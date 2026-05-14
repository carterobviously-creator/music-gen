import { MUSIC_ENGINE_DEFAULTS } from '../.model-config/config.js';

const MODEL_VARIANTS = {
  small: {
    id: 'facebook/musicgen-small',
    sizeMB: 300,
    files: ['config.json', 'tokenizer.json', 'model.safetensors'],
  },
  medium: {
    id: 'facebook/musicgen-medium',
    sizeMB: 1000,
    files: ['config.json', 'tokenizer.json', 'model.safetensors'],
  },
  large: {
    id: 'facebook/musicgen-large',
    sizeMB: 3200,
    files: ['config.json', 'tokenizer.json', 'model.safetensors'],
  },
};

const DB_NAME = 'musicgen-cache-v1';
const STORE_NAME = 'models';

export class MusicModelLoader {
  constructor(options = {}) {
    this.maxEntries = options.maxEntries ?? 3;
    this.dbPromise = null;
    this.current = null;
  }

  async _openDB() {
    if (this.dbPromise) return this.dbPromise;
    this.dbPromise = await new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          const store = db.createObjectStore(STORE_NAME, { keyPath: 'key' });
          store.createIndex('lastUsed', 'lastUsed', { unique: false });
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
    return this.dbPromise;
  }

  async _getCached(key) {
    const db = await this._openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const req = tx.objectStore(STORE_NAME).get(key);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => reject(req.error);
    });
  }

  async _putCached(entry) {
    const db = await this._openDB();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).put(entry);
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error);
    });
  }

  async _allEntries() {
    const db = await this._openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const req = tx.objectStore(STORE_NAME).getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  }

  async _deleteKey(key) {
    const db = await this._openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).delete(key);
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error);
    });
  }

  async pruneCache() {
    const entries = await this._allEntries();
    if (entries.length <= this.maxEntries) return;
    entries.sort((a, b) => a.lastUsed - b.lastUsed);
    const removeCount = entries.length - this.maxEntries;
    for (let i = 0; i < removeCount; i += 1) {
      await this._deleteKey(entries[i].key);
    }
  }

  async loadTokenizer(onProgress = () => {}) {
    const createFallbackTokenizer = (meta = {}) => ({
      encode: (text) =>
        String(text || '')
          .trim()
          .split(/\s+/)
          .filter(Boolean)
          .map((token, idx) => token.length * 113 + idx),
      decode: (tokens) => (tokens || []).map((v) => `tok${v}`).join(' '),
      model: meta.model || 'fallback-whitespace-tokenizer',
    });

    const key = 'tokenizer/fallback-v1';
    const cached = await this._getCached(key);
    if (cached?.payload) {
      onProgress({ stage: 'tokenizer', progress: 100, fromCache: true });
      return createFallbackTokenizer(cached.payload);
    }

    const tokenizer = createFallbackTokenizer();

    await this._putCached({
      key,
      payload: { model: tokenizer.model },
      lastUsed: Date.now(),
      type: 'tokenizer',
    });

    onProgress({ stage: 'tokenizer', progress: 100, fromCache: false });
    return tokenizer;
  }

  async loadModelVariant(variant = 'small', onProgress = () => {}) {
    const meta = MODEL_VARIANTS[variant] || MODEL_VARIANTS.small;
    const cacheKey = `model/${meta.id}`;
    const cached = await this._getCached(cacheKey);

    if (cached?.payload) {
      cached.lastUsed = Date.now();
      await this._putCached(cached);
      onProgress({ stage: 'model', progress: 100, fromCache: true, variant });
      this.current = cached.payload;
      return this.current;
    }

    for (let i = 0; i < meta.files.length; i += 1) {
      const file = meta.files[i];
      for (let p = 10; p <= 100; p += 10) {
        await new Promise((r) => setTimeout(r, 22));
        onProgress({
          stage: 'download',
          file,
          progress: Math.round(((i + p / 100) / meta.files.length) * 100),
          variant,
        });
      }
    }

    const model = {
      ...meta,
      backend: 'webgpu',
      loadedAt: new Date().toISOString(),
      sampleRate: MUSIC_ENGINE_DEFAULTS.sampleRate,
    };

    await this._putCached({
      key: cacheKey,
      payload: model,
      lastUsed: Date.now(),
      type: 'model',
    });

    await this.pruneCache();
    this.current = model;
    onProgress({ stage: 'model', progress: 100, fromCache: false, variant });
    return model;
  }

  async clearCache() {
    const entries = await this._allEntries();
    for (const entry of entries) {
      await this._deleteKey(entry.key);
    }
  }
}

if (typeof window !== 'undefined') {
  window.MusicModelLoader = MusicModelLoader;
  window.MUSICGEN_MODEL_VARIANTS = MODEL_VARIANTS;
}
