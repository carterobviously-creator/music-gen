export const MUSIC_ENGINE_DEFAULTS = {
  sampleRate: 32000,
  bitDepth: 16,
  channels: 1,
  maxSeconds: 30,
  minSeconds: 5,
  defaultTemperature: 0.75,
  powerPreference: 'high-performance',
};

export const GENRE_DEFINITIONS = {
  electronic: 'pulsing synths, punchy drums, futuristic textures',
  indie: 'warm guitars, intimate vocals, organic room ambience',
  lofi: 'dusty drums, vinyl crackle, mellow keys, soft saturation',
  ambient: 'evolving pads, long reverbs, sparse tonal movement',
  cinematic: 'orchestral layers, dramatic swells, emotional motifs',
  house: 'four-on-the-floor kick, groove bass, dancefloor momentum',
  techno: 'hypnotic sequencers, deep kick, industrial atmosphere',
  hiphop: 'boom-bap drums, bass groove, head-nod rhythm pocket',
  trap: '808 bass, sharp hats, modern urban energy',
  jazz: 'upright bass, brushed drums, rich harmonic language',
  rock: 'driven guitars, live drums, energetic performance',
  classical: 'strings and woodwinds, elegant dynamics, concert hall feel',
  synthwave: 'retro analog synths, neon arps, nostalgic pulse',
};

const ENVELOPE = Object.freeze({ attack: 0.03, release: 0.12 });
const DEFAULT_PROMPT_TOKENS = new Uint32Array([17, 23, 37, 53]);
const DEFAULT_EMPTY_TOKEN = 11;
// Knuth multiplicative hash constant for spreading 16-bit/8-bit char values in 32-bit space.
const TOKEN_HASH_MULTIPLIER = 2654435761;
// Small salt mixed into all prompt tokens to avoid sparse low-value token outputs.
const TOKEN_INDEX_SALT = 97;
const SEED_TEMPO_FACTOR = 13;
const SEED_TEMPERATURE_FACTOR = 1000;

export class WebGPUMusicEngine {
  constructor(options = {}) {
    this.settings = { ...MUSIC_ENGINE_DEFAULTS, ...options };
    this.device = null;
    this.adapter = null;
    this.ready = false;
  }

  async initGPU() {
    if (!('gpu' in navigator)) {
      throw new Error('WebGPU is not available in this browser.');
    }

    this.adapter = await navigator.gpu.requestAdapter({
      powerPreference: this.settings.powerPreference,
      forceFallbackAdapter: false,
    });

    if (!this.adapter) {
      throw new Error('No compatible GPU adapter found.');
    }

    this.device = await this.adapter.requestDevice({
      requiredFeatures: [],
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(
          this.adapter.limits.maxStorageBufferBindingSize,
          256 * 1024 * 1024,
        ),
      },
    });

    this.ready = true;
    return this.device;
  }

  encodePromptTokens(prompt) {
    const normalized = String(prompt || '').toLowerCase().trim();
    if (!normalized) return DEFAULT_PROMPT_TOKENS;

    const out = new Uint32Array(Math.min(256, normalized.length));
    for (let i = 0; i < out.length; i += 1) {
      const code = normalized.charCodeAt(i);
      out[i] = ((code * TOKEN_HASH_MULTIPLIER) >>> 0) ^ TOKEN_INDEX_SALT;
    }
    return out;
  }

  _seedFromTokens(tokens, tempo, temperature) {
    let seed = Math.floor((tempo * SEED_TEMPO_FACTOR + temperature * SEED_TEMPERATURE_FACTOR) % 2147483647) || 1;
    for (let i = 0; i < tokens.length; i += 1) {
      seed = (seed ^ tokens[i]) % 2147483647;
      if (seed <= 0) seed += 2147483646;
    }
    return seed;
  }

  _nextRandom(state) {
    state.seed = (state.seed * 48271) % 2147483647;
    return state.seed / 2147483647;
  }

  async generateMusic({ promptTokens, durationSec, tempo, temperature = 0.75 }) {
    if (!this.ready) {
      await this.initGPU();
    }

    const sampleRate = this.settings.sampleRate;
    const length = Math.max(1, Math.floor(durationSec * sampleRate));
    const tokens = promptTokens instanceof Uint32Array ? promptTokens : new Uint32Array(promptTokens || [DEFAULT_EMPTY_TOKEN]);

    const state = {
      seed: this._seedFromTokens(tokens, tempo, temperature),
    };

    // Compute pipeline creation verifies WebGPU compute support.
    const shaderModule = this.device.createShaderModule({
      code: `
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
      }
      `,
    });

    this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'main' },
    });

    const audio = new Float32Array(length);
    const beatHz = Math.max(0.6, tempo / 60);
    const attackFrames = Math.max(1, Math.floor(sampleRate * ENVELOPE.attack));
    const releaseFrames = Math.max(1, Math.floor(sampleRate * ENVELOPE.release));

    for (let i = 0; i < length; i += 1) {
      const t = i / sampleRate;
      const harmonic = Math.sin(2 * Math.PI * (110 + beatHz * 30) * t)
        + 0.5 * Math.sin(2 * Math.PI * (220 + beatHz * 15) * t)
        + 0.25 * Math.sin(2 * Math.PI * (440 + beatHz * 5) * t);
      const noise = (this._nextRandom(state) * 2 - 1) * 0.22 * temperature;
      const groove = Math.sin(2 * Math.PI * beatHz * t) * 0.15;
      audio[i] = (harmonic * 0.42 + groove + noise) * (0.88 + 0.12 * Math.sin(t * 0.6));

      if (i < attackFrames) audio[i] *= i / attackFrames;
      if (i > length - releaseFrames) audio[i] *= (length - i) / releaseFrames;
    }

    this.normalizeSignal(audio);
    return { audio, sampleRate };
  }

  normalizeSignal(floatBuffer) {
    let peak = 0;
    for (let i = 0; i < floatBuffer.length; i += 1) {
      const abs = Math.abs(floatBuffer[i]);
      if (abs > peak) peak = abs;
    }
    if (peak < 1e-6) return floatBuffer;
    const gain = 0.92 / peak;
    for (let i = 0; i < floatBuffer.length; i += 1) {
      const val = floatBuffer[i] * gain;
      floatBuffer[i] = Math.max(-1, Math.min(1, val));
    }
    return floatBuffer;
  }

  float32ToWav(floatBuffer, sampleRate = this.settings.sampleRate) {
    const bytesPerSample = this.settings.bitDepth / 8;
    const dataSize = floatBuffer.length * bytesPerSample;
    const wav = new ArrayBuffer(44 + dataSize);
    const view = new DataView(wav);

    const writeString = (offset, text) => {
      for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, this.settings.channels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * this.settings.channels * bytesPerSample, true);
    view.setUint16(32, this.settings.channels * bytesPerSample, true);
    view.setUint16(34, this.settings.bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);

    for (let i = 0; i < floatBuffer.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, floatBuffer[i]));
      view.setInt16(44 + i * bytesPerSample, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    }

    return new Blob([wav], { type: 'audio/wav' });
  }
}

if (typeof window !== 'undefined') {
  window.WebGPUMusicEngine = WebGPUMusicEngine;
  window.GENRE_DEFINITIONS = GENRE_DEFINITIONS;
  window.MUSIC_ENGINE_DEFAULTS = MUSIC_ENGINE_DEFAULTS;
}
