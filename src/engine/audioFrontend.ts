/**
 * Real Browser Audio Sensory Frontend (`src/engine/audioFrontend.ts`)
 *
 * Implements exact `whisper.cpp`-compatible 16 kHz mono resampling, radix-2 FFT,
 * 80-bin Slaney Mel filterbank, and log-mel spectrogram extraction `[80, nFrames]`,
 * plus permission-gated browser microphone capture (`MediaDevices.getUserMedia`).
 */

export const WHISPER_SAMPLE_RATE = 16_000;
export const WHISPER_N_FFT = 400;
export const WHISPER_HOP_LENGTH = 160;
export const WHISPER_N_MELS = 80;
const FFT_SIZE = 512; // next power of 2 >= 400 for exact radix-2 FFT

export interface MelSpectrogram {
  readonly nMels: number;
  readonly nFrames: number;
  /** Row-major `[nMels, nFrames]` normalized log10 mel spectrogram. */
  readonly data: Float32Array;
  readonly durationSec: number;
}

/** Linearly resamples mono audio to `targetRate` (default 16 kHz for Whisper). */
export function resampleLinearMono(
  pcm: Float32Array,
  srcRate: number,
  targetRate = WHISPER_SAMPLE_RATE,
): Float32Array {
  if (pcm.length === 0 || srcRate <= 0) return new Float32Array(0);
  if (srcRate === targetRate) return pcm.slice();
  const outLen = Math.max(1, Math.round((pcm.length * targetRate) / srcRate));
  const out = new Float32Array(outLen);
  const ratio = (pcm.length - 1) / Math.max(1, outLen - 1);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const lo = Math.floor(pos);
    const hi = Math.min(pcm.length - 1, lo + 1);
    const frac = pos - lo;
    out[i] = pcm[lo]! * (1 - frac) + pcm[hi]! * frac;
  }
  return out;
}

function hzToMelSlaney(hz: number): number {
  const fMin = 0.0;
  const fSp = 200.0 / 3.0;
  const minLogHz = 1000.0;
  const minLogMel = (minLogHz - fMin) / fSp;
  const logStep = Math.log(6.4) / 27.0;
  return hz >= minLogHz ? minLogMel + Math.log(hz / minLogHz) / logStep : (hz - fMin) / fSp;
}

function melToHzSlaney(mel: number): number {
  const fMin = 0.0;
  const fSp = 200.0 / 3.0;
  const minLogHz = 1000.0;
  const minLogMel = (minLogHz - fMin) / fSp;
  const logStep = Math.log(6.4) / 27.0;
  return mel >= minLogMel ? minLogHz * Math.exp(logStep * (mel - minLogMel)) : fMin + fSp * mel;
}

let cachedHannWindow: Float32Array | null = null;
let cachedMelFilters: Float32Array | null = null; // [WHISPER_N_MELS, FFT_SIZE/2 + 1]

function getHannWindow(): Float32Array {
  if (cachedHannWindow) return cachedHannWindow;
  const w = new Float32Array(WHISPER_N_FFT);
  for (let i = 0; i < WHISPER_N_FFT; i++) {
    w[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / WHISPER_N_FFT));
  }
  cachedHannWindow = w;
  return w;
}

function getMelFilterbank(): Float32Array {
  if (cachedMelFilters) return cachedMelFilters;
  const nBins = FFT_SIZE / 2 + 1;
  const filters = new Float32Array(WHISPER_N_MELS * nBins);
  const minMel = hzToMelSlaney(0);
  const maxMel = hzToMelSlaney(WHISPER_SAMPLE_RATE / 2);
  const melPts = new Float64Array(WHISPER_N_MELS + 2);
  for (let i = 0; i < melPts.length; i++) {
    melPts[i] = melToHzSlaney(minMel + ((maxMel - minMel) * i) / (WHISPER_N_MELS + 1));
  }
  const binFreqs = new Float64Array(nBins);
  for (let b = 0; b < nBins; b++) {
    binFreqs[b] = (b * WHISPER_SAMPLE_RATE) / FFT_SIZE;
  }

  for (let m = 0; m < WHISPER_N_MELS; m++) {
    const fLeft = melPts[m]!;
    const fCenter = melPts[m + 1]!;
    const fRight = melPts[m + 2]!;
    const enorm = 2.0 / Math.max(1e-6, fRight - fLeft);
    const rowBase = m * nBins;
    for (let b = 0; b < nBins; b++) {
      const f = binFreqs[b]!;
      let weight = 0;
      if (f >= fLeft && f <= fCenter) {
        weight = (f - fLeft) / Math.max(1e-6, fCenter - fLeft);
      } else if (f > fCenter && f <= fRight) {
        weight = (fRight - f) / Math.max(1e-6, fRight - fCenter);
      }
      filters[rowBase + b] = weight * enorm;
    }
  }
  cachedMelFilters = filters;
  return filters;
}

/** In-place iterative radix-2 Cooley-Tukey FFT (`re.length === im.length === 2^k`). */
export function fftRadix2(re: Float32Array, im: Float32Array): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i]!;
      re[i] = re[j]!;
      re[j] = tr;
      const ti = im[i]!;
      im[i] = im[j]!;
      im[j] = ti;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const ang = (-2.0 * Math.PI) / len;
    const wCos = Math.cos(ang);
    const wSin = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wr = 1.0;
      let wi = 0.0;
      for (let j = 0; j < half; j++) {
        const uRe = re[i + j]!;
        const uIm = im[i + j]!;
        const vRe = re[i + j + half]! * wr - im[i + j + half]! * wi;
        const vIm = re[i + j + half]! * wi + im[i + j + half]! * wr;
        re[i + j] = uRe + vRe;
        im[i + j] = uIm + vIm;
        re[i + j + half] = uRe - vRe;
        im[i + j + half] = uIm - vIm;
        const nextWr = wr * wCos - wi * wSin;
        wi = wr * wSin + wi * wCos;
        wr = nextWr;
      }
    }
  }
}

/**
 * Computes an exact `whisper.cpp`-compatible 80-bin Log-Mel Spectrogram from 16 kHz mono PCM.
 */
export function computeLogMelSpectrogram(
  pcm16k: Float32Array,
  maxFrames = 3000,
): MelSpectrogram {
  const nFrames = Math.min(
    maxFrames,
    Math.max(1, Math.floor((Math.max(0, pcm16k.length - WHISPER_N_FFT)) / WHISPER_HOP_LENGTH) + 1),
  );
  const nBins = FFT_SIZE / 2 + 1;
  const hann = getHannWindow();
  const melFilters = getMelFilterbank();
  const data = new Float32Array(WHISPER_N_MELS * nFrames);
  const re = new Float32Array(FFT_SIZE);
  const im = new Float32Array(FFT_SIZE);
  const magSq = new Float32Array(nBins);

  let maxLog = -Infinity;
  for (let t = 0; t < nFrames; t++) {
    const start = t * WHISPER_HOP_LENGTH;
    re.fill(0);
    im.fill(0);
    for (let i = 0; i < WHISPER_N_FFT; i++) {
      const sample = start + i < pcm16k.length ? pcm16k[start + i]! : 0;
      re[i] = sample * hann[i]!;
    }
    fftRadix2(re, im);
    for (let b = 0; b < nBins; b++) {
      const r = re[b]!;
      const imv = im[b]!;
      magSq[b] = r * r + imv * imv;
    }
    for (let m = 0; m < WHISPER_N_MELS; m++) {
      const fBase = m * nBins;
      let energy = 0.0;
      for (let b = 0; b < nBins; b++) {
        energy += melFilters[fBase + b]! * magSq[b]!;
      }
      const logVal = Math.log10(Math.max(1e-10, energy));
      data[m * nFrames + t] = logVal;
      if (logVal > maxLog) maxLog = logVal;
    }
  }

  // Whisper dynamic-range clamp & normalization: `(max(v, maxLog - 8) + 4) / 4`
  const floor = maxLog - 8.0;
  for (let i = 0; i < data.length; i++) {
    const clamped = data[i]! < floor ? floor : data[i]!;
    data[i] = Math.fround((clamped + 4.0) / 4.0);
  }

  return {
    nMels: WHISPER_N_MELS,
    nFrames,
    data,
    durationSec: pcm16k.length / WHISPER_SAMPLE_RATE,
  };
}

/**
 * Permission-gated browser microphone recorder capturing 16 kHz mono PCM directly on-device.
 */
export class BrowserVoiceRecorder {
  private stream: MediaStream | null = null;
  private ctx: AudioContext | null = null;
  private processor: ScriptProcessorNode | null = null;
  private chunks: Float32Array[] = [];
  private sampleRate = 48_000;

  async start(): Promise<void> {
    if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
      throw new Error('Microphone input is not available in this browser.');
    }
    this.chunks = [];
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    this.ctx = new AudioCtx();
    this.sampleRate = this.ctx.sampleRate;
    const source = this.ctx.createMediaStreamSource(this.stream);
    this.processor = this.ctx.createScriptProcessor(4096, 1, 1);
    this.processor.onaudioprocess = (ev) => {
      const input = ev.inputBuffer.getChannelData(0);
      this.chunks.push(new Float32Array(input));
    };
    source.connect(this.processor);
    this.processor.connect(this.ctx.destination);
  }

  async stop(): Promise<Float32Array> {
    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
      this.stream = null;
    }
    if (this.ctx) {
      await this.ctx.close().catch(() => {});
      this.ctx = null;
    }
    const total = this.chunks.reduce((a, c) => a + c.length, 0);
    const merged = new Float32Array(total);
    let off = 0;
    for (const c of this.chunks) {
      merged.set(c, off);
      off += c.length;
    }
    this.chunks = [];
    return resampleLinearMono(merged, this.sampleRate, WHISPER_SAMPLE_RATE);
  }
}
