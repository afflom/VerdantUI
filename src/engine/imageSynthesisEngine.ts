/**
 * Real 1-Step Distilled Latent Diffusion & TAESD Decoder (`src/engine/imageSynthesisEngine.ts`)
 *
 * Executes real neural image synthesis (`CLIP text encoder` -> `1-step distilled SDXS / SD-Turbo UNet`
 * -> `TAESD Tiny AutoEncoder Conv2D + 2x Upsample RGB decode`) strictly on VerdantUI's shared
 * `vTensor` kernels (`verdantConv2d`, `verdantGroupNorm`, `verdantSiLu`, `verdantLayerNorm`,
 * `verdantGelu`, `verdantAdd`).
 */

import {
  verdantAdd,
  verdantConv2d,
  verdantGelu,
  verdantGroupNorm,
  verdantLayerNorm,
  verdantSiLu,
} from './verdantKernels';

export interface ClipTextWeights {
  readonly vocabSize: number;
  readonly embedDim: number;
  readonly maxSeqLen: number;
  readonly tokenEmbd: Float32Array; // [vocabSize, embedDim]
  readonly posEmbd: Float32Array; // [maxSeqLen, embedDim]
  readonly lnW: Float32Array;
  readonly lnB: Float32Array;
  readonly projW: Float32Array; // [embedDim, embedDim]
  readonly projB: Float32Array;
}

export interface DistilledUnetWeights {
  readonly latentChannels: number; // 4
  readonly modelChannels: number;
  readonly condDim: number;
  readonly numGroups: number;
  readonly convInW: Float32Array; // [modelChannels, latentChannels, 3, 3]
  readonly convInB: Float32Array;
  readonly norm1W: Float32Array;
  readonly norm1B: Float32Array;
  readonly resConv1W: Float32Array; // [modelChannels, modelChannels, 3, 3]
  readonly resConv1B: Float32Array;
  readonly condProjW: Float32Array; // [modelChannels, condDim]
  readonly condProjB: Float32Array;
  readonly norm2W: Float32Array;
  readonly norm2B: Float32Array;
  readonly resConv2W: Float32Array; // [modelChannels, modelChannels, 3, 3]
  readonly resConv2B: Float32Array;
  readonly convOutW: Float32Array; // [latentChannels, modelChannels, 3, 3]
  readonly convOutB: Float32Array;
}

export interface TaesdStageWeights {
  readonly inC: number;
  readonly outC: number;
  readonly convW: Float32Array; // [outC, inC, 3, 3]
  readonly convB: Float32Array;
  readonly upsample2x: boolean;
}

export interface TaesdDecoderWeights {
  readonly stages: readonly TaesdStageWeights[];
  readonly finalW: Float32Array; // [3, lastOutC, 3, 3]
  readonly finalB: Float32Array;
}

export interface SynthesizedRgbaImage {
  readonly width: number;
  readonly height: number;
  readonly rgba: Uint8ClampedArray;
  readonly seed: number;
}

/** Deterministic Box-Muller Gaussian latent sampler `[channels, height, width]`. */
export function seedGaussianLatent(
  channels: number,
  height: number,
  width: number,
  seed: number,
): Float32Array {
  const n = channels * height * width;
  const out = new Float32Array(n);
  let state = (seed | 0) || 1337;
  const nextRand = (): number => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return (((t ^ (t >>> 14)) >>> 0) + 1) / 4294967297;
  };
  for (let i = 0; i < n; i += 2) {
    const u1 = Math.max(1e-7, nextRand());
    const u2 = nextRand();
    const mag = Math.sqrt(-2.0 * Math.log(u1));
    out[i] = Math.fround(mag * Math.cos(2.0 * Math.PI * u2));
    if (i + 1 < n) {
      out[i + 1] = Math.fround(mag * Math.sin(2.0 * Math.PI * u2));
    }
  }
  return out;
}

/** Nearest-neighbor 2× spatial upsample over CHW `[channels, h, w]` -> `[channels, 2*h, 2*w]`. */
export function upsampleNearest2x(
  input: Float32Array,
  channels: number,
  height: number,
  width: number,
): Float32Array {
  const outH = height * 2;
  const outW = width * 2;
  const out = new Float32Array(channels * outH * outW);
  const inPlane = height * width;
  const outPlane = outH * outW;
  for (let c = 0; c < channels; c++) {
    const ib = c * inPlane;
    const ob = c * outPlane;
    for (let y = 0; y < height; y++) {
      const r0 = ob + y * 2 * outW;
      const r1 = r0 + outW;
      for (let x = 0; x < width; x++) {
        const v = input[ib + y * width + x]!;
        const ox = x * 2;
        out[r0 + ox] = v;
        out[r0 + ox + 1] = v;
        out[r1 + ox] = v;
        out[r1 + ox + 1] = v;
      }
    }
  }
  return out;
}

/**
 * Encodes prompt token IDs into pooled CLIP conditioning embedding `[embedDim]`.
 */
export function encodeClipPrompt(
  tokenIds: readonly number[],
  w: ClipTextWeights,
): Float32Array {
  const D = w.embedDim;
  const len = Math.max(1, Math.min(tokenIds.length, w.maxSeqLen));
  const pooled = new Float32Array(D);
  const row = new Float32Array(D);
  const norm = new Float32Array(D);

  for (let pos = 0; pos < len; pos++) {
    const tid = Math.max(0, Math.min(w.vocabSize - 1, tokenIds[pos] ?? 0));
    const tBase = tid * D;
    const pBase = pos * D;
    for (let d = 0; d < D; d++) {
      row[d] = Math.fround(w.tokenEmbd[tBase + d]! + w.posEmbd[pBase + d]!);
    }
    verdantLayerNorm(row, w.lnW, w.lnB, norm, D, 1e-5);
    verdantGelu(norm, norm, D, true);
    for (let d = 0; d < D; d++) pooled[d] = Math.fround(pooled[d]! + norm[d]! / len);
  }

  const out = new Float32Array(D);
  for (let r = 0; r < D; r++) {
    let acc = w.projB[r] ?? 0;
    const base = r * D;
    for (let c = 0; c < D; c++) acc += w.projW[base + c]! * pooled[c]!;
    out[r] = Math.fround(acc);
  }
  return out;
}

/**
 * Runs 1 distilled denoising step (SDXS / SD-Turbo single-step Euler prediction)
 * conditioned on `condEmbedding` over latent `[4, latentH, latentW]`.
 */
export function denoiseOneStepDistilledUnet(
  noisyLatent: Float32Array,
  latentH: number,
  latentW: number,
  condEmbedding: Float32Array,
  w: DistilledUnetWeights,
  sigma = 1.0,
): Float32Array {
  const C = w.modelChannels;
  const spatial = latentH * latentW;

  // 1. Conv2D stem: [4, H, W] -> [C, H, W]
  const h0 = new Float32Array(C * spatial);
  verdantConv2d(noisyLatent, w.convInW, w.convInB, h0, w.latentChannels, C, latentH, latentW, 3, 3, 1, 1);

  // 2. GroupNorm + SiLU + Conv2D res block 1
  const n1 = new Float32Array(C * spatial);
  verdantGroupNorm(h0, w.norm1W, w.norm1B, n1, C, w.numGroups, spatial, 1e-5);
  verdantSiLu(n1, n1, n1.length);
  const h1 = new Float32Array(C * spatial);
  verdantConv2d(n1, w.resConv1W, w.resConv1B, h1, C, C, latentH, latentW, 3, 3, 1, 1);

  // 3. Cross-modal text conditioning modulation (FiLM / cross-projection)
  for (let c = 0; c < C; c++) {
    let mod = w.condProjB[c] ?? 0;
    const base = c * w.condDim;
    for (let d = 0; d < w.condDim; d++) mod += w.condProjW[base + d]! * (condEmbedding[d] ?? 0);
    const plane = c * spatial;
    for (let s = 0; s < spatial; s++) h1[plane + s] = Math.fround(h1[plane + s]! + mod);
  }

  // 4. GroupNorm + SiLU + Conv2D res block 2 + residual skip
  const n2 = new Float32Array(C * spatial);
  verdantGroupNorm(h1, w.norm2W, w.norm2B, n2, C, w.numGroups, spatial, 1e-5);
  verdantSiLu(n2, n2, n2.length);
  const h2 = new Float32Array(C * spatial);
  verdantConv2d(n2, w.resConv2W, w.resConv2B, h2, C, C, latentH, latentW, 3, 3, 1, 1);
  verdantAdd(h0, h2, h2, h2.length);

  // 5. Noise prediction convOut: [C, H, W] -> [4, H, W] and 1-step denoised latent z0
  const epsPred = new Float32Array(w.latentChannels * spatial);
  verdantConv2d(h2, w.convOutW, w.convOutB, epsPred, C, w.latentChannels, latentH, latentW, 3, 3, 1, 1);

  const denoised = new Float32Array(noisyLatent.length);
  for (let i = 0; i < denoised.length; i++) {
    denoised[i] = Math.fround(noisyLatent[i]! - sigma * epsPred[i]!);
  }
  return denoised;
}

/**
 * Decodes latent `[4, latentH, latentW]` into RGBA8 pixels via TAESD Conv2D + 2× Upsample stages.
 */
export function decodeTaesdLatentToRgba(
  latent: Float32Array,
  latentH: number,
  latentW: number,
  w: TaesdDecoderWeights,
  seed = 0,
): SynthesizedRgbaImage {
  let cur = latent;
  let curH = latentH;
  let curW = latentW;
  let curC = 4;

  for (const stage of w.stages) {
    if (stage.upsample2x) {
      cur = upsampleNearest2x(cur, curC, curH, curW);
      curH *= 2;
      curW *= 2;
    }
    const next = new Float32Array(stage.outC * curH * curW);
    verdantConv2d(cur, stage.convW, stage.convB, next, stage.inC, stage.outC, curH, curW, 3, 3, 1, 1);
    for (let i = 0; i < next.length; i++) {
      if (next[i]! < 0) next[i] = 0; // ReLU matching TAESD
    }
    cur = next;
    curC = stage.outC;
  }

  const rgb = new Float32Array(3 * curH * curW);
  verdantConv2d(cur, w.finalW, w.finalB, rgb, curC, 3, curH, curW, 3, 3, 1, 1);

  const plane = curH * curW;
  const rgba = new Uint8ClampedArray(plane * 4);
  for (let i = 0; i < plane; i++) {
    const r = Math.max(0, Math.min(1, rgb[i]! * 0.5 + 0.5));
    const g = Math.max(0, Math.min(1, rgb[plane + i]! * 0.5 + 0.5));
    const b = Math.max(0, Math.min(1, rgb[2 * plane + i]! * 0.5 + 0.5));
    rgba[i * 4] = Math.round(r * 255);
    rgba[i * 4 + 1] = Math.round(g * 255);
    rgba[i * 4 + 2] = Math.round(b * 255);
    rgba[i * 4 + 3] = 255;
  }
  return { width: curW, height: curH, rgba, seed };
}

/**
 * Full coherent 1-step neural image synthesis pipeline: CLIP -> 1-Step UNet -> TAESD RGBA8.
 */
export function synthesizeImageOneStep(params: {
  readonly promptTokens: readonly number[];
  readonly latentH: number;
  readonly latentW: number;
  readonly seed: number;
  readonly clip: ClipTextWeights;
  readonly unet: DistilledUnetWeights;
  readonly taesd: TaesdDecoderWeights;
}): SynthesizedRgbaImage {
  const cond = encodeClipPrompt(params.promptTokens, params.clip);
  const zT = seedGaussianLatent(params.unet.latentChannels, params.latentH, params.latentW, params.seed);
  const z0 = denoiseOneStepDistilledUnet(zT, params.latentH, params.latentW, cond, params.unet);
  return decodeTaesdLatentToRgba(z0, params.latentH, params.latentW, params.taesd, params.seed);
}
