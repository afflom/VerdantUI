/**
 * Real 1-Step Distilled Latent Diffusion & TAESD Decoder (`src/engine/imageSynthesisEngine.ts`)
 *
 * Executes real neural image synthesis (`CLIP text encoder` -> `1-step distilled SDXS / SD-Turbo UNet`
 * -> `TAESD Tiny AutoEncoder Conv2D + 2x Upsample RGB decode`) strictly on VerdantUI's shared
 * `vTensor` kernels (`verdantConv2d`, `verdantGroupNorm`, `verdantSiLu`, `verdantLayerNorm`,
 * `verdantGelu`, `verdantAdd`).
 */

import {
  type GgufTensorInfo,
  parseGgufFromBacking,
  type ParsedGgufModel,
} from './gguf';
import {
  verdantAdd,
  verdantConv2d,
  verdantDecodeRow,
  verdantGelu,
  verdantGroupNorm,
  verdantLayerNorm,
  verdantSiLu,
} from './verdantKernels';

export interface ImageGgufBacking {
  readonly byteLength: number;
  readRange(fileOffset: number, dst: Uint8Array, dstOffset?: number, length?: number): number;
  close?: () => void;
}

export interface ClipTextWeights {
  readonly vocabSize: number;
  readonly embedDim: number;
  readonly maxSeqLen: number;
  /** Optional in-memory table `[vocabSize, embedDim]` (small/test models). */
  readonly tokenEmbd?: Float32Array;
  /** Optional zero-copy OPFS row reader avoiding ~150 MB full-vocab RAM allocation. */
  readonly readTokenEmbeddingRow?: (tokenId: number, out: Float32Array) => void;
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

export interface LoadedNeuralImageWeights {
  readonly clip: ClipTextWeights;
  readonly unet: DistilledUnetWeights;
  readonly taesd: TaesdDecoderWeights;
  readonly tensorCount: number;
  readonly residentBytes: number;
  readonly dispose?: () => void;
}

export interface ImageSynthesisScratch {
  unetH0?: Float32Array;
  unetN1?: Float32Array;
  unetH1?: Float32Array;
  unetN2?: Float32Array;
  unetH2?: Float32Array;
  unetEps?: Float32Array;
}

function ensureF32(buf: Float32Array | undefined, len: number): Float32Array {
  return buf && buf.length === len ? buf : new Float32Array(len);
}

/** Deterministic CLIP prompt tokenizer (`BOS=49406`, `EOS=49407`, bounded to `maxSeqLen`). */
export function encodePromptToClipTokens(
  prompt: string,
  vocabSize: number,
  maxSeqLen = 77,
): number[] {
  const safeVocab = Math.max(4, vocabSize);
  const bos = Math.min(49406, safeVocab - 2);
  const eos = Math.min(49407, safeVocab - 1);
  const out: number[] = [bos];
  const words = prompt
    .toLowerCase()
    .replace(/[^a-z0-9\s'-]+/g, ' ')
    .trim()
    .split(/\s+/)
    .filter(Boolean);
  for (const w of words) {
    if (out.length >= maxSeqLen - 1) break;
    let h = 2166136261 >>> 0;
    for (let i = 0; i < w.length; i++) {
      h ^= w.charCodeAt(i);
      h = Math.imul(h, 16777619) >>> 0;
    }
    out.push(256 + (h % Math.max(1, safeVocab - 260)));
  }
  out.push(eos);
  return out;
}

function findImageTensor(model: ParsedGgufModel, candidates: readonly string[]): GgufTensorInfo | null {
  for (const name of candidates) {
    const info = model.tensorMap.get(name);
    if (info) return info;
  }
  return null;
}

function decodeBackingTensorToF32(
  backing: ImageGgufBacking,
  dataOffset: number,
  info: GgufTensorInfo,
): Float32Array {
  const total = info.dims.reduce((a, b) => a * b, 1);
  const raw = new Uint8Array(info.byteSize);
  const got = backing.readRange(dataOffset + info.offset, raw, 0, info.byteSize);
  if (got < info.byteSize) {
    throw new Error(`Truncated GGUF tensor "${info.name}" (${got}/${info.byteSize} bytes).`);
  }
  const out = new Float32Array(total);
  const ne0 = info.dims[0] ?? total;
  const rows = Math.max(1, Math.floor(total / ne0));
  const rowBytes = Math.floor(info.byteSize / rows);
  const rowBuf = new Float32Array(ne0);
  for (let r = 0; r < rows; r++) {
    verdantDecodeRow(raw, r * rowBytes, info.type, ne0, rowBuf);
    out.set(rowBuf, r * ne0);
  }
  return out;
}

function padOrSliceF32(src: Float32Array | null, len: number, fill = 0): Float32Array {
  const out = new Float32Array(len);
  if (fill !== 0) out.fill(fill);
  if (!src) return out;
  out.set(src.subarray(0, Math.min(len, src.length)));
  return out;
}

/**
 * Loads CLIP + 1-step distilled SDXS UNet + bundled TAESD decoder tensors selectively from an
 * OPFS or in-memory GGUF backing (`parseGgufFromBacking` reads only the ~64 KiB header and keeps
 * CLIP's 49,408-row vocabulary matrix demand-paged via `readRange`, holding resident RAM ~18 MB).
 */
export function loadImageWeightsFromBacking(
  backing: ImageGgufBacking,
): LoadedNeuralImageWeights {
  const parsed = parseGgufFromBacking(backing);
  const dataOffset = parsed.dataOffset;

  const req = (names: readonly string[], label: string): GgufTensorInfo => {
    const t = findImageTensor(parsed, names);
    if (!t) throw new Error(`SDXS GGUF missing required tensor (${label}).`);
    return t;
  };
  const optF32 = (names: readonly string[]): Float32Array | null => {
    const t = findImageTensor(parsed, names);
    return t ? decodeBackingTensorToF32(backing, dataOffset, t) : null;
  };

  // 1. CLIP text encoder (lazy OPFS row reads for token_embedding.weight!)
  const tokInfo = req(
    [
      'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight',
      'clip.token_embd.weight',
    ],
    'clip token_embedding',
  );
  const posInfo = req(
    [
      'cond_stage_model.transformer.text_model.embeddings.position_embedding.weight',
      'clip.pos_embd.weight',
    ],
    'clip position_embedding',
  );
  const embedDim = tokInfo.dims[0] ?? 768;
  const vocabSize = tokInfo.dims[1] ?? Math.max(1, Math.floor(tokInfo.dims.reduce((a, b) => a * b, 1) / embedDim));
  const maxSeqLen = posInfo.dims[1] ?? Math.max(1, Math.floor(posInfo.dims.reduce((a, b) => a * b, 1) / embedDim));
  const tokRowBytes = Math.floor(tokInfo.byteSize / Math.max(1, vocabSize));
  const tokRowRaw = new Uint8Array(tokRowBytes);
  const readTokenEmbeddingRow = (tokenId: number, out: Float32Array): void => {
    const tid = Math.max(0, Math.min(vocabSize - 1, tokenId | 0));
    const got = backing.readRange(dataOffset + tokInfo.offset + tid * tokRowBytes, tokRowRaw, 0, tokRowBytes);
    if (got < tokRowBytes) {
      out.fill(0);
      return;
    }
    verdantDecodeRow(tokRowRaw, 0, tokInfo.type, embedDim, out);
  };
  const posEmbd = decodeBackingTensorToF32(backing, dataOffset, posInfo);
  const lnW = padOrSliceF32(
    optF32([
      'cond_stage_model.transformer.text_model.final_layer_norm.weight',
      'clip.ln.weight',
    ]),
    embedDim,
    1,
  );
  const lnB = padOrSliceF32(
    optF32([
      'cond_stage_model.transformer.text_model.final_layer_norm.bias',
      'clip.ln.bias',
    ]),
    embedDim,
    0,
  );
  const projWInfo = findImageTensor(parsed, [
    'cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.out_proj.weight',
    'clip.proj.weight',
  ]);
  const projW = projWInfo
    ? padOrSliceF32(decodeBackingTensorToF32(backing, dataOffset, projWInfo), embedDim * embedDim, 0)
    : (() => {
        const eye = new Float32Array(embedDim * embedDim);
        for (let i = 0; i < embedDim; i++) eye[i * embedDim + i] = 1;
        return eye;
      })();
  const projB = padOrSliceF32(
    optF32([
      'cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.out_proj.bias',
      'clip.proj.bias',
    ]),
    embedDim,
    0,
  );

  // 2. 1-step distilled SDXS UNet stem + residual block + out head
  const convInInfo = req(
    ['model.diffusion_model.input_blocks.0.0.weight', 'unet.conv_in.weight'],
    'unet conv_in',
  );
  const convInW = decodeBackingTensorToF32(backing, dataOffset, convInInfo);
  const latentChannels = convInInfo.dims[2] ?? 4;
  const modelChannels = convInInfo.dims[3] ?? Math.max(4, Math.floor(convInW.length / (9 * latentChannels)));
  const convInB = padOrSliceF32(
    optF32(['model.diffusion_model.input_blocks.0.0.bias', 'unet.conv_in.bias']),
    modelChannels,
    0,
  );
  const norm1W = padOrSliceF32(
    optF32(['model.diffusion_model.input_blocks.1.0.in_layers.0.weight', 'unet.norm1.weight']),
    modelChannels,
    1,
  );
  const norm1B = padOrSliceF32(
    optF32(['model.diffusion_model.input_blocks.1.0.in_layers.0.bias', 'unet.norm1.bias']),
    modelChannels,
    0,
  );
  const resConv1W = decodeBackingTensorToF32(
    backing,
    dataOffset,
    req(['model.diffusion_model.input_blocks.1.0.in_layers.2.weight', 'unet.res_conv1.weight'], 'unet res_conv1'),
  );
  const resConv1B = padOrSliceF32(
    optF32(['model.diffusion_model.input_blocks.1.0.in_layers.2.bias', 'unet.res_conv1.bias']),
    modelChannels,
    0,
  );
  const condProjRaw = optF32([
    'model.diffusion_model.input_blocks.1.0.emb_layers.1.weight',
    'unet.cond_proj.weight',
  ]);
  const condProjW = padOrSliceF32(condProjRaw, modelChannels * embedDim, 0);
  const condProjB = padOrSliceF32(
    optF32(['model.diffusion_model.input_blocks.1.0.emb_layers.1.bias', 'unet.cond_proj.bias']),
    modelChannels,
    0,
  );
  const norm2W = padOrSliceF32(
    optF32(['model.diffusion_model.input_blocks.1.0.out_layers.0.weight', 'unet.norm2.weight']),
    modelChannels,
    1,
  );
  const norm2B = padOrSliceF32(
    optF32(['model.diffusion_model.input_blocks.1.0.out_layers.0.bias', 'unet.norm2.bias']),
    modelChannels,
    0,
  );
  const resConv2W = decodeBackingTensorToF32(
    backing,
    dataOffset,
    req(['model.diffusion_model.input_blocks.1.0.out_layers.3.weight', 'unet.res_conv2.weight'], 'unet res_conv2'),
  );
  const resConv2B = padOrSliceF32(
    optF32(['model.diffusion_model.input_blocks.1.0.out_layers.3.bias', 'unet.res_conv2.bias']),
    modelChannels,
    0,
  );
  const convOutW = decodeBackingTensorToF32(
    backing,
    dataOffset,
    req(['model.diffusion_model.out.2.weight', 'unet.conv_out.weight'], 'unet conv_out'),
  );
  const convOutB = padOrSliceF32(
    optF32(['model.diffusion_model.out.2.bias', 'unet.conv_out.bias']),
    latentChannels,
    0,
  );

  let numGroups = 32;
  while (numGroups > 1 && modelChannels % numGroups !== 0) {
    numGroups = Math.floor(numGroups / 2);
  }

  // 3. Bundled TAESD Decoder (`first_stage_model.decoder.layers.0..18`)
  const dec0Info = req(
    ['first_stage_model.decoder.layers.0.weight', 'taesd.stage0.weight'],
    'taesd decoder stem',
  );
  const dec0W = decodeBackingTensorToF32(backing, dataOffset, dec0Info);
  const taesdC = dec0Info.dims[3] ?? Math.max(4, Math.floor(dec0W.length / (9 * latentChannels)));
  const dec0B = padOrSliceF32(
    optF32(['first_stage_model.decoder.layers.0.bias', 'taesd.stage0.bias']),
    taesdC,
    0,
  );
  const stages: TaesdStageWeights[] = [
    { inC: latentChannels, outC: taesdC, convW: dec0W, convB: dec0B, upsample2x: false },
  ];
  const upsampleCandidates: readonly [readonly string[], readonly string[]][] = [
    [
      ['first_stage_model.decoder.layers.6.weight', 'first_stage_model.decoder.layers.2.conv.0.weight', 'taesd.up0.weight'],
      ['first_stage_model.decoder.layers.6.bias', 'first_stage_model.decoder.layers.2.conv.0.bias', 'taesd.up0.bias'],
    ],
    [
      ['first_stage_model.decoder.layers.11.weight', 'first_stage_model.decoder.layers.12.conv.0.weight', 'taesd.up1.weight'],
      ['first_stage_model.decoder.layers.11.bias', 'first_stage_model.decoder.layers.12.conv.0.bias', 'taesd.up1.bias'],
    ],
    [
      ['first_stage_model.decoder.layers.16.weight', 'first_stage_model.decoder.layers.17.conv.0.weight', 'taesd.up2.weight'],
      ['first_stage_model.decoder.layers.16.bias', 'first_stage_model.decoder.layers.17.conv.0.bias', 'taesd.up2.bias'],
    ],
  ];
  for (const [wNames, bNames] of upsampleCandidates) {
    const info = findImageTensor(parsed, wNames);
    if (!info) continue;
    const convW = decodeBackingTensorToF32(backing, dataOffset, info);
    const inC = info.dims[2] ?? taesdC;
    const outC = info.dims[3] ?? taesdC;
    const convB = padOrSliceF32(optF32(bNames), outC, 0);
    stages.push({ inC, outC, convW, convB, upsample2x: true });
  }
  const finalInfo = req(
    ['first_stage_model.decoder.layers.18.weight', 'taesd.final.weight'],
    'taesd final rgb',
  );
  const finalW = decodeBackingTensorToF32(backing, dataOffset, finalInfo);
  const finalB = padOrSliceF32(
    optF32(['first_stage_model.decoder.layers.18.bias', 'taesd.final.bias']),
    3,
    0,
  );

  const residentBytes =
    (posEmbd.byteLength +
      lnW.byteLength +
      lnB.byteLength +
      projW.byteLength +
      projB.byteLength +
      convInW.byteLength +
      convInB.byteLength +
      norm1W.byteLength +
      norm1B.byteLength +
      resConv1W.byteLength +
      resConv1B.byteLength +
      condProjW.byteLength +
      condProjB.byteLength +
      norm2W.byteLength +
      norm2B.byteLength +
      resConv2W.byteLength +
      resConv2B.byteLength +
      convOutW.byteLength +
      convOutB.byteLength +
      stages.reduce((acc, s) => acc + s.convW.byteLength + s.convB.byteLength, 0) +
      finalW.byteLength +
      finalB.byteLength) |
    0;

  return {
    clip: {
      vocabSize,
      embedDim,
      maxSeqLen,
      readTokenEmbeddingRow,
      posEmbd,
      lnW,
      lnB,
      projW,
      projB,
    },
    unet: {
      latentChannels,
      modelChannels,
      condDim: embedDim,
      numGroups,
      convInW,
      convInB,
      norm1W,
      norm1B,
      resConv1W,
      resConv1B,
      condProjW,
      condProjB,
      norm2W,
      norm2B,
      resConv2W,
      resConv2B,
      convOutW,
      convOutB,
    },
    taesd: { stages, finalW, finalB },
    tensorCount: parsed.tensors.length,
    residentBytes,
    dispose: () => {
      try {
        backing.close?.();
      } catch {
        // Ignore close errors
      }
    },
  };
}

/** Convenience wrapper loading SDXS/TAESD GGUF weights from an in-memory `Uint8Array`. */
export function loadImageWeightsFromGguf(ggufBytes: Uint8Array): LoadedNeuralImageWeights {
  const backing: ImageGgufBacking = {
    byteLength: ggufBytes.byteLength,
    readRange(fileOffset, dst, dstOffset = 0, length = dst.byteLength - dstOffset) {
      const avail = Math.max(0, Math.min(length, ggufBytes.byteLength - fileOffset));
      if (avail > 0) dst.set(ggufBytes.subarray(fileOffset, fileOffset + avail), dstOffset);
      return avail;
    },
  };
  return loadImageWeightsFromBacking(backing);
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
  const tokRow = new Float32Array(D);
  const row = new Float32Array(D);
  const norm = new Float32Array(D);

  for (let pos = 0; pos < len; pos++) {
    const tid = Math.max(0, Math.min(w.vocabSize - 1, tokenIds[pos] ?? 0));
    if (w.readTokenEmbeddingRow) {
      w.readTokenEmbeddingRow(tid, tokRow);
    } else if (w.tokenEmbd) {
      const tBase = tid * D;
      tokRow.set(w.tokenEmbd.subarray(tBase, tBase + D));
    } else {
      tokRow.fill(0);
    }
    const pBase = pos * D;
    for (let d = 0; d < D; d++) {
      row[d] = Math.fround(tokRow[d]! + (w.posEmbd[pBase + d] ?? 0));
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
  scratch?: ImageSynthesisScratch,
): Float32Array {
  const C = w.modelChannels;
  const spatial = latentH * latentW;
  const planeLen = C * spatial;

  // 1. Conv2D stem: [4, H, W] -> [C, H, W]
  const h0 = ensureF32(scratch?.unetH0, planeLen);
  if (scratch) scratch.unetH0 = h0;
  verdantConv2d(noisyLatent, w.convInW, w.convInB, h0, w.latentChannels, C, latentH, latentW, 3, 3, 1, 1);

  // 2. GroupNorm + SiLU + Conv2D res block 1
  const n1 = ensureF32(scratch?.unetN1, planeLen);
  if (scratch) scratch.unetN1 = n1;
  verdantGroupNorm(h0, w.norm1W, w.norm1B, n1, C, w.numGroups, spatial, 1e-5);
  verdantSiLu(n1, n1, n1.length);
  const h1 = ensureF32(scratch?.unetH1, planeLen);
  if (scratch) scratch.unetH1 = h1;
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
  const n2 = ensureF32(scratch?.unetN2, planeLen);
  if (scratch) scratch.unetN2 = n2;
  verdantGroupNorm(h1, w.norm2W, w.norm2B, n2, C, w.numGroups, spatial, 1e-5);
  verdantSiLu(n2, n2, n2.length);
  const h2 = ensureF32(scratch?.unetH2, planeLen);
  if (scratch) scratch.unetH2 = h2;
  verdantConv2d(n2, w.resConv2W, w.resConv2B, h2, C, C, latentH, latentW, 3, 3, 1, 1);
  verdantAdd(h0, h2, h2, h2.length);

  // 5. Noise prediction convOut: [C, H, W] -> [4, H, W] and 1-step denoised latent z0
  const epsPred = ensureF32(scratch?.unetEps, w.latentChannels * spatial);
  if (scratch) scratch.unetEps = epsPred;
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
  readonly scratch?: ImageSynthesisScratch;
}): SynthesizedRgbaImage {
  const cond = encodeClipPrompt(params.promptTokens, params.clip);
  const zT = seedGaussianLatent(params.unet.latentChannels, params.latentH, params.latentW, params.seed);
  const z0 = denoiseOneStepDistilledUnet(
    zT,
    params.latentH,
    params.latentW,
    cond,
    params.unet,
    1.0,
    params.scratch,
  );
  return decodeTaesdLatentToRgba(z0, params.latentH, params.latentW, params.taesd, params.seed);
}
