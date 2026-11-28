/**
 * Real 1-Step Distilled Latent Diffusion & TAESD Decoder (`src/engine/imageSynthesisEngine.ts`)
 *
 * Executes real neural image synthesis (`CLIP text encoder` -> `1-step distilled SDXS / SD-Turbo UNet`
 * -> `TAESD Tiny AutoEncoder Conv2D + 2x Upsample RGB decode`) strictly on VerdantUI's shared
 * `vTensor` kernels (`verdantConv2d`, `verdantGroupNorm`, `verdantSiLu`, `verdantLayerNorm`,
 * `verdantGelu`, `verdantAdd`).
 */

import {
  GgmlType,
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
import {
  verdantConv2dSimd,
  verdantLinearBatchSimd,
  type VerdantWasmArena,
} from './verdantWasmSimd';
import { getSharedOpfsCasStore } from './vramCasStore';

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

export interface StreamedSdxsGraphHandle {
  readonly backing: ImageGgufBacking;
  readonly parsed: ParsedGgufModel;
  readonly dataOffset: number;
  readonly clipLayerIndices: readonly number[];
  readonly inputBlockIndices: readonly number[];
  readonly outputBlockIndices: readonly number[];
  readonly hasMiddleBlock: boolean;
  readonly taesdResidualIndices: readonly number[];
  readonly clipBpeVocab?: ClipBpeVocabulary | null;
}

export interface LoadedNeuralImageWeights {
  readonly clip: ClipTextWeights;
  readonly unet: DistilledUnetWeights;
  readonly taesd: TaesdDecoderWeights;
  readonly tensorCount: number;
  readonly residentBytes: number;
  /** Optional OPFS-streamed full-graph handle for 12-layer CLIP + cross-attention UNet + 3-stage TAESD. */
  readonly streamGraph?: StreamedSdxsGraphHandle;
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

/** OpenAI CLIP byte-level BPE vocabulary (`0 KiB` static bundle footprint; loaded from GGUF KV or OPFS). */
export interface ClipBpeVocabulary {
  readonly encoder: ReadonlyMap<string, number>;
  readonly bpeRanks: ReadonlyMap<string, number>;
  readonly bosTokenId: number;
  readonly eosTokenId: number;
  readonly source: 'gguf-kv' | 'opfs-cache' | 'custom';
}

const CLIP_BYTE_TO_UNICODE: readonly string[] = (() => {
  const bs: number[] = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n += 1;
    }
  }
  const table: string[] = new Array(256);
  for (let i = 0; i < bs.length; i++) {
    table[bs[i]!] = String.fromCharCode(cs[i]!);
  }
  return table;
})();

const clipUtf8Encoder = new TextEncoder();

/** Parses OpenAI CLIP `vocab.json` and `merges.txt` into a fast BPE lookup structure. */
export function parseClipBpeFromJsonAndMerges(
  vocabJson: string,
  mergesText: string,
  source: ClipBpeVocabulary['source'] = 'custom',
): ClipBpeVocabulary {
  const raw = JSON.parse(vocabJson) as Record<string, number>;
  const encoder = new Map<string, number>();
  for (const [k, v] of Object.entries(raw)) {
    if (typeof v === 'number' && Number.isFinite(v)) encoder.set(k, v);
  }
  const bpeRanks = new Map<string, number>();
  const lines = mergesText.split(/\r?\n/);
  let rank = 0;
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const parts = line.split(/\s+/);
    if (parts.length >= 2) {
      bpeRanks.set(`${parts[0]}\u0000${parts[1]}`, rank++);
    }
  }
  return {
    encoder,
    bpeRanks,
    bosTokenId: encoder.get('<|startoftext|>') ?? 49406,
    eosTokenId: encoder.get('<|endoftext|>') ?? 49407,
    source,
  };
}

/** Extracts CLIP BPE vocabulary directly from GGUF metadata KV when embedded. */
export function parseClipBpeFromGguf(parsed: ParsedGgufModel): ClipBpeVocabulary | null {
  const tokens = parsed.metadata['tokenizer.ggml.tokens'];
  if (!Array.isArray(tokens) || tokens.length < 256) return null;
  const encoder = new Map<string, number>();
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    if (typeof t === 'string') encoder.set(t, i);
  }
  const merges = parsed.metadata['tokenizer.ggml.merges'];
  const bpeRanks = new Map<string, number>();
  if (Array.isArray(merges)) {
    let rank = 0;
    for (const m of merges) {
      if (typeof m !== 'string') continue;
      const parts = m.trim().split(/\s+/);
      if (parts.length >= 2) bpeRanks.set(`${parts[0]}\u0000${parts[1]}`, rank++);
    }
  }
  return {
    encoder,
    bpeRanks,
    bosTokenId: encoder.get('<|startoftext|>') ?? Math.min(49406, tokens.length - 2),
    eosTokenId: encoder.get('<|endoftext|>') ?? Math.min(49407, tokens.length - 1),
    source: 'gguf-kv',
  };
}

/** Exact OpenAI CLIP byte-level BPE encoder (`BOS=49406`, `EOS=49407`, `</w>` word boundary). */
export function encodeClipBpeTokens(
  prompt: string,
  vocab: ClipBpeVocabulary,
  maxSeqLen = 77,
): number[] {
  const out: number[] = [vocab.bosTokenId];
  const cleaned = prompt.trim().replace(/\s+/g, ' ').toLowerCase();
  const wordRe = /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-z]+|[0-9]|[^\s a-z0-9]+/g;
  const matches = cleaned.match(wordRe) ?? [];

  for (const token of matches) {
    if (out.length >= maxSeqLen - 1) break;
    const utf8 = clipUtf8Encoder.encode(token);
    const chars: string[] = [];
    for (let i = 0; i < utf8.length; i++) {
      chars.push(CLIP_BYTE_TO_UNICODE[utf8[i]!]!);
    }
    if (chars.length === 0) continue;
    chars[chars.length - 1] = `${chars[chars.length - 1]}</w>`;

    let word = chars;
    while (word.length > 1) {
      let bestRank = Infinity;
      let bestFirst = '';
      let bestSecond = '';
      for (let i = 0; i < word.length - 1; i++) {
        const r = vocab.bpeRanks.get(`${word[i]}\u0000${word[i + 1]}`);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          bestFirst = word[i]!;
          bestSecond = word[i + 1]!;
        }
      }
      if (!Number.isFinite(bestRank)) break;
      const merged: string[] = [];
      let i = 0;
      while (i < word.length) {
        if (i < word.length - 1 && word[i] === bestFirst && word[i + 1] === bestSecond) {
          merged.push(bestFirst + bestSecond);
          i += 2;
        } else {
          merged.push(word[i]!);
          i += 1;
        }
      }
      word = merged;
    }

    for (const piece of word) {
      if (out.length >= maxSeqLen - 1) break;
      const id = vocab.encoder.get(piece);
      if (id !== undefined) {
        out.push(id);
      }
    }
  }
  out.push(vocab.eosTokenId);
  return out;
}

const CLIP_BPE_VOCAB_URL =
  'https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/vocab.json';
const CLIP_BPE_MERGES_URL =
  'https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/merges.txt';
let cachedClipBpeVocab: ClipBpeVocabulary | null = null;
let clipBpeLoadPromise: Promise<ClipBpeVocabulary | null> | null = null;

/**
 * Lazily loads/caches standard OpenAI CLIP BPE vocab & merges in OPFS (`0 KiB` static bundle bloat).
 * When `allowNetwork` is false, reads strictly from OPFS cache if already persisted.
 */
export async function ensureClipBpeVocabularyCached(
  allowNetwork: boolean,
): Promise<ClipBpeVocabulary | null> {
  if (cachedClipBpeVocab) return cachedClipBpeVocab;
  if (clipBpeLoadPromise) return clipBpeLoadPromise;
  clipBpeLoadPromise = (async () => {
    try {
      const storage = typeof navigator !== 'undefined' ? navigator.storage : undefined;
      let dir: FileSystemDirectoryHandle | null = null;
      if (storage && typeof storage.getDirectory === 'function') {
        try {
          const root = await storage.getDirectory();
          dir = await root.getDirectoryHandle('verdant-vram-v1', { create: true });
          const [vH, mH] = await Promise.all([
            dir.getFileHandle('clip-bpe-vocab.json'),
            dir.getFileHandle('clip-bpe-merges.txt'),
          ]);
          const [vText, mText] = await Promise.all([
            (await vH.getFile()).text(),
            (await mH.getFile()).text(),
          ]);
          if (vText.length > 1024 && mText.length > 1024) {
            cachedClipBpeVocab = parseClipBpeFromJsonAndMerges(vText, mText, 'opfs-cache');
            return cachedClipBpeVocab;
          }
        } catch {
          // Cache miss in OPFS
        }
      }
      if (!allowNetwork || typeof fetch !== 'function') return null;
      const [vResp, mResp] = await Promise.all([
        fetch(CLIP_BPE_VOCAB_URL),
        fetch(CLIP_BPE_MERGES_URL),
      ]);
      if (!vResp.ok || !mResp.ok) return null;
      const [vText, mText] = await Promise.all([vResp.text(), mResp.text()]);
      const parsed = parseClipBpeFromJsonAndMerges(vText, mText, 'opfs-cache');
      // Register CLIP BPE assets inside the unified Content-Addressed OPFS store for global dedup
      const cas = getSharedOpfsCasStore();
      void Promise.all([
        cas.putObject({
          kind: 'tokenizer-bpe',
          bytes: clipUtf8Encoder.encode(vText),
          ownerId: 'system:clip-bpe-vocab',
          salienceQ10: 960,
          sourceUrl: CLIP_BPE_VOCAB_URL,
        }),
        cas.putObject({
          kind: 'tokenizer-bpe',
          bytes: clipUtf8Encoder.encode(mText),
          ownerId: 'system:clip-bpe-merges',
          salienceQ10: 960,
          sourceUrl: CLIP_BPE_MERGES_URL,
        }),
      ]).catch(() => {});
      if (dir) {
        try {
          const writeText = async (name: string, content: string) => {
            const fh = await dir!.getFileHandle(name, { create: true });
            const wr = await fh.createWritable();
            await wr.write(content);
            await wr.close();
          };
          await Promise.all([
            writeText('clip-bpe-vocab.json', vText),
            writeText('clip-bpe-merges.txt', mText),
          ]);
        } catch {
          // Keep in-memory vocabulary even if OPFS write is unavailable
        }
      }
      cachedClipBpeVocab = parsed;
      return parsed;
    } catch {
      return null;
    } finally {
      clipBpeLoadPromise = null;
    }
  })();
  return clipBpeLoadPromise;
}

/**
 * CLIP prompt tokenizer: requires a verified byte-level CLIP BPE vocabulary
 * (`BOS=49406`, `EOS=49407`) and refuses explicitly when unavailable (zero hash guessing).
 */
export function encodePromptToClipTokens(
  prompt: string,
  _vocabSize: number,
  maxSeqLen = 77,
  bpeVocab?: ClipBpeVocabulary | null,
): number[] {
  if (!bpeVocab) {
    throw new Error(
      'refused: CLIP BPE vocabulary is required to encode image prompts; hash-based token guessing is not permitted',
    );
  }
  return encodeClipBpeTokens(prompt, bpeVocab, maxSeqLen);
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

  // Discover full OPFS-streamable CLIP layers, UNet input/middle/output blocks & TAESD residual blocks
  const clipLayerSet = new Set<number>();
  const inputBlockSet = new Set<number>();
  const outputBlockSet = new Set<number>();
  const taesdResSet = new Set<number>();
  let hasMiddleBlock = false;
  for (const t of parsed.tensors) {
    const clipM = t.name.match(/^cond_stage_model\.transformer\.text_model\.encoder\.layers\.(\d+)\./);
    if (clipM) clipLayerSet.add(Number(clipM[1]));
    const inM = t.name.match(/^model\.diffusion_model\.input_blocks\.(\d+)\./);
    if (inM) inputBlockSet.add(Number(inM[1]));
    const outM = t.name.match(/^model\.diffusion_model\.output_blocks\.(\d+)\./);
    if (outM) outputBlockSet.add(Number(outM[1]));
    if (t.name.startsWith('model.diffusion_model.middle_block.')) hasMiddleBlock = true;
    const trM = t.name.match(/^first_stage_model\.decoder\.layers\.(\d+)\.conv\.0\.weight$/);
    if (trM) taesdResSet.add(Number(trM[1]));
  }

  const streamGraph: StreamedSdxsGraphHandle = {
    backing,
    parsed,
    dataOffset,
    clipLayerIndices: Array.from(clipLayerSet).sort((a, b) => a - b),
    inputBlockIndices: Array.from(inputBlockSet).sort((a, b) => a - b),
    outputBlockIndices: Array.from(outputBlockSet).sort((a, b) => a - b),
    hasMiddleBlock,
    taesdResidualIndices: Array.from(taesdResSet).sort((a, b) => a - b),
    clipBpeVocab: parseClipBpeFromGguf(parsed),
  };

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
    streamGraph,
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

// ============================================================================
// Milestone B: Layer-Streamed Full SDXS-512 Graph over OPFS (`610` tensors)
//   - 12-Layer Causal CLIP Text Transformer (sequence + pooled conditioning)
//   - Sinusoidal Timestep Embedding + Spatial Multi-Head Cross-Attention
//   - Layer-Streamed UNet (input_blocks -> middle_block -> skip output_blocks -> out)
//   - Progressive 3-Stage Residual TAESD (128 / 256 / 384 / 512px) with 16.6ms yields
// ============================================================================

function linearMatVec(
  x: Float32Array,
  xOff: number,
  w: Float32Array,
  b: Float32Array | null,
  out: Float32Array,
  outOff: number,
  inDim: number,
  outDim: number,
): void {
  for (let r = 0; r < outDim; r++) {
    let acc = b ? (b[r] ?? 0) : 0;
    const rowBase = r * inDim;
    for (let c = 0; c < inDim; c++) {
      acc += (w[rowBase + c] ?? 0) * (x[xOff + c] ?? 0);
    }
    out[outOff + r] = Math.fround(acc);
  }
}

/** Exact DDPM/SDXS sinusoidal timestep embedding `[dim]`. */
export function sinusoidalTimestepEmbedding(timestep: number, dim: number): Float32Array {
  const out = new Float32Array(dim);
  const half = Math.floor(dim / 2);
  const logMax = Math.log(10000);
  for (let i = 0; i < half; i++) {
    const freq = Math.exp((-logMax * i) / Math.max(1, half));
    const arg = timestep * freq;
    out[i] = Math.fround(Math.cos(arg));
    out[half + i] = Math.fround(Math.sin(arg));
  }
  return out;
}

/**
 * Multi-head Spatial Cross-Attention (`SDXS UNet SpatialTransformer`):
 * Attends spatial tokens `x` (`[C, H*W]`, CHW) against CLIP sequence tokens `condSeq` (`[seqLen, condDim]`).
 */
export function applySpatialCrossAttention(params: {
  readonly x: Float32Array;
  readonly channels: number;
  readonly spatial: number;
  readonly condSeq: Float32Array;
  readonly seqLen: number;
  readonly condDim: number;
  readonly numHeads: number;
  readonly toQW: Float32Array;
  readonly toKW: Float32Array;
  readonly toVW: Float32Array;
  readonly toOutW: Float32Array;
  readonly toOutB: Float32Array | null;
  readonly out: Float32Array;
}): void {
  const {
    x,
    channels: C,
    spatial: S,
    condSeq,
    seqLen: L,
    condDim,
    numHeads,
    toQW,
    toKW,
    toVW,
    toOutW,
    toOutB,
    out,
  } = params;
  const heads = Math.max(1, C % numHeads === 0 ? numHeads : 1);
  const headDim = Math.max(1, Math.floor(C / heads));
  const invScale = 1.0 / Math.sqrt(headDim);

  // Project K and V once for the CLIP token sequence: [L, C]
  const K = new Float32Array(L * C);
  const V = new Float32Array(L * C);
  for (let t = 0; t < L; t++) {
    linearMatVec(condSeq, t * condDim, toKW, null, K, t * C, condDim, C);
    linearMatVec(condSeq, t * condDim, toVW, null, V, t * C, condDim, C);
  }

  const tok = new Float32Array(C);
  const qVec = new Float32Array(C);
  const attnVec = new Float32Array(C);
  const projOut = new Float32Array(C);
  const logits = new Float32Array(L);

  for (let s = 0; s < S; s++) {
    for (let c = 0; c < C; c++) tok[c] = x[c * S + s]!;
    linearMatVec(tok, 0, toQW, null, qVec, 0, C, C);
    attnVec.fill(0);

    for (let h = 0; h < heads; h++) {
      const hOff = h * headDim;
      let maxLogit = -Infinity;
      for (let t = 0; t < L; t++) {
        let dot = 0.0;
        const kOff = t * C + hOff;
        for (let d = 0; d < headDim; d++) {
          dot += qVec[hOff + d]! * K[kOff + d]!;
        }
        const scaled = dot * invScale;
        logits[t] = scaled;
        if (scaled > maxLogit) maxLogit = scaled;
      }
      let sumExp = 0.0;
      for (let t = 0; t < L; t++) {
        const e = Math.exp(logits[t]! - maxLogit);
        logits[t] = e;
        sumExp += e;
      }
      const invSum = sumExp > 0 ? 1.0 / sumExp : 0;
      for (let t = 0; t < L; t++) {
        const w = logits[t]! * invSum;
        const vOff = t * C + hOff;
        for (let d = 0; d < headDim; d++) {
          attnVec[hOff + d] = Math.fround(attnVec[hOff + d]! + w * V[vOff + d]!);
        }
      }
    }

    linearMatVec(attnVec, 0, toOutW, toOutB, projOut, 0, C, C);
    for (let c = 0; c < C; c++) {
      out[c * S + s] = Math.fround(tok[c]! + projOut[c]!);
    }
  }
}

/**
 * Layer-streamed multi-layer CLIP text encoder over OPFS backing (`layers.0..N-1`).
 * Decodes one transformer layer at a time so peak RAM stays tiny while emitting BOTH
 * full sequence embeddings `[seqLen, embedDim]` (for UNet cross-attention) and pooled `[embedDim]`.
 */
export async function encodeClipSequenceStreamed(
  tokenIds: readonly number[],
  loaded: LoadedNeuralImageWeights,
  opts?: {
    readonly yieldFrame?: () => Promise<void>;
    readonly signal?: AbortSignal;
    readonly arena?: VerdantWasmArena | null;
  },
): Promise<{
  readonly sequence: Float32Array;
  readonly seqLen: number;
  readonly pooled: Float32Array;
  readonly layersExecuted: number;
}> {
  const { clip, streamGraph } = loaded;
  const D = clip.embedDim;
  const seqLen = Math.max(1, Math.min(tokenIds.length, clip.maxSeqLen));
  const hidden = new Float32Array(seqLen * D);
  const tokRow = new Float32Array(D);

  for (let pos = 0; pos < seqLen; pos++) {
    const tid = Math.max(0, Math.min(clip.vocabSize - 1, tokenIds[pos] ?? 0));
    if (clip.readTokenEmbeddingRow) {
      clip.readTokenEmbeddingRow(tid, tokRow);
    } else if (clip.tokenEmbd) {
      tokRow.set(clip.tokenEmbd.subarray(tid * D, tid * D + D));
    } else {
      tokRow.fill(0);
    }
    const pBase = pos * D;
    for (let d = 0; d < D; d++) {
      hidden[pBase + d] = Math.fround(tokRow[d]! + (clip.posEmbd[pBase + d] ?? 0));
    }
  }

  let layersExecuted = 0;
  if (streamGraph && streamGraph.clipLayerIndices.length > 0) {
    const { backing, parsed, dataOffset } = streamGraph;
    const loadOpt = (name: string): Float32Array | null => {
      const info = parsed.tensorMap.get(name);
      return info ? decodeBackingTensorToF32(backing, dataOffset, info) : null;
    };
    const normBuf = new Float32Array(seqLen * D);
    const Q = new Float32Array(seqLen * D);
    const K = new Float32Array(seqLen * D);
    const V = new Float32Array(seqLen * D);
    const attn = new Float32Array(seqLen * D);
    const projBatch = new Float32Array(seqLen * D);
    const heads = D >= 64 && D % 8 === 0 ? 8 : Math.max(1, D % 4 === 0 ? 4 : 1);
    const headDim = Math.floor(D / heads);
    const invScale = 1.0 / Math.sqrt(headDim);
    const logits = new Float32Array(seqLen);

    for (const layerIdx of streamGraph.clipLayerIndices) {
      if (opts?.signal?.aborted) break;
      const prefix = `cond_stage_model.transformer.text_model.encoder.layers.${layerIdx}`;
      const qWInfo = parsed.tensorMap.get(`${prefix}.self_attn.q_proj.weight`);
      const kWInfo = parsed.tensorMap.get(`${prefix}.self_attn.k_proj.weight`);
      const vWInfo = parsed.tensorMap.get(`${prefix}.self_attn.v_proj.weight`);
      const oWInfo = parsed.tensorMap.get(`${prefix}.self_attn.out_proj.weight`);
      if (!qWInfo || !kWInfo || !vWInfo || !oWInfo) continue;

      const ln1W = padOrSliceF32(loadOpt(`${prefix}.layer_norm1.weight`), D, 1);
      const ln1B = padOrSliceF32(loadOpt(`${prefix}.layer_norm1.bias`), D, 0);
      for (let t = 0; t < seqLen; t++) {
        verdantLayerNorm(
          hidden.subarray(t * D, (t + 1) * D),
          ln1W,
          ln1B,
          normBuf.subarray(t * D, (t + 1) * D),
          D,
          1e-5,
        );
      }
      runStreamedLinearBatch({
        backing,
        dataOffset,
        wInfo: qWInfo,
        bInfo: parsed.tensorMap.get(`${prefix}.self_attn.q_proj.bias`) ?? null,
        input: normBuf,
        out: Q,
        inDim: D,
        outDim: D,
        batchSize: seqLen,
        arena: opts?.arena,
      });
      runStreamedLinearBatch({
        backing,
        dataOffset,
        wInfo: kWInfo,
        bInfo: parsed.tensorMap.get(`${prefix}.self_attn.k_proj.bias`) ?? null,
        input: normBuf,
        out: K,
        inDim: D,
        outDim: D,
        batchSize: seqLen,
        arena: opts?.arena,
      });
      runStreamedLinearBatch({
        backing,
        dataOffset,
        wInfo: vWInfo,
        bInfo: parsed.tensorMap.get(`${prefix}.self_attn.v_proj.bias`) ?? null,
        input: normBuf,
        out: V,
        inDim: D,
        outDim: D,
        batchSize: seqLen,
        arena: opts?.arena,
      });

      // Causal self-attention across [0..t]
      attn.fill(0);
      for (let t = 0; t < seqLen; t++) {
        for (let h = 0; h < heads; h++) {
          const hOff = h * headDim;
          let maxL = -Infinity;
          for (let j = 0; j <= t; j++) {
            let dot = 0;
            for (let d = 0; d < headDim; d++) {
              dot += Q[t * D + hOff + d]! * K[j * D + hOff + d]!;
            }
            const sc = dot * invScale;
            logits[j] = sc;
            if (sc > maxL) maxL = sc;
          }
          let sumE = 0;
          for (let j = 0; j <= t; j++) {
            const e = Math.exp(logits[j]! - maxL);
            logits[j] = e;
            sumE += e;
          }
          const invSum = sumE > 0 ? 1 / sumE : 0;
          for (let j = 0; j <= t; j++) {
            const w = logits[j]! * invSum;
            for (let d = 0; d < headDim; d++) {
              attn[t * D + hOff + d] = Math.fround(
                attn[t * D + hOff + d]! + w * V[j * D + hOff + d]!,
              );
            }
          }
        }
      }
      runStreamedLinearBatch({
        backing,
        dataOffset,
        wInfo: oWInfo,
        bInfo: parsed.tensorMap.get(`${prefix}.self_attn.out_proj.bias`) ?? null,
        input: attn,
        out: projBatch,
        inDim: D,
        outDim: D,
        batchSize: seqLen,
        arena: opts?.arena,
      });
      verdantAdd(hidden, projBatch, hidden, hidden.length);

      // Feed-forward MLP (layer_norm2 -> fc1 -> GELU -> fc2 + residual, native Q8_0 SIMD!)
      const fc1Info = parsed.tensorMap.get(`${prefix}.mlp.fc1.weight`);
      const fc2Info = parsed.tensorMap.get(`${prefix}.mlp.fc2.weight`);
      if (fc1Info && fc2Info) {
        const ln2W = padOrSliceF32(loadOpt(`${prefix}.layer_norm2.weight`), D, 1);
        const ln2B = padOrSliceF32(loadOpt(`${prefix}.layer_norm2.bias`), D, 0);
        const mlpDim = fc1Info.dims[1] ?? D * 4;
        const mlpMid = new Float32Array(seqLen * mlpDim);
        for (let t = 0; t < seqLen; t++) {
          verdantLayerNorm(
            hidden.subarray(t * D, (t + 1) * D),
            ln2W,
            ln2B,
            normBuf.subarray(t * D, (t + 1) * D),
            D,
            1e-5,
          );
        }
        runStreamedLinearBatch({
          backing,
          dataOffset,
          wInfo: fc1Info,
          bInfo: parsed.tensorMap.get(`${prefix}.mlp.fc1.bias`) ?? null,
          input: normBuf,
          out: mlpMid,
          inDim: D,
          outDim: mlpDim,
          batchSize: seqLen,
          arena: opts?.arena,
        });
        verdantGelu(mlpMid, mlpMid, mlpMid.length, true);
        runStreamedLinearBatch({
          backing,
          dataOffset,
          wInfo: fc2Info,
          bInfo: parsed.tensorMap.get(`${prefix}.mlp.fc2.bias`) ?? null,
          input: mlpMid,
          out: projBatch,
          inDim: mlpDim,
          outDim: D,
          batchSize: seqLen,
          arena: opts?.arena,
        });
        verdantAdd(hidden, projBatch, hidden, hidden.length);
      }

      layersExecuted += 1;
      if (opts?.yieldFrame) await opts.yieldFrame();
    }
  }

  // Final layer norm across sequence + EOS/mean pooled vector
  const normRow = new Float32Array(D);
  const pooled = new Float32Array(D);
  for (let t = 0; t < seqLen; t++) {
    verdantLayerNorm(hidden.subarray(t * D, (t + 1) * D), clip.lnW, clip.lnB, normRow, D, 1e-5);
    hidden.set(normRow, t * D);
    for (let d = 0; d < D; d++) {
      pooled[d] = Math.fround(pooled[d]! + normRow[d]! / seqLen);
    }
  }
  if (layersExecuted === 0) {
    // Minimal synthetic GGUF fallback: apply legacy pooled projection
    const projected = new Float32Array(D);
    linearMatVec(pooled, 0, clip.projW, clip.projB, projected, 0, D, D);
    pooled.set(projected);
  }

  return { sequence: hidden, seqLen, pooled, layersExecuted };
}

function readBackingRawTensor(
  backing: ImageGgufBacking,
  dataOffset: number,
  info: GgufTensorInfo,
): Uint8Array {
  const raw = new Uint8Array(info.byteSize);
  const got = backing.readRange(dataOffset + info.offset, raw, 0, info.byteSize);
  if (got < info.byteSize) {
    throw new Error(`Truncated GGUF tensor "${info.name}" (${got}/${info.byteSize} bytes).`);
  }
  return raw;
}

function runStreamedLinearBatch(params: {
  readonly backing: ImageGgufBacking;
  readonly dataOffset: number;
  readonly wInfo: GgufTensorInfo;
  readonly bInfo: GgufTensorInfo | null;
  readonly input: Float32Array;
  readonly out: Float32Array;
  readonly inDim: number;
  readonly outDim: number;
  readonly batchSize: number;
  readonly arena?: VerdantWasmArena | null;
}): void {
  const { backing, dataOffset, wInfo, bInfo, input, out, inDim, outDim, batchSize, arena } = params;
  const bias = bInfo ? decodeBackingTensorToF32(backing, dataOffset, bInfo) : null;
  if (
    wInfo.type === GgmlType.Q8_0 ||
    wInfo.type === GgmlType.F16 ||
    wInfo.type === GgmlType.F32
  ) {
    const raw = readBackingRawTensor(backing, dataOffset, wInfo);
    const usedSimd = verdantLinearBatchSimd({
      input,
      weightRaw: raw,
      weightType: wInfo.type,
      bias,
      out,
      inDim,
      outDim,
      batchSize,
      arena,
    });
    if (usedSimd) return;
  }
  const wF32 = decodeBackingTensorToF32(backing, dataOffset, wInfo);
  for (let b = 0; b < batchSize; b++) {
    linearMatVec(input, b * inDim, wF32, bias, out, b * outDim, inDim, outDim);
  }
}

function runStreamedConv2d(params: {
  readonly backing: ImageGgufBacking;
  readonly dataOffset: number;
  readonly wInfo: GgufTensorInfo;
  readonly bInfo: GgufTensorInfo | null;
  readonly input: Float32Array;
  readonly out: Float32Array;
  readonly inC: number;
  readonly outC: number;
  readonly height: number;
  readonly width: number;
  readonly kH: number;
  readonly kW: number;
  readonly stride?: number;
  readonly padding?: number;
  readonly arena?: VerdantWasmArena | null;
}): { readonly outH: number; readonly outW: number } {
  const {
    backing,
    dataOffset,
    wInfo,
    bInfo,
    input,
    out,
    inC,
    outC,
    height,
    width,
    kH,
    kW,
    stride = 1,
    padding = 0,
    arena,
  } = params;
  const bias = bInfo ? decodeBackingTensorToF32(backing, dataOffset, bInfo) : null;
  if (wInfo.type === GgmlType.F16 || wInfo.type === GgmlType.F32) {
    const raw = readBackingRawTensor(backing, dataOffset, wInfo);
    const res = verdantConv2dSimd({
      input,
      weightRaw: raw,
      weightType: wInfo.type,
      bias,
      out,
      inC,
      outC,
      height,
      width,
      kH,
      kW,
      stride,
      padding,
      arena,
    });
    return { outH: res.outH, outW: res.outW };
  }
  const wF32 = decodeBackingTensorToF32(backing, dataOffset, wInfo);
  return verdantConv2d(input, wF32, bias, out, inC, outC, height, width, kH, kW, stride, padding);
}

function concatChannelsCHW(
  a: Float32Array,
  aC: number,
  b: Float32Array,
  bC: number,
  spatial: number,
): Float32Array {
  const out = new Float32Array((aC + bC) * spatial);
  out.set(a.subarray(0, aC * spatial), 0);
  out.set(b.subarray(0, bC * spatial), aC * spatial);
  return out;
}

function pickGroupCount(channels: number): number {
  let g = 32;
  while (g > 1 && channels % g !== 0) g = Math.floor(g / 2);
  return g;
}

function runResBlockStreamed(params: {
  readonly x: Float32Array;
  readonly inC: number;
  readonly height: number;
  readonly width: number;
  readonly prefix: string;
  readonly tEmb: Float32Array;
  readonly backing: ImageGgufBacking;
  readonly parsed: ParsedGgufModel;
  readonly dataOffset: number;
  readonly arena?: VerdantWasmArena | null;
}): { readonly out: Float32Array; readonly outC: number; readonly peakBytes: number } {
  const { x, inC, height, width, prefix, tEmb, backing, parsed, dataOffset, arena } = params;
  const spatial = height * width;
  const conv1Info = parsed.tensorMap.get(`${prefix}.in_layers.2.weight`)!;
  const outC = conv1Info.dims[3] ?? inC;

  const optF32 = (name: string): Float32Array | null => {
    const info = parsed.tensorMap.get(name);
    return info ? decodeBackingTensorToF32(backing, dataOffset, info) : null;
  };

  // 1. GroupNorm + SiLU + Conv2d 3x3 (inC -> outC)
  const n1 = new Float32Array(inC * spatial);
  const gn1W = padOrSliceF32(optF32(`${prefix}.in_layers.0.weight`), inC, 1);
  const gn1B = padOrSliceF32(optF32(`${prefix}.in_layers.0.bias`), inC, 0);
  verdantGroupNorm(x, gn1W, gn1B, n1, inC, pickGroupCount(inC), spatial, 1e-5);
  verdantSiLu(n1, n1, n1.length);

  const h = new Float32Array(outC * spatial);
  runStreamedConv2d({
    backing,
    dataOffset,
    wInfo: conv1Info,
    bInfo: parsed.tensorMap.get(`${prefix}.in_layers.2.bias`) ?? null,
    input: n1,
    out: h,
    inC,
    outC,
    height,
    width,
    kH: 3,
    kW: 3,
    stride: 1,
    padding: 1,
    arena,
  });

  // 2. Timestep conditioning (SiLU(tEmb) -> emb_layers.1 -> broadcast per channel)
  const embWInfo = parsed.tensorMap.get(`${prefix}.emb_layers.1.weight`);
  if (embWInfo && tEmb.length > 0) {
    const tInDim = embWInfo.dims[0] ?? tEmb.length;
    const tAct = new Float32Array(tInDim);
    const copyLen = Math.min(tInDim, tEmb.length);
    for (let i = 0; i < copyLen; i++) tAct[i] = tEmb[i]!;
    verdantSiLu(tAct, tAct, tInDim);
    const embOut = new Float32Array(outC);
    runStreamedLinearBatch({
      backing,
      dataOffset,
      wInfo: embWInfo,
      bInfo: parsed.tensorMap.get(`${prefix}.emb_layers.1.bias`) ?? null,
      input: tAct,
      out: embOut,
      inDim: tInDim,
      outDim: outC,
      batchSize: 1,
      arena,
    });
    for (let c = 0; c < outC; c++) {
      const add = embOut[c]!;
      const base = c * spatial;
      for (let s = 0; s < spatial; s++) h[base + s] = Math.fround(h[base + s]! + add);
    }
  }

  // 3. GroupNorm + SiLU + Conv2d 3x3 (outC -> outC)
  const n2 = new Float32Array(outC * spatial);
  const gn2W = padOrSliceF32(optF32(`${prefix}.out_layers.0.weight`), outC, 1);
  const gn2B = padOrSliceF32(optF32(`${prefix}.out_layers.0.bias`), outC, 0);
  verdantGroupNorm(h, gn2W, gn2B, n2, outC, pickGroupCount(outC), spatial, 1e-5);
  verdantSiLu(n2, n2, n2.length);

  const conv2Info = parsed.tensorMap.get(`${prefix}.out_layers.3.weight`);
  const out = new Float32Array(outC * spatial);
  if (conv2Info) {
    runStreamedConv2d({
      backing,
      dataOffset,
      wInfo: conv2Info,
      bInfo: parsed.tensorMap.get(`${prefix}.out_layers.3.bias`) ?? null,
      input: n2,
      out,
      inC: outC,
      outC,
      height,
      width,
      kH: 3,
      kW: 3,
      stride: 1,
      padding: 1,
      arena,
    });
  } else {
    out.set(n2);
  }

  // 4. Residual skip (1x1 skip_connection when inC !== outC)
  const skipInfo = parsed.tensorMap.get(`${prefix}.skip_connection.weight`);
  if (skipInfo) {
    const skipBuf = new Float32Array(outC * spatial);
    runStreamedConv2d({
      backing,
      dataOffset,
      wInfo: skipInfo,
      bInfo: parsed.tensorMap.get(`${prefix}.skip_connection.bias`) ?? null,
      input: x,
      out: skipBuf,
      inC,
      outC,
      height,
      width,
      kH: 1,
      kW: 1,
      stride: 1,
      padding: 0,
      arena,
    });
    verdantAdd(out, skipBuf, out, out.length);
  } else if (inC === outC) {
    verdantAdd(out, x, out, out.length);
  }

  const peakBytes = n1.byteLength + h.byteLength + n2.byteLength + out.byteLength;
  return { out, outC, peakBytes };
}

function runSpatialTransformerStreamed(params: {
  readonly x: Float32Array;
  readonly channels: number;
  readonly height: number;
  readonly width: number;
  readonly prefix: string;
  readonly condSeq: Float32Array;
  readonly seqLen: number;
  readonly condDim: number;
  readonly backing: ImageGgufBacking;
  readonly parsed: ParsedGgufModel;
  readonly dataOffset: number;
  readonly arena?: VerdantWasmArena | null;
}): { readonly out: Float32Array; readonly executed: boolean } {
  const { x, channels: C, height, width, prefix, condSeq, seqLen, condDim, backing, parsed, dataOffset, arena } =
    params;
  const spatial = height * width;
  const tbPrefix = `${prefix}.transformer_blocks.0`;
  const q2Info = parsed.tensorMap.get(`${tbPrefix}.attn2.to_q.weight`);
  const k2Info = parsed.tensorMap.get(`${tbPrefix}.attn2.to_k.weight`);
  const v2Info = parsed.tensorMap.get(`${tbPrefix}.attn2.to_v.weight`);
  const o2Info = parsed.tensorMap.get(`${tbPrefix}.attn2.to_out.0.weight`);
  if (!q2Info || !k2Info || !v2Info || !o2Info) {
    return { out: x, executed: false };
  }

  const optF32 = (name: string): Float32Array | null => {
    const info = parsed.tensorMap.get(name);
    return info ? decodeBackingTensorToF32(backing, dataOffset, info) : null;
  };

  // GroupNorm + proj_in (CHW -> [spatial, C] token matrix)
  const normed = new Float32Array(C * spatial);
  const gnW = padOrSliceF32(optF32(`${prefix}.norm.weight`), C, 1);
  const gnB = padOrSliceF32(optF32(`${prefix}.norm.bias`), C, 0);
  verdantGroupNorm(x, gnW, gnB, normed, C, pickGroupCount(C), spatial, 1e-6);

  const projInInfo = parsed.tensorMap.get(`${prefix}.proj_in.weight`);
  const tokens = new Float32Array(spatial * C);
  if (projInInfo && projInInfo.dims.length === 4) {
    const projectedCHW = new Float32Array(C * spatial);
    runStreamedConv2d({
      backing,
      dataOffset,
      wInfo: projInInfo,
      bInfo: parsed.tensorMap.get(`${prefix}.proj_in.bias`) ?? null,
      input: normed,
      out: projectedCHW,
      inC: C,
      outC: C,
      height,
      width,
      kH: 1,
      kW: 1,
      stride: 1,
      padding: 0,
      arena,
    });
    for (let s = 0; s < spatial; s++) {
      for (let c = 0; c < C; c++) tokens[s * C + c] = projectedCHW[c * spatial + s]!;
    }
  } else {
    const rawTokens = new Float32Array(spatial * C);
    for (let s = 0; s < spatial; s++) {
      for (let c = 0; c < C; c++) rawTokens[s * C + c] = normed[c * spatial + s]!;
    }
    if (projInInfo) {
      runStreamedLinearBatch({
        backing,
        dataOffset,
        wInfo: projInInfo,
        bInfo: parsed.tensorMap.get(`${prefix}.proj_in.bias`) ?? null,
        input: rawTokens,
        out: tokens,
        inDim: C,
        outDim: C,
        batchSize: spatial,
        arena,
      });
    } else {
      tokens.set(rawTokens);
    }
  }

  const heads = C >= 64 && C % 8 === 0 ? 8 : Math.max(1, C % 4 === 0 ? 4 : 1);
  const headDim = Math.floor(C / heads);
  const invScale = 1.0 / Math.sqrt(headDim);
  const normTok = new Float32Array(spatial * C);

  const runAttnPass = (
    attnPrefix: string,
    kvSrc: Float32Array,
    kvLen: number,
    kvDim: number,
  ): void => {
    const qInfo = parsed.tensorMap.get(`${attnPrefix}.to_q.weight`);
    const kInfo = parsed.tensorMap.get(`${attnPrefix}.to_k.weight`);
    const vInfo = parsed.tensorMap.get(`${attnPrefix}.to_v.weight`);
    const oInfo = parsed.tensorMap.get(`${attnPrefix}.to_out.0.weight`);
    if (!qInfo || !kInfo || !vInfo || !oInfo) return;

    const Q = new Float32Array(spatial * C);
    const K = new Float32Array(kvLen * C);
    const V = new Float32Array(kvLen * C);
    runStreamedLinearBatch({
      backing,
      dataOffset,
      wInfo: qInfo,
      bInfo: parsed.tensorMap.get(`${attnPrefix}.to_q.bias`) ?? null,
      input: normTok,
      out: Q,
      inDim: C,
      outDim: C,
      batchSize: spatial,
      arena,
    });
    runStreamedLinearBatch({
      backing,
      dataOffset,
      wInfo: kInfo,
      bInfo: parsed.tensorMap.get(`${attnPrefix}.to_k.bias`) ?? null,
      input: kvSrc,
      out: K,
      inDim: kvDim,
      outDim: C,
      batchSize: kvLen,
      arena,
    });
    runStreamedLinearBatch({
      backing,
      dataOffset,
      wInfo: vInfo,
      bInfo: parsed.tensorMap.get(`${attnPrefix}.to_v.bias`) ?? null,
      input: kvSrc,
      out: V,
      inDim: kvDim,
      outDim: C,
      batchSize: kvLen,
      arena,
    });

    const attnOut = new Float32Array(spatial * C);
    const logits = new Float32Array(kvLen);
    for (let s = 0; s < spatial; s++) {
      const qBase = s * C;
      const oBase = s * C;
      for (let h = 0; h < heads; h++) {
        const hOff = h * headDim;
        let maxL = -Infinity;
        for (let t = 0; t < kvLen; t++) {
          let dot = 0;
          const kBase = t * C + hOff;
          for (let d = 0; d < headDim; d++) dot += Q[qBase + hOff + d]! * K[kBase + d]!;
          const sc = dot * invScale;
          logits[t] = sc;
          if (sc > maxL) maxL = sc;
        }
        let sumE = 0;
        for (let t = 0; t < kvLen; t++) {
          const e = Math.exp(logits[t]! - maxL);
          logits[t] = e;
          sumE += e;
        }
        const inv = sumE > 0 ? 1 / sumE : 0;
        for (let t = 0; t < kvLen; t++) {
          const w = logits[t]! * inv;
          const vBase = t * C + hOff;
          for (let d = 0; d < headDim; d++) {
            attnOut[oBase + hOff + d] = Math.fround(attnOut[oBase + hOff + d]! + w * V[vBase + d]!);
          }
        }
      }
    }

    const projected = new Float32Array(spatial * C);
    runStreamedLinearBatch({
      backing,
      dataOffset,
      wInfo: oInfo,
      bInfo: parsed.tensorMap.get(`${attnPrefix}.to_out.0.bias`) ?? null,
      input: attnOut,
      out: projected,
      inDim: C,
      outDim: C,
      batchSize: spatial,
      arena,
    });
    verdantAdd(tokens, projected, tokens, tokens.length);
  };

  // Block 0: Self-Attention (attn1) -> Cross-Attention (attn2) -> GEGLU FeedForward (ff.net)
  const ln1W = padOrSliceF32(optF32(`${tbPrefix}.norm1.weight`), C, 1);
  const ln1B = padOrSliceF32(optF32(`${tbPrefix}.norm1.bias`), C, 0);
  for (let s = 0; s < spatial; s++) {
    verdantLayerNorm(
      tokens.subarray(s * C, (s + 1) * C),
      ln1W,
      ln1B,
      normTok.subarray(s * C, (s + 1) * C),
      C,
      1e-5,
    );
  }
  runAttnPass(`${tbPrefix}.attn1`, normTok, spatial, C);

  const ln2W = padOrSliceF32(optF32(`${tbPrefix}.norm2.weight`), C, 1);
  const ln2B = padOrSliceF32(optF32(`${tbPrefix}.norm2.bias`), C, 0);
  for (let s = 0; s < spatial; s++) {
    verdantLayerNorm(
      tokens.subarray(s * C, (s + 1) * C),
      ln2W,
      ln2B,
      normTok.subarray(s * C, (s + 1) * C),
      C,
      1e-5,
    );
  }
  runAttnPass(`${tbPrefix}.attn2`, condSeq, seqLen, condDim);

  const ff0Info = parsed.tensorMap.get(`${tbPrefix}.ff.net.0.proj.weight`);
  const ff2Info = parsed.tensorMap.get(`${tbPrefix}.ff.net.2.weight`);
  if (ff0Info && ff2Info) {
    const ln3W = padOrSliceF32(optF32(`${tbPrefix}.norm3.weight`), C, 1);
    const ln3B = padOrSliceF32(optF32(`${tbPrefix}.norm3.bias`), C, 0);
    for (let s = 0; s < spatial; s++) {
      verdantLayerNorm(
        tokens.subarray(s * C, (s + 1) * C),
        ln3W,
        ln3B,
        normTok.subarray(s * C, (s + 1) * C),
        C,
        1e-5,
      );
    }
    const ffOut2 = ff0Info.dims[1] ?? C * 8;
    const ffDim = Math.floor(ffOut2 / 2);
    const ffProj = new Float32Array(spatial * ffOut2);
    runStreamedLinearBatch({
      backing,
      dataOffset,
      wInfo: ff0Info,
      bInfo: parsed.tensorMap.get(`${tbPrefix}.ff.net.0.proj.bias`) ?? null,
      input: normTok,
      out: ffProj,
      inDim: C,
      outDim: ffOut2,
      batchSize: spatial,
      arena,
    });
    // GEGLU activation: val * GELU(gate)
    const gated = new Float32Array(spatial * ffDim);
    const gateBuf = new Float32Array(ffDim);
    for (let s = 0; s < spatial; s++) {
      const src = s * ffOut2;
      const dst = s * ffDim;
      for (let d = 0; d < ffDim; d++) gateBuf[d] = ffProj[src + ffDim + d]!;
      verdantGelu(gateBuf, gateBuf, ffDim, false);
      for (let d = 0; d < ffDim; d++) gated[dst + d] = Math.fround(ffProj[src + d]! * gateBuf[d]!);
    }
    const ffOut = new Float32Array(spatial * C);
    runStreamedLinearBatch({
      backing,
      dataOffset,
      wInfo: ff2Info,
      bInfo: parsed.tensorMap.get(`${tbPrefix}.ff.net.2.bias`) ?? null,
      input: gated,
      out: ffOut,
      inDim: ffDim,
      outDim: C,
      batchSize: spatial,
      arena,
    });
    verdantAdd(tokens, ffOut, tokens, tokens.length);
  }

  // proj_out + outer skip x -> CHW
  const projOutInfo = parsed.tensorMap.get(`${prefix}.proj_out.weight`);
  const outCHW = new Float32Array(C * spatial);
  if (projOutInfo && projOutInfo.dims.length === 4) {
    const tmpCHW = new Float32Array(C * spatial);
    for (let s = 0; s < spatial; s++) {
      for (let c = 0; c < C; c++) tmpCHW[c * spatial + s] = tokens[s * C + c]!;
    }
    runStreamedConv2d({
      backing,
      dataOffset,
      wInfo: projOutInfo,
      bInfo: parsed.tensorMap.get(`${prefix}.proj_out.bias`) ?? null,
      input: tmpCHW,
      out: outCHW,
      inC: C,
      outC: C,
      height,
      width,
      kH: 1,
      kW: 1,
      stride: 1,
      padding: 0,
      arena,
    });
  } else {
    const finalTok = new Float32Array(spatial * C);
    if (projOutInfo) {
      runStreamedLinearBatch({
        backing,
        dataOffset,
        wInfo: projOutInfo,
        bInfo: parsed.tensorMap.get(`${prefix}.proj_out.bias`) ?? null,
        input: tokens,
        out: finalTok,
        inDim: C,
        outDim: C,
        batchSize: spatial,
        arena,
      });
    } else {
      finalTok.set(tokens);
    }
    for (let s = 0; s < spatial; s++) {
      for (let c = 0; c < C; c++) outCHW[c * spatial + s] = finalTok[s * C + c]!;
    }
  }
  verdantAdd(outCHW, x, outCHW, outCHW.length);
  return { out: outCHW, executed: true };
}

function runTaesdResidualBlockStreamed(params: {
  readonly x: Float32Array;
  readonly channels: number;
  readonly height: number;
  readonly width: number;
  readonly layerIdx: number;
  readonly backing: ImageGgufBacking;
  readonly parsed: ParsedGgufModel;
  readonly dataOffset: number;
  readonly arena?: VerdantWasmArena | null;
}): Float32Array {
  const { x, channels: C, height, width, layerIdx, backing, parsed, dataOffset, arena } = params;
  const prefix = `first_stage_model.decoder.layers.${layerIdx}`;
  const c0 = parsed.tensorMap.get(`${prefix}.conv.0.weight`);
  const c2 = parsed.tensorMap.get(`${prefix}.conv.2.weight`);
  const c4 = parsed.tensorMap.get(`${prefix}.conv.4.weight`);
  if (!c0 || !c2 || !c4) return x;

  const bufA = new Float32Array(C * height * width);
  const bufB = new Float32Array(C * height * width);
  runStreamedConv2d({
    backing,
    dataOffset,
    wInfo: c0,
    bInfo: parsed.tensorMap.get(`${prefix}.conv.0.bias`) ?? null,
    input: x,
    out: bufA,
    inC: C,
    outC: C,
    height,
    width,
    kH: 3,
    kW: 3,
    stride: 1,
    padding: 1,
    arena,
  });
  for (let i = 0; i < bufA.length; i++) if (bufA[i]! < 0) bufA[i] = 0;

  runStreamedConv2d({
    backing,
    dataOffset,
    wInfo: c2,
    bInfo: parsed.tensorMap.get(`${prefix}.conv.2.bias`) ?? null,
    input: bufA,
    out: bufB,
    inC: C,
    outC: C,
    height,
    width,
    kH: 3,
    kW: 3,
    stride: 1,
    padding: 1,
    arena,
  });
  for (let i = 0; i < bufB.length; i++) if (bufB[i]! < 0) bufB[i] = 0;

  runStreamedConv2d({
    backing,
    dataOffset,
    wInfo: c4,
    bInfo: parsed.tensorMap.get(`${prefix}.conv.4.bias`) ?? null,
    input: bufB,
    out: bufA,
    inC: C,
    outC: C,
    height,
    width,
    kH: 3,
    kW: 3,
    stride: 1,
    padding: 1,
    arena,
  });
  // TAESD Block residual: ReLU(conv4(ReLU(conv2(ReLU(conv0(x))))) + x)
  for (let i = 0; i < bufA.length; i++) {
    const v = bufA[i]! + x[i]!;
    bufA[i] = v > 0 ? Math.fround(v) : 0;
  }
  return bufA;
}

/**
 * Full OPFS layer-streamed 1-step distilled SDXS-512 synthesis (`12-Layer Q8_0 CLIP ->
 * Exact SDXS-512 Skip-Connected Cross-Attention UNet -> Full 19-Layer Residual TAESD`)
 * supporting all 4 `streaming.rs` resolutions (`128 / 256 / 384 / 512px`) with native
 * `F16`/`Q8_0` WASM SIMD and `< 64 MB` peak scratch.
 */
export async function synthesizeDistilledImageAsync(params: {
  readonly promptTokens: readonly number[];
  readonly latentH: number;
  readonly latentW: number;
  readonly seed: number;
  readonly loaded: LoadedNeuralImageWeights;
  readonly scratch?: ImageSynthesisScratch;
  readonly arena?: VerdantWasmArena | null;
  readonly yieldFrame?: () => Promise<void>;
  readonly signal?: AbortSignal;
}): Promise<
  SynthesizedRgbaImage & {
    readonly clipLayersExecuted: number;
    readonly crossAttnBlocksExecuted: number;
    readonly taesdResidualBlocksExecuted: number;
    readonly unetBlocksExecuted: number;
    readonly peakScratchBytes: number;
  }
> {
  const { promptTokens, latentH, latentW, seed, loaded, scratch, arena, yieldFrame, signal } = params;
  const clipRes = await encodeClipSequenceStreamed(promptTokens, loaded, { yieldFrame, signal, arena });
  const zT = seedGaussianLatent(loaded.unet.latentChannels, latentH, latentW, seed);
  let crossAttnBlocksExecuted = 0;
  let unetBlocksExecuted = 0;
  let taesdResidualBlocksExecuted = 0;
  let peakScratchBytes = zT.byteLength;

  const trackPeak = (bytes: number) => {
    if (bytes > peakScratchBytes) peakScratchBytes = bytes;
  };

  let z0: Float32Array;
  const sg = loaded.streamGraph;
  const hasFullSdxsUnet = Boolean(
    sg &&
      sg.inputBlockIndices.length >= 2 &&
      sg.outputBlockIndices.length >= 2 &&
      sg.parsed.tensorMap.has('model.diffusion_model.out.2.weight'),
  );

  if (hasFullSdxsUnet && sg && !signal?.aborted) {
    const { backing, parsed, dataOffset } = sg;
    const optF32 = (name: string): Float32Array | null => {
      const info = parsed.tensorMap.get(name);
      return info ? decodeBackingTensorToF32(backing, dataOffset, info) : null;
    };

    // 1. Exact SDXS-512 timestep embedding (`t = 999` -> time_embed.0 [320->1280] -> SiLU -> time_embed.2 [1280->1280])
    const tSin = sinusoidalTimestepEmbedding(999, loaded.unet.modelChannels);
    const te0Info = parsed.tensorMap.get('model.diffusion_model.time_embed.0.weight');
    const te2Info = parsed.tensorMap.get('model.diffusion_model.time_embed.2.weight');
    let tEmb: Float32Array = tSin;
    if (te0Info && te2Info) {
      const teMidDim = te0Info.dims[1] ?? loaded.unet.modelChannels * 4;
      const teOutDim = te2Info.dims[1] ?? teMidDim;
      const teMid = new Float32Array(teMidDim);
      runStreamedLinearBatch({
        backing,
        dataOffset,
        wInfo: te0Info,
        bInfo: parsed.tensorMap.get('model.diffusion_model.time_embed.0.bias') ?? null,
        input: tSin,
        out: teMid,
        inDim: te0Info.dims[0] ?? tSin.length,
        outDim: teMidDim,
        batchSize: 1,
        arena,
      });
      verdantSiLu(teMid, teMid, teMidDim);
      const teOut = new Float32Array(teOutDim);
      runStreamedLinearBatch({
        backing,
        dataOffset,
        wInfo: te2Info,
        bInfo: parsed.tensorMap.get('model.diffusion_model.time_embed.2.bias') ?? null,
        input: teMid,
        out: teOut,
        inDim: te2Info.dims[0] ?? teMidDim,
        outDim: teOutDim,
        batchSize: 1,
        arena,
      });
      tEmb = teOut;
    }

    // Standard SD v1 scaled_linear Euler 1-step conditioning at t = 999 (sigma ~ 14.6146)
    const sigma999 = 14.6146;
    const invInputScale = 1.0 / Math.sqrt(sigma999 * sigma999 + 1.0);
    let cur: Float32Array = new Float32Array(zT.length);
    for (let i = 0; i < zT.length; i++) cur[i] = Math.fround(zT[i]! * invInputScale);
    let curC = loaded.unet.latentChannels;
    let curH = latentH;
    let curW = latentW;

    const skips: Array<{ readonly data: Float32Array; readonly C: number; readonly H: number; readonly W: number }> = [];

    // 2. Encoder (`input_blocks`: `{0, 1, 3, 4, 6, 7}` on live sdxs-512-tinySDdistilled)
    for (const idx of sg.inputBlockIndices) {
      if (signal?.aborted) break;
      const base = `model.diffusion_model.input_blocks.${idx}`;
      const dsInfo = parsed.tensorMap.get(`${base}.0.op.weight`);
      const resInfo = parsed.tensorMap.get(`${base}.0.in_layers.2.weight`);
      const stemInfo = parsed.tensorMap.get(`${base}.0.weight`);

      if (dsInfo) {
        const outC = dsInfo.dims[3] ?? curC;
        const nextH = Math.floor((curH + 2 - 3) / 2) + 1;
        const nextW = Math.floor((curW + 2 - 3) / 2) + 1;
        const next = new Float32Array(outC * nextH * nextW);
        runStreamedConv2d({
          backing,
          dataOffset,
          wInfo: dsInfo,
          bInfo: parsed.tensorMap.get(`${base}.0.op.bias`) ?? null,
          input: cur,
          out: next,
          inC: curC,
          outC,
          height: curH,
          width: curW,
          kH: 3,
          kW: 3,
          stride: 2,
          padding: 1,
          arena,
        });
        cur = next;
        curC = outC;
        curH = nextH;
        curW = nextW;
        unetBlocksExecuted += 1;
      } else if (resInfo) {
        const res = runResBlockStreamed({
          x: cur,
          inC: curC,
          height: curH,
          width: curW,
          prefix: `${base}.0`,
          tEmb,
          backing,
          parsed,
          dataOffset,
          arena,
        });
        cur = res.out;
        curC = res.outC;
        trackPeak(res.peakBytes);
        unetBlocksExecuted += 1;
      } else if (stemInfo) {
        const outC = stemInfo.dims[3] ?? loaded.unet.modelChannels;
        const next = new Float32Array(outC * curH * curW);
        runStreamedConv2d({
          backing,
          dataOffset,
          wInfo: stemInfo,
          bInfo: parsed.tensorMap.get(`${base}.0.bias`) ?? null,
          input: cur,
          out: next,
          inC: curC,
          outC,
          height: curH,
          width: curW,
          kH: 3,
          kW: 3,
          stride: 1,
          padding: 1,
          arena,
        });
        cur = next;
        curC = outC;
        unetBlocksExecuted += 1;
      }

      if (parsed.tensorMap.has(`${base}.1.transformer_blocks.0.attn2.to_q.weight`)) {
        const attn = runSpatialTransformerStreamed({
          x: cur,
          channels: curC,
          height: curH,
          width: curW,
          prefix: `${base}.1`,
          condSeq: clipRes.sequence,
          seqLen: clipRes.seqLen,
          condDim: loaded.clip.embedDim,
          backing,
          parsed,
          dataOffset,
          arena,
        });
        cur = attn.out;
        if (attn.executed) crossAttnBlocksExecuted += 1;
      }

      skips.push({ data: cur, C: curC, H: curH, W: curW });
      trackPeak(cur.byteLength);
      if (yieldFrame) await yieldFrame();
    }

    // 3. Optional `middle_block` (generic SD1.5 / SD-Turbo / synthetic GGUFs; absent on sdxs-512-tinySDdistilled)
    if (sg.hasMiddleBlock && !signal?.aborted) {
      if (parsed.tensorMap.has('model.diffusion_model.middle_block.0.in_layers.2.weight')) {
        const r0 = runResBlockStreamed({
          x: cur,
          inC: curC,
          height: curH,
          width: curW,
          prefix: 'model.diffusion_model.middle_block.0',
          tEmb,
          backing,
          parsed,
          dataOffset,
          arena,
        });
        cur = r0.out;
        curC = r0.outC;
        trackPeak(r0.peakBytes);
        unetBlocksExecuted += 1;
      }
      if (parsed.tensorMap.has('model.diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_q.weight')) {
        const attn = runSpatialTransformerStreamed({
          x: cur,
          channels: curC,
          height: curH,
          width: curW,
          prefix: 'model.diffusion_model.middle_block.1',
          condSeq: clipRes.sequence,
          seqLen: clipRes.seqLen,
          condDim: loaded.clip.embedDim,
          backing,
          parsed,
          dataOffset,
          arena,
        });
        cur = attn.out;
        if (attn.executed) crossAttnBlocksExecuted += 1;
      }
      if (parsed.tensorMap.has('model.diffusion_model.middle_block.2.in_layers.2.weight')) {
        const r2 = runResBlockStreamed({
          x: cur,
          inC: curC,
          height: curH,
          width: curW,
          prefix: 'model.diffusion_model.middle_block.2',
          tEmb,
          backing,
          parsed,
          dataOffset,
          arena,
        });
        cur = r2.out;
        curC = r2.outC;
        trackPeak(r2.peakBytes);
        unetBlocksExecuted += 1;
      }
      if (yieldFrame) await yieldFrame();
    }

    // 4. Decoder (`output_blocks.0..7` + `up_blocks.0.attentions.{0,1}` with skip concatenations)
    for (const idx of sg.outputBlockIndices) {
      if (signal?.aborted) break;
      const base = `model.diffusion_model.output_blocks.${idx}`;
      const hasRes = parsed.tensorMap.has(`${base}.0.in_layers.2.weight`);

      if (hasRes) {
        const skip = skips.pop();
        if (skip && skip.H === curH && skip.W === curW) {
          cur = concatChannelsCHW(cur, curC, skip.data, skip.C, curH * curW);
          curC += skip.C;
          trackPeak(cur.byteLength);
        }
        const res = runResBlockStreamed({
          x: cur,
          inC: curC,
          height: curH,
          width: curW,
          prefix: `${base}.0`,
          tEmb,
          backing,
          parsed,
          dataOffset,
          arena,
        });
        cur = res.out;
        curC = res.outC;
        trackPeak(res.peakBytes);
        unetBlocksExecuted += 1;
      }

      // SpatialTransformer (either `output_blocks.${idx}.1` or SDXS `up_blocks.0.attentions.${idx}`)
      const directAttnPrefix = `${base}.1`;
      const sdxsUpAttnPrefix = `model.diffusion_model.up_blocks.0.attentions.${idx}`;
      const attnPrefix = parsed.tensorMap.has(`${directAttnPrefix}.transformer_blocks.0.attn2.to_q.weight`)
        ? directAttnPrefix
        : parsed.tensorMap.has(`${sdxsUpAttnPrefix}.transformer_blocks.0.attn2.to_q.weight`)
          ? sdxsUpAttnPrefix
          : null;
      if (attnPrefix) {
        const attn = runSpatialTransformerStreamed({
          x: cur,
          channels: curC,
          height: curH,
          width: curW,
          prefix: attnPrefix,
          condSeq: clipRes.sequence,
          seqLen: clipRes.seqLen,
          condDim: loaded.clip.embedDim,
          backing,
          parsed,
          dataOffset,
          arena,
        });
        cur = attn.out;
        if (attn.executed) crossAttnBlocksExecuted += 1;
      }

      // Optional 2x upsample conv (`${base}.1.conv.weight` or `${base}.2.conv.weight`)
      const upInfo =
        parsed.tensorMap.get(`${base}.1.conv.weight`) ??
        parsed.tensorMap.get(`${base}.2.conv.weight`);
      const upBiasInfo =
        parsed.tensorMap.get(`${base}.1.conv.bias`) ??
        parsed.tensorMap.get(`${base}.2.conv.bias`) ??
        null;
      if (upInfo) {
        const up = upsampleNearest2x(cur, curC, curH, curW);
        curH *= 2;
        curW *= 2;
        const outC = upInfo.dims[3] ?? curC;
        const next = new Float32Array(outC * curH * curW);
        runStreamedConv2d({
          backing,
          dataOffset,
          wInfo: upInfo,
          bInfo: upBiasInfo,
          input: up,
          out: next,
          inC: curC,
          outC,
          height: curH,
          width: curW,
          kH: 3,
          kW: 3,
          stride: 1,
          padding: 1,
          arena,
        });
        cur = next;
        curC = outC;
        trackPeak(cur.byteLength);
        unetBlocksExecuted += 1;
      }

      if (yieldFrame) await yieldFrame();
    }

    // 5. Final `out.0` GroupNorm + SiLU + `out.2` Conv2d (`320 -> 4`) & 1-step Euler update
    const outNorm = new Float32Array(curC * curH * curW);
    const outGnW = padOrSliceF32(optF32('model.diffusion_model.out.0.weight'), curC, 1);
    const outGnB = padOrSliceF32(optF32('model.diffusion_model.out.0.bias'), curC, 0);
    verdantGroupNorm(cur, outGnW, outGnB, outNorm, curC, pickGroupCount(curC), curH * curW, 1e-5);
    verdantSiLu(outNorm, outNorm, outNorm.length);

    const epsPred = new Float32Array(loaded.unet.latentChannels * curH * curW);
    runStreamedConv2d({
      backing,
      dataOffset,
      wInfo: parsed.tensorMap.get('model.diffusion_model.out.2.weight')!,
      bInfo: parsed.tensorMap.get('model.diffusion_model.out.2.bias') ?? null,
      input: outNorm,
      out: epsPred,
      inC: curC,
      outC: loaded.unet.latentChannels,
      height: curH,
      width: curW,
      kH: 3,
      kW: 3,
      stride: 1,
      padding: 1,
      arena,
    });
    z0 = new Float32Array(zT.length);
    for (let i = 0; i < z0.length; i++) {
      z0[i] = Math.fround(zT[i]! - sigma999 * epsPred[i]!);
    }
  } else {
    // Minimal synthetic GGUF fallback (supports optional single-stage cross-attention)
    const tEmb = sinusoidalTimestepEmbedding(999, loaded.clip.embedDim);
    const blendedCond = new Float32Array(loaded.clip.embedDim);
    for (let i = 0; i < blendedCond.length; i++) {
      blendedCond[i] = Math.fround(clipRes.pooled[i]! + 0.1 * (tEmb[i] ?? 0));
    }
    z0 = denoiseOneStepDistilledUnet(zT, latentH, latentW, blendedCond, loaded.unet, 1.0, scratch);
    if (sg && !signal?.aborted) {
      const { backing, parsed, dataOffset } = sg;
      const spatial = latentH * latentW;
      const C = loaded.unet.modelChannels;
      const attnPrefixes: string[] = [];
      for (const t of parsed.tensors) {
        if (t.name.endsWith('.attn2.to_q.weight')) {
          attnPrefixes.push(t.name.slice(0, -'.attn2.to_q.weight'.length));
        }
      }
      if (attnPrefixes.length > 0 && spatial <= 4096) {
        const hIn = new Float32Array(C * spatial);
        const hOut = new Float32Array(C * spatial);
        verdantConv2d(z0, loaded.unet.convInW, loaded.unet.convInB, hIn, loaded.unet.latentChannels, C, latentH, latentW, 3, 3, 1, 1);
        for (const prefix of attnPrefixes) {
          if (signal?.aborted) break;
          const qInfo = parsed.tensorMap.get(`${prefix}.attn2.to_q.weight`);
          const kInfo = parsed.tensorMap.get(`${prefix}.attn2.to_k.weight`);
          const vInfo = parsed.tensorMap.get(`${prefix}.attn2.to_v.weight`);
          const oInfo = parsed.tensorMap.get(`${prefix}.attn2.to_out.0.weight`);
          if (!qInfo || !kInfo || !vInfo || !oInfo) continue;
          const blockC = qInfo.dims[0] ?? C;
          const blockCond = kInfo.dims[0] ?? loaded.clip.embedDim;
          if (blockC !== C || blockCond !== loaded.clip.embedDim) continue;
          applySpatialCrossAttention({
            x: hIn,
            channels: C,
            spatial,
            condSeq: clipRes.sequence,
            seqLen: clipRes.seqLen,
            condDim: loaded.clip.embedDim,
            numHeads: C >= 64 && C % 8 === 0 ? 8 : Math.max(1, C % 4 === 0 ? 4 : 1),
            toQW: decodeBackingTensorToF32(backing, dataOffset, qInfo),
            toKW: decodeBackingTensorToF32(backing, dataOffset, kInfo),
            toVW: decodeBackingTensorToF32(backing, dataOffset, vInfo),
            toOutW: decodeBackingTensorToF32(backing, dataOffset, oInfo),
            toOutB: parsed.tensorMap.has(`${prefix}.attn2.to_out.0.bias`)
              ? decodeBackingTensorToF32(backing, dataOffset, parsed.tensorMap.get(`${prefix}.attn2.to_out.0.bias`)!)
              : null,
            out: hOut,
          });
          hIn.set(hOut);
          crossAttnBlocksExecuted += 1;
          if (yieldFrame) await yieldFrame();
        }
        if (crossAttnBlocksExecuted > 0) {
          const refinedEps = new Float32Array(z0.length);
          verdantConv2d(hIn, loaded.unet.convOutW, loaded.unet.convOutB, refinedEps, C, loaded.unet.latentChannels, latentH, latentW, 3, 3, 1, 1);
          const nextZ0 = new Float32Array(z0.length);
          for (let i = 0; i < z0.length; i++) {
            nextZ0[i] = Math.fround(0.5 * z0[i]! + 0.5 * (zT[i]! - refinedEps[i]!));
          }
          z0 = nextZ0;
        }
      }
    }
  }

  if (yieldFrame) await yieldFrame();

  // 6. Exact 19-Layer Residual TAESD Decoder (with Clamp + all 10 residual blocks + native [0, 1] RGB!)
  let cur = z0;
  let curH = latentH;
  let curW = latentW;
  let curC = loaded.unet.latentChannels;
  const hasRealTaesdResiduals = Boolean(sg && sg.taesdResidualIndices.length > 0);

  if (hasRealTaesdResiduals && sg) {
    const { backing, parsed, dataOffset } = sg;
    // TAESD Decoder entry Clamp: 3 * tanh(z / 3)
    const clamped = new Float32Array(cur.length);
    for (let i = 0; i < cur.length; i++) {
      clamped[i] = Math.fround(3.0 * Math.tanh(cur[i]! / 3.0));
    }
    cur = clamped;

    const runLayerConv = (layerIdx: number, applyRelu: boolean, upsampleFirst: boolean) => {
      const wInfo = parsed.tensorMap.get(`first_stage_model.decoder.layers.${layerIdx}.weight`);
      if (!wInfo) return;
      if (upsampleFirst) {
        cur = upsampleNearest2x(cur, curC, curH, curW);
        curH *= 2;
        curW *= 2;
        trackPeak(cur.byteLength);
      }
      const outC = wInfo.dims[3] ?? curC;
      const next = new Float32Array(outC * curH * curW);
      trackPeak(next.byteLength);
      runStreamedConv2d({
        backing,
        dataOffset,
        wInfo,
        bInfo: parsed.tensorMap.get(`first_stage_model.decoder.layers.${layerIdx}.bias`) ?? null,
        input: cur,
        out: next,
        inC: curC,
        outC,
        height: curH,
        width: curW,
        kH: 3,
        kW: 3,
        stride: 1,
        padding: 1,
        arena,
      });
      if (applyRelu) {
        for (let i = 0; i < next.length; i++) if (next[i]! < 0) next[i] = 0;
      }
      cur = next;
      curC = outC;
    };

    const runResGroup = async (indices: readonly number[]) => {
      for (const idx of indices) {
        if (signal?.aborted) break;
        if (parsed.tensorMap.has(`first_stage_model.decoder.layers.${idx}.conv.0.weight`)) {
          cur = runTaesdResidualBlockStreamed({
            x: cur,
            channels: curC,
            height: curH,
            width: curW,
            layerIdx: idx,
            backing,
            parsed,
            dataOffset,
            arena,
          });
          taesdResidualBlocksExecuted += 1;
          trackPeak(cur.byteLength);
          if (yieldFrame) await yieldFrame();
        }
      }
    };

    runLayerConv(0, true, false);
    await runResGroup([2, 3, 4]);
    runLayerConv(6, false, true);
    await runResGroup([7, 8, 9]);
    runLayerConv(11, false, true);
    await runResGroup([12, 13, 14]);
    runLayerConv(16, false, true);
    await runResGroup([17]);
    runLayerConv(18, false, false);

    // TAESD outputs RGB natively in [0, 1]
    const plane = curH * curW;
    const rgba = new Uint8ClampedArray(plane * 4);
    for (let i = 0; i < plane; i++) {
      rgba[i * 4] = Math.round(Math.max(0, Math.min(1, cur[i]!)) * 255);
      rgba[i * 4 + 1] = Math.round(Math.max(0, Math.min(1, cur[plane + i]!)) * 255);
      rgba[i * 4 + 2] = Math.round(Math.max(0, Math.min(1, cur[2 * plane + i]!)) * 255);
      rgba[i * 4 + 3] = 255;
    }
    return {
      width: curW,
      height: curH,
      rgba,
      seed,
      clipLayersExecuted: clipRes.layersExecuted,
      crossAttnBlocksExecuted,
      taesdResidualBlocksExecuted,
      unetBlocksExecuted,
      peakScratchBytes,
    };
  }

  // Legacy minimal TAESD fallback (synthetic unit-test GGUFs without residual blocks)
  for (const stage of loaded.taesd.stages) {
    if (stage.upsample2x) {
      cur = upsampleNearest2x(cur, curC, curH, curW);
      curH *= 2;
      curW *= 2;
      trackPeak(cur.byteLength);
    }
    const next = new Float32Array(stage.outC * curH * curW);
    trackPeak(next.byteLength);
    verdantConv2d(cur, stage.convW, stage.convB, next, stage.inC, stage.outC, curH, curW, 3, 3, 1, 1);
    for (let i = 0; i < next.length; i++) {
      if (next[i]! < 0) next[i] = 0;
    }
    cur = next;
    curC = stage.outC;
    if (yieldFrame) await yieldFrame();
  }

  const rgb = new Float32Array(3 * curH * curW);
  trackPeak(rgb.byteLength);
  verdantConv2d(cur, loaded.taesd.finalW, loaded.taesd.finalB, rgb, curC, 3, curH, curW, 3, 3, 1, 1);

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

  return {
    width: curW,
    height: curH,
    rgba,
    seed,
    clipLayersExecuted: clipRes.layersExecuted,
    crossAttnBlocksExecuted,
    taesdResidualBlocksExecuted,
    unetBlocksExecuted,
    peakScratchBytes,
  };
}
