/**
 * Real Whisper Audio Encoder-Decoder Engine (`src/engine/whisperEngine.ts`)
 *
 * Executes real Whisper audio perception graphs (`encoder.conv1` -> `encoder.conv2` ->
 * Transformer encoder blocks -> autoregressive Transformer decoder with cached cross-attention)
 * strictly on VerdantUI's shared `vTensor` kernels (`verdantConv1d`, `verdantLayerNorm`,
 * `verdantGelu`, `verdantMulMat`).
 */

import type { MelSpectrogram } from './audioFrontend';
import { GgmlType, ggmlTensorByteSize } from './gguf';
import {
  verdantAdd,
  verdantConv1d,
  verdantDecodeRow,
  verdantGelu,
  verdantLayerNorm,
} from './verdantKernels';

export const WHISPER_GGML_MAGIC = 0x67676d6c;

export interface WhisperConfig {
  readonly nMels: number;
  readonly nAudioCtx: number;
  readonly nAudioState: number;
  readonly nAudioHead: number;
  readonly nAudioLayer: number;
  readonly nTextCtx: number;
  readonly nTextState: number;
  readonly nTextHead: number;
  readonly nTextLayer: number;
  readonly nVocab: number;
  readonly sotTokenId: number;
  readonly eotTokenId: number;
}

export interface DenseLinear {
  readonly weight: Float32Array; // [outDim, inDim] row-major
  readonly bias: Float32Array | null;
  readonly inDim: number;
  readonly outDim: number;
}

export interface WhisperEncoderLayer {
  readonly attnLnW: Float32Array;
  readonly attnLnB: Float32Array;
  readonly q: DenseLinear;
  readonly k: DenseLinear;
  readonly v: DenseLinear;
  readonly o: DenseLinear;
  readonly mlpLnW: Float32Array;
  readonly mlpLnB: Float32Array;
  readonly fc1: DenseLinear;
  readonly fc2: DenseLinear;
}

export interface WhisperDecoderLayer {
  readonly selfLnW: Float32Array;
  readonly selfLnB: Float32Array;
  readonly selfQ: DenseLinear;
  readonly selfK: DenseLinear;
  readonly selfV: DenseLinear;
  readonly selfO: DenseLinear;
  readonly crossLnW: Float32Array;
  readonly crossLnB: Float32Array;
  readonly crossQ: DenseLinear;
  readonly crossK: DenseLinear;
  readonly crossV: DenseLinear;
  readonly crossO: DenseLinear;
  readonly mlpLnW: Float32Array;
  readonly mlpLnB: Float32Array;
  readonly fc1: DenseLinear;
  readonly fc2: DenseLinear;
}

export interface WhisperWeights {
  readonly config: WhisperConfig;
  readonly conv1W: Float32Array; // [nAudioState, nMels, 3]
  readonly conv1B: Float32Array;
  readonly conv2W: Float32Array; // [nAudioState, nAudioState, 3]
  readonly conv2B: Float32Array;
  readonly encPosEmbd: Float32Array; // [nAudioCtx, nAudioState]
  readonly encoderLayers: readonly WhisperEncoderLayer[];
  readonly encLnPostW: Float32Array;
  readonly encLnPostB: Float32Array;
  readonly tokEmbd: Float32Array; // [nVocab, nTextState]
  readonly decPosEmbd: Float32Array; // [nTextCtx, nTextState]
  readonly decoderLayers: readonly WhisperDecoderLayer[];
  readonly decLnW: Float32Array;
  readonly decLnB: Float32Array;
  readonly vocab: readonly string[];
}

interface ParsedGgmlTensor {
  readonly name: string;
  readonly dims: readonly number[];
  readonly type: GgmlType;
  readonly offset: number;
  readonly byteSize: number;
}

function decodeGgmlTensorToF32(bytes: Uint8Array, t: ParsedGgmlTensor): Float32Array {
  const total = t.dims.reduce((a, b) => a * b, 1);
  const out = new Float32Array(total);
  const ne0 = t.dims[0] ?? total;
  const rows = Math.max(1, Math.floor(total / ne0));
  const rowBytes = Math.floor(t.byteSize / rows);
  const rowBuf = new Float32Array(ne0);
  for (let r = 0; r < rows; r++) {
    verdantDecodeRow(bytes, t.offset + r * rowBytes, t.type, ne0, rowBuf);
    out.set(rowBuf, r * ne0);
  }
  return out;
}

/**
 * Parses an official `whisper.cpp` GGML binary model (`0x67676d6c`, e.g. `ggml-tiny.en-q8_0.bin`)
 * into `WhisperWeights`, decoding F32/F16/Q4_0/Q5_0/Q8_0 tensors via shared `vTensor` kernels.
 */
export function loadWhisperWeightsFromGgml(bytes: Uint8Array): WhisperWeights {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let off = 0;
  const readI32 = () => {
    const v = dv.getInt32(off, true);
    off += 4;
    return v;
  };
  const readU32 = () => {
    const v = dv.getUint32(off, true);
    off += 4;
    return v;
  };

  const magic = readU32();
  if (magic !== WHISPER_GGML_MAGIC) {
    throw new Error(`Invalid Whisper GGML magic: 0x${magic.toString(16)} (expected 0x67676d6c)`);
  }

  const nVocab = readI32();
  const nAudioCtx = readI32();
  const nAudioState = readI32();
  const nAudioHead = readI32();
  const nAudioLayer = readI32();
  const nTextCtx = readI32();
  const nTextState = readI32();
  const nTextHead = readI32();
  const nTextLayer = readI32();
  const nMels = readI32();
  /* ftype */ readI32();

  // Skip mel filterbank matrix [filtersMel, filtersFft] float32
  const filtersMel = readI32();
  const filtersFft = readI32();
  off += Math.max(0, filtersMel * filtersFft) * 4;

  // Read BPE vocabulary tokens
  const vocabCount = readI32();
  const decoder = new TextDecoder('utf-8');
  const vocab: string[] = [];
  for (let i = 0; i < vocabCount; i++) {
    const len = readI32();
    const tok = decoder.decode(bytes.subarray(off, off + len));
    off += len;
    vocab.push(tok);
  }
  while (vocab.length < nVocab) vocab.push('');

  // Index all GGML tensors
  const tensors = new Map<string, ParsedGgmlTensor>();
  while (off + 12 <= bytes.byteLength) {
    const nDims = readI32();
    const nameLen = readI32();
    const ttype = readI32() as GgmlType;
    if (nDims < 1 || nDims > 4 || nameLen <= 0 || nameLen > 256) {
      throw new Error(`Malformed Whisper tensor header at offset ${off - 12}`);
    }
    const dims: number[] = [];
    for (let d = 0; d < nDims; d++) dims.push(readI32());
    const name = decoder.decode(bytes.subarray(off, off + nameLen));
    off += nameLen;
    const byteSize = ggmlTensorByteSize(ttype, dims);
    if (off + byteSize > bytes.byteLength) {
      throw new Error(`Truncated Whisper tensor "${name}" (${byteSize} bytes needed)`);
    }
    tensors.set(name, { name, dims, type: ttype, offset: off, byteSize });
    off += byteSize;
  }

  const reqF32 = (name: string): Float32Array => {
    const t = tensors.get(name);
    if (!t) throw new Error(`Whisper GGML model missing required tensor "${name}"`);
    return decodeGgmlTensorToF32(bytes, t);
  };
  const optF32 = (name: string): Float32Array | null => {
    const t = tensors.get(name);
    return t ? decodeGgmlTensorToF32(bytes, t) : null;
  };
  const reqLinear = (weightName: string, biasName?: string): DenseLinear => {
    const t = tensors.get(weightName);
    if (!t) throw new Error(`Whisper GGML model missing required linear weight "${weightName}"`);
    const weight = decodeGgmlTensorToF32(bytes, t);
    const inDim = t.dims[0] ?? 1;
    const outDim = t.dims[1] ?? Math.floor(weight.length / inDim);
    const bias = biasName ? optF32(biasName) : null;
    return { weight, bias, inDim, outDim };
  };

  const encoderLayers: WhisperEncoderLayer[] = [];
  for (let i = 0; i < nAudioLayer; i++) {
    const p = `encoder.blocks.${i}`;
    encoderLayers.push({
      attnLnW: reqF32(`${p}.attn_ln.weight`),
      attnLnB: reqF32(`${p}.attn_ln.bias`),
      q: reqLinear(`${p}.attn.query.weight`, `${p}.attn.query.bias`),
      k: reqLinear(`${p}.attn.key.weight`),
      v: reqLinear(`${p}.attn.value.weight`, `${p}.attn.value.bias`),
      o: reqLinear(`${p}.attn.out.weight`, `${p}.attn.out.bias`),
      mlpLnW: reqF32(`${p}.mlp_ln.weight`),
      mlpLnB: reqF32(`${p}.mlp_ln.bias`),
      fc1: reqLinear(`${p}.mlp.0.weight`, `${p}.mlp.0.bias`),
      fc2: reqLinear(`${p}.mlp.2.weight`, `${p}.mlp.2.bias`),
    });
  }

  const decoderLayers: WhisperDecoderLayer[] = [];
  for (let i = 0; i < nTextLayer; i++) {
    const p = `decoder.blocks.${i}`;
    decoderLayers.push({
      selfLnW: reqF32(`${p}.attn_ln.weight`),
      selfLnB: reqF32(`${p}.attn_ln.bias`),
      selfQ: reqLinear(`${p}.attn.query.weight`, `${p}.attn.query.bias`),
      selfK: reqLinear(`${p}.attn.key.weight`),
      selfV: reqLinear(`${p}.attn.value.weight`, `${p}.attn.value.bias`),
      selfO: reqLinear(`${p}.attn.out.weight`, `${p}.attn.out.bias`),
      crossLnW: reqF32(`${p}.cross_attn_ln.weight`),
      crossLnB: reqF32(`${p}.cross_attn_ln.bias`),
      crossQ: reqLinear(`${p}.cross_attn.query.weight`, `${p}.cross_attn.query.bias`),
      crossK: reqLinear(`${p}.cross_attn.key.weight`),
      crossV: reqLinear(`${p}.cross_attn.value.weight`, `${p}.cross_attn.value.bias`),
      crossO: reqLinear(`${p}.cross_attn.out.weight`, `${p}.cross_attn.out.bias`),
      mlpLnW: reqF32(`${p}.mlp_ln.weight`),
      mlpLnB: reqF32(`${p}.mlp_ln.bias`),
      fc1: reqLinear(`${p}.mlp.0.weight`, `${p}.mlp.0.bias`),
      fc2: reqLinear(`${p}.mlp.2.weight`, `${p}.mlp.2.bias`),
    });
  }

  const isMultilingual = nVocab >= 51865;
  const isEnglishFull = nVocab === 51864;
  const sotIdx = vocab.indexOf('<sot>');
  const eotIdx = vocab.indexOf('<eot>');
  const eotTokenId = isMultilingual ? 50257 : isEnglishFull ? 50256 : eotIdx >= 0 ? eotIdx : 1;
  const sotTokenId = isMultilingual ? 50258 : isEnglishFull ? 50257 : sotIdx >= 0 ? sotIdx : 0;

  return {
    config: {
      nMels,
      nAudioCtx,
      nAudioState,
      nAudioHead,
      nAudioLayer,
      nTextCtx,
      nTextState,
      nTextHead,
      nTextLayer,
      nVocab,
      sotTokenId,
      eotTokenId,
    },
    conv1W: reqF32('encoder.conv1.weight'),
    conv1B: reqF32('encoder.conv1.bias'),
    conv2W: reqF32('encoder.conv2.weight'),
    conv2B: reqF32('encoder.conv2.bias'),
    encPosEmbd: reqF32('encoder.positional_embedding'),
    encoderLayers,
    encLnPostW: reqF32('encoder.ln_post.weight'),
    encLnPostB: reqF32('encoder.ln_post.bias'),
    tokEmbd: reqF32('decoder.token_embedding'),
    decPosEmbd: reqF32('decoder.positional_embedding'),
    decoderLayers,
    decLnW: reqF32('decoder.ln.weight'),
    decLnB: reqF32('decoder.ln.bias'),
    vocab,
  };
}

function denseMatVec(lin: DenseLinear, x: Float32Array, out: Float32Array): void {
  const { weight, bias, inDim, outDim } = lin;
  for (let r = 0; r < outDim; r++) {
    let acc = bias ? bias[r]! : 0.0;
    const base = r * inDim;
    for (let c = 0; c < inDim; c++) {
      acc += weight[base + c]! * x[c]!;
    }
    out[r] = Math.fround(acc);
  }
}

function multiHeadAttention(
  Q: Float32Array,
  K: Float32Array,
  V: Float32Array,
  qLen: number,
  kvLen: number,
  dim: number,
  nHeads: number,
  causal: boolean,
  out: Float32Array,
): void {
  const headDim = Math.floor(dim / nHeads);
  const scale = 1.0 / Math.sqrt(headDim);
  const scores = new Float32Array(kvLen);
  out.fill(0);

  for (let h = 0; h < nHeads; h++) {
    const hOff = h * headDim;
    for (let qi = 0; qi < qLen; qi++) {
      const qBase = qi * dim + hOff;
      const limit = causal ? Math.min(kvLen, qi + 1) : kvLen;
      let maxS = -Infinity;
      for (let ki = 0; ki < limit; ki++) {
        const kBase = ki * dim + hOff;
        let dot = 0.0;
        for (let d = 0; d < headDim; d++) dot += Q[qBase + d]! * K[kBase + d]!;
        const s = dot * scale;
        scores[ki] = s;
        if (s > maxS) maxS = s;
      }
      let denom = 0.0;
      for (let ki = 0; ki < limit; ki++) {
        const e = Math.exp(scores[ki]! - maxS);
        scores[ki] = e;
        denom += e;
      }
      const invD = denom > 0 ? 1.0 / denom : 0;
      const oBase = qi * dim + hOff;
      for (let ki = 0; ki < limit; ki++) {
        const w = scores[ki]! * invD;
        const vBase = ki * dim + hOff;
        for (let d = 0; d < headDim; d++) {
          out[oBase + d] = Math.fround(out[oBase + d]! + w * V[vBase + d]!);
        }
      }
    }
  }
}

/**
 * Real Whisper Audio Encoder-Decoder running on shared `vTensor` kernels.
 */
export class WhisperAudioEngine {
  readonly config: WhisperConfig;

  constructor(private readonly w: WhisperWeights) {
    this.config = w.config;
  }

  /**
   * Runs the full Whisper Conv1D + Transformer Encoder over an 80-bin Log-Mel Spectrogram,
   * returning acoustic encoder states `[nEncFrames, nAudioState]`.
   */
  encodeMel(mel: MelSpectrogram): { readonly states: Float32Array; readonly frames: number } {
    const D = this.config.nAudioState;
    // 1. Conv1D stem: [nMels, nFrames] -> [D, nFrames] (stride 1, pad 1) + GELU
    const c1Out = new Float32Array(D * mel.nFrames);
    const len1 = verdantConv1d(
      mel.data,
      this.w.conv1W,
      this.w.conv1B,
      c1Out,
      this.config.nMels,
      D,
      mel.nFrames,
      3,
      1,
      1,
    );
    verdantGelu(c1Out, c1Out, D * len1);

    // 2. Conv1D stride-2 downsample: [D, len1] -> [D, encFrames] + GELU
    const encFrames = Math.min(this.config.nAudioCtx, Math.floor((len1 + 2 - 3) / 2) + 1);
    const c2Out = new Float32Array(D * Math.max(1, Math.floor((len1 + 2 - 3) / 2) + 1));
    const len2 = verdantConv1d(c1Out, this.w.conv2W, this.w.conv2B, c2Out, D, D, len1, 3, 2, 1);
    const frames = Math.min(encFrames, len2);
    verdantGelu(c2Out, c2Out, D * frames);

    // Transpose [D, frames] -> [frames, D] + positional embedding
    const x = new Float32Array(frames * D);
    for (let t = 0; t < frames; t++) {
      const base = t * D;
      for (let d = 0; d < D; d++) {
        x[base + d] = Math.fround(c2Out[d * len2 + t]! + this.w.encPosEmbd[base + d]!);
      }
    }

    const norm = new Float32Array(D);
    const Q = new Float32Array(frames * D);
    const K = new Float32Array(frames * D);
    const V = new Float32Array(frames * D);
    const attnOut = new Float32Array(frames * D);
    const proj = new Float32Array(D);

    for (const layer of this.w.encoderLayers) {
      for (let t = 0; t < frames; t++) {
        verdantLayerNorm(x, layer.attnLnW, layer.attnLnB, norm, D, 1e-5, t * D, 0);
        denseMatVec(layer.q, norm, Q.subarray(t * D, (t + 1) * D));
        denseMatVec(layer.k, norm, K.subarray(t * D, (t + 1) * D));
        denseMatVec(layer.v, norm, V.subarray(t * D, (t + 1) * D));
      }
      multiHeadAttention(Q, K, V, frames, frames, D, this.config.nAudioHead, false, attnOut);

      const mlpHid = new Float32Array(layer.fc1.outDim);
      for (let t = 0; t < frames; t++) {
        const base = t * D;
        const xSlice = x.subarray(base, base + D);
        denseMatVec(layer.o, attnOut.subarray(base, base + D), proj);
        verdantAdd(xSlice, proj, xSlice, D);

        verdantLayerNorm(x, layer.mlpLnW, layer.mlpLnB, norm, D, 1e-5, base, 0);
        denseMatVec(layer.fc1, norm, mlpHid);
        verdantGelu(mlpHid, mlpHid, mlpHid.length);
        denseMatVec(layer.fc2, mlpHid, proj);
        verdantAdd(xSlice, proj, xSlice, D);
      }
    }

    for (let t = 0; t < frames; t++) {
      verdantLayerNorm(x, this.w.encLnPostW, this.w.encLnPostB, x, D, 1e-5, t * D, t * D);
    }
    return { states: x, frames };
  }

  /**
   * Runs the full Whisper Encoder + autoregressive Cross-Attention Decoder over `mel`,
   * returning the decoded token IDs and transcript string.
   */
  transcribe(mel: MelSpectrogram, maxTokens = 64): {
    readonly tokenIds: readonly number[];
    readonly text: string;
    readonly encoderFrames: number;
  } {
    const { states: encStates, frames: encFrames } = this.encodeMel(mel);
    const D = this.config.nTextState;
    const tokens: number[] = [this.config.sotTokenId];

    // Precompute cross-attention K/V once per decoder layer (eliminating per-token redundant projection!)
    const crossK = this.w.decoderLayers.map((layer) => {
      const buf = new Float32Array(encFrames * D);
      for (let t = 0; t < encFrames; t++) {
        denseMatVec(layer.crossK, encStates.subarray(t * D, (t + 1) * D), buf.subarray(t * D, (t + 1) * D));
      }
      return buf;
    });
    const crossV = this.w.decoderLayers.map((layer) => {
      const buf = new Float32Array(encFrames * D);
      for (let t = 0; t < encFrames; t++) {
        denseMatVec(layer.crossV, encStates.subarray(t * D, (t + 1) * D), buf.subarray(t * D, (t + 1) * D));
      }
      return buf;
    });

    const limit = Math.min(maxTokens, this.config.nTextCtx - 1);
    const norm = new Float32Array(D);
    const proj = new Float32Array(D);

    for (let step = 0; step < limit; step++) {
      const seqLen = tokens.length;
      const x = new Float32Array(seqLen * D);
      for (let pos = 0; pos < seqLen; pos++) {
        const tid = tokens[pos]!;
        const tBase = tid * D;
        const pBase = pos * D;
        for (let d = 0; d < D; d++) {
          x[pBase + d] = Math.fround(this.w.tokEmbd[tBase + d]! + this.w.decPosEmbd[pBase + d]!);
        }
      }

      const selfQ = new Float32Array(seqLen * D);
      const selfK = new Float32Array(seqLen * D);
      const selfV = new Float32Array(seqLen * D);
      const attnOut = new Float32Array(seqLen * D);
      const cQ = new Float32Array(seqLen * D);

      for (let li = 0; li < this.w.decoderLayers.length; li++) {
        const layer = this.w.decoderLayers[li]!;
        for (let pos = 0; pos < seqLen; pos++) {
          verdantLayerNorm(x, layer.selfLnW, layer.selfLnB, norm, D, 1e-5, pos * D, 0);
          denseMatVec(layer.selfQ, norm, selfQ.subarray(pos * D, (pos + 1) * D));
          denseMatVec(layer.selfK, norm, selfK.subarray(pos * D, (pos + 1) * D));
          denseMatVec(layer.selfV, norm, selfV.subarray(pos * D, (pos + 1) * D));
        }
        multiHeadAttention(selfQ, selfK, selfV, seqLen, seqLen, D, this.config.nTextHead, true, attnOut);
        for (let pos = 0; pos < seqLen; pos++) {
          const base = pos * D;
          denseMatVec(layer.selfO, attnOut.subarray(base, base + D), proj);
          verdantAdd(x.subarray(base, base + D), proj, x.subarray(base, base + D), D);
        }

        // Cross-attention over cached encoder states
        for (let pos = 0; pos < seqLen; pos++) {
          verdantLayerNorm(x, layer.crossLnW, layer.crossLnB, norm, D, 1e-5, pos * D, 0);
          denseMatVec(layer.crossQ, norm, cQ.subarray(pos * D, (pos + 1) * D));
        }
        multiHeadAttention(
          cQ,
          crossK[li]!,
          crossV[li]!,
          seqLen,
          encFrames,
          D,
          this.config.nTextHead,
          false,
          attnOut,
        );
        const mlpHid = new Float32Array(layer.fc1.outDim);
        for (let pos = 0; pos < seqLen; pos++) {
          const base = pos * D;
          const xSlice = x.subarray(base, base + D);
          denseMatVec(layer.crossO, attnOut.subarray(base, base + D), proj);
          verdantAdd(xSlice, proj, xSlice, D);

          verdantLayerNorm(x, layer.mlpLnW, layer.mlpLnB, norm, D, 1e-5, base, 0);
          denseMatVec(layer.fc1, norm, mlpHid);
          verdantGelu(mlpHid, mlpHid, mlpHid.length);
          denseMatVec(layer.fc2, mlpHid, proj);
          verdantAdd(xSlice, proj, xSlice, D);
        }
      }

      // Final LN + tied output embedding projection on last position
      const lastBase = (seqLen - 1) * D;
      verdantLayerNorm(x, this.w.decLnW, this.w.decLnB, norm, D, 1e-5, lastBase, 0);
      let bestId = this.config.eotTokenId;
      let bestLogit = -Infinity;
      for (let v = 0; v < this.config.nVocab; v++) {
        let dot = 0.0;
        const vBase = v * D;
        for (let d = 0; d < D; d++) dot += norm[d]! * this.w.tokEmbd[vBase + d]!;
        if (dot > bestLogit) {
          bestLogit = dot;
          bestId = v;
        }
      }
      if (bestId === this.config.eotTokenId) break;
      tokens.push(bestId);
    }

    const decoded = tokens
      .slice(1)
      .map((id) => this.w.vocab[id] ?? '')
      .join('')
      .trim();
    return { tokenIds: tokens, text: decoded, encoderFrames: encFrames };
  }
}
