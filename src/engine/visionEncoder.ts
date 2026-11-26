/**
 * Real SigLIP / ViT Vision & Video Encoder (`src/engine/visionEncoder.ts`)
 *
 * Executes real multimodal projector (`mmproj`) GGUF weights (e.g. `SmolVLM2-256M-Video-Instruct`,
 * `SmolVLM-256M-Instruct`, `CLIP` / `SigLIP` ViT) directly on VerdantUI's shared `vTensor`
 * kernels (`verdantConv2d`, `verdantLayerNorm`, `verdantGelu`, `verdantMulMat`), producing real
 * `[n_visual_tokens, d_llm]` embeddings that inject straight into `LlamaVTensorRuntime.prefillEmbeddingSequence`
 * with zero intermediate serialization and zero foreign runtimes.
 */

import { GgmlType, type GgufTensorInfo, parseGGUF, type ParsedGgufModel } from './gguf';
import {
  verdantAdd,
  verdantConv2d,
  verdantDecodeRow,
  verdantGelu,
  verdantLayerNorm,
  verdantMulMat,
} from './verdantKernels';

export interface VisionEncoderConfig {
  readonly imageSize: number;
  readonly patchSize: number;
  readonly hiddenSize: number;
  readonly intermediateSize: number;
  readonly numLayers: number;
  readonly numHeads: number;
  readonly projectionDim: number;
  readonly scaleFactor: number; // pixel-shuffle spatial merge factor (e.g. 2 or 3 for SmolVLM, 1 for plain ViT)
  readonly layerNormEps: number;
  readonly useQuickGelu: boolean;
  readonly imageMean: readonly [number, number, number];
  readonly imageStd: readonly [number, number, number];
}

export interface VisionLayerWeights {
  readonly ln1W: Float32Array;
  readonly ln1B: Float32Array | null;
  readonly qW: TensorSlice;
  readonly qB: Float32Array | null;
  readonly kW: TensorSlice;
  readonly kB: Float32Array | null;
  readonly vW: TensorSlice;
  readonly vB: Float32Array | null;
  readonly oW: TensorSlice;
  readonly oB: Float32Array | null;
  readonly ln2W: Float32Array;
  readonly ln2B: Float32Array | null;
  readonly fc1W: TensorSlice;
  readonly fc1B: Float32Array | null;
  readonly fc2W: TensorSlice;
  readonly fc2B: Float32Array | null;
}

export interface TensorSlice {
  readonly bytes: Uint8Array;
  readonly offset: number;
  readonly type: GgmlType;
  readonly rows: number;
  readonly cols: number;
}

export interface VisionWeights {
  readonly config: VisionEncoderConfig;
  readonly patchWeight: Float32Array; // [hiddenSize, 3, patchSize, patchSize]
  readonly patchBias: Float32Array | null;
  readonly posEmbd: Float32Array; // [numPatches, hiddenSize]
  readonly layers: readonly VisionLayerWeights[];
  readonly postLnW: Float32Array | null;
  readonly postLnB: Float32Array | null;
  readonly connectorW1: TensorSlice;
  readonly connectorB1: Float32Array | null;
  readonly connectorW2?: TensorSlice | null;
  readonly connectorB2?: Float32Array | null;
}

export interface VisualFrameInput {
  readonly rgba: Uint8Array | Uint8ClampedArray;
  readonly width: number;
  readonly height: number;
}

/**
 * Bilinearly resizes RGBA8 pixels to `dstSize × dstSize` and normalizes RGB channels
 * into CHW `Float32Array[3 * dstSize * dstSize]` using the model's exact mean/std.
 */
export function preprocessRgbaToChw(
  rgba: Uint8Array | Uint8ClampedArray,
  srcW: number,
  srcH: number,
  dstSize: number,
  mean: readonly [number, number, number] = [0.5, 0.5, 0.5],
  std: readonly [number, number, number] = [0.5, 0.5, 0.5],
): Float32Array {
  const out = new Float32Array(3 * dstSize * dstSize);
  const plane = dstSize * dstSize;
  const invStd0 = 1 / (std[0] || 0.5);
  const invStd1 = 1 / (std[1] || 0.5);
  const invStd2 = 1 / (std[2] || 0.5);

  const scaleX = srcW > 1 && dstSize > 1 ? (srcW - 1) / (dstSize - 1) : 0;
  const scaleY = srcH > 1 && dstSize > 1 ? (srcH - 1) / (dstSize - 1) : 0;

  for (let dy = 0; dy < dstSize; dy++) {
    const sy = dy * scaleY;
    const y0 = Math.min(srcH - 1, Math.floor(sy));
    const y1 = Math.min(srcH - 1, y0 + 1);
    const fy = sy - y0;
    for (let dx = 0; dx < dstSize; dx++) {
      const sx = dx * scaleX;
      const x0 = Math.min(srcW - 1, Math.floor(sx));
      const x1 = Math.min(srcW - 1, x0 + 1);
      const fx = sx - x0;

      const idx00 = (y0 * srcW + x0) * 4;
      const idx10 = (y0 * srcW + x1) * 4;
      const idx01 = (y1 * srcW + x0) * 4;
      const idx11 = (y1 * srcW + x1) * 4;

      const w00 = (1 - fx) * (1 - fy);
      const w10 = fx * (1 - fy);
      const w01 = (1 - fx) * fy;
      const w11 = fx * fy;

      const r =
        ((rgba[idx00]! * w00 + rgba[idx10]! * w10 + rgba[idx01]! * w01 + rgba[idx11]! * w11) / 255 -
          mean[0]) *
        invStd0;
      const g =
        ((rgba[idx00 + 1]! * w00 +
          rgba[idx10 + 1]! * w10 +
          rgba[idx01 + 1]! * w01 +
          rgba[idx11 + 1]! * w11) /
          255 -
          mean[1]) *
        invStd1;
      const b =
        ((rgba[idx00 + 2]! * w00 +
          rgba[idx10 + 2]! * w10 +
          rgba[idx01 + 2]! * w01 +
          rgba[idx11 + 2]! * w11) /
          255 -
          mean[2]) *
        invStd2;

      const dstIdx = dy * dstSize + dx;
      out[dstIdx] = Math.fround(r);
      out[plane + dstIdx] = Math.fround(g);
      out[2 * plane + dstIdx] = Math.fround(b);
    }
  }
  return out;
}

function decodeTensorToF32(bytes: Uint8Array, dataOffset: number, info: GgufTensorInfo): Float32Array {
  const total = info.dims.reduce((a, b) => a * b, 1);
  const out = new Float32Array(total);
  const ne0 = info.dims[0] ?? total;
  const rows = Math.max(1, Math.floor(total / ne0));
  const rowBytes = Math.floor(info.byteSize / rows);
  const rowBuf = new Float32Array(ne0);
  for (let r = 0; r < rows; r++) {
    verdantDecodeRow(bytes, dataOffset + info.offset + r * rowBytes, info.type, ne0, rowBuf);
    out.set(rowBuf, r * ne0);
  }
  return out;
}

function findTensor(model: ParsedGgufModel, candidates: readonly string[]): GgufTensorInfo | null {
  for (const name of candidates) {
    const t = model.tensorMap.get(name);
    if (t) return t;
  }
  return null;
}

function toTensorSlice(bytes: Uint8Array, dataOffset: number, info: GgufTensorInfo): TensorSlice {
  return {
    bytes,
    offset: dataOffset + info.offset,
    type: info.type,
    cols: info.dims[0] ?? 1,
    rows: info.dims[1] ?? 1,
  };
}

function metaNum(model: ParsedGgufModel, keys: readonly string[], fallback: number): number {
  for (const k of keys) {
    const v = model.metadata[k];
    if (typeof v === 'number' && Number.isFinite(v)) return v;
    if (typeof v === 'bigint') return Number(v);
  }
  return fallback;
}

function metaVec3(
  model: ParsedGgufModel,
  key: string,
  fallback: readonly [number, number, number],
): readonly [number, number, number] {
  const v = model.metadata[key];
  if (Array.isArray(v) && v.length >= 3) {
    return [Number(v[0]) || fallback[0], Number(v[1]) || fallback[1], Number(v[2]) || fallback[2]];
  }
  return fallback;
}

/**
 * Loads a real SigLIP / ViT `mmproj` GGUF buffer (`SmolVLM2`, `SmolVLM`, `CLIP`) into `VisionWeights`.
 */
export function loadVisionWeightsFromGguf(ggufBytes: Uint8Array): VisionWeights {
  const parsed = parseGGUF(ggufBytes);
  const dataOffset = parsed.dataOffset;

  const imageSize = metaNum(parsed, ['clip.vision.image_size'], 384);
  const patchSize = metaNum(parsed, ['clip.vision.patch_size'], 14);
  const hiddenSize = metaNum(parsed, ['clip.vision.embedding_length'], 768);
  const intermediateSize = metaNum(parsed, ['clip.vision.feed_forward_length'], hiddenSize * 4);
  const numLayers = metaNum(parsed, ['clip.vision.block_count'], 12);
  const numHeads = metaNum(parsed, ['clip.vision.attention.head_count'], 12);
  const projectionDim = metaNum(parsed, ['clip.vision.projection_dim'], hiddenSize);
  const scaleFactor = metaNum(parsed, ['clip.vision.proj_scale_factor', 'clip.vision.scale_factor'], 1);
  const layerNormEps = metaNum(parsed, ['clip.vision.attention.layer_norm_epsilon'], 1e-6);
  const useQuickGelu = Boolean(parsed.metadata['clip.use_gelu'] === false);
  const imageMean = metaVec3(parsed, 'clip.vision.image_mean', [0.5, 0.5, 0.5]);
  const imageStd = metaVec3(parsed, 'clip.vision.image_std', [0.5, 0.5, 0.5]);

  const patchInfo = findTensor(parsed, [
    'v.patch_embd.weight',
    'v.vision_model.embeddings.patch_embedding.weight',
  ]);
  if (!patchInfo) throw new Error('Vision GGUF is missing patch embedding weight tensor.');
  const patchWeight = decodeTensorToF32(ggufBytes, dataOffset, patchInfo);
  const patchBiasInfo = findTensor(parsed, [
    'v.patch_embd.bias',
    'v.vision_model.embeddings.patch_embedding.bias',
  ]);
  const patchBias = patchBiasInfo ? decodeTensorToF32(ggufBytes, dataOffset, patchBiasInfo) : null;

  const posInfo = findTensor(parsed, [
    'v.position_embd.weight',
    'v.vision_model.embeddings.position_embedding.weight',
  ]);
  if (!posInfo) throw new Error('Vision GGUF is missing position embedding weight tensor.');
  const posEmbd = decodeTensorToF32(ggufBytes, dataOffset, posInfo);

  const layers: VisionLayerWeights[] = [];
  for (let i = 0; i < numLayers; i++) {
    const req = (names: readonly string[], label: string): GgufTensorInfo => {
      const t = findTensor(parsed, names);
      if (!t) throw new Error(`Vision GGUF missing layer ${i} ${label}`);
      return t;
    };
    const optF32 = (names: readonly string[]): Float32Array | null => {
      const t = findTensor(parsed, names);
      return t ? decodeTensorToF32(ggufBytes, dataOffset, t) : null;
    };

    const ln1W = decodeTensorToF32(
      ggufBytes,
      dataOffset,
      req([`v.blk.${i}.ln1.weight`, `v.vision_model.encoder.layers.${i}.layer_norm1.weight`], 'ln1.weight'),
    );
    const ln1B = optF32([`v.blk.${i}.ln1.bias`, `v.vision_model.encoder.layers.${i}.layer_norm1.bias`]);
    const qW = toTensorSlice(
      ggufBytes,
      dataOffset,
      req([`v.blk.${i}.attn_q.weight`, `v.vision_model.encoder.layers.${i}.self_attn.q_proj.weight`], 'attn_q.weight'),
    );
    const qB = optF32([`v.blk.${i}.attn_q.bias`, `v.vision_model.encoder.layers.${i}.self_attn.q_proj.bias`]);
    const kW = toTensorSlice(
      ggufBytes,
      dataOffset,
      req([`v.blk.${i}.attn_k.weight`, `v.vision_model.encoder.layers.${i}.self_attn.k_proj.weight`], 'attn_k.weight'),
    );
    const kB = optF32([`v.blk.${i}.attn_k.bias`, `v.vision_model.encoder.layers.${i}.self_attn.k_proj.bias`]);
    const vW = toTensorSlice(
      ggufBytes,
      dataOffset,
      req([`v.blk.${i}.attn_v.weight`, `v.vision_model.encoder.layers.${i}.self_attn.v_proj.weight`], 'attn_v.weight'),
    );
    const vB = optF32([`v.blk.${i}.attn_v.bias`, `v.vision_model.encoder.layers.${i}.self_attn.v_proj.bias`]);
    const oW = toTensorSlice(
      ggufBytes,
      dataOffset,
      req([`v.blk.${i}.attn_out.weight`, `v.vision_model.encoder.layers.${i}.self_attn.out_proj.weight`], 'attn_out.weight'),
    );
    const oB = optF32([`v.blk.${i}.attn_out.bias`, `v.vision_model.encoder.layers.${i}.self_attn.out_proj.bias`]);
    const ln2W = decodeTensorToF32(
      ggufBytes,
      dataOffset,
      req([`v.blk.${i}.ln2.weight`, `v.vision_model.encoder.layers.${i}.layer_norm2.weight`], 'ln2.weight'),
    );
    const ln2B = optF32([`v.blk.${i}.ln2.bias`, `v.vision_model.encoder.layers.${i}.layer_norm2.bias`]);
    const fc1W = toTensorSlice(
      ggufBytes,
      dataOffset,
      req([`v.blk.${i}.ffn_down.weight`, `v.blk.${i}.ffn_up.weight`, `v.vision_model.encoder.layers.${i}.mlp.fc1.weight`], 'mlp.fc1.weight'),
    );
    const fc1B = optF32([`v.blk.${i}.ffn_down.bias`, `v.blk.${i}.ffn_up.bias`, `v.vision_model.encoder.layers.${i}.mlp.fc1.bias`]);
    const fc2W = toTensorSlice(
      ggufBytes,
      dataOffset,
      req([`v.blk.${i}.ffn_up.weight`, `v.blk.${i}.ffn_down.weight`, `v.vision_model.encoder.layers.${i}.mlp.fc2.weight`], 'mlp.fc2.weight'),
    );
    const fc2B = optF32([`v.blk.${i}.ffn_up.bias`, `v.blk.${i}.ffn_down.bias`, `v.vision_model.encoder.layers.${i}.mlp.fc2.bias`]);

    layers.push({ ln1W, ln1B, qW, qB, kW, kB, vW, vB, oW, oB, ln2W, ln2B, fc1W, fc1B, fc2W, fc2B });
  }

  const postLnWInfo = findTensor(parsed, ['v.post_ln.weight', 'v.vision_model.post_layernorm.weight']);
  const postLnBInfo = findTensor(parsed, ['v.post_ln.bias', 'v.vision_model.post_layernorm.bias']);
  const postLnW = postLnWInfo ? decodeTensorToF32(ggufBytes, dataOffset, postLnWInfo) : null;
  const postLnB = postLnBInfo ? decodeTensorToF32(ggufBytes, dataOffset, postLnBInfo) : null;

  const mm0Info = findTensor(parsed, ['mm.0.weight', 'mm.model.connector.modality_projection.proj.weight', 'mm.input_norm.weight']);
  if (!mm0Info) throw new Error('Vision GGUF missing multimodal projector tensor (`mm.0.weight`).');
  const mm0BInfo = findTensor(parsed, ['mm.0.bias', 'mm.model.connector.modality_projection.proj.bias']);
  const mm2Info = findTensor(parsed, ['mm.2.weight']);
  const mm2BInfo = findTensor(parsed, ['mm.2.bias']);

  return {
    config: {
      imageSize,
      patchSize,
      hiddenSize,
      intermediateSize,
      numLayers,
      numHeads,
      projectionDim,
      scaleFactor,
      layerNormEps,
      useQuickGelu,
      imageMean,
      imageStd,
    },
    patchWeight,
    patchBias,
    posEmbd,
    layers,
    postLnW,
    postLnB,
    connectorW1: toTensorSlice(ggufBytes, dataOffset, mm0Info),
    connectorB1: mm0BInfo ? decodeTensorToF32(ggufBytes, dataOffset, mm0BInfo) : null,
    connectorW2: mm2Info ? toTensorSlice(ggufBytes, dataOffset, mm2Info) : null,
    connectorB2: mm2BInfo ? decodeTensorToF32(ggufBytes, dataOffset, mm2BInfo) : null,
  };
}

function linearWithBias(slice: TensorSlice, x: Float32Array, bias: Float32Array | null, out: Float32Array): void {
  verdantMulMat(slice.bytes, slice.offset, slice.type, slice.cols, slice.rows, x, out);
  if (bias) {
    for (let r = 0; r < slice.rows; r++) {
      out[r] = Math.fround(out[r]! + bias[r]!);
    }
  }
}

/**
 * Real SigLIP / ViT Vision & Video Encoder executing strictly on shared `vTensor` kernels.
 */
export class SiglipVisionEncoder {
  readonly config: VisionEncoderConfig;

  constructor(private readonly weights: VisionWeights) {
    this.config = weights.config;
  }

  /**
   * Encodes one RGBA image frame into real projected visual embedding tokens `[nTokens, projectionDim]`
   * ready for direct injection into `LlamaVTensorRuntime.prefillEmbeddingSequence`.
   */
  encodeFrame(frame: VisualFrameInput): readonly Float32Array[] {
    const cfg = this.config;
    const chw = preprocessRgbaToChw(
      frame.rgba,
      frame.width,
      frame.height,
      cfg.imageSize,
      cfg.imageMean,
      cfg.imageStd,
    );

    const grid = Math.floor(cfg.imageSize / cfg.patchSize);
    const numPatches = grid * grid;
    const D = cfg.hiddenSize;

    // 1. Real 2D Patch Embedding Convolution: [3, H, W] -> [D, grid, grid]
    const convOut = new Float32Array(D * numPatches);
    verdantConv2d(
      chw,
      this.weights.patchWeight,
      this.weights.patchBias,
      convOut,
      3,
      D,
      cfg.imageSize,
      cfg.imageSize,
      cfg.patchSize,
      cfg.patchSize,
      cfg.patchSize,
      0,
    );

    // Transpose [D, numPatches] -> [numPatches, D] and add learned positional embeddings
    const hidden = new Float32Array(numPatches * D);
    for (let p = 0; p < numPatches; p++) {
      const pBase = p * D;
      for (let d = 0; d < D; d++) {
        const posVal = pBase + d < this.weights.posEmbd.length ? this.weights.posEmbd[pBase + d]! : 0;
        hidden[pBase + d] = Math.fround(convOut[d * numPatches + p]! + posVal);
      }
    }

    // 2. Real SigLIP Vision Transformer Blocks (bidirectional Multi-Head Attention + GELU MLP)
    const headDim = Math.floor(D / cfg.numHeads);
    const attnScale = 1.0 / Math.sqrt(headDim);
    const normBuf = new Float32Array(D);
    const Q = new Float32Array(numPatches * D);
    const K = new Float32Array(numPatches * D);
    const V = new Float32Array(numPatches * D);
    const attnOut = new Float32Array(numPatches * D);
    const projRow = new Float32Array(D);
    const scores = new Float32Array(numPatches);
    const mlpHid = new Float32Array(cfg.intermediateSize);

    for (const layer of this.weights.layers) {
      // Pre-LN + Q/K/V projections for all patches
      for (let p = 0; p < numPatches; p++) {
        verdantLayerNorm(hidden, layer.ln1W, layer.ln1B, normBuf, D, cfg.layerNormEps, p * D, 0);
        const qView = Q.subarray(p * D, (p + 1) * D);
        const kView = K.subarray(p * D, (p + 1) * D);
        const vView = V.subarray(p * D, (p + 1) * D);
        linearWithBias(layer.qW, normBuf, layer.qB, qView);
        linearWithBias(layer.kW, normBuf, layer.kB, kView);
        linearWithBias(layer.vW, normBuf, layer.vB, vView);
      }

      // Bidirectional Multi-Head Attention
      attnOut.fill(0);
      for (let h = 0; h < cfg.numHeads; h++) {
        const hOff = h * headDim;
        for (let p = 0; p < numPatches; p++) {
          const qBase = p * D + hOff;
          let maxS = -Infinity;
          for (let q = 0; q < numPatches; q++) {
            const kBase = q * D + hOff;
            let dot = 0.0;
            for (let d = 0; d < headDim; d++) dot += Q[qBase + d]! * K[kBase + d]!;
            const s = dot * attnScale;
            scores[q] = s;
            if (s > maxS) maxS = s;
          }
          let denom = 0.0;
          for (let q = 0; q < numPatches; q++) {
            const e = Math.exp(scores[q]! - maxS);
            scores[q] = e;
            denom += e;
          }
          const invD = denom > 0 ? 1.0 / denom : 0;
          const outBase = p * D + hOff;
          for (let q = 0; q < numPatches; q++) {
            const w = scores[q]! * invD;
            const vBase = q * D + hOff;
            for (let d = 0; d < headDim; d++) {
              attnOut[outBase + d] = Math.fround(attnOut[outBase + d]! + w * V[vBase + d]!);
            }
          }
        }
      }

      // Output projection + residual + Post-Attn LN + GELU MLP + residual
      for (let p = 0; p < numPatches; p++) {
        const pBase = p * D;
        const hiddenSlice = hidden.subarray(pBase, pBase + D);
        linearWithBias(layer.oW, attnOut.subarray(pBase, pBase + D), layer.oB, projRow);
        verdantAdd(hiddenSlice, projRow, hiddenSlice, D);

        verdantLayerNorm(hidden, layer.ln2W, layer.ln2B, normBuf, D, cfg.layerNormEps, pBase, 0);
        linearWithBias(layer.fc1W, normBuf, layer.fc1B, mlpHid);
        verdantGelu(mlpHid, mlpHid, layer.fc1W.rows, cfg.useQuickGelu);
        linearWithBias(layer.fc2W, mlpHid, layer.fc2B, projRow);
        verdantAdd(hiddenSlice, projRow, hiddenSlice, D);
      }
    }

    // 3. Final Post-LayerNorm
    if (this.weights.postLnW) {
      for (let p = 0; p < numPatches; p++) {
        verdantLayerNorm(hidden, this.weights.postLnW, this.weights.postLnB, hidden, D, cfg.layerNormEps, p * D, p * D);
      }
    }

    // 4. Spatial Pixel-Shuffle Merge (for SmolVLM / SmolVLM2) + Multimodal Connector Projection
    const sf = Math.max(1, cfg.scaleFactor);
    const outGrid = Math.max(1, Math.floor(grid / sf));
    const mergedDim = D * sf * sf;
    const mergedBuf = new Float32Array(mergedDim);
    const tokens: Float32Array[] = [];

    for (let gy = 0; gy < outGrid; gy++) {
      for (let gx = 0; gx < outGrid; gx++) {
        let cursor = 0;
        for (let sy = 0; sy < sf; sy++) {
          for (let sx = 0; sx < sf; sx++) {
            const pIdx = (gy * sf + sy) * grid + (gx * sf + sx);
            mergedBuf.set(hidden.subarray(pIdx * D, (pIdx + 1) * D), cursor);
            cursor += D;
          }
        }
        const stage1 = new Float32Array(this.weights.connectorW1.rows);
        linearWithBias(this.weights.connectorW1, mergedBuf, this.weights.connectorB1, stage1);
        if (this.weights.connectorW2) {
          verdantGelu(stage1, stage1, stage1.length, cfg.useQuickGelu);
          const stage2 = new Float32Array(this.weights.connectorW2.rows);
          linearWithBias(this.weights.connectorW2, stage1, this.weights.connectorB2 ?? null, stage2);
          tokens.push(stage2);
        } else {
          tokens.push(stage1);
        }
      }
    }

    return tokens;
  }

  /** Encodes multiple video frames sequentially into one coherent visual embedding stream. */
  encodeVideoFrames(frames: readonly VisualFrameInput[]): readonly Float32Array[] {
    const all: Float32Array[] = [];
    for (const f of frames) {
      for (const tok of this.encodeFrame(f)) all.push(tok);
    }
    return all;
  }
}
