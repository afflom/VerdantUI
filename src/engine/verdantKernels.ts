/**
 * Verdant Tensor deterministic numerical primitives substrate.
 *
 * Per github.com/afflom/vTensor architecture:
 * - Verdant Tensor is the sole source of tensor computation executed by the vTensor GGML provider.
 * - RMS_NORM and reductions accumulate in IEEE 754 F64 before narrowing to F32 (VT-45 / VT-46).
 * - Quantized row products decode exact upstream GGML block layouts (F32, F16, BF16, Q4_0, Q4_1,
 *   Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K) matching ggml-quants.c bit-for-bit.
 * - ROPE (normal & NeoX), SWIGLU, and FLASH_ATTN_EXT preserve upstream GGML transformer semantics.
 */

import { GgmlType } from './gguf';

export const VERDANT_TENSOR_VERSION = '2.22.4';
export const VERDANT_TENSOR_COMMIT = '9d22f5ac00000000000000000000000000000000';

/** IEEE 754 binary16 (F16) to binary32 (F32) exact conversion. */
export function f16ToF32(h: number): number {
  const sign = (h & 0x8000) !== 0 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x03ff;

  if (exp === 0) {
    if (frac === 0) return sign * 0;
    return sign * Math.pow(2, -14) * (frac / 1024);
  }
  if (exp === 0x1f) {
    return frac === 0 ? sign * Infinity : NaN;
  }
  return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

/** Precomputed F16 lookup table (65,536 entries) for fast deterministic decoding. */
const F16_LUT = new Float32Array(65536);
for (let i = 0; i < 65536; i++) {
  F16_LUT[i] = f16ToF32(i);
}

export function f16BitsToF32(bits: number): number {
  return F16_LUT[bits & 0xffff];
}

const bf16Buf = new ArrayBuffer(4);
const bf16U32 = new Uint32Array(bf16Buf);
const bf16F32 = new Float32Array(bf16Buf);

/** BF16 (top 16 bits of IEEE 754 F32) to F32 conversion. */
export function bf16BitsToF32(bits: number): number {
  bf16U32[0] = (bits & 0xffff) << 16;
  return bf16F32[0];
}

/** Precomputed BF16 lookup table (65,536 entries) for fast deterministic decoding. */
const BF16_LUT = new Float32Array(65536);
for (let i = 0; i < 65536; i++) {
  BF16_LUT[i] = bf16BitsToF32(i);
}

/**
 * Upstream ggml-quants.c `get_scale_min_k4(j, q)` helper for Q4_K and Q5_K 6-bit packed scales/mins.
 * Packs `[scale, min]` as `(scale << 8) | min` to avoid heap allocation in hot loops.
 */
function getScaleMinK4Packed(j: number, bytes: Uint8Array, scalesOffset: number): number {
  if (j < 4) {
    const sc = bytes[scalesOffset + j] & 63;
    const m = bytes[scalesOffset + j + 4] & 63;
    return (sc << 8) | m;
  }
  const bHi = bytes[scalesOffset + j + 4];
  const sc = (bHi & 0x0f) | ((bytes[scalesOffset + j - 4] >> 6) << 4);
  const m = (bHi >> 4) | ((bytes[scalesOffset + j] >> 6) << 4);
  return (sc << 8) | m;
}

/** Zero-allocation scratch buffer for unpacking 16 signed Q3_K scales [-32..31]. */
const q3kScalesScratch = new Int8Array(16);

/** Unpacks upstream ggml-quants.c Q3_K 12-byte packed 6-bit scales into `q3kScalesScratch`. */
function unpackQ3KScales(bytes: Uint8Array, scOff: number): Int8Array {
  const kmask1 = 0x03030303;
  const kmask2 = 0x0f0f0f0f;
  const a0 =
    (bytes[scOff] |
      (bytes[scOff + 1] << 8) |
      (bytes[scOff + 2] << 16) |
      (bytes[scOff + 3] << 24)) >>>
    0;
  const a1 =
    (bytes[scOff + 4] |
      (bytes[scOff + 5] << 8) |
      (bytes[scOff + 6] << 16) |
      (bytes[scOff + 7] << 24)) >>>
    0;
  const tmp =
    (bytes[scOff + 8] |
      (bytes[scOff + 9] << 8) |
      (bytes[scOff + 10] << 16) |
      (bytes[scOff + 11] << 24)) >>>
    0;

  const u0 = ((a0 & kmask2) | (((tmp >>> 0) & kmask1) << 4)) >>> 0;
  const u1 = ((a1 & kmask2) | (((tmp >>> 2) & kmask1) << 4)) >>> 0;
  const u2 = (((a0 >>> 4) & kmask2) | (((tmp >>> 4) & kmask1) << 4)) >>> 0;
  const u3 = (((a1 >>> 4) & kmask2) | (((tmp >>> 6) & kmask1) << 4)) >>> 0;

  q3kScalesScratch[0] = (u0 & 0x3f) - 32;
  q3kScalesScratch[1] = ((u0 >>> 8) & 0x3f) - 32;
  q3kScalesScratch[2] = ((u0 >>> 16) & 0x3f) - 32;
  q3kScalesScratch[3] = ((u0 >>> 24) & 0x3f) - 32;
  q3kScalesScratch[4] = (u1 & 0x3f) - 32;
  q3kScalesScratch[5] = ((u1 >>> 8) & 0x3f) - 32;
  q3kScalesScratch[6] = ((u1 >>> 16) & 0x3f) - 32;
  q3kScalesScratch[7] = ((u1 >>> 24) & 0x3f) - 32;
  q3kScalesScratch[8] = (u2 & 0x3f) - 32;
  q3kScalesScratch[9] = ((u2 >>> 8) & 0x3f) - 32;
  q3kScalesScratch[10] = ((u2 >>> 16) & 0x3f) - 32;
  q3kScalesScratch[11] = ((u2 >>> 24) & 0x3f) - 32;
  q3kScalesScratch[12] = (u3 & 0x3f) - 32;
  q3kScalesScratch[13] = ((u3 >>> 8) & 0x3f) - 32;
  q3kScalesScratch[14] = ((u3 >>> 16) & 0x3f) - 32;
  q3kScalesScratch[15] = ((u3 >>> 24) & 0x3f) - 32;
  return q3kScalesScratch;
}

/** Decode a single quantized or floating-point row from raw bytes into F32 destination. */
export function verdantDecodeRow(
  srcBytes: Uint8Array,
  srcByteOffset: number,
  srcType: GgmlType,
  cols: number,
  dst: Float32Array,
  dstOffset = 0,
): void {
  if (srcType === GgmlType.F32) {
    const absOff = srcBytes.byteOffset + srcByteOffset;
    if ((absOff & 3) === 0) {
      const f32 = new Float32Array(srcBytes.buffer, absOff, cols);
      dst.set(f32, dstOffset);
      return;
    }
    const view = new DataView(srcBytes.buffer, absOff, cols * 4);
    for (let i = 0; i < cols; i++) {
      dst[dstOffset + i] = view.getFloat32(i * 4, true);
    }
    return;
  }

  if (srcType === GgmlType.F16) {
    const absOff = srcBytes.byteOffset + srcByteOffset;
    if ((absOff & 1) === 0) {
      const u16 = new Uint16Array(srcBytes.buffer, absOff, cols);
      for (let i = 0; i < cols; i++) {
        dst[dstOffset + i] = F16_LUT[u16[i]];
      }
      return;
    }
    const view = new DataView(srcBytes.buffer, absOff, cols * 2);
    for (let i = 0; i < cols; i++) {
      dst[dstOffset + i] = F16_LUT[view.getUint16(i * 2, true)];
    }
    return;
  }

  if (srcType === GgmlType.BF16) {
    const absOff = srcBytes.byteOffset + srcByteOffset;
    if ((absOff & 1) === 0) {
      const u16 = new Uint16Array(srcBytes.buffer, absOff, cols);
      for (let i = 0; i < cols; i++) {
        dst[dstOffset + i] = BF16_LUT[u16[i]];
      }
      return;
    }
    const view = new DataView(srcBytes.buffer, absOff, cols * 2);
    for (let i = 0; i < cols; i++) {
      dst[dstOffset + i] = BF16_LUT[view.getUint16(i * 2, true)];
    }
    return;
  }

  if (srcType === GgmlType.Q8_0) {
    // Q8_0 block: 2-byte F16 delta + 32 int8 quants = 34 bytes per 32 elements.
    const numBlocks = cols / 32;
    let ptr = srcByteOffset;
    for (let b = 0; b < numBlocks; b++) {
      const scale = F16_LUT[srcBytes[ptr] | (srcBytes[ptr + 1] << 8)];
      ptr += 2;
      const outBase = dstOffset + b * 32;
      for (let j = 0; j < 32; j++) {
        const q = (srcBytes[ptr + j] << 24) >> 24;
        dst[outBase + j] = q * scale;
      }
      ptr += 32;
    }
    return;
  }

  if (srcType === GgmlType.Q4_0) {
    // Q4_0 block: 2-byte F16 delta + 16 packed bytes = 18 bytes per 32 elements.
    const numBlocks = cols / 32;
    let ptr = srcByteOffset;
    for (let b = 0; b < numBlocks; b++) {
      const scale = F16_LUT[srcBytes[ptr] | (srcBytes[ptr + 1] << 8)];
      ptr += 2;
      const outBase = dstOffset + b * 32;
      for (let j = 0; j < 16; j++) {
        const packed = srcBytes[ptr + j];
        dst[outBase + j] = ((packed & 0x0f) - 8) * scale;
        dst[outBase + j + 16] = ((packed >> 4) - 8) * scale;
      }
      ptr += 16;
    }
    return;
  }

  if (srcType === GgmlType.Q4_1) {
    // Q4_1 block: 2-byte F16 delta + 2-byte F16 min + 16 packed bytes = 20 bytes per 32 elements.
    const numBlocks = cols / 32;
    let ptr = srcByteOffset;
    for (let b = 0; b < numBlocks; b++) {
      const d = F16_LUT[srcBytes[ptr] | (srcBytes[ptr + 1] << 8)];
      const m = F16_LUT[srcBytes[ptr + 2] | (srcBytes[ptr + 3] << 8)];
      ptr += 4;
      const outBase = dstOffset + b * 32;
      for (let j = 0; j < 16; j++) {
        const packed = srcBytes[ptr + j];
        dst[outBase + j] = (packed & 0x0f) * d + m;
        dst[outBase + j + 16] = (packed >> 4) * d + m;
      }
      ptr += 16;
    }
    return;
  }

  if (srcType === GgmlType.Q5_0) {
    // Q5_0 block: 2-byte F16 delta + 4-byte uint32 high bits + 16 packed low nibbles = 22 bytes per 32 elements.
    const numBlocks = cols / 32;
    let ptr = srcByteOffset;
    for (let b = 0; b < numBlocks; b++) {
      const d = F16_LUT[srcBytes[ptr] | (srcBytes[ptr + 1] << 8)];
      const qh =
        (srcBytes[ptr + 2] |
          (srcBytes[ptr + 3] << 8) |
          (srcBytes[ptr + 4] << 16) |
          (srcBytes[ptr + 5] << 24)) >>>
        0;
      ptr += 6;
      const outBase = dstOffset + b * 32;
      for (let j = 0; j < 16; j++) {
        const packed = srcBytes[ptr + j];
        const xh0 = ((qh >>> j) & 1) << 4;
        const xh1 = ((qh >>> (j + 16)) & 1) << 4;
        dst[outBase + j] = (((packed & 0x0f) | xh0) - 16) * d;
        dst[outBase + j + 16] = (((packed >> 4) | xh1) - 16) * d;
      }
      ptr += 16;
    }
    return;
  }

  if (srcType === GgmlType.Q5_1) {
    // Q5_1 block: 2-byte F16 delta + 2-byte F16 min + 4-byte uint32 high bits + 16 packed low nibbles = 24 bytes.
    const numBlocks = cols / 32;
    let ptr = srcByteOffset;
    for (let b = 0; b < numBlocks; b++) {
      const d = F16_LUT[srcBytes[ptr] | (srcBytes[ptr + 1] << 8)];
      const m = F16_LUT[srcBytes[ptr + 2] | (srcBytes[ptr + 3] << 8)];
      const qh =
        (srcBytes[ptr + 4] |
          (srcBytes[ptr + 5] << 8) |
          (srcBytes[ptr + 6] << 16) |
          (srcBytes[ptr + 7] << 24)) >>>
        0;
      ptr += 8;
      const outBase = dstOffset + b * 32;
      for (let j = 0; j < 16; j++) {
        const packed = srcBytes[ptr + j];
        const xh0 = ((qh >>> j) & 1) << 4;
        const xh1 = ((qh >>> (j + 16)) & 1) << 4;
        dst[outBase + j] = ((packed & 0x0f) | xh0) * d + m;
        dst[outBase + j + 16] = ((packed >> 4) | xh1) * d + m;
      }
      ptr += 16;
    }
    return;
  }

  if (srcType === GgmlType.Q4_K) {
    // Q4_K super-block (256 elements = 144 bytes): d(2), dmin(2), scales(12), qs(128).
    const numBlocks = cols / 256;
    let ptr = srcByteOffset;
    for (let b = 0; b < numBlocks; b++) {
      const d = F16_LUT[srcBytes[ptr] | (srcBytes[ptr + 1] << 8)];
      const dmin = F16_LUT[srcBytes[ptr + 2] | (srcBytes[ptr + 3] << 8)];
      const scalesOff = ptr + 4;
      let qPtr = ptr + 16;
      let outPtr = dstOffset + b * 256;

      for (let is = 0; is < 8; is += 2) {
        const sm0 = getScaleMinK4Packed(is, srcBytes, scalesOff);
        const d1 = d * (sm0 >> 8);
        const m1 = dmin * (sm0 & 0xff);
        const sm1 = getScaleMinK4Packed(is + 1, srcBytes, scalesOff);
        const d2 = d * (sm1 >> 8);
        const m2 = dmin * (sm1 & 0xff);

        for (let l = 0; l < 32; l++) {
          const packed = srcBytes[qPtr + l];
          dst[outPtr + l] = d1 * (packed & 0x0f) - m1;
          dst[outPtr + 32 + l] = d2 * (packed >> 4) - m2;
        }
        qPtr += 32;
        outPtr += 64;
      }
      ptr += 144;
    }
    return;
  }

  if (srcType === GgmlType.Q5_K) {
    // Q5_K super-block (256 elements = 176 bytes): d(2), dmin(2), scales(12), qh(32), qs(128).
    const numBlocks = cols / 256;
    let ptr = srcByteOffset;
    for (let b = 0; b < numBlocks; b++) {
      const d = F16_LUT[srcBytes[ptr] | (srcBytes[ptr + 1] << 8)];
      const dmin = F16_LUT[srcBytes[ptr + 2] | (srcBytes[ptr + 3] << 8)];
      const scalesOff = ptr + 4;
      const qhOff = ptr + 16;
      let qPtr = ptr + 48;
      let outPtr = dstOffset + b * 256;

      let u1 = 1;
      let u2 = 2;
      for (let is = 0; is < 8; is += 2) {
        const sm0 = getScaleMinK4Packed(is, srcBytes, scalesOff);
        const d1 = d * (sm0 >> 8);
        const m1 = dmin * (sm0 & 0xff);
        const sm1 = getScaleMinK4Packed(is + 1, srcBytes, scalesOff);
        const d2 = d * (sm1 >> 8);
        const m2 = dmin * (sm1 & 0xff);

        for (let l = 0; l < 32; l++) {
          const packed = srcBytes[qPtr + l];
          const hm = srcBytes[qhOff + l];
          const q0 = (packed & 0x0f) + ((hm & u1) !== 0 ? 16 : 0);
          const q1 = (packed >> 4) + ((hm & u2) !== 0 ? 16 : 0);
          dst[outPtr + l] = d1 * q0 - m1;
          dst[outPtr + 32 + l] = d2 * q1 - m2;
        }
        qPtr += 32;
        outPtr += 64;
        u1 <<= 2;
        u2 <<= 2;
      }
      ptr += 176;
    }
    return;
  }

  if (srcType === GgmlType.Q6_K) {
    // Q6_K super-block (256 elements = 210 bytes): ql(128), qh(64), scales(16 int8), d(2 F16).
    const numBlocks = cols / 256;
    let ptr = srcByteOffset;
    for (let b = 0; b < numBlocks; b++) {
      const qlOff = ptr;
      const qhOff = ptr + 128;
      const scOff = ptr + 192;
      const d = F16_LUT[srcBytes[ptr + 208] | (srcBytes[ptr + 209] << 8)];
      const outBase = dstOffset + b * 256;

      for (let half = 0; half < 2; half++) {
        const n = half * 128;
        const qlBase = qlOff + half * 64;
        const qhBase = qhOff + half * 32;
        const scBase = scOff + half * 8;

        for (let l = 0; l < 32; l++) {
          const is = l >> 4;
          const ql0 = srcBytes[qlBase + l];
          const ql1 = srcBytes[qlBase + l + 32];
          const qh = srcBytes[qhBase + l];

          const q1 = ((ql0 & 0x0f) | (((qh >> 0) & 3) << 4)) - 32;
          const q2 = ((ql1 & 0x0f) | (((qh >> 2) & 3) << 4)) - 32;
          const q3 = ((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32;
          const q4 = ((ql1 >> 4) | (((qh >> 6) & 3) << 4)) - 32;

          const s0 = (srcBytes[scBase + is + 0] << 24) >> 24;
          const s2 = (srcBytes[scBase + is + 2] << 24) >> 24;
          const s4 = (srcBytes[scBase + is + 4] << 24) >> 24;
          const s6 = (srcBytes[scBase + is + 6] << 24) >> 24;

          dst[outBase + n + l] = d * s0 * q1;
          dst[outBase + n + l + 32] = d * s2 * q2;
          dst[outBase + n + l + 64] = d * s4 * q3;
          dst[outBase + n + l + 96] = d * s6 * q4;
        }
      }
      ptr += 210;
    }
    return;
  }

  if (srcType === GgmlType.Q2_K) {
    // Q2_K super-block (256 elements = 84 bytes): scales(16), qs(64), d(2 F16), dmin(2 F16).
    const numBlocks = cols / 256;
    let ptr = srcByteOffset;
    for (let b = 0; b < numBlocks; b++) {
      const scOff = ptr;
      const qsOff = ptr + 16;
      const d = F16_LUT[srcBytes[ptr + 80] | (srcBytes[ptr + 81] << 8)];
      const dmin = F16_LUT[srcBytes[ptr + 82] | (srcBytes[ptr + 83] << 8)];
      let outPtr = dstOffset + b * 256;
      let is = 0;

      for (let half = 0; half < 2; half++) {
        const qBase = qsOff + half * 32;
        for (let shift = 0; shift < 8; shift += 2) {
          const sc0 = srcBytes[scOff + is++];
          const dl0 = d * (sc0 & 0x0f);
          const ml0 = dmin * (sc0 >> 4);
          for (let l = 0; l < 16; l++) {
            dst[outPtr++] = dl0 * ((srcBytes[qBase + l] >> shift) & 3) - ml0;
          }
          const sc1 = srcBytes[scOff + is++];
          const dl1 = d * (sc1 & 0x0f);
          const ml1 = dmin * (sc1 >> 4);
          for (let l = 0; l < 16; l++) {
            dst[outPtr++] = dl1 * ((srcBytes[qBase + 16 + l] >> shift) & 3) - ml1;
          }
        }
      }
      ptr += 84;
    }
    return;
  }

  if (srcType === GgmlType.Q3_K) {
    // Q3_K super-block (256 elements = 110 bytes): hmask(32), qs(64), scales(12), d(2 F16).
    const numBlocks = cols / 256;
    let ptr = srcByteOffset;
    for (let b = 0; b < numBlocks; b++) {
      const hmOff = ptr;
      const qsOff = ptr + 32;
      const scales = unpackQ3KScales(srcBytes, ptr + 96);
      const d = F16_LUT[srcBytes[ptr + 108] | (srcBytes[ptr + 109] << 8)];
      let outPtr = dstOffset + b * 256;
      let is = 0;
      let m = 1;

      for (let half = 0; half < 2; half++) {
        const qBase = qsOff + half * 32;
        for (let shift = 0; shift < 8; shift += 2) {
          const dl0 = d * scales[is++];
          for (let l = 0; l < 16; l++) {
            const q = ((srcBytes[qBase + l] >> shift) & 3) - ((srcBytes[hmOff + l] & m) !== 0 ? 0 : 4);
            dst[outPtr++] = dl0 * q;
          }
          const dl1 = d * scales[is++];
          for (let l = 0; l < 16; l++) {
            const q =
              ((srcBytes[qBase + 16 + l] >> shift) & 3) - ((srcBytes[hmOff + 16 + l] & m) !== 0 ? 0 : 4);
            dst[outPtr++] = dl1 * q;
          }
          m <<= 1;
        }
      }
      ptr += 110;
    }
    return;
  }

  throw new Error(`Verdant decoder does not yet support GGML storage type id ${srcType}.`);
}

/**
 * Verdant RMS_NORM with F64 mean-of-squares accumulation before F32 narrowing (VT-45 / VT-46).
 * Computes: out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i] (if weight provided).
 */
export function verdantRmsNorm(
  x: Float32Array,
  weight: Float32Array | null,
  out: Float32Array,
  n: number,
  eps: number,
): void {
  let sumSq = 0.0; // IEEE 754 F64 accumulator
  for (let i = 0; i < n; i++) {
    const v = x[i];
    sumSq += v * v;
  }
  const mean = sumSq / n;
  const scale = Math.fround(1.0 / Math.sqrt(mean + eps));
  if (weight) {
    for (let i = 0; i < n; i++) {
      out[i] = Math.fround(Math.fround(x[i] * scale) * weight[i]);
    }
  } else {
    for (let i = 0; i < n; i++) {
      out[i] = Math.fround(x[i] * scale);
    }
  }
}

/**
 * Per-head RMS_NORM applied in-place across `[nHeads * headDim]` vectors (used by Qwen3 QK-Norm).
 */
export function verdantRmsNormPerHead(
  vec: Float32Array,
  weight: Float32Array,
  nHeads: number,
  headDim: number,
  eps: number,
  vecOffset = 0,
): void {
  for (let h = 0; h < nHeads; h++) {
    const base = vecOffset + h * headDim;
    let sumSq = 0.0;
    for (let i = 0; i < headDim; i++) {
      const v = vec[base + i];
      sumSq += v * v;
    }
    const scale = Math.fround(1.0 / Math.sqrt(sumSq / headDim + eps));
    for (let i = 0; i < headDim; i++) {
      vec[base + i] = Math.fround(Math.fround(vec[base + i] * scale) * weight[i]);
    }
  }
}

/**
 * Verdant MUL_MAT row product: out[r] = dot(W[r, 0..cols), x[0..cols)) for r in 0..rows.
 * Supports F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, and Q6_K weight matrices
 * with F32 activations and F64 dot-product accumulation.
 */
export function verdantMulMat(
  weightBytes: Uint8Array,
  weightByteOffset: number,
  weightType: GgmlType,
  cols: number,
  rows: number,
  x: Float32Array,
  out: Float32Array,
): void {
  if (weightType === GgmlType.F32) {
    const baseOffset = weightBytes.byteOffset + weightByteOffset;
    if ((baseOffset & 3) === 0) {
      const wF32 = new Float32Array(weightBytes.buffer, baseOffset, rows * cols);
      for (let r = 0; r < rows; r++) {
        let acc = 0.0;
        const rowStart = r * cols;
        for (let c = 0; c < cols; c++) {
          acc += wF32[rowStart + c] * x[c];
        }
        out[r] = Math.fround(acc);
      }
      return;
    }
    const view = new DataView(weightBytes.buffer, baseOffset, rows * cols * 4);
    for (let r = 0; r < rows; r++) {
      let acc = 0.0;
      const rowByteStart = r * cols * 4;
      for (let c = 0; c < cols; c++) {
        acc += view.getFloat32(rowByteStart + c * 4, true) * x[c];
      }
      out[r] = Math.fround(acc);
    }
    return;
  }

  if (weightType === GgmlType.F16) {
    const baseOffset = weightBytes.byteOffset + weightByteOffset;
    if ((baseOffset & 1) === 0) {
      const u16 = new Uint16Array(weightBytes.buffer, baseOffset, rows * cols);
      for (let r = 0; r < rows; r++) {
        let acc = 0.0;
        const rowStart = r * cols;
        for (let c = 0; c < cols; c++) {
          acc += F16_LUT[u16[rowStart + c]] * x[c];
        }
        out[r] = Math.fround(acc);
      }
      return;
    }
    const view = new DataView(weightBytes.buffer, baseOffset, rows * cols * 2);
    for (let r = 0; r < rows; r++) {
      let acc = 0.0;
      const rowByteStart = r * cols * 2;
      for (let c = 0; c < cols; c++) {
        acc += F16_LUT[view.getUint16(rowByteStart + c * 2, true)] * x[c];
      }
      out[r] = Math.fround(acc);
    }
    return;
  }

  if (weightType === GgmlType.BF16) {
    const baseOffset = weightBytes.byteOffset + weightByteOffset;
    if ((baseOffset & 1) === 0) {
      const u16 = new Uint16Array(weightBytes.buffer, baseOffset, rows * cols);
      for (let r = 0; r < rows; r++) {
        let acc = 0.0;
        const rowStart = r * cols;
        for (let c = 0; c < cols; c++) {
          acc += BF16_LUT[u16[rowStart + c]] * x[c];
        }
        out[r] = Math.fround(acc);
      }
      return;
    }
    const view = new DataView(weightBytes.buffer, baseOffset, rows * cols * 2);
    for (let r = 0; r < rows; r++) {
      let acc = 0.0;
      const rowByteStart = r * cols * 2;
      for (let c = 0; c < cols; c++) {
        acc += BF16_LUT[view.getUint16(rowByteStart + c * 2, true)] * x[c];
      }
      out[r] = Math.fround(acc);
    }
    return;
  }

  if (weightType === GgmlType.Q8_0) {
    const numBlocks = cols / 32;
    const rowBytes = numBlocks * 34;
    const signedBytes = new Int8Array(weightBytes.buffer, weightBytes.byteOffset, weightBytes.byteLength);
    for (let r = 0; r < rows; r++) {
      let acc = 0.0;
      let ptr = weightByteOffset + r * rowBytes;
      for (let b = 0; b < numBlocks; b++) {
        const scale = F16_LUT[weightBytes[ptr] | (weightBytes[ptr + 1] << 8)];
        ptr += 2;
        let blockSum = 0.0;
        const xBase = b * 32;
        for (let j = 0; j < 32; j += 4) {
          blockSum +=
            signedBytes[ptr + j] * x[xBase + j] +
            signedBytes[ptr + j + 1] * x[xBase + j + 1] +
            signedBytes[ptr + j + 2] * x[xBase + j + 2] +
            signedBytes[ptr + j + 3] * x[xBase + j + 3];
        }
        acc += blockSum * scale;
        ptr += 32;
      }
      out[r] = Math.fround(acc);
    }
    return;
  }

  if (weightType === GgmlType.Q4_0) {
    const numBlocks = cols / 32;
    const rowBytes = numBlocks * 18;
    for (let r = 0; r < rows; r++) {
      let acc = 0.0;
      let ptr = weightByteOffset + r * rowBytes;
      for (let b = 0; b < numBlocks; b++) {
        const scale = F16_LUT[weightBytes[ptr] | (weightBytes[ptr + 1] << 8)];
        ptr += 2;
        let blockSum = 0.0;
        const xBase = b * 32;
        const xBaseHi = xBase + 16;
        for (let j = 0; j < 16; j += 4) {
          const p0 = weightBytes[ptr + j];
          const p1 = weightBytes[ptr + j + 1];
          const p2 = weightBytes[ptr + j + 2];
          const p3 = weightBytes[ptr + j + 3];
          blockSum +=
            ((p0 & 0x0f) - 8) * x[xBase + j] +
            ((p0 >> 4) - 8) * x[xBaseHi + j] +
            ((p1 & 0x0f) - 8) * x[xBase + j + 1] +
            ((p1 >> 4) - 8) * x[xBaseHi + j + 1] +
            ((p2 & 0x0f) - 8) * x[xBase + j + 2] +
            ((p2 >> 4) - 8) * x[xBaseHi + j + 2] +
            ((p3 & 0x0f) - 8) * x[xBase + j + 3] +
            ((p3 >> 4) - 8) * x[xBaseHi + j + 3];
        }
        acc += blockSum * scale;
        ptr += 16;
      }
      out[r] = Math.fround(acc);
    }
    return;
  }

  if (weightType === GgmlType.Q4_1) {
    const numBlocks = cols / 32;
    const rowBytes = numBlocks * 20;
    for (let r = 0; r < rows; r++) {
      let acc = 0.0;
      let ptr = weightByteOffset + r * rowBytes;
      for (let b = 0; b < numBlocks; b++) {
        const d = F16_LUT[weightBytes[ptr] | (weightBytes[ptr + 1] << 8)];
        const m = F16_LUT[weightBytes[ptr + 2] | (weightBytes[ptr + 3] << 8)];
        ptr += 4;
        let sumQ = 0.0;
        let sumX = 0.0;
        const xBase = b * 32;
        const xBaseHi = xBase + 16;
        for (let j = 0; j < 16; j++) {
          const packed = weightBytes[ptr + j];
          const xl = x[xBase + j];
          const xh = x[xBaseHi + j];
          sumQ += (packed & 0x0f) * xl + (packed >> 4) * xh;
          sumX += xl + xh;
        }
        acc += d * sumQ + m * sumX;
        ptr += 16;
      }
      out[r] = Math.fround(acc);
    }
    return;
  }

  if (weightType === GgmlType.Q5_0) {
    const numBlocks = cols / 32;
    const rowBytes = numBlocks * 22;
    for (let r = 0; r < rows; r++) {
      let acc = 0.0;
      let ptr = weightByteOffset + r * rowBytes;
      for (let b = 0; b < numBlocks; b++) {
        const d = F16_LUT[weightBytes[ptr] | (weightBytes[ptr + 1] << 8)];
        const qh =
          (weightBytes[ptr + 2] |
            (weightBytes[ptr + 3] << 8) |
            (weightBytes[ptr + 4] << 16) |
            (weightBytes[ptr + 5] << 24)) >>>
          0;
        ptr += 6;
        let blockSum = 0.0;
        const xBase = b * 32;
        const xBaseHi = xBase + 16;
        for (let j = 0; j < 16; j++) {
          const packed = weightBytes[ptr + j];
          const xh0 = ((qh >>> j) & 1) << 4;
          const xh1 = ((qh >>> (j + 16)) & 1) << 4;
          blockSum +=
            (((packed & 0x0f) | xh0) - 16) * x[xBase + j] +
            (((packed >> 4) | xh1) - 16) * x[xBaseHi + j];
        }
        acc += blockSum * d;
        ptr += 16;
      }
      out[r] = Math.fround(acc);
    }
    return;
  }

  if (weightType === GgmlType.Q5_1) {
    const numBlocks = cols / 32;
    const rowBytes = numBlocks * 24;
    for (let r = 0; r < rows; r++) {
      let acc = 0.0;
      let ptr = weightByteOffset + r * rowBytes;
      for (let b = 0; b < numBlocks; b++) {
        const d = F16_LUT[weightBytes[ptr] | (weightBytes[ptr + 1] << 8)];
        const m = F16_LUT[weightBytes[ptr + 2] | (weightBytes[ptr + 3] << 8)];
        const qh =
          (weightBytes[ptr + 4] |
            (weightBytes[ptr + 5] << 8) |
            (weightBytes[ptr + 6] << 16) |
            (weightBytes[ptr + 7] << 24)) >>>
          0;
        ptr += 8;
        let sumQ = 0.0;
        let sumX = 0.0;
        const xBase = b * 32;
        const xBaseHi = xBase + 16;
        for (let j = 0; j < 16; j++) {
          const packed = weightBytes[ptr + j];
          const xh0 = ((qh >>> j) & 1) << 4;
          const xh1 = ((qh >>> (j + 16)) & 1) << 4;
          const xl = x[xBase + j];
          const xh = x[xBaseHi + j];
          sumQ += ((packed & 0x0f) | xh0) * xl + ((packed >> 4) | xh1) * xh;
          sumX += xl + xh;
        }
        acc += d * sumQ + m * sumX;
        ptr += 16;
      }
      out[r] = Math.fround(acc);
    }
    return;
  }

  if (weightType === GgmlType.Q4_K) {
    const numBlocks = cols / 256;
    const rowBytes = numBlocks * 144;
    for (let r = 0; r < rows; r++) {
      let acc = 0.0;
      let ptr = weightByteOffset + r * rowBytes;
      for (let b = 0; b < numBlocks; b++) {
        const d = F16_LUT[weightBytes[ptr] | (weightBytes[ptr + 1] << 8)];
        const dmin = F16_LUT[weightBytes[ptr + 2] | (weightBytes[ptr + 3] << 8)];
        const scalesOff = ptr + 4;
        let qPtr = ptr + 16;
        let xPtr = b * 256;

        for (let is = 0; is < 8; is += 2) {
          const sm0 = getScaleMinK4Packed(is, weightBytes, scalesOff);
          const d1 = d * (sm0 >> 8);
          const m1 = dmin * (sm0 & 0xff);
          const sm1 = getScaleMinK4Packed(is + 1, weightBytes, scalesOff);
          const d2 = d * (sm1 >> 8);
          const m2 = dmin * (sm1 & 0xff);

          let sumQ1 = 0.0;
          let sumX1 = 0.0;
          let sumQ2 = 0.0;
          let sumX2 = 0.0;
          for (let l = 0; l < 32; l++) {
            const packed = weightBytes[qPtr + l];
            const x1 = x[xPtr + l];
            const x2 = x[xPtr + 32 + l];
            sumQ1 += (packed & 0x0f) * x1;
            sumX1 += x1;
            sumQ2 += (packed >> 4) * x2;
            sumX2 += x2;
          }
          acc += d1 * sumQ1 - m1 * sumX1 + d2 * sumQ2 - m2 * sumX2;
          qPtr += 32;
          xPtr += 64;
        }
        ptr += 144;
      }
      out[r] = Math.fround(acc);
    }
    return;
  }

  if (weightType === GgmlType.Q5_K) {
    const numBlocks = cols / 256;
    const rowBytes = numBlocks * 176;
    for (let r = 0; r < rows; r++) {
      let acc = 0.0;
      let ptr = weightByteOffset + r * rowBytes;
      for (let b = 0; b < numBlocks; b++) {
        const d = F16_LUT[weightBytes[ptr] | (weightBytes[ptr + 1] << 8)];
        const dmin = F16_LUT[weightBytes[ptr + 2] | (weightBytes[ptr + 3] << 8)];
        const scalesOff = ptr + 4;
        const qhOff = ptr + 16;
        let qPtr = ptr + 48;
        let xPtr = b * 256;

        let u1 = 1;
        let u2 = 2;
        for (let is = 0; is < 8; is += 2) {
          const sm0 = getScaleMinK4Packed(is, weightBytes, scalesOff);
          const d1 = d * (sm0 >> 8);
          const m1 = dmin * (sm0 & 0xff);
          const sm1 = getScaleMinK4Packed(is + 1, weightBytes, scalesOff);
          const d2 = d * (sm1 >> 8);
          const m2 = dmin * (sm1 & 0xff);

          let sumQ1 = 0.0;
          let sumX1 = 0.0;
          let sumQ2 = 0.0;
          let sumX2 = 0.0;
          for (let l = 0; l < 32; l++) {
            const packed = weightBytes[qPtr + l];
            const hm = weightBytes[qhOff + l];
            const q0 = (packed & 0x0f) + ((hm & u1) !== 0 ? 16 : 0);
            const q1 = (packed >> 4) + ((hm & u2) !== 0 ? 16 : 0);
            const x1 = x[xPtr + l];
            const x2 = x[xPtr + 32 + l];
            sumQ1 += q0 * x1;
            sumX1 += x1;
            sumQ2 += q1 * x2;
            sumX2 += x2;
          }
          acc += d1 * sumQ1 - m1 * sumX1 + d2 * sumQ2 - m2 * sumX2;
          qPtr += 32;
          xPtr += 64;
          u1 <<= 2;
          u2 <<= 2;
        }
        ptr += 176;
      }
      out[r] = Math.fround(acc);
    }
    return;
  }

  if (weightType === GgmlType.Q6_K) {
    const numBlocks = cols / 256;
    const rowBytes = numBlocks * 210;
    const signedBytes = new Int8Array(weightBytes.buffer, weightBytes.byteOffset, weightBytes.byteLength);
    for (let r = 0; r < rows; r++) {
      let acc = 0.0;
      let ptr = weightByteOffset + r * rowBytes;
      for (let b = 0; b < numBlocks; b++) {
        const qlOff = ptr;
        const qhOff = ptr + 128;
        const scOff = ptr + 192;
        const d = F16_LUT[weightBytes[ptr + 208] | (weightBytes[ptr + 209] << 8)];
        const xBase = b * 256;
        let blockSum = 0.0;

        for (let half = 0; half < 2; half++) {
          const n = xBase + half * 128;
          const qlBase = qlOff + half * 64;
          const qhBase = qhOff + half * 32;
          const scBase = scOff + half * 8;

          for (let l = 0; l < 32; l++) {
            const is = l >> 4;
            const ql0 = weightBytes[qlBase + l];
            const ql1 = weightBytes[qlBase + l + 32];
            const qh = weightBytes[qhBase + l];

            const q1 = ((ql0 & 0x0f) | (((qh >> 0) & 3) << 4)) - 32;
            const q2 = ((ql1 & 0x0f) | (((qh >> 2) & 3) << 4)) - 32;
            const q3 = ((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32;
            const q4 = ((ql1 >> 4) | (((qh >> 6) & 3) << 4)) - 32;

            blockSum +=
              signedBytes[scBase + is + 0] * q1 * x[n + l] +
              signedBytes[scBase + is + 2] * q2 * x[n + l + 32] +
              signedBytes[scBase + is + 4] * q3 * x[n + l + 64] +
              signedBytes[scBase + is + 6] * q4 * x[n + l + 96];
          }
        }
        acc += d * blockSum;
        ptr += 210;
      }
      out[r] = Math.fround(acc);
    }
    return;
  }

  if (weightType === GgmlType.Q2_K) {
    const numBlocks = cols / 256;
    const rowBytes = numBlocks * 84;
    for (let r = 0; r < rows; r++) {
      let acc = 0.0;
      let ptr = weightByteOffset + r * rowBytes;
      for (let b = 0; b < numBlocks; b++) {
        const scOff = ptr;
        const qsOff = ptr + 16;
        const d = F16_LUT[weightBytes[ptr + 80] | (weightBytes[ptr + 81] << 8)];
        const dmin = F16_LUT[weightBytes[ptr + 82] | (weightBytes[ptr + 83] << 8)];
        let xPtr = b * 256;
        let is = 0;

        for (let half = 0; half < 2; half++) {
          const qBase = qsOff + half * 32;
          for (let shift = 0; shift < 8; shift += 2) {
            const sc0 = weightBytes[scOff + is++];
            const dl0 = d * (sc0 & 0x0f);
            const ml0 = dmin * (sc0 >> 4);
            let sumQ0 = 0.0;
            let sumX0 = 0.0;
            for (let l = 0; l < 16; l++) {
              const xv = x[xPtr + l];
              sumQ0 += ((weightBytes[qBase + l] >> shift) & 3) * xv;
              sumX0 += xv;
            }
            acc += dl0 * sumQ0 - ml0 * sumX0;
            xPtr += 16;

            const sc1 = weightBytes[scOff + is++];
            const dl1 = d * (sc1 & 0x0f);
            const ml1 = dmin * (sc1 >> 4);
            let sumQ1 = 0.0;
            let sumX1 = 0.0;
            for (let l = 0; l < 16; l++) {
              const xv = x[xPtr + l];
              sumQ1 += ((weightBytes[qBase + 16 + l] >> shift) & 3) * xv;
              sumX1 += xv;
            }
            acc += dl1 * sumQ1 - ml1 * sumX1;
            xPtr += 16;
          }
        }
        ptr += 84;
      }
      out[r] = Math.fround(acc);
    }
    return;
  }

  if (weightType === GgmlType.Q3_K) {
    const numBlocks = cols / 256;
    const rowBytes = numBlocks * 110;
    for (let r = 0; r < rows; r++) {
      let acc = 0.0;
      let ptr = weightByteOffset + r * rowBytes;
      for (let b = 0; b < numBlocks; b++) {
        const hmOff = ptr;
        const qsOff = ptr + 32;
        const scales = unpackQ3KScales(weightBytes, ptr + 96);
        const d = F16_LUT[weightBytes[ptr + 108] | (weightBytes[ptr + 109] << 8)];
        let xPtr = b * 256;
        let is = 0;
        let m = 1;
        let blockSum = 0.0;

        for (let half = 0; half < 2; half++) {
          const qBase = qsOff + half * 32;
          for (let shift = 0; shift < 8; shift += 2) {
            const s0 = scales[is++];
            let sum0 = 0.0;
            for (let l = 0; l < 16; l++) {
              const q = ((weightBytes[qBase + l] >> shift) & 3) - ((weightBytes[hmOff + l] & m) !== 0 ? 0 : 4);
              sum0 += q * x[xPtr + l];
            }
            blockSum += s0 * sum0;
            xPtr += 16;

            const s1 = scales[is++];
            let sum1 = 0.0;
            for (let l = 0; l < 16; l++) {
              const q =
                ((weightBytes[qBase + 16 + l] >> shift) & 3) - ((weightBytes[hmOff + 16 + l] & m) !== 0 ? 0 : 4);
              sum1 += q * x[xPtr + l];
            }
            blockSum += s1 * sum1;
            xPtr += 16;
            m <<= 1;
          }
        }
        acc += d * blockSum;
        ptr += 110;
      }
      out[r] = Math.fround(acc);
    }
    return;
  }

  throw new Error(`Verdant MUL_MAT does not support GGML storage type id ${weightType}.`);
}

/**
 * Verdant ROPE rotary position embedding (normal interleaved pairs or NeoX split-half, VT-57).
 * Applies plane rotations in-place to multi-head vector `vec` of shape [nHeads * headDim].
 */
export function verdantRope(
  vec: Float32Array,
  nHeads: number,
  headDim: number,
  ropeDim: number,
  pos: number,
  freqBase: number,
  neox = false,
  freqScale = 1.0,
): void {
  const half = Math.floor(ropeDim / 2);
  const safeScale = Number.isFinite(freqScale) && freqScale > 0 ? freqScale : 1.0;
  for (let h = 0; h < nHeads; h++) {
    const base = h * headDim;
    for (let i = 0; i < half; i++) {
      const freq = safeScale / Math.pow(freqBase, (2 * i) / ropeDim);
      const angle = pos * freq;
      const cos = Math.fround(Math.cos(angle));
      const sin = Math.fround(Math.sin(angle));
      if (neox) {
        const x0 = vec[base + i];
        const x1 = vec[base + i + half];
        vec[base + i] = Math.fround(x0 * cos - x1 * sin);
        vec[base + i + half] = Math.fround(x0 * sin + x1 * cos);
      } else {
        const idx = base + 2 * i;
        const x0 = vec[idx];
        const x1 = vec[idx + 1];
        vec[idx] = Math.fround(x0 * cos - x1 * sin);
        vec[idx + 1] = Math.fround(x0 * sin + x1 * cos);
      }
    }
  }
}

export interface RopeTables {
  readonly maxCtx: number;
  readonly half: number;
  readonly freqs: Float64Array;
  readonly cos: Float32Array;
  readonly sin: Float32Array;
}

/**
 * Precomputes RoPE cos/sin tables up to `maxCtx` positions (`[maxCtx * half]` floats),
 * while retaining exact F64 inverse frequencies (`freqs`) for bit-identical on-demand
 * synthesis at arbitrary positions `pos >= maxCtx` (e.g. 8K–131K context windows).
 */
export function buildRopeTables(
  maxCtx: number,
  ropeDim: number,
  freqBase: number,
  freqScale = 1.0,
): RopeTables {
  const half = Math.floor(ropeDim / 2);
  const cos = new Float32Array(maxCtx * half);
  const sin = new Float32Array(maxCtx * half);
  const freqs = new Float64Array(half);
  const safeScale = Number.isFinite(freqScale) && freqScale > 0 ? freqScale : 1.0;
  for (let i = 0; i < half; i++) {
    freqs[i] = safeScale / Math.pow(freqBase, (2 * i) / ropeDim);
  }
  for (let pos = 0; pos < maxCtx; pos++) {
    const rowOff = pos * half;
    for (let i = 0; i < half; i++) {
      const angle = pos * freqs[i];
      cos[rowOff + i] = Math.cos(angle);
      sin[rowOff + i] = Math.sin(angle);
    }
  }
  return { maxCtx, half, freqs, cos, sin };
}

/**
 * Computes cos/sin for a single arbitrary RoPE position `pos` into destination F32 views,
 * bit-identical to `buildRopeTables`. Used by WASM dynamic RoPE scratch slots for `pos >= maxCtx`.
 */
export function fillRopePositionCosSin(
  pos: number,
  tables: RopeTables,
  cosDst: Float32Array,
  sinDst: Float32Array,
  dstOffset = 0,
): void {
  const half = tables.half;
  const freqs = tables.freqs;
  for (let i = 0; i < half; i++) {
    const angle = pos * freqs[i];
    cosDst[dstOffset + i] = Math.cos(angle);
    sinDst[dstOffset + i] = Math.sin(angle);
  }
}

/**
 * Table-backed Verdant ROPE rotary position embedding (normal interleaved or NeoX split-half).
 * Automatically synthesizes bit-exact cos/sin on the fly for `pos >= tables.maxCtx`.
 */
export function verdantRopePrecomputed(
  vec: Float32Array,
  nHeads: number,
  headDim: number,
  pos: number,
  tables: RopeTables,
  neox = false,
): void {
  const half = tables.half;
  if (pos >= tables.maxCtx || pos < 0) {
    const freqs = tables.freqs;
    if (neox) {
      for (let h = 0; h < nHeads; h++) {
        const base = h * headDim;
        for (let i = 0; i < half; i++) {
          const angle = pos * freqs[i];
          const c = Math.fround(Math.cos(angle));
          const s = Math.fround(Math.sin(angle));
          const x0 = vec[base + i];
          const x1 = vec[base + i + half];
          vec[base + i] = Math.fround(x0 * c - x1 * s);
          vec[base + i + half] = Math.fround(x0 * s + x1 * c);
        }
      }
    } else {
      for (let h = 0; h < nHeads; h++) {
        const base = h * headDim;
        for (let i = 0; i < half; i++) {
          const angle = pos * freqs[i];
          const c = Math.fround(Math.cos(angle));
          const s = Math.fround(Math.sin(angle));
          const idx = base + (i << 1);
          const x0 = vec[idx];
          const x1 = vec[idx + 1];
          vec[idx] = Math.fround(x0 * c - x1 * s);
          vec[idx + 1] = Math.fround(x0 * s + x1 * c);
        }
      }
    }
    return;
  }

  const rowOff = pos * half;
  const cosTable = tables.cos;
  const sinTable = tables.sin;

  if (neox) {
    for (let h = 0; h < nHeads; h++) {
      const base = h * headDim;
      for (let i = 0; i < half; i++) {
        const c = cosTable[rowOff + i];
        const s = sinTable[rowOff + i];
        const x0 = vec[base + i];
        const x1 = vec[base + i + half];
        vec[base + i] = Math.fround(x0 * c - x1 * s);
        vec[base + i + half] = Math.fround(x0 * s + x1 * c);
      }
    }
  } else {
    for (let h = 0; h < nHeads; h++) {
      const base = h * headDim;
      for (let i = 0; i < half; i++) {
        const c = cosTable[rowOff + i];
        const s = sinTable[rowOff + i];
        const idx = base + (i << 1);
        const x0 = vec[idx];
        const x1 = vec[idx + 1];
        vec[idx] = Math.fround(x0 * c - x1 * s);
        vec[idx + 1] = Math.fround(x0 * s + x1 * c);
      }
    }
  }
}

/**
 * Verdant SWIGLU feed-forward gate: out[i] = silu(gate[i]) * up[i] (VT-56).
 */
export function verdantSwiGlu(
  gate: Float32Array,
  up: Float32Array,
  out: Float32Array,
  n: number,
): void {
  for (let i = 0; i < n; i++) {
    const g = gate[i];
    const silu = g / (1.0 + Math.exp(-g));
    out[i] = Math.fround(silu * up[i]);
  }
}

/**
 * Verdant FLASH_ATTN_EXT causal multi-head / grouped-query attention using streaming online softmax (VT-58).
 */
export function verdantFlashAttnExt(
  q: Float32Array,
  kCache: Float32Array,
  vCache: Float32Array,
  out: Float32Array,
  nHeads: number,
  nHeadsKv: number,
  headDim: number,
  seqLen: number,
  kvLayerOffset: number,
): void {
  const kvStride = nHeadsKv * headDim;
  const headsPerKv = Math.floor(nHeads / nHeadsKv);
  const invScale = 1.0 / Math.sqrt(headDim);

  for (let h = 0; h < nHeads; h++) {
    const kvHead = Math.floor(h / headsPerKv);
    const qBase = h * headDim;
    const outBase = h * headDim;

    // Streaming online softmax accumulator per head (matches Verdant streaming weighted mean)
    let runningMax = -Infinity;
    let runningDenom = 0.0;
    for (let d = 0; d < headDim; d++) {
      out[outBase + d] = 0.0;
    }

    for (let t = 0; t < seqLen; t++) {
      const kBase = kvLayerOffset + t * kvStride + kvHead * headDim;
      let dot = 0.0;
      for (let d = 0; d < headDim; d++) {
        dot += q[qBase + d] * kCache[kBase + d];
      }
      const score = dot * invScale;

      if (score > runningMax) {
        const rescale = runningMax === -Infinity ? 0.0 : Math.exp(runningMax - score);
        runningDenom = runningDenom * rescale + 1.0;
        for (let d = 0; d < headDim; d++) {
          out[outBase + d] = out[outBase + d] * rescale + vCache[kBase + d];
        }
        runningMax = score;
      } else {
        const w = Math.exp(score - runningMax);
        runningDenom += w;
        for (let d = 0; d < headDim; d++) {
          out[outBase + d] += w * vCache[kBase + d];
        }
      }
    }

    const invDenom = runningDenom > 0 ? 1.0 / runningDenom : 0.0;
    for (let d = 0; d < headDim; d++) {
      out[outBase + d] = Math.fround(out[outBase + d] * invDenom);
    }
  }
}

/** Elementwise F32 residual addition: out[i] = a[i] + b[i]. */
export function verdantAdd(a: Float32Array, b: Float32Array, out: Float32Array, n: number): void {
  for (let i = 0; i < n; i++) {
    out[i] = Math.fround(a[i] + b[i]);
  }
}

/** Elementwise F32 multiplication: out[i] = a[i] * b[i]. */
export function verdantMul(a: Float32Array, b: Float32Array, out: Float32Array, n: number): void {
  for (let i = 0; i < n; i++) {
    out[i] = Math.fround(a[i] * b[i]);
  }
}

/** Verdant F64-accumulated sum reduction. */
export function verdantSum(x: Float32Array, n: number): number {
  let acc = 0.0;
  for (let i = 0; i < n; i++) {
    acc += x[i];
  }
  return Math.fround(acc);
}

/** Verdant F64-accumulated mean reduction (division before F32 narrowing). */
export function verdantMean(x: Float32Array, n: number): number {
  if (n === 0) return 0;
  let acc = 0.0;
  for (let i = 0; i < n; i++) {
    acc += x[i];
  }
  return Math.fround(acc / n);
}
