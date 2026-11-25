/**
 * Verdant Tensor WebAssembly 128-bit SIMD Kernel Accelerator.
 *
 * Extrapolated from github.com/afflom/vTensor (Verdant numerical substrate + zero-copy linear arena)
 * and github.com/afflom/vRAM (word-kernel acceleration + 256-entry lookup tables):
 * - Compiles a self-contained, deterministic WebAssembly SIMD128 (v128) module at startup with zero
 *   external binary dependencies.
 * - Owns a unified growable WebAssembly.Memory linear arena so VTensorBuffer weights, KV cache,
 *   vRAM cold compressed pages, and scratch tensors execute in-place with zero JS-to-WASM copies.
 * - Implements 128-bit SIMD MUL_MAT across all 11 GGML formats (F32, F16, BF16, Q8_0, Q4_0, Q4_1,
 *   Q5_0, Q5_1, Q4_K, Q5_K, Q6_K) with IEEE 754 F64 row/block accumulation, algebraic blockSumX
 *   factoring, 2-row activation register tiling, and batched multi-token prefill kernels.
 * - Implements SIMD RMS_NORM (F64 sum-of-squares, VT-46) and SIMD dot-product helpers for both hot
 *   F32 and vRAM cold Q8 KV-cache attention.
 */

import { GgmlType } from './gguf';
import { f16ToF32 } from './verdantKernels';

/** Fixed low-memory reserved layout inside the WASM linear arena (512 KiB total). */
export const WASM_F16_LUT_OFFSET = 0; // 65536 * 4 = 262,144 bytes
export const WASM_BLOCK_SUM_X_OFFSET = 262_144; // 8192 * 8 = 65,536 bytes (up to 262,144 cols)
export const WASM_SCRATCH_BASE_OFFSET = 327_680; // 196,608 bytes reserved internal scratch
export const WASM_ATTN_SCORES_OFFSET = WASM_SCRATCH_BASE_OFFSET; // 64 bytes (16 x f32)
export const WASM_ATTN_WEIGHTS_OFFSET = WASM_SCRATCH_BASE_OFFSET + 64; // 64 bytes (16 x f32)
export const WASM_KV_QUANT_ERR_OFFSET = WASM_SCRATCH_BASE_OFFSET + 128; // 16 bytes (f32 maxErr, f32 maxSig)
export const WASM_TOPK_IDS_OFFSET = WASM_SCRATCH_BASE_OFFSET + 256; // up to 1024 x i32 (4 KiB)
export const WASM_TOPK_VALS_OFFSET = WASM_SCRATCH_BASE_OFFSET + 4352; // up to 1024 x f32 (4 KiB)
/** Per-worker 128-byte attention scratch (64B scores + 64B weights, 2 full cache lines) for up to 32 threads. */
export const WASM_ATTN_WORKER_SCRATCH_BASE = WASM_SCRATCH_BASE_OFFSET + 8448; // 128-byte aligned (4 KiB total)
export const WASM_ATTN_WORKER_SCRATCH_STRIDE = 128;
export const WASM_DYNAMIC_ARENA_START = 524_288; // 512 KiB aligned start for VTensorBuffers

/**
 * Packed Q8_0_ACT activation block layout (48 bytes per 32 floats, 16-byte aligned):
 *   +0:  f32 d_x      (maxAbs / 127)
 *   +4:  i32 sum_lo   (sum of q[0..15], used by Q6_K 16-element sub-groups)
 *   +8:  i32 sum_hi   (sum of q[16..31], used by Q6_K 16-element sub-groups)
 *   +12: i32 sum_all  (sum_lo + sum_hi, used by Q4_0/Q4_1/Q5_0/Q5_1/Q4_K/Q5_K)
 *   +16: i8[16] x_lo  (16-byte aligned signed int8 activations 0..15)
 *   +32: i8[16] x_hi  (16-byte aligned signed int8 activations 16..31)
 */
export const Q8_0_ACT_BLOCK_BYTES = 48;

/**
 * Packed Q8_K_ACT super-block activation layout (320 bytes per 256 floats, 16-byte aligned).
 * Implements `uor-matmul` narrow-lane accumulation ("encode to float ONCE per 256-elem super-block"):
 *   +0:   f32 d_x          (maxAbs / 127 across all 256 elements)
 *   +4:   12 bytes pad     (16-byte alignment for SIMD loads)
 *   +16:  i16[16] bsums16  (16 x i16 exact sums of each 16-elem slice, for Q6_K via 2x dot_i16x8)
 *   +48:  i16[8]  bsums32  (8  x i16 exact sums of each 32-elem sub-block, for Q4_K via 1x dot_i16x8)
 *   +64:  i8[256] qs       (16 x v128 contiguous signed int8 activations)
 */
export const Q8_K_ACT_BLOCK_BYTES = 320;

const WASM_PAGE_BYTES = 65_536;

// WASM value types
const I32 = 0x7f;
const I64 = 0x7e;
const F32 = 0x7d;
const F64 = 0x7c;
const V128 = 0x7b;
const VOID = 0x40;

// LEB128 encoders
function uleb(v: number): number[] {
  const out: number[] = [];
  let n = v >>> 0;
  do {
    let b = n & 0x7f;
    n >>>= 7;
    if (n !== 0) b |= 0x80;
    out.push(b);
  } while (n !== 0);
  return out;
}

function sleb(v: number): number[] {
  const out: number[] = [];
  let more = true;
  let n = v | 0;
  while (more) {
    let b = n & 0x7f;
    n >>= 7;
    if ((n === 0 && (b & 0x40) === 0) || (n === -1 && (b & 0x40) !== 0)) {
      more = false;
    } else {
      b |= 0x80;
    }
    out.push(b);
  }
  return out;
}

function f32Bytes(v: number): number[] {
  const buf = new ArrayBuffer(4);
  new DataView(buf).setFloat32(0, v, true);
  return Array.from(new Uint8Array(buf));
}

function f64Bytes(v: number): number[] {
  const buf = new ArrayBuffer(8);
  new DataView(buf).setFloat64(0, v, true);
  return Array.from(new Uint8Array(buf));
}

function utf8(s: string): number[] {
  const bytes = Array.from(new TextEncoder().encode(s));
  return [...uleb(bytes.length), ...bytes];
}

function section(id: number, body: number[]): number[] {
  return [id, ...uleb(body.length), ...body];
}

class FuncBuilder {
  readonly params: number[];
  readonly results: number[];
  private readonly localTypes: number[] = [];
  readonly code: number[] = [];

  constructor(params: number[], results: number[] = []) {
    this.params = params;
    this.results = results;
  }

  local(type: number): number {
    const idx = this.params.length + this.localTypes.length;
    this.localTypes.push(type);
    return idx;
  }

  emit(...bytes: number[]): this {
    for (let i = 0; i < bytes.length; i++) this.code.push(bytes[i]);
    return this;
  }

  // Control flow
  block(type = VOID): this { return this.emit(0x02, type); }
  loop(type = VOID): this { return this.emit(0x03, type); }
  if_(type = VOID): this { return this.emit(0x04, type); }
  else_(): this { return this.emit(0x05); }
  end(): this { return this.emit(0x0b); }
  br(depth: number): this { return this.emit(0x0c, ...uleb(depth)); }
  brIf(depth: number): this { return this.emit(0x0d, ...uleb(depth)); }
  ret(): this { return this.emit(0x0f); }
  call(fnIdx: number): this { return this.emit(0x10, ...uleb(fnIdx)); }
  drop(): this { return this.emit(0x1a); }

  // Locals
  get(idx: number): this { return this.emit(0x20, ...uleb(idx)); }
  set(idx: number): this { return this.emit(0x21, ...uleb(idx)); }
  tee(idx: number): this { return this.emit(0x22, ...uleb(idx)); }

  // Scalar memory
  i32Load(offset = 0, align = 2): this { return this.emit(0x28, align, ...uleb(offset)); }
  f32Load(offset = 0, align = 2): this { return this.emit(0x2a, align, ...uleb(offset)); }
  f64Load(offset = 0, align = 3): this { return this.emit(0x2b, align, ...uleb(offset)); }
  i32Load8S(offset = 0): this { return this.emit(0x2c, 0, ...uleb(offset)); }
  i32Load8U(offset = 0): this { return this.emit(0x2d, 0, ...uleb(offset)); }
  i32Load16U(offset = 0, align = 0): this { return this.emit(0x2f, align, ...uleb(offset)); }
  i32Store(offset = 0, align = 2): this { return this.emit(0x36, align, ...uleb(offset)); }
  f32Store(offset = 0, align = 2): this { return this.emit(0x38, align, ...uleb(offset)); }
  f64Store(offset = 0, align = 3): this { return this.emit(0x39, align, ...uleb(offset)); }

  // Constants
  i32Const(v: number): this { return this.emit(0x41, ...sleb(v)); }
  i64Const(v: number): this { return this.emit(0x42, ...sleb(v)); }
  f32Const(v: number): this { return this.emit(0x43, ...f32Bytes(v)); }
  f64Const(v: number): this { return this.emit(0x44, ...f64Bytes(v)); }

  // Bulk memory (0xfc prefix)
  memoryCopy(): this { return this.emit(0xfc, 0x0a, 0x00, 0x00); }
  memoryFill(): this { return this.emit(0xfc, 0x0b, 0x00); }

  // i32 arithmetic & comparison
  i32Eqz(): this { return this.emit(0x45); }
  i32Eq(): this { return this.emit(0x46); }
  i32Ne(): this { return this.emit(0x47); }
  i32LtS(): this { return this.emit(0x48); }
  i32LtU(): this { return this.emit(0x49); }
  i32GtS(): this { return this.emit(0x4a); }
  i32LeS(): this { return this.emit(0x4c); }
  i32GeS(): this { return this.emit(0x4e); }
  i32Add(): this { return this.emit(0x6a); }
  i32Sub(): this { return this.emit(0x6b); }
  i32Mul(): this { return this.emit(0x6c); }
  i32DivU(): this { return this.emit(0x6e); }
  i32And(): this { return this.emit(0x71); }
  i32Or(): this { return this.emit(0x72); }
  i32Shl(): this { return this.emit(0x74); }
  i32ShrS(): this { return this.emit(0x75); }
  i32ShrU(): this { return this.emit(0x76); }
  i64Or(): this { return this.emit(0x84); }
  i64Shl(): this { return this.emit(0x86); }
  i64ShrU(): this { return this.emit(0x88); }
  i32WrapI64(): this { return this.emit(0xa7); }
  i64ExtendI32U(): this { return this.emit(0xad); }

  // f32 & f64 arithmetic
  f32Eq(): this { return this.emit(0x5b); }
  f32Ne(): this { return this.emit(0x5c); }
  f32Gt(): this { return this.emit(0x5e); }
  f32Nearest(): this { return this.emit(0x90); }
  f32Sqrt(): this { return this.emit(0x91); }
  f32Add(): this { return this.emit(0x92); }
  f32Sub(): this { return this.emit(0x93); }
  f32Mul(): this { return this.emit(0x94); }
  f32Div(): this { return this.emit(0x95); }
  f32Min(): this { return this.emit(0x96); }
  f32Max(): this { return this.emit(0x97); }
  f64Sqrt(): this { return this.emit(0x9f); }
  f64Add(): this { return this.emit(0xa0); }
  f64Sub(): this { return this.emit(0xa1); }
  f64Mul(): this { return this.emit(0xa2); }
  f64Div(): this { return this.emit(0xa3); }
  i32TruncSatF32S(): this { return this.emit(0xfc, 0x00); }
  f32ConvertI32S(): this { return this.emit(0xb2); }
  f32DemoteF64(): this { return this.emit(0xb6); }
  f64ConvertI32S(): this { return this.emit(0xb7); }
  f64PromoteF32(): this { return this.emit(0xbb); }
  f32ReinterpretI32(): this { return this.emit(0xbe); }

  // SIMD 128-bit (0xfd prefix)
  v128Load(offset = 0, align = 0): this { return this.emit(0xfd, 0x00, align, ...uleb(offset)); }
  v128Load8x8S(offset = 0): this { return this.emit(0xfd, 0x01, 0, ...uleb(offset)); }
  v128Load16x4U(offset = 0): this { return this.emit(0xfd, 0x04, 0, ...uleb(offset)); }
  v128Load64Zero(offset = 0): this { return this.emit(0xfd, 0x5d, 0, ...uleb(offset)); }
  v128Store(offset = 0, align = 0): this { return this.emit(0xfd, 0x0b, align, ...uleb(offset)); }
  v128Const(bytes16: number[]): this { return this.emit(0xfd, 0x0c, ...bytes16); }
  i8x16Shuffle(lanes16: number[]): this { return this.emit(0xfd, 0x0d, ...lanes16); }
  i8x16Swizzle(): this { return this.emit(0xfd, 0x0e); }
  i8x16Splat(): this { return this.emit(0xfd, 0x0f); }
  i16x8Splat(): this { return this.emit(0xfd, 0x10); }
  i32x4Splat(): this { return this.emit(0xfd, 0x11); }
  f32x4Splat(): this { return this.emit(0xfd, 0x13); }
  i16x8ReplaceLane(lane: number): this { return this.emit(0xfd, 0x1a, lane); }
  i32x4Extract(lane: number): this { return this.emit(0xfd, 0x1b, lane); }
  i32x4ReplaceLane(lane: number): this { return this.emit(0xfd, 0x1c, lane); }
  f32x4Extract(lane: number): this { return this.emit(0xfd, 0x1f, lane); }
  f32x4ReplaceLane(lane: number): this { return this.emit(0xfd, 0x20, lane); }
  i8x16Ne(): this { return this.emit(0xfd, 0x24); }
  f32x4Gt(): this { return this.emit(0xfd, 0x44); }
  v128And(): this { return this.emit(0xfd, 0x4e); }
  v128AndNot(): this { return this.emit(0xfd, 0x4f); }
  v128Or(): this { return this.emit(0xfd, 0x50); }
  i8x16NarrowI16x8S(): this { return this.emit(0xfd, 0x65); }
  f32x4Nearest(): this { return this.emit(0xfd, 0x6a); }
  i8x16Shl(): this { return this.emit(0xfd, 0x6b); }
  i8x16ShrU(): this { return this.emit(0xfd, 0x6d); }
  i8x16Sub(): this { return this.emit(0xfd, 0x71); }
  i16x8ExtaddPairwiseI8x16S(): this { return this.emit(0xfd, 0x7c); }
  i32x4ExtaddPairwiseI16x8S(): this { return this.emit(0xfd, 0x7e); }
  i16x8NarrowI32x4S(): this { return this.emit(0xfd, 0x85, 0x01); }
  i16x8ExtendLowI8x16S(): this { return this.emit(0xfd, 0x87, 0x01); }
  i16x8ExtendHighI8x16S(): this { return this.emit(0xfd, 0x88, 0x01); }
  i16x8ExtendLowI8x16U(): this { return this.emit(0xfd, 0x89, 0x01); }
  i16x8ExtendHighI8x16U(): this { return this.emit(0xfd, 0x8a, 0x01); }
  i16x8Add(): this { return this.emit(0xfd, 0x8e, 0x01); }
  i16x8ExtmulLowI8x16S(): this { return this.emit(0xfd, 0x9c, 0x01); }
  i16x8ExtmulHighI8x16S(): this { return this.emit(0xfd, 0x9d, 0x01); }
  i32x4Bitmask(): this { return this.emit(0xfd, 0xa4, 0x01); }
  i32x4ExtendLowI16x8S(): this { return this.emit(0xfd, 0xa7, 0x01); }
  i32x4ExtendHighI16x8S(): this { return this.emit(0xfd, 0xa8, 0x01); }
  i32x4ExtendLowI16x8U(): this { return this.emit(0xfd, 0xa9, 0x01); }
  i32x4ExtendHighI16x8U(): this { return this.emit(0xfd, 0xaa, 0x01); }
  i32x4Shl(): this { return this.emit(0xfd, 0xab, 0x01); }
  i32x4Add(): this { return this.emit(0xfd, 0xae, 0x01); }
  i32x4Sub(): this { return this.emit(0xfd, 0xb1, 0x01); }
  i32x4Mul(): this { return this.emit(0xfd, 0xb5, 0x01); }
  i32x4DotI16x8S(): this { return this.emit(0xfd, 0xba, 0x01); }
  i32x4ExtmulLowI16x8S(): this { return this.emit(0xfd, 0xbc, 0x01); }
  i32x4ExtmulHighI16x8S(): this { return this.emit(0xfd, 0xbd, 0x01); }
  f32x4Abs(): this { return this.emit(0xfd, 0xe0, 0x01); }
  f32x4Neg(): this { return this.emit(0xfd, 0xe1, 0x01); }
  f32x4Add(): this { return this.emit(0xfd, 0xe4, 0x01); }
  f32x4Sub(): this { return this.emit(0xfd, 0xe5, 0x01); }
  f32x4Mul(): this { return this.emit(0xfd, 0xe6, 0x01); }
  f32x4Div(): this { return this.emit(0xfd, 0xe7, 0x01); }
  f32x4Pmin(): this { return this.emit(0xfd, 0xea, 0x01); }
  f32x4Pmax(): this { return this.emit(0xfd, 0xeb, 0x01); }
  i32x4TruncSatF32x4S(): this { return this.emit(0xfd, 0xf8, 0x01); }
  f32x4ConvertI32x4S(): this { return this.emit(0xfd, 0xfa, 0x01); }
  f32x4ConvertI32x4U(): this { return this.emit(0xfd, 0xfb, 0x01); }

  // Relaxed SIMD (hardware VNNI/SDOT + FMA, gated by dynamic capability probe)
  f32x4RelaxedMadd(): this { return this.emit(0xfd, 0x85, 0x02); }
  i16x8RelaxedDotI8x16I7x16S(): this { return this.emit(0xfd, 0x92, 0x02); }
  i32x4RelaxedDotI8x16I7x16AddS(): this { return this.emit(0xfd, 0x93, 0x02); }

  /**
   * Produces an `i16x8` vector whose 8 signed lanes sum to `dot(xI8, wI7)` (with `wI7 ∈ [0..63]`
   * and `xI8 ∈ [-127..127]`, each lane `∈ [-16002, +16002]`, strictly overflow-free in `i16`).
   * Lets callers fuse sub-block scale multiply directly into `i32x4.dot_i16x8_s(p, i16x8.splat(sc))`
   * (1x `vpmaddubsw` + 1x `vpmaddwd`, zero `pmulld`!).
   */
  pushDotI8I7ToI16x8(xI8: number, wI7: number, useRelaxed: boolean): this {
    if (useRelaxed) {
      this.get(xI8).get(wI7).i16x8RelaxedDotI8x16I7x16S();
    } else {
      this.get(xI8).get(wI7).i16x8ExtmulLowI8x16S()
        .get(xI8).get(wI7).i16x8ExtmulHighI8x16S()
        .i16x8Add();
    }
    return this;
  }

  /**
   * 4-row vectorized horizontal transpose-reduction:
   * Reduces four `i32x4` accumulators (`v0..v3`) into a single `i32x4` vector
   * `[hsum(v0), hsum(v1), hsum(v2), hsum(v3)]` in 6 SIMD ops (zero scalar lane extracts!).
   */
  hsum4RowsI32x4(
    v0: number,
    v1: number,
    v2: number,
    v3: number,
    t0: number,
    t1: number,
    dst: number,
  ): this {
    const LO_PAIRS = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23];
    const HI_PAIRS = [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31];
    const EVEN_LANES = [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27];
    const ODD_LANES = [4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31];
    this.get(v0).get(v1).i8x16Shuffle(LO_PAIRS)
      .get(v0).get(v1).i8x16Shuffle(HI_PAIRS)
      .i32x4Add().set(t0);
    this.get(v2).get(v3).i8x16Shuffle(LO_PAIRS)
      .get(v2).get(v3).i8x16Shuffle(HI_PAIRS)
      .i32x4Add().set(t1);
    this.get(t0).get(t1).i8x16Shuffle(EVEN_LANES)
      .get(t0).get(t1).i8x16Shuffle(ODD_LANES)
      .i32x4Add().set(dst);
    return this;
  }

  /** Helper: load F16 scalar from `ptr + byteOff`, look up in `F16_LUT`, return f32. */
  loadF16AsF32(ptrLocal: number, byteOff = 0): this {
    this.get(ptrLocal).i32Load16U(byteOff).i32Const(2).i32Shl().f32Load(WASM_F16_LUT_OFFSET);
    return this;
  }

  /** Helper: load F16 scalar from `ptr + byteOff`, look up in `F16_LUT`, promote to f64. */
  loadF16AsF64(ptrLocal: number, byteOff = 0): this {
    this.loadF16AsF32(ptrLocal, byteOff).f64PromoteF32();
    return this;
  }

  /** Helper: horizontal sum of a `v128` (`i32x4`) as signed `i32`. */
  hsumI32x4(vecLocal: number): this {
    this.get(vecLocal).i32x4Extract(0);
    this.get(vecLocal).i32x4Extract(1).i32Add();
    this.get(vecLocal).i32x4Extract(2).i32Add();
    this.get(vecLocal).i32x4Extract(3).i32Add();
    return this;
  }

  /**
   * Helper: accumulate 16-lane dot product of signed `xI8` (`[-127, 127]`) with non-negative 7-bit
   * `wI7` (`[0, 63] ⊂ [0, 127]`) into `accI32` (`i32x4`).
   * Uses 1-instruction hardware VNNI/SDOT (`i32x4.relaxed_dot_i8x16_i7x16_add_s`) on Relaxed SIMD,
   * or 2x standard `i32x4.dot_i16x8_s` on baseline WASM SIMD128.
   */
  dotI8I7Add(xI8: number, wI7: number, accI32: number, useRelaxed: boolean): this {
    if (useRelaxed) {
      this.get(xI8).get(wI7).get(accI32).i32x4RelaxedDotI8x16I7x16AddS().set(accI32);
    } else {
      this.get(xI8).i16x8ExtendLowI8x16S()
        .get(wI7).i16x8ExtendLowI8x16U()
        .i32x4DotI16x8S()
        .get(accI32).i32x4Add().set(accI32);
      this.get(xI8).i16x8ExtendHighI8x16S()
        .get(wI7).i16x8ExtendHighI8x16U()
        .i32x4DotI16x8S()
        .get(accI32).i32x4Add().set(accI32);
    }
    return this;
  }

  /**
   * Helper: accumulate 16-lane exact signed `i8 × i8` dot product into `accI32` (`i32x4`)
   * using standard SIMD128 `i32x4.dot_i16x8_s`.
   */
  dotI8I8Add(xI8: number, wI8: number, accI32: number): this {
    this.get(xI8).i16x8ExtendLowI8x16S()
      .get(wI8).i16x8ExtendLowI8x16S()
      .i32x4DotI16x8S()
      .get(accI32).i32x4Add().set(accI32);
    this.get(xI8).i16x8ExtendHighI8x16S()
      .get(wI8).i16x8ExtendHighI8x16S()
      .i32x4DotI16x8S()
      .get(accI32).i32x4Add().set(accI32);
    return this;
  }

  /** Helper: horizontal sum of a `v128` (`f32x4`) promoted to `f64`. */
  hsumF32x4ToF64(vecLocal: number): this {
    this.get(vecLocal).f32x4Extract(0).f64PromoteF32();
    this.get(vecLocal).f32x4Extract(1).f64PromoteF32().f64Add();
    this.get(vecLocal).f32x4Extract(2).f64PromoteF32().f64Add();
    this.get(vecLocal).f32x4Extract(3).f64PromoteF32().f64Add();
    return this;
  }

  /**
   * Helper: accumulate dot product of 16 unsigned bytes in `u8Vec` against 16 floats at `xPtr + xByteOff`
   * into `f32x4` accumulator `accVec`. Uses scratch `v128` local `tmp16`.
   */
  dotU8x16WithX(u8Vec: number, xPtr: number, xByteOff: number, accVec: number, tmp16: number): this {
    // Low 8 bytes -> 2 x f32x4
    this.get(u8Vec).i16x8ExtendLowI8x16U().tee(tmp16);
    this.i32x4ExtendLowI16x8U().f32x4ConvertI32x4S();
    this.get(xPtr).v128Load(xByteOff).f32x4Mul().get(accVec).f32x4Add().set(accVec);

    this.get(tmp16).i32x4ExtendHighI16x8U().f32x4ConvertI32x4S();
    this.get(xPtr).v128Load(xByteOff + 16).f32x4Mul().get(accVec).f32x4Add().set(accVec);

    // High 8 bytes -> 2 x f32x4
    this.get(u8Vec).i16x8ExtendHighI8x16U().tee(tmp16);
    this.i32x4ExtendLowI16x8U().f32x4ConvertI32x4S();
    this.get(xPtr).v128Load(xByteOff + 32).f32x4Mul().get(accVec).f32x4Add().set(accVec);

    this.get(tmp16).i32x4ExtendHighI16x8U().f32x4ConvertI32x4S();
    this.get(xPtr).v128Load(xByteOff + 48).f32x4Mul().get(accVec).f32x4Add().set(accVec);
    return this;
  }

  /**
   * Helper: accumulate dot product of 16 signed int8 bytes in `s8Vec` against 16 floats at `xPtr + xByteOff`
   * into `f32x4` accumulator `accVec`. Uses scratch `v128` local `tmp16`.
   */
  dotS8x16WithX(s8Vec: number, xPtr: number, xByteOff: number, accVec: number, tmp16: number): this {
    this.get(s8Vec).i16x8ExtendLowI8x16S().tee(tmp16);
    this.i32x4ExtendLowI16x8S().f32x4ConvertI32x4S();
    this.get(xPtr).v128Load(xByteOff).f32x4Mul().get(accVec).f32x4Add().set(accVec);

    this.get(tmp16).i32x4ExtendHighI16x8S().f32x4ConvertI32x4S();
    this.get(xPtr).v128Load(xByteOff + 16).f32x4Mul().get(accVec).f32x4Add().set(accVec);

    this.get(s8Vec).i16x8ExtendHighI8x16S().tee(tmp16);
    this.i32x4ExtendLowI16x8S().f32x4ConvertI32x4S();
    this.get(xPtr).v128Load(xByteOff + 32).f32x4Mul().get(accVec).f32x4Add().set(accVec);

    this.get(tmp16).i32x4ExtendHighI16x8S().f32x4ConvertI32x4S();
    this.get(xPtr).v128Load(xByteOff + 48).f32x4Mul().get(accVec).f32x4Add().set(accVec);
    return this;
  }

  /**
   * Helper: accumulate dot product of 16 unsigned bytes in `u8Vec` against 4 pre-loaded `v128` x locals
   * (`x0..x3`) into `accVec`. Used for 2-row tiling where `x` stays in SIMD registers!
   */
  dotU8x16WithRegs(
    u8Vec: number,
    x0: number,
    x1: number,
    x2: number,
    x3: number,
    accVec: number,
    tmp16: number,
  ): this {
    this.get(u8Vec).i16x8ExtendLowI8x16U().tee(tmp16);
    this.i32x4ExtendLowI16x8U().f32x4ConvertI32x4S().get(x0).f32x4Mul().get(accVec).f32x4Add().set(accVec);
    this.get(tmp16).i32x4ExtendHighI16x8U().f32x4ConvertI32x4S().get(x1).f32x4Mul().get(accVec).f32x4Add().set(accVec);

    this.get(u8Vec).i16x8ExtendHighI8x16U().tee(tmp16);
    this.i32x4ExtendLowI16x8U().f32x4ConvertI32x4S().get(x2).f32x4Mul().get(accVec).f32x4Add().set(accVec);
    this.get(tmp16).i32x4ExtendHighI16x8U().f32x4ConvertI32x4S().get(x3).f32x4Mul().get(accVec).f32x4Add().set(accVec);
    return this;
  }

  /**
   * Helper: accumulate dot product of 16 signed int8 bytes in `s8Vec` against 4 pre-loaded `v128` x locals.
   */
  dotS8x16WithRegs(
    s8Vec: number,
    x0: number,
    x1: number,
    x2: number,
    x3: number,
    accVec: number,
    tmp16: number,
  ): this {
    this.get(s8Vec).i16x8ExtendLowI8x16S().tee(tmp16);
    this.i32x4ExtendLowI16x8S().f32x4ConvertI32x4S().get(x0).f32x4Mul().get(accVec).f32x4Add().set(accVec);
    this.get(tmp16).i32x4ExtendHighI16x8S().f32x4ConvertI32x4S().get(x1).f32x4Mul().get(accVec).f32x4Add().set(accVec);

    this.get(s8Vec).i16x8ExtendHighI8x16S().tee(tmp16);
    this.i32x4ExtendLowI16x8S().f32x4ConvertI32x4S().get(x2).f32x4Mul().get(accVec).f32x4Add().set(accVec);
    this.get(tmp16).i32x4ExtendHighI16x8S().f32x4ConvertI32x4S().get(x3).f32x4Mul().get(accVec).f32x4Add().set(accVec);
    return this;
  }

  encodeBody(): number[] {
    // Group consecutive locals of identical type
    const groups: [number, number][] = [];
    for (const t of this.localTypes) {
      if (groups.length > 0 && groups[groups.length - 1][1] === t) {
        groups[groups.length - 1][0] += 1;
      } else {
        groups.push([1, t]);
      }
    }
    const header: number[] = [...uleb(groups.length)];
    for (const [count, t] of groups) {
      header.push(...uleb(count), t);
    }
    const body = [...header, ...this.code, 0x0b];
    return [...uleb(body.length), ...body];
  }
}

const ZERO_16 = new Array<number>(16).fill(0);
const MASK_0F_16 = new Array<number>(16).fill(0x0f);
const MASK_03_16 = new Array<number>(16).fill(0x03);
const VAL_16_16 = new Array<number>(16).fill(16);
const VAL_32_16 = new Array<number>(16).fill(32);
const SWIZ_LOW16_BITS = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1];
const SWIZ_HIGH16_BITS = [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3];
const BIT_MASKS_8X2 = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

/** Builds the helper function `precompute_block_sum_x(xPtr, cols, blockSumOffset)` (32-elem blocks). */
function buildPrecomputeBlockSumX(): FuncBuilder {
  // params: xPtr (0), numBlocks (1), dstPtr (2)
  const f = new FuncBuilder([I32, I32, I32]);
  const b = f.local(I32);
  const xp = f.local(I32);
  const dp = f.local(I32);
  const vsum = f.local(V128);

  f.i32Const(0).set(b);
  f.get(0).set(xp);
  f.get(2).set(dp);

  f.block().loop();
  f.get(b).get(1).i32GeS().brIf(1);

  f.get(xp).v128Load(0)
    .get(xp).v128Load(16).f32x4Add()
    .get(xp).v128Load(32).f32x4Add()
    .get(xp).v128Load(48).f32x4Add()
    .get(xp).v128Load(64).f32x4Add()
    .get(xp).v128Load(80).f32x4Add()
    .get(xp).v128Load(96).f32x4Add()
    .get(xp).v128Load(112).f32x4Add()
    .set(vsum);

  f.get(dp).hsumF32x4ToF64(vsum).f64Store(0);

  f.get(xp).i32Const(128).i32Add().set(xp);
  f.get(dp).i32Const(8).i32Add().set(dp);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();
  return f;
}

/**
 * Builds `mul_mat_f32(wPtr, xPtr, outPtr, cols, rows)` with 4-row register blocking
 * and hardware FMA (`f32x4.relaxed_madd`) when Relaxed SIMD is active.
 */
function buildMulMatF32(useRelaxed: boolean): FuncBuilder {
  // params: wPtr(0), xPtr(1), outPtr(2), cols(3), rows(4)
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const c = f.local(I32);
  const rowBytes = f.local(I32);
  const p0 = f.local(I32);
  const p1 = f.local(I32);
  const p2 = f.local(I32);
  const p3 = f.local(I32);
  const xp = f.local(I32);
  const op = f.local(I32);
  const v0 = f.local(V128);
  const v1 = f.local(V128);
  const v2 = f.local(V128);
  const v3 = f.local(V128);
  const xv = f.local(V128);
  const a0 = f.local(F64);
  const a1 = f.local(F64);
  const a2 = f.local(F64);
  const a3 = f.local(F64);
  const xScalar = f.local(F64);
  const cols4 = f.local(I32);

  f.get(3).i32Const(2).i32Shl().set(rowBytes);
  f.get(3).i32Const(-4).i32And().set(cols4);
  f.i32Const(0).set(r);

  const emitMadd = (pLoc: number, vLoc: number) => {
    if (useRelaxed) {
      f.get(xv).get(pLoc).v128Load(0).get(vLoc).f32x4RelaxedMadd().set(vLoc);
    } else {
      f.get(xv).get(pLoc).v128Load(0).f32x4Mul().get(vLoc).f32x4Add().set(vLoc);
    }
  };

  // 4-row register-blocked loop (4x fewer activation loads, 4 independent FMA accumulators)
  f.block().loop();
  f.get(r).i32Const(3).i32Add().get(4).i32GeS().brIf(1);

  f.get(0).get(r).get(rowBytes).i32Mul().i32Add()
    .tee(p0).get(rowBytes).i32Add()
    .tee(p1).get(rowBytes).i32Add()
    .tee(p2).get(rowBytes).i32Add().set(p3);
  f.get(1).set(xp);
  f.v128Const(ZERO_16).tee(v0).tee(v1).tee(v2).set(v3);
  f.i32Const(0).set(c);

  f.block().loop();
  f.get(c).get(cols4).i32GeS().brIf(1);
  f.get(xp).v128Load(0).set(xv);
  emitMadd(p0, v0);
  emitMadd(p1, v1);
  emitMadd(p2, v2);
  emitMadd(p3, v3);
  f.get(p0).i32Const(16).i32Add().set(p0);
  f.get(p1).i32Const(16).i32Add().set(p1);
  f.get(p2).i32Const(16).i32Add().set(p2);
  f.get(p3).i32Const(16).i32Add().set(p3);
  f.get(xp).i32Const(16).i32Add().set(xp);
  f.get(c).i32Const(4).i32Add().set(c);
  f.br(0).end().end();

  f.hsumF32x4ToF64(v0).set(a0);
  f.hsumF32x4ToF64(v1).set(a1);
  f.hsumF32x4ToF64(v2).set(a2);
  f.hsumF32x4ToF64(v3).set(a3);

  // Scalar tail
  f.block().loop();
  f.get(c).get(3).i32GeS().brIf(1);
  f.get(xp).f32Load(0).f64PromoteF32().set(xScalar);
  f.get(a0).get(p0).f32Load(0).f64PromoteF32().get(xScalar).f64Mul().f64Add().set(a0);
  f.get(a1).get(p1).f32Load(0).f64PromoteF32().get(xScalar).f64Mul().f64Add().set(a1);
  f.get(a2).get(p2).f32Load(0).f64PromoteF32().get(xScalar).f64Mul().f64Add().set(a2);
  f.get(a3).get(p3).f32Load(0).f64PromoteF32().get(xScalar).f64Mul().f64Add().set(a3);
  f.get(p0).i32Const(4).i32Add().set(p0);
  f.get(p1).i32Const(4).i32Add().set(p1);
  f.get(p2).i32Const(4).i32Add().set(p2);
  f.get(p3).i32Const(4).i32Add().set(p3);
  f.get(xp).i32Const(4).i32Add().set(xp);
  f.get(c).i32Const(1).i32Add().set(c);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().set(op);
  f.get(op).get(a0).f32DemoteF64().f32Store(0);
  f.get(op).get(a1).f32DemoteF64().f32Store(4);
  f.get(op).get(a2).f32DemoteF64().f32Store(8);
  f.get(op).get(a3).f32DemoteF64().f32Store(12);
  f.get(r).i32Const(4).i32Add().set(r);
  f.br(0).end().end();

  // Remainder rows (0..3)
  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(p0);
  f.get(1).set(xp);
  f.v128Const(ZERO_16).set(v0);
  f.i32Const(0).set(c);
  f.block().loop();
  f.get(c).get(cols4).i32GeS().brIf(1);
  f.get(xp).v128Load(0).set(xv);
  emitMadd(p0, v0);
  f.get(p0).i32Const(16).i32Add().set(p0);
  f.get(xp).i32Const(16).i32Add().set(xp);
  f.get(c).i32Const(4).i32Add().set(c);
  f.br(0).end().end();
  f.hsumF32x4ToF64(v0).set(a0);
  f.block().loop();
  f.get(c).get(3).i32GeS().brIf(1);
  f.get(a0).get(p0).f32Load(0).f64PromoteF32().get(xp).f32Load(0).f64PromoteF32().f64Mul().f64Add().set(a0);
  f.get(p0).i32Const(4).i32Add().set(p0);
  f.get(xp).i32Const(4).i32Add().set(xp);
  f.get(c).i32Const(1).i32Add().set(c);
  f.br(0).end().end();
  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(a0).f32DemoteF64().f32Store(0);
  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();

  return f;
}

/** Builds `mul_mat_bf16(wPtr, xPtr, outPtr, cols, rows)` using native SIMD `load16x4_u` + `shl 16`. */
function buildMulMatBF16(): FuncBuilder {
  // params: wPtr(0), xPtr(1), outPtr(2), cols(3), rows(4)
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const c = f.local(I32);
  const rowBytes = f.local(I32);
  const p0 = f.local(I32);
  const xp = f.local(I32);
  const v0 = f.local(V128);
  const a0 = f.local(F64);
  const cols4 = f.local(I32);

  f.get(3).i32Const(1).i32Shl().set(rowBytes);
  f.get(3).i32Const(-4).i32And().set(cols4);
  f.i32Const(0).set(r);

  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(p0);
  f.get(1).set(xp);
  f.v128Const(ZERO_16).set(v0);
  f.i32Const(0).set(c);

  f.block().loop();
  f.get(c).get(cols4).i32GeS().brIf(1);
  f.get(p0).v128Load16x4U(0).i32Const(16).i32x4Shl()
    .get(xp).v128Load(0).f32x4Mul().get(v0).f32x4Add().set(v0);
  f.get(p0).i32Const(8).i32Add().set(p0);
  f.get(xp).i32Const(16).i32Add().set(xp);
  f.get(c).i32Const(4).i32Add().set(c);
  f.br(0).end().end();

  f.hsumF32x4ToF64(v0).set(a0);
  f.block().loop();
  f.get(c).get(3).i32GeS().brIf(1);
  f.get(a0)
    .get(p0).i32Load16U(0).i32Const(16).i32Shl().emit(0xbe).f64PromoteF32()
    .get(xp).f32Load(0).f64PromoteF32().f64Mul().f64Add().set(a0);
  f.get(p0).i32Const(2).i32Add().set(p0);
  f.get(xp).i32Const(4).i32Add().set(xp);
  f.get(c).i32Const(1).i32Add().set(c);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(a0).f32DemoteF64().f32Store(0);
  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();
  return f;
}

/** Builds 4-row register-blocked `mul_mat_f16(wPtr, xPtr, outPtr, cols, rows)` using WASM-resident F16 LUT. */
function buildMulMatF16(): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const c = f.local(I32);
  const rowBytes = f.local(I32);
  const p0 = f.local(I32);
  const p1 = f.local(I32);
  const p2 = f.local(I32);
  const p3 = f.local(I32);
  const xp = f.local(I32);
  const op = f.local(I32);
  const a0 = f.local(F64);
  const a1 = f.local(F64);
  const a2 = f.local(F64);
  const a3 = f.local(F64);
  const x0 = f.local(F64);
  const x1 = f.local(F64);
  const cols2 = f.local(I32);

  f.get(3).i32Const(1).i32Shl().set(rowBytes);
  f.get(3).i32Const(-2).i32And().set(cols2);
  f.i32Const(0).set(r);

  // 4-row register-blocked loop (4x fewer activation loads + 2x inner unroll)
  f.block().loop();
  f.get(r).i32Const(3).i32Add().get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add()
    .tee(p0).get(rowBytes).i32Add()
    .tee(p1).get(rowBytes).i32Add()
    .tee(p2).get(rowBytes).i32Add().set(p3);
  f.get(1).set(xp);
  f.f64Const(0).tee(a0).tee(a1).tee(a2).set(a3);
  f.i32Const(0).set(c);

  f.block().loop();
  f.get(c).get(cols2).i32GeS().brIf(1);
  f.get(xp).f32Load(0).f64PromoteF32().set(x0);
  f.get(xp).f32Load(4).f64PromoteF32().set(x1);
  f.get(a0).loadF16AsF64(p0, 0).get(x0).f64Mul().f64Add().loadF16AsF64(p0, 2).get(x1).f64Mul().f64Add().set(a0);
  f.get(a1).loadF16AsF64(p1, 0).get(x0).f64Mul().f64Add().loadF16AsF64(p1, 2).get(x1).f64Mul().f64Add().set(a1);
  f.get(a2).loadF16AsF64(p2, 0).get(x0).f64Mul().f64Add().loadF16AsF64(p2, 2).get(x1).f64Mul().f64Add().set(a2);
  f.get(a3).loadF16AsF64(p3, 0).get(x0).f64Mul().f64Add().loadF16AsF64(p3, 2).get(x1).f64Mul().f64Add().set(a3);
  f.get(p0).i32Const(4).i32Add().set(p0);
  f.get(p1).i32Const(4).i32Add().set(p1);
  f.get(p2).i32Const(4).i32Add().set(p2);
  f.get(p3).i32Const(4).i32Add().set(p3);
  f.get(xp).i32Const(8).i32Add().set(xp);
  f.get(c).i32Const(2).i32Add().set(c);
  f.br(0).end().end();

  f.get(c).get(3).i32LtS().if_();
  f.get(xp).f32Load(0).f64PromoteF32().set(x0);
  f.get(a0).loadF16AsF64(p0, 0).get(x0).f64Mul().f64Add().set(a0);
  f.get(a1).loadF16AsF64(p1, 0).get(x0).f64Mul().f64Add().set(a1);
  f.get(a2).loadF16AsF64(p2, 0).get(x0).f64Mul().f64Add().set(a2);
  f.get(a3).loadF16AsF64(p3, 0).get(x0).f64Mul().f64Add().set(a3);
  f.end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().set(op);
  f.get(op).get(a0).f32DemoteF64().f32Store(0);
  f.get(op).get(a1).f32DemoteF64().f32Store(4);
  f.get(op).get(a2).f32DemoteF64().f32Store(8);
  f.get(op).get(a3).f32DemoteF64().f32Store(12);
  f.get(r).i32Const(4).i32Add().set(r);
  f.br(0).end().end();

  // Remainder rows (0..3)
  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(p0);
  f.get(1).set(xp);
  f.f64Const(0).set(a0);
  f.i32Const(0).set(c);
  f.block().loop();
  f.get(c).get(3).i32GeS().brIf(1);
  f.get(a0).loadF16AsF64(p0, 0).get(xp).f32Load(0).f64PromoteF32().f64Mul().f64Add().set(a0);
  f.get(p0).i32Const(2).i32Add().set(p0);
  f.get(xp).i32Const(4).i32Add().set(xp);
  f.get(c).i32Const(1).i32Add().set(c);
  f.br(0).end().end();
  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(a0).f32DemoteF64().f32Store(0);
  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();
  return f;
}

/**
 * Builds `mul_mat_q8_0(wPtr, xPtr, outPtr, cols, rows)` with 2-row activation register tiling.
 * Each 32-weight block = 34 bytes (2-byte F16 scale + 32 int8 quants).
 */
function buildMulMatQ8_0(): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const b = f.local(I32);
  const numBlocks = f.local(I32);
  const rowBytes = f.local(I32);
  const p0 = f.local(I32);
  const p1 = f.local(I32);
  const xp = f.local(I32);
  const a0 = f.local(F64);
  const a1 = f.local(F64);
  const x0 = f.local(V128);
  const x1 = f.local(V128);
  const x2 = f.local(V128);
  const x3 = f.local(V128);
  const x4 = f.local(V128);
  const x5 = f.local(V128);
  const x6 = f.local(V128);
  const x7 = f.local(V128);
  const vsum0 = f.local(V128);
  const vsum1 = f.local(V128);
  const tmp16 = f.local(V128);
  const qv = f.local(V128);

  f.get(3).i32Const(5).i32ShrU().tee(numBlocks).i32Const(34).i32Mul().set(rowBytes);
  f.i32Const(0).set(r);

  // 2-row tiled loop
  f.block().loop();
  f.get(r).i32Const(1).i32Add().get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().tee(p0).get(rowBytes).i32Add().set(p1);
  f.get(1).set(xp);
  f.f64Const(0).set(a0);
  f.f64Const(0).set(a1);
  f.i32Const(0).set(b);

  f.block().loop();
  f.get(b).get(numBlocks).i32GeS().brIf(1);

  // Load 32 floats of x into 8 v128 registers once for both rows
  f.get(xp).v128Load(0).set(x0);
  f.get(xp).v128Load(16).set(x1);
  f.get(xp).v128Load(32).set(x2);
  f.get(xp).v128Load(48).set(x3);
  f.get(xp).v128Load(64).set(x4);
  f.get(xp).v128Load(80).set(x5);
  f.get(xp).v128Load(96).set(x6);
  f.get(xp).v128Load(112).set(x7);

  // Row 0
  f.v128Const(ZERO_16).set(vsum0);
  f.get(p0).v128Load(2).set(qv);
  f.dotS8x16WithRegs(qv, x0, x1, x2, x3, vsum0, tmp16);
  f.get(p0).v128Load(18).set(qv);
  f.dotS8x16WithRegs(qv, x4, x5, x6, x7, vsum0, tmp16);
  f.get(a0).hsumF32x4ToF64(vsum0).loadF16AsF64(p0, 0).f64Mul().f64Add().set(a0);

  // Row 1
  f.v128Const(ZERO_16).set(vsum1);
  f.get(p1).v128Load(2).set(qv);
  f.dotS8x16WithRegs(qv, x0, x1, x2, x3, vsum1, tmp16);
  f.get(p1).v128Load(18).set(qv);
  f.dotS8x16WithRegs(qv, x4, x5, x6, x7, vsum1, tmp16);
  f.get(a1).hsumF32x4ToF64(vsum1).loadF16AsF64(p1, 0).f64Mul().f64Add().set(a1);

  f.get(p0).i32Const(34).i32Add().set(p0);
  f.get(p1).i32Const(34).i32Add().set(p1);
  f.get(xp).i32Const(128).i32Add().set(xp);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(a0).f32DemoteF64().f32Store(0);
  f.get(2).get(r).i32Const(1).i32Add().i32Const(2).i32Shl().i32Add().get(a1).f32DemoteF64().f32Store(0);
  f.get(r).i32Const(2).i32Add().set(r);
  f.br(0).end().end();

  // Odd remainder row
  f.get(r).get(4).i32LtS().if_();
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(p0);
  f.get(1).set(xp);
  f.f64Const(0).set(a0);
  f.i32Const(0).set(b);
  f.block().loop();
  f.get(b).get(numBlocks).i32GeS().brIf(1);
  f.v128Const(ZERO_16).set(vsum0);
  f.get(p0).v128Load(2).set(qv);
  f.dotS8x16WithX(qv, xp, 0, vsum0, tmp16);
  f.get(p0).v128Load(18).set(qv);
  f.dotS8x16WithX(qv, xp, 64, vsum0, tmp16);
  f.get(a0).hsumF32x4ToF64(vsum0).loadF16AsF64(p0, 0).f64Mul().f64Add().set(a0);
  f.get(p0).i32Const(34).i32Add().set(p0);
  f.get(xp).i32Const(128).i32Add().set(xp);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();
  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(a0).f32DemoteF64().f32Store(0);
  f.end();

  return f;
}

/**
 * Builds `mul_mat_q4_0(wPtr, xPtr, outPtr, cols, rows)` with:
 * - Precomputed `blockSumX` factoring (`sum((q-8)*x) = sum(q*x) - 8*blockSumX[b]`)
 * - 2-row activation register tiling (loads x[0..31] into v128 registers ONCE per 2 rows)
 */
function buildMulMatQ4_0(precomputeFnIdx: number): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const b = f.local(I32);
  const numBlocks = f.local(I32);
  const rowBytes = f.local(I32);
  const p0 = f.local(I32);
  const p1 = f.local(I32);
  const xp = f.local(I32);
  const bp = f.local(I32);
  const a0 = f.local(F64);
  const a1 = f.local(F64);
  const bx8 = f.local(F64);
  const mask0f = f.local(V128);
  const x0 = f.local(V128);
  const x1 = f.local(V128);
  const x2 = f.local(V128);
  const x3 = f.local(V128);
  const x4 = f.local(V128);
  const x5 = f.local(V128);
  const x6 = f.local(V128);
  const x7 = f.local(V128);
  const packed = f.local(V128);
  const nib = f.local(V128);
  const vsum0 = f.local(V128);
  const vsum1 = f.local(V128);
  const tmp16 = f.local(V128);

  f.get(3).i32Const(5).i32ShrU().tee(numBlocks).i32Const(18).i32Mul().set(rowBytes);
  f.get(1).get(numBlocks).i32Const(WASM_BLOCK_SUM_X_OFFSET).call(precomputeFnIdx);
  f.v128Const(MASK_0F_16).set(mask0f);
  f.i32Const(0).set(r);

  // 2-row tiled loop
  f.block().loop();
  f.get(r).i32Const(1).i32Add().get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().tee(p0).get(rowBytes).i32Add().set(p1);
  f.get(1).set(xp);
  f.i32Const(WASM_BLOCK_SUM_X_OFFSET).set(bp);
  f.f64Const(0).set(a0);
  f.f64Const(0).set(a1);
  f.i32Const(0).set(b);

  f.block().loop();
  f.get(b).get(numBlocks).i32GeS().brIf(1);

  f.get(bp).f64Load(0).f64Const(8.0).f64Mul().set(bx8);
  f.get(xp).v128Load(0).set(x0);
  f.get(xp).v128Load(16).set(x1);
  f.get(xp).v128Load(32).set(x2);
  f.get(xp).v128Load(48).set(x3);
  f.get(xp).v128Load(64).set(x4);
  f.get(xp).v128Load(80).set(x5);
  f.get(xp).v128Load(96).set(x6);
  f.get(xp).v128Load(112).set(x7);

  // Row 0
  f.v128Const(ZERO_16).set(vsum0);
  f.get(p0).v128Load(2).tee(packed).get(mask0f).v128And().set(nib);
  f.dotU8x16WithRegs(nib, x0, x1, x2, x3, vsum0, tmp16);
  f.get(packed).i32Const(4).i8x16ShrU().set(nib);
  f.dotU8x16WithRegs(nib, x4, x5, x6, x7, vsum0, tmp16);
  f.get(a0).hsumF32x4ToF64(vsum0).get(bx8).f64Sub().loadF16AsF64(p0, 0).f64Mul().f64Add().set(a0);

  // Row 1
  f.v128Const(ZERO_16).set(vsum1);
  f.get(p1).v128Load(2).tee(packed).get(mask0f).v128And().set(nib);
  f.dotU8x16WithRegs(nib, x0, x1, x2, x3, vsum1, tmp16);
  f.get(packed).i32Const(4).i8x16ShrU().set(nib);
  f.dotU8x16WithRegs(nib, x4, x5, x6, x7, vsum1, tmp16);
  f.get(a1).hsumF32x4ToF64(vsum1).get(bx8).f64Sub().loadF16AsF64(p1, 0).f64Mul().f64Add().set(a1);

  f.get(p0).i32Const(18).i32Add().set(p0);
  f.get(p1).i32Const(18).i32Add().set(p1);
  f.get(xp).i32Const(128).i32Add().set(xp);
  f.get(bp).i32Const(8).i32Add().set(bp);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(a0).f32DemoteF64().f32Store(0);
  f.get(2).get(r).i32Const(1).i32Add().i32Const(2).i32Shl().i32Add().get(a1).f32DemoteF64().f32Store(0);
  f.get(r).i32Const(2).i32Add().set(r);
  f.br(0).end().end();

  // Odd remainder row
  f.get(r).get(4).i32LtS().if_();
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(p0);
  f.get(1).set(xp);
  f.i32Const(WASM_BLOCK_SUM_X_OFFSET).set(bp);
  f.f64Const(0).set(a0);
  f.i32Const(0).set(b);
  f.block().loop();
  f.get(b).get(numBlocks).i32GeS().brIf(1);
  f.v128Const(ZERO_16).set(vsum0);
  f.get(p0).v128Load(2).tee(packed).get(mask0f).v128And().set(nib);
  f.dotU8x16WithX(nib, xp, 0, vsum0, tmp16);
  f.get(packed).i32Const(4).i8x16ShrU().set(nib);
  f.dotU8x16WithX(nib, xp, 64, vsum0, tmp16);
  f.get(a0)
    .hsumF32x4ToF64(vsum0).get(bp).f64Load(0).f64Const(8.0).f64Mul().f64Sub()
    .loadF16AsF64(p0, 0).f64Mul().f64Add().set(a0);
  f.get(p0).i32Const(18).i32Add().set(p0);
  f.get(xp).i32Const(128).i32Add().set(xp);
  f.get(bp).i32Const(8).i32Add().set(bp);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();
  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(a0).f32DemoteF64().f32Store(0);
  f.end();

  return f;
}

/**
 * Builds `mul_mat_q4_0_batch4(wPtr, xBatchPtr, outBatchPtr, cols, rows)`:
 * Processes 4 prompt tokens simultaneously per weight read!
 * Decodes each 32-element Q4_0 block ONCE into 8 `f32x4` registers and multiplies against all 4
 * token vectors, cutting prefill weight-decode work by 4x!
 * Layout: `xBatchPtr` holds 4 contiguous `[cols]` F32 vectors; `outBatchPtr` holds 4 contiguous `[rows]` F32 vectors.
 */
function buildMulMatQ4_0Batch4(precomputeFnIdx: number): FuncBuilder {
  // params: wPtr(0), xBatchPtr(1), outBatchPtr(2), cols(3), rows(4), oStrideBytes(5)
  const f = new FuncBuilder([I32, I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const b = f.local(I32);
  const numBlocks = f.local(I32);
  const rowBytes = f.local(I32);
  const xStride = f.local(I32);
  const oStride = f.local(I32);
  const p = f.local(I32);
  const xp0 = f.local(I32);
  const xp1 = f.local(I32);
  const xp2 = f.local(I32);
  const xp3 = f.local(I32);
  const bp = f.local(I32);
  const bsStride = f.local(I32);
  const a0 = f.local(F64);
  const a1 = f.local(F64);
  const a2 = f.local(F64);
  const a3 = f.local(F64);
  const scale = f.local(F64);
  const mask0f = f.local(V128);
  const packed = f.local(V128);
  const nib = f.local(V128);
  const tmp16 = f.local(V128);
  const w0 = f.local(V128);
  const w1 = f.local(V128);
  const w2 = f.local(V128);
  const w3 = f.local(V128);
  const s0 = f.local(V128);
  const s1 = f.local(V128);
  const s2 = f.local(V128);
  const s3 = f.local(V128);

  f.get(3).i32Const(5).i32ShrU().tee(numBlocks).i32Const(18).i32Mul().set(rowBytes);
  f.get(3).i32Const(2).i32Shl().set(xStride);
  f.get(5).set(oStride);
  f.get(numBlocks).i32Const(3).i32Shl().set(bsStride);

  // Precompute blockSumX for all 4 tokens in WASM_BLOCK_SUM_X_OFFSET
  for (let t = 0; t < 4; t++) {
    f.get(1);
    if (t > 0) f.get(xStride).i32Const(t).i32Mul().i32Add();
    f.get(numBlocks);
    f.i32Const(WASM_BLOCK_SUM_X_OFFSET);
    if (t > 0) f.get(bsStride).i32Const(t).i32Mul().i32Add();
    f.call(precomputeFnIdx);
  }

  f.v128Const(MASK_0F_16).set(mask0f);
  f.i32Const(0).set(r);

  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);

  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(p);
  f.get(1).tee(xp0).get(xStride).i32Add().tee(xp1).get(xStride).i32Add().tee(xp2).get(xStride).i32Add().set(xp3);
  f.i32Const(WASM_BLOCK_SUM_X_OFFSET).set(bp);
  f.f64Const(0).set(a0);
  f.f64Const(0).set(a1);
  f.f64Const(0).set(a2);
  f.f64Const(0).set(a3);
  f.i32Const(0).set(b);

  f.block().loop();
  f.get(b).get(numBlocks).i32GeS().brIf(1);

  f.loadF16AsF64(p, 0).set(scale);

  const xps = [xp0, xp1, xp2, xp3];
  const sums = [s0, s1, s2, s3];
  const accs = [a0, a1, a2, a3];
  const ws = [w0, w1, w2, w3];

  // Wave 1: low 16 nibbles -> w0..w3 (fits cleanly in 16 XMM registers with zero spills)
  f.get(p).v128Load(2).tee(packed).get(mask0f).v128And().set(nib);
  f.get(nib).i16x8ExtendLowI8x16U().tee(tmp16).i32x4ExtendLowI16x8U().f32x4ConvertI32x4S().set(w0);
  f.get(tmp16).i32x4ExtendHighI16x8U().f32x4ConvertI32x4S().set(w1);
  f.get(nib).i16x8ExtendHighI8x16U().tee(tmp16).i32x4ExtendLowI16x8U().f32x4ConvertI32x4S().set(w2);
  f.get(tmp16).i32x4ExtendHighI16x8U().f32x4ConvertI32x4S().set(w3);

  for (let t = 0; t < 4; t++) {
    f.get(ws[0]).get(xps[t]).v128Load(0).f32x4Mul()
      .get(ws[1]).get(xps[t]).v128Load(16).f32x4Mul().f32x4Add()
      .get(ws[2]).get(xps[t]).v128Load(32).f32x4Mul().f32x4Add()
      .get(ws[3]).get(xps[t]).v128Load(48).f32x4Mul().f32x4Add()
      .set(sums[t]);
  }

  // Wave 2: high 16 nibbles -> reuse w0..w3
  f.get(packed).i32Const(4).i8x16ShrU().set(nib);
  f.get(nib).i16x8ExtendLowI8x16U().tee(tmp16).i32x4ExtendLowI16x8U().f32x4ConvertI32x4S().set(w0);
  f.get(tmp16).i32x4ExtendHighI16x8U().f32x4ConvertI32x4S().set(w1);
  f.get(nib).i16x8ExtendHighI8x16U().tee(tmp16).i32x4ExtendLowI16x8U().f32x4ConvertI32x4S().set(w2);
  f.get(tmp16).i32x4ExtendHighI16x8U().f32x4ConvertI32x4S().set(w3);

  for (let t = 0; t < 4; t++) {
    f.get(sums[t])
      .get(ws[0]).get(xps[t]).v128Load(64).f32x4Mul().f32x4Add()
      .get(ws[1]).get(xps[t]).v128Load(80).f32x4Mul().f32x4Add()
      .get(ws[2]).get(xps[t]).v128Load(96).f32x4Mul().f32x4Add()
      .get(ws[3]).get(xps[t]).v128Load(112).f32x4Mul().f32x4Add()
      .set(sums[t]);

    // acc[t] += scale * (hsum(s[t]) - 8.0 * blockSumX[t, b])
    f.get(accs[t]).hsumF32x4ToF64(sums[t]);
    f.get(bp);
    if (t > 0) f.get(bsStride).i32Const(t).i32Mul().i32Add();
    f.f64Load(0).f64Const(8.0).f64Mul().f64Sub().get(scale).f64Mul().f64Add().set(accs[t]);
  }

  f.get(p).i32Const(18).i32Add().set(p);
  f.get(xp0).i32Const(128).i32Add().set(xp0);
  f.get(xp1).i32Const(128).i32Add().set(xp1);
  f.get(xp2).i32Const(128).i32Add().set(xp2);
  f.get(xp3).i32Const(128).i32Add().set(xp3);
  f.get(bp).i32Const(8).i32Add().set(bp);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();

  // Store 4 outputs
  for (let t = 0; t < 4; t++) {
    f.get(2);
    if (t > 0) f.get(oStride).i32Const(t).i32Mul().i32Add();
    f.get(r).i32Const(2).i32Shl().i32Add().get(accs[t]).f32DemoteF64().f32Store(0);
  }

  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();
  return f;
}

/** Builds `mul_mat_q4_1(wPtr, xPtr, outPtr, cols, rows)` with precomputed `blockSumX`. */
function buildMulMatQ4_1(precomputeFnIdx: number): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const b = f.local(I32);
  const numBlocks = f.local(I32);
  const rowBytes = f.local(I32);
  const p = f.local(I32);
  const xp = f.local(I32);
  const bp = f.local(I32);
  const acc = f.local(F64);
  const mask0f = f.local(V128);
  const packed = f.local(V128);
  const nib = f.local(V128);
  const vsum = f.local(V128);
  const tmp16 = f.local(V128);

  f.get(3).i32Const(5).i32ShrU().tee(numBlocks).i32Const(20).i32Mul().set(rowBytes);
  f.get(1).get(numBlocks).i32Const(WASM_BLOCK_SUM_X_OFFSET).call(precomputeFnIdx);
  f.v128Const(MASK_0F_16).set(mask0f);
  f.i32Const(0).set(r);

  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(p);
  f.get(1).set(xp);
  f.i32Const(WASM_BLOCK_SUM_X_OFFSET).set(bp);
  f.f64Const(0).set(acc);
  f.i32Const(0).set(b);

  f.block().loop();
  f.get(b).get(numBlocks).i32GeS().brIf(1);
  f.v128Const(ZERO_16).set(vsum);
  f.get(p).v128Load(4).tee(packed).get(mask0f).v128And().set(nib);
  f.dotU8x16WithX(nib, xp, 0, vsum, tmp16);
  f.get(packed).i32Const(4).i8x16ShrU().set(nib);
  f.dotU8x16WithX(nib, xp, 64, vsum, tmp16);

  // acc += d * sumQ + m * blockSumX[b]
  f.get(acc)
    .hsumF32x4ToF64(vsum).loadF16AsF64(p, 0).f64Mul().f64Add()
    .get(bp).f64Load(0).loadF16AsF64(p, 2).f64Mul().f64Add()
    .set(acc);

  f.get(p).i32Const(20).i32Add().set(p);
  f.get(xp).i32Const(128).i32Add().set(xp);
  f.get(bp).i32Const(8).i32Add().set(bp);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc).f32DemoteF64().f32Store(0);
  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();
  return f;
}

/**
 * Builds `mul_mat_q5_0(wPtr, xPtr, outPtr, cols, rows)` with branchless SIMD `qh` bit-expansion
 * and precomputed `blockSumX` factoring (`sum((q-16)*x) = sum(q*x) - 16*blockSumX[b]`).
 */
function buildMulMatQ5_0(precomputeFnIdx: number): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const b = f.local(I32);
  const numBlocks = f.local(I32);
  const rowBytes = f.local(I32);
  const p = f.local(I32);
  const xp = f.local(I32);
  const bp = f.local(I32);
  const acc = f.local(F64);
  const mask0f = f.local(V128);
  const swizLo = f.local(V128);
  const swizHi = f.local(V128);
  const bitMask = f.local(V128);
  const val16 = f.local(V128);
  const vzero = f.local(V128);
  const qhVec = f.local(V128);
  const packed = f.local(V128);
  const q5 = f.local(V128);
  const vsum = f.local(V128);
  const tmp16 = f.local(V128);

  f.get(3).i32Const(5).i32ShrU().tee(numBlocks).i32Const(22).i32Mul().set(rowBytes);
  f.get(1).get(numBlocks).i32Const(WASM_BLOCK_SUM_X_OFFSET).call(precomputeFnIdx);

  f.v128Const(MASK_0F_16).set(mask0f);
  f.v128Const(SWIZ_LOW16_BITS).set(swizLo);
  f.v128Const(SWIZ_HIGH16_BITS).set(swizHi);
  f.v128Const(BIT_MASKS_8X2).set(bitMask);
  f.v128Const(VAL_16_16).set(val16);
  f.v128Const(ZERO_16).set(vzero);
  f.i32Const(0).set(r);

  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(p);
  f.get(1).set(xp);
  f.i32Const(WASM_BLOCK_SUM_X_OFFSET).set(bp);
  f.f64Const(0).set(acc);
  f.i32Const(0).set(b);

  f.block().loop();
  f.get(b).get(numBlocks).i32GeS().brIf(1);

  f.v128Const(ZERO_16).set(vsum);
  f.get(p).i32Load(2, 0).i32x4Splat().set(qhVec);
  f.get(p).v128Load(6).set(packed);

  // Low 16 elements: (packed & 0x0f) | (((qh >>> j) & 1) << 4)
  f.get(qhVec).get(swizLo).i8x16Swizzle().get(bitMask).v128And().get(vzero).i8x16Ne().get(val16).v128And()
    .get(packed).get(mask0f).v128And().v128Or().set(q5);
  f.dotU8x16WithX(q5, xp, 0, vsum, tmp16);

  // High 16 elements: (packed >> 4) | (((qh >>> (j+16)) & 1) << 4)
  f.get(qhVec).get(swizHi).i8x16Swizzle().get(bitMask).v128And().get(vzero).i8x16Ne().get(val16).v128And()
    .get(packed).i32Const(4).i8x16ShrU().v128Or().set(q5);
  f.dotU8x16WithX(q5, xp, 64, vsum, tmp16);

  f.get(acc)
    .hsumF32x4ToF64(vsum).get(bp).f64Load(0).f64Const(16.0).f64Mul().f64Sub()
    .loadF16AsF64(p, 0).f64Mul().f64Add().set(acc);

  f.get(p).i32Const(22).i32Add().set(p);
  f.get(xp).i32Const(128).i32Add().set(xp);
  f.get(bp).i32Const(8).i32Add().set(bp);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc).f32DemoteF64().f32Store(0);
  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();
  return f;
}

/** Builds `mul_mat_q5_1(wPtr, xPtr, outPtr, cols, rows)` with SIMD `qh` expansion + `blockSumX`. */
function buildMulMatQ5_1(precomputeFnIdx: number): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const b = f.local(I32);
  const numBlocks = f.local(I32);
  const rowBytes = f.local(I32);
  const p = f.local(I32);
  const xp = f.local(I32);
  const bp = f.local(I32);
  const acc = f.local(F64);
  const mask0f = f.local(V128);
  const swizLo = f.local(V128);
  const swizHi = f.local(V128);
  const bitMask = f.local(V128);
  const val16 = f.local(V128);
  const vzero = f.local(V128);
  const qhVec = f.local(V128);
  const packed = f.local(V128);
  const q5 = f.local(V128);
  const vsum = f.local(V128);
  const tmp16 = f.local(V128);

  f.get(3).i32Const(5).i32ShrU().tee(numBlocks).i32Const(24).i32Mul().set(rowBytes);
  f.get(1).get(numBlocks).i32Const(WASM_BLOCK_SUM_X_OFFSET).call(precomputeFnIdx);

  f.v128Const(MASK_0F_16).set(mask0f);
  f.v128Const(SWIZ_LOW16_BITS).set(swizLo);
  f.v128Const(SWIZ_HIGH16_BITS).set(swizHi);
  f.v128Const(BIT_MASKS_8X2).set(bitMask);
  f.v128Const(VAL_16_16).set(val16);
  f.v128Const(ZERO_16).set(vzero);
  f.i32Const(0).set(r);

  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(p);
  f.get(1).set(xp);
  f.i32Const(WASM_BLOCK_SUM_X_OFFSET).set(bp);
  f.f64Const(0).set(acc);
  f.i32Const(0).set(b);

  f.block().loop();
  f.get(b).get(numBlocks).i32GeS().brIf(1);

  f.v128Const(ZERO_16).set(vsum);
  f.get(p).i32Load(4, 0).i32x4Splat().set(qhVec);
  f.get(p).v128Load(8).set(packed);

  f.get(qhVec).get(swizLo).i8x16Swizzle().get(bitMask).v128And().get(vzero).i8x16Ne().get(val16).v128And()
    .get(packed).get(mask0f).v128And().v128Or().set(q5);
  f.dotU8x16WithX(q5, xp, 0, vsum, tmp16);

  f.get(qhVec).get(swizHi).i8x16Swizzle().get(bitMask).v128And().get(vzero).i8x16Ne().get(val16).v128And()
    .get(packed).i32Const(4).i8x16ShrU().v128Or().set(q5);
  f.dotU8x16WithX(q5, xp, 64, vsum, tmp16);

  f.get(acc)
    .hsumF32x4ToF64(vsum).loadF16AsF64(p, 0).f64Mul().f64Add()
    .get(bp).f64Load(0).loadF16AsF64(p, 2).f64Mul().f64Add()
    .set(acc);

  f.get(p).i32Const(24).i32Add().set(p);
  f.get(xp).i32Const(128).i32Add().set(xp);
  f.get(bp).i32Const(8).i32Add().set(bp);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc).f32DemoteF64().f32Store(0);
  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();
  return f;
}

/**
 * Helper: emits `get_scale_min_k4_packed(j, scalesPtr)` inline, leaving `(sc << 8) | m` on stack.
 */
function emitGetScaleMinK4(f: FuncBuilder, jLocal: number, scalesPtr: number, bHiLocal: number): void {
  f.get(jLocal).i32Const(4).i32LtS().if_(I32);
  // (bytes[scales+j] & 63) << 8 | (bytes[scales+j+4] & 63)
  f.get(scalesPtr).get(jLocal).i32Add().i32Load8U(0).i32Const(63).i32And().i32Const(8).i32Shl();
  f.get(scalesPtr).get(jLocal).i32Add().i32Load8U(4).i32Const(63).i32And().i32Or();
  f.else_();
  f.get(scalesPtr).get(jLocal).i32Add().i32Load8U(4).set(bHiLocal);
  // sc = (bHi & 0x0f) | ((bytes[scales+j-4] >> 6) << 4)
  f.get(bHiLocal).i32Const(0x0f).i32And()
    .get(scalesPtr).get(jLocal).i32Add().i32Const(4).i32Sub().i32Load8U(0).i32Const(6).i32ShrU().i32Const(4).i32Shl()
    .i32Or().i32Const(8).i32Shl();
  // m = (bHi >> 4) | ((bytes[scales+j] >> 6) << 4)
  f.get(bHiLocal).i32Const(4).i32ShrU()
    .get(scalesPtr).get(jLocal).i32Add().i32Load8U(0).i32Const(6).i32ShrU().i32Const(4).i32Shl()
    .i32Or().i32Or();
  f.end();
}

/**
 * Builds `mul_mat_q4_k(wPtr, xPtr, outPtr, cols, rows)` (256-elem super-blocks = 144 bytes).
 * Precomputes 32-element `blockSumX` once per call; processes 64 weights per sub-iteration in SIMD.
 */
function buildMulMatQ4_K(precomputeFnIdx: number): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const b = f.local(I32);
  const is = f.local(I32);
  const numBlocks = f.local(I32);
  const rowBytes = f.local(I32);
  const p = f.local(I32);
  const qPtr = f.local(I32);
  const scalesPtr = f.local(I32);
  const xp = f.local(I32);
  const bp = f.local(I32);
  const sm = f.local(I32);
  const bHi = f.local(I32);
  const acc = f.local(F64);
  const d = f.local(F64);
  const dmin = f.local(F64);
  const d1 = f.local(F64);
  const m1 = f.local(F64);
  const d2 = f.local(F64);
  const m2 = f.local(F64);
  const mask0f = f.local(V128);
  const pk0 = f.local(V128);
  const pk1 = f.local(V128);
  const nib = f.local(V128);
  const vsum1 = f.local(V128);
  const vsum2 = f.local(V128);
  const tmp16 = f.local(V128);

  f.get(3).i32Const(8).i32ShrU().tee(numBlocks).i32Const(144).i32Mul().set(rowBytes);
  // Precompute 32-element block sums (cols / 32 blocks)
  f.get(1).get(3).i32Const(5).i32ShrU().i32Const(WASM_BLOCK_SUM_X_OFFSET).call(precomputeFnIdx);
  f.v128Const(MASK_0F_16).set(mask0f);
  f.i32Const(0).set(r);

  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(p);
  f.get(1).set(xp);
  f.i32Const(WASM_BLOCK_SUM_X_OFFSET).set(bp);
  f.f64Const(0).set(acc);
  f.i32Const(0).set(b);

  f.block().loop();
  f.get(b).get(numBlocks).i32GeS().brIf(1);

  f.loadF16AsF64(p, 0).set(d);
  f.loadF16AsF64(p, 2).set(dmin);
  f.get(p).i32Const(4).i32Add().set(scalesPtr);
  f.get(p).i32Const(16).i32Add().set(qPtr);
  f.i32Const(0).set(is);

  f.block().loop();
  f.get(is).i32Const(8).i32GeS().brIf(1);

  emitGetScaleMinK4(f, is, scalesPtr, bHi);
  f.set(sm);
  f.get(d).get(sm).i32Const(8).i32ShrU().f64ConvertI32S().f64Mul().set(d1);
  f.get(dmin).get(sm).i32Const(0xff).i32And().f64ConvertI32S().f64Mul().set(m1);

  f.get(is).i32Const(1).i32Add().set(is);
  emitGetScaleMinK4(f, is, scalesPtr, bHi);
  f.set(sm);
  f.get(d).get(sm).i32Const(8).i32ShrU().f64ConvertI32S().f64Mul().set(d2);
  f.get(dmin).get(sm).i32Const(0xff).i32And().f64ConvertI32S().f64Mul().set(m2);
  f.get(is).i32Const(1).i32Add().set(is);

  // Load 32 packed bytes (64 weights)
  f.get(qPtr).v128Load(0).set(pk0);
  f.get(qPtr).v128Load(16).set(pk1);

  // Low nibbles -> sub-block 1 (x[0..31])
  f.v128Const(ZERO_16).set(vsum1);
  f.get(pk0).get(mask0f).v128And().set(nib);
  f.dotU8x16WithX(nib, xp, 0, vsum1, tmp16);
  f.get(pk1).get(mask0f).v128And().set(nib);
  f.dotU8x16WithX(nib, xp, 64, vsum1, tmp16);

  // High nibbles -> sub-block 2 (x[32..63])
  f.v128Const(ZERO_16).set(vsum2);
  f.get(pk0).i32Const(4).i8x16ShrU().set(nib);
  f.dotU8x16WithX(nib, xp, 128, vsum2, tmp16);
  f.get(pk1).i32Const(4).i8x16ShrU().set(nib);
  f.dotU8x16WithX(nib, xp, 192, vsum2, tmp16);

  f.get(acc)
    .get(d1).hsumF32x4ToF64(vsum1).f64Mul().f64Add()
    .get(m1).get(bp).f64Load(0).f64Mul().f64Sub()
    .get(d2).hsumF32x4ToF64(vsum2).f64Mul().f64Add()
    .get(m2).get(bp).f64Load(8).f64Mul().f64Sub()
    .set(acc);

  f.get(qPtr).i32Const(32).i32Add().set(qPtr);
  f.get(xp).i32Const(256).i32Add().set(xp);
  f.get(bp).i32Const(16).i32Add().set(bp);
  f.br(0).end().end();

  f.get(p).i32Const(144).i32Add().set(p);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc).f32DemoteF64().f32Store(0);
  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();
  return f;
}

/**
 * Builds `mul_mat_q5_k(wPtr, xPtr, outPtr, cols, rows)` (256-elem super-blocks = 176 bytes).
 */
function buildMulMatQ5_K(precomputeFnIdx: number): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const b = f.local(I32);
  const is = f.local(I32);
  const u1 = f.local(I32);
  const u2 = f.local(I32);
  const numBlocks = f.local(I32);
  const rowBytes = f.local(I32);
  const p = f.local(I32);
  const qPtr = f.local(I32);
  const qhPtr = f.local(I32);
  const scalesPtr = f.local(I32);
  const xp = f.local(I32);
  const bp = f.local(I32);
  const sm = f.local(I32);
  const bHi = f.local(I32);
  const acc = f.local(F64);
  const d = f.local(F64);
  const dmin = f.local(F64);
  const d1 = f.local(F64);
  const m1 = f.local(F64);
  const d2 = f.local(F64);
  const m2 = f.local(F64);
  const mask0f = f.local(V128);
  const val16 = f.local(V128);
  const vzero = f.local(V128);
  const pk0 = f.local(V128);
  const pk1 = f.local(V128);
  const hm0 = f.local(V128);
  const hm1 = f.local(V128);
  const u1v = f.local(V128);
  const u2v = f.local(V128);
  const q5 = f.local(V128);
  const vsum1 = f.local(V128);
  const vsum2 = f.local(V128);
  const tmp16 = f.local(V128);

  f.get(3).i32Const(8).i32ShrU().tee(numBlocks).i32Const(176).i32Mul().set(rowBytes);
  f.get(1).get(3).i32Const(5).i32ShrU().i32Const(WASM_BLOCK_SUM_X_OFFSET).call(precomputeFnIdx);
  f.v128Const(MASK_0F_16).set(mask0f);
  f.v128Const(VAL_16_16).set(val16);
  f.v128Const(ZERO_16).set(vzero);
  f.i32Const(0).set(r);

  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(p);
  f.get(1).set(xp);
  f.i32Const(WASM_BLOCK_SUM_X_OFFSET).set(bp);
  f.f64Const(0).set(acc);
  f.i32Const(0).set(b);

  f.block().loop();
  f.get(b).get(numBlocks).i32GeS().brIf(1);

  f.loadF16AsF64(p, 0).set(d);
  f.loadF16AsF64(p, 2).set(dmin);
  f.get(p).i32Const(4).i32Add().set(scalesPtr);
  f.get(p).i32Const(16).i32Add().set(qhPtr);
  f.get(p).i32Const(48).i32Add().set(qPtr);
  f.get(qhPtr).v128Load(0).set(hm0);
  f.get(qhPtr).v128Load(16).set(hm1);
  f.i32Const(1).set(u1);
  f.i32Const(2).set(u2);
  f.i32Const(0).set(is);

  f.block().loop();
  f.get(is).i32Const(8).i32GeS().brIf(1);

  emitGetScaleMinK4(f, is, scalesPtr, bHi);
  f.set(sm);
  f.get(d).get(sm).i32Const(8).i32ShrU().f64ConvertI32S().f64Mul().set(d1);
  f.get(dmin).get(sm).i32Const(0xff).i32And().f64ConvertI32S().f64Mul().set(m1);

  f.get(is).i32Const(1).i32Add().set(is);
  emitGetScaleMinK4(f, is, scalesPtr, bHi);
  f.set(sm);
  f.get(d).get(sm).i32Const(8).i32ShrU().f64ConvertI32S().f64Mul().set(d2);
  f.get(dmin).get(sm).i32Const(0xff).i32And().f64ConvertI32S().f64Mul().set(m2);
  f.get(is).i32Const(1).i32Add().set(is);

  f.get(u1).i8x16Splat().set(u1v);
  f.get(u2).i8x16Splat().set(u2v);
  f.get(qPtr).v128Load(0).set(pk0);
  f.get(qPtr).v128Load(16).set(pk1);

  // Sub-block 1
  f.v128Const(ZERO_16).set(vsum1);
  f.get(hm0).get(u1v).v128And().get(vzero).i8x16Ne().get(val16).v128And()
    .get(pk0).get(mask0f).v128And().v128Or().set(q5);
  f.dotU8x16WithX(q5, xp, 0, vsum1, tmp16);
  f.get(hm1).get(u1v).v128And().get(vzero).i8x16Ne().get(val16).v128And()
    .get(pk1).get(mask0f).v128And().v128Or().set(q5);
  f.dotU8x16WithX(q5, xp, 64, vsum1, tmp16);

  // Sub-block 2
  f.v128Const(ZERO_16).set(vsum2);
  f.get(hm0).get(u2v).v128And().get(vzero).i8x16Ne().get(val16).v128And()
    .get(pk0).i32Const(4).i8x16ShrU().v128Or().set(q5);
  f.dotU8x16WithX(q5, xp, 128, vsum2, tmp16);
  f.get(hm1).get(u2v).v128And().get(vzero).i8x16Ne().get(val16).v128And()
    .get(pk1).i32Const(4).i8x16ShrU().v128Or().set(q5);
  f.dotU8x16WithX(q5, xp, 192, vsum2, tmp16);

  f.get(acc)
    .get(d1).hsumF32x4ToF64(vsum1).f64Mul().f64Add()
    .get(m1).get(bp).f64Load(0).f64Mul().f64Sub()
    .get(d2).hsumF32x4ToF64(vsum2).f64Mul().f64Add()
    .get(m2).get(bp).f64Load(8).f64Mul().f64Sub()
    .set(acc);

  f.get(qPtr).i32Const(32).i32Add().set(qPtr);
  f.get(xp).i32Const(256).i32Add().set(xp);
  f.get(bp).i32Const(16).i32Add().set(bp);
  f.get(u1).i32Const(2).i32Shl().set(u1);
  f.get(u2).i32Const(2).i32Shl().set(u2);
  f.br(0).end().end();

  f.get(p).i32Const(176).i32Add().set(p);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc).f32DemoteF64().f32Store(0);
  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();
  return f;
}

/**
 * Builds `mul_mat_q6_k(wPtr, xPtr, outPtr, cols, rows)` (256-elem super-blocks = 210 bytes).
 * Processes each 16-element sub-group with constant int8 scale across all 16 elements in SIMD!
 */
function buildMulMatQ6_K(): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const b = f.local(I32);
  const half = f.local(I32);
  const chunk = f.local(I32);
  const numBlocks = f.local(I32);
  const rowBytes = f.local(I32);
  const p = f.local(I32);
  const qlPtr = f.local(I32);
  const qhPtr = f.local(I32);
  const scPtr = f.local(I32);
  const xp = f.local(I32);
  const acc = f.local(F64);
  const blockSum = f.local(F64);
  const mask0f = f.local(V128);
  const mask03 = f.local(V128);
  const val32 = f.local(V128);
  const ql0 = f.local(V128);
  const ql1 = f.local(V128);
  const qh = f.local(V128);
  const q6 = f.local(V128);
  const vsum = f.local(V128);
  const tmp16 = f.local(V128);

  f.get(3).i32Const(8).i32ShrU().tee(numBlocks).i32Const(210).i32Mul().set(rowBytes);
  f.v128Const(MASK_0F_16).set(mask0f);
  f.v128Const(MASK_03_16).set(mask03);
  f.v128Const(VAL_32_16).set(val32);
  f.i32Const(0).set(r);

  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(p);
  f.get(1).set(xp);
  f.f64Const(0).set(acc);
  f.i32Const(0).set(b);

  f.block().loop();
  f.get(b).get(numBlocks).i32GeS().brIf(1);

  f.f64Const(0).set(blockSum);
  f.get(p).set(qlPtr);
  f.get(p).i32Const(128).i32Add().set(qhPtr);
  f.get(p).i32Const(192).i32Add().set(scPtr);
  f.i32Const(0).set(half);

  f.block().loop();
  f.get(half).i32Const(2).i32GeS().brIf(1);

  f.i32Const(0).set(chunk);
  f.block().loop();
  f.get(chunk).i32Const(2).i32GeS().brIf(1);

  // chunk=0 -> l=0..15 (is=0); chunk=1 -> l=16..31 (is=1)
  f.get(qlPtr).get(chunk).i32Const(4).i32Shl().i32Add().v128Load(0).set(ql0);
  f.get(qlPtr).get(chunk).i32Const(4).i32Shl().i32Add().v128Load(32).set(ql1);
  f.get(qhPtr).get(chunk).i32Const(4).i32Shl().i32Add().v128Load(0).set(qh);

  // q1 = ((ql0 & 0x0f) | (((qh >> 0) & 3) << 4)) - 32 -> x[n + chunk*16]
  f.get(ql0).get(mask0f).v128And()
    .get(qh).get(mask03).v128And().i32Const(4).i8x16Shl().v128Or()
    .get(val32).i8x16Sub().set(q6);
  f.v128Const(ZERO_16).set(vsum);
  f.dotS8x16WithX(q6, xp, 0, vsum, tmp16);
  f.get(blockSum)
    .hsumF32x4ToF64(vsum)
    .get(scPtr).get(chunk).i32Add().i32Load8S(0).f64ConvertI32S().f64Mul().f64Add().set(blockSum);

  // q2 = ((ql1 & 0x0f) | (((qh >> 2) & 3) << 4)) - 32 -> x[n + 32 + chunk*16]
  f.get(ql1).get(mask0f).v128And()
    .get(qh).i32Const(2).i8x16ShrU().get(mask03).v128And().i32Const(4).i8x16Shl().v128Or()
    .get(val32).i8x16Sub().set(q6);
  f.v128Const(ZERO_16).set(vsum);
  f.dotS8x16WithX(q6, xp, 128, vsum, tmp16);
  f.get(blockSum)
    .hsumF32x4ToF64(vsum)
    .get(scPtr).get(chunk).i32Add().i32Load8S(2).f64ConvertI32S().f64Mul().f64Add().set(blockSum);

  // q3 = ((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32 -> x[n + 64 + chunk*16]
  f.get(ql0).i32Const(4).i8x16ShrU()
    .get(qh).i32Const(4).i8x16ShrU().get(mask03).v128And().i32Const(4).i8x16Shl().v128Or()
    .get(val32).i8x16Sub().set(q6);
  f.v128Const(ZERO_16).set(vsum);
  f.dotS8x16WithX(q6, xp, 256, vsum, tmp16);
  f.get(blockSum)
    .hsumF32x4ToF64(vsum)
    .get(scPtr).get(chunk).i32Add().i32Load8S(4).f64ConvertI32S().f64Mul().f64Add().set(blockSum);

  // q4 = ((ql1 >> 4) | (((qh >> 6) & 3) << 4)) - 32 -> x[n + 96 + chunk*16]
  f.get(ql1).i32Const(4).i8x16ShrU()
    .get(qh).i32Const(6).i8x16ShrU().get(mask03).v128And().i32Const(4).i8x16Shl().v128Or()
    .get(val32).i8x16Sub().set(q6);
  f.v128Const(ZERO_16).set(vsum);
  f.dotS8x16WithX(q6, xp, 384, vsum, tmp16);
  f.get(blockSum)
    .hsumF32x4ToF64(vsum)
    .get(scPtr).get(chunk).i32Add().i32Load8S(6).f64ConvertI32S().f64Mul().f64Add().set(blockSum);

  f.get(xp).i32Const(64).i32Add().set(xp);
  f.get(chunk).i32Const(1).i32Add().set(chunk);
  f.br(0).end().end();

  // Advance xp past the remaining 96 floats of this 128-element half (we advanced 2*16=32 floats)
  f.get(xp).i32Const(384).i32Add().set(xp);
  f.get(qlPtr).i32Const(64).i32Add().set(qlPtr);
  f.get(qhPtr).i32Const(32).i32Add().set(qhPtr);
  f.get(scPtr).i32Const(8).i32Add().set(scPtr);
  f.get(half).i32Const(1).i32Add().set(half);
  f.br(0).end().end();

  f.get(acc).get(blockSum).loadF16AsF64(p, 208).f64Mul().f64Add().set(acc);
  f.get(p).i32Const(210).i32Add().set(p);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc).f32DemoteF64().f32Store(0);
  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();
  return f;
}

/**
 * Builds lock-free strict F32-activation `mul_mat_q2_k(wPtr, xPtr, outPtr, cols, rows)`
 * (256-elem super-blocks = 84 bytes: scales[16], qs[64], d f16, dmin f16).
 * Accumulates in F64 with zero shared scratch -> 100% multi-core parallel-safe.
 */
function buildMulMatQ2_K(): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const b = f.local(I32);
  const numBlocks = f.local(I32);
  const rowBytes = f.local(I32);
  const p = f.local(I32);
  const xp = f.local(I32);
  const sc = f.local(I32);
  const acc = f.local(F64);
  const d = f.local(F64);
  const dmin = f.local(F64);

  const mask03 = f.local(V128);
  const q0 = f.local(V128);
  const q1 = f.local(V128);
  const qv = f.local(V128);
  const vsum = f.local(V128);
  const vsumX = f.local(V128);
  const tmp16 = f.local(V128);

  f.get(3).i32Const(8).i32ShrU().tee(numBlocks).i32Const(84).i32Mul().set(rowBytes);
  f.i32Const(0x03).i8x16Splat().set(mask03);
  f.i32Const(0).set(r);

  const emitSub16 = (qSrc: number, shift: number, scOff: number) => {
    f.get(qSrc);
    if (shift > 0) f.i32Const(shift).i8x16ShrU();
    f.get(mask03).v128And().set(qv);
    f.v128Const(ZERO_16).set(vsum);
    f.dotU8x16WithX(qv, xp, 0, vsum, tmp16);
    // Sum 16 floats at xp for exact dmin subtraction
    f.get(xp).v128Load(0).get(xp).v128Load(16).f32x4Add()
      .get(xp).v128Load(32).f32x4Add().get(xp).v128Load(48).f32x4Add().set(vsumX);
    f.get(p).i32Load8U(scOff).set(sc);
    f.get(acc)
      .get(d).get(sc).i32Const(0x0f).i32And().f64ConvertI32S().f64Mul()
      .hsumF32x4ToF64(vsum).f64Mul().f64Add()
      .get(dmin).get(sc).i32Const(4).i32ShrU().f64ConvertI32S().f64Mul()
      .hsumF32x4ToF64(vsumX).f64Mul().f64Sub()
      .set(acc);
    f.get(xp).i32Const(64).i32Add().set(xp);
  };

  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(p);
  f.get(1).set(xp);
  f.f64Const(0).set(acc);
  f.i32Const(0).set(b);

  f.block().loop();
  f.get(b).get(numBlocks).i32GeS().brIf(1);
  f.loadF16AsF64(p, 80).set(d);
  f.loadF16AsF64(p, 82).set(dmin);

  for (let h = 0; h < 2; h++) {
    f.get(p).v128Load(16 + h * 32).set(q0);
    f.get(p).v128Load(32 + h * 32).set(q1);
    for (let s = 0; s < 4; s++) {
      emitSub16(q0, s * 2, h * 8 + s * 2);
      emitSub16(q1, s * 2, h * 8 + s * 2 + 1);
    }
  }

  f.get(p).i32Const(84).i32Add().set(p);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc).f32DemoteF64().f32Store(0);
  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();
  return f;
}

/**
 * Builds lock-free strict F32-activation `mul_mat_q3_k(wPtr, xPtr, outPtr, cols, rows)`
 * (256-elem super-blocks = 110 bytes: hmask[32], qs[64], scales[12], d f16).
 * Accumulates in F64 with zero shared scratch -> 100% multi-core parallel-safe.
 */
function buildMulMatQ3_K(): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const b = f.local(I32);
  const numBlocks = f.local(I32);
  const rowBytes = f.local(I32);
  const p = f.local(I32);
  const xp = f.local(I32);
  const a0 = f.local(I32);
  const a1 = f.local(I32);
  const tmp = f.local(I32);
  const u0 = f.local(I32);
  const u1 = f.local(I32);
  const u2 = f.local(I32);
  const u3 = f.local(I32);
  const acc = f.local(F64);
  const blockSum = f.local(F64);

  const mask03 = f.local(V128);
  const mask04 = f.local(V128);
  const val4 = f.local(V128);
  const hm0 = f.local(V128);
  const hm1 = f.local(V128);
  const q0 = f.local(V128);
  const q1 = f.local(V128);
  const qv = f.local(V128);
  const vsum = f.local(V128);
  const tmp16 = f.local(V128);

  f.get(3).i32Const(8).i32ShrU().tee(numBlocks).i32Const(110).i32Mul().set(rowBytes);
  f.i32Const(0x03).i8x16Splat().set(mask03);
  f.i32Const(0x04).i8x16Splat().set(mask04);
  f.i32Const(4).i8x16Splat().set(val4);
  f.i32Const(0).set(r);

  const pushScaleF64 = (scIdx: number) => {
    const uLoc = scIdx < 4 ? u0 : scIdx < 8 ? u1 : scIdx < 12 ? u2 : u3;
    const byteIdx = scIdx & 3;
    f.get(uLoc);
    if (byteIdx > 0) f.i32Const(byteIdx * 8).i32ShrU();
    if (byteIdx < 3) f.i32Const(0xff).i32And();
    f.i32Const(32).i32Sub().f64ConvertI32S();
  };

  const emitSub16 = (qSrc: number, hmSrc: number, shift: number, bitIdx: number, scIdx: number) => {
    f.get(qSrc);
    if (shift > 0) f.i32Const(shift).i8x16ShrU();
    f.get(mask03).v128And();
    f.get(hmSrc);
    if (bitIdx < 2) {
      f.i32Const(2 - bitIdx).i8x16Shl();
    } else if (bitIdx > 2) {
      f.i32Const(bitIdx - 2).i8x16ShrU();
    }
    f.get(mask04).v128And().v128Or().get(val4).i8x16Sub().set(qv);
    f.v128Const(ZERO_16).set(vsum);
    f.dotS8x16WithX(qv, xp, 0, vsum, tmp16);
    f.get(blockSum).hsumF32x4ToF64(vsum);
    pushScaleF64(scIdx);
    f.f64Mul().f64Add().set(blockSum);
    f.get(xp).i32Const(64).i32Add().set(xp);
  };

  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(p);
  f.get(1).set(xp);
  f.f64Const(0).set(acc);
  f.i32Const(0).set(b);

  f.block().loop();
  f.get(b).get(numBlocks).i32GeS().brIf(1);

  // SWAR unpack 12-byte packed 6-bit Q3_K scales into u0..u3 (16 unsigned 6-bit values; -32 gives signed)
  f.get(p).i32Load(96).set(a0);
  f.get(p).i32Load(100).set(a1);
  f.get(p).i32Load(104).set(tmp);
  f.get(a0).i32Const(0x0f0f0f0f).i32And()
    .get(tmp).i32Const(0x03030303).i32And().i32Const(4).i32Shl().i32Or().set(u0);
  f.get(a1).i32Const(0x0f0f0f0f).i32And()
    .get(tmp).i32Const(2).i32ShrU().i32Const(0x03030303).i32And().i32Const(4).i32Shl().i32Or().set(u1);
  f.get(a0).i32Const(4).i32ShrU().i32Const(0x0f0f0f0f).i32And()
    .get(tmp).i32Const(4).i32ShrU().i32Const(0x03030303).i32And().i32Const(4).i32Shl().i32Or().set(u2);
  f.get(a1).i32Const(4).i32ShrU().i32Const(0x0f0f0f0f).i32And()
    .get(tmp).i32Const(6).i32ShrU().i32Const(0x03030303).i32And().i32Const(4).i32Shl().i32Or().set(u3);

  f.get(p).v128Load(0).set(hm0);
  f.get(p).v128Load(16).set(hm1);
  f.f64Const(0).set(blockSum);

  for (let h = 0; h < 2; h++) {
    f.get(p).v128Load(32 + h * 32).set(q0);
    f.get(p).v128Load(48 + h * 32).set(q1);
    for (let s = 0; s < 4; s++) {
      const bitIdx = h * 4 + s;
      emitSub16(q0, hm0, s * 2, bitIdx, h * 8 + s * 2);
      emitSub16(q1, hm1, s * 2, bitIdx, h * 8 + s * 2 + 1);
    }
  }

  f.get(acc).get(blockSum).loadF16AsF64(p, 108).f64Mul().f64Add().set(acc);
  f.get(p).i32Const(110).i32Add().set(p);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc).f32DemoteF64().f32Store(0);
  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();
  return f;
}

/**
 * Builds `rms_norm_f32(xPtr, wPtr, outPtr, n, eps)` with 4-way unrolled IEEE-754 F64
 * sum-of-squares (exact VT-46 precision, breaking F64 latency chains 4x) and SIMD128 scaling.
 */
function buildRmsNormF32(): FuncBuilder {
  // params: xPtr(0), wPtr(1), outPtr(2), n(3), eps(4: f64)
  const f = new FuncBuilder([I32, I32, I32, I32, F64]);
  const i = f.local(I32);
  const xp = f.local(I32);
  const s0 = f.local(F64);
  const s1 = f.local(F64);
  const s2 = f.local(F64);
  const s3 = f.local(F64);
  const sumSq = f.local(F64);
  const v = f.local(F64);
  const scale = f.local(F32);
  const vscale = f.local(V128);
  const n4 = f.local(I32);

  f.get(3).i32Const(-4).i32And().set(n4);
  f.f64Const(0).set(s0);
  f.f64Const(0).set(s1);
  f.f64Const(0).set(s2);
  f.f64Const(0).set(s3);
  f.get(0).set(xp);
  f.i32Const(0).set(i);

  // 4-way unrolled IEEE-754 F64 sum-of-squares (hides F64 add latency while preserving F64 precision)
  f.block().loop();
  f.get(i).get(n4).i32GeS().brIf(1);
  f.get(xp).f32Load(0).f64PromoteF32().tee(v).get(v).f64Mul().get(s0).f64Add().set(s0);
  f.get(xp).f32Load(4).f64PromoteF32().tee(v).get(v).f64Mul().get(s1).f64Add().set(s1);
  f.get(xp).f32Load(8).f64PromoteF32().tee(v).get(v).f64Mul().get(s2).f64Add().set(s2);
  f.get(xp).f32Load(12).f64PromoteF32().tee(v).get(v).f64Mul().get(s3).f64Add().set(s3);
  f.get(xp).i32Const(16).i32Add().set(xp);
  f.get(i).i32Const(4).i32Add().set(i);
  f.br(0).end().end();

  f.get(s0).get(s1).f64Add().get(s2).get(s3).f64Add().f64Add().set(sumSq);

  // Scalar tail
  f.block().loop();
  f.get(i).get(3).i32GeS().brIf(1);
  f.get(xp).f32Load(0).f64PromoteF32().tee(v).get(v).f64Mul().get(sumSq).f64Add().set(sumSq);
  f.get(xp).i32Const(4).i32Add().set(xp);
  f.get(i).i32Const(1).i32Add().set(i);
  f.br(0).end().end();

  // scale = f32(1.0 / sqrt(sumSq / n + eps))
  f.f64Const(1.0)
    .get(sumSq).get(3).f64ConvertI32S().f64Div().get(4).f64Add().f64Sqrt()
    .f64Div().f32DemoteF64().tee(scale).f32x4Splat().set(vscale);

  f.i32Const(0).set(i);

  // SIMD pass: out[i..i+3] = (x * scale) * w
  f.block().loop();
  f.get(i).get(n4).i32GeS().brIf(1);
  f.get(2).get(i).i32Const(2).i32Shl().i32Add()
    .get(0).get(i).i32Const(2).i32Shl().i32Add().v128Load(0).get(vscale).f32x4Mul()
    .get(1).get(i).i32Const(2).i32Shl().i32Add().v128Load(0).f32x4Mul()
    .v128Store(0);
  f.get(i).i32Const(4).i32Add().set(i);
  f.br(0).end().end();

  // Scalar tail
  f.block().loop();
  f.get(i).get(3).i32GeS().brIf(1);
  f.get(2).get(i).i32Const(2).i32Shl().i32Add()
    .get(0).get(i).i32Const(2).i32Shl().i32Add().f32Load(0).get(scale).f32Mul()
    .get(1).get(i).i32Const(2).i32Shl().i32Add().f32Load(0).f32Mul()
    .f32Store(0);
  f.get(i).i32Const(1).i32Add().set(i);
  f.br(0).end().end();

  return f;
}

/**
 * Builds `zero_bytes(dstPtr, byteLen)` via WASM bulk `memory.fill(0)` for zero-alloc buffer clears.
 */
function buildZeroBytes(): FuncBuilder {
  const f = new FuncBuilder([I32, I32]);
  f.get(0).i32Const(0).get(1).memoryFill();
  return f;
}

/**
 * Builds `quantize_kv_page_f32_to_q8(srcF32Ptr, dstQ8Ptr, dstScalePtr, totalVectors, headDim, errOutPtr)`.
 * Vectorized 16-wide SIMD128 KV-cache page quantizer (eliminates JS hot-to-cold latency spikes on
 * 7B–32B & 30B-A3B models) while tracking exact SIMD `maxErr` and `maxSig` for vRAM challenger admission.
 * Requires `headDim >= 16 && (headDim & 15) === 0`.
 */
function buildQuantizeKvPageF32ToQ8(): FuncBuilder {
  // params: srcF32Ptr(0), dstQ8Ptr(1), dstScalePtr(2), totalVectors(3), headDim(4), errOutPtr(5)
  const f = new FuncBuilder([I32, I32, I32, I32, I32, I32]);
  const vec = f.local(I32);
  const d = f.local(I32);
  const headBytesF32 = f.local(I32);
  const srcBase = f.local(I32);
  const dstBase = f.local(I32);
  const sp = f.local(I32);
  const dp = f.local(I32);
  const vmax0 = f.local(V128);
  const vmax1 = f.local(V128);
  const vMax = f.local(F32);
  const scale = f.local(F32);
  const inv = f.local(F32);
  const globalMaxSig = f.local(F32);
  const globalMaxErrV = f.local(V128);
  const vInv = f.local(V128);
  const vScale = f.local(V128);
  const vNeg127 = f.local(V128);
  const vPos127 = f.local(V128);
  const x0 = f.local(V128);
  const qf = f.local(V128);
  const q0 = f.local(V128);
  const q1 = f.local(V128);
  const q2 = f.local(V128);
  const q3 = f.local(V128);

  f.get(4).i32Const(2).i32Shl().set(headBytesF32);
  f.f32Const(0).set(globalMaxSig);
  f.f32Const(0).f32x4Splat().set(globalMaxErrV);
  f.f32Const(-127.0).f32x4Splat().set(vNeg127);
  f.f32Const(127.0).f32x4Splat().set(vPos127);
  f.get(0).set(srcBase);
  f.get(1).set(dstBase);
  f.i32Const(0).set(vec);

  const emitQuantChunk = (off: number, qiLoc: number) => {
    f.get(sp).v128Load(off).tee(x0)
      .get(vInv).f32x4Mul().f32x4Nearest()
      .get(vNeg127).f32x4Pmax()
      .get(vPos127).f32x4Pmin()
      .tee(qf)
      .i32x4TruncSatF32x4S().set(qiLoc);
    // Track exact reconstruction error |x - q * scale|
    f.get(globalMaxErrV)
      .get(x0).get(qf).get(vScale).f32x4Mul().f32x4Sub().f32x4Abs()
      .f32x4Pmax().set(globalMaxErrV);
  };

  f.block().loop();
  f.get(vec).get(3).i32GeS().brIf(1);

  // Pass 1: find maxAbs across `headDim` floats (16 floats per iteration)
  f.f32Const(0).f32x4Splat().tee(vmax0).set(vmax1);
  f.get(srcBase).set(sp);
  f.i32Const(0).set(d);
  f.block().loop();
  f.get(d).get(4).i32GeS().brIf(1);
  f.get(vmax0)
    .get(sp).v128Load(0).f32x4Abs().f32x4Pmax()
    .get(sp).v128Load(16).f32x4Abs().f32x4Pmax()
    .set(vmax0);
  f.get(vmax1)
    .get(sp).v128Load(32).f32x4Abs().f32x4Pmax()
    .get(sp).v128Load(48).f32x4Abs().f32x4Pmax()
    .set(vmax1);
  f.get(sp).i32Const(64).i32Add().set(sp);
  f.get(d).i32Const(16).i32Add().set(d);
  f.br(0).end().end();

  f.get(vmax0).get(vmax1).f32x4Pmax().set(vmax0);
  f.get(vmax0).f32x4Extract(0)
    .get(vmax0).f32x4Extract(1).f32Max()
    .get(vmax0).f32x4Extract(2).f32Max()
    .get(vmax0).f32x4Extract(3).f32Max()
    .set(vMax);

  f.get(globalMaxSig).get(vMax).f32Max().set(globalMaxSig);

  f.get(vMax).f32Const(0).f32Gt().if_();
  f.get(vMax).f32Const(127.0).f32Div().set(scale);
  f.f32Const(127.0).get(vMax).f32Div().set(inv);
  f.else_();
  f.f32Const(0).set(scale);
  f.f32Const(0).set(inv);
  f.end();

  // Store per-vector scale
  f.get(2).get(vec).i32Const(2).i32Shl().i32Add().get(scale).f32Store(0);

  f.get(inv).f32x4Splat().set(vInv);
  f.get(scale).f32x4Splat().set(vScale);

  // Pass 2: quantize + narrow 16 f32 -> 16 i8 + accumulate SIMD maxErr
  f.get(srcBase).set(sp);
  f.get(dstBase).set(dp);
  f.i32Const(0).set(d);
  f.block().loop();
  f.get(d).get(4).i32GeS().brIf(1);
  emitQuantChunk(0, q0);
  emitQuantChunk(16, q1);
  emitQuantChunk(32, q2);
  emitQuantChunk(48, q3);
  f.get(dp)
    .get(q0).get(q1).i16x8NarrowI32x4S()
    .get(q2).get(q3).i16x8NarrowI32x4S()
    .i8x16NarrowI16x8S()
    .v128Store(0);
  f.get(sp).i32Const(64).i32Add().set(sp);
  f.get(dp).i32Const(16).i32Add().set(dp);
  f.get(d).i32Const(16).i32Add().set(d);
  f.br(0).end().end();

  f.get(srcBase).get(headBytesF32).i32Add().set(srcBase);
  f.get(dstBase).get(4).i32Add().set(dstBase);
  f.get(vec).i32Const(1).i32Add().set(vec);
  f.br(0).end().end();

  // Write [maxErr, maxSig] to errOutPtr
  f.get(5)
    .get(globalMaxErrV).f32x4Extract(0)
    .get(globalMaxErrV).f32x4Extract(1).f32Max()
    .get(globalMaxErrV).f32x4Extract(2).f32Max()
    .get(globalMaxErrV).f32x4Extract(3).f32Max()
    .f32Store(0);
  f.get(5).get(globalMaxSig).f32Store(4);

  return f;
}

/**
 * Builds `dot_f32(aPtr, bPtr, n) -> f64` SIMD helper for hot F32 attention.
 */
function buildDotF32(): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32], [F64]);
  const i = f.local(I32);
  const n4 = f.local(I32);
  const vsum = f.local(V128);
  const acc = f.local(F64);

  f.get(2).i32Const(-4).i32And().set(n4);
  f.v128Const(ZERO_16).set(vsum);
  f.i32Const(0).set(i);

  f.block().loop();
  f.get(i).get(n4).i32GeS().brIf(1);
  f.get(0).get(i).i32Const(2).i32Shl().i32Add().v128Load(0)
    .get(1).get(i).i32Const(2).i32Shl().i32Add().v128Load(0)
    .f32x4Mul().get(vsum).f32x4Add().set(vsum);
  f.get(i).i32Const(4).i32Add().set(i);
  f.br(0).end().end();

  f.hsumF32x4ToF64(vsum).set(acc);

  f.block().loop();
  f.get(i).get(2).i32GeS().brIf(1);
  f.get(acc)
    .get(0).get(i).i32Const(2).i32Shl().i32Add().f32Load(0).f64PromoteF32()
    .get(1).get(i).i32Const(2).i32Shl().i32Add().f32Load(0).f64PromoteF32()
    .f64Mul().f64Add().set(acc);
  f.get(i).i32Const(1).i32Add().set(i);
  f.br(0).end().end();

  f.get(acc);
  return f;
}

/**
 * Builds `dot_q8_f32(q8Ptr, xPtr, n) -> f64` for vRAM Cold-Page Q8 KV-cache attention!
 * Computes dot product of int8 vector `q8[0..n]` with F32 vector `x[0..n]` in SIMD.
 */
function buildDotQ8F32(): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32], [F64]);
  const i = f.local(I32);
  const n16 = f.local(I32);
  const qv = f.local(V128);
  const vsum = f.local(V128);
  const tmp16 = f.local(V128);
  const acc = f.local(F64);
  const xp = f.local(I32);

  f.get(2).i32Const(-16).i32And().set(n16);
  f.v128Const(ZERO_16).set(vsum);
  f.get(1).set(xp);
  f.i32Const(0).set(i);

  f.block().loop();
  f.get(i).get(n16).i32GeS().brIf(1);
  f.get(0).get(i).i32Add().v128Load(0).set(qv);
  f.dotS8x16WithX(qv, xp, 0, vsum, tmp16);
  f.get(xp).i32Const(64).i32Add().set(xp);
  f.get(i).i32Const(16).i32Add().set(i);
  f.br(0).end().end();

  f.hsumF32x4ToF64(vsum).set(acc);

  f.block().loop();
  f.get(i).get(2).i32GeS().brIf(1);
  f.get(acc)
    .get(0).get(i).i32Add().i32Load8S(0).f64ConvertI32S()
    .get(1).get(i).i32Const(2).i32Shl().i32Add().f32Load(0).f64PromoteF32()
    .f64Mul().f64Add().set(acc);
  f.get(i).i32Const(1).i32Add().set(i);
  f.br(0).end().end();

  f.get(acc);
  return f;
}

/**
 * Builds `axpy_f32(dstPtr, srcPtr, n, scale)` (dst[i] += scale * src[i]) in SIMD.
 */
function buildAxpyF32(): FuncBuilder {
  // params: dstPtr(0), srcPtr(1), n(2), scale(3: f32)
  const f = new FuncBuilder([I32, I32, I32, F32]);
  const i = f.local(I32);
  const n4 = f.local(I32);
  const vs = f.local(V128);

  f.get(2).i32Const(-4).i32And().set(n4);
  f.get(3).f32x4Splat().set(vs);
  f.i32Const(0).set(i);

  f.block().loop();
  f.get(i).get(n4).i32GeS().brIf(1);
  f.get(0).get(i).i32Const(2).i32Shl().i32Add()
    .get(0).get(i).i32Const(2).i32Shl().i32Add().v128Load(0)
    .get(1).get(i).i32Const(2).i32Shl().i32Add().v128Load(0).get(vs).f32x4Mul()
    .f32x4Add().v128Store(0);
  f.get(i).i32Const(4).i32Add().set(i);
  f.br(0).end().end();

  f.block().loop();
  f.get(i).get(2).i32GeS().brIf(1);
  f.get(0).get(i).i32Const(2).i32Shl().i32Add()
    .get(0).get(i).i32Const(2).i32Shl().i32Add().f32Load(0)
    .get(1).get(i).i32Const(2).i32Shl().i32Add().f32Load(0).get(3).f32Mul()
    .emit(0x92).f32Store(0);
  f.get(i).i32Const(1).i32Add().set(i);
  f.br(0).end().end();
  return f;
}

/**
 * Builds `axpy_q8_f32(dstPtr, q8Ptr, n, scale)` (dst[i] += scale * q8[i]) for vRAM cold Q8 V accumulation.
 */
function buildAxpyQ8F32(): FuncBuilder {
  // params: dstPtr(0), q8Ptr(1), n(2), scale(3: f32)
  const f = new FuncBuilder([I32, I32, I32, F32]);
  const i = f.local(I32);
  const n16 = f.local(I32);
  const dp = f.local(I32);
  const vs = f.local(V128);
  const qv = f.local(V128);
  const tmp16 = f.local(V128);

  f.get(2).i32Const(-16).i32And().set(n16);
  f.get(3).f32x4Splat().set(vs);
  f.get(0).set(dp);
  f.i32Const(0).set(i);

  f.block().loop();
  f.get(i).get(n16).i32GeS().brIf(1);
  f.get(1).get(i).i32Add().v128Load(0).set(qv);

  // Low 8 bytes
  f.get(qv).i16x8ExtendLowI8x16S().set(tmp16);
  f.get(dp)
    .get(dp).v128Load(0)
    .get(tmp16).i32x4ExtendLowI16x8S().f32x4ConvertI32x4S().get(vs).f32x4Mul().f32x4Add().v128Store(0);
  f.get(dp)
    .get(dp).v128Load(16)
    .get(tmp16).i32x4ExtendHighI16x8S().f32x4ConvertI32x4S().get(vs).f32x4Mul().f32x4Add().v128Store(16);

  // High 8 bytes
  f.get(qv).i16x8ExtendHighI8x16S().set(tmp16);
  f.get(dp)
    .get(dp).v128Load(32)
    .get(tmp16).i32x4ExtendLowI16x8S().f32x4ConvertI32x4S().get(vs).f32x4Mul().f32x4Add().v128Store(32);
  f.get(dp)
    .get(dp).v128Load(48)
    .get(tmp16).i32x4ExtendHighI16x8S().f32x4ConvertI32x4S().get(vs).f32x4Mul().f32x4Add().v128Store(48);

  f.get(dp).i32Const(64).i32Add().set(dp);
  f.get(i).i32Const(16).i32Add().set(i);
  f.br(0).end().end();

  // Scalar tail
  f.block().loop();
  f.get(i).get(2).i32GeS().brIf(1);
  f.get(0).get(i).i32Const(2).i32Shl().i32Add()
    .get(0).get(i).i32Const(2).i32Shl().i32Add().f32Load(0)
    .get(1).get(i).i32Add().i32Load8S(0).emit(0xb2).get(3).f32Mul()
    .emit(0x92).f32Store(0);
  f.get(i).i32Const(1).i32Add().set(i);
  f.br(0).end().end();
  return f;
}

/**
 * Builds `scale_add_f32(dstPtr, srcPtr, n, rescale, addScale)` (dst[i] = dst[i] * rescale + src[i] * addScale).
 */
function buildScaleAddF32(): FuncBuilder {
  // params: dstPtr(0), srcPtr(1), n(2), rescale(3: f32), addScale(4: f32)
  const f = new FuncBuilder([I32, I32, I32, F32, F32]);
  const i = f.local(I32);
  const n4 = f.local(I32);
  const vr = f.local(V128);
  const va = f.local(V128);

  f.get(2).i32Const(-4).i32And().set(n4);
  f.get(3).f32x4Splat().set(vr);
  f.get(4).f32x4Splat().set(va);
  f.i32Const(0).set(i);

  f.block().loop();
  f.get(i).get(n4).i32GeS().brIf(1);
  f.get(0).get(i).i32Const(2).i32Shl().i32Add()
    .get(0).get(i).i32Const(2).i32Shl().i32Add().v128Load(0).get(vr).f32x4Mul()
    .get(1).get(i).i32Const(2).i32Shl().i32Add().v128Load(0).get(va).f32x4Mul()
    .f32x4Add().v128Store(0);
  f.get(i).i32Const(4).i32Add().set(i);
  f.br(0).end().end();

  f.block().loop();
  f.get(i).get(2).i32GeS().brIf(1);
  f.get(0).get(i).i32Const(2).i32Shl().i32Add()
    .get(0).get(i).i32Const(2).i32Shl().i32Add().f32Load(0).get(3).f32Mul()
    .get(1).get(i).i32Const(2).i32Shl().i32Add().f32Load(0).get(4).f32Mul()
    .emit(0x92).f32Store(0);
  f.get(i).i32Const(1).i32Add().set(i);
  f.br(0).end().end();
  return f;
}

/**
 * Builds `scale_f32(dstPtr, n, scale)` (dst[i] *= scale) in SIMD.
 */
function buildScaleF32(): FuncBuilder {
  const f = new FuncBuilder([I32, I32, F32]);
  const i = f.local(I32);
  const n4 = f.local(I32);
  const vs = f.local(V128);

  f.get(1).i32Const(-4).i32And().set(n4);
  f.get(2).f32x4Splat().set(vs);
  f.i32Const(0).set(i);

  f.block().loop();
  f.get(i).get(n4).i32GeS().brIf(1);
  f.get(0).get(i).i32Const(2).i32Shl().i32Add()
    .get(0).get(i).i32Const(2).i32Shl().i32Add().v128Load(0).get(vs).f32x4Mul().v128Store(0);
  f.get(i).i32Const(4).i32Add().set(i);
  f.br(0).end().end();

  f.block().loop();
  f.get(i).get(1).i32GeS().brIf(1);
  f.get(0).get(i).i32Const(2).i32Shl().i32Add()
    .get(0).get(i).i32Const(2).i32Shl().i32Add().f32Load(0).get(2).f32Mul().f32Store(0);
  f.get(i).i32Const(1).i32Add().set(i);
  f.br(0).end().end();
  return f;
}

/**
 * quantize_f32_to_q8_0(xPtr, q8Ptr, cols):
 * Packs `cols` (multiple of 32) f32 activations into 48-byte aligned Q8_0_ACT blocks:
 *   +0: f32 d_x, +4: i32 sum_lo, +8: i32 sum_hi, +12: i32 sum_all, +16: i8[16] lo, +32: i8[16] hi
 */
function buildQuantizeF32ToQ8_0(): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32]); // 0:xPtr, 1:q8Ptr, 2:cols
  const nBlocks = f.local(I32);
  const b = f.local(I32);
  const xp = f.local(I32);
  const qp = f.local(I32);
  const amax = f.local(F32);
  const d = f.local(F32);
  const invD = f.local(F32);
  const vmax = f.local(V128);
  const vinv = f.local(V128);
  const q0 = f.local(V128);
  const q1 = f.local(V128);
  const q2 = f.local(V128);
  const q3 = f.local(V128);
  const sumV = f.local(V128);
  const sumLo = f.local(I32);
  const sumHi = f.local(I32);

  f.get(2).i32Const(5).i32ShrS().set(nBlocks);
  f.get(0).set(xp);
  f.get(1).set(qp);
  f.i32Const(0).set(b);

  f.block().loop();
  f.get(b).get(nBlocks).i32GeS().brIf(1);

  // Compute amax across all 32 floats via 8 SIMD abs + pmax ops
  f.get(xp).v128Load(0).f32x4Abs()
    .get(xp).v128Load(16).f32x4Abs().f32x4Pmax()
    .get(xp).v128Load(32).f32x4Abs().f32x4Pmax()
    .get(xp).v128Load(48).f32x4Abs().f32x4Pmax()
    .get(xp).v128Load(64).f32x4Abs().f32x4Pmax()
    .get(xp).v128Load(80).f32x4Abs().f32x4Pmax()
    .get(xp).v128Load(96).f32x4Abs().f32x4Pmax()
    .get(xp).v128Load(112).f32x4Abs().f32x4Pmax()
    .set(vmax);

  f.get(vmax).f32x4Extract(0)
    .get(vmax).f32x4Extract(1).f32Max()
    .get(vmax).f32x4Extract(2).f32Max()
    .get(vmax).f32x4Extract(3).f32Max()
    .set(amax);

  f.get(amax).f32Const(0).f32Gt().if_();
  f.get(amax).f32Const(127).f32Div().set(d);
  f.f32Const(127).get(amax).f32Div().set(invD);
  f.else_();
  f.f32Const(0).set(d);
  f.f32Const(0).set(invD);
  f.end();

  f.get(invD).f32x4Splat().set(vinv);

  // First 16 floats -> q0..q3 -> sum_lo + x_lo
  f.get(xp).v128Load(0).get(vinv).f32x4Mul().f32x4Nearest().i32x4TruncSatF32x4S().set(q0);
  f.get(xp).v128Load(16).get(vinv).f32x4Mul().f32x4Nearest().i32x4TruncSatF32x4S().set(q1);
  f.get(xp).v128Load(32).get(vinv).f32x4Mul().f32x4Nearest().i32x4TruncSatF32x4S().set(q2);
  f.get(xp).v128Load(48).get(vinv).f32x4Mul().f32x4Nearest().i32x4TruncSatF32x4S().set(q3);

  f.get(q0).get(q1).i32x4Add().get(q2).i32x4Add().get(q3).i32x4Add().set(sumV);
  f.hsumI32x4(sumV).set(sumLo);

  f.get(qp)
    .get(q0).get(q1).i16x8NarrowI32x4S()
    .get(q2).get(q3).i16x8NarrowI32x4S()
    .i8x16NarrowI16x8S()
    .v128Store(16, 4);

  // Next 16 floats -> q0..q3 -> sum_hi + x_hi
  f.get(xp).v128Load(64).get(vinv).f32x4Mul().f32x4Nearest().i32x4TruncSatF32x4S().set(q0);
  f.get(xp).v128Load(80).get(vinv).f32x4Mul().f32x4Nearest().i32x4TruncSatF32x4S().set(q1);
  f.get(xp).v128Load(96).get(vinv).f32x4Mul().f32x4Nearest().i32x4TruncSatF32x4S().set(q2);
  f.get(xp).v128Load(112).get(vinv).f32x4Mul().f32x4Nearest().i32x4TruncSatF32x4S().set(q3);

  f.get(q0).get(q1).i32x4Add().get(q2).i32x4Add().get(q3).i32x4Add().set(sumV);
  f.hsumI32x4(sumV).set(sumHi);

  f.get(qp)
    .get(q0).get(q1).i16x8NarrowI32x4S()
    .get(q2).get(q3).i16x8NarrowI32x4S()
    .i8x16NarrowI16x8S()
    .v128Store(32, 4);

  // Store header: d_x, sum_lo, sum_hi, sum_all
  f.get(qp).get(d).f32Store(0);
  f.get(qp).get(sumLo).i32Store(4);
  f.get(qp).get(sumHi).i32Store(8);
  f.get(qp).get(sumLo).get(sumHi).i32Add().i32Store(12);

  f.get(xp).i32Const(128).i32Add().set(xp);
  f.get(qp).i32Const(Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();

  return f;
}

/**
 * 4-row register-blocked Q4_0 × Q8_0_ACT VNNI/Int8 MUL_MAT: (wPtr, q8Ptr, outPtr, cols, rows)
 * Uses 4-row vectorized horizontal-transpose epilogue (`acc4: f32x4`) — zero scalar lane extracts
 * in the main loop, cutting epilogue instruction count by 4x!
 */
function buildMulMatQ4_0Q8(useRelaxed: boolean): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const nBlocks = f.local(I32);
  const rowBytes = f.local(I32);
  const wp0 = f.local(I32);
  const wp1 = f.local(I32);
  const wp2 = f.local(I32);
  const wp3 = f.local(I32);
  const qp = f.local(I32);
  const b = f.local(I32);
  const sumAll8 = f.local(I32);
  const dx = f.local(F32);
  const acc0 = f.local(F32);
  const mask0f = f.local(V128);
  const zero = f.local(V128);
  const xlo = f.local(V128);
  const xhi = f.local(V128);
  const raw = f.local(V128);
  const nib = f.local(V128);
  const dot0 = f.local(V128);
  const dot1 = f.local(V128);
  const dot2 = f.local(V128);
  const dot3 = f.local(V128);
  const acc4 = f.local(V128);
  const dVec = f.local(V128);

  f.i32Const(0x0f).i8x16Splat().set(mask0f);
  f.i32Const(0).i32x4Splat().set(zero);
  f.get(3).i32Const(5).i32ShrS().set(nBlocks);
  f.get(nBlocks).i32Const(18).i32Mul().set(rowBytes);
  f.i32Const(0).set(r);

  const emitRowDot = (wpLoc: number, dotLoc: number) => {
    f.get(zero).set(dotLoc);
    f.get(wpLoc).v128Load(2).tee(raw).get(mask0f).v128And().set(nib);
    f.dotI8I7Add(xlo, nib, dotLoc, useRelaxed);
    f.get(raw).i32Const(4).i8x16ShrU().set(nib);
    f.dotI8I7Add(xhi, nib, dotLoc, useRelaxed);
  };

  // 4-row register-blocked loop with vectorized f32x4 epilogue
  f.block().loop();
  f.get(r).i32Const(3).i32Add().get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add()
    .tee(wp0).get(rowBytes).i32Add()
    .tee(wp1).get(rowBytes).i32Add()
    .tee(wp2).get(rowBytes).i32Add().set(wp3);
  f.get(1).set(qp);
  f.f32Const(0).f32x4Splat().set(acc4);
  f.i32Const(0).set(b);

  f.block().loop();
  f.get(b).get(nBlocks).i32GeS().brIf(1);
  f.get(qp).v128Load(16, 4).set(xlo);
  f.get(qp).v128Load(32, 4).set(xhi);

  emitRowDot(wp0, dot0);
  emitRowDot(wp1, dot1);
  emitRowDot(wp2, dot2);
  emitRowDot(wp3, dot3);

  // Reduce all 4 rows in 6 SIMD ops (zero scalar lane extracts!)
  f.hsum4RowsI32x4(dot0, dot1, dot2, dot3, raw, nib, dot0);
  f.get(dot0)
    .get(qp).i32Load(12).i32Const(3).i32Shl().i32x4Splat()
    .i32x4Sub().f32x4ConvertI32x4S().set(dot0);

  // Pack [d0, d1, d2, d3] * dx and accumulate into acc4
  f.get(zero)
    .loadF16AsF32(wp0, 0).f32x4ReplaceLane(0)
    .loadF16AsF32(wp1, 0).f32x4ReplaceLane(1)
    .loadF16AsF32(wp2, 0).f32x4ReplaceLane(2)
    .loadF16AsF32(wp3, 0).f32x4ReplaceLane(3)
    .get(qp).f32Load(0).f32x4Splat().f32x4Mul().set(dVec);

  if (useRelaxed) {
    f.get(dot0).get(dVec).get(acc4).f32x4RelaxedMadd().set(acc4);
  } else {
    f.get(dot0).get(dVec).f32x4Mul().get(acc4).f32x4Add().set(acc4);
  }

  f.get(wp0).i32Const(18).i32Add().set(wp0);
  f.get(wp1).i32Const(18).i32Add().set(wp1);
  f.get(wp2).i32Const(18).i32Add().set(wp2);
  f.get(wp3).i32Const(18).i32Add().set(wp3);
  f.get(qp).i32Const(Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc4).v128Store(0);
  f.get(r).i32Const(4).i32Add().set(r);
  f.br(0).end().end();

  // Remainder rows (0..3)
  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(wp0);
  f.get(1).set(qp);
  f.f32Const(0).set(acc0);
  f.i32Const(0).set(b);
  f.block().loop();
  f.get(b).get(nBlocks).i32GeS().brIf(1);
  f.get(qp).f32Load(0).set(dx);
  f.get(qp).i32Load(12).i32Const(3).i32Shl().set(sumAll8);
  f.get(qp).v128Load(16, 4).set(xlo);
  f.get(qp).v128Load(32, 4).set(xhi);
  emitRowDot(wp0, dot0);
  f.hsumI32x4(dot0).get(sumAll8).i32Sub().f32ConvertI32S();
  f.loadF16AsF32(wp0, 0).get(dx).f32Mul().f32Mul().get(acc0).f32Add().set(acc0);
  f.get(wp0).i32Const(18).i32Add().set(wp0);
  f.get(qp).i32Const(Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();
  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc0).f32Store(0);
  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();

  return f;
}

/**
 * 4-row × 4-token register-blocked Q4_0 × Q8_0_ACT VNNI/Int8 prefill kernel:
 * `(wPtr, q8BatchPtr, outBatchPtr, cols, rows, oStrideBytes)`.
 *
 * Tiles 4 output rows across each 32-element activation block so `qp0..qp3` (192 B = 3 cache lines)
 * stay 100% L1D-hot across all 4 rows and `sum4`/`dx4` are packed ONCE per 4 rows (4× less L2/L3
 * activation bandwidth on large FFN dims like 13,824–27,648!). Strictly 14 V128 locals (zero spills).
 */
function buildMulMatQ4_0Q8Batch4(useRelaxed: boolean): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const nBlocks = f.local(I32);
  const rowBytes = f.local(I32);
  const qStride = f.local(I32);
  const oStride = f.local(I32);
  const wp0 = f.local(I32);
  const wp1 = f.local(I32);
  const wp2 = f.local(I32);
  const wp3 = f.local(I32);
  const qp0 = f.local(I32);
  const qp1 = f.local(I32);
  const qp2 = f.local(I32);
  const qp3 = f.local(I32);
  const op = f.local(I32);
  const b = f.local(I32);

  // Strictly 14 V128 locals (<= 16 XMM registers -> zero spills!)
  const mask0f = f.local(V128);
  const acc0 = f.local(V128);
  const acc1 = f.local(V128);
  const acc2 = f.local(V128);
  const acc3 = f.local(V128);
  const sum4 = f.local(V128);
  const dx4 = f.local(V128);
  const wlo = f.local(V128);
  const whi = f.local(V128);
  const xtmp = f.local(V128);
  const dot0 = f.local(V128);
  const dot1 = f.local(V128);
  const dot2 = f.local(V128);
  const dot3 = f.local(V128);

  f.i32Const(0x0f).i8x16Splat().set(mask0f);
  f.get(3).i32Const(5).i32ShrS().set(nBlocks);
  f.get(nBlocks).i32Const(18).i32Mul().set(rowBytes);
  f.get(nBlocks).i32Const(Q8_0_ACT_BLOCK_BYTES).i32Mul().set(qStride);
  f.get(5).set(oStride);
  f.i32Const(0).set(r);

  const emitTokDot = (qpLoc: number, dotLoc: number) => {
    f.i32Const(0).i32x4Splat().set(dotLoc);
    f.get(qpLoc).v128Load(16, 4).set(xtmp);
    f.dotI8I7Add(xtmp, wlo, dotLoc, useRelaxed);
    f.get(qpLoc).v128Load(32, 4).set(xtmp);
    f.dotI8I7Add(xtmp, whi, dotLoc, useRelaxed);
  };

  const emitRowBlock = (wpLoc: number, accLoc: number) => {
    f.get(wpLoc).v128Load(2).tee(xtmp).get(mask0f).v128And().set(wlo);
    f.get(xtmp).i32Const(4).i8x16ShrU().set(whi);

    emitTokDot(qp0, dot0);
    emitTokDot(qp1, dot1);
    emitTokDot(qp2, dot2);
    emitTokDot(qp3, dot3);

    // Reuse wlo/whi as scratch for 6-op 4-token horizontal sum
    f.hsum4RowsI32x4(dot0, dot1, dot2, dot3, wlo, whi, dot0);
    f.get(dot0).get(sum4).i32x4Sub().f32x4ConvertI32x4S().set(dot0);
    f.loadF16AsF32(wpLoc, 0).f32x4Splat().get(dx4).f32x4Mul().set(wlo);
    if (useRelaxed) {
      f.get(dot0).get(wlo).get(accLoc).f32x4RelaxedMadd().set(accLoc);
    } else {
      f.get(dot0).get(wlo).f32x4Mul().get(accLoc).f32x4Add().set(accLoc);
    }
  };

  const loadSharedBlockMeta = () => {
    f.i32Const(0).i32x4Splat()
      .get(qp0).i32Load(12).i32x4ReplaceLane(0)
      .get(qp1).i32Load(12).i32x4ReplaceLane(1)
      .get(qp2).i32Load(12).i32x4ReplaceLane(2)
      .get(qp3).i32Load(12).i32x4ReplaceLane(3)
      .i32Const(3).i32x4Shl()
      .set(sum4);
    f.f32Const(0).f32x4Splat()
      .get(qp0).f32Load(0).f32x4ReplaceLane(0)
      .get(qp1).f32Load(0).f32x4ReplaceLane(1)
      .get(qp2).f32Load(0).f32x4ReplaceLane(2)
      .get(qp3).f32Load(0).f32x4ReplaceLane(3)
      .set(dx4);
  };

  const storeRowTokens = (rowIndexExpr: () => void, accLoc: number) => {
    rowIndexExpr();
    f.i32Const(2).i32Shl().get(2).i32Add().tee(op).get(accLoc).f32x4Extract(0).f32Store(0);
    f.get(op).get(oStride).i32Add().tee(op).get(accLoc).f32x4Extract(1).f32Store(0);
    f.get(op).get(oStride).i32Add().tee(op).get(accLoc).f32x4Extract(2).f32Store(0);
    f.get(op).get(oStride).i32Add().get(accLoc).f32x4Extract(3).f32Store(0);
  };

  // Main 4-row × 4-token loop
  f.block().loop();
  f.get(r).i32Const(3).i32Add().get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add()
    .tee(wp0).get(rowBytes).i32Add()
    .tee(wp1).get(rowBytes).i32Add()
    .tee(wp2).get(rowBytes).i32Add().set(wp3);
  f.get(1).tee(qp0).get(qStride).i32Add().tee(qp1).get(qStride).i32Add().tee(qp2).get(qStride).i32Add().set(qp3);
  f.f32Const(0).f32x4Splat().tee(acc0).tee(acc1).tee(acc2).set(acc3);
  f.i32Const(0).set(b);

  f.block().loop();
  f.get(b).get(nBlocks).i32GeS().brIf(1);
  loadSharedBlockMeta();
  emitRowBlock(wp0, acc0);
  emitRowBlock(wp1, acc1);
  emitRowBlock(wp2, acc2);
  emitRowBlock(wp3, acc3);

  f.get(wp0).i32Const(18).i32Add().set(wp0);
  f.get(wp1).i32Const(18).i32Add().set(wp1);
  f.get(wp2).i32Const(18).i32Add().set(wp2);
  f.get(wp3).i32Const(18).i32Add().set(wp3);
  f.get(qp0).i32Const(Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp0);
  f.get(qp1).i32Const(Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp1);
  f.get(qp2).i32Const(Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp2);
  f.get(qp3).i32Const(Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp3);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();

  storeRowTokens(() => f.get(r), acc0);
  storeRowTokens(() => f.get(r).i32Const(1).i32Add(), acc1);
  storeRowTokens(() => f.get(r).i32Const(2).i32Add(), acc2);
  storeRowTokens(() => f.get(r).i32Const(3).i32Add(), acc3);
  f.get(r).i32Const(4).i32Add().set(r);
  f.br(0).end().end();

  // Remainder rows (0..3)
  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(wp0);
  f.get(1).tee(qp0).get(qStride).i32Add().tee(qp1).get(qStride).i32Add().tee(qp2).get(qStride).i32Add().set(qp3);
  f.f32Const(0).f32x4Splat().set(acc0);
  f.i32Const(0).set(b);

  f.block().loop();
  f.get(b).get(nBlocks).i32GeS().brIf(1);
  loadSharedBlockMeta();
  emitRowBlock(wp0, acc0);
  f.get(wp0).i32Const(18).i32Add().set(wp0);
  f.get(qp0).i32Const(Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp0);
  f.get(qp1).i32Const(Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp1);
  f.get(qp2).i32Const(Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp2);
  f.get(qp3).i32Const(Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp3);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();

  storeRowTokens(() => f.get(r), acc0);
  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();

  return f;
}

/** 4-row register-blocked Q4_1 × Q8_0_ACT VNNI/Int8 MUL_MAT: (wPtr, q8Ptr, outPtr, cols, rows) */
function buildMulMatQ4_1Q8(useRelaxed: boolean): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const nBlocks = f.local(I32);
  const rowBytes = f.local(I32);
  const wp0 = f.local(I32);
  const wp1 = f.local(I32);
  const wp2 = f.local(I32);
  const wp3 = f.local(I32);
  const qp = f.local(I32);
  const op = f.local(I32);
  const b = f.local(I32);
  const dx = f.local(F32);
  const sumAllF = f.local(F32);
  const acc0 = f.local(F32);
  const acc1 = f.local(F32);
  const acc2 = f.local(F32);
  const acc3 = f.local(F32);
  const mask0f = f.local(V128);
  const zero = f.local(V128);
  const xlo = f.local(V128);
  const xhi = f.local(V128);
  const raw = f.local(V128);
  const nib = f.local(V128);
  const dotV = f.local(V128);

  f.i32Const(0x0f).i8x16Splat().set(mask0f);
  f.i32Const(0).i32x4Splat().set(zero);
  f.get(3).i32Const(5).i32ShrS().set(nBlocks);
  f.get(nBlocks).i32Const(20).i32Mul().set(rowBytes);
  f.i32Const(0).set(r);

  const emitQ41Row = (wpLoc: number, accLoc: number) => {
    f.get(zero).set(dotV);
    f.get(wpLoc).v128Load(4).tee(raw).get(mask0f).v128And().set(nib);
    f.dotI8I7Add(xlo, nib, dotV, useRelaxed);
    f.get(raw).i32Const(4).i8x16ShrU().set(nib);
    f.dotI8I7Add(xhi, nib, dotV, useRelaxed);
    f.loadF16AsF32(wpLoc, 0).hsumI32x4(dotV).f32ConvertI32S().f32Mul();
    f.loadF16AsF32(wpLoc, 2).get(sumAllF).f32Mul().f32Add();
    f.get(dx).f32Mul().get(accLoc).f32Add().set(accLoc);
  };

  // 4-row register-blocked loop
  f.block().loop();
  f.get(r).i32Const(3).i32Add().get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add()
    .tee(wp0).get(rowBytes).i32Add()
    .tee(wp1).get(rowBytes).i32Add()
    .tee(wp2).get(rowBytes).i32Add().set(wp3);
  f.get(1).set(qp);
  f.f32Const(0).tee(acc0).tee(acc1).tee(acc2).set(acc3);
  f.i32Const(0).set(b);

  f.block().loop();
  f.get(b).get(nBlocks).i32GeS().brIf(1);
  f.get(qp).f32Load(0).set(dx);
  f.get(qp).i32Load(12).f32ConvertI32S().set(sumAllF);
  f.get(qp).v128Load(16, 4).set(xlo);
  f.get(qp).v128Load(32, 4).set(xhi);

  emitQ41Row(wp0, acc0);
  emitQ41Row(wp1, acc1);
  emitQ41Row(wp2, acc2);
  emitQ41Row(wp3, acc3);

  f.get(wp0).i32Const(20).i32Add().set(wp0);
  f.get(wp1).i32Const(20).i32Add().set(wp1);
  f.get(wp2).i32Const(20).i32Add().set(wp2);
  f.get(wp3).i32Const(20).i32Add().set(wp3);
  f.get(qp).i32Const(Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().set(op);
  f.get(op).get(acc0).f32Store(0);
  f.get(op).get(acc1).f32Store(4);
  f.get(op).get(acc2).f32Store(8);
  f.get(op).get(acc3).f32Store(12);
  f.get(r).i32Const(4).i32Add().set(r);
  f.br(0).end().end();

  // Remainder rows (0..3)
  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(wp0);
  f.get(1).set(qp);
  f.f32Const(0).set(acc0);
  f.i32Const(0).set(b);
  f.block().loop();
  f.get(b).get(nBlocks).i32GeS().brIf(1);
  f.get(qp).f32Load(0).set(dx);
  f.get(qp).i32Load(12).f32ConvertI32S().set(sumAllF);
  f.get(qp).v128Load(16, 4).set(xlo);
  f.get(qp).v128Load(32, 4).set(xhi);
  emitQ41Row(wp0, acc0);
  f.get(wp0).i32Const(20).i32Add().set(wp0);
  f.get(qp).i32Const(Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();
  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc0).f32Store(0);
  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();
  return f;
}

/** 4-row register-blocked Q5_0 / Q5_1 × Q8_0_ACT VNNI/Int8 builder (`hasMin = false` -> Q5_0, `true` -> Q5_1). */
function buildMulMatQ5Q8(hasMin: boolean, useRelaxed: boolean): FuncBuilder {
  const blockSize = hasMin ? 24 : 22;
  const qhOff = hasMin ? 4 : 2;
  const qsOff = hasMin ? 8 : 6;
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const nBlocks = f.local(I32);
  const rowBytes = f.local(I32);
  const wp0 = f.local(I32);
  const wp1 = f.local(I32);
  const wp2 = f.local(I32);
  const wp3 = f.local(I32);
  const qp = f.local(I32);
  const op = f.local(I32);
  const b = f.local(I32);
  const sumAll16 = f.local(I32);
  const dx = f.local(F32);
  const sumAllF = f.local(F32);
  const acc0 = f.local(F32);
  const acc1 = f.local(F32);
  const acc2 = f.local(F32);
  const acc3 = f.local(F32);
  const mask0f = f.local(V128);
  const bitMask = f.local(V128);
  const swizLo = f.local(V128);
  const swizHi = f.local(V128);
  const bit16 = f.local(V128);
  const zero = f.local(V128);
  const xlo = f.local(V128);
  const xhi = f.local(V128);
  const qhVec = f.local(V128);
  const raw = f.local(V128);
  const q5v = f.local(V128);
  const dotV = f.local(V128);

  f.i32Const(0x0f).i8x16Splat().set(mask0f);
  f.v128Const([1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128]).set(bitMask);
  f.v128Const([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]).set(swizLo);
  f.v128Const([2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3]).set(swizHi);
  f.i32Const(16).i8x16Splat().set(bit16);
  f.i32Const(0).i32x4Splat().set(zero);

  f.get(3).i32Const(5).i32ShrS().set(nBlocks);
  f.get(nBlocks).i32Const(blockSize).i32Mul().set(rowBytes);
  f.i32Const(0).set(r);

  const emitQ5Row = (wpLoc: number, accLoc: number) => {
    f.get(zero).set(dotV);
    f.get(wpLoc).i32Load(qhOff, 0).i32x4Splat().set(qhVec);
    f.get(wpLoc).v128Load(qsOff).set(raw);

    f.get(qhVec).get(swizLo).i8x16Swizzle().get(bitMask).v128And().get(zero).i8x16Ne().get(bit16).v128And();
    f.get(raw).get(mask0f).v128And().v128Or().set(q5v);
    f.dotI8I7Add(xlo, q5v, dotV, useRelaxed);

    f.get(qhVec).get(swizHi).i8x16Swizzle().get(bitMask).v128And().get(zero).i8x16Ne().get(bit16).v128And();
    f.get(raw).i32Const(4).i8x16ShrU().v128Or().set(q5v);
    f.dotI8I7Add(xhi, q5v, dotV, useRelaxed);

    if (hasMin) {
      f.loadF16AsF32(wpLoc, 0).hsumI32x4(dotV).f32ConvertI32S().f32Mul();
      f.loadF16AsF32(wpLoc, 2).get(sumAllF).f32Mul().f32Add();
      f.get(dx).f32Mul().get(accLoc).f32Add().set(accLoc);
    } else {
      f.hsumI32x4(dotV).get(sumAll16).i32Sub().f32ConvertI32S();
      f.loadF16AsF32(wpLoc, 0).get(dx).f32Mul().f32Mul().get(accLoc).f32Add().set(accLoc);
    }
  };

  // 4-row register-blocked loop
  f.block().loop();
  f.get(r).i32Const(3).i32Add().get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add()
    .tee(wp0).get(rowBytes).i32Add()
    .tee(wp1).get(rowBytes).i32Add()
    .tee(wp2).get(rowBytes).i32Add().set(wp3);
  f.get(1).set(qp);
  f.f32Const(0).tee(acc0).tee(acc1).tee(acc2).set(acc3);
  f.i32Const(0).set(b);

  f.block().loop();
  f.get(b).get(nBlocks).i32GeS().brIf(1);
  f.get(qp).f32Load(0).set(dx);
  if (hasMin) {
    f.get(qp).i32Load(12).f32ConvertI32S().set(sumAllF);
  } else {
    f.get(qp).i32Load(12).i32Const(4).i32Shl().set(sumAll16);
  }
  f.get(qp).v128Load(16, 4).set(xlo);
  f.get(qp).v128Load(32, 4).set(xhi);

  emitQ5Row(wp0, acc0);
  emitQ5Row(wp1, acc1);
  emitQ5Row(wp2, acc2);
  emitQ5Row(wp3, acc3);

  f.get(wp0).i32Const(blockSize).i32Add().set(wp0);
  f.get(wp1).i32Const(blockSize).i32Add().set(wp1);
  f.get(wp2).i32Const(blockSize).i32Add().set(wp2);
  f.get(wp3).i32Const(blockSize).i32Add().set(wp3);
  f.get(qp).i32Const(Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().set(op);
  f.get(op).get(acc0).f32Store(0);
  f.get(op).get(acc1).f32Store(4);
  f.get(op).get(acc2).f32Store(8);
  f.get(op).get(acc3).f32Store(12);
  f.get(r).i32Const(4).i32Add().set(r);
  f.br(0).end().end();

  // Remainder rows (0..3)
  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(wp0);
  f.get(1).set(qp);
  f.f32Const(0).set(acc0);
  f.i32Const(0).set(b);
  f.block().loop();
  f.get(b).get(nBlocks).i32GeS().brIf(1);
  f.get(qp).f32Load(0).set(dx);
  if (hasMin) {
    f.get(qp).i32Load(12).f32ConvertI32S().set(sumAllF);
  } else {
    f.get(qp).i32Load(12).i32Const(4).i32Shl().set(sumAll16);
  }
  f.get(qp).v128Load(16, 4).set(xlo);
  f.get(qp).v128Load(32, 4).set(xhi);
  emitQ5Row(wp0, acc0);
  f.get(wp0).i32Const(blockSize).i32Add().set(wp0);
  f.get(qp).i32Const(Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();
  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc0).f32Store(0);
  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();
  return f;
}

/** 4-row register-blocked Q8_0 × Q8_0_ACT Int8 MUL_MAT with vectorized f32x4 epilogue. */
function buildMulMatQ8_0Q8(useRelaxed: boolean): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const nBlocks = f.local(I32);
  const rowBytes = f.local(I32);
  const wp0 = f.local(I32);
  const wp1 = f.local(I32);
  const wp2 = f.local(I32);
  const wp3 = f.local(I32);
  const qp = f.local(I32);
  const b = f.local(I32);
  const dx = f.local(F32);
  const acc0 = f.local(F32);
  const zero = f.local(V128);
  const xlo = f.local(V128);
  const xhi = f.local(V128);
  const wv = f.local(V128);
  const t1 = f.local(V128);
  const dot0 = f.local(V128);
  const dot1 = f.local(V128);
  const dot2 = f.local(V128);
  const dot3 = f.local(V128);
  const acc4 = f.local(V128);
  const dVec = f.local(V128);

  f.i32Const(0).i32x4Splat().set(zero);
  f.get(3).i32Const(5).i32ShrS().set(nBlocks);
  f.get(nBlocks).i32Const(34).i32Mul().set(rowBytes);
  f.i32Const(0).set(r);

  const emitQ8Dot = (wpLoc: number, dotLoc: number) => {
    f.get(zero).set(dotLoc);
    f.get(wpLoc).v128Load(2).set(wv);
    f.dotI8I8Add(xlo, wv, dotLoc);
    f.get(wpLoc).v128Load(18).set(wv);
    f.dotI8I8Add(xhi, wv, dotLoc);
  };

  // 4-row register-blocked loop
  f.block().loop();
  f.get(r).i32Const(3).i32Add().get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add()
    .tee(wp0).get(rowBytes).i32Add()
    .tee(wp1).get(rowBytes).i32Add()
    .tee(wp2).get(rowBytes).i32Add().set(wp3);
  f.get(1).set(qp);
  f.f32Const(0).f32x4Splat().set(acc4);
  f.i32Const(0).set(b);

  f.block().loop();
  f.get(b).get(nBlocks).i32GeS().brIf(1);
  f.get(qp).v128Load(16, 4).set(xlo);
  f.get(qp).v128Load(32, 4).set(xhi);

  emitQ8Dot(wp0, dot0);
  emitQ8Dot(wp1, dot1);
  emitQ8Dot(wp2, dot2);
  emitQ8Dot(wp3, dot3);

  f.hsum4RowsI32x4(dot0, dot1, dot2, dot3, wv, t1, dot0);
  f.get(dot0).f32x4ConvertI32x4S().set(dot0);
  f.get(zero)
    .loadF16AsF32(wp0, 0).f32x4ReplaceLane(0)
    .loadF16AsF32(wp1, 0).f32x4ReplaceLane(1)
    .loadF16AsF32(wp2, 0).f32x4ReplaceLane(2)
    .loadF16AsF32(wp3, 0).f32x4ReplaceLane(3)
    .get(qp).f32Load(0).f32x4Splat().f32x4Mul().set(dVec);

  if (useRelaxed) {
    f.get(dot0).get(dVec).get(acc4).f32x4RelaxedMadd().set(acc4);
  } else {
    f.get(dot0).get(dVec).f32x4Mul().get(acc4).f32x4Add().set(acc4);
  }

  f.get(wp0).i32Const(34).i32Add().set(wp0);
  f.get(wp1).i32Const(34).i32Add().set(wp1);
  f.get(wp2).i32Const(34).i32Add().set(wp2);
  f.get(wp3).i32Const(34).i32Add().set(wp3);
  f.get(qp).i32Const(Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc4).v128Store(0);
  f.get(r).i32Const(4).i32Add().set(r);
  f.br(0).end().end();

  // Remainder rows (0..3)
  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(wp0);
  f.get(1).set(qp);
  f.f32Const(0).set(acc0);
  f.i32Const(0).set(b);
  f.block().loop();
  f.get(b).get(nBlocks).i32GeS().brIf(1);
  f.get(qp).f32Load(0).set(dx);
  f.get(qp).v128Load(16, 4).set(xlo);
  f.get(qp).v128Load(32, 4).set(xhi);
  emitQ8Dot(wp0, dot0);
  f.hsumI32x4(dot0).f32ConvertI32S().loadF16AsF32(wp0, 0).get(dx).f32Mul().f32Mul().get(acc0).f32Add().set(acc0);
  f.get(wp0).i32Const(34).i32Add().set(wp0);
  f.get(qp).i32Const(Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();
  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc0).f32Store(0);
  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();

  return f;
}

/**
 * 4-token batched Q8_0 × Q8_0_ACT prefill kernel with vectorized 4-lane transpose epilogue
 * (`hsum4RowsI32x4` + `f32x4RelaxedMadd`, zero scalar extracts inside the block loop).
 */
function buildMulMatQ8_0Q8Batch4(useRelaxed: boolean): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const nBlocks = f.local(I32);
  const rowBytes = f.local(I32);
  const qStride = f.local(I32);
  const oStride = f.local(I32);
  const wp = f.local(I32);
  const qp0 = f.local(I32);
  const qp1 = f.local(I32);
  const qp2 = f.local(I32);
  const qp3 = f.local(I32);
  const b = f.local(I32);
  const zero = f.local(V128);
  const wlo = f.local(V128);
  const whi = f.local(V128);
  const xtmp = f.local(V128);
  const dot0 = f.local(V128);
  const dot1 = f.local(V128);
  const dot2 = f.local(V128);
  const dot3 = f.local(V128);
  const t0 = f.local(V128);
  const t1 = f.local(V128);
  const acc4 = f.local(V128);

  f.i32Const(0).i32x4Splat().set(zero);
  f.get(3).i32Const(5).i32ShrS().set(nBlocks);
  f.get(nBlocks).i32Const(34).i32Mul().set(rowBytes);
  f.get(nBlocks).i32Const(Q8_0_ACT_BLOCK_BYTES).i32Mul().set(qStride);
  f.get(5).set(oStride);
  f.i32Const(0).set(r);

  const emitTokDot = (qpLoc: number, dotLoc: number) => {
    f.get(zero).set(dotLoc);
    f.get(qpLoc).v128Load(16, 4).set(xtmp);
    f.dotI8I8Add(xtmp, wlo, dotLoc);
    f.get(qpLoc).v128Load(32, 4).set(xtmp);
    f.dotI8I8Add(xtmp, whi, dotLoc);
  };

  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(wp);
  f.get(1).tee(qp0).get(qStride).i32Add().tee(qp1).get(qStride).i32Add().tee(qp2).get(qStride).i32Add().set(qp3);
  f.f32Const(0).f32x4Splat().set(acc4);
  f.i32Const(0).set(b);

  f.block().loop();
  f.get(b).get(nBlocks).i32GeS().brIf(1);
  f.get(wp).v128Load(2).set(wlo);
  f.get(wp).v128Load(18).set(whi);

  emitTokDot(qp0, dot0);
  emitTokDot(qp1, dot1);
  emitTokDot(qp2, dot2);
  emitTokDot(qp3, dot3);

  f.hsum4RowsI32x4(dot0, dot1, dot2, dot3, t0, t1, dot0);
  f.get(dot0).f32x4ConvertI32x4S().set(dot0);

  f.get(zero)
    .get(qp0).f32Load(0).f32x4ReplaceLane(0)
    .get(qp1).f32Load(0).f32x4ReplaceLane(1)
    .get(qp2).f32Load(0).f32x4ReplaceLane(2)
    .get(qp3).f32Load(0).f32x4ReplaceLane(3)
    .loadF16AsF32(wp, 0).f32x4Splat().f32x4Mul()
    .set(t1);

  if (useRelaxed) {
    f.get(dot0).get(t1).get(acc4).f32x4RelaxedMadd().set(acc4);
  } else {
    f.get(dot0).get(t1).f32x4Mul().get(acc4).f32x4Add().set(acc4);
  }

  f.get(wp).i32Const(34).i32Add().set(wp);
  f.get(qp0).i32Const(Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp0);
  f.get(qp1).i32Const(Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp1);
  f.get(qp2).i32Const(Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp2);
  f.get(qp3).i32Const(Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp3);
  f.get(b).i32Const(1).i32Add().set(b);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().tee(wp).get(acc4).f32x4Extract(0).f32Store(0);
  f.get(wp).get(oStride).i32Add().tee(wp).get(acc4).f32x4Extract(1).f32Store(0);
  f.get(wp).get(oStride).i32Add().tee(wp).get(acc4).f32x4Extract(2).f32Store(0);
  f.get(wp).get(oStride).i32Add().get(acc4).f32x4Extract(3).f32Store(0);

  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();
  return f;
}

/**
 * Helper to emit Q4_K scale/min decoding for sub-block `j` (`0..7`) at `wp + 4` into `scLoc` & `mLoc` (f32).
 */
function emitQ4KScaleMinF32(f: FuncBuilder, wp: number, j: number, scLoc: number, mLoc: number): void {
  if (j < 4) {
    f.get(wp).i32Load8U(4 + j).i32Const(63).i32And().f32ConvertI32S().set(scLoc);
    f.get(wp).i32Load8U(4 + j + 4).i32Const(63).i32And().f32ConvertI32S().set(mLoc);
  } else {
    f.get(wp).i32Load8U(4 + j + 4).i32Const(0x0f).i32And()
      .get(wp).i32Load8U(4 + j - 4).i32Const(6).i32ShrU().i32Const(4).i32Shl().i32Or()
      .f32ConvertI32S().set(scLoc);
    f.get(wp).i32Load8U(4 + j + 4).i32Const(4).i32ShrU()
      .get(wp).i32Load8U(4 + j).i32Const(6).i32ShrU().i32Const(4).i32Shl().i32Or()
      .f32ConvertI32S().set(mLoc);
  }
}

/**
 * 4-row register-blocked Q4_K × Q8_0_ACT VNNI/Int8 MUL_MAT: (wPtr, q8Ptr, outPtr, cols, rows)
 * Loads each 64-element pair of Q8_0_ACT blocks (`xlo0, xhi0, xlo1, xhi1`, `dx`, `sum_all`) ONCE
 * across 4 output rows, cutting activation L1 traffic by 75% while saturating 4 independent F32
 * accumulators (`acc0..acc3`) with <= 10 live V128 registers (zero stack spills).
 */
function buildMulMatQ4KQ8(useRelaxed: boolean): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const nSuper = f.local(I32);
  const rowBytes = f.local(I32);
  const wp0 = f.local(I32);
  const wp1 = f.local(I32);
  const wp2 = f.local(I32);
  const wp3 = f.local(I32);
  const qp = f.local(I32);
  const op = f.local(I32);
  const sb = f.local(I32);
  const d0 = f.local(F32);
  const dmin0 = f.local(F32);
  const d1 = f.local(F32);
  const dmin1 = f.local(F32);
  const d2 = f.local(F32);
  const dmin2 = f.local(F32);
  const d3 = f.local(F32);
  const dmin3 = f.local(F32);
  const sc = f.local(F32);
  const mn = f.local(F32);
  const dxLo = f.local(F32);
  const dxHi = f.local(F32);
  const sumLoF = f.local(F32);
  const sumHiF = f.local(F32);
  const acc0 = f.local(F32);
  const acc1 = f.local(F32);
  const acc2 = f.local(F32);
  const acc3 = f.local(F32);
  const mask0f = f.local(V128);
  const zero = f.local(V128);
  const xlo0 = f.local(V128);
  const xhi0 = f.local(V128);
  const xlo1 = f.local(V128);
  const xhi1 = f.local(V128);
  const raw0 = f.local(V128);
  const raw1 = f.local(V128);
  const nib = f.local(V128);
  const dotV = f.local(V128);

  f.i32Const(0x0f).i8x16Splat().set(mask0f);
  f.i32Const(0).i32x4Splat().set(zero);
  f.get(3).i32Const(8).i32ShrS().set(nSuper);
  f.get(nSuper).i32Const(144).i32Mul().set(rowBytes);
  f.i32Const(0).set(r);

  const emitChunkRow = (wpLoc: number, dLoc: number, dminLoc: number, accLoc: number, c: number) => {
    const qsBase = 16 + c * 32;
    f.get(wpLoc).v128Load(qsBase).set(raw0);
    f.get(wpLoc).v128Load(qsBase + 16).set(raw1);

    // Sub-block 2c (low nibbles against pre-loaded xlo0, xhi0)
    emitQ4KScaleMinF32(f, wpLoc, c * 2, sc, mn);
    f.get(zero).set(dotV);
    f.get(raw0).get(mask0f).v128And().set(nib);
    f.dotI8I7Add(xlo0, nib, dotV, useRelaxed);
    f.get(raw1).get(mask0f).v128And().set(nib);
    f.dotI8I7Add(xhi0, nib, dotV, useRelaxed);
    f.get(dLoc).get(sc).f32Mul().hsumI32x4(dotV).f32ConvertI32S().f32Mul();
    f.get(dminLoc).get(mn).f32Mul().get(sumLoF).f32Mul().f32Sub();
    f.get(dxLo).f32Mul().get(accLoc).f32Add().set(accLoc);

    // Sub-block 2c + 1 (high nibbles against pre-loaded xlo1, xhi1)
    emitQ4KScaleMinF32(f, wpLoc, c * 2 + 1, sc, mn);
    f.get(zero).set(dotV);
    f.get(raw0).i32Const(4).i8x16ShrU().set(nib);
    f.dotI8I7Add(xlo1, nib, dotV, useRelaxed);
    f.get(raw1).i32Const(4).i8x16ShrU().set(nib);
    f.dotI8I7Add(xhi1, nib, dotV, useRelaxed);
    f.get(dLoc).get(sc).f32Mul().hsumI32x4(dotV).f32ConvertI32S().f32Mul();
    f.get(dminLoc).get(mn).f32Mul().get(sumHiF).f32Mul().f32Sub();
    f.get(dxHi).f32Mul().get(accLoc).f32Add().set(accLoc);
  };

  // 4-row register-blocked loop
  f.block().loop();
  f.get(r).i32Const(3).i32Add().get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add()
    .tee(wp0).get(rowBytes).i32Add()
    .tee(wp1).get(rowBytes).i32Add()
    .tee(wp2).get(rowBytes).i32Add().set(wp3);
  f.get(1).set(qp);
  f.f32Const(0).tee(acc0).tee(acc1).tee(acc2).set(acc3);
  f.i32Const(0).set(sb);

  f.block().loop();
  f.get(sb).get(nSuper).i32GeS().brIf(1);
  f.loadF16AsF32(wp0, 0).set(d0);
  f.loadF16AsF32(wp0, 2).set(dmin0);
  f.loadF16AsF32(wp1, 0).set(d1);
  f.loadF16AsF32(wp1, 2).set(dmin1);
  f.loadF16AsF32(wp2, 0).set(d2);
  f.loadF16AsF32(wp2, 2).set(dmin2);
  f.loadF16AsF32(wp3, 0).set(d3);
  f.loadF16AsF32(wp3, 2).set(dmin3);

  for (let c = 0; c < 4; c++) {
    const qOffLo = c * 2 * Q8_0_ACT_BLOCK_BYTES;
    const qOffHi = (c * 2 + 1) * Q8_0_ACT_BLOCK_BYTES;

    f.get(qp).f32Load(qOffLo).set(dxLo);
    f.get(qp).i32Load(qOffLo + 12).f32ConvertI32S().set(sumLoF);
    f.get(qp).v128Load(qOffLo + 16, 4).set(xlo0);
    f.get(qp).v128Load(qOffLo + 32, 4).set(xhi0);

    f.get(qp).f32Load(qOffHi).set(dxHi);
    f.get(qp).i32Load(qOffHi + 12).f32ConvertI32S().set(sumHiF);
    f.get(qp).v128Load(qOffHi + 16, 4).set(xlo1);
    f.get(qp).v128Load(qOffHi + 32, 4).set(xhi1);

    emitChunkRow(wp0, d0, dmin0, acc0, c);
    emitChunkRow(wp1, d1, dmin1, acc1, c);
    emitChunkRow(wp2, d2, dmin2, acc2, c);
    emitChunkRow(wp3, d3, dmin3, acc3, c);
  }

  f.get(wp0).i32Const(144).i32Add().set(wp0);
  f.get(wp1).i32Const(144).i32Add().set(wp1);
  f.get(wp2).i32Const(144).i32Add().set(wp2);
  f.get(wp3).i32Const(144).i32Add().set(wp3);
  f.get(qp).i32Const(8 * Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(sb).i32Const(1).i32Add().set(sb);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().set(op);
  f.get(op).get(acc0).f32Store(0);
  f.get(op).get(acc1).f32Store(4);
  f.get(op).get(acc2).f32Store(8);
  f.get(op).get(acc3).f32Store(12);
  f.get(r).i32Const(4).i32Add().set(r);
  f.br(0).end().end();

  // Remainder rows (0..3)
  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(wp0);
  f.get(1).set(qp);
  f.f32Const(0).set(acc0);
  f.i32Const(0).set(sb);

  f.block().loop();
  f.get(sb).get(nSuper).i32GeS().brIf(1);
  f.loadF16AsF32(wp0, 0).set(d0);
  f.loadF16AsF32(wp0, 2).set(dmin0);

  for (let c = 0; c < 4; c++) {
    const qOffLo = c * 2 * Q8_0_ACT_BLOCK_BYTES;
    const qOffHi = (c * 2 + 1) * Q8_0_ACT_BLOCK_BYTES;
    f.get(qp).f32Load(qOffLo).set(dxLo);
    f.get(qp).i32Load(qOffLo + 12).f32ConvertI32S().set(sumLoF);
    f.get(qp).v128Load(qOffLo + 16, 4).set(xlo0);
    f.get(qp).v128Load(qOffLo + 32, 4).set(xhi0);
    f.get(qp).f32Load(qOffHi).set(dxHi);
    f.get(qp).i32Load(qOffHi + 12).f32ConvertI32S().set(sumHiF);
    f.get(qp).v128Load(qOffHi + 16, 4).set(xlo1);
    f.get(qp).v128Load(qOffHi + 32, 4).set(xhi1);
    emitChunkRow(wp0, d0, dmin0, acc0, c);
  }

  f.get(wp0).i32Const(144).i32Add().set(wp0);
  f.get(qp).i32Const(8 * Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(sb).i32Const(1).i32Add().set(sb);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc0).f32Store(0);
  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();
  return f;
}

/** 4-token batched Q4_K × Q8_0_ACT VNNI/Int8 prefill kernel. */
function buildMulMatQ4KQ8Batch4(useRelaxed: boolean): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const nSuper = f.local(I32);
  const rowBytes = f.local(I32);
  const qStride = f.local(I32);
  const oStride = f.local(I32);
  const wp = f.local(I32);
  const qp0 = f.local(I32);
  const qp1 = f.local(I32);
  const qp2 = f.local(I32);
  const qp3 = f.local(I32);
  const sb = f.local(I32);
  const d = f.local(F32);
  const dmin = f.local(F32);
  const dSc = f.local(F32);
  const dMn = f.local(F32);
  const acc0 = f.local(F32);
  const acc1 = f.local(F32);
  const acc2 = f.local(F32);
  const acc3 = f.local(F32);
  const mask0f = f.local(V128);
  const zero = f.local(V128);
  const raw0 = f.local(V128);
  const raw1 = f.local(V128);
  const w0 = f.local(V128);
  const w1 = f.local(V128);
  const xtmp = f.local(V128);
  const dotV = f.local(V128);

  f.i32Const(0x0f).i8x16Splat().set(mask0f);
  f.i32Const(0).i32x4Splat().set(zero);
  f.get(3).i32Const(8).i32ShrS().set(nSuper);
  f.get(nSuper).i32Const(144).i32Mul().set(rowBytes);
  f.get(nSuper).i32Const(8 * Q8_0_ACT_BLOCK_BYTES).i32Mul().set(qStride);
  f.get(5).set(oStride);
  f.i32Const(0).set(r);

  const emitTokSub = (qpLoc: number, qOff: number, accLoc: number) => {
    f.get(zero).set(dotV);
    f.get(qpLoc).v128Load(qOff + 16, 4).set(xtmp);
    f.dotI8I7Add(xtmp, w0, dotV, useRelaxed);
    f.get(qpLoc).v128Load(qOff + 32, 4).set(xtmp);
    f.dotI8I7Add(xtmp, w1, dotV, useRelaxed);
    f.get(dSc).hsumI32x4(dotV).f32ConvertI32S().f32Mul();
    f.get(dMn).get(qpLoc).i32Load(qOff + 12).f32ConvertI32S().f32Mul().f32Sub();
    f.get(qpLoc).f32Load(qOff).f32Mul().get(accLoc).f32Add().set(accLoc);
  };

  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(wp);
  f.get(1).tee(qp0).get(qStride).i32Add().tee(qp1).get(qStride).i32Add().tee(qp2).get(qStride).i32Add().set(qp3);
  f.f32Const(0).set(acc0);
  f.f32Const(0).set(acc1);
  f.f32Const(0).set(acc2);
  f.f32Const(0).set(acc3);
  f.i32Const(0).set(sb);

  f.block().loop();
  f.get(sb).get(nSuper).i32GeS().brIf(1);
  f.loadF16AsF32(wp, 0).set(d);
  f.loadF16AsF32(wp, 2).set(dmin);

  for (let c = 0; c < 4; c++) {
    const qsBase = 16 + c * 32;
    const qOffLo = (c * 2) * Q8_0_ACT_BLOCK_BYTES;
    const qOffHi = (c * 2 + 1) * Q8_0_ACT_BLOCK_BYTES;

    f.get(wp).v128Load(qsBase).set(raw0);
    f.get(wp).v128Load(qsBase + 16).set(raw1);

    // Sub-block 2c (low nibbles) across all 4 tokens
    emitQ4KScaleMinF32(f, wp, c * 2, dSc, dMn);
    f.get(dSc).get(d).f32Mul().set(dSc);
    f.get(dMn).get(dmin).f32Mul().set(dMn);
    f.get(raw0).get(mask0f).v128And().set(w0);
    f.get(raw1).get(mask0f).v128And().set(w1);
    emitTokSub(qp0, qOffLo, acc0);
    emitTokSub(qp1, qOffLo, acc1);
    emitTokSub(qp2, qOffLo, acc2);
    emitTokSub(qp3, qOffLo, acc3);

    // Sub-block 2c + 1 (high nibbles) across all 4 tokens
    emitQ4KScaleMinF32(f, wp, c * 2 + 1, dSc, dMn);
    f.get(dSc).get(d).f32Mul().set(dSc);
    f.get(dMn).get(dmin).f32Mul().set(dMn);
    f.get(raw0).i32Const(4).i8x16ShrU().set(w0);
    f.get(raw1).i32Const(4).i8x16ShrU().set(w1);
    emitTokSub(qp0, qOffHi, acc0);
    emitTokSub(qp1, qOffHi, acc1);
    emitTokSub(qp2, qOffHi, acc2);
    emitTokSub(qp3, qOffHi, acc3);
  }

  f.get(wp).i32Const(144).i32Add().set(wp);
  f.get(qp0).i32Const(8 * Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp0);
  f.get(qp1).i32Const(8 * Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp1);
  f.get(qp2).i32Const(8 * Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp2);
  f.get(qp3).i32Const(8 * Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp3);
  f.get(sb).i32Const(1).i32Add().set(sb);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().tee(wp).get(acc0).f32Store(0);
  f.get(wp).get(oStride).i32Add().tee(wp).get(acc1).f32Store(0);
  f.get(wp).get(oStride).i32Add().tee(wp).get(acc2).f32Store(0);
  f.get(wp).get(oStride).i32Add().get(acc3).f32Store(0);

  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();
  return f;
}

/** Q5_K × Q8_0_ACT VNNI/Int8 MUL_MAT: (wPtr, q8Ptr, outPtr, cols, rows) */
function buildMulMatQ5KQ8(useRelaxed: boolean): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const nSuper = f.local(I32);
  const rowBytes = f.local(I32);
  const wp = f.local(I32);
  const qp = f.local(I32);
  const sb = f.local(I32);
  const d = f.local(F32);
  const dmin = f.local(F32);
  const sc = f.local(F32);
  const mn = f.local(F32);
  const acc = f.local(F32);
  const mask0f = f.local(V128);
  const bit16 = f.local(V128);
  const zero = f.local(V128);
  const qh0 = f.local(V128);
  const qh1 = f.local(V128);
  const raw0 = f.local(V128);
  const raw1 = f.local(V128);
  const q5v = f.local(V128);
  const xtmp = f.local(V128);
  const dotV = f.local(V128);

  f.i32Const(0x0f).i8x16Splat().set(mask0f);
  f.i32Const(16).i8x16Splat().set(bit16);
  f.i32Const(0).i32x4Splat().set(zero);
  f.get(3).i32Const(8).i32ShrS().set(nSuper);
  f.get(nSuper).i32Const(176).i32Mul().set(rowBytes);
  f.i32Const(0).set(r);

  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(wp);
  f.get(1).set(qp);
  f.f32Const(0).set(acc);
  f.i32Const(0).set(sb);

  f.block().loop();
  f.get(sb).get(nSuper).i32GeS().brIf(1);
  f.loadF16AsF32(wp, 0).set(d);
  f.loadF16AsF32(wp, 2).set(dmin);
  f.get(wp).v128Load(16).set(qh0);
  f.get(wp).v128Load(32).set(qh1);

  for (let c = 0; c < 4; c++) {
    const qsBase = 48 + c * 32;
    const qOffLo = (c * 2) * Q8_0_ACT_BLOCK_BYTES;
    const qOffHi = (c * 2 + 1) * Q8_0_ACT_BLOCK_BYTES;
    const u1 = 1 << (c * 2);
    const u2 = 1 << (c * 2 + 1);

    f.get(wp).v128Load(qsBase).set(raw0);
    f.get(wp).v128Load(qsBase + 16).set(raw1);

    // Sub-block 2c (low nibbles + qh bit u1)
    emitQ4KScaleMinF32(f, wp, c * 2, sc, mn);
    f.get(zero).set(dotV);
    f.get(raw0).get(mask0f).v128And()
      .get(qh0).i32Const(u1).i8x16Splat().v128And().get(zero).i8x16Ne().get(bit16).v128And()
      .v128Or().set(q5v);
    f.get(qp).v128Load(qOffLo + 16, 4).set(xtmp);
    f.dotI8I7Add(xtmp, q5v, dotV, useRelaxed);

    f.get(raw1).get(mask0f).v128And()
      .get(qh1).i32Const(u1).i8x16Splat().v128And().get(zero).i8x16Ne().get(bit16).v128And()
      .v128Or().set(q5v);
    f.get(qp).v128Load(qOffLo + 32, 4).set(xtmp);
    f.dotI8I7Add(xtmp, q5v, dotV, useRelaxed);

    f.get(d).get(sc).f32Mul().hsumI32x4(dotV).f32ConvertI32S().f32Mul();
    f.get(dmin).get(mn).f32Mul().get(qp).i32Load(qOffLo + 12).f32ConvertI32S().f32Mul().f32Sub();
    f.get(qp).f32Load(qOffLo).f32Mul().get(acc).f32Add().set(acc);

    // Sub-block 2c + 1 (high nibbles + qh bit u2)
    emitQ4KScaleMinF32(f, wp, c * 2 + 1, sc, mn);
    f.get(zero).set(dotV);
    f.get(raw0).i32Const(4).i8x16ShrU()
      .get(qh0).i32Const(u2).i8x16Splat().v128And().get(zero).i8x16Ne().get(bit16).v128And()
      .v128Or().set(q5v);
    f.get(qp).v128Load(qOffHi + 16, 4).set(xtmp);
    f.dotI8I7Add(xtmp, q5v, dotV, useRelaxed);

    f.get(raw1).i32Const(4).i8x16ShrU()
      .get(qh1).i32Const(u2).i8x16Splat().v128And().get(zero).i8x16Ne().get(bit16).v128And()
      .v128Or().set(q5v);
    f.get(qp).v128Load(qOffHi + 32, 4).set(xtmp);
    f.dotI8I7Add(xtmp, q5v, dotV, useRelaxed);

    f.get(d).get(sc).f32Mul().hsumI32x4(dotV).f32ConvertI32S().f32Mul();
    f.get(dmin).get(mn).f32Mul().get(qp).i32Load(qOffHi + 12).f32ConvertI32S().f32Mul().f32Sub();
    f.get(qp).f32Load(qOffHi).f32Mul().get(acc).f32Add().set(acc);
  }

  f.get(wp).i32Const(176).i32Add().set(wp);
  f.get(qp).i32Const(8 * Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(sb).i32Const(1).i32Add().set(sb);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc).f32Store(0);
  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();
  return f;
}

/**
 * 4-row register-blocked Q6_K × Q8_0_ACT VNNI/Int8 MUL_MAT: (wPtr, q8Ptr, outPtr, cols, rows)
 * Loads each `(h, s)` slice of 4 Q8_0_ACT blocks (`x0..x3`, `sum32_0..3`, `dx0..3`) ONCE across
 * 4 output rows — accelerating Q6_K LM heads (49K–152K vocab) and FFN down projections by 1.5x–2x.
 */
function buildMulMatQ6KQ8(useRelaxed: boolean): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const nSuper = f.local(I32);
  const rowBytes = f.local(I32);
  const wp0 = f.local(I32);
  const wp1 = f.local(I32);
  const wp2 = f.local(I32);
  const wp3 = f.local(I32);
  const qp = f.local(I32);
  const op = f.local(I32);
  const sb = f.local(I32);
  const sum32_0 = f.local(I32);
  const sum32_1 = f.local(I32);
  const sum32_2 = f.local(I32);
  const sum32_3 = f.local(I32);
  const dx0 = f.local(F32);
  const dx1 = f.local(F32);
  const dx2 = f.local(F32);
  const dx3 = f.local(F32);
  const acc0 = f.local(F32);
  const acc1 = f.local(F32);
  const acc2 = f.local(F32);
  const acc3 = f.local(F32);
  const sb0 = f.local(F32);
  const sb1 = f.local(F32);
  const sb2 = f.local(F32);
  const sb3 = f.local(F32);
  const mask0f = f.local(V128);
  const mask30 = f.local(V128);
  const zero = f.local(V128);
  const x0 = f.local(V128);
  const x1 = f.local(V128);
  const x2 = f.local(V128);
  const x3 = f.local(V128);
  const qla = f.local(V128);
  const qlb = f.local(V128);
  const qhv = f.local(V128);
  const q6v = f.local(V128);
  const dotV = f.local(V128);

  f.i32Const(0x0f).i8x16Splat().set(mask0f);
  f.i32Const(0x30).i8x16Splat().set(mask30);
  f.i32Const(0).i32x4Splat().set(zero);
  f.get(3).i32Const(8).i32ShrS().set(nSuper);
  f.get(nSuper).i32Const(210).i32Mul().set(rowBytes);
  f.i32Const(0).set(r);

  const loadSliceActivations = (blkBase: number, xVecOff: number, sumOff: number) => {
    const o0 = (blkBase + 0) * Q8_0_ACT_BLOCK_BYTES;
    const o1 = (blkBase + 1) * Q8_0_ACT_BLOCK_BYTES;
    const o2 = (blkBase + 2) * Q8_0_ACT_BLOCK_BYTES;
    const o3 = (blkBase + 3) * Q8_0_ACT_BLOCK_BYTES;
    f.get(qp).v128Load(o0 + xVecOff, 4).set(x0);
    f.get(qp).v128Load(o1 + xVecOff, 4).set(x1);
    f.get(qp).v128Load(o2 + xVecOff, 4).set(x2);
    f.get(qp).v128Load(o3 + xVecOff, 4).set(x3);
    f.get(qp).i32Load(o0 + sumOff).i32Const(5).i32Shl().set(sum32_0);
    f.get(qp).i32Load(o1 + sumOff).i32Const(5).i32Shl().set(sum32_1);
    f.get(qp).i32Load(o2 + sumOff).i32Const(5).i32Shl().set(sum32_2);
    f.get(qp).i32Load(o3 + sumOff).i32Const(5).i32Shl().set(sum32_3);
    f.get(qp).f32Load(o0).set(dx0);
    f.get(qp).f32Load(o1).set(dx1);
    f.get(qp).f32Load(o2).set(dx2);
    f.get(qp).f32Load(o3).set(dx3);
  };

  const emitSliceRow = (
    wpLoc: number,
    sbLoc: number,
    qlBase: number,
    qhBase: number,
    scBase: number,
    s: number,
  ) => {
    const sOff = s * 16;
    f.get(wpLoc).v128Load(qlBase + sOff).set(qla);
    f.get(wpLoc).v128Load(qlBase + 32 + sOff).set(qlb);
    f.get(wpLoc).v128Load(qhBase + sOff).set(qhv);

    const emitGroup = (
      xLoc: number,
      sum32Loc: number,
      dxLoc: number,
      scOff: number,
      buildQ6: () => void,
    ) => {
      buildQ6();
      f.set(q6v);
      f.get(zero).set(dotV);
      f.dotI8I7Add(xLoc, q6v, dotV, useRelaxed);
      f.hsumI32x4(dotV).get(sum32Loc).i32Sub().f32ConvertI32S();
      f.get(wpLoc).i32Load8S(scBase + scOff).f32ConvertI32S().f32Mul();
      f.get(dxLoc).f32Mul().get(sbLoc).f32Add().set(sbLoc);
    };

    emitGroup(x0, sum32_0, dx0, 0 + s, () => {
      f.get(qla).get(mask0f).v128And()
        .get(qhv).i32Const(4).i8x16Shl().get(mask30).v128And().v128Or();
    });
    emitGroup(x1, sum32_1, dx1, 2 + s, () => {
      f.get(qlb).get(mask0f).v128And()
        .get(qhv).i32Const(2).i8x16Shl().get(mask30).v128And().v128Or();
    });
    emitGroup(x2, sum32_2, dx2, 4 + s, () => {
      f.get(qla).i32Const(4).i8x16ShrU()
        .get(qhv).get(mask30).v128And().v128Or();
    });
    emitGroup(x3, sum32_3, dx3, 6 + s, () => {
      f.get(qlb).i32Const(4).i8x16ShrU()
        .get(qhv).i32Const(2).i8x16ShrU().get(mask30).v128And().v128Or();
    });
  };

  // 4-row register-blocked loop
  f.block().loop();
  f.get(r).i32Const(3).i32Add().get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add()
    .tee(wp0).get(rowBytes).i32Add()
    .tee(wp1).get(rowBytes).i32Add()
    .tee(wp2).get(rowBytes).i32Add().set(wp3);
  f.get(1).set(qp);
  f.f32Const(0).tee(acc0).tee(acc1).tee(acc2).set(acc3);
  f.i32Const(0).set(sb);

  f.block().loop();
  f.get(sb).get(nSuper).i32GeS().brIf(1);
  f.f32Const(0).tee(sb0).tee(sb1).tee(sb2).set(sb3);

  for (let h = 0; h < 2; h++) {
    const qlBase = h * 64;
    const qhBase = 128 + h * 32;
    const scBase = 192 + h * 8;
    const blkBase = h * 4;

    for (let s = 0; s < 2; s++) {
      const xVecOff = s === 0 ? 16 : 32;
      const sumOff = s === 0 ? 4 : 8;
      loadSliceActivations(blkBase, xVecOff, sumOff);
      emitSliceRow(wp0, sb0, qlBase, qhBase, scBase, s);
      emitSliceRow(wp1, sb1, qlBase, qhBase, scBase, s);
      emitSliceRow(wp2, sb2, qlBase, qhBase, scBase, s);
      emitSliceRow(wp3, sb3, qlBase, qhBase, scBase, s);
    }
  }

  f.get(sb0).loadF16AsF32(wp0, 208).f32Mul().get(acc0).f32Add().set(acc0);
  f.get(sb1).loadF16AsF32(wp1, 208).f32Mul().get(acc1).f32Add().set(acc1);
  f.get(sb2).loadF16AsF32(wp2, 208).f32Mul().get(acc2).f32Add().set(acc2);
  f.get(sb3).loadF16AsF32(wp3, 208).f32Mul().get(acc3).f32Add().set(acc3);

  f.get(wp0).i32Const(210).i32Add().set(wp0);
  f.get(wp1).i32Const(210).i32Add().set(wp1);
  f.get(wp2).i32Const(210).i32Add().set(wp2);
  f.get(wp3).i32Const(210).i32Add().set(wp3);
  f.get(qp).i32Const(8 * Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(sb).i32Const(1).i32Add().set(sb);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().set(op);
  f.get(op).get(acc0).f32Store(0);
  f.get(op).get(acc1).f32Store(4);
  f.get(op).get(acc2).f32Store(8);
  f.get(op).get(acc3).f32Store(12);
  f.get(r).i32Const(4).i32Add().set(r);
  f.br(0).end().end();

  // Remainder rows (0..3)
  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(wp0);
  f.get(1).set(qp);
  f.f32Const(0).set(acc0);
  f.i32Const(0).set(sb);

  f.block().loop();
  f.get(sb).get(nSuper).i32GeS().brIf(1);
  f.f32Const(0).set(sb0);
  for (let h = 0; h < 2; h++) {
    const qlBase = h * 64;
    const qhBase = 128 + h * 32;
    const scBase = 192 + h * 8;
    const blkBase = h * 4;
    for (let s = 0; s < 2; s++) {
      const xVecOff = s === 0 ? 16 : 32;
      const sumOff = s === 0 ? 4 : 8;
      loadSliceActivations(blkBase, xVecOff, sumOff);
      emitSliceRow(wp0, sb0, qlBase, qhBase, scBase, s);
    }
  }
  f.get(sb0).loadF16AsF32(wp0, 208).f32Mul().get(acc0).f32Add().set(acc0);
  f.get(wp0).i32Const(210).i32Add().set(wp0);
  f.get(qp).i32Const(8 * Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(sb).i32Const(1).i32Add().set(sb);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc0).f32Store(0);
  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();
  return f;
}

/** 4-token batched Q6_K × Q8_0_ACT VNNI/Int8 prefill kernel. */
function buildMulMatQ6KQ8Batch4(useRelaxed: boolean): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const nSuper = f.local(I32);
  const rowBytes = f.local(I32);
  const qStride = f.local(I32);
  const oStride = f.local(I32);
  const wp = f.local(I32);
  const qp0 = f.local(I32);
  const qp1 = f.local(I32);
  const qp2 = f.local(I32);
  const qp3 = f.local(I32);
  const sb = f.local(I32);
  const scF = f.local(F32);
  const dF = f.local(F32);
  const acc0 = f.local(F32);
  const acc1 = f.local(F32);
  const acc2 = f.local(F32);
  const acc3 = f.local(F32);
  const sb0 = f.local(F32);
  const sb1 = f.local(F32);
  const sb2 = f.local(F32);
  const sb3 = f.local(F32);
  const mask0f = f.local(V128);
  const mask30 = f.local(V128);
  const zero = f.local(V128);
  const qla = f.local(V128);
  const qlb = f.local(V128);
  const qhv = f.local(V128);
  const q6v = f.local(V128);
  const xtmp = f.local(V128);
  const dotV = f.local(V128);

  f.i32Const(0x0f).i8x16Splat().set(mask0f);
  f.i32Const(0x30).i8x16Splat().set(mask30);
  f.i32Const(0).i32x4Splat().set(zero);
  f.get(3).i32Const(8).i32ShrS().set(nSuper);
  f.get(nSuper).i32Const(210).i32Mul().set(rowBytes);
  f.get(nSuper).i32Const(8 * Q8_0_ACT_BLOCK_BYTES).i32Mul().set(qStride);
  f.get(5).set(oStride);
  f.i32Const(0).set(r);

  const emitTokQ6 = (qpLoc: number, qBlkOff: number, xVecOff: number, sumOff: number, sbLoc: number) => {
    f.get(zero).set(dotV);
    f.get(qpLoc).v128Load(qBlkOff + xVecOff, 4).set(xtmp);
    f.dotI8I7Add(xtmp, q6v, dotV, useRelaxed);
    f.hsumI32x4(dotV).get(qpLoc).i32Load(qBlkOff + sumOff).i32Const(5).i32Shl().i32Sub().f32ConvertI32S();
    f.get(scF).f32Mul().get(qpLoc).f32Load(qBlkOff).f32Mul().get(sbLoc).f32Add().set(sbLoc);
  };

  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(wp);
  f.get(1).tee(qp0).get(qStride).i32Add().tee(qp1).get(qStride).i32Add().tee(qp2).get(qStride).i32Add().set(qp3);
  f.f32Const(0).set(acc0);
  f.f32Const(0).set(acc1);
  f.f32Const(0).set(acc2);
  f.f32Const(0).set(acc3);
  f.i32Const(0).set(sb);

  f.block().loop();
  f.get(sb).get(nSuper).i32GeS().brIf(1);
  f.f32Const(0).set(sb0);
  f.f32Const(0).set(sb1);
  f.f32Const(0).set(sb2);
  f.f32Const(0).set(sb3);

  for (let h = 0; h < 2; h++) {
    const qlBase = h * 64;
    const qhBase = 128 + h * 32;
    const scBase = 192 + h * 8;
    const blkBase = h * 4;

    for (let s = 0; s < 2; s++) {
      const sOff = s * 16;
      const xVecOff = s === 0 ? 16 : 32;
      const sumOff = s === 0 ? 4 : 8;

      f.get(wp).v128Load(qlBase + sOff).set(qla);
      f.get(wp).v128Load(qlBase + 32 + sOff).set(qlb);
      f.get(wp).v128Load(qhBase + sOff).set(qhv);

      const emitGroup4 = (gIdx: number, scOff: number, buildQ6: () => void) => {
        const qBlkOff = (blkBase + gIdx) * Q8_0_ACT_BLOCK_BYTES;
        f.get(wp).i32Load8S(scBase + scOff).f32ConvertI32S().set(scF);
        buildQ6();
        f.set(q6v);
        emitTokQ6(qp0, qBlkOff, xVecOff, sumOff, sb0);
        emitTokQ6(qp1, qBlkOff, xVecOff, sumOff, sb1);
        emitTokQ6(qp2, qBlkOff, xVecOff, sumOff, sb2);
        emitTokQ6(qp3, qBlkOff, xVecOff, sumOff, sb3);
      };

      emitGroup4(0, 0 + s, () => {
        f.get(qla).get(mask0f).v128And()
          .get(qhv).i32Const(4).i8x16Shl().get(mask30).v128And().v128Or();
      });
      emitGroup4(1, 2 + s, () => {
        f.get(qlb).get(mask0f).v128And()
          .get(qhv).i32Const(2).i8x16Shl().get(mask30).v128And().v128Or();
      });
      emitGroup4(2, 4 + s, () => {
        f.get(qla).i32Const(4).i8x16ShrU()
          .get(qhv).get(mask30).v128And().v128Or();
      });
      emitGroup4(3, 6 + s, () => {
        f.get(qlb).i32Const(4).i8x16ShrU()
          .get(qhv).i32Const(2).i8x16ShrU().get(mask30).v128And().v128Or();
      });
    }
  }

  f.loadF16AsF32(wp, 208).set(dF);
  f.get(sb0).get(dF).f32Mul().get(acc0).f32Add().set(acc0);
  f.get(sb1).get(dF).f32Mul().get(acc1).f32Add().set(acc1);
  f.get(sb2).get(dF).f32Mul().get(acc2).f32Add().set(acc2);
  f.get(sb3).get(dF).f32Mul().get(acc3).f32Add().set(acc3);

  f.get(wp).i32Const(210).i32Add().set(wp);
  f.get(qp0).i32Const(8 * Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp0);
  f.get(qp1).i32Const(8 * Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp1);
  f.get(qp2).i32Const(8 * Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp2);
  f.get(qp3).i32Const(8 * Q8_0_ACT_BLOCK_BYTES).i32Add().set(qp3);
  f.get(sb).i32Const(1).i32Add().set(sb);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().tee(wp).get(acc0).f32Store(0);
  f.get(wp).get(oStride).i32Add().tee(wp).get(acc1).f32Store(0);
  f.get(wp).get(oStride).i32Add().tee(wp).get(acc2).f32Store(0);
  f.get(wp).get(oStride).i32Add().get(acc3).f32Store(0);

  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();
  return f;
}

/** 4-token batched F32 MUL_MAT (uses hardware FMA `f32x4.relaxed_madd` when supported). */
function buildMulMatF32Batch4(useRelaxed: boolean): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const c = f.local(I32);
  const rowBytes = f.local(I32);
  const oStride = f.local(I32);
  const wp = f.local(I32);
  const xp0 = f.local(I32);
  const xp1 = f.local(I32);
  const xp2 = f.local(I32);
  const xp3 = f.local(I32);
  const wv = f.local(V128);
  const s0 = f.local(V128);
  const s1 = f.local(V128);
  const s2 = f.local(V128);
  const s3 = f.local(V128);

  f.get(3).i32Const(2).i32Shl().set(rowBytes);
  f.get(5).set(oStride);
  f.i32Const(0).set(r);

  const emitMadd = (xpLoc: number, sLoc: number) => {
    if (useRelaxed) {
      f.get(wv).get(xpLoc).v128Load(0).get(sLoc).f32x4RelaxedMadd().set(sLoc);
    } else {
      f.get(wv).get(xpLoc).v128Load(0).f32x4Mul().get(sLoc).f32x4Add().set(sLoc);
    }
  };

  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(wp);
  f.get(1).tee(xp0).get(rowBytes).i32Add().tee(xp1).get(rowBytes).i32Add().tee(xp2).get(rowBytes).i32Add().set(xp3);
  f.f32Const(0).f32x4Splat().tee(s0).tee(s1).tee(s2).set(s3);
  f.i32Const(0).set(c);

  f.block().loop();
  f.get(c).i32Const(4).i32Add().get(3).i32GtS().brIf(1);
  f.get(wp).v128Load(0).set(wv);
  emitMadd(xp0, s0);
  emitMadd(xp1, s1);
  emitMadd(xp2, s2);
  emitMadd(xp3, s3);
  f.get(wp).i32Const(16).i32Add().set(wp);
  f.get(xp0).i32Const(16).i32Add().set(xp0);
  f.get(xp1).i32Const(16).i32Add().set(xp1);
  f.get(xp2).i32Const(16).i32Add().set(xp2);
  f.get(xp3).i32Const(16).i32Add().set(xp3);
  f.get(c).i32Const(4).i32Add().set(c);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().tee(wp).hsumF32x4ToF64(s0).f32DemoteF64().f32Store(0);
  f.get(wp).get(oStride).i32Add().tee(wp).hsumF32x4ToF64(s1).f32DemoteF64().f32Store(0);
  f.get(wp).get(oStride).i32Add().tee(wp).hsumF32x4ToF64(s2).f32DemoteF64().f32Store(0);
  f.get(wp).get(oStride).i32Add().hsumF32x4ToF64(s3).f32DemoteF64().f32Store(0);

  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();
  return f;
}

/**
 * dot_q4_f32(q4Ptr, xPtr, n) -> f64:
 * vRAM Tier-2 Cold Q4 KV attention dot product.
 * Each 16-byte chunk holds 32 symmetric 4-bit signed values `q ∈ [-7..+7]` encoded as `u4 = q + 8`
 * (low nibbles = indices 0..15, high nibbles = indices 16..31).
 */
function buildDotQ4F32(): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32], [F64]); // 0:q4Ptr, 1:xPtr, 2:n
  const i = f.local(I32);
  const qp = f.local(I32);
  const xp = f.local(I32);
  const mask0f = f.local(V128);
  const bias8 = f.local(V128);
  const raw = f.local(V128);
  const s8 = f.local(V128);
  const acc = f.local(V128);
  const tmp16 = f.local(V128);

  f.i32Const(0x0f).i8x16Splat().set(mask0f);
  f.i32Const(8).i8x16Splat().set(bias8);
  f.f32Const(0).f32x4Splat().set(acc);
  f.get(0).set(qp);
  f.get(1).set(xp);
  f.i32Const(0).set(i);

  f.block().loop();
  f.get(i).i32Const(32).i32Add().get(2).i32GtS().brIf(1);
  f.get(qp).v128Load(0).tee(raw).get(mask0f).v128And().get(bias8).i8x16Sub().set(s8);
  f.dotS8x16WithX(s8, xp, 0, acc, tmp16);
  f.get(raw).i32Const(4).i8x16ShrU().get(bias8).i8x16Sub().set(s8);
  f.dotS8x16WithX(s8, xp, 64, acc, tmp16);
  f.get(qp).i32Const(16).i32Add().set(qp);
  f.get(xp).i32Const(128).i32Add().set(xp);
  f.get(i).i32Const(32).i32Add().set(i);
  f.br(0).end().end();

  f.hsumF32x4ToF64(acc);
  return f;
}

/**
 * axpy_q4_f32(dstPtr, q4Ptr, n, scale):
 * vRAM Tier-2 Cold Q4 value accumulation: dst[j] += (q4[j] - 8) * scale.
 */
function buildAxpyQ4F32(): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, F32]); // 0:dstPtr, 1:q4Ptr, 2:n, 3:scale
  const i = f.local(I32);
  const dp = f.local(I32);
  const qp = f.local(I32);
  const mask0f = f.local(V128);
  const bias8 = f.local(V128);
  const vscale = f.local(V128);
  const raw = f.local(V128);
  const s8 = f.local(V128);
  const tmp16 = f.local(V128);

  f.i32Const(0x0f).i8x16Splat().set(mask0f);
  f.i32Const(8).i8x16Splat().set(bias8);
  f.get(3).f32x4Splat().set(vscale);
  f.get(0).set(dp);
  f.get(1).set(qp);
  f.i32Const(0).set(i);

  const emitAxpy16 = (baseOff: number) => {
    f.get(s8).i16x8ExtendLowI8x16S().set(tmp16);
    f.get(dp)
      .get(tmp16).i32x4ExtendLowI16x8S().f32x4ConvertI32x4S().get(vscale).f32x4Mul()
      .get(dp).v128Load(baseOff).f32x4Add().v128Store(baseOff);
    f.get(dp)
      .get(tmp16).i32x4ExtendHighI16x8S().f32x4ConvertI32x4S().get(vscale).f32x4Mul()
      .get(dp).v128Load(baseOff + 16).f32x4Add().v128Store(baseOff + 16);

    f.get(s8).i16x8ExtendHighI8x16S().set(tmp16);
    f.get(dp)
      .get(tmp16).i32x4ExtendLowI16x8S().f32x4ConvertI32x4S().get(vscale).f32x4Mul()
      .get(dp).v128Load(baseOff + 32).f32x4Add().v128Store(baseOff + 32);
    f.get(dp)
      .get(tmp16).i32x4ExtendHighI16x8S().f32x4ConvertI32x4S().get(vscale).f32x4Mul()
      .get(dp).v128Load(baseOff + 48).f32x4Add().v128Store(baseOff + 48);
  };

  f.block().loop();
  f.get(i).i32Const(32).i32Add().get(2).i32GtS().brIf(1);
  f.get(qp).v128Load(0).tee(raw).get(mask0f).v128And().get(bias8).i8x16Sub().set(s8);
  emitAxpy16(0);
  f.get(raw).i32Const(4).i8x16ShrU().get(bias8).i8x16Sub().set(s8);
  emitAxpy16(64);
  f.get(qp).i32Const(16).i32Add().set(qp);
  f.get(dp).i32Const(128).i32Add().set(dp);
  f.get(i).i32Const(32).i32Add().set(i);
  f.br(0).end().end();
  return f;
}

/**
 * Builds `add_f32(aPtr, bPtr, outPtr, n)` — 4× unrolled 16-float SIMD128 residual/bias vector add.
 */
function buildAddF32(): FuncBuilder {
  // params: aPtr(0), bPtr(1), outPtr(2), n(3)
  const f = new FuncBuilder([I32, I32, I32, I32]);
  const i = f.local(I32);
  const n16 = f.local(I32);
  const n4 = f.local(I32);
  const ap = f.local(I32);
  const bp = f.local(I32);
  const op = f.local(I32);

  f.get(3).i32Const(-16).i32And().set(n16);
  f.get(3).i32Const(-4).i32And().set(n4);
  f.get(0).set(ap);
  f.get(1).set(bp);
  f.get(2).set(op);
  f.i32Const(0).set(i);

  f.block().loop();
  f.get(i).get(n16).i32GeS().brIf(1);
  f.get(op).get(ap).v128Load(0).get(bp).v128Load(0).f32x4Add().v128Store(0);
  f.get(op).get(ap).v128Load(16).get(bp).v128Load(16).f32x4Add().v128Store(16);
  f.get(op).get(ap).v128Load(32).get(bp).v128Load(32).f32x4Add().v128Store(32);
  f.get(op).get(ap).v128Load(48).get(bp).v128Load(48).f32x4Add().v128Store(48);
  f.get(ap).i32Const(64).i32Add().set(ap);
  f.get(bp).i32Const(64).i32Add().set(bp);
  f.get(op).i32Const(64).i32Add().set(op);
  f.get(i).i32Const(16).i32Add().set(i);
  f.br(0).end().end();

  f.block().loop();
  f.get(i).get(n4).i32GeS().brIf(1);
  f.get(op).get(ap).v128Load(0).get(bp).v128Load(0).f32x4Add().v128Store(0);
  f.get(ap).i32Const(16).i32Add().set(ap);
  f.get(bp).i32Const(16).i32Add().set(bp);
  f.get(op).i32Const(16).i32Add().set(op);
  f.get(i).i32Const(4).i32Add().set(i);
  f.br(0).end().end();

  f.block().loop();
  f.get(i).get(3).i32GeS().brIf(1);
  f.get(op).get(ap).f32Load(0).get(bp).f32Load(0).f32Add().f32Store(0);
  f.get(ap).i32Const(4).i32Add().set(ap);
  f.get(bp).i32Const(4).i32Add().set(bp);
  f.get(op).i32Const(4).i32Add().set(op);
  f.get(i).i32Const(1).i32Add().set(i);
  f.br(0).end().end();

  return f;
}

/**
 * Builds `swiglu_f32(gatePtr, upPtr, outPtr, n)` — 4-wide SIMD128 SiLU(gate) * up
 * with range-reduced degree-5 minimax exp(-g) (< 2e-6 relative error, zero JS Math.exp loops).
 */
function buildSwigluF32(useRelaxed: boolean): FuncBuilder {
  // params: gatePtr(0), upPtr(1), outPtr(2), n(3)
  const f = new FuncBuilder([I32, I32, I32, I32]);
  const i = f.local(I32);
  const n4 = f.local(I32);
  const gp = f.local(I32);
  const up = f.local(I32);
  const op = f.local(I32);
  const g = f.local(V128);
  const z = f.local(V128);
  const k = f.local(V128);
  const r = f.local(V128);
  const p = f.local(V128);
  const exp2k = f.local(V128);
  const cMin = f.local(V128);
  const cMax = f.local(V128);
  const cLog2e = f.local(V128);
  const cLn2 = f.local(V128);
  const cOne = f.local(V128);
  const cHalf = f.local(V128);
  const c6 = f.local(V128);
  const c24 = f.local(V128);
  const c120 = f.local(V128);
  const cBias127 = f.local(V128);

  f.f32Const(-88.0).f32x4Splat().set(cMin);
  f.f32Const(88.0).f32x4Splat().set(cMax);
  f.f32Const(1.4426950408889634).f32x4Splat().set(cLog2e);
  f.f32Const(0.6931471805599453).f32x4Splat().set(cLn2);
  f.f32Const(1.0).f32x4Splat().set(cOne);
  f.f32Const(0.5).f32x4Splat().set(cHalf);
  f.f32Const(0.1666666716337204).f32x4Splat().set(c6);
  f.f32Const(0.0416666679084301).f32x4Splat().set(c24);
  f.f32Const(0.008333333767950535).f32x4Splat().set(c120);
  f.i32Const(127).i32x4Splat().set(cBias127);

  const emitMaddConst = (polyLoc: number, rLoc: number, constLoc: number) => {
    if (useRelaxed) {
      f.get(polyLoc).get(rLoc).get(constLoc).f32x4RelaxedMadd().set(polyLoc);
    } else {
      f.get(polyLoc).get(rLoc).f32x4Mul().get(constLoc).f32x4Add().set(polyLoc);
    }
  };

  // Evaluates silu(g) = g / (1 + exp(clamp(-g, -88, 88))) on v128 register `g`, leaves result on stack
  const emitSiluVec = () => {
    f.get(g).f32x4Neg().get(cMin).f32x4Pmax().get(cMax).f32x4Pmin().set(z);
    f.get(z).get(cLog2e).f32x4Mul().f32x4Nearest().set(k);
    f.get(z).get(k).get(cLn2).f32x4Mul().f32x4Sub().set(r);
    f.get(c120).set(p);
    emitMaddConst(p, r, c24);
    emitMaddConst(p, r, c6);
    emitMaddConst(p, r, cHalf);
    emitMaddConst(p, r, cOne);
    emitMaddConst(p, r, cOne);
    f.get(k).i32x4TruncSatF32x4S().get(cBias127).i32x4Add().i32Const(23).i32x4Shl().set(exp2k);
    f.get(g).get(cOne).get(p).get(exp2k).f32x4Mul().f32x4Add().f32x4Div();
  };

  f.get(3).i32Const(-4).i32And().set(n4);
  f.get(0).set(gp);
  f.get(1).set(up);
  f.get(2).set(op);
  f.i32Const(0).set(i);

  f.block().loop();
  f.get(i).get(n4).i32GeS().brIf(1);
  f.get(gp).v128Load(0).set(g);
  f.get(op);
  emitSiluVec();
  f.get(up).v128Load(0).f32x4Mul().v128Store(0);
  f.get(gp).i32Const(16).i32Add().set(gp);
  f.get(up).i32Const(16).i32Add().set(up);
  f.get(op).i32Const(16).i32Add().set(op);
  f.get(i).i32Const(4).i32Add().set(i);
  f.br(0).end().end();

  f.block().loop();
  f.get(i).get(3).i32GeS().brIf(1);
  f.get(gp).f32Load(0).f32x4Splat().set(g);
  f.get(op);
  emitSiluVec();
  f.f32x4Extract(0).get(up).f32Load(0).f32Mul().f32Store(0);
  f.get(gp).i32Const(4).i32Add().set(gp);
  f.get(up).i32Const(4).i32Add().set(up);
  f.get(op).i32Const(4).i32Add().set(op);
  f.get(i).i32Const(1).i32Add().set(i);
  f.br(0).end().end();

  return f;
}

/**
 * Fused 16-token Hot F32 page attention scores:
 * `scores[t] = dot(Q, K_t) * invScale` for `t = 0 .. count-1` in ONE WASM call (16x fewer calls!).
 */
function buildAttnPageScoresF32(useRelaxed: boolean): FuncBuilder {
  // params: qPtr(0), kBasePtr(1), scoresPtr(2), headDim(3), kvStrideBytes(4), count(5), invScale(6:f32)
  const f = new FuncBuilder([I32, I32, I32, I32, I32, I32, F32]);
  const t = f.local(I32);
  const d = f.local(I32);
  const d4 = f.local(I32);
  const qp = f.local(I32);
  const kp = f.local(I32);
  const kRow = f.local(I32);
  const sp = f.local(I32);
  const acc = f.local(V128);
  const sum = f.local(F64);

  f.get(3).i32Const(-4).i32And().set(d4);
  f.get(1).set(kRow);
  f.get(2).set(sp);
  f.i32Const(0).set(t);

  f.block().loop();
  f.get(t).get(5).i32GeS().brIf(1);
  f.get(0).set(qp);
  f.get(kRow).set(kp);
  f.f32Const(0).f32x4Splat().set(acc);
  f.i32Const(0).set(d);

  f.block().loop();
  f.get(d).get(d4).i32GeS().brIf(1);
  if (useRelaxed) {
    f.get(qp).v128Load(0).get(kp).v128Load(0).get(acc).f32x4RelaxedMadd().set(acc);
  } else {
    f.get(qp).v128Load(0).get(kp).v128Load(0).f32x4Mul().get(acc).f32x4Add().set(acc);
  }
  f.get(qp).i32Const(16).i32Add().set(qp);
  f.get(kp).i32Const(16).i32Add().set(kp);
  f.get(d).i32Const(4).i32Add().set(d);
  f.br(0).end().end();

  f.hsumF32x4ToF64(acc).set(sum);
  f.block().loop();
  f.get(d).get(3).i32GeS().brIf(1);
  f.get(qp).f32Load(0).get(kp).f32Load(0).f32Mul().f64PromoteF32().get(sum).f64Add().set(sum);
  f.get(qp).i32Const(4).i32Add().set(qp);
  f.get(kp).i32Const(4).i32Add().set(kp);
  f.get(d).i32Const(1).i32Add().set(d);
  f.br(0).end().end();

  f.get(sp).get(sum).f32DemoteF64().get(6).f32Mul().f32Store(0);
  f.get(kRow).get(4).i32Add().set(kRow);
  f.get(sp).i32Const(4).i32Add().set(sp);
  f.get(t).i32Const(1).i32Add().set(t);
  f.br(0).end().end();

  return f;
}

/**
 * Fused 16-token Hot F32 page value accumulation:
 * Rescales `out[d..d+3] *= rescale` ONCE in registers, adds `sum_t weights[t] * V_t[d..d+3]`,
 * and stores `out[d..d+3]` ONCE (16x less `out` memory traffic + 16x fewer JS-WASM calls!).
 */
function buildAttnPageAccumF32(useRelaxed: boolean): FuncBuilder {
  // params: outPtr(0), vBasePtr(1), weightsPtr(2), headDim(3), kvStrideBytes(4), count(5), rescale(6:f32)
  const f = new FuncBuilder([I32, I32, I32, I32, I32, I32, F32]);
  const d = f.local(I32);
  const d4 = f.local(I32);
  const t = f.local(I32);
  const vp = f.local(I32);
  const wp = f.local(I32);
  const op = f.local(I32);
  const acc = f.local(V128);
  const vRescale = f.local(V128);
  const accS = f.local(F32);

  f.get(3).i32Const(-4).i32And().set(d4);
  f.get(6).f32x4Splat().set(vRescale);
  f.get(0).set(op);
  f.i32Const(0).set(d);

  // Outer loop over 4-float SIMD lanes of headDim: load out[d..d+3] ONCE, accumulate all `count` tokens
  f.block().loop();
  f.get(d).get(d4).i32GeS().brIf(1);
  f.get(op).v128Load(0).get(vRescale).f32x4Mul().set(acc);
  f.get(1).get(d).i32Const(2).i32Shl().i32Add().set(vp);
  f.get(2).set(wp);
  f.i32Const(0).set(t);

  f.block().loop();
  f.get(t).get(5).i32GeS().brIf(1);
  if (useRelaxed) {
    f.get(vp).v128Load(0).get(wp).f32Load(0).f32x4Splat().get(acc).f32x4RelaxedMadd().set(acc);
  } else {
    f.get(vp).v128Load(0).get(wp).f32Load(0).f32x4Splat().f32x4Mul().get(acc).f32x4Add().set(acc);
  }
  f.get(vp).get(4).i32Add().set(vp);
  f.get(wp).i32Const(4).i32Add().set(wp);
  f.get(t).i32Const(1).i32Add().set(t);
  f.br(0).end().end();

  f.get(op).get(acc).v128Store(0);
  f.get(op).i32Const(16).i32Add().set(op);
  f.get(d).i32Const(4).i32Add().set(d);
  f.br(0).end().end();

  // Scalar tail if headDim is not a multiple of 4
  f.block().loop();
  f.get(d).get(3).i32GeS().brIf(1);
  f.get(op).f32Load(0).get(6).f32Mul().set(accS);
  f.get(1).get(d).i32Const(2).i32Shl().i32Add().set(vp);
  f.get(2).set(wp);
  f.i32Const(0).set(t);
  f.block().loop();
  f.get(t).get(5).i32GeS().brIf(1);
  f.get(vp).f32Load(0).get(wp).f32Load(0).f32Mul().get(accS).f32Add().set(accS);
  f.get(vp).get(4).i32Add().set(vp);
  f.get(wp).i32Const(4).i32Add().set(wp);
  f.get(t).i32Const(1).i32Add().set(t);
  f.br(0).end().end();
  f.get(op).get(accS).f32Store(0);
  f.get(op).i32Const(4).i32Add().set(op);
  f.get(d).i32Const(1).i32Add().set(d);
  f.br(0).end().end();

  return f;
}

/**
 * Fused 16-token Warm Q8 page attention scores:
 * `scores[t] = dot_q8_f32(kQ8_t, Q) * kScale_t * invScale` for `t = 0 .. count-1`.
 */
function buildAttnPageScoresQ8(dotQ8Idx: number): FuncBuilder {
  // params: qPtr(0), kQ8BasePtr(1), kScBasePtr(2), scoresPtr(3), headDim(4), kvStrideBytes(5), scStrideBytes(6), count(7), invScale(8:f32)
  const f = new FuncBuilder([I32, I32, I32, I32, I32, I32, I32, I32, F32]);
  const t = f.local(I32);
  const kp = f.local(I32);
  const scp = f.local(I32);
  const sp = f.local(I32);

  f.get(1).set(kp);
  f.get(2).set(scp);
  f.get(3).set(sp);
  f.i32Const(0).set(t);

  f.block().loop();
  f.get(t).get(7).i32GeS().brIf(1);
  f.get(sp);
  f.get(kp).get(0).get(4).call(dotQ8Idx).f32DemoteF64();
  f.get(scp).f32Load(0).f32Mul().get(8).f32Mul().f32Store(0);
  f.get(kp).get(5).i32Add().set(kp);
  f.get(scp).get(6).i32Add().set(scp);
  f.get(sp).i32Const(4).i32Add().set(sp);
  f.get(t).i32Const(1).i32Add().set(t);
  f.br(0).end().end();

  return f;
}

/**
 * Fused 16-token Warm Q8 page value accumulation:
 * Rescales `out` once and accumulates `sum_t weights[t] * vQ8_t` in ONE WASM call.
 */
function buildAttnPageAccumQ8(scaleIdx: number, axpyQ8Idx: number): FuncBuilder {
  // params: outPtr(0), vQ8BasePtr(1), weightsPtr(2), headDim(3), kvStrideBytes(4), count(5), rescale(6:f32)
  const f = new FuncBuilder([I32, I32, I32, I32, I32, I32, F32]);
  const t = f.local(I32);
  const vp = f.local(I32);
  const wp = f.local(I32);

  f.get(0).get(3).get(6).call(scaleIdx);
  f.get(1).set(vp);
  f.get(2).set(wp);
  f.i32Const(0).set(t);

  f.block().loop();
  f.get(t).get(5).i32GeS().brIf(1);
  f.get(0).get(vp).get(3).get(wp).f32Load(0).call(axpyQ8Idx);
  f.get(vp).get(4).i32Add().set(vp);
  f.get(wp).i32Const(4).i32Add().set(wp);
  f.get(t).i32Const(1).i32Add().set(t);
  f.br(0).end().end();

  return f;
}

/**
 * Fused 16-token Cold Q4 page attention scores:
 * `scores[t] = dot_q4_f32(kQ4_t, Q) * kScale_t * invScale` for `t = 0 .. count-1`.
 */
function buildAttnPageScoresQ4(dotQ4Idx: number): FuncBuilder {
  // params: qPtr(0), kQ4BasePtr(1), kScBasePtr(2), scoresPtr(3), headDim(4), kvStrideBytes(5), scStrideBytes(6), count(7), invScale(8:f32)
  const f = new FuncBuilder([I32, I32, I32, I32, I32, I32, I32, I32, F32]);
  const t = f.local(I32);
  const kp = f.local(I32);
  const scp = f.local(I32);
  const sp = f.local(I32);

  f.get(1).set(kp);
  f.get(2).set(scp);
  f.get(3).set(sp);
  f.i32Const(0).set(t);

  f.block().loop();
  f.get(t).get(7).i32GeS().brIf(1);
  f.get(sp);
  f.get(kp).get(0).get(4).call(dotQ4Idx).f32DemoteF64();
  f.get(scp).f32Load(0).f32Mul().get(8).f32Mul().f32Store(0);
  f.get(kp).get(5).i32Add().set(kp);
  f.get(scp).get(6).i32Add().set(scp);
  f.get(sp).i32Const(4).i32Add().set(sp);
  f.get(t).i32Const(1).i32Add().set(t);
  f.br(0).end().end();

  return f;
}

/**
 * Fused 16-token Cold Q4 page value accumulation:
 * Rescales `out` once and accumulates `sum_t weights[t] * vQ4_t` in ONE WASM call.
 */
function buildAttnPageAccumQ4(scaleIdx: number, axpyQ4Idx: number): FuncBuilder {
  // params: outPtr(0), vQ4BasePtr(1), weightsPtr(2), headDim(3), kvStrideBytes(4), count(5), rescale(6:f32)
  const f = new FuncBuilder([I32, I32, I32, I32, I32, I32, F32]);
  const t = f.local(I32);
  const vp = f.local(I32);
  const wp = f.local(I32);

  f.get(0).get(3).get(6).call(scaleIdx);
  f.get(1).set(vp);
  f.get(2).set(wp);
  f.i32Const(0).set(t);

  f.block().loop();
  f.get(t).get(5).i32GeS().brIf(1);
  f.get(0).get(vp).get(3).get(wp).f32Load(0).call(axpyQ4Idx);
  f.get(vp).get(4).i32Add().set(vp);
  f.get(wp).i32Const(4).i32Add().set(wp);
  f.get(t).i32Const(1).i32Add().set(t);
  f.br(0).end().end();

  return f;
}

/**
 * Zero-JS-view bulk memory copy inside Verdant WASM linear memory: `copy_bytes(dstPtr, srcPtr, byteLen)`.
 */
function buildCopyBytes(): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32]);
  f.get(0).get(1).get(2).memoryCopy();
  return f;
}

/**
 * In-Verdant F16 row decode via the low-memory F16 LUT: `decode_row_f16(srcPtr, dstPtr, n)`.
 */
function buildDecodeRowF16(): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32]);
  const i = f.local(I32);
  const sp = f.local(I32);
  const dp = f.local(I32);

  f.get(0).set(sp);
  f.get(1).set(dp);
  f.i32Const(0).set(i);

  f.block().loop();
  f.get(i).get(2).i32GeS().brIf(1);
  f.get(dp).loadF16AsF32(sp, 0).f32Store(0);
  f.get(sp).i32Const(2).i32Add().set(sp);
  f.get(dp).i32Const(4).i32Add().set(dp);
  f.get(i).i32Const(1).i32Add().set(i);
  f.br(0).end().end();

  return f;
}

/**
 * 100% In-Verdant WASM SIMD RoPE:
 * `rope_f32(vecPtr, cosPtr, sinPtr, nHeads, headDim, isNeox)`
 * Supports both NeoX split-half (`x[i], x[i + half]`, 4-wide f32x4 SIMD) and
 * LLaMA interleaved pairs (`x[2i], x[2i + 1]`) directly in Verdant linear memory!
 */
function buildRopeF32(): FuncBuilder {
  // params: vecPtr(0), cosPtr(1), sinPtr(2), nHeads(3), headDim(4), isNeox(5)
  const f = new FuncBuilder([I32, I32, I32, I32, I32, I32]);
  const h = f.local(I32);
  const half = f.local(I32);
  const halfBytes = f.local(I32);
  const headBytes = f.local(I32);
  const hPtr = f.local(I32);
  const p0 = f.local(I32);
  const p1 = f.local(I32);
  const cp = f.local(I32);
  const sp = f.local(I32);
  const i = f.local(I32);
  const x0 = f.local(F32);
  const x1 = f.local(F32);
  const c = f.local(F32);
  const s = f.local(F32);
  const v0 = f.local(V128);
  const v1 = f.local(V128);
  const vc = f.local(V128);
  const vs = f.local(V128);

  f.get(4).i32Const(1).i32ShrS().set(half);
  f.get(half).i32Const(2).i32Shl().set(halfBytes);
  f.get(4).i32Const(2).i32Shl().set(headBytes);
  f.i32Const(0).set(h);
  f.get(0).set(hPtr);

  f.block().loop();
  f.get(h).get(3).i32GeS().brIf(1);

  f.get(5).if_();
  // --- NeoX split-half RoPE: 4-wide f32x4 SIMD ---
  f.get(hPtr).set(p0);
  f.get(hPtr).get(halfBytes).i32Add().set(p1);
  f.get(1).set(cp);
  f.get(2).set(sp);
  f.i32Const(0).set(i);

  f.block().loop();
  f.get(i).i32Const(3).i32Add().get(half).i32GeS().brIf(1);
  f.get(p0).v128Load(0).set(v0);
  f.get(p1).v128Load(0).set(v1);
  f.get(cp).v128Load(0).set(vc);
  f.get(sp).v128Load(0).set(vs);
  // out0 = v0 * vc - v1 * vs
  f.get(p0).get(v0).get(vc).f32x4Mul().get(v1).get(vs).f32x4Mul().f32x4Sub().v128Store(0);
  // out1 = v0 * vs + v1 * vc
  f.get(p1).get(v0).get(vs).f32x4Mul().get(v1).get(vc).f32x4Mul().f32x4Add().v128Store(0);
  f.get(p0).i32Const(16).i32Add().set(p0);
  f.get(p1).i32Const(16).i32Add().set(p1);
  f.get(cp).i32Const(16).i32Add().set(cp);
  f.get(sp).i32Const(16).i32Add().set(sp);
  f.get(i).i32Const(4).i32Add().set(i);
  f.br(0).end().end();

  // NeoX scalar tail
  f.block().loop();
  f.get(i).get(half).i32GeS().brIf(1);
  f.get(p0).f32Load(0).set(x0);
  f.get(p1).f32Load(0).set(x1);
  f.get(cp).f32Load(0).set(c);
  f.get(sp).f32Load(0).set(s);
  f.get(p0).get(x0).get(c).f32Mul().get(x1).get(s).f32Mul().f32Sub().f32Store(0);
  f.get(p1).get(x0).get(s).f32Mul().get(x1).get(c).f32Mul().f32Add().f32Store(0);
  f.get(p0).i32Const(4).i32Add().set(p0);
  f.get(p1).i32Const(4).i32Add().set(p1);
  f.get(cp).i32Const(4).i32Add().set(cp);
  f.get(sp).i32Const(4).i32Add().set(sp);
  f.get(i).i32Const(1).i32Add().set(i);
  f.br(0).end().end();

  f.else_();
  // --- Standard LLaMA interleaved RoPE: (2i, 2i+1) ---
  f.get(hPtr).set(p0);
  f.get(1).set(cp);
  f.get(2).set(sp);
  f.i32Const(0).set(i);

  f.block().loop();
  f.get(i).get(half).i32GeS().brIf(1);
  f.get(p0).f32Load(0).set(x0);
  f.get(p0).f32Load(4).set(x1);
  f.get(cp).f32Load(0).set(c);
  f.get(sp).f32Load(0).set(s);
  f.get(p0).get(x0).get(c).f32Mul().get(x1).get(s).f32Mul().f32Sub().f32Store(0);
  f.get(p0).get(x0).get(s).f32Mul().get(x1).get(c).f32Mul().f32Add().f32Store(4);
  f.get(p0).i32Const(8).i32Add().set(p0);
  f.get(cp).i32Const(4).i32Add().set(cp);
  f.get(sp).i32Const(4).i32Add().set(sp);
  f.get(i).i32Const(1).i32Add().set(i);
  f.br(0).end().end();
  f.end();

  f.get(hPtr).get(headBytes).i32Add().set(hPtr);
  f.get(h).i32Const(1).i32Add().set(h);
  f.br(0).end().end();

  return f;
}

/**
 * 100% In-Verdant Per-Head RMSNorm (Qwen3 QK-Norm):
 * `rms_norm_per_head_f32(vecPtr, wPtr, nHeads, headDim, eps)`
 */
function buildRmsNormPerHeadF32(rmsNormIdx: number): FuncBuilder {
  // params: vecPtr(0), wPtr(1), nHeads(2), headDim(3), eps(4:f64)
  const f = new FuncBuilder([I32, I32, I32, I32, F64]);
  const h = f.local(I32);
  const ptr = f.local(I32);
  const headBytes = f.local(I32);

  f.get(0).set(ptr);
  f.get(3).i32Const(2).i32Shl().set(headBytes);
  f.i32Const(0).set(h);

  f.block().loop();
  f.get(h).get(2).i32GeS().brIf(1);
  f.get(ptr).get(1).get(ptr).get(3).get(4).call(rmsNormIdx);
  f.get(ptr).get(headBytes).i32Add().set(ptr);
  f.get(h).i32Const(1).i32Add().set(h);
  f.br(0).end().end();

  return f;
}

/**
 * Exact range-reduced degree-5 minimax `exp(x)` in float32 (`< 2e-7` relative error on `[-88, 88]`).
 * Used inside Verdant WASM `flash_attn_layer` so softmax never exits Verdant to JS `Math.exp`!
 */
function buildFastExpF32(): FuncBuilder {
  const f = new FuncBuilder([F32], [F32]);
  const x = f.local(F32);
  const kf = f.local(F32);
  const r = f.local(F32);
  const p = f.local(F32);

  // Clamp x to [-87.0, 88.0] so (k + 127) stays strictly inside valid IEEE-754 normal exponent range
  f.get(0).f32Const(-87.0).f32Max().f32Const(88.0).f32Min().set(x);
  // k = nearest(x * log2(e))
  f.get(x).f32Const(1.4426950408889634).f32Mul().f32Nearest().set(kf);
  // r = x - k * ln(2)  (|r| <= 0.3466)
  f.get(x).get(kf).f32Const(0.6931471805599453).f32Mul().f32Sub().set(r);

  // Horner evaluation: 1 + r*(1 + r*(0.5 + r*(1/6 + r*(1/24 + r*(1/120)))))
  f.f32Const(0.008333333767950535).get(r).f32Mul().f32Const(0.0416666679084301).f32Add()
    .get(r).f32Mul().f32Const(0.1666666716337204).f32Add()
    .get(r).f32Mul().f32Const(0.5).f32Add()
    .get(r).f32Mul().f32Const(1.0).f32Add()
    .get(r).f32Mul().f32Const(1.0).f32Add().set(p);

  // pow2k = f32.reinterpret_i32((i32(kf) + 127) << 23)
  f.get(kf).i32TruncSatF32S().i32Const(127).i32Add().i32Const(23).i32Shl().f32ReinterpretI32();
  f.get(p).f32Mul();
  return f;
}

/**
 * 100% In-Verdant Multi-Core Parallelizable 3-Tier FlashAttention (`flash_attn_layer_range`):
 * Executes Q heads `[hStart, hEnd)` × ALL `ceil(seqLen / 16)` pages across Hot F32 (state=1),
 * Warm Q8 (state=2), and Cold Q4 (state=3) with in-WASM online softmax, in-WASM `fast_exp_f32`,
 * and per-worker isolated 128-byte scratch (`scratchBase`: 64B scores + 64B weights) so `P`
 * shared-WASM threads scale long-context attention linearly with bit-exact parity!
 */
function buildFlashAttnLayerRange(
  fastExpIdx: number,
  scaleIdx: number,
  scoresF32Idx: number,
  accumF32Idx: number,
  scoresQ8Idx: number,
  accumQ8Idx: number,
  scoresQ4Idx: number,
  accumQ4Idx: number,
): FuncBuilder {
  // 29 I32 params:
  // 0: qPtr, 1: outPtr, 2: pageStatePtr, 3: pageSlotPtr
  // 4: hotKLayerPtr, 5: hotVLayerPtr
  // 6: q8KLayerPtr, 7: q8VLayerPtr, 8: q8KScLayerPtr, 9: q8VScLayerPtr
  // 10: q4KLayerPtr, 11: q4VLayerPtr, 12: q4KScLayerPtr, 13: q4VScLayerPtr
  // 14: nHeads, 15: nHeadsKv, 16: headDim, 17: seqLen
  // 18: hotPageBytes, 19: q8PageBytes, 20: q4PageBytes, 21: scPageBytes
  // 22: hotKvStrideBytes, 23: q8KvStrideBytes, 24: q4KvStrideBytes, 25: scStrideBytes
  // 26: hStart, 27: hEnd, 28: scratchBase (64B scores @ +0, 64B weights @ +64)
  const f = new FuncBuilder(new Array(29).fill(I32));
  const headsPerKv = f.local(I32);
  const numPages = f.local(I32);
  const headBytes = f.local(I32);
  const headQ4Bytes = f.local(I32);
  const invScale = f.local(F32);
  const h = f.local(I32);
  const kvHead = f.local(I32);
  const qH = f.local(I32);
  const oH = f.local(I32);
  const p = f.local(I32);
  const pageEnd = f.local(I32);
  const state = f.local(I32);
  const slot = f.local(I32);
  const t = f.local(I32);
  const baseOff = f.local(I32);
  const scOff = f.local(I32);
  const vScPtr = f.local(I32);
  const weightsPtr = f.local(I32);
  const hasPrev = f.local(I32);
  const runningMax = f.local(F32);
  const runningDenom = f.local(F32);
  const pageMax = f.local(F32);
  const newMax = f.local(F32);
  const rescale = f.local(F32);
  const pageSum = f.local(F32);
  const wVal = f.local(F32);
  const scTmp = f.local(F32);

  // Precompute layer invariants
  f.get(14).get(15).i32DivU().set(headsPerKv);
  f.get(17).i32Const(15).i32Add().i32Const(4).i32ShrS().set(numPages);
  f.get(16).i32Const(2).i32Shl().set(headBytes);
  f.get(24).get(15).i32DivU().set(headQ4Bytes);
  f.f32Const(1.0).get(16).f32ConvertI32S().f32Sqrt().f32Div().set(invScale);
  f.get(28).i32Const(64).i32Add().set(weightsPtr);

  // Helper to emit in-WASM online softmax update for `pageEnd` scores at `scratchBase` (param 28).
  const emitPageSoftmax = (useVScale: boolean) => {
    // 1. Find pageMax over scores[0..pageEnd-1]
    f.get(28).f32Load(0).set(pageMax);
    f.i32Const(1).set(t);
    f.block().loop();
    f.get(t).get(pageEnd).i32GeS().brIf(1);
    f.get(28).get(t).i32Const(2).i32Shl().i32Add().f32Load(0).set(scTmp);
    f.get(scTmp).get(pageMax).f32Gt().if_();
    f.get(scTmp).set(pageMax);
    f.end();
    f.get(t).i32Const(1).i32Add().set(t);
    f.br(0).end().end();

    // 2. newMax & rescale
    f.get(hasPrev).if_();
    f.get(pageMax).get(runningMax).f32Max().set(newMax);
    f.get(runningMax).get(newMax).f32Sub().call(fastExpIdx).set(rescale);
    f.else_();
    f.get(pageMax).set(newMax);
    f.f32Const(0.0).set(rescale);
    f.i32Const(1).set(hasPrev);
    f.end();

    // 3. weights[t] = exp(score[t] - newMax) (* vScale[t]), pageSum += exp(...)
    f.f32Const(0.0).set(pageSum);
    f.i32Const(0).set(t);
    f.block().loop();
    f.get(t).get(pageEnd).i32GeS().brIf(1);
    f.get(28).get(t).i32Const(2).i32Shl().i32Add().f32Load(0)
      .get(newMax).f32Sub().call(fastExpIdx).set(wVal);
    f.get(pageSum).get(wVal).f32Add().set(pageSum);
    if (useVScale) {
      f.get(weightsPtr).get(t).i32Const(2).i32Shl().i32Add()
        .get(wVal).get(vScPtr).f32Load(0).f32Mul().f32Store(0);
      f.get(vScPtr).get(25).i32Add().set(vScPtr);
    } else {
      f.get(weightsPtr).get(t).i32Const(2).i32Shl().i32Add()
        .get(wVal).f32Store(0);
    }
    f.get(t).i32Const(1).i32Add().set(t);
    f.br(0).end().end();

    // 4. runningDenom = runningDenom * rescale + pageSum; runningMax = newMax
    f.get(runningDenom).get(rescale).f32Mul().get(pageSum).f32Add().set(runningDenom);
    f.get(newMax).set(runningMax);
  };

  // Loop over Q heads [hStart, hEnd)
  f.get(26).set(h);
  f.get(0).get(26).get(headBytes).i32Mul().i32Add().set(qH);
  f.get(1).get(26).get(headBytes).i32Mul().i32Add().set(oH);

  f.block().loop();
  f.get(h).get(27).i32GeS().brIf(1);
  f.get(h).get(headsPerKv).i32DivU().set(kvHead);

  // Zero out[h] directly via bulk memory.fill
  f.get(oH).i32Const(0).get(headBytes).memoryFill();
  f.i32Const(0).set(hasPrev);
  f.f32Const(-1e38).set(runningMax);
  f.f32Const(0.0).set(runningDenom);

  // Loop over all active KV pages p = 0 .. numPages-1
  f.i32Const(0).set(p);
  f.block().loop();
  f.get(p).get(numPages).i32GeS().brIf(1);

  // pageEnd = min(16, seqLen - (p << 4))
  f.get(17).get(p).i32Const(4).i32Shl().i32Sub().tee(pageEnd);
  f.i32Const(16).i32GtS().if_();
  f.i32Const(16).set(pageEnd);
  f.end();

  f.get(2).get(p).i32Const(2).i32Shl().i32Add().i32Load(0).set(state);
  f.get(3).get(p).i32Const(2).i32Shl().i32Add().i32Load(0).set(slot);

  // --- Tier 0: HOT_RESIDENT (state == 1) ---
  f.get(state).i32Const(1).i32Eq().if_();
  f.get(slot).get(18).i32Mul().get(kvHead).get(headBytes).i32Mul().i32Add().set(baseOff);
  f.get(qH).get(4).get(baseOff).i32Add().get(28)
    .get(16).get(22).get(pageEnd).get(invScale).call(scoresF32Idx);
  emitPageSoftmax(false);
  f.get(oH).get(5).get(baseOff).i32Add().get(weightsPtr)
    .get(16).get(22).get(pageEnd).get(rescale).call(accumF32Idx);

  f.else_();
  // --- Tier 1: COLD_COMPRESSED Warm Q8 (state == 2) ---
  f.get(state).i32Const(2).i32Eq().if_();
  f.get(slot).get(19).i32Mul().get(kvHead).get(16).i32Mul().i32Add().set(baseOff);
  f.get(slot).get(21).i32Mul().get(kvHead).i32Const(2).i32Shl().i32Add().set(scOff);
  f.get(qH).get(6).get(baseOff).i32Add().get(8).get(scOff).i32Add()
    .get(28).get(16).get(23).get(25).get(pageEnd).get(invScale)
    .call(scoresQ8Idx);
  f.get(9).get(scOff).i32Add().set(vScPtr);
  emitPageSoftmax(true);
  f.get(oH).get(7).get(baseOff).i32Add().get(weightsPtr)
    .get(16).get(23).get(pageEnd).get(rescale).call(accumQ8Idx);

  f.else_();
  // --- Tier 2: COLD_Q4 (state == 3 && q4KLayerPtr != 0) ---
  f.get(state).i32Const(3).i32Eq().if_();
  f.get(10).i32Const(0).i32Ne().if_();
  f.get(slot).get(20).i32Mul().get(kvHead).get(headQ4Bytes).i32Mul().i32Add().set(baseOff);
  f.get(slot).get(21).i32Mul().get(kvHead).i32Const(2).i32Shl().i32Add().set(scOff);
  f.get(qH).get(10).get(baseOff).i32Add().get(12).get(scOff).i32Add()
    .get(28).get(16).get(24).get(25).get(pageEnd).get(invScale)
    .call(scoresQ4Idx);
  f.get(13).get(scOff).i32Add().set(vScPtr);
  emitPageSoftmax(true);
  f.get(oH).get(11).get(baseOff).i32Add().get(weightsPtr)
    .get(16).get(24).get(pageEnd).get(rescale).call(accumQ4Idx);
  f.end();
  f.end();
  f.end();
  f.end();

  f.get(p).i32Const(1).i32Add().set(p);
  f.br(0).end().end();

  // Final normalization: out[h] *= 1.0 / runningDenom
  f.get(runningDenom).f32Const(0.0).f32Gt().if_();
  f.get(oH).get(16).f32Const(1.0).get(runningDenom).f32Div().call(scaleIdx);
  f.end();

  f.get(qH).get(headBytes).i32Add().set(qH);
  f.get(oH).get(headBytes).i32Add().set(oH);
  f.get(h).i32Const(1).i32Add().set(h);
  f.br(0).end().end();

  return f;
}

/** Backwards-compatible whole-layer wrapper delegating to `flash_attn_layer_range(0, nHeads)`. */
function buildFlashAttnLayer(flashAttnRangeIdx: number): FuncBuilder {
  const f = new FuncBuilder(new Array(26).fill(I32));
  for (let i = 0; i < 26; i++) f.get(i);
  f.i32Const(0).get(14).i32Const(WASM_ATTN_SCORES_OFFSET).call(flashAttnRangeIdx);
  return f;
}

/**
 * 100% In-Verdant WASM SIMD128 `argmax_f32(logitsPtr, n) -> bestId` (`VT-217` / `verdant_row_ranking`):
 * Scans 4 f32 logits per iteration with `f32x4.gt` + `i32x4.bitmask`, breaking ties strictly by
 * lower lane index and guaranteeing NaNs never outrank finite or `-Infinity` logits.
 */
function buildArgmaxF32(): FuncBuilder {
  // params: 0=logitsPtr, 1=n -> result: i32 bestId
  const f = new FuncBuilder([I32, I32], [I32]);
  const bestId = f.local(I32);
  const bestVal = f.local(F32);
  const vBest = f.local(V128);
  const v = f.local(V128);
  const i = f.local(I32);
  const n4 = f.local(I32);
  const p = f.local(I32);
  const val = f.local(F32);

  // Guard n <= 0
  f.get(1).i32Const(0).i32LeS().if_();
  f.i32Const(0).ret();
  f.end();

  f.i32Const(0).set(bestId);
  f.f32Const(-Infinity).set(bestVal);
  f.get(bestVal).f32x4Splat().set(vBest);

  f.i32Const(0).set(i);
  f.get(1).i32Const(-4).i32And().set(n4);
  f.get(0).set(p);

  // 4-wide SIMD fast-skip loop (NaN lanes never satisfy f32x4.gt -> immune to NaN poisoning!)
  f.block().loop();
  f.get(i).get(n4).i32GeS().brIf(1);
  f.get(p).v128Load(0).tee(v).get(vBest).f32x4Gt().i32x4Bitmask().if_();
  // Lane 0..3 exact scalar tie-breaking (strictly > preserves first-index VT-217 parity)
  for (let lane = 0; lane < 4; lane++) {
    f.get(v).f32x4Extract(lane).tee(val).get(bestVal).f32Gt().if_();
    f.get(val).set(bestVal);
    if (lane === 0) {
      f.get(i).set(bestId);
    } else {
      f.get(i).i32Const(lane).i32Add().set(bestId);
    }
    f.end();
  }
  f.get(bestVal).f32x4Splat().set(vBest);
  f.end();
  f.get(i).i32Const(4).i32Add().set(i);
  f.get(p).i32Const(16).i32Add().set(p);
  f.br(0).end().end();

  // Scalar tail for remaining n % 4 elements
  f.block().loop();
  f.get(i).get(1).i32GeS().brIf(1);
  f.get(p).f32Load(0).tee(val).get(bestVal).f32Gt().if_();
  f.get(val).set(bestVal);
  f.get(i).set(bestId);
  f.end();
  f.get(i).i32Const(1).i32Add().set(i);
  f.get(p).i32Const(4).i32Add().set(p);
  f.br(0).end().end();

  // Edge case: if all non-NaN elements are -Infinity and logits[0] is NaN, select first non-NaN (-Inf)
  f.get(bestVal).f32Const(-Infinity).f32Eq().if_();
  f.get(0).f32Load(0).tee(val).get(val).f32Ne().if_();
  f.i32Const(1).set(i);
  f.block().loop();
  f.get(i).get(1).i32GeS().brIf(1);
  f.get(0).get(i).i32Const(2).i32Shl().i32Add().f32Load(0).tee(val).get(val).f32Eq().if_();
  f.get(i).set(bestId);
  f.br(2);
  f.end();
  f.get(i).i32Const(1).i32Add().set(i);
  f.br(0).end().end();
  f.end();
  f.end();

  f.get(bestId);
  return f;
}

/**
 * 100% In-Verdant WASM SIMD128 `topk_f32(logitsPtr, n, k, outIdsPtr, outValsPtr) -> effK`:
 * Selects exact `VT-217` / `verdant_row_ranking` top-K `(id, logit)` pairs in a single SIMD pass
 * with zero heap allocations, sorting NaNs strictly last and preserving lower lane indices on ties.
 */
function buildTopkF32(): FuncBuilder {
  // params: 0=logitsPtr, 1=n, 2=k, 3=outIdsPtr, 4=outValsPtr -> result: i32 effK
  const f = new FuncBuilder([I32, I32, I32, I32, I32], [I32]);
  const effK = f.local(I32);
  const minSlot = f.local(I32);
  const minId = f.local(I32);
  const minVal = f.local(F32);
  const vMin = f.local(V128);
  const v = f.local(V128);
  const i = f.local(I32);
  const j = f.local(I32);
  const p = f.local(I32);
  const nLimit4 = f.local(I32);
  const val = f.local(F32);
  const tmp = f.local(F32);
  const tmpId = f.local(I32);
  const slotOff = f.local(I32);

  // effK = min(k, n)
  f.get(2).get(1).i32GtS().if_();
  f.get(1).set(effK);
  f.else_();
  f.get(2).set(effK);
  f.end();

  f.get(effK).i32Const(0).i32LeS().if_();
  f.i32Const(0).ret();
  f.end();

  // Helper: update (minVal, minId, minSlot) when slot `slotIdx` with `(candVal, candId)` is strictly
  // worse under VT-217 total order: `candVal < minVal` OR `(candVal == minVal && candId > minId)`.
  const emitCheckMinSlot = (slotIdxLocal: number, candValLocal: number, candIdLocal: number) => {
    f.get(minVal).get(candValLocal).f32Gt().if_();
    f.get(candValLocal).set(minVal);
    f.get(candIdLocal).set(minId);
    f.get(slotIdxLocal).set(minSlot);
    f.else_();
    f.get(candValLocal).get(minVal).f32Eq().if_();
    f.get(candIdLocal).get(minId).i32GtS().if_();
    f.get(candIdLocal).set(minId);
    f.get(slotIdxLocal).set(minSlot);
    f.end();
    f.end();
    f.end();
  };

  // Phase 1: seed slots [0..effK-1] with logits[0..effK-1] (normalizing NaNs -> -Infinity so finite logits always evict them)
  f.get(3).i32Const(0).i32Store(0);
  f.get(0).f32Load(0).set(val);
  f.get(val).get(val).f32Ne().if_();
  f.f32Const(-Infinity).set(val);
  f.end();
  f.get(val).set(minVal);
  f.get(4).get(minVal).f32Store(0);
  f.i32Const(0).set(minSlot);
  f.i32Const(0).set(minId);

  f.i32Const(1).set(j);
  f.block().loop();
  f.get(j).get(effK).i32GeS().brIf(1);
  f.get(j).i32Const(2).i32Shl().set(slotOff);
  f.get(0).get(slotOff).i32Add().f32Load(0).set(val);
  f.get(val).get(val).f32Ne().if_();
  f.f32Const(-Infinity).set(val);
  f.end();
  f.get(3).get(slotOff).i32Add().get(j).i32Store(0);
  f.get(4).get(slotOff).i32Add().get(val).f32Store(0);
  emitCheckMinSlot(j, val, j);
  f.get(j).i32Const(1).i32Add().set(j);
  f.br(0).end().end();

  // Helper to replace worst slot (lowest value, highest ID on tie) when `val > minVal` (NaNs never pass f32.gt!)
  const emitInsertIfGreater = (laneOffset: number) => {
    f.get(val).get(minVal).f32Gt().if_();
    f.get(minSlot).i32Const(2).i32Shl().set(slotOff);
    if (laneOffset === 0) {
      f.get(3).get(slotOff).i32Add().get(i).i32Store(0);
    } else {
      f.get(3).get(slotOff).i32Add().get(i).i32Const(laneOffset).i32Add().i32Store(0);
    }
    f.get(4).get(slotOff).i32Add().get(val).f32Store(0);

    // Recompute worst slot (minVal, maxId on tie) over [0..effK-1]
    f.get(4).f32Load(0).set(minVal);
    f.get(3).i32Load(0).set(minId);
    f.i32Const(0).set(minSlot);
    f.i32Const(1).set(j);
    f.block().loop();
    f.get(j).get(effK).i32GeS().brIf(1);
    f.get(j).i32Const(2).i32Shl().set(slotOff);
    f.get(4).get(slotOff).i32Add().f32Load(0).set(tmp);
    f.get(3).get(slotOff).i32Add().i32Load(0).set(tmpId);
    emitCheckMinSlot(j, tmp, tmpId);
    f.get(j).i32Const(1).i32Add().set(j);
    f.br(0).end().end();
    f.end();
  };

  // Phase 2: 4-wide SIMD fast-skip loop for i = effK .. n - 4
  f.get(minVal).f32x4Splat().set(vMin);
  f.get(effK).set(i);
  f.get(0).get(effK).i32Const(2).i32Shl().i32Add().set(p);
  f.get(1).i32Const(3).i32Sub().set(nLimit4);

  f.block().loop();
  f.get(i).get(nLimit4).i32GeS().brIf(1);
  f.get(p).v128Load(0).tee(v).get(vMin).f32x4Gt().i32x4Bitmask().if_();
  for (let lane = 0; lane < 4; lane++) {
    f.get(v).f32x4Extract(lane).set(val);
    emitInsertIfGreater(lane);
  }
  f.get(minVal).f32x4Splat().set(vMin);
  f.end();
  f.get(i).i32Const(4).i32Add().set(i);
  f.get(p).i32Const(16).i32Add().set(p);
  f.br(0).end().end();

  // Phase 3: scalar tail for remaining elements
  f.block().loop();
  f.get(i).get(1).i32GeS().brIf(1);
  f.get(p).f32Load(0).set(val);
  emitInsertIfGreater(0);
  f.get(i).i32Const(1).i32Add().set(i);
  f.get(p).i32Const(4).i32Add().set(p);
  f.br(0).end().end();

  f.get(effK);
  return f;
}

/**
 * Verdant/Prism `Q8_K` super-block activation quantizer: `(xPtr, q8kPtr, cols)`.
 * Packs 256 floats into 320 bytes (`f32 d`, `i16[16] bsums16`, `i16[8] bsums32`, `i8[256] qs`).
 * Enables `uor-matmul` narrow-lane accumulation ("encode to float ONCE per 256-elem super-block")
 * for both `Q4_K` and `Q6_K`.
 */
function buildQuantizeF32ToQ8K(): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32]);
  const nSuper = f.local(I32);
  const sb = f.local(I32);
  const xp = f.local(I32);
  const qp = f.local(I32);
  const j = f.local(I32);
  const amax = f.local(F32);
  const d = f.local(F32);
  const invD = f.local(F32);
  const vmax = f.local(V128);
  const vinv = f.local(V128);
  const q0 = f.local(V128);
  const q1 = f.local(V128);
  const q2 = f.local(V128);
  const q3 = f.local(V128);
  const sumV = f.local(V128);
  const bsumsLo = f.local(V128);
  const bsumsHi = f.local(V128);
  const zero = f.local(V128);

  f.i32Const(0).i32x4Splat().set(zero);
  f.get(2).i32Const(8).i32ShrS().set(nSuper);
  f.get(0).set(xp);
  f.get(1).set(qp);
  f.i32Const(0).set(sb);

  f.block().loop();
  f.get(sb).get(nSuper).i32GeS().brIf(1);

  // Pass 1: find amax across all 256 floats (64 x f32x4 vectors = 16 iterations of 16 floats)
  f.f32Const(0).f32x4Splat().set(vmax);
  f.i32Const(0).set(j);
  f.block().loop();
  f.get(j).i32Const(1024).i32GeS().brIf(1);
  f.get(vmax)
    .get(xp).get(j).i32Add().v128Load(0).f32x4Abs().f32x4Pmax()
    .get(xp).get(j).i32Add().v128Load(16).f32x4Abs().f32x4Pmax()
    .get(xp).get(j).i32Add().v128Load(32).f32x4Abs().f32x4Pmax()
    .get(xp).get(j).i32Add().v128Load(48).f32x4Abs().f32x4Pmax()
    .set(vmax);
  f.get(j).i32Const(64).i32Add().set(j);
  f.br(0).end().end();

  f.get(vmax).f32x4Extract(0)
    .get(vmax).f32x4Extract(1).f32Max()
    .get(vmax).f32x4Extract(2).f32Max()
    .get(vmax).f32x4Extract(3).f32Max()
    .set(amax);

  f.get(amax).f32Const(0).f32Gt().if_();
  f.get(amax).f32Const(127).f32Div().set(d);
  f.f32Const(127).get(amax).f32Div().set(invD);
  f.else_();
  f.f32Const(0).set(d);
  f.f32Const(0).set(invD);
  f.end();

  f.get(qp).get(d).f32Store(0);
  f.get(invD).f32x4Splat().set(vinv);
  f.get(zero).tee(bsumsLo).set(bsumsHi);

  // Pass 2: quantize all 16 slices of 16 floats -> qs[64..319] + bsums16[0..15]
  for (let s = 0; s < 16; s++) {
    const xOff = s * 64;
    f.get(xp).v128Load(xOff).get(vinv).f32x4Mul().f32x4Nearest().i32x4TruncSatF32x4S().set(q0);
    f.get(xp).v128Load(xOff + 16).get(vinv).f32x4Mul().f32x4Nearest().i32x4TruncSatF32x4S().set(q1);
    f.get(xp).v128Load(xOff + 32).get(vinv).f32x4Mul().f32x4Nearest().i32x4TruncSatF32x4S().set(q2);
    f.get(xp).v128Load(xOff + 48).get(vinv).f32x4Mul().f32x4Nearest().i32x4TruncSatF32x4S().set(q3);

    f.get(q0).get(q1).i32x4Add().get(q2).i32x4Add().get(q3).i32x4Add().set(sumV);
    const targetBsum = s < 8 ? bsumsLo : bsumsHi;
    f.get(targetBsum).hsumI32x4(sumV).i16x8ReplaceLane(s & 7).set(targetBsum);

    f.get(qp)
      .get(q0).get(q1).i16x8NarrowI32x4S()
      .get(q2).get(q3).i16x8NarrowI32x4S()
      .i8x16NarrowI16x8S()
      .v128Store(64 + s * 16, 4);
  }

  // Store bsums16[0..15] at +16..+47 and bsums32[0..7] (pairwise sums) at +48..+63
  f.get(qp).get(bsumsLo).v128Store(16, 4);
  f.get(qp).get(bsumsHi).v128Store(32, 4);
  f.get(qp)
    .get(bsumsLo).i32x4ExtaddPairwiseI16x8S()
    .get(bsumsHi).i32x4ExtaddPairwiseI16x8S()
    .i16x8NarrowI32x4S()
    .v128Store(48, 4);

  f.get(xp).i32Const(1024).i32Add().set(xp);
  f.get(qp).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(sb).i32Const(1).i32Add().set(sb);
  f.br(0).end().end();

  return f;
}

/**
 * Verdant/Prism 4-row register-blocked `Q4_K × Q8_K` Narrow-Lane Integer Contraction:
 * `(wPtr, q8kPtr, outPtr, cols, rows)`.
 * - SWAR 3-load `u32` scale/min unpacking (0 scale loads inside chunk loop)
 * - 1x `i32x4.dot_i16x8_s` computes all 8 sub-block min terms per row at once
 * - Pure `i32x4` narrow-lane super-block accumulation (70x i32 headroom, 0 scalar hsum inside sub-blocks)
 * - Vectorized 4-row `hsum4RowsI32x4` epilogue encodes to float ONCE per 256 weights!
 */
function buildMulMatQ4KQ8K(useRelaxed: boolean): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const nSuper = f.local(I32);
  const rowBytes = f.local(I32);
  const wp0 = f.local(I32);
  const wp1 = f.local(I32);
  const wp2 = f.local(I32);
  const wp3 = f.local(I32);
  const qp = f.local(I32);
  const sb = f.local(I32);
  const u0 = f.local(I32);
  const u1 = f.local(I32);
  const u2 = f.local(I32);
  // Pack 8x u8 scales per row into ONE 64-bit GPR (4 I64 locals instead of 8 I32 -> frees 4 x86-64 GPRs!)
  const sc_0 = f.local(I64);
  const sc_1 = f.local(I64);
  const sc_2 = f.local(I64);
  const sc_3 = f.local(I64);
  const mn03 = f.local(I32);
  const mn47 = f.local(I32);
  const accS = f.local(F32);
  const dxS = f.local(F32);

  const mask0f = f.local(V128);
  const acc4 = f.local(V128);
  const xlo0 = f.local(V128);
  const xhi0 = f.local(V128);
  const xlo1 = f.local(V128);
  const xhi1 = f.local(V128);
  const sumi0 = f.local(V128);
  const sumi1 = f.local(V128);
  const sumi2 = f.local(V128);
  const sumi3 = f.local(V128);
  const raw0 = f.local(V128);
  const raw1 = f.local(V128);
  const nib = f.local(V128);
  const dotV = f.local(V128);

  f.i32Const(0x0f).i8x16Splat().set(mask0f);
  f.get(3).i32Const(8).i32ShrS().set(nSuper);
  f.get(nSuper).i32Const(144).i32Mul().set(rowBytes);
  f.i32Const(0).set(r);

  // SWAR unpack scales into packed i64 `sc64Loc` and compute minDotV = dot(mins[0..7], bsums32V) into dstMinLoc
  const unpackScalesAndMinDot = (
    wpLoc: number,
    sc64Loc: number,
    bsums32Loc: number,
    dstMinLoc: number,
  ) => {
    f.get(wpLoc).i32Load(4).set(u0);
    f.get(wpLoc).i32Load(8).set(u1);
    f.get(wpLoc).i32Load(12).set(u2);
    // sc03 = u0 & 0x3f3f3f3f; sc47 = (u2 & 0x0f0f0f0f) | ((u0 >>> 2) & 0x30303030)
    // Pack into single i64 GPR: sc03 | (sc47 << 32)
    f.get(u0).i32Const(0x3f3f3f3f).i32And().i64ExtendI32U()
      .get(u2).i32Const(0x0f0f0f0f).i32And()
      .get(u0).i32Const(2).i32ShrU().i32Const(0x30303030).i32And()
      .i32Or().i64ExtendI32U().i64Const(32).i64Shl()
      .i64Or().set(sc64Loc);
    // mn03 = u1 & 0x3f3f3f3f; mn47 = ((u2 >>> 4) & 0x0f0f0f0f) | ((u1 >>> 2) & 0x30303030)
    f.get(u1).i32Const(0x3f3f3f3f).i32And().set(mn03);
    f.get(u2).i32Const(4).i32ShrU().i32Const(0x0f0f0f0f).i32And()
      .get(u1).i32Const(2).i32ShrU().i32Const(0x30303030).i32And()
      .i32Or().set(mn47);
    // Pack mn03 + mn47 -> i16x8 -> dot with bsums32V
    f.i32Const(0).i32x4Splat().get(mn03).i32x4ReplaceLane(0).get(mn47).i32x4ReplaceLane(1)
      .i16x8ExtendLowI8x16U()
      .get(bsums32Loc).i32x4DotI16x8S().set(dstMinLoc);
  };

  // Extract byte `k` (0..7) from packed i64 scale local and splat into i16x8 (for 1-op pmaddwd!)
  const emitScaleSplat16 = (sc64Loc: number, byteIdx: number) => {
    f.get(sc64Loc);
    if (byteIdx > 0) {
      f.i64Const(byteIdx * 8).i64ShrU();
    }
    f.i32WrapI64();
    if (byteIdx < 7) {
      f.i32Const(0xff).i32And();
    }
    f.i16x8Splat();
  };

  const emitChunkRow = (
    wpLoc: number,
    sc64Loc: number,
    sumiLoc: number,
    c: number,
  ) => {
    const qsBase = 16 + c * 32;
    const scLoIdx = c * 2;
    const scHiIdx = scLoIdx + 1;

    f.get(wpLoc).v128Load(qsBase).set(raw0);
    f.get(wpLoc).v128Load(qsBase + 16).set(raw1);

    // Sub-block 2c (low nibbles): 2x vpmaddubsw -> i16x8.add (max 7620 << 32767) -> 1x vpmaddwd(sc)!
    f.get(raw0).get(mask0f).v128And().set(nib);
    f.pushDotI8I7ToI16x8(xlo0, nib, useRelaxed);
    f.get(raw1).get(mask0f).v128And().set(nib);
    f.pushDotI8I7ToI16x8(xhi0, nib, useRelaxed);
    f.i16x8Add();
    emitScaleSplat16(sc64Loc, scLoIdx);
    f.i32x4DotI16x8S().get(sumiLoc).i32x4Add().set(sumiLoc);

    // Sub-block 2c+1 (high nibbles): 2x vpmaddubsw -> i16x8.add -> 1x vpmaddwd(sc)!
    f.get(raw0).i32Const(4).i8x16ShrU().set(nib);
    f.pushDotI8I7ToI16x8(xlo1, nib, useRelaxed);
    f.get(raw1).i32Const(4).i8x16ShrU().set(nib);
    f.pushDotI8I7ToI16x8(xhi1, nib, useRelaxed);
    f.i16x8Add();
    emitScaleSplat16(sc64Loc, scHiIdx);
    f.i32x4DotI16x8S().get(sumiLoc).i32x4Add().set(sumiLoc);
  };

  // 4-row register-blocked loop (<=14 live V128 registers -> zero XMM spills!)
  f.block().loop();
  f.get(r).i32Const(3).i32Add().get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add()
    .tee(wp0).get(rowBytes).i32Add()
    .tee(wp1).get(rowBytes).i32Add()
    .tee(wp2).get(rowBytes).i32Add().set(wp3);
  f.get(1).set(qp);
  f.f32Const(0).f32x4Splat().set(acc4);
  f.i32Const(0).set(sb);

  f.block().loop();
  f.get(sb).get(nSuper).i32GeS().brIf(1);

  f.get(qp).f32Load(0).set(dxS);
  // Load bsums32[0..7] into xlo0 temporarily and fold all 4 rows' min terms into acc4 up front
  f.get(qp).v128Load(48, 4).set(xlo0);
  unpackScalesAndMinDot(wp0, sc_0, xlo0, sumi0);
  unpackScalesAndMinDot(wp1, sc_1, xlo0, sumi1);
  unpackScalesAndMinDot(wp2, sc_2, xlo0, sumi2);
  unpackScalesAndMinDot(wp3, sc_3, xlo0, sumi3);
  f.hsum4RowsI32x4(sumi0, sumi1, sumi2, sumi3, raw0, raw1, dotV);
  // acc4 -= f32x4(minSums4) * dmin4 * dxS
  f.f32Const(0).f32x4Splat()
    .loadF16AsF32(wp0, 2).f32x4ReplaceLane(0)
    .loadF16AsF32(wp1, 2).f32x4ReplaceLane(1)
    .loadF16AsF32(wp2, 2).f32x4ReplaceLane(2)
    .loadF16AsF32(wp3, 2).f32x4ReplaceLane(3)
    .get(dxS).f32x4Splat().f32x4Mul().set(raw0);
  f.get(acc4)
    .get(dotV).f32x4ConvertI32x4S().get(raw0).f32x4Mul()
    .f32x4Sub().set(acc4);

  // Zero narrow-lane i32x4 accumulators for the 8 sub-blocks
  f.i32Const(0).i32x4Splat().tee(sumi0).tee(sumi1).tee(sumi2).set(sumi3);

  for (let c = 0; c < 4; c++) {
    const xBase = 64 + c * 64;
    f.get(qp).v128Load(xBase, 4).set(xlo0);
    f.get(qp).v128Load(xBase + 16, 4).set(xhi0);
    f.get(qp).v128Load(xBase + 32, 4).set(xlo1);
    f.get(qp).v128Load(xBase + 48, 4).set(xhi1);

    emitChunkRow(wp0, sc_0, sumi0, c);
    emitChunkRow(wp1, sc_1, sumi1, c);
    emitChunkRow(wp2, sc_2, sumi2, c);
    emitChunkRow(wp3, sc_3, sumi3, c);
  }

  // Encode to float ONCE per 256-element super-block for all 4 rows simultaneously!
  f.hsum4RowsI32x4(sumi0, sumi1, sumi2, sumi3, raw0, raw1, dotV);
  f.f32Const(0).f32x4Splat()
    .loadF16AsF32(wp0, 0).f32x4ReplaceLane(0)
    .loadF16AsF32(wp1, 0).f32x4ReplaceLane(1)
    .loadF16AsF32(wp2, 0).f32x4ReplaceLane(2)
    .loadF16AsF32(wp3, 0).f32x4ReplaceLane(3)
    .get(dxS).f32x4Splat().f32x4Mul().set(raw0);
  if (useRelaxed) {
    f.get(dotV).f32x4ConvertI32x4S().get(raw0).get(acc4).f32x4RelaxedMadd().set(acc4);
  } else {
    f.get(dotV).f32x4ConvertI32x4S().get(raw0).f32x4Mul().get(acc4).f32x4Add().set(acc4);
  }

  f.get(wp0).i32Const(144).i32Add().set(wp0);
  f.get(wp1).i32Const(144).i32Add().set(wp1);
  f.get(wp2).i32Const(144).i32Add().set(wp2);
  f.get(wp3).i32Const(144).i32Add().set(wp3);
  f.get(qp).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(sb).i32Const(1).i32Add().set(sb);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc4).v128Store(0);
  f.get(r).i32Const(4).i32Add().set(r);
  f.br(0).end().end();

  // Remainder rows (0..3)
  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(wp0);
  f.get(1).set(qp);
  f.f32Const(0).set(accS);
  f.i32Const(0).set(sb);

  f.block().loop();
  f.get(sb).get(nSuper).i32GeS().brIf(1);
  f.get(qp).f32Load(0).set(dxS);
  f.get(qp).v128Load(48, 4).set(xlo0);
  unpackScalesAndMinDot(wp0, sc_0, xlo0, sumi1);
  f.i32Const(0).i32x4Splat().set(sumi0);
  for (let c = 0; c < 4; c++) {
    const xBase = 64 + c * 64;
    f.get(qp).v128Load(xBase, 4).set(xlo0);
    f.get(qp).v128Load(xBase + 16, 4).set(xhi0);
    f.get(qp).v128Load(xBase + 32, 4).set(xlo1);
    f.get(qp).v128Load(xBase + 48, 4).set(xhi1);
    emitChunkRow(wp0, sc_0, sumi0, c);
  }
  f.loadF16AsF32(wp0, 0).hsumI32x4(sumi0).f32ConvertI32S().f32Mul()
    .loadF16AsF32(wp0, 2).hsumI32x4(sumi1).f32ConvertI32S().f32Mul()
    .f32Sub().get(dxS).f32Mul().get(accS).f32Add().set(accS);

  f.get(wp0).i32Const(144).i32Add().set(wp0);
  f.get(qp).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(sb).i32Const(1).i32Add().set(sb);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(accS).f32Store(0);
  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();

  return f;
}

/**
 * Verdant/Prism 4-token batched prefill `Q4_K × Q8_K` Narrow-Lane Contraction:
 * `(wPtr, q8kBatchPtr, outBatchPtr, cols, rows, oStrideBytes)`.
 * Unpacks each `Q4_K` weight super-block ONCE across all 4 prompt tokens, accumulates in pure
 * `i32x4`, and encodes to float ONCE per 256 weights via `hsum4RowsI32x4`.
 */
function buildMulMatQ4KQ8KBatch4(useRelaxed: boolean): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const nSuper = f.local(I32);
  const rowBytes = f.local(I32);
  const qStride = f.local(I32);
  const oStride = f.local(I32);
  const wp = f.local(I32);
  const qp0 = f.local(I32);
  const qp1 = f.local(I32);
  const qp2 = f.local(I32);
  const qp3 = f.local(I32);
  const op = f.local(I32);
  const sb = f.local(I32);
  const u0 = f.local(I32);
  const u1 = f.local(I32);
  const u2 = f.local(I32);
  const sc03 = f.local(I32);
  const sc47 = f.local(I32);
  const mn03 = f.local(I32);
  const mn47 = f.local(I32);

  const mask0f = f.local(V128);
  const acc4 = f.local(V128);
  const dx4 = f.local(V128);
  const sumi0 = f.local(V128);
  const sumi1 = f.local(V128);
  const sumi2 = f.local(V128);
  const sumi3 = f.local(V128);
  const raw0 = f.local(V128);
  const raw1 = f.local(V128);
  const wlo0 = f.local(V128);
  const wlo1 = f.local(V128);
  const scV = f.local(V128);
  const xtmp = f.local(V128);
  const dotV = f.local(V128);

  f.i32Const(0x0f).i8x16Splat().set(mask0f);
  f.get(3).i32Const(8).i32ShrS().set(nSuper);
  f.get(nSuper).i32Const(144).i32Mul().set(rowBytes);
  f.get(nSuper).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Mul().set(qStride);
  f.get(5).set(oStride);
  f.i32Const(0).set(r);

  // Broadcast i16 lane `k` (0..7) from XMM `dotV` (holding sc[0..7]) in 1 single VPSHUFB (zero GPR->XMM stalls!)
  const emitScaleShuffle16 = (laneIdx: number) => {
    const lo = laneIdx * 2;
    const hi = lo + 1;
    f.get(dotV).get(dotV).i8x16Shuffle([lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi]).set(scV);
  };

  const emitTokSub = (qpLoc: number, xOff: number, sumiLoc: number) => {
    f.get(qpLoc).v128Load(xOff, 4).set(xtmp);
    f.pushDotI8I7ToI16x8(xtmp, wlo0, useRelaxed);
    f.get(qpLoc).v128Load(xOff + 16, 4).set(xtmp);
    f.pushDotI8I7ToI16x8(xtmp, wlo1, useRelaxed);
    f.i16x8Add().get(scV).i32x4DotI16x8S().get(sumiLoc).i32x4Add().set(sumiLoc);
  };

  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(wp);
  f.get(1).tee(qp0)
    .get(qStride).i32Add().tee(qp1)
    .get(qStride).i32Add().tee(qp2)
    .get(qStride).i32Add().set(qp3);
  f.f32Const(0).f32x4Splat().set(acc4);
  f.i32Const(0).set(sb);

  f.block().loop();
  f.get(sb).get(nSuper).i32GeS().brIf(1);

  f.get(wp).i32Load(4).set(u0);
  f.get(wp).i32Load(8).set(u1);
  f.get(wp).i32Load(12).set(u2);
  f.get(u0).i32Const(0x3f3f3f3f).i32And().set(sc03);
  f.get(u1).i32Const(0x3f3f3f3f).i32And().set(mn03);
  f.get(u2).i32Const(0x0f0f0f0f).i32And()
    .get(u0).i32Const(2).i32ShrU().i32Const(0x30303030).i32And()
    .i32Or().set(sc47);
  f.get(u2).i32Const(4).i32ShrU().i32Const(0x0f0f0f0f).i32And()
    .get(u1).i32Const(2).i32ShrU().i32Const(0x30303030).i32And()
    .i32Or().set(mn47);

  f.f32Const(0).f32x4Splat()
    .get(qp0).f32Load(0).f32x4ReplaceLane(0)
    .get(qp1).f32Load(0).f32x4ReplaceLane(1)
    .get(qp2).f32Load(0).f32x4ReplaceLane(2)
    .get(qp3).f32Load(0).f32x4ReplaceLane(3)
    .set(dx4);

  f.i32Const(0).i32x4Splat().get(mn03).i32x4ReplaceLane(0).get(mn47).i32x4ReplaceLane(1)
    .i16x8ExtendLowI8x16U().set(scV);
  f.get(scV).get(qp0).v128Load(48, 4).i32x4DotI16x8S().set(sumi0);
  f.get(scV).get(qp1).v128Load(48, 4).i32x4DotI16x8S().set(sumi1);
  f.get(scV).get(qp2).v128Load(48, 4).i32x4DotI16x8S().set(sumi2);
  f.get(scV).get(qp3).v128Load(48, 4).i32x4DotI16x8S().set(sumi3);
  f.hsum4RowsI32x4(sumi0, sumi1, sumi2, sumi3, raw0, raw1, dotV);
  f.get(acc4)
    .get(dotV).f32x4ConvertI32x4S()
    .loadF16AsF32(wp, 2).f32x4Splat().get(dx4).f32x4Mul().f32x4Mul()
    .f32x4Sub().set(acc4);

  // Pack all 8 scales into XMM `dotV` (dead during chunk loop) as i16x8 -> frees sc03/sc47 GPRs!
  f.i32Const(0).i32x4Splat().get(sc03).i32x4ReplaceLane(0).get(sc47).i32x4ReplaceLane(1)
    .i16x8ExtendLowI8x16U().set(dotV);

  f.i32Const(0).i32x4Splat().tee(sumi0).tee(sumi1).tee(sumi2).set(sumi3);

  for (let c = 0; c < 4; c++) {
    const qsBase = 16 + c * 32;
    const xBase = 64 + c * 64;

    f.get(wp).v128Load(qsBase).set(raw0);
    f.get(wp).v128Load(qsBase + 16).set(raw1);

    f.get(raw0).get(mask0f).v128And().set(wlo0);
    f.get(raw1).get(mask0f).v128And().set(wlo1);
    emitScaleShuffle16(c * 2);
    emitTokSub(qp0, xBase, sumi0);
    emitTokSub(qp1, xBase, sumi1);
    emitTokSub(qp2, xBase, sumi2);
    emitTokSub(qp3, xBase, sumi3);

    f.get(raw0).i32Const(4).i8x16ShrU().set(wlo0);
    f.get(raw1).i32Const(4).i8x16ShrU().set(wlo1);
    emitScaleShuffle16(c * 2 + 1);
    emitTokSub(qp0, xBase + 32, sumi0);
    emitTokSub(qp1, xBase + 32, sumi1);
    emitTokSub(qp2, xBase + 32, sumi2);
    emitTokSub(qp3, xBase + 32, sumi3);
  }

  f.hsum4RowsI32x4(sumi0, sumi1, sumi2, sumi3, raw0, raw1, dotV);
  f.loadF16AsF32(wp, 0).f32x4Splat().get(dx4).f32x4Mul().set(raw0);
  if (useRelaxed) {
    f.get(dotV).f32x4ConvertI32x4S().get(raw0).get(acc4).f32x4RelaxedMadd().set(acc4);
  } else {
    f.get(dotV).f32x4ConvertI32x4S().get(raw0).f32x4Mul().get(acc4).f32x4Add().set(acc4);
  }

  f.get(wp).i32Const(144).i32Add().set(wp);
  f.get(qp0).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp0);
  f.get(qp1).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp1);
  f.get(qp2).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp2);
  f.get(qp3).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp3);
  f.get(sb).i32Const(1).i32Add().set(sb);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().tee(op).get(acc4).f32x4Extract(0).f32Store(0);
  f.get(op).get(oStride).i32Add().tee(op).get(acc4).f32x4Extract(1).f32Store(0);
  f.get(op).get(oStride).i32Add().tee(op).get(acc4).f32x4Extract(2).f32Store(0);
  f.get(op).get(oStride).i32Add().get(acc4).f32x4Extract(3).f32Store(0);

  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();

  return f;
}

/**
 * Verdant Tensor `v2.22.4` 2-row register-blocked `Q5_K × Q8_K` Narrow-Lane Integer Contraction:
 * `(wPtr, q8kPtr, outPtr, cols, rows)`.
 * - SWAR 3-load `u32` scale/min unpacking + 1-op `i32x4.dot_i16x8_s` min term per row
 * - Single-shift/zero-shift `mask10 = 0x10` 5th-bit extraction from `qh[32]`
 * - Pure `i32x4` narrow-lane super-block accumulation (33.82x i32 headroom, 2.08x i16 headroom)
 * - Strictly 13 live V128 registers (zero x86-64 XMM spills in V8 TurboFan!)
 */
function buildMulMatQ5KQ8K(useRelaxed: boolean): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const nSuper = f.local(I32);
  const rowBytes = f.local(I32);
  const wp0 = f.local(I32);
  const wp1 = f.local(I32);
  const qp = f.local(I32);
  const sb = f.local(I32);
  const u0 = f.local(I32);
  const u1 = f.local(I32);
  const u2 = f.local(I32);
  const sc03_0 = f.local(I32);
  const sc47_0 = f.local(I32);
  const sc03_1 = f.local(I32);
  const sc47_1 = f.local(I32);
  const mn03 = f.local(I32);
  const mn47 = f.local(I32);
  const acc0 = f.local(F32);
  const acc1 = f.local(F32);
  const dxS = f.local(F32);

  // Strictly 13 V128 locals -> 3 free XMM scratch registers on x86-64 (0 spills!)
  const mask0f = f.local(V128);
  const mask10 = f.local(V128);
  const sumi0 = f.local(V128);
  const sumi1 = f.local(V128);
  const xlo0 = f.local(V128);
  const xhi0 = f.local(V128);
  const xlo1 = f.local(V128);
  const xhi1 = f.local(V128);
  const qh0 = f.local(V128);
  const qh1 = f.local(V128);
  const raw0 = f.local(V128);
  const raw1 = f.local(V128);
  const q5v = f.local(V128);

  f.i32Const(0x0f).i8x16Splat().set(mask0f);
  f.i32Const(0x10).i8x16Splat().set(mask10);
  f.get(3).i32Const(8).i32ShrS().set(nSuper);
  f.get(nSuper).i32Const(176).i32Mul().set(rowBytes);
  f.i32Const(0).set(r);

  const unpackScalesAndMinDot = (
    wpLoc: number,
    sc03Loc: number,
    sc47Loc: number,
    bsums32Loc: number,
    dstMinLoc: number,
  ) => {
    f.get(wpLoc).i32Load(4).set(u0);
    f.get(wpLoc).i32Load(8).set(u1);
    f.get(wpLoc).i32Load(12).set(u2);
    f.get(u0).i32Const(0x3f3f3f3f).i32And().set(sc03Loc);
    f.get(u1).i32Const(0x3f3f3f3f).i32And().set(mn03);
    f.get(u2).i32Const(0x0f0f0f0f).i32And()
      .get(u0).i32Const(2).i32ShrU().i32Const(0x30303030).i32And()
      .i32Or().set(sc47Loc);
    f.get(u2).i32Const(4).i32ShrU().i32Const(0x0f0f0f0f).i32And()
      .get(u1).i32Const(2).i32ShrU().i32Const(0x30303030).i32And()
      .i32Or().set(mn47);
    f.i32Const(0).i32x4Splat().get(mn03).i32x4ReplaceLane(0).get(mn47).i32x4ReplaceLane(1)
      .i16x8ExtendLowI8x16U()
      .get(bsums32Loc).i32x4DotI16x8S().set(dstMinLoc);
  };

  const emitScaleSplat16 = (scWordLoc: number, byteIdx: number) => {
    f.get(scWordLoc);
    if (byteIdx > 0) f.i32Const(byteIdx * 8).i32ShrU();
    if (byteIdx < 3) f.i32Const(0xff).i32And();
    f.i16x8Splat();
  };

  // Shift bit `bitIdx` (0..7) of `qhLoc` into bit 4 (0x10): 0 shifts for bit 4, 1 shift otherwise!
  const emitQhBit4 = (qhLoc: number, bitIdx: number) => {
    f.get(qhLoc);
    if (bitIdx < 4) {
      f.i32Const(4 - bitIdx).i8x16Shl();
    } else if (bitIdx > 4) {
      f.i32Const(bitIdx - 4).i8x16ShrU();
    }
    f.get(mask10).v128And();
  };

  const emitChunkRow = (
    wpLoc: number,
    sc03Loc: number,
    sc47Loc: number,
    sumiLoc: number,
    c: number,
  ) => {
    const qsBase = 48 + c * 32;
    const scWord = c < 2 ? sc03Loc : sc47Loc;
    const scLoIdx = (c & 1) * 2;
    const scHiIdx = scLoIdx + 1;
    const bitLo = c * 2;
    const bitHi = c * 2 + 1;

    f.get(wpLoc).v128Load(16).set(qh0);
    f.get(wpLoc).v128Load(32).set(qh1);
    f.get(wpLoc).v128Load(qsBase).set(raw0);
    f.get(wpLoc).v128Load(qsBase + 16).set(raw1);

    // Sub-block 2c (low nibbles + qh bit 2c): 2x vpmaddubsw -> i16x8.add (max 15748 <= 32767) -> pmaddwd(sc)
    f.get(raw0).get(mask0f).v128And();
    emitQhBit4(qh0, bitLo);
    f.v128Or().set(q5v);
    f.pushDotI8I7ToI16x8(xlo0, q5v, useRelaxed);

    f.get(raw1).get(mask0f).v128And();
    emitQhBit4(qh1, bitLo);
    f.v128Or().set(q5v);
    f.pushDotI8I7ToI16x8(xhi0, q5v, useRelaxed);

    f.i16x8Add();
    emitScaleSplat16(scWord, scLoIdx);
    f.i32x4DotI16x8S().get(sumiLoc).i32x4Add().set(sumiLoc);

    // Sub-block 2c+1 (high nibbles + qh bit 2c+1)
    f.get(raw0).i32Const(4).i8x16ShrU();
    emitQhBit4(qh0, bitHi);
    f.v128Or().set(q5v);
    f.pushDotI8I7ToI16x8(xlo1, q5v, useRelaxed);

    f.get(raw1).i32Const(4).i8x16ShrU();
    emitQhBit4(qh1, bitHi);
    f.v128Or().set(q5v);
    f.pushDotI8I7ToI16x8(xhi1, q5v, useRelaxed);

    f.i16x8Add();
    emitScaleSplat16(scWord, scHiIdx);
    f.i32x4DotI16x8S().get(sumiLoc).i32x4Add().set(sumiLoc);
  };

  // 2-row zero-spill main loop
  f.block().loop();
  f.get(r).i32Const(1).i32Add().get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().tee(wp0).get(rowBytes).i32Add().set(wp1);
  f.get(1).set(qp);
  f.f32Const(0).tee(acc0).set(acc1);
  f.i32Const(0).set(sb);

  f.block().loop();
  f.get(sb).get(nSuper).i32GeS().brIf(1);

  f.get(qp).f32Load(0).set(dxS);
  f.get(qp).v128Load(48, 4).set(xlo0);
  unpackScalesAndMinDot(wp0, sc03_0, sc47_0, xlo0, sumi0);
  unpackScalesAndMinDot(wp1, sc03_1, sc47_1, xlo0, sumi1);
  f.get(acc0).hsumI32x4(sumi0).f32ConvertI32S().loadF16AsF32(wp0, 2).get(dxS).f32Mul().f32Mul().f32Sub().set(acc0);
  f.get(acc1).hsumI32x4(sumi1).f32ConvertI32S().loadF16AsF32(wp1, 2).get(dxS).f32Mul().f32Mul().f32Sub().set(acc1);

  f.i32Const(0).i32x4Splat().tee(sumi0).set(sumi1);
  for (let c = 0; c < 4; c++) {
    const xBase = 64 + c * 64;
    f.get(qp).v128Load(xBase, 4).set(xlo0);
    f.get(qp).v128Load(xBase + 16, 4).set(xhi0);
    f.get(qp).v128Load(xBase + 32, 4).set(xlo1);
    f.get(qp).v128Load(xBase + 48, 4).set(xhi1);
    emitChunkRow(wp0, sc03_0, sc47_0, sumi0, c);
    emitChunkRow(wp1, sc03_1, sc47_1, sumi1, c);
  }

  f.hsumI32x4(sumi0).f32ConvertI32S().loadF16AsF32(wp0, 0).get(dxS).f32Mul().f32Mul()
    .get(acc0).f32Add().set(acc0);
  f.hsumI32x4(sumi1).f32ConvertI32S().loadF16AsF32(wp1, 0).get(dxS).f32Mul().f32Mul()
    .get(acc1).f32Add().set(acc1);

  f.get(wp0).i32Const(176).i32Add().set(wp0);
  f.get(wp1).i32Const(176).i32Add().set(wp1);
  f.get(qp).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(sb).i32Const(1).i32Add().set(sb);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc0).f32Store(0);
  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc1).f32Store(4);
  f.get(r).i32Const(2).i32Add().set(r);
  f.br(0).end().end();

  // Odd trailing row (0..1)
  f.block();
  f.get(r).get(4).i32GeS().brIf(0);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(wp0);
  f.get(1).set(qp);
  f.f32Const(0).set(acc0);
  f.i32Const(0).set(sb);

  f.block().loop();
  f.get(sb).get(nSuper).i32GeS().brIf(1);
  f.get(qp).f32Load(0).set(dxS);
  f.get(qp).v128Load(48, 4).set(xlo0);
  unpackScalesAndMinDot(wp0, sc03_0, sc47_0, xlo0, sumi1);
  f.i32Const(0).i32x4Splat().set(sumi0);
  for (let c = 0; c < 4; c++) {
    const xBase = 64 + c * 64;
    f.get(qp).v128Load(xBase, 4).set(xlo0);
    f.get(qp).v128Load(xBase + 16, 4).set(xhi0);
    f.get(qp).v128Load(xBase + 32, 4).set(xlo1);
    f.get(qp).v128Load(xBase + 48, 4).set(xhi1);
    emitChunkRow(wp0, sc03_0, sc47_0, sumi0, c);
  }
  f.loadF16AsF32(wp0, 0).hsumI32x4(sumi0).f32ConvertI32S().f32Mul()
    .loadF16AsF32(wp0, 2).hsumI32x4(sumi1).f32ConvertI32S().f32Mul()
    .f32Sub().get(dxS).f32Mul().get(acc0).f32Add().set(acc0);

  f.get(wp0).i32Const(176).i32Add().set(wp0);
  f.get(qp).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(sb).i32Const(1).i32Add().set(sb);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc0).f32Store(0);
  f.end();

  return f;
}

/**
 * Verdant Tensor `v2.22.4` 4-token batched prefill `Q5_K × Q8_K` Narrow-Lane Contraction:
 * `(wPtr, q8kBatchPtr, outBatchPtr, cols, rows, oStrideBytes)`.
 * Unpacks each `Q5_K` weight super-block (`qh[32]` + `qs[128]` + SWAR scales) ONCE across all
 * 4 prompt tokens, accumulates in pure `i32x4`, and encodes to float ONCE per 256 weights.
 */
function buildMulMatQ5KQ8KBatch4(useRelaxed: boolean): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const nSuper = f.local(I32);
  const rowBytes = f.local(I32);
  const qStride = f.local(I32);
  const oStride = f.local(I32);
  const wp = f.local(I32);
  const qp0 = f.local(I32);
  const qp1 = f.local(I32);
  const qp2 = f.local(I32);
  const qp3 = f.local(I32);
  const op = f.local(I32);
  const sb = f.local(I32);
  const u0 = f.local(I32);
  const u1 = f.local(I32);
  const u2 = f.local(I32);
  const sc03 = f.local(I32);
  const sc47 = f.local(I32);
  const mn03 = f.local(I32);
  const mn47 = f.local(I32);

  // Strictly 14 V128 locals -> 2 free XMM scratch registers on x86-64 (0 spills!)
  const mask0f = f.local(V128);
  const mask10 = f.local(V128);
  const acc4 = f.local(V128);
  const sumi0 = f.local(V128);
  const sumi1 = f.local(V128);
  const sumi2 = f.local(V128);
  const sumi3 = f.local(V128);
  const raw0 = f.local(V128);
  const raw1 = f.local(V128);
  const w0 = f.local(V128);
  const w1 = f.local(V128);
  const sc8V = f.local(V128);
  const scV = f.local(V128);
  const xtmp = f.local(V128);

  f.i32Const(0x0f).i8x16Splat().set(mask0f);
  f.i32Const(0x10).i8x16Splat().set(mask10);
  f.get(3).i32Const(8).i32ShrS().set(nSuper);
  f.get(nSuper).i32Const(176).i32Mul().set(rowBytes);
  f.get(nSuper).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Mul().set(qStride);
  f.get(5).set(oStride);
  f.i32Const(0).set(r);

  // Broadcast i16 lane `k` (0..7) from XMM `sc8V` in 1 single VPSHUFB (zero GPR->XMM stalls!)
  const emitScaleShuffle16 = (laneIdx: number) => {
    const lo = laneIdx * 2;
    const hi = lo + 1;
    f.get(sc8V).get(sc8V).i8x16Shuffle([lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi]).set(scV);
  };

  const emitQhBit4FromMem = (qhOff: number, bitIdx: number) => {
    f.get(wp).v128Load(qhOff);
    if (bitIdx < 4) {
      f.i32Const(4 - bitIdx).i8x16Shl();
    } else if (bitIdx > 4) {
      f.i32Const(bitIdx - 4).i8x16ShrU();
    }
    f.get(mask10).v128And();
  };

  const emitTokSub = (qpLoc: number, xOff: number, sumiLoc: number) => {
    f.get(qpLoc).v128Load(xOff, 4).set(xtmp);
    f.pushDotI8I7ToI16x8(xtmp, w0, useRelaxed);
    f.get(qpLoc).v128Load(xOff + 16, 4).set(xtmp);
    f.pushDotI8I7ToI16x8(xtmp, w1, useRelaxed);
    f.i16x8Add().get(scV).i32x4DotI16x8S().get(sumiLoc).i32x4Add().set(sumiLoc);
  };

  const pushDx4 = () => {
    f.f32Const(0).f32x4Splat()
      .get(qp0).f32Load(0).f32x4ReplaceLane(0)
      .get(qp1).f32Load(0).f32x4ReplaceLane(1)
      .get(qp2).f32Load(0).f32x4ReplaceLane(2)
      .get(qp3).f32Load(0).f32x4ReplaceLane(3);
  };

  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(wp);
  f.get(1).tee(qp0)
    .get(qStride).i32Add().tee(qp1)
    .get(qStride).i32Add().tee(qp2)
    .get(qStride).i32Add().set(qp3);
  f.f32Const(0).f32x4Splat().set(acc4);
  f.i32Const(0).set(sb);

  f.block().loop();
  f.get(sb).get(nSuper).i32GeS().brIf(1);

  // SWAR unpack scales & mins ONCE for all 4 tokens
  f.get(wp).i32Load(4).set(u0);
  f.get(wp).i32Load(8).set(u1);
  f.get(wp).i32Load(12).set(u2);
  f.get(u0).i32Const(0x3f3f3f3f).i32And().set(sc03);
  f.get(u1).i32Const(0x3f3f3f3f).i32And().set(mn03);
  f.get(u2).i32Const(0x0f0f0f0f).i32And()
    .get(u0).i32Const(2).i32ShrU().i32Const(0x30303030).i32And()
    .i32Or().set(sc47);
  f.get(u2).i32Const(4).i32ShrU().i32Const(0x0f0f0f0f).i32And()
    .get(u1).i32Const(2).i32ShrU().i32Const(0x30303030).i32And()
    .i32Or().set(mn47);

  // Compute min dots for all 4 tokens via 1x dot_i16x8 each, fold into acc4
  f.i32Const(0).i32x4Splat().get(mn03).i32x4ReplaceLane(0).get(mn47).i32x4ReplaceLane(1)
    .i16x8ExtendLowI8x16U().set(scV);
  f.get(scV).get(qp0).v128Load(48, 4).i32x4DotI16x8S().set(sumi0);
  f.get(scV).get(qp1).v128Load(48, 4).i32x4DotI16x8S().set(sumi1);
  f.get(scV).get(qp2).v128Load(48, 4).i32x4DotI16x8S().set(sumi2);
  f.get(scV).get(qp3).v128Load(48, 4).i32x4DotI16x8S().set(sumi3);
  f.hsum4RowsI32x4(sumi0, sumi1, sumi2, sumi3, raw0, raw1, w0);
  pushDx4();
  f.loadF16AsF32(wp, 2).f32x4Splat().f32x4Mul().set(raw0);
  f.get(acc4)
    .get(w0).f32x4ConvertI32x4S().get(raw0).f32x4Mul()
    .f32x4Sub().set(acc4);

  // Pack all 8 scales into XMM `sc8V` as i16x8 -> frees sc03/sc47 GPRs across chunk loop!
  f.i32Const(0).i32x4Splat().get(sc03).i32x4ReplaceLane(0).get(sc47).i32x4ReplaceLane(1)
    .i16x8ExtendLowI8x16U().set(sc8V);

  f.i32Const(0).i32x4Splat().tee(sumi0).tee(sumi1).tee(sumi2).set(sumi3);

  for (let c = 0; c < 4; c++) {
    const qsBase = 48 + c * 32;
    const xBase = 64 + c * 64;
    const bitLo = c * 2;
    const bitHi = c * 2 + 1;

    f.get(wp).v128Load(qsBase).set(raw0);
    f.get(wp).v128Load(qsBase + 16).set(raw1);

    // Sub-block 2c (low nibbles + qh bit 2c) across all 4 tokens
    f.get(raw0).get(mask0f).v128And();
    emitQhBit4FromMem(16, bitLo);
    f.v128Or().set(w0);
    f.get(raw1).get(mask0f).v128And();
    emitQhBit4FromMem(32, bitLo);
    f.v128Or().set(w1);
    emitScaleShuffle16(c * 2);
    emitTokSub(qp0, xBase, sumi0);
    emitTokSub(qp1, xBase, sumi1);
    emitTokSub(qp2, xBase, sumi2);
    emitTokSub(qp3, xBase, sumi3);

    // Sub-block 2c+1 (high nibbles + qh bit 2c+1) across all 4 tokens
    f.get(raw0).i32Const(4).i8x16ShrU();
    emitQhBit4FromMem(16, bitHi);
    f.v128Or().set(w0);
    f.get(raw1).i32Const(4).i8x16ShrU();
    emitQhBit4FromMem(32, bitHi);
    f.v128Or().set(w1);
    emitScaleShuffle16(c * 2 + 1);
    emitTokSub(qp0, xBase + 32, sumi0);
    emitTokSub(qp1, xBase + 32, sumi1);
    emitTokSub(qp2, xBase + 32, sumi2);
    emitTokSub(qp3, xBase + 32, sumi3);
  }

  // Encode all 4 tokens ONCE per 256-weight super-block!
  f.hsum4RowsI32x4(sumi0, sumi1, sumi2, sumi3, raw0, raw1, w0);
  pushDx4();
  f.loadF16AsF32(wp, 0).f32x4Splat().f32x4Mul().set(raw0);
  if (useRelaxed) {
    f.get(w0).f32x4ConvertI32x4S().get(raw0).get(acc4).f32x4RelaxedMadd().set(acc4);
  } else {
    f.get(w0).f32x4ConvertI32x4S().get(raw0).f32x4Mul().get(acc4).f32x4Add().set(acc4);
  }

  f.get(wp).i32Const(176).i32Add().set(wp);
  f.get(qp0).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp0);
  f.get(qp1).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp1);
  f.get(qp2).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp2);
  f.get(qp3).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp3);
  f.get(sb).i32Const(1).i32Add().set(sb);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().tee(op).get(acc4).f32x4Extract(0).f32Store(0);
  f.get(op).get(oStride).i32Add().tee(op).get(acc4).f32x4Extract(1).f32Store(0);
  f.get(op).get(oStride).i32Add().tee(op).get(acc4).f32x4Extract(2).f32Store(0);
  f.get(op).get(oStride).i32Add().get(acc4).f32x4Extract(3).f32Store(0);

  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();

  return f;
}

/**
 * Verdant/Prism 2-row register-blocked `Q6_K × Q8_K` Narrow-Lane Integer Contraction:
 * `(wPtr, q8kPtr, outPtr, cols, rows)`.
 * - Absorbs all 16 sub-blocks' `-32 * sc[j] * bsums16[j]` offset via TWO `i32x4.dot_i16x8_s` ops
 * - Fuses every sub-block scale multiply into 1-op `i32x4.dot_i16x8_s(vpmaddubsw, i16x8.splat(sc))`
 *   (ZERO `pmulld`, ZERO `qhv` reloads!)
 * - `mask30 = 0x30` bit-shift elimination saves 16 `i8x16` vector shifts (~32 x86-64 ops) per super-block
 * - Strictly 12 live V128 registers (zero x86-64 XMM spills in V8 TurboFan!) + compact rolled `h` loop
 */
function buildMulMatQ6KQ8K(useRelaxed: boolean): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const nSuper = f.local(I32);
  const rowBytes = f.local(I32);
  const wp0 = f.local(I32);
  const wp1 = f.local(I32);
  const qp = f.local(I32);
  const sb = f.local(I32);
  const acc0 = f.local(F32);
  const acc1 = f.local(F32);
  const dxS = f.local(F32);

  // Strictly 13 V128 locals -> 3 free XMM scratch registers on x86-64 (0 spills!)
  // Eliminates 8 extra GPR pointer locals & 32 GPR->XMM i16x8Splat stalls per super-block!
  const mask0f = f.local(V128);
  const mask30 = f.local(V128);
  const sumi0 = f.local(V128);
  const sumi1 = f.local(V128);
  const x0 = f.local(V128);
  const x1 = f.local(V128);
  const x2 = f.local(V128);
  const x3 = f.local(V128);
  const qla = f.local(V128);
  const qlb = f.local(V128);
  const qhv = f.local(V128);
  const q6v = f.local(V128);
  const scHalf = f.local(V128);

  f.i32Const(0x0f).i8x16Splat().set(mask0f);
  f.i32Const(0x30).i8x16Splat().set(mask30);
  f.get(3).i32Const(8).i32ShrS().set(nSuper);
  f.get(nSuper).i32Const(210).i32Mul().set(rowBytes);
  f.i32Const(0).set(r);

  // Initialize sumiLoc = -32 * sum_{j=0..15} sc[j] * bsums16[j]
  const initRowOffset = (wpLoc: number, sumiLoc: number) => {
    f.get(wpLoc).v128Load(192).set(qhv);
    f.i32Const(0).i32x4Splat()
      .get(qhv).i16x8ExtendLowI8x16S().get(x0).i32x4DotI16x8S()
      .get(qhv).i16x8ExtendHighI8x16S().get(x1).i32x4DotI16x8S()
      .i32x4Add().i32Const(5).i32x4Shl()
      .i32x4Sub().set(sumiLoc);
  };

  // Emit all 4 groups of slice `s` (0 or 1) in half `h` (0 or 1) purely in XMM domain
  const emitSliceRow = (
    wpLoc: number,
    sumiLoc: number,
    h: number,
    s: number,
  ) => {
    const qlBase = h * 64 + s * 16;
    const qhBase = 128 + h * 32 + s * 16;
    f.get(wpLoc).v128Load(192);
    if (h === 0) f.i16x8ExtendLowI8x16S().set(scHalf);
    else f.i16x8ExtendHighI8x16S().set(scHalf);
    f.get(wpLoc).v128Load(qlBase).set(qla);
    f.get(wpLoc).v128Load(qlBase + 32).set(qlb);
    f.get(wpLoc).v128Load(qhBase).set(qhv);

    const emitGroup = (xLoc: number, scLane: number, buildQ6: () => void) => {
      const lo = scLane * 2;
      const hi = lo + 1;
      buildQ6();
      f.set(q6v);
      f.pushDotI8I7ToI16x8(xLoc, q6v, useRelaxed);
      f.get(scHalf).get(scHalf).i8x16Shuffle([lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi])
        .i32x4DotI16x8S().get(sumiLoc).i32x4Add().set(sumiLoc);
    };

    emitGroup(x0, 0 + s, () => {
      f.get(qla).get(mask0f).v128And()
        .get(qhv).i32Const(4).i8x16Shl().get(mask30).v128And().v128Or();
    });
    emitGroup(x1, 2 + s, () => {
      f.get(qlb).get(mask0f).v128And()
        .get(qhv).i32Const(2).i8x16Shl().get(mask30).v128And().v128Or();
    });
    emitGroup(x2, 4 + s, () => {
      f.get(qla).i32Const(4).i8x16ShrU()
        .get(qhv).get(mask30).v128And().v128Or();
    });
    emitGroup(x3, 6 + s, () => {
      f.get(qlb).i32Const(4).i8x16ShrU()
        .get(qhv).i32Const(2).i8x16ShrU().get(mask30).v128And().v128Or();
    });
  };

  // 2-row zero-spill main loop with unrolled halves (zero pointer-increment GPR overhead)
  f.block().loop();
  f.get(r).i32Const(1).i32Add().get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().tee(wp0).get(rowBytes).i32Add().set(wp1);
  f.get(1).set(qp);
  f.f32Const(0).tee(acc0).set(acc1);
  f.i32Const(0).set(sb);

  f.block().loop();
  f.get(sb).get(nSuper).i32GeS().brIf(1);

  f.get(qp).f32Load(0).set(dxS);
  f.get(qp).v128Load(16, 4).set(x0);
  f.get(qp).v128Load(32, 4).set(x1);
  initRowOffset(wp0, sumi0);
  initRowOffset(wp1, sumi1);

  for (let h = 0; h < 2; h++) {
    const xHalf = 64 + h * 128;
    for (let s = 0; s < 2; s++) {
      const sOff = s * 16;
      f.get(qp).v128Load(xHalf + 0 + sOff, 4).set(x0);
      f.get(qp).v128Load(xHalf + 32 + sOff, 4).set(x1);
      f.get(qp).v128Load(xHalf + 64 + sOff, 4).set(x2);
      f.get(qp).v128Load(xHalf + 96 + sOff, 4).set(x3);
      emitSliceRow(wp0, sumi0, h, s);
      emitSliceRow(wp1, sumi1, h, s);
    }
  }

  f.hsumI32x4(sumi0).f32ConvertI32S().loadF16AsF32(wp0, 208).get(dxS).f32Mul().f32Mul()
    .get(acc0).f32Add().set(acc0);
  f.hsumI32x4(sumi1).f32ConvertI32S().loadF16AsF32(wp1, 208).get(dxS).f32Mul().f32Mul()
    .get(acc1).f32Add().set(acc1);

  f.get(wp0).i32Const(210).i32Add().set(wp0);
  f.get(wp1).i32Const(210).i32Add().set(wp1);
  f.get(qp).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(sb).i32Const(1).i32Add().set(sb);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc0).f32Store(0);
  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc1).f32Store(4);
  f.get(r).i32Const(2).i32Add().set(r);
  f.br(0).end().end();

  // Odd trailing row (0..1)
  f.block();
  f.get(r).get(4).i32GeS().brIf(0);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(wp0);
  f.get(1).set(qp);
  f.f32Const(0).set(acc0);
  f.i32Const(0).set(sb);

  f.block().loop();
  f.get(sb).get(nSuper).i32GeS().brIf(1);
  f.get(qp).f32Load(0).set(dxS);
  f.get(qp).v128Load(16, 4).set(x0);
  f.get(qp).v128Load(32, 4).set(x1);
  initRowOffset(wp0, sumi0);

  for (let h = 0; h < 2; h++) {
    const xHalf = 64 + h * 128;
    for (let s = 0; s < 2; s++) {
      const sOff = s * 16;
      f.get(qp).v128Load(xHalf + 0 + sOff, 4).set(x0);
      f.get(qp).v128Load(xHalf + 32 + sOff, 4).set(x1);
      f.get(qp).v128Load(xHalf + 64 + sOff, 4).set(x2);
      f.get(qp).v128Load(xHalf + 96 + sOff, 4).set(x3);
      emitSliceRow(wp0, sumi0, h, s);
    }
  }

  f.hsumI32x4(sumi0).f32ConvertI32S().loadF16AsF32(wp0, 208).get(dxS).f32Mul().f32Mul()
    .get(acc0).f32Add().set(acc0);

  f.get(wp0).i32Const(210).i32Add().set(wp0);
  f.get(qp).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(sb).i32Const(1).i32Add().set(sb);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc0).f32Store(0);
  f.end();

  return f;
}

/**
 * Verdant/Prism 4-token batched prefill `Q6_K × Q8_K` Narrow-Lane Contraction:
 * `(wPtr, q8kBatchPtr, outBatchPtr, cols, rows, oStrideBytes)`.
 */
function buildMulMatQ6KQ8KBatch4(useRelaxed: boolean): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const nSuper = f.local(I32);
  const rowBytes = f.local(I32);
  const qStride = f.local(I32);
  const oStride = f.local(I32);
  const wp = f.local(I32);
  const qp0 = f.local(I32);
  const qp1 = f.local(I32);
  const qp2 = f.local(I32);
  const qp3 = f.local(I32);
  const op = f.local(I32);
  const sb = f.local(I32);

  const mask0f = f.local(V128);
  const mask30 = f.local(V128);
  const acc4 = f.local(V128);
  const sumi0 = f.local(V128);
  const sumi1 = f.local(V128);
  const sumi2 = f.local(V128);
  const sumi3 = f.local(V128);
  const qla = f.local(V128);
  const qlb = f.local(V128);
  const qhv = f.local(V128);
  const q6v = f.local(V128);
  const scV = f.local(V128);
  const xtmp = f.local(V128);
  const dotV = f.local(V128);

  f.i32Const(0x0f).i8x16Splat().set(mask0f);
  f.i32Const(0x30).i8x16Splat().set(mask30);
  f.get(3).i32Const(8).i32ShrS().set(nSuper);
  f.get(nSuper).i32Const(210).i32Mul().set(rowBytes);
  f.get(nSuper).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Mul().set(qStride);
  f.get(5).set(oStride);
  f.i32Const(0).set(r);

  const initTokOffset = (qpLoc: number, sumiLoc: number) => {
    f.i32Const(0).i32x4Splat()
      .get(qla).get(qpLoc).v128Load(16, 4).i32x4DotI16x8S()
      .get(qlb).get(qpLoc).v128Load(32, 4).i32x4DotI16x8S()
      .i32x4Add().i32Const(5).i32x4Shl()
      .i32x4Sub().set(sumiLoc);
  };

  const emitTokQ6 = (qpLoc: number, xOff: number, sumiLoc: number) => {
    f.get(qpLoc).v128Load(xOff, 4).set(xtmp);
    f.pushDotI8I7ToI16x8(xtmp, q6v, useRelaxed);
    f.get(scV).i32x4DotI16x8S().get(sumiLoc).i32x4Add().set(sumiLoc);
  };

  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(wp);
  f.get(1).tee(qp0)
    .get(qStride).i32Add().tee(qp1)
    .get(qStride).i32Add().tee(qp2)
    .get(qStride).i32Add().set(qp3);
  f.f32Const(0).f32x4Splat().set(acc4);
  f.i32Const(0).set(sb);

  f.block().loop();
  f.get(sb).get(nSuper).i32GeS().brIf(1);

  // Load 16 scales -> extend to qla (low 8) & qlb (high 8), absorb -32*sc*bsum for all 4 tokens
  f.get(wp).v128Load(192).set(qhv);
  f.get(qhv).i16x8ExtendLowI8x16S().set(qla);
  f.get(qhv).i16x8ExtendHighI8x16S().set(qlb);
  initTokOffset(qp0, sumi0);
  initTokOffset(qp1, sumi1);
  initTokOffset(qp2, sumi2);
  initTokOffset(qp3, sumi3);

  for (let h = 0; h < 2; h++) {
    const qlBase = h * 64;
    const qhBase = 128 + h * 32;
    const xHalf = 64 + h * 128;

    // Extend 8 signed i8 scales for this half into `dotV` (dead during h-loop) -> 1-op VPSHUFB per sub-block!
    f.get(wp).v128Load(192);
    if (h === 0) f.i16x8ExtendLowI8x16S().set(dotV);
    else f.i16x8ExtendHighI8x16S().set(dotV);

    for (let s = 0; s < 2; s++) {
      const sOff = s * 16;
      f.get(wp).v128Load(qlBase + sOff).set(qla);
      f.get(wp).v128Load(qlBase + 32 + sOff).set(qlb);
      f.get(wp).v128Load(qhBase + sOff).set(qhv);

      const emitGroup4 = (xOff: number, scLane: number, buildQ6: () => void) => {
        const lo = scLane * 2;
        const hi = lo + 1;
        buildQ6();
        f.set(q6v);
        f.get(dotV).get(dotV).i8x16Shuffle([lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi]).set(scV);
        emitTokQ6(qp0, xOff, sumi0);
        emitTokQ6(qp1, xOff, sumi1);
        emitTokQ6(qp2, xOff, sumi2);
        emitTokQ6(qp3, xOff, sumi3);
      };

      emitGroup4(xHalf + 0 + sOff, 0 + s, () => {
        f.get(qla).get(mask0f).v128And()
          .get(qhv).i32Const(4).i8x16Shl().get(mask30).v128And().v128Or();
      });
      emitGroup4(xHalf + 32 + sOff, 2 + s, () => {
        f.get(qlb).get(mask0f).v128And()
          .get(qhv).i32Const(2).i8x16Shl().get(mask30).v128And().v128Or();
      });
      emitGroup4(xHalf + 64 + sOff, 4 + s, () => {
        f.get(qla).i32Const(4).i8x16ShrU()
          .get(qhv).get(mask30).v128And().v128Or();
      });
      emitGroup4(xHalf + 96 + sOff, 6 + s, () => {
        f.get(qlb).i32Const(4).i8x16ShrU()
          .get(qhv).i32Const(2).i8x16ShrU().get(mask30).v128And().v128Or();
      });
    }
  }

  f.hsum4RowsI32x4(sumi0, sumi1, sumi2, sumi3, qla, qlb, dotV);
  f.f32Const(0).f32x4Splat()
    .get(qp0).f32Load(0).f32x4ReplaceLane(0)
    .get(qp1).f32Load(0).f32x4ReplaceLane(1)
    .get(qp2).f32Load(0).f32x4ReplaceLane(2)
    .get(qp3).f32Load(0).f32x4ReplaceLane(3)
    .loadF16AsF32(wp, 208).f32x4Splat().f32x4Mul().set(qla);
  if (useRelaxed) {
    f.get(dotV).f32x4ConvertI32x4S().get(qla).get(acc4).f32x4RelaxedMadd().set(acc4);
  } else {
    f.get(dotV).f32x4ConvertI32x4S().get(qla).f32x4Mul().get(acc4).f32x4Add().set(acc4);
  }

  f.get(wp).i32Const(210).i32Add().set(wp);
  f.get(qp0).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp0);
  f.get(qp1).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp1);
  f.get(qp2).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp2);
  f.get(qp3).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp3);
  f.get(sb).i32Const(1).i32Add().set(sb);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().tee(op).get(acc4).f32x4Extract(0).f32Store(0);
  f.get(op).get(oStride).i32Add().tee(op).get(acc4).f32x4Extract(1).f32Store(0);
  f.get(op).get(oStride).i32Add().tee(op).get(acc4).f32x4Extract(2).f32Store(0);
  f.get(op).get(oStride).i32Add().get(acc4).f32x4Extract(3).f32Store(0);

  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();

  return f;
}

/**
 * Verdant/Prism 4-row register-blocked `Q2_K × Q8_K` Narrow-Lane Integer Contraction:
 * `(wPtr, q8kPtr, outPtr, cols, rows)`.
 * - 12 live V128 registers (zero x86-64 XMM spills!)
 * - Computes all 16 `dmin` sub-block terms in 2x `i32x4.dot_i16x8_s` against `Q8_K.bsums16`
 * - Pure `i32x4` narrow-lane accumulation (1,467x i32 headroom, 43x i16 headroom)
 * - Encodes to float ONCE per 256 weights via `hsum4RowsI32x4`!
 */
function buildMulMatQ2KQ8K(useRelaxed: boolean): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const nSuper = f.local(I32);
  const rowBytes = f.local(I32);
  const wp0 = f.local(I32);
  const wp1 = f.local(I32);
  const wp2 = f.local(I32);
  const wp3 = f.local(I32);
  const qp = f.local(I32);
  const sb = f.local(I32);
  const accS = f.local(F32);
  const dxS = f.local(F32);

  // Strictly 12 V128 locals -> 4 free XMM scratch registers (0 spills!)
  const mask03 = f.local(V128);
  const acc4 = f.local(V128);
  const sumi0 = f.local(V128);
  const sumi1 = f.local(V128);
  const sumi2 = f.local(V128);
  const sumi3 = f.local(V128);
  const q0 = f.local(V128);
  const q1 = f.local(V128);
  const qv = f.local(V128);
  const x0 = f.local(V128);
  const x1 = f.local(V128);
  const dotV = f.local(V128);

  f.i32Const(0x03).i8x16Splat().set(mask03);
  f.get(3).i32Const(8).i32ShrS().set(nSuper);
  f.get(nSuper).i32Const(84).i32Mul().set(rowBytes);
  f.i32Const(0).set(r);

  // Compute dot(ml[0..15], bsums16[0..15]) where x0 = bsums16[0..7], x1 = bsums16[8..15]
  const emitMinDot = (wpLoc: number, dstLoc: number) => {
    f.get(wpLoc).v128Load(0).i32Const(4).i8x16ShrU().set(qv);
    f.get(qv).i16x8ExtendLowI8x16U().get(x0).i32x4DotI16x8S()
      .get(qv).i16x8ExtendHighI8x16U().get(x1).i32x4DotI16x8S()
      .i32x4Add().set(dstLoc);
  };

  // XMM-domain scale broadcast: 1 single VPSHUFB per sub-block (zero scalar loads, zero GPR->XMM stalls!)
  const emitScaleShuffle = (scHalfLoc: number, laneIdx: number) => {
    const lo = laneIdx * 2;
    const hi = lo + 1;
    f.get(scHalfLoc).get(scHalfLoc).i8x16Shuffle([lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi]);
  };

  // Accumulate all 16 sub-blocks of one row `wpLoc` into `sumiLoc` purely in XMM domain
  const emitRowQ2 = (wpLoc: number, sumiLoc: number) => {
    // Load all 16 4-bit scales once into dotV (dead during emitRowQ2), extend low/high halves into x1
    f.get(wpLoc).v128Load(0).v128Const([15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]).v128And().set(dotV);
    for (let h = 0; h < 2; h++) {
      if (h === 0) f.get(dotV).i16x8ExtendLowI8x16U().set(x1);
      else f.get(dotV).i16x8ExtendHighI8x16U().set(x1);
      f.get(wpLoc).v128Load(16 + h * 32).set(q0);
      f.get(wpLoc).v128Load(32 + h * 32).set(q1);
      for (let s = 0; s < 4; s++) {
        const shift = s * 2;
        const xOff = 64 + h * 128 + s * 32;
        f.get(qp).v128Load(xOff, 4).set(x0);
        f.get(q0);
        if (shift > 0) f.i32Const(shift).i8x16ShrU();
        if (s < 3) f.get(mask03).v128And();
        f.set(qv);
        f.pushDotI8I7ToI16x8(x0, qv, useRelaxed);
        emitScaleShuffle(x1, s * 2);
        f.i32x4DotI16x8S().get(sumiLoc).i32x4Add().set(sumiLoc);

        f.get(qp).v128Load(xOff + 16, 4).set(x0);
        f.get(q1);
        if (shift > 0) f.i32Const(shift).i8x16ShrU();
        if (s < 3) f.get(mask03).v128And();
        f.set(qv);
        f.pushDotI8I7ToI16x8(x0, qv, useRelaxed);
        emitScaleShuffle(x1, s * 2 + 1);
        f.i32x4DotI16x8S().get(sumiLoc).i32x4Add().set(sumiLoc);
      }
    }
  };

  // 4-row register-blocked loop
  f.block().loop();
  f.get(r).i32Const(3).i32Add().get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add()
    .tee(wp0).get(rowBytes).i32Add()
    .tee(wp1).get(rowBytes).i32Add()
    .tee(wp2).get(rowBytes).i32Add().set(wp3);
  f.get(1).set(qp);
  f.f32Const(0).f32x4Splat().set(acc4);
  f.i32Const(0).set(sb);

  f.block().loop();
  f.get(sb).get(nSuper).i32GeS().brIf(1);

  f.get(qp).f32Load(0).set(dxS);
  f.get(qp).v128Load(16, 4).set(x0);
  f.get(qp).v128Load(32, 4).set(x1);
  emitMinDot(wp0, sumi0);
  emitMinDot(wp1, sumi1);
  emitMinDot(wp2, sumi2);
  emitMinDot(wp3, sumi3);
  f.hsum4RowsI32x4(sumi0, sumi1, sumi2, sumi3, q0, q1, dotV);
  f.f32Const(0).f32x4Splat()
    .loadF16AsF32(wp0, 82).f32x4ReplaceLane(0)
    .loadF16AsF32(wp1, 82).f32x4ReplaceLane(1)
    .loadF16AsF32(wp2, 82).f32x4ReplaceLane(2)
    .loadF16AsF32(wp3, 82).f32x4ReplaceLane(3)
    .get(dxS).f32x4Splat().f32x4Mul().set(q0);
  f.get(acc4)
    .get(dotV).f32x4ConvertI32x4S().get(q0).f32x4Mul()
    .f32x4Sub().set(acc4);

  f.i32Const(0).i32x4Splat().tee(sumi0).tee(sumi1).tee(sumi2).set(sumi3);
  emitRowQ2(wp0, sumi0);
  emitRowQ2(wp1, sumi1);
  emitRowQ2(wp2, sumi2);
  emitRowQ2(wp3, sumi3);

  f.hsum4RowsI32x4(sumi0, sumi1, sumi2, sumi3, q0, q1, dotV);
  f.f32Const(0).f32x4Splat()
    .loadF16AsF32(wp0, 80).f32x4ReplaceLane(0)
    .loadF16AsF32(wp1, 80).f32x4ReplaceLane(1)
    .loadF16AsF32(wp2, 80).f32x4ReplaceLane(2)
    .loadF16AsF32(wp3, 80).f32x4ReplaceLane(3)
    .get(dxS).f32x4Splat().f32x4Mul().set(q0);
  if (useRelaxed) {
    f.get(dotV).f32x4ConvertI32x4S().get(q0).get(acc4).f32x4RelaxedMadd().set(acc4);
  } else {
    f.get(dotV).f32x4ConvertI32x4S().get(q0).f32x4Mul().get(acc4).f32x4Add().set(acc4);
  }

  f.get(wp0).i32Const(84).i32Add().set(wp0);
  f.get(wp1).i32Const(84).i32Add().set(wp1);
  f.get(wp2).i32Const(84).i32Add().set(wp2);
  f.get(wp3).i32Const(84).i32Add().set(wp3);
  f.get(qp).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(sb).i32Const(1).i32Add().set(sb);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc4).v128Store(0);
  f.get(r).i32Const(4).i32Add().set(r);
  f.br(0).end().end();

  // Remainder rows (0..3)
  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(wp0);
  f.get(1).set(qp);
  f.f32Const(0).set(accS);
  f.i32Const(0).set(sb);

  f.block().loop();
  f.get(sb).get(nSuper).i32GeS().brIf(1);
  f.get(qp).f32Load(0).set(dxS);
  f.get(qp).v128Load(16, 4).set(x0);
  f.get(qp).v128Load(32, 4).set(x1);
  emitMinDot(wp0, sumi1);
  f.i32Const(0).i32x4Splat().set(sumi0);
  emitRowQ2(wp0, sumi0);
  f.loadF16AsF32(wp0, 80).hsumI32x4(sumi0).f32ConvertI32S().f32Mul()
    .loadF16AsF32(wp0, 82).hsumI32x4(sumi1).f32ConvertI32S().f32Mul()
    .f32Sub().get(dxS).f32Mul().get(accS).f32Add().set(accS);

  f.get(wp0).i32Const(84).i32Add().set(wp0);
  f.get(qp).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(sb).i32Const(1).i32Add().set(sb);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(accS).f32Store(0);
  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();

  return f;
}

/**
 * Verdant/Prism 4-token batched prefill `Q2_K × Q8_K` Narrow-Lane Contraction:
 * `(wPtr, q8kBatchPtr, outBatchPtr, cols, rows, oStrideBytes)`.
 */
function buildMulMatQ2KQ8KBatch4(useRelaxed: boolean): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const nSuper = f.local(I32);
  const rowBytes = f.local(I32);
  const qStride = f.local(I32);
  const oStride = f.local(I32);
  const wp = f.local(I32);
  const qp0 = f.local(I32);
  const qp1 = f.local(I32);
  const qp2 = f.local(I32);
  const qp3 = f.local(I32);
  const op = f.local(I32);
  const sb = f.local(I32);

  const mask03 = f.local(V128);
  const acc4 = f.local(V128);
  const dx4 = f.local(V128);
  const sumi0 = f.local(V128);
  const sumi1 = f.local(V128);
  const sumi2 = f.local(V128);
  const sumi3 = f.local(V128);
  const q0 = f.local(V128);
  const q1 = f.local(V128);
  const qv = f.local(V128);
  const scV = f.local(V128);
  const xtmp = f.local(V128);
  const dotV = f.local(V128);

  f.i32Const(0x03).i8x16Splat().set(mask03);
  f.get(3).i32Const(8).i32ShrS().set(nSuper);
  f.get(nSuper).i32Const(84).i32Mul().set(rowBytes);
  f.get(nSuper).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Mul().set(qStride);
  f.get(5).set(oStride);
  f.i32Const(0).set(r);

  const emitTokMin = (qpLoc: number, dstLoc: number) => {
    f.get(q0).get(qpLoc).v128Load(16, 4).i32x4DotI16x8S()
      .get(q1).get(qpLoc).v128Load(32, 4).i32x4DotI16x8S()
      .i32x4Add().set(dstLoc);
  };

  const emitTokSub = (qpLoc: number, xOff: number, sumiLoc: number) => {
    f.get(qpLoc).v128Load(xOff, 4).set(xtmp);
    f.pushDotI8I7ToI16x8(xtmp, qv, useRelaxed);
    f.get(scV).i32x4DotI16x8S().get(sumiLoc).i32x4Add().set(sumiLoc);
  };

  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(wp);
  f.get(1).tee(qp0)
    .get(qStride).i32Add().tee(qp1)
    .get(qStride).i32Add().tee(qp2)
    .get(qStride).i32Add().set(qp3);
  f.f32Const(0).f32x4Splat().set(acc4);
  f.i32Const(0).set(sb);

  f.block().loop();
  f.get(sb).get(nSuper).i32GeS().brIf(1);

  f.f32Const(0).f32x4Splat()
    .get(qp0).f32Load(0).f32x4ReplaceLane(0)
    .get(qp1).f32Load(0).f32x4ReplaceLane(1)
    .get(qp2).f32Load(0).f32x4ReplaceLane(2)
    .get(qp3).f32Load(0).f32x4ReplaceLane(3)
    .set(dx4);

  // Min term across all 4 tokens
  f.get(wp).v128Load(0).i32Const(4).i8x16ShrU().set(qv);
  f.get(qv).i16x8ExtendLowI8x16U().set(q0);
  f.get(qv).i16x8ExtendHighI8x16U().set(q1);
  emitTokMin(qp0, sumi0);
  emitTokMin(qp1, sumi1);
  emitTokMin(qp2, sumi2);
  emitTokMin(qp3, sumi3);
  f.hsum4RowsI32x4(sumi0, sumi1, sumi2, sumi3, q0, q1, dotV);
  f.get(acc4)
    .get(dotV).f32x4ConvertI32x4S()
    .loadF16AsF32(wp, 82).f32x4Splat().get(dx4).f32x4Mul().f32x4Mul()
    .f32x4Sub().set(acc4);

  f.i32Const(0).i32x4Splat().tee(sumi0).tee(sumi1).tee(sumi2).set(sumi3);

  // XMM-domain scale broadcast: extend 8 scales per half into `dotV` once, then 1-op VPSHUFB per sub-block
  for (let h = 0; h < 2; h++) {
    f.get(wp).v128Load(0).v128Const([15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]).v128And();
    if (h === 0) f.i16x8ExtendLowI8x16U().set(dotV);
    else f.i16x8ExtendHighI8x16U().set(dotV);
    f.get(wp).v128Load(16 + h * 32).set(q0);
    f.get(wp).v128Load(32 + h * 32).set(q1);
    for (let s = 0; s < 4; s++) {
      const shift = s * 2;
      const xOff = 64 + h * 128 + s * 32;
      const l0 = s * 4;
      const l1 = l0 + 2;

      f.get(q0);
      if (shift > 0) f.i32Const(shift).i8x16ShrU();
      if (s < 3) f.get(mask03).v128And();
      f.set(qv);
      f.get(dotV).get(dotV).i8x16Shuffle([l0, l0 + 1, l0, l0 + 1, l0, l0 + 1, l0, l0 + 1, l0, l0 + 1, l0, l0 + 1, l0, l0 + 1, l0, l0 + 1]).set(scV);
      emitTokSub(qp0, xOff, sumi0);
      emitTokSub(qp1, xOff, sumi1);
      emitTokSub(qp2, xOff, sumi2);
      emitTokSub(qp3, xOff, sumi3);

      f.get(q1);
      if (shift > 0) f.i32Const(shift).i8x16ShrU();
      if (s < 3) f.get(mask03).v128And();
      f.set(qv);
      f.get(dotV).get(dotV).i8x16Shuffle([l1, l1 + 1, l1, l1 + 1, l1, l1 + 1, l1, l1 + 1, l1, l1 + 1, l1, l1 + 1, l1, l1 + 1, l1, l1 + 1]).set(scV);
      emitTokSub(qp0, xOff + 16, sumi0);
      emitTokSub(qp1, xOff + 16, sumi1);
      emitTokSub(qp2, xOff + 16, sumi2);
      emitTokSub(qp3, xOff + 16, sumi3);
    }
  }

  f.hsum4RowsI32x4(sumi0, sumi1, sumi2, sumi3, q0, q1, dotV);
  f.loadF16AsF32(wp, 80).f32x4Splat().get(dx4).f32x4Mul().set(q0);
  if (useRelaxed) {
    f.get(dotV).f32x4ConvertI32x4S().get(q0).get(acc4).f32x4RelaxedMadd().set(acc4);
  } else {
    f.get(dotV).f32x4ConvertI32x4S().get(q0).f32x4Mul().get(acc4).f32x4Add().set(acc4);
  }

  f.get(wp).i32Const(84).i32Add().set(wp);
  f.get(qp0).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp0);
  f.get(qp1).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp1);
  f.get(qp2).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp2);
  f.get(qp3).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp3);
  f.get(sb).i32Const(1).i32Add().set(sb);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().tee(op).get(acc4).f32x4Extract(0).f32Store(0);
  f.get(op).get(oStride).i32Add().tee(op).get(acc4).f32x4Extract(1).f32Store(0);
  f.get(op).get(oStride).i32Add().tee(op).get(acc4).f32x4Extract(2).f32Store(0);
  f.get(op).get(oStride).i32Add().get(acc4).f32x4Extract(3).f32Store(0);

  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();

  return f;
}

/**
 * Verdant/Prism 2-row zero-spill `Q3_K × Q8_K` Narrow-Lane Integer Contraction:
 * `(wPtr, q8kPtr, outPtr, cols, rows)`.
 * - Factors `q = q_u3 - 4` (`q_u3 ∈ [0..7]`) so `-4 * sum_j(s_j * bsums16[j])` is absorbed up front
 *   in 2x `i32x4.dot_i16x8_s` + `shl(2)`, allowing hardware unsigned-i7 VNNI dots!
 * - Strictly 11 live V128 registers (zero x86-64 XMM spills!) & 295x i32 headroom.
 */
function buildMulMatQ3KQ8K(useRelaxed: boolean): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const nSuper = f.local(I32);
  const rowBytes = f.local(I32);
  const wp0 = f.local(I32);
  const wp1 = f.local(I32);
  const qp = f.local(I32);
  const sb = f.local(I32);
  const a0 = f.local(I32);
  const a1 = f.local(I32);
  const tmp = f.local(I32);
  const u0 = f.local(I32);
  const u1 = f.local(I32);
  const u2 = f.local(I32);
  const u3 = f.local(I32);
  const acc0 = f.local(F32);
  const acc1 = f.local(F32);
  const dxS = f.local(F32);

  // Strictly 13 V128 locals -> 3 free XMM scratch registers (0 spills!)
  const mask03 = f.local(V128);
  const mask04 = f.local(V128);
  const sumi0 = f.local(V128);
  const sumi1 = f.local(V128);
  const hm0 = f.local(V128);
  const hm1 = f.local(V128);
  const q0 = f.local(V128);
  const q1 = f.local(V128);
  const qv = f.local(V128);
  const x0 = f.local(V128);
  const sc16V = f.local(V128);
  const scHalf = f.local(V128);
  const x1 = f.local(V128);

  f.i32Const(0x03).i8x16Splat().set(mask03);
  f.i32Const(0x04).i8x16Splat().set(mask04);
  f.get(3).i32Const(8).i32ShrS().set(nSuper);
  f.get(nSuper).i32Const(110).i32Mul().set(rowBytes);
  f.i32Const(0).set(r);

  // XMM-domain signed scale broadcast: 1 single VPSHUFB per sub-block (zero GPR->XMM stalls!)
  const emitScaleShuffleQ3 = (laneIdx: number) => {
    const lo = laneIdx * 2;
    const hi = lo + 1;
    f.get(scHalf).get(scHalf).i8x16Shuffle([lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi, lo, hi]);
  };

  // Unpack scales of `wpLoc`, initialize `sumiLoc = -4 * sum_j(sc[j] * bsums16[j])`, and run all 16 sub-blocks
  const emitRowQ3 = (wpLoc: number, sumiLoc: number) => {
    f.get(wpLoc).i32Load(96).set(a0);
    f.get(wpLoc).i32Load(100).set(a1);
    f.get(wpLoc).i32Load(104).set(tmp);
    f.get(a0).i32Const(0x0f0f0f0f).i32And()
      .get(tmp).i32Const(0x03030303).i32And().i32Const(4).i32Shl().i32Or().set(u0);
    f.get(a1).i32Const(0x0f0f0f0f).i32And()
      .get(tmp).i32Const(2).i32ShrU().i32Const(0x03030303).i32And().i32Const(4).i32Shl().i32Or().set(u1);
    f.get(a0).i32Const(4).i32ShrU().i32Const(0x0f0f0f0f).i32And()
      .get(tmp).i32Const(4).i32ShrU().i32Const(0x03030303).i32And().i32Const(4).i32Shl().i32Or().set(u2);
    f.get(a1).i32Const(4).i32ShrU().i32Const(0x0f0f0f0f).i32And()
      .get(tmp).i32Const(6).i32ShrU().i32Const(0x03030303).i32And().i32Const(4).i32Shl().i32Or().set(u3);

    // Pack signed i8 scales [-32..31] into sc16V (reused for both min-offset AND sub-block shuffles!)
    f.i32Const(0).i32x4Splat()
      .get(u0).i32x4ReplaceLane(0)
      .get(u1).i32x4ReplaceLane(1)
      .get(u2).i32x4ReplaceLane(2)
      .get(u3).i32x4ReplaceLane(3)
      .i32Const(32).i8x16Splat().i8x16Sub().set(sc16V);
    f.i32Const(0).i32x4Splat()
      .get(sc16V).i16x8ExtendLowI8x16S().get(qp).v128Load(16, 4).i32x4DotI16x8S()
      .get(sc16V).i16x8ExtendHighI8x16S().get(qp).v128Load(32, 4).i32x4DotI16x8S()
      .i32x4Add().i32Const(2).i32x4Shl()
      .i32x4Sub().set(sumiLoc);

    f.get(wpLoc).v128Load(0).set(hm0);
    f.get(wpLoc).v128Load(16).set(hm1);

    for (let h = 0; h < 2; h++) {
      if (h === 0) f.get(sc16V).i16x8ExtendLowI8x16S().set(scHalf);
      else f.get(sc16V).i16x8ExtendHighI8x16S().set(scHalf);
      f.get(wpLoc).v128Load(32 + h * 32).set(q0);
      f.get(wpLoc).v128Load(48 + h * 32).set(q1);
      for (let s = 0; s < 4; s++) {
        const shift = s * 2;
        const bitIdx = h * 4 + s;
        const xOff = 64 + h * 128 + s * 32;

        // First 16 elements
        f.get(qp).v128Load(xOff, 4).set(x0);
        f.get(q0);
        if (shift > 0) f.i32Const(shift).i8x16ShrU();
        if (s < 3) f.get(mask03).v128And();
        f.get(hm0);
        if (bitIdx < 2) f.i32Const(2 - bitIdx).i8x16Shl();
        else if (bitIdx > 2) f.i32Const(bitIdx - 2).i8x16ShrU();
        f.get(mask04).v128And().v128Or().set(qv);
        f.pushDotI8I7ToI16x8(x0, qv, useRelaxed);
        emitScaleShuffleQ3(s * 2);
        f.i32x4DotI16x8S().get(sumiLoc).i32x4Add().set(sumiLoc);

        // Next 16 elements
        f.get(qp).v128Load(xOff + 16, 4).set(x1);
        f.get(q1);
        if (shift > 0) f.i32Const(shift).i8x16ShrU();
        if (s < 3) f.get(mask03).v128And();
        f.get(hm1);
        if (bitIdx < 2) f.i32Const(2 - bitIdx).i8x16Shl();
        else if (bitIdx > 2) f.i32Const(bitIdx - 2).i8x16ShrU();
        f.get(mask04).v128And().v128Or().set(qv);
        f.pushDotI8I7ToI16x8(x1, qv, useRelaxed);
        emitScaleShuffleQ3(s * 2 + 1);
        f.i32x4DotI16x8S().get(sumiLoc).i32x4Add().set(sumiLoc);
      }
    }
  };

  // 2-row zero-spill main loop
  f.block().loop();
  f.get(r).i32Const(1).i32Add().get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().tee(wp0).get(rowBytes).i32Add().set(wp1);
  f.get(1).set(qp);
  f.f32Const(0).tee(acc0).set(acc1);
  f.i32Const(0).set(sb);

  f.block().loop();
  f.get(sb).get(nSuper).i32GeS().brIf(1);
  f.get(qp).f32Load(0).set(dxS);
  emitRowQ3(wp0, sumi0);
  emitRowQ3(wp1, sumi1);

  f.hsumI32x4(sumi0).f32ConvertI32S().loadF16AsF32(wp0, 108).get(dxS).f32Mul().f32Mul()
    .get(acc0).f32Add().set(acc0);
  f.hsumI32x4(sumi1).f32ConvertI32S().loadF16AsF32(wp1, 108).get(dxS).f32Mul().f32Mul()
    .get(acc1).f32Add().set(acc1);

  f.get(wp0).i32Const(110).i32Add().set(wp0);
  f.get(wp1).i32Const(110).i32Add().set(wp1);
  f.get(qp).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(sb).i32Const(1).i32Add().set(sb);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc0).f32Store(0);
  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc1).f32Store(4);
  f.get(r).i32Const(2).i32Add().set(r);
  f.br(0).end().end();

  // Odd trailing row (0..1)
  f.block();
  f.get(r).get(4).i32GeS().brIf(0);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(wp0);
  f.get(1).set(qp);
  f.f32Const(0).set(acc0);
  f.i32Const(0).set(sb);

  f.block().loop();
  f.get(sb).get(nSuper).i32GeS().brIf(1);
  f.get(qp).f32Load(0).set(dxS);
  emitRowQ3(wp0, sumi0);
  f.hsumI32x4(sumi0).f32ConvertI32S().loadF16AsF32(wp0, 108).get(dxS).f32Mul().f32Mul()
    .get(acc0).f32Add().set(acc0);

  f.get(wp0).i32Const(110).i32Add().set(wp0);
  f.get(qp).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp);
  f.get(sb).i32Const(1).i32Add().set(sb);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().get(acc0).f32Store(0);
  f.end();

  return f;
}

/**
 * Verdant/Prism 4-token batched prefill `Q3_K × Q8_K` Narrow-Lane Contraction:
 * `(wPtr, q8kBatchPtr, outBatchPtr, cols, rows, oStrideBytes)`.
 */
function buildMulMatQ3KQ8KBatch4(useRelaxed: boolean): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32, I32]);
  const r = f.local(I32);
  const nSuper = f.local(I32);
  const rowBytes = f.local(I32);
  const qStride = f.local(I32);
  const oStride = f.local(I32);
  const wp = f.local(I32);
  const qp0 = f.local(I32);
  const qp1 = f.local(I32);
  const qp2 = f.local(I32);
  const qp3 = f.local(I32);
  const op = f.local(I32);
  const sb = f.local(I32);
  const a0 = f.local(I32);
  const a1 = f.local(I32);
  const tmp = f.local(I32);
  const u0 = f.local(I32);
  const u1 = f.local(I32);
  const u2 = f.local(I32);
  const u3 = f.local(I32);

  const mask03 = f.local(V128);
  const mask04 = f.local(V128);
  const acc4 = f.local(V128);
  const sumi0 = f.local(V128);
  const sumi1 = f.local(V128);
  const sumi2 = f.local(V128);
  const sumi3 = f.local(V128);
  const hm0 = f.local(V128);
  const hm1 = f.local(V128);
  const q0 = f.local(V128);
  const q1 = f.local(V128);
  const qv = f.local(V128);
  const scV = f.local(V128);
  const xtmp = f.local(V128);
  const dotV = f.local(V128);

  f.i32Const(0x03).i8x16Splat().set(mask03);
  f.i32Const(0x04).i8x16Splat().set(mask04);
  f.get(3).i32Const(8).i32ShrS().set(nSuper);
  f.get(nSuper).i32Const(110).i32Mul().set(rowBytes);
  f.get(nSuper).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Mul().set(qStride);
  f.get(5).set(oStride);
  f.i32Const(0).set(r);

  const initTokOffset = (qpLoc: number, sumiLoc: number) => {
    f.i32Const(0).i32x4Splat()
      .get(q0).get(qpLoc).v128Load(16, 4).i32x4DotI16x8S()
      .get(q1).get(qpLoc).v128Load(32, 4).i32x4DotI16x8S()
      .i32x4Add().i32Const(2).i32x4Shl()
      .i32x4Sub().set(sumiLoc);
  };

  const setScaleSplat16 = (scIdx: number) => {
    const uLoc = scIdx < 4 ? u0 : scIdx < 8 ? u1 : scIdx < 12 ? u2 : u3;
    const byteIdx = scIdx & 3;
    f.get(uLoc);
    if (byteIdx > 0) f.i32Const(byteIdx * 8).i32ShrU();
    if (byteIdx < 3) f.i32Const(0xff).i32And();
    f.i32Const(32).i32Sub().i16x8Splat().set(scV);
  };

  const emitTokQ3 = (qpLoc: number, xOff: number, sumiLoc: number) => {
    f.get(qpLoc).v128Load(xOff, 4).set(xtmp);
    f.pushDotI8I7ToI16x8(xtmp, qv, useRelaxed);
    f.get(scV).i32x4DotI16x8S().get(sumiLoc).i32x4Add().set(sumiLoc);
  };

  f.block().loop();
  f.get(r).get(4).i32GeS().brIf(1);
  f.get(0).get(r).get(rowBytes).i32Mul().i32Add().set(wp);
  f.get(1).tee(qp0)
    .get(qStride).i32Add().tee(qp1)
    .get(qStride).i32Add().tee(qp2)
    .get(qStride).i32Add().set(qp3);
  f.f32Const(0).f32x4Splat().set(acc4);
  f.i32Const(0).set(sb);

  f.block().loop();
  f.get(sb).get(nSuper).i32GeS().brIf(1);

  f.get(wp).i32Load(96).set(a0);
  f.get(wp).i32Load(100).set(a1);
  f.get(wp).i32Load(104).set(tmp);
  f.get(a0).i32Const(0x0f0f0f0f).i32And()
    .get(tmp).i32Const(0x03030303).i32And().i32Const(4).i32Shl().i32Or().set(u0);
  f.get(a1).i32Const(0x0f0f0f0f).i32And()
    .get(tmp).i32Const(2).i32ShrU().i32Const(0x03030303).i32And().i32Const(4).i32Shl().i32Or().set(u1);
  f.get(a0).i32Const(4).i32ShrU().i32Const(0x0f0f0f0f).i32And()
    .get(tmp).i32Const(4).i32ShrU().i32Const(0x03030303).i32And().i32Const(4).i32Shl().i32Or().set(u2);
  f.get(a1).i32Const(4).i32ShrU().i32Const(0x0f0f0f0f).i32And()
    .get(tmp).i32Const(6).i32ShrU().i32Const(0x03030303).i32And().i32Const(4).i32Shl().i32Or().set(u3);

  f.i32Const(0).i32x4Splat()
    .get(u0).i32x4ReplaceLane(0)
    .get(u1).i32x4ReplaceLane(1)
    .get(u2).i32x4ReplaceLane(2)
    .get(u3).i32x4ReplaceLane(3)
    .i32Const(32).i8x16Splat().i8x16Sub().set(qv);
  f.get(qv).i16x8ExtendLowI8x16S().set(q0);
  f.get(qv).i16x8ExtendHighI8x16S().set(q1);
  initTokOffset(qp0, sumi0);
  initTokOffset(qp1, sumi1);
  initTokOffset(qp2, sumi2);
  initTokOffset(qp3, sumi3);

  f.get(wp).v128Load(0).set(hm0);
  f.get(wp).v128Load(16).set(hm1);

  for (let h = 0; h < 2; h++) {
    f.get(wp).v128Load(32 + h * 32).set(q0);
    f.get(wp).v128Load(48 + h * 32).set(q1);
    for (let s = 0; s < 4; s++) {
      const shift = s * 2;
      const bitIdx = h * 4 + s;
      const xOff = 64 + h * 128 + s * 32;
      const scIdx0 = h * 8 + s * 2;

      f.get(q0);
      if (shift > 0) f.i32Const(shift).i8x16ShrU();
      f.get(mask03).v128And();
      f.get(hm0);
      if (bitIdx < 2) f.i32Const(2 - bitIdx).i8x16Shl();
      else if (bitIdx > 2) f.i32Const(bitIdx - 2).i8x16ShrU();
      f.get(mask04).v128And().v128Or().set(qv);
      setScaleSplat16(scIdx0);
      emitTokQ3(qp0, xOff, sumi0);
      emitTokQ3(qp1, xOff, sumi1);
      emitTokQ3(qp2, xOff, sumi2);
      emitTokQ3(qp3, xOff, sumi3);

      f.get(q1);
      if (shift > 0) f.i32Const(shift).i8x16ShrU();
      f.get(mask03).v128And();
      f.get(hm1);
      if (bitIdx < 2) f.i32Const(2 - bitIdx).i8x16Shl();
      else if (bitIdx > 2) f.i32Const(bitIdx - 2).i8x16ShrU();
      f.get(mask04).v128And().v128Or().set(qv);
      setScaleSplat16(scIdx0 + 1);
      emitTokQ3(qp0, xOff + 16, sumi0);
      emitTokQ3(qp1, xOff + 16, sumi1);
      emitTokQ3(qp2, xOff + 16, sumi2);
      emitTokQ3(qp3, xOff + 16, sumi3);
    }
  }

  f.hsum4RowsI32x4(sumi0, sumi1, sumi2, sumi3, q0, q1, dotV);
  f.f32Const(0).f32x4Splat()
    .get(qp0).f32Load(0).f32x4ReplaceLane(0)
    .get(qp1).f32Load(0).f32x4ReplaceLane(1)
    .get(qp2).f32Load(0).f32x4ReplaceLane(2)
    .get(qp3).f32Load(0).f32x4ReplaceLane(3)
    .loadF16AsF32(wp, 108).f32x4Splat().f32x4Mul().set(q0);
  if (useRelaxed) {
    f.get(dotV).f32x4ConvertI32x4S().get(q0).get(acc4).f32x4RelaxedMadd().set(acc4);
  } else {
    f.get(dotV).f32x4ConvertI32x4S().get(q0).f32x4Mul().get(acc4).f32x4Add().set(acc4);
  }

  f.get(wp).i32Const(110).i32Add().set(wp);
  f.get(qp0).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp0);
  f.get(qp1).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp1);
  f.get(qp2).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp2);
  f.get(qp3).i32Const(Q8_K_ACT_BLOCK_BYTES).i32Add().set(qp3);
  f.get(sb).i32Const(1).i32Add().set(sb);
  f.br(0).end().end();

  f.get(2).get(r).i32Const(2).i32Shl().i32Add().tee(op).get(acc4).f32x4Extract(0).f32Store(0);
  f.get(op).get(oStride).i32Add().tee(op).get(acc4).f32x4Extract(1).f32Store(0);
  f.get(op).get(oStride).i32Add().tee(op).get(acc4).f32x4Extract(2).f32Store(0);
  f.get(op).get(oStride).i32Add().get(acc4).f32x4Extract(3).f32Store(0);

  f.get(r).i32Const(1).i32Add().set(r);
  f.br(0).end().end();

  return f;
}

/**
 * Exact WASM SIMD port of `UOR-Foundation/uor-matmul/crates/uor-matmul-kernels/src/isa/wasm.rs::simd_build_v4`:
 * `(space, block, bookPtr, actsPtr, outPtr)` where `rows = 16` (`V = 4` v128 registers of `i32x4`).
 * Precomputes the L1-resident partial-sum codebook table for every code `c ∈ [0..space)` across
 * `block` consecutive weight positions and `16` activation rows simultaneously.
 */
function buildUorTableBuildI8V4(): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const c = f.local(I32);
  const p = f.local(I32);
  const bp = f.local(I32);
  const ap = f.local(I32);
  const op = f.local(I32);
  const e0 = f.local(V128);
  const e1 = f.local(V128);
  const e2 = f.local(V128);
  const e3 = f.local(V128);
  const zero = f.local(V128);
  const w = f.local(V128);
  const xv = f.local(V128);
  const lo = f.local(V128);
  const hi = f.local(V128);

  f.i32Const(0).i32x4Splat().set(zero);
  f.i32Const(0).set(c);
  f.get(2).set(bp); // bookPtr (code-major: space x block i8)
  f.get(4).set(op); // outPtr (space x 16 i32 = space * 64 bytes)

  f.block().loop();
  f.get(c).get(0).i32GeS().brIf(1);
  f.get(zero).tee(e0).tee(e1).tee(e2).set(e3);
  f.get(3).set(ap); // actsPtr (block x 16 i8 = block * 16 bytes)
  f.i32Const(0).set(p);

  f.block().loop();
  f.get(p).get(1).i32GeS().brIf(1);
  f.get(bp).get(p).i32Add().i32Load8S(0).i16x8Splat().set(w);
  f.get(ap).v128Load(0).set(xv);
  f.get(xv).i16x8ExtendLowI8x16S().set(lo);
  f.get(xv).i16x8ExtendHighI8x16S().set(hi);
  f.get(e0).get(lo).get(w).i32x4ExtmulLowI16x8S().i32x4Add().set(e0);
  f.get(e1).get(lo).get(w).i32x4ExtmulHighI16x8S().i32x4Add().set(e1);
  f.get(e2).get(hi).get(w).i32x4ExtmulLowI16x8S().i32x4Add().set(e2);
  f.get(e3).get(hi).get(w).i32x4ExtmulHighI16x8S().i32x4Add().set(e3);
  f.get(ap).i32Const(16).i32Add().set(ap);
  f.get(p).i32Const(1).i32Add().set(p);
  f.br(0).end().end();

  f.get(op).get(e0).v128Store(0);
  f.get(op).get(e1).v128Store(16);
  f.get(op).get(e2).v128Store(32);
  f.get(op).get(e3).v128Store(48);
  f.get(bp).get(1).i32Add().set(bp);
  f.get(op).i32Const(64).i32Add().set(op);
  f.get(c).i32Const(1).i32Add().set(c);
  f.br(0).end().end();

  return f;
}

/**
 * Exact WASM SIMD port of `UOR-Foundation/uor-matmul/crates/uor-matmul-kernels/src/isa/wasm.rs::simd_codes_v4_u1_u8`:
 * `(depth, slabWords, stackPtr, codesPtr, lanePtr)` (`rows = 16`, `group = 1`, `u8` codes).
 * Zero-multiply, zero-branch masked gather accumulation directly over coded weight bytes!
 */
function buildUorTableGatherU8V4(): FuncBuilder {
  const f = new FuncBuilder([I32, I32, I32, I32, I32]);
  const slot = f.local(I32);
  const mask = f.local(I32);
  const slabBytes = f.local(I32);
  const words = f.local(I32);
  const cp = f.local(I32);
  const entry = f.local(I32);
  const a0 = f.local(V128);
  const a1 = f.local(V128);
  const a2 = f.local(V128);
  const a3 = f.local(V128);

  // mask = (slabWords >> 4) - 1 (each entry = 16 i32 words = 64 bytes = 1 << 6)
  f.get(1).i32Const(4).i32ShrU().i32Const(1).i32Sub().set(mask);
  f.get(1).i32Const(2).i32Shl().set(slabBytes);
  f.get(2).set(words);
  f.get(3).set(cp);

  f.get(4).v128Load(0).set(a0);
  f.get(4).v128Load(16).set(a1);
  f.get(4).v128Load(32).set(a2);
  f.get(4).v128Load(48).set(a3);

  f.i32Const(0).set(slot);
  f.block().loop();
  f.get(slot).get(0).i32GeS().brIf(1);
  // entry = words + ((code & mask) << 6) — zero comparison, zero branch!
  f.get(words)
    .get(cp).i32Load8U(0).get(mask).i32And().i32Const(6).i32Shl()
    .i32Add().set(entry);
  f.get(a0).get(entry).v128Load(0).i32x4Add().set(a0);
  f.get(a1).get(entry).v128Load(16).i32x4Add().set(a1);
  f.get(a2).get(entry).v128Load(32).i32x4Add().set(a2);
  f.get(a3).get(entry).v128Load(48).i32x4Add().set(a3);
  f.get(cp).i32Const(1).i32Add().set(cp);
  f.get(words).get(slabBytes).i32Add().set(words);
  f.get(slot).i32Const(1).i32Add().set(slot);
  f.br(0).end().end();

  f.get(4).get(a0).v128Store(0);
  f.get(4).get(a1).v128Store(16);
  f.get(4).get(a2).v128Store(32);
  f.get(4).get(a3).v128Store(48);
  return f;
}

export function buildVerdantSimdModuleBytes(
  useRelaxed: boolean,
  sharedMemory = false,
): Uint8Array<ArrayBuffer> {
  const funcs: Array<{ name: string; builder: FuncBuilder }> = [];
  const add = (name: string, builder: FuncBuilder): number => {
    const idx = funcs.length;
    funcs.push({ name, builder });
    return idx;
  };

  const precomputeIdx = add('precompute_block_sum_x', buildPrecomputeBlockSumX());
  // Strict F32-activation F64-accumulated kernels (all 11 GGML formats, 4-row tiled F32)
  add('mul_mat_f32', buildMulMatF32(useRelaxed));
  add('mul_mat_f16', buildMulMatF16());
  add('mul_mat_bf16', buildMulMatBF16());
  add('mul_mat_q8_0', buildMulMatQ8_0());
  add('mul_mat_q4_0', buildMulMatQ4_0(precomputeIdx));
  add('mul_mat_q4_0_batch4', buildMulMatQ4_0Batch4(precomputeIdx));
  add('mul_mat_q4_1', buildMulMatQ4_1(precomputeIdx));
  add('mul_mat_q5_0', buildMulMatQ5_0(precomputeIdx));
  add('mul_mat_q5_1', buildMulMatQ5_1(precomputeIdx));
  add('mul_mat_q2_k', buildMulMatQ2_K());
  add('mul_mat_q3_k', buildMulMatQ3_K());
  add('mul_mat_q4_k', buildMulMatQ4_K(precomputeIdx));
  add('mul_mat_q5_k', buildMulMatQ5_K(precomputeIdx));
  add('mul_mat_q6_k', buildMulMatQ6_K());

  // High-throughput Q8_0-activation integer dot-product kernels (Relaxed VNNI / SIMD128 Int8)
  add('quantize_f32_to_q8_0', buildQuantizeF32ToQ8_0());
  add('mul_mat_q4_0_q8', buildMulMatQ4_0Q8(useRelaxed));
  add('mul_mat_q4_0_q8_batch4', buildMulMatQ4_0Q8Batch4(useRelaxed));
  add('mul_mat_q4_1_q8', buildMulMatQ4_1Q8(useRelaxed));
  add('mul_mat_q5_0_q8', buildMulMatQ5Q8(false, useRelaxed));
  add('mul_mat_q5_1_q8', buildMulMatQ5Q8(true, useRelaxed));
  add('mul_mat_q8_0_q8', buildMulMatQ8_0Q8(useRelaxed));
  add('mul_mat_q8_0_q8_batch4', buildMulMatQ8_0Q8Batch4(useRelaxed));
  add('mul_mat_q4_k_q8', buildMulMatQ4KQ8(useRelaxed));
  add('mul_mat_q4_k_q8_batch4', buildMulMatQ4KQ8Batch4(useRelaxed));
  add('mul_mat_q5_k_q8', buildMulMatQ5KQ8(useRelaxed));
  add('mul_mat_q6_k_q8', buildMulMatQ6KQ8(useRelaxed));
  add('mul_mat_q6_k_q8_batch4', buildMulMatQ6KQ8Batch4(useRelaxed));
  add('mul_mat_f32_batch4', buildMulMatF32Batch4(useRelaxed));

  // Verdant/Prism Q8_K super-block narrow-lane integer contraction ("encode once per 256 weights")
  // + uor-matmul tabulated zero-multiply TableBuild / TableGatherCodesU8 kernels
  add('quantize_f32_to_q8_k', buildQuantizeF32ToQ8K());
  add('mul_mat_q2_k_q8k', buildMulMatQ2KQ8K(useRelaxed));
  add('mul_mat_q2_k_q8k_batch4', buildMulMatQ2KQ8KBatch4(useRelaxed));
  add('mul_mat_q3_k_q8k', buildMulMatQ3KQ8K(useRelaxed));
  add('mul_mat_q3_k_q8k_batch4', buildMulMatQ3KQ8KBatch4(useRelaxed));
  add('mul_mat_q4_k_q8k', buildMulMatQ4KQ8K(useRelaxed));
  add('mul_mat_q4_k_q8k_batch4', buildMulMatQ4KQ8KBatch4(useRelaxed));
  add('mul_mat_q5_k_q8k', buildMulMatQ5KQ8K(useRelaxed));
  add('mul_mat_q5_k_q8k_batch4', buildMulMatQ5KQ8KBatch4(useRelaxed));
  add('mul_mat_q6_k_q8k', buildMulMatQ6KQ8K(useRelaxed));
  add('mul_mat_q6_k_q8k_batch4', buildMulMatQ6KQ8KBatch4(useRelaxed));
  add('uor_table_build_i8_v4', buildUorTableBuildI8V4());
  add('uor_table_gather_u8_v4', buildUorTableGatherU8V4());

  // Normalization, activations, RoPE, bulk copy/zero & 3-tier vRAM attention + KV quantization primitives
  const rmsNormIdx = add('rms_norm_f32', buildRmsNormF32());
  add('rms_norm_per_head_f32', buildRmsNormPerHeadF32(rmsNormIdx));
  add('add_f32', buildAddF32());
  add('swiglu_f32', buildSwigluF32(useRelaxed));
  add('copy_bytes', buildCopyBytes());
  add('zero_bytes', buildZeroBytes());
  add('quantize_kv_page_f32_to_q8', buildQuantizeKvPageF32ToQ8());
  add('decode_row_f16', buildDecodeRowF16());
  add('rope_f32', buildRopeF32());
  const fastExpIdx = add('fast_exp_f32', buildFastExpF32());
  add('dot_f32', buildDotF32());
  const dotQ8Idx = add('dot_q8_f32', buildDotQ8F32());
  const dotQ4Idx = add('dot_q4_f32', buildDotQ4F32());
  add('axpy_f32', buildAxpyF32());
  const axpyQ8Idx = add('axpy_q8_f32', buildAxpyQ8F32());
  const axpyQ4Idx = add('axpy_q4_f32', buildAxpyQ4F32());
  add('scale_add_f32', buildScaleAddF32());
  const scaleIdx = add('scale_f32', buildScaleF32());

  // Fused 16-token page-level attention kernels & 100% in-Verdant multi-core FlashAttention
  const scoresF32Idx = add('attn_page_scores_f32', buildAttnPageScoresF32(useRelaxed));
  const accumF32Idx = add('attn_page_accum_f32', buildAttnPageAccumF32(useRelaxed));
  const scoresQ8Idx = add('attn_page_scores_q8', buildAttnPageScoresQ8(dotQ8Idx));
  const accumQ8Idx = add('attn_page_accum_q8', buildAttnPageAccumQ8(scaleIdx, axpyQ8Idx));
  const scoresQ4Idx = add('attn_page_scores_q4', buildAttnPageScoresQ4(dotQ4Idx));
  const accumQ4Idx = add('attn_page_accum_q4', buildAttnPageAccumQ4(scaleIdx, axpyQ4Idx));
  const flashAttnRangeIdx = add(
    'flash_attn_layer_range',
    buildFlashAttnLayerRange(
      fastExpIdx,
      scaleIdx,
      scoresF32Idx,
      accumF32Idx,
      scoresQ8Idx,
      accumQ8Idx,
      scoresQ4Idx,
      accumQ4Idx,
    ),
  );
  add('flash_attn_layer', buildFlashAttnLayer(flashAttnRangeIdx));
  add('argmax_f32', buildArgmaxF32());
  add('topk_f32', buildTopkF32());

  // Type section (deduplicated signatures)
  const sigKey = (p: number[], r: number[]) => `${p.join(',')}|${r.join(',')}`;
  const sigMap = new Map<string, number>();
  const typeEntries: number[] = [];
  const funcTypeIndices: number[] = [];

  for (const { builder } of funcs) {
    const key = sigKey(builder.params, builder.results);
    let idx = sigMap.get(key);
    if (idx === undefined) {
      idx = sigMap.size;
      sigMap.set(key, idx);
      typeEntries.push(
        0x60,
        ...uleb(builder.params.length),
        ...builder.params,
        ...uleb(builder.results.length),
        ...builder.results,
      );
    }
    funcTypeIndices.push(idx);
  }

  // Import section: env.memory (min 16 pages = 1 MiB, max 65536 pages = 4 GiB WASM32 architectural ceiling; 0x03 = shared)
  const importBody = [
    0x01,
    ...utf8('env'),
    ...utf8('memory'),
    0x02,
    sharedMemory ? 0x03 : 0x01,
    ...uleb(16),
    ...uleb(65536),
  ];

  // Function section
  const funcSectionBody = [...uleb(funcs.length), ...funcTypeIndices.flatMap((i) => uleb(i))];

  // Export section
  const exportEntries: number[] = [...uleb(funcs.length)];
  funcs.forEach(({ name }, idx) => {
    exportEntries.push(...utf8(name), 0x00, ...uleb(idx));
  });

  // Code section
  const codeSectionBody: number[] = [...uleb(funcs.length)];
  for (const { builder } of funcs) {
    codeSectionBody.push(...builder.encodeBody());
  }

  return new Uint8Array([
    0x00, 0x61, 0x73, 0x6d,
    0x01, 0x00, 0x00, 0x00,
    ...section(1, [...uleb(sigMap.size), ...typeEntries]),
    ...section(2, importBody),
    ...section(3, funcSectionBody),
    ...section(7, exportEntries),
    ...section(10, codeSectionBody),
  ]);
}

export interface VerdantSimdExports {
  mul_mat_f32(wPtr: number, xPtr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_f16(wPtr: number, xPtr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_bf16(wPtr: number, xPtr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_q8_0(wPtr: number, xPtr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_q4_0(wPtr: number, xPtr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_q4_0_batch4(wPtr: number, xBatchPtr: number, outBatchPtr: number, cols: number, rows: number, oStrideBytes: number): void;
  mul_mat_q4_1(wPtr: number, xPtr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_q5_0(wPtr: number, xPtr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_q5_1(wPtr: number, xPtr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_q2_k(wPtr: number, xPtr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_q3_k(wPtr: number, xPtr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_q4_k(wPtr: number, xPtr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_q5_k(wPtr: number, xPtr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_q6_k(wPtr: number, xPtr: number, outPtr: number, cols: number, rows: number): void;
  quantize_f32_to_q8_0(xPtr: number, q8Ptr: number, cols: number): void;
  mul_mat_q4_0_q8(wPtr: number, q8Ptr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_q4_0_q8_batch4(wPtr: number, q8BatchPtr: number, outBatchPtr: number, cols: number, rows: number, oStrideBytes: number): void;
  mul_mat_q4_1_q8(wPtr: number, q8Ptr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_q5_0_q8(wPtr: number, q8Ptr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_q5_1_q8(wPtr: number, q8Ptr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_q8_0_q8(wPtr: number, q8Ptr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_q8_0_q8_batch4(wPtr: number, q8BatchPtr: number, outBatchPtr: number, cols: number, rows: number, oStrideBytes: number): void;
  mul_mat_q4_k_q8(wPtr: number, q8Ptr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_q4_k_q8_batch4(wPtr: number, q8BatchPtr: number, outBatchPtr: number, cols: number, rows: number, oStrideBytes: number): void;
  mul_mat_q5_k_q8(wPtr: number, q8Ptr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_q6_k_q8(wPtr: number, q8Ptr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_q6_k_q8_batch4(wPtr: number, q8BatchPtr: number, outBatchPtr: number, cols: number, rows: number, oStrideBytes: number): void;
  mul_mat_f32_batch4(wPtr: number, xBatchPtr: number, outBatchPtr: number, cols: number, rows: number, oStrideBytes: number): void;
  rms_norm_f32(xPtr: number, wPtr: number, outPtr: number, n: number, eps: number): void;
  rms_norm_per_head_f32(vecPtr: number, wPtr: number, nHeads: number, headDim: number, eps: number): void;
  add_f32(aPtr: number, bPtr: number, outPtr: number, n: number): void;
  swiglu_f32(gatePtr: number, upPtr: number, outPtr: number, n: number): void;
  copy_bytes(dstPtr: number, srcPtr: number, byteLen: number): void;
  zero_bytes(dstPtr: number, byteLen: number): void;
  quantize_kv_page_f32_to_q8(
    srcF32Ptr: number,
    dstQ8Ptr: number,
    dstScalePtr: number,
    totalVectors: number,
    headDim: number,
    errOutPtr: number,
  ): void;
  decode_row_f16(srcPtr: number, dstPtr: number, n: number): void;
  rope_f32(vecPtr: number, cosPtr: number, sinPtr: number, nHeads: number, headDim: number, isNeox: number): void;
  fast_exp_f32(x: number): number;
  dot_f32(aPtr: number, bPtr: number, n: number): number;
  dot_q8_f32(q8Ptr: number, xPtr: number, n: number): number;
  dot_q4_f32(q4Ptr: number, xPtr: number, n: number): number;
  axpy_f32(dstPtr: number, srcPtr: number, n: number, scale: number): void;
  axpy_q8_f32(dstPtr: number, q8Ptr: number, n: number, scale: number): void;
  axpy_q4_f32(dstPtr: number, q4Ptr: number, n: number, scale: number): void;
  scale_add_f32(dstPtr: number, srcPtr: number, n: number, rescale: number, addScale: number): void;
  scale_f32(dstPtr: number, n: number, scale: number): void;
  attn_page_scores_f32(qPtr: number, kBasePtr: number, scoresPtr: number, headDim: number, kvStrideBytes: number, count: number, invScale: number): void;
  attn_page_accum_f32(outPtr: number, vBasePtr: number, weightsPtr: number, headDim: number, kvStrideBytes: number, count: number, rescale: number): void;
  attn_page_scores_q8(qPtr: number, kQ8BasePtr: number, kScBasePtr: number, scoresPtr: number, headDim: number, kvStrideBytes: number, scStrideBytes: number, count: number, invScale: number): void;
  attn_page_accum_q8(outPtr: number, vQ8BasePtr: number, weightsPtr: number, headDim: number, kvStrideBytes: number, count: number, rescale: number): void;
  attn_page_scores_q4(qPtr: number, kQ4BasePtr: number, kScBasePtr: number, scoresPtr: number, headDim: number, kvStrideBytes: number, scStrideBytes: number, count: number, invScale: number): void;
  attn_page_accum_q4(outPtr: number, vQ4BasePtr: number, weightsPtr: number, headDim: number, kvStrideBytes: number, count: number, rescale: number): void;
  flash_attn_layer_range(
    qPtr: number,
    outPtr: number,
    pageStatePtr: number,
    pageSlotPtr: number,
    hotKLayerPtr: number,
    hotVLayerPtr: number,
    q8KLayerPtr: number,
    q8VLayerPtr: number,
    q8KScLayerPtr: number,
    q8VScLayerPtr: number,
    q4KLayerPtr: number,
    q4VLayerPtr: number,
    q4KScLayerPtr: number,
    q4VScLayerPtr: number,
    nHeads: number,
    nHeadsKv: number,
    headDim: number,
    seqLen: number,
    hotPageBytes: number,
    q8PageBytes: number,
    q4PageBytes: number,
    scPageBytes: number,
    hotKvStrideBytes: number,
    q8KvStrideBytes: number,
    q4KvStrideBytes: number,
    scStrideBytes: number,
    hStart: number,
    hEnd: number,
    scratchBase: number,
  ): void;
  flash_attn_layer(
    qPtr: number,
    outPtr: number,
    pageStatePtr: number,
    pageSlotPtr: number,
    hotKLayerPtr: number,
    hotVLayerPtr: number,
    q8KLayerPtr: number,
    q8VLayerPtr: number,
    q8KScLayerPtr: number,
    q8VScLayerPtr: number,
    q4KLayerPtr: number,
    q4VLayerPtr: number,
    q4KScLayerPtr: number,
    q4VScLayerPtr: number,
    nHeads: number,
    nHeadsKv: number,
    headDim: number,
    seqLen: number,
    hotPageBytes: number,
    q8PageBytes: number,
    q4PageBytes: number,
    scPageBytes: number,
    hotKvStrideBytes: number,
    q8KvStrideBytes: number,
    q4KvStrideBytes: number,
    scStrideBytes: number,
  ): void;
  argmax_f32(logitsPtr: number, n: number): number;
  topk_f32(logitsPtr: number, n: number, k: number, outIdsPtr: number, outValsPtr: number): number;
  quantize_f32_to_q8_k(xPtr: number, q8kPtr: number, cols: number): void;
  mul_mat_q2_k_q8k(wPtr: number, q8kPtr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_q2_k_q8k_batch4(wPtr: number, q8kBatchPtr: number, outBatchPtr: number, cols: number, rows: number, oStrideBytes: number): void;
  mul_mat_q3_k_q8k(wPtr: number, q8kPtr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_q3_k_q8k_batch4(wPtr: number, q8kBatchPtr: number, outBatchPtr: number, cols: number, rows: number, oStrideBytes: number): void;
  mul_mat_q4_k_q8k(wPtr: number, q8kPtr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_q4_k_q8k_batch4(wPtr: number, q8kBatchPtr: number, outBatchPtr: number, cols: number, rows: number, oStrideBytes: number): void;
  mul_mat_q5_k_q8k(wPtr: number, q8kPtr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_q5_k_q8k_batch4(wPtr: number, q8kBatchPtr: number, outBatchPtr: number, cols: number, rows: number, oStrideBytes: number): void;
  mul_mat_q6_k_q8k(wPtr: number, q8kPtr: number, outPtr: number, cols: number, rows: number): void;
  mul_mat_q6_k_q8k_batch4(wPtr: number, q8kBatchPtr: number, outBatchPtr: number, cols: number, rows: number, oStrideBytes: number): void;
  uor_table_build_i8_v4(space: number, block: number, bookPtr: number, actsPtr: number, outPtr: number): void;
  uor_table_gather_u8_v4(depth: number, slabWords: number, stackPtr: number, codesPtr: number, lanePtr: number): void;
}

export const enum ParallelMatOp {
  MUL_MAT = 1,
  MUL_MAT_Q8 = 2,
  MUL_MAT_BATCH4 = 3,
  MUL_MAT_BATCH4_Q8 = 4,
  MUL_MAT_Q8K = 5,
  MUL_MAT_BATCH4_Q8K = 6,
}

export interface MatGroupEntry {
  readonly weightType: GgmlType;
  readonly wPtr: number;
  readonly outPtr: number;
  readonly rows: number;
  readonly oStrideBytes: number;
}

export interface MoeWorkerScratchLayout {
  readonly offGateOut: number;
  readonly offUpOut: number;
  readonly offHid: number;
  readonly offQ8Ffn: number;
  readonly offExpProj: number;
  readonly offAccum: number;
  readonly strideBytes: number;
}

export function computeMoeWorkerScratchLayout(
  dim: number,
  expFfnDim: number,
  batchSize = 1,
): MoeWorkerScratchLayout {
  const b = Math.max(1, batchSize | 0);
  const align64 = (n: number) => (n + 63) & ~63;
  const ffnBytes = align64(b * expFfnDim * 4);
  const dimBytes = align64(b * dim * 4);
  const q8PerTok = Math.max(
    Math.ceil(expFfnDim / 256) * Q8_K_ACT_BLOCK_BYTES,
    Math.ceil(expFfnDim / 32) * Q8_0_ACT_BLOCK_BYTES,
    64,
  );
  const q8Bytes = align64(b * q8PerTok);
  const offGateOut = 0;
  const offUpOut = offGateOut + ffnBytes;
  const offHid = offUpOut + ffnBytes;
  const offQ8Ffn = offHid + ffnBytes;
  const offExpProj = offQ8Ffn + q8Bytes;
  const offAccum = offExpProj + dimBytes;
  const strideBytes = offAccum + dimBytes;
  return { offGateOut, offUpOut, offHid, offQ8Ffn, offExpProj, offAccum, strideBytes };
}

export interface MoeExpertParallelDispatch {
  readonly expertCount: number;
  readonly dim: number;
  readonly expFfnDim: number;
  readonly gateUpOp: ParallelMatOp;
  readonly downOp: ParallelMatOp;
  readonly gateType: GgmlType;
  readonly upType: GgmlType;
  readonly downType: GgmlType;
  readonly xOrQ8Ptr: number;
  readonly xF32Ptr: number;
  readonly outPtr: number;
  readonly workerScratchBase: number;
  readonly layout: MoeWorkerScratchLayout;
  readonly gatePtrs: Int32Array;
  readonly upPtrs: Int32Array;
  readonly downPtrs: Int32Array;
  readonly weights: Float32Array;
  /** When true, adds the reduced result onto existing `outPtr` contents instead of overwriting. */
  readonly accumulate?: boolean;
}

export interface MoeExpertBatch4ParallelDispatch {
  readonly uniqueCount: number;
  readonly dim: number;
  readonly expFfnDim: number;
  readonly gateUpBatchOp: ParallelMatOp;
  readonly gateUpSingleOp: ParallelMatOp;
  readonly downBatchOp: ParallelMatOp;
  readonly downSingleOp: ParallelMatOp;
  readonly gateType: GgmlType;
  readonly upType: GgmlType;
  readonly downType: GgmlType;
  readonly bXOrQ8Ptr: number;
  readonly bXF32Ptr: number;
  readonly xQ8StrideBytes: number;
  readonly outPtr: number;
  readonly workerScratchBase: number;
  readonly layout: MoeWorkerScratchLayout;
  readonly gatePtrs: Int32Array;
  readonly upPtrs: Int32Array;
  readonly downPtrs: Int32Array;
  readonly masks: Uint8Array | Int32Array;
  readonly weights: Float32Array;
  /** When true, adds the reduced result onto existing `outPtr` contents instead of overwriting. */
  readonly accumulate?: boolean;
}

export interface VTensorThreadPoolLike {
  readonly isActive: boolean;
  readonly workerCount: number;
  readonly parallelThresholdBytes: number;
  readonly parallelDispatches: number;
  readonly barrierLatencyUs: number;
  dispatchParallel(
    op: ParallelMatOp,
    weightType: GgmlType,
    wPtr: number,
    xOrQ8Ptr: number,
    xF32Ptr: number,
    outPtr: number,
    cols: number,
    rows: number,
    oStrideBytes: number,
    runLocalStripe: (rowStart: number, rowCount: number) => void,
  ): void;
  dispatchParallelGroup(
    op: ParallelMatOp,
    entries: readonly MatGroupEntry[],
    xOrQ8Ptr: number,
    xF32Ptr: number,
    cols: number,
    runLocalEntry: (entryIdx: number, rowStart: number, rowCount: number) => void,
  ): void;
  dispatchParallelMoeExperts(
    params: MoeExpertParallelDispatch,
    kernels: VerdantSimdExports,
  ): void;
  dispatchParallelMoeExpertsBatch4(
    params: MoeExpertBatch4ParallelDispatch,
    kernels: VerdantSimdExports,
  ): void;
  dispatchParallelFlashAttn(
    qPtr: number,
    outPtr: number,
    pageStatePtr: number,
    pageSlotPtr: number,
    hotKLayerPtr: number,
    hotVLayerPtr: number,
    q8KLayerPtr: number,
    q8VLayerPtr: number,
    q8KScLayerPtr: number,
    q8VScLayerPtr: number,
    q4KLayerPtr: number,
    q4VLayerPtr: number,
    q4KScLayerPtr: number,
    q4VScLayerPtr: number,
    nHeads: number,
    nHeadsKv: number,
    headDim: number,
    seqLen: number,
    hotPageBytes: number,
    q8PageBytes: number,
    q4PageBytes: number,
    scPageBytes: number,
    hotKvStrideBytes: number,
    q8KvStrideBytes: number,
    q4KvStrideBytes: number,
    scStrideBytes: number,
    runLocalHeads: (hStart: number, hEnd: number, scratchBase: number) => void,
  ): void;
}

export interface ArenaCalibrationResult {
  readonly relaxedSimdActive: boolean;
  readonly memoryBandwidthGBs: number;
  readonly computeGops: number;
  readonly ridgeIntensityOpsPerByte: number;
  readonly calibrationMs: number;
}

let cachedCompiledModule: WebAssembly.Module | null = null;
let cachedModuleByteLength = 0;
let cachedSharedCompiledModule: WebAssembly.Module | null = null;
let cachedSharedModuleByteLength = 0;
let cachedSimdSupport: boolean | null = null;
let cachedSharedWasmSupport: boolean | null = null;
let cachedRelaxedSimdSupport: boolean | null = null;

/** Computes exact byte length of one quantized or floating-point weight matrix row. */
export function weightRowByteSize(weightType: GgmlType, cols: number): number {
  switch (weightType) {
    case GgmlType.F32: return cols * 4;
    case GgmlType.F16:
    case GgmlType.BF16: return cols * 2;
    case GgmlType.Q4_0: return (cols >> 5) * 18;
    case GgmlType.Q4_1: return (cols >> 5) * 20;
    case GgmlType.Q5_0: return (cols >> 5) * 22;
    case GgmlType.Q5_1: return (cols >> 5) * 24;
    case GgmlType.Q8_0: return (cols >> 5) * 34;
    case GgmlType.Q2_K: return (cols >> 8) * 84;
    case GgmlType.Q3_K: return (cols >> 8) * 110;
    case GgmlType.Q4_K: return (cols >> 8) * 144;
    case GgmlType.Q5_K: return (cols >> 8) * 176;
    case GgmlType.Q6_K: return (cols >> 8) * 210;
    default: return cols * 4;
  }
}

/** Dispatches a single row-stripe directly onto a `VerdantSimdExports` instance (shared by coordinator & worker threads). */
export function executeWasmMatStripe(
  kernels: VerdantSimdExports,
  op: ParallelMatOp,
  weightType: GgmlType,
  wPtr: number,
  xOrQ8Ptr: number,
  xF32Ptr: number,
  outPtr: number,
  cols: number,
  rows: number,
  oStrideBytes: number,
): void {
  if (op === ParallelMatOp.MUL_MAT_BATCH4_Q8K) {
    switch (weightType) {
      case GgmlType.Q2_K:
        kernels.mul_mat_q2_k_q8k_batch4(wPtr, xOrQ8Ptr, outPtr, cols, rows, oStrideBytes);
        return;
      case GgmlType.Q3_K:
        kernels.mul_mat_q3_k_q8k_batch4(wPtr, xOrQ8Ptr, outPtr, cols, rows, oStrideBytes);
        return;
      case GgmlType.Q4_K:
        kernels.mul_mat_q4_k_q8k_batch4(wPtr, xOrQ8Ptr, outPtr, cols, rows, oStrideBytes);
        return;
      case GgmlType.Q5_K:
        kernels.mul_mat_q5_k_q8k_batch4(wPtr, xOrQ8Ptr, outPtr, cols, rows, oStrideBytes);
        return;
      case GgmlType.Q6_K:
        kernels.mul_mat_q6_k_q8k_batch4(wPtr, xOrQ8Ptr, outPtr, cols, rows, oStrideBytes);
        return;
      case GgmlType.F32:
        kernels.mul_mat_f32_batch4(wPtr, xF32Ptr, outPtr, cols, rows, oStrideBytes);
        return;
      default:
        break;
    }
  } else if (op === ParallelMatOp.MUL_MAT_Q8K) {
    switch (weightType) {
      case GgmlType.Q2_K:
        kernels.mul_mat_q2_k_q8k(wPtr, xOrQ8Ptr, outPtr, cols, rows);
        return;
      case GgmlType.Q3_K:
        kernels.mul_mat_q3_k_q8k(wPtr, xOrQ8Ptr, outPtr, cols, rows);
        return;
      case GgmlType.Q4_K:
        kernels.mul_mat_q4_k_q8k(wPtr, xOrQ8Ptr, outPtr, cols, rows);
        return;
      case GgmlType.Q5_K:
        kernels.mul_mat_q5_k_q8k(wPtr, xOrQ8Ptr, outPtr, cols, rows);
        return;
      case GgmlType.Q6_K:
        kernels.mul_mat_q6_k_q8k(wPtr, xOrQ8Ptr, outPtr, cols, rows);
        return;
      default:
        break;
    }
  } else if (op === ParallelMatOp.MUL_MAT_BATCH4_Q8) {
    switch (weightType) {
      case GgmlType.Q4_0:
        kernels.mul_mat_q4_0_q8_batch4(wPtr, xOrQ8Ptr, outPtr, cols, rows, oStrideBytes);
        return;
      case GgmlType.Q4_K:
        kernels.mul_mat_q4_k_q8_batch4(wPtr, xOrQ8Ptr, outPtr, cols, rows, oStrideBytes);
        return;
      case GgmlType.Q6_K:
        kernels.mul_mat_q6_k_q8_batch4(wPtr, xOrQ8Ptr, outPtr, cols, rows, oStrideBytes);
        return;
      case GgmlType.Q8_0:
        kernels.mul_mat_q8_0_q8_batch4(wPtr, xOrQ8Ptr, outPtr, cols, rows, oStrideBytes);
        return;
      case GgmlType.F32:
        kernels.mul_mat_f32_batch4(wPtr, xF32Ptr, outPtr, cols, rows, oStrideBytes);
        return;
      default:
        break;
    }
  } else if (op === ParallelMatOp.MUL_MAT_BATCH4) {
    if (weightType === GgmlType.F32) {
      kernels.mul_mat_f32_batch4(wPtr, xOrQ8Ptr, outPtr, cols, rows, oStrideBytes);
      return;
    }
    if (weightType === GgmlType.Q4_0) {
      kernels.mul_mat_q4_0_batch4(wPtr, xOrQ8Ptr, outPtr, cols, rows, oStrideBytes);
      return;
    }
  } else if (op === ParallelMatOp.MUL_MAT_Q8) {
    switch (weightType) {
      case GgmlType.Q4_0: kernels.mul_mat_q4_0_q8(wPtr, xOrQ8Ptr, outPtr, cols, rows); return;
      case GgmlType.Q4_1: kernels.mul_mat_q4_1_q8(wPtr, xOrQ8Ptr, outPtr, cols, rows); return;
      case GgmlType.Q5_0: kernels.mul_mat_q5_0_q8(wPtr, xOrQ8Ptr, outPtr, cols, rows); return;
      case GgmlType.Q5_1: kernels.mul_mat_q5_1_q8(wPtr, xOrQ8Ptr, outPtr, cols, rows); return;
      case GgmlType.Q8_0: kernels.mul_mat_q8_0_q8(wPtr, xOrQ8Ptr, outPtr, cols, rows); return;
      case GgmlType.Q4_K: kernels.mul_mat_q4_k_q8(wPtr, xOrQ8Ptr, outPtr, cols, rows); return;
      case GgmlType.Q5_K: kernels.mul_mat_q5_k_q8(wPtr, xOrQ8Ptr, outPtr, cols, rows); return;
      case GgmlType.Q6_K: kernels.mul_mat_q6_k_q8(wPtr, xOrQ8Ptr, outPtr, cols, rows); return;
      default:
        break;
    }
  }

  // Fallback / MUL_MAT (strict F32 activation)
  const xp =
    op === ParallelMatOp.MUL_MAT_Q8 || op === ParallelMatOp.MUL_MAT_Q8K ? xF32Ptr : xOrQ8Ptr;
  switch (weightType) {
    case GgmlType.F32: kernels.mul_mat_f32(wPtr, xp, outPtr, cols, rows); return;
    case GgmlType.F16: kernels.mul_mat_f16(wPtr, xp, outPtr, cols, rows); return;
    case GgmlType.BF16: kernels.mul_mat_bf16(wPtr, xp, outPtr, cols, rows); return;
    case GgmlType.Q8_0: kernels.mul_mat_q8_0(wPtr, xp, outPtr, cols, rows); return;
    case GgmlType.Q4_0: kernels.mul_mat_q4_0(wPtr, xp, outPtr, cols, rows); return;
    case GgmlType.Q4_1: kernels.mul_mat_q4_1(wPtr, xp, outPtr, cols, rows); return;
    case GgmlType.Q5_0: kernels.mul_mat_q5_0(wPtr, xp, outPtr, cols, rows); return;
    case GgmlType.Q5_1: kernels.mul_mat_q5_1(wPtr, xp, outPtr, cols, rows); return;
    case GgmlType.Q2_K: kernels.mul_mat_q2_k(wPtr, xp, outPtr, cols, rows); return;
    case GgmlType.Q3_K: kernels.mul_mat_q3_k(wPtr, xp, outPtr, cols, rows); return;
    case GgmlType.Q4_K: kernels.mul_mat_q4_k(wPtr, xp, outPtr, cols, rows); return;
    case GgmlType.Q5_K: kernels.mul_mat_q5_k(wPtr, xp, outPtr, cols, rows); return;
    case GgmlType.Q6_K: kernels.mul_mat_q6_k(wPtr, xp, outPtr, cols, rows); return;
    default:
      return;
  }
}

/** Probes whether WebAssembly Relaxed SIMD (hardware VNNI/SDOT + FMA) is supported. */
export function isVerdantRelaxedSimdSupported(): boolean {
  if (cachedRelaxedSimdSupport !== null) return cachedRelaxedSimdSupport;
  try {
    const f = new FuncBuilder([], [I32]);
    f.i32Const(0).v128Load(0)
      .i32Const(16).v128Load(0)
      .i32Const(32).v128Load(0)
      .i32x4RelaxedDotI8x16I7x16AddS()
      .i32x4Extract(0);
    const f2 = new FuncBuilder([], [F32]);
    f2.i32Const(0).v128Load(0)
      .i32Const(16).v128Load(0)
      .i32Const(32).v128Load(0)
      .f32x4RelaxedMadd()
      .f32x4Extract(0);
    const f3 = new FuncBuilder([], [I32]);
    f3.i32Const(0).v128Load(0)
      .i32Const(16).v128Load(0)
      .i16x8RelaxedDotI8x16I7x16S()
      .i32x4Extract(0);
    const probeBytes = new Uint8Array([
      0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
      ...section(1, [0x02, 0x60, 0x00, 0x01, I32, 0x60, 0x00, 0x01, F32]),
      ...section(3, [0x03, 0x00, 0x01, 0x00]),
      ...section(5, [0x01, 0x00, 0x01]),
      ...section(10, [0x03, ...f.encodeBody(), ...f2.encodeBody(), ...f3.encodeBody()]),
    ]);
    cachedRelaxedSimdSupport = WebAssembly.validate(probeBytes);
  } catch {
    cachedRelaxedSimdSupport = false;
  }
  return cachedRelaxedSimdSupport;
}

export function isVerdantWasmSimdSupported(): boolean {
  if (cachedSimdSupport !== null) return cachedSimdSupport;
  try {
    const useRelaxed = isVerdantRelaxedSimdSupported();
    const bytes = buildVerdantSimdModuleBytes(useRelaxed, false);
    cachedSimdSupport = WebAssembly.validate(bytes);
    if (cachedSimdSupport) {
      cachedCompiledModule = new WebAssembly.Module(bytes);
      cachedModuleByteLength = bytes.byteLength;
    } else if (useRelaxed) {
      // Fallback cleanly to standard SIMD128 if relaxed module rejected
      cachedRelaxedSimdSupport = false;
      const stdBytes = buildVerdantSimdModuleBytes(false, false);
      cachedSimdSupport = WebAssembly.validate(stdBytes);
      if (cachedSimdSupport) {
        cachedCompiledModule = new WebAssembly.Module(stdBytes);
        cachedModuleByteLength = stdBytes.byteLength;
      }
    }
  } catch {
    cachedSimdSupport = false;
  }
  return cachedSimdSupport;
}

/** Probes whether shared WebAssembly.Memory (SharedArrayBuffer + COOP/COEP) is supported for multi-core SIMD. */
export function isVerdantSharedWasmSupported(): boolean {
  if (cachedSharedWasmSupport !== null) return cachedSharedWasmSupport;
  if (!isVerdantWasmSimdSupported()) {
    cachedSharedWasmSupport = false;
    return false;
  }
  if (typeof SharedArrayBuffer === 'undefined') {
    cachedSharedWasmSupport = false;
    return false;
  }
  if (typeof crossOriginIsolated !== 'undefined' && !crossOriginIsolated) {
    cachedSharedWasmSupport = false;
    return false;
  }
  try {
    const testMem = new WebAssembly.Memory({ initial: 1, maximum: 2, shared: true });
    if (!(testMem.buffer instanceof SharedArrayBuffer)) {
      cachedSharedWasmSupport = false;
      return false;
    }
    const useRelaxed = isVerdantRelaxedSimdSupported();
    const bytes = buildVerdantSimdModuleBytes(useRelaxed, true);
    if (WebAssembly.validate(bytes)) {
      cachedSharedCompiledModule = new WebAssembly.Module(bytes);
      cachedSharedModuleByteLength = bytes.byteLength;
      cachedSharedWasmSupport = true;
      return true;
    }
    cachedSharedWasmSupport = false;
  } catch {
    cachedSharedWasmSupport = false;
  }
  return cachedSharedWasmSupport;
}

function alignUp32(n: number): number {
  const rem = n & 31;
  return rem === 0 ? n : n + (32 - rem);
}

const cachedProbedMaxPages: { shared: number | null; unshared: number | null } = {
  shared: null,
  unshared: null,
};

/**
 * Dynamically probes the host engine's true maximum `WebAssembly.Memory` page ceiling
 * (up to 3.75 GiB = 61,440 pages on 64-bit V8/SpiderMonkey/JSC, gracefully stepping down on
 * constrained browsers/devices) with zero physical RAM commit (`initial: 1`).
 */
export function probeMaxWasmMemoryPages(shared: boolean): number {
  const key = shared ? 'shared' : 'unshared';
  const cached = cachedProbedMaxPages[key];
  if (cached !== null) return cached;

  // Candidate ceilings in 64 KiB pages: 3.75 GiB -> 3.5 GiB -> 3.0 GiB -> 2.5 GiB -> 2.0 GiB -> 1.5 GiB -> 1.0 GiB
  const candidates = [61440, 57344, 49152, 40960, 32768, 24576, 16384];
  for (const maxPages of candidates) {
    try {
      const probe = new WebAssembly.Memory({
        initial: 1,
        maximum: maxPages,
        ...(shared ? { shared: true } : {}),
      });
      if (probe.buffer.byteLength >= 65536) {
        cachedProbedMaxPages[key] = maxPages;
        return maxPages;
      }
    } catch {
      // Try next lower ceiling
    }
  }
  cachedProbedMaxPages[key] = 16384;
  return 16384;
}

/**
 * Unified WASM linear memory arena + SIMD128/Relaxed-SIMD kernel dispatcher owned by each
 * VTensorBackendRegistry. Supports both single-core and zero-copy SharedArrayBuffer multi-core
 * WebAssembly.Memory execution with Theorem T3 deterministic row-striped partitioning.
 */
export class VerdantWasmArena {
  readonly memory: WebAssembly.Memory;
  readonly compiledModule: WebAssembly.Module;
  readonly kernels: VerdantSimdExports;
  readonly moduleByteSize: number;
  readonly relaxedSimdActive: boolean;
  readonly isSharedMemory: boolean;
  readonly maxPages: number;
  readonly maxCapacityBytes: number;

  private cursor = WASM_DYNAMIC_ARENA_START;
  private gen = 1;
  private cachedU8: Uint8Array;
  private cachedCalibration: ArenaCalibrationResult | null = null;
  private threadPool: VTensorThreadPoolLike | null = null;

  constructor(optionsOrPages: number | { initialPages?: number; shared?: boolean; maxPages?: number } = 64) {
    if (!isVerdantWasmSimdSupported() || !cachedCompiledModule) {
      throw new Error('WebAssembly SIMD128 is not supported in this runtime.');
    }
    const initialPages =
      typeof optionsOrPages === 'number' ? optionsOrPages : (optionsOrPages.initialPages ?? 64);
    const wantShared =
      typeof optionsOrPages === 'number' ? false : (optionsOrPages.shared ?? false);
    const requestedMaxPages =
      typeof optionsOrPages === 'number' ? undefined : optionsOrPages.maxPages;

    this.relaxedSimdActive = isVerdantRelaxedSimdSupported();
    const useShared = wantShared && isVerdantSharedWasmSupported() && cachedSharedCompiledModule !== null;
    this.isSharedMemory = useShared;
    this.compiledModule = useShared ? cachedSharedCompiledModule! : cachedCompiledModule;
    this.moduleByteSize = useShared ? cachedSharedModuleByteLength : cachedModuleByteLength;

    const probedMax = probeMaxWasmMemoryPages(useShared);
    const maxPages = Math.max(
      Math.max(16, initialPages),
      Math.min(probedMax, requestedMaxPages ?? probedMax),
    );
    this.maxPages = maxPages;
    this.maxCapacityBytes = maxPages * 65536;

    this.memory = new WebAssembly.Memory({
      initial: Math.max(16, initialPages),
      maximum: maxPages,
      ...(useShared ? { shared: true } : {}),
    });
    const instance = new WebAssembly.Instance(this.compiledModule, {
      env: { memory: this.memory },
    });
    this.kernels = instance.exports as unknown as VerdantSimdExports;
    this.cachedU8 = new Uint8Array(this.memory.buffer);

    // Populate the 65,536-entry F16 lookup table at offset 0 once per arena
    const f16Lut = new Float32Array(this.memory.buffer, WASM_F16_LUT_OFFSET, 65536);
    for (let i = 0; i < 65536; i++) {
      f16Lut[i] = f16ToF32(i);
    }
  }

  attachThreadPool(pool: VTensorThreadPoolLike | null): void {
    this.threadPool = pool;
  }

  get activeThreadPool(): VTensorThreadPoolLike | null {
    return this.threadPool;
  }

  get generation(): number {
    return this.gen;
  }

  get capacityBytes(): number {
    return this.memory.buffer.byteLength;
  }

  get usedBytes(): number {
    return this.cursor;
  }

  bytes(): Uint8Array {
    if (this.cachedU8.buffer !== this.memory.buffer) {
      this.cachedU8 = new Uint8Array(this.memory.buffer);
    }
    return this.cachedU8;
  }

  /** Allocates a 32-byte-aligned contiguous region inside WASM linear memory, growing if needed. */
  allocRegion(byteLength: number): number {
    const alignedStart = alignUp32(this.cursor);
    const alignedSize = alignUp32(Math.max(32, byteLength));
    const neededEnd = alignedStart + alignedSize;

    const currentBytes = this.memory.buffer.byteLength;
    if (neededEnd > currentBytes) {
      const extraBytes = neededEnd - currentBytes;
      const extraPages = Math.ceil(extraBytes / WASM_PAGE_BYTES);
      this.memory.grow(extraPages);
      this.gen += 1;
      this.cachedU8 = new Uint8Array(this.memory.buffer);
    }

    this.cursor = neededEnd;
    return alignedStart;
  }

  /** Resets dynamic buffer allocations when a model is unloaded (retains F16 LUT intact). */
  reset(): void {
    this.cursor = WASM_DYNAMIC_ARENA_START;
  }

  /**
   * Live Hardware Calibration Probe (< 8 ms):
   * Empirically measures sustained quantized weight-read bandwidth (B_mem, GB/s) on a cache-spanning
   * ~11.25 MiB matrix and peak VNNI/Int8 vector throughput (F_peak, GOPS/s) on an L1-resident batch-4 tile.
   * Restores the arena cursor afterward so zero bytes remain allocated.
   */
  runCalibrationProbe(): ArenaCalibrationResult {
    if (this.cachedCalibration) return this.cachedCalibration;

    const savedCursor = this.cursor;
    const t0 = performance.now();

    const cols = 1024; // 32 blocks of 32
    const rowsBw = 20480; // 20480 * 32 * 18 = 11,796,480 bytes (~11.25 MiB, exceeds L2 cache)
    const rowsCmp = 128; // 128 * 32 * 18 = 73,728 bytes (L1/L2 resident compute tile)
    const wPtr = this.allocRegion(rowsBw * (cols / 32) * 18);
    const xPtr = this.allocRegion(cols * 4 * 4);
    const q8Ptr = this.allocRegion((cols / 32) * Q8_0_ACT_BLOCK_BYTES * 4);
    const outPtr = this.allocRegion(rowsBw * 4);

    // Seed non-zero finite values in activations
    const f32v = new Float32Array(this.memory.buffer, xPtr, cols * 4);
    for (let i = 0; i < cols * 4; i++) {
      f32v[i] = ((i & 31) - 16) * 0.03125;
    }
    const qStride = (cols / 32) * Q8_0_ACT_BLOCK_BYTES;
    for (let t = 0; t < 4; t++) {
      this.kernels.quantize_f32_to_q8_0(xPtr + t * cols * 4, q8Ptr + t * qStride, cols);
    }

    // Warmup JIT so V8 finishes Liftoff -> TurboFan tier-up before timing
    for (let w = 0; w < 4; w++) {
      this.kernels.mul_mat_q4_0_q8(wPtr, q8Ptr, outPtr, cols, rowsCmp);
      this.kernels.mul_mat_q4_0_q8_batch4(wPtr, q8Ptr, outPtr, cols, rowsCmp, rowsCmp * 4);
    }
    this.kernels.mul_mat_q4_0_q8(wPtr, q8Ptr, outPtr, cols, rowsBw);

    // 1. Sustained weight-read bandwidth sweep (median of 3 sweeps over ~11.25 MiB)
    const bytesPerSweep = rowsBw * (cols / 32) * 18;
    const bwSamples: number[] = [];
    for (let burst = 0; burst < 3; burst++) {
      const tBw0 = performance.now();
      this.kernels.mul_mat_q4_0_q8(wPtr, q8Ptr, outPtr, cols, rowsBw);
      bwSamples.push(Math.max(1e-4, (performance.now() - tBw0) / 1000));
    }
    bwSamples.sort((a, b) => a - b);
    const medianBwSec = bwSamples[1];
    const memoryBandwidthGBs = Math.max(0.5, Math.min(200, (bytesPerSweep / 1e9) / medianBwSec));

    // 2. Peak VNNI/Int8 compute sweep (median of 3 bursts over 4x L1-resident batch-4 tiles)
    const cmpIters = 4;
    const opsPerBurst = cmpIters * 4 * 2 * rowsCmp * cols;
    const cmpSamples: number[] = [];
    for (let burst = 0; burst < 3; burst++) {
      const tCmp0 = performance.now();
      for (let i = 0; i < cmpIters; i++) {
        this.kernels.mul_mat_q4_0_q8_batch4(wPtr, q8Ptr, outPtr, cols, rowsCmp, rowsCmp * 4);
      }
      cmpSamples.push(Math.max(1e-4, (performance.now() - tCmp0) / 1000));
    }
    cmpSamples.sort((a, b) => a - b);
    const medianCmpSec = cmpSamples[1];
    const computeGops = Math.max(1.0, Math.min(1000, (opsPerBurst / 1e9) / medianCmpSec));

    const ridgeIntensityOpsPerByte = computeGops / Math.max(0.1, memoryBandwidthGBs);
    const calibrationMs = performance.now() - t0;

    this.cursor = savedCursor;
    this.cachedCalibration = {
      relaxedSimdActive: this.relaxedSimdActive,
      memoryBandwidthGBs: Number(memoryBandwidthGBs.toFixed(2)),
      computeGops: Number(computeGops.toFixed(2)),
      ridgeIntensityOpsPerByte: Number(ridgeIntensityOpsPerByte.toFixed(2)),
      calibrationMs: Number(calibrationMs.toFixed(2)),
    };
    return this.cachedCalibration;
  }

  /** Quantizes an F32 activation vector into packed 48-byte Q8_0_ACT blocks in WASM memory. */
  quantizeActivation(xPtr: number, q8Ptr: number, cols: number): void {
    this.kernels.quantize_f32_to_q8_0(xPtr, q8Ptr, cols);
  }

  /** Quantizes an F32 activation vector into packed 320-byte Q8_K super-blocks in WASM memory. */
  quantizeActivationQ8K(xPtr: number, q8kPtr: number, cols: number): void {
    this.kernels.quantize_f32_to_q8_k(xPtr, q8kPtr, cols);
  }

  /** Returns whether `weightType` supports fast Q8_0-activation integer dot-product kernels. */
  supportsQ8Activation(weightType: GgmlType): boolean {
    switch (weightType) {
      case GgmlType.Q4_0:
      case GgmlType.Q4_1:
      case GgmlType.Q5_0:
      case GgmlType.Q5_1:
      case GgmlType.Q8_0:
      case GgmlType.Q4_K:
      case GgmlType.Q5_K:
      case GgmlType.Q6_K:
        return true;
      default:
        return false;
    }
  }

  /** Returns whether strict F32-activation `mulMat` is lock-free (does not touch shared `WASM_BLOCK_SUM_X_OFFSET`). */
  private isStrictParallelSafe(weightType: GgmlType): boolean {
    return (
      weightType === GgmlType.F32 ||
      weightType === GgmlType.F16 ||
      weightType === GgmlType.BF16 ||
      weightType === GgmlType.Q8_0 ||
      weightType === GgmlType.Q2_K ||
      weightType === GgmlType.Q3_K ||
      weightType === GgmlType.Q6_K
    );
  }

  /** Dispatches strict F32-activation MUL_MAT directly on WASM linear memory byte pointers. */
  mulMat(
    weightType: GgmlType,
    wPtr: number,
    xPtr: number,
    outPtr: number,
    cols: number,
    rows: number,
  ): void {
    const pool = this.threadPool;
    if (
      pool &&
      pool.isActive &&
      rows >= 8 &&
      this.isStrictParallelSafe(weightType) &&
      weightRowByteSize(weightType, cols) * rows >= pool.parallelThresholdBytes
    ) {
      const rowBytes = weightRowByteSize(weightType, cols);
      pool.dispatchParallel(
        ParallelMatOp.MUL_MAT,
        weightType,
        wPtr,
        xPtr,
        xPtr,
        outPtr,
        cols,
        rows,
        rows * 4,
        (rowStart, rowCount) => {
          executeWasmMatStripe(
            this.kernels,
            ParallelMatOp.MUL_MAT,
            weightType,
            wPtr + rowStart * rowBytes,
            xPtr,
            xPtr,
            outPtr + rowStart * 4,
            cols,
            rowCount,
            rows * 4,
          );
        },
      );
      return;
    }

    executeWasmMatStripe(this.kernels, ParallelMatOp.MUL_MAT, weightType, wPtr, xPtr, xPtr, outPtr, cols, rows, rows * 4);
  }

  /**
   * Dispatches high-throughput Q8_0-activation integer dot-product MUL_MAT (Relaxed VNNI / SIMD128 Int8).
   * Automatically parallelizes across all P cores via Theorem T3 row-stripes when weightBytes >= W_par*(P).
   */
  mulMatQ8(
    weightType: GgmlType,
    wPtr: number,
    q8Ptr: number,
    xF32Ptr: number,
    outPtr: number,
    cols: number,
    rows: number,
  ): void {
    const pool = this.threadPool;
    if (
      pool &&
      pool.isActive &&
      rows >= 8 &&
      (this.supportsQ8Activation(weightType) || this.isStrictParallelSafe(weightType)) &&
      weightRowByteSize(weightType, cols) * rows >= pool.parallelThresholdBytes
    ) {
      const rowBytes = weightRowByteSize(weightType, cols);
      pool.dispatchParallel(
        ParallelMatOp.MUL_MAT_Q8,
        weightType,
        wPtr,
        q8Ptr,
        xF32Ptr,
        outPtr,
        cols,
        rows,
        rows * 4,
        (rowStart, rowCount) => {
          executeWasmMatStripe(
            this.kernels,
            ParallelMatOp.MUL_MAT_Q8,
            weightType,
            wPtr + rowStart * rowBytes,
            q8Ptr,
            xF32Ptr,
            outPtr + rowStart * 4,
            cols,
            rowCount,
            rows * 4,
          );
        },
      );
      return;
    }

    if (this.supportsQ8Activation(weightType)) {
      executeWasmMatStripe(this.kernels, ParallelMatOp.MUL_MAT_Q8, weightType, wPtr, q8Ptr, xF32Ptr, outPtr, cols, rows, rows * 4);
    } else {
      this.mulMat(weightType, wPtr, xF32Ptr, outPtr, cols, rows);
    }
  }

  /**
   * Batched multi-token strict F32-activation MUL_MAT (`batchSize` tokens packed contiguously).
   */
  mulMatBatch(
    weightType: GgmlType,
    wPtr: number,
    xBatchPtr: number,
    outBatchPtr: number,
    cols: number,
    rows: number,
    batchSize: number,
  ): void {
    const xStride = cols * 4;
    const oStride = rows * 4;
    const pool = this.threadPool;
    const rowBytes = weightRowByteSize(weightType, cols);
    const canParallelF32 =
      pool !== null &&
      pool.isActive &&
      rows >= 8 &&
      weightType === GgmlType.F32 &&
      rowBytes * rows >= pool.parallelThresholdBytes;

    let t = 0;
    if (weightType === GgmlType.Q4_0) {
      while (t + 4 <= batchSize) {
        this.kernels.mul_mat_q4_0_batch4(
          wPtr,
          xBatchPtr + t * xStride,
          outBatchPtr + t * oStride,
          cols,
          rows,
          oStride,
        );
        t += 4;
      }
    } else if (weightType === GgmlType.F32) {
      while (t + 4 <= batchSize) {
        const xb = xBatchPtr + t * xStride;
        const ob = outBatchPtr + t * oStride;
        if (canParallelF32 && pool) {
          pool.dispatchParallel(
            ParallelMatOp.MUL_MAT_BATCH4,
            weightType,
            wPtr,
            xb,
            xb,
            ob,
            cols,
            rows,
            oStride,
            (rowStart, rowCount) => {
              this.kernels.mul_mat_f32_batch4(
                wPtr + rowStart * rowBytes,
                xb,
                ob + rowStart * 4,
                cols,
                rowCount,
                oStride,
              );
            },
          );
        } else {
          this.kernels.mul_mat_f32_batch4(wPtr, xb, ob, cols, rows, oStride);
        }
        t += 4;
      }
    }
    while (t < batchSize) {
      this.mulMat(weightType, wPtr, xBatchPtr + t * xStride, outBatchPtr + t * oStride, cols, rows);
      t += 1;
    }
  }

  /**
   * Universal batched multi-token Q8_0-activation VNNI/Int8 MUL_MAT (`batchSize` pre-quantized tokens).
   * Uses fused 4-token zero-spill VNNI kernels for Q4_0, Q4_K, Q6_K, Q8_0, and F32 — and scales
   * across all P CPU cores via zero-copy shared WASM linear memory on large models!
   */
  mulMatBatchQ8(
    weightType: GgmlType,
    wPtr: number,
    q8BatchPtr: number,
    xF32BatchPtr: number,
    outBatchPtr: number,
    cols: number,
    rows: number,
    batchSize: number,
  ): void {
    const qStride = (cols >> 5) * Q8_0_ACT_BLOCK_BYTES;
    const xStride = cols * 4;
    const oStride = rows * 4;
    const pool = this.threadPool;
    const rowBytes = weightRowByteSize(weightType, cols);
    const hasBatch4Kernel =
      weightType === GgmlType.Q4_0 ||
      weightType === GgmlType.Q4_K ||
      weightType === GgmlType.Q6_K ||
      weightType === GgmlType.Q8_0 ||
      weightType === GgmlType.F32;
    const useParallel =
      pool !== null &&
      pool.isActive &&
      rows >= 8 &&
      hasBatch4Kernel &&
      rowBytes * rows >= pool.parallelThresholdBytes;

    let t = 0;
    if (hasBatch4Kernel) {
      while (t + 4 <= batchSize) {
        const qb = q8BatchPtr + t * qStride;
        const xb = xF32BatchPtr + t * xStride;
        const ob = outBatchPtr + t * oStride;
        if (useParallel && pool) {
          pool.dispatchParallel(
            ParallelMatOp.MUL_MAT_BATCH4_Q8,
            weightType,
            wPtr,
            qb,
            xb,
            ob,
            cols,
            rows,
            oStride,
            (rowStart, rowCount) => {
              executeWasmMatStripe(
                this.kernels,
                ParallelMatOp.MUL_MAT_BATCH4_Q8,
                weightType,
                wPtr + rowStart * rowBytes,
                qb,
                xb,
                ob + rowStart * 4,
                cols,
                rowCount,
                oStride,
              );
            },
          );
        } else {
          executeWasmMatStripe(
            this.kernels,
            ParallelMatOp.MUL_MAT_BATCH4_Q8,
            weightType,
            wPtr,
            qb,
            xb,
            ob,
            cols,
            rows,
            oStride,
          );
        }
        t += 4;
      }
    }

    while (t < batchSize) {
      this.mulMatQ8(
        weightType,
        wPtr,
        q8BatchPtr + t * qStride,
        xF32BatchPtr + t * xStride,
        outBatchPtr + t * oStride,
        cols,
        rows,
      );
      t += 1;
    }
  }

  /**
   * Fused multi-projection single-barrier dispatch (`Q+K+V` or `Gate+Up`) sharing a single input
   * vector across all `P` cores in ONE wakeup/join — cutting barrier syncs per layer from 7 to 4
   * and parallelizing small GQA `K`/`V` projections alongside `Q`!
   */
  mulMatQ8Group(
    entries: readonly MatGroupEntry[],
    q8Ptr: number,
    xF32Ptr: number,
    cols: number,
    useQ8: boolean,
  ): void {
    const pool = this.threadPool;
    const op = useQ8 ? ParallelMatOp.MUL_MAT_Q8 : ParallelMatOp.MUL_MAT;
    if (pool && pool.isActive && entries.length >= 2) {
      let totalBytes = 0;
      let allParallelSafe = true;
      for (let i = 0; i < entries.length; i++) {
        const e = entries[i];
        const safe = useQ8
          ? this.supportsQ8Activation(e.weightType) || this.isStrictParallelSafe(e.weightType)
          : this.isStrictParallelSafe(e.weightType);
        if (!safe || e.rows < 8) {
          allParallelSafe = false;
          break;
        }
        totalBytes += weightRowByteSize(e.weightType, cols) * e.rows;
      }
      if (allParallelSafe && totalBytes >= pool.parallelThresholdBytes) {
        pool.dispatchParallelGroup(op, entries, useQ8 ? q8Ptr : xF32Ptr, xF32Ptr, cols, (idx, rowStart, rowCount) => {
          const e = entries[idx];
          const rb = weightRowByteSize(e.weightType, cols);
          executeWasmMatStripe(
            this.kernels,
            op,
            e.weightType,
            e.wPtr + rowStart * rb,
            useQ8 ? q8Ptr : xF32Ptr,
            xF32Ptr,
            e.outPtr + rowStart * 4,
            cols,
            rowCount,
            e.oStrideBytes,
          );
        });
        return;
      }
    }

    for (let i = 0; i < entries.length; i++) {
      const e = entries[i];
      if (useQ8) {
        this.mulMatQ8(e.weightType, e.wPtr, q8Ptr, xF32Ptr, e.outPtr, cols, e.rows);
      } else {
        this.mulMat(e.weightType, e.wPtr, xF32Ptr, e.outPtr, cols, e.rows);
      }
    }
  }

  /** Fused 4-token batched multi-projection single-barrier dispatch (`Q+K+V` or `Gate+Up` prefill). */
  mulMatBatch4Q8Group(
    entries: readonly MatGroupEntry[],
    q8BatchPtr: number,
    xF32BatchPtr: number,
    cols: number,
    useQ8: boolean,
  ): void {
    const pool = this.threadPool;
    const op = useQ8 ? ParallelMatOp.MUL_MAT_BATCH4_Q8 : ParallelMatOp.MUL_MAT_BATCH4;
    if (pool && pool.isActive && entries.length >= 2) {
      let totalBytes = 0;
      let allBatch4Safe = true;
      for (let i = 0; i < entries.length; i++) {
        const e = entries[i];
        const hasB4 = useQ8
          ? e.weightType === GgmlType.Q4_0 ||
            e.weightType === GgmlType.Q4_K ||
            e.weightType === GgmlType.Q6_K ||
            e.weightType === GgmlType.Q8_0 ||
            e.weightType === GgmlType.F32
          : e.weightType === GgmlType.F32;
        if (!hasB4 || e.rows < 8) {
          allBatch4Safe = false;
          break;
        }
        totalBytes += weightRowByteSize(e.weightType, cols) * e.rows;
      }
      if (allBatch4Safe && totalBytes >= pool.parallelThresholdBytes) {
        pool.dispatchParallelGroup(
          op,
          entries,
          useQ8 ? q8BatchPtr : xF32BatchPtr,
          xF32BatchPtr,
          cols,
          (idx, rowStart, rowCount) => {
            const e = entries[idx];
            const rb = weightRowByteSize(e.weightType, cols);
            executeWasmMatStripe(
              this.kernels,
              op,
              e.weightType,
              e.wPtr + rowStart * rb,
              useQ8 ? q8BatchPtr : xF32BatchPtr,
              xF32BatchPtr,
              e.outPtr + rowStart * 4,
              cols,
              rowCount,
              e.oStrideBytes,
            );
          },
        );
        return;
      }
    }

    for (let i = 0; i < entries.length; i++) {
      const e = entries[i];
      if (useQ8) {
        this.mulMatBatchQ8(e.weightType, e.wPtr, q8BatchPtr, xF32BatchPtr, e.outPtr, cols, e.rows, 4);
      } else {
        this.mulMatBatch(e.weightType, e.wPtr, xF32BatchPtr, e.outPtr, cols, e.rows, 4);
      }
    }
  }

  /** Returns true if `weightType` supports Verdant/Prism Q8_K super-block narrow-lane contraction. */
  supportsQ8KActivation(weightType: GgmlType): boolean {
    return (
      weightType === GgmlType.Q2_K ||
      weightType === GgmlType.Q3_K ||
      weightType === GgmlType.Q4_K ||
      weightType === GgmlType.Q5_K ||
      weightType === GgmlType.Q6_K
    );
  }

  /**
   * Executes whole-layer 3-tier FlashAttention (`Hot F32 + Warm Q8 + Cold Q4`) in-Verdant,
   * automatically distributing Q heads across all `P` shared-WASM threads on long contexts
   * (`seqLen >= 192 && nHeads >= 2`) with zero-copy, zero-race per-worker scratch and bit-exact parity.
   */
  flashAttnLayer(
    qPtr: number,
    outPtr: number,
    pageStatePtr: number,
    pageSlotPtr: number,
    hotKLayerPtr: number,
    hotVLayerPtr: number,
    q8KLayerPtr: number,
    q8VLayerPtr: number,
    q8KScLayerPtr: number,
    q8VScLayerPtr: number,
    q4KLayerPtr: number,
    q4VLayerPtr: number,
    q4KScLayerPtr: number,
    q4VScLayerPtr: number,
    nHeads: number,
    nHeadsKv: number,
    headDim: number,
    seqLen: number,
    hotPageBytes: number,
    q8PageBytes: number,
    q4PageBytes: number,
    scPageBytes: number,
    hotKvStrideBytes: number,
    q8KvStrideBytes: number,
    q4KvStrideBytes: number,
    scStrideBytes: number,
  ): void {
    const pool = this.threadPool;
    // Adaptive work-based parallel FlashAttention floor: unlocks multi-core at seqLen >= 64
    // on 16H–32H × 128D models (~80–150 us work vs ~0.6 us barrier) while bypassing tiny models.
    if (pool && pool.isActive && nHeads >= 2 && nHeads * headDim * seqLen >= 131_072) {
      pool.dispatchParallelFlashAttn(
        qPtr,
        outPtr,
        pageStatePtr,
        pageSlotPtr,
        hotKLayerPtr,
        hotVLayerPtr,
        q8KLayerPtr,
        q8VLayerPtr,
        q8KScLayerPtr,
        q8VScLayerPtr,
        q4KLayerPtr,
        q4VLayerPtr,
        q4KScLayerPtr,
        q4VScLayerPtr,
        nHeads,
        nHeadsKv,
        headDim,
        seqLen,
        hotPageBytes,
        q8PageBytes,
        q4PageBytes,
        scPageBytes,
        hotKvStrideBytes,
        q8KvStrideBytes,
        q4KvStrideBytes,
        scStrideBytes,
        (hStart, hEnd, scratchBase) => {
          this.kernels.flash_attn_layer_range(
            qPtr,
            outPtr,
            pageStatePtr,
            pageSlotPtr,
            hotKLayerPtr,
            hotVLayerPtr,
            q8KLayerPtr,
            q8VLayerPtr,
            q8KScLayerPtr,
            q8VScLayerPtr,
            q4KLayerPtr,
            q4VLayerPtr,
            q4KScLayerPtr,
            q4VScLayerPtr,
            nHeads,
            nHeadsKv,
            headDim,
            seqLen,
            hotPageBytes,
            q8PageBytes,
            q4PageBytes,
            scPageBytes,
            hotKvStrideBytes,
            q8KvStrideBytes,
            q4KvStrideBytes,
            scStrideBytes,
            hStart,
            hEnd,
            scratchBase,
          );
        },
      );
      return;
    }
    this.kernels.flash_attn_layer(
      qPtr,
      outPtr,
      pageStatePtr,
      pageSlotPtr,
      hotKLayerPtr,
      hotVLayerPtr,
      q8KLayerPtr,
      q8VLayerPtr,
      q8KScLayerPtr,
      q8VScLayerPtr,
      q4KLayerPtr,
      q4VLayerPtr,
      q4KScLayerPtr,
      q4VScLayerPtr,
      nHeads,
      nHeadsKv,
      headDim,
      seqLen,
      hotPageBytes,
      q8PageBytes,
      q4PageBytes,
      scPageBytes,
      hotKvStrideBytes,
      q8KvStrideBytes,
      q4KvStrideBytes,
      scStrideBytes,
    );
  }

  /**
   * Verdant/Prism `Q8_K` super-block narrow-lane MUL_MAT (`Q4_K` / `Q6_K`), parallelized across
   * all `P` workers via Theorem T3 row-stripes with zero copy and bit-exact determinism.
   */
  mulMatQ8K(
    weightType: GgmlType,
    wPtr: number,
    q8kPtr: number,
    xF32Ptr: number,
    outPtr: number,
    cols: number,
    rows: number,
  ): void {
    if (!this.supportsQ8KActivation(weightType) || (cols & 255) !== 0) {
      this.mulMat(weightType, wPtr, xF32Ptr, outPtr, cols, rows);
      return;
    }
    const pool = this.threadPool;
    if (
      pool &&
      pool.isActive &&
      rows >= 8 &&
      weightRowByteSize(weightType, cols) * rows >= pool.parallelThresholdBytes
    ) {
      const rowBytes = weightRowByteSize(weightType, cols);
      pool.dispatchParallel(
        ParallelMatOp.MUL_MAT_Q8K,
        weightType,
        wPtr,
        q8kPtr,
        xF32Ptr,
        outPtr,
        cols,
        rows,
        rows * 4,
        (rowStart, rowCount) => {
          executeWasmMatStripe(
            this.kernels,
            ParallelMatOp.MUL_MAT_Q8K,
            weightType,
            wPtr + rowStart * rowBytes,
            q8kPtr,
            xF32Ptr,
            outPtr + rowStart * 4,
            cols,
            rowCount,
            rows * 4,
          );
        },
      );
      return;
    }
    executeWasmMatStripe(
      this.kernels,
      ParallelMatOp.MUL_MAT_Q8K,
      weightType,
      wPtr,
      q8kPtr,
      xF32Ptr,
      outPtr,
      cols,
      rows,
      rows * 4,
    );
  }

  /**
   * Fused multi-projection single-barrier `Q8_K` dispatch (`Q+K+V` or `Gate+Up` for K-quant models,
   * including mixed `Q4_K` + `Q6_K` groups like Qwen2.5 `attn_qkv`).
   */
  mulMatQ8KGroup(
    entries: readonly MatGroupEntry[],
    q8kPtr: number,
    xF32Ptr: number,
    cols: number,
  ): void {
    const pool = this.threadPool;
    const op = ParallelMatOp.MUL_MAT_Q8K;
    if (pool && pool.isActive && entries.length >= 2 && (cols & 255) === 0) {
      let totalBytes = 0;
      let allKQuant = true;
      for (let i = 0; i < entries.length; i++) {
        const e = entries[i];
        if (!this.supportsQ8KActivation(e.weightType) || e.rows < 8) {
          allKQuant = false;
          break;
        }
        totalBytes += weightRowByteSize(e.weightType, cols) * e.rows;
      }
      if (allKQuant && totalBytes >= pool.parallelThresholdBytes) {
        pool.dispatchParallelGroup(op, entries, q8kPtr, xF32Ptr, cols, (idx, rowStart, rowCount) => {
          const e = entries[idx];
          const rb = weightRowByteSize(e.weightType, cols);
          executeWasmMatStripe(
            this.kernels,
            op,
            e.weightType,
            e.wPtr + rowStart * rb,
            q8kPtr,
            xF32Ptr,
            e.outPtr + rowStart * 4,
            cols,
            rowCount,
            e.oStrideBytes,
          );
        });
        return;
      }
    }
    for (let i = 0; i < entries.length; i++) {
      const e = entries[i];
      this.mulMatQ8K(e.weightType, e.wPtr, q8kPtr, xF32Ptr, e.outPtr, cols, e.rows);
    }
  }

  /** Batched multi-token `Q8_K` narrow-lane prefill MUL_MAT (`batchSize` tokens packed contiguously). */
  mulMatBatchQ8K(
    weightType: GgmlType,
    wPtr: number,
    q8kBatchPtr: number,
    xF32BatchPtr: number,
    outBatchPtr: number,
    cols: number,
    rows: number,
    batchSize: number,
  ): void {
    const qStride = (cols >>> 8) * Q8_K_ACT_BLOCK_BYTES;
    const xStride = cols * 4;
    const oStride = rows * 4;
    const pool = this.threadPool;
    const rowBytes = weightRowByteSize(weightType, cols);
    const canParallel =
      pool !== null &&
      pool.isActive &&
      rows >= 8 &&
      this.supportsQ8KActivation(weightType) &&
      (cols & 255) === 0 &&
      rowBytes * rows >= pool.parallelThresholdBytes;

    let t = 0;
    if (this.supportsQ8KActivation(weightType) && (cols & 255) === 0) {
      while (t + 4 <= batchSize) {
        const qb = q8kBatchPtr + t * qStride;
        const xb = xF32BatchPtr + t * xStride;
        const ob = outBatchPtr + t * oStride;
        if (canParallel && pool) {
          pool.dispatchParallel(
            ParallelMatOp.MUL_MAT_BATCH4_Q8K,
            weightType,
            wPtr,
            qb,
            xb,
            ob,
            cols,
            rows,
            oStride,
            (rowStart, rowCount) => {
              executeWasmMatStripe(
                this.kernels,
                ParallelMatOp.MUL_MAT_BATCH4_Q8K,
                weightType,
                wPtr + rowStart * rowBytes,
                qb,
                xb,
                ob + rowStart * 4,
                cols,
                rowCount,
                oStride,
              );
            },
          );
        } else {
          executeWasmMatStripe(
            this.kernels,
            ParallelMatOp.MUL_MAT_BATCH4_Q8K,
            weightType,
            wPtr,
            qb,
            xb,
            ob,
            cols,
            rows,
            oStride,
          );
        }
        t += 4;
      }
    }
    while (t < batchSize) {
      this.mulMatQ8K(
        weightType,
        wPtr,
        q8kBatchPtr + t * qStride,
        xF32BatchPtr + t * xStride,
        outBatchPtr + t * oStride,
        cols,
        rows,
      );
      t += 1;
    }
  }

  /**
   * Single-barrier Expert-Parallel Sparse MoE dispatch (`OP_MOE_EXPERTS`):
   * Distributes active routed experts `0..K-1` across all `P` cores in ONE thread barrier,
   * executing `gate`, `up`, `swiglu_f32`, `quantize`, `down`, and SIMD `axpy_f32` in L1-resident
   * per-worker scratch, then deterministically summing worker accumulators.
   */
  dispatchMoeExpertsParallel(params: MoeExpertParallelDispatch): boolean {
    const pool = this.threadPool;
    if (!pool || !pool.isActive || params.expertCount < 2) {
      return false;
    }
    pool.dispatchParallelMoeExperts(params, this.kernels);
    return true;
  }

  /**
   * Single-barrier Expert-Parallel 4-token Prefill Sparse MoE dispatch (`OP_MOE_EXPERTS_BATCH4`):
   * Distributes all `uniqueCount` routed experts across all `P` cores in ONE barrier — using
   * Batch-4 contractions when all 4 tokens hit an expert (`mask === 0b1111`) and L1-hot per-token
   * contractions for sparse masks (`popcount(mask) ∈ {1,2,3}`), with SIMD reduction across cores!
   */
  dispatchMoeExpertsBatch4Parallel(params: MoeExpertBatch4ParallelDispatch): boolean {
    const pool = this.threadPool;
    if (!pool || !pool.isActive || params.uniqueCount < 2) {
      return false;
    }
    pool.dispatchParallelMoeExpertsBatch4(params, this.kernels);
    return true;
  }

  /** Fused 4-token batched multi-projection single-barrier `Q8_K` prefill dispatch. */
  mulMatBatch4Q8KGroup(
    entries: readonly MatGroupEntry[],
    q8kBatchPtr: number,
    xF32BatchPtr: number,
    cols: number,
  ): void {
    const pool = this.threadPool;
    const op = ParallelMatOp.MUL_MAT_BATCH4_Q8K;
    if (pool && pool.isActive && entries.length >= 2 && (cols & 255) === 0) {
      let totalBytes = 0;
      let allKQuant = true;
      for (let i = 0; i < entries.length; i++) {
        const e = entries[i];
        if (!this.supportsQ8KActivation(e.weightType) || e.rows < 8) {
          allKQuant = false;
          break;
        }
        totalBytes += weightRowByteSize(e.weightType, cols) * e.rows;
      }
      if (allKQuant && totalBytes >= pool.parallelThresholdBytes) {
        pool.dispatchParallelGroup(op, entries, q8kBatchPtr, xF32BatchPtr, cols, (idx, rowStart, rowCount) => {
          const e = entries[idx];
          const rb = weightRowByteSize(e.weightType, cols);
          executeWasmMatStripe(
            this.kernels,
            op,
            e.weightType,
            e.wPtr + rowStart * rb,
            q8kBatchPtr,
            xF32BatchPtr,
            e.outPtr + rowStart * 4,
            cols,
            rowCount,
            e.oStrideBytes,
          );
        });
        return;
      }
    }
    for (let i = 0; i < entries.length; i++) {
      const e = entries[i];
      this.mulMatBatchQ8K(e.weightType, e.wPtr, q8kBatchPtr, xF32BatchPtr, e.outPtr, cols, e.rows, 4);
    }
  }
}
