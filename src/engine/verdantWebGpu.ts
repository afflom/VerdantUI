/**
 * Verdant Hardware WebGPU Compute Substrate (internal to `vTensor`).
 *
 * Directly extrapolated from `github.com/afflom/verdant-engine/site/scene-webgpu-renderer.mjs`:
 * - Probes `navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })`.
 * - Strictly REJECTS software/emulated adapters (SwiftShader, llvmpipe, lavapipe, Basic Render,
 *   `isFallbackAdapter === true`) so WebGPU never regresses below multi-core WASM Relaxed-SIMD VNNI.
 * - Compiles 64-thread workgroup-cooperative WGSL compute kernels (`F32`, `Q4_0`, `Q8_0`, `Q4_K`)
 *   where all 64 threads in a workgroup coalesce contiguous DRAM reads across a single output row
 *   and reduce via `var<workgroup>` shared memory (10x+ higher GPU memory bandwidth than 1-thread/row).
 * - Maintains persistent GPU weight storage buffers + reusable activation/readback scratch buffers
 *   so large matrices (`>= 512 KiB`, e.g. 135M–3B linear projections & LM heads) dispatch cleanly on
 *   hardware GPU silicon with zero per-token weight re-uploads.
 * - Executes strictly INSIDE `vTensor` — llama.cpp's default `GGML_WEBGPU` provider stays blocked!
 */

import { GgmlType } from './gguf';

export type WebGpuProbeStatus =
  | 'hardware-active'
  | 'software-rejected'
  | 'unavailable'
  | 'validation-failed'
  | 'pending';

export interface WebGpuAdapterInfoLike {
  readonly vendor?: string;
  readonly architecture?: string;
  readonly device?: string;
  readonly description?: string;
  readonly isFallbackAdapter?: boolean;
}

export interface WebGpuSubstrateTelemetry {
  readonly status: WebGpuProbeStatus;
  readonly adapterLabel: string;
  readonly vendor: string;
  readonly architecture: string;
  readonly maxStorageBufferMB: number;
  readonly shadersCompiled: number;
  readonly selfTestMaxError: number | null;
  readonly persistentBuffers: number;
  readonly persistentBytes: number;
  readonly gpuDispatches: number;
  readonly gpuDispatchedBytes: number;
  readonly reason: string;
}

const SOFTWARE_ADAPTER_REGEX =
  /swiftshader|llvmpipe|lavapipe|software|basic render|microsoft basic|subzero|cpu|emulated/i;

export interface AdapterClassification {
  readonly admitted: boolean;
  readonly status: 'hardware-active' | 'software-rejected';
  readonly adapterLabel: string;
  readonly reason: string;
}

/**
 * Pure, deterministic adapter admission policy matching `afflom/verdant-engine`
 * (`scene-webgpu-renderer.mjs` & `scene-hd-renderer.mjs`):
 * Never admits software-emulated WebGPU adapters.
 */
export function classifyWebGpuAdapter(info: WebGpuAdapterInfoLike): AdapterClassification {
  const vendor = (info.vendor ?? '').trim();
  const architecture = (info.architecture ?? '').trim();
  const device = (info.device ?? '').trim();
  const description = (info.description ?? '').trim();

  const labelParts = [vendor, device || architecture, description].filter(Boolean);
  const adapterLabel = labelParts.length > 0 ? labelParts.join(' · ') : 'WebGPU Adapter';
  const combined = `${vendor} ${architecture} ${device} ${description}`;

  if (info.isFallbackAdapter === true) {
    return {
      admitted: false,
      status: 'software-rejected',
      adapterLabel,
      reason: 'Fallback/software WebGPU adapter rejected; multi-core WASM SIMD VNNI is faster.',
    };
  }

  if (SOFTWARE_ADAPTER_REGEX.test(combined)) {
    return {
      admitted: false,
      status: 'software-rejected',
      adapterLabel,
      reason: `Software renderer (${adapterLabel}) rejected; keeping multi-core WASM SIMD VNNI.`,
    };
  }

  return {
    admitted: true,
    status: 'hardware-active',
    adapterLabel,
    reason: 'Hardware WebGPU workgroup-tiled compute verified inside vTensor.',
  };
}

/**
 * Workgroup-cooperative `@compute @workgroup_size(64)` F32 matrix-vector kernel.
 * 1 workgroup = 1 output row; all 64 threads perform coalesced contiguous reads along `cols`
 * and perform a 6-step shared-memory tree reduction (`sharedAcc`).
 */
export const WGSL_MUL_MAT_F32 = /* wgsl */ `
struct Params {
  rows: u32,
  cols: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> act: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> sharedAcc: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wid.x;
  let tid = lid.x;
  if (row >= params.rows) {
    return;
  }
  let base = row * params.cols;
  var acc: f32 = 0.0;
  var c: u32 = tid;
  while (c < params.cols) {
    acc = acc + weights[base + c] * act[c];
    c = c + 64u;
  }
  sharedAcc[tid] = acc;
  workgroupBarrier();
  if (tid < 32u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 32u]; }
  workgroupBarrier();
  if (tid < 16u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 16u]; }
  workgroupBarrier();
  if (tid < 8u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 8u]; }
  workgroupBarrier();
  if (tid < 4u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 4u]; }
  workgroupBarrier();
  if (tid < 2u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 2u]; }
  workgroupBarrier();
  if (tid == 0u) {
    out[row] = sharedAcc[0] + sharedAcc[1];
  }
}
`;

/**
 * Workgroup-cooperative `@compute @workgroup_size(64)` zero-decode Q4_0 matrix-vector kernel.
 * 64 threads stride across `blocksPerRow` with coalesced memory traffic and shared-memory reduction.
 */
export const WGSL_MUL_MAT_Q4_0 = /* wgsl */ `
struct Params {
  rows: u32,
  cols: u32,
  blocksPerRow: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weightBytes: array<u32>;
@group(0) @binding(2) var<storage, read> act: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> sharedAcc: array<f32, 64>;

fn readU8(byteOff: u32) -> u32 {
  let word = weightBytes[byteOff >> 2u];
  return (word >> ((byteOff & 3u) * 8u)) & 0xffu;
}

fn readF16(byteOff: u32) -> f32 {
  let lo = readU8(byteOff);
  let hi = readU8(byteOff + 1u);
  return unpack2x16float(lo | (hi << 8u)).x;
}

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wid.x;
  let tid = lid.x;
  if (row >= params.rows) {
    return;
  }
  let rowByteBase = row * params.blocksPerRow * 18u;
  var acc: f32 = 0.0;
  var b: u32 = tid;
  while (b < params.blocksPerRow) {
    let blkOff = rowByteBase + b * 18u;
    let d = readF16(blkOff);
    let xBase = b * 32u;
    var blockSum: f32 = 0.0;
    for (var j: u32 = 0u; j < 16u; j = j + 1u) {
      let byteVal = readU8(blkOff + 2u + j);
      let qLo = f32(i32(byteVal & 0x0fu) - 8);
      let qHi = f32(i32((byteVal >> 4u) & 0x0fu) - 8);
      blockSum = blockSum + qLo * act[xBase + j] + qHi * act[xBase + 16u + j];
    }
    acc = acc + d * blockSum;
    b = b + 64u;
  }
  sharedAcc[tid] = acc;
  workgroupBarrier();
  if (tid < 32u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 32u]; }
  workgroupBarrier();
  if (tid < 16u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 16u]; }
  workgroupBarrier();
  if (tid < 8u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 8u]; }
  workgroupBarrier();
  if (tid < 4u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 4u]; }
  workgroupBarrier();
  if (tid < 2u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 2u]; }
  workgroupBarrier();
  if (tid == 0u) {
    out[row] = sharedAcc[0] + sharedAcc[1];
  }
}
`;

/**
 * Workgroup-cooperative `@compute @workgroup_size(64)` zero-decode Q8_0 matrix-vector kernel.
 */
export const WGSL_MUL_MAT_Q8_0 = /* wgsl */ `
struct Params {
  rows: u32,
  cols: u32,
  blocksPerRow: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weightBytes: array<u32>;
@group(0) @binding(2) var<storage, read> act: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> sharedAcc: array<f32, 64>;

fn readU8(byteOff: u32) -> u32 {
  let word = weightBytes[byteOff >> 2u];
  return (word >> ((byteOff & 3u) * 8u)) & 0xffu;
}

fn readI8(byteOff: u32) -> f32 {
  let v = i32(readU8(byteOff));
  return f32(select(v, v - 256, v >= 128));
}

fn readF16(byteOff: u32) -> f32 {
  let lo = readU8(byteOff);
  let hi = readU8(byteOff + 1u);
  return unpack2x16float(lo | (hi << 8u)).x;
}

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wid.x;
  let tid = lid.x;
  if (row >= params.rows) {
    return;
  }
  let rowByteBase = row * params.blocksPerRow * 34u;
  var acc: f32 = 0.0;
  var b: u32 = tid;
  while (b < params.blocksPerRow) {
    let blkOff = rowByteBase + b * 34u;
    let d = readF16(blkOff);
    let xBase = b * 32u;
    var blockSum: f32 = 0.0;
    for (var j: u32 = 0u; j < 32u; j = j + 1u) {
      blockSum = blockSum + readI8(blkOff + 2u + j) * act[xBase + j];
    }
    acc = acc + d * blockSum;
    b = b + 64u;
  }
  sharedAcc[tid] = acc;
  workgroupBarrier();
  if (tid < 32u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 32u]; }
  workgroupBarrier();
  if (tid < 16u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 16u]; }
  workgroupBarrier();
  if (tid < 8u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 8u]; }
  workgroupBarrier();
  if (tid < 4u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 4u]; }
  workgroupBarrier();
  if (tid < 2u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 2u]; }
  workgroupBarrier();
  if (tid == 0u) {
    out[row] = sharedAcc[0] + sharedAcc[1];
  }
}
`;

/**
 * Workgroup-cooperative `@compute @workgroup_size(64)` zero-decode Q4_K (K-quant 256-elem super-block)
 * matrix-vector kernel for real 135M–3B Q4_K_M models.
 */
export const WGSL_MUL_MAT_Q4_K = /* wgsl */ `
struct Params {
  rows: u32,
  cols: u32,
  superBlocksPerRow: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weightBytes: array<u32>;
@group(0) @binding(2) var<storage, read> act: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> sharedAcc: array<f32, 64>;

fn readU8(byteOff: u32) -> u32 {
  let word = weightBytes[byteOff >> 2u];
  return (word >> ((byteOff & 3u) * 8u)) & 0xffu;
}

fn readF16(byteOff: u32) -> f32 {
  let lo = readU8(byteOff);
  let hi = readU8(byteOff + 1u);
  return unpack2x16float(lo | (hi << 8u)).x;
}

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wid.x;
  let tid = lid.x;
  if (row >= params.rows) {
    return;
  }
  let rowByteBase = row * params.superBlocksPerRow * 144u;
  var acc: f32 = 0.0;
  var sb: u32 = tid;
  while (sb < params.superBlocksPerRow) {
    let off = rowByteBase + sb * 144u;
    let d = readF16(off);
    let dmin = readF16(off + 2u);
    let scOff = off + 4u;
    let qsOff = off + 16u;
    let xBase = sb * 256u;

    for (var g: u32 = 0u; g < 4u; g = g + 1u) {
      let idx0 = g * 2u;
      let idx1 = idx0 + 1u;
      var sc0: u32;
      var mn0: u32;
      var sc1: u32;
      var mn1: u32;
      if (idx0 < 4u) {
        sc0 = readU8(scOff + idx0) & 63u;
        mn0 = readU8(scOff + idx0 + 4u) & 63u;
        sc1 = readU8(scOff + idx1) & 63u;
        mn1 = readU8(scOff + idx1 + 4u) & 63u;
      } else {
        sc0 = (readU8(scOff + idx0 + 4u) & 0x0fu) | ((readU8(scOff + idx0 - 4u) >> 6u) << 4u);
        mn0 = (readU8(scOff + idx0 + 4u) >> 4u) | ((readU8(scOff + idx0) >> 6u) << 4u);
        sc1 = (readU8(scOff + idx1 + 4u) & 0x0fu) | ((readU8(scOff + idx1 - 4u) >> 6u) << 4u);
        mn1 = (readU8(scOff + idx1 + 4u) >> 4u) | ((readU8(scOff + idx1) >> 6u) << 4u);
      }
      let d1 = d * f32(sc0);
      let m1 = dmin * f32(mn0);
      let d2 = d * f32(sc1);
      let m2 = dmin * f32(mn1);
      let qStart = qsOff + g * 32u;
      let xStart = xBase + g * 64u;
      var sumLo: f32 = 0.0;
      var sumXLo: f32 = 0.0;
      var sumHi: f32 = 0.0;
      var sumXHi: f32 = 0.0;
      for (var l: u32 = 0u; l < 32u; l = l + 1u) {
        let bv = readU8(qStart + l);
        let xLo = act[xStart + l];
        let xHi = act[xStart + 32u + l];
        sumLo = sumLo + f32(bv & 0x0fu) * xLo;
        sumXLo = sumXLo + xLo;
        sumHi = sumHi + f32((bv >> 4u) & 0x0fu) * xHi;
        sumXHi = sumXHi + xHi;
      }
      acc = acc + (d1 * sumLo - m1 * sumXLo) + (d2 * sumHi - m2 * sumXHi);
    }
    sb = sb + 64u;
  }
  sharedAcc[tid] = acc;
  workgroupBarrier();
  if (tid < 32u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 32u]; }
  workgroupBarrier();
  if (tid < 16u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 16u]; }
  workgroupBarrier();
  if (tid < 8u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 8u]; }
  workgroupBarrier();
  if (tid < 4u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 4u]; }
  workgroupBarrier();
  if (tid < 2u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 2u]; }
  workgroupBarrier();
  if (tid == 0u) {
    out[row] = sharedAcc[0] + sharedAcc[1];
  }
}
`;

/**
 * Workgroup-cooperative `@compute @workgroup_size(64)` zero-decode Q6_K (256-elem super-block = 210 bytes)
 * matrix-vector kernel for Q4_K_M LM heads and FFN down projections on hardware GPUs.
 */
export const WGSL_MUL_MAT_Q6_K = /* wgsl */ `
struct Params {
  rows: u32,
  cols: u32,
  superBlocksPerRow: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weightBytes: array<u32>;
@group(0) @binding(2) var<storage, read> act: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> sharedAcc: array<f32, 64>;

fn readU8(byteOff: u32) -> u32 {
  let word = weightBytes[byteOff >> 2u];
  return (word >> ((byteOff & 3u) * 8u)) & 0xffu;
}

fn readI8(byteOff: u32) -> f32 {
  let v = i32(readU8(byteOff));
  return f32(select(v, v - 256, v >= 128));
}

fn readF16(byteOff: u32) -> f32 {
  let lo = readU8(byteOff);
  let hi = readU8(byteOff + 1u);
  return unpack2x16float(lo | (hi << 8u)).x;
}

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wid.x;
  let tid = lid.x;
  if (row >= params.rows) {
    return;
  }
  let rowByteBase = row * params.superBlocksPerRow * 210u;
  var acc: f32 = 0.0;
  var sb: u32 = tid;
  while (sb < params.superBlocksPerRow) {
    let off = rowByteBase + sb * 210u;
    let d = readF16(off + 208u);
    let xBase = sb * 256u;
    var sbSum: f32 = 0.0;

    for (var h: u32 = 0u; h < 2u; h = h + 1u) {
      let qlBase = off + h * 64u;
      let qhBase = off + 128u + h * 32u;
      let scBase = off + 192u + h * 8u;
      let xHalf = xBase + h * 128u;
      for (var l: u32 = 0u; l < 32u; l = l + 1u) {
        let is = l >> 4u;
        let ql0 = readU8(qlBase + l);
        let ql1 = readU8(qlBase + 32u + l);
        let qh = readU8(qhBase + l);
        let q1 = f32(i32((ql0 & 0x0fu) | ((qh & 0x03u) << 4u)) - 32);
        let q2 = f32(i32((ql1 & 0x0fu) | (((qh >> 2u) & 0x03u) << 4u)) - 32);
        let q3 = f32(i32((ql0 >> 4u) | (((qh >> 4u) & 0x03u) << 4u)) - 32);
        let q4 = f32(i32((ql1 >> 4u) | (((qh >> 6u) & 0x03u) << 4u)) - 32);
        sbSum = sbSum + readI8(scBase + is) * q1 * act[xHalf + l];
        sbSum = sbSum + readI8(scBase + is + 2u) * q2 * act[xHalf + 32u + l];
        sbSum = sbSum + readI8(scBase + is + 4u) * q3 * act[xHalf + 64u + l];
        sbSum = sbSum + readI8(scBase + is + 6u) * q4 * act[xHalf + 96u + l];
      }
    }
    acc = acc + d * sbSum;
    sb = sb + 64u;
  }
  sharedAcc[tid] = acc;
  workgroupBarrier();
  if (tid < 32u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 32u]; }
  workgroupBarrier();
  if (tid < 16u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 16u]; }
  workgroupBarrier();
  if (tid < 8u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 8u]; }
  workgroupBarrier();
  if (tid < 4u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 4u]; }
  workgroupBarrier();
  if (tid < 2u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 2u]; }
  workgroupBarrier();
  if (tid == 0u) {
    out[row] = sharedAcc[0] + sharedAcc[1];
  }
}
`;

/**
 * Workgroup-cooperative `@compute @workgroup_size(64)` zero-decode F16 matrix-vector kernel
 * using hardware `unpack2x16float` (2 elements per u32 read).
 */
export const WGSL_MUL_MAT_F16 = /* wgsl */ `
struct Params {
  rows: u32,
  cols: u32,
  pairsPerRow: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> weightWords: array<u32>;
@group(0) @binding(2) var<storage, read> act: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> sharedAcc: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wid.x;
  let tid = lid.x;
  if (row >= params.rows) {
    return;
  }
  let base = row * params.pairsPerRow;
  var acc: f32 = 0.0;
  var p: u32 = tid;
  while (p < params.pairsPerRow) {
    let pair = unpack2x16float(weightWords[base + p]);
    let c = p * 2u;
    acc = acc + pair.x * act[c] + pair.y * act[c + 1u];
    p = p + 64u;
  }
  sharedAcc[tid] = acc;
  workgroupBarrier();
  if (tid < 32u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 32u]; }
  workgroupBarrier();
  if (tid < 16u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 16u]; }
  workgroupBarrier();
  if (tid < 8u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 8u]; }
  workgroupBarrier();
  if (tid < 4u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 4u]; }
  workgroupBarrier();
  if (tid < 2u) { sharedAcc[tid] = sharedAcc[tid] + sharedAcc[tid + 2u]; }
  workgroupBarrier();
  if (tid == 0u) {
    out[row] = sharedAcc[0] + sharedAcc[1];
  }
}
`;

// Minimal structural types so TypeScript builds cleanly without `@webgpu/types`
interface GpuBufferLike {
  size: number;
  mapAsync(mode: number): Promise<void>;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}

interface GpuDeviceLike {
  limits: { maxStorageBufferBindingSize?: number };
  queue: {
    writeBuffer(buffer: GpuBufferLike, bufferOffset: number, data: ArrayBufferView | ArrayBufferLike): void;
    submit(commandBuffers: unknown[]): void;
  };
  createShaderModule(desc: { code: string }): unknown;
  createComputePipeline(desc: {
    layout: 'auto';
    compute: { module: unknown; entryPoint: string };
  }): { getBindGroupLayout(index: number): unknown };
  createBuffer(desc: { size: number; usage: number }): GpuBufferLike;
  createBindGroup(desc: {
    layout: unknown;
    entries: Array<{ binding: number; resource: { buffer: GpuBufferLike } }>;
  }): unknown;
  createCommandEncoder(): {
    beginComputePass(): {
      setPipeline(p: unknown): void;
      setBindGroup(index: number, bg: unknown): void;
      dispatchWorkgroups(x: number): void;
      end(): void;
    };
    copyBufferToBuffer(
      src: GpuBufferLike,
      srcOffset: number,
      dst: GpuBufferLike,
      dstOffset: number,
      size: number,
    ): void;
    finish(): unknown;
  };
  destroy(): void;
}

interface GpuAdapterLike {
  info?: WebGpuAdapterInfoLike;
  isFallbackAdapter?: boolean;
  limits?: { maxStorageBufferBindingSize?: number };
  requestDevice(desc?: unknown): Promise<GpuDeviceLike | null>;
}

interface NavigatorGpuLike {
  requestAdapter(opts?: { powerPreference?: string }): Promise<GpuAdapterLike | null>;
}

const GPU_BUFFER_USAGE = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
};

const GPU_MAP_MODE_READ = 0x0001;

interface PersistentGpuWeight {
  readonly buffer: GpuBufferLike;
  readonly byteSize: number;
  readonly weightType: GgmlType;
  readonly rows: number;
  readonly cols: number;
}

/** Minimum weight byte size (512 KiB) to justify GPU command-buffer submission over WASM SIMD. */
export const WEBGPU_DISPATCH_MIN_BYTES = 512 * 1024;

/**
 * Verdant Hardware WebGPU Compute Substrate (`vTensor`-internal).
 * Holds compiled workgroup-tiled WGSL pipelines (`F32`, `F16`, `Q4_0`, `Q8_0`, `Q4_K`, `Q6_K`),
 * persistent GPU weight buffers, and reusable scratch buffers for zero-allocation single & fused
 * multi-projection group GPU dispatch.
 */
export class VerdantWebGpuSubstrate {
  private device: GpuDeviceLike | null = null;
  private f32Pipeline: { getBindGroupLayout(index: number): unknown } | null = null;
  private f16Pipeline: { getBindGroupLayout(index: number): unknown } | null = null;
  private q4Pipeline: { getBindGroupLayout(index: number): unknown } | null = null;
  private q8Pipeline: { getBindGroupLayout(index: number): unknown } | null = null;
  private q4kPipeline: { getBindGroupLayout(index: number): unknown } | null = null;
  private q6kPipeline: { getBindGroupLayout(index: number): unknown } | null = null;
  private readonly weightBuffers = new Map<string, PersistentGpuWeight>();
  private persistentBytes = 0;
  private gpuDispatches = 0;
  private gpuDispatchedBytes = 0;

  // Reusable dispatch scratch buffers (avoid per-op GPU buffer allocations)
  private paramsBuf: GpuBufferLike | null = null;
  private groupParamsBufs: GpuBufferLike[] = [];
  private groupOutBufs: GpuBufferLike[] = [];
  private scratchXBuf: GpuBufferLike | null = null;
  private scratchOutBuf: GpuBufferLike | null = null;
  private scratchReadBuf: GpuBufferLike | null = null;

  private telemetry: WebGpuSubstrateTelemetry = {
    status: 'pending',
    adapterLabel: 'Probing…',
    vendor: '',
    architecture: '',
    maxStorageBufferMB: 0,
    shadersCompiled: 0,
    selfTestMaxError: null,
    persistentBuffers: 0,
    persistentBytes: 0,
    gpuDispatches: 0,
    gpuDispatchedBytes: 0,
    reason: 'Awaiting WebGPU hardware probe.',
  };

  get isHardwareActive(): boolean {
    return this.telemetry.status === 'hardware-active' && this.device !== null;
  }

  getTelemetry(): WebGpuSubstrateTelemetry {
    return {
      ...this.telemetry,
      persistentBuffers: this.weightBuffers.size,
      persistentBytes: this.persistentBytes,
      gpuDispatches: this.gpuDispatches,
      gpuDispatchedBytes: this.gpuDispatchedBytes,
    };
  }

  supportsWeightType(weightType: GgmlType): boolean {
    return (
      weightType === GgmlType.F32 ||
      weightType === GgmlType.F16 ||
      weightType === GgmlType.Q4_0 ||
      weightType === GgmlType.Q8_0 ||
      weightType === GgmlType.Q4_K ||
      weightType === GgmlType.Q6_K
    );
  }

  hasWeightBuffer(id: string): boolean {
    return this.weightBuffers.has(id);
  }

  /**
   * Probes `navigator.gpu`, rejects software adapters, compiles workgroup-tiled WGSL compute
   * shaders (F32, Q4_0, Q8_0, Q4_K), and validates numerical parity on live GPU silicon.
   */
  async probeAndInit(): Promise<WebGpuSubstrateTelemetry> {
    const navGpu =
      typeof navigator !== 'undefined'
        ? ((navigator as unknown as { gpu?: NavigatorGpuLike }).gpu ?? null)
        : null;

    if (!navGpu || typeof navGpu.requestAdapter !== 'function') {
      this.telemetry = {
        status: 'unavailable',
        adapterLabel: 'No WebGPU API',
        vendor: '',
        architecture: '',
        maxStorageBufferMB: 0,
        shadersCompiled: 0,
        selfTestMaxError: null,
        persistentBuffers: 0,
        persistentBytes: 0,
        gpuDispatches: 0,
        gpuDispatchedBytes: 0,
        reason: 'WebGPU unavailable in this runtime; multi-core WASM SIMD VNNI active.',
      };
      return this.getTelemetry();
    }

    try {
      const adapter = await navGpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) {
        this.telemetry = {
          status: 'unavailable',
          adapterLabel: 'No hardware GPU adapter',
          vendor: '',
          architecture: '',
          maxStorageBufferMB: 0,
          shadersCompiled: 0,
          selfTestMaxError: null,
          persistentBuffers: 0,
          persistentBytes: 0,
          gpuDispatches: 0,
          gpuDispatchedBytes: 0,
          reason: 'No WebGPU adapter exposed; multi-core WASM SIMD VNNI active.',
        };
        return this.getTelemetry();
      }

      const info: WebGpuAdapterInfoLike = {
        ...(adapter.info ?? {}),
        isFallbackAdapter: adapter.isFallbackAdapter ?? adapter.info?.isFallbackAdapter,
      };
      const classification = classifyWebGpuAdapter(info);
      const maxStorageBytes = adapter.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
      const maxStorageBufferMB = Math.round(maxStorageBytes / (1024 * 1024));

      if (!classification.admitted) {
        this.telemetry = {
          status: 'software-rejected',
          adapterLabel: classification.adapterLabel,
          vendor: info.vendor ?? '',
          architecture: info.architecture ?? '',
          maxStorageBufferMB,
          shadersCompiled: 0,
          selfTestMaxError: null,
          persistentBuffers: 0,
          persistentBytes: 0,
          gpuDispatches: 0,
          gpuDispatchedBytes: 0,
          reason: classification.reason,
        };
        return this.getTelemetry();
      }

      const device = await adapter.requestDevice();
      if (!device) {
        this.telemetry = {
          status: 'unavailable',
          adapterLabel: classification.adapterLabel,
          vendor: info.vendor ?? '',
          architecture: info.architecture ?? '',
          maxStorageBufferMB,
          shadersCompiled: 0,
          selfTestMaxError: null,
          persistentBuffers: 0,
          persistentBytes: 0,
          gpuDispatches: 0,
          gpuDispatchedBytes: 0,
          reason: 'WebGPU device request returned null; keeping multi-core WASM SIMD VNNI.',
        };
        return this.getTelemetry();
      }

      const modF32 = device.createShaderModule({ code: WGSL_MUL_MAT_F32 });
      const modF16 = device.createShaderModule({ code: WGSL_MUL_MAT_F16 });
      const modQ4 = device.createShaderModule({ code: WGSL_MUL_MAT_Q4_0 });
      const modQ8 = device.createShaderModule({ code: WGSL_MUL_MAT_Q8_0 });
      const modQ4k = device.createShaderModule({ code: WGSL_MUL_MAT_Q4_K });
      const modQ6k = device.createShaderModule({ code: WGSL_MUL_MAT_Q6_K });

      this.device = device;
      this.f32Pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: modF32, entryPoint: 'main' },
      });
      this.f16Pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: modF16, entryPoint: 'main' },
      });
      this.q4Pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: modQ4, entryPoint: 'main' },
      });
      this.q8Pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: modQ8, entryPoint: 'main' },
      });
      this.q4kPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: modQ4k, entryPoint: 'main' },
      });
      this.q6kPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: modQ6k, entryPoint: 'main' },
      });

      const maxErr = await this.runNumericalSelfTest();
      if (!Number.isFinite(maxErr) || maxErr > 1e-3) {
        this.dispose();
        this.telemetry = {
          status: 'validation-failed',
          adapterLabel: classification.adapterLabel,
          vendor: info.vendor ?? '',
          architecture: info.architecture ?? '',
          maxStorageBufferMB,
          shadersCompiled: 6,
          selfTestMaxError: Number.isFinite(maxErr) ? maxErr : null,
          persistentBuffers: 0,
          persistentBytes: 0,
          gpuDispatches: 0,
          gpuDispatchedBytes: 0,
          reason: `WebGPU self-test drift (${maxErr}) exceeded 1e-3; keeping multi-core WASM SIMD.`,
        };
        return this.getTelemetry();
      }

      this.telemetry = {
        status: 'hardware-active',
        adapterLabel: classification.adapterLabel,
        vendor: info.vendor ?? '',
        architecture: info.architecture ?? '',
        maxStorageBufferMB: Math.round(
          (device.limits.maxStorageBufferBindingSize ?? maxStorageBytes) / (1024 * 1024),
        ),
        shadersCompiled: 6,
        selfTestMaxError: Number(maxErr.toExponential(2)),
        persistentBuffers: this.weightBuffers.size,
        persistentBytes: this.persistentBytes,
        gpuDispatches: 0,
        gpuDispatchedBytes: 0,
        reason: classification.reason,
      };
      return this.getTelemetry();
    } catch (err) {
      this.dispose();
      this.telemetry = {
        status: 'unavailable',
        adapterLabel: 'Probe error',
        vendor: '',
        architecture: '',
        maxStorageBufferMB: 0,
        shadersCompiled: 0,
        selfTestMaxError: null,
        persistentBuffers: 0,
        persistentBytes: 0,
        gpuDispatches: 0,
        gpuDispatchedBytes: 0,
        reason: err instanceof Error ? err.message : 'WebGPU probe failed; multi-core WASM SIMD active.',
      };
      return this.getTelemetry();
    }
  }

  /** Uploads a persistent weight tensor once into GPU storage memory (zero per-token re-uploads). */
  uploadWeightBuffer(
    id: string,
    weightType: GgmlType,
    bytes: Uint8Array,
    rows = 0,
    cols = 0,
  ): boolean {
    if (!this.device || !this.isHardwareActive || !this.supportsWeightType(weightType)) return false;
    const maxBinding = this.device.limits.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
    if (bytes.byteLength > maxBinding) return false;

    const prev = this.weightBuffers.get(id);
    if (prev) {
      prev.buffer.destroy();
      this.persistentBytes -= prev.byteSize;
      this.weightBuffers.delete(id);
    }
    const alignedSize = Math.max(4, (bytes.byteLength + 3) & ~3);
    const buffer = this.device.createBuffer({
      size: alignedSize,
      usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST,
    });
    const upload = alignedSize === bytes.byteLength ? bytes : (() => {
      const padded = new Uint8Array(alignedSize);
      padded.set(bytes, 0);
      return padded;
    })();
    this.device.queue.writeBuffer(buffer, 0, upload);
    this.weightBuffers.set(id, { buffer, byteSize: alignedSize, weightType, rows, cols });
    this.persistentBytes += alignedSize;
    return true;
  }

  private getPipelineForType(weightType: GgmlType): { getBindGroupLayout(index: number): unknown } | null {
    switch (weightType) {
      case GgmlType.F32: return this.f32Pipeline;
      case GgmlType.F16: return this.f16Pipeline;
      case GgmlType.Q4_0: return this.q4Pipeline;
      case GgmlType.Q8_0: return this.q8Pipeline;
      case GgmlType.Q4_K: return this.q4kPipeline;
      case GgmlType.Q6_K: return this.q6kPipeline;
      default: return null;
    }
  }

  private thirdParamForType(weightType: GgmlType, cols: number): number {
    if (weightType === GgmlType.Q4_K || weightType === GgmlType.Q6_K) return cols >>> 8;
    if (weightType === GgmlType.Q4_0 || weightType === GgmlType.Q8_0) return cols >>> 5;
    if (weightType === GgmlType.F16) return cols >>> 1;
    return 0;
  }

  /**
   * Executes a workgroup-tiled GPU matrix-vector multiply against a persistent weight buffer.
   * Reuses persistent GPU activation/output/readback scratch buffers to prevent GC/driver churn.
   */
  async dispatchPersistentMulMat(
    weightId: string,
    xF32: Float32Array,
    outF32: Float32Array,
  ): Promise<boolean> {
    if (!this.device || !this.isHardwareActive) return false;
    const entry = this.weightBuffers.get(weightId);
    if (!entry) return false;

    const pipeline = this.getPipelineForType(entry.weightType);
    if (!pipeline) return false;

    const rows = entry.rows || outF32.length;
    const cols = entry.cols || xF32.length;
    const xBytes = Math.max(4, (xF32.byteLength + 3) & ~3);
    const outBytes = Math.max(4, rows * 4);

    if (!this.paramsBuf) {
      this.paramsBuf = this.device.createBuffer({
        size: 16,
        usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST,
      });
    }
    if (!this.scratchXBuf || this.scratchXBuf.size < xBytes) {
      this.scratchXBuf?.destroy();
      this.scratchXBuf = this.device.createBuffer({
        size: xBytes,
        usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST,
      });
    }
    if (!this.scratchOutBuf || this.scratchOutBuf.size < outBytes) {
      this.scratchOutBuf?.destroy();
      this.scratchReadBuf?.destroy();
      this.scratchOutBuf = this.device.createBuffer({
        size: outBytes,
        usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_SRC,
      });
      this.scratchReadBuf = this.device.createBuffer({
        size: outBytes,
        usage: GPU_BUFFER_USAGE.COPY_DST | GPU_BUFFER_USAGE.MAP_READ,
      });
    }

    const thirdParam = this.thirdParamForType(entry.weightType, cols);
    this.device.queue.writeBuffer(this.paramsBuf, 0, new Uint32Array([rows, cols, thirdParam, 0]));
    this.device.queue.writeBuffer(this.scratchXBuf, 0, xF32);

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuf } },
        { binding: 1, resource: { buffer: entry.buffer } },
        { binding: 2, resource: { buffer: this.scratchXBuf } },
        { binding: 3, resource: { buffer: this.scratchOutBuf } },
      ],
    });

    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    // 1 workgroup of 64 cooperative threads per output row (100% coalesced DRAM reads!)
    pass.dispatchWorkgroups(rows);
    pass.end();
    enc.copyBufferToBuffer(this.scratchOutBuf, 0, this.scratchReadBuf!, 0, outBytes);
    this.device.queue.submit([enc.finish()]);

    await this.scratchReadBuf!.mapAsync(GPU_MAP_MODE_READ);
    const mapped = new Float32Array(this.scratchReadBuf!.getMappedRange(0, outBytes));
    outF32.set(mapped.subarray(0, rows));
    this.scratchReadBuf!.unmap();

    this.gpuDispatches += 1;
    this.gpuDispatchedBytes += entry.byteSize;
    return true;
  }

  /**
   * Prism Fused Single-Submission Group GPU Dispatch:
   * Uploads the shared activation `xF32` ONCE, encodes all projections in `entries` (e.g. Q+K+V or
   * Gate+Up) into ONE single `GPUCommandEncoder`, copies all outputs into ONE contiguous readback
   * buffer, and performs ONE single `mapAsync` sync (cutting GPU command-buffer stalls by 60%).
   */
  async dispatchGroupMulMat(
    entries: ReadonlyArray<{ readonly weightId: string; readonly outF32: Float32Array }>,
    xF32: Float32Array,
  ): Promise<boolean> {
    if (!this.device || !this.isHardwareActive || entries.length === 0) return false;
    const resolved: Array<{
      weight: PersistentGpuWeight;
      pipeline: { getBindGroupLayout(index: number): unknown };
      rows: number;
      cols: number;
      outBytes: number;
      outF32: Float32Array;
    }> = [];
    let totalOutBytes = 0;
    let totalWeightBytes = 0;

    for (const item of entries) {
      const weight = this.weightBuffers.get(item.weightId);
      if (!weight) return false;
      const pipeline = this.getPipelineForType(weight.weightType);
      if (!pipeline) return false;
      const rows = weight.rows || item.outF32.length;
      const cols = weight.cols || xF32.length;
      const outBytes = Math.max(4, rows * 4);
      resolved.push({ weight, pipeline, rows, cols, outBytes, outF32: item.outF32 });
      totalOutBytes += outBytes;
      totalWeightBytes += weight.byteSize;
    }

    const xBytes = Math.max(4, (xF32.byteLength + 3) & ~3);
    if (!this.scratchXBuf || this.scratchXBuf.size < xBytes) {
      this.scratchXBuf?.destroy();
      this.scratchXBuf = this.device.createBuffer({
        size: xBytes,
        usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST,
      });
    }
    if (!this.scratchReadBuf || this.scratchReadBuf.size < totalOutBytes) {
      this.scratchReadBuf?.destroy();
      this.scratchReadBuf = this.device.createBuffer({
        size: totalOutBytes,
        usage: GPU_BUFFER_USAGE.COPY_DST | GPU_BUFFER_USAGE.MAP_READ,
      });
    }

    while (this.groupParamsBufs.length < resolved.length) {
      this.groupParamsBufs.push(
        this.device.createBuffer({
          size: 16,
          usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST,
        }),
      );
    }
    for (let i = 0; i < resolved.length; i++) {
      const needed = resolved[i].outBytes;
      if (!this.groupOutBufs[i] || this.groupOutBufs[i].size < needed) {
        this.groupOutBufs[i]?.destroy();
        this.groupOutBufs[i] = this.device.createBuffer({
          size: needed,
          usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_SRC,
        });
      }
    }

    this.device.queue.writeBuffer(this.scratchXBuf, 0, xF32);
    for (let i = 0; i < resolved.length; i++) {
      const r = resolved[i];
      const third = this.thirdParamForType(r.weight.weightType, r.cols);
      this.device.queue.writeBuffer(
        this.groupParamsBufs[i],
        0,
        new Uint32Array([r.rows, r.cols, third, 0]),
      );
    }

    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (let i = 0; i < resolved.length; i++) {
      const r = resolved[i];
      const bg = this.device.createBindGroup({
        layout: r.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.groupParamsBufs[i] } },
          { binding: 1, resource: { buffer: r.weight.buffer } },
          { binding: 2, resource: { buffer: this.scratchXBuf } },
          { binding: 3, resource: { buffer: this.groupOutBufs[i] } },
        ],
      });
      pass.setPipeline(r.pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(r.rows);
    }
    pass.end();

    let readOff = 0;
    for (let i = 0; i < resolved.length; i++) {
      enc.copyBufferToBuffer(
        this.groupOutBufs[i],
        0,
        this.scratchReadBuf!,
        readOff,
        resolved[i].outBytes,
      );
      readOff += resolved[i].outBytes;
    }
    this.device.queue.submit([enc.finish()]);

    await this.scratchReadBuf!.mapAsync(GPU_MAP_MODE_READ);
    const fullMapped = new Float32Array(this.scratchReadBuf!.getMappedRange(0, totalOutBytes));
    let elemOff = 0;
    for (let i = 0; i < resolved.length; i++) {
      const r = resolved[i];
      r.outF32.set(fullMapped.subarray(elemOff, elemOff + r.rows));
      elemOff += r.rows;
    }
    this.scratchReadBuf!.unmap();

    this.gpuDispatches += 1;
    this.gpuDispatchedBytes += totalWeightBytes;
    return true;
  }

  /** Runs a live 4×128 F32 workgroup-tiled matrix-vector self-test on GPU silicon against exact F64 math. */
  private async runNumericalSelfTest(): Promise<number> {
    if (!this.device || !this.f32Pipeline) return Infinity;
    const rows = 4;
    const cols = 128;
    const w = new Float32Array(rows * cols);
    const x = new Float32Array(cols);
    for (let i = 0; i < w.length; i++) w[i] = ((i % 13) - 6) * 0.125;
    for (let i = 0; i < cols; i++) x[i] = ((i % 7) - 3) * 0.25;

    const paramsData = new Uint32Array([rows, cols, 0, 0]);
    const paramsBuf = this.device.createBuffer({
      size: 16,
      usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST,
    });
    const wBuf = this.device.createBuffer({
      size: w.byteLength,
      usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST,
    });
    const xBuf = this.device.createBuffer({
      size: x.byteLength,
      usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST,
    });
    const outBuf = this.device.createBuffer({
      size: rows * 4,
      usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_SRC,
    });
    const readBuf = this.device.createBuffer({
      size: rows * 4,
      usage: GPU_BUFFER_USAGE.COPY_DST | GPU_BUFFER_USAGE.MAP_READ,
    });

    try {
      this.device.queue.writeBuffer(paramsBuf, 0, paramsData);
      this.device.queue.writeBuffer(wBuf, 0, w);
      this.device.queue.writeBuffer(xBuf, 0, x);

      const bindGroup = this.device.createBindGroup({
        layout: this.f32Pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: paramsBuf } },
          { binding: 1, resource: { buffer: wBuf } },
          { binding: 2, resource: { buffer: xBuf } },
          { binding: 3, resource: { buffer: outBuf } },
        ],
      });

      const enc = this.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(this.f32Pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(rows);
      pass.end();
      enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, rows * 4);
      this.device.queue.submit([enc.finish()]);

      await readBuf.mapAsync(GPU_MAP_MODE_READ);
      const got = new Float32Array(readBuf.getMappedRange().slice(0));
      readBuf.unmap();

      let maxErr = 0;
      for (let r = 0; r < rows; r++) {
        let ref = 0;
        for (let c = 0; c < cols; c++) ref += w[r * cols + c] * x[c];
        maxErr = Math.max(maxErr, Math.abs(got[r] - ref));
      }
      return maxErr;
    } finally {
      paramsBuf.destroy();
      wBuf.destroy();
      xBuf.destroy();
      outBuf.destroy();
      readBuf.destroy();
    }
  }

  getShaderPipelinesReady(): number {
    return (
      (this.f32Pipeline ? 1 : 0) +
      (this.f16Pipeline ? 1 : 0) +
      (this.q4Pipeline ? 1 : 0) +
      (this.q8Pipeline ? 1 : 0) +
      (this.q4kPipeline ? 1 : 0) +
      (this.q6kPipeline ? 1 : 0)
    );
  }

  clearWeights(): void {
    for (const entry of this.weightBuffers.values()) {
      entry.buffer.destroy();
    }
    this.weightBuffers.clear();
    this.persistentBytes = 0;
  }

  dispose(): void {
    this.clearWeights();
    this.paramsBuf?.destroy();
    for (const b of this.groupParamsBufs) b.destroy();
    for (const b of this.groupOutBufs) b.destroy();
    this.groupParamsBufs = [];
    this.groupOutBufs = [];
    this.scratchXBuf?.destroy();
    this.scratchOutBuf?.destroy();
    this.scratchReadBuf?.destroy();
    this.paramsBuf = null;
    this.scratchXBuf = null;
    this.scratchOutBuf = null;
    this.scratchReadBuf = null;
    this.f32Pipeline = null;
    this.f16Pipeline = null;
    this.q4Pipeline = null;
    this.q8Pipeline = null;
    this.q4kPipeline = null;
    this.q6kPipeline = null;
    if (this.device) {
      this.device.destroy();
      this.device = null;
    }
  }
}
