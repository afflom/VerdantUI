#!/usr/bin/env node
/**
 * VerdantUI Model-Catalog Performance Baseline & Scaling Benchmark (`scripts/bench-catalog.ts`)
 *
 * Measures real, deterministic multi-core Verdant WASM SIMD compute throughput across EVERY
 * model geometry in `MODEL_CATALOG` (0.5B compact -> 32B dense & 30B-A3B Sparse MoE):
 *   1. Thread-count scaling sweep (`P = 1, 2, 4, physical, logical`) to measure barrier latency
 *      and true core saturation on dense Q4_K / Q4_0 and Sparse MoE workloads.
 *   2. Exact per-layer decode (`B=1`), per-layer batched prefill (`B=4`), Sparse MoE Top-K
 *      expert execution, and LM-head vocabulary projection (`V=128K..152K`) timing, projecting
 *      full-model compute `tok/s` and effective DRAM bandwidth (GB/s) for all 11 catalog models.
 *
 * Run with: `npm run bench:catalog` (add `--json` for machine-readable output).
 */

import os from 'node:os';
import { GgmlType, getGgmlTypeTrait } from '../src/engine/gguf';
import { MODEL_CATALOG, type CatalogModel } from '../src/engine/modelCatalog';
import {
  computeMoeWorkerScratchLayout,
  ParallelMatOp,
  Q8_0_ACT_BLOCK_BYTES,
  Q8_K_ACT_BLOCK_BYTES,
  VerdantWasmArena,
  weightRowByteSize,
  type MatGroupEntry,
  type MoeWorkerScratchLayout,
} from '../src/engine/verdantWasmSimd';
import { VTensorThreadPool } from '../src/engine/vtensorThreadPool';

interface ThreadSweepResult {
  readonly workers: number;
  readonly barrierUs: number;
  readonly q4kDecodeMs: number;
  readonly q4kBandwidthGBs: number;
  readonly q40DecodeMs: number;
  readonly q40BandwidthGBs: number;
  readonly moeLayerMs: number;
  readonly speedupVsP1: number;
}

interface CatalogBenchRow {
  readonly id: string;
  readonly shortName: string;
  readonly quant: string;
  readonly isMoe: boolean;
  readonly layers: number;
  readonly activeWeightMB: number;
  readonly layerDecodeMs: number;
  readonly moeFfnMs: number;
  readonly lmHeadMs: number;
  readonly layerPrefillB4Ms: number;
  readonly projectedDecodeTokSec: number;
  readonly projectedPrefillTokSec: number;
  readonly effectiveBandwidthGBs: number;
}

function alignUp64(n: number): number {
  return (n + 63) & ~63;
}

function median(values: readonly number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >>> 1;
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) * 0.5;
}

function fillSyntheticWeights(dst: Uint8Array, seed: number): void {
  let s = seed | 0;
  const u32 = new Uint32Array(dst.buffer, dst.byteOffset, dst.byteLength >>> 2);
  for (let i = 0; i < u32.length; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) | 0;
    // Keep F16 scales finite and small (clear top exponent bits)
    u32[i] = s & 0x3bff3bff;
  }
}

function fillSyntheticF32(dst: Float32Array, seed: number): void {
  let s = seed | 0;
  for (let i = 0; i < dst.length; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) | 0;
    dst[i] = ((s >>> 16) / 32768 - 1) * 0.5;
  }
}

function runSingleDenseOrMoeLayer(
  arena: VerdantWasmArena,
  wType: GgmlType,
  dim: number,
  qDim: number,
  kvDim: number,
  ffnDim: number,
  isMoe: boolean,
  activeExperts: number,
  ptrs: {
    wQ: number;
    wK: number;
    wV: number;
    wO: number;
    wGate: number;
    wUp: number;
    wDown: number;
    x: number;
    q8: number;
    q8k: number;
    q: number;
    k: number;
    v: number;
    attnProj: number;
    ffnGate: number;
    ffnUp: number;
    ffnHid: number;
    q8Ffn: number;
    q8kFfn: number;
    ffnProj: number;
    expProj: number;
    expStrideGate: number;
    expStrideDown: number;
    moeWorkerScratch: number;
    moeLayout: MoeWorkerScratchLayout | null;
    moeGatePtrs: Int32Array;
    moeUpPtrs: Int32Array;
    moeDownPtrs: Int32Array;
    moeWeights: Float32Array;
  },
): { totalMs: number; moeMs: number } {
  const useQ8K = arena.supportsQ8KActivation(wType) && (dim & 255) === 0;
  const ffnUseQ8K = arena.supportsQ8KActivation(wType) && (ffnDim & 255) === 0;
  const qkvEntries: MatGroupEntry[] = [
    { weightType: wType, wPtr: ptrs.wQ, outPtr: ptrs.q, rows: qDim, oStrideBytes: qDim * 4 },
    { weightType: wType, wPtr: ptrs.wK, outPtr: ptrs.k, rows: kvDim, oStrideBytes: kvDim * 4 },
    { weightType: wType, wPtr: ptrs.wV, outPtr: ptrs.v, rows: kvDim, oStrideBytes: kvDim * 4 },
  ];

  const t0 = performance.now();

  // 1. QKV fused projection
  if (useQ8K) {
    arena.quantizeActivationQ8K(ptrs.x, ptrs.q8k, dim);
    arena.mulMatQ8KGroup(qkvEntries, ptrs.q8k, ptrs.x, dim);
  } else {
    arena.quantizeActivation(ptrs.x, ptrs.q8, dim);
    arena.mulMatQ8Group(qkvEntries, ptrs.q8, ptrs.x, dim, true);
  }

  // 2. O projection
  const oUseQ8K = arena.supportsQ8KActivation(wType) && (qDim & 255) === 0;
  if (oUseQ8K) {
    arena.quantizeActivationQ8K(ptrs.q, ptrs.q8k, qDim);
    arena.mulMatQ8K(wType, ptrs.wO, ptrs.q8k, ptrs.q, ptrs.attnProj, qDim, dim);
  } else {
    arena.quantizeActivation(ptrs.q, ptrs.q8, qDim);
    arena.mulMatQ8(wType, ptrs.wO, ptrs.q8, ptrs.q, ptrs.attnProj, qDim, dim);
  }
  arena.kernels.add_f32(ptrs.x, ptrs.attnProj, ptrs.x, dim);

  // 3. FFN (Dense or Sparse MoE Top-K)
  const tFfn0 = performance.now();
  if (useQ8K) {
    arena.quantizeActivationQ8K(ptrs.x, ptrs.q8k, dim);
  } else {
    arena.quantizeActivation(ptrs.x, ptrs.q8, dim);
  }

  if (isMoe) {
    // Mirrors llamaRuntime.ts executeSparseMoeSingleRouted: single-barrier OP_MOE_EXPERTS when pooled
    const dispatched =
      ptrs.moeLayout !== null &&
      ptrs.moeWorkerScratch >= 0 &&
      arena.dispatchMoeExpertsParallel({
        expertCount: activeExperts,
        dim,
        expFfnDim: ffnDim,
        gateUpOp: useQ8K ? ParallelMatOp.MUL_MAT_Q8K : ParallelMatOp.MUL_MAT_Q8,
        downOp: ffnUseQ8K ? ParallelMatOp.MUL_MAT_Q8K : ParallelMatOp.MUL_MAT_Q8,
        gateType: wType,
        upType: wType,
        downType: wType,
        xOrQ8Ptr: useQ8K ? ptrs.q8k : ptrs.q8,
        xF32Ptr: ptrs.x,
        outPtr: ptrs.ffnProj,
        workerScratchBase: ptrs.moeWorkerScratch,
        layout: ptrs.moeLayout,
        gatePtrs: ptrs.moeGatePtrs,
        upPtrs: ptrs.moeUpPtrs,
        downPtrs: ptrs.moeDownPtrs,
        weights: ptrs.moeWeights,
      });
    if (!dispatched) {
      arena.kernels.zero_bytes(ptrs.ffnProj, dim * 4);
      for (let e = 0; e < activeExperts; e++) {
        const ge: MatGroupEntry[] = [
          { weightType: wType, wPtr: ptrs.wGate + e * ptrs.expStrideGate, outPtr: ptrs.ffnGate, rows: ffnDim, oStrideBytes: ffnDim * 4 },
          { weightType: wType, wPtr: ptrs.wUp + e * ptrs.expStrideGate, outPtr: ptrs.ffnUp, rows: ffnDim, oStrideBytes: ffnDim * 4 },
        ];
        const dPtr = ptrs.wDown + e * ptrs.expStrideDown;
        if (useQ8K) {
          arena.mulMatQ8KGroup(ge, ptrs.q8k, ptrs.x, dim);
        } else {
          arena.mulMatQ8Group(ge, ptrs.q8, ptrs.x, dim, true);
        }
        arena.kernels.swiglu_f32(ptrs.ffnGate, ptrs.ffnUp, ptrs.ffnHid, ffnDim);
        if (ffnUseQ8K) {
          arena.quantizeActivationQ8K(ptrs.ffnHid, ptrs.q8kFfn, ffnDim);
          arena.mulMatQ8K(wType, dPtr, ptrs.q8kFfn, ptrs.ffnHid, ptrs.expProj, ffnDim, dim);
        } else {
          arena.quantizeActivation(ptrs.ffnHid, ptrs.q8Ffn, ffnDim);
          arena.mulMatQ8(wType, dPtr, ptrs.q8Ffn, ptrs.ffnHid, ptrs.expProj, ffnDim, dim);
        }
        arena.kernels.axpy_f32(ptrs.ffnProj, ptrs.expProj, 1 / activeExperts, dim);
      }
    }
  } else {
    const ge: MatGroupEntry[] = [
      { weightType: wType, wPtr: ptrs.wGate, outPtr: ptrs.ffnGate, rows: ffnDim, oStrideBytes: ffnDim * 4 },
      { weightType: wType, wPtr: ptrs.wUp, outPtr: ptrs.ffnUp, rows: ffnDim, oStrideBytes: ffnDim * 4 },
    ];
    if (useQ8K) {
      arena.mulMatQ8KGroup(ge, ptrs.q8k, ptrs.x, dim);
    } else {
      arena.mulMatQ8Group(ge, ptrs.q8, ptrs.x, dim, true);
    }
    arena.kernels.swiglu_f32(ptrs.ffnGate, ptrs.ffnUp, ptrs.ffnHid, ffnDim);
    if (ffnUseQ8K) {
      arena.quantizeActivationQ8K(ptrs.ffnHid, ptrs.q8kFfn, ffnDim);
      arena.mulMatQ8K(wType, ptrs.wDown, ptrs.q8kFfn, ptrs.ffnHid, ptrs.ffnProj, ffnDim, dim);
    } else {
      arena.quantizeActivation(ptrs.ffnHid, ptrs.q8Ffn, ffnDim);
      arena.mulMatQ8(wType, ptrs.wDown, ptrs.q8Ffn, ptrs.ffnHid, ptrs.ffnProj, ffnDim, dim);
    }
  }
  const moeMs = performance.now() - tFfn0;
  arena.kernels.add_f32(ptrs.x, ptrs.ffnProj, ptrs.x, dim);
  const totalMs = performance.now() - t0;
  return { totalMs, moeMs };
}

function runBatch4DenseOrMoeLayer(
  arena: VerdantWasmArena,
  wType: GgmlType,
  dim: number,
  qDim: number,
  kvDim: number,
  ffnDim: number,
  isMoe: boolean,
  activeExperts: number,
  ptrs: {
    wQ: number;
    wK: number;
    wV: number;
    wO: number;
    wGate: number;
    wUp: number;
    wDown: number;
    xB4: number;
    q8B4: number;
    q8kB4: number;
    qB4: number;
    kB4: number;
    vB4: number;
    attnProjB4: number;
    ffnGateB4: number;
    ffnUpB4: number;
    ffnHidB4: number;
    q8FfnB4: number;
    q8kFfnB4: number;
    ffnProjB4: number;
    expStrideGate: number;
    expStrideDown: number;
  },
): number {
  const B = 4;
  const useQ8K = arena.supportsQ8KActivation(wType) && (dim & 255) === 0;
  const ffnUseQ8K = arena.supportsQ8KActivation(wType) && (ffnDim & 255) === 0;
  const qkDimStride = (dim >>> 8) * Q8_K_ACT_BLOCK_BYTES;
  const qDimStride = (dim >>> 5) * Q8_0_ACT_BLOCK_BYTES;
  const qkFfnStride = (ffnDim >>> 8) * Q8_K_ACT_BLOCK_BYTES;
  const qFfnStride = (ffnDim >>> 5) * Q8_0_ACT_BLOCK_BYTES;

  const qkvEntries: MatGroupEntry[] = [
    { weightType: wType, wPtr: ptrs.wQ, outPtr: ptrs.qB4, rows: qDim, oStrideBytes: qDim * 4 },
    { weightType: wType, wPtr: ptrs.wK, outPtr: ptrs.kB4, rows: kvDim, oStrideBytes: kvDim * 4 },
    { weightType: wType, wPtr: ptrs.wV, outPtr: ptrs.vB4, rows: kvDim, oStrideBytes: kvDim * 4 },
  ];

  const t0 = performance.now();
  if (useQ8K) {
    for (let t = 0; t < B; t++) {
      arena.quantizeActivationQ8K(ptrs.xB4 + t * dim * 4, ptrs.q8kB4 + t * qkDimStride, dim);
    }
    arena.mulMatBatch4Q8KGroup(qkvEntries, ptrs.q8kB4, ptrs.xB4, dim);
  } else {
    for (let t = 0; t < B; t++) {
      arena.quantizeActivation(ptrs.xB4 + t * dim * 4, ptrs.q8B4 + t * qDimStride, dim);
    }
    arena.mulMatBatch4Q8Group(qkvEntries, ptrs.q8B4, ptrs.xB4, dim, true);
  }

  const oUseQ8K = arena.supportsQ8KActivation(wType) && (qDim & 255) === 0;
  if (oUseQ8K) {
    const qkQStride = (qDim >>> 8) * Q8_K_ACT_BLOCK_BYTES;
    for (let t = 0; t < B; t++) {
      arena.quantizeActivationQ8K(ptrs.qB4 + t * qDim * 4, ptrs.q8kB4 + t * qkQStride, qDim);
    }
    arena.mulMatBatchQ8K(wType, ptrs.wO, ptrs.q8kB4, ptrs.qB4, ptrs.attnProjB4, qDim, dim, B);
  } else {
    const qQStride = (qDim >>> 5) * Q8_0_ACT_BLOCK_BYTES;
    for (let t = 0; t < B; t++) {
      arena.quantizeActivation(ptrs.qB4 + t * qDim * 4, ptrs.q8B4 + t * qQStride, qDim);
    }
    arena.mulMatBatchQ8(wType, ptrs.wO, ptrs.q8B4, ptrs.qB4, ptrs.attnProjB4, qDim, dim, B);
  }

  if (useQ8K) {
    for (let t = 0; t < B; t++) {
      arena.quantizeActivationQ8K(ptrs.xB4 + t * dim * 4, ptrs.q8kB4 + t * qkDimStride, dim);
    }
  } else {
    for (let t = 0; t < B; t++) {
      arena.quantizeActivation(ptrs.xB4 + t * dim * 4, ptrs.q8B4 + t * qDimStride, dim);
    }
  }

  const nPasses = isMoe ? activeExperts : 1;
  for (let e = 0; e < nPasses; e++) {
    const ge: MatGroupEntry[] = [
      { weightType: wType, wPtr: ptrs.wGate + e * ptrs.expStrideGate, outPtr: ptrs.ffnGateB4, rows: ffnDim, oStrideBytes: ffnDim * 4 },
      { weightType: wType, wPtr: ptrs.wUp + e * ptrs.expStrideGate, outPtr: ptrs.ffnUpB4, rows: ffnDim, oStrideBytes: ffnDim * 4 },
    ];
    const dPtr = ptrs.wDown + e * ptrs.expStrideDown;
    if (useQ8K) {
      arena.mulMatBatch4Q8KGroup(ge, ptrs.q8kB4, ptrs.xB4, dim);
    } else {
      arena.mulMatBatch4Q8Group(ge, ptrs.q8B4, ptrs.xB4, dim, true);
    }
    arena.kernels.swiglu_f32(ptrs.ffnGateB4, ptrs.ffnUpB4, ptrs.ffnHidB4, B * ffnDim);
    if (ffnUseQ8K) {
      for (let t = 0; t < B; t++) {
        arena.quantizeActivationQ8K(ptrs.ffnHidB4 + t * ffnDim * 4, ptrs.q8kFfnB4 + t * qkFfnStride, ffnDim);
      }
      arena.mulMatBatchQ8K(wType, dPtr, ptrs.q8kFfnB4, ptrs.ffnHidB4, ptrs.ffnProjB4, ffnDim, dim, B);
    } else {
      for (let t = 0; t < B; t++) {
        arena.quantizeActivation(ptrs.ffnHidB4 + t * ffnDim * 4, ptrs.q8FfnB4 + t * qFfnStride, ffnDim);
      }
      arena.mulMatBatchQ8(wType, dPtr, ptrs.q8FfnB4, ptrs.ffnHidB4, ptrs.ffnProjB4, ffnDim, dim, B);
    }
  }
  return performance.now() - t0;
}

async function runThreadSweep(): Promise<{ bestWorkers: number; results: ThreadSweepResult[] }> {
  const logicalCpus = Math.max(1, os.cpus().length);
  const candidates = Array.from(
    new Set([1, 2, 4, Math.floor(logicalCpus / 2), 6, 8, logicalCpus].filter((c) => c >= 1 && c <= logicalCpus)),
  ).sort((a, b) => a - b);

  // Allocate a single shared arena large enough for 14B/32B single-projection + MoE sweep
  const arena = new VerdantWasmArena({ initialPages: 2048, shared: true }); // 128 MiB
  const cols = 4096;
  const rows = 12288; // Qwen3 8B FFN gate
  const q4kBytes = weightRowByteSize(GgmlType.Q4_K, cols) * rows;
  const q40Bytes = weightRowByteSize(GgmlType.Q4_0, cols) * rows;

  const wQ4kPtr = arena.allocRegion(q4kBytes);
  const wQ40Ptr = arena.allocRegion(q40Bytes);
  const xPtr = arena.allocRegion(cols * 4);
  const q8Ptr = arena.allocRegion((cols >>> 5) * Q8_0_ACT_BLOCK_BYTES);
  const q8kPtr = arena.allocRegion((cols >>> 8) * Q8_K_ACT_BLOCK_BYTES);
  const outPtr = arena.allocRegion(rows * 4);

  // Qwen3-30B-A3B MoE expert slice: 8 experts × (gate+up [768×2048] + down [2048×768]) in Q4_K
  const mDim = 2048;
  const mFfn = 768;
  const mExpK = 8;
  const mGateBytes = alignUp64(weightRowByteSize(GgmlType.Q4_K, mDim) * mFfn);
  const mDownBytes = alignUp64(weightRowByteSize(GgmlType.Q4_K, mFfn) * mDim);
  const mGatePtr = arena.allocRegion(mGateBytes * mExpK);
  const mUpPtr = arena.allocRegion(mGateBytes * mExpK);
  const mDownPtr = arena.allocRegion(mDownBytes * mExpK);
  const mGateOut = arena.allocRegion(mFfn * 4);
  const mUpOut = arena.allocRegion(mFfn * 4);
  const mHid = arena.allocRegion(mFfn * 4);
  const mQ8kFfn = arena.allocRegion((mFfn >>> 8) * Q8_K_ACT_BLOCK_BYTES);
  const mProj = arena.allocRegion(mDim * 4);
  const mExpProj = arena.allocRegion(mDim * 4);
  const mLayout = computeMoeWorkerScratchLayout(mDim, mFfn);
  const mWorkerScratch = arena.allocRegion(mLayout.strideBytes * Math.max(1, logicalCpus));
  const mGatePtrs = new Int32Array(mExpK);
  const mUpPtrs = new Int32Array(mExpK);
  const mDownPtrs = new Int32Array(mExpK);
  const mWeights = new Float32Array(mExpK);
  for (let e = 0; e < mExpK; e++) {
    mGatePtrs[e] = mGatePtr + e * mGateBytes;
    mUpPtrs[e] = mUpPtr + e * mGateBytes;
    mDownPtrs[e] = mDownPtr + e * mDownBytes;
    mWeights[e] = 1 / mExpK;
  }

  fillSyntheticWeights(arena.bytes().subarray(wQ4kPtr, wQ4kPtr + q4kBytes), 11);
  fillSyntheticWeights(arena.bytes().subarray(wQ40Ptr, wQ40Ptr + q40Bytes), 22);
  fillSyntheticWeights(arena.bytes().subarray(mGatePtr, mGatePtr + mGateBytes * mExpK), 33);
  fillSyntheticWeights(arena.bytes().subarray(mUpPtr, mUpPtr + mGateBytes * mExpK), 44);
  fillSyntheticWeights(arena.bytes().subarray(mDownPtr, mDownPtr + mDownBytes * mExpK), 55);
  fillSyntheticF32(new Float32Array(arena.memory.buffer, xPtr, cols), 77);
  arena.quantizeActivation(xPtr, q8Ptr, cols);
  arena.quantizeActivationQ8K(xPtr, q8kPtr, cols);

  const results: ThreadSweepResult[] = [];
  let p1Q4kMs = 0;

  for (const w of candidates) {
    let pool: VTensorThreadPool | null = null;
    if (w > 1) {
      pool = await VTensorThreadPool.create(arena, { maxWorkers: w, parallelThresholdBytes: 32 * 1024 });
      arena.attachThreadPool(pool);
    } else {
      arena.attachThreadPool(null);
    }

    // Warmup
    for (let i = 0; i < 2; i++) {
      arena.mulMatQ8K(GgmlType.Q4_K, wQ4kPtr, q8kPtr, xPtr, outPtr, cols, rows);
      arena.mulMatQ8(GgmlType.Q4_0, wQ40Ptr, q8Ptr, xPtr, outPtr, cols, rows);
    }

    const iters = 6;
    const t0 = performance.now();
    for (let i = 0; i < iters; i++) {
      arena.mulMatQ8K(GgmlType.Q4_K, wQ4kPtr, q8kPtr, xPtr, outPtr, cols, rows);
    }
    const q4kDecodeMs = (performance.now() - t0) / iters;

    const t1 = performance.now();
    for (let i = 0; i < iters; i++) {
      arena.mulMatQ8(GgmlType.Q4_0, wQ40Ptr, q8Ptr, xPtr, outPtr, cols, rows);
    }
    const q40DecodeMs = (performance.now() - t1) / iters;

    // MoE Top-8 layer benchmark (single-barrier OP_MOE_EXPERTS when pooled, WASM SIMD axpy otherwise)
    const t2 = performance.now();
    for (let it = 0; it < iters; it++) {
      const dispatched = arena.dispatchMoeExpertsParallel({
        expertCount: mExpK,
        dim: mDim,
        expFfnDim: mFfn,
        gateUpOp: ParallelMatOp.MUL_MAT_Q8K,
        downOp: ParallelMatOp.MUL_MAT_Q8K,
        gateType: GgmlType.Q4_K,
        upType: GgmlType.Q4_K,
        downType: GgmlType.Q4_K,
        xOrQ8Ptr: q8kPtr,
        xF32Ptr: xPtr,
        outPtr: mProj,
        workerScratchBase: mWorkerScratch,
        layout: mLayout,
        gatePtrs: mGatePtrs,
        upPtrs: mUpPtrs,
        downPtrs: mDownPtrs,
        weights: mWeights,
      });
      if (!dispatched) {
        arena.kernels.zero_bytes(mProj, mDim * 4);
        for (let e = 0; e < mExpK; e++) {
          const ge: MatGroupEntry[] = [
            { weightType: GgmlType.Q4_K, wPtr: mGatePtr + e * mGateBytes, outPtr: mGateOut, rows: mFfn, oStrideBytes: mFfn * 4 },
            { weightType: GgmlType.Q4_K, wPtr: mUpPtr + e * mGateBytes, outPtr: mUpOut, rows: mFfn, oStrideBytes: mFfn * 4 },
          ];
          arena.mulMatQ8KGroup(ge, q8kPtr, xPtr, mDim);
          arena.kernels.swiglu_f32(mGateOut, mUpOut, mHid, mFfn);
          arena.quantizeActivationQ8K(mHid, mQ8kFfn, mFfn);
          arena.mulMatQ8K(GgmlType.Q4_K, mDownPtr + e * mDownBytes, mQ8kFfn, mHid, mExpProj, mFfn, mDim);
          arena.kernels.axpy_f32(mProj, mExpProj, 0.125, mDim);
        }
      }
    }
    const moeLayerMs = (performance.now() - t2) / iters;

    if (w === 1) p1Q4kMs = q4kDecodeMs;

    results.push({
      workers: pool ? pool.workerCount : 1,
      barrierUs: pool ? pool.barrierLatencyUs : 0,
      q4kDecodeMs,
      q4kBandwidthGBs: q4kBytes / (q4kDecodeMs * 1e6),
      q40DecodeMs,
      q40BandwidthGBs: q40Bytes / (q40DecodeMs * 1e6),
      moeLayerMs,
      speedupVsP1: p1Q4kMs / Math.max(0.001, q4kDecodeMs),
    });

    if (pool) {
      arena.attachThreadPool(null);
      await pool.disposeAsync();
    }
  }

  let best = results[0];
  for (const r of results) {
    if (r.q4kDecodeMs < best.q4kDecodeMs) best = r;
  }
  return { bestWorkers: best.workers, results };
}

async function benchmarkModel(model: CatalogModel, workers: number): Promise<CatalogBenchRow> {
  const g = model.geometry;
  // Real GGUF models with dim % 256 !== 0 (e.g. Qwen2.5 0.5B dim=896) store those projections in Q8_0/Q4_0 (32-elem blocks)
  const wType =
    g.dominantGgmlType === GgmlType.Q4_K && (g.embeddingLength & 255) !== 0
      ? GgmlType.Q4_0
      : g.dominantGgmlType;
  const dim = g.embeddingLength;
  const qDim = g.headCount * g.headDim;
  const kvDim = g.headCountKv * g.headDim;
  const ffnDim = g.isMoe ? g.expertFeedForwardLength : g.feedForwardLength;
  const activeExperts = g.isMoe ? g.expertUsedCount : 1;

  // Size arena for 1 full layer + LM head tile (4096 rows, scaled to full vocabSize)
  const lmTileRows = Math.min(g.vocabSize, 4096);
  const wQBytes = alignUp64(weightRowByteSize(wType, dim) * qDim);
  const wKVBytes = alignUp64(weightRowByteSize(wType, dim) * kvDim);
  const wOBytes = alignUp64(weightRowByteSize(wType, qDim) * dim);
  const expStrideGate = alignUp64(weightRowByteSize(wType, dim) * ffnDim);
  const expStrideDown = alignUp64(weightRowByteSize(wType, ffnDim) * dim);
  const wLmTileBytes = alignUp64(weightRowByteSize(wType, dim) * lmTileRows);

  const totalWeightsBytes =
    wQBytes +
    wKVBytes * 2 +
    wOBytes +
    (expStrideGate * 2 + expStrideDown) * activeExperts +
    wLmTileBytes;
  const scratchBytes =
    alignUp64((dim + qDim + kvDim * 2 + ffnDim * 3 + lmTileRows) * 4 * 5) + 4 * 1024 * 1024;
  const neededPages = Math.ceil((totalWeightsBytes + scratchBytes + 2 * 1024 * 1024) / 65536);

  const arena = new VerdantWasmArena({ initialPages: neededPages, shared: workers > 1 });
  const pool =
    workers > 1
      ? await VTensorThreadPool.create(arena, { maxWorkers: workers, parallelThresholdBytes: 32 * 1024 })
      : null;
  arena.attachThreadPool(pool);

  try {
    const wQ = arena.allocRegion(wQBytes);
    const wK = arena.allocRegion(wKVBytes);
    const wV = arena.allocRegion(wKVBytes);
    const wO = arena.allocRegion(wOBytes);
    const wGate = arena.allocRegion(expStrideGate * activeExperts);
    const wUp = arena.allocRegion(expStrideGate * activeExperts);
    const wDown = arena.allocRegion(expStrideDown * activeExperts);
    const wLm = arena.allocRegion(wLmTileBytes);

    fillSyntheticWeights(arena.bytes().subarray(wQ, wQ + wQBytes), 1);
    fillSyntheticWeights(arena.bytes().subarray(wK, wK + wKVBytes), 2);
    fillSyntheticWeights(arena.bytes().subarray(wV, wV + wKVBytes), 3);
    fillSyntheticWeights(arena.bytes().subarray(wO, wO + wOBytes), 4);
    fillSyntheticWeights(arena.bytes().subarray(wGate, wGate + expStrideGate * activeExperts), 5);
    fillSyntheticWeights(arena.bytes().subarray(wUp, wUp + expStrideGate * activeExperts), 6);
    fillSyntheticWeights(arena.bytes().subarray(wDown, wDown + expStrideDown * activeExperts), 7);
    fillSyntheticWeights(arena.bytes().subarray(wLm, wLm + wLmTileBytes), 8);

    const maxActCols = Math.max(dim, qDim);
    const x = arena.allocRegion(dim * 4);
    const q8 = arena.allocRegion((maxActCols >>> 5) * Q8_0_ACT_BLOCK_BYTES);
    const q8k = arena.allocRegion(Math.max(1, maxActCols >>> 8) * Q8_K_ACT_BLOCK_BYTES);
    const q = arena.allocRegion(qDim * 4);
    const k = arena.allocRegion(kvDim * 4);
    const v = arena.allocRegion(kvDim * 4);
    const attnProj = arena.allocRegion(dim * 4);
    const ffnGate = arena.allocRegion(ffnDim * 4);
    const ffnUp = arena.allocRegion(ffnDim * 4);
    const ffnHid = arena.allocRegion(ffnDim * 4);
    const q8Ffn = arena.allocRegion((ffnDim >>> 5) * Q8_0_ACT_BLOCK_BYTES);
    const q8kFfn = arena.allocRegion(Math.max(1, ffnDim >>> 8) * Q8_K_ACT_BLOCK_BYTES);
    const ffnProj = arena.allocRegion(dim * 4);
    const expProj = arena.allocRegion(dim * 4);
    const logitsTile = arena.allocRegion(lmTileRows * 4);
    const moeLayout = g.isMoe ? computeMoeWorkerScratchLayout(dim, ffnDim) : null;
    const moeWorkerScratch =
      moeLayout !== null ? arena.allocRegion(moeLayout.strideBytes * Math.max(1, workers)) : -1;
    const moeGatePtrs = new Int32Array(activeExperts);
    const moeUpPtrs = new Int32Array(activeExperts);
    const moeDownPtrs = new Int32Array(activeExperts);
    const moeWeights = new Float32Array(activeExperts);
    for (let e = 0; e < activeExperts; e++) {
      moeGatePtrs[e] = wGate + e * expStrideGate;
      moeUpPtrs[e] = wUp + e * expStrideGate;
      moeDownPtrs[e] = wDown + e * expStrideDown;
      moeWeights[e] = 1 / activeExperts;
    }

    const xB4 = arena.allocRegion(4 * dim * 4);
    const q8B4 = arena.allocRegion(4 * (maxActCols >>> 5) * Q8_0_ACT_BLOCK_BYTES);
    const q8kB4 = arena.allocRegion(4 * Math.max(1, maxActCols >>> 8) * Q8_K_ACT_BLOCK_BYTES);
    const qB4 = arena.allocRegion(4 * qDim * 4);
    const kB4 = arena.allocRegion(4 * kvDim * 4);
    const vB4 = arena.allocRegion(4 * kvDim * 4);
    const attnProjB4 = arena.allocRegion(4 * dim * 4);
    const ffnGateB4 = arena.allocRegion(4 * ffnDim * 4);
    const ffnUpB4 = arena.allocRegion(4 * ffnDim * 4);
    const ffnHidB4 = arena.allocRegion(4 * ffnDim * 4);
    const q8FfnB4 = arena.allocRegion(4 * (ffnDim >>> 5) * Q8_0_ACT_BLOCK_BYTES);
    const q8kFfnB4 = arena.allocRegion(4 * Math.max(1, ffnDim >>> 8) * Q8_K_ACT_BLOCK_BYTES);
    const ffnProjB4 = arena.allocRegion(4 * dim * 4);

    fillSyntheticF32(new Float32Array(arena.memory.buffer, x, dim), 101);
    fillSyntheticF32(new Float32Array(arena.memory.buffer, xB4, 4 * dim), 202);

    const singlePtrs = {
      wQ, wK, wV, wO, wGate, wUp, wDown, x, q8, q8k, q, k, v, attnProj,
      ffnGate, ffnUp, ffnHid, q8Ffn, q8kFfn, ffnProj, expProj, expStrideGate, expStrideDown,
      moeWorkerScratch, moeLayout, moeGatePtrs, moeUpPtrs, moeDownPtrs, moeWeights,
    };
    const b4Ptrs = {
      wQ, wK, wV, wO, wGate, wUp, wDown, xB4, q8B4, q8kB4, qB4, kB4, vB4,
      attnProjB4, ffnGateB4, ffnUpB4, ffnHidB4, q8FfnB4, q8kFfnB4, ffnProjB4, expStrideGate, expStrideDown,
    };

    // Warmup all hot kernels (single-token, batch-4, and LM-head) across workers so TurboFan tiers up before timing
    const lmUseQ8K = arena.supportsQ8KActivation(wType) && (dim & 255) === 0;
    if (lmUseQ8K) arena.quantizeActivationQ8K(x, q8k, dim);
    else arena.quantizeActivation(x, q8, dim);
    const warmupIters = totalWeightsBytes > 40 * 1024 * 1024 ? 2 : 3;
    for (let w = 0; w < warmupIters; w++) {
      runSingleDenseOrMoeLayer(arena, wType, dim, qDim, kvDim, ffnDim, g.isMoe, activeExperts, singlePtrs);
      runBatch4DenseOrMoeLayer(arena, wType, dim, qDim, kvDim, ffnDim, g.isMoe, activeExperts, b4Ptrs);
      if (lmUseQ8K) arena.mulMatQ8K(wType, wLm, q8k, x, logitsTile, dim, lmTileRows);
      else arena.mulMatQ8(wType, wLm, q8, x, logitsTile, dim, lmTileRows);
    }

    const iters = totalWeightsBytes > 40 * 1024 * 1024 ? 5 : 7;
    const layerSamples: number[] = [];
    const moeSamples: number[] = [];
    for (let i = 0; i < iters; i++) {
      const r = runSingleDenseOrMoeLayer(arena, wType, dim, qDim, kvDim, ffnDim, g.isMoe, activeExperts, singlePtrs);
      layerSamples.push(r.totalMs);
      moeSamples.push(r.moeMs);
    }
    const layerDecodeMs = median(layerSamples);
    const moeFfnMs = median(moeSamples);

    const b4Samples: number[] = [];
    for (let i = 0; i < iters; i++) {
      b4Samples.push(runBatch4DenseOrMoeLayer(arena, wType, dim, qDim, kvDim, ffnDim, g.isMoe, activeExperts, b4Ptrs));
    }
    const layerPrefillB4Ms = median(b4Samples);

    // LM-head tile measurement scaled to full vocab
    const lmSamples: number[] = [];
    for (let i = 0; i < iters; i++) {
      const tLm0 = performance.now();
      if (lmUseQ8K) arena.mulMatQ8K(wType, wLm, q8k, x, logitsTile, dim, lmTileRows);
      else arena.mulMatQ8(wType, wLm, q8, x, logitsTile, dim, lmTileRows);
      lmSamples.push(performance.now() - tLm0);
    }
    const lmTileMs = median(lmSamples);
    const lmHeadMs = lmTileMs * (g.vocabSize / lmTileRows);

    const activeLayerBytes =
      wQBytes + wKVBytes * 2 + wOBytes + (expStrideGate * 2 + expStrideDown) * activeExperts;
    const lmFullBytes = weightRowByteSize(wType, dim) * g.vocabSize;
    const totalActiveBytes = activeLayerBytes * g.blockCount + lmFullBytes;
    const fullDecodeMs = layerDecodeMs * g.blockCount + lmHeadMs;
    const fullPrefillB4Ms = layerPrefillB4Ms * g.blockCount;

    return {
      id: model.id,
      shortName: model.shortName,
      quant: getGgmlTypeTrait(wType).name,
      isMoe: g.isMoe,
      layers: g.blockCount,
      activeWeightMB: Number((totalActiveBytes / (1024 * 1024)).toFixed(1)),
      layerDecodeMs: Number(layerDecodeMs.toFixed(2)),
      moeFfnMs: Number(moeFfnMs.toFixed(2)),
      lmHeadMs: Number(lmHeadMs.toFixed(2)),
      layerPrefillB4Ms: Number(layerPrefillB4Ms.toFixed(2)),
      projectedDecodeTokSec: Number((1000 / Math.max(0.01, fullDecodeMs)).toFixed(2)),
      projectedPrefillTokSec: Number((4000 / Math.max(0.01, fullPrefillB4Ms)).toFixed(2)),
      effectiveBandwidthGBs: Number((totalActiveBytes / (fullDecodeMs * 1e6)).toFixed(2)),
    };
  } finally {
    arena.attachThreadPool(null);
    if (pool) await pool.disposeAsync();
  }
}

async function main(): Promise<void> {
  const jsonMode = process.argv.includes('--json');
  const workersArgIdx = process.argv.indexOf('--workers');
  const forcedWorkers =
    workersArgIdx >= 0 && process.argv[workersArgIdx + 1]
      ? Number(process.argv[workersArgIdx + 1])
      : undefined;

  const cpuInfo = os.cpus()[0]?.model ?? 'Unknown CPU';
  const logicalCpus = os.cpus().length;
  const sweep = await runThreadSweep();
  const targetWorkers = forcedWorkers ?? sweep.bestWorkers;

  const rows: CatalogBenchRow[] = [];
  for (const model of MODEL_CATALOG) {
    rows.push(await benchmarkModel(model, targetWorkers));
  }

  if (jsonMode) {
    console.log(
      JSON.stringify(
        {
          cpu: cpuInfo,
          logicalCpus,
          selectedWorkers: targetWorkers,
          threadSweep: sweep.results,
          catalog: rows,
        },
        null,
        2,
      ),
    );
    return;
  }

  console.log(`VerdantUI Catalog Performance Baseline`);
  console.log(`Host: ${cpuInfo} (${logicalCpus} logical CPUs) · Selected P=${targetWorkers}\n`);

  console.log('=== 1. Multi-Core Thread Scaling Sweep (Q4_K/Q4_0 [12288×4096] + Qwen3-30B-A3B MoE Top-8) ===');
  console.log(
    'P'.padStart(3),
    'Barrier'.padStart(9),
    'Q4_K ms'.padStart(9),
    'Q4_K GB/s'.padStart(10),
    'Q4_0 ms'.padStart(9),
    'Q4_0 GB/s'.padStart(10),
    'MoE8 ms'.padStart(9),
    'vs P=1'.padStart(8),
  );
  for (const r of sweep.results) {
    console.log(
      String(r.workers).padStart(3),
      `${r.barrierUs.toFixed(1)}us`.padStart(9),
      r.q4kDecodeMs.toFixed(2).padStart(9),
      r.q4kBandwidthGBs.toFixed(1).padStart(10),
      r.q40DecodeMs.toFixed(2).padStart(9),
      r.q40BandwidthGBs.toFixed(1).padStart(10),
      r.moeLayerMs.toFixed(2).padStart(9),
      `${r.speedupVsP1.toFixed(2)}x`.padStart(8),
    );
  }

  console.log(`\n=== 2. All ${rows.length} Catalog Models (Compute Throughput at P=${targetWorkers}) ===`);
  console.log(
    'Model'.padEnd(28),
    'Type'.padEnd(6),
    'ActMB'.padStart(7),
    'Layer1'.padStart(8),
    'FFN/MoE'.padStart(8),
    'LMHead'.padStart(8),
    'Decode'.padStart(10),
    'PrefillB4'.padStart(11),
    'BW GB/s'.padStart(8),
  );
  for (const r of rows) {
    console.log(
      r.shortName.padEnd(28),
      r.quant.padEnd(6),
      r.activeWeightMB.toFixed(0).padStart(7),
      `${r.layerDecodeMs.toFixed(2)}ms`.padStart(8),
      `${r.moeFfnMs.toFixed(2)}ms`.padStart(8),
      `${r.lmHeadMs.toFixed(1)}ms`.padStart(8),
      `${r.projectedDecodeTokSec.toFixed(2)} t/s`.padStart(10),
      `${r.projectedPrefillTokSec.toFixed(2)} t/s`.padStart(11),
      r.effectiveBandwidthGBs.toFixed(1).padStart(8),
    );
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
