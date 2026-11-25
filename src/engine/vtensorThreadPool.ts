/**
 * vTensor Zero-Copy Multi-Core Shared-WASM SIMD Thread Pool (Prism Theorem T3 / VT-59).
 *
 * Eliminates the single-core scaling cliff on 135M–3B+ parameter models by instantiating the
 * exact same Verdant WASM Relaxed-SIMD / SIMD128 binary across `P = min(8, hardwareConcurrency)`
 * persistent compute workers sharing a single `WebAssembly.Memory({ shared: true })` arena.
 *
 * Architectural properties:
 * - ZERO weight copies, ZERO activation copies, ZERO per-op `postMessage` allocations.
 * - Cache-line-isolated (64-byte strided) `Int32Array` atomic control block prevents false sharing.
 * - Theorem T3 deterministic 4-row-aligned striping (`partitionRowStripes`) guarantees bit-exact
 *   output parity with `P = 1` across every GGML format and both single-token and 4-token prefill.
 * - Automatically bypassed for small matrices (`weightBytes < W_par*(P)`) so tiny models pay 0 us
 *   synchronization overhead while 135M–3B matrices scale linearly across CPU cores.
 */

import { GgmlType } from './gguf';
import {
  executeWasmMatStripe,
  ParallelMatOp,
  VerdantWasmArena,
  WASM_ATTN_WORKER_SCRATCH_BASE,
  WASM_ATTN_WORKER_SCRATCH_STRIDE,
  weightRowByteSize,
  type MatGroupEntry,
  type VerdantSimdExports,
  type VTensorThreadPoolLike,
} from './verdantWasmSimd';

export const CTRL_JOB_SEQ = 0;
export const CTRL_OP_CODE = 1;
export const CTRL_WEIGHT_TYPE = 2;
export const CTRL_W_PTR = 3;
export const CTRL_X_OR_Q8_PTR = 4;
export const CTRL_X_F32_PTR = 5;
export const CTRL_OUT_PTR = 6;
export const CTRL_COLS = 7;
export const CTRL_TOTAL_ROWS = 8;
export const CTRL_ROW_BYTES = 9;
export const CTRL_O_STRIDE_BYTES = 10;
export const CTRL_READY_COUNT = 11;
export const CTRL_GROUP_OP = 12;
export const CTRL_GROUP_COUNT = 13;
export const CTRL_SLEEPING_HELPERS = 14;
export const CTRL_COORDINATOR_WAITING = 15;

// Fused multi-projection group descriptor block (3 entries × 8 ints = 24 ints, padded to 32 ints)
export const GROUP_DESC_BASE = 16; // byte offset 64 (cache-line aligned)
export const GROUP_DESC_STRIDE = 8;
export const MAX_GROUP_ENTRIES = 3;
export const GDESC_WEIGHT_TYPE = 0;
export const GDESC_W_PTR = 1;
export const GDESC_OUT_PTR = 2;
export const GDESC_ROW_BYTES = 3;
export const GDESC_O_STRIDE_BYTES = 4;
export const GDESC_ROWS = 5;

export const WORKER_SLOT_BASE = 48; // byte offset 192 (cache-line aligned)
export const WORKER_SLOT_STRIDE = 16; // 64 bytes per worker slot (zero false sharing)
export const SLOT_ROW_START = 0;
export const SLOT_ROW_COUNT = 1;
export const SLOT_DONE_SEQ = 2;
export const SLOT_G1_ROW_START = 3;
export const SLOT_G1_ROW_COUNT = 4;
export const SLOT_G2_ROW_START = 5;
export const SLOT_G2_ROW_COUNT = 6;

export const OP_GROUP_DISPATCH = 50;
export const OP_PING = 51;
export const OP_FLASH_ATTN_RANGE = 52;
export const OP_SHUTDOWN = 99;
export const MAX_POOL_WORKERS = 32;
/** 26-int FlashAttention descriptor overlaid onto indices 16..41 (strictly < WORKER_SLOT_BASE = 48). */
export const ATTN_DESC_BASE = GROUP_DESC_BASE;

/** Active spin window (~400–800 µs) keeps helper threads hot across attention/RMSNorm gaps without futex syscalls. */
export const WORKER_ACTIVE_SPIN = 80_000;
export const COORDINATOR_JOIN_SPIN = 80_000;

/** Default Theorem T3 parallel dispatch floor (48 KiB): unlocks multi-core across 135M–3B models. */
export const DEFAULT_PARALLEL_THRESHOLD_BYTES = 48 * 1024;

export interface VTensorThreadPoolTelemetry {
  readonly active: boolean;
  readonly workerCount: number;
  readonly helperThreads: number;
  readonly sharedMemoryActive: boolean;
  readonly parallelThresholdBytes: number;
  readonly barrierLatencyUs: number;
  readonly parallelDispatches: number;
  readonly parallelBytesProcessed: number;
  readonly reason: string;
}

/**
 * Persistent compute loop executed by every helper thread (browser Web Worker or Node worker_thread).
 * Uses an adaptive active spin (~150–250 µs) before `Atomics.wait` to eliminate mid-token futex sleeps.
 */
export function runComputeWorkerLoop(
  kernels: VerdantSimdExports,
  ctrl: Int32Array,
  workerIndex: number,
): void {
  const slotBase = WORKER_SLOT_BASE + workerIndex * WORKER_SLOT_STRIDE;
  Atomics.add(ctrl, CTRL_READY_COUNT, 1);
  Atomics.notify(ctrl, CTRL_READY_COUNT, 1);

  let lastSeq = 0;
  for (;;) {
    let seq = Atomics.load(ctrl, CTRL_JOB_SEQ);
    for (let spin = 0; spin < WORKER_ACTIVE_SPIN && seq === lastSeq; spin++) {
      seq = Atomics.load(ctrl, CTRL_JOB_SEQ);
    }
    while (seq === lastSeq) {
      Atomics.add(ctrl, CTRL_SLEEPING_HELPERS, 1);
      seq = Atomics.load(ctrl, CTRL_JOB_SEQ);
      if (seq === lastSeq) {
        Atomics.wait(ctrl, CTRL_JOB_SEQ, lastSeq, 25);
        seq = Atomics.load(ctrl, CTRL_JOB_SEQ);
      }
      Atomics.sub(ctrl, CTRL_SLEEPING_HELPERS, 1);
    }
    lastSeq = seq;

    // Release-acquire invariant (ECMA-262 §29): Atomics.load(CTRL_JOB_SEQ) synchronizes-with
    // coordinator's Atomics.store(CTRL_JOB_SEQ), so all descriptor fields written prior to
    // publishJob() are safely visible via plain typed-array reads with zero redundant fences.
    const op = ctrl[CTRL_OP_CODE];
    if (op === OP_SHUTDOWN) {
      Atomics.store(ctrl, slotBase + SLOT_DONE_SEQ, seq);
      if (Atomics.load(ctrl, CTRL_COORDINATOR_WAITING) !== 0) {
        Atomics.notify(ctrl, slotBase + SLOT_DONE_SEQ, 1);
      }
      break;
    }

    if (op === OP_PING) {
      Atomics.store(ctrl, slotBase + SLOT_DONE_SEQ, seq);
      if (Atomics.load(ctrl, CTRL_COORDINATOR_WAITING) !== 0) {
        Atomics.notify(ctrl, slotBase + SLOT_DONE_SEQ, 1);
      }
      continue;
    }

    if (op === OP_FLASH_ATTN_RANGE) {
      const hCount = ctrl[slotBase + SLOT_ROW_COUNT];
      if (hCount > 0) {
        const hStart = ctrl[slotBase + SLOT_ROW_START];
        const d = ATTN_DESC_BASE;
        const scratchBase = WASM_ATTN_WORKER_SCRATCH_BASE + (workerIndex + 1) * WASM_ATTN_WORKER_SCRATCH_STRIDE;
        kernels.flash_attn_layer_range(
          ctrl[d + 0] >>> 0,
          ctrl[d + 1] >>> 0,
          ctrl[d + 2] >>> 0,
          ctrl[d + 3] >>> 0,
          ctrl[d + 4] >>> 0,
          ctrl[d + 5] >>> 0,
          ctrl[d + 6] >>> 0,
          ctrl[d + 7] >>> 0,
          ctrl[d + 8] >>> 0,
          ctrl[d + 9] >>> 0,
          ctrl[d + 10] >>> 0,
          ctrl[d + 11] >>> 0,
          ctrl[d + 12] >>> 0,
          ctrl[d + 13] >>> 0,
          ctrl[d + 14],
          ctrl[d + 15],
          ctrl[d + 16],
          ctrl[d + 17],
          ctrl[d + 18],
          ctrl[d + 19],
          ctrl[d + 20],
          ctrl[d + 21],
          ctrl[d + 22],
          ctrl[d + 23],
          ctrl[d + 24],
          ctrl[d + 25],
          hStart,
          hStart + hCount,
          scratchBase,
        );
      }
      Atomics.store(ctrl, slotBase + SLOT_DONE_SEQ, seq);
      if (Atomics.load(ctrl, CTRL_COORDINATOR_WAITING) !== 0) {
        Atomics.notify(ctrl, slotBase + SLOT_DONE_SEQ, 1);
      }
      continue;
    }

    if (op === OP_GROUP_DISPATCH) {
      const groupOp = ctrl[CTRL_GROUP_OP] as ParallelMatOp;
      const groupCount = ctrl[CTRL_GROUP_COUNT];
      const xOrQ8Ptr = ctrl[CTRL_X_OR_Q8_PTR] >>> 0;
      const xF32Ptr = ctrl[CTRL_X_F32_PTR] >>> 0;
      const cols = ctrl[CTRL_COLS];

      for (let g = 0; g < groupCount; g++) {
        const startOff = g === 0 ? SLOT_ROW_START : g === 1 ? SLOT_G1_ROW_START : SLOT_G2_ROW_START;
        const countOff = g === 0 ? SLOT_ROW_COUNT : g === 1 ? SLOT_G1_ROW_COUNT : SLOT_G2_ROW_COUNT;
        const rowCount = ctrl[slotBase + countOff];
        if (rowCount > 0) {
          const rowStart = ctrl[slotBase + startOff];
          const dBase = GROUP_DESC_BASE + g * GROUP_DESC_STRIDE;
          const weightType = ctrl[dBase + GDESC_WEIGHT_TYPE] as GgmlType;
          const wPtr = ctrl[dBase + GDESC_W_PTR] >>> 0;
          const outPtr = ctrl[dBase + GDESC_OUT_PTR] >>> 0;
          const rowBytes = ctrl[dBase + GDESC_ROW_BYTES];
          const oStrideBytes = ctrl[dBase + GDESC_O_STRIDE_BYTES];

          executeWasmMatStripe(
            kernels,
            groupOp,
            weightType,
            wPtr + rowStart * rowBytes,
            xOrQ8Ptr,
            xF32Ptr,
            outPtr + rowStart * 4,
            cols,
            rowCount,
            oStrideBytes,
          );
        }
      }

      Atomics.store(ctrl, slotBase + SLOT_DONE_SEQ, seq);
      if (Atomics.load(ctrl, CTRL_COORDINATOR_WAITING) !== 0) {
        Atomics.notify(ctrl, slotBase + SLOT_DONE_SEQ, 1);
      }
      continue;
    }

    const rowCount = ctrl[slotBase + SLOT_ROW_COUNT];
    if (rowCount > 0) {
      const rowStart = ctrl[slotBase + SLOT_ROW_START];
      const weightType = ctrl[CTRL_WEIGHT_TYPE] as GgmlType;
      const wPtr = ctrl[CTRL_W_PTR] >>> 0;
      const xOrQ8Ptr = ctrl[CTRL_X_OR_Q8_PTR] >>> 0;
      const xF32Ptr = ctrl[CTRL_X_F32_PTR] >>> 0;
      const outPtr = ctrl[CTRL_OUT_PTR] >>> 0;
      const cols = ctrl[CTRL_COLS];
      const rowBytes = ctrl[CTRL_ROW_BYTES];
      const oStrideBytes = ctrl[CTRL_O_STRIDE_BYTES];

      executeWasmMatStripe(
        kernels,
        op as ParallelMatOp,
        weightType,
        wPtr + rowStart * rowBytes,
        xOrQ8Ptr,
        xF32Ptr,
        outPtr + rowStart * 4,
        cols,
        rowCount,
        oStrideBytes,
      );
    }

    Atomics.store(ctrl, slotBase + SLOT_DONE_SEQ, seq);
    if (Atomics.load(ctrl, CTRL_COORDINATOR_WAITING) !== 0) {
      Atomics.notify(ctrl, slotBase + SLOT_DONE_SEQ, 1);
    }
  }
}

interface ManagedWorkerHandle {
  terminate(): void | Promise<number>;
}

/**
 * Zero-copy multi-core WASM SIMD coordinator pool owned by `VTensorBackendRegistry`.
 * Coordinator executes stripe 0 locally while `P - 1` persistent helper threads execute stripes `1..P-1`
 * concurrently on the shared `WebAssembly.Memory` arena.
 */
export class VTensorThreadPool implements VTensorThreadPoolLike {
  readonly workerCount: number;
  readonly helperCount: number;
  private readonly ctrlSab: SharedArrayBuffer;
  private readonly ctrl: Int32Array;
  private readonly workers: ManagedWorkerHandle[] = [];
  private readonly stripeStarts: Int32Array;
  private readonly stripeCounts: Int32Array;
  private readonly localGroupStarts = new Int32Array(MAX_GROUP_ENTRIES);
  private readonly localGroupCounts = new Int32Array(MAX_GROUP_ENTRIES);
  private active = false;
  private jobSeq = 0;
  private dispatches = 0;
  private bytesProcessed = 0;
  private thresholdBytes: number;
  private measuredBarrierUs = 3.5;
  private reason: string;

  private constructor(
    workerCount: number,
    ctrlSab: SharedArrayBuffer,
    thresholdBytes: number,
  ) {
    this.workerCount = workerCount;
    this.helperCount = Math.max(0, workerCount - 1);
    this.ctrlSab = ctrlSab;
    this.ctrl = new Int32Array(ctrlSab);
    this.stripeStarts = new Int32Array(workerCount);
    this.stripeCounts = new Int32Array(workerCount);
    this.thresholdBytes = thresholdBytes;
    this.reason = 'Initializing multi-core vTensor WASM pool…';
  }

  get isActive(): boolean {
    return this.active && this.helperCount > 0;
  }

  get parallelThresholdBytes(): number {
    return this.thresholdBytes;
  }

  get barrierLatencyUs(): number {
    return this.measuredBarrierUs;
  }

  setParallelThresholdBytes(bytes: number): void {
    if (Number.isFinite(bytes) && bytes >= 16 * 1024) {
      this.thresholdBytes = Math.min(2 * 1024 * 1024, Math.floor(bytes));
    }
  }

  get parallelDispatches(): number {
    return this.dispatches;
  }

  getTelemetry(): VTensorThreadPoolTelemetry {
    return {
      active: this.isActive,
      workerCount: this.isActive ? this.workerCount : 1,
      helperThreads: this.isActive ? this.helperCount : 0,
      sharedMemoryActive: this.isActive,
      parallelThresholdBytes: this.thresholdBytes,
      barrierLatencyUs: Number(this.measuredBarrierUs.toFixed(1)),
      parallelDispatches: this.dispatches,
      parallelBytesProcessed: this.bytesProcessed,
      reason: this.reason,
    };
  }

  /**
   * Boots `P - 1` persistent helper workers attached to `arena.memory` (`WebAssembly.Memory({ shared: true })`).
   * Works seamlessly in both Browser Web Workers (nested module workers) and Node.js (`worker_threads`).
   */
  static async create(
    arena: VerdantWasmArena,
    options?: {
      maxWorkers?: number;
      parallelThresholdBytes?: number;
      readyTimeoutMs?: number;
    },
  ): Promise<VTensorThreadPool | null> {
    if (!arena.isSharedMemory || typeof SharedArrayBuffer === 'undefined') {
      return null;
    }

    const hwCores =
      typeof navigator !== 'undefined' && typeof navigator.hardwareConcurrency === 'number'
        ? navigator.hardwareConcurrency
        : 4;
    const requestedMax = options?.maxWorkers ?? MAX_POOL_WORKERS;
    const totalWorkers = Math.max(1, Math.min(MAX_POOL_WORKERS, requestedMax, Math.floor(hwCores)));
    if (totalWorkers <= 1) {
      return null;
    }

    const ctrlSab = new SharedArrayBuffer((WORKER_SLOT_BASE + totalWorkers * WORKER_SLOT_STRIDE) * 4);
    const pool = new VTensorThreadPool(
      totalWorkers,
      ctrlSab,
      options?.parallelThresholdBytes ?? DEFAULT_PARALLEL_THRESHOLD_BYTES,
    );

    const readyTimeoutMs = options?.readyTimeoutMs ?? 3000;
    const hasBrowserWorker = typeof Worker === 'function';

    try {
      if (hasBrowserWorker) {
        for (let w = 0; w < pool.helperCount; w++) {
          const worker = new Worker(new URL('./vtensorCompute.worker.ts', import.meta.url), {
            type: 'module',
          });
          worker.postMessage({
            wasmModule: arena.compiledModule,
            memory: arena.memory,
            ctrlSab: pool.ctrlSab,
            workerIndex: w,
          });
          pool.workers.push(worker);
        }
      } else {
        // Node.js worker_threads path (used by engine unit tests & large-scale benchmarks)
        const wtSpec = 'node:' + 'worker_threads';
        const wt = (await import(/* @vite-ignore */ wtSpec)) as {
          Worker: new (
            code: string,
            opts: { eval: boolean; workerData: unknown },
          ) => ManagedWorkerHandle & { unref?: () => void };
        };
        const inlineWorkerScript = `
          const { workerData } = require('node:worker_threads');
          const { wasmModule, memory, ctrlSab, workerIndex } = workerData;
          const instance = new WebAssembly.Instance(wasmModule, { env: { memory } });
          const k = instance.exports;
          const ctrl = new Int32Array(ctrlSab);
          const slotBase = ${WORKER_SLOT_BASE} + workerIndex * ${WORKER_SLOT_STRIDE};
          Atomics.add(ctrl, ${CTRL_READY_COUNT}, 1);
          Atomics.notify(ctrl, ${CTRL_READY_COUNT}, 1);

          function runStripe(op, wt, wPtr, xOrQ8, xF32, outPtr, cols, rows, oStride) {
            if (op === ${ParallelMatOp.MUL_MAT_BATCH4_Q8K}) {
              if (wt === ${GgmlType.Q2_K}) { k.mul_mat_q2_k_q8k_batch4(wPtr, xOrQ8, outPtr, cols, rows, oStride); return; }
              if (wt === ${GgmlType.Q3_K}) { k.mul_mat_q3_k_q8k_batch4(wPtr, xOrQ8, outPtr, cols, rows, oStride); return; }
              if (wt === ${GgmlType.Q4_K}) { k.mul_mat_q4_k_q8k_batch4(wPtr, xOrQ8, outPtr, cols, rows, oStride); return; }
              if (wt === ${GgmlType.Q5_K}) { k.mul_mat_q5_k_q8k_batch4(wPtr, xOrQ8, outPtr, cols, rows, oStride); return; }
              if (wt === ${GgmlType.Q6_K}) { k.mul_mat_q6_k_q8k_batch4(wPtr, xOrQ8, outPtr, cols, rows, oStride); return; }
              if (wt === ${GgmlType.F32}) { k.mul_mat_f32_batch4(wPtr, xF32, outPtr, cols, rows, oStride); return; }
            } else if (op === ${ParallelMatOp.MUL_MAT_Q8K}) {
              if (wt === ${GgmlType.Q2_K}) { k.mul_mat_q2_k_q8k(wPtr, xOrQ8, outPtr, cols, rows); return; }
              if (wt === ${GgmlType.Q3_K}) { k.mul_mat_q3_k_q8k(wPtr, xOrQ8, outPtr, cols, rows); return; }
              if (wt === ${GgmlType.Q4_K}) { k.mul_mat_q4_k_q8k(wPtr, xOrQ8, outPtr, cols, rows); return; }
              if (wt === ${GgmlType.Q5_K}) { k.mul_mat_q5_k_q8k(wPtr, xOrQ8, outPtr, cols, rows); return; }
              if (wt === ${GgmlType.Q6_K}) { k.mul_mat_q6_k_q8k(wPtr, xOrQ8, outPtr, cols, rows); return; }
            } else if (op === ${ParallelMatOp.MUL_MAT_BATCH4_Q8}) {
              if (wt === ${GgmlType.Q4_0}) { k.mul_mat_q4_0_q8_batch4(wPtr, xOrQ8, outPtr, cols, rows, oStride); return; }
              if (wt === ${GgmlType.Q4_K}) { k.mul_mat_q4_k_q8_batch4(wPtr, xOrQ8, outPtr, cols, rows, oStride); return; }
              if (wt === ${GgmlType.Q6_K}) { k.mul_mat_q6_k_q8_batch4(wPtr, xOrQ8, outPtr, cols, rows, oStride); return; }
              if (wt === ${GgmlType.Q8_0}) { k.mul_mat_q8_0_q8_batch4(wPtr, xOrQ8, outPtr, cols, rows, oStride); return; }
              if (wt === ${GgmlType.F32}) { k.mul_mat_f32_batch4(wPtr, xF32, outPtr, cols, rows, oStride); return; }
            } else if (op === ${ParallelMatOp.MUL_MAT_BATCH4}) {
              if (wt === ${GgmlType.F32}) { k.mul_mat_f32_batch4(wPtr, xOrQ8, outPtr, cols, rows, oStride); return; }
              if (wt === ${GgmlType.Q4_0}) { k.mul_mat_q4_0_batch4(wPtr, xOrQ8, outPtr, cols, rows, oStride); return; }
            } else if (op === ${ParallelMatOp.MUL_MAT_Q8}) {
              if (wt === ${GgmlType.Q4_0}) { k.mul_mat_q4_0_q8(wPtr, xOrQ8, outPtr, cols, rows); return; }
              if (wt === ${GgmlType.Q4_1}) { k.mul_mat_q4_1_q8(wPtr, xOrQ8, outPtr, cols, rows); return; }
              if (wt === ${GgmlType.Q5_0}) { k.mul_mat_q5_0_q8(wPtr, xOrQ8, outPtr, cols, rows); return; }
              if (wt === ${GgmlType.Q5_1}) { k.mul_mat_q5_1_q8(wPtr, xOrQ8, outPtr, cols, rows); return; }
              if (wt === ${GgmlType.Q8_0}) { k.mul_mat_q8_0_q8(wPtr, xOrQ8, outPtr, cols, rows); return; }
              if (wt === ${GgmlType.Q4_K}) { k.mul_mat_q4_k_q8(wPtr, xOrQ8, outPtr, cols, rows); return; }
              if (wt === ${GgmlType.Q5_K}) { k.mul_mat_q5_k_q8(wPtr, xOrQ8, outPtr, cols, rows); return; }
              if (wt === ${GgmlType.Q6_K}) { k.mul_mat_q6_k_q8(wPtr, xOrQ8, outPtr, cols, rows); return; }
            }
            const xp = (op === ${ParallelMatOp.MUL_MAT_Q8} || op === ${ParallelMatOp.MUL_MAT_Q8K}) ? xF32 : xOrQ8;
            if (wt === ${GgmlType.F32}) { k.mul_mat_f32(wPtr, xp, outPtr, cols, rows); return; }
            if (wt === ${GgmlType.F16}) { k.mul_mat_f16(wPtr, xp, outPtr, cols, rows); return; }
            if (wt === ${GgmlType.BF16}) { k.mul_mat_bf16(wPtr, xp, outPtr, cols, rows); return; }
            if (wt === ${GgmlType.Q8_0}) { k.mul_mat_q8_0(wPtr, xp, outPtr, cols, rows); return; }
            if (wt === ${GgmlType.Q4_0}) { k.mul_mat_q4_0(wPtr, xp, outPtr, cols, rows); return; }
            if (wt === ${GgmlType.Q4_1}) { k.mul_mat_q4_1(wPtr, xp, outPtr, cols, rows); return; }
            if (wt === ${GgmlType.Q5_0}) { k.mul_mat_q5_0(wPtr, xp, outPtr, cols, rows); return; }
            if (wt === ${GgmlType.Q5_1}) { k.mul_mat_q5_1(wPtr, xp, outPtr, cols, rows); return; }
            if (wt === ${GgmlType.Q2_K}) { k.mul_mat_q2_k(wPtr, xp, outPtr, cols, rows); return; }
            if (wt === ${GgmlType.Q3_K}) { k.mul_mat_q3_k(wPtr, xp, outPtr, cols, rows); return; }
            if (wt === ${GgmlType.Q4_K}) { k.mul_mat_q4_k(wPtr, xp, outPtr, cols, rows); return; }
            if (wt === ${GgmlType.Q5_K}) { k.mul_mat_q5_k(wPtr, xp, outPtr, cols, rows); return; }
            if (wt === ${GgmlType.Q6_K}) { k.mul_mat_q6_k(wPtr, xp, outPtr, cols, rows); return; }
          }

          let lastSeq = 0;
          for (;;) {
            let seq = Atomics.load(ctrl, ${CTRL_JOB_SEQ});
            for (let spin = 0; spin < ${WORKER_ACTIVE_SPIN} && seq === lastSeq; spin++) {
              seq = Atomics.load(ctrl, ${CTRL_JOB_SEQ});
            }
            while (seq === lastSeq) {
              Atomics.add(ctrl, ${CTRL_SLEEPING_HELPERS}, 1);
              seq = Atomics.load(ctrl, ${CTRL_JOB_SEQ});
              if (seq === lastSeq) {
                Atomics.wait(ctrl, ${CTRL_JOB_SEQ}, lastSeq, 25);
                seq = Atomics.load(ctrl, ${CTRL_JOB_SEQ});
              }
              Atomics.sub(ctrl, ${CTRL_SLEEPING_HELPERS}, 1);
            }
            lastSeq = seq;
            const op = ctrl[${CTRL_OP_CODE}];
            if (op === ${OP_SHUTDOWN}) {
              Atomics.store(ctrl, slotBase + ${SLOT_DONE_SEQ}, seq);
              if (Atomics.load(ctrl, ${CTRL_COORDINATOR_WAITING}) !== 0) {
                Atomics.notify(ctrl, slotBase + ${SLOT_DONE_SEQ}, 1);
              }
              break;
            }
            if (op === ${OP_PING}) {
              Atomics.store(ctrl, slotBase + ${SLOT_DONE_SEQ}, seq);
              if (Atomics.load(ctrl, ${CTRL_COORDINATOR_WAITING}) !== 0) {
                Atomics.notify(ctrl, slotBase + ${SLOT_DONE_SEQ}, 1);
              }
              continue;
            }
            if (op === ${OP_FLASH_ATTN_RANGE}) {
              const hCount = ctrl[slotBase + ${SLOT_ROW_COUNT}];
              if (hCount > 0) {
                const hStart = ctrl[slotBase + ${SLOT_ROW_START}];
                const d = ${ATTN_DESC_BASE};
                const scratchBase = ${WASM_ATTN_WORKER_SCRATCH_BASE} + (workerIndex + 1) * ${WASM_ATTN_WORKER_SCRATCH_STRIDE};
                k.flash_attn_layer_range(
                  ctrl[d + 0] >>> 0,
                  ctrl[d + 1] >>> 0,
                  ctrl[d + 2] >>> 0,
                  ctrl[d + 3] >>> 0,
                  ctrl[d + 4] >>> 0,
                  ctrl[d + 5] >>> 0,
                  ctrl[d + 6] >>> 0,
                  ctrl[d + 7] >>> 0,
                  ctrl[d + 8] >>> 0,
                  ctrl[d + 9] >>> 0,
                  ctrl[d + 10] >>> 0,
                  ctrl[d + 11] >>> 0,
                  ctrl[d + 12] >>> 0,
                  ctrl[d + 13] >>> 0,
                  ctrl[d + 14],
                  ctrl[d + 15],
                  ctrl[d + 16],
                  ctrl[d + 17],
                  ctrl[d + 18],
                  ctrl[d + 19],
                  ctrl[d + 20],
                  ctrl[d + 21],
                  ctrl[d + 22],
                  ctrl[d + 23],
                  ctrl[d + 24],
                  ctrl[d + 25],
                  hStart,
                  hStart + hCount,
                  scratchBase,
                );
              }
              Atomics.store(ctrl, slotBase + ${SLOT_DONE_SEQ}, seq);
              if (Atomics.load(ctrl, ${CTRL_COORDINATOR_WAITING}) !== 0) {
                Atomics.notify(ctrl, slotBase + ${SLOT_DONE_SEQ}, 1);
              }
              continue;
            }
            if (op === ${OP_GROUP_DISPATCH}) {
              const groupOp = ctrl[${CTRL_GROUP_OP}];
              const groupCount = ctrl[${CTRL_GROUP_COUNT}];
              const xOrQ8 = ctrl[${CTRL_X_OR_Q8_PTR}] >>> 0;
              const xF32 = ctrl[${CTRL_X_F32_PTR}] >>> 0;
              const cols = ctrl[${CTRL_COLS}];
              for (let g = 0; g < groupCount; g++) {
                const startOff = g === 0 ? ${SLOT_ROW_START} : g === 1 ? ${SLOT_G1_ROW_START} : ${SLOT_G2_ROW_START};
                const countOff = g === 0 ? ${SLOT_ROW_COUNT} : g === 1 ? ${SLOT_G1_ROW_COUNT} : ${SLOT_G2_ROW_COUNT};
                const rowCount = ctrl[slotBase + countOff];
                if (rowCount > 0) {
                  const rowStart = ctrl[slotBase + startOff];
                  const dBase = ${GROUP_DESC_BASE} + g * ${GROUP_DESC_STRIDE};
                  const wt = ctrl[dBase + ${GDESC_WEIGHT_TYPE}];
                  const wPtr = ctrl[dBase + ${GDESC_W_PTR}] >>> 0;
                  const outPtr = ctrl[dBase + ${GDESC_OUT_PTR}] >>> 0;
                  const rowBytes = ctrl[dBase + ${GDESC_ROW_BYTES}];
                  const oStride = ctrl[dBase + ${GDESC_O_STRIDE_BYTES}];
                  runStripe(groupOp, wt, wPtr + rowStart * rowBytes, xOrQ8, xF32, outPtr + rowStart * 4, cols, rowCount, oStride);
                }
              }
              Atomics.store(ctrl, slotBase + ${SLOT_DONE_SEQ}, seq);
              if (Atomics.load(ctrl, ${CTRL_COORDINATOR_WAITING}) !== 0) {
                Atomics.notify(ctrl, slotBase + ${SLOT_DONE_SEQ}, 1);
              }
              continue;
            }
            const rowCount = ctrl[slotBase + ${SLOT_ROW_COUNT}];
            if (rowCount > 0) {
              const rowStart = ctrl[slotBase + ${SLOT_ROW_START}];
              const wt = ctrl[${CTRL_WEIGHT_TYPE}];
              const wPtr = ctrl[${CTRL_W_PTR}] >>> 0;
              const xOrQ8 = ctrl[${CTRL_X_OR_Q8_PTR}] >>> 0;
              const xF32 = ctrl[${CTRL_X_F32_PTR}] >>> 0;
              const outPtr = ctrl[${CTRL_OUT_PTR}] >>> 0;
              const cols = ctrl[${CTRL_COLS}];
              const rowBytes = ctrl[${CTRL_ROW_BYTES}];
              const oStride = ctrl[${CTRL_O_STRIDE_BYTES}];
              runStripe(op, wt, wPtr + rowStart * rowBytes, xOrQ8, xF32, outPtr + rowStart * 4, cols, rowCount, oStride);
            }
            Atomics.store(ctrl, slotBase + ${SLOT_DONE_SEQ}, seq);
            if (Atomics.load(ctrl, ${CTRL_COORDINATOR_WAITING}) !== 0) {
              Atomics.notify(ctrl, slotBase + ${SLOT_DONE_SEQ}, 1);
            }
          }
        `;
        for (let w = 0; w < pool.helperCount; w++) {
          const worker = new wt.Worker(inlineWorkerScript, {
            eval: true,
            workerData: {
              wasmModule: arena.compiledModule,
              memory: arena.memory,
              ctrlSab: pool.ctrlSab,
              workerIndex: w,
            },
          });
          worker.unref?.();
          pool.workers.push(worker);
        }
      }

      // Wait asynchronously until all helper threads instantiate the shared WASM module
      const deadline = performance.now() + readyTimeoutMs;
      while (Atomics.load(pool.ctrl, CTRL_READY_COUNT) < pool.helperCount) {
        if (performance.now() > deadline) {
          pool.dispose();
          return null;
        }
        await new Promise((resolve) => setTimeout(resolve, 4));
      }

      pool.active = true;
      pool.measuredBarrierUs = pool.measureBarrierLatencyUs();
      pool.reason = `${pool.workerCount} shared-WASM SIMD threads active (1 coordinator + ${pool.helperCount} workers · ~${pool.measuredBarrierUs.toFixed(1)} µs spin barrier).`;
      return pool;
    } catch {
      pool.dispose();
      return null;
    }
  }

  /**
   * Zero-allocation Theorem T3 deterministic row striping matching `partitionRowStripes` exactly.
   * Writes stripe ranges into preallocated `this.stripeStarts`/`this.stripeCounts` and returns active count.
   */
  private fillRowStripes(totalRows: number, rowAlignment = 4): number {
    const safeWorkers = Math.max(1, Math.min(this.workerCount, Math.max(1, totalRows)));
    if (safeWorkers === 1 || totalRows <= rowAlignment) {
      this.stripeStarts[0] = 0;
      this.stripeCounts[0] = totalRows;
      return 1;
    }
    const align = Math.max(1, rowAlignment);
    const totalGroups = Math.floor(totalRows / align);
    const remainderRows = totalRows - totalGroups * align;
    const groupsPerWorker = Math.floor(totalGroups / safeWorkers);
    const extraGroups = totalGroups % safeWorkers;

    let cursor = 0;
    let countOut = 0;
    for (let w = 0; w < safeWorkers; w++) {
      const groups = groupsPerWorker + (w < extraGroups ? 1 : 0);
      let count = groups * align;
      if (w === safeWorkers - 1) {
        count += remainderRows;
      }
      if (count > 0) {
        this.stripeStarts[countOut] = cursor;
        this.stripeCounts[countOut] = count;
        cursor += count;
        countOut += 1;
      }
    }
    return countOut;
  }

  /** Publishes `nextSeq` and wakes helpers ONLY if at least one helper entered kernel futex wait. */
  private publishJob(nextSeq: number): void {
    const ctrl = this.ctrl;
    Atomics.store(ctrl, CTRL_JOB_SEQ, nextSeq);
    if (Atomics.load(ctrl, CTRL_SLEEPING_HELPERS) > 0) {
      Atomics.notify(ctrl, CTRL_JOB_SEQ, this.helperCount);
    }
  }

  /**
   * Empirically measures hot spin-barrier round-trip latency in microseconds so Theorem T3
   * derives the true host parallel threshold `W_par*(P)` rather than a pessimistic constant.
   */
  private measureBarrierLatencyUs(): number {
    if (!this.active || this.helperCount === 0) return 0;
    const ctrl = this.ctrl;
    const pingOnce = () => {
      const nextSeq = (this.jobSeq + 1) | 0;
      this.jobSeq = nextSeq;
      ctrl[CTRL_OP_CODE] = OP_PING;
      this.publishJob(nextSeq);
      this.joinHelpers(nextSeq);
    };
    // Warm up helper threads into active spin window
    for (let i = 0; i < 6; i++) pingOnce();
    const samples = 16;
    const t0 = performance.now();
    for (let i = 0; i < samples; i++) pingOnce();
    const elapsedUs = ((performance.now() - t0) * 1000) / samples;
    return Math.max(0.4, Math.min(18, elapsedUs));
  }

  private joinHelpers(nextSeq: number): void {
    const ctrl = this.ctrl;
    for (let w = 0; w < this.helperCount; w++) {
      const doneIdx = WORKER_SLOT_BASE + w * WORKER_SLOT_STRIDE + SLOT_DONE_SEQ;
      let done = Atomics.load(ctrl, doneIdx);
      for (let spin = 0; spin < COORDINATOR_JOIN_SPIN && done !== nextSeq; spin++) {
        done = Atomics.load(ctrl, doneIdx);
      }
      while (done !== nextSeq) {
        Atomics.store(ctrl, CTRL_COORDINATOR_WAITING, 1);
        done = Atomics.load(ctrl, doneIdx);
        if (done !== nextSeq) {
          try {
            Atomics.wait(ctrl, doneIdx, done, 5);
          } catch {
            // Non-blocking fallback if ever invoked in an environment where Atomics.wait is forbidden
          }
          done = Atomics.load(ctrl, doneIdx);
        }
        Atomics.store(ctrl, CTRL_COORDINATOR_WAITING, 0);
      }
    }
  }

  /**
   * Synchronous zero-copy, zero-allocation parallel matrix dispatch using Theorem T3 deterministic
   * 4-row-aligned stripes.
   */
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
  ): void {
    if (!this.active || this.helperCount === 0) {
      runLocalStripe(0, rows);
      return;
    }

    const activeStripes = this.fillRowStripes(rows, 4);
    if (activeStripes <= 1) {
      runLocalStripe(0, rows);
      return;
    }

    const rowBytes = weightRowByteSize(weightType, cols);
    const ctrl = this.ctrl;
    const nextSeq = (this.jobSeq + 1) | 0;
    this.jobSeq = nextSeq;

    // Plain descriptor stores published by the single release fence in publishJob(nextSeq)
    ctrl[CTRL_OP_CODE] = op;
    ctrl[CTRL_WEIGHT_TYPE] = weightType;
    ctrl[CTRL_W_PTR] = wPtr | 0;
    ctrl[CTRL_X_OR_Q8_PTR] = xOrQ8Ptr | 0;
    ctrl[CTRL_X_F32_PTR] = xF32Ptr | 0;
    ctrl[CTRL_OUT_PTR] = outPtr | 0;
    ctrl[CTRL_COLS] = cols;
    ctrl[CTRL_TOTAL_ROWS] = rows;
    ctrl[CTRL_ROW_BYTES] = rowBytes;
    ctrl[CTRL_O_STRIDE_BYTES] = oStrideBytes;

    // Assign stripes 1..P-1 to helper threads 0..P-2
    for (let w = 0; w < this.helperCount; w++) {
      const slotBase = WORKER_SLOT_BASE + w * WORKER_SLOT_STRIDE;
      const stripeIdx = w + 1;
      if (stripeIdx < activeStripes) {
        ctrl[slotBase + SLOT_ROW_START] = this.stripeStarts[stripeIdx];
        ctrl[slotBase + SLOT_ROW_COUNT] = this.stripeCounts[stripeIdx];
      } else {
        ctrl[slotBase + SLOT_ROW_START] = 0;
        ctrl[slotBase + SLOT_ROW_COUNT] = 0;
      }
    }

    // Release barrier (zero futex syscall when helpers are spinning!)
    this.publishJob(nextSeq);

    // Coordinator computes stripe 0 immediately on its own WASM SIMD instance
    runLocalStripe(this.stripeStarts[0], this.stripeCounts[0]);

    this.joinHelpers(nextSeq);
    this.dispatches += 1;
    this.bytesProcessed += rowBytes * rows;
  }

  /**
   * Fused multi-projection single-barrier dispatch (`Q+K+V` or `Gate+Up`):
   * Executes up to 3 sibling projections sharing one input vector across all `P` threads in ONE
   * wakeup/join barrier — cutting barriers/layer from 7 to 4 and parallelizing small `K`/`V` heads!
   */
  dispatchParallelGroup(
    op: ParallelMatOp,
    entries: readonly MatGroupEntry[],
    xOrQ8Ptr: number,
    xF32Ptr: number,
    cols: number,
    runLocalStripe: (entryIndex: number, rowStart: number, rowCount: number) => void,
  ): void {
    if (!this.active || this.helperCount === 0 || entries.length === 0) {
      for (let i = 0; i < entries.length; i++) {
        runLocalStripe(i, 0, entries[i].rows);
      }
      return;
    }

    // Handle arbitrary entry counts in chunks of MAX_GROUP_ENTRIES (3)
    for (let baseIdx = 0; baseIdx < entries.length; baseIdx += MAX_GROUP_ENTRIES) {
      const groupCount = Math.min(MAX_GROUP_ENTRIES, entries.length - baseIdx);
      const ctrl = this.ctrl;
      const nextSeq = (this.jobSeq + 1) | 0;
      this.jobSeq = nextSeq;

      ctrl[CTRL_OP_CODE] = OP_GROUP_DISPATCH;
      ctrl[CTRL_GROUP_OP] = op;
      ctrl[CTRL_GROUP_COUNT] = groupCount;
      ctrl[CTRL_X_OR_Q8_PTR] = xOrQ8Ptr | 0;
      ctrl[CTRL_X_F32_PTR] = xF32Ptr | 0;
      ctrl[CTRL_COLS] = cols;

      const localStarts = this.localGroupStarts;
      const localCounts = this.localGroupCounts;
      let chunkBytes = 0;

      for (let g = 0; g < groupCount; g++) {
        const e = entries[baseIdx + g];
        const rowBytes = weightRowByteSize(e.weightType, cols);
        chunkBytes += rowBytes * e.rows;
        const dBase = GROUP_DESC_BASE + g * GROUP_DESC_STRIDE;
        ctrl[dBase + GDESC_WEIGHT_TYPE] = e.weightType;
        ctrl[dBase + GDESC_W_PTR] = e.wPtr | 0;
        ctrl[dBase + GDESC_OUT_PTR] = e.outPtr | 0;
        ctrl[dBase + GDESC_ROW_BYTES] = rowBytes;
        ctrl[dBase + GDESC_O_STRIDE_BYTES] = e.oStrideBytes;
        ctrl[dBase + GDESC_ROWS] = e.rows;

        const activeStripes = this.fillRowStripes(e.rows, 4);
        localStarts[g] = this.stripeStarts[0];
        localCounts[g] = this.stripeCounts[0];

        const startOff = g === 0 ? SLOT_ROW_START : g === 1 ? SLOT_G1_ROW_START : SLOT_G2_ROW_START;
        const countOff = g === 0 ? SLOT_ROW_COUNT : g === 1 ? SLOT_G1_ROW_COUNT : SLOT_G2_ROW_COUNT;
        for (let w = 0; w < this.helperCount; w++) {
          const slotBase = WORKER_SLOT_BASE + w * WORKER_SLOT_STRIDE;
          const stripeIdx = w + 1;
          if (stripeIdx < activeStripes) {
            ctrl[slotBase + startOff] = this.stripeStarts[stripeIdx];
            ctrl[slotBase + countOff] = this.stripeCounts[stripeIdx];
          } else {
            ctrl[slotBase + startOff] = 0;
            ctrl[slotBase + countOff] = 0;
          }
        }
      }

      this.publishJob(nextSeq);

      for (let g = 0; g < groupCount; g++) {
        if (localCounts[g] > 0) {
          runLocalStripe(baseIdx + g, localStarts[g], localCounts[g]);
        }
      }

      this.joinHelpers(nextSeq);
      this.dispatches += 1;
      this.bytesProcessed += chunkBytes;
    }
  }

  /**
   * Synchronous zero-copy multi-core 3-tier FlashAttention dispatch across Q heads `[0, nHeads)`.
   * Each thread owns an isolated 128-byte scratch buffer (`WASM_ATTN_WORKER_SCRATCH_BASE + id * 128`)
   * so long-context attention scales linearly across cores with bit-exact parity vs `P = 1`.
   */
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
  ): void {
    if (!this.active || this.helperCount === 0 || nHeads < 2) {
      runLocalHeads(0, nHeads, WASM_ATTN_WORKER_SCRATCH_BASE);
      return;
    }

    const activeStripes = this.fillRowStripes(nHeads, 1);
    if (activeStripes <= 1) {
      runLocalHeads(0, nHeads, WASM_ATTN_WORKER_SCRATCH_BASE);
      return;
    }

    const ctrl = this.ctrl;
    const nextSeq = (this.jobSeq + 1) | 0;
    this.jobSeq = nextSeq;
    const d = ATTN_DESC_BASE;

    ctrl[CTRL_OP_CODE] = OP_FLASH_ATTN_RANGE;
    ctrl[d + 0] = qPtr | 0;
    ctrl[d + 1] = outPtr | 0;
    ctrl[d + 2] = pageStatePtr | 0;
    ctrl[d + 3] = pageSlotPtr | 0;
    ctrl[d + 4] = hotKLayerPtr | 0;
    ctrl[d + 5] = hotVLayerPtr | 0;
    ctrl[d + 6] = q8KLayerPtr | 0;
    ctrl[d + 7] = q8VLayerPtr | 0;
    ctrl[d + 8] = q8KScLayerPtr | 0;
    ctrl[d + 9] = q8VScLayerPtr | 0;
    ctrl[d + 10] = q4KLayerPtr | 0;
    ctrl[d + 11] = q4VLayerPtr | 0;
    ctrl[d + 12] = q4KScLayerPtr | 0;
    ctrl[d + 13] = q4VScLayerPtr | 0;
    ctrl[d + 14] = nHeads;
    ctrl[d + 15] = nHeadsKv;
    ctrl[d + 16] = headDim;
    ctrl[d + 17] = seqLen;
    ctrl[d + 18] = hotPageBytes;
    ctrl[d + 19] = q8PageBytes;
    ctrl[d + 20] = q4PageBytes;
    ctrl[d + 21] = scPageBytes;
    ctrl[d + 22] = hotKvStrideBytes;
    ctrl[d + 23] = q8KvStrideBytes;
    ctrl[d + 24] = q4KvStrideBytes;
    ctrl[d + 25] = scStrideBytes;

    for (let w = 0; w < this.helperCount; w++) {
      const slotBase = WORKER_SLOT_BASE + w * WORKER_SLOT_STRIDE;
      const stripeIdx = w + 1;
      if (stripeIdx < activeStripes) {
        ctrl[slotBase + SLOT_ROW_START] = this.stripeStarts[stripeIdx];
        ctrl[slotBase + SLOT_ROW_COUNT] = this.stripeCounts[stripeIdx];
      } else {
        ctrl[slotBase + SLOT_ROW_START] = 0;
        ctrl[slotBase + SLOT_ROW_COUNT] = 0;
      }
    }

    this.publishJob(nextSeq);

    // Coordinator computes stripe 0 on its dedicated scratch slot (workerId = 0)
    runLocalHeads(
      this.stripeStarts[0],
      this.stripeStarts[0] + this.stripeCounts[0],
      WASM_ATTN_WORKER_SCRATCH_BASE,
    );

    this.joinHelpers(nextSeq);
    this.dispatches += 1;
  }

  dispose(): void {
    if (this.workers.length > 0) {
      const nextSeq = (this.jobSeq + 1) | 0;
      this.jobSeq = nextSeq;
      Atomics.store(this.ctrl, CTRL_OP_CODE, OP_SHUTDOWN);
      Atomics.store(this.ctrl, CTRL_JOB_SEQ, nextSeq);
      Atomics.notify(this.ctrl, CTRL_JOB_SEQ, this.workers.length);
      for (const w of this.workers) {
        try {
          void w.terminate();
        } catch {
          // ignore
        }
      }
      this.workers.length = 0;
    }
    this.active = false;
  }
}
