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
import { partitionRowStripes } from './performanceContract';
import {
  executeWasmMatStripe,
  ParallelMatOp,
  VerdantWasmArena,
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
export const OP_SHUTDOWN = 99;
export const MAX_POOL_WORKERS = 8;

/** Active spin window (~150–250 µs) keeps helper threads hot across back-to-back transformer layers. */
export const WORKER_ACTIVE_SPIN = 24_000;
export const COORDINATOR_JOIN_SPIN = 40_000;

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
      Atomics.wait(ctrl, CTRL_JOB_SEQ, lastSeq, 25);
      seq = Atomics.load(ctrl, CTRL_JOB_SEQ);
    }
    lastSeq = seq;

    const op = Atomics.load(ctrl, CTRL_OP_CODE);
    if (op === OP_SHUTDOWN) {
      Atomics.store(ctrl, slotBase + SLOT_DONE_SEQ, seq);
      Atomics.notify(ctrl, slotBase + SLOT_DONE_SEQ, 1);
      break;
    }

    if (op === OP_PING) {
      Atomics.store(ctrl, slotBase + SLOT_DONE_SEQ, seq);
      Atomics.notify(ctrl, slotBase + SLOT_DONE_SEQ, 1);
      continue;
    }

    if (op === OP_GROUP_DISPATCH) {
      const groupOp = Atomics.load(ctrl, CTRL_GROUP_OP) as ParallelMatOp;
      const groupCount = Atomics.load(ctrl, CTRL_GROUP_COUNT);
      const xOrQ8Ptr = Atomics.load(ctrl, CTRL_X_OR_Q8_PTR);
      const xF32Ptr = Atomics.load(ctrl, CTRL_X_F32_PTR);
      const cols = Atomics.load(ctrl, CTRL_COLS);

      for (let g = 0; g < groupCount; g++) {
        const startOff = g === 0 ? SLOT_ROW_START : g === 1 ? SLOT_G1_ROW_START : SLOT_G2_ROW_START;
        const countOff = g === 0 ? SLOT_ROW_COUNT : g === 1 ? SLOT_G1_ROW_COUNT : SLOT_G2_ROW_COUNT;
        const rowCount = Atomics.load(ctrl, slotBase + countOff);
        if (rowCount > 0) {
          const rowStart = Atomics.load(ctrl, slotBase + startOff);
          const dBase = GROUP_DESC_BASE + g * GROUP_DESC_STRIDE;
          const weightType = Atomics.load(ctrl, dBase + GDESC_WEIGHT_TYPE) as GgmlType;
          const wPtr = Atomics.load(ctrl, dBase + GDESC_W_PTR);
          const outPtr = Atomics.load(ctrl, dBase + GDESC_OUT_PTR);
          const rowBytes = Atomics.load(ctrl, dBase + GDESC_ROW_BYTES);
          const oStrideBytes = Atomics.load(ctrl, dBase + GDESC_O_STRIDE_BYTES);

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
      Atomics.notify(ctrl, slotBase + SLOT_DONE_SEQ, 1);
      continue;
    }

    const rowCount = Atomics.load(ctrl, slotBase + SLOT_ROW_COUNT);
    if (rowCount > 0) {
      const rowStart = Atomics.load(ctrl, slotBase + SLOT_ROW_START);
      const weightType = Atomics.load(ctrl, CTRL_WEIGHT_TYPE) as GgmlType;
      const wPtr = Atomics.load(ctrl, CTRL_W_PTR);
      const xOrQ8Ptr = Atomics.load(ctrl, CTRL_X_OR_Q8_PTR);
      const xF32Ptr = Atomics.load(ctrl, CTRL_X_F32_PTR);
      const outPtr = Atomics.load(ctrl, CTRL_OUT_PTR);
      const cols = Atomics.load(ctrl, CTRL_COLS);
      const rowBytes = Atomics.load(ctrl, CTRL_ROW_BYTES);
      const oStrideBytes = Atomics.load(ctrl, CTRL_O_STRIDE_BYTES);

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
    Atomics.notify(ctrl, slotBase + SLOT_DONE_SEQ, 1);
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

    const ctrlSab = new SharedArrayBuffer((WORKER_SLOT_BASE + MAX_POOL_WORKERS * WORKER_SLOT_STRIDE) * 4);
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
            if (op === ${ParallelMatOp.MUL_MAT_BATCH4_Q8}) {
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
            const xp = op === ${ParallelMatOp.MUL_MAT_Q8} ? xF32 : xOrQ8;
            if (wt === ${GgmlType.F32}) { k.mul_mat_f32(wPtr, xp, outPtr, cols, rows); return; }
            if (wt === ${GgmlType.F16}) { k.mul_mat_f16(wPtr, xp, outPtr, cols, rows); return; }
            if (wt === ${GgmlType.BF16}) { k.mul_mat_bf16(wPtr, xp, outPtr, cols, rows); return; }
            if (wt === ${GgmlType.Q8_0}) { k.mul_mat_q8_0(wPtr, xp, outPtr, cols, rows); return; }
            if (wt === ${GgmlType.Q6_K}) { k.mul_mat_q6_k(wPtr, xp, outPtr, cols, rows); return; }
          }

          let lastSeq = 0;
          for (;;) {
            let seq = Atomics.load(ctrl, ${CTRL_JOB_SEQ});
            for (let spin = 0; spin < ${WORKER_ACTIVE_SPIN} && seq === lastSeq; spin++) {
              seq = Atomics.load(ctrl, ${CTRL_JOB_SEQ});
            }
            while (seq === lastSeq) {
              Atomics.wait(ctrl, ${CTRL_JOB_SEQ}, lastSeq, 25);
              seq = Atomics.load(ctrl, ${CTRL_JOB_SEQ});
            }
            lastSeq = seq;
            const op = Atomics.load(ctrl, ${CTRL_OP_CODE});
            if (op === ${OP_SHUTDOWN}) {
              Atomics.store(ctrl, slotBase + ${SLOT_DONE_SEQ}, seq);
              Atomics.notify(ctrl, slotBase + ${SLOT_DONE_SEQ}, 1);
              break;
            }
            if (op === ${OP_PING}) {
              Atomics.store(ctrl, slotBase + ${SLOT_DONE_SEQ}, seq);
              Atomics.notify(ctrl, slotBase + ${SLOT_DONE_SEQ}, 1);
              continue;
            }
            if (op === ${OP_GROUP_DISPATCH}) {
              const groupOp = Atomics.load(ctrl, ${CTRL_GROUP_OP});
              const groupCount = Atomics.load(ctrl, ${CTRL_GROUP_COUNT});
              const xOrQ8 = Atomics.load(ctrl, ${CTRL_X_OR_Q8_PTR});
              const xF32 = Atomics.load(ctrl, ${CTRL_X_F32_PTR});
              const cols = Atomics.load(ctrl, ${CTRL_COLS});
              for (let g = 0; g < groupCount; g++) {
                const startOff = g === 0 ? ${SLOT_ROW_START} : g === 1 ? ${SLOT_G1_ROW_START} : ${SLOT_G2_ROW_START};
                const countOff = g === 0 ? ${SLOT_ROW_COUNT} : g === 1 ? ${SLOT_G1_ROW_COUNT} : ${SLOT_G2_ROW_COUNT};
                const rowCount = Atomics.load(ctrl, slotBase + countOff);
                if (rowCount > 0) {
                  const rowStart = Atomics.load(ctrl, slotBase + startOff);
                  const dBase = ${GROUP_DESC_BASE} + g * ${GROUP_DESC_STRIDE};
                  const wt = Atomics.load(ctrl, dBase + ${GDESC_WEIGHT_TYPE});
                  const wPtr = Atomics.load(ctrl, dBase + ${GDESC_W_PTR});
                  const outPtr = Atomics.load(ctrl, dBase + ${GDESC_OUT_PTR});
                  const rowBytes = Atomics.load(ctrl, dBase + ${GDESC_ROW_BYTES});
                  const oStride = Atomics.load(ctrl, dBase + ${GDESC_O_STRIDE_BYTES});
                  runStripe(groupOp, wt, wPtr + rowStart * rowBytes, xOrQ8, xF32, outPtr + rowStart * 4, cols, rowCount, oStride);
                }
              }
              Atomics.store(ctrl, slotBase + ${SLOT_DONE_SEQ}, seq);
              Atomics.notify(ctrl, slotBase + ${SLOT_DONE_SEQ}, 1);
              continue;
            }
            const rowCount = Atomics.load(ctrl, slotBase + ${SLOT_ROW_COUNT});
            if (rowCount > 0) {
              const rowStart = Atomics.load(ctrl, slotBase + ${SLOT_ROW_START});
              const wt = Atomics.load(ctrl, ${CTRL_WEIGHT_TYPE});
              const wPtr = Atomics.load(ctrl, ${CTRL_W_PTR});
              const xOrQ8 = Atomics.load(ctrl, ${CTRL_X_OR_Q8_PTR});
              const xF32 = Atomics.load(ctrl, ${CTRL_X_F32_PTR});
              const outPtr = Atomics.load(ctrl, ${CTRL_OUT_PTR});
              const cols = Atomics.load(ctrl, ${CTRL_COLS});
              const rowBytes = Atomics.load(ctrl, ${CTRL_ROW_BYTES});
              const oStride = Atomics.load(ctrl, ${CTRL_O_STRIDE_BYTES});
              runStripe(op, wt, wPtr + rowStart * rowBytes, xOrQ8, xF32, outPtr + rowStart * 4, cols, rowCount, oStride);
            }
            Atomics.store(ctrl, slotBase + ${SLOT_DONE_SEQ}, seq);
            Atomics.notify(ctrl, slotBase + ${SLOT_DONE_SEQ}, 1);
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
   * Empirically measures hot spin-barrier round-trip latency in microseconds so Theorem T3
   * derives the true host parallel threshold `W_par*(P)` rather than a pessimistic constant.
   */
  private measureBarrierLatencyUs(): number {
    if (!this.active || this.helperCount === 0) return 0;
    const ctrl = this.ctrl;
    const pingOnce = () => {
      const nextSeq = (this.jobSeq + 1) | 0;
      this.jobSeq = nextSeq;
      Atomics.store(ctrl, CTRL_OP_CODE, OP_PING);
      Atomics.store(ctrl, CTRL_JOB_SEQ, nextSeq);
      Atomics.notify(ctrl, CTRL_JOB_SEQ, this.helperCount);
      this.joinHelpers(nextSeq);
    };
    // Warm up helper threads into active spin window
    for (let i = 0; i < 6; i++) pingOnce();
    const samples = 16;
    const t0 = performance.now();
    for (let i = 0; i < samples; i++) pingOnce();
    const elapsedUs = ((performance.now() - t0) * 1000) / samples;
    return Math.max(0.8, Math.min(18, elapsedUs));
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
        try {
          Atomics.wait(ctrl, doneIdx, done, 5);
        } catch {
          // Non-blocking fallback if ever invoked in an environment where Atomics.wait is forbidden
        }
        done = Atomics.load(ctrl, doneIdx);
      }
    }
  }

  /**
   * Synchronous zero-copy parallel matrix dispatch using Theorem T3 deterministic 4-row-aligned stripes.
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

    const stripes = partitionRowStripes(rows, this.workerCount, 4);
    if (stripes.length <= 1) {
      runLocalStripe(0, rows);
      return;
    }

    const rowBytes = weightRowByteSize(weightType, cols);
    const ctrl = this.ctrl;
    const nextSeq = (this.jobSeq + 1) | 0;
    this.jobSeq = nextSeq;

    Atomics.store(ctrl, CTRL_OP_CODE, op);
    Atomics.store(ctrl, CTRL_WEIGHT_TYPE, weightType);
    Atomics.store(ctrl, CTRL_W_PTR, wPtr);
    Atomics.store(ctrl, CTRL_X_OR_Q8_PTR, xOrQ8Ptr);
    Atomics.store(ctrl, CTRL_X_F32_PTR, xF32Ptr);
    Atomics.store(ctrl, CTRL_OUT_PTR, outPtr);
    Atomics.store(ctrl, CTRL_COLS, cols);
    Atomics.store(ctrl, CTRL_TOTAL_ROWS, rows);
    Atomics.store(ctrl, CTRL_ROW_BYTES, rowBytes);
    Atomics.store(ctrl, CTRL_O_STRIDE_BYTES, oStrideBytes);

    // Assign stripes 1..P-1 to helper threads 0..P-2
    for (let w = 0; w < this.helperCount; w++) {
      const slotBase = WORKER_SLOT_BASE + w * WORKER_SLOT_STRIDE;
      const stripe = stripes[w + 1];
      if (stripe) {
        Atomics.store(ctrl, slotBase + SLOT_ROW_START, stripe.rowStart);
        Atomics.store(ctrl, slotBase + SLOT_ROW_COUNT, stripe.rowCount);
      } else {
        Atomics.store(ctrl, slotBase + SLOT_ROW_START, 0);
        Atomics.store(ctrl, slotBase + SLOT_ROW_COUNT, 0);
      }
    }

    // Release barrier + wake all helper workers
    Atomics.store(ctrl, CTRL_JOB_SEQ, nextSeq);
    Atomics.notify(ctrl, CTRL_JOB_SEQ, this.helperCount);

    // Coordinator computes stripe 0 immediately on its own WASM SIMD instance
    const localStripe = stripes[0];
    runLocalStripe(localStripe.rowStart, localStripe.rowCount);

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

      Atomics.store(ctrl, CTRL_OP_CODE, OP_GROUP_DISPATCH);
      Atomics.store(ctrl, CTRL_GROUP_OP, op);
      Atomics.store(ctrl, CTRL_GROUP_COUNT, groupCount);
      Atomics.store(ctrl, CTRL_X_OR_Q8_PTR, xOrQ8Ptr);
      Atomics.store(ctrl, CTRL_X_F32_PTR, xF32Ptr);
      Atomics.store(ctrl, CTRL_COLS, cols);

      const localStarts: number[] = [0, 0, 0];
      const localCounts: number[] = [0, 0, 0];
      let chunkBytes = 0;

      for (let g = 0; g < groupCount; g++) {
        const e = entries[baseIdx + g];
        const rowBytes = weightRowByteSize(e.weightType, cols);
        chunkBytes += rowBytes * e.rows;
        const dBase = GROUP_DESC_BASE + g * GROUP_DESC_STRIDE;
        Atomics.store(ctrl, dBase + GDESC_WEIGHT_TYPE, e.weightType);
        Atomics.store(ctrl, dBase + GDESC_W_PTR, e.wPtr);
        Atomics.store(ctrl, dBase + GDESC_OUT_PTR, e.outPtr);
        Atomics.store(ctrl, dBase + GDESC_ROW_BYTES, rowBytes);
        Atomics.store(ctrl, dBase + GDESC_O_STRIDE_BYTES, e.oStrideBytes);
        Atomics.store(ctrl, dBase + GDESC_ROWS, e.rows);

        const stripes = partitionRowStripes(e.rows, this.workerCount, 4);
        localStarts[g] = stripes[0].rowStart;
        localCounts[g] = stripes[0].rowCount;

        const startOff = g === 0 ? SLOT_ROW_START : g === 1 ? SLOT_G1_ROW_START : SLOT_G2_ROW_START;
        const countOff = g === 0 ? SLOT_ROW_COUNT : g === 1 ? SLOT_G1_ROW_COUNT : SLOT_G2_ROW_COUNT;
        for (let w = 0; w < this.helperCount; w++) {
          const slotBase = WORKER_SLOT_BASE + w * WORKER_SLOT_STRIDE;
          const stripe = stripes[w + 1];
          if (stripe) {
            Atomics.store(ctrl, slotBase + startOff, stripe.rowStart);
            Atomics.store(ctrl, slotBase + countOff, stripe.rowCount);
          } else {
            Atomics.store(ctrl, slotBase + startOff, 0);
            Atomics.store(ctrl, slotBase + countOff, 0);
          }
        }
      }

      Atomics.store(ctrl, CTRL_JOB_SEQ, nextSeq);
      Atomics.notify(ctrl, CTRL_JOB_SEQ, this.helperCount);

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
