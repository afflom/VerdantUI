/**
 * Persistent vTensor Compute Sub-Worker (Browser Module Web Worker).
 *
 * Instantiates the shared-memory Verdant WASM SIMD128 / Relaxed-SIMD binary directly on the
 * coordinator's `WebAssembly.Memory({ shared: true })` and runs a lock-free `Atomics` stripe
 * execution loop with zero copies and zero `postMessage` overhead per operator.
 */

import type { VerdantSimdExports } from './verdantWasmSimd';
import { runComputeWorkerLoop } from './vtensorThreadPool';

interface InitComputeWorkerMessage {
  wasmModule: WebAssembly.Module;
  memory: WebAssembly.Memory;
  ctrlSab: SharedArrayBuffer;
  workerIndex: number;
}

self.onmessage = (event: MessageEvent<InitComputeWorkerMessage>) => {
  const { wasmModule, memory, ctrlSab, workerIndex } = event.data;
  const instance = new WebAssembly.Instance(wasmModule, {
    env: { memory },
  });
  const kernels = instance.exports as unknown as VerdantSimdExports;
  const ctrl = new Int32Array(ctrlSab);
  runComputeWorkerLoop(kernels, ctrl, workerIndex);
};
