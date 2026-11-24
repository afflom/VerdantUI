import test from 'node:test';
import assert from 'node:assert/strict';
import { GgmlType } from './gguf';
import {
  PRISM_NARROW_LANE_PROOFS,
  PRISM_NATIVE_BACKEND_CONFIG,
  PrismCensusLedger,
  VRAM_PAGE_BYTES,
  VRAM_READ_WINDOW_PAGES,
  canonicalObjectiveScore,
  canonicalTieBreak,
  compilePrismMatrixSchedule,
  planVramReadWindows,
  prismAlignedOffset,
  prismContentDigest,
  prismDecideMetadataOp,
  prismDecideNativeDevice,
  prismPlanBackingBytes,
  proveNarrowLaneExactness,
  runPrismVerificationSuite,
  selectCertifiedPrismRoute,
  uorLexicographicArgmax,
  uorSlabCodes,
  uorTableWords,
  validateMatrixTriple,
  type PrismHostCapabilities,
} from './prismKernel';
import {
  Q8_K_ACT_BLOCK_BYTES,
  VerdantWasmArena,
  type MatGroupEntry,
} from './verdantWasmSimd';
import { verdantMulMat } from './verdantKernels';
import { classifyWebGpuAdapter } from './verdantWebGpu';
import { runVTensorConformanceSuite } from './vtensorProvider';

function makeRng(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 0xffffffff;
  };
}

function fillQ4K(dst: Uint8Array, rows: number, cols: number, seed = 101): void {
  const rng = makeRng(seed);
  const blocksPerRow = cols / 256;
  const rowBytes = blocksPerRow * 144;
  for (let r = 0; r < rows; r++) {
    for (let b = 0; b < blocksPerRow; b++) {
      const off = r * rowBytes + b * 144;
      dst[off] = 0x00;
      dst[off + 1] = 0x2c; // F16 0.0625
      dst[off + 2] = 0x00;
      dst[off + 3] = 0x28; // F16 0.03125
      for (let i = 4; i < 144; i++) {
        dst[off + i] = Math.floor(rng() * 256) & 0xff;
      }
    }
  }
}

function fillQ6K(dst: Uint8Array, rows: number, cols: number, seed = 202): void {
  const rng = makeRng(seed);
  const blocksPerRow = cols / 256;
  const rowBytes = blocksPerRow * 210;
  for (let r = 0; r < rows; r++) {
    for (let b = 0; b < blocksPerRow; b++) {
      const off = r * rowBytes + b * 210;
      for (let i = 0; i < 208; i++) {
        dst[off + i] = Math.floor(rng() * 256) & 0xff;
      }
      dst[off + 208] = 0x00;
      dst[off + 209] = 0x28; // F16 0.03125
    }
  }
}

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na * nb);
  return denom > 0 ? dot / denom : 1;
}

test('Prism NativeBackend theorems, conformancy refusals & narrow-lane proofs hold exactly', () => {
  assert.equal(PRISM_NATIVE_BACKEND_CONFIG.alignment, 64);
  assert.equal(prismPlanBackingBytes(0), 0);
  assert.equal(prismPlanBackingBytes(128), 128 + 63);
  assert.equal(prismAlignedOffset(0), 0);
  assert.equal(prismAlignedOffset(64), 0);
  assert.equal(prismAlignedOffset(65), 63);
  assert.equal(prismAlignedOffset(127), 1);

  assert.equal(prismDecideNativeDevice(0), true);
  assert.equal(prismDecideNativeDevice(-1), false);
  assert.equal(prismDecideNativeDevice(1), false);
  assert.equal(prismDecideMetadataOp(0, 128), true);
  assert.equal(prismDecideMetadataOp(35, 128), true);
  assert.equal(prismDecideMetadataOp(24, 0), true);
  assert.equal(prismDecideMetadataOp(24, 128), false);

  // Planted-defect refusal (CT-05)
  const badCols = validateMatrixTriple(1, 512, 256, 1024, 1, 1024, 1024, 1);
  assert.equal(badCols.ok, false);
  assert.equal(badCols.reason, 'Nonconformant');

  const badRowStride = validateMatrixTriple(4, 256, 256, 64, 4, 64, 0, 1);
  assert.equal(badRowStride.ok, false);
  assert.equal(badRowStride.reason, 'OutputAliasesItself');

  const badOverlap = validateMatrixTriple(4, 256, 256, 64, 4, 64, 32, 1);
  assert.equal(badOverlap.ok, false);
  assert.equal(badOverlap.reason, 'OutputAliasesItself');

  assert.equal(validateMatrixTriple(4, 256, 256, 64, 4, 64, 64, 1).ok, true);

  // Narrow-lane exactness proofs & overflow refusal
  assert.equal(PRISM_NARROW_LANE_PROOFS.q4k_x_q8k.fitsI32Exactly, true);
  assert.equal(PRISM_NARROW_LANE_PROOFS.q6k_x_q8k.fitsI32Exactly, true);
  assert.equal(PRISM_NARROW_LANE_PROOFS.q40_x_q80.fitsI32Exactly, true);
  assert.equal(PRISM_NARROW_LANE_PROOFS.q80_x_q80.fitsI32Exactly, true);
  assert.equal(PRISM_NARROW_LANE_PROOFS.i8_lane_max_k.fitsI32Exactly, true);
  assert.equal(proveNarrowLaneExactness('overflow', 128, 128, 256, 1024).fitsI32Exactly, false);

  // Tabulated slab & AUTH-TF1-D6 lexicographic argmax
  assert.equal(uorSlabCodes(0), 0);
  assert.equal(uorSlabCodes(1), 1);
  assert.equal(uorSlabCodes(3), 4);
  assert.equal(uorSlabCodes(16), 16);
  assert.equal(uorSlabCodes(255), 256);
  assert.equal(uorTableWords(12, 16, 4), 16 * 16 * 4);

  const arg = uorLexicographicArgmax([0.5, 7.25, 7.25, 7.25, -2]);
  assert.equal(arg.index, 1);
  assert.equal(arg.value, 7.25);
});

test('Prism CanonicalObjective, CostSurface routing, schedule compilation, vRAM windows & ledger', () => {
  assert.equal(
    canonicalObjectiveScore({
      dramReadCost: 0,
      computeOpCost: 0,
      quantEncodeCost: 0,
      syncBarrierCost: 0,
      hostTransferCost: 0,
    }),
    0,
  );
  assert.ok(canonicalTieBreak('wasm-q8-4row', 'wasm-q8k-4row') < 0);
  assert.equal(canonicalTieBreak('wasm-q8k-4row', 'wasm-q8k-4row'), 0);
  assert.equal(prismContentDigest('verdant-prism'), prismContentDigest('verdant-prism'));

  const baseHost: PrismHostCapabilities = {
    wasmSimd: true,
    relaxedVnni: true,
    hardwareWebGpu: false,
    effectiveWorkers: 4,
    memoryBandwidthGBs: 16,
    computeGops: 32,
    barrierLatencyUs: 3,
    precisionMode: 'q8-vnni',
  };

  // K-quant selects Q8_K narrow-lane 4-row route
  const q4kCert = selectCertifiedPrismRoute(baseHost, {
    weightType: GgmlType.Q4_K,
    rows: 2048,
    cols: 2048,
    batchSize: 1,
    groupSize: 3,
    weightBytes: 2048 * 8 * 144,
    gpuResident: false,
  });
  assert.equal(q4kCert.selectedRoute, 'wasm-q8k-4row');

  // Legacy Q4_0 selects Q8_0 4-row route
  const q40Cert = selectCertifiedPrismRoute(baseHost, {
    weightType: GgmlType.Q4_0,
    rows: 2048,
    cols: 2048,
    batchSize: 1,
    groupSize: 2,
    weightBytes: 2048 * 64 * 18,
    gpuResident: false,
  });
  assert.equal(q40Cert.selectedRoute, 'wasm-q8-4row');

  // Hardware GPU resident matrix selects webgpu-wgsl-group
  const gpuCert = selectCertifiedPrismRoute(
    { ...baseHost, hardwareWebGpu: true },
    {
      weightType: GgmlType.Q4_K,
      rows: 4096,
      cols: 2048,
      batchSize: 1,
      groupSize: 3,
      weightBytes: 4096 * 8 * 144,
      gpuResident: true,
    },
  );
  assert.equal(gpuCert.selectedRoute, 'webgpu-wgsl-group');

  // Strict F64 precision mode selects wasm-f32-strict
  const strictCert = selectCertifiedPrismRoute(
    { ...baseHost, precisionMode: 'strict-f64' },
    {
      weightType: GgmlType.Q4_K,
      rows: 2048,
      cols: 2048,
      batchSize: 1,
      groupSize: 1,
      weightBytes: 2048 * 8 * 144,
      gpuResident: false,
    },
  );
  assert.equal(strictCert.selectedRoute, 'wasm-f32-strict');

  // Schedule compilation with one planted invalid shape
  const sched = compilePrismMatrixSchedule(baseHost, [
    {
      label: 'blk.0.attn_q',
      weightType: GgmlType.Q4_K,
      rows: 2048,
      cols: 2048,
      batchSize: 1,
      groupSize: 3,
      weightBytes: 2048 * 8 * 144,
      gpuResident: false,
    },
    {
      label: 'blk.0.bad_op',
      weightType: GgmlType.Q4_K,
      rows: -1,
      cols: 2048,
      batchSize: 1,
      groupSize: 1,
      weightBytes: 0,
      gpuResident: false,
    },
  ]);
  assert.equal(sched.q8kOps, 1);
  assert.equal(sched.refusedOps, 1);
  assert.equal(sched.instructions[0].kind, 'multiply');
  assert.equal(sched.instructions[1].kind, 'refused');

  // vRAM ReadPolicy::window bounded 64-page coalescing
  const windows = planVramReadWindows([
    { byteOffset: 0, byteLength: VRAM_PAGE_BYTES * 2 },
    { byteOffset: VRAM_PAGE_BYTES * 2, byteLength: VRAM_PAGE_BYTES * 3 },
    { byteOffset: VRAM_PAGE_BYTES * 100, byteLength: VRAM_PAGE_BYTES * 70 },
  ]);
  assert.equal(windows.length, 3);
  assert.equal(windows[0].startPage, 0);
  assert.equal(windows[0].pageCount, 5);
  assert.equal(windows[1].startPage, 100);
  assert.equal(windows[1].pageCount, VRAM_READ_WINDOW_PAGES);
  assert.equal(windows[2].startPage, 100 + VRAM_READ_WINDOW_PAGES);
  assert.equal(windows[2].pageCount, 6);

  // Census ledger & verification suite
  const ledger = new PrismCensusLedger();
  ledger.recordQ8K(512, 2048, 4);
  ledger.recordQ80();
  ledger.recordTabulated();
  ledger.recordWebGpuGroup();
  ledger.recordStrictF32();
  ledger.recordFusedBarrier();
  const snap = ledger.snapshot();
  assert.equal(snap.q8kDispatches, 1);
  assert.equal(snap.q8kSuperBlocks, 512 * 8 * 4);
  assert.equal(snap.q80Dispatches, 1);
  assert.equal(snap.uorTabulatedDispatches, 1);
  assert.equal(snap.webGpuGroupDispatches, 1);
  assert.equal(snap.strictF32Dispatches, 1);
  assert.equal(snap.fusedGroupBarriers, 1);
  assert.equal(snap.narrowLaneVerified, true);

  const suite = runPrismVerificationSuite();
  assert.equal(suite.length, 5);
  for (const check of suite) {
    assert.equal(check.passed, true, `${check.id} failed: ${check.detail}`);
  }
});

test('WASM SIMD uor_table_build_i8_v4 + uor_table_gather_u8_v4 are bit-exact across code spaces and depths', () => {
  const arena = new VerdantWasmArena({ initialPages: 16, shared: false });
  const rng = makeRng(202611);

  for (const space of [4, 8, 16]) {
    for (const depth of [1, 4]) {
      const block = 8;
      const slabWords = space * 16;
      const bookPtr = arena.allocRegion(space * block);
      const actsPtr = arena.allocRegion(block * 16);
      const stackPtr = arena.allocRegion(depth * slabWords * 4);
      const codesPtr = arena.allocRegion(depth);
      const lanePtr = arena.allocRegion(16 * 4);

      const expectedStack = new Int32Array(depth * slabWords);
      const codes = new Uint8Array(depth);

      for (let d = 0; d < depth; d++) {
        const book = new Int8Array(arena.memory.buffer, bookPtr, space * block);
        const acts = new Int8Array(arena.memory.buffer, actsPtr, block * 16);
        for (let i = 0; i < book.length; i++) book[i] = Math.floor(rng() * 127) - 63;
        for (let i = 0; i < acts.length; i++) acts[i] = Math.floor(rng() * 127) - 63;

        const outPtr = stackPtr + d * slabWords * 4;
        arena.kernels.uor_table_build_i8_v4(space, block, bookPtr, actsPtr, outPtr);

        const gotSlab = new Int32Array(arena.memory.buffer, outPtr, slabWords);
        for (let c = 0; c < space; c++) {
          for (let r = 0; r < 16; r++) {
            let acc = 0;
            for (let p = 0; p < block; p++) {
              acc += book[c * block + p] * acts[p * 16 + r];
            }
            assert.equal(
              gotSlab[c * 16 + r],
              acc,
              `uor_table_build_i8_v4 mismatch (space=${space}, d=${d}, c=${c}, r=${r})`,
            );
            expectedStack[d * slabWords + c * 16 + r] = acc;
          }
        }
        codes[d] = Math.floor(rng() * space) % space;
      }

      new Uint8Array(arena.memory.buffer, codesPtr, depth).set(codes);
      const lane = new Int32Array(arena.memory.buffer, lanePtr, 16);
      lane.fill(0);
      arena.kernels.uor_table_gather_u8_v4(depth, slabWords, stackPtr, codesPtr, lanePtr);

      for (let r = 0; r < 16; r++) {
        let ref = 0;
        for (let d = 0; d < depth; d++) {
          ref += expectedStack[d * slabWords + codes[d] * 16 + r];
        }
        assert.equal(
          lane[r],
          ref,
          `uor_table_gather_u8_v4 mismatch (space=${space}, depth=${depth}, r=${r})`,
        );
      }
    }
  }
});

test('Verdant/Prism Q4_K×Q8_K and Q6_K×Q8_K match reference across remainder rows, batch-4, and fused groups', () => {
  const arena = new VerdantWasmArena({ initialPages: 64, shared: false });
  const cols = 512;
  const q8kStride = (cols >>> 8) * Q8_K_ACT_BLOCK_BYTES;
  const rng = makeRng(90909);

  const xBatch = new Float32Array(4 * cols);
  for (let i = 0; i < xBatch.length; i++) xBatch[i] = (rng() - 0.5) * 2.0;

  const xBatchOff = arena.allocRegion(4 * cols * 4);
  const q8kBatchOff = arena.allocRegion(4 * q8kStride);
  new Float32Array(arena.memory.buffer, xBatchOff, 4 * cols).set(xBatch);
  for (let t = 0; t < 4; t++) {
    arena.quantizeActivationQ8K(xBatchOff + t * cols * 4, q8kBatchOff + t * q8kStride, cols);
  }

  for (const rows of [1, 2, 3, 5, 13, 19]) {
    for (const fmt of [
      { name: 'Q4_K', type: GgmlType.Q4_K, rowBytes: (cols / 256) * 144, fill: fillQ4K },
      { name: 'Q6_K', type: GgmlType.Q6_K, rowBytes: (cols / 256) * 210, fill: fillQ6K },
    ]) {
      const wBytesLen = rows * fmt.rowBytes;
      const wOff = arena.allocRegion(wBytesLen);
      const wSlice = arena.bytes().subarray(wOff, wOff + wBytesLen);
      fmt.fill(wSlice, rows, cols, rows * 37 + fmt.type);

      const ySingleOff = arena.allocRegion(rows * 4);
      const yBatchOff = arena.allocRegion(4 * rows * 4);

      arena.mulMatQ8K(fmt.type, wOff, q8kBatchOff, xBatchOff, ySingleOff, cols, rows);
      arena.mulMatBatchQ8K(fmt.type, wOff, q8kBatchOff, xBatchOff, yBatchOff, cols, rows, 4);

      const gotSingle = new Float32Array(arena.memory.buffer, ySingleOff, rows);
      const gotBatch = new Float32Array(arena.memory.buffer, yBatchOff, 4 * rows);

      // Token 0 single vs batch-4 must agree closely
      for (let r = 0; r < rows; r++) {
        assert.ok(Number.isFinite(gotSingle[r]));
        assert.ok(
          Math.abs(gotSingle[r] - gotBatch[r]) < 1e-4,
          `${fmt.name} rows=${rows} token 0 single vs batch-4 mismatch at row ${r}`,
        );
      }

      // Check all 4 tokens against reference verdantMulMat
      for (let t = 0; t < 4; t++) {
        const ref = new Float32Array(rows);
        verdantMulMat(wSlice, 0, fmt.type, cols, rows, xBatch.subarray(t * cols, (t + 1) * cols), ref);
        const gotT = gotBatch.subarray(t * rows, (t + 1) * rows);
        let maxErr = 0;
        let refNorm = 0;
        for (let r = 0; r < rows; r++) {
          maxErr = Math.max(maxErr, Math.abs(gotT[r] - ref[r]));
          refNorm = Math.max(refNorm, Math.abs(ref[r]));
        }
        const relErr = maxErr / Math.max(0.05, refNorm);
        assert.ok(
          relErr < 0.035,
          `${fmt.name}×Q8_K rows=${rows} tok=${t} relErr=${relErr.toFixed(4)} (maxErr=${maxErr.toFixed(5)})`,
        );
        if (rows >= 13) {
          const cos = cosineSimilarity(gotT, ref);
          assert.ok(cos > 0.999, `${fmt.name}×Q8_K rows=${rows} tok=${t} cos=${cos}`);
        }
      }
    }
  }

  // Mixed Q4_K + Q6_K fused group single-token & batch-4
  const r0 = 24;
  const r1 = 16;
  const w0Off = arena.allocRegion(r0 * (cols / 256) * 144);
  const w1Off = arena.allocRegion(r1 * (cols / 256) * 210);
  fillQ4K(arena.bytes().subarray(w0Off, w0Off + r0 * (cols / 256) * 144), r0, cols, 701);
  fillQ6K(arena.bytes().subarray(w1Off, w1Off + r1 * (cols / 256) * 210), r1, cols, 702);

  const g0Single = arena.allocRegion(r0 * 4);
  const g1Single = arena.allocRegion(r1 * 4);
  const g0Ref = arena.allocRegion(r0 * 4);
  const g1Ref = arena.allocRegion(r1 * 4);
  const g0Batch = arena.allocRegion(4 * r0 * 4);
  const g1Batch = arena.allocRegion(4 * r1 * 4);

  const groupSingle: MatGroupEntry[] = [
    { weightType: GgmlType.Q4_K, wPtr: w0Off, outPtr: g0Single, rows: r0, oStrideBytes: r0 * 4 },
    { weightType: GgmlType.Q6_K, wPtr: w1Off, outPtr: g1Single, rows: r1, oStrideBytes: r1 * 4 },
  ];
  const groupBatch: MatGroupEntry[] = [
    { weightType: GgmlType.Q4_K, wPtr: w0Off, outPtr: g0Batch, rows: r0, oStrideBytes: r0 * 4 },
    { weightType: GgmlType.Q6_K, wPtr: w1Off, outPtr: g1Batch, rows: r1, oStrideBytes: r1 * 4 },
  ];

  arena.mulMatQ8KGroup(groupSingle, q8kBatchOff, xBatchOff, cols);
  arena.mulMatBatch4Q8KGroup(groupBatch, q8kBatchOff, xBatchOff, cols);
  arena.mulMatQ8K(GgmlType.Q4_K, w0Off, q8kBatchOff, xBatchOff, g0Ref, cols, r0);
  arena.mulMatQ8K(GgmlType.Q6_K, w1Off, q8kBatchOff, xBatchOff, g1Ref, cols, r1);

  assert.deepEqual(
    Array.from(new Float32Array(arena.memory.buffer, g0Single, r0)),
    Array.from(new Float32Array(arena.memory.buffer, g0Ref, r0)),
  );
  assert.deepEqual(
    Array.from(new Float32Array(arena.memory.buffer, g1Single, r1)),
    Array.from(new Float32Array(arena.memory.buffer, g1Ref, r1)),
  );
  const g0B0 = new Float32Array(arena.memory.buffer, g0Batch, r0);
  const g0S = new Float32Array(arena.memory.buffer, g0Single, r0);
  for (let i = 0; i < r0; i++) {
    assert.ok(Math.abs(g0B0[i] - g0S[i]) < 1e-4);
  }
});

test('WebGPU hardware-only admission rejects software renderers and admits hardware silicon', () => {
  assert.equal(
    classifyWebGpuAdapter({ vendor: 'Google', description: 'SwiftShader Device (Subzero)' }).admitted,
    false,
  );
  assert.equal(
    classifyWebGpuAdapter({ vendor: 'Mesa', device: 'llvmpipe (LLVM 17.0)' }).admitted,
    false,
  );
  assert.equal(
    classifyWebGpuAdapter({ vendor: 'Apple', device: 'Apple M3 Max', isFallbackAdapter: true }).admitted,
    false,
  );
  const hw = classifyWebGpuAdapter({ vendor: 'Apple', architecture: 'metal-3', device: 'Apple M3 Max' });
  assert.equal(hw.admitted, true);
  assert.equal(hw.status, 'hardware-active');
});

test('Full vTensor conformance suite passes 100% including VT-67 & PRISM-01..05 with zero CPU fallbacks', () => {
  const report = runVTensorConformanceSuite();
  assert.equal(
    report.passedCount,
    report.totalCount,
    `Failed checks: ${report.checks
      .filter((c) => !c.passed)
      .map((c) => `${c.id}: ${c.detail}`)
      .join('; ')}`,
  );
  const ids = new Set(report.checks.map((c) => c.id));
  for (const requiredId of ['VT-67', 'PRISM-01', 'PRISM-02', 'PRISM-03', 'PRISM-04', 'PRISM-05']) {
    assert.ok(ids.has(requiredId), `Missing required conformance gate ${requiredId}`);
  }
});
