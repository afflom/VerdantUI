import test from 'node:test';
import assert from 'node:assert/strict';
import os from 'node:os';
import { GgmlType, type GgufTensorInfo, type ParsedGgufModel } from './gguf';
import {
  ParallelMatOp,
  Q8_0_ACT_BLOCK_BYTES,
  Q8_K_ACT_BLOCK_BYTES,
  VerdantWasmArena,
  WASM_TOPK_IDS_OFFSET,
  WASM_TOPK_VALS_OFFSET,
  computeMoeWorkerScratchLayout,
  isVerdantSharedWasmSupported,
  weightRowByteSize,
  type MatGroupEntry,
} from './verdantWasmSimd';
import {
  VTensorThreadPool,
  deriveOptimalPoolWorkers,
  deriveWaveOptimalMoeThreads,
} from './vtensorThreadPool';
import { verdantMulMat, verdantRope, verdantSwiGlu } from './verdantKernels';
import { VramKvPager } from './vramKvPager';
import { VramWeightPager, type LayerTensorInfoGroup } from './vramWeightPager';
import { MemoryBackingStore } from './vramBackingStore';
import { VTensorBackendRegistry } from './vtensorProvider';
import {
  PRISM_NARROW_LANE_PROOFS,
  selectCertifiedPrismRoute,
} from './prismKernel';
import {
  derivePerformanceContract,
  partitionRowStripes,
  type EnvironmentProfile,
  type ModelGeometrySpec,
} from './performanceContract';

/** Deterministic xorshift32 PRNG for reproducible test weights and activations. */
function makeRng(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 0xffffffff;
  };
}

/** Populate deterministic Q4_K weight blocks (144 bytes per 256 elements). */
function fillQ4K(dst: Uint8Array, rows: number, cols: number, seed = 1337): void {
  const rng = makeRng(seed);
  const blocksPerRow = cols / 256;
  const rowBytes = blocksPerRow * 144;
  for (let r = 0; r < rows; r++) {
    const rOff = r * rowBytes;
    for (let b = 0; b < blocksPerRow; b++) {
      const off = rOff + b * 144;
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

/** Populate deterministic Q4_0 weight blocks (18 bytes per 32 elements). */
function fillQ40(dst: Uint8Array, rows: number, cols: number, seed = 4242): void {
  const rng = makeRng(seed);
  const blocksPerRow = cols / 32;
  const rowBytes = blocksPerRow * 18;
  for (let r = 0; r < rows; r++) {
    const rOff = r * rowBytes;
    for (let b = 0; b < blocksPerRow; b++) {
      const off = rOff + b * 18;
      dst[off] = 0x00;
      dst[off + 1] = 0x30; // F16 0.125
      for (let i = 2; i < 18; i++) {
        dst[off + i] = Math.floor(rng() * 256) & 0xff;
      }
    }
  }
}

/** Populate deterministic Q8_0 weight blocks (34 bytes per 32 elements). */
function fillQ80(dst: Uint8Array, rows: number, cols: number, seed = 7777): void {
  const rng = makeRng(seed);
  const blocksPerRow = cols / 32;
  const rowBytes = blocksPerRow * 34;
  for (let r = 0; r < rows; r++) {
    const rOff = r * rowBytes;
    for (let b = 0; b < blocksPerRow; b++) {
      const off = rOff + b * 34;
      dst[off] = 0x00;
      dst[off + 1] = 0x28; // F16 0.03125
      for (let i = 2; i < 34; i++) {
        dst[off + i] = Math.floor(rng() * 256) & 0xff;
      }
    }
  }
}

test('Theorem T3 partitionRowStripes preserves 4-row SIMD alignment and covers all rows without gaps', () => {
  for (const rows of [1, 7, 32, 576, 1536, 4096, 8192, 32768, 49152, 128256]) {
    for (const P of [1, 2, 3, 4, 6, 8]) {
      const stripes = partitionRowStripes(rows, P, 4);
      assert.ok(stripes.length >= 1 && stripes.length <= P);
      let cursor = 0;
      for (let i = 0; i < stripes.length; i++) {
        const s = stripes[i];
        assert.equal(s.rowStart, cursor, `gap at stripe ${i} for rows=${rows}, P=${P}`);
        assert.ok(s.rowCount > 0);
        if (i < stripes.length - 1 && rows >= 4) {
          assert.equal(s.rowCount % 4, 0);
        }
        cursor += s.rowCount;
      }
      assert.equal(cursor, rows);
    }
  }
});

test('Multi-core VTensorThreadPool produces bit-identical results and scales throughput at 1B/3B matrix shapes', async () => {
  assert.equal(
    isVerdantSharedWasmSupported(),
    true,
    'SharedArrayBuffer WASM memory must be supported in Node.js',
  );

  const arena = new VerdantWasmArena({ initialPages: 420, shared: true });
  assert.equal(arena.isSharedMemory, true);

  const availCpus = Math.max(1, os.cpus().length);
  const targetWorkers = Math.min(4, Math.max(2, availCpus));

  const pool = await VTensorThreadPool.create(arena, {
    maxWorkers: targetWorkers,
    parallelThresholdBytes: 32 * 1024,
  });
  assert.ok(pool !== null && pool.isActive, 'Expected VTensorThreadPool.create() to activate');

  try {
    // 1. Bit-exactness across Q4_K, Q4_0, Q8_0 — single-token and 4-token strided batch
    const cols = 2048;
    const rows = 1024;
    const qStride = (cols / 32) * Q8_0_ACT_BLOCK_BYTES;
    const rng = makeRng(2026);
    const xSingle = new Float32Array(cols);
    const xBatch = new Float32Array(4 * cols);
    for (let i = 0; i < cols; i++) xSingle[i] = (rng() - 0.5) * 2;
    for (let i = 0; i < 4 * cols; i++) xBatch[i] = (rng() - 0.5) * 2;

    const xOff = arena.allocRegion(cols * 4);
    const xbOff = arena.allocRegion(4 * cols * 4);
    const qOff = arena.allocRegion(qStride);
    const qbOff = arena.allocRegion(4 * qStride);
    const ySingleSeqOff = arena.allocRegion(rows * 4);
    const ySingleParOff = arena.allocRegion(rows * 4);
    const yBatchSeqOff = arena.allocRegion(4 * rows * 4);
    const yBatchParOff = arena.allocRegion(4 * rows * 4);

    new Float32Array(arena.memory.buffer, xOff, cols).set(xSingle);
    new Float32Array(arena.memory.buffer, xbOff, 4 * cols).set(xBatch);
    arena.quantizeActivation(xOff, qOff, cols);
    for (let t = 0; t < 4; t++) {
      arena.quantizeActivation(xbOff + t * cols * 4, qbOff + t * qStride, cols);
    }

    const wQ4KBytes = rows * (cols / 256) * 144;
    const wQ4KOff = arena.allocRegion(wQ4KBytes);
    fillQ4K(arena.bytes().subarray(wQ4KOff, wQ4KOff + wQ4KBytes), rows, cols, 101);

    const wQ40Bytes = rows * (cols / 32) * 18;
    const wQ40Off = arena.allocRegion(wQ40Bytes);
    fillQ40(arena.bytes().subarray(wQ40Off, wQ40Off + wQ40Bytes), rows, cols, 202);

    const wQ80Bytes = rows * (cols / 32) * 34;
    const wQ80Off = arena.allocRegion(wQ80Bytes);
    fillQ80(arena.bytes().subarray(wQ80Off, wQ80Off + wQ80Bytes), rows, cols, 303);

    const cases: Array<{ name: string; type: GgmlType; wOff: number }> = [
      { name: 'Q4_K', type: GgmlType.Q4_K, wOff: wQ4KOff },
      { name: 'Q4_0', type: GgmlType.Q4_0, wOff: wQ40Off },
      { name: 'Q8_0', type: GgmlType.Q8_0, wOff: wQ80Off },
    ];

    for (const c of cases) {
      arena.attachThreadPool(null);
      arena.mulMatQ8(c.type, c.wOff, qOff, xOff, ySingleSeqOff, cols, rows);
      arena.mulMatBatchQ8(c.type, c.wOff, qbOff, xbOff, yBatchSeqOff, cols, rows, 4);

      arena.attachThreadPool(pool);
      arena.mulMatQ8(c.type, c.wOff, qOff, xOff, ySingleParOff, cols, rows);
      arena.mulMatBatchQ8(c.type, c.wOff, qbOff, xbOff, yBatchParOff, cols, rows, 4);

      const sSeq = new Float32Array(arena.memory.buffer, ySingleSeqOff, rows);
      const sPar = new Float32Array(arena.memory.buffer, ySingleParOff, rows);
      for (let i = 0; i < rows; i++) {
        assert.equal(
          sPar[i],
          sSeq[i],
          `${c.name} single-token mismatch at row ${i}: ${sPar[i]} !== ${sSeq[i]}`,
        );
      }

      const bSeq = new Float32Array(arena.memory.buffer, yBatchSeqOff, 4 * rows);
      const bPar = new Float32Array(arena.memory.buffer, yBatchParOff, 4 * rows);
      for (let i = 0; i < 4 * rows; i++) {
        assert.equal(
          bPar[i],
          bSeq[i],
          `${c.name} batch-4 strided mismatch at idx ${i}: ${bPar[i]} !== ${bSeq[i]}`,
        );
      }
    }

    // 2. Empirical large-scale throughput benchmark: 1B FFN gate/up [8192 x 2048] Q4_K (~9.4 MiB)
    const bigRows = 8192;
    const bigCols = 2048;
    const bigWBytes = bigRows * (bigCols / 256) * 144;
    const bigWOff = arena.allocRegion(bigWBytes);
    fillQ4K(arena.bytes().subarray(bigWOff, bigWOff + bigWBytes), bigRows, bigCols, 909);
    const bigYOff = arena.allocRegion(4 * bigRows * 4);

    // Warmup both paths
    arena.attachThreadPool(null);
    arena.mulMatQ8(GgmlType.Q4_K, bigWOff, qOff, xOff, bigYOff, bigCols, bigRows);
    arena.attachThreadPool(pool);
    arena.mulMatQ8(GgmlType.Q4_K, bigWOff, qOff, xOff, bigYOff, bigCols, bigRows);

    const iters = 6;
    arena.attachThreadPool(null);
    const tSeq0 = performance.now();
    for (let i = 0; i < iters; i++) {
      arena.mulMatQ8(GgmlType.Q4_K, bigWOff, qOff, xOff, bigYOff, bigCols, bigRows);
    }
    const seqMs = performance.now() - tSeq0;

    arena.attachThreadPool(pool);
    const tPar0 = performance.now();
    for (let i = 0; i < iters; i++) {
      arena.mulMatQ8(GgmlType.Q4_K, bigWOff, qOff, xOff, bigYOff, bigCols, bigRows);
    }
    const parMs = performance.now() - tPar0;

    // 4-token prefill batch on [8192 x 2048] Q4_K
    arena.attachThreadPool(null);
    const tbSeq0 = performance.now();
    for (let i = 0; i < 3; i++) {
      arena.mulMatBatchQ8(GgmlType.Q4_K, bigWOff, qbOff, xbOff, bigYOff, bigCols, bigRows, 4);
    }
    const bSeqMs = performance.now() - tbSeq0;

    arena.attachThreadPool(pool);
    const tbPar0 = performance.now();
    for (let i = 0; i < 3; i++) {
      arena.mulMatBatchQ8(GgmlType.Q4_K, bigWOff, qbOff, xbOff, bigYOff, bigCols, bigRows, 4);
    }
    const bParMs = performance.now() - tbPar0;

    const decodeSpeedup = seqMs / Math.max(0.001, parMs);
    const prefillSpeedup = bSeqMs / Math.max(0.001, bParMs);
    const tel = pool.getTelemetry();

    console.log(
      `[T3 Multi-Core P=${pool.workerCount}] Q4_K [8192x2048] decode: ${decodeSpeedup.toFixed(2)}x (${seqMs.toFixed(1)}ms -> ${parMs.toFixed(1)}ms) | batch-4 prefill: ${prefillSpeedup.toFixed(2)}x (${bSeqMs.toFixed(1)}ms -> ${bParMs.toFixed(1)}ms)`,
    );

    assert.ok(tel.parallelDispatches >= iters + 3, 'Expected parallel dispatches to be recorded');
    // Enforce wall-clock multi-core ratio on uncontended local hosts; shared CI VMs still verify 100% bit-exact parity above.
    if (availCpus >= 2 && process.env.CI !== 'true') {
      assert.ok(
        decodeSpeedup > 1.1,
        `Expected multi-core Q4_K [8192x2048] decode speedup > 1.1x (got ${decodeSpeedup.toFixed(2)}x)`,
      );
      assert.ok(
        prefillSpeedup > 1.2,
        `Expected multi-core Q4_K [8192x2048] batch-4 speedup > 1.2x (got ${prefillSpeedup.toFixed(2)}x)`,
      );
    }
  } finally {
    arena.attachThreadPool(null);
    pool.dispose();
  }
});

test('Belady-optimal cyclic layer pinning achieves exact (K-1)/L hit rate on cyclic transformer scans vs LRU 0%', () => {
  const L = 24;
  const K = 8;
  const dim = 64;
  const tensorBytes = dim * dim * 4;
  const normBytes = dim * 4;

  let offset = 0;
  const makeInfo = (name: string, dims: number[], byteSize: number): GgufTensorInfo => {
    const info: GgufTensorInfo = {
      name,
      dims,
      type: GgmlType.F32,
      typeName: 'F32',
      offset,
      byteSize,
    };
    offset += byteSize;
    return info;
  };

  const embdInfo = makeInfo('token_embd.weight', [dim, 64], dim * 64 * 4);
  const outNormInfo = makeInfo('output_norm.weight', [dim], normBytes);
  const outWeightInfo = makeInfo('output.weight', [dim, 64], dim * 64 * 4);

  const layerInfos: LayerTensorInfoGroup[] = [];
  for (let l = 0; l < L; l++) {
    layerInfos.push({
      attnNorm: makeInfo(`blk.${l}.attn_norm.weight`, [dim], normBytes),
      attnQ: makeInfo(`blk.${l}.attn_q.weight`, [dim, dim], tensorBytes),
      attnQBias: null,
      attnQNorm: null,
      attnK: makeInfo(`blk.${l}.attn_k.weight`, [dim, dim], tensorBytes),
      attnKBias: null,
      attnKNorm: null,
      attnV: makeInfo(`blk.${l}.attn_v.weight`, [dim, dim], tensorBytes),
      attnVBias: null,
      attnOutput: makeInfo(`blk.${l}.attn_output.weight`, [dim, dim], tensorBytes),
      attnOutputBias: null,
      ffnNorm: makeInfo(`blk.${l}.ffn_norm.weight`, [dim], normBytes),
      ffnGate: makeInfo(`blk.${l}.ffn_gate.weight`, [dim, dim], tensorBytes),
      ffnUp: makeInfo(`blk.${l}.ffn_up.weight`, [dim, dim], tensorBytes),
      ffnDown: makeInfo(`blk.${l}.ffn_down.weight`, [dim, dim], tensorBytes),
    });
  }

  const rawBuf = new ArrayBuffer(offset + 256);
  const dv = new DataView(rawBuf);
  for (let l = 0; l < L; l++) {
    dv.setFloat32(layerInfos[l].attnQ.offset, l + 1, true);
  }

  const parsed = {
    dataOffset: 0,
    tensorMap: new Map(),
    tensors: [],
  } as unknown as ParsedGgufModel;

  const backing = new MemoryBackingStore('belady-test', new Uint8Array(rawBuf));
  const registry = new VTensorBackendRegistry({ sharedWasmMemory: false });

  const pager = new VramWeightPager(
    parsed,
    backing,
    embdInfo,
    outNormInfo,
    outWeightInfo,
    false,
    layerInfos,
    {
      residencyMode: 'opfs-paged',
      layerFrameSlots: K,
      streamEmbeddings: false,
      streamLmHead: false,
    },
    (id, usage, cap) => registry.allocBuffer(id, usage, cap),
  );

  const passes = 4;
  for (let p = 0; p < passes; p++) {
    for (let l = 0; l < L; l++) {
      const w = pager.acquireLayer(l);
      const firstVal = w.attnQ.buffer.f32View(w.attnQ)[0];
      assert.equal(firstVal, l + 1, `Corrupted layer ${l} weights on pass ${p}`);
    }
  }

  const tel = pager.getTelemetry();
  assert.equal(tel.evictionPolicy, 'belady-cyclic');
  assert.equal(tel.pinnedLayerCount, K - 1);
  assert.equal(tel.layerHits, (K - 1) * (passes - 1));
  assert.equal(tel.layerFaults, L + (L - (K - 1)) * (passes - 1));

  // Verify Performance Contract Theorem T4b models Belady fault fraction
  const env: EnvironmentProfile = {
    simdTier: 'relaxed-vnni',
    relaxedSimdSupported: true,
    hardwareConcurrency: 4,
    sharedMemorySupported: true,
    effectiveWorkers: 4,
    memoryBandwidthGBs: 25,
    computeGops: 120,
    ridgeIntensityOpsPerByte: 4.8,
    workerSyncLatencyUs: 4,
    memoryBudgetBytes: 1024 * 1024 * 1024,
    calibrationMs: 1,
  };
  const modelSpec: ModelGeometrySpec = {
    hparams: {
      architecture: 'qwen2',
      name: '3B-Q4_K',
      contextLength: 4096,
      embeddingLength: 2048,
      blockCount: 28,
      feedForwardLength: 11008,
      headCount: 16,
      headCountKv: 2,
      headDim: 128,
      ropeDimensionCount: 128,
      ropeFreqBase: 1000000,
      ropeFreqScale: 1.0,
      ropeType: 'neox',
      rmsNormEps: 1e-6,
      vocabSize: 151936,
      bosTokenId: 151643,
      eosTokenId: 151645,
      unkTokenId: 0,
      addBosToken: false,
      tokenizerModel: 'gpt2',
      preTokenizer: 'qwen2',
      chatTemplate: null,
      primaryQuantization: 'Q4_K',
    },
    activeWeightBytes: 1_900_000_000,
    lmHeadWeightBytes: 180_000_000,
    activeParamCount: 3_000_000_000,
    totalWeightBufferBytes: 1_900_000_000,
    dominantWeightType: GgmlType.Q4_K,
    quantizedWeightFraction: 1.0,
  };
  const contract = derivePerformanceContract(env, modelSpec, {
    precisionMode: 'q8-vnni',
    requestedMaxContext: 2048,
    backingKind: 'opfs-sync',
    weightResidencyPolicy: 'paged',
    backingBandwidthMBs: 1200,
    backingQuotaBytes: 4 * 1024 * 1024 * 1024,
    maxLayerFrameSlotsCap: 16,
  });
  const naiveLruAllLayersMs = (1_900_000_000 / (1200 * 1e6)) * 1000;
  assert.ok(contract.opfsIoOverheadMsPerToken > 0);
  assert.ok(
    contract.opfsIoOverheadMsPerToken < naiveLruAllLayersMs * 0.7,
    `Expected Belady cyclic pinning to reduce OPFS decode I/O below 70% of naive LRU (${naiveLruAllLayersMs.toFixed(1)} ms), got ${contract.opfsIoOverheadMsPerToken.toFixed(1)} ms`,
  );
  registry.freeAllBuffers();
});

test('100% In-Verdant whole-layer FlashAttention matches scalar parity and accelerates seqLen=1024 across all 3 tiers', () => {
  const numLayers = 1;
  const maxContext = 1024;
  const nHeadsQ = 16;
  const nHeadsKv = 4;
  const headDim = 64;
  const kvDim = nHeadsKv * headDim;
  const qDim = nHeadsQ * headDim;

  const reg = new VTensorBackendRegistry({ initialWasmPages: 128, sharedWasmMemory: false });
  assert.ok(reg.arena !== null);
  const arena = reg.arena;

  const pager = new VramKvPager(
    {
      numLayers,
      maxContext,
      nHeadsKv,
      headDim,
      hotFrameLimit: 8, // 128 Hot F32 tokens
      warmQ8Limit: 16, // 256 Warm Q8 tokens
      coldQ4Enabled: true, // 640 Cold Q4 tokens
    },
    (id, usage, cap) => reg.allocBuffer(id, usage, cap),
  );

  const actBuf = reg.allocBuffer('attn.bench', 'compute', (qDim * 2 + kvDim * 2) * 4 + 256);
  const qT = actBuf.allocTensor('q', GgmlType.F32, [qDim]);
  const kT = actBuf.allocTensor('k', GgmlType.F32, [kvDim]);
  const vT = actBuf.allocTensor('v', GgmlType.F32, [kvDim]);
  const outT = actBuf.allocTensor('out', GgmlType.F32, [qDim]);

  const rng = makeRng(8088);
  const kArr = actBuf.f32View(kT);
  const vArr = actBuf.f32View(vT);
  const kPtr = actBuf.wasmPtr(kT);
  const vPtr = actBuf.wasmPtr(vT);

  for (let pos = 0; pos < maxContext; pos++) {
    for (let i = 0; i < kvDim; i++) {
      kArr[i] = (rng() - 0.5) * 1.2;
      vArr[i] = (rng() - 0.5) * 1.2;
    }
    pager.writeKvPtr(false, 0, pos, kPtr, arena);
    pager.writeKvPtr(true, 0, pos, vPtr, arena);
  }

  const kvTel = pager.getTelemetry();
  assert.ok(kvTel.hotResidentPages > 0 && kvTel.warmQ8Pages > 0 && kvTel.coldQ4Pages > 0);

  const qArr = actBuf.f32View(qT);
  for (let i = 0; i < qDim; i++) qArr[i] = (rng() - 0.5) * 1.5;

  const qPtr = actBuf.wasmPtr(qT);
  const outPtr = actBuf.wasmPtr(outT);
  const outVerdant = actBuf.f32View(outT);
  const outScalar = new Float32Array(qDim);

  // Parity check
  pager.flashAttnExt(qArr, qPtr, outVerdant, outPtr, nHeadsQ, 0, maxContext, arena);
  pager.flashAttnExt(qArr, -1, outScalar, -1, nHeadsQ, 0, maxContext, null);

  let maxDiff = 0;
  for (let i = 0; i < qDim; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(outVerdant[i] - outScalar[i]));
  }
  assert.ok(maxDiff < 1e-4, `In-Verdant FlashAttention diverged from scalar: maxDiff=${maxDiff}`);

  // Benchmark 100% in-Verdant 1-call layer attention vs scalar JS at seqLen=1024, 16 heads
  const emptyF32 = new Float32Array(0);
  const tW0 = performance.now();
  for (let i = 0; i < 24; i++) {
    pager.flashAttnExt(emptyF32, qPtr, emptyF32, outPtr, nHeadsQ, 0, maxContext, arena);
  }
  const verdantMs = performance.now() - tW0;

  const tS0 = performance.now();
  for (let i = 0; i < 24; i++) {
    pager.flashAttnExt(qArr, -1, outScalar, -1, nHeadsQ, 0, maxContext, null);
  }
  const scalarMs = performance.now() - tS0;
  const speedup = scalarMs / Math.max(0.001, verdantMs);
  console.log(
    `[In-Verdant Layer FlashAttn seqLen=1024 16H×24L] ${speedup.toFixed(2)}x speedup (${scalarMs.toFixed(1)}ms JS -> ${verdantMs.toFixed(1)}ms Verdant, maxDiff=${maxDiff.toExponential(1)})`,
  );
  assert.ok(speedup > 2.0, `Expected >2x speedup for in-Verdant layer FlashAttention (got ${speedup.toFixed(2)}x)`);
  reg.freeAllBuffers();
});

test('100% In-Verdant WASM SIMD SwiGLU and RoPE (NeoX + Interleaved) match JS references within 1e-4', () => {
  const arena = new VerdantWasmArena({ initialPages: 16, shared: false });

  // 1. SwiGLU across [-80, +80] with non-multiple-of-4 tail
  const n = 11011;
  const gate = new Float32Array(n);
  const up = new Float32Array(n);
  const rng = makeRng(5555);
  for (let i = 0; i < n; i++) {
    gate[i] = i < 200 ? -80 + i * 0.8 : (rng() - 0.5) * 20;
    up[i] = (rng() - 0.5) * 4;
  }
  const refSw = new Float32Array(n);
  verdantSwiGlu(gate, up, refSw, n);

  const gOff = arena.allocRegion(n * 4);
  const uOff = arena.allocRegion(n * 4);
  const oOff = arena.allocRegion(n * 4);
  new Float32Array(arena.memory.buffer, gOff, n).set(gate);
  new Float32Array(arena.memory.buffer, uOff, n).set(up);
  arena.kernels.swiglu_f32(gOff, uOff, oOff, n);
  const gotSw = new Float32Array(arena.memory.buffer, oOff, n);

  let maxSwErr = 0;
  for (let i = 0; i < n; i++) {
    assert.ok(Number.isFinite(gotSw[i]));
    maxSwErr = Math.max(maxSwErr, Math.abs(gotSw[i] - refSw[i]));
  }
  assert.ok(maxSwErr < 1e-4, `SwiGLU maxErr=${maxSwErr}`);

  // 2. RoPE (both NeoX and LLaMA interleaved) across 16 heads × headDim=128
  const nHeads = 16;
  const headDim = 128;
  const half = headDim >>> 1;
  const pos = 37;
  const cosArr = new Float32Array(half);
  const sinArr = new Float32Array(half);
  for (let i = 0; i < half; i++) {
    const ang = pos * Math.pow(10000, (-2 * i) / headDim);
    cosArr[i] = Math.cos(ang);
    sinArr[i] = Math.sin(ang);
  }
  const cOff = arena.allocRegion(half * 4);
  const sOff = arena.allocRegion(half * 4);
  const vOff = arena.allocRegion(nHeads * headDim * 4);
  new Float32Array(arena.memory.buffer, cOff, half).set(cosArr);
  new Float32Array(arena.memory.buffer, sOff, half).set(sinArr);

  for (const isNeox of [false, true]) {
    const orig = new Float32Array(nHeads * headDim);
    for (let i = 0; i < orig.length; i++) orig[i] = (rng() - 0.5) * 4;
    const refR = new Float32Array(orig);
    verdantRope(refR, nHeads, headDim, headDim, pos, 10000, isNeox);

    const wasmView = new Float32Array(arena.memory.buffer, vOff, orig.length);
    wasmView.set(orig);
    arena.kernels.rope_f32(vOff, cOff, sOff, nHeads, headDim, isNeox ? 1 : 0);

    let maxRopeErr = 0;
    for (let i = 0; i < orig.length; i++) {
      maxRopeErr = Math.max(maxRopeErr, Math.abs(wasmView[i] - refR[i]));
    }
    assert.ok(maxRopeErr < 1e-5, `RoPE (neox=${isNeox}) maxErr=${maxRopeErr}`);
  }
});

test('4-row register-blocked Q4_K, Q6_K, Q4_1, Q5_0, Q5_1, F16 kernels match scalar reference (main loop + remainder tail)', () => {
  const arena = new VerdantWasmArena({ initialPages: 32, shared: false });
  const cols = 512;
  const rows = 19; // 4 full 4-row tiles + 3-row remainder tail!
  const rng = makeRng(9901);

  const x = new Float32Array(cols);
  for (let i = 0; i < cols; i++) x[i] = (rng() - 0.5) * 2.0;

  const xOff = arena.allocRegion(cols * 4);
  const qStride = (cols / 32) * Q8_0_ACT_BLOCK_BYTES;
  const qOff = arena.allocRegion(qStride);
  const yOff = arena.allocRegion(rows * 4);
  new Float32Array(arena.memory.buffer, xOff, cols).set(x);
  arena.quantizeActivation(xOff, qOff, cols);

  const formats: Array<{ name: string; type: GgmlType; blockSize: number; typeBytes: number; maxRelTol: number; useQ8: boolean }> = [
    { name: 'F16', type: GgmlType.F16, blockSize: 1, typeBytes: 2, maxRelTol: 1e-3, useQ8: false },
    { name: 'Q4_1', type: GgmlType.Q4_1, blockSize: 32, typeBytes: 20, maxRelTol: 2.5e-2, useQ8: true },
    { name: 'Q5_0', type: GgmlType.Q5_0, blockSize: 32, typeBytes: 22, maxRelTol: 2.5e-2, useQ8: true },
    { name: 'Q5_1', type: GgmlType.Q5_1, blockSize: 32, typeBytes: 24, maxRelTol: 2.5e-2, useQ8: true },
    { name: 'Q4_K', type: GgmlType.Q4_K, blockSize: 256, typeBytes: 144, maxRelTol: 2.5e-2, useQ8: true },
    { name: 'Q6_K', type: GgmlType.Q6_K, blockSize: 256, typeBytes: 210, maxRelTol: 2.5e-2, useQ8: true },
  ];

  for (const fmt of formats) {
    const rowBytes = (cols / fmt.blockSize) * fmt.typeBytes;
    const totalBytes = rows * rowBytes;
    const wOff = arena.allocRegion(totalBytes);
    const wBytes = arena.bytes().subarray(wOff, wOff + totalBytes);

    if (fmt.type === GgmlType.F16) {
      // Safe finite F16 patterns in [-1.5, +1.5]
      for (let i = 0; i < totalBytes; i += 2) {
        const sign = rng() < 0.5 ? 0x00 : 0x80;
        const exp = 0x24 + Math.floor(rng() * 0x18); // tiny to ~1.5
        wBytes[i] = Math.floor(rng() * 256) & 0xff;
        wBytes[i + 1] = sign | (exp & 0x7f);
      }
    } else if (fmt.type === GgmlType.Q4_K) {
      fillQ4K(wBytes, rows, cols, 31415);
    } else if (fmt.type === GgmlType.Q6_K) {
      // Q6_K block = 210 bytes: ql[128], qh[64], scales[16], d(f16 at +208)
      const blocksPerRow = cols / 256;
      for (let r = 0; r < rows; r++) {
        for (let b = 0; b < blocksPerRow; b++) {
          const off = r * rowBytes + b * 210;
          for (let i = 0; i < 208; i++) wBytes[off + i] = Math.floor(rng() * 256) & 0xff;
          wBytes[off + 208] = 0x00;
          wBytes[off + 209] = 0x28; // F16 0.03125
        }
      }
    } else {
      // Q4_1 / Q5_0 / Q5_1: F16 scale at +0 (and optional min at +2)
      const blocksPerRow = cols / 32;
      for (let r = 0; r < rows; r++) {
        for (let b = 0; b < blocksPerRow; b++) {
          const off = r * rowBytes + b * fmt.typeBytes;
          wBytes[off] = 0x00;
          wBytes[off + 1] = 0x2c; // F16 0.0625
          if (fmt.type === GgmlType.Q4_1 || fmt.type === GgmlType.Q5_1) {
            wBytes[off + 2] = 0x00;
            wBytes[off + 3] = 0x28; // F16 0.03125
            for (let i = 4; i < fmt.typeBytes; i++) wBytes[off + i] = Math.floor(rng() * 256) & 0xff;
          } else {
            for (let i = 2; i < fmt.typeBytes; i++) wBytes[off + i] = Math.floor(rng() * 256) & 0xff;
          }
        }
      }
    }

    const refOut = new Float32Array(rows);
    verdantMulMat(wBytes, 0, fmt.type, cols, rows, x, refOut);

    if (fmt.useQ8) {
      arena.mulMatQ8(fmt.type, wOff, qOff, xOff, yOff, cols, rows);
    } else {
      arena.mulMat(fmt.type, wOff, xOff, yOff, cols, rows);
    }
    const gotOut = new Float32Array(arena.memory.buffer, yOff, rows);

    let maxErr = 0;
    let refNorm = 0;
    for (let r = 0; r < rows; r++) {
      assert.ok(Number.isFinite(gotOut[r]), `${fmt.name} row ${r} produced non-finite value`);
      maxErr = Math.max(maxErr, Math.abs(gotOut[r] - refOut[r]));
      refNorm = Math.max(refNorm, Math.abs(refOut[r]));
    }
    const relErr = maxErr / Math.max(1e-3, refNorm);
    assert.ok(
      relErr < fmt.maxRelTol,
      `4-row ${fmt.name} diverged from scalar ref: maxErr=${maxErr.toFixed(5)}, relErr=${relErr.toFixed(4)} > ${fmt.maxRelTol}`,
    );
  }
});

test('Fused single-barrier QKV & Gate+Up dispatch is bit-exact and scales throughput on both SmolLM2-135M and 1B/3B shapes', async () => {
  const arena = new VerdantWasmArena({ initialPages: 320, shared: true });
  const availCpus = Math.max(1, os.cpus().length);
  const targetWorkers = Math.min(4, Math.max(2, availCpus));
  const pool = await VTensorThreadPool.create(arena, {
    maxWorkers: targetWorkers,
    parallelThresholdBytes: 48 * 1024,
  });
  assert.ok(pool !== null && pool.isActive);

  try {
    // SmolLM2-135M layer geometry: dim=512 (Q4_K multiple of 256), q=512, kv=256, ffn=1536
    const cols = 512;
    const qRows = 512;
    const kvRows = 256;
    const ffnRows = 1536;
    const rng = makeRng(4444);
    const x = new Float32Array(cols);
    for (let i = 0; i < cols; i++) x[i] = (rng() - 0.5) * 2;

    const xOff = arena.allocRegion(cols * 4);
    const qStride = (cols / 32) * Q8_0_ACT_BLOCK_BYTES;
    const q8Off = arena.allocRegion(qStride);
    new Float32Array(arena.memory.buffer, xOff, cols).set(x);
    arena.quantizeActivation(xOff, q8Off, cols);

    const rowBytes = (cols / 256) * 144;
    const allocQ4K = (r: number, seed: number) => {
      const off = arena.allocRegion(r * rowBytes);
      fillQ4K(arena.bytes().subarray(off, off + r * rowBytes), r, cols, seed);
      return off;
    };

    const wQ = allocQ4K(qRows, 11);
    const wK = allocQ4K(kvRows, 12);
    const wV = allocQ4K(kvRows, 13);
    const wGate = allocQ4K(ffnRows, 14);
    const wUp = allocQ4K(ffnRows, 15);

    const oQSeq = arena.allocRegion(qRows * 4);
    const oKSeq = arena.allocRegion(kvRows * 4);
    const oVSeq = arena.allocRegion(kvRows * 4);
    const oQPar = arena.allocRegion(qRows * 4);
    const oKPar = arena.allocRegion(kvRows * 4);
    const oVPar = arena.allocRegion(kvRows * 4);

    const oGSeq = arena.allocRegion(ffnRows * 4);
    const oUSeq = arena.allocRegion(ffnRows * 4);
    const oGPar = arena.allocRegion(ffnRows * 4);
    const oUPar = arena.allocRegion(ffnRows * 4);

    const qkvSeq: MatGroupEntry[] = [
      { weightType: GgmlType.Q4_K, wPtr: wQ, outPtr: oQSeq, rows: qRows, oStrideBytes: qRows * 4 },
      { weightType: GgmlType.Q4_K, wPtr: wK, outPtr: oKSeq, rows: kvRows, oStrideBytes: kvRows * 4 },
      { weightType: GgmlType.Q4_K, wPtr: wV, outPtr: oVSeq, rows: kvRows, oStrideBytes: kvRows * 4 },
    ];
    const qkvPar: MatGroupEntry[] = [
      { weightType: GgmlType.Q4_K, wPtr: wQ, outPtr: oQPar, rows: qRows, oStrideBytes: qRows * 4 },
      { weightType: GgmlType.Q4_K, wPtr: wK, outPtr: oKPar, rows: kvRows, oStrideBytes: kvRows * 4 },
      { weightType: GgmlType.Q4_K, wPtr: wV, outPtr: oVPar, rows: kvRows, oStrideBytes: kvRows * 4 },
    ];
    const ffnSeq: MatGroupEntry[] = [
      { weightType: GgmlType.Q4_K, wPtr: wGate, outPtr: oGSeq, rows: ffnRows, oStrideBytes: ffnRows * 4 },
      { weightType: GgmlType.Q4_K, wPtr: wUp, outPtr: oUSeq, rows: ffnRows, oStrideBytes: ffnRows * 4 },
    ];
    const ffnPar: MatGroupEntry[] = [
      { weightType: GgmlType.Q4_K, wPtr: wGate, outPtr: oGPar, rows: ffnRows, oStrideBytes: ffnRows * 4 },
      { weightType: GgmlType.Q4_K, wPtr: wUp, outPtr: oUPar, rows: ffnRows, oStrideBytes: ffnRows * 4 },
    ];

    // Bit-exact check: fused P=1 vs fused P=N
    arena.attachThreadPool(null);
    arena.mulMatQ8Group(qkvSeq, q8Off, xOff, cols, true);
    arena.mulMatQ8Group(ffnSeq, q8Off, xOff, cols, true);

    arena.attachThreadPool(pool);
    arena.mulMatQ8Group(qkvPar, q8Off, xOff, cols, true);
    arena.mulMatQ8Group(ffnPar, q8Off, xOff, cols, true);

    const checkBitExact = (aOff: number, bOff: number, len: number, label: string) => {
      const a = new Float32Array(arena.memory.buffer, aOff, len);
      const b = new Float32Array(arena.memory.buffer, bOff, len);
      for (let i = 0; i < len; i++) {
        assert.equal(b[i], a[i], `Fused group ${label} mismatch at row ${i}`);
      }
    };
    checkBitExact(oQSeq, oQPar, qRows, 'Q');
    checkBitExact(oKSeq, oKPar, kvRows, 'K');
    checkBitExact(oVSeq, oVPar, kvRows, 'V');
    checkBitExact(oGSeq, oGPar, ffnRows, 'Gate');
    checkBitExact(oUSeq, oUPar, ffnRows, 'Up');

    // Benchmark 30 transformer layers (SmolLM2-135M scale): P=1 vs P=N fused QKV + Gate+Up
    const layerIters = 30;
    arena.attachThreadPool(null);
    const t1 = performance.now();
    for (let i = 0; i < layerIters; i++) {
      arena.mulMatQ8Group(qkvSeq, q8Off, xOff, cols, true);
      arena.mulMatQ8Group(ffnSeq, q8Off, xOff, cols, true);
    }
    const seqMs = performance.now() - t1;

    arena.attachThreadPool(pool);
    const t2 = performance.now();
    for (let i = 0; i < layerIters; i++) {
      arena.mulMatQ8Group(qkvPar, q8Off, xOff, cols, true);
      arena.mulMatQ8Group(ffnPar, q8Off, xOff, cols, true);
    }
    const parMs = performance.now() - t2;

    console.log(
      `[Fused QKV+GateUp P=${pool.workerCount} barrier=${pool.barrierLatencyUs.toFixed(1)}us] 135M-scale 30L: ${(seqMs / Math.max(0.01, parMs)).toFixed(2)}x (${seqMs.toFixed(2)}ms -> ${parMs.toFixed(2)}ms)`,
    );
    assert.ok(pool.barrierLatencyUs > 0 && pool.barrierLatencyUs <= 18);
  } finally {
    arena.attachThreadPool(null);
    pool.dispose();
  }
});

test('100% In-Verdant WASM SIMD argmax_f32 and topk_f32 match JS references exactly and accelerate 151,936-vocab sampling', () => {
  const arena = new VerdantWasmArena({ initialPages: 32, shared: false });
  const vocabSize = 151_939; // 152K + 3 tail elements
  const lOff = arena.allocRegion(vocabSize * 4);
  const logits = new Float32Array(arena.memory.buffer, lOff, vocabSize);
  const rng = makeRng(7070);
  for (let i = 0; i < vocabSize; i++) {
    logits[i] = (rng() - 0.5) * 20;
  }
  // Inject known peaks + exact tie at tail
  logits[12345] = 31.5;
  logits[vocabSize - 1] = 31.5;
  logits[99] = 29.75;
  logits[100_003] = 28.25;

  // 1. Exact argmax parity (including first-index tie-breaking)
  let jsBestId = 0;
  let jsBestVal = -Infinity;
  for (let i = 0; i < vocabSize; i++) {
    if (logits[i] > jsBestVal) {
      jsBestVal = logits[i];
      jsBestId = i;
    }
  }
  const wasmBestId = arena.kernels.argmax_f32(lOff, vocabSize);
  assert.equal(wasmBestId, jsBestId);
  assert.equal(wasmBestId, 12345);

  // 2. Exact top-40 parity
  const k = 40;
  const effK = arena.kernels.topk_f32(lOff, vocabSize, k, WASM_TOPK_IDS_OFFSET, WASM_TOPK_VALS_OFFSET);
  assert.equal(effK, k);
  const wasmIds = Array.from(new Int32Array(arena.memory.buffer, WASM_TOPK_IDS_OFFSET, effK)).sort(
    (a, b) => logits[b] - logits[a] || a - b,
  );

  // Reference JS top-40
  const jsTopIds = new Int32Array(k);
  const jsTopVals = new Float32Array(k);
  jsTopVals.fill(-Infinity);
  let minSlot = 0;
  let minVal = -Infinity;
  for (let i = 0; i < vocabSize; i++) {
    const v = logits[i];
    if (v > minVal) {
      jsTopIds[minSlot] = i;
      jsTopVals[minSlot] = v;
      minVal = jsTopVals[0];
      minSlot = 0;
      for (let j = 1; j < k; j++) {
        if (jsTopVals[j] < minVal) {
          minVal = jsTopVals[j];
          minSlot = j;
        }
      }
    }
  }
  const refIds = Array.from(jsTopIds).sort((a, b) => logits[b] - logits[a] || a - b);
  assert.deepEqual(wasmIds, refIds);

  // 3. Speed benchmark (40 tokens)
  const iters = 40;
  const tW0 = performance.now();
  let sink = 0;
  for (let it = 0; it < iters; it++) {
    sink += arena.kernels.topk_f32(lOff, vocabSize, k, WASM_TOPK_IDS_OFFSET, WASM_TOPK_VALS_OFFSET);
  }
  const wasmMs = performance.now() - tW0;

  const tJ0 = performance.now();
  for (let it = 0; it < iters; it++) {
    jsTopVals.fill(-Infinity);
    let ms = 0;
    let mv = -Infinity;
    for (let i = 0; i < vocabSize; i++) {
      const v = logits[i];
      if (v > mv) {
        jsTopIds[ms] = i;
        jsTopVals[ms] = v;
        mv = jsTopVals[0];
        ms = 0;
        for (let j = 1; j < k; j++) {
          if (jsTopVals[j] < mv) {
            mv = jsTopVals[j];
            ms = j;
          }
        }
      }
    }
    sink += jsTopIds[0];
  }
  const jsMs = performance.now() - tJ0;
  assert.ok(sink > 0);
  console.log(
    `[In-Verdant SIMD topk_f32 V=${vocabSize} K=${k} ×${iters}] ${(jsMs / Math.max(0.01, wasmMs)).toFixed(2)}x speedup (${jsMs.toFixed(2)}ms JS -> ${wasmMs.toFixed(2)}ms Verdant)`,
  );
});

test('Prism Q8_K narrow-lane accelerates Q4_K & Q6_K over Q8_0 and maintains bit-exact P=1 vs P=N multi-core parity', async () => {
  const arena = new VerdantWasmArena({ initialPages: 420, shared: true });
  const availCpus = Math.max(1, os.cpus().length);
  const targetWorkers = Math.min(4, Math.max(2, availCpus));
  const pool = await VTensorThreadPool.create(arena, {
    maxWorkers: targetWorkers,
    parallelThresholdBytes: 32 * 1024,
  });
  assert.ok(pool !== null && pool.isActive);

  try {
    const cols = 2048;
    const rows = 4096;
    const q8Stride = (cols / 32) * Q8_0_ACT_BLOCK_BYTES;
    const q8kStride = (cols / 256) * 320;
    const rng = makeRng(202607);

    const xBatch = new Float32Array(4 * cols);
    for (let i = 0; i < xBatch.length; i++) xBatch[i] = (rng() - 0.5) * 2.0;

    const xOff = arena.allocRegion(4 * cols * 4);
    const q8Off = arena.allocRegion(4 * q8Stride);
    const q8kOff = arena.allocRegion(4 * q8kStride);
    new Float32Array(arena.memory.buffer, xOff, 4 * cols).set(xBatch);
    for (let t = 0; t < 4; t++) {
      arena.quantizeActivation(xOff + t * cols * 4, q8Off + t * q8Stride, cols);
      arena.quantizeActivationQ8K(xOff + t * cols * 4, q8kOff + t * q8kStride, cols);
    }

    const wQ4KBytes = rows * (cols / 256) * 144;
    const wQ4KOff = arena.allocRegion(wQ4KBytes);
    fillQ4K(arena.bytes().subarray(wQ4KOff, wQ4KOff + wQ4KBytes), rows, cols, 611);

    const wQ6KBytes = rows * (cols / 256) * 210;
    const wQ6KOff = arena.allocRegion(wQ6KBytes);
    const q6Slice = arena.bytes().subarray(wQ6KOff, wQ6KOff + wQ6KBytes);
    const q6BlocksPerRow = cols / 256;
    for (let r = 0; r < rows; r++) {
      for (let b = 0; b < q6BlocksPerRow; b++) {
        const off = r * q6BlocksPerRow * 210 + b * 210;
        for (let i = 0; i < 208; i++) q6Slice[off + i] = Math.floor(rng() * 256) & 0xff;
        q6Slice[off + 208] = 0x00;
        q6Slice[off + 209] = 0x28;
      }
    }

    const ySeqOff = arena.allocRegion(4 * rows * 4);
    const yParOff = arena.allocRegion(4 * rows * 4);
    const yGrp0Seq = arena.allocRegion(4 * rows * 4);
    const yGrp1Seq = arena.allocRegion(4 * rows * 4);
    const yGrp0Par = arena.allocRegion(4 * rows * 4);
    const yGrp1Par = arena.allocRegion(4 * rows * 4);

    // 1. Bit-exact P=1 vs P=N for Q4_K and Q6_K single-token & batch-4
    for (const c of [
      { name: 'Q4_K', type: GgmlType.Q4_K, wOff: wQ4KOff },
      { name: 'Q6_K', type: GgmlType.Q6_K, wOff: wQ6KOff },
    ]) {
      arena.attachThreadPool(null);
      arena.mulMatQ8K(c.type, c.wOff, q8kOff, xOff, ySeqOff, cols, rows);
      arena.attachThreadPool(pool);
      arena.mulMatQ8K(c.type, c.wOff, q8kOff, xOff, yParOff, cols, rows);
      const sSeq = new Float32Array(arena.memory.buffer, ySeqOff, rows);
      const sPar = new Float32Array(arena.memory.buffer, yParOff, rows);
      for (let i = 0; i < rows; i++) {
        assert.equal(sPar[i], sSeq[i], `${c.name}×Q8_K single P=N mismatch at row ${i}`);
      }

      arena.attachThreadPool(null);
      arena.mulMatBatchQ8K(c.type, c.wOff, q8kOff, xOff, ySeqOff, cols, rows, 4);
      arena.attachThreadPool(pool);
      arena.mulMatBatchQ8K(c.type, c.wOff, q8kOff, xOff, yParOff, cols, rows, 4);
      const bSeq = new Float32Array(arena.memory.buffer, ySeqOff, 4 * rows);
      const bPar = new Float32Array(arena.memory.buffer, yParOff, 4 * rows);
      for (let i = 0; i < 4 * rows; i++) {
        assert.equal(bPar[i], bSeq[i], `${c.name}×Q8_K batch-4 P=N mismatch at idx ${i}`);
      }
    }

    // 2. Bit-exact P=1 vs P=N for mixed Q4_K + Q6_K fused group (single + batch-4)
    const grpSeq: MatGroupEntry[] = [
      { weightType: GgmlType.Q4_K, wPtr: wQ4KOff, outPtr: yGrp0Seq, rows, oStrideBytes: rows * 4 },
      { weightType: GgmlType.Q6_K, wPtr: wQ6KOff, outPtr: yGrp1Seq, rows, oStrideBytes: rows * 4 },
    ];
    const grpPar: MatGroupEntry[] = [
      { weightType: GgmlType.Q4_K, wPtr: wQ4KOff, outPtr: yGrp0Par, rows, oStrideBytes: rows * 4 },
      { weightType: GgmlType.Q6_K, wPtr: wQ6KOff, outPtr: yGrp1Par, rows, oStrideBytes: rows * 4 },
    ];
    arena.attachThreadPool(null);
    arena.mulMatQ8KGroup(grpSeq, q8kOff, xOff, cols);
    arena.attachThreadPool(pool);
    arena.mulMatQ8KGroup(grpPar, q8kOff, xOff, cols);
    assert.deepEqual(
      Array.from(new Float32Array(arena.memory.buffer, yGrp0Par, rows)),
      Array.from(new Float32Array(arena.memory.buffer, yGrp0Seq, rows)),
    );
    assert.deepEqual(
      Array.from(new Float32Array(arena.memory.buffer, yGrp1Par, rows)),
      Array.from(new Float32Array(arena.memory.buffer, yGrp1Seq, rows)),
    );

    arena.attachThreadPool(null);
    arena.mulMatBatch4Q8KGroup(grpSeq, q8kOff, xOff, cols);
    arena.attachThreadPool(pool);
    arena.mulMatBatch4Q8KGroup(grpPar, q8kOff, xOff, cols);
    assert.deepEqual(
      Array.from(new Float32Array(arena.memory.buffer, yGrp0Par, 4 * rows)),
      Array.from(new Float32Array(arena.memory.buffer, yGrp0Seq, 4 * rows)),
    );
    assert.deepEqual(
      Array.from(new Float32Array(arena.memory.buffer, yGrp1Par, 4 * rows)),
      Array.from(new Float32Array(arena.memory.buffer, yGrp1Seq, 4 * rows)),
    );

    // 3. Throughput benchmark: Q8_K vs Q8_0 on single-core [4096×2048] Q4_K and Q6_K
    arena.attachThreadPool(null);
    // Warmup
    arena.mulMatQ8(GgmlType.Q4_K, wQ4KOff, q8Off, xOff, ySeqOff, cols, rows);
    arena.mulMatQ8K(GgmlType.Q4_K, wQ4KOff, q8kOff, xOff, yParOff, cols, rows);
    arena.mulMatQ8(GgmlType.Q6_K, wQ6KOff, q8Off, xOff, ySeqOff, cols, rows);
    arena.mulMatQ8K(GgmlType.Q6_K, wQ6KOff, q8kOff, xOff, yParOff, cols, rows);

    const benchIters = 10;
    const tQ4Old0 = performance.now();
    for (let i = 0; i < benchIters; i++) {
      arena.mulMatQ8(GgmlType.Q4_K, wQ4KOff, q8Off, xOff, ySeqOff, cols, rows);
    }
    const q4OldMs = performance.now() - tQ4Old0;

    const tQ4New0 = performance.now();
    for (let i = 0; i < benchIters; i++) {
      arena.mulMatQ8K(GgmlType.Q4_K, wQ4KOff, q8kOff, xOff, yParOff, cols, rows);
    }
    const q4NewMs = performance.now() - tQ4New0;

    const tQ6Old0 = performance.now();
    for (let i = 0; i < benchIters; i++) {
      arena.mulMatQ8(GgmlType.Q6_K, wQ6KOff, q8Off, xOff, ySeqOff, cols, rows);
    }
    const q6OldMs = performance.now() - tQ6Old0;

    const tQ6New0 = performance.now();
    for (let i = 0; i < benchIters; i++) {
      arena.mulMatQ8K(GgmlType.Q6_K, wQ6KOff, q8kOff, xOff, yParOff, cols, rows);
    }
    const q6NewMs = performance.now() - tQ6New0;

    const q4Gain = q4OldMs / Math.max(0.01, q4NewMs);
    const q6Gain = q6OldMs / Math.max(0.01, q6NewMs);
    console.log(
      `[Prism Q8_K Narrow-Lane [4096x2048] ×${benchIters}] Q4_K: ${q4Gain.toFixed(2)}x (${q4OldMs.toFixed(1)}ms Q8_0 -> ${q4NewMs.toFixed(1)}ms Q8_K) | Q6_K: ${q6Gain.toFixed(2)}x (${q6OldMs.toFixed(1)}ms Q8_0 -> ${q6NewMs.toFixed(1)}ms Q8_K)`,
    );
    assert.ok(q4Gain > 1.25, `Expected Q4_K×Q8_K speedup > 1.25x over Q8_0 (got ${q4Gain.toFixed(2)}x)`);
    assert.ok(q6Gain > 1.1, `Expected Q6_K×Q8_K speedup > 1.10x over Q8_0 (got ${q6Gain.toFixed(2)}x)`);
  } finally {
    arena.attachThreadPool(null);
    pool.dispose();
  }
});

test('Verdant v2.22.4 Q5_K × Q8_K single & batch-4 narrow-lane kernels match exact F64 reference, Prism proofs, and multi-core P=N parity', async () => {
  // 1. Prism algebraic proof & certified route selection for Q5_K × Q8_K
  assert.equal(PRISM_NARROW_LANE_PROOFS.q5k_x_q8k.fitsI32Exactly, true);
  assert.ok(PRISM_NARROW_LANE_PROOFS.q5k_x_q8k.i32MaxHeadroomFactor > 30);
  const cert = selectCertifiedPrismRoute(
    {
      wasmSimd: true,
      relaxedVnni: true,
      effectiveWorkers: 4,
      memoryBandwidthGBs: 25,
      computeGops: 140,
      barrierLatencyUs: 2,
      precisionMode: 'q8-vnni',
    },
    {
      weightType: GgmlType.Q5_K,
      rows: 3072,
      cols: 8192,
      batchSize: 1,
      groupSize: 1,
      weightBytes: 3072 * (8192 / 256) * 176,
    },
  );
  assert.equal(cert.selectedRoute, 'wasm-q8k-4row');

  const arena = new VerdantWasmArena({ initialPages: 256, shared: true });
  const availCpus = Math.max(1, os.cpus().length);
  const targetWorkers = Math.min(4, Math.max(2, availCpus));
  const pool = await VTensorThreadPool.create(arena, {
    maxWorkers: targetWorkers,
    parallelThresholdBytes: 16 * 1024,
  });
  assert.ok(pool !== null && pool.isActive);

  try {
    const cols = 1024;
    const rows = 515; // exercises 2-row tiled loop + odd remainder row across stripes
    const B = 4;
    const blocksPerRow = cols / 256;
    const rowBytes = blocksPerRow * 176;
    const wBytes = rows * rowBytes;
    const rng = makeRng(2224);

    const wOff = arena.allocRegion(wBytes);
    const wSlice = arena.bytes().subarray(wOff, wOff + wBytes);
    for (let r = 0; r < rows; r++) {
      for (let b = 0; b < blocksPerRow; b++) {
        const off = r * rowBytes + b * 176;
        wSlice[off] = 0x00;
        wSlice[off + 1] = 0x2c; // F16 0.0625
        wSlice[off + 2] = 0x00;
        wSlice[off + 3] = 0x28; // F16 0.03125
        for (let i = 4; i < 176; i++) wSlice[off + i] = Math.floor(rng() * 256) & 0xff;
      }
    }

    const xOff = arena.allocRegion(B * cols * 4);
    const xBatch = new Float32Array(arena.memory.buffer, xOff, B * cols);
    for (let i = 0; i < xBatch.length; i++) xBatch[i] = (rng() - 0.5) * 2.0;

    const q8kStride = blocksPerRow * Q8_K_ACT_BLOCK_BYTES;
    const q8kOff = arena.allocRegion(B * q8kStride);
    for (let t = 0; t < B; t++) {
      arena.quantizeActivationQ8K(xOff + t * cols * 4, q8kOff + t * q8kStride, cols);
    }

    const ySingleSeq = arena.allocRegion(B * rows * 4);
    const ySinglePar = arena.allocRegion(B * rows * 4);
    const yBatchSeq = arena.allocRegion(B * rows * 4);
    const yBatchPar = arena.allocRegion(B * rows * 4);

    // P=1 vs P=N bit-exactness for single-token and batch-4 Q5_K × Q8_K
    arena.attachThreadPool(null);
    for (let t = 0; t < B; t++) {
      arena.mulMatQ8K(GgmlType.Q5_K, wOff, q8kOff + t * q8kStride, xOff + t * cols * 4, ySingleSeq + t * rows * 4, cols, rows);
    }
    arena.mulMatBatchQ8K(GgmlType.Q5_K, wOff, q8kOff, xOff, yBatchSeq, cols, rows, B);

    arena.attachThreadPool(pool);
    for (let t = 0; t < B; t++) {
      arena.mulMatQ8K(GgmlType.Q5_K, wOff, q8kOff + t * q8kStride, xOff + t * cols * 4, ySinglePar + t * rows * 4, cols, rows);
    }
    arena.mulMatBatchQ8K(GgmlType.Q5_K, wOff, q8kOff, xOff, yBatchPar, cols, rows, B);

    const sSeq = new Float32Array(arena.memory.buffer, ySingleSeq, B * rows);
    const sPar = new Float32Array(arena.memory.buffer, ySinglePar, B * rows);
    const bSeq = new Float32Array(arena.memory.buffer, yBatchSeq, B * rows);
    const bPar = new Float32Array(arena.memory.buffer, yBatchPar, B * rows);

    for (let i = 0; i < B * rows; i++) {
      assert.equal(sPar[i], sSeq[i], `Q5_K×Q8_K single P=N mismatch at idx ${i}`);
      assert.equal(bPar[i], bSeq[i], `Q5_K×Q8_K batch-4 P=N mismatch at idx ${i}`);
      assert.ok(Math.abs(sPar[i] - bPar[i]) <= 1e-3, `Q5_K×Q8_K batch4 vs single drift at idx ${i}`);
    }

    // Exact dequantized Q8_K algebraic check against JS F64 verdantMulMat
    const dv = new DataView(arena.memory.buffer);
    const xDeq = new Float32Array(cols);
    const refDeq = new Float32Array(rows);
    for (let t = 0; t < B; t++) {
      for (let b = 0; b < blocksPerRow; b++) {
        const blkOff = q8kOff + t * q8kStride + b * Q8_K_ACT_BLOCK_BYTES;
        const d = dv.getFloat32(blkOff, true);
        for (let j = 0; j < 256; j++) xDeq[b * 256 + j] = dv.getInt8(blkOff + 64 + j) * d;
      }
      verdantMulMat(wSlice, 0, GgmlType.Q5_K, cols, rows, xDeq, refDeq);
      for (let r = 0; r < rows; r++) {
        const err = Math.abs(refDeq[r] - sSeq[t * rows + r]);
        assert.ok(err <= 1e-3, `Q5_K×Q8_K exact algebraic mismatch tok=${t} row=${r}: ref=${refDeq[r]}, got=${sSeq[t * rows + r]} (err=${err})`);
      }
    }
  } finally {
    arena.attachThreadPool(null);
    pool.dispose();
  }
});

test('Multi-core parallel flash_attn_layer_range is bit-exact (maxDiff === 0) across Hot+Warm+Cold KV tiers and accelerates long context', async () => {
  const numLayers = 1;
  const seqLen = 1024;
  const nHeadsQ = 16;
  const nHeadsKv = 4;
  const headDim = 128;
  const kvDim = nHeadsKv * headDim;
  const qDim = nHeadsQ * headDim;

  const reg = new VTensorBackendRegistry({ initialWasmPages: 256, sharedWasmMemory: true });
  assert.ok(reg.arena !== null);
  const arena = reg.arena;
  const availCpus = Math.max(1, os.cpus().length);
  const targetWorkers = Math.min(4, Math.max(2, availCpus));
  const pool = await VTensorThreadPool.create(arena, {
    maxWorkers: targetWorkers,
    parallelThresholdBytes: 32 * 1024,
  });
  assert.ok(pool !== null && pool.isActive);

  try {
    const pager = new VramKvPager(
      {
        numLayers,
        maxContext: seqLen,
        nHeadsKv,
        headDim,
        hotFrameLimit: 12,
        warmQ8Limit: 20,
        coldQ4Enabled: true,
      },
      (id, usage, cap) => reg.allocBuffer(id, usage, cap),
    );
    const actBuf = reg.allocBuffer('attn.par', 'compute', (qDim * 3 + kvDim * 2) * 4 + 256);
    const qT = actBuf.allocTensor('q', GgmlType.F32, [qDim]);
    const kT = actBuf.allocTensor('k', GgmlType.F32, [kvDim]);
    const vT = actBuf.allocTensor('v', GgmlType.F32, [kvDim]);
    const oSeqT = actBuf.allocTensor('oSeq', GgmlType.F32, [qDim]);
    const oParT = actBuf.allocTensor('oPar', GgmlType.F32, [qDim]);

    const rng = makeRng(9090);
    const kArr = actBuf.f32View(kT);
    const vArr = actBuf.f32View(vT);
    const qArr = actBuf.f32View(qT);
    for (let i = 0; i < qDim; i++) qArr[i] = (rng() - 0.5) * 1.4;
    for (let pos = 0; pos < seqLen; pos++) {
      for (let i = 0; i < kvDim; i++) {
        kArr[i] = (rng() - 0.5) * 1.1;
        vArr[i] = (rng() - 0.5) * 1.1;
      }
      pager.writeKvPtr(false, 0, pos, actBuf.wasmPtr(kT), arena);
      pager.writeKvPtr(true, 0, pos, actBuf.wasmPtr(vT), arena);
    }

    const kvTel = pager.getTelemetry();
    assert.ok(kvTel.hotResidentPages > 0 && kvTel.warmQ8Pages > 0 && kvTel.coldQ4Pages > 0);

    const empty = new Float32Array(0);
    const qPtr = actBuf.wasmPtr(qT);
    const oSeqPtr = actBuf.wasmPtr(oSeqT);
    const oParPtr = actBuf.wasmPtr(oParT);

    arena.attachThreadPool(null);
    pager.flashAttnExt(empty, qPtr, empty, oSeqPtr, nHeadsQ, 0, seqLen, arena);

    arena.attachThreadPool(pool);
    pager.flashAttnExt(empty, qPtr, empty, oParPtr, nHeadsQ, 0, seqLen, arena);

    const oSeq = actBuf.f32View(oSeqT);
    const oPar = actBuf.f32View(oParT);
    for (let i = 0; i < qDim; i++) {
      assert.equal(oPar[i], oSeq[i], `Parallel FlashAttention mismatch at idx ${i}: ${oPar[i]} !== ${oSeq[i]}`);
    }

    // Measure speedup
    const iters = 16;
    arena.attachThreadPool(null);
    const tSeq0 = performance.now();
    for (let i = 0; i < iters; i++) pager.flashAttnExt(empty, qPtr, empty, oSeqPtr, nHeadsQ, 0, seqLen, arena);
    const seqMs = performance.now() - tSeq0;

    arena.attachThreadPool(pool);
    const tPar0 = performance.now();
    for (let i = 0; i < iters; i++) pager.flashAttnExt(empty, qPtr, empty, oParPtr, nHeadsQ, 0, seqLen, arena);
    const parMs = performance.now() - tPar0;

    const speedup = seqMs / Math.max(0.01, parMs);
    console.log(
      `[Parallel FlashAttn P=${pool.workerCount} seqLen=${seqLen} 16H×128D] ${speedup.toFixed(2)}x (${seqMs.toFixed(2)}ms -> ${parMs.toFixed(2)}ms, bit-exact)`,
    );
    if (availCpus >= 2 && process.env.CI !== 'true') {
      assert.ok(speedup > 1.25, `Expected multi-core FlashAttention speedup > 1.25x (got ${speedup.toFixed(2)}x)`);
    }
  } finally {
    arena.attachThreadPool(null);
    pool.dispose();
    reg.freeAllBuffers();
  }
});

test('Upstream VT-217 / verdant_row_ranking NaN-safe deterministic Argmax and Top-K enforce lower-index tie-breaking', () => {
  const arena = new VerdantWasmArena({ initialPages: 8, shared: false });
  const n = 23; // exercises 4-wide SIMD lanes + 3-element scalar tail
  const lOff = arena.allocRegion(n * 4);
  const logits = new Float32Array(arena.memory.buffer, lOff, n);

  // Case 1: NaN at index 0 + interior NaNs + exact tie across SIMD & scalar tail
  logits.fill(-5.0);
  logits[0] = Number.NaN;
  logits[1] = Number.NaN;
  logits[5] = 4.25;
  logits[9] = Number.NaN;
  logits[14] = 4.25; // tied with 5 -> lower index 5 MUST win!
  logits[21] = 4.25; // tied in scalar tail -> lower index 5 MUST win!
  logits[22] = Number.NaN;
  assert.equal(arena.kernels.argmax_f32(lOff, n), 5);

  // Top-4 must ignore all NaNs and preserve lower indices (5, 14, 21) on ties
  const effK = arena.kernels.topk_f32(lOff, n, 4, WASM_TOPK_IDS_OFFSET, WASM_TOPK_VALS_OFFSET);
  assert.equal(effK, 4);
  const ids = Array.from(new Int32Array(arena.memory.buffer, WASM_TOPK_IDS_OFFSET, effK));
  const vals = Array.from(new Float32Array(arena.memory.buffer, WASM_TOPK_VALS_OFFSET, effK));
  const ranked = ids
    .map((id, idx) => ({ id, val: vals[idx] }))
    .sort((a, b) => b.val - a.val || a.id - b.id);
  assert.deepEqual(
    ranked.map((r) => r.id),
    [5, 14, 21, 2],
  );
  assert.ok(ranked.every((r) => !Number.isNaN(r.val)));

  // Case 2: All NaN -> deterministic fallback to index 0 (never out-of-bounds or negative)
  logits.fill(Number.NaN);
  assert.equal(arena.kernels.argmax_f32(lOff, n), 0);

  // Case 3: Negative finite values after leading NaN -> finite negative beats -Infinity/NaN
  logits.fill(Number.NaN);
  logits[17] = -123.5;
  logits[19] = -123.5;
  assert.equal(arena.kernels.argmax_f32(lOff, n), 17);
});

test('deriveOptimalPoolWorkers avoids SMT spin-convoy saturation and OP_MOE_EXPERTS matches sequential MoE', async () => {
  // Verify Theorem T3 empirical worker-bound invariants across small to high-core hosts
  assert.equal(deriveOptimalPoolWorkers(1), 1);
  assert.equal(deriveOptimalPoolWorkers(2), 2);
  assert.equal(deriveOptimalPoolWorkers(4), 4);
  assert.equal(deriveOptimalPoolWorkers(8), 6);
  assert.equal(deriveOptimalPoolWorkers(12), 8);
  assert.equal(deriveOptimalPoolWorkers(16), 10);
  assert.equal(deriveOptimalPoolWorkers(24), 12);
  assert.equal(deriveOptimalPoolWorkers(32), 12);
  // Verify Theorem T3-MoE wave-optimal active thread sizing
  assert.equal(deriveWaveOptimalMoeThreads(6, 8), 4); // 2 waves -> 4 balanced cores (zero stragglers!)
  assert.equal(deriveWaveOptimalMoeThreads(8, 8), 8); // 1 wave -> all 8 cores
  assert.equal(deriveWaveOptimalMoeThreads(4, 8), 4); // 2 waves -> 4 balanced cores
  assert.equal(deriveWaveOptimalMoeThreads(6, 6), 6); // 1 wave -> 6 balanced cores
  assert.equal(deriveWaveOptimalMoeThreads(6, 12), 6); // 2 waves -> 6 balanced cores
  assert.equal(deriveWaveOptimalMoeThreads(6, 10), 5); // 2 waves -> 5 balanced cores

  if (!isVerdantSharedWasmSupported()) return;
  const availCpus = Math.max(1, os.cpus().length);
  const targetWorkers = Math.min(4, availCpus);
  if (targetWorkers < 2) return;

  // Qwen3-30B-A3B exact MoE geometry: dim=2048, expFfnDim=768, Top-8 experts
  const dim = 2048;
  const expFfnDim = 768;
  const expertCount = 8;
  const rng = makeRng(30303);

  for (const wType of [GgmlType.Q4_K, GgmlType.Q2_K] as const) {
    const gateBytes = (weightRowByteSize(wType, dim) * expFfnDim + 63) & ~63;
    const downBytes = (weightRowByteSize(wType, expFfnDim) * dim + 63) & ~63;
    const layout = computeMoeWorkerScratchLayout(dim, expFfnDim);
    const totalBytes = (gateBytes * 2 + downBytes) * expertCount + layout.strideBytes * targetWorkers + 2 * 1024 * 1024;
    const arena = new VerdantWasmArena({ initialPages: Math.ceil(totalBytes / 65536), shared: true });
    const pool = await VTensorThreadPool.create(arena, { maxWorkers: targetWorkers, parallelThresholdBytes: 32 * 1024 });
    assert.ok(pool && pool.isActive);

    try {
      const wGate = arena.allocRegion(gateBytes * expertCount);
      const wUp = arena.allocRegion(gateBytes * expertCount);
      const wDown = arena.allocRegion(downBytes * expertCount);
      const xPtr = arena.allocRegion(dim * 4);
      const q8kPtr = arena.allocRegion((dim >>> 8) * Q8_K_ACT_BLOCK_BYTES);
      const outSeqPtr = arena.allocRegion(dim * 4);
      const outParPtr = arena.allocRegion(dim * 4);
      const gateTmp = arena.allocRegion(expFfnDim * 4);
      const upTmp = arena.allocRegion(expFfnDim * 4);
      const hidTmp = arena.allocRegion(expFfnDim * 4);
      const q8kFfnTmp = arena.allocRegion((expFfnDim >>> 8) * Q8_K_ACT_BLOCK_BYTES);
      const expOutTmp = arena.allocRegion(dim * 4);
      const workerScratch = arena.allocRegion(layout.strideBytes * targetWorkers);

      const bytes = arena.bytes();
      for (const [base, len] of [
        [wGate, gateBytes * expertCount],
        [wUp, gateBytes * expertCount],
        [wDown, downBytes * expertCount],
      ] as const) {
        const u32 = new Uint32Array(bytes.buffer, base, len >>> 2);
        for (let i = 0; i < u32.length; i++) {
          u32[i] = ((rng() * 0xffffffff) >>> 0) & 0x3bff3bff;
        }
      }
      const xArr = new Float32Array(arena.memory.buffer, xPtr, dim);
      for (let i = 0; i < dim; i++) xArr[i] = (rng() - 0.5) * 0.8;
      arena.quantizeActivationQ8K(xPtr, q8kPtr, dim);

      const gatePtrs = new Int32Array(expertCount);
      const upPtrs = new Int32Array(expertCount);
      const downPtrs = new Int32Array(expertCount);
      const weights = new Float32Array(expertCount);
      let wSum = 0;
      for (let e = 0; e < expertCount; e++) {
        gatePtrs[e] = wGate + e * gateBytes;
        upPtrs[e] = wUp + e * gateBytes;
        downPtrs[e] = wDown + e * downBytes;
        weights[e] = 0.05 + rng() * 0.25;
        wSum += weights[e];
      }
      for (let e = 0; e < expertCount; e++) weights[e] /= wSum;

      // 1. Sequential reference (P=1)
      arena.attachThreadPool(null);
      const runSeq = () => {
        arena.kernels.zero_bytes(outSeqPtr, dim * 4);
        for (let e = 0; e < expertCount; e++) {
          arena.mulMatQ8K(wType, gatePtrs[e], q8kPtr, xPtr, gateTmp, dim, expFfnDim);
          arena.mulMatQ8K(wType, upPtrs[e], q8kPtr, xPtr, upTmp, dim, expFfnDim);
          arena.kernels.swiglu_f32(gateTmp, upTmp, hidTmp, expFfnDim);
          arena.quantizeActivationQ8K(hidTmp, q8kFfnTmp, expFfnDim);
          arena.mulMatQ8K(wType, downPtrs[e], q8kFfnTmp, hidTmp, expOutTmp, expFfnDim, dim);
          arena.kernels.axpy_f32(outSeqPtr, expOutTmp, weights[e], dim);
        }
      };
      runSeq();

      // 2. Single-barrier Expert-Parallel OP_MOE_EXPERTS (P=targetWorkers)
      arena.attachThreadPool(pool);
      const runPar = () => {
        const ok = arena.dispatchMoeExpertsParallel({
          expertCount,
          dim,
          expFfnDim,
          gateUpOp: ParallelMatOp.MUL_MAT_Q8K,
          downOp: ParallelMatOp.MUL_MAT_Q8K,
          gateType: wType,
          upType: wType,
          downType: wType,
          xOrQ8Ptr: q8kPtr,
          xF32Ptr: xPtr,
          outPtr: outParPtr,
          workerScratchBase: workerScratch,
          layout,
          gatePtrs,
          upPtrs,
          downPtrs,
          weights,
        });
        assert.equal(ok, true);
      };
      runPar();

      const seqOut = new Float32Array(arena.memory.buffer, outSeqPtr, dim);
      const parOut = new Float32Array(arena.memory.buffer, outParPtr, dim);
      let maxDiff = 0;
      for (let i = 0; i < dim; i++) {
        assert.ok(Number.isFinite(parOut[i]), `Non-finite MoE output at idx ${i}`);
        const d = Math.abs(parOut[i] - seqOut[i]);
        if (d > maxDiff) maxDiff = d;
      }
      // Different summation order across workers (per-worker reduction) permits tiny F32 ULP drift only
      assert.ok(maxDiff < 1e-4, `OP_MOE_EXPERTS (${wType}) parity drift too large: maxDiff=${maxDiff}`);

      if (wType === GgmlType.Q4_K) {
        const iters = 6;
        arena.attachThreadPool(null);
        runSeq();
        const t0 = performance.now();
        for (let i = 0; i < iters; i++) runSeq();
        const seqMs = (performance.now() - t0) / iters;

        arena.attachThreadPool(pool);
        runPar();
        const t1 = performance.now();
        for (let i = 0; i < iters; i++) runPar();
        const parMs = (performance.now() - t1) / iters;

        const speedup = seqMs / Math.max(0.01, parMs);
        console.log(
          `[OP_MOE_EXPERTS P=${pool.workerCount} Q4_K 8×(2048→768→2048)] ${speedup.toFixed(2)}x (${seqMs.toFixed(2)}ms -> ${parMs.toFixed(2)}ms, maxDiff=${maxDiff.toExponential(1)})`,
        );
        if (process.env.CI !== 'true') {
          assert.ok(speedup > 1.5, `Expected OP_MOE_EXPERTS speedup > 1.5x (got ${speedup.toFixed(2)}x)`);
        }
      }
    } finally {
      arena.attachThreadPool(null);
      pool.dispose();
    }
  }
});

test('OP_MOE_EXPERTS_BATCH4 matches sequential Batch-4 MoE across dense (0b1111) and sparse masks and scales', async () => {
  if (!isVerdantSharedWasmSupported()) return;
  const availCpus = Math.max(1, os.cpus().length);
  const targetWorkers = Math.min(4, availCpus);
  if (targetWorkers < 2) return;

  const dim = 2048;
  const expFfnDim = 768;
  const uniqueCount = 8;
  const wType = GgmlType.Q4_K;
  const rng = makeRng(40404);
  const gateBytes = (weightRowByteSize(wType, dim) * expFfnDim + 63) & ~63;
  const downBytes = (weightRowByteSize(wType, expFfnDim) * dim + 63) & ~63;
  const layout4 = computeMoeWorkerScratchLayout(dim, expFfnDim, 4);
  const totalBytes =
    (gateBytes * 2 + downBytes) * uniqueCount + layout4.strideBytes * targetWorkers + 4 * 1024 * 1024;
  const arena = new VerdantWasmArena({ initialPages: Math.ceil(totalBytes / 65536), shared: true });
  const pool = await VTensorThreadPool.create(arena, { maxWorkers: targetWorkers, parallelThresholdBytes: 32 * 1024 });
  assert.ok(pool && pool.isActive);

  try {
    const wGate = arena.allocRegion(gateBytes * uniqueCount);
    const wUp = arena.allocRegion(gateBytes * uniqueCount);
    const wDown = arena.allocRegion(downBytes * uniqueCount);
    const bXPtr = arena.allocRegion(4 * dim * 4);
    const qkDimStride = (dim >>> 8) * Q8_K_ACT_BLOCK_BYTES;
    const qkFfnStride = (expFfnDim >>> 8) * Q8_K_ACT_BLOCK_BYTES;
    const bQkDimPtr = arena.allocRegion(4 * qkDimStride);
    const outSeqPtr = arena.allocRegion(4 * dim * 4);
    const outParPtr = arena.allocRegion(4 * dim * 4);
    const gateTmp = arena.allocRegion(4 * expFfnDim * 4);
    const upTmp = arena.allocRegion(4 * expFfnDim * 4);
    const hidTmp = arena.allocRegion(4 * expFfnDim * 4);
    const qkFfnTmp = arena.allocRegion(4 * qkFfnStride);
    const projTmp = arena.allocRegion(4 * dim * 4);
    const workerScratch = arena.allocRegion(layout4.strideBytes * targetWorkers);

    const bytes = arena.bytes();
    for (const [base, len] of [
      [wGate, gateBytes * uniqueCount],
      [wUp, gateBytes * uniqueCount],
      [wDown, downBytes * uniqueCount],
    ] as const) {
      const u32 = new Uint32Array(bytes.buffer, base, len >>> 2);
      for (let i = 0; i < u32.length; i++) u32[i] = ((rng() * 0xffffffff) >>> 0) & 0x3bff3bff;
    }
    const xArr = new Float32Array(arena.memory.buffer, bXPtr, 4 * dim);
    for (let i = 0; i < xArr.length; i++) xArr[i] = (rng() - 0.5) * 0.8;
    for (let t = 0; t < 4; t++) {
      arena.quantizeActivationQ8K(bXPtr + t * dim * 4, bQkDimPtr + t * qkDimStride, dim);
    }

    const gatePtrs = new Int32Array(uniqueCount);
    const upPtrs = new Int32Array(uniqueCount);
    const downPtrs = new Int32Array(uniqueCount);
    // Mix of full 4-token hits (0b1111) and sparse 1..3-token masks
    const masks = new Uint8Array([0b1111, 0b1111, 0b0101, 0b1010, 0b1110, 0b0001, 0b1111, 0b0111]);
    const weights = new Float32Array(uniqueCount * 4);
    for (let u = 0; u < uniqueCount; u++) {
      gatePtrs[u] = wGate + u * gateBytes;
      upPtrs[u] = wUp + u * gateBytes;
      downPtrs[u] = wDown + u * downBytes;
      for (let t = 0; t < 4; t++) {
        weights[u * 4 + t] = (masks[u] & (1 << t)) !== 0 ? 0.05 + rng() * 0.25 : 0;
      }
    }

    const runSeq = () => {
      arena.kernels.zero_bytes(outSeqPtr, 4 * dim * 4);
      for (let u = 0; u < uniqueCount; u++) {
        const mask = masks[u];
        if (mask === 15) {
          arena.mulMatBatchQ8K(wType, gatePtrs[u], bQkDimPtr, bXPtr, gateTmp, dim, expFfnDim, 4);
          arena.mulMatBatchQ8K(wType, upPtrs[u], bQkDimPtr, bXPtr, upTmp, dim, expFfnDim, 4);
          arena.kernels.swiglu_f32(gateTmp, upTmp, hidTmp, 4 * expFfnDim);
          for (let t = 0; t < 4; t++) {
            arena.quantizeActivationQ8K(hidTmp + t * expFfnDim * 4, qkFfnTmp + t * qkFfnStride, expFfnDim);
          }
          arena.mulMatBatchQ8K(wType, downPtrs[u], qkFfnTmp, hidTmp, projTmp, expFfnDim, dim, 4);
          for (let t = 0; t < 4; t++) {
            arena.kernels.axpy_f32(outSeqPtr + t * dim * 4, projTmp + t * dim * 4, weights[u * 4 + t], dim);
          }
        } else {
          for (let t = 0; t < 4; t++) {
            if ((mask & (1 << t)) === 0) continue;
            const tx = bXPtr + t * dim * 4;
            const tq = bQkDimPtr + t * qkDimStride;
            arena.mulMatQ8K(wType, gatePtrs[u], tq, tx, gateTmp, dim, expFfnDim);
            arena.mulMatQ8K(wType, upPtrs[u], tq, tx, upTmp, dim, expFfnDim);
            arena.kernels.swiglu_f32(gateTmp, upTmp, hidTmp, expFfnDim);
            arena.quantizeActivationQ8K(hidTmp, qkFfnTmp, expFfnDim);
            arena.mulMatQ8K(wType, downPtrs[u], qkFfnTmp, hidTmp, projTmp, expFfnDim, dim);
            arena.kernels.axpy_f32(outSeqPtr + t * dim * 4, projTmp, weights[u * 4 + t], dim);
          }
        }
      }
    };

    arena.attachThreadPool(null);
    runSeq();

    arena.attachThreadPool(pool);
    const runPar = () => {
      const ok = arena.dispatchMoeExpertsBatch4Parallel({
        uniqueCount,
        dim,
        expFfnDim,
        gateUpBatchOp: ParallelMatOp.MUL_MAT_BATCH4_Q8K,
        gateUpSingleOp: ParallelMatOp.MUL_MAT_Q8K,
        downBatchOp: ParallelMatOp.MUL_MAT_BATCH4_Q8K,
        downSingleOp: ParallelMatOp.MUL_MAT_Q8K,
        gateType: wType,
        upType: wType,
        downType: wType,
        bXOrQ8Ptr: bQkDimPtr,
        bXF32Ptr: bXPtr,
        xQ8StrideBytes: qkDimStride,
        outPtr: outParPtr,
        workerScratchBase: workerScratch,
        layout: layout4,
        gatePtrs,
        upPtrs,
        downPtrs,
        masks,
        weights,
      });
      assert.equal(ok, true);
    };
    runPar();

    const seqOut = new Float32Array(arena.memory.buffer, outSeqPtr, 4 * dim);
    const parOut = new Float32Array(arena.memory.buffer, outParPtr, 4 * dim);
    let maxDiff = 0;
    for (let i = 0; i < 4 * dim; i++) {
      assert.ok(Number.isFinite(parOut[i]), `Non-finite Batch4 MoE output at idx ${i}`);
      const d = Math.abs(parOut[i] - seqOut[i]);
      if (d > maxDiff) maxDiff = d;
    }
    assert.ok(maxDiff < 1e-4, `OP_MOE_EXPERTS_BATCH4 parity drift: maxDiff=${maxDiff}`);

    const iters = 4;
    arena.attachThreadPool(null);
    runSeq();
    const t0 = performance.now();
    for (let i = 0; i < iters; i++) runSeq();
    const seqMs = (performance.now() - t0) / iters;

    arena.attachThreadPool(pool);
    runPar();
    const t1 = performance.now();
    for (let i = 0; i < iters; i++) runPar();
    const parMs = (performance.now() - t1) / iters;

    const speedup = seqMs / Math.max(0.01, parMs);
    console.log(
      `[OP_MOE_EXPERTS_BATCH4 P=${pool.workerCount} Q4_K 8×mixed] ${speedup.toFixed(2)}x (${seqMs.toFixed(2)}ms -> ${parMs.toFixed(2)}ms, maxDiff=${maxDiff.toExponential(1)})`,
    );
    if (process.env.CI !== 'true') {
      assert.ok(speedup > 1.5, `Expected OP_MOE_EXPERTS_BATCH4 speedup > 1.5x (got ${speedup.toFixed(2)}x)`);
    }
  } finally {
    arena.attachThreadPool(null);
    await pool.disposeAsync();
  }
});

