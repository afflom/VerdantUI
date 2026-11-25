import test from 'node:test';
import assert from 'node:assert/strict';
import { GgmlType } from './gguf';
import {
  computeVramKvPageSizes,
  deriveHostMemoryBudgetBytes,
  derivePerformanceContract,
  evaluateContractCompliance,
  partitionRowStripes,
  SPILL_FREE_BATCH_WIDTH,
  WASM32_SAFE_MEMORY_CAP_BYTES,
  type EnvironmentProfile,
  type ModelGeometrySpec,
} from './performanceContract';
import { probeMaxWasmMemoryPages, VerdantWasmArena } from './verdantWasmSimd';
import { VRAM_LM_HEAD_TILE_MAX_BYTES, VRAM_LM_HEAD_TILE_TARGET_BYTES } from './vramWeightPager';

const SAMPLE_ENV: EnvironmentProfile = {
  simdTier: 'relaxed-vnni',
  relaxedSimdSupported: true,
  hardwareConcurrency: 4,
  sharedMemorySupported: false,
  effectiveWorkers: 1,
  memoryBandwidthGBs: 3.6,
  computeGops: 18.0,
  ridgeIntensityOpsPerByte: 5.0,
  workerSyncLatencyUs: 0,
  memoryBudgetBytes: WASM32_SAFE_MEMORY_CAP_BYTES,
  calibrationMs: 3.5,
};

function makeModelSpec(params: {
  name: string;
  layers: number;
  dim: number;
  ffnDim: number;
  heads: number;
  kvHeads: number;
  vocab: number;
  weightMB: number;
  ctx: number;
  type: GgmlType;
  quantFraction?: number;
}): ModelGeometrySpec {
  const headDim = Math.floor(params.dim / params.heads);
  const bytes = Math.round(params.weightMB * 1024 * 1024);
  const avgBpw = params.type === GgmlType.F32 ? 4 : 0.5625;
  return {
    hparams: {
      architecture: 'llama',
      name: params.name,
      contextLength: params.ctx,
      embeddingLength: params.dim,
      blockCount: params.layers,
      feedForwardLength: params.ffnDim,
      headCount: params.heads,
      headCountKv: params.kvHeads,
      headDim,
      ropeDimensionCount: headDim,
      ropeFreqBase: 10000,
      ropeFreqScale: 1.0,
      ropeType: 'norm',
      rmsNormEps: 1e-5,
      vocabSize: params.vocab,
      bosTokenId: 1,
      eosTokenId: 2,
      unkTokenId: 0,
      addBosToken: true,
      tokenizerModel: 'gpt2',
      preTokenizer: 'default',
      chatTemplate: null,
      primaryQuantization: params.type === GgmlType.F32 ? 'F32' : 'Q4_K',
    },
    activeWeightBytes: bytes,
    lmHeadWeightBytes: Math.round(bytes * 0.14),
    activeParamCount: Math.round(bytes / avgBpw),
    totalWeightBufferBytes: bytes,
    dominantWeightType: params.type,
    quantizedWeightFraction: params.quantFraction ?? (params.type === GgmlType.F32 ? 0 : 1),
  };
}

test('Prism Theorems T1 & T2: monotonic roofline scaling from 260K to 3B and spill-free batch width', () => {
  const m260k = makeModelSpec({
    name: '260K',
    layers: 5,
    dim: 64,
    ffnDim: 172,
    heads: 8,
    kvHeads: 4,
    vocab: 512,
    weightMB: 1.1,
    ctx: 512,
    type: GgmlType.F32,
  });
  const m135m = makeModelSpec({
    name: '135M',
    layers: 30,
    dim: 576,
    ffnDim: 1536,
    heads: 9,
    kvHeads: 3,
    vocab: 49152,
    weightMB: 87.6,
    ctx: 8192,
    type: GgmlType.Q4_0,
  });
  const m360m = makeModelSpec({
    name: '360M',
    layers: 32,
    dim: 960,
    ffnDim: 2560,
    heads: 15,
    kvHeads: 5,
    vocab: 49152,
    weightMB: 258.1,
    ctx: 8192,
    type: GgmlType.Q4_K,
  });
  const m1b = makeModelSpec({
    name: '1B',
    layers: 16,
    dim: 2048,
    ffnDim: 8192,
    heads: 32,
    kvHeads: 8,
    vocab: 128256,
    weightMB: 807.6,
    ctx: 8192,
    type: GgmlType.Q4_K,
  });

  const c260k = derivePerformanceContract(SAMPLE_ENV, m260k, { requestedMaxContext: 512 });
  const c135m = derivePerformanceContract(SAMPLE_ENV, m135m, { requestedMaxContext: 1024 });
  const c360m = derivePerformanceContract(SAMPLE_ENV, m360m, { requestedMaxContext: 1024 });
  const c1b = derivePerformanceContract(SAMPLE_ENV, m1b, { requestedMaxContext: 2048 });

  // Theorem T1 monotonicity: larger active weight footprint => strictly lower decode ceiling
  assert.ok(c260k.rooflineDecodeTokSec > c135m.rooflineDecodeTokSec);
  assert.ok(c135m.rooflineDecodeTokSec > c360m.rooflineDecodeTokSec);
  assert.ok(c360m.rooflineDecodeTokSec > c1b.rooflineDecodeTokSec);

  // Theorem T2: F32 model disables Q8 activations; quantized models enable Q8-VNNI
  assert.equal(c260k.useQ8Activations, false);
  assert.equal(c135m.useQ8Activations, true);
  assert.equal(c360m.useQ8Activations, true);
  assert.equal(c1b.useQ8Activations, true);
  assert.ok(c135m.optimalPrefillBatchSize >= 1 && c135m.optimalPrefillBatchSize <= SPILL_FREE_BATCH_WIDTH);

  // Strict F64 precision mode disables Q8 activations and adjusts T1 ceiling accordingly
  const c135mStrict = derivePerformanceContract(SAMPLE_ENV, m135m, {
    precisionMode: 'strict-f64',
    requestedMaxContext: 1024,
  });
  assert.equal(c135mStrict.useQ8Activations, false);
  assert.ok(c135mStrict.rooflineDecodeTokSec <= c135m.rooflineDecodeTokSec);
});

test('Prism Theorem T3: deterministic row-striped partitioning covers all rows without overlap or gaps', () => {
  for (const totalRows of [1, 2, 7, 15, 64, 576, 49152]) {
    for (const workers of [1, 2, 3, 4, 8]) {
      const stripes = partitionRowStripes(totalRows, workers, 2);
      let cursor = 0;
      for (const s of stripes) {
        assert.equal(s.rowStart, cursor, `Gap or overlap at rows=${totalRows}, P=${workers}`);
        assert.ok(s.rowCount > 0);
        cursor += s.rowCount;
      }
      assert.equal(cursor, totalRows);
    }
  }
});

test('Prism Theorem T4: 3-tier Hot/Warm/Cold-Q4 capacity law safely scales context under tight memory budgets', () => {
  const sizes = computeVramKvPageSizes(32, 8, 128);
  assert.ok(sizes.hotF32Bytes > sizes.warmQ8Bytes * 3.5);
  assert.ok(sizes.warmQ8Bytes > sizes.coldQ4Bytes * 1.7);

  // 1B model requesting 2048 ctx must automatically engage 3-tier Cold Q4 paging
  const m1b = makeModelSpec({
    name: '1B',
    layers: 16,
    dim: 2048,
    ffnDim: 8192,
    heads: 32,
    kvHeads: 8,
    vocab: 128256,
    weightMB: 807.6,
    ctx: 8192,
    type: GgmlType.Q4_K,
  });
  const c1b = derivePerformanceContract(SAMPLE_ENV, m1b, { requestedMaxContext: 2048 });
  assert.equal(c1b.coldQ4Enabled, true);
  assert.ok(c1b.projectedKvContractBytes < c1b.projectedRawF32KvBytes * 0.45);
  assert.ok(c1b.projectedTotalArenaBytes <= SAMPLE_ENV.memoryBudgetBytes);

  // Constrained budget where KV exceeds RAM engages Tier-3 Swapped Q4 while preserving FULL 16,384 context!
  const tightEnv: EnvironmentProfile = { ...SAMPLE_ENV, memoryBudgetBytes: 120 * 1024 * 1024 };
  const mBigKv = makeModelSpec({
    name: 'TightKv',
    layers: 32,
    dim: 1024,
    ffnDim: 3072,
    heads: 16,
    kvHeads: 16,
    vocab: 32000,
    weightMB: 40,
    ctx: 16384,
    type: GgmlType.Q4_K,
  });
  const cTight = derivePerformanceContract(tightEnv, mBigKv, { requestedMaxContext: 16384 });
  assert.equal(cTight.effectiveMaxContext, 16384);
  assert.equal(cTight.contextScaledToFitBudget, false);
  assert.equal(cTight.coldQ4Enabled, true);
  assert.equal(cTight.swapQ4Enabled, true);
  assert.ok(cTight.coldQ4RamLimit < cTight.totalKvPages);
  assert.ok(cTight.projectedTotalArenaBytes <= tightEnv.memoryBudgetBytes);

  // Theorem T4b + Full Context (3.25 GiB probed WASM32 ceiling):
  // Llama-3.2-3B (~1.88 GiB Q4_K_M) stays 100% resident-all (0 ms OPFS weight I/O/token)
  // while preserving FULL 131,072 context via 4-tier vRAM KV swap!
  const m3b = makeModelSpec({
    name: 'Llama-3.2-3B-Q4_K_M',
    layers: 28,
    dim: 3072,
    ffnDim: 8192,
    heads: 24,
    kvHeads: 8,
    vocab: 128256,
    weightMB: 1926,
    ctx: 131072,
    type: GgmlType.Q4_K,
  });
  const c3bResident = derivePerformanceContract(SAMPLE_ENV, m3b, {
    backingKind: 'opfs-sync',
    weightResidencyPolicy: 'auto',
    backingQuotaBytes: 16 * 1024 * 1024 * 1024,
  });
  assert.equal(c3bResident.effectiveMaxContext, 131072);
  assert.equal(c3bResident.swapQ4Enabled, true);
  assert.equal(c3bResident.weightResidencyMode, 'resident-all');
  assert.equal(c3bResident.streamEmbeddings, false);
  assert.equal(c3bResident.streamLmHead, false);
  assert.equal(c3bResident.opfsIoOverheadMsPerToken, 0);
  assert.ok(c3bResident.projectedTotalArenaBytes <= SAMPLE_ENV.memoryBudgetBytes);

  // Under a constrained 1.5 GiB budget (e.g. mobile/OOM step-down), the same 3B model
  // automatically engages OPFS weight paging + 4-tier KV swap without dropping context!
  const constrainedEnv: EnvironmentProfile = {
    ...SAMPLE_ENV,
    memoryBudgetBytes: Math.floor(1.5 * 1024 * 1024 * 1024),
  };
  const c3bPaged = derivePerformanceContract(constrainedEnv, m3b, {
    backingKind: 'opfs-sync',
    weightResidencyPolicy: 'auto',
    backingQuotaBytes: 16 * 1024 * 1024 * 1024,
  });
  assert.equal(c3bPaged.effectiveMaxContext, 131072);
  assert.equal(c3bPaged.swapQ4Enabled, true);
  assert.equal(c3bPaged.weightResidencyMode, 'opfs-paged');
  assert.equal(c3bPaged.streamEmbeddings, true);
  assert.equal(c3bPaged.streamLmHead, true);
  assert.ok(c3bPaged.weightLayerFrameSlots >= 1 && c3bPaged.weightLayerFrameSlots < 28);
  assert.ok(c3bPaged.projectedResidentWeightBytes < m3b.totalWeightBufferBytes);
  assert.ok(c3bPaged.projectedTotalArenaBytes <= constrainedEnv.memoryBudgetBytes);
  assert.ok(c3bPaged.opfsIoOverheadMsPerToken > 0);
  assert.ok(
    c3bPaged.lmHeadTileTargetBytes >= VRAM_LM_HEAD_TILE_TARGET_BYTES &&
      c3bPaged.lmHeadTileTargetBytes <= VRAM_LM_HEAD_TILE_MAX_BYTES,
  );

  // 7B model (4.35 GB) on 3.25 GiB host engages OPFS weight paging with scaled 32 MiB LM-head tile!
  const m7b = makeModelSpec({
    name: 'Qwen2.5-7B-Q4_K_M',
    layers: 28,
    dim: 3584,
    ffnDim: 18944,
    heads: 28,
    kvHeads: 4,
    vocab: 152064,
    weightMB: 4350,
    ctx: 32768,
    type: GgmlType.Q4_K,
  });
  const c7b = derivePerformanceContract(SAMPLE_ENV, m7b, {
    backingKind: 'opfs-sync',
    weightResidencyPolicy: 'auto',
    backingQuotaBytes: 16 * 1024 * 1024 * 1024,
  });
  assert.equal(c7b.weightResidencyMode, 'opfs-paged');
  assert.equal(c7b.lmHeadTileTargetBytes, VRAM_LM_HEAD_TILE_MAX_BYTES);
  assert.ok(c7b.projectedTotalArenaBytes <= SAMPLE_ENV.memoryBudgetBytes);
});

test('Dynamic WASM32 ceiling probe unlocks > 2 GiB arena budget without contrived caps', () => {
  const maxPages = probeMaxWasmMemoryPages(true);
  assert.ok(maxPages >= 32768, `Expected probed WASM32 maxPages >= 32768, got ${maxPages}`);
  const arena = new VerdantWasmArena({ initialPages: 32, shared: true });
  assert.equal(arena.maxPages, maxPages);
  assert.equal(arena.maxCapacityBytes, maxPages * 65536);

  // Probed 3.75 GiB -> 3.25 GiB safe budget; probed 2.0 GiB -> 1.625 GiB safe budget
  const budget375 = deriveHostMemoryBudgetBytes(61440 * 65536);
  assert.equal(budget375, WASM32_SAFE_MEMORY_CAP_BYTES);
  const budget200 = deriveHostMemoryBudgetBytes(32768 * 65536);
  assert.equal(budget200, 32768 * 65536 - 384 * 1024 * 1024);
});

test('Prism Theorem T5: compliance evaluation distinguishes UPHELD, EXCEEDS_ESTIMATE, and BELOW_CONTRACT', () => {
  const m135m = makeModelSpec({
    name: '135M',
    layers: 30,
    dim: 576,
    ffnDim: 1536,
    heads: 9,
    kvHeads: 3,
    vocab: 49152,
    weightMB: 87.6,
    ctx: 8192,
    type: GgmlType.Q4_0,
  });
  const base = derivePerformanceContract(SAMPLE_ENV, m135m, { requestedMaxContext: 1024 });
  assert.equal(base.verdict, 'CALIBRATED');

  const upheld = evaluateContractCompliance(base, base.rooflineDecodeTokSec * 0.78, 120, 0);
  assert.equal(upheld.verdict, 'UPHELD');
  assert.ok(upheld.rooflineEfficiencyPct >= 75 && upheld.rooflineEfficiencyPct <= 81);

  const exceeds = evaluateContractCompliance(base, base.rooflineDecodeTokSec * 1.05, 160, 0);
  assert.equal(exceeds.verdict, 'EXCEEDS_ESTIMATE');

  const belowSlow = evaluateContractCompliance(base, base.rooflineDecodeTokSec * 0.2, 20, 0);
  assert.equal(belowSlow.verdict, 'BELOW_CONTRACT');

  // Any CPU fallback immediately violates the contract regardless of speed!
  const belowFallback = evaluateContractCompliance(base, base.rooflineDecodeTokSec * 0.9, 120, 1);
  assert.equal(belowFallback.verdict, 'BELOW_CONTRACT');
});
