import test from 'node:test';
import assert from 'node:assert/strict';
import { GgmlType } from './gguf';
import {
  computeVramKvPageSizes,
  derivePerformanceContract,
  evaluateContractCompliance,
  partitionRowStripes,
  SPILL_FREE_BATCH_WIDTH,
  WASM32_SAFE_MEMORY_CAP_BYTES,
  type EnvironmentProfile,
  type ModelGeometrySpec,
} from './performanceContract';

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

  // Constrained budget where KV itself exceeds available memory safely scales context down
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
  assert.equal(cTight.contextScaledToFitBudget, true);
  assert.ok(cTight.effectiveMaxContext < 16384);
  assert.ok(cTight.effectiveMaxContext >= 64);

  // Theorem T4b: 3B 1.88 GB model exceeds WASM32 1.5 GiB budget -> automatically engages OPFS-backed vRAM weight paging!
  const m3b = makeModelSpec({
    name: 'Llama-3.2-3B-Q4_K_M',
    layers: 28,
    dim: 3072,
    ffnDim: 8192,
    heads: 24,
    kvHeads: 8,
    vocab: 128256,
    weightMB: 1926,
    ctx: 8192,
    type: GgmlType.Q4_K,
  });
  const c3b = derivePerformanceContract(SAMPLE_ENV, m3b, {
    requestedMaxContext: 2048,
    backingKind: 'opfs-sync',
    weightResidencyPolicy: 'auto',
    backingQuotaBytes: 16 * 1024 * 1024 * 1024,
  });
  assert.equal(c3b.weightResidencyMode, 'opfs-paged');
  assert.equal(c3b.streamEmbeddings, true);
  assert.equal(c3b.streamLmHead, true);
  assert.ok(c3b.weightLayerFrameSlots >= 1 && c3b.weightLayerFrameSlots < 28);
  assert.ok(c3b.projectedResidentWeightBytes < m3b.totalWeightBufferBytes);
  assert.ok(c3b.projectedTotalArenaBytes <= SAMPLE_ENV.memoryBudgetBytes);
  assert.ok(c3b.opfsIoOverheadMsPerToken > 0);
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
