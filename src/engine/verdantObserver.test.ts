import test from 'node:test';
import assert from 'node:assert/strict';
import {
  INTERNAL_BOOTSTRAP_MODEL,
  MODEL_CATALOG,
  MODEL_CATEGORIES,
  findCatalogModelById,
  getModelsByCategory,
  isInternalBootstrapModel,
} from './modelCatalog';
import {
  buildFallbackEnvironmentProfile,
  observeCatalogScaling,
  type ObserverRuntimeInput,
} from './verdantObserver';
import { DEFAULT_SAMPLING_CONFIG } from './llamaRuntime';

const GIB = 1024 * 1024 * 1024;

function makeBaseInput(overrides: Partial<ObserverRuntimeInput> = {}): ObserverRuntimeInput {
  return {
    activeModelId: INTERNAL_BOOTSTRAP_MODEL.id,
    activeModelName: INTERNAL_BOOTSTRAP_MODEL.name,
    env: {
      ...buildFallbackEnvironmentProfile(),
      memoryBandwidthGBs: 28,
      computeGops: 320,
      ridgeIntensityOpsPerByte: 320 / 28,
      memoryBudgetBytes: Math.floor(3.25 * GIB),
    },
    sampling: DEFAULT_SAMPLING_CONFIG,
    precisionMode: 'q8-vnni',
    weightResidencyPolicy: 'auto',
    opfsStorage: {
      supported: true,
      usageBytes: 0,
      quotaBytes: 64 * GIB,
      storedModels: [],
      partialModels: [],
    },
    backingBandwidthMBs: 3200,
    ...overrides,
  };
}

test('MODEL_CATALOG excludes dev/conformance models and spans 0.5B to 30B–32B across factual categories', () => {
  assert.ok(MODEL_CATALOG.length >= 10);
  for (const model of MODEL_CATALOG) {
    assert.ok(
      !/stories260k|stories15m|tinyllama.*stories/i.test(model.id + ' ' + model.name),
      `Dev/conformance checkpoint leaked into user-facing MODEL_CATALOG: ${model.id}`,
    );
    assert.ok(model.url.startsWith('https://huggingface.co/'), `Expected verified HF URL for ${model.id}`);
    assert.equal(model.bundled, false);
  }
  assert.equal(isInternalBootstrapModel(INTERNAL_BOOTSTRAP_MODEL.id), true);
  assert.equal(isInternalBootstrapModel('qwen3-coder-30b-a3b-q4_k_m'), false);

  for (const cat of MODEL_CATEGORIES) {
    const models = getModelsByCategory(cat.id);
    assert.ok(models.length >= 2, `Category ${cat.id} should have at least 2 models`);
  }

  const maxBillions = Math.max(...MODEL_CATALOG.map((m) => m.geometry.paramsBillions));
  const minBillions = Math.min(...MODEL_CATALOG.map((m) => m.geometry.paramsBillions));
  assert.ok(maxBillions >= 30, `Expected catalog to scale to >= 30B, got ${maxBillions}B`);
  assert.ok(minBillions <= 0.5, `Expected compact category to include 0.5B, got ${minBillions}B`);
});

test('Verdant scaling evaluator fits 30B-A3B Sparse MoE within WASM32 arena and proves superior throughput over 32B Dense', () => {
  const input = makeBaseInput();
  const report = observeCatalogScaling(input, 'coding');

  const moe30b = report.models.find((m) => m.model.id === 'qwen3-coder-30b-a3b-q4_k_m');
  const dense32b = report.models.find((m) => m.model.id === 'qwen2.5-coder-32b-instruct-q4_0');
  assert.ok(moe30b, 'Expected Qwen3-Coder 30B-A3B in coding category');
  assert.ok(dense32b, 'Expected Qwen2.5-Coder 32B Dense in coding category');

  assert.equal(moe30b.residencyStrategy, 'opfs-sparse-moe');
  assert.equal(dense32b.residencyStrategy, 'opfs-paged-dimm');
  assert.ok(moe30b.moeExpertCacheSlots > 0, 'Expected bounded LRU Sparse Expert Cache slots > 0');
  assert.equal(dense32b.moeExpertCacheSlots, 0);

  // Both 30B+ checkpoints must remain strictly bounded inside the 3.25 GiB WASM32 host budget
  assert.ok(
    moe30b.projectedResidentBytes <= report.envelope.wasmBudgetBytes,
    `MoE 30B resident bytes (${moe30b.projectedResidentBytes}) exceeded WASM budget (${report.envelope.wasmBudgetBytes})`,
  );
  assert.ok(
    dense32b.projectedResidentBytes <= report.envelope.wasmBudgetBytes,
    `Dense 32B resident bytes (${dense32b.projectedResidentBytes}) exceeded WASM budget (${report.envelope.wasmBudgetBytes})`,
  );

  // Sparse MoE (3.3B active) must project substantially faster decode than 32B Dense (32.5B active)
  assert.ok(
    moe30b.projectedDecodeTokSec > dense32b.projectedDecodeTokSec * 2,
    `Expected MoE 30B-A3B (${moe30b.projectedDecodeTokSec} tok/s) > 2x Dense 32B (${dense32b.projectedDecodeTokSec} tok/s)`,
  );

  // Qwen3-Coder 30B-A3B Q2_K routes through native Q8_K narrow-lane WASM SIMD (zero scalar fallback)
  const moe30bQ2k = report.models.find((m) => m.model.id === 'qwen3-coder-30b-a3b-q2_k');
  assert.ok(moe30bQ2k, 'Expected Qwen3-Coder 30B-A3B (Q2_K) in coding category');
  assert.match(moe30bQ2k.activationQuantLabel, /Q8_K/i);
  assert.equal(moe30bQ2k.prefillChunkTokens, 32);
  assert.equal(dense32b.prefillChunkTokens, 32);
  assert.ok(
    moe30bQ2k.projectedDecodeTokSec >= moe30b.projectedDecodeTokSec,
    `Expected compact Q2_K 30B-A3B (${moe30bQ2k.projectedDecodeTokSec} tok/s) >= Q4_K_M (${moe30b.projectedDecodeTokSec} tok/s)`,
  );
});

test('Verdant scaling evaluator handles OPFS demand-cache fallback, reclamation, precision, and residency checks', () => {
  // 1. Unbound bootstrap + strict-f64 + tight OPFS with inactive cached model
  const inactiveCached = findCatalogModelById('qwen2.5-coder-14b-instruct-q4_0')!;
  const report = observeCatalogScaling(
    makeBaseInput({
      activeModelId: INTERNAL_BOOTSTRAP_MODEL.id,
      precisionMode: 'strict-f64',
      opfsStorage: {
        supported: true,
        usageBytes: 12 * GIB,
        quotaBytes: 22 * GIB,
        storedModels: [
          {
            modelId: inactiveCached.id,
            fileName: `${inactiveCached.id}.gguf`,
            byteLength: inactiveCached.approximateBytes,
            lastModified: Date.now(),
          },
        ],
        partialModels: [],
      },
    }),
    'coding',
  );

  assert.equal(report.bootstrapUnbound, true);
  assert.ok(
    report.refinements.some((r) => r.priority === 'critical' && r.action.kind === 'bind-model'),
    'Expected critical bind-model suggestion when bootstrap is unbound',
  );
  assert.ok(
    report.refinements.some(
      (r) => r.priority === 'high' && r.action.kind === 'reclaim-unused-opfs',
    ),
    'Expected high reclaim-unused-opfs suggestion when target 30B exceeds remaining OPFS quota',
  );
  assert.ok(
    report.refinements.some(
      (r) =>
        r.priority === 'high' &&
        r.action.kind === 'set-precision' &&
        r.action.mode === 'q8-vnni',
    ),
    'Expected high Q8-VNNI precision suggestion when strict-f64 is enabled',
  );

  // Target 30B model seamlessly uses OPFS demand-paged slab cache under 10 GiB free OPFS
  const moe30b = report.models.find((m) => m.model.id === 'qwen3-coder-30b-a3b-q4_k_m')!;
  assert.equal(moe30b.readiness, 'ready-demand-cache');
  assert.equal(moe30b.storageMode, 'demand-cache');
  assert.ok(moe30b.opfsMissingBytes > 0);

  // Only truly exhausted origins (< 128 MiB free and nothing reclaimable) report quota-constrained
  const exhaustedReport = observeCatalogScaling(
    makeBaseInput({
      opfsStorage: {
        supported: true,
        usageBytes: 22 * GIB - 32 * 1024 * 1024,
        quotaBytes: 22 * GIB,
        storedModels: [],
        partialModels: [],
      },
    }),
    'coding',
  );
  const exhausted30b = exhaustedReport.models.find((m) => m.model.id === 'qwen3-coder-30b-a3b-q4_k_m')!;
  assert.equal(exhausted30b.readiness, 'quota-constrained');

  // 2. Forced paged policy on a small 0.5B compact model triggers continuous restore-auto-residency
  const compactReport = observeCatalogScaling(
    makeBaseInput({
      activeModelId: 'qwen2.5-0.5b-instruct-q4_k_m',
      activeModelName: 'Qwen2.5 0.5B Instruct (Q4_K_M)',
      weightResidencyPolicy: 'paged',
    }),
    'compact',
  );

  assert.equal(compactReport.bootstrapUnbound, false);
  const compactSummary = compactReport.categorySummaries.find((s) => s.category.id === 'compact');
  assert.equal(compactSummary?.activeInCategory, true);
  assert.ok(
    compactReport.refinements.some(
      (r) =>
        r.priority === 'continuous' &&
        r.action.kind === 'set-residency' &&
        r.action.policy === 'auto',
    ),
    'Expected continuous suggestion to restore auto residency when active 0.5B model fits in RAM',
  );
});
