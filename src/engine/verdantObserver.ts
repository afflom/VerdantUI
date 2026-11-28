/**
 * Internal Verdant Scaling & Storage Evaluator (`afflom/verdant-engine` observer_refine.rs parity).
 *
 * Pure, decoupled analysis module that inspects the host execution envelope
 * (WASM32 budget, SIMD tier, worker threads, calibrated bandwidth, OPFS storage quota) and
 * evaluates catalog checkpoints using Prism Theorems T1–T4b (`derivePerformanceContract`).
 * Never exposes internal primitives or fabricated prompt personas in the UI.
 */

import { getGgmlTypeTrait, type LlamaHyperParams } from './gguf';
import {
  MODEL_CATEGORIES,
  formatBytes,
  getModelsByCategory,
  isInternalBootstrapModel,
  type CatalogModel,
  type ModelCategory,
  type ModelCategoryInfo,
} from './modelCatalog';
import {
  derivePerformanceContract,
  INTEGER_DOT_COMPATIBLE_TYPES,
  type EnvironmentProfile,
  type ModelGeometrySpec,
  type PrecisionContractMode,
} from './performanceContract';
import type { OpfsQuotaSnapshot } from './vramBackingStore';
import type { WeightResidencyPolicy } from './vramWeightPager';
import type { SamplingConfig } from './llamaRuntime';

export type ObserverRefinementPriority = 'critical' | 'high' | 'medium' | 'continuous';

export type ModelResidencyStrategy = 'resident-all' | 'opfs-paged-dimm' | 'opfs-sparse-moe';

export type ModelReadinessStatus =
  | 'active'
  | 'ready-cached'
  | 'paused-resumable'
  | 'ready-to-stream'
  | 'ready-demand-cache'
  | 'quota-constrained';

export interface ObserverHostEnvelope {
  readonly simdTierLabel: string;
  readonly workerThreads: number;
  readonly sharedMemoryEnabled: boolean;
  readonly wasmBudgetBytes: number;
  readonly memoryBandwidthGBs: number;
  readonly computeGops: number;
  readonly opfsSupported: boolean;
  readonly opfsUsedBytes: number;
  readonly opfsQuotaBytes: number;
  readonly opfsAvailableBytes: number;
  readonly cachedModelCount: number;
  readonly partialModelCount: number;
}

export interface CatalogModelAssessment {
  readonly model: CatalogModel;
  readonly readiness: ModelReadinessStatus;
  readonly residencyStrategy: ModelResidencyStrategy;
  readonly residencyStrategyLabel: string;
  readonly projectedResidentBytes: number;
  readonly projectedDecodeTokSec: number;
  readonly projectedPrefillTokSec: number;
  readonly activationQuantLabel: string;
  readonly prefillChunkTokens: number;
  readonly pinnedLayerFrames: number;
  readonly totalLayers: number;
  readonly moeExpertCacheSlots: number;
  readonly opfsFits: boolean;
  readonly opfsMissingBytes: number;
  readonly storageMode?: 'full' | 'demand-cache';
  readonly partialReceivedBytes: number;
  readonly partialTotalBytes: number;
  readonly partialPercent: number | null;
}

export type ObserverRefinementAction =
  | { readonly kind: 'bind-model'; readonly modelId: string }
  | { readonly kind: 'set-precision'; readonly mode: PrecisionContractMode }
  | { readonly kind: 'set-residency'; readonly policy: WeightResidencyPolicy }
  | { readonly kind: 'reclaim-unused-opfs'; readonly reclaimModelIds: readonly string[] };

export interface ObserverRefinementProposal {
  readonly id: string;
  readonly priority: ObserverRefinementPriority;
  readonly title: string;
  readonly rationale: string;
  readonly impactLabel: string;
  readonly actionLabel: string;
  readonly action: ObserverRefinementAction;
}

export interface CategorySummaryObservation {
  readonly category: ModelCategoryInfo;
  readonly activeInCategory: boolean;
  readonly cachedCount: number;
  readonly totalCount: number;
}

export interface VerdantObserverReport {
  readonly activeCategory: ModelCategoryInfo;
  readonly envelope: ObserverHostEnvelope;
  readonly categorySummaries: readonly CategorySummaryObservation[];
  readonly models: readonly CatalogModelAssessment[];
  readonly refinements: readonly ObserverRefinementProposal[];
  readonly bootstrapUnbound: boolean;
}

export interface ObserverRuntimeInput {
  readonly activeModelId: string | null;
  readonly activeModelName: string | null;
  readonly env: EnvironmentProfile | null;
  readonly sampling: SamplingConfig;
  readonly precisionMode: PrecisionContractMode;
  readonly weightResidencyPolicy: WeightResidencyPolicy;
  readonly opfsStorage: OpfsQuotaSnapshot;
  readonly backingBandwidthMBs?: number;
}

export function buildGeometrySpecForCatalogModel(model: CatalogModel): ModelGeometrySpec {
  const g = model.geometry;
  const trait = getGgmlTypeTrait(g.dominantGgmlType);
  const bytesPerParam = trait.typeSizeBytes / trait.blockSize;

  const qDim = g.headCount * g.headDim;
  const kvDim = g.headCountKv * g.headDim;
  const attnParamsPerLayer =
    g.embeddingLength * qDim +
    g.embeddingLength * kvDim +
    g.embeddingLength * kvDim +
    qDim * g.embeddingLength;
  const routerParamsPerLayer = g.isMoe ? g.embeddingLength * g.expertCount : 0;
  const moeAttentionLayerBytes = Math.ceil(
    (attnParamsPerLayer + routerParamsPerLayer) * bytesPerParam,
  );

  const ffnWidth = g.isMoe ? g.expertFeedForwardLength : g.feedForwardLength;
  const singleFfnTripleParams = 3 * g.embeddingLength * ffnWidth;
  const moeSingleExpertTripleBytes = g.isMoe
    ? Math.ceil(singleFfnTripleParams * bytesPerParam)
    : 0;

  const lmHeadParams = g.vocabSize * g.embeddingLength;
  const lmHeadWeightBytes = Math.ceil(lmHeadParams * bytesPerParam);

  const activeFfnParamsPerLayer = g.isMoe
    ? singleFfnTripleParams * g.expertUsedCount
    : singleFfnTripleParams;
  const activeBodyParams =
    g.blockCount * (attnParamsPerLayer + routerParamsPerLayer + activeFfnParamsPerLayer);
  const activeParamCount = activeBodyParams + lmHeadParams;

  const activeWeightBytes = Math.min(
    model.approximateBytes,
    Math.ceil(activeParamCount * bytesPerParam),
  );

  const hparams: LlamaHyperParams = {
    architecture: g.architecture,
    name: model.name,
    vocabSize: g.vocabSize,
    contextLength: g.nativeContextTokens,
    embeddingLength: g.embeddingLength,
    feedForwardLength: g.feedForwardLength,
    headCount: g.headCount,
    headCountKv: g.headCountKv,
    headDim: g.headDim,
    blockCount: g.blockCount,
    ropeDimensionCount: g.headDim,
    ropeFreqBase: 1_000_000,
    ropeFreqScale: 1.0,
    ropeType: g.architecture === 'llama' ? 'norm' : 'neox',
    rmsNormEps: 1e-6,
    bosTokenId: 1,
    eosTokenId: 2,
    unkTokenId: 0,
    addBosToken: false,
    tokenizerModel: 'gpt2',
    preTokenizer: 'qwen2',
    chatTemplate: null,
    primaryQuantization: trait.name,
    expertCount: g.expertCount,
    expertUsedCount: g.expertUsedCount,
    expertFeedForwardLength: g.expertFeedForwardLength,
    expertSharedCount: 0,
    expertWeightsNorm: g.isMoe,
  };

  return {
    hparams,
    activeWeightBytes,
    lmHeadWeightBytes,
    activeParamCount,
    totalWeightBufferBytes: model.approximateBytes,
    dominantWeightType: g.dominantGgmlType,
    quantizedWeightFraction: INTEGER_DOT_COMPATIBLE_TYPES.has(g.dominantGgmlType) ? 0.98 : 0.85,
    isMoe: g.isMoe,
    expertCount: g.expertCount,
    expertUsedCount: g.expertUsedCount,
    moeAttentionLayerBytes,
    moeSingleExpertTripleBytes,
  };
}

function formatSimdLabel(env: EnvironmentProfile): string {
  switch (env.simdTier) {
    case 'relaxed-vnni':
      return 'WASM Relaxed-SIMD VNNI';
    case 'simd128-int8':
      return 'WASM SIMD128 Int8';
    case 'simd128-f32':
      return 'WASM SIMD128 F32';
    default:
      return 'Scalar Fallback';
  }
}

export function assessCatalogModel(
  model: CatalogModel,
  env: EnvironmentProfile,
  input: ObserverRuntimeInput,
): CatalogModelAssessment {
  const stored = input.opfsStorage.storedModels.find((m) => m.modelId === model.id);
  const partial = !stored
    ? (input.opfsStorage.partialModels ?? []).find((p) => p.modelId === model.id)
    : undefined;

  const isActive = input.activeModelId === model.id;
  const quotaBytes = input.opfsStorage.quotaBytes;
  const usageBytes = input.opfsStorage.usageBytes;
  const availableBytes = quotaBytes > 0 ? Math.max(0, quotaBytes - usageBytes) : 0;
  const remainingDownloadBytes = stored
    ? 0
    : Math.max(0, model.approximateBytes - (partial?.receivedBytes ?? 0));

  const safetyHeadroom = 256 * 1024 * 1024;
  const opfsFits =
    Boolean(stored) ||
    quotaBytes <= 0 ||
    remainingDownloadBytes + safetyHeadroom <= availableBytes;
  const opfsMissingBytes = opfsFits
    ? 0
    : Math.max(0, remainingDownloadBytes + safetyHeadroom - availableBytes);
  const reclaimableInactiveBytes = input.opfsStorage.storedModels
    .filter((m) => m.modelId !== input.activeModelId && m.modelId !== model.id)
    .reduce((sum, m) => sum + m.byteLength, 0);
  const canUseDemandCache =
    input.opfsStorage.supported &&
    (quotaBytes <= 0 || availableBytes + reclaimableInactiveBytes >= 128 * 1024 * 1024);

  const spec = buildGeometrySpecForCatalogModel(model);
  const contract = derivePerformanceContract(env, spec, {
    precisionMode: input.precisionMode,
    backingKind: !input.opfsStorage.supported
      ? 'memory'
      : stored?.storageMode === 'demand-cache' || (!opfsFits && canUseDemandCache)
        ? 'opfs-demand'
        : 'opfs-sync',
    weightResidencyPolicy: input.weightResidencyPolicy,
    backingBandwidthMBs: input.backingBandwidthMBs ?? 3200,
    backingQuotaBytes:
      quotaBytes > 0 ? quotaBytes : Math.max(model.approximateBytes * 2, 32 * 1024 * 1024 * 1024),
  });

  let residencyStrategy: ModelResidencyStrategy;
  let residencyStrategyLabel: string;
  if (contract.weightResidencyMode === 'resident-all') {
    residencyStrategy = 'resident-all';
    residencyStrategyLabel = 'In Memory';
  } else if (model.geometry.isMoe) {
    residencyStrategy = 'opfs-sparse-moe';
    residencyStrategyLabel = `Sparse MoE (${contract.moeExpertCacheSlots}E cache · ${contract.weightLayerFrameSlots}/${model.geometry.blockCount}L)`;
  } else {
    residencyStrategy = 'opfs-paged-dimm';
    residencyStrategyLabel = `Paged (${contract.weightLayerFrameSlots}/${model.geometry.blockCount}L resident)`;
  }

  let readiness: ModelReadinessStatus;
  if (isActive) {
    readiness = 'active';
  } else if (stored) {
    readiness = 'ready-cached';
  } else if (partial) {
    readiness = 'paused-resumable';
  } else if (opfsFits) {
    readiness = 'ready-to-stream';
  } else if (canUseDemandCache) {
    readiness = 'ready-demand-cache';
  } else {
    readiness = 'quota-constrained';
  }

  const partialPercent =
    partial && partial.totalBytes > 0
      ? Math.min(99, Math.max(1, Math.round((partial.receivedBytes / partial.totalBytes) * 100)))
      : null;

  return {
    model,
    readiness,
    residencyStrategy,
    residencyStrategyLabel,
    projectedResidentBytes: contract.projectedResidentWeightBytes,
    projectedDecodeTokSec: Number(contract.rooflineDecodeTokSec.toFixed(1)),
    projectedPrefillTokSec: Number(contract.rooflinePrefillTokSec.toFixed(1)),
    activationQuantLabel: contract.activationQuantLabel,
    prefillChunkTokens: contract.prefillChunkTokens,
    pinnedLayerFrames: contract.weightLayerFrameSlots,
    totalLayers: model.geometry.blockCount,
    moeExpertCacheSlots: contract.moeExpertCacheSlots,
    opfsFits,
    opfsMissingBytes,
    storageMode: stored?.storageMode ?? (!opfsFits && canUseDemandCache ? 'demand-cache' : 'full'),
    partialReceivedBytes: partial?.receivedBytes ?? 0,
    partialTotalBytes: partial?.totalBytes ?? model.approximateBytes,
    partialPercent,
  };
}

export function observeCatalogScaling(
  input: ObserverRuntimeInput,
  category: ModelCategory = 'all',
): VerdantObserverReport {
  if (!input.env) {
    throw new Error(
      'refused: observeCatalogScaling requires a measured EnvironmentProfile; synthetic hardware assumptions are not permitted',
    );
  }
  const env = input.env;
  const activeCategory = MODEL_CATEGORIES.find((c) => c.id === category) ?? MODEL_CATEGORIES[0];
  const bootstrapUnbound = isInternalBootstrapModel(input.activeModelId);

  const opfsStored = input.opfsStorage.storedModels;
  const opfsPartials = input.opfsStorage.partialModels ?? [];
  const opfsUsedBytes =
    input.opfsStorage.usageBytes > 0
      ? input.opfsStorage.usageBytes
      : opfsStored.reduce((sum, m) => sum + m.byteLength, 0) +
        opfsPartials.reduce((sum, p) => sum + p.receivedBytes, 0);
  const opfsQuotaBytes = input.opfsStorage.quotaBytes;
  const opfsAvailableBytes = opfsQuotaBytes > 0 ? Math.max(0, opfsQuotaBytes - opfsUsedBytes) : 0;

  const envelope: ObserverHostEnvelope = {
    simdTierLabel: formatSimdLabel(env),
    workerThreads: env.effectiveWorkers,
    sharedMemoryEnabled: env.sharedMemorySupported,
    wasmBudgetBytes: env.memoryBudgetBytes,
    memoryBandwidthGBs: Number(env.memoryBandwidthGBs.toFixed(1)),
    computeGops: Number(env.computeGops.toFixed(0)),
    opfsSupported: input.opfsStorage.supported,
    opfsUsedBytes,
    opfsQuotaBytes,
    opfsAvailableBytes,
    cachedModelCount: opfsStored.length,
    partialModelCount: opfsPartials.length,
  };

  // Assess every catalog model ONCE and reuse across active view + category summaries
  const allCatalogModels = getModelsByCategory('all');
  const assessedById = new Map<string, CatalogModelAssessment>();
  for (let i = 0; i < allCatalogModels.length; i++) {
    const m = allCatalogModels[i];
    assessedById.set(m.id, assessCatalogModel(m, env, input));
  }

  const categoryModels = getModelsByCategory(activeCategory.id);
  const assessments = categoryModels.map(
    (m) => assessedById.get(m.id) ?? assessCatalogModel(m, env, input),
  );

  const categorySummaries: CategorySummaryObservation[] = MODEL_CATEGORIES.map((cat) => {
    const modelsInCat = getModelsByCategory(cat.id);
    let cachedCount = 0;
    for (let i = 0; i < modelsInCat.length; i++) {
      const a = assessedById.get(modelsInCat[i].id);
      if (a && (a.readiness === 'ready-cached' || a.readiness === 'active')) {
        cachedCount += 1;
      }
    }
    return {
      category: cat,
      activeInCategory: !bootstrapUnbound && modelsInCat.some((m) => m.id === input.activeModelId),
      cachedCount,
      totalCount: modelsInCat.length,
    };
  });

  const refinements: ObserverRefinementProposal[] = [];

  if (bootstrapUnbound && assessments[0]) {
    const best =
      assessments.find((a) => a.readiness === 'ready-cached') ??
      assessments.find((a) => a.opfsFits) ??
      assessments[0];
    refinements.push({
      id: 'bind-initial-model',
      priority: 'critical',
      title: `Load ${best.model.shortName}`,
      rationale: 'No user model is currently loaded.',
      impactLabel: `${best.model.sizeLabel} · ~${best.projectedDecodeTokSec} tok/s*`,
      actionLabel: best.readiness === 'ready-cached' ? `Load ${best.model.shortName}` : `Download ${best.model.shortName}`,
      action: { kind: 'bind-model', modelId: best.model.id },
    });
  }

  const firstUnfitting = assessments.find((a) => !a.opfsFits);
  if (firstUnfitting && opfsStored.length > 0) {
    const reclaimable = opfsStored.filter((m) => m.modelId !== input.activeModelId);
    const reclaimBytes = reclaimable.reduce((acc, m) => acc + m.byteLength, 0);
    if (reclaimable.length > 0 && reclaimBytes > 0) {
      refinements.push({
        id: 'reclaim-opfs-storage',
        priority: 'high',
        title: `Reclaim ${formatBytes(reclaimBytes)} unused storage`,
        rationale: `Clearing ${reclaimable.length} inactive cached model${reclaimable.length > 1 ? 's' : ''} frees room for full local caching.`,
        impactLabel: `+${formatBytes(reclaimBytes)} free`,
        actionLabel: 'Reclaim Inactive Cache',
        action: {
          kind: 'reclaim-unused-opfs',
          reclaimModelIds: reclaimable.map((m) => m.modelId),
        },
      });
    }
  }

  if (input.precisionMode !== 'q8-vnni') {
    refinements.push({
      id: 'enable-vnni-narrow-lane',
      priority: 'high',
      title: 'Engage Q8-VNNI narrow-lane precision',
      rationale:
        'Strict F64 promotion unpacks every quantized block to F32 inside the inner loop. Switching to Q8-VNNI restores ~1.9× roofline bandwidth.',
      impactLabel: '~1.9× decode throughput',
      actionLabel: 'Switch to Q8-VNNI',
      action: { kind: 'set-precision', mode: 'q8-vnni' },
    });
  }

  const activeAssessment = assessments.find((a) => a.readiness === 'active');
  if (
    input.weightResidencyPolicy === 'paged' &&
    activeAssessment &&
    activeAssessment.model.approximateBytes < env.memoryBudgetBytes * 0.8
  ) {
    refinements.push({
      id: 'restore-auto-residency',
      priority: 'continuous',
      title: 'Restore automatic memory residency',
      rationale: `${activeAssessment.model.shortName} (${activeAssessment.model.sizeLabel}) fits completely inside memory.`,
      impactLabel: '0 page faults',
      actionLabel: 'Set Residency Auto',
      action: { kind: 'set-residency', policy: 'auto' },
    });
  }

  return {
    activeCategory,
    envelope,
    categorySummaries,
    models: assessments,
    refinements,
    bootstrapUnbound,
  };
}
