/**
 * Client-side llama.cpp Inference Host bound exclusively to the afflom/vTensor GGML Provider
 * governed by the Prism Formal Performance Contract (Theorems T1–T5):
 * - Calibrated Roofline Bounds (T1): measures host weight bandwidth & VNNI compute throughput on load.
 * - Optimal Spill-Free Batch Width & Shared Q8_0 Activations (T2): quantizes `xNorm` / `attnOut` /
 *   `ffnHidden` ONCE per projection group (`Q/K/V`, `gate/up`) and dispatches hardware Relaxed-SIMD
 *   VNNI (`i32x4.relaxed_dot_i8x16_i7x16_add_s`) or SIMD128 Int8 dot products across all 8 quantized
 *   GGML formats, with universal 4-token batched prompt prefill (Q4_0, Q4_K, Q6_K, Q8_0, F32).
 * - Toggleable Precision Policy: `'q8-vnni'` (default hardware-optimal) or `'strict-f64'` (exact F32×W
 *   F64-accumulated verification mode) switchable live without reloading model weights.
 * - vRAM 3-Tier Bounded-Memory Paged KV-Cache (T4): contract-derived Hot F32 (`H*`), Warm Q8 (`W*`),
 *   and Cold Q4 (`~7.1x`) tiers with automatic reclaim, cascade, and safe context scaling (`C_eff*`).
 * - Live Contract Compliance Verification (T5): computes real-time roofline efficiency `eta` & verdict.
 */

import {
  GgmlType,
  getGgmlTypeTrait,
  type GgufTensorInfo,
  type LlamaHyperParams,
  type ParsedGgufModel,
  parseGgufFromBacking,
  parseGgufFromBackingAsync,
} from './gguf';
import {
  derivePerformanceContract,
  evaluateContractCompliance,
  INTEGER_DOT_COMPATIBLE_TYPES,
  SPILL_FREE_BATCH_WIDTH,
  type ModelGeometrySpec,
  type PerformanceContract,
  type PrecisionContractMode,
} from './performanceContract';
import { GgufTokenizer } from './tokenizer';
import {
  buildRopeTables,
  fillRopePositionCosSin,
  type RopeTables,
  verdantAdd,
  verdantDecodeRow,
  verdantMulMat,
  verdantRmsNorm,
  verdantRmsNormPerHead,
  verdantRopePrecomputed,
  verdantSwiGlu,
} from './verdantKernels';
import {
  computeMoeWorkerScratchLayout,
  ParallelMatOp,
  Q8_0_ACT_BLOCK_BYTES,
  Q8_K_ACT_BLOCK_BYTES,
  WASM_TOPK_IDS_OFFSET,
  WASM_TOPK_VALS_OFFSET,
  type MatGroupEntry,
  type MoeWorkerScratchLayout,
  type VerdantWasmArena,
} from './verdantWasmSimd';
import {
  type CompiledVTensorGraph,
  type GgmlGraphNode,
  GgmlOp,
  VTENSOR_CONTRACT,
  VTensorBackendRegistry,
  type VTensorHandle,
  type VTensorTelemetrySnapshot,
} from './vtensorProvider';
import { VramKvPager } from './vramKvPager';
import {
  calibrateBackingReadBandwidthMBs,
  MemoryBackingStore,
  OpfsDemandVramBackingStore,
  readHotMetadataBundleFromOpfs,
  writeHotMetadataBundleToOpfs,
  type ByteRange,
  type VramBackingKind,
  type VramBackingStore,
} from './vramBackingStore';
import {
  planPriorityWeightExtents,
  type ExpertPrewarmPlan,
} from './unifiedExpertScheduler';
import {
  computeBackingContentFingerprint,
  computeBackingContentFingerprintAsync,
  readHotEmbeddingsCacheFromOpfs,
  readParsedGgufBootCacheFromOpfs,
  readRopeTablesCacheFromOpfs,
  writeHotEmbeddingsCacheToOpfs,
  writeParsedGgufBootCacheToOpfs,
  writeRopeTablesCacheToOpfs,
  type RopeCacheParams,
} from './vramBootCache';
import {
  collectHotMetadataTensorInfos,
  computeLmHeadStripeRows,
  VramWeightPager,
  type LayerTensorInfoGroup,
  type PagedLayerWeights,
  type WeightResidencyPolicy,
} from './vramWeightPager';
import {
  deriveUnifiedStoreAllocation,
  VRAM_KV_PAGE_TOKENS,
  VramUnifiedStore,
  type UnifiedStoreTelemetry,
} from './vramUnifiedStore';
import {
  createCooperativeYielder,
  FrameBudgetProfiler,
  GENERATION_SLICE_BUDGET_US,
  SPECULATIVE_SLICE_BUDGET_US,
  type CooperativeYielder,
} from './vramSwapPolicy';
import {
  compilePrismMatrixSchedule,
  PrismCensusLedger,
  type PrismCompiledSchedule,
  type PrismHostCapabilities,
  type PrismOpDescriptor,
} from './prismKernel';

export const SUPPORTED_ARCHITECTURES: ReadonlySet<string> = new Set<string>([
  'llama',
  'qwen2',
  'qwen2moe',
  'qwen3',
  'qwen3moe',
  'mistral',
  'mixtral',
  'granite',
  'smollm3',
  'internlm2',
  'olmo2',
]);

export const SUPPORTED_VERDANT_STORAGE_TYPES: ReadonlySet<GgmlType> = new Set<GgmlType>([
  GgmlType.F32,
  GgmlType.F16,
  GgmlType.BF16,
  GgmlType.Q4_0,
  GgmlType.Q4_1,
  GgmlType.Q5_0,
  GgmlType.Q5_1,
  GgmlType.Q8_0,
  GgmlType.Q2_K,
  GgmlType.Q3_K,
  GgmlType.Q4_K,
  GgmlType.Q5_K,
  GgmlType.Q6_K,
]);

export interface SamplingConfig {
  readonly temperature: number;
  readonly topP: number;
  readonly topK: number;
  readonly repeatPenalty: number;
  readonly maxTokens: number;
  /** When true (default), decode continues up to the selected model's full context length. */
  readonly fullContextOutput?: boolean;
  readonly seed?: number;
}

export const DEFAULT_SAMPLING_CONFIG: SamplingConfig = {
  temperature: 0.6,
  topP: 0.9,
  topK: 40,
  repeatPenalty: 1.1,
  maxTokens: 2048,
  fullContextOutput: true,
};

export interface LoadedModelSummary {
  readonly id: string;
  readonly displayName: string;
  readonly source: string;
  readonly fileByteLength: number;
  readonly tensorCount: number;
  readonly hparams: LlamaHyperParams;
  readonly maxContextWindow: number;
  readonly backingKind?: VramBackingKind;
}

export interface PrefillProgressEvent {
  readonly completedTokens: number;
  readonly totalTokens: number;
  readonly reusedPrefixTokens: number;
  readonly detail?: string;
}

export interface AsyncLoadProgressEvent {
  readonly percent: number;
  readonly detail: string;
}

interface AsyncLayerPlan {
  readonly layerIdx: number;
  readonly acquireNode: GgmlGraphNode | null;
  readonly preMoeNodes: readonly GgmlGraphNode[];
  readonly moe?: {
    readonly routeAndGetExperts: () => { readonly count: number; readonly expertIds: number[] };
    readonly executeRouted: (count: number) => void;
    readonly residualNode: GgmlGraphNode;
  };
  readonly denseFfnNodes: readonly GgmlGraphNode[];
}

interface SingleAsyncGraphPlan {
  readonly embedNode: GgmlGraphNode;
  readonly layers: readonly AsyncLayerPlan[];
  readonly lmNormAndQuantNodes: readonly GgmlGraphNode[];
  readonly pinnedLmHeadNode: GgmlGraphNode | null;
  readonly executeLmHeadStripe:
    | ((weightStripe: VTensorHandle, rowStart: number, rowCount: number) => void)
    | null;
}

interface Batch4AsyncGraphPlan {
  readonly embedNode: GgmlGraphNode;
  readonly layers: readonly AsyncLayerPlan[];
}

export interface SpeculativePrefillProgress {
  readonly readyTokens: number;
  readonly targetTokens: number;
}

export interface SpeculativePrefillResult {
  readonly targetTokens: number;
  readonly readyTokens: number;
  readonly reusedLiveTokens: number;
  readonly restoredUnifiedTokens: number;
  readonly newlyPrefilledTokens: number;
  readonly aborted: boolean;
}

export interface TokenStreamEvent {
  readonly tokenId: number;
  readonly textDelta: string;
  readonly accumulatedText: string;
  readonly generatedTokens: number;
  readonly promptTokens: number;
  readonly contextWindow?: number;
  readonly promptTruncatedTokens?: number;
  readonly tokensPerSecond: number;
  readonly finished: boolean;
}

interface ScratchTensors {
  readonly x: VTensorHandle;
  readonly xNorm: VTensorHandle;
  readonly q: VTensorHandle;
  readonly k: VTensorHandle;
  readonly v: VTensorHandle;
  readonly attnOut: VTensorHandle;
  readonly attnProj: VTensorHandle;
  readonly ffnGate: VTensorHandle;
  readonly ffnUp: VTensorHandle;
  readonly ffnHidden: VTensorHandle;
  readonly ffnProj: VTensorHandle;
  readonly logits: VTensorHandle;
  // Sparse MoE reusable scratch vectors (null for dense models)
  readonly moeRouterLogits: VTensorHandle | null;
  readonly moeExpertProj: VTensorHandle | null;
  readonly moeWorkerScratch: VTensorHandle | null;
  readonly moeWorkerScratchLayout: MoeWorkerScratchLayout | null;
  readonly moeBatch4WorkerScratchLayout: MoeWorkerScratchLayout | null;
  readonly moeWorkerScratchSlots: number;
  // Shared packed Q8_0_ACT & Q8_K_ACT scratch vectors (amortized across Q/K/V and gate/up!)
  readonly q8Dim: VTensorHandle | null;
  readonly q8AttnOut: VTensorHandle | null;
  readonly q8Ffn: VTensorHandle | null;
  readonly q8kDim: VTensorHandle | null;
  readonly q8kAttnOut: VTensorHandle | null;
  readonly q8kFfn: VTensorHandle | null;
}

interface Batch4ScratchTensors {
  readonly x: VTensorHandle;
  readonly xNorm: VTensorHandle;
  readonly q: VTensorHandle;
  readonly k: VTensorHandle;
  readonly v: VTensorHandle;
  readonly attnOut: VTensorHandle;
  readonly attnProj: VTensorHandle;
  readonly ffnGate: VTensorHandle;
  readonly ffnUp: VTensorHandle;
  readonly ffnHidden: VTensorHandle;
  readonly ffnProj: VTensorHandle;
  // Shared batch-4 packed Q8_0_ACT & Q8_K_ACT scratch vectors
  readonly q8Dim: VTensorHandle | null;
  readonly q8AttnOut: VTensorHandle | null;
  readonly q8Ffn: VTensorHandle | null;
  readonly q8kDim: VTensorHandle | null;
  readonly q8kAttnOut: VTensorHandle | null;
  readonly q8kFfn: VTensorHandle | null;
}

const PREFILL_BATCH_SIZE = SPILL_FREE_BATCH_WIDTH;
/** Up to 8 Batch-4 micro-batches (32 tokens) per OPFS layer acquire during Layer-Major Chunked Prefill. */
const PREFILL_LAYER_MAJOR_MAX_MICROBATCHES = 8;

function alignUp32(n: number): number {
  const rem = n % 32;
  return rem === 0 ? n : n + (32 - rem);
}

/** Deterministic PRNG (Mulberry32) when seed is provided, else Math.random(). */
function createRng(seed?: number): () => number {
  if (seed === undefined) return Math.random;
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Reusable scratch buffers for zero-allocation repeat-penalty restoration (max 64 unique tokens)
const savedPenaltyIds = new Int32Array(64);
const savedPenaltyVals = new Float32Array(64);
// Reusable scratch buffers for zero-allocation Top-K sampling (k <= 1024)
const samplerTopkIds = new Int32Array(1024);
const samplerTopkVals = new Float32Array(1024);
const samplerTopkProbs = new Float64Array(1024);
let cachedTopkMemBuf: ArrayBufferLike | null = null;
let cachedTopkIdsView: Int32Array | null = null;
let cachedTopkValsView: Float32Array | null = null;

/**
 * Zero-allocation token sampler: applies repeat penalty in-place to the <=64 unique recent token
 * entries and restores them before returning (avoids copying 49K–152K floats per token!).
 */
export function sampleNextToken(
  logits: Float32Array,
  recentTokens: readonly number[],
  config: SamplingConfig,
  rng: () => number,
  arena?: VerdantWasmArena | null,
  logitsPtr = -1,
): number {
  const vocabSize = logits.length;
  let modifiedCount = 0;

  // Apply repeat penalty in-place for <=64 unique recent tokens and save originals for restoration
  if (config.repeatPenalty > 1.0 && recentTokens.length > 0) {
    const windowStart = Math.max(0, recentTokens.length - 64);
    for (let i = windowStart; i < recentTokens.length; i++) {
      const id = recentTokens[i];
      if (id < 0 || id >= vocabSize) continue;
      let dup = false;
      for (let j = 0; j < modifiedCount; j++) {
        if (savedPenaltyIds[j] === id) {
          dup = true;
          break;
        }
      }
      if (dup) continue;
      const orig = logits[id];
      savedPenaltyIds[modifiedCount] = id;
      savedPenaltyVals[modifiedCount] = orig;
      modifiedCount += 1;
      logits[id] = orig > 0 ? orig / config.repeatPenalty : orig * config.repeatPenalty;
    }
  }

  try {
    // Greedy argmax when temperature <= 0.01 (100% in-Verdant WASM SIMD when arena-backed,
    // guarded against non-finite logits so numerical spikes never poison greedy output)
    if (config.temperature <= 0.01) {
      if (arena && logitsPtr >= 0) {
        const wasmBest = arena.kernels.argmax_f32(logitsPtr, vocabSize);
        if (wasmBest >= 0 && wasmBest < vocabSize && Number.isFinite(logits[wasmBest])) {
          return wasmBest;
        }
      }
      let bestId = 0;
      let bestVal = -Infinity;
      for (let i = 0; i < vocabSize; i++) {
        const v = logits[i];
        if (Number.isFinite(v) && v > bestVal) {
          bestVal = v;
          bestId = i;
        }
      }
      return bestId;
    }

    // Partial Top-K selection (100% in-Verdant WASM SIMD for k <= 1024, zero JS allocations!)
    const k = Math.max(1, Math.min(config.topK, vocabSize));
    let candIds: Int32Array;
    let candCount: number;

    if (k <= 1024) {
      candIds = samplerTopkIds;
      if (arena && logitsPtr >= 0) {
        const rawCount = arena.kernels.topk_f32(
          logitsPtr,
          vocabSize,
          k,
          WASM_TOPK_IDS_OFFSET,
          WASM_TOPK_VALS_OFFSET,
        );
        const memBuf = arena.memory.buffer;
        if (cachedTopkMemBuf !== memBuf || !cachedTopkIdsView || !cachedTopkValsView) {
          cachedTopkMemBuf = memBuf;
          cachedTopkIdsView = new Int32Array(memBuf, WASM_TOPK_IDS_OFFSET, 1024);
          cachedTopkValsView = new Float32Array(memBuf, WASM_TOPK_VALS_OFFSET, 1024);
        }
        candCount = 0;
        for (let i = 0; i < rawCount; i++) {
          const val = cachedTopkValsView[i];
          if (Number.isFinite(val)) {
            samplerTopkIds[candCount] = cachedTopkIdsView[i];
            samplerTopkVals[candCount] = val;
            candCount += 1;
          }
        }
      } else {
        candCount = 0;
        for (let i = 0; i < k; i++) {
          samplerTopkIds[i] = 0;
          samplerTopkVals[i] = -Infinity;
        }
        let minSlot = 0;
        let minVal = -Infinity;
        for (let i = 0; i < vocabSize; i++) {
          const v = logits[i];
          if (!Number.isFinite(v)) continue;
          if (v > minVal) {
            if (candCount < k) candCount += 1;
            samplerTopkIds[minSlot] = i;
            samplerTopkVals[minSlot] = v;
            minVal = samplerTopkVals[0];
            minSlot = 0;
            for (let j = 1; j < k; j++) {
              if (samplerTopkVals[j] < minVal) {
                minVal = samplerTopkVals[j];
                minSlot = j;
              }
            }
          }
        }
        // Compact out any unfilled -Infinity slots when fewer than k finite logits exist
        if (candCount < k) {
          let write = 0;
          for (let i = 0; i < k; i++) {
            if (Number.isFinite(samplerTopkVals[i])) {
              samplerTopkIds[write] = samplerTopkIds[i];
              samplerTopkVals[write] = samplerTopkVals[i];
              write += 1;
            }
          }
          candCount = write;
        }
      }

      // In-place descending sort of candCount <= 1024 items (VT-217: ties break by lower token ID)
      for (let i = 1; i < candCount; i++) {
        const keyId = samplerTopkIds[i];
        const keyVal = samplerTopkVals[i];
        let j = i - 1;
        while (
          j >= 0 &&
          (samplerTopkVals[j] < keyVal ||
            (samplerTopkVals[j] === keyVal && samplerTopkIds[j] > keyId))
        ) {
          samplerTopkIds[j + 1] = samplerTopkIds[j];
          samplerTopkVals[j + 1] = samplerTopkVals[j];
          j--;
        }
        samplerTopkIds[j + 1] = keyId;
        samplerTopkVals[j + 1] = keyVal;
      }
    } else {
      const topIds = new Int32Array(k);
      const topVals = new Float32Array(k);
      topVals.fill(-Infinity);
      let minSlot = 0;
      let minVal = -Infinity;
      for (let i = 0; i < vocabSize; i++) {
        const v = logits[i];
        if (!Number.isFinite(v)) continue;
        if (v > minVal) {
          topIds[minSlot] = i;
          topVals[minSlot] = v;
          minVal = topVals[0];
          minSlot = 0;
          for (let j = 1; j < k; j++) {
            if (topVals[j] < minVal || (topVals[j] === minVal && topIds[j] > topIds[minSlot])) {
              minVal = topVals[j];
              minSlot = j;
            }
          }
        }
      }
      const finiteIds: number[] = [];
      for (let i = 0; i < k; i++) {
        if (Number.isFinite(topVals[i])) finiteIds.push(topIds[i]);
      }
      finiteIds.sort((a, b) => logits[b] - logits[a] || a - b);
      candIds = Int32Array.from(finiteIds);
      candCount = finiteIds.length;
    }

    // Finite-logit fallback: if every logit was non-finite (e.g. numerical spike), pick deterministic argmax/0
    if (candCount === 0) {
      let bestId = 0;
      let bestVal = -Infinity;
      for (let i = 0; i < vocabSize; i++) {
        const v = logits[i];
        if (Number.isFinite(v) && v > bestVal) {
          bestVal = v;
          bestId = i;
        }
      }
      return bestId;
    }

    const invTemp = 1.0 / config.temperature;
    const maxLogit = logits[candIds[0]] * invTemp;
    if (!Number.isFinite(maxLogit)) {
      return candIds[0];
    }

    const probs = candCount <= 1024 ? samplerTopkProbs : new Float64Array(candCount);
    let sumExp = 0;
    for (let i = 0; i < candCount; i++) {
      const p = Math.exp(logits[candIds[i]] * invTemp - maxLogit);
      const safeP = Number.isFinite(p) ? p : 0;
      probs[i] = safeP;
      sumExp += safeP;
    }
    if (!(sumExp > 0) || !Number.isFinite(sumExp)) {
      return candIds[0];
    }
    const invSum = 1.0 / sumExp;
    for (let i = 0; i < candCount; i++) {
      probs[i] *= invSum;
    }

    // Top-P (nucleus) truncation
    const topP = Math.max(0.05, Math.min(1.0, config.topP));
    let cumProb = 0;
    let cutoff = candCount;
    for (let i = 0; i < candCount; i++) {
      cumProb += probs[i];
      if (cumProb >= topP) {
        cutoff = i + 1;
        break;
      }
    }

    let renorm = 0;
    for (let i = 0; i < cutoff; i++) renorm += probs[i];
    if (!(renorm > 0) || !Number.isFinite(renorm)) {
      return candIds[0];
    }

    const r = rng() * renorm;
    let running = 0;
    for (let i = 0; i < cutoff; i++) {
      running += probs[i];
      if (r <= running) {
        return candIds[i];
      }
    }
    return candIds[cutoff - 1];
  } finally {
    // Restore original logits so callers inspecting logits see unmutated values
    for (let i = 0; i < modifiedCount; i++) {
      logits[savedPenaltyIds[i]] = savedPenaltyVals[i];
    }
  }
}

export class LlamaVTensorRuntime {
  readonly backend: VTensorBackendRegistry;

  private modelSummary: LoadedModelSummary | null = null;
  private tokenizer: GgufTokenizer | null = null;
  private hparams: LlamaHyperParams | null = null;
  private ropeTables: RopeTables | null = null;
  private ropeCosHandle: VTensorHandle | null = null;
  private ropeSinHandle: VTensorHandle | null = null;

  private kvPager: VramKvPager | null = null;
  private cachedTokens: number[] = [];
  private activeThreadId: string | null = null;
  private unifiedStore: VramUnifiedStore | null = null;
  private readonly committedPageKeys = new Set<string>();

  // OPFS-backed vRAM Virtual DIMM Weight Pager & parsed header state
  private backingStore: VramBackingStore | null = null;
  private modelContentFingerprint: string | null = null;
  private parsedHeader: ParsedGgufModel | null = null;
  private weightPager: VramWeightPager | null = null;
  private weightResidencyPolicy: WeightResidencyPolicy = 'auto';
  private backingQuotaBytes: number | undefined = undefined;
  private calibratedBackingBandwidthMBs: number | undefined = undefined;
  private embdInfo: GgufTensorInfo | null = null;
  private outNormInfo: GgufTensorInfo | null = null;
  private outWeightInfo: GgufTensorInfo | null = null;
  private isTiedOutput = false;
  private layerInfos: LayerTensorInfoGroup[] = [];
  private preloadedRopeTables: RopeTables | null = null;
  private preloadedRopeParams: RopeCacheParams | null = null;

  private scratch: ScratchTensors | null = null;
  private batchScratch: Batch4ScratchTensors | null = null;
  private maxContextWindow = 1024;
  private requestedMaxContext = 1024;

  // Prism Performance Contract & Compiled Kernel Schedule state
  private precisionMode: PrecisionContractMode = 'q8-vnni';
  private modelSpec: ModelGeometrySpec | null = null;
  private performanceContract: PerformanceContract | null = null;
  private readonly prismLedger = new PrismCensusLedger();
  private prismSchedule: PrismCompiledSchedule | null = null;
  private prismSingleBodyTally = { q8kOps: 0, q8kSb: 0, q80Ops: 0, strictOps: 0, fusedBarriers: 0 };
  private prismSingleLmTally = { q8kOps: 0, q8kSb: 0, q80Ops: 0, strictOps: 0 };
  private prismBatch4Tally = { q8kOps: 0, q8kSb: 0, q80Ops: 0, strictOps: 0, fusedBarriers: 0 };

  // Compiled reusable vTensor graph plans & mutable scalar runtime state
  private compiledSingleGraph: CompiledVTensorGraph | null = null;
  private singleGraphBodyNodeCount = 0;
  private compiledBatch4Graph: CompiledVTensorGraph | null = null;
  private batchEmbedNode: GgmlGraphNode | null = null;
  private batchLayerAcquireNodes: Array<GgmlGraphNode | null> = [];
  private batchLayerBodyNodes: GgmlGraphNode[][] = [];
  private prefillChunkX: Float32Array | null = null;
  private singleAsyncPlan: SingleAsyncGraphPlan | null = null;
  private batchAsyncPlan: Batch4AsyncGraphPlan | null = null;
  private preloadedHotTensors: ReadonlyMap<string, Uint8Array> | null = null;
  private readonly singleExec: {
    tokenId: number;
    pos: number;
    seqLen: number;
    externalEmbedding: Float32Array | null;
  } = { tokenId: 0, pos: 0, seqLen: 1, externalEmbedding: null };
  private readonly batchExec = { tokens: new Int32Array(PREFILL_BATCH_SIZE), startPos: 0 };

  // Verdant 16,666 µs FrameBudgetProfiler + zero-clamp MessageChannel cooperative yielder (Theorem T7)
  readonly frameBudget: FrameBudgetProfiler;
  private readonly yielder: CooperativeYielder;

  constructor(backend?: VTensorBackendRegistry) {
    const registry = backend ?? new VTensorBackendRegistry();
    // Enforce exclusive binding to vTensor (throws if any other provider were requested)
    this.backend = registry.selectBackend(VTENSOR_CONTRACT.providerName);
    this.yielder = createCooperativeYielder();
    this.frameBudget = new FrameBudgetProfiler(undefined, this.yielder.transport);
    this.backend.attachFrameBudgetTelemetry(() => this.frameBudget.getTelemetry());
  }

  /**
   * Initialize the multi-core SharedArrayBuffer WASM SIMD thread pool (Theorem T3)
   * inside vTensor so large projection matrices scale across P CPU cores.
   */
  async initThreadPool(options?: {
    maxWorkers?: number;
    parallelThresholdBytes?: number;
  }): Promise<void> {
    await this.backend.initThreadPool(options);
  }

  /** Terminate helper threads owned by the vTensor WASM SIMD thread pool. */
  disposeThreadPool(): void {
    this.backend.disposeThreadPool();
  }

  /** Marks the start of a continuous generation or prefill compute slice. */
  beginGenerationSlice(): void {
    this.frameBudget.beginSlice();
  }

  /**
   * Zero-clamp cooperative yield gated by the 16,666 µs Verdant frame budget KPI.
   * Replaces per-token `setTimeout(0)` (4 ms Chromium clamp) with a ~20 µs `MessageChannel`
   * turn emitted only when slice time reaches 12 ms (or every 16 tokens), keeping cancel
   * responsiveness sub-frame while freeing 100% of CPU cycles for WASM SIMD throughput.
   */
  async maybeYieldGenerationStep(stepDurationUs: number): Promise<void> {
    if (
      this.frameBudget.recordStepAndShouldYield(stepDurationUs, GENERATION_SLICE_BUDGET_US, 16)
    ) {
      await this.yielder.yieldNow();
    }
  }

  get isLoaded(): boolean {
    return this.modelSummary !== null;
  }

  getModelSummary(): LoadedModelSummary | null {
    return this.modelSummary;
  }

  getPrecisionMode(): PrecisionContractMode {
    return this.precisionMode;
  }

  getWeightResidencyPolicy(): WeightResidencyPolicy {
    return this.weightResidencyPolicy;
  }

  getPerformanceContract(): PerformanceContract | null {
    return this.performanceContract;
  }

  /**
   * Switches between hardware-optimal `'q8-vnni'` (Relaxed-SIMD VNNI / Int8) and exact
   * `'strict-f64'` (F32×W F64-accumulated) live without reloading weights.
   */
  setPrecisionMode(mode: PrecisionContractMode): PerformanceContract | null {
    if (this.precisionMode === mode && this.performanceContract) {
      return this.performanceContract;
    }
    this.precisionMode = mode;
    if (this.modelSpec && this.hparams && this.scratch) {
      const env = this.backend.calibrateEnvironment();
      this.performanceContract = derivePerformanceContract(env, this.modelSpec, {
        precisionMode: mode,
        requestedMaxContext: this.requestedMaxContext,
        backingKind: this.backingStore?.kind ?? 'memory',
        weightResidencyPolicy: this.weightResidencyPolicy,
        backingBandwidthMBs: this.calibratedBackingBandwidthMBs,
        backingQuotaBytes: this.backingQuotaBytes,
      });
      this.buildCompiledGraphs();
    }
    return this.performanceContract;
  }

  /**
   * Switches between Prism Theorem T4 automatic residency (`'auto'`) and bounded-RAM
   * OPFS-backed vRAM DIMM weight paging (`'paged'`) live from the already-open backing store
   * without re-downloading or re-parsing GGUF weights.
   */
  setWeightResidencyPolicy(policy: WeightResidencyPolicy): PerformanceContract | null {
    if (this.weightResidencyPolicy === policy && this.performanceContract) {
      return this.performanceContract;
    }
    this.weightResidencyPolicy = policy;
    if (this.backingStore && this.parsedHeader && this.modelSpec && this.hparams && this.modelSummary) {
      this.allocateFromActiveBacking();
    }
    return this.performanceContract;
  }

  getTokenizer(): GgufTokenizer {
    if (!this.tokenizer) {
      throw new Error('No GGUF model loaded.');
    }
    return this.tokenizer;
  }

  getTelemetry(): VTensorTelemetrySnapshot {
    return this.backend.getTelemetry();
  }

  getUnifiedStore(): VramUnifiedStore | null {
    return this.unifiedStore;
  }

  /** Returns the deterministic `<headerHash64>-<byteLength>` content fingerprint of the loaded model. */
  getContentFingerprint(): string | null {
    return this.modelContentFingerprint;
  }

  /**
   * Computes a priority-ordered, interruptible background prewarm plan for the active
   * `opfs-demand` model (early layers -> body layers -> LM-head stripes, strictly skipping
   * already-cached slabs and cold Sparse MoE expert tables).
   */
  getDemandPrewarmPlan(): ExpertPrewarmPlan | null {
    if (
      !this.backingStore ||
      this.backingStore.kind !== 'opfs-demand' ||
      !this.weightPager ||
      !this.hparams
    ) {
      return null;
    }
    const wp = this.weightPager;
    const demandStore =
      this.backingStore instanceof OpfsDemandVramBackingStore ? this.backingStore : null;
    return planPriorityWeightExtents({
      totalLayers: this.hparams.blockCount,
      isMoe: Boolean(this.modelSpec?.isMoe),
      getLayerRanges: (layerIdx) => wp.getLayerMissingRanges(layerIdx),
      lmHeadStripes: wp.getLmHeadStripeSpecs(),
      isRangeCached: demandStore
        ? (fileOffset, byteLength) => demandStore.isRangeCached(fileOffset, byteLength)
        : undefined,
    });
  }

  /**
   * Cooperative, abortable background prefetch of a priority weight slice into OPFS.
   * Preempts immediately when `signal` aborts (e.g., when the user submits a prompt).
   */
  async prefetchDemandRanges(
    ranges: readonly ByteRange[],
    signal?: AbortSignal,
  ): Promise<void> {
    if (!this.backingStore?.prefetchRanges || ranges.length === 0) return;
    await this.backingStore.prefetchRanges(ranges, {
      signal,
      concurrency: 6,
    });
  }

  /**
   * Returns true when every active transformer layer and LM-head stripe of an `opfs-demand`
   * model is already cached locally in OPFS (0 network fetches required).
   */
  isDemandModelFullyWarm(): boolean {
    if (!this.backingStore || this.backingStore.kind !== 'opfs-demand') {
      return true;
    }
    const plan = this.getDemandPrewarmPlan();
    return plan !== null && plan.slices.length === 0;
  }

  /** Flushes any dirty OPFS demand slab writes and slab directory index cleanly to storage. */
  flushDemandBacking(): void {
    if (this.backingStore instanceof OpfsDemandVramBackingStore) {
      this.backingStore.flush();
    }
  }

  /**
   * Pre-warms pinned WASM layer frames, lazy-pinned full-vocab LM-head cache, and an optional
   * initial system-prompt KV prefix into live WASM RAM directly from local OPFS (0 network bytes
   * when OPFS slabs are warm, or fetching any missing embedding row slabs for `bootPrompt`).
   * Guarantees instant, steady-state TTFT & tok/s on the user's very first prompt!
   */
  async warmRuntimeForInstantFirstToken(
    bootPrompt?: string,
    signal?: AbortSignal,
    onStage?: (detail: string) => void,
  ): Promise<{ readonly warmedLayers: number; readonly lmHeadCached: boolean; readonly readyTokens: number }> {
    if (!this.weightPager || !this.hparams) {
      return { warmedLayers: 0, lmHeadCached: false, readyTokens: 0 };
    }
    const wp = this.weightPager;
    let warmedLayers = 0;

    // 1. Pre-load pinned layer frames into WASM RAM from local OPFS so Turn 1 pays 0 OPFS layer reads
    if (wp.residencyMode === 'opfs-paged' && !wp.isMoe) {
      const pinTarget = Math.min(wp.totalLayers, wp.layerFrameSlots);
      onStage?.('Loading assistant into memory');
      for (let l = 0; l < pinTarget; l++) {
        LlamaVTensorRuntime.throwIfAborted(signal);
        if (
          this.backingStore instanceof OpfsDemandVramBackingStore &&
          wp.getLayerMissingRanges(l).some(
            (r) => !(this.backingStore as OpfsDemandVramBackingStore).isRangeCached(r.fileOffset, r.byteLength),
          )
        ) {
          break;
        }
        wp.acquireLayer(l);
        warmedLayers += 1;
        if (l % 4 === 3) {
          await this.yielder.yieldNow();
        }
      }
    }

    // 2. Pre-populate lazy-pinned full-vocab LM-head WASM cache from local OPFS (0 OPFS reads on decode!)
    let lmHeadCached = wp.getCachedFullLmHead() !== null;
    if (!lmHeadCached && wp.streamLmHead) {
      const stripes = wp.getLmHeadStripeSpecs();
      const allStripesOnDisk =
        !(this.backingStore instanceof OpfsDemandVramBackingStore) ||
        stripes.every((s) =>
          (this.backingStore as OpfsDemandVramBackingStore).isRangeCached(s.fileOffset, s.byteLength),
        );
      if (allStripesOnDisk) {
        for (let i = 0; i < stripes.length; i++) {
          LlamaVTensorRuntime.throwIfAborted(signal);
          wp.acquireLmHeadStripe(stripes[i].rowStart, stripes[i].rowCount);
          await this.yielder.yieldNow();
        }
        lmHeadCached = wp.getCachedFullLmHead() !== null;
      }
    }

    // 3. Pre-compute initial system-prompt KV prefix (fetching any missing prompt-token embedding rows
    //    first so speculative prefill succeeds even on a cold boot!)
    let readyTokens = 0;
    if (bootPrompt && this.tokenizer && this.kvPager) {
      const encoded = this.tokenizer.encode(bootPrompt, this.tokenizer.addBosToken);
      if (encoded.length > 0) {
        const embdMissing = wp.getEmbeddingMissingRanges(encoded);
        if (embdMissing.length > 0 && this.backingStore?.prefetchRanges) {
          await this.backingStore.prefetchRanges(embdMissing, { signal, concurrency: 6 });
        }
        onStage?.('Preparing context');
        const spec = await this.speculativePrefill(bootPrompt, signal);
        readyTokens = spec.readyTokens;
        if (!spec.aborted && spec.newlyPrefilledTokens > 0) {
          await this.persistUnifiedStoreToOpfs().catch(() => 0);
        }
      }
    }

    this.flushDemandBacking();
    return { warmedLayers, lmHeadCached, readyTokens };
  }

  async hydrateUnifiedStoreFromOpfs(
    opfsDirectoryOverride?: FileSystemDirectoryHandle,
  ): Promise<number> {
    if (!this.unifiedStore) return 0;
    const kvCount = await this.unifiedStore.hydrateFromOpfs(opfsDirectoryOverride);
    const wp = this.weightPager;
    const fp = this.modelContentFingerprint;
    if (wp && fp) {
      const embdSig = wp.getHotEmbeddingSignature();
      if (embdSig) {
        const rows = await readHotEmbeddingsCacheFromOpfs(fp, embdSig, opfsDirectoryOverride).catch(
          () => null,
        );
        if (rows && rows.length > 0) {
          wp.hydrateHotEmbeddingRows(rows);
        }
      }
    }
    return kvCount;
  }

  async persistUnifiedStoreToOpfs(
    opfsDirectoryOverride?: FileSystemDirectoryHandle,
  ): Promise<number> {
    if (!this.unifiedStore) return 0;
    const kvCount = await this.unifiedStore.persistToOpfs(opfsDirectoryOverride);
    const wp = this.weightPager;
    const fp = this.modelContentFingerprint;
    if (wp && fp) {
      const embdSig = wp.getHotEmbeddingSignature();
      if (embdSig) {
        const hotRows = wp.exportHotEmbeddingRows(256);
        if (hotRows.length > 0) {
          await writeHotEmbeddingsCacheToOpfs(fp, embdSig, hotRows, opfsDirectoryOverride).catch(
            () => false,
          );
        }
      }
    }
    return kvCount;
  }

  unload(): void {
    this.backend.freeAllBuffers();
    if (this.backingStore) {
      this.backingStore.close();
      this.backingStore = null;
    }
    this.modelContentFingerprint = null;
    this.parsedHeader = null;
    this.weightPager = null;
    this.embdInfo = null;
    this.outNormInfo = null;
    this.outWeightInfo = null;
    this.isTiedOutput = false;
    this.layerInfos = [];
    this.kvPager = null;
    this.cachedTokens = [];
    this.activeThreadId = null;
    this.unifiedStore = null;
    this.committedPageKeys.clear();
    this.ropeTables = null;
    this.preloadedRopeTables = null;
    this.preloadedRopeParams = null;
    this.scratch = null;
    this.batchScratch = null;
    this.compiledSingleGraph = null;
    this.compiledBatch4Graph = null;
    this.batchEmbedNode = null;
    this.batchLayerAcquireNodes = [];
    this.batchLayerBodyNodes = [];
    this.prefillChunkX = null;
    this.singleAsyncPlan = null;
    this.batchAsyncPlan = null;
    this.preloadedHotTensors = null;
    this.singleGraphBodyNodeCount = 0;
    this.modelSummary = null;
    this.modelSpec = null;
    this.performanceContract = null;
    this.prismLedger.reset();
    this.prismSchedule = null;
    this.tokenizer = null;
    this.hparams = null;
  }

  /**
   * Backward-compatible entry point wrapping an in-memory GGUF ArrayBuffer / Uint8Array in a
   * `MemoryBackingStore` and delegating to `loadFromBacking`.
   */
  loadFromBuffer(
    id: string,
    displayName: string,
    source: string,
    buffer: ArrayBuffer | Uint8Array,
    maxContextCap?: number,
  ): LoadedModelSummary {
    const backing = new MemoryBackingStore(id, buffer);
    return this.loadFromBacking(id, displayName, source, backing, { maxContextCap });
  }

  /**
   * Parses ONLY the GGUF header from an OPFS or memory `VramBackingStore` (never allocating a
   * full multi-GB JS ArrayBuffer!), calibrates the host environment, solves the Prism Performance
   * Contract (Theorems T1–T4b), binds either resident-all or OPFS-paged vRAM weight frames,
   * initializes the 3-tier vRAM KV pager, precomputes RoPE tables, and compiles reusable graphs.
   */
  loadFromBacking(
    id: string,
    displayName: string,
    source: string,
    backing: VramBackingStore,
    options?: {
      maxContextCap?: number;
      weightResidencyPolicy?: WeightResidencyPolicy;
      backingQuotaBytes?: number;
    },
  ): LoadedModelSummary {
    this.unload();
    this.backend.resetMetrics();

    if (options?.weightResidencyPolicy) {
      this.weightResidencyPolicy = options.weightResidencyPolicy;
    }
    this.backingQuotaBytes = options?.backingQuotaBytes;

    let parsed: ParsedGgufModel;
    try {
      parsed = parseGgufFromBacking(backing);
      this.modelContentFingerprint = computeBackingContentFingerprint(backing);
    } catch (err) {
      backing.close();
      throw err;
    }

    return this.finishLoadFromParsedBacking(id, displayName, source, backing, parsed, options?.maxContextCap);
  }

  /**
   * Async demand-aware model loader used by the Web Worker for multi-GB remote & OPFS models.
   * For `opfs-demand` & `opfs-sync` backings:
   *  1. Computes the 4 KiB GGUF content fingerprint (`<headerHash64>-<byteLength>`) and checks
   *     `<fp>.hdr.bin` to bypass re-fetching/re-parsing 8–12 MiB 150K-vocab GGUF headers.
   *  2. Preloads `<fp>.rope.bin` and `<fp>.hot.bin` in parallel with lock-free read-only OPFS I/O,
   *     falling back to coalesced exact HTTP Ranges only on first boot.
   *  3. Honors `AbortSignal` throughout so cancelling at 97% returns immediately.
   */
  async loadFromBackingAsync(
    id: string,
    displayName: string,
    source: string,
    backing: VramBackingStore,
    options?: {
      maxContextCap?: number;
      weightResidencyPolicy?: WeightResidencyPolicy;
      backingQuotaBytes?: number;
      signal?: AbortSignal;
      onProgress?: (progress: AsyncLoadProgressEvent) => void;
      opfsDirectoryOverride?: FileSystemDirectoryHandle;
    },
  ): Promise<LoadedModelSummary> {
    this.unload();
    this.backend.resetMetrics();

    if (options?.weightResidencyPolicy) {
      this.weightResidencyPolicy = options.weightResidencyPolicy;
    }
    this.backingQuotaBytes = options?.backingQuotaBytes;

    const useOpfsBootCache = backing.isPersistent || Boolean(options?.opfsDirectoryOverride);
    let contentFingerprint: string | null = null;
    let parsed: ParsedGgufModel;
    let headerLoadedFromBootCache = false;
    try {
      options?.onProgress?.({ percent: 97, detail: `Reading ${displayName} header` });
      contentFingerprint = await computeBackingContentFingerprintAsync(backing, options?.signal);
      this.modelContentFingerprint = contentFingerprint;

      const cachedParsed = useOpfsBootCache
        ? await readParsedGgufBootCacheFromOpfs(
            contentFingerprint,
            contentFingerprint,
            backing.byteLength,
            options?.opfsDirectoryOverride,
          )
        : null;

      if (cachedParsed) {
        parsed = cachedParsed;
        headerLoadedFromBootCache = true;
        options?.onProgress?.({
          percent: 97,
          detail: `Loaded ${displayName} header from local cache`,
        });
      } else {
        parsed = await parseGgufFromBackingAsync(backing, {
          signal: options?.signal,
          onProgress: (detail) => options?.onProgress?.({ percent: 97, detail }),
        });
      }
    } catch (err) {
      backing.close();
      throw err;
    }

    if (options?.signal?.aborted) {
      backing.close();
      throw new DOMException('Aborted', 'AbortError');
    }

    const prepared = this.prepareParsedBacking(id, displayName, source, backing, parsed, options?.maxContextCap);

    // Persist parsed GGUF header + tokenizer vocab/merges boot cache on cold parse.
    // In real browser OPFS (`!options?.opfsDirectoryOverride`), run non-blocking in the background
    // so serializing/flushing an 8 MB 150K-vocab header never delays first-turn readiness!
    if (useOpfsBootCache && !headerLoadedFromBootCache && contentFingerprint && !options?.signal?.aborted) {
      const writeHdrPromise = writeParsedGgufBootCacheToOpfs(
        contentFingerprint,
        contentFingerprint,
        parsed,
        options?.opfsDirectoryOverride,
      ).catch(() => false);
      if (options?.opfsDirectoryOverride) {
        await writeHdrPromise;
      }
    }

    // Solve contract once upfront so we can preload RoPE LUT + hot metadata tensors concurrently!
    const env = this.backend.calibrateEnvironment();
    const contract = derivePerformanceContract(env, prepared.modelSpec, {
      precisionMode: this.precisionMode,
      requestedMaxContext: this.requestedMaxContext,
      backingKind: backing.kind,
      weightResidencyPolicy: this.weightResidencyPolicy,
      backingBandwidthMBs: this.calibratedBackingBandwidthMBs,
      backingQuotaBytes: this.backingQuotaBytes,
    });

    const hp = parsed.hparams;
    const ropeParams: RopeCacheParams = {
      maxCtx: Math.min(contract.effectiveMaxContext, 4096),
      ropeDim: hp.ropeDimensionCount,
      freqBase: hp.ropeFreqBase,
      freqScale: hp.ropeFreqScale,
    };

    const preloadRopePromise: Promise<void> =
      useOpfsBootCache && contentFingerprint
        ? readRopeTablesCacheFromOpfs(
            contentFingerprint,
            ropeParams,
            options?.opfsDirectoryOverride,
          )
            .then((cachedRope) => {
              if (cachedRope) {
                this.preloadedRopeTables = cachedRope;
                this.preloadedRopeParams = ropeParams;
              }
            })
            .catch(() => {})
        : Promise.resolve();

    // Preload tiny hot metadata tensors asynchronously on demand-paged backings (zero sync XHRs!)
    const preloadHotPromise: Promise<void> = (async () => {
      if (!backing.readRangesDirectAsync && !backing.prefetchRanges) return;
      if (contract.weightResidencyMode === 'opfs-paged') {
        const hotInfos = collectHotMetadataTensorInfos(
          prepared.embdInfo,
          prepared.outNormInfo,
          prepared.outWeightInfo,
          prepared.isTiedOutput,
          prepared.layerInfos,
          {
            streamEmbeddings: contract.streamEmbeddings,
            streamLmHead: contract.streamLmHead,
          },
        );
        const smallInfos = hotInfos.filter((t) => t.byteSize <= 256 * 1024);
        const largeInfos = hotInfos.filter((t) => t.byteSize > 256 * 1024);

        const preloaded = new Map<string, Uint8Array>();
        if (smallInfos.length > 0 && backing.readRangesDirectAsync) {
          // Key `.hot.bin` by contentFingerprint so renamed/mirror/custom-URL mounts share it too,
          // with transparent fallback to legacy `<id>.hot.bin`!
          const hotKey = contentFingerprint ?? id;
          const hotSig = {
            dataOffset: parsed.dataOffset,
            backingByteLength: backing.byteLength,
            expectedTensors: smallInfos,
          };
          let cachedHot = useOpfsBootCache
            ? await readHotMetadataBundleFromOpfs(hotKey, hotSig, options?.opfsDirectoryOverride)
            : null;
          if (!cachedHot && useOpfsBootCache && hotKey !== id) {
            cachedHot = await readHotMetadataBundleFromOpfs(id, hotSig, options?.opfsDirectoryOverride);
          }

          if (cachedHot) {
            options?.onProgress?.({
              percent: 98,
              detail: `Loaded ${displayName} layer metadata from local cache`,
            });
            for (let i = 0; i < smallInfos.length; i++) {
              const name = smallInfos[i].name;
              const slice = cachedHot.get(name);
              if (slice) preloaded.set(name, slice);
            }
          } else {
            options?.onProgress?.({
              percent: 98,
              detail: `Loading ${displayName} layer metadata`,
            });
            const rawSlices = await backing.readRangesDirectAsync(
              smallInfos.map((t) => ({
                fileOffset: parsed.dataOffset + t.offset,
                byteLength: t.byteSize,
              })),
              {
                signal: options?.signal,
                onProgress: (done, total) => {
                  const pct = total > 0 ? 97 + Math.min(2, Math.round((done / total) * 2)) : 98;
                  options?.onProgress?.({
                    percent: pct,
                    detail: `Loading layer metadata (${done}/${total})`,
                  });
                },
              },
            );
            for (let i = 0; i < smallInfos.length; i++) {
              preloaded.set(smallInfos[i].name, rawSlices[i]);
            }
            if (useOpfsBootCache && !options?.signal?.aborted) {
              await writeHotMetadataBundleToOpfs(
                hotKey,
                { dataOffset: parsed.dataOffset, backingByteLength: backing.byteLength },
                preloaded,
                options?.opfsDirectoryOverride,
              ).catch(() => {});
            }
          }
        }
        if (largeInfos.length > 0 && backing.prefetchRanges) {
          options?.onProgress?.({
            percent: 99,
            detail: `Caching ${displayName} embeddings`,
          });
          await backing.prefetchRanges(
            largeInfos.map((t) => ({
              fileOffset: parsed.dataOffset + t.offset,
              byteLength: t.byteSize,
            })),
            { signal: options?.signal },
          );
        }
        if (preloaded.size > 0) {
          this.preloadedHotTensors = preloaded;
        }
      } else if (backing.prefetchRanges) {
        options?.onProgress?.({
          percent: 98,
          detail: `Loading ${displayName} weights`,
        });
        await backing.prefetchRanges(
          [
            {
              fileOffset: parsed.dataOffset,
              byteLength: Math.max(0, backing.byteLength - parsed.dataOffset),
            },
          ],
          { signal: options?.signal },
        );
      }
    })();

    try {
      await Promise.all([preloadRopePromise, preloadHotPromise]);
    } catch (err) {
      this.unload();
      throw err;
    }

    if (options?.signal?.aborted) {
      this.unload();
      throw new DOMException('Aborted', 'AbortError');
    }

    const ropeWasCached = this.preloadedRopeTables !== null;
    options?.onProgress?.({ percent: 99, detail: `Initializing ${displayName}` });
    this.allocateFromActiveBacking();

    // Persist computed RoPE cos/sin LUT to `<fp>.rope.bin` on cold miss (non-blocking in browser OPFS)
    if (
      useOpfsBootCache &&
      !ropeWasCached &&
      contentFingerprint &&
      this.ropeTables &&
      !options?.signal?.aborted
    ) {
      const actualRopeParams: RopeCacheParams = {
        maxCtx: this.ropeTables.maxCtx,
        ropeDim: hp.ropeDimensionCount,
        freqBase: hp.ropeFreqBase,
        freqScale: hp.ropeFreqScale,
      };
      const writeRopePromise = writeRopeTablesCacheToOpfs(
        contentFingerprint,
        this.ropeTables,
        actualRopeParams,
        options?.opfsDirectoryOverride,
      ).catch(() => false);
      if (options?.opfsDirectoryOverride) {
        await writeRopePromise;
      }
    }

    return this.modelSummary!;
  }

  private finishLoadFromParsedBacking(
    id: string,
    displayName: string,
    source: string,
    backing: VramBackingStore,
    parsed: ParsedGgufModel,
    maxContextCap?: number,
  ): LoadedModelSummary {
    this.prepareParsedBacking(id, displayName, source, backing, parsed, maxContextCap);
    this.allocateFromActiveBacking();
    return this.modelSummary!;
  }

  private prepareParsedBacking(
    id: string,
    displayName: string,
    source: string,
    backing: VramBackingStore,
    parsed: ParsedGgufModel,
    maxContextCap?: number,
  ): {
    embdInfo: GgufTensorInfo;
    outNormInfo: GgufTensorInfo;
    outWeightInfo: GgufTensorInfo;
    isTiedOutput: boolean;
    layerInfos: LayerTensorInfoGroup[];
    modelSpec: ModelGeometrySpec;
  } {

    const hp = parsed.hparams;
    const hasDenseFfn0 =
      parsed.tensorMap.has('blk.0.ffn_gate.weight') &&
      parsed.tensorMap.has('blk.0.ffn_up.weight') &&
      parsed.tensorMap.has('blk.0.ffn_down.weight');
    const hasMoeFfn0 =
      parsed.tensorMap.has('blk.0.ffn_gate_inp.weight') &&
      parsed.tensorMap.has('blk.0.ffn_gate_exps.weight') &&
      parsed.tensorMap.has('blk.0.ffn_up_exps.weight') &&
      parsed.tensorMap.has('blk.0.ffn_down_exps.weight');

    const hasStandardTransformerTopology =
      parsed.tensorMap.has('token_embd.weight') &&
      parsed.tensorMap.has('output_norm.weight') &&
      parsed.tensorMap.has('blk.0.attn_norm.weight') &&
      parsed.tensorMap.has('blk.0.attn_q.weight') &&
      parsed.tensorMap.has('blk.0.attn_k.weight') &&
      parsed.tensorMap.has('blk.0.attn_v.weight') &&
      parsed.tensorMap.has('blk.0.attn_output.weight') &&
      parsed.tensorMap.has('blk.0.ffn_norm.weight') &&
      (hasDenseFfn0 || hasMoeFfn0);

    if (!SUPPORTED_ARCHITECTURES.has(hp.architecture) && !hasStandardTransformerTopology) {
      backing.close();
      throw new Error(
        `GGUF architecture "${hp.architecture}" uses an unsupported tensor graph topology. Supported architectures: llama, qwen2, qwen2moe, qwen3, qwen3moe, mistral, mixtral, granite, smollm3, internlm2, olmo2.`,
      );
    }

    let tokenizer: GgufTokenizer;
    try {
      tokenizer = new GgufTokenizer(parsed);
    } catch (err) {
      backing.close();
      throw err;
    }

    const requireTensor = (name: string) => {
      const info = parsed.tensorMap.get(name);
      if (!info) {
        backing.close();
        throw new Error(`GGUF model is missing required tensor "${name}".`);
      }
      return info;
    };

    const optionalTensor = (name: string) => parsed.tensorMap.get(name) ?? null;

    const embdInfo = requireTensor('token_embd.weight');
    const outNormInfo = requireTensor('output_norm.weight');
    const outWeightInfo = parsed.tensorMap.get('output.weight') ?? embdInfo;
    const isTiedOutput = !parsed.tensorMap.has('output.weight');

    const tensorInfosToLoad: GgufTensorInfo[] = [embdInfo, outNormInfo];
    if (!isTiedOutput) tensorInfosToLoad.push(outWeightInfo);

    const layerInfos: LayerTensorInfoGroup[] = [];
    let anyMoeLayer = false;
    for (let l = 0; l < hp.blockCount; l++) {
      const ffnGateInp = optionalTensor(`blk.${l}.ffn_gate_inp.weight`);
      const ffnGateExps = optionalTensor(`blk.${l}.ffn_gate_exps.weight`);
      const ffnUpExps = optionalTensor(`blk.${l}.ffn_up_exps.weight`);
      const ffnDownExps = optionalTensor(`blk.${l}.ffn_down_exps.weight`);
      const layerIsMoe = Boolean(ffnGateInp && ffnGateExps && ffnUpExps && ffnDownExps);
      if (layerIsMoe) anyMoeLayer = true;

      const layer: LayerTensorInfoGroup = {
        attnNorm: requireTensor(`blk.${l}.attn_norm.weight`),
        attnQ: requireTensor(`blk.${l}.attn_q.weight`),
        attnQBias: optionalTensor(`blk.${l}.attn_q.bias`),
        attnQNorm: optionalTensor(`blk.${l}.attn_q_norm.weight`),
        attnK: requireTensor(`blk.${l}.attn_k.weight`),
        attnKBias: optionalTensor(`blk.${l}.attn_k.bias`),
        attnKNorm: optionalTensor(`blk.${l}.attn_k_norm.weight`),
        attnV: requireTensor(`blk.${l}.attn_v.weight`),
        attnVBias: optionalTensor(`blk.${l}.attn_v.bias`),
        attnOutput: requireTensor(`blk.${l}.attn_output.weight`),
        attnOutputBias: optionalTensor(`blk.${l}.attn_output.bias`),
        ffnNorm: requireTensor(`blk.${l}.ffn_norm.weight`),
        ffnGate: layerIsMoe ? null : requireTensor(`blk.${l}.ffn_gate.weight`),
        ffnUp: layerIsMoe ? null : requireTensor(`blk.${l}.ffn_up.weight`),
        ffnDown: layerIsMoe ? null : requireTensor(`blk.${l}.ffn_down.weight`),
        ffnGateInp,
        ffnGateExps,
        ffnUpExps,
        ffnDownExps,
      };
      layerInfos.push(layer);
      tensorInfosToLoad.push(
        layer.attnNorm,
        layer.attnQ,
        layer.attnK,
        layer.attnV,
        layer.attnOutput,
        layer.ffnNorm,
      );
      if (layer.ffnGate) tensorInfosToLoad.push(layer.ffnGate);
      if (layer.ffnUp) tensorInfosToLoad.push(layer.ffnUp);
      if (layer.ffnDown) tensorInfosToLoad.push(layer.ffnDown);
      if (layer.ffnGateInp) tensorInfosToLoad.push(layer.ffnGateInp);
      if (layer.ffnGateExps) tensorInfosToLoad.push(layer.ffnGateExps);
      if (layer.ffnUpExps) tensorInfosToLoad.push(layer.ffnUpExps);
      if (layer.ffnDownExps) tensorInfosToLoad.push(layer.ffnDownExps);
      if (layer.attnQBias) tensorInfosToLoad.push(layer.attnQBias);
      if (layer.attnQNorm) tensorInfosToLoad.push(layer.attnQNorm);
      if (layer.attnKBias) tensorInfosToLoad.push(layer.attnKBias);
      if (layer.attnKNorm) tensorInfosToLoad.push(layer.attnKNorm);
      if (layer.attnVBias) tensorInfosToLoad.push(layer.attnVBias);
      if (layer.attnOutputBias) tensorInfosToLoad.push(layer.attnOutputBias);
    }

    for (const info of tensorInfosToLoad) {
      if (!SUPPORTED_VERDANT_STORAGE_TYPES.has(info.type)) {
        const traitName = getGgmlTypeTrait(info.type).name;
        backing.close();
        throw new Error(
          `Tensor "${info.name}" uses quantization ${traitName}, which is not yet supported in this browser Verdant build. Supported formats: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K.`,
        );
      }
    }

    // Compute exact ModelGeometrySpec for Prism Theorems T1–T4b (distinguishing active vs total bytes on MoE!)
    let weightsCapacity = 0;
    for (const info of tensorInfosToLoad) {
      weightsCapacity += alignUp32(info.byteSize) + 32;
    }

    let activeWeightBytes = outWeightInfo.byteSize;
    let activeParamCount = (outWeightInfo.dims[0] ?? 1) * (outWeightInfo.dims[1] ?? 1);
    let quantizedLinearBytes = INTEGER_DOT_COMPATIBLE_TYPES.has(outWeightInfo.type)
      ? outWeightInfo.byteSize
      : 0;
    const typeByteCounts = new Map<GgmlType, number>();
    typeByteCounts.set(outWeightInfo.type, outWeightInfo.byteSize);

    let moeAttentionLayerBytesSum = 0;
    let moeSingleExpertTripleBytes = 0;
    let detectedExpertCount = hp.expertCount ?? 0;
    const detectedExpertUsedCount = Math.max(1, hp.expertUsedCount ?? (anyMoeLayer ? 2 : 0));

    for (const l of layerInfos) {
      const attnWeights = [l.attnQ, l.attnK, l.attnV, l.attnOutput];
      for (const w of attnWeights) {
        activeWeightBytes += w.byteSize;
        activeParamCount += (w.dims[0] ?? 1) * (w.dims[1] ?? 1);
        if (INTEGER_DOT_COMPATIBLE_TYPES.has(w.type)) {
          quantizedLinearBytes += w.byteSize;
        }
        typeByteCounts.set(w.type, (typeByteCounts.get(w.type) ?? 0) + w.byteSize);
        moeAttentionLayerBytesSum += w.byteSize;
      }

      if (l.ffnGateInp && l.ffnGateExps && l.ffnUpExps && l.ffnDownExps) {
        moeAttentionLayerBytesSum += l.ffnGateInp.byteSize;
        activeWeightBytes += l.ffnGateInp.byteSize;
        activeParamCount += (l.ffnGateInp.dims[0] ?? 1) * (l.ffnGateInp.dims[1] ?? 1);

        const nExps = Math.max(1, l.ffnGateExps.dims[2] ?? hp.expertCount ?? 1);
        detectedExpertCount = Math.max(detectedExpertCount, nExps);
        const kUsed = Math.min(nExps, detectedExpertUsedCount);
        const singleGateBytes = Math.floor(l.ffnGateExps.byteSize / nExps);
        const singleUpBytes = Math.floor(l.ffnUpExps.byteSize / nExps);
        const singleDownBytes = Math.floor(l.ffnDownExps.byteSize / nExps);
        moeSingleExpertTripleBytes = Math.max(
          moeSingleExpertTripleBytes,
          singleGateBytes + singleUpBytes + singleDownBytes,
        );

        const expTensors: Array<{ w: GgufTensorInfo; singleBytes: number }> = [
          { w: l.ffnGateExps, singleBytes: singleGateBytes },
          { w: l.ffnUpExps, singleBytes: singleUpBytes },
          { w: l.ffnDownExps, singleBytes: singleDownBytes },
        ];
        for (const { w, singleBytes } of expTensors) {
          const activeExpBytes = singleBytes * kUsed;
          activeWeightBytes += activeExpBytes;
          activeParamCount += (w.dims[0] ?? 1) * (w.dims[1] ?? 1) * kUsed;
          if (INTEGER_DOT_COMPATIBLE_TYPES.has(w.type)) {
            quantizedLinearBytes += activeExpBytes;
          }
          typeByteCounts.set(w.type, (typeByteCounts.get(w.type) ?? 0) + activeExpBytes);
        }
      } else if (l.ffnGate && l.ffnUp && l.ffnDown) {
        const denseFfn = [l.ffnGate, l.ffnUp, l.ffnDown];
        for (const w of denseFfn) {
          activeWeightBytes += w.byteSize;
          activeParamCount += (w.dims[0] ?? 1) * (w.dims[1] ?? 1);
          if (INTEGER_DOT_COMPATIBLE_TYPES.has(w.type)) {
            quantizedLinearBytes += w.byteSize;
          }
          typeByteCounts.set(w.type, (typeByteCounts.get(w.type) ?? 0) + w.byteSize);
        }
      }
    }

    let dominantWeightType = outWeightInfo.type;
    let dominantBytes = -1;
    for (const [t, bytes] of typeByteCounts.entries()) {
      if (bytes > dominantBytes) {
        dominantBytes = bytes;
        dominantWeightType = t;
      }
    }

    const modelSpec: ModelGeometrySpec = {
      hparams: hp,
      activeWeightBytes,
      lmHeadWeightBytes: outWeightInfo.byteSize,
      activeParamCount,
      totalWeightBufferBytes: weightsCapacity,
      dominantWeightType,
      quantizedWeightFraction: activeWeightBytes > 0 ? quantizedLinearBytes / activeWeightBytes : 0,
      isMoe: anyMoeLayer,
      expertCount: anyMoeLayer ? detectedExpertCount : 0,
      expertUsedCount: anyMoeLayer ? detectedExpertUsedCount : 0,
      moeAttentionLayerBytes: anyMoeLayer
        ? Math.ceil(moeAttentionLayerBytesSum / Math.max(1, layerInfos.length))
        : 0,
      moeSingleExpertTripleBytes,
    };

    this.backingStore = backing;
    this.calibratedBackingBandwidthMBs = calibrateBackingReadBandwidthMBs(backing);
    this.parsedHeader = parsed;
    this.embdInfo = embdInfo;
    this.outNormInfo = outNormInfo;
    this.outWeightInfo = outWeightInfo;
    this.isTiedOutput = isTiedOutput;
    this.layerInfos = layerInfos;
    this.hparams = hp;
    this.tokenizer = tokenizer;
    this.modelSpec = modelSpec;
    this.requestedMaxContext =
      maxContextCap != null && maxContextCap > 0
        ? Math.min(hp.contextLength, maxContextCap)
        : hp.contextLength;

    this.modelSummary = {
      id,
      displayName,
      source,
      fileByteLength: parsed.fileByteLength,
      tensorCount: parsed.tensors.length,
      hparams: hp,
      maxContextWindow: this.requestedMaxContext,
      backingKind: backing.kind,
    };

    return {
      embdInfo,
      outNormInfo,
      outWeightInfo,
      isTiedOutput,
      layerInfos,
      modelSpec,
    };
  }

  /**
   * Allocates vTensor weight buffers (resident-all or OPFS-backed vRAM paged DIMM frames),
   * 3-tier vRAM KV cache, compute scratch, RoPE LUT, and compiled Prism graphs from the active
   * backing store. Automatically steps down from `resident-all` -> `opfs-paged` (and trims frame
   * slots/context) if WASM32 memory growth is constrained on tight devices.
   */
  private allocateFromActiveBacking(): void {
    const attempts: Array<{
      policy: WeightResidencyPolicy;
      maxLayerFrameSlotsCap?: number;
      memoryBudgetBytesOverride?: number;
    }> = [
      { policy: this.weightResidencyPolicy },
      { policy: 'paged', memoryBudgetBytesOverride: 1536 * 1024 * 1024 },
      { policy: 'paged', memoryBudgetBytesOverride: 768 * 1024 * 1024, maxLayerFrameSlotsCap: 4 },
      { policy: 'paged', memoryBudgetBytesOverride: 512 * 1024 * 1024, maxLayerFrameSlotsCap: 1 },
    ];

    let lastErr: unknown = null;
    for (let i = 0; i < attempts.length; i++) {
      const plan = attempts[i];
      try {
        this.allocateWithPlan(
          plan.policy,
          plan.maxLayerFrameSlotsCap,
          undefined,
          plan.memoryBudgetBytesOverride,
        );
        return;
      } catch (err) {
        lastErr = err;
        const msg = err instanceof Error ? err.message : String(err);
        const isMemoryConstraint =
          err instanceof RangeError ||
          /out of memory|unable to grow|wasm|memory|allocation|overflow/i.test(msg);
        if (!isMemoryConstraint || i === attempts.length - 1) {
          throw err;
        }
        // Step down cleanly after freeing partial buffers from the failed attempt
        this.backend.freeAllBuffers();
      }
    }
    if (lastErr) throw lastErr;
  }

  private allocateWithPlan(
    policy: WeightResidencyPolicy,
    maxLayerFrameSlotsCap?: number,
    maxContextOverride?: number,
    memoryBudgetBytesOverride?: number,
  ): void {
    const backing = this.backingStore!;
    const parsed = this.parsedHeader!;
    const hp = this.hparams!;
    const modelSpec = this.modelSpec!;
    const embdInfo = this.embdInfo!;
    const outNormInfo = this.outNormInfo!;
    const outWeightInfo = this.outWeightInfo!;

    this.backend.freeAllBuffers();

    // 0. Calibrate hardware & solve the formal Prism Performance Contract (Theorems T1–T4b)
    const env = this.backend.calibrateEnvironment();
    const effectiveBudgetBytes =
      memoryBudgetBytesOverride && memoryBudgetBytesOverride > 0
        ? Math.min(env.memoryBudgetBytes, memoryBudgetBytesOverride)
        : env.memoryBudgetBytes;
    const contract = derivePerformanceContract(env, modelSpec, {
      precisionMode: this.precisionMode,
      requestedMaxContext: maxContextOverride ?? this.requestedMaxContext,
      backingKind: backing.kind,
      weightResidencyPolicy: policy,
      backingBandwidthMBs: this.calibratedBackingBandwidthMBs,
      backingQuotaBytes: this.backingQuotaBytes,
      maxLayerFrameSlotsCap,
      memoryBudgetBytesOverride: effectiveBudgetBytes,
    });
    if (this.backend.activeThreadPool && Number.isFinite(contract.parallelThresholdBytes)) {
      this.backend.activeThreadPool.setParallelThresholdBytes(contract.parallelThresholdBytes);
    }

    // 0b. Solve Theorem T6 Unified Virtual Store Global Space Allocation (weights + hot embeddings + KV prefixes)
    const embdTrait = getGgmlTypeTrait(embdInfo.type);
    const embdRowBytes = ((embdInfo.dims[0] ?? hp.embeddingLength) / embdTrait.blockSize) * embdTrait.typeSizeBytes;
    const headDim = hp.headDim;
    const q8BytesPerHalfPage = hp.blockCount * VRAM_KV_PAGE_TOKENS * hp.headCountKv * headDim;
    const scaleFloatsPerHalfPage = hp.blockCount * VRAM_KV_PAGE_TOKENS * hp.headCountKv;
    const kvPageBytesWarmQ8 = 2 * (q8BytesPerHalfPage + scaleFloatsPerHalfPage * 4);

    const unifiedPlan = deriveUnifiedStoreAllocation({
      globalBudgetBytes: effectiveBudgetBytes,
      weightAndLiveKvBytes: contract.projectedTotalArenaBytes,
      embdRowBytes,
      streamEmbeddings: contract.streamEmbeddings,
      vocabSize: hp.vocabSize,
      kvPageBytesWarmQ8,
    });

    // 1. Allocate model weights through OPFS-backed vRAM Weight Pager (resident-all or opfs-paged + Zipfian hot embedding slab + Sparse MoE LRU expert cache)
    const weightPager = new VramWeightPager(
      parsed,
      backing,
      embdInfo,
      outNormInfo,
      outWeightInfo,
      this.isTiedOutput,
      this.layerInfos,
      {
        residencyMode: contract.weightResidencyMode,
        layerFrameSlots: contract.weightLayerFrameSlots,
        streamEmbeddings: contract.streamEmbeddings,
        streamLmHead: contract.streamLmHead,
        lazyPinLmHead: contract.lazyPinLmHead,
        lmHeadStripeRows: computeLmHeadStripeRows(outWeightInfo, contract.lmHeadTileTargetBytes),
        hotEmbeddingCacheSlots: unifiedPlan.hotEmbeddingCacheSlots,
        moeExpertCacheSlots: contract.moeExpertCacheSlots,
      },
      (bufId, usage, cap) => this.backend.allocBuffer(bufId, usage, cap),
      this.preloadedHotTensors ?? undefined,
    );
    this.backend.attachVramWeightTelemetry(() => weightPager.getTelemetry());

    // 2. Allocate vRAM 4-Tier Paged KV-Cache driven strictly by Prism Theorem T4 (preserves full context)
    const maxCtx = contract.effectiveMaxContext;
    const qDim = hp.headCount * headDim;
    const kvDim = hp.headCountKv * headDim;
    const kvPager = new VramKvPager(
      {
        numLayers: hp.blockCount,
        maxContext: maxCtx,
        nHeadsKv: hp.headCountKv,
        headDim,
        hotFrameLimit: contract.hotFrameLimit,
        warmQ8Limit: contract.warmQ8Limit,
        coldQ4Enabled: contract.coldQ4Enabled,
        coldQ4PoolLimit: contract.coldQ4RamLimit,
      },
      (bufId, usage, cap) => this.backend.allocBuffer(bufId, usage, cap),
    );
    this.backend.attachVramKvTelemetry(() => kvPager.getTelemetry());

    // 2b. Initialize or retain the Unified Virtual Store keyed by underlying GGUF contentFingerprint
    //     so renamed, custom-URL, or mirror-deduplicated mounts share warm KV prefixes seamlessly!
    const modelId = this.modelSummary?.id ?? backing.modelId;
    const contentFingerprint = this.modelContentFingerprint ?? undefined;
    const expectedIdentityKey = contentFingerprint ?? modelId;
    const existingStore = this.unifiedStore;
    const reuseExistingStore =
      existingStore !== null &&
      existingStore.identityKey === expectedIdentityKey &&
      existingStore.geometry.numLayers === hp.blockCount &&
      existingStore.geometry.nHeadsKv === hp.headCountKv &&
      existingStore.geometry.headDim === headDim;

    const unifiedStore = reuseExistingStore
      ? existingStore
      : new VramUnifiedStore(
          {
            modelId,
            contentFingerprint,
            numLayers: hp.blockCount,
            nHeadsKv: hp.headCountKv,
            headDim,
            pageTokens: VRAM_KV_PAGE_TOKENS,
            q8BytesPerHalfPage,
            scaleFloatsPerHalfPage,
          },
          unifiedPlan,
          env.memoryBudgetBytes,
        );
    this.unifiedStore = unifiedStore;
    this.committedPageKeys.clear();
    this.backend.attachUnifiedStoreTelemetry((): UnifiedStoreTelemetry => {
      const wt = weightPager.getTelemetry();
      const kvT = kvPager.getTelemetry();
      return unifiedStore.getTelemetry({
        weightSlabBytes: wt.residentWeightBytes,
        liveKvSlabBytes: kvT.allocatedBytes,
        embeddingCacheBytes: wt.embdCacheBytes,
        embdCacheSlots: wt.embdCacheSlots,
        embdCacheResidentCount: wt.embdCacheResidentCount,
        embdCacheHits: wt.embdCacheHits,
        embdCacheMisses: wt.embdCacheMisses,
      });
    });

    // 3. Allocate single-token + batched-4 compute scratch tensors (including packed Q8_0_ACT & MoE slots)
    const dim = hp.embeddingLength;
    const ffnDim = Math.max(hp.feedForwardLength, hp.expertFeedForwardLength ?? 0);
    const moeExpertCount = modelSpec.isMoe ? Math.max(1, modelSpec.expertCount ?? hp.expertCount ?? 1) : 0;
    const vocab = hp.vocabSize;
    const B = PREFILL_BATCH_SIZE;

    const canQuantizeDim = (dim & 31) === 0;
    const canQuantizeQDim = (qDim & 31) === 0;
    const canQuantizeFfn = (ffnDim & 31) === 0;
    const q8DimBytes = canQuantizeDim ? (dim / 32) * Q8_0_ACT_BLOCK_BYTES : 0;
    const q8AttnOutBytes = canQuantizeQDim ? (qDim / 32) * Q8_0_ACT_BLOCK_BYTES : 0;
    const q8FfnBytes = canQuantizeFfn ? (ffnDim / 32) * Q8_0_ACT_BLOCK_BYTES : 0;

    const canQuantizeQ8KDim = (dim & 255) === 0;
    const canQuantizeQ8KQDim = (qDim & 255) === 0;
    const canQuantizeQ8KFfn = (ffnDim & 255) === 0;
    const q8kDimBytes = canQuantizeQ8KDim ? (dim >>> 8) * Q8_K_ACT_BLOCK_BYTES : 0;
    const q8kAttnOutBytes = canQuantizeQ8KQDim ? (qDim >>> 8) * Q8_K_ACT_BLOCK_BYTES : 0;
    const q8kFfnBytes = canQuantizeQ8KFfn ? (ffnDim >>> 8) * Q8_K_ACT_BLOCK_BYTES : 0;

    const moeWorkerScratchLayout =
      moeExpertCount > 0 ? computeMoeWorkerScratchLayout(dim, ffnDim, 1) : null;
    const moeBatch4WorkerScratchLayout =
      moeExpertCount > 0 ? computeMoeWorkerScratchLayout(dim, ffnDim, B) : null;
    const moeWorkerScratchSlots =
      moeExpertCount > 0
        ? Math.max(1, Math.min(16, this.backend.activeThreadPool?.workerCount ?? 1))
        : 0;
    const moeWorkerScratchBytes =
      moeBatch4WorkerScratchLayout !== null
        ? moeBatch4WorkerScratchLayout.strideBytes * moeWorkerScratchSlots
        : 0;

    const scratchCapacity =
      alignUp32(dim * 4) * 5 +
      alignUp32(qDim * 4) * 2 +
      alignUp32(kvDim * 4) * 2 +
      alignUp32(ffnDim * 4) * 3 +
      alignUp32(vocab * 4) +
      (moeExpertCount > 0 ? alignUp32(moeExpertCount * 4) + alignUp32(moeWorkerScratchBytes) + 128 : 0) +
      alignUp32(dim * B * 4) * 4 +
      alignUp32(qDim * B * 4) * 2 +
      alignUp32(kvDim * B * 4) * 2 +
      alignUp32(ffnDim * B * 4) * 3 +
      alignUp32(q8DimBytes) * (B + 2) +
      alignUp32(q8AttnOutBytes) * (B + 2) +
      alignUp32(q8FfnBytes) * (B + 2) +
      alignUp32(q8kDimBytes) * (B + 2) +
      alignUp32(q8kAttnOutBytes) * (B + 2) +
      alignUp32(q8kFfnBytes) * (B + 2) +
      4096;

    const computeBuffer = this.backend.allocBuffer(
      'vtensor.compute_scratch',
      'compute',
      scratchCapacity,
    );

    const scratch: ScratchTensors = {
      x: computeBuffer.allocTensor('act.x', GgmlType.F32, [dim]),
      xNorm: computeBuffer.allocTensor('act.x_norm', GgmlType.F32, [dim]),
      q: computeBuffer.allocTensor('act.q', GgmlType.F32, [qDim]),
      k: computeBuffer.allocTensor('act.k', GgmlType.F32, [kvDim]),
      v: computeBuffer.allocTensor('act.v', GgmlType.F32, [kvDim]),
      attnOut: computeBuffer.allocTensor('act.attn_out', GgmlType.F32, [qDim]),
      attnProj: computeBuffer.allocTensor('act.attn_proj', GgmlType.F32, [dim]),
      ffnGate: computeBuffer.allocTensor('act.ffn_gate', GgmlType.F32, [ffnDim]),
      ffnUp: computeBuffer.allocTensor('act.ffn_up', GgmlType.F32, [ffnDim]),
      ffnHidden: computeBuffer.allocTensor('act.ffn_hidden', GgmlType.F32, [ffnDim]),
      ffnProj: computeBuffer.allocTensor('act.ffn_proj', GgmlType.F32, [dim]),
      logits: computeBuffer.allocTensor('act.logits', GgmlType.F32, [vocab]),
      moeRouterLogits:
        moeExpertCount > 0
          ? computeBuffer.allocTensor('act.moe_router', GgmlType.F32, [moeExpertCount])
          : null,
      moeExpertProj:
        moeExpertCount > 0
          ? computeBuffer.allocTensor('act.moe_exp_proj', GgmlType.F32, [dim])
          : null,
      moeWorkerScratch:
        moeWorkerScratchBytes > 0
          ? computeBuffer.allocTensor('act.moe_worker_scratch', GgmlType.I8, [moeWorkerScratchBytes])
          : null,
      moeWorkerScratchLayout,
      moeBatch4WorkerScratchLayout,
      moeWorkerScratchSlots,
      q8Dim: q8DimBytes > 0 ? computeBuffer.allocTensor('act.q8_dim', GgmlType.I8, [q8DimBytes]) : null,
      q8AttnOut:
        q8AttnOutBytes > 0
          ? computeBuffer.allocTensor('act.q8_attn_out', GgmlType.I8, [q8AttnOutBytes])
          : null,
      q8Ffn: q8FfnBytes > 0 ? computeBuffer.allocTensor('act.q8_ffn', GgmlType.I8, [q8FfnBytes]) : null,
      q8kDim:
        q8kDimBytes > 0 ? computeBuffer.allocTensor('act.q8k_dim', GgmlType.I8, [q8kDimBytes]) : null,
      q8kAttnOut:
        q8kAttnOutBytes > 0
          ? computeBuffer.allocTensor('act.q8k_attn_out', GgmlType.I8, [q8kAttnOutBytes])
          : null,
      q8kFfn:
        q8kFfnBytes > 0 ? computeBuffer.allocTensor('act.q8k_ffn', GgmlType.I8, [q8kFfnBytes]) : null,
    };

    const batchScratch: Batch4ScratchTensors = {
      x: computeBuffer.allocTensor('batch.x', GgmlType.F32, [dim * B]),
      xNorm: computeBuffer.allocTensor('batch.x_norm', GgmlType.F32, [dim * B]),
      q: computeBuffer.allocTensor('batch.q', GgmlType.F32, [qDim * B]),
      k: computeBuffer.allocTensor('batch.k', GgmlType.F32, [kvDim * B]),
      v: computeBuffer.allocTensor('batch.v', GgmlType.F32, [kvDim * B]),
      attnOut: computeBuffer.allocTensor('batch.attn_out', GgmlType.F32, [qDim * B]),
      attnProj: computeBuffer.allocTensor('batch.attn_proj', GgmlType.F32, [dim * B]),
      ffnGate: computeBuffer.allocTensor('batch.ffn_gate', GgmlType.F32, [ffnDim * B]),
      ffnUp: computeBuffer.allocTensor('batch.ffn_up', GgmlType.F32, [ffnDim * B]),
      ffnHidden: computeBuffer.allocTensor('batch.ffn_hidden', GgmlType.F32, [ffnDim * B]),
      ffnProj: computeBuffer.allocTensor('batch.ffn_proj', GgmlType.F32, [dim * B]),
      q8Dim:
        q8DimBytes > 0
          ? computeBuffer.allocTensor('batch.q8_dim', GgmlType.I8, [q8DimBytes * B])
          : null,
      q8AttnOut:
        q8AttnOutBytes > 0
          ? computeBuffer.allocTensor('batch.q8_attn_out', GgmlType.I8, [q8AttnOutBytes * B])
          : null,
      q8Ffn:
        q8FfnBytes > 0
          ? computeBuffer.allocTensor('batch.q8_ffn', GgmlType.I8, [q8FfnBytes * B])
          : null,
      q8kDim:
        q8kDimBytes > 0
          ? computeBuffer.allocTensor('batch.q8k_dim', GgmlType.I8, [q8kDimBytes * B])
          : null,
      q8kAttnOut:
        q8kAttnOutBytes > 0
          ? computeBuffer.allocTensor('batch.q8k_attn_out', GgmlType.I8, [q8kAttnOutBytes * B])
          : null,
      q8kFfn:
        q8kFfnBytes > 0
          ? computeBuffer.allocTensor('batch.q8k_ffn', GgmlType.I8, [q8kFfnBytes * B])
          : null,
    };

    // 4. Precompute bounded RoPE cos/sin LUT (<= 4096 rows) + PREFILL_BATCH_SIZE dynamic scratch slots
    //    so arbitrary full contexts (8K–131K+) never OOM WASM memory while remaining bit-exact.
    //    Reuses OPFS `<fp>.rope.bin` directly on reload (0 trig evaluations) when geometry matches!
    const ropeLutMaxPos = Math.min(maxCtx, 4096);
    const cachedRope =
      this.preloadedRopeTables &&
      this.preloadedRopeParams &&
      this.preloadedRopeParams.maxCtx === ropeLutMaxPos &&
      this.preloadedRopeParams.ropeDim === hp.ropeDimensionCount &&
      this.preloadedRopeParams.freqBase === hp.ropeFreqBase &&
      this.preloadedRopeParams.freqScale === hp.ropeFreqScale
        ? this.preloadedRopeTables
        : null;
    const ropeTables =
      cachedRope ??
      buildRopeTables(
        ropeLutMaxPos,
        hp.ropeDimensionCount,
        hp.ropeFreqBase,
        hp.ropeFreqScale,
      );
    const ropeTotalFloats = (ropeLutMaxPos + PREFILL_BATCH_SIZE) * ropeTables.half;
    const ropeTableBytes = alignUp32(ropeTotalFloats * 4);
    const ropeBuf = this.backend.allocBuffer(
      'vtensor.rope_lut',
      'compute',
      ropeTableBytes * 2 + 128,
    );
    const ropeCosHandle = ropeBuf.allocTensor('rope.cos', GgmlType.F32, [ropeTotalFloats]);
    const ropeSinHandle = ropeBuf.allocTensor('rope.sin', GgmlType.F32, [ropeTotalFloats]);
    ropeBuf.f32View(ropeCosHandle).set(ropeTables.cos, 0);
    ropeBuf.f32View(ropeSinHandle).set(ropeTables.sin, 0);

    this.weightPager = weightPager;
    this.kvPager = kvPager;
    this.cachedTokens = [];
    this.ropeTables = ropeTables;
    this.ropeCosHandle = ropeCosHandle;
    this.ropeSinHandle = ropeSinHandle;
    this.scratch = scratch;
    this.batchScratch = batchScratch;
    this.maxContextWindow = maxCtx;
    this.performanceContract = contract;
    this.backend.attachPerformanceContract(() => this.performanceContract);
    this.backend.attachPrismTelemetry(() => (this.prismSchedule ? this.prismLedger.snapshot() : null));

    if (this.modelSummary) {
      this.modelSummary = {
        ...this.modelSummary,
        maxContextWindow: maxCtx,
        backingKind: backing.kind,
      };
    }

    // 5. Compile reusable Prism-admitted single-token & batch-4 vTensor execution graphs ONCE
    this.buildCompiledGraphs();
  }

  /**
   * Builds and Prism-admits reusable single-token and batched-4 execution graphs once at load time
   * (or when toggling precision mode or OPFS weight residency).
   * - In `'resident-all'` mode: binds static WASM weight pointers directly (0% indirection overhead).
   * - In `'opfs-paged'` mode: emits ONE `vram_dimm_acquire` node per layer (`afflom/vRAM` commit
   *   `3f27a941`), streaming all 7 projections (and all `B* = 4` batched prefill tokens) from a
   *   single committed DIMM publication, plus single-row OPFS embeddings and row-striped LM-head!
   */
  private buildCompiledGraphs(): void {
    const hp = this.hparams!;
    const wp = this.weightPager!;
    const s = this.scratch!;
    const bs = this.batchScratch!;
    const kv = this.kvPager!;
    const rope = this.ropeTables!;
    const b = this.backend;
    const arena = b.arena;
    const headDim = hp.headDim;
    const dim = hp.embeddingLength;
    const qDim = hp.headCount * headDim;
    const kvDim = hp.headCountKv * headDim;
    const ffnDim = Math.max(hp.feedForwardLength, hp.expertFeedForwardLength ?? 0);
    const useNeoxRope = hp.ropeType === 'neox';
    const moeTopK = Math.max(1, hp.expertUsedCount ?? 2);
    const moeNormalizeWeights = hp.expertWeightsNorm !== false;
    const moeTopIds = new Int32Array(moeTopK);
    const moeTopProbs = new Float64Array(moeTopK);
    const moeSoftmaxBuf = new Float64Array(Math.max(1, this.modelSpec?.expertCount ?? hp.expertCount ?? 1));
    // Preallocated mutable 2-entry tuple so Sparse MoE dispatches create 0 JS heap objects per expert
    const moeGateUpGroupEntries: Array<{
      weightType: GgmlType;
      wPtr: number;
      outPtr: number;
      rows: number;
      oStrideBytes: number;
    }> = [
      { weightType: GgmlType.F32, wPtr: 0, outPtr: 0, rows: 0, oStrideBytes: 0 },
      { weightType: GgmlType.F32, wPtr: 0, outPtr: 0, rows: 0, oStrideBytes: 0 },
    ];
    // Preallocated zero-alloc expert tables for single-token & Batch-4 Sparse MoE dispatch
    const moeSingleGatePtrs = new Int32Array(moeTopK);
    const moeSingleUpPtrs = new Int32Array(moeTopK);
    const moeSingleDownPtrs = new Int32Array(moeTopK);
    const moeSingleWeightsF32 = new Float32Array(moeTopK);
    const moeBatchMaxUnique = PREFILL_BATCH_SIZE * moeTopK;
    const moeBatchExpIds = new Int32Array(moeBatchMaxUnique);
    const moeBatchExpMask = new Uint8Array(moeBatchMaxUnique);
    const moeBatchExpWeights = new Float64Array(moeBatchMaxUnique * PREFILL_BATCH_SIZE);
    const moeBatchSparseIdx = new Int32Array(moeBatchMaxUnique);
    const moeBatchGatePtrs = new Int32Array(moeBatchMaxUnique);
    const moeBatchUpPtrs = new Int32Array(moeBatchMaxUnique);
    const moeBatchDownPtrs = new Int32Array(moeBatchMaxUnique);
    const moeBatchSparseMasks = new Uint8Array(moeBatchMaxUnique);
    const moeBatchWeightsF32 = new Float32Array(moeBatchMaxUnique * PREFILL_BATCH_SIZE);

    const routeTopKFromLogits = (rLogits: Float32Array, nExperts: number, kUsed: number): void => {
      let maxL = Number.NEGATIVE_INFINITY;
      for (let i = 0; i < nExperts; i++) {
        const v = rLogits[i];
        if (v > maxL) maxL = v;
      }
      let sumExp = 0;
      for (let i = 0; i < nExperts; i++) {
        const p = Math.exp(rLogits[i] - maxL);
        moeSoftmaxBuf[i] = p;
        sumExp += p;
      }
      const invSum = sumExp > 0 ? 1 / sumExp : 1 / nExperts;
      for (let i = 0; i < nExperts; i++) {
        moeSoftmaxBuf[i] *= invSum;
      }

      for (let k = 0; k < kUsed; k++) {
        moeTopIds[k] = -1;
        moeTopProbs[k] = -1;
      }
      for (let i = 0; i < nExperts; i++) {
        const p = moeSoftmaxBuf[i];
        if (p > moeTopProbs[kUsed - 1]) {
          let pos = kUsed - 1;
          while (pos > 0 && p > moeTopProbs[pos - 1]) {
            moeTopProbs[pos] = moeTopProbs[pos - 1];
            moeTopIds[pos] = moeTopIds[pos - 1];
            pos -= 1;
          }
          moeTopProbs[pos] = p;
          moeTopIds[pos] = i;
        }
      }
      if (moeNormalizeWeights) {
        let topSum = 0;
        for (let k = 0; k < kUsed; k++) topSum += moeTopProbs[k];
        if (topSum > 0) {
          for (let k = 0; k < kUsed; k++) moeTopProbs[k] /= topSum;
        }
      }
    };

    const routeSparseMoeSingle = (
      routerWeight: VTensorHandle,
      xNormHandle: VTensorHandle,
      xNormOffsetFloats: number,
    ): number => {
      const nExperts = routerWeight.dims[1] ?? 1;
      const kUsed = Math.min(moeTopK, nExperts);
      const routerLogits = s.moeRouterLogits!;

      // 1. Router projection: routerLogits[nExperts] = routerWeight[nExperts x dim] . xNorm[dim]
      const xInView = xNormHandle.buffer.f32View(xNormHandle);
      const xSlice = xNormOffsetFloats === 0 ? xInView : xInView.subarray(xNormOffsetFloats, xNormOffsetFloats + dim);
      if (xNormOffsetFloats !== 0) {
        s.xNorm.buffer.f32View(s.xNorm).set(xSlice, 0);
      }
      const xActHandle = xNormOffsetFloats === 0 ? xNormHandle : s.xNorm;
      const useRouterWasm =
        arena !== null &&
        xActHandle.buffer.isWasmBacked &&
        routerLogits.buffer.isWasmBacked &&
        routerWeight.buffer.isWasmBacked;

      if (useRouterWasm && arena) {
        arena.mulMat(
          routerWeight.type,
          routerWeight.buffer.wasmPtr(routerWeight),
          xActHandle.buffer.wasmPtr(xActHandle),
          routerLogits.buffer.wasmPtr(routerLogits),
          dim,
          nExperts,
        );
      } else {
        verdantMulMat(
          routerWeight.buffer.rawBytes(),
          routerWeight.byteOffset,
          routerWeight.type,
          dim,
          nExperts,
          xActHandle.buffer.f32View(xActHandle),
          routerLogits.buffer.f32View(routerLogits),
        );
      }

      // 2. Numerically exact Float64 softmax + deterministic Top-K selection
      routeTopKFromLogits(routerLogits.buffer.f32View(routerLogits), nExperts, kUsed);
      return kUsed;
    };

    const executeSparseMoeSingleRouted = (
      layerIdx: number,
      xNormHandle: VTensorHandle,
      xNormOffsetFloats: number,
      q8DimHandle: VTensorHandle | null,
      q8kDimHandle: VTensorHandle | null,
      outProjHandle: VTensorHandle,
      outProjOffsetFloats: number,
      expFfnDim: number,
      kUsed: number,
    ): void => {
      const expertProj = s.moeExpertProj!;
      const xActHandle = xNormOffsetFloats === 0 ? xNormHandle : s.xNorm;
      const useWasm =
        arena !== null &&
        xActHandle.buffer.isWasmBacked &&
        s.ffnGate.buffer.isWasmBacked &&
        s.ffnUp.buffer.isWasmBacked &&
        s.ffnHidden.buffer.isWasmBacked &&
        expertProj.buffer.isWasmBacked;

      const xPtr = useWasm ? xActHandle.buffer.wasmPtr(xActHandle) : -1;
      const gateOutPtr = useWasm ? s.ffnGate.buffer.wasmPtr(s.ffnGate) : -1;
      const upOutPtr = useWasm ? s.ffnUp.buffer.wasmPtr(s.ffnUp) : -1;
      const hidPtr = useWasm ? s.ffnHidden.buffer.wasmPtr(s.ffnHidden) : -1;
      const expProjPtr = useWasm ? expertProj.buffer.wasmPtr(expertProj) : -1;

      // Quantize xActHandle ONCE across all K selected experts if gate/up support Q8_K or Q8_0
      const infoL = this.layerInfos[layerIdx];
      const expGateType = infoL.ffnGateExps!.type;
      const expUpType = infoL.ffnUpExps!.type;
      const expDownType = infoL.ffnDownExps!.type;
      const expInQ8K =
        useWasm &&
        arena !== null &&
        q8kDimHandle !== null &&
        q8kDimHandle.buffer.isWasmBacked &&
        canQ8K(expGateType, dim, q8kDimHandle) &&
        canQ8K(expUpType, dim, q8kDimHandle);
      const expInQ8 =
        !expInQ8K &&
        useWasm &&
        useQ8 &&
        arena !== null &&
        q8DimHandle !== null &&
        q8DimHandle.buffer.isWasmBacked &&
        (arena.supportsQ8Activation(expGateType) || arena.supportsQ8Activation(expUpType));
      const qkDimPtr = expInQ8K && q8kDimHandle ? q8kDimHandle.buffer.wasmPtr(q8kDimHandle) : -1;
      const qDimPtr = expInQ8 && q8DimHandle ? q8DimHandle.buffer.wasmPtr(q8DimHandle) : -1;

      if (expInQ8K && arena) {
        arena.quantizeActivationQ8K(xPtr, qkDimPtr, dim);
      } else if (expInQ8 && arena) {
        arena.quantizeActivation(xPtr, qDimPtr, dim);
      }

      const expDownQ8K =
        useWasm &&
        arena !== null &&
        s.q8kFfn !== null &&
        s.q8kFfn.buffer.isWasmBacked &&
        canQ8K(expDownType, expFfnDim, s.q8kFfn);
      const expDownQ8 =
        !expDownQ8K &&
        useWasm &&
        useQ8 &&
        arena !== null &&
        s.q8Ffn !== null &&
        s.q8Ffn.buffer.isWasmBacked &&
        arena.supportsQ8Activation(expDownType);
      const qkFfnPtr = expDownQ8K && s.q8kFfn ? s.q8kFfn.buffer.wasmPtr(s.q8kFfn) : -1;
      const qFfnPtr = expDownQ8 && s.q8Ffn ? s.q8Ffn.buffer.wasmPtr(s.q8Ffn) : -1;

      // 3. Single-barrier Expert-Parallel MoE fast path (`OP_MOE_EXPERTS`) when thread pool is active.
      //    Chunks into conflict-free waves of `<= wp.moeExpertCacheCapacity` experts so WASM
      //    paged expert cache slots can NEVER be evicted and overwritten while queued in parallel!
      const outBasePtr = useWasm && outProjHandle.buffer.isWasmBacked
        ? outProjHandle.buffer.wasmPtr(outProjHandle) + outProjOffsetFloats * 4
        : -1;
      const pool = this.backend.activeThreadPool;
      const maxSingleWave = Math.min(16, Math.max(1, wp.moeExpertCacheCapacity));
      if (
        useWasm &&
        arena &&
        outBasePtr >= 0 &&
        pool &&
        pool.isActive &&
        kUsed >= 2 &&
        maxSingleWave >= 2 &&
        s.moeWorkerScratch &&
        s.moeWorkerScratchLayout &&
        s.moeWorkerScratch.buffer.isWasmBacked &&
        s.moeWorkerScratchSlots >= Math.min(pool.workerCount, Math.min(kUsed, maxSingleWave))
      ) {
        let waveCount = 0;
        let dispatchedWaves = 0;
        let allHomogeneousWasm = true;

        const flushSingleWave = (): boolean => {
          if (waveCount === 0) return true;
          if (waveCount >= 2) {
            const ok = arena.dispatchMoeExpertsParallel({
              expertCount: waveCount,
              dim,
              expFfnDim,
              gateUpOp: expInQ8K
                ? ParallelMatOp.MUL_MAT_Q8K
                : expInQ8
                  ? ParallelMatOp.MUL_MAT_Q8
                  : ParallelMatOp.MUL_MAT,
              downOp: expDownQ8K
                ? ParallelMatOp.MUL_MAT_Q8K
                : expDownQ8
                  ? ParallelMatOp.MUL_MAT_Q8
                  : ParallelMatOp.MUL_MAT,
              gateType: expGateType,
              upType: expUpType,
              downType: expDownType,
              xOrQ8Ptr: expInQ8K ? qkDimPtr : expInQ8 ? qDimPtr : xPtr,
              xF32Ptr: xPtr,
              outPtr: outBasePtr,
              workerScratchBase: s.moeWorkerScratch!.buffer.wasmPtr(s.moeWorkerScratch!),
              layout: s.moeWorkerScratchLayout!,
              gatePtrs: moeSingleGatePtrs,
              upPtrs: moeSingleUpPtrs,
              downPtrs: moeSingleDownPtrs,
              weights: moeSingleWeightsF32,
              accumulate: dispatchedWaves > 0,
            });
            if (!ok) return false;
          } else {
            // Tail wave of 1 expert: row-stripe across all P cores via mulMatQ8KGroup/mulMatQ8Group
            if (dispatchedWaves === 0) {
              arena.kernels.zero_bytes(outBasePtr, dim * 4);
            }
            const g0 = moeGateUpGroupEntries[0];
            const g1 = moeGateUpGroupEntries[1];
            g0.weightType = expGateType;
            g0.wPtr = moeSingleGatePtrs[0];
            g0.outPtr = gateOutPtr;
            g0.rows = expFfnDim;
            g0.oStrideBytes = expFfnDim * 4;
            g1.weightType = expUpType;
            g1.wPtr = moeSingleUpPtrs[0];
            g1.outPtr = upOutPtr;
            g1.rows = expFfnDim;
            g1.oStrideBytes = expFfnDim * 4;
            const dPtr = moeSingleDownPtrs[0];
            if (expInQ8K) {
              arena.mulMatQ8KGroup(moeGateUpGroupEntries, qkDimPtr, xPtr, dim);
            } else {
              arena.mulMatQ8Group(moeGateUpGroupEntries, expInQ8 ? qDimPtr : xPtr, xPtr, dim, expInQ8);
            }
            arena.kernels.swiglu_f32(gateOutPtr, upOutPtr, hidPtr, expFfnDim);
            if (expDownQ8K) {
              arena.quantizeActivationQ8K(hidPtr, qkFfnPtr, expFfnDim);
              arena.mulMatQ8K(expDownType, dPtr, qkFfnPtr, hidPtr, expProjPtr, expFfnDim, dim);
            } else if (expDownQ8) {
              arena.quantizeActivation(hidPtr, qFfnPtr, expFfnDim);
              arena.mulMatQ8(expDownType, dPtr, qFfnPtr, hidPtr, expProjPtr, expFfnDim, dim);
            } else {
              arena.mulMat(expDownType, dPtr, hidPtr, expProjPtr, expFfnDim, dim);
            }
            arena.kernels.axpy_f32(outBasePtr, expProjPtr, moeSingleWeightsF32[0], dim);
          }
          dispatchedWaves += 1;
          waveCount = 0;
          return true;
        };

        for (let k = 0; k < kUsed; k++) {
          const expId = moeTopIds[k];
          const weight = moeTopProbs[k];
          if (expId < 0 || weight === 0) continue;
          const exp = wp.acquireExpertSlice(layerIdx, expId);
          if (
            !exp.gate.buffer.isWasmBacked ||
            !exp.up.buffer.isWasmBacked ||
            !exp.down.buffer.isWasmBacked ||
            exp.gate.type !== expGateType ||
            exp.up.type !== expUpType ||
            exp.down.type !== expDownType
          ) {
            allHomogeneousWasm = false;
            break;
          }
          moeSingleGatePtrs[waveCount] = exp.gate.buffer.wasmPtr(exp.gate);
          moeSingleUpPtrs[waveCount] = exp.up.buffer.wasmPtr(exp.up);
          moeSingleDownPtrs[waveCount] = exp.down.buffer.wasmPtr(exp.down);
          moeSingleWeightsF32[waveCount] = weight;
          waveCount += 1;
          if (waveCount === maxSingleWave) {
            if (!flushSingleWave()) {
              allHomogeneousWasm = false;
              break;
            }
          }
        }
        if (allHomogeneousWasm && (waveCount > 0 || dispatchedWaves > 0)) {
          if (waveCount === 0 || (dispatchedWaves > 0 ? flushSingleWave() : waveCount >= 2 && flushSingleWave())) {
            return;
          }
        }
      }

      // Single-thread / fallback path with WASM SIMD zero_bytes + axpy_f32 accumulation
      if (useWasm && arena && outBasePtr >= 0) {
        arena.kernels.zero_bytes(outBasePtr, dim * 4);
      } else {
        const outAll = outProjHandle.buffer.f32View(outProjHandle);
        const outSlice =
          outProjOffsetFloats === 0
            ? outAll.subarray(0, dim)
            : outAll.subarray(outProjOffsetFloats, outProjOffsetFloats + dim);
        outSlice.fill(0);
      }

      const ge0 = moeGateUpGroupEntries[0];
      const ge1 = moeGateUpGroupEntries[1];
      ge0.outPtr = gateOutPtr;
      ge0.rows = expFfnDim;
      ge0.oStrideBytes = expFfnDim * 4;
      ge1.outPtr = upOutPtr;
      ge1.rows = expFfnDim;
      ge1.oStrideBytes = expFfnDim * 4;

      for (let k = 0; k < kUsed; k++) {
        const expId = moeTopIds[k];
        const weight = moeTopProbs[k];
        if (expId < 0 || weight === 0) continue;

        const exp = wp.acquireExpertSlice(layerIdx, expId);
        if (useWasm && arena && exp.gate.buffer.isWasmBacked && exp.up.buffer.isWasmBacked && exp.down.buffer.isWasmBacked) {
          ge0.weightType = exp.gate.type;
          ge0.wPtr = exp.gate.buffer.wasmPtr(exp.gate);
          ge1.weightType = exp.up.type;
          ge1.wPtr = exp.up.buffer.wasmPtr(exp.up);
          const dPtr = exp.down.buffer.wasmPtr(exp.down);
          if (expInQ8K) {
            arena.mulMatQ8KGroup(moeGateUpGroupEntries, qkDimPtr, xPtr, dim);
          } else {
            arena.mulMatQ8Group(moeGateUpGroupEntries, expInQ8 ? qDimPtr : xPtr, xPtr, dim, expInQ8);
          }
          arena.kernels.swiglu_f32(gateOutPtr, upOutPtr, hidPtr, expFfnDim);
          if (expDownQ8K) {
            arena.quantizeActivationQ8K(hidPtr, qkFfnPtr, expFfnDim);
            arena.mulMatQ8K(exp.down.type, dPtr, qkFfnPtr, hidPtr, expProjPtr, expFfnDim, dim);
          } else if (expDownQ8) {
            arena.quantizeActivation(hidPtr, qFfnPtr, expFfnDim);
            arena.mulMatQ8(exp.down.type, dPtr, qFfnPtr, hidPtr, expProjPtr, expFfnDim, dim);
          } else {
            arena.mulMat(exp.down.type, dPtr, hidPtr, expProjPtr, expFfnDim, dim);
          }
          if (outBasePtr >= 0) {
            arena.kernels.axpy_f32(outBasePtr, expProjPtr, weight, dim);
            continue;
          }
        } else {
          const xView = xActHandle.buffer.f32View(xActHandle);
          const gView = s.ffnGate.buffer.f32View(s.ffnGate).subarray(0, expFfnDim);
          const uView = s.ffnUp.buffer.f32View(s.ffnUp).subarray(0, expFfnDim);
          const hView = s.ffnHidden.buffer.f32View(s.ffnHidden).subarray(0, expFfnDim);
          const eView = expertProj.buffer.f32View(expertProj);
          verdantMulMat(exp.gate.buffer.rawBytes(), exp.gate.byteOffset, exp.gate.type, dim, expFfnDim, xView, gView);
          verdantMulMat(exp.up.buffer.rawBytes(), exp.up.byteOffset, exp.up.type, dim, expFfnDim, xView, uView);
          verdantSwiGlu(gView, uView, hView, expFfnDim);
          verdantMulMat(exp.down.buffer.rawBytes(), exp.down.byteOffset, exp.down.type, expFfnDim, dim, hView, eView);
        }

        const outAll = outProjHandle.buffer.f32View(outProjHandle);
        const outSlice =
          outProjOffsetFloats === 0
            ? outAll.subarray(0, dim)
            : outAll.subarray(outProjOffsetFloats, outProjOffsetFloats + dim);
        const eOut = expertProj.buffer.f32View(expertProj);
        for (let i = 0; i < dim; i++) {
          outSlice[i] += weight * eOut[i];
        }
      }
    };

    const runSparseMoeSingle = (
      layerIdx: number,
      routerWeight: VTensorHandle,
      xNormHandle: VTensorHandle,
      xNormOffsetFloats: number,
      q8DimHandle: VTensorHandle | null,
      q8kDimHandle: VTensorHandle | null,
      outProjHandle: VTensorHandle,
      outProjOffsetFloats: number,
      expFfnDim: number,
    ): void => {
      const kUsed = routeSparseMoeSingle(routerWeight, xNormHandle, xNormOffsetFloats);
      executeSparseMoeSingleRouted(
        layerIdx,
        xNormHandle,
        xNormOffsetFloats,
        q8DimHandle,
        q8kDimHandle,
        outProjHandle,
        outProjOffsetFloats,
        expFfnDim,
        kUsed,
      );
    };

    const routeSparseMoeBatch4 = (routerWeight: VTensorHandle): number => {
      const B4 = PREFILL_BATCH_SIZE;
      const nExperts = routerWeight.dims[1] ?? 1;
      const kUsed = Math.min(moeTopK, nExperts);
      const routerLogits = s.moeRouterLogits!;
      const useRouterWasm =
        arena !== null &&
        bs.xNorm.buffer.isWasmBacked &&
        routerLogits.buffer.isWasmBacked &&
        routerWeight.buffer.isWasmBacked;

      const bXNormPtr = useRouterWasm ? bs.xNorm.buffer.wasmPtr(bs.xNorm) : -1;
      const routerOutPtr = useRouterWasm ? routerLogits.buffer.wasmPtr(routerLogits) : -1;
      const xStrideBytes = dim * 4;
      const xAll = bs.xNorm.buffer.f32View(bs.xNorm);
      const rLogitsView = routerLogits.buffer.f32View(routerLogits);

      // 1. Route all 4 tokens and group by unique expertId (zero JS allocations)
      let uniqueCount = 0;
      for (let t = 0; t < B4; t++) {
        if (useRouterWasm && arena) {
          arena.mulMat(
            routerWeight.type,
            routerWeight.buffer.wasmPtr(routerWeight),
            bXNormPtr + t * xStrideBytes,
            routerOutPtr,
            dim,
            nExperts,
          );
        } else {
          verdantMulMat(
            routerWeight.buffer.rawBytes(),
            routerWeight.byteOffset,
            routerWeight.type,
            dim,
            nExperts,
            xAll.subarray(t * dim, (t + 1) * dim),
            rLogitsView,
          );
        }
        routeTopKFromLogits(rLogitsView, nExperts, kUsed);
        for (let k = 0; k < kUsed; k++) {
          const expId = moeTopIds[k];
          const weight = moeTopProbs[k];
          if (expId < 0 || weight === 0) continue;
          let u = -1;
          for (let j = 0; j < uniqueCount; j++) {
            if (moeBatchExpIds[j] === expId) {
              u = j;
              break;
            }
          }
          if (u < 0) {
            u = uniqueCount++;
            moeBatchExpIds[u] = expId;
            moeBatchExpMask[u] = 0;
            const wBase = u * B4;
            moeBatchExpWeights[wBase] = 0;
            moeBatchExpWeights[wBase + 1] = 0;
            moeBatchExpWeights[wBase + 2] = 0;
            moeBatchExpWeights[wBase + 3] = 0;
          }
          moeBatchExpMask[u] |= 1 << t;
          moeBatchExpWeights[u * B4 + t] = weight;
        }
      }
      return uniqueCount;
    };

    /**
     * Expert-Grouped Batch-4 Sparse MoE prefill kernel:
     * Executes all `uniqueCount` routed experts populated by `routeSparseMoeBatch4()`.
     */
    const executeSparseMoeBatch4Routed = (
      layerIdx: number,
      expFfnDim: number,
      uniqueCount: number,
    ): void => {
      const B4 = PREFILL_BATCH_SIZE;
      const expertProj = s.moeExpertProj!;
      const useWasm =
        arena !== null &&
        bs.xNorm.buffer.isWasmBacked &&
        bs.ffnGate.buffer.isWasmBacked &&
        bs.ffnUp.buffer.isWasmBacked &&
        bs.ffnHidden.buffer.isWasmBacked &&
        bs.attnProj.buffer.isWasmBacked &&
        s.ffnGate.buffer.isWasmBacked &&
        s.ffnUp.buffer.isWasmBacked &&
        s.ffnHidden.buffer.isWasmBacked &&
        expertProj.buffer.isWasmBacked;

      const bXNormPtr = useWasm ? bs.xNorm.buffer.wasmPtr(bs.xNorm) : -1;
      const xStrideBytes = dim * 4;
      const xAll = bs.xNorm.buffer.f32View(bs.xNorm);

      // 2. Quantize all 4 token activations ONCE into bs.q8kDim or bs.q8Dim
      const infoL = this.layerInfos[layerIdx];
      const expGateType = infoL.ffnGateExps!.type;
      const expUpType = infoL.ffnUpExps!.type;
      const expDownType = infoL.ffnDownExps!.type;

      const expInQ8K =
        useWasm &&
        arena !== null &&
        bs.q8kDim !== null &&
        bs.q8kDim.buffer.isWasmBacked &&
        canQ8K(expGateType, dim, bs.q8kDim) &&
        canQ8K(expUpType, dim, bs.q8kDim);
      const expInQ8 =
        !expInQ8K &&
        useWasm &&
        useQ8 &&
        arena !== null &&
        bs.q8Dim !== null &&
        bs.q8Dim.buffer.isWasmBacked &&
        (arena.supportsQ8Activation(expGateType) || arena.supportsQ8Activation(expUpType));
      const bQkDimPtr = expInQ8K && bs.q8kDim ? bs.q8kDim.buffer.wasmPtr(bs.q8kDim) : -1;
      const bQDimPtr = expInQ8 && bs.q8Dim ? bs.q8Dim.buffer.wasmPtr(bs.q8Dim) : -1;
      const qkDimStride = (dim >>> 8) * Q8_K_ACT_BLOCK_BYTES;
      const qDimStride = (dim >>> 5) * Q8_0_ACT_BLOCK_BYTES;

      if (expInQ8K && arena) {
        for (let t = 0; t < B4; t++) {
          arena.quantizeActivationQ8K(bXNormPtr + t * xStrideBytes, bQkDimPtr + t * qkDimStride, dim);
        }
      } else if (expInQ8 && arena) {
        for (let t = 0; t < B4; t++) {
          arena.quantizeActivation(bXNormPtr + t * xStrideBytes, bQDimPtr + t * qDimStride, dim);
        }
      }

      const expDownQ8K =
        useWasm &&
        arena !== null &&
        s.q8kFfn !== null &&
        s.q8kFfn.buffer.isWasmBacked &&
        canQ8K(expDownType, expFfnDim, s.q8kFfn);
      const expDownQ8 =
        !expDownQ8K &&
        useWasm &&
        useQ8 &&
        arena !== null &&
        s.q8Ffn !== null &&
        s.q8Ffn.buffer.isWasmBacked &&
        arena.supportsQ8Activation(expDownType);
      const qkFfnPtr = expDownQ8K && s.q8kFfn ? s.q8kFfn.buffer.wasmPtr(s.q8kFfn) : -1;
      const qFfnPtr = expDownQ8 && s.q8Ffn ? s.q8Ffn.buffer.wasmPtr(s.q8Ffn) : -1;
      const bQkFfnPtr =
        expDownQ8K && bs.q8kFfn && bs.q8kFfn.buffer.isWasmBacked ? bs.q8kFfn.buffer.wasmPtr(bs.q8kFfn) : -1;
      const bQFfnPtr =
        expDownQ8 && bs.q8Ffn && bs.q8Ffn.buffer.isWasmBacked ? bs.q8Ffn.buffer.wasmPtr(bs.q8Ffn) : -1;
      const qkFfnStride = (expFfnDim >>> 8) * Q8_K_ACT_BLOCK_BYTES;
      const qFfnStride = (expFfnDim >>> 5) * Q8_0_ACT_BLOCK_BYTES;

      const sGateOutPtr = useWasm ? s.ffnGate.buffer.wasmPtr(s.ffnGate) : -1;
      const sUpOutPtr = useWasm ? s.ffnUp.buffer.wasmPtr(s.ffnUp) : -1;
      const sHidPtr = useWasm ? s.ffnHidden.buffer.wasmPtr(s.ffnHidden) : -1;
      const sExpProjPtr = useWasm ? expertProj.buffer.wasmPtr(expertProj) : -1;

      const bGateOutPtr = useWasm ? bs.ffnGate.buffer.wasmPtr(bs.ffnGate) : -1;
      const bUpOutPtr = useWasm ? bs.ffnUp.buffer.wasmPtr(bs.ffnUp) : -1;
      const bHidPtr = useWasm ? bs.ffnHidden.buffer.wasmPtr(bs.ffnHidden) : -1;
      // Reuse bs.attnProj ([B4 x dim], already consumed at attn_residual_b4) for Batch-4 expert down output
      const bTmpProjPtr = useWasm ? bs.attnProj.buffer.wasmPtr(bs.attnProj) : -1;
      const bTmpProjView = bs.attnProj.buffer.f32View(bs.attnProj);

      const bFfnProjPtr = useWasm && bs.ffnProj.buffer.isWasmBacked ? bs.ffnProj.buffer.wasmPtr(bs.ffnProj) : -1;
      const outAll = bs.ffnProj.buffer.f32View(bs.ffnProj);
      if (useWasm && arena && bFfnProjPtr >= 0) {
        arena.kernels.zero_bytes(bFfnProjPtr, B4 * dim * 4);
      } else {
        outAll.fill(0);
      }

      const ge0 = moeGateUpGroupEntries[0];
      const ge1 = moeGateUpGroupEntries[1];
      const eOutView = expertProj.buffer.f32View(expertProj);

      // Hybrid Theorem T3-MoE Batch-4 dispatch:
      // 1) Heavy 4-token hits (`mask === 0b1111`): row-striped Batch-4 across ALL P cores (100% core balance + L3-resident)
      // 2) Light sparse hits (`popcount(mask) ∈ {1,2,3}`): executed in conflict-free waves of
      //    `<= wp.moeExpertCacheCapacity` experts via OP_MOE_EXPERTS_BATCH4 so WASM paged expert
      //    cache slots can NEVER be evicted while queued in parallel!
      const pool = b.activeThreadPool;
      const canPoolSparse =
        useWasm &&
        arena !== null &&
        bFfnProjPtr >= 0 &&
        pool !== null &&
        pool.isActive &&
        s.moeWorkerScratch !== null &&
        s.moeBatch4WorkerScratchLayout !== null &&
        s.moeWorkerScratch.buffer.isWasmBacked;

      let sparseIdxCount = 0;
      let ranDenseHit = false;

      // Pass 1: execute all full 4-token hits (`mask === 0b1111`) row-striped across P cores;
      // defer sparse hits until Pass 2 so dense acquires never evict queued sparse cache slots.
      for (let u = 0; u < uniqueCount; u++) {
        const mask = moeBatchExpMask[u];
        if (mask !== 0b1111) {
          moeBatchSparseIdx[sparseIdxCount++] = u;
          continue;
        }
        const expId = moeBatchExpIds[u];
        const exp = wp.acquireExpertSlice(layerIdx, expId);
        const expWasm =
          useWasm &&
          arena !== null &&
          exp.gate.buffer.isWasmBacked &&
          exp.up.buffer.isWasmBacked &&
          exp.down.buffer.isWasmBacked;

        if (expWasm && arena) {
          ranDenseHit = true;
          const gPtr = exp.gate.buffer.wasmPtr(exp.gate);
          const uPtr = exp.up.buffer.wasmPtr(exp.up);
          const dPtr = exp.down.buffer.wasmPtr(exp.down);
          ge0.weightType = exp.gate.type;
          ge0.wPtr = gPtr;
          ge0.rows = expFfnDim;
          ge0.oStrideBytes = expFfnDim * 4;
          ge0.outPtr = bGateOutPtr;
          ge1.weightType = exp.up.type;
          ge1.wPtr = uPtr;
          ge1.rows = expFfnDim;
          ge1.oStrideBytes = expFfnDim * 4;
          ge1.outPtr = bUpOutPtr;

          if (expInQ8K) {
            arena.mulMatBatch4Q8KGroup(moeGateUpGroupEntries, bQkDimPtr, bXNormPtr, dim);
          } else {
            arena.mulMatBatch4Q8Group(moeGateUpGroupEntries, expInQ8 ? bQDimPtr : bXNormPtr, bXNormPtr, dim, expInQ8);
          }
          arena.kernels.swiglu_f32(bGateOutPtr, bUpOutPtr, bHidPtr, B4 * expFfnDim);
          if (expDownQ8K && bQkFfnPtr >= 0) {
            for (let t = 0; t < B4; t++) {
              arena.quantizeActivationQ8K(bHidPtr + t * expFfnDim * 4, bQkFfnPtr + t * qkFfnStride, expFfnDim);
            }
            arena.mulMatBatchQ8K(exp.down.type, dPtr, bQkFfnPtr, bHidPtr, bTmpProjPtr, expFfnDim, dim, B4);
          } else if (expDownQ8 && bQFfnPtr >= 0) {
            for (let t = 0; t < B4; t++) {
              arena.quantizeActivation(bHidPtr + t * expFfnDim * 4, bQFfnPtr + t * qFfnStride, expFfnDim);
            }
            arena.mulMatBatchQ8(exp.down.type, dPtr, bQFfnPtr, bHidPtr, bTmpProjPtr, expFfnDim, dim, B4);
          } else {
            arena.mulMatBatch(exp.down.type, dPtr, bHidPtr, bTmpProjPtr, expFfnDim, dim, B4);
          }
          for (let t = 0; t < B4; t++) {
            const weight = moeBatchExpWeights[u * B4 + t];
            if (bFfnProjPtr >= 0) {
              arena.kernels.axpy_f32(bFfnProjPtr + t * xStrideBytes, bTmpProjPtr + t * xStrideBytes, weight, dim);
            } else {
              const base = t * dim;
              for (let i = 0; i < dim; i++) {
                outAll[base + i] += weight * bTmpProjView[base + i];
              }
            }
          }
        } else {
          moeBatchSparseIdx[sparseIdxCount++] = u;
        }
      }

      if (sparseIdxCount === 0) return;

      const runSparseExpertSequential = (u: number): void => {
        const mask = moeBatchExpMask[u];
        const exp = wp.acquireExpertSlice(layerIdx, moeBatchExpIds[u]);
        const expWasm =
          useWasm &&
          arena !== null &&
          exp.gate.buffer.isWasmBacked &&
          exp.up.buffer.isWasmBacked &&
          exp.down.buffer.isWasmBacked;

        if (expWasm && arena) {
          const gPtr = exp.gate.buffer.wasmPtr(exp.gate);
          const uPtr = exp.up.buffer.wasmPtr(exp.up);
          const dPtr = exp.down.buffer.wasmPtr(exp.down);
          ge0.weightType = exp.gate.type;
          ge0.wPtr = gPtr;
          ge0.rows = expFfnDim;
          ge0.oStrideBytes = expFfnDim * 4;
          ge0.outPtr = sGateOutPtr;
          ge1.weightType = exp.up.type;
          ge1.wPtr = uPtr;
          ge1.rows = expFfnDim;
          ge1.oStrideBytes = expFfnDim * 4;
          ge1.outPtr = sUpOutPtr;
          for (let t = 0; t < B4; t++) {
            if ((mask & (1 << t)) === 0) continue;
            const weight = moeBatchExpWeights[u * B4 + t];
            const txPtr = bXNormPtr + t * xStrideBytes;
            if (expInQ8K) {
              arena.mulMatQ8KGroup(moeGateUpGroupEntries, bQkDimPtr + t * qkDimStride, txPtr, dim);
            } else {
              arena.mulMatQ8Group(
                moeGateUpGroupEntries,
                expInQ8 ? bQDimPtr + t * qDimStride : txPtr,
                txPtr,
                dim,
                expInQ8,
              );
            }
            arena.kernels.swiglu_f32(sGateOutPtr, sUpOutPtr, sHidPtr, expFfnDim);
            if (expDownQ8K) {
              arena.quantizeActivationQ8K(sHidPtr, qkFfnPtr, expFfnDim);
              arena.mulMatQ8K(exp.down.type, dPtr, qkFfnPtr, sHidPtr, sExpProjPtr, expFfnDim, dim);
            } else if (expDownQ8) {
              arena.quantizeActivation(sHidPtr, qFfnPtr, expFfnDim);
              arena.mulMatQ8(exp.down.type, dPtr, qFfnPtr, sHidPtr, sExpProjPtr, expFfnDim, dim);
            } else {
              arena.mulMat(exp.down.type, dPtr, sHidPtr, sExpProjPtr, expFfnDim, dim);
            }
            if (bFfnProjPtr >= 0) {
              arena.kernels.axpy_f32(bFfnProjPtr + t * xStrideBytes, sExpProjPtr, weight, dim);
            } else {
              const base = t * dim;
              for (let i = 0; i < dim; i++) {
                outAll[base + i] += weight * eOutView[i];
              }
            }
          }
        } else {
          const gView = s.ffnGate.buffer.f32View(s.ffnGate).subarray(0, expFfnDim);
          const uView = s.ffnUp.buffer.f32View(s.ffnUp).subarray(0, expFfnDim);
          const hView = s.ffnHidden.buffer.f32View(s.ffnHidden).subarray(0, expFfnDim);
          for (let t = 0; t < B4; t++) {
            if ((mask & (1 << t)) === 0) continue;
            const weight = moeBatchExpWeights[u * B4 + t];
            const xView = xAll.subarray(t * dim, (t + 1) * dim);
            verdantMulMat(exp.gate.buffer.rawBytes(), exp.gate.byteOffset, exp.gate.type, dim, expFfnDim, xView, gView);
            verdantMulMat(exp.up.buffer.rawBytes(), exp.up.byteOffset, exp.up.type, dim, expFfnDim, xView, uView);
            verdantSwiGlu(gView, uView, hView, expFfnDim);
            verdantMulMat(exp.down.buffer.rawBytes(), exp.down.byteOffset, exp.down.type, expFfnDim, dim, hView, eOutView);
            const base = t * dim;
            for (let i = 0; i < dim; i++) {
              outAll[base + i] += weight * eOutView[i];
            }
          }
        }
      };

      // Pass 2: Execute sparse experts in conflict-free waves of `<= min(32, moeExpertCacheCapacity)`
      const maxBatchWave = Math.min(32, Math.max(1, wp.moeExpertCacheCapacity));
      if (canPoolSparse && maxBatchWave >= 2 && sparseIdxCount >= 2 && arena && pool) {
        let accumulatedOut = ranDenseHit;
        for (let wStart = 0; wStart < sparseIdxCount; wStart += maxBatchWave) {
          const wEnd = Math.min(sparseIdxCount, wStart + maxBatchWave);
          const wTarget = wEnd - wStart;
          if (wTarget < 2 || s.moeWorkerScratchSlots < Math.min(pool.workerCount, wTarget)) {
            for (let wi = wStart; wi < wEnd; wi++) {
              runSparseExpertSequential(moeBatchSparseIdx[wi]);
            }
            accumulatedOut = true;
            continue;
          }

          let waveAllWasm = true;
          let waveSameTypes = true;
          let waveGType: GgmlType = expGateType;
          let waveUType: GgmlType = expUpType;
          let waveDType: GgmlType = expDownType;
          for (let wi = 0; wi < wTarget; wi++) {
            const u = moeBatchSparseIdx[wStart + wi];
            const exp = wp.acquireExpertSlice(layerIdx, moeBatchExpIds[u]);
            if (
              !exp.gate.buffer.isWasmBacked ||
              !exp.up.buffer.isWasmBacked ||
              !exp.down.buffer.isWasmBacked
            ) {
              waveAllWasm = false;
              break;
            }
            if (wi === 0) {
              waveGType = exp.gate.type;
              waveUType = exp.up.type;
              waveDType = exp.down.type;
            } else if (
              exp.gate.type !== waveGType ||
              exp.up.type !== waveUType ||
              exp.down.type !== waveDType
            ) {
              waveSameTypes = false;
            }
            moeBatchGatePtrs[wi] = exp.gate.buffer.wasmPtr(exp.gate);
            moeBatchUpPtrs[wi] = exp.up.buffer.wasmPtr(exp.up);
            moeBatchDownPtrs[wi] = exp.down.buffer.wasmPtr(exp.down);
            moeBatchSparseMasks[wi] = moeBatchExpMask[u];
            const dstW = wi * B4;
            const srcW = u * B4;
            moeBatchWeightsF32[dstW + 0] = moeBatchExpWeights[srcW + 0];
            moeBatchWeightsF32[dstW + 1] = moeBatchExpWeights[srcW + 1];
            moeBatchWeightsF32[dstW + 2] = moeBatchExpWeights[srcW + 2];
            moeBatchWeightsF32[dstW + 3] = moeBatchExpWeights[srcW + 3];
          }

          let dispatched = false;
          if (waveAllWasm && waveSameTypes) {
            dispatched = arena.dispatchMoeExpertsBatch4Parallel({
              uniqueCount: wTarget,
              dim,
              expFfnDim,
              gateUpBatchOp: expInQ8K
                ? ParallelMatOp.MUL_MAT_BATCH4_Q8K
                : expInQ8
                  ? ParallelMatOp.MUL_MAT_BATCH4_Q8
                  : ParallelMatOp.MUL_MAT_BATCH4,
              gateUpSingleOp: expInQ8K
                ? ParallelMatOp.MUL_MAT_Q8K
                : expInQ8
                  ? ParallelMatOp.MUL_MAT_Q8
                  : ParallelMatOp.MUL_MAT,
              downBatchOp: expDownQ8K
                ? ParallelMatOp.MUL_MAT_BATCH4_Q8K
                : expDownQ8
                  ? ParallelMatOp.MUL_MAT_BATCH4_Q8
                  : ParallelMatOp.MUL_MAT_BATCH4,
              downSingleOp: expDownQ8K
                ? ParallelMatOp.MUL_MAT_Q8K
                : expDownQ8
                  ? ParallelMatOp.MUL_MAT_Q8
                  : ParallelMatOp.MUL_MAT,
              gateType: waveGType,
              upType: waveUType,
              downType: waveDType,
              bXOrQ8Ptr: expInQ8K ? bQkDimPtr : expInQ8 ? bQDimPtr : bXNormPtr,
              bXF32Ptr: bXNormPtr,
              xQ8StrideBytes: expInQ8K ? qkDimStride : expInQ8 ? qDimStride : xStrideBytes,
              outPtr: bFfnProjPtr,
              workerScratchBase: s.moeWorkerScratch!.buffer.wasmPtr(s.moeWorkerScratch!),
              layout: s.moeBatch4WorkerScratchLayout!,
              gatePtrs: moeBatchGatePtrs,
              upPtrs: moeBatchUpPtrs,
              downPtrs: moeBatchDownPtrs,
              masks: moeBatchSparseMasks,
              weights: moeBatchWeightsF32,
              accumulate: accumulatedOut,
            });
          }
          if (!dispatched) {
            for (let wi = wStart; wi < wEnd; wi++) {
              runSparseExpertSequential(moeBatchSparseIdx[wi]);
            }
          }
          accumulatedOut = true;
        }
        return;
      }

      // Sequential fallback for remaining sparse experts (single-core or <=1 sparse expert or non-WASM)
      for (let si = 0; si < sparseIdxCount; si++) {
        runSparseExpertSequential(moeBatchSparseIdx[si]);
      }
    };

    const runSparseMoeBatch4 = (
      layerIdx: number,
      routerWeight: VTensorHandle,
      expFfnDim: number,
    ): void => {
      const uniqueCount = routeSparseMoeBatch4(routerWeight);
      executeSparseMoeBatch4Routed(layerIdx, expFfnDim, uniqueCount);
    };
    const embdInfo = this.embdInfo!;
    const outWeightInfo = this.outWeightInfo!;
    const embdTrait = getGgmlTypeTrait(embdInfo.type);
    const embdRowBytes = ((embdInfo.dims[0] ?? dim) / embdTrait.blockSize) * embdTrait.typeSizeBytes;

    const isPaged = wp.residencyMode === 'opfs-paged';
    const residentLayers = wp.getAllResidentLayers();
    const pinnedTokenEmbd = wp.getPinnedTokenEmbd();
    const pinnedOutputWeight = wp.getPinnedOutputWeight();
    const layerCount = this.layerInfos.length;

    const useQ8 = Boolean(this.performanceContract?.useQ8Activations && arena !== null);
    const kvTierLabel = kv.swapQ4Enabled
      ? 'F32/Q8/Q4/Swap vRAM'
      : kv.coldQ4Enabled
        ? 'F32/Q8/Q4 vRAM'
        : 'F32/Q8 vRAM';
    const attnLabel = arena
      ? kv.swapQ4Enabled
        ? 'F32/Q8/Q4/Swap SIMD'
        : kv.coldQ4Enabled
          ? 'F32/Q8/Q4 SIMD'
          : 'F32/Q8 SIMD'
      : kvTierLabel;

    // Compile formal Prism MatrixExecution schedule & cost-surface certificates for this model
    const contractEnv = this.performanceContract?.env;
    const prismHost: PrismHostCapabilities = {
      wasmSimd: arena !== null,
      relaxedVnni: arena?.relaxedSimdActive ?? false,
      effectiveWorkers: contractEnv?.effectiveWorkers ?? 1,
      memoryBandwidthGBs: contractEnv?.memoryBandwidthGBs ?? 4,
      computeGops: contractEnv?.computeGops ?? 8,
      barrierLatencyUs: contractEnv?.workerSyncLatencyUs ?? 2,
      precisionMode: this.precisionMode,
    };
    const scheduleOps: Array<PrismOpDescriptor & { readonly label: string }> = [];
    for (let l = 0; l < layerCount; l++) {
      const li = this.layerInfos[l];
      const pushOp = (label: string, info: GgufTensorInfo, groupSize: number) => {
        scheduleOps.push({
          label,
          weightType: info.type,
          rows: info.dims[1] ?? 1,
          cols: info.dims[0] ?? 1,
          batchSize: 1,
          groupSize,
          weightBytes: info.byteSize,
        });
      };
      pushOp(`blk.${l}.attn_q`, li.attnQ, 3);
      pushOp(`blk.${l}.attn_k`, li.attnK, 3);
      pushOp(`blk.${l}.attn_v`, li.attnV, 3);
      pushOp(`blk.${l}.attn_output`, li.attnOutput, 1);
      if (li.ffnGateInp && li.ffnGateExps && li.ffnUpExps && li.ffnDownExps) {
        pushOp(`blk.${l}.ffn_gate_inp`, li.ffnGateInp, 1);
        const nExp = Math.max(1, hp.expertCount ?? li.ffnGateExps.dims[2] ?? 1);
        const expUsed = Math.max(1, hp.expertUsedCount ?? 1);
        const pushMoeExp = (label: string, info: GgufTensorInfo, groupSize: number) => {
          scheduleOps.push({
            label,
            weightType: info.type,
            rows: info.dims[1] ?? 1,
            cols: info.dims[0] ?? 1,
            batchSize: expUsed,
            groupSize,
            weightBytes: Math.ceil((info.byteSize / nExp) * expUsed),
          });
        };
        pushMoeExp(`blk.${l}.ffn_gate_exps`, li.ffnGateExps, 2);
        pushMoeExp(`blk.${l}.ffn_up_exps`, li.ffnUpExps, 2);
        pushMoeExp(`blk.${l}.ffn_down_exps`, li.ffnDownExps, 1);
      } else if (li.ffnGate && li.ffnUp && li.ffnDown) {
        pushOp(`blk.${l}.ffn_gate`, li.ffnGate, 2);
        pushOp(`blk.${l}.ffn_up`, li.ffnUp, 2);
        pushOp(`blk.${l}.ffn_down`, li.ffnDown, 1);
      }
    }
    scheduleOps.push({
      label: 'output.weight',
      weightType: outWeightInfo.type,
      rows: outWeightInfo.dims[1] ?? hp.vocabSize,
      cols: outWeightInfo.dims[0] ?? dim,
      batchSize: 1,
      groupSize: 1,
      weightBytes: outWeightInfo.byteSize,
    });
    const compiledPrism = compilePrismMatrixSchedule(prismHost, scheduleOps);
    this.prismSchedule = compiledPrism;
    this.prismLedger.scheduleDigest = compiledPrism.modelDigest;
    this.prismLedger.activePrimaryRoute =
      compiledPrism.q8kOps > 0
        ? 'wasm-q8k-4row'
        : compiledPrism.q80Ops > 0
          ? 'wasm-q8-4row'
          : 'wasm-f32-strict';

    const singleBodyTally = { q8kOps: 0, q8kSb: 0, q80Ops: 0, strictOps: 0, fusedBarriers: 0 };
    const singleLmTally = { q8kOps: 0, q8kSb: 0, q80Ops: 0, strictOps: 0 };
    const batch4Tally = { q8kOps: 0, q8kSb: 0, q80Ops: 0, strictOps: 0, fusedBarriers: 0 };

    const canQ8K = (weightType: GgmlType, cols: number, q8kHandle: VTensorHandle | null): boolean =>
      useQ8 &&
      q8kHandle !== null &&
      arena !== null &&
      arena.supportsQ8KActivation(weightType) &&
      (cols & 255) === 0;

    // Helper: static resident-all single linear node (0% overhead fast path)
    const makeSingleLinear = (
      label: string,
      weight: VTensorHandle,
      q8Act: VTensorHandle | null,
      q8kAct: VTensorHandle | null,
      f32Act: VTensorHandle,
      out: VTensorHandle,
    ): GgmlGraphNode => {
      const cols = weight.dims[0] ?? 1;
      if (canQ8K(weight.type, cols, q8kAct) && q8kAct) {
        return b.nodeMulMatQ8K(label, weight, q8kAct, f32Act, out);
      }
      if (useQ8 && q8Act && arena?.supportsQ8Activation(weight.type)) {
        return b.nodeMulMatQ8(label, weight, q8Act, f32Act, out);
      }
      return b.nodeMulMat(label, weight, f32Act, out);
    };

    // Helper: OPFS-paged single linear node executing on the layer's committed DIMM publication
    const makePagedSingleLinear = (
      label: string,
      info: GgufTensorInfo,
      activePub: { current: PagedLayerWeights | null },
      selectHandle: (pub: PagedLayerWeights) => VTensorHandle,
      q8Act: VTensorHandle | null,
      q8kAct: VTensorHandle | null,
      f32Act: VTensorHandle,
      out: VTensorHandle,
    ): GgmlGraphNode => {
      const cols = info.dims[0] ?? 1;
      const rows = info.dims[1] ?? 1;
      const trait = getGgmlTypeTrait(info.type);
      const useWasm = arena !== null && f32Act.buffer.isWasmBacked && out.buffer.isWasmBacked;
      const useLinearQ8K =
        useWasm &&
        canQ8K(info.type, cols, q8kAct) &&
        q8kAct !== null &&
        q8kAct.buffer.isWasmBacked;
      const useLinearQ8 =
        !useLinearQ8K &&
        useQ8 &&
        useWasm &&
        q8Act !== null &&
        q8Act.buffer.isWasmBacked &&
        arena.supportsQ8Activation(info.type);
      const qkPtr = useLinearQ8K && q8kAct ? q8kAct.buffer.wasmPtr(q8kAct) : -1;
      const qPtr = useLinearQ8 && q8Act ? q8Act.buffer.wasmPtr(q8Act) : -1;
      const xPtr = useWasm ? f32Act.buffer.wasmPtr(f32Act) : -1;
      const oPtr = useWasm ? out.buffer.wasmPtr(out) : -1;

      return {
        op: GgmlOp.MUL_MAT,
        label,
        shapeDesc: `[${rows}×${cols}] · [${cols}] (vRAM)`,
        srcType: useLinearQ8K
          ? `${trait.name}×Q8_K ${arena.relaxedSimdActive ? 'VNNI' : 'Int8'}`
          : useLinearQ8
            ? `${trait.name}×Q8 ${arena.relaxedSimdActive ? 'VNNI' : 'Int8'}`
            : `${trait.name}×F32`,
        execute: () => {
          const w = selectHandle(activePub.current!);
          if (useWasm && arena && w.buffer.isWasmBacked) {
            const wPtr = w.buffer.wasmPtr(w);
            if (useLinearQ8K) {
              arena.mulMatQ8K(w.type, wPtr, qkPtr, xPtr, oPtr, cols, rows);
            } else if (useLinearQ8) {
              arena.mulMatQ8(w.type, wPtr, qPtr, xPtr, oPtr, cols, rows);
            } else {
              arena.mulMat(w.type, wPtr, xPtr, oPtr, cols, rows);
            }
          } else {
            verdantMulMat(
              w.buffer.rawBytes(),
              w.byteOffset,
              w.type,
              cols,
              rows,
              f32Act.buffer.f32View(f32Act),
              out.buffer.f32View(out),
            );
          }
        },
      };
    };

    // Helper: Fused single-barrier multi-projection node (`Q+K+V` or `Gate+Up`) for resident weights
    const makeSingleLinearGroup = (
      label: string,
      specs: readonly { weight: VTensorHandle; out: VTensorHandle }[],
      q8Act: VTensorHandle | null,
      q8kAct: VTensorHandle | null,
      f32Act: VTensorHandle,
      groupUseQ8: boolean,
      groupUseQ8K: boolean,
    ): GgmlGraphNode => {
      const cols = specs[0].weight.dims[0] ?? 1;
      const totalRows = specs.reduce((acc, sp) => acc + (sp.weight.dims[1] ?? 1), 0);
      const trait = getGgmlTypeTrait(specs[0].weight.type);
      const useWasm =
        arena !== null &&
        f32Act.buffer.isWasmBacked &&
        specs.every((sp) => sp.weight.buffer.isWasmBacked && sp.out.buffer.isWasmBacked);
      const qkPtr = groupUseQ8K && useWasm && q8kAct?.buffer.isWasmBacked ? q8kAct.buffer.wasmPtr(q8kAct) : -1;
      const qPtr = !groupUseQ8K && groupUseQ8 && useWasm && q8Act?.buffer.isWasmBacked ? q8Act.buffer.wasmPtr(q8Act) : -1;
      const xPtr = useWasm ? f32Act.buffer.wasmPtr(f32Act) : -1;
      const entries: MatGroupEntry[] = useWasm
        ? specs.map((sp) => {
            const rows = sp.weight.dims[1] ?? 1;
            return {
              weightType: sp.weight.type,
              wPtr: sp.weight.buffer.wasmPtr(sp.weight),
              outPtr: sp.out.buffer.wasmPtr(sp.out),
              rows,
              oStrideBytes: rows * 4,
            };
          })
        : [];
      const activeQ8K = groupUseQ8K && qkPtr >= 0;
      const activeQ8 = !activeQ8K && groupUseQ8 && qPtr >= 0;

      return {
        op: GgmlOp.MUL_MAT,
        label,
        shapeDesc: `[${totalRows}×${cols}] · [${cols}] (fused ${specs.length}×)`,
        srcType: activeQ8K
          ? `${trait.name}×Q8_K ${arena?.relaxedSimdActive ? 'VNNI' : 'Int8'}`
          : activeQ8
            ? `${trait.name}×Q8 ${arena?.relaxedSimdActive ? 'VNNI' : 'Int8'}`
            : useWasm
              ? `${trait.name}×F32 SIMD`
              : `${trait.name}×F32`,
        execute: () => {
          if (useWasm && arena) {
            if (activeQ8K) {
              arena.mulMatQ8KGroup(entries, qkPtr, xPtr, cols);
            } else {
              arena.mulMatQ8Group(entries, activeQ8 ? qPtr : xPtr, xPtr, cols, activeQ8);
            }
          } else {
            const xView = f32Act.buffer.f32View(f32Act);
            for (let i = 0; i < specs.length; i++) {
              const w = specs[i].weight;
              const o = specs[i].out;
              verdantMulMat(w.buffer.rawBytes(), w.byteOffset, w.type, cols, w.dims[1] ?? 1, xView, o.buffer.f32View(o));
            }
          }
        },
      };
    };

    // Helper: Fused single-barrier multi-projection node (`Q+K+V` or `Gate+Up`) for OPFS-paged weights
    const makePagedSingleLinearGroup = (
      label: string,
      specs: readonly {
        info: GgufTensorInfo;
        selectHandle: (pub: PagedLayerWeights) => VTensorHandle;
        out: VTensorHandle;
      }[],
      activePub: { current: PagedLayerWeights | null },
      q8Act: VTensorHandle | null,
      q8kAct: VTensorHandle | null,
      f32Act: VTensorHandle,
      groupUseQ8: boolean,
      groupUseQ8K: boolean,
    ): GgmlGraphNode => {
      const cols = specs[0].info.dims[0] ?? 1;
      const totalRows = specs.reduce((acc, sp) => acc + (sp.info.dims[1] ?? 1), 0);
      const trait = getGgmlTypeTrait(specs[0].info.type);
      const useWasm =
        arena !== null &&
        f32Act.buffer.isWasmBacked &&
        specs.every((sp) => sp.out.buffer.isWasmBacked);
      const qkPtr = groupUseQ8K && useWasm && q8kAct?.buffer.isWasmBacked ? q8kAct.buffer.wasmPtr(q8kAct) : -1;
      const qPtr = !groupUseQ8K && groupUseQ8 && useWasm && q8Act?.buffer.isWasmBacked ? q8Act.buffer.wasmPtr(q8Act) : -1;
      const xPtr = useWasm ? f32Act.buffer.wasmPtr(f32Act) : -1;
      const entries: Array<{
        weightType: GgmlType;
        wPtr: number;
        outPtr: number;
        rows: number;
        oStrideBytes: number;
      }> = useWasm
        ? specs.map((sp) => {
            const rows = sp.info.dims[1] ?? 1;
            return {
              weightType: sp.info.type,
              wPtr: 0,
              outPtr: sp.out.buffer.wasmPtr(sp.out),
              rows,
              oStrideBytes: rows * 4,
            };
          })
        : [];
      const activeQ8K = groupUseQ8K && qkPtr >= 0;
      const activeQ8 = !activeQ8K && groupUseQ8 && qPtr >= 0;

      return {
        op: GgmlOp.MUL_MAT,
        label,
        shapeDesc: `[${totalRows}×${cols}] · [${cols}] (vRAM fused ${specs.length}×)`,
        srcType: activeQ8K
          ? `${trait.name}×Q8_K ${arena?.relaxedSimdActive ? 'VNNI' : 'Int8'}`
          : activeQ8
            ? `${trait.name}×Q8 ${arena?.relaxedSimdActive ? 'VNNI' : 'Int8'}`
            : `${trait.name}×F32`,
        execute: () => {
          const pub = activePub.current!;
          if (useWasm && arena) {
            let allWasm = true;
            for (let i = 0; i < specs.length; i++) {
              const w = specs[i].selectHandle(pub);
              if (!w.buffer.isWasmBacked) {
                allWasm = false;
                break;
              }
              entries[i].wPtr = w.buffer.wasmPtr(w);
            }
            if (allWasm) {
              if (activeQ8K) {
                arena.mulMatQ8KGroup(entries, qkPtr, xPtr, cols);
              } else {
                arena.mulMatQ8Group(entries, activeQ8 ? qPtr : xPtr, xPtr, cols, activeQ8);
              }
              return;
            }
          }
          const xView = f32Act.buffer.f32View(f32Act);
          for (let i = 0; i < specs.length; i++) {
            const w = specs[i].selectHandle(pub);
            const o = specs[i].out;
            verdantMulMat(w.buffer.rawBytes(), w.byteOffset, w.type, cols, w.dims[1] ?? 1, xView, o.buffer.f32View(o));
          }
        },
      };
    };

    // --- A. Single-token reusable graph ---
    const singleNodes: GgmlGraphNode[] = [];
    const execState = this.singleExec;
    const singleActivePub: { current: PagedLayerWeights | null } = { current: null };

    const emptyF32 = new Float32Array(0);
    const ropeCosPtr =
      arena && this.ropeCosHandle && this.ropeCosHandle.buffer.isWasmBacked
        ? this.ropeCosHandle.buffer.wasmPtr(this.ropeCosHandle)
        : -1;
    const ropeSinPtr =
      arena && this.ropeSinHandle && this.ropeSinHandle.buffer.isWasmBacked
        ? this.ropeSinHandle.buffer.wasmPtr(this.ropeSinHandle)
        : -1;
    const canWasmRope =
      arena !== null && ropeCosPtr >= 0 && ropeSinPtr >= 0 && hp.ropeDimensionCount === headDim;
    const ropePosStrideBytes = (hp.ropeDimensionCount >>> 1) * 4;
    const ropeLutMaxPos = rope.maxCtx;
    const ropeHalf = rope.half;
    const ropeCosHandleRef = this.ropeCosHandle;
    const ropeSinHandleRef = this.ropeSinHandle;
    const dynamicRopePosCache = new Int32Array(PREFILL_BATCH_SIZE).fill(-1);
    const resolveRopeOffsetBytes = (pos: number, slotIdx: number): number => {
      if (pos < ropeLutMaxPos) {
        return pos * ropePosStrideBytes;
      }
      const rowIdx = ropeLutMaxPos + slotIdx;
      if (dynamicRopePosCache[slotIdx] !== pos && ropeCosHandleRef && ropeSinHandleRef) {
        fillRopePositionCosSin(
          pos,
          rope,
          ropeCosHandleRef.buffer.f32View(ropeCosHandleRef),
          ropeSinHandleRef.buffer.f32View(ropeSinHandleRef),
          rowIdx * ropeHalf,
        );
        dynamicRopePosCache[slotIdx] = pos;
      }
      return rowIdx * ropePosStrideBytes;
    };
    const neoxFlag = useNeoxRope ? 1 : 0;

    const sxPtr = arena && s.x.buffer.isWasmBacked ? s.x.buffer.wasmPtr(s.x) : -1;
    const pinnedEmbdWasmPtr =
      arena && pinnedTokenEmbd && pinnedTokenEmbd.buffer.isWasmBacked
        ? pinnedTokenEmbd.buffer.wasmPtr(pinnedTokenEmbd)
        : -1;

    singleNodes.push({
      op: GgmlOp.GET_ROWS,
      label: 'token_embd.get_rows',
      shapeDesc: wp.streamEmbeddings ? `[${dim}] <- OPFS row` : `[${dim}] <- token`,
      srcType: wp.streamEmbeddings ? `${embdTrait.name} OPFS` : embdTrait.name,
      execute: () => {
        if (execState.externalEmbedding) {
          s.x.buffer.f32View(s.x).set(execState.externalEmbedding);
          return;
        }
        if (pinnedTokenEmbd) {
          if (arena && sxPtr >= 0 && pinnedEmbdWasmPtr >= 0) {
            if (pinnedTokenEmbd.type === GgmlType.F32) {
              arena.kernels.copy_bytes(sxPtr, pinnedEmbdWasmPtr + execState.tokenId * embdRowBytes, dim * 4);
              return;
            }
            if (pinnedTokenEmbd.type === GgmlType.F16) {
              arena.kernels.decode_row_f16(pinnedEmbdWasmPtr + execState.tokenId * embdRowBytes, sxPtr, dim);
              return;
            }
          }
          verdantDecodeRow(
            pinnedTokenEmbd.buffer.rawBytes(),
            pinnedTokenEmbd.byteOffset + execState.tokenId * embdRowBytes,
            pinnedTokenEmbd.type,
            dim,
            s.x.buffer.f32View(s.x),
            0,
          );
        } else {
          const { handle, rowIndex } = wp.acquireEmbeddingRow(execState.tokenId);
          if (arena && sxPtr >= 0 && handle.buffer.isWasmBacked) {
            const rowWasmPtr = handle.buffer.wasmPtr(handle) + rowIndex * embdRowBytes;
            if (handle.type === GgmlType.F32) {
              arena.kernels.copy_bytes(sxPtr, rowWasmPtr, dim * 4);
              return;
            }
            if (handle.type === GgmlType.F16) {
              arena.kernels.decode_row_f16(rowWasmPtr, sxPtr, dim);
              return;
            }
          }
          verdantDecodeRow(
            handle.buffer.rawBytes(),
            handle.byteOffset + rowIndex * embdRowBytes,
            handle.type,
            dim,
            s.x.buffer.f32View(s.x),
            0,
          );
        }
      },
    });

    const singleAsyncLayers: AsyncLayerPlan[] = [];

    for (let l = 0; l < layerCount; l++) {
      const layerIdx = l;
      const info = this.layerInfos[l];
      const hot = isPaged ? wp.getHotLayer(l) : residentLayers![l];
      const lw = isPaged ? null : residentLayers![l];
      let singleLayerAcquireNode: GgmlGraphNode | null = null;

      if (isPaged) {
        // Single DIMM publication for all 7 projections in layer `l` (afflom/vRAM 3f27a941)
        singleLayerAcquireNode = {
          op: GgmlOp.CPY,
          label: `blk.${l}.vram_dimm_acquire`,
          shapeDesc: `vRAM DIMM L${l} -> frame`,
          srcType:
            wp.backingKind === 'opfs-sync'
              ? 'OPFS DIMM'
              : wp.backingKind === 'opfs-demand'
                ? 'OPFS Demand'
                : wp.backingKind === 'local-file-sync'
                  ? 'Local Disk'
                  : 'vRAM DIMM',
          execute: () => {
            singleActivePub.current = wp.acquireLayer(layerIdx);
          },
        };
        singleNodes.push(singleLayerAcquireNode);
      }
      const singlePreMoeStartIdx = singleNodes.length;

      const allAttnQ8K =
        canQ8K(info.attnQ.type, dim, s.q8kDim) &&
        canQ8K(info.attnK.type, dim, s.q8kDim) &&
        canQ8K(info.attnV.type, dim, s.q8kDim);
      const anyAttnQ8 =
        !allAttnQ8K &&
        useQ8 &&
        s.q8Dim !== null &&
        arena !== null &&
        (arena.supportsQ8Activation(info.attnQ.type) ||
          arena.supportsQ8Activation(info.attnK.type) ||
          arena.supportsQ8Activation(info.attnV.type));

      singleNodes.push(b.nodeRmsNorm(`blk.${l}.attn_norm`, s.x, hot.attnNorm, s.xNorm, hp.rmsNormEps));
      if (allAttnQ8K && s.q8kDim) {
        singleNodes.push(b.nodeQuantizeQ8K(`blk.${l}.q8k_attn_in`, s.xNorm, s.q8kDim, dim));
        singleBodyTally.q8kOps += 3;
        singleBodyTally.q8kSb += (qDim + kvDim + kvDim) * (dim >>> 8);
      } else if (anyAttnQ8 && s.q8Dim) {
        singleNodes.push(b.nodeQuantizeQ8(`blk.${l}.q8_attn_in`, s.xNorm, s.q8Dim, dim));
        singleBodyTally.q80Ops += 3;
      } else {
        singleBodyTally.strictOps += 3;
      }
      singleBodyTally.fusedBarriers += 1;

      singleNodes.push(
        isPaged
          ? makePagedSingleLinearGroup(
              `blk.${l}.attn_qkv`,
              [
                { info: info.attnQ, selectHandle: (p) => p.attnQ, out: s.q },
                { info: info.attnK, selectHandle: (p) => p.attnK, out: s.k },
                { info: info.attnV, selectHandle: (p) => p.attnV, out: s.v },
              ],
              singleActivePub,
              s.q8Dim,
              s.q8kDim,
              s.xNorm,
              anyAttnQ8,
              allAttnQ8K,
            )
          : makeSingleLinearGroup(
              `blk.${l}.attn_qkv`,
              [
                { weight: lw!.attnQ, out: s.q },
                { weight: lw!.attnK, out: s.k },
                { weight: lw!.attnV, out: s.v },
              ],
              s.q8Dim,
              s.q8kDim,
              s.xNorm,
              anyAttnQ8,
              allAttnQ8K,
            ),
      );
      if (hot.attnQBias) {
        singleNodes.push(b.nodeAdd(`blk.${l}.attn_q_bias`, s.q, hot.attnQBias, s.q));
      }
      if (hot.attnKBias) {
        singleNodes.push(b.nodeAdd(`blk.${l}.attn_k_bias`, s.k, hot.attnKBias, s.k));
      }
      if (hot.attnVBias) {
        singleNodes.push(b.nodeAdd(`blk.${l}.attn_v_bias`, s.v, hot.attnVBias, s.v));
      }
      const qPtr = s.q.buffer.wasmPtr(s.q);
      const kPtr = s.k.buffer.wasmPtr(s.k);
      const vPtr = s.v.buffer.wasmPtr(s.v);
      const attnOutPtr = s.attnOut.buffer.wasmPtr(s.attnOut);
      const useWasmAct = arena !== null && qPtr >= 0 && kPtr >= 0 && vPtr >= 0 && attnOutPtr >= 0;

      if (hot.attnQNorm) {
        const qNormHandle = hot.attnQNorm;
        const qNormWPtr = useWasmAct && qNormHandle.buffer.isWasmBacked ? qNormHandle.buffer.wasmPtr(qNormHandle) : -1;
        singleNodes.push({
          op: GgmlOp.RMS_NORM,
          label: `blk.${l}.attn_q_norm`,
          shapeDesc: `[${hp.headCount}×${headDim}] (per-head)`,
          srcType: qNormWPtr >= 0 ? 'F32 SIMD' : 'F32',
          execute: () => {
            if (arena && qNormWPtr >= 0) {
              arena.kernels.rms_norm_per_head_f32(qPtr, qNormWPtr, hp.headCount, headDim, hp.rmsNormEps);
            } else {
              verdantRmsNormPerHead(
                s.q.buffer.f32View(s.q),
                qNormHandle.buffer.f32View(qNormHandle),
                hp.headCount,
                headDim,
                hp.rmsNormEps,
              );
            }
          },
        });
      }
      if (hot.attnKNorm) {
        const kNormHandle = hot.attnKNorm;
        const kNormWPtr = useWasmAct && kNormHandle.buffer.isWasmBacked ? kNormHandle.buffer.wasmPtr(kNormHandle) : -1;
        singleNodes.push({
          op: GgmlOp.RMS_NORM,
          label: `blk.${l}.attn_k_norm`,
          shapeDesc: `[${hp.headCountKv}×${headDim}] (per-head)`,
          srcType: kNormWPtr >= 0 ? 'F32 SIMD' : 'F32',
          execute: () => {
            if (arena && kNormWPtr >= 0) {
              arena.kernels.rms_norm_per_head_f32(kPtr, kNormWPtr, hp.headCountKv, headDim, hp.rmsNormEps);
            } else {
              verdantRmsNormPerHead(
                s.k.buffer.f32View(s.k),
                kNormHandle.buffer.f32View(kNormHandle),
                hp.headCountKv,
                headDim,
                hp.rmsNormEps,
              );
            }
          },
        });
      }

      singleNodes.push(
        {
          op: GgmlOp.ROPE,
          label: `blk.${l}.rope_q`,
          shapeDesc: `[${hp.headCount}×${headDim}] (Verdant LUT)`,
          srcType: canWasmRope && useWasmAct ? 'F32 SIMD' : 'F32',
          execute: () => {
            if (canWasmRope && useWasmAct && arena) {
              const pOff = resolveRopeOffsetBytes(execState.pos, 0);
              arena.kernels.rope_f32(qPtr, ropeCosPtr + pOff, ropeSinPtr + pOff, hp.headCount, headDim, neoxFlag);
            } else {
              verdantRopePrecomputed(
                s.q.buffer.f32View(s.q),
                hp.headCount,
                headDim,
                execState.pos,
                rope,
                useNeoxRope,
              );
            }
          },
        },
        {
          op: GgmlOp.ROPE,
          label: `blk.${l}.rope_k`,
          shapeDesc: `[${hp.headCountKv}×${headDim}] (Verdant LUT)`,
          srcType: canWasmRope && useWasmAct ? 'F32 SIMD' : 'F32',
          execute: () => {
            if (canWasmRope && useWasmAct && arena) {
              const pOff = resolveRopeOffsetBytes(execState.pos, 0);
              arena.kernels.rope_f32(kPtr, ropeCosPtr + pOff, ropeSinPtr + pOff, hp.headCountKv, headDim, neoxFlag);
            } else {
              verdantRopePrecomputed(
                s.k.buffer.f32View(s.k),
                hp.headCountKv,
                headDim,
                execState.pos,
                rope,
                useNeoxRope,
              );
            }
          },
        },
        {
          op: GgmlOp.SET_ROWS,
          label: `blk.${l}.k_cache_set`,
          shapeDesc: `vram_kv[${l}] <- [${kvDim}]`,
          srcType: kvTierLabel,
          execute: () => {
            if (useWasmAct && arena) {
              kv.writeKvPtr(false, layerIdx, execState.pos, kPtr, arena);
            } else {
              kv.writeKv(false, layerIdx, execState.pos, s.k.buffer.f32View(s.k));
            }
          },
        },
        {
          op: GgmlOp.SET_ROWS,
          label: `blk.${l}.v_cache_set`,
          shapeDesc: `vram_kv[${l}] <- [${kvDim}]`,
          srcType: kvTierLabel,
          execute: () => {
            if (useWasmAct && arena) {
              kv.writeKvPtr(true, layerIdx, execState.pos, vPtr, arena);
            } else {
              kv.writeKv(true, layerIdx, execState.pos, s.v.buffer.f32View(s.v));
            }
          },
        },
        {
          op: GgmlOp.FLASH_ATTN_EXT,
          label: `blk.${l}.flash_attn_ext`,
          shapeDesc: `Q[${hp.headCount}×${headDim}] × vRAM KV`,
          srcType: attnLabel,
          execute: () => {
            kv.flashAttnExt(
              useWasmAct ? emptyF32 : s.q.buffer.f32View(s.q),
              qPtr,
              useWasmAct ? emptyF32 : s.attnOut.buffer.f32View(s.attnOut),
              attnOutPtr,
              hp.headCount,
              layerIdx,
              execState.seqLen,
              arena,
            );
          },
        },
      );

      const attnOutQ8K = canQ8K(info.attnOutput.type, qDim, s.q8kAttnOut);
      if (attnOutQ8K && s.q8kAttnOut) {
        singleNodes.push(b.nodeQuantizeQ8K(`blk.${l}.q8k_attn_out`, s.attnOut, s.q8kAttnOut, qDim));
        singleBodyTally.q8kOps += 1;
        singleBodyTally.q8kSb += dim * (qDim >>> 8);
      } else if (useQ8 && s.q8AttnOut && arena?.supportsQ8Activation(info.attnOutput.type)) {
        singleNodes.push(b.nodeQuantizeQ8(`blk.${l}.q8_attn_out`, s.attnOut, s.q8AttnOut, qDim));
        singleBodyTally.q80Ops += 1;
      } else {
        singleBodyTally.strictOps += 1;
      }
      singleNodes.push(
        isPaged
          ? makePagedSingleLinear(
              `blk.${l}.attn_output`,
              info.attnOutput,
              singleActivePub,
              (p) => p.attnOutput,
              s.q8AttnOut,
              s.q8kAttnOut,
              s.attnOut,
              s.attnProj,
            )
          : makeSingleLinear(`blk.${l}.attn_output`, lw!.attnOutput, s.q8AttnOut, s.q8kAttnOut, s.attnOut, s.attnProj),
      );
      if (hot.attnOutputBias) {
        singleNodes.push(b.nodeAdd(`blk.${l}.attn_output_bias`, s.attnProj, hot.attnOutputBias, s.attnProj));
      }
      singleNodes.push(
        b.nodeAdd(`blk.${l}.attn_residual`, s.x, s.attnProj, s.x),
        b.nodeRmsNorm(`blk.${l}.ffn_norm`, s.x, hot.ffnNorm, s.xNorm, hp.rmsNormEps),
      );

      const singlePreMoeNodes = singleNodes.slice(singlePreMoeStartIdx);

      if (info.ffnGateInp && info.ffnGateExps && info.ffnUpExps && info.ffnDownExps) {
        const expFfnDim = info.ffnGateExps.dims[1] ?? ffnDim;
        const nExps = info.ffnGateExps.dims[2] ?? hp.expertCount ?? 1;
        const expTrait = getGgmlTypeTrait(info.ffnGateExps.type);
        const residualNode = b.nodeAdd(`blk.${l}.ffn_residual`, s.x, s.ffnProj, s.x);
        singleNodes.push(
          {
            op: GgmlOp.MUL_MAT,
            label: `blk.${l}.ffn_moe_sparse`,
            shapeDesc: `Top-${Math.min(moeTopK, nExps)}/${nExps} × [${expFfnDim}×${dim}] (vRAM MoE)`,
            srcType: `${expTrait.name} Sparse MoE`,
            execute: () => {
              const routerW = isPaged ? singleActivePub.current!.ffnGateInp! : lw!.ffnGateInp!;
              runSparseMoeSingle(layerIdx, routerW, s.xNorm, 0, s.q8Dim, s.q8kDim, s.ffnProj, 0, expFfnDim);
            },
          },
          residualNode,
        );
        singleAsyncLayers.push({
          layerIdx,
          acquireNode: singleLayerAcquireNode,
          preMoeNodes: singlePreMoeNodes,
          moe: {
            routeAndGetExperts: () => {
              const routerW = isPaged ? singleActivePub.current!.ffnGateInp! : lw!.ffnGateInp!;
              const kUsed = routeSparseMoeSingle(routerW, s.xNorm, 0);
              const expertIds: number[] = [];
              for (let k = 0; k < kUsed; k++) {
                const id = moeTopIds[k];
                if (id >= 0 && moeTopProbs[k] !== 0) {
                  expertIds.push(id);
                }
              }
              return { count: kUsed, expertIds };
            },
            executeRouted: (kUsed: number) => {
              executeSparseMoeSingleRouted(
                layerIdx,
                s.xNorm,
                0,
                s.q8Dim,
                s.q8kDim,
                s.ffnProj,
                0,
                expFfnDim,
                kUsed,
              );
            },
            residualNode,
          },
          denseFfnNodes: [],
        });
      } else {
        const singleDenseStartIdx = singleNodes.length;
        const ffnGateInfo = info.ffnGate!;
        const ffnUpInfo = info.ffnUp!;
        const ffnDownInfo = info.ffnDown!;
        const allFfnInQ8K =
          canQ8K(ffnGateInfo.type, dim, s.q8kDim) && canQ8K(ffnUpInfo.type, dim, s.q8kDim);
        const anyFfnInQ8 =
          !allFfnInQ8K &&
          useQ8 &&
          s.q8Dim !== null &&
          arena !== null &&
          (arena.supportsQ8Activation(ffnGateInfo.type) || arena.supportsQ8Activation(ffnUpInfo.type));
        if (allFfnInQ8K && s.q8kDim) {
          singleNodes.push(b.nodeQuantizeQ8K(`blk.${l}.q8k_ffn_in`, s.xNorm, s.q8kDim, dim));
          singleBodyTally.q8kOps += 2;
          singleBodyTally.q8kSb += (ffnDim * 2) * (dim >>> 8);
        } else if (anyFfnInQ8 && s.q8Dim) {
          singleNodes.push(b.nodeQuantizeQ8(`blk.${l}.q8_ffn_in`, s.xNorm, s.q8Dim, dim));
          singleBodyTally.q80Ops += 2;
        } else {
          singleBodyTally.strictOps += 2;
        }
        singleBodyTally.fusedBarriers += 1;

        singleNodes.push(
          isPaged
            ? makePagedSingleLinearGroup(
                `blk.${l}.ffn_gate_up`,
                [
                  { info: ffnGateInfo, selectHandle: (p) => p.ffnGate!, out: s.ffnGate },
                  { info: ffnUpInfo, selectHandle: (p) => p.ffnUp!, out: s.ffnUp },
                ],
                singleActivePub,
                s.q8Dim,
                s.q8kDim,
                s.xNorm,
                anyFfnInQ8,
                allFfnInQ8K,
              )
            : makeSingleLinearGroup(
                `blk.${l}.ffn_gate_up`,
                [
                  { weight: lw!.ffnGate!, out: s.ffnGate },
                  { weight: lw!.ffnUp!, out: s.ffnUp },
                ],
                s.q8Dim,
                s.q8kDim,
                s.xNorm,
                anyFfnInQ8,
                allFfnInQ8K,
              ),
          b.nodeSwiGlu(`blk.${l}.ffn_swiglu`, s.ffnGate, s.ffnUp, s.ffnHidden),
        );

        const ffnDownQ8K = canQ8K(ffnDownInfo.type, ffnDim, s.q8kFfn);
        if (ffnDownQ8K && s.q8kFfn) {
          singleNodes.push(b.nodeQuantizeQ8K(`blk.${l}.q8k_ffn_down`, s.ffnHidden, s.q8kFfn, ffnDim));
          singleBodyTally.q8kOps += 1;
          singleBodyTally.q8kSb += dim * (ffnDim >>> 8);
        } else if (useQ8 && s.q8Ffn && arena?.supportsQ8Activation(ffnDownInfo.type)) {
          singleNodes.push(b.nodeQuantizeQ8(`blk.${l}.q8_ffn_down`, s.ffnHidden, s.q8Ffn, ffnDim));
          singleBodyTally.q80Ops += 1;
        } else {
          singleBodyTally.strictOps += 1;
        }
        singleNodes.push(
          isPaged
            ? makePagedSingleLinear(
                `blk.${l}.ffn_down`,
                ffnDownInfo,
                singleActivePub,
                (p) => p.ffnDown!,
                s.q8Ffn,
                s.q8kFfn,
                s.ffnHidden,
                s.ffnProj,
              )
            : makeSingleLinear(`blk.${l}.ffn_down`, lw!.ffnDown!, s.q8Ffn, s.q8kFfn, s.ffnHidden, s.ffnProj),
          b.nodeAdd(`blk.${l}.ffn_residual`, s.x, s.ffnProj, s.x),
        );
        singleAsyncLayers.push({
          layerIdx,
          acquireNode: singleLayerAcquireNode,
          preMoeNodes: singlePreMoeNodes,
          denseFfnNodes: singleNodes.slice(singleDenseStartIdx),
        });
      }
    }

    // Record exact boundary before LM head so non-final prefill tokens can skip output.weight!
    this.singleGraphBodyNodeCount = singleNodes.length;
    const lmNormStartIdx = singleNodes.length;

    singleNodes.push(b.nodeRmsNorm('output_norm', s.x, wp.outputNorm, s.xNorm, hp.rmsNormEps));
    const lmHeadQ8K = canQ8K(outWeightInfo.type, dim, s.q8kDim);
    if (lmHeadQ8K && s.q8kDim) {
      singleNodes.push(b.nodeQuantizeQ8K('output.q8k_in', s.xNorm, s.q8kDim, dim));
      singleLmTally.q8kOps = 1;
      singleLmTally.q8kSb = (outWeightInfo.dims[1] ?? hp.vocabSize) * (dim >>> 8);
    } else if (useQ8 && s.q8Dim && arena?.supportsQ8Activation(outWeightInfo.type)) {
      singleNodes.push(b.nodeQuantizeQ8('output.q8_in', s.xNorm, s.q8Dim, dim));
      singleLmTally.q80Ops = 1;
    } else {
      singleLmTally.strictOps = 1;
    }

    const lmNormAndQuantNodes = singleNodes.slice(lmNormStartIdx);
    let pinnedLmHeadNode: GgmlGraphNode | null = null;
    let executeLmHeadStripe:
      | ((weightStripe: VTensorHandle, rowStart: number, rowCount: number) => void)
      | null = null;

    if (pinnedOutputWeight) {
      pinnedLmHeadNode = makeSingleLinear(
        'output.weight',
        pinnedOutputWeight,
        s.q8Dim,
        s.q8kDim,
        s.xNorm,
        s.logits,
      );
      singleNodes.push(pinnedLmHeadNode);
    } else {
      // Row-striped OPFS LM-head streaming: bit-exact logits inside a bounded <= 6 MiB WASM tile!
      const outCols = outWeightInfo.dims[0] ?? dim;
      const outRows = outWeightInfo.dims[1] ?? hp.vocabSize;
      const outTrait = getGgmlTypeTrait(outWeightInfo.type);
      const useWasmLm = arena !== null && s.xNorm.buffer.isWasmBacked && s.logits.buffer.isWasmBacked;
      const useLmQ8K =
        useWasmLm &&
        lmHeadQ8K &&
        s.q8kDim !== null &&
        s.q8kDim.buffer.isWasmBacked;
      const useLmQ8 =
        !useLmQ8K &&
        useQ8 &&
        useWasmLm &&
        s.q8Dim !== null &&
        s.q8Dim.buffer.isWasmBacked &&
        arena.supportsQ8Activation(outWeightInfo.type);
      const qkPtr = useLmQ8K && s.q8kDim ? s.q8kDim.buffer.wasmPtr(s.q8kDim) : -1;
      const qPtr = useLmQ8 && s.q8Dim ? s.q8Dim.buffer.wasmPtr(s.q8Dim) : -1;
      const xPtr = useWasmLm ? s.xNorm.buffer.wasmPtr(s.xNorm) : -1;
      const oBasePtr = useWasmLm ? s.logits.buffer.wasmPtr(s.logits) : -1;

      const runLmHeadStripe = (weightStripe: VTensorHandle, rowStart: number, rowCount: number) => {
        if (useWasmLm && arena && weightStripe.buffer.isWasmBacked) {
          const wPtr = weightStripe.buffer.wasmPtr(weightStripe);
          const stripeOutPtr = oBasePtr + rowStart * 4;
          if (useLmQ8K) {
            arena.mulMatQ8K(weightStripe.type, wPtr, qkPtr, xPtr, stripeOutPtr, outCols, rowCount);
          } else if (useLmQ8) {
            arena.mulMatQ8(weightStripe.type, wPtr, qPtr, xPtr, stripeOutPtr, outCols, rowCount);
          } else {
            arena.mulMat(weightStripe.type, wPtr, xPtr, stripeOutPtr, outCols, rowCount);
          }
        } else {
          verdantMulMat(
            weightStripe.buffer.rawBytes(),
            weightStripe.byteOffset,
            weightStripe.type,
            outCols,
            rowCount,
            s.xNorm.buffer.f32View(s.xNorm),
            s.logits.buffer.f32View(s.logits).subarray(rowStart, rowStart + rowCount),
          );
        }
      };
      executeLmHeadStripe = runLmHeadStripe;

      singleNodes.push({
        op: GgmlOp.MUL_MAT,
        label: 'output.weight',
        shapeDesc: `[${outRows}×${outCols}] · [${outCols}] (OPFS striped)`,
        srcType: useLmQ8K
          ? `${outTrait.name}×Q8_K ${arena.relaxedSimdActive ? 'VNNI' : 'Int8'} OPFS`
          : useLmQ8
            ? `${outTrait.name}×Q8 ${arena.relaxedSimdActive ? 'VNNI' : 'Int8'} OPFS`
            : `${outTrait.name}×F32 OPFS`,
        execute: () => {
          wp.forEachLmHeadStripe(runLmHeadStripe);
        },
      });
    }

    this.compiledSingleGraph = b.compileGraph(singleNodes);
    this.singleAsyncPlan = {
      embedNode: singleNodes[0],
      layers: singleAsyncLayers,
      lmNormAndQuantNodes,
      pinnedLmHeadNode,
      executeLmHeadStripe,
    };
    this.prismSingleBodyTally = singleBodyTally;
    this.prismSingleLmTally = singleLmTally;

    // --- B. Universal Batched 4-token prefill graph (amortizes weights + OPFS restores 4×!) ---
    const batchNodes: GgmlGraphNode[] = [];
    const batchState = this.batchExec;
    const B = PREFILL_BATCH_SIZE;
    const batchActivePub: { current: PagedLayerWeights | null } = { current: null };

    const makeBatchQuantizeQ8 = (
      label: string,
      xBatch: VTensorHandle,
      q8Batch: VTensorHandle,
      cols: number,
    ): GgmlGraphNode => {
      const useWasm = arena !== null && xBatch.buffer.isWasmBacked && q8Batch.buffer.isWasmBacked;
      const xBase = useWasm ? xBatch.buffer.wasmPtr(xBatch) : -1;
      const qBase = useWasm ? q8Batch.buffer.wasmPtr(q8Batch) : -1;
      const xStride = cols * 4;
      const qStride = (cols / 32) * Q8_0_ACT_BLOCK_BYTES;
      return {
        op: GgmlOp.CPY,
        label,
        shapeDesc: `F32[${B}×${cols}] -> Q8_0_ACT`,
        srcType: arena?.relaxedSimdActive ? 'Q8_0 VNNI B4' : 'Q8_0 Int8 B4',
        execute: () => {
          if (useWasm && arena) {
            for (let t = 0; t < B; t++) {
              arena.quantizeActivation(xBase + t * xStride, qBase + t * qStride, cols);
            }
          }
        },
      };
    };

    const makeBatchQuantizeQ8K = (
      label: string,
      xBatch: VTensorHandle,
      q8kBatch: VTensorHandle,
      cols: number,
    ): GgmlGraphNode => {
      const useWasm = arena !== null && xBatch.buffer.isWasmBacked && q8kBatch.buffer.isWasmBacked;
      const xBase = useWasm ? xBatch.buffer.wasmPtr(xBatch) : -1;
      const qBase = useWasm ? q8kBatch.buffer.wasmPtr(q8kBatch) : -1;
      const xStride = cols * 4;
      const qStride = (cols >>> 8) * Q8_K_ACT_BLOCK_BYTES;
      return {
        op: GgmlOp.CPY,
        label,
        shapeDesc: `F32[${B}×${cols}] -> Q8_K`,
        srcType: arena?.relaxedSimdActive ? 'Q8_K VNNI B4' : 'Q8_K Int8 B4',
        execute: () => {
          if (useWasm && arena) {
            for (let t = 0; t < B; t++) {
              arena.quantizeActivationQ8K(xBase + t * xStride, qBase + t * qStride, cols);
            }
          }
        },
      };
    };

    const makeBatchMulMat = (
      label: string,
      weight: VTensorHandle,
      q8Batch: VTensorHandle | null,
      q8kBatch: VTensorHandle | null,
      xBatch: VTensorHandle,
      outBatch: VTensorHandle,
    ): GgmlGraphNode => {
      const cols = weight.dims[0];
      const rows = weight.dims[1];
      const trait = getGgmlTypeTrait(weight.type);
      const useWasm = arena !== null && weight.buffer.isWasmBacked;
      const useBatchQ8K = useWasm && canQ8K(weight.type, cols, q8kBatch) && q8kBatch !== null;
      const useBatchQ8 = !useBatchQ8K && useQ8 && useWasm && q8Batch !== null && arena.supportsQ8Activation(weight.type);
      const wPtr = useWasm ? weight.buffer.wasmPtr(weight) : -1;
      const qkPtr = useBatchQ8K && q8kBatch ? q8kBatch.buffer.wasmPtr(q8kBatch) : -1;
      const qPtr = useBatchQ8 && q8Batch ? q8Batch.buffer.wasmPtr(q8Batch) : -1;
      const xPtr = useWasm ? xBatch.buffer.wasmPtr(xBatch) : -1;
      const oPtr = useWasm ? outBatch.buffer.wasmPtr(outBatch) : -1;

      return {
        op: GgmlOp.MUL_MAT,
        label,
        shapeDesc: `[${rows}×${cols}] · [${B}×${cols}]`,
        srcType: useBatchQ8K
          ? `${trait.name}×Q8_K ${arena.relaxedSimdActive ? 'VNNI' : 'Int8'} B4`
          : useBatchQ8
            ? `${trait.name}×Q8 ${arena.relaxedSimdActive ? 'VNNI' : 'Int8'} B4`
            : useWasm
              ? `${trait.name}×F32 B4`
              : `${trait.name}×F32`,
        execute: () => {
          if (useWasm && arena) {
            if (useBatchQ8K) {
              arena.mulMatBatchQ8K(weight.type, wPtr, qkPtr, xPtr, oPtr, cols, rows, B);
            } else if (useBatchQ8) {
              arena.mulMatBatchQ8(weight.type, wPtr, qPtr, xPtr, oPtr, cols, rows, B);
            } else {
              arena.mulMatBatch(weight.type, wPtr, xPtr, oPtr, cols, rows, B);
            }
          } else {
            const xAll = xBatch.buffer.f32View(xBatch);
            const oAll = outBatch.buffer.f32View(outBatch);
            const raw = weight.buffer.rawBytes();
            for (let t = 0; t < B; t++) {
              verdantMulMat(
                raw,
                weight.byteOffset,
                weight.type,
                cols,
                rows,
                xAll.subarray(t * cols, (t + 1) * cols),
                oAll.subarray(t * rows, (t + 1) * rows),
              );
            }
          }
        },
      };
    };

    const makePagedBatchMulMat = (
      label: string,
      info: GgufTensorInfo,
      selectHandle: (pub: PagedLayerWeights) => VTensorHandle,
      q8Batch: VTensorHandle | null,
      q8kBatch: VTensorHandle | null,
      xBatch: VTensorHandle,
      outBatch: VTensorHandle,
    ): GgmlGraphNode => {
      const cols = info.dims[0] ?? 1;
      const rows = info.dims[1] ?? 1;
      const trait = getGgmlTypeTrait(info.type);
      const useWasm = arena !== null && xBatch.buffer.isWasmBacked && outBatch.buffer.isWasmBacked;
      const useBatchQ8K =
        useWasm &&
        canQ8K(info.type, cols, q8kBatch) &&
        q8kBatch !== null &&
        q8kBatch.buffer.isWasmBacked;
      const useBatchQ8 =
        !useBatchQ8K &&
        useQ8 &&
        useWasm &&
        q8Batch !== null &&
        q8Batch.buffer.isWasmBacked &&
        arena.supportsQ8Activation(info.type);
      const qkPtr = useBatchQ8K && q8kBatch ? q8kBatch.buffer.wasmPtr(q8kBatch) : -1;
      const qPtr = useBatchQ8 && q8Batch ? q8Batch.buffer.wasmPtr(q8Batch) : -1;
      const xPtr = useWasm ? xBatch.buffer.wasmPtr(xBatch) : -1;
      const oPtr = useWasm ? outBatch.buffer.wasmPtr(outBatch) : -1;

      return {
        op: GgmlOp.MUL_MAT,
        label,
        shapeDesc: `[${rows}×${cols}] · [${B}×${cols}] (vRAM)`,
        srcType: useBatchQ8K
          ? `${trait.name}×Q8_K ${arena.relaxedSimdActive ? 'VNNI' : 'Int8'} B4`
          : useBatchQ8
            ? `${trait.name}×Q8 ${arena.relaxedSimdActive ? 'VNNI' : 'Int8'} B4`
            : useWasm
              ? `${trait.name}×F32 B4`
              : `${trait.name}×F32`,
        execute: () => {
          const w = selectHandle(batchActivePub.current!);
          if (useWasm && arena && w.buffer.isWasmBacked) {
            const wPtr = w.buffer.wasmPtr(w);
            if (useBatchQ8K) {
              arena.mulMatBatchQ8K(w.type, wPtr, qkPtr, xPtr, oPtr, cols, rows, B);
            } else if (useBatchQ8) {
              arena.mulMatBatchQ8(w.type, wPtr, qPtr, xPtr, oPtr, cols, rows, B);
            } else {
              arena.mulMatBatch(w.type, wPtr, xPtr, oPtr, cols, rows, B);
            }
          } else {
            const xAll = xBatch.buffer.f32View(xBatch);
            const oAll = outBatch.buffer.f32View(outBatch);
            const raw = w.buffer.rawBytes();
            for (let t = 0; t < B; t++) {
              verdantMulMat(
                raw,
                w.byteOffset,
                w.type,
                cols,
                rows,
                xAll.subarray(t * cols, (t + 1) * cols),
                oAll.subarray(t * rows, (t + 1) * rows),
              );
            }
          }
        },
      };
    };

    const makeBatchRmsNorm = (
      label: string,
      xBatch: VTensorHandle,
      weight: VTensorHandle,
      outBatch: VTensorHandle,
    ): GgmlGraphNode => {
      const useWasm = arena !== null && xBatch.buffer.isWasmBacked;
      const xBase = useWasm ? xBatch.buffer.wasmPtr(xBatch) : -1;
      const wPtr = useWasm ? weight.buffer.wasmPtr(weight) : -1;
      const oBase = useWasm ? outBatch.buffer.wasmPtr(outBatch) : -1;
      const strideBytes = dim * 4;
      return {
        op: GgmlOp.RMS_NORM,
        label,
        shapeDesc: `[${B}×${dim}]`,
        srcType: useWasm ? 'F32 SIMD B4' : 'F32',
        execute: () => {
          if (useWasm && arena) {
            for (let t = 0; t < B; t++) {
              arena.kernels.rms_norm_f32(
                xBase + t * strideBytes,
                wPtr,
                oBase + t * strideBytes,
                dim,
                hp.rmsNormEps,
              );
            }
          } else {
            const xAll = xBatch.buffer.f32View(xBatch);
            const wView = weight.buffer.f32View(weight);
            const oAll = outBatch.buffer.f32View(outBatch);
            for (let t = 0; t < B; t++) {
              verdantRmsNorm(
                xAll.subarray(t * dim, (t + 1) * dim),
                wView,
                oAll.subarray(t * dim, (t + 1) * dim),
                dim,
                hp.rmsNormEps,
              );
            }
          }
        },
      };
    };

    const makeBatchBiasAdd = (
      label: string,
      vecBatch: VTensorHandle,
      bias: VTensorHandle,
      vecLen: number,
    ): GgmlGraphNode => {
      const useWasmBias = arena !== null && vecBatch.buffer.isWasmBacked && bias.buffer.isWasmBacked;
      const vBase = useWasmBias ? vecBatch.buffer.wasmPtr(vecBatch) : -1;
      const bPtr = useWasmBias ? bias.buffer.wasmPtr(bias) : -1;
      const vStrideBytes = vecLen * 4;
      return {
        op: GgmlOp.ADD,
        label,
        shapeDesc: `[${B}×${vecLen}]`,
        srcType: useWasmBias ? 'F32 SIMD B4' : 'F32',
        execute: () => {
          if (useWasmBias && arena) {
            for (let t = 0; t < B; t++) {
              const vp = vBase + t * vStrideBytes;
              arena.kernels.add_f32(vp, bPtr, vp, vecLen);
            }
          } else {
            const vAll = vecBatch.buffer.f32View(vecBatch);
            const bView = bias.buffer.f32View(bias);
            for (let t = 0; t < B; t++) {
              const off = t * vecLen;
              for (let i = 0; i < vecLen; i++) {
                vAll[off + i] = Math.fround(vAll[off + i] + bView[i]);
              }
            }
          }
        },
      };
    };

    // Helper: Fused single-barrier B4 multi-projection node (`Q+K+V` or `Gate+Up` prefill)
    const makeBatchLinearGroup = (
      label: string,
      specs: readonly { weight: VTensorHandle; outBatch: VTensorHandle }[],
      q8Batch: VTensorHandle | null,
      q8kBatch: VTensorHandle | null,
      xBatch: VTensorHandle,
      groupUseQ8: boolean,
      groupUseQ8K: boolean,
    ): GgmlGraphNode => {
      const cols = specs[0].weight.dims[0] ?? 1;
      const totalRows = specs.reduce((acc, sp) => acc + (sp.weight.dims[1] ?? 1), 0);
      const trait = getGgmlTypeTrait(specs[0].weight.type);
      const useWasm =
        arena !== null &&
        xBatch.buffer.isWasmBacked &&
        specs.every((sp) => sp.weight.buffer.isWasmBacked && sp.outBatch.buffer.isWasmBacked);
      const qkPtr = groupUseQ8K && useWasm && q8kBatch?.buffer.isWasmBacked ? q8kBatch.buffer.wasmPtr(q8kBatch) : -1;
      const qPtr = !groupUseQ8K && groupUseQ8 && useWasm && q8Batch?.buffer.isWasmBacked ? q8Batch.buffer.wasmPtr(q8Batch) : -1;
      const xPtr = useWasm ? xBatch.buffer.wasmPtr(xBatch) : -1;
      const entries: MatGroupEntry[] = useWasm
        ? specs.map((sp) => {
            const rows = sp.weight.dims[1] ?? 1;
            return {
              weightType: sp.weight.type,
              wPtr: sp.weight.buffer.wasmPtr(sp.weight),
              outPtr: sp.outBatch.buffer.wasmPtr(sp.outBatch),
              rows,
              oStrideBytes: rows * 4,
            };
          })
        : [];
      const activeQ8K = groupUseQ8K && qkPtr >= 0;
      const activeQ8 = !activeQ8K && groupUseQ8 && qPtr >= 0;

      return {
        op: GgmlOp.MUL_MAT,
        label,
        shapeDesc: `[${totalRows}×${cols}] · [${B}×${cols}] (fused ${specs.length}×)`,
        srcType: activeQ8K
          ? `${trait.name}×Q8_K B${B} ${arena?.relaxedSimdActive ? 'VNNI' : 'Int8'}`
          : activeQ8
            ? `${trait.name}×Q8 B${B} ${arena?.relaxedSimdActive ? 'VNNI' : 'Int8'}`
            : useWasm
              ? `${trait.name}×F32 B${B} SIMD`
              : `${trait.name}×F32 B${B}`,
        execute: () => {
          if (useWasm && arena) {
            if (activeQ8K) {
              arena.mulMatBatch4Q8KGroup(entries, qkPtr, xPtr, cols);
            } else {
              arena.mulMatBatch4Q8Group(entries, activeQ8 ? qPtr : xPtr, xPtr, cols, activeQ8);
            }
          } else {
            const xAll = xBatch.buffer.f32View(xBatch);
            for (let i = 0; i < specs.length; i++) {
              const w = specs[i].weight;
              const rows = w.dims[1] ?? 1;
              const oAll = specs[i].outBatch.buffer.f32View(specs[i].outBatch);
              const wRaw = w.buffer.rawBytes();
              for (let t = 0; t < B; t++) {
                verdantMulMat(
                  wRaw,
                  w.byteOffset,
                  w.type,
                  cols,
                  rows,
                  xAll.subarray(t * cols, (t + 1) * cols),
                  oAll.subarray(t * rows, (t + 1) * rows),
                );
              }
            }
          }
        },
      };
    };

    const makePagedBatchLinearGroup = (
      label: string,
      specs: readonly {
        info: GgufTensorInfo;
        selectHandle: (pub: PagedLayerWeights) => VTensorHandle;
        outBatch: VTensorHandle;
      }[],
      q8Batch: VTensorHandle | null,
      q8kBatch: VTensorHandle | null,
      xBatch: VTensorHandle,
      groupUseQ8: boolean,
      groupUseQ8K: boolean,
    ): GgmlGraphNode => {
      const cols = specs[0].info.dims[0] ?? 1;
      const totalRows = specs.reduce((acc, sp) => acc + (sp.info.dims[1] ?? 1), 0);
      const trait = getGgmlTypeTrait(specs[0].info.type);
      const useWasm =
        arena !== null &&
        xBatch.buffer.isWasmBacked &&
        specs.every((sp) => sp.outBatch.buffer.isWasmBacked);
      const qkPtr = groupUseQ8K && useWasm && q8kBatch?.buffer.isWasmBacked ? q8kBatch.buffer.wasmPtr(q8kBatch) : -1;
      const qPtr = !groupUseQ8K && groupUseQ8 && useWasm && q8Batch?.buffer.isWasmBacked ? q8Batch.buffer.wasmPtr(q8Batch) : -1;
      const xPtr = useWasm ? xBatch.buffer.wasmPtr(xBatch) : -1;
      const entries: Array<{
        weightType: GgmlType;
        wPtr: number;
        outPtr: number;
        rows: number;
        oStrideBytes: number;
      }> = useWasm
        ? specs.map((sp) => {
            const rows = sp.info.dims[1] ?? 1;
            return {
              weightType: sp.info.type,
              wPtr: 0,
              outPtr: sp.outBatch.buffer.wasmPtr(sp.outBatch),
              rows,
              oStrideBytes: rows * 4,
            };
          })
        : [];
      const activeQ8K = groupUseQ8K && qkPtr >= 0;
      const activeQ8 = !activeQ8K && groupUseQ8 && qPtr >= 0;

      return {
        op: GgmlOp.MUL_MAT,
        label,
        shapeDesc: `[${totalRows}×${cols}] · [${B}×${cols}] (vRAM fused ${specs.length}×)`,
        srcType: activeQ8K
          ? `${trait.name}×Q8_K B${B} ${arena?.relaxedSimdActive ? 'VNNI' : 'Int8'}`
          : activeQ8
            ? `${trait.name}×Q8 B${B} ${arena?.relaxedSimdActive ? 'VNNI' : 'Int8'}`
            : `${trait.name}×F32 B${B}`,
        execute: () => {
          const pub = batchActivePub.current!;
          if (useWasm && arena) {
            let allWasm = true;
            for (let i = 0; i < specs.length; i++) {
              const w = specs[i].selectHandle(pub);
              if (!w.buffer.isWasmBacked) {
                allWasm = false;
                break;
              }
              entries[i].wPtr = w.buffer.wasmPtr(w);
            }
            if (allWasm) {
              if (activeQ8K) {
                arena.mulMatBatch4Q8KGroup(entries, qkPtr, xPtr, cols);
              } else {
                arena.mulMatBatch4Q8Group(entries, activeQ8 ? qPtr : xPtr, xPtr, cols, activeQ8);
              }
              return;
            }
          }
          const xAll = xBatch.buffer.f32View(xBatch);
          for (let i = 0; i < specs.length; i++) {
            const w = specs[i].selectHandle(pub);
            const rows = w.dims[1] ?? 1;
            const oAll = specs[i].outBatch.buffer.f32View(specs[i].outBatch);
            const wRaw = w.buffer.rawBytes();
            for (let t = 0; t < B; t++) {
              verdantMulMat(
                wRaw,
                w.byteOffset,
                w.type,
                cols,
                rows,
                xAll.subarray(t * cols, (t + 1) * cols),
                oAll.subarray(t * rows, (t + 1) * rows),
              );
            }
          }
        },
      };
    };

    const batchEmbedNode: GgmlGraphNode = {
      op: GgmlOp.GET_ROWS,
      label: 'token_embd.get_rows_b4',
      shapeDesc: wp.streamEmbeddings ? `[${B}×${dim}] <- OPFS` : `[${B}×${dim}]`,
      srcType: wp.streamEmbeddings ? `${embdTrait.name} OPFS` : embdTrait.name,
      execute: () => {
        const dst = bs.x.buffer.f32View(bs.x);
        if (pinnedTokenEmbd) {
          const raw = pinnedTokenEmbd.buffer.rawBytes();
          for (let t = 0; t < B; t++) {
            verdantDecodeRow(
              raw,
              pinnedTokenEmbd.byteOffset + batchState.tokens[t] * embdRowBytes,
              pinnedTokenEmbd.type,
              dim,
              dst,
              t * dim,
            );
          }
        } else {
          for (let t = 0; t < B; t++) {
            const { handle, rowIndex } = wp.acquireEmbeddingRow(batchState.tokens[t]);
            verdantDecodeRow(
              handle.buffer.rawBytes(),
              handle.byteOffset + rowIndex * embdRowBytes,
              handle.type,
              dim,
              dst,
              t * dim,
            );
          }
        }
      },
    };
    batchNodes.push(batchEmbedNode);
    const batchLayerAcquireNodes: Array<GgmlGraphNode | null> = new Array(layerCount).fill(null);
    const batchLayerBodyNodes: GgmlGraphNode[][] = new Array(layerCount);
    const batchAsyncLayers: AsyncLayerPlan[] = [];

    for (let l = 0; l < layerCount; l++) {
      const layerIdx = l;
      const info = this.layerInfos[l];
      const hot = isPaged ? wp.getHotLayer(l) : residentLayers![l];
      const lw = isPaged ? null : residentLayers![l];
      const bqBasePtr = bs.q.buffer.wasmPtr(bs.q);
      const bAttnOutBasePtr = bs.attnOut.buffer.wasmPtr(bs.attnOut);

      if (isPaged) {
        // Amortizes ONE OPFS coalesced layer restore across all B* = 4 (or Layer-Major C = 32) prefill tokens!
        const acqNode: GgmlGraphNode = {
          op: GgmlOp.CPY,
          label: `blk.${l}.vram_dimm_acquire_b4`,
          shapeDesc: `vRAM DIMM L${l} -> B${B} frame`,
          srcType:
            wp.backingKind === 'opfs-sync'
              ? 'OPFS DIMM B4'
              : wp.backingKind === 'opfs-demand'
                ? 'OPFS Demand B4'
                : wp.backingKind === 'local-file-sync'
                  ? 'Local Disk B4'
                  : 'vRAM DIMM B4',
          execute: () => {
            batchActivePub.current = wp.acquireLayer(layerIdx);
          },
        };
        batchLayerAcquireNodes[l] = acqNode;
        batchNodes.push(acqNode);
      }
      const layerBodyStartIdx = batchNodes.length;

      batchNodes.push(makeBatchRmsNorm(`blk.${l}.attn_norm_b4`, bs.x, hot.attnNorm, bs.xNorm));
      const allAttnQ8K =
        canQ8K(info.attnQ.type, dim, bs.q8kDim) &&
        canQ8K(info.attnK.type, dim, bs.q8kDim) &&
        canQ8K(info.attnV.type, dim, bs.q8kDim);
      const anyAttnQ8 =
        !allAttnQ8K &&
        useQ8 &&
        bs.q8Dim !== null &&
        arena !== null &&
        (arena.supportsQ8Activation(info.attnQ.type) ||
          arena.supportsQ8Activation(info.attnK.type) ||
          arena.supportsQ8Activation(info.attnV.type));
      if (allAttnQ8K && bs.q8kDim) {
        batchNodes.push(makeBatchQuantizeQ8K(`blk.${l}.q8k_attn_in_b4`, bs.xNorm, bs.q8kDim, dim));
        batch4Tally.q8kOps += 3;
        batch4Tally.q8kSb += (qDim + kvDim + kvDim) * (dim >>> 8) * B;
      } else if (anyAttnQ8 && bs.q8Dim) {
        batchNodes.push(makeBatchQuantizeQ8(`blk.${l}.q8_attn_in_b4`, bs.xNorm, bs.q8Dim, dim));
        batch4Tally.q80Ops += 3;
      } else {
        batch4Tally.strictOps += 3;
      }
      batch4Tally.fusedBarriers += 1;

      batchNodes.push(
        isPaged
          ? makePagedBatchLinearGroup(
              `blk.${l}.attn_qkv_b4`,
              [
                { info: info.attnQ, selectHandle: (p) => p.attnQ, outBatch: bs.q },
                { info: info.attnK, selectHandle: (p) => p.attnK, outBatch: bs.k },
                { info: info.attnV, selectHandle: (p) => p.attnV, outBatch: bs.v },
              ],
              bs.q8Dim,
              bs.q8kDim,
              bs.xNorm,
              anyAttnQ8,
              allAttnQ8K,
            )
          : makeBatchLinearGroup(
              `blk.${l}.attn_qkv_b4`,
              [
                { weight: lw!.attnQ, outBatch: bs.q },
                { weight: lw!.attnK, outBatch: bs.k },
                { weight: lw!.attnV, outBatch: bs.v },
              ],
              bs.q8Dim,
              bs.q8kDim,
              bs.xNorm,
              anyAttnQ8,
              allAttnQ8K,
            ),
      );
      if (hot.attnQBias) {
        batchNodes.push(makeBatchBiasAdd(`blk.${l}.attn_q_bias_b4`, bs.q, hot.attnQBias, qDim));
      }
      if (hot.attnKBias) {
        batchNodes.push(makeBatchBiasAdd(`blk.${l}.attn_k_bias_b4`, bs.k, hot.attnKBias, kvDim));
      }
      if (hot.attnVBias) {
        batchNodes.push(makeBatchBiasAdd(`blk.${l}.attn_v_bias_b4`, bs.v, hot.attnVBias, kvDim));
      }
      const bkBasePtr = bs.k.buffer.wasmPtr(bs.k);
      const bvBasePtr = bs.v.buffer.wasmPtr(bs.v);
      const useWasmBatchAct =
        arena !== null && bqBasePtr >= 0 && bkBasePtr >= 0 && bvBasePtr >= 0 && bAttnOutBasePtr >= 0;
      const qStrideBytes = qDim * 4;
      const kvStrideBytes = kvDim * 4;

      if (hot.attnQNorm) {
        const qNormHandle = hot.attnQNorm;
        const qNormWPtr = useWasmBatchAct && qNormHandle.buffer.isWasmBacked ? qNormHandle.buffer.wasmPtr(qNormHandle) : -1;
        batchNodes.push({
          op: GgmlOp.RMS_NORM,
          label: `blk.${l}.attn_q_norm_b4`,
          shapeDesc: `[${B}×${hp.headCount}×${headDim}]`,
          srcType: qNormWPtr >= 0 ? 'F32 SIMD B4' : 'F32',
          execute: () => {
            if (arena && qNormWPtr >= 0) {
              for (let t = 0; t < B; t++) {
                arena.kernels.rms_norm_per_head_f32(
                  bqBasePtr + t * qStrideBytes,
                  qNormWPtr,
                  hp.headCount,
                  headDim,
                  hp.rmsNormEps,
                );
              }
            } else {
              const qAll = bs.q.buffer.f32View(bs.q);
              const w = qNormHandle.buffer.f32View(qNormHandle);
              for (let t = 0; t < B; t++) {
                verdantRmsNormPerHead(qAll, w, hp.headCount, headDim, hp.rmsNormEps, t * qDim);
              }
            }
          },
        });
      }
      if (hot.attnKNorm) {
        const kNormHandle = hot.attnKNorm;
        const kNormWPtr = useWasmBatchAct && kNormHandle.buffer.isWasmBacked ? kNormHandle.buffer.wasmPtr(kNormHandle) : -1;
        batchNodes.push({
          op: GgmlOp.RMS_NORM,
          label: `blk.${l}.attn_k_norm_b4`,
          shapeDesc: `[${B}×${hp.headCountKv}×${headDim}]`,
          srcType: kNormWPtr >= 0 ? 'F32 SIMD B4' : 'F32',
          execute: () => {
            if (arena && kNormWPtr >= 0) {
              for (let t = 0; t < B; t++) {
                arena.kernels.rms_norm_per_head_f32(
                  bkBasePtr + t * kvStrideBytes,
                  kNormWPtr,
                  hp.headCountKv,
                  headDim,
                  hp.rmsNormEps,
                );
              }
            } else {
              const kAll = bs.k.buffer.f32View(bs.k);
              const w = kNormHandle.buffer.f32View(kNormHandle);
              for (let t = 0; t < B; t++) {
                verdantRmsNormPerHead(kAll, w, hp.headCountKv, headDim, hp.rmsNormEps, t * kvDim);
              }
            }
          },
        });
      }

      batchNodes.push(
        {
          op: GgmlOp.ROPE,
          label: `blk.${l}.rope_qk_b4`,
          shapeDesc: `[${B}×(${hp.headCount}+${hp.headCountKv})×${headDim}]`,
          srcType: canWasmRope && useWasmBatchAct ? 'F32 SIMD B4' : 'F32',
          execute: () => {
            if (canWasmRope && useWasmBatchAct && arena) {
              for (let t = 0; t < B; t++) {
                const pOff = resolveRopeOffsetBytes(batchState.startPos + t, t);
                arena.kernels.rope_f32(
                  bqBasePtr + t * qStrideBytes,
                  ropeCosPtr + pOff,
                  ropeSinPtr + pOff,
                  hp.headCount,
                  headDim,
                  neoxFlag,
                );
                arena.kernels.rope_f32(
                  bkBasePtr + t * kvStrideBytes,
                  ropeCosPtr + pOff,
                  ropeSinPtr + pOff,
                  hp.headCountKv,
                  headDim,
                  neoxFlag,
                );
              }
            } else {
              const qAll = bs.q.buffer.f32View(bs.q);
              const kAll = bs.k.buffer.f32View(bs.k);
              for (let t = 0; t < B; t++) {
                const p = batchState.startPos + t;
                verdantRopePrecomputed(
                  qAll.subarray(t * qDim, (t + 1) * qDim),
                  hp.headCount,
                  headDim,
                  p,
                  rope,
                  useNeoxRope,
                );
                verdantRopePrecomputed(
                  kAll.subarray(t * kvDim, (t + 1) * kvDim),
                  hp.headCountKv,
                  headDim,
                  p,
                  rope,
                  useNeoxRope,
                );
              }
            }
          },
        },
        {
          op: GgmlOp.SET_ROWS,
          label: `blk.${l}.kv_cache_set_b4`,
          shapeDesc: `vram_kv[${l}] <- [${B}×${kvDim}]`,
          srcType: kvTierLabel,
          execute: () => {
            if (useWasmBatchAct && arena) {
              for (let t = 0; t < B; t++) {
                const p = batchState.startPos + t;
                kv.writeKvPtr(false, layerIdx, p, bkBasePtr + t * kvStrideBytes, arena);
                kv.writeKvPtr(true, layerIdx, p, bvBasePtr + t * kvStrideBytes, arena);
              }
            } else {
              const kAll = bs.k.buffer.f32View(bs.k);
              const vAll = bs.v.buffer.f32View(bs.v);
              for (let t = 0; t < B; t++) {
                const p = batchState.startPos + t;
                kv.writeKv(false, layerIdx, p, kAll.subarray(t * kvDim, (t + 1) * kvDim));
                kv.writeKv(true, layerIdx, p, vAll.subarray(t * kvDim, (t + 1) * kvDim));
              }
            }
          },
        },
        {
          op: GgmlOp.FLASH_ATTN_EXT,
          label: `blk.${l}.flash_attn_ext_b4`,
          shapeDesc: `[${B}×${hp.headCount}×${headDim}] × vRAM KV`,
          srcType: attnLabel,
          execute: () => {
            if (useWasmBatchAct && arena) {
              for (let t = 0; t < B; t++) {
                const seqLen = batchState.startPos + t + 1;
                kv.flashAttnExt(
                  emptyF32,
                  bqBasePtr + t * qStrideBytes,
                  emptyF32,
                  bAttnOutBasePtr + t * qStrideBytes,
                  hp.headCount,
                  layerIdx,
                  seqLen,
                  arena,
                );
              }
            } else {
              const qAll = bs.q.buffer.f32View(bs.q);
              const oAll = bs.attnOut.buffer.f32View(bs.attnOut);
              for (let t = 0; t < B; t++) {
                const seqLen = batchState.startPos + t + 1;
                kv.flashAttnExt(
                  qAll.subarray(t * qDim, (t + 1) * qDim),
                  bqBasePtr >= 0 ? bqBasePtr + t * qStrideBytes : -1,
                  oAll.subarray(t * qDim, (t + 1) * qDim),
                  bAttnOutBasePtr >= 0 ? bAttnOutBasePtr + t * qStrideBytes : -1,
                  hp.headCount,
                  layerIdx,
                  seqLen,
                  arena,
                );
              }
            }
          },
        },
      );

      const attnOutB4Q8K = canQ8K(info.attnOutput.type, qDim, bs.q8kAttnOut);
      if (attnOutB4Q8K && bs.q8kAttnOut) {
        batchNodes.push(
          makeBatchQuantizeQ8K(`blk.${l}.q8k_attn_out_b4`, bs.attnOut, bs.q8kAttnOut, qDim),
        );
        batch4Tally.q8kOps += 1;
        batch4Tally.q8kSb += dim * (qDim >>> 8) * B;
      } else if (useQ8 && bs.q8AttnOut && arena?.supportsQ8Activation(info.attnOutput.type)) {
        batchNodes.push(
          makeBatchQuantizeQ8(`blk.${l}.q8_attn_out_b4`, bs.attnOut, bs.q8AttnOut, qDim),
        );
        batch4Tally.q80Ops += 1;
      } else {
        batch4Tally.strictOps += 1;
      }
      batchNodes.push(
        isPaged
          ? makePagedBatchMulMat(
              `blk.${l}.attn_output_b4`,
              info.attnOutput,
              (p) => p.attnOutput,
              bs.q8AttnOut,
              bs.q8kAttnOut,
              bs.attnOut,
              bs.attnProj,
            )
          : makeBatchMulMat(
              `blk.${l}.attn_output_b4`,
              lw!.attnOutput,
              bs.q8AttnOut,
              bs.q8kAttnOut,
              bs.attnOut,
              bs.attnProj,
            ),
      );
      if (hot.attnOutputBias) {
        batchNodes.push(makeBatchBiasAdd(`blk.${l}.attn_output_bias_b4`, bs.attnProj, hot.attnOutputBias, dim));
      }
      const bxPtr = bs.x.buffer.wasmPtr(bs.x);
      const bAttnProjPtr = bs.attnProj.buffer.wasmPtr(bs.attnProj);
      const bFfnProjPtr = bs.ffnProj.buffer.wasmPtr(bs.ffnProj);
      const useBatchAddSimd =
        arena !== null && bxPtr >= 0 && bAttnProjPtr >= 0 && bFfnProjPtr >= 0;

      batchNodes.push(
        {
          op: GgmlOp.ADD,
          label: `blk.${l}.attn_residual_b4`,
          shapeDesc: `[${B}×${dim}]`,
          srcType: useBatchAddSimd ? 'F32 SIMD' : 'F32',
          execute: () => {
            if (useBatchAddSimd && arena) {
              arena.kernels.add_f32(bxPtr, bAttnProjPtr, bxPtr, B * dim);
            } else {
              const bx = bs.x.buffer.f32View(bs.x);
              verdantAdd(bx, bs.attnProj.buffer.f32View(bs.attnProj), bx, B * dim);
            }
          },
        },
        makeBatchRmsNorm(`blk.${l}.ffn_norm_b4`, bs.x, hot.ffnNorm, bs.xNorm),
      );

      const batchPreMoeNodes = batchNodes.slice(layerBodyStartIdx);

      if (info.ffnGateInp && info.ffnGateExps && info.ffnUpExps && info.ffnDownExps) {
        const expFfnDim = info.ffnGateExps.dims[1] ?? ffnDim;
        const nExps = info.ffnGateExps.dims[2] ?? hp.expertCount ?? 1;
        const expTrait = getGgmlTypeTrait(info.ffnGateExps.type);
        const batchResidualNode: GgmlGraphNode = {
          op: GgmlOp.ADD,
          label: `blk.${l}.ffn_residual_b4`,
          shapeDesc: `[${B}×${dim}]`,
          srcType: useBatchAddSimd ? 'F32 SIMD' : 'F32',
          execute: () => {
            if (useBatchAddSimd && arena) {
              arena.kernels.add_f32(bxPtr, bFfnProjPtr, bxPtr, B * dim);
            } else {
              const bx = bs.x.buffer.f32View(bs.x);
              verdantAdd(bx, bs.ffnProj.buffer.f32View(bs.ffnProj), bx, B * dim);
            }
          },
        };
        batchNodes.push(
          {
            op: GgmlOp.MUL_MAT,
            label: `blk.${l}.ffn_moe_sparse_b4`,
            shapeDesc: `B${B} × Top-${Math.min(moeTopK, nExps)}/${nExps} × [${expFfnDim}×${dim}] (vRAM MoE)`,
            srcType: `${expTrait.name} Sparse MoE B${B}`,
            execute: () => {
              const routerW = isPaged ? batchActivePub.current!.ffnGateInp! : lw!.ffnGateInp!;
              runSparseMoeBatch4(layerIdx, routerW, expFfnDim);
            },
          },
          batchResidualNode,
        );
        batchAsyncLayers.push({
          layerIdx,
          acquireNode: batchLayerAcquireNodes[l],
          preMoeNodes: batchPreMoeNodes,
          moe: {
            routeAndGetExperts: () => {
              const routerW = isPaged ? batchActivePub.current!.ffnGateInp! : lw!.ffnGateInp!;
              const uniqueCount = routeSparseMoeBatch4(routerW);
              const expertIds: number[] = [];
              for (let u = 0; u < uniqueCount; u++) {
                expertIds.push(moeBatchExpIds[u]);
              }
              return { count: uniqueCount, expertIds };
            },
            executeRouted: (uniqueCount: number) => {
              executeSparseMoeBatch4Routed(layerIdx, expFfnDim, uniqueCount);
            },
            residualNode: batchResidualNode,
          },
          denseFfnNodes: [],
        });
      } else {
        const batchDenseStartIdx = batchNodes.length;
        const ffnGateInfo = info.ffnGate!;
        const ffnUpInfo = info.ffnUp!;
        const ffnDownInfo = info.ffnDown!;
        const allFfnInB4Q8K =
          canQ8K(ffnGateInfo.type, dim, bs.q8kDim) && canQ8K(ffnUpInfo.type, dim, bs.q8kDim);
        const anyFfnInQ8 =
          !allFfnInB4Q8K &&
          useQ8 &&
          bs.q8Dim !== null &&
          arena !== null &&
          (arena.supportsQ8Activation(ffnGateInfo.type) || arena.supportsQ8Activation(ffnUpInfo.type));
        if (allFfnInB4Q8K && bs.q8kDim) {
          batchNodes.push(makeBatchQuantizeQ8K(`blk.${l}.q8k_ffn_in_b4`, bs.xNorm, bs.q8kDim, dim));
          batch4Tally.q8kOps += 2;
          batch4Tally.q8kSb += (ffnDim * 2) * (dim >>> 8) * B;
        } else if (anyFfnInQ8 && bs.q8Dim) {
          batchNodes.push(makeBatchQuantizeQ8(`blk.${l}.q8_ffn_in_b4`, bs.xNorm, bs.q8Dim, dim));
          batch4Tally.q80Ops += 2;
        } else {
          batch4Tally.strictOps += 2;
        }
        batch4Tally.fusedBarriers += 1;

        batchNodes.push(
          isPaged
            ? makePagedBatchLinearGroup(
                `blk.${l}.ffn_gate_up_b4`,
                [
                  { info: ffnGateInfo, selectHandle: (p) => p.ffnGate!, outBatch: bs.ffnGate },
                  { info: ffnUpInfo, selectHandle: (p) => p.ffnUp!, outBatch: bs.ffnUp },
                ],
                bs.q8Dim,
                bs.q8kDim,
                bs.xNorm,
                anyFfnInQ8,
                allFfnInB4Q8K,
              )
            : makeBatchLinearGroup(
                `blk.${l}.ffn_gate_up_b4`,
                [
                  { weight: lw!.ffnGate!, outBatch: bs.ffnGate },
                  { weight: lw!.ffnUp!, outBatch: bs.ffnUp },
                ],
                bs.q8Dim,
                bs.q8kDim,
                bs.xNorm,
                anyFfnInQ8,
                allFfnInB4Q8K,
              ),
          {
            op: GgmlOp.SWIGLU,
            label: `blk.${l}.ffn_swiglu_b4`,
            shapeDesc: `[${B}×${ffnDim}]`,
            srcType: useBatchAddSimd ? 'F32 SIMD' : 'F32',
            execute: () => {
              if (useBatchAddSimd && arena) {
                arena.kernels.swiglu_f32(
                  bs.ffnGate.buffer.wasmPtr(bs.ffnGate),
                  bs.ffnUp.buffer.wasmPtr(bs.ffnUp),
                  bs.ffnHidden.buffer.wasmPtr(bs.ffnHidden),
                  B * ffnDim,
                );
              } else {
                verdantSwiGlu(
                  bs.ffnGate.buffer.f32View(bs.ffnGate),
                  bs.ffnUp.buffer.f32View(bs.ffnUp),
                  bs.ffnHidden.buffer.f32View(bs.ffnHidden),
                  B * ffnDim,
                );
              }
            },
          },
        );

        const ffnDownB4Q8K = canQ8K(ffnDownInfo.type, ffnDim, bs.q8kFfn);
        if (ffnDownB4Q8K && bs.q8kFfn) {
          batchNodes.push(makeBatchQuantizeQ8K(`blk.${l}.q8k_ffn_down_b4`, bs.ffnHidden, bs.q8kFfn, ffnDim));
          batch4Tally.q8kOps += 1;
          batch4Tally.q8kSb += dim * (ffnDim >>> 8) * B;
        } else if (useQ8 && bs.q8Ffn && arena?.supportsQ8Activation(ffnDownInfo.type)) {
          batchNodes.push(makeBatchQuantizeQ8(`blk.${l}.q8_ffn_down_b4`, bs.ffnHidden, bs.q8Ffn, ffnDim));
          batch4Tally.q80Ops += 1;
        } else {
          batch4Tally.strictOps += 1;
        }
        batchNodes.push(
          isPaged
            ? makePagedBatchMulMat(
                `blk.${l}.ffn_down_b4`,
                ffnDownInfo,
                (p) => p.ffnDown!,
                bs.q8Ffn,
                bs.q8kFfn,
                bs.ffnHidden,
                bs.ffnProj,
              )
            : makeBatchMulMat(
                `blk.${l}.ffn_down_b4`,
                lw!.ffnDown!,
                bs.q8Ffn,
                bs.q8kFfn,
                bs.ffnHidden,
                bs.ffnProj,
              ),
          {
            op: GgmlOp.ADD,
            label: `blk.${l}.ffn_residual_b4`,
            shapeDesc: `[${B}×${dim}]`,
            srcType: useBatchAddSimd ? 'F32 SIMD' : 'F32',
            execute: () => {
              if (useBatchAddSimd && arena) {
                arena.kernels.add_f32(bxPtr, bFfnProjPtr, bxPtr, B * dim);
              } else {
                const bx = bs.x.buffer.f32View(bs.x);
                verdantAdd(bx, bs.ffnProj.buffer.f32View(bs.ffnProj), bx, B * dim);
              }
            },
          },
        );
        batchAsyncLayers.push({
          layerIdx,
          acquireNode: batchLayerAcquireNodes[l],
          preMoeNodes: batchPreMoeNodes,
          denseFfnNodes: batchNodes.slice(batchDenseStartIdx),
        });
      }
      batchLayerBodyNodes[l] = batchNodes.slice(layerBodyStartIdx);
    }

    this.compiledBatch4Graph = b.compileGraph(batchNodes);
    this.batchAsyncPlan = {
      embedNode: batchEmbedNode,
      layers: batchAsyncLayers,
    };
    this.batchEmbedNode = batchEmbedNode;
    this.batchLayerAcquireNodes = batchLayerAcquireNodes;
    this.batchLayerBodyNodes = batchLayerBodyNodes;
    this.prefillChunkX = isPaged
      ? new Float32Array(PREFILL_LAYER_MAJOR_MAX_MICROBATCHES * B * dim)
      : null;
    this.prismBatch4Tally = batch4Tally;
  }

  resetKvCache(): void {
    this.kvPager?.reset();
    this.cachedTokens = [];
    this.committedPageKeys.clear();
  }

  /**
   * Aligns the live vRAM KV pager with `tokens` up to `maxReusableCap`:
   * 1. Reuses every matching token already live in `kvPager`.
   * 2. If `VramUnifiedStore` holds additional contiguous 16-token compressed Q8 prefix blocks
   *    beyond the live match (e.g., across threads or hydrated from OPFS), restores them directly
   *    into Tier-1 Warm Q8 pages in O(1) without recomputing transformer layers!
   */
  private alignAndRestorePrefix(
    tokens: readonly number[],
    maxReusableCap: number,
  ): { commonPrefix: number; reusedLive: number; restoredUnified: number } {
    const kv = this.kvPager!;
    const cap = Math.max(0, Math.min(tokens.length, maxReusableCap));

    let liveMatch = 0;
    const maxLive = Math.min(this.cachedTokens.length, cap);
    while (liveMatch < maxLive && this.cachedTokens[liveMatch] === tokens[liveMatch]) {
      liveMatch += 1;
    }

    let reusedLive = liveMatch;
    let restoredUnified = 0;

    if (this.unifiedStore && cap >= VRAM_KV_PAGE_TOKENS) {
      const chain = this.unifiedStore.lookupPrefixChain(tokens, cap);
      const unifiedReach = chain.length * VRAM_KV_PAGE_TOKENS;
      if (unifiedReach > liveMatch) {
        const startPage = Math.floor(liveMatch / VRAM_KV_PAGE_TOKENS);
        const keepTokens = startPage * VRAM_KV_PAGE_TOKENS;
        if (keepTokens < this.cachedTokens.length) {
          kv.truncateTo(keepTokens);
          this.cachedTokens.length = keepTokens;
        }
        reusedLive = keepTokens;

        for (let p = startPage; p < chain.length; p++) {
          const entry = chain[p];
          const ok = kv.importQ8BlockIntoPage(p, entry.data);
          if (!ok) break;
          for (let t = 0; t < VRAM_KV_PAGE_TOKENS; t++) {
            this.cachedTokens.push(tokens[p * VRAM_KV_PAGE_TOKENS + t]);
          }
          this.committedPageKeys.add(entry.blockKey);
          restoredUnified += VRAM_KV_PAGE_TOKENS;
        }
        return {
          commonPrefix: reusedLive + restoredUnified,
          reusedLive,
          restoredUnified,
        };
      }
    }

    if (liveMatch < this.cachedTokens.length) {
      kv.truncateTo(liveMatch);
      this.cachedTokens.length = liveMatch;
    }
    return {
      commonPrefix: liveMatch,
      reusedLive: liveMatch,
      restoredUnified: 0,
    };
  }

  /**
   * Commits any newly completed 16-token KV pages from live `kvPager` into the content-addressed
   * `VramUnifiedStore` so subsequent turns, threads, and browser reloads reuse them instantaneously.
   */
  private commitFullPagesToUnifiedStore(tokens: readonly number[]): number {
    if (!this.unifiedStore || !this.kvPager || tokens.length < VRAM_KV_PAGE_TOKENS) return 0;
    const keys = this.unifiedStore.computePrefixKeys(tokens);
    let committed = 0;
    for (const { pageIndex, key, slice } of keys) {
      if (this.committedPageKeys.has(key)) continue;
      const block = this.kvPager.exportPageAsQ8Block(pageIndex);
      if (!block) continue;
      this.unifiedStore.putBlock(key, pageIndex, slice, block);
      this.committedPageKeys.add(key);
      committed += 1;
    }
    return committed;
  }

  /**
   * Continuous Speculative Prefill Engine:
   * Cooperatively pre-warms system prompts and live composer drafts into `kvPager` +
   * `VramUnifiedStore` without running the LM head (`computeLogits = false`), yielding between
   * batches so user submission preempts in milliseconds while keeping every token already warmed!
   */
  async speculativePrefill(
    prompt: string,
    signal?: AbortSignal,
    onProgress?: (progress: SpeculativePrefillProgress) => void,
    threadId?: string,
  ): Promise<SpeculativePrefillResult> {
    // Never trigger network downloads during background speculative prefill:
    // on `opfs-demand`, run ONLY once all active layers are already 100% resident in local OPFS!
    const isDemandBacking = this.backingStore?.kind === 'opfs-demand';
    if (
      !this.tokenizer ||
      !this.hparams ||
      !this.kvPager ||
      !prompt ||
      (isDemandBacking && !this.isDemandModelFullyWarm())
    ) {
      return {
        targetTokens: 0,
        readyTokens: 0,
        reusedLiveTokens: 0,
        restoredUnifiedTokens: 0,
        newlyPrefilledTokens: 0,
        aborted: false,
      };
    }

    const encoded = this.tokenizer.encode(prompt, this.tokenizer.addBosToken);
    const safeCap = Math.max(0, this.maxContextWindow - 1);
    const targetTokens = encoded.length > safeCap ? encoded.slice(0, safeCap) : encoded;
    if (targetTokens.length === 0) {
      return {
        targetTokens: 0,
        readyTokens: 0,
        reusedLiveTokens: 0,
        restoredUnifiedTokens: 0,
        newlyPrefilledTokens: 0,
        aborted: false,
      };
    }

    // Strict 0-network guarantee on warm `opfs-demand`: if any prompt token's embedding row slab
    // is not yet cached locally in OPFS, skip speculative prefill without touching the network.
    if (
      isDemandBacking &&
      this.weightPager &&
      this.backingStore instanceof OpfsDemandVramBackingStore
    ) {
      const embdMissing = this.weightPager.getEmbeddingMissingRanges(targetTokens);
      for (let k = 0; k < embdMissing.length; k++) {
        if (!this.backingStore.isRangeCached(embdMissing[k].fileOffset, embdMissing[k].byteLength)) {
          return {
            targetTokens: 0,
            readyTokens: 0,
            reusedLiveTokens: 0,
            restoredUnifiedTokens: 0,
            newlyPrefilledTokens: 0,
            aborted: false,
          };
        }
      }
    }

    this.unifiedStore?.setSpeculativeActive(true);
    let i = 0;
    let newlyPrefilled = 0;
    let reusedLive = 0;
    let restoredUnified = 0;
    let aborted = false;

    try {
      // Strict thread isolation: wipe live KV frames whenever switching threads so partial tail
      // K/V state from another thread can NEVER bleed into this thread (shared system prompt blocks
      // still restore cleanly from content-addressed VramUnifiedStore).
      if (threadId && this.activeThreadId !== null && threadId !== this.activeThreadId) {
        this.resetKvCache();
      }
      if (threadId) {
        this.activeThreadId = threadId;
      }

      const aligned = this.alignAndRestorePrefix(targetTokens, targetTokens.length);
      i = aligned.commonPrefix;
      reusedLive = aligned.reusedLive;
      restoredUnified = aligned.restoredUnified;

      onProgress?.({ readyTokens: i, targetTokens: targetTokens.length });

      const batchWidth = this.performanceContract?.optimalPrefillBatchSize ?? PREFILL_BATCH_SIZE;
      this.frameBudget.beginSlice();

      while (i < targetTokens.length) {
        if (signal?.aborted) {
          aborted = true;
          break;
        }

        const tStep0 = performance.now();
        const chunkBatches =
          this.backend.isWasmSimdActive && batchWidth >= PREFILL_BATCH_SIZE
            ? this.computeLayerMajorMicroBatches(i, targetTokens.length)
            : 0;
        if (chunkBatches >= 2) {
          const chunkTokens = chunkBatches * PREFILL_BATCH_SIZE;
          this.evalPrefillChunkLayerMajor(targetTokens, i, i, chunkBatches);
          for (let k = 0; k < chunkTokens; k++) {
            this.cachedTokens.push(targetTokens[i + k]);
          }
          i += chunkTokens;
          newlyPrefilled += chunkTokens;
          this.commitFullPagesToUnifiedStore(this.cachedTokens);
          onProgress?.({ readyTokens: i, targetTokens: targetTokens.length });
          const stepUs = (performance.now() - tStep0) * 1000;
          if (
            i < targetTokens.length &&
            this.frameBudget.recordStepAndShouldYield(stepUs, SPECULATIVE_SLICE_BUDGET_US, 2)
          ) {
            await this.yielder.yieldNow();
          }
          continue;
        }
        if (
          this.backend.isWasmSimdActive &&
          batchWidth >= PREFILL_BATCH_SIZE &&
          i + PREFILL_BATCH_SIZE <= targetTokens.length
        ) {
          this.evalPrefillBatch4(targetTokens, i, i);
          for (let k = 0; k < PREFILL_BATCH_SIZE; k++) {
            this.cachedTokens.push(targetTokens[i + k]);
          }
          i += PREFILL_BATCH_SIZE;
          newlyPrefilled += PREFILL_BATCH_SIZE;
          this.commitFullPagesToUnifiedStore(this.cachedTokens);
          onProgress?.({ readyTokens: i, targetTokens: targetTokens.length });
          const stepUs = (performance.now() - tStep0) * 1000;
          if (
            i < targetTokens.length &&
            this.frameBudget.recordStepAndShouldYield(stepUs, SPECULATIVE_SLICE_BUDGET_US, 4)
          ) {
            await this.yielder.yieldNow();
          }
          continue;
        }

        this.evalToken(targetTokens[i], i, false, false);
        this.cachedTokens.push(targetTokens[i]);
        i += 1;
        newlyPrefilled += 1;

        if (i % VRAM_KV_PAGE_TOKENS === 0) {
          this.commitFullPagesToUnifiedStore(this.cachedTokens);
        }
        onProgress?.({ readyTokens: i, targetTokens: targetTokens.length });

        const stepUs = (performance.now() - tStep0) * 1000;
        if (
          i < targetTokens.length &&
          this.frameBudget.recordStepAndShouldYield(stepUs, SPECULATIVE_SLICE_BUDGET_US, 4)
        ) {
          await this.yielder.yieldNow();
        }
      }

      this.commitFullPagesToUnifiedStore(this.cachedTokens);
      if (newlyPrefilled > 0 || restoredUnified > 0 || reusedLive > 0) {
        this.unifiedStore?.recordSpeculativePrefill(newlyPrefilled, i);
      }

      return {
        targetTokens: targetTokens.length,
        readyTokens: i,
        reusedLiveTokens: reusedLive,
        restoredUnifiedTokens: restoredUnified,
        newlyPrefilledTokens: newlyPrefilled,
        aborted,
      };
    } finally {
      this.kvPager?.truncateTo(this.cachedTokens.length);
      this.unifiedStore?.setSpeculativeActive(false);
    }
  }

  /**
   * Executes the compiled reusable single-token ggml_cgraph exclusively through vTensor.
   * When `computeLogits` is false (non-final prefill tokens), skips the LM head projection
   * (`output_norm` + `output.weight`) with 100% mathematical exactness.
   */
  evalToken(
    tokenId: number,
    pos: number,
    captureTrace = false,
    computeLogits = true,
  ): Float32Array {
    if (!this.compiledSingleGraph || !this.scratch) {
      throw new Error('Cannot evaluate token before a GGUF model is loaded.');
    }

    if (pos < 0 || pos >= this.maxContextWindow) {
      throw new Error(
        `Token position ${pos} exceeds vTensor KV-cache context window (${this.maxContextWindow}).`,
      );
    }

    this.singleExec.tokenId = tokenId;
    this.singleExec.pos = pos;
    this.singleExec.seqLen = pos + 1;
    this.singleExec.externalEmbedding = null;

    this.backend.executeCompiledGraph(this.compiledSingleGraph, {
      captureTrace,
      profilePerOp: captureTrace,
      nodeLimit: computeLogits ? undefined : this.singleGraphBodyNodeCount,
    });

    const bt = this.prismSingleBodyTally;
    this.prismLedger.q8kDispatches += bt.q8kOps;
    this.prismLedger.q8kSuperBlocks += bt.q8kSb;
    this.prismLedger.q80Dispatches += bt.q80Ops;
    this.prismLedger.strictF32Dispatches += bt.strictOps;
    this.prismLedger.fusedGroupBarriers += bt.fusedBarriers;
    if (computeLogits) {
      const lt = this.prismSingleLmTally;
      this.prismLedger.q8kDispatches += lt.q8kOps;
      this.prismLedger.q8kSuperBlocks += lt.q8kSb;
      this.prismLedger.q80Dispatches += lt.q80Ops;
      this.prismLedger.strictF32Dispatches += lt.strictOps;
    }

    return this.scratch.logits.buffer.f32View(this.scratch.logits);
  }

  /**
   * Decodes one token's F32 embedding vector directly from the model's `token_embd` tensor.
   * Useful for multimodal interleaving and exact verification tests.
   */
  decodeTokenEmbedding(tokenId: number): Float32Array {
    if (!this.hparams || !this.weightPager) {
      throw new Error('Cannot decode token embedding before a GGUF model is loaded.');
    }
    const dim = this.hparams.embeddingLength;
    const out = new Float32Array(dim);
    const pinned = this.weightPager.getPinnedTokenEmbd();
    if (pinned) {
      const trait = getGgmlTypeTrait(pinned.type);
      const rowBytes = (dim / trait.blockSize) * trait.typeSizeBytes;
      verdantDecodeRow(pinned.buffer.rawBytes(), pinned.byteOffset + tokenId * rowBytes, pinned.type, dim, out);
      return out;
    }
    const { handle, rowIndex } = this.weightPager.acquireEmbeddingRow(tokenId);
    const trait = getGgmlTypeTrait(handle.type);
    const rowBytes = (dim / trait.blockSize) * trait.typeSizeBytes;
    verdantDecodeRow(handle.buffer.rawBytes(), handle.byteOffset + rowIndex * rowBytes, handle.type, dim, out);
    return out;
  }

  /**
   * Injects a real external multimodal embedding vector (e.g. from Vision `SigLIP` `mmproj`)
   * directly into Layer 0 of the compiled `vTensor` graph at position `pos` with zero
   * serialization and bit-exact KV-cache continuity.
   */
  evalEmbedding(
    embedding: Float32Array,
    pos: number,
    computeLogits = false,
  ): Float32Array {
    if (!this.compiledSingleGraph || !this.scratch || !this.hparams) {
      throw new Error('Cannot evaluate external embedding before a GGUF model is loaded.');
    }
    const dim = this.hparams.embeddingLength;
    if (embedding.length !== dim) {
      throw new Error(
        `Multimodal embedding dimension mismatch: got ${embedding.length}, expected ${dim}.`,
      );
    }
    if (pos < 0 || pos >= this.maxContextWindow) {
      throw new Error(
        `Embedding position ${pos} exceeds vTensor KV-cache context window (${this.maxContextWindow}).`,
      );
    }

    this.singleExec.tokenId = 0;
    this.singleExec.pos = pos;
    this.singleExec.seqLen = pos + 1;
    this.singleExec.externalEmbedding = embedding;
    try {
      this.backend.executeCompiledGraph(this.compiledSingleGraph, {
        captureTrace: false,
        profilePerOp: false,
        nodeLimit: computeLogits ? undefined : this.singleGraphBodyNodeCount,
      });
    } finally {
      this.singleExec.externalEmbedding = null;
    }
    return this.scratch.logits.buffer.f32View(this.scratch.logits);
  }

  /**
   * Prefills a contiguous sequence of multimodal embeddings (e.g. visual tokens from `visionEncoder`)
   * directly into the active KV cache starting at `startPos`, returning the next KV position.
   */
  prefillEmbeddingSequence(
    embeddings: readonly Float32Array[],
    startPos: number,
    computeFinalLogits = false,
  ): { readonly nextPos: number; readonly logits: Float32Array | null } {
    let pos = startPos;
    let lastLogits: Float32Array | null = null;
    for (let i = 0; i < embeddings.length; i++) {
      const isLast = i === embeddings.length - 1;
      const logits = this.evalEmbedding(embeddings[i]!, pos, computeFinalLogits && isLast);
      if (isLast && computeFinalLogits) lastLogits = logits;
      pos += 1;
    }
    return { nextPos: pos, logits: lastLogits };
  }

  /**
   * Executes 4 non-final prefill tokens in a single batched pass through all transformer layers,
   * amortizing quantized weight reads 4× across Q4_0, Q4_K, Q6_K, Q8_0, and F32 and skipping LM head.
   */
  private evalPrefillBatch4(tokens: readonly number[], offset: number, startPos: number): void {
    if (!this.compiledBatch4Graph) {
      throw new Error('Compiled batch-4 graph unavailable.');
    }
    this.batchExec.tokens[0] = tokens[offset];
    this.batchExec.tokens[1] = tokens[offset + 1];
    this.batchExec.tokens[2] = tokens[offset + 2];
    this.batchExec.tokens[3] = tokens[offset + 3];
    this.batchExec.startPos = startPos;
    this.backend.executeCompiledGraph(this.compiledBatch4Graph, {
      captureTrace: false,
      profilePerOp: false,
    });
    const b4 = this.prismBatch4Tally;
    this.prismLedger.q8kDispatches += b4.q8kOps;
    this.prismLedger.q8kSuperBlocks += b4.q8kSb;
    this.prismLedger.q80Dispatches += b4.q80Ops;
    this.prismLedger.strictF32Dispatches += b4.strictOps;
    this.prismLedger.fusedGroupBarriers += b4.fusedBarriers;
  }

  /**
   * Computes how many Batch-4 micro-batches (`2..8`, i.e., `8..32` tokens) can execute in a single
   * Layer-Major OPFS-Paged prefill chunk starting at `startPos` while keeping every touched KV page
   * simultaneously HOT_RESIDENT across all layers (zero mid-chunk partial-layer KV evictions!).
   */
  private computeLayerMajorMicroBatches(startPos: number, endExclusive: number): number {
    if (
      !this.weightPager ||
      this.weightPager.residencyMode !== 'opfs-paged' ||
      !this.kvPager ||
      !this.batchEmbedNode ||
      !this.prefillChunkX ||
      this.batchLayerBodyNodes.length === 0
    ) {
      return 0;
    }
    const maxAvailBatches = Math.min(
      PREFILL_LAYER_MAJOR_MAX_MICROBATCHES,
      Math.floor((endExclusive - startPos) / PREFILL_BATCH_SIZE),
    );
    if (maxAvailBatches < 2) return 0;

    const startPage = Math.floor(startPos / VRAM_KV_PAGE_TOKENS);
    const maxPagesInChunk = Math.max(1, this.kvPager.hotFrameLimit - (startPage > 0 ? 1 : 0));
    let m = maxAvailBatches;
    while (m >= 2) {
      const endPos = startPos + m * PREFILL_BATCH_SIZE - 1;
      const endPage = Math.floor(endPos / VRAM_KV_PAGE_TOKENS);
      if (endPage - startPage + 1 <= maxPagesInChunk) {
        return m;
      }
      m -= 1;
    }
    return 0;
  }

  /**
   * Layer-Major Chunked Prefill for OPFS-Paged models (`C = 4 * microBatches`, up to 32 tokens):
   * Acquires each transformer layer from OPFS ONCE for the entire `C`-token chunk, cutting OPFS
   * weight I/O during multi-token prompt prefill by up to 8× with mathematically bit-exact KV state!
   */
  private evalPrefillChunkLayerMajor(
    tokens: readonly number[],
    offset: number,
    startPos: number,
    microBatches: number,
  ): void {
    const bs = this.batchScratch!;
    const hp = this.hparams!;
    const embedNode = this.batchEmbedNode!;
    const chunkX = this.prefillChunkX!;
    const batchFloats = PREFILL_BATCH_SIZE * hp.embeddingLength;
    const bsXView = bs.x.buffer.f32View(bs.x);

    // 1. Embed all micro-batches into compact chunkX staging slab (<= 640 KB even on 32B!)
    for (let m = 0; m < microBatches; m++) {
      const baseTok = offset + m * PREFILL_BATCH_SIZE;
      this.batchExec.tokens[0] = tokens[baseTok];
      this.batchExec.tokens[1] = tokens[baseTok + 1];
      this.batchExec.tokens[2] = tokens[baseTok + 2];
      this.batchExec.tokens[3] = tokens[baseTok + 3];
      embedNode.execute();
      chunkX.set(bsXView, m * batchFloats);
    }

    // 2. Sweep Layer-Major: restore layer `l` from OPFS ONCE, run all `microBatches` causally!
    const layerCount = this.batchLayerBodyNodes.length;
    for (let l = 0; l < layerCount; l++) {
      const acqNode = this.batchLayerAcquireNodes[l];
      if (acqNode) acqNode.execute();
      const bodyNodes = this.batchLayerBodyNodes[l];
      const nodeCount = bodyNodes.length;
      for (let m = 0; m < microBatches; m++) {
        const sliceStart = m * batchFloats;
        bsXView.set(chunkX.subarray(sliceStart, sliceStart + batchFloats), 0);
        this.batchExec.startPos = startPos + m * PREFILL_BATCH_SIZE;
        for (let n = 0; n < nodeCount; n++) {
          bodyNodes[n].execute();
        }
        chunkX.set(bsXView, sliceStart);
      }
    }

    const b4 = this.prismBatch4Tally;
    this.prismLedger.q8kDispatches += b4.q8kOps * microBatches;
    this.prismLedger.q8kSuperBlocks += b4.q8kSb * microBatches;
    this.prismLedger.q80Dispatches += b4.q80Ops * microBatches;
    this.prismLedger.strictF32Dispatches += b4.strictOps * microBatches;
    this.prismLedger.fusedGroupBarriers += b4.fusedBarriers * microBatches;
  }

  private static isAbortError(err: unknown): boolean {
    if (!err) return false;
    if (err instanceof DOMException && err.name === 'AbortError') return true;
    if (err instanceof Error && err.name === 'AbortError') return true;
    return false;
  }

  private static throwIfAborted(signal?: AbortSignal): void {
    if (signal?.aborted) {
      throw new DOMException('Aborted', 'AbortError');
    }
  }

  /**
   * Async demand-paged single-token forward pass:
   * Prefetches all missing embedding, layer, routed MoE expert, and LM-head stripe ranges
   * concurrently via abortable `fetch()` before touching synchronous vRAM weight handles,
   * eliminating blocking sync XHRs and enabling live layer progress + instant Stop cancellation.
   */
  private async evalTokenAsync(
    tokenId: number,
    pos: number,
    computeLogits: boolean,
    signal?: AbortSignal,
    onDetail?: (detail: string) => void,
  ): Promise<Float32Array> {
    const plan = this.singleAsyncPlan;
    const wp = this.weightPager;
    const backing = this.backingStore;
    if (!plan || !this.scratch || !wp || !backing?.prefetchRanges) {
      return this.evalToken(tokenId, pos, false, computeLogits);
    }
    if (pos < 0 || pos >= this.maxContextWindow) {
      throw new Error(
        `Token position ${pos} exceeds vTensor KV-cache context window (${this.maxContextWindow}).`,
      );
    }

    LlamaVTensorRuntime.throwIfAborted(signal);
    this.singleExec.tokenId = tokenId;
    this.singleExec.pos = pos;
    this.singleExec.seqLen = pos + 1;

    const embdMissing = wp.getEmbeddingMissingRanges([tokenId]);
    if (embdMissing.length > 0) {
      await backing.prefetchRanges(embdMissing, { signal });
    }
    LlamaVTensorRuntime.throwIfAborted(signal);
    plan.embedNode.execute();

    const layerCount = plan.layers.length;
    const demandStats =
      backing instanceof OpfsDemandVramBackingStore ? backing.getStats() : null;
    const poolBytes = demandStats ? demandStats.maxSlots * demandStats.slabBytes : 0;
    const estLayerBytes = Math.max(
      1,
      (this.modelSpec?.activeWeightBytes ?? 0) / Math.max(1, layerCount),
    );
    const lookaheadDepth =
      !demandStats || poolBytes >= estLayerBytes * 5
        ? 3
        : poolBytes >= estLayerBytes * 3
          ? 1
          : 0;
    const inflightLayers = new Map<number, Promise<void>>();
    const scheduleLayerLookahead = (startIdx: number) => {
      if (lookaheadDepth <= 0) return;
      for (let ahead = startIdx; ahead < Math.min(layerCount, startIdx + lookaheadDepth); ahead++) {
        if (inflightLayers.has(ahead) || plan.layers[ahead]?.moe) continue;
        const nextMissing = wp.getLayerMissingRanges(ahead);
        if (nextMissing.length > 0) {
          const p = backing.prefetchRanges!(nextMissing, { signal });
          p.catch(() => {});
          inflightLayers.set(ahead, p);
        }
      }
    };
    try {
      for (let l = 0; l < layerCount; l++) {
        LlamaVTensorRuntime.throwIfAborted(signal);
        const tLayer0 = performance.now();
        const layerPlan = plan.layers[l];
        const pendingThisLayer = inflightLayers.get(l);
        if (pendingThisLayer) {
          inflightLayers.delete(l);
          await pendingThisLayer;
        }
        const layerMissing = wp.getLayerMissingRanges(l);
        if (layerMissing.length > 0) {
          onDetail?.(`Layer ${l + 1}/${layerCount}`);
          await backing.prefetchRanges(layerMissing, { signal });
        } else if (l === 0 || (l + 1) % 4 === 0 || l === layerCount - 1) {
          onDetail?.(`Layer ${l + 1}/${layerCount}`);
        }
        LlamaVTensorRuntime.throwIfAborted(signal);

        layerPlan.acquireNode?.execute();
        // Keep a rolling prefetch pipeline saturated AFTER layer `l` is safely in WASM RAM
        if (!layerPlan.moe && l + 1 < layerCount) {
          scheduleLayerLookahead(l + 1);
        }

        for (let n = 0; n < layerPlan.preMoeNodes.length; n++) {
          layerPlan.preMoeNodes[n].execute();
        }

        if (layerPlan.moe) {
          const routed = layerPlan.moe.routeAndGetExperts();
          const expMissing = wp.getExpertMissingRanges(l, routed.expertIds);
          if (expMissing.length > 0) {
            onDetail?.(`Layer ${l + 1}/${layerCount} · loading experts`);
            await backing.prefetchRanges(expMissing, { signal });
          }
          LlamaVTensorRuntime.throwIfAborted(signal);
          layerPlan.moe.executeRouted(routed.count);
          layerPlan.moe.residualNode.execute();
        } else {
          for (let n = 0; n < layerPlan.denseFfnNodes.length; n++) {
            layerPlan.denseFfnNodes[n].execute();
          }
        }

        if (performance.now() - tLayer0 >= 8) {
          await this.yielder.yieldNow();
        }
      }
    } finally {
      for (const p of inflightLayers.values()) {
        p.catch(() => {});
      }
      inflightLayers.clear();
    }

    if (computeLogits) {
      LlamaVTensorRuntime.throwIfAborted(signal);
      for (let n = 0; n < plan.lmNormAndQuantNodes.length; n++) {
        plan.lmNormAndQuantNodes[n].execute();
      }
      if (plan.pinnedLmHeadNode) {
        plan.pinnedLmHeadNode.execute();
      } else if (plan.executeLmHeadStripe) {
        const cachedFullLmHead = wp.getCachedFullLmHead();
        if (cachedFullLmHead) {
          // Zero-OPFS fast path: full-vocab LM-head is already cached in WASM RAM -> 1 single multi-core pass!
          plan.executeLmHeadStripe(cachedFullLmHead, 0, this.hparams!.vocabSize);
        } else {
          const stripes = wp.getLmHeadStripeSpecs();
          const stripeBytesEst = stripes[0]?.byteLength ?? 0;
          const stripeLookahead =
            !demandStats || poolBytes >= stripeBytesEst * 3 ? 2 : 0;
          const inflightStripes = new Map<number, Promise<void>>();
          const scheduleStripes = (startIdx: number) => {
            if (stripeLookahead <= 0) return;
            for (let s = startIdx; s < Math.min(stripes.length, startIdx + stripeLookahead); s++) {
              if (inflightStripes.has(s)) continue;
              const p = backing.prefetchRanges!([stripes[s]], { signal });
              p.catch(() => {});
              inflightStripes.set(s, p);
            }
          };
          try {
            for (let sIdx = 0; sIdx < stripes.length; sIdx++) {
              LlamaVTensorRuntime.throwIfAborted(signal);
              const tStripe0 = performance.now();
              const spec = stripes[sIdx];
              const pct = Math.min(99, Math.max(1, Math.round(((sIdx + 1) / stripes.length) * 100)));
              onDetail?.(`Computing logits (${pct}%)`);
              const pendingStripe = inflightStripes.get(sIdx);
              if (pendingStripe) {
                inflightStripes.delete(sIdx);
                await pendingStripe;
              }
              if (
                !(backing instanceof OpfsDemandVramBackingStore) ||
                !backing.isRangeCached(spec.fileOffset, spec.byteLength)
              ) {
                await backing.prefetchRanges([spec], { signal });
              }
              LlamaVTensorRuntime.throwIfAborted(signal);
              const weightStripe = wp.acquireLmHeadStripe(spec.rowStart, spec.rowCount);
              // Keep upcoming LM-head stripes prefetching concurrently with WASM SIMD projection
              if (sIdx + 1 < stripes.length) {
                scheduleStripes(sIdx + 1);
              }
              plan.executeLmHeadStripe(weightStripe, spec.rowStart, spec.rowCount);
              if (performance.now() - tStripe0 >= 12) {
                await this.yielder.yieldNow();
              }
            }
          } finally {
            for (const p of inflightStripes.values()) {
              p.catch(() => {});
            }
            inflightStripes.clear();
          }
        }
      }
    }

    const bt = this.prismSingleBodyTally;
    this.prismLedger.q8kDispatches += bt.q8kOps;
    this.prismLedger.q8kSuperBlocks += bt.q8kSb;
    this.prismLedger.q80Dispatches += bt.q80Ops;
    this.prismLedger.strictF32Dispatches += bt.strictOps;
    this.prismLedger.fusedGroupBarriers += bt.fusedBarriers;
    if (computeLogits) {
      const lt = this.prismSingleLmTally;
      this.prismLedger.q8kDispatches += lt.q8kOps;
      this.prismLedger.q8kSuperBlocks += lt.q8kSb;
      this.prismLedger.q80Dispatches += lt.q80Ops;
      this.prismLedger.strictF32Dispatches += lt.strictOps;
    }

    return this.scratch.logits.buffer.f32View(this.scratch.logits);
  }

  /**
   * Async demand-paged Batch-4 prefill pass (4 non-final tokens simultaneously).
   */
  private async evalPrefillBatch4Async(
    tokens: readonly number[],
    offset: number,
    startPos: number,
    signal?: AbortSignal,
    onDetail?: (detail: string) => void,
  ): Promise<void> {
    const plan = this.batchAsyncPlan;
    const wp = this.weightPager;
    const backing = this.backingStore;
    if (!plan || !wp || !backing?.prefetchRanges) {
      this.evalPrefillBatch4(tokens, offset, startPos);
      return;
    }

    LlamaVTensorRuntime.throwIfAborted(signal);
    const batchToks = [
      tokens[offset],
      tokens[offset + 1],
      tokens[offset + 2],
      tokens[offset + 3],
    ];
    this.batchExec.tokens[0] = batchToks[0];
    this.batchExec.tokens[1] = batchToks[1];
    this.batchExec.tokens[2] = batchToks[2];
    this.batchExec.tokens[3] = batchToks[3];
    this.batchExec.startPos = startPos;

    const embdMissing = wp.getEmbeddingMissingRanges(batchToks);
    if (embdMissing.length > 0) {
      await backing.prefetchRanges(embdMissing, { signal });
    }
    LlamaVTensorRuntime.throwIfAborted(signal);
    plan.embedNode.execute();

    const layerCount = plan.layers.length;
    let pendingNextLayerPrefetch: Promise<void> | null = null;
    try {
      for (let l = 0; l < layerCount; l++) {
        LlamaVTensorRuntime.throwIfAborted(signal);
        const tLayer0 = performance.now();
        const layerPlan = plan.layers[l];
        if (pendingNextLayerPrefetch) {
          await pendingNextLayerPrefetch;
          pendingNextLayerPrefetch = null;
        }
        const layerMissing = wp.getLayerMissingRanges(l);
        if (layerMissing.length > 0) {
          onDetail?.(`Layer ${l + 1}/${layerCount}`);
          await backing.prefetchRanges(layerMissing, { signal });
        } else if (l === 0 || (l + 1) % 4 === 0 || l === layerCount - 1) {
          onDetail?.(`Layer ${l + 1}/${layerCount}`);
        }
        LlamaVTensorRuntime.throwIfAborted(signal);

        layerPlan.acquireNode?.execute();
        if (!layerPlan.moe && l + 1 < layerCount) {
          const nextMissing = wp.getLayerMissingRanges(l + 1);
          if (nextMissing.length > 0) {
            pendingNextLayerPrefetch = backing.prefetchRanges(nextMissing, { signal });
          }
        }

        for (let n = 0; n < layerPlan.preMoeNodes.length; n++) {
          layerPlan.preMoeNodes[n].execute();
        }
        if (layerPlan.moe) {
          const routed = layerPlan.moe.routeAndGetExperts();
          const expMissing = wp.getExpertMissingRanges(l, routed.expertIds);
          if (expMissing.length > 0) {
            onDetail?.(`Layer ${l + 1}/${layerCount} · loading experts`);
            await backing.prefetchRanges(expMissing, { signal });
          }
          LlamaVTensorRuntime.throwIfAborted(signal);
          layerPlan.moe.executeRouted(routed.count);
          layerPlan.moe.residualNode.execute();
        } else {
          for (let n = 0; n < layerPlan.denseFfnNodes.length; n++) {
            layerPlan.denseFfnNodes[n].execute();
          }
        }

        if (performance.now() - tLayer0 >= 8) {
          await this.yielder.yieldNow();
        }
      }
    } finally {
      if (pendingNextLayerPrefetch) {
        pendingNextLayerPrefetch.catch(() => {});
      }
    }

    const b4 = this.prismBatch4Tally;
    this.prismLedger.q8kDispatches += b4.q8kOps;
    this.prismLedger.q8kSuperBlocks += b4.q8kSb;
    this.prismLedger.q80Dispatches += b4.q80Ops;
    this.prismLedger.strictF32Dispatches += b4.strictOps;
    this.prismLedger.fusedGroupBarriers += b4.fusedBarriers;
  }

  /**
   * Async Layer-Major Chunked Prefill (`C = 4 * microBatches`, up to 32 tokens):
   * Prefetches each layer ONCE across the entire chunk via parallel abortable `fetch()`,
   * then routes and prefetches missing MoE experts per micro-batch with live progress.
   */
  private async evalPrefillChunkLayerMajorAsync(
    tokens: readonly number[],
    offset: number,
    startPos: number,
    microBatches: number,
    signal?: AbortSignal,
    onDetail?: (detail: string) => void,
  ): Promise<void> {
    const plan = this.batchAsyncPlan;
    const wp = this.weightPager;
    const backing = this.backingStore;
    const bs = this.batchScratch;
    const hp = this.hparams;
    const chunkX = this.prefillChunkX;
    if (!plan || !wp || !backing?.prefetchRanges || !bs || !hp || !chunkX) {
      this.evalPrefillChunkLayerMajor(tokens, offset, startPos, microBatches);
      return;
    }

    LlamaVTensorRuntime.throwIfAborted(signal);
    const chunkTokensCount = microBatches * PREFILL_BATCH_SIZE;
    const chunkTokenIds: number[] = new Array(chunkTokensCount);
    for (let k = 0; k < chunkTokensCount; k++) {
      chunkTokenIds[k] = tokens[offset + k];
    }
    const embdMissing = wp.getEmbeddingMissingRanges(chunkTokenIds);
    if (embdMissing.length > 0) {
      await backing.prefetchRanges(embdMissing, { signal });
    }
    LlamaVTensorRuntime.throwIfAborted(signal);

    const batchFloats = PREFILL_BATCH_SIZE * hp.embeddingLength;
    const bsXView = bs.x.buffer.f32View(bs.x);

    for (let m = 0; m < microBatches; m++) {
      const baseTok = offset + m * PREFILL_BATCH_SIZE;
      this.batchExec.tokens[0] = tokens[baseTok];
      this.batchExec.tokens[1] = tokens[baseTok + 1];
      this.batchExec.tokens[2] = tokens[baseTok + 2];
      this.batchExec.tokens[3] = tokens[baseTok + 3];
      plan.embedNode.execute();
      chunkX.set(bsXView, m * batchFloats);
    }

    const layerCount = plan.layers.length;
    const demandStats =
      backing instanceof OpfsDemandVramBackingStore ? backing.getStats() : null;
    const poolBytes = demandStats ? demandStats.maxSlots * demandStats.slabBytes : 0;
    const estLayerBytes = Math.max(
      1,
      (this.modelSpec?.activeWeightBytes ?? 0) / Math.max(1, layerCount),
    );
    const lookaheadDepth =
      !demandStats || poolBytes >= estLayerBytes * 5
        ? 3
        : poolBytes >= estLayerBytes * 3
          ? 1
          : 0;
    const inflightLayers = new Map<number, Promise<void>>();
    const scheduleLayerLookahead = (startIdx: number) => {
      if (lookaheadDepth <= 0) return;
      for (let ahead = startIdx; ahead < Math.min(layerCount, startIdx + lookaheadDepth); ahead++) {
        if (inflightLayers.has(ahead) || plan.layers[ahead]?.moe) continue;
        const nextMissing = wp.getLayerMissingRanges(ahead);
        if (nextMissing.length > 0) {
          const p = backing.prefetchRanges!(nextMissing, { signal });
          p.catch(() => {});
          inflightLayers.set(ahead, p);
        }
      }
    };
    try {
      for (let l = 0; l < layerCount; l++) {
        LlamaVTensorRuntime.throwIfAborted(signal);
        const tLayer0 = performance.now();
        const layerPlan = plan.layers[l];
        const pendingThisLayer = inflightLayers.get(l);
        if (pendingThisLayer) {
          inflightLayers.delete(l);
          await pendingThisLayer;
        }
        const layerMissing = wp.getLayerMissingRanges(l);
        if (layerMissing.length > 0) {
          onDetail?.(`Layer ${l + 1}/${layerCount}`);
          await backing.prefetchRanges(layerMissing, { signal });
        } else if (l === 0 || (l + 1) % 4 === 0 || l === layerCount - 1) {
          onDetail?.(`Layer ${l + 1}/${layerCount}`);
        }
        LlamaVTensorRuntime.throwIfAborted(signal);

        layerPlan.acquireNode?.execute();
        // Overlap rolling OPFS/network prefetch across ALL `microBatches` AFTER layer `l` is in WASM RAM
        if (!layerPlan.moe && l + 1 < layerCount) {
          scheduleLayerLookahead(l + 1);
        }

        for (let m = 0; m < microBatches; m++) {
          LlamaVTensorRuntime.throwIfAborted(signal);
          const sliceStart = m * batchFloats;
          bsXView.set(chunkX.subarray(sliceStart, sliceStart + batchFloats), 0);
          this.batchExec.startPos = startPos + m * PREFILL_BATCH_SIZE;

          for (let n = 0; n < layerPlan.preMoeNodes.length; n++) {
            layerPlan.preMoeNodes[n].execute();
          }
          if (layerPlan.moe) {
            const routed = layerPlan.moe.routeAndGetExperts();
            const expMissing = wp.getExpertMissingRanges(l, routed.expertIds);
            if (expMissing.length > 0) {
              onDetail?.(`Layer ${l + 1}/${layerCount} · loading experts`);
              await backing.prefetchRanges(expMissing, { signal });
            }
            LlamaVTensorRuntime.throwIfAborted(signal);
            layerPlan.moe.executeRouted(routed.count);
            layerPlan.moe.residualNode.execute();
          } else {
            for (let n = 0; n < layerPlan.denseFfnNodes.length; n++) {
              layerPlan.denseFfnNodes[n].execute();
            }
          }
          chunkX.set(bsXView, sliceStart);
        }

        if (performance.now() - tLayer0 >= 8) {
          await this.yielder.yieldNow();
        }
      }
    } finally {
      for (const p of inflightLayers.values()) {
        p.catch(() => {});
      }
      inflightLayers.clear();
    }

    const b4 = this.prismBatch4Tally;
    this.prismLedger.q8kDispatches += b4.q8kOps * microBatches;
    this.prismLedger.q8kSuperBlocks += b4.q8kSb * microBatches;
    this.prismLedger.q80Dispatches += b4.q80Ops * microBatches;
    this.prismLedger.strictF32Dispatches += b4.strictOps * microBatches;
    this.prismLedger.fusedGroupBarriers += b4.fusedBarriers * microBatches;
  }

  /**
   * Returns the exact token count for `prompt` under the active GGUF model's tokenizer
   * (including BOS when configured by the model), or 0 if no model is loaded.
   */
  countTokens(prompt: string): number {
    if (!this.tokenizer || !prompt) return 0;
    return this.tokenizer.encode(prompt, this.tokenizer.addBosToken).length;
  }

  /**
   * Runs prompt prefill (with bit-exact KV-cache prefix reuse, 4-tier vRAM paging, contract-driven
   * batched-4 layer execution, and LM-head skipping) followed by autoregressive decoding through
   * vTensor up to the model's full native context length, continuously verifying Theorem T5.
   */
  async *generate(
    prompt: string,
    config: SamplingConfig = DEFAULT_SAMPLING_CONFIG,
    signal?: AbortSignal,
    onPrefillProgress?: (progress: PrefillProgressEvent) => void,
    threadId?: string,
    prefixEmbeddings?: readonly Float32Array[],
  ): AsyncGenerator<TokenStreamEvent, void, unknown> {
    if (!this.tokenizer || !this.hparams || !this.kvPager) {
      throw new Error('No GGUF model loaded in llama.cpp + vTensor runtime.');
    }

    // Strict thread isolation: reset live KV pages whenever switching threads so history from
    // another thread can NEVER conflate with this conversation.
    if (threadId && this.activeThreadId !== null && threadId !== this.activeThreadId) {
      this.resetKvCache();
    }
    if (threadId) {
      this.activeThreadId = threadId;
    }

    // Native coherent multimodal embedding injection: when dimension-matched visual embeddings
    // (e.g. SigLIP mmproj) are provided, prefill them directly into Layer 0 ahead of text tokens!
    const validPrefixEmbeddings =
      prefixEmbeddings &&
      prefixEmbeddings.length > 0 &&
      prefixEmbeddings.every((e) => e.length === this.hparams!.embeddingLength)
        ? prefixEmbeddings
        : null;
    let posOffset = 0;
    if (validPrefixEmbeddings) {
      this.resetKvCache();
      const injected = this.prefillEmbeddingSequence(validPrefixEmbeddings, 0, false);
      posOffset = injected.nextPos;
    }

    const promptTokens = this.tokenizer.encode(prompt, this.tokenizer.addBosToken);
    // Allow users to input up to 100% of the selected model's native context window with zero artificial reservation.
    // On extreme overflow, preserve the leading system/anchor prefix alongside the recent tail so instructions
    // are never amputated and the pinned KV prefix remains reusable!
    const maxPrompt = Math.max(1, this.maxContextWindow - posOffset);
    const promptTruncatedTokens = Math.max(0, promptTokens.length - maxPrompt);
    let truncatedPrompt: number[];
    if (promptTokens.length > maxPrompt) {
      const keepHead = Math.min(
        Math.max(1, this.kvPager.getPinnedPrefixPages() * 16),
        Math.floor(maxPrompt * 0.25),
      );
      const keepTail = Math.max(1, maxPrompt - keepHead);
      truncatedPrompt = [
        ...promptTokens.slice(0, keepHead),
        ...promptTokens.slice(promptTokens.length - keepTail),
      ];
    } else {
      truncatedPrompt = promptTokens;
    }

    if (truncatedPrompt.length === 0) {
      truncatedPrompt = [this.hparams.bosTokenId];
    }

    // Bit-exact live KV-cache prefix reuse + Unified Virtual Store Q8 block restore
    const aligned =
      posOffset > 0
        ? { commonPrefix: 0, reusedLive: 0, restoredUnified: 0 }
        : this.alignAndRestorePrefix(truncatedPrompt, truncatedPrompt.length - 1);
    const commonPrefix = aligned.commonPrefix;
    // Pin leading system/anchor KV pages above lossy Cold-Q4 demotion as multi-turn prefix stabilizes
    const anchorPages = Math.max(
      this.kvPager.getPinnedPrefixPages(),
      Math.min(16, Math.max(1, Math.floor(Math.min(truncatedPrompt.length, Math.max(64, commonPrefix)) / 16))),
    );
    this.kvPager.setPinnedPrefixPages(anchorPages);
    this.unifiedStore?.setPinnedPrefixPages(anchorPages);
    const estMsPerPrefillToken =
      this.performanceContract && this.performanceContract.rooflinePrefillTokSec > 0
        ? 1000 / this.performanceContract.rooflinePrefillTokSec
        : 0;
    this.unifiedStore?.recordPromptTurnReuse({
      totalPromptTokens: truncatedPrompt.length,
      reusedLiveTokens: aligned.reusedLive,
      restoredUnifiedTokens: aligned.restoredUnified,
      estimatedMsPerPrefillToken: estMsPerPrefillToken,
    });

    const rng = createRng(config.seed);
    const history: number[] = [...truncatedPrompt];
    const streamDecoder = this.tokenizer.createStreamDecoder();

    // Prefill pass: batch B* non-final tokens at a time when contract enables batching
    const prefillStart = performance.now();
    let logits: Float32Array | null = null;
    const tokensToEval = truncatedPrompt.length - commonPrefix;
    const lastIdx = truncatedPrompt.length - 1;
    const batchWidth = this.performanceContract?.optimalPrefillBatchSize ?? PREFILL_BATCH_SIZE;
    const useAsyncDemand = Boolean(
      this.backingStore?.prefetchRanges && this.singleAsyncPlan && this.batchAsyncPlan,
    );
    let i = commonPrefix;
    this.frameBudget.beginSlice();

    const emitStageDetail = (completedTokens: number, detail?: string) => {
      if (!onPrefillProgress) return;
      onPrefillProgress({
        completedTokens,
        totalTokens: tokensToEval,
        reusedPrefixTokens: commonPrefix,
        detail,
      });
    };

    try {
      while (i < truncatedPrompt.length) {
        if (signal?.aborted) return;

        const tStep0 = performance.now();
        const chunkBatches =
          this.backend.isWasmSimdActive && batchWidth >= PREFILL_BATCH_SIZE
            ? this.computeLayerMajorMicroBatches(i, lastIdx)
            : 0;
        if (chunkBatches >= 2) {
          const chunkTokens = chunkBatches * PREFILL_BATCH_SIZE;
          if (useAsyncDemand) {
            await this.evalPrefillChunkLayerMajorAsync(
              truncatedPrompt,
              i,
              i + posOffset,
              chunkBatches,
              signal,
              (detail) => emitStageDetail(i - commonPrefix, detail),
            );
          } else {
            this.evalPrefillChunkLayerMajor(truncatedPrompt, i, i + posOffset, chunkBatches);
          }
          if (posOffset === 0) {
            for (let k = 0; k < chunkTokens; k++) {
              this.cachedTokens.push(truncatedPrompt[i + k]);
            }
          }
          i += chunkTokens;

          emitStageDetail(i - commonPrefix);
          const stepUs = (performance.now() - tStep0) * 1000;
          if (
            this.frameBudget.recordStepAndShouldYield(stepUs, GENERATION_SLICE_BUDGET_US, 2)
          ) {
            await this.yielder.yieldNow();
          }
          continue;
        }

        if (
          this.backend.isWasmSimdActive &&
          batchWidth >= PREFILL_BATCH_SIZE &&
          i + PREFILL_BATCH_SIZE <= lastIdx
        ) {
          if (useAsyncDemand) {
            await this.evalPrefillBatch4Async(
              truncatedPrompt,
              i,
              i + posOffset,
              signal,
              (detail) => emitStageDetail(i - commonPrefix, detail),
            );
          } else {
            this.evalPrefillBatch4(truncatedPrompt, i, i + posOffset);
          }
          if (posOffset === 0) {
            for (let k = 0; k < PREFILL_BATCH_SIZE; k++) {
              this.cachedTokens.push(truncatedPrompt[i + k]);
            }
          }
          i += PREFILL_BATCH_SIZE;

          emitStageDetail(i - commonPrefix);
          const stepUs = (performance.now() - tStep0) * 1000;
          if (
            this.frameBudget.recordStepAndShouldYield(stepUs, GENERATION_SLICE_BUDGET_US, 8)
          ) {
            await this.yielder.yieldNow();
          }
          continue;
        }

        const isLastPrefill = i === lastIdx;
        if (useAsyncDemand) {
          logits = await this.evalTokenAsync(
            truncatedPrompt[i],
            i + posOffset,
            isLastPrefill,
            signal,
            (detail) => emitStageDetail(i - commonPrefix, detail),
          );
        } else {
          logits = this.evalToken(truncatedPrompt[i], i + posOffset, isLastPrefill, isLastPrefill);
        }
        if (posOffset === 0) {
          this.cachedTokens.push(truncatedPrompt[i]);
        }
        i += 1;

        emitStageDetail(i - commonPrefix);
        if (!isLastPrefill) {
          const stepUs = (performance.now() - tStep0) * 1000;
          if (
            this.frameBudget.recordStepAndShouldYield(stepUs, GENERATION_SLICE_BUDGET_US, 8)
          ) {
            await this.yielder.yieldNow();
          }
        }
      }

      if (!logits) {
        logits = useAsyncDemand
          ? await this.evalTokenAsync(this.hparams.bosTokenId, 0, true, signal, (detail) =>
              emitStageDetail(0, detail),
            )
          : this.evalToken(this.hparams.bosTokenId, 0, true, true);
        this.cachedTokens = [this.hparams.bosTokenId];
      }

      // Commit completed prompt prefix pages into Unified Virtual Store immediately after prefill
      if (posOffset === 0) {
        this.commitFullPagesToUnifiedStore(this.cachedTokens);
      }

      const prefillElapsedSec = Math.max(0.001, (performance.now() - prefillStart) / 1000);
      const observedPrefillTokSec = Number((Math.max(1, tokensToEval) / prefillElapsedSec).toFixed(1));

      const decodeStart = performance.now();
      let pos = posOffset + truncatedPrompt.length;
      let accumulated = '';
      const requestedMaxGen =
        config.fullContextOutput || config.maxTokens <= 0
          ? this.maxContextWindow
          : config.maxTokens;
      const remainingSlots = Math.max(0, this.maxContextWindow - pos);
      // Final prompt logits can always emit 1 token without writing a new KV slot; each subsequent token
      // consumes 1 KV slot up to `this.maxContextWindow - 1`, allowing responses up to full context length.
      const maxPossibleGen = Math.min(this.maxContextWindow, remainingSlots + 1);
      const maxGen = Math.max(1, Math.min(requestedMaxGen, maxPossibleGen));

      const updateContract = (tokSec: number) => {
        if (this.performanceContract && tokSec > 0) {
          this.performanceContract = evaluateContractCompliance(
            this.performanceContract,
            tokSec,
            observedPrefillTokSec,
            0,
          );
        }
      };

      for (let step = 0; step < maxGen; step++) {
        if (signal?.aborted) return;

        const nextToken = sampleNextToken(
          logits,
          history,
          config,
          rng,
          this.backend.arena,
          this.scratch ? this.scratch.logits.buffer.wasmPtr(this.scratch.logits) : -1,
        );
        const isStopToken =
          this.tokenizer.eogTokenIds.has(nextToken) ||
          (nextToken === this.hparams.bosTokenId && this.hparams.bosTokenId !== this.hparams.eosTokenId);

        if (isStopToken) {
          const trailing = streamDecoder.flush();
          if (trailing) accumulated += trailing;
          const elapsedSec = Math.max(0.001, (performance.now() - decodeStart) / 1000);
          const tokSec = Number((step / elapsedSec).toFixed(1));
          if (step >= 1) updateContract(tokSec);
          yield {
            tokenId: nextToken,
            textDelta: trailing,
            accumulatedText: accumulated,
            generatedTokens: step,
            promptTokens: truncatedPrompt.length,
            contextWindow: this.maxContextWindow,
            promptTruncatedTokens,
            tokensPerSecond: tokSec,
            finished: true,
          };
          return;
        }

        history.push(nextToken);
        const generatedCount = step + 1;
        const finished = generatedCount >= maxGen;
        let delta = streamDecoder.push(nextToken);
        if (finished) {
          delta += streamDecoder.flush();
        }
        accumulated += delta;

        const elapsedSec = Math.max(0.001, (performance.now() - decodeStart) / 1000);
        const tokSec = Number((generatedCount / elapsedSec).toFixed(1));
        if (generatedCount === 2 || generatedCount % 4 === 0 || finished) {
          updateContract(tokSec);
        }

        yield {
          tokenId: nextToken,
          textDelta: delta,
          accumulatedText: accumulated,
          generatedTokens: generatedCount,
          promptTokens: truncatedPrompt.length,
          contextWindow: this.maxContextWindow,
          promptTruncatedTokens,
          tokensPerSecond: tokSec,
          finished,
        };

        if (!finished && pos < this.maxContextWindow) {
          if (useAsyncDemand) {
            logits = await this.evalTokenAsync(nextToken, pos, true, signal);
          } else {
            logits = this.evalToken(nextToken, pos, step === 0, true);
          }
          if (posOffset === 0) {
            this.cachedTokens.push(nextToken);
          }
          pos += 1;
        }
      }
    } catch (err) {
      if (signal?.aborted || LlamaVTensorRuntime.isAbortError(err)) {
        return;
      }
      throw err;
    } finally {
      // Roll back any partially-written KV slots past `cachedTokens.length` (e.g. mid-chunk abort)
      // before committing completed 16-token KV pages into the Unified Store.
      if (posOffset === 0) {
        this.kvPager?.truncateTo(this.cachedTokens.length);
        this.commitFullPagesToUnifiedStore(this.cachedTokens);
      } else {
        this.resetKvCache();
      }
      this.flushDemandBacking();
    }
  }
}
