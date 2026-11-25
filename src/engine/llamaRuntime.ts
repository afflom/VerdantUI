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
  Q8_0_ACT_BLOCK_BYTES,
  Q8_K_ACT_BLOCK_BYTES,
  WASM_TOPK_IDS_OFFSET,
  WASM_TOPK_VALS_OFFSET,
  type MatGroupEntry,
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
  type VramBackingKind,
  type VramBackingStore,
} from './vramBackingStore';
import {
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
import type { WebGpuSubstrateTelemetry } from './verdantWebGpu';
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
  'qwen3',
  'mistral',
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
    // Greedy argmax when temperature <= 0.01 (100% in-Verdant WASM SIMD when arena-backed)
    if (config.temperature <= 0.01) {
      if (arena && logitsPtr >= 0) {
        return arena.kernels.argmax_f32(logitsPtr, vocabSize);
      }
      let bestId = 0;
      let bestVal = -Infinity;
      for (let i = 0; i < vocabSize; i++) {
        const v = logits[i];
        if (v > bestVal) {
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
        candCount = arena.kernels.topk_f32(
          logitsPtr,
          vocabSize,
          k,
          WASM_TOPK_IDS_OFFSET,
          WASM_TOPK_VALS_OFFSET,
        );
        const idsView = new Int32Array(arena.memory.buffer, WASM_TOPK_IDS_OFFSET, candCount);
        const valsView = new Float32Array(arena.memory.buffer, WASM_TOPK_VALS_OFFSET, candCount);
        for (let i = 0; i < candCount; i++) {
          samplerTopkIds[i] = idsView[i];
          samplerTopkVals[i] = valsView[i];
        }
      } else {
        candCount = k;
        for (let i = 0; i < k; i++) {
          samplerTopkIds[i] = 0;
          samplerTopkVals[i] = -Infinity;
        }
        let minSlot = 0;
        let minVal = -Infinity;
        for (let i = 0; i < vocabSize; i++) {
          const v = logits[i];
          if (v > minVal) {
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
      }

      // In-place descending sort of candCount <= 1024 items (zero heap allocations)
      for (let i = 1; i < candCount; i++) {
        const keyId = samplerTopkIds[i];
        const keyVal = samplerTopkVals[i];
        let j = i - 1;
        while (j >= 0 && samplerTopkVals[j] < keyVal) {
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
        if (v > minVal) {
          topIds[minSlot] = i;
          topVals[minSlot] = v;
          minVal = topVals[0];
          minSlot = 0;
          for (let j = 1; j < k; j++) {
            if (topVals[j] < minVal) {
              minVal = topVals[j];
              minSlot = j;
            }
          }
        }
      }
      const sorted = Array.from(topIds).sort((a, b) => logits[b] - logits[a]);
      candIds = Int32Array.from(sorted);
      candCount = sorted.length;
    }

    const invTemp = 1.0 / config.temperature;
    const maxLogit = logits[candIds[0]] * invTemp;

    const probs = candCount <= 1024 ? samplerTopkProbs : new Float64Array(candCount);
    let sumExp = 0;
    for (let i = 0; i < candCount; i++) {
      const p = Math.exp(logits[candIds[i]] * invTemp - maxLogit);
      probs[i] = p;
      sumExp += p;
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
  private unifiedStore: VramUnifiedStore | null = null;
  private readonly committedPageKeys = new Set<string>();

  // OPFS-backed vRAM Virtual DIMM Weight Pager & parsed header state
  private backingStore: VramBackingStore | null = null;
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
  private readonly singleExec = { tokenId: 0, pos: 0, seqLen: 1 };
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

  attachWebGpuTelemetry(provider: (() => WebGpuSubstrateTelemetry) | null): void {
    this.backend.attachWebGpuTelemetry(provider);
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

  async hydrateUnifiedStoreFromOpfs(
    opfsDirectoryOverride?: FileSystemDirectoryHandle,
  ): Promise<number> {
    if (!this.unifiedStore) return 0;
    return this.unifiedStore.hydrateFromOpfs(opfsDirectoryOverride);
  }

  async persistUnifiedStoreToOpfs(
    opfsDirectoryOverride?: FileSystemDirectoryHandle,
  ): Promise<number> {
    if (!this.unifiedStore) return 0;
    return this.unifiedStore.persistToOpfs(opfsDirectoryOverride);
  }

  unload(): void {
    this.backend.freeAllBuffers();
    if (this.backingStore) {
      this.backingStore.close();
      this.backingStore = null;
    }
    this.parsedHeader = null;
    this.weightPager = null;
    this.embdInfo = null;
    this.outNormInfo = null;
    this.outWeightInfo = null;
    this.isTiedOutput = false;
    this.layerInfos = [];
    this.kvPager = null;
    this.cachedTokens = [];
    this.unifiedStore = null;
    this.committedPageKeys.clear();
    this.ropeTables = null;
    this.scratch = null;
    this.batchScratch = null;
    this.compiledSingleGraph = null;
    this.compiledBatch4Graph = null;
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
    } catch (err) {
      backing.close();
      throw err;
    }

    const hp = parsed.hparams;
    const hasStandardTransformerTopology =
      parsed.tensorMap.has('token_embd.weight') &&
      parsed.tensorMap.has('output_norm.weight') &&
      parsed.tensorMap.has('blk.0.attn_norm.weight') &&
      parsed.tensorMap.has('blk.0.attn_q.weight') &&
      parsed.tensorMap.has('blk.0.attn_k.weight') &&
      parsed.tensorMap.has('blk.0.attn_v.weight') &&
      parsed.tensorMap.has('blk.0.attn_output.weight') &&
      parsed.tensorMap.has('blk.0.ffn_norm.weight') &&
      parsed.tensorMap.has('blk.0.ffn_gate.weight') &&
      parsed.tensorMap.has('blk.0.ffn_up.weight') &&
      parsed.tensorMap.has('blk.0.ffn_down.weight');

    if (!SUPPORTED_ARCHITECTURES.has(hp.architecture) && !hasStandardTransformerTopology) {
      backing.close();
      throw new Error(
        `GGUF architecture "${hp.architecture}" uses an unsupported tensor graph topology. Supported architectures: llama, qwen2, qwen3, mistral, granite, smollm3, internlm2, olmo2 (or any LLaMA-compatible SwiGLU GGUF).`,
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
    for (let l = 0; l < hp.blockCount; l++) {
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
        ffnGate: requireTensor(`blk.${l}.ffn_gate.weight`),
        ffnUp: requireTensor(`blk.${l}.ffn_up.weight`),
        ffnDown: requireTensor(`blk.${l}.ffn_down.weight`),
      };
      layerInfos.push(layer);
      tensorInfosToLoad.push(
        layer.attnNorm,
        layer.attnQ,
        layer.attnK,
        layer.attnV,
        layer.attnOutput,
        layer.ffnNorm,
        layer.ffnGate,
        layer.ffnUp,
        layer.ffnDown,
      );
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

    // Compute exact ModelGeometrySpec for Prism Theorems T1–T4b
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

    for (const l of layerInfos) {
      const linearWeights = [
        l.attnQ,
        l.attnK,
        l.attnV,
        l.attnOutput,
        l.ffnGate,
        l.ffnUp,
        l.ffnDown,
      ];
      for (const w of linearWeights) {
        activeWeightBytes += w.byteSize;
        activeParamCount += (w.dims[0] ?? 1) * (w.dims[1] ?? 1);
        if (INTEGER_DOT_COMPATIBLE_TYPES.has(w.type)) {
          quantizedLinearBytes += w.byteSize;
        }
        typeByteCounts.set(w.type, (typeByteCounts.get(w.type) ?? 0) + w.byteSize);
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
    };

    const maxContextCap = options?.maxContextCap;
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

    this.allocateFromActiveBacking();
    return this.modelSummary!;
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

    // 1. Allocate model weights through OPFS-backed vRAM Weight Pager (resident-all or opfs-paged + Zipfian hot embedding slab)
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
        lmHeadStripeRows: computeLmHeadStripeRows(outWeightInfo, contract.lmHeadTileTargetBytes),
        hotEmbeddingCacheSlots: unifiedPlan.hotEmbeddingCacheSlots,
      },
      (bufId, usage, cap) => this.backend.allocBuffer(bufId, usage, cap),
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

    // 2b. Initialize or retain the Unified Virtual Store for this model's geometry
    const modelId = this.modelSummary?.id ?? backing.modelId;
    const existingStore = this.unifiedStore;
    const reuseExistingStore =
      existingStore !== null &&
      existingStore.geometry.modelId === modelId &&
      existingStore.geometry.numLayers === hp.blockCount &&
      existingStore.geometry.nHeadsKv === hp.headCountKv &&
      existingStore.geometry.headDim === headDim;

    const unifiedStore = reuseExistingStore
      ? existingStore
      : new VramUnifiedStore(
          {
            modelId,
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

    // 3. Allocate single-token + batched-4 compute scratch tensors (including packed Q8_0_ACT slots)
    const dim = hp.embeddingLength;
    const ffnDim = hp.feedForwardLength;
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

    const scratchCapacity =
      alignUp32(dim * 4) * 4 +
      alignUp32(qDim * 4) * 2 +
      alignUp32(kvDim * 4) * 2 +
      alignUp32(ffnDim * 4) * 3 +
      alignUp32(vocab * 4) +
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
    const ropeLutMaxPos = Math.min(maxCtx, 4096);
    const ropeTables = buildRopeTables(
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
    const ffnDim = hp.feedForwardLength;
    const useNeoxRope = hp.ropeType === 'neox';
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
      hardwareWebGpu: false,
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
          gpuResident: false,
        });
      };
      pushOp(`blk.${l}.attn_q`, li.attnQ, 3);
      pushOp(`blk.${l}.attn_k`, li.attnK, 3);
      pushOp(`blk.${l}.attn_v`, li.attnV, 3);
      pushOp(`blk.${l}.attn_output`, li.attnOutput, 1);
      pushOp(`blk.${l}.ffn_gate`, li.ffnGate, 2);
      pushOp(`blk.${l}.ffn_up`, li.ffnUp, 2);
      pushOp(`blk.${l}.ffn_down`, li.ffnDown, 1);
    }
    scheduleOps.push({
      label: 'output.weight',
      weightType: outWeightInfo.type,
      rows: outWeightInfo.dims[1] ?? hp.vocabSize,
      cols: outWeightInfo.dims[0] ?? dim,
      batchSize: 1,
      groupSize: 1,
      weightBytes: outWeightInfo.byteSize,
      gpuResident: false,
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

    for (let l = 0; l < layerCount; l++) {
      const layerIdx = l;
      const info = this.layerInfos[l];
      const hot = isPaged ? wp.getHotLayer(l) : residentLayers![l];
      const lw = isPaged ? null : residentLayers![l];

      if (isPaged) {
        // Single DIMM publication for all 7 projections in layer `l` (afflom/vRAM 3f27a941)
        singleNodes.push({
          op: GgmlOp.CPY,
          label: `blk.${l}.vram_dimm_acquire`,
          shapeDesc: `vRAM DIMM L${l} -> frame`,
          srcType: wp.backingKind === 'opfs-sync' ? 'OPFS DIMM' : 'vRAM DIMM',
          execute: () => {
            singleActivePub.current = wp.acquireLayer(layerIdx);
          },
        });
      }

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

      const allFfnInQ8K =
        canQ8K(info.ffnGate.type, dim, s.q8kDim) && canQ8K(info.ffnUp.type, dim, s.q8kDim);
      const anyFfnInQ8 =
        !allFfnInQ8K &&
        useQ8 &&
        s.q8Dim !== null &&
        arena !== null &&
        (arena.supportsQ8Activation(info.ffnGate.type) || arena.supportsQ8Activation(info.ffnUp.type));
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
                { info: info.ffnGate, selectHandle: (p) => p.ffnGate, out: s.ffnGate },
                { info: info.ffnUp, selectHandle: (p) => p.ffnUp, out: s.ffnUp },
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
                { weight: lw!.ffnGate, out: s.ffnGate },
                { weight: lw!.ffnUp, out: s.ffnUp },
              ],
              s.q8Dim,
              s.q8kDim,
              s.xNorm,
              anyFfnInQ8,
              allFfnInQ8K,
            ),
        b.nodeSwiGlu(`blk.${l}.ffn_swiglu`, s.ffnGate, s.ffnUp, s.ffnHidden),
      );

      const ffnDownQ8K = canQ8K(info.ffnDown.type, ffnDim, s.q8kFfn);
      if (ffnDownQ8K && s.q8kFfn) {
        singleNodes.push(b.nodeQuantizeQ8K(`blk.${l}.q8k_ffn_down`, s.ffnHidden, s.q8kFfn, ffnDim));
        singleBodyTally.q8kOps += 1;
        singleBodyTally.q8kSb += dim * (ffnDim >>> 8);
      } else if (useQ8 && s.q8Ffn && arena?.supportsQ8Activation(info.ffnDown.type)) {
        singleNodes.push(b.nodeQuantizeQ8(`blk.${l}.q8_ffn_down`, s.ffnHidden, s.q8Ffn, ffnDim));
        singleBodyTally.q80Ops += 1;
      } else {
        singleBodyTally.strictOps += 1;
      }
      singleNodes.push(
        isPaged
          ? makePagedSingleLinear(
              `blk.${l}.ffn_down`,
              info.ffnDown,
              singleActivePub,
              (p) => p.ffnDown,
              s.q8Ffn,
              s.q8kFfn,
              s.ffnHidden,
              s.ffnProj,
            )
          : makeSingleLinear(`blk.${l}.ffn_down`, lw!.ffnDown, s.q8Ffn, s.q8kFfn, s.ffnHidden, s.ffnProj),
        b.nodeAdd(`blk.${l}.ffn_residual`, s.x, s.ffnProj, s.x),
      );
    }

    // Record exact boundary before LM head so non-final prefill tokens can skip output.weight!
    this.singleGraphBodyNodeCount = singleNodes.length;

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

    if (pinnedOutputWeight) {
      singleNodes.push(
        makeSingleLinear('output.weight', pinnedOutputWeight, s.q8Dim, s.q8kDim, s.xNorm, s.logits),
      );
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
          wp.forEachLmHeadStripe((weightStripe, rowStart, rowCount) => {
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
          });
        },
      });
    }

    this.compiledSingleGraph = b.compileGraph(singleNodes);
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

    batchNodes.push({
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
    });

    for (let l = 0; l < layerCount; l++) {
      const layerIdx = l;
      const info = this.layerInfos[l];
      const hot = isPaged ? wp.getHotLayer(l) : residentLayers![l];
      const lw = isPaged ? null : residentLayers![l];
      const bqBasePtr = bs.q.buffer.wasmPtr(bs.q);
      const bAttnOutBasePtr = bs.attnOut.buffer.wasmPtr(bs.attnOut);

      if (isPaged) {
        // Amortizes ONE OPFS coalesced layer restore across all B* = 4 prefill tokens!
        batchNodes.push({
          op: GgmlOp.CPY,
          label: `blk.${l}.vram_dimm_acquire_b4`,
          shapeDesc: `vRAM DIMM L${l} -> B${B} frame`,
          srcType: wp.backingKind === 'opfs-sync' ? 'OPFS DIMM B4' : 'vRAM DIMM B4',
          execute: () => {
            batchActivePub.current = wp.acquireLayer(layerIdx);
          },
        });
      }

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

      const allFfnInB4Q8K =
        canQ8K(info.ffnGate.type, dim, bs.q8kDim) && canQ8K(info.ffnUp.type, dim, bs.q8kDim);
      const anyFfnInQ8 =
        !allFfnInB4Q8K &&
        useQ8 &&
        bs.q8Dim !== null &&
        arena !== null &&
        (arena.supportsQ8Activation(info.ffnGate.type) || arena.supportsQ8Activation(info.ffnUp.type));
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
                { info: info.ffnGate, selectHandle: (p) => p.ffnGate, outBatch: bs.ffnGate },
                { info: info.ffnUp, selectHandle: (p) => p.ffnUp, outBatch: bs.ffnUp },
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
                { weight: lw!.ffnGate, outBatch: bs.ffnGate },
                { weight: lw!.ffnUp, outBatch: bs.ffnUp },
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

      const ffnDownB4Q8K = canQ8K(info.ffnDown.type, ffnDim, bs.q8kFfn);
      if (ffnDownB4Q8K && bs.q8kFfn) {
        batchNodes.push(makeBatchQuantizeQ8K(`blk.${l}.q8k_ffn_down_b4`, bs.ffnHidden, bs.q8kFfn, ffnDim));
        batch4Tally.q8kOps += 1;
        batch4Tally.q8kSb += dim * (ffnDim >>> 8) * B;
      } else if (useQ8 && bs.q8Ffn && arena?.supportsQ8Activation(info.ffnDown.type)) {
        batchNodes.push(makeBatchQuantizeQ8(`blk.${l}.q8_ffn_down_b4`, bs.ffnHidden, bs.q8Ffn, ffnDim));
        batch4Tally.q80Ops += 1;
      } else {
        batch4Tally.strictOps += 1;
      }
      batchNodes.push(
        isPaged
          ? makePagedBatchMulMat(
              `blk.${l}.ffn_down_b4`,
              info.ffnDown,
              (p) => p.ffnDown,
              bs.q8Ffn,
              bs.q8kFfn,
              bs.ffnHidden,
              bs.ffnProj,
            )
          : makeBatchMulMat(
              `blk.${l}.ffn_down_b4`,
              lw!.ffnDown,
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
    }

    this.compiledBatch4Graph = b.compileGraph(batchNodes);
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
  ): Promise<SpeculativePrefillResult> {
    if (!this.tokenizer || !this.hparams || !this.kvPager || !prompt) {
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

    this.unifiedStore?.setSpeculativeActive(true);
    let i = 0;
    let newlyPrefilled = 0;
    let reusedLive = 0;
    let restoredUnified = 0;
    let aborted = false;

    try {
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
  ): AsyncGenerator<TokenStreamEvent, void, unknown> {
    if (!this.tokenizer || !this.hparams || !this.kvPager) {
      throw new Error('No GGUF model loaded in llama.cpp + vTensor runtime.');
    }

    const promptTokens = this.tokenizer.encode(prompt, this.tokenizer.addBosToken);
    // Allow users to input up to 100% of the selected model's native context window with zero artificial reservation.
    const maxPrompt = Math.max(1, this.maxContextWindow);
    const promptTruncatedTokens = Math.max(0, promptTokens.length - maxPrompt);
    let truncatedPrompt: number[];
    if (promptTokens.length > maxPrompt) {
      truncatedPrompt = this.tokenizer.addBosToken
        ? [this.hparams.bosTokenId, ...promptTokens.slice(promptTokens.length - maxPrompt + 1)]
        : promptTokens.slice(promptTokens.length - maxPrompt);
    } else {
      truncatedPrompt = promptTokens;
    }

    if (truncatedPrompt.length === 0) {
      truncatedPrompt = [this.hparams.bosTokenId];
    }

    // Bit-exact live KV-cache prefix reuse + Unified Virtual Store Q8 block restore
    const aligned = this.alignAndRestorePrefix(truncatedPrompt, truncatedPrompt.length - 1);
    const commonPrefix = aligned.commonPrefix;
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
    let i = commonPrefix;
    this.frameBudget.beginSlice();

    while (i < truncatedPrompt.length) {
      if (signal?.aborted) return;

      const tStep0 = performance.now();
      if (
        this.backend.isWasmSimdActive &&
        batchWidth >= PREFILL_BATCH_SIZE &&
        i + PREFILL_BATCH_SIZE <= lastIdx
      ) {
        this.evalPrefillBatch4(truncatedPrompt, i, i);
        for (let k = 0; k < PREFILL_BATCH_SIZE; k++) {
          this.cachedTokens.push(truncatedPrompt[i + k]);
        }
        i += PREFILL_BATCH_SIZE;

        if (onPrefillProgress) {
          onPrefillProgress({
            completedTokens: i - commonPrefix,
            totalTokens: tokensToEval,
            reusedPrefixTokens: commonPrefix,
          });
        }
        const stepUs = (performance.now() - tStep0) * 1000;
        if (
          this.frameBudget.recordStepAndShouldYield(stepUs, GENERATION_SLICE_BUDGET_US, 8)
        ) {
          await this.yielder.yieldNow();
        }
        continue;
      }

      const isLastPrefill = i === lastIdx;
      logits = this.evalToken(truncatedPrompt[i], i, isLastPrefill, isLastPrefill);
      this.cachedTokens.push(truncatedPrompt[i]);
      i += 1;

      if (onPrefillProgress) {
        onPrefillProgress({
          completedTokens: i - commonPrefix,
          totalTokens: tokensToEval,
          reusedPrefixTokens: commonPrefix,
        });
      }
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
      logits = this.evalToken(this.hparams.bosTokenId, 0, true, true);
      this.cachedTokens = [this.hparams.bosTokenId];
    }

    // Commit completed prompt prefix pages into Unified Virtual Store immediately after prefill
    this.commitFullPagesToUnifiedStore(this.cachedTokens);

    const prefillElapsedSec = Math.max(0.001, (performance.now() - prefillStart) / 1000);
    const observedPrefillTokSec = Number((Math.max(1, tokensToEval) / prefillElapsedSec).toFixed(1));

    const decodeStart = performance.now();
    let pos = truncatedPrompt.length;
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

    try {
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
          logits = this.evalToken(nextToken, pos, step === 0, true);
          this.cachedTokens.push(nextToken);
          pos += 1;
        }
      }
    } finally {
      // Persist any newly completed assistant-turn 16-token KV pages into Unified Store
      this.commitFullPagesToUnifiedStore(this.cachedTokens);
    }
  }
}
