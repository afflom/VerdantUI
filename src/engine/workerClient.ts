/**
 * Main-thread client managing the Verdant inference Web Worker,
 * OPFS-backed vRAM persistence for downloaded & imported multi-GB GGUF weights,
 * imported ModelContext harness state, download cancellation, prefill progress reporting,
 * and reactive state subscriptions.
 */

import type { ThreadMessage } from '@assistant-ui/react';
import type { Tool } from 'assistant-stream';
import { formatSpeculativePrefixForLlama } from './assistantAdapter';
import {
  estimateTextTokens,
  UnifiedCognitiveContextGovernor,
} from './cognitiveContext';
import {
  UNIFIED_VERDANT_SYSTEM_PROMPT,
  VerdantCognitiveObserver,
  type CognitiveObserverSnapshot,
  type CognitiveRouteCertificate,
} from './cognitiveObserver';
import {
  loadSavedHarness,
  saveHarnessToStorage,
  type ImportedHarnessSpec,
} from './harnessConfig';
import type { PerceivedAudioReport, WorkerRequest, WorkerResponse } from './inference.worker';
import {
  DEFAULT_SAMPLING_CONFIG,
  type LoadedModelSummary,
  type PrefillProgressEvent,
  type SamplingConfig,
  type TokenStreamEvent,
} from './llamaRuntime';
import {
  INTERNAL_BOOTSTRAP_MODEL,
  MODEL_CATALOG,
  findCatalogModelById,
  formatBytes,
  isInternalBootstrapModel,
  shouldPreferDemandStreaming,
  type CatalogModel,
} from './modelCatalog';
import { selectInitialUnifiedBackbone } from './unifiedExpertScheduler';
import type { MultimodalCoherenceTelemetry } from './multimodalCoordinator';
import type { PrecisionContractMode } from './performanceContract';
import type {
  PerceivedImageReport,
  SemanticVisualQaReport,
  SemanticVisualQaTarget,
} from './visionCritiqueLoop';
import type { OpfsQuotaSnapshot } from './vramBackingStore';
import { trimToStableSpeculativePrefix } from './vramUnifiedStore';
import type { WeightResidencyPolicy } from './vramWeightPager';
import {
  type ConformanceSuiteReport,
  DISABLED_DEFAULT_GGML_PROVIDERS,
  VTENSOR_CONTRACT,
  type VTensorTelemetrySnapshot,
} from './vtensorProvider';

const LEGACY_GGUF_CACHE_NAME = 'vtensor-gguf-models-v1';
const PREFS_STORAGE_KEY = 'vtensor-assistant:prefs:v1';
const LAST_MODEL_STORAGE_KEY = 'vtensor-assistant:last-model:v1';
const MODEL_STREAM_STORAGE_KEY = 'verdant-ui:model-stream-enabled:v1';
const STREAMING_CONSENT_STORAGE_KEY = 'verdant-ui:streaming-consent:v1';

export type StreamingConsentState = 'pending' | 'accepted' | 'paused';

export function loadSavedStreamingConsent(): StreamingConsentState {
  if (typeof window === 'undefined' || typeof window.localStorage === 'undefined') {
    return 'pending';
  }
  try {
    const saved = window.localStorage.getItem(STREAMING_CONSENT_STORAGE_KEY);
    if (saved === 'accepted' || saved === 'paused' || saved === 'pending') {
      return saved;
    }
    const legacy = window.localStorage.getItem(MODEL_STREAM_STORAGE_KEY);
    if (legacy === 'true') return 'accepted';
    if (legacy === 'false') return 'paused';
    return 'pending';
  } catch {
    return 'pending';
  }
}

export function saveStreamingConsent(consent: StreamingConsentState): void {
  if (typeof window === 'undefined' || typeof window.localStorage === 'undefined') return;
  try {
    window.localStorage.setItem(STREAMING_CONSENT_STORAGE_KEY, consent);
    window.localStorage.setItem(MODEL_STREAM_STORAGE_KEY, consent === 'accepted' ? 'true' : 'false');
  } catch {
    // Ignore storage errors
  }
}

export type EngineStatus = 'idle' | 'downloading' | 'loading' | 'ready' | 'generating' | 'error';
export type PromptFormatMode = 'auto' | 'chatml' | 'continuation' | 'instruct';

export interface SpeculativePrefillState {
  readonly active: boolean;
  readonly readyTokens: number;
  readonly targetTokens: number;
}

export interface BackgroundPrewarmState {
  readonly active: boolean;
  readonly phase: 'idle' | 'streaming' | 'warming-context' | 'ready' | 'error';
  readonly percent: number;
  readonly warmBytes: number;
  readonly targetBytes: number;
  readonly casBytesSaved: number;
  readonly bytesPerSecond: number;
  readonly etaSeconds: number | null;
  readonly contextWarmed: boolean;
  readonly errorMessage: string | null;
}

const INITIAL_BACKGROUND_PREWARM: BackgroundPrewarmState = {
  active: false,
  phase: 'idle',
  percent: 0,
  warmBytes: 0,
  targetBytes: 0,
  casBytesSaved: 0,
  bytesPerSecond: 0,
  etaSeconds: null,
  contextWarmed: false,
  errorMessage: null,
};

export interface EngineState {
  readonly status: EngineStatus;
  readonly progressPercent: number;
  readonly statusDetail: string;
  readonly errorMessage: string | null;
  readonly model: LoadedModelSummary | null;
  readonly telemetry: VTensorTelemetrySnapshot;
  readonly lastTokensPerSec: number;
  readonly lastGeneratedTokens: number;
  readonly lastPromptTokens: number;
  readonly lastPromptTruncatedTokens: number;
  readonly prefillProgress: PrefillProgressEvent | null;
  readonly speculativePrefill: SpeculativePrefillState;
  readonly backgroundPrewarm: BackgroundPrewarmState;
  readonly sampling: SamplingConfig;
  readonly samplingCustomized: boolean;
  readonly promptMode: PromptFormatMode;
  readonly precisionMode: PrecisionContractMode;
  readonly weightResidencyPolicy: WeightResidencyPolicy;
  readonly streamingConsent: StreamingConsentState;
  readonly modelStreamEnabled: boolean;
  readonly lastRouteCertificate: CognitiveRouteCertificate | null;
  readonly observerSnapshot: CognitiveObserverSnapshot | null;
  readonly importedHarness: ImportedHarnessSpec | null;
  readonly cachedModelIds: readonly string[];
  readonly opfsStorage: OpfsQuotaSnapshot;
  readonly conformanceReport: ConformanceSuiteReport | null;
  readonly conformanceRunning: boolean;
  readonly multimodalTelemetry: MultimodalCoherenceTelemetry | null;
  readonly pendingImageContext: PerceivedImageReport | null;
  readonly pendingAudioContext: PerceivedAudioReport | null;
  /** Active non-preempted part IDs currently inferring or queued in the unified worker. */
  readonly activePartTaskIds: readonly string[];
}

const INITIAL_OPFS_STORAGE: OpfsQuotaSnapshot = {
  supported: false,
  usageBytes: 0,
  quotaBytes: 0,
  storedModels: [],
  partialModels: [],
};

const INITIAL_TELEMETRY: VTensorTelemetrySnapshot = {
  activeProvider: VTENSOR_CONTRACT.providerName,
  disabledProviders: [...DISABLED_DEFAULT_GGML_PROVIDERS],
  ggmlAbiVersion: VTENSOR_CONTRACT.ggmlAbiVersion,
  verdantVersion: VTENSOR_CONTRACT.verdantVersion,
  llamaCppPinnedCommit: VTENSOR_CONTRACT.llamaCppPinnedCommit,
  kernelAcceleration: 'WASM Relaxed SIMD (VNNI)',
  wasmModuleBytes: 0,
  compiledGraphReuses: 0,
  totalGraphRuns: 0,
  totalOpsExecuted: 0,
  cpuFallbackOps: 0,
  vramKv: null,
  vramWeights: null,
  unifiedStore: null,
  performanceContract: null,
  frameBudget: null,
  threadPool: null,
  prism: null,
  buffers: [],
  opMetrics: [],
  lastGraphTrace: [],
  lastGraphNodeCount: 0,
};

export interface HeaderEngineSummary {
  readonly streamingConsent: StreamingConsentState;
  readonly modelStreamEnabled: boolean;
  readonly status: EngineStatus;
  readonly progressPercent: number;
  readonly initStatusDetail: string;
  readonly backgroundPrewarmActive: boolean;
  readonly backgroundPrewarmPercent: number;
  readonly backgroundPrewarmPhase: BackgroundPrewarmState['phase'];
  readonly backgroundPrewarmBps: number;
  readonly backgroundPrewarmEtaSec: number | null;
  readonly backgroundPrewarmError: string | null;
  readonly contextWarmed: boolean;
  readonly modelId: string | null;
  readonly modelDisplayName: string | null;
  readonly errorMessage: string | null;
  readonly activePartTaskCount: number;
}

export interface AppShellEngineSummary {
  readonly streamingConsent: StreamingConsentState;
  readonly model: LoadedModelSummary | null;
  readonly opfsStorage: OpfsQuotaSnapshot;
  readonly sampling: SamplingConfig;
  readonly samplingCustomized: boolean;
  readonly importedHarness: ImportedHarnessSpec | null;
}

type Listener = (state: EngineState) => void;
type VoidListener = () => void;

function loadSavedPrefs(): {
  sampling: SamplingConfig;
  promptMode: PromptFormatMode;
  precisionMode: PrecisionContractMode;
  weightResidencyPolicy: WeightResidencyPolicy;
} {
  const defaults = {
    sampling: DEFAULT_SAMPLING_CONFIG,
    promptMode: 'auto' as PromptFormatMode,
    precisionMode: 'q8-vnni' as PrecisionContractMode,
    weightResidencyPolicy: 'auto' as WeightResidencyPolicy,
  };
  if (typeof window === 'undefined' || typeof window.localStorage === 'undefined') {
    return defaults;
  }
  try {
    const raw = window.localStorage.getItem(PREFS_STORAGE_KEY);
    if (!raw) return defaults;
    const parsed = JSON.parse(raw) as Partial<{
      sampling: Partial<SamplingConfig>;
      promptMode: PromptFormatMode;
      precisionMode: PrecisionContractMode;
      weightResidencyPolicy: WeightResidencyPolicy;
    }>;
    const s = parsed.sampling ?? {};
    const sampling: SamplingConfig = {
      temperature:
        typeof s.temperature === 'number' && s.temperature >= 0 && s.temperature <= 2
          ? s.temperature
          : DEFAULT_SAMPLING_CONFIG.temperature,
      topP:
        typeof s.topP === 'number' && s.topP >= 0.05 && s.topP <= 1
          ? s.topP
          : DEFAULT_SAMPLING_CONFIG.topP,
      topK:
        typeof s.topK === 'number' && s.topK >= 1 && s.topK <= 200
          ? Math.round(s.topK)
          : DEFAULT_SAMPLING_CONFIG.topK,
      repeatPenalty:
        typeof s.repeatPenalty === 'number' && s.repeatPenalty >= 1 && s.repeatPenalty <= 2
          ? s.repeatPenalty
          : DEFAULT_SAMPLING_CONFIG.repeatPenalty,
      maxTokens:
        typeof s.maxTokens === 'number' && s.maxTokens >= 1 && s.maxTokens <= 1_048_576
          ? Math.round(s.maxTokens)
          : DEFAULT_SAMPLING_CONFIG.maxTokens,
      fullContextOutput:
        typeof s.fullContextOutput === 'boolean'
          ? s.fullContextOutput
          : DEFAULT_SAMPLING_CONFIG.fullContextOutput,
    };
    const promptMode: PromptFormatMode =
      parsed.promptMode === 'chatml' ||
      parsed.promptMode === 'continuation' ||
      parsed.promptMode === 'instruct' ||
      parsed.promptMode === 'auto'
        ? parsed.promptMode
        : 'auto';
    const precisionMode: PrecisionContractMode =
      parsed.precisionMode === 'strict-f64' ? 'strict-f64' : 'q8-vnni';
    const weightResidencyPolicy: WeightResidencyPolicy =
      parsed.weightResidencyPolicy === 'paged' ? 'paged' : 'auto';
    return { sampling, promptMode, precisionMode, weightResidencyPolicy };
  } catch {
    return defaults;
  }
}

function savePrefs(
  sampling: SamplingConfig,
  promptMode: PromptFormatMode,
  precisionMode: PrecisionContractMode,
  weightResidencyPolicy: WeightResidencyPolicy,
): void {
  if (typeof window === 'undefined' || typeof window.localStorage === 'undefined') return;
  try {
    window.localStorage.setItem(
      PREFS_STORAGE_KEY,
      JSON.stringify({
        sampling,
        promptMode,
        precisionMode,
        weightResidencyPolicy,
      }),
    );
  } catch {
    // Ignore storage errors
  }
}

function readLastModelId(): string | null {
  if (typeof window === 'undefined' || typeof window.localStorage === 'undefined') return null;
  try {
    return window.localStorage.getItem(LAST_MODEL_STORAGE_KEY);
  } catch {
    return null;
  }
}

function saveLastModelId(id: string): void {
  if (typeof window === 'undefined' || typeof window.localStorage === 'undefined') return;
  try {
    window.localStorage.setItem(LAST_MODEL_STORAGE_KEY, id);
  } catch {
    // Ignore storage errors
  }
}

function computeCachedIds(opfsStorage: OpfsQuotaSnapshot): string[] {
  const ids = new Set<string>();
  for (const entry of opfsStorage.storedModels) {
    ids.add(entry.modelId);
  }
  return Array.from(ids);
}

export class LlamaVTensorWorkerClient {
  private worker: Worker;
  private reqCounter = 0;
  private activeLoadRequestId: string | null = null;
  private speculativeWordTimer: ReturnType<typeof setTimeout> | null = null;
  private speculativeTailTimer: ReturnType<typeof setTimeout> | null = null;
  private lastSpeculativePrompt = '';
  private activeSpeculativeThreadId: string | null = null;
  private lastTokenStatePatchMs = 0;
  private activeThreadId = 'default-thread';
  private activeToolRegistry: Record<string, Tool<any, any>> | undefined;
  readonly cognitiveObserver = new VerdantCognitiveObserver();
  readonly cognitiveContext = new UnifiedCognitiveContextGovernor();

  getActiveThreadId(): string {
    return this.activeThreadId;
  }

  setActiveThreadId(threadId: string): void {
    this.activeThreadId = threadId || 'default-thread';
  }

  setActiveToolRegistry(tools: Record<string, Tool<any, any>> | undefined): void {
    this.activeToolRegistry = tools;
  }
  private readonly listeners = new Set<Listener>();
  private readonly voidListeners = new Set<VoidListener>();
  private readonly headerSummaryListeners = new Set<VoidListener>();
  private readonly appShellSummaryListeners = new Set<VoidListener>();
  private headerSummary: HeaderEngineSummary;
  private appShellSummary: AppShellEngineSummary;
  private readonly pending = new Map<
    string,
    {
      resolve: (value: unknown) => void;
      reject: (err: Error) => void;
      onToken?: (event: TokenStreamEvent) => void;
    }
  >();

  private state: EngineState;
  private disposed = false;
  private lastRequestedCatalogModel: CatalogModel | null = null;
  private activeLoadPromise: Promise<LoadedModelSummary> | null = null;
  private readonly activeGenerationPartByReq = new Map<string, string>();
  private focusedPartId = 'verdant-part-home';
  private readonly handlePageUnload = (): void => {
    if (this.disposed) return;
    this.disposed = true;
    this.clearSpeculativeTimers();
    try {
      this.worker.postMessage({ type: 'shutdown' } satisfies WorkerRequest);
    } catch {
      // Ignore closed worker
    }
    this.worker.terminate();
  };

  constructor() {
    const saved = loadSavedPrefs();
    const savedHarness = loadSavedHarness();
    const savedConsent = loadSavedStreamingConsent();
    const savedStreamEnabled = savedConsent === 'accepted';
    if (!savedStreamEnabled) {
      this.cognitiveObserver.grants.deny('network-weights');
    } else {
      this.cognitiveObserver.grants.reset('network-weights');
      this.cognitiveObserver.grants.grant('network-weights');
    }
    this.state = {
      status: 'idle',
      progressPercent: 0,
      statusDetail: savedStreamEnabled ? 'Ready' : savedConsent === 'paused' ? 'Offline only' : 'Ready',
      errorMessage: null,
      model: null,
      telemetry: INITIAL_TELEMETRY,
      lastTokensPerSec: 0,
      lastGeneratedTokens: 0,
      lastPromptTokens: 0,
      lastPromptTruncatedTokens: 0,
      prefillProgress: null,
      speculativePrefill: { active: false, readyTokens: 0, targetTokens: 0 },
      backgroundPrewarm: INITIAL_BACKGROUND_PREWARM,
      sampling: saved.sampling,
      samplingCustomized: false,
      promptMode: saved.promptMode,
      precisionMode: saved.precisionMode,
      weightResidencyPolicy: saved.weightResidencyPolicy,
      streamingConsent: savedConsent,
      modelStreamEnabled: savedStreamEnabled,
      lastRouteCertificate: null,
      observerSnapshot: null,
      importedHarness: savedHarness,
      cachedModelIds: [],
      opfsStorage: INITIAL_OPFS_STORAGE,
      conformanceReport: null,
      conformanceRunning: false,
      multimodalTelemetry: null,
      pendingImageContext: null,
      pendingAudioContext: null,
      activePartTaskIds: [],
    };
    this.headerSummary = {
      streamingConsent: savedConsent,
      modelStreamEnabled: savedStreamEnabled,
      status: 'idle',
      progressPercent: 0,
      initStatusDetail: '',
      backgroundPrewarmActive: false,
      backgroundPrewarmPercent: 0,
      backgroundPrewarmPhase: 'idle',
      backgroundPrewarmBps: 0,
      backgroundPrewarmEtaSec: null,
      backgroundPrewarmError: null,
      contextWarmed: false,
      modelId: null,
      modelDisplayName: null,
      errorMessage: null,
      activePartTaskCount: 0,
    };
    this.appShellSummary = {
      streamingConsent: savedConsent,
      model: null,
      opfsStorage: INITIAL_OPFS_STORAGE,
      sampling: saved.sampling,
      samplingCustomized: false,
      importedHarness: savedHarness,
    };

    this.worker = this.spawnWorker();

    if (typeof window !== 'undefined') {
      // Synchronously release OPFS SyncAccessHandles on page reload/navigation so the reloaded
      // document never hits NoModificationAllowedError racing the old worker's teardown!
      window.addEventListener('pagehide', this.handlePageUnload);
      window.addEventListener('beforeunload', this.handlePageUnload);
    }

    if (saved.precisionMode !== 'q8-vnni') {
      void this.updatePrecisionMode(saved.precisionMode).catch(() => {});
    }
    if (saved.weightResidencyPolicy !== 'auto') {
      void this.updateWeightResidencyPolicy(saved.weightResidencyPolicy).catch(() => {});
    }
    void this.refreshOpfsStorage().catch(() => {});
  }

  private spawnWorker(): Worker {
    const worker = new Worker(new URL('./inference.worker.ts', import.meta.url), {
      type: 'module',
    });
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      this.handleWorkerMessage(event.data);
    };
    const handleWorkerFailure = (errorMsg: string) => {
      this.activeLoadRequestId = null;
      this.activeLoadPromise = null;
      this.activeGenerationPartByReq.clear();
      for (const [, pendingEntry] of this.pending) {
        pendingEntry.reject(new Error(errorMsg));
      }
      this.pending.clear();
      this.patchState({
        status: 'error',
        model: null,
        progressPercent: 0,
        prefillProgress: null,
        backgroundPrewarm: INITIAL_BACKGROUND_PREWARM,
        conformanceRunning: false,
        activePartTaskIds: [],
        errorMessage: errorMsg,
        statusDetail: 'Worker restarting',
      });
      if (!this.disposed && this.worker === worker) {
        try {
          worker.terminate();
        } catch {
          // Ignore termination error
        }
        this.worker = this.spawnWorker();
        if (this.state.precisionMode !== 'q8-vnni') {
          void this.sendPrecisionMode(this.state.precisionMode).catch(() => {});
        }
        if (this.state.weightResidencyPolicy !== 'auto') {
          void this.updateWeightResidencyPolicy(this.state.weightResidencyPolicy).catch(() => {});
        }
      }
    };
    worker.onerror = (err) => {
      handleWorkerFailure(err.message || 'Web Worker runtime error.');
    };
    worker.onmessageerror = () => {
      handleWorkerFailure('Web Worker message serialization error.');
    };
    return worker;
  }

  private postRpcWithTimeout<T>(
    req: WorkerRequest & { readonly requestId: string },
    timeoutMs: number,
    transfer?: Transferable[],
  ): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        if (this.pending.delete(req.requestId)) {
          reject(new Error(`Worker request "${req.type}" timed out after ${Math.round(timeoutMs / 1000)}s.`));
        }
      }, timeoutMs);
      this.pending.set(req.requestId, {
        resolve: (v) => {
          clearTimeout(timer);
          resolve(v as T);
        },
        reject: (err) => {
          clearTimeout(timer);
          reject(err);
        },
      });
      if (transfer && transfer.length > 0) {
        this.worker.postMessage(req, transfer);
      } else {
        this.worker.postMessage(req);
      }
    });
  }

  private sendPrecisionMode(mode: PrecisionContractMode): Promise<VTensorTelemetrySnapshot> {
    const requestId = this.nextRequestId('prec');
    return this.postRpcWithTimeout<VTensorTelemetrySnapshot>(
      { type: 'set_precision_mode', requestId, mode },
      8_000,
    );
  }

  /**
   * Executes a critical acceptance/sealing step on the deterministic `strict-f64` contract
   * without mutating the user's persisted preview preference, restoring the previous mode in `finally`.
   */
  async runWithDeterministicPrecision<T>(fn: () => Promise<T>): Promise<T> {
    const prevMode = this.state.precisionMode;
    if (prevMode !== 'strict-f64') {
      this.patchState({ precisionMode: 'strict-f64' });
      await this.sendPrecisionMode('strict-f64').catch(() => undefined);
    }
    try {
      return await fn();
    } finally {
      if (prevMode !== 'strict-f64') {
        this.patchState({ precisionMode: prevMode });
        await this.sendPrecisionMode(prevMode).catch(() => undefined);
      }
    }
  }

  getState(): EngineState {
    return this.state;
  }

  getHeaderSummary(): HeaderEngineSummary {
    return this.headerSummary;
  }

  getAppShellSummary(): AppShellEngineSummary {
    return this.appShellSummary;
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    listener(this.state);
    return () => {
      this.listeners.delete(listener);
    };
  }

  subscribeStore(listener: VoidListener): () => void {
    this.voidListeners.add(listener);
    return () => {
      this.voidListeners.delete(listener);
    };
  }

  /**
   * Fine-grained subscription for top-bar & sidebar status UI: fires ONLY when
   * `modelStreamEnabled`, `status`, `progressPercent`, `model.id`, `errorMessage`, or
   * `activePartTaskIds.length` changes — NEVER on 120 ms decode telemetry ticks.
   */
  subscribeHeaderSummary(listener: VoidListener): () => void {
    this.headerSummaryListeners.add(listener);
    return () => {
      this.headerSummaryListeners.delete(listener);
    };
  }

  /**
   * Fine-grained subscription for root App shell state (`model`, `opfsStorage`, `sampling`,
   * `importedHarness`) — 0 root re-renders during token streaming or prefill telemetry ticks.
   */
  subscribeAppShellSummary(listener: VoidListener): () => void {
    this.appShellSummaryListeners.add(listener);
    return () => {
      this.appShellSummaryListeners.delete(listener);
    };
  }

  /**
   * Immediately mounts the optimal demand-first System Stream backbone (~4-8 MB header+norms,
   * ready in ~1-2s) and starts priority-ordered background prewarming into OPFS CAS.
   */
  private async warmUnifiedModelOnConsent(
    storageOverride?: OpfsQuotaSnapshot,
  ): Promise<LoadedModelSummary> {
    const storage =
      storageOverride ?? (await this.refreshOpfsStorage().catch(() => this.state.opfsStorage));
    const cachedModelIds = computeCachedIds(storage);
    const backbone = selectInitialUnifiedBackbone({
      cachedModelIds,
      lastModelId: readLastModelId(),
      hardwareConcurrency:
        typeof navigator !== 'undefined' && typeof navigator.hardwareConcurrency === 'number'
          ? navigator.hardwareConcurrency
          : 4,
    });
    return this.loadCatalogModel(backbone);
  }

  /**
   * Waits for an in-flight model mount (e.g., consent-triggered ~1-2s header mount) to finish
   * without rejecting user turns or double-downloading.
   */
  async waitForActiveLoad(signal?: AbortSignal): Promise<LoadedModelSummary | null> {
    const pendingLoad = this.activeLoadPromise;
    if (!pendingLoad) return this.state.model;
    if (signal?.aborted) return null;
    if (!signal) {
      return pendingLoad.catch(() => this.state.model);
    }
    return new Promise<LoadedModelSummary | null>((resolve) => {
      const onAbort = () => resolve(null);
      signal.addEventListener('abort', onAbort, { once: true });
      pendingLoad
        .then((summary) => {
          signal.removeEventListener('abort', onAbort);
          resolve(summary);
        })
        .catch(() => {
          signal.removeEventListener('abort', onAbort);
          resolve(this.state.model);
        });
    });
  }

  /**
   * Boots the user's demand-first Unified Model backbone immediately when streaming consent is
   * already accepted (re-attaching warm OPFS slabs or mounting header+norms in ~1-2s with
   * cooperative background warming). When consent is `'pending'` or `'paused'` (Offline-Only),
   * boots the local offline bootstrap runtime with 0 network bytes.
   */
  async bootInitialModel(): Promise<LoadedModelSummary> {
    // Avoid racing OPFS handle creation when coi-serviceworker.js is negotiating a one-shot
    // COOP/COEP reload on first visit (prevents transient OPFS lock during reload handover).
    if (typeof window !== 'undefined') {
      const coiWin = window as unknown as {
        __VERDANT_COI_RELOADING?: boolean;
        __VERDANT_COI_PENDING?: boolean;
      };
      if (coiWin.__VERDANT_COI_PENDING && !coiWin.__VERDANT_COI_RELOADING) {
        const deadline = Date.now() + 450;
        while (coiWin.__VERDANT_COI_PENDING && !coiWin.__VERDANT_COI_RELOADING && Date.now() < deadline) {
          await new Promise((resolve) => setTimeout(resolve, 25));
        }
      }
      if (coiWin.__VERDANT_COI_RELOADING) {
        // Page is reloading into cross-origin isolation right now — don't lock OPFS!
        return new Promise<LoadedModelSummary>(() => {});
      }
    }

    if (this.state.streamingConsent === 'accepted') {
      this.cognitiveObserver.chronology.beginObject({
        objectId: 'app:init',
        modality: 'system',
        kind: 'app-init',
        label: 'Application initialization',
        budgetMs: this.cognitiveObserver.chronology.slas.appInitTargetMs,
      });
      this.patchState({
        status: 'downloading',
        progressPercent: 2,
        errorMessage: null,
        statusDetail: 'Checking local cache',
      });
    }
    const storage = await this.refreshOpfsStorage().catch(() => INITIAL_OPFS_STORAGE);
    if (this.state.streamingConsent === 'accepted') {
      return this.warmUnifiedModelOnConsent(storage);
    }
    return this.loadCatalogModel(INTERNAL_BOOTSTRAP_MODEL);
  }

  /**
   * Sets the user's browser-streaming consent (`'accepted'` vs `'paused'` Offline-Only Mode).
   * - `'accepted'` grants `network-weights` and IMMEDIATELY mounts the demand-first System Stream
   *   backbone (~4-8 MB header+norms, ready in ~1-2s) + starts interruptible priority-ordered
   *   background warming into OPFS CAS (NEVER waiting until the user submits a prompt!).
   * - `'paused'` immediately pauses background prewarming, cancels in-flight transfers, and denies
   *   `network-weights` (`FK-07` strict offline mode).
   */
  async setStreamingConsent(consent: StreamingConsentState): Promise<void> {
    saveStreamingConsent(consent);
    const enabled = consent === 'accepted';
    if (enabled) {
      this.cognitiveObserver.grants.reset('network-weights');
      this.cognitiveObserver.grants.grant('network-weights');
      const activeId = this.state.model?.id;
      const alreadyLoadedRealModel =
        activeId !== undefined && activeId !== INTERNAL_BOOTSTRAP_MODEL.id;
      const isAlreadyLoading =
        this.state.status === 'downloading' || this.state.status === 'loading';
      this.patchState({
        streamingConsent: 'accepted',
        modelStreamEnabled: true,
        errorMessage: null,
        ...(alreadyLoadedRealModel || isAlreadyLoading
          ? { statusDetail: alreadyLoadedRealModel ? 'Ready' : this.state.statusDetail }
          : {
              status: 'downloading',
              progressPercent: 2,
              statusDetail: 'Checking local cache',
            }),
      });

      if (!alreadyLoadedRealModel && !isAlreadyLoading) {
        this.cognitiveObserver.chronology.beginObject({
          objectId: 'app:init',
          modality: 'system',
          kind: 'app-init',
          label: 'Application initialization',
          budgetMs: this.cognitiveObserver.chronology.slas.appInitTargetMs,
        });
        await this.warmUnifiedModelOnConsent().catch(() => undefined);
      }
    } else {
      this.cognitiveObserver.grants.deny('network-weights');
      try {
        this.worker.postMessage({ type: 'pause_prewarm' } satisfies WorkerRequest);
      } catch {
        // Ignore closed worker
      }
      this.cancelDownload();
      this.patchState({
        streamingConsent: consent,
        modelStreamEnabled: false,
        backgroundPrewarm: INITIAL_BACKGROUND_PREWARM,
        status: this.state.model ? 'ready' : 'idle',
        progressPercent: 0,
        statusDetail: consent === 'paused' ? 'Offline only' : 'Ready',
        errorMessage: null,
      });
      if (consent === 'paused' && !this.state.model) {
        await this.loadCatalogModel(INTERNAL_BOOTSTRAP_MODEL).catch(() => undefined);
      }
    }
  }

  async setModelStreamEnabled(enabled: boolean): Promise<void> {
    return this.setStreamingConsent(enabled ? 'accepted' : 'paused');
  }

  /**
   * Imports a validated standard harness specification (mapped to Assistant-UI's ModelContext).
   * Applies any explicitly provided model parameters from the harness without fabricating defaults.
   */
  setImportedHarness(harness: ImportedHarnessSpec | null): void {
    saveHarnessToStorage(harness);
    const nextSampling: SamplingConfig = harness
      ? {
          ...this.state.sampling,
          ...(typeof harness.callSettings?.temperature === 'number'
            ? { temperature: harness.callSettings.temperature }
            : {}),
          ...(typeof harness.callSettings?.topP === 'number'
            ? { topP: harness.callSettings.topP }
            : {}),
          ...(typeof harness.topK === 'number' ? { topK: harness.topK } : {}),
          ...(typeof harness.repeatPenalty === 'number'
            ? { repeatPenalty: harness.repeatPenalty }
            : {}),
          ...(typeof harness.callSettings?.maxTokens === 'number'
            ? { maxTokens: harness.callSettings.maxTokens, fullContextOutput: false }
            : {}),
        }
      : this.state.sampling;

    const nextPromptMode = harness?.promptMode ?? this.state.promptMode;

    savePrefs(
      nextSampling,
      nextPromptMode,
      this.state.precisionMode,
      this.state.weightResidencyPolicy,
    );
    this.lastSpeculativePrompt = '';
    this.patchState({
      importedHarness: harness,
      sampling: nextSampling,
      promptMode: nextPromptMode,
    });
  }

  updateSampling(patch: Partial<SamplingConfig>): void {
    const nextSampling = { ...this.state.sampling, ...patch };
    savePrefs(
      nextSampling,
      this.state.promptMode,
      this.state.precisionMode,
      this.state.weightResidencyPolicy,
    );
    this.patchState({ sampling: nextSampling, samplingCustomized: true });
  }

  updatePromptMode(promptMode: PromptFormatMode): void {
    savePrefs(
      this.state.sampling,
      promptMode,
      this.state.precisionMode,
      this.state.weightResidencyPolicy,
    );
    this.patchState({ promptMode });
  }

  async updatePrecisionMode(precisionMode: PrecisionContractMode): Promise<VTensorTelemetrySnapshot> {
    savePrefs(
      this.state.sampling,
      this.state.promptMode,
      precisionMode,
      this.state.weightResidencyPolicy,
    );
    this.patchState({ precisionMode });
    return this.sendPrecisionMode(precisionMode);
  }

  async updateWeightResidencyPolicy(
    policy: WeightResidencyPolicy,
  ): Promise<VTensorTelemetrySnapshot> {
    savePrefs(
      this.state.sampling,
      this.state.promptMode,
      this.state.precisionMode,
      policy,
    );
    this.patchState({ weightResidencyPolicy: policy });
    const requestId = this.nextRequestId('wres');
    return this.postRpcWithTimeout<VTensorTelemetrySnapshot>(
      { type: 'set_weight_residency_policy', requestId, policy },
      8_000,
    );
  }

  resetSampling(): void {
    savePrefs(DEFAULT_SAMPLING_CONFIG, 'auto', 'q8-vnni', 'auto');
    this.patchState({
      sampling: DEFAULT_SAMPLING_CONFIG,
      samplingCustomized: false,
      promptMode: 'auto',
      precisionMode: 'q8-vnni',
      weightResidencyPolicy: 'auto',
    });
    void this.updatePrecisionMode('q8-vnni').catch(() => {});
    void this.updateWeightResidencyPolicy('auto').catch(() => {});
  }

  cancelDownload(): void {
    if (this.activeLoadRequestId) {
      this.worker.postMessage({
        type: 'cancel',
        requestId: this.activeLoadRequestId,
      } satisfies WorkerRequest);
      this.activeLoadRequestId = null;
    }
  }

  async runConformanceCheck(): Promise<ConformanceSuiteReport> {
    this.patchState({ conformanceRunning: true });
    const requestId = this.nextRequestId('conf');
    return new Promise<ConformanceSuiteReport>((resolve, reject) => {
      this.pending.set(requestId, {
        resolve: resolve as (v: unknown) => void,
        reject: (err) => {
          this.patchState({ conformanceRunning: false });
          reject(err);
        },
      });
      this.worker.postMessage({ type: 'run_conformance', requestId } satisfies WorkerRequest);
    });
  }

  async countTokens(prompt: string): Promise<{ tokenCount: number; maxContextWindow: number }> {
    if (!this.state.model || !prompt) {
      return { tokenCount: 0, maxContextWindow: this.state.model?.maxContextWindow ?? 0 };
    }
    const requestId = this.nextRequestId('tok');
    return new Promise<{ tokenCount: number; maxContextWindow: number }>((resolve, reject) => {
      this.pending.set(requestId, {
        resolve: resolve as (v: unknown) => void,
        reject,
      });
      this.worker.postMessage({
        type: 'count_tokens',
        requestId,
        prompt,
      } satisfies WorkerRequest);
    });
  }

  async refreshOpfsStorage(): Promise<OpfsQuotaSnapshot> {
    const requestId = this.nextRequestId('opfs');
    return new Promise<OpfsQuotaSnapshot>((resolve, reject) => {
      this.pending.set(requestId, {
        resolve: resolve as (v: unknown) => void,
        reject,
      });
      this.worker.postMessage({ type: 'inspect_opfs', requestId } satisfies WorkerRequest);
    });
  }

  async deleteCachedModel(modelId: string): Promise<OpfsQuotaSnapshot> {
    const requestId = this.nextRequestId('del');
    return new Promise<OpfsQuotaSnapshot>((resolve, reject) => {
      this.pending.set(requestId, {
        resolve: resolve as (v: unknown) => void,
        reject,
      });
      this.worker.postMessage({
        type: 'delete_opfs_model',
        requestId,
        modelId,
      } satisfies WorkerRequest);
    });
  }

  async clearModelCache(): Promise<void> {
    if (typeof caches !== 'undefined') {
      try {
        await caches.delete(LEGACY_GGUF_CACHE_NAME);
      } catch {
        // Ignore storage error
      }
    }
    const requestId = this.nextRequestId('clr');
    await new Promise<OpfsQuotaSnapshot>((resolve, reject) => {
      this.pending.set(requestId, {
        resolve: resolve as (v: unknown) => void,
        reject,
      });
      this.worker.postMessage({ type: 'clear_opfs_models', requestId } satisfies WorkerRequest);
    });
  }

  private patchState(patch: Partial<EngineState>): void {
    this.state = { ...this.state, ...patch };
    for (const listener of this.listeners) {
      listener(this.state);
    }
    for (const listener of this.voidListeners) {
      listener();
    }

    const prevH = this.headerSummary;
    const nextModelId = this.state.model?.id ?? null;
    const nextModelName = this.state.model?.displayName ?? null;
    const nextActiveCount = this.state.activePartTaskIds.length;
    const nextPrewarmActive = this.state.backgroundPrewarm.active;
    const nextPrewarmPercent = this.state.backgroundPrewarm.percent;
    const nextPrewarmPhase = this.state.backgroundPrewarm.phase;
    const nextPrewarmBps = this.state.backgroundPrewarm.bytesPerSecond;
    const nextPrewarmEtaSec = this.state.backgroundPrewarm.etaSeconds;
    const nextPrewarmError = this.state.backgroundPrewarm.errorMessage;
    const nextContextWarmed = this.state.backgroundPrewarm.contextWarmed;
    const nextInitStatusDetail =
      this.state.status === 'downloading' ||
      this.state.status === 'loading' ||
      this.state.status === 'error'
        ? this.state.statusDetail
        : '';
    if (
      prevH.streamingConsent !== this.state.streamingConsent ||
      prevH.modelStreamEnabled !== this.state.modelStreamEnabled ||
      prevH.status !== this.state.status ||
      prevH.progressPercent !== this.state.progressPercent ||
      prevH.initStatusDetail !== nextInitStatusDetail ||
      prevH.backgroundPrewarmActive !== nextPrewarmActive ||
      prevH.backgroundPrewarmPercent !== nextPrewarmPercent ||
      prevH.backgroundPrewarmPhase !== nextPrewarmPhase ||
      prevH.backgroundPrewarmBps !== nextPrewarmBps ||
      prevH.backgroundPrewarmEtaSec !== nextPrewarmEtaSec ||
      prevH.backgroundPrewarmError !== nextPrewarmError ||
      prevH.contextWarmed !== nextContextWarmed ||
      prevH.modelId !== nextModelId ||
      prevH.modelDisplayName !== nextModelName ||
      prevH.errorMessage !== this.state.errorMessage ||
      prevH.activePartTaskCount !== nextActiveCount
    ) {
      this.headerSummary = {
        streamingConsent: this.state.streamingConsent,
        modelStreamEnabled: this.state.modelStreamEnabled,
        status: this.state.status,
        progressPercent: this.state.progressPercent,
        initStatusDetail: nextInitStatusDetail,
        backgroundPrewarmActive: nextPrewarmActive,
        backgroundPrewarmPercent: nextPrewarmPercent,
        backgroundPrewarmPhase: nextPrewarmPhase,
        backgroundPrewarmBps: nextPrewarmBps,
        backgroundPrewarmEtaSec: nextPrewarmEtaSec,
        backgroundPrewarmError: nextPrewarmError,
        contextWarmed: nextContextWarmed,
        modelId: nextModelId,
        modelDisplayName: nextModelName,
        errorMessage: this.state.errorMessage,
        activePartTaskCount: nextActiveCount,
      };
      for (const listener of this.headerSummaryListeners) {
        listener();
      }
    }

    const prevS = this.appShellSummary;
    if (
      prevS.streamingConsent !== this.state.streamingConsent ||
      prevS.model !== this.state.model ||
      prevS.opfsStorage !== this.state.opfsStorage ||
      prevS.sampling !== this.state.sampling ||
      prevS.samplingCustomized !== this.state.samplingCustomized ||
      prevS.importedHarness !== this.state.importedHarness
    ) {
      this.appShellSummary = {
        streamingConsent: this.state.streamingConsent,
        model: this.state.model,
        opfsStorage: this.state.opfsStorage,
        sampling: this.state.sampling,
        samplingCustomized: this.state.samplingCustomized,
        importedHarness: this.state.importedHarness,
      };
      for (const listener of this.appShellSummaryListeners) {
        listener();
      }
    }
  }

  private nextRequestId(prefix: string): string {
    this.reqCounter += 1;
    return `${prefix}-${this.reqCounter}`;
  }

  private clearSpeculativeTimers(): void {
    if (this.speculativeWordTimer !== null) {
      clearTimeout(this.speculativeWordTimer);
      this.speculativeWordTimer = null;
    }
    if (this.speculativeTailTimer !== null) {
      clearTimeout(this.speculativeTailTimer);
      this.speculativeTailTimer = null;
    }
  }

  private canSpeculativePrefillModel(): boolean {
    if (this.state.status !== 'ready' || !this.state.model) return false;
    if (isInternalBootstrapModel(this.state.model.id)) return false;
    if (this.state.model.backingKind === 'opfs-demand') {
      return !this.state.backgroundPrewarm.active && this.state.backgroundPrewarm.percent >= 100;
    }
    return true;
  }

  private dispatchSpeculativePrefill(prompt: string, threadId?: string): void {
    if (!prompt || !this.canSpeculativePrefillModel()) return;
    if (prompt === this.lastSpeculativePrompt && threadId === this.activeSpeculativeThreadId) return;
    this.lastSpeculativePrompt = prompt;
    if (threadId !== undefined) {
      this.activeSpeculativeThreadId = threadId;
    }
    const requestId = this.nextRequestId('spec');
    this.worker.postMessage({
      type: 'speculative_prefill',
      requestId,
      prompt,
      threadId,
    } satisfies WorkerRequest);
  }

  recordCognitiveObservation(
    certificate: CognitiveRouteCertificate,
    snapshot: CognitiveObserverSnapshot,
  ): void {
    this.patchState({
      lastRouteCertificate: certificate,
      observerSnapshot: snapshot,
    });
  }

  /**
   * Seamlessly activates/streams a CostSurface-selected neural route on demand (inline within a turn),
   * honoring an external AbortSignal if the user clicks Stop.
   */
  async ensureUnifiedNeuralRoute(
    modelId: string,
    signal?: AbortSignal,
  ): Promise<LoadedModelSummary> {
    if (this.state.model && this.state.model.id === modelId) {
      return this.state.model;
    }
    const target = findCatalogModelById(modelId);
    if (!target) {
      throw new Error(`Unified neural route "${modelId}" not found in catalog.`);
    }
    if (signal?.aborted) {
      throw new Error('Download cancelled.');
    }
    const onAbort = () => this.cancelDownload();
    signal?.addEventListener('abort', onAbort, { once: true });
    try {
      return await this.loadCatalogModel(target);
    } finally {
      signal?.removeEventListener('abort', onAbort);
    }
  }

  /**
   * Schedules cooperative Continuous Speculative Prefill for the active thread and live composer draft.
   */
  scheduleSpeculativePrefill(
    messages: readonly ThreadMessage[],
    draftText: string,
    systemMessage?: string,
    threadId?: string,
  ): void {
    this.clearSpeculativeTimers();
    if (threadId && this.activeSpeculativeThreadId && threadId !== this.activeSpeculativeThreadId) {
      this.lastSpeculativePrompt = '';
    }
    if (!this.canSpeculativePrefillModel()) return;
    const model = this.state.model!;
    const mode = this.state.importedHarness?.promptMode ?? this.state.promptMode;
    const systemPrompt =
      systemMessage ?? this.state.importedHarness?.system ?? UNIFIED_VERDANT_SYSTEM_PROMPT;
    const tid = threadId || this.activeThreadId;
    const latchedTools =
      this.cognitiveObserver.isToolsLatched(tid) &&
      this.activeToolRegistry &&
      Object.keys(this.activeToolRegistry).length > 0
        ? this.activeToolRegistry
        : undefined;
    const projected = this.cognitiveContext.projectContext(tid, messages, {
      latestUserText: '',
      belief: this.cognitiveObserver.getThreadSnapshot(tid).belief,
      maxContextWindow: model.maxContextWindow,
      systemTokensEstimate: estimateTextTokens(systemPrompt) + (latchedTools ? 380 : 0),
    });
    const specOpts = {
      tools: latchedTools,
      epochManifestBlock: projected.epochManifestBlock,
      preserveEmptyThinkPrefix: true,
    };
    const stableWordDraft = trimToStableSpeculativePrefix(draftText, false);
    const fullDraft = trimToStableSpeculativePrefix(draftText, true);

    const wordPrompt = formatSpeculativePrefixForLlama(
      projected.messages,
      stableWordDraft,
      model,
      mode,
      systemPrompt,
      specOpts,
    );
    if (
      wordPrompt &&
      (wordPrompt !== this.lastSpeculativePrompt || threadId !== this.activeSpeculativeThreadId)
    ) {
      const delayMs = draftText.trim().length === 0 ? 60 : 160;
      this.speculativeWordTimer = setTimeout(() => {
        this.speculativeWordTimer = null;
        this.dispatchSpeculativePrefill(wordPrompt, threadId);
      }, delayMs);
    }

    if (fullDraft !== stableWordDraft && fullDraft.length > 0) {
      const tailPrompt = formatSpeculativePrefixForLlama(
        projected.messages,
        fullDraft,
        model,
        mode,
        systemPrompt,
        specOpts,
      );
      if (
        tailPrompt &&
        (tailPrompt !== this.lastSpeculativePrompt || threadId !== this.activeSpeculativeThreadId)
      ) {
        this.speculativeTailTimer = setTimeout(() => {
          this.speculativeTailTimer = null;
          this.dispatchSpeculativePrefill(tailPrompt, threadId);
        }, 460);
      }
    }
  }

  private handleWorkerMessage(msg: WorkerResponse): void {
    if (msg.type === 'speculative_status') {
      this.patchState({
        speculativePrefill: {
          active: msg.active,
          readyTokens: msg.readyTokens,
          targetTokens: msg.targetTokens,
        },
        ...(msg.telemetry ? { telemetry: msg.telemetry } : {}),
      });
      return;
    }

    if (msg.type === 'prewarm_status') {
      const wasWarm =
        !this.state.backgroundPrewarm.active && this.state.backgroundPrewarm.contextWarmed;
      if (typeof msg.bytesPerSecond === 'number' && msg.bytesPerSecond > 0) {
        this.cognitiveObserver.chronology.recordNetworkThroughput(msg.bytesPerSecond);
      }
      if (!msg.active && msg.contextWarmed) {
        this.cognitiveObserver.chronology.completeObject('app:init');
      }
      this.patchState({
        backgroundPrewarm: {
          active: msg.active,
          phase: msg.phase,
          percent: msg.percent,
          warmBytes: msg.warmBytes,
          targetBytes: msg.targetBytes,
          casBytesSaved: msg.casBytesSaved,
          bytesPerSecond: msg.bytesPerSecond ?? 0,
          etaSeconds: msg.etaSeconds ?? null,
          contextWarmed: msg.contextWarmed,
          errorMessage: msg.errorMessage ?? null,
        },
      });
      // Once warm & context-ready, record bootPrefix as already prefilled so live composer draft
      // speculative prefill seamlessly extends from the warmed system-prompt KV prefix!
      if (
        !wasWarm &&
        !msg.active &&
        msg.contextWarmed &&
        this.state.status === 'ready' &&
        this.state.model &&
        !isInternalBootstrapModel(this.state.model.id)
      ) {
        const bootSystem = this.state.importedHarness?.system ?? UNIFIED_VERDANT_SYSTEM_PROMPT;
        const bootPrefix = formatSpeculativePrefixForLlama(
          [],
          '',
          this.state.model,
          this.state.importedHarness?.promptMode ?? this.state.promptMode,
          bootSystem,
        );
        if (bootPrefix) {
          this.lastSpeculativePrompt = bootPrefix;
        }
      }
      return;
    }

    const entry = this.pending.get(msg.requestId);
    if (!entry) return;

    if (msg.type === 'progress') {
      this.patchState({
        status: msg.percent < 96 ? 'downloading' : 'loading',
        progressPercent: msg.percent,
        statusDetail: msg.detail,
      });
      return;
    }

    if (msg.type === 'loaded') {
      this.pending.delete(msg.requestId);
      if (this.activeLoadRequestId === msg.requestId) {
        this.activeLoadRequestId = null;
      }
      if (MODEL_CATALOG.some((m) => m.id === msg.summary.id)) {
        saveLastModelId(msg.summary.id);
      }
      this.clearSpeculativeTimers();
      this.lastSpeculativePrompt = '';
      const isInternalBootstrap = isInternalBootstrapModel(msg.summary.id);
      this.patchState({
        status: 'ready',
        progressPercent: 100,
        statusDetail: isInternalBootstrap
          ? 'Ready on your device'
          : `${msg.summary.displayName} ready`,
        errorMessage: null,
        prefillProgress: null,
        speculativePrefill: { active: false, readyTokens: 0, targetTokens: 0 },
        backgroundPrewarm: INITIAL_BACKGROUND_PREWARM,
        model: msg.summary,
        telemetry: msg.telemetry,
        opfsStorage: msg.opfsStorage,
        cachedModelIds: computeCachedIds(msg.opfsStorage),
      });
      // Warm initial prompt prefix for real resident models (skip bootstrap & demand-paged models)
      if (!isInternalBootstrap && msg.summary.backingKind !== 'opfs-demand') {
        this.cognitiveObserver.chronology.completeObject('app:init');
        const bootSystem = this.state.importedHarness?.system ?? UNIFIED_VERDANT_SYSTEM_PROMPT;
        const bootPrefix = formatSpeculativePrefixForLlama(
          [],
          '',
          msg.summary,
          this.state.importedHarness?.promptMode ?? this.state.promptMode,
          bootSystem,
        );
        if (bootPrefix) {
          this.dispatchSpeculativePrefill(bootPrefix);
        }
      }
      entry.resolve(msg.summary);
      return;
    }

    if (msg.type === 'prefill') {
      const p = msg.progress;
      this.patchState({
        prefillProgress: p,
        statusDetail: 'Preparing response…',
      });
      return;
    }

    if (msg.type === 'token') {
      // Deliver every token immediately to Assistant-UI's stream consumer with zero latency.
      entry.onToken?.(msg.event);
      // Throttle top-level React App engineState re-renders to ~120ms so fast decode (50-150 tok/s)
      // never saturates the main thread with redundant full-tree React reconciliation.
      const nowMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
      const shouldPatchUi =
        this.state.prefillProgress !== null ||
        msg.telemetry !== undefined ||
        msg.event.finished ||
        nowMs - this.lastTokenStatePatchMs >= 120;
      if (shouldPatchUi) {
        this.lastTokenStatePatchMs = nowMs;
        this.patchState({
          prefillProgress: null,
          statusDetail: 'Generating response',
          lastTokensPerSec: msg.event.tokensPerSecond,
          lastGeneratedTokens: msg.event.generatedTokens,
          lastPromptTokens: msg.event.promptTokens,
          lastPromptTruncatedTokens: msg.event.promptTruncatedTokens ?? 0,
          ...(msg.telemetry ? { telemetry: msg.telemetry } : {}),
        });
      }
      return;
    }

    if (msg.type === 'done') {
      this.pending.delete(msg.requestId);
      this.activeGenerationPartByReq.delete(msg.requestId);
      const remainingParts = Array.from(new Set(this.activeGenerationPartByReq.values()));
      this.patchState({
        status: remainingParts.length > 0 ? 'generating' : 'ready',
        prefillProgress: null,
        statusDetail:
          remainingParts.length > 0
            ? 'Generating response'
            : `${this.state.model?.displayName ?? 'Model'} ready`,
        telemetry: msg.telemetry,
        activePartTaskIds: remainingParts,
        ...(msg.multimodalTelemetry ? { multimodalTelemetry: msg.multimodalTelemetry } : {}),
      });
      entry.resolve(undefined);
      return;
    }

    if (msg.type === 'telemetry') {
      this.pending.delete(msg.requestId);
      this.patchState({
        telemetry: msg.telemetry,
        ...(msg.multimodalTelemetry ? { multimodalTelemetry: msg.multimodalTelemetry } : {}),
      });
      entry.resolve(msg.telemetry);
      return;
    }

    if (msg.type === 'image_perceived') {
      this.pending.delete(msg.requestId);
      this.patchState({
        pendingImageContext: msg.report,
        multimodalTelemetry: msg.multimodalTelemetry,
      });
      entry.resolve(msg.report);
      return;
    }

    if (msg.type === 'audio_perceived') {
      this.pending.delete(msg.requestId);
      this.patchState({
        pendingAudioContext: msg.report,
        multimodalTelemetry: msg.multimodalTelemetry,
      });
      entry.resolve(msg.report);
      return;
    }

    if (msg.type === 'audio_synthesized') {
      this.pending.delete(msg.requestId);
      this.patchState({ multimodalTelemetry: msg.multimodalTelemetry });
      entry.resolve({ wavBytes: new Uint8Array(msg.wavBuffer), sampleRate: msg.sampleRate, durationSec: msg.durationSec });
      return;
    }

    if (msg.type === 'image_synthesized') {
      this.pending.delete(msg.requestId);
      this.patchState({ multimodalTelemetry: msg.multimodalTelemetry });
      entry.resolve({
        neuralAvailable: msg.neuralAvailable,
        width: msg.width,
        height: msg.height,
        rgba: msg.rgbaBuffer ? new Uint8ClampedArray(msg.rgbaBuffer) : null,
      });
      return;
    }

    if (msg.type === 'visual_qa_result') {
      this.pending.delete(msg.requestId);
      this.patchState({ multimodalTelemetry: msg.multimodalTelemetry });
      entry.resolve(msg.report);
      return;
    }

    if (msg.type === 'token_count') {
      this.pending.delete(msg.requestId);
      entry.resolve({ tokenCount: msg.tokenCount, maxContextWindow: msg.maxContextWindow });
      return;
    }

    if (msg.type === 'opfs_storage') {
      this.pending.delete(msg.requestId);
      const nextModel = msg.modelUnloaded ? null : this.state.model;
      this.patchState({
        model: nextModel,
        ...(msg.modelUnloaded ? { status: 'idle' as EngineStatus, statusDetail: 'Model unloaded' } : {}),
        opfsStorage: msg.storage,
        cachedModelIds: computeCachedIds(msg.storage),
      });
      entry.resolve(msg.storage);
      return;
    }

    if (msg.type === 'conformance') {
      this.pending.delete(msg.requestId);
      this.patchState({
        conformanceReport: msg.report,
        conformanceRunning: false,
      });
      entry.resolve(msg.report);
      return;
    }

    if (msg.type === 'error') {
      this.pending.delete(msg.requestId);
      const removedActivePart = this.activeGenerationPartByReq.delete(msg.requestId);
      const remainingParts = removedActivePart
        ? Array.from(new Set(this.activeGenerationPartByReq.values()))
        : this.state.activePartTaskIds;
      if (this.activeLoadRequestId === msg.requestId) {
        this.activeLoadRequestId = null;
      }
      const isCancelled = msg.message === 'Download cancelled.';
      const nextModel = msg.modelUnloaded ? null : this.state.model;
      this.patchState({
        model: nextModel,
        status:
          remainingParts.length > 0
            ? 'generating'
            : nextModel
              ? 'ready'
              : isCancelled
                ? 'idle'
                : 'error',
        progressPercent: 0,
        prefillProgress: null,
        activePartTaskIds: remainingParts,
        errorMessage: isCancelled ? null : msg.message,
        statusDetail: isCancelled ? 'Download paused' : 'Model error',
        ...(msg.opfsStorage
          ? {
              opfsStorage: msg.opfsStorage,
              cachedModelIds: computeCachedIds(msg.opfsStorage),
            }
          : {}),
      });
      entry.reject(new Error(msg.message));
    }
  }

  private resolveWorkerFetchUrl(url: string): string {
    if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(url)) {
      return url;
    }
    if (typeof document !== 'undefined' && document.baseURI) {
      return new URL(url.replace(/^\/+/, ''), document.baseURI).href;
    }
    if (typeof window !== 'undefined' && window.location?.href) {
      return new URL(url.replace(/^\/+/, ''), window.location.href).href;
    }
    return url;
  }

  async loadCatalogModel(model: CatalogModel): Promise<LoadedModelSummary> {
    this.lastRequestedCatalogModel = model;
    const isCachedLocally = this.state.cachedModelIds.includes(model.id);
    if (
      model.id !== INTERNAL_BOOTSTRAP_MODEL.id &&
      !isCachedLocally &&
      this.state.streamingConsent === 'paused'
    ) {
      throw new Error(
        'Offline-only mode is active. Turn off Offline-Only Mode in Settings → Content Streams to stream new capabilities.',
      );
    }
    if (model.id !== INTERNAL_BOOTSTRAP_MODEL.id && this.state.streamingConsent === 'pending') {
      saveStreamingConsent('accepted');
      this.cognitiveObserver.grants.reset('network-weights');
      this.cognitiveObserver.grants.grant('network-weights');
      this.patchState({ streamingConsent: 'accepted', modelStreamEnabled: true });
    }
    return this.loadFromUrl(
      model.id,
      model.name,
      model.url,
      model.bundled,
      shouldPreferDemandStreaming(model),
    );
  }

  private buildBootSpeculativePrompt(modelId: string): string | undefined {
    if (isInternalBootstrapModel(modelId)) return undefined;
    const bootSystem = this.state.importedHarness?.system ?? UNIFIED_VERDANT_SYSTEM_PROMPT;
    const prompt = formatSpeculativePrefixForLlama(
      [],
      '',
      modelId,
      this.state.importedHarness?.promptMode ?? this.state.promptMode,
      bootSystem,
    );
    return prompt || undefined;
  }

  /** Retries the most recently requested catalog model (or resumes interrupted prewarming). */
  async retryLastLoad(): Promise<LoadedModelSummary> {
    if (
      this.state.model &&
      !isInternalBootstrapModel(this.state.model.id) &&
      this.state.backgroundPrewarm.phase === 'error'
    ) {
      const bootPrompt = this.buildBootSpeculativePrompt(this.state.model.id);
      this.patchState({
        errorMessage: null,
        backgroundPrewarm: {
          ...this.state.backgroundPrewarm,
          active: true,
          phase: 'streaming',
          errorMessage: null,
        },
      });
      this.worker.postMessage({
        type: 'resume_prewarm',
        bootPrompt,
      } satisfies WorkerRequest);
      return this.state.model;
    }
    if (this.lastRequestedCatalogModel) {
      return this.loadCatalogModel(this.lastRequestedCatalogModel);
    }
    return this.bootInitialModel();
  }

  clearError(): void {
    if (this.state.errorMessage !== null) {
      this.patchState({
        errorMessage: null,
        status: this.state.model ? 'ready' : 'idle',
      });
    }
  }

  async loadFromUrl(
    id: string,
    displayName: string,
    url: string,
    skipOpfs = false,
    preferDemandStreaming = false,
  ): Promise<LoadedModelSummary> {
    this.cancelDownload();
    const requestId = this.nextRequestId('url');
    this.activeLoadRequestId = requestId;
    const resolvedUrl = this.resolveWorkerFetchUrl(url);
    const bootPrompt = skipOpfs ? undefined : this.buildBootSpeculativePrompt(id);

    if (!isInternalBootstrapModel(id)) {
      this.patchState({
        status: 'downloading',
        progressPercent: 2,
        errorMessage: null,
        backgroundPrewarm: INITIAL_BACKGROUND_PREWARM,
        statusDetail: preferDemandStreaming ? `Connecting ${displayName}` : `Loading ${displayName}`,
      });
    }

    const promise = new Promise<LoadedModelSummary>((resolve, reject) => {
      this.pending.set(requestId, {
        resolve: resolve as (v: unknown) => void,
        reject,
      });
      this.worker.postMessage({
        type: 'load_url',
        requestId,
        id,
        displayName,
        url: resolvedUrl,
        skipOpfs,
        preferDemandStreaming,
        weightResidencyPolicy: this.state.weightResidencyPolicy,
        bootPrompt,
      } satisfies WorkerRequest);
    });
    this.activeLoadPromise = promise;
    void promise.finally(() => {
      if (this.activeLoadPromise === promise) {
        this.activeLoadPromise = null;
      }
    });
    return promise;
  }

  async loadFromFile(file: File): Promise<LoadedModelSummary> {
    if (!file.name.toLowerCase().endsWith('.gguf')) {
      const err = new Error('Please select a valid .gguf model file.');
      this.patchState({ errorMessage: err.message });
      throw err;
    }
    this.cancelDownload();
    const requestId = this.nextRequestId('file');
    this.activeLoadRequestId = requestId;

    this.patchState({
      status: 'downloading',
      progressPercent: 3,
      errorMessage: null,
      statusDetail: `Opening ${file.name} (${formatBytes(file.size)})`,
    });

    return new Promise<LoadedModelSummary>((resolve, reject) => {
      this.pending.set(requestId, {
        resolve: resolve as (v: unknown) => void,
        reject,
      });
      this.worker.postMessage({
        type: 'load_file',
        requestId,
        id: `local-${file.name}`,
        displayName: file.name.replace(/\.gguf$/i, ''),
        source: `local://${file.name}`,
        file,
        weightResidencyPolicy: this.state.weightResidencyPolicy,
      } satisfies WorkerRequest);
    });
  }

  async loadFromArrayBuffer(
    id: string,
    displayName: string,
    source: string,
    buffer: ArrayBuffer,
  ): Promise<LoadedModelSummary> {
    this.patchState({
      status: 'loading',
      progressPercent: 95,
      errorMessage: null,
      statusDetail: `Loading ${displayName}`,
    });

    const requestId = this.nextRequestId('load');
    return new Promise<LoadedModelSummary>((resolve, reject) => {
      this.pending.set(requestId, {
        resolve: resolve as (v: unknown) => void,
        reject,
      });
      const msg: WorkerRequest = {
        type: 'load_buffer',
        requestId,
        id,
        displayName,
        source,
        buffer,
      };
      this.worker.postMessage(msg, [buffer]);
    });
  }

  /**
   * Updates the Observer's focused part slot so the unified worker prioritizes the visible
   * part's queued tasks without aborting or interrupting any sibling part's active inference.
   */
  setFocusedPart(partId: string): void {
    if (!partId || partId === this.focusedPartId) return;
    this.focusedPartId = partId;
    try {
      this.worker.postMessage({ type: 'set_focus_part', partId } satisfies WorkerRequest);
    } catch {
      // Ignore transient worker restart
    }
  }

  async *streamGenerate(
    prompt: string,
    signal?: AbortSignal,
    samplingOverride?: SamplingConfig,
    threadId?: string,
    visualKappa?: string,
    opts?: { readonly partId?: string; readonly salience?: number },
  ): AsyncGenerator<TokenStreamEvent, void, unknown> {
    if (!this.state.model) {
      throw new Error('Wait for the model to finish loading before sending a message.');
    }

    const requestId = this.nextRequestId('gen');
    const partId = opts?.partId ?? threadId ?? 'verdant-part-home';
    const queue: TokenStreamEvent[] = [];
    let queueHead = 0;
    let notifyNext: (() => void) | null = null;
    let done = false;
    let cancelSent = false;
    let streamError: Error | null = null;

    const sendCancelOnce = () => {
      if (cancelSent) return;
      cancelSent = true;
      this.worker.postMessage({ type: 'cancel', requestId } satisfies WorkerRequest);
    };

    const wake = () => {
      if (notifyNext) {
        const fn = notifyNext;
        notifyNext = null;
        fn();
      }
    };

    const onAbort = () => {
      sendCancelOnce();
      done = true;
      wake();
    };

    if (signal?.aborted) {
      return;
    }
    signal?.addEventListener('abort', onAbort, { once: true });

    this.clearSpeculativeTimers();
    this.lastSpeculativePrompt = '';
    this.lastTokenStatePatchMs = 0;

    this.activeGenerationPartByReq.set(requestId, partId);
    const activeParts = Array.from(new Set(this.activeGenerationPartByReq.values()));

    this.patchState({
      status: 'generating',
      prefillProgress: null,
      speculativePrefill: { ...this.state.speculativePrefill, active: false },
      errorMessage: null,
      activePartTaskIds: activeParts,
      statusDetail: 'Preparing response',
    });

    this.pending.set(requestId, {
      resolve: () => {
        done = true;
        wake();
      },
      reject: (err) => {
        streamError = err;
        done = true;
        wake();
      },
      onToken: (ev) => {
        queue.push(ev);
        wake();
      },
    });

    const req: WorkerRequest = {
      type: 'generate',
      requestId,
      prompt,
      sampling: samplingOverride ?? this.state.sampling,
      threadId,
      partId,
      salience: opts?.salience,
      visualKappa,
    };
    this.worker.postMessage(req);

    try {
      while (!done || queueHead < queue.length) {
        if (queueHead < queue.length) {
          const item = queue[queueHead++]!;
          if (queueHead >= queue.length) {
            queue.length = 0;
            queueHead = 0;
          }
          yield item;
        } else if (!done) {
          await new Promise<void>((resolve) => {
            notifyNext = resolve;
          });
        }
      }
      if (streamError) {
        throw streamError;
      }
    } finally {
      signal?.removeEventListener('abort', onAbort);
      this.activeGenerationPartByReq.delete(requestId);
      if (this.pending.has(requestId)) {
        sendCancelOnce();
        this.pending.delete(requestId);
        const remainingParts = Array.from(new Set(this.activeGenerationPartByReq.values()));
        this.patchState({
          status: remainingParts.length > 0 ? 'generating' : 'ready',
          prefillProgress: null,
          activePartTaskIds: remainingParts,
          statusDetail:
            remainingParts.length > 0
              ? 'Generating response'
              : `${this.state.model?.displayName ?? 'Model'} ready`,
        });
      }
    }
  }

  /**
   * Speculatively perceives an attached image inside the Web Worker (zero-copy RGBA transfer)
   * using the shared `MultimodalCoherenceCoordinator` (`kappa` 0 ms cache on repeat images).
   */
  async perceiveImage(
    name: string,
    width: number,
    height: number,
    rgba: Uint8Array,
  ): Promise<PerceivedImageReport> {
    const requestId = this.nextRequestId('img');
    const copy = new Uint8Array(rgba.byteLength);
    copy.set(rgba);
    return this.postRpcWithTimeout<PerceivedImageReport>(
      {
        type: 'perceive_image',
        requestId,
        name,
        width,
        height,
        rgbaBuffer: copy.buffer,
        allowAutoLoad: this.state.modelStreamEnabled,
      },
      15_000,
      [copy.buffer],
    );
  }

  clearPendingImageContext(): void {
    if (this.state.pendingImageContext !== null) {
      this.patchState({ pendingImageContext: null });
    }
  }

  /**
   * Perceives a captured 16 kHz mono voice clip inside the Web Worker (zero-copy transfer),
   * computing Log-Mel acoustic features and caching via content-addressed `kappa`.
   */
  async perceiveAudio(pcm16k: Float32Array, sampleRate = 16_000): Promise<PerceivedAudioReport> {
    const requestId = this.nextRequestId('aud');
    const copy = new Float32Array(pcm16k.length);
    copy.set(pcm16k);
    return this.postRpcWithTimeout<PerceivedAudioReport>(
      {
        type: 'perceive_audio',
        requestId,
        sampleRate,
        pcmBuffer: copy.buffer,
        allowAutoLoad: this.state.modelStreamEnabled,
      },
      15_000,
      [copy.buffer],
    );
  }

  clearPendingAudioContext(): void {
    if (this.state.pendingAudioContext !== null) {
      this.patchState({ pendingAudioContext: null });
    }
  }

  /**
   * Synthesizes a multi-track stereo modal audio clip directly inside the shared Web Worker
   * and returns the zero-copy transferred 16-bit stereo RIFF/WAVE bytes.
   */
  async synthesizeAudioClip(params: {
    readonly title: string;
    readonly style: 'ambient-pad' | 'melodic-arpeggio' | 'rhythm-pulse' | 'sound-effects';
    readonly bpm: number;
    readonly rootMidi: number;
    readonly scale: 'pentatonic' | 'major' | 'minor' | 'dorian';
    readonly durationSec: number;
  }): Promise<{ readonly wavBytes: Uint8Array; readonly sampleRate: number; readonly durationSec: number }> {
    const requestId = this.nextRequestId('syn');
    return this.postRpcWithTimeout(
      {
        type: 'synthesize_audio',
        requestId,
        ...params,
      },
      15_000,
    );
  }

  /**
   * Routes resident SDXS/TAESD neural image weights inside the shared Web Worker
   * when cached locally (or streams lazily when Local AI is On), returning
   * `{ neuralAvailable: false }` in <1ms when Local AI is Off.
   */
  async synthesizeImageLayer(params: {
    readonly prompt: string;
    readonly seed: number;
    readonly latentH?: number;
    readonly latentW?: number;
  }): Promise<{
    readonly neuralAvailable: boolean;
    readonly width: number;
    readonly height: number;
    readonly rgba: Uint8ClampedArray | null;
  }> {
    const requestId = this.nextRequestId('img');
    return this.postRpcWithTimeout(
      {
        type: 'synthesize_image',
        requestId,
        allowAutoLoad: this.state.modelStreamEnabled,
        ...params,
      },
      20_000,
    );
  }

  /**
   * Runs Semantic Vision QA inside the shared Web Worker (non-KV lane, zero-copy RGBA transfer)
   * comparing a rendered Studio frame against a unified semantic concept target (`"dog"`, etc.).
   */
  async evaluateVisualSemanticQa(params: {
    readonly width: number;
    readonly height: number;
    readonly rgba: Uint8Array | Uint8ClampedArray;
    readonly target: SemanticVisualQaTarget;
    readonly minAcceptScoreQ10?: number;
  }): Promise<SemanticVisualQaReport> {
    const requestId = this.nextRequestId('vqa');
    const copy = new Uint8Array(params.rgba.byteLength);
    copy.set(params.rgba);
    return this.postRpcWithTimeout<SemanticVisualQaReport>(
      {
        type: 'qa_visual_semantic',
        requestId,
        width: params.width,
        height: params.height,
        rgbaBuffer: copy.buffer,
        target: params.target,
        allowAutoLoad: this.state.modelStreamEnabled,
        minAcceptScoreQ10: params.minAcceptScoreQ10,
      },
      20_000,
      [copy.buffer],
    );
  }

  /**
   * Consumes any pending speculative visual/audio sensory observations for the active turn
   * so they attach cleanly at the tail (and inject cached SigLIP embeddings via `visualKappa`
   * when dimension-matched) without mutating prior cached KV turns.
   */
  consumePendingSensory(): { readonly textBlock: string | null; readonly visualKappa?: string } {
    const img = this.state.pendingImageContext;
    const aud = this.state.pendingAudioContext;
    if (!img && !aud) return { textBlock: null };
    const lines: string[] = [];
    if (img) lines.push(`[Visual Perception] ${img.summary}`);
    if (aud) {
      lines.push(
        aud.transcript
          ? `[Voice Input] ${aud.transcript}`
          : `[Audio Perception] ${aud.summary}`,
      );
    }
    this.patchState({ pendingImageContext: null, pendingAudioContext: null });
    return { textBlock: lines.join('\n'), visualKappa: img?.kappa };
  }

  consumePendingSensoryBlock(): string | null {
    return this.consumePendingSensory().textBlock;
  }

  dispose(): void {
    this.disposed = true;
    if (typeof window !== 'undefined') {
      window.removeEventListener('pagehide', this.handlePageUnload);
      window.removeEventListener('beforeunload', this.handlePageUnload);
    }
    this.clearSpeculativeTimers();
    this.cancelDownload();
    try {
      this.worker.postMessage({ type: 'shutdown' } satisfies WorkerRequest);
    } catch {
      // Ignore closed worker
    }
    this.worker.terminate();
    this.listeners.clear();
    this.voidListeners.clear();
    this.headerSummaryListeners.clear();
    this.appShellSummaryListeners.clear();
    this.pending.clear();
  }
}
