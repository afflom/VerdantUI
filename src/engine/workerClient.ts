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
  pickDefaultNeuralCatalogModel,
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
  type CatalogModel,
} from './modelCatalog';
import type { MultimodalCoherenceTelemetry } from './multimodalCoordinator';
import type { PrecisionContractMode } from './performanceContract';
import type { PerceivedImageReport } from './visionCritiqueLoop';
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

export function loadSavedModelStreamEnabled(): boolean {
  if (typeof window === 'undefined' || typeof window.localStorage === 'undefined') {
    return false;
  }
  try {
    return window.localStorage.getItem(MODEL_STREAM_STORAGE_KEY) === 'true';
  } catch {
    return false;
  }
}

export function saveModelStreamEnabled(enabled: boolean): void {
  if (typeof window === 'undefined' || typeof window.localStorage === 'undefined') return;
  try {
    window.localStorage.setItem(MODEL_STREAM_STORAGE_KEY, enabled ? 'true' : 'false');
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
  readonly sampling: SamplingConfig;
  readonly samplingCustomized: boolean;
  readonly promptMode: PromptFormatMode;
  readonly precisionMode: PrecisionContractMode;
  readonly weightResidencyPolicy: WeightResidencyPolicy;
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

type Listener = (state: EngineState) => void;

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
    const savedStreamEnabled = loadSavedModelStreamEnabled();
    if (!savedStreamEnabled) {
      this.cognitiveObserver.grants.deny('network-weights');
    } else {
      this.cognitiveObserver.grants.reset('network-weights');
      this.cognitiveObserver.grants.grant('network-weights');
    }
    this.state = {
      status: 'idle',
      progressPercent: 0,
      statusDetail: savedStreamEnabled ? 'Starting local AI' : 'Local AI off',
      errorMessage: null,
      model: null,
      telemetry: INITIAL_TELEMETRY,
      lastTokensPerSec: 0,
      lastGeneratedTokens: 0,
      lastPromptTokens: 0,
      lastPromptTruncatedTokens: 0,
      prefillProgress: null,
      speculativePrefill: { active: false, readyTokens: 0, targetTokens: 0 },
      sampling: saved.sampling,
      samplingCustomized: false,
      promptMode: saved.promptMode,
      precisionMode: saved.precisionMode,
      weightResidencyPolicy: saved.weightResidencyPolicy,
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
    };

    this.worker = new Worker(new URL('./inference.worker.ts', import.meta.url), {
      type: 'module',
    });
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      this.handleWorkerMessage(event.data);
    };
    const handleWorkerFailure = (errorMsg: string) => {
      this.activeLoadRequestId = null;
      for (const [, pendingEntry] of this.pending) {
        pendingEntry.reject(new Error(errorMsg));
      }
      this.pending.clear();
      this.patchState({
        status: this.state.model ? 'ready' : 'error',
        progressPercent: 0,
        prefillProgress: null,
        conformanceRunning: false,
        errorMessage: errorMsg,
        statusDetail: 'Worker error',
      });
    };
    this.worker.onerror = (err) => {
      handleWorkerFailure(err.message || 'Web Worker runtime error.');
    };
    this.worker.onmessageerror = () => {
      handleWorkerFailure('Web Worker message serialization error.');
    };

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

  getState(): EngineState {
    return this.state;
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    listener(this.state);
    return () => {
      this.listeners.delete(listener);
    };
  }

  /**
   * Boots the user's previously persisted OPFS catalog model if available,
   * otherwise initializes the internal offline bootstrap checkpoint silently.
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

    const storage = await this.refreshOpfsStorage().catch(() => INITIAL_OPFS_STORAGE);
    const cachedIds = computeCachedIds(storage);
    // Respect persisted Model Stream toggle: disabled upon first entry (never auto-downloads).
    // When enabled across reload, immediately resume/load the remembered or unified default model!
    if (this.state.modelStreamEnabled) {
      const lastId = readLastModelId();
      if (lastId && cachedIds.includes(lastId)) {
        const remembered = MODEL_CATALOG.find((m) => m.id === lastId);
        if (remembered) {
          return this.loadCatalogModel(remembered);
        }
      }
      const target = pickDefaultNeuralCatalogModel(cachedIds);
      return this.loadCatalogModel(target);
    }
    return this.loadCatalogModel(INTERNAL_BOOTSTRAP_MODEL);
  }

  /**
   * Toggles the persistent Model Stream setting (`false` on first visit, remembered across reloads).
   * - Turning ON grants `network-weights` and immediately starts streaming/activating the unified model.
   * - Turning OFF cancels any in-flight stream and denies `network-weights` (`FK-07`).
   */
  async setModelStreamEnabled(enabled: boolean): Promise<void> {
    saveModelStreamEnabled(enabled);
    if (enabled) {
      this.cognitiveObserver.grants.reset('network-weights');
      this.cognitiveObserver.grants.grant('network-weights');
      this.patchState({ modelStreamEnabled: true, errorMessage: null });

      const activeId = this.state.model?.id;
      const alreadyLoadedRealModel =
        activeId !== undefined && activeId !== INTERNAL_BOOTSTRAP_MODEL.id;
      if (!alreadyLoadedRealModel && this.state.status !== 'downloading' && this.state.status !== 'loading') {
        const storage = await this.refreshOpfsStorage().catch(() => this.state.opfsStorage);
        const cachedIds = computeCachedIds(storage);
        const lastId = readLastModelId();
        const remembered =
          lastId && cachedIds.includes(lastId)
            ? MODEL_CATALOG.find((m) => m.id === lastId)
            : undefined;
        const target = remembered ?? pickDefaultNeuralCatalogModel(cachedIds);
        await this.loadCatalogModel(target);
      }
    } else {
      this.cognitiveObserver.grants.deny('network-weights');
      this.cancelDownload();
      this.patchState({
        modelStreamEnabled: false,
        status: this.state.model ? 'ready' : 'idle',
        progressPercent: 0,
        statusDetail: 'Local AI off',
        errorMessage: null,
      });
    }
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
    const requestId = this.nextRequestId('prec');
    return new Promise<VTensorTelemetrySnapshot>((resolve, reject) => {
      this.pending.set(requestId, {
        resolve: resolve as (v: unknown) => void,
        reject,
      });
      this.worker.postMessage({
        type: 'set_precision_mode',
        requestId,
        mode: precisionMode,
      } satisfies WorkerRequest);
    });
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
    return new Promise<VTensorTelemetrySnapshot>((resolve, reject) => {
      this.pending.set(requestId, {
        resolve: resolve as (v: unknown) => void,
        reject,
      });
      this.worker.postMessage({
        type: 'set_weight_residency_policy',
        requestId,
        policy,
      } satisfies WorkerRequest);
    });
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

  private dispatchSpeculativePrefill(prompt: string, threadId?: string): void {
    if (this.state.status !== 'ready' || !this.state.model || !prompt) return;
    if (this.state.model.backingKind === 'opfs-demand') return;
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
    if (this.state.status !== 'ready' || !this.state.model) return;
    if (this.state.model.backingKind === 'opfs-demand') return;

    const model = this.state.model;
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
        model: msg.summary,
        telemetry: msg.telemetry,
        opfsStorage: msg.opfsStorage,
        cachedModelIds: computeCachedIds(msg.opfsStorage),
      });
      // Warm initial prompt prefix for real resident models (skip bootstrap & demand-paged models)
      if (!isInternalBootstrap && msg.summary.backingKind !== 'opfs-demand') {
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
      this.patchState({
        status: 'ready',
        prefillProgress: null,
        statusDetail: `${this.state.model?.displayName ?? 'Model'} ready`,
        telemetry: msg.telemetry,
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
      if (this.activeLoadRequestId === msg.requestId) {
        this.activeLoadRequestId = null;
      }
      const isCancelled = msg.message === 'Download cancelled.';
      const nextModel = msg.modelUnloaded ? null : this.state.model;
      this.patchState({
        model: nextModel,
        status: nextModel ? 'ready' : isCancelled ? 'idle' : 'error',
        progressPercent: 0,
        prefillProgress: null,
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
    if (model.id !== INTERNAL_BOOTSTRAP_MODEL.id && !this.state.modelStreamEnabled) {
      saveModelStreamEnabled(true);
      this.cognitiveObserver.grants.reset('network-weights');
      this.cognitiveObserver.grants.grant('network-weights');
      this.patchState({ modelStreamEnabled: true });
    }
    return this.loadFromUrl(model.id, model.name, model.url, model.bundled);
  }

  /** Retries the most recently requested catalog model (or re-runs initial boot). */
  async retryLastLoad(): Promise<LoadedModelSummary> {
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
  ): Promise<LoadedModelSummary> {
    this.cancelDownload();
    const requestId = this.nextRequestId('url');
    this.activeLoadRequestId = requestId;
    const resolvedUrl = this.resolveWorkerFetchUrl(url);

    if (!isInternalBootstrapModel(id)) {
      this.patchState({
        status: 'downloading',
        progressPercent: 2,
        errorMessage: null,
        statusDetail: `Loading ${displayName}`,
      });
    }

    return new Promise<LoadedModelSummary>((resolve, reject) => {
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
        weightResidencyPolicy: this.state.weightResidencyPolicy,
      } satisfies WorkerRequest);
    });
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

  async *streamGenerate(
    prompt: string,
    signal?: AbortSignal,
    samplingOverride?: SamplingConfig,
    threadId?: string,
    visualKappa?: string,
  ): AsyncGenerator<TokenStreamEvent, void, unknown> {
    if (!this.state.model) {
      throw new Error('Wait for the model to finish loading before sending a message.');
    }

    const requestId = this.nextRequestId('gen');
    const queue: TokenStreamEvent[] = [];
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

    this.patchState({
      status: 'generating',
      prefillProgress: null,
      speculativePrefill: { ...this.state.speculativePrefill, active: false },
      errorMessage: null,
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
      visualKappa,
    };
    this.worker.postMessage(req);

    try {
      while (!done || queue.length > 0) {
        if (queue.length > 0) {
          yield queue.shift()!;
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
      if (this.pending.has(requestId)) {
        sendCancelOnce();
        this.pending.delete(requestId);
        this.patchState({
          status: 'ready',
          prefillProgress: null,
          statusDetail: `${this.state.model?.displayName ?? 'Model'} ready`,
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
    return new Promise<PerceivedImageReport>((resolve, reject) => {
      this.pending.set(requestId, {
        resolve: resolve as (v: unknown) => void,
        reject,
      });
      const msg: WorkerRequest = {
        type: 'perceive_image',
        requestId,
        name,
        width,
        height,
        rgbaBuffer: copy.buffer,
        allowAutoLoad: this.state.modelStreamEnabled,
      };
      this.worker.postMessage(msg, [copy.buffer]);
    });
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
    return new Promise<PerceivedAudioReport>((resolve, reject) => {
      this.pending.set(requestId, {
        resolve: resolve as (v: unknown) => void,
        reject,
      });
      const msg: WorkerRequest = {
        type: 'perceive_audio',
        requestId,
        sampleRate,
        pcmBuffer: copy.buffer,
        allowAutoLoad: this.state.modelStreamEnabled,
      };
      this.worker.postMessage(msg, [copy.buffer]);
    });
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
    return new Promise((resolve, reject) => {
      this.pending.set(requestId, {
        resolve: resolve as (v: unknown) => void,
        reject,
      });
      this.worker.postMessage({
        type: 'synthesize_audio',
        requestId,
        ...params,
      } satisfies WorkerRequest);
    });
  }

  /**
   * Routes resident SDXS/TAESD neural image weights inside the shared Web Worker
   * when cached locally, returning `{ neuralAvailable: false }` in <1ms otherwise.
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
    return new Promise((resolve, reject) => {
      this.pending.set(requestId, {
        resolve: resolve as (v: unknown) => void,
        reject,
      });
      this.worker.postMessage({
        type: 'synthesize_image',
        requestId,
        ...params,
      } satisfies WorkerRequest);
    });
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
    this.pending.clear();
  }
}
