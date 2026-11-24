/**
 * Main-thread client managing the llama.cpp + vTensor Web Worker,
 * OPFS-backed vRAM persistence for downloaded & imported multi-GB GGUF weights,
 * download cancellation, prefill progress reporting, and reactive state subscriptions.
 */

import type { ThreadMessage } from '@assistant-ui/react';
import { formatSpeculativePrefixForLlama } from './assistantAdapter';
import type { WorkerRequest, WorkerResponse } from './inference.worker';
import {
  DEFAULT_SAMPLING_CONFIG,
  type LoadedModelSummary,
  type PrefillProgressEvent,
  type SamplingConfig,
  type TokenStreamEvent,
} from './llamaRuntime';
import {
  BUNDLED_DEFAULT_MODEL,
  MODEL_CATALOG,
  formatBytes,
  type CatalogModel,
} from './modelCatalog';
import type { PrecisionContractMode } from './performanceContract';
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
  readonly prefillProgress: PrefillProgressEvent | null;
  readonly speculativePrefill: SpeculativePrefillState;
  readonly sampling: SamplingConfig;
  readonly promptMode: PromptFormatMode;
  readonly precisionMode: PrecisionContractMode;
  readonly weightResidencyPolicy: WeightResidencyPolicy;
  readonly cachedModelIds: readonly string[];
  readonly opfsStorage: OpfsQuotaSnapshot;
  readonly conformanceReport: ConformanceSuiteReport | null;
  readonly conformanceRunning: boolean;
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
  webgpu: null,
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
        typeof s.maxTokens === 'number' && s.maxTokens >= 16 && s.maxTokens <= 512
          ? Math.round(s.maxTokens)
          : DEFAULT_SAMPLING_CONFIG.maxTokens,
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
      JSON.stringify({ sampling, promptMode, precisionMode, weightResidencyPolicy }),
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
  const ids = new Set<string>(MODEL_CATALOG.filter((m) => m.bundled).map((m) => m.id));
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

  constructor() {
    const saved = loadSavedPrefs();
    this.state = {
      status: 'idle',
      progressPercent: 0,
      statusDetail: 'Initializing vTensor GGML provider',
      errorMessage: null,
      model: null,
      telemetry: INITIAL_TELEMETRY,
      lastTokensPerSec: 0,
      lastGeneratedTokens: 0,
      prefillProgress: null,
      speculativePrefill: { active: false, readyTokens: 0, targetTokens: 0 },
      sampling: saved.sampling,
      promptMode: saved.promptMode,
      precisionMode: saved.precisionMode,
      weightResidencyPolicy: saved.weightResidencyPolicy,
      cachedModelIds: MODEL_CATALOG.filter((m) => m.bundled).map((m) => m.id),
      opfsStorage: INITIAL_OPFS_STORAGE,
      conformanceReport: null,
      conformanceRunning: false,
    };

    this.worker = new Worker(new URL('./inference.worker.ts', import.meta.url), {
      type: 'module',
    });
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      this.handleWorkerMessage(event.data);
    };
    this.worker.onerror = (err) => {
      const errorMsg = err.message || 'Web Worker runtime error.';
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
   * otherwise falls back to the instant offline bundled GGUF model.
   */
  async bootInitialModel(): Promise<LoadedModelSummary> {
    const storage = await this.refreshOpfsStorage().catch(() => INITIAL_OPFS_STORAGE);
    const cachedIds = computeCachedIds(storage);
    const lastId = readLastModelId();
    if (lastId && cachedIds.includes(lastId)) {
      const remembered = MODEL_CATALOG.find((m) => m.id === lastId);
      if (remembered) {
        return this.loadCatalogModel(remembered);
      }
    }
    return this.loadCatalogModel(BUNDLED_DEFAULT_MODEL);
  }

  updateSampling(patch: Partial<SamplingConfig>): void {
    const nextSampling = { ...this.state.sampling, ...patch };
    savePrefs(
      nextSampling,
      this.state.promptMode,
      this.state.precisionMode,
      this.state.weightResidencyPolicy,
    );
    this.patchState({ sampling: nextSampling });
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
    savePrefs(this.state.sampling, this.state.promptMode, this.state.precisionMode, policy);
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
    // Also clean up any legacy Cache API entries from earlier sessions
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

  private dispatchSpeculativePrefill(prompt: string): void {
    if (this.state.status !== 'ready' || !this.state.model || !prompt) return;
    if (prompt === this.lastSpeculativePrompt) return;
    this.lastSpeculativePrompt = prompt;
    const requestId = this.nextRequestId('spec');
    this.worker.postMessage({
      type: 'speculative_prefill',
      requestId,
      prompt,
    } satisfies WorkerRequest);
  }

  /**
   * Schedules cooperative Continuous Speculative Prefill for the active thread and live composer draft.
   * Debounces cleanly at word boundaries (~160 ms) and idle pause tail (~460 ms) so prefill runs
   * continuously while the user composes without delaying typing or submission.
   */
  scheduleSpeculativePrefill(messages: readonly ThreadMessage[], draftText: string): void {
    this.clearSpeculativeTimers();
    if (this.state.status !== 'ready' || !this.state.model) return;

    const model = this.state.model;
    const mode = this.state.promptMode;
    const stableWordDraft = trimToStableSpeculativePrefix(draftText, false);
    const fullDraft = trimToStableSpeculativePrefix(draftText, true);

    const wordPrompt = formatSpeculativePrefixForLlama(messages, stableWordDraft, model, mode);
    if (wordPrompt && wordPrompt !== this.lastSpeculativePrompt) {
      const delayMs = draftText.trim().length === 0 ? 60 : 160;
      this.speculativeWordTimer = setTimeout(() => {
        this.speculativeWordTimer = null;
        this.dispatchSpeculativePrefill(wordPrompt);
      }, delayMs);
    }

    if (fullDraft !== stableWordDraft && fullDraft.length > 0) {
      const tailPrompt = formatSpeculativePrefixForLlama(messages, fullDraft, model, mode);
      if (tailPrompt && tailPrompt !== this.lastSpeculativePrompt) {
        this.speculativeTailTimer = setTimeout(() => {
          this.speculativeTailTimer = null;
          this.dispatchSpeculativePrefill(tailPrompt);
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
      this.patchState({
        status: 'ready',
        progressPercent: 100,
        statusDetail: `${msg.summary.displayName} bound to vTensor`,
        errorMessage: null,
        prefillProgress: null,
        speculativePrefill: { active: false, readyTokens: 0, targetTokens: 0 },
        model: msg.summary,
        telemetry: msg.telemetry,
        opfsStorage: msg.opfsStorage,
        cachedModelIds: computeCachedIds(msg.opfsStorage),
      });
      // Immediately warm the active template's system prompt into Unified Store / KV cache
      const bootPrefix = formatSpeculativePrefixForLlama([], '', msg.summary, this.state.promptMode);
      if (bootPrefix) {
        this.dispatchSpeculativePrefill(bootPrefix);
      }
      entry.resolve(msg.summary);
      return;
    }

    if (msg.type === 'prefill') {
      const p = msg.progress;
      const prefixNote = p.reusedPrefixTokens > 0 ? ` (reused ${p.reusedPrefixTokens} KV)` : '';
      this.patchState({
        prefillProgress: p,
        statusDetail: `Prefilling prompt ${p.completedTokens}/${p.totalTokens}${prefixNote}`,
      });
      return;
    }

    if (msg.type === 'token') {
      this.patchState({
        prefillProgress: null,
        statusDetail: 'Generating tokens on vTensor',
        lastTokensPerSec: msg.event.tokensPerSecond,
        lastGeneratedTokens: msg.event.generatedTokens,
        ...(msg.telemetry ? { telemetry: msg.telemetry } : {}),
      });
      entry.onToken?.(msg.event);
      return;
    }

    if (msg.type === 'done') {
      this.pending.delete(msg.requestId);
      this.patchState({
        status: 'ready',
        prefillProgress: null,
        statusDetail: `${this.state.model?.displayName ?? 'Model'} ready`,
        telemetry: msg.telemetry,
      });
      entry.resolve(undefined);
      return;
    }

    if (msg.type === 'telemetry') {
      this.pending.delete(msg.requestId);
      this.patchState({ telemetry: msg.telemetry });
      entry.resolve(msg.telemetry);
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
        statusDetail: isCancelled ? 'Download paused (resumable in OPFS)' : 'Inference error',
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

  async loadCatalogModel(model: CatalogModel): Promise<LoadedModelSummary> {
    return this.loadFromUrl(model.id, model.name, model.url, model.bundled);
  }

  /**
   * Streams a GGUF URL directly into OPFS inside the Web Worker (zero main-thread full-file
   * ArrayBuffer copy, enabling 1B–3B+ models to download and run within bounded memory).
   */
  async loadFromUrl(
    id: string,
    displayName: string,
    url: string,
    skipOpfs = false,
  ): Promise<LoadedModelSummary> {
    this.cancelDownload();
    const requestId = this.nextRequestId('url');
    this.activeLoadRequestId = requestId;

    this.patchState({
      status: 'downloading',
      progressPercent: 2,
      errorMessage: null,
      statusDetail: skipOpfs
        ? `Loading bundled ${displayName}`
        : `Opening OPFS vRAM stream for ${displayName}`,
    });

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
        url,
        skipOpfs,
        weightResidencyPolicy: this.state.weightResidencyPolicy,
      } satisfies WorkerRequest);
    });
  }

  /**
   * Streams a local `.gguf` File handle directly into OPFS inside the Web Worker in 4 MiB slices
   * (supporting 2 GB–4 GB+ local models with ~4 MiB peak JS heap).
   */
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
      statusDetail: `Streaming ${file.name} (${formatBytes(file.size)}) into OPFS vRAM`,
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
      statusDetail: `Allocating vTensor buffers for ${displayName}`,
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
  ): AsyncGenerator<TokenStreamEvent, void, unknown> {
    if (!this.state.model) {
      throw new Error('Wait for the GGUF model to finish loading into vTensor buffers.');
    }

    const requestId = this.nextRequestId('gen');
    const queue: TokenStreamEvent[] = [];
    let notifyNext: (() => void) | null = null;
    let done = false;
    let streamError: Error | null = null;

    const wake = () => {
      if (notifyNext) {
        const fn = notifyNext;
        notifyNext = null;
        fn();
      }
    };

    const onAbort = () => {
      this.worker.postMessage({ type: 'cancel', requestId } satisfies WorkerRequest);
      done = true;
      wake();
    };

    if (signal?.aborted) {
      return;
    }
    signal?.addEventListener('abort', onAbort, { once: true });

    this.clearSpeculativeTimers();
    this.lastSpeculativePrompt = '';

    this.patchState({
      status: 'generating',
      prefillProgress: null,
      speculativePrefill: { ...this.state.speculativePrefill, active: false },
      errorMessage: null,
      statusDetail: 'Prefilling prompt on vTensor',
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
      sampling: this.state.sampling,
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
        this.worker.postMessage({ type: 'cancel', requestId } satisfies WorkerRequest);
        this.pending.delete(requestId);
        this.patchState({
          status: 'ready',
          prefillProgress: null,
          statusDetail: 'Generation cancelled',
        });
      }
    }
  }

  dispose(): void {
    this.clearSpeculativeTimers();
    this.cancelDownload();
    this.worker.terminate();
    this.listeners.clear();
    this.pending.clear();
  }
}
