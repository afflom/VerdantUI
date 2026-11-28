/// <reference lib="webworker" />

import { computeLogMelSpectrogram } from './audioFrontend';
import {
  composeMultiTrackArrangement,
  encodeStereoWav,
  masterStereoPcm,
  synthesizeCompositionStereoPcm,
} from './audioSynthesisEngine';
import {
  encodePromptToClipTokens,
  ensureClipBpeVocabularyCached,
  loadImageWeightsFromBacking,
  synthesizeDistilledImageAsync,
  type ImageSynthesisScratch,
  type LoadedNeuralImageWeights,
} from './imageSynthesisEngine';
import {
  LlamaVTensorRuntime,
  type LoadedModelSummary,
  type PrefillProgressEvent,
  type SamplingConfig,
  type TokenStreamEvent,
} from './llamaRuntime';
import { MULTIMODAL_CORE_MODELS, formatBytes } from './modelCatalog';
import {
  MultimodalCoherenceCoordinator,
  computeMediaKappa,
  type MultimodalCoherenceTelemetry,
} from './multimodalCoordinator';
import type { PrecisionContractMode } from './performanceContract';
import {
  evaluateSemanticVisualQa,
  perceiveImageFrame,
  type PerceivedImageReport,
  type SemanticVisualQaReport,
  type SemanticVisualQaTarget,
} from './visionCritiqueLoop';
import { SiglipVisionEncoder, loadVisionWeightsFromGguf } from './visionEncoder';
import { WhisperAudioEngine, loadWhisperWeightsFromGgml } from './whisperEngine';
import {
  OPFS_HANDOVER_CHANNEL_NAME,
  clearAllOpfsModels,
  deleteOpfsModel,
  inspectOpfsVramStorage,
  streamFileToVramBacking,
  streamUrlToVramBacking,
  type OpfsQuotaSnapshot,
  type VramBackingKind,
  type VramBackingStore,
} from './vramBackingStore';
import {
  createCooperativeYielder,
  shouldCheckpointAtKvPageBoundary,
} from './vramSwapPolicy';
import { getSharedOpfsCasStore } from './vramCasStore';
import type { WeightResidencyPolicy } from './vramWeightPager';
import {
  type ConformanceSuiteReport,
  runVTensorConformanceSuite,
  type VTensorTelemetrySnapshot,
} from './vtensorProvider';

export type WorkerRequest =
  | {
      type: 'load_buffer';
      requestId: string;
      id: string;
      displayName: string;
      source: string;
      buffer: ArrayBuffer;
      maxContextCap?: number;
    }
  | {
      type: 'load_url';
      requestId: string;
      id: string;
      displayName: string;
      url: string;
      skipOpfs?: boolean;
      preferDemandStreaming?: boolean;
      maxContextCap?: number;
      weightResidencyPolicy?: WeightResidencyPolicy;
      bootPrompt?: string;
    }
  | {
      type: 'load_file';
      requestId: string;
      id: string;
      displayName: string;
      source: string;
      file: File;
      maxContextCap?: number;
      weightResidencyPolicy?: WeightResidencyPolicy;
    }
  | {
      type: 'generate';
      requestId: string;
      prompt: string;
      sampling: SamplingConfig;
      threadId?: string;
      partId?: string;
      salience?: number;
      visualKappa?: string;
    }
  | {
      type: 'set_focus_part';
      partId: string;
    }
  | {
      type: 'cancel';
      requestId: string;
    }
  | {
      type: 'get_telemetry';
      requestId: string;
    }
  | {
      type: 'run_conformance';
      requestId: string;
    }
  | {
      type: 'set_precision_mode';
      requestId: string;
      mode: PrecisionContractMode;
    }
  | {
      type: 'set_weight_residency_policy';
      requestId: string;
      policy: WeightResidencyPolicy;
    }
  | {
      type: 'inspect_opfs';
      requestId: string;
    }
  | {
      type: 'delete_opfs_model';
      requestId: string;
      modelId: string;
    }
  | {
      type: 'clear_opfs_models';
      requestId: string;
    }
  | {
      type: 'speculative_prefill';
      requestId: string;
      prompt: string;
      threadId?: string;
    }
  | {
      type: 'count_tokens';
      requestId: string;
      prompt: string;
    }
  | {
      type: 'perceive_image';
      requestId: string;
      name: string;
      width: number;
      height: number;
      rgbaBuffer: ArrayBuffer;
      allowAutoLoad?: boolean;
    }
  | {
      type: 'perceive_audio';
      requestId: string;
      sampleRate: number;
      pcmBuffer: ArrayBuffer;
      allowAutoLoad?: boolean;
    }
  | {
      type: 'synthesize_audio';
      requestId: string;
      title: string;
      style: 'ambient-pad' | 'melodic-arpeggio' | 'rhythm-pulse' | 'sound-effects';
      bpm: number;
      rootMidi: number;
      scale: 'pentatonic' | 'major' | 'minor' | 'dorian';
      durationSec: number;
    }
  | {
      type: 'synthesize_image';
      requestId: string;
      prompt: string;
      seed: number;
      latentH?: number;
      latentW?: number;
      allowAutoLoad?: boolean;
    }
  | {
      type: 'qa_visual_semantic';
      requestId: string;
      width: number;
      height: number;
      rgbaBuffer: ArrayBuffer;
      target: SemanticVisualQaTarget;
      allowAutoLoad?: boolean;
      minAcceptScoreQ10?: number;
    }
  | {
      type: 'pause_prewarm';
    }
  | {
      type: 'resume_prewarm';
      bootPrompt?: string;
    }
  | {
      type: 'shutdown';
    };

export interface PerceivedAudioReport {
  readonly kappa: string;
  readonly durationSec: number;
  readonly frames: number;
  readonly rmsEnergy: number;
  readonly speechDetected: boolean;
  readonly transcript: string;
  readonly cacheHit: boolean;
  readonly summary: string;
}

export type WorkerResponse =
  | {
      type: 'progress';
      requestId: string;
      percent: number;
      detail: string;
      receivedBytes: number;
      totalBytes: number;
      backingKind: VramBackingKind;
    }
  | {
      type: 'loaded';
      requestId: string;
      summary: LoadedModelSummary;
      telemetry: VTensorTelemetrySnapshot;
      opfsStorage: OpfsQuotaSnapshot;
    }
  | {
      type: 'prefill';
      requestId: string;
      progress: PrefillProgressEvent;
    }
  | {
      type: 'token';
      requestId: string;
      event: TokenStreamEvent;
      telemetry?: VTensorTelemetrySnapshot;
    }
  | {
      type: 'done';
      requestId: string;
      telemetry: VTensorTelemetrySnapshot;
      multimodalTelemetry?: MultimodalCoherenceTelemetry;
    }
  | {
      type: 'telemetry';
      requestId: string;
      telemetry: VTensorTelemetrySnapshot;
      multimodalTelemetry?: MultimodalCoherenceTelemetry;
    }
  | {
      type: 'image_perceived';
      requestId: string;
      report: PerceivedImageReport;
      multimodalTelemetry: MultimodalCoherenceTelemetry;
    }
  | {
      type: 'audio_perceived';
      requestId: string;
      report: PerceivedAudioReport;
      multimodalTelemetry: MultimodalCoherenceTelemetry;
    }
  | {
      type: 'audio_synthesized';
      requestId: string;
      wavBuffer: ArrayBuffer;
      sampleRate: number;
      durationSec: number;
      multimodalTelemetry: MultimodalCoherenceTelemetry;
    }
  | {
      type: 'image_synthesized';
      requestId: string;
      neuralAvailable: boolean;
      width: number;
      height: number;
      rgbaBuffer: ArrayBuffer | null;
      multimodalTelemetry: MultimodalCoherenceTelemetry;
    }
  | {
      type: 'visual_qa_result';
      requestId: string;
      report: SemanticVisualQaReport;
      multimodalTelemetry: MultimodalCoherenceTelemetry;
    }
  | {
      type: 'opfs_storage';
      requestId: string;
      storage: OpfsQuotaSnapshot;
      modelUnloaded?: boolean;
    }
  | {
      type: 'conformance';
      requestId: string;
      report: ConformanceSuiteReport;
    }
  | {
      type: 'speculative_status';
      requestId: string;
      active: boolean;
      readyTokens: number;
      targetTokens: number;
      telemetry?: VTensorTelemetrySnapshot;
    }
  | {
      type: 'prewarm_status';
      active: boolean;
      phase: 'streaming' | 'warming-context' | 'ready' | 'error';
      modelId: string;
      warmBytes: number;
      targetBytes: number;
      percent: number;
      casBytesSaved: number;
      bytesPerSecond?: number;
      etaSeconds?: number | null;
      contextWarmed: boolean;
      errorMessage?: string;
    }
  | {
      type: 'token_count';
      requestId: string;
      tokenCount: number;
      maxContextWindow: number;
    }
  | {
      type: 'error';
      requestId: string;
      message: string;
      opfsStorage?: OpfsQuotaSnapshot;
      modelUnloaded?: boolean;
    };

const runtime = new LlamaVTensorRuntime();
const multimodalCoordinator = new MultimodalCoherenceCoordinator();
let activeVisionEncoder: SiglipVisionEncoder | null = null;
let activeWhisperEngine: WhisperAudioEngine | null = null;
let residentNeuralImageWeights: LoadedNeuralImageWeights | null = null;
let imageLoadPromise: Promise<LoadedNeuralImageWeights | null> | null = null;
const imageSynthesisScratch: ImageSynthesisScratch = {};
function getResidentNeuralImageWeights(): LoadedNeuralImageWeights | null {
  return residentNeuralImageWeights;
}
function disposeResidentNeuralImageWeights(): void {
  if (residentNeuralImageWeights) {
    residentNeuralImageWeights.dispose?.();
    residentNeuralImageWeights = null;
  }
}
async function ensureImageWeightsOnDemand(): Promise<LoadedNeuralImageWeights | null> {
  if (residentNeuralImageWeights) return residentNeuralImageWeights;
  if (imageLoadPromise) return imageLoadPromise;
  const spec = MULTIMODAL_CORE_MODELS.find((m) => m.modality === 'image-synthesis');
  if (!spec) return null;
  imageLoadPromise = (async () => {
    try {
      const backing = await streamUrlToVramBacking(spec.id, spec.url, {
        skipOpfs: false,
        casStore: getSharedOpfsCasStore(),
      });
      try {
        const loaded = loadImageWeightsFromBacking(backing);
        residentNeuralImageWeights = loaded;
        multimodalCoordinator.activateModality({
          modality: 'image-synthesis',
          modelId: spec.id,
          residentBytes: loaded.residentBytes,
          evict: () => {
            disposeResidentNeuralImageWeights();
          },
        });
        return loaded;
      } catch (err) {
        backing.close();
        throw err;
      }
    } catch {
      return null;
    } finally {
      imageLoadPromise = null;
    }
  })();
  return imageLoadPromise;
}
let visionLoadPromise: Promise<SiglipVisionEncoder | null> | null = null;
let whisperLoadPromise: Promise<WhisperAudioEngine | null> | null = null;
const threadPoolReadyPromise = runtime.initThreadPool().catch(() => undefined);

function readBackingBytes(backing: VramBackingStore): Uint8Array {
  const out = new Uint8Array(backing.byteLength);
  let pos = 0;
  const chunk = 8 * 1024 * 1024;
  while (pos < backing.byteLength) {
    const len = Math.min(chunk, backing.byteLength - pos);
    const got = backing.readRange(pos, out, pos, len);
    if (got <= 0) break;
    pos += got;
  }
  return out;
}

async function ensureVisionEncoderOnDemand(): Promise<SiglipVisionEncoder | null> {
  if (activeVisionEncoder) return activeVisionEncoder;
  if (visionLoadPromise) return visionLoadPromise;
  const spec = MULTIMODAL_CORE_MODELS.find((m) => m.modality === 'vision');
  if (!spec) return null;
  visionLoadPromise = (async () => {
    try {
      const backing = await streamUrlToVramBacking(spec.id, spec.url, {
        skipOpfs: false,
        casStore: getSharedOpfsCasStore(),
      });
      try {
        const bytes = readBackingBytes(backing);
        const weights = loadVisionWeightsFromGguf(bytes);
        activeVisionEncoder = new SiglipVisionEncoder(weights);
        multimodalCoordinator.activateModality({
          modality: 'vision',
          modelId: spec.id,
          residentBytes: bytes.byteLength,
          evict: () => {
            activeVisionEncoder = null;
          },
        });
        return activeVisionEncoder;
      } finally {
        backing.close();
      }
    } catch {
      return null;
    } finally {
      visionLoadPromise = null;
    }
  })();
  return visionLoadPromise;
}

async function ensureWhisperEngineOnDemand(): Promise<WhisperAudioEngine | null> {
  if (activeWhisperEngine) return activeWhisperEngine;
  if (whisperLoadPromise) return whisperLoadPromise;
  const spec = MULTIMODAL_CORE_MODELS.find((m) => m.modality === 'audio');
  if (!spec) return null;
  whisperLoadPromise = (async () => {
    try {
      const backing = await streamUrlToVramBacking(spec.id, spec.url, {
        skipOpfs: false,
        casStore: getSharedOpfsCasStore(),
      });
      try {
        const bytes = readBackingBytes(backing);
        const weights = loadWhisperWeightsFromGgml(bytes);
        activeWhisperEngine = new WhisperAudioEngine(weights);
        multimodalCoordinator.activateModality({
          modality: 'audio',
          modelId: spec.id,
          residentBytes: bytes.byteLength,
          evict: () => {
            activeWhisperEngine = null;
          },
        });
        return activeWhisperEngine;
      } finally {
        backing.close();
      }
    } catch {
      return null;
    } finally {
      whisperLoadPromise = null;
    }
  })();
  return whisperLoadPromise;
}

const activeControllers = new Map<string, AbortController>();
let activeSpeculativeController: AbortController | null = null;
let activeSpeculativePromise: Promise<void> | null = null;
let activePrewarmController: AbortController | null = null;
let activePrewarmPromise: Promise<void> | null = null;
let prewarmAllowed = true;
let activeBootPrompt: string | undefined;
let contextWarmedModelId: string | null = null;
let activeLoadController: AbortController | null = null;
let activeLoadPromise: Promise<void> | null = null;
let activeGenerationController: AbortController | null = null;
let activeGenerationPromise: Promise<void> | null = null;
let activeGenerationCount = 0;
let activeGenerationPartId: string | null = null;
let checkpointRequested = false;
let focusedPartId = 'verdant-part-home';
const cooperativeYielder = createCooperativeYielder();

interface QueuedGenerationJob {
  readonly req: Extract<WorkerRequest, { type: 'generate' }>;
  readonly controller: AbortController;
  readonly enqueuedSeq: number;
  readonly continuationPrompt?: string;
  readonly continuationAccumulatedText?: string;
  readonly continuationTokensEmitted?: number;
  readonly remainingMaxTokens?: number;
}
let generationEnqueueSeq = 0;
const pendingGenerationQueue: QueuedGenerationJob[] = [];
let generationPumpActive = false;

let activeOpfsPersistPromise: Promise<void> | null = null;
let opfsPersistEpoch = 0;

/**
 * Cancels any queued background `.unified.kv` OPFS persist and waits for an in-flight write to close
 * its SyncAccessHandle cleanly before model loads, deletions, handover, or worker shutdown.
 */
async function drainBackgroundOpfsPersist(): Promise<void> {
  opfsPersistEpoch += 1;
  while (activeOpfsPersistPromise) {
    const pending = activeOpfsPersistPromise;
    try {
      await pending;
    } catch {
      // Ignore best-effort persistence failures
    }
  }
}

/**
 * Schedules serialized, coalesced background OPFS KV persistence AFTER posting `{ type: 'done' }`
 * so interactive turn completion has 0 ms storage/flush latency while warm prefixes still survive reloads.
 */
function scheduleBackgroundOpfsPersist(): void {
  const epoch = ++opfsPersistEpoch;
  const prev = activeOpfsPersistPromise ?? Promise.resolve();
  const task = prev
    .catch(() => undefined)
    .then(async () => {
      if (
        epoch !== opfsPersistEpoch ||
        activeGenerationCount > 0 ||
        activeLoadController !== null ||
        !runtime.isLoaded
      ) {
        return;
      }
      await runtime.persistUnifiedStoreToOpfs().catch(() => 0);
    });
  const tracked = task.finally(() => {
    if (activeOpfsPersistPromise === tracked) {
      activeOpfsPersistPromise = null;
    }
  });
  activeOpfsPersistPromise = tracked;
}

const workerInstanceId =
  typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `w-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

// Cross-tab / reload OPFS lease handover channel: if a newer reloading tab/worker requests
// an OPFS file held by an idle older worker, immediately close our SyncAccessHandle!
let opfsHandoverChannel: BroadcastChannel | null = null;
if (typeof BroadcastChannel === 'function') {
  try {
    opfsHandoverChannel = new BroadcastChannel(OPFS_HANDOVER_CHANNEL_NAME);
    opfsHandoverChannel.onmessage = (ev: MessageEvent<{ type?: string; senderId?: string; modelId?: string | null }>) => {
      const msg = ev.data;
      if (!msg || msg.type !== 'release-opfs' || msg.senderId === workerInstanceId) {
        return;
      }
      if (activeGenerationCount > 0 || activeLoadController !== null) {
        return;
      }
      const loaded = runtime.getModelSummary();
      const holdsOpfs =
        loaded !== null &&
        (loaded.backingKind === 'opfs-sync' || loaded.backingKind === 'opfs-demand');
      if (holdsOpfs && (!msg.modelId || msg.modelId === loaded.id)) {
        void preemptBackgroundPrewarm()
          .then(() => drainBackgroundOpfsPersist())
          .finally(() => {
            if (activeGenerationCount > 0 || activeLoadController !== null) return;
            runtime.unload();
            try {
              opfsHandoverChannel?.postMessage({
                type: 'released',
                senderId: workerInstanceId,
                modelId: loaded.id,
              });
            } catch {
              // Ignore closed channel
            }
          });
      }
    };
  } catch {
    opfsHandoverChannel = null;
  }
}

/**
 * Serializes model loads so rapid model switches never overlap OPFS handles or race runtime state.
 * Loops until no in-flight load owns the slot so concurrent back-to-back callers cannot race.
 */
async function preemptActiveLoad(): Promise<void> {
  while (activeLoadController || activeLoadPromise) {
    const ctrl = activeLoadController;
    const pending = activeLoadPromise;
    if (ctrl) ctrl.abort();
    if (pending) {
      try {
        await pending;
      } catch {
        // Previous load cleaned up
      }
    } else {
      break;
    }
  }
}

const workerScope = self as unknown as DedicatedWorkerGlobalScope;

function post(msg: WorkerResponse): void {
  workerScope.postMessage(msg);
}

/**
 * Strictly preempts any running cooperative speculative prefill before high-priority
 * generation, model load, or residency mutations. Already-prefilled tokens remain preserved
 * in the live vRAM KV pager and Unified Virtual Store!
 */
async function preemptSpeculativePrefill(): Promise<void> {
  while (activeSpeculativeController || activeSpeculativePromise) {
    const ctrl = activeSpeculativeController;
    const pending = activeSpeculativePromise;
    if (ctrl) ctrl.abort();
    if (pending) {
      try {
        await pending;
      } catch {
        // Ignore speculative abort/cancellation errors
      }
    } else {
      break;
    }
  }
}

/**
 * Immediately preempts interruptible background weight prewarming (<1 ms) when a user turn,
 * model switch, offline toggle, or storage mutation arrives. Already-warmed OPFS CAS slabs
 * remain persisted and immediately reusable.
 */
async function preemptBackgroundPrewarm(): Promise<void> {
  while (activePrewarmController || activePrewarmPromise) {
    const ctrl = activePrewarmController;
    const pending = activePrewarmPromise;
    if (ctrl) ctrl.abort();
    if (pending) {
      try {
        await pending;
      } catch {
        // Ignore prewarm abort/cancellation errors
      }
    } else {
      break;
    }
  }
}

/**
 * High-throughput, interruptible initialization & background prewarmer:
 *   1. Streams priority uncached weight slices (`32 MiB` slices via direct-CDN 16 MiB coalesced
 *      HTTP Range fetches, strictly skipping cold Sparse MoE expert tables) with live speed/ETA.
 *   2. Once OPFS slabs are 100% warm, pins active WASM layers + lazy full-vocab LM-head and
 *      pre-computes the initial system-prompt KV prefix (`warmRuntimeForInstantFirstToken`) so
 *      the user's VERY FIRST prompt achieves instant TTFT and peak steady-state tok/s!
 *   3. Preempts in < 1 ms on `generate`, `pause_prewarm`, or model switch, keeping all warmed slabs.
 */
function startBackgroundPrewarm(): void {
  if (
    !prewarmAllowed ||
    !runtime.isLoaded ||
    activeGenerationCount > 0 ||
    pendingGenerationQueue.length > 0 ||
    activeLoadController !== null ||
    activePrewarmController !== null
  ) {
    return;
  }
  const summary = runtime.getModelSummary();
  if (!summary) return;
  const modelId = summary.id;

  // For already-resident/full-OPFS models, warm pinned WASM frames + system-prompt KV directly
  const isDemand = summary.backingKind === 'opfs-demand';
  const plan = isDemand ? runtime.getDemandPrewarmPlan() : null;
  const targetBytes = plan?.totalTargetBytes ?? summary.fileByteLength;
  let warmBytes = plan?.alreadyCachedBytes ?? targetBytes;
  const slices = plan?.slices ?? [];

  if (slices.length === 0 && contextWarmedModelId === modelId) {
    post({
      type: 'prewarm_status',
      active: false,
      phase: 'ready',
      modelId,
      warmBytes: targetBytes,
      targetBytes,
      percent: 100,
      casBytesSaved: getSharedOpfsCasStore().getTelemetry().bytesSaved,
      contextWarmed: true,
    });
    return;
  }

  const controller = new AbortController();
  activePrewarmController = controller;

  let task: Promise<void> | null = null;
  task = (async () => {
    let lastPost = 0;
    const tStartMs = performance.now();
    let downloadedThisRun = 0;
    let finishedWarm = false;
    let errorMessage: string | undefined;

    const emit = (
      active: boolean,
      phase: 'streaming' | 'warming-context' | 'ready' | 'error',
      contextWarmed: boolean,
      force = false,
      errMsg?: string,
    ) => {
      const now = performance.now();
      const pct =
        targetBytes > 0 ? Math.min(100, Math.round((warmBytes / targetBytes) * 100)) : 100;
      const elapsedSec = Math.max(0.1, (now - tStartMs) / 1000);
      const bps = downloadedThisRun > 0 ? Math.round(downloadedThisRun / elapsedSec) : 0;
      const remaining = Math.max(0, targetBytes - warmBytes);
      const eta = bps >= 64 * 1024 && remaining > 0 ? remaining / bps : null;
      if (force || now - lastPost >= 200 || pct === 100) {
        lastPost = now;
        post({
          type: 'prewarm_status',
          active,
          phase,
          modelId,
          warmBytes,
          targetBytes,
          percent: pct,
          casBytesSaved: getSharedOpfsCasStore().getTelemetry().bytesSaved,
          bytesPerSecond: bps > 0 ? bps : undefined,
          etaSeconds: eta,
          contextWarmed,
          ...(errMsg ? { errorMessage: errMsg } : {}),
        });
      }
    };

    try {
      if (slices.length > 0) {
        emit(true, 'streaming', false, true);
        for (const slice of slices) {
          if (
            controller.signal.aborted ||
            !prewarmAllowed ||
            activeGenerationCount > 0 ||
            pendingGenerationQueue.length > 0 ||
            activeLoadController !== null ||
            runtime.getModelSummary()?.id !== modelId
          ) {
            return;
          }
          await runtime.prefetchDemandRanges(slice.ranges, controller.signal);
          warmBytes = Math.min(targetBytes, warmBytes + slice.byteLength);
          downloadedThisRun += slice.byteLength;
          emit(true, 'streaming', false, false);
          await cooperativeYielder.yieldNow();
        }
      }

      if (
        controller.signal.aborted ||
        !prewarmAllowed ||
        activeGenerationCount > 0 ||
        pendingGenerationQueue.length > 0 ||
        activeLoadController !== null ||
        runtime.getModelSummary()?.id !== modelId
      ) {
        return;
      }

      // Stage 3: Pin WASM layers + full-vocab LM-head + pre-compute system-prompt KV prefix
      warmBytes = targetBytes;
      emit(true, 'warming-context', false, true);
      await runtime.warmRuntimeForInstantFirstToken(activeBootPrompt, controller.signal);
      if (!controller.signal.aborted && runtime.getModelSummary()?.id === modelId) {
        contextWarmedModelId = modelId;
        finishedWarm = true;
      }
    } catch (err) {
      const isAbort =
        controller.signal.aborted ||
        (err instanceof DOMException && err.name === 'AbortError') ||
        (err instanceof Error && err.name === 'AbortError');
      if (!isAbort) {
        errorMessage = err instanceof Error ? err.message : String(err);
      }
    } finally {
      runtime.flushDemandBacking();
      if (activePrewarmController === controller) {
        activePrewarmController = null;
      }
      if (activePrewarmPromise === task) {
        activePrewarmPromise = null;
      }
      if (finishedWarm) {
        emit(false, 'ready', true, true);
      } else if (errorMessage) {
        emit(false, 'error', false, true, errorMessage);
      } else {
        emit(false, warmBytes >= targetBytes ? 'warming-context' : 'streaming', contextWarmedModelId === modelId, true);
      }
    }
  })();
  activePrewarmPromise = task;
}

/**
 * Strictly serializes inference generations so switching threads or triggering a new prompt
 * NEVER interleaves two generators over the shared `kvPager` / `cachedTokens` / `scratch` state!
 */
async function preemptActiveGeneration(): Promise<void> {
  // Drain queued jobs only on destructive administrative actions (model switch/delete/shutdown)
  while (pendingGenerationQueue.length > 0) {
    const queued = pendingGenerationQueue.shift()!;
    queued.controller.abort();
    activeControllers.delete(queued.req.requestId);
    post({
      type: 'done',
      requestId: queued.req.requestId,
      telemetry: runtime.getTelemetry(),
      multimodalTelemetry: multimodalCoordinator.getTelemetry(),
    });
  }
  while (activeGenerationController || activeGenerationPromise) {
    const ctrl = activeGenerationController;
    const pending = activeGenerationPromise;
    if (ctrl) ctrl.abort();
    if (pending) {
      try {
        await pending;
      } catch {
        // Previous generation cleaned up
      }
    } else {
      break;
    }
  }
}

/**
 * Picks the highest-salience queued generation job (focused `partId` first, then explicit
 * `kernel10` salience, then FIFO) so independent parts NEVER preempt or corrupt each other's
 * live `kvPager` state while foreground focus still responds first.
 */
function dequeueNextGenerationJob(): QueuedGenerationJob | undefined {
  if (pendingGenerationQueue.length === 0) return undefined;
  let bestIdx = 0;
  let bestScore = -1;
  for (let i = 0; i < pendingGenerationQueue.length; i++) {
    const job = pendingGenerationQueue[i]!;
    const isFocused = (job.req.partId ?? job.req.threadId ?? '') === focusedPartId ? 10_000 : 0;
    const salience = job.req.salience ?? 512;
    const score = isFocused + salience;
    if (score > bestScore) {
      bestScore = score;
      bestIdx = i;
    }
  }
  return pendingGenerationQueue.splice(bestIdx, 1)[0];
}

async function executeSingleGenerationJob(job: QueuedGenerationJob): Promise<void> {
  const { req, controller } = job;
  if (controller.signal.aborted) {
    activeControllers.delete(req.requestId);
    post({
      type: 'done',
      requestId: req.requestId,
      telemetry: runtime.getTelemetry(),
      multimodalTelemetry: multimodalCoordinator.getTelemetry(),
    });
    return;
  }

  await preemptBackgroundPrewarm();
  await preemptSpeculativePrefill();
  opfsPersistEpoch += 1;
  const jobPartId = req.partId ?? req.threadId ?? 'verdant-part-home';
  const sliceController = new AbortController();
  const onParentAbort = () => sliceController.abort();
  controller.signal.addEventListener('abort', onParentAbort, { once: true });
  activeGenerationController = controller;
  activeGenerationPartId = jobPartId;
  checkpointRequested = false;
  activeGenerationCount += 1;

  let finishGen!: () => void;
  const genDone = new Promise<void>((resolve) => {
    finishGen = resolve;
  });
  activeGenerationPromise = genDone;
  let shouldPersistAfterTurn = false;
  let checkpointed = false;

  try {
    let sliceCount = 0;
    const prevAccumText = job.continuationAccumulatedText ?? '';
    const prevTokensEmitted = job.continuationTokensEmitted ?? 0;
    const activePrompt = job.continuationPrompt ?? req.prompt;
    const targetMaxTokens = job.remainingMaxTokens ?? req.sampling.maxTokens;
    const sliceSampling: SamplingConfig =
      targetMaxTokens === req.sampling.maxTokens
        ? req.sampling
        : { ...req.sampling, maxTokens: targetMaxTokens };
    let lastSliceAccum = '';
    let lastPrefillPost = 0;
    let lastPrefillDetail: string | undefined;
    let lastTelemetryPost = 0;
    runtime.beginGenerationSlice();
    let lastStepMs = performance.now();
    const cachedVisual = req.visualKappa
      ? multimodalCoordinator.getCachedVisual(req.visualKappa)
      : undefined;
    const modelEmbDim = runtime.getModelSummary()?.hparams.embeddingLength ?? 0;
    const prefixEmbeddings =
      cachedVisual &&
      cachedVisual.embeddings.length > 0 &&
      cachedVisual.embeddingDim === modelEmbDim
        ? cachedVisual.embeddings
        : undefined;

    for await (const tokenEvent of runtime.generate(
      activePrompt,
      sliceSampling,
      sliceController.signal,
      (progress) => {
        const now = performance.now();
        const detailChanged =
          progress.detail !== undefined && progress.detail !== lastPrefillDetail;
        if (
          lastPrefillPost === 0 ||
          progress.completedTokens === 1 ||
          progress.completedTokens === progress.totalTokens ||
          (detailChanged && now - lastPrefillPost >= 100) ||
          now - lastPrefillPost > 120
        ) {
          lastPrefillPost = now;
          lastPrefillDetail = progress.detail;
          post({
            type: 'prefill',
            requestId: req.requestId,
            progress,
          });
        }
      },
      req.threadId,
      prefixEmbeddings,
    )) {
      sliceCount += 1;
      lastSliceAccum = tokenEvent.accumulatedText;
      const totalEmitted = prevTokensEmitted + sliceCount;
      const remainingTokens = Math.max(0, targetMaxTokens - sliceCount);
      const hasWaitingFocusedJob = pendingGenerationQueue.some(
        (q) => (q.req.partId ?? q.req.threadId ?? '') === focusedPartId,
      );
      const willCheckpoint =
        !tokenEvent.finished &&
        !controller.signal.aborted &&
        shouldCheckpointAtKvPageBoundary({
          checkpointRequested,
          jobPartId,
          focusedPartId,
          hasWaitingFocusedJob,
          emittedTokensInSlice: sliceCount,
          totalEmittedTokens: totalEmitted,
          remainingMaxTokens: remainingTokens,
        });

      const mergedEvent: TokenStreamEvent =
        prevAccumText.length > 0 || prevTokensEmitted > 0 || willCheckpoint
          ? {
              ...tokenEvent,
              accumulatedText: prevAccumText + tokenEvent.accumulatedText,
              generatedTokens: totalEmitted,
              finished: willCheckpoint ? false : tokenEvent.finished,
            }
          : tokenEvent;

      const nowMs = performance.now();
      const includeTelemetry =
        sliceCount === 1 || mergedEvent.finished || nowMs - lastTelemetryPost >= 150;
      if (includeTelemetry) {
        lastTelemetryPost = nowMs;
      }
      post({
        type: 'token',
        requestId: req.requestId,
        event: mergedEvent,
        telemetry: includeTelemetry ? runtime.getTelemetry() : undefined,
      });
      const stepUs = (nowMs - lastStepMs) * 1000;
      lastStepMs = nowMs;
      multimodalCoordinator.recordFrame(stepUs);

      if (willCheckpoint) {
        checkpointed = true;
        sliceController.abort();
        pendingGenerationQueue.push({
          req,
          controller,
          enqueuedSeq: job.enqueuedSeq,
          continuationPrompt: activePrompt + lastSliceAccum,
          continuationAccumulatedText: prevAccumText + lastSliceAccum,
          continuationTokensEmitted: totalEmitted,
          remainingMaxTokens: remainingTokens,
        });
        break;
      }

      await runtime.maybeYieldGenerationStep(stepUs);
    }

    if (!checkpointed) {
      shouldPersistAfterTurn = !controller.signal.aborted;
      post({
        type: 'done',
        requestId: req.requestId,
        telemetry: runtime.getTelemetry(),
        multimodalTelemetry: multimodalCoordinator.getTelemetry(),
      });
    }
  } catch (err) {
    if (!checkpointed) {
      post({
        type: 'error',
        requestId: req.requestId,
        message: err instanceof Error ? err.message : String(err),
      });
    }
  } finally {
    controller.signal.removeEventListener('abort', onParentAbort);
    activeGenerationCount = Math.max(0, activeGenerationCount - 1);
    if (!checkpointed) {
      activeControllers.delete(req.requestId);
    }
    if (activeGenerationController === controller) {
      activeGenerationController = null;
      activeGenerationPartId = null;
    }
    checkpointRequested = false;
    if (activeGenerationPromise === genDone) {
      activeGenerationPromise = null;
    }
    finishGen();
    if (
      !checkpointed &&
      activeGenerationCount === 0 &&
      pendingGenerationQueue.length === 0 &&
      activeLoadController === null
    ) {
      if (shouldPersistAfterTurn) {
        scheduleBackgroundOpfsPersist();
      }
      startBackgroundPrewarm();
    }
  }
}

async function pumpGenerationQueue(): Promise<void> {
  if (generationPumpActive) return;
  generationPumpActive = true;
  try {
    while (pendingGenerationQueue.length > 0) {
      const next = dequeueNextGenerationJob();
      if (!next) break;
      await executeSingleGenerationJob(next);
    }
  } finally {
    generationPumpActive = false;
  }
}

function formatEta(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds <= 0) return '';
  if (seconds < 60) return `${Math.ceil(seconds)}s left`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.ceil(seconds % 60);
  return `${mins}m ${secs}s left`;
}

function formatStreamProgressDetail(
  p: {
    receivedBytes: number;
    totalBytes: number;
    backingKind: VramBackingKind;
    bytesPerSecond?: number;
    etaSeconds?: number | null;
    resumedFromBytes?: number;
    retryAttempt?: number;
    maxRetries?: number;
    stage?: string;
    note?: string;
  },
  isLocalFile = false,
): string {
  if (p.stage === 'retrying' && p.note) {
    return p.note;
  }
  if (p.stage === 'verifying') {
    return p.note ?? 'Verifying model file integrity…';
  }
  const baseLabel = isLocalFile
    ? p.backingKind === 'local-file-sync'
      ? 'Opening local model'
      : 'Importing local model'
    : p.backingKind === 'opfs-demand'
      ? p.note ?? 'Preparing on-demand model cache'
      : p.resumedFromBytes && p.resumedFromBytes > 0
        ? 'Resuming download'
        : 'Downloading';

  const sizePart =
    p.totalBytes > 0
      ? `${formatBytes(p.receivedBytes)} / ${formatBytes(p.totalBytes)}`
      : formatBytes(p.receivedBytes);

  const metrics: string[] = [baseLabel, sizePart];
  if (p.bytesPerSecond && p.bytesPerSecond >= 64 * 1024) {
    metrics.push(`${(p.bytesPerSecond / (1024 * 1024)).toFixed(1)} MB/s`);
  }
  const eta = formatEta(p.etaSeconds);
  if (eta) {
    metrics.push(eta);
  }
  return metrics.join(' · ');
}

workerScope.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const req = event.data;

  try {
    if (req.type === 'shutdown') {
      prewarmAllowed = false;
      if (activePrewarmController) activePrewarmController.abort();
      if (activeSpeculativeController) activeSpeculativeController.abort();
      if (activeGenerationController) activeGenerationController.abort();
      if (activeLoadController) activeLoadController.abort();
      for (const ctrl of activeControllers.values()) ctrl.abort();
      activeControllers.clear();
      await drainBackgroundOpfsPersist();
      await getSharedOpfsCasStore().flushLedger().catch(() => undefined);
      disposeResidentNeuralImageWeights();
      runtime.unload();
      runtime.disposeThreadPool();
      try {
        opfsHandoverChannel?.close();
      } catch {
        // Ignore
      }
      opfsHandoverChannel = null;
      return;
    }

    if (req.type === 'pause_prewarm') {
      prewarmAllowed = false;
      await preemptBackgroundPrewarm();
      return;
    }

    if (req.type === 'resume_prewarm') {
      prewarmAllowed = true;
      if (req.bootPrompt) activeBootPrompt = req.bootPrompt;
      startBackgroundPrewarm();
      return;
    }

    if (req.type === 'load_buffer') {
      await preemptBackgroundPrewarm();
      await preemptSpeculativePrefill();
      await preemptActiveGeneration();
      await preemptActiveLoad();
      await drainBackgroundOpfsPersist();
      await threadPoolReadyPromise;
      const summary = runtime.loadFromBuffer(
        req.id,
        req.displayName,
        req.source,
        req.buffer,
        req.maxContextCap,
      );
      await runtime.hydrateUnifiedStoreFromOpfs().catch(() => 0);
      const opfsStorage = await inspectOpfsVramStorage();
      post({
        type: 'loaded',
        requestId: req.requestId,
        summary,
        telemetry: runtime.getTelemetry(),
        opfsStorage,
      });
      return;
    }

    if (req.type === 'load_url') {
      await preemptBackgroundPrewarm();
      await preemptSpeculativePrefill();
      await preemptActiveGeneration();
      await preemptActiveLoad();
      await drainBackgroundOpfsPersist();
      await threadPoolReadyPromise;
      prewarmAllowed = true;
      activeBootPrompt = req.bootPrompt;
      contextWarmedModelId = null;
      const controller = new AbortController();
      activeControllers.set(req.requestId, controller);
      activeLoadController = controller;
      let modelUnloaded = false;
      let acquiredBacking: VramBackingStore | null = null;
      let committedToRuntime = false;
      let shouldStartPrewarm = false;
      let finishLoad!: () => void;
      const loadDone = new Promise<void>((resolve) => {
        finishLoad = resolve;
      });
      activeLoadPromise = loadDone;
      try {
        // Release existing SyncAccessHandle only when reloading the exact same model ID
        if (runtime.getModelSummary()?.id === req.id) {
          runtime.unload();
          modelUnloaded = true;
        }
        const activeIdBeforeLoad = runtime.getModelSummary()?.id;
        const backing = await streamUrlToVramBacking(req.id, req.url, {
          skipOpfs: req.skipOpfs,
          preferDemandStreaming: req.preferDemandStreaming,
          casStore: getSharedOpfsCasStore(),
          activeModelId: activeIdBeforeLoad,
          signal: controller.signal,
          onProgress: (p) => {
            post({
              type: 'progress',
              requestId: req.requestId,
              percent: p.percent,
              detail: formatStreamProgressDetail(p, false),
              receivedBytes: p.receivedBytes,
              totalBytes: p.totalBytes,
              backingKind: p.backingKind,
            });
          },
        });
        acquiredBacking = backing;

        post({
          type: 'progress',
          requestId: req.requestId,
          percent: 97,
          detail: `Loading ${req.displayName}`,
          receivedBytes: backing.byteLength,
          totalBytes: backing.byteLength,
          backingKind: backing.kind,
        });

        const preStorage = await inspectOpfsVramStorage();
        if (preStorage.quotaBytes > 0) {
          getSharedOpfsCasStore().setCapacityCeiling(
            2 * 1024 * 1024 * 1024,
            Math.max(256 * 1024 * 1024, Math.floor(preStorage.quotaBytes * 0.45)),
            preStorage.quotaBytes,
          );
        }
        committedToRuntime = true;
        const summary = await runtime.loadFromBackingAsync(
          req.id,
          req.displayName,
          req.url,
          backing,
          {
            maxContextCap: req.maxContextCap,
            weightResidencyPolicy: req.weightResidencyPolicy,
            backingQuotaBytes: preStorage.quotaBytes > 0 ? preStorage.quotaBytes : undefined,
            signal: controller.signal,
            onProgress: (p) => {
              post({
                type: 'progress',
                requestId: req.requestId,
                percent: p.percent,
                detail: p.detail,
                receivedBytes: backing.byteLength,
                totalBytes: backing.byteLength,
                backingKind: backing.kind,
              });
            },
          },
        );
        await runtime.hydrateUnifiedStoreFromOpfs().catch(() => 0);
        const opfsStorage = await inspectOpfsVramStorage();
        post({
          type: 'loaded',
          requestId: req.requestId,
          summary,
          telemetry: runtime.getTelemetry(),
          opfsStorage,
        });
        shouldStartPrewarm = !req.skipOpfs;
      } catch (err) {
        if (acquiredBacking && !committedToRuntime) {
          try {
            acquiredBacking.close();
          } catch {
            // Ignore duplicate close
          }
        }
        const opfsStorage = await inspectOpfsVramStorage().catch(() => undefined);
        const isAbort =
          (err instanceof DOMException && err.name === 'AbortError') ||
          (err instanceof Error && err.name === 'AbortError');
        post({
          type: 'error',
          requestId: req.requestId,
          message: isAbort
            ? 'Download cancelled.'
            : err instanceof Error
              ? err.message
              : String(err),
          opfsStorage,
          modelUnloaded,
        });
      } finally {
        activeControllers.delete(req.requestId);
        if (activeLoadController === controller) activeLoadController = null;
        if (activeLoadPromise === loadDone) activeLoadPromise = null;
        finishLoad();
        if (shouldStartPrewarm) {
          startBackgroundPrewarm();
        }
      }
      return;
    }

    if (req.type === 'load_file') {
      await preemptBackgroundPrewarm();
      await preemptSpeculativePrefill();
      await preemptActiveGeneration();
      await preemptActiveLoad();
      await drainBackgroundOpfsPersist();
      await threadPoolReadyPromise;
      const controller = new AbortController();
      activeControllers.set(req.requestId, controller);
      activeLoadController = controller;
      let modelUnloaded = false;
      let acquiredBacking: VramBackingStore | null = null;
      let committedToRuntime = false;
      let finishLoad!: () => void;
      const loadDone = new Promise<void>((resolve) => {
        finishLoad = resolve;
      });
      activeLoadPromise = loadDone;
      try {
        if (runtime.getModelSummary()?.id === req.id) {
          runtime.unload();
          modelUnloaded = true;
        }
        const preStorage = await inspectOpfsVramStorage();
        const backing = await streamFileToVramBacking(
          req.id,
          req.file,
          (p) => {
            post({
              type: 'progress',
              requestId: req.requestId,
              percent: p.percent,
              detail: formatStreamProgressDetail(p, true),
              receivedBytes: p.receivedBytes,
              totalBytes: p.totalBytes,
              backingKind: p.backingKind,
            });
          },
          controller.signal,
        );
        acquiredBacking = backing;

        post({
          type: 'progress',
          requestId: req.requestId,
          percent: 97,
          detail: `Loading ${req.displayName}`,
          receivedBytes: backing.byteLength,
          totalBytes: backing.byteLength,
          backingKind: backing.kind,
        });

        committedToRuntime = true;
        const summary = await runtime.loadFromBackingAsync(
          req.id,
          req.displayName,
          req.source,
          backing,
          {
            maxContextCap: req.maxContextCap,
            weightResidencyPolicy: req.weightResidencyPolicy,
            backingQuotaBytes: preStorage.quotaBytes > 0 ? preStorage.quotaBytes : undefined,
            signal: controller.signal,
            onProgress: (p) => {
              post({
                type: 'progress',
                requestId: req.requestId,
                percent: p.percent,
                detail: p.detail,
                receivedBytes: backing.byteLength,
                totalBytes: backing.byteLength,
                backingKind: backing.kind,
              });
            },
          },
        );
        await runtime.hydrateUnifiedStoreFromOpfs().catch(() => 0);
        const opfsStorage = await inspectOpfsVramStorage();
        post({
          type: 'loaded',
          requestId: req.requestId,
          summary,
          telemetry: runtime.getTelemetry(),
          opfsStorage,
        });
      } catch (err) {
        if (acquiredBacking && !committedToRuntime) {
          try {
            acquiredBacking.close();
          } catch {
            // Ignore duplicate close
          }
        }
        const opfsStorage = await inspectOpfsVramStorage().catch(() => undefined);
        const isAbort =
          (err instanceof DOMException && err.name === 'AbortError') ||
          (err instanceof Error && err.name === 'AbortError');
        post({
          type: 'error',
          requestId: req.requestId,
          message: isAbort
            ? 'Download cancelled.'
            : err instanceof Error
              ? err.message
              : String(err),
          opfsStorage,
          modelUnloaded,
        });
      } finally {
        activeControllers.delete(req.requestId);
        if (activeLoadController === controller) activeLoadController = null;
        if (activeLoadPromise === loadDone) activeLoadPromise = null;
        finishLoad();
      }
      return;
    }

    if (req.type === 'set_focus_part') {
      focusedPartId = req.partId || 'verdant-part-home';
      return;
    }

    if (req.type === 'cancel') {
      // Remove from pending queue if not started yet (never affects sibling parts!)
      const qIdx = pendingGenerationQueue.findIndex((j) => j.req.requestId === req.requestId);
      if (qIdx >= 0) {
        const [removed] = pendingGenerationQueue.splice(qIdx, 1);
        removed?.controller.abort();
        activeControllers.delete(req.requestId);
        post({
          type: 'done',
          requestId: req.requestId,
          telemetry: runtime.getTelemetry(),
          multimodalTelemetry: multimodalCoordinator.getTelemetry(),
        });
        return;
      }
      const controller = activeControllers.get(req.requestId);
      if (controller) {
        controller.abort();
        activeControllers.delete(req.requestId);
      }
      return;
    }

    if (req.type === 'get_telemetry') {
      post({
        type: 'telemetry',
        requestId: req.requestId,
        telemetry: runtime.getTelemetry(),
        multimodalTelemetry: multimodalCoordinator.getTelemetry(),
      });
      return;
    }

    if (req.type === 'perceive_image') {
      const rgba = new Uint8Array(req.rgbaBuffer);
      // Warm SigLIP mmproj in the background when Local AI is enabled without blocking perception
      if (req.allowAutoLoad && !activeVisionEncoder && !visionLoadPromise) {
        void ensureVisionEncoderOnDemand();
      }
      if (!activeVisionEncoder) {
        multimodalCoordinator.activateModality({
          modality: 'vision',
          modelId: 'verdant-vision-sense',
          residentBytes: rgba.byteLength,
          evict: () => {},
        });
      }
      const report = await perceiveImageFrame(
        req.name,
        { rgba, width: req.width, height: req.height },
        { visionEncoder: activeVisionEncoder, coordinator: multimodalCoordinator },
      );
      post({
        type: 'image_perceived',
        requestId: req.requestId,
        report,
        multimodalTelemetry: multimodalCoordinator.getTelemetry(),
      });
      return;
    }

    if (req.type === 'perceive_audio') {
      const pcm = new Float32Array(req.pcmBuffer);
      const whisper =
        activeWhisperEngine ?? (req.allowAutoLoad ? await ensureWhisperEngineOnDemand() : null);
      if (!whisper) {
        multimodalCoordinator.activateModality({
          modality: 'audio',
          modelId: 'verdant-audio-sense',
          residentBytes: pcm.byteLength,
          evict: () => {},
        });
      }
      const kappa = computeMediaKappa('audio', pcm, `${req.sampleRate}hz:${pcm.length}`);
      let energySum = 0;
      for (let i = 0; i < pcm.length; i++) energySum += pcm[i]! * pcm[i]!;
      const rmsEnergy = pcm.length > 0 ? Math.sqrt(energySum / pcm.length) : 0;
      const speechDetected = rmsEnergy >= 0.008;
      const mel = computeLogMelSpectrogram(pcm);
      const { entry, cacheHit } = await multimodalCoordinator.getOrTranscribeAudio(
        kappa,
        async () => {
          const t0 = performance.now();
          const text = whisper && speechDetected ? whisper.transcribe(mel, 48).text : '';
          return {
            durationSec: mel.durationSec,
            text,
            transcribeMs: Math.max(0.5, performance.now() - t0),
          };
        },
      );
      const summary = entry.text
        ? `Transcribed voice (${entry.durationSec.toFixed(1)}s): "${entry.text}"`
        : speechDetected
          ? `Voice clip attached (${entry.durationSec.toFixed(1)}s, ${mel.nFrames} acoustic frames, speech activity detected).`
          : `Audio clip attached (${entry.durationSec.toFixed(1)}s, low ambient level).`;
      post({
        type: 'audio_perceived',
        requestId: req.requestId,
        report: {
          kappa,
          durationSec: entry.durationSec,
          frames: mel.nFrames,
          rmsEnergy,
          speechDetected,
          transcript: entry.text,
          cacheHit,
          summary,
        },
        multimodalTelemetry: multimodalCoordinator.getTelemetry(),
      });
      return;
    }

    if (req.type === 'synthesize_audio') {
      const comp = composeMultiTrackArrangement({
        title: req.title,
        style: req.style,
        bpm: req.bpm,
        rootMidi: req.rootMidi,
        scale: req.scale,
        durationSec: req.durationSec,
      });
      const rawPcm = synthesizeCompositionStereoPcm(comp, 22_050);
      const pcm = masterStereoPcm(rawPcm.left, rawPcm.right, rawPcm.sampleRate);
      const wav = encodeStereoWav(pcm.left, pcm.right, pcm.sampleRate);
      multimodalCoordinator.activateModality({
        modality: 'audio',
        modelId: 'verdant-modal-audio-synth',
        residentBytes: wav.byteLength,
        evict: () => {},
      });
      const transferCopy = new Uint8Array(wav.byteLength);
      transferCopy.set(wav);
      const transferBuf = transferCopy.buffer;
      workerScope.postMessage(
        {
          type: 'audio_synthesized',
          requestId: req.requestId,
          wavBuffer: transferBuf,
          sampleRate: pcm.sampleRate,
          durationSec: comp.durationSec,
          multimodalTelemetry: multimodalCoordinator.getTelemetry(),
        } satisfies WorkerResponse,
        [transferBuf],
      );
      return;
    }

    if (req.type === 'synthesize_image') {
      const t0 = performance.now();
      const neuralWeights =
        getResidentNeuralImageWeights() ??
        (req.allowAutoLoad ? await ensureImageWeightsOnDemand() : null);
      if (!neuralWeights) {
        post({
          type: 'image_synthesized',
          requestId: req.requestId,
          neuralAvailable: false,
          width: 0,
          height: 0,
          rgbaBuffer: null,
          multimodalTelemetry: multimodalCoordinator.getTelemetry(),
        });
        return;
      }
      const bpeVocab =
        neuralWeights.streamGraph?.clipBpeVocab ??
        (await ensureClipBpeVocabularyCached(Boolean(req.allowAutoLoad)));
      const promptTokens = encodePromptToClipTokens(
        req.prompt,
        neuralWeights.clip.vocabSize,
        neuralWeights.clip.maxSeqLen,
        bpeVocab,
      );
      const synth = await synthesizeDistilledImageAsync({
        promptTokens,
        latentH: Math.max(8, Math.min(64, req.latentH ?? 16)),
        latentW: Math.max(8, Math.min(64, req.latentW ?? 16)),
        seed: req.seed,
        loaded: neuralWeights,
        scratch: imageSynthesisScratch,
        yieldFrame: () => cooperativeYielder.yieldNow(),
      });
      multimodalCoordinator.frameOracle.recordFrame((performance.now() - t0) * 1000);
      const copy = new Uint8Array(synth.rgba.byteLength);
      copy.set(synth.rgba);
      const transferBuf = copy.buffer;
      workerScope.postMessage(
        {
          type: 'image_synthesized',
          requestId: req.requestId,
          neuralAvailable: true,
          width: synth.width,
          height: synth.height,
          rgbaBuffer: transferBuf,
          multimodalTelemetry: multimodalCoordinator.getTelemetry(),
        } satisfies WorkerResponse,
        [transferBuf],
      );
      return;
    }

    if (req.type === 'qa_visual_semantic') {
      const t0 = performance.now();
      const rgba = new Uint8Array(req.rgbaBuffer);
      if (req.allowAutoLoad && !activeVisionEncoder && !visionLoadPromise) {
        void ensureVisionEncoderOnDemand();
      }
      const runVlmCritique = async (
        visionKappa: string,
        target: SemanticVisualQaTarget,
      ): Promise<string | null> => {
        if (!activeVisionEncoder || !runtime.isLoaded || activeGenerationCount > 0) {
          return null;
        }
        const cached = multimodalCoordinator.getCachedVisual(visionKappa);
        const modelEmbDim = runtime.getModelSummary()?.hparams.embeddingLength ?? 0;
        if (!cached || cached.embeddings.length === 0 || cached.embeddingDim !== modelEmbDim) {
          return null;
        }
        const qaCtrl = new AbortController();
        const qaTimer = setTimeout(() => qaCtrl.abort(), 4500);
        let text = '';
        activeGenerationCount += 1;
        try {
          const prompt = `<|im_start|>system\nYou are a concise visual QA critic. Reply ONLY with "ACCEPTED" if the image clearly depicts the target subject and attributes, or "REFINE: <short missing visual attribute>".<|im_end|>\n<|im_start|>user\nTarget subject: ${target.canonicalSubject}. Attributes: ${target.attributes.slice(0, 4).join(', ')}. Lighting: ${target.lighting ?? 'balanced'}. Evaluate the attached visual embeddings.<|im_end|>\n<|im_start|>assistant\n`;
          for await (const ev of runtime.generate(
            prompt,
            {
              temperature: 0.1,
              topP: 0.9,
              topK: 20,
              repeatPenalty: 1.05,
              maxTokens: 32,
              fullContextOutput: false,
            },
            qaCtrl.signal,
            undefined,
            'qa:vision',
            cached.embeddings,
          )) {
            text = ev.accumulatedText;
          }
        } catch {
          return null;
        } finally {
          clearTimeout(qaTimer);
          activeGenerationCount = Math.max(0, activeGenerationCount - 1);
        }
        return text.trim() || null;
      };
      const report = await evaluateSemanticVisualQa(
        { rgba, width: req.width, height: req.height },
        req.target,
        {
          visionEncoder: activeVisionEncoder,
          coordinator: multimodalCoordinator,
          runVlmCritique,
          minAcceptScoreQ10: req.minAcceptScoreQ10,
        },
      );
      multimodalCoordinator.frameOracle.recordFrame((performance.now() - t0) * 1000);
      post({
        type: 'visual_qa_result',
        requestId: req.requestId,
        report,
        multimodalTelemetry: multimodalCoordinator.getTelemetry(),
      });
      return;
    }

    if (req.type === 'count_tokens') {
      const tokenCount = runtime.countTokens(req.prompt);
      const maxContextWindow = runtime.getModelSummary()?.maxContextWindow ?? 0;
      post({
        type: 'token_count',
        requestId: req.requestId,
        tokenCount,
        maxContextWindow,
      });
      return;
    }

    if (req.type === 'run_conformance') {
      const report = runVTensorConformanceSuite();
      post({
        type: 'conformance',
        requestId: req.requestId,
        report,
      });
      return;
    }

    if (req.type === 'set_precision_mode') {
      await preemptSpeculativePrefill();
      await preemptActiveGeneration();
      runtime.setPrecisionMode(req.mode);
      post({
        type: 'telemetry',
        requestId: req.requestId,
        telemetry: runtime.getTelemetry(),
      });
      return;
    }

    if (req.type === 'set_weight_residency_policy') {
      await preemptBackgroundPrewarm();
      await preemptSpeculativePrefill();
      await preemptActiveGeneration();
      runtime.setWeightResidencyPolicy(req.policy);
      post({
        type: 'telemetry',
        requestId: req.requestId,
        telemetry: runtime.getTelemetry(),
      });
      return;
    }

    if (req.type === 'inspect_opfs') {
      const storage = await inspectOpfsVramStorage();
      post({
        type: 'opfs_storage',
        requestId: req.requestId,
        storage,
      });
      return;
    }

    if (req.type === 'delete_opfs_model') {
      await preemptBackgroundPrewarm();
      await preemptSpeculativePrefill();
      await preemptActiveGeneration();
      await drainBackgroundOpfsPersist();
      let modelUnloaded = false;
      if (runtime.getModelSummary()?.id === req.modelId) {
        runtime.unload();
        modelUnloaded = true;
      }
      await deleteOpfsModel(req.modelId);
      await getSharedOpfsCasStore().releaseOwner(req.modelId).catch(() => undefined);
      const storage = await inspectOpfsVramStorage();
      post({
        type: 'opfs_storage',
        requestId: req.requestId,
        storage,
        modelUnloaded,
      });
      return;
    }

    if (req.type === 'clear_opfs_models') {
      await preemptBackgroundPrewarm();
      await preemptSpeculativePrefill();
      await preemptActiveGeneration();
      await drainBackgroundOpfsPersist();
      const activeId = runtime.getModelSummary()?.id;
      const before = await inspectOpfsVramStorage();
      const activeKind = runtime.getModelSummary()?.backingKind;
      const canWipeDir = !activeId || (activeKind !== 'opfs-sync' && activeKind !== 'opfs-demand');
      if (canWipeDir) {
        await clearAllOpfsModels();
        await getSharedOpfsCasStore().clearAll().catch(() => undefined);
      } else {
        for (const entry of before.storedModels) {
          if (entry.modelId !== activeId) {
            await deleteOpfsModel(entry.modelId);
            await getSharedOpfsCasStore().releaseOwner(entry.modelId).catch(() => undefined);
          }
        }
        for (const partial of before.partialModels ?? []) {
          if (partial.modelId !== activeId) {
            await deleteOpfsModel(partial.modelId);
            await getSharedOpfsCasStore().releaseOwner(partial.modelId).catch(() => undefined);
          }
        }
      }
      const storage = await inspectOpfsVramStorage();
      post({
        type: 'opfs_storage',
        requestId: req.requestId,
        storage,
      });
      return;
    }

    if (req.type === 'speculative_prefill') {
      if (
        activeGenerationCount > 0 ||
        !runtime.isLoaded ||
        (runtime.getModelSummary()?.backingKind === 'opfs-demand' &&
          !runtime.isDemandModelFullyWarm())
      ) {
        return;
      }
      await preemptSpeculativePrefill();
      if (
        activeGenerationCount > 0 ||
        !runtime.isLoaded ||
        (runtime.getModelSummary()?.backingKind === 'opfs-demand' &&
          !runtime.isDemandModelFullyWarm())
      ) {
        return;
      }

      const controller = new AbortController();
      activeSpeculativeController = controller;
      let lastPost = 0;

      let task: Promise<void> | null = null;
      task = (async () => {
        try {
          const result = await runtime.speculativePrefill(
            req.prompt,
            controller.signal,
            (p) => {
              const now = performance.now();
              if (p.readyTokens === p.targetTokens || now - lastPost > 140) {
                lastPost = now;
                post({
                  type: 'speculative_status',
                  requestId: req.requestId,
                  active: p.readyTokens < p.targetTokens,
                  readyTokens: p.readyTokens,
                  targetTokens: p.targetTokens,
                });
              }
            },
            req.threadId,
          );
          if (!result.aborted && result.newlyPrefilledTokens > 0) {
            await runtime.persistUnifiedStoreToOpfs().catch(() => 0);
          }
          post({
            type: 'speculative_status',
            requestId: req.requestId,
            active: false,
            readyTokens: result.readyTokens,
            targetTokens: result.targetTokens,
            telemetry: runtime.getTelemetry(),
          });
        } finally {
          if (activeSpeculativeController === controller) {
            activeSpeculativeController = null;
          }
          if (activeSpeculativePromise === task) {
            activeSpeculativePromise = null;
          }
        }
      })();
      activeSpeculativePromise = task;
      await task;
      return;
    }

    if (req.type === 'generate') {
      // Non-preemptive salience-ordered scheduling: NEVER abort another part's active generation!
      // If a focused foreground part arrives while a background part is running, request a clean
      // 64-token KV page-boundary checkpoint so foreground responds rapidly with zero lost tokens.
      const controller = new AbortController();
      activeControllers.set(req.requestId, controller);
      const incomingPartId = req.partId ?? req.threadId ?? 'verdant-part-home';
      if (
        activeGenerationCount > 0 &&
        incomingPartId === focusedPartId &&
        activeGenerationPartId !== null &&
        activeGenerationPartId !== focusedPartId
      ) {
        checkpointRequested = true;
      }
      pendingGenerationQueue.push({
        req,
        controller,
        enqueuedSeq: ++generationEnqueueSeq,
      });
      void pumpGenerationQueue();
      return;
    }
  } catch (err) {
    if ('requestId' in req) {
      post({
        type: 'error',
        requestId: req.requestId,
        message: err instanceof Error ? err.message : String(err),
      });
    }
  }
};
