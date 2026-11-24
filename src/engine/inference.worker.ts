/// <reference lib="webworker" />

import {
  LlamaVTensorRuntime,
  type LoadedModelSummary,
  type PrefillProgressEvent,
  type SamplingConfig,
  type TokenStreamEvent,
} from './llamaRuntime';
import { formatBytes } from './modelCatalog';
import type { PrecisionContractMode } from './performanceContract';
import {
  clearAllOpfsModels,
  deleteOpfsModel,
  inspectOpfsVramStorage,
  streamFileToVramBacking,
  streamUrlToVramBacking,
  type OpfsQuotaSnapshot,
  type VramBackingKind,
} from './vramBackingStore';
import type { WeightResidencyPolicy } from './vramWeightPager';
import {
  type ConformanceSuiteReport,
  runVTensorConformanceSuite,
  type VTensorTelemetrySnapshot,
} from './vtensorProvider';
import { VerdantWebGpuSubstrate } from './verdantWebGpu';

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
      maxContextCap?: number;
      weightResidencyPolicy?: WeightResidencyPolicy;
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
    };

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
    }
  | {
      type: 'telemetry';
      requestId: string;
      telemetry: VTensorTelemetrySnapshot;
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
      type: 'error';
      requestId: string;
      message: string;
      opfsStorage?: OpfsQuotaSnapshot;
      modelUnloaded?: boolean;
    };

const runtime = new LlamaVTensorRuntime();
const webgpuSubstrate = new VerdantWebGpuSubstrate();
runtime.attachWebGpuTelemetry(() => webgpuSubstrate.getTelemetry());
void webgpuSubstrate.probeAndInit();
const threadPoolReadyPromise = runtime.initThreadPool().catch(() => undefined);

const activeControllers = new Map<string, AbortController>();
let activeSpeculativeController: AbortController | null = null;
let activeSpeculativePromise: Promise<void> | null = null;
let activeGenerationCount = 0;

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
  if (activeSpeculativeController) {
    activeSpeculativeController.abort();
  }
  if (activeSpeculativePromise) {
    try {
      await activeSpeculativePromise;
    } catch {
      // Ignore speculative abort/cancellation errors
    }
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
    return p.note ?? 'Verifying GGUF integrity in OPFS vRAM…';
  }
  const baseLabel = isLocalFile
    ? p.backingKind === 'opfs-sync'
      ? 'Streaming local GGUF to OPFS vRAM'
      : 'Reading local GGUF'
    : p.backingKind === 'opfs-sync'
      ? p.resumedFromBytes && p.resumedFromBytes > 0
        ? 'Resuming OPFS vRAM stream'
        : 'Streaming to OPFS vRAM'
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
    if (req.type === 'load_buffer') {
      await preemptSpeculativePrefill();
      await threadPoolReadyPromise;
      const summary = runtime.loadFromBuffer(
        req.id,
        req.displayName,
        req.source,
        req.buffer,
        req.maxContextCap ?? 1024,
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
      await preemptSpeculativePrefill();
      await threadPoolReadyPromise;
      const controller = new AbortController();
      activeControllers.set(req.requestId, controller);
      let modelUnloaded = false;
      try {
        // Release existing SyncAccessHandle only when reloading the exact same model ID
        if (runtime.getModelSummary()?.id === req.id) {
          runtime.unload();
          modelUnloaded = true;
        }
        const preStorage = await inspectOpfsVramStorage();
        const backing = await streamUrlToVramBacking(req.id, req.url, {
          skipOpfs: req.skipOpfs,
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

        post({
          type: 'progress',
          requestId: req.requestId,
          percent: 97,
          detail:
            backing.kind === 'opfs-sync'
              ? `Binding OPFS vRAM & Prism contract for ${req.displayName}`
              : `Allocating vTensor buffers for ${req.displayName}`,
          receivedBytes: backing.byteLength,
          totalBytes: backing.byteLength,
          backingKind: backing.kind,
        });

        const summary = runtime.loadFromBacking(req.id, req.displayName, req.url, backing, {
          maxContextCap: req.maxContextCap ?? 1024,
          weightResidencyPolicy: req.weightResidencyPolicy,
          backingQuotaBytes: preStorage.quotaBytes > 0 ? preStorage.quotaBytes : undefined,
        });
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
      }
      return;
    }

    if (req.type === 'load_file') {
      await preemptSpeculativePrefill();
      await threadPoolReadyPromise;
      const controller = new AbortController();
      activeControllers.set(req.requestId, controller);
      let modelUnloaded = false;
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

        post({
          type: 'progress',
          requestId: req.requestId,
          percent: 97,
          detail: `Binding OPFS vRAM & Prism contract for ${req.displayName}`,
          receivedBytes: backing.byteLength,
          totalBytes: backing.byteLength,
          backingKind: backing.kind,
        });

        const summary = runtime.loadFromBacking(req.id, req.displayName, req.source, backing, {
          maxContextCap: req.maxContextCap ?? 1024,
          weightResidencyPolicy: req.weightResidencyPolicy,
          backingQuotaBytes: preStorage.quotaBytes > 0 ? preStorage.quotaBytes : undefined,
        });
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
      }
      return;
    }

    if (req.type === 'cancel') {
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
      runtime.setPrecisionMode(req.mode);
      post({
        type: 'telemetry',
        requestId: req.requestId,
        telemetry: runtime.getTelemetry(),
      });
      return;
    }

    if (req.type === 'set_weight_residency_policy') {
      await preemptSpeculativePrefill();
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
      await preemptSpeculativePrefill();
      let modelUnloaded = false;
      if (runtime.getModelSummary()?.id === req.modelId) {
        runtime.unload();
        modelUnloaded = true;
      }
      await deleteOpfsModel(req.modelId);
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
      await preemptSpeculativePrefill();
      const activeId = runtime.getModelSummary()?.id;
      const before = await inspectOpfsVramStorage();
      const canWipeDir = !activeId || runtime.getModelSummary()?.backingKind !== 'opfs-sync';
      if (canWipeDir) {
        await clearAllOpfsModels();
      } else {
        for (const entry of before.storedModels) {
          if (entry.modelId !== activeId) {
            await deleteOpfsModel(entry.modelId);
          }
        }
        for (const partial of before.partialModels ?? []) {
          if (partial.modelId !== activeId) {
            await deleteOpfsModel(partial.modelId);
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
      if (activeGenerationCount > 0 || !runtime.isLoaded) {
        return;
      }
      await preemptSpeculativePrefill();
      if (activeGenerationCount > 0 || !runtime.isLoaded) {
        return;
      }

      const controller = new AbortController();
      activeSpeculativeController = controller;
      let lastPost = 0;

      let task: Promise<void> | null = null;
      task = (async () => {
        try {
          const result = await runtime.speculativePrefill(req.prompt, controller.signal, (p) => {
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
          });
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
      await preemptSpeculativePrefill();
      const controller = new AbortController();
      activeControllers.set(req.requestId, controller);
      activeGenerationCount += 1;

      try {
        let count = 0;
        let lastPrefillPost = 0;
        let lastTelemetryPost = 0;
        runtime.beginGenerationSlice();
        let lastStepMs = performance.now();
        for await (const tokenEvent of runtime.generate(
          req.prompt,
          req.sampling,
          controller.signal,
          (progress) => {
            const now = performance.now();
            if (
              progress.completedTokens === 1 ||
              progress.completedTokens === progress.totalTokens ||
              now - lastPrefillPost > 120
            ) {
              lastPrefillPost = now;
              post({
                type: 'prefill',
                requestId: req.requestId,
                progress,
              });
            }
          },
        )) {
          count += 1;
          const nowMs = performance.now();
          const includeTelemetry =
            count === 1 || tokenEvent.finished || nowMs - lastTelemetryPost >= 150;
          if (includeTelemetry) {
            lastTelemetryPost = nowMs;
          }
          post({
            type: 'token',
            requestId: req.requestId,
            event: tokenEvent,
            telemetry: includeTelemetry ? runtime.getTelemetry() : undefined,
          });
          // Theorem T7 zero-clamp MessageChannel cooperative yield gated by 16,666 µs frame budget
          const stepUs = (nowMs - lastStepMs) * 1000;
          lastStepMs = nowMs;
          await runtime.maybeYieldGenerationStep(stepUs);
        }

        await runtime.persistUnifiedStoreToOpfs().catch(() => 0);
        post({
          type: 'done',
          requestId: req.requestId,
          telemetry: runtime.getTelemetry(),
        });
      } finally {
        activeGenerationCount = Math.max(0, activeGenerationCount - 1);
        activeControllers.delete(req.requestId);
      }
    }
  } catch (err) {
    post({
      type: 'error',
      requestId: req.requestId,
      message: err instanceof Error ? err.message : String(err),
    });
  }
};
