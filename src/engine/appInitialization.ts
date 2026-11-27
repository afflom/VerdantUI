/**
 * VerdantUI Application Initialization State Machine (`src/engine/appInitialization.ts`)
 *
 * Pure, deterministic state derivation for the Application Initializing Screen shown after
 * streaming consent (and on returning startup while mounting) until VerdantUI is ready to use.
 * Keeps all user-facing copy plain, concise, and free of internal engineering jargon.
 */

import { isInternalBootstrapModel } from './modelCatalog';
import type { EngineStatus, StreamingConsentState } from './workerClient';

export type InitStepId = 'workspace' | 'capabilities' | 'runtime';
export type InitStepStatus = 'pending' | 'active' | 'done' | 'error';

export interface InitStepItem {
  readonly id: InitStepId;
  readonly label: string;
  readonly detail: string;
  readonly status: InitStepStatus;
}

export type AppInitPhase = 'hidden' | 'consent' | 'initializing' | 'error' | 'ready';

export interface AppInitializationInput {
  readonly streamingConsent: StreamingConsentState;
  readonly engineStatus: EngineStatus;
  readonly progressPercent: number;
  readonly initStatusDetail: string;
  readonly modelId: string | null;
  readonly errorMessage: string | null;
  readonly storesHydrated: boolean;
  /** Live OPFS + WASM + system-context warming status for instant Turn 1 TTFT. */
  readonly prewarmActive?: boolean;
  readonly prewarmPercent?: number;
  readonly prewarmPhase?: 'idle' | 'streaming' | 'warming-context' | 'ready' | 'error';
  readonly prewarmBytesPerSec?: number;
  readonly prewarmEtaSeconds?: number | null;
  readonly prewarmError?: string | null;
  readonly contextWarmed?: boolean;
  /** True once the user has already reached a usable session in the current mode. */
  readonly sessionUnlocked: boolean;
}

export interface AppInitializationView {
  readonly visible: boolean;
  readonly phase: AppInitPhase;
  readonly headline: string;
  readonly statusLine: string;
  readonly overallPercent: number;
  readonly steps: readonly InitStepItem[];
  readonly errorMessage: string | null;
  readonly canRetry: boolean;
  readonly canSwitchOffline: boolean;
}

/**
 * Translates raw worker/loader progress strings into calm, plain-English initialization status
 * lines without leaking internal file formats or engine identifiers.
 */
export function formatInitStatusLine(
  rawDetail: string,
  engineStatus: EngineStatus,
  progressPercent: number,
): string {
  const cleaned = rawDetail.trim();
  if (engineStatus === 'loading' || progressPercent >= 96) {
    return 'Starting assistant on your device…';
  }
  if (/retrying/i.test(cleaned)) {
    return 'Reconnecting to continue setup…';
  }
  if (/verifying/i.test(cleaned)) {
    return 'Verifying downloaded files…';
  }
  if (/checking local cache/i.test(cleaned)) {
    return 'Checking saved data on your device…';
  }
  if (/preparing on-demand|connecting/i.test(cleaned)) {
    return 'Connecting capabilities to your browser…';
  }
  if (/loading layer|caching|initializing/i.test(cleaned)) {
    return 'Preparing assistant on your device…';
  }
  if (engineStatus === 'downloading') {
    return progressPercent > 0
      ? `Streaming core capabilities (${Math.min(99, Math.round(progressPercent))}%)…`
      : 'Connecting capabilities to your browser…';
  }
  return 'Initializing VerdantUI…';
}

function formatCompactTransferRate(bytesPerSec: number, etaSeconds: number | null | undefined): string {
  const parts: string[] = [];
  if (bytesPerSec >= 128 * 1024) {
    parts.push(`${(bytesPerSec / (1024 * 1024)).toFixed(1)} MB/s`);
  }
  if (etaSeconds != null && Number.isFinite(etaSeconds) && etaSeconds > 0.5) {
    if (etaSeconds < 60) {
      parts.push(`${Math.ceil(etaSeconds)}s left`);
    } else {
      const m = Math.floor(etaSeconds / 60);
      const s = Math.ceil(etaSeconds % 60);
      parts.push(`${m}m ${s}s left`);
    }
  }
  return parts.join(' · ');
}

export function isVerdantUsableForConsent(params: {
  readonly streamingConsent: StreamingConsentState;
  readonly engineStatus: EngineStatus;
  readonly modelId: string | null;
  readonly storesHydrated: boolean;
  readonly contextWarmed?: boolean;
  readonly prewarmPhase?: 'idle' | 'streaming' | 'warming-context' | 'ready' | 'error';
}): boolean {
  if (!params.storesHydrated) return false;
  if (params.streamingConsent === 'pending') {
    return true;
  }
  if (params.streamingConsent === 'paused') {
    return (
      params.modelId !== null &&
      params.engineStatus !== 'downloading' &&
      params.engineStatus !== 'loading'
    );
  }
  // Accepted streaming consent: unlock VerdantUI ONLY once the unified core runtime,
  // active capabilities, AND initial context are warm so Turn 1 achieves instant TTFT & peak tok/s!
  const hasRealRuntime = params.modelId !== null && !isInternalBootstrapModel(params.modelId);
  const runtimeReady =
    hasRealRuntime &&
    (params.engineStatus === 'ready' || params.engineStatus === 'generating');
  if (!runtimeReady) return false;
  return Boolean(params.contextWarmed || params.prewarmPhase === 'ready');
}

export function deriveAppInitializationState(
  input: AppInitializationInput,
): AppInitializationView {
  const {
    streamingConsent,
    engineStatus,
    progressPercent,
    initStatusDetail,
    modelId,
    errorMessage,
    storesHydrated,
    prewarmActive = false,
    prewarmPercent = 0,
    prewarmPhase = 'idle',
    prewarmBytesPerSec = 0,
    prewarmEtaSeconds = null,
    prewarmError = null,
    contextWarmed = false,
    sessionUnlocked,
  } = input;

  // First-time visitor: present the Splash Initialization Screen with built-in consent directly
  // on page load (returning users with saved consent bypass this phase automatically).
  if (streamingConsent === 'pending') {
    return {
      visible: true,
      phase: 'consent',
      headline: 'Welcome to VerdantUI',
      statusLine:
        'VerdantUI runs privately in your browser and streams capabilities and your chosen content sources directly to your device as needed.',
      overallPercent: 0,
      steps: [],
      errorMessage: null,
      canRetry: false,
      canSwitchOffline: true,
    };
  }

  const usable = isVerdantUsableForConsent({
    streamingConsent,
    engineStatus,
    modelId,
    storesHydrated,
    contextWarmed,
    prewarmPhase,
  });

  if (sessionUnlocked || usable) {
    return {
      visible: false,
      phase: 'ready',
      headline: 'VerdantUI is ready',
      statusLine: 'Ready to use',
      overallPercent: 100,
      steps: [
        { id: 'workspace', label: 'Workspace & saved library', detail: 'Ready', status: 'done' },
        {
          id: 'capabilities',
          label: streamingConsent === 'paused' ? 'Offline mode' : 'Browser capabilities',
          detail: 'Ready',
          status: 'done',
        },
        { id: 'runtime', label: 'Assistant & context', detail: 'Ready', status: 'done' },
      ],
      errorMessage: null,
      canRetry: false,
      canSwitchOffline: false,
    };
  }

  const clampedMountProgress = Math.max(0, Math.min(100, Math.round(progressPercent)));
  const clampedPrewarmPct = Math.max(0, Math.min(100, Math.round(prewarmPercent)));
  const hasRealRuntime = modelId !== null && !isInternalBootstrapModel(modelId);
  const combinedError = errorMessage || prewarmError;
  const hasInitError =
    streamingConsent === 'accepted' &&
    combinedError !== null &&
    !prewarmActive &&
    engineStatus !== 'downloading' &&
    engineStatus !== 'loading' &&
    (!hasRealRuntime || prewarmPhase === 'error');

  // Offline-only startup: brief hydration + local bootstrap check
  if (streamingConsent === 'paused') {
    const overallPercent = storesHydrated ? 70 : 30;
    return {
      visible: true,
      phase: 'initializing',
      headline: 'Starting VerdantUI offline',
      statusLine: storesHydrated
        ? 'Preparing offline workspace…'
        : 'Loading your saved spaces and library…',
      overallPercent,
      steps: [
        {
          id: 'workspace',
          label: 'Workspace & saved library',
          detail: storesHydrated ? 'Ready' : 'Loading…',
          status: storesHydrated ? 'done' : 'active',
        },
        {
          id: 'capabilities',
          label: 'Offline mode',
          detail: 'Local only',
          status: 'done',
        },
        {
          id: 'runtime',
          label: 'Local workspace runtime',
          detail: modelId ? 'Ready' : 'Starting…',
          status: modelId ? 'done' : storesHydrated ? 'active' : 'pending',
        },
      ],
      errorMessage: null,
      canRetry: false,
      canSwitchOffline: false,
    };
  }

  // Accepted streaming initialization:
  // Stage 1: Workspace & saved library (0-8%)
  // Stage 2: Streaming core capabilities to browser storage (8-86%)
  // Stage 3: Loading assistant into memory & preparing context for instant first response (86-100%)
  const workspaceStep: InitStepItem = {
    id: 'workspace',
    label: 'Workspace & saved library',
    detail: storesHydrated ? 'Ready' : 'Loading saved spaces…',
    status: storesHydrated ? 'done' : 'active',
  };

  const capabilitiesReady =
    hasRealRuntime && (prewarmPhase === 'warming-context' || prewarmPhase === 'ready' || contextWarmed);
  const rateSuffix = formatCompactTransferRate(prewarmBytesPerSec, prewarmEtaSeconds);
  const capabilitiesDetail = hasInitError
    ? 'Connection interrupted'
    : capabilitiesReady
      ? 'Ready'
      : hasRealRuntime
        ? clampedPrewarmPct > 0
          ? `${clampedPrewarmPct}%`
          : 'Starting…'
        : clampedMountProgress > 0 && engineStatus === 'downloading'
          ? `Connecting…`
          : 'Connecting…';

  const capabilitiesStep: InitStepItem = {
    id: 'capabilities',
    label: 'Core capabilities',
    detail: capabilitiesDetail,
    status: hasInitError
      ? 'error'
      : capabilitiesReady
        ? 'done'
        : storesHydrated || engineStatus === 'downloading' || hasRealRuntime
          ? 'active'
          : 'pending',
  };

  const warmingContextNow = hasRealRuntime && prewarmPhase === 'warming-context';
  const runtimeStep: InitStepItem = {
    id: 'runtime',
    label: 'Assistant & context',
    detail: hasInitError
      ? 'Waiting to retry'
      : contextWarmed
        ? 'Ready'
        : warmingContextNow
          ? 'Preparing context…'
          : 'Waiting…',
    status: hasInitError
      ? 'pending'
      : contextWarmed
        ? 'done'
        : warmingContextNow
          ? 'active'
          : 'pending',
  };

  // Smooth monotonic progress across all 3 stages:
  // Workspace: 0-8%, Header mount: 8-14%, Streaming core capabilities: 14-86%, Warming context: 86-98%
  let overallPercent = storesHydrated ? 8 : 4;
  if (!hasRealRuntime) {
    if (engineStatus === 'downloading') {
      overallPercent = Math.max(overallPercent, Math.round(8 + (clampedMountProgress / 100) * 5));
    } else if (engineStatus === 'loading') {
      overallPercent = Math.max(overallPercent, 14);
    }
  } else if (prewarmPhase === 'warming-context') {
    overallPercent = 92;
  } else if (prewarmPhase === 'ready' || contextWarmed) {
    overallPercent = storesHydrated ? 100 : 98;
  } else {
    overallPercent = Math.max(14, Math.round(14 + (clampedPrewarmPct / 100) * 72));
  }

  if (hasInitError) {
    return {
      visible: true,
      phase: 'error',
      headline: 'Unable to finish setup',
      statusLine: combinedError || 'Could not reach streaming capabilities.',
      overallPercent,
      steps: [workspaceStep, capabilitiesStep, runtimeStep],
      errorMessage: combinedError,
      canRetry: true,
      canSwitchOffline: true,
    };
  }

  const liveStatusLine = !hasRealRuntime
    ? formatInitStatusLine(initStatusDetail, engineStatus, clampedMountProgress)
    : warmingContextNow
      ? 'Loading assistant into memory and preparing context…'
      : rateSuffix
        ? `Streaming core capabilities (${clampedPrewarmPct}% · ${rateSuffix})`
        : clampedPrewarmPct > 0
          ? `Streaming core capabilities (${clampedPrewarmPct}%)…`
          : 'Streaming core capabilities to your browser…';

  return {
    visible: true,
    phase: 'initializing',
    headline: 'Initializing VerdantUI',
    statusLine: liveStatusLine,
    overallPercent,
    steps: [workspaceStep, capabilitiesStep, runtimeStep],
    errorMessage: null,
    canRetry: false,
    canSwitchOffline: true,
  };
}
