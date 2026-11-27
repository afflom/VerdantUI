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

export type AppInitPhase = 'hidden' | 'initializing' | 'error' | 'ready';

export interface AppInitializationInput {
  readonly streamingConsent: StreamingConsentState;
  readonly engineStatus: EngineStatus;
  readonly progressPercent: number;
  readonly initStatusDetail: string;
  readonly modelId: string | null;
  readonly errorMessage: string | null;
  readonly storesHydrated: boolean;
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

export function isVerdantUsableForConsent(params: {
  readonly streamingConsent: StreamingConsentState;
  readonly engineStatus: EngineStatus;
  readonly modelId: string | null;
  readonly storesHydrated: boolean;
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
  // Accepted streaming consent: ready as soon as the real demand-first runtime is mounted
  // (background warming continues silently after the screen clears).
  const hasRealRuntime = params.modelId !== null && !isInternalBootstrapModel(params.modelId);
  return (
    hasRealRuntime &&
    (params.engineStatus === 'ready' || params.engineStatus === 'generating')
  );
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
    sessionUnlocked,
  } = input;

  // Before consent is decided, show the consent banner inside the app (not the initializing screen).
  if (streamingConsent === 'pending') {
    return {
      visible: false,
      phase: 'hidden',
      headline: 'Initializing VerdantUI',
      statusLine: 'Waiting for confirmation',
      overallPercent: 0,
      steps: [],
      errorMessage: null,
      canRetry: false,
      canSwitchOffline: false,
    };
  }

  const usable = isVerdantUsableForConsent({
    streamingConsent,
    engineStatus,
    modelId,
    storesHydrated,
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
        { id: 'runtime', label: 'Assistant runtime', detail: 'Ready', status: 'done' },
      ],
      errorMessage: null,
      canRetry: false,
      canSwitchOffline: false,
    };
  }

  const clampedProgress = Math.max(0, Math.min(100, Math.round(progressPercent)));
  const hasRealRuntime = modelId !== null && !isInternalBootstrapModel(modelId);
  const hasInitError =
    streamingConsent === 'accepted' &&
    !hasRealRuntime &&
    errorMessage !== null &&
    engineStatus !== 'downloading' &&
    engineStatus !== 'loading';

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

  // Accepted streaming initialization (or error recovery)
  const workspaceStep: InitStepItem = {
    id: 'workspace',
    label: 'Workspace & saved library',
    detail: storesHydrated ? 'Ready' : 'Loading saved spaces…',
    status: storesHydrated ? 'done' : 'active',
  };

  const capabilitiesDone = hasRealRuntime || engineStatus === 'loading';
  const capabilitiesStep: InitStepItem = {
    id: 'capabilities',
    label: 'Browser capabilities',
    detail: hasInitError
      ? 'Connection interrupted'
      : capabilitiesDone
        ? 'Connected'
        : clampedProgress > 0 && engineStatus === 'downloading'
          ? `Connecting (${clampedProgress}%)`
          : 'Connecting…',
    status: hasInitError
      ? 'error'
      : capabilitiesDone
        ? 'done'
        : storesHydrated || engineStatus === 'downloading'
          ? 'active'
          : 'pending',
  };

  const runtimeStep: InitStepItem = {
    id: 'runtime',
    label: 'Assistant startup',
    detail: hasInitError
      ? 'Waiting to retry'
      : hasRealRuntime
        ? 'Ready'
        : engineStatus === 'loading'
          ? 'Starting on your device…'
          : 'Waiting…',
    status: hasInitError
      ? 'pending'
      : hasRealRuntime
        ? 'done'
        : engineStatus === 'loading'
          ? 'active'
          : 'pending',
  };

  // Smooth monotonic progress across all 3 phases:
  // Workspace hydration: 0-15%, Capabilities stream: 15-82%, Runtime init: 82-99%
  let overallPercent = 8;
  if (storesHydrated) {
    overallPercent = 15;
  }
  if (engineStatus === 'downloading') {
    const streamFraction = Math.min(1, clampedProgress / 96);
    overallPercent = Math.max(overallPercent, Math.round(15 + streamFraction * 67));
  } else if (engineStatus === 'loading') {
    const loadFraction = Math.max(0, Math.min(1, (clampedProgress - 96) / 4));
    overallPercent = Math.max(82, Math.round(82 + loadFraction * 16));
  } else if (hasRealRuntime) {
    overallPercent = storesHydrated ? 100 : 96;
  }

  if (hasInitError) {
    return {
      visible: true,
      phase: 'error',
      headline: 'Unable to finish setup',
      statusLine: errorMessage || 'Could not reach streaming capabilities.',
      overallPercent,
      steps: [workspaceStep, capabilitiesStep, runtimeStep],
      errorMessage,
      canRetry: true,
      canSwitchOffline: true,
    };
  }

  return {
    visible: true,
    phase: 'initializing',
    headline: 'Initializing VerdantUI',
    statusLine: formatInitStatusLine(initStatusDetail, engineStatus, clampedProgress),
    overallPercent,
    steps: [workspaceStep, capabilitiesStep, runtimeStep],
    errorMessage: null,
    canRetry: false,
    canSwitchOffline: true,
  };
}
