/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Suspense, lazy, useCallback, useEffect, useMemo, useState, type FC } from 'react';
import {
  AssistantRuntimeProvider,
  SimpleTextAttachmentAdapter,
  useAui,
  useLocalRuntime,
  useRemoteThreadListRuntime,
} from '@assistant-ui/react';
import { Menu, Moon, Shield, ShieldOff, SlidersHorizontal, Sun } from 'lucide-react';
import { SettingsDrawer } from './components/SettingsDrawer';
import { ThreadSidebar } from './components/ThreadSidebar';
import { ThreadView } from './components/ThreadView';
import { createLlamaVTensorAdapter } from './engine/assistantAdapter';

const VTensorInspector = lazy(() =>
  import('./components/VTensorInspector').then((m) => ({ default: m.VTensorInspector })),
);
const VerdantStudioView = lazy(() =>
  import('./components/VerdantStudioView').then((m) => ({ default: m.VerdantStudioView })),
);
import type { SamplingConfig } from './engine/llamaRuntime';
import {
  buildUnifiedVerdantModelContext,
  createVerdantClientTools,
} from './engine/cognitiveObserver';
import {
  DEFAULT_USER_SPACES,
  type ContextCompartmentId,
} from './engine/unifiedPossessionModel';
import type { ImportedHarnessSpec } from './engine/harnessConfig';
import { isInternalBootstrapModel } from './engine/modelCatalog';
import { ObserverStreamBus } from './engine/observerStreamBus';
import { PrismBuildWorkerClient } from './engine/prismBuildClient';
import { PrismHostPool } from './engine/prismHostPool';
import {
  PrismStore,
  type PublishedPrismSummary,
  type VerdantPrismProject,
} from './engine/prismStudio';
import { createVTensorThreadListAdapter } from './engine/threadStorage';
import { LlamaVTensorWorkerClient, type EngineState } from './engine/workerClient';

const THEME_STORAGE_KEY = 'vtensor-assistant:theme:v1';

function readInitialDarkMode(): boolean {
  if (typeof window === 'undefined' || typeof window.localStorage === 'undefined') {
    return false;
  }
  try {
    return window.localStorage.getItem(THEME_STORAGE_KEY) === 'dark';
  } catch {
    return false;
  }
}

/**
 * Registers the single unified VerdantUI model context (verified client tools, unified system
 * contract, and optional imported custom harness override) directly into `@assistant-ui/react`'s
 * standard `ModelContext` registry (`aui.modelContext.register`).
 */
const VerdantModelContextBridge: FC<{
  importedHarness: ImportedHarnessSpec | null;
  sampling: SamplingConfig;
  samplingCustomized: boolean;
  workerClient: LlamaVTensorWorkerClient;
  publishedPrisms: readonly PublishedPrismSummary[];
  prismPool: PrismHostPool;
  opaqueConstraintsNotice: string;
}> = ({
  importedHarness,
  sampling,
  samplingCustomized,
  workerClient,
  publishedPrisms,
  prismPool,
  opaqueConstraintsNotice,
}) => {
  const aui = useAui();
  const clientTools = useMemo(
    () =>
      createVerdantClientTools(
        () => {
          const s = workerClient.getState();
          return {
            model: s.model,
            telemetry: s.telemetry,
            opfsStorage: s.opfsStorage,
            lastTokensPerSec: s.lastTokensPerSec,
          };
        },
        {
          prisms: publishedPrisms,
          dispatchPrism: (prismId, payload, signal) =>
            prismPool.dispatchPublishedPrism(prismId, payload, { signal }),
        },
        {
          recallTurns: (q) =>
            workerClient.cognitiveContext.recallTurns(workerClient.getActiveThreadId(), q),
        },
        () => workerClient.cognitiveObserver.possession.getActiveCompartmentId(),
      ),
    [workerClient, publishedPrisms, prismPool],
  );

  useEffect(() => {
    workerClient.setActiveToolRegistry(clientTools);
  }, [workerClient, clientTools]);

  useEffect(() => {
    const context = buildUnifiedVerdantModelContext(
      clientTools,
      importedHarness,
      samplingCustomized
        ? { temperature: sampling.temperature, topP: sampling.topP }
        : undefined,
      opaqueConstraintsNotice,
    );
    return aui.modelContext.register({
      getModelContext: () => context,
    });
  }, [
    aui,
    clientTools,
    importedHarness,
    samplingCustomized,
    sampling.temperature,
    sampling.topP,
    opaqueConstraintsNotice,
  ]);

  return null;
};

type ActiveWorkspaceView =
  | { readonly kind: 'thread' }
  | { readonly kind: 'prism'; readonly prismId: string; readonly mode: 'studio' | 'app' };

export default function App() {
  const workerClient = useMemo(() => new LlamaVTensorWorkerClient(), []);
  const observerBus = useMemo(() => new ObserverStreamBus(), []);
  const prismBuildClient = useMemo(() => new PrismBuildWorkerClient(observerBus), [observerBus]);
  const prismStore = useMemo(() => new PrismStore(), []);
  const prismPool = useMemo(
    () =>
      new PrismHostPool(
        observerBus,
        prismStore,
        workerClient.cognitiveObserver.grants,
        workerClient,
        undefined,
        prismBuildClient,
      ),
    [observerBus, prismStore, workerClient, prismBuildClient],
  );
  const modelAdapter = useMemo(
    () =>
      createLlamaVTensorAdapter(workerClient, {
        bus: observerBus,
        getPublishedPrisms: () =>
          workerClient.cognitiveObserver.getVisiblePublishedPrismsForSpace(
            prismStore.listPublishedSummaries(),
          ),
        onOpenPrism: (prismId) => {
          setActiveView({ kind: 'prism', prismId, mode: 'app' });
        },
        resolveOrCreatePrismForToolCall: async (toolName, args, meta) => {
          const { resolveOrCreatePrismForToolCall } = await import('./engine/prismToolResolver');
          return resolveOrCreatePrismForToolCall({
            toolName,
            args,
            latestUserText: meta.latestUserText,
            activeSpace: workerClient.cognitiveObserver.possession.getActiveCompartmentId(),
            store: prismStore,
            pool: prismPool,
            observer: workerClient.cognitiveObserver,
            signal: meta.signal,
          });
        },
      }),
    [workerClient, observerBus, prismStore, prismPool],
  );
  const attachmentAdapter = useMemo(() => new SimpleTextAttachmentAdapter(), []);
  const threadListAdapter = useMemo(() => createVTensorThreadListAdapter(), []);

  const useThreadRuntime = useCallback(
    () =>
      useLocalRuntime(modelAdapter, {
        maxSteps: 5,
        adapters: {
          attachments: attachmentAdapter,
        },
      }),
    [modelAdapter, attachmentAdapter],
  );

  const runtime = useRemoteThreadListRuntime({
    runtimeHook: useThreadRuntime,
    adapter: threadListAdapter,
  });

  const [engineState, setEngineState] = useState<EngineState>(() => workerClient.getState());
  const [prisms, setPrisms] = useState<readonly VerdantPrismProject[]>(() => prismStore.list());
  const [activeView, setActiveView] = useState<ActiveWorkspaceView>({ kind: 'thread' });
  const [prismToast, setPrismToast] = useState<{ title: string; body: string } | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [debugPanelOpen, setDebugPanelOpen] = useState(false);
  const debugMode = useMemo(() => {
    if (typeof window === 'undefined') return false;
    return new URLSearchParams(window.location.search).get('debug') === '1';
  }, []);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [darkMode, setDarkMode] = useState<boolean>(readInitialDarkMode);
  const [activeSpace, setActiveSpace] = useState<ContextCompartmentId>('personal');
  const [sharingPaused, setSharingPaused] = useState(false);

  const publishedPrisms = useMemo(
    () => prismStore.listPublishedSummaries(),
    // Recompute whenever `prisms` state updates
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [prismStore, prisms],
  );

  // Sync owned Prisms, active space, and global pause switch into unified possession model,
  // and filter visible Published Prisms strictly to the active Space (zero cross-space name/detail leaks!)
  const { opaqueConstraintsNotice, visiblePublishedPrisms } = useMemo(() => {
    workerClient.cognitiveObserver.possession.setActiveCompartment(activeSpace);
    workerClient.cognitiveObserver.possession.setHalted(sharingPaused);
    workerClient.cognitiveObserver.syncPublishedPrisms(prisms);
    return {
      opaqueConstraintsNotice:
        workerClient.cognitiveObserver.getOpaqueConstraintsPromptForSpace(activeSpace),
      visiblePublishedPrisms: workerClient.cognitiveObserver.getVisiblePublishedPrismsForSpace(
        publishedPrisms,
        activeSpace,
      ),
    };
  }, [workerClient, activeSpace, sharingPaused, prisms, publishedPrisms]);

  useEffect(() => {
    let toastTimer: ReturnType<typeof setTimeout> | null = null;
    prismPool.setCallbacks({
      onNotify: ({ title, body }) => {
        setPrismToast({ title, body });
        if (toastTimer) clearTimeout(toastTimer);
        toastTimer = setTimeout(() => setPrismToast(null), 3500);
      },
      onOpenPrism: (prismId) => {
        setActiveView({ kind: 'prism', prismId, mode: 'app' });
      },
    });
    const unsubStore = prismStore.subscribe(setPrisms);
    void prismStore.hydrate();
    const unsubBus = observerBus.subscribe((ev) => {
      if (ev.endpointKind === 'prism') {
        workerClient.cognitiveObserver.recordSensoryEvent(ev);
      }
    });
    return () => {
      if (toastTimer) clearTimeout(toastTimer);
      unsubStore();
      unsubBus();
      prismPool.dispose();
      prismBuildClient.dispose();
    };
  }, [prismStore, observerBus, workerClient, prismPool, prismBuildClient]);

  useEffect(() => {
    const unsubscribe = workerClient.subscribe(setEngineState);

    if (!workerClient.getState().model) {
      void workerClient.bootInitialModel().catch(() => {
        // Error state is surfaced via engineState.errorMessage
      });
    }

    return () => {
      unsubscribe();
    };
  }, [workerClient]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const mq = window.matchMedia('(max-width: 1279px)');
    const sync = (matches: boolean) => {
      if (matches) {
        setSettingsOpen(false);
        setDebugPanelOpen(false);
      }
    };
    sync(mq.matches);
    const handler = (e: MediaQueryListEvent) => sync(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (darkMode) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, darkMode ? 'dark' : 'light');
    } catch {
      // Ignore storage error
    }
  }, [darkMode]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      if (mobileSidebarOpen) {
        setMobileSidebarOpen(false);
        return;
      }
      if ((settingsOpen || debugPanelOpen) && typeof window !== 'undefined' && window.innerWidth < 1280) {
        setSettingsOpen(false);
        setDebugPanelOpen(false);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [mobileSidebarOpen, settingsOpen, debugPanelOpen]);

  const handleClearHarness = useCallback(() => {
    workerClient.setImportedHarness(null);
  }, [workerClient]);

  const handleToggleModelStream = useCallback(() => {
    const next = !workerClient.getState().modelStreamEnabled;
    void workerClient.setModelStreamEnabled(next).catch(() => {});
  }, [workerClient]);

  const handleCancelDownload = useCallback(() => {
    void workerClient.setModelStreamEnabled(false).catch(() => {});
  }, [workerClient]);

  const handleRetryLoad = useCallback(() => {
    void workerClient.retryLastLoad().catch(() => {});
  }, [workerClient]);

  const handleDismissError = useCallback(() => {
    workerClient.clearError();
  }, [workerClient]);

  const handleToggleInspector = useCallback(() => {
    setSettingsOpen((prev) => !prev);
  }, []);

  const handleSpeculativeDraftChange = useCallback(
    (
      messages: Parameters<LlamaVTensorWorkerClient['scheduleSpeculativePrefill']>[0],
      draft: string,
      system?: string,
      threadId?: string,
    ) => {
      workerClient.scheduleSpeculativePrefill(messages, draft, system, threadId);
    },
    [workerClient],
  );

  const handlePerceiveImage = useCallback(
    (name: string, width: number, height: number, rgba: Uint8Array) =>
      workerClient.perceiveImage(name, width, height, rgba),
    [workerClient],
  );

  const handleClearPendingImage = useCallback(() => {
    workerClient.clearPendingImageContext();
  }, [workerClient]);

  const handlePerceiveAudio = useCallback(
    (pcm16k: Float32Array) => workerClient.perceiveAudio(pcm16k),
    [workerClient],
  );

  const handleClearPendingAudio = useCallback(() => {
    workerClient.clearPendingAudioContext();
  }, [workerClient]);

  const handleClearCache = useCallback(async () => {
    await workerClient.clearModelCache();
  }, [workerClient]);

  const handleRunConformance = useCallback(async () => {
    await workerClient.runConformanceCheck();
  }, [workerClient]);

  const handleCreatePrism = useCallback(() => {
    void prismStore.createDeclarative('undefined').then(async (created) => {
      if (activeSpace && activeSpace !== 'personal') {
        await prismStore.applyStudioOp(
          created.id,
          { op: 'set-concept-profile', compartmentId: activeSpace },
          { pushRevision: false },
        );
      }
      setActiveView({ kind: 'prism', prismId: created.id, mode: 'studio' });
    });
  }, [prismStore, activeSpace]);

  const handleImportHoloFile = useCallback(
    async (file: File) => {
      try {
        const bytes = new Uint8Array(await file.arrayBuffer());
        if (bytes[0] === 0x7b) {
          const imported = await prismStore.importHolo(new TextDecoder().decode(bytes), false);
          setActiveView({ kind: 'prism', prismId: imported.id, mode: 'studio' });
          return;
        }
        const verified = await prismBuildClient.verifyArchive({ holoBytes: bytes });
        if (!verified.identitiesVerified) {
          setPrismToast({ title: 'Import refused', body: `${file.name} failed BLAKE3 integrity verification.` });
          return;
        }
        const imported = await prismStore.importBinaryHolo(bytes, verified, true);
        setActiveView({ kind: 'prism', prismId: imported.id, mode: 'app' });
        setPrismToast({
          title: `Verified ${verified.name}`,
          body: `${verified.vectorsPassed}/${verified.vectorsTotal} acceptance vectors passed.`,
        });
      } catch (err) {
        setPrismToast({
          title: 'Import failed',
          body: err instanceof Error ? err.message : String(err),
        });
      }
    },
    [prismBuildClient, prismStore],
  );

  const handleSelectPrism = useCallback((prismId: string, mode: 'studio' | 'app') => {
    setActiveView({ kind: 'prism', prismId, mode });
  }, []);

  const handleOpenPrismApp = useCallback((prismId: string) => {
    setActiveView({ kind: 'prism', prismId, mode: 'app' });
  }, []);

  const handleEditPrismStudio = useCallback((prismId: string) => {
    setActiveView({ kind: 'prism', prismId, mode: 'studio' });
  }, []);

  const handleDeletePrism = useCallback(
    (prismId: string) => {
      prismPool.evictPrism(prismId);
      void prismStore.delete(prismId).then(() => {
        setActiveView((prev) =>
          prev.kind === 'prism' && prev.prismId === prismId ? { kind: 'thread' } : prev,
        );
      });
    },
    [prismStore, prismPool],
  );

  const handleEnableModelStream = useCallback(() => {
    void workerClient.setModelStreamEnabled(true).catch(() => {});
  }, [workerClient]);

  const focusedPrism =
    activeView.kind === 'prism'
      ? prisms.find((p) => p.id === activeView.prismId) ?? null
      : null;

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <VerdantModelContextBridge
        importedHarness={engineState.importedHarness}
        sampling={engineState.sampling}
        samplingCustomized={engineState.samplingCustomized}
        workerClient={workerClient}
        publishedPrisms={visiblePublishedPrisms}
        prismPool={prismPool}
        opaqueConstraintsNotice={opaqueConstraintsNotice}
      />
      <div className="flex h-screen w-screen flex-col overflow-hidden bg-[#F8F9FA] text-slate-900 dark:bg-[#0B0F14] dark:text-slate-100">
        {/* Top Bar Contract: Single-line brand wordmark on left, concise workspace actions on right */}
        <header className="flex h-14 shrink-0 items-center justify-between gap-3 border-b border-slate-200 bg-white px-3.5 sm:px-5 dark:border-slate-800 dark:bg-[#0F141C]">
          <div className="flex min-w-0 items-center gap-2.5">
            <button
              type="button"
              onClick={() => setMobileSidebarOpen(true)}
              aria-label="Open conversation threads"
              className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-50 lg:hidden dark:border-slate-800 dark:text-slate-300 dark:hover:bg-slate-800"
            >
              <Menu className="h-4 w-4" />
            </button>
            <span className="font-display text-[15px] font-bold tracking-tight text-slate-950 whitespace-nowrap truncate dark:text-white">
              VerdantUI
            </span>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {/* Plain-language Active Space selector & global Pause Sharing kill switch (DEC-010 / DEC-011) */}
            <div className="hidden md:flex items-center gap-1 rounded-lg border border-slate-200 bg-slate-50/80 p-0.5 dark:border-slate-700 dark:bg-[#141B26]">
              {DEFAULT_USER_SPACES.map((sp) => {
                const isActive = activeSpace === sp.id;
                return (
                  <button
                    key={sp.id}
                    type="button"
                    onClick={() => setActiveSpace(sp.id)}
                    title={`Switch active space to ${sp.label} (keeps private details isolated across spaces)`}
                    className={`rounded-md px-2 py-1 text-[11px] font-medium transition-colors ${
                      isActive
                        ? 'bg-emerald-700 text-white'
                        : 'text-slate-600 hover:text-slate-900 dark:text-slate-300 dark:hover:text-white'
                    }`}
                  >
                    {sp.label}
                  </button>
                );
              })}
            </div>

            <button
              type="button"
              onClick={() => setSharingPaused((p) => !p)}
              aria-pressed={sharingPaused}
              title={
                sharingPaused
                  ? 'External & cross-space sharing is paused. Click to resume privacy-shielded availability.'
                  : 'Privacy Shield active: private details never leave their space. Click to pause all external sharing.'
              }
              className={`hidden sm:inline-flex h-9 items-center gap-1.5 rounded-lg border px-2.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                sharingPaused
                  ? 'border-amber-500/50 bg-amber-50 text-amber-900 hover:bg-amber-100 dark:border-amber-500/40 dark:bg-amber-950/50 dark:text-amber-200'
                  : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-200 dark:hover:bg-slate-800'
              }`}
            >
              {sharingPaused ? <ShieldOff className="h-3.5 w-3.5" /> : <Shield className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />}
              <span>{sharingPaused ? 'Paused' : 'Private'}</span>
            </button>

            {(() => {
              const streamOn = engineState.modelStreamEnabled;
              const isStreaming =
                engineState.status === 'downloading' || engineState.status === 'loading';
              const hasRealModel =
                engineState.model !== null && !isInternalBootstrapModel(engineState.model.id);
              const hasError = streamOn && engineState.errorMessage !== null;
              const statusLabel = !streamOn
                ? 'Off'
                : isStreaming
                  ? `${engineState.progressPercent}%`
                  : hasError
                    ? 'Error'
                    : hasRealModel
                      ? 'Ready'
                      : 'On';
              return (
                <button
                  type="button"
                  role="switch"
                  aria-checked={streamOn}
                  data-testid="model-stream-toggle"
                  title={
                    streamOn
                      ? 'Local AI is active on your device. Click to pause.'
                      : 'Local AI is off. Click to enable private on-device AI.'
                  }
                  onClick={handleToggleModelStream}
                  className={`inline-flex h-9 items-center gap-2 rounded-lg border px-2.5 sm:px-3 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                    streamOn
                      ? 'border-emerald-700/40 bg-emerald-50 text-emerald-950 hover:bg-emerald-100/80 dark:border-emerald-500/40 dark:bg-emerald-950/50 dark:text-emerald-200 dark:hover:bg-emerald-950/80'
                      : 'border-slate-200 bg-white text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-300 dark:hover:bg-slate-800'
                  }`}
                >
                  <span
                    className={`inline-block h-2 w-2 rounded-full ${
                      !streamOn
                        ? 'bg-slate-400 dark:bg-slate-500'
                        : hasError
                          ? 'bg-red-500'
                          : isStreaming
                            ? 'animate-pulse bg-emerald-500'
                            : 'bg-emerald-600 dark:bg-emerald-400'
                    }`}
                  />
                  <span>Local AI</span>
                  <span className="font-mono text-[11px] tabular-nums opacity-80">
                    {statusLabel}
                  </span>
                </button>
              );
            })()}

            <button
              type="button"
              onClick={handleToggleInspector}
              aria-pressed={settingsOpen}
              className={`inline-flex h-9 items-center gap-1.5 rounded-lg px-2.5 sm:px-3 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                settingsOpen
                  ? 'bg-[#0F5132] text-white hover:bg-[#0B3D25]'
                  : 'border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-200 dark:hover:bg-slate-800'
              }`}
            >
              <SlidersHorizontal className="h-3.5 w-3.5" />
              <span>Settings</span>
            </button>

            <button
              type="button"
              onClick={() => setDarkMode((d) => !d)}
              aria-label={darkMode ? 'Switch to light theme' : 'Switch to dark theme'}
              className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-300 dark:hover:bg-slate-800 transition-colors shrink-0"
            >
              {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
          </div>
        </header>

        {/* Main 3-zone workbench */}
        <div className="flex min-h-0 flex-1 overflow-hidden">
          <ThreadSidebar
            engineState={engineState}
            mobileOpen={mobileSidebarOpen}
            onCloseMobile={() => setMobileSidebarOpen(false)}
            onOpenInspector={() => setSettingsOpen(true)}
            activeView={activeView}
            onSelectThreadView={() => setActiveView({ kind: 'thread' })}
            prisms={prisms}
            onCreatePrism={handleCreatePrism}
            onImportHoloFile={(file) => void handleImportHoloFile(file)}
            onSelectPrism={handleSelectPrism}
            onDeletePrism={handleDeletePrism}
          />

          <main className="min-w-0 flex-1 overflow-hidden">
            {activeView.kind === 'prism' && focusedPrism ? (
              <Suspense fallback={null}>
                <VerdantStudioView
                  project={focusedPrism}
                  mode={activeView.mode}
                  onChangeMode={(nextMode) =>
                    setActiveView({ kind: 'prism', prismId: focusedPrism.id, mode: nextMode })
                  }
                  store={prismStore}
                  bus={observerBus}
                  bridgeRuntime={prismPool}
                  buildClient={prismBuildClient}
                  workerClient={workerClient}
                  engineState={engineState}
                  darkMode={darkMode}
                  onEnableModelStream={handleEnableModelStream}
                  onSelectProject={handleSelectPrism}
                />
              </Suspense>
            ) : (
              <ThreadView
                engineState={engineState}
                onCancelDownload={handleCancelDownload}
                onRetryLoad={handleRetryLoad}
                onDismissError={handleDismissError}
                onToggleInspector={handleToggleInspector}
                onOpenPrism={handleOpenPrismApp}
                onEditPrism={handleEditPrismStudio}
                publishedPrisms={visiblePublishedPrisms}
                onSpeculativeDraftChange={handleSpeculativeDraftChange}
                onPerceiveImage={handlePerceiveImage}
                onClearPendingImage={handleClearPendingImage}
                onPerceiveAudio={handlePerceiveAudio}
                onClearPendingAudio={handleClearPendingAudio}
              />
            )}
          </main>

          <SettingsDrawer
            open={settingsOpen && !debugPanelOpen}
            onClose={() => setSettingsOpen(false)}
            darkMode={darkMode}
            onToggleDarkMode={() => setDarkMode((d) => !d)}
            activeSpace={activeSpace}
            onSelectSpace={setActiveSpace}
            sharingPaused={sharingPaused}
            onToggleSharingPaused={() => setSharingPaused((p) => !p)}
            cachedBytes={engineState.opfsStorage?.usageBytes ?? 0}
            onClearCache={handleClearCache}
            debugMode={debugMode}
            onOpenDebugPanel={() => {
              setSettingsOpen(false);
              setDebugPanelOpen(true);
            }}
          />

          {debugMode && debugPanelOpen && (
            <Suspense fallback={null}>
              <VTensorInspector
                open={debugPanelOpen}
                onClose={() => setDebugPanelOpen(false)}
                engineState={engineState}
                publishedPrisms={publishedPrisms}
                observerBus={observerBus}
                onClearHarness={handleClearHarness}
                onClearCache={handleClearCache}
                onUpdateSampling={(patch) => workerClient.updateSampling(patch)}
                onUpdatePromptMode={(mode) => workerClient.updatePromptMode(mode)}
                onUpdatePrecisionMode={(mode) => void workerClient.updatePrecisionMode(mode)}
                onUpdateWeightResidencyPolicy={(policy) =>
                  void workerClient.updateWeightResidencyPolicy(policy)
                }
                onResetSampling={() => workerClient.resetSampling()}
                onRunConformance={handleRunConformance}
              />
            </Suspense>
          )}
        </div>

        {prismToast && (
          <div
            role="status"
            aria-live="polite"
            className="fixed right-4 bottom-4 z-50 max-w-xs rounded-xl border border-slate-200 bg-white px-3.5 py-2.5 text-xs shadow-lg dark:border-slate-700 dark:bg-[#141B26]"
          >
            <div className="font-semibold text-slate-900 dark:text-white">{prismToast.title}</div>
            {prismToast.body && (
              <div className="mt-0.5 text-slate-500 dark:text-slate-400">{prismToast.body}</div>
            )}
          </div>
        )}
      </div>
    </AssistantRuntimeProvider>
  );
}
