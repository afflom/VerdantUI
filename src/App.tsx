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
import { Menu, Moon, SlidersHorizontal, Sun } from 'lucide-react';
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
import type { ImportedHarnessSpec } from './engine/harnessConfig';
import { isInternalBootstrapModel } from './engine/modelCatalog';
import { ObserverStreamBus } from './engine/observerStreamBus';
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
}> = ({
  importedHarness,
  sampling,
  samplingCustomized,
  workerClient,
  publishedPrisms,
  prismPool,
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
      ),
    [workerClient, publishedPrisms, prismPool],
  );

  useEffect(() => {
    const context = buildUnifiedVerdantModelContext(
      clientTools,
      importedHarness,
      samplingCustomized
        ? { temperature: sampling.temperature, topP: sampling.topP }
        : undefined,
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
  ]);

  return null;
};

type ActiveWorkspaceView =
  | { readonly kind: 'thread' }
  | { readonly kind: 'prism'; readonly prismId: string; readonly mode: 'studio' | 'app' };

export default function App() {
  const workerClient = useMemo(() => new LlamaVTensorWorkerClient(), []);
  const observerBus = useMemo(() => new ObserverStreamBus(), []);
  const prismStore = useMemo(() => new PrismStore(), []);
  const prismPool = useMemo(
    () =>
      new PrismHostPool(
        observerBus,
        prismStore,
        workerClient.cognitiveObserver.grants,
        workerClient,
      ),
    [observerBus, prismStore, workerClient],
  );
  const modelAdapter = useMemo(
    () =>
      createLlamaVTensorAdapter(workerClient, {
        bus: observerBus,
        getPublishedPrisms: () => prismStore.listPublishedSummaries(),
      }),
    [workerClient, observerBus, prismStore],
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
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [darkMode, setDarkMode] = useState<boolean>(readInitialDarkMode);

  const publishedPrisms = useMemo(
    () => prismStore.listPublishedSummaries(),
    // Recompute whenever `prisms` state updates
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [prismStore, prisms],
  );

  useEffect(() => {
    const unsubStore = prismStore.subscribe(setPrisms);
    void prismStore.hydrate();
    const unsubBus = observerBus.subscribe((ev) => {
      if (ev.endpointKind === 'prism') {
        workerClient.cognitiveObserver.recordSensoryEvent(ev);
      }
    });
    return () => {
      unsubStore();
      unsubBus();
      prismPool.dispose();
    };
  }, [prismStore, observerBus, workerClient, prismPool]);

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
      if (matches) setInspectorOpen(false);
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
      if (inspectorOpen && typeof window !== 'undefined' && window.innerWidth < 1280) {
        setInspectorOpen(false);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [mobileSidebarOpen, inspectorOpen]);

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
    setInspectorOpen((prev) => !prev);
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

  const handleCountTokens = useCallback(
    (prompt: string) => workerClient.countTokens(prompt),
    [workerClient],
  );

  const handleClearCache = useCallback(async () => {
    await workerClient.clearModelCache();
  }, [workerClient]);

  const handleRunConformance = useCallback(async () => {
    await workerClient.runConformanceCheck();
  }, [workerClient]);

  const handleCreatePrism = useCallback(() => {
    void prismStore.create('blank').then((created) => {
      setActiveView({ kind: 'prism', prismId: created.id, mode: 'studio' });
    });
  }, [prismStore]);

  const handleSelectPrism = useCallback((prismId: string, mode: 'studio' | 'app') => {
    setActiveView({ kind: 'prism', prismId, mode });
  }, []);

  const handleOpenPrismApp = useCallback((prismId: string) => {
    setActiveView({ kind: 'prism', prismId, mode: 'app' });
  }, []);

  const handleDeletePrism = useCallback(
    (prismId: string) => {
      void prismStore.delete(prismId).then(() => {
        setActiveView((prev) =>
          prev.kind === 'prism' && prev.prismId === prismId ? { kind: 'thread' } : prev,
        );
      });
    },
    [prismStore],
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
        publishedPrisms={publishedPrisms}
        prismPool={prismPool}
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
                      ? `Model Stream enabled (${engineState.statusDetail}). Click to pause/disable.`
                      : 'Model Stream disabled (remembers setting across reloads). Click to stream local neural weights.'
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
                  <span>Model Stream</span>
                  <span className="font-mono text-[11px] tabular-nums opacity-80">
                    {statusLabel}
                  </span>
                </button>
              );
            })()}

            <button
              type="button"
              onClick={handleToggleInspector}
              aria-pressed={inspectorOpen}
              className={`inline-flex h-9 items-center gap-1.5 rounded-lg px-2.5 sm:px-3 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                inspectorOpen
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
            onOpenInspector={() => setInspectorOpen(true)}
            activeView={activeView}
            onSelectThreadView={() => setActiveView({ kind: 'thread' })}
            prisms={prisms}
            onCreatePrism={handleCreatePrism}
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
                onSpeculativeDraftChange={handleSpeculativeDraftChange}
                onCountTokens={handleCountTokens}
              />
            )}
          </main>

          {inspectorOpen && (
            <Suspense fallback={null}>
              <VTensorInspector
                open={inspectorOpen}
                onClose={() => setInspectorOpen(false)}
                engineState={engineState}
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
      </div>
    </AssistantRuntimeProvider>
  );
}
