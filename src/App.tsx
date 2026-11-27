/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  Suspense,
  lazy,
  memo,
  useCallback,
  useEffect,
  useMemo,
  useState,
  useSyncExternalStore,
  type ComponentProps,
  type FC,
} from 'react';
import {
  AssistantRuntimeProvider,
  SimpleTextAttachmentAdapter,
  useAui,
  useLocalRuntime,
  useRemoteThreadListRuntime,
} from '@assistant-ui/react';
import { Menu, Moon, SlidersHorizontal, Sun } from 'lucide-react';
import { PrismSurfaceShell } from './components/PrismSurfaceShell';
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
const GalleryView = lazy(() =>
  import('./components/GalleryView').then((m) => ({ default: m.GalleryView })),
);
import type { SamplingConfig } from './engine/llamaRuntime';
import {
  buildUnifiedVerdantModelContext,
  createVerdantClientTools,
} from './engine/cognitiveObserver';
import {
  GALLERY_LABEL_BY_TYPE,
  type ContentArtifactRecord,
  type GalleryMediaType,
} from './engine/contentGalleryStore';
import type { ImportedHarnessSpec } from './engine/harnessConfig';
import { isInternalBootstrapModel } from './engine/modelCatalog';
import {
  galleryPrismId,
  VERDANT_HOME_PRISM_ID,
  VERDANT_SETTINGS_PRISM_ID,
  type PrismSurfaceMode,
} from './engine/observerUiSynthesizer';
import type { PrismHostPool } from './engine/prismHostPool';
import type {
  PublishedPrismSummary,
  VerdantPrismProject,
} from './engine/prismStudio';
import type { StudioChatBridge } from './engine/studioChatBridge';
import { createVTensorThreadListAdapter } from './engine/threadStorage';
import type { UnifiedExpertRegistry } from './engine/unifiedModelBasis';
import { getVerdantRootPrismRuntime } from './engine/verdantRootPrism';
import type { LlamaVTensorWorkerClient } from './engine/workerClient';

const LocalAiStatusPill: FC<{
  readonly workerClient: LlamaVTensorWorkerClient;
  readonly onToggle: () => void;
}> = memo(({ workerClient, onToggle }) => {
  const summary = useSyncExternalStore(
    useCallback((cb) => workerClient.subscribeHeaderSummary(cb), [workerClient]),
    useCallback(() => workerClient.getHeaderSummary(), [workerClient]),
  );
  const streamOn = summary.modelStreamEnabled;
  const isStreaming = summary.status === 'downloading' || summary.status === 'loading';
  const hasRealModel = summary.modelId !== null && !isInternalBootstrapModel(summary.modelId);
  const hasError = streamOn && summary.errorMessage !== null;
  const statusLabel = !streamOn
    ? 'Off'
    : isStreaming
      ? `${summary.progressPercent}%`
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
      onClick={onToggle}
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
      <span className="font-mono text-[11px] tabular-nums opacity-80">{statusLabel}</span>
    </button>
  );
});

const SubscribedThreadSidebar: FC<
  Omit<ComponentProps<typeof ThreadSidebar>, 'engineState' | 'headerSummary'> & {
    readonly workerClient: LlamaVTensorWorkerClient;
  }
> = memo(({ workerClient, ...rest }) => {
  const headerSummary = useSyncExternalStore(
    useCallback((cb) => workerClient.subscribeHeaderSummary(cb), [workerClient]),
    useCallback(() => workerClient.getHeaderSummary(), [workerClient]),
  );
  return <ThreadSidebar {...rest} headerSummary={headerSummary} />;
});

const SubscribedThreadView: FC<
  Omit<ComponentProps<typeof ThreadView>, 'engineState'> & {
    readonly workerClient: LlamaVTensorWorkerClient;
  }
> = ({ workerClient, ...rest }) => {
  const engineState = useSyncExternalStore(
    useCallback((cb) => workerClient.subscribeStore(cb), [workerClient]),
    useCallback(() => workerClient.getState(), [workerClient]),
  );
  return <ThreadView {...rest} engineState={engineState} />;
};

const SubscribedStudioView: FC<
  Omit<ComponentProps<typeof VerdantStudioView>, 'engineState'> & {
    readonly workerClient: LlamaVTensorWorkerClient;
  }
> = ({ workerClient, ...rest }) => {
  const engineState = useSyncExternalStore(
    useCallback((cb) => workerClient.subscribeStore(cb), [workerClient]),
    useCallback(() => workerClient.getState(), [workerClient]),
  );
  return <VerdantStudioView {...rest} workerClient={workerClient} engineState={engineState} />;
};

const SubscribedVTensorInspector: FC<
  Omit<ComponentProps<typeof VTensorInspector>, 'engineState'> & {
    readonly workerClient: LlamaVTensorWorkerClient;
  }
> = ({ workerClient, ...rest }) => {
  const engineState = useSyncExternalStore(
    useCallback((cb) => workerClient.subscribeStore(cb), [workerClient]),
    useCallback(() => workerClient.getState(), [workerClient]),
  );
  return <VTensorInspector {...rest} engineState={engineState} />;
};

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
  studioChatBridge: () => Promise<StudioChatBridge>;
  expertRegistry: UnifiedExpertRegistry;
  opaqueConstraintsNotice: string;
}> = ({
  importedHarness,
  sampling,
  samplingCustomized,
  workerClient,
  publishedPrisms,
  prismPool,
  studioChatBridge,
  expertRegistry,
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
        studioChatBridge,
        expertRegistry,
      ),
    [workerClient, publishedPrisms, prismPool, studioChatBridge, expertRegistry],
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
  | { readonly kind: 'prism'; readonly prismId: string; readonly mode: 'studio' | 'app' }
  | { readonly kind: 'gallery'; readonly mediaType: GalleryMediaType }
  | { readonly kind: 'settings' };

export default function App() {
  // Single live Root Verdant Prism Runtime (`verdant-ui-root`) rendered by the Observer
  // and maintained by Verdant Studio — decouples ALL inference & synthesis from view mounting.
  const rootPrism = useMemo(() => getVerdantRootPrismRuntime(), []);
  const {
    workerClient,
    observerBus,
    prismStore,
    galleryStore,
    expertRegistry,
    prismPool,
    systemPartPrisms,
  } = rootPrism;
  const getPrismBuildClient = useCallback(
    () => rootPrism.getPrismBuildClient(),
    [rootPrism],
  );
  const getStudioChatBridge = useCallback(
    () => rootPrism.getStudioChatBridge(),
    [rootPrism],
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
          const [{ resolveOrCreatePrismForToolCall }, bridge] = await Promise.all([
            import('./engine/prismToolResolver'),
            getStudioChatBridge(),
          ]);
          return resolveOrCreatePrismForToolCall({
            toolName,
            args,
            latestUserText: meta.latestUserText,
            activeSpace: workerClient.cognitiveObserver.possession.getActiveCompartmentId(),
            store: prismStore,
            pool: prismPool,
            observer: workerClient.cognitiveObserver,
            galleries: galleryStore,
            registry: expertRegistry,
            studioChatBridge: bridge,
            signal: meta.signal,
          });
        },
      }),
    [workerClient, observerBus, prismStore, prismPool, galleryStore, expertRegistry, getStudioChatBridge],
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

  const appShellSummary = useSyncExternalStore(
    useCallback((cb) => workerClient.subscribeAppShellSummary(cb), [workerClient]),
    useCallback(() => workerClient.getAppShellSummary(), [workerClient]),
  );
  const [prisms, setPrisms] = useState<readonly VerdantPrismProject[]>(() => prismStore.list());
  const [galleryArtifacts, setGalleryArtifacts] = useState<readonly ContentArtifactRecord[]>(() =>
    galleryStore.listAll(),
  );
  const [activeView, setActiveView] = useState<ActiveWorkspaceView>({ kind: 'thread' });
  const [surfaceMode, setSurfaceMode] = useState<PrismSurfaceMode>('preview');
  const [userTypingInSurface, setUserTypingInSurface] = useState(false);
  const [viewportSize, setViewportSize] = useState<{ width: number; height: number }>(() => ({
    width: typeof window !== 'undefined' ? window.innerWidth : 1280,
    height: typeof window !== 'undefined' ? window.innerHeight : 800,
  }));
  const [prismToast, setPrismToast] = useState<{ title: string; body: string } | null>(null);
  const [debugPanelOpen, setDebugPanelOpen] = useState(false);
  const debugMode = useMemo(() => {
    if (typeof window === 'undefined') return false;
    return new URLSearchParams(window.location.search).get('debug') === '1';
  }, []);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [darkMode, setDarkMode] = useState<boolean>(readInitialDarkMode);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const onResize = () =>
      setViewportSize({ width: window.innerWidth, height: window.innerHeight });
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  // Reset surface mode back to Preview when switching between Prisms
  useEffect(() => {
    setSurfaceMode('preview');
    setUserTypingInSurface(false);
  }, [activeView.kind, activeView.kind === 'prism' ? activeView.prismId : activeView.kind === 'gallery' ? activeView.mediaType : '']);

  const publishedPrisms = useMemo(
    () => prismStore.listPublishedSummaries(),
    // Recompute whenever `prisms` state updates
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [prismStore, prisms],
  );

  // Sync owned Prisms into the unified possession model, expert registry, and global gallery index
  // cleanly in an effect (NEVER mutate/emit stores synchronously during render!).
  useEffect(() => {
    workerClient.cognitiveObserver.syncPublishedPrisms(prisms);
    expertRegistry.syncPrismsAsExperts(prisms);
    galleryStore.syncFromPrisms(prisms);
  }, [workerClient, expertRegistry, galleryStore, prisms]);

  // Pure derived workspace views (zero store mutations during render)
  const { opaqueConstraintsNotice, visiblePublishedPrisms, spacePrisms, galleryCounts } = useMemo(() => {
    const activeSpace = workerClient.cognitiveObserver.possession.getActiveCompartmentId();
    const spacesOnly = prisms.filter(
      (p) => (p.declaration?.view?.proceduralOrgan?.mediaType ?? p.declaration?.concept.mediaType) === 'space',
    );
    const counts: Record<GalleryMediaType, number> = {
      picture: 0,
      video: 0,
      audio: 0,
      'written-work': 0,
      'interactive-game': 0,
      'research-program': 0,
      'tool-organ': 0,
    };
    for (const rec of galleryArtifacts) {
      counts[rec.mediaType] += 1;
    }
    return {
      opaqueConstraintsNotice:
        workerClient.cognitiveObserver.getOpaqueConstraintsPromptForSpace(activeSpace),
      visiblePublishedPrisms: workerClient.cognitiveObserver.getVisiblePublishedPrismsForSpace(
        publishedPrisms,
        activeSpace,
      ),
      spacePrisms: spacesOnly,
      galleryCounts: counts,
    };
  }, [workerClient, prisms, publishedPrisms, galleryArtifacts]);

  // Couple Living Training Basis user facets into VerdantCognitiveObserver active inference
  useEffect(() => {
    workerClient.cognitiveObserver.setUserBasisFacetsProvider(() =>
      expertRegistry.basis.getUserFacetsQ10(),
    );
    return () => {
      workerClient.cognitiveObserver.setUserBasisFacetsProvider(null);
    };
  }, [workerClient, expertRegistry]);

  // Register downloaded/resident neural weight experts into the unified VerdantUI model
  useEffect(() => {
    if (appShellSummary.model && !isInternalBootstrapModel(appShellSummary.model.id)) {
      const storedEntry = appShellSummary.opfsStorage?.storedModels.find(
        (m) => m.modelId === appShellSummary.model!.id,
      );
      expertRegistry.registerNeuralWeightExpert({
        modality: 'language',
        modelId: appShellSummary.model.id,
        label: appShellSummary.model.displayName,
        residency: 'hot-vtensor',
        residentBytes: storedEntry?.byteLength ?? 0,
      });
    }
    for (const stored of appShellSummary.opfsStorage?.storedModels ?? []) {
      const mod = /whisper/i.test(stored.modelId)
        ? 'speech'
        : /vision|mmproj|siglip|vlm/i.test(stored.modelId)
          ? 'vision'
          : /sdxs|taesd/i.test(stored.modelId)
            ? 'image-neural'
            : 'language';
      expertRegistry.registerNeuralWeightExpert({
        modality: mod,
        modelId: stored.modelId,
        label: stored.modelId,
        residency: appShellSummary.model?.id === stored.modelId ? 'hot-vtensor' : 'warm-opfs',
        residentBytes: stored.byteLength,
      });
    }
  }, [appShellSummary.model, appShellSummary.opfsStorage, expertRegistry]);

  useEffect(() => {
    let toastTimer: ReturnType<typeof setTimeout> | null = null;
    rootPrism.setNavigationCallbacks({
      onOpenPrism: (prismId) => setActiveView({ kind: 'prism', prismId, mode: 'app' }),
      onOpenGallery: (mediaType) => setActiveView({ kind: 'gallery', mediaType }),
    });
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
    const unsubGallery = galleryStore.subscribe(setGalleryArtifacts);
    void Promise.all([prismStore.hydrate(), galleryStore.hydrate(), expertRegistry.basis.hydrate()]).then(() => {
      galleryStore.syncFromPrisms(prismStore.list());
      expertRegistry.syncPrismsAsExperts(prismStore.list());
    });
    const unsubBus = observerBus.subscribe((ev) => {
      if (ev.endpointKind === 'prism') {
        workerClient.cognitiveObserver.recordSensoryEvent(ev);
      }
    });
    return () => {
      if (toastTimer) clearTimeout(toastTimer);
      unsubStore();
      unsubGallery();
      unsubBus();
    };
  }, [rootPrism, prismStore, galleryStore, expertRegistry, observerBus, workerClient, prismPool]);

  useEffect(() => {
    workerClient.cognitiveObserver.setModalityResidencyProvider(() => {
      const st = workerClient.getState();
      const loadedModel = st.model !== null && !isInternalBootstrapModel(st.model.id);
      const mm = st.multimodalTelemetry;
      const activeMods = mm?.activeModalities ?? [];
      return {
        language: loadedModel,
        image: activeMods.includes('image-synthesis') || st.cachedModelIds.includes('sdxs-512-q8-taesd'),
        vision: activeMods.includes('vision') || st.cachedModelIds.includes('smolvlm2-256m-mmproj'),
        audio: activeMods.includes('audio') || st.cachedModelIds.includes('whisper-tiny-en-q8'),
        basis: expertRegistry.basis.snapshot().adaptationSteps > 0,
      };
    });
    return () => {
      workerClient.cognitiveObserver.setModalityResidencyProvider(null);
    };
  }, [workerClient, expertRegistry]);

  useEffect(() => {
    if (!workerClient.getState().model) {
      void workerClient.bootInitialModel().catch(() => {
        // Error state is surfaced via scoped header/thread subscriptions
      });
    }
  }, [workerClient]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const mq = window.matchMedia('(max-width: 1279px)');
    const sync = (matches: boolean) => {
      if (matches) {
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
      if (debugPanelOpen && typeof window !== 'undefined' && window.innerWidth < 1280) {
        setDebugPanelOpen(false);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [mobileSidebarOpen, debugPanelOpen]);

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
    setActiveView((prev) => (prev.kind === 'settings' ? { kind: 'thread' } : { kind: 'settings' }));
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

  const handleCreateSpace = useCallback(() => {
    void getStudioChatBridge()
      .then((bridge) =>
        bridge.manageSpacesAndPrisms({
          action: 'create',
          name: `Space ${spacePrisms.length + 1}`,
        }),
      )
      .then((res) => {
        setActiveView({ kind: 'prism', prismId: res.prismId, mode: 'studio' });
      });
  }, [getStudioChatBridge, spacePrisms.length]);

  const handleCreateInGallery = useCallback(
    (mediaType: GalleryMediaType) => {
      void getStudioChatBridge()
        .then((bridge) =>
          bridge.createArtifactFromPrompt({
            prompt: `Create a new ${GALLERY_LABEL_BY_TYPE[mediaType].toLowerCase()}`,
            forcedMediaType: mediaType,
          }),
        )
        .then((res) => {
          setActiveView({ kind: 'prism', prismId: res.prismId, mode: 'studio' });
        });
    },
    [getStudioChatBridge],
  );

  const handleCreateFromGalleryPrompt = useCallback(
    async (mediaType: GalleryMediaType, prompt: string): Promise<string> => {
      const bridge = await getStudioChatBridge();
      const res = await bridge.createArtifactFromPrompt({
        prompt,
        forcedMediaType: mediaType,
      });
      setActiveView({ kind: 'prism', prismId: res.prismId, mode: 'studio' });
      return `Created "${res.prismName}" in ${GALLERY_LABEL_BY_TYPE[mediaType]}.`;
    },
    [getStudioChatBridge],
  );

  const activePartId =
    activeView.kind === 'prism'
      ? activeView.prismId
      : activeView.kind === 'gallery'
        ? galleryPrismId(activeView.mediaType)
        : activeView.kind === 'settings'
          ? VERDANT_SETTINGS_PRISM_ID
          : VERDANT_HOME_PRISM_ID;

  // Shift focus pointer and salience priority inside the unified root runtime without aborting sibling parts
  useEffect(() => {
    rootPrism.setFocusPart(activePartId, userTypingInSurface);
  }, [rootPrism, activePartId, userTypingInSurface]);

  const busyBackgroundParts = useSyncExternalStore(
    useCallback((cb) => rootPrism.subscribeActiveTaskIds(cb), [rootPrism]),
    useCallback(() => rootPrism.getActiveBusyPartIds(), [rootPrism]),
  );

  const busyMediaTypes = useMemo(() => {
    const set = new Set<GalleryMediaType>();
    for (const partId of busyBackgroundParts) {
      const rec = galleryStore.get(partId);
      if (rec) {
        set.add(rec.mediaType);
        continue;
      }
      const p = prisms.find((item) => item.id === partId);
      const mt = p?.declaration?.view?.proceduralOrgan?.mediaType ?? p?.declaration?.concept.mediaType;
      if (mt && mt !== 'space') {
        set.add(mt);
      }
    }
    return set;
  }, [busyBackgroundParts, galleryStore, prisms]);

  const handleSelectSystemPrism = useCallback(
    (prismId: string) => {
      if (prismId === VERDANT_HOME_PRISM_ID) {
        setActiveView({ kind: 'thread' });
        return;
      }
      if (prismId === VERDANT_SETTINGS_PRISM_ID) {
        setActiveView({ kind: 'settings' });
        return;
      }
      if (prismId.startsWith('verdant-part-gallery-')) {
        const mt = prismId.slice('verdant-part-gallery-'.length) as GalleryMediaType;
        setActiveView({ kind: 'gallery', mediaType: mt });
        return;
      }
      const existing = prismStore.get(prismId);
      if (existing) {
        setActiveView({
          kind: 'prism',
          prismId: existing.id,
          mode: existing.published ? 'app' : 'studio',
        });
      }
    },
    [prismStore],
  );

  const handleAssignToSpace = useCallback(
    (artifactId: string, spaceId: string) => {
      galleryStore.assignToSpace(artifactId, spaceId);
      void import('./engine/studioChatBridge').then(({ syncSpacePrismView }) =>
        syncSpacePrismView(spaceId, {
          store: prismStore,
          galleries: galleryStore,
          observer: workerClient.cognitiveObserver,
          registry: expertRegistry,
        }),
      );
    },
    [galleryStore, prismStore, workerClient, expertRegistry],
  );

  const handleImportHoloFile = useCallback(
    async (file: File) => {
      try {
        const bytes = new Uint8Array(await file.arrayBuffer());
        if (bytes[0] === 0x7b) {
          const imported = await prismStore.importHolo(new TextDecoder().decode(bytes), false);
          setActiveView({ kind: 'prism', prismId: imported.id, mode: 'studio' });
          return;
        }
        const buildClient = await getPrismBuildClient();
        const verified = await buildClient.verifyArchive({ holoBytes: bytes });
        if (!verified.identitiesVerified) {
          setPrismToast({ title: 'Import refused', body: `${file.name} failed integrity verification.` });
          return;
        }
        const imported = await prismStore.importBinaryHolo(bytes, verified, true);
        setActiveView({ kind: 'prism', prismId: imported.id, mode: 'app' });
        setPrismToast({
          title: `Verified ${verified.name}`,
          body: `${verified.vectorsPassed}/${verified.vectorsTotal} verification checks passed.`,
        });
      } catch (err) {
        setPrismToast({
          title: 'Import failed',
          body: err instanceof Error ? err.message : String(err),
        });
      }
    },
    [getPrismBuildClient, prismStore],
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

  const focusedPrism =
    activeView.kind === 'prism'
      ? prisms.find((p) => p.id === activeView.prismId) ?? null
      : null;

  // Procedurally synthesize the Root Prism & active part via 2-tier cached layout projection
  const observerFrame = useMemo(
    () =>
      rootPrism.getObserverFrame({
        activeView,
        focusedPrism,
        prisms,
        surfaceMode,
        viewportWidthPx: viewportSize.width,
        viewportHeightPx: viewportSize.height,
        darkMode,
        userTyping: userTypingInSurface,
      }),
    [
      rootPrism,
      activeView,
      focusedPrism,
      prisms,
      surfaceMode,
      viewportSize.width,
      viewportSize.height,
      darkMode,
      userTypingInSurface,
    ],
  );

  const settingsActive = activeView.kind === 'settings';

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <VerdantModelContextBridge
        importedHarness={appShellSummary.importedHarness}
        sampling={appShellSummary.sampling}
        samplingCustomized={appShellSummary.samplingCustomized}
        workerClient={workerClient}
        publishedPrisms={visiblePublishedPrisms}
        prismPool={prismPool}
        studioChatBridge={getStudioChatBridge}
        expertRegistry={expertRegistry}
        opaqueConstraintsNotice={opaqueConstraintsNotice}
      />
      <div
        data-root-prism-id={observerFrame.rootPrism.id}
        data-synthesis-digest={observerFrame.synthesisDigest.slice(0, 12)}
        style={observerFrame.directive.tokens as React.CSSProperties}
        className="flex h-screen w-screen flex-col overflow-hidden bg-[#F8F9FA] text-slate-900 dark:bg-[#0B0F14] dark:text-slate-100"
      >
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
            <button
              type="button"
              onClick={() => setActiveView({ kind: 'thread' })}
              className="font-display text-[15px] font-bold tracking-tight text-slate-950 whitespace-nowrap truncate hover:opacity-80 dark:text-white"
            >
              VerdantUI
            </button>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <LocalAiStatusPill
              workerClient={workerClient}
              onToggle={handleToggleModelStream}
            />

            <button
              type="button"
              onClick={handleToggleInspector}
              aria-pressed={settingsActive}
              className={`inline-flex h-9 items-center gap-1.5 rounded-lg px-2.5 sm:px-3 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                settingsActive
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

        {/* Main 3-zone workbench: every zone is a procedurally synthesized Verdant Prism */}
        <div className="flex min-h-0 flex-1 overflow-hidden">
          <SubscribedThreadSidebar
            workerClient={workerClient}
            busyPartIds={busyBackgroundParts}
            busyMediaTypes={busyMediaTypes}
            mobileOpen={mobileSidebarOpen}
            onCloseMobile={() => setMobileSidebarOpen(false)}
            onOpenInspector={() => setActiveView({ kind: 'settings' })}
            activeView={activeView}
            onSelectThreadView={() => setActiveView({ kind: 'thread' })}
            spaces={spacePrisms}
            galleryCounts={galleryCounts}
            onCreateSpace={handleCreateSpace}
            onSelectGallery={(mediaType) => setActiveView({ kind: 'gallery', mediaType })}
            onImportHoloFile={(file) => void handleImportHoloFile(file)}
            onSelectPrism={handleSelectPrism}
            onDeletePrism={handleDeletePrism}
          />

          <main
            className="min-w-0 flex-1 overflow-hidden"
            data-active-part-id={activePartId}
            data-busy-parts={busyBackgroundParts.length}
          >
            {/* Keep Home Thread mounted persistently so navigating to Galleries, Settings, or Prisms NEVER interrupts active streaming */}
            <div
              className={activeView.kind === 'thread' ? 'h-full w-full' : 'hidden'}
              aria-hidden={activeView.kind !== 'thread'}
            >
              <PrismSurfaceShell
                frame={observerFrame}
                mode={activeView.kind === 'thread' ? surfaceMode : 'preview'}
                onChangeMode={setSurfaceMode}
                onSelectChildPrism={handleSelectSystemPrism}
                previewContent={
                  <SubscribedThreadView
                    workerClient={workerClient}
                    onCancelDownload={handleCancelDownload}
                    onRetryLoad={handleRetryLoad}
                    onDismissError={handleDismissError}
                    onToggleInspector={handleToggleInspector}
                    onOpenPrism={handleOpenPrismApp}
                    onEditPrism={handleEditPrismStudio}
                    getPrismById={(id) => prismStore.get(id)}
                    spaces={spacePrisms}
                    onAssignToSpace={handleAssignToSpace}
                    observerBus={observerBus}
                    bridgeRuntime={prismPool}
                    darkMode={darkMode}
                    publishedPrisms={visiblePublishedPrisms}
                    onSpeculativeDraftChange={handleSpeculativeDraftChange}
                    onPerceiveImage={handlePerceiveImage}
                    onClearPendingImage={handleClearPendingImage}
                    onPerceiveAudio={handlePerceiveAudio}
                    onClearPendingAudio={handleClearPendingAudio}
                  />
                }
              />
            </div>

            {activeView.kind === 'settings' && (
              <SettingsDrawer
                frame={observerFrame}
                surfaceMode={surfaceMode}
                onChangeSurfaceMode={setSurfaceMode}
                onSelectSystemPrism={handleSelectSystemPrism}
                onTypingChange={setUserTypingInSurface}
                darkMode={darkMode}
                onToggleDarkMode={() => setDarkMode((d) => !d)}
                onSetDarkMode={setDarkMode}
                cachedBytes={appShellSummary.opfsStorage?.usageBytes ?? 0}
                onClearCache={handleClearCache}
                debugMode={debugMode}
                onOpenDebugPanel={() => setDebugPanelOpen(true)}
              />
            )}

            {activeView.kind === 'gallery' && (
              <Suspense fallback={null}>
                <GalleryView
                  mediaType={activeView.mediaType}
                  frame={observerFrame}
                  surfaceMode={surfaceMode}
                  onChangeSurfaceMode={setSurfaceMode}
                  onSelectSystemPrism={handleSelectSystemPrism}
                  onTypingChange={setUserTypingInSurface}
                  items={galleryStore.listByGallery(activeView.mediaType)}
                  categories={galleryStore.listCategoriesForGallery(activeView.mediaType)}
                  spaces={spacePrisms}
                  onCreateInGallery={handleCreateInGallery}
                  onCreateFromPrompt={handleCreateFromGalleryPrompt}
                  onOpenArtifact={(id, mode) => setActiveView({ kind: 'prism', prismId: id, mode })}
                  onDeleteArtifact={handleDeletePrism}
                  onAssignToSpace={handleAssignToSpace}
                  onOpenSpace={(spaceId) => setActiveView({ kind: 'prism', prismId: spaceId, mode: 'app' })}
                />
              </Suspense>
            )}

            {activeView.kind === 'prism' && focusedPrism && (
              <Suspense fallback={null}>
                <SubscribedStudioView
                  project={focusedPrism}
                  mode={activeView.mode}
                  frame={observerFrame}
                  surfaceMode={surfaceMode}
                  onChangeSurfaceMode={setSurfaceMode}
                  onChangeMode={(nextMode) =>
                    setActiveView({ kind: 'prism', prismId: focusedPrism.id, mode: nextMode })
                  }
                  onSelectSystemPrism={handleSelectSystemPrism}
                  onTypingChange={setUserTypingInSurface}
                  store={prismStore}
                  bus={observerBus}
                  bridgeRuntime={prismPool}
                  getBuildClient={getPrismBuildClient}
                  workerClient={workerClient}
                  darkMode={darkMode}
                  getStudioChatBridge={getStudioChatBridge}
                  rootPrism={rootPrism}
                  onSelectProject={handleSelectPrism}
                />
              </Suspense>
            )}
          </main>

          {debugMode && debugPanelOpen && (
            <Suspense fallback={null}>
              <SubscribedVTensorInspector
                open={debugPanelOpen}
                onClose={() => setDebugPanelOpen(false)}
                workerClient={workerClient}
                publishedPrisms={publishedPrisms}
                unifiedModelTelemetry={expertRegistry.telemetry()}
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
