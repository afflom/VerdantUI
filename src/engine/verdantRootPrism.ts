/**
 * Single Live Root Verdant Prism Runtime (`src/engine/verdantRootPrism.ts`)
 *
 * Implements VerdantUI itself as ONE live Verdant Prism runtime instance (`verdant-ui-root`),
 * rendered by the Observer and maintained by Verdant Studio:
 * - Owns the unified substrate (`workerClient`, `observerBus`, `prismStore`, `galleryStore`,
 *   `expertRegistry`, `prismPool`, lazy `PrismBuildWorkerClient`, lazy `StudioChatBridge`).
 * - Decouples ALL inference, Studio synthesis, Publish sealing, and `Lod0 -> Lod3` reconciliation
 *   tasks from React view mounting so navigating between Home, Galleries, Settings, Spaces, and
 *   Prisms NEVER interrupts, aborts, or resets inference running in another part of VerdantUI.
 * - Uses `verdant-engine` `kernel10` neuromorphic salience (`stepNeuromorphicSalienceRing`) to
 *   order bounded concurrent background reconciliations (`MAX_CONCURRENT_RECONCILIATIONS = 2`)
 *   with monotonic per-part epoch guards and bounded stage watchdogs.
 * - Uses 2-tier Observer Frame memoization + fine-grained external store subscriptions so token
 *   decode and background LOD ticks cause ZERO full-tree React re-render storms and ZERO
 *   redundant Tropical UI layout solves.
 */

import { formatMessagesForLlama, stripReasoningTags } from './assistantAdapter';
import {
  ContentGalleryStore,
  type ContentArtifactRecord,
  type GalleryMediaType,
} from './contentGalleryStore';
import { ContentStreamCatalogStore } from './contentStreamCatalog';
import { isInternalBootstrapModel } from './modelCatalog';
import type {
  ObserverChronology,
  TemporalCoherenceSnapshot,
  TimelinessState,
} from './observerChronology';
import { ObserverStreamBus } from './observerStreamBus';
import {
  buildCanonicalWorkspaceUiTree,
  buildSystemPartPrisms,
  descriptorForUserPrism,
  galleryPrismId,
  solveAndVerifyUiTree,
  synthesizeObserverPrismFrame,
  VERDANT_HOME_PRISM_ID,
  VERDANT_SETTINGS_PRISM_ID,
  type PrismSurfaceMode,
  type SolvedUiLayout,
  type SynthesizedPrismFrame,
  type VerdantSystemPrismDescriptor,
} from './observerUiSynthesizer';
import type { PrismBuildWorkerClient } from './prismBuildClient';
import { PrismHostPool } from './prismHostPool';
import type { AudioWeights, PictureWeights } from './prismMediaOrgans';
import { PrismStore, type VerdantPrismProject } from './prismStudio';
import type { StudioChatBridge } from './studioChatBridge';
import {
  UnifiedExpertRegistry,
  type SemanticRevisionRecord,
} from './unifiedModelBasis';
import {
  canonicalizePartCommandMonoid,
  stepNeuromorphicSalienceRing,
  type PartCommandOp,
} from './verdantKernels';
import type { PrismLodLabel, PrismLodRank } from './verdantStudioApi';
import { LlamaVTensorWorkerClient, type EngineState } from './workerClient';

export type PartTaskStatus =
  | 'idle'
  | 'contextualizing'
  | 'synthesizing'
  | 'qa-reconciling'
  | 'publishing'
  | 'ready'
  | 'error';

export interface PartRuntimeTask {
  readonly partId: string;
  readonly status: PartTaskStatus;
  readonly busy: boolean;
  readonly lodRank: PrismLodRank;
  readonly lodLabel: PrismLodLabel;
  readonly qaScoreQ10: number;
  readonly semanticVersion: number;
  readonly epoch: number;
  readonly label: string;
  readonly banner: { readonly ok: boolean; readonly text: string } | null;
  readonly updatedAt: number;
  readonly startedAtWallMs?: number;
  readonly elapsedMs?: number;
  readonly timelinessQ10?: number;
  readonly timelinessState?: TimelinessState;
}

export interface RootFocusSnapshot {
  readonly activePartId: string;
  readonly attendedSlot: number;
  readonly foveatedLodRank: PrismLodRank;
  readonly salienceByPart: Readonly<Record<string, number>>;
}

interface QueuedReconciliationJob {
  readonly prismId: string;
  readonly prompt: string;
  readonly initialRevision: SemanticRevisionRecord;
  readonly startRank: PrismLodRank;
  readonly maxRank: PrismLodRank;
  readonly epoch: number;
  readonly controller: AbortController;
  readonly resolve: () => void;
}

const MAX_CONCURRENT_RECONCILIATIONS = 2;
const LOD_LABELS: readonly PrismLodLabel[] = [
  'Lod0CoarseProxy',
  'Lod1MediumDetail',
  'Lod2HighDetail',
  'Lod3LosslessSource',
];
const LOD_USER_STATUS: readonly string[] = [
  'Creating draft…',
  'Refining details…',
  'Checking quality…',
  'Finishing full resolution…',
];

const IDLE_PART_TASK: PartRuntimeTask = Object.freeze({
  partId: '',
  status: 'idle',
  busy: false,
  lodRank: 3,
  lodLabel: 'Lod3LosslessSource',
  qaScoreQ10: 0,
  semanticVersion: 0,
  epoch: 0,
  label: '',
  banner: null,
  updatedAt: 0,
});

export interface VerdantRootPrismOptions {
  readonly workerClient?: LlamaVTensorWorkerClient;
  readonly observerBus?: ObserverStreamBus;
  readonly prismStore?: PrismStore;
  readonly galleryStore?: ContentGalleryStore;
  readonly contentStreams?: ContentStreamCatalogStore;
  readonly expertRegistry?: UnifiedExpertRegistry;
}

export class VerdantRootPrismRuntime {
  readonly workerClient: LlamaVTensorWorkerClient;
  readonly observerBus: ObserverStreamBus;
  readonly prismStore: PrismStore;
  readonly galleryStore: ContentGalleryStore;
  readonly contentStreams: ContentStreamCatalogStore;
  readonly expertRegistry: UnifiedExpertRegistry;
  readonly prismPool: PrismHostPool;
  readonly systemPartPrisms: readonly VerdantSystemPrismDescriptor[];

  private buildClientInstance: PrismBuildWorkerClient | null = null;
  private buildClientPromise: Promise<PrismBuildWorkerClient> | null = null;
  private studioBridgePromise: Promise<StudioChatBridge> | null = null;
  private onOpenHomeCallback: (() => void) | null = null;
  private onOpenSettingsCallback: (() => void) | null = null;
  private onOpenPrismCallback: ((prismId: string, mode?: 'app' | 'studio') => void) | null = null;
  private onOpenGalleryCallback: ((mediaType: GalleryMediaType) => void) | null = null;
  private onSetSurfaceModeCallback: ((mode: PrismSurfaceMode) => void) | null = null;
  private onSetDarkModeCallback: ((dark: boolean) => void) | null = null;
  private onSetStreamingConsentCallback: ((next: 'accepted' | 'paused') => void) | null = null;
  private onGetUiStateCallback: (() => {
    readonly activeView: string;
    readonly surfaceMode: PrismSurfaceMode;
    readonly darkMode: boolean;
    readonly storesHydrated: boolean;
  }) | null = null;

  // Live Observer focus & kernel10 salience state
  private activePartId: string = VERDANT_HOME_PRISM_ID;
  private membraneMilli: readonly number[] = [0, 0, 0, 0, 0, 0, 0, 0];
  private focusSnapshot: RootFocusSnapshot = {
    activePartId: VERDANT_HOME_PRISM_ID,
    attendedSlot: 0,
    foveatedLodRank: 3,
    salienceByPart: { [VERDANT_HOME_PRISM_ID]: 1024 },
  };

  // Per-part non-interruptible task ledger & bounded reconciliation queue
  private readonly partTasks = new Map<string, PartRuntimeTask>();
  private readonly partEpochs = new Map<string, number>();
  private readonly partAbortControllers = new Map<string, AbortController>();
  private readonly pendingReconciliations: QueuedReconciliationJob[] = [];
  private activeReconciliations = 0;
  private activePartIdsSnapshot: readonly string[] = [];

  // 2-tier Observer Frame structural cache (0 Tropical solves during token/LOD ticks)
  private cachedFrameKey = '';
  private cachedFrame: SynthesizedPrismFrame | null = null;
  private cachedLayoutKey = '';
  private cachedLayout: SolvedUiLayout | null = null;
  private layoutSolveCount = 0;

  // Fine-grained subscriber sets for zero-churn React `useSyncExternalStore`
  private readonly focusListeners = new Set<() => void>();
  private readonly activeTasksListeners = new Set<() => void>();
  private readonly partTaskListeners = new Map<string, Set<() => void>>();

  constructor(opts?: VerdantRootPrismOptions) {
    this.workerClient = opts?.workerClient ?? new LlamaVTensorWorkerClient();
    this.observerBus = opts?.observerBus ?? new ObserverStreamBus();
    this.prismStore = opts?.prismStore ?? new PrismStore();
    this.galleryStore = opts?.galleryStore ?? new ContentGalleryStore();
    this.contentStreams = opts?.contentStreams ?? new ContentStreamCatalogStore();
    this.expertRegistry = opts?.expertRegistry ?? new UnifiedExpertRegistry();
    this.systemPartPrisms = buildSystemPartPrisms();
    this.prismPool = new PrismHostPool(
      this.observerBus,
      this.prismStore,
      this.workerClient.cognitiveObserver.grants,
      this.workerClient,
      undefined,
      () => this.getPrismBuildClient(),
    );
  }

  setNavigationCallbacks(cbs: {
    readonly onOpenHome?: () => void;
    readonly onOpenSettings?: () => void;
    readonly onOpenPrism?: (prismId: string, mode?: 'app' | 'studio') => void;
    readonly onOpenGallery?: (mediaType: GalleryMediaType) => void;
    readonly onSetSurfaceMode?: (mode: PrismSurfaceMode) => void;
    readonly onSetDarkMode?: (dark: boolean) => void;
    readonly onSetStreamingConsent?: (next: 'accepted' | 'paused') => void;
    readonly onGetUiState?: () => {
      readonly activeView: string;
      readonly surfaceMode: PrismSurfaceMode;
      readonly darkMode: boolean;
      readonly storesHydrated: boolean;
    };
  }): void {
    this.onOpenHomeCallback = cbs.onOpenHome ?? null;
    this.onOpenSettingsCallback = cbs.onOpenSettings ?? null;
    this.onOpenPrismCallback = cbs.onOpenPrism ?? null;
    this.onOpenGalleryCallback = cbs.onOpenGallery ?? null;
    this.onSetSurfaceModeCallback = cbs.onSetSurfaceMode ?? null;
    this.onSetDarkModeCallback = cbs.onSetDarkMode ?? null;
    this.onSetStreamingConsentCallback = cbs.onSetStreamingConsent ?? null;
    this.onGetUiStateCallback = cbs.onGetUiState ?? null;
  }

  /** Unified wall-clock + monotonic chronological spine shared with the Cognitive Observer. */
  get chronology(): ObserverChronology {
    return this.workerClient.cognitiveObserver.chronology;
  }

  /** Returns live cross-modality timeliness coherence across all parts, turns, and streams. */
  getTemporalCoherenceSnapshot(): TemporalCoherenceSnapshot {
    return this.chronology.snapshot();
  }

  async getPrismBuildClient(): Promise<PrismBuildWorkerClient> {
    if (this.buildClientInstance) return this.buildClientInstance;
    if (!this.buildClientPromise) {
      this.buildClientPromise = import('./prismBuildClient').then(({ PrismBuildWorkerClient }) => {
        const inst = new PrismBuildWorkerClient(this.observerBus);
        this.buildClientInstance = inst;
        return inst;
      });
    }
    return this.buildClientPromise;
  }

  getStudioChatBridge(): Promise<StudioChatBridge> {
    if (!this.studioBridgePromise) {
      this.studioBridgePromise = import('./studioChatBridge').then(
        ({ StudioChatBridge }) =>
          new StudioChatBridge({
            store: this.prismStore,
            pool: this.prismPool,
            observer: this.workerClient.cognitiveObserver,
            galleries: this.galleryStore,
            contentStreams: this.contentStreams,
            registry: this.expertRegistry,
            runLocalStudioPrompt: (systemPrompt, userPrompt, maxTokens, partId) =>
              this.runLocalLanguageHelper(systemPrompt, userPrompt, maxTokens, partId),
            synthesizeAudioWav: (params) => this.workerClient.synthesizeAudioClip(params),
            synthesizeImageLayer: (params) => this.workerClient.synthesizeImageLayer(params),
            evaluateVisualQa: (params) => this.workerClient.evaluateVisualSemanticQa(params),
            runWithDeterministicPrecision: (fn) =>
              this.workerClient.runWithDeterministicPrecision(fn),
            scheduleProgressiveReconciliation: (job) => {
              this.enqueueProgressiveReconciliation(job);
            },
            onOpenPrism: (prismId) => {
              this.onOpenPrismCallback?.(prismId);
            },
            onOpenGallery: (mediaType) => {
              this.onOpenGalleryCallback?.(mediaType);
            },
          }),
      );
    }
    return this.studioBridgePromise;
  }

  async runLocalLanguageHelper(
    systemPrompt: string,
    userPrompt: string,
    maxTokens: number,
    partId = 'studio:chat-router',
    signal?: AbortSignal,
  ): Promise<string> {
    const st = this.workerClient.getState();
    if (!st.model || isInternalBootstrapModel(st.model.id)) return '';
    const formatted = formatMessagesForLlama(
      [
        {
          id: `studio_chat_${Date.now()}`,
          role: 'user',
          createdAt: new Date(),
          content: [{ type: 'text', text: userPrompt }],
          attachments: [],
          metadata: { custom: {} },
        },
      ],
      st.model,
      st.promptMode,
      systemPrompt,
      { thinkingMode: 'off' },
    );
    const ctrl = signal ? null : new AbortController();
    const activeSig = signal ?? ctrl!.signal;
    const salience = this.focusSnapshot.salienceByPart[partId] ?? 512;
    let out = '';
    for await (const ev of this.workerClient.streamGenerate(
      formatted,
      activeSig,
      { ...st.sampling, temperature: 0.25, maxTokens },
      `studio:${partId}`,
      undefined,
      { partId, salience },
    )) {
      out = stripReasoningTags(ev.accumulatedText);
    }
    return out;
  }

  /**
   * Shifts the Observer's viewport focus pointer (`activePartId`) and steps `kernel10`
   * neuromorphic salience — NEVER aborting or interrupting any task running in another part!
   */
  setFocusPart(partId: string, userTyping = false): RootFocusSnapshot {
    const target = partId || VERDANT_HOME_PRISM_ID;
    const trackedIds = Array.from(
      new Set<string>([
        VERDANT_HOME_PRISM_ID,
        target,
        ...this.partTasks.keys(),
        ...this.pendingReconciliations.map((j) => j.prismId),
      ]),
    );
    const focusIdx = Math.max(0, trackedIds.indexOf(target));
    const drive = [
      (focusIdx + 1) * 100,
      userTyping ? 850 : 400,
      target === this.activePartId ? 150 : 600,
      this.activeReconciliations * 200,
      userTyping ? 900 : 500,
      0,
      focusIdx * 100,
      16,
    ];
    const stepped = stepNeuromorphicSalienceRing(
      this.membraneMilli,
      drive,
      trackedIds.length,
      focusIdx,
    );
    this.membraneMilli = stepped.nextMembraneMilli;

    const salienceByPart: Record<string, number> = {};
    for (let i = 0; i < trackedIds.length; i++) {
      salienceByPart[trackedIds[i]!] = stepped.entitySalienceMilli[i] ?? 256;
    }

    const changed = target !== this.activePartId;
    this.activePartId = target;
    this.focusSnapshot = {
      activePartId: target,
      attendedSlot: stepped.attendedSlot,
      foveatedLodRank: stepped.foveatedLodRank,
      salienceByPart,
    };

    if (changed) {
      this.workerClient.setFocusedPart(target);
      for (const fn of this.focusListeners) fn();
    }
    return this.focusSnapshot;
  }

  getFocusSnapshot(): RootFocusSnapshot {
    return this.focusSnapshot;
  }

  /** Fuses rapid UI/part commands via `runtime_monoid.rs` algebraic canonicalization. */
  canonicalizeCommands(ops: readonly PartCommandOp[]) {
    return canonicalizePartCommandMonoid(ops);
  }

  /**
   * 2-tier cached Observer Frame projection (`ui.rs` Tropical `(max,+)` / `(min,+)`):
   * Re-solves layout ONLY when viewport size, surfaceMode, darkMode, typing, or active part
   * identity/count structurally changes — 0 layout solves during token decode or LOD ticks.
   */
  getObserverFrame(params: {
    readonly activeView:
      | { readonly kind: 'thread' }
      | { readonly kind: 'prism'; readonly prismId: string; readonly mode: 'studio' | 'app' }
      | { readonly kind: 'gallery'; readonly mediaType: GalleryMediaType }
      | { readonly kind: 'settings' };
    readonly focusedPrism: VerdantPrismProject | null;
    readonly prisms: readonly VerdantPrismProject[];
    readonly surfaceMode: PrismSurfaceMode;
    readonly viewportWidthPx: number;
    readonly viewportHeightPx: number;
    readonly darkMode: boolean;
    readonly userTyping: boolean;
  }): SynthesizedPrismFrame {
    const activeKey =
      params.activeView.kind === 'prism'
        ? `prism:${params.activeView.prismId}:${params.focusedPrism?.draft.name ?? ''}:${params.focusedPrism?.draft.digest ?? ''}`
        : params.activeView.kind === 'gallery'
          ? `gallery:${params.activeView.mediaType}`
          : params.activeView.kind;
    const key = [
      activeKey,
      params.prisms.length,
      params.surfaceMode,
      params.viewportWidthPx,
      params.viewportHeightPx,
      params.darkMode ? 1 : 0,
      params.userTyping ? 1 : 0,
    ].join('|');

    if (this.cachedFrame && this.cachedFrameKey === key) {
      return this.cachedFrame;
    }

    const view = params.activeView;
    const activePrismDescriptor =
      view.kind === 'prism' && params.focusedPrism
        ? descriptorForUserPrism(params.focusedPrism)
        : view.kind === 'gallery'
          ? this.systemPartPrisms.find((p) => p.id === galleryPrismId(view.mediaType)) ??
            this.systemPartPrisms[0]!
          : view.kind === 'settings'
            ? this.systemPartPrisms.find((p) => p.id === VERDANT_SETTINGS_PRISM_ID) ??
              this.systemPartPrisms[0]!
            : this.systemPartPrisms.find((p) => p.id === VERDANT_HOME_PRISM_ID) ??
              this.systemPartPrisms[0]!;

    const includeSidebar = params.viewportWidthPx >= 1024;
    const layoutKey = [
      activePrismDescriptor.id,
      params.surfaceMode,
      params.viewportWidthPx,
      params.viewportHeightPx,
      includeSidebar ? 1 : 0,
    ].join('|');
    let solvedLayout = this.cachedLayout;
    if (!solvedLayout || this.cachedLayoutKey !== layoutKey) {
      const uiTree = buildCanonicalWorkspaceUiTree(
        activePrismDescriptor.id,
        params.surfaceMode,
        includeSidebar,
      );
      solvedLayout = solveAndVerifyUiTree(uiTree, params.viewportWidthPx, params.viewportHeightPx);
      this.cachedLayoutKey = layoutKey;
      this.cachedLayout = solvedLayout;
      this.layoutSolveCount += 1;
    }

    const frame = synthesizeObserverPrismFrame({
      activePrism: activePrismDescriptor,
      partPrisms: this.systemPartPrisms,
      userPrisms: params.prisms,
      surfaceMode: params.surfaceMode,
      viewportWidthPx: params.viewportWidthPx,
      viewportHeightPx: params.viewportHeightPx,
      darkMode: params.darkMode,
      userTyping: params.userTyping,
      cachedLayout: solvedLayout,
    });
    this.cachedFrameKey = key;
    this.cachedFrame = frame;
    return frame;
  }

  getLayoutSolveCount(): number {
    return this.layoutSolveCount;
  }

  // ---- Fine-grained subscriptions (zero full-tree React re-render storms) ----

  subscribeFocus(listener: () => void): () => void {
    this.focusListeners.add(listener);
    return () => this.focusListeners.delete(listener);
  }

  subscribeActiveTaskIds(listener: () => void): () => void {
    this.activeTasksListeners.add(listener);
    return () => this.activeTasksListeners.delete(listener);
  }

  getActiveBusyPartIds(): readonly string[] {
    return this.activePartIdsSnapshot;
  }

  subscribePartTask(partId: string, listener: () => void): () => void {
    let set = this.partTaskListeners.get(partId);
    if (!set) {
      set = new Set();
      this.partTaskListeners.set(partId, set);
    }
    set.add(listener);
    return () => {
      set?.delete(listener);
      if (set && set.size === 0) this.partTaskListeners.delete(partId);
    };
  }

  getPartTask(partId: string): PartRuntimeTask {
    return this.partTasks.get(partId) ?? IDLE_PART_TASK;
  }

  private bumpPartEpoch(partId: string): number {
    const next = (this.partEpochs.get(partId) ?? 0) + 1;
    this.partEpochs.set(partId, next);
    return next;
  }

  private setPartTask(partId: string, patch: Partial<PartRuntimeTask>): PartRuntimeTask {
    const prev = this.partTasks.get(partId) ?? { ...IDLE_PART_TASK, partId };
    const next: PartRuntimeTask = {
      ...prev,
      ...patch,
      partId,
      updatedAt: Date.now(),
    };
    this.partTasks.set(partId, next);

    // Update compact snapshot of busy part IDs only when membership changes
    const busyIds: string[] = [];
    for (const [id, task] of this.partTasks) {
      if (task.busy) busyIds.push(id);
    }
    const prevIds = this.activePartIdsSnapshot;
    const idsChanged =
      busyIds.length !== prevIds.length || busyIds.some((id, idx) => prevIds[idx] !== id);
    if (idsChanged) {
      this.activePartIdsSnapshot = busyIds;
      for (const fn of this.activeTasksListeners) fn();
    }

    const listeners = this.partTaskListeners.get(partId);
    if (listeners) {
      for (const fn of listeners) fn();
    }
    return next;
  }

  clearPartBanner(partId: string): void {
    const prev = this.partTasks.get(partId);
    if (prev?.banner) {
      this.setPartTask(partId, { banner: null });
    }
  }

  /** Cancels ONLY the explicitly targeted part's in-flight task — sibling parts continue cleanly. */
  cancelPartTask(partId: string): void {
    this.bumpPartEpoch(partId);
    const ctrl = this.partAbortControllers.get(partId);
    if (ctrl) {
      ctrl.abort();
      this.partAbortControllers.delete(partId);
    }
    for (let i = this.pendingReconciliations.length - 1; i >= 0; i--) {
      if (this.pendingReconciliations[i]!.prismId === partId) {
        const [removed] = this.pendingReconciliations.splice(i, 1);
        removed?.controller.abort();
        removed?.resolve();
      }
    }
    this.chronology.completeObject(`prism:${partId}`);
    this.setPartTask(partId, {
      status: 'idle',
      busy: false,
      label: '',
    });
  }

  /**
   * Enqueues a background `Lod0 -> Lod3` reconciliation loop inside `VerdantRootPrismRuntime`
   * with bounded concurrency (`MAX_CONCURRENT_RECONCILIATIONS = 2`), `kernel10` salience ordering,
   * monotonic epoch guards, and bounded watchdog timeouts.
   */
  enqueueProgressiveReconciliation(params: {
    readonly prismId: string;
    readonly prompt: string;
    readonly initialRevision: SemanticRevisionRecord;
    readonly startRank?: PrismLodRank;
    readonly maxRank?: PrismLodRank;
  }): Promise<void> {
    const epoch = this.bumpPartEpoch(params.prismId);
    const prevCtrl = this.partAbortControllers.get(params.prismId);
    if (prevCtrl) prevCtrl.abort();
    const controller = new AbortController();
    this.partAbortControllers.set(params.prismId, controller);

    // Replace any pending lower-epoch job for the same Prism
    for (let i = this.pendingReconciliations.length - 1; i >= 0; i--) {
      if (this.pendingReconciliations[i]!.prismId === params.prismId) {
        const [old] = this.pendingReconciliations.splice(i, 1);
        old?.controller.abort();
        old?.resolve();
      }
    }

    const startRank: PrismLodRank = params.startRank ?? 1;
    const maxRank: PrismLodRank = params.maxRank ?? 3;
    this.setPartTask(params.prismId, {
      status: 'qa-reconciling',
      busy: true,
      lodRank: startRank,
      lodLabel: LOD_LABELS[startRank]!,
      semanticVersion: params.initialRevision.version,
      epoch,
      label: LOD_USER_STATUS[startRank]!,
    });

    return new Promise<void>((resolve) => {
      this.pendingReconciliations.push({
        prismId: params.prismId,
        prompt: params.prompt,
        initialRevision: params.initialRevision,
        startRank,
        maxRank,
        epoch,
        controller,
        resolve,
      });
      void this.pumpReconciliationQueue();
    });
  }

  private dequeueSalientReconciliation(): QueuedReconciliationJob | undefined {
    if (this.pendingReconciliations.length === 0) return undefined;
    let bestIdx = 0;
    let bestScore = -1;
    for (let i = 0; i < this.pendingReconciliations.length; i++) {
      const job = this.pendingReconciliations[i]!;
      const isFocused = job.prismId === this.activePartId ? 10_000 : 0;
      const sal = this.focusSnapshot.salienceByPart[job.prismId] ?? 512;
      if (isFocused + sal > bestScore) {
        bestScore = isFocused + sal;
        bestIdx = i;
      }
    }
    return this.pendingReconciliations.splice(bestIdx, 1)[0];
  }

  private async pumpReconciliationQueue(): Promise<void> {
    while (
      this.activeReconciliations < MAX_CONCURRENT_RECONCILIATIONS &&
      this.pendingReconciliations.length > 0
    ) {
      const job = this.dequeueSalientReconciliation();
      if (!job) break;
      this.activeReconciliations += 1;
      void this.runReconciliationJob(job).finally(() => {
        this.activeReconciliations = Math.max(0, this.activeReconciliations - 1);
        job.resolve();
        void this.pumpReconciliationQueue();
      });
    }
  }

  private async runReconciliationJob(job: QueuedReconciliationJob): Promise<void> {
    const isStale = () =>
      job.controller.signal.aborted || this.partEpochs.get(job.prismId) !== job.epoch;
    const prismObjId = `prism:${job.prismId}`;
    const stageBudgetMs = this.chronology.derivePrismStageBudgetMs(job.maxRank);
    const chronoRec = this.chronology.beginObject({
      objectId: prismObjId,
      modality: 'prism',
      kind: 'prism-part',
      label: job.prompt.slice(0, 64) || job.prismId,
      budgetMs: stageBudgetMs,
      periodMs: 1_000,
    });
    this.setPartTask(job.prismId, {
      startedAtWallMs: chronoRec.bornAtWallMs,
      elapsedMs: 0,
      timelinessQ10: 1024,
      timelinessState: 'on-time',
    });
    let watchdog: ReturnType<typeof setTimeout> | null = null;
    try {
      watchdog = setTimeout(() => {
        job.controller.abort();
      }, stageBudgetMs);

      const { runProgressivePrismReconciliation } = await import('./prismReconciliationLoop');
      const res = await runProgressivePrismReconciliation({
        prismId: job.prismId,
        prompt: job.prompt,
        initialRevision: job.initialRevision,
        startRank: job.startRank,
        maxRank: job.maxRank,
        isStale,
        onProgress: (ev) => {
          if (isStale()) return;
          const obs = this.chronology.observeObject(prismObjId);
          this.setPartTask(job.prismId, {
            status: ev.lodRank >= job.maxRank ? 'ready' : 'qa-reconciling',
            busy: ev.lodRank < job.maxRank,
            lodRank: ev.lodRank,
            lodLabel: ev.lodLabel,
            qaScoreQ10: ev.qaScoreQ10,
            semanticVersion: ev.semanticVersion,
            epoch: job.epoch,
            label: ev.statusLabel,
            ...(obs
              ? {
                  elapsedMs: obs.elapsedMs,
                  timelinessQ10: obs.timelinessQ10,
                  timelinessState: obs.state,
                }
              : {}),
          });
        },
        deps: {
          store: this.prismStore,
          galleries: this.galleryStore,
          registry: this.expertRegistry,
          // Keep background LOD1..LOD3 visual/audio reconciliation strictly non-blocking:
          // never contend with or wipe the active chat thread's KV cache on the single worker,
          // and avoid slow ~20s CPU SDXS passes that overwrite crisp procedural scenes!
          runLocalLanguagePrompt: undefined,
          synthesizeImageLayer: undefined,
          evaluateVisualQa: (p) => this.workerClient.evaluateVisualSemanticQa(p),
          runWithDeterministicPrecision: (fn) =>
            this.workerClient.runWithDeterministicPrecision(fn),
          syncSpacesForArtifact: async (pid) => {
            const rec = this.galleryStore.get(pid);
            if (!rec || rec.spaceIds.length === 0) return;
            const { syncSpacePrismView } = await import('./studioChatBridge');
            for (const sid of rec.spaceIds) {
              await syncSpacePrismView(sid, {
                store: this.prismStore,
                galleries: this.galleryStore,
                observer: this.workerClient.cognitiveObserver,
                registry: this.expertRegistry,
              });
            }
          },
        },
      });

      const finalObs = this.chronology.completeObject(prismObjId);
      if (!isStale()) {
        this.setPartTask(job.prismId, {
          status: 'ready',
          busy: false,
          lodRank: res.finalRank,
          lodLabel: LOD_LABELS[res.finalRank]!,
          qaScoreQ10: res.qaScoreQ10,
          semanticVersion: res.finalRevision.version,
          epoch: job.epoch,
          label: '',
          ...(finalObs
            ? {
                elapsedMs: finalObs.elapsedMs,
                timelinessQ10: finalObs.timelinessQ10,
                timelinessState: finalObs.state,
              }
            : {}),
        });
      }
    } catch {
      const finalObs = this.chronology.completeObject(prismObjId);
      if (!isStale()) {
        this.setPartTask(job.prismId, {
          status: 'ready',
          busy: false,
          label: '',
          ...(finalObs
            ? {
                elapsedMs: finalObs.elapsedMs,
                timelinessQ10: finalObs.timelinessQ10,
                timelinessState: finalObs.state,
              }
            : {}),
        });
      }
    } finally {
      if (watchdog !== null) clearTimeout(watchdog);
      if (this.partAbortControllers.get(job.prismId) === job.controller) {
        this.partAbortControllers.delete(job.prismId);
      }
    }
  }

  /**
   * Executes an in-Prism natural-language refinement or companion generation inside
   * `VerdantRootPrismRuntime` so navigating to another Prism or tab NEVER aborts it!
   */
  async dispatchPrismPrompt(
    prismId: string,
    instruction: string,
  ): Promise<{ readonly ok: boolean; readonly text: string; readonly openPrismId?: string }> {
    const trimmed = instruction.trim();
    if (!trimmed) return { ok: false, text: '' };
    const epoch = this.bumpPartEpoch(prismId);
    const prevCtrl = this.partAbortControllers.get(prismId);
    if (prevCtrl) prevCtrl.abort();
    const ctrl = new AbortController();
    this.partAbortControllers.set(prismId, ctrl);

    const isStale = () => ctrl.signal.aborted || this.partEpochs.get(prismId) !== epoch;
    const isCompanionReq =
      /\b(generate|create|make|add|compose)\b[\s\S]{0,36}\b(cover\s*art|book\s*cover|album\s*cover|box\s*art|logo|brand\s*mark|emblem|soundtrack|theme\s*song|background\s*music)\b/i.test(
        trimmed,
      );

    const prismObjId = `prism:${prismId}`;
    const chronoRec = this.chronology.beginObject({
      objectId: prismObjId,
      modality: 'prism',
      kind: 'prism-part',
      label: trimmed.slice(0, 64),
      budgetMs: this.chronology.derivePrismStageBudgetMs(2),
    });
    this.setPartTask(prismId, {
      status: 'synthesizing',
      busy: true,
      epoch,
      label: isCompanionReq ? 'Creating companion…' : 'Refining…',
      banner: null,
      startedAtWallMs: chronoRec.bornAtWallMs,
      elapsedMs: 0,
      timelinessQ10: 1024,
      timelinessState: 'on-time',
    });

    try {
      const bridge = await this.getStudioChatBridge();
      const res = await bridge.refinePrismFromPrompt({
        target: prismId,
        instruction: trimmed,
      });
      const finalObs = this.chronology.completeObject(prismObjId);
      if (isStale()) return { ok: false, text: '' };

      const openedCompanion = res.prismId !== prismId ? res.prismId : undefined;
      this.setPartTask(prismId, {
        status: 'ready',
        busy: false,
        epoch,
        label: '',
        ...(finalObs
          ? {
              elapsedMs: finalObs.elapsedMs,
              timelinessQ10: finalObs.timelinessQ10,
              timelinessState: finalObs.state,
            }
          : {}),
        banner: {
          ok: true,
          text: openedCompanion
            ? `Created "${res.prismName}" from this creation.`
            : res.summary || 'Updated from your prompt.',
        },
      });
      return { ok: true, text: res.summary, openPrismId: openedCompanion };
    } catch (err) {
      this.chronology.completeObject(prismObjId);
      if (isStale()) return { ok: false, text: '' };
      const msg = err instanceof Error ? err.message : String(err);
      this.setPartTask(prismId, {
        status: 'error',
        busy: false,
        epoch,
        label: '',
        banner: { ok: false, text: msg },
      });
      return { ok: false, text: msg };
    } finally {
      if (this.partAbortControllers.get(prismId) === ctrl) {
        this.partAbortControllers.delete(prismId);
      }
    }
  }

  /**
   * Executes pre-publish visual/audio quality check + deterministic `strict-f64` `.holo` sealing
   * inside `VerdantRootPrismRuntime` so navigating away mid-publish still completes cleanly!
   */
  async publishPrism(prismId: string): Promise<{ readonly ok: boolean; readonly text: string }> {
    const project = this.prismStore.get(prismId);
    if (!project) return { ok: false, text: 'Creation not found.' };
    const declaration = project.declaration ?? null;
    const isDeclarative = project.kind === 'declarative' || declaration !== null;

    this.prismPool.evictPrism(prismId);
    if (!isDeclarative) {
      await this.prismStore.publish(prismId);
      this.setPartTask(prismId, {
        status: 'ready',
        busy: false,
        label: '',
        banner: { ok: true, text: 'Published.' },
      });
      return { ok: true, text: 'Published.' };
    }

    if (!declaration) {
      const msg = 'Describe your idea in the prompt bar below first so we can finish building it.';
      this.setPartTask(prismId, { banner: { ok: false, text: msg } });
      return { ok: false, text: msg };
    }

    const epoch = this.bumpPartEpoch(prismId);
    this.setPartTask(prismId, {
      status: 'publishing',
      busy: true,
      epoch,
      label: 'Publishing…',
      banner: null,
    });

    try {
      const [{ compilePrismDeclaration }, { evaluateVerdantRouting }] = await Promise.all([
        import('./prismCompiler'),
        import('./verdantStandardsRouter'),
      ]);
      const compiled = compilePrismDeclaration(declaration);
      const readiness = evaluateVerdantRouting(declaration);
      const hasProceduralOrgan = Boolean(declaration.view?.proceduralOrgan);
      if (!compiled || (!readiness.readyToSeal && !hasProceduralOrgan)) {
        const msg = 'Describe your idea in the prompt bar below first so we can finish building it.';
        this.setPartTask(prismId, { status: 'idle', busy: false, label: '', banner: { ok: false, text: msg } });
        return { ok: false, text: msg };
      }

      let qualityNote = '';
      const organ = declaration.view?.proceduralOrgan;
      if (organ?.mediaType === 'picture') {
        const { renderPictureWithCoherenceCheck } = await import('./proceduralPictureRenderer');
        const check = await renderPictureWithCoherenceCheck(organ.weights as PictureWeights, {
          maxPreviewDim: 384,
        });
        if (!check.coherence.coherent) {
          qualityNote = ' — low visual contrast noted';
        }
      } else if (organ?.mediaType === 'audio') {
        const {
          analyzeAudioFidelity,
          composeMultiTrackArrangement,
          masterStereoPcm,
          synthesizeCompositionStereoPcm,
        } = await import('./audioSynthesisEngine');
        const aw = organ.weights as AudioWeights;
        const comp = composeMultiTrackArrangement({
          title: aw.title,
          style: aw.style,
          bpm: aw.bpm,
          rootMidi: aw.rootMidi,
          scale: aw.scale,
          durationSec: Math.min(6, aw.durationSec),
        });
        const raw = synthesizeCompositionStereoPcm(comp, 22_050);
        const mastered = masterStereoPcm(raw.left, raw.right, raw.sampleRate);
        const fidelity = analyzeAudioFidelity(mastered.left, mastered.right);
        if (fidelity.hasNaN || !fidelity.nonSilent || fidelity.clippedSamples > 0) {
          qualityNote = ' — audio level adjusted';
        }
      }

      let sealed = project.sealedDraft ?? null;
      if (!sealed) {
        const buildClient = await this.getPrismBuildClient();
        const res = await this.workerClient.runWithDeterministicPrecision(() =>
          buildClient.buildAndSeal({
            prismId: project.id,
            slug: declaration.metadata.slug,
            appFiles: compiled.appFiles,
          }),
        );
        if (!res.ok || !res.holo) {
          const msg = res.message || 'Could not publish right now.';
          this.setPartTask(prismId, { status: 'error', busy: false, epoch, label: '', banner: { ok: false, text: msg } });
          return { ok: false, text: msg };
        }
        sealed = {
          bytes: res.holo.bytes,
          address: res.holo.address,
          schema: 'prism/holo/6',
          semanticId: res.holo.semanticId,
          proved: res.proved,
          theorems: res.theorems.map((t) => ({
            theorem: t.theorem,
            status: t.status,
            axioms: t.axioms,
          })),
          vectorsPassed: res.vectorsPassed,
          vectorsTotal: res.vectorsTotal,
          carries: res.holo.carries,
          sealedAt: Date.now(),
        };
        await this.prismStore.saveSealedDraft(project.id, sealed);
      }
      await this.prismStore.publishSealedHolo(project.id, sealed);
      const msg = qualityNote
        ? `"${project.draft.name}" published${qualityNote}.`
        : `"${project.draft.name}" verified and published.`;
      this.setPartTask(prismId, {
        status: 'ready',
        busy: false,
        epoch,
        label: '',
        banner: { ok: true, text: msg },
      });
      return { ok: true, text: msg };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.setPartTask(prismId, {
        status: 'error',
        busy: false,
        epoch,
        label: '',
        banner: { ok: false, text: msg },
      });
      return { ok: false, text: msg };
    }
  }

  getEngineState(): EngineState {
    return this.workerClient.getState();
  }

  /**
   * Produces a unified, live OTEL + Chronology + Observer + Substrate self-report for direct
   * introspection and second-order cybernetic evaluation (`window.__VERDANT_OBSERVER__`).
   */
  getObserverFieldReport() {
    const engine = this.workerClient.getState();
    const chrono = this.chronology.snapshot();
    const obs = this.workerClient.cognitiveObserver;
    const activeThreadId =
      typeof this.workerClient.getActiveThreadId === 'function'
        ? this.workerClient.getActiveThreadId()
        : 'default';
    const threadSnap = obs.getThreadSnapshot(activeThreadId);
    const activeModel =
      engine.model && !isInternalBootstrapModel(engine.model.id) ? engine.model : null;
    const recentSpans = this.observerBus
      .getRecentSpans()
      .slice(-24)
      .map((s) => ({
        name: s.name,
        endpointKind: s.endpointKind,
        endpointId: s.endpointId,
        status: s.status,
        startTimeMs: s.startTimeMs,
        durationMs: s.durationMs ?? null,
        attributes: s.attributes,
        events: s.events.map((e) => ({
          name: e.name,
          timestampMs: e.timestampMs,
          status: e.status,
          attributes: e.attributes,
        })),
      }));
    return {
      timestampMs: chrono.instant.wallClockMs,
      runtime: {
        status: engine.status,
        statusDetail: engine.statusDetail,
        model: activeModel
          ? {
              id: activeModel.id,
              displayName: activeModel.displayName,
              backingKind: activeModel.backingKind,
              quantization: activeModel.hparams.primaryQuantization,
              contextWindow: activeModel.maxContextWindow,
            }
          : null,
        kernelAcceleration: engine.telemetry?.kernelAcceleration ?? 'WASM SIMD',
        workers: engine.telemetry?.threadPool?.workerCount ?? 1,
        lastTokensPerSec: engine.lastTokensPerSec ?? 0,
        opfsUsageBytes: engine.opfsStorage?.usageBytes ?? 0,
        opfsQuotaBytes: engine.opfsStorage?.quotaBytes ?? 0,
        unifiedStore: engine.telemetry?.unifiedStore
          ? {
              kvPrefixBlocks: engine.telemetry.unifiedStore.kvPrefixBlocks,
              kvPrefixMaxBlocks: engine.telemetry.unifiedStore.kvPrefixMaxBlocks,
              kvPrefixPersistedOpfs: engine.telemetry.unifiedStore.kvPrefixPersistedOpfs,
              unifiedRestoredTokens: engine.telemetry.unifiedStore.unifiedRestoredTokens,
              speculativeReadyTokens: engine.telemetry.unifiedStore.speculativeReadyTokens,
              lastTurnPrefillHitRatePct: engine.telemetry.unifiedStore.lastTurnPrefillHitRatePct,
              estimatedTtftSavedMs: engine.telemetry.unifiedStore.estimatedTtftSavedMs,
            }
          : null,
        cachedModels: (engine.opfsStorage?.storedModels ?? []).map((m) => ({
          modelId: m.modelId,
          byteLength: m.byteLength,
          storageMode: m.storageMode ?? 'full',
        })),
      },
      chronology: chrono,
      observer: {
        activeSpace: obs.possession.getActiveCompartmentId(),
        publishedPrismCount: this.prismStore.listPublishedSummaries().length,
        turnCount: threadSnap.turnCount,
        lastSurpriseQ10: threadSnap.lastSurpriseQ10,
        lastCertificate: threadSnap.lastCertificate
          ? {
              routeId: threadSnap.lastCertificate.routeId,
              label: threadSnap.lastCertificate.label,
              kind: threadSnap.lastCertificate.kind,
              modelId: threadSnap.lastCertificate.modelId,
              thinkingPolicy: threadSnap.lastCertificate.thinkingPolicy,
              attachTools: threadSnap.lastCertificate.attachTools,
              freeEnergy: threadSnap.lastCertificate.freeEnergy,
              qualityBudget: threadSnap.lastCertificate.qualityBudget,
              temporalAnchor: threadSnap.lastCertificate.temporalAnchor,
              rationale: threadSnap.lastCertificate.rationale,
            }
          : null,
      },
      otel: {
        spanCount: recentSpans.length,
        recentSpans,
      },
    };
  }

  /**
   * Active Observer GUI Perception (`Studio.Gui.Perception` · `REQ-AGENT-USERS` · `REQ-AGENT-OBSERVER`).
   * Lets an agent perceive the exact same parts, views, galleries, spaces, streams, tasks, and
   * layout compliance as a human user, recording a traced perception event on `observerBus`.
   */
  perceiveGuiState() {
    const span = this.observerBus.startSpan('observer.perceive', {
      endpointKind: 'observer',
      endpointId: this.activePartId,
    });
    const uiState = this.onGetUiStateCallback?.() ?? {
      activeView: this.activePartId,
      surfaceMode: 'preview' as PrismSurfaceMode,
      darkMode: false,
      storesHydrated: true,
    };
    const allPrisms = this.prismStore.list();
    const galleryCounts = this.galleryStore.galleryCounts();
    const allStreams = this.contentStreams.listAll();
    const enabledStreams = allStreams.filter((s) => s.enabled);
    const activeTasks = this.getActiveBusyPartIds().map((id) => this.getPartTask(id));
    const field = this.getObserverFieldReport();
    span.setAttributes({
      'verdant.active_part': this.activePartId,
      'verdant.prism_count': allPrisms.length,
      'verdant.enabled_streams': enabledStreams.length,
    });
    span.end('ok');
    return {
      ui: {
        ...uiState,
        activePartId: this.activePartId,
        layoutCompliant: this.cachedLayout ? this.cachedLayout.certificate.ok : true,
        layoutSolveCount: this.layoutSolveCount,
      },
      parts: this.systemPartPrisms.map((p) => ({
        id: p.id,
        name: p.name,
        kind: p.kind,
        subtitle: p.subtitle,
      })),
      galleries: galleryCounts,
      prisms: allPrisms.map((p) => ({
        id: p.id,
        name: p.published?.name ?? p.draft.name,
        mediaType: p.declaration?.view?.proceduralOrgan?.mediaType ?? p.declaration?.concept.mediaType ?? 'tool-organ',
        published: Boolean(p.published),
        spaceIds: this.galleryStore.get(p.id)?.spaceIds ?? [],
        task: this.getPartTask(p.id),
      })),
      contentStreams: {
        total: allStreams.length,
        enabled: enabledStreams.length,
        enabledIds: enabledStreams.map((s) => s.id),
      },
      activeTasks,
      runtime: field.runtime,
      chronology: field.chronology,
      observer: field.observer,
    };
  }

  /**
   * Active Observer Self-Diagnosis (`REQ-SELF-HEALING` · `REQ-AGENT-OBSERVER`).
   * Delegates system-wide health and feature-completeness evaluation directly to the Observer.
   */
  diagnoseSystemHealth(): {
    readonly overallHealthQ10: number;
    readonly status: 'healthy' | 'attention' | 'degraded';
    readonly featureCoverage: Readonly<Record<string, number>>;
    readonly findings: ReadonlyArray<{
      readonly severity: 'info' | 'warn' | 'error';
      readonly subsystem: string;
      readonly message: string;
      readonly recommendedAction?: string;
    }>;
  } {
    const span = this.observerBus.startSpan('observer.diagnose', {
      endpointKind: 'observer',
      endpointId: 'verdant-ui-root',
    });
    const findings: Array<{
      readonly severity: 'info' | 'warn' | 'error';
      readonly subsystem: string;
      readonly message: string;
      readonly recommendedAction?: string;
    }> = [];
    const field = this.getObserverFieldReport();
    const chrono = field.chronology;
    const uStore = field.runtime.unifiedStore;
    let score = 1024;

    if (chrono.overdueCount > 0) {
      score -= Math.min(320, chrono.overdueCount * 160);
      findings.push({
        severity: 'error',
        subsystem: 'chronology',
        message: `${chrono.overdueCount} active chronological object(s) overdue (coherence ${chrono.coherenceQ10}/1024).`,
        recommendedAction: 'Run observer.enact({ type: "self_heal" }) or inspect overdue task spans.',
      });
    } else if (chrono.coherenceQ10 < 512) {
      score -= 128;
      findings.push({
        severity: 'warn',
        subsystem: 'chronology',
        message: `Global timeliness coherence is ${chrono.coherenceQ10}/1024 across modalities.`,
        recommendedAction: 'Review recent turn TTFT vs calibrated throughput budgets.',
      });
    }

    if (field.runtime.model && uStore && !uStore.kvPrefixPersistedOpfs && uStore.kvPrefixBlocks > 0) {
      score -= 96;
      findings.push({
        severity: 'warn',
        subsystem: 'kv-cache',
        message: 'Unified KV prefix blocks exist in RAM but have not yet persisted to OPFS.',
        recommendedAction: 'Run observer.enact({ type: "self_heal" }) to flush unified KV persistence.',
      });
    }

    for (const [mod, q10] of Object.entries(chrono.modalityCoherenceQ10)) {
      if (typeof q10 === 'number' && q10 < 384) {
        score -= 128;
        findings.push({
          severity: 'warn',
          subsystem: `chronology:${mod}`,
          message: `Modality "${mod}" timeliness coherence is low (${q10}/1024).`,
          recommendedAction: `Inspect recent "${mod}" spans for prompt bloat or stalls.`,
        });
      }
    }

    const recentSpans = this.observerBus.getRecentSpans();
    const recentCancelled = recentSpans.filter(
      (s) => s.name === 'thread.turn' && s.attributes['verdant.cancelled'] === true,
    );
    if (recentCancelled.length > 0) {
      const lastCancel = recentCancelled[recentCancelled.length - 1]!;
      score -= 96;
      findings.push({
        severity: 'warn',
        subsystem: 'inference',
        message: `${recentCancelled.length} recent cancelled chat turn(s) (latest ran ${Math.round(Number(lastCancel.attributes['verdant.total_ms'] ?? lastCancel.durationMs ?? 0) / 1000)}s).`,
        recommendedAction: 'Check prompt token count, tool attachment, and thinking token budget.',
      });
    }

    const recentErrors = recentSpans.filter((s) => s.status === 'error');
    if (recentErrors.length > 0) {
      score -= Math.min(256, recentErrors.length * 96);
      findings.push({
        severity: 'error',
        subsystem: 'otel',
        message: `${recentErrors.length} recent OTEL error span(s) recorded (latest: ${recentErrors[recentErrors.length - 1]!.name}).`,
        recommendedAction: 'Inspect recentSpans() error attributes and reconcile the failing part.',
      });
    }

    if (this.cachedLayout && !this.cachedLayout.certificate.ok) {
      score -= 192;
      findings.push({
        severity: 'error',
        subsystem: 'ui-layout',
        message: 'Synthesized UI layout certificate failed verification.',
        recommendedAction: 'Re-solve layout tree on resize or inspect PrismSurfaceShell bounds.',
      });
    }

    const galleryCounts = this.galleryStore.galleryCounts();
    const prisms = this.prismStore.list();
    const spacesCount = prisms.filter(
      (p) => (p.declaration?.view?.proceduralOrgan?.mediaType ?? p.declaration?.concept.mediaType) === 'space',
    ).length;
    const featureCoverage: Record<string, number> = {
      ...galleryCounts,
      spaces: spacesCount,
      publishedPrisms: this.prismStore.listPublishedSummaries().length,
      enabledContentStreams: this.contentStreams.listAll().filter((s) => s.enabled).length,
      cachedModels: field.runtime.cachedModels.length,
    };

    const unindexedPrisms = prisms.filter((p) => {
      const mt = p.declaration?.view?.proceduralOrgan?.mediaType ?? p.declaration?.concept.mediaType;
      return mt && mt !== 'space' && !this.galleryStore.get(p.id);
    });
    if (unindexedPrisms.length > 0) {
      score -= 96;
      findings.push({
        severity: 'warn',
        subsystem: 'galleries',
        message: `${unindexedPrisms.length} Prism(s) missing from Gallery index.`,
        recommendedAction: 'Run observer.enact({ type: "self_heal" }) to synchronize gallery & expert indexes.',
      });
    }

    if (findings.length === 0) {
      findings.push({
        severity: 'info',
        subsystem: 'root',
        message: 'All monitored subsystems (runtime, OPFS, KV cache, chronology, galleries, UI layout) are healthy.',
      });
    }

    const overallHealthQ10 = Math.max(0, Math.min(1024, Math.round(score)));
    const status =
      overallHealthQ10 >= 768 ? 'healthy' : overallHealthQ10 >= 448 ? 'attention' : 'degraded';
    span.setAttributes({
      'verdant.health_q10': overallHealthQ10,
      'verdant.findings': findings.length,
    });
    span.end('ok');
    return { overallHealthQ10, status, featureCoverage, findings };
  }

  /**
   * Self-healing reconciliation (`REQ-SELF-HEALING`): synchronizes Gallery and Expert Basis
   * indexes from authoritative Prism storage and clears stale errored tasks.
   */
  reconcileSelfHeal(): { readonly syncedPrisms: number; readonly clearedErrors: number } {
    const prisms = this.prismStore.list();
    this.galleryStore.syncFromPrisms(prisms);
    this.expertRegistry.syncPrismsAsExperts(prisms);
    let clearedErrors = 0;
    for (const [partId, task] of this.partTasks.entries()) {
      if (task.status === 'error' && !task.busy) {
        this.setPartTask(partId, { status: 'idle', busy: false, label: '', banner: null });
        clearedErrors += 1;
      }
    }
    return { syncedPrisms: prisms.length, clearedErrors };
  }

  /**
   * Guarded GUI Action Enactment (`Studio.Gui.Enactment` · `REQ-AGENT-USERS` · `REQ-AGENT-OBSERVER`).
   * Executes the exact same actions available to human users under `ContainerGrantMatrix` (FK-07)
   * and records every enactment on the W3C OTEL `observerBus`.
   */
  async enactGuiAction(
    action:
      | { readonly type: 'open_home' }
      | { readonly type: 'open_settings' }
      | { readonly type: 'open_gallery'; readonly mediaType: GalleryMediaType }
      | { readonly type: 'open_prism'; readonly prismId: string; readonly mode?: 'app' | 'studio' }
      | { readonly type: 'set_surface_mode'; readonly mode: PrismSurfaceMode }
      | { readonly type: 'set_dark_mode'; readonly dark: boolean }
      | { readonly type: 'set_space'; readonly spaceId: string }
      | { readonly type: 'set_streaming_consent'; readonly consent: 'accepted' | 'paused' }
      | { readonly type: 'toggle_content_stream'; readonly streamId: string; readonly enabled: boolean }
      | { readonly type: 'create_artifact'; readonly prompt: string; readonly mediaType?: GalleryMediaType; readonly spaceName?: string }
      | { readonly type: 'refine_prism'; readonly prismId: string; readonly instruction: string }
      | { readonly type: 'publish_prism'; readonly prismId: string }
      | { readonly type: 'delete_prism'; readonly prismId: string }
      | { readonly type: 'self_heal' },
  ): Promise<{ readonly ok: boolean; readonly action: string; readonly result?: unknown; readonly error?: string }> {
    const span = this.observerBus.startSpan('observer.enact', {
      endpointKind: 'observer',
      endpointId: this.activePartId,
      attributes: { 'verdant.action': action.type },
    });
    try {
      let result: unknown;
      switch (action.type) {
        case 'open_home':
          this.onOpenHomeCallback?.();
          result = { view: 'thread' };
          break;
        case 'open_settings':
          this.onOpenSettingsCallback?.();
          result = { view: 'settings' };
          break;
        case 'open_gallery':
          this.onOpenGalleryCallback?.(action.mediaType);
          result = { view: 'gallery', mediaType: action.mediaType };
          break;
        case 'open_prism': {
          const p = this.prismStore.get(action.prismId);
          if (!p) throw new Error(`Prism "${action.prismId}" not found.`);
          this.onOpenPrismCallback?.(action.prismId, action.mode ?? 'app');
          result = { view: 'prism', prismId: action.prismId, mode: action.mode ?? 'app' };
          break;
        }
        case 'set_surface_mode':
          this.onSetSurfaceModeCallback?.(action.mode);
          result = { surfaceMode: action.mode };
          break;
        case 'set_dark_mode':
          if (typeof action.dark !== 'boolean') {
            throw new Error('set_dark_mode requires a boolean "dark" field.');
          }
          this.onSetDarkModeCallback?.(action.dark);
          result = { darkMode: action.dark };
          break;
        case 'set_space':
          this.workerClient.cognitiveObserver.possession.setActiveCompartment(action.spaceId);
          result = { activeSpace: this.workerClient.cognitiveObserver.possession.getActiveCompartmentId() };
          break;
        case 'set_streaming_consent':
          this.onSetStreamingConsentCallback?.(action.consent);
          result = { streamingConsent: action.consent };
          break;
        case 'toggle_content_stream':
          this.contentStreams.toggleStream(action.streamId, action.enabled);
          result = { streamId: action.streamId, enabled: action.enabled };
          break;
        case 'create_artifact': {
          if (!this.workerClient.cognitiveObserver.grants.isAllowed('prism-runtime')) {
            throw new Error('Prism creation denied by ContainerGrantMatrix (FK-07).');
          }
          const bridge = await this.getStudioChatBridge();
          result = await bridge.createArtifactFromPrompt({
            prompt: action.prompt,
            forcedMediaType: action.mediaType,
            spaceName: action.spaceName,
          });
          break;
        }
        case 'refine_prism': {
          if (!this.workerClient.cognitiveObserver.grants.isAllowed('prism-runtime')) {
            throw new Error('Prism refinement denied by ContainerGrantMatrix (FK-07).');
          }
          const bridge = await this.getStudioChatBridge();
          result = await bridge.refinePrismFromPrompt({
            target: action.prismId,
            instruction: action.instruction,
          });
          break;
        }
        case 'publish_prism':
          result = await this.publishPrism(action.prismId);
          break;
        case 'delete_prism':
          this.cancelPartTask(action.prismId);
          await this.prismStore.delete(action.prismId);
          this.galleryStore.syncFromPrisms(this.prismStore.list());
          result = { deleted: action.prismId };
          break;
        case 'self_heal':
          result = this.reconcileSelfHeal();
          break;
      }
      span.end('ok');
      return { ok: true, action: action.type, result };
    } catch (err) {
      const error = err instanceof Error ? err.message : String(err);
      span.end('error', error);
      return { ok: false, action: action.type, error };
    }
  }

  listGalleryArtifacts(): readonly ContentArtifactRecord[] {
    return this.galleryStore.listAll();
  }

  dispose(): void {
    for (const ctrl of this.partAbortControllers.values()) ctrl.abort();
    this.partAbortControllers.clear();
    this.pendingReconciliations.length = 0;
    this.prismPool.dispose();
    this.buildClientInstance?.dispose();
    this.workerClient.dispose();
  }
}

let singletonRootPrism: VerdantRootPrismRuntime | null = null;

/** Returns the single live `verdant-ui-root` Prism runtime instance for the browser session. */
export function getVerdantRootPrismRuntime(): VerdantRootPrismRuntime {
  if (!singletonRootPrism) {
    singletonRootPrism = new VerdantRootPrismRuntime();
    if (typeof window !== 'undefined') {
      const root = singletonRootPrism;
      (window as unknown as Record<string, unknown>).__VERDANT_OBSERVER__ = Object.freeze({
        fieldReport: () => root.getObserverFieldReport(),
        perceive: () => root.perceiveGuiState(),
        diagnose: () => root.diagnoseSystemHealth(),
        enact: (action: Parameters<VerdantRootPrismRuntime['enactGuiAction']>[0]) =>
          root.enactGuiAction(action),
        chronology: () => root.getTemporalCoherenceSnapshot(),
        recentSpans: () => root.observerBus.getRecentSpans(),
        otlpJson: () => root.observerBus.toOtlpJson(),
        routeCertificate: () =>
          root.workerClient.cognitiveObserver.getThreadSnapshot(
            root.workerClient.getActiveThreadId(),
          ).lastCertificate,
      });
    }
  }
  return singletonRootPrism;
}
