/**
 * Verdant Studio & Published App Workspace (`src/components/VerdantStudioView.tsx`)
 *
 * Implements DEC-011 (Zero Technical Jargon) & DEC-012 (Zero User Tunables):
 * - Studio Mode: Conversational Creator (`StudioGuidedWizard`) + Live Preview & simple Details.
 * - App Mode: Full-stage live published creation + clean Details overlay.
 * - Compilation, formal verification, and sandboxing happen silently behind the scenes on Publish.
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type FC,
} from 'react';
import type { ThreadMessage } from '@assistant-ui/react';
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  Download,
  ExternalLink,
  Eye,
  Globe,
  Info,
  Loader2,
  Monitor,
  RotateCcw,
  Smartphone,
  Sparkles,
  Undo2,
  Upload,
  X,
} from 'lucide-react';
import { formatMessagesForLlama, stripReasoningTags } from '../engine/assistantAdapter';
import type { ObserverStreamBus, OtelTraceContext } from '../engine/observerStreamBus';
import type { PrismBuildWorkerClient } from '../engine/prismBuildClient';
import { compilePrismDeclaration } from '../engine/prismCompiler';
import type { PrismBridgeRuntime } from '../engine/prismHostPool';
import type {
  PrismStore,
  SealedHoloRecord,
  VerdantPrismProject,
} from '../engine/prismStudio';
import {
  checkDeclaration,
  evaluateDeclaredModel,
  type DeclarativeStarterId,
  type StudioApiOp,
} from '../engine/verdantStudioApi';
import { evaluateVerdantRouting } from '../engine/verdantStandardsRouter';
import {
  buildLocalInterviewExtractionPrompt,
  computeGuidedWizardState,
  mergeLocalModelExtraction,
  type ExtractedPrismIntent,
  type GuidedStepId,
} from '../engine/prismInterviewEngine';
import { PRISM_MEDIA_CATALOG } from '../engine/prismMediaOrgans';
import { isInternalBootstrapModel } from '../engine/modelCatalog';
import type { EngineState, LlamaVTensorWorkerClient } from '../engine/workerClient';
import {
  PrismSandboxFrame,
  type HoloviewMountConfig,
  type PrismRuntimeError,
} from './PrismSandboxFrame';
import { StudioGuidedWizard } from './StudioGuidedWizard';

interface VerdantStudioViewProps {
  project: VerdantPrismProject;
  mode: 'studio' | 'app';
  onChangeMode: (mode: 'studio' | 'app') => void;
  store: PrismStore;
  bus: ObserverStreamBus;
  bridgeRuntime: PrismBridgeRuntime;
  buildClient: PrismBuildWorkerClient;
  workerClient: LlamaVTensorWorkerClient;
  engineState: EngineState;
  darkMode: boolean;
  onEnableModelStream: () => void;
  onSelectProject: (prismId: string, mode: 'studio' | 'app') => void;
}

type PreviewPaneMode = 'preview' | 'details';

export const VerdantStudioView: FC<VerdantStudioViewProps> = ({
  project,
  mode,
  onChangeMode,
  store,
  bus,
  bridgeRuntime,
  buildClient,
  workerClient,
  engineState,
  darkMode,
  onSelectProject,
}) => {
  const [previewPaneMode, setPreviewPaneMode] = useState<PreviewPaneMode>('preview');
  const [guidedStep, setGuidedStep] = useState<GuidedStepId>('anchor-idea');
  const [detailsModalOpen, setDetailsModalOpen] = useState(false);
  const [viewportMode, setViewportMode] = useState<'desktop' | 'mobile'>('desktop');
  const [reloadNonce, setReloadNonce] = useState(0);
  const [runtimeError, setRuntimeError] = useState<PrismRuntimeError | null>(null);
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState(project.draft.name);
  const [justPublished, setJustPublished] = useState(false);
  const [publishing, setPublishing] = useState(false);
  const [statusBanner, setStatusBanner] = useState<{ ok: boolean; text: string } | null>(null);
  const [draggingFile, setDraggingFile] = useState(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const activeAgentAbortRef = useRef<AbortController | null>(null);
  const declaration = project.declaration ?? null;
  const isDeclarative = project.kind === 'declarative' || declaration !== null;
  const isExternalHolo = project.kind === 'external-holo';

  const activeHolo: SealedHoloRecord | null =
    mode === 'app'
      ? (project.publishedHolo ?? project.sealedDraft ?? null)
      : (project.sealedDraft ?? project.publishedHolo ?? null);

  const activeSnapshot = mode === 'app' && project.published ? project.published : project.draft;

  const compiled = useMemo(
    () => (declaration ? compilePrismDeclaration(declaration) : null),
    [declaration],
  );
  const checkReport = useMemo(
    () => (declaration ? checkDeclaration(declaration) : null),
    [declaration],
  );
  const routing = useMemo(
    () => (declaration ? evaluateVerdantRouting(declaration) : null),
    [declaration],
  );

  useEffect(() => {
    activeAgentAbortRef.current?.abort();
    activeAgentAbortRef.current = null;
    setTitleDraft(project.draft.name);
    setRuntimeError(null);
    setStatusBanner(null);
    setDetailsModalOpen(false);
    setPreviewPaneMode('preview');
    const initialRouting = project.declaration ? evaluateVerdantRouting(project.declaration) : null;
    const initialCheck = project.declaration ? checkDeclaration(project.declaration) : null;
    if (project.declaration && initialRouting && initialCheck) {
      setGuidedStep(computeGuidedWizardState(project.declaration, initialRouting, initialCheck).recommendedStep);
    } else {
      setGuidedStep('anchor-idea');
    }
  }, [project.id, project.draft.name, project.declaration?.schema]);

  useEffect(() => {
    if (!detailsModalOpen && previewPaneMode !== 'details') return;
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === 'Escape') {
        if (detailsModalOpen) setDetailsModalOpen(false);
        if (previewPaneMode === 'details') setPreviewPaneMode('preview');
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [detailsModalOpen, previewPaneMode]);

  // Closed-loop visual check: listen for rendered canvas sensory reports from this creation
  useEffect(() => {
    return bus.subscribe((ev) => {
      if (ev.endpointKind !== 'prism' || ev.endpointId !== project.id || ev.name !== 'sensory') {
        return;
      }
      const body = ev.body as { modality?: string; coherent?: boolean } | null;
      if (!body || body.modality !== 'vision') return;
      if (body.coherent === false) {
        setStatusBanner({
          ok: false,
          text: 'Visual check: low contrast detected — try brighter colors or a new variation.',
        });
      } else if (body.coherent === true) {
        setStatusBanner((prev) =>
          prev && prev.text.startsWith('Visual check:') ? null : prev,
        );
      }
    });
  }, [bus, project.id]);

  const holoviewConfig = useMemo<HoloviewMountConfig | null>(() => {
    const invokeModel = async (requestLine: string, traceContext?: OtelTraceContext): Promise<string> => {
      if (activeHolo && activeHolo.bytes.length > 0) {
        return buildClient.runHoloGuest({
          prismId: project.id,
          address: activeHolo.address,
          holoBytes: activeHolo.bytes,
          request: requestLine,
          parentContext: traceContext,
        });
      }
      if (declaration) {
        return evaluateDeclaredModel(declaration, requestLine).response;
      }
      throw new Error('Creation is not ready yet.');
    };

    if (isExternalHolo && activeHolo?.viewFiles) {
      const files = activeHolo.viewFiles;
      const entry = activeHolo.viewEntry ?? 'index.html';
      return {
        indexHtml: files[entry] ?? files['index.html'] ?? '',
        prismCss: files['prism.css'] ?? files['styles.css'] ?? '',
        prismUiJs: files['prism-ui.js'] ?? '',
        appJs: files['app.js'] ?? '',
        onInvoke: invokeModel,
      };
    }

    if (compiled) {
      return {
        indexHtml: compiled.appFiles['view/index.html'] ?? '',
        prismCss: compiled.compiledCss,
        prismUiJs: compiled.compiledUiJs,
        appJs: compiled.appFiles['view/app.js'] ?? '',
        onInvoke: invokeModel,
      };
    }

    return null;
  }, [activeHolo, buildClient, compiled, declaration, isExternalHolo, project.id]);

  const handleDispatchOp = useCallback(
    async (op: StudioApiOp, opts?: { readonly pushRevision?: boolean }): Promise<boolean> => {
      setStatusBanner(null);
      const res = await store.applyStudioOp(project.id, op, opts);
      if (!res.ok) {
        setStatusBanner({ ok: false, text: res.message });
        return false;
      }
      return true;
    },
    [project.id, store],
  );

  const handleLoadStarter = useCallback(
    (starter: DeclarativeStarterId) => {
      void store.createDeclarative(starter).then((created) => {
        onSelectProject(created.id, 'studio');
      });
    },
    [onSelectProject, store],
  );

  const handleCommitTitle = async () => {
    setEditingTitle(false);
    const trimmed = titleDraft.trim();
    if (!trimmed || trimmed === project.draft.name) {
      setTitleDraft(project.draft.name);
      return;
    }
    if (isDeclarative) {
      await handleDispatchOp({ op: 'set-metadata', name: trimmed });
    } else {
      await store.updateDraft(project.id, (prev) => ({ ...prev, name: trimmed }), { pushRevision: false });
    }
  };

  const handleUndo = async () => {
    if (isDeclarative) {
      await store.undoDeclaration(project.id);
    } else {
      await store.undoDraft(project.id);
    }
  };

  /** Silently verifies, compiles, and publishes in one smooth user action. */
  const handlePublish = useCallback(async () => {
    bridgeRuntime.evictPrism(project.id);
    setStatusBanner(null);

    if (!isDeclarative) {
      await store.publish(project.id);
      setJustPublished(true);
      setTimeout(() => setJustPublished(false), 2000);
      return;
    }

    if (!declaration || !compiled || (routing && !routing.readyToSeal)) {
      setStatusBanner({
        ok: false,
        text: 'Describe your idea in Studio first so we can finish building it.',
      });
      return;
    }

    setPublishing(true);
    try {
      let sealed = project.sealedDraft ?? null;
      if (!sealed) {
        const res = await buildClient.buildAndSeal({
          prismId: project.id,
          slug: declaration.metadata.slug,
          appFiles: compiled.appFiles,
        });
        if (!res.ok || !res.holo) {
          setStatusBanner({ ok: false, text: res.message || 'Could not publish right now.' });
          return;
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
        await store.saveSealedDraft(project.id, sealed);
      }
      await store.publishSealedHolo(project.id, sealed);
      setJustPublished(true);
      setStatusBanner({ ok: true, text: `"${project.draft.name}" is now published and ready to use.` });
      setTimeout(() => setJustPublished(false), 2000);
    } catch (error) {
      setStatusBanner({
        ok: false,
        text: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setPublishing(false);
    }
  }, [bridgeRuntime, buildClient, compiled, declaration, isDeclarative, project.draft.name, project.id, project.sealedDraft, routing, store]);

  const handleExportCreation = () => {
    const slug = (declaration?.metadata.slug || project.draft.name)
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-') || 'creation';
    if (activeHolo && activeHolo.bytes.length > 0) {
      const blob = new Blob([activeHolo.bytes.slice().buffer], { type: 'application/octet-stream' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${slug}.verdant`;
      a.click();
      URL.revokeObjectURL(url);
      return;
    }
    const json = store.exportHolo(project.id, mode === 'app');
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${slug}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const importCreationFromFile = useCallback(
    async (file: File) => {
      setStatusBanner(null);
      try {
        const buf = new Uint8Array(await file.arrayBuffer());
        if (buf[0] === 0x7b) {
          const text = new TextDecoder().decode(buf);
          const imported = await store.importHolo(text, false);
          onSelectProject(imported.id, 'studio');
          return;
        }
        setPublishing(true);
        const verified = await buildClient.verifyArchive({ prismId: project.id, holoBytes: buf });
        if (!verified.identitiesVerified) {
          setStatusBanner({ ok: false, text: `Could not verify ${file.name}.` });
          return;
        }
        const imported = await store.importBinaryHolo(buf, verified, true);
        onSelectProject(imported.id, 'app');
      } catch (error) {
        setStatusBanner({
          ok: false,
          text: `Import failed: ${error instanceof Error ? error.message : String(error)}`,
        });
      } finally {
        setPublishing(false);
      }
    },
    [buildClient, onSelectProject, project.id, store],
  );

  const handleImportFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (file) await importCreationFromFile(file);
  };

  const handleDropFile = async (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDraggingFile(false);
    const file = e.dataTransfer?.files?.[0];
    if (file && /\.(holo|verdant|json)$/i.test(file.name)) {
      await importCreationFromFile(file);
    }
  };

  const hasLocalModelReady =
    engineState.modelStreamEnabled &&
    engineState.model !== null &&
    !isInternalBootstrapModel(engineState.model.id);

  const handleSynthesizeIdeaWithLocalModel = useCallback(
    async (sentence: string, fallback: ExtractedPrismIntent): Promise<ExtractedPrismIntent> => {
      if (!hasLocalModelReady || !engineState.model) return fallback;
      const ac = new AbortController();
      const timeoutId = setTimeout(() => ac.abort(), 12000);
      try {
        const syntheticUser: ThreadMessage = {
          id: `idea_${Date.now()}`,
          role: 'user',
          createdAt: new Date(),
          content: [{ type: 'text', text: sentence }],
          attachments: [],
          metadata: { custom: {} },
        } as ThreadMessage;
        const prompt = formatMessagesForLlama(
          [syntheticUser],
          engineState.model,
          engineState.promptMode,
          buildLocalInterviewExtractionPrompt(sentence),
          { thinkingMode: 'off' },
        );
        let raw = '';
        for await (const ev of workerClient.streamGenerate(
          prompt,
          ac.signal,
          { ...engineState.sampling, temperature: 0.15, maxTokens: 512 },
          `studio-interview:${project.id}`,
        )) {
          raw = stripReasoningTags(ev.accumulatedText);
        }
        return mergeLocalModelExtraction(raw, fallback);
      } catch {
        return fallback;
      } finally {
        clearTimeout(timeoutId);
      }
    },
    [engineState.model, engineState.promptMode, engineState.sampling, hasLocalModelReady, project.id, workerClient],
  );

  useEffect(() => {
    return () => {
      activeAgentAbortRef.current?.abort();
      activeAgentAbortRef.current = null;
    };
  }, []);

  const handleCancelStudioAgent = useCallback(() => {
    activeAgentAbortRef.current?.abort();
    activeAgentAbortRef.current = null;
  }, []);

  const handleAppendStudioMessage = useCallback(
    async (msg: {
      readonly role: 'user' | 'assistant';
      readonly content: string;
      readonly badges?: readonly string[];
    }) => {
      await store.appendMessage(project.id, msg);
    },
    [project.id, store],
  );

  const handleRunStudioAgentPrompt = useCallback(
    async (
      systemPrompt: string,
      userPrompt: string,
      maxTokens: number,
      onProgress?: (charsGenerated: number) => void,
    ): Promise<string> => {
      if (!hasLocalModelReady || !engineState.model) {
        throw new Error('Local AI is not ready yet.');
      }
      activeAgentAbortRef.current?.abort();
      const ac = new AbortController();
      activeAgentAbortRef.current = ac;

      const syntheticUser: ThreadMessage = {
        id: `agent_${Date.now()}`,
        role: 'user',
        createdAt: new Date(),
        content: [{ type: 'text', text: userPrompt }],
        attachments: [],
        metadata: { custom: {} },
      } as ThreadMessage;
      const prompt = formatMessagesForLlama(
        [syntheticUser],
        engineState.model,
        engineState.promptMode,
        systemPrompt,
        { thinkingMode: 'off' },
      );
      let raw = '';
      let lastProgressAt = 0;
      try {
        for await (const ev of workerClient.streamGenerate(
          prompt,
          ac.signal,
          { ...engineState.sampling, temperature: 0.25, maxTokens },
          `studio-agent:${project.id}`,
        )) {
          if (ac.signal.aborted) break;
          raw = stripReasoningTags(ev.accumulatedText);
          const now = Date.now();
          if (onProgress && now - lastProgressAt >= 160) {
            lastProgressAt = now;
            onProgress(raw.length);
          }
        }
        if (ac.signal.aborted) {
          throw new DOMException('Cancelled.', 'AbortError');
        }
        onProgress?.(raw.length);
        return raw;
      } finally {
        if (activeAgentAbortRef.current === ac) {
          activeAgentAbortRef.current = null;
        }
      }
    },
    [engineState.model, engineState.promptMode, engineState.sampling, hasLocalModelReady, project.id, workerClient],
  );

  const effectivePreviewDark =
    compiled && !compiled.themeModes.includes('light') ? true : darkMode;

  const isPublished = project.published !== null;
  const isModified =
    isPublished &&
    (project.published!.digest !== project.draft.digest ||
      (project.publishedHolo?.address ?? null) !== (project.sealedDraft?.address ?? null));
  const canUndo = isDeclarative
    ? (project.declarationRevisions?.length ?? 0) > 0
    : project.revisions.length > 0;
  const mediaLabel = declaration?.view?.proceduralOrgan
    ? PRISM_MEDIA_CATALOG[declaration.view.proceduralOrgan.mediaType].label
    : 'Creation';

  const renderDetailsPanel = (onClose?: () => void) => (
    <div className="flex h-full min-h-0 min-w-0 flex-col overflow-hidden bg-white dark:bg-[#0B0F14]">
      <div className="flex shrink-0 items-center justify-between border-b border-zinc-200 px-4 py-2.5 dark:border-zinc-800">
        <div className="flex items-center gap-2">
          <Info className="h-4 w-4 text-emerald-600" />
          <span className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
            {project.draft.name}
          </span>
          <span className="rounded-full bg-zinc-100 px-2 py-0.5 text-[10px] font-medium text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300">
            {mediaLabel}
          </span>
        </div>
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="inline-flex items-center gap-1 rounded-md border border-zinc-200 px-2.5 py-1 text-xs font-medium text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-800"
          >
            <Eye className="h-3.5 w-3.5" />
            <span>Back</span>
          </button>
        )}
      </div>

      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4 text-xs">
        <div className="rounded-xl border border-zinc-200 p-3.5 dark:border-zinc-800">
          <p className="font-semibold text-zinc-900 dark:text-zinc-100">About</p>
          <p className="mt-1 leading-relaxed text-zinc-600 dark:text-zinc-400">
            {declaration?.concept.purpose ||
              project.draft.summary ||
              'Runs privately on your device inside VerdantUI.'}
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={handleExportCreation}
              className="inline-flex items-center gap-1.5 rounded-lg bg-zinc-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900"
            >
              <Download className="h-3.5 w-3.5" />
              Export
            </button>
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="inline-flex items-center gap-1.5 rounded-lg border border-zinc-200 px-3 py-1.5 text-xs font-medium text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-800"
            >
              <Upload className="h-3.5 w-3.5" />
              Import
            </button>
          </div>
        </div>

        {isPublished && (
          <div className="space-y-2.5 rounded-xl border border-zinc-200 p-3.5 dark:border-zinc-800">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-zinc-900 dark:text-zinc-100">
                Permissions
              </span>
              <button
                type="button"
                onClick={async () => {
                  bridgeRuntime.evictPrism(project.id);
                  await store.unpublish(project.id);
                  setDetailsModalOpen(false);
                  onChangeMode('studio');
                }}
                className="rounded px-2 py-0.5 text-[11px] font-medium text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-950/40"
              >
                Unpublish
              </button>
            </div>
            <p className="text-[11px] text-zinc-500">
              All permissions start off unless you enable them here.
            </p>
            <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-3">
              {(
                [
                  ['storage', 'Save data locally'],
                  ['browser', 'Downloads & alerts'],
                  ['network', 'Internet access'],
                ] as const
              ).map(([key, label]) => (
                <label
                  key={key}
                  className="flex cursor-pointer items-center justify-between rounded-lg border border-zinc-200 px-2.5 py-2 text-[11px] dark:border-zinc-800"
                >
                  <span>{label}</span>
                  <input
                    type="checkbox"
                    checked={activeSnapshot.grants[key]}
                    onChange={(e) => {
                      bridgeRuntime.evictPrism(project.id);
                      void store.updatePublishedGrants(project.id, (prev) => ({
                        ...prev,
                        [key]: e.target.checked,
                      }));
                    }}
                    className="h-3.5 w-3.5 accent-emerald-600"
                  />
                </label>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div
      className="relative flex h-full min-h-0 flex-col bg-[#F8F9FA] dark:bg-[#0B0F14]"
      onDragOver={(e) => {
        if (e.dataTransfer?.types?.includes('Files')) {
          e.preventDefault();
          if (!draggingFile) setDraggingFile(true);
        }
      }}
      onDragLeave={(e) => {
        if (e.currentTarget.contains(e.relatedTarget as Node | null)) return;
        setDraggingFile(false);
      }}
      onDrop={(e) => void handleDropFile(e)}
    >
      {draggingFile && (
        <div className="pointer-events-none absolute inset-0 z-40 flex items-center justify-center bg-emerald-950/50 backdrop-blur-xs">
          <div className="rounded-xl border-2 border-dashed border-emerald-400 bg-white/95 px-6 py-4 text-center shadow-xl dark:bg-zinc-900/95">
            <Upload className="mx-auto h-6 w-6 text-emerald-600" />
            <p className="mt-2 text-sm font-semibold text-zinc-900 dark:text-zinc-100">
              Drop file to open offline
            </p>
          </div>
        </div>
      )}
      <input
        ref={fileInputRef}
        type="file"
        accept=".holo,.verdant,.json,application/octet-stream,application/json"
        className="hidden"
        onChange={handleImportFileChange}
      />

      {/* Compact Focus Bar */}
      <header className="flex h-11 shrink-0 items-center justify-between gap-2 border-b border-zinc-200 bg-white px-3 dark:border-zinc-800 dark:bg-[#0E131B]">
        <div className="flex min-w-0 items-center gap-2">
          <span
            className={`h-2 w-2 shrink-0 rounded-full ${
              isPublished && !isModified
                ? 'bg-emerald-500'
                : isPublished
                  ? 'bg-amber-500'
                  : 'bg-zinc-400'
            }`}
            title={isPublished ? (isModified ? 'Unpublished edits' : 'Published') : 'Draft'}
          />
          {editingTitle ? (
            <input
              type="text"
              autoFocus
              value={titleDraft}
              onChange={(e) => setTitleDraft(e.target.value)}
              onBlur={() => void handleCommitTitle()}
              onKeyDown={(e) => {
                if (e.key === 'Enter') void handleCommitTitle();
                if (e.key === 'Escape') {
                  setTitleDraft(project.draft.name);
                  setEditingTitle(false);
                }
              }}
              className="w-44 rounded border border-emerald-500 bg-transparent px-1.5 py-0.5 text-sm font-semibold focus:outline-none"
            />
          ) : (
            <button
              type="button"
              onClick={() => setEditingTitle(true)}
              className="truncate text-sm font-semibold text-zinc-900 hover:underline dark:text-zinc-100"
              title="Click to rename"
            >
              {project.draft.name}
            </button>
          )}
        </div>

        <div className="flex shrink-0 items-center gap-1.5">
          {mode === 'studio' && (
            <>
              <button
                type="button"
                disabled={!canUndo}
                onClick={() => void handleUndo()}
                title="Undo last change"
                className="inline-flex h-7 items-center gap-1 rounded-md border border-zinc-200 px-2 text-xs text-zinc-600 hover:bg-zinc-100 disabled:opacity-40 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
              >
                <Undo2 className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Undo</span>
              </button>
              <button
                type="button"
                disabled={publishing}
                onClick={() => void handlePublish()}
                className="inline-flex h-7 items-center gap-1 rounded-md bg-emerald-700 px-2.5 text-xs font-medium text-white hover:bg-emerald-800 disabled:opacity-50"
              >
                {publishing ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : justPublished ? (
                  <Check className="h-3.5 w-3.5" />
                ) : (
                  <Globe className="h-3.5 w-3.5" />
                )}
                <span>
                  {publishing
                    ? 'Publishing…'
                    : justPublished
                      ? 'Published'
                      : isPublished
                        ? isModified
                          ? 'Update'
                          : 'Published'
                        : 'Publish'}
                </span>
              </button>
            </>
          )}

          {mode === 'app' && (
            <button
              type="button"
              aria-pressed={detailsModalOpen}
              onClick={() => setDetailsModalOpen((v) => !v)}
              className={`inline-flex h-7 items-center gap-1 rounded-md border px-2.5 text-xs font-medium transition-colors ${
                detailsModalOpen
                  ? 'border-emerald-600 bg-emerald-50 text-emerald-900 dark:border-emerald-500 dark:bg-emerald-950/60 dark:text-emerald-200'
                  : 'border-zinc-200 text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-800'
              }`}
            >
              <Info className="h-3.5 w-3.5" />
              {detailsModalOpen ? 'Back to App' : 'Details'}
            </button>
          )}

          {!isExternalHolo && (
            <div className="inline-flex rounded-lg border border-zinc-200 bg-zinc-100 p-0.5 text-[11px] dark:border-zinc-700 dark:bg-zinc-900">
              <button
                type="button"
                onClick={() => {
                  setDetailsModalOpen(false);
                  onChangeMode('studio');
                }}
                className={`rounded-md px-2 py-0.5 font-medium transition-colors ${
                  mode === 'studio'
                    ? 'bg-white text-zinc-900 shadow-xs dark:bg-zinc-800 dark:text-zinc-100'
                    : 'text-zinc-500 hover:text-zinc-800 dark:text-zinc-400'
                }`}
              >
                Studio
              </button>
              <button
                type="button"
                disabled={!isPublished}
                onClick={() => {
                  setDetailsModalOpen(false);
                  onChangeMode('app');
                }}
                className={`rounded-md px-2 py-0.5 font-medium transition-colors disabled:opacity-40 ${
                  mode === 'app'
                    ? 'bg-white text-zinc-900 shadow-xs dark:bg-zinc-800 dark:text-zinc-100'
                    : 'text-zinc-500 hover:text-zinc-800 dark:text-zinc-400'
                }`}
              >
                App
              </button>
            </div>
          )}
        </div>
      </header>

      {(publishing || statusBanner) && (
        <div
          className={`flex shrink-0 items-center justify-between gap-2 border-b px-3 py-1.5 text-xs ${
            publishing || statusBanner?.ok
              ? 'border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-900/50 dark:bg-emerald-950/40 dark:text-emerald-200'
              : 'border-red-200 bg-red-50 text-red-900 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-200'
          }`}
        >
          <div className="flex min-w-0 items-center gap-2">
            {publishing ? (
              <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" />
            ) : statusBanner?.ok ? (
              <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
            ) : (
              <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
            )}
            <span className="truncate">
              {publishing ? 'Publishing your creation…' : statusBanner?.text}
            </span>
          </div>
          {!publishing && statusBanner && (
            <button
              type="button"
              onClick={() => setStatusBanner(null)}
              className="shrink-0 rounded p-0.5 hover:bg-black/5"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      )}

      {mode === 'app' ? (
        <div className="relative min-h-0 min-w-0 flex-1 overflow-hidden">
          <div className={`h-full w-full ${detailsModalOpen ? 'invisible pointer-events-none' : ''}`}>
            <PrismSandboxFrame
              prismId={project.id}
              snapshot={activeSnapshot}
              holoview={holoviewConfig}
              darkMode={effectivePreviewDark}
              bus={bus}
              bridgeRuntime={bridgeRuntime}
              reloadNonce={reloadNonce}
              onRuntimeError={setRuntimeError}
              className="h-full w-full border-0"
            />
          </div>
          {detailsModalOpen && (
            <div className="absolute inset-0 z-20 flex flex-col overflow-hidden bg-white dark:bg-[#0B0F14]">
              <div className="flex shrink-0 items-center justify-end gap-2 border-b border-zinc-200 px-4 py-2 dark:border-zinc-800">
                {!isExternalHolo && (
                  <button
                    type="button"
                    onClick={() => {
                      setDetailsModalOpen(false);
                      onChangeMode('studio');
                    }}
                    className="inline-flex items-center gap-1 rounded-md border border-zinc-200 px-2.5 py-1 text-xs font-medium hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
                  >
                    <ExternalLink className="h-3 w-3" />
                    Edit in Studio
                  </button>
                )}
              </div>
              <div className="min-h-0 flex-1 overflow-hidden">
                {renderDetailsPanel(() => setDetailsModalOpen(false))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden lg:flex-row">
          {/* Left Pane: Conversational Studio Creator */}
          <div className="flex h-1/2 min-h-0 min-w-0 w-full flex-col overflow-hidden border-b border-zinc-200 lg:h-full lg:w-[440px] lg:shrink-0 lg:border-r lg:border-b-0 dark:border-zinc-800">
            {isDeclarative && declaration && checkReport && routing ? (
              <StudioGuidedWizard
                key={`guided-${project.id}`}
                declaration={declaration}
                routing={routing}
                checkReport={checkReport}
                messages={project.messages}
                onAppendMessage={handleAppendStudioMessage}
                activeStep={guidedStep}
                onSelectStep={setGuidedStep}
                onDispatchOp={handleDispatchOp}
                onSynthesizeWithLocalModel={handleSynthesizeIdeaWithLocalModel}
                onRunStudioAgentPrompt={handleRunStudioAgentPrompt}
                onCancelStudioAgent={handleCancelStudioAgent}
                hasLocalModelReady={hasLocalModelReady}
                engineStatus={engineState.status}
                engineProgressPercent={engineState.progressPercent}
                onOpenPreview={() => setPreviewPaneMode('preview')}
                onPublish={() => void handlePublish()}
                publishing={publishing}
              />
            ) : (
              <div className="flex h-full flex-col items-center justify-center gap-3 p-6 text-center text-xs text-zinc-500">
                <Sparkles className="h-8 w-8 text-emerald-600/70" />
                <p className="max-w-xs text-sm font-medium text-zinc-800 dark:text-zinc-200">
                  Imported Creation
                </p>
                <button
                  type="button"
                  onClick={() => handleLoadStarter('undefined')}
                  className="rounded-lg bg-emerald-700 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-800"
                >
                  Start a New Creation
                </button>
              </div>
            )}
          </div>

          {/* Right Pane: Live Preview & Simple Details */}
          <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
            <div className="relative z-20 flex shrink-0 items-center justify-between gap-2 border-b border-zinc-200 bg-white px-3 py-1.5 dark:border-zinc-800 dark:bg-[#0E131B]">
              <div className="inline-flex rounded-lg border border-zinc-200 bg-zinc-100 p-0.5 text-xs dark:border-zinc-700 dark:bg-zinc-900">
                <button
                  type="button"
                  aria-pressed={previewPaneMode === 'preview'}
                  onClick={() => setPreviewPaneMode('preview')}
                  className={`inline-flex items-center gap-1 rounded-md px-2.5 py-1 font-medium transition-colors ${
                    previewPaneMode === 'preview'
                      ? 'bg-white text-zinc-900 shadow-xs dark:bg-zinc-800 dark:text-zinc-100'
                      : 'text-zinc-500 hover:text-zinc-800 dark:text-zinc-400'
                  }`}
                >
                  <Eye className="h-3.5 w-3.5" />
                  Preview
                </button>
                <button
                  type="button"
                  aria-pressed={previewPaneMode === 'details'}
                  onClick={() =>
                    setPreviewPaneMode((m) => (m === 'details' ? 'preview' : 'details'))
                  }
                  className={`inline-flex items-center gap-1 rounded-md px-2.5 py-1 font-medium transition-colors ${
                    previewPaneMode === 'details'
                      ? 'bg-white text-zinc-900 shadow-xs dark:bg-zinc-800 dark:text-zinc-100'
                      : 'text-zinc-500 hover:text-zinc-800 dark:text-zinc-400'
                  }`}
                >
                  <Info className="h-3.5 w-3.5" />
                  Details
                </button>
              </div>

              <div className="flex items-center gap-1.5">
                {previewPaneMode === 'preview' && (!isDeclarative || routing?.readyToPreview) && (
                  <>
                    <button
                      type="button"
                      onClick={() => setViewportMode('desktop')}
                      className={`rounded p-1 ${viewportMode === 'desktop' ? 'bg-zinc-200/80 text-zinc-900 dark:bg-zinc-800 dark:text-zinc-100' : 'text-zinc-400'}`}
                      title="Desktop view"
                    >
                      <Monitor className="h-3.5 w-3.5" />
                    </button>
                    <button
                      type="button"
                      onClick={() => setViewportMode('mobile')}
                      className={`rounded p-1 ${viewportMode === 'mobile' ? 'bg-zinc-200/80 text-zinc-900 dark:bg-zinc-800 dark:text-zinc-100' : 'text-zinc-400'}`}
                      title="Mobile view"
                    >
                      <Smartphone className="h-3.5 w-3.5" />
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setRuntimeError(null);
                        setReloadNonce((n) => n + 1);
                      }}
                      className="rounded p-1 text-zinc-500 hover:bg-zinc-100 dark:hover:bg-zinc-800"
                      title="Reload preview"
                    >
                      <RotateCcw className="h-3.5 w-3.5" />
                    </button>
                  </>
                )}
              </div>
            </div>

            {runtimeError && (
              <div className="flex shrink-0 items-center justify-between gap-2 border-b border-red-200 bg-red-50 px-3 py-1.5 text-xs text-red-800 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-200">
                <span className="truncate">{runtimeError.message}</span>
                <button
                  type="button"
                  onClick={() => setRuntimeError(null)}
                  className="rounded px-1.5 py-0.5 text-[11px] hover:bg-red-100 dark:hover:bg-red-900/50"
                >
                  Dismiss
                </button>
              </div>
            )}

            <div className="relative min-h-0 flex-1 overflow-hidden bg-zinc-100/80 dark:bg-[#090D12]">
              {(!isDeclarative || !routing || routing.readyToPreview) ? (
                <div
                  className={`flex h-full w-full justify-center ${
                    previewPaneMode !== 'preview' ? 'invisible pointer-events-none' : ''
                  }`}
                >
                  <div
                    className={`h-full transition-all ${
                      viewportMode === 'mobile'
                        ? 'my-2 h-[calc(100%-16px)] w-[390px] overflow-hidden rounded-2xl border border-zinc-300 shadow-lg dark:border-zinc-700'
                        : 'w-full'
                    }`}
                  >
                    <PrismSandboxFrame
                      prismId={project.id}
                      snapshot={activeSnapshot}
                      holoview={holoviewConfig}
                      darkMode={effectivePreviewDark}
                      bus={bus}
                      bridgeRuntime={bridgeRuntime}
                      reloadNonce={reloadNonce}
                      onRuntimeError={setRuntimeError}
                      className="h-full w-full border-0"
                    />
                  </div>
                </div>
              ) : (
                previewPaneMode === 'preview' && (
                  <div className="flex h-full flex-col items-center justify-center p-8 text-center">
                    <div className="max-w-sm space-y-2.5">
                      <div className="mx-auto flex h-11 w-11 items-center justify-center rounded-2xl bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400">
                        <Sparkles className="h-5 w-5" />
                      </div>
                      <h2 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">
                        Your live preview appears here
                      </h2>
                      <p className="text-xs leading-relaxed text-zinc-500 dark:text-zinc-400">
                        Describe a game, picture, audio track, story, video, research space, or
                        everyday tool on the left to bring it to life immediately.
                      </p>
                    </div>
                  </div>
                )
              )}

              {previewPaneMode === 'details' && (
                <div className="absolute inset-0 z-10 overflow-hidden">
                  {renderDetailsPanel(() => setPreviewPaneMode('preview'))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
