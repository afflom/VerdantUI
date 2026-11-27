/**
 * Unified Procedurally Synthesized Prism View (`src/components/VerdantStudioView.tsx`)
 *
 * Renders EVERY user-owned Artifact Prism and Space Prism through the universal
 * `PrismSurfaceShell` contract (`DEC-011`, `DEC-012`, `DEC-013`, `DEC-014`, `DEC-015`):
 *   - Full-stage live Rendering (Preview) Panel (zero cramped side splits, zero stacked toolbars)
 *   - Single calm `Preview | Details` toggle
 *   - Homescreen-grade Natural Language Interface bar right beneath the canvas
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
  Globe,
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
import type {
  PrismSurfaceMode,
  SynthesizedPrismFrame,
} from '../engine/observerUiSynthesizer';
import type { ObserverStreamBus, OtelTraceContext } from '../engine/observerStreamBus';
import type { PrismBuildWorkerClient } from '../engine/prismBuildClient';
import { compilePrismDeclaration } from '../engine/prismCompiler';
import type { PrismHostPool } from '../engine/prismHostPool';
import type {
  PrismStore,
  SealedHoloRecord,
  VerdantPrismProject,
} from '../engine/prismStudio';
import {
  checkDeclaration,
  dispatchStudioOp,
  evaluateDeclaredModel,
  type StudioApiOp,
} from '../engine/verdantStudioApi';
import { evaluateVerdantRouting } from '../engine/verdantStandardsRouter';
import {
  buildLocalInterviewExtractionPrompt,
  computeGuidedWizardState,
  mergeLocalModelExtraction,
  parseIdeaSentence,
  reconcileWizardTurn,
  type ExtractedPrismIntent,
  type GuidedStepId,
} from '../engine/prismInterviewEngine';
import {
  assembleProceduralPrismOps,
  buildAgentCoreModelOps,
  buildWeightRefinementPrompt,
  buildWeightRoutingDiscoveryPrompt,
  classifyArtifactMediaType,
  parseDiscoveryAgentResponse,
  type AgentChatTurn,
} from '../engine/prismStudioAgent';
import { PRISM_MEDIA_CATALOG, type PrismMediaType } from '../engine/prismMediaOrgans';
import { isInternalBootstrapModel } from '../engine/modelCatalog';
import type { EngineState, LlamaVTensorWorkerClient } from '../engine/workerClient';
import {
  PrismSandboxFrame,
  type HoloviewMountConfig,
  type PrismRuntimeError,
} from './PrismSandboxFrame';
import { PrismSurfaceShell, type PrismSurfaceMessage } from './PrismSurfaceShell';

interface VerdantStudioViewProps {
  project: VerdantPrismProject;
  mode: 'studio' | 'app';
  frame: SynthesizedPrismFrame;
  surfaceMode: PrismSurfaceMode;
  onChangeSurfaceMode: (mode: PrismSurfaceMode) => void;
  onChangeMode: (mode: 'studio' | 'app') => void;
  onSelectSystemPrism: (prismId: string) => void;
  onTypingChange?: (typing: boolean) => void;
  store: PrismStore;
  bus: ObserverStreamBus;
  bridgeRuntime: PrismHostPool;
  buildClient: PrismBuildWorkerClient;
  workerClient: LlamaVTensorWorkerClient;
  engineState: EngineState;
  darkMode: boolean;
  onSelectProject: (prismId: string, mode: 'studio' | 'app') => void;
}

function isAbortError(err: unknown): boolean {
  return (
    (err instanceof DOMException && err.name === 'AbortError') ||
    (err instanceof Error && (err.name === 'AbortError' || /aborted|cancelled/i.test(err.message)))
  );
}

export const VerdantStudioView: FC<VerdantStudioViewProps> = ({
  project,
  mode,
  frame,
  surfaceMode,
  onChangeSurfaceMode,
  onChangeMode,
  onSelectSystemPrism,
  onTypingChange,
  store,
  bus,
  bridgeRuntime,
  buildClient,
  workerClient,
  engineState,
  darkMode,
  onSelectProject,
}) => {
  const [guidedStep, setGuidedStep] = useState<GuidedStepId>('anchor-idea');
  const [viewportMode, setViewportMode] = useState<'desktop' | 'mobile'>('desktop');
  const [reloadNonce, setReloadNonce] = useState(0);
  const [runtimeError, setRuntimeError] = useState<PrismRuntimeError | null>(null);
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState(project.draft.name);
  const [justPublished, setJustPublished] = useState(false);
  const [publishing, setPublishing] = useState(false);
  const [nlBusy, setNlBusy] = useState(false);
  const [nlBusyLabel, setNlBusyLabel] = useState('Updating…');
  const [statusBanner, setStatusBanner] = useState<{ ok: boolean; text: string } | null>(null);
  const [draggingFile, setDraggingFile] = useState(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const activeAgentAbortRef = useRef<AbortController | null>(null);
  const runTokenRef = useRef(0);
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
  const readiness = useMemo(
    () => (declaration ? evaluateVerdantRouting(declaration) : null),
    [declaration],
  );

  useEffect(() => {
    activeAgentAbortRef.current?.abort();
    activeAgentAbortRef.current = null;
    setTitleDraft(project.draft.name);
    setRuntimeError(null);
    setStatusBanner(null);
    const initialRouting = project.declaration ? evaluateVerdantRouting(project.declaration) : null;
    const initialCheck = project.declaration ? checkDeclaration(project.declaration) : null;
    if (project.declaration && initialRouting && initialCheck) {
      setGuidedStep(computeGuidedWizardState(project.declaration, initialRouting, initialCheck).recommendedStep);
    } else {
      setGuidedStep('anchor-idea');
    }
  }, [project.id, project.draft.name, project.declaration?.schema]);

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

  const handlePublish = useCallback(async () => {
    bridgeRuntime.evictPrism(project.id);
    setStatusBanner(null);

    if (!isDeclarative) {
      await store.publish(project.id);
      setJustPublished(true);
      setTimeout(() => setJustPublished(false), 2000);
      return;
    }

    if (!declaration || !compiled || (readiness && !readiness.readyToSeal)) {
      setStatusBanner({
        ok: false,
        text: 'Describe your idea in the prompt bar below first so we can finish building it.',
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
      setStatusBanner({ ok: true, text: `"${project.draft.name}" is now published.` });
      setTimeout(() => setJustPublished(false), 2000);
    } catch (error) {
      setStatusBanner({
        ok: false,
        text: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setPublishing(false);
    }
  }, [bridgeRuntime, buildClient, compiled, declaration, isDeclarative, project.draft.name, project.id, project.sealedDraft, readiness, store]);

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

  const handleCancelNl = useCallback(() => {
    runTokenRef.current += 1;
    activeAgentAbortRef.current?.abort();
    activeAgentAbortRef.current = null;
    setNlBusy(false);
  }, []);

  const handleRunStudioAgentPrompt = useCallback(
    async (systemPrompt: string, userPrompt: string, maxTokens: number): Promise<string> => {
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
      try {
        for await (const ev of workerClient.streamGenerate(
          prompt,
          ac.signal,
          { ...engineState.sampling, temperature: 0.25, maxTokens },
          `studio-agent:${project.id}`,
        )) {
          if (ac.signal.aborted) break;
          raw = stripReasoningTags(ev.accumulatedText);
        }
        if (ac.signal.aborted) throw new DOMException('Cancelled.', 'AbortError');
        return raw;
      } finally {
        if (activeAgentAbortRef.current === ac) activeAgentAbortRef.current = null;
      }
    },
    [engineState.model, engineState.promptMode, engineState.sampling, hasLocalModelReady, project.id, workerClient],
  );

  /**
   * Universal Natural Language handler for this Prism:
   * handles view toggles ("show details", "publish"), live document commands,
   * and full procedural Prism synthesis/refinement.
   */
  const handleSubmitNaturalLanguage = useCallback(
    async (rawMessage: string) => {
      const message = rawMessage.trim();
      if (!message || nlBusy) return;
      const lower = message.toLowerCase();

      await store.appendMessage(project.id, { role: 'user', content: message });

      // Direct Prism surface commands
      if (/^(?:show|open|view)\s+(?:prism\s+)?details\b/.test(lower)) {
        onChangeSurfaceMode('details');
        await store.appendMessage(project.id, { role: 'assistant', content: 'Switched to Prism Details.' });
        return;
      }
      if (/^(?:show|open|back\s+to)\s+preview\b/.test(lower)) {
        onChangeSurfaceMode('preview');
        await store.appendMessage(project.id, { role: 'assistant', content: 'Switched to live Preview.' });
        return;
      }
      if (/^publish\b/.test(lower)) {
        await handlePublish();
        await store.appendMessage(project.id, { role: 'assistant', content: `Published "${project.draft.name}".` });
        return;
      }

      const activeMediaType =
        declaration?.view?.proceduralOrgan?.mediaType ?? declaration?.concept.mediaType;

      // Live in-place Document commands (read/write toggle, add section, continue drafting)
      if (
        activeMediaType === 'written-work' &&
        project.published !== null &&
        /\b(read|reading|write|edit|add\s+(?:a\s+)?(?:chapter|section)|new\s+(?:chapter|section)|continue|draft|expand|tighten|rewrite|paragraph)\b/.test(lower)
      ) {
        setNlBusy(true);
        setNlBusyLabel('Updating document…');
        try {
          await bridgeRuntime.dispatchPublishedPrism(project.id, { prompt: message });
          onChangeSurfaceMode('preview');
          await store.appendMessage(project.id, {
            role: 'assistant',
            content: 'Updated your document live in Preview.',
          });
          return;
        } catch {
          // Fall through to declarative refinement if sandbox isn't accepting input
        } finally {
          setNlBusy(false);
        }
      }

      if (!declaration || !readiness || !checkReport) {
        await store.appendMessage(project.id, {
          role: 'assistant',
          content: 'This imported creation runs in read-only mode.',
        });
        return;
      }

      const nextHistory: AgentChatTurn[] = [
        ...project.messages.map((m) => ({
          role: m.role === 'user' ? ('user' as const) : ('studio' as const),
          text: m.content,
        })),
        { role: 'user', text: message },
      ];

      const classified = classifyArtifactMediaType(message);
      const isProceduralFlow =
        Boolean(declaration.view?.proceduralOrgan) ||
        Boolean(declaration.concept.mediaType) ||
        classified.mode === 'procedural-organ';

      if (isProceduralFlow) {
        const resolvedMediaType: PrismMediaType =
          declaration.view?.proceduralOrgan?.mediaType ??
          declaration.concept.mediaType ??
          classified.mediaType;

        const runToken = ++runTokenRef.current;
        const userTurns = nextHistory.filter((t) => t.role === 'user').length;
        const isRefining = Boolean(declaration.view?.proceduralOrgan);

        let rawModelJson = '';
        if (hasLocalModelReady) {
          setNlBusy(true);
          setNlBusyLabel(
            isRefining
              ? `Updating ${PRISM_MEDIA_CATALOG[resolvedMediaType].label.toLowerCase()}…`
              : `Creating ${PRISM_MEDIA_CATALOG[resolvedMediaType].label.toLowerCase()}…`,
          );
          try {
            const prompt = isRefining
              ? buildWeightRefinementPrompt(nextHistory, declaration, message)
              : buildWeightRoutingDiscoveryPrompt(nextHistory, declaration, userTurns, resolvedMediaType);
            rawModelJson = await handleRunStudioAgentPrompt(prompt, message, 768);
          } catch (err) {
            if (isAbortError(err) || runTokenRef.current !== runToken) {
              setNlBusy(false);
              await store.appendMessage(project.id, {
                role: 'assistant',
                content: 'Stopped. Send another prompt whenever you are ready.',
              });
              return;
            }
            rawModelJson = '';
          } finally {
            if (runTokenRef.current === runToken) setNlBusy(false);
          }
          if (runTokenRef.current !== runToken) return;
        }

        const { spec } = parseDiscoveryAgentResponse(
          rawModelJson,
          message,
          declaration,
          userTurns,
          resolvedMediaType,
        );

        if (spec.readyToBuild || isRefining) {
          const assembled = assembleProceduralPrismOps(declaration, spec, message);
          if (assembled.ops.length > 0) {
            await handleDispatchOp({ op: 'batch', ops: assembled.ops });
          }
          if (mode === 'app') onChangeMode('studio');
          await store.appendMessage(project.id, {
            role: 'assistant',
            content: isRefining
              ? `Updated "${spec.name}" and refreshed the preview.`
              : `"${spec.name}" is ready in Preview.`,
          });
          setGuidedStep('ready-to-seal');
          onChangeSurfaceMode('preview');
        } else {
          const coreOps = buildAgentCoreModelOps(declaration, spec, message);
          if (coreOps.length > 0) {
            let working = declaration;
            for (const op of coreOps) {
              const r = dispatchStudioOp(working, op);
              if (r.ok) working = r.declaration;
            }
            await handleDispatchOp({ op: 'batch', ops: coreOps });
          }
          await store.appendMessage(project.id, {
            role: 'assistant',
            content: spec.reply,
          });
          setGuidedStep('confirm-context');
        }
        return;
      }

      let preParsedIntent: ExtractedPrismIntent | undefined;
      if ((readiness.isUndefined || guidedStep === 'anchor-idea') && hasLocalModelReady) {
        const runToken = ++runTokenRef.current;
        setNlBusy(true);
        setNlBusyLabel('Understanding your idea…');
        try {
          const deterministic = parseIdeaSentence(message);
          preParsedIntent = await handleSynthesizeIdeaWithLocalModel(message, deterministic);
        } catch {
          preParsedIntent = parseIdeaSentence(message);
        } finally {
          if (runTokenRef.current === runToken) setNlBusy(false);
        }
      }

      const result = reconcileWizardTurn(declaration, guidedStep, message, preParsedIntent);
      if (result.ops.length > 0) {
        await handleDispatchOp({ op: 'batch', ops: result.ops });
      }
      await store.appendMessage(project.id, {
        role: 'assistant',
        content: result.assistantReply,
      });
      setGuidedStep(result.nextStep);
      if (result.openPreview) onChangeSurfaceMode('preview');
    },
    [
      activeSnapshot,
      bridgeRuntime,
      checkReport,
      declaration,
      guidedStep,
      handleDispatchOp,
      handlePublish,
      handleRunStudioAgentPrompt,
      handleSynthesizeIdeaWithLocalModel,
      hasLocalModelReady,
      mode,
      nlBusy,
      onChangeMode,
      onChangeSurfaceMode,
      project.draft.name,
      project.id,
      project.messages,
      readiness,
      store,
    ],
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

  const nlMessages = useMemo<readonly PrismSurfaceMessage[]>(
    () =>
      project.messages.map((m) => ({
        id: m.id,
        role: m.role,
        content: m.content,
      })),
    [project.messages],
  );

  const titleSlot = editingTitle ? (
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
      className="w-48 rounded-md border border-emerald-600 bg-transparent px-2 py-0.5 text-sm font-semibold text-slate-900 focus:outline-none dark:text-slate-100"
    />
  ) : (
    <button
      type="button"
      onClick={() => setEditingTitle(true)}
      className="truncate text-sm font-semibold tracking-tight text-slate-900 hover:underline dark:text-slate-100"
      title="Click to rename"
    >
      {project.draft.name}
    </button>
  );

  const statusDotClass =
    isPublished && !isModified
      ? 'bg-emerald-500'
      : isPublished
        ? 'bg-amber-500'
        : 'bg-slate-400';

  const headerActions = (
    <div className="flex items-center gap-1.5">
      {surfaceMode === 'preview' && (!isDeclarative || readiness?.readyToPreview) && (
        <div className="hidden items-center gap-0.5 sm:flex">
          <button
            type="button"
            onClick={() => setViewportMode('desktop')}
            className={`rounded-md p-1.5 transition-colors ${
              viewportMode === 'desktop'
                ? 'bg-slate-100 text-slate-900 dark:bg-slate-800 dark:text-slate-100'
                : 'text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
            }`}
            title="Desktop view"
            aria-label="Desktop view"
          >
            <Monitor className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            onClick={() => setViewportMode('mobile')}
            className={`rounded-md p-1.5 transition-colors ${
              viewportMode === 'mobile'
                ? 'bg-slate-100 text-slate-900 dark:bg-slate-800 dark:text-slate-100'
                : 'text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
            }`}
            title="Mobile view"
            aria-label="Mobile view"
          >
            <Smartphone className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            onClick={() => {
              setRuntimeError(null);
              setReloadNonce((n) => n + 1);
            }}
            className="rounded-md p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-700 dark:hover:bg-slate-800 dark:hover:text-slate-200"
            title="Reload preview"
            aria-label="Reload preview"
          >
            <RotateCcw className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      {canUndo && (
        <button
          type="button"
          onClick={() => void handleUndo()}
          title="Undo last change"
          className="inline-flex h-7 items-center gap-1 rounded-md border border-slate-200 px-2 text-xs text-slate-600 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800"
        >
          <Undo2 className="h-3.5 w-3.5" />
          <span className="hidden md:inline">Undo</span>
        </button>
      )}

      {!isExternalHolo && (!isPublished || isModified || justPublished || publishing) && (
        <button
          type="button"
          disabled={publishing}
          onClick={() => void handlePublish()}
          className="inline-flex h-7 items-center gap-1 rounded-md bg-[#0F5132] px-2.5 text-xs font-medium text-white hover:bg-[#0B3D25] disabled:opacity-50"
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
                  ? 'Update'
                  : 'Publish'}
          </span>
        </button>
      )}
    </div>
  );

  const banner = (publishing || statusBanner || runtimeError) ? (
    <div className="shrink-0">
      {(publishing || statusBanner) && (
        <div
          className={`flex items-center justify-between gap-2 border-b px-4 py-1.5 text-xs ${
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
      {runtimeError && (
        <div className="flex items-center justify-between gap-2 border-b border-red-200 bg-red-50 px-4 py-1.5 text-xs text-red-800 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-200">
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
    </div>
  ) : null;

  const previewContent =
    !isDeclarative || !readiness || readiness.readyToPreview ? (
      <div className="flex h-full w-full justify-center overflow-hidden bg-[#F8F9FA] dark:bg-[#0B0F14]">
        <div
          className={`h-full transition-all ${
            viewportMode === 'mobile'
              ? 'my-3 h-[calc(100%-24px)] w-[390px] overflow-hidden rounded-2xl border border-slate-300 shadow-lg dark:border-slate-700'
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
      <div className="flex h-full flex-col items-center justify-center p-8 text-center">
        <div className="max-w-md space-y-3">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400">
            <Sparkles className="h-5 w-5" />
          </div>
          <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
            Describe what you want to create
          </h2>
          <p className="text-sm leading-relaxed text-slate-500 dark:text-slate-400">
            Use the prompt bar below to describe a document, picture, video, audio track, game,
            research notebook, Space, or everyday utility.
          </p>
        </div>
      </div>
    );

  const detailsExtra = (
    <div className="space-y-4">
      <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-[#0F141C]">
        <div className="text-xs font-semibold text-slate-900 dark:text-slate-100">
          Portability
        </div>
        <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
          Export this Prism as a portable offline archive or import another saved creation.
        </p>
        <div className="mt-3 flex flex-wrap gap-2">
          <button
            type="button"
            onClick={handleExportCreation}
            className="inline-flex items-center gap-1.5 rounded-lg bg-slate-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-800 dark:bg-slate-100 dark:text-slate-900"
          >
            <Download className="h-3.5 w-3.5" />
            <span>Export</span>
          </button>
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-200 dark:hover:bg-slate-800"
          >
            <Upload className="h-3.5 w-3.5" />
            <span>Import</span>
          </button>
        </div>
      </div>

      {isPublished && (
        <div className="space-y-2.5 rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-[#0F141C]">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-900 dark:text-slate-100">
              Permissions
            </span>
            <button
              type="button"
              onClick={async () => {
                bridgeRuntime.evictPrism(project.id);
                await store.unpublish(project.id);
                onChangeMode('studio');
              }}
              className="rounded px-2 py-0.5 text-[11px] font-medium text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-950/40"
            >
              Unpublish
            </button>
          </div>
          <p className="text-[11px] text-slate-500 dark:text-slate-400">
            All permissions start off unless you enable them here.
          </p>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
            {(
              [
                ['storage', 'Save data locally'],
                ['browser', 'Downloads & alerts'],
                ['network', 'Internet access'],
              ] as const
            ).map(([key, label]) => (
              <label
                key={key}
                className="flex cursor-pointer items-center justify-between rounded-lg border border-slate-200 px-3 py-2 text-xs text-slate-700 dark:border-slate-800 dark:text-slate-200"
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
  );

  return (
    <div
      className="relative flex h-full min-h-0 flex-col"
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
          <div className="rounded-xl border-2 border-dashed border-emerald-400 bg-white/95 px-6 py-4 text-center shadow-xl dark:bg-slate-900/95">
            <Upload className="mx-auto h-6 w-6 text-emerald-600" />
            <p className="mt-2 text-sm font-semibold text-slate-900 dark:text-slate-100">
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

      <PrismSurfaceShell
        frame={frame}
        mode={surfaceMode}
        onChangeMode={onChangeSurfaceMode}
        titleSlot={titleSlot}
        statusDotClass={statusDotClass}
        headerActions={headerActions}
        banner={banner}
        previewContent={previewContent}
        detailsExtra={detailsExtra}
        onSelectChildPrism={onSelectSystemPrism}
        onSubmitNaturalLanguage={handleSubmitNaturalLanguage}
        onCancelNaturalLanguage={handleCancelNl}
        nlBusy={nlBusy}
        nlBusyLabel={nlBusyLabel}
        nlMessages={nlMessages}
        onTypingChange={onTypingChange}
      />
    </div>
  );
};
