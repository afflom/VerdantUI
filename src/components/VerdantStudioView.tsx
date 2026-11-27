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
  assembleProceduralPrismOps,
  buildAgentCoreModelOps,
  buildWeightRefinementPrompt,
  buildWeightRoutingDiscoveryPrompt,
  classifyArtifactMediaType,
  parseDiscoveryAgentResponse,
  type AgentChatTurn,
} from '../engine/prismStudioAgent';
import {
  PRISM_MEDIA_CATALOG,
  type AudioWeights,
  type PictureWeights,
  type PrismMediaType,
} from '../engine/prismMediaOrgans';
import { isInternalBootstrapModel } from '../engine/modelCatalog';
import type { StudioChatBridge } from '../engine/studioChatBridge';
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
  getBuildClient: () => Promise<PrismBuildWorkerClient>;
  workerClient: LlamaVTensorWorkerClient;
  engineState: EngineState;
  darkMode: boolean;
  getStudioChatBridge?: () => Promise<StudioChatBridge>;
  onSelectProject: (prismId: string, mode: 'studio' | 'app') => void;
}

const PIN_DISPLAY_LABELS: Readonly<Record<string, string>> = {
  bpm: 'Tempo',
  lighting: 'Lighting',
  palette: 'Palette',
  scale: 'Scale',
  style: 'Style',
  coverFormat: 'Format',
  generator: 'Scene',
};

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
  getBuildClient,
  workerClient,
  engineState,
  darkMode,
  getStudioChatBridge,
  onSelectProject,
}) => {
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
        const buildClient = await getBuildClient();
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
  }, [activeHolo, getBuildClient, compiled, declaration, isExternalHolo, project.id]);

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
      // Automatic pre-publish visual / audio quality check
      let qualityNote = '';
      const organ = declaration.view?.proceduralOrgan;
      if (organ?.mediaType === 'picture') {
        const { renderPictureWithCoherenceCheck } = await import('../engine/proceduralPictureRenderer');
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
        } = await import('../engine/audioSynthesisEngine');
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
        const buildClient = await getBuildClient();
        const res = await workerClient.runWithDeterministicPrecision(() =>
          buildClient.buildAndSeal({
            prismId: project.id,
            slug: declaration.metadata.slug,
            appFiles: compiled.appFiles,
          }),
        );
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
      setStatusBanner({
        ok: true,
        text: qualityNote
          ? `"${project.draft.name}" published${qualityNote}.`
          : `"${project.draft.name}" verified and published.`,
      });
      setTimeout(() => setJustPublished(false), 2000);
    } catch (error) {
      setStatusBanner({
        ok: false,
        text: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setPublishing(false);
    }
  }, [bridgeRuntime, getBuildClient, compiled, declaration, isDeclarative, project.draft.name, project.id, project.sealedDraft, readiness, store, workerClient]);

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
        const buildClient = await getBuildClient();
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
    [getBuildClient, onSelectProject, project.id, store],
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

      // Companion media derived directly from this Prism's features ("Generate cover art for this", "Make a soundtrack for this")
      const companionMatch = message.match(
        /\b(?:generate|create|make|compose|design|add)\s+(?:a\s+|an\s+)?(cover\s+art|book\s+cover|album\s+cover|game\s+cover|logo|brand\s*mark|soundtrack|theme\s+song)\b/i,
      );
      if (companionMatch && getStudioChatBridge) {
        const requestedKind = companionMatch[1]!.toLowerCase();
        const isAudio = /soundtrack|theme\s+song/.test(requestedKind);
        const isLogo = /logo|brand\s*mark/.test(requestedKind);
        setNlBusy(true);
        setNlBusyLabel(isAudio ? 'Composing soundtrack…' : isLogo ? 'Designing logo…' : 'Creating cover art…');
        try {
          const bridge = await getStudioChatBridge();
          const companion = await bridge.refinePrismFromPrompt({
            target: project.id,
            instruction: message,
          });
          await store.appendMessage(project.id, {
            role: 'assistant',
            content: `Created **${companion.prismName}** from this Prism's features and saved it to ${companion.galleryLabel}.`,
          });
          onSelectProject(companion.prismId, 'app');
          return;
        } finally {
          setNlBusy(false);
        }
      }

      // Live in-place organ commands (Document read/write/draft; Audio/Video playback; Picture new variation)
      if (
        project.published !== null &&
        ((activeMediaType === 'written-work' &&
          /\b(read|reading|write|edit|add\s+(?:a\s+)?(?:chapter|section)|new\s+(?:chapter|section)|continue|draft|expand|tighten|rewrite|paragraph)\b/.test(lower)) ||
          ((activeMediaType === 'audio' || activeMediaType === 'video') &&
            /^(?:play|pause|stop|resume|scene\s+\d+)$/i.test(message)) ||
          (activeMediaType === 'picture' && /^(?:new\s+variation|another\s+variation|shuffle|re-?roll)\b/i.test(message)))
      ) {
        setNlBusy(true);
        setNlBusyLabel(activeMediaType === 'written-work' ? 'Updating document…' : 'Updating live preview…');
        try {
          await bridgeRuntime.dispatchPublishedPrism(project.id, { prompt: message });
          onChangeSurfaceMode('preview');
          await store.appendMessage(project.id, {
            role: 'assistant',
            content:
              activeMediaType === 'written-work'
                ? 'Updated your document live in Preview.'
                : 'Updated live Preview.',
          });
          return;
        } catch {
          // Fall through to declarative refinement if sandbox isn't accepting input
        } finally {
          setNlBusy(false);
        }
      }

      // Pin-preserving natural-language refinement for any existing procedural Prism via StudioChatBridge
      if (declaration?.view?.proceduralOrgan && getStudioChatBridge) {
        setNlBusy(true);
        setNlBusyLabel(`Refining ${PRISM_MEDIA_CATALOG[declaration.view.proceduralOrgan.mediaType].label.toLowerCase()}…`);
        try {
          const bridge = await getStudioChatBridge();
          const refined = await bridge.refinePrismFromPrompt({
            target: project.id,
            instruction: message,
          });
          await store.appendMessage(project.id, {
            role: 'assistant',
            content: `Updated "${refined.prismName}" while preserving your pinned preferences.`,
          });
          onChangeSurfaceMode('preview');
          return;
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
      }
    },
    [
      activeSnapshot,
      bridgeRuntime,
      checkReport,
      declaration,
      getStudioChatBridge,
      handleDispatchOp,
      handlePublish,
      handleRunStudioAgentPrompt,
      hasLocalModelReady,
      mode,
      nlBusy,
      onChangeMode,
      onChangeSurfaceMode,
      onSelectProject,
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

  const seedText = declaration?.concept.seedText?.trim() || '';
  const pinnedFields = declaration?.concept.pinnedFields ?? [];
  const connectedPrisms = useMemo(() => {
    const all = store.list();
    const directIds = new Set(declaration?.concept.linkedPrismIds ?? []);
    const connected: VerdantPrismProject[] = [];
    const seen = new Set<string>([project.id]);
    for (const p of all) {
      if (seen.has(p.id)) continue;
      const linksBack = p.declaration?.concept.linkedPrismIds?.includes(project.id) ?? false;
      if (directIds.has(p.id) || linksBack) {
        seen.add(p.id);
        connected.push(p);
      }
    }
    return connected;
  }, [declaration?.concept.linkedPrismIds, project.id, store]);

  const handleUnpinField = useCallback(
    async (field: string) => {
      const nextPins = pinnedFields.filter((f) => f !== field);
      await handleDispatchOp({ op: 'set-concept-profile', pinnedFields: nextPins });
      if (project.published) {
        await store.publish(project.id);
      }
    },
    [handleDispatchOp, pinnedFields, project.id, project.published, store],
  );

  const detailsExtra = (
    <div className="space-y-4">
      {seedText && (
        <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-[#0F141C]">
          <div className="text-xs font-semibold text-slate-900 dark:text-slate-100">
            Original Prompt
          </div>
          <p className="mt-1.5 text-xs leading-relaxed text-slate-600 dark:text-slate-300">
            “{seedText}”
          </p>
        </div>
      )}

      {pinnedFields.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-[#0F141C]">
          <div className="text-xs font-semibold text-slate-900 dark:text-slate-100">
            Pinned Preferences
          </div>
          <p className="mt-1 text-[11px] text-slate-500 dark:text-slate-400">
            Kept steady when you refine in natural language. Click any item to unpin it.
          </p>
          <div className="mt-2.5 flex flex-wrap gap-1.5">
            {pinnedFields.map((field) => (
              <button
                key={field}
                type="button"
                data-testid={`unpin-${field}`}
                onClick={() => void handleUnpinField(field)}
                className="inline-flex items-center gap-1 rounded-full border border-emerald-700/25 bg-emerald-50 px-2.5 py-1 text-[11px] font-medium text-emerald-900 hover:bg-emerald-100 dark:border-emerald-500/30 dark:bg-emerald-950/50 dark:text-emerald-200 dark:hover:bg-emerald-900/60"
                title={`Unpin ${PIN_DISPLAY_LABELS[field] ?? field}`}
              >
                <span>{PIN_DISPLAY_LABELS[field] ?? field}</span>
                <X className="h-3 w-3 opacity-70" />
              </button>
            ))}
          </div>
        </div>
      )}

      {connectedPrisms.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-[#0F141C]">
          <div className="text-xs font-semibold text-slate-900 dark:text-slate-100">
            Connected Creations
          </div>
          <div className="mt-2.5 grid grid-cols-1 gap-2 sm:grid-cols-2">
            {connectedPrisms.map((linked) => {
              const linkedType =
                linked.declaration?.view?.proceduralOrgan?.mediaType ??
                linked.declaration?.concept.mediaType;
              const typeLabel = linkedType ? PRISM_MEDIA_CATALOG[linkedType].label : 'Creation';
              return (
                <button
                  key={linked.id}
                  type="button"
                  data-testid={`connected-prism-${linked.id}`}
                  onClick={() => onSelectProject(linked.id, 'app')}
                  className="flex items-center justify-between gap-2 rounded-lg border border-slate-200 bg-slate-50/70 px-3 py-2 text-left transition-colors hover:border-emerald-600/40 hover:bg-emerald-50/50 dark:border-slate-800 dark:bg-[#141B26] dark:hover:border-emerald-500/40"
                >
                  <div className="min-w-0">
                    <div className="truncate text-xs font-semibold text-slate-900 dark:text-slate-100">
                      {linked.published?.name ?? linked.draft.name}
                    </div>
                    <div className="text-[11px] text-slate-500 dark:text-slate-400">
                      {typeLabel}
                    </div>
                  </div>
                  <span className="shrink-0 text-[11px] font-medium text-emerald-700 dark:text-emerald-400">
                    Open
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}

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
