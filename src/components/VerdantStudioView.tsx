/**
 * Verdant Studio & Published Prism App Workspace (`src/components/VerdantStudioView.tsx`)
 *
 * The declarative GUI experience for the `afflom/verdant-studio` API and the
 * `uor.foundation` Prism skill (`prism-code@0.11.5`):
 * - Collapsed ~40px Prism Focus Bar (editable title, Verified/Draft/Modified status,
 *   short BLAKE3/SHA address, Undo, Check & Build [real WASM LexLean + Lean 4.32.2],
 *   Publish, and Studio | App toggle)
 * - Studio Mode: Split Declarative API Builder (`StudioDeclarativeBuilder`) + Real-Time
 *   Compiled `prism.design/1` Preview with the `Preview | Prism` segmented toggle right
 *   on the Preview toolbar
 * - App Mode: Full-stage live Published/Imported Prism backed by verified WASM guest +
 *   clean centered Prism Package/Verification modal
 */

import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent, type DragEvent, type FC } from 'react';
import type { ThreadMessage } from '@assistant-ui/react';
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  Code2,
  Compass,
  Copy,
  Download,
  ExternalLink,
  Eye,
  FileCheck2,
  Globe,
  Layers,
  Loader2,
  Monitor,
  Play,
  RotateCcw,
  ShieldCheck,
  SlidersHorizontal,
  Smartphone,
  Sparkles,
  Undo2,
  Upload,
  X,
} from 'lucide-react';
import { formatMessagesForLlama, stripReasoningTags } from '../engine/assistantAdapter';
import type { ObserverStreamBus, OtelTraceContext } from '../engine/observerStreamBus';
import type { PrismBuildStage, PrismBuildWorkerClient } from '../engine/prismBuildClient';
import { compilePrismDeclaration } from '../engine/prismCompiler';
import type { PrismBridgeRuntime } from '../engine/prismHostPool';
import {
  buildStudioApiProposerPrompt,
  extractStudioApiOpsFromModelOutput,
  shortPrismDigest,
  type PrismStore,
  type SealedHoloRecord,
  type VerdantPrismProject,
} from '../engine/prismStudio';
import {
  checkDeclaration,
  computeAgenda,
  evaluateDeclaredModel,
  type DeclarativeStarterId,
  type StudioApiOp,
  type StudioWizardStage,
} from '../engine/verdantStudioApi';
import { evaluateVerdantRouting } from '../engine/verdantStandardsRouter';
import {
  buildLocalInterviewExtractionPrompt,
  computeGuidedWizardState,
  mergeLocalModelExtraction,
  type ExtractedPrismIntent,
  type GuidedStepId,
} from '../engine/prismInterviewEngine';
import { isInternalBootstrapModel } from '../engine/modelCatalog';
import type { EngineState, LlamaVTensorWorkerClient } from '../engine/workerClient';
import { ModelBlueprintCanvas } from './ModelBlueprintCanvas';
import { PrismSandboxFrame, type HoloviewMountConfig, type PrismRuntimeError } from './PrismSandboxFrame';
import { StudioDeclarativeBuilder } from './StudioDeclarativeBuilder';
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

type PrismPanelTab = 'overview' | 'package' | 'compiled';
type PreviewPaneMode = 'blueprint' | 'preview' | 'prism';

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
  onEnableModelStream,
  onSelectProject,
}) => {
  const [previewPaneMode, setPreviewPaneMode] = useState<PreviewPaneMode>('blueprint');
  const [leftPaneMode, setLeftPaneMode] = useState<'guided' | 'advanced'>('guided');
  const [guidedStep, setGuidedStep] = useState<GuidedStepId>('anchor-idea');
  const [wizardStage, setWizardStage] = useState<StudioWizardStage>('concept');
  const [appPrismModalOpen, setAppPrismModalOpen] = useState(false);
  const [panelTab, setPanelTab] = useState<PrismPanelTab>('overview');
  const [selectedCompiledFile, setSelectedCompiledFile] = useState('app/src/App.lex.tex');
  const [digestCopied, setDigestCopied] = useState(false);
  const [viewportMode, setViewportMode] = useState<'desktop' | 'mobile'>('desktop');
  const [reloadNonce, setReloadNonce] = useState(0);
  const [runtimeError, setRuntimeError] = useState<PrismRuntimeError | null>(null);
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState(project.draft.name);
  const [justPublished, setJustPublished] = useState(false);

  // Build / Verify WASM Worker state
  const [building, setBuilding] = useState(false);
  const [buildStage, setBuildStage] = useState<PrismBuildStage | null>(null);
  const [buildDetail, setBuildDetail] = useState<string>('');
  const [buildBanner, setBuildBanner] = useState<{ ok: boolean; text: string } | null>(null);

  // Declarative API error & optional NL->Ops proposal state
  const [apiError, setApiError] = useState<{ code: string; message: string; remediation: string } | null>(null);
  const [proposingOps, setProposingOps] = useState(false);
  const [proposedOps, setProposedOps] = useState<readonly StudioApiOp[] | null>(null);

  // Interactive test runner for Package / External Holo
  const [testRequestLine, setTestRequestLine] = useState('');
  const [testAnswer, setTestAnswer] = useState<string | null>(null);
  const [testRunning, setTestRunning] = useState(false);
  const [draggingHolo, setDraggingHolo] = useState(false);
  const [compiledCopied, setCompiledCopied] = useState(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const declaration = project.declaration ?? null;
  const isDeclarative = project.kind === 'declarative' || declaration !== null;
  const isExternalHolo = project.kind === 'external-holo';

  const activeHolo: SealedHoloRecord | null =
    mode === 'app'
      ? (project.publishedHolo ?? project.sealedDraft ?? null)
      : (project.sealedDraft ?? project.publishedHolo ?? null);

  const activeSnapshot = mode === 'app' && project.published ? project.published : project.draft;

  // Mechanically compile declaration for instant <10ms preview + compiled files viewer
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
  const agenda = useMemo(
    () => (declaration ? computeAgenda(declaration) : []),
    [declaration],
  );

  // Sync local title & reset ephemeral state on project switch
  useEffect(() => {
    setTitleDraft(project.draft.name);
    setRuntimeError(null);
    setBuildBanner(null);
    setApiError(null);
    setProposedOps(null);
    setTestAnswer(null);
    setAppPrismModalOpen(false);
    const initialRouting = project.declaration ? evaluateVerdantRouting(project.declaration) : null;
    const initialCheck = project.declaration ? checkDeclaration(project.declaration) : null;
    setPreviewPaneMode(initialRouting && !initialRouting.readyToPreview ? 'blueprint' : 'preview');
    setWizardStage(initialRouting ? initialRouting.recommendedStage : 'concept');
    if (project.declaration && initialRouting && initialCheck) {
      setGuidedStep(computeGuidedWizardState(project.declaration, initialRouting, initialCheck).recommendedStep);
    } else {
      setGuidedStep('anchor-idea');
    }
  }, [project.id, project.draft.name, project.declaration?.schema]);

  // Escape closes Prism inspector back to Preview (if ready) or Blueprint (Studio mode) / App (App mode)
  useEffect(() => {
    if (!appPrismModalOpen && previewPaneMode !== 'prism') return;
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === 'Escape') {
        if (appPrismModalOpen) setAppPrismModalOpen(false);
        if (previewPaneMode === 'prism') {
          setPreviewPaneMode(routing?.readyToPreview ? 'preview' : 'blueprint');
        }
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [appPrismModalOpen, previewPaneMode, routing?.readyToPreview]);

  // Build `holoview` mount config when we have a compiled declaration or external binary .holo
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
      throw new Error('No compiled Prism model available.');
    };

    // Prefer decoded HOLOVIEW from an external binary .holo archive
    if (isExternalHolo && activeHolo?.viewFiles) {
      const files = activeHolo.viewFiles;
      const entry = activeHolo.viewEntry ?? 'index.html';
      return {
        indexHtml: files[entry] ?? files['index.html'] ?? '',
        prismCss: files['prism.css'] ?? '',
        prismUiJs: files['prism-ui.js'] ?? '',
        appJs: files['app.js'] ?? '',
        onInvoke: invokeModel,
      };
    }

    // Otherwise mount the mechanically compiled `prism.design/1` page
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
      setApiError(null);
      setBuildBanner(null);
      const res = await store.applyStudioOp(project.id, op, opts);
      if (!res.ok) {
        setApiError({ code: res.code, message: res.message, remediation: res.remediation });
        bus.emit({
          name: `studio.api.refused`,
          endpointKind: 'studio',
          endpointId: project.id,
          status: 'error',
          attributes: { 'verdant.api.op': op.op, 'verdant.api.code': res.code },
          body: res.message,
        });
        return false;
      }
      bus.emit({
        name: `studio.api.${op.op}`,
        endpointKind: 'studio',
        endpointId: project.id,
        attributes: { 'verdant.api.op': op.op },
        body: res.message,
      });
      return true;
    },
    [bus, project.id, store],
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

  /** Runs real LexLean WASM + Lean 4.32.2 WASM kernel + binary `prism/holo/6` seal in the Web Worker. */
  const handleCheckAndBuild = useCallback(async (): Promise<SealedHoloRecord | null> => {
    if (!declaration) return null;
    if (routing && !routing.readyToSeal) {
      const nextStageReport = routing.stages.find((s) => s.stage === routing.recommendedStage);
      setWizardStage(routing.recommendedStage);
      setBuildBanner({
        ok: false,
        text: `Complete ${nextStageReport?.label ?? 'Core Models'} before building: ${nextStageReport?.missing[0] ?? 'Model incomplete.'}`,
      });
      return null;
    }
    if (!compiled) {
      setWizardStage('concept');
      setBuildBanner({
        ok: false,
        text: 'Define Concept, System, Security, and Design models before building.',
      });
      return null;
    }
    if (checkReport && !checkReport.ok) {
      const first = checkReport.diagnostics[0];
      setBuildBanner({
        ok: false,
        text: `Fix before building: ${first?.message ?? 'Declaration check failed.'}`,
      });
      return null;
    }

    setBuilding(true);
    setBuildStage('loading-toolchain');
    setBuildDetail('Preparing BLAKE3-verified WASM toolchain…');
    setBuildBanner(null);

    try {
      const res = await buildClient.buildAndSeal({
        prismId: project.id,
        slug: declaration.metadata.slug,
        appFiles: compiled.appFiles,
        onStage: (stage, detail) => {
          setBuildStage(stage);
          setBuildDetail(detail);
        },
      });

      if (!res.ok || !res.holo) {
        const reason = res.message || 'Build refused.';
        setBuildBanner({ ok: false, text: reason });
        await store.applyStudioOp(
          project.id,
          {
            op: 'record-error',
            error: {
              stage: res.stage === 'prove' ? 'prove' : res.stage === 'check' ? 'check' : 'build',
              code: res.diagnostics[0]?.code ?? `build.${res.stage}`,
              refusal: reason,
            },
          },
          { pushRevision: false },
        );
        return null;
      }

      const sealed: SealedHoloRecord = {
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
      setBuildBanner({
        ok: true,
        text: `Verified & sealed prism/holo/6 (${(sealed.bytes.length / 1024).toFixed(1)} KB · ${sealed.theorems.length} proved · ${sealed.vectorsPassed}/${sealed.vectorsTotal} examples · ${res.ms}ms)`,
      });
      return sealed;
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      setBuildBanner({ ok: false, text: msg });
      return null;
    } finally {
      setBuilding(false);
      setBuildStage(null);
      setBuildDetail('');
    }
  }, [buildClient, checkReport, compiled, declaration, project.id, routing, store]);

  const handlePublish = async () => {
    bridgeRuntime.evictPrism(project.id);
    if (isDeclarative) {
      let sealed = project.sealedDraft ?? null;
      if (!sealed) {
        sealed = await handleCheckAndBuild();
        if (!sealed) return;
      }
      await store.publishSealedHolo(project.id, sealed);
    } else {
      await store.publish(project.id);
    }
    setJustPublished(true);
    setTimeout(() => setJustPublished(false), 2000);
  };

  const handleExportHolo = () => {
    if (activeHolo && activeHolo.bytes.length > 0) {
      const blob = new Blob([activeHolo.bytes.slice().buffer], { type: 'application/octet-stream' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${declaration?.metadata.slug ?? 'prism'}.holo`;
      a.click();
      URL.revokeObjectURL(url);
      return;
    }
    const json = store.exportHolo(project.id, mode === 'app');
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${project.draft.name.toLowerCase().replace(/[^a-z0-9]+/g, '-') || 'prism'}.holo.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const importHoloFromFile = useCallback(
    async (file: File) => {
      setBuildBanner(null);
      try {
        const buf = new Uint8Array(await file.arrayBuffer());
        const startsWithBrace = buf[0] === 0x7b;
        if (startsWithBrace) {
          const text = new TextDecoder().decode(buf);
          const imported = await store.importHolo(text, false);
          onSelectProject(imported.id, 'studio');
          return;
        }
        setBuilding(true);
        setBuildStage('verifying');
        setBuildDetail(`Verifying ${file.name}…`);
        const verified = await buildClient.verifyArchive({ prismId: project.id, holoBytes: buf });
        if (!verified.identitiesVerified) {
          setBuildBanner({ ok: false, text: `Refused ${file.name}: BLAKE3 archive integrity failed.` });
          return;
        }
        const imported = await store.importBinaryHolo(buf, verified, true);
        setBuildBanner({
          ok: true,
          text: `Verified ${file.name} (${verified.vectorsPassed}/${verified.vectorsTotal} vectors passed).`,
        });
        onSelectProject(imported.id, 'app');
      } catch (error) {
        setBuildBanner({
          ok: false,
          text: `Import failed: ${error instanceof Error ? error.message : String(error)}`,
        });
      } finally {
        setBuilding(false);
        setBuildStage(null);
        setBuildDetail('');
      }
    },
    [buildClient, onSelectProject, project.id, store],
  );

  const handleImportFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (file) await importHoloFromFile(file);
  };

  const handleDropHolo = async (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDraggingHolo(false);
    const file = e.dataTransfer?.files?.[0];
    if (file && /\.(holo|json)$/i.test(file.name)) {
      await importHoloFromFile(file);
    }
  };

  const handleReverifyOffline = async () => {
    if (!activeHolo) return;
    setBuilding(true);
    setBuildStage('verifying');
    setBuildDetail('Verifying BLAKE3 identities, replaying vectors & re-proving with Lean WASM…');
    try {
      const res = await buildClient.verifyArchive({
        prismId: project.id,
        holoBytes: activeHolo.bytes,
        reprove: activeHolo.carries.snapshot,
        onStage: (stage, detail) => {
          setBuildStage(stage);
          setBuildDetail(detail);
        },
      });
      if (res.reproved) {
        const updatedSealed: SealedHoloRecord = {
          ...activeHolo,
          proved: res.reproved.passed,
          theorems: res.reproved.theorems.map((t) => ({
            theorem: t.name.split('.').pop() ?? t.name,
            status: t.status,
            axioms: t.axioms,
          })),
          vectorsPassed: res.vectorsPassed,
          vectorsTotal: res.vectorsTotal,
        };
        await store.saveSealedDraft(project.id, updatedSealed);
      }
      const reproveMsg = res.reproved
        ? res.reproved.passed
          ? ` · Lean re-proved ${res.reproved.theorems.length} theorem(s)`
          : ` · Re-proof failed (${res.reproved.reason ?? 'refused'})`
        : '';
      const srcMsg =
        res.sourcesMatch === true ? ' · sources match snapshot' : res.sourcesMatch === false ? ' · sources mismatch' : '';
      setBuildBanner({
        ok: res.ok,
        text: res.ok
          ? `Offline verification PASSED · ${res.address.slice(0, 19)}… · ${res.vectorsPassed}/${res.vectorsTotal} vectors${reproveMsg}${srcMsg}`
          : `Offline verification FAILED · ${res.vectorsPassed}/${res.vectorsTotal} vectors${reproveMsg}${srcMsg}`,
      });
    } catch (err) {
      setBuildBanner({ ok: false, text: err instanceof Error ? err.message : String(err) });
    } finally {
      setBuilding(false);
      setBuildStage(null);
      setBuildDetail('');
    }
  };

  const handleRunTestLine = async () => {
    if (!holoviewConfig) return;
    setTestRunning(true);
    try {
      const ans = await holoviewConfig.onInvoke(testRequestLine);
      setTestAnswer(ans);
    } catch (err) {
      setTestAnswer(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setTestRunning(false);
    }
  };

  const handleProposeOpsFromPrompt = async (promptText: string) => {
    if (!declaration) return;
    const hasRealModel =
      engineState.model !== null && !isInternalBootstrapModel(engineState.model.id);
    if (!hasRealModel) {
      setApiError({
        code: 'model.not-ready',
        message: 'Local model is not loaded yet.',
        remediation: 'Enable Model Stream in the top bar and wait for it to finish loading.',
      });
      return;
    }

    setProposingOps(true);
    setProposedOps(null);
    setApiError(null);
    try {
      const syntheticUser: ThreadMessage = {
        id: `propose_${Date.now()}`,
        role: 'user',
        createdAt: new Date(),
        content: [{ type: 'text', text: promptText }],
        attachments: [],
        metadata: { custom: {} },
      } as ThreadMessage;
      const prompt = formatMessagesForLlama(
        [syntheticUser],
        engineState.model!,
        engineState.promptMode,
        buildStudioApiProposerPrompt(declaration),
        { thinkingMode: 'off' },
      );
      let raw = '';
      const ac = new AbortController();
      for await (const ev of workerClient.streamGenerate(
        prompt,
        ac.signal,
        { ...engineState.sampling, temperature: 0.2, maxTokens: 768 },
        `studio-api:${project.id}`,
      )) {
        raw = stripReasoningTags(ev.accumulatedText);
      }
      const ops = extractStudioApiOpsFromModelOutput(raw);
      if (!ops) {
        setApiError({
          code: 'proposer.no-ops',
          message: 'Could not extract structured Studio API operations from model response.',
          remediation: 'Use the visual Features / Data / Design controls directly, or try a more specific rule description.',
        });
      } else {
        setProposedOps(ops);
      }
    } catch (err) {
      setApiError({
        code: 'proposer.failed',
        message: err instanceof Error ? err.message : String(err),
        remediation: 'Retry or use the declarative controls directly.',
      });
    } finally {
      setProposingOps(false);
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

  const handleSelectBlueprintStage = useCallback(
    (stage: StudioWizardStage) => {
      setWizardStage(stage);
      const mapToGuided: Record<StudioWizardStage, GuidedStepId> = {
        concept: routing?.isUndefined ? 'anchor-idea' : 'confirm-context',
        architecture: 'refine-rules',
        security: 'safety-check',
        design: 'design-feel',
        standards: 'routed-standards',
        seal: 'ready-to-seal',
      };
      setGuidedStep(mapToGuided[stage]);
    },
    [routing?.isUndefined],
  );

  // Dark-only presets (`noir`, `midnight`) must always render in dark mode regardless of host UI mode
  const effectivePreviewDark =
    compiled && !compiled.themeModes.includes('light') ? true : darkMode;

  const displayAddress = activeHolo?.address ?? activeSnapshot.digest;
  const isPublished = project.published !== null;
  const isModified =
    isPublished &&
    (project.published!.digest !== project.draft.digest ||
      (project.publishedHolo?.address ?? null) !== (project.sealedDraft?.address ?? null));
  const canUndo = isDeclarative
    ? (project.declarationRevisions?.length ?? 0) > 0
    : project.revisions.length > 0;

  const renderPrismInspector = (onClose?: () => void, closeLabel = 'Back to Preview') => (
    <div className="flex h-full min-h-0 min-w-0 flex-col overflow-hidden bg-white dark:bg-[#0B0F14]">
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-zinc-200 px-4 py-2 dark:border-zinc-800">
        <div className="flex flex-wrap items-center gap-1">
          {(
            [
              ['overview', 'Overview', ShieldCheck],
              ['package', 'Package (.holo)', FileCheck2],
              ...(compiled ? [['compiled', 'Compiled Files', Code2] as const] : []),
            ] as readonly [PrismPanelTab, string, typeof ShieldCheck][]
          ).map(([id, label, Icon]) => (
            <button
              key={id}
              type="button"
              onClick={() => setPanelTab(id)}
              className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                panelTab === id
                  ? 'bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900'
                  : 'text-zinc-600 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-800'
              }`}
            >
              <Icon className="h-3.5 w-3.5" />
              {label}
            </button>
          ))}
        </div>
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="inline-flex shrink-0 items-center gap-1 rounded-md border border-zinc-200 px-2.5 py-1 text-xs font-medium text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-800"
          >
            <Eye className="h-3.5 w-3.5" />
            <span>{closeLabel}</span>
          </button>
        )}
      </div>

      <div className="min-h-0 min-w-0 flex-1 overflow-y-auto p-4 text-xs">
        {panelTab === 'overview' && (
          <div className="space-y-4">
            {/* Honest Claim Verification Cards */}
            <div className="grid gap-2 sm:grid-cols-3">
              <div className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
                  Model Rules
                </p>
                <p className="mt-1 flex items-center gap-1.5 text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                  {activeHolo?.proved ? (
                    <>
                      <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                      <span>Proved</span>
                    </>
                  ) : (
                    <>
                      <AlertTriangle className="h-4 w-4 text-amber-500" />
                      <span>Unsealed Draft</span>
                    </>
                  )}
                </p>
                <p className="mt-0.5 text-[11px] text-zinc-500">
                  {activeHolo?.proved
                    ? `${activeHolo.theorems.length || 'All'} theorem(s) accepted by WASM kernel`
                    : 'Click Check & Build to prove and seal'}
                </p>
              </div>
              <div className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
                  Acceptance Examples
                </p>
                <p className="mt-1 text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                  {activeHolo
                    ? `${activeHolo.vectorsPassed} / ${activeHolo.vectorsTotal} Replayed`
                    : checkReport
                      ? `${checkReport.vectorsPassed} / ${checkReport.vectorsTotal} Passing`
                      : 'N/A'}
                </p>
                <p className="mt-0.5 text-[11px] text-zinc-500">
                  {activeHolo ? 'Tested on portable WASM guest' : 'Evaluated on draft rules'}
                </p>
              </div>
              <div className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
                  Interface
                </p>
                <p className="mt-1 text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                  {declaration ? `prism.design/1 (${declaration.theme.preset})` : 'HOLOVIEW'}
                </p>
                <p className="mt-0.5 text-[11px] text-zinc-500">
                  Authored view · zero rules in page JS
                </p>
              </div>
            </div>

            {/* Content Address */}
            <div className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-semibold text-zinc-500">
                  {activeHolo ? 'UOR Content Address (BLAKE3)' : 'Draft SHA-256 Digest'}
                </span>
                <button
                  type="button"
                  onClick={() => {
                    void navigator.clipboard?.writeText(displayAddress);
                    setDigestCopied(true);
                    setTimeout(() => setDigestCopied(false), 1500);
                  }}
                  className="inline-flex items-center gap-1 rounded px-2 py-0.5 text-[11px] text-zinc-600 hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-800"
                >
                  {digestCopied ? <Check className="h-3 w-3 text-emerald-600" /> : <Copy className="h-3 w-3" />}
                  {digestCopied ? 'Copied' : 'Copy'}
                </button>
              </div>
              <code className="mt-1 block break-all font-mono text-[11px] text-zinc-800 dark:text-zinc-200">
                {displayAddress}
              </code>
            </div>

            {/* Core Spine & Routed Standards Summary */}
            {routing && (
              <div className="space-y-1.5 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-semibold text-zinc-600 dark:text-zinc-300">
                    Verdant Prism Core Models &amp; Routed Standards
                  </span>
                  <span className="font-mono text-[10px] text-zinc-500">
                    Core {routing.summary.coreComplete}/4 · Layered {routing.summary.activeLayeredCount} · Excluded {routing.summary.excludedCount}
                  </span>
                </div>
                <div className="flex flex-wrap gap-1 pt-0.5">
                  {routing.coreSpine.map((c) => (
                    <span
                      key={c.authority.id}
                      className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                        c.settledCount >= c.totalCount
                          ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300'
                          : 'bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300'
                      }`}
                    >
                      {c.authority.name} ({c.settledCount}/{c.totalCount})
                    </span>
                  ))}
                  {routing.activeLayeredStandards.map((s) => (
                    <span
                      key={s.authority.id}
                      className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                        s.settledCount >= s.totalCount
                          ? 'bg-sky-100 text-sky-800 dark:bg-sky-950/60 dark:text-sky-300'
                          : 'bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300'
                      }`}
                    >
                      {s.authority.name} ({s.settledCount}/{s.totalCount})
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Traceability Matrix (REQ-D9) */}
            {checkReport && (
              <div className="space-y-1.5">
                <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                  Traceability Matrix (Requirement → Rule → View)
                </h4>
                <div className="overflow-hidden rounded-lg border border-zinc-200 dark:border-zinc-800">
                  <table className="w-full text-left text-[11px]">
                    <thead className="bg-zinc-50 text-zinc-500 dark:bg-zinc-900/70">
                      <tr>
                        <th className="px-2.5 py-1.5 font-medium">Requirement</th>
                        <th className="px-2.5 py-1.5 font-medium">Rules / Refusals</th>
                        <th className="px-2.5 py-1.5 font-medium">View Elements</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-zinc-200 dark:divide-zinc-800">
                      {checkReport.traceability.map((row) => (
                        <tr key={row.requirementId}>
                          <td className="px-2.5 py-1.5 font-mono font-semibold">{row.requirementId}</td>
                          <td className="px-2.5 py-1.5 font-mono text-zinc-600 dark:text-zinc-300">
                            {[...row.featureIds, ...row.refusalIds].join(', ') || '—'}
                          </td>
                          <td className="px-2.5 py-1.5 text-zinc-600 dark:text-zinc-300">
                            {row.viewElementIds.length} bound
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Live Published Capabilities (App Mode quick controls) */}
            {mode === 'app' && isPublished && (
              <div className="space-y-1.5 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-semibold text-zinc-600 dark:text-zinc-300">
                    Runtime Capabilities (Deny by Default)
                  </span>
                  <button
                    type="button"
                    onClick={async () => {
                      bridgeRuntime.evictPrism(project.id);
                      await store.unpublish(project.id);
                      setAppPrismModalOpen(false);
                      onChangeMode('studio');
                    }}
                    className="rounded px-2 py-0.5 text-[11px] text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-950/40"
                  >
                    Unpublish
                  </button>
                </div>
                <div className="grid grid-cols-3 gap-1.5 pt-1">
                  {(
                    [
                      ['storage', 'Local Storage'],
                      ['browser', 'Browser Utilities'],
                      ['network', 'Internet Access'],
                    ] as const
                  ).map(([key, label]) => (
                    <label
                      key={key}
                      className="flex cursor-pointer items-center justify-between rounded border border-zinc-200 px-2 py-1.5 text-[11px] dark:border-zinc-800"
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

            {/* Quick Live Model Invocation Test */}
            {holoviewConfig && (
              <div className="space-y-1.5 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
                <span className="text-[11px] font-semibold text-zinc-600 dark:text-zinc-300">
                  Direct Protocol Test ({activeHolo ? 'Portable WASM Guest' : 'Instant Draft Evaluator'})
                </span>
                <div className="flex gap-1.5">
                  <input
                    type="text"
                    value={testRequestLine}
                    onChange={(e) => setTestRequestLine(e.target.value)}
                    placeholder={
                      declaration?.vectors[0]?.request
                        ? `e.g. ${declaration.vectors[0].request}`
                        : 'Type a request line…'
                    }
                    className="flex-1 rounded border border-zinc-300 bg-transparent px-2 py-1 font-mono text-xs dark:border-zinc-700"
                  />
                  <button
                    type="button"
                    disabled={testRunning}
                    onClick={() => void handleRunTestLine()}
                    className="inline-flex items-center gap-1 rounded bg-emerald-700 px-2.5 py-1 text-xs font-medium text-white hover:bg-emerald-800"
                  >
                    <Play className="h-3 w-3" />
                    Run
                  </button>
                </div>
                {testAnswer !== null && (
                  <p className="rounded bg-zinc-100 px-2 py-1.5 font-mono text-xs text-zinc-900 dark:bg-zinc-900 dark:text-zinc-100">
                    {testAnswer}
                  </p>
                )}
              </div>
            )}
          </div>
        )}

        {panelTab === 'package' && (
          <div className="space-y-4">
            <div className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <p className="font-semibold text-zinc-900 dark:text-zinc-100">
                    {activeHolo ? `${activeHolo.schema} Binary Archive` : 'Unsealed Declarative Draft'}
                  </p>
                  <p className="mt-0.5 text-[11px] text-zinc-500">
                    {activeHolo
                      ? `${(activeHolo.bytes.length / 1024).toFixed(1)} KB · Snapshot: ${activeHolo.carries.snapshot ? 'Yes' : 'No'} · Sources: ${activeHolo.carries.sources ? 'Yes' : 'No'} · Proof: ${activeHolo.carries.proof ? 'Yes' : 'No'}`
                      : 'Click Check & Build to compile and seal a portable binary .holo archive.'}
                  </p>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {activeHolo && (
                    <button
                      type="button"
                      disabled={building}
                      onClick={() => void handleReverifyOffline()}
                      className="inline-flex items-center gap-1 rounded-md border border-zinc-200 px-2.5 py-1 text-xs font-medium hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
                    >
                      <ShieldCheck className="h-3.5 w-3.5 text-emerald-600" />
                      Verify &amp; Re-Prove
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={handleExportHolo}
                    className="inline-flex items-center gap-1 rounded-md bg-zinc-900 px-2.5 py-1 text-xs font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900"
                  >
                    <Download className="h-3.5 w-3.5" />
                    Export .holo
                  </button>
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="inline-flex items-center gap-1 rounded-md border border-zinc-200 px-2.5 py-1 text-xs font-medium hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
                  >
                    <Upload className="h-3.5 w-3.5" />
                    Import .holo
                  </button>
                </div>
              </div>
            </div>

            {activeHolo && activeHolo.theorems.length > 0 && (
              <div className="space-y-1.5">
                <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                  Kernel-Verified Theorems ({activeHolo.theorems.length})
                </h4>
                <div className="space-y-1">
                  {activeHolo.theorems.map((t, i) => (
                    <div
                      key={i}
                      className="flex items-center justify-between rounded border border-zinc-200 px-2.5 py-1.5 dark:border-zinc-800"
                    >
                      <code className="font-mono text-[11px]">{t.theorem}</code>
                      <span className="rounded bg-emerald-100 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300">
                        {t.status}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {panelTab === 'compiled' && compiled && (
          <div className="flex h-full min-h-0 min-w-0 flex-col gap-2">
            <div className="flex flex-wrap items-center justify-between gap-1">
              <div className="flex flex-wrap gap-1">
                {Object.keys(compiled.files).map((filePath) => (
                  <button
                    key={filePath}
                    type="button"
                    onClick={() => setSelectedCompiledFile(filePath)}
                    className={`rounded px-2 py-1 font-mono text-[11px] ${
                      selectedCompiledFile === filePath
                        ? 'bg-emerald-700 text-white'
                        : 'bg-zinc-100 text-zinc-600 hover:bg-zinc-200 dark:bg-zinc-900 dark:text-zinc-300'
                    }`}
                  >
                    {filePath}
                  </button>
                ))}
              </div>
              <button
                type="button"
                onClick={() => {
                  void navigator.clipboard?.writeText(compiled.files[selectedCompiledFile] ?? '');
                  setCompiledCopied(true);
                  setTimeout(() => setCompiledCopied(false), 1400);
                }}
                className="inline-flex shrink-0 items-center gap-1 rounded border border-zinc-200 px-2 py-1 text-[11px] text-zinc-600 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
              >
                {compiledCopied ? <Check className="h-3 w-3 text-emerald-600" /> : <Copy className="h-3 w-3" />}
                {compiledCopied ? 'Copied' : 'Copy'}
              </button>
            </div>
            <pre className="min-h-[260px] w-full max-w-full flex-1 overflow-auto rounded-lg border border-zinc-200 bg-zinc-50 p-3 font-mono text-[11px] leading-relaxed text-zinc-800 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-200">
              {compiled.files[selectedCompiledFile] ?? ''}
            </pre>
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
          if (!draggingHolo) setDraggingHolo(true);
        }
      }}
      onDragLeave={(e) => {
        if (e.currentTarget.contains(e.relatedTarget as Node | null)) return;
        setDraggingHolo(false);
      }}
      onDrop={(e) => void handleDropHolo(e)}
    >
      {draggingHolo && (
        <div className="pointer-events-none absolute inset-0 z-40 flex items-center justify-center bg-emerald-950/50 backdrop-blur-xs">
          <div className="rounded-xl border-2 border-dashed border-emerald-400 bg-white/95 px-6 py-4 text-center shadow-xl dark:bg-zinc-900/95">
            <Upload className="mx-auto h-6 w-6 text-emerald-600" />
            <p className="mt-2 text-sm font-semibold text-zinc-900 dark:text-zinc-100">
              Drop .holo archive to verify &amp; run offline
            </p>
          </div>
        </div>
      )}
      <input
        ref={fileInputRef}
        type="file"
        accept=".holo,.json,application/octet-stream,application/json"
        className="hidden"
        onChange={handleImportFileChange}
      />

      {/* Compact ~40px Prism Focus Bar */}
      <header className="flex h-11 shrink-0 items-center justify-between gap-2 border-b border-zinc-200 bg-white px-3 dark:border-zinc-800 dark:bg-[#0E131B]">
        <div className="flex min-w-0 items-center gap-2">
          <span
            className={`h-2 w-2 shrink-0 rounded-full ${
              activeHolo?.proved && isPublished && !isModified
                ? 'bg-emerald-500'
                : isPublished
                  ? 'bg-amber-500'
                  : 'bg-zinc-400'
            }`}
            title={isPublished ? (isModified ? 'Modified since last publish' : 'Published') : 'Draft'}
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
              title="Click to rename Prism"
            >
              {project.draft.name}
            </button>
          )}
          <span className="hidden rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-[10px] text-zinc-500 sm:inline dark:bg-zinc-800 dark:text-zinc-400">
            {activeHolo ? `${activeHolo.address.slice(0, 15)}…` : shortPrismDigest(activeSnapshot.digest)}
          </span>
          {activeHolo?.proved && (
            <span className="hidden items-center gap-1 rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-semibold text-emerald-800 md:inline-flex dark:bg-emerald-950/70 dark:text-emerald-300">
              <ShieldCheck className="h-3 w-3" />
              Proved .holo
            </span>
          )}
          {project.kind === 'legacy-v1' && (
            <span className="rounded-full bg-zinc-100 px-2 py-0.5 text-[10px] font-medium text-zinc-500 dark:bg-zinc-800">
              Legacy v1
            </span>
          )}
        </div>

        <div className="flex shrink-0 items-center gap-1.5">
          {mode === 'studio' && (
            <>
              <button
                type="button"
                disabled={!canUndo}
                onClick={() => void handleUndo()}
                title="Undo last declaration change"
                className="inline-flex h-7 items-center gap-1 rounded-md border border-zinc-200 px-2 text-xs text-zinc-600 hover:bg-zinc-100 disabled:opacity-40 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
              >
                <Undo2 className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Undo</span>
              </button>
              {isDeclarative && (
                <button
                  type="button"
                  disabled={building}
                  onClick={() => void handleCheckAndBuild()}
                  className="inline-flex h-7 items-center gap-1.5 rounded-md border border-emerald-600/40 bg-emerald-50 px-2.5 text-xs font-medium text-emerald-800 hover:bg-emerald-100 disabled:opacity-60 dark:border-emerald-700/60 dark:bg-emerald-950/50 dark:text-emerald-200"
                >
                  {building ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <ShieldCheck className="h-3.5 w-3.5" />
                  )}
                  <span>{building ? (buildStage ?? 'Building') : 'Check & Build'}</span>
                </button>
              )}
              <button
                type="button"
                disabled={building}
                onClick={() => void handlePublish()}
                className="inline-flex h-7 items-center gap-1 rounded-md bg-emerald-700 px-2.5 text-xs font-medium text-white hover:bg-emerald-800 disabled:opacity-50"
              >
                {justPublished ? <Check className="h-3.5 w-3.5" /> : <Globe className="h-3.5 w-3.5" />}
                <span>{justPublished ? 'Published' : isPublished ? (isModified ? 'Update' : 'Published') : 'Publish'}</span>
              </button>
            </>
          )}

          {mode === 'app' && (
            <button
              type="button"
              aria-pressed={appPrismModalOpen}
              onClick={() => setAppPrismModalOpen((v) => !v)}
              className={`inline-flex h-7 items-center gap-1 rounded-md border px-2.5 text-xs font-medium transition-colors ${
                appPrismModalOpen
                  ? 'border-emerald-600 bg-emerald-50 text-emerald-900 dark:border-emerald-500 dark:bg-emerald-950/60 dark:text-emerald-200'
                  : 'border-zinc-200 text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-800'
              }`}
            >
              <Layers className="h-3.5 w-3.5" />
              {appPrismModalOpen ? 'Back to App' : 'Prism'}
            </button>
          )}

          {!isExternalHolo && (
            <div className="inline-flex rounded-lg border border-zinc-200 bg-zinc-100 p-0.5 text-[11px] dark:border-zinc-700 dark:bg-zinc-900">
              <button
                type="button"
                onClick={() => {
                  setAppPrismModalOpen(false);
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
                  setAppPrismModalOpen(false);
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

      {/* Live Build Progress / Result Banner */}
      {(building || buildBanner) && (
        <div
          className={`flex shrink-0 items-center justify-between gap-2 border-b px-3 py-1.5 text-xs ${
            building
              ? 'border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-900/50 dark:bg-emerald-950/40 dark:text-emerald-200'
              : buildBanner?.ok
                ? 'border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-900/50 dark:bg-emerald-950/40 dark:text-emerald-200'
                : 'border-red-200 bg-red-50 text-red-900 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-200'
          }`}
        >
          <div className="flex min-w-0 items-center gap-2">
            {building ? (
              <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin" />
            ) : buildBanner?.ok ? (
              <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
            ) : (
              <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
            )}
            <span className="truncate">{building ? buildDetail : buildBanner?.text}</span>
          </div>
          {!building && buildBanner && (
            <button
              type="button"
              onClick={() => setBuildBanner(null)}
              className="shrink-0 rounded p-0.5 hover:bg-black/5"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      )}

      {/* Main Content Stage */}
      {mode === 'app' ? (
        <div className="relative min-h-0 min-w-0 flex-1 overflow-hidden">
          <div className={`h-full w-full ${appPrismModalOpen ? 'invisible pointer-events-none' : ''}`}>
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
          {appPrismModalOpen && (
            <div className="absolute inset-0 z-20 flex flex-col overflow-hidden bg-white dark:bg-[#0B0F14]">
              <div className="flex shrink-0 items-center justify-between border-b border-zinc-200 px-4 py-2 dark:border-zinc-800">
                <div className="flex items-center gap-2">
                  <Layers className="h-4 w-4 text-emerald-600" />
                  <span className="text-sm font-semibold">{project.draft.name} · Prism Package</span>
                </div>
                <div className="flex items-center gap-2">
                  {!isExternalHolo && (
                    <button
                      type="button"
                      onClick={() => {
                        setAppPrismModalOpen(false);
                        onChangeMode('studio');
                      }}
                      className="inline-flex items-center gap-1 rounded-md border border-zinc-200 px-2.5 py-1 text-xs font-medium hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
                    >
                      <ExternalLink className="h-3 w-3" />
                      Open in Studio
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => setAppPrismModalOpen(false)}
                    className="inline-flex items-center gap-1 rounded-md bg-zinc-900 px-2.5 py-1 text-xs font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900"
                  >
                    <X className="h-3.5 w-3.5" />
                    Close
                  </button>
                </div>
              </div>
              <div className="min-h-0 min-w-0 flex-1 overflow-hidden">
                {renderPrismInspector(() => setAppPrismModalOpen(false), 'Back to App')}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden lg:flex-row">
          {/* Left Pane: Guided Conversational Creator | Advanced Model Editor */}
          <div className="flex h-1/2 min-h-0 min-w-0 w-full flex-col overflow-hidden border-b border-zinc-200 lg:h-full lg:w-[460px] lg:shrink-0 lg:border-r lg:border-b-0 dark:border-zinc-800">
            {isDeclarative && declaration && checkReport && routing ? (
              <div className="flex h-full min-h-0 min-w-0 flex-col">
                {/* Compact Guided | Advanced Switcher */}
                <div className="flex shrink-0 items-center justify-between gap-2 border-b border-zinc-200 bg-zinc-50/70 px-3 py-1.5 dark:border-zinc-800 dark:bg-[#0B0F14]">
                  <div className="inline-flex rounded-lg border border-zinc-200 bg-white p-0.5 text-[11px] dark:border-zinc-700 dark:bg-zinc-900">
                    <button
                      type="button"
                      onClick={() => setLeftPaneMode('guided')}
                      className={`inline-flex items-center gap-1 rounded-md px-2.5 py-0.5 font-medium transition-colors ${
                        leftPaneMode === 'guided'
                          ? 'bg-emerald-700 text-white shadow-xs'
                          : 'text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100'
                      }`}
                    >
                      <Sparkles className="h-3 w-3" />
                      Guided
                    </button>
                    <button
                      type="button"
                      onClick={() => setLeftPaneMode('advanced')}
                      className={`inline-flex items-center gap-1 rounded-md px-2.5 py-0.5 font-medium transition-colors ${
                        leftPaneMode === 'advanced'
                          ? 'bg-zinc-900 text-white shadow-xs dark:bg-zinc-100 dark:text-zinc-900'
                          : 'text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100'
                      }`}
                    >
                      <SlidersHorizontal className="h-3 w-3" />
                      Advanced
                    </button>
                  </div>
                  <span className="text-[10px] font-medium text-zinc-500">
                    {routing.readyToSeal
                      ? 'Ready to prove'
                      : `${routing.summary.coreComplete}/4 core models`}
                  </span>
                </div>

                <div className="min-h-0 flex-1 overflow-hidden">
                  {leftPaneMode === 'guided' ? (
                    <StudioGuidedWizard
                      key={`guided-${project.id}`}
                      declaration={declaration}
                      routing={routing}
                      checkReport={checkReport}
                      activeStep={guidedStep}
                      onSelectStep={setGuidedStep}
                      onDispatchOp={handleDispatchOp}
                      onSynthesizeWithLocalModel={handleSynthesizeIdeaWithLocalModel}
                      hasLocalModelReady={hasLocalModelReady}
                      onOpenPreview={() => setPreviewPaneMode('preview')}
                      onCheckAndBuild={() => void handleCheckAndBuild()}
                      onPublish={() => void handlePublish()}
                      building={building}
                      onSwitchToAdvanced={() => setLeftPaneMode('advanced')}
                    />
                  ) : (
                    <StudioDeclarativeBuilder
                      key={project.id}
                      declaration={declaration}
                      checkReport={checkReport}
                      routing={routing}
                      agenda={agenda}
                      activeStage={wizardStage}
                      onSelectStage={setWizardStage}
                      onDispatchOp={handleDispatchOp}
                      onLoadStarter={handleLoadStarter}
                      onImportHoloFile={() => fileInputRef.current?.click()}
                      onCheckAndBuild={() => void handleCheckAndBuild()}
                      onPublish={() => void handlePublish()}
                      building={building}
                      apiError={apiError}
                      onClearApiError={() => setApiError(null)}
                      modelStreamEnabled={engineState.modelStreamEnabled}
                      onEnableModelStream={onEnableModelStream}
                      proposingOps={proposingOps}
                      proposedOps={proposedOps}
                      onProposeOpsFromPrompt={(p) => void handleProposeOpsFromPrompt(p)}
                      onApplyProposedOps={async () => {
                        if (!proposedOps) return;
                        const ok = await handleDispatchOp({ op: 'batch', ops: proposedOps });
                        if (ok) setProposedOps(null);
                      }}
                      onDismissProposedOps={() => setProposedOps(null)}
                    />
                  )}
                </div>
              </div>
            ) : (
              <div className="flex h-full flex-col items-center justify-center gap-3 p-6 text-center text-xs text-zinc-500">
                <Layers className="h-8 w-8 text-emerald-600/70" />
                <p className="max-w-xs text-sm font-medium text-zinc-800 dark:text-zinc-200">
                  {isExternalHolo
                    ? 'Imported Verified .holo Archive'
                    : 'Legacy v1 HTML/CSS/JS Prism'}
                </p>
                <p className="max-w-xs text-xs text-zinc-500">
                  {isExternalHolo
                    ? 'This binary archive runs directly on its verified portable WASM guest. Inspect its BLAKE3 address and vectors in the Prism tab.'
                    : 'This project was saved in the earlier v1 format. Start a new Declarative Prism to compile and prove real binary .holo v6 archives.'}
                </p>
                <button
                  type="button"
                  onClick={() => handleLoadStarter('undefined')}
                  className="rounded-lg bg-emerald-700 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-800"
                >
                  New Verdant Prism
                </button>
              </div>
            )}
          </div>

          {/* Right Pane: Model Blueprint / Live Compiled Preview / Prism Toggle */}
          <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
            <div className="relative z-20 flex shrink-0 items-center justify-between gap-2 border-b border-zinc-200 bg-white px-3 py-1.5 dark:border-zinc-800 dark:bg-[#0E131B]">
              <div className="inline-flex rounded-lg border border-zinc-200 bg-zinc-100 p-0.5 text-xs dark:border-zinc-700 dark:bg-zinc-900">
                {isDeclarative && routing && (
                  <button
                    type="button"
                    aria-pressed={previewPaneMode === 'blueprint'}
                    onClick={() => setPreviewPaneMode('blueprint')}
                    className={`inline-flex items-center gap-1 rounded-md px-2.5 py-1 font-medium transition-colors ${
                      previewPaneMode === 'blueprint'
                        ? 'bg-white text-zinc-900 shadow-xs dark:bg-zinc-800 dark:text-zinc-100'
                        : 'text-zinc-500 hover:text-zinc-800 dark:text-zinc-400'
                    }`}
                  >
                    <Compass className="h-3.5 w-3.5" />
                    Blueprint
                  </button>
                )}
                <button
                  type="button"
                  disabled={Boolean(isDeclarative && routing && !routing.readyToPreview)}
                  aria-pressed={previewPaneMode === 'preview'}
                  onClick={() => setPreviewPaneMode('preview')}
                  title={
                    isDeclarative && routing && !routing.readyToPreview
                      ? 'Complete Concept, System, and Design to synthesize live Preview'
                      : 'Live compiled Prism preview'
                  }
                  className={`inline-flex items-center gap-1 rounded-md px-2.5 py-1 font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
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
                  aria-pressed={previewPaneMode === 'prism'}
                  onClick={() =>
                    setPreviewPaneMode((m) =>
                      m === 'prism' ? (routing?.readyToPreview ? 'preview' : 'blueprint') : 'prism',
                    )
                  }
                  className={`inline-flex items-center gap-1 rounded-md px-2.5 py-1 font-medium transition-colors ${
                    previewPaneMode === 'prism'
                      ? 'bg-white text-zinc-900 shadow-xs dark:bg-zinc-800 dark:text-zinc-100'
                      : 'text-zinc-500 hover:text-zinc-800 dark:text-zinc-400'
                  }`}
                >
                  <Layers className="h-3.5 w-3.5" />
                  Prism
                </button>
              </div>

              <div className="flex items-center gap-1.5">
                {previewPaneMode === 'preview' && (
                  <>
                    <button
                      type="button"
                      onClick={() => setViewportMode('desktop')}
                      className={`rounded p-1 ${viewportMode === 'desktop' ? 'bg-zinc-200/80 text-zinc-900 dark:bg-zinc-800 dark:text-zinc-100' : 'text-zinc-400'}`}
                      title="Desktop viewport"
                    >
                      <Monitor className="h-3.5 w-3.5" />
                    </button>
                    <button
                      type="button"
                      onClick={() => setViewportMode('mobile')}
                      className={`rounded p-1 ${viewportMode === 'mobile' ? 'bg-zinc-200/80 text-zinc-900 dark:bg-zinc-800 dark:text-zinc-100' : 'text-zinc-400'}`}
                      title="Mobile viewport"
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
                <span className="truncate">Runtime error: {runtimeError.message}</span>
                <button
                  type="button"
                  onClick={() => setRuntimeError(null)}
                  className="rounded px-1.5 py-0.5 text-[11px] hover:bg-red-100 dark:hover:bg-red-900/50"
                >
                  Dismiss
                </button>
              </div>
            )}

            {/* Blueprint, Preview & Prism Stack */}
            <div className="relative min-h-0 flex-1 overflow-hidden bg-zinc-100/80 dark:bg-[#090D12]">
              {(!isDeclarative || !routing || routing.readyToPreview) && (
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
              )}

              {(previewPaneMode === 'blueprint' || (previewPaneMode === 'preview' && !routing?.readyToPreview)) && declaration && routing && (
                <div className="absolute inset-0 z-10 overflow-hidden">
                  <ModelBlueprintCanvas
                    declaration={declaration}
                    routing={routing}
                    activeStage={wizardStage}
                    onSelectStage={handleSelectBlueprintStage}
                    onOpenPreview={() => setPreviewPaneMode('preview')}
                  />
                </div>
              )}

              {previewPaneMode === 'prism' && (
                <div className="absolute inset-0 z-10 overflow-hidden">
                  {renderPrismInspector(
                    () => setPreviewPaneMode(routing?.readyToPreview ? 'preview' : 'blueprint'),
                    routing?.readyToPreview ? 'Back to Preview' : 'Back to Blueprint',
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
