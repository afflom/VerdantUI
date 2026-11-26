/**
 * Continuous Conversational Verdant Studio Agent Wizard (`src/components/StudioGuidedWizard.tsx`)
 *
 * Pairs the Creator with the 100% client-side Verdant Studio Agent over Verdant Studio's
 * formal Core Models & Standards:
 * - Every stage is populated, reconciled, and refined through plain-English conversation —
 *   zero manual forms or hardcoded app templates.
 * - For open-ended interactive apps & games (e.g. "This Prism will be the video game 'Minesweeper'"):
 *   asks 1–2 tailored clarifying questions, reconciles all Core Models & applicable standards,
 *   and synthesizes the real, playable interactive app (`admit-authored-view`) packed into
 *   a verifiable `prism/holo/6` archive.
 * - When the local on-device model is off and an open-ended app needs agent synthesis,
 *   offers a 1-click "Turn On Local Studio Agent & Continue" action inline and auto-resumes
 *   the moment the model finishes loading.
 * - Zero exposure of internal Verdant constructs (no Observer, OTel, vTensor, LexLean).
 */

import { useCallback, useEffect, useMemo, useRef, useState, type FC, type FormEvent } from 'react';
import {
  Check,
  Cpu,
  Globe,
  Loader2,
  Send,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Wrench,
} from 'lucide-react';
import {
  dispatchStudioOp,
  type DeclarationCheckReport,
  type PrismDeclaration,
  type StudioApiOp,
} from '../engine/verdantStudioApi';
import type { VerdantRoutingReport } from '../engine/verdantStandardsRouter';
import {
  buildStudioPromptForStep,
  computeGuidedWizardState,
  parseIdeaSentence,
  reconcileWizardTurn,
  type ExtractedPrismIntent,
  type GuidedStepId,
} from '../engine/prismInterviewEngine';
import {
  assembleCompleteAuthoredPrismOps,
  buildAgentCoreModelOps,
  buildDiscoveryAgentPrompt,
  buildInteractiveAppPrompt,
  buildRefineInteractiveAppPrompt,
  classifyIdeaRealizationMode,
  extractAuthoredAppFromModelOutput,
  extractCleanIdeaTitle,
  parseDiscoveryAgentResponse,
  type AgentChatTurn,
} from '../engine/prismStudioAgent';
import type { EngineStatus } from '../engine/workerClient';
import type { PrismRuntimeError } from './PrismSandboxFrame';

interface StudioGuidedWizardProps {
  declaration: PrismDeclaration;
  routing: VerdantRoutingReport;
  checkReport: DeclarationCheckReport;
  activeStep: GuidedStepId;
  onSelectStep: (step: GuidedStepId) => void;
  onDispatchOp: (op: StudioApiOp, opts?: { readonly pushRevision?: boolean }) => Promise<boolean>;
  onSynthesizeWithLocalModel?: (sentence: string, fallback: ExtractedPrismIntent) => Promise<ExtractedPrismIntent>;
  onRunStudioAgentPrompt?: (
    systemPrompt: string,
    userPrompt: string,
    maxTokens: number,
    onProgress?: (charsGenerated: number) => void,
  ) => Promise<string>;
  hasLocalModelReady: boolean;
  engineStatus: EngineStatus;
  engineProgressPercent: number;
  engineStatusDetail: string;
  loadedModelName: string | null;
  onEnableModelStream: () => void;
  runtimeError: PrismRuntimeError | null;
  onClearRuntimeError: () => void;
  onOpenPreview: () => void;
  onCheckAndBuild: () => void;
  onPublish: () => void;
  building: boolean;
  onSwitchToAdvanced: () => void;
}

interface WizardChatEntry {
  readonly id: string;
  readonly role: 'studio' | 'user';
  readonly text: string;
  readonly badges?: readonly string[];
  readonly needsLocalModel?: boolean;
}

const STARTER_SUGGESTIONS: readonly string[] = [
  'This Prism will be the video game "Minesweeper"',
  'Build a Pomodoro focus timer with customizable work/break sessions',
  'Check if a new expense fits my monthly budget, warn when over, and refuse zero amounts.',
];

export const StudioGuidedWizard: FC<StudioGuidedWizardProps> = ({
  declaration,
  routing,
  checkReport,
  activeStep,
  onSelectStep,
  onDispatchOp,
  onSynthesizeWithLocalModel,
  onRunStudioAgentPrompt,
  hasLocalModelReady,
  engineStatus,
  engineProgressPercent,
  engineStatusDetail,
  loadedModelName,
  onEnableModelStream,
  runtimeError,
  onClearRuntimeError,
  onOpenPreview,
  onCheckAndBuild,
  onPublish,
  building,
  onSwitchToAdvanced,
}) => {
  const [inputText, setInputText] = useState('');
  const [thinking, setThinking] = useState(false);
  const [agentPhaseLabel, setAgentPhaseLabel] = useState('Reconciling your Prism model…');
  const [transcript, setTranscript] = useState<readonly WizardChatEntry[]>([]);
  const [agentQuickReplies, setAgentQuickReplies] = useState<readonly string[] | null>(null);
  const [interactiveSessionActive, setInteractiveSessionActive] = useState(
    Boolean(declaration.view?.authoredView),
  );
  const [pendingAgentMessage, setPendingAgentMessage] = useState<string | null>(null);
  const scrollEndRef = useRef<HTMLDivElement | null>(null);
  const transcriptRef = useRef(transcript);
  transcriptRef.current = transcript;

  const guided = useMemo(
    () => computeGuidedWizardState(declaration, routing, checkReport),
    [declaration, routing, checkReport],
  );

  const currentPrompt = useMemo(
    () => buildStudioPromptForStep(declaration, routing, checkReport, activeStep),
    [declaration, routing, checkReport, activeStep],
  );

  useEffect(() => {
    scrollEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [transcript.length, activeStep, thinking, agentPhaseLabel, pendingAgentMessage]);

  const isModelLoading = engineStatus === 'downloading' || engineStatus === 'loading';

  const runInteractiveBuildPass = useCallback(
    async (
      history: readonly AgentChatTurn[],
      workingDecl: PrismDeclaration,
      spec: ReturnType<typeof parseDiscoveryAgentResponse>['spec'] | null,
      userMessage: string,
    ): Promise<boolean> => {
      if (!onRunStudioAgentPrompt) return false;
      const appTitle = spec?.name || workingDecl.metadata.name || 'your Prism';
      setAgentPhaseLabel(`Building interactive ${appTitle}…`);
      const appPrompt = buildInteractiveAppPrompt(history, workingDecl);
      const rawApp = await onRunStudioAgentPrompt(
        appPrompt,
        `Build the complete, playable interactive ${appTitle} application now.`,
        3072,
        (chars) => {
          setAgentPhaseLabel(`Building interactive ${appTitle}… (${chars.toLocaleString()} chars)`);
        },
      );
      let extracted = extractAuthoredAppFromModelOutput(rawApp);

      // Single automatic format-repair retry if first pass missed one of the fenced blocks
      if (!extracted.ok) {
        setAgentPhaseLabel(`Refining ${appTitle} structure…`);
        const retryRaw = await onRunStudioAgentPrompt(
          appPrompt,
          `Your previous draft was incomplete (${extracted.reason}). Output ONLY the three complete fenced blocks: \`\`\`html, \`\`\`css, and \`\`\`js for ${appTitle}.`,
          3072,
          (chars) => {
            setAgentPhaseLabel(`Finishing ${appTitle}… (${chars.toLocaleString()} chars)`);
          },
        );
        extracted = extractAuthoredAppFromModelOutput(retryRaw);
      }

      if (!extracted.ok) {
        setTranscript((prev) => [
          ...prev,
          {
            id: `s_${Date.now()}_${Math.random().toString(36).slice(2, 5)}`,
            role: 'studio',
            text: `I drafted the Core Models for "${appTitle}", but the on-device model didn't finish the interactive app code (${extracted.reason}). Try saying "Build it again" or switching to a larger coding model in the top bar.`,
          },
        ]);
        setAgentQuickReplies(['Build it again', 'Keep it simpler and build now']);
        return false;
      }

      const assembled = assembleCompleteAuthoredPrismOps(workingDecl, spec, extracted.view, userMessage);
      if (assembled.ops.length > 0) {
        await onDispatchOp({ op: 'batch', ops: assembled.ops });
      }
      setTranscript((prev) => [
        ...prev,
        {
          id: `s_${Date.now()}_${Math.random().toString(36).slice(2, 5)}`,
          role: 'studio',
          text: `Your interactive ${appTitle} Prism is live in the Preview pane! All Core Models, safety boundaries, and applicable checks are reconciled. Play with it on the right, ask me to tweak anything in plain English, or click Prove & Seal when ready.`,
          badges: assembled.badges,
        },
      ]);
      setAgentQuickReplies([
        'Make it look sleeker in dark mode',
        'Add difficulty or mode options',
        'Add a reset & score tracker',
      ]);
      onSelectStep('ready-to-seal');
      onOpenPreview();
      return true;
    },
    [onDispatchOp, onOpenPreview, onRunStudioAgentPrompt, onSelectStep],
  );

  const runInteractiveRefinement = useCallback(
    async (
      instruction: string,
      errToFix?: PrismRuntimeError | null,
    ) => {
      if (!onRunStudioAgentPrompt || !declaration.view?.authoredView) return;
      setThinking(true);
      setAgentPhaseLabel(
        errToFix
          ? `Diagnosing & fixing "${errToFix.message.slice(0, 48)}"…`
          : `Updating ${declaration.metadata.name}…`,
      );
      try {
        const history: AgentChatTurn[] = [
          ...transcriptRef.current.map((e) => ({ role: e.role, text: e.text })),
          { role: 'user', text: instruction },
        ];
        const prompt = buildRefineInteractiveAppPrompt(
          history,
          declaration,
          declaration.view.authoredView,
          instruction,
          errToFix ?? null,
        );
        const raw = await onRunStudioAgentPrompt(
          prompt,
          errToFix ? `Fix the runtime error: ${errToFix.message}` : instruction,
          3072,
          (chars) => {
            setAgentPhaseLabel(
              errToFix
                ? `Applying fix to ${declaration.metadata.name}… (${chars.toLocaleString()} chars)`
                : `Updating ${declaration.metadata.name}… (${chars.toLocaleString()} chars)`,
            );
          },
        );
        const extracted = extractAuthoredAppFromModelOutput(raw);
        if (!extracted.ok) {
          setTranscript((prev) => [
            ...prev,
            {
              id: `s_${Date.now()}_${Math.random().toString(36).slice(2, 5)}`,
              role: 'studio',
              text: `I couldn't apply that update cleanly (${extracted.reason}). Could you rephrase what you'd like changed?`,
            },
          ]);
          return;
        }
        const assembled = assembleCompleteAuthoredPrismOps(declaration, null, extracted.view, instruction);
        if (assembled.ops.length > 0) {
          await onDispatchOp({ op: 'batch', ops: assembled.ops });
        }
        if (errToFix) {
          onClearRuntimeError();
        }
        setTranscript((prev) => [
          ...prev,
          {
            id: `s_${Date.now()}_${Math.random().toString(36).slice(2, 5)}`,
            role: 'studio',
            text: errToFix
              ? `Fixed the runtime error and refreshed "${declaration.metadata.name}" in Preview.`
              : `Updated "${declaration.metadata.name}" and reconciled all checks. Take a look in the Preview pane!`,
            badges: errToFix ? ['Fixed runtime error', 'Preview refreshed'] : assembled.badges,
          },
        ]);
        onOpenPreview();
      } catch (err) {
        setTranscript((prev) => [
          ...prev,
          {
            id: `s_${Date.now()}_${Math.random().toString(36).slice(2, 5)}`,
            role: 'studio',
            text: `Update interrupted: ${err instanceof Error ? err.message : String(err)}`,
          },
        ]);
      } finally {
        setThinking(false);
      }
    },
    [declaration, onClearRuntimeError, onDispatchOp, onOpenPreview, onRunStudioAgentPrompt],
  );

  const submitNaturalMessage = useCallback(
    async (rawMessage: string, opts?: { readonly skipUserBubble?: boolean }) => {
      const message = rawMessage.trim();
      if (!message || thinking) return;

      const currentTranscript = transcriptRef.current;
      const nextHistory: AgentChatTurn[] = [
        ...currentTranscript.map((e) => ({ role: e.role, text: e.text })),
        ...(opts?.skipUserBubble ? [] : [{ role: 'user' as const, text: message }]),
      ];

      if (!opts?.skipUserBubble) {
        const userEntry: WizardChatEntry = {
          id: `u_${Date.now()}_${Math.random().toString(36).slice(2, 5)}`,
          role: 'user',
          text: message,
        };
        setTranscript((prev) => [...prev, userEntry]);
        setInputText('');
      }

      const modeCheck = classifyIdeaRealizationMode(message);
      const isInteractiveFlow =
        interactiveSessionActive ||
        Boolean(declaration.view?.authoredView) ||
        ((routing.isUndefined || activeStep === 'anchor-idea') && modeCheck.mode === 'interactive-app');

      // Interactive Prism creation / refinement — powered by the 100% local Verdant Studio Agent
      if (isInteractiveFlow) {
        setInteractiveSessionActive(true);

        if (!hasLocalModelReady || !onRunStudioAgentPrompt) {
          const title =
            declaration.metadata.name && declaration.metadata.name !== 'Untitled Prism'
              ? declaration.metadata.name
              : extractCleanIdeaTitle(message);
          setPendingAgentMessage(message);
          setTranscript((prev) => [
            ...prev,
            {
              id: `s_${Date.now()}_${Math.random().toString(36).slice(2, 5)}`,
              role: 'studio',
              text: `I'd love to build a real, playable "${title}" Prism with you from scratch — no hardcoded templates. Turn on the on-device Studio Agent below (runs 100% locally in your browser) and I'll pick up right here automatically.`,
              needsLocalModel: true,
            },
          ]);
          return;
        }

        // Case A: Already built an interactive authored app -> refine it directly!
        if (declaration.view?.authoredView) {
          await runInteractiveRefinement(message, null);
          return;
        }

        // Case B: Discovery & Build flow with the Verdant Studio Agent
        setThinking(true);
        setAgentPhaseLabel('Understanding your idea & designing Core Models…');
        try {
          const userTurns = nextHistory.filter((t) => t.role === 'user').length;
          const discoveryPrompt = buildDiscoveryAgentPrompt(nextHistory, declaration, userTurns);
          const rawDiscovery = await onRunStudioAgentPrompt(
            discoveryPrompt,
            message,
            1024,
            (chars) => {
              setAgentPhaseLabel(`Drafting Core Models… (${chars.toLocaleString()} chars)`);
            },
          );
          const { spec } = parseDiscoveryAgentResponse(rawDiscovery, message, declaration, userTurns);

          // Apply Core Model anchor ops right away so the live Blueprint populates immediately
          const coreOps = buildAgentCoreModelOps(declaration, spec, message);
          let workingDecl = declaration;
          if (coreOps.length > 0) {
            for (const op of coreOps) {
              const r = dispatchStudioOp(workingDecl, op);
              if (r.ok) workingDecl = r.declaration;
            }
            await onDispatchOp({ op: 'batch', ops: coreOps });
          }

          if (spec.readyToBuild) {
            await runInteractiveBuildPass(nextHistory, workingDecl, spec, message);
          } else {
            setTranscript((prev) => [
              ...prev,
              {
                id: `s_${Date.now()}_${Math.random().toString(36).slice(2, 5)}`,
                role: 'studio',
                text: spec.reply,
                badges: [
                  `Concept: ${spec.name}`,
                  `${spec.inputs.length} core model input(s)`,
                ],
              },
            ]);
            setAgentQuickReplies(spec.quickReplies);
            onSelectStep('confirm-context');
          }
        } catch (err) {
          setTranscript((prev) => [
            ...prev,
            {
              id: `s_${Date.now()}_${Math.random().toString(36).slice(2, 5)}`,
              role: 'studio',
              text: `Studio Agent ran into an issue: ${err instanceof Error ? err.message : String(err)}. Try sending your message again.`,
            },
          ]);
        } finally {
          setThinking(false);
        }
        return;
      }

      // Structured numeric rule / policy / calculator flow (works instantaneously offline + local-model enhanced)
      let preParsedIntent: ExtractedPrismIntent | undefined;
      if (
        (routing.isUndefined || activeStep === 'anchor-idea') &&
        hasLocalModelReady &&
        onSynthesizeWithLocalModel
      ) {
        setThinking(true);
        setAgentPhaseLabel('Extracting your Prism structure…');
        try {
          const deterministic = parseIdeaSentence(message);
          preParsedIntent = await onSynthesizeWithLocalModel(message, deterministic);
        } catch {
          preParsedIntent = parseIdeaSentence(message);
        } finally {
          setThinking(false);
        }
      }

      const result = reconcileWizardTurn(declaration, activeStep, message, preParsedIntent);
      if (result.ops.length > 0) {
        await onDispatchOp({ op: 'batch', ops: result.ops });
      }

      const studioEntry: WizardChatEntry = {
        id: `s_${Date.now()}_${Math.random().toString(36).slice(2, 5)}`,
        role: 'studio',
        text: result.assistantReply,
        badges: result.reconciledBadges,
      };
      setTranscript((prev) => [...prev, studioEntry]);
      setAgentQuickReplies(null);
      onSelectStep(result.nextStep);
      if (result.openPreview) {
        onOpenPreview();
      }
    },
    [
      activeStep,
      declaration,
      hasLocalModelReady,
      interactiveSessionActive,
      onDispatchOp,
      onOpenPreview,
      onRunStudioAgentPrompt,
      onSelectStep,
      onSynthesizeWithLocalModel,
      routing.isUndefined,
      runInteractiveBuildPass,
      runInteractiveRefinement,
      thinking,
    ],
  );

  // Auto-resume queued open-ended request the instant the local on-device model finishes loading!
  useEffect(() => {
    if (!hasLocalModelReady || !pendingAgentMessage || thinking) return;
    const queued = pendingAgentMessage;
    setPendingAgentMessage(null);
    void submitNaturalMessage(queued, { skipUserBubble: true });
  }, [hasLocalModelReady, pendingAgentMessage, submitNaturalMessage, thinking]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    void submitNaturalMessage(inputText);
  };

  const quickReplyList = useMemo(() => {
    if (agentQuickReplies && agentQuickReplies.length > 0) return agentQuickReplies;
    if (routing.isUndefined || activeStep === 'anchor-idea') return STARTER_SUGGESTIONS;
    return currentPrompt.quickReplies;
  }, [activeStep, agentQuickReplies, currentPrompt.quickReplies, routing.isUndefined]);

  const showDefaultPromptCard =
    transcript.length === 0 ||
    (!interactiveSessionActive && !pendingAgentMessage) ||
    routing.readyToSeal;

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-col bg-white dark:bg-[#0E131B]">
      {/* Compact Plain-Language Progress Rail */}
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-1 border-b border-zinc-200 px-3 py-2 dark:border-zinc-800">
        <div className="flex flex-wrap items-center gap-1">
          {guided.steps
            .filter((s) => s.visible)
            .map((st) => {
              const active = activeStep === st.id;
              return (
                <button
                  key={st.id}
                  type="button"
                  onClick={() => onSelectStep(st.id)}
                  className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11px] font-medium transition-colors ${
                    active
                      ? 'bg-emerald-700 text-white'
                      : st.complete
                        ? 'bg-emerald-50 text-emerald-800 hover:bg-emerald-100 dark:bg-emerald-950/40 dark:text-emerald-300'
                        : 'bg-zinc-100 text-zinc-600 hover:bg-zinc-200 dark:bg-zinc-800/80 dark:text-zinc-400'
                  }`}
                >
                  {st.complete && <Check className="h-3 w-3" />}
                  <span>{st.label}</span>
                </button>
              );
            })}
        </div>
        <span
          className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium ${
            hasLocalModelReady
              ? 'bg-emerald-50 text-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-300'
              : 'bg-zinc-100 text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400'
          }`}
          title={
            hasLocalModelReady
              ? `Studio Agent ready (${loadedModelName ?? 'local model'})`
              : 'Turn on the local model to generate custom interactive Prisms from scratch'
          }
        >
          <Cpu className="h-2.5 w-2.5" />
          <span>{hasLocalModelReady ? 'Agent Ready' : isModelLoading ? `Loading ${Math.round(engineProgressPercent)}%` : 'Agent Off'}</span>
        </span>
      </div>

      {/* Continuous Conversational Transcript */}
      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3.5 text-xs">
        {transcript.map((entry) => (
          <div
            key={entry.id}
            className={`flex flex-col ${entry.role === 'user' ? 'items-end' : 'items-start'}`}
          >
            <div
              className={`max-w-[92%] rounded-2xl px-3.5 py-2.5 leading-relaxed ${
                entry.role === 'user'
                  ? 'bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900'
                  : 'border border-zinc-200 bg-zinc-50 text-zinc-800 dark:border-zinc-800 dark:bg-zinc-900/70 dark:text-zinc-200'
              }`}
            >
              <p className="whitespace-pre-line">{entry.text}</p>
              {entry.badges && entry.badges.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {entry.badges.map((b, bi) => (
                    <span
                      key={bi}
                      className="inline-flex items-center gap-1 rounded-md bg-emerald-100/90 px-2 py-0.5 text-[10px] font-medium text-emerald-900 dark:bg-emerald-950/70 dark:text-emerald-200"
                    >
                      <Check className="h-2.5 w-2.5" />
                      {b}
                    </span>
                  ))}
                </div>
              )}
              {entry.needsLocalModel && !hasLocalModelReady && (
                <div className="mt-2.5 space-y-2 rounded-xl border border-emerald-200 bg-white p-2.5 dark:border-emerald-900/60 dark:bg-[#0B0F14]">
                  {isModelLoading ? (
                    <div className="space-y-1.5">
                      <div className="flex items-center justify-between text-[11px] font-medium text-emerald-800 dark:text-emerald-300">
                        <span className="inline-flex items-center gap-1.5">
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          Loading on-device Studio Agent…
                        </span>
                        <span>{Math.round(engineProgressPercent)}%</span>
                      </div>
                      <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
                        <div
                          className="h-full rounded-full bg-emerald-600 transition-all"
                          style={{ width: `${Math.max(4, Math.min(100, engineProgressPercent))}%` }}
                        />
                      </div>
                      {engineStatusDetail && (
                        <p className="truncate text-[10px] text-zinc-500">{engineStatusDetail}</p>
                      )}
                    </div>
                  ) : (
                    <button
                      type="button"
                      onClick={onEnableModelStream}
                      className="inline-flex w-full items-center justify-center gap-1.5 rounded-lg bg-emerald-700 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-800"
                    >
                      <Cpu className="h-3.5 w-3.5" />
                      Turn On Local Studio Agent &amp; Continue
                    </button>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Live Runtime Error Self-Healing Card */}
        {runtimeError && declaration.view?.authoredView && hasLocalModelReady && !thinking && (
          <div className="rounded-xl border border-amber-200 bg-amber-50/80 p-3 text-xs text-amber-950 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-200">
            <p className="font-medium">Preview spotted an error: {runtimeError.message}</p>
            <button
              type="button"
              onClick={() =>
                void runInteractiveRefinement(
                  `Fix runtime error: ${runtimeError.message}`,
                  runtimeError,
                )
              }
              className="mt-2 inline-flex items-center gap-1.5 rounded-lg bg-amber-700 px-3 py-1 text-[11px] font-medium text-white hover:bg-amber-800"
            >
              <Wrench className="h-3 w-3" />
              Ask Studio Agent to Fix
            </button>
          </div>
        )}

        {/* Current Active Studio Prompt / Seal Card */}
        {showDefaultPromptCard && (
          <div className="flex flex-col items-start">
            <div className="max-w-[96%] space-y-2.5 rounded-2xl border border-emerald-200/90 bg-emerald-50/45 px-3.5 py-3 text-zinc-800 dark:border-emerald-900/60 dark:bg-emerald-950/20 dark:text-zinc-200">
              <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-emerald-700 dark:text-emerald-400">
                <Sparkles className="h-3 w-3" />
                <span>Verdant Studio</span>
              </div>
              <p className="leading-relaxed">
                {transcript.length === 0
                  ? 'What should your Prism be? Describe any app, game, or tool in plain English — like "This Prism will be the video game Minesweeper" — and we will design and build it together.'
                  : currentPrompt.promptText}
              </p>

              {/* Primary Prove & Publish CTA when completely reconciled */}
              {routing.readyToSeal && (
                <div className="flex flex-wrap gap-2 pt-1">
                  <button
                    type="button"
                    disabled={building}
                    onClick={onCheckAndBuild}
                    className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-700 px-3.5 py-1.5 text-xs font-medium text-white hover:bg-emerald-800 disabled:opacity-50"
                  >
                    {building ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <ShieldCheck className="h-3.5 w-3.5" />
                    )}
                    Prove &amp; Seal (.holo)
                  </button>
                  <button
                    type="button"
                    disabled={building}
                    onClick={onPublish}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-600 bg-white px-3.5 py-1.5 text-xs font-medium text-emerald-800 hover:bg-emerald-50 disabled:opacity-50 dark:border-emerald-700 dark:bg-transparent dark:text-emerald-200 dark:hover:bg-emerald-950/40"
                  >
                    <Globe className="h-3.5 w-3.5" />
                    Publish to App
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

        {thinking && (
          <div className="flex items-center gap-2 rounded-xl border border-emerald-200/70 bg-emerald-50/40 px-3 py-2 text-[11px] text-emerald-900 dark:border-emerald-900/50 dark:bg-emerald-950/20 dark:text-emerald-200">
            <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-emerald-600" />
            <span className="truncate">{agentPhaseLabel}</span>
          </div>
        )}
        <div ref={scrollEndRef} />
      </div>

      {/* Natural-Language Quick Replies + Persistent Chat Input */}
      <div className="shrink-0 space-y-2 border-t border-zinc-200 bg-zinc-50/70 p-3 dark:border-zinc-800 dark:bg-[#0B0F14]">
        {quickReplyList.length > 0 && !pendingAgentMessage && (
          <div className="flex flex-wrap gap-1.5">
            {quickReplyList.map((reply) => (
              <button
                key={reply}
                type="button"
                disabled={thinking}
                onClick={() => void submitNaturalMessage(reply)}
                className="rounded-full border border-zinc-200 bg-white px-2.5 py-1 text-left text-[11px] text-zinc-700 transition-colors hover:border-emerald-500 hover:bg-emerald-50/50 disabled:opacity-45 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300 dark:hover:border-emerald-600"
              >
                {reply}
              </button>
            ))}
          </div>
        )}

        <form onSubmit={handleSubmit} className="flex items-end gap-1.5">
          <textarea
            rows={2}
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                void submitNaturalMessage(inputText);
              }
            }}
            placeholder={
              declaration.view?.authoredView
                ? 'Ask to add features, change rules, or restyle your Prism…'
                : currentPrompt.placeholder
            }
            className="min-w-0 flex-1 resize-none rounded-xl border border-zinc-300 bg-white px-3 py-2 text-xs leading-relaxed text-zinc-900 placeholder:text-zinc-400 focus:border-emerald-600 focus:outline-none dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100"
          />
          <button
            type="submit"
            disabled={!inputText.trim() || thinking}
            className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-emerald-700 text-white hover:bg-emerald-800 disabled:opacity-45"
            title="Send response"
          >
            {thinking ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </button>
        </form>

        <div className="flex items-center justify-between text-[10px] text-zinc-500">
          <span>Describe any idea or change in plain English</span>
          <button
            type="button"
            onClick={onSwitchToAdvanced}
            className="inline-flex items-center gap-1 font-medium text-zinc-600 hover:text-emerald-700 dark:text-zinc-400 dark:hover:text-emerald-400"
          >
            <SlidersHorizontal className="h-3 w-3" />
            Advanced Model
          </button>
        </div>
      </div>
    </div>
  );
};
