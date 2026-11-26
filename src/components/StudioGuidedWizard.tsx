/**
 * Conversational Verdant Studio Weight-Routing Wizard (`src/components/StudioGuidedWizard.tsx`)
 *
 * Implements DEC-009 (Zero LLM-Written JavaScript) & DEC-011 (Plain Human Language):
 * - Classifies the user's idea across the 7 MediaTypes (Picture, Audio, Written Work, Video,
 *   Game, Research Program, Tool) and activates ONLY the relevant Studio features.
 * - Routes structured declarative weights (`route-prism-weights`) — NEVER generates raw JS.
 * - Supports importing & attaching local media weight files (.gguf / .safetensors) onto
 *   compatible Picture, Audio, Video, Game, or Written-Work Prisms.
 * - Works smoothly both with the on-device local model AND instantaneously out of the box.
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type FC,
  type FormEvent,
} from 'react';
import {
  Check,
  Cpu,
  Globe,
  Loader2,
  Send,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Square,
  Upload,
} from 'lucide-react';
import {
  dispatchStudioOp,
  type DeclarationCheckReport,
  type PrismDeclaration,
  type StudioApiOp,
} from '../engine/verdantStudioApi';
import type { VerdantRoutingReport } from '../engine/verdantStandardsRouter';
import type { StudioChatMessage } from '../engine/prismStudio';
import {
  buildStudioPromptForStep,
  computeGuidedWizardState,
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
import {
  compatibleModalitiesForMediaType,
  createImportedWeightBinding,
  PRISM_MEDIA_CATALOG,
  type MediaModelWeightBinding,
  type PrismMediaType,
} from '../engine/prismMediaOrgans';
import type { EngineStatus } from '../engine/workerClient';
import type { PrismRuntimeError } from './PrismSandboxFrame';

interface StudioGuidedWizardProps {
  declaration: PrismDeclaration;
  routing: VerdantRoutingReport;
  checkReport: DeclarationCheckReport;
  messages: readonly StudioChatMessage[];
  onAppendMessage: (msg: {
    readonly role: 'user' | 'assistant';
    readonly content: string;
    readonly badges?: readonly string[];
    readonly needsLocalModel?: boolean;
  }) => Promise<void>;
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
  onCancelStudioAgent?: () => void;
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

const STARTER_SUGGESTIONS: readonly string[] = [
  'This Prism will be the video game "Minesweeper"',
  'Create an emerald geometric sunburst picture',
  'Create a calm ambient synth soundscape',
  'Healthy Habits: protect 5pm to 6pm daily for exercise',
];

const MEDIA_PILL_ORDER: readonly PrismMediaType[] = [
  'interactive-game',
  'picture',
  'audio',
  'written-work',
  'video',
  'research-program',
  'tool-organ',
];

function isAbortError(err: unknown): boolean {
  return (
    (err instanceof DOMException && err.name === 'AbortError') ||
    (err instanceof Error && (err.name === 'AbortError' || /aborted|cancelled/i.test(err.message)))
  );
}

export const StudioGuidedWizard: FC<StudioGuidedWizardProps> = ({
  declaration,
  routing,
  checkReport,
  messages,
  onAppendMessage,
  activeStep,
  onSelectStep,
  onDispatchOp,
  onSynthesizeWithLocalModel,
  onRunStudioAgentPrompt,
  onCancelStudioAgent,
  hasLocalModelReady,
  engineStatus,
  engineProgressPercent,
  loadedModelName,
  onOpenPreview,
  onCheckAndBuild,
  onPublish,
  building,
  onSwitchToAdvanced,
}) => {
  const [inputText, setInputText] = useState('');
  const [thinking, setThinking] = useState(false);
  const [agentPhaseLabel, setAgentPhaseLabel] = useState('Routing Prism weights…');
  const [agentQuickReplies, setAgentQuickReplies] = useState<readonly string[] | null>(null);
  const [selectedMediaType, setSelectedMediaType] = useState<PrismMediaType | null>(
    () => declaration.view?.proceduralOrgan?.mediaType ?? declaration.concept.mediaType ?? null,
  );
  const [attachedWeights, setAttachedWeights] = useState<readonly MediaModelWeightBinding[]>(
    () => declaration.view?.proceduralOrgan?.attachedWeights ?? [],
  );

  const scrollEndRef = useRef<HTMLDivElement | null>(null);
  const weightFileInputRef = useRef<HTMLInputElement | null>(null);
  const messagesRef = useRef(messages);
  messagesRef.current = messages;
  const runTokenRef = useRef(0);

  useEffect(() => {
    const declMedia = declaration.view?.proceduralOrgan?.mediaType ?? declaration.concept.mediaType ?? null;
    if (declMedia) setSelectedMediaType(declMedia);
    if (declaration.view?.proceduralOrgan?.attachedWeights) {
      setAttachedWeights(declaration.view.proceduralOrgan.attachedWeights);
    }
  }, [declaration.concept.mediaType, declaration.view?.proceduralOrgan]);

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
  }, [messages.length, activeStep, thinking]);

  const isModelLoading = engineStatus === 'downloading' || engineStatus === 'loading';

  const handleStopAgent = useCallback(() => {
    runTokenRef.current += 1;
    onCancelStudioAgent?.();
    setThinking(false);
  }, [onCancelStudioAgent]);

  const handleAttachWeightFile = useCallback(
    async (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      e.target.value = '';
      if (!file) return;
      const activeMedia: PrismMediaType = selectedMediaType ?? 'picture';
      const modality = compatibleModalitiesForMediaType(activeMedia)[0] ?? 'text';
      const binding = createImportedWeightBinding({
        fileName: file.name,
        byteLength: file.size,
        modality,
      });
      const nextBindings = [
        ...attachedWeights.filter((b) => b.id !== binding.id),
        binding,
      ];
      setAttachedWeights(nextBindings);

      if (declaration.view?.proceduralOrgan) {
        const primaryReq = declaration.requirements[0]?.id ?? 'REQ-01';
        await onDispatchOp({
          op: 'route-prism-weights',
          mediaType: declaration.view.proceduralOrgan.mediaType,
          activeOrgans: declaration.view.proceduralOrgan.activeOrgans,
          weights: declaration.view.proceduralOrgan.weights,
          attachedWeights: nextBindings,
          compartmentId: declaration.concept.compartmentId,
          citation: { kind: 'req', requirementId: primaryReq },
        });
      }

      await onAppendMessage({
        role: 'assistant',
        content: `Attached local ${modality} model weights "${file.name}" (${(file.size / (1024 * 1024)).toFixed(1)} MB) to this ${PRISM_MEDIA_CATALOG[activeMedia].label} Prism.`,
        badges: [`Routed ${modality} weights`],
      });
    },
    [attachedWeights, declaration, onAppendMessage, onDispatchOp, selectedMediaType],
  );

  const submitNaturalMessage = useCallback(
    async (rawMessage: string) => {
      const message = rawMessage.trim();
      if (!message || thinking) return;

      const nextHistory: AgentChatTurn[] = [
        ...messagesRef.current.map((m) => ({
          role: m.role === 'user' ? ('user' as const) : ('studio' as const),
          text: m.content,
        })),
        { role: 'user', text: message },
      ];

      setInputText('');
      await onAppendMessage({ role: 'user', content: message });

      const classified = classifyArtifactMediaType(message);
      const isProceduralFlow =
        selectedMediaType !== null ||
        Boolean(declaration.view?.proceduralOrgan) ||
        classified.mode === 'procedural-organ';

      if (isProceduralFlow) {
        const activeMediaType: PrismMediaType =
          selectedMediaType ??
          declaration.view?.proceduralOrgan?.mediaType ??
          declaration.concept.mediaType ??
          classified.mediaType;
        setSelectedMediaType(activeMediaType);

        const runToken = ++runTokenRef.current;
        const userTurns = nextHistory.filter((t) => t.role === 'user').length;
        const isRefining = Boolean(declaration.view?.proceduralOrgan);

        let rawModelJson = '';
        if (hasLocalModelReady && onRunStudioAgentPrompt) {
          setThinking(true);
          setAgentPhaseLabel(
            isRefining
              ? `Routing updated ${PRISM_MEDIA_CATALOG[activeMediaType].label} weights…`
              : `Routing ${PRISM_MEDIA_CATALOG[activeMediaType].label} features…`,
          );
          try {
            const prompt = isRefining
              ? buildWeightRefinementPrompt(nextHistory, declaration, message)
              : buildWeightRoutingDiscoveryPrompt(nextHistory, declaration, userTurns, activeMediaType);
            rawModelJson = await onRunStudioAgentPrompt(prompt, message, 768, (chars) => {
              if (runTokenRef.current === runToken) {
                setAgentPhaseLabel(`Routing Prism weights… (${chars.toLocaleString()} chars)`);
              }
            });
          } catch (err) {
            if (isAbortError(err) || runTokenRef.current !== runToken) {
              setThinking(false);
              await onAppendMessage({
                role: 'assistant',
                content: 'Stopped routing. Send another message whenever you are ready.',
              });
              return;
            }
            // Fall through cleanly to deterministic weight routing if local inference errored
            rawModelJson = '';
          } finally {
            if (runTokenRef.current === runToken) {
              setThinking(false);
            }
          }
          if (runTokenRef.current !== runToken) return;
        }

        const { spec } = parseDiscoveryAgentResponse(
          rawModelJson,
          message,
          declaration,
          userTurns,
          activeMediaType,
        );

        if (spec.readyToBuild || isRefining) {
          const assembled = assembleProceduralPrismOps(declaration, spec, message);
          const finalOps =
            attachedWeights.length > 0
              ? assembled.ops.map((op) =>
                  op.op === 'route-prism-weights' ? { ...op, attachedWeights } : op,
                )
              : assembled.ops;
          if (finalOps.length > 0) {
            await onDispatchOp({ op: 'batch', ops: finalOps });
          }
          await onAppendMessage({
            role: 'assistant',
            content: isRefining
              ? `Updated your ${PRISM_MEDIA_CATALOG[spec.mediaType].label} Prism "${spec.name}" and refreshed Preview.`
              : `Your ${PRISM_MEDIA_CATALOG[spec.mediaType].label} Prism "${spec.name}" is synthesized and live in Preview! Ask me in plain English to adjust any settings, rules, or style.`,
            badges: assembled.badges,
          });
          setAgentQuickReplies(spec.quickReplies);
          onSelectStep('ready-to-seal');
          onOpenPreview();
        } else {
          // Anchor Core Models right away so the live Blueprint updates immediately while asking 1–2 quick questions
          const coreOps = buildAgentCoreModelOps(declaration, spec, message);
          if (coreOps.length > 0) {
            let working = declaration;
            for (const op of coreOps) {
              const r = dispatchStudioOp(working, op);
              if (r.ok) working = r.declaration;
            }
            await onDispatchOp({ op: 'batch', ops: coreOps });
          }
          await onAppendMessage({
            role: 'assistant',
            content: spec.reply,
            badges: [
              PRISM_MEDIA_CATALOG[spec.mediaType].label,
              `${spec.activeOrgans.length} active feature(s)`,
            ],
          });
          setAgentQuickReplies(spec.quickReplies);
          onSelectStep('confirm-context');
        }
        return;
      }

      // Structured numeric policy / calculator flow
      let preParsedIntent: ExtractedPrismIntent | undefined;
      if (
        (routing.isUndefined || activeStep === 'anchor-idea') &&
        hasLocalModelReady &&
        onSynthesizeWithLocalModel
      ) {
        const runToken = ++runTokenRef.current;
        setThinking(true);
        setAgentPhaseLabel('Extracting your Prism rules…');
        try {
          const deterministic = parseIdeaSentence(message);
          preParsedIntent = await onSynthesizeWithLocalModel(message, deterministic);
        } catch {
          preParsedIntent = parseIdeaSentence(message);
        } finally {
          if (runTokenRef.current === runToken) setThinking(false);
        }
      }

      const result = reconcileWizardTurn(declaration, activeStep, message, preParsedIntent);
      if (result.ops.length > 0) {
        await onDispatchOp({ op: 'batch', ops: result.ops });
      }
      await onAppendMessage({
        role: 'assistant',
        content: result.assistantReply,
        badges: result.reconciledBadges,
      });
      setAgentQuickReplies(null);
      onSelectStep(result.nextStep);
      if (result.openPreview) onOpenPreview();
    },
    [
      activeStep,
      attachedWeights,
      declaration,
      hasLocalModelReady,
      onAppendMessage,
      onDispatchOp,
      onOpenPreview,
      onRunStudioAgentPrompt,
      onSelectStep,
      onSynthesizeWithLocalModel,
      routing.isUndefined,
      selectedMediaType,
      thinking,
    ],
  );

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    void submitNaturalMessage(inputText);
  };

  const quickReplyList = useMemo(() => {
    if (agentQuickReplies && agentQuickReplies.length > 0) return agentQuickReplies;
    if (routing.isUndefined || activeStep === 'anchor-idea') return STARTER_SUGGESTIONS;
    return currentPrompt.quickReplies;
  }, [activeStep, agentQuickReplies, currentPrompt.quickReplies, routing.isUndefined]);

  const activeMediaForUi =
    selectedMediaType ?? declaration.view?.proceduralOrgan?.mediaType ?? declaration.concept.mediaType ?? null;

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
              ? `Local assistant active (${loadedModelName ?? 'on-device model'})`
              : 'Instant procedural studio ready (turn on a local model for richer weight routing)'
          }
        >
          <Cpu className="h-2.5 w-2.5" />
          <span>
            {hasLocalModelReady
              ? 'Local AI Ready'
              : isModelLoading
                ? `Loading ${Math.round(engineProgressPercent)}%`
                : 'Instant Studio'}
          </span>
        </span>
      </div>

      {/* Sparse Artifact Type Selector Rail */}
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-1.5 border-b border-zinc-100 bg-zinc-50/70 px-3 py-1.5 dark:border-zinc-800/80 dark:bg-[#0B0F14]">
        <div className="flex flex-wrap items-center gap-1">
          {MEDIA_PILL_ORDER.map((mt) => {
            const selected = activeMediaForUi === mt;
            return (
              <button
                key={mt}
                type="button"
                onClick={() => setSelectedMediaType(selected ? null : mt)}
                className={`rounded-full px-2 py-0.5 text-[10px] font-medium transition-colors ${
                  selected
                    ? 'bg-emerald-700 text-white'
                    : 'border border-zinc-200 bg-white text-zinc-600 hover:border-emerald-400 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300'
                }`}
              >
                {PRISM_MEDIA_CATALOG[mt].label}
              </button>
            );
          })}
        </div>
        <div className="flex items-center gap-1">
          <input
            ref={weightFileInputRef}
            type="file"
            accept=".gguf,.safetensors,.bin"
            className="hidden"
            onChange={(e) => void handleAttachWeightFile(e)}
          />
          <button
            type="button"
            onClick={() => weightFileInputRef.current?.click()}
            className="inline-flex items-center gap-1 rounded-md border border-zinc-200 bg-white px-2 py-0.5 text-[10px] font-medium text-zinc-700 hover:border-emerald-500 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300"
            title="Import local audio, image, video, or text model weights to enhance this Prism"
          >
            <Upload className="h-2.5 w-2.5" />
            <span>
              {attachedWeights.length > 0
                ? `${attachedWeights.length} Model Weight${attachedWeights.length === 1 ? '' : 's'}`
                : 'Attach Weights'}
            </span>
          </button>
        </div>
      </div>

      {/* Persistent Unified Conversational Transcript */}
      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3.5 text-xs">
        {messages.map((entry) => (
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
              <p className="whitespace-pre-line">{entry.content}</p>
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
            </div>
          </div>
        ))}

        {(messages.length === 0 || routing.readyToSeal) && (
          <div className="flex flex-col items-start">
            <div className="max-w-[96%] space-y-2.5 rounded-2xl border border-emerald-200/90 bg-emerald-50/45 px-3.5 py-3 text-zinc-800 dark:border-emerald-900/60 dark:bg-emerald-950/20 dark:text-zinc-200">
              <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-emerald-700 dark:text-emerald-400">
                <Sparkles className="h-3 w-3" />
                <span>Verdant Studio</span>
              </div>
              <p className="leading-relaxed">
                {messages.length === 0
                  ? 'What would you like to create? Pick an artifact type above or describe a game, picture, audio track, novel, video, research program, or personal tool in plain English.'
                  : currentPrompt.promptText}
              </p>

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
                    Verify &amp; Seal
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
          <div className="flex items-center justify-between gap-2 rounded-xl border border-emerald-200/70 bg-emerald-50/40 px-3 py-2 text-[11px] text-emerald-900 dark:border-emerald-900/50 dark:bg-emerald-950/20 dark:text-emerald-200">
            <div className="flex min-w-0 items-center gap-2">
              <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-emerald-600" />
              <span className="truncate">{agentPhaseLabel}</span>
            </div>
            {onCancelStudioAgent && (
              <button
                type="button"
                onClick={handleStopAgent}
                className="inline-flex shrink-0 items-center gap-1 rounded-md border border-emerald-300 bg-white px-2 py-0.5 text-[10px] font-medium text-emerald-900 hover:bg-emerald-50 dark:border-emerald-800 dark:bg-zinc-900 dark:text-emerald-200"
              >
                <Square className="h-2.5 w-2.5 fill-current" />
                Stop
              </button>
            )}
          </div>
        )}
        <div ref={scrollEndRef} />
      </div>

      {/* Natural-Language Quick Replies + Persistent Chat Input */}
      <div className="shrink-0 space-y-2 border-t border-zinc-200 bg-zinc-50/70 p-3 dark:border-zinc-800 dark:bg-[#0B0F14]">
        {quickReplyList.length > 0 && (
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
              declaration.view?.proceduralOrgan
                ? 'Ask to change difficulty, board size, style, schedule window, or rules…'
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
          <span>Describe any idea or tweak in plain English</span>
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
