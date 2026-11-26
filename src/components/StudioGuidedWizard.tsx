/**
 * Conversational Verdant Studio Creator (`src/components/StudioGuidedWizard.tsx`)
 *
 * Implements DEC-011 (Zero Technical Jargon), DEC-012 (Zero User Tunables / Zero Manual Uploads),
 * and DEC-013 (Automatic Natural-Language Media Selection — no media-type or space button clutter).
 */

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FC,
  type FormEvent,
} from 'react';
import {
  Check,
  Globe,
  Loader2,
  Send,
  Sparkles,
  Square,
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
import type { EngineStatus } from '../engine/workerClient';

interface StudioGuidedWizardProps {
  declaration: PrismDeclaration;
  routing: VerdantRoutingReport;
  checkReport: DeclarationCheckReport;
  messages: readonly StudioChatMessage[];
  onAppendMessage: (msg: {
    readonly role: 'user' | 'assistant';
    readonly content: string;
    readonly badges?: readonly string[];
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
  onOpenPreview: () => void;
  onPublish: () => void;
  publishing: boolean;
}

const STARTER_SUGGESTIONS: readonly string[] = [
  'Make the game "Minesweeper"',
  'Create an emerald geometric sunburst picture',
  'Compose a calm ambient soundscape',
  'Healthy Habits: protect 5pm to 6pm daily for exercise',
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
  onOpenPreview,
  onPublish,
  publishing,
}) => {
  const [inputText, setInputText] = useState('');
  const [thinking, setThinking] = useState(false);
  const [phaseLabel, setPhaseLabel] = useState('Designing…');
  const [agentQuickReplies, setAgentQuickReplies] = useState<readonly string[] | null>(null);

  const scrollEndRef = useRef<HTMLDivElement | null>(null);
  const messagesRef = useRef(messages);
  messagesRef.current = messages;
  const runTokenRef = useRef(0);

  const currentPrompt = useMemo(
    () => buildStudioPromptForStep(declaration, routing, checkReport, activeStep),
    [declaration, routing, checkReport, activeStep],
  );

  useEffect(() => {
    scrollEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages.length, activeStep, thinking]);

  const isLoadingAi = engineStatus === 'downloading' || engineStatus === 'loading';

  const handleStopAgent = useCallback(() => {
    runTokenRef.current += 1;
    onCancelStudioAgent?.();
    setThinking(false);
  }, [onCancelStudioAgent]);

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
        Boolean(declaration.view?.proceduralOrgan) ||
        Boolean(declaration.concept.mediaType) ||
        classified.mode === 'procedural-organ';

      if (isProceduralFlow) {
        const activeMediaType: PrismMediaType =
          declaration.view?.proceduralOrgan?.mediaType ??
          declaration.concept.mediaType ??
          classified.mediaType;

        const runToken = ++runTokenRef.current;
        const userTurns = nextHistory.filter((t) => t.role === 'user').length;
        const isRefining = Boolean(declaration.view?.proceduralOrgan);

        let rawModelJson = '';
        if (hasLocalModelReady && onRunStudioAgentPrompt) {
          setThinking(true);
          setPhaseLabel(
            isRefining
              ? `Updating your ${PRISM_MEDIA_CATALOG[activeMediaType].label.toLowerCase()}…`
              : `Creating your ${PRISM_MEDIA_CATALOG[activeMediaType].label.toLowerCase()}…`,
          );
          try {
            const prompt = isRefining
              ? buildWeightRefinementPrompt(nextHistory, declaration, message)
              : buildWeightRoutingDiscoveryPrompt(nextHistory, declaration, userTurns, activeMediaType);
            rawModelJson = await onRunStudioAgentPrompt(prompt, message, 768);
          } catch (err) {
            if (isAbortError(err) || runTokenRef.current !== runToken) {
              setThinking(false);
              await onAppendMessage({
                role: 'assistant',
                content: 'Stopped. Send another message whenever you are ready.',
              });
              return;
            }
            rawModelJson = '';
          } finally {
            if (runTokenRef.current === runToken) setThinking(false);
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
          if (assembled.ops.length > 0) {
            await onDispatchOp({ op: 'batch', ops: assembled.ops });
          }
          await onAppendMessage({
            role: 'assistant',
            content: isRefining
              ? `Updated "${spec.name}" and refreshed the preview.`
              : `"${spec.name}" is ready in Preview! Tell me in plain English if you want to change anything.`,
            badges: [PRISM_MEDIA_CATALOG[spec.mediaType].label],
          });
          setAgentQuickReplies(spec.quickReplies);
          onSelectStep('ready-to-seal');
          onOpenPreview();
        } else {
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
            badges: [PRISM_MEDIA_CATALOG[spec.mediaType].label],
          });
          setAgentQuickReplies(spec.quickReplies);
          onSelectStep('confirm-context');
        }
        return;
      }

      let preParsedIntent: ExtractedPrismIntent | undefined;
      if (
        (routing.isUndefined || activeStep === 'anchor-idea') &&
        hasLocalModelReady &&
        onSynthesizeWithLocalModel
      ) {
        const runToken = ++runTokenRef.current;
        setThinking(true);
        setPhaseLabel('Understanding your idea…');
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
      });
      setAgentQuickReplies(null);
      onSelectStep(result.nextStep);
      if (result.openPreview) onOpenPreview();
    },
    [
      activeStep,
      declaration,
      hasLocalModelReady,
      onAppendMessage,
      onDispatchOp,
      onOpenPreview,
      onRunStudioAgentPrompt,
      onSelectStep,
      onSynthesizeWithLocalModel,
      routing.isUndefined,
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

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-col bg-white dark:bg-[#0E131B]">
      {isLoadingAi && (
        <div className="flex shrink-0 items-center justify-end border-b border-zinc-100 bg-white px-3 py-1.5 text-[11px] dark:border-zinc-800/80 dark:bg-[#0E131B]">
          <span className="text-[10px] font-medium text-zinc-500 tabular-nums">
            Preparing AI {Math.round(engineProgressPercent)}%
          </span>
        </div>
      )}

      {/* Conversational Transcript */}
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

        {(messages.length === 0 || routing.readyToPreview) && (
          <div className="flex flex-col items-start">
            <div className="max-w-[96%] space-y-2.5 rounded-2xl border border-emerald-200/90 bg-emerald-50/45 px-3.5 py-3 text-zinc-800 dark:border-emerald-900/60 dark:bg-emerald-950/20 dark:text-zinc-200">
              <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-emerald-700 dark:text-emerald-400">
                <Sparkles className="h-3 w-3" />
                <span>Verdant Studio</span>
              </div>
              <p className="leading-relaxed">
                {messages.length === 0
                  ? 'What would you like to make? Describe a game, picture, audio track, story, video, research space, or everyday tool in plain English.'
                  : 'Your creation is ready. Ask for any adjustments, or publish it to use alongside your conversations.'}
              </p>

              {routing.readyToPreview && (
                <div className="flex flex-wrap gap-2 pt-1">
                  <button
                    type="button"
                    disabled={publishing}
                    onClick={onPublish}
                    className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-700 px-3.5 py-1.5 text-xs font-medium text-white hover:bg-emerald-800 disabled:opacity-50"
                  >
                    {publishing ? (
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    ) : (
                      <Globe className="h-3.5 w-3.5" />
                    )}
                    Publish
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
              <span className="truncate">{phaseLabel}</span>
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

      {/* Quick Suggestions + Input */}
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
                ? 'Ask to change difficulty, size, style, schedule, or rules…'
                : 'Describe what you want to make…'
            }
            className="min-w-0 flex-1 resize-none rounded-xl border border-zinc-300 bg-white px-3 py-2 text-xs leading-relaxed text-zinc-900 placeholder:text-zinc-400 focus:border-emerald-600 focus:outline-none dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100"
          />
          <button
            type="submit"
            disabled={!inputText.trim() || thinking}
            className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-emerald-700 text-white hover:bg-emerald-800 disabled:opacity-45"
            title="Send"
          >
            {thinking ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </button>
        </form>
      </div>
    </div>
  );
};
