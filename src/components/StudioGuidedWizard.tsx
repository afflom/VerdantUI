/**
 * Continuous Conversational Prism Wizard (`src/components/StudioGuidedWizard.tsx`)
 *
 * Pure natural-language creator experience over Verdant Studio's formal Core Models & Standards:
 * - Every stage of the wizard is populated and reconciled through natural-language dialogue —
 *   zero manual forms, checkboxes, or engineering fields required.
 * - Starts from a single-sentence idea prompt, then continues guiding the Creator through
 *   Context, Rules, Safety, Look & Feel, and Applicable Protections (or reconciles everything
 *   autonomously on request), updating the live Blueprint & Preview on every turn.
 * - Zero exposure of internal Verdant constructs (no Observer, OTel, vTensor, LexLean).
 * - Full granular editing remains available anytime via the `Advanced` toggle.
 */

import { useEffect, useMemo, useRef, useState, type FC, type FormEvent } from 'react';
import {
  Check,
  Globe,
  Loader2,
  Send,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
} from 'lucide-react';
import type {
  DeclarationCheckReport,
  PrismDeclaration,
  StudioApiOp,
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

interface StudioGuidedWizardProps {
  declaration: PrismDeclaration;
  routing: VerdantRoutingReport;
  checkReport: DeclarationCheckReport;
  activeStep: GuidedStepId;
  onSelectStep: (step: GuidedStepId) => void;
  onDispatchOp: (op: StudioApiOp, opts?: { readonly pushRevision?: boolean }) => Promise<boolean>;
  onSynthesizeWithLocalModel?: (sentence: string, fallback: ExtractedPrismIntent) => Promise<ExtractedPrismIntent>;
  hasLocalModelReady: boolean;
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
}

export const StudioGuidedWizard: FC<StudioGuidedWizardProps> = ({
  declaration,
  routing,
  checkReport,
  activeStep,
  onSelectStep,
  onDispatchOp,
  onSynthesizeWithLocalModel,
  hasLocalModelReady,
  onOpenPreview,
  onCheckAndBuild,
  onPublish,
  building,
  onSwitchToAdvanced,
}) => {
  const [inputText, setInputText] = useState('');
  const [thinking, setThinking] = useState(false);
  const [transcript, setTranscript] = useState<readonly WizardChatEntry[]>([]);
  const scrollEndRef = useRef<HTMLDivElement | null>(null);

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
  }, [transcript.length, activeStep, thinking]);

  const submitNaturalMessage = async (rawMessage: string, skipLocalModel = false) => {
    const message = rawMessage.trim();
    if (!message || thinking) return;

    const userEntry: WizardChatEntry = {
      id: `u_${Date.now()}_${Math.random().toString(36).slice(2, 5)}`,
      role: 'user',
      text: message,
    };
    setTranscript((prev) => [...prev, userEntry]);
    setInputText('');

    let preParsedIntent: ExtractedPrismIntent | undefined;
    if ((routing.isUndefined || activeStep === 'anchor-idea') && hasLocalModelReady && onSynthesizeWithLocalModel && !skipLocalModel) {
      setThinking(true);
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
    onSelectStep(result.nextStep);
    if (result.openPreview) {
      onOpenPreview();
    }
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    void submitNaturalMessage(inputText);
  };

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-col bg-white dark:bg-[#0E131B]">
      {/* Compact Plain-Language Progress Rail */}
      <div className="flex shrink-0 flex-wrap items-center gap-1 border-b border-zinc-200 px-3 py-2 dark:border-zinc-800">
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
              <p>{entry.text}</p>
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

        {/* Current Active Studio Prompt Card */}
        <div className="flex flex-col items-start">
          <div className="max-w-[96%] space-y-2.5 rounded-2xl border border-emerald-200/90 bg-emerald-50/45 px-3.5 py-3 text-zinc-800 dark:border-emerald-900/60 dark:bg-emerald-950/20 dark:text-zinc-200">
            <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-emerald-700 dark:text-emerald-400">
              <Sparkles className="h-3 w-3" />
              <span>Verdant Studio</span>
            </div>
            <p className="leading-relaxed">{currentPrompt.promptText}</p>

            {/* Primary Prove & Publish CTA when completely reconciled */}
            {routing.readyToSeal && (
              <div className="flex flex-wrap gap-2 pt-1">
                <button
                  type="button"
                  disabled={building}
                  onClick={onCheckAndBuild}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-700 px-3.5 py-1.5 text-xs font-medium text-white hover:bg-emerald-800 disabled:opacity-50"
                >
                  {building ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ShieldCheck className="h-3.5 w-3.5" />}
                  Prove &amp; Seal (.holo)
                </button>
                <button
                  type="button"
                  disabled={building}
                  onClick={onPublish}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-600 bg-white px-3 py-1.5 text-xs font-medium text-emerald-800 hover:bg-emerald-50 disabled:opacity-50 dark:border-emerald-700 dark:bg-transparent dark:text-emerald-200 dark:hover:bg-emerald-950/40"
                >
                  <Globe className="h-3.5 w-3.5" />
                  Publish to App
                </button>
              </div>
            )}
          </div>
        </div>

        {thinking && (
          <div className="flex items-center gap-2 text-[11px] text-zinc-500">
            <Loader2 className="h-3.5 w-3.5 animate-spin text-emerald-600" />
            <span>Reconciling your Prism model…</span>
          </div>
        )}
        <div ref={scrollEndRef} />
      </div>

      {/* Natural-Language Quick Replies + Persistent Chat Input */}
      <div className="shrink-0 space-y-2 border-t border-zinc-200 bg-zinc-50/70 p-3 dark:border-zinc-800 dark:bg-[#0B0F14]">
        {currentPrompt.quickReplies.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {currentPrompt.quickReplies.map((reply) => (
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
            placeholder={currentPrompt.placeholder}
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
          <span>Describe any change in plain English</span>
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
