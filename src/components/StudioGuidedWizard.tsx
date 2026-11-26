/**
 * Conversational Prompt-Driven Prism Wizard (`src/components/StudioGuidedWizard.tsx`)
 *
 * Calm, plain-English creator flow over Verdant Studio's formal Core Models & Standards Router:
 * - Zero raw engineering jargon and zero internal Verdant constructs (no Observer, OTel, vTensor, LexLean).
 * - Starts with a single-sentence idea prompt, transparently distinguishing facts extracted
 *   "From your idea" vs "Suggested" defaults (R2 honesty).
 * - Progressively guides the Creator through:
 *   1. Idea Anchor -> 2. Audience & Data -> 3. Inputs & Rules -> 4. Safety -> 5. Look & Feel
 *   -> [Compliance (only if triggered)] -> Prove & Publish.
 * - Every action dispatches strictly through closed `dispatchStudioOp()` calls (full Undo support).
 */

import { useMemo, useState, type FC, type FormEvent } from 'react';
import {
  AlertTriangle,
  ArrowRight,
  Check,
  CheckCircle2,
  Eye,
  Globe,
  Loader2,
  Palette,
  Scale,
  Shield,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Trash2,
} from 'lucide-react';
import {
  PRISM_THEME_PRESETS,
  type DataClassification,
  type DeclarationCheckReport,
  type PrismAudience,
  type PrismDeclaration,
  type PrismDomain,
  type PrismJurisdiction,
  type PrismViewpoint,
  type StudioApiOp,
} from '../engine/verdantStudioApi';
import type { VerdantRoutingReport } from '../engine/verdantStandardsRouter';
import {
  INSPIRATION_PROMPTS,
  computeGuidedWizardState,
  intentToAnchorOps,
  parseFollowUpInstruction,
  parseIdeaSentence,
  type ExtractedPrismIntent,
  type FollowUpParseResult,
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

const AUDIENCE_CHOICES: readonly { readonly id: PrismAudience; readonly label: string; readonly note: string }[] = [
  { id: 'personal', label: 'Just me', note: 'Private, offline personal tool' },
  { id: 'team', label: 'My team', note: 'Shared internal policy or calculator' },
  { id: 'public-adults', label: 'Public (adults)', note: 'Customers or public visitors' },
  { id: 'includes-minors', label: 'Includes kids / minors', note: 'Adds child-safety protections' },
];

const DATA_CHOICES: readonly { readonly id: DataClassification; readonly label: string }[] = [
  { id: 'none', label: 'No personal info' },
  { id: 'operational', label: 'Business / operational' },
  { id: 'personal', label: 'Personal info (age, score, etc.)' },
  { id: 'sensitive-regulated', label: 'Sensitive / health' },
];

const REGION_CHOICES: readonly { readonly id: PrismJurisdiction; readonly label: string }[] = [
  { id: 'none', label: 'Local / personal' },
  { id: 'us', label: 'United States' },
  { id: 'eu', label: 'European Union' },
  { id: 'uk', label: 'United Kingdom' },
  { id: 'global', label: 'Global' },
];

const DOMAIN_LABELS: Record<PrismDomain, string> = {
  general: 'General Utility',
  finance: 'Finance & Budgeting',
  health: 'Health & Wellness',
  education: 'Education',
  'games-media': 'Games & Media',
  'enterprise-ops': 'Operations & Policy',
};

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
  const [ideaText, setIdeaText] = useState(() => declaration.decisions[0]?.quote ?? declaration.concept.purpose ?? '');
  const [analyzing, setAnalyzing] = useState(false);
  const [draftIntent, setDraftIntent] = useState<ExtractedPrismIntent | null>(null);
  const [selectedViewpoint, setSelectedViewpoint] = useState<PrismViewpoint>(
    () => declaration.view?.viewpoint ?? 'decision-form',
  );

  // Step 3 plain-English refinement input
  const [followUpText, setFollowUpText] = useState('');
  const [followUpResult, setFollowUpResult] = useState<FollowUpParseResult | null>(null);

  const guided = useMemo(
    () => computeGuidedWizardState(declaration, routing, checkReport),
    [declaration, routing, checkReport],
  );

  const handleAnalyzeIdea = async (e?: FormEvent) => {
    e?.preventDefault();
    const trimmed = ideaText.trim();
    if (!trimmed) return;

    const deterministic = parseIdeaSentence(trimmed);
    if (hasLocalModelReady && onSynthesizeWithLocalModel) {
      setAnalyzing(true);
      try {
        const enhanced = await onSynthesizeWithLocalModel(trimmed, deterministic);
        setDraftIntent(enhanced);
        setSelectedViewpoint(enhanced.recommendedViewpoint);
      } catch {
        setDraftIntent(deterministic);
        setSelectedViewpoint(deterministic.recommendedViewpoint);
      } finally {
        setAnalyzing(false);
      }
      return;
    }

    setDraftIntent(deterministic);
    setSelectedViewpoint(deterministic.recommendedViewpoint);
  };

  const handleUseFastParserNow = () => {
    const trimmed = ideaText.trim();
    if (!trimmed) return;
    const deterministic = parseIdeaSentence(trimmed);
    setAnalyzing(false);
    setDraftIntent(deterministic);
    setSelectedViewpoint(deterministic.recommendedViewpoint);
  };

  const handleApplyIntent = async () => {
    if (!draftIntent) return;
    const ops = intentToAnchorOps(declaration, draftIntent);
    const ok = await onDispatchOp({ op: 'batch', ops });
    if (ok) {
      setDraftIntent(null);
      onSelectStep('confirm-context');
    }
  };

  const handleToggleRegion = async (jur: PrismJurisdiction) => {
    const cur = declaration.concept.jurisdictions;
    const next: PrismJurisdiction[] =
      jur === 'none'
        ? ['none']
        : cur.includes(jur)
          ? cur.filter((j) => j !== jur)
          : [...cur.filter((j) => j !== 'none'), jur];
    await onDispatchOp({ op: 'set-concept-profile', jurisdictions: next.length > 0 ? next : ['none'] });
  };

  const handleParseFollowUp = (e: FormEvent) => {
    e.preventDefault();
    if (!followUpText.trim()) return;
    setFollowUpResult(parseFollowUpInstruction(declaration, followUpText));
  };

  const handleApplyFollowUp = async () => {
    if (!followUpResult || !followUpResult.ok) return;
    const ok = await onDispatchOp({ op: 'batch', ops: followUpResult.ops });
    if (ok) {
      setFollowUpText('');
      setFollowUpResult(null);
    }
  };

  const handleSynthesizeDesign = async () => {
    const ok = await onDispatchOp({ op: 'synthesize-view-from-system', viewpoint: selectedViewpoint });
    if (ok) {
      onOpenPreview();
      onSelectStep(routing.activeLayeredStandards.length > 0 ? 'routed-standards' : 'ready-to-seal');
    }
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

      {/* Main Conversational Body */}
      <div className="min-h-0 flex-1 overflow-y-auto p-4 text-xs">
        {/* ============================================================ */}
        {/* STEP 1: SINGLE-SENTENCE IDEA ANCHOR                           */}
        {/* ============================================================ */}
        {activeStep === 'anchor-idea' && (
          <div className="space-y-4">
            <div className="space-y-1">
              <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                What should your Prism do?
              </h3>
              <p className="text-xs leading-relaxed text-zinc-500 dark:text-zinc-400">
                Describe your idea in one sentence. Verdant Studio will map out your inputs, rules, and checks automatically—and walk you through confirming each step.
              </p>
            </div>

            <form onSubmit={(e) => void handleAnalyzeIdea(e)} className="space-y-2.5">
              <textarea
                rows={3}
                value={ideaText}
                onChange={(e) => setIdeaText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    void handleAnalyzeIdea();
                  }
                }}
                placeholder="e.g. Check if a new expense fits my monthly budget, warn when it goes over, and refuse zero amounts."
                className="w-full resize-none rounded-xl border border-zinc-300 bg-zinc-50/50 p-3 text-xs leading-relaxed text-zinc-900 placeholder:text-zinc-400 focus:border-emerald-600 focus:bg-white focus:outline-none dark:border-zinc-700 dark:bg-zinc-900/50 dark:text-zinc-100 dark:focus:bg-zinc-950"
              />
              <div className="flex flex-wrap items-center justify-between gap-2">
                <button
                  type="submit"
                  disabled={!ideaText.trim() || analyzing}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-700 px-3.5 py-2 text-xs font-medium text-white hover:bg-emerald-800 disabled:opacity-45"
                >
                  {analyzing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
                  {analyzing ? 'Understanding your idea…' : 'Understand My Idea'}
                </button>
                {analyzing && (
                  <button
                    type="button"
                    onClick={handleUseFastParserNow}
                    className="text-[11px] font-medium text-zinc-500 hover:underline"
                  >
                    Use instant parser
                  </button>
                )}
              </div>
            </form>

            {/* Optional Inspiration Chips (Fill textarea only — never mutate Prism!) */}
            {!draftIntent && (
              <div className="space-y-1.5 pt-1">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
                  Or try an example sentence
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {INSPIRATION_PROMPTS.map((item) => (
                    <button
                      key={item.label}
                      type="button"
                      onClick={() => setIdeaText(item.sentence)}
                      className="rounded-lg border border-zinc-200 bg-zinc-50 px-2.5 py-1 text-[11px] text-zinc-700 hover:border-emerald-400 hover:bg-emerald-50/40 dark:border-zinc-800 dark:bg-zinc-900/60 dark:text-zinc-300"
                    >
                      {item.label}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* R2-Honest Understanding Preview Card */}
            {draftIntent && (
              <div className="space-y-3 rounded-xl border border-emerald-300/80 bg-emerald-50/40 p-3.5 dark:border-emerald-900/70 dark:bg-emerald-950/20">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <p className="text-xs font-semibold text-emerald-950 dark:text-emerald-200">
                      Here’s what I understood
                    </p>
                    <p className="text-[11px] text-zinc-600 dark:text-zinc-400">
                      Review below—you can refine every detail in the next steps or undo anytime.
                    </p>
                  </div>
                </div>

                <div className="space-y-2 rounded-lg bg-white/90 p-3 dark:bg-[#0B0F14]">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-semibold text-zinc-900 dark:text-zinc-100">{draftIntent.name.value}</span>
                    <span className="rounded bg-emerald-100 px-1.5 py-0.5 text-[9px] font-semibold text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300">
                      {DOMAIN_LABELS[draftIntent.domain.value]}
                    </span>
                  </div>

                  {/* Inputs */}
                  <div>
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
                      Inputs ({draftIntent.inputs.length})
                    </p>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {draftIntent.inputs.map((inp) => (
                        <span
                          key={inp.name}
                          className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[11px] ${
                            inp.source === 'stated'
                              ? 'border-emerald-300 bg-emerald-50/70 text-emerald-900 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-200'
                              : 'border-zinc-200 bg-zinc-50 text-zinc-600 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300'
                          }`}
                        >
                          <span>{inp.label}{inp.unit ? ` (${inp.unit})` : ''}</span>
                          <span className="text-[9px] opacity-65">
                            {inp.source === 'stated' ? 'from idea' : 'suggested'}
                          </span>
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Rules & Refusals */}
                  <div className="space-y-1">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
                      Rules &amp; Safety Checks
                    </p>
                    {draftIntent.rules.map((r, i) => (
                      <div key={i} className="flex items-center justify-between gap-2 text-[11px]">
                        <span className="text-zinc-800 dark:text-zinc-200">• {r.statement}</span>
                        <span className="shrink-0 text-[9px] text-zinc-400">
                          {r.source === 'stated' ? 'from idea' : 'suggested'}
                        </span>
                      </div>
                    ))}
                    {draftIntent.refusals.map((ref, i) => (
                      <div key={i} className="flex items-center justify-between gap-2 text-[11px]">
                        <span className="text-red-700 dark:text-red-300">• Refuse: “{ref.failureText}”</span>
                        <span className="shrink-0 text-[9px] text-zinc-400">
                          {ref.source === 'stated' ? 'from idea' : 'suggested'}
                        </span>
                      </div>
                    ))}
                  </div>

                  {/* Context Tag Summary */}
                  <div className="flex flex-wrap gap-1 border-t border-zinc-200 pt-2 text-[10px] text-zinc-500 dark:border-zinc-800">
                    <span className="rounded bg-zinc-100 px-1.5 py-0.5 dark:bg-zinc-800">
                      Audience: {draftIntent.audience.value} ({draftIntent.audience.source})
                    </span>
                    <span className="rounded bg-zinc-100 px-1.5 py-0.5 dark:bg-zinc-800">
                      Data: {draftIntent.dataClassification.value} ({draftIntent.dataClassification.source})
                    </span>
                    <span className="rounded bg-zinc-100 px-1.5 py-0.5 dark:bg-zinc-800">
                      Regions: {draftIntent.jurisdictions.value.join(', ').toUpperCase()}
                    </span>
                  </div>
                </div>

                <div className="flex items-center justify-between gap-2">
                  <button
                    type="button"
                    onClick={() => setDraftIntent(null)}
                    className="rounded-lg border border-zinc-300 px-3 py-1.5 text-xs font-medium text-zinc-700 hover:bg-white dark:border-zinc-700 dark:text-zinc-300"
                  >
                    Edit Sentence
                  </button>
                  <button
                    type="button"
                    onClick={() => void handleApplyIntent()}
                    className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-700 px-3.5 py-1.5 text-xs font-medium text-white hover:bg-emerald-800"
                  >
                    Apply &amp; Continue
                    <ArrowRight className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ============================================================ */}
        {/* STEP 2: CONFIRM AUDIENCE, DATA & REGIONS                      */}
        {/* ============================================================ */}
        {activeStep === 'confirm-context' && (
          <div className="space-y-4">
            <div className="space-y-1">
              <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                Who will use this, and what data does it touch?
              </h3>
              <p className="text-xs text-zinc-500">
                These choices automatically turn on only the privacy, safety, or regional protections your Prism actually needs.
              </p>
            </div>

            <div className="space-y-3">
              <div>
                <p className="mb-1.5 text-[11px] font-semibold text-zinc-700 dark:text-zinc-300">Who is it for?</p>
                <div className="grid grid-cols-2 gap-1.5">
                  {AUDIENCE_CHOICES.map((opt) => (
                    <button
                      key={opt.id}
                      type="button"
                      onClick={() => void onDispatchOp({ op: 'set-concept-profile', audience: opt.id })}
                      className={`rounded-lg border p-2 text-left transition-colors ${
                        declaration.concept.audience === opt.id
                          ? 'border-emerald-600 bg-emerald-50/70 dark:border-emerald-500 dark:bg-emerald-950/30'
                          : 'border-zinc-200 hover:bg-zinc-50 dark:border-zinc-800 dark:hover:bg-zinc-900'
                      }`}
                    >
                      <p className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">{opt.label}</p>
                      <p className="text-[10px] text-zinc-500">{opt.note}</p>
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <p className="mb-1.5 text-[11px] font-semibold text-zinc-700 dark:text-zinc-300">Kind of data entered</p>
                <div className="flex flex-wrap gap-1.5">
                  {DATA_CHOICES.map((opt) => (
                    <button
                      key={opt.id}
                      type="button"
                      onClick={() => void onDispatchOp({ op: 'set-concept-profile', dataClassification: opt.id })}
                      className={`rounded-lg border px-2.5 py-1 text-[11px] font-medium ${
                        declaration.concept.dataClassification === opt.id
                          ? 'border-emerald-600 bg-emerald-600 text-white'
                          : 'border-zinc-200 text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300'
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <p className="mb-1.5 text-[11px] font-semibold text-zinc-700 dark:text-zinc-300">Regions served</p>
                <div className="flex flex-wrap gap-1.5">
                  {REGION_CHOICES.map((opt) => {
                    const selected = declaration.concept.jurisdictions.includes(opt.id);
                    return (
                      <button
                        key={opt.id}
                        type="button"
                        onClick={() => void handleToggleRegion(opt.id)}
                        className={`rounded-lg border px-2.5 py-1 text-[11px] font-medium ${
                          selected
                            ? 'border-zinc-900 bg-zinc-900 text-white dark:border-zinc-100 dark:bg-zinc-100 dark:text-zinc-900'
                            : 'border-zinc-200 text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300'
                        }`}
                      >
                        {opt.label}
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 pt-1">
                <label className="flex cursor-pointer items-center justify-between rounded-lg border border-zinc-200 px-2.5 py-2 dark:border-zinc-800">
                  <span className="text-[11px] font-medium">On-device AI assistance</span>
                  <input
                    type="checkbox"
                    checked={declaration.concept.usesLocalAi}
                    onChange={(e) =>
                      void onDispatchOp({ op: 'set-concept-profile', usesLocalAi: e.target.checked })
                    }
                    className="h-3.5 w-3.5 accent-emerald-600"
                  />
                </label>
                <label className="flex cursor-pointer items-center justify-between rounded-lg border border-zinc-200 px-2.5 py-2 dark:border-zinc-800">
                  <span className="text-[11px] font-medium">Payments / checkout</span>
                  <input
                    type="checkbox"
                    checked={declaration.concept.involvesPayments}
                    onChange={(e) =>
                      void onDispatchOp({ op: 'set-concept-profile', involvesPayments: e.target.checked })
                    }
                    className="h-3.5 w-3.5 accent-emerald-600"
                  />
                </label>
              </div>
            </div>

            {/* Live Standards Routing Feedback */}
            <div className="rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2 text-[11px] text-zinc-600 dark:border-zinc-800 dark:bg-zinc-900/50 dark:text-zinc-300">
              {routing.activeLayeredStandards.length === 0 ? (
                <span>Clean personal setup — only the core checks apply, with zero extra compliance paperwork.</span>
              ) : (
                <span>
                  Based on your answers, we will guide you through{' '}
                  <strong>{routing.activeLayeredStandards.map((s) => s.authority.name).join(', ')}</strong>.
                </span>
              )}
            </div>

            <div className="flex justify-end">
              <button
                type="button"
                onClick={() => onSelectStep('refine-rules')}
                className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-700 px-3.5 py-1.5 text-xs font-medium text-white hover:bg-emerald-800"
              >
                Confirm Context &amp; Review Rules
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* STEP 3: PLAIN-ENGLISH RULES & REFINEMENT                      */}
        {/* ============================================================ */}
        {activeStep === 'refine-rules' && (
          <div className="space-y-4">
            <div className="space-y-1">
              <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                Confirm your inputs, rules, and refusals
              </h3>
              <p className="text-xs text-zinc-500">
                Check what your Prism accepts and decides. Type any addition below in plain English.
              </p>
            </div>

            {/* Inputs */}
            <div className="space-y-1.5 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
                Inputs ({declaration.encoding?.fields.length ?? 0})
              </p>
              <div className="flex flex-wrap gap-1.5">
                {(declaration.encoding?.fields ?? []).map((f) => (
                  <span
                    key={f.name}
                    className="rounded-md bg-zinc-100 px-2 py-1 text-[11px] font-medium text-zinc-800 dark:bg-zinc-800 dark:text-zinc-200"
                  >
                    {f.label}
                    {f.choices ? ` (${f.choices.join(' / ')})` : f.unit ? ` (${f.unit})` : ''}
                  </span>
                ))}
              </div>
            </div>

            {/* Plain-English Rules & Refusals */}
            <div className="space-y-1.5 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <div className="flex items-center justify-between">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
                  Rules ({guided.plainRules.length}) &amp; Refusals ({guided.plainRefusals.length})
                </p>
                <span className="text-[10px] font-medium text-emerald-700 dark:text-emerald-400">
                  {checkReport.vectorsPassed}/{checkReport.vectorsTotal} examples passing
                </span>
              </div>
              {guided.plainRules.map((r) => (
                <div key={r.id} className="flex items-center justify-between gap-2 rounded bg-zinc-50 px-2.5 py-1.5 dark:bg-zinc-900/60">
                  <span className="text-[11px] text-zinc-800 dark:text-zinc-200">{r.summary}</span>
                  {guided.plainRules.length > 1 && (
                    <button
                      type="button"
                      onClick={() => void onDispatchOp({ op: 'remove-feature', id: r.id })}
                      className="shrink-0 text-zinc-400 hover:text-red-500"
                      title="Remove rule"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  )}
                </div>
              ))}
              {guided.plainRefusals.map((ref) => (
                <div key={ref.id} className="flex items-center justify-between gap-2 rounded bg-red-50/70 px-2.5 py-1.5 dark:bg-red-950/20">
                  <span className="text-[11px] text-red-800 dark:text-red-200">{ref.summary}</span>
                  <button
                    type="button"
                    onClick={() => void onDispatchOp({ op: 'remove-refusal', id: ref.id })}
                    className="shrink-0 text-zinc-400 hover:text-red-500"
                    title="Remove refusal"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </div>
              ))}
            </div>

            {/* Natural-Language Refinement Input */}
            <form onSubmit={handleParseFollowUp} className="space-y-2 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <p className="text-[11px] font-medium text-zinc-700 dark:text-zinc-300">
                Add a rule, refusal, or input in plain English
              </p>
              <div className="flex gap-1.5">
                <input
                  type="text"
                  value={followUpText}
                  onChange={(e) => setFollowUpText(e.target.value)}
                  placeholder='e.g. "Refuse if amount is over 5000" or "Add a field for priority (Low, Medium, High)"'
                  className="min-w-0 flex-1 rounded-md border border-zinc-300 bg-transparent px-2.5 py-1.5 text-xs dark:border-zinc-700"
                />
                <button
                  type="submit"
                  disabled={!followUpText.trim()}
                  className="shrink-0 rounded-md bg-zinc-900 px-3 py-1.5 text-xs font-medium text-white disabled:opacity-40 dark:bg-zinc-100 dark:text-zinc-900"
                >
                  Check
                </button>
              </div>
              {followUpResult && (
                <div
                  className={`rounded-md p-2 text-[11px] ${
                    followUpResult.ok
                      ? 'border border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-800 dark:bg-emerald-950/30 dark:text-emerald-200'
                      : 'border border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-950/30 dark:text-amber-200'
                  }`}
                >
                  {followUpResult.ok ? (
                    <div className="flex items-center justify-between gap-2">
                      <span>{followUpResult.summary}</span>
                      <button
                        type="button"
                        onClick={() => void handleApplyFollowUp()}
                        className="shrink-0 rounded bg-emerald-700 px-2.5 py-0.5 text-[10px] font-semibold text-white"
                      >
                        Apply Change
                      </button>
                    </div>
                  ) : (
                    <span>{followUpResult.hint}</span>
                  )}
                </div>
              )}
            </form>

            <div className="flex justify-end">
              <button
                type="button"
                onClick={async () => {
                  await onDispatchOp({ op: 'sync-stride-threats' });
                  onSelectStep('safety-check');
                }}
                className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-700 px-3.5 py-1.5 text-xs font-medium text-white hover:bg-emerald-800"
              >
                Confirm Rules &amp; Check Safety
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* STEP 4: PLAIN-ENGLISH SAFETY & PRIVACY CHECK                  */}
        {/* ============================================================ */}
        {activeStep === 'safety-check' && (
          <div className="space-y-4">
            <div className="space-y-1">
              <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                Safety &amp; privacy check
              </h3>
              <p className="text-xs text-zinc-500">
                Verdant isolates every Prism in an offline sandbox automatically. Confirm any remaining items below.
              </p>
            </div>

            {declaration.security.threats.length === 0 ? (
              <button
                type="button"
                onClick={() => void onDispatchOp({ op: 'sync-stride-threats' })}
                className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-700 px-3.5 py-2 text-xs font-medium text-white"
              >
                <Shield className="h-3.5 w-3.5" />
                Run Safety Check
              </button>
            ) : (
              <>
                <div className="flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50/70 px-3 py-2 text-xs text-emerald-900 dark:border-emerald-900/60 dark:bg-emerald-950/30 dark:text-emerald-200">
                  <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600" />
                  <span>
                    {guided.autoProtectedThreats} of {declaration.security.threats.length} safety checks are verified and protected.
                  </span>
                </div>

                {guided.openSafetyItems.length > 0 ? (
                  <div className="space-y-2.5">
                    {guided.openSafetyItems.map((item) => (
                      <div
                        key={item.id}
                        className="space-y-2 rounded-lg border border-amber-300/80 bg-amber-50/50 p-3 dark:border-amber-900/60 dark:bg-amber-950/20"
                      >
                        <p className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">{item.title}</p>
                        <p className="text-[11px] text-zinc-600 dark:text-zinc-300">{item.plainPrompt}</p>
                        <div className="flex flex-wrap gap-1.5 pt-0.5">
                          {item.quickActions.map((act) => (
                            <button
                              key={act.label}
                              type="button"
                              onClick={() => void onDispatchOp(act.op)}
                              className="rounded-md bg-emerald-700 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-emerald-800"
                            >
                              {act.label}
                            </button>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-emerald-700 dark:text-emerald-400">
                    All safety boundaries are verified—no open concerns remain.
                  </p>
                )}

                <div className="flex justify-end">
                  <button
                    type="button"
                    onClick={() => onSelectStep('design-feel')}
                    className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-700 px-3.5 py-1.5 text-xs font-medium text-white hover:bg-emerald-800"
                  >
                    Continue to Look &amp; Feel
                    <ArrowRight className="h-3.5 w-3.5" />
                  </button>
                </div>
              </>
            )}
          </div>
        )}

        {/* ============================================================ */}
        {/* STEP 5: LOOK & FEEL (UNLOCKS LIVE PREVIEW)                    */}
        {/* ============================================================ */}
        {activeStep === 'design-feel' && (
          <div className="space-y-4">
            <div className="space-y-1">
              <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                Choose how your Prism looks and feels
              </h3>
              <p className="text-xs text-zinc-500">
                Pick a layout and color palette—Verdant checks readability and builds your interactive preview automatically.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-1.5">
              {(
                [
                  ['decision-form', 'Decision Form', 'Clean form with clear verdict card'],
                  ['calculator', 'Live Calculator', 'Recalculates live as you adjust inputs'],
                  ['ledger-tracker', 'Tracker + History', 'Summary totals with recent log'],
                  ['request-reply', 'Q&A Transcript', 'Step-by-step question & answer feed'],
                ] as readonly [PrismViewpoint, string, string][]
              ).map(([vp, label, desc]) => (
                <button
                  key={vp}
                  type="button"
                  onClick={() => setSelectedViewpoint(vp)}
                  className={`rounded-lg border p-2.5 text-left transition-colors ${
                    selectedViewpoint === vp
                      ? 'border-emerald-600 bg-emerald-50/70 dark:border-emerald-500 dark:bg-emerald-950/30'
                      : 'border-zinc-200 hover:bg-zinc-50 dark:border-zinc-800'
                  }`}
                >
                  <p className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">{label}</p>
                  <p className="mt-0.5 text-[10px] text-zinc-500">{desc}</p>
                </button>
              ))}
            </div>

            <div className="space-y-1.5">
              <p className="text-[11px] font-semibold text-zinc-700 dark:text-zinc-300">Theme Palette</p>
              <div className="grid grid-cols-2 gap-1.5">
                {PRISM_THEME_PRESETS.map((preset) => (
                  <button
                    key={preset.id}
                    type="button"
                    onClick={() => void onDispatchOp({ op: 'set-theme', theme: { preset: preset.id } })}
                    className={`flex items-center justify-between gap-2 rounded-lg border px-2.5 py-2 text-left ${
                      declaration.theme.preset === preset.id
                        ? 'border-emerald-600 bg-emerald-50/60 dark:border-emerald-500 dark:bg-emerald-950/30'
                        : 'border-zinc-200 hover:bg-zinc-50 dark:border-zinc-800'
                    }`}
                  >
                    <span className="text-xs font-medium">{preset.name}</span>
                    <div className="flex gap-0.5">
                      {preset.swatches.map((hex) => (
                        <span
                          key={hex}
                          className="h-3.5 w-3.5 rounded-full border border-black/15"
                          style={{ backgroundColor: hex }}
                        />
                      ))}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            <div className="flex items-center justify-between gap-2 pt-1">
              <span className="inline-flex items-center gap-1 text-[11px] text-emerald-700 dark:text-emerald-400">
                <Palette className="h-3.5 w-3.5" />
                Contrast &amp; accessibility checked automatically
              </span>
              <button
                type="button"
                disabled={(declaration.encoding?.fields.length ?? 0) === 0}
                onClick={() => void handleSynthesizeDesign()}
                className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-700 px-3.5 py-2 text-xs font-medium text-white hover:bg-emerald-800 disabled:opacity-45"
              >
                <Eye className="h-3.5 w-3.5" />
                {declaration.view ? 'Update & Open Preview' : 'Build Interface & Open Preview'}
              </button>
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* STEP 6: CONDITIONAL COMPLIANCE CHECKS                         */}
        {/* ============================================================ */}
        {activeStep === 'routed-standards' && (
          <div className="space-y-4">
            <div className="space-y-1">
              <h3 className="flex items-center gap-1.5 text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                <Scale className="h-4 w-4 text-emerald-600" />
                Required protections for your setup
              </h3>
              <p className="text-xs text-zinc-500">
                Because of the audience, data, or regions you selected, these standards apply. Settle each with one click.
              </p>
            </div>

            <div className="space-y-2.5">
              {routing.activeLayeredStandards.map((std) => (
                <div key={std.authority.id} className="space-y-2 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
                  <div>
                    <p className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">{std.authority.name}</p>
                    <p className="text-[11px] text-zinc-500">{std.reason}</p>
                  </div>
                  {std.obligations.map((ob) => (
                    <div
                      key={ob.id}
                      className={`flex items-start justify-between gap-2 rounded-md border p-2 ${
                        ob.settled
                          ? 'border-emerald-200 bg-emerald-50/50 dark:border-emerald-900/50 dark:bg-emerald-950/20'
                          : 'border-amber-200 bg-amber-50/50 dark:border-amber-900/50 dark:bg-amber-950/20'
                      }`}
                    >
                      <div className="min-w-0">
                        <p className="text-[11px] font-semibold text-zinc-900 dark:text-zinc-100">{ob.title}</p>
                        <p className="text-[10px] text-zinc-500">{ob.evidence ?? ob.detail}</p>
                      </div>
                      {ob.settled ? (
                        <span className="inline-flex shrink-0 items-center gap-1 rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-semibold text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300">
                          <Check className="h-3 w-3" />
                          Done
                        </span>
                      ) : ob.suggestedOp ? (
                        <button
                          type="button"
                          onClick={() => void onDispatchOp(ob.suggestedOp!)}
                          className="shrink-0 rounded bg-emerald-700 px-2.5 py-1 text-[10px] font-semibold text-white hover:bg-emerald-800"
                        >
                          Apply Protection
                        </button>
                      ) : null}
                    </div>
                  ))}
                </div>
              ))}
            </div>

            <div className="flex justify-end">
              <button
                type="button"
                onClick={() => onSelectStep('ready-to-seal')}
                className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-700 px-3.5 py-1.5 text-xs font-medium text-white hover:bg-emerald-800"
              >
                Continue to Prove &amp; Publish
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        )}

        {/* ============================================================ */}
        {/* STEP 7: PROVE & PUBLISH                                       */}
        {/* ============================================================ */}
        {activeStep === 'ready-to-seal' && (
          <div className="space-y-4">
            <div className="space-y-1">
              <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-100">
                {routing.readyToSeal ? 'Your Prism is ready to prove & publish' : 'Almost ready'}
              </h3>
              <p className="text-xs text-zinc-500">
                {routing.readyToSeal
                  ? 'Every core check and required protection is complete. Build to mathematically prove your rules and package a portable .holo archive.'
                  : 'Complete any remaining item below to unlock proof & publishing.'}
              </p>
            </div>

            <div className="space-y-1.5">
              {guided.steps
                .filter((s) => s.id !== 'ready-to-seal' && s.visible)
                .map((st) => (
                  <button
                    key={st.id}
                    type="button"
                    onClick={() => onSelectStep(st.id)}
                    className="flex w-full items-center justify-between rounded-lg border border-zinc-200 px-3 py-2 text-left hover:bg-zinc-50 dark:border-zinc-800 dark:hover:bg-zinc-900"
                  >
                    <span className="flex items-center gap-2 font-medium text-zinc-900 dark:text-zinc-100">
                      {st.complete ? (
                        <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                      ) : (
                        <AlertTriangle className="h-4 w-4 text-amber-500" />
                      )}
                      {st.label}
                    </span>
                    <span className="text-[11px] text-zinc-500">{st.complete ? 'Complete' : 'Needs review'}</span>
                  </button>
                ))}
            </div>

            <div className="flex flex-wrap gap-2 pt-1">
              <button
                type="button"
                disabled={!routing.readyToSeal || building}
                onClick={onCheckAndBuild}
                className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-700 px-4 py-2 text-xs font-medium text-white hover:bg-emerald-800 disabled:opacity-45"
              >
                {building ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ShieldCheck className="h-3.5 w-3.5" />}
                Prove &amp; Seal (.holo)
              </button>
              <button
                type="button"
                disabled={!routing.readyToSeal || building}
                onClick={onPublish}
                className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-600 px-3.5 py-2 text-xs font-medium text-emerald-800 hover:bg-emerald-50 disabled:opacity-45 dark:border-emerald-700 dark:text-emerald-200 dark:hover:bg-emerald-950/40"
              >
                <Globe className="h-3.5 w-3.5" />
                Publish to App
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Subtle Footer Link to Advanced Model Inspector */}
      <div className="flex shrink-0 items-center justify-between border-t border-zinc-200 bg-zinc-50/70 px-3 py-2 text-[11px] text-zinc-500 dark:border-zinc-800 dark:bg-[#0B0F14]">
        <span>Guided Creator</span>
        <button
          type="button"
          onClick={onSwitchToAdvanced}
          className="inline-flex items-center gap-1 font-medium text-zinc-700 hover:text-emerald-700 dark:text-zinc-300 dark:hover:text-emerald-400"
        >
          <SlidersHorizontal className="h-3 w-3" />
          Open Advanced Model Editor
        </button>
      </div>
    </div>
  );
};
