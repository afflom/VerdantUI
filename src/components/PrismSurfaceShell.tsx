/**
 * Universal Procedurally Synthesized Prism Surface (`src/components/PrismSurfaceShell.tsx`)
 *
 * Implements the universal VerdantUI interaction contract across EVERY part of VerdantUI
 * (Home, Galleries, Spaces, Settings, Documents, Pictures, Videos, Audio, Games, Research, Utilities):
 *   1. Clean single-row header with `Preview | Details` toggle (zero stacked toolbars).
 *   2. Full-stage, serene Rendering (Preview) Panel matching the homescreen aesthetic.
 *   3. Calm editorial Prism Details view showing purpose, identity, part hierarchy, and controls.
 *   4. Homescreen-grade Natural Language Interface bar (optional when the child surface supplies
 *      its own full conversational composer such as Home ThreadView).
 */

import {
  useEffect,
  useRef,
  useState,
  type FC,
  type FormEvent,
  type ReactNode,
} from 'react';
import {
  ArrowUp,
  Eye,
  Info,
  Layers,
  Loader2,
  MessageSquare,
  ShieldCheck,
  Sparkles,
  Square,
  X,
} from 'lucide-react';
import type {
  PrismSurfaceMode,
  SynthesizedPrismFrame,
  VerdantSystemPrismDescriptor,
} from '../engine/observerUiSynthesizer';
import { shortPrismDigest } from '../engine/prismStudio';

export interface PrismSurfaceMessage {
  readonly id: string;
  readonly role: 'user' | 'assistant';
  readonly content: string;
}

interface PrismSurfaceShellProps {
  /** Procedurally synthesized layout & compliance frame for this Prism. */
  frame: SynthesizedPrismFrame;
  /** Current Preview vs Details surface mode. */
  mode: PrismSurfaceMode;
  onChangeMode: (mode: PrismSurfaceMode) => void;
  /** Optional inline title element (e.g., click-to-rename title on user Prisms). */
  titleSlot?: ReactNode;
  /** Optional subtle status dot next to title. */
  statusDotClass?: string;
  /** Optional concise header actions (e.g., Undo, Publish). */
  headerActions?: ReactNode;
  /** Optional subtle status banner directly under the header. */
  banner?: ReactNode;
  /** Main live rendered Preview surface. */
  previewContent: ReactNode;
  /** Optional extra controls inside the Details view (Permissions, Export/Import, Storage, etc.). */
  detailsExtra?: ReactNode;
  /** Child Prism navigation callback from the Details part-tree list. */
  onSelectChildPrism?: (prismId: string) => void;
  /** Natural language command handler. Omit only when previewContent already includes the full composer (Home). */
  onSubmitNaturalLanguage?: (prompt: string) => void | Promise<void>;
  /** Cancel an in-progress natural-language action. */
  onCancelNaturalLanguage?: () => void;
  /** Busy indicator for the natural-language bar. */
  nlBusy?: boolean;
  nlBusyLabel?: string;
  /** Conversation history turns for this Prism's natural-language interface. */
  nlMessages?: readonly PrismSurfaceMessage[];
  /** Optional placeholder override for the natural-language bar. */
  nlPlaceholder?: string;
  /** Optional quick suggestions shown when idle. */
  nlSuggestions?: readonly string[];
  /** Notify parent when the user starts/stops typing (for calm peripheral chrome). */
  onTypingChange?: (typing: boolean) => void;
}

const CAPABILITY_LABELS: Readonly<Record<string, string>> = {
  'natural-language-interface': 'Natural language control',
  'live-preview': 'Live interactive preview',
  'prism-details': 'Inspectable Prism details',
  'local-index': 'Local on-device index',
  'multimodal-io': 'Text, image, and voice input',
  'local-storage': 'Private on-device storage',
  'sandboxed-runtime': 'Isolated private sandbox',
  'self-similar-prism-root': 'Unified workspace Prism',
  'procedural-ui-synthesis': 'Adaptive layout synthesis',
  'local-only-privacy': '100% private · zero servers',
  navigation: 'Workspace navigation',
};

function renderPartRow(
  part: VerdantSystemPrismDescriptor,
  activeId: string,
  onSelect?: (id: string) => void,
) {
  const isCurrent = part.id === activeId;
  return (
    <button
      key={part.id}
      type="button"
      onClick={() => onSelect?.(part.id)}
      disabled={!onSelect}
      className={`flex w-full items-center justify-between gap-3 rounded-xl border px-3.5 py-2.5 text-left transition-colors ${
        isCurrent
          ? 'border-emerald-600/40 bg-emerald-50/70 text-emerald-950 dark:border-emerald-500/40 dark:bg-emerald-950/40 dark:text-emerald-100'
          : 'border-slate-200 bg-white text-slate-800 hover:border-slate-300 hover:bg-slate-50 dark:border-slate-800 dark:bg-[#121821] dark:text-slate-200 dark:hover:border-slate-700'
      }`}
    >
      <div className="min-w-0">
        <div className="flex items-center gap-2 text-xs font-semibold">
          <Layers className="h-3.5 w-3.5 shrink-0 text-emerald-700 dark:text-emerald-400" />
          <span className="truncate">{part.name}</span>
          {isCurrent && (
            <span className="rounded-full bg-emerald-700 px-1.5 py-0.5 text-[10px] font-medium text-white">
              Active
            </span>
          )}
        </div>
        <p className="mt-0.5 truncate text-[11px] text-slate-500 dark:text-slate-400">
          {part.subtitle}
        </p>
      </div>
      <span className="shrink-0 font-mono text-[10px] text-slate-400">
        #{shortPrismDigest(part.digest)}
      </span>
    </button>
  );
}

export const PrismSurfaceShell: FC<PrismSurfaceShellProps> = ({
  frame,
  mode,
  onChangeMode,
  titleSlot,
  statusDotClass,
  headerActions,
  banner,
  previewContent,
  detailsExtra,
  onSelectChildPrism,
  onSubmitNaturalLanguage,
  onCancelNaturalLanguage,
  nlBusy = false,
  nlBusyLabel = 'Updating…',
  nlMessages = [],
  nlPlaceholder,
  nlSuggestions,
  onTypingChange,
}) => {
  const [input, setInput] = useState('');
  const [historyOpen, setHistoryOpen] = useState(false);
  const historyEndRef = useRef<HTMLDivElement | null>(null);
  const prism = frame.activePartPrism;
  const calmChrome = frame.directive.calmPeripheralUi;
  const effectiveSuggestions = nlSuggestions ?? prism.nlSuggestions;

  useEffect(() => {
    if (historyOpen) {
      historyEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [historyOpen, nlMessages.length]);

  useEffect(() => {
    if (mode !== 'details') return;
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === 'Escape') onChangeMode('preview');
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [mode, onChangeMode]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || nlBusy || !onSubmitNaturalLanguage) return;
    setInput('');
    onTypingChange?.(false);
    void onSubmitNaturalLanguage(text);
  };

  const handleSuggestion = (text: string) => {
    if (nlBusy || !onSubmitNaturalLanguage) return;
    void onSubmitNaturalLanguage(text);
  };

  const latestReply =
    nlMessages.length > 0 && nlMessages[nlMessages.length - 1]!.role === 'assistant'
      ? nlMessages[nlMessages.length - 1]!
      : null;

  return (
    <section
      aria-label={prism.name}
      data-prism-id={prism.id}
      data-prism-kind={prism.kind}
      data-prism-digest={shortPrismDigest(prism.digest)}
      data-ui-compliant={frame.layout.certificate.ok ? 'true' : 'false'}
      style={frame.directive.tokens as React.CSSProperties}
      className="relative flex h-full min-h-0 flex-col overflow-hidden bg-[#F8F9FA] dark:bg-[#0B0F14]"
    >
      {/* Single calm header bar: Title + Preview/Details toggle + minimal actions */}
      <header
        className={`flex h-11 shrink-0 items-center justify-between gap-3 border-b border-slate-200/80 bg-white/95 px-4 backdrop-blur-xs transition-opacity dark:border-slate-800/80 dark:bg-[#0F141C]/95 ${
          calmChrome ? 'opacity-90 hover:opacity-100' : 'opacity-100'
        }`}
      >
        <div className="flex min-w-0 items-center gap-2">
          {statusDotClass && <span className={`h-2 w-2 shrink-0 rounded-full ${statusDotClass}`} />}
          {titleSlot ?? (
            <h1 className="truncate text-sm font-semibold tracking-tight text-slate-900 dark:text-slate-100">
              {prism.name}
            </h1>
          )}
        </div>

        <div className="flex shrink-0 items-center gap-2">
          {headerActions}

          {/* Universal Preview | Details segmented toggle */}
          <div
            role="group"
            aria-label="Surface view"
            className="inline-flex rounded-lg border border-slate-200 bg-slate-100/90 p-0.5 text-xs dark:border-slate-700/80 dark:bg-slate-900"
          >
            <button
              type="button"
              data-testid="prism-toggle-preview"
              aria-pressed={mode === 'preview'}
              onClick={() => onChangeMode('preview')}
              className={`inline-flex items-center gap-1 rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors ${
                mode === 'preview'
                  ? 'bg-white text-slate-950 shadow-xs dark:bg-slate-800 dark:text-white'
                  : 'text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
              }`}
            >
              <Eye className="h-3.5 w-3.5" />
              <span>Preview</span>
            </button>
            <button
              type="button"
              data-testid="prism-toggle-details"
              aria-pressed={mode === 'details'}
              onClick={() => onChangeMode(mode === 'details' ? 'preview' : 'details')}
              className={`inline-flex items-center gap-1 rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors ${
                mode === 'details'
                  ? 'bg-white text-slate-950 shadow-xs dark:bg-slate-800 dark:text-white'
                  : 'text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
              }`}
            >
              <Info className="h-3.5 w-3.5" />
              <span>Details</span>
            </button>
          </div>
        </div>
      </header>

      {banner}

      {/* Main Rendering (Preview) Stage + Details Overlay */}
      <div className="relative min-h-0 flex-1 overflow-hidden">
        <div
          className={`h-full w-full ${
            mode === 'details' ? 'invisible pointer-events-none' : ''
          }`}
        >
          {previewContent}
        </div>

        {mode === 'details' && (
          <div className="absolute inset-0 z-20 overflow-y-auto bg-[#F8F9FA] dark:bg-[#0B0F14]">
            <div className="mx-auto max-w-2xl space-y-5 px-5 py-7">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-700/20 bg-emerald-50 px-2.5 py-0.5 text-[11px] font-medium text-emerald-900 dark:border-emerald-500/30 dark:bg-emerald-950/50 dark:text-emerald-200">
                    <Sparkles className="h-3 w-3" />
                    <span>Prism Details</span>
                  </span>
                  <h2 className="mt-2 text-xl font-bold tracking-tight text-slate-950 dark:text-white">
                    {prism.name}
                  </h2>
                  <p className="mt-1 text-sm leading-relaxed text-slate-600 dark:text-slate-400">
                    {prism.purpose}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => onChangeMode('preview')}
                  className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-200 dark:hover:bg-slate-800"
                >
                  <Eye className="h-3.5 w-3.5" />
                  <span>Back to Preview</span>
                </button>
              </div>

              {/* Identity & Privacy summary */}
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                <div className="rounded-xl border border-slate-200 bg-white p-3.5 dark:border-slate-800 dark:bg-[#0F141C]">
                  <div className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
                    Identity
                  </div>
                  <div className="mt-1 font-mono text-xs font-semibold text-slate-900 dark:text-slate-100">
                    #{shortPrismDigest(prism.digest)}
                  </div>
                </div>
                <div className="rounded-xl border border-slate-200 bg-white p-3.5 dark:border-slate-800 dark:bg-[#0F141C]">
                  <div className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
                    Workspace
                  </div>
                  <div className="mt-1 text-xs font-semibold text-slate-900 dark:text-slate-100">
                    {frame.rootPrism.name}
                  </div>
                </div>
                <div className="rounded-xl border border-slate-200 bg-white p-3.5 dark:border-slate-800 dark:bg-[#0F141C]">
                  <div className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
                    Layout Check
                  </div>
                  <div className="mt-1 inline-flex items-center gap-1 text-xs font-semibold text-emerald-700 dark:text-emerald-400">
                    <ShieldCheck className="h-3.5 w-3.5" />
                    <span>{frame.layout.certificate.ok ? 'Verified' : 'Checking'}</span>
                  </div>
                </div>
              </div>

              {/* Capabilities */}
              {prism.capabilities.length > 0 && (
                <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-[#0F141C]">
                  <div className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                    Capabilities
                  </div>
                  <div className="mt-2.5 flex flex-wrap gap-1.5">
                    {prism.capabilities.map((cap) => (
                      <span
                        key={cap}
                        className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-medium text-slate-700 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-300"
                      >
                        {CAPABILITY_LABELS[cap] ?? cap}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Part-specific extra Details controls (Export, Permissions, Storage, etc.) */}
              {detailsExtra}

              {/* Self-similar VerdantUI Prism Parts */}
              <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-[#0F141C]">
                <div className="flex items-center justify-between gap-2">
                  <div>
                    <div className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                      VerdantUI Prisms
                    </div>
                    <p className="mt-0.5 text-[11px] text-slate-500 dark:text-slate-400">
                      Every part of VerdantUI is its own Prism with a live preview, details view, and natural language interface.
                    </p>
                  </div>
                  <span className="font-mono text-[11px] text-slate-400 tabular-nums">
                    {frame.partPrisms.length} parts
                  </span>
                </div>
                <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
                  {frame.partPrisms
                    .filter((p) => p.kind !== 'navigation-rail')
                    .map((part) => renderPartRow(part, prism.id, onSelectChildPrism))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Homescreen-grade Natural Language Interface Bar (present on every part) */}
      {onSubmitNaturalLanguage && (
        <div className="shrink-0 border-t border-slate-200/80 bg-[#F8F9FA]/95 px-4 pt-2.5 pb-3 backdrop-blur-xs dark:border-slate-800/80 dark:bg-[#0B0F14]/95">
          <div className="mx-auto max-w-3xl">
            {/* Collapsible Conversation History Drawer */}
            {historyOpen && nlMessages.length > 0 && (
              <div className="mb-2.5 max-h-52 overflow-y-auto rounded-xl border border-slate-200 bg-white p-3 shadow-xs dark:border-slate-800 dark:bg-[#121821]">
                <div className="mb-2 flex items-center justify-between text-[11px] font-semibold text-slate-500 dark:text-slate-400">
                  <span>Conversation</span>
                  <button
                    type="button"
                    onClick={() => setHistoryOpen(false)}
                    aria-label="Hide conversation"
                    className="rounded p-0.5 hover:bg-slate-100 dark:hover:bg-slate-800"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
                <div className="space-y-2 text-xs">
                  {nlMessages.map((m) => (
                    <div
                      key={m.id}
                      className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div
                        className={`max-w-[85%] rounded-xl px-3 py-1.5 leading-relaxed ${
                          m.role === 'user'
                            ? 'bg-[#0F5132] text-white'
                            : 'bg-slate-100 text-slate-800 dark:bg-slate-800/80 dark:text-slate-200'
                        }`}
                      >
                        {m.content}
                      </div>
                    </div>
                  ))}
                  <div ref={historyEndRef} />
                </div>
              </div>
            )}

            {/* Concise status reply toast line when history is collapsed */}
            {!historyOpen && latestReply && (
              <div className="mb-2 flex items-center justify-between gap-2 px-1 text-xs text-slate-600 dark:text-slate-400">
                <span className="truncate">{latestReply.content}</span>
                {nlMessages.length > 1 && (
                  <button
                    type="button"
                    onClick={() => setHistoryOpen(true)}
                    className="shrink-0 text-[11px] font-medium text-emerald-700 hover:underline dark:text-emerald-400"
                  >
                    History ({nlMessages.length})
                  </button>
                )}
              </div>
            )}

            {/* Quick suggestions when input is empty and not busy */}
            {!input && !nlBusy && effectiveSuggestions.length > 0 && !historyOpen && (
              <div className="mb-2 flex flex-wrap gap-1.5">
                {effectiveSuggestions.slice(0, 3).map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => handleSuggestion(s)}
                    className="rounded-full border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-medium text-slate-600 hover:border-emerald-700/40 hover:bg-emerald-50/50 hover:text-emerald-950 dark:border-slate-800 dark:bg-[#121821] dark:text-slate-300 dark:hover:border-emerald-500/40 dark:hover:bg-emerald-950/40 dark:hover:text-emerald-200 transition-colors"
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}

            {/* Homescreen-styled Natural Language Composer */}
            <form
              onSubmit={handleSubmit}
              className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-1.5 shadow-xs focus-within:border-emerald-800 dark:border-slate-800 dark:bg-[#121821] dark:focus-within:border-emerald-500 transition-colors"
            >
              {nlMessages.length > 0 && (
                <button
                  type="button"
                  onClick={() => setHistoryOpen((v) => !v)}
                  aria-label="Toggle conversation history"
                  title="Conversation history"
                  className={`inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg transition-colors ${
                    historyOpen
                      ? 'bg-emerald-50 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300'
                      : 'text-slate-400 hover:bg-slate-100 hover:text-slate-700 dark:hover:bg-slate-800 dark:hover:text-slate-200'
                  }`}
                >
                  <MessageSquare className="h-4 w-4" />
                </button>
              )}

              <input
                type="text"
                aria-label={`Ask ${prism.name}`}
                value={input}
                onChange={(e) => {
                  const val = e.target.value;
                  setInput(val);
                  onTypingChange?.(val.trim().length > 0);
                }}
                placeholder={nlPlaceholder ?? prism.nlPlaceholder}
                disabled={nlBusy}
                className="min-w-0 flex-1 bg-transparent py-1.5 text-sm text-slate-900 outline-none placeholder:text-slate-400 disabled:opacity-60 dark:text-slate-100 dark:placeholder:text-slate-500"
              />

              {nlBusy ? (
                <div className="flex shrink-0 items-center gap-2">
                  <span className="hidden items-center gap-1.5 text-xs text-slate-500 sm:inline-flex dark:text-slate-400">
                    <Loader2 className="h-3.5 w-3.5 animate-spin text-emerald-700 dark:text-emerald-400" />
                    <span>{nlBusyLabel}</span>
                  </span>
                  {onCancelNaturalLanguage && (
                    <button
                      type="button"
                      onClick={onCancelNaturalLanguage}
                      aria-label="Stop"
                      className="inline-flex h-8 items-center gap-1 rounded-lg bg-slate-900 px-2.5 text-xs font-medium text-white hover:bg-slate-800 dark:bg-slate-100 dark:text-slate-950"
                    >
                      <Square className="h-3 w-3 fill-current" />
                      <span>Stop</span>
                    </button>
                  )}
                </div>
              ) : (
                <button
                  type="submit"
                  disabled={!input.trim()}
                  aria-label="Send"
                  className="inline-flex h-8 shrink-0 items-center gap-1 rounded-lg bg-[#0F5132] px-3 text-xs font-medium text-white hover:bg-[#0B3D25] disabled:cursor-not-allowed disabled:opacity-40 transition-colors"
                >
                  <span>Send</span>
                  <ArrowUp className="h-3.5 w-3.5" />
                </button>
              )}
            </form>
          </div>
        </div>
      )}
    </section>
  );
};
