import { createContext, useContext, useEffect, useMemo, useState, type FC } from 'react';
import {
  ActionBarPrimitive,
  AttachmentPrimitive,
  AuiIf,
  BranchPickerPrimitive,
  ComposerPrimitive,
  ErrorPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useAui,
  useAuiState,
  type ReasoningMessagePartProps,
  type ThreadMessage,
  type ToolCallMessagePartComponent,
} from '@assistant-ui/react';
import { MarkdownTextPrimitive } from '@assistant-ui/react-markdown';
import remarkGfm from 'remark-gfm';
import {
  AlertCircle,
  ArrowDown,
  ArrowUp,
  Brain,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Copy,
  ExternalLink,
  FileText,
  Loader2,
  Paperclip,
  Pencil,
  RefreshCw,
  SlidersHorizontal,
  Sparkles,
  Square,
  Wrench,
  X,
} from 'lucide-react';
import { formatMessagesForLlama } from '../engine/assistantAdapter';
import {
  UNIFIED_VERDANT_SUGGESTIONS,
  UNIFIED_VERDANT_SYSTEM_PROMPT,
  type CognitiveRouteCertificate,
} from '../engine/cognitiveObserver';
import { isInternalBootstrapModel } from '../engine/modelCatalog';
import { shortPrismDigest, type PublishedPrismSummary } from '../engine/prismStudio';
import type { EngineState } from '../engine/workerClient';

interface ThreadViewProps {
  engineState: EngineState;
  onCancelDownload: () => void;
  onRetryLoad?: () => void;
  onDismissError?: () => void;
  onToggleInspector: () => void;
  onOpenPrism?: (prismId: string) => void;
  onEditPrism?: (prismId: string) => void;
  publishedPrisms?: readonly PublishedPrismSummary[];
  onSpeculativeDraftChange?: (
    messages: readonly ThreadMessage[],
    draftText: string,
    systemMessage?: string,
    threadId?: string,
  ) => void;
  onCountTokens?: (prompt: string) => Promise<{ tokenCount: number; maxContextWindow: number }>;
}

const SpeculativePrefillDriver: FC<{
  modelId: string | null;
  modelReady: boolean;
  promptMode: EngineState['promptMode'];
  systemMessage: string | undefined;
  onSpeculativeDraftChange?: (
    messages: readonly ThreadMessage[],
    draftText: string,
    systemMessage?: string,
    threadId?: string,
  ) => void;
}> = ({ modelId, modelReady, promptMode, systemMessage, onSpeculativeDraftChange }) => {
  const composerText = useAuiState((s) => s.composer.text);
  const messages = useAuiState((s) => s.thread.messages);
  const isRunning = useAuiState((s) => s.thread.isRunning);
  const threadId = useAuiState((s) => s.threads.mainThreadId);

  useEffect(() => {
    if (!modelReady || !modelId || isRunning || !onSpeculativeDraftChange) return;
    onSpeculativeDraftChange(messages, composerText, systemMessage, threadId);
  }, [
    modelId,
    modelReady,
    promptMode,
    systemMessage,
    isRunning,
    messages,
    composerText,
    threadId,
    onSpeculativeDraftChange,
  ]);

  return null;
};

const ComposerContextMeter: FC<{
  engineState: EngineState;
  modelReady: boolean;
  onCountTokens?: (prompt: string) => Promise<{ tokenCount: number; maxContextWindow: number }>;
}> = ({ engineState, modelReady, onCountTokens }) => {
  const composerText = useAuiState((s) => s.composer.text);
  const messages = useAuiState((s) => s.thread.messages);
  const [tokenCount, setTokenCount] = useState<number>(0);

  const model = engineState.model;
  const maxCtx = model?.maxContextWindow ?? 0;
  const isBootstrapUnbound = isInternalBootstrapModel(model?.id);
  const systemMessage = engineState.importedHarness?.system ?? UNIFIED_VERDANT_SYSTEM_PROMPT;
  const effectivePromptMode = engineState.importedHarness?.promptMode ?? engineState.promptMode;

  useEffect(() => {
    if (!modelReady || !model || isBootstrapUnbound || !onCountTokens) {
      setTokenCount(0);
      return;
    }
    const trimmedDraft = composerText.trim();
    if (messages.length === 0 && trimmedDraft.length === 0) {
      setTokenCount(0);
      return;
    }
    let cancelled = false;
    const timer = setTimeout(() => {
      const syntheticMessages: ThreadMessage[] =
        trimmedDraft.length > 0
          ? [
              ...messages,
              {
                id: 'draft-preview',
                role: 'user',
                createdAt: new Date(),
                content: [{ type: 'text', text: composerText }],
                attachments: [],
                metadata: { custom: {} },
              } as ThreadMessage,
            ]
          : [...messages];
      const formatted = formatMessagesForLlama(
        syntheticMessages,
        model,
        effectivePromptMode,
        systemMessage,
      );
      void onCountTokens(formatted)
        .then((res) => {
          if (!cancelled) setTokenCount(res.tokenCount);
        })
        .catch(() => {
          // Ignore transient count errors while switching models
        });
    }, 120);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [
    composerText,
    messages,
    model,
    modelReady,
    isBootstrapUnbound,
    effectivePromptMode,
    systemMessage,
    onCountTokens,
  ]);

  if (!model || isBootstrapUnbound || maxCtx <= 0) return null;

  const isOverContext = tokenCount > maxCtx;
  const displayTokens =
    tokenCount > 0
      ? `${tokenCount.toLocaleString()} / ${maxCtx.toLocaleString()} ctx`
      : `${maxCtx.toLocaleString()} ctx`;

  return (
    <>
      <span aria-hidden="true">·</span>
      <span
        data-testid="composer-context-meter"
        title={
          isOverContext
            ? `Prompt is ${tokenCount.toLocaleString()} tokens; oldest ${(tokenCount - maxCtx).toLocaleString()} tokens will slide outside the ${maxCtx.toLocaleString()}-token window.`
            : `Full native context supported: input & output up to ${maxCtx.toLocaleString()} tokens.`
        }
        className={`whitespace-nowrap shrink-0 font-mono text-[11px] tabular-nums ${
          isOverContext
            ? 'font-medium text-amber-700 dark:text-amber-400'
            : 'text-slate-500 dark:text-slate-400'
        }`}
      >
        {displayTokens}
      </span>
    </>
  );
};

const ComposerPrismMentionPicker: FC<{
  publishedPrisms: readonly PublishedPrismSummary[];
  forceOpen: boolean;
  onClose: () => void;
}> = ({ publishedPrisms, forceOpen, onClose }) => {
  const aui = useAui();
  const composerText = useAuiState((s) => s.composer.text);
  const [activeIdx, setActiveIdx] = useState(0);

  const trimmed = composerText.trimStart();
  const atMatch = trimmed.match(/^@([^\n:]*)$/);
  const query = atMatch ? atMatch[1].toLowerCase() : '';
  const open = publishedPrisms.length > 0 && (forceOpen || Boolean(atMatch));

  const matches = useMemo(() => {
    if (!open) return [];
    return publishedPrisms.filter((p) => {
      if (!query) return true;
      return (
        p.name.toLowerCase().includes(query) ||
        p.slug.toLowerCase().includes(query) ||
        p.summary.toLowerCase().includes(query)
      );
    });
  }, [open, publishedPrisms, query]);

  useEffect(() => {
    setActiveIdx(0);
  }, [query, open]);

  useEffect(() => {
    if (!open || matches.length === 0) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setActiveIdx((i) => (i + 1) % matches.length);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setActiveIdx((i) => (i - 1 + matches.length) % matches.length);
      } else if (e.key === 'Tab') {
        e.preventDefault();
        const picked = matches[activeIdx] ?? matches[0];
        if (picked) {
          aui.composer().setText(`@${picked.name} `);
          onClose();
        }
      } else if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', onKeyDown, true);
    return () => window.removeEventListener('keydown', onKeyDown, true);
  }, [open, matches, activeIdx, aui, onClose]);

  if (!open || matches.length === 0) return null;

  return (
    <div
      role="listbox"
      aria-label="Published Prisms"
      className="mx-3 mt-2 rounded-lg border border-slate-200 bg-slate-50/90 p-1.5 shadow-xs dark:border-slate-700 dark:bg-[#0F141C]"
    >
      <div className="mb-1 flex items-center justify-between px-2 text-[10px] font-semibold tracking-wide text-slate-500 uppercase dark:text-slate-400">
        <span>Published Prisms · Tab to insert</span>
        {forceOpen && (
          <button
            type="button"
            onClick={onClose}
            className="text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
          >
            Close
          </button>
        )}
      </div>
      <div className="space-y-0.5">
        {matches.map((p, idx) => (
          <button
            key={p.id}
            type="button"
            role="option"
            aria-selected={idx === activeIdx}
            onMouseEnter={() => setActiveIdx(idx)}
            onClick={() => {
              aui.composer().setText(`@${p.name} `);
              onClose();
            }}
            className={`flex w-full items-center justify-between gap-2 rounded-md px-2 py-1.5 text-left text-xs transition-colors ${
              idx === activeIdx
                ? 'bg-emerald-50 dark:bg-emerald-950/60'
                : 'hover:bg-emerald-50/60 dark:hover:bg-emerald-950/30'
            }`}
          >
            <span className="inline-flex min-w-0 items-center gap-1.5 font-medium text-slate-900 dark:text-slate-100">
              <Sparkles className="h-3.5 w-3.5 shrink-0 text-emerald-700 dark:text-emerald-400" />
              <span className="truncate">@{p.name}</span>
            </span>
            <span className="truncate text-[11px] text-slate-500 dark:text-slate-400">
              {p.summary}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
};

const MarkdownText: FC = () => (
  <div className="aui-md">
    <MarkdownTextPrimitive remarkPlugins={[remarkGfm]} />
  </div>
);

/**
 * Official Assistant-UI `ReasoningMessagePart` renderer.
 * Displays model chain-of-thought (`<think>...</think>`) in a clean, collapsible disclosure
 * separate from the visible assistant reply. Collapsed by default once complete.
 */
const ReasoningPart: FC<ReasoningMessagePartProps> = ({ text, status }) => {
  const isStreaming = status?.type === 'running';
  const [expanded, setExpanded] = useState(false);

  if (!text.trim()) return null;

  return (
    <div
      data-testid="reasoning-part"
      className="mb-3 rounded-lg border border-slate-200/80 bg-slate-50/70 dark:border-slate-800 dark:bg-[#0F141C]"
    >
      <button
        type="button"
        aria-expanded={expanded}
        onClick={() => setExpanded((prev) => !prev)}
        className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-xs font-medium text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200 transition-colors"
      >
        <span className="inline-flex items-center gap-1.5">
          {expanded ? (
            <ChevronDown className="h-3.5 w-3.5 shrink-0" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5 shrink-0" />
          )}
          {isStreaming ? (
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-600 dark:bg-emerald-400" />
          ) : (
            <Brain className="h-3.5 w-3.5 shrink-0 text-emerald-700 dark:text-emerald-400" />
          )}
          <span>{isStreaming ? 'Thinking...' : 'Thought process'}</span>
        </span>
        <span className="font-mono text-[11px] text-slate-400 dark:text-slate-500 tabular-nums">
          {expanded ? 'Hide' : 'Show'}
        </span>
      </button>
      {expanded && (
        <div className="border-t border-slate-200/80 px-3 py-2.5 text-xs leading-relaxed whitespace-pre-wrap text-slate-600 dark:border-slate-800 dark:text-slate-400">
          {text}
        </div>
      )}
    </div>
  );
};

function formatToolJson(value: unknown): string {
  if (value === undefined) return '';
  if (typeof value === 'string') return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

/**
 * Official Assistant-UI `ToolCallMessagePartComponent` fallback renderer.
 * Visualizes client-side Agentic tool calls (`calculate`, `inspect_runtime`, `workspace_notes`,
 * `current_datetime`, `analyze_text`) with live execution status and collapsible I/O inspection.
 */
const VerdantToolCallPart: ToolCallMessagePartComponent = ({
  toolName,
  argsText,
  args,
  result,
  isError,
  status,
}) => {
  const { onOpenPrism, onEditPrism } = useContext(AssistantMessageContext);
  const [expanded, setExpanded] = useState(false);
  const isRunning = status?.type === 'running' || (result === undefined && !isError);
  const formattedArgs = argsText?.trim() || formatToolJson(args);
  const formattedResult = formatToolJson(result);
  const prismPayload =
    result &&
    typeof result === 'object' &&
    (result as Record<string, unknown>).__prismInvocation === true
      ? (result as {
          prismId: string;
          prismName: string;
          digest: string;
          output: unknown;
        })
      : null;

  if (prismPayload) {
    const outObj =
      prismPayload.output && typeof prismPayload.output === 'object' && !Array.isArray(prismPayload.output)
        ? (prismPayload.output as Record<string, unknown>)
        : null;
    const summaryEntries = outObj
      ? Object.entries(outObj)
          .filter(([, v]) => typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean')
          .slice(0, 4)
      : [];

    return (
      <div
        data-testid={`prism-stream-card-${prismPayload.prismId}`}
        className="my-2.5 rounded-xl border border-emerald-700/30 bg-emerald-50/40 dark:border-emerald-500/30 dark:bg-emerald-950/20"
      >
        <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2">
          <button
            type="button"
            aria-expanded={expanded}
            onClick={() => setExpanded((prev) => !prev)}
            className="flex min-w-0 flex-1 items-center gap-1.5 text-left text-xs font-semibold text-emerald-950 dark:text-emerald-200"
          >
            {expanded ? (
              <ChevronDown className="h-3.5 w-3.5 shrink-0 text-emerald-700 dark:text-emerald-400" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5 shrink-0 text-emerald-700 dark:text-emerald-400" />
            )}
            <Sparkles className="h-3.5 w-3.5 shrink-0 text-emerald-700 dark:text-emerald-400" />
            <span className="truncate">{prismPayload.prismName}</span>
            <span className="font-mono text-[10px] font-normal text-emerald-800/75 dark:text-emerald-400/75">
              #{shortPrismDigest(prismPayload.digest)}
            </span>
          </button>
          <div className="flex shrink-0 items-center gap-1.5">
            {onEditPrism && (
              <button
                type="button"
                onClick={() => onEditPrism(prismPayload.prismId)}
                className="inline-flex items-center rounded-md border border-emerald-700/30 bg-white px-2 py-1 text-[11px] font-medium text-emerald-900 hover:bg-emerald-50 dark:border-emerald-500/30 dark:bg-[#0F141C] dark:text-emerald-300 dark:hover:bg-emerald-950/50 transition-colors"
              >
                Studio
              </button>
            )}
            {onOpenPrism && (
              <button
                type="button"
                onClick={() => onOpenPrism(prismPayload.prismId)}
                className="inline-flex items-center gap-1 rounded-md bg-[#0F5132] px-2 py-1 text-[11px] font-medium text-white hover:bg-[#0B3D25] transition-colors"
              >
                <ExternalLink className="h-3 w-3" />
                <span>Open App</span>
              </button>
            )}
          </div>
        </div>
        {summaryEntries.length > 0 && !expanded && (
          <div className="flex flex-wrap gap-x-3 gap-y-1 border-t border-emerald-700/15 px-3 py-1.5 text-[11px] text-slate-700 dark:border-emerald-500/15 dark:text-slate-300">
            {summaryEntries.map(([k, v]) => (
              <span key={k}>
                <span className="text-slate-500 dark:text-slate-400">{k}:</span>{' '}
                <strong className="font-mono font-medium">{String(v)}</strong>
              </span>
            ))}
          </div>
        )}
        {expanded && (
          <div className="border-t border-emerald-700/20 px-3 py-2 text-[11px] dark:border-emerald-500/20">
            <pre className="overflow-x-auto rounded-md bg-white p-2 font-mono text-slate-700 dark:bg-[#0B0F14] dark:text-slate-300">
              {formatToolJson(prismPayload.output)}
            </pre>
          </div>
        )}
      </div>
    );
  }

  return (
    <div
      data-testid={`tool-call-${toolName}`}
      className="my-2.5 rounded-lg border border-slate-200/90 bg-slate-50/80 dark:border-slate-800 dark:bg-[#0F141C]"
    >
      <button
        type="button"
        aria-expanded={expanded}
        onClick={() => setExpanded((prev) => !prev)}
        className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-xs font-medium text-slate-700 hover:text-slate-950 dark:text-slate-300 dark:hover:text-white transition-colors"
      >
        <span className="inline-flex min-w-0 items-center gap-1.5">
          {expanded ? (
            <ChevronDown className="h-3.5 w-3.5 shrink-0 text-slate-400" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5 shrink-0 text-slate-400" />
          )}
          <Wrench className="h-3.5 w-3.5 shrink-0 text-emerald-700 dark:text-emerald-400" />
          <span className="truncate font-mono text-[11px]">{toolName}</span>
        </span>
        <span className="inline-flex shrink-0 items-center gap-1.5 text-[11px]">
          {isRunning ? (
            <>
              <Loader2 className="h-3.5 w-3.5 animate-spin text-emerald-700 dark:text-emerald-400" />
              <span className="text-emerald-800 dark:text-emerald-400">Running</span>
            </>
          ) : isError ? (
            <>
              <AlertCircle className="h-3.5 w-3.5 text-red-600 dark:text-red-400" />
              <span className="text-red-700 dark:text-red-400">Error</span>
            </>
          ) : (
            <>
              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-700 dark:text-emerald-400" />
              <span className="text-slate-500 dark:text-slate-400">Done</span>
            </>
          )}
        </span>
      </button>

      {expanded && (
        <div className="space-y-2 border-t border-slate-200/80 px-3 py-2.5 text-[11px] dark:border-slate-800">
          {formattedArgs && (
            <div>
              <div className="mb-1 font-semibold tracking-wide text-slate-500 uppercase dark:text-slate-400">
                Arguments
              </div>
              <pre className="overflow-x-auto rounded-md bg-white p-2 font-mono text-slate-700 dark:bg-[#0B0F14] dark:text-slate-300">
                {formattedArgs}
              </pre>
            </div>
          )}
          {result !== undefined && (
            <div>
              <div className="mb-1 font-semibold tracking-wide text-slate-500 uppercase dark:text-slate-400">
                {isError ? 'Error' : 'Result'}
              </div>
              <pre
                className={`overflow-x-auto rounded-md p-2 font-mono ${
                  isError
                    ? 'bg-red-50 text-red-700 dark:bg-red-950/40 dark:text-red-300'
                    : 'bg-white text-slate-700 dark:bg-[#0B0F14] dark:text-slate-300'
                }`}
              >
                {formattedResult}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const ComposerAttachmentItem: FC = () => (
  <AttachmentPrimitive.Root className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-slate-50 px-2 py-1 text-xs text-slate-700 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-300">
    <FileText className="h-3.5 w-3.5 shrink-0 text-slate-500 dark:text-slate-400" />
    <span className="max-w-[160px] truncate">
      <AttachmentPrimitive.Name />
    </span>
    <AttachmentPrimitive.Remove
      aria-label="Remove attachment"
      className="inline-flex h-4 w-4 items-center justify-center rounded text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
    >
      <X className="h-3 w-3" />
    </AttachmentPrimitive.Remove>
  </AttachmentPrimitive.Root>
);

const UserAttachmentItem: FC = () => (
  <AttachmentPrimitive.Root className="inline-flex items-center gap-1.5 rounded-md border border-slate-200 bg-white px-2.5 py-1 text-xs text-slate-600 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-300">
    <FileText className="h-3.5 w-3.5 shrink-0 text-slate-400" />
    <span className="max-w-[200px] truncate">
      <AttachmentPrimitive.Name />
    </span>
  </AttachmentPrimitive.Root>
);

const BranchPicker: FC = () => (
  <BranchPickerPrimitive.Root
    hideWhenSingleBranch
    className="inline-flex items-center gap-1 text-xs text-slate-500 dark:text-slate-400 tabular-nums"
  >
    <BranchPickerPrimitive.Previous
      aria-label="Previous response branch"
      className="inline-flex h-7 w-7 items-center justify-center rounded-md hover:bg-slate-100 hover:text-slate-900 dark:hover:bg-slate-800 dark:hover:text-slate-100 transition-colors focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-emerald-700"
    >
      <ChevronLeft className="h-3.5 w-3.5" />
    </BranchPickerPrimitive.Previous>
    <span className="px-1 font-mono text-[11px]">
      <BranchPickerPrimitive.Number /> / <BranchPickerPrimitive.Count />
    </span>
    <BranchPickerPrimitive.Next
      aria-label="Next response branch"
      className="inline-flex h-7 w-7 items-center justify-center rounded-md hover:bg-slate-100 hover:text-slate-900 dark:hover:bg-slate-800 dark:hover:text-slate-100 transition-colors focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-emerald-700"
    >
      <ChevronRight className="h-3.5 w-3.5" />
    </BranchPickerPrimitive.Next>
  </BranchPickerPrimitive.Root>
);

const UserMessage: FC = () => (
  <MessagePrimitive.Root className="group mx-auto w-full max-w-3xl px-4 py-3 sm:px-6">
    <div className="flex flex-col items-end gap-1.5">
      <div className="flex flex-wrap justify-end gap-1.5">
        <MessagePrimitive.Attachments components={{ Attachment: UserAttachmentItem }} />
      </div>
      <div className="max-w-[85%] rounded-2xl bg-slate-900 px-4 py-3 text-[15px] leading-relaxed text-white dark:bg-slate-100 dark:text-slate-950">
        <MessagePrimitive.Parts />
      </div>
      <div className="flex items-center gap-2">
        <BranchPicker />
        <ActionBarPrimitive.Root
          hideWhenRunning
          autohide="not-last"
          className="flex items-center gap-1 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity"
        >
          <ActionBarPrimitive.Edit
            aria-label="Edit message"
            className="inline-flex h-8 items-center gap-1.5 rounded-md px-2 text-xs font-medium text-slate-500 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-100 transition-colors whitespace-nowrap"
          >
            <Pencil className="h-3.5 w-3.5" />
            <span>Edit</span>
          </ActionBarPrimitive.Edit>
        </ActionBarPrimitive.Root>
      </div>
    </div>
  </MessagePrimitive.Root>
);

const EditComposer: FC = () => (
  <div className="mx-auto w-full max-w-3xl px-4 py-3 sm:px-6">
    <ComposerPrimitive.Root className="rounded-xl border border-slate-300 bg-white p-3 dark:border-slate-700 dark:bg-[#121821]">
      <ComposerPrimitive.Input
        aria-label="Edit message prompt"
        className="w-full resize-none bg-transparent text-[15px] text-slate-900 outline-none placeholder:text-slate-400 dark:text-slate-100"
        rows={3}
      />
      <div className="mt-3 flex items-center justify-end gap-2">
        <ComposerPrimitive.Cancel className="rounded-lg border border-slate-200 px-3.5 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800 transition-colors whitespace-nowrap">
          Cancel
        </ComposerPrimitive.Cancel>
        <ComposerPrimitive.Send className="rounded-lg bg-[#0F5132] px-3.5 py-1.5 text-xs font-medium text-white hover:bg-[#0B3D25] transition-colors whitespace-nowrap">
          Update and run
        </ComposerPrimitive.Send>
      </div>
    </ComposerPrimitive.Root>
  </div>
);

interface AssistantMessageContextValue {
  readonly statusDetail: string;
  readonly onInspectRoute: () => void;
  readonly onOpenPrism?: (prismId: string) => void;
  readonly onEditPrism?: (prismId: string) => void;
}

const AssistantMessageContext = createContext<AssistantMessageContextValue>({
  statusDetail: '',
  onInspectRoute: () => {},
});

const AssistantMessage: FC = () => {
  const { statusDetail, onInspectRoute } = useContext(AssistantMessageContext);
  const prismRoute = useAuiState(
    (s) =>
      (
        s.message.metadata.custom as
          | { readonly prismRoute?: CognitiveRouteCertificate }
          | undefined
      )?.prismRoute,
  );
  return (
    <MessagePrimitive.Root className="group mx-auto w-full max-w-3xl px-4 py-3 sm:px-6">
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between gap-2 text-xs text-slate-500 dark:text-slate-400">
          <span className="font-semibold text-slate-800 dark:text-slate-200">VerdantUI</span>
          {prismRoute && (
            <button
              type="button"
              onClick={onInspectRoute}
              title={prismRoute.rationale}
              className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 font-mono text-[10px] text-slate-500 hover:bg-slate-100 hover:text-emerald-800 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-emerald-400 transition-colors"
            >
              <span>{prismRoute.label}</span>
              <span aria-hidden="true">·</span>
              <span>#{prismRoute.digest}</span>
            </button>
          )}
        </div>

        <div className="rounded-xl border border-slate-200/90 bg-white px-4 py-3.5 text-slate-800 dark:border-slate-800 dark:bg-[#121821] dark:text-slate-100">
          <AuiIf condition={(s) => s.message.isLast && s.thread.isRunning && s.message.parts.length === 0}>
            <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400 tabular-nums">
              <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-[#0F5132] dark:bg-emerald-400" />
              <span>{statusDetail || 'Preparing response...'}</span>
            </div>
          </AuiIf>
          <MessagePrimitive.Parts
            components={{
              Text: MarkdownText,
              Reasoning: ReasoningPart,
              tools: {
                Fallback: VerdantToolCallPart,
              },
            }}
          />
          <MessagePrimitive.Error>
            <ErrorPrimitive.Root className="mt-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-300">
              <ErrorPrimitive.Message />
            </ErrorPrimitive.Root>
          </MessagePrimitive.Error>
        </div>

        <div className="flex items-center gap-2">
          <BranchPicker />
          <ActionBarPrimitive.Root
            hideWhenRunning
            autohide="not-last"
            className="flex items-center gap-1 text-slate-500 dark:text-slate-400"
          >
            <ActionBarPrimitive.Copy
              aria-label="Copy response"
              className="inline-flex h-8 items-center gap-1.5 rounded-md px-2 text-xs font-medium hover:bg-slate-100 hover:text-slate-900 dark:hover:bg-slate-800 dark:hover:text-slate-100 transition-colors whitespace-nowrap"
            >
              <AuiIf condition={(s) => s.message.isCopied}>
                <Check className="h-3.5 w-3.5 text-emerald-700 dark:text-emerald-400" />
                <span>Copied</span>
              </AuiIf>
              <AuiIf condition={(s) => !s.message.isCopied}>
                <Copy className="h-3.5 w-3.5" />
                <span>Copy</span>
              </AuiIf>
            </ActionBarPrimitive.Copy>

            <ActionBarPrimitive.Reload
              aria-label="Regenerate response"
              className="inline-flex h-8 items-center gap-1.5 rounded-md px-2 text-xs font-medium hover:bg-slate-100 hover:text-slate-900 dark:hover:bg-slate-800 dark:hover:text-slate-100 transition-colors whitespace-nowrap"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              <span>Regenerate</span>
            </ActionBarPrimitive.Reload>
          </ActionBarPrimitive.Root>
        </div>
      </div>
    </MessagePrimitive.Root>
  );
};

export const ThreadView: FC<ThreadViewProps> = ({
  engineState,
  onCancelDownload,
  onRetryLoad,
  onDismissError,
  onToggleInspector,
  onOpenPrism,
  onEditPrism,
  publishedPrisms = [],
  onSpeculativeDraftChange,
  onCountTokens,
}) => {
  const [prismPickerOpen, setPrismPickerOpen] = useState(false);
  const isDownloading = engineState.status === 'downloading';
  const isModelLoading = isDownloading || engineState.status === 'loading';
  const modelReady = engineState.model !== null && !isModelLoading;
  const isBootstrapUnbound = isInternalBootstrapModel(engineState.model?.id);
  const importedHarness = engineState.importedHarness;
  const effectiveSystemMessage = importedHarness?.system ?? UNIFIED_VERDANT_SYSTEM_PROMPT;

  const activeSuggestions =
    importedHarness?.suggestions && importedHarness.suggestions.length > 0
      ? importedHarness.suggestions
      : UNIFIED_VERDANT_SUGGESTIONS;

  const assistantContextValue = useMemo<AssistantMessageContextValue>(
    () => ({
      statusDetail: engineState.statusDetail,
      onInspectRoute: onToggleInspector,
      onOpenPrism,
      onEditPrism,
    }),
    [engineState.statusDetail, onToggleInspector, onOpenPrism, onEditPrism],
  );

  return (
    <AssistantMessageContext.Provider value={assistantContextValue}>
      <ThreadPrimitive.Root className="relative flex h-full flex-col overflow-hidden bg-[#F8F9FA] dark:bg-[#0B0F14]">
      <SpeculativePrefillDriver
        modelId={engineState.model?.id ?? null}
        modelReady={modelReady && !isBootstrapUnbound}
        promptMode={importedHarness?.promptMode ?? engineState.promptMode}
        systemMessage={effectiveSystemMessage}
        onSpeculativeDraftChange={onSpeculativeDraftChange}
      />
      <ThreadPrimitive.Viewport className="flex flex-1 flex-col overflow-y-auto scroll-smooth">
        <div className="flex-1">
          <ThreadPrimitive.Empty>
            <div className="mx-auto flex max-w-3xl flex-col px-4 pt-12 pb-8 sm:px-6 sm:pt-16">
              <div className="mb-7">
                <h1
                  className="text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl dark:text-slate-50"
                  style={{ textWrap: 'balance' }}
                >
                  VerdantUI
                </h1>
                <p className="mt-2.5 max-w-2xl text-[15px] leading-relaxed text-slate-600 dark:text-slate-400">
                  A free, private, 100% on-device AI workspace running directly in your browser.
                  Every prompt adapts automatically across built-in tools, deeper reasoning, and
                  your own Prisms — zero accounts, zero tracking, zero servers.
                </p>

                <div className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-500 dark:text-slate-400 tabular-nums">
                  <span>
                    Model Stream:{' '}
                    <strong className="font-medium text-slate-800 dark:text-slate-200">
                      {!engineState.modelStreamEnabled && isBootstrapUnbound
                        ? 'Off (instant built-in tools active)'
                        : isModelLoading
                          ? `${engineState.statusDetail} (${engineState.progressPercent}%)`
                          : isBootstrapUnbound
                            ? 'Ready'
                            : (engineState.model?.displayName ?? 'VerdantUI')}
                    </strong>
                  </span>
                  {!isBootstrapUnbound && engineState.model && (
                    <>
                      <span aria-hidden="true">·</span>
                      <span>
                        Format:{' '}
                        <span className="font-mono">
                          {engineState.model.hparams.primaryQuantization}
                        </span>
                      </span>
                      <span aria-hidden="true">·</span>
                      <span>
                        Context:{' '}
                        <span className="font-mono">
                          {engineState.model.maxContextWindow.toLocaleString()} tok
                        </span>
                      </span>
                    </>
                  )}
                  {importedHarness && (
                    <>
                      <span aria-hidden="true">·</span>
                      <span>
                        Override:{' '}
                        <strong className="font-medium text-slate-800 dark:text-slate-200">
                          {importedHarness.name}
                        </strong>
                      </span>
                    </>
                  )}
                </div>
              </div>

              {isModelLoading && (
                <div
                  role="status"
                  aria-live="polite"
                  className="mb-6 rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-[#121821]"
                >
                  <div className="flex items-center justify-between gap-3 text-xs text-slate-600 dark:text-slate-300">
                    <span className="truncate">{engineState.statusDetail}</span>
                    <div className="flex shrink-0 items-center gap-2">
                      <span className="font-mono tabular-nums">{engineState.progressPercent}%</span>
                      <button
                        type="button"
                        onClick={onCancelDownload}
                        className="rounded-md border border-slate-200 px-2 py-0.5 text-[11px] font-medium text-slate-700 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-800"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                  <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
                    <div
                      className="h-full bg-[#0F5132] transition-transform duration-200 dark:bg-emerald-500"
                      style={{
                        width: '100%',
                        transform: `scaleX(${Math.max(0.05, engineState.progressPercent / 100)})`,
                        transformOrigin: 'left',
                      }}
                    />
                  </div>
                </div>
              )}

              {activeSuggestions.length > 0 && (
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  {activeSuggestions.map((item) => (
                    <ThreadPrimitive.Suggestion
                      key={item.prompt}
                      prompt={item.prompt}
                      method="replace"
                      autoSend={!isModelLoading}
                      className="flex flex-col items-start rounded-xl border border-slate-200 bg-white p-4 text-left hover:border-slate-300 hover:bg-slate-50/70 dark:border-slate-800 dark:bg-[#121821] dark:hover:border-slate-700 dark:hover:bg-[#161E29] transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-700"
                    >
                      <span className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                        {item.label ?? item.prompt}
                      </span>
                      {item.label && (
                        <span className="mt-1 line-clamp-2 text-xs text-slate-500 dark:text-slate-400">
                          {item.prompt}
                        </span>
                      )}
                    </ThreadPrimitive.Suggestion>
                  ))}
                </div>
              )}
            </div>
          </ThreadPrimitive.Empty>

          <div className="py-4">
            <ThreadPrimitive.Messages
              components={{
                UserMessage,
                EditComposer,
                AssistantMessage,
              }}
            />
          </div>
        </div>

        <ThreadPrimitive.ViewportFooter className="sticky bottom-0 mx-auto mt-auto w-full max-w-3xl bg-gradient-to-t from-[#F8F9FA] via-[#F8F9FA] to-transparent px-4 pt-6 pb-4 sm:px-6 dark:from-[#0B0F14] dark:via-[#0B0F14]">
          {isModelLoading && (
            <div
              role="status"
              aria-live="polite"
              className="mb-3 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-xs shadow-xs dark:border-slate-800 dark:bg-[#121821]"
            >
              <div className="flex items-center justify-between gap-3 text-slate-600 dark:text-slate-300">
                <span className="truncate">{engineState.statusDetail}</span>
                <div className="flex shrink-0 items-center gap-2">
                  <span className="font-mono tabular-nums">{engineState.progressPercent}%</span>
                  <button
                    type="button"
                    onClick={onCancelDownload}
                    className="rounded-md border border-slate-200 px-2 py-0.5 text-[11px] font-medium text-slate-700 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-800"
                  >
                    Cancel
                  </button>
                </div>
              </div>
              <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
                <div
                  className="h-full bg-[#0F5132] transition-transform duration-200 dark:bg-emerald-500"
                  style={{
                    width: '100%',
                    transform: `scaleX(${Math.max(0.05, engineState.progressPercent / 100)})`,
                    transformOrigin: 'left',
                  }}
                />
              </div>
            </div>
          )}

          {engineState.errorMessage && (
            <div
              role="alert"
              className="mb-3 flex items-start justify-between gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-2.5 text-xs text-red-700 shadow-xs dark:border-red-900/60 dark:bg-red-950/60 dark:text-red-200"
            >
              <span className="min-w-0 flex-1 leading-relaxed">{engineState.errorMessage}</span>
              <div className="flex shrink-0 items-center gap-1.5">
                {onRetryLoad && (
                  <button
                    type="button"
                    onClick={onRetryLoad}
                    className="inline-flex items-center gap-1 rounded-md border border-red-300 bg-white px-2.5 py-1 text-[11px] font-medium text-red-700 hover:bg-red-100 dark:border-red-800 dark:bg-red-950 dark:text-red-200 dark:hover:bg-red-900/60 transition-colors"
                  >
                    <RefreshCw className="h-3 w-3" />
                    <span>Retry</span>
                  </button>
                )}
                {onDismissError && (
                  <button
                    type="button"
                    onClick={onDismissError}
                    aria-label="Dismiss error"
                    className="inline-flex h-6 w-6 items-center justify-center rounded-md text-red-500 hover:bg-red-100 hover:text-red-700 dark:text-red-400 dark:hover:bg-red-900/60 dark:hover:text-red-200 transition-colors"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
            </div>
          )}
          <div className="mb-2 flex justify-center">
            <ThreadPrimitive.ScrollToBottom
              aria-label="Scroll to latest message"
              className="inline-flex h-8 items-center gap-1.5 rounded-full border border-slate-200 bg-white px-3 text-xs font-medium text-slate-700 shadow-xs hover:bg-slate-50 disabled:invisible dark:border-slate-700 dark:bg-[#121821] dark:text-slate-200 dark:hover:bg-slate-800"
            >
              <ArrowDown className="h-3.5 w-3.5" />
              <span>Latest</span>
            </ThreadPrimitive.ScrollToBottom>
          </div>

          <ComposerPrimitive.Root className="rounded-xl border border-slate-200 bg-white shadow-xs focus-within:border-emerald-800 dark:border-slate-800 dark:bg-[#121821] dark:focus-within:border-emerald-500 transition-colors">
            <ComposerPrismMentionPicker
              publishedPrisms={publishedPrisms}
              forceOpen={prismPickerOpen}
              onClose={() => setPrismPickerOpen(false)}
            />

            <div className="flex flex-wrap gap-1.5 px-3 pt-2.5 empty:hidden">
              <ComposerPrimitive.Attachments
                components={{ Attachment: ComposerAttachmentItem }}
              />
            </div>

            <ComposerPrimitive.Input
              aria-label="Message prompt"
              placeholder={
                publishedPrisms.length > 0
                  ? 'Ask VerdantUI anything, or type @ to address a Published Prism...'
                  : 'Ask VerdantUI anything...'
              }
              rows={2}
              autoFocus
              className="max-h-56 w-full resize-none overflow-y-auto bg-transparent px-4 pt-3.5 pb-2 text-[15px] text-slate-900 outline-none placeholder:text-slate-400 dark:text-slate-100 dark:placeholder:text-slate-500"
            />

            <div className="flex flex-wrap items-center justify-between gap-2 border-t border-slate-100 px-3 py-2 dark:border-slate-800/80">
              <div className="flex min-w-0 flex-wrap items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                <ComposerPrimitive.AddAttachment
                  aria-label="Attach text or code file"
                  title="Attach text or code file"
                  className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-slate-500 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-100 transition-colors"
                >
                  <Paperclip className="h-3.5 w-3.5" />
                </ComposerPrimitive.AddAttachment>
                <span aria-hidden="true">·</span>
                <button
                  type="button"
                  onClick={onToggleInspector}
                  className="max-w-[200px] truncate font-medium text-slate-700 hover:text-emerald-800 dark:text-slate-300 dark:hover:text-emerald-400 transition-colors"
                >
                  VerdantUI
                </button>
                {publishedPrisms.length > 0 && (
                  <>
                    <span aria-hidden="true">·</span>
                    <button
                      type="button"
                      onClick={() => setPrismPickerOpen((prev) => !prev)}
                      className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 font-medium text-emerald-800 hover:bg-emerald-50 dark:text-emerald-400 dark:hover:bg-emerald-950/50 transition-colors"
                      title="Address a Published Prism with @"
                    >
                      <Sparkles className="h-3 w-3" />
                      <span>
                        {publishedPrisms.length} Prism{publishedPrisms.length === 1 ? '' : 's'}
                      </span>
                    </button>
                  </>
                )}
                <ComposerContextMeter
                  engineState={engineState}
                  modelReady={modelReady}
                  onCountTokens={onCountTokens}
                />
                {engineState.lastRouteCertificate && (
                  <>
                    <span className="hidden sm:inline" aria-hidden="true">
                      ·
                    </span>
                    <button
                      type="button"
                      onClick={onToggleInspector}
                      title={engineState.lastRouteCertificate.rationale}
                      className="hidden shrink-0 items-center gap-1 whitespace-nowrap font-mono text-[11px] hover:text-emerald-800 sm:inline-flex dark:hover:text-emerald-400 transition-colors"
                    >
                      <SlidersHorizontal className="h-3 w-3" />
                      <span>#{engineState.lastRouteCertificate.digest}</span>
                    </button>
                  </>
                )}
                {engineState.prefillProgress ? (
                  <>
                    <span className="hidden sm:inline" aria-hidden="true">
                      ·
                    </span>
                    <span className="hidden truncate font-mono text-[11px] tabular-nums text-emerald-800 dark:text-emerald-400 sm:inline">
                      {engineState.prefillProgress.totalTokens > 0 &&
                      engineState.prefillProgress.completedTokens <
                        engineState.prefillProgress.totalTokens
                        ? `prefill ${engineState.prefillProgress.completedTokens}/${engineState.prefillProgress.totalTokens}${engineState.prefillProgress.detail ? ` · ${engineState.prefillProgress.detail}` : ''}`
                        : (engineState.prefillProgress.detail ??
                          `prefill ${engineState.prefillProgress.completedTokens}/${engineState.prefillProgress.totalTokens}`)}
                    </span>
                  </>
                ) : engineState.lastTokensPerSec > 0 ? (
                  <>
                    <span className="hidden sm:inline" aria-hidden="true">
                      ·
                    </span>
                    <span className="hidden font-mono tabular-nums sm:inline">
                      {engineState.lastTokensPerSec.toFixed(1)} tok/s
                    </span>
                  </>
                ) : null}
              </div>

              <div className="flex shrink-0 items-center gap-2">
                <AuiIf condition={(s) => s.thread.isRunning}>
                  <ComposerPrimitive.Cancel
                    aria-label="Stop generation"
                    className="inline-flex h-9 items-center gap-1.5 rounded-lg bg-slate-900 px-3.5 text-xs font-medium text-white hover:bg-slate-800 dark:bg-slate-100 dark:text-slate-950 dark:hover:bg-slate-200 transition-colors whitespace-nowrap"
                  >
                    <Square className="h-3 w-3 fill-current" />
                    <span>Stop</span>
                  </ComposerPrimitive.Cancel>
                </AuiIf>
                <AuiIf condition={(s) => !s.thread.isRunning}>
                  <ComposerPrimitive.Send
                    disabled={isModelLoading}
                    aria-label="Send message"
                    className="inline-flex h-9 items-center gap-1.5 rounded-lg bg-[#0F5132] px-3.5 text-xs font-medium text-white hover:bg-[#0B3D25] disabled:cursor-not-allowed disabled:opacity-45 transition-colors whitespace-nowrap"
                  >
                    <span>Send</span>
                    <ArrowUp className="h-3.5 w-3.5" />
                  </ComposerPrimitive.Send>
                </AuiIf>
              </div>
            </div>
          </ComposerPrimitive.Root>
        </ThreadPrimitive.ViewportFooter>
      </ThreadPrimitive.Viewport>
      </ThreadPrimitive.Root>
    </AssistantMessageContext.Provider>
  );
};
