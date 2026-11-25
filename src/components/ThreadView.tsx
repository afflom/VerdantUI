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
  useAuiState,
  type ReasoningMessagePartProps,
  type ThreadMessage,
} from '@assistant-ui/react';
import { MarkdownTextPrimitive } from '@assistant-ui/react-markdown';
import remarkGfm from 'remark-gfm';
import {
  ArrowDown,
  ArrowUp,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Copy,
  FileJson,
  FileText,
  Layers,
  Paperclip,
  Pencil,
  RefreshCw,
  SlidersHorizontal,
  Square,
  X,
} from 'lucide-react';
import { detectChatTemplate, formatMessagesForLlama } from '../engine/assistantAdapter';
import { isInternalBootstrapModel } from '../engine/modelCatalog';
import type { EngineState } from '../engine/workerClient';

interface ThreadViewProps {
  engineState: EngineState;
  onOpenModels: () => void;
  onOpenHarness: () => void;
  onCancelDownload: () => void;
  onToggleInspector: () => void;
  onSpeculativeDraftChange?: (
    messages: readonly ThreadMessage[],
    draftText: string,
    systemMessage?: string,
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
  ) => void;
}> = ({ modelId, modelReady, promptMode, systemMessage, onSpeculativeDraftChange }) => {
  const composerText = useAuiState((s) => s.composer.text);
  const messages = useAuiState((s) => s.thread.messages);
  const isRunning = useAuiState((s) => s.thread.isRunning);

  useEffect(() => {
    if (!modelReady || !modelId || isRunning || !onSpeculativeDraftChange) return;
    onSpeculativeDraftChange(messages, composerText, systemMessage);
  }, [
    modelId,
    modelReady,
    promptMode,
    systemMessage,
    isRunning,
    messages,
    composerText,
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
  const systemMessage = engineState.importedHarness?.system;
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
    <div className="mb-3 rounded-lg border border-slate-200/80 bg-slate-50/70 dark:border-slate-800 dark:bg-[#0F141C]">
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
  readonly senderLabel: string;
  readonly statusDetail: string;
}

const AssistantMessageContext = createContext<AssistantMessageContextValue>({
  senderLabel: 'VerdantUI',
  statusDetail: '',
});

const AssistantMessage: FC = () => {
  const { senderLabel, statusDetail } = useContext(AssistantMessageContext);
  return (
    <MessagePrimitive.Root className="group mx-auto w-full max-w-3xl px-4 py-3 sm:px-6">
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
          <span className="font-semibold text-slate-800 dark:text-slate-200">{senderLabel}</span>
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
  onOpenModels,
  onOpenHarness,
  onCancelDownload,
  onToggleInspector,
  onSpeculativeDraftChange,
  onCountTokens,
}) => {
  const isDownloading = engineState.status === 'downloading';
  const isModelLoading = isDownloading || engineState.status === 'loading';
  const modelReady = engineState.model !== null && !isModelLoading;
  const isBootstrapUnbound = isInternalBootstrapModel(engineState.model?.id);
  const importedHarness = engineState.importedHarness;
  const detectedTemplate = detectChatTemplate(engineState.model);

  const senderLabel =
    !isBootstrapUnbound && engineState.model?.displayName
      ? engineState.model.displayName
      : 'VerdantUI';

  const assistantContextValue = useMemo<AssistantMessageContextValue>(
    () => ({
      senderLabel,
      statusDetail: engineState.statusDetail,
    }),
    [senderLabel, engineState.statusDetail],
  );

  return (
    <AssistantMessageContext.Provider value={assistantContextValue}>
      <ThreadPrimitive.Root className="relative flex h-full flex-col overflow-hidden bg-[#F8F9FA] dark:bg-[#0B0F14]">
      <SpeculativePrefillDriver
        modelId={engineState.model?.id ?? null}
        modelReady={modelReady}
        promptMode={importedHarness?.promptMode ?? engineState.promptMode}
        systemMessage={importedHarness?.system}
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
                  Run open-weight GGUF models directly in your browser. Models run unmodified by
                  default, or with an imported harness configuration.
                </p>

                <div className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-500 dark:text-slate-400 tabular-nums">
                  <span>
                    Model:{' '}
                    <strong className="font-medium text-slate-800 dark:text-slate-200">
                      {isBootstrapUnbound
                        ? 'None selected'
                        : (engineState.model?.displayName ?? 'Loading...')}
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
                      <span aria-hidden="true">·</span>
                      <span>
                        Template: <span className="font-mono">{detectedTemplate}</span>
                      </span>
                    </>
                  )}
                  <span aria-hidden="true">·</span>
                  <span>
                    Harness:{' '}
                    <strong className="font-medium text-slate-800 dark:text-slate-200">
                      {importedHarness ? importedHarness.name : 'None'}
                    </strong>
                  </span>
                </div>
              </div>

              {isBootstrapUnbound && !isModelLoading && (
                <div
                  data-testid="model-selection-banner"
                  className="mb-6 flex flex-col gap-3 rounded-xl border border-slate-200 bg-white p-4 sm:flex-row sm:items-center sm:justify-between dark:border-slate-800 dark:bg-[#121821]"
                >
                  <div className="min-w-0">
                    <div className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                      Select a model to get started
                    </div>
                    <p className="mt-0.5 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
                      Download a model from the catalog, open a local{' '}
                      <code className="font-mono">.gguf</code> file directly from disk, or import a
                      JSON harness configuration.
                    </p>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <button
                      type="button"
                      onClick={onOpenModels}
                      className="inline-flex h-9 items-center gap-1.5 rounded-lg bg-[#0F5132] px-3.5 text-xs font-medium text-white hover:bg-[#0B3D25] transition-colors whitespace-nowrap"
                    >
                      <Layers className="h-3.5 w-3.5" />
                      <span>Browse Models</span>
                    </button>
                    <button
                      type="button"
                      onClick={onOpenHarness}
                      className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-200 dark:hover:bg-slate-800 transition-colors whitespace-nowrap"
                    >
                      <FileJson className="h-3.5 w-3.5" />
                      <span>Import Harness</span>
                    </button>
                  </div>
                </div>
              )}

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
                      {isModelLoading && (
                        <button
                          type="button"
                          onClick={onCancelDownload}
                          className="rounded-md border border-slate-200 px-2 py-0.5 text-[11px] font-medium text-slate-700 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-800"
                        >
                          Cancel
                        </button>
                      )}
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

              {engineState.errorMessage && (
                <div
                  role="alert"
                  className="mb-6 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-xs text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-300"
                >
                  {engineState.errorMessage}
                </div>
              )}

              {/* Only render starter suggestions if explicitly provided by an imported harness */}
              {importedHarness?.suggestions && importedHarness.suggestions.length > 0 && (
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  {importedHarness.suggestions.map((item) => (
                    <ThreadPrimitive.Suggestion
                      key={item.prompt}
                      prompt={item.prompt}
                      method="replace"
                      autoSend={modelReady && !isBootstrapUnbound}
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
            <div className="flex flex-wrap gap-1.5 px-3 pt-2.5 empty:hidden">
              <ComposerPrimitive.Attachments
                components={{ Attachment: ComposerAttachmentItem }}
              />
            </div>

            <ComposerPrimitive.Input
              aria-label="Message prompt"
              placeholder={
                modelReady
                  ? isBootstrapUnbound
                    ? 'Select a model in Models to start chatting...'
                    : `Message ${engineState.model?.displayName ?? 'VerdantUI'}...`
                  : 'Loading model...'
              }
              rows={2}
              autoFocus
              className="max-h-56 w-full resize-none overflow-y-auto bg-transparent px-4 pt-3.5 pb-2 text-[15px] text-slate-900 outline-none placeholder:text-slate-400 dark:text-slate-100 dark:placeholder:text-slate-500"
            />

            <div className="flex items-center justify-between gap-3 border-t border-slate-100 px-3 py-2 dark:border-slate-800/80">
              <div className="flex min-w-0 items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
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
                  onClick={onOpenModels}
                  className="max-w-[200px] truncate font-medium text-slate-700 hover:text-emerald-800 dark:text-slate-300 dark:hover:text-emerald-400 transition-colors"
                >
                  {isBootstrapUnbound
                    ? 'Select Model'
                    : (engineState.model?.displayName ?? 'Loading model...')}
                </button>
                <ComposerContextMeter
                  engineState={engineState}
                  modelReady={modelReady}
                  onCountTokens={onCountTokens}
                />
                <span aria-hidden="true">·</span>
                <button
                  type="button"
                  onClick={onToggleInspector}
                  className="inline-flex shrink-0 items-center gap-1 whitespace-nowrap hover:text-slate-900 dark:hover:text-slate-200 transition-colors"
                >
                  <SlidersHorizontal className="h-3 w-3" />
                  <span className="font-mono tabular-nums">
                    temp {engineState.sampling.temperature.toFixed(1)}
                  </span>
                </button>
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
                    disabled={!modelReady}
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
