import { useEffect, type FC } from 'react';
import {
  ActionBarPrimitive,
  AuiIf,
  BranchPickerPrimitive,
  ComposerPrimitive,
  ErrorPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useAuiState,
  type ThreadMessage,
} from '@assistant-ui/react';
import { MarkdownTextPrimitive } from '@assistant-ui/react-markdown';
import remarkGfm from 'remark-gfm';
import {
  ArrowDown,
  ArrowUp,
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  Download,
  Pencil,
  RefreshCw,
  SlidersHorizontal,
  Square,
} from 'lucide-react';
import { detectChatTemplate } from '../engine/assistantAdapter';
import { RECOMMENDED_INSTRUCT_MODEL } from '../engine/modelCatalog';
import type { EngineState } from '../engine/workerClient';

interface ThreadViewProps {
  engineState: EngineState;
  onOpenModels: () => void;
  onDownloadRecommended: () => void;
  onCancelDownload: () => void;
  onToggleInspector: () => void;
  onSpeculativeDraftChange?: (messages: readonly ThreadMessage[], draftText: string) => void;
}

const SpeculativePrefillDriver: FC<{
  modelId: string | null;
  modelReady: boolean;
  promptMode: EngineState['promptMode'];
  onSpeculativeDraftChange?: (messages: readonly ThreadMessage[], draftText: string) => void;
}> = ({ modelId, modelReady, promptMode, onSpeculativeDraftChange }) => {
  const composerText = useAuiState((s) => s.composer.text);
  const messages = useAuiState((s) => s.thread.messages);
  const isRunning = useAuiState((s) => s.thread.isRunning);

  useEffect(() => {
    if (!modelReady || !modelId || isRunning || !onSpeculativeDraftChange) return;
    onSpeculativeDraftChange(messages, composerText);
  }, [modelId, modelReady, promptMode, isRunning, messages, composerText, onSpeculativeDraftChange]);

  return null;
};

const INSTRUCT_STARTER_PROMPTS = [
  {
    title: 'Explain an engineering concept',
    description: 'Clear summary formatted via native ChatML',
    prompt: 'Explain how rotary position embeddings (RoPE) work in transformers in 3 concise bullet points.',
  },
  {
    title: 'Write a Python utility',
    description: 'Generate clean code locally in Web Worker',
    prompt: 'Write a clean Python function to check whether a string is a palindrome, ignoring case and spaces.',
  },
  {
    title: 'Quick factual Q&A',
    description: 'Verify instruction-following accuracy on vTensor',
    prompt: 'What is the capital of France, and what river runs through it? Answer in two short sentences.',
  },
  {
    title: 'Compare two data structures',
    description: 'Structured reasoning with KV-cache prefix reuse',
    prompt: 'Briefly compare a hash table and a binary search tree for lookup speed and ordering.',
  },
] as const;

const STORY_STARTER_PROMPTS = [
  {
    title: 'Continue a classic story',
    description: 'Run 5-layer GQA inference via vTensor',
    prompt: 'Once upon a time, a small silver bird found a glowing key in the garden',
  },
  {
    title: 'Test greedy determinism',
    description: 'Verify F64-accumulated RMS_NORM & RoPE',
    prompt: 'One sunny morning, Tom and his dog walked down the quiet forest path',
  },
  {
    title: 'Explore multi-turn context',
    description: 'Exercise vTensor KV-cache SET_ROWS writes',
    prompt: 'Lily built a wooden boat out of sticks and leaves because she wanted to sail',
  },
  {
    title: 'Short fable opening',
    description: 'Stream tokens live from Web Worker',
    prompt: 'Long ago in a quiet valley, a curious fox discovered an old clockmaker',
  },
] as const;

const MarkdownText: FC = () => (
  <div className="aui-md">
    <MarkdownTextPrimitive remarkPlugins={[remarkGfm]} />
  </div>
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

const AssistantMessage: FC = () => (
  <MessagePrimitive.Root className="group mx-auto w-full max-w-3xl px-4 py-3 sm:px-6">
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
        <span className="font-semibold text-slate-800 dark:text-slate-200">vTensor Assistant</span>
        <span aria-hidden="true">·</span>
        <span className="font-mono text-[11px]">llama.cpp / vTensor</span>
      </div>

      <div className="rounded-xl border border-slate-200/90 bg-white px-4 py-3.5 text-slate-800 dark:border-slate-800 dark:bg-[#121821] dark:text-slate-100">
        <MessagePrimitive.Parts
          components={{
            Text: MarkdownText,
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

export const ThreadView: FC<ThreadViewProps> = ({
  engineState,
  onOpenModels,
  onDownloadRecommended,
  onCancelDownload,
  onToggleInspector,
  onSpeculativeDraftChange,
}) => {
  const isDownloading = engineState.status === 'downloading';
  const isModelLoading = isDownloading || engineState.status === 'loading';
  const modelReady = engineState.model !== null && !isModelLoading;
  const detectedTemplate = detectChatTemplate(engineState.model);
  const isStoryModel = detectedTemplate === 'continuation';
  const starterPrompts = isStoryModel ? STORY_STARTER_PROMPTS : INSTRUCT_STARTER_PROMPTS;
  const recommendedCached = engineState.cachedModelIds.includes(RECOMMENDED_INSTRUCT_MODEL.id);

  return (
    <ThreadPrimitive.Root className="relative flex h-full flex-col overflow-hidden bg-[#F8F9FA] dark:bg-[#0B0F14]">
      <SpeculativePrefillDriver
        modelId={engineState.model?.id ?? null}
        modelReady={modelReady}
        promptMode={engineState.promptMode}
        onSpeculativeDraftChange={onSpeculativeDraftChange}
      />
      <ThreadPrimitive.Viewport className="flex flex-1 flex-col overflow-y-auto scroll-smooth">
        <div className="flex-1">
          <ThreadPrimitive.Empty>
            <div className="mx-auto flex max-w-3xl flex-col px-4 pt-10 pb-8 sm:px-6 sm:pt-14">
              <div className="mb-7">
                <p className="text-xs font-medium text-emerald-800 dark:text-emerald-400">
                  Client-side assistant-ui · llama.cpp · afflom/vTensor
                </p>
                <h1
                  className="mt-2 text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl dark:text-slate-50"
                  style={{ textWrap: 'balance' }}
                >
                  Local GGUF inference on the vTensor GGML provider
                </h1>
                <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-slate-600 dark:text-slate-400">
                  Every transformer graph executes inside your browser Web Worker using{' '}
                  <span className="font-medium text-slate-800 dark:text-slate-200">
                    Verdant Tensor
                  </span>{' '}
                  primitives through the vTensor GGML v3 backend contract, with default GGML CPU,
                  BLAS, and WebGPU providers disabled.
                </p>

                <div className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-500 dark:text-slate-400 tabular-nums">
                  <span>
                    Model:{' '}
                    <strong className="font-medium text-slate-800 dark:text-slate-200">
                      {engineState.model?.displayName ?? 'Loading stories260K.gguf'}
                    </strong>
                  </span>
                  <span aria-hidden="true">·</span>
                  <span>
                    Quantization:{' '}
                    <span className="font-mono">
                      {engineState.model?.hparams.primaryQuantization ?? 'F32'}
                    </span>
                  </span>
                  <span aria-hidden="true">·</span>
                  <span>
                    Template: <span className="font-mono">{detectedTemplate}</span>
                  </span>
                  <span aria-hidden="true">·</span>
                  <span>
                    Provider: <span className="font-mono">vTensor (0 CPU fallbacks)</span>
                  </span>
                </div>
              </div>

              {isStoryModel && !isModelLoading && (
                <div className="mb-6 flex flex-col gap-3 rounded-xl border border-slate-200 bg-white p-4 sm:flex-row sm:items-center sm:justify-between dark:border-slate-800 dark:bg-[#121821]">
                  <div className="min-w-0">
                    <div className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                      Upgrade to {RECOMMENDED_INSTRUCT_MODEL.name}
                    </div>
                    <p className="mt-0.5 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
                      TinyLLaMA Stories 260K is bundled for instant boot. Download a full
                      instruction-tuned ChatML assistant ({RECOMMENDED_INSTRUCT_MODEL.sizeLabel})
                      or pick SmolLM2-360M / Qwen2.5-0.5B from Models.
                    </p>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <button
                      type="button"
                      onClick={onDownloadRecommended}
                      className="inline-flex h-9 items-center gap-1.5 rounded-lg bg-[#0F5132] px-3.5 text-xs font-medium text-white hover:bg-[#0B3D25] transition-colors whitespace-nowrap"
                    >
                      <Download className="h-3.5 w-3.5" />
                      <span>
                        {recommendedCached
                          ? 'Load SmolLM2 135M'
                          : `Download (${RECOMMENDED_INSTRUCT_MODEL.sizeLabel})`}
                      </span>
                    </button>
                    <button
                      type="button"
                      onClick={onOpenModels}
                      className="inline-flex h-9 items-center rounded-lg border border-slate-200 bg-white px-3 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-200 dark:hover:bg-slate-800 transition-colors whitespace-nowrap"
                    >
                      All Models
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
                    <div className="flex items-center gap-2 shrink-0">
                      <span className="font-mono tabular-nums">{engineState.progressPercent}%</span>
                      {isDownloading && (
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

              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                {starterPrompts.map((item) => (
                  <ThreadPrimitive.Suggestion
                    key={item.title}
                    prompt={item.prompt}
                    method="replace"
                    autoSend={modelReady}
                    className="flex flex-col items-start rounded-xl border border-slate-200 bg-white p-4 text-left hover:border-slate-300 hover:bg-slate-50/70 dark:border-slate-800 dark:bg-[#121821] dark:hover:border-slate-700 dark:hover:bg-[#161E29] transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-700"
                  >
                    <span className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                      {item.title}
                    </span>
                    <span className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                      {item.description}
                    </span>
                  </ThreadPrimitive.Suggestion>
                ))}
              </div>
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
            <ComposerPrimitive.Input
              aria-label="Message prompt"
              placeholder={
                modelReady
                  ? isStoryModel
                    ? 'Write a story opening or switch to SmolLM2 / Qwen2.5 in Models...'
                    : 'Ask anything — runs locally on llama.cpp + vTensor...'
                  : 'Loading GGUF weights into vTensor buffers...'
              }
              rows={2}
              autoFocus
              className="max-h-40 w-full resize-none bg-transparent px-4 pt-3.5 pb-2 text-[15px] text-slate-900 outline-none placeholder:text-slate-400 dark:text-slate-100 dark:placeholder:text-slate-500"
            />

            <div className="flex items-center justify-between gap-3 border-t border-slate-100 px-3 py-2 dark:border-slate-800/80">
              <div className="flex min-w-0 items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                <button
                  type="button"
                  onClick={onOpenModels}
                  className="truncate max-w-[200px] font-medium text-slate-700 hover:text-emerald-800 dark:text-slate-300 dark:hover:text-emerald-400 transition-colors"
                >
                  {engineState.model?.displayName ?? 'Loading GGUF...'}
                </button>
                <span aria-hidden="true">·</span>
                <button
                  type="button"
                  onClick={onToggleInspector}
                  className="inline-flex items-center gap-1 whitespace-nowrap shrink-0 hover:text-slate-900 dark:hover:text-slate-200 transition-colors"
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
                      prefill {engineState.prefillProgress.completedTokens}/
                      {engineState.prefillProgress.totalTokens}
                    </span>
                  </>
                ) : engineState.speculativePrefill.active ? (
                  <>
                    <span className="hidden sm:inline" aria-hidden="true">
                      ·
                    </span>
                    <span className="hidden truncate font-mono text-[11px] tabular-nums text-emerald-800 dark:text-emerald-400 sm:inline">
                      warming {engineState.speculativePrefill.readyTokens}/
                      {engineState.speculativePrefill.targetTokens} tok
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
                    {engineState.speculativePrefill.readyTokens > 0 && (
                      <>
                        <span className="hidden md:inline" aria-hidden="true">
                          ·
                        </span>
                        <span className="hidden font-mono text-[11px] tabular-nums text-emerald-800 dark:text-emerald-400 md:inline">
                          warmed {engineState.speculativePrefill.readyTokens} tok
                        </span>
                      </>
                    )}
                  </>
                ) : (
                  engineState.speculativePrefill.readyTokens > 0 && (
                    <>
                      <span className="hidden sm:inline" aria-hidden="true">
                        ·
                      </span>
                      <span className="hidden font-mono text-[11px] tabular-nums text-emerald-800 dark:text-emerald-400 sm:inline">
                        warmed {engineState.speculativePrefill.readyTokens} tok
                      </span>
                    </>
                  )
                )}
              </div>

              <div className="flex items-center gap-2 shrink-0">
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
  );
};
