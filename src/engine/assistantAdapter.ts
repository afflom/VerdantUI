/**
 * Official `@assistant-ui/react` `ChatModelAdapter` connecting `useLocalRuntime`
 * directly to the client-side Verdant inference worker.
 *
 * Strictly honors Assistant-UI's `ModelContext` (`context.system`, `context.callSettings`,
 * `context.config`, `context.tools`) and `LocalThreadRuntimeCore` multi-step tool loop:
 * - Passes `unstable_threadId` through to the worker so switching threads never conflates KV state.
 * - Orchestrates native Qwen3 / thinking-model control prefixes (`<think>\n` for Thinking mode,
 *   `<think>\n\n</think>\n\n` for Fast mode) so thinking models never get confused.
 * - Parses reasoning traces (`<think>...</think>`, `<thought>...</thought>`, including prefilled
 *   `startsInReasoning: true`) into official Assistant-UI `ReasoningMessagePart` objects.
 * - Parses `<tool_call>...</tool_call>` blocks into official Assistant-UI `ToolCallMessagePart`
 *   objects and yields `status: { type: 'requires-action', reason: 'tool-calls' }` so Assistant-UI's
 *   `ToolInvocationTracker` executes client-side tools and resumes the next synthesis turn automatically.
 * - Strips reasoning traces from prior assistant turns and includes text file attachments.
 */

import type {
  ChatModelAdapter,
  ReasoningMessagePart,
  TextMessagePart,
  ThreadMessage,
  ToolCallMessagePart,
} from '@assistant-ui/react';
import type { Tool } from 'assistant-stream';
import type { ReadonlyJSONObject } from 'assistant-stream/utils';
import { applyCallSettingsToSampling } from './harnessConfig';
import type { LoadedModelSummary } from './llamaRuntime';
import { estimateTextTokens } from './cognitiveContext';
import {
  formatDirectOperatorSummary,
  UNIFIED_VERDANT_SYSTEM_PROMPT,
  type VerdantThinkingPolicy,
} from './cognitiveObserver';
import type { ObserverStreamBus } from './observerStreamBus';
import type { PublishedPrismSummary } from './prismStudio';
import type { LlamaVTensorWorkerClient, PromptFormatMode } from './workerClient';

export type DetectedChatTemplate =
  | 'chatml'
  | 'llama3'
  | 'gemma'
  | 'phi3'
  | 'zephyr'
  | 'llama2'
  | 'continuation'
  | 'plain';

export interface FormatPromptOptions {
  /** Controls Qwen3/thinking model prefix (`'on'` -> `<think>\n`, `'off'` -> `<think>\n\n</think>\n\n`). */
  readonly thinkingMode?: VerdantThinkingPolicy;
  /** Optional Assistant-UI `ModelContext.tools` schemas serialized into `<tools>...</tools>`. */
  readonly tools?: Record<string, Tool<any, any>>;
  /** Optional in-progress assistant message from `unstable_getMessage()` carrying settled tool results for Step 2+. */
  readonly pendingAssistantMessage?: ThreadMessage | null;
  /** Optional frozen `LOD1` Lossless Episodic Index appended cleanly to the epoch system prefix. */
  readonly epochManifestBlock?: string | null;
  /** Optional foveated `LOD3` bit-exact recalled turns placed at the active tail (preserves KV prefix!). */
  readonly foveatedRecallBlock?: string | null;
  /**
   * When true on thinking-capable ChatML models (Qwen3), preserves `<think>\n\n</think>\n\n`
   * ahead of prior non-reasoning assistant replies so Turn `k+1` matches Turn `k`'s live KV
   * tokens bit-for-bit (eliminating 100% assistant-reply KV cache invalidation!).
   */
  readonly preserveEmptyThinkPrefix?: boolean;
}

export interface ParseAssistantStreamOptions {
  /** True when the prompt itself ended with `<think>\n` so generation begins inside a reasoning block. */
  readonly startsInReasoning?: boolean;
  /** True when Agentic tool calling is enabled and `<tool_call>...</tool_call>` should be parsed into ToolCallMessageParts. */
  readonly parseToolCalls?: boolean;
  /** Stable prefix used to generate deterministic `toolCallId`s within a single roundtrip. */
  readonly idPrefix?: string;
}

export type ParsedAssistantContentPart =
  | ReasoningMessagePart
  | TextMessagePart
  | ToolCallMessagePart;

const REASONING_TAG_PAIRS: readonly { readonly open: string; readonly close: string }[] = [
  { open: '<think>', close: '</think>' },
  { open: '<thought>', close: '</thought>' },
  { open: '<|thought|>', close: '<|/thought|>' },
];

const TOOL_CALL_OPEN = '<tool_call>';
const TOOL_CALL_CLOSE = '</tool_call>';
const TOOL_RESPONSE_OPEN = '<tool_response>';

export function detectChatTemplate(
  model: Pick<LoadedModelSummary, 'id' | 'hparams'> | null | undefined,
): DetectedChatTemplate {
  if (!model) return 'plain';
  const tmpl = model.hparams.chatTemplate ?? '';
  if (tmpl.includes('<|im_start|>')) return 'chatml';
  if (tmpl.includes('<|start_header_id|>')) return 'llama3';
  if (tmpl.includes('<start_of_turn>')) return 'gemma';
  if (tmpl.includes('<|user|>') && tmpl.includes('<|end|>')) return 'phi3';
  if (tmpl.includes('<|assistant|>')) return 'zephyr';
  if (tmpl.includes('[INST]')) return 'llama2';
  if (
    model.id.toLowerCase().includes('stories') ||
    model.hparams.name.toLowerCase().includes('stories')
  ) {
    return 'continuation';
  }
  return 'plain';
}

/**
 * Detects whether the loaded model natively uses `<think>...</think>` hybrid reasoning control
 * (e.g., Qwen3 dense 4B/8B, QwQ, DeepSeek-R1) rather than pure non-thinking instruct models.
 */
export function isThinkingCapableModel(
  modelOrId?: Pick<LoadedModelSummary, 'id' | 'hparams'> | string | null,
): boolean {
  if (!modelOrId) return false;
  if (typeof modelOrId === 'string') {
    const lower = modelOrId.toLowerCase();
    if (lower.includes('coder')) return false;
    return (
      lower.includes('qwen3') ||
      lower.includes('qwq') ||
      lower.includes('deepseek-r1') ||
      lower.includes('thinking')
    );
  }
  const tmpl = (modelOrId.hparams.chatTemplate ?? '').toLowerCase();
  if (tmpl.includes('<think>')) return true;
  const hay = `${modelOrId.id} ${modelOrId.hparams.name}`.toLowerCase();
  if (hay.includes('coder')) return false;
  return (
    hay.includes('qwen3') ||
    hay.includes('qwq') ||
    hay.includes('deepseek-r1') ||
    hay.includes('thinking')
  );
}

export function resolveEffectiveTemplate(
  modelOrId?: Pick<LoadedModelSummary, 'id' | 'hparams'> | string,
  mode: PromptFormatMode = 'auto',
): DetectedChatTemplate {
  if (mode === 'continuation') return 'continuation';
  if (mode === 'chatml') return 'chatml';
  if (mode === 'instruct') return 'plain';
  if (typeof modelOrId === 'string') {
    return modelOrId.toLowerCase().includes('stories') ? 'continuation' : 'plain';
  }
  return detectChatTemplate(modelOrId);
}

/** Returns true when `formatMessagesForLlama` prefills `<think>\n` for this model & thinking policy. */
export function shouldPrefillOpenThink(
  modelOrId?: Pick<LoadedModelSummary, 'id' | 'hparams'> | string | null,
  mode: PromptFormatMode = 'auto',
  thinkingMode: VerdantThinkingPolicy = 'auto',
): boolean {
  if (thinkingMode !== 'on') return false;
  const tmpl = resolveEffectiveTemplate(modelOrId ?? undefined, mode);
  return tmpl === 'chatml' && isThinkingCapableModel(modelOrId);
}

/** Returns true when `formatMessagesForLlama` prefills `<think>\n\n</think>\n\n` to disable monologue in Fast mode. */
export function shouldPrefillEmptyThink(
  modelOrId?: Pick<LoadedModelSummary, 'id' | 'hparams'> | string | null,
  mode: PromptFormatMode = 'auto',
  thinkingMode: VerdantThinkingPolicy = 'auto',
): boolean {
  if (thinkingMode !== 'off') return false;
  const tmpl = resolveEffectiveTemplate(modelOrId ?? undefined, mode);
  return tmpl === 'chatml' && isThinkingCapableModel(modelOrId);
}

/**
 * Strips completed or unclosed reasoning tags from stored assistant text so reasoning monologues
 * from previous turns (or legacy persisted threads) are never re-fed as conversational context.
 */
export function stripReasoningTags(text: string): string {
  let cleaned = text;
  for (const { open, close } of REASONING_TAG_PAIRS) {
    const escapedOpen = open.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const escapedClose = close.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    cleaned = cleaned.replace(new RegExp(`${escapedOpen}[\\s\\S]*?${escapedClose}`, 'gi'), '');
    cleaned = cleaned.replace(new RegExp(`${escapedOpen}[\\s\\S]*$`, 'gi'), '');
  }
  return cleaned.trim();
}

/**
 * Serializes Assistant-UI `ModelContext.tools` into standard Hermes/Qwen3 `<tools>` JSON schemas.
 */
export function formatToolDeclarations(tools?: Record<string, Tool<any, any>>): string {
  if (!tools) return '';
  const entries = Object.entries(tools).filter(([, t]) => !t.disabled);
  if (entries.length === 0) return '';
  const specs = entries.map(([name, t]) => ({
    type: 'function',
    function: {
      name,
      description: t.description ?? `Execute ${name}`,
      parameters:
        t.parameters && typeof t.parameters === 'object'
          ? t.parameters
          : { type: 'object', properties: {} },
    },
  }));
  return `<tools>\n${specs.map((s) => JSON.stringify(s)).join('\n')}\n</tools>`;
}

/** Returns true once raw output contains a completed `</tool_call>` block so generation can stop cleanly. */
export function hasCompleteToolCallTag(rawText: string): boolean {
  const lower = rawText.toLowerCase();
  const openIdx = lower.indexOf(TOOL_CALL_OPEN);
  if (openIdx === -1) return false;
  const closeIdx = lower.indexOf(TOOL_CALL_CLOSE, openIdx + TOOL_CALL_OPEN.length);
  return closeIdx !== -1;
}

/**
 * Computes the length of the longest suffix of `text` that is a strict prefix of any
 * tag in `tags` (case-insensitive), so streaming chunks never flash partial `<thi` or `</tool_`
 * fragments before the tag completes.
 */
function trailingPartialTagLength(text: string, tags: readonly string[]): number {
  const lower = text.toLowerCase();
  let maxLen = 0;
  for (const tag of tags) {
    const limit = Math.min(tag.length - 1, lower.length);
    for (let len = limit; len > maxLen; len--) {
      if (lower.endsWith(tag.slice(0, len))) {
        maxLen = len;
        break;
      }
    }
  }
  return maxLen;
}

function parseToolCallJson(
  rawInner: string,
  toolCallId: string,
): ToolCallMessagePart | null {
  const cleaned = rawInner
    .trim()
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/i, '')
    .trim();
  if (!cleaned) return null;
  try {
    const parsed = JSON.parse(cleaned) as {
      name?: unknown;
      arguments?: unknown;
      args?: unknown;
      parameters?: unknown;
    };
    if (!parsed || typeof parsed !== 'object' || typeof parsed.name !== 'string') {
      return null;
    }
    const toolName = parsed.name.trim();
    if (!toolName) return null;
    const rawArgs = parsed.arguments ?? parsed.args ?? parsed.parameters ?? {};
    let argsObj: ReadonlyJSONObject;
    if (typeof rawArgs === 'string') {
      try {
        const reparsed = JSON.parse(rawArgs);
        argsObj =
          reparsed && typeof reparsed === 'object' && !Array.isArray(reparsed)
            ? (reparsed as ReadonlyJSONObject)
            : {};
      } catch {
        argsObj = {};
      }
    } else if (rawArgs && typeof rawArgs === 'object' && !Array.isArray(rawArgs)) {
      argsObj = rawArgs as ReadonlyJSONObject;
    } else {
      argsObj = {};
    }
    return {
      type: 'tool-call',
      toolCallId,
      toolName,
      args: argsObj,
      argsText: JSON.stringify(argsObj),
    };
  } catch {
    return null;
  }
}

/**
 * Incrementally parses raw streamed model output into official Assistant-UI content parts:
 * - `{ type: 'reasoning', text }` for `<think>...</think>` / `<thought>...</thought>` blocks
 * - `{ type: 'tool-call', toolCallId, toolName, args, argsText }` for `<tool_call>...</tool_call>` blocks
 * - `{ type: 'text', text }` for visible answer text
 *
 * Supports `startsInReasoning: true` (when the prompt itself prefilled `<think>\n`) and holds back
 * incomplete tag tails during streaming.
 */
export function parseAssistantStreamParts(
  rawText: string,
  isComplete = false,
  options?: ParseAssistantStreamOptions,
): readonly ParsedAssistantContentPart[] {
  if (!rawText) return [];

  const parseTools = Boolean(options?.parseToolCalls);
  const idPrefix = options?.idPrefix ?? 'tc';

  // Truncate any hallucinated `<tool_response>` if the model tried to fabricate a tool output
  let workingText = rawText;
  if (parseTools) {
    const respIdx = workingText.toLowerCase().indexOf(TOOL_RESPONSE_OPEN);
    if (respIdx !== -1) {
      workingText = workingText.slice(0, respIdx);
    }
  }
  if (!workingText) return [];

  const openTags = parseTools
    ? [...REASONING_TAG_PAIRS.map((p) => p.open), TOOL_CALL_OPEN, TOOL_RESPONSE_OPEN]
    : REASONING_TAG_PAIRS.map((p) => p.open);
  const parts: ParsedAssistantContentPart[] = [];
  let cursor = 0;
  let activeMode: 'text' | 'reasoning' | 'tool-call' = options?.startsInReasoning ? 'reasoning' : 'text';
  let activeCloseTag: string | null = options?.startsInReasoning ? '</think>' : null;
  let afterBlock = false;
  let toolSeq = 0;

  const pushTextOrReasoning = (type: 'reasoning' | 'text', slice: string) => {
    const normalized =
      type === 'reasoning'
        ? slice.trim()
        : afterBlock
          ? slice.replace(/^\s+/, '')
          : slice;
    if (!normalized) return;
    if (type === 'text') {
      afterBlock = false;
    }
    const prev = parts[parts.length - 1];
    if (prev && prev.type === type) {
      parts[parts.length - 1] = {
        type,
        text: type === 'reasoning' ? `${prev.text}\n\n${normalized}` : `${prev.text}${normalized}`,
      };
    } else {
      parts.push({ type, text: normalized });
    }
  };

  const lower = workingText.toLowerCase();

  while (cursor < workingText.length) {
    if (activeMode === 'text') {
      let bestIdx = -1;
      let bestKind: 'reasoning' | 'tool-call' = 'reasoning';
      let bestOpenLen = 0;
      let bestClose = '';

      for (const pair of REASONING_TAG_PAIRS) {
        const idx = lower.indexOf(pair.open, cursor);
        if (idx !== -1 && (bestIdx === -1 || idx < bestIdx)) {
          bestIdx = idx;
          bestKind = 'reasoning';
          bestOpenLen = pair.open.length;
          bestClose = pair.close;
        }
      }

      if (parseTools) {
        const tcIdx = lower.indexOf(TOOL_CALL_OPEN, cursor);
        if (tcIdx !== -1 && (bestIdx === -1 || tcIdx < bestIdx)) {
          bestIdx = tcIdx;
          bestKind = 'tool-call';
          bestOpenLen = TOOL_CALL_OPEN.length;
          bestClose = TOOL_CALL_CLOSE;
        }
      }

      if (bestIdx === -1) {
        const remaining = workingText.slice(cursor);
        const holdback = isComplete ? 0 : trailingPartialTagLength(remaining, openTags);
        const safeText = holdback > 0 ? remaining.slice(0, remaining.length - holdback) : remaining;
        pushTextOrReasoning('text', safeText);
        break;
      }

      if (bestIdx > cursor) {
        pushTextOrReasoning('text', workingText.slice(cursor, bestIdx));
      }
      cursor = bestIdx + bestOpenLen;
      activeMode = bestKind;
      activeCloseTag = bestClose;
    } else if (activeMode === 'reasoning') {
      const closeTag = activeCloseTag!;
      const closeIdx = lower.indexOf(closeTag, cursor);
      if (closeIdx === -1) {
        const remaining = workingText.slice(cursor);
        const holdback = isComplete ? 0 : trailingPartialTagLength(remaining, [closeTag]);
        const safeReasoning =
          holdback > 0 ? remaining.slice(0, remaining.length - holdback) : remaining;
        if (safeReasoning.trim().length > 0) {
          pushTextOrReasoning('reasoning', safeReasoning);
        }
        break;
      }

      const reasoningSlice = workingText.slice(cursor, closeIdx);
      if (reasoningSlice.trim().length > 0) {
        pushTextOrReasoning('reasoning', reasoningSlice);
      }
      cursor = closeIdx + closeTag.length;
      activeMode = 'text';
      activeCloseTag = null;
      afterBlock = true;
    } else {
      // activeMode === 'tool-call'
      const closeIdx = lower.indexOf(TOOL_CALL_CLOSE, cursor);
      if (closeIdx === -1) {
        // Hold back incomplete `<tool_call>` JSON until `</tool_call>` arrives (or emit raw text if truncated on complete)
        if (isComplete) {
          const fallbackParsed = parseToolCallJson(workingText.slice(cursor), `${idPrefix}_${toolSeq}`);
          if (fallbackParsed) {
            toolSeq += 1;
            parts.push(fallbackParsed);
          }
        }
        break;
      }

      const inner = workingText.slice(cursor, closeIdx);
      const parsedCall = parseToolCallJson(inner, `${idPrefix}_${toolSeq}`);
      if (parsedCall) {
        toolSeq += 1;
        parts.push(parsedCall);
      }
      cursor = closeIdx + TOOL_CALL_CLOSE.length;
      activeMode = 'text';
      activeCloseTag = null;
      afterBlock = true;
    }
  }

  return parts;
}

interface ExtractedTurn {
  readonly role: 'system' | 'user' | 'assistant' | 'tool';
  readonly text: string;
  /** True when this assistant turn was produced with no visible reasoning block (e.g. Fast/empty-think). */
  readonly emptyThinkAssistant?: boolean;
}

function serializeToolResultPayload(part: ToolCallMessagePart): string {
  const payload = part.isError
    ? { name: part.toolName, error: part.result ?? 'Tool execution failed.' }
    : { name: part.toolName, result: part.result ?? null };
  return `<tool_response>\n${JSON.stringify(payload)}\n</tool_response>`;
}

function extractTurns(
  messages: readonly ThreadMessage[],
  pendingAssistantMessage?: ThreadMessage | null,
): ExtractedTurn[] {
  const allMessages =
    pendingAssistantMessage &&
    pendingAssistantMessage.role === 'assistant' &&
    pendingAssistantMessage.content.length > 0 &&
    !messages.some((m) => m.id === pendingAssistantMessage.id)
      ? [...messages, pendingAssistantMessage]
      : messages;

  const turns: ExtractedTurn[] = [];

  for (const m of allMessages) {
    if (m.role === 'user' || m.role === 'system') {
      const contentChunks = (m.content as readonly { type?: string; text?: string }[])
        .filter(
          (part): part is { type: 'text'; text: string } =>
            part.type === 'text' && typeof part.text === 'string',
        )
        .map((part) => part.text);

      // Include text file attachments attached via Assistant-UI's SimpleTextAttachmentAdapter
      if (m.role === 'user' && Array.isArray(m.attachments)) {
        for (const att of m.attachments) {
          if (!Array.isArray(att.content)) continue;
          const attText = (att.content as readonly { type?: string; text?: string }[])
            .filter((p): p is { type: 'text'; text: string } => p.type === 'text' && typeof p.text === 'string')
            .map((p) => p.text)
            .join('\n')
            .trim();
          if (attText) {
            contentChunks.push(`[Attached file: ${att.name}]\n${attText}`);
          }
        }
      }

      const joined = contentChunks.join('\n').trim();
      if (joined) {
        turns.push({ role: m.role, text: joined });
      }
      continue;
    }

    if (m.role === 'assistant') {
      const hasReasoningPart = m.content.some(
        (p) => p.type === 'reasoning' && typeof (p as { text?: unknown }).text === 'string' && String((p as { text?: string }).text ?? '').trim().length > 0,
      );
      // Preserve sequential assistant text / tool-call / tool-response interleaving across multi-step agent runs
      let pendingAssistantChunks: string[] = [];
      let pendingToolResponses: string[] = [];
      let firstAssistantChunk = true;

      const flushAssistantAndTools = () => {
        const aText = pendingAssistantChunks.join('\n').trim();
        if (aText) {
          turns.push({
            role: 'assistant',
            text: aText,
            emptyThinkAssistant: firstAssistantChunk && !hasReasoningPart,
          });
          firstAssistantChunk = false;
        }
        pendingAssistantChunks = [];
        if (pendingToolResponses.length > 0) {
          turns.push({ role: 'tool', text: pendingToolResponses.join('\n') });
          pendingToolResponses = [];
        }
      };

      for (const part of m.content) {
        if (part.type === 'text') {
          const clean = stripReasoningTags(part.text);
          if (clean) {
            if (pendingToolResponses.length > 0) {
              flushAssistantAndTools();
            }
            pendingAssistantChunks.push(clean);
          }
        } else if (part.type === 'tool-call') {
          const callJson = JSON.stringify({ name: part.toolName, arguments: part.args ?? {} });
          pendingAssistantChunks.push(`${TOOL_CALL_OPEN}\n${callJson}\n${TOOL_CALL_CLOSE}`);
          if (part.result !== undefined) {
            pendingToolResponses.push(serializeToolResultPayload(part));
          }
        }
      }
      flushAssistantAndTools();
    }
  }

  return turns;
}

function buildEffectiveSystemPrompt(
  explicitSystem: string,
  tools?: Record<string, Tool<any, any>>,
  epochManifestBlock?: string | null,
): string {
  const parts: string[] = [];
  if (explicitSystem) parts.push(explicitSystem);
  const toolBlock = formatToolDeclarations(tools);
  if (toolBlock) parts.push(toolBlock);
  if (epochManifestBlock?.trim()) parts.push(epochManifestBlock.trim());
  return parts.join('\n\n');
}

function attachFoveatedRecallToTail(
  turns: ExtractedTurn[],
  foveatedRecallBlock?: string | null,
): ExtractedTurn[] {
  const recall = foveatedRecallBlock?.trim();
  if (!recall) return turns;
  for (let i = turns.length - 1; i >= 0; i--) {
    if (turns[i].role === 'user') {
      const next = [...turns];
      next[i] = { ...next[i], text: `${recall}\n\n${next[i].text}` };
      return next;
    }
  }
  return turns;
}

/**
 * Formats assistant-ui ThreadMessage history into the exact prompt format expected
 * by the loaded GGUF model (ChatML, Llama-3, Gemma, Phi-3, Zephyr, Llama-2, or continuation).
 *
 * Strictly emits a `system` block ONLY when a non-empty `systemPrompt` or `options.tools` is
 * supplied via Assistant-UI's `ModelContext` (or already present as a `role: 'system'` message).
 */
export function formatMessagesForLlama(
  messages: readonly ThreadMessage[],
  modelOrId?: Pick<LoadedModelSummary, 'id' | 'hparams'> | string,
  mode: PromptFormatMode = 'auto',
  systemPrompt?: string,
  options?: FormatPromptOptions,
): string {
  const explicitSystem = buildEffectiveSystemPrompt(
    systemPrompt?.trim() || '',
    options?.tools,
    options?.epochManifestBlock,
  );
  const turns = attachFoveatedRecallToTail(
    extractTurns(messages, options?.pendingAssistantMessage),
    options?.foveatedRecallBlock,
  );
  const effectiveTemplate = resolveEffectiveTemplate(modelOrId, mode);
  const thinkingMode = options?.thinkingMode ?? 'auto';
  const keepEmptyThink =
    Boolean(options?.preserveEmptyThinkPrefix) &&
    effectiveTemplate === 'chatml' &&
    isThinkingCapableModel(modelOrId);

  if (effectiveTemplate === 'continuation') {
    const parts = turns.map((t) => t.text);
    if (parts.length === 0) return 'Once upon a time';
    if (parts.length === 1) return parts[0];
    return parts.join(' ');
  }

  if (effectiveTemplate === 'chatml') {
    const hasSystem = turns.some((t) => t.role === 'system');
    const chunks: string[] = [];
    if (!hasSystem && explicitSystem) {
      chunks.push(`<|im_start|>system\n${explicitSystem}<|im_end|>\n`);
    }
    for (const t of turns) {
      const chatRole = t.role === 'tool' ? 'user' : t.role;
      const body =
        keepEmptyThink && t.role === 'assistant' && t.emptyThinkAssistant
          ? `<think>\n\n</think>\n\n${t.text}`
          : t.text;
      chunks.push(`<|im_start|>${chatRole}\n${body}<|im_end|>\n`);
    }
    if (shouldPrefillOpenThink(modelOrId, mode, thinkingMode)) {
      chunks.push('<|im_start|>assistant\n<think>\n');
    } else if (shouldPrefillEmptyThink(modelOrId, mode, thinkingMode)) {
      chunks.push('<|im_start|>assistant\n<think>\n\n</think>\n\n');
    } else {
      chunks.push('<|im_start|>assistant\n');
    }
    return chunks.join('');
  }

  if (effectiveTemplate === 'llama3') {
    const hasSystem = turns.some((t) => t.role === 'system');
    const chunks: string[] = ['<|begin_of_text|>'];
    if (!hasSystem && explicitSystem) {
      chunks.push(`<|start_header_id|>system<|end_header_id|>\n\n${explicitSystem}<|eot_id|>`);
    }
    for (const t of turns) {
      const role = t.role === 'tool' ? 'ipython' : t.role;
      chunks.push(`<|start_header_id|>${role}<|end_header_id|>\n\n${t.text}<|eot_id|>`);
    }
    chunks.push('<|start_header_id|>assistant<|end_header_id|>\n\n');
    return chunks.join('');
  }

  if (effectiveTemplate === 'gemma') {
    const chunks: string[] = [];
    let pendingSystem = !turns.some((t) => t.role === 'system') ? explicitSystem : '';
    for (const t of turns) {
      if (t.role === 'system') {
        pendingSystem = pendingSystem ? `${pendingSystem}\n\n${t.text}` : t.text;
        continue;
      }
      const role = t.role === 'assistant' ? 'model' : 'user';
      const body = role === 'user' && pendingSystem ? `${pendingSystem}\n\n${t.text}` : t.text;
      if (role === 'user') pendingSystem = '';
      chunks.push(`<start_of_turn>${role}\n${body}<end_of_turn>\n`);
    }
    chunks.push('<start_of_turn>model\n');
    return chunks.join('');
  }

  if (effectiveTemplate === 'phi3') {
    const hasSystem = turns.some((t) => t.role === 'system');
    const chunks: string[] = [];
    if (!hasSystem && explicitSystem) {
      chunks.push(`<|system|>\n${explicitSystem}<|end|>\n`);
    }
    for (const t of turns) {
      const role = t.role === 'tool' ? 'user' : t.role;
      chunks.push(`<|${role}|>\n${t.text}<|end|>\n`);
    }
    chunks.push('<|assistant|>\n');
    return chunks.join('');
  }

  if (effectiveTemplate === 'zephyr') {
    const hasSystem = turns.some((t) => t.role === 'system');
    const chunks: string[] = [];
    if (!hasSystem && explicitSystem) {
      chunks.push(`<|system|>\n${explicitSystem}</s>\n`);
    }
    for (const t of turns) {
      const role = t.role === 'tool' ? 'user' : t.role;
      chunks.push(`<|${role}|>\n${t.text}</s>\n`);
    }
    chunks.push('<|assistant|>\n');
    return chunks.join('');
  }

  if (effectiveTemplate === 'llama2') {
    const chunks: string[] = [];
    let pendingSystem = !turns.some((t) => t.role === 'system') ? explicitSystem : '';
    for (const t of turns) {
      if (t.role === 'system') {
        pendingSystem = pendingSystem ? `${pendingSystem}\n\n${t.text}` : t.text;
        continue;
      }
      if (t.role === 'user' || t.role === 'tool') {
        const body = pendingSystem ? `<<SYS>>\n${pendingSystem}\n<</SYS>>\n\n${t.text}` : t.text;
        pendingSystem = '';
        chunks.push(`[INST] ${body} [/INST]`);
      } else if (t.role === 'assistant') {
        chunks.push(` ${t.text} </s>`);
      }
    }
    return chunks.join('');
  }

  // Standard multi-turn instruction prompt fallback
  const lines: string[] = [];
  const hasSystem = turns.some((t) => t.role === 'system');
  if (!hasSystem && explicitSystem) {
    lines.push(`System: ${explicitSystem}`);
  }
  for (const t of turns) {
    if (t.role === 'system') {
      lines.push(`System: ${t.text}`);
    } else if (t.role === 'user' || t.role === 'tool') {
      lines.push(`User: ${t.text}`);
    } else if (t.role === 'assistant') {
      lines.push(`Assistant: ${t.text}`);
    }
  }
  lines.push('Assistant:');
  return lines.join('\n');
}

/**
 * Formats the exact open prefix for Continuous Speculative Prefill while the user is typing
 * (or at boot when `draftPrefix === ''`).
 *
 * Invariant: for every chat template and any `fullText` whose `.trim()` starts with
 * `draftPrefix.trimStart()`, `formatMessagesForLlama([...messages, user(fullText)], model, mode, systemPrompt, options)`
 * strictly `startsWith(formatSpeculativePrefixForLlama(messages, draftPrefix, model, mode, systemPrompt, options))`!
 */
export function formatSpeculativePrefixForLlama(
  messages: readonly ThreadMessage[],
  draftPrefix: string,
  modelOrId?: Pick<LoadedModelSummary, 'id' | 'hparams'> | string,
  mode: PromptFormatMode = 'auto',
  systemPrompt?: string,
  options?: Pick<FormatPromptOptions, 'tools' | 'epochManifestBlock' | 'preserveEmptyThinkPrefix'>,
): string {
  const explicitSystem = buildEffectiveSystemPrompt(
    systemPrompt?.trim() || '',
    options?.tools,
    options?.epochManifestBlock,
  );
  const turns = extractTurns(messages);
  const effectiveTemplate = resolveEffectiveTemplate(modelOrId, mode);
  const openDraft = draftPrefix.trimStart();
  const keepEmptyThink =
    Boolean(options?.preserveEmptyThinkPrefix) &&
    effectiveTemplate === 'chatml' &&
    isThinkingCapableModel(modelOrId);

  if (effectiveTemplate === 'continuation') {
    if (turns.length === 0) return openDraft;
    const prev = turns.map((t) => t.text).join(' ');
    return openDraft ? `${prev} ${openDraft}` : `${prev} `;
  }

  if (effectiveTemplate === 'chatml') {
    const hasSystem = turns.some((t) => t.role === 'system');
    const chunks: string[] = [];
    if (!hasSystem && explicitSystem) {
      chunks.push(`<|im_start|>system\n${explicitSystem}<|im_end|>\n`);
    }
    for (const t of turns) {
      const chatRole = t.role === 'tool' ? 'user' : t.role;
      const body =
        keepEmptyThink && t.role === 'assistant' && t.emptyThinkAssistant
          ? `<think>\n\n</think>\n\n${t.text}`
          : t.text;
      chunks.push(`<|im_start|>${chatRole}\n${body}<|im_end|>\n`);
    }
    chunks.push(`<|im_start|>user\n${openDraft}`);
    return chunks.join('');
  }

  if (effectiveTemplate === 'llama3') {
    const hasSystem = turns.some((t) => t.role === 'system');
    const chunks: string[] = ['<|begin_of_text|>'];
    if (!hasSystem && explicitSystem) {
      chunks.push(`<|start_header_id|>system<|end_header_id|>\n\n${explicitSystem}<|eot_id|>`);
    }
    for (const t of turns) {
      const role = t.role === 'tool' ? 'ipython' : t.role;
      chunks.push(`<|start_header_id|>${role}<|end_header_id|>\n\n${t.text}<|eot_id|>`);
    }
    chunks.push(`<|start_header_id|>user<|end_header_id|>\n\n${openDraft}`);
    return chunks.join('');
  }

  if (effectiveTemplate === 'gemma') {
    const chunks: string[] = [];
    let pendingSystem = !turns.some((t) => t.role === 'system') ? explicitSystem : '';
    for (const t of turns) {
      if (t.role === 'system') {
        pendingSystem = pendingSystem ? `${pendingSystem}\n\n${t.text}` : t.text;
        continue;
      }
      const role = t.role === 'assistant' ? 'model' : 'user';
      const body = role === 'user' && pendingSystem ? `${pendingSystem}\n\n${t.text}` : t.text;
      if (role === 'user') pendingSystem = '';
      chunks.push(`<start_of_turn>${role}\n${body}<end_of_turn>\n`);
    }
    const draftBody = pendingSystem ? `${pendingSystem}\n\n${openDraft}` : openDraft;
    chunks.push(`<start_of_turn>user\n${draftBody}`);
    return chunks.join('');
  }

  if (effectiveTemplate === 'phi3') {
    const hasSystem = turns.some((t) => t.role === 'system');
    const chunks: string[] = [];
    if (!hasSystem && explicitSystem) {
      chunks.push(`<|system|>\n${explicitSystem}<|end|>\n`);
    }
    for (const t of turns) {
      const role = t.role === 'tool' ? 'user' : t.role;
      chunks.push(`<|${role}|>\n${t.text}<|end|>\n`);
    }
    chunks.push(`<|user|>\n${openDraft}`);
    return chunks.join('');
  }

  if (effectiveTemplate === 'zephyr') {
    const hasSystem = turns.some((t) => t.role === 'system');
    const chunks: string[] = [];
    if (!hasSystem && explicitSystem) {
      chunks.push(`<|system|>\n${explicitSystem}</s>\n`);
    }
    for (const t of turns) {
      const role = t.role === 'tool' ? 'user' : t.role;
      chunks.push(`<|${role}|>\n${t.text}</s>\n`);
    }
    chunks.push(`<|user|>\n${openDraft}`);
    return chunks.join('');
  }

  if (effectiveTemplate === 'llama2') {
    const chunks: string[] = [];
    let pendingSystem = !turns.some((t) => t.role === 'system') ? explicitSystem : '';
    for (const t of turns) {
      if (t.role === 'system') {
        pendingSystem = pendingSystem ? `${pendingSystem}\n\n${t.text}` : t.text;
        continue;
      }
      if (t.role === 'user' || t.role === 'tool') {
        const body = pendingSystem ? `<<SYS>>\n${pendingSystem}\n<</SYS>>\n\n${t.text}` : t.text;
        pendingSystem = '';
        chunks.push(`[INST] ${body} [/INST]`);
      } else if (t.role === 'assistant') {
        chunks.push(` ${t.text} </s>`);
      }
    }
    const draftBody = pendingSystem ? `<<SYS>>\n${pendingSystem}\n<</SYS>>\n\n${openDraft}` : openDraft;
    chunks.push(`[INST] ${draftBody}`);
    return chunks.join('');
  }

  // Standard multi-turn instruction prompt fallback
  const lines: string[] = [];
  const hasSystem = turns.some((t) => t.role === 'system');
  if (!hasSystem && explicitSystem) {
    lines.push(`System: ${explicitSystem}`);
  }
  for (const t of turns) {
    if (t.role === 'system') {
      lines.push(`System: ${t.text}`);
    } else if (t.role === 'user' || t.role === 'tool') {
      lines.push(`User: ${t.text}`);
    } else if (t.role === 'assistant') {
      lines.push(`Assistant: ${t.text}`);
    }
  }
  lines.push(`User: ${openDraft}`);
  return lines.join('\n');
}

function extractLatestUserInput(messages: readonly ThreadMessage[]): {
  text: string;
  hasAttachment: boolean;
} {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role !== 'user') continue;
    const text = (m.content as readonly { type?: string; text?: string }[])
      .filter(
        (p): p is { type: 'text'; text: string } =>
          p.type === 'text' && typeof p.text === 'string',
      )
      .map((p) => p.text)
      .join('\n')
      .trim();
    const hasAttachment = Array.isArray(m.attachments) && m.attachments.length > 0;
    return { text, hasAttachment };
  }
  return { text: '', hasAttachment: false };
}

function hasPendingToolResults(msg: ThreadMessage | null | undefined): boolean {
  if (!msg || msg.role !== 'assistant' || !Array.isArray(msg.content)) return false;
  return (msg.content as readonly { type?: string; result?: unknown }[]).some(
    (p) => p.type === 'tool-call' && p.result !== undefined,
  );
}

export interface StudioAdapterBridge {
  readonly bus: ObserverStreamBus;
  getPublishedPrisms(): readonly PublishedPrismSummary[];
  onOpenPrism?: (prismId: string) => void;
  /**
   * Resolves any model `<tool_call>` not satisfied by an already-registered tool:
   * reuses an existing Space-visible Published Prism or autonomously creates, publishes,
   * and executes a new procedural Prism in the active Space (`DEC-013`).
   */
  resolveOrCreatePrismForToolCall?: (
    toolName: string,
    args: Record<string, unknown>,
    meta: { readonly latestUserText: string; readonly signal?: AbortSignal },
  ) => Promise<{
    readonly __prismInvocation: true;
    readonly autoCreated: boolean;
    readonly prismId: string;
    readonly prismName: string;
    readonly digest: string;
    readonly mediaType?: string;
    readonly summary?: string;
    readonly galleryLabel?: string;
    readonly categories?: readonly string[];
    readonly previewImageDataUrl?: string;
    readonly previewImageWidth?: number;
    readonly previewImageHeight?: number;
    readonly audioWavDataUrl?: string;
    readonly audioDurationSec?: number;
    readonly writtenExcerpt?: string;
    readonly output: unknown;
  }>;
}

export function createLlamaVTensorAdapter(
  client: LlamaVTensorWorkerClient,
  studioBridge?: StudioAdapterBridge,
): ChatModelAdapter {
  let roundtripSeq = 0;

  return {
    async *run({
      messages,
      abortSignal,
      context,
      unstable_threadId,
      unstable_getMessage,
    }) {
      roundtripSeq += 1;
      const stepIdPrefix = `tc_${Date.now().toString(36)}_${roundtripSeq}`;
      const threadId = unstable_threadId || 'default-thread';
      const pendingAssistantMessage = unstable_getMessage ? unstable_getMessage() : null;
      const isToolContinuationStep = hasPendingToolResults(pendingAssistantMessage);
      const { text: latestUserText, hasAttachment } = extractLatestUserInput(messages);

      const turnSpan = studioBridge?.bus.startSpan('thread.turn', {
        endpointKind: 'thread',
        endpointId: threadId,
        kind: 'SERVER',
        attributes: {
          'verdant.thread.id': threadId,
          'verdant.input.chars': latestUserText.length,
        },
      });

      const initialState = client.getState();
      client.setActiveThreadId(threadId);
      client.cognitiveContext.ingestMessages(threadId, messages);
      const ledgerStats = client.cognitiveContext.getThreadLedgerStats(threadId);
      const publishedPrisms = studioBridge?.getPublishedPrisms() ?? [];
      const { certificate, snapshot } = client.cognitiveObserver.selectRoute({
        threadId,
        messages,
        latestUserText,
        hasAttachment,
        activeModel: initialState.model,
        cachedModelIds: initialState.cachedModelIds,
        isToolContinuationStep,
        hasImportedHarness: initialState.importedHarness !== null,
        publishedPrisms,
        hasConsolidatedHistory: ledgerStats.consolidatedTurns > 0,
      });
      client.recordCognitiveObservation(certificate, snapshot);
      turnSpan?.setAttributes({
        'verdant.route.id': certificate.routeId,
        'verdant.route.kind': certificate.kind,
      });

      const streamStartTime = Date.now();

      // ---------------------------------------------------------------------
      // Path A: LOD0 Verified Deterministic Settlement (`REQ-SETTLEMENT-LOD0`)
      // Executes instantaneously (<2ms, 0 bytes downloaded, works on first paint!)
      // ---------------------------------------------------------------------
      if (certificate.kind === 'lod0-verified' && certificate.directMatch) {
        const dm = certificate.directMatch;
        const toolDef = context?.tools?.[dm.toolName];
        if (toolDef?.execute) {
          const toolCallId = `${stepIdPrefix}_lod0`;
          const argsText = JSON.stringify(dm.args);
          try {
            const runTool = () =>
              toolDef.execute!(dm.args as never, {
                toolCallId,
                abortSignal,
                human: async () => undefined,
              } as never);
            const result = studioBridge
              ? await studioBridge.bus.runWithTraceContext(turnSpan?.context, runTool)
              : await runTool();
            if (dm.prismId && /^\/open\s+/i.test(latestUserText.trim())) {
              studioBridge?.onOpenPrism?.(dm.prismId);
            }
            const summaryText = formatDirectOperatorSummary(dm.toolName, result);
            const updatedSnap = client.cognitiveObserver.recordTurnOutcome(threadId, {
              factKey: dm.toolName,
              factSummary: summaryText,
            });
            client.recordCognitiveObservation(certificate, updatedSnap);
            const totalStreamTime = Date.now() - streamStartTime;
            turnSpan?.end('ok');
            yield {
              content: [
                {
                  type: 'tool-call' as const,
                  toolCallId,
                  toolName: dm.toolName,
                  args: dm.args as ReadonlyJSONObject,
                  argsText,
                  result,
                },
                {
                  type: 'text' as const,
                  text: summaryText,
                },
              ],
              status: { type: 'complete', reason: 'stop' },
              metadata: {
                custom: { prismRoute: certificate },
                timing: {
                  streamStartTime,
                  firstTokenTime: Date.now(),
                  totalStreamTime,
                  tokenCount: 0,
                  tokensPerSecond: 0,
                  totalChunks: 1,
                  toolCallCount: 1,
                },
              },
            };
            return;
          } catch (err) {
            const updatedSnap = client.cognitiveObserver.recordTurnOutcome(threadId, {
              hadToolError: true,
            });
            client.recordCognitiveObservation(certificate, updatedSnap);
            const msg = err instanceof Error ? err.message : String(err);
            turnSpan?.end('error', msg);
            yield {
              content: [{ type: 'text' as const, text: `Unable to evaluate \`${dm.toolName}\`: ${msg}` }],
              status: { type: 'complete', reason: 'stop' },
              metadata: { custom: { prismRoute: certificate } },
            };
            return;
          }
        }
      }

      // ---------------------------------------------------------------------
      // Path B: Neural Stream Gate (`FK-07` ContainerGrantMatrix + persistent Model Stream toggle)
      // NEVER start a surprise network download when submitting a prompt while Model Stream is Off!
      // ---------------------------------------------------------------------
      if (certificate.kind === 'neural-on-demand' && certificate.modelId) {
        const isAlreadyCachedLocally = initialState.cachedModelIds.includes(certificate.modelId);
        const streamAllowed =
          initialState.modelStreamEnabled &&
          client.cognitiveObserver.grants.isAllowed('network-weights');

        if (!isAlreadyCachedLocally && !streamAllowed) {
          turnSpan?.end('ok');
          yield {
            content: [
              {
                type: 'text' as const,
                text: 'Local AI is currently **Off**. Turn **Local AI** on in the top bar to enable private on-device responses.\n\n*(Quick actions, Studio creations, and your saved apps addressed with `@AppName` work immediately even while Local AI is off.)*',
              },
            ],
            status: { type: 'complete', reason: 'stop' },
            metadata: { custom: { prismRoute: certificate } },
          };
          return;
        }

        if (initialState.status === 'downloading' || initialState.status === 'loading') {
          yield {
            content: [
              {
                type: 'text' as const,
                text: `Local AI is still getting ready (${initialState.progressPercent}% · ${initialState.statusDetail}). Wait a moment and send again.`,
              },
            ],
            status: { type: 'complete', reason: 'stop' },
            metadata: { custom: { prismRoute: certificate } },
          };
          return;
        }

        if (abortSignal.aborted) return;
        await client.ensureUnifiedNeuralRoute(certificate.modelId, abortSignal);
        if (abortSignal.aborted) return;
      }

      const state = client.getState();
      if (!state.model || state.model.id === 'stories260k-f32') {
        yield {
          content: [
            {
              type: 'text' as const,
              text: 'Local AI is currently **Off**. Turn **Local AI** on in the top bar to start private on-device AI.',
            },
          ],
          status: { type: 'complete', reason: 'stop' },
          metadata: { custom: { prismRoute: certificate } },
        };
        return;
      }

      // ---------------------------------------------------------------------
      // Path C: Unified Neural Execution (adaptive depth + conditional tools)
      // ---------------------------------------------------------------------
      const allTools =
        context?.tools && Object.keys(context.tools).length > 0 ? context.tools : undefined;
      const activeTools = certificate.attachTools ? allTools : undefined;
      // Always parse `<tool_call>...</tool_call>` if the model emits one so spontaneous tool calls
      // NEVER leak raw XML to the user and instead resolve cleanly through Prisms (`DEC-013`).
      const parseToolCalls = true;

      const systemMessage = context?.system ?? UNIFIED_VERDANT_SYSTEM_PROMPT;
      const baseSampling = state.samplingCustomized
        ? state.sampling
        : {
            ...state.sampling,
            temperature: certificate.recommendedSampling.temperature,
            topP: certificate.recommendedSampling.topP,
            topK: certificate.recommendedSampling.topK,
            repeatPenalty: certificate.recommendedSampling.repeatPenalty,
          };
      const effectiveSampling = applyCallSettingsToSampling(
        baseSampling,
        state.importedHarness?.callSettings ??
          (state.samplingCustomized ? context?.callSettings : undefined),
        state.importedHarness,
      );
      const effectivePromptMode = state.importedHarness?.promptMode ?? state.promptMode;
      const thinkingPolicy = certificate.thinkingPolicy;
      const startsInReasoning = shouldPrefillOpenThink(
        state.model,
        effectivePromptMode,
        thinkingPolicy,
      );

      // Project bounded, lossless O(1) active cognitive context with epoch-aligned KV stability
      // and foveated lossless turn refaulting at the tail
      const projected = client.cognitiveContext.projectContext(threadId, messages, {
        latestUserText,
        belief: snapshot.belief,
        maxContextWindow: state.model.maxContextWindow,
        systemTokensEstimate: estimateTextTokens(systemMessage) + (activeTools ? 380 : 0),
      });

      const sensory = client.consumePendingSensory();
      const combinedTailRecall = [projected.foveatedRecallBlock, sensory.textBlock]
        .filter((s): s is string => Boolean(s && s.trim()))
        .join('\n\n') || null;

      const prompt = formatMessagesForLlama(
        projected.messages,
        state.model,
        effectivePromptMode,
        systemMessage,
        {
          thinkingMode: thinkingPolicy,
          tools: activeTools,
          pendingAssistantMessage,
          epochManifestBlock: projected.epochManifestBlock,
          foveatedRecallBlock: combinedTailRecall,
          preserveEmptyThinkPrefix: true,
        },
      );

      // Link an internal controller so tool calls stop cleanly right after `</tool_call>`
      const runController = new AbortController();
      const onExternalAbort = () => runController.abort();
      if (abortSignal.aborted) {
        runController.abort();
      } else {
        abortSignal.addEventListener('abort', onExternalAbort, { once: true });
      }

      let firstTokenTime: number | undefined;
      let totalChunks = 0;
      let latestText = '';
      let lastPromptTokens = 0;
      let lastGeneratedTokens = 0;
      let lastTokensPerSecond = 0;

      try {
        for await (const event of client.streamGenerate(
          prompt,
          runController.signal,
          effectiveSampling,
          threadId,
          sensory.visualKappa,
        )) {
          if (firstTokenTime === undefined) {
            firstTokenTime = Date.now();
          }
          totalChunks += 1;
          latestText = event.accumulatedText;
          lastPromptTokens = event.promptTokens;
          lastGeneratedTokens = event.generatedTokens;
          lastTokensPerSecond = event.tokensPerSecond;

          const parts = parseAssistantStreamParts(latestText, false, {
            startsInReasoning,
            parseToolCalls,
            idPrefix: stepIdPrefix,
          });
          const completedToolCall = parseToolCalls && hasCompleteToolCallTag(latestText);
          const toolCallCount = parts.filter((p) => p.type === 'tool-call').length;

          if (parts.length > 0 && !completedToolCall) {
            yield {
              content: parts,
              metadata: {
                custom: { prismRoute: certificate },
                timing: {
                  streamStartTime,
                  firstTokenTime,
                  tokenCount: lastGeneratedTokens,
                  tokensPerSecond: lastTokensPerSecond,
                  totalChunks,
                  toolCallCount,
                },
              },
            };
          }

          if (completedToolCall) {
            runController.abort();
            break;
          }
        }
      } finally {
        abortSignal.removeEventListener('abort', onExternalAbort);
      }

      if (abortSignal.aborted) {
        return;
      }

      const rawFinalParts = parseAssistantStreamParts(latestText, true, {
        startsInReasoning,
        parseToolCalls,
        idPrefix: stepIdPrefix,
      });

      // Autonomous Prism Tool-Call Resolution (`DEC-013`):
      // Any `<tool_call>` whose `toolName` is not already registered in `context.tools`
      // is resolved through a Published Prism — automatically synthesizing & publishing a new
      // procedural Prism in the active Space if none exists yet!
      const registeredTools = context?.tools ?? {};
      const finalParts: ParsedAssistantContentPart[] = [];
      let hasUnresolvedToolCalls = false;
      const autoCreatedPrisms: string[] = [];

      for (const part of rawFinalParts) {
        if (part.type !== 'tool-call') {
          finalParts.push(part);
          continue;
        }
        const registered = registeredTools[part.toolName];
        if (registered?.execute) {
          hasUnresolvedToolCalls = true;
          finalParts.push(part);
          continue;
        }
        if (studioBridge?.resolveOrCreatePrismForToolCall) {
          try {
            const prismResult = await studioBridge.resolveOrCreatePrismForToolCall(
              part.toolName,
              (part.args && typeof part.args === 'object' ? part.args : {}) as Record<string, unknown>,
              { latestUserText, signal: abortSignal },
            );
            if (prismResult.autoCreated) {
              autoCreatedPrisms.push(prismResult.prismName);
            }
            finalParts.push({
              ...part,
              result: prismResult,
            });
          } catch (err) {
            finalParts.push({
              ...part,
              isError: true,
              result: err instanceof Error ? err.message : String(err),
            });
          }
        } else {
          hasUnresolvedToolCalls = true;
          finalParts.push(part);
        }
      }

      if (
        autoCreatedPrisms.length > 0 &&
        !finalParts.some((p) => p.type === 'text' && p.text.trim().length > 0)
      ) {
        finalParts.push({
          type: 'text',
          text: `Created and ran **${autoCreatedPrisms.join(', ')}** to handle this request. Open it above to interact or customize it in Studio.`,
        });
      }

      const totalStreamTime = Date.now() - streamStartTime;
      const toolCallsEmitted = finalParts.filter((p) => p.type === 'tool-call');
      const hasToolCalls = hasUnresolvedToolCalls;
      if (hasToolCalls && studioBridge && turnSpan) {
        // Keep active W3C trace context available while Assistant-UI's ToolInvocationTracker runs tools
        studioBridge.bus.setActiveTraceContext(turnSpan.context);
      } else {
        studioBridge?.bus.setActiveTraceContext(undefined);
      }
      turnSpan?.end('ok');

      if (finalParts.length > 0) {
        // Close the second-order cybernetic feedback loop for neural turns too
        const visibleReply = finalParts
          .filter((p): p is TextMessagePart => p.type === 'text')
          .map((p) => p.text)
          .join(' ')
          .replace(/\s+/g, ' ')
          .trim();
        if (visibleReply && !hasToolCalls) {
          const updatedSnap = client.cognitiveObserver.recordTurnOutcome(threadId, {
            factKey: `turn_${snapshot.turnCount}`,
            factSummary: visibleReply.slice(0, 140),
          });
          client.recordCognitiveObservation(certificate, updatedSnap);
        }
        yield {
          content: finalParts,
          status: hasToolCalls
            ? { type: 'requires-action', reason: 'tool-calls' }
            : { type: 'complete', reason: 'stop' },
          metadata: {
            custom: { prismRoute: certificate },
            steps: [
              {
                usage: {
                  inputTokens: lastPromptTokens,
                  outputTokens: lastGeneratedTokens,
                },
              },
            ],
            timing: {
              streamStartTime,
              firstTokenTime,
              totalStreamTime,
              tokenCount: lastGeneratedTokens,
              tokensPerSecond: lastTokensPerSecond,
              totalChunks,
              toolCallCount: toolCallsEmitted.length,
            },
          },
        };
      } else {
        yield {
          content: [
            {
              type: 'text',
              text: '(End of sequence token generated by model.)',
            },
          ],
          status: { type: 'complete', reason: 'stop' },
          metadata: {
            custom: { prismRoute: certificate },
            timing: {
              streamStartTime,
              firstTokenTime,
              totalStreamTime,
              tokenCount: lastGeneratedTokens,
              tokensPerSecond: lastTokensPerSecond,
              totalChunks,
              toolCallCount: 0,
            },
          },
        };
      }
    },
  };
}
