/**
 * Official @assistant-ui/react ChatModelAdapter connecting useLocalRuntime
 * directly to the client-side Verdant inference worker.
 *
 * Strictly honors Assistant-UI's `ModelContext` (`context.system`, `context.callSettings`,
 * `context.config`):
 * - Never injects a fabricated/default system prompt when `context.system` is omitted.
 * - Parses reasoning traces (`<think>...</think>`, `<thought>...</thought>`) into official
 *   Assistant-UI `ReasoningMessagePart` (`type: 'reasoning'`) objects separate from
 *   user-facing `TextMessagePart` (`type: 'text'`) replies.
 * - Strips reasoning traces from prior assistant turns so multi-turn prompts stay clean.
 * - Includes text file attachments from `SimpleTextAttachmentAdapter`.
 */

import type {
  ChatModelAdapter,
  ReasoningMessagePart,
  TextMessagePart,
  ThreadMessage,
} from '@assistant-ui/react';
import { applyCallSettingsToSampling } from './harnessConfig';
import type { LoadedModelSummary } from './llamaRuntime';
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

const REASONING_TAG_PAIRS: readonly { readonly open: string; readonly close: string }[] = [
  { open: '<think>', close: '</think>' },
  { open: '<thought>', close: '</thought>' },
  { open: '<|thought|>', close: '<|/thought|>' },
];

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

function resolveEffectiveTemplate(
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
 * Computes the length of the longest suffix of `text` that is a strict prefix of any
 * tag in `tags` (case-insensitive), so streaming chunks never flash partial `<thi` or `</thi`
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

/**
 * Incrementally parses raw streamed model output into official Assistant-UI content parts:
 * - `{ type: 'reasoning', text }` for `<think>...</think>` / `<thought>...</thought>` blocks
 * - `{ type: 'text', text }` for visible answer text
 *
 * When `isComplete` is false (during streaming), partial tag prefixes at the tail are safely
 * held back until disambiguated, and an open `<think>` block streams live into a `reasoning` part.
 */
export function parseAssistantStreamParts(
  rawText: string,
  isComplete = false,
): readonly (ReasoningMessagePart | TextMessagePart)[] {
  if (!rawText) return [];

  const openTags = REASONING_TAG_PAIRS.map((p) => p.open);
  const parts: Array<ReasoningMessagePart | TextMessagePart> = [];
  let cursor = 0;
  let activeCloseTag: string | null = null;
  let afterReasoningBlock = false;

  const pushPart = (type: 'reasoning' | 'text', slice: string) => {
    const normalized =
      type === 'reasoning'
        ? slice.trim()
        : afterReasoningBlock
          ? slice.replace(/^\s+/, '')
          : slice;
    if (!normalized) return;
    if (type === 'text') {
      afterReasoningBlock = false;
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

  const lower = rawText.toLowerCase();

  while (cursor < rawText.length) {
    if (activeCloseTag === null) {
      // Find the earliest opening reasoning tag at or after cursor
      let bestIdx = -1;
      let bestPair: { readonly open: string; readonly close: string } | null = null;
      for (const pair of REASONING_TAG_PAIRS) {
        const idx = lower.indexOf(pair.open, cursor);
        if (idx !== -1 && (bestIdx === -1 || idx < bestIdx)) {
          bestIdx = idx;
          bestPair = pair;
        }
      }

      if (bestIdx === -1 || !bestPair) {
        const remaining = rawText.slice(cursor);
        const holdback = isComplete ? 0 : trailingPartialTagLength(remaining, openTags);
        const safeText = holdback > 0 ? remaining.slice(0, remaining.length - holdback) : remaining;
        pushPart('text', safeText);
        break;
      }

      if (bestIdx > cursor) {
        pushPart('text', rawText.slice(cursor, bestIdx));
      }
      cursor = bestIdx + bestPair.open.length;
      activeCloseTag = bestPair.close;
    } else {
      const closeIdx = lower.indexOf(activeCloseTag, cursor);
      if (closeIdx === -1) {
        const remaining = rawText.slice(cursor);
        const holdback = isComplete ? 0 : trailingPartialTagLength(remaining, [activeCloseTag]);
        const safeReasoning =
          holdback > 0 ? remaining.slice(0, remaining.length - holdback) : remaining;
        if (safeReasoning.trim().length > 0) {
          pushPart('reasoning', safeReasoning);
        }
        break;
      }

      const reasoningSlice = rawText.slice(cursor, closeIdx);
      if (reasoningSlice.trim().length > 0) {
        pushPart('reasoning', reasoningSlice);
      }
      cursor = closeIdx + activeCloseTag.length;
      activeCloseTag = null;
      afterReasoningBlock = true;
    }
  }

  return parts;
}

function extractTurns(
  messages: readonly ThreadMessage[],
): Array<{ role: 'system' | 'user' | 'assistant'; text: string }> {
  const extractText = (msg: ThreadMessage): string => {
    const contentChunks = msg.content
      .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
      .map((part) => part.text);

    // Include text file attachments attached via Assistant-UI's SimpleTextAttachmentAdapter
    if (msg.role === 'user' && Array.isArray(msg.attachments)) {
      for (const att of msg.attachments) {
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
    return msg.role === 'assistant' ? stripReasoningTags(joined) : joined;
  };

  const turns: Array<{ role: 'system' | 'user' | 'assistant'; text: string }> = [];
  for (const m of messages) {
    const text = extractText(m);
    if (!text) continue;
    if (m.role === 'user' || m.role === 'assistant' || m.role === 'system') {
      turns.push({ role: m.role, text });
    }
  }
  return turns;
}

/**
 * Formats assistant-ui ThreadMessage history into the exact prompt format expected
 * by the loaded GGUF model (ChatML, Llama-3, Gemma, Phi-3, Zephyr, Llama-2, or continuation).
 *
 * Strictly emits a `system` block ONLY when a non-empty `systemPrompt` is explicitly supplied
 * via Assistant-UI's `ModelContext` (or already present as a `role: 'system'` message).
 */
export function formatMessagesForLlama(
  messages: readonly ThreadMessage[],
  modelOrId?: Pick<LoadedModelSummary, 'id' | 'hparams'> | string,
  mode: PromptFormatMode = 'auto',
  systemPrompt?: string,
): string {
  const explicitSystem = systemPrompt?.trim() || '';
  const turns = extractTurns(messages);
  const effectiveTemplate = resolveEffectiveTemplate(modelOrId, mode);

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
      chunks.push(`<|im_start|>${t.role}\n${t.text}<|im_end|>\n`);
    }
    chunks.push('<|im_start|>assistant\n');
    return chunks.join('');
  }

  if (effectiveTemplate === 'llama3') {
    const hasSystem = turns.some((t) => t.role === 'system');
    const chunks: string[] = ['<|begin_of_text|>'];
    if (!hasSystem && explicitSystem) {
      chunks.push(`<|start_header_id|>system<|end_header_id|>\n\n${explicitSystem}<|eot_id|>`);
    }
    for (const t of turns) {
      chunks.push(`<|start_header_id|>${t.role}<|end_header_id|>\n\n${t.text}<|eot_id|>`);
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
      chunks.push(`<|${t.role}|>\n${t.text}<|end|>\n`);
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
      chunks.push(`<|${t.role}|>\n${t.text}</s>\n`);
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
      if (t.role === 'user') {
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
    } else if (t.role === 'user') {
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
 * `draftPrefix.trimStart()`, `formatMessagesForLlama([...messages, user(fullText)], model, mode, systemPrompt)`
 * strictly `startsWith(formatSpeculativePrefixForLlama(messages, draftPrefix, model, mode, systemPrompt))`!
 */
export function formatSpeculativePrefixForLlama(
  messages: readonly ThreadMessage[],
  draftPrefix: string,
  modelOrId?: Pick<LoadedModelSummary, 'id' | 'hparams'> | string,
  mode: PromptFormatMode = 'auto',
  systemPrompt?: string,
): string {
  const explicitSystem = systemPrompt?.trim() || '';
  const turns = extractTurns(messages);
  const effectiveTemplate = resolveEffectiveTemplate(modelOrId, mode);
  const openDraft = draftPrefix.trimStart();

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
      chunks.push(`<|im_start|>${t.role}\n${t.text}<|im_end|>\n`);
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
      chunks.push(`<|start_header_id|>${t.role}<|end_header_id|>\n\n${t.text}<|eot_id|>`);
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
      chunks.push(`<|${t.role}|>\n${t.text}<|end|>\n`);
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
      chunks.push(`<|${t.role}|>\n${t.text}</s>\n`);
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
      if (t.role === 'user') {
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
    } else if (t.role === 'user') {
      lines.push(`User: ${t.text}`);
    } else if (t.role === 'assistant') {
      lines.push(`Assistant: ${t.text}`);
    }
  }
  lines.push(`User: ${openDraft}`);
  return lines.join('\n');
}

export function createLlamaVTensorAdapter(
  client: LlamaVTensorWorkerClient,
): ChatModelAdapter {
  return {
    async *run({ messages, abortSignal, context }) {
      const state = client.getState();
      if (!state.model) {
        throw new Error('Model is still loading. Please wait a moment and try again.');
      }

      // Strictly read system instructions and callSettings from Assistant-UI's ModelContext
      const systemMessage = context?.system;
      const effectiveSampling = applyCallSettingsToSampling(
        state.sampling,
        context?.callSettings,
        state.importedHarness,
      );
      const effectivePromptMode = state.importedHarness?.promptMode ?? state.promptMode;

      const prompt = formatMessagesForLlama(
        messages,
        state.model,
        effectivePromptMode,
        systemMessage,
      );

      const streamStartTime = Date.now();
      let firstTokenTime: number | undefined;
      let totalChunks = 0;
      let latestText = '';
      let lastGeneratedTokens = 0;
      let lastTokensPerSecond = 0;

      for await (const event of client.streamGenerate(prompt, abortSignal, effectiveSampling)) {
        if (firstTokenTime === undefined) {
          firstTokenTime = Date.now();
        }
        totalChunks += 1;
        latestText = event.accumulatedText;
        lastGeneratedTokens = event.generatedTokens;
        lastTokensPerSecond = event.tokensPerSecond;

        const parts = parseAssistantStreamParts(latestText, false);
        if (parts.length > 0) {
          yield {
            content: parts,
            metadata: {
              timing: {
                streamStartTime,
                firstTokenTime,
                tokenCount: lastGeneratedTokens,
                tokensPerSecond: lastTokensPerSecond,
                totalChunks,
                toolCallCount: 0,
              },
            },
          };
        }
      }

      const finalParts = parseAssistantStreamParts(latestText, true);
      const totalStreamTime = Date.now() - streamStartTime;

      if (finalParts.length > 0) {
        yield {
          content: finalParts,
          metadata: {
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
      } else {
        yield {
          content: [
            {
              type: 'text',
              text: '(End of sequence token generated by model.)',
            },
          ],
          metadata: {
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
