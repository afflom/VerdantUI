/**
 * Official @assistant-ui/react ChatModelAdapter connecting useLocalRuntime
 * directly to the client-side llama.cpp + afflom/vTensor inference engine,
 * with automatic GGUF chat-template detection (ChatML, Llama-3, Zephyr, Llama-2, Story).
 */

import type { ChatModelAdapter, ThreadMessage } from '@assistant-ui/react';
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

const DEFAULT_SYSTEM_PROMPT = 'You are a helpful, concise, and accurate AI assistant.';

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

function extractTurns(
  messages: readonly ThreadMessage[],
): Array<{ role: 'system' | 'user' | 'assistant'; text: string }> {
  const extractText = (msg: ThreadMessage): string =>
    msg.content
      .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
      .map((part) => part.text)
      .join('\n')
      .trim();

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
 * by the loaded GGUF model (ChatML for SmolLM2 / Qwen2.5, Llama-3, Zephyr, or continuation).
 */
export function formatMessagesForLlama(
  messages: readonly ThreadMessage[],
  modelOrId?: Pick<LoadedModelSummary, 'id' | 'hparams'> | string,
  mode: PromptFormatMode = 'auto',
): string {
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
    if (!hasSystem) {
      chunks.push(`<|im_start|>system\n${DEFAULT_SYSTEM_PROMPT}<|im_end|>\n`);
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
    if (!hasSystem) {
      chunks.push(`<|start_header_id|>system<|end_header_id|>\n\n${DEFAULT_SYSTEM_PROMPT}<|eot_id|>`);
    }
    for (const t of turns) {
      chunks.push(`<|start_header_id|>${t.role}<|end_header_id|>\n\n${t.text}<|eot_id|>`);
    }
    chunks.push('<|start_header_id|>assistant<|end_header_id|>\n\n');
    return chunks.join('');
  }

  if (effectiveTemplate === 'gemma') {
    const chunks: string[] = [];
    for (const t of turns) {
      const role = t.role === 'assistant' ? 'model' : 'user';
      chunks.push(`<start_of_turn>${role}\n${t.text}<end_of_turn>\n`);
    }
    chunks.push('<start_of_turn>model\n');
    return chunks.join('');
  }

  if (effectiveTemplate === 'phi3') {
    const hasSystem = turns.some((t) => t.role === 'system');
    const chunks: string[] = [];
    if (!hasSystem) {
      chunks.push(`<|system|>\n${DEFAULT_SYSTEM_PROMPT}<|end|>\n`);
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
    if (!hasSystem) {
      chunks.push(`<|system|>\n${DEFAULT_SYSTEM_PROMPT}</s>\n`);
    }
    for (const t of turns) {
      chunks.push(`<|${t.role}|>\n${t.text}</s>\n`);
    }
    chunks.push('<|assistant|>\n');
    return chunks.join('');
  }

  if (effectiveTemplate === 'llama2') {
    const chunks: string[] = [];
    for (const t of turns) {
      if (t.role === 'user') {
        chunks.push(`[INST] ${t.text} [/INST]`);
      } else if (t.role === 'assistant') {
        chunks.push(` ${t.text} </s>`);
      }
    }
    return chunks.join('');
  }

  // Standard multi-turn instruction prompt fallback
  const lines: string[] = [];
  for (const t of turns) {
    if (t.role === 'user') {
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
 * `draftPrefix.trimStart()`, `formatMessagesForLlama([...messages, user(fullText)], model, mode)`
 * strictly `startsWith(formatSpeculativePrefixForLlama(messages, draftPrefix, model, mode))`!
 * Because the closing turn delimiter is omitted until submission, every speculatively prefilled
 * KV token is 100% reusable bit-for-bit when the user continues typing or presses Enter.
 */
export function formatSpeculativePrefixForLlama(
  messages: readonly ThreadMessage[],
  draftPrefix: string,
  modelOrId?: Pick<LoadedModelSummary, 'id' | 'hparams'> | string,
  mode: PromptFormatMode = 'auto',
): string {
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
    if (!hasSystem) {
      chunks.push(`<|im_start|>system\n${DEFAULT_SYSTEM_PROMPT}<|im_end|>\n`);
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
    if (!hasSystem) {
      chunks.push(`<|start_header_id|>system<|end_header_id|>\n\n${DEFAULT_SYSTEM_PROMPT}<|eot_id|>`);
    }
    for (const t of turns) {
      chunks.push(`<|start_header_id|>${t.role}<|end_header_id|>\n\n${t.text}<|eot_id|>`);
    }
    chunks.push(`<|start_header_id|>user<|end_header_id|>\n\n${openDraft}`);
    return chunks.join('');
  }

  if (effectiveTemplate === 'gemma') {
    const chunks: string[] = [];
    for (const t of turns) {
      const role = t.role === 'assistant' ? 'model' : 'user';
      chunks.push(`<start_of_turn>${role}\n${t.text}<end_of_turn>\n`);
    }
    chunks.push(`<start_of_turn>user\n${openDraft}`);
    return chunks.join('');
  }

  if (effectiveTemplate === 'phi3') {
    const hasSystem = turns.some((t) => t.role === 'system');
    const chunks: string[] = [];
    if (!hasSystem) {
      chunks.push(`<|system|>\n${DEFAULT_SYSTEM_PROMPT}<|end|>\n`);
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
    if (!hasSystem) {
      chunks.push(`<|system|>\n${DEFAULT_SYSTEM_PROMPT}</s>\n`);
    }
    for (const t of turns) {
      chunks.push(`<|${t.role}|>\n${t.text}</s>\n`);
    }
    chunks.push(`<|user|>\n${openDraft}`);
    return chunks.join('');
  }

  if (effectiveTemplate === 'llama2') {
    const chunks: string[] = [];
    for (const t of turns) {
      if (t.role === 'user') {
        chunks.push(`[INST] ${t.text} [/INST]`);
      } else if (t.role === 'assistant') {
        chunks.push(` ${t.text} </s>`);
      }
    }
    chunks.push(`[INST] ${openDraft}`);
    return chunks.join('');
  }

  // Standard multi-turn instruction prompt fallback
  const lines: string[] = [];
  for (const t of turns) {
    if (t.role === 'user') {
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
    async *run({ messages, abortSignal }) {
      const state = client.getState();
      if (!state.model) {
        throw new Error(
          'Model is still loading into vTensor GGML buffers. Please wait a moment and try again.',
        );
      }

      const prompt = formatMessagesForLlama(messages, state.model, state.promptMode);
      let latestText = '';

      for await (const event of client.streamGenerate(prompt, abortSignal)) {
        latestText = event.accumulatedText;
        yield {
          content: [
            {
              type: 'text',
              text: latestText,
            },
          ],
        };
      }

      if (!latestText.trim()) {
        yield {
          content: [
            {
              type: 'text',
              text: '(End of sequence token generated by model.)',
            },
          ],
        };
      }
    },
  };
}
