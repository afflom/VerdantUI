import test from 'node:test';
import assert from 'node:assert/strict';
import type { ThreadMessage } from '@assistant-ui/react';
import { formatMessagesForLlama, formatSpeculativePrefixForLlama } from './assistantAdapter';

function makeMsg(role: 'user' | 'assistant', text: string): ThreadMessage {
  return {
    id: `${role}-${text.slice(0, 8)}`,
    role,
    content: [{ type: 'text', text }],
    createdAt: new Date(0),
    status: { type: 'complete', reason: 'stop' },
    metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} },
    attachments: [],
  } as unknown as ThreadMessage;
}

test('formatMessagesForLlama auto mode picks continuation for TinyStories models and chat turns for general models', () => {
  const messages: ThreadMessage[] = [
    makeMsg('user', 'Once upon a time'),
    makeMsg('assistant', 'there was a fox.'),
    makeMsg('user', 'The fox found a key.'),
  ];

  const autoStory = formatMessagesForLlama(messages, 'stories260k-f32', 'auto');
  assert.equal(autoStory, 'Once upon a time there was a fox. The fox found a key.');

  const autoGeneral = formatMessagesForLlama(messages, 'llama-3-instruct-q4', 'auto');
  assert.equal(
    autoGeneral,
    'User: Once upon a time\nAssistant: there was a fox.\nUser: The fox found a key.\nAssistant:',
  );
});

test('formatMessagesForLlama respects explicit chatml, continuation, and instruct overrides', () => {
  const messages: ThreadMessage[] = [makeMsg('user', 'Hello world')];

  assert.equal(
    formatMessagesForLlama(messages, 'stories260k-f32', 'instruct'),
    'User: Hello world\nAssistant:',
  );
  assert.equal(
    formatMessagesForLlama(messages, 'llama-3-instruct-q4', 'continuation'),
    'Hello world',
  );
  const chatml = formatMessagesForLlama(messages, 'stories260k-f32', 'chatml');
  assert.match(chatml, /^<\|im_start\|>system\n[\s\S]+<\|im_end\|>\n<\|im_start\|>user\nHello world<\|im_end\|>\n<\|im_start\|>assistant\n$/);
});

test('formatMessagesForLlama auto-detects Gemma and Phi-3 chat templates from tokenizer metadata', () => {
  const messages: ThreadMessage[] = [
    makeMsg('user', 'Explain gravity briefly.'),
    makeMsg('assistant', 'Gravity pulls mass together.'),
    makeMsg('user', 'Give an example.'),
  ];

  const gemmaPrompt = formatMessagesForLlama(
    messages,
    {
      id: 'gemma-2-2b-it',
      hparams: { name: 'gemma-2-2b-it', chatTemplate: '<start_of_turn>user\n{{ msg }}<end_of_turn>' } as never,
    },
    'auto',
  );
  assert.ok(gemmaPrompt.includes('<start_of_turn>user\n'));
  assert.ok(gemmaPrompt.endsWith('<start_of_turn>model\n'));

  const phi3Prompt = formatMessagesForLlama(
    messages,
    {
      id: 'phi-3-mini-4k-instruct',
      hparams: { name: 'phi-3-mini', chatTemplate: '<|user|>\n{{ msg }}<|end|>\n<|assistant|>\n' } as never,
    },
    'auto',
  );
  assert.ok(phi3Prompt.startsWith('<|system|>\n'));
  assert.ok(phi3Prompt.endsWith('<|assistant|>\n'));
});

test('formatSpeculativePrefixForLlama maintains strict byte-prefix parity with formatMessagesForLlama across all 8 templates', () => {
  const history: ThreadMessage[] = [
    makeMsg('user', 'Hello there'),
    makeMsg('assistant', 'Hi! How can I help?'),
  ];
  const fullDraft = 'What is the capital of France?';
  const stablePrefix = 'What is the capital ';

  const templates = [
    { label: 'chatml', model: { id: 'smollm2', hparams: { name: 'smollm2', chatTemplate: '<|im_start|>' } as never } },
    { label: 'llama3', model: { id: 'llama3', hparams: { name: 'llama3', chatTemplate: '<|start_header_id|>' } as never } },
    { label: 'gemma', model: { id: 'gemma2', hparams: { name: 'gemma2', chatTemplate: '<start_of_turn>' } as never } },
    { label: 'phi3', model: { id: 'phi3', hparams: { name: 'phi3', chatTemplate: '<|user|><|end|>' } as never } },
    { label: 'zephyr', model: { id: 'zephyr', hparams: { name: 'zephyr', chatTemplate: '<|assistant|>' } as never } },
    { label: 'llama2', model: { id: 'llama2', hparams: { name: 'llama2', chatTemplate: '[INST]' } as never } },
    { label: 'continuation', model: { id: 'stories260k', hparams: { name: 'stories260k', chatTemplate: null } as never } },
    { label: 'plain', model: { id: 'generic-llm', hparams: { name: 'generic-llm', chatTemplate: null } as never } },
  ];

  for (const { label, model } of templates) {
    // 1. Empty history (boot warm-up + first user turn)
    const finalTurn0 = formatMessagesForLlama([makeMsg('user', fullDraft)], model, 'auto');
    const bootWarm0 = formatSpeculativePrefixForLlama([], '', model, 'auto');
    const draftWarm0 = formatSpeculativePrefixForLlama([], stablePrefix, model, 'auto');
    assert.ok(
      finalTurn0.startsWith(bootWarm0),
      `[${label}] turn-0 final prompt must start with boot warm-up prefix`,
    );
    assert.ok(
      finalTurn0.startsWith(draftWarm0),
      `[${label}] turn-0 final prompt must start with speculative draft prefix`,
    );

    // 2. Multi-turn history
    const finalMulti = formatMessagesForLlama([...history, makeMsg('user', fullDraft)], model, 'auto');
    const bootMulti = formatSpeculativePrefixForLlama(history, '', model, 'auto');
    const draftMulti = formatSpeculativePrefixForLlama(history, stablePrefix, model, 'auto');
    assert.ok(
      finalMulti.startsWith(bootMulti),
      `[${label}] multi-turn final prompt must start with empty-draft warm prefix`,
    );
    assert.ok(
      finalMulti.startsWith(draftMulti),
      `[${label}] multi-turn final prompt must start with speculative draft prefix`,
    );
    assert.ok(
      draftMulti.startsWith(bootMulti),
      `[${label}] speculative draft prefix must monotonically extend boot prefix`,
    );
  }
});
