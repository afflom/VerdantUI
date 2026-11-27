import test from 'node:test';
import assert from 'node:assert/strict';
import type { ThreadMessage } from '@assistant-ui/react';
import {
  formatMessagesForLlama,
  formatSpeculativePrefixForLlama,
  parseAssistantStreamParts,
  stripReasoningTags,
} from './assistantAdapter';

function makeMsg(
  role: 'user' | 'assistant' | 'system',
  text: string,
  extra?: Partial<ThreadMessage>,
): ThreadMessage {
  return {
    id: `${role}-${text.slice(0, 8)}`,
    role,
    content: [{ type: 'text', text }],
    createdAt: new Date(0),
    status: { type: 'complete', reason: 'stop' },
    metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} },
    attachments: [],
    ...extra,
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

test('formatMessagesForLlama NEVER injects a system prompt when ModelContext.system is omitted', () => {
  const messages: ThreadMessage[] = [makeMsg('user', 'Hi. My name is Alex.')];

  const chatmlNoSystem = formatMessagesForLlama(messages, 'stories260k-f32', 'chatml');
  assert.equal(
    chatmlNoSystem,
    '<|im_start|>user\nHi. My name is Alex.<|im_end|>\n<|im_start|>assistant\n',
  );
  assert.ok(!chatmlNoSystem.includes('<|im_start|>system'));

  const llama3NoSystem = formatMessagesForLlama(
    messages,
    { id: 'llama-3.2-3b', hparams: { name: 'llama-3.2', chatTemplate: '<|start_header_id|>' } as never },
    'auto',
  );
  assert.ok(!llama3NoSystem.includes('<|start_header_id|>system<|end_header_id|>'));
  assert.ok(llama3NoSystem.includes('<|start_header_id|>user<|end_header_id|>\n\nHi. My name is Alex.<|eot_id|>'));

  const phi3NoSystem = formatMessagesForLlama(
    messages,
    { id: 'phi-3-mini', hparams: { name: 'phi-3-mini', chatTemplate: '<|user|>\n{{ msg }}<|end|>\n<|assistant|>\n' } as never },
    'auto',
  );
  assert.ok(!phi3NoSystem.includes('<|system|>'));
  assert.equal(phi3NoSystem, '<|user|>\nHi. My name is Alex.<|end|>\n<|assistant|>\n');
});

test('formatMessagesForLlama emits explicit ModelContext.system across templates when provided by an imported harness', () => {
  const messages: ThreadMessage[] = [makeMsg('user', 'Hello world')];

  assert.equal(
    formatMessagesForLlama(messages, 'stories260k-f32', 'instruct'),
    'User: Hello world\nAssistant:',
  );
  assert.equal(
    formatMessagesForLlama(messages, 'stories260k-f32', 'instruct', 'Be concise.'),
    'System: Be concise.\nUser: Hello world\nAssistant:',
  );
  assert.equal(
    formatMessagesForLlama(messages, 'llama-3-instruct-q4', 'continuation', 'Ignored in raw continuation'),
    'Hello world',
  );

  const chatmlWithSystem = formatMessagesForLlama(
    messages,
    'stories260k-f32',
    'chatml',
    'You are a code reviewer.',
  );
  assert.equal(
    chatmlWithSystem,
    '<|im_start|>system\nYou are a code reviewer.<|im_end|>\n<|im_start|>user\nHello world<|im_end|>\n<|im_start|>assistant\n',
  );

  const phi3WithSystem = formatMessagesForLlama(
    messages,
    { id: 'phi-3-mini', hparams: { name: 'phi-3-mini', chatTemplate: '<|user|>\n{{ msg }}<|end|>\n<|assistant|>\n' } as never },
    'auto',
    'You are terse.',
  );
  assert.equal(
    phi3WithSystem,
    '<|system|>\nYou are terse.<|end|>\n<|user|>\nHello world<|end|>\n<|assistant|>\n',
  );
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
  assert.ok(phi3Prompt.startsWith('<|user|>\n'));
  assert.ok(phi3Prompt.endsWith('<|assistant|>\n'));
});

test('parseAssistantStreamParts separates <think> blocks into official ReasoningMessagePart and holds back partial tags while streaming', () => {
  // Streaming partial opening tag must not leak "<thi" into text
  assert.deepEqual(parseAssistantStreamParts('<thi', false), []);

  // Streaming inside open <think> block emits a reasoning part and holds back partial "</thi"
  assert.deepEqual(parseAssistantStreamParts('<think>Let me greet Alex</thi', false), [
    { type: 'reasoning', text: 'Let me greet Alex' },
  ]);

  // Complete <think> + visible reply splits cleanly into reasoning + text without raw tags
  const completed = parseAssistantStreamParts(
    '<think>\nThe user introduced themselves as Alex.\n</think>\n\nHello, Alex! How can I help you today?',
    true,
  );
  assert.deepEqual(completed, [
    { type: 'reasoning', text: 'The user introduced themselves as Alex.' },
    { type: 'text', text: 'Hello, Alex! How can I help you today?' },
  ]);

  // Empty <think></think> produces only the visible answer
  assert.deepEqual(parseAssistantStreamParts('<think>   </think>\nHello, Alex!', true), [
    { type: 'text', text: 'Hello, Alex!' },
  ]);

  // Plain text without reasoning tags passes through verbatim
  assert.deepEqual(parseAssistantStreamParts('Hello, Alex!', true), [
    { type: 'text', text: 'Hello, Alex!' },
  ]);
});

test('stripReasoningTags and multi-turn formatting exclude ReasoningMessagePart and legacy <think> traces while preserving attachments', () => {
  assert.equal(
    stripReasoningTags('<think>internal monologue</think>\n\nHello, Alex!'),
    'Hello, Alex!',
  );

  const multiTurn: ThreadMessage[] = [
    makeMsg('user', 'Hi. My name is Alex.', {
      attachments: [
        {
          id: 'att-1',
          type: 'document',
          name: 'notes.txt',
          contentType: 'text/plain',
          content: [{ type: 'text', text: 'Project: VerdantUI' }],
          status: { type: 'complete' },
        } as never,
      ],
    }),
    {
      ...makeMsg('assistant', 'Hello, Alex!'),
      content: [
        { type: 'reasoning', text: 'Private chain of thought that must not re-enter prompt.' },
        { type: 'text', text: 'Hello, Alex!' },
      ],
    } as unknown as ThreadMessage,
    makeMsg('user', 'What project am I working on?'),
  ];

  const prompt = formatMessagesForLlama(multiTurn, 'qwen', 'chatml');
  assert.ok(!prompt.includes('Private chain of thought'));
  assert.ok(prompt.includes('[Attached file: notes.txt]\nProject: VerdantUI'));
  assert.ok(prompt.includes('<|im_start|>assistant\nHello, Alex!<|im_end|>'));
});

test('formatSpeculativePrefixForLlama maintains strict byte-prefix parity with formatMessagesForLlama across all 8 templates (with and without system prompt)', () => {
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

  for (const systemPrompt of [undefined, 'Imported harness system prompt.']) {
    for (const { label, model } of templates) {
      // 1. Empty history (boot warm-up + first user turn)
      const finalTurn0 = formatMessagesForLlama([makeMsg('user', fullDraft)], model, 'auto', systemPrompt);
      const bootWarm0 = formatSpeculativePrefixForLlama([], '', model, 'auto', systemPrompt);
      const draftWarm0 = formatSpeculativePrefixForLlama([], stablePrefix, model, 'auto', systemPrompt);
      assert.ok(
        finalTurn0.startsWith(bootWarm0),
        `[${label}] turn-0 final prompt must start with boot warm-up prefix (system=${Boolean(systemPrompt)})`,
      );
      assert.ok(
        finalTurn0.startsWith(draftWarm0),
        `[${label}] turn-0 final prompt must start with speculative draft prefix (system=${Boolean(systemPrompt)})`,
      );

      // 2. Multi-turn history
      const finalMulti = formatMessagesForLlama([...history, makeMsg('user', fullDraft)], model, 'auto', systemPrompt);
      const bootMulti = formatSpeculativePrefixForLlama(history, '', model, 'auto', systemPrompt);
      const draftMulti = formatSpeculativePrefixForLlama(history, stablePrefix, model, 'auto', systemPrompt);
      assert.ok(
        finalMulti.startsWith(bootMulti),
        `[${label}] multi-turn final prompt must start with empty-draft warm prefix (system=${Boolean(systemPrompt)})`,
      );
      assert.ok(
        finalMulti.startsWith(draftMulti),
        `[${label}] multi-turn final prompt must start with speculative draft prefix (system=${Boolean(systemPrompt)})`,
      );
      assert.ok(
        draftMulti.startsWith(bootMulti),
        `[${label}] speculative draft prefix must monotonically extend boot prefix (system=${Boolean(systemPrompt)})`,
      );
    }
  }
});

test('formatMessagesForLlama and formatSpeculativePrefixForLlama redact large base64 data URLs from tool results', () => {
  const fakePng = `data:image/png;base64,${'A'.repeat(64_000)}`;
  const fakeWav = `data:audio/wav;base64,${'B'.repeat(120_000)}`;
  const msgs: ThreadMessage[] = [
    makeMsg('user', 'Create a logo'),
    {
      id: 'a-tool-1',
      role: 'assistant',
      createdAt: new Date(),
      status: { type: 'complete', reason: 'stop' },
      content: [
        {
          type: 'tool-call',
          toolCallId: 'tc_1',
          toolName: 'studio_create_artifact',
          args: { prompt: 'Create a logo' },
          argsText: '{"prompt":"Create a logo"}',
          result: {
            prismName: 'Meridian Logo',
            previewImageDataUrl: fakePng,
            audioWavDataUrl: fakeWav,
          },
        },
        { type: 'text', text: 'Created Meridian Logo.' },
      ],
      metadata: { unstable_state: null, unstable_annotations: [], unstable_data: [], steps: [], custom: {} },
    },
    makeMsg('user', 'Now make cover art'),
  ];
  const model = 'qwen3-0.6b-q4_0';
  const prompt = formatMessagesForLlama(msgs, model, 'auto');
  const spec = formatSpeculativePrefixForLlama(msgs.slice(0, 2), 'Now make', model, 'auto');
  assert.ok(prompt.length < 2000, `Prompt bloated to ${prompt.length} chars`);
  assert.ok(spec.length < 2000, `Speculative prompt bloated to ${spec.length} chars`);
  assert.ok(!prompt.includes('AAAA'), 'Raw PNG base64 must never leak into prompt');
  assert.ok(!prompt.includes('BBBB'), 'Raw WAV base64 must never leak into prompt');
  assert.ok(prompt.includes('inline image/png artifact rendered in UI'));
  assert.ok(prompt.includes('inline audio/wav artifact rendered in UI'));
});
