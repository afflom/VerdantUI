import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import type { ThreadMessage } from '@assistant-ui/react';
import {
  formatMessagesForLlama,
  hasCompleteToolCallTag,
  isThinkingCapableModel,
  parseAssistantStreamParts,
  shouldPrefillEmptyThink,
  shouldPrefillOpenThink,
} from './assistantAdapter';
import { LlamaVTensorRuntime, type LoadedModelSummary } from './llamaRuntime';
import {
  buildVerdantModelContext,
  createVerdantClientTools,
  evaluateMathExpression,
  findTierForModelId,
  recommendVerdantTier,
  resolveTierCatalogModel,
  VERDANT_HARNESS_MODES,
  VERDANT_RUNTIME_TIERS,
} from './verdantHarness';

const STORIES_260K_PATH = path.resolve(process.cwd(), 'public/models/stories260K.gguf');

function makeUserMsg(text: string): ThreadMessage {
  return {
    id: `user-${text.slice(0, 8)}`,
    role: 'user',
    content: [{ type: 'text', text }],
    createdAt: new Date(0),
    attachments: [],
    metadata: { custom: {} },
  } as unknown as ThreadMessage;
}

const QWEN3_4B_SUMMARY = {
  id: 'qwen3-4b-q4_k_m',
  displayName: 'Qwen3 4B (Q4_K_M)',
  hparams: {
    name: 'Qwen3-4B',
    architecture: 'qwen3',
    chatTemplate: '<|im_start|>{{ role }}\n{{ content }}<|im_end|>\n',
  },
} as unknown as LoadedModelSummary;

const QWEN25_15B_SUMMARY = {
  id: 'qwen2.5-1.5b-instruct-q4_k_m',
  displayName: 'Qwen2.5 1.5B Instruct (Q4_K_M)',
  hparams: {
    name: 'Qwen2.5-1.5B-Instruct',
    architecture: 'qwen2',
    chatTemplate: '<|im_start|>{{ role }}\n{{ content }}<|im_end|>\n',
  },
} as unknown as LoadedModelSummary;

test('LlamaVTensorRuntime enforces strict per-thread KV isolation with zero cross-thread conflation', async () => {
  const bytes = fs.readFileSync(STORIES_260K_PATH);
  const sharedRuntime = new LlamaVTensorRuntime();
  sharedRuntime.loadFromBuffer('stories260k-f32', 'Stories 260K', '/models/stories260K.gguf', bytes);

  const freshRuntime = new LlamaVTensorRuntime();
  freshRuntime.loadFromBuffer('stories260k-f32', 'Stories 260K', '/models/stories260K.gguf', bytes);

  const greedy = { temperature: 0, topP: 1, topK: 1, repeatPenalty: 1.0, maxTokens: 16, seed: 7 };

  // 1. Run Thread A to populate KV cache with completely different context
  let threadAText = '';
  for await (const ev of sharedRuntime.generate(
    'One sunny morning, a blue cat climbed the tall apple tree.',
    greedy,
    undefined,
    undefined,
    'thread-alpha',
  )) {
    threadAText = ev.accumulatedText;
  }
  assert.ok(threadAText.length > 0);

  // 2. Abort a mid-stream generation on Thread A to test KV truncation hygiene
  const abortCtrl = new AbortController();
  let abortedTokens = 0;
  for await (const _ of sharedRuntime.generate(
    'One sunny morning, a blue cat climbed the tall apple tree and saw a big red bird.',
    { ...greedy, maxTokens: 32 },
    abortCtrl.signal,
    undefined,
    'thread-alpha',
  )) {
    abortedTokens++;
    if (abortedTokens >= 2) {
      abortCtrl.abort();
    }
  }

  // 3. Switch to Thread B on the shared runtime — must match freshRuntime bit-for-bit
  const threadBPrompt = 'Once upon a time, a little dog found a shiny bell.';
  let sharedThreadBText = '';
  for await (const ev of sharedRuntime.generate(
    threadBPrompt,
    greedy,
    undefined,
    undefined,
    'thread-beta',
  )) {
    sharedThreadBText = ev.accumulatedText;
  }

  let freshThreadBText = '';
  for await (const ev of freshRuntime.generate(
    threadBPrompt,
    greedy,
    undefined,
    undefined,
    'thread-beta',
  )) {
    freshThreadBText = ev.accumulatedText;
  }

  assert.equal(
    sharedThreadBText,
    freshThreadBText,
    'Thread B output on shared runtime must match a completely clean runtime bit-for-bit',
  );
  assert.ok(sharedThreadBText.length > 5);
});

test('Qwen3 Fast vs Thinking mode prefills exact Qwen3 thinking-control tags and parses streaming reasoning', () => {
  assert.equal(isThinkingCapableModel(QWEN3_4B_SUMMARY), true);
  assert.equal(isThinkingCapableModel(QWEN25_15B_SUMMARY), false);

  assert.equal(shouldPrefillEmptyThink(QWEN3_4B_SUMMARY, 'auto', 'off'), true);
  assert.equal(shouldPrefillOpenThink(QWEN3_4B_SUMMARY, 'auto', 'on'), true);
  assert.equal(shouldPrefillEmptyThink(QWEN25_15B_SUMMARY, 'auto', 'off'), false);
  assert.equal(shouldPrefillOpenThink(QWEN25_15B_SUMMARY, 'auto', 'on'), false);

  const msgs = [makeUserMsg('What is 2 + 2?')];

  // Fast mode (thinkingMode: 'off') on Qwen3 prefills empty <think>\n\n</think>\n\n
  const fastPrompt = formatMessagesForLlama(msgs, QWEN3_4B_SUMMARY, 'auto', VERDANT_HARNESS_MODES.fast.system, {
    thinkingMode: 'off',
  });
  assert.ok(
    fastPrompt.endsWith('<|im_start|>assistant\n<think>\n\n</think>\n\n'),
    `Expected Fast mode Qwen3 empty think prefill, got: ${fastPrompt}`,
  );

  // Thinking mode (thinkingMode: 'on') on Qwen3 prefills opening <think>\n
  const thinkingPrompt = formatMessagesForLlama(
    msgs,
    QWEN3_4B_SUMMARY,
    'auto',
    VERDANT_HARNESS_MODES.thinking.system,
    { thinkingMode: 'on' },
  );
  assert.ok(
    thinkingPrompt.endsWith('<|im_start|>assistant\n<think>\n'),
    `Expected Thinking mode Qwen3 open think prefill, got: ${thinkingPrompt}`,
  );

  // Streaming parser with startsInReasoning=true routes prefilled thought into ReasoningMessagePart
  const midStream = parseAssistantStreamParts('Let me add 2 and 2 carefully.</thi', false, {
    startsInReasoning: true,
  });
  assert.equal(midStream.length, 1);
  assert.equal(midStream[0].type, 'reasoning');
  assert.equal(midStream[0].text, 'Let me add 2 and 2 carefully.');

  const finished = parseAssistantStreamParts('Let me add 2 and 2 carefully.\n</think>\n\nThe answer is 4.', true, {
    startsInReasoning: true,
  });
  assert.equal(finished.length, 2);
  assert.equal(finished[0].type, 'reasoning');
  assert.equal(finished[0].text, 'Let me add 2 and 2 carefully.');
  assert.equal(finished[1].type, 'text');
  assert.equal(finished[1].text, 'The answer is 4.');
});

test('Agentic tool-calling protocol formats <tools>/<tool_call>/<tool_response> and parses ToolCallMessagePart', () => {
  const tools = createVerdantClientTools(() => ({
    model: QWEN3_4B_SUMMARY,
    telemetry: {
      activeProvider: 'vTensor',
      verdantVersion: '2.22.4',
      kernelAcceleration: 'wasm-simd',
      threadPool: { active: true, workerCount: 4, barrierLatencyUs: 12 },
      performanceContract: { verdict: 'OPTIMAL' },
    } as never,
    opfsStorage: { supported: true, storedModels: [] } as never,
    lastTokensPerSec: 18.5,
    harnessMode: 'agentic',
  }));

  const step1Prompt = formatMessagesForLlama(
    [makeUserMsg('Compute 144 * 37')],
    QWEN3_4B_SUMMARY,
    'auto',
    VERDANT_HARNESS_MODES.agentic.system,
    { thinkingMode: 'auto', tools },
  );
  assert.ok(step1Prompt.includes('<tools>'));
  assert.ok(step1Prompt.includes('"name":"calculate"'));
  assert.ok(step1Prompt.includes('</tools>'));

  // Model emits a tool call (with optional hallucinated <tool_response> that must be stripped)
  const rawStep1Output = [
    '<think>I should use the calculate tool.</think>',
    '<tool_call>',
    '{"name": "calculate", "arguments": {"expression": "144 * 37"}}',
    '</tool_call>',
    '<tool_response>{"result": 999999}</tool_response>',
  ].join('\n');

  assert.equal(hasCompleteToolCallTag(rawStep1Output), true);
  const step1Parts = parseAssistantStreamParts(rawStep1Output, true, { parseToolCalls: true });
  assert.equal(step1Parts[0].type, 'reasoning');
  assert.equal(step1Parts[1].type, 'tool-call');
  if (step1Parts[1].type === 'tool-call') {
    assert.equal(step1Parts[1].toolName, 'calculate');
    assert.deepEqual(step1Parts[1].args, { expression: '144 * 37' });
  }
  // Ensure hallucinated <tool_response> is never leaked as visible text
  assert.ok(!step1Parts.some((p) => p.type === 'text' && p.text.includes('999999')));

  // Step 2: Assistant-UI passes settled tool-call result on pendingAssistantMessage
  const pendingAssistant = {
    id: 'assistant-step-1',
    role: 'assistant',
    content: [
      {
        type: 'tool-call',
        toolCallId: 'call_1',
        toolName: 'calculate',
        args: { expression: '144 * 37' },
        argsText: '{"expression":"144 * 37"}',
        result: { expression: '144 * 37', result: 5328, formatted: '5,328' },
      },
    ],
  } as unknown as ThreadMessage;

  const step2Prompt = formatMessagesForLlama(
    [makeUserMsg('Compute 144 * 37')],
    QWEN3_4B_SUMMARY,
    'auto',
    VERDANT_HARNESS_MODES.agentic.system,
    { thinkingMode: 'auto', tools, pendingAssistantMessage: pendingAssistant },
  );
  assert.ok(step2Prompt.includes('<tool_call>\n{"name":"calculate","arguments":{"expression":"144 * 37"}}\n</tool_call>'));
  assert.ok(step2Prompt.includes('<tool_response>\n{"name":"calculate","result":{"expression":"144 * 37","result":5328,"formatted":"5,328"}}\n</tool_response>'));
  assert.ok(step2Prompt.endsWith('<|im_start|>assistant\n'));
});

test('evaluateMathExpression and all 5 Verdant client tools run deterministically with zero eval', async () => {
  assert.equal(evaluateMathExpression('(144 * 37) + sqrt(625)'), 5353);
  assert.equal(evaluateMathExpression('2 ^ 10 - 24'), 1000);
  assert.equal(evaluateMathExpression('-5 + max(3, 9, 1) * 2'), 13);
  assert.ok(Math.abs(evaluateMathExpression('sin(pi / 2) + ln(e)') - 2) < 1e-12);
  assert.throws(() => evaluateMathExpression('1 / 0'), /Division by zero/);
  assert.throws(() => evaluateMathExpression('process.exit(1)'), /Unsupported character|Unknown identifier/);

  const tools = createVerdantClientTools(() => ({
    model: QWEN3_4B_SUMMARY,
    telemetry: {
      activeProvider: 'vTensor',
      verdantVersion: '2.22.4',
      kernelAcceleration: 'wasm-simd',
      threadPool: { active: true, workerCount: 4, barrierLatencyUs: 15 },
      performanceContract: { verdict: 'OPTIMAL' },
    } as never,
    opfsStorage: {
      supported: true,
      storedModels: [{ modelId: 'qwen3-4b-q4_k_m', byteLength: 2_500_000_000, storageMode: 'full' }],
    } as never,
    lastTokensPerSec: 21.4,
    harnessMode: 'agentic',
  }));

  const calcRes = await tools.calculate.execute!({ expression: '12 * 12' }, {} as never);
  assert.equal(calcRes.result, 144);

  const dtRes = await tools.current_datetime.execute!({}, {} as never);
  assert.equal(typeof dtRes.iso, 'string');
  assert.ok(dtRes.unixMs > 0);

  const rtRes = await tools.inspect_runtime.execute!({}, {} as never);
  assert.equal(rtRes.provider, 'vTensor');
  assert.equal(rtRes.threadPool.workers, 4);
  assert.equal(rtRes.opfs.cachedModels.length, 1);

  const writeRes = await tools.workspace_notes.execute!(
    { action: 'write', title: 'Test Note', content: 'Hello Verdant' },
    {} as never,
  );
  assert.equal(writeRes.saved, true);
  const readRes = await tools.workspace_notes.execute!({ action: 'read', title: 'Test Note' }, {} as never);
  assert.equal(readRes.found, true);
  assert.equal(readRes.note.content, 'Hello Verdant');
  const delRes = await tools.workspace_notes.execute!({ action: 'delete', title: 'Test Note' }, {} as never);
  assert.equal(delRes.deleted, true);

  const statsRes = await tools.analyze_text.execute!({ text: 'alpha beta\ngamma', mode: 'stats' }, {} as never);
  assert.deepEqual(statsRes, { chars: 16, words: 3, lines: 2 });

  const jsonRes = await tools.analyze_text.execute!({ text: '{"a":1,"b":2}', mode: 'json' }, {} as never);
  assert.equal(jsonRes.validJson, true);
  assert.deepEqual(jsonRes.keys, ['a', 'b']);

  const searchRes = await tools.analyze_text.execute!(
    { text: 'line one\nVerdant SIMD\nline three', mode: 'search', query: 'verdant' },
    {} as never,
  );
  assert.equal(searchRes.matchCount, 1);
  assert.equal(searchRes.matches[0].line, 2);
});

test('buildVerdantModelContext and VERDANT_RUNTIME_TIERS orchestrate curated tiers and harness modes cleanly', () => {
  for (const tier of VERDANT_RUNTIME_TIERS) {
    const model = resolveTierCatalogModel(tier.id);
    assert.equal(model.id, tier.modelId);
    assert.equal(findTierForModelId(model.id)?.id, tier.id);
  }

  // Recommend prefers cached managed tier
  assert.equal(recommendVerdantTier(['qwen3-8b-q4_k_m']).id, 'pro');
  assert.equal(recommendVerdantTier(['qwen3-4b-q4_k_m', 'qwen3-8b-q4_k_m']).id, 'balanced');

  const clientTools = createVerdantClientTools(() => ({
    model: null,
    telemetry: {} as never,
    opfsStorage: { supported: false, storedModels: [] } as never,
    lastTokensPerSec: 0,
    harnessMode: 'fast',
  }));

  const fastCtx = buildVerdantModelContext('fast', clientTools, null);
  assert.equal(fastCtx.tools, undefined);
  assert.equal(fastCtx.config?.reasoningEffort, 'low');

  const thinkingCtx = buildVerdantModelContext('thinking', clientTools, null);
  assert.equal(thinkingCtx.tools, undefined);
  assert.equal(thinkingCtx.config?.reasoningEffort, 'high');

  const agenticCtx = buildVerdantModelContext('agentic', clientTools, null);
  assert.ok(agenticCtx.tools && Object.keys(agenticCtx.tools).length === 5);

  const overriddenCtx = buildVerdantModelContext('agentic', clientTools, {
    id: 'test-harness',
    importedAt: 0,
    name: 'Custom Override',
    sourceLabel: 'test.json',
    system: 'Custom system prompt',
    callSettings: { temperature: 0.1 },
  });
  assert.equal(overriddenCtx.system, 'Custom system prompt');
  assert.equal(overriddenCtx.callSettings?.temperature, 0.1);
  assert.ok(overriddenCtx.tools && Object.keys(overriddenCtx.tools).length === 5);
});
