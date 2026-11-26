import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import type { ThreadMessage } from '@assistant-ui/react';
import {
  formatMessagesForLlama,
  formatSpeculativePrefixForLlama,
  hasCompleteToolCallTag,
  isThinkingCapableModel,
  parseAssistantStreamParts,
  shouldPrefillEmptyThink,
  shouldPrefillOpenThink,
} from './assistantAdapter';
import {
  ContainerGrantMatrix,
  UNIFIED_VERDANT_SYSTEM_PROMPT,
  VerdantCognitiveObserver,
  buildUnifiedVerdantModelContext,
  computePredictionSurpriseQ10,
  createVerdantClientTools,
  dotCognitiveQ10,
  evaluateMathExpression,
  formatDirectOperatorSummary,
  harmonicIntegrateBelief,
  observeInputSensoryState,
} from './cognitiveObserver';
import { LlamaVTensorRuntime, type LoadedModelSummary } from './llamaRuntime';

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

  for await (const _ of sharedRuntime.generate(
    'One sunny morning, a blue cat climbed the tall apple tree.',
    greedy,
    undefined,
    undefined,
    'thread-alpha',
  )) {
    // warm Thread A
  }

  // Abort mid-stream on Thread A to verify KV truncation hygiene
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
    if (abortedTokens >= 2) abortCtrl.abort();
  }

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

  assert.equal(sharedThreadBText, freshThreadBText);
  assert.ok(sharedThreadBText.length > 5);
});

test('ContainerGrantMatrix enforces FK-07 denial-wins capability lattice', () => {
  const grants = new ContainerGrantMatrix();
  assert.equal(grants.isAllowed('math-eval'), true);
  assert.equal(grants.isAllowed('network-weights'), true);

  grants.deny('math-eval');
  assert.equal(grants.isAllowed('math-eval'), false);

  // Subsequent grant MUST NOT override explicit denial (FK-07)
  grants.grant('math-eval');
  assert.equal(grants.isAllowed('math-eval'), false);

  grants.reset('math-eval');
  assert.equal(grants.isAllowed('math-eval'), true);
});

test('Exact integer Walsh-Hadamard harmonic integration and Predictive Coding surprise preserve Q10 bounds', () => {
  const prior = [256, 256, 256, 256, 256, 256, 256, 256] as const;
  const obs = [960, 896, 768, 128, 640, 320, 768, 64] as const;

  const surprise = computePredictionSurpriseQ10(prior, obs);
  assert.ok(surprise > 200 && surprise <= 1024);

  const next = harmonicIntegrateBelief(prior, obs, surprise);
  for (const v of next) {
    assert.ok(Number.isInteger(v) && v >= 0 && v <= 1024);
  }
  // Exact integer determinism
  assert.deepEqual(next, harmonicIntegrateBelief(prior, obs, surprise));

  const resonance = dotCognitiveQ10(next, obs);
  assert.ok(Number.isInteger(resonance) && resonance > 0);
});

test('VerdantCognitiveObserver settles LOD0 verified routes, adaptive System-1/System-2 neural depth, retry escalation, and thread isolation', () => {
  const observer = new VerdantCognitiveObserver();

  // 1. Cold browser (no real model loaded): pure math settles deterministically at LOD0 (0 bytes downloaded)
  const mathRes = observer.selectRoute({
    threadId: 'thread-1',
    messages: [makeUserMsg('Calculate (144 * 37) + sqrt(625)')],
    latestUserText: 'Calculate (144 * 37) + sqrt(625)',
    hasAttachment: false,
    activeModel: null,
    cachedModelIds: [],
    isToolContinuationStep: false,
    hasImportedHarness: false,
  });
  assert.equal(mathRes.certificate.kind, 'lod0-verified');
  assert.equal(mathRes.certificate.settlementRule, 'REQ-SETTLEMENT-LOD0');
  assert.equal(mathRes.certificate.directMatch?.toolName, 'calculate');
  assert.equal(mathRes.certificate.attachTools, false);

  // 2. Cold browser + open-ended question promotes seamlessly to default neural route
  const coldNeural = observer.selectRoute({
    threadId: 'thread-2',
    messages: [makeUserMsg('Say hello in French.')],
    latestUserText: 'Say hello in French.',
    hasAttachment: false,
    activeModel: null,
    cachedModelIds: [],
    isToolContinuationStep: false,
    hasImportedHarness: false,
  });
  assert.equal(coldNeural.certificate.kind, 'neural-on-demand');
  assert.equal(coldNeural.certificate.settlementRule, 'REQ-SETTLEMENT-PROMOTE');
  assert.equal(coldNeural.certificate.thinkingPolicy, 'off');
  assert.equal(coldNeural.certificate.attachTools, false);

  // 3. Same prompt re-submitted (user clicks Regenerate) -> prediction error escalates deductive depth!
  const retryRes = observer.selectRoute({
    threadId: 'thread-2',
    messages: [makeUserMsg('Say hello in French.')],
    latestUserText: 'Say hello in French.',
    hasAttachment: false,
    activeModel: QWEN3_4B_SUMMARY,
    cachedModelIds: ['qwen3-4b-q4_k_m'],
    isToolContinuationStep: false,
    hasImportedHarness: false,
  });
  assert.equal(retryRes.certificate.escalatedFromRetry, true);
  assert.equal(retryRes.certificate.thinkingPolicy, 'on');
  assert.equal(retryRes.certificate.kind, 'neural-resident');

  // 4. Deep algorithmic / proof prompt automatically activates System-2 reasoning (`thinkingPolicy: 'on'`)
  const deepRes = observer.selectRoute({
    threadId: 'thread-3',
    messages: [
      makeUserMsg(
        'Design an O(1) LRU cache with second-chance frequency protection and prove why eviction stays bounded.',
      ),
    ],
    latestUserText:
      'Design an O(1) LRU cache with second-chance frequency protection and prove why eviction stays bounded.',
    hasAttachment: false,
    activeModel: QWEN3_4B_SUMMARY,
    cachedModelIds: ['qwen3-4b-q4_k_m'],
    isToolContinuationStep: false,
    hasImportedHarness: false,
  });
  assert.equal(deepRes.certificate.thinkingPolicy, 'on');
  assert.equal(deepRes.certificate.attachTools, false);

  // 5. Compound task requiring math + local notes activates grounded tool composition
  const compoundRes = observer.selectRoute({
    threadId: 'thread-4',
    messages: [
      makeUserMsg(
        'Calculate (144 * 37) + sqrt(625) and save the result to a workspace note titled "Calc".',
      ),
    ],
    latestUserText:
      'Calculate (144 * 37) + sqrt(625) and save the result to a workspace note titled "Calc".',
    hasAttachment: false,
    activeModel: QWEN3_4B_SUMMARY,
    cachedModelIds: ['qwen3-4b-q4_k_m'],
    isToolContinuationStep: false,
    hasImportedHarness: false,
  });
  assert.equal(compoundRes.certificate.kind, 'neural-resident');
  assert.equal(compoundRes.certificate.attachTools, true);

  // 6. Strict per-thread cognitive isolation: thread-1 turn count & beliefs unaffected by thread-2/3/4
  assert.equal(observer.getThreadSnapshot('thread-1').turnCount, 1);
  assert.equal(observer.getThreadSnapshot('thread-2').retryEscalationLevel, 1);
  assert.equal(observer.getThreadSnapshot('thread-3').retryEscalationLevel, 0);

  // 7. Denying math-eval via ContainerGrantMatrix refuses LOD0 math and routes to neural instead
  observer.grants.deny('math-eval');
  const deniedMath = observer.selectRoute({
    threadId: 'thread-5',
    messages: [makeUserMsg('Calculate 12 * 12')],
    latestUserText: 'Calculate 12 * 12',
    hasAttachment: false,
    activeModel: QWEN3_4B_SUMMARY,
    cachedModelIds: ['qwen3-4b-q4_k_m'],
    isToolContinuationStep: false,
    hasImportedHarness: false,
  });
  assert.equal(deniedMath.certificate.kind, 'neural-resident');
});

test('Unified prompt formatting, Qwen3 thinking control, tool parsing, and speculative prefill parity hold across all postures', () => {
  assert.equal(isThinkingCapableModel(QWEN3_4B_SUMMARY), true);
  assert.equal(isThinkingCapableModel(QWEN25_15B_SUMMARY), false);
  assert.equal(shouldPrefillEmptyThink(QWEN3_4B_SUMMARY, 'auto', 'off'), true);
  assert.equal(shouldPrefillOpenThink(QWEN3_4B_SUMMARY, 'auto', 'on'), true);

  const msgs = [makeUserMsg('Explain WebAssembly SIMD briefly.')];
  const directPrompt = formatMessagesForLlama(
    msgs,
    QWEN3_4B_SUMMARY,
    'auto',
    UNIFIED_VERDANT_SYSTEM_PROMPT,
    { thinkingMode: 'off' },
  );
  assert.ok(directPrompt.endsWith('<|im_start|>assistant\n<think>\n\n</think>\n\n'));

  const deepPrompt = formatMessagesForLlama(
    msgs,
    QWEN3_4B_SUMMARY,
    'auto',
    UNIFIED_VERDANT_SYSTEM_PROMPT,
    { thinkingMode: 'on' },
  );
  assert.ok(deepPrompt.endsWith('<|im_start|>assistant\n<think>\n'));

  // Speculative boot & draft prefixes are strict byte prefixes of final unified prompts
  const bootSpec = formatSpeculativePrefixForLlama(
    [],
    '',
    QWEN3_4B_SUMMARY,
    'auto',
    UNIFIED_VERDANT_SYSTEM_PROMPT,
  );
  const draftSpec = formatSpeculativePrefixForLlama(
    [],
    'Explain WebAssembly ',
    QWEN3_4B_SUMMARY,
    'auto',
    UNIFIED_VERDANT_SYSTEM_PROMPT,
  );
  assert.ok(directPrompt.startsWith(bootSpec));
  assert.ok(directPrompt.startsWith(draftSpec));
  assert.ok(deepPrompt.startsWith(bootSpec));
  assert.ok(deepPrompt.startsWith(draftSpec));

  // Streaming parser separates prefilled <think> cleanly and strips hallucinated <tool_response>
  const reasoned = parseAssistantStreamParts('Check edge cases.\n</think>\n\nAnswer: 42.', true, {
    startsInReasoning: true,
  });
  assert.equal(reasoned[0].type, 'reasoning');
  assert.equal(reasoned[1].type, 'text');
  assert.equal(reasoned[1].text, 'Answer: 42.');

  const rawTool =
    '<tool_call>{"name":"calculate","arguments":{"expression":"6*7"}}</tool_call><tool_response>999</tool_response>';
  assert.equal(hasCompleteToolCallTag(rawTool), true);
  const toolParts = parseAssistantStreamParts(rawTool, true, { parseToolCalls: true });
  assert.equal(toolParts.length, 1);
  assert.equal(toolParts[0].type, 'tool-call');
});

test('evaluateMathExpression, all 5 client tools, and unified ModelContext run deterministically with zero eval', async () => {
  assert.equal(evaluateMathExpression('(144 * 37) + sqrt(625)'), 5353);
  assert.equal(evaluateMathExpression('2 ^ 10 - 24'), 1000);
  assert.throws(() => evaluateMathExpression('1 / 0'), /Division by zero/);
  assert.throws(() => evaluateMathExpression('process.exit(1)'), /Unsupported character|Unknown identifier/);

  const tools = createVerdantClientTools(() => ({
    model: QWEN3_4B_SUMMARY,
    telemetry: {
      activeProvider: 'vTensor',
      verdantVersion: '2.22.4',
      kernelAcceleration: 'wasm-simd',
      threadPool: { active: true, workerCount: 4, barrierLatencyUs: 12 },
      performanceContract: { verdict: 'UPHELD' },
    } as never,
    opfsStorage: {
      supported: true,
      usageBytes: 0,
      quotaBytes: 0,
      storedModels: [{ modelId: 'qwen3-4b-q4_k_m', byteLength: 2_500_000_000, storageMode: 'full' }],
      partialModels: [],
    } as never,
    lastTokensPerSec: 22.0,
  }));

  const calc = await tools.calculate.execute!({ expression: '(144 * 37) + sqrt(625)' }, {} as never);
  assert.equal(calc.result, 5353);
  assert.ok(formatDirectOperatorSummary('calculate', calc).includes('5,353'));

  const rt = await tools.inspect_runtime.execute!({}, {} as never);
  assert.equal(rt.provider, 'vTensor');
  assert.ok(formatDirectOperatorSummary('inspect_runtime', rt).includes('4 WASM SIMD workers'));

  const dt = await tools.current_datetime.execute!({}, {} as never);
  assert.ok(dt.unixMs > 0);

  await tools.workspace_notes.execute!(
    { action: 'write', title: 'Observer', content: 'Coherent' },
    {} as never,
  );
  const readBack = await tools.workspace_notes.execute!(
    { action: 'read', title: 'Observer' },
    {} as never,
  );
  assert.equal(readBack.found, true);
  assert.equal(readBack.note.content, 'Coherent');
  await tools.workspace_notes.execute!({ action: 'delete', title: 'Observer' }, {} as never);

  const jsonCheck = await tools.analyze_text.execute!(
    { text: '{"ok":true}', mode: 'json' },
    {} as never,
  );
  assert.equal(jsonCheck.validJson, true);

  const obs = observeInputSensoryState([], 'Inspect my local Verdant runtime', false);
  assert.equal(obs.directMatch?.toolName, 'inspect_runtime');

  const ctx = buildUnifiedVerdantModelContext(tools, null, { temperature: 0.4, topP: 0.85 });
  assert.equal(ctx.system, UNIFIED_VERDANT_SYSTEM_PROMPT);
  assert.equal(ctx.callSettings?.temperature, 0.4);
  assert.ok(ctx.tools && Object.keys(ctx.tools).length === 5);

  // When Model Stream is disabled (`network-weights` denied on cold browser), LOD0 verified routes
  // still settle instantaneously while cold network promotion is refused by FK-07.
  const coldObserver = new VerdantCognitiveObserver();
  coldObserver.grants.deny('network-weights');
  const coldMath = coldObserver.selectRoute({
    threadId: 'cold-stream-off',
    messages: [makeUserMsg('Calculate 9 * 9')],
    latestUserText: 'Calculate 9 * 9',
    hasAttachment: false,
    activeModel: null,
    cachedModelIds: [],
    isToolContinuationStep: false,
    hasImportedHarness: false,
  });
  assert.equal(coldMath.certificate.kind, 'lod0-verified');
  assert.equal(coldObserver.grants.isAllowed('network-weights'), false);
});
