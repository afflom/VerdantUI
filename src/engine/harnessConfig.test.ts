import test from 'node:test';
import assert from 'node:assert/strict';
import {
  applyCallSettingsToSampling,
  buildModelContextFromHarness,
  matchCatalogModelForHarness,
  parseHarnessConfigJson,
  serializeHarnessConfig,
} from './harnessConfig';
import { DEFAULT_SAMPLING_CONFIG } from './llamaRuntime';

test('buildModelContextFromHarness returns empty object when no harness is imported (zero injected system prompt)', () => {
  const ctx = buildModelContextFromHarness(null);
  assert.deepEqual(ctx, {});
  assert.equal(ctx.system, undefined);
  assert.equal(ctx.callSettings, undefined);
  assert.equal(ctx.config, undefined);
});

test('parseHarnessConfigJson parses official Assistant-UI ModelContext JSON', () => {
  const raw = JSON.stringify({
    name: 'Code Reviewer',
    description: 'Strict diff review harness',
    system: 'Review code diffs concisely and list concrete bugs first.',
    callSettings: {
      temperature: 0.2,
      topP: 0.85,
      maxTokens: 1024,
      seed: 42,
    },
    config: {
      modelName: 'Qwen3-Coder 30B-A3B',
      reasoningEffort: 'medium',
    },
    topK: 32,
    repeatPenalty: 1.08,
    promptMode: 'chatml',
    suggestions: [{ label: 'Audit diff', prompt: 'Review the attached patch for regressions.' }],
  });

  const spec = parseHarnessConfigJson(raw, 'code-reviewer.json');
  assert.equal(spec.name, 'Code Reviewer');
  assert.equal(spec.description, 'Strict diff review harness');
  assert.equal(spec.system, 'Review code diffs concisely and list concrete bugs first.');
  assert.deepEqual(spec.callSettings, {
    temperature: 0.2,
    topP: 0.85,
    maxTokens: 1024,
    seed: 42,
  });
  assert.deepEqual(spec.config, {
    modelName: 'Qwen3-Coder 30B-A3B',
    reasoningEffort: 'medium',
  });
  assert.equal(spec.topK, 32);
  assert.equal(spec.repeatPenalty, 1.08);
  assert.equal(spec.promptMode, 'chatml');
  assert.deepEqual(spec.suggestions, [
    { label: 'Audit diff', prompt: 'Review the attached patch for regressions.' },
  ]);

  const modelCtx = buildModelContextFromHarness(spec);
  assert.equal(modelCtx.system, spec.system);
  assert.equal(modelCtx.callSettings?.temperature, 0.2);
  assert.equal(modelCtx.config?.modelName, 'Qwen3-Coder 30B-A3B');
});

test('parseHarnessConfigJson parses OpenAI Chat Completions / Agent JSON format', () => {
  const raw = JSON.stringify({
    model: 'qwen2.5-coder-7b-instruct-q4_k_m',
    temperature: 0.15,
    top_p: 0.9,
    max_tokens: 512,
    messages: [
      { role: 'system', content: 'You output valid TypeScript only.' },
      { role: 'user', content: 'Ignored sample user turn' },
    ],
  });

  const spec = parseHarnessConfigJson(raw, 'ts-agent.json');
  assert.equal(spec.name, 'ts-agent');
  assert.equal(spec.system, 'You output valid TypeScript only.');
  assert.equal(spec.callSettings?.temperature, 0.15);
  assert.equal(spec.callSettings?.topP, 0.9);
  assert.equal(spec.callSettings?.maxTokens, 512);
  assert.equal(spec.config?.modelName, 'qwen2.5-coder-7b-instruct-q4_k_m');
});

test('parseHarnessConfigJson rejects malformed JSON, non-objects, and empty unrecognized configs', () => {
  assert.throws(() => parseHarnessConfigJson('{not-json'), /Invalid JSON harness file/i);
  assert.throws(() => parseHarnessConfigJson('[1, 2, 3]'), /must be a JSON object/i);
  assert.throws(() => parseHarnessConfigJson('{"foo":"bar"}'), /did not include any recognized ModelContext fields/i);
});

test('applyCallSettingsToSampling merges ModelContext callSettings and harness extensions without clobbering defaults', () => {
  const harness = parseHarnessConfigJson(
    JSON.stringify({
      system: 'Be brief.',
      callSettings: { temperature: 0.1, topP: 0.8, maxTokens: 256 },
      topK: 20,
      repeatPenalty: 1.15,
    }),
  );

  const merged = applyCallSettingsToSampling(DEFAULT_SAMPLING_CONFIG, harness.callSettings, harness);
  assert.equal(merged.temperature, 0.1);
  assert.equal(merged.topP, 0.8);
  assert.equal(merged.maxTokens, 256);
  assert.equal(merged.fullContextOutput, false);
  assert.equal(merged.topK, 20);
  assert.equal(merged.repeatPenalty, 1.15);

  const untouched = applyCallSettingsToSampling(DEFAULT_SAMPLING_CONFIG, undefined, null);
  assert.deepEqual(untouched, DEFAULT_SAMPLING_CONFIG);
});

test('matchCatalogModelForHarness resolves exact IDs and human model names from catalog', () => {
  const byName = parseHarnessConfigJson(
    JSON.stringify({ config: { modelName: 'Qwen3-Coder 30B-A3B' } }),
  );
  const matched = matchCatalogModelForHarness(byName);
  assert.ok(matched);
  assert.equal(matched.id, 'qwen3-coder-30b-a3b-q4_k_m');

  const byId = parseHarnessConfigJson(
    JSON.stringify({ model: 'llama-3.2-3b-instruct-q4_k_m' }),
  );
  const matchedId = matchCatalogModelForHarness(byId);
  assert.ok(matchedId);
  assert.equal(matchedId.id, 'llama-3.2-3b-instruct-q4_k_m');

  assert.equal(matchCatalogModelForHarness(null), null);
});

test('serializeHarnessConfig round-trips cleanly through parseHarnessConfigJson', () => {
  const original = parseHarnessConfigJson(
    JSON.stringify({
      name: 'Roundtrip Harness',
      system: 'Answer in one sentence.',
      callSettings: { temperature: 0.3, topP: 0.92, maxTokens: 200 },
      config: { modelName: 'Llama 3.2 3B' },
    }),
  );

  const json = serializeHarnessConfig(original, DEFAULT_SAMPLING_CONFIG, 'Llama 3.2 3B');
  const reparsed = parseHarnessConfigJson(json, 'roundtrip.json');
  assert.equal(reparsed.name, 'Roundtrip Harness');
  assert.equal(reparsed.system, 'Answer in one sentence.');
  assert.equal(reparsed.callSettings?.temperature, 0.3);
  assert.equal(reparsed.callSettings?.topP, 0.92);
  assert.equal(reparsed.callSettings?.maxTokens, 200);
  assert.equal(reparsed.config?.modelName, 'Llama 3.2 3B');
});
