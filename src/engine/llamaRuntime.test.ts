import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { LlamaVTensorRuntime } from './llamaRuntime';
import { GgmlOp } from './vtensorProvider';

const STORIES_260K_PATH = path.resolve(process.cwd(), 'public/models/stories260K.gguf');

test('LlamaVTensorRuntime loads stories260K.gguf into vTensor buffers and generates coherent tokens with zero CPU fallback', async () => {
  const bytes = fs.readFileSync(STORIES_260K_PATH);
  const runtime = new LlamaVTensorRuntime();

  const summary = runtime.loadFromBuffer(
    'stories260k-f32',
    'TinyLLaMA Stories 260K',
    '/models/stories260K.gguf',
    bytes,
  );

  assert.equal(summary.tensorCount, 48);
  assert.equal(summary.hparams.blockCount, 5);

  let text = '';
  let generatedCount = 0;

  for await (const ev of runtime.generate(
    'Once upon a time',
    {
      temperature: 0,
      topP: 1,
      topK: 1,
      repeatPenalty: 1.0,
      maxTokens: 24,
      seed: 42,
    },
  )) {
    text = ev.accumulatedText;
    generatedCount = ev.generatedTokens;
  }

  assert.ok(generatedCount > 5, `Expected >5 generated tokens, got ${generatedCount}`);
  assert.ok(text.trim().length > 10, `Expected readable continuation text, got "${text}"`);
  // Verify the story continuation contains genuine English words produced by stories260K
  assert.match(text, /[a-zA-Z]{3,}/);

  const telemetry = runtime.getTelemetry();
  assert.equal(telemetry.activeProvider, 'vTensor');
  assert.equal(telemetry.cpuFallbackOps, 0);
  assert.ok(telemetry.totalGraphRuns > 5);
  assert.ok(telemetry.totalOpsExecuted > 100);
  assert.ok(telemetry.buffers.length >= 6); // weights, kv_hot, kv_warm, kv_pagetable, compute_scratch, rope_tables
  const bufferUsages = new Set(telemetry.buffers.map((b) => b.usage));
  assert.ok(bufferUsages.has('weights') && bufferUsages.has('kv_cache') && bufferUsages.has('compute'));
  assert.ok(telemetry.compiledGraphReuses > 5);
  assert.ok(telemetry.vramKv !== null);
  assert.ok(telemetry.vramKv.hotResidentPages >= 1);

  const executedOps = new Set(telemetry.opMetrics.map((m) => m.op));
  assert.ok(executedOps.has(GgmlOp.GET_ROWS));
  assert.ok(executedOps.has(GgmlOp.RMS_NORM));
  assert.ok(executedOps.has(GgmlOp.MUL_MAT));
  assert.ok(executedOps.has(GgmlOp.ROPE));
  assert.ok(executedOps.has(GgmlOp.SET_ROWS));
  assert.ok(executedOps.has(GgmlOp.FLASH_ATTN_EXT));
  assert.ok(executedOps.has(GgmlOp.SWIGLU));
  assert.ok(executedOps.has(GgmlOp.ADD));
});

test('LlamaVTensorRuntime rejects out-of-bounds context positions explicitly', () => {
  const bytes = fs.readFileSync(STORIES_260K_PATH);
  const runtime = new LlamaVTensorRuntime();
  const summary = runtime.loadFromBuffer(
    'stories260k-f32',
    'TinyLLaMA Stories 260K',
    '/models/stories260K.gguf',
    bytes,
  );

  assert.throws(
    () => runtime.evalToken(summary.hparams.bosTokenId, -1),
    /exceeds vTensor KV-cache context window/,
  );
  assert.throws(
    () => runtime.evalToken(summary.hparams.bosTokenId, summary.hparams.contextLength),
    /exceeds vTensor KV-cache context window/,
  );
});

test('LlamaVTensorRuntime KV-cache prefix reuse matches fresh evaluation bit-for-bit', async () => {
  const bytes = fs.readFileSync(STORIES_260K_PATH);
  const runtime = new LlamaVTensorRuntime();
  runtime.loadFromBuffer('stories260k-f32', 'TinyLLaMA Stories 260K', '/models/stories260K.gguf', bytes);

  const greedy = { temperature: 0, topP: 1, topK: 1, repeatPenalty: 1.0, maxTokens: 16 };

  // Warm cache with prefix
  for await (const _ of runtime.generate('Once upon a time', greedy)) {
    // consume
  }

  // Run extended prompt reusing KV prefix
  let reusedCount = 0;
  let reusedText = '';
  for await (const ev of runtime.generate(
    'Once upon a time, a small silver bird',
    greedy,
    undefined,
    (p) => {
      reusedCount = p.reusedPrefixTokens;
    },
  )) {
    reusedText = ev.accumulatedText;
  }

  assert.ok(reusedCount >= 3, `Expected KV prefix reuse >= 3 tokens, got ${reusedCount}`);

  // Reset KV cache and run identical extended prompt from scratch
  runtime.resetKvCache();
  let freshText = '';
  for await (const ev of runtime.generate('Once upon a time, a small silver bird', greedy)) {
    freshText = ev.accumulatedText;
  }

  assert.equal(reusedText, freshText);
});

test('LlamaVTensorRuntime WASM SIMD batched prefill matches Scalar JS single-token greedy output exactly', async () => {
  const { VTensorBackendRegistry } = await import('./vtensorProvider');
  const bytes = fs.readFileSync(STORIES_260K_PATH);

  const wasmRuntime = new LlamaVTensorRuntime(new VTensorBackendRegistry({ enableWasmSimd: true }));
  const scalarRuntime = new LlamaVTensorRuntime(new VTensorBackendRegistry({ enableWasmSimd: false }));

  wasmRuntime.loadFromBuffer('stories260k-f32', 'TinyLLaMA Stories 260K', '/models/stories260K.gguf', bytes);
  scalarRuntime.loadFromBuffer('stories260k-f32', 'TinyLLaMA Stories 260K', '/models/stories260K.gguf', bytes);

  const prompt = 'Once upon a time, there was a little girl named Lily who lived in a small house';
  const greedy = { temperature: 0, topP: 1, topK: 1, repeatPenalty: 1.0, maxTokens: 20 };

  let wasmText = '';
  for await (const ev of wasmRuntime.generate(prompt, greedy)) wasmText = ev.accumulatedText;

  let scalarText = '';
  for await (const ev of scalarRuntime.generate(prompt, greedy)) scalarText = ev.accumulatedText;

  assert.match(
    wasmRuntime.getTelemetry().kernelAcceleration,
    /^WASM (Relaxed SIMD \(VNNI\)|SIMD128 \(Int8\))$/,
  );
  assert.equal(scalarRuntime.getTelemetry().kernelAcceleration, 'Scalar JS');
  assert.equal(wasmText, scalarText);

  // Verify Prism Performance Contract is attached, evaluated, and supports live precisionMode switching
  const contract = wasmRuntime.getTelemetry().performanceContract;
  assert.ok(contract !== null);
  assert.ok(contract.rooflineDecodeTokSec > 0);
  assert.ok(contract.observedDecodeTokSec > 0);
  assert.ok(contract.verdict === 'UPHELD' || contract.verdict === 'EXCEEDS_ESTIMATE');

  const strictContract = wasmRuntime.setPrecisionMode('strict-f64');
  assert.equal(strictContract?.precisionMode, 'strict-f64');
  assert.equal(wasmRuntime.getPrecisionMode(), 'strict-f64');
});

test('LlamaVTensorRuntime supports full context-length input (100%) and full context-length output without artificial caps', async () => {
  const bytes = fs.readFileSync(STORIES_260K_PATH);
  const runtime = new LlamaVTensorRuntime();
  const summary = runtime.loadFromBuffer(
    'stories260k-f32',
    'TinyLLaMA Stories 260K',
    '/models/stories260K.gguf',
    bytes,
  );
  const maxCtx = summary.maxContextWindow;
  assert.equal(maxCtx, summary.hparams.contextLength);
  assert.equal(maxCtx, 2048);

  // 1. Build a prompt that exceeds the old 1024 cap & fills ~98% of the full 2048 context window
  const phrase = 'Once upon a time there was a little silver bird in a quiet forest. ';
  const phraseToks = Math.max(1, runtime.countTokens(phrase) - 1);
  const repeats = Math.max(1, Math.floor((maxCtx - 24) / phraseToks));
  const longPrompt = phrase.repeat(repeats);
  const nearFullTokens = runtime.countTokens(longPrompt);
  assert.ok(
    nearFullTokens > 1024 && nearFullTokens <= maxCtx - 8,
    `Expected near-full prompt tokens (>1024 and <=${maxCtx - 8}), got ${nearFullTokens}`,
  );

  let lastPromptTokens = 0;
  let lastTruncated = -1;
  let lastGen = 0;
  for await (const ev of runtime.generate(longPrompt, {
    temperature: 0,
    topP: 1,
    topK: 1,
    repeatPenalty: 1.0,
    maxTokens: maxCtx,
    fullContextOutput: true,
    seed: 1,
  })) {
    lastPromptTokens = ev.promptTokens;
    lastTruncated = ev.promptTruncatedTokens ?? 0;
    lastGen = ev.generatedTokens;
  }
  assert.equal(lastPromptTokens, nearFullTokens, 'Zero prompt truncation for prompts fitting inside maxContextWindow');
  assert.equal(lastTruncated, 0);
  assert.ok(lastGen >= 1, 'Model emits tokens from near-full prompt without KV overflow');

  // 2. Prompt at 100% of maxContextWindow (exact 2048-token boundary): zero truncation and safely emits 1 token without OOB
  const needed = Math.max(1, maxCtx - nearFullTokens);
  const exactFullPrompt = longPrompt + ' a'.repeat(needed);
  const exactCount = runtime.countTokens(exactFullPrompt);
  assert.ok(exactCount >= maxCtx, `Expected exact boundary prompt >= ${maxCtx} tokens, got ${exactCount}`);
  let fullBoundaryPromptTokens = 0;
  let fullBoundaryGen = 0;
  for await (const ev of runtime.generate(exactFullPrompt, {
    temperature: 0,
    topP: 1,
    topK: 1,
    repeatPenalty: 1.0,
    maxTokens: maxCtx,
    fullContextOutput: true,
    seed: 1,
  })) {
    fullBoundaryPromptTokens = ev.promptTokens;
    fullBoundaryGen = ev.generatedTokens;
  }
  assert.equal(fullBoundaryPromptTokens, Math.min(exactCount, maxCtx));
  assert.equal(fullBoundaryGen, 1);
  assert.equal(runtime.getTelemetry().cpuFallbackOps, 0);

  // 3. Short prompt with fullContextOutput: true generates beyond custom 16-token cap (up to EOS or full 2048 ctx)
  runtime.resetKvCache();
  let fullOutGen = 0;
  for await (const ev of runtime.generate('Once', {
    temperature: 0.7,
    topP: 0.95,
    topK: 40,
    repeatPenalty: 1.05,
    maxTokens: 16, // Ignored when fullContextOutput: true
    fullContextOutput: true,
    seed: 7,
  })) {
    fullOutGen = ev.generatedTokens;
  }
  assert.ok(fullOutGen > 16, `Expected fullContextOutput to exceed custom 16-token cap, got ${fullOutGen}`);
});

test('Bounded RoPE dynamic synthesis (pos >= 4096) matches full precomputed buildRopeTables bit-for-bit', async () => {
  const { buildRopeTables, fillRopePositionCosSin, verdantRopePrecomputed } = await import(
    './verdantKernels'
  );
  const fullTables = buildRopeTables(5000, 64, 1000000, 1.0);
  const boundedTables = buildRopeTables(4096, 64, 1000000, 1.0);

  for (const pos of [4096, 4097, 4500, 4999]) {
    const cosDyn = new Float32Array(32);
    const sinDyn = new Float32Array(32);
    fillRopePositionCosSin(pos, boundedTables, cosDyn, sinDyn, 0);
    for (let i = 0; i < 32; i++) {
      assert.equal(cosDyn[i], fullTables.cos[pos * 32 + i]);
      assert.equal(sinDyn[i], fullTables.sin[pos * 32 + i]);
    }

    for (const neox of [false, true]) {
      const vecFull = new Float32Array(128);
      const vecBounded = new Float32Array(128);
      for (let i = 0; i < 128; i++) {
        const v = Math.fround(Math.sin(i * 0.37 + pos * 0.01));
        vecFull[i] = v;
        vecBounded[i] = v;
      }
      verdantRopePrecomputed(vecFull, 2, 64, pos, fullTables, neox);
      verdantRopePrecomputed(vecBounded, 2, 64, pos, boundedTables, neox);
      assert.deepEqual(vecBounded, vecFull);
    }
  }
});


