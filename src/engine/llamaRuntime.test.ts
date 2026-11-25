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

