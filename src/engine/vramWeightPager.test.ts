import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { parseGGUF, parseGgufFromBacking } from './gguf';
import { LlamaVTensorRuntime } from './llamaRuntime';
import { MemoryBackingStore } from './vramBackingStore';

const STORIES_260K_PATH = path.resolve(process.cwd(), 'public/models/stories260K.gguf');

test('parseGgufFromBacking matches full-buffer parseGGUF metadata and tensor offsets identically', () => {
  const bytes = new Uint8Array(fs.readFileSync(STORIES_260K_PATH));
  const fullParsed = parseGGUF(bytes);
  const backing = new MemoryBackingStore('stories260k-header-test', bytes);
  const headerParsed = parseGgufFromBacking(backing);

  assert.equal(headerParsed.version, fullParsed.version);
  assert.equal(headerParsed.dataOffset, fullParsed.dataOffset);
  assert.equal(headerParsed.fileByteLength, fullParsed.fileByteLength);
  assert.equal(headerParsed.tensors.length, fullParsed.tensors.length);
  assert.deepEqual(headerParsed.hparams, fullParsed.hparams);
  for (let i = 0; i < fullParsed.tensors.length; i++) {
    assert.deepEqual(headerParsed.tensors[i], fullParsed.tensors[i]);
  }
  backing.close();
});

test('OPFS-paged vRAM weight residency produces BIT-EXACT identical greedy tokens to resident-all with bounded RAM and zero CPU fallbacks', async () => {
  const bytes = new Uint8Array(fs.readFileSync(STORIES_260K_PATH));
  const prompt = 'Once upon a time, there was a little girl named Lily.';
  const sampling = {
    temperature: 0,
    topP: 1,
    topK: 1,
    repeatPenalty: 1.0,
    maxTokens: 20,
    seed: 42,
  };

  // 1. Baseline: resident-all (auto)
  const rtResident = new LlamaVTensorRuntime();
  rtResident.loadFromBacking(
    'stories260k',
    'Stories 260K',
    'test://stories260K',
    new MemoryBackingStore('stories260k', bytes),
    { maxContextCap: 256, weightResidencyPolicy: 'auto' },
  );
  const residentTokens: number[] = [];
  let residentText = '';
  for await (const ev of rtResident.generate(prompt, sampling)) {
    if (!ev.finished || ev.textDelta.length > 0) {
      residentTokens.push(ev.tokenId);
    }
    residentText = ev.accumulatedText;
  }
  const telResident = rtResident.getTelemetry();
  assert.equal(telResident.cpuFallbackOps, 0);
  assert.equal(telResident.vramWeights?.residencyMode, 'resident-all');

  // 2. OPFS-paged mode (bounded layer frame pool + streamed single-row embeddings + striped LM head)
  const rtPaged = new LlamaVTensorRuntime();
  rtPaged.loadFromBacking(
    'stories260k',
    'Stories 260K',
    'test://stories260K',
    new MemoryBackingStore('stories260k', bytes),
    { maxContextCap: 256, weightResidencyPolicy: 'paged' },
  );
  const pagedTokens: number[] = [];
  let pagedText = '';
  for await (const ev of rtPaged.generate(prompt, sampling)) {
    if (!ev.finished || ev.textDelta.length > 0) {
      pagedTokens.push(ev.tokenId);
    }
    pagedText = ev.accumulatedText;
  }
  const telPaged = rtPaged.getTelemetry();
  assert.equal(telPaged.cpuFallbackOps, 0);
  assert.ok(telPaged.vramWeights, 'Expected vramWeights telemetry');
  assert.equal(telPaged.vramWeights.residencyMode, 'opfs-paged');
  assert.equal(telPaged.vramWeights.streamEmbeddings, true);
  assert.equal(telPaged.vramWeights.streamLmHead, true);
  assert.ok(
    telPaged.vramWeights.layerFrameSlots < telPaged.vramWeights.totalLayers,
    `Expected bounded layer frame pool < ${telPaged.vramWeights.totalLayers}, got ${telPaged.vramWeights.layerFrameSlots}`,
  );
  assert.ok(telPaged.vramWeights.layerFaults > 0, 'Expected layer faults during streaming');
  assert.ok(telPaged.vramWeights.slotReclaims > 0, 'Expected zero-alloc slot reclaims');
  assert.ok(
    telPaged.vramWeights.residentWeightBytes < telResident.vramWeights!.residentWeightBytes,
    `Expected paged resident RAM (${telPaged.vramWeights.residentWeightBytes}) < resident-all (${telResident.vramWeights!.residentWeightBytes})`,
  );

  // Bit-exact greedy parity proof!
  assert.deepEqual(pagedTokens, residentTokens);
  assert.equal(pagedText, residentText);
  assert.ok(pagedText.length > 10);

  // 3. Live toggle paged -> auto -> paged without reloading file
  rtPaged.setWeightResidencyPolicy('auto');
  assert.equal(rtPaged.getTelemetry().vramWeights?.residencyMode, 'resident-all');
  rtPaged.setWeightResidencyPolicy('paged');
  assert.equal(rtPaged.getTelemetry().vramWeights?.residencyMode, 'opfs-paged');

  let liveText = '';
  for await (const ev of rtPaged.generate(prompt, sampling)) {
    liveText = ev.accumulatedText;
  }
  assert.equal(liveText, residentText);

  rtResident.unload();
  rtPaged.unload();
});
