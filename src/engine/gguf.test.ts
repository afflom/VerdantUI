import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { GgmlType, ggmlTensorByteSize, parseGGUF, sliceGgufTensorBytes } from './gguf';

const STORIES_260K_PATH = path.resolve(process.cwd(), 'public/models/stories260K.gguf');

test('parseGGUF parses real stories260K.gguf metadata and 48 tensors accurately', () => {
  const bytes = fs.readFileSync(STORIES_260K_PATH);
  const parsed = parseGGUF(bytes);

  assert.equal(parsed.version, 3);
  assert.equal(parsed.tensors.length, 48);
  assert.equal(parsed.hparams.architecture, 'llama');
  assert.equal(parsed.hparams.embeddingLength, 64);
  assert.equal(parsed.hparams.feedForwardLength, 172);
  assert.equal(parsed.hparams.headCount, 8);
  assert.equal(parsed.hparams.headCountKv, 4);
  assert.equal(parsed.hparams.blockCount, 5);
  assert.equal(parsed.hparams.vocabSize, 512);
  assert.equal(parsed.hparams.primaryQuantization, 'F32');

  const embd = parsed.tensorMap.get('token_embd.weight');
  assert.ok(embd);
  assert.deepEqual(embd.dims, [64, 512]);
  assert.equal(embd.byteSize, 64 * 512 * 4);

  const slice = sliceGgufTensorBytes(parsed, embd);
  assert.equal(slice.byteLength, embd.byteSize);
});

test('ggmlTensorByteSize matches upstream GGML block sizes for F32, F16, Q8_0, and Q4_0', () => {
  assert.equal(ggmlTensorByteSize(GgmlType.F32, [64, 16]), 64 * 16 * 4);
  assert.equal(ggmlTensorByteSize(GgmlType.F16, [64, 16]), 64 * 16 * 2);
  // Q8_0: 34 bytes per 32 elements -> 2 blocks per 64-element row -> 68 bytes/row
  assert.equal(ggmlTensorByteSize(GgmlType.Q8_0, [64, 10]), 680);
  // Q4_0: 18 bytes per 32 elements -> 2 blocks per 64-element row -> 36 bytes/row
  assert.equal(ggmlTensorByteSize(GgmlType.Q4_0, [64, 10]), 360);
});

test('parseGGUF rejects malformed magic header', () => {
  const invalid = new Uint8Array(32);
  invalid.set([0x4e, 0x4f, 0x50, 0x45]); // "NOPE"
  assert.throws(() => parseGGUF(invalid), /Invalid GGUF magic header/);
});
