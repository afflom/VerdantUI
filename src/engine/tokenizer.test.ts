import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { parseGGUF, type ParsedGgufModel } from './gguf';
import { GgufTokenizer } from './tokenizer';

const STORIES_260K_PATH = path.resolve(process.cwd(), 'public/models/stories260K.gguf');

test('GgufTokenizer round-trips ASCII text cleanly on stories260K.gguf', () => {
  const bytes = fs.readFileSync(STORIES_260K_PATH);
  const parsed = parseGGUF(bytes);
  const tokenizer = new GgufTokenizer(parsed);

  const tokens = tokenizer.encode('Once upon a time', true);
  assert.equal(tokens[0], parsed.hparams.bosTokenId);
  assert.ok(tokens.length > 2);

  const decoded = tokenizer.decode(tokens);
  assert.equal(decoded, 'Once upon a time');
});

test('GgufTokenizer createStreamDecoder buffers split multi-byte UTF-8 <0xXX> fallback tokens without corruption', () => {
  // Synthetic minimal vocab containing SentencePiece tokens + <0xXX> byte fallback entries for:
  // "café" (é = 0xC3 0xA9) and "✨" (0xE2 0x9C 0xA8) and "🚀" (0xF0 0x9F 0x9A 0x80)
  const vocab = [
    '<unk>',
    '<s>',
    '</s>',
    '\u2581caf',
    '<0xC3>',
    '<0xA9>',
    '\u2581',
    '<0xE2>',
    '<0x9C>',
    '<0xA8>',
    '<0xF0>',
    '<0x9F>',
    '<0x9A>',
    '<0x80>',
  ];

  const mockModel: ParsedGgufModel = {
    version: 3,
    alignment: 32,
    dataOffset: 0,
    fileByteLength: 0,
    metadata: {
      'tokenizer.ggml.tokens': vocab,
      'tokenizer.ggml.scores': vocab.map(() => 0),
    },
    tensors: [],
    tensorMap: new Map(),
    rawBytes: new Uint8Array(0),
    hparams: {
      architecture: 'llama',
      name: 'mock-tokenizer',
      contextLength: 256,
      embeddingLength: 64,
      blockCount: 1,
      feedForwardLength: 128,
      headCount: 4,
      headCountKv: 4,
      headDim: 16,
      rmsNormEps: 1e-5,
      ropeFreqBase: 10000,
      ropeFreqScale: 1.0,
      ropeDimensionCount: 16,
      ropeType: 'norm',
      vocabSize: vocab.length,
      bosTokenId: 1,
      eosTokenId: 2,
      unkTokenId: 0,
      addBosToken: true,
      tokenizerModel: 'llama',
      preTokenizer: 'default',
      chatTemplate: null,
      primaryQuantization: 'F32',
    },
  };

  const tokenizer = new GgufTokenizer(mockModel);
  const stream = tokenizer.createStreamDecoder();

  const sequence = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 2];
  const chunks: string[] = [];
  for (const id of sequence) {
    const piece = stream.push(id);
    if (piece) chunks.push(piece);
  }
  const tail = stream.flush();
  if (tail) chunks.push(tail);

  const combined = chunks.join('');
  assert.equal(combined, 'café ✨🚀');
  assert.ok(!combined.includes('\uFFFD'), 'Should never emit U+FFFD replacement characters for valid split UTF-8');
});

test('GgufTokenizer GPT-2 byte-level BPE handles atomic special tokens, merges, and multi-byte UTF-8', () => {
  // GPT-2 byte encodings: space='Ġ' (\u0120), newline='Ċ' (\u010A), 'é'=0xC3 0xA9 -> 'Ã' (\u00C3) + '©' (\u00A9)
  const vocab = [
    '<|endoftext|>', // 0 (control)
    '<|im_start|>',  // 1 (control)
    '<|im_end|>',    // 2 (control)
    'u', 's', 'e', 'r', 'us', 'user',
    '\u010A',        // 9: '\n'
    'H', 'i', 'Hi',
    '\u0120',        // 13: ' '
    'c', 'a', 'f', '\u00C3', '\u00A9',
  ];
  const tokenTypes = vocab.map((_, idx) => (idx <= 2 ? 3 : 1));
  const merges = ['u s', 'us e', 'use r', 'H i'];

  const mockGpt2: ParsedGgufModel = {
    version: 3,
    alignment: 32,
    dataOffset: 0,
    fileByteLength: 0,
    metadata: {
      'tokenizer.ggml.tokens': vocab,
      'tokenizer.ggml.token_type': tokenTypes,
      'tokenizer.ggml.merges': merges,
      'tokenizer.ggml.add_bos_token': false,
    },
    tensors: [],
    tensorMap: new Map(),
    rawBytes: new Uint8Array(0),
    hparams: {
      architecture: 'llama',
      name: 'mock-gpt2',
      contextLength: 256,
      embeddingLength: 64,
      blockCount: 1,
      feedForwardLength: 128,
      headCount: 4,
      headCountKv: 4,
      headDim: 16,
      rmsNormEps: 1e-5,
      ropeFreqBase: 100000,
      ropeFreqScale: 1.0,
      ropeDimensionCount: 16,
      ropeType: 'norm',
      vocabSize: vocab.length,
      bosTokenId: 1,
      eosTokenId: 2,
      unkTokenId: 0,
      addBosToken: false,
      tokenizerModel: 'gpt2',
      preTokenizer: 'smollm',
      chatTemplate: '<|im_start|>',
      primaryQuantization: 'Q4_0',
    },
  };

  const tok = new GgufTokenizer(mockGpt2);
  assert.equal(tok.isByteLevelBpe, true);
  assert.equal(tok.addBosToken, false);
  assert.ok(tok.eogTokenIds.has(0));
  assert.ok(tok.eogTokenIds.has(2));

  const encoded = tok.encode('<|im_start|>user\nHi caf\u00E9<|im_end|>');
  assert.equal(encoded[0], 1); // atomic <|im_start|>
  assert.equal(encoded[encoded.length - 1], 2); // atomic <|im_end|>

  // Streaming decode suppresses control tokens and reconstructs spaces, newlines, and UTF-8 'é'
  const decoded = tok.decode(encoded);
  assert.equal(decoded, 'user\nHi caf\u00E9');
});
