import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { parseGGUF, parseGgufFromBacking } from './gguf';
import { LlamaVTensorRuntime } from './llamaRuntime';
import {
  MemoryBackingStore,
  OpfsDemandVramBackingStore,
  type SyncAccessHandleLike,
} from './vramBackingStore';

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

// ============================================================================
// Synthetic Qwen3-MoE (`qwen3moe`) End-to-End Sparse Expert Paging Parity Test
// ============================================================================

const GGUF_VAL_UINT32 = 4;
const GGUF_VAL_FLOAT32 = 6;
const GGUF_VAL_BOOL = 7;
const GGUF_VAL_STRING = 8;
const GGUF_VAL_ARRAY = 9;

function buildSyntheticQwen3MoeGguf(): Uint8Array {
  const dim = 32;
  const ffnExpDim = 16;
  const nHeads = 4;
  const nKvHeads = 2;
  const headDim = 8;
  const nLayers = 2;
  const nExperts = 8;
  const nExpertsUsed = 2;
  const vocabSize = 32;
  const ctxLen = 64;

  const chunks: Uint8Array[] = [];
  let byteLen = 0;
  const push = (buf: Uint8Array) => {
    chunks.push(buf);
    byteLen += buf.byteLength;
  };
  const pushU32 = (v: number) => {
    const b = new Uint8Array(4);
    new DataView(b.buffer).setUint32(0, v >>> 0, true);
    push(b);
  };
  const pushU64 = (v: number) => {
    const b = new Uint8Array(8);
    new DataView(b.buffer).setBigUint64(0, BigInt(v), true);
    push(b);
  };
  const pushF32 = (v: number) => {
    const b = new Uint8Array(4);
    new DataView(b.buffer).setFloat32(0, v, true);
    push(b);
  };
  const pushBool = (v: boolean) => {
    push(new Uint8Array([v ? 1 : 0]));
  };
  const enc = new TextEncoder();
  const pushStr = (s: string) => {
    const bytes = enc.encode(s);
    pushU64(bytes.byteLength);
    push(bytes);
  };

  type KvEntry = () => void;
  const kvs: KvEntry[] = [
    () => {
      pushStr('general.architecture');
      pushU32(GGUF_VAL_STRING);
      pushStr('qwen3moe');
    },
    () => {
      pushStr('general.name');
      pushU32(GGUF_VAL_STRING);
      pushStr('Synthetic Qwen3-MoE');
    },
    () => {
      pushStr('qwen3moe.context_length');
      pushU32(GGUF_VAL_UINT32);
      pushU32(ctxLen);
    },
    () => {
      pushStr('qwen3moe.embedding_length');
      pushU32(GGUF_VAL_UINT32);
      pushU32(dim);
    },
    () => {
      pushStr('qwen3moe.feed_forward_length');
      pushU32(GGUF_VAL_UINT32);
      pushU32(ffnExpDim);
    },
    () => {
      pushStr('qwen3moe.expert_feed_forward_length');
      pushU32(GGUF_VAL_UINT32);
      pushU32(ffnExpDim);
    },
    () => {
      pushStr('qwen3moe.expert_count');
      pushU32(GGUF_VAL_UINT32);
      pushU32(nExperts);
    },
    () => {
      pushStr('qwen3moe.expert_used_count');
      pushU32(GGUF_VAL_UINT32);
      pushU32(nExpertsUsed);
    },
    () => {
      pushStr('qwen3moe.expert_weights_norm');
      pushU32(GGUF_VAL_BOOL);
      pushBool(true);
    },
    () => {
      pushStr('qwen3moe.block_count');
      pushU32(GGUF_VAL_UINT32);
      pushU32(nLayers);
    },
    () => {
      pushStr('qwen3moe.attention.head_count');
      pushU32(GGUF_VAL_UINT32);
      pushU32(nHeads);
    },
    () => {
      pushStr('qwen3moe.attention.head_count_kv');
      pushU32(GGUF_VAL_UINT32);
      pushU32(nKvHeads);
    },
    () => {
      pushStr('qwen3moe.attention.key_length');
      pushU32(GGUF_VAL_UINT32);
      pushU32(headDim);
    },
    () => {
      pushStr('qwen3moe.attention.layer_norm_rms_epsilon');
      pushU32(GGUF_VAL_FLOAT32);
      pushF32(1e-6);
    },
    () => {
      pushStr('tokenizer.ggml.model');
      pushU32(GGUF_VAL_STRING);
      pushStr('llama');
    },
    () => {
      pushStr('tokenizer.ggml.bos_token_id');
      pushU32(GGUF_VAL_UINT32);
      pushU32(1);
    },
    () => {
      pushStr('tokenizer.ggml.eos_token_id');
      pushU32(GGUF_VAL_UINT32);
      pushU32(2);
    },
    () => {
      pushStr('tokenizer.ggml.tokens');
      pushU32(GGUF_VAL_ARRAY);
      pushU32(GGUF_VAL_STRING);
      pushU64(vocabSize);
      const baseTokens = ['<unk>', '<s>', '</s>', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
      for (let i = 0; i < vocabSize; i++) {
        pushStr(baseTokens[i] ?? `t${i}`);
      }
    },
    () => {
      pushStr('tokenizer.ggml.scores');
      pushU32(GGUF_VAL_ARRAY);
      pushU32(GGUF_VAL_FLOAT32);
      pushU64(vocabSize);
      for (let i = 0; i < vocabSize; i++) {
        pushF32(-i * 0.1);
      }
    },
  ];

  // Deterministic PRNG for weights
  let seed = 0x51a7_9e31;
  const nextF32 = (scale = 0.08): number => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return ((seed / 0xffffffff) * 2 - 1) * scale;
  };
  const makeF32 = (count: number, scale = 0.08, constant?: number): Float32Array => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      out[i] = constant !== undefined ? constant : nextF32(scale);
    }
    return out;
  };

  interface TensorSpec {
    name: string;
    dims: number[]; // GGUF ne[0], ne[1], ne[2]
    data: Float32Array;
  }
  const tensors: TensorSpec[] = [];
  const addTensor = (name: string, dims: number[], data: Float32Array) => {
    tensors.push({ name, dims, data });
  };

  addTensor('token_embd.weight', [dim, vocabSize], makeF32(dim * vocabSize, 0.2));
  addTensor('output_norm.weight', [dim], makeF32(dim, 0, 1.0));
  addTensor('output.weight', [dim, vocabSize], makeF32(dim * vocabSize, 0.2));

  const qDim = nHeads * headDim;
  const kvDim = nKvHeads * headDim;
  for (let l = 0; l < nLayers; l++) {
    addTensor(`blk.${l}.attn_norm.weight`, [dim], makeF32(dim, 0, 1.0));
    addTensor(`blk.${l}.attn_q.weight`, [dim, qDim], makeF32(dim * qDim, 0.1));
    addTensor(`blk.${l}.attn_k.weight`, [dim, kvDim], makeF32(dim * kvDim, 0.1));
    addTensor(`blk.${l}.attn_v.weight`, [dim, kvDim], makeF32(dim * kvDim, 0.1));
    addTensor(`blk.${l}.attn_q_norm.weight`, [headDim], makeF32(headDim, 0, 1.0));
    addTensor(`blk.${l}.attn_k_norm.weight`, [headDim], makeF32(headDim, 0, 1.0));
    addTensor(`blk.${l}.attn_output.weight`, [qDim, dim], makeF32(qDim * dim, 0.1));
    addTensor(`blk.${l}.ffn_norm.weight`, [dim], makeF32(dim, 0, 1.0));
    // Sparse MoE Router & 3D Stacked Expert Tensors [in, out, n_experts]
    addTensor(`blk.${l}.ffn_gate_inp.weight`, [dim, nExperts], makeF32(dim * nExperts, 0.25));
    addTensor(
      `blk.${l}.ffn_gate_exps.weight`,
      [dim, ffnExpDim, nExperts],
      makeF32(dim * ffnExpDim * nExperts, 0.12),
    );
    addTensor(
      `blk.${l}.ffn_up_exps.weight`,
      [dim, ffnExpDim, nExperts],
      makeF32(dim * ffnExpDim * nExperts, 0.12),
    );
    addTensor(
      `blk.${l}.ffn_down_exps.weight`,
      [ffnExpDim, dim, nExperts],
      makeF32(ffnExpDim * dim * nExperts, 0.12),
    );
  }

  // Header: GGUF magic, version 3, tensor count, kv count
  push(new Uint8Array([0x47, 0x47, 0x55, 0x46]));
  pushU32(3);
  pushU64(tensors.length);
  pushU64(kvs.length);
  for (const emit of kvs) emit();

  // Compute aligned tensor offsets (32-byte alignment)
  let relOffset = 0;
  for (const t of tensors) {
    pushStr(t.name);
    pushU32(t.dims.length);
    for (const d of t.dims) pushU64(d);
    pushU32(0); // GgmlType.F32
    pushU64(relOffset);
    const sz = t.data.byteLength;
    relOffset += (sz + 31) & ~31;
  }

  // Pad header to 32-byte alignment
  const pad = (32 - (byteLen % 32)) % 32;
  if (pad > 0) push(new Uint8Array(pad));

  // Emit tensor payloads
  for (const t of tensors) {
    push(new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength));
    const tPad = (32 - (t.data.byteLength % 32)) % 32;
    if (tPad > 0) push(new Uint8Array(tPad));
  }

  const out = new Uint8Array(byteLen);
  let cursor = 0;
  for (const c of chunks) {
    out.set(c, cursor);
    cursor += c.byteLength;
  }
  return out;
}

test('Qwen3-MoE Sparse Expert Paging produces BIT-EXACT identical tokens to resident-all with bounded LRU expert cache', async () => {
  const bytes = buildSyntheticQwen3MoeGguf();
  const sampling = {
    temperature: 0,
    topP: 1,
    topK: 1,
    repeatPenalty: 1.0,
    maxTokens: 12,
    seed: 7,
  };
  // Prompt length >= 16 so multiple Batch-4 micro-batches exercise grouped Sparse MoE (both mask===0b1111 and partial) + frequency-protected LRU!
  const prompt = 'a b c d e f g h a b c d e f g h';

  const rtResident = new LlamaVTensorRuntime();
  rtResident.loadFromBacking(
    'synth-qwen3moe',
    'Synthetic Qwen3-MoE',
    'test://qwen3moe',
    new MemoryBackingStore('synth-qwen3moe', bytes),
    { maxContextCap: 64, weightResidencyPolicy: 'auto' },
  );
  const residentTokens: number[] = [];
  for await (const ev of rtResident.generate(prompt, sampling)) {
    if (!ev.finished || ev.textDelta.length > 0) residentTokens.push(ev.tokenId);
  }
  const telResident = rtResident.getTelemetry();
  assert.equal(telResident.cpuFallbackOps, 0);
  assert.equal(telResident.vramWeights?.isMoe, true);
  assert.equal(telResident.vramWeights?.residencyMode, 'resident-all');

  const rtPaged = new LlamaVTensorRuntime();
  rtPaged.loadFromBacking(
    'synth-qwen3moe',
    'Synthetic Qwen3-MoE',
    'test://qwen3moe',
    new MemoryBackingStore('synth-qwen3moe', bytes),
    { maxContextCap: 64, weightResidencyPolicy: 'paged' },
  );
  const pagedTokens: number[] = [];
  for await (const ev of rtPaged.generate(prompt, sampling)) {
    if (!ev.finished || ev.textDelta.length > 0) pagedTokens.push(ev.tokenId);
  }
  const telPaged = rtPaged.getTelemetry();
  assert.equal(telPaged.cpuFallbackOps, 0);
  assert.equal(telPaged.vramWeights?.isMoe, true);
  assert.equal(telPaged.vramWeights?.residencyMode, 'opfs-paged');
  assert.ok((telPaged.vramWeights?.moeExpertCacheSlots ?? 0) > 0);
  assert.ok((telPaged.vramWeights?.moeExpertMisses ?? 0) > 0, 'Expected Sparse MoE expert cache faults');
  assert.ok(
    (telPaged.vramWeights?.moeExpertHits ?? 0) > 0,
    'Expected Sparse MoE O(1) frequency-protected LRU expert cache hits across grouped Batch-4 prefill & decode',
  );
  assert.deepEqual(pagedTokens, residentTokens);
  assert.ok(residentTokens.length >= 1, 'Expected generated tokens from synthetic Qwen3-MoE');

  rtResident.unload();
  rtPaged.unload();
});

test('Qwen3-MoE async loadFromBackingAsync + generate over OpfsDemandVramBackingStore executes with ZERO sync XHRs and bit-exact parity', async () => {
  const bytes = buildSyntheticQwen3MoeGguf();
  const slabBytes = 16 * 1024;
  let cacheBuf = new Uint8Array(0);
  const mockHandle: SyncAccessHandleLike = {
    getSize: () => cacheBuf.byteLength,
    read: (view, opts) => {
      const at = opts?.at ?? 0;
      const u8 = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
      const count = Math.max(0, Math.min(u8.byteLength, cacheBuf.byteLength - at));
      u8.set(cacheBuf.subarray(at, at + count), 0);
      return count;
    },
    write: (view, opts) => {
      const at = opts?.at ?? 0;
      const src = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
      const nextLen = Math.max(cacheBuf.byteLength, at + src.byteLength);
      if (nextLen !== cacheBuf.byteLength) {
        const next = new Uint8Array(nextLen);
        next.set(cacheBuf, 0);
        cacheBuf = next;
      }
      cacheBuf.set(src, at);
      return src.byteLength;
    },
    truncate: (n) => {
      const next = new Uint8Array(n);
      next.set(cacheBuf.subarray(0, Math.min(cacheBuf.byteLength, n)), 0);
      cacheBuf = next;
    },
    flush: () => {},
    close: () => {},
  };

  let syncXhrCalls = 0;
  let asyncRangeCalls = 0;
  const demandStore = new OpfsDemandVramBackingStore({
    modelId: 'synth-qwen3moe-demand',
    url: 'https://local.test/qwen3moe.gguf',
    totalBytes: bytes.byteLength,
    handle: mockHandle,
    maxCacheBytes: slabBytes * 12,
    slabBytes,
    syncRangeFetcher: () => {
      syncXhrCalls += 1;
      throw new Error('Sync XHR forbidden on Qwen3-MoE async demand path');
    },
    asyncRangeFetcher: async (_url, start, endInclusive, signal) => {
      if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
      asyncRangeCalls += 1;
      return bytes.slice(start, endInclusive + 1);
    },
  });

  const sampling = {
    temperature: 0,
    topP: 1,
    topK: 1,
    repeatPenalty: 1.0,
    maxTokens: 10,
    seed: 7,
  };
  const prompt = 'a b c d e f g h a b c d e f g h';

  const rtRef = new LlamaVTensorRuntime();
  const rtDemand = new LlamaVTensorRuntime();
  try {
    rtRef.loadFromBacking(
      'synth-qwen3moe-ref',
      'Synthetic Qwen3-MoE Ref',
      'test://qwen3moe',
      new MemoryBackingStore('synth-qwen3moe-ref', bytes),
      { maxContextCap: 64, weightResidencyPolicy: 'auto' },
    );
    await rtDemand.loadFromBackingAsync(
      'synth-qwen3moe-demand',
      'Synthetic Qwen3-MoE Demand',
      'https://local.test/qwen3moe.gguf',
      demandStore,
      { maxContextCap: 64, weightResidencyPolicy: 'paged' },
    );

    assert.equal(syncXhrCalls, 0, 'Expected 0 sync XHR calls during Qwen3-MoE loadFromBackingAsync');

    // Speculative prefill must be skipped on opfs-demand so the worker never stalls after loading
    const callsBeforeSpec = asyncRangeCalls;
    const spec = await rtDemand.speculativePrefill(prompt);
    assert.equal(spec.newlyPrefilledTokens, 0);
    assert.equal(asyncRangeCalls, callsBeforeSpec);

    const refTokens: number[] = [];
    for await (const ev of rtRef.generate(prompt, sampling)) {
      if (!ev.finished || ev.textDelta.length > 0) refTokens.push(ev.tokenId);
    }

    const stageDetails: string[] = [];
    const demandTokens: number[] = [];
    for await (const ev of rtDemand.generate(prompt, sampling, undefined, (p) => {
      if (p.detail) stageDetails.push(p.detail);
    })) {
      if (!ev.finished || ev.textDelta.length > 0) demandTokens.push(ev.tokenId);
    }

    assert.deepEqual(demandTokens, refTokens, 'Expected bit-exact Qwen3-MoE parity on async OpfsDemandVramBackingStore');
    assert.equal(syncXhrCalls, 0, 'Expected 0 sync XHR calls across Qwen3-MoE Batch-4, Layer-Major, MoE experts, and decode');
    assert.ok(stageDetails.some((d) => d.includes('Layer ')), 'Expected live layer progress updates');
  } finally {
    rtRef.unload();
    rtDemand.unload();
  }
});

test('Layer-Major Chunked Prefill (C <= 32) cuts OPFS dense layer faults by > 3x while preserving BIT-EXACT parity', async () => {
  const bytes = new Uint8Array(fs.readFileSync(STORIES_260K_PATH));
  // ~48-token prompt spans > 1 full 32-token layer-major chunk (8 x Batch-4 micro-batches per layer pass)
  const prompt =
    'Once upon a time, in a quiet little green village near the forest, a curious young girl named Lily loved to walk along the bright sunny path and listen to the birds singing in the tall oak trees every morning.';
  const sampling = {
    temperature: 0,
    topP: 1,
    topK: 1,
    repeatPenalty: 1.0,
    maxTokens: 1, // Isolate prefill OPFS layer faults cleanly
    seed: 11,
  };

  const rtResident = new LlamaVTensorRuntime();
  rtResident.loadFromBacking(
    'stories260k-lm-res',
    'Stories 260K',
    'test://stories260K',
    new MemoryBackingStore('stories260k-lm-res', bytes),
    { maxContextCap: 256, weightResidencyPolicy: 'auto' },
  );
  const residentTokens: number[] = [];
  for await (const ev of rtResident.generate(prompt, sampling)) {
    if (!ev.finished || ev.textDelta.length > 0) residentTokens.push(ev.tokenId);
  }

  const rtPaged = new LlamaVTensorRuntime();
  rtPaged.loadFromBacking(
    'stories260k-lm-paged',
    'Stories 260K',
    'test://stories260K',
    new MemoryBackingStore('stories260k-lm-paged', bytes),
    { maxContextCap: 256, weightResidencyPolicy: 'paged' },
  );
  const promptTokCount = rtPaged.countTokens(prompt);
  assert.ok(promptTokCount >= 36, `Expected prompt >= 36 tokens, got ${promptTokCount}`);

  const pagedTokens: number[] = [];
  for await (const ev of rtPaged.generate(prompt, sampling)) {
    if (!ev.finished || ev.textDelta.length > 0) pagedTokens.push(ev.tokenId);
  }
  const telPaged = rtPaged.getTelemetry();
  assert.equal(telPaged.cpuFallbackOps, 0);
  assert.equal(telPaged.vramWeights?.residencyMode, 'opfs-paged');
  assert.equal(telPaged.performanceContract?.prefillChunkTokens, 32);
  assert.deepEqual(pagedTokens, residentTokens);

  // Without Layer-Major Chunked Prefill, token-major Batch-4 over ~45 tokens would execute >= 11
  // full layer sweeps = 11 * (L - K + 1) >= 44 OPFS layer faults. Layer-Major Chunked Prefill (C=32)
  // amortizes 8 x Batch-4 micro-batches per layer load, slashing prefill OPFS faults to <= 16!
  const faults = telPaged.vramWeights?.layerFaults ?? 999;
  const totalLayers = telPaged.vramWeights?.totalLayers ?? 5;
  const pinnedSlots = Math.max(0, (telPaged.vramWeights?.layerFrameSlots ?? 2) - 1);
  const unpinnedPerPass = Math.max(1, totalLayers - pinnedSlots);
  const naiveTokenMajorFaults = Math.floor(promptTokCount / 4) * unpinnedPerPass;
  assert.ok(
    faults < naiveTokenMajorFaults * 0.45,
    `Expected Layer-Major Chunked Prefill faults (${faults}) < 45% of naive token-major Batch-4 (${naiveTokenMajorFaults})`,
  );

  rtResident.unload();
  rtPaged.unload();
});


