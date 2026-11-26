import test from 'node:test';
import assert from 'node:assert/strict';
import { GgmlType } from './gguf';
import { sampleNextToken, type SamplingConfig } from './llamaRuntime';
import { VramKvPager, VramPageState, VRAM_KV_PAGE_TOKENS } from './vramKvPager';
import { FrameBudgetProfiler } from './vramSwapPolicy';
import {
  VramUnifiedStore,
  computePrefixBlockHash,
  type CompressedKvPageBlock,
  type UnifiedStoreAllocationPlan,
  type UnifiedStoreGeometry,
} from './vramUnifiedStore';
import { VTensorBackendRegistry } from './vtensorProvider';

function makeDummyBlock(geometry: UnifiedStoreGeometry, seed: number): CompressedKvPageBlock {
  const kQ8 = new Int8Array(geometry.q8BytesPerHalfPage);
  const vQ8 = new Int8Array(geometry.q8BytesPerHalfPage);
  const kScales = new Float32Array(geometry.scaleFloatsPerHalfPage);
  const vScales = new Float32Array(geometry.scaleFloatsPerHalfPage);
  for (let i = 0; i < kQ8.length; i++) {
    kQ8[i] = ((i * 7 + seed * 13) % 251) - 125;
    vQ8[i] = ((i * 11 + seed * 17) % 251) - 125;
  }
  for (let i = 0; i < kScales.length; i++) {
    kScales[i] = 0.01 * ((i + seed) % 19 + 1);
    vScales[i] = 0.02 * ((i + seed) % 13 + 1);
  }
  return { kQ8, vQ8, kScales, vScales };
}

const TEST_GEOMETRY: UnifiedStoreGeometry = {
  modelId: 'test-model',
  contentFingerprint: 'fp-deadbeef-1234',
  numLayers: 2,
  nHeadsKv: 2,
  headDim: 8,
  pageTokens: VRAM_KV_PAGE_TOKENS,
  q8BytesPerHalfPage: 2 * VRAM_KV_PAGE_TOKENS * 2 * 8,
  scaleFloatsPerHalfPage: 2 * VRAM_KV_PAGE_TOKENS * 2,
};

const BYTES_PER_BLOCK =
  TEST_GEOMETRY.q8BytesPerHalfPage * 2 + TEST_GEOMETRY.scaleFloatsPerHalfPage * 8;

const TEST_ALLOC: UnifiedStoreAllocationPlan = {
  hotEmbeddingCacheSlots: 0,
  hotEmbeddingSlabBytes: 0,
  kvPrefixSlabBudgetBytes: BYTES_PER_BLOCK * 8,
  maxKvPrefixBlocks: 8,
  bytesPerKvBlock: BYTES_PER_BLOCK,
};

test('VramUnifiedStore UKV3 round-trips bit-exact and rejects bit-flipped or truncated payloads cleanly', () => {
  const writer = new VramUnifiedStore(TEST_GEOMETRY, TEST_ALLOC, BYTES_PER_BLOCK * 16);
  const tokens = Array.from({ length: VRAM_KV_PAGE_TOKENS * 2 }, (_, i) => 100 + i);
  const keys = writer.computePrefixKeys(tokens);
  for (let i = 0; i < keys.length; i++) {
    writer.putBlock(keys[i].key, keys[i].pageIndex, keys[i].slice, makeDummyBlock(TEST_GEOMETRY, i + 1));
  }

  const snapshot = writer.serializeSnapshotBytes();
  assert.ok(snapshot.byteLength > 36);
  // Magic must be "UKV3" (0x33564b55 LE)
  assert.equal(
    new DataView(snapshot.buffer, snapshot.byteOffset, snapshot.byteLength).getUint32(0, true),
    0x33564b55,
  );

  // Valid hydration succeeds bit-exact
  const cleanReader = new VramUnifiedStore(TEST_GEOMETRY, TEST_ALLOC, BYTES_PER_BLOCK * 16);
  assert.equal(cleanReader.hydrateFromBytes(snapshot), 2);
  const chain = cleanReader.lookupPrefixChain(tokens);
  assert.equal(chain.length, 2);
  assert.equal(chain[0].data.kQ8[5], writer.getBlock(keys[0].key)!.data.kQ8[5]);

  // Single-bit flip in body must be rejected with 0 hydrated blocks and zero thrown error
  const corruptedBody = snapshot.slice();
  corruptedBody[Math.floor(corruptedBody.byteLength / 2)] ^= 0x04;
  const corruptReaderA = new VramUnifiedStore(TEST_GEOMETRY, TEST_ALLOC, BYTES_PER_BLOCK * 16);
  assert.equal(corruptReaderA.hydrateFromBytes(corruptedBody), 0);
  assert.equal(corruptReaderA.blockCount, 0);

  // Single-bit flip in 8-byte integrity trailer must be rejected
  const corruptedTrailer = snapshot.slice();
  corruptedTrailer[corruptedTrailer.byteLength - 1] ^= 0x80;
  const corruptReaderB = new VramUnifiedStore(TEST_GEOMETRY, TEST_ALLOC, BYTES_PER_BLOCK * 16);
  assert.equal(corruptReaderB.hydrateFromBytes(corruptedTrailer), 0);
  assert.equal(corruptReaderB.blockCount, 0);

  // Truncated mid-write snapshot must be rejected cleanly
  const truncated = snapshot.slice(0, snapshot.byteLength - 5);
  const truncReader = new VramUnifiedStore(TEST_GEOMETRY, TEST_ALLOC, BYTES_PER_BLOCK * 16);
  assert.equal(truncReader.hydrateFromBytes(truncated), 0);
  assert.equal(truncReader.blockCount, 0);
});

test('VramUnifiedStore protects multi-page pinned prefixes [0..pinnedPrefixPages-1] during eviction', () => {
  const compactAlloc: UnifiedStoreAllocationPlan = {
    ...TEST_ALLOC,
    kvPrefixSlabBudgetBytes: BYTES_PER_BLOCK * 4,
    maxKvPrefixBlocks: 4,
  };
  const store = new VramUnifiedStore(TEST_GEOMETRY, compactAlloc, BYTES_PER_BLOCK * 8);
  store.setPinnedPrefixPages(3); // Pages 0, 1, 2 pinned

  const pageKeys: string[] = [];
  let parent = 'root';
  for (let p = 0; p < 6; p++) {
    const toks = Array.from({ length: VRAM_KV_PAGE_TOKENS }, (_, t) => p * 100 + t);
    const key = computePrefixBlockHash(store.identityKey, parent, toks);
    pageKeys.push(key);
    parent = key;
    store.putBlock(key, p, toks, makeDummyBlock(TEST_GEOMETRY, p + 1));
  }

  // Capacity is 4: pinned pages 0, 1, 2 MUST remain resident while older unpinned pages (3, 4) evict
  assert.equal(store.blockCount, 4);
  assert.equal(store.hasBlock(pageKeys[0]), true);
  assert.equal(store.hasBlock(pageKeys[1]), true);
  assert.equal(store.hasBlock(pageKeys[2]), true);
  assert.equal(store.hasBlock(pageKeys[5]), true);
  assert.equal(store.hasBlock(pageKeys[3]), false);
});

test('VramKvPager never throws hot-frame exhaustion even under NaN/Infinity outlier KV pages', () => {
  const reg = new VTensorBackendRegistry({ initialWasmPages: 16 });
  const pager = new VramKvPager(
    {
      numLayers: 2,
      maxContext: 96,
      nHeadsKv: 2,
      headDim: 16,
      hotFrameLimit: 2,
      warmQ8Limit: 2,
      coldQ4Enabled: true,
    },
    (id, usage, bytes) => reg.allocBuffer(id, usage, bytes),
  );

  // Poison pages 1 and 2 with NaN / Infinity values that fail nominal challenger tolerances
  // while keeping sink page 0 and active tail page 3 finite
  const poison = new Float32Array(2 * 16);
  poison.fill(Number.NaN);
  poison[0] = Number.POSITIVE_INFINITY;
  poison[1] = 1.5;

  const normalK = new Float32Array(32).fill(0.25);
  const normalV = new Float32Array(32).fill(-0.25);

  for (let pos = 0; pos < 80; pos++) {
    const isPoisonedPage = pos >= 16 && pos < 48;
    for (let l = 0; l < 2; l++) {
      pager.writeKv(false, l, pos, isPoisonedPage ? poison : normalK);
      pager.writeKv(true, l, pos, isPoisonedPage ? poison : normalV);
    }
  }

  // Pager made total progress across 5 pages with hotFrameLimit = 2 without crashing,
  // using multi-victim fallback first and sanitized emergency reclaim when all hot pages were poisoned!
  assert.ok(pager.getForcedReclaims() >= 1);
  assert.notEqual(pager.getPageState(1), VramPageState.HOT_RESIDENT);
  assert.notEqual(pager.getPageState(2), VramPageState.HOT_RESIDENT);
  assert.equal(pager.getPageState(4), VramPageState.HOT_RESIDENT);

  // Attention over all 80 tokens produces strictly finite output (no NaN contamination)
  const q = new Float32Array(2 * 16).fill(0.1);
  const out = new Float32Array(2 * 16);
  pager.flashAttnExt(q, -1, out, -1, 2, 0, 80, null);
  for (let i = 0; i < out.length; i++) {
    assert.equal(Number.isFinite(out[i]), true);
  }
});

test('VramKvPager keeps O(1) active-tier attention telemetry exact across decode and prefix reads', () => {
  const reg = new VTensorBackendRegistry({ initialWasmPages: 16 });
  const pager = new VramKvPager(
    {
      numLayers: 1,
      maxContext: 64,
      nHeadsKv: 2,
      headDim: 16,
      hotFrameLimit: 2,
      warmQ8Limit: 4,
      coldQ4Enabled: false,
    },
    (id, usage, bytes) => reg.allocBuffer(id, usage, bytes),
  );

  const kv = new Float32Array(2 * 16).fill(0.5);
  // Fill 3 pages (48 tokens): 2 HOT_RESIDENT + 1 COLD_COMPRESSED
  for (let pos = 0; pos < 48; pos++) {
    pager.writeKv(false, 0, pos, kv);
    pager.writeKv(true, 0, pos, kv);
  }
  const before = pager.getTelemetry();
  assert.equal(before.hotResidentPages, 2);
  assert.equal(before.warmQ8Pages, 1);

  const scratch = reg.allocBuffer('test.scratch', 'compute', 1024);
  const qH = scratch.allocTensor('q', GgmlType.F32, [32]);
  const outH = scratch.allocTensor('out', GgmlType.F32, [32]);
  scratch.f32View(qH).fill(0.2);

  // Full decode read (seqLen = 48 -> numPages = 3 >= watermark 3 -> O(1) fast path)
  pager.flashAttnExt(
    scratch.f32View(qH),
    scratch.wasmPtr(qH),
    scratch.f32View(outH),
    scratch.wasmPtr(outH),
    2,
    0,
    48,
    reg.arena,
  );
  const afterFull = pager.getTelemetry();
  assert.equal(afterFull.hotAttentionReads, 2 * 2); // 2 heads * 2 hot pages
  assert.equal(afterFull.coldAttentionReads, 2 * 1); // 2 heads * 1 warm page

  // Partial prefix read (seqLen = 16 -> numPages = 1 < watermark 3 -> exact subset scan)
  pager.flashAttnExt(
    scratch.f32View(qH),
    scratch.wasmPtr(qH),
    scratch.f32View(outH),
    scratch.wasmPtr(outH),
    2,
    0,
    16,
    reg.arena,
  );
  const afterPrefix = pager.getTelemetry();
  // Page 0 is HOT_RESIDENT (sink pinned), so adding 2 heads * 1 hot page and 0 cold pages
  assert.equal(afterPrefix.hotAttentionReads, afterFull.hotAttentionReads + 2);
  assert.equal(afterPrefix.coldAttentionReads, afterFull.coldAttentionReads);
});

test('sampleNextToken guards greedy, Top-K/Top-P, large Top-K, and all-NaN logits cleanly', () => {
  const baseCfg: SamplingConfig = {
    temperature: 0.7,
    topK: 4,
    topP: 0.9,
    repeatPenalty: 1.0,
    maxTokens: 32,
    seed: 42,
  };

  const logits = new Float32Array([
    Number.NaN,
    Number.POSITIVE_INFINITY,
    -2.5,
    4.25,
    Number.NEGATIVE_INFINITY,
    1.0,
  ]);

  // Greedy path must choose finite argmax (token 3 = 4.25), never NaN/Infinity index
  const greedyTok = sampleNextToken(logits, [], { ...baseCfg, temperature: 0 }, () => 0.5);
  assert.equal(greedyTok, 3);

  // Top-K/Top-P path must never select NaN/Infinity indices (0, 1, 4)
  for (const r of [0.0, 0.25, 0.5, 0.99]) {
    const tok = sampleNextToken(logits, [], baseCfg, () => r);
    assert.ok([2, 3, 5].includes(tok), `unexpected token ${tok}`);
  }

  // Large Top-K (> 1024) path also filters non-finite logits cleanly
  const largeLogits = new Float32Array(1100);
  largeLogits.fill(Number.NaN);
  largeLogits[777] = 3.5;
  assert.equal(sampleNextToken(largeLogits, [], { ...baseCfg, topK: 1050 }, () => 0.5), 777);

  // All-NaN edge case falls back deterministically to token 0 without throwing
  const allNan = new Float32Array([Number.NaN, Number.NaN, Number.NaN]);
  assert.equal(sampleNextToken(allNan, [], baseCfg, () => 0.5), 0);
  assert.equal(sampleNextToken(allNan, [], { ...baseCfg, temperature: 0 }, () => 0.5), 0);
});

test('FrameBudgetProfiler maintains bounded O(1) state and exact 3-sample median across 10,500 steps', () => {
  const profiler = new FrameBudgetProfiler(16_666, 'message-channel');
  profiler.beginSlice(0);
  let nowMs = 0;
  for (let i = 0; i < 10_500; i++) {
    const stepUs = i % 3 === 0 ? 4_000 : i % 3 === 1 ? 6_000 : 8_000;
    nowMs += stepUs / 1000;
    profiler.recordStepAndShouldYield(stepUs, 12_000, 16, nowMs);
  }
  const tel = profiler.getTelemetry();
  assert.equal(tel.totalSteps, 10_500);
  assert.equal(tel.medianStepUs, 6_000);
  assert.equal(tel.withinFrameBudget, true);
  assert.ok(tel.yieldsExecuted > 0);
  assert.ok(tel.yieldsAvoided > 0);
});
