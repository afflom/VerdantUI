import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { LlamaVTensorRuntime } from './llamaRuntime';
import { MemoryBackingStore, type SyncAccessHandleLike } from './vramBackingStore';
import {
  computePrefixBlockHash,
  deriveUnifiedStoreAllocation,
  trimToStableSpeculativePrefix,
  VramUnifiedStore,
  type CompressedKvPageBlock,
} from './vramUnifiedStore';

function makeMockBlock(seed: number, q8Bytes = 64, scaleFloats = 4): CompressedKvPageBlock {
  const kQ8 = new Int8Array(q8Bytes);
  const vQ8 = new Int8Array(q8Bytes);
  const kScales = new Float32Array(scaleFloats);
  const vScales = new Float32Array(scaleFloats);
  for (let i = 0; i < q8Bytes; i++) {
    kQ8[i] = ((seed * 17 + i * 7) % 240) - 120;
    vQ8[i] = ((seed * 23 - i * 5) % 240) - 120;
  }
  for (let i = 0; i < scaleFloats; i++) {
    kScales[i] = 0.01 * (seed + i + 1);
    vScales[i] = 0.02 * (seed + i + 1);
  }
  return { kQ8, vQ8, kScales, vScales };
}

function makeMockOpfsDir(): FileSystemDirectoryHandle {
  const files = new Map<string, Uint8Array>();

  const makeSyncHandle = (name: string): SyncAccessHandleLike => ({
    getSize: () => files.get(name)?.byteLength ?? 0,
    read: (buffer: ArrayBufferView, options?: { at?: number }) => {
      const data = files.get(name) ?? new Uint8Array(0);
      const at = options?.at ?? 0;
      const dst = new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
      const avail = Math.max(0, Math.min(dst.byteLength, data.byteLength - at));
      if (avail > 0) dst.set(data.subarray(at, at + avail), 0);
      return avail;
    },
    write: (buffer: ArrayBufferView, options?: { at?: number }) => {
      const at = options?.at ?? 0;
      const src = new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
      const cur = files.get(name) ?? new Uint8Array(0);
      const nextLen = Math.max(cur.byteLength, at + src.byteLength);
      const next = new Uint8Array(nextLen);
      next.set(cur, 0);
      next.set(src, at);
      files.set(name, next);
      return src.byteLength;
    },
    truncate: (newSize: number) => {
      const cur = files.get(name) ?? new Uint8Array(0);
      const next = new Uint8Array(newSize);
      next.set(cur.subarray(0, Math.min(cur.byteLength, newSize)), 0);
      files.set(name, next);
    },
    flush: () => {},
    close: () => {},
  });

  return {
    getFileHandle: async (name: string, options?: { create?: boolean }) => {
      if (!files.has(name)) {
        if (!options?.create) throw new Error(`NotFoundError: ${name}`);
        files.set(name, new Uint8Array(0));
      }
      return {
        createSyncAccessHandle: async () => makeSyncHandle(name),
      };
    },
  } as unknown as FileSystemDirectoryHandle;
}

test('Theorem T6 deriveUnifiedStoreAllocation respects global budget across pinned and OPFS-streamed regimes', () => {
  const pinnedPlan = deriveUnifiedStoreAllocation({
    globalBudgetBytes: 512 * 1024 * 1024,
    weightAndLiveKvBytes: 120 * 1024 * 1024,
    embdRowBytes: 2304,
    streamEmbeddings: false,
    vocabSize: 49152,
    kvPageBytesWarmQ8: 32 * 1024,
  });
  assert.equal(pinnedPlan.hotEmbeddingCacheSlots, 0);
  assert.equal(pinnedPlan.hotEmbeddingSlabBytes, 0);
  assert.ok(pinnedPlan.maxKvPrefixBlocks >= 16);

  const pagedPlan = deriveUnifiedStoreAllocation({
    globalBudgetBytes: 512 * 1024 * 1024,
    weightAndLiveKvBytes: 220 * 1024 * 1024,
    embdRowBytes: 3072,
    streamEmbeddings: true,
    vocabSize: 128256,
    kvPageBytesWarmQ8: 64 * 1024,
  });
  assert.ok(pagedPlan.hotEmbeddingCacheSlots >= 32 && pagedPlan.hotEmbeddingCacheSlots <= 512);
  assert.ok(
    220 * 1024 * 1024 + pagedPlan.hotEmbeddingSlabBytes + pagedPlan.kvPrefixSlabBudgetBytes <=
      512 * 1024 * 1024,
  );
});

test('VramUnifiedStore content-addressed prefix hashing deduplicates shared prefixes and protects page-0 system sinks', () => {
  const plan = {
    hotEmbeddingCacheSlots: 0,
    hotEmbeddingSlabBytes: 0,
    kvPrefixSlabBudgetBytes: 64 * 1024,
    maxKvPrefixBlocks: 3,
    bytesPerKvBlock: 160,
  };
  const store = new VramUnifiedStore(
    {
      modelId: 'smollm2-135m',
      numLayers: 1,
      nHeadsKv: 1,
      headDim: 4,
      pageTokens: 16,
      q8BytesPerHalfPage: 64,
      scaleFloatsPerHalfPage: 4,
    },
    plan,
    1024 * 1024,
  );

  const page0 = Array.from({ length: 16 }, (_, i) => i + 1);
  const page1A = Array.from({ length: 16 }, (_, i) => i + 100);
  const page1B = Array.from({ length: 16 }, (_, i) => i + 200);
  const page1C = Array.from({ length: 16 }, (_, i) => i + 300);

  const k0 = computePrefixBlockHash('smollm2-135m', 'root', page0);
  const k1A = computePrefixBlockHash('smollm2-135m', k0, page1A);
  const k1B = computePrefixBlockHash('smollm2-135m', k0, page1B);
  const k1C = computePrefixBlockHash('smollm2-135m', k0, page1C);

  assert.equal(store.putBlock(k0, 0, page0, makeMockBlock(1)), true);
  // Deduplicate identical page 0 from another thread
  assert.equal(store.putBlock(k0, 0, page0, makeMockBlock(1)), false);
  assert.equal(store.putBlock(k1A, 1, page1A, makeMockBlock(2)), true);
  assert.equal(store.putBlock(k1B, 1, page1B, makeMockBlock(3)), true);
  // Inserting 4th block into maxBlocks=3 must evict page1A (non-sink) while preserving page0 (system sink)
  assert.equal(store.putBlock(k1C, 1, page1C, makeMockBlock(4)), true);

  assert.equal(store.hasBlock(k0), true, 'System sink page 0 must survive eviction');
  assert.equal(store.hasBlock(k1A), false, 'Oldest non-sink block should be evicted');
  assert.equal(store.hasBlock(k1C), true);

  const chainB = store.lookupPrefixChain([...page0, ...page1B]);
  assert.equal(chainB.length, 2);
  assert.equal(chainB[0].blockKey, k0);
  assert.equal(chainB[1].blockKey, k1B);
});

test('VramUnifiedStore OPFS UKV1 binary persist and hydrate round-trips exactly across sessions', async () => {
  const opfsDir = makeMockOpfsDir();
  const geometry = {
    modelId: 'opfs-roundtrip-model',
    numLayers: 2,
    nHeadsKv: 2,
    headDim: 4,
    pageTokens: 16,
    q8BytesPerHalfPage: 2 * 16 * 2 * 4,
    scaleFloatsPerHalfPage: 2 * 16 * 2,
  };
  const plan = {
    hotEmbeddingCacheSlots: 0,
    hotEmbeddingSlabBytes: 0,
    kvPrefixSlabBudgetBytes: 1024 * 1024,
    maxKvPrefixBlocks: 8,
    bytesPerKvBlock: 1024,
  };

  const writer = new VramUnifiedStore(geometry, plan, 4 * 1024 * 1024);
  const p0 = Array.from({ length: 16 }, (_, i) => i + 10);
  const p1 = Array.from({ length: 16 }, (_, i) => i + 50);
  const k0 = computePrefixBlockHash(geometry.modelId, 'root', p0);
  const k1 = computePrefixBlockHash(geometry.modelId, k0, p1);
  const b0 = makeMockBlock(11, geometry.q8BytesPerHalfPage, geometry.scaleFloatsPerHalfPage);
  const b1 = makeMockBlock(22, geometry.q8BytesPerHalfPage, geometry.scaleFloatsPerHalfPage);

  writer.putBlock(k0, 0, p0, b0);
  writer.putBlock(k1, 1, p1, b1);

  const persisted = await writer.persistToOpfs(opfsDir);
  assert.equal(persisted, 2);

  // Hydrate into a fresh store (simulating browser reload)
  const reader = new VramUnifiedStore(geometry, plan, 4 * 1024 * 1024);
  const hydrated = await reader.hydrateFromOpfs(opfsDir);
  assert.equal(hydrated, 2);

  const restored = reader.lookupPrefixChain([...p0, ...p1]);
  assert.equal(restored.length, 2);
  assert.deepEqual(Array.from(restored[0].data.kQ8), Array.from(b0.kQ8));
  assert.deepEqual(Array.from(restored[1].data.vQ8), Array.from(b1.vQ8));
  assert.deepEqual(Array.from(restored[0].data.kScales), Array.from(b0.kScales));

  // Stale/mismatched geometry must be safely ignored
  const mismatched = new VramUnifiedStore({ ...geometry, numLayers: 99 }, plan, 4 * 1024 * 1024);
  assert.equal(await mismatched.hydrateFromOpfs(opfsDir), 0);

  // Chained prefix persistence: pinned early system-prompt pages 0..13 MUST survive OPFS persistence
  // even when 30 newer tail pages are added afterwards (never evicted in favor of orphan tail pages!)
  const bigPlan = { ...plan, maxKvPrefixBlocks: 64, kvPrefixSlabBudgetBytes: 64 * 1024 * 1024 };
  const chainedStore = new VramUnifiedStore(geometry, bigPlan, 64 * 1024 * 1024);
  chainedStore.setPinnedPrefixPages(14);
  const allTokens: number[] = [];
  for (let i = 0; i < 40 * geometry.pageTokens; i++) allTokens.push((i % 250) + 1);
  const allKeys = chainedStore.computePrefixKeys(allTokens);
  for (const k of allKeys) {
    chainedStore.putBlock(
      k.key,
      k.pageIndex,
      k.slice,
      makeMockBlock(k.pageIndex + 1, geometry.q8BytesPerHalfPage, geometry.scaleFloatsPerHalfPage),
    );
  }
  const snapBytes = chainedStore.serializeSnapshotBytes(24);
  const reloaded = new VramUnifiedStore(geometry, bigPlan, 64 * 1024 * 1024);
  assert.equal(reloaded.hydrateFromBytes(snapBytes), 24);
  const restoredChain = reloaded.lookupPrefixChain(allTokens);
  assert.ok(
    restoredChain.length >= 14,
    `Expected all 14 pinned boot prefix pages to survive OPFS persistence, got ${restoredChain.length}`,
  );
});

test('trimToStableSpeculativePrefix trims cleanly to word boundaries while typing and preserves full draft on pause', () => {
  assert.equal(trimToStableSpeculativePrefix('What is the cap', false), 'What is the ');
  assert.equal(trimToStableSpeculativePrefix('What is the cap', true), 'What is the cap');
  assert.equal(trimToStableSpeculativePrefix('Hello world ', false), 'Hello world ');
  assert.equal(trimToStableSpeculativePrefix('Singleword', false), '');
  assert.equal(trimToStableSpeculativePrefix('Singleword', true), 'Singleword');
});

test('LlamaVTensorRuntime speculative prefill, Zipfian hot embedding cache, and cross-thread Unified Store KV restore match cold baseline', async () => {
  const modelPath = path.resolve(process.cwd(), 'public/models/stories260K.gguf');
  const bytes = new Uint8Array(fs.readFileSync(modelPath));

  const sharedPrefix =
    'Once upon a time, a little silver bird flew over the tall green trees and found a shiny golden key near the quiet river. ';
  const fullPrompt = `${sharedPrefix}The bird picked up the key and`;

  // 1. Cold baseline (resident-all, no prior speculative prefill)
  const baseline = new LlamaVTensorRuntime();
  baseline.loadFromBuffer('stories260k-base', 'Stories 260K', 'bundled', bytes, 256);
  const baselineTokens: number[] = [];
  for await (const ev of baseline.generate(fullPrompt, {
    temperature: 0,
    topP: 1,
    topK: 1,
    repeatPenalty: 1,
    maxTokens: 10,
    seed: 1,
  })) {
    baselineTokens.push(ev.tokenId);
  }

  // 2. OPFS-paged runtime with Zipfian Hot Embedding Cache + Continuous Speculative Prefill
  const unifiedRt = new LlamaVTensorRuntime();
  unifiedRt.loadFromBacking(
    'stories260k-unified',
    'Stories 260K',
    'bundled',
    new MemoryBackingStore('stories260k-unified', bytes),
    { maxContextCap: 256, weightResidencyPolicy: 'paged' },
  );

  // Speculatively prefill the shared prefix while the user is still typing!
  const specRes = await unifiedRt.speculativePrefill(sharedPrefix);
  assert.ok(specRes.readyTokens >= 20, `Expected >=20 speculatively prefilled tokens, got ${specRes.readyTokens}`);
  assert.equal(specRes.aborted, false);

  const afterSpecTel = unifiedRt.getTelemetry();
  assert.ok(afterSpecTel.unifiedStore !== null);
  assert.ok(
    afterSpecTel.unifiedStore.kvPrefixBlocks >= 1,
    `Expected >=1 committed 16-token KV prefix block in Unified Store, got ${afterSpecTel.unifiedStore.kvPrefixBlocks}`,
  );
  assert.ok(
    afterSpecTel.unifiedStore.embdCacheSlots >= 32 && afterSpecTel.unifiedStore.embdCacheHits > 0,
    `Expected Zipfian hot embedding cache hits during prefill, got slots=${afterSpecTel.unifiedStore.embdCacheSlots}, hits=${afterSpecTel.unifiedStore.embdCacheHits}`,
  );

  // Now submit the full prompt: must reuse the speculatively warmed tokens and match baseline bit-for-bit!
  let reportedReused = 0;
  const specGenTokens: number[] = [];
  for await (const ev of unifiedRt.generate(
    fullPrompt,
    { temperature: 0, topP: 1, topK: 1, repeatPenalty: 1, maxTokens: 10, seed: 1 },
    undefined,
    (p) => {
      reportedReused = Math.max(reportedReused, p.reusedPrefixTokens);
    },
  )) {
    specGenTokens.push(ev.tokenId);
  }

  assert.ok(
    reportedReused >= specRes.readyTokens - 2,
    `Expected generate() to reuse speculatively warmed prefix (${reportedReused} vs ${specRes.readyTokens})`,
  );
  assert.deepEqual(
    specGenTokens,
    baselineTokens,
    'Speculative prefill must produce bit-identical greedy tokens to cold baseline',
  );

  // 3. Cross-thread Unified Store KV restore: wipe live KV cache (new thread) and run a new turn with the same prefix!
  unifiedRt.resetKvCache();
  let restoredReused = 0;
  const crossThreadTokens: number[] = [];
  for await (const ev of unifiedRt.generate(
    `${sharedPrefix}A small brown rabbit watched`,
    { temperature: 0, topP: 1, topK: 1, repeatPenalty: 1, maxTokens: 6, seed: 1 },
    undefined,
    (p) => {
      restoredReused = Math.max(restoredReused, p.reusedPrefixTokens);
    },
  )) {
    crossThreadTokens.push(ev.tokenId);
  }

  const finalTel = unifiedRt.getTelemetry().unifiedStore!;
  assert.ok(
    restoredReused >= 16,
    `Expected >=16 tokens restored from Unified Store after resetKvCache(), got ${restoredReused}`,
  );
  assert.ok(
    finalTel.unifiedRestoredTokens >= 16,
    `Expected telemetry unifiedRestoredTokens >= 16, got ${finalTel.unifiedRestoredTokens}`,
  );
  assert.ok(crossThreadTokens.length >= 4);
  assert.equal(unifiedRt.getTelemetry().cpuFallbackOps, 0);
});
