import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { parseGGUF, parseGgufFromBacking } from './gguf';
import { LlamaVTensorRuntime } from './llamaRuntime';
import { GgufTokenizer } from './tokenizer';
import { buildRopeTables } from './verdantKernels';
import {
  MemoryBackingStore,
  OpfsDemandVramBackingStore,
  deleteOpfsModel,
  type SyncAccessHandleLike,
} from './vramBackingStore';
import {
  computeBackingContentFingerprint,
  deserializeHotEmbeddingsCache,
  deserializeParsedGgufBootCache,
  deserializeRopeTablesCache,
  readHotEmbeddingsCacheFromOpfs,
  readParsedGgufBootCacheFromOpfs,
  readRopeTablesCacheFromOpfs,
  serializeHotEmbeddingsCache,
  serializeParsedGgufBootCache,
  serializeRopeTablesCache,
  writeHotEmbeddingsCacheToOpfs,
  writeParsedGgufBootCacheToOpfs,
  writeRopeTablesCacheToOpfs,
} from './vramBootCache';
import {
  VramUnifiedStore,
  computePrefixBlockHash,
  type CompressedKvPageBlock,
} from './vramUnifiedStore';
import { VramWeightPager, type LayerTensorInfoGroup } from './vramWeightPager';
import { VTensorBackendRegistry } from './vtensorProvider';

function createMockOpfsDir() {
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

  const dir = {
    async getFileHandle(name: string, options?: { create?: boolean }) {
      if (!files.has(name)) {
        if (!options?.create) throw new DOMException(`NotFound: ${name}`, 'NotFoundError');
        files.set(name, new Uint8Array(0));
      }
      return {
        kind: 'file',
        name,
        async getFile() {
          const data = files.get(name) ?? new Uint8Array(0);
          return new Blob([data.slice()]);
        },
        async createSyncAccessHandle() {
          return makeSyncHandle(name);
        },
      };
    },
    async removeEntry(name: string) {
      if (!files.has(name)) throw new DOMException(`NotFound: ${name}`, 'NotFoundError');
      files.delete(name);
    },
    async *entries(): AsyncIterableIterator<[string, FileSystemHandle]> {
      for (const name of files.keys()) {
        yield [name, { kind: 'file', name } as FileSystemHandle];
      }
    },
  } as unknown as FileSystemDirectoryHandle;

  return { dir, files };
}

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

test('Parsed GGUF header+tokenizer boot cache (VHDR) and RoPE cache (VROP) round-trip bit-exactly and invalidate on signature mismatch', async () => {
  const modelPath = path.resolve(process.cwd(), 'public/models/stories260K.gguf');
  const fileBytes = new Uint8Array(fs.readFileSync(modelPath));
  const backing = new MemoryBackingStore('stories-hdr', fileBytes);
  const fp = computeBackingContentFingerprint(backing);
  const parsed = parseGgufFromBacking(backing);
  const { dir, files } = createMockOpfsDir();

  // 1. Header + tokenizer boot cache round-trip
  const serialized = serializeParsedGgufBootCache(parsed, fp);
  const restored = deserializeParsedGgufBootCache(serialized, fp, fileBytes.byteLength);
  assert.ok(restored !== null);
  assert.equal(restored.version, parsed.version);
  assert.equal(restored.dataOffset, parsed.dataOffset);
  assert.equal(restored.tensors.length, parsed.tensors.length);
  assert.deepEqual(restored.hparams, parsed.hparams);
  assert.ok(
    (restored.metadata['tokenizer.ggml.scores'] as unknown) instanceof Float32Array,
    'Expected scores restored directly as Float32Array',
  );

  // Verify tokenizer encode/decode parity between fresh parseGGUF and cached VHDR parse
  const tokRef = new GgufTokenizer(parseGGUF(fileBytes));
  const tokCached = new GgufTokenizer(restored);
  const samplePrompt = 'Once upon a time, a little robot built a fast garden.';
  assert.deepEqual(tokCached.encode(samplePrompt), tokRef.encode(samplePrompt));
  assert.equal(tokCached.decode(tokCached.encode(samplePrompt)), tokRef.decode(tokRef.encode(samplePrompt)));

  // Signature invalidation on wrong fingerprint or fileByteLength
  assert.equal(deserializeParsedGgufBootCache(serialized, 'deadbeef-1234', fileBytes.byteLength), null);
  assert.equal(deserializeParsedGgufBootCache(serialized, fp, fileBytes.byteLength + 1), null);

  // OPFS read/write round-trip
  assert.equal(await writeParsedGgufBootCacheToOpfs(fp, fp, parsed, dir), true);
  assert.equal(files.has(`${fp}.hdr.bin`), true);
  const fromOpfs = await readParsedGgufBootCacheFromOpfs(fp, fp, fileBytes.byteLength, dir);
  assert.ok(fromOpfs !== null);
  assert.equal(fromOpfs.tensors.length, parsed.tensors.length);

  // 2. Precomputed RoPE LUT cache round-trip & bit-exactness
  const ropeParams = { maxCtx: 512, ropeDim: 64, freqBase: 10000, freqScale: 1.0 };
  const ropeRef = buildRopeTables(ropeParams.maxCtx, ropeParams.ropeDim, ropeParams.freqBase, ropeParams.freqScale);
  const ropeBin = serializeRopeTablesCache(ropeRef, ropeParams);
  const ropeBack = deserializeRopeTablesCache(ropeBin, ropeParams);
  assert.ok(ropeBack !== null);
  assert.deepEqual(ropeBack.freqs, ropeRef.freqs);
  assert.deepEqual(ropeBack.cos, ropeRef.cos);
  assert.deepEqual(ropeBack.sin, ropeRef.sin);
  assert.equal(deserializeRopeTablesCache(ropeBin, { ...ropeParams, maxCtx: 1024 }), null);
  assert.equal(deserializeRopeTablesCache(ropeBin, { ...ropeParams, freqBase: 1_000_000 }), null);

  assert.equal(await writeRopeTablesCacheToOpfs(fp, ropeRef, ropeParams, dir), true);
  assert.equal(files.has(`${fp}.rope.bin`), true);
  const ropeOpfs = await readRopeTablesCacheFromOpfs(fp, ropeParams, dir);
  assert.ok(ropeOpfs !== null);
  assert.deepEqual(ropeOpfs.cos, ropeRef.cos);
  assert.deepEqual(ropeOpfs.sin, ropeRef.sin);
});

test('Cross-model content-addressed VramUnifiedStore shares KV prefixes across aliases and upgrades legacy <modelId>.unified.kv', async () => {
  const { dir, files } = createMockOpfsDir();
  const sharedFp = 'a1b2c3d4e5f60708-12345678';
  const plan = {
    hotEmbeddingCacheSlots: 0,
    hotEmbeddingSlabBytes: 0,
    kvPrefixSlabBudgetBytes: 1024 * 1024,
    maxKvPrefixBlocks: 8,
    bytesPerKvBlock: 1024,
  };
  const baseGeom = {
    numLayers: 2,
    nHeadsKv: 2,
    headDim: 4,
    pageTokens: 16,
    q8BytesPerHalfPage: 2 * 16 * 2 * 4,
    scaleFloatsPerHalfPage: 2 * 16 * 2,
  };

  // 1. Model alias A writes KV prefix blocks keyed by shared contentFingerprint
  const storeA = new VramUnifiedStore(
    { ...baseGeom, modelId: 'catalog-qwen3-alias', contentFingerprint: sharedFp },
    plan,
    4 * 1024 * 1024,
  );
  const tokens = Array.from({ length: 32 }, (_, i) => i + 10);
  const keysA = storeA.computePrefixKeys(tokens);
  assert.equal(keysA.length, 2);
  storeA.putBlock(keysA[0].key, 0, keysA[0].slice, makeMockBlock(1, baseGeom.q8BytesPerHalfPage, baseGeom.scaleFloatsPerHalfPage));
  storeA.putBlock(keysA[1].key, 1, keysA[1].slice, makeMockBlock(2, baseGeom.q8BytesPerHalfPage, baseGeom.scaleFloatsPerHalfPage));
  assert.equal(await storeA.persistToOpfs(dir), 2);
  assert.equal(files.has(`${sharedFp}.unified.kv`), true, 'Expected KV cache persisted under contentFingerprint');

  // 2. Renamed custom-URL alias B hydrates from the SAME contentFingerprint file and hits 100% of A's KV blocks!
  const storeB = new VramUnifiedStore(
    { ...baseGeom, modelId: 'custom-mirror-alias', contentFingerprint: sharedFp },
    plan,
    4 * 1024 * 1024,
  );
  assert.equal(await storeB.hydrateFromOpfs(dir), 2);
  const chainB = storeB.lookupPrefixChain(tokens);
  assert.equal(chainB.length, 2, 'Expected renamed alias B to hit 100% of alias A KV prefix blocks');

  // 3. Legacy upgrade: a store written before contentFingerprint (under `<modelId>.unified.kv`) hydrates
  //    into a fingerprinted store and transparently re-chains keys to `contentFingerprint`!
  const { dir: legacyDir } = createMockOpfsDir();
  const legacyWriter = new VramUnifiedStore(
    { ...baseGeom, modelId: 'legacy-smollm' },
    plan,
    4 * 1024 * 1024,
  );
  const legacyK0 = computePrefixBlockHash('legacy-smollm', 'root', tokens.slice(0, 16));
  const legacyK1 = computePrefixBlockHash('legacy-smollm', legacyK0, tokens.slice(16, 32));
  legacyWriter.putBlock(legacyK0, 0, tokens.slice(0, 16), makeMockBlock(7, baseGeom.q8BytesPerHalfPage, baseGeom.scaleFloatsPerHalfPage));
  legacyWriter.putBlock(legacyK1, 1, tokens.slice(16, 32), makeMockBlock(8, baseGeom.q8BytesPerHalfPage, baseGeom.scaleFloatsPerHalfPage));
  assert.equal(await legacyWriter.persistToOpfs(legacyDir), 2);

  const upgraded = new VramUnifiedStore(
    { ...baseGeom, modelId: 'legacy-smollm', contentFingerprint: sharedFp },
    plan,
    4 * 1024 * 1024,
  );
  assert.equal(await upgraded.hydrateFromOpfs(legacyDir), 2);
  const upgradedChain = upgraded.lookupPrefixChain(tokens);
  assert.equal(upgradedChain.length, 2, 'Expected legacy KV blocks to re-chain under contentFingerprint');
});

test('Zipfian hot embedding row cache (VEMB) shares warmed embeddings across reloads with 0 backing reads', async () => {
  const modelPath = path.resolve(process.cwd(), 'public/models/stories260K.gguf');
  const fileBytes = new Uint8Array(fs.readFileSync(modelPath));
  const { dir, files } = createMockOpfsDir();

  const buildPager = (backing: MemoryBackingStore) => {
    const parsed = parseGgufFromBacking(backing);
    const embdInfo = parsed.tensorMap.get('token_embd.weight')!;
    const outNormInfo = parsed.tensorMap.get('output_norm.weight')!;
    const outWeightInfo = parsed.tensorMap.get('output.weight') ?? embdInfo;
    const isTiedOutput = !parsed.tensorMap.has('output.weight');
    const layerInfos: LayerTensorInfoGroup[] = [];
    for (let l = 0; l < parsed.hparams.blockCount; l++) {
      layerInfos.push({
        attnNorm: parsed.tensorMap.get(`blk.${l}.attn_norm.weight`)!,
        attnQ: parsed.tensorMap.get(`blk.${l}.attn_q.weight`)!,
        attnQBias: null,
        attnQNorm: null,
        attnK: parsed.tensorMap.get(`blk.${l}.attn_k.weight`)!,
        attnKBias: null,
        attnKNorm: null,
        attnV: parsed.tensorMap.get(`blk.${l}.attn_v.weight`)!,
        attnVBias: null,
        attnOutput: parsed.tensorMap.get(`blk.${l}.attn_output.weight`)!,
        attnOutputBias: null,
        ffnNorm: parsed.tensorMap.get(`blk.${l}.ffn_norm.weight`)!,
        ffnGate: parsed.tensorMap.get(`blk.${l}.ffn_gate.weight`)!,
        ffnUp: parsed.tensorMap.get(`blk.${l}.ffn_up.weight`)!,
        ffnDown: parsed.tensorMap.get(`blk.${l}.ffn_down.weight`)!,
        ffnGateInp: null,
        ffnGateExps: null,
        ffnUpExps: null,
        ffnDownExps: null,
      });
    }
    const reg = new VTensorBackendRegistry();
    const pager = new VramWeightPager(
      parsed,
      backing,
      embdInfo,
      outNormInfo,
      outWeightInfo,
      isTiedOutput,
      layerInfos,
      {
        residencyMode: 'opfs-paged',
        layerFrameSlots: 2,
        streamEmbeddings: true,
        streamLmHead: true,
        lmHeadStripeRows: 64,
        hotEmbeddingCacheSlots: 32,
      },
      (id, usage, cap) => reg.allocBuffer(id, usage, cap),
    );
    return { pager, reg };
  };

  const backingA = new MemoryBackingStore('alias-embd-a', fileBytes);
  const fp = computeBackingContentFingerprint(backingA);
  const { pager: pagerA, reg: regA } = buildPager(backingA);
  try {
    const hotTokens = [1, 4, 12, 27, 42];
    for (const tid of hotTokens) {
      pagerA.acquireEmbeddingRow(tid);
    }
    const sig = pagerA.getHotEmbeddingSignature();
    assert.ok(sig !== null);
    const exported = pagerA.exportHotEmbeddingRows(32);
    assert.equal(exported.length, hotTokens.length);

    // Binary codec invalidation check
    const rawBin = serializeHotEmbeddingsCache(sig, exported);
    assert.equal(deserializeHotEmbeddingsCache(rawBin, { ...sig, embdRowBytes: sig.embdRowBytes + 4 }), null);

    assert.equal(await writeHotEmbeddingsCacheToOpfs(fp, sig, exported, dir), true);
    assert.equal(files.has(`${fp}.embd.bin`), true);
  } finally {
    regA.freeAllBuffers();
  }

  // Alias B hydrates hot embeddings from OPFS and hits ALL 5 tokens with 0 backing misses!
  const backingB = new MemoryBackingStore('alias-embd-b', fileBytes);
  const { pager: pagerB, reg: regB } = buildPager(backingB);
  try {
    const sigB = pagerB.getHotEmbeddingSignature()!;
    const rows = await readHotEmbeddingsCacheFromOpfs(fp, sigB, dir);
    assert.ok(rows !== null && rows.length === 5);
    assert.equal(pagerB.hydrateHotEmbeddingRows(rows), 5);
    for (const tid of [1, 4, 12, 27, 42]) {
      pagerB.acquireEmbeddingRow(tid);
    }
    const tel = pagerB.getTelemetry();
    assert.equal(tel.embdCacheHits, 5, 'Expected 5/5 hot embedding cache hits after OPFS hydration');
    assert.equal(tel.embdCacheMisses, 0, 'Expected 0 backing reads for hydrated hot embeddings');
  } finally {
    regB.freeAllBuffers();
  }
});

test('LlamaVTensorRuntime.loadFromBackingAsync shares .hdr.bin, .rope.bin, .hot.bin, and .unified.kv across renamed aliases with 0 header/hot-meta Range fetches', async () => {
  const modelPath = path.resolve(process.cwd(), 'public/models/stories260K.gguf');
  const fileBytes = new Uint8Array(fs.readFileSync(modelPath));
  const { dir, files } = createMockOpfsDir();
  const slabBytes = 32 * 1024;

  const openDemand = async (
    modelId: string,
    stats: { largeHeaderDirectCalls: number; hotMetaDirectCalls: number },
  ) => {
    const cacheHandle = (await dir.getFileHandle(`${modelId}.vram.cache`, {
      create: true,
    })) as FileSystemFileHandle & { createSyncAccessHandle: () => Promise<SyncAccessHandleLike> };
    const syncHandle = await cacheHandle.createSyncAccessHandle();
    const store = new OpfsDemandVramBackingStore({
      modelId,
      url: `https://mirror.test/${modelId}.gguf`,
      totalBytes: fileBytes.byteLength,
      handle: syncHandle,
      maxCacheBytes: slabBytes * 12,
      slabBytes,
      asyncRangeFetcher: async (_u, start, end) => fileBytes.slice(start, end + 1),
      syncRangeFetcher: (_u, start, end) => fileBytes.slice(start, end + 1),
    });
    const origDirect = store.readRangesDirectAsync.bind(store);
    store.readRangesDirectAsync = async (ranges, opts) => {
      if (ranges.length > 1) {
        stats.hotMetaDirectCalls += 1;
      } else if (ranges.length === 1 && ranges[0].byteLength > 4096) {
        stats.largeHeaderDirectCalls += 1;
      }
      return origDirect(ranges, opts);
    };
    return store;
  };

  // 1. Cold boot under alias A: parses full header, computes RoPE, fetches hot metadata, warms KV prefix
  const statsA = { largeHeaderDirectCalls: 0, hotMetaDirectCalls: 0 };
  const storeA = await openDemand('catalog-stories', statsA);
  const rtA = new LlamaVTensorRuntime();
  const prompt = 'Once upon a time there was a little green frog who lived near a sunny pond in the forest and';
  const tokensA: number[] = [];
  let fp = '';
  try {
    await rtA.loadFromBackingAsync('catalog-stories', 'Stories A', 'opfs-demand://a', storeA, {
      weightResidencyPolicy: 'paged',
      opfsDirectoryOverride: dir,
    });
    fp = rtA.getContentFingerprint() ?? '';
    assert.ok(fp.length > 16, 'Expected deterministic contentFingerprint');
    assert.equal(statsA.largeHeaderDirectCalls, 1, 'Cold boot must read full GGUF header once');
    assert.equal(statsA.hotMetaDirectCalls, 1, 'Cold boot must fetch hot metadata once');
    assert.equal(files.has(`${fp}.hdr.bin`), true, 'Expected .hdr.bin persisted');
    assert.equal(files.has(`${fp}.rope.bin`), true, 'Expected .rope.bin persisted');
    assert.equal(files.has(`${fp}.hot.bin`), true, 'Expected .hot.bin persisted');

    for await (const ev of rtA.generate(prompt, {
      temperature: 0,
      topP: 1,
      topK: 1,
      repeatPenalty: 1.0,
      maxTokens: 6,
      fullContextOutput: false,
    })) {
      tokensA.push(ev.tokenId);
    }
    const persistedKv = await rtA.persistUnifiedStoreToOpfs(dir);
    assert.ok(persistedKv >= 1, 'Expected at least 1 KV prefix block persisted');
    assert.equal(files.has(`${fp}.unified.kv`), true);
  } finally {
    rtA.unload();
  }

  // 2. Warm boot under DIFFERENT alias B: skips full header parse, skips RoPE trig, skips hot-meta Range fetch, AND shares KV prefixes!
  const statsB = { largeHeaderDirectCalls: 0, hotMetaDirectCalls: 0 };
  const storeB = await openDemand('custom-url-stories', statsB);
  const rtB = new LlamaVTensorRuntime();
  const tokensB: number[] = [];
  try {
    await rtB.loadFromBackingAsync('custom-url-stories', 'Stories B', 'opfs-demand://b', storeB, {
      weightResidencyPolicy: 'paged',
      opfsDirectoryOverride: dir,
    });
    assert.equal(rtB.getContentFingerprint(), fp);
    assert.equal(
      statsB.largeHeaderDirectCalls,
      0,
      'Expected alias B to skip full GGUF header read via shared <fp>.hdr.bin',
    );
    assert.equal(
      statsB.hotMetaDirectCalls,
      0,
      'Expected alias B to skip hot-metadata Range fetches via shared <fp>.hot.bin',
    );
    const hydratedKv = await rtB.hydrateUnifiedStoreFromOpfs(dir);
    assert.ok(hydratedKv >= 1, 'Expected alias B to hydrate alias A KV prefix blocks');

    for await (const ev of rtB.generate(prompt, {
      temperature: 0,
      topP: 1,
      topK: 1,
      repeatPenalty: 1.0,
      maxTokens: 6,
      fullContextOutput: false,
    })) {
      tokensB.push(ev.tokenId);
    }
    assert.deepEqual(tokensB, tokensA, 'Expected bit-exact greedy tokens across deduplicated boot & KV caches');
  } finally {
    rtB.unload();
  }
});

test('deleteOpfsModel preserves shared fingerprinted boot/KV sidecars while another committed model references them, then cleans up on last delete', async () => {
  const { dir, files } = createMockOpfsDir();
  const headerHash64 = '1122334455667788';
  const totalBytes = 4096;
  const fp = `${headerHash64}-${totalBytes}`;

  files.set('model-a.gguf', new Uint8Array(64));
  files.set(
    'model-a.meta',
    new TextEncoder().encode(
      JSON.stringify({
        version: 1,
        modelId: 'model-a',
        url: 'https://cdn.test/a.gguf',
        totalBytes,
        headerHash64,
        committedAt: Date.now(),
      }),
    ),
  );
  files.set('model-b.gguf', new Uint8Array(64));
  files.set(
    'model-b.meta',
    new TextEncoder().encode(
      JSON.stringify({
        version: 1,
        modelId: 'model-b',
        url: 'https://cdn.test/b.gguf',
        totalBytes,
        headerHash64,
        committedAt: Date.now(),
      }),
    ),
  );
  for (const suffix of ['.hdr.bin', '.rope.bin', '.hot.bin', '.embd.bin', '.unified.kv']) {
    files.set(`${fp}${suffix}`, new Uint8Array([1, 2, 3, 4]));
  }

  // Deleting model-a keeps shared `<fp>.*` caches because model-b still references the fingerprint!
  await deleteOpfsModel('model-a', dir);
  assert.equal(files.has('model-a.gguf'), false);
  assert.equal(files.has(`${fp}.hdr.bin`), true);
  assert.equal(files.has(`${fp}.unified.kv`), true);

  // Deleting model-b (last reference) cleans up all shared `<fp>.*` caches!
  await deleteOpfsModel('model-b', dir);
  assert.equal(files.has('model-b.gguf'), false);
  for (const suffix of ['.hdr.bin', '.rope.bin', '.hot.bin', '.embd.bin', '.unified.kv']) {
    assert.equal(files.has(`${fp}${suffix}`), false, `Expected ${fp}${suffix} cleaned up on last delete`);
  }
});
