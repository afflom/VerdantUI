import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { GgmlType } from './gguf';
import {
  discoverRemoteExpertManifest,
  planPriorityWeightExtents,
  selectInitialUnifiedBackbone,
} from './unifiedExpertScheduler';
import { MemoryCasBackend, OpfsCasStore } from './vramCasStore';
import {
  OpfsDemandVramBackingStore,
  type SyncAccessHandleLike,
} from './vramBackingStore';
import {
  derivePerformanceContract,
  type EnvironmentProfile,
  type ModelGeometrySpec,
} from './performanceContract';

test('discoverRemoteExpertManifest parses real GGUF header via HTTP Range with zero full-file download and CAS slab-0 registration', async () => {
  const ggufPath = resolve(process.cwd(), 'public/models/stories260K.gguf');
  const rawFile = new Uint8Array(await readFile(ggufPath));
  const totalBytes = rawFile.byteLength;
  const rangeRequests: string[] = [];

  const mockFetch: typeof fetch = async (_input, init) => {
    const hdrs = init?.headers as Record<string, string> | undefined;
    const range = hdrs?.Range ?? hdrs?.range ?? '';
    rangeRequests.push(range);
    const match = /^bytes=(\d+)-(\d+)$/.exec(range);
    assert.ok(match, `Expected explicit HTTP Range header, got "${range}"`);
    const start = Number(match[1]);
    const end = Math.min(totalBytes - 1, Number(match[2]));
    const slice = rawFile.subarray(start, end + 1);
    return new Response(slice, {
      status: 206,
      headers: {
        'Content-Range': `bytes ${start}-${end}/${totalBytes}`,
        ETag: '"stories260k-v1"',
      },
    });
  };

  const backend = new MemoryCasBackend();
  const cas = new OpfsCasStore({ backend, budgetBytes: 8 * 1024 * 1024 });

  const manifest = await discoverRemoteExpertManifest({
    url: 'https://example.invalid/stories260K.gguf',
    modelId: 'stories260k-probe',
    fetchFn: mockFetch,
    casStore: cas,
  });

  assert.equal(rangeRequests.length, 1);
  assert.equal(manifest.architecture, 'llama');
  assert.equal(manifest.blockCount, 5);
  assert.equal(manifest.fileByteLength, totalBytes);
  assert.ok(
    manifest.headerBytesFetched <= 65_536 && manifest.headerBytesFetched < totalBytes,
    `Expected compact header-only probe (< file size), fetched ${manifest.headerBytesFetched}/${totalBytes}`,
  );
  assert.equal(manifest.isMoe, false);
  assert.equal(manifest.etag, '"stories260k-v1"');
  assert.equal(backend.physicalObjectCount(), 1, 'Header slab-0 must be registered in OPFS CAS');
});

test('planPriorityWeightExtents orders early attention -> body -> LM-head, caps slice sizes <= 4 MiB, skips cached ranges, and excludes cold MoE experts', () => {
  const MB = 1024 * 1024;
  // Dense 6-layer model: layers 0-1 are early-attention, layers 2-5 are body-layers
  const densePlan = planPriorityWeightExtents({
    totalLayers: 6,
    isMoe: false,
    getLayerRanges: (l) => [{ fileOffset: (l + 1) * 10 * MB, byteLength: 6 * MB }],
    lmHeadStripes: [{ fileOffset: 100 * MB, byteLength: 8 * MB }],
    // Mark layer 0 as already cached in OPFS
    isRangeCached: (off) => off === 10 * MB,
    maxSliceBytes: 4 * MB,
  });

  assert.equal(densePlan.alreadyCachedBytes, 6 * MB);
  assert.equal(densePlan.totalTargetBytes, 6 * 6 * MB + 8 * MB);
  assert.equal(densePlan.missingBytes, 5 * 6 * MB + 8 * MB);
  assert.ok(densePlan.slices.every((s) => s.byteLength <= 4 * MB), 'Every prewarm slice must be <= 4 MiB');

  const tiers = densePlan.slices.map((s) => s.tier);
  const firstBody = tiers.indexOf('body-layers');
  const firstLm = tiers.indexOf('lm-head');
  assert.ok(firstBody > 0, 'Early attention slices must precede body layers');
  assert.ok(firstLm > firstBody, 'Body layer slices must precede LM-head stripes');
  assert.ok(
    !densePlan.slices.some((s) => s.layerIndex === 0),
    'Already-cached layer 0 must be omitted from prewarm slices',
  );

  // Sparse MoE model: all attention+router layers are early-attention priority (cold FFN experts stay virtual)
  const moePlan = planPriorityWeightExtents({
    totalLayers: 4,
    isMoe: true,
    getLayerRanges: (l) => [{ fileOffset: (l + 1) * 4 * MB, byteLength: 1 * MB }],
    lmHeadStripes: [{ fileOffset: 500 * MB, byteLength: 2 * MB }],
    maxSliceBytes: 4 * MB,
  });
  assert.equal(moePlan.slices.filter((s) => s.tier === 'body-layers').length, 0);
  assert.equal(moePlan.slices.filter((s) => s.tier === 'early-attention').length, 4);
  assert.equal(moePlan.slices.filter((s) => s.tier === 'lm-head').length, 1);
});

test('selectInitialUnifiedBackbone prioritizes remembered/warm OPFS models and picks demand-first backbone immediately on cold consent', () => {
  const remembered = selectInitialUnifiedBackbone({
    cachedModelIds: ['qwen2.5-1.5b-instruct-q4_k_m', 'qwen3-8b-q4_k_m'],
    lastModelId: 'qwen3-8b-q4_k_m',
    hardwareConcurrency: 8,
  });
  assert.equal(remembered.id, 'qwen3-8b-q4_k_m');

  const warmFallback = selectInitialUnifiedBackbone({
    cachedModelIds: ['qwen2.5-1.5b-instruct-q4_k_m'],
    lastModelId: null,
    hardwareConcurrency: 8,
  });
  assert.equal(warmFallback.id, 'qwen2.5-1.5b-instruct-q4_k_m');

  const coldDesktop = selectInitialUnifiedBackbone({
    cachedModelIds: [],
    lastModelId: null,
    hardwareConcurrency: 8,
  });
  assert.equal(coldDesktop.id, 'qwen3-4b-q4_k_m');

  const coldLowCore = selectInitialUnifiedBackbone({
    cachedModelIds: [],
    lastModelId: null,
    hardwareConcurrency: 2,
  });
  assert.equal(coldLowCore.id, 'qwen2.5-1.5b-instruct-q4_k_m');
});

test('derivePerformanceContract enforces opfs-paged + streamed embeddings/LM-head on opfs-demand backings (zero monolithic upfront pull)', () => {
  const env: EnvironmentProfile = {
    simdTier: 'relaxed-vnni',
    relaxedSimdSupported: true,
    hardwareConcurrency: 8,
    sharedMemorySupported: true,
    effectiveWorkers: 6,
    memoryBandwidthGBs: 24,
    computeGops: 120,
    ridgeIntensityOpsPerByte: 5,
    workerSyncLatencyUs: 15,
    memoryBudgetBytes: 3_500_000_000,
    calibrationMs: 2,
  };
  const model: ModelGeometrySpec = {
    hparams: {
      architecture: 'qwen3',
      name: 'Qwen3-4B',
      vocabSize: 151_936,
      contextLength: 40_960,
      embeddingLength: 2560,
      feedForwardLength: 9728,
      blockCount: 36,
      headCount: 32,
      headCountKv: 8,
      headDim: 128,
      rmsNormEps: 1e-6,
      ropeFreqBase: 1_000_000,
      ropeFreqScale: 1,
      ropeDimensionCount: 128,
      ropeType: 'neox',
      bosTokenId: 151_643,
      eosTokenId: 151_645,
      unkTokenId: 0,
      addBosToken: false,
      tokenizerModel: 'gpt2',
      preTokenizer: 'qwen2',
      chatTemplate: null,
      primaryQuantization: 'Q4_K',
    },
    activeWeightBytes: 2_400_000_000,
    lmHeadWeightBytes: 210_000_000,
    activeParamCount: 4_000_000_000,
    totalWeightBufferBytes: 2_400_000_000,
    dominantWeightType: GgmlType.Q4_K,
    quantizedWeightFraction: 0.99,
  };
  const contract = derivePerformanceContract(env, model, {
    backingKind: 'opfs-demand',
    weightResidencyPolicy: 'auto',
  });

  assert.equal(contract.weightResidencyMode, 'opfs-paged');
  assert.equal(contract.streamEmbeddings, true);
  assert.equal(contract.streamLmHead, true);
  assert.equal(
    contract.lazyPinLmHead,
    true,
    'opfs-demand auto residency must enable lazyPinLmHead when full-vocab LM-head fits in WASM budget (0 OPFS reads on tokens 2..N)',
  );
  assert.equal(
    contract.opfsIoOverheadMsPerToken,
    0,
    'When all layers and lazy-pinned LM-head fit in WASM RAM, steady-state per-token OPFS I/O overhead must be 0 ms',
  );
  assert.equal(
    contract.weightLayerFrameSlots,
    36,
    'opfs-demand auto residency must pin all 36 layers when they fit in WASM budget (0 OPFS layer reads per decode token)',
  );
});

test('OpfsDemandVramBackingStore coalesces contiguous missing slabs into multi-slab Range fetches and batches OPFS flushes', async () => {
  const slabBytes = 4096;
  const totalSlabs = 10;
  const totalBytes = slabBytes * totalSlabs;
  const remoteFile = new Uint8Array(totalBytes);
  for (let i = 0; i < totalBytes; i++) remoteFile[i] = (i * 13 + 7) & 0xff;

  let backingData = new Uint8Array(0);
  let flushCalls = 0;
  const memHandle: SyncAccessHandleLike = {
    getSize: () => backingData.byteLength,
    truncate: (sz: number) => {
      const next = new Uint8Array(sz);
      next.set(backingData.subarray(0, Math.min(sz, backingData.byteLength)));
      backingData = next;
    },
    read: (buf, opts) => {
      const view =
        buf instanceof ArrayBuffer
          ? new Uint8Array(buf)
          : new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
      const off = opts?.at ?? 0;
      const len = Math.min(view.byteLength, Math.max(0, backingData.byteLength - off));
      view.set(backingData.subarray(off, off + len), 0);
      return len;
    },
    write: (buf, opts) => {
      const view =
        buf instanceof ArrayBuffer
          ? new Uint8Array(buf)
          : new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
      const off = opts?.at ?? 0;
      const end = off + view.byteLength;
      if (end > backingData.byteLength) {
        const next = new Uint8Array(end);
        next.set(backingData);
        backingData = next;
      }
      backingData.set(view, off);
      return view.byteLength;
    },
    flush: () => {
      flushCalls += 1;
    },
    close: () => {},
  };

  const rangeCalls: Array<{ start: number; end: number }> = [];
  const store = new OpfsDemandVramBackingStore({
    modelId: 'coalesce-test',
    url: 'https://example.invalid/model.gguf',
    totalBytes,
    handle: memHandle,
    maxCacheBytes: totalBytes,
    slabBytes,
    asyncRangeFetcher: async (_url, start, end) => {
      rangeCalls.push({ start, end });
      return remoteFile.subarray(start, end + 1);
    },
  });

  // Prefetch 8 contiguous slabs -> MUST coalesce into 1 single HTTP Range fetch (not 8!)
  await store.prefetchRanges([{ fileOffset: 0, byteLength: 8 * slabBytes }]);
  assert.equal(
    rangeCalls.length,
    1,
    `Expected 8 contiguous slabs to coalesce into 1 HTTP Range fetch, got ${rangeCalls.length}`,
  );
  assert.equal(rangeCalls[0]!.start, 0);
  assert.equal(rangeCalls[0]!.end, 8 * slabBytes - 1);
  // Batched flush: < 32 slabs written -> 0 blocking fsync flushes during mid-pass prefetch!
  assert.equal(flushCalls, 0);

  // Verify bit-exact readback across all 8 slabs
  const readBack = new Uint8Array(8 * slabBytes);
  assert.equal(store.readRange(0, readBack, 0, readBack.byteLength), readBack.byteLength);
  assert.deepEqual(readBack, remoteFile.subarray(0, 8 * slabBytes));

  // Explicit flush persists cleanly once
  store.flush();
  assert.equal(flushCalls, 1);
  store.close();
});
