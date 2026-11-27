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
});
