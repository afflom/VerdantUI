import test from 'node:test';
import assert from 'node:assert/strict';
import {
  MemoryCasBackend,
  OpfsCasStore,
  computeCasKappa,
  computeCasUtilityPerByte,
} from './vramCasStore';
import {
  MODEL_CATALOG,
  getModelCapacityQ10,
  getModelStreamingStrategy,
  shouldPreferDemandStreaming,
} from './modelCatalog';

function makeSlab(seed: number, length = 4096): Uint8Array {
  const out = new Uint8Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = (seed * 31 + i * 17) & 0xff;
  }
  return out;
}

test('OpfsCasStore deduplicates identical slabs across models with 0 duplicate physical writes and 0 redundant fetches', async () => {
  const backend = new MemoryCasBackend();
  const cas = new OpfsCasStore({ backend, budgetBytes: 1024 * 1024 });

  const sharedBytes = makeSlab(7, 8192);
  let fetchesA = 0;
  const first = await cas.resolveOrStoreSlab({
    contentFingerprint: 'fp-model-a',
    slabIndex: 0,
    ownerModelId: 'qwen3-4b-q4_k_m',
    salienceQ10: 700,
    fetchFn: async () => {
      fetchesA += 1;
      return sharedBytes;
    },
  });
  assert.equal(fetchesA, 1);
  assert.equal(first.hit, false);
  assert.equal(first.dedupLinked, false);
  assert.equal(backend.physicalObjectCount(), 1);

  // Same fingerprint + slabIndex hits extent cache with 0 network calls
  const hit = await cas.resolveOrStoreSlab({
    contentFingerprint: 'fp-model-a',
    slabIndex: 0,
    ownerModelId: 'qwen3-4b-q4_k_m',
    fetchFn: async () => {
      fetchesA += 1;
      return sharedBytes;
    },
  });
  assert.equal(fetchesA, 1, 'Extent cache hit must not invoke fetchFn');
  assert.equal(hit.hit, true);
  assert.equal(hit.kappa, first.kappa);

  // Different model / mirror with identical slab bytes links refOwners with 0 duplicate physical objects
  const second = await cas.resolveOrStoreSlab({
    contentFingerprint: 'fp-model-b-mirror',
    slabIndex: 3,
    ownerModelId: 'qwen3-30b-a3b-q4_k_m',
    salienceQ10: 850,
    fetchFn: async () => sharedBytes.slice(),
  });
  assert.equal(second.kappa, first.kappa);
  assert.equal(second.dedupLinked, true);
  assert.equal(backend.physicalObjectCount(), 1, 'Identical kappa must NEVER duplicate physical storage');

  const entry = cas.getEntry(first.kappa)!;
  assert.equal(entry.refOwners.length, 2);
  assert.ok(entry.refOwners.includes('qwen3-4b-q4_k_m'));
  assert.ok(entry.refOwners.includes('qwen3-30b-a3b-q4_k_m'));
  assert.ok(cas.getTelemetry().bytesSaved >= sharedBytes.byteLength * 2);
});

test('OpfsCasStore evicts lowest utility-per-byte entries under capacity ceiling while protecting active kappas', async () => {
  const backend = new MemoryCasBackend();
  const cas = new OpfsCasStore({ backend, budgetBytes: 14_000 });

  // Small high-salience tokenizer (high utility / byte)
  const tokBytes = makeSlab(1, 2048);
  const tok = await cas.putObject({
    kind: 'tokenizer-bpe',
    bytes: tokBytes,
    ownerId: 'clip-bpe',
    salienceQ10: 960,
  });

  // Large low-salience cold slab (lowest utility / byte)
  const coldSlab = makeSlab(2, 6144);
  const cold = await cas.putObject({
    kind: 'weight-slab',
    bytes: coldSlab,
    ownerId: 'cold-model',
    salienceQ10: 120,
  });

  // Protected slab (total resident: 13,312 bytes <= 14,000 budget)
  const pinnedSlab = makeSlab(3, 5120);
  const pinned = await cas.putObject({
    kind: 'weight-slab',
    bytes: pinnedSlab,
    ownerId: 'active-moe',
    salienceQ10: 200,
    protectedKappas: new Set([tok.kappa]),
  });

  assert.ok(
    computeCasUtilityPerByte(cas.getEntry(tok.kappa)!) >
      computeCasUtilityPerByte(cas.getEntry(cold.kappa)!),
  );

  // Adding another 5 KiB slab forces eviction over 14 KiB budget -> coldSlab is evicted first, tok & pinned stay!
  const freshSlab = makeSlab(4, 5120);
  await cas.putObject({
    kind: 'weight-slab',
    bytes: freshSlab,
    ownerId: 'active-moe',
    salienceQ10: 800,
    protectedKappas: new Set([pinned.kappa, tok.kappa]),
  });

  assert.equal(cas.getEntry(cold.kappa), undefined, 'Cold lowest-utility slab must be evicted first');
  assert.ok(cas.getEntry(tok.kappa), 'High-utility tokenizer must survive eviction');
  assert.ok(cas.getEntry(pinned.kappa), 'Protected kappa must survive eviction');
  assert.ok(cas.getResidentBytes() <= 14_000);
  assert.ok(cas.getTelemetry().evictions >= 1);

  // Release owner cleans up unreferenced objects while keeping shared objects alive
  const sharedKappa = computeCasKappa('boot-header', makeSlab(9, 1024));
  await cas.putObject({ kind: 'boot-header', bytes: makeSlab(9, 1024), ownerId: 'm1', salienceQ10: 900 });
  await cas.putObject({ kind: 'boot-header', bytes: makeSlab(9, 1024), ownerId: 'm2', salienceQ10: 900 });
  await cas.releaseOwner('m1');
  assert.ok(cas.getEntry(sharedKappa), 'Shared object must stay alive while m2 holds a ref');
  await cas.releaseOwner('m2');
  assert.equal(cas.getEntry(sharedKappa), undefined, 'Object with 0 remaining owners must be reclaimed');
});

test('Model catalog marks Sparse MoE and large models for demand-first streaming with smooth monotonic capacity Q10', () => {
  const moe30b = MODEL_CATALOG.find((m) => m.id === 'qwen3-30b-a3b-q4_k_m');
  const qwen8b = MODEL_CATALOG.find((m) => m.id === 'qwen3-8b-q4_k_m');
  const qwen4b = MODEL_CATALOG.find((m) => m.id === 'qwen3-4b-q4_k_m');
  const qwen05b = MODEL_CATALOG.find((m) => m.id === 'qwen2.5-0.5b-instruct-q4_k_m');
  assert.ok(moe30b && qwen8b && qwen4b && qwen05b);

  assert.equal(getModelStreamingStrategy(moe30b), 'demand-sparse-moe');
  assert.equal(shouldPreferDemandStreaming(moe30b), true);
  assert.equal(getModelStreamingStrategy(qwen8b), 'demand-layered');
  assert.equal(shouldPreferDemandStreaming(qwen8b), true);
  assert.equal(getModelStreamingStrategy(qwen05b), 'compact-resident');
  assert.equal(shouldPreferDemandStreaming(qwen05b), false);

  assert.ok(getModelCapacityQ10(qwen05b) < getModelCapacityQ10(qwen4b));
  assert.ok(getModelCapacityQ10(qwen4b) < getModelCapacityQ10(qwen8b));
  assert.ok(getModelCapacityQ10(qwen8b) < getModelCapacityQ10(moe30b));
});
