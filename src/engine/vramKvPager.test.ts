import test from 'node:test';
import assert from 'node:assert/strict';
import { GgmlType } from './gguf';
import { verdantFlashAttnExt } from './verdantKernels';
import { VTensorBackendRegistry } from './vtensorProvider';
import {
  admitColdKvPage,
  VramKvPager,
  vramMembership,
  VramPageState,
  vramTrim,
} from './vramKvPager';

test('vRAM primitives: 64-bit residency bitset, trim governor, and challenger admission gate', () => {
  let word = 0n;
  word = vramMembership(word, 0, true);
  word = vramMembership(word, 5, true);
  assert.equal(word, 0x21n);
  word = vramMembership(word, 5, false);
  assert.equal(word, 0x01n);

  assert.equal(vramTrim(4, 4), false);
  assert.equal(vramTrim(4, 5), true);

  assert.equal(admitColdKvPage(1024, 272, 0.004, 1.0), true);
  assert.equal(admitColdKvPage(1024, 1024, 0.0, 1.0), false); // must strictly reduce bytes
  assert.equal(admitColdKvPage(1024, 272, 0.5, 1.0), false); // rejects excessive reconstruction drift
});

test('VramKvPager reclaims cold pages into Q8 backing, preserves attention accuracy, and refaults on truncation', () => {
  const reg = new VTensorBackendRegistry({ initialWasmPages: 16 });
  const headDim = 32;
  const nHeads = 4;
  const nHeadsKv = 2;
  const kvDim = nHeadsKv * headDim;
  const maxCtx = 96;

  const pager = new VramKvPager(
    { numLayers: 2, maxContext: maxCtx, nHeadsKv, headDim, hotFrameLimit: 3 },
    (id, usage, cap) => reg.allocBuffer(id, usage, cap),
  );

  const fullK = new Float32Array(2 * maxCtx * kvDim);
  const fullV = new Float32Array(2 * maxCtx * kvDim);
  const tmpK = new Float32Array(kvDim);
  const tmpV = new Float32Array(kvDim);

  // Fill 76 positions (5 pages -> forces automatic reclaim of pages 1 and 2 while keeping page 0 sink hot)
  for (let pos = 0; pos < 76; pos++) {
    for (let l = 0; l < 2; l++) {
      for (let d = 0; d < kvDim; d++) {
        tmpK[d] = Math.sin(pos * 0.19 + l * 1.1 + d * 0.07);
        tmpV[d] = Math.cos(pos * 0.13 - l * 0.7 + d * 0.11);
      }
      fullK.set(tmpK, (l * maxCtx + pos) * kvDim);
      fullV.set(tmpV, (l * maxCtx + pos) * kvDim);
      pager.writeKv(false, l, pos, tmpK);
      pager.writeKv(true, l, pos, tmpV);
    }
  }

  const tel = pager.getTelemetry();
  assert.equal(pager.getPageState(0), VramPageState.HOT_RESIDENT); // sink page pinned
  assert.equal(pager.getPageState(1), VramPageState.COLD_COMPRESSED);
  assert.equal(pager.getPageState(2), VramPageState.COLD_COMPRESSED);
  assert.equal(tel.hotResidentPages, 3);
  assert.equal(tel.coldCompressedPages, 2);
  assert.ok(tel.reclaimCycles >= 2);
  assert.ok(tel.savedBytes > 0);
  assert.ok(tel.compressionRatio > 1.2);

  // Compare fused hot F32 + cold Q8 WASM SIMD attention against uncompressed F32 reference
  const scratch = reg.allocBuffer('test.scratch', 'compute', 2048);
  const qH = scratch.allocTensor('q', GgmlType.F32, [nHeads * headDim]);
  const oH = scratch.allocTensor('o', GgmlType.F32, [nHeads * headDim]);
  const qView = scratch.f32View(qH);
  for (let i = 0; i < qView.length; i++) qView[i] = Math.sin(i * 0.27);

  for (let l = 0; l < 2; l++) {
    const refOut = new Float32Array(nHeads * headDim);
    verdantFlashAttnExt(qView, fullK, fullV, refOut, nHeads, nHeadsKv, headDim, 76, l * maxCtx * kvDim);
    pager.flashAttnExt(
      qView,
      scratch.wasmPtr(qH),
      scratch.f32View(oH),
      scratch.wasmPtr(oH),
      nHeads,
      l,
      76,
      reg.arena,
    );
    const got = scratch.f32View(oH);
    for (let i = 0; i < got.length; i++) {
      assert.ok(Math.abs(got[i] - refOut[i]) < 0.02, `Layer ${l} dim ${i} drift ${Math.abs(got[i] - refOut[i])}`);
    }
  }

  // Truncating back into cold page 1 (pos 22) should release pages 2..4 and refault page 1 hot!
  pager.truncateTo(22);
  assert.equal(pager.getPageState(1), VramPageState.HOT_RESIDENT);
  assert.equal(pager.getPageState(2), VramPageState.EMPTY);
  assert.equal(pager.getTelemetry().refaultCount, 1);

  reg.freeAllBuffers();
});

test('VramKvPager 3-tier Hot F32 -> Warm Q8 -> Cold Q4 cascade attends via WASM SIMD Q4 and refaults cleanly', () => {
  const reg = new VTensorBackendRegistry({ initialWasmPages: 24 });
  const headDim = 64;
  const nHeads = 4;
  const nHeadsKv = 2;
  const kvDim = nHeadsKv * headDim;
  const maxCtx = 160; // 10 pages

  const pager = new VramKvPager(
    {
      numLayers: 1,
      maxContext: maxCtx,
      nHeadsKv,
      headDim,
      hotFrameLimit: 2,
      warmQ8Limit: 2,
      coldQ4Enabled: true,
    },
    (id, usage, cap) => reg.allocBuffer(id, usage, cap),
  );

  const fullK = new Float32Array(maxCtx * kvDim);
  const fullV = new Float32Array(maxCtx * kvDim);
  const tmpK = new Float32Array(kvDim);
  const tmpV = new Float32Array(kvDim);

  // Fill 116 positions (8 pages: page 0 hot sink, page 7 hot active, 2 warm Q8, 4 cold Q4!)
  for (let pos = 0; pos < 116; pos++) {
    for (let d = 0; d < kvDim; d++) {
      tmpK[d] = Math.sin(pos * 0.17 + d * 0.09);
      tmpV[d] = Math.cos(pos * 0.11 - d * 0.13);
    }
    fullK.set(tmpK, pos * kvDim);
    fullV.set(tmpV, pos * kvDim);
    pager.writeKv(false, 0, pos, tmpK);
    pager.writeKv(true, 0, pos, tmpV);
  }

  const tel = pager.getTelemetry();
  assert.equal(tel.coldQ4Enabled, true);
  assert.equal(pager.getPageState(0), VramPageState.HOT_RESIDENT);
  assert.equal(pager.getPageState(1), VramPageState.COLD_Q4);
  assert.equal(tel.hotResidentPages, 2);
  assert.equal(tel.warmQ8Pages, 2);
  assert.equal(tel.coldQ4Pages, 4);
  assert.ok(tel.tierCascadeCycles >= 4);
  assert.ok(tel.compressionRatio > 2.2);

  // Verify 3-tier attention (Hot F32 + Warm Q8 + Cold Q4 via WASM SIMD dot_q4_f32 / axpy_q4_f32)
  const scratch = reg.allocBuffer('test3t.scratch', 'compute', 4096);
  const qH = scratch.allocTensor('q', GgmlType.F32, [nHeads * headDim]);
  const oH = scratch.allocTensor('o', GgmlType.F32, [nHeads * headDim]);
  const qView = scratch.f32View(qH);
  for (let i = 0; i < qView.length; i++) qView[i] = Math.cos(i * 0.19);

  const refOut = new Float32Array(nHeads * headDim);
  verdantFlashAttnExt(qView, fullK, fullV, refOut, nHeads, nHeadsKv, headDim, 116, 0);
  pager.flashAttnExt(
    qView,
    scratch.wasmPtr(qH),
    scratch.f32View(oH),
    scratch.wasmPtr(oH),
    nHeads,
    0,
    116,
    reg.arena,
  );
  const got = scratch.f32View(oH);
  let maxErr = 0;
  for (let i = 0; i < got.length; i++) {
    maxErr = Math.max(maxErr, Math.abs(got[i] - refOut[i]));
  }
  assert.ok(maxErr < 0.08, `3-tier Cold Q4 attention max error ${maxErr}`);
  assert.ok(pager.getTelemetry().coldQ4AttentionReads > 0);

  // Truncating back into a Cold Q4 page (page 1, pos 20) refaults it cleanly back to Hot F32!
  pager.truncateTo(20);
  assert.equal(pager.getPageState(1), VramPageState.HOT_RESIDENT);
  assert.equal(pager.getTelemetry().coldQ4Pages, 0);
  assert.equal(pager.getTelemetry().refaultCount, 1);

  reg.freeAllBuffers();
});

test('VramKvPager 4-tier Hot -> Warm Q8 -> Cold Q4 -> Tier-3 SWAPPED_Q4 preserves full context with bounded WASM RAM', () => {
  const reg = new VTensorBackendRegistry({ initialWasmPages: 24 });
  const headDim = 64;
  const nHeads = 4;
  const nHeadsKv = 2;
  const kvDim = nHeadsKv * headDim;
  const maxCtx = 224; // 14 pages

  // Bound Cold Q4 WASM pool to just 2 slots so remaining pages spill to Tier-3 SWAPPED_Q4!
  const pager = new VramKvPager(
    {
      numLayers: 1,
      maxContext: maxCtx,
      nHeadsKv,
      headDim,
      hotFrameLimit: 2,
      warmQ8Limit: 2,
      coldQ4Enabled: true,
      coldQ4PoolLimit: 2,
    },
    (id, usage, cap) => reg.allocBuffer(id, usage, cap),
  );

  const fullK = new Float32Array(maxCtx * kvDim);
  const fullV = new Float32Array(maxCtx * kvDim);
  const tmpK = new Float32Array(kvDim);
  const tmpV = new Float32Array(kvDim);

  // Fill 180 positions (12 pages: 2 hot + 2 warm + 2 cold Q4 + 6 swapped Q4!)
  for (let pos = 0; pos < 180; pos++) {
    for (let d = 0; d < kvDim; d++) {
      tmpK[d] = Math.sin(pos * 0.13 + d * 0.07);
      tmpV[d] = Math.cos(pos * 0.17 - d * 0.11);
    }
    fullK.set(tmpK, pos * kvDim);
    fullV.set(tmpV, pos * kvDim);
    pager.writeKv(false, 0, pos, tmpK);
    pager.writeKv(true, 0, pos, tmpV);
  }

  const tel = pager.getTelemetry();
  assert.equal(tel.swapQ4Enabled, true);
  assert.equal(pager.getPageState(0), VramPageState.HOT_RESIDENT);
  assert.equal(pager.getPageState(1), VramPageState.SWAPPED_Q4);
  assert.equal(tel.hotResidentPages, 2);
  assert.equal(tel.warmQ8Pages, 2);
  assert.equal(tel.coldQ4Pages, 2);
  assert.equal(tel.swappedQ4Pages, 6);
  assert.ok(tel.swapCycles >= 6);

  // Verify 4-tier streaming attention accuracy against full uncompressed F32 reference
  // Also exercises empty Float32Array(0) + WASM pointer materialization path used by compiled graphs!
  const scratch = reg.allocBuffer('test4t.scratch', 'compute', 4096);
  const qH = scratch.allocTensor('q', GgmlType.F32, [nHeads * headDim]);
  const oH = scratch.allocTensor('o', GgmlType.F32, [nHeads * headDim]);
  const qView = scratch.f32View(qH);
  for (let i = 0; i < qView.length; i++) qView[i] = Math.sin(i * 0.23);

  const refOut = new Float32Array(nHeads * headDim);
  verdantFlashAttnExt(qView, fullK, fullV, refOut, nHeads, nHeadsKv, headDim, 180, 0);
  pager.flashAttnExt(
    new Float32Array(0),
    scratch.wasmPtr(qH),
    new Float32Array(0),
    scratch.wasmPtr(oH),
    nHeads,
    0,
    180,
    reg.arena,
  );
  const got = scratch.f32View(oH);
  let maxErr = 0;
  for (let i = 0; i < got.length; i++) {
    maxErr = Math.max(maxErr, Math.abs(got[i] - refOut[i]));
  }
  assert.ok(maxErr < 0.08, `4-tier SWAPPED_Q4 attention max error ${maxErr}`);

  // Truncating back into a SWAPPED_Q4 page (page 1, pos 20) refaults it cleanly back to Hot F32!
  pager.truncateTo(20);
  assert.equal(pager.getPageState(1), VramPageState.HOT_RESIDENT);
  assert.equal(pager.getTelemetry().swappedQ4Pages, 0);
  assert.equal(pager.getTelemetry().coldQ4Pages, 0);
  assert.equal(pager.getTelemetry().refaultCount, 1);

  reg.freeAllBuffers();
});

