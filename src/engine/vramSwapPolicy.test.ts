import test from 'node:test';
import assert from 'node:assert/strict';
import {
  compressUnifiedKvBytes,
  createCooperativeYielder,
  decideUorcChallengerAdmission,
  decodeRepeatedBytes,
  decodeSelectorPage,
  decompressUnifiedKvBytes,
  factorRepeatedBytes,
  fitSelectorPage,
  FRAME_BUDGET_US,
  FrameBudgetProfiler,
  GENERATION_SLICE_BUDGET_US,
  shiftedRepeatedBytes,
  swapBreakEven,
  swapCopyCount,
  swapFactorDominance,
  swapMiddle,
  swapServiceBudget,
  swapTwoWithin,
  UnifiedPageCodecTag,
} from './vramSwapPolicy';

test('Lean 4 SwapPolicy Theorem T7 invariants hold across medians, budgets, break-even, and factor dominance', () => {
  // All 6 permutations of (5, 12, 20) must yield median 12
  const perms: Array<[number, number, number]> = [
    [5, 12, 20],
    [5, 20, 12],
    [12, 5, 20],
    [12, 20, 5],
    [20, 5, 12],
    [20, 12, 5],
  ];
  for (const [a, b, c] of perms) {
    assert.equal(swapMiddle(a, b, c), 12);
    assert.equal(swapTwoWithin(a, b, c, 12), true);
    assert.equal(swapTwoWithin(a, b, c, 11), false);
  }

  const okBudget = swapServiceBudget(30, 8_000, 100, 12_000, FRAME_BUDGET_US);
  assert.deepEqual(okBudget, {
    transportWithin: true,
    serviceWithin: true,
    ceilingValid: true,
    totalWithinZram: true,
  });

  const overBudget = swapServiceBudget(500, 17_000, 100, 12_000, FRAME_BUDGET_US);
  assert.equal(overBudget.transportWithin, false);
  assert.equal(overBudget.serviceWithin, false);
  assert.equal(overBudget.totalWithinZram, false);

  assert.equal(swapBreakEven(100, 50, 200, 50), true);
  assert.equal(swapBreakEven(100, 51, 200, 50), false);
  assert.equal(swapFactorDominance(137, 4096, 28), true);
  assert.equal(swapFactorDominance(4097, 4096, 28), false);
  assert.equal(swapCopyCount(7, 3), 3);
  assert.equal(swapCopyCount(2, 9), 2);
  assert.equal(decideUorcChallengerAdmission(137, 4096), true);
  assert.equal(decideUorcChallengerAdmission(4096, 4096), false);
});

test('Exact Repeated (dyadic + shifted) and 2-Symbol Selector codecs achieve bit-exact lossless round-trips', () => {
  // 1. Dyadic exact repetition (4096 B -> 64 B unit)
  const dyadic = new Uint8Array(4096);
  for (let i = 0; i < dyadic.length; i++) dyadic[i] = ((i & 63) * 19 + 7) & 0xff;
  const dyadicFit = factorRepeatedBytes(dyadic);
  assert.ok(dyadicFit !== null);
  assert.equal(dyadicFit.unit.byteLength, 64);
  assert.equal(dyadicFit.halvings, 6);
  assert.deepEqual(Array.from(decodeRepeatedBytes(dyadicFit.unit, dyadic.length)), Array.from(dyadic));

  // 2. Non-dyadic shifted period (period 15 over 300 bytes)
  const shifted = new Uint8Array(300);
  for (let i = 0; i < shifted.length; i++) shifted[i] = ((i % 15) * 23 + 5) & 0xff;
  const shiftedFit = shiftedRepeatedBytes(shifted);
  assert.ok(shiftedFit !== null);
  assert.equal(shiftedFit.unit.byteLength, 15);
  assert.deepEqual(Array.from(decodeRepeatedBytes(shiftedFit.unit, shifted.length)), Array.from(shifted));

  // 3. 2-Symbol shared column-mask selector (4096 B -> 137 B, 29.9× lossless!)
  const selectorPage = new Uint8Array(4096);
  const mask = 0xf0f0_aa55_33cc_0f0fn;
  for (let r = 0; r < 64; r++) {
    const sym0 = (r * 7 + 11) & 0xff;
    const sym1 = (r * 13 + 97) & 0xff;
    for (let c = 0; c < 64; c++) {
      selectorPage[r * 64 + c] = ((mask >> BigInt(c)) & 1n) === 1n ? sym1 : sym0;
    }
  }
  const selFit = fitSelectorPage(selectorPage);
  assert.ok(selFit !== null);
  assert.equal(selFit.encodedBytes, 137);
  const selDecoded = decodeSelectorPage(selFit.columnsMask, selFit.rowSymbols, selFit.rowCount, 64);
  assert.deepEqual(Array.from(selDecoded), Array.from(selectorPage));

  // 4. Unified UKV2 codec dispatcher chooses Selector on 2-symbol page and RAW on incompressible noise
  const packedSel = compressUnifiedKvBytes(selectorPage);
  assert.equal(packedSel.codec, UnifiedPageCodecTag.SELECTOR_2SYM);
  assert.equal(packedSel.savedBytes, 4096 - 137);
  assert.deepEqual(
    Array.from(decompressUnifiedKvBytes(packedSel.payload, 4096)),
    Array.from(selectorPage),
  );

  const randomLike = new Uint8Array(512);
  for (let i = 0; i < randomLike.length; i++) {
    randomLike[i] = ((i * i * 37 + i * 17 + 13) ^ (i >>> 2)) & 0xff;
  }
  const packedRaw = compressUnifiedKvBytes(randomLike);
  assert.equal(packedRaw.codec, UnifiedPageCodecTag.RAW);
  assert.equal(packedRaw.savedBytes, 0);
  assert.deepEqual(
    Array.from(decompressUnifiedKvBytes(packedRaw.payload, randomLike.length)),
    Array.from(randomLike),
  );
});

test('Zero-clamp MessageChannel yielder and FrameBudgetProfiler enforce 16,666 us KPI without per-token timer stalls', async () => {
  const yielder = createCooperativeYielder();
  try {
    assert.equal(yielder.transport, 'message-channel');
    const t0 = performance.now();
    for (let i = 0; i < 10; i++) {
      await yielder.yieldNow();
    }
    const elapsedMs = performance.now() - t0;
    // 10 consecutive MessageChannel turns must complete promptly without stalling
    assert.ok(elapsedMs < 250, `Expected 10 MessageChannel yields < 250 ms, took ${elapsedMs.toFixed(2)} ms`);
  } finally {
    yielder.dispose();
  }

  const profiler = new FrameBudgetProfiler(FRAME_BUDGET_US, 'message-channel');
  profiler.beginSlice(100);
  assert.equal(profiler.recordStepAndShouldYield(900, GENERATION_SLICE_BUDGET_US, 16, 100.9), false);
  assert.equal(profiler.recordStepAndShouldYield(1_100, GENERATION_SLICE_BUDGET_US, 16, 102.0), false);
  assert.equal(profiler.recordStepAndShouldYield(10_500, GENERATION_SLICE_BUDGET_US, 16, 112.5), true);

  const tel = profiler.getTelemetry();
  assert.equal(tel.yieldTransport, 'message-channel');
  assert.equal(tel.totalSteps, 3);
  assert.equal(tel.yieldsExecuted, 1);
  assert.equal(tel.yieldsAvoided, 2);
  assert.equal(tel.medianStepUs, 1_100);
  assert.equal(tel.withinFrameBudget, true);
});
