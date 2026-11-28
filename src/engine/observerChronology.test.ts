import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ObserverChronology,
  classifyUserCadence,
  computeKnowledgeFreshnessQ10,
  computePeriodicPhaseQ10,
  computeTimelinessPressureQ10,
} from './observerChronology';
import { VerdantCognitiveObserver } from './cognitiveObserver';
import type { LoadedModelSummary } from './llamaRuntime';
import { LivingTrainingBasis } from './unifiedModelBasis';
import { MultimodalCoherenceCoordinator } from './multimodalCoordinator';

test('ObserverChronology senses dual-clock wall/monotonic time, periodicity, and object timeliness pressure', () => {
  let wallMs = new Date('2026-11-27T14:30:00Z').getTime();
  let monoMs = 1_000;
  const chron = new ObserverChronology({
    clock: {
      wallNow: () => wallMs,
      monoNow: () => monoMs,
    },
  });

  const t0 = chron.now();
  assert.equal(t0.wallClockMs, wallMs);
  assert.equal(t0.monoMs, 1_000);
  assert.ok(t0.circadianPhaseQ10 >= 0 && t0.circadianPhaseQ10 <= 1024);

  // Register a part with a 2,000 ms wall-clock budget
  chron.beginObject({
    objectId: 'prism:demo',
    modality: 'prism',
    kind: 'prism-part',
    label: 'Demo Prism',
    budgetMs: 2_000,
    periodMs: 500,
  });

  // At 400 ms elapsed: on-time, low pressure
  monoMs += 400;
  wallMs += 400;
  const early = chron.observeObject('prism:demo')!;
  assert.equal(early.state, 'on-time');
  assert.ok(early.timelinessQ10 > 768);
  assert.equal(chron.getRemainingBudgetMs('prism:demo'), 1_600);

  // At 1,600 ms elapsed (80% budget): tight
  monoMs += 1_200;
  wallMs += 1_200;
  const tight = chron.observeObject('prism:demo')!;
  assert.equal(tight.state, 'tight');
  assert.ok(tight.pressureQ10 >= 512);

  // At 2,400 ms elapsed (past deadline): overdue
  monoMs += 800;
  wallMs += 800;
  const overdue = chron.observeObject('prism:demo')!;
  assert.equal(overdue.state, 'overdue');
  assert.equal(chron.getRemainingBudgetMs('prism:demo'), 0);
  assert.equal(chron.snapshot().overdueCount, 1);

  // Completing the object preserves recorded duration and clears active overdue count
  const done = chron.completeObject('prism:demo')!;
  assert.equal(done.elapsedMs, 2_400);
  assert.equal(chron.snapshot().overdueCount, 0);
});

test('ObserverChronology senses periodic rhythm, knowledge freshness decay, and live throughput wall-clock projections', () => {
  let wallMs = 1_700_000_000_000;
  let monoMs = 5_000;
  const chron = new ObserverChronology({
    clock: {
      wallNow: () => wallMs,
      monoNow: () => monoMs,
    },
  });

  // Sense periodic 60 Hz frame cadence
  chron.pulseObject({
    objectId: 'modal:frame',
    modality: 'video',
    kind: 'modal-frame',
    label: 'Render Frame',
    periodMs: 17,
  });
  monoMs += 16;
  wallMs += 16;
  const p2 = chron.pulseObject({
    objectId: 'modal:frame',
    modality: 'video',
    kind: 'modal-frame',
    label: 'Render Frame',
    periodMs: 17,
  });
  assert.equal(p2.observedPeriodMs, 16);

  // Sense rapid user turn periodicity -> tightens adaptive stream timeout
  chron.observeUserTurn();
  wallMs += 5_000;
  monoMs += 5_000;
  const rapid = chron.observeUserTurn();
  assert.equal(rapid.cadence, 'rapid');
  assert.ok(chron.deriveAdaptiveStreamTimeoutMs('balanced') < 2_600);

  // Calibrate real network & inference throughput -> accurate wall-clock TTFT projection
  chron.recordNetworkThroughput(10 * 1024 * 1024); // 10 MB/s
  chron.recordInferenceThroughput({ prefillTokSec: 50, decodeTokSec: 20 });
  const projectedMs = chron.projectTurnTtftMs({
    missingTransferBytes: 20 * 1024 * 1024, // ~2000 ms
    uncachedPrefillTokens: 100, // ~2000 ms
    thinkingTokensEstimate: 0,
  });
  assert.ok(projectedMs > 2_000 && projectedMs < 7_000);

  // Exact-integer freshness decay & phase helpers
  assert.equal(computeKnowledgeFreshnessQ10(0, 60_000), 1024);
  assert.equal(computeKnowledgeFreshnessQ10(60_000, 60_000), 512);
  assert.ok(computeKnowledgeFreshnessQ10(200_000, 60_000) < 256);
  assert.equal(computePeriodicPhaseQ10(250, 1000), 256);
  assert.ok(computeTimelinessPressureQ10(100, 1000) < computeTimelinessPressureQ10(900, 1000));
  assert.equal(classifyUserCadence(4_000), 'rapid');
  assert.equal(classifyUserCadence(30_000), 'interactive');
  assert.equal(classifyUserCadence(120_000), 'deliberative');
});

test('ObserverChronology anchors opaque schedule constraints to the current local wall clock', () => {
  const fixedDate = new Date(2026, 10, 27, 9, 30, 0); // Friday 09:30 local (day=5, minute=570)
  const chron = new ObserverChronology({
    clock: {
      wallNow: () => fixedDate.getTime(),
      monoNow: () => 0,
    },
  });

  const active = chron.getActiveScheduleWindows([
    {
      kind: 'availability-block',
      targetCompartmentId: 'work',
      opaqueSourceKappa: 'k1',
      startMinuteOfDay: 540, // 09:00
      endMinuteOfDay: 600, // 10:00
      daysOfWeek: [5],
      weightQ10: 960,
      detail: null,
    },
    {
      kind: 'availability-block',
      targetCompartmentId: 'work',
      opaqueSourceKappa: 'k2',
      startMinuteOfDay: 720, // 12:00
      endMinuteOfDay: 780, // 13:00
      daysOfWeek: [5],
      weightQ10: 960,
      detail: null,
    },
  ]);
  assert.equal(active.length, 1);
  assert.equal(active[0]!.opaqueSourceKappa, 'k1');
});

test('Cognitive Observer, LivingTrainingBasis, and MultimodalCoherenceCoordinator share unified chronological coherence', async () => {
  // 1. Cognitive Observer emits temporalAnchor + temporalCoherence on route selection
  const observer = new VerdantCognitiveObserver();
  const { certificate, snapshot } = observer.selectRoute({
    threadId: 'chrono-thread',
    messages: [],
    latestUserText: 'Explain how tidal resonance works briefly.',
    hasAttachment: false,
    activeModel: {
      id: 'qwen3-4b-q4_k_m',
      displayName: 'Qwen3-4B',
      backingKind: 'opfs-demand',
      maxContextWindow: 4096,
      hparams: {
        name: 'Qwen3-4B',
        architecture: 'qwen3',
      },
    } as unknown as LoadedModelSummary,
    cachedModelIds: ['qwen3-4b-q4_k_m'],
    isToolContinuationStep: false,
    hasImportedHarness: false,
  });
  assert.ok(certificate.temporalAnchor, 'Route certificate must carry a temporalAnchor');
  assert.ok(certificate.temporalAnchor.wallClockMs > 0);
  assert.ok(certificate.temporalAnchor.projectedTtftMs >= 0);
  assert.ok(certificate.temporalAnchor.streamTimeoutMs >= 1_200);
  assert.ok(snapshot.temporalCoherence, 'Observer snapshot must include temporalCoherence');
  assert.ok(snapshot.temporalCoherence.coherenceQ10 > 0);

  // 2. LivingTrainingBasis ranks fresh knowledge strictly ahead of stale expired facts
  const basis = new LivingTrainingBasis();
  const nowMs = 1_800_000_000_000;
  basis.recordStreamedKnowledge({
    kappa: 'know:stale-solar',
    streamId: 'stream-news',
    catalogId: 'science',
    provider: 'rss',
    title: 'Solar Flare Activity Archive',
    sourceUrl: 'https://example.org/old',
    summary: 'Historical solar flare geomagnetic storm telemetry report.',
    categories: ['solar', 'space'],
    fetchedAt: nowMs - 48 * 3600 * 1000, // 48 hours old (stale past 6h TTL)
    ttlMs: 6 * 3600 * 1000,
  });
  basis.recordStreamedKnowledge({
    kappa: 'know:fresh-solar',
    streamId: 'stream-news',
    catalogId: 'science',
    provider: 'rss',
    title: 'Solar Flare Activity Live',
    sourceUrl: 'https://example.org/live',
    summary: 'Fresh solar flare geomagnetic storm telemetry report.',
    categories: ['solar', 'space'],
    fetchedAt: nowMs - 10 * 60 * 1000, // 10 minutes old (fresh)
    ttlMs: 6 * 3600 * 1000,
  });
  const recalled = basis.recallKnowledge({ text: 'solar flare telemetry', limit: 2, nowMs });
  assert.equal(recalled.length, 2);
  assert.equal(recalled[0]!.kappa, 'know:fresh-solar');
  assert.equal(recalled[1]!.kappa, 'know:stale-solar');

  // 3. MultimodalCoherenceCoordinator pulses Vision, Audio, and Frame clocks onto ObserverChronology
  const multimodal = new MultimodalCoherenceCoordinator(128 * 1024 * 1024, observer.chronology);
  multimodal.recordFrame(12_000);
  await multimodal.getOrEncodeVisual('vis:kappa-1', async () => ({
    width: 64,
    height: 64,
    tokenCount: 16,
    embeddingDim: 8,
    embeddings: [],
    encodeMs: 12,
  }));
  await multimodal.getOrTranscribeAudio('aud:kappa-1', async () => ({
    durationSec: 2,
    text: 'hello world',
    transcribeMs: 18,
  }));
  const mmTel = multimodal.getTelemetry();
  assert.ok(mmTel.temporalCoherenceQ10 >= 512);
  const snapAfter = observer.chronology.snapshot();
  assert.ok(snapAfter.modalityCoherenceQ10.vision !== undefined);
  assert.ok(snapAfter.modalityCoherenceQ10.audio !== undefined);
  assert.ok(snapAfter.modalityCoherenceQ10.video !== undefined);
});

test('ObserverChronology.deriveTurnBudgetMs keeps SLA floor before calibration and scales smoothly on slow CPU hardware', () => {
  const chrono = new ObserverChronology({
    initialThroughput: { prefillTokensPerSec: 10, decodeTokensPerSec: 3, turnSamples: 0 },
  });
  // Before any completed turns, preserve the interactive SLA floor
  assert.equal(chrono.deriveTurnBudgetMs(12_000, 2_500), 2_500);
  // Once calibrated, grade on-pace hardware execution fairly while still capping at 60s
  chrono.recordInferenceThroughput({ prefillTokSec: 10, decodeTokSec: 3 });
  const calibrated = chrono.deriveTurnBudgetMs(10_000, 2_500);
  assert.equal(calibrated, 16_000);
  assert.equal(chrono.deriveTurnBudgetMs(500, 2_500), 2_500);
  assert.equal(chrono.deriveTurnBudgetMs(100_000, 2_500), 60_000);

  // Idempotent completeObject(): calling completeObject() again later must NOT overwrite original duration
  let mono = 1_000;
  const clk = { wallNow: () => 1_700_000_000_000 + mono, monoNow: () => mono };
  const c2 = new ObserverChronology({ clock: clk });
  c2.beginObject({ objectId: 'app:init', modality: 'system', kind: 'app-init', label: 'Boot', budgetMs: 25_000 });
  mono = 4_000;
  const first = c2.completeObject('app:init')!;
  assert.equal(first.elapsedMs, 3_000);
  mono = 180_000;
  const second = c2.completeObject('app:init')!;
  assert.equal(second.elapsedMs, 3_000, 'Second completeObject call must preserve initial completion duration');
  assert.equal(c2.snapshot().modalityCoherenceQ10.system, first.timelinessQ10);
});
