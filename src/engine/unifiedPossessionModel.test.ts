import test from 'node:test';
import assert from 'node:assert/strict';
import {
  evaluateContextBoundary,
  formatOpaqueConstraintsForContext,
  ObserverPossessionStore,
  projectOpaqueConstraints,
  verifyZeroCrossContextLeak,
} from './unifiedPossessionModel';
import { VerdantCognitiveObserver } from './cognitiveObserver';
import { buildHoloviewSandboxDocument } from './prismSandboxHtml';
import { PrismHostPool } from './prismHostPool';
import { ObserverStreamBus } from './observerStreamBus';
import { PrismStore } from './prismStudio';
import type { LlamaVTensorWorkerClient } from './workerClient';

test('ObserverPossessionStore establishes Observer ⊇ User ⊇ {Threads, Prisms, Compartments} hierarchy', () => {
  const store = new ObserverPossessionStore('personal');
  store.bindThreadToCompartment('thread-work-1', 'work');
  store.bindPrismToCompartment('prism-habits', 'personal', ['Healthy Habits', 'evening cardio']);

  const user = store.getUser();
  assert.equal(user.userId, 'owner');
  assert.ok(user.ownedThreadIds.includes('thread-work-1'));
  assert.ok(user.ownedPrismIds.includes('prism-habits'));
  assert.ok(user.compartments.work.ownedThreadIds.includes('thread-work-1'));
  assert.ok(user.compartments.personal.attachedPrismIds.includes('prism-habits'));
});

test('Healthy Habits (personal) projects zero-detail opaque busy blocks to Work with ZERO personal leakage', () => {
  const store = new ObserverPossessionStore('work');
  store.bindPrismToCompartment('prism-healthy-habits', 'personal', [
    'Healthy Habits',
    'evening yoga',
    'cardio workout',
  ]);
  store.upsertScheduleConstraint('personal', {
    id: 'habit-daily-exercise',
    privateLabel: 'Evening cardio workout',
    sourcePrismId: 'prism-healthy-habits',
    startMinuteOfDay: 17 * 60, // 17:00
    endMinuteOfDay: 18 * 60,   // 18:00
    daysOfWeek: [0, 1, 2, 3, 4, 5, 6],
    weightQ10: 960,
    shareOpaqueAvailability: true,
  });

  // Work space receives the exact protected time window as an opaque constraint
  const projected = projectOpaqueConstraints(store.getUser(), 'work');
  assert.equal(projected.length, 1);
  assert.equal(projected[0].startMinuteOfDay, 1020);
  assert.equal(projected[0].endMinuteOfDay, 1080);
  assert.equal(projected[0].detail, null);

  // Zero-leak verification: neither JSON nor prompt text contains ANY personal detail
  const serialized = JSON.stringify(projected) + '\n' + formatOpaqueConstraintsForContext(projected);
  const lower = serialized.toLowerCase();
  for (const forbidden of ['healthy habits', 'cardio', 'workout', 'yoga', 'personal', 'prism-healthy-habits']) {
    assert.equal(lower.includes(forbidden), false, `Leaked personal term "${forbidden}" into work projection`);
  }
  assert.ok(serialized.includes('17:00–18:00'));

  // Taint detector catches accidental personal detail in an outbound work calendar invite!
  const leaks = verifyZeroCrossContextLeak(
    'Sorry, I have my evening cardio workout at 5pm.',
    [store.getUser().compartments.personal],
  );
  assert.ok(leaks.length > 0);
});

test('evaluateContextBoundary enforces kill switch, regulated hashonly/deny, zero cross-space raw transfer, and seen-bytes approval', () => {
  const store = new ObserverPossessionStore('work');
  store.bindPrismToCompartment('prism-habits', 'personal', ['Healthy Habits', 'therapy session']);

  // 1. Kill switch denies everything
  const haltedRes = evaluateContextBoundary({
    sourceCompartmentId: 'work',
    destination: 'calendar',
    label: 'public',
    payloadText: 'Team standup at 10am',
    ownerApproved: true,
    bytesMatchSeen: true,
    halted: true,
  });
  assert.equal(haltedRes.verdict, 'deny');

  // 2. Regulated credential/PHI -> hashonly locally, deny outbound
  const localCred = evaluateContextBoundary({
    sourceCompartmentId: 'personal',
    destination: 'browser-local',
    label: 'confidential',
    payloadText: 'token ghp_1234567890abcdefghijklmno',
    halted: false,
  });
  assert.equal(localCred.verdict, 'hashonly');
  assert.equal(localCred.detector, 'credential');

  const outboundCred = evaluateContextBoundary({
    sourceCompartmentId: 'personal',
    destination: 'mail',
    label: 'confidential',
    payloadText: 'token ghp_1234567890abcdefghijklmno',
    ownerApproved: true,
    bytesMatchSeen: true,
    halted: false,
  });
  assert.equal(outboundCred.verdict, 'deny');

  // 3. Raw cross-compartment transfer is denied unless opaque projection
  const rawCross = evaluateContextBoundary({
    sourceCompartmentId: 'personal',
    targetCompartmentId: 'work',
    destination: 'foreign-compartment',
    label: 'internal',
    payloadText: 'Private note',
    isOpaqueProjection: false,
    halted: false,
  });
  assert.equal(rawCross.verdict, 'deny');

  const opaqueCross = evaluateContextBoundary({
    sourceCompartmentId: 'personal',
    targetCompartmentId: 'work',
    destination: 'foreign-compartment',
    label: 'internal',
    payloadText: '17:00-18:00 busy',
    isOpaqueProjection: true,
    halted: false,
  });
  assert.equal(opaqueCross.verdict, 'allow');

  // 4. Outbound work email leaking personal compartment secret is DENIED even with ownerApproved
  const taintedOutbound = evaluateContextBoundary({
    sourceCompartmentId: 'work',
    destination: 'mail',
    label: 'internal',
    payloadText: 'Cannot meet at 5pm due to Healthy Habits.',
    ownerApproved: true,
    bytesMatchSeen: true,
    halted: false,
    foreignCompartments: [store.getUser().compartments.personal],
  });
  assert.equal(taintedOutbound.verdict, 'deny');
  assert.ok(taintedOutbound.leakedForeignTerms.length > 0);

  // 5. Clean outbound requires owner approval on exact seen bytes
  const unapproved = evaluateContextBoundary({
    sourceCompartmentId: 'work',
    destination: 'calendar',
    label: 'internal',
    payloadText: 'Let us schedule the design review for 3:00 PM.',
    ownerApproved: false,
    bytesMatchSeen: true,
    halted: false,
    foreignCompartments: [store.getUser().compartments.personal],
  });
  assert.equal(unapproved.verdict, 'ask');

  const approved = evaluateContextBoundary({
    sourceCompartmentId: 'work',
    destination: 'calendar',
    label: 'internal',
    payloadText: 'Let us schedule the design review for 3:00 PM.',
    ownerApproved: true,
    bytesMatchSeen: true,
    halted: false,
    foreignCompartments: [store.getUser().compartments.personal],
  });
  assert.equal(approved.verdict, 'allow');
});

test('Published Prisms act as Observer organs: sensory ingestion (with PHI hashonly) and gated action impulses', () => {
  const observer = new VerdantCognitiveObserver();
  observer.possession.registerPublishedPrismOrgan({
    prismId: 'prism-habits',
    name: 'Healthy Habits',
    slug: 'healthy-habits',
    digest: 'abc123',
    mediaType: 'tool-organ',
    compartmentId: 'personal',
    cognitiveWeightsQ10: [256, 256, 128, 640, 128, 384, 256, 768],
    sensoryChannels: ['browser', 'schedule'],
    impulseChannels: ['notify', 'schedule-constraint'],
    shareOpaqueConstraints: true,
  });

  // PHI sensory frame is stored strictly as hashonly pointer (raw text never persisted)
  const sens = observer.possession.ingestSensoryFrame({
    id: 'sf-1',
    prismId: 'prism-habits',
    compartmentId: 'personal',
    channel: 'schedule',
    sensitivity: 'confidential',
    payloadText: 'blood pressure check 120/80 at 5pm',
    receivedAt: Date.now(),
  });
  assert.equal(sens.verdict, 'hashonly');
  const frames = observer.possession.getRecentSensoryFrames();
  assert.equal(frames.length, 1);
  assert.ok(frames[0].payloadText.startsWith('sha256:'));
  assert.equal(frames[0].payloadText.includes('blood pressure'), false);

  // Action impulse inside same compartment succeeds; raw cross-compartment impulse is refused
  const okImp = observer.possession.dispatchActionImpulse({
    id: 'imp-1',
    targetPrismId: 'prism-habits',
    sourceCompartmentId: 'personal',
    targetCompartmentId: 'personal',
    kind: 'notify',
    payload: { title: 'Stretch reminder' },
    opaqueOnly: false,
    ownerApproved: true,
    emittedAt: Date.now(),
  });
  assert.equal(okImp.verdict, 'allow');

  const blockedImp = observer.possession.dispatchActionImpulse({
    id: 'imp-2',
    targetPrismId: 'prism-habits',
    sourceCompartmentId: 'work',
    targetCompartmentId: 'personal',
    kind: 'stream-emit',
    payload: 'raw work payload',
    opaqueOnly: false,
    ownerApproved: true,
    emittedAt: Date.now(),
  });
  assert.equal(blockedImp.verdict, 'deny');
});

test('buildHoloviewSandboxDocument exposes window.verdant alongside prism.invoke and PrismHostPool mediates emits & space-isolated KV', async () => {
  const html = buildHoloviewSandboxDocument({
    prismId: 'prism-habits',
    channelId: 'ch_test',
    digest: 'abc123',
    indexHtml: '<div id="organ-root"></div>',
    prismCss: '',
    prismUiJs: '',
    appJs: '',
    darkMode: true,
  });
  assert.ok(html.includes('globalThis.prism = Object.freeze'));
  assert.ok(html.includes('window.verdant = Object.freeze'));
  assert.ok(html.includes("post('emit'"));

  const bus = new ObserverStreamBus();
  const store = new PrismStore();
  const observer = new VerdantCognitiveObserver();
  const proj = await store.createDeclarative('undefined');
  await store.applyStudioOp(proj.id, {
    op: 'set-concept-profile',
    compartmentId: 'personal',
  });

  const fakeWorker = { cognitiveObserver: observer } as unknown as LlamaVTensorWorkerClient;
  const pool = new PrismHostPool(bus, store, observer.grants, fakeWorker);
  const grants = { observer: true, storage: true, browser: true, network: false };

  // 1. Live `constraint` emit populates zero-detail opaque window for `work`
  const emitRes = pool.handlePrismEmit(proj.id, grants, 'constraint', {
    opaqueOnly: true,
    windows: [{ startMinuteOfDay: 1020, endMinuteOfDay: 1080, daysOfWeek: [0, 1, 2, 3, 4, 5, 6] }],
  });
  assert.equal(emitRes.accepted, true);
  assert.equal(observer.possession.getProjectedConstraintsFor('work').length, 1);

  // 2. Toggling window off (empty windows array) prunes the constraint cleanly
  pool.handlePrismEmit(proj.id, grants, 'constraint', { opaqueOnly: true, windows: [] });
  assert.equal(observer.possession.getProjectedConstraintsFor('work').length, 0);

  // 3. Kill switch halts constraint & impulse emits immediately
  observer.possession.setHalted(true);
  const haltedEmit = pool.handlePrismEmit(proj.id, grants, 'constraint', {
    opaqueOnly: true,
    windows: [{ startMinuteOfDay: 1020, endMinuteOfDay: 1080 }],
  });
  assert.equal(haltedEmit.accepted, false);
  observer.possession.setHalted(false);

  // 4. Compartment-isolated KV storage: `personal` vs `work` never bleed
  await pool.setKv(proj.id, grants, 'notes', 'personal-only-note');
  assert.equal(await pool.getKv(proj.id, grants, 'notes'), 'personal-only-note');
  await store.applyStudioOp(proj.id, { op: 'set-concept-profile', compartmentId: 'work' });
  assert.equal(await pool.getKv(proj.id, grants, 'notes'), null);

  pool.dispose();
});

test('Unified model enforces Space-scoped Published Prism visibility, organ pruning, thread ownership, and Space-isolated notes', async () => {
  const observer = new VerdantCognitiveObserver();
  const store = new PrismStore();
  const personalProj = await store.createDeclarative('undefined');
  await store.applyStudioOp(personalProj.id, { op: 'set-metadata', name: 'Healthy Habits' });
  await store.applyStudioOp(personalProj.id, { op: 'set-concept-profile', compartmentId: 'personal' });
  await store.publish(personalProj.id);

  const workProj = await store.createDeclarative('undefined');
  await store.applyStudioOp(workProj.id, { op: 'set-metadata', name: 'Sprint Tracker' });
  await store.applyStudioOp(workProj.id, { op: 'set-concept-profile', compartmentId: 'work' });
  await store.publish(workProj.id);

  observer.syncPublishedPrisms(store.list());
  const allPublished = store.listPublishedSummaries();

  // In Work space, Personal Healthy Habits is NOT exposed as a tool / mention
  const workVisible = observer.getVisiblePublishedPrismsForSpace(allPublished, 'work');
  assert.equal(workVisible.some((p) => p.name === 'Sprint Tracker'), true);
  assert.equal(workVisible.some((p) => p.name === 'Healthy Habits'), false);

  // In Personal space, Healthy Habits IS visible and Sprint Tracker is isolated to Work
  const personalVisible = observer.getVisiblePublishedPrismsForSpace(allPublished, 'personal');
  assert.equal(personalVisible.some((p) => p.name === 'Healthy Habits'), true);
  assert.equal(personalVisible.some((p) => p.name === 'Sprint Tracker'), false);

  // Unpublishing removes the organ cleanly from possession
  await store.unpublish(personalProj.id);
  observer.syncPublishedPrisms(store.list());
  assert.equal(observer.possession.listPublishedOrgans('personal').length, 0);

  // selectRoute binds active thread ownership to active Space (`Observer ⊇ User ⊇ Thread`)
  observer.possession.setActiveCompartment('work');
  observer.selectRoute({
    threadId: 'thread-work-42',
    messages: [],
    latestUserText: 'Calculate 7 * 8',
    hasAttachment: false,
    activeModel: null,
    cachedModelIds: [],
    isToolContinuationStep: false,
    hasImportedHarness: false,
  });
  assert.ok(observer.possession.getUser().ownedThreadIds.includes('thread-work-42'));
  assert.ok(observer.possession.getUser().compartments.work.ownedThreadIds.includes('thread-work-42'));
});
