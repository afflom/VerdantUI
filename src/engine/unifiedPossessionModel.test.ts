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
