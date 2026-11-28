import assert from 'node:assert/strict';
import test from 'node:test';
import {
  canonicalizePartCommandMonoid,
  stepNeuromorphicSalienceRing,
  verdantGatedDeltaStepF32,
} from './verdantKernels';
import {
  encodeSemanticVector32,
  UnifiedExpertRegistry,
} from './unifiedModelBasis';
import { VerdantRootPrismRuntime } from './verdantRootPrism';
import { VERDANT_HOME_PRISM_ID } from './observerUiSynthesizer';
import type { LlamaVTensorWorkerClient } from './workerClient';
import { VerdantCognitiveObserver } from './cognitiveObserver';
import { shouldCheckpointAtKvPageBoundary } from './vramSwapPolicy';

function createStubWorkerClient(): {
  client: LlamaVTensorWorkerClient;
  focusHistory: string[];
} {
  const focusHistory: string[] = [];
  const observer = new VerdantCognitiveObserver();
  const stub = {
    cognitiveObserver: observer,
    setFocusedPart(partId: string) {
      focusHistory.push(partId);
    },
    getState() {
      return {
        status: 'ready',
        model: null,
        modelStreamEnabled: false,
        sampling: { temperature: 0.7, topP: 0.9, topK: 40, repeatPenalty: 1.1, maxTokens: 256, fullContextOutput: false },
        promptMode: 'auto',
        precisionMode: 'q8-vnni',
        activePartTaskIds: [],
      };
    },
    async synthesizeImageLayer() {
      return { neuralAvailable: false, width: 0, height: 0, rgba: null };
    },
    async runWithDeterministicPrecision<T>(fn: () => Promise<T>): Promise<T> {
      return fn();
    },
    dispose() {},
  } as unknown as LlamaVTensorWorkerClient;
  return { client: stub, focusHistory };
}

test('verdantGatedDeltaStepF32, kernel10 salience, and runtime_monoid fusion match verdant-engine laws', () => {
  // 1. Gated Delta Rule (`delta_rule.rs` parity): residual decreases strictly on repeated association
  const d = 8;
  const state = new Float32Array(d * d);
  const scores = new Float32Array(d);
  const scratch = new Float32Array(d * d + 2 * d);
  const key = new Float32Array([1, 0, 0, 0, 0, 0, 0, 0]);
  const value = new Float32Array([0.5, -0.5, 0.25, 0.75, 0, 0, 0, 0]);
  const decay = new Float32Array([-0.02]);

  const r1 = verdantGatedDeltaStepF32({ query: key, key, value, logDecay: decay, gain: 0.6 }, state, scores, scratch);
  const r2 = verdantGatedDeltaStepF32({ query: key, key, value, logDecay: decay, gain: 0.6 }, state, scores, scratch);
  const r3 = verdantGatedDeltaStepF32({ query: key, key, value, logDecay: decay, gain: 0.6 }, state, scores, scratch);
  assert.ok(r2.residualNorm < r1.residualNorm, 'Gated delta residual must drop on step 2');
  assert.ok(r3.residualNorm < r2.residualNorm, 'Gated delta residual must drop on step 3');
  assert.ok(scores[0]! > 0 && scores[1]! < 0, 'Query readout must preserve signed associative recall');

  // 2. Neuromorphic LIF/SSM salience (`kernel10`): attended slot wins while background slots retain baseline > 0
  const sal = stepNeuromorphicSalienceRing(
    [100, 100, 100, 100, 100, 100, 100, 100],
    [300, 500, 200, 100, 500, 0, 200, 16],
    4,
    2,
  );
  assert.equal(sal.attendedSlot, 2);
  assert.ok(sal.entitySalienceMilli[2]! > sal.entitySalienceMilli[0]!);
  assert.ok(sal.entitySalienceMilli[0]! >= 128, 'Background slots must retain nonzero baseline salience');
  assert.ok(sal.foveatedLodRank >= 0 && sal.foveatedLodRank <= 3);

  // 3. Command Monoid (`runtime_monoid.rs`): cancels self-inverse toggles, fuses deltas, commutes independent slots
  const report = canonicalizePartCommandMonoid([
    { id: 1, targetSlot: 'prism:a', kind: { type: 'toggle', key: 'details' } },
    { id: 2, targetSlot: 'prism:a', kind: { type: 'toggle', key: 'details' } },
    { id: 3, targetSlot: 'prism:a', kind: { type: 'param-delta', key: 'bpm', delta: 4 } },
    { id: 4, targetSlot: 'prism:a', kind: { type: 'param-delta', key: 'bpm', delta: 6 } },
    { id: 5, targetSlot: 'prism:b', kind: { type: 'param-delta', key: 'density', delta: 10 } },
  ]);
  assert.equal(report.involutionCancellations, 1);
  assert.equal(report.additiveMerges, 1);
  assert.equal(report.fusedCount, 2);
  assert.equal(report.parallelLayers.length, 1, 'Independent target slots must schedule in parallel layer 0');
  assert.equal((report.fusedOps[0]!.kind as { delta: number }).delta, 10);
});

test('VerdantRootPrismRuntime shifts focus without interrupting sibling parts and caches Observer layout frames', async () => {
  const { client, focusHistory } = createStubWorkerClient();
  const root = new VerdantRootPrismRuntime({ workerClient: client });

  // 2-tier Observer Frame cache returns the exact same object reference on steady-state reads
  const frame1 = root.getObserverFrame({
    activeView: { kind: 'thread' },
    focusedPrism: null,
    prisms: [],
    surfaceMode: 'preview',
    viewportWidthPx: 1280,
    viewportHeightPx: 800,
    darkMode: false,
    userTyping: false,
  });
  const frame2 = root.getObserverFrame({
    activeView: { kind: 'thread' },
    focusedPrism: null,
    prisms: [],
    surfaceMode: 'preview',
    viewportWidthPx: 1280,
    viewportHeightPx: 800,
    darkMode: false,
    userTyping: false,
  });
  assert.equal(frame1, frame2, 'Unchanged viewport/focus must hit 2-tier Observer Frame cache');

  // Create two separate Prisms and run background reconciliation on Prism A while focusing Prism B and Home
  const bridge = await root.getStudioChatBridge();
  const dogPic = await bridge.createArtifactFromPrompt({
    prompt: 'Create a sunlit portrait picture of a golden dog',
    forcedMediaType: 'picture',
  });
  const oceanAudio = await bridge.createArtifactFromPrompt({
    prompt: 'Compose a calm ambient ocean soundtrack',
    forcedMediaType: 'audio',
  });

  // Rapidly navigate across parts while background reconciliations are active -> ZERO cross-part aborts!
  root.setFocusPart(dogPic.prismId);
  root.setFocusPart(oceanAudio.prismId);
  root.setFocusPart(VERDANT_HOME_PRISM_ID);
  assert.ok(focusHistory.includes(dogPic.prismId));
  assert.ok(focusHistory.includes(oceanAudio.prismId));

  // Explicitly enqueue both and await completion without either preempting the other
  const revDog = root.expertRegistry.semantics.getConceptForPrism(dogPic.prismId)!;
  const revAudio = root.expertRegistry.semantics.getConceptForPrism(oceanAudio.prismId)!;
  assert.ok(revDog, 'Dog picture must be anchored in UnifiedSemanticAssociator');
  assert.ok(revAudio, 'Ocean audio must be anchored in UnifiedSemanticAssociator');

  await Promise.all([
    root.enqueueProgressiveReconciliation({
      prismId: dogPic.prismId,
      prompt: 'golden dog',
      initialRevision: revDog,
      startRank: 1,
      maxRank: 3,
    }),
    root.enqueueProgressiveReconciliation({
      prismId: oceanAudio.prismId,
      prompt: 'calm ocean soundtrack',
      initialRevision: revAudio,
      startRank: 1,
      maxRank: 3,
    }),
  ]);

  const dogTask = root.getPartTask(dogPic.prismId);
  const audioTask = root.getPartTask(oceanAudio.prismId);
  assert.equal(dogTask.lodRank, 3, 'Dog Prism must converge to Lod3LosslessSource');
  assert.equal(audioTask.lodRank, 3, 'Audio Prism must converge to Lod3LosslessSource');
  assert.equal(dogTask.busy, false);
  assert.equal(audioTask.busy, false);

  // Explicit cancellation of one part must not throw or corrupt another
  root.cancelPartTask(dogPic.prismId);
  assert.equal(root.getPartTask(oceanAudio.prismId).lodRank, 3);
  root.dispose();
});

test('UnifiedSemanticAssociator binds cross-modal concepts with parent-linked kappa revisions and residual checks', () => {
  const registry = new UnifiedExpertRegistry();
  const w1 = registry.semantics.writeSemanticObservation('dog', {
    canonicalSubject: 'dog',
    contextualBrief: 'Warm golden retriever sitting in sunlit grass.',
    attributes: ['golden fur', 'alert ears'],
    lighting: 'daylight',
    palette: ['#1a2e1a', '#d99b43', '#f4d38c', '#f8fafc'],
    linkedPrismId: 'prism-dog-pic',
  });
  assert.equal(w1.revision.version, 1);
  assert.equal(w1.revision.parentKappa, null);
  assert.ok(w1.revision.kappa.length >= 8);

  const w2 = registry.semantics.writeSemanticObservation('dog', {
    canonicalSubject: 'dog',
    contextualBrief: 'Warm golden retriever with sharper muzzle detail in sunlit grass.',
    attributes: ['expressive eyes'],
    linkedPrismId: 'prism-dog-pic',
  });
  assert.equal(w2.revision.version, 2);
  assert.equal(w2.revision.parentKappa, w1.revision.kappa, 'v2 must chain parentKappa to v1');
  assert.ok(w2.revision.attributes.includes('golden fur') && w2.revision.attributes.includes('expressive eyes'));

  // Adjacent modality recall ("make a video of that dog") resolves the exact same concept!
  const recalled = registry.semantics.recallAssociatedSemantics('make a video of that dog', 'prism-dog-pic');
  assert.ok(recalled);
  assert.equal(recalled.conceptId, w2.revision.conceptId);
  assert.equal(recalled.version, 2);

  // Semantic residual check
  const matchVec = encodeSemanticVector32(['dog', 'golden fur', 'alert ears', 'expressive eyes']);
  const res = registry.semantics.computeSemanticResidual(w2.revision.conceptId, matchVec);
  assert.ok(Number.isFinite(res.residualNorm) && res.cosineSimilarity > 0.2);
});

test('Observer Tropical layout geometry cache executes 0 solves across focusedPrism draft.digest LOD ticks', async () => {
  const { client } = createStubWorkerClient();
  const root = new VerdantRootPrismRuntime({ workerClient: client });
  const bridge = await root.getStudioChatBridge();
  const created = await bridge.createArtifactFromPrompt({
    prompt: 'Create a picture of a golden dog',
    forcedMediaType: 'picture',
  });

  const proj0 = root.prismStore.get(created.prismId)!;
  const f0 = root.getObserverFrame({
    activeView: { kind: 'prism', prismId: created.prismId, mode: 'studio' },
    focusedPrism: proj0,
    prisms: root.prismStore.list(),
    surfaceMode: 'preview',
    viewportWidthPx: 1440,
    viewportHeightPx: 900,
    darkMode: false,
    userTyping: false,
  });
  const solvesAfterFirst = root.getLayoutSolveCount();
  assert.equal(solvesAfterFirst, 1);

  // Mutate focusedPrism draft digest (simulating Lod0 -> Lod1 -> Lod2 -> Lod3 reconciliation commits)
  await root.prismStore.applyStudioOp(
    created.prismId,
    { op: 'set-metadata', description: 'Refined Lod2 golden dog' },
    { pushRevision: false },
  );
  const proj1 = root.prismStore.get(created.prismId)!;
  assert.notEqual(proj1.draft.digest, proj0.draft.digest);

  const f1 = root.getObserverFrame({
    activeView: { kind: 'prism', prismId: created.prismId, mode: 'studio' },
    focusedPrism: proj1,
    prisms: root.prismStore.list(),
    surfaceMode: 'preview',
    viewportWidthPx: 1440,
    viewportHeightPx: 900,
    darkMode: true,
    userTyping: true,
  });
  assert.equal(
    root.getLayoutSolveCount(),
    solvesAfterFirst,
    'Digest/theme/typing updates must reuse cached SolvedUiLayout with 0 Tropical solves',
  );
  assert.equal(f1.layout.layoutDigest, f0.layout.layoutDigest);

  // Changing viewport or surfaceMode must cleanly trigger a new Tropical layout solve
  root.getObserverFrame({
    activeView: { kind: 'prism', prismId: created.prismId, mode: 'studio' },
    focusedPrism: proj1,
    prisms: root.prismStore.list(),
    surfaceMode: 'details',
    viewportWidthPx: 1440,
    viewportHeightPx: 900,
    darkMode: true,
    userTyping: false,
  });
  assert.equal(root.getLayoutSolveCount(), solvesAfterFirst + 1);
  root.dispose();
});

test('64-token KV page-boundary checkpoint yields long background part cleanly for waiting focused part', () => {
  assert.equal(
    shouldCheckpointAtKvPageBoundary({
      checkpointRequested: true,
      jobPartId: 'bg-thread',
      focusedPartId: 'fg-studio',
      hasWaitingFocusedJob: true,
      emittedTokensInSlice: 64,
      totalEmittedTokens: 128,
      remainingMaxTokens: 128,
    }),
    true,
    'Background part at exact 64-token page boundary (>=128 total, >=32 left) must checkpoint cleanly',
  );

  // Must NOT checkpoint off-boundary (e.g. token 129) or when the job IS the focused part
  assert.equal(
    shouldCheckpointAtKvPageBoundary({
      checkpointRequested: true,
      jobPartId: 'bg-thread',
      focusedPartId: 'fg-studio',
      hasWaitingFocusedJob: true,
      emittedTokensInSlice: 65,
      totalEmittedTokens: 129,
      remainingMaxTokens: 127,
    }),
    false,
  );
  assert.equal(
    shouldCheckpointAtKvPageBoundary({
      checkpointRequested: true,
      jobPartId: 'fg-studio',
      focusedPartId: 'fg-studio',
      hasWaitingFocusedJob: true,
      emittedTokensInSlice: 64,
      totalEmittedTokens: 128,
      remainingMaxTokens: 128,
    }),
    false,
  );
});

test('Active Observer perceive(), diagnose(), and guarded enact() work under FK-07 and emit OTEL spans', async () => {
  const { client } = createStubWorkerClient();
  const root = new VerdantRootPrismRuntime({ workerClient: client });
  let currentView = 'thread';
  let currentDark = false;
  root.setNavigationCallbacks({
    onOpenHome: () => { currentView = 'thread'; },
    onOpenSettings: () => { currentView = 'settings'; },
    onOpenGallery: (mt) => { currentView = `gallery:${mt}`; },
    onOpenPrism: (id) => { currentView = `prism:${id}`; },
    onSetDarkMode: (d) => { currentDark = d; },
    onGetUiState: () => ({
      activeView: currentView,
      surfaceMode: 'preview',
      darkMode: currentDark,
      storesHydrated: true,
    }),
  });

  const p0 = root.perceiveGuiState();
  assert.equal(p0.ui.activeView, 'thread');
  assert.ok(p0.parts.length >= 4);

  const navRes = await root.enactGuiAction({ type: 'open_gallery', mediaType: 'audio' });
  assert.equal(navRes.ok, true);
  assert.equal(currentView, 'gallery:audio');

  const createRes = await root.enactGuiAction({
    type: 'create_artifact',
    prompt: 'Compose a calm ambient synth track at 82 BPM',
    mediaType: 'audio',
  });
  assert.equal(createRes.ok, true);

  const diag = root.diagnoseSystemHealth();
  assert.equal(diag.status, 'healthy');
  assert.ok(diag.overallHealthQ10 >= 768);
  assert.equal(diag.featureCoverage.audio, 1);

  const healRes = await root.enactGuiAction({ type: 'self_heal' });
  assert.equal(healRes.ok, true);

  // FK-07 guard: denying prism-runtime must block create_artifact cleanly
  client.cognitiveObserver.grants.deny('prism-runtime');
  const blocked = await root.enactGuiAction({
    type: 'create_artifact',
    prompt: 'Create a picture of a mountain',
  });
  assert.equal(blocked.ok, false);
  assert.match(blocked.error ?? '', /ContainerGrantMatrix/i);

  const spanNames = root.observerBus.getRecentSpans().map((s) => s.name);
  assert.ok(spanNames.includes('observer.perceive'));
  assert.ok(spanNames.includes('observer.diagnose'));
  assert.ok(spanNames.includes('observer.enact'));
  root.dispose();
});
