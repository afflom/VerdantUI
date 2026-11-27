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
