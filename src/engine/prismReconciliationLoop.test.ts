import assert from 'node:assert/strict';
import test from 'node:test';
import { VerdantCognitiveObserver } from './cognitiveObserver';
import { ContentGalleryStore } from './contentGalleryStore';
import { ObserverStreamBus } from './observerStreamBus';
import { PrismHostPool } from './prismHostPool';
import {
  contextualizePromptToUnifiedWeights,
  runProgressivePrismReconciliation,
} from './prismReconciliationLoop';
import { PrismStore } from './prismStudio';
import { StudioChatBridge } from './studioChatBridge';
import { UnifiedExpertRegistry } from './unifiedModelBasis';
import { evaluateSemanticVisualQa } from './visionCritiqueLoop';
import { renderPictureWithCoherenceCheck } from './proceduralPictureRenderer';
import { buildPictureRuntimeJs } from './prismVisualRuntimes';
import type { PictureWeights } from './prismMediaOrgans';

test('Closed-loop "a dog" synthesis reconciles Lod0 -> Lod3 with Vision QA and shares semantics across adjacent Gallery MediaTypes', async () => {
  const store = new PrismStore();
  const bus = new ObserverStreamBus();
  const observer = new VerdantCognitiveObserver();
  const galleries = new ContentGalleryStore();
  const registry = new UnifiedExpertRegistry();
  const pool = new PrismHostPool(bus, store, observer.grants, null);

  const bridge = new StudioChatBridge({
    store,
    pool,
    observer,
    galleries,
    registry,
  });

  // 1. User prompts "a dog" as a Picture -> Language contextualization writes UnifiedSemanticWeights
  const dogPic = await bridge.createArtifactFromPrompt({
    prompt: 'Create a picture of a dog in sunlit grass',
    forcedMediaType: 'picture',
  });
  const picProj = store.get(dogPic.prismId)!;
  assert.ok(picProj.declaration?.concept.semanticConceptId, 'Picture Prism must record semanticConceptId');

  const initialRev = registry.semantics.getConcept(picProj.declaration!.concept.semanticConceptId!)!;
  assert.ok(initialRev);

  // 2. Drive monotonic 4-tier progressive reconciliation Lod0 -> Lod1 -> Lod2 -> Lod3 with Vision QA
  const seenRanks: number[] = [];
  const recon = await runProgressivePrismReconciliation({
    prismId: dogPic.prismId,
    prompt: 'a dog in sunlit grass',
    initialRevision: initialRev,
    startRank: 0,
    maxRank: 3,
    onProgress: (ev) => {
      if (!seenRanks.includes(ev.lodRank)) seenRanks.push(ev.lodRank);
    },
    deps: { store, galleries, registry },
  });

  assert.equal(recon.cancelled, false);
  assert.equal(recon.finalRank, 3, 'Must converge to Lod3LosslessSource');
  assert.deepEqual(seenRanks, [0, 1, 2, 3], 'LOD progression must be strictly monotonic 0 -> 1 -> 2 -> 3');

  const reconciledProj = store.get(dogPic.prismId)!;
  assert.equal(reconciledProj.declaration?.concept.lodState?.label, 'Lod3LosslessSource');
  assert.ok((reconciledProj.declaration?.concept.lodState?.qaPasses ?? 0) >= 4);

  // 3. Adjacent modality ("make a video of that dog") inherits the SAME unified "dog" semantic anchor!
  const ctxVideo = await contextualizePromptToUnifiedWeights({
    prompt: 'Make a short video of that dog',
    mediaType: 'video',
    activePrismId: dogPic.prismId,
    deps: { registry },
  });
  assert.equal(ctxVideo.inheritedAnchor, true, 'Adjacent video must inherit reconciled dog anchor');
  assert.equal(ctxVideo.revision.conceptId, recon.finalRevision.conceptId);
  assert.ok(ctxVideo.revision.version > recon.finalRevision.version);

  // 4. Negative QA check: blank frame must be refused by evaluateSemanticVisualQa
  const blankRgba = new Uint8ClampedArray(64 * 64 * 4);
  for (let i = 3; i < blankRgba.length; i += 4) blankRgba[i] = 255;
  const blankQa = await evaluateSemanticVisualQa(
    { rgba: blankRgba, width: 64, height: 64 },
    {
      conceptId: initialRev.conceptId,
      canonicalSubject: 'dog',
      contextualBrief: initialRev.contextualBrief,
      attributes: initialRev.attributes,
      lighting: 'daylight',
      semanticVector: initialRev.semanticVector,
    },
  );
  assert.equal(blankQa.accepted, false, 'Blank frame must be refused by semantic Vision QA');
  assert.ok(blankQa.missingAttributes.length > 0);
  pool.dispose();
});

test('refinePrismFromPrompt advances UnifiedSemanticAssociator (vN->vN+1) and schedules Lod1->Lod3 reconciliation', async () => {
  const store = new PrismStore();
  const bus = new ObserverStreamBus();
  const observer = new VerdantCognitiveObserver();
  const galleries = new ContentGalleryStore();
  const registry = new UnifiedExpertRegistry();
  const pool = new PrismHostPool(bus, store, observer.grants, null);

  const scheduled: Array<{ prismId: string; version: number; startRank?: number; maxRank?: number }> = [];
  const bridge = new StudioChatBridge({
    store,
    pool,
    observer,
    galleries,
    registry,
    scheduleProgressiveReconciliation: (job) => {
      scheduled.push({
        prismId: job.prismId,
        version: job.initialRevision.version,
        startRank: job.startRank,
        maxRank: job.maxRank,
      });
    },
  });

  const created = await bridge.createArtifactFromPrompt({
    prompt: 'Create a picture of a golden dog',
    forcedMediaType: 'picture',
  });
  const p0 = store.get(created.prismId)!;
  const v1 = p0.declaration?.concept.semanticVersion ?? 0;
  const k1 = p0.declaration?.concept.semanticKappa ?? '';
  assert.ok(v1 >= 1 && k1.length >= 8);

  await bridge.refinePrismFromPrompt({
    target: created.prismId,
    instruction: 'Make the lighting golden hour with warmer sunlight',
  });
  const p1 = store.get(created.prismId)!;
  const v2 = p1.declaration?.concept.semanticVersion ?? 0;
  assert.ok(v2 > v1, 'Refinement must advance semanticVersion (vN -> vN+1)');
  assert.notEqual(p1.declaration?.concept.semanticKappa, k1);

  const latestRec = registry.semantics.getConcept(p1.declaration!.concept.semanticConceptId!)!;
  assert.equal(latestRec.parentKappa, k1, 'Refined semantic revision must chain parentKappa to v1');
  assert.equal(scheduled.length, 2, 'Both create and refine must schedule progressive reconciliation');
  assert.equal(scheduled[1]!.startRank, 1);
  assert.equal(scheduled[1]!.maxRank, 3);
  pool.dispose();
});

test('100% neural primary Picture display persists reconciledImageDataUrl into sandbox and strips it from IDB undo revisions', async () => {
  // Distinct sharp neural layer (bright gold/indigo high-contrast pattern) to prove 100% primary display
  const nlW = 64;
  const nlH = 64;
  const nlRgba = new Uint8ClampedArray(nlW * nlH * 4);
  for (let y = 0; y < nlH; y++) {
    for (let x = 0; x < nlW; x++) {
      const idx = (y * nlW + x) * 4;
      const hi = ((x >> 1) + (y >> 1)) % 2 === 0;
      nlRgba[idx] = hi ? 240 : 20;
      nlRgba[idx + 1] = hi ? 180 : 30;
      nlRgba[idx + 2] = hi ? 40 : 210;
      nlRgba[idx + 3] = 255;
    }
  }

  const baseWeights: PictureWeights = {
    mediaType: 'picture',
    generator: 'landscape-vista',
    lighting: 'golden-hour',
    width: 160,
    height: 120,
    seed: 1234,
    symmetry: 4,
    density: 60,
    palette: ['#000000', '#111111', '#222222', '#333333'],
    caption: 'A golden dog',
  };

  const rendered = await renderPictureWithCoherenceCheck(baseWeights, {
    maxPreviewDim: 160,
    neuralLayer: { width: nlW, height: nlH, rgba: nlRgba },
  });
  // Top-left pixel (0,0) must match the neural layer's (240, 180, 40) at 100% fidelity (not 24% blended over #000000!)
  assert.ok(rendered.rgba[0]! >= 230 && rendered.rgba[1]! >= 170, 'Neural layer must render at 100% primary fidelity');
  assert.ok(
    rendered.weights.reconciledImageDataUrl?.startsWith('data:image/'),
    'PictureWeights must persist compact reconciledImageDataUrl',
  );

  // Live Picture Prism sandbox runtime must embed and draw reconciledImageDataUrl
  const runtimeJs = buildPictureRuntimeJs(rendered.weights);
  assert.ok(runtimeJs.includes('reconciledImageDataUrl') && runtimeJs.includes('useReconciledImage'));

  // PrismStore must strip reconciledImageDataUrl from archived declarationRevisions undo snapshots
  const store = new PrismStore();
  const bus = new ObserverStreamBus();
  const observer = new VerdantCognitiveObserver();
  const galleries = new ContentGalleryStore();
  const registry = new UnifiedExpertRegistry();
  const pool = new PrismHostPool(bus, store, observer.grants, null);
  const bridge = new StudioChatBridge({ store, pool, observer, galleries, registry });
  const created = await bridge.createArtifactFromPrompt({
    prompt: 'Create a picture of a golden dog',
    forcedMediaType: 'picture',
  });
  await store.applyStudioOp(
    created.prismId,
    {
      op: 'reconcile-semantic-weights',
      semanticConceptId: 'dog',
      semanticKappa: 'sha256:test',
      semanticVersion: 2,
      lodState: {
        rank: 3,
        label: 'Lod3LosslessSource',
        qaScoreQ10: 900,
        qaPasses: 1,
        qaSummary: 'Accepted',
        updatedAt: Date.now(),
      },
      weights: rendered.weights,
    },
    { pushRevision: false },
  );
  // Push a second revision so the first declaration with reconciledImageDataUrl enters declarationRevisions
  await store.applyStudioOp(
    created.prismId,
    { op: 'set-metadata', description: 'Second edit' },
    { pushRevision: true },
  );
  const updated = store.get(created.prismId)!;
  const activeW = updated.declaration?.view?.proceduralOrgan?.weights as PictureWeights;
  const archivedW = updated.declarationRevisions?.[updated.declarationRevisions.length - 1]?.view?.proceduralOrgan
    ?.weights as PictureWeights;
  assert.ok(activeW.reconciledImageDataUrl, 'Active PictureWeights keeps reconciledImageDataUrl');
  assert.equal(archivedW.reconciledImageDataUrl, undefined, 'Archived undo snapshot must strip reconciledImageDataUrl');
  pool.dispose();

  // Verify runVlmCritique callback integration & calibrated residualNorm
  const qaAccepted = await evaluateSemanticVisualQa(
    { rgba: rendered.rgba, width: rendered.width, height: rendered.height },
    {
      conceptId: 'dog',
      canonicalSubject: 'dog',
      contextualBrief: 'golden dog',
      attributes: ['golden fur'],
      lighting: 'golden-hour',
    },
    { runVlmCritique: async () => 'ACCEPTED' },
  );
  assert.equal(qaAccepted.vlmCritique, 'ACCEPTED');
  assert.equal(qaAccepted.accepted, true);
  assert.ok(qaAccepted.residualNorm < 0.35);

  const qaRefused = await evaluateSemanticVisualQa(
    { rgba: rendered.rgba, width: rendered.width, height: rendered.height },
    {
      conceptId: 'dog',
      canonicalSubject: 'dog',
      contextualBrief: 'golden dog',
      attributes: ['golden fur'],
      lighting: 'golden-hour',
    },
    { runVlmCritique: async () => 'REFINE: missing dog face detail' },
  );
  assert.equal(qaRefused.accepted, false);
  assert.ok(qaRefused.residualNorm > qaAccepted.residualNorm);
});
