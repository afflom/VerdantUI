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
