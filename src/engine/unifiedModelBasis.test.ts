import test from 'node:test';
import assert from 'node:assert/strict';
import {
  LivingTrainingBasis,
  UnifiedExpertRegistry,
  facetsQ10ForMediaType,
} from './unifiedModelBasis';

test('UnifiedExpertRegistry unifies neural weights, procedural organs, user basis, and growing skills', () => {
  const basis = new LivingTrainingBasis();
  const registry = new UnifiedExpertRegistry(basis);

  const initial = registry.telemetry();
  assert.equal(initial.proceduralExperts, 8);
  assert.equal(initial.neuralExperts, 0);
  assert.equal(initial.adaptationSteps, 0);

  // Register downloaded neural weight experts into the same unified model
  registry.registerNeuralWeightExpert({
    modality: 'language',
    modelId: 'qwen3-0.6b-q4',
    label: 'Qwen3 0.6B',
    residency: 'hot-vtensor',
    residentBytes: 420_000_000,
  });
  registry.registerNeuralWeightExpert({
    modality: 'vision',
    modelId: 'smolvlm2-mmproj',
    label: 'SigLIP Vision',
    residency: 'warm-opfs',
    residentBytes: 92_000_000,
  });

  // Online adaptation grows the non-static training basis
  const snap1 = basis.recordAdaptation({
    mediaType: 'picture',
    expertId: 'content:pic1',
    kappa: 'kappa_pic1',
    title: 'Emerald Mountain Lake',
    summary: 'Misty alpine sunset lake',
    categories: ['Landscapes'],
    pictureStyle: 'landscape-vista',
    pictureLighting: 'golden-hour',
    newlyAcquiredSkill: true,
  });

  assert.equal(snap1.adaptationSteps, 1);
  assert.equal(snap1.acquiredSkillIds.includes('content:pic1'), true);
  assert.equal(basis.getStyleAffinities().preferredPictureStyle, 'landscape-vista');
  assert.equal(basis.getStyleAffinities().preferredPictureLighting, 'golden-hour');
  assert.ok((basis.getStyleAffinities().mediaAffinityQ10.picture ?? 0) > 512);

  // Basis recall finds created artifacts & skills by keyword
  const recalled = basis.recallKnowledge({ text: 'alpine sunset' });
  assert.equal(recalled.length, 1);
  assert.equal(recalled[0]!.title, 'Emerald Mountain Lake');

  // Exact integer Q10 expert ranking works coherently across all expert kinds
  const ranked = registry.rankExpertsForVector(facetsQ10ForMediaType('picture'), { limit: 5 });
  assert.ok(ranked.length >= 3);
  assert.ok(ranked[0]!.resonanceQ10 > 0);

  const tele = registry.telemetry();
  assert.equal(tele.neuralExperts, 2);
  assert.equal(tele.hotResidentBytes, 420_000_000);
  assert.equal(tele.warmOpfsBytes, 92_000_000);
  assert.equal(tele.acquiredSkills, 1);
  assert.equal(tele.knowledgeEntries, 1);
});
