import test from 'node:test';
import assert from 'node:assert/strict';
import { ContentGalleryStore, inferCategoriesForArtifact } from './contentGalleryStore';
import { PrismStore } from './prismStudio';
import { assembleProceduralPrismOps, parseDiscoveryAgentResponse } from './prismStudioAgent';

test('ContentGalleryStore globally indexes artifacts by MediaType, sub-categories, and multi-Space membership', async () => {
  const store = new PrismStore();
  const galleries = new ContentGalleryStore();

  // Create a Picture Prism
  const picProj = await store.createDeclarative('undefined');
  const picSpec = parseDiscoveryAgentResponse(
    '',
    'Generate a misty alpine mountain lake sunset picture',
    picProj.declaration!,
    2,
    'picture',
  ).spec;
  await store.applyStudioOp(
    picProj.id,
    { op: 'batch', ops: assembleProceduralPrismOps(picProj.declaration!, picSpec, 'alpine mountain lake sunset').ops },
    { pushRevision: false },
  );
  await store.publish(picProj.id);

  // Create a Space Prism
  const spaceProj = await store.createDeclarative('undefined');
  const spaceSpec = parseDiscoveryAgentResponse(
    JSON.stringify({ name: 'Alpine Expedition', mediaType: 'space', readyToBuild: true, weights: { mediaType: 'space', title: 'Alpine Expedition', layout: 'grid', items: [] } }),
    'Create a Space called Alpine Expedition',
    spaceProj.declaration!,
    2,
    'space',
  ).spec;
  await store.applyStudioOp(
    spaceProj.id,
    { op: 'batch', ops: assembleProceduralPrismOps(spaceProj.declaration!, spaceSpec, 'Alpine Expedition').ops },
    { pushRevision: false },
  );
  await store.publish(spaceProj.id);

  galleries.syncFromPrisms(store.list());
  assert.equal(galleries.galleryCounts().picture, 1);
  // Space Prisms are NOT placed inside content galleries
  assert.equal(galleries.listAll().length, 1);

  const pics = galleries.listByGallery('picture');
  assert.equal(pics.length, 1);
  assert.ok(pics[0]!.categories.includes('Landscapes'));

  // Assign artifact into the Space Prism
  galleries.assignToSpace(picProj.id, spaceProj.id);
  const spaceItems = galleries.listForSpace(spaceProj.id);
  assert.equal(spaceItems.length, 1);
  assert.equal(spaceItems[0]!.id, picProj.id);

  // Filter gallery by Space and by Category
  assert.equal(galleries.listByGallery('picture', { spaceId: spaceProj.id }).length, 1);
  assert.equal(galleries.listByGallery('picture', { category: 'Landscapes' }).length, 1);
  assert.equal(galleries.listByGallery('picture', { category: 'Pixel Art' }).length, 0);

  // Category inference covers diverse media types cleanly
  assert.ok(inferCategoriesForArtifact('audio', 'Lo-Fi Beat', '112 bpm drum pulse', { style: 'rhythm-pulse' }).includes('Beats & Rhythm'));
  assert.ok(inferCategoriesForArtifact('interactive-game', 'Breakout', 'arcade bricks', { archetype: 'arcade-canvas' }).includes('Arcade'));
});
