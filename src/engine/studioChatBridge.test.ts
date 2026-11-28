import test from 'node:test';
import assert from 'node:assert/strict';
import { VerdantCognitiveObserver } from './cognitiveObserver';
import { ContentGalleryStore } from './contentGalleryStore';
import { ObserverStreamBus } from './observerStreamBus';
import { PrismHostPool } from './prismHostPool';
import { PrismStore } from './prismStudio';
import { detectStudioChatIntent, StudioChatBridge } from './studioChatBridge';
import { UnifiedExpertRegistry } from './unifiedModelBasis';

test('detectStudioChatIntent recognizes creation, refinement, Space curation, and Gallery intents without hijacking questions', () => {
  assert.equal(detectStudioChatIntent('Generate a picture of a misty emerald mountain lake at sunset')?.toolName, 'studio_create_artifact');
  assert.equal(detectStudioChatIntent('Compose a calm ambient dorian synth track at 84 BPM')?.toolName, 'studio_create_artifact');
  assert.equal(detectStudioChatIntent('Build a playable Breakout arcade game')?.toolName, 'studio_create_artifact');
  assert.equal(detectStudioChatIntent('Create a Space called Mars Mission')?.toolName, 'studio_manage_spaces');
  const addLatestPic = detectStudioChatIntent('Add that picture to Mars Mission');
  assert.equal(addLatestPic?.toolName, 'studio_manage_spaces');
  assert.equal(addLatestPic?.args.artifact, 'latest');
  assert.equal(addLatestPic?.args.space, 'Mars Mission');
  const addBudget = detectStudioChatIntent('Add that budget tracker to Science Lab');
  assert.equal(addBudget?.toolName, 'studio_manage_spaces');
  assert.equal(addBudget?.args.artifact, 'latest');
  assert.equal(addBudget?.args.space, 'Science Lab');
  const addNamed = detectStudioChatIntent('Add "Budget Guard" to Science Lab');
  assert.equal(addNamed?.args.artifact, 'Budget Guard');
  assert.equal(addNamed?.args.space, 'Science Lab');
  assert.equal(detectStudioChatIntent('Show my Pictures gallery')?.toolName, 'studio_browse_gallery');
  assert.equal(detectStudioChatIntent('Update that picture to midnight with stars')?.toolName, 'studio_refine_prism');
  assert.equal(detectStudioChatIntent('List my Prisms')?.toolName, 'studio_manage_spaces');

  // Informational questions & ordinary English sentences must NEVER be hijacked
  assert.equal(detectStudioChatIntent('What is a nebula?'), null);
  assert.equal(detectStudioChatIntent('Explain how Minesweeper works'), null);
  assert.equal(detectStudioChatIntent('How do video codecs work?'), null);
  assert.equal(detectStudioChatIntent('Write a Python function to sort a list'), null);
  assert.equal(detectStudioChatIntent('Open a savings account'), null);
  assert.equal(detectStudioChatIntent('Delete duplicate rows in SQL'), null);

  // Extended REQ-PRISM-WORKS natural-language creation intents
  assert.equal(detectStudioChatIntent('Create a research report on battery recycling')?.toolName, 'studio_create_artifact');
  assert.equal(detectStudioChatIntent('Build a pomodoro timer app')?.toolName, 'studio_create_artifact');
  assert.equal(detectStudioChatIntent('Open my prism Mars Mission')?.toolName, 'studio_manage_spaces');
});

test('StudioChatBridge creates inline PNG Pictures, Stereo WAV Audio, Games, Spaces-as-Prisms, refines, and deletes cleanly', async () => {
  const store = new PrismStore();
  const bus = new ObserverStreamBus();
  const observer = new VerdantCognitiveObserver();
  const pool = new PrismHostPool(bus, store, observer.grants, null);
  const galleries = new ContentGalleryStore();
  const registry = new UnifiedExpertRegistry();

  const bridge = new StudioChatBridge({ store, pool, observer, galleries, registry });

  // 1. Create a high-spec Picture directly from chat -> real inline PNG + Pictures gallery + Living Basis adaptation
  const picRes = await bridge.createArtifactFromPrompt({
    prompt: 'Generate a picture of a misty emerald mountain lake at sunset',
  });
  assert.equal(picRes.mediaType, 'picture');
  assert.equal(picRes.galleryLabel, 'Pictures');
  assert.ok(picRes.previewImageDataUrl?.startsWith('data:image/png;base64,'));
  assert.equal(galleries.galleryCounts().picture, 1);
  assert.equal(registry.basis.snapshot().adaptationSteps, 1);

  // 2. Create a Space Prism and attach the latest picture to it
  const spaceRes = await bridge.manageSpacesAndPrisms({
    action: 'create',
    name: 'Alpine Expedition',
    addLatest: true,
  });
  assert.equal(spaceRes.mediaType, 'space');
  assert.equal(galleries.listForSpace(spaceRes.prismId).length, 1);

  // 3. Create Stereo WAV Audio directly from chat attached to that Space
  const audioRes = await bridge.createArtifactFromPrompt({
    prompt: 'Compose a calm ambient dorian synth track at 86 BPM',
    spaceName: 'Alpine Expedition',
  });
  assert.equal(audioRes.mediaType, 'audio');
  assert.ok(audioRes.audioWavDataUrl?.startsWith('data:audio/wav;base64,'));
  assert.equal(audioRes.spaceName, 'Alpine Expedition');
  assert.equal(galleries.listForSpace(spaceRes.prismId).length, 2);

  // 4. Refine the picture in natural language ("midnight with stars")
  const refRes = await bridge.refinePrismFromPrompt({
    target: picRes.prismName,
    instruction: 'midnight with a starry sky',
  });
  assert.equal(refRes.action, 'update');
  assert.ok(refRes.previewImageDataUrl?.startsWith('data:image/png;base64,'));

  // 5. Browse Pictures gallery & delete cleanly
  const galRes = bridge.browseGallery('picture');
  assert.equal(galRes.prismsList?.length, 1);

  const delRes = await bridge.manageSpacesAndPrisms({ action: 'delete', target: audioRes.prismName });
  assert.equal(delRes.action, 'delete');
  assert.equal(galleries.galleryCounts().audio, 0);

  // 6. Media-aware summaries, Pomodoro focus-timer, Written-Work starter prose & clean naming
  const synthRes = await bridge.createArtifactFromPrompt({
    prompt: 'Compose an upbeat dorian synthwave soundtrack at 118 BPM with drums',
    forcedMediaType: 'audio',
  });
  const synthProj = store.get(synthRes.prismId)!;
  assert.ok(!synthProj.draft.summary.includes('clear focal silhouette'), 'Audio must not use picture brief template');
  assert.ok(!/^n\s+upbeat/i.test(synthProj.draft.summary), 'Must not slice "an " into "n "');

  const docRes = await bridge.createArtifactFromPrompt({
    prompt: 'Write chapter 1 of a sci-fi novel about an orbital greenhouse',
    forcedMediaType: 'written-work',
  });
  assert.ok(docRes.prismName.includes('1') && /sci-fi/i.test(docRes.prismName), `Expected clean numeric/hyphenated title, got "${docRes.prismName}"`);
  const docWeights = store.get(docRes.prismId)?.declaration?.view?.proceduralOrgan?.weights;
  assert.ok(docWeights?.mediaType === 'written-work' && docWeights.chapters[0]!.body.trim().length > 40, 'Written work Chapter 1 must start with real prose');

  const pomoRes = await bridge.createArtifactFromPrompt({
    prompt: 'Build a pomodoro timer app',
    forcedMediaType: 'tool-organ',
  });
  const pomoWeights = store.get(pomoRes.prismId)?.declaration?.view?.proceduralOrgan?.weights;
  assert.equal(pomoWeights?.mediaType === 'tool-organ' ? pomoWeights.role : null, 'focus-timer');

  const flashRes = await bridge.createArtifactFromPrompt({
    prompt: 'Build a flashcards study deck for organic chemistry',
    forcedMediaType: 'tool-organ',
  });
  const flashWeights = store.get(flashRes.prismId)?.declaration?.view?.proceduralOrgan?.weights;
  assert.equal(flashWeights?.mediaType === 'tool-organ' ? flashWeights.role : null, 'flashcards');

  const budgetRes = await bridge.createArtifactFromPrompt({
    prompt: 'Build a monthly budget tracker for household expenses',
    forcedMediaType: 'tool-organ',
  });
  const budgetProj = store.get(budgetRes.prismId)!;
  const budgetWeights = budgetProj.declaration?.view?.proceduralOrgan?.weights;
  assert.equal(budgetWeights?.mediaType === 'tool-organ' ? budgetWeights.role : null, 'budget-tracker');
  assert.equal(budgetProj.draft.grants.storage, true, 'Stateful tool-organ Prisms must have scoped local storage enabled (FK-07)');

  // 7. Utility compartment visibility, deterministic fast-path refinement (<50ms, 0 neural calls), & procedural status dispatch
  const visible = observer.getVisiblePublishedPrismsForSpace(store.listPublishedSummaries(), 'personal');
  assert.ok(
    visible.some((p) => p.id === budgetRes.prismId),
    'Budget Guard tool-organ must be visible in default personal Published Prism registry',
  );
  let neuralCalls = 0;
  const guardedBridge = new StudioChatBridge({
    store,
    pool,
    observer,
    galleries,
    registry,
    runLocalStudioPrompt: async () => {
      neuralCalls += 1;
      return '';
    },
  });
  await guardedBridge.refinePrismFromPrompt({
    target: budgetRes.prismId,
    instruction: 'Add expense $18 lunch',
  });
  assert.equal(neuralCalls, 0, 'Deterministic "Add expense $18 lunch" must NOT invoke slow neural rewrite');
  const updatedBudget = store.get(budgetRes.prismId)!;
  const bw = updatedBudget.declaration?.view?.proceduralOrgan?.weights as { expenses?: Array<{ label: string; amount: number }> };
  assert.ok(
    bw.expenses?.some((e) => e.label.toLowerCase() === 'lunch' && e.amount === 18),
    'Budget Guard weights must contain the newly added $18 lunch expense',
  );
  const statusRes = await pool.dispatchPublishedPrism(budgetRes.prismId, { action: 'input', message: 'status' });
  const statusObj = statusRes.result as Record<string, unknown>;
  assert.equal(statusObj.refused, false, 'Procedural status dispatch must NOT refuse or stringify [object Object]');
  assert.equal(statusObj.request, 'status');
  assert.equal(statusObj.status, 'ok');

  pool.dispose();
});
