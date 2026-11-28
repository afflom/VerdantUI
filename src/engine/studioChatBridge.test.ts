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
  assert.equal(detectStudioChatIntent('Add that picture to Mars Mission')?.toolName, 'studio_manage_spaces');
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

  pool.dispose();
});
