import test from 'node:test';
import assert from 'node:assert/strict';
import {
  inferPictureWeightsFromPrompt,
  renderPictureWithCoherenceCheck,
} from './proceduralPictureRenderer';
import type { PictureGeneratorKind } from './prismMediaOrgans';

test('inferPictureWeightsFromPrompt maps natural-language descriptions to appropriate scene generators & palettes', () => {
  const vista = inferPictureWeightsFromPrompt('A misty pine mountain lake at sunset', 'Mountain Lake');
  assert.equal(vista.generator, 'landscape-vista');
  assert.equal(vista.lighting, 'golden-hour');

  const nebula = inferPictureWeightsFromPrompt('Deep space cosmic nebula with ringed planet at midnight', 'Nebula');
  assert.equal(nebula.generator, 'cosmic-nebula');
  assert.equal(nebula.lighting, 'midnight');

  const arch = inferPictureWeightsFromPrompt('Emerald glass greenhouse architecture in rain', 'Greenhouse');
  assert.equal(arch.generator, 'architectural-scene');

  const bot = inferPictureWeightsFromPrompt('Sunlit botanical blossom garden', 'Garden');
  assert.equal(bot.generator, 'botanical-garden');

  const ocean = inferPictureWeightsFromPrompt('Moonlit ocean seascape with cresting waves', 'Ocean');
  assert.equal(ocean.generator, 'ocean-seascape');
});

test('All 10 procedural picture scene generators produce deterministic coherent RGBA & valid PNG data URLs', async () => {
  const allGenerators: readonly PictureGeneratorKind[] = [
    'landscape-vista',
    'cosmic-nebula',
    'architectural-scene',
    'botanical-garden',
    'ocean-seascape',
    'abstract-flow',
    'geometric-mosaic',
    'constellation-atlas',
    'poster-editorial',
    'pixel-scene',
  ];

  for (const gen of allGenerators) {
    const res = await renderPictureWithCoherenceCheck(
      {
        mediaType: 'picture',
        generator: gen,
        lighting: gen === 'cosmic-nebula' ? 'midnight' : 'golden-hour',
        width: 240,
        height: 160,
        seed: 7331,
        symmetry: 6,
        density: 56,
        palette: ['#0a1614', '#0f766e', '#f59e0b', '#fef3c7'],
        caption: `Test ${gen}`,
      },
      { maxPreviewDim: 240 },
    );
    assert.equal(res.width, 240);
    assert.equal(res.height, 160);
    assert.equal(res.rgba.length, 240 * 160 * 4);
    assert.ok(res.pngDataUrl.startsWith('data:image/png;base64,'), `Expected valid PNG data URL for ${gen}`);
    assert.equal(res.coherence.coherent, true, `Generator ${gen} failed coherence: ${JSON.stringify(res.coherence)}`);
    assert.ok(res.coherence.nonUniformRatio > 0.05, `Generator ${gen} rendered nearly blank`);
  }
});
