import assert from 'node:assert/strict';
import test from 'node:test';
import {
  analyzeAudioFidelity,
  composeMultiTrackArrangement,
  encodeCompositionToMidi,
  encodeStereoWav,
  masterStereoPcm,
  synthesizeCompositionStereoPcm,
} from './audioSynthesisEngine';
import { VerdantCognitiveObserver } from './cognitiveObserver';
import { ContentGalleryStore } from './contentGalleryStore';
import { MULTIMODAL_CORE_MODELS } from './modelCatalog';
import { ObserverStreamBus } from './observerStreamBus';
import { compilePrismDeclaration } from './prismCompiler';
import { PrismHostPool } from './prismHostPool';
import { PrismStore } from './prismStudio';
import { classifyArtifactMediaType } from './prismStudioAgent';
import {
  inferPictureWeightsFromPrompt,
  renderPictureWithCoherenceCheck,
  synthesizeLogoSvg,
} from './proceduralPictureRenderer';
import { StudioChatBridge, detectStudioChatIntent } from './studioChatBridge';
import { UnifiedExpertRegistry } from './unifiedModelBasis';

function createTestBridge() {
  const store = new PrismStore();
  const observer = new VerdantCognitiveObserver();
  const galleries = new ContentGalleryStore();
  const registry = new UnifiedExpertRegistry();
  const bus = new ObserverStreamBus();
  const pool = new PrismHostPool(bus, store, observer.grants, null);
  const bridge = new StudioChatBridge({ store, pool, observer, galleries, registry });
  return { store, observer, galleries, registry, bridge };
}

test('SDXS image expert pin points to verified Q8_0 GGUF revision (no 404 q4_0 URL)', () => {
  const imgSpec = MULTIMODAL_CORE_MODELS.find((m) => m.modality === 'image-synthesis');
  assert.ok(imgSpec);
  assert.ok(imgSpec.url.includes('sdxs-512-tinySDdistilled_Q8_0.gguf'));
  assert.equal(imgSpec.url.includes('q4_0'), false);
});

test('Prompt classification accurately separates Logo, Book Cover Art, Album Cover Art, Soundtrack, and Games', () => {
  assert.equal(
    classifyArtifactMediaType('Create a minimalist emerald geometric logo for Meridian').mediaType,
    'picture',
  );
  assert.equal(
    classifyArtifactMediaType('Book cover art for The Glass Archipelago').mediaType,
    'picture',
  );
  assert.equal(
    classifyArtifactMediaType('Album cover art for Midnight Echoes').mediaType,
    'picture',
  );
  assert.equal(
    classifyArtifactMediaType('Game cover art for Starfall Bastion').mediaType,
    'picture',
  );
  assert.equal(
    classifyArtifactMediaType('Build a cozy alpine puzzle game with cover art, a logo, and a soundtrack').mediaType,
    'interactive-game',
  );
  assert.equal(
    classifyArtifactMediaType('Compose a calm 78 BPM ambient nocturnal soundtrack').mediaType,
    'audio',
  );

  const logoIntent = detectStudioChatIntent('Create a minimalist emerald geometric logo for Meridian');
  assert.equal(logoIntent?.toolName, 'studio_create_artifact');
  const companionIntent = detectStudioChatIntent('Generate cover art for this');
  assert.equal(companionIntent?.toolName, 'studio_refine_prism');
});

test('Logo and Cover Art synthesize valid SVG 2, PNG 3 signatures, and pass closed-loop visual coherence', async () => {
  // 1. Logo Mark (SVG 2 + PNG 3)
  const logoW = inferPictureWeightsFromPrompt('Create a minimalist emerald geometric logo for Meridian', 'Meridian');
  assert.equal(logoW.generator, 'logo-mark');
  assert.equal(logoW.width, 640);
  assert.equal(logoW.height, 640);
  const svg = logoW.svgMarkup ?? synthesizeLogoSvg(logoW);
  assert.ok(svg.startsWith('<svg xmlns="http://www.w3.org/2000/svg"'));
  assert.ok(svg.includes('MERIDIAN'));
  assert.equal(svg.includes('<script'), false);
  const renderedLogo = await renderPictureWithCoherenceCheck(logoW, { maxPreviewDim: 320 });
  assert.ok(renderedLogo.pngDataUrl.startsWith('data:image/png;base64,'));
  assert.equal(renderedLogo.coherence.coherent, true);

  // Verify real PNG 8-byte signature in base64 payload
  const pngBytes = Buffer.from(renderedLogo.pngDataUrl.split(',')[1]!, 'base64');
  assert.deepEqual(Array.from(pngBytes.subarray(0, 8)), [137, 80, 78, 71, 13, 10, 26, 10]);

  // 2. Book Cover Art (2:3 vertical ratio)
  const bookW = inferPictureWeightsFromPrompt('Book cover art for The Glass Archipelago', 'The Glass Archipelago');
  assert.equal(bookW.generator, 'cover-art');
  assert.equal(bookW.coverFormat, 'book');
  assert.equal(bookW.width, 512);
  assert.equal(bookW.height, 768);
  const renderedCover = await renderPictureWithCoherenceCheck(bookW, { maxPreviewDim: 360 });
  assert.equal(renderedCover.coherence.coherent, true);
  assert.ok(renderedCover.pngDataUrl.startsWith('data:image/png;base64,'));
});

test('Audio mastering enforces headroom, zero clipping/NaN, valid RIFF/WAVE, and valid Standard MIDI 1.0', () => {
  const comp = composeMultiTrackArrangement({
    title: 'Nocturnal Horizon',
    style: 'ambient-pad',
    bpm: 78,
    rootMidi: 57,
    scale: 'minor',
    durationSec: 6,
  });
  const raw = synthesizeCompositionStereoPcm(comp, 22_050);
  const mastered = masterStereoPcm(raw.left, raw.right, raw.sampleRate);
  const report = analyzeAudioFidelity(mastered.left, mastered.right);
  assert.equal(report.hasNaN, false);
  assert.equal(report.clippedSamples, 0);
  assert.equal(report.nonSilent, true);
  assert.ok(report.peakDbfs <= -0.9, `Expected peak <= -0.9 dBFS, got ${report.peakDbfs}`);
  assert.ok(report.rmsDbfs > -32 && report.rmsDbfs < -8, `Unexpected RMS ${report.rmsDbfs} dBFS`);

  const wav = encodeStereoWav(mastered.left, mastered.right, mastered.sampleRate);
  assert.equal(new TextDecoder().decode(wav.subarray(0, 4)), 'RIFF');
  assert.equal(new TextDecoder().decode(wav.subarray(8, 12)), 'WAVE');

  const midi = encodeCompositionToMidi(comp);
  assert.equal(new TextDecoder().decode(midi.subarray(0, 4)), 'MThd');
  assert.equal(new TextDecoder().decode(midi.subarray(14, 18)), 'MTrk');
  assert.ok(midi.byteLength > 40);
});

test('1-word seeds, feature-derived companion generation, and pin-preserving tuning stay coherent end-to-end', async () => {
  const { store, galleries, bridge } = createTestBridge();

  // 1-word seed ("Minesweeper") compiles cleanly with zero missing fields
  const singleWord = await bridge.createArtifactFromPrompt({ prompt: 'Minesweeper' });
  assert.equal(singleWord.mediaType, 'interactive-game');
  const swProj = store.get(singleWord.prismId);
  assert.ok(swProj?.declaration);
  assert.ok(compilePrismDeclaration(swProj.declaration) !== null);

  // Multi-facet request generates primary Game + coherent Cover Art + Logo + Soundtrack indexed in Galleries
  const gameBundle = await bridge.createArtifactFromPrompt({
    prompt: 'Build a cozy alpine puzzle game with cover art, a logo, and a soundtrack',
  });
  assert.equal(gameBundle.mediaType, 'interactive-game');
  const parent = store.get(gameBundle.prismId)!;
  const linked = parent.declaration?.concept.linkedPrismIds ?? [];
  assert.equal(linked.length, 3, `Expected 3 companion facets (cover-art, logo, soundtrack), got ${linked.length}`);
  assert.ok(galleries.listByGallery('picture').length >= 2);
  assert.ok(galleries.listByGallery('audio').length >= 1);

  // Pin-preserving tuning on Audio: user pins 118 BPM, then tunes mood to "darker minor" -> BPM stays 118, scale becomes minor
  const track = await bridge.createArtifactFromPrompt({
    prompt: 'Compose a bright 118 BPM melodic synth track',
  });
  assert.equal(track.mediaType, 'audio');
  const beforeProj = store.get(track.prismId)!;
  assert.equal((beforeProj.declaration?.view?.proceduralOrgan?.weights as { bpm?: number })?.bpm, 118);
  assert.ok(beforeProj.declaration?.concept.pinnedFields?.includes('bpm'));

  await bridge.refinePrismFromPrompt({
    target: track.prismId,
    instruction: 'Make the mood darker and more nocturnal',
  });
  const afterProj = store.get(track.prismId)!;
  const afterW = afterProj.declaration?.view?.proceduralOrgan?.weights as { bpm?: number; scale?: string };
  assert.equal(afterW.bpm, 118, 'User-pinned BPM must be preserved across mood tuning');
  assert.equal(afterW.scale, 'minor', 'Unpinned scale should adapt to darker nocturnal mood');

  // Concise noun-phrase media seeds and brand subjects must route to creation without bogus Space capture
  const coverIntent = detectStudioChatIntent('Book cover art for The Glass Archipelago');
  assert.equal(coverIntent?.toolName, 'studio_create_artifact');
  assert.equal(coverIntent?.args.spaceName, undefined);
  const logoIntent = detectStudioChatIntent('Create a minimalist emerald geometric logo for Meridian');
  assert.equal(logoIntent?.toolName, 'studio_create_artifact');
  assert.equal(logoIntent?.args.spaceName, undefined);

  // Observer active modality residency reporting
  const obs = new VerdantCognitiveObserver();
  obs.setModalityResidencyProvider(() => ({
    language: true,
    image: true,
    vision: false,
    audio: true,
    basis: true,
  }));
  const residency = obs.getActiveModalityResidency();
  assert.equal(residency.language, true);
  assert.equal(residency.image, true);
  assert.equal(residency.audio, true);
  assert.equal(residency.basis, true);
});
