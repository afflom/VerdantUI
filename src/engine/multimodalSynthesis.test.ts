import test from 'node:test';
import assert from 'node:assert/strict';
import {
  synthesizeImageOneStep,
  type ClipTextWeights,
  type DistilledUnetWeights,
  type TaesdDecoderWeights,
} from './imageSynthesisEngine';
import {
  composeMultiTrackArrangement,
  encodeStereoWav,
  synthesizeCompositionStereoPcm,
} from './audioSynthesisEngine';
import { evaluateRenderedFrameCoherence, perceiveImageFrame } from './visionCritiqueLoop';
import { MultimodalCoherenceCoordinator } from './multimodalCoordinator';
import { GgmlType } from './gguf';
import { SiglipVisionEncoder, type TensorSlice, type VisionWeights } from './visionEncoder';

function f32Slice(rows: number, cols: number, fill = 0.04): TensorSlice {
  const arr = new Float32Array(rows * cols);
  for (let i = 0; i < arr.length; i++) arr[i] = Math.sin(i * 0.23) * fill;
  return { bytes: new Uint8Array(arr.buffer), offset: 0, type: GgmlType.F32, rows, cols };
}

test('1-step distilled neural image synthesis (CLIP -> SDXS UNet -> TAESD) produces valid bounded RGBA8 pixels', () => {
  const clip: ClipTextWeights = {
    vocabSize: 32,
    embedDim: 8,
    maxSeqLen: 16,
    tokenEmbd: new Float32Array(32 * 8).map((_, i) => Math.sin(i * 0.19) * 0.2),
    posEmbd: new Float32Array(16 * 8).map((_, i) => Math.cos(i * 0.11) * 0.05),
    lnW: new Float32Array(8).fill(1),
    lnB: new Float32Array(8),
    projW: new Float32Array(8 * 8).map((_, i) => Math.sin(i * 0.37) * 0.15),
    projB: new Float32Array(8),
  };

  const C = 8;
  const unet: DistilledUnetWeights = {
    latentChannels: 4,
    modelChannels: C,
    condDim: 8,
    numGroups: 2,
    convInW: new Float32Array(C * 4 * 9).fill(0.03),
    convInB: new Float32Array(C),
    norm1W: new Float32Array(C).fill(1),
    norm1B: new Float32Array(C),
    resConv1W: new Float32Array(C * C * 9).fill(0.02),
    resConv1B: new Float32Array(C),
    condProjW: new Float32Array(C * 8).fill(0.04),
    condProjB: new Float32Array(C),
    norm2W: new Float32Array(C).fill(1),
    norm2B: new Float32Array(C),
    resConv2W: new Float32Array(C * C * 9).fill(0.02),
    resConv2B: new Float32Array(C),
    convOutW: new Float32Array(4 * C * 9).fill(0.02),
    convOutB: new Float32Array(4),
  };

  // Two 2x upsample stages: 4x4 latent -> 16x16 RGB image
  const taesd: TaesdDecoderWeights = {
    stages: [
      { inC: 4, outC: 8, convW: new Float32Array(8 * 4 * 9).map((_, i) => Math.sin(i) * 0.15), convB: new Float32Array(8).fill(0.1), upsample2x: true },
      { inC: 8, outC: 6, convW: new Float32Array(6 * 8 * 9).map((_, i) => Math.cos(i) * 0.12), convB: new Float32Array(6).fill(0.05), upsample2x: true },
    ],
    finalW: new Float32Array(3 * 6 * 9).map((_, i) => Math.sin(i * 0.5) * 0.15),
    finalB: new Float32Array(3),
  };

  const imgA = synthesizeImageOneStep({
    promptTokens: [1, 5, 12, 23],
    latentH: 4,
    latentW: 4,
    seed: 42,
    clip,
    unet,
    taesd,
  });
  const imgB = synthesizeImageOneStep({
    promptTokens: [1, 5, 12, 23],
    latentH: 4,
    latentW: 4,
    seed: 42,
    clip,
    unet,
    taesd,
  });

  assert.equal(imgA.width, 16);
  assert.equal(imgA.height, 16);
  assert.equal(imgA.rgba.length, 16 * 16 * 4);
  // Deterministic given identical seed + prompt
  assert.deepEqual(Array.from(imgA.rgba), Array.from(imgB.rgba));
  // Every alpha channel is 255 and RGB values are valid finite bytes
  for (let i = 0; i < imgA.rgba.length; i += 4) {
    assert.equal(imgA.rgba[i + 3], 255);
  }
});

test('Multi-track modal audio engine composes polyphonic arrangement and renders non-clipping stereo RIFF/WAVE', () => {
  const comp = composeMultiTrackArrangement({
    title: 'Calm Emerald Dawn',
    style: 'ambient-pad',
    bpm: 92,
    rootMidi: 60,
    scale: 'pentatonic',
    durationSec: 3,
  });
  assert.ok(comp.events.length >= 8);
  assert.ok(comp.events.some((e) => e.timbre === 'warm-pad'));
  assert.ok(comp.events.some((e) => e.timbre === 'plucked-bass'));
  assert.ok(comp.events.some((e) => e.timbre === 'modal-bell'));

  const pcm = synthesizeCompositionStereoPcm(comp, 16_000);
  assert.equal(pcm.left.length, 48_000);
  assert.equal(pcm.right.length, 48_000);

  let maxAbs = 0;
  let energy = 0;
  for (let i = 0; i < pcm.left.length; i++) {
    const al = Math.abs(pcm.left[i]!);
    const ar = Math.abs(pcm.right[i]!);
    if (al > maxAbs) maxAbs = al;
    if (ar > maxAbs) maxAbs = ar;
    energy += al + ar;
  }
  assert.ok(energy > 100, 'Synthesized audio must be non-silent');
  assert.ok(maxAbs <= 1.0, 'Soft limiter must prevent digital clipping');

  const wav = encodeStereoWav(pcm.left, pcm.right, pcm.sampleRate);
  assert.equal(new TextDecoder().decode(wav.subarray(0, 4)), 'RIFF');
  assert.equal(new TextDecoder().decode(wav.subarray(8, 12)), 'WAVE');
  assert.equal(wav.byteLength, 44 + 48_000 * 4);
});

test('Closed-loop Vision self-critique catches blank frames, approves coherent visuals, and hits 0ms kappa cache on repeat', async () => {
  // 1. Blank frame is flagged cleanly
  const blank = new Uint8Array(16 * 16 * 4);
  for (let i = 0; i < blank.length; i += 4) blank[i + 3] = 255;
  const blankRep = await evaluateRenderedFrameCoherence({ rgba: blank, width: 16, height: 16 });
  assert.equal(blankRep.coherent, false);
  assert.ok(blankRep.suggestions.length > 0);

  // 2. High-contrast geometric visual passes and caches SigLIP encoding via kappa
  const rich = new Uint8Array(16 * 16 * 4);
  for (let y = 0; y < 16; y++) {
    for (let x = 0; x < 16; x++) {
      const i = (y * 16 + x) * 4;
      const hi = (x + y) % 2 === 0;
      rich[i] = hi ? 16 : 240;
      rich[i + 1] = hi ? 185 : 248;
      rich[i + 2] = hi ? 129 : 255;
      rich[i + 3] = 255;
    }
  }

  const D = 8;
  const vWeights: VisionWeights = {
    config: {
      imageSize: 8,
      patchSize: 4,
      hiddenSize: D,
      intermediateSize: 16,
      numLayers: 1,
      numHeads: 2,
      projectionDim: 16,
      scaleFactor: 2,
      layerNormEps: 1e-6,
      useQuickGelu: false,
      imageMean: [0.5, 0.5, 0.5],
      imageStd: [0.5, 0.5, 0.5],
    },
    patchWeight: new Float32Array(D * 3 * 16).fill(0.05),
    patchBias: new Float32Array(D),
    posEmbd: new Float32Array(4 * D),
    layers: [
      {
        ln1W: new Float32Array(D).fill(1),
        ln1B: new Float32Array(D),
        qW: f32Slice(D, D),
        qB: new Float32Array(D),
        kW: f32Slice(D, D),
        kB: new Float32Array(D),
        vW: f32Slice(D, D),
        vB: new Float32Array(D),
        oW: f32Slice(D, D),
        oB: new Float32Array(D),
        ln2W: new Float32Array(D).fill(1),
        ln2B: new Float32Array(D),
        fc1W: f32Slice(16, D),
        fc1B: new Float32Array(16),
        fc2W: f32Slice(D, 16),
        fc2B: new Float32Array(D),
      },
    ],
    postLnW: new Float32Array(D).fill(1),
    postLnB: new Float32Array(D),
    connectorW1: f32Slice(16, D * 4),
    connectorB1: new Float32Array(16),
  };

  const encoder = new SiglipVisionEncoder(vWeights);
  const coord = new MultimodalCoherenceCoordinator();
  const pass1 = await evaluateRenderedFrameCoherence({ rgba: rich, width: 16, height: 16 }, { visionEncoder: encoder, coordinator: coord });
  const pass2 = await evaluateRenderedFrameCoherence({ rgba: rich, width: 16, height: 16 }, { visionEncoder: encoder, coordinator: coord });

  assert.equal(pass1.coherent, true);
  assert.equal(pass1.visionTokensEncoded, 1);
  assert.equal(pass1.visionCacheHit, false);
  assert.equal(pass2.visionCacheHit, true);

  // 3. Full perceiveImageFrame extracts dominant palette hex swatches + human summary + 0ms kappa cache
  const freshCoord = new MultimodalCoherenceCoordinator();
  const pA = await perceiveImageFrame('emerald-grid.png', { rgba: rich, width: 16, height: 16 }, { coordinator: freshCoord });
  const pB = await perceiveImageFrame('emerald-grid.png', { rgba: rich, width: 16, height: 16 }, { coordinator: freshCoord });
  assert.ok(pA.dominantColors.length >= 2);
  assert.ok(pA.dominantColors.every((c) => /^#[0-9a-f]{6}$/.test(c)));
  assert.ok(pA.summary.includes('emerald-grid.png'));
  assert.equal(pA.visionCacheHit, false);
  assert.equal(pB.visionCacheHit, true);
  assert.equal(freshCoord.getTelemetry().visualCacheHits, 1);
});
