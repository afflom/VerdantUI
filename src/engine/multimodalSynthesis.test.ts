import test from 'node:test';
import assert from 'node:assert/strict';
import {
  encodePromptToClipTokens,
  loadImageWeightsFromBacking,
  loadImageWeightsFromGguf,
  sinusoidalTimestepEmbedding,
  synthesizeDistilledImageAsync,
  synthesizeImageOneStep,
  type ClipTextWeights,
  type DistilledUnetWeights,
  type ImageSynthesisScratch,
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

function buildSyntheticSdxsGguf(tensors: readonly { readonly name: string; readonly dims: readonly number[]; readonly data: Float32Array }[]): Uint8Array {
  const enc = new TextEncoder();
  const headerChunks: Uint8Array[] = [];
  const pushU32 = (v: number) => { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, v, true); headerChunks.push(b); };
  const pushU64 = (v: number) => { const b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, BigInt(v), true); headerChunks.push(b); };
  const pushStr = (s: string) => { const bytes = enc.encode(s); pushU64(bytes.length); headerChunks.push(bytes); };

  headerChunks.push(enc.encode('GGUF'));
  pushU32(3);
  pushU64(tensors.length);
  pushU64(0); // kvCount=0 matching real sdxs-512-tinySDdistilled_Q8_0.gguf

  let dataOffsetRel = 0;
  for (const t of tensors) {
    pushStr(t.name);
    pushU32(t.dims.length);
    for (const d of t.dims) pushU64(d);
    pushU32(GgmlType.F32);
    pushU64(dataOffsetRel);
    const bytes = t.data.byteLength;
    dataOffsetRel += Math.ceil(bytes / 32) * 32;
  }

  const rawHeaderLen = headerChunks.reduce((a, c) => a + c.byteLength, 0);
  const alignedDataOffset = Math.ceil(rawHeaderLen / 32) * 32;
  const full = new Uint8Array(alignedDataOffset + dataOffsetRel);
  let pos = 0;
  for (const c of headerChunks) { full.set(c, pos); pos += c.byteLength; }

  let rel = 0;
  for (const t of tensors) {
    full.set(new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength), alignedDataOffset + rel);
    rel += Math.ceil(t.data.byteLength / 32) * 32;
  }
  return full;
}

test('loadImageWeightsFromBacking selectively range-reads SDXS+TAESD GGUF with lazy CLIP rows and reusable scratch parity', () => {
  const D = 8, V = 64, S = 16, C = 8, TC = 8;
  const fill = (n: number, k: number) => new Float32Array(n).map((_, i) => Math.sin((i + 1) * k) * 0.05);
  const gguf = buildSyntheticSdxsGguf([
    { name: 'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight', dims: [D, V], data: fill(D * V, 0.13) },
    { name: 'cond_stage_model.transformer.text_model.embeddings.position_embedding.weight', dims: [D, S], data: fill(D * S, 0.17) },
    { name: 'cond_stage_model.transformer.text_model.final_layer_norm.weight', dims: [D], data: new Float32Array(D).fill(1) },
    { name: 'cond_stage_model.transformer.text_model.final_layer_norm.bias', dims: [D], data: new Float32Array(D) },
    { name: 'cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.out_proj.weight', dims: [D, D], data: fill(D * D, 0.11) },
    { name: 'cond_stage_model.transformer.text_model.encoder.layers.0.self_attn.out_proj.bias', dims: [D], data: new Float32Array(D) },
    { name: 'model.diffusion_model.input_blocks.0.0.weight', dims: [3, 3, 4, C], data: fill(3 * 3 * 4 * C, 0.07) },
    { name: 'model.diffusion_model.input_blocks.0.0.bias', dims: [C], data: new Float32Array(C) },
    { name: 'model.diffusion_model.input_blocks.1.0.in_layers.0.weight', dims: [C], data: new Float32Array(C).fill(1) },
    { name: 'model.diffusion_model.input_blocks.1.0.in_layers.0.bias', dims: [C], data: new Float32Array(C) },
    { name: 'model.diffusion_model.input_blocks.1.0.in_layers.2.weight', dims: [3, 3, C, C], data: fill(3 * 3 * C * C, 0.09) },
    { name: 'model.diffusion_model.input_blocks.1.0.in_layers.2.bias', dims: [C], data: new Float32Array(C) },
    { name: 'model.diffusion_model.input_blocks.1.0.emb_layers.1.weight', dims: [D, C], data: fill(D * C, 0.12) },
    { name: 'model.diffusion_model.input_blocks.1.0.emb_layers.1.bias', dims: [C], data: new Float32Array(C) },
    { name: 'model.diffusion_model.input_blocks.1.0.out_layers.0.weight', dims: [C], data: new Float32Array(C).fill(1) },
    { name: 'model.diffusion_model.input_blocks.1.0.out_layers.0.bias', dims: [C], data: new Float32Array(C) },
    { name: 'model.diffusion_model.input_blocks.1.0.out_layers.3.weight', dims: [3, 3, C, C], data: fill(3 * 3 * C * C, 0.14) },
    { name: 'model.diffusion_model.input_blocks.1.0.out_layers.3.bias', dims: [C], data: new Float32Array(C) },
    { name: 'model.diffusion_model.out.2.weight', dims: [3, 3, C, 4], data: fill(3 * 3 * C * 4, 0.06) },
    { name: 'model.diffusion_model.out.2.bias', dims: [4], data: new Float32Array(4) },
    { name: 'first_stage_model.decoder.layers.0.weight', dims: [3, 3, 4, TC], data: fill(3 * 3 * 4 * TC, 0.08) },
    { name: 'first_stage_model.decoder.layers.0.bias', dims: [TC], data: new Float32Array(TC) },
    { name: 'first_stage_model.decoder.layers.6.weight', dims: [3, 3, TC, TC], data: fill(3 * 3 * TC * TC, 0.05) },
    { name: 'first_stage_model.decoder.layers.18.weight', dims: [3, 3, TC, 3], data: fill(3 * 3 * TC * 3, 0.09) },
    { name: 'first_stage_model.decoder.layers.18.bias', dims: [3], data: new Float32Array(3) },
  ]);

  let closed = false;
  let rangeReads = 0;
  const loaded = loadImageWeightsFromBacking({
    byteLength: gguf.byteLength,
    readRange(fileOffset, dst, dstOffset = 0, length = dst.byteLength - dstOffset) {
      rangeReads += 1;
      const avail = Math.max(0, Math.min(length, gguf.byteLength - fileOffset));
      if (avail > 0) dst.set(gguf.subarray(fileOffset, fileOffset + avail), dstOffset);
      return avail;
    },
    close() { closed = true; },
  });

  assert.equal(loaded.tensorCount, 25);
  assert.equal(loaded.clip.tokenEmbd, undefined, 'CLIP vocab must stay demand-paged via readTokenEmbeddingRow');
  assert.equal(typeof loaded.clip.readTokenEmbeddingRow, 'function');

  const tokens = encodePromptToClipTokens('emerald alpine sunrise', loaded.clip.vocabSize, loaded.clip.maxSeqLen);
  const scratch: ImageSynthesisScratch = {};
  const readsBefore = rangeReads;
  const img1 = synthesizeImageOneStep({ promptTokens: tokens, latentH: 4, latentW: 4, seed: 42, ...loaded, scratch });
  const img2 = synthesizeImageOneStep({ promptTokens: tokens, latentH: 4, latentW: 4, seed: 42, ...loaded, scratch });
  assert.ok(rangeReads > readsBefore, 'Expected lazy CLIP row reads during prompt encoding');
  assert.equal(img1.width, 8);
  assert.equal(img1.height, 8);
  assert.deepEqual(Array.from(img1.rgba), Array.from(img2.rgba), 'Reusable scratch must stay bit-exact deterministic');
  loaded.dispose?.();
  assert.equal(closed, true);

  // Negative checks: corrupted header and incomplete GGUF throw explicitly
  assert.throws(() => loadImageWeightsFromGguf(new Uint8Array([1, 2, 3, 4])));
  assert.throws(() => loadImageWeightsFromGguf(buildSyntheticSdxsGguf([])));
});

test('synthesizeDistilledImageAsync streams multi-layer CLIP, cross-attention UNet, and 3-stage TAESD across Lod0-Lod3 resolutions with bounded scratch', async () => {
  const D = 8, V = 64, S = 16, C = 8, TC = 4;
  const fill = (n: number, k: number) => new Float32Array(n).map((_, i) => Math.sin((i + 1) * k) * 0.04);
  const gguf = buildSyntheticSdxsGguf([
    { name: 'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight', dims: [D, V], data: fill(D * V, 0.13) },
    { name: 'cond_stage_model.transformer.text_model.embeddings.position_embedding.weight', dims: [D, S], data: fill(D * S, 0.17) },
    { name: 'cond_stage_model.transformer.text_model.final_layer_norm.weight', dims: [D], data: new Float32Array(D).fill(1) },
    { name: 'cond_stage_model.transformer.text_model.final_layer_norm.bias', dims: [D], data: new Float32Array(D) },
    ...[0, 1].flatMap((l) => [
      { name: `cond_stage_model.transformer.text_model.encoder.layers.${l}.layer_norm1.weight`, dims: [D], data: new Float32Array(D).fill(1) },
      { name: `cond_stage_model.transformer.text_model.encoder.layers.${l}.layer_norm1.bias`, dims: [D], data: new Float32Array(D) },
      { name: `cond_stage_model.transformer.text_model.encoder.layers.${l}.self_attn.q_proj.weight`, dims: [D, D], data: fill(D * D, 0.11 + l) },
      { name: `cond_stage_model.transformer.text_model.encoder.layers.${l}.self_attn.k_proj.weight`, dims: [D, D], data: fill(D * D, 0.12 + l) },
      { name: `cond_stage_model.transformer.text_model.encoder.layers.${l}.self_attn.v_proj.weight`, dims: [D, D], data: fill(D * D, 0.13 + l) },
      { name: `cond_stage_model.transformer.text_model.encoder.layers.${l}.self_attn.out_proj.weight`, dims: [D, D], data: fill(D * D, 0.14 + l) },
      { name: `cond_stage_model.transformer.text_model.encoder.layers.${l}.layer_norm2.weight`, dims: [D], data: new Float32Array(D).fill(1) },
      { name: `cond_stage_model.transformer.text_model.encoder.layers.${l}.layer_norm2.bias`, dims: [D], data: new Float32Array(D) },
      { name: `cond_stage_model.transformer.text_model.encoder.layers.${l}.mlp.fc1.weight`, dims: [D, D * 2], data: fill(D * D * 2, 0.15 + l) },
      { name: `cond_stage_model.transformer.text_model.encoder.layers.${l}.mlp.fc2.weight`, dims: [D * 2, D], data: fill(D * D * 2, 0.16 + l) },
    ]),
    { name: 'model.diffusion_model.input_blocks.0.0.weight', dims: [3, 3, 4, C], data: fill(3 * 3 * 4 * C, 0.07) },
    { name: 'model.diffusion_model.input_blocks.0.0.bias', dims: [C], data: new Float32Array(C) },
    { name: 'model.diffusion_model.input_blocks.1.0.in_layers.0.weight', dims: [C], data: new Float32Array(C).fill(1) },
    { name: 'model.diffusion_model.input_blocks.1.0.in_layers.0.bias', dims: [C], data: new Float32Array(C) },
    { name: 'model.diffusion_model.input_blocks.1.0.in_layers.2.weight', dims: [3, 3, C, C], data: fill(3 * 3 * C * C, 0.09) },
    { name: 'model.diffusion_model.input_blocks.1.0.in_layers.2.bias', dims: [C], data: new Float32Array(C) },
    { name: 'model.diffusion_model.input_blocks.1.0.emb_layers.1.weight', dims: [D, C], data: fill(D * C, 0.12) },
    { name: 'model.diffusion_model.input_blocks.1.0.emb_layers.1.bias', dims: [C], data: new Float32Array(C) },
    { name: 'model.diffusion_model.input_blocks.1.0.out_layers.0.weight', dims: [C], data: new Float32Array(C).fill(1) },
    { name: 'model.diffusion_model.input_blocks.1.0.out_layers.0.bias', dims: [C], data: new Float32Array(C) },
    { name: 'model.diffusion_model.input_blocks.1.0.out_layers.3.weight', dims: [3, 3, C, C], data: fill(3 * 3 * C * C, 0.14) },
    { name: 'model.diffusion_model.input_blocks.1.0.out_layers.3.bias', dims: [C], data: new Float32Array(C) },
    { name: 'model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_q.weight', dims: [C, C], data: fill(C * C, 0.21) },
    { name: 'model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight', dims: [D, C], data: fill(D * C, 0.22) },
    { name: 'model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_v.weight', dims: [D, C], data: fill(D * C, 0.23) },
    { name: 'model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_out.0.weight', dims: [C, C], data: fill(C * C, 0.24) },
    { name: 'model.diffusion_model.out.2.weight', dims: [3, 3, C, 4], data: fill(3 * 3 * C * 4, 0.06) },
    { name: 'model.diffusion_model.out.2.bias', dims: [4], data: new Float32Array(4) },
    { name: 'first_stage_model.decoder.layers.0.weight', dims: [3, 3, 4, TC], data: fill(3 * 3 * 4 * TC, 0.08) },
    { name: 'first_stage_model.decoder.layers.0.bias', dims: [TC], data: new Float32Array(TC) },
    { name: 'first_stage_model.decoder.layers.6.weight', dims: [3, 3, TC, TC], data: fill(3 * 3 * TC * TC, 0.05) },
    { name: 'first_stage_model.decoder.layers.11.weight', dims: [3, 3, TC, TC], data: fill(3 * 3 * TC * TC, 0.06) },
    { name: 'first_stage_model.decoder.layers.16.weight', dims: [3, 3, TC, TC], data: fill(3 * 3 * TC * TC, 0.07) },
    { name: 'first_stage_model.decoder.layers.18.weight', dims: [3, 3, TC, 3], data: fill(3 * 3 * TC * 3, 0.09) },
    { name: 'first_stage_model.decoder.layers.18.bias', dims: [3], data: new Float32Array(3) },
  ]);

  const tEmb = sinusoidalTimestepEmbedding(999, D);
  assert.equal(tEmb.length, D);
  assert.ok(tEmb.every((v) => Number.isFinite(v)));

  const loaded = loadImageWeightsFromGguf(gguf);
  const tokens = encodePromptToClipTokens('a golden dog sitting in sunlit grass', V, S);
  let yields = 0;

  // Verify Lod0CoarseProxy (16 -> 128x128) and Lod3LosslessSource (64 -> 512x512)
  const lod0 = await synthesizeDistilledImageAsync({
    promptTokens: tokens,
    latentH: 16,
    latentW: 16,
    seed: 42,
    loaded,
    yieldFrame: async () => { yields += 1; },
  });
  assert.equal(lod0.width, 128);
  assert.equal(lod0.height, 128);
  assert.equal(lod0.clipLayersExecuted, 2);
  assert.equal(lod0.crossAttnBlocksExecuted, 1);
  assert.ok(yields >= 4, 'Must yield cooperatively across CLIP, UNet cross-attention, and TAESD stages');
  assert.ok(lod0.peakScratchBytes < 64 * 1024 * 1024);

  const lod3 = await synthesizeDistilledImageAsync({
    promptTokens: tokens,
    latentH: 64,
    latentW: 64,
    seed: 42,
    loaded,
    yieldFrame: async () => { yields += 1; },
  });
  assert.equal(lod3.width, 512);
  assert.equal(lod3.height, 512);
  assert.equal(lod3.rgba.length, 512 * 512 * 4);
  assert.ok(lod3.peakScratchBytes < 64 * 1024 * 1024, 'Lod3 512x512 scratch must stay strictly under 64 MB');
});
