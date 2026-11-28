import test from 'node:test';
import assert from 'node:assert/strict';
import {
  encodeClipBpeTokens,
  encodePromptToClipTokens,
  loadImageWeightsFromBacking,
  loadImageWeightsFromGguf,
  parseClipBpeFromJsonAndMerges,
  sinusoidalTimestepEmbedding,
  synthesizeDistilledImageAsync,
  synthesizeImageOneStep,
  type ClipTextWeights,
  type DistilledUnetWeights,
  type ImageSynthesisScratch,
  type TaesdDecoderWeights,
} from './imageSynthesisEngine';
import { verdantConv2d } from './verdantKernels';
import { getSharedComputeWasmArena, verdantConv2dSimd } from './verdantWasmSimd';
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

function makeTestClipBpeVocab(vocabSize: number) {
  const tokens: Record<string, number> = {
    '<|startoftext|>': Math.min(49406, vocabSize - 2),
    '<|endoftext|>': Math.min(49407, vocabSize - 1),
  };
  let nextId = 1;
  for (const ch of 'abcdefghijklmnopqrstuvwxyz ') {
    if (nextId < vocabSize - 2) tokens[ch] = nextId++;
    if (nextId < vocabSize - 2) tokens[`${ch}</w>`] = nextId++;
  }
  return parseClipBpeFromJsonAndMerges(JSON.stringify(tokens), '#version: 0.2\n', 'opfs-cache');
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

  assert.throws(
    () => encodePromptToClipTokens('emerald alpine sunrise', loaded.clip.vocabSize, loaded.clip.maxSeqLen, null),
    /refused:.*CLIP BPE/i,
    'encodePromptToClipTokens must refuse explicitly when CLIP BPE vocab is missing',
  );
  const testVocab = makeTestClipBpeVocab(loaded.clip.vocabSize);
  const tokens = encodePromptToClipTokens('emerald alpine sunrise', loaded.clip.vocabSize, loaded.clip.maxSeqLen, testVocab);
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
  const tokens = encodePromptToClipTokens('a golden dog sitting in sunlit grass', V, S, makeTestClipBpeVocab(V));
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

function floatToF16Bits(val: number): number {
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = val;
  const x = u32[0]!;
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 127 + 15;
  let frac = (x >>> 13) & 0x03ff;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 0x7c00;
  return sign | (exp << 10) | frac;
}

test('verdantConv2dSimd matches scalar verdantConv2d for F32 and native F16 weights with 0 WASM arena leak', () => {
  const inC = 4;
  const outC = 8;
  const H = 6;
  const W = 6;
  const kH = 3;
  const kW = 3;
  const input = new Float32Array(inC * H * W);
  for (let i = 0; i < input.length; i++) input[i] = Math.sin(i * 0.37) * 0.5;
  const wF32 = new Float32Array(outC * inC * kH * kW);
  for (let i = 0; i < wF32.length; i++) wF32[i] = Math.cos(i * 0.19) * 0.25;
  const bias = new Float32Array(outC);
  for (let i = 0; i < outC; i++) bias[i] = (i - 3.5) * 0.05;

  const expected = new Float32Array(outC * H * W);
  verdantConv2d(input, wF32, bias, expected, inC, outC, H, W, kH, kW, 1, 1);

  const arena = getSharedComputeWasmArena();
  const usedBefore = arena?.usedBytes ?? 0;

  const outF32 = new Float32Array(outC * H * W);
  const resF32 = verdantConv2dSimd({
    input,
    weightRaw: new Uint8Array(wF32.buffer, wF32.byteOffset, wF32.byteLength),
    weightType: GgmlType.F32,
    bias,
    out: outF32,
    inC,
    outC,
    height: H,
    width: W,
    kH,
    kW,
    stride: 1,
    padding: 1,
    arena,
  });
  assert.equal(resF32.outH, H);
  assert.equal(resF32.outW, W);
  assert.equal(arena?.usedBytes ?? 0, usedBefore, 'withScratchScope must leak 0 bytes after F32 Conv2d');
  for (let i = 0; i < expected.length; i++) {
    assert.ok(Math.abs(outF32[i]! - expected[i]!) < 1e-4, `F32 mismatch at ${i}`);
  }

  // Native F16 weights (zero F16->F32 inflation)
  const wF16 = new Uint16Array(wF32.length);
  for (let i = 0; i < wF32.length; i++) wF16[i] = floatToF16Bits(wF32[i]!);
  const outF16 = new Float32Array(outC * H * W);
  verdantConv2dSimd({
    input,
    weightRaw: new Uint8Array(wF16.buffer, wF16.byteOffset, wF16.byteLength),
    weightType: GgmlType.F16,
    bias,
    out: outF16,
    inC,
    outC,
    height: H,
    width: W,
    kH,
    kW,
    stride: 1,
    padding: 1,
    arena,
  });
  assert.equal(arena?.usedBytes ?? 0, usedBefore, 'withScratchScope must leak 0 bytes after F16 Conv2d');
  for (let i = 0; i < expected.length; i++) {
    assert.ok(Math.abs(outF16[i]! - expected[i]!) < 2e-3, `F16 mismatch at ${i}`);
  }
});

test('encodeClipBpeTokens executes exact OpenAI CLIP byte-level BPE merges ("dog" -> 1929)', () => {
  const vocabJson = JSON.stringify({
    'a</w>': 320,
    'd': 67,
    'o': 78,
    'g</w>': 326,
    'do': 400,
    'dog</w>': 1929,
    'g': 70,
    'l': 75,
    'e': 68,
    'n</w>': 333,
    'go': 401,
    'gol': 402,
    'gold': 403,
    'golde': 404,
    'golden</w>': 3878,
    '<|startoftext|>': 49406,
    '<|endoftext|>': 49407,
  });
  const mergesText = [
    '#version: 0.2',
    'd o',
    'do g</w>',
    'g o',
    'go l',
    'gol d',
    'gold e',
    'golde n</w>',
  ].join('\n');

  const vocab = parseClipBpeFromJsonAndMerges(vocabJson, mergesText, 'custom');
  const tokens = encodeClipBpeTokens('A Golden Dog', vocab, 77);
  assert.deepEqual(tokens, [49406, 320, 3878, 1929, 49407]);
});

test('Exact verified SDXS-512 UNet topology (input_blocks {0,1,3,4,6,7} -> output_blocks {0..7} + up_blocks.0.attentions) + 10 TAESD residual blocks executes end-to-end', async () => {
  const V = 32;
  const D = 8;
  const S = 16;
  const C0 = 4; // scaled stand-in for 320
  const C1 = 8; // scaled stand-in for 640
  const C2 = 16; // scaled stand-in for 1280
  const TE = 16; // scaled stand-in for time_embed 1280
  const TC = 4; // scaled stand-in for TAESD 64
  const fill = (len: number, seed: number) => {
    const out = new Float32Array(len);
    for (let i = 0; i < len; i++) out[i] = Math.sin(i * 0.29 + seed) * 0.04;
    return out;
  };
  const ones = (len: number) => new Float32Array(len).fill(1);
  const zeros = (len: number) => new Float32Array(len);

  const resBlock = (prefix: string, inC: number, outC: number, s: number) => [
    { name: `${prefix}.in_layers.0.weight`, dims: [inC], data: ones(inC) },
    { name: `${prefix}.in_layers.0.bias`, dims: [inC], data: zeros(inC) },
    { name: `${prefix}.in_layers.2.weight`, dims: [3, 3, inC, outC], data: fill(9 * inC * outC, s) },
    { name: `${prefix}.in_layers.2.bias`, dims: [outC], data: zeros(outC) },
    { name: `${prefix}.emb_layers.1.weight`, dims: [TE, outC], data: fill(TE * outC, s + 0.1) },
    { name: `${prefix}.emb_layers.1.bias`, dims: [outC], data: zeros(outC) },
    { name: `${prefix}.out_layers.0.weight`, dims: [outC], data: ones(outC) },
    { name: `${prefix}.out_layers.0.bias`, dims: [outC], data: zeros(outC) },
    { name: `${prefix}.out_layers.3.weight`, dims: [3, 3, outC, outC], data: fill(9 * outC * outC, s + 0.2) },
    { name: `${prefix}.out_layers.3.bias`, dims: [outC], data: zeros(outC) },
    ...(inC !== outC
      ? [
          { name: `${prefix}.skip_connection.weight`, dims: [1, 1, inC, outC], data: fill(inC * outC, s + 0.3) },
          { name: `${prefix}.skip_connection.bias`, dims: [outC], data: zeros(outC) },
        ]
      : []),
  ];

  const spatialAttn = (prefix: string, ch: number, s: number) => {
    const tb = `${prefix}.transformer_blocks.0`;
    return [
      { name: `${prefix}.norm.weight`, dims: [ch], data: ones(ch) },
      { name: `${prefix}.norm.bias`, dims: [ch], data: zeros(ch) },
      { name: `${prefix}.proj_in.weight`, dims: [1, 1, ch, ch], data: fill(ch * ch, s) },
      { name: `${prefix}.proj_in.bias`, dims: [ch], data: zeros(ch) },
      { name: `${tb}.norm1.weight`, dims: [ch], data: ones(ch) },
      { name: `${tb}.norm1.bias`, dims: [ch], data: zeros(ch) },
      { name: `${tb}.attn1.to_q.weight`, dims: [ch, ch], data: fill(ch * ch, s + 0.1) },
      { name: `${tb}.attn1.to_k.weight`, dims: [ch, ch], data: fill(ch * ch, s + 0.2) },
      { name: `${tb}.attn1.to_v.weight`, dims: [ch, ch], data: fill(ch * ch, s + 0.3) },
      { name: `${tb}.attn1.to_out.0.weight`, dims: [ch, ch], data: fill(ch * ch, s + 0.4) },
      { name: `${tb}.norm2.weight`, dims: [ch], data: ones(ch) },
      { name: `${tb}.norm2.bias`, dims: [ch], data: zeros(ch) },
      { name: `${tb}.attn2.to_q.weight`, dims: [ch, ch], data: fill(ch * ch, s + 0.5) },
      { name: `${tb}.attn2.to_k.weight`, dims: [D, ch], data: fill(D * ch, s + 0.6) },
      { name: `${tb}.attn2.to_v.weight`, dims: [D, ch], data: fill(D * ch, s + 0.7) },
      { name: `${tb}.attn2.to_out.0.weight`, dims: [ch, ch], data: fill(ch * ch, s + 0.8) },
      { name: `${tb}.norm3.weight`, dims: [ch], data: ones(ch) },
      { name: `${tb}.norm3.bias`, dims: [ch], data: zeros(ch) },
      { name: `${tb}.ff.net.0.proj.weight`, dims: [ch, ch * 2], data: fill(ch * ch * 2, s + 0.9) },
      { name: `${tb}.ff.net.2.weight`, dims: [ch, ch], data: fill(ch * ch, s + 1.0) },
      { name: `${prefix}.proj_out.weight`, dims: [1, 1, ch, ch], data: fill(ch * ch, s + 1.1) },
      { name: `${prefix}.proj_out.bias`, dims: [ch], data: zeros(ch) },
    ];
  };

  const taesdRes = (idx: number, s: number) => [
    { name: `first_stage_model.decoder.layers.${idx}.conv.0.weight`, dims: [3, 3, TC, TC], data: fill(9 * TC * TC, s) },
    { name: `first_stage_model.decoder.layers.${idx}.conv.0.bias`, dims: [TC], data: zeros(TC) },
    { name: `first_stage_model.decoder.layers.${idx}.conv.2.weight`, dims: [3, 3, TC, TC], data: fill(9 * TC * TC, s + 0.1) },
    { name: `first_stage_model.decoder.layers.${idx}.conv.2.bias`, dims: [TC], data: zeros(TC) },
    { name: `first_stage_model.decoder.layers.${idx}.conv.4.weight`, dims: [3, 3, TC, TC], data: fill(9 * TC * TC, s + 0.2) },
    { name: `first_stage_model.decoder.layers.${idx}.conv.4.bias`, dims: [TC], data: zeros(TC) },
  ];

  const gguf = buildSyntheticSdxsGguf([
    { name: 'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight', dims: [D, V], data: fill(D * V, 1) },
    { name: 'cond_stage_model.transformer.text_model.embeddings.position_embedding.weight', dims: [D, S], data: fill(D * S, 2) },
    { name: 'cond_stage_model.transformer.text_model.final_layer_norm.weight', dims: [D], data: ones(D) },
    { name: 'cond_stage_model.transformer.text_model.final_layer_norm.bias', dims: [D], data: zeros(D) },
    { name: 'model.diffusion_model.time_embed.0.weight', dims: [C0, TE], data: fill(C0 * TE, 3) },
    { name: 'model.diffusion_model.time_embed.0.bias', dims: [TE], data: zeros(TE) },
    { name: 'model.diffusion_model.time_embed.2.weight', dims: [TE, TE], data: fill(TE * TE, 4) },
    { name: 'model.diffusion_model.time_embed.2.bias', dims: [TE], data: zeros(TE) },
    // Exact live input_blocks: {0, 1, 3, 4, 6, 7} (NO blocks 2 or 5, NO middle_block!)
    { name: 'model.diffusion_model.input_blocks.0.0.weight', dims: [3, 3, 4, C0], data: fill(36 * C0, 5) },
    { name: 'model.diffusion_model.input_blocks.0.0.bias', dims: [C0], data: zeros(C0) },
    ...resBlock('model.diffusion_model.input_blocks.1.0', C0, C0, 6),
    { name: 'model.diffusion_model.input_blocks.3.0.op.weight', dims: [3, 3, C0, C0], data: fill(9 * C0 * C0, 7) },
    { name: 'model.diffusion_model.input_blocks.3.0.op.bias', dims: [C0], data: zeros(C0) },
    ...resBlock('model.diffusion_model.input_blocks.4.0', C0, C1, 8),
    ...spatialAttn('model.diffusion_model.input_blocks.4.1', C1, 9),
    { name: 'model.diffusion_model.input_blocks.6.0.op.weight', dims: [3, 3, C1, C1], data: fill(9 * C1 * C1, 10) },
    { name: 'model.diffusion_model.input_blocks.6.0.op.bias', dims: [C1], data: zeros(C1) },
    ...resBlock('model.diffusion_model.input_blocks.7.0', C1, C2, 11),
    ...spatialAttn('model.diffusion_model.input_blocks.7.1', C2, 12),
    // Exact live output_blocks: {0..7} + up_blocks.0.attentions.{0,1}
    ...resBlock('model.diffusion_model.output_blocks.0.0', C2 + C2, C2, 13),
    ...spatialAttn('model.diffusion_model.up_blocks.0.attentions.0', C2, 14),
    ...resBlock('model.diffusion_model.output_blocks.1.0', C2 + C1, C2, 15),
    ...spatialAttn('model.diffusion_model.up_blocks.0.attentions.1', C2, 16),
    { name: 'model.diffusion_model.output_blocks.2.1.conv.weight', dims: [3, 3, C2, C2], data: fill(9 * C2 * C2, 17) },
    { name: 'model.diffusion_model.output_blocks.2.1.conv.bias', dims: [C2], data: zeros(C2) },
    ...resBlock('model.diffusion_model.output_blocks.3.0', C2 + C1, C1, 18),
    ...spatialAttn('model.diffusion_model.output_blocks.3.1', C1, 19),
    ...resBlock('model.diffusion_model.output_blocks.4.0', C1 + C0, C1, 20),
    ...spatialAttn('model.diffusion_model.output_blocks.4.1', C1, 21),
    { name: 'model.diffusion_model.output_blocks.5.2.conv.weight', dims: [3, 3, C1, C1], data: fill(9 * C1 * C1, 22) },
    { name: 'model.diffusion_model.output_blocks.5.2.conv.bias', dims: [C1], data: zeros(C1) },
    ...resBlock('model.diffusion_model.output_blocks.6.0', C1 + C0, C0, 23),
    ...resBlock('model.diffusion_model.output_blocks.7.0', C0 + C0, C0, 24),
    { name: 'model.diffusion_model.out.0.weight', dims: [C0], data: ones(C0) },
    { name: 'model.diffusion_model.out.0.bias', dims: [C0], data: zeros(C0) },
    { name: 'model.diffusion_model.out.2.weight', dims: [3, 3, C0, 4], data: fill(36 * C0, 25) },
    { name: 'model.diffusion_model.out.2.bias', dims: [4], data: zeros(4) },
    // Exact 19-layer TAESD decoder with all 10 residual blocks
    { name: 'first_stage_model.decoder.layers.0.weight', dims: [3, 3, 4, TC], data: fill(36 * TC, 26) },
    { name: 'first_stage_model.decoder.layers.0.bias', dims: [TC], data: zeros(TC) },
    ...taesdRes(2, 27),
    ...taesdRes(3, 28),
    ...taesdRes(4, 29),
    { name: 'first_stage_model.decoder.layers.6.weight', dims: [3, 3, TC, TC], data: fill(9 * TC * TC, 30) },
    ...taesdRes(7, 31),
    ...taesdRes(8, 32),
    ...taesdRes(9, 33),
    { name: 'first_stage_model.decoder.layers.11.weight', dims: [3, 3, TC, TC], data: fill(9 * TC * TC, 34) },
    ...taesdRes(12, 35),
    ...taesdRes(13, 36),
    ...taesdRes(14, 37),
    { name: 'first_stage_model.decoder.layers.16.weight', dims: [3, 3, TC, TC], data: fill(9 * TC * TC, 38) },
    ...taesdRes(17, 39),
    { name: 'first_stage_model.decoder.layers.18.weight', dims: [3, 3, TC, 3], data: fill(27 * TC, 40) },
    { name: 'first_stage_model.decoder.layers.18.bias', dims: [3], data: new Float32Array([0.5, 0.5, 0.5]) },
  ]);

  const loaded = loadImageWeightsFromGguf(gguf);
  assert.equal(loaded.streamGraph?.hasMiddleBlock, false);
  assert.deepEqual(loaded.streamGraph?.inputBlockIndices, [0, 1, 3, 4, 6, 7]);
  assert.deepEqual(loaded.streamGraph?.outputBlockIndices, [0, 1, 2, 3, 4, 5, 6, 7]);
  assert.equal(loaded.streamGraph?.taesdResidualIndices.length, 10);

  const tokens = encodePromptToClipTokens('a golden dog', V, S, makeTestClipBpeVocab(V));
  const res = await synthesizeDistilledImageAsync({
    promptTokens: tokens,
    latentH: 8,
    latentW: 8,
    seed: 7,
    loaded,
  });

  assert.equal(res.width, 64);
  assert.equal(res.height, 64);
  assert.equal(res.crossAttnBlocksExecuted, 6, 'All 6 SpatialTransformer cross-attention blocks must execute');
  assert.equal(res.taesdResidualBlocksExecuted, 10, 'All 10 TAESD residual blocks must execute');
  assert.ok(res.unetBlocksExecuted >= 14, 'All encoder/decoder UNet blocks must execute');
  assert.ok(res.peakScratchBytes < 64 * 1024 * 1024);
  assert.equal(res.rgba.length, 64 * 64 * 4);
});
