import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { GgmlType } from './gguf';
import { LlamaVTensorRuntime } from './llamaRuntime';
import {
  verdantConv1d,
  verdantConv2d,
  verdantGelu,
  verdantGroupNorm,
  verdantLayerNorm,
  verdantSiLu,
} from './verdantKernels';
import {
  preprocessRgbaToChw,
  SiglipVisionEncoder,
  type TensorSlice,
  type VisionWeights,
} from './visionEncoder';
import {
  computeLogMelSpectrogram,
  resampleLinearMono,
  WHISPER_SAMPLE_RATE,
} from './audioFrontend';
import {
  WHISPER_GGML_MAGIC,
  WhisperAudioEngine,
  loadWhisperWeightsFromGgml,
  type DenseLinear,
  type WhisperWeights,
} from './whisperEngine';
import {
  computeMediaKappa,
  FrameBudgetOracle,
  MultimodalCoherenceCoordinator,
} from './multimodalCoordinator';

function f32Slice(rows: number, cols: number, fill = 0.05): TensorSlice {
  const arr = new Float32Array(rows * cols);
  for (let i = 0; i < arr.length; i++) arr[i] = Math.sin(i * 0.31) * fill;
  return {
    bytes: new Uint8Array(arr.buffer),
    offset: 0,
    type: GgmlType.F32,
    rows,
    cols,
  };
}

function makeDense(outDim: number, inDim: number, scale = 0.05): DenseLinear {
  const weight = new Float32Array(outDim * inDim);
  for (let i = 0; i < weight.length; i++) weight[i] = Math.cos(i * 0.17) * scale;
  return { weight, bias: new Float32Array(outDim), inDim, outDim };
}

test('Multimodal vTensor kernels (LayerNorm, GELU, SiLU, GroupNorm, Conv1D, Conv2D) match exact math', () => {
  // LayerNorm: mean 0, unit variance
  const x = new Float32Array([1, 2, 3, 4]);
  const lnOut = new Float32Array(4);
  verdantLayerNorm(x, null, null, lnOut, 4, 1e-6);
  const mean = (lnOut[0]! + lnOut[1]! + lnOut[2]! + lnOut[3]!) / 4;
  assert.ok(Math.abs(mean) < 1e-5);
  assert.ok(lnOut[0]! < 0 && lnOut[3]! > 0);

  // GELU & SiLU
  const actIn = new Float32Array([-2, 0, 2]);
  const geluOut = new Float32Array(3);
  const siluOut = new Float32Array(3);
  verdantGelu(actIn, geluOut, 3);
  verdantSiLu(actIn, siluOut, 3);
  assert.equal(geluOut[1], 0);
  assert.ok(geluOut[2]! > 1.9 && geluOut[2]! < 2.0);
  assert.equal(siluOut[1], 0);
  assert.ok(siluOut[2]! > 1.7 && siluOut[2]! < 1.8);

  // GroupNorm over [4, 2] with 2 groups
  const gnIn = new Float32Array([1, 3, 5, 7, 10, 20, 30, 40]);
  const gnOut = new Float32Array(8);
  verdantGroupNorm(gnIn, null, null, gnOut, 4, 2, 2, 1e-6);
  const g0Mean = (gnOut[0]! + gnOut[1]! + gnOut[2]! + gnOut[3]!) / 4;
  assert.ok(Math.abs(g0Mean) < 1e-5);

  // Conv1D identity kernel [1, 1, 3] = [0, 1, 0] with pad=1
  const c1In = new Float32Array([2, 4, 6, 8]);
  const c1W = new Float32Array([0, 1, 0]);
  const c1Out = new Float32Array(4);
  const c1Len = verdantConv1d(c1In, c1W, null, c1Out, 1, 1, 4, 3, 1, 1);
  assert.equal(c1Len, 4);
  assert.deepEqual(Array.from(c1Out), [2, 4, 6, 8]);

  // Conv2D 2x2 stride-2 sum filter over 4x4 image
  const c2In = new Float32Array(16).fill(1);
  const c2W = new Float32Array([1, 1, 1, 1]);
  const c2Out = new Float32Array(4);
  const { outH, outW } = verdantConv2d(c2In, c2W, null, c2Out, 1, 1, 4, 4, 2, 2, 2, 0);
  assert.equal(outH, 2);
  assert.equal(outW, 2);
  assert.deepEqual(Array.from(c2Out), [4, 4, 4, 4]);
});

test('LlamaVTensorRuntime.prefillEmbeddingSequence produces BIT-EXACT logits matching evalToken on real stories260K.gguf', () => {
  const modelPath = path.resolve(process.cwd(), 'public/models/stories260K.gguf');
  const bytes = new Uint8Array(fs.readFileSync(modelPath));

  const rtA = new LlamaVTensorRuntime();
  rtA.loadFromBuffer('stories260k', 'Stories 260K', 'local', bytes);
  const tokens = [1, 15, 42, 7];
  let logitsA = new Float32Array(0);
  for (let i = 0; i < tokens.length; i++) {
    logitsA = rtA.evalToken(tokens[i]!, i, false, i === tokens.length - 1).slice();
  }

  const rtB = new LlamaVTensorRuntime();
  rtB.loadFromBuffer('stories260k', 'Stories 260K', 'local', bytes);
  const embds = tokens.map((t) => rtB.decodeTokenEmbedding(t));
  const resB = rtB.prefillEmbeddingSequence(embds, 0, true);
  assert.equal(resB.nextPos, tokens.length);
  assert.ok(resB.logits);

  for (let i = 0; i < logitsA.length; i++) {
    assert.equal(resB.logits[i], logitsA[i], `Bit-exact mismatch at logit ${i}`);
  }
});

test('SiglipVisionEncoder encodes real RGBA frames & video clips into projected embeddings that inject straight into LlamaVTensorRuntime', async () => {
  // Compact SigLIP-compatible projector matching stories260K embedding dim (64)
  const D = 16;
  const imgSize = 8;
  const patchSize = 4; // 2x2 = 4 patches
  const sf = 2; // pixel-shuffle merges 2x2 patches -> 1 visual token of dim D*4 = 64 -> projects to 64!
  const weights: VisionWeights = {
    config: {
      imageSize: imgSize,
      patchSize,
      hiddenSize: D,
      intermediateSize: 32,
      numLayers: 2,
      numHeads: 4,
      projectionDim: 64,
      scaleFactor: sf,
      layerNormEps: 1e-6,
      useQuickGelu: false,
      imageMean: [0.5, 0.5, 0.5],
      imageStd: [0.5, 0.5, 0.5],
    },
    patchWeight: new Float32Array(D * 3 * patchSize * patchSize).fill(0.05),
    patchBias: new Float32Array(D),
    posEmbd: new Float32Array(4 * D),
    layers: [0, 1].map(() => ({
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
      fc1W: f32Slice(32, D),
      fc1B: new Float32Array(32),
      fc2W: f32Slice(D, 32),
      fc2B: new Float32Array(D),
    })),
    postLnW: new Float32Array(D).fill(1),
    postLnB: new Float32Array(D),
    connectorW1: f32Slice(64, D * sf * sf),
    connectorB1: new Float32Array(64),
  };

  const encoder = new SiglipVisionEncoder(weights);
  const rgba = new Uint8Array(16 * 16 * 4);
  for (let i = 0; i < rgba.length; i += 4) {
    rgba[i] = (i * 7) & 255;
    rgba[i + 1] = (i * 13) & 255;
    rgba[i + 2] = (i * 19) & 255;
    rgba[i + 3] = 255;
  }

  const chw = preprocessRgbaToChw(rgba, 16, 16, imgSize);
  assert.equal(chw.length, 3 * imgSize * imgSize);

  const imgTokens = encoder.encodeFrame({ rgba, width: 16, height: 16 });
  assert.equal(imgTokens.length, 1);
  assert.equal(imgTokens[0]!.length, 64);
  assert.ok(imgTokens[0]!.every((v) => Number.isFinite(v)));

  const vidTokens = encoder.encodeVideoFrames([
    { rgba, width: 16, height: 16 },
    { rgba, width: 16, height: 16 },
  ]);
  assert.equal(vidTokens.length, 2);

  // Inject real visual tokens straight into real LlamaVTensorRuntime KV cache AND run full generate()!
  const modelPath = path.resolve(process.cwd(), 'public/models/stories260K.gguf');
  const rt = new LlamaVTensorRuntime();
  rt.loadFromBuffer('stories260k', 'Stories 260K', 'local', new Uint8Array(fs.readFileSync(modelPath)));
  const injected = rt.prefillEmbeddingSequence(vidTokens, 0, true);
  assert.equal(injected.nextPos, 2);
  assert.ok(injected.logits && injected.logits.every((v) => Number.isFinite(v)));

  let generatedWithVision = '';
  for await (const ev of rt.generate(
    'Once upon a time',
    { temperature: 0, topP: 1, topK: 1, repeatPenalty: 1.0, maxTokens: 6, seed: 1 },
    undefined,
    undefined,
    'vision-thread',
    vidTokens,
  )) {
    generatedWithVision = ev.accumulatedText;
  }
  assert.ok(generatedWithVision.length > 0);
});

test('Audio frontend Log-Mel spectrogram distinguishes frequencies and WhisperAudioEngine runs real encoder-decoder pass', () => {
  // Generate 0.5s sine tones at 440 Hz vs 2400 Hz and verify Mel energy peak shifts cleanly upward
  const makeTone = (hz: number) => {
    const pcm = new Float32Array(WHISPER_SAMPLE_RATE / 2);
    for (let i = 0; i < pcm.length; i++) pcm[i] = Math.sin((2 * Math.PI * hz * i) / WHISPER_SAMPLE_RATE);
    return pcm;
  };
  const melLow = computeLogMelSpectrogram(makeTone(440));
  const melHigh = computeLogMelSpectrogram(makeTone(2400));
  assert.equal(melLow.nMels, 80);
  assert.ok(melLow.nFrames > 20);

  const peakMelBin = (mel: { nMels: number; nFrames: number; data: Float32Array }) => {
    let bestBin = 0;
    let bestEnergy = -Infinity;
    for (let m = 0; m < mel.nMels; m++) {
      let sum = 0;
      for (let t = 0; t < mel.nFrames; t++) sum += mel.data[m * mel.nFrames + t]!;
      if (sum > bestEnergy) {
        bestEnergy = sum;
        bestBin = m;
      }
    }
    return bestBin;
  };
  assert.ok(peakMelBin(melHigh) > peakMelBin(melLow) + 15);

  const resampled = resampleLinearMono(new Float32Array(4800), 48000, 16000);
  assert.equal(resampled.length, 1600);

  // Full Whisper Conv1D + Encoder + Cross-Attention Decoder forward pass
  const D = 16;
  const nVocab = 8;
  const tokEmbd = new Float32Array(nVocab * D);
  for (let i = 0; i < tokEmbd.length; i++) tokEmbd[i] = Math.sin(i * 0.4) * 0.1;
  // Bias EOT token (id=1) cleanly so greedy decode terminates deterministically
  for (let d = 0; d < D; d++) tokEmbd[1 * D + d] = 1.0;

  const wWeights: WhisperWeights = {
    config: {
      nMels: 80,
      nAudioCtx: 64,
      nAudioState: D,
      nAudioHead: 4,
      nAudioLayer: 1,
      nTextCtx: 16,
      nTextState: D,
      nTextHead: 4,
      nTextLayer: 1,
      nVocab,
      sotTokenId: 0,
      eotTokenId: 1,
    },
    conv1W: new Float32Array(D * 80 * 3).fill(0.01),
    conv1B: new Float32Array(D),
    conv2W: new Float32Array(D * D * 3).fill(0.01),
    conv2B: new Float32Array(D),
    encPosEmbd: new Float32Array(64 * D),
    encoderLayers: [
      {
        attnLnW: new Float32Array(D).fill(1),
        attnLnB: new Float32Array(D),
        q: makeDense(D, D),
        k: makeDense(D, D),
        v: makeDense(D, D),
        o: makeDense(D, D),
        mlpLnW: new Float32Array(D).fill(1),
        mlpLnB: new Float32Array(D),
        fc1: makeDense(32, D),
        fc2: makeDense(D, 32),
      },
    ],
    encLnPostW: new Float32Array(D).fill(1),
    encLnPostB: new Float32Array(D),
    tokEmbd,
    decPosEmbd: new Float32Array(16 * D),
    decoderLayers: [
      {
        selfLnW: new Float32Array(D).fill(1),
        selfLnB: new Float32Array(D),
        selfQ: makeDense(D, D),
        selfK: makeDense(D, D),
        selfV: makeDense(D, D),
        selfO: makeDense(D, D),
        crossLnW: new Float32Array(D).fill(1),
        crossLnB: new Float32Array(D),
        crossQ: makeDense(D, D),
        crossK: makeDense(D, D),
        crossV: makeDense(D, D),
        crossO: makeDense(D, D),
        mlpLnW: new Float32Array(D).fill(1),
        mlpLnB: new Float32Array(D),
        fc1: makeDense(32, D),
        fc2: makeDense(D, 32),
      },
    ],
    decLnW: new Float32Array(D).fill(1),
    decLnB: new Float32Array(D),
    vocab: ['<sot>', '<eot>', 'hello', ' world', ' verdant', ' audio', ' test', '.'],
  };

  const whisper = new WhisperAudioEngine(wWeights);
  const res = whisper.transcribe(melLow, 4);
  assert.ok(res.encoderFrames > 0);
  assert.ok(res.tokenIds.length >= 1);
});

test('MultimodalCoherenceCoordinator enforces kappa 0ms cache reuse, sparse modality eviction, and 64-frame oracle stability', async () => {
  const coord = new MultimodalCoherenceCoordinator(150 * 1024 * 1024); // 150 MB aux cap
  let visionEvicted = false;

  coord.activateModality({
    modality: 'vision',
    modelId: 'smolvlm2-mmproj',
    residentBytes: 104 * 1024 * 1024,
    evict: () => {
      visionEvicted = true;
    },
  });
  // Activating audio (60 MB) pushes aux total to 164 MB > 150 MB -> evicts idle vision cleanly!
  coord.activateModality({
    modality: 'audio',
    modelId: 'whisper-tiny',
    residentBytes: 60 * 1024 * 1024,
    evict: () => {},
  });
  assert.equal(visionEvicted, true);
  assert.equal(coord.getTelemetry().modalityEvictions, 1);

  // Content-addressed kappa cache returns instant hit on repeat visual frame
  const pixels = new Uint8Array(64 * 64 * 4).fill(128);
  const kappa = computeMediaKappa('visual', pixels, '64x64');
  let encodeCalls = 0;
  const first = await coord.getOrEncodeVisual(kappa, async () => {
    encodeCalls += 1;
    return { width: 64, height: 64, tokenCount: 1, embeddingDim: 64, embeddings: [new Float32Array(64)], encodeMs: 12 };
  });
  const second = await coord.getOrEncodeVisual(kappa, async () => {
    encodeCalls += 1;
    return { width: 64, height: 64, tokenCount: 1, embeddingDim: 64, embeddings: [new Float32Array(64)], encodeMs: 12 };
  });
  assert.equal(encodeCalls, 1);
  assert.equal(first.cacheHit, false);
  assert.equal(second.cacheHit, true);

  // 64-frame FrameBudgetOracle adapts compute slice smoothly under heavy frames
  const oracle = new FrameBudgetOracle(16_666);
  const initialSlice = oracle.report().recommendedSliceUs;
  for (let i = 0; i < 32; i++) oracle.recordFrame(24_000);
  const slowed = oracle.report();
  assert.ok(slowed.recommendedSliceUs < initialSlice);
  assert.equal(slowed.stable, false);
  for (let i = 0; i < 64; i++) oracle.recordFrame(6_000);
  const recovered = oracle.report();
  assert.equal(recovered.stable, true);
  assert.ok(recovered.recommendedSliceUs > slowed.recommendedSliceUs);
});

test('loadWhisperWeightsFromGgml parses real 0x67676d6c whisper.cpp binary format and runs transcription', () => {
  const D = 8;
  const nMels = 80;
  const nAudioCtx = 16;
  const nTextCtx = 8;
  const nVocab = 6;
  const vocab = ['<sot>', '<eot>', 'hello', ' verdant', ' world', '.'];
  const enc = new TextEncoder();

  const chunks: Uint8Array[] = [];
  const pushI32 = (v: number) => {
    const b = new Uint8Array(4);
    new DataView(b.buffer).setInt32(0, v, true);
    chunks.push(b);
  };
  const pushU32 = (v: number) => {
    const b = new Uint8Array(4);
    new DataView(b.buffer).setUint32(0, v, true);
    chunks.push(b);
  };
  const pushTensor = (name: string, dims: number[], data: Float32Array) => {
    const nameBytes = enc.encode(name);
    pushI32(dims.length);
    pushI32(nameBytes.length);
    pushI32(GgmlType.F32);
    for (const d of dims) pushI32(d);
    chunks.push(nameBytes);
    chunks.push(new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)));
  };

  pushU32(WHISPER_GGML_MAGIC);
  for (const hp of [nVocab, nAudioCtx, D, 2, 1, nTextCtx, D, 2, 1, nMels, 0]) pushI32(hp);
  // Empty 0x0 mel filters section
  pushI32(0);
  pushI32(0);
  // Vocab section
  pushI32(vocab.length);
  for (const tok of vocab) {
    const tb = enc.encode(tok);
    pushI32(tb.length);
    chunks.push(tb);
  }

  const ones = (n: number) => new Float32Array(n).fill(1);
  const zeros = (n: number) => new Float32Array(n);
  const small = (n: number) => new Float32Array(n).map((_, i) => Math.sin(i * 0.3) * 0.02);

  pushTensor('encoder.conv1.weight', [3, nMels, D], small(3 * nMels * D));
  pushTensor('encoder.conv1.bias', [D], zeros(D));
  pushTensor('encoder.conv2.weight', [3, D, D], small(3 * D * D));
  pushTensor('encoder.conv2.bias', [D], zeros(D));
  pushTensor('encoder.positional_embedding', [D, nAudioCtx], zeros(D * nAudioCtx));
  pushTensor('encoder.blocks.0.attn_ln.weight', [D], ones(D));
  pushTensor('encoder.blocks.0.attn_ln.bias', [D], zeros(D));
  pushTensor('encoder.blocks.0.attn.query.weight', [D, D], small(D * D));
  pushTensor('encoder.blocks.0.attn.query.bias', [D], zeros(D));
  pushTensor('encoder.blocks.0.attn.key.weight', [D, D], small(D * D));
  pushTensor('encoder.blocks.0.attn.value.weight', [D, D], small(D * D));
  pushTensor('encoder.blocks.0.attn.value.bias', [D], zeros(D));
  pushTensor('encoder.blocks.0.attn.out.weight', [D, D], small(D * D));
  pushTensor('encoder.blocks.0.attn.out.bias', [D], zeros(D));
  pushTensor('encoder.blocks.0.mlp_ln.weight', [D], ones(D));
  pushTensor('encoder.blocks.0.mlp_ln.bias', [D], zeros(D));
  pushTensor('encoder.blocks.0.mlp.0.weight', [D, 16], small(D * 16));
  pushTensor('encoder.blocks.0.mlp.0.bias', [16], zeros(16));
  pushTensor('encoder.blocks.0.mlp.2.weight', [16, D], small(16 * D));
  pushTensor('encoder.blocks.0.mlp.2.bias', [D], zeros(D));
  pushTensor('encoder.ln_post.weight', [D], ones(D));
  pushTensor('encoder.ln_post.bias', [D], zeros(D));

  const tokEmbd = small(nVocab * D);
  for (let d = 0; d < D; d++) tokEmbd[1 * D + d] = 1.0; // bias <eot> cleanly
  pushTensor('decoder.token_embedding', [D, nVocab], tokEmbd);
  pushTensor('decoder.positional_embedding', [D, nTextCtx], zeros(D * nTextCtx));
  pushTensor('decoder.blocks.0.attn_ln.weight', [D], ones(D));
  pushTensor('decoder.blocks.0.attn_ln.bias', [D], zeros(D));
  pushTensor('decoder.blocks.0.attn.query.weight', [D, D], small(D * D));
  pushTensor('decoder.blocks.0.attn.query.bias', [D], zeros(D));
  pushTensor('decoder.blocks.0.attn.key.weight', [D, D], small(D * D));
  pushTensor('decoder.blocks.0.attn.value.weight', [D, D], small(D * D));
  pushTensor('decoder.blocks.0.attn.value.bias', [D], zeros(D));
  pushTensor('decoder.blocks.0.attn.out.weight', [D, D], small(D * D));
  pushTensor('decoder.blocks.0.attn.out.bias', [D], zeros(D));
  pushTensor('decoder.blocks.0.cross_attn_ln.weight', [D], ones(D));
  pushTensor('decoder.blocks.0.cross_attn_ln.bias', [D], zeros(D));
  pushTensor('decoder.blocks.0.cross_attn.query.weight', [D, D], small(D * D));
  pushTensor('decoder.blocks.0.cross_attn.query.bias', [D], zeros(D));
  pushTensor('decoder.blocks.0.cross_attn.key.weight', [D, D], small(D * D));
  pushTensor('decoder.blocks.0.cross_attn.value.weight', [D, D], small(D * D));
  pushTensor('decoder.blocks.0.cross_attn.value.bias', [D], zeros(D));
  pushTensor('decoder.blocks.0.cross_attn.out.weight', [D, D], small(D * D));
  pushTensor('decoder.blocks.0.cross_attn.out.bias', [D], zeros(D));
  pushTensor('decoder.blocks.0.mlp_ln.weight', [D], ones(D));
  pushTensor('decoder.blocks.0.mlp_ln.bias', [D], zeros(D));
  pushTensor('decoder.blocks.0.mlp.0.weight', [D, 16], small(D * 16));
  pushTensor('decoder.blocks.0.mlp.0.bias', [16], zeros(16));
  pushTensor('decoder.blocks.0.mlp.2.weight', [16, D], small(16 * D));
  pushTensor('decoder.blocks.0.mlp.2.bias', [D], zeros(D));
  pushTensor('decoder.ln.weight', [D], ones(D));
  pushTensor('decoder.ln.bias', [D], zeros(D));

  const totalLen = chunks.reduce((a, c) => a + c.byteLength, 0);
  const bin = new Uint8Array(totalLen);
  let cursor = 0;
  for (const c of chunks) {
    bin.set(c, cursor);
    cursor += c.byteLength;
  }

  const loaded = loadWhisperWeightsFromGgml(bin);
  assert.equal(loaded.config.nVocab, nVocab);
  assert.equal(loaded.config.nAudioState, D);
  assert.equal(loaded.config.sotTokenId, 0);
  assert.equal(loaded.config.eotTokenId, 1);

  const engine = new WhisperAudioEngine(loaded);
  const mel = computeLogMelSpectrogram(new Float32Array(WHISPER_SAMPLE_RATE / 4).fill(0.1));
  const out = engine.transcribe(mel, 4);
  assert.ok(out.encoderFrames > 0);
  assert.equal(out.tokenIds[0], 0);
});
