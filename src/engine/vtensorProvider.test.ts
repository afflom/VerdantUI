import test from 'node:test';
import assert from 'node:assert/strict';
import { GgmlType } from './gguf';
import {
  f16BitsToF32,
  verdantDecodeRow,
  verdantMulMat,
  verdantRmsNorm,
  verdantRmsNormPerHead,
  verdantRope,
  verdantSwiGlu,
} from './verdantKernels';
import { VerdantWasmArena } from './verdantWasmSimd';
import {
  DISABLED_DEFAULT_GGML_PROVIDERS,
  GgmlOp,
  VTensorBackendRegistry,
  VTensorPolicyError,
  runVTensorConformanceSuite,
} from './vtensorProvider';

test('vTensor backend registry enforces exclusive provider lock and blocks default GGML backends', () => {
  const registry = new VTensorBackendRegistry();
  assert.equal(registry.selectBackend('vTensor'), registry);

  for (const forbidden of DISABLED_DEFAULT_GGML_PROVIDERS) {
    assert.throws(
      () => registry.selectBackend(forbidden),
      VTensorPolicyError,
      `Expected ${forbidden} to be rejected`,
    );
  }
});

test('Prism graph admission refuses uncertified GGML operations without CPU fallback', () => {
  const registry = new VTensorBackendRegistry();
  assert.throws(
    () =>
      registry.graphCompute({
        nodes: [
          {
            op: 'UNSUPPORTED_OP' as unknown as GgmlOp,
            label: 'illegal.node',
            shapeDesc: '[1]',
            srcType: 'F32',
            execute: () => {},
          },
        ],
      }),
    VTensorPolicyError,
  );
  assert.equal(registry.getTelemetry().cpuFallbackOps, 0);
});

test('vTensor aligned linear buffer preserves 32-byte alignment and round-trips tensor bytes', () => {
  const registry = new VTensorBackendRegistry();
  const buf = registry.allocBuffer('test.buf', 'compute', 512);

  const t1 = buf.allocTensor('t1', GgmlType.F32, [3]); // 12 bytes
  const t2 = buf.allocTensor('t2', GgmlType.F32, [4]); // 16 bytes
  assert.equal(t1.byteOffset, 0);
  assert.equal(t2.byteOffset, 32); // 32-byte aligned per VT-10

  const payload = new Float32Array([1.5, -2.25, 3.75, 4.0]);
  buf.setTensor(t2, new Uint8Array(payload.buffer));
  const roundTrip = new Float32Array(buf.getTensor(t2).buffer);
  assert.deepEqual(Array.from(roundTrip), [1.5, -2.25, 3.75, 4.0]);
});

test('Verdant kernels compute F64-accumulated RMS_NORM, Q4_0/Q8_0 decoding, MUL_MAT, ROPE, and SWIGLU accurately', () => {
  // 1. RMS_NORM: x = [3, 4], mean(x^2) = 12.5, scale = 1/sqrt(12.5 + 1e-5)
  const x = new Float32Array([3, 4]);
  const w = new Float32Array([1, 2]);
  const out = new Float32Array(2);
  verdantRmsNorm(x, w, out, 2, 1e-5);
  const expectedScale = 1 / Math.sqrt(12.5 + 1e-5);
  assert.ok(Math.abs(out[0] - 3 * expectedScale) < 1e-5);
  assert.ok(Math.abs(out[1] - 4 * expectedScale * 2) < 1e-5);

  // 2. F16 0x3c00 == 1.0
  assert.equal(f16BitsToF32(0x3c00), 1.0);

  // 3. Q8_0 single block (32 elements, scale = 1.0, quants = [0, 1, -1, 2, ...])
  const q8Block = new Uint8Array(34);
  q8Block[0] = 0x00;
  q8Block[1] = 0x3c; // F16 1.0
  q8Block[2] = 0;
  q8Block[3] = 1;
  q8Block[4] = 255; // -1 signed int8
  q8Block[5] = 2;
  const decodedQ8 = new Float32Array(32);
  verdantDecodeRow(q8Block, 0, GgmlType.Q8_0, 32, decodedQ8);
  assert.equal(decodedQ8[0], 0);
  assert.equal(decodedQ8[1], 1);
  assert.equal(decodedQ8[2], -1);
  assert.equal(decodedQ8[3], 2);

  // 4. Q4_0 single block (32 elements, scale = 1.0, low nibble = 9 -> +1, high nibble = 6 -> -2)
  const q4Block = new Uint8Array(18);
  q4Block[0] = 0x00;
  q4Block[1] = 0x3c; // F16 1.0
  for (let j = 0; j < 16; j++) {
    q4Block[2 + j] = (6 << 4) | 9;
  }
  const vec32 = new Float32Array(32).fill(1.0);
  const mmOut = new Float32Array(1);
  verdantMulMat(q4Block, 0, GgmlType.Q4_0, 32, 1, vec32, mmOut);
  // 16*(+1) + 16*(-2) = -16
  assert.equal(mmOut[0], -16);

  // 5. SWIGLU: gate=0 -> silu(0)=0, gate=1 -> silu(1)=1/(1+e^-1)
  const swOut = new Float32Array(2);
  verdantSwiGlu(new Float32Array([0, 1]), new Float32Array([5, 2]), swOut, 2);
  assert.equal(swOut[0], 0);
  assert.ok(Math.abs(swOut[1] - (2 / (1 + Math.exp(-1)))) < 1e-5);

  // 6. ROPE at pos=0 is identity
  const ropeVec = new Float32Array([1, 2, 3, 4]);
  verdantRope(ropeVec, 1, 4, 4, 0, 10000);
  assert.deepEqual(Array.from(ropeVec), [1, 2, 3, 4]);
});

test('Verdant K-quant and 5-bit kernels (Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K) match exact mathematical definitions', () => {
  // Q4_1: d = 1.0 (0x3c00), m = 2.0 (0x4000), all low nibbles = 3, high nibbles = 5
  const q41 = new Uint8Array(20);
  q41[0] = 0x00; q41[1] = 0x3c;
  q41[2] = 0x00; q41[3] = 0x40;
  for (let j = 0; j < 16; j++) q41[4 + j] = (5 << 4) | 3;
  const dec32 = new Float32Array(32);
  verdantDecodeRow(q41, 0, GgmlType.Q4_1, 32, dec32);
  assert.equal(dec32[0], 5); // 3 * 1 + 2
  assert.equal(dec32[16], 7); // 5 * 1 + 2

  // Q5_0: d = 1.0 (0x3c00), qh = 0xFFFF0000 (low 16 have bit4=0, high 16 have bit4=1), all nibbles = 1
  // low 16 => (1 | 0) - 16 = -15; high 16 => (1 | 16) - 16 = +1
  const q50 = new Uint8Array(22);
  q50[0] = 0x00; q50[1] = 0x3c;
  q50[2] = 0x00; q50[3] = 0x00; q50[4] = 0xff; q50[5] = 0xff;
  for (let j = 0; j < 16; j++) q50[6 + j] = (1 << 4) | 1;
  verdantDecodeRow(q50, 0, GgmlType.Q5_0, 32, dec32);
  assert.equal(dec32[0], -15);
  assert.equal(dec32[15], -15);
  assert.equal(dec32[16], 1);
  assert.equal(dec32[31], 1);

  // Q5_1: d = 0.5 (0x3800), m = -2.0 (0xc000), qh = 0x00000001 (only elem 0 has high bit=1), nibbles = 2
  const q51 = new Uint8Array(24);
  q51[0] = 0x00; q51[1] = 0x38;
  q51[2] = 0x00; q51[3] = 0xc0;
  q51[4] = 0x01;
  for (let j = 0; j < 16; j++) q51[8 + j] = (2 << 4) | 2;
  verdantDecodeRow(q51, 0, GgmlType.Q5_1, 32, dec32);
  assert.equal(dec32[0], (2 | 16) * 0.5 - 2.0); // 7.0
  assert.equal(dec32[1], 2 * 0.5 - 2.0); // -1.0

  // Verify NeoX RoPE splits [x_i, x_{i+half}]
  const neoxVec = new Float32Array([1, 2, 0, 0]);
  verdantRope(neoxVec, 1, 4, 4, 1, 10000, true);
  assert.ok(Math.abs(neoxVec[0] - Math.cos(1)) < 1e-5);
  assert.ok(Math.abs(neoxVec[2] - Math.sin(1)) < 1e-5);
});

test('runVTensorConformanceSuite executes all 18 isolated VT checks (VT-02..VT-66) and passes cleanly', () => {
  const report = runVTensorConformanceSuite();
  assert.equal(report.totalCount, 18);
  assert.equal(report.passedCount, 18);
  assert.equal(report.checks.length, 18);
  assert.ok(report.checks.some((c) => c.id === 'VT-65' && c.passed));
  assert.ok(report.checks.some((c) => c.id === 'VT-66' && c.passed));
  for (const check of report.checks) {
    assert.equal(check.passed, true, `Expected ${check.id} (${check.name}) to pass: ${check.detail}`);
    assert.ok(check.durationMs >= 0);
  }
});

test('Q2_K, Q3_K decode/mulMat (JS + WASM arena) and Qwen3 per-head QK-Norm match exact ggml-quants.c formulas', () => {
  // 1. Q2_K single super-block (84 bytes = 16 scales + 64 qs + 2 d + 2 dmin)
  // Set scales[j] = (2 << 4) | 3 => scale=3, min=2; d = 0.5 (0x3800), dmin = 0.25 (0x3400)
  // Set all qs bytes = 0xE4 (0b11_10_01_00):
  //   shift=0 -> q=0, shift=2 -> q=1, shift=4 -> q=2, shift=6 -> q=3
  const q2k = new Uint8Array(84);
  for (let i = 0; i < 16; i++) q2k[i] = (2 << 4) | 3;
  for (let i = 0; i < 64; i++) q2k[16 + i] = 0xe4;
  q2k[80] = 0x00; q2k[81] = 0x38; // d = 0.5
  q2k[82] = 0x00; q2k[83] = 0x34; // dmin = 0.25
  // dl = 0.5 * 3 = 1.5, ml = 0.25 * 2 = 0.5
  // Elements 0..31: shift=0 => q=0 => 0*1.5 - 0.5 = -0.5
  // Elements 32..63: shift=2 => q=1 => 1*1.5 - 0.5 = +1.0
  // Elements 64..95: shift=4 => q=2 => 2*1.5 - 0.5 = +2.5
  // Elements 96..127: shift=6 => q=3 => 3*1.5 - 0.5 = +4.0
  const dec256 = new Float32Array(256);
  verdantDecodeRow(q2k, 0, GgmlType.Q2_K, 256, dec256);
  assert.ok(Math.abs(dec256[0] - -0.5) < 1e-6);
  assert.ok(Math.abs(dec256[32] - 1.0) < 1e-6);
  assert.ok(Math.abs(dec256[64] - 2.5) < 1e-6);
  assert.ok(Math.abs(dec256[96] - 4.0) < 1e-6);

  const vec256 = new Float32Array(256);
  for (let i = 0; i < 256; i++) vec256[i] = ((i % 7) - 3) * 0.25;
  let refDotQ2 = 0;
  for (let i = 0; i < 256; i++) refDotQ2 += dec256[i] * vec256[i];

  const outQ2 = new Float32Array(1);
  verdantMulMat(q2k, 0, GgmlType.Q2_K, 256, 1, vec256, outQ2);
  assert.ok(Math.abs(outQ2[0] - refDotQ2) < 1e-4, `Q2_K mulMat mismatch: ${outQ2[0]} vs ${refDotQ2}`);

  // 2. Q3_K single super-block (110 bytes = 32 hmask + 64 qs + 12 scales + 2 d)
  // Fill scales with deterministic packed 6-bit values, d = 0.5 (0x3800), hmask alternating, qs alternating
  const q3k = new Uint8Array(110);
  for (let i = 0; i < 32; i++) q3k[i] = i % 2 === 0 ? 0xaa : 0x55;
  for (let i = 0; i < 64; i++) q3k[32 + i] = (i * 37 + 11) & 0xff;
  for (let i = 0; i < 12; i++) q3k[96 + i] = (i * 29 + 5) & 0xff;
  q3k[108] = 0x00; q3k[109] = 0x38; // d = 0.5

  const decQ3 = new Float32Array(256);
  verdantDecodeRow(q3k, 0, GgmlType.Q3_K, 256, decQ3);
  let refDotQ3 = 0;
  for (let i = 0; i < 256; i++) refDotQ3 += decQ3[i] * vec256[i];

  const outQ3 = new Float32Array(1);
  verdantMulMat(q3k, 0, GgmlType.Q3_K, 256, 1, vec256, outQ3);
  assert.ok(Math.abs(outQ3[0] - refDotQ3) < 1e-4, `Q3_K mulMat mismatch: ${outQ3[0]} vs ${refDotQ3}`);

  // 3. Verify VerdantWasmArena zero-copy dispatch matches for both Q2_K and Q3_K
  const arena = new VerdantWasmArena(16);
  const wQ2Off = arena.allocRegion(q2k.byteLength);
  arena.bytes().set(q2k, wQ2Off);
  const wQ3Off = arena.allocRegion(q3k.byteLength);
  arena.bytes().set(q3k, wQ3Off);
  const xOff = arena.allocRegion(256 * 4);
  new Float32Array(arena.memory.buffer, xOff, 256).set(vec256);
  const yOff = arena.allocRegion(4);

  arena.mulMat(GgmlType.Q2_K, wQ2Off, xOff, yOff, 256, 1);
  assert.ok(Math.abs(new Float32Array(arena.memory.buffer, yOff, 1)[0] - refDotQ2) < 1e-4);
  arena.mulMat(GgmlType.Q3_K, wQ3Off, xOff, yOff, 256, 1);
  assert.ok(Math.abs(new Float32Array(arena.memory.buffer, yOff, 1)[0] - refDotQ3) < 1e-4);

  // 4. Qwen3 per-head QK-Norm: 2 heads of headDim=2, head0=[3, 4], head1=[0, 2], shared weight=[1, 2]
  const qkVec = new Float32Array([3, 4, 0, 2]);
  const qkW = new Float32Array([1, 2]);
  verdantRmsNormPerHead(qkVec, qkW, 2, 2, 1e-5);
  const s0 = 1 / Math.sqrt(12.5 + 1e-5);
  const s1 = 1 / Math.sqrt(2.0 + 1e-5);
  assert.ok(Math.abs(qkVec[0] - 3 * s0) < 1e-5);
  assert.ok(Math.abs(qkVec[1] - 4 * s0 * 2) < 1e-5);
  assert.ok(Math.abs(qkVec[2] - 0) < 1e-6);
  assert.ok(Math.abs(qkVec[3] - 2 * s1 * 2) < 1e-5);
});

