import test from 'node:test';
import assert from 'node:assert/strict';
import { GgmlType } from './gguf';
import { verdantMulMat, verdantRmsNorm, buildRopeTables, verdantRope, verdantRopePrecomputed } from './verdantKernels';
import {
  buildVerdantSimdModuleBytes,
  isVerdantWasmSimdSupported,
  Q8_0_ACT_BLOCK_BYTES,
  Q8_K_ACT_BLOCK_BYTES,
  VerdantWasmArena,
} from './verdantWasmSimd';

test('Verdant WASM SIMD128 MUL_MAT matches JS F64 reference across all 11 GGML formats and batch-4 Q4_0', () => {
  assert.equal(isVerdantWasmSimdSupported(), true);
  assert.equal(WebAssembly.validate(buildVerdantSimdModuleBytes(false)), true);
  const arena = new VerdantWasmArena(64);

  const formats: Array<[string, GgmlType, number, number, number, number]> = [
    ['F32', GgmlType.F32, 1, 4, 256, 1e-4],
    ['F16', GgmlType.F16, 1, 2, 256, 1e-4],
    ['BF16', GgmlType.BF16, 1, 2, 256, 1e-4],
    ['Q8_0', GgmlType.Q8_0, 32, 34, 256, 5e-4],
    ['Q4_0', GgmlType.Q4_0, 32, 18, 256, 5e-4],
    ['Q4_1', GgmlType.Q4_1, 32, 20, 256, 5e-4],
    ['Q5_0', GgmlType.Q5_0, 32, 22, 256, 5e-4],
    ['Q5_1', GgmlType.Q5_1, 32, 24, 256, 5e-4],
    ['Q2_K', GgmlType.Q2_K, 256, 84, 512, 1e-3],
    ['Q3_K', GgmlType.Q3_K, 256, 110, 512, 1e-3],
    ['Q4_K', GgmlType.Q4_K, 256, 144, 512, 1e-3],
    ['Q5_K', GgmlType.Q5_K, 256, 176, 512, 1e-3],
    ['Q6_K', GgmlType.Q6_K, 256, 210, 512, 1e-3],
  ];

  const rows = 13; // odd count exercises both 2-row tiled loop and odd tail
  for (const [name, type, bs, bbytes, cols, tol] of formats) {
    arena.reset();
    const wBytes = rows * (cols / bs) * bbytes;
    const wPtr = arena.allocRegion(wBytes);
    const xPtr = arena.allocRegion(cols * 4);
    const outPtr = arena.allocRegion(rows * 4);

    const memU8 = arena.bytes();
    for (let i = 0; i < wBytes; i++) memU8[wPtr + i] = (i * 37 + 13) & 0xff;

    if (type === GgmlType.F32) {
      const f32 = new Float32Array(arena.memory.buffer, wPtr, rows * cols);
      for (let i = 0; i < f32.length; i++) f32[i] = Math.sin(i * 0.13);
    } else if (type === GgmlType.F16 || type === GgmlType.BF16) {
      const u16 = new Uint16Array(arena.memory.buffer, wPtr, rows * cols);
      for (let i = 0; i < u16.length; i++) {
        u16[i] = type === GgmlType.F16 ? (0x3400 | (i & 0x3ff)) : (0x3f00 | (i & 0x7f));
      }
    } else {
      const nBlk = rows * (cols / bs);
      for (let b = 0; b < nBlk; b++) {
        const off = wPtr + b * bbytes;
        if (type === GgmlType.Q2_K) {
          memU8[off + 80] = 0x00;
          memU8[off + 81] = 0x34;
          memU8[off + 82] = 0x00;
          memU8[off + 83] = 0x30;
        } else if (type === GgmlType.Q3_K) {
          memU8[off + 108] = 0x00;
          memU8[off + 109] = 0x34;
        } else if (type === GgmlType.Q6_K) {
          memU8[off + 208] = 0x00;
          memU8[off + 209] = 0x34;
        } else {
          memU8[off] = 0x00;
          memU8[off + 1] = 0x34;
          if (type === GgmlType.Q4_1 || type === GgmlType.Q5_1 || type === GgmlType.Q4_K || type === GgmlType.Q5_K) {
            memU8[off + 2] = 0x00;
            memU8[off + 3] = 0x30;
          }
        }
      }
    }

    const xView = new Float32Array(arena.memory.buffer, xPtr, cols);
    for (let i = 0; i < cols; i++) xView[i] = Math.cos(i * 0.07) * 0.5;

    const refOut = new Float32Array(rows);
    verdantMulMat(memU8.subarray(wPtr, wPtr + wBytes), 0, type, cols, rows, xView, refOut);

    arena.mulMat(type, wPtr, xPtr, outPtr, cols, rows);
    const got = new Float32Array(arena.memory.buffer, outPtr, rows);

    for (let r = 0; r < rows; r++) {
      const diff = Math.abs(refOut[r] - got[r]);
      assert.ok(
        Number.isFinite(got[r]) && diff <= tol,
        `${name} row ${r}: expected ${refOut[r]}, got ${got[r]} (diff ${diff})`,
      );
    }
  }

  // Test mul_mat_q4_0_batch4 against 4 independent single-token reference calls
  arena.reset();
  const cols = 256;
  const wBytes = rows * (cols / 32) * 18;
  const wPtr = arena.allocRegion(wBytes);
  const xBatchPtr = arena.allocRegion(cols * 4 * 4);
  const outBatchPtr = arena.allocRegion(rows * 4 * 4);
  const memU8 = arena.bytes();
  for (let i = 0; i < wBytes; i++) memU8[wPtr + i] = (i * 29 + 5) & 0xff;
  for (let b = 0; b < rows * (cols / 32); b++) {
    memU8[wPtr + b * 18] = 0x00;
    memU8[wPtr + b * 18 + 1] = 0x34;
  }
  const xBatch = new Float32Array(arena.memory.buffer, xBatchPtr, cols * 4);
  for (let i = 0; i < xBatch.length; i++) xBatch[i] = Math.sin(i * 0.09);

  arena.mulMatBatch(GgmlType.Q4_0, wPtr, xBatchPtr, outBatchPtr, cols, rows, 4);
  const outBatch = new Float32Array(arena.memory.buffer, outBatchPtr, rows * 4);

  const refSingle = new Float32Array(rows);
  for (let t = 0; t < 4; t++) {
    verdantMulMat(
      memU8.subarray(wPtr, wPtr + wBytes),
      0,
      GgmlType.Q4_0,
      cols,
      rows,
      xBatch.subarray(t * cols, (t + 1) * cols),
      refSingle,
    );
    for (let r = 0; r < rows; r++) {
      const diff = Math.abs(refSingle[r] - outBatch[t * rows + r]);
      assert.ok(diff <= 5e-4, `Q4_0 batch4 token ${t} row ${r} diff ${diff}`);
    }
  }
});

test('Verdant Relaxed-SIMD VNNI / Int8 Q8_0-activation MUL_MAT & batch-4 prefill match F64 reference across all 8 quant formats + F32 batch-4', () => {
  const arena = new VerdantWasmArena(64);
  const rows = 11;
  const cols = 512;
  const B = 4;

  const formats: Array<[string, GgmlType, number, number]> = [
    ['Q4_0', GgmlType.Q4_0, 32, 18],
    ['Q4_1', GgmlType.Q4_1, 32, 20],
    ['Q5_0', GgmlType.Q5_0, 32, 22],
    ['Q5_1', GgmlType.Q5_1, 32, 24],
    ['Q8_0', GgmlType.Q8_0, 32, 34],
    ['Q4_K', GgmlType.Q4_K, 256, 144],
    ['Q5_K', GgmlType.Q5_K, 256, 176],
    ['Q6_K', GgmlType.Q6_K, 256, 210],
  ];

  for (const [name, type, bs, bbytes] of formats) {
    arena.reset();
    const wBytes = rows * (cols / bs) * bbytes;
    const wPtr = arena.allocRegion(wBytes);
    const xPtr = arena.allocRegion(cols * B * 4);
    const qStride = (cols / 32) * Q8_0_ACT_BLOCK_BYTES;
    const q8Ptr = arena.allocRegion(qStride * B);
    const outS = arena.allocRegion(rows * 4);
    const outB = arena.allocRegion(rows * B * 4);

    const memU8 = arena.bytes();
    for (let i = 0; i < wBytes; i++) memU8[wPtr + i] = (i * 41 + 9) & 0xff;
    const nBlk = rows * (cols / bs);
    for (let b = 0; b < nBlk; b++) {
      const off = wPtr + b * bbytes;
      if (type === GgmlType.Q6_K) {
        memU8[off + 208] = 0x00;
        memU8[off + 209] = 0x34;
      } else {
        memU8[off] = 0x00;
        memU8[off + 1] = 0x34;
        if (type === GgmlType.Q4_1 || type === GgmlType.Q5_1 || type === GgmlType.Q4_K || type === GgmlType.Q5_K) {
          memU8[off + 2] = 0x00;
          memU8[off + 3] = 0x30;
        }
      }
    }

    const xBatch = new Float32Array(arena.memory.buffer, xPtr, cols * B);
    for (let t = 0; t < B; t++) {
      for (let i = 0; i < cols; i++) {
        xBatch[t * cols + i] = Math.sin(i * 0.11 + t * 0.65) * 0.75;
      }
      arena.quantizeActivation(xPtr + t * cols * 4, q8Ptr + t * qStride, cols);
    }

    // Batch-4 Q8×W
    arena.mulMatBatchQ8(type, wPtr, q8Ptr, xPtr, outB, cols, rows, B);
    const gotB = new Float32Array(arena.memory.buffer, outB, rows * B);

    const dv = new DataView(arena.memory.buffer);
    const xDeq = new Float32Array(cols);
    const refDeq = new Float32Array(rows);
    const refF32 = new Float32Array(rows);
    for (let t = 0; t < B; t++) {
      const qTokOff = q8Ptr + t * qStride;
      for (let blk = 0; blk < cols / 32; blk++) {
        const off = qTokOff + blk * Q8_0_ACT_BLOCK_BYTES;
        const d = dv.getFloat32(off, true);
        for (let j = 0; j < 32; j++) xDeq[blk * 32 + j] = dv.getInt8(off + 16 + j) * d;
      }

      verdantMulMat(memU8.subarray(wPtr, wPtr + wBytes), 0, type, cols, rows, xDeq, refDeq);
      verdantMulMat(
        memU8.subarray(wPtr, wPtr + wBytes),
        0,
        type,
        cols,
        rows,
        xBatch.subarray(t * cols, (t + 1) * cols),
        refF32,
      );
      arena.mulMatQ8(type, wPtr, qTokOff, xPtr + t * cols * 4, outS, cols, rows);
      const gotS = new Float32Array(arena.memory.buffer, outS, rows);

      let xErrSq = 0;
      let xNormSq = 0;
      for (let i = 0; i < cols; i++) {
        const xv = xBatch[t * cols + i];
        const dx = xv - xDeq[i];
        xErrSq += dx * dx;
        xNormSq += xv * xv;
      }
      const actL2Rel = Math.sqrt(xErrSq) / Math.max(1e-6, Math.sqrt(xNormSq));
      assert.ok(actL2Rel < 0.01, `${name} tok=${t} Q8_0 activation L2 error ${actL2Rel}`);

      let gotF32ErrSq = 0;
      let deqF32ErrSq = 0;
      for (let r = 0; r < rows; r++) {
        const exactDiff = Math.abs(refDeq[r] - gotS[r]);
        assert.ok(
          Number.isFinite(gotS[r]) && exactDiff <= 2e-3,
          `${name} Q8_0 tok=${t} row=${r}: refDeq=${refDeq[r]}, got=${gotS[r]} (diff=${exactDiff})`,
        );
        const batchDelta = Math.abs(gotS[r] - gotB[t * rows + r]);
        assert.ok(
          batchDelta <= 1e-4,
          `${name} batch4 vs single tok=${t} row=${r}: single=${gotS[r]}, batch=${gotB[t * rows + r]}`,
        );
        const eg = refF32[r] - gotS[r];
        const ed = refF32[r] - refDeq[r];
        gotF32ErrSq += eg * eg;
        deqF32ErrSq += ed * ed;
      }
      assert.ok(
        Math.sqrt(gotF32ErrSq) <= Math.sqrt(deqF32ErrSq) + 5e-3,
        `${name} tok=${t} excess error beyond Q8_0 quantization`,
      );
    }
  }

  // Also test F32 batch-4 relaxed FMA kernel
  arena.reset();
  const wPtr = arena.allocRegion(rows * cols * 4);
  const xPtr = arena.allocRegion(cols * B * 4);
  const outB = arena.allocRegion(rows * B * 4);
  const wf = new Float32Array(arena.memory.buffer, wPtr, rows * cols);
  const xf = new Float32Array(arena.memory.buffer, xPtr, cols * B);
  for (let i = 0; i < wf.length; i++) wf[i] = Math.cos(i * 0.05) * 0.4;
  for (let i = 0; i < xf.length; i++) xf[i] = Math.sin(i * 0.07) * 0.5;
  arena.mulMatBatch(GgmlType.F32, wPtr, xPtr, outB, cols, rows, B);
  const gotF32B = new Float32Array(arena.memory.buffer, outB, rows * B);
  const refF32 = new Float32Array(rows);
  for (let t = 0; t < B; t++) {
    verdantMulMat(
      arena.bytes().subarray(wPtr, wPtr + rows * cols * 4),
      0,
      GgmlType.F32,
      cols,
      rows,
      xf.subarray(t * cols, (t + 1) * cols),
      refF32,
    );
    for (let r = 0; r < rows; r++) {
      assert.ok(Math.abs(refF32[r] - gotF32B[t * rows + r]) <= 1e-4);
    }
  }
});

test('Verdant WASM SIMD Cold-Q4 attention kernels (dot_q4_f32 / axpy_q4_f32) and calibration probe', () => {
  const arena = new VerdantWasmArena(32);
  const beforeCursor = arena.usedBytes;
  const cal = arena.runCalibrationProbe();
  assert.ok(cal.memoryBandwidthGBs > 0 && cal.computeGops > 0 && cal.ridgeIntensityOpsPerByte > 0);
  assert.equal(arena.usedBytes, beforeCursor); // calibration resets cursor with zero leaked bytes!

  const n = 64;
  const q4Ptr = arena.allocRegion(n / 2);
  const xPtr = arena.allocRegion(n * 4);
  const dstPtr = arena.allocRegion(n * 4);
  const mem = arena.bytes();
  const x = new Float32Array(arena.memory.buffer, xPtr, n);
  const dst = new Float32Array(arena.memory.buffer, dstPtr, n);

  const vals = new Int8Array(n);
  for (let i = 0; i < n; i++) {
    vals[i] = ((i * 5) % 15) - 7; // [-7..+7]
    x[i] = Math.sin(i * 0.21);
    dst[i] = 0.25 * i;
  }
  // Pack split-nibble 32-element chunks
  let bOff = q4Ptr;
  for (let c = 0; c < n; c += 32) {
    for (let j = 0; j < 16; j++) {
      mem[bOff++] = ((vals[c + j] + 8) & 0x0f) | (((vals[c + 16 + j] + 8) & 0x0f) << 4);
    }
  }

  let refDot = 0;
  for (let i = 0; i < n; i++) refDot += vals[i] * x[i];
  const gotDot = arena.kernels.dot_q4_f32(q4Ptr, xPtr, n);
  assert.ok(Math.abs(refDot - gotDot) <= 1e-4, `dot_q4_f32: ${refDot} vs ${gotDot}`);

  const expectedDst = Float32Array.from(dst, (v, i) => v + vals[i] * 0.35);
  arena.kernels.axpy_q4_f32(dstPtr, q4Ptr, n, 0.35);
  for (let i = 0; i < n; i++) {
    assert.ok(Math.abs(expectedDst[i] - dst[i]) <= 1e-4, `axpy_q4_f32 idx ${i}`);
  }
});

test('Verdant WASM SIMD RMS_NORM and precomputed RoPE tables match scalar references', () => {
  const arena = new VerdantWasmArena(16);
  const n = 131; // non-multiple of 4 exercises SIMD + scalar tail
  const xPtr = arena.allocRegion(n * 4);
  const wPtr = arena.allocRegion(n * 4);
  const oPtr = arena.allocRegion(n * 4);
  const x = new Float32Array(arena.memory.buffer, xPtr, n);
  const w = new Float32Array(arena.memory.buffer, wPtr, n);
  for (let i = 0; i < n; i++) {
    x[i] = Math.sin(i * 0.3) * 2.5;
    w[i] = 0.8 + Math.cos(i * 0.1) * 0.2;
  }
  const ref = new Float32Array(n);
  verdantRmsNorm(x, w, ref, n, 1e-5);
  arena.kernels.rms_norm_f32(xPtr, wPtr, oPtr, n, 1e-5);
  const got = new Float32Array(arena.memory.buffer, oPtr, n);
  for (let i = 0; i < n; i++) {
    assert.ok(Math.abs(ref[i] - got[i]) <= 1e-5, `RMS_NORM idx ${i} mismatch`);
  }

  // Precomputed RoPE table parity for both normal and NeoX modes
  const tables = buildRopeTables(32, 64, 100000);
  for (const neox of [false, true]) {
    const vRef = new Float32Array(128);
    const vLut = new Float32Array(128);
    for (let i = 0; i < 128; i++) vRef[i] = vLut[i] = Math.sin(i * 0.23);
    verdantRope(vRef, 2, 64, 64, 17, 100000, neox);
    verdantRopePrecomputed(vLut, 2, 64, 17, tables, neox);
    for (let i = 0; i < 128; i++) {
      assert.ok(Math.abs(vRef[i] - vLut[i]) <= 1e-5, `RoPE neox=${neox} idx ${i} drift`);
    }
  }

  // Verify 4x-unrolled WASM SIMD add_f32 (both out-of-place and in-place residual add)
  const addN = 147;
  const aPtr = arena.allocRegion(addN * 4);
  const bPtr = arena.allocRegion(addN * 4);
  const outAddPtr = arena.allocRegion(addN * 4);
  const a = new Float32Array(arena.memory.buffer, aPtr, addN);
  const b = new Float32Array(arena.memory.buffer, bPtr, addN);
  const expectedAdd = new Float32Array(addN);
  for (let i = 0; i < addN; i++) {
    a[i] = Math.sin(i * 0.19) * 3.0;
    b[i] = Math.cos(i * 0.27) * 1.5;
    expectedAdd[i] = Math.fround(a[i] + b[i]);
  }
  arena.kernels.add_f32(aPtr, bPtr, outAddPtr, addN);
  const gotAdd = new Float32Array(arena.memory.buffer, outAddPtr, addN);
  for (let i = 0; i < addN; i++) {
    assert.ok(Math.abs(gotAdd[i] - expectedAdd[i]) <= 1e-6, `add_f32 out-of-place idx ${i}`);
  }
  arena.kernels.add_f32(aPtr, bPtr, aPtr, addN);
  for (let i = 0; i < addN; i++) {
    assert.ok(Math.abs(a[i] - expectedAdd[i]) <= 1e-6, `add_f32 in-place idx ${i}`);
  }
});

test('Verdant WASM SIMD/VNNI Q2_K & Q3_K × Q8_K narrow-lane (4-row / 2-row + batch-4) match exact F64 dequantized reference', () => {
  const arena = new VerdantWasmArena(64);
  const cols = 512;
  const B = 4;

  const cases: Array<[string, GgmlType, number, number]> = [
    ['Q2_K', GgmlType.Q2_K, 84, 17],
    ['Q2_K-aligned', GgmlType.Q2_K, 84, 32],
    ['Q3_K', GgmlType.Q3_K, 110, 17],
    ['Q3_K-aligned', GgmlType.Q3_K, 110, 32],
  ];

  for (const [label, type, bbytes, rows] of cases) {
    arena.reset();
    const nSuper = cols >>> 8;
    const wBytes = rows * nSuper * bbytes;
    const wPtr = arena.allocRegion(wBytes);
    const xPtr = arena.allocRegion(cols * B * 4);
    const qStride = nSuper * Q8_K_ACT_BLOCK_BYTES;
    const q8kPtr = arena.allocRegion(qStride * B);
    const outS = arena.allocRegion(rows * 4);
    const outB = arena.allocRegion(rows * B * 4);

    const memU8 = arena.bytes();
    for (let i = 0; i < wBytes; i++) memU8[wPtr + i] = (i * 59 + 17) & 0xff;
    for (let b = 0; b < rows * nSuper; b++) {
      const off = wPtr + b * bbytes;
      if (type === GgmlType.Q2_K) {
        memU8[off + 80] = 0x00;
        memU8[off + 81] = 0x34;
        memU8[off + 82] = 0x00;
        memU8[off + 83] = 0x30;
      } else {
        memU8[off + 108] = 0x00;
        memU8[off + 109] = 0x34;
      }
    }

    const xBatch = new Float32Array(arena.memory.buffer, xPtr, cols * B);
    for (let t = 0; t < B; t++) {
      for (let i = 0; i < cols; i++) {
        xBatch[t * cols + i] = Math.sin(i * 0.09 + t * 0.71) * 0.85;
      }
      arena.quantizeActivationQ8K(xPtr + t * cols * 4, q8kPtr + t * qStride, cols);
    }

    arena.mulMatBatchQ8K(type, wPtr, q8kPtr, xPtr, outB, cols, rows, B);
    const gotB = new Float32Array(arena.memory.buffer, outB, rows * B);

    const dv = new DataView(arena.memory.buffer);
    const xDeq = new Float32Array(cols);
    const refDeq = new Float32Array(rows);
    for (let t = 0; t < B; t++) {
      const qTokOff = q8kPtr + t * qStride;
      // Reconstruct exact dequantized Q8_K vector: d at +0, bsums16 at +16, bsums32 at +48, qs[256] at +64
      for (let sb = 0; sb < nSuper; sb++) {
        const sbOff = qTokOff + sb * Q8_K_ACT_BLOCK_BYTES;
        const d = dv.getFloat32(sbOff, true);
        for (let j = 0; j < 256; j++) {
          xDeq[sb * 256 + j] = dv.getInt8(sbOff + 64 + j) * d;
        }
      }
      verdantMulMat(memU8.subarray(wPtr, wPtr + wBytes), 0, type, cols, rows, xDeq, refDeq);
      arena.mulMatQ8K(type, wPtr, qTokOff, xPtr + t * cols * 4, outS, cols, rows);
      const gotS = new Float32Array(arena.memory.buffer, outS, rows);

      for (let r = 0; r < rows; r++) {
        const diff = Math.abs(refDeq[r] - gotS[r]);
        assert.ok(
          Number.isFinite(gotS[r]) && diff <= 2e-3,
          `${label} tok=${t} row=${r}: refDeq=${refDeq[r]} got=${gotS[r]} (diff=${diff})`,
        );
        const bDiff = Math.abs(gotS[r] - gotB[t * rows + r]);
        assert.ok(
          bDiff <= 1e-4,
          `${label} batch4 vs single tok=${t} row=${r}: single=${gotS[r]} batch=${gotB[t * rows + r]}`,
        );
      }
    }
  }
});

