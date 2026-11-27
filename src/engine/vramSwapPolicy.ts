/**
 * vRAM Swap Latency Bounds (Theorem T7), Exact Page Repetition Factoring,
 * 2-Symbol Column-Mask Selector, and Verdant 16,666 µs FrameBudgetProfiler.
 *
 * Directly extrapolated from:
 * - github.com/afflom/vRAM (`app/src/SwapPolicy.lex.tex`, `device/src/repeated.rs`, `device/src/selector.rs`)
 * - github.com/afflom/verdant-engine (`FrameBudgetProfiler` 16,666 µs KPI)
 *
 * Key guarantees:
 * 1. Zero-clamp cooperative yielding via `MessageChannel` (~20 µs macrotask turn instead of
 *    Chromium's 4,000 µs nested `setTimeout(0)` clamp), gated by a 16,666 µs frame budget.
 * 2. Proven Lean 4 SwapPolicy invariants (`swapMiddle`, `swapTwoWithin`, `swapServiceBudget`,
 *    `swapBreakEven`, `swapFactorDominance`, `swapCopyCount`).
 * 3. Bit-exact lossless page compression (`Repeated` dyadic/shifted factoring + `Selector`
 *    2-symbol shared column bitmask) admitted strictly via `decideUorcChallengerAdmission`.
 */

/** Verdant Engine 60 Hz interactive frame budget KPI (16,666 µs = 16.666 ms). */
export const FRAME_BUDGET_US = 16_666;

/** Cooperative yield slice target during active token generation (leaves 4.6 ms UI headroom). */
export const GENERATION_SLICE_BUDGET_US = 12_000;

/** Cooperative yield slice target during background speculative prefill (sub-half-frame). */
export const SPECULATIVE_SLICE_BUDGET_US = 8_000;

// ---------------------------------------------------------------------------
// 1. Proven SwapPolicy Invariants (afflom/vRAM app/src/SwapPolicy.lex.tex)
// ---------------------------------------------------------------------------

/** Exact 3-sample median (`SwapPolicy.middle`). */
export function swapMiddle(a: number, b: number, c: number): number {
  if (a <= b) {
    if (b <= c) return b;
    return a <= c ? c : a;
  }
  if (a <= c) return a;
  return b <= c ? c : b;
}

/**
 * `SwapPolicy.twoWithin`: returns true iff at least 2 of 3 samples are `<= bound`.
 * Proved in Lean 4 to be equivalent to `swapMiddle(a, b, c) <= bound`.
 */
export function swapTwoWithin(a: number, b: number, c: number, bound: number): boolean {
  return swapMiddle(a, b, c) <= bound;
}

export interface ServiceBudgetCheck {
  readonly transportWithin: boolean;
  readonly serviceWithin: boolean;
  readonly ceilingValid: boolean;
  readonly totalWithinZram: boolean;
}

/**
 * `SwapPolicy.serviceBudget`: verifies that bounded transport + bounded service stay
 * strictly below the incumbent zram/baseline latency ceiling.
 */
export function swapServiceBudget(
  transportUs: number,
  serviceUs: number,
  transportMaxUs: number,
  serviceMaxUs: number,
  baselineUs: number,
): ServiceBudgetCheck {
  const transportWithin = transportUs <= transportMaxUs;
  const serviceWithin = serviceUs <= serviceMaxUs;
  const ceilingValid = transportMaxUs + serviceMaxUs <= baselineUs;
  return {
    transportWithin,
    serviceWithin,
    ceilingValid,
    totalWithinZram: transportUs + serviceUs <= baselineUs,
  };
}

/**
 * `SwapPolicy.zstdBreakEven`: challenger beats incumbent iff extra encode/decode overhead
 * does not exceed the transport/storage latency saved (`extra <= saved`).
 */
export function swapBreakEven(
  sharedUs: number,
  extraUs: number,
  serviceUs: number,
  savedUs: number,
): boolean {
  return sharedUs + extraUs + serviceUs <= sharedUs + serviceUs + savedUs && extraUs <= savedUs;
}

/**
 * `SwapPolicy.factorDominance`: monotonic dominance check (`challenger <= incumbent` preserves
 * total cost ordering when adding identical fixed framing `fixedCost`).
 */
export function swapFactorDominance(
  challengerBytes: number,
  incumbentBytes: number,
  fixedOverheadBytes = 0,
): boolean {
  return (
    challengerBytes <= incumbentBytes &&
    challengerBytes + fixedOverheadBytes <= incumbentBytes + fixedOverheadBytes
  );
}

/** `SwapPolicy.copyBudget`: exact number of pages copied in one transfer burst. */
export function swapCopyCount(ready: number, missing: number): number {
  return Math.min(Math.max(0, Math.floor(ready)), Math.max(0, Math.floor(missing)));
}

/** Strict challenger admission gate (`decide_uorc_challenger_admission` in afflom/vRAM). */
export function decideUorcChallengerAdmission(
  challengerBytes: number,
  incumbentBytes: number,
): boolean {
  return challengerBytes > 0 && challengerBytes < incumbentBytes;
}

// ---------------------------------------------------------------------------
// 2. Exact Page Repetition Factoring (afflom/vRAM device/src/repeated.rs)
// ---------------------------------------------------------------------------

export interface RepeatedFactorResult {
  readonly kind: 'dyadic' | 'shifted';
  readonly unit: Uint8Array;
  readonly halvings: number;
  readonly originalLength: number;
  readonly encodedBytes: number;
}

/**
 * Dyadic exact page repetition factoring (`Repeated::factor` in afflom/vRAM):
 * Repeatedly halves the buffer whenever the first half equals the second half byte-for-byte.
 * Returns `null` if no strictly smaller exact factor exists.
 */
export function factorRepeatedBytes(bytes: Uint8Array): RepeatedFactorResult | null {
  const originalLength = bytes.byteLength;
  if (originalLength < 8) return null;

  let len = originalLength;
  let halvings = 0;

  while (len >= 2 && (len & 1) === 0) {
    const half = len >>> 1;
    let equal = true;
    for (let i = 0; i < half; i++) {
      if (bytes[i] !== bytes[half + i]) {
        equal = false;
        break;
      }
    }
    if (!equal) break;
    len = half;
    halvings += 1;
  }

  if (halvings === 0) return null;
  const encodedBytes = 1 + 4 + len; // codec tag + u32 unitLen + unit payload
  if (!decideUorcChallengerAdmission(encodedBytes, originalLength)) {
    return null;
  }

  return {
    kind: 'dyadic',
    unit: bytes.slice(0, len),
    halvings,
    originalLength,
    encodedBytes,
  };
}

/**
 * Non-dyadic exact periodic factoring (`Repeated::shifted` in afflom/vRAM):
 * Scans for the smallest period `d` (`1 <= d <= maxPeriod`) using an 8-byte prefix window
 * match before verifying full exact periodicity `bytes[i] === bytes[i % d]`.
 */
export function shiftedRepeatedBytes(
  bytes: Uint8Array,
  incumbentBytes: number = bytes.byteLength,
): RepeatedFactorResult | null {
  const n = bytes.byteLength;
  const headerBytes = 5; // 1B tag + 4B unitLen
  const maxPeriod = Math.min(n >>> 1, Math.max(0, incumbentBytes - headerBytes - 1));
  if (n < 16 || maxPeriod < 1) return null;

  // Fast 1-byte constant page check (extremely common in zero-padded KV regions)
  const b0 = bytes[0];
  let allSame = true;
  for (let i = 1; i < n; i++) {
    if (bytes[i] !== b0) {
      allSame = false;
      break;
    }
  }
  if (allSame && decideUorcChallengerAdmission(headerBytes + 1, incumbentBytes)) {
    return {
      kind: 'shifted',
      unit: bytes.slice(0, 1),
      halvings: 0,
      originalLength: n,
      encodedBytes: headerBytes + 1,
    };
  }

  // 8-byte prefix signature scan for exact period d
  const p0 = bytes[0];
  const p1 = bytes[1];
  const p2 = bytes[2];
  const p3 = bytes[3];
  const p4 = bytes[4];
  const p5 = bytes[5];
  const p6 = bytes[6];
  const p7 = bytes[7];

  for (let d = 2; d <= maxPeriod; d++) {
    if (d + 8 <= n) {
      if (
        bytes[d] !== p0 ||
        bytes[d + 1] !== p1 ||
        bytes[d + 2] !== p2 ||
        bytes[d + 3] !== p3 ||
        bytes[d + 4] !== p4 ||
        bytes[d + 5] !== p5 ||
        bytes[d + 6] !== p6 ||
        bytes[d + 7] !== p7
      ) {
        continue;
      }
    } else if (bytes[d] !== p0) {
      continue;
    }

    let periodic = true;
    for (let i = d; i < n; i++) {
      if (bytes[i] !== bytes[i - d]) {
        periodic = false;
        break;
      }
    }
    if (periodic) {
      const encodedBytes = headerBytes + d;
      if (decideUorcChallengerAdmission(encodedBytes, incumbentBytes)) {
        return {
          kind: 'shifted',
          unit: bytes.slice(0, d),
          halvings: 0,
          originalLength: n,
          encodedBytes,
        };
      }
      return null;
    }
  }

  return null;
}

/** Reconstructs an exact byte buffer from a repeated unit via doubling `copyWithin`. */
export function decodeRepeatedBytes(unit: Uint8Array, expectedLength: number): Uint8Array {
  const out = new Uint8Array(expectedLength);
  if (expectedLength === 0 || unit.byteLength === 0) return out;
  const firstLen = Math.min(unit.byteLength, expectedLength);
  out.set(unit.subarray(0, firstLen), 0);
  let filled = firstLen;
  while (filled < expectedLength) {
    const chunk = Math.min(filled, expectedLength - filled);
    out.copyWithin(filled, 0, chunk);
    filled += chunk;
  }
  return out;
}

// ---------------------------------------------------------------------------
// 3. 2-Symbol Column-Mask Selector (afflom/vRAM device/src/selector.rs)
// ---------------------------------------------------------------------------

export interface SelectorPageResult {
  readonly rowWidth: number;
  readonly rowCount: number;
  /** Shared 64-bit column selector mask: bit `c` picks `rowSymbols[2*r + 1]` vs `rowSymbols[2*r]`. */
  readonly columnsMask: bigint;
  /** Two symbols `(sym0, sym1)` per 64-byte row. */
  readonly rowSymbols: Uint8Array;
  readonly encodedBytes: number;
}

/**
 * Fits a 2-symbol shared-column-mask selector across 64-byte rows (`Selector::fit` in afflom/vRAM).
 * Achieves ~30× bit-exact lossless compression on structured/sparse pages (137 B vs 4,096 B).
 */
export function fitSelectorPage(
  bytes: Uint8Array,
  incumbentBytes: number = bytes.byteLength,
  rowWidth = 64,
): SelectorPageResult | null {
  if (rowWidth !== 64 || bytes.byteLength < rowWidth || bytes.byteLength % rowWidth !== 0) {
    return null;
  }
  const rowCount = bytes.byteLength / rowWidth;
  const encodedBytes = 1 + 8 + rowCount * 2; // 1B tag + 8B u64 mask + 2B per row
  if (!decideUorcChallengerAdmission(encodedBytes, incumbentBytes)) {
    return null;
  }

  const rowSymbols = new Uint8Array(rowCount * 2);
  let maskEstablished = false;
  let sharedMask = 0n;

  for (let r = 0; r < rowCount; r++) {
    const base = r * rowWidth;
    const sym0 = bytes[base];
    let sym1 = sym0;
    let hasSecond = false;
    let rowMask = 0n;

    for (let c = 1; c < rowWidth; c++) {
      const v = bytes[base + c];
      if (v === sym0) continue;
      if (!hasSecond) {
        sym1 = v;
        hasSecond = true;
        rowMask |= 1n << BigInt(c);
      } else if (v === sym1) {
        rowMask |= 1n << BigInt(c);
      } else {
        // More than 2 distinct byte symbols in this row -> reject immediately
        return null;
      }
    }

    if (hasSecond) {
      if (!maskEstablished) {
        sharedMask = rowMask;
        maskEstablished = true;
      } else if (rowMask !== sharedMask) {
        // Try swapped symbol polarity (sym0 <-> sym1) in case row inverted
        const fullMask = (1n << BigInt(rowWidth)) - 1n;
        if ((rowMask ^ fullMask) === sharedMask) {
          rowSymbols[r * 2] = sym1;
          rowSymbols[r * 2 + 1] = sym0;
          continue;
        }
        return null;
      }
    }

    rowSymbols[r * 2] = sym0;
    rowSymbols[r * 2 + 1] = sym1;
  }

  return {
    rowWidth,
    rowCount,
    columnsMask: sharedMask,
    rowSymbols,
    encodedBytes,
  };
}

/** Exact lossless decoder for a 2-symbol column-mask selector page. */
export function decodeSelectorPage(
  columnsMask: bigint,
  rowSymbols: Uint8Array,
  rowCount: number,
  rowWidth = 64,
): Uint8Array {
  const out = new Uint8Array(rowCount * rowWidth);
  // Precompute column bit flags once (0 or 1) to avoid BigInt ops in inner row loop
  const colSel = new Uint8Array(rowWidth);
  for (let c = 0; c < rowWidth; c++) {
    colSel[c] = Number((columnsMask >> BigInt(c)) & 1n);
  }
  for (let r = 0; r < rowCount; r++) {
    const sym0 = rowSymbols[r * 2];
    const sym1 = rowSymbols[r * 2 + 1];
    const base = r * rowWidth;
    for (let c = 0; c < rowWidth; c++) {
      out[base + c] = colSel[c] === 0 ? sym0 : sym1;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// 4. Unified KV Block Exact Lossless Codec (UKV2 payload compression)
// ---------------------------------------------------------------------------

export const enum UnifiedPageCodecTag {
  RAW = 0,
  DYADIC_REPEATED = 1,
  SELECTOR_2SYM = 2,
  SHIFTED_REPEATED = 3,
}

export interface CompressedPayloadResult {
  readonly codec: UnifiedPageCodecTag;
  readonly payload: Uint8Array;
  readonly rawBytes: number;
  readonly savedBytes: number;
}

/**
 * Compresses an arbitrary KV byte payload with 0% precision loss using afflom/vRAM's
 * challenger admission order: Dyadic Repeated -> 2-Symbol Selector -> Shifted Repeated -> Raw.
 */
export function compressUnifiedKvBytes(raw: Uint8Array): CompressedPayloadResult {
  const rawBytes = raw.byteLength;
  let bestCodec = UnifiedPageCodecTag.RAW;
  let bestBytes = rawBytes + 1;
  let bestPayload: Uint8Array | null = null;

  // 1. Dyadic exact repetition (O(N), ultra-fast)
  const dyadic = factorRepeatedBytes(raw);
  if (dyadic && dyadic.encodedBytes < bestBytes) {
    const out = new Uint8Array(5 + dyadic.unit.byteLength);
    out[0] = UnifiedPageCodecTag.DYADIC_REPEATED;
    new DataView(out.buffer).setUint32(1, dyadic.unit.byteLength, true);
    out.set(dyadic.unit, 5);
    bestCodec = UnifiedPageCodecTag.DYADIC_REPEATED;
    bestBytes = out.byteLength;
    bestPayload = out;
  }

  // 2. 2-Symbol shared column-mask selector (O(N), up to 30× lossless reduction)
  const selector = fitSelectorPage(raw, bestBytes, 64);
  if (selector && selector.encodedBytes < bestBytes) {
    const out = new Uint8Array(9 + selector.rowSymbols.byteLength);
    out[0] = UnifiedPageCodecTag.SELECTOR_2SYM;
    new DataView(out.buffer).setBigUint64(1, selector.columnsMask, true);
    out.set(selector.rowSymbols, 9);
    bestCodec = UnifiedPageCodecTag.SELECTOR_2SYM;
    bestBytes = out.byteLength;
    bestPayload = out;
  }

  // 3. Non-dyadic shifted repetition (only when raw is modest so scan stays sub-millisecond)
  if (rawBytes <= 16_384) {
    const shifted = shiftedRepeatedBytes(raw, bestBytes);
    if (shifted && shifted.encodedBytes < bestBytes) {
      const out = new Uint8Array(5 + shifted.unit.byteLength);
      out[0] = UnifiedPageCodecTag.SHIFTED_REPEATED;
      new DataView(out.buffer).setUint32(1, shifted.unit.byteLength, true);
      out.set(shifted.unit, 5);
      bestCodec = UnifiedPageCodecTag.SHIFTED_REPEATED;
      bestBytes = out.byteLength;
      bestPayload = out;
    }
  }

  if (bestPayload && bestPayload.byteLength < rawBytes) {
    return {
      codec: bestCodec,
      payload: bestPayload,
      rawBytes,
      savedBytes: rawBytes - bestPayload.byteLength,
    };
  }

  // Fallback: Raw payload prefixed with 1-byte RAW tag (no compression admitted)
  const rawTagged = new Uint8Array(1 + rawBytes);
  rawTagged[0] = UnifiedPageCodecTag.RAW;
  rawTagged.set(raw, 1);
  return {
    codec: UnifiedPageCodecTag.RAW,
    payload: rawTagged,
    rawBytes,
    savedBytes: 0,
  };
}

/** Exact lossless decoder for `compressUnifiedKvBytes`. */
export function decompressUnifiedKvBytes(
  payload: Uint8Array,
  expectedLength: number,
): Uint8Array {
  if (payload.byteLength === 0) {
    return new Uint8Array(expectedLength);
  }
  const tag = payload[0] as UnifiedPageCodecTag;
  if (tag === UnifiedPageCodecTag.RAW) {
    return payload.slice(1, 1 + expectedLength);
  }
  if (
    tag === UnifiedPageCodecTag.DYADIC_REPEATED ||
    tag === UnifiedPageCodecTag.SHIFTED_REPEATED
  ) {
    if (payload.byteLength < 5) {
      throw new Error('Truncated repeated KV page payload.');
    }
    const unitLen = new DataView(payload.buffer, payload.byteOffset, payload.byteLength).getUint32(
      1,
      true,
    );
    const unit = payload.subarray(5, 5 + unitLen);
    return decodeRepeatedBytes(unit, expectedLength);
  }
  if (tag === UnifiedPageCodecTag.SELECTOR_2SYM) {
    if (payload.byteLength < 9 || expectedLength % 64 !== 0) {
      throw new Error('Malformed selector KV page payload.');
    }
    const mask = new DataView(payload.buffer, payload.byteOffset, payload.byteLength).getBigUint64(
      1,
      true,
    );
    const rowCount = expectedLength / 64;
    const rowSymbols = payload.subarray(9, 9 + rowCount * 2);
    return decodeSelectorPage(mask, rowSymbols, rowCount, 64);
  }
  throw new Error(`Unknown unified KV codec tag: ${String(tag)}`);
}

// ---------------------------------------------------------------------------
// 5. Zero-Clamp MessageChannel Cooperative Yielder + FrameBudgetProfiler
// ---------------------------------------------------------------------------

export interface CooperativeYielder {
  readonly transport: 'message-channel' | 'timeout-fallback';
  yieldNow(): Promise<void>;
  dispose(): void;
}

export interface KvCheckpointDecisionInput {
  readonly checkpointRequested: boolean;
  readonly jobPartId: string;
  readonly focusedPartId: string;
  readonly hasWaitingFocusedJob: boolean;
  readonly emittedTokensInSlice: number;
  readonly totalEmittedTokens: number;
  readonly remainingMaxTokens: number;
}

/**
 * Pure 64-token KV page-boundary checkpoint predicate:
 * yields a long-running background part cleanly at an exact 64-token KV page boundary
 * when the user's focused foreground part is waiting, preserving all committed KV pages.
 */
export function shouldCheckpointAtKvPageBoundary(input: KvCheckpointDecisionInput): boolean {
  return (
    input.checkpointRequested &&
    input.hasWaitingFocusedJob &&
    input.jobPartId !== input.focusedPartId &&
    input.emittedTokensInSlice >= 64 &&
    input.totalEmittedTokens >= 128 &&
    input.totalEmittedTokens % 64 === 0 &&
    input.remainingMaxTokens >= 32
  );
}

/**
 * Creates a zero-clamp macrotask yielder using `MessageChannel`.
 *
 * Why this matters:
 * Chromium/WebKit clamp nested `setTimeout(fn, 0)` calls to >= 4,000 µs (4 ms) after 5 iterations,
 * capping fast client-side models at ~250 tok/s and adding 4 ms pure idle stall to EVERY token!
 * `MessageChannel` schedules a true macrotask turn (delivering `cancel` / `speculative_prefill`
 * worker messages immediately) in ~15–30 µs with ZERO 4 ms timer clamping!
 */
export function createCooperativeYielder(): CooperativeYielder {
  if (typeof MessageChannel !== 'undefined') {
    const channel = new MessageChannel();
    const port1 = channel.port1 as MessagePort & { ref?: () => void; unref?: () => void };
    const port2 = channel.port2 as MessagePort & { ref?: () => void; unref?: () => void };
    const queue: Array<() => void> = [];
    port1.onmessage = () => {
      const next = queue.shift();
      if (queue.length === 0) {
        port1.unref?.();
      }
      if (next) next();
    };
    port1.unref?.();
    port2.unref?.();
    return {
      transport: 'message-channel',
      yieldNow: () =>
        new Promise<void>((resolve) => {
          if (queue.length === 0) {
            port1.ref?.();
          }
          queue.push(resolve);
          port2.postMessage(undefined);
        }),
      dispose: () => {
        port1.onmessage = null;
        port1.unref?.();
        port2.unref?.();
        port1.close();
        port2.close();
        while (queue.length > 0) {
          queue.shift()?.();
        }
      },
    };
  }

  return {
    transport: 'timeout-fallback',
    yieldNow: () => new Promise<void>((resolve) => setTimeout(resolve, 0)),
    dispose: () => {},
  };
}

export interface FrameBudgetTelemetry {
  readonly frameBudgetUs: number;
  readonly yieldTransport: 'message-channel' | 'timeout-fallback';
  readonly medianStepUs: number;
  readonly lastSliceUs: number;
  readonly totalSteps: number;
  readonly yieldsExecuted: number;
  readonly yieldsAvoided: number;
  readonly withinFrameBudget: boolean;
}

/**
 * Verdant Engine FrameBudgetProfiler (16,666 µs KPI) coupled with vRAM SwapPolicy's
 * 3-sample median (`swapMiddle`) and `swapTwoWithin` stability proof.
 *
 * Eliminates unconditional per-token timer yields: yields cooperatively via zero-clamp
 * `MessageChannel` ONLY when accumulated compute slice time reaches the frame slice budget
 * (or every `maxStepsWithoutYield` steps) so cancel/preempt latency stays < 16.67 ms while
 * freeing 100% of CPU cycles for WASM SIMD token throughput!
 */
export class FrameBudgetProfiler {
  readonly frameBudgetUs: number;
  private readonly yieldTransport: 'message-channel' | 'timeout-fallback';
  private readonly recentStepsUs: [number, number, number] = [0, 0, 0];
  private sampleCount = 0;
  private totalSteps = 0;
  private yieldsExecuted = 0;
  private yieldsAvoided = 0;
  private lastSliceUs = 0;
  private sliceStartMs = 0;
  private stepsInSlice = 0;

  constructor(
    frameBudgetUs: number = FRAME_BUDGET_US,
    yieldTransport: 'message-channel' | 'timeout-fallback' =
      typeof MessageChannel !== 'undefined' ? 'message-channel' : 'timeout-fallback',
  ) {
    this.frameBudgetUs = frameBudgetUs;
    this.yieldTransport = yieldTransport;
    this.sliceStartMs = performance.now();
  }

  /** Resets the current work slice timer (call at start of a generation or prefill run). */
  beginSlice(nowMs: number = performance.now()): void {
    this.sliceStartMs = nowMs;
    this.stepsInSlice = 0;
  }

  /**
   * Records a completed token/batch step duration and returns whether the worker should
   * yield a zero-clamp `MessageChannel` turn right now to uphold the 16,666 µs KPI.
   */
  recordStepAndShouldYield(
    stepDurationUs: number,
    sliceBudgetUs: number = GENERATION_SLICE_BUDGET_US,
    maxStepsWithoutYield = 16,
    nowMs: number = performance.now(),
  ): boolean {
    const safeStepUs = Math.max(0, stepDurationUs);
    this.recentStepsUs[this.sampleCount % 3] = safeStepUs;
    this.sampleCount += 1;
    this.totalSteps += 1;
    this.stepsInSlice += 1;

    const sliceElapsedUs = Math.max(safeStepUs, (nowMs - this.sliceStartMs) * 1000);
    this.lastSliceUs = sliceElapsedUs;

    if (sliceElapsedUs >= sliceBudgetUs || this.stepsInSlice >= maxStepsWithoutYield) {
      this.yieldsExecuted += 1;
      this.sliceStartMs = nowMs;
      this.stepsInSlice = 0;
      return true;
    }

    this.yieldsAvoided += 1;
    return false;
  }

  getMedianStepUs(): number {
    if (this.sampleCount === 0) return 0;
    if (this.sampleCount === 1) return this.recentStepsUs[0];
    if (this.sampleCount === 2) return (this.recentStepsUs[0] + this.recentStepsUs[1]) * 0.5;
    return swapMiddle(this.recentStepsUs[0], this.recentStepsUs[1], this.recentStepsUs[2]);
  }

  isWithinFrameBudget(): boolean {
    if (this.sampleCount < 3) {
      return this.getMedianStepUs() <= this.frameBudgetUs;
    }
    return swapTwoWithin(
      this.recentStepsUs[0],
      this.recentStepsUs[1],
      this.recentStepsUs[2],
      this.frameBudgetUs,
    );
  }

  getTelemetry(): FrameBudgetTelemetry {
    return {
      frameBudgetUs: this.frameBudgetUs,
      yieldTransport: this.yieldTransport,
      medianStepUs: Math.round(this.getMedianStepUs()),
      lastSliceUs: Math.round(this.lastSliceUs),
      totalSteps: this.totalSteps,
      yieldsExecuted: this.yieldsExecuted,
      yieldsAvoided: this.yieldsAvoided,
      withinFrameBudget: this.isWithinFrameBudget(),
    };
  }
}
