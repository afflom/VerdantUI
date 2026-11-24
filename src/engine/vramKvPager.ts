/**
 * vRAM-Inspired 3-Tier Hot/Warm/Cold Paged KV-Cache & Automatic Reclaim Governor.
 *
 * Extrapolated directly from github.com/afflom/vRAM (device/DESIGN.md, device/OPTIMIZER.md,
 * device/CAPACITY.md, device/src/lifecycle.rs) and governed by Prism Theorem T4:
 * - Partitions KV context slots into fixed 16-token pages tracked by 64-bit word membership bitsets
 *   (`vramMembership`, mirroring vRAM's 64-frame residency groups).
 * - Tier 0 (Hot F32): Bounded working set (`H*` frames derived by Theorem T4: Page 0 attention sink
 *   + active tail pages) kept in F32 resident vTensor frames for zero-overhead writes and hot attention.
 * - Tier 1 (Warm Q8): When resident hot frames exceed `H*`, `vramTrim` + `reclaimColdestPage`
 *   captures the oldest non-sink hot page into symmetric per-head int8 (Q8, ~3.8x compression)
 *   verified by the vRAM Challenger Admission Gate (`admitColdKvPage`).
 * - Tier 2 (Cold Q4): When `coldQ4Enabled` (long context / large model scaling per Theorem T4) and
 *   Warm Q8 pages exceed `W*`, cascades the oldest Warm Q8 page into packed symmetric 4-bit Q4
 *   (~7.1x compression vs F32) verified by `admitColdQ4Page`, releasing the Warm Q8 slot!
 * - Zero-fault 3-Tier Attention: FLASH_ATTN_EXT attends directly across Hot F32 (`dot_f32`),
 *   Warm Q8 (`dot_q8_f32`), and Cold Q4 (`dot_q4_f32`) in WASM SIMD without expanding pages.
 * - Refault on Prefix Truncation (`vramFault`): truncating mid-page into either Warm Q8 or Cold Q4
 *   cleanly expands that page back into a resident Hot F32 frame.
 */

import { GgmlType } from './gguf';
import type { VTensorBuffer, VTensorHandle } from './vtensorProvider';
import {
  type VerdantWasmArena,
  WASM_ATTN_SCORES_OFFSET,
  WASM_ATTN_WEIGHTS_OFFSET,
} from './verdantWasmSimd';

export const VRAM_KV_PAGE_TOKENS = 16;

export const enum VramPageState {
  EMPTY = 0,
  HOT_RESIDENT = 1,
  COLD_COMPRESSED = 2, // Tier 1: Warm Q8
  COLD_Q4 = 3, // Tier 2: Cold Q4 (~7.1x compression)
}

export interface VramKvTelemetry {
  readonly pageTokens: number;
  readonly maxPages: number;
  readonly hotFrameLimit: number;
  readonly warmQ8Limit: number;
  readonly coldQ4Enabled: boolean;
  readonly hotResidentPages: number;
  /** Total non-hot compressed pages (Warm Q8 + Cold Q4) for backward compatibility. */
  readonly coldCompressedPages: number;
  readonly warmQ8Pages: number;
  readonly coldQ4Pages: number;
  readonly activePages: number;
  readonly rawEquivalentBytes: number;
  readonly allocatedBytes: number;
  readonly savedBytes: number;
  readonly compressionRatio: number;
  readonly reclaimCycles: number;
  readonly tierCascadeCycles: number;
  readonly refaultCount: number;
  readonly hotAttentionReads: number;
  readonly coldAttentionReads: number;
  readonly coldQ4AttentionReads: number;
  readonly residencyBitsetHex: string;
}

/**
 * Mirrors vRAM's `vram_membership(word, bit, present)` 64-bit residency bitset update.
 */
export function vramMembership(word: bigint, bitIndex: number, present: boolean): bigint {
  const mask = 1n << BigInt(bitIndex & 63);
  return present ? word | mask : word & ~mask;
}

/**
 * Mirrors vRAM's `vram_trim(limit, resident)` automatic reclaim governor decision.
 */
export function vramTrim(limit: number, resident: number): boolean {
  return resident > limit;
}

/**
 * Mirrors vRAM's challenger admission gate for Tier 1 (Warm Q8):
 * requires strict byte reduction and bounded quantization error before releasing the F32 frame.
 */
export function admitColdKvPage(
  rawF32Bytes: number,
  compressedBytes: number,
  maxAbsError: number,
  maxAbsSignal: number,
): boolean {
  if (compressedBytes >= rawF32Bytes) return false;
  if (!Number.isFinite(maxAbsError)) return false;
  // Symmetric int8 max theoretical step error is scale/2 = maxAbs / 254; allow small FP margin
  const tolerance = Math.max(1e-4, maxAbsSignal * 0.012);
  return maxAbsError <= tolerance;
}

/**
 * Mirrors vRAM's challenger admission gate for Tier 2 (Warm Q8 -> Cold Q4 cascade):
 * requires strict byte reduction vs Q8 and bounded 4-bit symmetric reconstruction error.
 */
export function admitColdQ4Page(
  warmQ8Bytes: number,
  coldQ4Bytes: number,
  maxAbsError: number,
  maxAbsSignal: number,
): boolean {
  if (coldQ4Bytes >= warmQ8Bytes) return false;
  if (!Number.isFinite(maxAbsError)) return false;
  // Symmetric 4-bit [-7, 7] max theoretical step error is scale/2 = maxAbs / 14 (~7.14%); allow margin
  const tolerance = Math.max(5e-4, maxAbsSignal * 0.095);
  return maxAbsError <= tolerance;
}

export interface VramKvPagerConfig {
  readonly numLayers: number;
  readonly maxContext: number;
  readonly nHeadsKv: number;
  readonly headDim: number;
  readonly hotFrameLimit?: number;
  readonly warmQ8Limit?: number;
  readonly coldQ4Enabled?: boolean;
}

export class VramKvPager {
  readonly numLayers: number;
  readonly maxContext: number;
  readonly nHeadsKv: number;
  readonly headDim: number;
  readonly kvDim: number;
  readonly maxPages: number;
  readonly hotFrameLimit: number;
  readonly warmQ8Limit: number;
  readonly coldQ4Enabled: boolean;

  // Per-page geometry (for K or V individually)
  readonly hotPageFloats: number; // numLayers * PAGE_TOKENS * kvDim
  readonly coldPageQ8Bytes: number; // numLayers * PAGE_TOKENS * kvDim (int8)
  readonly coldPageQ4Bytes: number; // numLayers * PAGE_TOKENS * ceil(kvDim / 2) (packed u4)
  readonly coldPageScaleFloats: number; // numLayers * PAGE_TOKENS * nHeadsKv (f32 scales)

  private readonly warmQ8PoolSize: number;
  private readonly coldQ4PoolSize: number;
  private readonly headQ4Bytes: number;

  // Backing vTensor handles (live directly inside WASM linear memory when SIMD arena is active)
  private readonly hotKHandle: VTensorHandle;
  private readonly hotVHandle: VTensorHandle;
  private readonly coldKQ8Handle: VTensorHandle;
  private readonly coldVQ8Handle: VTensorHandle;
  private readonly coldKScaleHandle: VTensorHandle;
  private readonly coldVScaleHandle: VTensorHandle;

  // Optional Tier-2 Cold Q4 backing handles (allocated only when coldQ4Enabled is true)
  private readonly coldKQ4Handle: VTensorHandle | null;
  private readonly coldVQ4Handle: VTensorHandle | null;
  private readonly coldKQ4ScaleHandle: VTensorHandle | null;
  private readonly coldVQ4ScaleHandle: VTensorHandle | null;

  // Verdant WASM-resident KV page directory (i32[maxPages] state & active slot) for 1-call whole-layer FlashAttention
  private readonly wasmPageStateHandle: VTensorHandle;
  private readonly wasmPageSlotHandle: VTensorHandle;

  // Page directory & 64-bit residency bitsets (vram_membership groups)
  private readonly pageState: Uint8Array;
  private readonly pageHotSlot: Int32Array;
  private readonly pageColdSlot: Int32Array;
  private readonly pageQ4Slot: Int32Array;
  private readonly freeHotSlots: number[] = [];
  private readonly freeColdSlots: number[] = [];
  private readonly freeQ4Slots: number[] = [];
  private readonly hotBitsets: BigUint64Array;
  private readonly coldBitsets: BigUint64Array;
  private readonly coldQ4Bitsets: BigUint64Array;

  private hotResidentCount = 0;
  private warmQ8Count = 0;
  private coldQ4Count = 0;
  private reclaimCycles = 0;
  private tierCascadeCycles = 0;
  private refaultCount = 0;
  private hotAttentionReads = 0;
  private coldAttentionReads = 0;
  private coldQ4AttentionReads = 0;

  constructor(
    config: VramKvPagerConfig,
    allocBuffer: (id: string, usage: 'kv_cache', capacityBytes: number) => VTensorBuffer,
  ) {
    this.numLayers = config.numLayers;
    this.maxContext = config.maxContext;
    this.nHeadsKv = config.nHeadsKv;
    this.headDim = config.headDim;
    this.kvDim = config.nHeadsKv * config.headDim;
    this.maxPages = Math.max(1, Math.ceil(config.maxContext / VRAM_KV_PAGE_TOKENS));

    // Contract-driven limits (or clean conservative fallback when instantiated standalone)
    const defaultHot = Math.min(this.maxPages, Math.max(4, Math.min(8, Math.ceil(Math.sqrt(this.maxPages)))));
    this.hotFrameLimit = Math.max(2, Math.min(this.maxPages, config.hotFrameLimit ?? defaultHot));

    const wantColdQ4 = Boolean(config.coldQ4Enabled) && this.headDim >= 2 && (this.headDim & 1) === 0;
    const requestedWarm = config.warmQ8Limit ?? this.maxPages;
    this.coldQ4Enabled = wantColdQ4 && this.maxPages > this.hotFrameLimit + Math.max(1, requestedWarm);
    this.warmQ8Limit = this.coldQ4Enabled
      ? Math.max(1, Math.min(this.maxPages - this.hotFrameLimit, requestedWarm))
      : this.maxPages;

    this.headQ4Bytes = this.headDim >>> 1;
    this.hotPageFloats = this.numLayers * VRAM_KV_PAGE_TOKENS * this.kvDim;
    this.coldPageQ8Bytes = this.numLayers * VRAM_KV_PAGE_TOKENS * this.kvDim;
    this.coldPageQ4Bytes = this.numLayers * VRAM_KV_PAGE_TOKENS * this.nHeadsKv * this.headQ4Bytes;
    this.coldPageScaleFloats = this.numLayers * VRAM_KV_PAGE_TOKENS * this.nHeadsKv;

    // Warm Q8 pool only needs warmQ8Limit + 1 transient slot when Cold Q4 handles overflow
    this.warmQ8PoolSize = this.coldQ4Enabled
      ? Math.min(this.maxPages, this.warmQ8Limit + 1)
      : Math.max(1, this.maxPages);
    this.coldQ4PoolSize = this.coldQ4Enabled
      ? Math.max(1, this.maxPages - this.hotFrameLimit)
      : 0;

    const align32 = (n: number) => ((n + 31) & ~31);

    // 1. Tier-0 Hot Resident F32 frame pool (only `hotFrameLimit` pages in F32!)
    const hotFrameBytes = this.hotFrameLimit * this.hotPageFloats * 4;
    const hotBuf = allocBuffer(
      'vtensor.kv_hot_frames',
      'kv_cache',
      align32(hotFrameBytes) * 2 + 128,
    );
    this.hotKHandle = hotBuf.allocTensor('kv.hot_k', GgmlType.F32, [
      this.kvDim,
      VRAM_KV_PAGE_TOKENS,
      this.numLayers * this.hotFrameLimit,
    ]);
    this.hotVHandle = hotBuf.allocTensor('kv.hot_v', GgmlType.F32, [
      this.kvDim,
      VRAM_KV_PAGE_TOKENS,
      this.numLayers * this.hotFrameLimit,
    ]);

    // 2. Tier-1 Warm Q8 compressed backing store
    const warmQ8BytesTotal = this.warmQ8PoolSize * this.coldPageQ8Bytes;
    const warmScaleBytesTotal = this.warmQ8PoolSize * this.coldPageScaleFloats * 4;
    const warmBuf = allocBuffer(
      'vtensor.vram_cold_kv',
      'kv_cache',
      align32(warmQ8BytesTotal) * 2 + align32(warmScaleBytesTotal) * 2 + 256,
    );
    this.coldKQ8Handle = warmBuf.allocTensor('kv.cold_k_q8', GgmlType.I8, [
      this.kvDim,
      VRAM_KV_PAGE_TOKENS,
      this.numLayers * this.warmQ8PoolSize,
    ]);
    this.coldVQ8Handle = warmBuf.allocTensor('kv.cold_v_q8', GgmlType.I8, [
      this.kvDim,
      VRAM_KV_PAGE_TOKENS,
      this.numLayers * this.warmQ8PoolSize,
    ]);
    this.coldKScaleHandle = warmBuf.allocTensor('kv.cold_k_scale', GgmlType.F32, [
      this.nHeadsKv,
      VRAM_KV_PAGE_TOKENS,
      this.numLayers * this.warmQ8PoolSize,
    ]);
    this.coldVScaleHandle = warmBuf.allocTensor('kv.cold_v_scale', GgmlType.F32, [
      this.nHeadsKv,
      VRAM_KV_PAGE_TOKENS,
      this.numLayers * this.warmQ8PoolSize,
    ]);

    // 3. Tier-2 Cold Q4 backing store (enabled automatically by Theorem T4 for long context / large models)
    if (this.coldQ4Enabled && this.coldQ4PoolSize > 0) {
      const q4BytesTotal = this.coldQ4PoolSize * this.coldPageQ4Bytes;
      const q4ScaleTotal = this.coldQ4PoolSize * this.coldPageScaleFloats * 4;
      const q4Buf = allocBuffer(
        'vtensor.vram_cold_q4',
        'kv_cache',
        align32(q4BytesTotal) * 2 + align32(q4ScaleTotal) * 2 + 256,
      );
      this.coldKQ4Handle = q4Buf.allocTensor('kv.cold_k_q4', GgmlType.I8, [
        this.nHeadsKv * this.headQ4Bytes,
        VRAM_KV_PAGE_TOKENS,
        this.numLayers * this.coldQ4PoolSize,
      ]);
      this.coldVQ4Handle = q4Buf.allocTensor('kv.cold_v_q4', GgmlType.I8, [
        this.nHeadsKv * this.headQ4Bytes,
        VRAM_KV_PAGE_TOKENS,
        this.numLayers * this.coldQ4PoolSize,
      ]);
      this.coldKQ4ScaleHandle = q4Buf.allocTensor('kv.cold_k_q4_scale', GgmlType.F32, [
        this.nHeadsKv,
        VRAM_KV_PAGE_TOKENS,
        this.numLayers * this.coldQ4PoolSize,
      ]);
      this.coldVQ4ScaleHandle = q4Buf.allocTensor('kv.cold_v_q4_scale', GgmlType.F32, [
        this.nHeadsKv,
        VRAM_KV_PAGE_TOKENS,
        this.numLayers * this.coldQ4PoolSize,
      ]);
    } else {
      this.coldKQ4Handle = null;
      this.coldVQ4Handle = null;
      this.coldKQ4ScaleHandle = null;
      this.coldVQ4ScaleHandle = null;
    }

    // 4. Verdant WASM-resident page directory (state + active slot per page for zero-JS-loop flash_attn_layer)
    const ptBytes = align32(this.maxPages * 4);
    const ptBuf = allocBuffer('vtensor.vram_kv_pagetable', 'kv_cache', ptBytes * 2 + 128);
    this.wasmPageStateHandle = ptBuf.allocTensor('kv.pt_state', GgmlType.F32, [this.maxPages]);
    this.wasmPageSlotHandle = ptBuf.allocTensor('kv.pt_slot', GgmlType.F32, [this.maxPages]);

    this.pageState = new Uint8Array(this.maxPages);
    this.pageHotSlot = new Int32Array(this.maxPages).fill(-1);
    this.pageColdSlot = new Int32Array(this.maxPages).fill(-1);
    this.pageQ4Slot = new Int32Array(this.maxPages).fill(-1);

    const numGroups = Math.max(1, Math.ceil(this.maxPages / 64));
    this.hotBitsets = new BigUint64Array(numGroups);
    this.coldBitsets = new BigUint64Array(numGroups);
    this.coldQ4Bitsets = new BigUint64Array(numGroups);

    this.reset();
  }

  private syncWasmPageEntry(pageIdx: number, state: VramPageState, slot: number): void {
    const raw = this.wasmPageStateHandle.buffer.rawBytes();
    const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    dv.setInt32(this.wasmPageStateHandle.byteOffset + (pageIdx << 2), state, true);
    dv.setInt32(this.wasmPageSlotHandle.byteOffset + (pageIdx << 2), slot, true);
  }

  reset(): void {
    this.pageState.fill(VramPageState.EMPTY);
    this.pageHotSlot.fill(-1);
    this.pageColdSlot.fill(-1);
    this.pageQ4Slot.fill(-1);
    const rawPt = this.wasmPageStateHandle.buffer.rawBytes();
    const dvPt = new DataView(rawPt.buffer, rawPt.byteOffset, rawPt.byteLength);
    for (let p = 0; p < this.maxPages; p++) {
      dvPt.setInt32(this.wasmPageStateHandle.byteOffset + (p << 2), VramPageState.EMPTY, true);
      dvPt.setInt32(this.wasmPageSlotHandle.byteOffset + (p << 2), -1, true);
    }
    this.hotBitsets.fill(0n);
    this.coldBitsets.fill(0n);
    this.coldQ4Bitsets.fill(0n);

    this.freeHotSlots.length = 0;
    for (let s = this.hotFrameLimit - 1; s >= 0; s--) {
      this.freeHotSlots.push(s);
    }
    this.freeColdSlots.length = 0;
    for (let s = this.warmQ8PoolSize - 1; s >= 0; s--) {
      this.freeColdSlots.push(s);
    }
    this.freeQ4Slots.length = 0;
    for (let s = this.coldQ4PoolSize - 1; s >= 0; s--) {
      this.freeQ4Slots.push(s);
    }

    this.hotResidentCount = 0;
    this.warmQ8Count = 0;
    this.coldQ4Count = 0;
    this.reclaimCycles = 0;
    this.tierCascadeCycles = 0;
    this.refaultCount = 0;
    this.hotAttentionReads = 0;
    this.coldAttentionReads = 0;
    this.coldQ4AttentionReads = 0;
  }

  private setHotBit(pageIdx: number, present: boolean): void {
    const g = pageIdx >>> 6;
    this.hotBitsets[g] = vramMembership(this.hotBitsets[g], pageIdx & 63, present);
  }

  private setColdBit(pageIdx: number, present: boolean): void {
    const g = pageIdx >>> 6;
    this.coldBitsets[g] = vramMembership(this.coldBitsets[g], pageIdx & 63, present);
  }

  private setQ4Bit(pageIdx: number, present: boolean): void {
    const g = pageIdx >>> 6;
    this.coldQ4Bitsets[g] = vramMembership(this.coldQ4Bitsets[g], pageIdx & 63, present);
  }

  /**
   * Ensures `pageIdx` is resident in a hot F32 frame (allocating a free frame or running
   * vRAM's automatic reclaim governor + Tier-1->Tier-2 cascade if needed).
   * If the page was previously Warm Q8 or Cold Q4, performs a vRAM refault back to F32!
   */
  ensurePageHot(pageIdx: number): number {
    if (this.pageState[pageIdx] === VramPageState.HOT_RESIDENT) {
      return this.pageHotSlot[pageIdx];
    }

    // Trigger vRAM automatic reclaim governor if no free hot slots remain
    if (this.freeHotSlots.length === 0 || vramTrim(this.hotFrameLimit, this.hotResidentCount + 1)) {
      this.reclaimColdestPage(pageIdx);
    }

    const hotSlot = this.freeHotSlots.pop();
    if (hotSlot === undefined) {
      throw new Error(`vRAM KV pager exhausted hot frame pool (${this.hotFrameLimit} frames).`);
    }

    if (this.pageState[pageIdx] === VramPageState.COLD_COMPRESSED) {
      // Refault from Tier-1 Warm Q8 back into hot F32 frame
      const coldSlot = this.pageColdSlot[pageIdx];
      this.expandColdToHot(coldSlot, hotSlot);
      this.freeColdSlots.push(coldSlot);
      this.pageColdSlot[pageIdx] = -1;
      this.warmQ8Count -= 1;
      this.setColdBit(pageIdx, false);
      this.refaultCount += 1;
    } else if (this.pageState[pageIdx] === VramPageState.COLD_Q4) {
      // Refault from Tier-2 Cold Q4 back into hot F32 frame
      const q4Slot = this.pageQ4Slot[pageIdx];
      this.expandColdQ4ToHot(q4Slot, hotSlot);
      this.freeQ4Slots.push(q4Slot);
      this.pageQ4Slot[pageIdx] = -1;
      this.coldQ4Count -= 1;
      this.setQ4Bit(pageIdx, false);
      this.refaultCount += 1;
    }

    this.pageState[pageIdx] = VramPageState.HOT_RESIDENT;
    this.pageHotSlot[pageIdx] = hotSlot;
    this.syncWasmPageEntry(pageIdx, VramPageState.HOT_RESIDENT, hotSlot);
    this.hotResidentCount += 1;
    this.setHotBit(pageIdx, true);
    return hotSlot;
  }

  /**
   * Reclaims the oldest non-sink, non-active resident hot page into Tier-1 Warm Q8 backing.
   * When Tier-2 Cold Q4 is enabled and Warm Q8 exceeds `warmQ8Limit`, cascades the oldest Warm Q8
   * page down to Cold Q4 first so a Warm Q8 slot is always free.
   */
  private reclaimColdestPage(protectPageIdx: number): void {
    // Pin page 0 as the attention sink when possible; evict lowest pageIdx > 0 that isn't protected
    let victim = -1;
    for (let p = 1; p < this.maxPages; p++) {
      if (p !== protectPageIdx && this.pageState[p] === VramPageState.HOT_RESIDENT) {
        victim = p;
        break;
      }
    }
    if (victim === -1) {
      if (protectPageIdx !== 0 && this.pageState[0] === VramPageState.HOT_RESIDENT) {
        victim = 0;
      } else {
        return;
      }
    }

    // Ensure a Warm Q8 slot is available (cascading oldest Warm Q8 -> Cold Q4 when over warmQ8Limit)
    if (
      this.coldQ4Enabled &&
      (this.freeColdSlots.length === 0 || vramTrim(this.warmQ8Limit, this.warmQ8Count + 1))
    ) {
      this.cascadeWarmToColdQ4(protectPageIdx);
    }

    const hotSlot = this.pageHotSlot[victim];
    const coldSlot = this.freeColdSlots.pop();
    if (coldSlot === undefined) return;

    const admitted = this.captureHotToCold(hotSlot, coldSlot);
    if (!admitted) {
      this.freeColdSlots.push(coldSlot);
      return;
    }

    this.pageState[victim] = VramPageState.COLD_COMPRESSED;
    this.pageHotSlot[victim] = -1;
    this.pageColdSlot[victim] = coldSlot;
    this.syncWasmPageEntry(victim, VramPageState.COLD_COMPRESSED, coldSlot);
    this.hotResidentCount -= 1;
    this.warmQ8Count += 1;
    this.setHotBit(victim, false);
    this.setColdBit(victim, true);
    this.freeHotSlots.push(hotSlot);
    this.reclaimCycles += 1;

    // Enforce warmQ8Limit after admission as well
    if (this.coldQ4Enabled && vramTrim(this.warmQ8Limit, this.warmQ8Count)) {
      this.cascadeWarmToColdQ4(protectPageIdx);
    }
  }

  /**
   * Tier-1 -> Tier-2 Cascade: compresses the oldest Warm Q8 page into packed 4-bit Cold Q4
   * (~7.1x vs F32), verified by the vRAM Cold-Q4 challenger admission gate.
   */
  private cascadeWarmToColdQ4(protectPageIdx: number): void {
    if (!this.coldQ4Enabled || !this.coldKQ4Handle) return;
    let victim = -1;
    for (let p = 1; p < this.maxPages; p++) {
      if (p !== protectPageIdx && this.pageState[p] === VramPageState.COLD_COMPRESSED) {
        victim = p;
        break;
      }
    }
    if (victim === -1) return;

    const q4Slot = this.freeQ4Slots.pop();
    if (q4Slot === undefined) return;

    const warmSlot = this.pageColdSlot[victim];
    const admitted = this.captureWarmQ8ToColdQ4(warmSlot, q4Slot);
    if (!admitted) {
      this.freeQ4Slots.push(q4Slot);
      return;
    }

    this.pageState[victim] = VramPageState.COLD_Q4;
    this.pageColdSlot[victim] = -1;
    this.pageQ4Slot[victim] = q4Slot;
    this.syncWasmPageEntry(victim, VramPageState.COLD_Q4, q4Slot);
    this.warmQ8Count -= 1;
    this.coldQ4Count += 1;
    this.setColdBit(victim, false);
    this.setQ4Bit(victim, true);
    this.freeColdSlots.push(warmSlot);
    this.tierCascadeCycles += 1;
  }

  /** Compresses one F32 hot page across all layers into symmetric per-head int8 (Warm Q8) backing. */
  private captureHotToCold(hotSlot: number, coldSlot: number): boolean {
    const hotK = this.hotKHandle.buffer.f32View(this.hotKHandle);
    const hotV = this.hotVHandle.buffer.f32View(this.hotVHandle);
    const coldKQ8 = this.coldKQ8Handle.buffer.i8View(this.coldKQ8Handle);
    const coldVQ8 = this.coldVQ8Handle.buffer.i8View(this.coldVQ8Handle);
    const coldKSc = this.coldKScaleHandle.buffer.f32View(this.coldKScaleHandle);
    const coldVSc = this.coldVScaleHandle.buffer.f32View(this.coldVScaleHandle);

    const headDim = this.headDim;
    const totalVectors = this.numLayers * VRAM_KV_PAGE_TOKENS * this.nHeadsKv;
    const hotBase = hotSlot * this.hotPageFloats;
    const coldQ8Base = coldSlot * this.coldPageQ8Bytes;
    const coldScBase = coldSlot * this.coldPageScaleFloats;

    let maxErr = 0;
    let maxSig = 0;

    for (let vec = 0; vec < totalVectors; vec++) {
      const fOff = hotBase + vec * headDim;
      const qOff = coldQ8Base + vec * headDim;

      // Quantize K head vector
      let kMax = 0;
      for (let d = 0; d < headDim; d++) {
        const a = Math.abs(hotK[fOff + d]);
        if (a > kMax) kMax = a;
      }
      if (kMax > maxSig) maxSig = kMax;
      const kScale = kMax > 0 ? kMax / 127.0 : 0.0;
      const kInv = kMax > 0 ? 127.0 / kMax : 0.0;
      coldKSc[coldScBase + vec] = kScale;
      for (let d = 0; d < headDim; d++) {
        const orig = hotK[fOff + d];
        const q = Math.max(-127, Math.min(127, Math.round(orig * kInv)));
        coldKQ8[qOff + d] = q;
        const err = Math.abs(orig - q * kScale);
        if (err > maxErr) maxErr = err;
      }

      // Quantize V head vector
      let vMax = 0;
      for (let d = 0; d < headDim; d++) {
        const a = Math.abs(hotV[fOff + d]);
        if (a > vMax) vMax = a;
      }
      if (vMax > maxSig) maxSig = vMax;
      const vScale = vMax > 0 ? vMax / 127.0 : 0.0;
      const vInv = vMax > 0 ? 127.0 / vMax : 0.0;
      coldVSc[coldScBase + vec] = vScale;
      for (let d = 0; d < headDim; d++) {
        const orig = hotV[fOff + d];
        const q = Math.max(-127, Math.min(127, Math.round(orig * vInv)));
        coldVQ8[qOff + d] = q;
        const err = Math.abs(orig - q * vScale);
        if (err > maxErr) maxErr = err;
      }
    }

    const rawBytes = this.hotPageFloats * 4 * 2;
    const compressedBytes = (this.coldPageQ8Bytes + this.coldPageScaleFloats * 4) * 2;
    return admitColdKvPage(rawBytes, compressedBytes, maxErr, maxSig);
  }

  /**
   * Compresses one Warm Q8 page into packed 4-bit Cold Q4 backing.
   * Split-nibble layout matches WASM `dot_q4_f32` / `axpy_q4_f32`:
   * For every 32-element chunk (or half-headDim if headDim < 32):
   *   low nibbles of bytes [0..half-1] store elements [0..half-1] biased by +8 (u4 in [1..15])
   *   high nibbles of bytes [0..half-1] store elements [half..2*half-1] biased by +8.
   */
  private captureWarmQ8ToColdQ4(warmSlot: number, q4Slot: number): boolean {
    const coldKQ8 = this.coldKQ8Handle.buffer.i8View(this.coldKQ8Handle);
    const coldVQ8 = this.coldVQ8Handle.buffer.i8View(this.coldVQ8Handle);
    const coldKSc = this.coldKScaleHandle.buffer.f32View(this.coldKScaleHandle);
    const coldVSc = this.coldVScaleHandle.buffer.f32View(this.coldVScaleHandle);

    const kQ4Raw = this.coldKQ4Handle!.buffer.rawBytes();
    const vQ4Raw = this.coldVQ4Handle!.buffer.rawBytes();
    const kQ4BaseByte = this.coldKQ4Handle!.byteOffset + q4Slot * this.coldPageQ4Bytes;
    const vQ4BaseByte = this.coldVQ4Handle!.byteOffset + q4Slot * this.coldPageQ4Bytes;
    const kQ4Sc = this.coldKQ4ScaleHandle!.buffer.f32View(this.coldKQ4ScaleHandle!);
    const vQ4Sc = this.coldVQ4ScaleHandle!.buffer.f32View(this.coldVQ4ScaleHandle!);

    const headDim = this.headDim;
    const headQ4Bytes = this.headQ4Bytes;
    const chunkSpan = headDim >= 32 && (headDim & 31) === 0 ? 32 : headDim;
    const halfSpan = chunkSpan >>> 1;
    const totalVectors = this.numLayers * VRAM_KV_PAGE_TOKENS * this.nHeadsKv;

    const warmQ8Base = warmSlot * this.coldPageQ8Bytes;
    const warmScBase = warmSlot * this.coldPageScaleFloats;
    const q4ScBase = q4Slot * this.coldPageScaleFloats;

    let maxErr = 0;
    let maxSig = 0;

    const packVec = (
      srcQ8: Int8Array,
      srcOff: number,
      srcScale: number,
      dstBytes: Uint8Array,
      dstOff: number,
    ): number => {
      let maxAbs = 0;
      for (let d = 0; d < headDim; d++) {
        const v = Math.abs(srcQ8[srcOff + d] * srcScale);
        if (v > maxAbs) maxAbs = v;
      }
      if (maxAbs > maxSig) maxSig = maxAbs;
      const scale4 = maxAbs > 0 ? maxAbs / 7.0 : 0.0;
      const inv4 = maxAbs > 0 ? 7.0 / maxAbs : 0.0;

      let outByte = dstOff;
      for (let c = 0; c < headDim; c += chunkSpan) {
        for (let j = 0; j < halfSpan; j++) {
          const origLo = srcQ8[srcOff + c + j] * srcScale;
          const origHi = srcQ8[srcOff + c + halfSpan + j] * srcScale;
          const qLo = Math.max(-7, Math.min(7, Math.round(origLo * inv4)));
          const qHi = Math.max(-7, Math.min(7, Math.round(origHi * inv4)));
          dstBytes[outByte++] = ((qLo + 8) & 0x0f) | (((qHi + 8) & 0x0f) << 4);
          const eLo = Math.abs(origLo - qLo * scale4);
          const eHi = Math.abs(origHi - qHi * scale4);
          if (eLo > maxErr) maxErr = eLo;
          if (eHi > maxErr) maxErr = eHi;
        }
      }
      return scale4;
    };

    for (let vec = 0; vec < totalVectors; vec++) {
      const q8Off = warmQ8Base + vec * headDim;
      const q4Off = vec * headQ4Bytes;
      kQ4Sc[q4ScBase + vec] = packVec(
        coldKQ8,
        q8Off,
        coldKSc[warmScBase + vec],
        kQ4Raw,
        kQ4BaseByte + q4Off,
      );
      vQ4Sc[q4ScBase + vec] = packVec(
        coldVQ8,
        q8Off,
        coldVSc[warmScBase + vec],
        vQ4Raw,
        vQ4BaseByte + q4Off,
      );
    }

    const warmBytes = (this.coldPageQ8Bytes + this.coldPageScaleFloats * 4) * 2;
    const coldQ4Bytes = (this.coldPageQ4Bytes + this.coldPageScaleFloats * 4) * 2;
    return admitColdQ4Page(warmBytes, coldQ4Bytes, maxErr, maxSig);
  }

  /** Refault path (Tier-1 Warm Q8 -> Tier-0 Hot F32). */
  private expandColdToHot(coldSlot: number, hotSlot: number): void {
    const hotK = this.hotKHandle.buffer.f32View(this.hotKHandle);
    const hotV = this.hotVHandle.buffer.f32View(this.hotVHandle);
    const coldKQ8 = this.coldKQ8Handle.buffer.i8View(this.coldKQ8Handle);
    const coldVQ8 = this.coldVQ8Handle.buffer.i8View(this.coldVQ8Handle);
    const coldKSc = this.coldKScaleHandle.buffer.f32View(this.coldKScaleHandle);
    const coldVSc = this.coldVScaleHandle.buffer.f32View(this.coldVScaleHandle);

    const headDim = this.headDim;
    const totalVectors = this.numLayers * VRAM_KV_PAGE_TOKENS * this.nHeadsKv;
    const hotBase = hotSlot * this.hotPageFloats;
    const coldQ8Base = coldSlot * this.coldPageQ8Bytes;
    const coldScBase = coldSlot * this.coldPageScaleFloats;

    for (let vec = 0; vec < totalVectors; vec++) {
      const fOff = hotBase + vec * headDim;
      const qOff = coldQ8Base + vec * headDim;
      const ks = coldKSc[coldScBase + vec];
      const vs = coldVSc[coldScBase + vec];
      for (let d = 0; d < headDim; d++) {
        hotK[fOff + d] = coldKQ8[qOff + d] * ks;
        hotV[fOff + d] = coldVQ8[qOff + d] * vs;
      }
    }
  }

  /** Refault path (Tier-2 Cold Q4 -> Tier-0 Hot F32). */
  private expandColdQ4ToHot(q4Slot: number, hotSlot: number): void {
    const hotK = this.hotKHandle.buffer.f32View(this.hotKHandle);
    const hotV = this.hotVHandle.buffer.f32View(this.hotVHandle);
    const kQ4Raw = this.coldKQ4Handle!.buffer.rawBytes();
    const vQ4Raw = this.coldVQ4Handle!.buffer.rawBytes();
    const kQ4BaseByte = this.coldKQ4Handle!.byteOffset + q4Slot * this.coldPageQ4Bytes;
    const vQ4BaseByte = this.coldVQ4Handle!.byteOffset + q4Slot * this.coldPageQ4Bytes;
    const kQ4Sc = this.coldKQ4ScaleHandle!.buffer.f32View(this.coldKQ4ScaleHandle!);
    const vQ4Sc = this.coldVQ4ScaleHandle!.buffer.f32View(this.coldVQ4ScaleHandle!);

    const headDim = this.headDim;
    const headQ4Bytes = this.headQ4Bytes;
    const chunkSpan = headDim >= 32 && (headDim & 31) === 0 ? 32 : headDim;
    const halfSpan = chunkSpan >>> 1;
    const totalVectors = this.numLayers * VRAM_KV_PAGE_TOKENS * this.nHeadsKv;
    const hotBase = hotSlot * this.hotPageFloats;
    const q4ScBase = q4Slot * this.coldPageScaleFloats;

    for (let vec = 0; vec < totalVectors; vec++) {
      const fOff = hotBase + vec * headDim;
      const ks = kQ4Sc[q4ScBase + vec];
      const vs = vQ4Sc[q4ScBase + vec];
      let bOffK = kQ4BaseByte + vec * headQ4Bytes;
      let bOffV = vQ4BaseByte + vec * headQ4Bytes;

      for (let c = 0; c < headDim; c += chunkSpan) {
        for (let j = 0; j < halfSpan; j++) {
          const bk = kQ4Raw[bOffK++];
          const bv = vQ4Raw[bOffV++];
          hotK[fOff + c + j] = ((bk & 0x0f) - 8) * ks;
          hotK[fOff + c + halfSpan + j] = ((bk >>> 4) - 8) * ks;
          hotV[fOff + c + j] = ((bv & 0x0f) - 8) * vs;
          hotV[fOff + c + halfSpan + j] = ((bv >>> 4) - 8) * vs;
        }
      }
    }
  }

  /**
   * Truncates the paged KV cache after `validTokens` (used on multi-turn prefix reuse divergence).
   * Releases any pages strictly past the prefix across all 3 tiers and ensures the active write page is hot.
   */
  truncateTo(validTokens: number): void {
    const keepPages = validTokens > 0 ? Math.ceil(validTokens / VRAM_KV_PAGE_TOKENS) : 0;
    for (let p = keepPages; p < this.maxPages; p++) {
      const st = this.pageState[p];
      if (st === VramPageState.HOT_RESIDENT) {
        this.freeHotSlots.push(this.pageHotSlot[p]);
        this.pageHotSlot[p] = -1;
        this.pageState[p] = VramPageState.EMPTY;
        this.hotResidentCount -= 1;
        this.setHotBit(p, false);
      } else if (st === VramPageState.COLD_COMPRESSED) {
        this.freeColdSlots.push(this.pageColdSlot[p]);
        this.pageColdSlot[p] = -1;
        this.pageState[p] = VramPageState.EMPTY;
        this.warmQ8Count -= 1;
        this.setColdBit(p, false);
      } else if (st === VramPageState.COLD_Q4) {
        this.freeQ4Slots.push(this.pageQ4Slot[p]);
        this.pageQ4Slot[p] = -1;
        this.pageState[p] = VramPageState.EMPTY;
        this.coldQ4Count -= 1;
        this.setQ4Bit(p, false);
      }
    }
    // If validTokens lands mid-page on a Warm Q8 or Cold Q4 page, refault that tail page back to F32
    if (validTokens > 0 && validTokens % VRAM_KV_PAGE_TOKENS !== 0) {
      const tailPage = Math.floor(validTokens / VRAM_KV_PAGE_TOKENS);
      if (
        this.pageState[tailPage] === VramPageState.COLD_COMPRESSED ||
        this.pageState[tailPage] === VramPageState.COLD_Q4
      ) {
        this.ensurePageHot(tailPage);
      }
    }
  }

  /** Stores a layer's K or V vector at `pos` into the page's resident hot F32 frame. */
  writeKv(isValue: boolean, layerIndex: number, pos: number, src: Float32Array): void {
    const pageIdx = pos >>> 4; // pos / 16
    const slotInPage = pos & 15;
    const hotSlot = this.ensurePageHot(pageIdx);
    const dst = isValue
      ? this.hotVHandle.buffer.f32View(this.hotVHandle)
      : this.hotKHandle.buffer.f32View(this.hotKHandle);
    const offset =
      hotSlot * this.hotPageFloats +
      (layerIndex * VRAM_KV_PAGE_TOKENS + slotInPage) * this.kvDim;
    dst.set(src, offset);
  }

  /**
   * 100% In-Verdant KV cache write via WASM bulk `memory.copy`:
   * zero JS Float32Array view allocations!
   */
  writeKvPtr(
    isValue: boolean,
    layerIndex: number,
    pos: number,
    srcWasmPtr: number,
    arena: VerdantWasmArena,
  ): void {
    const pageIdx = pos >>> 4;
    const slotInPage = pos & 15;
    const hotSlot = this.ensurePageHot(pageIdx);
    const handle = isValue ? this.hotVHandle : this.hotKHandle;
    const dstPtr =
      handle.buffer.wasmPtr(handle) +
      (hotSlot * this.hotPageFloats +
        (layerIndex * VRAM_KV_PAGE_TOKENS + slotInPage) * this.kvDim) *
        4;
    arena.kernels.copy_bytes(dstPtr, srcWasmPtr, this.kvDim * 4);
  }

  /**
   * Executes causal grouped-query FLASH_ATTN_EXT directly across all 3 vRAM tiers:
   * Tier-0 Hot F32, Tier-1 Warm Q8, and Tier-2 Cold Q4 using streaming online softmax.
   * When `arena` is provided, executes the ENTIRE layer (all heads × all pages × online softmax)
   * inside ONE SINGLE Verdant WASM SIMD call (`flash_attn_layer`) with zero JS allocations!
   */
  flashAttnExt(
    q: Float32Array,
    qWasmPtr: number,
    out: Float32Array,
    outWasmPtr: number,
    nHeads: number,
    layerIndex: number,
    seqLen: number,
    arena: VerdantWasmArena | null,
  ): void {
    const headDim = this.headDim;
    const nHeadsKv = this.nHeadsKv;
    const numPages = Math.ceil(seqLen / VRAM_KV_PAGE_TOKENS);

    const hasQ4 = this.coldKQ4Handle !== null;
    const useWasm =
      arena !== null &&
      qWasmPtr >= 0 &&
      outWasmPtr >= 0 &&
      this.hotKHandle.buffer.isWasmBacked &&
      this.wasmPageStateHandle.buffer.isWasmBacked;
    const useWasmQ4 = useWasm && hasQ4 && headDim >= 32 && (headDim & 31) === 0;

    // Fast path: 100% In-Verdant Whole-Layer FlashAttention (1 WASM call per layer, 0 JS views, 0 JS Math.exp!)
    if (useWasm && arena && (!hasQ4 || useWasmQ4 || this.coldQ4Count === 0)) {
      const layerHotOffBytes = layerIndex * VRAM_KV_PAGE_TOKENS * this.kvDim * 4;
      const layerQ8OffBytes = layerIndex * VRAM_KV_PAGE_TOKENS * this.kvDim;
      const layerQ4OffBytes = layerIndex * VRAM_KV_PAGE_TOKENS * nHeadsKv * this.headQ4Bytes;
      const layerScOffBytes = layerIndex * VRAM_KV_PAGE_TOKENS * nHeadsKv * 4;

      arena.kernels.flash_attn_layer(
        qWasmPtr,
        outWasmPtr,
        this.wasmPageStateHandle.buffer.wasmPtr(this.wasmPageStateHandle),
        this.wasmPageSlotHandle.buffer.wasmPtr(this.wasmPageSlotHandle),
        this.hotKHandle.buffer.wasmPtr(this.hotKHandle) + layerHotOffBytes,
        this.hotVHandle.buffer.wasmPtr(this.hotVHandle) + layerHotOffBytes,
        this.coldKQ8Handle.buffer.wasmPtr(this.coldKQ8Handle) + layerQ8OffBytes,
        this.coldVQ8Handle.buffer.wasmPtr(this.coldVQ8Handle) + layerQ8OffBytes,
        this.coldKScaleHandle.buffer.wasmPtr(this.coldKScaleHandle) + layerScOffBytes,
        this.coldVScaleHandle.buffer.wasmPtr(this.coldVScaleHandle) + layerScOffBytes,
        useWasmQ4 ? this.coldKQ4Handle!.buffer.wasmPtr(this.coldKQ4Handle!) + layerQ4OffBytes : 0,
        useWasmQ4 ? this.coldVQ4Handle!.buffer.wasmPtr(this.coldVQ4Handle!) + layerQ4OffBytes : 0,
        useWasmQ4 ? this.coldKQ4ScaleHandle!.buffer.wasmPtr(this.coldKQ4ScaleHandle!) + layerScOffBytes : 0,
        useWasmQ4 ? this.coldVQ4ScaleHandle!.buffer.wasmPtr(this.coldVQ4ScaleHandle!) + layerScOffBytes : 0,
        nHeads,
        nHeadsKv,
        headDim,
        seqLen,
        this.hotPageFloats * 4,
        this.coldPageQ8Bytes,
        this.coldPageQ4Bytes,
        this.coldPageScaleFloats * 4,
        this.kvDim * 4,
        this.kvDim,
        nHeadsKv * this.headQ4Bytes,
        nHeadsKv * 4,
      );

      // Update tier telemetry counters in O(numPages) without touching any tensor data
      let hPages = 0;
      let wPages = 0;
      let q4Pages = 0;
      for (let p = 0; p < numPages; p++) {
        const st = this.pageState[p];
        if (st === VramPageState.HOT_RESIDENT) hPages += 1;
        else if (st === VramPageState.COLD_COMPRESSED) wPages += 1;
        else if (st === VramPageState.COLD_Q4) q4Pages += 1;
      }
      this.hotAttentionReads += nHeads * hPages;
      this.coldAttentionReads += nHeads * (wPages + q4Pages);
      this.coldQ4AttentionReads += nHeads * q4Pages;
      return;
    }

    const headsPerKv = Math.floor(nHeads / nHeadsKv);
    const invScale = 1.0 / Math.sqrt(headDim);

    const hotK = this.hotKHandle.buffer.f32View(this.hotKHandle);
    const hotV = this.hotVHandle.buffer.f32View(this.hotVHandle);
    const coldKQ8 = this.coldKQ8Handle.buffer.i8View(this.coldKQ8Handle);
    const coldVQ8 = this.coldVQ8Handle.buffer.i8View(this.coldVQ8Handle);
    const coldKSc = this.coldKScaleHandle.buffer.f32View(this.coldKScaleHandle);
    const coldVSc = this.coldVScaleHandle.buffer.f32View(this.coldVScaleHandle);

    const kQ4Raw = hasQ4 ? this.coldKQ4Handle!.buffer.rawBytes() : null;
    const vQ4Raw = hasQ4 ? this.coldVQ4Handle!.buffer.rawBytes() : null;
    const kQ4Sc = hasQ4 ? this.coldKQ4ScaleHandle!.buffer.f32View(this.coldKQ4ScaleHandle!) : null;
    const vQ4Sc = hasQ4 ? this.coldVQ4ScaleHandle!.buffer.f32View(this.coldVQ4ScaleHandle!) : null;

    const hotKBasePtr = useWasm ? this.hotKHandle.buffer.wasmPtr(this.hotKHandle) : 0;
    const hotVBasePtr = useWasm ? this.hotVHandle.buffer.wasmPtr(this.hotVHandle) : 0;
    const coldKQ8BasePtr = useWasm ? this.coldKQ8Handle.buffer.wasmPtr(this.coldKQ8Handle) : 0;
    const coldVQ8BasePtr = useWasm ? this.coldVQ8Handle.buffer.wasmPtr(this.coldVQ8Handle) : 0;
    const coldKScBasePtr = useWasm ? this.coldKScaleHandle.buffer.wasmPtr(this.coldKScaleHandle) : 0;
    const coldKQ4BasePtr = useWasmQ4 ? this.coldKQ4Handle!.buffer.wasmPtr(this.coldKQ4Handle!) : 0;
    const coldVQ4BasePtr = useWasmQ4 ? this.coldVQ4Handle!.buffer.wasmPtr(this.coldVQ4Handle!) : 0;
    const coldKQ4ScBasePtr = useWasmQ4 ? this.coldKQ4ScaleHandle!.buffer.wasmPtr(this.coldKQ4ScaleHandle!) : 0;
    const wk = arena?.kernels;
    const wasmScores = useWasm && arena
      ? new Float32Array(arena.memory.buffer, WASM_ATTN_SCORES_OFFSET, VRAM_KV_PAGE_TOKENS)
      : null;
    const wasmWeights = useWasm && arena
      ? new Float32Array(arena.memory.buffer, WASM_ATTN_WEIGHTS_OFFSET, VRAM_KV_PAGE_TOKENS)
      : null;

    const layerHotOff = layerIndex * VRAM_KV_PAGE_TOKENS * this.kvDim;
    const layerColdQ8Off = layerIndex * VRAM_KV_PAGE_TOKENS * this.kvDim;
    const layerColdQ4Off = layerIndex * VRAM_KV_PAGE_TOKENS * nHeadsKv * this.headQ4Bytes;
    const layerColdScOff = layerIndex * VRAM_KV_PAGE_TOKENS * nHeadsKv;
    const hotKvStrideBytes = this.kvDim * 4;
    const q8KvStrideBytes = this.kvDim;
    const q4KvStrideBytes = nHeadsKv * this.headQ4Bytes;
    const scStrideBytes = nHeadsKv * 4;
    const chunkSpan = headDim >= 32 && (headDim & 31) === 0 ? 32 : headDim;
    const halfSpan = chunkSpan >>> 1;

    for (let h = 0; h < nHeads; h++) {
      const kvHead = Math.floor(h / headsPerKv);
      const qBase = h * headDim;
      const outBase = h * headDim;
      const qPtr = qWasmPtr + qBase * 4;
      const oPtr = outWasmPtr + outBase * 4;

      let runningMax = -Infinity;
      let runningDenom = 0.0;
      for (let d = 0; d < headDim; d++) {
        out[outBase + d] = 0.0;
      }

      for (let p = 0; p < numPages; p++) {
        const pageEnd = Math.min(VRAM_KV_PAGE_TOKENS, seqLen - p * VRAM_KV_PAGE_TOKENS);
        const state = this.pageState[p];

        if (state === VramPageState.HOT_RESIDENT) {
          this.hotAttentionReads += 1;
          const slotBase = this.pageHotSlot[p] * this.hotPageFloats + layerHotOff;

          if (useWasm && wk && wasmScores && wasmWeights) {
            // Fused 16-token page attention: 2 WASM calls instead of 32 (16x fewer crossings + 16x less out[] traffic!)
            const pageHeadOffBytes = (slotBase + kvHead * headDim) * 4;
            wk.attn_page_scores_f32(
              qPtr,
              hotKBasePtr + pageHeadOffBytes,
              WASM_ATTN_SCORES_OFFSET,
              headDim,
              hotKvStrideBytes,
              pageEnd,
              invScale,
            );
            let pageMax = wasmScores[0];
            for (let t = 1; t < pageEnd; t++) {
              const s = wasmScores[t];
              if (s > pageMax) pageMax = s;
            }
            const newMax = pageMax > runningMax ? pageMax : runningMax;
            const rescale = runningMax === -Infinity ? 0.0 : Math.exp(runningMax - newMax);
            let pageSum = 0.0;
            for (let t = 0; t < pageEnd; t++) {
              const w = Math.exp(wasmScores[t] - newMax);
              wasmWeights[t] = w;
              pageSum += w;
            }
            runningDenom = runningDenom * rescale + pageSum;
            runningMax = newMax;
            wk.attn_page_accum_f32(
              oPtr,
              hotVBasePtr + pageHeadOffBytes,
              WASM_ATTN_WEIGHTS_OFFSET,
              headDim,
              hotKvStrideBytes,
              pageEnd,
              rescale,
            );
          } else {
            for (let t = 0; t < pageEnd; t++) {
              const vecOff = slotBase + t * this.kvDim + kvHead * headDim;
              let dot = 0.0;
              for (let d = 0; d < headDim; d++) dot += q[qBase + d] * hotK[vecOff + d];
              const score = dot * invScale;

              if (score > runningMax) {
                const rescale = runningMax === -Infinity ? 0.0 : Math.exp(runningMax - score);
                runningDenom = runningDenom * rescale + 1.0;
                for (let d = 0; d < headDim; d++) {
                  out[outBase + d] = out[outBase + d] * rescale + hotV[vecOff + d];
                }
                runningMax = score;
              } else {
                const w = Math.exp(score - runningMax);
                runningDenom += w;
                for (let d = 0; d < headDim; d++) {
                  out[outBase + d] += w * hotV[vecOff + d];
                }
              }
            }
          }
        } else if (state === VramPageState.COLD_COMPRESSED) {
          this.coldAttentionReads += 1;
          const coldSlot = this.pageColdSlot[p];
          const q8SlotBase = coldSlot * this.coldPageQ8Bytes + layerColdQ8Off;
          const scSlotBase = coldSlot * this.coldPageScaleFloats + layerColdScOff;

          if (useWasm && wk && wasmScores && wasmWeights) {
            const kQ8HeadPtr = coldKQ8BasePtr + q8SlotBase + kvHead * headDim;
            const vQ8HeadPtr = coldVQ8BasePtr + q8SlotBase + kvHead * headDim;
            const kScHeadPtr = coldKScBasePtr + (scSlotBase + kvHead) * 4;
            wk.attn_page_scores_q8(
              qPtr,
              kQ8HeadPtr,
              kScHeadPtr,
              WASM_ATTN_SCORES_OFFSET,
              headDim,
              q8KvStrideBytes,
              scStrideBytes,
              pageEnd,
              invScale,
            );
            let pageMax = wasmScores[0];
            for (let t = 1; t < pageEnd; t++) {
              const s = wasmScores[t];
              if (s > pageMax) pageMax = s;
            }
            const newMax = pageMax > runningMax ? pageMax : runningMax;
            const rescale = runningMax === -Infinity ? 0.0 : Math.exp(runningMax - newMax);
            let pageSum = 0.0;
            for (let t = 0; t < pageEnd; t++) {
              const w = Math.exp(wasmScores[t] - newMax);
              wasmWeights[t] = w * coldVSc[scSlotBase + t * nHeadsKv + kvHead];
              pageSum += w;
            }
            runningDenom = runningDenom * rescale + pageSum;
            runningMax = newMax;
            wk.attn_page_accum_q8(
              oPtr,
              vQ8HeadPtr,
              WASM_ATTN_WEIGHTS_OFFSET,
              headDim,
              q8KvStrideBytes,
              pageEnd,
              rescale,
            );
          } else {
            for (let t = 0; t < pageEnd; t++) {
              const q8Off = q8SlotBase + t * this.kvDim + kvHead * headDim;
              const scOff = scSlotBase + t * nHeadsKv + kvHead;
              const kScale = coldKSc[scOff];
              const vScale = coldVSc[scOff];

              let dot = 0.0;
              for (let d = 0; d < headDim; d++) dot += q[qBase + d] * coldKQ8[q8Off + d];
              const score = dot * kScale * invScale;

              if (score > runningMax) {
                const rescale = runningMax === -Infinity ? 0.0 : Math.exp(runningMax - score);
                runningDenom = runningDenom * rescale + 1.0;
                for (let d = 0; d < headDim; d++) {
                  out[outBase + d] = out[outBase + d] * rescale + coldVQ8[q8Off + d] * vScale;
                }
                runningMax = score;
              } else {
                const w = Math.exp(score - runningMax);
                runningDenom += w;
                const wv = w * vScale;
                for (let d = 0; d < headDim; d++) {
                  out[outBase + d] += wv * coldVQ8[q8Off + d];
                }
              }
            }
          }
        } else if (state === VramPageState.COLD_Q4 && kQ4Raw && vQ4Raw && kQ4Sc && vQ4Sc) {
          this.coldAttentionReads += 1;
          this.coldQ4AttentionReads += 1;
          const q4Slot = this.pageQ4Slot[p];
          const q4SlotBase = q4Slot * this.coldPageQ4Bytes + layerColdQ4Off;
          const scSlotBase = q4Slot * this.coldPageScaleFloats + layerColdScOff;

          if (useWasmQ4 && wk && wasmScores && wasmWeights) {
            const kQ4HeadPtr = coldKQ4BasePtr + q4SlotBase + kvHead * this.headQ4Bytes;
            const vQ4HeadPtr = coldVQ4BasePtr + q4SlotBase + kvHead * this.headQ4Bytes;
            const kScHeadPtr = coldKQ4ScBasePtr + (scSlotBase + kvHead) * 4;
            wk.attn_page_scores_q4(
              qPtr,
              kQ4HeadPtr,
              kScHeadPtr,
              WASM_ATTN_SCORES_OFFSET,
              headDim,
              q4KvStrideBytes,
              scStrideBytes,
              pageEnd,
              invScale,
            );
            let pageMax = wasmScores[0];
            for (let t = 1; t < pageEnd; t++) {
              const s = wasmScores[t];
              if (s > pageMax) pageMax = s;
            }
            const newMax = pageMax > runningMax ? pageMax : runningMax;
            const rescale = runningMax === -Infinity ? 0.0 : Math.exp(runningMax - newMax);
            let pageSum = 0.0;
            for (let t = 0; t < pageEnd; t++) {
              const w = Math.exp(wasmScores[t] - newMax);
              wasmWeights[t] = w * vQ4Sc[scSlotBase + t * nHeadsKv + kvHead];
              pageSum += w;
            }
            runningDenom = runningDenom * rescale + pageSum;
            runningMax = newMax;
            wk.attn_page_accum_q4(
              oPtr,
              vQ4HeadPtr,
              WASM_ATTN_WEIGHTS_OFFSET,
              headDim,
              q4KvStrideBytes,
              pageEnd,
              rescale,
            );
          } else {
            for (let t = 0; t < pageEnd; t++) {
              const q4RelOff = q4SlotBase + (t * nHeadsKv + kvHead) * this.headQ4Bytes;
              const scOff = scSlotBase + t * nHeadsKv + kvHead;
              const kScale = kQ4Sc[scOff];
              const vScale = vQ4Sc[scOff];

              let dot = 0.0;
              let bOff = this.coldKQ4Handle!.byteOffset + q4RelOff;
              for (let c = 0; c < headDim; c += chunkSpan) {
                for (let j = 0; j < halfSpan; j++) {
                  const byte = kQ4Raw[bOff++];
                  dot += q[qBase + c + j] * ((byte & 0x0f) - 8);
                  dot += q[qBase + c + halfSpan + j] * ((byte >>> 4) - 8);
                }
              }
              const score = dot * kScale * invScale;

              if (score > runningMax) {
                const rescale = runningMax === -Infinity ? 0.0 : Math.exp(runningMax - score);
                runningDenom = runningDenom * rescale + 1.0;
                for (let d = 0; d < headDim; d++) out[outBase + d] *= rescale;
                let bOffV = this.coldVQ4Handle!.byteOffset + q4RelOff;
                for (let c = 0; c < headDim; c += chunkSpan) {
                  for (let j = 0; j < halfSpan; j++) {
                    const byte = vQ4Raw[bOffV++];
                    out[outBase + c + j] += ((byte & 0x0f) - 8) * vScale;
                    out[outBase + c + halfSpan + j] += ((byte >>> 4) - 8) * vScale;
                  }
                }
                runningMax = score;
              } else {
                const w = Math.exp(score - runningMax);
                runningDenom += w;
                const wv = w * vScale;
                let bOffV = this.coldVQ4Handle!.byteOffset + q4RelOff;
                for (let c = 0; c < headDim; c += chunkSpan) {
                  for (let j = 0; j < halfSpan; j++) {
                    const byte = vQ4Raw[bOffV++];
                    out[outBase + c + j] += ((byte & 0x0f) - 8) * wv;
                    out[outBase + c + halfSpan + j] += ((byte >>> 4) - 8) * wv;
                  }
                }
              }
            }
          }
        }
      }

      const invDenom = runningDenom > 0 ? 1.0 / runningDenom : 0.0;
      if (useWasm && wk) {
        wk.scale_f32(oPtr, headDim, invDenom);
      } else {
        for (let d = 0; d < headDim; d++) {
          out[outBase + d] = Math.fround(out[outBase + d] * invDenom);
        }
      }
    }
  }

  /**
   * Captures a full 16-token KV page (from Hot F32 or Warm Q8) as a portable symmetric Q8 block
   * for content-addressed deduplication and OPFS persistence inside `VramUnifiedStore`.
   */
  exportPageAsQ8Block(pageIdx: number): {
    kQ8: Int8Array;
    vQ8: Int8Array;
    kScales: Float32Array;
    vScales: Float32Array;
  } | null {
    if (pageIdx < 0 || pageIdx >= this.maxPages) return null;
    const state = this.pageState[pageIdx] as VramPageState;

    if (state === VramPageState.COLD_COMPRESSED) {
      const coldSlot = this.pageColdSlot[pageIdx];
      if (coldSlot < 0) return null;
      const coldKQ8 = this.coldKQ8Handle.buffer.i8View(this.coldKQ8Handle);
      const coldVQ8 = this.coldVQ8Handle.buffer.i8View(this.coldVQ8Handle);
      const coldKSc = this.coldKScaleHandle.buffer.f32View(this.coldKScaleHandle);
      const coldVSc = this.coldVScaleHandle.buffer.f32View(this.coldVScaleHandle);
      const qBase = coldSlot * this.coldPageQ8Bytes;
      const sBase = coldSlot * this.coldPageScaleFloats;
      return {
        kQ8: coldKQ8.slice(qBase, qBase + this.coldPageQ8Bytes),
        vQ8: coldVQ8.slice(qBase, qBase + this.coldPageQ8Bytes),
        kScales: coldKSc.slice(sBase, sBase + this.coldPageScaleFloats),
        vScales: coldVSc.slice(sBase, sBase + this.coldPageScaleFloats),
      };
    }

    if (state === VramPageState.HOT_RESIDENT) {
      const hotSlot = this.pageHotSlot[pageIdx];
      if (hotSlot < 0) return null;
      const hotK = this.hotKHandle.buffer.f32View(this.hotKHandle);
      const hotV = this.hotVHandle.buffer.f32View(this.hotVHandle);
      const kQ8 = new Int8Array(this.coldPageQ8Bytes);
      const vQ8 = new Int8Array(this.coldPageQ8Bytes);
      const kScales = new Float32Array(this.coldPageScaleFloats);
      const vScales = new Float32Array(this.coldPageScaleFloats);

      const headDim = this.headDim;
      const totalVectors = this.numLayers * VRAM_KV_PAGE_TOKENS * this.nHeadsKv;
      const hotBase = hotSlot * this.hotPageFloats;

      let maxErr = 0;
      let maxSig = 0;
      for (let vec = 0; vec < totalVectors; vec++) {
        const fOff = hotBase + vec * headDim;
        const qOff = vec * headDim;

        let kMax = 0;
        for (let d = 0; d < headDim; d++) {
          const a = Math.abs(hotK[fOff + d]);
          if (a > kMax) kMax = a;
        }
        if (kMax > maxSig) maxSig = kMax;
        const kScale = kMax > 0 ? kMax / 127.0 : 0.0;
        const kInv = kMax > 0 ? 127.0 / kMax : 0.0;
        kScales[vec] = kScale;
        for (let d = 0; d < headDim; d++) {
          const orig = hotK[fOff + d];
          const q = Math.max(-127, Math.min(127, Math.round(orig * kInv)));
          kQ8[qOff + d] = q;
          const err = Math.abs(orig - q * kScale);
          if (err > maxErr) maxErr = err;
        }

        let vMax = 0;
        for (let d = 0; d < headDim; d++) {
          const a = Math.abs(hotV[fOff + d]);
          if (a > vMax) vMax = a;
        }
        if (vMax > maxSig) maxSig = vMax;
        const vScale = vMax > 0 ? vMax / 127.0 : 0.0;
        const vInv = vMax > 0 ? 127.0 / vMax : 0.0;
        vScales[vec] = vScale;
        for (let d = 0; d < headDim; d++) {
          const orig = hotV[fOff + d];
          const q = Math.max(-127, Math.min(127, Math.round(orig * vInv)));
          vQ8[qOff + d] = q;
          const err = Math.abs(orig - q * vScale);
          if (err > maxErr) maxErr = err;
        }
      }

      const rawBytes = this.hotPageFloats * 4 * 2;
      const compressedBytes = (this.coldPageQ8Bytes + this.coldPageScaleFloats * 4) * 2;
      if (!admitColdKvPage(rawBytes, compressedBytes, maxErr, maxSig)) {
        return null;
      }
      return { kQ8, vQ8, kScales, vScales };
    }

    return null;
  }

  /**
   * Restores a content-addressed Q8 KV block directly into Tier-1 Warm Q8 backing for `pageIdx`
   * with zero F32 expansion and zero transformer forward passes (immediately ready for WASM SIMD
   * `dot_q8_f32` / `axpy_q8_f32` attention).
   */
  importQ8BlockIntoPage(
    pageIdx: number,
    block: {
      readonly kQ8: Int8Array;
      readonly vQ8: Int8Array;
      readonly kScales: Float32Array;
      readonly vScales: Float32Array;
    },
  ): boolean {
    if (pageIdx < 0 || pageIdx >= this.maxPages) return false;
    if (
      block.kQ8.length !== this.coldPageQ8Bytes ||
      block.vQ8.length !== this.coldPageQ8Bytes ||
      block.kScales.length !== this.coldPageScaleFloats ||
      block.vScales.length !== this.coldPageScaleFloats
    ) {
      return false;
    }

    // Release any prior slot occupied by pageIdx cleanly
    const prevState = this.pageState[pageIdx] as VramPageState;
    if (prevState === VramPageState.HOT_RESIDENT) {
      this.freeHotSlots.push(this.pageHotSlot[pageIdx]);
      this.pageHotSlot[pageIdx] = -1;
      this.hotResidentCount -= 1;
      this.setHotBit(pageIdx, false);
    } else if (prevState === VramPageState.COLD_COMPRESSED) {
      this.freeColdSlots.push(this.pageColdSlot[pageIdx]);
      this.pageColdSlot[pageIdx] = -1;
      this.warmQ8Count -= 1;
      this.setColdBit(pageIdx, false);
    } else if (prevState === VramPageState.COLD_Q4) {
      this.freeQ4Slots.push(this.pageQ4Slot[pageIdx]);
      this.pageQ4Slot[pageIdx] = -1;
      this.coldQ4Count -= 1;
      this.setQ4Bit(pageIdx, false);
    }
    this.pageState[pageIdx] = VramPageState.EMPTY;
    this.syncWasmPageEntry(pageIdx, VramPageState.EMPTY, -1);

    if (
      this.coldQ4Enabled &&
      (this.freeColdSlots.length === 0 || vramTrim(this.warmQ8Limit, this.warmQ8Count + 1))
    ) {
      this.cascadeWarmToColdQ4(pageIdx);
    }

    const coldSlot = this.freeColdSlots.pop();
    if (coldSlot === undefined) return false;

    const coldKQ8 = this.coldKQ8Handle.buffer.i8View(this.coldKQ8Handle);
    const coldVQ8 = this.coldVQ8Handle.buffer.i8View(this.coldVQ8Handle);
    const coldKSc = this.coldKScaleHandle.buffer.f32View(this.coldKScaleHandle);
    const coldVSc = this.coldVScaleHandle.buffer.f32View(this.coldVScaleHandle);
    const qBase = coldSlot * this.coldPageQ8Bytes;
    const sBase = coldSlot * this.coldPageScaleFloats;

    coldKQ8.set(block.kQ8, qBase);
    coldVQ8.set(block.vQ8, qBase);
    coldKSc.set(block.kScales, sBase);
    coldVSc.set(block.vScales, sBase);

    this.pageState[pageIdx] = VramPageState.COLD_COMPRESSED;
    this.pageColdSlot[pageIdx] = coldSlot;
    this.syncWasmPageEntry(pageIdx, VramPageState.COLD_COMPRESSED, coldSlot);
    this.warmQ8Count += 1;
    this.setColdBit(pageIdx, true);
    return true;
  }

  getPageState(pageIdx: number): VramPageState {
    return this.pageState[pageIdx] as VramPageState;
  }

  getTelemetry(): VramKvTelemetry {
    const coldCompressedPages = this.warmQ8Count + this.coldQ4Count;
    const activePages = this.hotResidentCount + coldCompressedPages;
    const f32PageBytes = this.hotPageFloats * 4 * 2;
    const warmQ8PageBytes = (this.coldPageQ8Bytes + this.coldPageScaleFloats * 4) * 2;
    const coldQ4PageBytes = (this.coldPageQ4Bytes + this.coldPageScaleFloats * 4) * 2;

    const rawEquivalentBytes = Math.max(f32PageBytes * activePages, this.maxPages * f32PageBytes);
    const allocatedHotBytes = this.hotFrameLimit * f32PageBytes;
    const allocatedWarmBytes = this.warmQ8PoolSize * warmQ8PageBytes;
    const allocatedQ4Bytes = this.coldQ4PoolSize * coldQ4PageBytes;
    const allocatedBytes = allocatedHotBytes + allocatedWarmBytes + allocatedQ4Bytes;

    const activeRawBytes = activePages * f32PageBytes;
    const activeActualBytes =
      this.hotResidentCount * f32PageBytes +
      this.warmQ8Count * warmQ8PageBytes +
      this.coldQ4Count * coldQ4PageBytes;
    const savedBytes = Math.max(0, rawEquivalentBytes - allocatedBytes);
    const compressionRatio =
      activeActualBytes > 0
        ? Number((activeRawBytes / activeActualBytes).toFixed(2))
        : Number((rawEquivalentBytes / Math.max(1, allocatedBytes)).toFixed(2));

    const word0 = this.hotBitsets[0] ?? 0n;
    const residencyBitsetHex = `0x${word0.toString(16).padStart(16, '0')}`;

    return {
      pageTokens: VRAM_KV_PAGE_TOKENS,
      maxPages: this.maxPages,
      hotFrameLimit: this.hotFrameLimit,
      warmQ8Limit: this.warmQ8Limit,
      coldQ4Enabled: this.coldQ4Enabled,
      hotResidentPages: this.hotResidentCount,
      coldCompressedPages,
      warmQ8Pages: this.warmQ8Count,
      coldQ4Pages: this.coldQ4Count,
      activePages,
      rawEquivalentBytes,
      allocatedBytes,
      savedBytes,
      compressionRatio,
      reclaimCycles: this.reclaimCycles,
      tierCascadeCycles: this.tierCascadeCycles,
      refaultCount: this.refaultCount,
      hotAttentionReads: this.hotAttentionReads,
      coldAttentionReads: this.coldAttentionReads,
      coldQ4AttentionReads: this.coldQ4AttentionReads,
      residencyBitsetHex,
    };
  }
}
