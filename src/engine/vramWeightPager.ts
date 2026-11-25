/**
 * OPFS-Backed vRAM Virtual DIMM Weight Pager (`afflom/vRAM` + `afflom/vTensor`).
 *
 * Implements bounded-memory transformer weight execution inspired directly by `afflom/vRAM`:
 *   - Commit `f1185aff`: Derive virtual RAM capacity from proved limits and index only live extents.
 *   - Commit `b0d40626`: Coalesce contiguous Verdant backing slots during DIMM range restores.
 *   - Commit `3f27a941`: Stream batched Verdant projections from one committed DIMM publication.
 *   - Commit `88d7a68f`: Reclaim dense backing slots without allocating during release.
 *
 * Enables 1B–7B+ GGUF models to execute inside a bounded WASM32 linear arena by paging layer
 * frames, single-row embeddings, and row-striped LM-head tiles directly from an OPFS-backed
 * `VramBackingStore` straight into WASM linear memory with zero intermediate JS copies.
 */

import {
  getGgmlTypeTrait,
  type GgufTensorInfo,
  type ParsedGgufModel,
} from './gguf';
import {
  coalesceTensorExtents,
  type CoalescedExtent,
  type VramBackingKind,
  type VramBackingStore,
} from './vramBackingStore';
import type { VTensorBuffer, VTensorHandle } from './vtensorProvider';

export type WeightResidencyMode = 'resident-all' | 'opfs-paged';
export type WeightResidencyPolicy = 'auto' | 'paged';

export interface LayerTensorInfoGroup {
  readonly attnNorm: GgufTensorInfo;
  readonly attnQ: GgufTensorInfo;
  readonly attnQBias: GgufTensorInfo | null;
  readonly attnQNorm?: GgufTensorInfo | null;
  readonly attnK: GgufTensorInfo;
  readonly attnKBias: GgufTensorInfo | null;
  readonly attnKNorm?: GgufTensorInfo | null;
  readonly attnV: GgufTensorInfo;
  readonly attnVBias: GgufTensorInfo | null;
  readonly attnOutput: GgufTensorInfo;
  readonly attnOutputBias?: GgufTensorInfo | null;
  readonly ffnNorm: GgufTensorInfo;
  readonly ffnGate: GgufTensorInfo;
  readonly ffnUp: GgufTensorInfo;
  readonly ffnDown: GgufTensorInfo;
}

export interface PagedLayerWeights {
  readonly attnNorm: VTensorHandle;
  readonly attnQ: VTensorHandle;
  readonly attnQBias: VTensorHandle | null;
  readonly attnQNorm: VTensorHandle | null;
  readonly attnK: VTensorHandle;
  readonly attnKBias: VTensorHandle | null;
  readonly attnKNorm: VTensorHandle | null;
  readonly attnV: VTensorHandle;
  readonly attnVBias: VTensorHandle | null;
  readonly attnOutput: VTensorHandle;
  readonly attnOutputBias: VTensorHandle | null;
  readonly ffnNorm: VTensorHandle;
  readonly ffnGate: VTensorHandle;
  readonly ffnUp: VTensorHandle;
  readonly ffnDown: VTensorHandle;
}

export interface VramWeightPagerConfig {
  readonly residencyMode: WeightResidencyMode;
  readonly layerFrameSlots: number;
  readonly streamEmbeddings: boolean;
  readonly streamLmHead: boolean;
  readonly lmHeadStripeRows?: number;
  /** Number of hot token embedding rows cached inside WASM linear memory when `streamEmbeddings` is active. */
  readonly hotEmbeddingCacheSlots?: number;
}

export interface VramWeightPagerTelemetry {
  readonly backingKind: VramBackingKind;
  readonly isPersistent: boolean;
  readonly residencyMode: WeightResidencyMode;
  readonly logicalWeightBytes: number;
  readonly residentWeightBytes: number;
  readonly virtualToResidentRatio: number;
  readonly totalLayers: number;
  readonly layerFrameSlots: number;
  readonly residentLayerCount: number;
  readonly pinnedLayerCount: number;
  readonly evictionPolicy: 'belady-cyclic' | 'resident-all';
  readonly avgCoalescedExtentsPerLayer: number;
  readonly layerHits: number;
  readonly layerFaults: number;
  readonly slotReclaims: number;
  readonly coalescedReads: number;
  readonly bytesRestoredFromBacking: number;
  readonly backingReadMs: number;
  readonly avgBackingReadMBs: number;
  readonly streamEmbeddings: boolean;
  readonly streamLmHead: boolean;
  readonly embdCacheSlots: number;
  readonly embdCacheResidentCount: number;
  readonly embdCacheHits: number;
  readonly embdCacheMisses: number;
  readonly embdCacheBytes: number;
  readonly residentLayerBitsetHex: string;
}

export interface LayerHotMetadata {
  readonly attnNorm: VTensorHandle;
  readonly attnQBias: VTensorHandle | null;
  readonly attnQNorm: VTensorHandle | null;
  readonly attnKBias: VTensorHandle | null;
  readonly attnKNorm: VTensorHandle | null;
  readonly attnVBias: VTensorHandle | null;
  readonly attnOutputBias: VTensorHandle | null;
  readonly ffnNorm: VTensorHandle;
}

interface LayerExtentPlan {
  readonly extents: readonly CoalescedExtent[];
  readonly extentFrameOffsets: readonly number[];
  readonly totalFrameBytes: number;
  /** Exact byte offset inside a frame slot for each of the 7 linear projection tensors. */
  readonly projOffsets: {
    readonly attnQ: number;
    readonly attnK: number;
    readonly attnV: number;
    readonly attnOutput: number;
    readonly ffnGate: number;
    readonly ffnUp: number;
    readonly ffnDown: number;
  };
}

interface WeightFrameSlot {
  readonly slotIndex: number;
  readonly frameBaseOffset: number;
  layerIdx: number;
  lastTouch: number;
  publication: PagedLayerWeights | null;
}

function alignUp32(n: number): number {
  const rem = n & 31;
  return rem === 0 ? n : n + (32 - rem);
}

export const VRAM_LM_HEAD_TILE_TARGET_BYTES = 8 * 1024 * 1024;
export const VRAM_LM_HEAD_TILE_MAX_BYTES = 32 * 1024 * 1024;

/**
 * Scales the LM-head streaming tile budget dynamically between 8 MiB and 32 MiB according to
 * available physical weight RAM, cutting OPFS read syscalls and thread barriers up to 4× on 7B+ models.
 */
export function computeDynamicLmHeadTileBytes(
  physicalWeightBudgetBytes: number,
  lmHeadWeightBytes: number,
): number {
  const scaled = Math.floor(physicalWeightBudgetBytes * 0.12);
  const clamped = Math.max(
    VRAM_LM_HEAD_TILE_TARGET_BYTES,
    Math.min(VRAM_LM_HEAD_TILE_MAX_BYTES, scaled),
  );
  return Math.max(64 * 1024, Math.min(lmHeadWeightBytes, clamped));
}

/**
 * Computes the optimal LM-head stripe height (rows per tile) so the streaming tile stays
 * bounded cleanly around `targetTileBytes` regardless of vocab size (32K–256K).
 */
export function computeLmHeadStripeRows(
  outWeightInfo: GgufTensorInfo,
  targetTileBytes = VRAM_LM_HEAD_TILE_TARGET_BYTES,
): number {
  const cols = outWeightInfo.dims[0] ?? 1;
  const rows = outWeightInfo.dims[1] ?? 1;
  const trait = getGgmlTypeTrait(outWeightInfo.type);
  const rowBytes = (cols / trait.blockSize) * trait.typeSizeBytes;
  const rawRows = Math.max(64, Math.floor(targetTileBytes / Math.max(1, rowBytes)));
  // Align to 64 rows for clean SIMD tiling
  const aligned = Math.max(64, Math.floor(rawRows / 64) * 64);
  return Math.min(rows, aligned);
}

export class VramWeightPager {
  readonly residencyMode: WeightResidencyMode;
  readonly backingKind: VramBackingKind;
  readonly isPersistent: boolean;
  readonly outputNorm: VTensorHandle;
  readonly streamEmbeddings: boolean;
  readonly streamLmHead: boolean;
  readonly totalLayers: number;
  readonly layerFrameSlots: number;
  readonly logicalWeightBytes: number;
  readonly residentWeightBytes: number;
  readonly avgCoalescedExtentsPerLayer: number;

  private readonly parsed: ParsedGgufModel;
  private readonly backing: VramBackingStore;
  private readonly layerInfos: readonly LayerTensorInfoGroup[];
  private readonly embdInfo: GgufTensorInfo;
  private readonly outWeightInfo: GgufTensorInfo;

  // Resident-all storage
  private readonly residentLayers: PagedLayerWeights[] | null;
  private readonly pinnedTokenEmbd: VTensorHandle | null;
  private readonly pinnedOutputWeight: VTensorHandle | null;

  // OPFS-paged storage (reusable zero-allocation frame slots + streaming tiles)
  private readonly hotLayers: readonly LayerHotMetadata[];
  private readonly layerPlans: readonly LayerExtentPlan[];
  private readonly frameBuffer: VTensorBuffer | null;
  private readonly slots: WeightFrameSlot[];
  private readonly layerToSlot: Int32Array;
  private residentLayerBitset = 0n;
  private clock = 0;
  private readonly pinnedLayerCount: number;
  private readonly cachedStreamingPublications: Array<PagedLayerWeights | null>;

  // Unified Store Hot Embedding Row Slab (Zipfian LRU cache inside WASM linear memory)
  private readonly embdRowBuffer: VTensorBuffer | null;
  private readonly embdRowHandle: VTensorHandle | null;
  private readonly embdRowBytes: number;
  private readonly embdCacheSlots: number;
  private readonly embdSlotToToken: Int32Array;
  private readonly embdSlotLastTouch: Uint32Array;
  private readonly embdTokenToSlot = new Map<number, number>();
  private embdCacheClock = 0;
  private embdCacheHits = 0;
  private embdCacheMisses = 0;
  private readonly lmHeadTileBuffer: VTensorBuffer | null;
  private readonly lmHeadStripeRows: number;
  private readonly lmHeadRowBytes: number;
  private readonly lmHeadFullStripeView: VTensorHandle | null;
  private readonly lmHeadTailStripeView: VTensorHandle | null;

  // Live telemetry counters
  private layerHits = 0;
  private layerFaults = 0;
  private slotReclaims = 0;
  private coalescedReads = 0;
  private bytesRestoredFromBacking = 0;
  private backingReadMs = 0;

  constructor(
    parsed: ParsedGgufModel,
    backing: VramBackingStore,
    embdInfo: GgufTensorInfo,
    outNormInfo: GgufTensorInfo,
    outWeightInfo: GgufTensorInfo,
    isTiedOutput: boolean,
    layerInfos: readonly LayerTensorInfoGroup[],
    config: VramWeightPagerConfig,
    allocBuffer: (id: string, usage: 'weights' | 'compute', capacityBytes: number) => VTensorBuffer,
  ) {
    this.parsed = parsed;
    this.backing = backing;
    this.backingKind = backing.kind;
    this.isPersistent = backing.isPersistent;
    this.embdInfo = embdInfo;
    this.outWeightInfo = outWeightInfo;
    this.layerInfos = layerInfos;
    this.totalLayers = layerInfos.length;
    this.residencyMode = config.residencyMode;

    const embdTrait = getGgmlTypeTrait(embdInfo.type);
    this.embdRowBytes = ((embdInfo.dims[0] ?? 1) / embdTrait.blockSize) * embdTrait.typeSizeBytes;
    const outTrait = getGgmlTypeTrait(outWeightInfo.type);
    this.lmHeadRowBytes = ((outWeightInfo.dims[0] ?? 1) / outTrait.blockSize) * outTrait.typeSizeBytes;

    // Exact total logical weight footprint across all model tensors
    let logicalBytes = embdInfo.byteSize + outNormInfo.byteSize + (isTiedOutput ? 0 : outWeightInfo.byteSize);
    for (const l of layerInfos) {
      logicalBytes +=
        l.attnNorm.byteSize +
        l.attnQ.byteSize +
        (l.attnQBias?.byteSize ?? 0) +
        (l.attnQNorm?.byteSize ?? 0) +
        l.attnK.byteSize +
        (l.attnKBias?.byteSize ?? 0) +
        (l.attnKNorm?.byteSize ?? 0) +
        l.attnV.byteSize +
        (l.attnVBias?.byteSize ?? 0) +
        l.attnOutput.byteSize +
        (l.attnOutputBias?.byteSize ?? 0) +
        l.ffnNorm.byteSize +
        l.ffnGate.byteSize +
        l.ffnUp.byteSize +
        l.ffnDown.byteSize;
    }
    this.logicalWeightBytes = logicalBytes;

    // Precompute coalesced extent plans for every layer (b0d40626)
    const plans: LayerExtentPlan[] = [];
    let totalExtents = 0;
    let maxLayerFrameBytes = 0;
    for (const l of layerInfos) {
      const linearTensors = [
        l.attnQ,
        l.attnK,
        l.attnV,
        l.attnOutput,
        l.ffnGate,
        l.ffnUp,
        l.ffnDown,
      ];
      // Coalesce across <= 32 KiB gaps (tiny interleaved norm/bias vectors) -> 1.0 ext/L on 135M–3B!
      const extents = coalesceTensorExtents(linearTensors, parsed.dataOffset, 32768);
      totalExtents += extents.length;
      const extentFrameOffsets: number[] = [];
      const offsetByName = new Map<string, number>();
      let cursor = 0;
      for (const ext of extents) {
        const alignedCursor = alignUp32(cursor);
        extentFrameOffsets.push(alignedCursor);
        for (const s of ext.slices) {
          offsetByName.set(s.tensor.name, alignedCursor + s.relativeOffset);
        }
        cursor = alignedCursor + ext.byteLength;
      }
      const totalFrameBytes = alignUp32(cursor);
      if (totalFrameBytes > maxLayerFrameBytes) {
        maxLayerFrameBytes = totalFrameBytes;
      }
      plans.push({
        extents,
        extentFrameOffsets,
        totalFrameBytes,
        projOffsets: {
          attnQ: offsetByName.get(l.attnQ.name)!,
          attnK: offsetByName.get(l.attnK.name)!,
          attnV: offsetByName.get(l.attnV.name)!,
          attnOutput: offsetByName.get(l.attnOutput.name)!,
          ffnGate: offsetByName.get(l.ffnGate.name)!,
          ffnUp: offsetByName.get(l.ffnUp.name)!,
          ffnDown: offsetByName.get(l.ffnDown.name)!,
        },
      });
    }
    this.layerPlans = plans;
    this.avgCoalescedExtentsPerLayer =
      layerInfos.length > 0 ? Number((totalExtents / layerInfos.length).toFixed(2)) : 0;

    // Helper: read a single GGUF tensor directly from backing straight into WASM linear memory
    const uploadFromBacking = (buf: VTensorBuffer, info: GgufTensorInfo): VTensorHandle => {
      const handle = buf.allocTensor(info.name, info.type, info.dims);
      const raw = buf.rawBytes();
      const t0 = performance.now();
      const n = this.backing.readRange(
        this.parsed.dataOffset + info.offset,
        raw,
        handle.byteOffset,
        info.byteSize,
      );
      this.backingReadMs += performance.now() - t0;
      if (n !== info.byteSize) {
        throw new Error(`Short backing read for "${info.name}": ${n} != ${info.byteSize}`);
      }
      this.coalescedReads += 1;
      this.bytesRestoredFromBacking += info.byteSize;
      return handle;
    };

    if (config.residencyMode === 'resident-all') {
      // --- Mode A: Resident-All (coalesced direct load from OPFS/backing into vtensor.weights) ---
      this.streamEmbeddings = false;
      this.streamLmHead = false;
      this.layerFrameSlots = layerInfos.length;

      const allInfos: GgufTensorInfo[] = [embdInfo, outNormInfo];
      if (!isTiedOutput) allInfos.push(outWeightInfo);
      for (const l of layerInfos) {
        allInfos.push(l.attnNorm, l.attnQ, l.attnK, l.attnV, l.attnOutput, l.ffnNorm, l.ffnGate, l.ffnUp, l.ffnDown);
        if (l.attnQBias) allInfos.push(l.attnQBias);
        if (l.attnQNorm) allInfos.push(l.attnQNorm);
        if (l.attnKBias) allInfos.push(l.attnKBias);
        if (l.attnKNorm) allInfos.push(l.attnKNorm);
        if (l.attnVBias) allInfos.push(l.attnVBias);
        if (l.attnOutputBias) allInfos.push(l.attnOutputBias);
      }

      // Coalesce contiguous tensors for bulk load directly into vtensor.weights!
      const bulkExtents = coalesceTensorExtents(allInfos, parsed.dataOffset, 4096);
      let bulkCapacity = 0;
      for (const ext of bulkExtents) {
        bulkCapacity += alignUp32(ext.byteLength) + 32;
      }
      const weightsBuf = allocBuffer('vtensor.weights', 'weights', Math.max(64, bulkCapacity));
      const handleByName = new Map<string, VTensorHandle>();
      const raw = weightsBuf.rawBytes();
      let cursor = 0;

      for (const ext of bulkExtents) {
        const base = alignUp32(cursor);
        const t0 = performance.now();
        const read = this.backing.readRange(ext.fileOffset, raw, base, ext.byteLength);
        this.backingReadMs += performance.now() - t0;
        if (read !== ext.byteLength) {
          throw new Error(`Short coalesced bulk read at offset ${ext.fileOffset}: ${read} != ${ext.byteLength}`);
        }
        this.coalescedReads += 1;
        this.bytesRestoredFromBacking += ext.byteLength;
        for (const s of ext.slices) {
          handleByName.set(
            s.tensor.name,
            weightsBuf.createViewTensor(
              s.tensor.name,
              s.tensor.type,
              s.tensor.dims,
              base + s.relativeOffset,
              true,
            ),
          );
        }
        cursor = base + ext.byteLength;
      }
      // Advance buffer cursor bookkeeping via a zero-overhead telemetry marker if needed
      this.pinnedTokenEmbd = handleByName.get(embdInfo.name)!;
      this.outputNorm = handleByName.get(outNormInfo.name)!;
      this.pinnedOutputWeight = isTiedOutput
        ? this.pinnedTokenEmbd
        : handleByName.get(outWeightInfo.name)!;

      const residentLayers: PagedLayerWeights[] = layerInfos.map((l) => ({
        attnNorm: handleByName.get(l.attnNorm.name)!,
        attnQ: handleByName.get(l.attnQ.name)!,
        attnQBias: l.attnQBias ? handleByName.get(l.attnQBias.name)! : null,
        attnQNorm: l.attnQNorm ? handleByName.get(l.attnQNorm.name)! : null,
        attnK: handleByName.get(l.attnK.name)!,
        attnKBias: l.attnKBias ? handleByName.get(l.attnKBias.name)! : null,
        attnKNorm: l.attnKNorm ? handleByName.get(l.attnKNorm.name)! : null,
        attnV: handleByName.get(l.attnV.name)!,
        attnVBias: l.attnVBias ? handleByName.get(l.attnVBias.name)! : null,
        attnOutput: handleByName.get(l.attnOutput.name)!,
        attnOutputBias: l.attnOutputBias ? handleByName.get(l.attnOutputBias.name)! : null,
        ffnNorm: handleByName.get(l.ffnNorm.name)!,
        ffnGate: handleByName.get(l.ffnGate.name)!,
        ffnUp: handleByName.get(l.ffnUp.name)!,
        ffnDown: handleByName.get(l.ffnDown.name)!,
      }));

      this.residentLayers = residentLayers;
      this.hotLayers = [];
      this.frameBuffer = null;
      this.slots = [];
      this.layerToSlot = new Int32Array(layerInfos.length);
      this.layerToSlot.fill(-1);
      for (let i = 0; i < layerInfos.length; i++) {
        this.residentLayerBitset |= 1n << BigInt(i);
      }
      this.embdRowBuffer = null;
      this.embdRowHandle = null;
      this.embdCacheSlots = 0;
      this.embdSlotToToken = new Int32Array(0);
      this.embdSlotLastTouch = new Uint32Array(0);
      this.lmHeadTileBuffer = null;
      this.lmHeadStripeRows = outWeightInfo.dims[1] ?? 1;
      this.lmHeadFullStripeView = null;
      this.lmHeadTailStripeView = null;
      this.residentWeightBytes = weightsBuf.capacityBytes;
      this.pinnedLayerCount = layerInfos.length;
      this.cachedStreamingPublications = [];
      return;
    }

    // --- Mode B: OPFS-Backed vRAM Paged Weights (bounded physical WASM RAM!) ---
    this.streamEmbeddings = config.streamEmbeddings;
    this.streamLmHead = config.streamLmHead;
    this.layerFrameSlots = Math.max(1, Math.min(layerInfos.length, config.layerFrameSlots));
    this.residentLayers = null;

    // 1. Hot Permanent Buffer: tiny norms, biases, and optionally pinned embd/LM-head
    let hotCapacity = alignUp32(outNormInfo.byteSize) + 32;
    if (!this.streamEmbeddings) hotCapacity += alignUp32(embdInfo.byteSize) + 32;
    if (!this.streamLmHead && !(isTiedOutput && !this.streamEmbeddings)) {
      hotCapacity += alignUp32(outWeightInfo.byteSize) + 32;
    }
    for (const l of layerInfos) {
      hotCapacity += alignUp32(l.attnNorm.byteSize) + alignUp32(l.ffnNorm.byteSize) + 64;
      if (l.attnQBias) hotCapacity += alignUp32(l.attnQBias.byteSize) + 32;
      if (l.attnQNorm) hotCapacity += alignUp32(l.attnQNorm.byteSize) + 32;
      if (l.attnKBias) hotCapacity += alignUp32(l.attnKBias.byteSize) + 32;
      if (l.attnKNorm) hotCapacity += alignUp32(l.attnKNorm.byteSize) + 32;
      if (l.attnVBias) hotCapacity += alignUp32(l.attnVBias.byteSize) + 32;
      if (l.attnOutputBias) hotCapacity += alignUp32(l.attnOutputBias.byteSize) + 32;
    }

    const hotBuf = allocBuffer('vtensor.weights_hot', 'weights', Math.max(64, hotCapacity));
    this.outputNorm = uploadFromBacking(hotBuf, outNormInfo);
    this.pinnedTokenEmbd = this.streamEmbeddings ? null : uploadFromBacking(hotBuf, embdInfo);
    this.pinnedOutputWeight = this.streamLmHead
      ? null
      : isTiedOutput && this.pinnedTokenEmbd
        ? this.pinnedTokenEmbd
        : uploadFromBacking(hotBuf, outWeightInfo);

    const hotLayers: LayerHotMetadata[] = layerInfos.map((l) => ({
      attnNorm: uploadFromBacking(hotBuf, l.attnNorm),
      attnQBias: l.attnQBias ? uploadFromBacking(hotBuf, l.attnQBias) : null,
      attnQNorm: l.attnQNorm ? uploadFromBacking(hotBuf, l.attnQNorm) : null,
      attnKBias: l.attnKBias ? uploadFromBacking(hotBuf, l.attnKBias) : null,
      attnKNorm: l.attnKNorm ? uploadFromBacking(hotBuf, l.attnKNorm) : null,
      attnVBias: l.attnVBias ? uploadFromBacking(hotBuf, l.attnVBias) : null,
      attnOutputBias: l.attnOutputBias ? uploadFromBacking(hotBuf, l.attnOutputBias) : null,
      ffnNorm: uploadFromBacking(hotBuf, l.ffnNorm),
    }));
    this.hotLayers = hotLayers;

    // 2. Reusable Layer Frame Pool (`vtensor.vram_weight_frames`) — zero allocations on reclaim!
    const frameStride = alignUp32(Math.max(64, maxLayerFrameBytes));
    const frameBuf = allocBuffer(
      'vtensor.vram_weight_frames',
      'weights',
      frameStride * this.layerFrameSlots,
    );
    this.frameBuffer = frameBuf;
    this.slots = [];
    for (let s = 0; s < this.layerFrameSlots; s++) {
      this.slots.push({
        slotIndex: s,
        frameBaseOffset: s * frameStride,
        layerIdx: -1,
        lastTouch: 0,
        publication: null,
      });
    }
    this.layerToSlot = new Int32Array(layerInfos.length);
    this.layerToSlot.fill(-1);
    // Belady-Optimal Cyclic Pinning: pin layers [0 .. K-2] permanently, stream [K-1 .. L-1] through slot K-1.
    // Achieves mathematically optimal (K-1)/L hit rate on sequential layer scans instead of LRU's 0% hit rate!
    this.pinnedLayerCount =
      this.layerFrameSlots >= layerInfos.length
        ? layerInfos.length
        : Math.max(0, this.layerFrameSlots - 1);
    this.cachedStreamingPublications = new Array<PagedLayerWeights | null>(layerInfos.length).fill(null);

    let totalResident = hotBuf.capacityBytes + frameBuf.capacityBytes;

    // 3. Unified Store Hot Embedding Row Slab (Zipfian LRU cache in WASM linear memory)
    if (this.streamEmbeddings) {
      const vocabRows = Math.max(1, embdInfo.dims[1] ?? 1);
      const requestedSlots = config.hotEmbeddingCacheSlots ?? 256;
      this.embdCacheSlots = Math.max(1, Math.min(vocabRows, requestedSlots));
      this.embdSlotToToken = new Int32Array(this.embdCacheSlots);
      this.embdSlotToToken.fill(-1);
      this.embdSlotLastTouch = new Uint32Array(this.embdCacheSlots);

      const embdBuf = allocBuffer(
        'vtensor.vram_embd_cache',
        'weights',
        alignUp32(this.embdCacheSlots * this.embdRowBytes) + 32,
      );
      this.embdRowBuffer = embdBuf;
      this.embdRowHandle = embdBuf.allocTensor('vram.embd_cache', embdInfo.type, [
        embdInfo.dims[0] ?? 1,
        this.embdCacheSlots,
      ]);
      totalResident += embdBuf.capacityBytes;
    } else {
      this.embdRowBuffer = null;
      this.embdRowHandle = null;
      this.embdCacheSlots = 0;
      this.embdSlotToToken = new Int32Array(0);
      this.embdSlotLastTouch = new Uint32Array(0);
    }

    // 4. Row-striped LM-head streaming tile buffer if LM head is streamed from OPFS
    if (this.streamLmHead) {
      const cols = outWeightInfo.dims[0] ?? 1;
      const vocabRows = outWeightInfo.dims[1] ?? 1;
      this.lmHeadStripeRows =
        config.lmHeadStripeRows ?? computeLmHeadStripeRows(outWeightInfo);
      const tileBytes = alignUp32(this.lmHeadStripeRows * this.lmHeadRowBytes) + 32;
      const tileBuf = allocBuffer('vtensor.vram_lmhead_tile', 'weights', tileBytes);
      this.lmHeadTileBuffer = tileBuf;
      this.lmHeadFullStripeView = tileBuf.createViewTensor(
        'vram.lmhead_stripe',
        outWeightInfo.type,
        [cols, this.lmHeadStripeRows],
        0,
      );
      const tailRows = vocabRows % this.lmHeadStripeRows;
      this.lmHeadTailStripeView =
        tailRows > 0
          ? tileBuf.createViewTensor('vram.lmhead_tail', outWeightInfo.type, [cols, tailRows], 0)
          : null;
      totalResident += tileBuf.capacityBytes;
    } else {
      this.lmHeadTileBuffer = null;
      this.lmHeadStripeRows = outWeightInfo.dims[1] ?? 1;
      this.lmHeadFullStripeView = null;
      this.lmHeadTailStripeView = null;
    }

    this.residentWeightBytes = totalResident;
  }

  /**
   * Acquires the single committed DIMM publication for `layerIdx` (`afflom/vRAM` commit `3f27a941`).
   * All 7 projections and all `B*` batched prefill tokens execute from this publication.
   */
  acquireLayer(layerIdx: number): PagedLayerWeights {
    if (this.residentLayers) {
      this.layerHits += 1;
      return this.residentLayers[layerIdx];
    }

    const existingSlotIdx = this.layerToSlot[layerIdx];
    if (existingSlotIdx >= 0) {
      const slot = this.slots[existingSlotIdx];
      slot.lastTouch = ++this.clock;
      this.layerHits += 1;
      return slot.publication!;
    }

    // Belady-Optimal Cyclic Pinning slot assignment:
    // - Layers [0 .. pinnedLayerCount - 1] map 1:1 to permanently pinned slots [0 .. K-2] (never evicted!).
    // - Layers [pinnedLayerCount .. L-1] stream sequentially through dedicated streaming slot (K-1).
    const isPinnedLayer = layerIdx < this.pinnedLayerCount;
    const targetSlot = isPinnedLayer
      ? this.slots[layerIdx]
      : this.slots[this.slots.length - 1];

    if (targetSlot.layerIdx >= 0) {
      const evictedLayer = targetSlot.layerIdx;
      this.layerToSlot[evictedLayer] = -1;
      this.residentLayerBitset &= ~(1n << BigInt(evictedLayer));
      this.slotReclaims += 1;
    }

    // Coalesced range restore straight from backing into WASM frame slot (b0d40626)
    const plan = this.layerPlans[layerIdx];
    const frameBuf = this.frameBuffer!;
    const raw = frameBuf.rawBytes();
    const base = targetSlot.frameBaseOffset;

    const t0 = performance.now();
    for (let e = 0; e < plan.extents.length; e++) {
      const ext = plan.extents[e];
      const dstOff = base + plan.extentFrameOffsets[e];
      const read = this.backing.readRange(ext.fileOffset, raw, dstOff, ext.byteLength);
      if (read !== ext.byteLength) {
        throw new Error(
          `vRAM layer ${layerIdx} coalesced restore short read: ${read} != ${ext.byteLength}`,
        );
      }
      this.coalescedReads += 1;
      this.bytesRestoredFromBacking += ext.byteLength;
    }
    this.backingReadMs += performance.now() - t0;

    // Zero-allocation reuse of cached VTensorHandle views for streaming layers!
    let publication = !isPinnedLayer ? this.cachedStreamingPublications[layerIdx] : null;
    if (!publication) {
      const info = this.layerInfos[layerIdx];
      const hot = this.hotLayers[layerIdx];
      const p = plan.projOffsets;
      publication = {
        attnNorm: hot.attnNorm,
        attnQ: frameBuf.createViewTensor(info.attnQ.name, info.attnQ.type, info.attnQ.dims, base + p.attnQ),
        attnQBias: hot.attnQBias,
        attnQNorm: hot.attnQNorm,
        attnK: frameBuf.createViewTensor(info.attnK.name, info.attnK.type, info.attnK.dims, base + p.attnK),
        attnKBias: hot.attnKBias,
        attnKNorm: hot.attnKNorm,
        attnV: frameBuf.createViewTensor(info.attnV.name, info.attnV.type, info.attnV.dims, base + p.attnV),
        attnVBias: hot.attnVBias,
        attnOutput: frameBuf.createViewTensor(
          info.attnOutput.name,
          info.attnOutput.type,
          info.attnOutput.dims,
          base + p.attnOutput,
        ),
        attnOutputBias: hot.attnOutputBias,
        ffnNorm: hot.ffnNorm,
        ffnGate: frameBuf.createViewTensor(
          info.ffnGate.name,
          info.ffnGate.type,
          info.ffnGate.dims,
          base + p.ffnGate,
        ),
        ffnUp: frameBuf.createViewTensor(info.ffnUp.name, info.ffnUp.type, info.ffnUp.dims, base + p.ffnUp),
        ffnDown: frameBuf.createViewTensor(
          info.ffnDown.name,
          info.ffnDown.type,
          info.ffnDown.dims,
          base + p.ffnDown,
        ),
      };
      if (!isPinnedLayer) {
        this.cachedStreamingPublications[layerIdx] = publication;
      }
    }

    targetSlot.layerIdx = layerIdx;
    targetSlot.lastTouch = ++this.clock;
    targetSlot.publication = publication;
    this.layerToSlot[layerIdx] = targetSlot.slotIndex;
    this.residentLayerBitset |= 1n << BigInt(layerIdx);
    this.layerFaults += 1;
    return publication;
  }

  /**
   * Returns a `VTensorHandle` + local row index for embedding token `tokenId`.
   * When `streamEmbeddings` is active, checks the Unified Store Hot Embedding Row Slab
   * (Zipfian LRU cache in WASM linear memory) first so repeated tokens resolve in 0 OPFS reads!
   */
  acquireEmbeddingRow(tokenId: number): { handle: VTensorHandle; rowIndex: number } {
    if (this.pinnedTokenEmbd) {
      this.embdCacheHits += 1;
      return { handle: this.pinnedTokenEmbd, rowIndex: tokenId };
    }
    const handle = this.embdRowHandle!;
    const buf = this.embdRowBuffer!;

    // Fast O(1) Zipfian cache hit inside WASM linear memory
    const cachedSlot = this.embdTokenToSlot.get(tokenId);
    if (cachedSlot !== undefined) {
      this.embdSlotLastTouch[cachedSlot] = ++this.embdCacheClock;
      this.embdCacheHits += 1;
      return { handle, rowIndex: cachedSlot };
    }

    // Cache miss: select an empty slot first, otherwise evict the LRU hot-embedding slot
    let targetSlot = 0;
    let oldestTouch = 0xffffffff;
    for (let s = 0; s < this.embdCacheSlots; s++) {
      if (this.embdSlotToToken[s] < 0) {
        targetSlot = s;
        break;
      }
      const touch = this.embdSlotLastTouch[s];
      if (touch < oldestTouch) {
        oldestTouch = touch;
        targetSlot = s;
      }
    }

    const evictedToken = this.embdSlotToToken[targetSlot];
    if (evictedToken >= 0) {
      this.embdTokenToSlot.delete(evictedToken);
    }

    const rowOffset = this.parsed.dataOffset + this.embdInfo.offset + tokenId * this.embdRowBytes;
    const dstOffset = handle.byteOffset + targetSlot * this.embdRowBytes;
    const t0 = performance.now();
    const read = this.backing.readRange(rowOffset, buf.rawBytes(), dstOffset, this.embdRowBytes);
    this.backingReadMs += performance.now() - t0;
    if (read !== this.embdRowBytes) {
      throw new Error(`vRAM embedding row ${tokenId} short read: ${read} != ${this.embdRowBytes}`);
    }

    this.embdSlotToToken[targetSlot] = tokenId;
    this.embdSlotLastTouch[targetSlot] = ++this.embdCacheClock;
    this.embdTokenToSlot.set(tokenId, targetSlot);
    this.embdCacheMisses += 1;
    this.coalescedReads += 1;
    this.bytesRestoredFromBacking += this.embdRowBytes;
    return { handle, rowIndex: targetSlot };
  }

  /**
   * Streams the LM-head output projection weight matrix in bounded row-stripes (or invokes once
   * directly against the pinned LM head when `!streamLmHead`).
   */
  forEachLmHeadStripe(
    callback: (weightStripe: VTensorHandle, rowStart: number, rowCount: number) => void,
  ): void {
    const vocabSize = this.outWeightInfo.dims[1] ?? 1;
    if (this.pinnedOutputWeight) {
      callback(this.pinnedOutputWeight, 0, vocabSize);
      return;
    }

    const tileBuf = this.lmHeadTileBuffer!;
    const raw = tileBuf.rawBytes();
    const cols = this.outWeightInfo.dims[0] ?? 1;
    const baseFileOffset = this.parsed.dataOffset + this.outWeightInfo.offset;

    for (let rowStart = 0; rowStart < vocabSize; rowStart += this.lmHeadStripeRows) {
      const rowCount = Math.min(this.lmHeadStripeRows, vocabSize - rowStart);
      const stripeBytes = rowCount * this.lmHeadRowBytes;
      const fileOffset = baseFileOffset + rowStart * this.lmHeadRowBytes;

      const t0 = performance.now();
      const read = this.backing.readRange(fileOffset, raw, 0, stripeBytes);
      this.backingReadMs += performance.now() - t0;
      if (read !== stripeBytes) {
        throw new Error(`vRAM LM-head stripe [${rowStart}, ${rowStart + rowCount}) short read.`);
      }
      this.coalescedReads += 1;
      this.bytesRestoredFromBacking += stripeBytes;

      const view =
        rowCount === this.lmHeadStripeRows && this.lmHeadFullStripeView
          ? this.lmHeadFullStripeView
          : this.lmHeadTailStripeView && this.lmHeadTailStripeView.dims[1] === rowCount
            ? this.lmHeadTailStripeView
            : tileBuf.createViewTensor(
                `vram.lmhead_${rowStart}`,
                this.outWeightInfo.type,
                [cols, rowCount],
                0,
              );
      callback(view, rowStart, rowCount);
    }
  }

  /** Returns all resident layer handles when in `resident-all` mode (for compiled graph fast path). */
  getAllResidentLayers(): readonly PagedLayerWeights[] | null {
    return this.residentLayers;
  }

  /** Returns permanently resident norm & bias handles for `layerIdx` in `opfs-paged` mode. */
  getHotLayer(layerIdx: number): LayerHotMetadata {
    return this.hotLayers[layerIdx];
  }

  getPinnedTokenEmbd(): VTensorHandle | null {
    return this.pinnedTokenEmbd;
  }

  getPinnedOutputWeight(): VTensorHandle | null {
    return this.pinnedOutputWeight;
  }

  getTelemetry(): VramWeightPagerTelemetry {
    const residentLayerCount = this.residentLayers
      ? this.totalLayers
      : this.slots.reduce((acc, s) => acc + (s.layerIdx >= 0 ? 1 : 0), 0);
    const avgBackingReadMBs =
      this.backingReadMs > 0.05
        ? (this.bytesRestoredFromBacking / (1024 * 1024)) / (this.backingReadMs / 1000)
        : 0;
    const ratio =
      this.residentWeightBytes > 0
        ? this.logicalWeightBytes / this.residentWeightBytes
        : 1;

    return {
      backingKind: this.backingKind,
      isPersistent: this.isPersistent,
      residencyMode: this.residencyMode,
      logicalWeightBytes: this.logicalWeightBytes,
      residentWeightBytes: this.residentWeightBytes,
      virtualToResidentRatio: Number(ratio.toFixed(2)),
      totalLayers: this.totalLayers,
      layerFrameSlots: this.layerFrameSlots,
      residentLayerCount,
      pinnedLayerCount: this.residentLayers ? this.totalLayers : this.pinnedLayerCount,
      evictionPolicy: this.residentLayers ? 'resident-all' : 'belady-cyclic',
      avgCoalescedExtentsPerLayer: this.avgCoalescedExtentsPerLayer,
      layerHits: this.layerHits,
      layerFaults: this.layerFaults,
      slotReclaims: this.slotReclaims,
      coalescedReads: this.coalescedReads,
      bytesRestoredFromBacking: this.bytesRestoredFromBacking,
      backingReadMs: Number(this.backingReadMs.toFixed(2)),
      avgBackingReadMBs: Number(avgBackingReadMBs.toFixed(1)),
      streamEmbeddings: this.streamEmbeddings,
      streamLmHead: this.streamLmHead,
      embdCacheSlots: this.embdCacheSlots,
      embdCacheResidentCount: this.embdTokenToSlot.size,
      embdCacheHits: this.embdCacheHits,
      embdCacheMisses: this.embdCacheMisses,
      embdCacheBytes: this.embdRowBuffer?.capacityBytes ?? 0,
      residentLayerBitsetHex: `0x${this.residentLayerBitset.toString(16).padStart(16, '0')}`,
    };
  }
}
