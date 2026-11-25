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
  type ByteRange,
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
  /** Dense SwiGLU projections (null on Sparse MoE layers). */
  readonly ffnGate: GgufTensorInfo | null;
  readonly ffnUp: GgufTensorInfo | null;
  readonly ffnDown: GgufTensorInfo | null;
  /** Sparse Mixture-of-Experts router + 3D packed expert tensors (`qwen3moe` / `qwen2moe` / `mixtral`). */
  readonly ffnGateInp?: GgufTensorInfo | null;
  readonly ffnGateExps?: GgufTensorInfo | null;
  readonly ffnUpExps?: GgufTensorInfo | null;
  readonly ffnDownExps?: GgufTensorInfo | null;
}

export interface PagedExpertWeights {
  readonly gate: VTensorHandle;
  readonly up: VTensorHandle;
  readonly down: VTensorHandle;
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
  readonly ffnGate: VTensorHandle | null;
  readonly ffnUp: VTensorHandle | null;
  readonly ffnDown: VTensorHandle | null;
  readonly ffnGateInp: VTensorHandle | null;
}

export interface VramWeightPagerConfig {
  readonly residencyMode: WeightResidencyMode;
  readonly layerFrameSlots: number;
  readonly streamEmbeddings: boolean;
  readonly streamLmHead: boolean;
  readonly lmHeadStripeRows?: number;
  /** Number of hot token embedding rows cached inside WASM linear memory when `streamEmbeddings` is active. */
  readonly hotEmbeddingCacheSlots?: number;
  /** Number of Sparse MoE expert triples `(gate, up, down)` cached in WASM LRU memory (OPFS-paged MoE). */
  readonly moeExpertCacheSlots?: number;
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
  readonly isMoe?: boolean;
  readonly moeExpertCacheSlots?: number;
  readonly moeExpertResidentCount?: number;
  readonly moeExpertHits?: number;
  readonly moeExpertMisses?: number;
  readonly moeExpertCacheBytes?: number;
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
  /** Exact byte offset inside a frame slot for each linear projection tensor in this layer. */
  readonly projOffsets: {
    readonly attnQ: number;
    readonly attnK: number;
    readonly attnV: number;
    readonly attnOutput: number;
    readonly ffnGate: number;
    readonly ffnUp: number;
    readonly ffnDown: number;
    readonly ffnGateInp: number;
  };
}

interface MoeLayerExpertGeometry {
  readonly expertCount: number;
  readonly gateInfo: GgufTensorInfo;
  readonly upInfo: GgufTensorInfo;
  readonly downInfo: GgufTensorInfo;
  readonly gateSliceBytes: number;
  readonly upSliceBytes: number;
  readonly downSliceBytes: number;
  readonly gateDims2D: readonly [number, number];
  readonly upDims2D: readonly [number, number];
  readonly downDims2D: readonly [number, number];
}

interface MoeExpertCacheSlot {
  readonly slotIndex: number;
  readonly baseOffset: number;
  key: number;
  lastTouch: number;
  handles: PagedExpertWeights | null;
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

/**
 * Collects all small permanently-resident metadata tensors (norms, biases, QK-norms, and
 * optionally pinned embeddings/LM-head) needed to initialize `VramWeightPager` in `opfs-paged` mode.
 */
export function collectHotMetadataTensorInfos(
  embdInfo: GgufTensorInfo,
  outNormInfo: GgufTensorInfo,
  outWeightInfo: GgufTensorInfo,
  isTiedOutput: boolean,
  layerInfos: readonly LayerTensorInfoGroup[],
  options: { streamEmbeddings: boolean; streamLmHead: boolean },
): GgufTensorInfo[] {
  const list: GgufTensorInfo[] = [outNormInfo];
  if (!options.streamEmbeddings) list.push(embdInfo);
  if (!options.streamLmHead && !(isTiedOutput && !options.streamEmbeddings)) {
    list.push(outWeightInfo);
  }
  for (const l of layerInfos) {
    list.push(l.attnNorm, l.ffnNorm);
    if (l.attnQBias) list.push(l.attnQBias);
    if (l.attnQNorm) list.push(l.attnQNorm);
    if (l.attnKBias) list.push(l.attnKBias);
    if (l.attnKNorm) list.push(l.attnKNorm);
    if (l.attnVBias) list.push(l.attnVBias);
    if (l.attnOutputBias) list.push(l.attnOutputBias);
  }
  return list;
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

  // Unified Store Hot Embedding Row Slab (O(1) intrusive TypedArray LRU in WASM linear memory)
  private readonly embdRowBuffer: VTensorBuffer | null;
  private readonly embdRowHandle: VTensorHandle | null;
  private readonly embdRowBytes: number;
  private readonly embdCacheSlots: number;
  private readonly embdSlotToToken: Int32Array;
  private readonly embdPrev: Int32Array;
  private readonly embdNext: Int32Array;
  private embdHead = -1;
  private embdTail = -1;
  private embdFreeCursor = 0;
  private readonly embdTokenToSlot = new Map<number, number>();
  private embdCacheHits = 0;
  private embdCacheMisses = 0;
  private readonly lmHeadTileBuffer: VTensorBuffer | null;
  private readonly lmHeadStripeRows: number;
  private readonly lmHeadRowBytes: number;
  private readonly lmHeadFullStripeView: VTensorHandle | null;
  private readonly lmHeadTailStripeView: VTensorHandle | null;

  // Sparse Mixture-of-Experts (qwen3moe / qwen2moe / mixtral) O(1) Frequency-Protected WASM Expert Cache
  readonly isMoe: boolean;
  private readonly moeLayerGeo: Array<MoeLayerExpertGeometry | null>;
  private readonly moeMaxExperts: number;
  private readonly residentMoeExperts: Array<Array<PagedExpertWeights> | null>;
  private readonly moeCacheBuffer: VTensorBuffer | null;
  private readonly moeCacheSlots: MoeExpertCacheSlot[];
  private readonly moePrev: Int32Array;
  private readonly moeNext: Int32Array;
  private readonly moeSlotFreq: Uint8Array;
  private moeHead = -1;
  private moeTail = -1;
  private moeFreeCursor = 0;
  private readonly moeKeyToSlot = new Map<number, number>();
  private readonly moeGateOffsetInSlot: number;
  private readonly moeUpOffsetInSlot: number;
  private readonly moeDownOffsetInSlot: number;
  private moeClock = 0;
  private moeExpertHits = 0;
  private moeExpertMisses = 0;

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
    preloadedHotTensors?: ReadonlyMap<string, Uint8Array>,
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

    // Exact total logical weight footprint across all model tensors + Sparse MoE expert geometry
    let logicalBytes = embdInfo.byteSize + outNormInfo.byteSize + (isTiedOutput ? 0 : outWeightInfo.byteSize);
    const moeLayerGeo: Array<MoeLayerExpertGeometry | null> = [];
    let anyMoe = false;
    let maxExperts = 0;
    let maxGateSlice = 0;
    let maxUpSlice = 0;
    let maxDownSlice = 0;

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
        (l.ffnGate?.byteSize ?? 0) +
        (l.ffnUp?.byteSize ?? 0) +
        (l.ffnDown?.byteSize ?? 0) +
        (l.ffnGateInp?.byteSize ?? 0) +
        (l.ffnGateExps?.byteSize ?? 0) +
        (l.ffnUpExps?.byteSize ?? 0) +
        (l.ffnDownExps?.byteSize ?? 0);

      if (l.ffnGateInp && l.ffnGateExps && l.ffnUpExps && l.ffnDownExps) {
        anyMoe = true;
        const nExps = Math.max(1, l.ffnGateExps.dims[2] ?? parsed.hparams.expertCount ?? 1);
        maxExperts = Math.max(maxExperts, nExps);
        const gateSliceBytes = Math.floor(l.ffnGateExps.byteSize / nExps);
        const upSliceBytes = Math.floor(l.ffnUpExps.byteSize / nExps);
        const downSliceBytes = Math.floor(l.ffnDownExps.byteSize / nExps);
        maxGateSlice = Math.max(maxGateSlice, gateSliceBytes);
        maxUpSlice = Math.max(maxUpSlice, upSliceBytes);
        maxDownSlice = Math.max(maxDownSlice, downSliceBytes);
        moeLayerGeo.push({
          expertCount: nExps,
          gateInfo: l.ffnGateExps,
          upInfo: l.ffnUpExps,
          downInfo: l.ffnDownExps,
          gateSliceBytes,
          upSliceBytes,
          downSliceBytes,
          gateDims2D: [l.ffnGateExps.dims[0] ?? 1, l.ffnGateExps.dims[1] ?? 1],
          upDims2D: [l.ffnUpExps.dims[0] ?? 1, l.ffnUpExps.dims[1] ?? 1],
          downDims2D: [l.ffnDownExps.dims[0] ?? 1, l.ffnDownExps.dims[1] ?? 1],
        });
      } else {
        moeLayerGeo.push(null);
      }
    }
    this.logicalWeightBytes = logicalBytes;
    this.isMoe = anyMoe;
    this.moeLayerGeo = moeLayerGeo;
    this.moeMaxExperts = Math.max(1, maxExperts);

    // Precompute coalesced extent plans for every layer (b0d40626)
    // For MoE layers, frame slots hold attention projections + tiny router gate (`ffnGateInp`),
    // while 3D packed experts demand-page through the bounded WASM Sparse Expert Cache!
    const plans: LayerExtentPlan[] = [];
    let totalExtents = 0;
    let maxLayerFrameBytes = 0;
    for (const l of layerInfos) {
      const linearTensors: GgufTensorInfo[] = [
        l.attnQ,
        l.attnK,
        l.attnV,
        l.attnOutput,
      ];
      if (l.ffnGate) linearTensors.push(l.ffnGate);
      if (l.ffnUp) linearTensors.push(l.ffnUp);
      if (l.ffnDown) linearTensors.push(l.ffnDown);
      if (l.ffnGateInp) linearTensors.push(l.ffnGateInp);

      // Coalesce across <= 32 KiB gaps (tiny interleaved norm/bias vectors) -> 1.0 ext/L on dense!
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
          ffnGate: l.ffnGate ? (offsetByName.get(l.ffnGate.name) ?? -1) : -1,
          ffnUp: l.ffnUp ? (offsetByName.get(l.ffnUp.name) ?? -1) : -1,
          ffnDown: l.ffnDown ? (offsetByName.get(l.ffnDown.name) ?? -1) : -1,
          ffnGateInp: l.ffnGateInp ? (offsetByName.get(l.ffnGateInp.name) ?? -1) : -1,
        },
      });
    }
    this.layerPlans = plans;
    this.avgCoalescedExtentsPerLayer =
      layerInfos.length > 0 ? Number((totalExtents / layerInfos.length).toFixed(2)) : 0;

    // Helper: read a single GGUF tensor directly from preloaded buffer or backing straight into WASM linear memory
    const uploadFromBacking = (buf: VTensorBuffer, info: GgufTensorInfo): VTensorHandle => {
      const handle = buf.allocTensor(info.name, info.type, info.dims);
      const raw = buf.rawBytes();
      const preloaded = preloadedHotTensors?.get(info.name);
      if (preloaded && preloaded.byteLength === info.byteSize) {
        raw.set(preloaded, handle.byteOffset);
        this.coalescedReads += 1;
        this.bytesRestoredFromBacking += info.byteSize;
        return handle;
      }
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
        allInfos.push(l.attnNorm, l.attnQ, l.attnK, l.attnV, l.attnOutput, l.ffnNorm);
        if (l.ffnGate) allInfos.push(l.ffnGate);
        if (l.ffnUp) allInfos.push(l.ffnUp);
        if (l.ffnDown) allInfos.push(l.ffnDown);
        if (l.ffnGateInp) allInfos.push(l.ffnGateInp);
        if (l.ffnGateExps) allInfos.push(l.ffnGateExps);
        if (l.ffnUpExps) allInfos.push(l.ffnUpExps);
        if (l.ffnDownExps) allInfos.push(l.ffnDownExps);
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
        ffnGate: l.ffnGate ? handleByName.get(l.ffnGate.name)! : null,
        ffnUp: l.ffnUp ? handleByName.get(l.ffnUp.name)! : null,
        ffnDown: l.ffnDown ? handleByName.get(l.ffnDown.name)! : null,
        ffnGateInp: l.ffnGateInp ? handleByName.get(l.ffnGateInp.name)! : null,
      }));

      // Build zero-copy 2D expert slice views inside `vtensor.weights` for resident-all MoE models
      this.residentMoeExperts = moeLayerGeo.map((geo, lIdx) => {
        if (!geo) return null;
        const gBase = handleByName.get(geo.gateInfo.name)!.byteOffset;
        const uBase = handleByName.get(geo.upInfo.name)!.byteOffset;
        const dBase = handleByName.get(geo.downInfo.name)!.byteOffset;
        const slices: PagedExpertWeights[] = [];
        for (let e = 0; e < geo.expertCount; e++) {
          slices.push({
            gate: weightsBuf.createViewTensor(
              `blk.${lIdx}.ffn_gate_exp.${e}`,
              geo.gateInfo.type,
              geo.gateDims2D,
              gBase + e * geo.gateSliceBytes,
            ),
            up: weightsBuf.createViewTensor(
              `blk.${lIdx}.ffn_up_exp.${e}`,
              geo.upInfo.type,
              geo.upDims2D,
              uBase + e * geo.upSliceBytes,
            ),
            down: weightsBuf.createViewTensor(
              `blk.${lIdx}.ffn_down_exp.${e}`,
              geo.downInfo.type,
              geo.downDims2D,
              dBase + e * geo.downSliceBytes,
            ),
          });
        }
        return slices;
      });
      this.moeCacheBuffer = null;
      this.moeCacheSlots = [];
      this.moePrev = new Int32Array(0);
      this.moeNext = new Int32Array(0);
      this.moeSlotFreq = new Uint8Array(0);
      this.moeGateOffsetInSlot = 0;
      this.moeUpOffsetInSlot = 0;
      this.moeDownOffsetInSlot = 0;

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
      this.embdPrev = new Int32Array(0);
      this.embdNext = new Int32Array(0);
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

    // 3. Unified Store Hot Embedding Row Slab (O(1) intrusive TypedArray LRU in WASM linear memory)
    if (this.streamEmbeddings) {
      const vocabRows = Math.max(1, embdInfo.dims[1] ?? 1);
      const requestedSlots = config.hotEmbeddingCacheSlots ?? 256;
      this.embdCacheSlots = Math.max(1, Math.min(vocabRows, requestedSlots));
      this.embdSlotToToken = new Int32Array(this.embdCacheSlots);
      this.embdSlotToToken.fill(-1);
      this.embdPrev = new Int32Array(this.embdCacheSlots);
      this.embdPrev.fill(-1);
      this.embdNext = new Int32Array(this.embdCacheSlots);
      this.embdNext.fill(-1);

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
      this.embdPrev = new Int32Array(0);
      this.embdNext = new Int32Array(0);
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

    // 5. Bounded WASM LRU Sparse Expert Cache (`vtensor.vram_moe_experts`) for 30B-A3B MoE!
    this.residentMoeExperts = new Array<Array<PagedExpertWeights> | null>(layerInfos.length).fill(null);
    if (this.isMoe) {
      const gateOff = 0;
      const upOff = alignUp32(maxGateSlice);
      const downOff = alignUp32(upOff + maxUpSlice);
      const slotStride = alignUp32(Math.max(64, downOff + maxDownSlice));
      const requestedSlots = Math.max(
        2,
        config.moeExpertCacheSlots ?? Math.max(16, (parsed.hparams.expertUsedCount ?? 8) * 4),
      );
      const totalPossibleExperts = moeLayerGeo.reduce((acc, g) => acc + (g?.expertCount ?? 0), 0);
      const slotCount = Math.max(1, Math.min(totalPossibleExperts, requestedSlots));
      const moeBuf = allocBuffer('vtensor.vram_moe_experts', 'weights', slotStride * slotCount);
      this.moeCacheBuffer = moeBuf;
      this.moeGateOffsetInSlot = gateOff;
      this.moeUpOffsetInSlot = upOff;
      this.moeDownOffsetInSlot = downOff;
      this.moeCacheSlots = [];
      for (let s = 0; s < slotCount; s++) {
        this.moeCacheSlots.push({
          slotIndex: s,
          baseOffset: s * slotStride,
          key: -1,
          lastTouch: 0,
          handles: null,
        });
      }
      this.moePrev = new Int32Array(slotCount);
      this.moePrev.fill(-1);
      this.moeNext = new Int32Array(slotCount);
      this.moeNext.fill(-1);
      this.moeSlotFreq = new Uint8Array(slotCount);
      totalResident += moeBuf.capacityBytes;
    } else {
      this.moeCacheBuffer = null;
      this.moeCacheSlots = [];
      this.moePrev = new Int32Array(0);
      this.moeNext = new Int32Array(0);
      this.moeSlotFreq = new Uint8Array(0);
      this.moeGateOffsetInSlot = 0;
      this.moeUpOffsetInSlot = 0;
      this.moeDownOffsetInSlot = 0;
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
        ffnGate:
          info.ffnGate && p.ffnGate >= 0
            ? frameBuf.createViewTensor(
                info.ffnGate.name,
                info.ffnGate.type,
                info.ffnGate.dims,
                base + p.ffnGate,
              )
            : null,
        ffnUp:
          info.ffnUp && p.ffnUp >= 0
            ? frameBuf.createViewTensor(
                info.ffnUp.name,
                info.ffnUp.type,
                info.ffnUp.dims,
                base + p.ffnUp,
              )
            : null,
        ffnDown:
          info.ffnDown && p.ffnDown >= 0
            ? frameBuf.createViewTensor(
                info.ffnDown.name,
                info.ffnDown.type,
                info.ffnDown.dims,
                base + p.ffnDown,
              )
            : null,
        ffnGateInp:
          info.ffnGateInp && p.ffnGateInp >= 0
            ? frameBuf.createViewTensor(
                info.ffnGateInp.name,
                info.ffnGateInp.type,
                info.ffnGateInp.dims,
                base + p.ffnGateInp,
              )
            : null,
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

  /** O(1) intrusive doubly-linked list promotion to MRU head for MoE expert cache slots. */
  private promoteMoeSlot(idx: number): void {
    if (this.moeHead === idx) return;
    const prev = this.moePrev[idx];
    const next = this.moeNext[idx];
    if (prev >= 0) this.moeNext[prev] = next;
    if (next >= 0) this.moePrev[next] = prev;
    if (this.moeTail === idx) this.moeTail = prev;
    this.moePrev[idx] = -1;
    this.moeNext[idx] = this.moeHead;
    if (this.moeHead >= 0) this.moePrev[this.moeHead] = idx;
    this.moeHead = idx;
    if (this.moeTail < 0) this.moeTail = idx;
  }

  /** O(1) intrusive doubly-linked list promotion to MRU head for embedding row cache slots. */
  private promoteEmbdSlot(idx: number): void {
    if (this.embdHead === idx) return;
    const prev = this.embdPrev[idx];
    const next = this.embdNext[idx];
    if (prev >= 0) this.embdNext[prev] = next;
    if (next >= 0) this.embdPrev[next] = prev;
    if (this.embdTail === idx) this.embdTail = prev;
    this.embdPrev[idx] = -1;
    this.embdNext[idx] = this.embdHead;
    if (this.embdHead >= 0) this.embdPrev[this.embdHead] = idx;
    this.embdHead = idx;
    if (this.embdTail < 0) this.embdTail = idx;
  }

  /**
   * Acquires the 2D `(gate, up, down)` weight slices for routed expert `(layerIdx, expertIdx)`.
   * - In `resident-all` mode: returns the prebuilt zero-copy slice views inside `vtensor.weights`.
   * - In `opfs-paged` mode: checks the O(1) frequency-protected WASM Sparse Expert Cache
   *   (`vtensor.vram_moe_experts`) first; on miss, demand-pages ONLY this expert's 3 slices from
   *   OPFS directly into WASM memory with zero JS heap allocations on slot handle reuse!
   */
  acquireExpertSlice(layerIdx: number, expertIdx: number): PagedExpertWeights {
    const residentLayer = this.residentMoeExperts[layerIdx];
    if (residentLayer) {
      this.moeExpertHits += 1;
      return residentLayer[expertIdx];
    }

    const geo = this.moeLayerGeo[layerIdx];
    if (!geo) {
      throw new Error(`Layer ${layerIdx} is not configured as a Sparse MoE layer.`);
    }

    const key = layerIdx * this.moeMaxExperts + expertIdx;
    const cachedSlotIdx = this.moeKeyToSlot.get(key);
    if (cachedSlotIdx !== undefined) {
      const slot = this.moeCacheSlots[cachedSlotIdx];
      slot.lastTouch = ++this.moeClock;
      if (this.moeSlotFreq[cachedSlotIdx] < 15) {
        this.moeSlotFreq[cachedSlotIdx] += 1;
      }
      this.promoteMoeSlot(cachedSlotIdx);
      this.moeExpertHits += 1;
      return slot.handles!;
    }

    // O(1) free slot allocation or frequency-protected second-chance LRU eviction from tail
    let targetIdx: number;
    if (this.moeFreeCursor < this.moeCacheSlots.length) {
      targetIdx = this.moeFreeCursor++;
    } else {
      const maxSecondChances = Math.min(8, this.moeCacheSlots.length - 1);
      let passes = 0;
      while (passes < maxSecondChances && this.moeTail >= 0 && this.moeSlotFreq[this.moeTail] > 0) {
        const saved = this.moeTail;
        this.moeSlotFreq[saved] -= 1;
        this.promoteMoeSlot(saved);
        passes++;
      }
      targetIdx = this.moeTail >= 0 ? this.moeTail : 0;
    }

    const targetSlot = this.moeCacheSlots[targetIdx];
    if (targetSlot.key >= 0) {
      this.moeKeyToSlot.delete(targetSlot.key);
      this.slotReclaims += 1;
    }

    const buf = this.moeCacheBuffer!;
    const raw = buf.rawBytes();
    const base = targetSlot.baseOffset;
    const gateDst = base + this.moeGateOffsetInSlot;
    const upDst = base + this.moeUpOffsetInSlot;
    const downDst = base + this.moeDownOffsetInSlot;

    const gateFileOff = this.parsed.dataOffset + geo.gateInfo.offset + expertIdx * geo.gateSliceBytes;
    const upFileOff = this.parsed.dataOffset + geo.upInfo.offset + expertIdx * geo.upSliceBytes;
    const downFileOff = this.parsed.dataOffset + geo.downInfo.offset + expertIdx * geo.downSliceBytes;

    const t0 = performance.now();
    const rGate = this.backing.readRange(gateFileOff, raw, gateDst, geo.gateSliceBytes);
    const rUp = this.backing.readRange(upFileOff, raw, upDst, geo.upSliceBytes);
    const rDown = this.backing.readRange(downFileOff, raw, downDst, geo.downSliceBytes);
    this.backingReadMs += performance.now() - t0;

    if (rGate !== geo.gateSliceBytes || rUp !== geo.upSliceBytes || rDown !== geo.downSliceBytes) {
      throw new Error(
        `vRAM MoE expert (L${layerIdx}, E${expertIdx}) short OPFS restore.`,
      );
    }

    this.coalescedReads += 3;
    this.bytesRestoredFromBacking += geo.gateSliceBytes + geo.upSliceBytes + geo.downSliceBytes;
    this.moeExpertMisses += 1;

    // Zero-allocation reuse of slot VTensorHandles whenever expert shapes & types match (100% on uniform MoE)
    const existing = targetSlot.handles;
    let handles: PagedExpertWeights;
    if (
      existing &&
      existing.gate.type === geo.gateInfo.type &&
      existing.up.type === geo.upInfo.type &&
      existing.down.type === geo.downInfo.type &&
      existing.gate.dims[0] === geo.gateDims2D[0] &&
      existing.gate.dims[1] === geo.gateDims2D[1] &&
      existing.up.dims[0] === geo.upDims2D[0] &&
      existing.up.dims[1] === geo.upDims2D[1] &&
      existing.down.dims[0] === geo.downDims2D[0] &&
      existing.down.dims[1] === geo.downDims2D[1]
    ) {
      handles = existing;
    } else {
      handles = {
        gate: buf.createViewTensor(
          `vram.moe_g_s${targetSlot.slotIndex}`,
          geo.gateInfo.type,
          geo.gateDims2D,
          gateDst,
        ),
        up: buf.createViewTensor(
          `vram.moe_u_s${targetSlot.slotIndex}`,
          geo.upInfo.type,
          geo.upDims2D,
          upDst,
        ),
        down: buf.createViewTensor(
          `vram.moe_d_s${targetSlot.slotIndex}`,
          geo.downInfo.type,
          geo.downDims2D,
          downDst,
        ),
      };
      targetSlot.handles = handles;
    }

    targetSlot.key = key;
    targetSlot.lastTouch = ++this.moeClock;
    this.moeSlotFreq[targetIdx] = 0;
    this.promoteMoeSlot(targetIdx);
    this.moeKeyToSlot.set(key, targetIdx);
    return handles;
  }

  /**
   * Returns a `VTensorHandle` + local row index for embedding token `tokenId`.
   * When `streamEmbeddings` is active, checks the Unified Store Hot Embedding Row Slab
   * (O(1) intrusive TypedArray LRU cache in WASM linear memory) first so repeated tokens
   * resolve in 0 OPFS reads!
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
      this.promoteEmbdSlot(cachedSlot);
      this.embdCacheHits += 1;
      return { handle, rowIndex: cachedSlot };
    }

    // O(1) free slot selection or LRU tail eviction
    const targetSlot =
      this.embdFreeCursor < this.embdCacheSlots
        ? this.embdFreeCursor++
        : this.embdTail >= 0
          ? this.embdTail
          : 0;

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
    this.promoteEmbdSlot(targetSlot);
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

  /**
   * Loads a single LM-head row stripe `[rowStart, rowStart + rowCount)` from backing into the
   * bounded LM-head tile buffer (used by async demand-paged LM-head streaming).
   */
  acquireLmHeadStripe(rowStart: number, rowCount: number): VTensorHandle {
    if (this.pinnedOutputWeight) {
      return this.pinnedOutputWeight;
    }
    const tileBuf = this.lmHeadTileBuffer!;
    const raw = tileBuf.rawBytes();
    const cols = this.outWeightInfo.dims[0] ?? 1;
    const baseFileOffset = this.parsed.dataOffset + this.outWeightInfo.offset;
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

    return rowCount === this.lmHeadStripeRows && this.lmHeadFullStripeView
      ? this.lmHeadFullStripeView
      : this.lmHeadTailStripeView && this.lmHeadTailStripeView.dims[1] === rowCount
        ? this.lmHeadTailStripeView
        : tileBuf.createViewTensor(
            `vram.lmhead_${rowStart}`,
            this.outWeightInfo.type,
            [cols, rowCount],
            0,
          );
  }

  /** Returns backing byte ranges required to fault `layerIdx`, or `[]` if already resident in WASM. */
  getLayerMissingRanges(layerIdx: number): ByteRange[] {
    if (this.residentLayers || this.layerToSlot[layerIdx] >= 0) {
      return [];
    }
    const plan = this.layerPlans[layerIdx];
    const ranges: ByteRange[] = [];
    for (let e = 0; e < plan.extents.length; e++) {
      const ext = plan.extents[e];
      ranges.push({ fileOffset: ext.fileOffset, byteLength: ext.byteLength });
    }
    return ranges;
  }

  /**
   * Returns backing byte ranges for all routed `(layerIdx, expertIdx)` triples NOT currently
   * resident in the WASM Sparse Expert Cache.
   */
  getExpertMissingRanges(layerIdx: number, expertIds: readonly number[]): ByteRange[] {
    if (this.residentMoeExperts[layerIdx]) {
      return [];
    }
    const geo = this.moeLayerGeo[layerIdx];
    if (!geo) return [];
    const ranges: ByteRange[] = [];
    for (let i = 0; i < expertIds.length; i++) {
      const expId = expertIds[i];
      if (expId < 0 || expId >= geo.expertCount) continue;
      const key = layerIdx * this.moeMaxExperts + expId;
      if (this.moeKeyToSlot.has(key)) continue;
      ranges.push(
        {
          fileOffset: this.parsed.dataOffset + geo.gateInfo.offset + expId * geo.gateSliceBytes,
          byteLength: geo.gateSliceBytes,
        },
        {
          fileOffset: this.parsed.dataOffset + geo.upInfo.offset + expId * geo.upSliceBytes,
          byteLength: geo.upSliceBytes,
        },
        {
          fileOffset: this.parsed.dataOffset + geo.downInfo.offset + expId * geo.downSliceBytes,
          byteLength: geo.downSliceBytes,
        },
      );
    }
    return ranges;
  }

  /** Returns backing byte ranges for embedding token IDs not yet resident in the WASM hot row cache. */
  getEmbeddingMissingRanges(tokenIds: readonly number[]): ByteRange[] {
    if (this.pinnedTokenEmbd) return [];
    const ranges: ByteRange[] = [];
    const seen = new Set<number>();
    const base = this.parsed.dataOffset + this.embdInfo.offset;
    for (let i = 0; i < tokenIds.length; i++) {
      const tid = tokenIds[i];
      if (tid < 0 || seen.has(tid) || this.embdTokenToSlot.has(tid)) continue;
      seen.add(tid);
      ranges.push({
        fileOffset: base + tid * this.embdRowBytes,
        byteLength: this.embdRowBytes,
      });
    }
    return ranges;
  }

  /** Returns the row-stripe byte descriptors for streaming LM-head evaluation (`[]` if pinned). */
  getLmHeadStripeSpecs(): Array<{
    rowStart: number;
    rowCount: number;
    fileOffset: number;
    byteLength: number;
  }> {
    if (this.pinnedOutputWeight) return [];
    const vocabSize = this.outWeightInfo.dims[1] ?? 1;
    const baseFileOffset = this.parsed.dataOffset + this.outWeightInfo.offset;
    const specs: Array<{
      rowStart: number;
      rowCount: number;
      fileOffset: number;
      byteLength: number;
    }> = [];
    for (let rowStart = 0; rowStart < vocabSize; rowStart += this.lmHeadStripeRows) {
      const rowCount = Math.min(this.lmHeadStripeRows, vocabSize - rowStart);
      specs.push({
        rowStart,
        rowCount,
        fileOffset: baseFileOffset + rowStart * this.lmHeadRowBytes,
        byteLength: rowCount * this.lmHeadRowBytes,
      });
    }
    return specs;
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
      isMoe: this.isMoe,
      moeExpertCacheSlots: this.moeCacheSlots.length,
      moeExpertResidentCount: this.moeKeyToSlot.size,
      moeExpertHits: this.moeExpertHits,
      moeExpertMisses: this.moeExpertMisses,
      moeExpertCacheBytes: this.moeCacheBuffer?.capacityBytes ?? 0,
    };
  }
}
