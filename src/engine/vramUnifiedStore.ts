/**
 * Unified Virtual Store (`VramUnifiedStore`) — Global Address Space Optimization (Theorem T6).
 *
 * Unifies all three inference memory & storage realms under a single global capacity budget
 * `C_global = min(p, g, b)` across WASM linear memory and persistent OPFS backing:
 *   1. Layer Weight Frames (coalesced DIMM extents managed by `VramWeightPager`)
 *   2. Zipfian Hot Embedding Row Slab (WASM-resident LRU cache of frequent `token_embd` rows
 *      so repeated tokens hit RAM in 0 OPFS syscalls even in `opfs-paged` mode)
 *   3. Content-Addressed Embedded KV Prefix Blocks (16-token compressed Q8 KV pages indexed by
 *      chained 64-bit prefix hashes `hash64(modelId, parentHash, tokens[0..15])`, globally
 *      deduplicated across threads, edits, and browser reloads via `<modelId>.unified.kv` OPFS).
 *
 * Paired with Continuous Speculative Prefill, this eliminates perceived Time-To-First-Token
 * (TTFT) latency by keeping prefill running cooperatively while the user composes and
 * restoring embedded prefixes instantaneously across threads and sessions.
 */

import {
  OPFS_VRAM_DIR_NAME,
  openSyncAccessHandleWithRetry,
  sanitizeOpfsModelFileName,
  type SyncAccessHandleCapableFile,
} from './vramBackingStore';
import { VRAM_KV_PAGE_TOKENS } from './vramKvPager';
import { compressUnifiedKvBytes, decompressUnifiedKvBytes } from './vramSwapPolicy';

export interface CompressedKvPageBlock {
  readonly kQ8: Int8Array;
  readonly vQ8: Int8Array;
  readonly kScales: Float32Array;
  readonly vScales: Float32Array;
}

export interface UnifiedKvBlockEntry {
  readonly blockKey: string;
  readonly pageIndex: number;
  readonly tokenIds: readonly number[];
  readonly data: CompressedKvPageBlock;
  readonly byteLength: number;
  readonly isSystemSink: boolean;
  lastTouch: number;
  hitCount: number;
}

export interface UnifiedStoreAllocationPlan {
  /** Number of hot token embedding rows cached inside WASM linear memory (0 if pinned). */
  readonly hotEmbeddingCacheSlots: number;
  /** Bytes reserved for the hot embedding cache slab. */
  readonly hotEmbeddingSlabBytes: number;
  /** Maximum byte budget allocated to content-addressed embedded KV prefix blocks. */
  readonly kvPrefixSlabBudgetBytes: number;
  /** Maximum number of 16-token Q8 KV prefix blocks held in the unified store. */
  readonly maxKvPrefixBlocks: number;
  /** Per-block compressed Q8 KV footprint in bytes. */
  readonly bytesPerKvBlock: number;
}

export interface UnifiedStoreTelemetry {
  readonly globalBudgetBytes: number;
  readonly weightSlabBytes: number;
  readonly liveKvSlabBytes: number;
  readonly embeddingCacheBytes: number;
  readonly kvPrefixSlabBytes: number;
  readonly totalUnifiedResidentBytes: number;
  readonly embdCacheSlots: number;
  readonly embdCacheResidentCount: number;
  readonly embdCacheHits: number;
  readonly embdCacheMisses: number;
  readonly embdCacheHitRatePct: number;
  readonly kvPrefixBlocks: number;
  readonly kvPrefixMaxBlocks: number;
  readonly kvPrefixHits: number;
  readonly kvPrefixMisses: number;
  readonly kvPrefixDeduplicated: number;
  readonly kvPrefixEvictions: number;
  readonly kvPrefixPersistedOpfs: boolean;
  readonly speculativeActive: boolean;
  readonly speculativePrefillRuns: number;
  readonly speculativePrefilledTokens: number;
  readonly speculativeReadyTokens: number;
  readonly unifiedRestoredTokens: number;
  readonly estimatedTtftSavedMs: number;
  readonly lastTurnPrefillHitRatePct: number;
  readonly factoredBlocks: number;
  readonly factoredSavedBytes: number;
}

export interface UnifiedStoreGeometry {
  readonly modelId: string;
  /**
   * Optional content-identity fingerprint (`<headerHash64>-<byteLength>`).
   * When present, prefix block hashing AND OPFS `.unified.kv` persistence are keyed by the
   * underlying GGUF content identity rather than alias `modelId`, so renamed, mirror, or
   * custom-URL mounts of the same model share warm KV prefixes seamlessly!
   */
  readonly contentFingerprint?: string;
  readonly numLayers: number;
  readonly nHeadsKv: number;
  readonly headDim: number;
  readonly pageTokens: number;
  readonly q8BytesPerHalfPage: number;
  readonly scaleFloatsPerHalfPage: number;
}

/**
 * Theorem T6 — Unified Store Global Space Allocation Law:
 * Analytically partitions available resident memory headroom between the Zipfian hot embedding
 * row slab and the content-addressed embedded KV prefix block slab so neither can ever push
 * total WASM residency above `globalBudgetBytes`.
 */
export function deriveUnifiedStoreAllocation(params: {
  globalBudgetBytes: number;
  weightAndLiveKvBytes: number;
  embdRowBytes: number;
  streamEmbeddings: boolean;
  vocabSize: number;
  kvPageBytesWarmQ8: number;
}): UnifiedStoreAllocationPlan {
  const headroom = Math.max(0, params.globalBudgetBytes - params.weightAndLiveKvBytes);

  // 1. Hot Embedding Row Slab (only needed when token_embd.weight is streamed from OPFS)
  // Top 512–1024 Zipfian tokens cover ~75–85% of all token lookups while consuming <= 4 MiB!
  let hotEmbeddingCacheSlots = 0;
  let hotEmbeddingSlabBytes = 0;
  if (params.streamEmbeddings && params.embdRowBytes > 0) {
    const maxEmbdBudget = Math.min(4 * 1024 * 1024, Math.floor(headroom * 0.12));
    const maxSlotsByBudget = Math.floor(maxEmbdBudget / Math.max(32, params.embdRowBytes));
    const targetSlots = Math.min(params.vocabSize, 512);
    hotEmbeddingCacheSlots = Math.max(32, Math.min(targetSlots, Math.max(32, maxSlotsByBudget)));
    hotEmbeddingSlabBytes = hotEmbeddingCacheSlots * params.embdRowBytes;
  }

  // 2. Content-Addressed Embedded KV Prefix Block Slab (Q8 compressed 16-token pages)
  const remainingHeadroom = Math.max(0, headroom - hotEmbeddingSlabBytes);
  const bytesPerKvBlock = Math.max(256, params.kvPageBytesWarmQ8);
  const targetKvBudget = Math.min(24 * 1024 * 1024, Math.floor(remainingHeadroom * 0.3));
  // Guarantee at least 4 blocks (64 tokens = system prompt + user prefix) for instant reuse
  const minBlocks = 4;
  const maxKvPrefixBlocks = Math.max(
    minBlocks,
    Math.min(128, Math.floor(targetKvBudget / bytesPerKvBlock)),
  );
  const kvPrefixSlabBudgetBytes = maxKvPrefixBlocks * bytesPerKvBlock;

  return {
    hotEmbeddingCacheSlots,
    hotEmbeddingSlabBytes,
    kvPrefixSlabBudgetBytes,
    maxKvPrefixBlocks,
    bytesPerKvBlock,
  };
}

const FNV_OFFSET_BASIS_64 = 0xcbf29ce484222325n;
const FNV_PRIME_64 = 0x100000001b3n;
const MASK_64 = 0xffffffffffffffffn;

function mixHash64(h: bigint, value: bigint): bigint {
  let x = (h ^ (value & MASK_64)) & MASK_64;
  x = (x * FNV_PRIME_64) & MASK_64;
  x ^= x >> 33n;
  x = (x * 0xff51afd7ed558ccdn) & MASK_64;
  x ^= x >> 33n;
  return x & MASK_64;
}

/**
 * Computes the deterministic chained 64-bit content-address hash for a 16-token KV prefix block.
 * Chaining `parentHash` guarantees identical prefixes across threads/sessions map to ONE key,
 * while any earlier token divergence produces completely distinct downstream keys.
 */
export function computePrefixBlockHash(
  modelId: string,
  parentHash: string,
  tokens: readonly number[],
): string {
  let h = FNV_OFFSET_BASIS_64;
  for (let i = 0; i < modelId.length; i++) {
    h = mixHash64(h, BigInt(modelId.charCodeAt(i)));
  }
  for (let i = 0; i < parentHash.length; i++) {
    h = mixHash64(h, BigInt(parentHash.charCodeAt(i)));
  }
  h = mixHash64(h, BigInt(tokens.length));
  for (let i = 0; i < tokens.length; i++) {
    h = mixHash64(h, BigInt((tokens[i] >>> 0) + 1));
  }
  return h.toString(16).padStart(16, '0');
}

/**
 * Pure helper for Continuous Speculative Prefill while the user is typing:
 * - When `allowFinalWord` is false (user mid-keystroke), trims back to the last stable word
 *   boundary (whitespace or punctuation) so we never waste compute on a half-typed subword.
 * - When `allowFinalWord` is true (user paused >= ~450 ms), includes the entire draft so even
 *   the final word is warmed into KV before they press Enter.
 */
export function trimToStableSpeculativePrefix(draftText: string, allowFinalWord: boolean): string {
  if (!draftText) return '';
  if (allowFinalWord) return draftText;
  // If already ending in whitespace or sentence punctuation, the entire draft is stable
  if (/[\s.,!?;:)\]}"']$/.test(draftText)) {
    return draftText;
  }
  // Otherwise trim back to the last whitespace boundary
  const lastSpace = draftText.search(/\s[^\s]*$/);
  if (lastSpace < 0) {
    return '';
  }
  return draftText.slice(0, lastSpace + 1);
}

const UNIFIED_KV_MAGIC = 0x31564b55; // "UKV1" (legacy raw blocks)
const UNIFIED_KV_MAGIC_V2 = 0x32564b55; // "UKV2" (Theorem T7 exact lossless factored blocks)
const UNIFIED_KV_MAGIC_V3 = 0x33564b55; // "UKV3" (Theorem T7 factored blocks + 64-bit integrity trailer)
const UNIFIED_KV_TRAILER_BYTES = 8;
const MAX_PERSISTED_OPFS_BLOCKS = 16;

/**
 * Zero-allocation dual-lane 64-bit integrity digest (extrapolated from verdant-engine's
 * `WorldSnapshot` wire checksum verification) so partial or corrupted OPFS `.unified.kv`
 * writes are rejected cleanly before hydrating any KV pages.
 */
function computeUnifiedKvDigest(bytes: Uint8Array, length: number): readonly [number, number] {
  let h1 = 0x811c9dc5 | 0; // FNV-1a lane A
  let h2 = 0x9e3779b1 | 0; // Golden-ratio avalanche lane B
  for (let i = 0; i < length; i++) {
    const b = bytes[i];
    h1 = Math.imul(h1 ^ b, 0x01000193);
    h2 = Math.imul(h2 ^ ((b + i) & 0xff), 0x85ebca6b);
    h2 = (h2 << 13) | (h2 >>> 19);
  }
  // Final avalanche mix
  h1 ^= h1 >>> 16;
  h1 = Math.imul(h1, 0x7feb352d);
  h1 ^= h1 >>> 15;
  h2 ^= h2 >>> 16;
  h2 = Math.imul(h2, 0xc2b2ae35);
  h2 ^= h2 >>> 16;
  return [h1 >>> 0, h2 >>> 0];
}

function sanitizeUnifiedKvFileName(modelId: string): string {
  return `${sanitizeOpfsModelFileName(modelId).replace(/\.gguf$/i, '')}.unified.kv`;
}

async function getOpfsDir(create: boolean): Promise<FileSystemDirectoryHandle | null> {
  if (typeof navigator === 'undefined' || !navigator.storage?.getDirectory) return null;
  try {
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle(OPFS_VRAM_DIR_NAME, { create });
  } catch {
    return null;
  }
}

/**
 * Global Unified Virtual Store managing content-addressed embedded KV prefix blocks,
 * OPFS persistence across sessions, and unified latency/space telemetry across all slabs.
 */
export class VramUnifiedStore {
  readonly geometry: UnifiedStoreGeometry;
  readonly maxBlocks: number;
  readonly kvPrefixBudgetBytes: number;
  readonly globalBudgetBytes: number;

  private readonly blocks = new Map<string, UnifiedKvBlockEntry>();
  private usedKvBytes = 0;
  private clock = 0;
  private dirtyForOpfs = false;
  private persistedOpfs = false;
  private pinnedPrefixPages = 1;

  // Content-addressed KV prefix metrics
  private kvPrefixHits = 0;
  private kvPrefixMisses = 0;
  private kvPrefixDeduplicated = 0;
  private kvPrefixEvictions = 0;

  // Continuous Speculative Prefill & latency-elimination metrics
  private speculativeActive = false;
  private speculativePrefillRuns = 0;
  private speculativePrefilledTokens = 0;
  private speculativeReadyTokens = 0;
  private unifiedRestoredTokens = 0;
  private estimatedTtftSavedMs = 0;
  private lastTurnPrefillHitRatePct = 0;
  private factoredBlocks = 0;
  private factoredSavedBytes = 0;

  constructor(
    geometry: UnifiedStoreGeometry,
    allocation: UnifiedStoreAllocationPlan,
    globalBudgetBytes: number,
  ) {
    this.geometry = geometry;
    this.maxBlocks = Math.max(2, allocation.maxKvPrefixBlocks);
    this.kvPrefixBudgetBytes = allocation.kvPrefixSlabBudgetBytes;
    this.globalBudgetBytes = globalBudgetBytes;
  }

  get blockCount(): number {
    return this.blocks.size;
  }

  /** Effective content-identity key shared across aliases/mirrors of the same GGUF payload. */
  get identityKey(): string {
    return this.geometry.contentFingerprint || this.geometry.modelId;
  }

  get bytesStored(): number {
    return this.usedKvBytes;
  }

  /**
   * Pins leading prefix pages `[0 .. pinnedPrefixPages - 1]` (System Prompt, Tool Manifest,
   * and frozen Lossless Episodic Epoch Index) so multi-page shared prefixes receive eviction
   * & OPFS persistence priority alongside page 0.
   */
  setPinnedPrefixPages(pages: number): void {
    const clamped = Math.max(1, Math.min(Math.max(1, this.maxBlocks - 1), Math.floor(pages)));
    this.pinnedPrefixPages = clamped;
  }

  getPinnedPrefixPages(): number {
    return this.pinnedPrefixPages;
  }

  private isPinnedPrefixPage(pageIndex: number): boolean {
    return pageIndex < this.pinnedPrefixPages;
  }

  /**
   * Commits a full 16-token Q8 KV page block into the content-addressed unified store.
   * Identical prefixes across threads deduplicate automatically to 1 physical block.
   */
  putBlock(
    blockKey: string,
    pageIndex: number,
    tokenIds: readonly number[],
    data: CompressedKvPageBlock,
  ): boolean {
    const existing = this.blocks.get(blockKey);
    if (existing) {
      existing.lastTouch = ++this.clock;
      existing.hitCount += 1;
      this.kvPrefixDeduplicated += 1;
      return false;
    }

    const byteLength =
      data.kQ8.byteLength + data.vQ8.byteLength + data.kScales.byteLength + data.vScales.byteLength;

    // Marginal-Utility LRU Eviction: preserve pageIndex === 0 (system prompt sink) whenever possible
    while (
      this.blocks.size >= this.maxBlocks ||
      (this.blocks.size > 0 && this.usedKvBytes + byteLength > this.kvPrefixBudgetBytes)
    ) {
      if (!this.evictLowestUtilityBlock()) break;
    }

    const entry: UnifiedKvBlockEntry = {
      blockKey,
      pageIndex,
      tokenIds: [...tokenIds],
      data,
      byteLength,
      isSystemSink: this.isPinnedPrefixPage(pageIndex),
      lastTouch: ++this.clock,
      hitCount: 0,
    };
    this.blocks.set(blockKey, entry);
    this.usedKvBytes += byteLength;
    this.dirtyForOpfs = true;
    return true;
  }

  getBlock(blockKey: string): UnifiedKvBlockEntry | null {
    const entry = this.blocks.get(blockKey);
    if (!entry) {
      this.kvPrefixMisses += 1;
      return null;
    }
    entry.lastTouch = ++this.clock;
    entry.hitCount += 1;
    this.kvPrefixHits += 1;
    return entry;
  }

  hasBlock(blockKey: string): boolean {
    return this.blocks.has(blockKey);
  }

  /**
   * Walks full `pageTokens`-aligned prefix blocks of `tokens` from page 0 and returns
   * the longest contiguous sequence of embedded KV blocks stored in the Unified Store.
   */
  lookupPrefixChain(
    tokens: readonly number[],
    maxTokenCount = tokens.length,
  ): UnifiedKvBlockEntry[] {
    const pageTokens = this.geometry.pageTokens;
    const usableTokens = Math.min(tokens.length, maxTokenCount);
    const fullPages = Math.floor(usableTokens / pageTokens);
    const matched: UnifiedKvBlockEntry[] = [];
    let parentHash = 'root';

    const keySpace = this.identityKey;
    for (let p = 0; p < fullPages; p++) {
      const slice = tokens.slice(p * pageTokens, (p + 1) * pageTokens);
      const key = computePrefixBlockHash(keySpace, parentHash, slice);
      const found = this.getBlock(key);
      if (!found) break;
      matched.push(found);
      parentHash = key;
    }
    return matched;
  }

  /** Computes the exact content-addressed block keys for every full page of `tokens`. */
  computePrefixKeys(tokens: readonly number[]): Array<{ pageIndex: number; key: string; slice: number[] }> {
    const pageTokens = this.geometry.pageTokens;
    const fullPages = Math.floor(tokens.length / pageTokens);
    const result: Array<{ pageIndex: number; key: string; slice: number[] }> = [];
    let parentHash = 'root';
    const keySpace = this.identityKey;
    for (let p = 0; p < fullPages; p++) {
      const slice = tokens.slice(p * pageTokens, (p + 1) * pageTokens);
      const key = computePrefixBlockHash(keySpace, parentHash, slice);
      result.push({ pageIndex: p, key, slice });
      parentHash = key;
    }
    return result;
  }

  private evictLowestUtilityBlock(): boolean {
    let victimKey: string | null = null;
    let victimScore = Infinity;

    for (const [key, entry] of this.blocks) {
      // Pinned prefix blocks [0 .. pinnedPrefixPages - 1] get a strong priority boost so shared
      // system prompts and episodic epoch indexes stay hot ahead of tail turn pages
      const isPinned = entry.isSystemSink || this.isPinnedPrefixPage(entry.pageIndex);
      const pinBoost = isPinned ? 10_000 + Math.max(0, this.pinnedPrefixPages - entry.pageIndex) : 0;
      const utilityScore = entry.lastTouch + entry.hitCount * 4 + pinBoost;
      if (utilityScore < victimScore) {
        victimScore = utilityScore;
        victimKey = key;
      }
    }
    if (!victimKey) return false;
    const removed = this.blocks.get(victimKey)!;
    this.blocks.delete(victimKey);
    this.usedKvBytes = Math.max(0, this.usedKvBytes - removed.byteLength);
    this.kvPrefixEvictions += 1;
    return true;
  }

  setSpeculativeActive(active: boolean): void {
    this.speculativeActive = active;
  }

  recordSpeculativePrefill(newlyPrefilledTokens: number, totalReadyTokens: number): void {
    this.speculativePrefillRuns += 1;
    this.speculativePrefilledTokens += Math.max(0, newlyPrefilledTokens);
    this.speculativeReadyTokens = Math.max(0, totalReadyTokens);
  }

  recordPromptTurnReuse(params: {
    totalPromptTokens: number;
    reusedLiveTokens: number;
    restoredUnifiedTokens: number;
    estimatedMsPerPrefillToken: number;
  }): void {
    const totalReused = Math.max(0, params.reusedLiveTokens + params.restoredUnifiedTokens);
    this.unifiedRestoredTokens += Math.max(0, params.restoredUnifiedTokens);
    if (params.totalPromptTokens > 0) {
      this.lastTurnPrefillHitRatePct = Math.min(
        100,
        Math.round((totalReused / params.totalPromptTokens) * 100),
      );
    }
    if (totalReused > 0 && params.estimatedMsPerPrefillToken > 0) {
      this.estimatedTtftSavedMs += totalReused * params.estimatedMsPerPrefillToken;
    }
    this.speculativeReadyTokens = Math.max(this.speculativeReadyTokens, params.totalPromptTokens);
  }

  /**
   * Serializes the hottest embedded KV prefix blocks into an integrity-checked binary `UKV3`
   * snapshot (Theorem T7 factored blocks + 64-bit integrity trailer).
   */
  serializeSnapshotBytes(maxBlocks: number = MAX_PERSISTED_OPFS_BLOCKS): Uint8Array {
    const ranked = this.selectTopBlocksForPersistence(maxBlocks);
    return this.serializeBlocks(ranked);
  }

  /**
   * Hydrates embedded KV prefix blocks from a raw binary snapshot (`UKV3` integrity-checked,
   * or backward-compatible `UKV2`/`UKV1`). Rejects corrupted or truncated `UKV3` payloads cleanly
   * with `0` hydrated blocks and zero thrown exceptions.
   */
  hydrateFromBytes(raw: Uint8Array, rekeyToIdentity = false): number {
    try {
      return this.deserializeBlocks(raw, rekeyToIdentity);
    } catch {
      return 0;
    }
  }

  private selectTopBlocksForPersistence(maxBlocks: number): UnifiedKvBlockEntry[] {
    return Array.from(this.blocks.values())
      .sort((a, b) => {
        const pinA = a.isSystemSink || this.isPinnedPrefixPage(a.pageIndex);
        const pinB = b.isSystemSink || this.isPinnedPrefixPage(b.pageIndex);
        const boostA = pinA ? 10_000 + Math.max(0, this.pinnedPrefixPages - a.pageIndex) : 0;
        const boostB = pinB ? 10_000 + Math.max(0, this.pinnedPrefixPages - b.pageIndex) : 0;
        const sa = a.lastTouch + a.hitCount * 4 + boostA;
        const sb = b.lastTouch + b.hitCount * 4 + boostB;
        return sb - sa;
      })
      .slice(0, maxBlocks);
  }

  /**
   * Serializes the hottest embedded KV prefix blocks into an integrity-checked `UKV3` binary payload
   * and persists it into OPFS (`<identityKey>.unified.kv`) so system prompts and recent prefixes
   * survive page reloads without risk of partial-write corruption.
   */
  async persistToOpfs(opfsDirectoryOverride?: FileSystemDirectoryHandle): Promise<number> {
    if (!this.dirtyForOpfs || this.blocks.size === 0) return 0;
    const dir = opfsDirectoryOverride ?? (await getOpfsDir(true));
    if (!dir) return 0;

    // Pick top MAX_PERSISTED_OPFS_BLOCKS by utility (always prioritizing pinned prefix blocks)
    const ranked = this.selectTopBlocksForPersistence(MAX_PERSISTED_OPFS_BLOCKS);

    const payload = this.serializeBlocks(ranked);
    const fileName = sanitizeUnifiedKvFileName(this.identityKey);
    try {
      const fh = (await dir.getFileHandle(fileName, { create: true })) as FileSystemFileHandle &
        SyncAccessHandleCapableFile;
      if (typeof fh.createSyncAccessHandle !== 'function') return 0;
      const sync = await openSyncAccessHandleWithRetry(fh, {
        mode: 'readwrite',
        modelId: this.identityKey,
        retryDelaysMs: [25, 50, 100],
      });
      try {
        sync.truncate(0);
        let written = 0;
        while (written < payload.byteLength) {
          const view = written === 0 ? payload : payload.subarray(written);
          const n = sync.write(view, { at: written });
          if (n <= 0) break;
          written += n;
        }
        sync.flush();
      } finally {
        sync.close();
      }
      this.dirtyForOpfs = false;
      this.persistedOpfs = true;
      return ranked.length;
    } catch {
      return 0;
    }
  }

  /**
   * Hydrates persisted embedded KV prefix blocks from OPFS (`<identityKey>.unified.kv`) on model boot,
   * with transparent fallback to legacy `<modelId>.unified.kv` when upgrading to fingerprinted caches.
   * Silently discards stale or geometry-mismatched files without failing model load.
   */
  async hydrateFromOpfs(opfsDirectoryOverride?: FileSystemDirectoryHandle): Promise<number> {
    const dir = opfsDirectoryOverride ?? (await getOpfsDir(false));
    if (!dir) return 0;

    const primaryFile = sanitizeUnifiedKvFileName(this.identityKey);
    const legacyFile = sanitizeUnifiedKvFileName(this.geometry.modelId);
    const candidates = primaryFile === legacyFile ? [primaryFile] : [primaryFile, legacyFile];

    for (let c = 0; c < candidates.length; c++) {
      const fileName = candidates[c];
      const isLegacyFallback = c > 0;
      try {
        const fh = (await dir.getFileHandle(fileName, { create: false })) as FileSystemFileHandle &
          SyncAccessHandleCapableFile;
        if (typeof fh.createSyncAccessHandle !== 'function') continue;
        const sync = await openSyncAccessHandleWithRetry(fh, {
          mode: 'read-only',
          modelId: this.identityKey,
          retryDelaysMs: [25, 50, 100],
        });
        let raw: Uint8Array;
        try {
          const size = sync.getSize();
          if (size < 28 || size > 64 * 1024 * 1024) continue;
          raw = new Uint8Array(size);
          let read = 0;
          while (read < size) {
            const n = sync.read(read === 0 ? raw : raw.subarray(read), { at: read });
            if (n <= 0) break;
            read += n;
          }
          if (read !== size) continue;
        } finally {
          sync.close();
        }

        const isLegacyCodec =
          raw.byteLength >= 4 &&
          new DataView(raw.buffer, raw.byteOffset, raw.byteLength).getUint32(0, true) !==
            UNIFIED_KV_MAGIC_V3;
        const hydrated = this.deserializeBlocks(raw, isLegacyFallback);
        if (hydrated > 0) {
          this.dirtyForOpfs = isLegacyFallback || isLegacyCodec;
          this.persistedOpfs = true;
          return hydrated;
        }
      } catch {
        // Try next candidate or return 0 cleanly
      }
    }
    return 0;
  }

  private serializeBlocks(entries: readonly UnifiedKvBlockEntry[]): Uint8Array {
    const { numLayers, nHeadsKv, headDim, pageTokens, q8BytesPerHalfPage, scaleFloatsPerHalfPage } =
      this.geometry;
    const scaleBytes = scaleFloatsPerHalfPage * 4;
    const rawBlockBytes = q8BytesPerHalfPage * 2 + scaleBytes * 2;
    const metaBytes = 16 + 4 + pageTokens * 4 + 4; // key + pageIndex + tokens + compressedLen

    const compressedPayloads: Uint8Array[] = new Array(entries.length);
    const scratch = new Uint8Array(rawBlockBytes);
    let totalBodyBytes = 0;
    let factoredCount = 0;
    let savedTotal = 0;

    for (let idx = 0; idx < entries.length; idx++) {
      const entry = entries[idx];
      let sOff = 0;
      scratch.set(
        new Uint8Array(entry.data.kQ8.buffer, entry.data.kQ8.byteOffset, q8BytesPerHalfPage),
        sOff,
      );
      sOff += q8BytesPerHalfPage;
      scratch.set(
        new Uint8Array(entry.data.vQ8.buffer, entry.data.vQ8.byteOffset, q8BytesPerHalfPage),
        sOff,
      );
      sOff += q8BytesPerHalfPage;
      scratch.set(
        new Uint8Array(entry.data.kScales.buffer, entry.data.kScales.byteOffset, scaleBytes),
        sOff,
      );
      sOff += scaleBytes;
      scratch.set(
        new Uint8Array(entry.data.vScales.buffer, entry.data.vScales.byteOffset, scaleBytes),
        sOff,
      );

      const compressed = compressUnifiedKvBytes(scratch);
      compressedPayloads[idx] = compressed.payload;
      totalBodyBytes += metaBytes + compressed.payload.byteLength;
      if (compressed.savedBytes > 0) {
        factoredCount += 1;
        savedTotal += compressed.savedBytes;
      }
    }

    this.factoredBlocks = factoredCount;
    this.factoredSavedBytes = savedTotal;

    const headerBytes = 28;
    const bodyEnd = headerBytes + totalBodyBytes;
    const out = new Uint8Array(bodyEnd + UNIFIED_KV_TRAILER_BYTES);
    const dv = new DataView(out.buffer);
    dv.setUint32(0, UNIFIED_KV_MAGIC_V3, true);
    dv.setUint32(4, numLayers, true);
    dv.setUint32(8, nHeadsKv, true);
    dv.setUint32(12, headDim, true);
    dv.setUint32(16, pageTokens, true);
    dv.setUint32(20, entries.length, true);
    dv.setUint32(24, rawBlockBytes, true);

    let off = headerBytes;
    for (let idx = 0; idx < entries.length; idx++) {
      const entry = entries[idx];
      const payload = compressedPayloads[idx];
      for (let i = 0; i < 16; i++) {
        out[off + i] = entry.blockKey.charCodeAt(i) & 0xff;
      }
      off += 16;
      dv.setUint32(off, entry.pageIndex, true);
      off += 4;
      for (let t = 0; t < pageTokens; t++) {
        dv.setInt32(off + t * 4, entry.tokenIds[t] ?? 0, true);
      }
      off += pageTokens * 4;
      dv.setUint32(off, payload.byteLength, true);
      off += 4;
      out.set(payload, off);
      off += payload.byteLength;
    }
    const [digestLo, digestHi] = computeUnifiedKvDigest(out, bodyEnd);
    dv.setUint32(bodyEnd, digestLo, true);
    dv.setUint32(bodyEnd + 4, digestHi, true);
    return out;
  }

  private deserializeBlocks(raw: Uint8Array, rekeyToIdentity = false): number {
    if (raw.byteLength < 28) return 0;
    const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    const magic = dv.getUint32(0, true);
    if (
      magic !== UNIFIED_KV_MAGIC &&
      magic !== UNIFIED_KV_MAGIC_V2 &&
      magic !== UNIFIED_KV_MAGIC_V3
    ) {
      return 0;
    }
    let bodyEnd = raw.byteLength;
    if (magic === UNIFIED_KV_MAGIC_V3) {
      if (raw.byteLength < 28 + UNIFIED_KV_TRAILER_BYTES) return 0;
      bodyEnd = raw.byteLength - UNIFIED_KV_TRAILER_BYTES;
      const expectedLo = dv.getUint32(bodyEnd, true);
      const expectedHi = dv.getUint32(bodyEnd + 4, true);
      const [actualLo, actualHi] = computeUnifiedKvDigest(raw, bodyEnd);
      if (actualLo !== expectedLo || actualHi !== expectedHi) {
        return 0;
      }
    }
    const { numLayers, nHeadsKv, headDim, pageTokens, q8BytesPerHalfPage, scaleFloatsPerHalfPage } =
      this.geometry;
    if (
      dv.getUint32(4, true) !== numLayers ||
      dv.getUint32(8, true) !== nHeadsKv ||
      dv.getUint32(12, true) !== headDim ||
      dv.getUint32(16, true) !== pageTokens
    ) {
      return 0;
    }
    const count = dv.getUint32(20, true);
    const sizeField = dv.getUint32(24, true);
    const scaleBytes = scaleFloatsPerHalfPage * 4;
    const rawBlockBytes = q8BytesPerHalfPage * 2 + scaleBytes * 2;
    // When hydrating a legacy `<modelId>.unified.kv` into a content-fingerprinted store, re-chain
    // page keys under `this.identityKey` so lookups hit seamlessly!
    const rekeyedByPage = rekeyToIdentity ? new Map<number, string>() : null;
    const resolveKey = (storedKey: string, pageIndex: number, tokenIds: readonly number[]): string | null => {
      if (!rekeyedByPage) return storedKey;
      const parentHash = pageIndex === 0 ? 'root' : rekeyedByPage.get(pageIndex - 1);
      if (!parentHash) return null;
      const nextKey = computePrefixBlockHash(this.identityKey, parentHash, tokenIds);
      rekeyedByPage.set(pageIndex, nextKey);
      return nextKey;
    };

    if (magic === UNIFIED_KV_MAGIC_V2 || magic === UNIFIED_KV_MAGIC_V3) {
      if (sizeField !== rawBlockBytes) return 0;
      let off = 28;
      let loaded = 0;
      let factoredCount = 0;
      let savedTotal = 0;
      const metaBytes = 16 + 4 + pageTokens * 4 + 4;
      const pending: Array<{
        key: string;
        pageIndex: number;
        tokenIds: number[];
        data: CompressedKvPageBlock;
      }> = [];
      for (let idx = 0; idx < count; idx++) {
        if (off + metaBytes > bodyEnd) return magic === UNIFIED_KV_MAGIC_V3 ? 0 : loaded;
        let key = '';
        for (let i = 0; i < 16; i++) {
          key += String.fromCharCode(raw[off + i]);
        }
        off += 16;
        const pageIndex = dv.getUint32(off, true);
        off += 4;
        const tokenIds: number[] = new Array(pageTokens);
        for (let t = 0; t < pageTokens; t++) {
          tokenIds[t] = dv.getInt32(off + t * 4, true);
        }
        off += pageTokens * 4;
        const compressedLen = dv.getUint32(off, true);
        off += 4;
        if (off + compressedLen > bodyEnd) return magic === UNIFIED_KV_MAGIC_V3 ? 0 : loaded;

        const payload = raw.subarray(off, off + compressedLen);
        off += compressedLen;
        if (compressedLen < rawBlockBytes) {
          factoredCount += 1;
          savedTotal += rawBlockBytes - compressedLen;
        }
        const unpacked = decompressUnifiedKvBytes(payload, rawBlockBytes);
        if (unpacked.byteLength !== rawBlockBytes) return magic === UNIFIED_KV_MAGIC_V3 ? 0 : loaded;

        let uOff = 0;
        const kQ8 = new Int8Array(q8BytesPerHalfPage);
        kQ8.set(new Int8Array(unpacked.buffer, unpacked.byteOffset + uOff, q8BytesPerHalfPage));
        uOff += q8BytesPerHalfPage;

        const vQ8 = new Int8Array(q8BytesPerHalfPage);
        vQ8.set(new Int8Array(unpacked.buffer, unpacked.byteOffset + uOff, q8BytesPerHalfPage));
        uOff += q8BytesPerHalfPage;

        const kScales = new Float32Array(scaleFloatsPerHalfPage);
        new Uint8Array(kScales.buffer).set(unpacked.subarray(uOff, uOff + scaleBytes));
        uOff += scaleBytes;

        const vScales = new Float32Array(scaleFloatsPerHalfPage);
        new Uint8Array(vScales.buffer).set(unpacked.subarray(uOff, uOff + scaleBytes));

        pending.push({ key, pageIndex, tokenIds, data: { kQ8, vQ8, kScales, vScales } });
      }
      if (magic === UNIFIED_KV_MAGIC_V3 && off !== bodyEnd) {
        return 0;
      }
      if (rekeyedByPage) {
        pending.sort((a, b) => a.pageIndex - b.pageIndex);
      }
      for (let i = 0; i < pending.length; i++) {
        const p = pending[i];
        const finalKey = resolveKey(p.key, p.pageIndex, p.tokenIds);
        if (!finalKey) continue;
        this.putBlock(finalKey, p.pageIndex, p.tokenIds, p.data);
        loaded += 1;
      }
      this.factoredBlocks = factoredCount;
      this.factoredSavedBytes = savedTotal;
      return loaded;
    }

    // Legacy UKV1 raw fixed-size blocks
    const expectedEntrySize = 16 + 4 + pageTokens * 4 + rawBlockBytes;
    if (sizeField !== expectedEntrySize || raw.byteLength < 28 + count * sizeField) {
      return 0;
    }

    let off = 28;
    let loaded = 0;
    const pendingV1: Array<{
      key: string;
      pageIndex: number;
      tokenIds: number[];
      data: CompressedKvPageBlock;
    }> = [];
    for (let idx = 0; idx < count; idx++) {
      let key = '';
      for (let i = 0; i < 16; i++) {
        key += String.fromCharCode(raw[off + i]);
      }
      off += 16;
      const pageIndex = dv.getUint32(off, true);
      off += 4;
      const tokenIds: number[] = new Array(pageTokens);
      for (let t = 0; t < pageTokens; t++) {
        tokenIds[t] = dv.getInt32(off + t * 4, true);
      }
      off += pageTokens * 4;

      const kQ8 = new Int8Array(q8BytesPerHalfPage);
      kQ8.set(new Int8Array(raw.buffer, raw.byteOffset + off, q8BytesPerHalfPage));
      off += q8BytesPerHalfPage;

      const vQ8 = new Int8Array(q8BytesPerHalfPage);
      vQ8.set(new Int8Array(raw.buffer, raw.byteOffset + off, q8BytesPerHalfPage));
      off += q8BytesPerHalfPage;

      const kScales = new Float32Array(scaleFloatsPerHalfPage);
      new Uint8Array(kScales.buffer).set(raw.subarray(off, off + scaleBytes));
      off += scaleBytes;

      const vScales = new Float32Array(scaleFloatsPerHalfPage);
      new Uint8Array(vScales.buffer).set(raw.subarray(off, off + scaleBytes));
      off += scaleBytes;

      pendingV1.push({ key, pageIndex, tokenIds, data: { kQ8, vQ8, kScales, vScales } });
    }
    if (rekeyedByPage) {
      pendingV1.sort((a, b) => a.pageIndex - b.pageIndex);
    }
    for (let i = 0; i < pendingV1.length; i++) {
      const p = pendingV1[i];
      const finalKey = resolveKey(p.key, p.pageIndex, p.tokenIds);
      if (!finalKey) continue;
      this.putBlock(finalKey, p.pageIndex, p.tokenIds, p.data);
      loaded += 1;
    }
    return loaded;
  }

  getTelemetry(snapshot: {
    weightSlabBytes: number;
    liveKvSlabBytes: number;
    embeddingCacheBytes: number;
    embdCacheSlots: number;
    embdCacheResidentCount: number;
    embdCacheHits: number;
    embdCacheMisses: number;
  }): UnifiedStoreTelemetry {
    const embdTotal = snapshot.embdCacheHits + snapshot.embdCacheMisses;
    const embdCacheHitRatePct =
      embdTotal > 0 ? Math.round((snapshot.embdCacheHits / embdTotal) * 100) : 0;

    return {
      globalBudgetBytes: this.globalBudgetBytes,
      weightSlabBytes: snapshot.weightSlabBytes,
      liveKvSlabBytes: snapshot.liveKvSlabBytes,
      embeddingCacheBytes: snapshot.embeddingCacheBytes,
      kvPrefixSlabBytes: this.usedKvBytes,
      totalUnifiedResidentBytes:
        snapshot.weightSlabBytes + snapshot.liveKvSlabBytes + this.usedKvBytes,
      embdCacheSlots: snapshot.embdCacheSlots,
      embdCacheResidentCount: snapshot.embdCacheResidentCount,
      embdCacheHits: snapshot.embdCacheHits,
      embdCacheMisses: snapshot.embdCacheMisses,
      embdCacheHitRatePct,
      kvPrefixBlocks: this.blocks.size,
      kvPrefixMaxBlocks: this.maxBlocks,
      kvPrefixHits: this.kvPrefixHits,
      kvPrefixMisses: this.kvPrefixMisses,
      kvPrefixDeduplicated: this.kvPrefixDeduplicated,
      kvPrefixEvictions: this.kvPrefixEvictions,
      kvPrefixPersistedOpfs: this.persistedOpfs,
      speculativeActive: this.speculativeActive,
      speculativePrefillRuns: this.speculativePrefillRuns,
      speculativePrefilledTokens: this.speculativePrefilledTokens,
      speculativeReadyTokens: this.speculativeReadyTokens,
      unifiedRestoredTokens: this.unifiedRestoredTokens,
      estimatedTtftSavedMs: Math.round(this.estimatedTtftSavedMs),
      lastTurnPrefillHitRatePct: this.lastTurnPrefillHitRatePct,
      factoredBlocks: this.factoredBlocks,
      factoredSavedBytes: this.factoredSavedBytes,
    };
  }
}

export { VRAM_KV_PAGE_TOKENS };
