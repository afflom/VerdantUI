/**
 * Unified Expert Discovery & Priority-Ordered Retrieval Scheduler
 * (`src/engine/unifiedExpertScheduler.ts`)
 *
 * Optimizes how VerdantUI's Unified Model discovers, retrieves, prewarms, and reuses weights
 * across many models and experts without ever pulling an entire monolithic file upfront:
 *   1. Zero-download GGUF manifest discovery (`discoverRemoteExpertManifest`): fetches ONLY the
 *      compact header slice via HTTP Range (+ OPFS CAS dedup) to learn exact architecture,
 *      quantization, MoE expert topology, and tensor byte map before streaming any weights.
 *   2. Priority-ordered extent planning (`planPriorityWeightExtents`): partitions uncached weights
 *      into interruptible <= 4 MiB slices ordered by execution salience:
 *        - Priority 1: Early transformer layers (or all MoE attention + router gates)
 *        - Priority 2: Remaining dense body layers
 *        - Priority 3: Streamed LM-head output projection stripes
 *        - Excluded: Cold Sparse MoE expert tables (e.g. 120/128 unactivated experts stay virtual
 *          until a router gate actually selects them!)
 *   3. Consent-time initial backbone selection (`selectInitialUnifiedBackbone`): immediately
 *      warms the optimal demand-first System Stream backbone the moment the user consents.
 */

import { getGgmlTypeTrait, parseGGUF, type ParsedGgufModel } from './gguf';
import { MODEL_CATALOG, type CatalogModel } from './modelCatalog';
import type { ByteRange } from './vramBackingStore';
import type { DemandCasStoreLike } from './vramBackingStore';

export type PrewarmPriorityTier = 'early-attention' | 'body-layers' | 'lm-head';

export interface PrewarmSlice {
  readonly tier: PrewarmPriorityTier;
  readonly layerIndex?: number;
  readonly ranges: readonly ByteRange[];
  readonly byteLength: number;
}

export interface ExpertPrewarmPlan {
  readonly slices: readonly PrewarmSlice[];
  readonly totalTargetBytes: number;
  readonly alreadyCachedBytes: number;
  readonly missingBytes: number;
  readonly percentWarm: number;
}

export interface DiscoveredExpertManifest {
  readonly url: string;
  readonly contentFingerprint: string;
  readonly fileByteLength: number;
  readonly architecture: string;
  readonly name: string;
  readonly blockCount: number;
  readonly embeddingLength: number;
  readonly vocabSize: number;
  readonly isMoe: boolean;
  readonly expertCount: number;
  readonly expertUsedCount: number;
  readonly tensorCount: number;
  readonly primaryQuantization: string;
  readonly headerBytesFetched: number;
  readonly etag?: string;
}

const DEFAULT_PREWARM_SLICE_BYTES = 32 * 1024 * 1024; // 32 MiB interruptible slice (2 x 16 MiB direct-CDN coalesced fetches)
const INITIAL_DISCOVERY_HEADER_BYTES = 64 * 1024; // 64 KiB initial probe
const MAX_DISCOVERY_HEADER_BYTES = 8 * 1024 * 1024; // 8 MiB cap for 150K-vocab headers

function splitRangeIntoSlices(
  range: ByteRange,
  maxSliceBytes: number,
  tier: PrewarmPriorityTier,
  layerIndex?: number,
): PrewarmSlice[] {
  if (range.byteLength <= 0) return [];
  const slices: PrewarmSlice[] = [];
  let offset = range.fileOffset;
  let remaining = range.byteLength;
  while (remaining > 0) {
    const chunk = Math.min(remaining, maxSliceBytes);
    slices.push({
      tier,
      layerIndex,
      ranges: [{ fileOffset: offset, byteLength: chunk }],
      byteLength: chunk,
    });
    offset += chunk;
    remaining -= chunk;
  }
  return slices;
}

function packRangesIntoSlices(
  ranges: readonly ByteRange[],
  maxSliceBytes: number,
  tier: PrewarmPriorityTier,
  layerIndex?: number,
): PrewarmSlice[] {
  const out: PrewarmSlice[] = [];
  let bucket: ByteRange[] = [];
  let bucketBytes = 0;

  const flush = () => {
    if (bucket.length > 0) {
      out.push({ tier, layerIndex, ranges: bucket, byteLength: bucketBytes });
      bucket = [];
      bucketBytes = 0;
    }
  };

  for (const r of ranges) {
    if (r.byteLength <= 0) continue;
    if (r.byteLength > maxSliceBytes) {
      flush();
      out.push(...splitRangeIntoSlices(r, maxSliceBytes, tier, layerIndex));
      continue;
    }
    if (bucketBytes + r.byteLength > maxSliceBytes && bucket.length > 0) {
      flush();
    }
    bucket.push(r);
    bucketBytes += r.byteLength;
  }
  flush();
  return out;
}

/**
 * Builds an intelligent, priority-ordered, interruptible prewarm plan from a loaded model's
 * layer and LM-head ranges. Cold Sparse MoE expert tables are NEVER included — they remain
 * purely on-demand per router activation!
 */
export function planPriorityWeightExtents(params: {
  readonly totalLayers: number;
  readonly isMoe: boolean;
  readonly getLayerRanges: (layerIdx: number) => readonly ByteRange[];
  readonly lmHeadStripes: readonly ByteRange[];
  readonly isRangeCached?: (fileOffset: number, byteLength: number) => boolean;
  readonly maxSliceBytes?: number;
}): ExpertPrewarmPlan {
  const maxSlice = Math.max(256 * 1024, params.maxSliceBytes ?? DEFAULT_PREWARM_SLICE_BYTES);
  const earlyLayerCutoff = params.isMoe
    ? params.totalLayers // For Sparse MoE, layer ranges are ONLY tiny attention+router tensors!
    : Math.max(1, Math.ceil(params.totalLayers / 3));

  let totalTargetBytes = 0;
  let alreadyCachedBytes = 0;
  const earlySlices: PrewarmSlice[] = [];
  const bodySlices: PrewarmSlice[] = [];
  const lmHeadSlices: PrewarmSlice[] = [];

  const filterMissing = (ranges: readonly ByteRange[]): ByteRange[] => {
    const missing: ByteRange[] = [];
    for (const r of ranges) {
      if (r.byteLength <= 0) continue;
      totalTargetBytes += r.byteLength;
      if (params.isRangeCached?.(r.fileOffset, r.byteLength)) {
        alreadyCachedBytes += r.byteLength;
      } else {
        missing.push(r);
      }
    }
    return missing;
  };

  for (let l = 0; l < params.totalLayers; l++) {
    const missing = filterMissing(params.getLayerRanges(l));
    if (missing.length === 0) continue;
    if (l < earlyLayerCutoff) {
      earlySlices.push(...packRangesIntoSlices(missing, maxSlice, 'early-attention', l));
    } else {
      bodySlices.push(...packRangesIntoSlices(missing, maxSlice, 'body-layers', l));
    }
  }

  const missingLm = filterMissing(params.lmHeadStripes);
  if (missingLm.length > 0) {
    lmHeadSlices.push(...packRangesIntoSlices(missingLm, maxSlice, 'lm-head'));
  }

  const slices = [...earlySlices, ...bodySlices, ...lmHeadSlices];
  const missingBytes = Math.max(0, totalTargetBytes - alreadyCachedBytes);
  const percentWarm =
    totalTargetBytes <= 0
      ? 100
      : Math.min(100, Math.round((alreadyCachedBytes / totalTargetBytes) * 100));

  return {
    slices,
    totalTargetBytes,
    alreadyCachedBytes,
    missingBytes,
    percentWarm,
  };
}

/**
 * Selects the optimal demand-first System Stream backbone to mount immediately when the user
 * consents to browser streaming (or boots with saved consent).
 * - If any catalog model is already warm in OPFS, re-attaches the remembered/warmest one at 0 bytes.
 * - Otherwise mounts a fast, capable demand-first backbone (`qwen3-4b-q4_k_m` or `qwen2.5-1.5b`
 *   on low-core devices) in ~1-2s (~4-8 MB header+norms) while background slices warm silently.
 */
export function selectInitialUnifiedBackbone(params: {
  readonly cachedModelIds: readonly string[];
  readonly lastModelId: string | null;
  readonly hardwareConcurrency?: number;
}): CatalogModel {
  if (params.lastModelId && params.cachedModelIds.includes(params.lastModelId)) {
    const remembered = MODEL_CATALOG.find((m) => m.id === params.lastModelId);
    if (remembered) return remembered;
  }
  if (params.cachedModelIds.length > 0) {
    const cachedModels = MODEL_CATALOG.filter((m) => params.cachedModelIds.includes(m.id));
    if (cachedModels.length > 0) {
      return cachedModels[0]!;
    }
  }
  const cores = params.hardwareConcurrency ?? 4;
  const preferredId = cores >= 4 ? 'qwen3-4b-q4_k_m' : 'qwen2.5-1.5b-instruct-q4_k_m';
  return (
    MODEL_CATALOG.find((m) => m.id === preferredId) ??
    MODEL_CATALOG.find((m) => m.id === 'qwen2.5-1.5b-instruct-q4_k_m') ??
    MODEL_CATALOG[0]!
  );
}

/**
 * Zero-download GGUF manifest discovery: fetches ONLY the compact leading header bytes via
 * HTTP Range (cached in `OpfsCasStore` when supplied) to discover any model's architecture,
 * MoE expert geometry, quantization, and exact tensor map without pulling gigabytes of weights.
 */
export async function discoverRemoteExpertManifest(params: {
  readonly url: string;
  readonly modelId?: string;
  readonly fetchFn?: typeof fetch;
  readonly casStore?: DemandCasStoreLike;
  readonly signal?: AbortSignal;
}): Promise<DiscoveredExpertManifest> {
  const fetchImpl = params.fetchFn ?? fetch;
  let probeBytes = INITIAL_DISCOVERY_HEADER_BYTES;
  let totalFileBytes = 0;
  let etag: string | undefined;
  let parsed: ParsedGgufModel | null = null;
  let fetchedBytes = 0;

  while (probeBytes <= MAX_DISCOVERY_HEADER_BYTES) {
    const res = await fetchImpl(params.url, {
      headers: { Range: `bytes=0-${probeBytes - 1}` },
      signal: params.signal,
    });
    if (!res.ok && res.status !== 206) {
      throw new Error(`HTTP ${res.status} discovering expert manifest at ${params.url}`);
    }
    etag = res.headers.get('etag') ?? etag;
    const contentRange = res.headers.get('content-range');
    if (contentRange) {
      const m = contentRange.match(/\/(\d+)\s*$/);
      if (m) totalFileBytes = Number(m[1]);
    }
    const buf = new Uint8Array(await res.arrayBuffer());
    fetchedBytes = buf.byteLength;
    if (totalFileBytes <= 0) totalFileBytes = buf.byteLength;

    try {
      parsed = parseGGUF(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
      if (params.casStore && buf.byteLength > 0) {
        await params.casStore
          .resolveOrStoreSlab({
            contentFingerprint: etag ? `etag:${etag.replace(/"/g, '')}:${totalFileBytes}` : `url:${params.url}:${totalFileBytes}`,
            slabIndex: 0,
            ownerModelId: params.modelId ?? params.url,
            sourceUrl: params.url,
            etag,
            fetchFn: async () => buf,
          })
          .catch(() => undefined);
      }
      break;
    } catch {
      if (buf.byteLength < probeBytes || probeBytes >= MAX_DISCOVERY_HEADER_BYTES) {
        throw new Error(`Unable to parse GGUF header manifest within ${probeBytes} bytes for ${params.url}.`);
      }
      probeBytes = Math.min(MAX_DISCOVERY_HEADER_BYTES, probeBytes * 4);
    }
  }

  if (!parsed) {
    throw new Error(`Failed to discover GGUF manifest for ${params.url}.`);
  }

  const hp = parsed.hparams;
  const expertCount = hp.expertCount ?? 0;
  const isMoe =
    expertCount > 0 || parsed.tensorMap.has('blk.0.ffn_gate_exps.weight');
  const firstQuant = parsed.tensors[0] ? getGgmlTypeTrait(parsed.tensors[0].type).name : hp.primaryQuantization;

  return {
    url: params.url,
    contentFingerprint: etag
      ? `etag:${etag.replace(/"/g, '')}:${totalFileBytes}`
      : `url:${params.url}:${totalFileBytes}`,
    fileByteLength: totalFileBytes,
    architecture: hp.architecture,
    name: hp.name || params.modelId || params.url,
    blockCount: hp.blockCount,
    embeddingLength: hp.embeddingLength,
    vocabSize: hp.vocabSize,
    isMoe,
    expertCount,
    expertUsedCount: isMoe ? Math.max(1, hp.expertUsedCount ?? 2) : 0,
    tensorCount: parsed.tensors.length,
    primaryQuantization: hp.primaryQuantization || firstQuant,
    headerBytesFetched: fetchedBytes,
    etag,
  };
}
