/**
 * Content-Addressed OPFS Boot & Runtime Acceleration Caches (`VramBootCache`).
 *
 * Extrapolates OPFS content deduplication from the model downloader straight into the
 * inference runtime by keying immutable boot & warm-state sidecars on the underlying GGUF's
 * deterministic content fingerprint (`<headerHash64>-<byteLength>`):
 *   1. `<fp>.hdr.bin` (`VHDR` v1): Parsed GGUF metadata, tensor directory, and native-JSON
 *      vocab/merge slabs + raw TypedArray scores/types — skips 8–12 MiB remote Range header
 *      reads AND 300,000 per-token JS DataView/TextDecoder crossings on 150K-vocab models!
 *   2. `<fp>.rope.bin` (`VROP` v1): Precomputed RoPE F64 inverse frequencies and F32 cos/sin
 *      LUTs — eliminates up to 262,144 trig evaluations on cold reload with bit-exact parity.
 *   3. `<fp>.embd.bin` (`VEMB` v1): Zipfian Hot Embedding Row snapshot for `streamEmbeddings`
 *      (`opfs-paged`) models — shares warm token embeddings across reloads and renamed aliases.
 *
 * All reads use lock-free `{ mode: 'read-only' }` OPFS handles and degrade silently to `null`
 * on miss, corruption, or signature mismatch so boot caches can never fail a model load.
 */

import type { GgufMetadataValue, GgufTensorInfo, LlamaHyperParams, ParsedGgufModel } from './gguf';
import type { RopeTables } from './verdantKernels';
import {
  computeHeaderFingerprint64,
  getOpfsVramDirectory,
  isOpfsSyncSupported,
  openSyncAccessHandleWithRetry,
  sanitizeOpfsBaseName,
  writeAllSync,
  type SyncAccessHandleCapableFile,
  type VramBackingStore,
} from './vramBackingStore';

const HDR_CACHE_MAGIC = 0x52444856; // "VHDR"
const HDR_CACHE_VERSION = 1;
const HDR_CACHE_MAX_BYTES = 64 * 1024 * 1024;

const ROPE_CACHE_MAGIC = 0x504f5256; // "VROP"
const ROPE_CACHE_VERSION = 1;
const ROPE_CACHE_MAX_BYTES = 16 * 1024 * 1024;

const EMBD_CACHE_MAGIC = 0x424d4556; // "VEMB"
const EMBD_CACHE_VERSION = 1;
const EMBD_CACHE_MAX_BYTES = 16 * 1024 * 1024;

const utf8Encoder = new TextEncoder();
const utf8Decoder = new TextDecoder('utf-8');

/** Formats a deterministic cross-alias content fingerprint `<headerHash64>-<byteLength>`. */
export function formatContentFingerprint(headerHash64: string, byteLength: number): string {
  return `${headerHash64}-${Math.max(0, Math.floor(byteLength))}`;
}

/**
 * Synchronously computes the deterministic GGUF content fingerprint (`<headerHash64>-<byteLength>`)
 * from the first 4 KiB of any local or prewarmed `VramBackingStore`.
 */
export function computeBackingContentFingerprint(backing: VramBackingStore): string {
  const probeLen = Math.min(4096, Math.max(0, backing.byteLength));
  if (probeLen === 0) {
    return formatContentFingerprint('0000000000000000', 0);
  }
  const buf = new Uint8Array(probeLen);
  const read = backing.readRange(0, buf, 0, probeLen);
  const hash = computeHeaderFingerprint64(read === probeLen ? buf : buf.subarray(0, Math.max(0, read)));
  return formatContentFingerprint(hash, backing.byteLength);
}

/**
 * Asynchronously computes the deterministic GGUF content fingerprint without blocking XHRs on
 * cold `opfs-demand` stores whose slab 0 is not yet resident.
 */
export async function computeBackingContentFingerprintAsync(
  backing: VramBackingStore,
  signal?: AbortSignal,
): Promise<string> {
  const probeLen = Math.min(4096, Math.max(0, backing.byteLength));
  if (probeLen === 0) {
    return formatContentFingerprint('0000000000000000', 0);
  }
  if (backing.readRangesDirectAsync) {
    const [head] = await backing.readRangesDirectAsync(
      [{ fileOffset: 0, byteLength: probeLen }],
      { signal },
    );
    const hash = computeHeaderFingerprint64(head);
    return formatContentFingerprint(hash, backing.byteLength);
  }
  return computeBackingContentFingerprint(backing);
}

async function readOpfsSidecarBytes(
  fileName: string,
  maxBytes: number,
  modelId: string,
  dirOverride?: FileSystemDirectoryHandle,
): Promise<Uint8Array | null> {
  if (!dirOverride && !isOpfsSyncSupported()) return null;
  const dir = dirOverride ?? (await getOpfsVramDirectory(false));
  if (!dir) return null;
  try {
    const fh = (await dir.getFileHandle(fileName, { create: false })) as FileSystemFileHandle &
      SyncAccessHandleCapableFile;
    if (typeof fh.createSyncAccessHandle !== 'function') return null;
    const sync = await openSyncAccessHandleWithRetry(fh, {
      mode: 'read-only',
      modelId,
      retryDelaysMs: [25, 50],
    });
    try {
      const size = sync.getSize();
      if (size <= 0 || size > maxBytes) return null;
      const raw = new Uint8Array(size);
      let read = 0;
      while (read < size) {
        const n = sync.read(read === 0 ? raw : raw.subarray(read), { at: read });
        if (n <= 0) break;
        read += n;
      }
      return read === size ? raw : null;
    } finally {
      sync.close();
    }
  } catch {
    return null;
  }
}

async function writeOpfsSidecarBytes(
  fileName: string,
  payload: Uint8Array,
  maxBytes: number,
  modelId: string,
  dirOverride?: FileSystemDirectoryHandle,
): Promise<boolean> {
  if (payload.byteLength <= 0 || payload.byteLength > maxBytes) return false;
  if (!dirOverride && !isOpfsSyncSupported()) return false;
  const dir = dirOverride ?? (await getOpfsVramDirectory(true));
  if (!dir) return false;
  try {
    const fh = (await dir.getFileHandle(fileName, { create: true })) as FileSystemFileHandle &
      SyncAccessHandleCapableFile;
    if (typeof fh.createSyncAccessHandle !== 'function') return false;
    const sync = await openSyncAccessHandleWithRetry(fh, {
      mode: 'readwrite',
      modelId,
      retryDelaysMs: [25, 50, 100],
    });
    try {
      sync.truncate(0);
      writeAllSync(sync, payload, 0);
      sync.flush();
    } finally {
      sync.close();
    }
    return true;
  } catch {
    return false;
  }
}

// ============================================================================
// 1. Parsed GGUF Header + Tokenizer Boot Cache (`<key>.hdr.bin`, VHDR v1)
// ============================================================================

interface SerializedHeaderMeta {
  readonly fingerprint: string;
  readonly version: number;
  readonly alignment: number;
  readonly dataOffset: number;
  readonly fileByteLength: number;
  readonly hparams: LlamaHyperParams;
  readonly tensors: readonly GgufTensorInfo[];
  readonly eotTokenId?: number;
  readonly chatTemplate?: string | null;
}

/**
 * Serializes a parsed GGUF header + tokenizer vocab/merges into a compact `VHDR` v1 binary slab:
 * - JSON core metadata + tensor descriptors (~25–60 KiB)
 * - JSON string arrays for vocab & merges (parsed in 1 native C++ V8 JSON.parse call vs 300K JS loops)
 * - Raw `Float32Array` scores + raw `Int32Array` token_type bytes (0 JS parsing overhead)
 */
export function serializeParsedGgufBootCache(
  parsed: ParsedGgufModel,
  contentFingerprint: string,
): Uint8Array {
  const eotMeta = parsed.metadata['tokenizer.ggml.eot_token_id'];
  const chatMeta = parsed.metadata['tokenizer.chat_template'];
  const core: SerializedHeaderMeta = {
    fingerprint: contentFingerprint,
    version: parsed.version,
    alignment: parsed.alignment,
    dataOffset: parsed.dataOffset,
    fileByteLength: parsed.fileByteLength,
    hparams: parsed.hparams,
    tensors: parsed.tensors,
    eotTokenId: typeof eotMeta === 'number' ? eotMeta : undefined,
    chatTemplate: typeof chatMeta === 'string' ? chatMeta : parsed.hparams.chatTemplate,
  };

  const rawTokens = parsed.metadata['tokenizer.ggml.tokens'];
  const tokensArr = Array.isArray(rawTokens) ? rawTokens.map((t) => String(t)) : [];
  const rawMerges = parsed.metadata['tokenizer.ggml.merges'];
  const mergesArr = Array.isArray(rawMerges) ? rawMerges.map((m) => String(m)) : [];

  const rawScores = parsed.metadata['tokenizer.ggml.scores'];
  let scoresF32: Float32Array;
  if (rawScores instanceof Float32Array) {
    scoresF32 = rawScores;
  } else if (Array.isArray(rawScores) && rawScores.length > 0) {
    scoresF32 = new Float32Array(rawScores.length);
    for (let i = 0; i < rawScores.length; i++) {
      const v = rawScores[i];
      scoresF32[i] = typeof v === 'number' ? v : 0;
    }
  } else {
    scoresF32 = new Float32Array(0);
  }

  const rawTypes = parsed.metadata['tokenizer.ggml.token_type'];
  let typesI32: Int32Array;
  if (rawTypes instanceof Int32Array) {
    typesI32 = rawTypes;
  } else if (Array.isArray(rawTypes) && rawTypes.length > 0) {
    typesI32 = new Int32Array(rawTypes.length);
    for (let i = 0; i < rawTypes.length; i++) {
      const v = rawTypes[i];
      typesI32[i] = typeof v === 'number' ? v | 0 : 1;
    }
  } else {
    typesI32 = new Int32Array(0);
  }

  const coreBytes = utf8Encoder.encode(JSON.stringify(core));
  const tokensBytes = utf8Encoder.encode(JSON.stringify(tokensArr));
  const mergesBytes = mergesArr.length > 0 ? utf8Encoder.encode(JSON.stringify(mergesArr)) : new Uint8Array(0);
  const scoresBytes = new Uint8Array(scoresF32.buffer, scoresF32.byteOffset, scoresF32.byteLength);
  const typesBytes = new Uint8Array(typesI32.buffer, typesI32.byteOffset, typesI32.byteLength);

  // Header: 32 bytes = magic(4), version(4), coreLen(4), tokensLen(4), mergesLen(4), scoresCount(4), typesCount(4), reserved(4)
  const headerBytes = 32;
  const totalBytes =
    headerBytes +
    coreBytes.byteLength +
    tokensBytes.byteLength +
    mergesBytes.byteLength +
    scoresBytes.byteLength +
    typesBytes.byteLength;

  const out = new Uint8Array(totalBytes);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, HDR_CACHE_MAGIC, true);
  dv.setUint32(4, HDR_CACHE_VERSION, true);
  dv.setUint32(8, coreBytes.byteLength, true);
  dv.setUint32(12, tokensBytes.byteLength, true);
  dv.setUint32(16, mergesBytes.byteLength, true);
  dv.setUint32(20, scoresF32.length, true);
  dv.setUint32(24, typesI32.length, true);
  dv.setUint32(28, 0, true);

  let off = headerBytes;
  out.set(coreBytes, off);
  off += coreBytes.byteLength;
  out.set(tokensBytes, off);
  off += tokensBytes.byteLength;
  if (mergesBytes.byteLength > 0) {
    out.set(mergesBytes, off);
    off += mergesBytes.byteLength;
  }
  if (scoresBytes.byteLength > 0) {
    out.set(scoresBytes, off);
    off += scoresBytes.byteLength;
  }
  if (typesBytes.byteLength > 0) {
    out.set(typesBytes, off);
  }
  return out;
}

/** Deserializes and strictly validates a `VHDR` v1 boot cache against `(contentFingerprint, fileByteLength)`. */
export function deserializeParsedGgufBootCache(
  raw: Uint8Array,
  expectedFingerprint: string,
  expectedFileByteLength: number,
): ParsedGgufModel | null {
  if (raw.byteLength < 32) return null;
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  if (dv.getUint32(0, true) !== HDR_CACHE_MAGIC || dv.getUint32(4, true) !== HDR_CACHE_VERSION) {
    return null;
  }
  const coreLen = dv.getUint32(8, true);
  const tokensLen = dv.getUint32(12, true);
  const mergesLen = dv.getUint32(16, true);
  const scoresCount = dv.getUint32(20, true);
  const typesCount = dv.getUint32(24, true);
  const scoresByteLen = scoresCount * 4;
  const typesByteLen = typesCount * 4;
  const expectedTotal = 32 + coreLen + tokensLen + mergesLen + scoresByteLen + typesByteLen;
  if (coreLen === 0 || tokensLen === 0 || expectedTotal !== raw.byteLength) {
    return null;
  }

  try {
    let off = 32;
    const core = JSON.parse(utf8Decoder.decode(raw.subarray(off, off + coreLen))) as Partial<SerializedHeaderMeta>;
    off += coreLen;
    if (
      core.fingerprint !== expectedFingerprint ||
      core.fileByteLength !== expectedFileByteLength ||
      typeof core.version !== 'number' ||
      typeof core.alignment !== 'number' ||
      typeof core.dataOffset !== 'number' ||
      !core.hparams ||
      !Array.isArray(core.tensors)
    ) {
      return null;
    }

    const tokens = JSON.parse(utf8Decoder.decode(raw.subarray(off, off + tokensLen))) as unknown;
    off += tokensLen;
    if (!Array.isArray(tokens) || tokens.length === 0) return null;

    let merges: string[] | undefined;
    if (mergesLen > 0) {
      const parsedMerges = JSON.parse(utf8Decoder.decode(raw.subarray(off, off + mergesLen))) as unknown;
      if (!Array.isArray(parsedMerges)) return null;
      merges = parsedMerges as string[];
      off += mergesLen;
    }

    let scores: Float32Array | undefined;
    if (scoresCount > 0) {
      const alignedBuf = new ArrayBuffer(scoresByteLen);
      new Uint8Array(alignedBuf).set(raw.subarray(off, off + scoresByteLen));
      scores = new Float32Array(alignedBuf);
      off += scoresByteLen;
    }

    let tokenTypes: Int32Array | undefined;
    if (typesCount > 0) {
      const alignedBuf = new ArrayBuffer(typesByteLen);
      new Uint8Array(alignedBuf).set(raw.subarray(off, off + typesByteLen));
      tokenTypes = new Int32Array(alignedBuf);
    }

    const metadata: Record<string, GgufMetadataValue> = {
      'tokenizer.ggml.tokens': tokens as unknown as GgufMetadataValue,
    };
    if (scores) {
      // Store typed Float32Array directly (consumed zero-copy by GgufTokenizer)
      metadata['tokenizer.ggml.scores'] = scores as unknown as GgufMetadataValue;
    }
    if (tokenTypes) {
      metadata['tokenizer.ggml.token_type'] = tokenTypes as unknown as GgufMetadataValue;
    }
    if (merges) {
      metadata['tokenizer.ggml.merges'] = merges;
    }
    if (typeof core.eotTokenId === 'number') {
      metadata['tokenizer.ggml.eot_token_id'] = core.eotTokenId;
    }
    if (typeof core.chatTemplate === 'string') {
      metadata['tokenizer.chat_template'] = core.chatTemplate;
    }

    const tensorMap = new Map<string, GgufTensorInfo>();
    for (let i = 0; i < core.tensors.length; i++) {
      const t = core.tensors[i];
      tensorMap.set(t.name, t);
    }

    return {
      version: core.version,
      alignment: core.alignment,
      dataOffset: core.dataOffset,
      fileByteLength: core.fileByteLength,
      metadata,
      tensors: core.tensors,
      tensorMap,
      hparams: core.hparams,
      rawBytes: new Uint8Array(0),
    };
  } catch {
    return null;
  }
}

export async function readParsedGgufBootCacheFromOpfs(
  cacheKey: string,
  expectedFingerprint: string,
  expectedFileByteLength: number,
  dirOverride?: FileSystemDirectoryHandle,
): Promise<ParsedGgufModel | null> {
  const fileName = `${sanitizeOpfsBaseName(cacheKey)}.hdr.bin`;
  const raw = await readOpfsSidecarBytes(fileName, HDR_CACHE_MAX_BYTES, cacheKey, dirOverride);
  if (!raw) return null;
  return deserializeParsedGgufBootCache(raw, expectedFingerprint, expectedFileByteLength);
}

export async function writeParsedGgufBootCacheToOpfs(
  cacheKey: string,
  contentFingerprint: string,
  parsed: ParsedGgufModel,
  dirOverride?: FileSystemDirectoryHandle,
): Promise<boolean> {
  const payload = serializeParsedGgufBootCache(parsed, contentFingerprint);
  const fileName = `${sanitizeOpfsBaseName(cacheKey)}.hdr.bin`;
  return writeOpfsSidecarBytes(fileName, payload, HDR_CACHE_MAX_BYTES, cacheKey, dirOverride);
}

// ============================================================================
// 2. Precomputed RoPE Table Boot Cache (`<key>.rope.bin`, VROP v1)
// ============================================================================

export interface RopeCacheParams {
  readonly maxCtx: number;
  readonly ropeDim: number;
  readonly freqBase: number;
  readonly freqScale: number;
}

export function serializeRopeTablesCache(tables: RopeTables, params: RopeCacheParams): Uint8Array {
  const freqsBytes = tables.freqs.byteLength;
  const cosBytes = tables.cos.byteLength;
  const sinBytes = tables.sin.byteLength;
  const headerBytes = 40;
  const out = new Uint8Array(headerBytes + freqsBytes + cosBytes + sinBytes);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, ROPE_CACHE_MAGIC, true);
  dv.setUint32(4, ROPE_CACHE_VERSION, true);
  dv.setUint32(8, tables.maxCtx, true);
  dv.setUint32(12, params.ropeDim, true);
  dv.setUint32(16, tables.half, true);
  dv.setUint32(20, 0, true);
  dv.setFloat64(24, params.freqBase, true);
  dv.setFloat64(32, params.freqScale, true);

  let off = headerBytes;
  out.set(new Uint8Array(tables.freqs.buffer, tables.freqs.byteOffset, freqsBytes), off);
  off += freqsBytes;
  out.set(new Uint8Array(tables.cos.buffer, tables.cos.byteOffset, cosBytes), off);
  off += cosBytes;
  out.set(new Uint8Array(tables.sin.buffer, tables.sin.byteOffset, sinBytes), off);
  return out;
}

export function deserializeRopeTablesCache(
  raw: Uint8Array,
  expected: RopeCacheParams,
): RopeTables | null {
  if (raw.byteLength < 40) return null;
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  if (dv.getUint32(0, true) !== ROPE_CACHE_MAGIC || dv.getUint32(4, true) !== ROPE_CACHE_VERSION) {
    return null;
  }
  const maxCtx = dv.getUint32(8, true);
  const ropeDim = dv.getUint32(12, true);
  const half = dv.getUint32(16, true);
  const freqBase = dv.getFloat64(24, true);
  const freqScale = dv.getFloat64(32, true);
  const expectedHalf = Math.floor(expected.ropeDim / 2);
  if (
    maxCtx !== expected.maxCtx ||
    ropeDim !== expected.ropeDim ||
    half !== expectedHalf ||
    freqBase !== expected.freqBase ||
    freqScale !== expected.freqScale
  ) {
    return null;
  }
  const freqsBytes = half * 8;
  const trigBytes = maxCtx * half * 4;
  if (raw.byteLength !== 40 + freqsBytes + trigBytes * 2) {
    return null;
  }

  let off = 40;
  const freqsBuf = new ArrayBuffer(freqsBytes);
  new Uint8Array(freqsBuf).set(raw.subarray(off, off + freqsBytes));
  off += freqsBytes;

  const cosBuf = new ArrayBuffer(trigBytes);
  new Uint8Array(cosBuf).set(raw.subarray(off, off + trigBytes));
  off += trigBytes;

  const sinBuf = new ArrayBuffer(trigBytes);
  new Uint8Array(sinBuf).set(raw.subarray(off, off + trigBytes));

  return {
    maxCtx,
    half,
    freqs: new Float64Array(freqsBuf),
    cos: new Float32Array(cosBuf),
    sin: new Float32Array(sinBuf),
  };
}

export async function readRopeTablesCacheFromOpfs(
  cacheKey: string,
  expected: RopeCacheParams,
  dirOverride?: FileSystemDirectoryHandle,
): Promise<RopeTables | null> {
  const fileName = `${sanitizeOpfsBaseName(cacheKey)}.rope.bin`;
  const raw = await readOpfsSidecarBytes(fileName, ROPE_CACHE_MAX_BYTES, cacheKey, dirOverride);
  if (!raw) return null;
  return deserializeRopeTablesCache(raw, expected);
}

export async function writeRopeTablesCacheToOpfs(
  cacheKey: string,
  tables: RopeTables,
  params: RopeCacheParams,
  dirOverride?: FileSystemDirectoryHandle,
): Promise<boolean> {
  const payload = serializeRopeTablesCache(tables, params);
  const fileName = `${sanitizeOpfsBaseName(cacheKey)}.rope.bin`;
  return writeOpfsSidecarBytes(fileName, payload, ROPE_CACHE_MAX_BYTES, cacheKey, dirOverride);
}

// ============================================================================
// 3. Content-Addressed Hot Embedding Row Cache (`<key>.embd.bin`, VEMB v1)
// ============================================================================

export interface HotEmbeddingRowEntry {
  readonly tokenId: number;
  readonly bytes: Uint8Array;
}

export interface HotEmbeddingCacheSignature {
  readonly dataOffset: number;
  readonly backingByteLength: number;
  readonly embdOffset: number;
  readonly embdType: number;
  readonly embdRowBytes: number;
}

export function serializeHotEmbeddingsCache(
  sig: HotEmbeddingCacheSignature,
  rows: readonly HotEmbeddingRowEntry[],
): Uint8Array {
  const headerBytes = 44;
  const entryBytes = 4 + sig.embdRowBytes;
  const validRows = rows.filter((r) => r.bytes.byteLength === sig.embdRowBytes);
  const out = new Uint8Array(headerBytes + validRows.length * entryBytes);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, EMBD_CACHE_MAGIC, true);
  dv.setUint32(4, EMBD_CACHE_VERSION, true);
  dv.setFloat64(8, sig.dataOffset, true);
  dv.setFloat64(16, sig.backingByteLength, true);
  dv.setFloat64(24, sig.embdOffset, true);
  dv.setUint32(32, sig.embdType, true);
  dv.setUint32(36, sig.embdRowBytes, true);
  dv.setUint32(40, validRows.length, true);

  let off = headerBytes;
  for (let i = 0; i < validRows.length; i++) {
    dv.setInt32(off, validRows[i].tokenId, true);
    off += 4;
    out.set(validRows[i].bytes, off);
    off += sig.embdRowBytes;
  }
  return out;
}

export function deserializeHotEmbeddingsCache(
  raw: Uint8Array,
  expected: HotEmbeddingCacheSignature,
): HotEmbeddingRowEntry[] | null {
  if (raw.byteLength < 44) return null;
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  if (dv.getUint32(0, true) !== EMBD_CACHE_MAGIC || dv.getUint32(4, true) !== EMBD_CACHE_VERSION) {
    return null;
  }
  if (
    dv.getFloat64(8, true) !== expected.dataOffset ||
    dv.getFloat64(16, true) !== expected.backingByteLength ||
    dv.getFloat64(24, true) !== expected.embdOffset ||
    dv.getUint32(32, true) !== expected.embdType ||
    dv.getUint32(36, true) !== expected.embdRowBytes
  ) {
    return null;
  }
  const count = dv.getUint32(40, true);
  const entryBytes = 4 + expected.embdRowBytes;
  if (raw.byteLength !== 44 + count * entryBytes) return null;

  const rows: HotEmbeddingRowEntry[] = new Array(count);
  let off = 44;
  for (let i = 0; i < count; i++) {
    const tokenId = dv.getInt32(off, true);
    off += 4;
    const bytes = raw.slice(off, off + expected.embdRowBytes);
    off += expected.embdRowBytes;
    rows[i] = { tokenId, bytes };
  }
  return rows;
}

export async function readHotEmbeddingsCacheFromOpfs(
  cacheKey: string,
  expected: HotEmbeddingCacheSignature,
  dirOverride?: FileSystemDirectoryHandle,
): Promise<HotEmbeddingRowEntry[] | null> {
  const fileName = `${sanitizeOpfsBaseName(cacheKey)}.embd.bin`;
  const raw = await readOpfsSidecarBytes(fileName, EMBD_CACHE_MAX_BYTES, cacheKey, dirOverride);
  if (!raw) return null;
  return deserializeHotEmbeddingsCache(raw, expected);
}

export async function writeHotEmbeddingsCacheToOpfs(
  cacheKey: string,
  sig: HotEmbeddingCacheSignature,
  rows: readonly HotEmbeddingRowEntry[],
  dirOverride?: FileSystemDirectoryHandle,
): Promise<boolean> {
  if (rows.length === 0) return false;
  const payload = serializeHotEmbeddingsCache(sig, rows);
  const fileName = `${sanitizeOpfsBaseName(cacheKey)}.embd.bin`;
  return writeOpfsSidecarBytes(fileName, payload, EMBD_CACHE_MAX_BYTES, cacheKey, dirOverride);
}
