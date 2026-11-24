/**
 * OPFS-Backed vRAM Virtual DIMM Storage Layer (`afflom/vRAM` + `afflom/vTensor`).
 *
 * Implements the browser-native equivalent of `afflom/vRAM` (`host/ram_file.c`,
 * `app/src/CapacityPolicy.lex.tex`, and commits `f1185aff` / `b0d40626`):
 *   - Logical virtual capacity C_virtual = ceiling(p, g, b) = min(p, g, b) decoupled from
 *     physical resident WASM RAM.
 *   - Zero-spike streaming ingestion (4 MiB chunks) from Hugging Face URLs or local `.gguf`
 *     Files directly into Origin Private File System (OPFS) via `FileSystemSyncAccessHandle`.
 *   - Coalesced contiguous range restores (`b0d40626`): merges adjacent tensor extents separated
 *     by <= alignment gap into minimal synchronous `readRange()` syscalls straight into WASM
 *     linear memory views with zero intermediate JS copies.
 *   - Transparent `MemoryBackingStore` fallback for Node.js tests and restricted browsing contexts.
 */

import type { GgufTensorInfo } from './gguf';

export type VramBackingKind = 'opfs-sync' | 'memory';

export interface VramBackingStore {
  readonly kind: VramBackingKind;
  readonly modelId: string;
  readonly byteLength: number;
  readonly isPersistent: boolean;
  /**
   * Synchronously reads `length` bytes starting at `fileOffset` into `dst` at `dstOffset`.
   * Returns the exact number of bytes read.
   */
  readRange(fileOffset: number, dst: Uint8Array, dstOffset?: number, length?: number): number;
  /** Releases OS / OPFS file descriptor handles without deleting persisted OPFS data. */
  close(): void;
}

export interface CoalescedExtent {
  /** Absolute byte offset inside the GGUF backing file. */
  readonly fileOffset: number;
  /** Total contiguous bytes to restore in one backing read. */
  readonly byteLength: number;
  /** Individual tensor slices packed inside this coalesced extent. */
  readonly slices: readonly {
    readonly tensor: GgufTensorInfo;
    /** Byte offset of this tensor relative to `fileOffset` (guaranteed >= 0). */
    readonly relativeOffset: number;
  }[];
}

export interface OpfsStoredModelEntry {
  readonly modelId: string;
  readonly fileName: string;
  readonly byteLength: number;
  readonly lastModified: number;
}

export interface OpfsPartialModelEntry {
  readonly modelId: string;
  readonly fileName: string;
  readonly receivedBytes: number;
  readonly totalBytes: number;
  readonly url?: string;
  readonly lastModified: number;
}

export interface OpfsQuotaSnapshot {
  readonly supported: boolean;
  readonly usageBytes: number;
  readonly quotaBytes: number;
  readonly storedModels: readonly OpfsStoredModelEntry[];
  readonly partialModels?: readonly OpfsPartialModelEntry[];
}

export type StreamIngestStage = 'cached' | 'downloading' | 'retrying' | 'verifying';

export interface StreamIngestProgress {
  readonly receivedBytes: number;
  readonly totalBytes: number;
  readonly percent: number;
  readonly backingKind: VramBackingKind;
  readonly bytesPerSecond?: number;
  readonly etaSeconds?: number | null;
  readonly resumedFromBytes?: number;
  readonly retryAttempt?: number;
  readonly maxRetries?: number;
  readonly stage?: StreamIngestStage;
  readonly note?: string;
}

export const OPFS_VRAM_DIR_NAME = 'vram-gguf-v1';
export const OPFS_STREAM_CHUNK_BYTES = 4 * 1024 * 1024; // 4 MiB streaming window
export const OPFS_MAX_SYNC_IO_BYTES = 16 * 1024 * 1024; // 16 MiB chunked sync I/O bound
export const DOWNLOAD_MAX_RETRIES = 6;
export const DOWNLOAD_STALL_TIMEOUT_MS = 25_000;

/**
 * Exact translation of `afflom/vRAM` `app/src/CapacityPolicy.lex.tex`:
 *   ceiling(p, g, b) = min(p, min(g, b))
 * Proves monotone backing and bounded logical residency from (physical, guest, backing) limits.
 */
export function vramCapacityCeiling(
  physicalLimitBytes: number,
  guestTargetBytes: number,
  backingQuotaBytes: number,
): number {
  const p = Math.max(0, Math.floor(physicalLimitBytes));
  const g = Math.max(0, Math.floor(guestTargetBytes));
  const b = Math.max(0, Math.floor(backingQuotaBytes));
  return Math.min(p, g, b);
}

/**
 * Translation of `afflom/vRAM` commit `b0d40626`
 * ("Coalesce contiguous Verdant backing slots during DIMM range restores"):
 * Groups tensors sorted by absolute file offset and coalesces adjacent spans whose inter-tensor
 * padding gap is `<= maxGapBytes` into single contiguous backing reads.
 */
export function coalesceTensorExtents(
  tensors: readonly GgufTensorInfo[],
  dataOffset: number,
  maxGapBytes = 4096,
): CoalescedExtent[] {
  if (tensors.length === 0) return [];

  const sorted = [...tensors].sort((a, b) => a.offset - b.offset);
  const extents: CoalescedExtent[] = [];

  let curStart = dataOffset + sorted[0].offset;
  let curEnd = curStart + sorted[0].byteSize;
  let curSlices: Array<{ tensor: GgufTensorInfo; relativeOffset: number }> = [
    { tensor: sorted[0], relativeOffset: 0 },
  ];

  for (let i = 1; i < sorted.length; i++) {
    const t = sorted[i];
    const tStart = dataOffset + t.offset;
    const tEnd = tStart + t.byteSize;
    const gap = tStart - curEnd;

    if (gap >= 0 && gap <= maxGapBytes) {
      curSlices.push({ tensor: t, relativeOffset: tStart - curStart });
      if (tEnd > curEnd) curEnd = tEnd;
    } else {
      extents.push({
        fileOffset: curStart,
        byteLength: curEnd - curStart,
        slices: curSlices,
      });
      curStart = tStart;
      curEnd = tEnd;
      curSlices = [{ tensor: t, relativeOffset: 0 }];
    }
  }

  extents.push({
    fileOffset: curStart,
    byteLength: curEnd - curStart,
    slices: curSlices,
  });

  return extents;
}

/**
 * Empirically calibrates the host backing store's sequential read bandwidth (MB/s) using a
 * bounded 4 MiB coalesced sweep (median of 3 passes) so Prism Theorem T1/T4b derives exact
 * hardware-specific I/O bounds on both browser OPFS (~250–600 MB/s) and native page-cache.
 */
export function calibrateBackingReadBandwidthMBs(backing: VramBackingStore): number {
  const probeBytes = Math.min(backing.byteLength, 4 * 1024 * 1024);
  if (probeBytes < 64 * 1024) {
    return backing.kind === 'opfs-sync' ? 400 : 4000;
  }
  const chunkBytes = Math.min(probeBytes, 1024 * 1024);
  const scratch = new Uint8Array(chunkBytes);
  const samplesMBs: number[] = [];

  for (let pass = 0; pass < 3; pass++) {
    let totalRead = 0;
    const t0 = performance.now();
    for (let off = 0; off + chunkBytes <= probeBytes; off += chunkBytes) {
      totalRead += backing.readRange(off, scratch, 0, chunkBytes);
    }
    const elapsedSec = Math.max(0.0001, (performance.now() - t0) / 1000);
    if (totalRead > 0) {
      samplesMBs.push((totalRead / (1024 * 1024)) / elapsedSec);
    }
  }

  if (samplesMBs.length === 0) {
    return backing.kind === 'opfs-sync' ? 400 : 4000;
  }
  samplesMBs.sort((a, b) => a - b);
  const median = samplesMBs[Math.floor(samplesMBs.length / 2)];
  return Math.max(50, Math.min(12000, Number(median.toFixed(1))));
}

/**
 * In-memory `VramBackingStore` implementation for Node.js tests, bundled small models,
 * and browsers where OPFS `FileSystemSyncAccessHandle` is unavailable.
 */
export class MemoryBackingStore implements VramBackingStore {
  readonly kind: VramBackingKind = 'memory';
  readonly modelId: string;
  readonly byteLength: number;
  readonly isPersistent = false;
  private bytes: Uint8Array;

  constructor(modelId: string, source: ArrayBuffer | Uint8Array) {
    this.modelId = modelId;
    this.bytes = source instanceof Uint8Array ? source : new Uint8Array(source);
    this.byteLength = this.bytes.byteLength;
  }

  readRange(fileOffset: number, dst: Uint8Array, dstOffset = 0, length = dst.byteLength - dstOffset): number {
    if (fileOffset < 0 || fileOffset >= this.byteLength || length <= 0) {
      return 0;
    }
    const readable = Math.min(length, this.byteLength - fileOffset, dst.byteLength - dstOffset);
    if (readable <= 0) return 0;
    dst.set(this.bytes.subarray(fileOffset, fileOffset + readable), dstOffset);
    return readable;
  }

  /** Direct zero-copy view when caller knows the backing is in-memory. */
  rawView(): Uint8Array {
    return this.bytes;
  }

  close(): void {
    // Allow GC when backing is closed
    this.bytes = new Uint8Array(0);
  }
}

/** Minimal typed subset of the Web Worker OPFS `FileSystemSyncAccessHandle` API. */
export interface SyncAccessHandleLike {
  getSize(): number;
  read(buffer: ArrayBufferView, options?: { at?: number }): number;
  write(buffer: ArrayBufferView, options?: { at?: number }): number;
  truncate(newSize: number): void;
  flush(): void;
  close(): void;
}

/**
 * Synchronous OPFS-backed `VramBackingStore` running inside a Dedicated Web Worker.
 * Reads directly into WASM linear memory `Uint8Array` views via `SyncAccessHandle.read()`.
 */
export class OpfsSyncBackingStore implements VramBackingStore {
  readonly kind: VramBackingKind = 'opfs-sync';
  readonly modelId: string;
  readonly byteLength: number;
  readonly isPersistent = true;
  private handle: SyncAccessHandleLike | null;

  constructor(modelId: string, handle: SyncAccessHandleLike, byteLength?: number) {
    this.modelId = modelId;
    this.handle = handle;
    this.byteLength = byteLength ?? handle.getSize();
  }

  readRange(fileOffset: number, dst: Uint8Array, dstOffset = 0, length = dst.byteLength - dstOffset): number {
    if (!this.handle) {
      throw new Error(`OPFS backing store "${this.modelId}" is already closed.`);
    }
    if (fileOffset < 0 || fileOffset >= this.byteLength || length <= 0) {
      return 0;
    }
    const readable = Math.min(length, this.byteLength - fileOffset, dst.byteLength - dstOffset);
    if (readable <= 0) return 0;

    // Loop in <= 16 MiB chunks to survive partial OS/IPC SyncAccessHandle reads on huge extents
    let totalRead = 0;
    while (totalRead < readable) {
      const chunkLen = Math.min(OPFS_MAX_SYNC_IO_BYTES, readable - totalRead);
      const start = dstOffset + totalRead;
      const targetView =
        start === 0 && chunkLen === dst.byteLength ? dst : dst.subarray(start, start + chunkLen);
      const n = this.handle.read(targetView, { at: fileOffset + totalRead });
      if (n <= 0) break;
      totalRead += n;
    }
    return totalRead;
  }

  close(): void {
    if (this.handle) {
      try {
        this.handle.close();
      } catch {
        // Ignore duplicate close
      }
      this.handle = null;
    }
  }
}

/** Sanitizes a model identifier into a deterministic OPFS filename. */
export function sanitizeOpfsModelFileName(modelId: string): string {
  const safe = modelId.replace(/[^a-zA-Z0-9._-]+/g, '_').slice(0, 96) || 'model';
  return `${safe}.gguf`;
}

/** Detects whether Origin Private File System with Worker SyncAccessHandle is available. */
export function isOpfsSyncSupported(): boolean {
  try {
    const nav = typeof navigator !== 'undefined' ? navigator : undefined;
    if (!nav || !nav.storage || typeof nav.storage.getDirectory !== 'function') {
      return false;
    }
    const fshProto = (
      globalThis as unknown as {
        FileSystemFileHandle?: { prototype?: { createSyncAccessHandle?: unknown } };
      }
    ).FileSystemFileHandle?.prototype;
    return typeof fshProto?.createSyncAccessHandle === 'function';
  } catch {
    return false;
  }
}

async function getOpfsVramDirectory(create = true): Promise<FileSystemDirectoryHandle | null> {
  if (typeof navigator === 'undefined' || !navigator.storage?.getDirectory) {
    return null;
  }
  try {
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle(OPFS_VRAM_DIR_NAME, { create });
  } catch {
    return null;
  }
}

interface PartialDownloadMeta {
  url: string;
  totalBytes: number;
  etag?: string;
  lastModified?: string;
  receivedBytes: number;
  updatedAt: number;
}

/** Per-model serialization queue preventing OPFS `NoModificationAllowedError` on rapid retry/cancel. */
const opfsModelLocks = new Map<string, Promise<void>>();

async function withModelOpfsLock<T>(modelId: string, fn: () => Promise<T>): Promise<T> {
  const key = sanitizeOpfsModelFileName(modelId);
  const prev = opfsModelLocks.get(key) ?? Promise.resolve();
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const chained = prev.catch(() => {}).then(() => gate);
  opfsModelLocks.set(key, chained);

  await prev.catch(() => {});
  try {
    return await fn();
  } finally {
    release();
    if (opfsModelLocks.get(key) === chained) {
      opfsModelLocks.delete(key);
    }
  }
}

/** Writes every byte of `data` at `fileOffset`, handling partial `SyncAccessHandle.write()` returns. */
function writeAllSync(handle: SyncAccessHandleLike, data: Uint8Array, fileOffset: number): void {
  let written = 0;
  while (written < data.byteLength) {
    const view = written === 0 ? data : data.subarray(written);
    const n = handle.write(view, { at: fileOffset + written });
    if (n <= 0) {
      throw new Error(`OPFS write stalled at byte offset ${fileOffset + written}.`);
    }
    written += n;
  }
}

/** Validates the 24-byte GGUF v2/v3 header magic, version, and non-zero tensor count. */
export function validateGgufMagicHeader(header: Uint8Array): { valid: boolean; reason?: string } {
  if (header.byteLength < 24) {
    return { valid: false, reason: `File too short (${header.byteLength} bytes) for a GGUF header.` };
  }
  // ASCII "GGUF" = 0x47, 0x47, 0x55, 0x46
  if (header[0] !== 0x47 || header[1] !== 0x47 || header[2] !== 0x55 || header[3] !== 0x46) {
    const preview = String.fromCharCode(header[0], header[1], header[2], header[3]).replace(
      /[^\x20-\x7e]/g,
      '?',
    );
    return { valid: false, reason: `Invalid GGUF magic "${preview}" (expected "GGUF").` };
  }
  const view = new DataView(header.buffer, header.byteOffset, header.byteLength);
  const version = view.getUint32(4, true);
  if (version < 2 || version > 3) {
    return { valid: false, reason: `Unsupported GGUF version ${version} (supported: 2, 3).` };
  }
  const tensorCount = view.getBigUint64(8, true);
  if (tensorCount === 0n) {
    return { valid: false, reason: 'GGUF file declares 0 tensors.' };
  }
  return { valid: true };
}

function validateSyncHandleGgufHeader(handle: SyncAccessHandleLike): { valid: boolean; reason?: string } {
  const size = handle.getSize();
  if (size < 24) {
    return { valid: false, reason: `File too short (${size} bytes) for a GGUF header.` };
  }
  const header = new Uint8Array(24);
  const read = handle.read(header, { at: 0 });
  if (read < 24) {
    return { valid: false, reason: 'Unable to read 24-byte GGUF header from OPFS.' };
  }
  return validateGgufMagicHeader(header);
}

/** Parses HTTP `Content-Range: bytes <start>-<end>/<total|*>` strictly for Range resume safety. */
export function parseContentRangeHeader(
  header: string | null,
): { start: number; end: number; total: number | null } | null {
  if (!header) return null;
  const m = /^\s*bytes\s+(\d+)-(\d+)\/(\d+|\*)\s*$/i.exec(header);
  if (!m) return null;
  const start = Number(m[1]);
  const end = Number(m[2]);
  const total = m[3] === '*' ? null : Number(m[3]);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return null;
  if (total !== null && (!Number.isFinite(total) || total <= end)) return null;
  return { start, end, total };
}

function isRetryableHttpStatus(status: number): boolean {
  return status === 408 || status === 425 || status === 429 || (status >= 500 && status <= 599);
}

function isAbortError(err: unknown): boolean {
  if (!err) return false;
  if (err instanceof DOMException && err.name === 'AbortError') return true;
  if (err instanceof Error && err.name === 'AbortError') return true;
  return false;
}

function isQuotaError(err: unknown): boolean {
  if (!err) return false;
  if (err instanceof DOMException && err.name === 'QuotaExceededError') return true;
  const msg = err instanceof Error ? err.message : String(err);
  return /quota|no space|ENOSPC|storage/i.test(msg);
}

function formatShortSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

async function sleepWithAbort(ms: number, signal?: AbortSignal): Promise<void> {
  if (ms <= 0) {
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
    return;
  }
  await new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('Aborted', 'AbortError'));
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException('Aborted', 'AbortError'));
    };
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

/** Reads a stream chunk with an inactivity stall watchdog so hung CDN sockets auto-resume. */
async function readChunkWithWatchdog(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  timeoutMs: number,
  signal?: AbortSignal,
): Promise<ReadableStreamReadResult<Uint8Array>> {
  if (signal?.aborted) {
    throw new DOMException('Aborted', 'AbortError');
  }
  if (timeoutMs <= 0) {
    return reader.read();
  }
  return new Promise<ReadableStreamReadResult<Uint8Array>>((resolve, reject) => {
    let settled = false;
    const onAbort = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      void reader.cancel().catch(() => {});
      reject(new DOMException('Aborted', 'AbortError'));
    };
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      signal?.removeEventListener('abort', onAbort);
      void reader.cancel().catch(() => {});
      reject(new Error(`Network stream stalled for ${Math.round(timeoutMs / 1000)}s.`));
    }, timeoutMs);

    signal?.addEventListener('abort', onAbort, { once: true });
    reader.read().then(
      (res) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        signal?.removeEventListener('abort', onAbort);
        resolve(res);
      },
      (err) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        signal?.removeEventListener('abort', onAbort);
        reject(err);
      },
    );
  });
}

async function readPartMeta(
  dir: FileSystemDirectoryHandle,
  metaName: string,
): Promise<PartialDownloadMeta | null> {
  try {
    const h = await dir.getFileHandle(metaName, { create: false });
    const f = await h.getFile();
    const text = await f.text();
    const parsed = JSON.parse(text) as Partial<PartialDownloadMeta>;
    if (typeof parsed.url === 'string' && typeof parsed.totalBytes === 'number') {
      return {
        url: parsed.url,
        totalBytes: parsed.totalBytes,
        etag: typeof parsed.etag === 'string' ? parsed.etag : undefined,
        lastModified: typeof parsed.lastModified === 'string' ? parsed.lastModified : undefined,
        receivedBytes: typeof parsed.receivedBytes === 'number' ? parsed.receivedBytes : 0,
        updatedAt: typeof parsed.updatedAt === 'number' ? parsed.updatedAt : Date.now(),
      };
    }
    return null;
  } catch {
    return null;
  }
}

async function writePartMeta(
  dir: FileSystemDirectoryHandle,
  metaName: string,
  meta: PartialDownloadMeta,
): Promise<void> {
  try {
    const h = (await dir.getFileHandle(metaName, { create: true })) as FileSystemFileHandle & {
      createSyncAccessHandle?: () => Promise<SyncAccessHandleLike>;
    };
    const encoded = new TextEncoder().encode(JSON.stringify(meta));
    if (typeof h.createSyncAccessHandle === 'function') {
      const sync = await h.createSyncAccessHandle();
      try {
        sync.truncate(0);
        writeAllSync(sync, encoded, 0);
        sync.flush();
      } finally {
        sync.close();
      }
    }
  } catch {
    // Non-fatal metadata sidecar write failure
  }
}

/** Checks browser storage quota headroom before streaming large multi-GB models. */
async function assertStorageQuotaPreflight(additionalBytesNeeded: number): Promise<void> {
  if (additionalBytesNeeded <= 0) return;
  if (typeof navigator === 'undefined' || typeof navigator.storage?.estimate !== 'function') {
    return;
  }
  try {
    const est = await navigator.storage.estimate();
    if (typeof est.quota === 'number' && est.quota > 0 && typeof est.usage === 'number') {
      const freeBytes = est.quota - est.usage;
      // Leave a small 32 MiB safety buffer for OPFS metadata and KV/browser state
      if (additionalBytesNeeded > freeBytes + 32 * 1024 * 1024 && freeBytes < additionalBytesNeeded) {
        throw new Error(
          `Insufficient browser OPFS storage: need ${formatShortSize(additionalBytesNeeded)}, only ${formatShortSize(Math.max(0, freeBytes))} free of ${formatShortSize(est.quota)}. Clear cached models in the Model Manager to free space.`,
        );
      }
    }
  } catch (err) {
    if (err instanceof Error && err.message.startsWith('Insufficient browser OPFS storage:')) {
      throw err;
    }
    // Ignore unsupported estimate errors
  }
}

/** Opens an already-ingested OPFS GGUF file synchronously for vRAM paging, auto-evicting corrupt files. */
export async function openOpfsModelBacking(modelId: string): Promise<OpfsSyncBackingStore | null> {
  if (!isOpfsSyncSupported()) return null;
  const dir = await getOpfsVramDirectory(false);
  if (!dir) return null;
  const fileName = sanitizeOpfsModelFileName(modelId);
  try {
    const fileHandle = (await dir.getFileHandle(fileName, { create: false })) as FileSystemFileHandle & {
      createSyncAccessHandle: () => Promise<SyncAccessHandleLike>;
    };
    const syncHandle = await fileHandle.createSyncAccessHandle();
    const size = syncHandle.getSize();
    const check = validateSyncHandleGgufHeader(syncHandle);
    if (!check.valid) {
      syncHandle.close();
      // Auto-evict corrupt/truncated OPFS entry so caller seamlessly re-downloads
      await dir.removeEntry(fileName).catch(() => {});
      return null;
    }
    return new OpfsSyncBackingStore(modelId, syncHandle, size);
  } catch {
    return null;
  }
}

export interface ResumableStreamOptions {
  skipOpfs?: boolean;
  signal?: AbortSignal;
  onProgress?: (progress: StreamIngestProgress) => void;
  /** Maximum automatic retry attempts on transient network drops / stalls (default: 6). */
  maxRetries?: number;
  /** Per-chunk inactivity stall watchdog in milliseconds (default: 25,000 ms). */
  stallTimeoutMs?: number;
  /** Base retry backoff delay in milliseconds (default: 700 ms; set small in tests). */
  baseRetryDelayMs?: number;
  /** Injectable fetch implementation for deterministic unit tests. */
  fetchFn?: typeof fetch;
  /** Injectable OPFS directory handle for deterministic unit tests. */
  opfsDirectoryOverride?: FileSystemDirectoryHandle;
}

/**
 * Resumable, retry-hardened HTTP -> OPFS / Memory GGUF downloader.
 *
 * Guarantees rock-solid multi-GB downloads over unstable networks & CDNs:
 *  - Streams in 4 MiB chunks directly to OPFS (`<id>.gguf.part`) without holding full files in RAM.
 *  - Automatically resumes interrupted transfers via HTTP `Range: bytes=<offset>-` with strict
 *    `206 Partial Content` + `Content-Range` + `ETag` verification (up to 6 retries w/ backoff).
 *  - Detects silent CDN stalls via a per-chunk read watchdog AND rejects truncated streams where
 *    `receivedBytes < totalBytes` instead of committing corrupt GGUF files!
 *  - Preserves `.part` progress on user cancellation or transient failure so clicking "Resume"
 *    continues from the exact byte offset where it left off.
 *  - Verifies the GGUF v2/v3 magic header before atomically committing `.part` -> `.gguf`.
 */
export async function streamUrlToVramBacking(
  modelId: string,
  url: string,
  options?: ResumableStreamOptions,
): Promise<VramBackingStore> {
  const fetchImpl = options?.fetchFn ?? fetch;
  const maxRetries = options?.maxRetries ?? DOWNLOAD_MAX_RETRIES;
  const stallTimeoutMs = options?.stallTimeoutMs ?? DOWNLOAD_STALL_TIMEOUT_MS;
  const baseRetryDelayMs = options?.baseRetryDelayMs ?? 700;
  const useOpfs = !options?.skipOpfs && (Boolean(options?.opfsDirectoryOverride) || isOpfsSyncSupported());

  return withModelOpfsLock(modelId, async () => {
    // 1. Fast path: already verified & persisted in OPFS
    if (useOpfs && !options?.opfsDirectoryOverride) {
      const existing = await openOpfsModelBacking(modelId);
      if (existing) {
        options?.onProgress?.({
          receivedBytes: existing.byteLength,
          totalBytes: existing.byteLength,
          percent: 100,
          backingKind: 'opfs-sync',
          stage: 'cached',
        });
        return existing;
      }
    }

    // 2. OPFS resumable streaming path (zero JS heap spikes for 135M – 70B+ GGUFs)
    if (useOpfs) {
      const dir = options?.opfsDirectoryOverride ?? (await getOpfsVramDirectory(true));
      if (dir) {
        const fileName = sanitizeOpfsModelFileName(modelId);
        const tmpName = `${fileName}.part`;
        const metaName = `${fileName}.part.meta`;

        // Check for existing committed file in custom test directory too
        if (options?.opfsDirectoryOverride) {
          try {
            const existingH = (await dir.getFileHandle(fileName, { create: false })) as FileSystemFileHandle & {
              createSyncAccessHandle: () => Promise<SyncAccessHandleLike>;
            };
            const existingSync = await existingH.createSyncAccessHandle();
            const check = validateSyncHandleGgufHeader(existingSync);
            if (check.valid) {
              const size = existingSync.getSize();
              options?.onProgress?.({
                receivedBytes: size,
                totalBytes: size,
                percent: 100,
                backingKind: 'opfs-sync',
                stage: 'cached',
              });
              return new OpfsSyncBackingStore(modelId, existingSync, size);
            }
            existingSync.close();
            await dir.removeEntry(fileName).catch(() => {});
          } catch {
            // Not cached yet
          }
        }

        const existingMeta = await readPartMeta(dir, metaName);
        const tmpHandle = (await dir.getFileHandle(tmpName, { create: true })) as FileSystemFileHandle & {
          createSyncAccessHandle: () => Promise<SyncAccessHandleLike>;
        };
        const syncHandle = await tmpHandle.createSyncAccessHandle();
        const stageBuf = new Uint8Array(OPFS_STREAM_CHUNK_BYTES);
        let stageLen = 0;
        let writtenOffset = 0;
        let knownTotalBytes = 0;
        let knownEtag: string | undefined;
        let knownLastModified: string | undefined;
        let resumedFromBytes = 0;
        let handleClosed = false;

        const closeHandleSafe = () => {
          if (!handleClosed) {
            handleClosed = true;
            try {
              syncHandle.close();
            } catch {
              // Ignore duplicate close
            }
          }
        };

        const flushStage = () => {
          if (stageLen === 0) return;
          const slice = stageLen === stageBuf.byteLength ? stageBuf : stageBuf.subarray(0, stageLen);
          writeAllSync(syncHandle, slice, writtenOffset);
          writtenOffset += stageLen;
          stageLen = 0;
        };

        try {
          // Inspect existing `.part` file to resume seamlessly if it matches this URL
          const existingPartSize = syncHandle.getSize();
          if (existingPartSize > 0 && existingMeta && existingMeta.url === url) {
            if (existingMeta.totalBytes > 0 && existingPartSize <= existingMeta.totalBytes) {
              writtenOffset = existingPartSize;
              knownTotalBytes = existingMeta.totalBytes;
              knownEtag = existingMeta.etag;
              knownLastModified = existingMeta.lastModified;
              resumedFromBytes = existingPartSize;
            } else {
              syncHandle.truncate(0);
              writtenOffset = 0;
            }
          } else if (existingPartSize > 0) {
            // Stale `.part` without matching metadata -> start fresh
            syncHandle.truncate(0);
            writtenOffset = 0;
          }

          let attempt = 0;
          let speedWindowStart = performance.now();
          let speedWindowBytes = writtenOffset;
          let smoothedBps = 0;
          let lastReport = 0;
          let quotaChecked = false;

          while (true) {
            if (options?.signal?.aborted) {
              throw new DOMException('Aborted', 'AbortError');
            }

            // If we already have all expected bytes from a previous resumed run, break straight to verify!
            if (knownTotalBytes > 0 && writtenOffset === knownTotalBytes) {
              break;
            }

            const attemptController = new AbortController();
            const onExternalAbort = () => attemptController.abort();
            options?.signal?.addEventListener('abort', onExternalAbort, { once: true });

            try {
              const headers: Record<string, string> = {};
              const requestingRange = writtenOffset > 0;
              if (requestingRange) {
                headers['Range'] = `bytes=${writtenOffset}-`;
                if (knownEtag) {
                  headers['If-Range'] = knownEtag;
                }
              }

              const res = await fetchImpl(url, {
                headers: requestingRange ? headers : undefined,
                signal: attemptController.signal,
              });

              // Handle HTTP 416 Range Not Satisfiable (e.g., server file shrank or already complete)
              if (res.status === 416 && requestingRange) {
                const cr = res.headers.get('content-range');
                const starMatch = cr ? /bytes\s+\*\/(\d+)/i.exec(cr) : null;
                if (starMatch && Number(starMatch[1]) === writtenOffset && writtenOffset >= 24) {
                  knownTotalBytes = writtenOffset;
                  break;
                }
                // Reset and restart cleanly from byte 0
                syncHandle.truncate(0);
                writtenOffset = 0;
                knownTotalBytes = 0;
                knownEtag = undefined;
                resumedFromBytes = 0;
                continue;
              }

              if (!res.ok) {
                if (isRetryableHttpStatus(res.status) && attempt < maxRetries) {
                  throw new Error(`Transient HTTP ${res.status} while downloading ${modelId}.`);
                }
                throw new Error(`HTTP ${res.status} while downloading ${modelId}.`);
              }

              const respEtag = res.headers.get('etag') ?? undefined;
              const respLastMod = res.headers.get('last-modified') ?? undefined;
              const contentLen = Number(res.headers.get('content-length') ?? 0);

              if (requestingRange && res.status === 206) {
                const parsedRange = parseContentRangeHeader(res.headers.get('content-range'));
                if (!parsedRange || parsedRange.start !== writtenOffset) {
                  // Server returned unexpected range offset -> restart cleanly from 0
                  syncHandle.truncate(0);
                  writtenOffset = 0;
                  knownTotalBytes = 0;
                  resumedFromBytes = 0;
                  continue;
                }
                if (parsedRange.total !== null) {
                  if (knownTotalBytes > 0 && parsedRange.total !== knownTotalBytes) {
                    // Remote file changed size -> restart cleanly from 0
                    syncHandle.truncate(0);
                    writtenOffset = 0;
                    knownTotalBytes = parsedRange.total;
                    resumedFromBytes = 0;
                    continue;
                  }
                  knownTotalBytes = parsedRange.total;
                } else if (contentLen > 0) {
                  knownTotalBytes = writtenOffset + contentLen;
                }
              } else {
                // 200 OK (fresh download or server ignored Range / If-Range mismatch)
                if (writtenOffset > 0) {
                  syncHandle.truncate(0);
                  writtenOffset = 0;
                  resumedFromBytes = 0;
                }
                if (contentLen > 0) {
                  knownTotalBytes = contentLen;
                }
              }

              if (respEtag) knownEtag = respEtag;
              if (respLastMod) knownLastModified = respLastMod;

              // Preflight OPFS quota once we know exact totalBytes
              if (!quotaChecked && knownTotalBytes > 0) {
                quotaChecked = true;
                await assertStorageQuotaPreflight(Math.max(0, knownTotalBytes - writtenOffset));
              }

              // Persist sidecar metadata right as transfer starts/resumes
              await writePartMeta(dir, metaName, {
                url,
                totalBytes: knownTotalBytes,
                etag: knownEtag,
                lastModified: knownLastModified,
                receivedBytes: writtenOffset,
                updatedAt: Date.now(),
              });

              if (!res.body) {
                // Fallback if Response lacks streaming body
                const ab = await res.arrayBuffer();
                const u8 = new Uint8Array(ab);
                if (u8.byteLength > 0) {
                  writeAllSync(syncHandle, u8, writtenOffset);
                  writtenOffset += u8.byteLength;
                }
                if (knownTotalBytes === 0) knownTotalBytes = writtenOffset;
              } else {
                const reader = res.body.getReader();
                speedWindowStart = performance.now();
                speedWindowBytes = writtenOffset;
                let receivedInAttempt = 0;

                try {
                  while (true) {
                    const { done, value } = await readChunkWithWatchdog(
                      reader,
                      stallTimeoutMs,
                      attemptController.signal,
                    );
                    if (done) break;
                    if (!value || value.byteLength === 0) continue;

                    receivedInAttempt += value.byteLength;
                    let offset = 0;
                    while (offset < value.byteLength) {
                      const take = Math.min(stageBuf.byteLength - stageLen, value.byteLength - offset);
                      stageBuf.set(value.subarray(offset, offset + take), stageLen);
                      stageLen += take;
                      offset += take;
                      if (stageLen === stageBuf.byteLength) {
                        flushStage();
                      }
                    }

                    const received = writtenOffset + stageLen;
                    const now = performance.now();
                    if (now - lastReport > 100 || (knownTotalBytes > 0 && received >= knownTotalBytes)) {
                      const dtSec = Math.max(0.05, (now - speedWindowStart) / 1000);
                      const instBps = Math.max(0, (received - speedWindowBytes) / dtSec);
                      smoothedBps = smoothedBps > 0 ? smoothedBps * 0.7 + instBps * 0.3 : instBps;
                      if (dtSec > 2.5) {
                        speedWindowStart = now;
                        speedWindowBytes = received;
                      }
                      const etaSeconds =
                        knownTotalBytes > 0 && smoothedBps > 1024
                          ? Math.max(0, Math.round((knownTotalBytes - received) / smoothedBps))
                          : null;
                      lastReport = now;
                      const pct =
                        knownTotalBytes > 0
                          ? Math.min(95, Math.round((received / knownTotalBytes) * 95))
                          : 50;
                      options?.onProgress?.({
                        receivedBytes: received,
                        totalBytes: knownTotalBytes,
                        percent: pct,
                        backingKind: 'opfs-sync',
                        bytesPerSecond: Math.round(smoothedBps),
                        etaSeconds,
                        resumedFromBytes,
                        retryAttempt: attempt,
                        maxRetries,
                        stage: 'downloading',
                      });
                    }
                  }
                } finally {
                  // Flush every received byte to OPFS before exiting or retrying so ZERO bytes are lost!
                  flushStage();
                  syncHandle.flush();
                  try {
                    reader.releaseLock();
                  } catch {
                    // Ignore
                  }
                }

                // Reset retry counter whenever an attempt made meaningful forward progress (>= 1 MiB)
                if (receivedInAttempt >= 1024 * 1024) {
                  attempt = 0;
                }
              }

              // Strict completeness check: NEVER commit a silently truncated stream!
              if (knownTotalBytes > 0 && writtenOffset < knownTotalBytes) {
                throw new Error(
                  `Stream ended early at ${formatShortSize(writtenOffset)} / ${formatShortSize(knownTotalBytes)}.`,
                );
              }

              // Done!
              break;
            } catch (err) {
              attemptController.abort();
              flushStage();
              try {
                syncHandle.flush();
              } catch {
                // Ignore
              }

              // Persist latest received byte count so cancel or later retry resumes cleanly
              await writePartMeta(dir, metaName, {
                url,
                totalBytes: knownTotalBytes,
                etag: knownEtag,
                lastModified: knownLastModified,
                receivedBytes: writtenOffset,
                updatedAt: Date.now(),
              });

              if (isAbortError(err) || options?.signal?.aborted) {
                throw new DOMException('Aborted', 'AbortError');
              }
              if (isQuotaError(err)) {
                throw new Error(
                  `Browser storage quota exceeded while saving ${modelId}. Clear cached OPFS models in the Model Manager to free space.`,
                );
              }
              // Non-retryable 4xx HTTP errors fail immediately
              const errMsg = err instanceof Error ? err.message : String(err);
              const isHardHttp = /^HTTP (400|401|403|404|410)\b/.test(errMsg);
              if (isHardHttp || attempt >= maxRetries) {
                throw new Error(
                  `Download failed at ${formatShortSize(writtenOffset)}${knownTotalBytes > 0 ? ` / ${formatShortSize(knownTotalBytes)}` : ''} after ${attempt + 1} attempt(s): ${errMsg} (Partial progress saved in OPFS — click Resume to continue.)`,
                );
              }

              attempt += 1;
              const backoffMs = Math.min(10_000, baseRetryDelayMs * Math.pow(2, attempt - 1));
              options?.onProgress?.({
                receivedBytes: writtenOffset,
                totalBytes: knownTotalBytes,
                percent:
                  knownTotalBytes > 0
                    ? Math.min(95, Math.round((writtenOffset / knownTotalBytes) * 95))
                    : 50,
                backingKind: 'opfs-sync',
                resumedFromBytes: writtenOffset,
                retryAttempt: attempt,
                maxRetries,
                stage: 'retrying',
                note: `Connection interrupted — resuming from ${formatShortSize(writtenOffset)} (retry ${attempt}/${maxRetries})`,
              });
              await sleepWithAbort(backoffMs, options?.signal);
            } finally {
              options?.signal?.removeEventListener('abort', onExternalAbort);
            }
          }

          // 3. Verify GGUF v2/v3 magic header BEFORE committing `.part` -> `.gguf`
          options?.onProgress?.({
            receivedBytes: writtenOffset,
            totalBytes: knownTotalBytes || writtenOffset,
            percent: 96,
            backingKind: 'opfs-sync',
            stage: 'verifying',
            note: 'Verifying GGUF header integrity',
          });

          const integrity = validateSyncHandleGgufHeader(syncHandle);
          if (!integrity.valid) {
            closeHandleSafe();
            await dir.removeEntry(tmpName).catch(() => {});
            await dir.removeEntry(metaName).catch(() => {});
            throw new Error(
              `Downloaded file for "${modelId}" failed integrity verification: ${integrity.reason}`,
            );
          }

          syncHandle.flush();
          closeHandleSafe();

          // 4. Atomically commit `.part` -> final `.gguf`
          const movable = tmpHandle as FileSystemFileHandle & { move?: (name: string) => Promise<void> };
          if (typeof movable.move === 'function') {
            await movable.move(fileName);
          } else {
            const finalHandle = (await dir.getFileHandle(fileName, {
              create: true,
            })) as FileSystemFileHandle & {
              createSyncAccessHandle: () => Promise<SyncAccessHandleLike>;
            };
            const srcSync = await tmpHandle.createSyncAccessHandle();
            const dstSync = await finalHandle.createSyncAccessHandle();
            try {
              const totalSize = srcSync.getSize();
              dstSync.truncate(totalSize);
              let pos = 0;
              while (pos < totalSize) {
                const n = Math.min(stageBuf.byteLength, totalSize - pos);
                const view = stageBuf.subarray(0, n);
                const r = srcSync.read(view, { at: pos });
                if (r <= 0) break;
                writeAllSync(dstSync, r === n ? view : view.subarray(0, r), pos);
                pos += r;
              }
              dstSync.flush();
            } finally {
              srcSync.close();
              dstSync.close();
            }
            await dir.removeEntry(tmpName).catch(() => {});
          }
          await dir.removeEntry(metaName).catch(() => {});

          // Re-open final committed OPFS backing handle
          let committed: OpfsSyncBackingStore | null = null;
          if (options?.opfsDirectoryOverride) {
            const finalH = (await dir.getFileHandle(fileName, {
              create: false,
            })) as FileSystemFileHandle & {
              createSyncAccessHandle: () => Promise<SyncAccessHandleLike>;
            };
            const finalSync = await finalH.createSyncAccessHandle();
            committed = new OpfsSyncBackingStore(modelId, finalSync, finalSync.getSize());
          } else {
            committed = await openOpfsModelBacking(modelId);
          }
          if (!committed) {
            throw new Error(`Failed to open committed OPFS backing store for "${modelId}".`);
          }
          options?.onProgress?.({
            receivedBytes: committed.byteLength,
            totalBytes: committed.byteLength,
            percent: 100,
            backingKind: 'opfs-sync',
            stage: 'cached',
          });
          return committed;
        } catch (err) {
          closeHandleSafe();
          throw err;
        }
      }
    }

    // 3. Fallback path: in-memory streaming with automatic Range retries (small bundled models or OPFS-disabled)
    let attempt = 0;
    const chunks: Uint8Array[] = [];
    let received = 0;
    let totalBytes = 0;
    let lastReport = 0;

    while (true) {
      if (options?.signal?.aborted) {
        throw new DOMException('Aborted', 'AbortError');
      }
      const attemptController = new AbortController();
      const onExternalAbort = () => attemptController.abort();
      options?.signal?.addEventListener('abort', onExternalAbort, { once: true });

      try {
        const requestingRange = received > 0;
        const res = await fetchImpl(url, {
          headers: requestingRange ? { Range: `bytes=${received}-` } : undefined,
          signal: attemptController.signal,
        });
        if (!res.ok) {
          if (isRetryableHttpStatus(res.status) && attempt < maxRetries) {
            throw new Error(`Transient HTTP ${res.status} while downloading ${modelId}.`);
          }
          throw new Error(`HTTP ${res.status} while downloading ${modelId}.`);
        }

        const contentLen = Number(res.headers.get('content-length') ?? 0);
        if (requestingRange && res.status === 206) {
          const parsedRange = parseContentRangeHeader(res.headers.get('content-range'));
          if (!parsedRange || parsedRange.start !== received) {
            chunks.length = 0;
            received = 0;
            totalBytes = 0;
            continue;
          }
          if (parsedRange.total !== null) totalBytes = parsedRange.total;
        } else {
          if (received > 0) {
            chunks.length = 0;
            received = 0;
          }
          if (contentLen > 0) totalBytes = contentLen;
        }

        if (!res.body) {
          const ab = await res.arrayBuffer();
          const u8 = new Uint8Array(ab);
          chunks.push(u8);
          received += u8.byteLength;
          if (totalBytes === 0) totalBytes = received;
        } else {
          const reader = res.body.getReader();
          try {
            while (true) {
              const { done, value } = await readChunkWithWatchdog(
                reader,
                stallTimeoutMs,
                attemptController.signal,
              );
              if (done) break;
              if (!value || value.byteLength === 0) continue;
              chunks.push(value);
              received += value.byteLength;
              const now = performance.now();
              if (now - lastReport > 100 || (totalBytes > 0 && received >= totalBytes)) {
                lastReport = now;
                options?.onProgress?.({
                  receivedBytes: received,
                  totalBytes,
                  percent: totalBytes > 0 ? Math.min(95, Math.round((received / totalBytes) * 95)) : 50,
                  backingKind: 'memory',
                  stage: 'downloading',
                });
              }
            }
          } finally {
            try {
              reader.releaseLock();
            } catch {
              // Ignore
            }
          }
        }

        if (totalBytes > 0 && received < totalBytes) {
          throw new Error(`Stream ended early at ${received} / ${totalBytes} bytes.`);
        }
        break;
      } catch (err) {
        attemptController.abort();
        if (isAbortError(err) || options?.signal?.aborted) {
          throw new DOMException('Aborted', 'AbortError');
        }
        const errMsg = err instanceof Error ? err.message : String(err);
        const isHardHttp = /^HTTP (400|401|403|404|410)\b/.test(errMsg);
        if (isHardHttp || attempt >= maxRetries) {
          throw err;
        }
        attempt += 1;
        await sleepWithAbort(Math.min(8000, baseRetryDelayMs * Math.pow(2, attempt - 1)), options?.signal);
      } finally {
        options?.signal?.removeEventListener('abort', onExternalAbort);
      }
    }

    const combined = new Uint8Array(received);
    let pos = 0;
    for (const c of chunks) {
      combined.set(c, pos);
      pos += c.byteLength;
    }
    const integrity = validateGgufMagicHeader(combined);
    if (!integrity.valid) {
      throw new Error(`Downloaded file for "${modelId}" is not a valid GGUF: ${integrity.reason}`);
    }
    options?.onProgress?.({
      receivedBytes: combined.byteLength,
      totalBytes: combined.byteLength,
      percent: 100,
      backingKind: 'memory',
      stage: 'cached',
    });
    return new MemoryBackingStore(modelId, combined);
  });
}

/**
 * Streams a local user-dropped or selected `File` directly into OPFS in 4 MiB slices
 * (`file.slice()`), validating the GGUF header before atomic commit and supporting cancellation!
 */
export async function streamFileToVramBacking(
  modelId: string,
  file: File,
  onProgress?: (progress: StreamIngestProgress) => void,
  signal?: AbortSignal,
): Promise<VramBackingStore> {
  const totalBytes = file.size;

  // Immediate GGUF magic header check on first 24 bytes before copying gigabytes!
  if (totalBytes < 24) {
    throw new Error(`"${file.name}" is too small (${totalBytes} bytes) to be a valid GGUF file.`);
  }
  const headBuf = new Uint8Array(await file.slice(0, 24).arrayBuffer());
  const headCheck = validateGgufMagicHeader(headBuf);
  if (!headCheck.valid) {
    throw new Error(`"${file.name}" is not a valid GGUF model: ${headCheck.reason}`);
  }

  return withModelOpfsLock(modelId, async () => {
    if (isOpfsSyncSupported()) {
      await assertStorageQuotaPreflight(totalBytes);
      const dir = await getOpfsVramDirectory(true);
      if (dir) {
        const fileName = sanitizeOpfsModelFileName(modelId);
        const tmpName = `${fileName}.part`;
        const tmpHandle = (await dir.getFileHandle(tmpName, { create: true })) as FileSystemFileHandle & {
          createSyncAccessHandle: () => Promise<SyncAccessHandleLike>;
        };
        const syncHandle = await tmpHandle.createSyncAccessHandle();
        let handleClosed = false;
        const closeSafe = () => {
          if (!handleClosed) {
            handleClosed = true;
            try {
              syncHandle.close();
            } catch {
              // Ignore
            }
          }
        };

        try {
          syncHandle.truncate(0);
          let offset = 0;
          let lastReport = 0;
          const startTime = performance.now();

          while (offset < totalBytes) {
            if (signal?.aborted) {
              throw new DOMException('Aborted', 'AbortError');
            }
            const end = Math.min(totalBytes, offset + OPFS_STREAM_CHUNK_BYTES);
            const chunkBuf = await file.slice(offset, end).arrayBuffer();
            const chunkU8 = new Uint8Array(chunkBuf);
            writeAllSync(syncHandle, chunkU8, offset);
            offset = end;

            const now = performance.now();
            if (now - lastReport > 100 || offset >= totalBytes) {
              lastReport = now;
              const elapsedSec = Math.max(0.05, (now - startTime) / 1000);
              const bps = offset / elapsedSec;
              const etaSeconds = bps > 1024 ? Math.max(0, Math.round((totalBytes - offset) / bps)) : null;
              onProgress?.({
                receivedBytes: offset,
                totalBytes,
                percent: Math.min(95, Math.round((offset / Math.max(1, totalBytes)) * 95)),
                backingKind: 'opfs-sync',
                bytesPerSecond: Math.round(bps),
                etaSeconds,
                stage: 'downloading',
              });
            }
          }
          syncHandle.flush();
          closeSafe();

          const movable = tmpHandle as FileSystemFileHandle & { move?: (name: string) => Promise<void> };
          if (typeof movable.move === 'function') {
            await movable.move(fileName);
          } else {
            const finalHandle = (await dir.getFileHandle(fileName, {
              create: true,
            })) as FileSystemFileHandle & {
              createSyncAccessHandle: () => Promise<SyncAccessHandleLike>;
            };
            const srcSync = await tmpHandle.createSyncAccessHandle();
            const dstSync = await finalHandle.createSyncAccessHandle();
            const stageBuf = new Uint8Array(OPFS_STREAM_CHUNK_BYTES);
            try {
              dstSync.truncate(totalBytes);
              let pos = 0;
              while (pos < totalBytes) {
                const n = Math.min(stageBuf.byteLength, totalBytes - pos);
                const view = stageBuf.subarray(0, n);
                const r = srcSync.read(view, { at: pos });
                if (r <= 0) break;
                writeAllSync(dstSync, r === n ? view : view.subarray(0, r), pos);
                pos += r;
              }
              dstSync.flush();
            } finally {
              srcSync.close();
              dstSync.close();
            }
            await dir.removeEntry(tmpName).catch(() => {});
          }

          const committed = await openOpfsModelBacking(modelId);
          if (!committed) {
            throw new Error(`Unable to open OPFS backing file for "${file.name}".`);
          }
          onProgress?.({
            receivedBytes: totalBytes,
            totalBytes,
            percent: 100,
            backingKind: 'opfs-sync',
            stage: 'cached',
          });
          return committed;
        } catch (err) {
          closeSafe();
          await dir.removeEntry(tmpName).catch(() => {});
          throw err;
        }
      }
    }

    // Fallback when OPFS is unavailable
    const ab = await file.arrayBuffer();
    onProgress?.({
      receivedBytes: ab.byteLength,
      totalBytes: ab.byteLength,
      percent: 100,
      backingKind: 'memory',
      stage: 'cached',
    });
    return new MemoryBackingStore(modelId, ab);
  });
}

/** Lists all committed & partial GGUF models in the OPFS vRAM directory plus browser storage quota. */
export async function inspectOpfsVramStorage(): Promise<OpfsQuotaSnapshot> {
  const supported =
    typeof navigator !== 'undefined' && typeof navigator.storage?.getDirectory === 'function';
  let usageBytes = 0;
  let quotaBytes = 0;
  try {
    if (typeof navigator !== 'undefined' && typeof navigator.storage?.estimate === 'function') {
      const est = await navigator.storage.estimate();
      usageBytes = est.usage ?? 0;
      quotaBytes = est.quota ?? 0;
    }
  } catch {
    // Ignore estimate error
  }

  const storedModels: OpfsStoredModelEntry[] = [];
  const partialModels: OpfsPartialModelEntry[] = [];
  if (supported) {
    const dir = await getOpfsVramDirectory(false);
    if (dir) {
      const iterable = dir as FileSystemDirectoryHandle & {
        entries?: () => AsyncIterableIterator<[string, FileSystemHandle]>;
      };
      if (typeof iterable.entries === 'function') {
        for await (const [name, handle] of iterable.entries()) {
          if (handle.kind !== 'file') continue;
          if (name.endsWith('.gguf')) {
            try {
              const f = await (handle as FileSystemFileHandle).getFile();
              storedModels.push({
                modelId: name.replace(/\.gguf$/i, ''),
                fileName: name,
                byteLength: f.size,
                lastModified: f.lastModified,
              });
            } catch {
              // Ignore locked or transient file
            }
          } else if (name.endsWith('.gguf.part')) {
            try {
              const f = await (handle as FileSystemFileHandle).getFile();
              if (f.size > 0) {
                const baseName = name.slice(0, -'.part'.length);
                const meta = await readPartMeta(dir, `${baseName}.part.meta`);
                partialModels.push({
                  modelId: baseName.replace(/\.gguf$/i, ''),
                  fileName: name,
                  receivedBytes: f.size,
                  totalBytes: meta?.totalBytes ?? 0,
                  url: meta?.url,
                  lastModified: f.lastModified,
                });
              }
            } catch {
              // Ignore locked or transient file
            }
          }
        }
      }
    }
  }

  return {
    supported,
    usageBytes,
    quotaBytes,
    storedModels,
    partialModels,
  };
}

/** Deletes a single persisted or partial OPFS GGUF file (plus sidecar metadata) by model ID. */
export async function deleteOpfsModel(modelId: string): Promise<boolean> {
  return withModelOpfsLock(modelId, async () => {
    const dir = await getOpfsVramDirectory(false);
    if (!dir) return false;
    const base = sanitizeOpfsModelFileName(modelId);
    let removed = false;
    for (const candidate of [base, `${base}.part`, `${base}.part.meta`]) {
      try {
        await dir.removeEntry(candidate);
        removed = true;
      } catch {
        // Ignore missing sidecar/file
      }
    }
    return removed;
  });
}

/** Deletes all persisted and partial OPFS GGUF files in `vram-gguf-v1`. */
export async function clearAllOpfsModels(): Promise<void> {
  if (typeof navigator === 'undefined' || !navigator.storage?.getDirectory) return;
  try {
    const root = await navigator.storage.getDirectory();
    await root.removeEntry(OPFS_VRAM_DIR_NAME, { recursive: true });
  } catch {
    // Ignore if directory does not exist
  }
}
