import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { GgmlType, parseGgufFromBackingAsync, type GgufTensorInfo } from './gguf';
import { LlamaVTensorRuntime } from './llamaRuntime';
import {
  LocalFileSyncBackingStore,
  MemoryBackingStore,
  OPFS_DEMAND_SLAB_BYTES,
  OpfsDemandVramBackingStore,
  OpfsLockError,
  OpfsSyncBackingStore,
  coalesceTensorExtents,
  inspectOpfsVramStorage,
  isOpfsLockError,
  openOpfsDemandModelBacking,
  openOpfsModelBacking,
  openSyncAccessHandleWithRetry,
  parseContentRangeHeader,
  sanitizeOpfsModelFileName,
  streamUrlToVramBacking,
  validateGgufMagicHeader,
  vramCapacityCeiling,
  type FileReaderSyncLike,
  type StreamIngestProgress,
  type SyncAccessHandleLike,
} from './vramBackingStore';

function makeTensor(name: string, offset: number, byteSize: number): GgufTensorInfo {
  return {
    name,
    dims: [32, Math.max(1, Math.floor(byteSize / 18))],
    type: GgmlType.Q4_0,
    typeName: 'Q4_0',
    offset,
    byteSize,
  };
}

/** Creates a valid GGUF v3 binary payload of `totalBytes` with deterministic body bytes. */
function makeFakeGgufPayload(totalBytes: number): Uint8Array {
  const buf = new Uint8Array(Math.max(64, totalBytes));
  // Magic "GGUF"
  buf[0] = 0x47;
  buf[1] = 0x47;
  buf[2] = 0x55;
  buf[3] = 0x46;
  const view = new DataView(buf.buffer);
  view.setUint32(4, 3, true); // version 3
  view.setBigUint64(8, 4n, true); // 4 tensors
  view.setBigUint64(16, 0n, true); // 0 kv pairs
  for (let i = 24; i < buf.byteLength; i++) {
    buf[i] = (i * 73 + 19) & 0xff;
  }
  return buf;
}

/** Deterministic in-memory FileSystemDirectoryHandle mock supporting createSyncAccessHandle, entries(), and optional move(). */
function createMockOpfsDir(options?: { supportMove?: boolean }): {
  dir: FileSystemDirectoryHandle;
  files: Map<string, Uint8Array>;
  mtimes: Map<string, number>;
} {
  const supportMove = options?.supportMove ?? true;
  const files = new Map<string, Uint8Array>();
  const mtimes = new Map<string, number>();

  const makeHandle = (initialName: string): FileSystemFileHandle => {
    let currentName = initialName;
    const handle: Record<string, unknown> = {
      kind: 'file' as const,
      get name() {
        return currentName;
      },
      async getFile() {
        const data = files.get(currentName) ?? new Uint8Array(0);
        return {
          size: data.byteLength,
          lastModified: mtimes.get(currentName) ?? Date.now(),
          async text() {
            return new TextDecoder().decode(data);
          },
          async arrayBuffer() {
            return data.slice().buffer;
          },
        } as unknown as File;
      },
      async createSyncAccessHandle(): Promise<SyncAccessHandleLike> {
        let closed = false;
        return {
          getSize: () => (files.get(currentName) ?? new Uint8Array(0)).byteLength,
          read: (view, opts) => {
            if (closed) throw new Error('closed');
            const data = files.get(currentName) ?? new Uint8Array(0);
            const at = opts?.at ?? 0;
            const u8 = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
            const count = Math.max(0, Math.min(u8.byteLength, data.byteLength - at));
            u8.set(data.subarray(at, at + count), 0);
            return count;
          },
          write: (view, opts) => {
            if (closed) throw new Error('closed');
            const at = opts?.at ?? 0;
            const src = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
            const existing = files.get(currentName) ?? new Uint8Array(0);
            const nextLen = Math.max(existing.byteLength, at + src.byteLength);
            const next = nextLen === existing.byteLength ? existing : new Uint8Array(nextLen);
            if (next !== existing) next.set(existing, 0);
            next.set(src, at);
            files.set(currentName, next);
            mtimes.set(currentName, Date.now());
            return src.byteLength;
          },
          truncate: (newSize: number) => {
            if (closed) throw new Error('closed');
            const existing = files.get(currentName) ?? new Uint8Array(0);
            const next = new Uint8Array(newSize);
            next.set(existing.subarray(0, Math.min(existing.byteLength, newSize)), 0);
            files.set(currentName, next);
          },
          flush: () => {},
          close: () => {
            closed = true;
          },
        };
      },
    };
    if (supportMove) {
      handle.move = async (newName: string) => {
        const data = files.get(currentName) ?? new Uint8Array(0);
        const mt = mtimes.get(currentName) ?? Date.now();
        files.set(newName, data);
        mtimes.set(newName, mt);
        if (newName !== currentName) {
          files.delete(currentName);
          mtimes.delete(currentName);
          currentName = newName;
        }
      };
    }
    return handle as unknown as FileSystemFileHandle;
  };

  const dir = {
    kind: 'directory' as const,
    name: 'vram-gguf-v1',
    async getFileHandle(name: string, opts?: { create?: boolean }) {
      if (!files.has(name)) {
        if (opts?.create) {
          files.set(name, new Uint8Array(0));
          mtimes.set(name, Date.now());
        } else {
          throw new DOMException(`NotFoundError: ${name}`, 'NotFoundError');
        }
      }
      return makeHandle(name);
    },
    async removeEntry(name: string) {
      if (!files.has(name)) {
        throw new DOMException(`NotFoundError: ${name}`, 'NotFoundError');
      }
      files.delete(name);
      mtimes.delete(name);
    },
    async *entries(): AsyncIterableIterator<[string, FileSystemHandle]> {
      for (const name of Array.from(files.keys())) {
        yield [name, makeHandle(name)];
      }
    },
  } as unknown as FileSystemDirectoryHandle;

  return { dir, files, mtimes };
}

test('vramCapacityCeiling enforces afflom/vRAM CapacityPolicy min(p, g, b) monotonicity invariants', () => {
  assert.equal(vramCapacityCeiling(512, 1024, 4096), 512);
  assert.equal(vramCapacityCeiling(2048, 768, 4096), 768);
  assert.equal(vramCapacityCeiling(2048, 1024, 256), 256);
  // Monotone backing property: b <= c => ceiling(p, g, b) <= ceiling(p, g, c)
  for (const [p, g, b, c] of [
    [1000, 800, 400, 600],
    [1000, 800, 600, 1200],
    [500, 900, 100, 500],
  ]) {
    assert.ok(vramCapacityCeiling(p, g, b) <= vramCapacityCeiling(p, g, c));
  }
});

test('coalesceTensorExtents merges contiguous and small-gap layer tensors into minimal DIMM restores (b0d40626)', () => {
  const dataOffset = 4096;
  const layerTensors: GgufTensorInfo[] = [
    // Deliberately unsorted to verify deterministic offset sorting
    makeTensor('blk.0.attn_v.weight', 2080, 1024),
    makeTensor('blk.0.attn_q.weight', 0, 1024),
    makeTensor('blk.0.attn_k.weight', 1056, 1024), // 32-byte alignment gap -> coalesced!
    // Disjoint FFN tensors 64 KiB away -> second coalesced extent
    makeTensor('blk.0.ffn_up.weight', 70000 + 2080, 2048),
    makeTensor('blk.0.ffn_gate.weight', 70000, 2048),
  ];

  const extents = coalesceTensorExtents(layerTensors, dataOffset, 4096);
  assert.equal(extents.length, 2, `Expected 2 coalesced extents from 5 tensors, got ${extents.length}`);

  assert.equal(extents[0].fileOffset, dataOffset + 0);
  assert.equal(extents[0].byteLength, 2080 + 1024);
  assert.equal(extents[0].slices.length, 3);
  assert.equal(extents[0].slices[0].tensor.name, 'blk.0.attn_q.weight');
  assert.equal(extents[0].slices[0].relativeOffset, 0);
  assert.equal(extents[0].slices[1].relativeOffset, 1056);
  assert.equal(extents[0].slices[2].relativeOffset, 2080);

  assert.equal(extents[1].fileOffset, dataOffset + 70000);
  assert.equal(extents[1].byteLength, 2080 + 2048);
  assert.equal(extents[1].slices.length, 2);
});

test('MemoryBackingStore and OpfsSyncBackingStore read exact byte slices including partial OS reads', () => {
  const payload = new Uint8Array(8192);
  for (let i = 0; i < payload.length; i++) payload[i] = (i * 131 + 17) & 0xff;

  const memStore = new MemoryBackingStore('test-mem', payload);
  const dst1 = new Uint8Array(1024);
  const n1 = memStore.readRange(512, dst1, 0, 1024);
  assert.equal(n1, 1024);
  assert.deepEqual(dst1, payload.subarray(512, 1536));

  // Mock SyncAccessHandleLike that caps each read() at 257 bytes to prove readRange loops to completion
  let closed = false;
  const mockHandle: SyncAccessHandleLike = {
    getSize: () => payload.byteLength,
    read: (view, opts) => {
      const at = opts?.at ?? 0;
      const u8 = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
      const count = Math.min(257, u8.byteLength, payload.byteLength - at);
      u8.set(payload.subarray(at, at + count), 0);
      return count;
    },
    write: () => 0,
    truncate: () => {},
    flush: () => {},
    close: () => {
      closed = true;
    },
  };

  const opfsStore = new OpfsSyncBackingStore('llama-3.2-3b', mockHandle);
  assert.equal(opfsStore.kind, 'opfs-sync');
  assert.equal(opfsStore.isPersistent, true);
  const dst2 = new Uint8Array(1500);
  const n2 = opfsStore.readRange(4096, dst2, 100, 1024);
  assert.equal(n2, 1024);
  assert.deepEqual(dst2.subarray(100, 1124), payload.subarray(4096, 5120));

  opfsStore.close();
  assert.equal(closed, true);
  assert.throws(() => opfsStore.readRange(0, dst2), /already closed/);
  assert.equal(sanitizeOpfsModelFileName('Llama 3.2/3B:Q4_K_M'), 'Llama_3.2_3B_Q4_K_M.gguf');
});

test('validateGgufMagicHeader and parseContentRangeHeader strictly enforce wire invariants', () => {
  const valid = makeFakeGgufPayload(64);
  assert.equal(validateGgufMagicHeader(valid).valid, true);

  const badMagic = new Uint8Array(64).fill(0x41);
  assert.equal(validateGgufMagicHeader(badMagic).valid, false);

  const zeroTensors = makeFakeGgufPayload(64);
  new DataView(zeroTensors.buffer).setBigUint64(8, 0n, true);
  assert.equal(validateGgufMagicHeader(zeroTensors).valid, false);

  assert.deepEqual(parseContentRangeHeader('bytes 100-199/500'), { start: 100, end: 199, total: 500 });
  assert.deepEqual(parseContentRangeHeader('bytes 0-99/*'), { start: 0, end: 99, total: null });
  assert.equal(parseContentRangeHeader('bytes 200-100/500'), null);
  assert.equal(parseContentRangeHeader('invalid'), null);
});

test('streamUrlToVramBacking survives mid-stream network drops and silent early EOF via HTTP Range 206 resume', async () => {
  const payload = makeFakeGgufPayload(128 * 1024); // 128 KiB
  const { dir, files } = createMockOpfsDir();
  const seenRangeHeaders: (string | null)[] = [];
  let callIndex = 0;

  const flakyFetch: typeof fetch = async (_input, init) => {
    callIndex += 1;
    const reqHeaders = new Headers(init?.headers);
    const range = reqHeaders.get('Range');
    seenRangeHeaders.push(range);

    // Call 1: return 200 OK, yield 32 KiB on first pull, then throw mid-stream network drop on second pull
    if (callIndex === 1) {
      let sent = false;
      const stream = new ReadableStream<Uint8Array>({
        pull(controller) {
          if (!sent) {
            sent = true;
            controller.enqueue(payload.slice(0, 32 * 1024));
          } else {
            controller.error(new TypeError('network connection lost'));
          }
        },
      });
      return new Response(stream, {
        status: 200,
        headers: {
          'content-length': String(payload.byteLength),
          etag: '"etag-v1"',
        },
      });
    }

    // Call 2: transient 503 Service Unavailable
    if (callIndex === 2) {
      return new Response('busy', { status: 503 });
    }

    // Call 3: 206 Partial Content starting at 32 KiB, but silently truncates after another 40 KiB!
    if (callIndex === 3) {
      assert.equal(range, `bytes=${32 * 1024}-`);
      const start = 32 * 1024;
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(payload.slice(start, start + 40 * 1024));
          controller.close(); // Silent early EOF before full payload!
        },
      });
      return new Response(stream, {
        status: 206,
        headers: {
          'content-range': `bytes ${start}-${payload.byteLength - 1}/${payload.byteLength}`,
          'content-length': String(payload.byteLength - start),
          etag: '"etag-v1"',
        },
      });
    }

    // Call 4: 206 Partial Content finishing the remaining 56 KiB cleanly
    const start = 72 * 1024;
    assert.equal(range, `bytes=${start}-`);
    const rest = payload.slice(start);
    return new Response(rest, {
      status: 206,
      headers: {
        'content-range': `bytes ${start}-${payload.byteLength - 1}/${payload.byteLength}`,
        'content-length': String(rest.byteLength),
        etag: '"etag-v1"',
      },
    });
  };

  const events: StreamIngestProgress[] = [];
  const store = await streamUrlToVramBacking('flaky-model', 'https://example.test/flaky.gguf', {
    opfsDirectoryOverride: dir,
    fetchFn: flakyFetch,
    baseRetryDelayMs: 1,
    maxRetries: 5,
    onProgress: (p) => events.push(p),
  });

  try {
    assert.equal(callIndex, 4);
    assert.equal(store.byteLength, payload.byteLength);
    const readBack = new Uint8Array(payload.byteLength);
    assert.equal(store.readRange(0, readBack), payload.byteLength);
    assert.deepEqual(readBack, payload);
    assert.equal(files.has('flaky-model.gguf'), true);
    assert.equal(files.has('flaky-model.gguf.part'), false);
    assert.equal(files.has('flaky-model.gguf.part.meta'), false);
    assert.ok(events.some((e) => e.stage === 'retrying'));
  } finally {
    store.close();
  }
});

test('streamUrlToVramBacking preserves .part on user AbortSignal and resumes seamlessly on next call', async () => {
  const payload = makeFakeGgufPayload(96 * 1024);
  const { dir, files } = createMockOpfsDir();
  const controller1 = new AbortController();

  // First run: streams 48 KiB on pull #1, then user cancels while pull #2 is awaiting
  const fetch1: typeof fetch = async () => {
    let sent = false;
    const stream = new ReadableStream<Uint8Array>(
      {
        pull(ctrl) {
          if (!sent) {
            sent = true;
            ctrl.enqueue(payload.slice(0, 48 * 1024));
          } else {
            return new Promise<void>(() => {
              // Hang open until external AbortSignal fires
            });
          }
        },
      },
      { highWaterMark: 0 },
    );
    return new Response(stream, {
      status: 200,
      headers: {
        'content-length': String(payload.byteLength),
        etag: '"resume-etag"',
      },
    });
  };

  await assert.rejects(
    () =>
      streamUrlToVramBacking('resumable-model', 'https://example.test/resumable.gguf', {
        opfsDirectoryOverride: dir,
        fetchFn: fetch1,
        signal: controller1.signal,
        baseRetryDelayMs: 1,
        onProgress: (p) => {
          if (p.receivedBytes >= 48 * 1024) {
            controller1.abort();
          }
        },
      }),
    (err: unknown) => (err instanceof DOMException || err instanceof Error) && err.name === 'AbortError',
  );

  // `.part` and `.part.meta` must remain on disk for instant resume!
  assert.equal(files.has('resumable-model.gguf'), false);
  assert.equal(files.get('resumable-model.gguf.part')?.byteLength, 48 * 1024);
  assert.equal(files.has('resumable-model.gguf.part.meta'), true);

  // Second run: resumes directly from byte 48 KiB!
  let sawResumeRange = '';
  const fetch2: typeof fetch = async (_input, init) => {
    const range = new Headers(init?.headers).get('Range') ?? '';
    sawResumeRange = range;
    const start = 48 * 1024;
    const tail = payload.slice(start);
    return new Response(tail, {
      status: 206,
      headers: {
        'content-range': `bytes ${start}-${payload.byteLength - 1}/${payload.byteLength}`,
        'content-length': String(tail.byteLength),
        etag: '"resume-etag"',
      },
    });
  };

  const store = await streamUrlToVramBacking('resumable-model', 'https://example.test/resumable.gguf', {
    opfsDirectoryOverride: dir,
    fetchFn: fetch2,
    baseRetryDelayMs: 1,
  });
  try {
    assert.equal(sawResumeRange, `bytes=${48 * 1024}-`);
    const readBack = new Uint8Array(payload.byteLength);
    store.readRange(0, readBack);
    assert.deepEqual(readBack, payload);
  } finally {
    store.close();
  }
});

test('streamUrlToVramBacking rejects corrupt non-GGUF HTML/error payloads before committing to OPFS', async () => {
  const { dir, files } = createMockOpfsDir();
  const htmlPayload = new TextEncoder().encode('<!doctype html><html>Not a GGUF file!</html>');
  const badFetch: typeof fetch = async () =>
    new Response(htmlPayload, {
      status: 200,
      headers: { 'content-length': String(htmlPayload.byteLength) },
    });

  await assert.rejects(
    () =>
      streamUrlToVramBacking('bad-html', 'https://example.test/error.gguf', {
        opfsDirectoryOverride: dir,
        fetchFn: badFetch,
        baseRetryDelayMs: 1,
      }),
    /invalid GGUF magic|integrity verification/i,
  );
  assert.equal(files.has('bad-html.gguf'), false);
  assert.equal(files.has('bad-html.gguf.part'), false);
});

test('LocalFileSyncBackingStore reads arbitrary ranges directly from disk Blob with zero OPFS writes', () => {
  const payload = makeFakeGgufPayload(64 * 1024);
  const mockReader: FileReaderSyncLike = {
    readAsArrayBuffer(slice: Blob): ArrayBuffer {
      return (slice as unknown as { _buf: ArrayBuffer })._buf;
    },
  };
  // Custom Blob-like wrapper returning deterministic ArrayBuffers synchronously
  const syncBlob = {
    size: payload.byteLength,
    slice(start = 0, end = payload.byteLength) {
      return {
        _buf: payload.slice(start, end).buffer,
        size: Math.max(0, end - start),
      } as unknown as Blob;
    },
  } as unknown as Blob;

  const store = new LocalFileSyncBackingStore('local-30b', syncBlob, mockReader);
  assert.equal(store.kind, 'local-file-sync');
  assert.equal(store.isPersistent, false);
  assert.equal(store.byteLength, payload.byteLength);

  const dst = new Uint8Array(12_345);
  const n = store.readRange(20_000, dst, 100, 10_000);
  assert.equal(n, 10_000);
  assert.deepEqual(dst.subarray(100, 10_100), payload.subarray(20_000, 30_000));

  store.close();
  assert.throws(() => store.readRange(0, dst), /already closed/);
});

test('OpfsDemandVramBackingStore enforces O(1) bounded LRU slab cache, cross-slab reads, and persistent index reload', async () => {
  const slabBytes = 16 * 1024;
  const payload = makeFakeGgufPayload(slabBytes * 6); // 96 KiB = 6 slabs
  const { dir, files } = createMockOpfsDir();

  let syncFetches = 0;
  const mockRangeFetcher = (_url: string, start: number, endInclusive: number): Uint8Array => {
    syncFetches += 1;
    return payload.slice(start, endInclusive + 1);
  };

  const rangeFetch: typeof fetch = async (_input, init) => {
    const range = new Headers(init?.headers).get('Range');
    const m = range ? /bytes=(\d+)-(\d+)/.exec(range) : null;
    if (m) {
      const s = Number(m[1]);
      const e = Number(m[2]);
      const chunk = payload.slice(s, e + 1);
      return new Response(chunk, {
        status: 206,
        headers: {
          'content-range': `bytes ${s}-${e}/${payload.byteLength}`,
          'content-length': String(chunk.byteLength),
          'accept-ranges': 'bytes',
        },
      });
    }
    return new Response(payload.slice(), {
      status: 200,
      headers: {
        'content-length': String(payload.byteLength),
        'accept-ranges': 'bytes',
      },
    });
  };

  // Origin quota (56 KiB) < full model (96 KiB) -> seamlessly activates OpfsDemandVramBackingStore!
  const store = await streamUrlToVramBacking('qwen3-30b-demand', 'https://example.test/30b.gguf', {
    opfsDirectoryOverride: dir,
    fetchFn: rangeFetch,
    syncRangeFetcherOverride: mockRangeFetcher,
    demandSlabBytesOverride: slabBytes,
    storageEstimateOverride: async () => ({ usage: 0, quota: 56 * 1024 }),
    baseRetryDelayMs: 1,
  });

  assert.equal(store.kind, 'opfs-demand');
  assert.equal(store.byteLength, payload.byteLength);
  assert.ok(store instanceof OpfsDemandVramBackingStore);
  assert.ok(store.getStats().maxSlots <= 3, `Expected bounded <=3 slab slots, got ${store.getStats().maxSlots}`);

  // Cross-slab read spanning slabs 0, 1, 2 (slab 0 is already prewarmed -> 1 hit + 2 misses)
  const cross = new Uint8Array(slabBytes * 2);
  const readN = store.readRange(slabBytes - 4096, cross);
  assert.equal(readN, cross.byteLength);
  assert.deepEqual(cross, payload.subarray(slabBytes - 4096, slabBytes - 4096 + cross.byteLength));
  assert.equal(store.getStats().slabHits >= 1, true);
  assert.equal(syncFetches, 2);

  // Touching slabs 3, 4, 5 forces O(1) LRU evictions while keeping cache file strictly bounded!
  const tail = new Uint8Array(slabBytes * 3);
  assert.equal(store.readRange(slabBytes * 3, tail), tail.byteLength);
  assert.deepEqual(tail, payload.subarray(slabBytes * 3, slabBytes * 6));
  assert.ok(store.getStats().slabEvictions >= 2, 'Expected LRU slab evictions under tight OPFS budget');
  const cacheDiskSize = files.get('qwen3-30b-demand.vram.cache')?.byteLength ?? 0;
  assert.ok(
    cacheDiskSize <= store.getStats().maxSlots * slabBytes,
    `OPFS demand cache file (${cacheDiskSize}) exceeded bounded maxSlots * slabBytes`,
  );

  // Persist index on close, then inspect & reopen — warm slabs 4 & 5 must serve with 0 network fetches!
  store.close();
  const snap = await inspectOpfsVramStorage(dir);
  const demandEntry = snap.storedModels.find((m) => m.modelId === 'qwen3-30b-demand');
  assert.ok(demandEntry, 'Expected demand-cached model to appear in inspectOpfsVramStorage');
  assert.equal(demandEntry.storageMode, 'demand-cache');
  assert.equal(demandEntry.logicalByteLength, payload.byteLength);

  const fetchesBeforeReopen = syncFetches;
  const reopened = await openOpfsDemandModelBacking('qwen3-30b-demand', 'https://example.test/30b.gguf', {
    dirOverride: dir,
    syncRangeFetcher: mockRangeFetcher,
  });
  assert.ok(reopened, 'Expected openOpfsDemandModelBacking to reload persisted slab index');
  try {
    const warmCheck = new Uint8Array(4096);
    assert.equal(reopened.readRange(slabBytes * 5 + 1024, warmCheck), 4096);
    assert.deepEqual(warmCheck, payload.subarray(slabBytes * 5 + 1024, slabBytes * 5 + 5120));
    assert.equal(syncFetches, fetchesBeforeReopen, 'Expected warm slab hit after reopen with 0 network fetches');
  } finally {
    reopened.close();
  }
});

test('streamUrlToVramBacking auto-reclaims oldest inactive OPFS model before falling back and commits zero-copy without move()', async () => {
  const payload = makeFakeGgufPayload(64 * 1024);
  const { dir, files, mtimes } = createMockOpfsDir({ supportMove: false });

  // Seed two cached models: an old inactive model (50 KiB) and the active model (20 KiB)
  files.set('old-inactive.gguf', makeFakeGgufPayload(50 * 1024));
  mtimes.set('old-inactive.gguf', 1_000);
  files.set('active-model.gguf', makeFakeGgufPayload(20 * 1024));
  mtimes.set('active-model.gguf', 2_000);

  const okFetch: typeof fetch = async () =>
    new Response(payload.slice(), {
      status: 200,
      headers: {
        'content-length': String(payload.byteLength),
        'accept-ranges': 'bytes',
      },
    });

  // Total quota = 120 KiB, usage = 70 KiB (free = 50 KiB < 64 KiB needed) -> must evict old-inactive.gguf!
  const store = await streamUrlToVramBacking('new-model', 'https://example.test/new.gguf', {
    opfsDirectoryOverride: dir,
    fetchFn: okFetch,
    activeModelId: 'active-model',
    storageEstimateOverride: async () => ({
      usage:
        (files.get('old-inactive.gguf')?.byteLength ?? 0) +
        (files.get('active-model.gguf')?.byteLength ?? 0),
      quota: 120 * 1024,
    }),
    baseRetryDelayMs: 1,
  });

  try {
    assert.equal(store.kind, 'opfs-sync', 'Expected full opfs-sync after reclaiming old inactive model');
    assert.equal(files.has('old-inactive.gguf'), false, 'Expected oldest inactive model to be auto-reclaimed');
    assert.equal(files.has('active-model.gguf'), true, 'Expected active model to be protected from reclamation');
    assert.equal(files.has('new-model.gguf'), true);
    assert.equal(files.has('new-model.gguf.incomplete'), false, 'Expected .incomplete guard removed on commit');
    assert.equal(files.has('new-model.gguf.part'), false, 'Expected zero .part copy when move() is absent');

    const readBack = new Uint8Array(payload.byteLength);
    store.readRange(0, readBack);
    assert.deepEqual(readBack, payload);
  } finally {
    store.close();
  }
});

test('LlamaVTensorRuntime generates bit-exact greedy tokens over bounded OpfsDemandVramBackingStore vs MemoryBackingStore', async () => {
  const modelPath = path.resolve(process.cwd(), 'public/models/stories260K.gguf');
  const fileBytes = new Uint8Array(fs.readFileSync(modelPath));
  const { dir } = createMockOpfsDir();

  const slabBytes = 32 * 1024; // 32 KiB slabs over 1.13 MiB GGUF (~37 slabs)
  const cacheHandle = (await dir.getFileHandle('stories260k.vram.cache', {
    create: true,
  })) as FileSystemFileHandle & { createSyncAccessHandle: () => Promise<SyncAccessHandleLike> };
  const syncHandle = await cacheHandle.createSyncAccessHandle();

  // Strictly cap OPFS demand cache to 128 KiB (4 slabs for a 37-slab GGUF -> heavy LRU churn!)
  const demandStore = new OpfsDemandVramBackingStore({
    modelId: 'stories260k-demand',
    url: 'https://local.test/stories260K.gguf',
    totalBytes: fileBytes.byteLength,
    handle: syncHandle,
    maxCacheBytes: slabBytes * 4,
    slabBytes,
    syncRangeFetcher: (_url, start, endInclusive) => fileBytes.slice(start, endInclusive + 1),
  });
  const memStore = new MemoryBackingStore('stories260k-mem', fileBytes);

  const rtMem = new LlamaVTensorRuntime();
  const rtDemand = new LlamaVTensorRuntime();
  try {
    rtMem.loadFromBacking('stories260k-mem', 'Stories260K Mem', 'mem://stories260K', memStore);
    rtDemand.loadFromBacking('stories260k-demand', 'Stories260K Demand', 'opfs-demand://stories260K', demandStore, {
      weightResidencyPolicy: 'paged',
    });

    const sampling = {
      temperature: 0,
      topP: 1,
      topK: 1,
      repeatPenalty: 1.0,
      maxTokens: 12,
      fullContextOutput: false,
    };

    const memTokens: number[] = [];
    for await (const ev of rtMem.generate('Once upon a time', sampling)) {
      memTokens.push(ev.tokenId);
    }

    const demandTokens: number[] = [];
    for await (const ev of rtDemand.generate('Once upon a time', sampling)) {
      demandTokens.push(ev.tokenId);
    }

    assert.equal(demandTokens.length, 12);
    assert.deepEqual(
      demandTokens,
      memTokens,
      'Expected bit-exact greedy token parity between OpfsDemandVramBackingStore and MemoryBackingStore',
    );
    assert.ok(demandStore.getStats().slabEvictions > 0, 'Expected real OPFS slab evictions during paged inference');
  } finally {
    rtMem.unload();
    rtDemand.unload();
  }
});

test('OpfsDemandVramBackingStore uses 1 MiB default slabs, supports exact readRangesDirectAsync, parallel prefetchRanges, and AbortSignal', async () => {
  assert.equal(OPFS_DEMAND_SLAB_BYTES, 1024 * 1024);
  const slabBytes = 16 * 1024;
  const payload = makeFakeGgufPayload(slabBytes * 8); // 128 KiB
  const { dir } = createMockOpfsDir();
  const cacheHandle = (await dir.getFileHandle('async-demand.vram.cache', {
    create: true,
  })) as FileSystemFileHandle & { createSyncAccessHandle: () => Promise<SyncAccessHandleLike> };
  const syncHandle = await cacheHandle.createSyncAccessHandle();

  let syncCalls = 0;
  let asyncCalls = 0;
  let asyncBytesRequested = 0;

  const store = new OpfsDemandVramBackingStore({
    modelId: 'async-demand',
    url: 'https://example.test/async.gguf',
    totalBytes: payload.byteLength,
    handle: syncHandle,
    maxCacheBytes: slabBytes * 6,
    slabBytes,
    syncRangeFetcher: () => {
      syncCalls += 1;
      throw new Error('syncRangeFetcher should never be called when async prefetch is used');
    },
    asyncRangeFetcher: async (_url, start, endInclusive, signal) => {
      if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
      asyncCalls += 1;
      asyncBytesRequested += endInclusive - start + 1;
      await Promise.resolve();
      if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
      return payload.slice(start, endInclusive + 1);
    },
  });

  try {
    // 1. Direct exact reads for tiny scattered tensors must NOT fault slabs into the LRU cache
    const direct = await store.readRangesDirectAsync([
      { fileOffset: 128, byteLength: 64 },
      { fileOffset: 1024, byteLength: 128 }, // within 32 KiB gap -> coalesced into 1 HTTP request!
      { fileOffset: 96 * 1024, byteLength: 256 }, // far away -> 2nd HTTP request
    ]);
    assert.equal(direct.length, 3);
    assert.deepEqual(direct[0], payload.subarray(128, 192));
    assert.deepEqual(direct[1], payload.subarray(1024, 1152));
    assert.deepEqual(direct[2], payload.subarray(96 * 1024, 96 * 1024 + 256));
    assert.equal(asyncCalls, 2, 'Expected nearby tiny ranges to coalesce into 2 total Range fetches');
    assert.ok(asyncBytesRequested < 4096, `Expected exact small direct fetches, got ${asyncBytesRequested} B`);
    assert.equal(store.getStats().usedSlots, 0, 'Direct metadata reads must not pollute slab LRU');

    // 2. Parallel deduplicated slab prefetch -> subsequent sync readRange has 0 sync XHRs!
    await store.prefetchRanges([
      { fileOffset: 0, byteLength: slabBytes * 2 },
      { fileOffset: slabBytes, byteLength: slabBytes }, // duplicate slab 1 -> deduped
    ]);
    assert.equal(store.getStats().usedSlots, 2);
    const buf = new Uint8Array(slabBytes * 2);
    assert.equal(store.readRange(0, buf), slabBytes * 2);
    assert.deepEqual(buf, payload.subarray(0, slabBytes * 2));
    assert.equal(syncCalls, 0);

    // 3. AbortSignal immediately aborts prefetchRanges
    const ac = new AbortController();
    ac.abort();
    await assert.rejects(
      () => store.prefetchRanges([{ fileOffset: slabBytes * 4, byteLength: slabBytes * 2 }], { signal: ac.signal }),
      (err: unknown) => (err instanceof DOMException || err instanceof Error) && err.name === 'AbortError',
    );
  } finally {
    store.close();
  }
});

test('parseGgufFromBackingAsync and LlamaVTensorRuntime.loadFromBackingAsync + generate perform ZERO sync XHRs on OpfsDemandVramBackingStore', async () => {
  const modelPath = path.resolve(process.cwd(), 'public/models/stories260K.gguf');
  const fileBytes = new Uint8Array(fs.readFileSync(modelPath));
  const { dir } = createMockOpfsDir();

  const slabBytes = 32 * 1024;
  const cacheHandle = (await dir.getFileHandle('stories260k-zero-xhr.vram.cache', {
    create: true,
  })) as FileSystemFileHandle & { createSyncAccessHandle: () => Promise<SyncAccessHandleLike> };
  const syncHandle = await cacheHandle.createSyncAccessHandle();

  let syncXhrCalls = 0;
  let asyncFetchCalls = 0;
  const demandStore = new OpfsDemandVramBackingStore({
    modelId: 'stories260k-zero-xhr',
    url: 'https://local.test/stories260K.gguf',
    totalBytes: fileBytes.byteLength,
    handle: syncHandle,
    maxCacheBytes: slabBytes * 8,
    slabBytes,
    syncRangeFetcher: () => {
      syncXhrCalls += 1;
      throw new Error('Blocking sync XHR must NEVER be invoked during async demand execution!');
    },
    asyncRangeFetcher: async (_url, start, endInclusive, signal) => {
      if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
      asyncFetchCalls += 1;
      return fileBytes.slice(start, endInclusive + 1);
    },
  });

  // Verify async GGUF header parsing works with 0 sync XHRs
  const parsed = await parseGgufFromBackingAsync(demandStore);
  assert.ok(parsed.tensors.length > 0);
  assert.equal(syncXhrCalls, 0);

  const memStore = new MemoryBackingStore('stories260k-mem-ref', fileBytes);
  const rtMem = new LlamaVTensorRuntime();
  const rtDemand = new LlamaVTensorRuntime();
  try {
    rtMem.loadFromBacking('stories260k-mem-ref', 'Stories260K Mem', 'mem://stories260K', memStore);

    const loadPercents: number[] = [];
    await rtDemand.loadFromBackingAsync(
      'stories260k-zero-xhr',
      'Stories260K Async Demand',
      'opfs-demand://stories260K',
      demandStore,
      {
        weightResidencyPolicy: 'paged',
        onProgress: (p) => loadPercents.push(p.percent),
      },
    );
    assert.ok(loadPercents.length >= 2, 'Expected async load progress events between 97-99%');
    assert.equal(syncXhrCalls, 0, 'Expected 0 sync XHR calls during loadFromBackingAsync');

    // Speculative prefill must be a zero-cost no-op on opfs-demand
    const asyncBeforeSpec = asyncFetchCalls;
    const specRes = await rtDemand.speculativePrefill('Once upon a time');
    assert.equal(specRes.newlyPrefilledTokens, 0);
    assert.equal(asyncFetchCalls, asyncBeforeSpec, 'speculativePrefill must not fetch slabs on opfs-demand');

    const sampling = {
      temperature: 0,
      topP: 1,
      topK: 1,
      repeatPenalty: 1.0,
      maxTokens: 10,
      fullContextOutput: false,
    };

    const memTokens: number[] = [];
    for await (const ev of rtMem.generate('Once upon a time in a forest', sampling)) {
      memTokens.push(ev.tokenId);
    }

    const details: string[] = [];
    const demandTokens: number[] = [];
    for await (const ev of rtDemand.generate('Once upon a time in a forest', sampling, undefined, (p) => {
      if (p.detail) details.push(p.detail);
    })) {
      demandTokens.push(ev.tokenId);
    }

    assert.deepEqual(demandTokens, memTokens);
    assert.equal(syncXhrCalls, 0, 'Expected 0 sync XHR calls during full async generate()');
    assert.ok(details.some((d) => d.includes('Layer ')), 'Expected live layer progress details');

    // Verify immediate abort cleanly terminates without throwing
    const abortCtrl = new AbortController();
    let abortedTokens = 0;
    for await (const _ev of rtDemand.generate('Once upon a time', sampling, abortCtrl.signal, () => {
      abortCtrl.abort();
    })) {
      abortedTokens += 1;
    }
    assert.equal(abortedTokens, 0, 'Expected aborted prefill to stop before emitting tokens');
  } finally {
    rtMem.unload();
    rtDemand.unload();
  }
});

test('OPFS SyncAccessHandle reload lock recovery, read-only committed reads, and zero handle leaks', async () => {
  const validGguf = makeFakeGgufPayload(4096);

  // 1. Transient NoModificationAllowedError (simulated page reload teardown) recovers transparently
  let attempts = 0;
  let observedMode: string | undefined;
  let closed = false;
  const transientHandle = {
    createSyncAccessHandle: async (opts?: { mode?: 'read-only' | 'readwrite' | 'readwrite-unsafe' }) => {
      observedMode = opts?.mode;
      attempts += 1;
      if (attempts <= 2) {
        throw new DOMException(
          'Access Handles cannot be created if there is another open Access Handle',
          'NoModificationAllowedError',
        );
      }
      return {
        getSize: () => validGguf.byteLength,
        read: (buf: ArrayBufferView, o?: { at?: number }) => {
          const at = o?.at ?? 0;
          const dst = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
          const n = Math.min(dst.byteLength, Math.max(0, validGguf.byteLength - at));
          dst.set(validGguf.subarray(at, at + n), 0);
          return n;
        },
        write: () => 0,
        truncate: () => {},
        flush: () => {},
        close: () => {
          closed = true;
        },
      };
    },
  };

  const sync = await openSyncAccessHandleWithRetry(transientHandle, {
    mode: 'read-only',
    modelId: 'reload-test-model',
    retryDelaysMs: [5, 5, 5],
  });
  assert.equal(attempts, 3, 'Expected retry after 2 transient lock failures');
  assert.equal(observedMode, 'read-only', 'Committed read-only open must request mode: read-only');
  assert.equal(sync.getSize(), validGguf.byteLength);
  sync.close();
  assert.equal(closed, true);

  // 2. Older browser rejecting options bag with TypeError falls back cleanly to 0-arg createSyncAccessHandle()
  let zeroArgFallbackCalled = false;
  const legacyDir = {
    async getFileHandle(name: string) {
      if (name.endsWith('.incomplete')) throw new DOMException('Not found', 'NotFoundError');
      return {
        async createSyncAccessHandle(opts?: unknown) {
          if (opts !== undefined) {
            throw new TypeError('Unexpected options argument');
          }
          zeroArgFallbackCalled = true;
          return {
            getSize: () => validGguf.byteLength,
            read: (buf: ArrayBufferView, o?: { at?: number }) => {
              const at = o?.at ?? 0;
              const dst = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
              const n = Math.min(dst.byteLength, Math.max(0, validGguf.byteLength - at));
              dst.set(validGguf.subarray(at, at + n), 0);
              return n;
            },
            write: () => 0,
            truncate: () => {},
            flush: () => {},
            close: () => {},
          };
        },
      };
    },
  } as unknown as FileSystemDirectoryHandle;

  const backing = await openOpfsModelBacking('legacy-model', legacyDir, { retryDelaysMs: [5] });
  assert.ok(backing !== null);
  assert.equal(zeroArgFallbackCalled, true);
  backing.close();

  // 3. openOpfsDemandModelBacking closes syncCache (zero handle leak!) and throws OpfsLockError if syncIdx is locked
  let cacheHandleClosed = false;
  const idxMetaJson = JSON.stringify({
    version: 1,
    url: 'https://example.test/model.gguf',
    totalBytes: 4096,
    slabBytes: 1024,
    maxSlots: 4,
    slots: [[0, 0]],
    updatedAt: Date.now(),
  });
  const lockedDemandDir = {
    async getFileHandle(name: string) {
      if (name.endsWith('.vram.idx')) {
        return {
          async getFile() {
            return new Blob([idxMetaJson]);
          },
          async createSyncAccessHandle() {
            throw new DOMException('File is locked by another handle', 'NoModificationAllowedError');
          },
        };
      }
      if (name.endsWith('.vram.cache')) {
        return {
          async createSyncAccessHandle() {
            return {
              getSize: () => 4096,
              read: () => 0,
              write: () => 0,
              truncate: () => {},
              flush: () => {},
              close: () => {
                cacheHandleClosed = true;
              },
            };
          },
        };
      }
      throw new DOMException('Not found', 'NotFoundError');
    },
  } as unknown as FileSystemDirectoryHandle;

  await assert.rejects(
    () =>
      openOpfsDemandModelBacking('locked-demand-model', 'https://example.test/model.gguf', {
        dirOverride: lockedDemandDir,
        retryDelaysMs: [5, 5],
      }),
    (err: unknown) => {
      assert.ok(err instanceof OpfsLockError);
      assert.equal(isOpfsLockError(err), true);
      return true;
    },
  );
  assert.equal(
    cacheHandleClosed,
    true,
    'openOpfsDemandModelBacking MUST close syncCache when syncIdx fails to prevent permanent OPFS lock leak',
  );

  // 4. Overlapping double-buffered prefetchRanges deduplicates shared boundary slabs into 1 HTTP Range fetch
  const slabBytes = 16 * 1024;
  const payload = makeFakeGgufPayload(slabBytes * 4);
  const { dir: dedupDir } = createMockOpfsDir();
  const dedupHandle = (await dedupDir.getFileHandle('dedup.vram.cache', {
    create: true,
  })) as FileSystemFileHandle & { createSyncAccessHandle: () => Promise<SyncAccessHandleLike> };
  const dedupSync = await dedupHandle.createSyncAccessHandle();
  let fetchCount = 0;
  const store = new OpfsDemandVramBackingStore({
    modelId: 'dedup-test',
    url: 'https://example.test/dedup.gguf',
    totalBytes: payload.byteLength,
    handle: dedupSync,
    maxCacheBytes: slabBytes * 4,
    slabBytes,
    asyncRangeFetcher: async (_u, start, end) => {
      fetchCount += 1;
      await new Promise((resolve) => setTimeout(resolve, 15));
      return payload.subarray(start, end + 1);
    },
  });
  try {
    // Two overlapping double-buffered calls both touching slab 1 (bytes 16384..32767)
    await Promise.all([
      store.prefetchRanges([{ fileOffset: 8192, byteLength: 16384 }]),
      store.prefetchRanges([{ fileOffset: 24576, byteLength: 16384 }]),
    ]);
    // Slabs 0, 1, 2 -> exactly 3 fetches (NOT 4!)
    assert.equal(fetchCount, 3, `Expected shared boundary slab 1 to be deduplicated (got ${fetchCount} fetches)`);
    assert.equal(store.hasSlab(0), true);
    assert.equal(store.hasSlab(1), true);
    assert.equal(store.hasSlab(2), true);
  } finally {
    store.close();
  }
});
