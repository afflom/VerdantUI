import test from 'node:test';
import assert from 'node:assert/strict';
import { GgmlType, type GgufTensorInfo } from './gguf';
import {
  MemoryBackingStore,
  OpfsSyncBackingStore,
  coalesceTensorExtents,
  parseContentRangeHeader,
  sanitizeOpfsModelFileName,
  streamUrlToVramBacking,
  validateGgufMagicHeader,
  vramCapacityCeiling,
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

/** Deterministic in-memory FileSystemDirectoryHandle mock supporting createSyncAccessHandle & move(). */
function createMockOpfsDir(): {
  dir: FileSystemDirectoryHandle;
  files: Map<string, Uint8Array>;
} {
  const files = new Map<string, Uint8Array>();

  const makeHandle = (initialName: string): FileSystemFileHandle => {
    let currentName = initialName;
    const handle = {
      kind: 'file' as const,
      get name() {
        return currentName;
      },
      async getFile() {
        const data = files.get(currentName) ?? new Uint8Array(0);
        return {
          size: data.byteLength,
          lastModified: Date.now(),
          async text() {
            return new TextDecoder().decode(data);
          },
          async arrayBuffer() {
            return data.slice().buffer;
          },
        } as unknown as File;
      },
      async move(newName: string) {
        const data = files.get(currentName) ?? new Uint8Array(0);
        files.set(newName, data);
        if (newName !== currentName) {
          files.delete(currentName);
          currentName = newName;
        }
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
    return handle as unknown as FileSystemFileHandle;
  };

  const dir = {
    kind: 'directory' as const,
    name: 'vram-gguf-v1',
    async getFileHandle(name: string, opts?: { create?: boolean }) {
      if (!files.has(name)) {
        if (opts?.create) {
          files.set(name, new Uint8Array(0));
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
    },
  } as unknown as FileSystemDirectoryHandle;

  return { dir, files };
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
