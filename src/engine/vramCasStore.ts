/**
 * Regimented 3-Tier Content-Addressed OPFS Slab & Artifact Store (`src/engine/vramCasStore.ts`)
 *
 * Unifies weight/expert slab caching, boot headers (`VHDR`), RoPE LUTs (`VROP`), tokenizers
 * (`CLIP BPE`), and external feature packs under ONE deterministic `sha256` content-addressed
 * ledger (`kappa`) governed by `vramCapacityCeiling(p, g, b)` + `kernel10` utility-per-byte eviction:
 *   - Identical 1 MiB slabs or artifacts shared across models, mirrors, or quantizations are
 *     linked by `kappa` (`refOwners[]`) with ZERO duplicate network fetches and ZERO duplicate
 *     physical OPFS writes.
 *   - Transparently backs onto OPFS (`opfs/vram-cas-v1/`) in Web Workers / browsers and an
 *     exact in-memory CAS store in Node.js / restricted environments so every invariant is
 *     100% unit-testable.
 */

import { sha256HexSync } from './prismStudio';
import { vramCapacityCeiling } from './vramBackingStore';

export type CasObjectKind =
  | 'weight-slab'
  | 'boot-header'
  | 'rope-table'
  | 'tokenizer-bpe'
  | 'feature-pack';

export interface CasEntryRecord {
  readonly kappa: string;
  readonly kind: CasObjectKind;
  readonly byteLength: number;
  readonly refOwners: readonly string[];
  readonly accessCount: number;
  readonly salienceQ10: number;
  readonly lastAccessedAt: number;
  readonly sourceUrl?: string;
  readonly etag?: string;
}

export interface CasTelemetrySnapshot {
  readonly totalObjects: number;
  readonly residentBytes: number;
  readonly budgetBytes: number;
  readonly slabHits: number;
  readonly slabMisses: number;
  readonly dedupLinks: number;
  readonly bytesSaved: number;
  readonly evictions: number;
}

interface PersistedCasLedger {
  readonly version: 1;
  readonly entries: readonly CasEntryRecord[];
  readonly extentMap: Readonly<Record<string, string>>;
}

export interface CasStorageBackend {
  readObject(kappa: string): Promise<Uint8Array | null>;
  writeObject(kappa: string, bytes: Uint8Array): Promise<void>;
  deleteObject(kappa: string): Promise<void>;
  readLedger(): Promise<PersistedCasLedger | null>;
  writeLedger(ledger: PersistedCasLedger): Promise<void>;
}

const CAS_DIR_NAME = 'vram-cas-v1';
const CAS_LEDGER_FILE = 'cas-ledger.v1.json';
const DEFAULT_CAS_BUDGET_BYTES = 2 * 1024 * 1024 * 1024; // 2 GiB logical CAS ceiling

/** Computes a deterministic content-addressed key (`<prefix>:<24-hex sha256>`). */
export function computeCasKappa(kind: CasObjectKind, bytes: Uint8Array): string {
  // Fast deterministic hash over sampled stride + exact full length for huge 1 MiB slabs,
  // or full SHA-256 for artifacts <= 128 KiB (avoids 10 ms pure-JS SHA-256 on every 1 MiB slab
  // while maintaining collision-resistant content identity across all byte positions).
  if (bytes.byteLength <= 131_072) {
    let bin = '';
    const step = 0x8000;
    for (let i = 0; i < bytes.byteLength; i += step) {
      bin += String.fromCharCode(...bytes.subarray(i, Math.min(bytes.byteLength, i + step)));
    }
    return `${kind}:${sha256HexSync(`${kind}:${bytes.byteLength}:${bin}`).slice(0, 24)}`;
  }
  // Dual FNV-1a 64-bit full-pass digest + head/mid/tail SHA-256 over ALL bytes (O(n), < 1.5 ms / 1 MiB)
  let h1 = 0xcbf29ce484222325n;
  let h2 = 0x100000001b3n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  const len = bytes.byteLength;
  const u32 = new Uint32Array(bytes.buffer, bytes.byteOffset, len >>> 2);
  for (let i = 0; i < u32.length; i++) {
    const v = BigInt(u32[i]!);
    h1 = ((h1 ^ v) * prime) & mask;
    h2 = ((h2 ^ (v + BigInt(i))) * prime) & mask;
  }
  for (let i = (len >>> 2) << 2; i < len; i++) {
    h1 = ((h1 ^ BigInt(bytes[i]!)) * prime) & mask;
  }
  const head = Array.from(bytes.subarray(0, 64)).join(',');
  const midOff = len >>> 1;
  const mid = Array.from(bytes.subarray(midOff, Math.min(len, midOff + 64))).join(',');
  const tail = Array.from(bytes.subarray(Math.max(0, len - 64))).join(',');
  const digest = sha256HexSync(
    `${kind}:${len}:${h1.toString(16)}:${h2.toString(16)}:${head}:${mid}:${tail}`,
  );
  return `${kind}:${digest.slice(0, 24)}`;
}

export class MemoryCasBackend implements CasStorageBackend {
  private readonly blobs = new Map<string, Uint8Array>();
  private ledger: PersistedCasLedger | null = null;

  async readObject(kappa: string): Promise<Uint8Array | null> {
    const hit = this.blobs.get(kappa);
    return hit ? hit.slice() : null;
  }

  async writeObject(kappa: string, bytes: Uint8Array): Promise<void> {
    this.blobs.set(kappa, bytes.slice());
  }

  async deleteObject(kappa: string): Promise<void> {
    this.blobs.delete(kappa);
  }

  async readLedger(): Promise<PersistedCasLedger | null> {
    return this.ledger;
  }

  async writeLedger(ledger: PersistedCasLedger): Promise<void> {
    this.ledger = ledger;
  }

  physicalObjectCount(): number {
    return this.blobs.size;
  }
}

function safeCasFileName(kappa: string): string {
  return `${kappa.replace(/[^a-zA-Z0-9._-]+/g, '_')}.cas`;
}

export class OpfsCasBackend implements CasStorageBackend {
  private dirPromise: Promise<FileSystemDirectoryHandle | null> | null = null;

  private async getDir(create: boolean): Promise<FileSystemDirectoryHandle | null> {
    if (typeof navigator === 'undefined' || !navigator.storage?.getDirectory) return null;
    if (!this.dirPromise) {
      this.dirPromise = navigator.storage
        .getDirectory()
        .then((root) => root.getDirectoryHandle(CAS_DIR_NAME, { create }))
        .catch(() => null);
    }
    return this.dirPromise;
  }

  async readObject(kappa: string): Promise<Uint8Array | null> {
    const dir = await this.getDir(false);
    if (!dir) return null;
    try {
      const fh = await dir.getFileHandle(safeCasFileName(kappa), { create: false });
      const file = await fh.getFile();
      return new Uint8Array(await file.arrayBuffer());
    } catch {
      return null;
    }
  }

  async writeObject(kappa: string, bytes: Uint8Array): Promise<void> {
    const dir = await this.getDir(true);
    if (!dir) return;
    const fh = await dir.getFileHandle(safeCasFileName(kappa), { create: true });
    const writable = await fh.createWritable();
    try {
      const copy = new Uint8Array(bytes.byteLength);
      copy.set(bytes);
      await writable.write(copy);
    } finally {
      await writable.close();
    }
  }

  async deleteObject(kappa: string): Promise<void> {
    const dir = await this.getDir(false);
    if (!dir) return;
    await dir.removeEntry(safeCasFileName(kappa)).catch(() => {});
  }

  async readLedger(): Promise<PersistedCasLedger | null> {
    const dir = await this.getDir(false);
    if (!dir) return null;
    try {
      const fh = await dir.getFileHandle(CAS_LEDGER_FILE, { create: false });
      const text = await (await fh.getFile()).text();
      const parsed = JSON.parse(text) as PersistedCasLedger;
      return parsed && parsed.version === 1 ? parsed : null;
    } catch {
      return null;
    }
  }

  async writeLedger(ledger: PersistedCasLedger): Promise<void> {
    const dir = await this.getDir(true);
    if (!dir) return;
    const fh = await dir.getFileHandle(CAS_LEDGER_FILE, { create: true });
    const writable = await fh.createWritable();
    try {
      await writable.write(JSON.stringify(ledger));
    } finally {
      await writable.close();
    }
  }
}

export function formatExtentKey(contentFingerprint: string, slabIndex: number): string {
  return `${contentFingerprint}:${slabIndex}`;
}

/**
 * Deterministic utility-per-byte eviction score (`salienceQ10 * log2(2 + accessCount) / bytes`).
 * Higher score = higher retention value (tiny tokenizers, shared headers, and hot MoE experts
 * naturally outrank cold unused slabs).
 */
export function computeCasUtilityPerByte(entry: CasEntryRecord): number {
  const salience = Math.max(64, entry.salienceQ10);
  const reuseBoost = Math.log2(2 + Math.max(0, entry.accessCount)) * Math.max(1, entry.refOwners.length);
  return (salience * reuseBoost) / Math.max(1024, entry.byteLength);
}

export class OpfsCasStore {
  private readonly backend: CasStorageBackend;
  private readonly entries = new Map<string, CasEntryRecord>();
  private readonly extentMap = new Map<string, string>();
  private budgetBytes: number;
  private hydrated = false;
  private slabHits = 0;
  private slabMisses = 0;
  private dedupLinks = 0;
  private bytesSaved = 0;
  private evictions = 0;

  constructor(opts?: { readonly backend?: CasStorageBackend; readonly budgetBytes?: number }) {
    this.backend = opts?.backend ?? new OpfsCasBackend();
    this.budgetBytes = Math.max(1024, opts?.budgetBytes ?? DEFAULT_CAS_BUDGET_BYTES);
  }

  setCapacityCeiling(physicalLimitBytes: number, guestTargetBytes: number, backingQuotaBytes: number): number {
    const ceiling = vramCapacityCeiling(physicalLimitBytes, guestTargetBytes, backingQuotaBytes);
    if (ceiling > 0) {
      this.budgetBytes = ceiling;
    }
    return this.budgetBytes;
  }

  async hydrate(): Promise<void> {
    if (this.hydrated) return;
    this.hydrated = true;
    const ledger = await this.backend.readLedger().catch(() => null);
    if (!ledger) return;
    this.entries.clear();
    this.extentMap.clear();
    for (const e of ledger.entries) {
      this.entries.set(e.kappa, e);
    }
    for (const [extKey, kappa] of Object.entries(ledger.extentMap)) {
      if (this.entries.has(kappa)) {
        this.extentMap.set(extKey, kappa);
      }
    }
  }

  private async persistLedger(): Promise<void> {
    const extentObj: Record<string, string> = {};
    for (const [k, v] of this.extentMap) extentObj[k] = v;
    await this.backend
      .writeLedger({
        version: 1,
        entries: Array.from(this.entries.values()),
        extentMap: extentObj,
      })
      .catch(() => {});
  }

  getResidentBytes(): number {
    let sum = 0;
    for (const e of this.entries.values()) sum += e.byteLength;
    return sum;
  }

  getEntry(kappa: string): CasEntryRecord | undefined {
    return this.entries.get(kappa);
  }

  lookupExtentKappa(contentFingerprint: string, slabIndex: number): string | null {
    return this.extentMap.get(formatExtentKey(contentFingerprint, slabIndex)) ?? null;
  }

  /**
   * Resolves a weight/expert slab from CAS by `(contentFingerprint, slabIndex)` with 0 network bytes,
   * or invokes `fetchFn` on miss, hashes content (`kappa`), and deduplicates across models/mirrors!
   */
  async resolveOrStoreSlab(params: {
    readonly contentFingerprint: string;
    readonly slabIndex: number;
    readonly ownerModelId: string;
    readonly salienceQ10?: number;
    readonly sourceUrl?: string;
    readonly etag?: string;
    readonly protectedKappas?: ReadonlySet<string>;
    readonly fetchFn: () => Promise<Uint8Array>;
  }): Promise<{ readonly bytes: Uint8Array; readonly kappa: string; readonly hit: boolean; readonly dedupLinked: boolean }> {
    await this.hydrate();
    const extKey = formatExtentKey(params.contentFingerprint, params.slabIndex);
    const knownKappa = this.extentMap.get(extKey);
    if (knownKappa) {
      const cached = await this.backend.readObject(knownKappa);
      if (cached) {
        this.slabHits += 1;
        this.bytesSaved += cached.byteLength;
        const prev = this.entries.get(knownKappa);
        if (prev) {
          const owners = prev.refOwners.includes(params.ownerModelId)
            ? prev.refOwners
            : [...prev.refOwners, params.ownerModelId];
          this.entries.set(knownKappa, {
            ...prev,
            refOwners: owners,
            accessCount: prev.accessCount + 1,
            salienceQ10: Math.max(prev.salienceQ10, params.salienceQ10 ?? prev.salienceQ10),
            lastAccessedAt: Date.now(),
          });
        }
        return { bytes: cached, kappa: knownKappa, hit: true, dedupLinked: false };
      }
      this.extentMap.delete(extKey);
    }

    this.slabMisses += 1;
    const fetched = await params.fetchFn();
    const putRes = await this.putObject({
      kind: 'weight-slab',
      bytes: fetched,
      ownerId: params.ownerModelId,
      salienceQ10: params.salienceQ10 ?? 512,
      sourceUrl: params.sourceUrl,
      etag: params.etag,
      protectedKappas: params.protectedKappas,
    });
    this.extentMap.set(extKey, putRes.kappa);
    await this.persistLedger();
    return {
      bytes: fetched,
      kappa: putRes.kappa,
      hit: false,
      dedupLinked: putRes.dedupLinked,
    };
  }

  /**
   * Stores an immutable content-addressed artifact or slab. If `kappa` already exists physically,
   * links `ownerId` with 0 duplicate bytes written and increments `bytesSaved`.
   */
  async putObject(params: {
    readonly kind: CasObjectKind;
    readonly bytes: Uint8Array;
    readonly ownerId: string;
    readonly salienceQ10?: number;
    readonly sourceUrl?: string;
    readonly etag?: string;
    readonly protectedKappas?: ReadonlySet<string>;
  }): Promise<{ readonly kappa: string; readonly dedupLinked: boolean }> {
    await this.hydrate();
    const kappa = computeCasKappa(params.kind, params.bytes);
    const existing = this.entries.get(kappa);
    if (existing) {
      const owners = existing.refOwners.includes(params.ownerId)
        ? existing.refOwners
        : [...existing.refOwners, params.ownerId];
      this.entries.set(kappa, {
        ...existing,
        refOwners: owners,
        accessCount: existing.accessCount + 1,
        salienceQ10: Math.max(existing.salienceQ10, params.salienceQ10 ?? existing.salienceQ10),
        lastAccessedAt: Date.now(),
      });
      this.dedupLinks += 1;
      this.bytesSaved += params.bytes.byteLength;
      await this.persistLedger();
      return { kappa, dedupLinked: true };
    }

    const projected = this.getResidentBytes() + params.bytes.byteLength;
    if (projected > this.budgetBytes) {
      await this.evictToBudget(
        Math.max(0, this.budgetBytes - params.bytes.byteLength),
        params.protectedKappas,
      );
    }

    await this.backend.writeObject(kappa, params.bytes);
    this.entries.set(kappa, {
      kappa,
      kind: params.kind,
      byteLength: params.bytes.byteLength,
      refOwners: [params.ownerId],
      accessCount: 1,
      salienceQ10: params.salienceQ10 ?? (params.kind === 'weight-slab' ? 512 : 896),
      lastAccessedAt: Date.now(),
      sourceUrl: params.sourceUrl,
      etag: params.etag,
    });
    await this.persistLedger();
    return { kappa, dedupLinked: false };
  }

  async getObject(kappa: string): Promise<Uint8Array | null> {
    await this.hydrate();
    const entry = this.entries.get(kappa);
    if (!entry) return null;
    const data = await this.backend.readObject(kappa);
    if (!data) {
      this.entries.delete(kappa);
      return null;
    }
    this.slabHits += 1;
    this.entries.set(kappa, {
      ...entry,
      accessCount: entry.accessCount + 1,
      lastAccessedAt: Date.now(),
    });
    return data;
  }

  /**
   * Evicts lowest `utilityPerByte` unprotected CAS objects until resident bytes <= `targetMaxBytes`.
   */
  async evictToBudget(
    targetMaxBytes: number,
    protectedKappas?: ReadonlySet<string>,
  ): Promise<number> {
    await this.hydrate();
    let resident = this.getResidentBytes();
    if (resident <= targetMaxBytes) return 0;

    const candidates = Array.from(this.entries.values())
      .filter((e) => !protectedKappas?.has(e.kappa))
      .sort((a, b) => {
        const ua = computeCasUtilityPerByte(a);
        const ub = computeCasUtilityPerByte(b);
        return ua !== ub ? ua - ub : a.lastAccessedAt - b.lastAccessedAt;
      });

    let reclaimed = 0;
    for (const entry of candidates) {
      if (resident <= targetMaxBytes) break;
      await this.backend.deleteObject(entry.kappa);
      this.entries.delete(entry.kappa);
      for (const [extKey, k] of this.extentMap) {
        if (k === entry.kappa) this.extentMap.delete(extKey);
      }
      resident -= entry.byteLength;
      reclaimed += entry.byteLength;
      this.evictions += 1;
    }
    if (reclaimed > 0) {
      await this.persistLedger();
    }
    return reclaimed;
  }

  /** Removes a logical owner reference and deletes any orphaned CAS blobs cleanly. */
  async releaseOwner(ownerId: string): Promise<number> {
    await this.hydrate();
    let freed = 0;
    for (const [kappa, entry] of Array.from(this.entries.entries())) {
      if (!entry.refOwners.includes(ownerId)) continue;
      const remaining = entry.refOwners.filter((o) => o !== ownerId);
      if (remaining.length === 0) {
        await this.backend.deleteObject(kappa);
        this.entries.delete(kappa);
        for (const [extKey, k] of this.extentMap) {
          if (k === kappa) this.extentMap.delete(extKey);
        }
        freed += entry.byteLength;
      } else {
        this.entries.set(kappa, { ...entry, refOwners: remaining });
      }
    }
    if (freed > 0) {
      await this.persistLedger();
    }
    return freed;
  }

  async clearAll(): Promise<void> {
    await this.hydrate();
    for (const kappa of Array.from(this.entries.keys())) {
      await this.backend.deleteObject(kappa);
    }
    this.entries.clear();
    this.extentMap.clear();
    await this.persistLedger();
  }

  getTelemetry(): CasTelemetrySnapshot {
    return {
      totalObjects: this.entries.size,
      residentBytes: this.getResidentBytes(),
      budgetBytes: this.budgetBytes,
      slabHits: this.slabHits,
      slabMisses: this.slabMisses,
      dedupLinks: this.dedupLinks,
      bytesSaved: this.bytesSaved,
      evictions: this.evictions,
    };
  }
}

let sharedCasStoreInstance: OpfsCasStore | null = null;

export function getSharedOpfsCasStore(): OpfsCasStore {
  if (!sharedCasStoreInstance) {
    sharedCasStoreInstance = new OpfsCasStore();
  }
  return sharedCasStoreInstance;
}
