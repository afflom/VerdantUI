/**
 * User-Managed Content Stream Catalogs (`src/engine/contentStreamCatalog.ts`)
 *
 * Organizes all user-controlled external knowledge & content streams into clean,
 * discoverable Catalogs while keeping VerdantUI's internal System Stream strictly hidden:
 *   - ALL built-in streams start OFF (`enabled: false`) by default so the user explicitly
 *     chooses every external source that enters their VerdantUI knowledge basis.
 *   - Users can add custom public HTTPS streams (RSS/Atom, JSON API, GitHub repo/docs,
 *     or public web reference URLs), search & filter across catalogs, and toggle streams
 *     individually or by catalog.
 *   - Persisted locally in IndexedDB (`verdant-ui-streams-v1`) with graceful in-memory
 *     fallback for Node.js tests and restricted contexts. Zero servers, zero tracking.
 */

export type StreamCatalogId = 'reference' | 'science' | 'technology' | 'world' | 'custom';

export type ContentStreamProviderKind =
  | 'wikipedia'
  | 'wikidata'
  | 'wiktionary'
  | 'openalex'
  | 'arxiv'
  | 'crossref'
  | 'github'
  | 'npm'
  | 'crates'
  | 'mdn'
  | 'open-meteo'
  | 'wikimedia'
  | 'osm'
  | 'rss'
  | 'json-api'
  | 'web-url';

export interface StreamCatalogMeta {
  readonly id: StreamCatalogId;
  readonly name: string;
  readonly description: string;
  readonly order: number;
}

export interface ContentStreamSpec {
  readonly id: string;
  readonly catalogId: StreamCatalogId;
  readonly name: string;
  readonly description: string;
  readonly providerKind: ContentStreamProviderKind;
  readonly endpointUrl: string;
  readonly enabled: boolean;
  readonly builtIn: boolean;
  readonly freshnessTtlMs: number;
  readonly tags: readonly string[];
  readonly lastSyncedAt: number | null;
  readonly syncedItemCount: number;
  readonly lastError?: string | null;
}

export interface ContentStreamsSnapshot {
  readonly catalogs: readonly StreamCatalogMeta[];
  readonly streams: readonly ContentStreamSpec[];
  readonly enabledCount: number;
  readonly customCount: number;
  readonly updatedAt: number;
}

export const STREAM_CATALOGS: readonly StreamCatalogMeta[] = [
  {
    id: 'reference',
    name: 'Reference & Knowledge',
    description: 'Encyclopedias, structured facts, and language definitions.',
    order: 0,
  },
  {
    id: 'science',
    name: 'Science & Research',
    description: 'Open-access papers, preprints, abstracts, and scholarly citations.',
    order: 1,
  },
  {
    id: 'technology',
    name: 'Code & Documentation',
    description: 'Open-source repositories, package registries, and web standards.',
    order: 2,
  },
  {
    id: 'world',
    name: 'Live World & Places',
    description: 'Weather forecasts, public media descriptions, and geographic places.',
    order: 3,
  },
  {
    id: 'custom',
    name: 'Custom Streams',
    description: 'Feeds, datasets, repositories, and web pages you have added.',
    order: 4,
  },
];

const HOUR_MS = 60 * 60 * 1000;

/**
 * Built-in Content Streams — EVERY entry starts `enabled: false` (strict opt-in).
 * Core neural runtime weights belong to the hidden unmanaged System Stream and are NEVER listed here.
 */
export const BUILTIN_CONTENT_STREAMS: readonly ContentStreamSpec[] = [
  {
    id: 'stream:wikipedia',
    catalogId: 'reference',
    name: 'Wikipedia',
    description: 'Live encyclopedia articles, summaries, and topic overviews.',
    providerKind: 'wikipedia',
    endpointUrl: 'https://en.wikipedia.org/w/api.php',
    enabled: false,
    builtIn: true,
    freshnessTtlMs: 24 * HOUR_MS,
    tags: ['encyclopedia', 'history', 'people', 'places', 'general', 'facts'],
    lastSyncedAt: null,
    syncedItemCount: 0,
  },
  {
    id: 'stream:wikidata',
    catalogId: 'reference',
    name: 'Wikidata',
    description: 'Structured entity descriptions, relationships, and factual identifiers.',
    providerKind: 'wikidata',
    endpointUrl: 'https://www.wikidata.org/w/api.php',
    enabled: false,
    builtIn: true,
    freshnessTtlMs: 24 * HOUR_MS,
    tags: ['entities', 'structured', 'facts', 'attributes'],
    lastSyncedAt: null,
    syncedItemCount: 0,
  },
  {
    id: 'stream:wiktionary',
    catalogId: 'reference',
    name: 'Wiktionary',
    description: 'Word definitions, etymologies, and usage notes.',
    providerKind: 'wiktionary',
    endpointUrl: 'https://en.wiktionary.org/w/api.php',
    enabled: false,
    builtIn: true,
    freshnessTtlMs: 72 * HOUR_MS,
    tags: ['dictionary', 'definitions', 'language', 'vocabulary'],
    lastSyncedAt: null,
    syncedItemCount: 0,
  },
  {
    id: 'stream:openalex',
    catalogId: 'science',
    name: 'OpenAlex',
    description: 'Open catalog of scientific papers, abstracts, authors, and citations.',
    providerKind: 'openalex',
    endpointUrl: 'https://api.openalex.org/works',
    enabled: false,
    builtIn: true,
    freshnessTtlMs: 24 * HOUR_MS,
    tags: ['science', 'papers', 'research', 'studies', 'citations', 'academic'],
    lastSyncedAt: null,
    syncedItemCount: 0,
  },
  {
    id: 'stream:arxiv',
    catalogId: 'science',
    name: 'arXiv Preprints',
    description: 'Recent research preprints in computing, mathematics, and physics.',
    providerKind: 'arxiv',
    endpointUrl: 'https://export.arxiv.org/api/query',
    enabled: false,
    builtIn: true,
    freshnessTtlMs: 12 * HOUR_MS,
    tags: ['preprints', 'ai', 'math', 'physics', 'cs', 'papers', 'research'],
    lastSyncedAt: null,
    syncedItemCount: 0,
  },
  {
    id: 'stream:crossref',
    catalogId: 'science',
    name: 'Crossref Literature',
    description: 'Published journal metadata, DOIs, and bibliographic references.',
    providerKind: 'crossref',
    endpointUrl: 'https://api.crossref.org/works',
    enabled: false,
    builtIn: true,
    freshnessTtlMs: 24 * HOUR_MS,
    tags: ['journals', 'doi', 'citations', 'literature'],
    lastSyncedAt: null,
    syncedItemCount: 0,
  },
  {
    id: 'stream:github',
    catalogId: 'technology',
    name: 'GitHub Public Repositories',
    description: 'Open-source repositories, documentation READMEs, and recent releases.',
    providerKind: 'github',
    endpointUrl: 'https://api.github.com/search/repositories',
    enabled: false,
    builtIn: true,
    freshnessTtlMs: 6 * HOUR_MS,
    tags: ['code', 'github', 'software', 'libraries', 'open-source', 'docs'],
    lastSyncedAt: null,
    syncedItemCount: 0,
  },
  {
    id: 'stream:npm',
    catalogId: 'technology',
    name: 'npm Package Registry',
    description: 'Latest JavaScript and TypeScript package versions and descriptions.',
    providerKind: 'npm',
    endpointUrl: 'https://registry.npmjs.org',
    enabled: false,
    builtIn: true,
    freshnessTtlMs: 6 * HOUR_MS,
    tags: ['npm', 'javascript', 'typescript', 'packages', 'code'],
    lastSyncedAt: null,
    syncedItemCount: 0,
  },
  {
    id: 'stream:crates',
    catalogId: 'technology',
    name: 'crates.io Registry',
    description: 'Rust crate metadata, versions, and documentation links.',
    providerKind: 'crates',
    endpointUrl: 'https://crates.io/api/v1/crates',
    enabled: false,
    builtIn: true,
    freshnessTtlMs: 6 * HOUR_MS,
    tags: ['rust', 'crates', 'wasm', 'systems', 'code'],
    lastSyncedAt: null,
    syncedItemCount: 0,
  },
  {
    id: 'stream:mdn',
    catalogId: 'technology',
    name: 'MDN Web Docs',
    description: 'Browser APIs, HTML, CSS, and WebAssembly reference documentation.',
    providerKind: 'mdn',
    endpointUrl: 'https://developer.mozilla.org/api/v1/search',
    enabled: false,
    builtIn: true,
    freshnessTtlMs: 24 * HOUR_MS,
    tags: ['web', 'html', 'css', 'browser', 'api', 'docs'],
    lastSyncedAt: null,
    syncedItemCount: 0,
  },
  {
    id: 'stream:open-meteo',
    catalogId: 'world',
    name: 'Open-Meteo Weather',
    description: 'Live local weather conditions, temperatures, and forecasts.',
    providerKind: 'open-meteo',
    endpointUrl: 'https://api.open-meteo.com/v1/forecast',
    enabled: false,
    builtIn: true,
    freshnessTtlMs: 15 * 60 * 1000,
    tags: ['weather', 'forecast', 'temperature', 'climate', 'rain'],
    lastSyncedAt: null,
    syncedItemCount: 0,
  },
  {
    id: 'stream:wikimedia',
    catalogId: 'world',
    name: 'Wikimedia Commons',
    description: 'Public media descriptions and visual subject references.',
    providerKind: 'wikimedia',
    endpointUrl: 'https://commons.wikimedia.org/w/api.php',
    enabled: false,
    builtIn: true,
    freshnessTtlMs: 24 * HOUR_MS,
    tags: ['media', 'images', 'visual', 'commons', 'subjects'],
    lastSyncedAt: null,
    syncedItemCount: 0,
  },
  {
    id: 'stream:osm',
    catalogId: 'world',
    name: 'OpenStreetMap Places',
    description: 'Geographic locations, landmarks, and public place metadata.',
    providerKind: 'osm',
    endpointUrl: 'https://nominatim.openstreetmap.org/search',
    enabled: false,
    builtIn: true,
    freshnessTtlMs: 24 * HOUR_MS,
    tags: ['places', 'geography', 'maps', 'cities', 'landmarks'],
    lastSyncedAt: null,
    syncedItemCount: 0,
  },
];

const STREAMS_IDB_NAME = 'verdant-ui-streams-v1';
const STREAMS_IDB_STORE = 'streams';
const STREAMS_RECORD_KEY = 'catalog-state:v1';

interface PersistedStreamsState {
  readonly version: 1;
  readonly enabledOverrides: Readonly<Record<string, boolean>>;
  readonly syncMeta: Readonly<
    Record<string, { readonly lastSyncedAt: number | null; readonly syncedItemCount: number; readonly lastError?: string | null }>
  >;
  readonly customStreams: readonly ContentStreamSpec[];
}

let cachedStreamsDb: Promise<IDBDatabase> | null = null;

function openStreamsDb(): Promise<IDBDatabase> {
  if (cachedStreamsDb) return cachedStreamsDb;
  cachedStreamsDb = new Promise<IDBDatabase>((resolve, reject) => {
    const req = indexedDB.open(STREAMS_IDB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STREAMS_IDB_STORE)) {
        db.createObjectStore(STREAMS_IDB_STORE);
      }
    };
    req.onsuccess = () => {
      const db = req.result;
      db.onclose = () => {
        cachedStreamsDb = null;
      };
      db.onversionchange = () => {
        db.close();
        cachedStreamsDb = null;
      };
      resolve(db);
    };
    req.onerror = () => {
      cachedStreamsDb = null;
      reject(req.error);
    };
  });
  return cachedStreamsDb;
}

/** Validates and normalizes a user-supplied custom Content Stream URL (HTTPS only). */
export function normalizeCustomStreamInput(input: {
  readonly name?: string;
  readonly url: string;
  readonly description?: string;
  readonly formatHint?: 'auto' | 'rss' | 'json-api' | 'github' | 'web-url';
}): { readonly ok: true; readonly spec: Omit<ContentStreamSpec, 'id' | 'lastSyncedAt' | 'syncedItemCount'> } | { readonly ok: false; readonly error: string } {
  const rawUrl = input.url.trim();
  if (!rawUrl) {
    return { ok: false, error: 'Enter a public https:// address for the stream.' };
  }
  let parsed: URL;
  try {
    parsed = new URL(rawUrl);
  } catch {
    return { ok: false, error: 'That address is not a valid URL. Include https:// at the start.' };
  }
  if (parsed.protocol !== 'https:') {
    return { ok: false, error: 'Only secure https:// streams are supported.' };
  }
  const host = parsed.hostname.toLowerCase();
  if (
    host === 'localhost' ||
    host === '127.0.0.1' ||
    host === '0.0.0.0' ||
    host.endsWith('.local') ||
    /^10\./.test(host) ||
    /^192\.168\./.test(host) ||
    /^172\.(1[6-9]|2\d|3[0-1])\./.test(host)
  ) {
    return { ok: false, error: 'Please provide a public internet https:// address.' };
  }

  let providerKind: ContentStreamProviderKind = 'web-url';
  let endpointUrl = parsed.toString();
  let defaultName = input.name?.trim() || host.replace(/^www\./, '');
  let defaultDesc = input.description?.trim() || `Custom web stream from ${host}.`;
  const tags: string[] = [host.replace(/^www\./, '')];

  if (input.formatHint && input.formatHint !== 'auto') {
    providerKind = input.formatHint;
  } else if (host === 'github.com' || host === 'www.github.com') {
    const parts = parsed.pathname.split('/').filter(Boolean);
    if (parts.length >= 2) {
      const owner = parts[0]!;
      const repo = parts[1]!.replace(/\.git$/i, '');
      providerKind = 'github';
      endpointUrl = `https://api.github.com/repos/${owner}/${repo}`;
      if (!input.name?.trim()) defaultName = `${owner}/${repo}`;
      if (!input.description?.trim()) defaultDesc = `GitHub repository stream for ${owner}/${repo}.`;
      tags.push(owner.toLowerCase(), repo.toLowerCase(), 'github', 'code');
    }
  } else if (/\.(xml|rss|atom)(\?|$)/i.test(parsed.pathname) || /\/(feed|rss|atom)\b/i.test(parsed.pathname)) {
    providerKind = 'rss';
    if (!input.description?.trim()) defaultDesc = `RSS / Atom feed from ${host}.`;
    tags.push('feed', 'rss', 'updates');
  } else if (/\.json(\?|$)/i.test(parsed.pathname) || /\/api\//i.test(parsed.pathname)) {
    providerKind = 'json-api';
    if (!input.description?.trim()) defaultDesc = `Structured JSON data stream from ${host}.`;
    tags.push('json', 'data', 'api');
  }

  return {
    ok: true,
    spec: {
      catalogId: 'custom',
      name: defaultName.slice(0, 64),
      description: defaultDesc.slice(0, 160),
      providerKind,
      endpointUrl,
      enabled: true,
      builtIn: false,
      freshnessTtlMs: providerKind === 'rss' ? 30 * 60 * 1000 : 6 * HOUR_MS,
      tags,
    },
  };
}

export class ContentStreamCatalogStore {
  private readonly streamsById = new Map<string, ContentStreamSpec>();
  private readonly listeners = new Set<() => void>();
  private snapshotCache: ContentStreamsSnapshot;
  private hydrated = false;
  private persistDirty = false;
  private persistFlushPromise: Promise<void> | null = null;

  constructor() {
    for (const s of BUILTIN_CONTENT_STREAMS) {
      this.streamsById.set(s.id, { ...s });
    }
    this.snapshotCache = this.buildSnapshot();
  }

  private buildSnapshot(): ContentStreamsSnapshot {
    const streams = Array.from(this.streamsById.values());
    let enabledCount = 0;
    let customCount = 0;
    for (const s of streams) {
      if (s.enabled) enabledCount += 1;
      if (!s.builtIn) customCount += 1;
    }
    return {
      catalogs: STREAM_CATALOGS,
      streams,
      enabledCount,
      customCount,
      updatedAt: Date.now(),
    };
  }

  private schedulePersist(): void {
    if (typeof indexedDB === 'undefined') return;
    this.persistDirty = true;
    if (this.persistFlushPromise) return;
    this.persistFlushPromise = (async () => {
      await Promise.resolve();
      while (this.persistDirty) {
        this.persistDirty = false;
        await this.persist();
      }
    })().finally(() => {
      this.persistFlushPromise = null;
      if (this.persistDirty) {
        this.schedulePersist();
      }
    });
  }

  private notify(): void {
    this.snapshotCache = this.buildSnapshot();
    for (const fn of this.listeners) fn();
    this.schedulePersist();
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  getSnapshot(): ContentStreamsSnapshot {
    return this.snapshotCache;
  }

  listAll(): readonly ContentStreamSpec[] {
    return this.snapshotCache.streams;
  }

  getEnabledStreams(): readonly ContentStreamSpec[] {
    return this.snapshotCache.streams.filter((s) => s.enabled);
  }

  hasEnabledStreams(): boolean {
    return this.snapshotCache.enabledCount > 0;
  }

  getStream(id: string): ContentStreamSpec | undefined {
    return this.streamsById.get(id);
  }

  toggleStream(streamId: string, enabled?: boolean): boolean {
    const cur = this.streamsById.get(streamId);
    if (!cur) return false;
    const nextEnabled = enabled !== undefined ? enabled : !cur.enabled;
    if (cur.enabled === nextEnabled) return true;
    this.streamsById.set(streamId, { ...cur, enabled: nextEnabled, lastError: null });
    this.notify();
    return true;
  }

  toggleCatalog(catalogId: StreamCatalogId, enabled: boolean): number {
    let changed = 0;
    for (const [id, s] of this.streamsById) {
      if (s.catalogId === catalogId && s.enabled !== enabled) {
        this.streamsById.set(id, { ...s, enabled, lastError: null });
        changed += 1;
      }
    }
    if (changed > 0) this.notify();
    return changed;
  }

  addCustomStream(input: {
    readonly name?: string;
    readonly url: string;
    readonly description?: string;
    readonly formatHint?: 'auto' | 'rss' | 'json-api' | 'github' | 'web-url';
  }): { readonly ok: true; readonly stream: ContentStreamSpec } | { readonly ok: false; readonly error: string } {
    const normalized = normalizeCustomStreamInput(input);
    if (!normalized.ok) return normalized;

    // Deduplicate by exact endpointUrl
    for (const existing of this.streamsById.values()) {
      if (existing.endpointUrl.toLowerCase() === normalized.spec.endpointUrl.toLowerCase()) {
        if (!existing.enabled) {
          this.streamsById.set(existing.id, { ...existing, enabled: true });
          this.notify();
        }
        return { ok: true, stream: this.streamsById.get(existing.id)! };
      }
    }

    const id = `custom:${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
    const spec: ContentStreamSpec = {
      id,
      ...normalized.spec,
      lastSyncedAt: null,
      syncedItemCount: 0,
      lastError: null,
    };
    this.streamsById.set(id, spec);
    this.notify();
    return { ok: true, stream: spec };
  }

  removeCustomStream(streamId: string): boolean {
    const cur = this.streamsById.get(streamId);
    if (!cur || cur.builtIn) return false;
    this.streamsById.delete(streamId);
    this.notify();
    return true;
  }

  recordStreamSync(streamId: string, addedItems: number, error?: string | null): void {
    const cur = this.streamsById.get(streamId);
    if (!cur) return;
    this.streamsById.set(streamId, {
      ...cur,
      lastSyncedAt: error ? cur.lastSyncedAt : Date.now(),
      syncedItemCount: cur.syncedItemCount + Math.max(0, addedItems),
      lastError: error ?? null,
    });
    this.notify();
  }

  searchCatalogs(
    query: string,
    filter: 'all' | 'active' | 'custom' = 'all',
  ): readonly { readonly catalog: StreamCatalogMeta; readonly streams: readonly ContentStreamSpec[] }[] {
    const q = query.trim().toLowerCase();
    return STREAM_CATALOGS.map((catalog) => {
      const streams = this.snapshotCache.streams.filter((s) => {
        if (s.catalogId !== catalog.id) return false;
        if (filter === 'active' && !s.enabled) return false;
        if (filter === 'custom' && s.builtIn) return false;
        if (!q) return true;
        const hay = `${catalog.name} ${s.name} ${s.description} ${s.tags.join(' ')} ${s.endpointUrl}`.toLowerCase();
        return hay.includes(q);
      });
      return { catalog, streams };
    }).filter((group) => group.streams.length > 0 || (!q && filter === 'all' && group.catalog.id !== 'custom'));
  }

  async hydrate(): Promise<void> {
    if (this.hydrated || typeof indexedDB === 'undefined') {
      this.hydrated = true;
      return;
    }
    this.hydrated = true;
    try {
      const db = await openStreamsDb();
      const loaded = await new Promise<PersistedStreamsState | undefined>((resolve, reject) => {
        const tx = db.transaction(STREAMS_IDB_STORE, 'readonly');
        const req = tx.objectStore(STREAMS_IDB_STORE).get(STREAMS_RECORD_KEY);
        req.onsuccess = () => resolve(req.result as PersistedStreamsState | undefined);
        req.onerror = () => reject(req.error);
      });
      if (loaded && loaded.version === 1) {
        for (const base of BUILTIN_CONTENT_STREAMS) {
          const override = loaded.enabledOverrides?.[base.id];
          const meta = loaded.syncMeta?.[base.id];
          this.streamsById.set(base.id, {
            ...base,
            enabled: typeof override === 'boolean' ? override : base.enabled,
            lastSyncedAt: meta?.lastSyncedAt ?? null,
            syncedItemCount: meta?.syncedItemCount ?? 0,
            lastError: meta?.lastError ?? null,
          });
        }
        for (const custom of loaded.customStreams ?? []) {
          if (custom && typeof custom.id === 'string' && !custom.builtIn) {
            this.streamsById.set(custom.id, custom);
          }
        }
        this.snapshotCache = this.buildSnapshot();
        for (const fn of this.listeners) fn();
      }
    } catch {
      // Keep clean in-memory state
    }
  }

  async persist(): Promise<void> {
    if (typeof indexedDB === 'undefined') return;
    try {
      const enabledOverrides: Record<string, boolean> = {};
      const syncMeta: Record<string, { lastSyncedAt: number | null; syncedItemCount: number; lastError?: string | null }> = {};
      const customStreams: ContentStreamSpec[] = [];
      for (const s of this.streamsById.values()) {
        if (s.builtIn) {
          enabledOverrides[s.id] = s.enabled;
          if (s.lastSyncedAt !== null || s.syncedItemCount > 0 || s.lastError) {
            syncMeta[s.id] = {
              lastSyncedAt: s.lastSyncedAt,
              syncedItemCount: s.syncedItemCount,
              lastError: s.lastError,
            };
          }
        } else {
          customStreams.push(s);
        }
      }
      const payload: PersistedStreamsState = {
        version: 1,
        enabledOverrides,
        syncMeta,
        customStreams,
      };
      const db = await openStreamsDb();
      await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STREAMS_IDB_STORE, 'readwrite');
        tx.objectStore(STREAMS_IDB_STORE).put(payload, STREAMS_RECORD_KEY);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      });
    } catch {
      // Ignore quota errors
    }
  }
}
