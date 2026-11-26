/**
 * Global MediaType Gallery Index & Space-as-Prism Membership Store
 * (`src/engine/contentGalleryStore.ts`)
 *
 * Implements DEC-014 (Global MediaType Galleries + Spaces-as-Prisms):
 * - Every artifact created anywhere in VerdantUI (from Chat Threads, Spaces, or Studio)
 *   is indexed globally by its top-level MediaType Gallery:
 *     Pictures | Audio | Videos | Documents | Games | Research | Utilities
 *   and further organized with organic sub-categories/tags.
 * - Spaces are Verdant Prisms (`mediaType: 'space'`) that reference arbitrary content across
 *   all MediaTypes and render a live customizable View of their content.
 * - Content created inside a Space is simultaneously viewable in its MediaType Gallery and
 *   inside its Space(s) — single source of truth, zero duplication.
 */

import type { PrismMediaType } from './prismMediaOrgans';
import type { VerdantPrismProject } from './prismStudio';

export type GalleryMediaType = Exclude<PrismMediaType, 'space'>;

export interface GalleryDescriptor {
  readonly mediaType: GalleryMediaType;
  readonly label: string;
  readonly singularLabel: string;
  readonly description: string;
}

export const MEDIA_GALLERIES: readonly GalleryDescriptor[] = [
  {
    mediaType: 'picture',
    label: 'Pictures',
    singularLabel: 'Picture',
    description: 'Generated artwork, landscapes, scenes, posters, and visual designs.',
  },
  {
    mediaType: 'video',
    label: 'Videos',
    singularLabel: 'Video',
    description: 'Procedural motion timelines, storyboards, and recorded WebM clips.',
  },
  {
    mediaType: 'audio',
    label: 'Audio',
    singularLabel: 'Audio',
    description: 'Multi-track stereo music, ambient soundscapes, rhythms, and sound effects.',
  },
  {
    mediaType: 'written-work',
    label: 'Documents',
    singularLabel: 'Document',
    description: 'Stories, articles, manuscripts, outlines, and structured Markdown documents.',
  },
  {
    mediaType: 'interactive-game',
    label: 'Games',
    singularLabel: 'Game',
    description: 'Playable puzzles, board games, arcade games, and cellular simulations.',
  },
  {
    mediaType: 'research-program',
    label: 'Research',
    singularLabel: 'Research',
    description: 'Hypotheses, experiments, evidence logs, and structured synthesis.',
  },
  {
    mediaType: 'tool-organ',
    label: 'Utilities',
    singularLabel: 'Utility',
    description: 'Calculators, habit trackers, schedule planners, and personal workflows.',
  },
];

export const GALLERY_LABEL_BY_TYPE: Readonly<Record<GalleryMediaType, string>> = Object.freeze({
  picture: 'Pictures',
  video: 'Videos',
  audio: 'Audio',
  'written-work': 'Documents',
  'interactive-game': 'Games',
  'research-program': 'Research',
  'tool-organ': 'Utilities',
});

export interface ContentArtifactRecord {
  readonly id: string;
  readonly mediaType: GalleryMediaType;
  readonly title: string;
  readonly summary: string;
  readonly categories: readonly string[];
  readonly spaceIds: readonly string[];
  readonly kappa: string;
  readonly published: boolean;
  readonly previewDataUrl?: string;
  readonly audioDurationSec?: number;
  readonly createdAt: number;
  readonly updatedAt: number;
}

interface PersistedGalleryMeta {
  readonly categories: readonly string[];
  readonly spaceIds: readonly string[];
  readonly previewDataUrl?: string;
  readonly audioDurationSec?: number;
}

const GALLERY_IDB_NAME = 'verdant-ui-galleries-v1';
const GALLERY_IDB_STORE = 'artifact_meta';

/**
 * Derives clean human sub-categories from a Prism's title, purpose, and procedural weights.
 */
export function inferCategoriesForArtifact(
  mediaType: GalleryMediaType,
  title: string,
  summary: string,
  weights?: Record<string, unknown>,
): readonly string[] {
  const cats = new Set<string>();
  const hay = `${title} ${summary}`.toLowerCase();

  if (mediaType === 'picture') {
    const gen = String(weights?.generator ?? '');
    if (gen === 'landscape-vista' || /\b(mountain|landscape|lake|valley|forest|sunset|sunrise|alpine)\b/.test(hay)) cats.add('Landscapes');
    if (gen === 'cosmic-nebula' || gen === 'constellation' || gen === 'constellation-atlas' || /\b(space|nebula|galaxy|star|planet|cosmos)\b/.test(hay)) cats.add('Space & Cosmos');
    if (gen === 'architectural-scene' || /\b(architecture|city|building|greenhouse|arch|skyline|temple)\b/.test(hay)) cats.add('Architecture');
    if (gen === 'botanical-garden' || gen === 'botanical-fractal' || /\b(botanical|flower|garden|tree|plant|forest|leaf)\b/.test(hay)) cats.add('Botanical');
    if (gen === 'ocean-seascape' || gen === 'wave-field' || /\b(ocean|sea|wave|coast|water|aurora)\b/.test(hay)) cats.add('Seascape & Flow');
    if (gen === 'poster-editorial' || /\b(poster|editorial|cover|banner|graphic)\b/.test(hay)) cats.add('Posters');
    if (gen === 'pixel-scene' || gen === 'pixel-sprite' || /\b(pixel|retro|8-?bit|sprite)\b/.test(hay)) cats.add('Pixel Art');
    if (gen === 'geometric-mosaic' || gen === 'abstract-flow' || cats.size === 0) cats.add('Generative Art');
  } else if (mediaType === 'audio') {
    const style = String(weights?.style ?? '');
    if (style === 'ambient-pad' || /\b(ambient|calm|sleep|meditat|soundscape)\b/.test(hay)) cats.add('Ambient');
    if (style === 'rhythm-pulse' || /\b(beat|drum|lo-?fi|techno|pulse|rhythm)\b/.test(hay)) cats.add('Beats & Rhythm');
    if (style === 'melodic-arpeggio' || /\b(melody|arpeggio|synth|piano|song|track)\b/.test(hay)) cats.add('Melodic');
    if (style === 'sound-effects' || /\b(sfx|effect|chime|alert|bell)\b/.test(hay)) cats.add('Sound Effects');
    if (cats.size === 0) cats.add('Music');
  } else if (mediaType === 'video') {
    const style = String(weights?.style ?? '');
    if (style === 'kinetic-type' || /\b(title|intro|kinetic|launch|promo)\b/.test(hay)) cats.add('Motion Titles');
    if (style === 'storyboard-scenes' || /\b(storyboard|film|trailer|scene)\b/.test(hay)) cats.add('Storyboards');
    if (cats.size === 0) cats.add('Motion');
  } else if (mediaType === 'written-work') {
    const genre = String(weights?.genre ?? '').trim();
    if (genre) cats.add(genre);
    if (/\b(sci-?fi|space|mars|future)\b/.test(hay)) cats.add('Sci-Fi');
    if (/\b(essay|article|report|guide|spec)\b/.test(hay)) cats.add('Articles');
    if (cats.size === 0) cats.add('Writing');
  } else if (mediaType === 'interactive-game') {
    const arch = String(weights?.archetype ?? '');
    if (arch === 'arcade-canvas') cats.add('Arcade');
    else if (arch === 'turn-board') cats.add('Board Games');
    else if (arch === 'cellular-sim') cats.add('Simulations');
    else cats.add('Puzzles');
  } else if (mediaType === 'research-program') {
    cats.add('Studies');
    if (/\b(experiment|lab|benchmark|eval)\b/.test(hay)) cats.add('Experiments');
  } else {
    const role = String(weights?.role ?? '');
    if (role === 'habit-guardian') cats.add('Habits');
    else if (role === 'schedule-guard') cats.add('Schedules');
    else if (role === 'calculator') cats.add('Calculators');
    else cats.add('Workflows');
  }

  return [...cats].slice(0, 4);
}

/**
 * Global index of all content artifacts across all Spaces, backed by IndexedDB metadata
 * and automatically synchronized with `PrismStore`.
 */
export class ContentGalleryStore {
  private readonly records = new Map<string, ContentArtifactRecord>();
  private readonly metaById = new Map<string, PersistedGalleryMeta>();
  private readonly listeners = new Set<(records: readonly ContentArtifactRecord[]) => void>();
  private hydrated = false;

  subscribe(listener: (records: readonly ContentArtifactRecord[]) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private emit(): void {
    const snap = this.listAll();
    for (const fn of this.listeners) fn(snap);
  }

  listAll(): readonly ContentArtifactRecord[] {
    return [...this.records.values()].sort((a, b) => b.updatedAt - a.updatedAt);
  }

  get(id: string): ContentArtifactRecord | null {
    return this.records.get(id) ?? null;
  }

  /** Synchronizes every non-Space Prism into the global MediaType gallery index. */
  syncFromPrisms(prisms: readonly VerdantPrismProject[]): void {
    const seen = new Set<string>();
    const validSpaceIds = new Set<string>();
    for (const p of prisms) {
      const mt = p.declaration?.view?.proceduralOrgan?.mediaType ?? p.declaration?.concept.mediaType;
      if (mt === 'space') validSpaceIds.add(p.id);
    }

    for (const p of prisms) {
      const rawType: PrismMediaType =
        p.declaration?.view?.proceduralOrgan?.mediaType ??
        p.declaration?.concept.mediaType ??
        'tool-organ';
      if (rawType === 'space') continue;
      seen.add(p.id);
      const snap = p.published ?? p.draft;
      const meta = this.metaById.get(p.id);
      const weights = p.declaration?.view?.proceduralOrgan?.weights as unknown as Record<string, unknown> | undefined;
      const categories =
        meta && meta.categories.length > 0
          ? meta.categories
          : inferCategoriesForArtifact(rawType, snap.name, snap.summary, weights);
      const cleanedSpaces = (meta?.spaceIds ?? []).filter((sid) => validSpaceIds.has(sid));

      this.records.set(p.id, {
        id: p.id,
        mediaType: rawType,
        title: snap.name,
        summary: snap.summary || p.declaration?.concept.purpose || '',
        categories,
        spaceIds: cleanedSpaces,
        kappa: snap.digest,
        published: p.published !== null,
        previewDataUrl: meta?.previewDataUrl,
        audioDurationSec: meta?.audioDurationSec,
        createdAt: p.createdAt,
        updatedAt: p.updatedAt,
      });
    }

    for (const id of [...this.records.keys()]) {
      if (!seen.has(id)) {
        this.records.delete(id);
      }
    }
    this.emit();
  }

  /** Updates cached thumbnail, sub-categories, or Space memberships for an artifact. */
  setArtifactMeta(
    artifactId: string,
    patch: {
      readonly categories?: readonly string[];
      readonly addSpaceId?: string;
      readonly removeSpaceId?: string;
      readonly previewDataUrl?: string;
      readonly audioDurationSec?: number;
    },
  ): ContentArtifactRecord | null {
    const existing = this.records.get(artifactId);
    const prevMeta = this.metaById.get(artifactId) ?? {
      categories: existing?.categories ?? [],
      spaceIds: existing?.spaceIds ?? [],
      previewDataUrl: existing?.previewDataUrl,
      audioDurationSec: existing?.audioDurationSec,
    };

    const nextSpaces = new Set(prevMeta.spaceIds);
    if (patch.addSpaceId) nextSpaces.add(patch.addSpaceId);
    if (patch.removeSpaceId) nextSpaces.delete(patch.removeSpaceId);

    const nextMeta: PersistedGalleryMeta = {
      categories: patch.categories ? [...new Set(patch.categories.map((c) => c.trim()).filter(Boolean))].slice(0, 6) : prevMeta.categories,
      spaceIds: [...nextSpaces],
      previewDataUrl: patch.previewDataUrl ?? prevMeta.previewDataUrl,
      audioDurationSec: patch.audioDurationSec ?? prevMeta.audioDurationSec,
    };

    this.metaById.set(artifactId, nextMeta);
    void this.persistMeta(artifactId, nextMeta);

    if (existing) {
      const updated: ContentArtifactRecord = {
        ...existing,
        categories: nextMeta.categories,
        spaceIds: nextMeta.spaceIds,
        previewDataUrl: nextMeta.previewDataUrl,
        audioDurationSec: nextMeta.audioDurationSec,
      };
      this.records.set(artifactId, updated);
      this.emit();
      return updated;
    }
    return null;
  }

  assignToSpace(artifactId: string, spacePrismId: string): ContentArtifactRecord | null {
    return this.setArtifactMeta(artifactId, { addSpaceId: spacePrismId });
  }

  removeFromSpace(artifactId: string, spacePrismId: string): ContentArtifactRecord | null {
    return this.setArtifactMeta(artifactId, { removeSpaceId: spacePrismId });
  }

  listByGallery(
    mediaType: GalleryMediaType,
    opts?: { readonly category?: string; readonly spaceId?: string; readonly query?: string },
  ): readonly ContentArtifactRecord[] {
    const q = (opts?.query ?? '').trim().toLowerCase();
    return this.listAll().filter((rec) => {
      if (rec.mediaType !== mediaType) return false;
      if (opts?.category && !rec.categories.some((c) => c.toLowerCase() === opts.category!.toLowerCase())) {
        return false;
      }
      if (opts?.spaceId && !rec.spaceIds.includes(opts.spaceId)) {
        return false;
      }
      if (q) {
        const hay = `${rec.title} ${rec.summary} ${rec.categories.join(' ')}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }

  listCategoriesForGallery(mediaType: GalleryMediaType): readonly string[] {
    const counts = new Map<string, number>();
    for (const rec of this.records.values()) {
      if (rec.mediaType !== mediaType) continue;
      for (const c of rec.categories) {
        counts.set(c, (counts.get(c) ?? 0) + 1);
      }
    }
    return [...counts.entries()]
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .map(([cat]) => cat);
  }

  listForSpace(spacePrismId: string): readonly ContentArtifactRecord[] {
    return this.listAll().filter((rec) => rec.spaceIds.includes(spacePrismId));
  }

  galleryCounts(): Readonly<Record<GalleryMediaType, number>> {
    const out: Record<GalleryMediaType, number> = {
      picture: 0,
      video: 0,
      audio: 0,
      'written-work': 0,
      'interactive-game': 0,
      'research-program': 0,
      'tool-organ': 0,
    };
    for (const rec of this.records.values()) {
      out[rec.mediaType] += 1;
    }
    return out;
  }

  async hydrate(): Promise<void> {
    if (this.hydrated || typeof indexedDB === 'undefined') {
      this.hydrated = true;
      return;
    }
    this.hydrated = true;
    try {
      const db = await openGalleryDb();
      const rows = await new Promise<Array<{ id: string; meta: PersistedGalleryMeta }>>((resolve, reject) => {
        const out: Array<{ id: string; meta: PersistedGalleryMeta }> = [];
        const tx = db.transaction(GALLERY_IDB_STORE, 'readonly');
        const req = tx.objectStore(GALLERY_IDB_STORE).openCursor();
        req.onsuccess = () => {
          const cursor = req.result;
          if (!cursor) {
            resolve(out);
            return;
          }
          out.push({ id: String(cursor.key), meta: cursor.value as PersistedGalleryMeta });
          cursor.continue();
        };
        req.onerror = () => reject(req.error);
      });
      for (const row of rows) {
        this.metaById.set(row.id, row.meta);
        const existing = this.records.get(row.id);
        if (existing) {
          this.records.set(row.id, {
            ...existing,
            categories: row.meta.categories.length > 0 ? row.meta.categories : existing.categories,
            spaceIds: row.meta.spaceIds,
            previewDataUrl: row.meta.previewDataUrl ?? existing.previewDataUrl,
            audioDurationSec: row.meta.audioDurationSec ?? existing.audioDurationSec,
          });
        }
      }
      this.emit();
    } catch {
      // Ignore IDB errors cleanly
    }
  }

  private async persistMeta(id: string, meta: PersistedGalleryMeta): Promise<void> {
    if (typeof indexedDB === 'undefined') return;
    try {
      const db = await openGalleryDb();
      await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(GALLERY_IDB_STORE, 'readwrite');
        tx.objectStore(GALLERY_IDB_STORE).put(meta, id);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      });
    } catch {
      // Ignore quota errors
    }
  }
}

function openGalleryDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(GALLERY_IDB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(GALLERY_IDB_STORE)) {
        db.createObjectStore(GALLERY_IDB_STORE);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
