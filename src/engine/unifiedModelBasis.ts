/**
 * VerdantUI Unified Model Expert Registry & Non-Static Living Training Basis
 * (`src/engine/unifiedModelBasis.ts`)
 *
 * Implements DEC-014 (One Unified Model of Composable Experts & Living Training Basis):
 * - Every part of VerdantUI — downloaded neural weights (language, vision, speech, image),
 *   procedural multimodal organs (picture, audio, video, game, document, research, utility),
 *   user preferences, Space Prisms, and saved content artifacts — is registered as a coherent
 *   Expert inside ONE unified VerdantUI model sharing the `verdant-engine` / `vTensor` substrate.
 * - All experts inhabit the shared 8D exact-integer `Q10` (`0..1024`) cognitive space with
 *   content-addressed `kappa` digests and `O(nnz)` sparse activation tracking.
 * - The training basis is NEVER static: `LivingTrainingBasis` continuously adapts exact-integer
 *   `Q10` routing & style affinity weights from user creations, refinements, and interactions,
 *   acquires newly synthesized Prism Experts as permanent skills, and indexes content `kappa`
 *   knowledge locally in IndexedDB (`verdant-ui-basis-v1`, zero servers, 100% private).
 */

import {
  COGNITIVE_DIM,
  COGNITIVE_UNIT,
  dotCognitiveQ10,
  type CognitiveVector,
} from './cognitiveObserver';
import { prismContentDigest } from './prismKernel';
import type { PrismMediaType } from './prismMediaOrgans';
import type { VerdantPrismProject } from './prismStudio';

export type UnifiedExpertKind =
  | 'neural-weights'
  | 'procedural-organ'
  | 'space-prism'
  | 'content-artifact'
  | 'user-basis';

export type ExpertResidencyTier = 'hot-vtensor' | 'warm-opfs' | 'cold-declarative';

export interface UnifiedExpertDescriptor {
  readonly expertId: string;
  readonly kind: UnifiedExpertKind;
  readonly label: string;
  readonly mediaType?: PrismMediaType;
  /** Content-addressed SHA-256 identity (`kappa`) binding weights or declarative spec. */
  readonly kappa: string;
  /** 8D exact-integer Q10 signature in the shared Walsh-Hadamard cognitive space. */
  readonly facetsQ10: CognitiveVector;
  readonly residency: ExpertResidencyTier;
  readonly residentBytes: number;
  readonly activationCount: number;
  readonly lastActivatedAt: number;
  readonly tags?: readonly string[];
}

export interface BasisKnowledgeEntry {
  readonly kappa: string;
  readonly expertId: string;
  readonly kind: 'artifact' | 'space' | 'skill';
  readonly title: string;
  readonly mediaType: PrismMediaType;
  readonly summary: string;
  readonly categories: readonly string[];
  readonly updatedAt: number;
}

export interface UserStyleAffinities {
  readonly preferredPictureStyle?: string;
  readonly preferredPictureLighting?: string;
  readonly preferredPalette?: readonly [string, string, string, string];
  readonly preferredAudioScale?: 'pentatonic' | 'major' | 'minor' | 'dorian';
  readonly preferredAudioBpm?: number;
  readonly preferredVideoAspect?: '16:9' | '1:1' | '9:16';
  /** Per-MediaType Q10 familiarity/affinity scores (0..1024). */
  readonly mediaAffinityQ10: Readonly<Record<string, number>>;
}

export interface LivingBasisSnapshot {
  readonly version: 1;
  readonly adaptationSteps: number;
  readonly updatedAt: number;
  readonly userFacetsQ10: CognitiveVector;
  readonly styleAffinities: UserStyleAffinities;
  readonly acquiredSkillIds: readonly string[];
  readonly knowledgeIndex: readonly BasisKnowledgeEntry[];
  readonly basisDigest: string;
}

export interface UnifiedModelTelemetrySnapshot {
  readonly totalExperts: number;
  readonly neuralExperts: number;
  readonly proceduralExperts: number;
  readonly spaceExperts: number;
  readonly contentExperts: number;
  readonly hotResidentBytes: number;
  readonly warmOpfsBytes: number;
  readonly adaptationSteps: number;
  readonly acquiredSkills: number;
  readonly knowledgeEntries: number;
  readonly basisDigest: string;
  readonly topActiveExperts: readonly {
    readonly expertId: string;
    readonly kind: UnifiedExpertKind;
    readonly label: string;
    readonly residency: ExpertResidencyTier;
    readonly activations: number;
  }[];
}

const BASIS_IDB_NAME = 'verdant-ui-basis-v1';
const BASIS_IDB_STORE = 'basis';
const BASIS_RECORD_KEY = 'living-basis:owner';
const MAX_KNOWLEDGE_ENTRIES = 256;

function clampQ10(v: number): number {
  const n = Math.round(v);
  if (n < 0) return 0;
  if (n > COGNITIVE_UNIT) return COGNITIVE_UNIT;
  return n;
}

export function facetsQ10ForMediaType(mediaType: PrismMediaType): CognitiveVector {
  switch (mediaType) {
    case 'picture':
      return [384, 512, 384, 640, 256, 512, 768, 768];
    case 'audio':
      return [384, 448, 320, 640, 256, 512, 640, 768];
    case 'video':
      return [448, 576, 448, 640, 320, 640, 832, 832];
    case 'interactive-game':
      return [704, 704, 832, 640, 256, 512, 640, 832];
    case 'written-work':
      return [320, 640, 256, 576, 384, 768, 960, 576];
    case 'research-program':
      return [896, 960, 512, 768, 512, 768, 896, 832];
    case 'tool-organ':
      return [832, 576, 640, 896, 192, 640, 512, 960];
    case 'space':
      return [512, 640, 448, 896, 256, 896, 768, 896];
  }
}

const DEFAULT_PROCEDURAL_EXPERTS: readonly {
  readonly id: string;
  readonly label: string;
  readonly mediaType: PrismMediaType;
  readonly tags: readonly string[];
}[] = [
  { id: 'procedural:picture', label: 'Picture Procedural Synthesizer', mediaType: 'picture', tags: ['image', 'art', 'scene', 'png'] },
  { id: 'procedural:audio', label: 'Modal Stereo Audio Synthesizer', mediaType: 'audio', tags: ['audio', 'music', 'wav', 'synth'] },
  { id: 'procedural:video', label: 'Motion Video Compositor', mediaType: 'video', tags: ['video', 'motion', 'webm', 'storyboard'] },
  { id: 'procedural:game', label: 'Interactive Game Engine', mediaType: 'interactive-game', tags: ['game', 'simulation', 'puzzle', 'arcade'] },
  { id: 'procedural:document', label: 'Document & Story Studio', mediaType: 'written-work', tags: ['writing', 'document', 'story', 'markdown'] },
  { id: 'procedural:research', label: 'Research & Evidence Studio', mediaType: 'research-program', tags: ['research', 'hypothesis', 'experiment'] },
  { id: 'procedural:utility', label: 'Utility & Workflow Engine', mediaType: 'tool-organ', tags: ['utility', 'habit', 'schedule', 'calculator'] },
  { id: 'procedural:space', label: 'Space Prism Compositor', mediaType: 'space', tags: ['space', 'collection', 'view', 'prism'] },
];

const INITIAL_USER_FACETS: CognitiveVector = [512, 512, 512, 640, 384, 640, 576, 640];

function computeBasisDigest(
  adaptationSteps: number,
  userFacetsQ10: CognitiveVector,
  acquiredSkillIds: readonly string[],
  knowledgeIndex: readonly BasisKnowledgeEntry[],
): string {
  const payload = [
    adaptationSteps,
    userFacetsQ10.join(','),
    acquiredSkillIds.slice(-16).join(','),
    knowledgeIndex.slice(-16).map((k) => `${k.kappa.slice(0, 12)}:${k.mediaType}`).join(','),
  ].join('|');
  return prismContentDigest(`verdant-basis:v1:${payload}`);
}

/**
 * Persistent, non-static training basis for the unified VerdantUI model.
 * Continuously evolves exact-integer `Q10` preference weights, acquired Prism skill IDs,
 * and content-addressed `kappa` knowledge entries across sessions.
 */
export class LivingTrainingBasis {
  private adaptationSteps = 0;
  private updatedAt = Date.now();
  private userFacetsQ10: CognitiveVector = INITIAL_USER_FACETS;
  private styleAffinities: UserStyleAffinities = {
    mediaAffinityQ10: {
      picture: 512,
      audio: 512,
      video: 512,
      'written-work': 512,
      'interactive-game': 512,
      'research-program': 512,
      'tool-organ': 512,
      space: 512,
    },
  };
  private readonly acquiredSkillIds = new Set<string>();
  private readonly knowledgeByKappa = new Map<string, BasisKnowledgeEntry>();
  private readonly listeners = new Set<(snap: LivingBasisSnapshot) => void>();
  private hydrated = false;

  subscribe(listener: (snap: LivingBasisSnapshot) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private emit(): void {
    const snap = this.snapshot();
    for (const fn of this.listeners) fn(snap);
  }

  snapshot(): LivingBasisSnapshot {
    const acquired = [...this.acquiredSkillIds];
    const knowledge = [...this.knowledgeByKappa.values()].sort((a, b) => b.updatedAt - a.updatedAt);
    return {
      version: 1,
      adaptationSteps: this.adaptationSteps,
      updatedAt: this.updatedAt,
      userFacetsQ10: this.userFacetsQ10,
      styleAffinities: this.styleAffinities,
      acquiredSkillIds: acquired,
      knowledgeIndex: knowledge,
      basisDigest: computeBasisDigest(this.adaptationSteps, this.userFacetsQ10, acquired, knowledge),
    };
  }

  getStyleAffinities(): UserStyleAffinities {
    return this.styleAffinities;
  }

  getUserFacetsQ10(): CognitiveVector {
    return this.userFacetsQ10;
  }

  /**
   * Online continual `Q10` Hebbian / Predictive Coding adaptation step:
   * blends the observed interaction's cognitive signature into the living user basis
   * (7/8 retention + 1/8 online update — bounded strictly in `[0, 1024]`, zero float drift)
   * and updates learned media/style preferences.
   */
  recordAdaptation(event: {
    readonly mediaType: PrismMediaType;
    readonly expertId: string;
    readonly kappa: string;
    readonly title: string;
    readonly summary: string;
    readonly kind?: 'artifact' | 'space' | 'skill';
    readonly categories?: readonly string[];
    readonly pictureStyle?: string;
    readonly pictureLighting?: string;
    readonly palette?: readonly [string, string, string, string];
    readonly audioScale?: 'pentatonic' | 'major' | 'minor' | 'dorian';
    readonly audioBpm?: number;
    readonly videoAspect?: '16:9' | '1:1' | '9:16';
    readonly newlyAcquiredSkill?: boolean;
  }): LivingBasisSnapshot {
    const target = facetsQ10ForMediaType(event.mediaType);
    const nextFacets: number[] = new Array(COGNITIVE_DIM);
    for (let i = 0; i < COGNITIVE_DIM; i++) {
      nextFacets[i] = clampQ10((this.userFacetsQ10[i]! * 7 + target[i]!) >> 3);
    }
    this.userFacetsQ10 = nextFacets as unknown as CognitiveVector;

    const prevAff = this.styleAffinities.mediaAffinityQ10[event.mediaType] ?? 512;
    const nextMediaAff: Record<string, number> = {
      ...this.styleAffinities.mediaAffinityQ10,
      [event.mediaType]: clampQ10(prevAff + 48),
    };

    this.styleAffinities = {
      ...this.styleAffinities,
      ...(event.pictureStyle ? { preferredPictureStyle: event.pictureStyle } : {}),
      ...(event.pictureLighting ? { preferredPictureLighting: event.pictureLighting } : {}),
      ...(event.palette ? { preferredPalette: event.palette } : {}),
      ...(event.audioScale ? { preferredAudioScale: event.audioScale } : {}),
      ...(typeof event.audioBpm === 'number' ? { preferredAudioBpm: Math.round(event.audioBpm) } : {}),
      ...(event.videoAspect ? { preferredVideoAspect: event.videoAspect } : {}),
      mediaAffinityQ10: nextMediaAff,
    };

    if (event.newlyAcquiredSkill) {
      this.acquiredSkillIds.add(event.expertId);
    }

    this.knowledgeByKappa.set(event.kappa, {
      kappa: event.kappa,
      expertId: event.expertId,
      kind: event.kind ?? (event.mediaType === 'space' ? 'space' : 'artifact'),
      title: event.title.slice(0, 80),
      mediaType: event.mediaType,
      summary: event.summary.slice(0, 220),
      categories: (event.categories ?? []).slice(0, 8),
      updatedAt: Date.now(),
    });

    if (this.knowledgeByKappa.size > MAX_KNOWLEDGE_ENTRIES) {
      const oldest = [...this.knowledgeByKappa.values()].sort((a, b) => a.updatedAt - b.updatedAt)[0];
      if (oldest) this.knowledgeByKappa.delete(oldest.kappa);
    }

    this.adaptationSteps += 1;
    this.updatedAt = Date.now();
    this.emit();
    void this.persist();
    return this.snapshot();
  }

  /**
   * Retrieves relevant knowledge & skill entries from the non-static training basis
   * by keyword, mediaType, or category for grounding multi-turn chat responses.
   */
  recallKnowledge(query: {
    readonly text?: string;
    readonly mediaType?: PrismMediaType;
    readonly limit?: number;
  }): readonly BasisKnowledgeEntry[] {
    const q = (query.text ?? '').trim().toLowerCase();
    const tokens = q ? q.split(/\s+/).filter((t) => t.length >= 2) : [];
    const limit = Math.max(1, Math.min(12, query.limit ?? 5));

    const scored = [...this.knowledgeByKappa.values()]
      .filter((entry) => !query.mediaType || entry.mediaType === query.mediaType)
      .map((entry) => {
        if (tokens.length === 0) return { entry, score: 1 };
        const hay = `${entry.title} ${entry.summary} ${entry.categories.join(' ')} ${entry.mediaType}`.toLowerCase();
        let score = 0;
        for (const tok of tokens) {
          if (hay.includes(tok)) score += entry.title.toLowerCase().includes(tok) ? 3 : 1;
        }
        return { entry, score };
      })
      .filter((x) => x.score > 0)
      .sort((a, b) => b.score - a.score || b.entry.updatedAt - a.entry.updatedAt);

    return scored.slice(0, limit).map((x) => x.entry);
  }

  async hydrate(): Promise<void> {
    if (this.hydrated || typeof indexedDB === 'undefined') {
      this.hydrated = true;
      return;
    }
    this.hydrated = true;
    try {
      const db = await openBasisDb();
      const loaded = await new Promise<LivingBasisSnapshot | undefined>((resolve, reject) => {
        const tx = db.transaction(BASIS_IDB_STORE, 'readonly');
        const req = tx.objectStore(BASIS_IDB_STORE).get(BASIS_RECORD_KEY);
        req.onsuccess = () => resolve(req.result as LivingBasisSnapshot | undefined);
        req.onerror = () => reject(req.error);
      });
      if (loaded && loaded.version === 1) {
        this.adaptationSteps = loaded.adaptationSteps ?? 0;
        this.updatedAt = loaded.updatedAt ?? Date.now();
        if (Array.isArray(loaded.userFacetsQ10) && loaded.userFacetsQ10.length === COGNITIVE_DIM) {
          this.userFacetsQ10 = loaded.userFacetsQ10;
        }
        if (loaded.styleAffinities?.mediaAffinityQ10) {
          this.styleAffinities = loaded.styleAffinities;
        }
        this.acquiredSkillIds.clear();
        for (const id of loaded.acquiredSkillIds ?? []) this.acquiredSkillIds.add(id);
        this.knowledgeByKappa.clear();
        for (const entry of loaded.knowledgeIndex ?? []) {
          this.knowledgeByKappa.set(entry.kappa, entry);
        }
        this.emit();
      }
    } catch {
      // Fall back cleanly to in-memory basis
    }
  }

  async persist(): Promise<void> {
    if (typeof indexedDB === 'undefined') return;
    try {
      const db = await openBasisDb();
      const snap = this.snapshot();
      await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(BASIS_IDB_STORE, 'readwrite');
        tx.objectStore(BASIS_IDB_STORE).put(snap, BASIS_RECORD_KEY);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
      });
    } catch {
      // Ignore storage quota errors cleanly
    }
  }
}

function openBasisDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(BASIS_IDB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(BASIS_IDB_STORE)) {
        db.createObjectStore(BASIS_IDB_STORE);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/**
 * Single Unified Expert Registry for all parts of VerdantUI on `verdant-engine`.
 * Unifies downloaded neural weights, procedural media organs, Space Prisms, content artifacts,
 * and the living User Basis under one coherent address space and `Q10` resonance model.
 */
export class UnifiedExpertRegistry {
  private readonly experts = new Map<string, UnifiedExpertDescriptor>();
  readonly basis: LivingTrainingBasis;
  private readonly listeners = new Set<(snap: UnifiedModelTelemetrySnapshot) => void>();

  constructor(basis?: LivingTrainingBasis) {
    this.basis = basis ?? new LivingTrainingBasis();
    // Register all always-ready procedural experts on the shared verdant-engine substrate
    for (const pe of DEFAULT_PROCEDURAL_EXPERTS) {
      this.experts.set(pe.id, {
        expertId: pe.id,
        kind: 'procedural-organ',
        label: pe.label,
        mediaType: pe.mediaType,
        kappa: prismContentDigest(`verdant-expert:${pe.id}:v2`),
        facetsQ10: facetsQ10ForMediaType(pe.mediaType),
        residency: 'hot-vtensor',
        residentBytes: 0,
        activationCount: 0,
        lastActivatedAt: 0,
        tags: pe.tags,
      });
    }
    this.syncUserExpert();
  }

  subscribe(listener: (snap: UnifiedModelTelemetrySnapshot) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private emit(): void {
    const snap = this.telemetry();
    for (const fn of this.listeners) fn(snap);
  }

  private syncUserExpert(): void {
    const snap = this.basis.snapshot();
    this.experts.set('user:living-basis', {
      expertId: 'user:living-basis',
      kind: 'user-basis',
      label: 'Living User Model & Preferences',
      kappa: snap.basisDigest,
      facetsQ10: snap.userFacetsQ10,
      residency: 'hot-vtensor',
      residentBytes: 0,
      activationCount: snap.adaptationSteps,
      lastActivatedAt: snap.updatedAt,
      tags: ['user', 'preferences', 'memory'],
    });
  }

  /**
   * Registers or updates a downloaded neural weight expert inside the unified model
   * (language, vision, speech, or neural image synthesis on `VramUnifiedStore`).
   */
  registerNeuralWeightExpert(params: {
    readonly modality: 'language' | 'vision' | 'speech' | 'image-neural';
    readonly modelId: string;
    readonly label: string;
    readonly residency: ExpertResidencyTier;
    readonly residentBytes: number;
    readonly facetsQ10?: CognitiveVector;
  }): UnifiedExpertDescriptor {
    const expertId = `neural:${params.modality}:${params.modelId}`;
    const prev = this.experts.get(expertId);
    const defaultFacets: CognitiveVector =
      params.modality === 'vision'
        ? [512, 640, 384, 832, 384, 512, 640, 768]
        : params.modality === 'speech'
          ? [448, 384, 256, 896, 256, 640, 512, 640]
          : params.modality === 'image-neural'
            ? [384, 512, 384, 640, 256, 448, 832, 768]
            : [640, 832, 768, 640, 704, 832, 832, 832];
    const next: UnifiedExpertDescriptor = {
      expertId,
      kind: 'neural-weights',
      label: params.label,
      kappa: prismContentDigest(`${expertId}:${params.residentBytes}`),
      facetsQ10: params.facetsQ10 ?? defaultFacets,
      residency: params.residency,
      residentBytes: Math.max(0, Math.round(params.residentBytes)),
      activationCount: (prev?.activationCount ?? 0) + (params.residency === 'hot-vtensor' ? 1 : 0),
      lastActivatedAt: Date.now(),
      tags: [params.modality, params.modelId],
    };
    this.experts.set(expertId, next);
    this.emit();
    return next;
  }

  /** Marks an expert as activated during an inference or procedural synthesis turn. */
  recordActivation(expertId: string): void {
    const existing = this.experts.get(expertId);
    if (!existing) return;
    this.experts.set(expertId, {
      ...existing,
      activationCount: existing.activationCount + 1,
      lastActivatedAt: Date.now(),
    });
    this.emit();
  }

  /**
   * Synchronizes all Prisms (both Space Prisms and Content Artifact Prisms) into the
   * unified expert pool so the model can address every saved Space and artifact coherently.
   */
  syncPrismsAsExperts(prisms: readonly VerdantPrismProject[]): void {
    const liveIds = new Set<string>();
    for (const proj of prisms) {
      const mediaType: PrismMediaType =
        proj.declaration?.view?.proceduralOrgan?.mediaType ??
        proj.declaration?.concept.mediaType ??
        'tool-organ';
      const isSpace = mediaType === 'space';
      const expertId = isSpace ? `space:${proj.id}` : `content:${proj.id}`;
      liveIds.add(expertId);
      const prev = this.experts.get(expertId);
      const snap = proj.published ?? proj.draft;
      this.experts.set(expertId, {
        expertId,
        kind: isSpace ? 'space-prism' : 'content-artifact',
        label: snap.name,
        mediaType,
        kappa: snap.digest,
        facetsQ10: facetsQ10ForMediaType(mediaType),
        residency: proj.published ? 'warm-opfs' : 'cold-declarative',
        residentBytes: snap.app.html.length + snap.app.css.length + snap.app.js.length,
        activationCount: prev?.activationCount ?? 0,
        lastActivatedAt: prev?.lastActivatedAt ?? proj.updatedAt,
        tags: [mediaType, proj.published ? 'published' : 'draft'],
      });
    }
    for (const [id, ex] of this.experts.entries()) {
      if ((ex.kind === 'space-prism' || ex.kind === 'content-artifact') && !liveIds.has(id)) {
        this.experts.delete(id);
      }
    }
    this.syncUserExpert();
    this.emit();
  }

  /** Ranks experts by exact-integer `Q10` inner-product resonance with a target cognitive vector. */
  rankExpertsForVector(
    queryVector: CognitiveVector,
    opts?: { readonly kinds?: readonly UnifiedExpertKind[]; readonly limit?: number },
  ): readonly { readonly expert: UnifiedExpertDescriptor; readonly resonanceQ10: number }[] {
    const allow = opts?.kinds ? new Set(opts.kinds) : null;
    const limit = opts?.limit ?? 8;
    return [...this.experts.values()]
      .filter((e) => !allow || allow.has(e.kind))
      .map((expert) => ({
        expert,
        resonanceQ10: dotCognitiveQ10(queryVector, expert.facetsQ10),
      }))
      .sort((a, b) => b.resonanceQ10 - a.resonanceQ10 || b.expert.activationCount - a.expert.activationCount)
      .slice(0, limit);
  }

  listExperts(kind?: UnifiedExpertKind): readonly UnifiedExpertDescriptor[] {
    const all = [...this.experts.values()];
    return kind ? all.filter((e) => e.kind === kind) : all;
  }

  telemetry(): UnifiedModelTelemetrySnapshot {
    this.syncUserExpert();
    const all = [...this.experts.values()];
    const basisSnap = this.basis.snapshot();
    let neuralExperts = 0;
    let proceduralExperts = 0;
    let spaceExperts = 0;
    let contentExperts = 0;
    let hotResidentBytes = 0;
    let warmOpfsBytes = 0;

    for (const ex of all) {
      if (ex.kind === 'neural-weights') neuralExperts += 1;
      else if (ex.kind === 'procedural-organ') proceduralExperts += 1;
      else if (ex.kind === 'space-prism') spaceExperts += 1;
      else if (ex.kind === 'content-artifact') contentExperts += 1;
      if (ex.residency === 'hot-vtensor') hotResidentBytes += ex.residentBytes;
      else if (ex.residency === 'warm-opfs') warmOpfsBytes += ex.residentBytes;
    }

    const topActiveExperts = [...all]
      .sort((a, b) => b.activationCount - a.activationCount || b.lastActivatedAt - a.lastActivatedAt)
      .slice(0, 8)
      .map((e) => ({
        expertId: e.expertId,
        kind: e.kind,
        label: e.label,
        residency: e.residency,
        activations: e.activationCount,
      }));

    return {
      totalExperts: all.length,
      neuralExperts,
      proceduralExperts,
      spaceExperts,
      contentExperts,
      hotResidentBytes,
      warmOpfsBytes,
      adaptationSteps: basisSnap.adaptationSteps,
      acquiredSkills: basisSnap.acquiredSkillIds.length,
      knowledgeEntries: basisSnap.knowledgeIndex.length,
      basisDigest: basisSnap.basisDigest,
      topActiveExperts,
    };
  }
}
