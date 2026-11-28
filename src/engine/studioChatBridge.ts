/**
 * Chat-Native Unified Verdant Studio Orchestrator (`src/engine/studioChatBridge.ts`)
 *
 * Bridges Chat Threads directly to Verdant Studio, Global MediaType Galleries,
 * Spaces-as-Prisms, and the Unified Model's Living Training Basis (`DEC-013` & `DEC-014`):
 * - Natural-language artifact creation (Pictures, Audio, Videos, Documents, Games, Research, Utilities)
 *   synthesizes high-spec Prisms, indexes them into their Global MediaType Gallery (and optional Space),
 *   records online `Q10` basis adaptation, and renders real inline payloads (PNG, Stereo WAV, excerpts).
 * - Natural-language Space creation & curation manages Space Prisms (`mediaType: 'space'`) containing
 *   arbitrary mixed content and a live customizable Space View.
 * - Natural-language refinement, listing, gallery browsing, opening, publishing, renaming, and deleting
 *   work seamlessly in chat threads both when Local AI is Off and when Local AI is On.
 */

import {
  composeMultiTrackArrangement,
  encodeStereoWav,
  encodeStereoWavDataUrl,
  masterStereoPcm,
  synthesizeCompositionStereoPcm,
} from './audioSynthesisEngine';
import {
  observeInputSensoryState,
  senseUserQualityBudget,
  type VerdantCognitiveObserver,
} from './cognitiveObserver';
import type { ContentStreamCatalogStore } from './contentStreamCatalog';
import {
  GALLERY_LABEL_BY_TYPE,
  inferCategoriesForArtifact,
  type ContentGalleryStore,
  type GalleryMediaType,
} from './contentGalleryStore';
import type { PrismHostPool } from './prismHostPool';
import {
  PRISM_MEDIA_CATALOG,
  type AudioWeights,
  type PictureWeights,
  type PrismMediaType,
  type SpaceArtifactEntry,
  type SpaceWeights,
  type VideoWeights,
  type WrittenWorkWeights,
} from './prismMediaOrgans';
import type { PrismStore, VerdantPrismProject } from './prismStudio';
import {
  assembleProceduralPrismOps,
  buildWeightRefinementPrompt,
  classifyArtifactMediaType,
  extractCleanIdeaTitle,
  parseDiscoveryAgentResponse,
} from './prismStudioAgent';
import {
  encodeCompactThumbnailDataUrl,
  inferPictureWeightsFromPrompt,
  renderPictureWithCoherenceCheck,
} from './proceduralPictureRenderer';
import {
  contextualizePromptToUnifiedWeights,
  VERDANT_LOD_LADDER,
} from './prismReconciliationLoop';
import type {
  SemanticRevisionRecord,
  UnifiedExpertRegistry,
} from './unifiedModelBasis';
import type { PrismLodRank } from './verdantStudioApi';
import type {
  SemanticVisualQaReport,
  SemanticVisualQaTarget,
} from './visionCritiqueLoop';
export { detectStudioChatIntent, type DetectedStudioChatIntent } from './studioChatIntent';

export interface StudioChatArtifactResult {
  readonly __prismInvocation: true;
  readonly action: 'create' | 'update' | 'invoke' | 'list' | 'gallery' | 'open' | 'publish' | 'delete' | 'space';
  readonly autoCreated: boolean;
  readonly prismId: string;
  readonly prismName: string;
  readonly mediaType: PrismMediaType;
  readonly galleryLabel?: string;
  readonly categories?: readonly string[];
  readonly spaceId?: string;
  readonly spaceName?: string;
  readonly digest: string;
  readonly summary: string;
  readonly previewImageDataUrl?: string;
  readonly previewImageWidth?: number;
  readonly previewImageHeight?: number;
  readonly audioWavDataUrl?: string;
  readonly audioDurationSec?: number;
  readonly writtenExcerpt?: string;
  readonly coherenceNote?: string;
  readonly prismsList?: readonly {
    readonly id: string;
    readonly name: string;
    readonly mediaType: PrismMediaType;
    readonly published: boolean;
    readonly summary: string;
    readonly previewDataUrl?: string;
  }[];
  readonly output: unknown;
}

export interface StudioChatBridgeDeps {
  readonly store: PrismStore;
  readonly pool: PrismHostPool;
  readonly observer: VerdantCognitiveObserver;
  readonly galleries: ContentGalleryStore;
  readonly contentStreams?: ContentStreamCatalogStore;
  readonly registry: UnifiedExpertRegistry;
  readonly runLocalStudioPrompt?: (
    systemPrompt: string,
    userPrompt: string,
    maxTokens: number,
    partId?: string,
  ) => Promise<string>;
  readonly synthesizeAudioWav?: (params: {
    readonly title: string;
    readonly style: AudioWeights['style'];
    readonly bpm: number;
    readonly rootMidi: number;
    readonly scale: AudioWeights['scale'];
    readonly durationSec: number;
  }) => Promise<{ readonly wavBytes: Uint8Array; readonly durationSec: number }>;
  readonly synthesizeImageLayer?: (params: {
    readonly prompt: string;
    readonly seed: number;
    readonly latentH?: number;
    readonly latentW?: number;
  }) => Promise<{
    readonly neuralAvailable: boolean;
    readonly width: number;
    readonly height: number;
    readonly rgba: Uint8ClampedArray | null;
  }>;
  readonly evaluateVisualQa?: (params: {
    readonly width: number;
    readonly height: number;
    readonly rgba: Uint8Array | Uint8ClampedArray;
    readonly target: SemanticVisualQaTarget;
    readonly minAcceptScoreQ10?: number;
  }) => Promise<SemanticVisualQaReport>;
  readonly runWithDeterministicPrecision?: <T>(fn: () => Promise<T>) => Promise<T>;
  readonly scheduleProgressiveReconciliation?: (job: {
    readonly prismId: string;
    readonly prompt: string;
    readonly initialRevision: SemanticRevisionRecord;
    readonly startRank?: PrismLodRank;
    readonly maxRank?: PrismLodRank;
  }) => void;
  readonly onOpenPrism?: (prismId: string) => void;
  readonly onOpenGallery?: (mediaType: GalleryMediaType) => void;
}

function normKey(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, '');
}

/** Extracts feature keys explicitly stated by the user so iterative tuning preserves user pins. */
export function extractPinnedFieldsFromPrompt(prompt: string): readonly string[] {
  const lower = prompt.toLowerCase();
  const pins = new Set<string>();
  if (/\b\d{2,3}\s*bpm\b/.test(lower)) pins.add('bpm');
  if (/\b(pentatonic|major|minor|dorian)\b/.test(lower)) pins.add('scale');
  if (/\b(golden\s*hour|twilight|midnight|daylight|neon|noir|sunset|sunrise|dark|bright)\b/.test(lower)) {
    pins.add('lighting');
  }
  if (/\b(emerald|amber|crimson|cobalt|sapphire|violet|magenta|monochrome|terracotta|gold|pastel)\b/.test(lower)) {
    pins.add('palette');
  }
  if (/\b(16:9|4:3|1:1|9:16|square|portrait|widescreen)\b/.test(lower)) pins.add('aspect');
  if (/\b\d{1,2}\s*[x×]\s*\d{1,2}\b/.test(lower)) pins.add('dimensions');
  return Array.from(pins);
}

type CompanionFacetRequest = 'cover-art' | 'logo' | 'soundtrack';

function detectRequestedCompanions(prompt: string, primaryMediaType: PrismMediaType): readonly CompanionFacetRequest[] {
  if (primaryMediaType === 'space') return [];
  const lower = prompt.toLowerCase();
  const out: CompanionFacetRequest[] = [];
  if (primaryMediaType !== 'picture' && /\b(cover\s+art|book\s+cover|album\s+cover|box\s+art)\b/.test(lower)) {
    out.push('cover-art');
  }
  if (primaryMediaType !== 'picture' && /\b(logo|brand\s*mark|emblem)\b/.test(lower)) {
    out.push('logo');
  }
  if (primaryMediaType !== 'audio' && /\b(soundtrack|theme\s+song|background\s+music|ambient\s+music)\b/.test(lower)) {
    out.push('soundtrack');
  }
  return out;
}

export function findPrismByNameOrId(
  query: string | undefined,
  prisms: readonly VerdantPrismProject[],
  opts?: { readonly mediaType?: PrismMediaType },
): VerdantPrismProject | null {
  const pool = opts?.mediaType
    ? prisms.filter(
        (p) =>
          (p.declaration?.view?.proceduralOrgan?.mediaType ?? p.declaration?.concept.mediaType) ===
          opts.mediaType,
      )
    : prisms;
  if (pool.length === 0) return null;
  if (!query || !query.trim()) return pool[0] ?? null;
  const qLower = query.trim().toLowerCase();
  const qNorm = normKey(query);

  for (const p of pool) {
    if (p.id === query.trim()) return p;
    const name = (p.published?.name ?? p.draft.name).toLowerCase();
    if (name === qLower || normKey(name) === qNorm) return p;
  }
  for (const p of pool) {
    const name = (p.published?.name ?? p.draft.name).toLowerCase();
    if (qNorm.length >= 3 && (normKey(name).includes(qNorm) || qNorm.includes(normKey(name)))) {
      return p;
    }
  }
  return null;
}

/**
 * Rebuilds a Space Prism's declarative `SpaceWeights.items` from the live Global Content Index
 * so its Space View always displays up-to-date titles, summaries, and preview thumbnails.
 */
export async function syncSpacePrismView(
  spacePrismId: string,
  deps: Pick<StudioChatBridgeDeps, 'store' | 'galleries' | 'observer' | 'registry'>,
): Promise<VerdantPrismProject | null> {
  const proj = deps.store.get(spacePrismId);
  if (!proj?.declaration) return null;
  const curW = proj.declaration.view?.proceduralOrgan?.weights;
  const existingSpace: SpaceWeights =
    curW && curW.mediaType === 'space'
      ? curW
      : {
          mediaType: 'space',
          title: proj.draft.name,
          description: proj.declaration.concept.purpose || proj.draft.summary,
          layout: 'grid',
          accent: '#0f5132',
          items: [],
        };

  const linked = deps.galleries.listForSpace(spacePrismId);
  const items: SpaceArtifactEntry[] = linked.map((rec) => ({
    id: rec.id,
    title: rec.title,
    mediaType: rec.mediaType,
    summary: rec.summary,
    categories: rec.categories,
    previewDataUrl: rec.previewDataUrl,
  }));

  const primaryReq = proj.declaration.requirements[0]?.id ?? 'REQ-01';
  const nextWeights: SpaceWeights = { ...existingSpace, items };
  await deps.store.applyStudioOp(
    spacePrismId,
    {
      op: 'route-prism-weights',
      mediaType: 'space',
      activeOrgans: proj.declaration.view?.proceduralOrgan?.activeOrgans ?? ['palette-theme', 'narrative-outline', 'evidence-ledger'],
      weights: nextWeights,
      title: existingSpace.title,
      eyebrow: 'Space',
      subtitle: existingSpace.description,
      summary: `${items.length} curated item${items.length === 1 ? '' : 's'}`,
      citation: { kind: 'req', requirementId: primaryReq },
    },
    { pushRevision: false },
  );
  const pub = await deps.store.publish(spacePrismId);
  deps.observer.syncPublishedPrisms(deps.store.list());
  deps.registry.syncPrismsAsExperts(deps.store.list());
  return pub;
}

/**
 * Generates real rendered inline media artifacts (PNG for Pictures / Video poster, Stereo WAV for Audio,
 * formatted excerpt for Documents) directly from a synthesized Prism's declarative weights.
 */
async function materializeInlineArtifact(
  proj: VerdantPrismProject,
  deps: StudioChatBridgeDeps,
): Promise<{
  readonly previewImageDataUrl?: string;
  readonly thumbnailDataUrl?: string;
  readonly previewImageWidth?: number;
  readonly previewImageHeight?: number;
  readonly audioWavDataUrl?: string;
  readonly audioDurationSec?: number;
  readonly writtenExcerpt?: string;
  readonly coherenceNote?: string;
}> {
  const weights = proj.declaration?.view?.proceduralOrgan?.weights;
  if (!weights) return {};

  if (weights.mediaType === 'picture') {
    const pw = weights as PictureWeights;
    const residency = deps.observer.getActiveModalityResidency();
    let neuralLayer: { readonly width: number; readonly height: number; readonly rgba: Uint8ClampedArray } | null = null;
    if (deps.synthesizeImageLayer && (residency.image || !residency.language)) {
      try {
        const res = await deps.synthesizeImageLayer({ prompt: pw.caption || proj.draft.name, seed: pw.seed });
        if (res.neuralAvailable && res.rgba && res.width > 0 && res.height > 0) {
          neuralLayer = { width: res.width, height: res.height, rgba: res.rgba };
        }
      } catch {
        neuralLayer = null;
      }
    }
    const rendered = await renderPictureWithCoherenceCheck(pw, {
      maxPreviewDim: 640,
      neuralLayer,
    });
    return {
      previewImageDataUrl: rendered.pngDataUrl,
      thumbnailDataUrl: encodeCompactThumbnailDataUrl(rendered.rgba, rendered.width, rendered.height, 256),
      previewImageWidth: rendered.width,
      previewImageHeight: rendered.height,
      coherenceNote: rendered.coherence.coherent
        ? `${rendered.width}×${rendered.height} · Visual check passed`
        : `${rendered.width}×${rendered.height}`,
    };
  }

  if (weights.mediaType === 'audio') {
    const aw = weights as AudioWeights;
    try {
      if (deps.synthesizeAudioWav) {
        const res = await deps.synthesizeAudioWav({
          title: aw.title,
          style: aw.style,
          bpm: aw.bpm,
          rootMidi: aw.rootMidi,
          scale: aw.scale,
          durationSec: Math.min(10, aw.durationSec),
        });
        return {
          audioWavDataUrl: encodeStereoWavDataUrl(res.wavBytes),
          audioDurationSec: res.durationSec,
        };
      }
    } catch {
      // Fall back to direct main-thread modal synthesis
    }
    const comp = composeMultiTrackArrangement({
      title: aw.title,
      style: aw.style,
      bpm: aw.bpm,
      rootMidi: aw.rootMidi,
      scale: aw.scale,
      durationSec: Math.min(10, aw.durationSec),
    });
    const rawPcm = synthesizeCompositionStereoPcm(comp, 22_050);
    const pcm = masterStereoPcm(rawPcm.left, rawPcm.right, rawPcm.sampleRate);
    const wav = encodeStereoWav(pcm.left, pcm.right, pcm.sampleRate);
    return {
      audioWavDataUrl: encodeStereoWavDataUrl(wav),
      audioDurationSec: comp.durationSec,
    };
  }

  if (weights.mediaType === 'video') {
    const vw = weights as VideoWeights;
    const posterWeights: PictureWeights = {
      mediaType: 'picture',
      generator: 'abstract-flow',
      lighting: 'twilight',
      width: vw.aspectRatio === '1:1' ? 512 : vw.aspectRatio === '9:16' ? 360 : 640,
      height: vw.aspectRatio === '1:1' ? 512 : vw.aspectRatio === '9:16' ? 640 : 360,
      seed: (proj.createdAt % 900_000) + 1000,
      symmetry: 6,
      density: 54,
      palette: [vw.palette[0], vw.palette[1], vw.scenes[0]?.accent ?? '#f59e0b', vw.palette[2]],
      caption: vw.scenes[0]?.title ?? proj.draft.name,
    };
    const poster = await renderPictureWithCoherenceCheck(posterWeights, { maxPreviewDim: 512 });
    return {
      previewImageDataUrl: poster.pngDataUrl,
      thumbnailDataUrl: encodeCompactThumbnailDataUrl(poster.rgba, poster.width, poster.height, 256),
      previewImageWidth: poster.width,
      previewImageHeight: poster.height,
    };
  }

  if (weights.mediaType === 'written-work') {
    const ww = weights as WrittenWorkWeights;
    const lines = [
      `**${ww.title}** _(${ww.genre})_`,
      ww.premise,
      ...ww.chapters.slice(0, 4).map((c) => `- **${c.title}**: ${c.summary}`),
    ];
    return { writtenExcerpt: lines.join('\n') };
  }

  return {};
}

/**
 * Full unified Studio Chat Bridge instance used by `createVerdantClientTools` and `prismToolResolver`.
 */
export class StudioChatBridge {
  private lastTouchedPrismId: string | null = null;

  constructor(private readonly deps: StudioChatBridgeDeps) {}

  async createArtifactFromPrompt(params: {
    readonly prompt: string;
    readonly title?: string;
    readonly forcedMediaType?: PrismMediaType;
    readonly weightOverrides?: Record<string, unknown>;
    readonly spaceName?: string;
    readonly spaceId?: string;
  }): Promise<StudioChatArtifactResult> {
    const { store, observer, galleries, registry } = this.deps;
    const promptText = params.prompt.trim();
    const classified = classifyArtifactMediaType(promptText);
    const mediaType: PrismMediaType = params.forcedMediaType ?? classified.mediaType;
    const title = params.title?.trim() || extractCleanIdeaTitle(promptText);

    const created = await store.createDeclarative('undefined');
    const baseDecl = created.declaration!;
    const affinities = registry.basis.getStyleAffinities();
    const studioObs = observeInputSensoryState([], promptText, false);
    const studioSnap = observer.getThreadSnapshot('studio');
    const qualityBudget = senseUserQualityBudget({
      latestUserText: promptText,
      immediateObs: studioObs.vector,
      integratedBelief: studioSnap.belief,
      surpriseQ10: studioSnap.lastSurpriseQ10,
      retryEscalationLevel: studioSnap.retryEscalationLevel,
      globalUserFacetsQ10: registry.basis.getUserFacetsQ10(),
      hasEnabledContentStreams: Boolean(this.deps.contentStreams?.hasEnabledStreams()),
    });

    // Optional Zero-Cutoff Realtime Content Stream grounding (strictly opt-in user streams)
    let liveStreamEvidence: Array<{
      readonly hypothesisId: string;
      readonly verdict: 'supporting' | 'refuting' | 'observation';
      readonly text: string;
      readonly date: string;
    }> = [];
    let liveContextSummary = '';
    if (
      mediaType !== 'space' &&
      qualityBudget.shouldConsultContentStreams &&
      this.deps.contentStreams?.hasEnabledStreams() &&
      observer.grants.isAllowed('network-weights')
    ) {
      try {
        const { streamRealtimeWeightsToPrism } = await import('./realtimeWeightStreamer');
        const streamed = await streamRealtimeWeightsToPrism({
          query: promptText,
          catalogStore: this.deps.contentStreams,
          registry,
          maxStreams: qualityBudget.featureRichness === 'full' ? 3 : 2,
        });
        if (streamed.chunks.length > 0) {
          const today = new Date().toISOString().slice(0, 10);
          liveStreamEvidence = streamed.chunks.slice(0, 4).map((c) => ({
            hypothesisId: 'H1',
            verdict: 'observation' as const,
            text: `[${c.provider}] ${c.title}: ${c.canonicalText.slice(0, 220)} (${c.sourceUrl})`,
            date: today,
          }));
          liveContextSummary = streamed.chunks
            .slice(0, 2)
            .map((c) => `${c.title}: ${c.canonicalText.slice(0, 140)}`)
            .join(' · ');
        }
      } catch {
        // Offline or stream unreachable; proceed seamlessly with local basis
      }
    }

    // Stage 1: Instant deterministic cross-modal contextualization into Unified Semantic Associator
    // (delta_rule.rs) so LOD0 returns in < 500 ms without blocking chat or wiping the chat KV cache!
    const { revision: semanticRev, inheritedAnchor } = await contextualizePromptToUnifiedWeights({
      prompt: promptText,
      mediaType,
      prismId: created.id,
      activePrismId: this.lastTouchedPrismId ?? undefined,
      deps: {
        registry,
      },
    });

    // Seed picture weights using semantic prompt inference + associated cross-modal anchor + learned basis affinities
    const baseWeightsSeed: Record<string, unknown> =
      mediaType === 'picture'
        ? {
            ...inferPictureWeightsFromPrompt(promptText, title, affinities),
            ...(inheritedAnchor && semanticRev.palette ? { palette: semanticRev.palette } : {}),
            ...(inheritedAnchor && semanticRev.lighting ? { lighting: semanticRev.lighting } : {}),
            ...(params.weightOverrides ?? {}),
          }
        : mediaType === 'audio'
          ? {
              ...(!/\b(pentatonic|major|minor|dorian|lydian|phrygian|mixolydian|dark|nocturnal|melancholy|bright|happy|upbeat|jazz|dreamy)\b/i.test(
                promptText,
              ) && affinities.preferredAudioScale
                ? { scale: affinities.preferredAudioScale }
                : {}),
              ...(!/\b\d{2,3}\s*bpm\b|\b(fast|upbeat|energetic|synthwave|pulse|techno|calm|slow|ambient|chill|meditation|sleep|lofi)\b/i.test(
                promptText,
              ) && affinities.preferredAudioBpm
                ? { bpm: affinities.preferredAudioBpm }
                : {}),
              ...(params.weightOverrides ?? {}),
            }
          : mediaType === 'video'
            ? {
                ...(inheritedAnchor && semanticRev.palette
                  ? { palette: [semanticRev.palette[0], semanticRev.palette[1], semanticRev.palette[2]] }
                  : {}),
                ...(params.weightOverrides ?? {}),
              }
          : mediaType === 'research-program' && liveStreamEvidence.length > 0
            ? {
                evidence: liveStreamEvidence,
                ...(params.weightOverrides ?? {}),
              }
            : { ...(params.weightOverrides ?? {}) };

    // Synthesize LOD0 immediately from semantic prompt inference + living basis affinities
    // (never block LOD0 creation on a 512-token LLM roundtrip or wipe the active chat KV cache!).
    const seedJson = JSON.stringify({
      name: title,
      purpose:
        liveContextSummary
          ? `${semanticRev.contextualBrief || promptText.slice(0, 140)} (${liveContextSummary.slice(0, 140)})`
          : semanticRev.contextualBrief || promptText.slice(0, 220),
      mediaType,
      readyToBuild: true,
      weights: { mediaType, ...baseWeightsSeed },
      reply: `Created ${title}.`,
    });

    const { spec } = parseDiscoveryAgentResponse(seedJson, promptText, baseDecl, 2, mediaType);
    const mergedSpec = {
      ...spec,
      name: params.title?.trim() || spec.name || title,
      readyToBuild: true,
    };

    const assembled = assembleProceduralPrismOps(baseDecl, mergedSpec, promptText);
    const initialPins = extractPinnedFieldsFromPrompt(promptText);
    const initialLod = VERDANT_LOD_LADDER[0];
    await store.applyStudioOp(
      created.id,
      {
        op: 'batch',
        ops: [
          ...assembled.ops,
          {
            op: 'set-concept-profile',
            seedText: promptText,
            pinnedFields: initialPins,
            semanticConceptId: semanticRev.conceptId,
            semanticKappa: semanticRev.kappa,
            semanticVersion: semanticRev.version,
            lodState: {
              rank: initialLod.rank,
              label: initialLod.label,
              qaScoreQ10: 760,
              qaPasses: 0,
              qaSummary: initialLod.userStatusLabel,
              updatedAt: Date.now(),
            },
          },
        ],
      },
      { pushRevision: false },
    );
    const published = await store.publish(created.id);
    const activeProj = published ?? store.get(created.id)!;
    this.lastTouchedPrismId = activeProj.id;

    // Sync into unified possession model, expert registry & global gallery index
    const allPrisms = store.list();
    observer.syncPublishedPrisms(allPrisms);
    registry.syncPrismsAsExperts(allPrisms);
    registry.recordActivation(`procedural:${mediaType === 'written-work' ? 'document' : mediaType === 'interactive-game' ? 'game' : mediaType === 'research-program' ? 'research' : mediaType === 'tool-organ' ? 'utility' : mediaType}`);
    galleries.syncFromPrisms(allPrisms);

    // Materialize real inline PNG / Stereo WAV / Written excerpt
    const rendered = await materializeInlineArtifact(activeProj, this.deps);
    const weightsObj = activeProj.declaration?.view?.proceduralOrgan?.weights as unknown as Record<string, unknown> | undefined;
    const categories =
      mediaType !== 'space'
        ? inferCategoriesForArtifact(mediaType, activeProj.draft.name, activeProj.draft.summary, weightsObj)
        : ['Spaces'];

    // Optional Space attachment (by spaceId or spaceName)
    let resolvedSpaceId = params.spaceId;
    let resolvedSpaceName: string | undefined;
    if (mediaType !== 'space') {
      if (!resolvedSpaceId && params.spaceName) {
        const foundSpace = findPrismByNameOrId(params.spaceName, allPrisms, { mediaType: 'space' });
        if (foundSpace) {
          resolvedSpaceId = foundSpace.id;
          resolvedSpaceName = foundSpace.draft.name;
        }
      } else if (resolvedSpaceId) {
        resolvedSpaceName = store.get(resolvedSpaceId)?.draft.name;
      }

      galleries.setArtifactMeta(activeProj.id, {
        categories,
        addSpaceId: resolvedSpaceId,
        previewDataUrl: rendered.thumbnailDataUrl ?? rendered.previewImageDataUrl,
        audioDurationSec: rendered.audioDurationSec,
      });

      if (resolvedSpaceId) {
        await syncSpacePrismView(resolvedSpaceId, this.deps);
      }
    }

    // Record online adaptation step in the non-static Living Training Basis
    const picW = weightsObj?.mediaType === 'picture' ? (weightsObj as unknown as PictureWeights) : null;
    const audW = weightsObj?.mediaType === 'audio' ? (weightsObj as unknown as AudioWeights) : null;
    registry.basis.recordAdaptation({
      mediaType,
      expertId: mediaType === 'space' ? `space:${activeProj.id}` : `content:${activeProj.id}`,
      kappa: (activeProj.published ?? activeProj.draft).digest,
      title: activeProj.draft.name,
      summary: activeProj.draft.summary || promptText,
      kind: mediaType === 'space' ? 'space' : 'skill',
      categories,
      pictureStyle: picW?.generator,
      pictureLighting: picW?.lighting,
      palette: picW?.palette,
      audioScale: audW?.scale,
      audioBpm: audW?.bpm,
      newlyAcquiredSkill: true,
    });

    // Generate any explicitly requested companion facets coherently from this Prism's features
    const companionIds: string[] = [];
    if (!params.weightOverrides?.__isCompanion) {
      const requestedCompanions = detectRequestedCompanions(promptText, mediaType);
      for (const cKind of requestedCompanions) {
        const compRes = await this.generateCompanionFromPrism(activeProj, cKind, {
          spaceId: resolvedSpaceId,
        });
        if (compRes) companionIds.push(compRes.prismId);
      }
      if (companionIds.length > 0) {
        await store.applyStudioOp(
          activeProj.id,
          { op: 'set-concept-profile', linkedPrismIds: companionIds },
          { pushRevision: false },
        );
        await store.publish(activeProj.id);
        this.lastTouchedPrismId = activeProj.id;
      }
    }

    // Schedule non-interruptible background LOD reconciliation scaled to Observer's sensed user quality budget
    if (mediaType !== 'space' && this.deps.scheduleProgressiveReconciliation) {
      const targetMaxRank = Math.max(1, qualityBudget.targetLodRank) as PrismLodRank;
      this.deps.scheduleProgressiveReconciliation({
        prismId: activeProj.id,
        prompt: promptText,
        initialRevision: semanticRev,
        startRank: 1,
        maxRank: targetMaxRank,
      });
    }

    return {
      __prismInvocation: true,
      action: mediaType === 'space' ? 'space' : 'create',
      autoCreated: true,
      prismId: activeProj.id,
      prismName: activeProj.published?.name ?? activeProj.draft.name,
      mediaType,
      galleryLabel: mediaType === 'space' ? 'Spaces' : GALLERY_LABEL_BY_TYPE[mediaType],
      categories,
      spaceId: resolvedSpaceId,
      spaceName: resolvedSpaceName,
      digest: (activeProj.published ?? activeProj.draft).digest,
      summary:
        companionIds.length > 0
          ? `${activeProj.draft.summary || promptText.slice(0, 140)} (+${companionIds.length} companion ${companionIds.length === 1 ? 'asset' : 'assets'} saved to Galleries)`
          : activeProj.draft.summary || promptText.slice(0, 160),
      ...rendered,
      output: {
        status: 'synthesized',
        mediaType,
        gallery: mediaType === 'space' ? 'Spaces' : GALLERY_LABEL_BY_TYPE[mediaType],
        categories,
        ...(companionIds.length > 0 ? { linkedPrismIds: companionIds } : {}),
        ...(resolvedSpaceName ? { space: resolvedSpaceName } : {}),
      },
    };
  }

  /**
   * Derives a coherent companion Prism (`cover-art`, `logo`, or `soundtrack`) directly from
   * an existing Prism's features (title, purpose, palette, mood, theme) and links them cleanly.
   */
  async generateCompanionFromPrism(
    parent: VerdantPrismProject,
    kind: CompanionFacetRequest,
    opts?: { readonly spaceId?: string },
  ): Promise<StudioChatArtifactResult> {
    const parentName = parent.published?.name ?? parent.draft.name;
    const parentPurpose = parent.declaration?.concept.purpose || parent.draft.summary || parentName;
    const parentMedia =
      parent.declaration?.view?.proceduralOrgan?.mediaType ??
      parent.declaration?.concept.mediaType ??
      'tool-organ';
    const parentW = parent.declaration?.view?.proceduralOrgan?.weights as unknown as Record<string, unknown> | undefined;
    const inheritedPalette = Array.isArray(parentW?.palette) ? parentW.palette : undefined;
    const inheritedLighting = typeof parentW?.lighting === 'string' ? parentW.lighting : undefined;

    if (kind === 'soundtrack') {
      const prompt = `Compose a coherent soundtrack for "${parentName}": ${parentPurpose}`;
      return this.createArtifactFromPrompt({
        prompt,
        title: `${parentName} Soundtrack`,
        forcedMediaType: 'audio',
        spaceId: opts?.spaceId,
        weightOverrides: { __isCompanion: true, title: `${parentName} Soundtrack` },
      });
    }

    const isLogo = kind === 'logo';
    const coverFormat =
      parentMedia === 'written-work' ? 'book' : parentMedia === 'audio' ? 'album' : 'game';
    const prompt = isLogo
      ? `Create a geometric logo for "${parentName}": ${parentPurpose}`
      : `Create ${coverFormat} cover art for "${parentName}": ${parentPurpose}`;
    return this.createArtifactFromPrompt({
      prompt,
      title: isLogo ? `${parentName} Logo` : `${parentName} Cover Art`,
      forcedMediaType: 'picture',
      spaceId: opts?.spaceId,
      weightOverrides: {
        __isCompanion: true,
        generator: isLogo ? 'logo-mark' : 'cover-art',
        caption: parentName,
        subtitle: parentPurpose.slice(0, 64),
        ...(isLogo ? {} : { coverFormat }),
        ...(inheritedPalette ? { palette: inheritedPalette } : {}),
        ...(inheritedLighting ? { lighting: inheritedLighting } : {}),
      },
    });
  }

  async refinePrismFromPrompt(params: {
    readonly target?: string;
    readonly instruction: string;
  }): Promise<StudioChatArtifactResult> {
    const { store, observer, galleries, registry, pool } = this.deps;
    const allPrisms = store.list();
    const proj =
      params.target && params.target.toLowerCase() !== 'latest'
        ? findPrismByNameOrId(params.target, allPrisms)
        : (this.lastTouchedPrismId ? store.get(this.lastTouchedPrismId) : null) ?? allPrisms[0] ?? null;

    if (!proj || !proj.declaration) {
      return this.createArtifactFromPrompt({ prompt: params.instruction });
    }

    const decl = proj.declaration;
    const mediaType: PrismMediaType =
      decl.view?.proceduralOrgan?.mediaType ?? decl.concept.mediaType ?? 'tool-organ';

    // Handle feature-derived companion creation ("generate cover art", "generate a logo", "generate a soundtrack")
    const compMatch = params.instruction.match(
      /\b(?:generate|create|make|compose|design|add)\s+(?:a\s+|an\s+)?(cover\s+art|book\s+cover|album\s+cover|game\s+cover|logo|brand\s*mark|soundtrack|theme\s+song)\b/i,
    );
    if (compMatch) {
      const token = compMatch[1]!.toLowerCase();
      const kind: CompanionFacetRequest = /soundtrack|song/.test(token)
        ? 'soundtrack'
        : /logo|brand/.test(token)
          ? 'logo'
          : 'cover-art';
      const createdCompanion = await this.generateCompanionFromPrism(proj, kind);
      const nextLinked = Array.from(
        new Set([...(decl.concept.linkedPrismIds ?? []), createdCompanion.prismId]),
      );
      await store.applyStudioOp(
        proj.id,
        { op: 'set-concept-profile', linkedPrismIds: nextLinked },
        { pushRevision: false },
      );
      await store.publish(proj.id);
      return createdCompanion;
    }

    // Advance UnifiedSemanticAssociator (vN -> vN+1, parentKappa-chained) so refinements
    // propagate across adjacent modalities and trigger progressive Lod1->Lod3 reconciliation.
    const { revision: refinedSemanticRev } = await contextualizePromptToUnifiedWeights({
      prompt: `${proj.draft.name}: ${params.instruction}`,
      mediaType,
      prismId: proj.id,
      activePrismId: proj.id,
      deps: {
        registry,
        runLocalLanguagePrompt: this.deps.runLocalStudioPrompt
          ? (sys, usr, max) => this.deps.runLocalStudioPrompt!(sys, usr, max, proj.id)
          : undefined,
      },
    });

    let rawModelJson = '';
    if (this.deps.runLocalStudioPrompt) {
      try {
        const prompt = buildWeightRefinementPrompt([], decl, params.instruction);
        rawModelJson = await this.deps.runLocalStudioPrompt(prompt, params.instruction, 512, proj.id);
      } catch {
        rawModelJson = '';
      }
    }

    // Apply deterministic semantic overrides while preserving previously user-pinned fields
    const existingPins = new Set(decl.concept.pinnedFields ?? []);
    const newPins = extractPinnedFieldsFromPrompt(params.instruction);
    const mergedPins = Array.from(new Set([...existingPins, ...newPins]));

    const curW = decl.view?.proceduralOrgan?.weights as unknown as Record<string, unknown> | undefined;
    let patchWeights: Record<string, unknown> = { ...(curW ?? {}) };
    if (mediaType === 'picture') {
      const inferred = inferPictureWeightsFromPrompt(params.instruction, proj.draft.name);
      const pinSet = new Set(newPins);
      patchWeights = {
        ...patchWeights,
        ...((!existingPins.has('lighting') || pinSet.has('lighting')) ? { lighting: inferred.lighting } : {}),
        ...((!existingPins.has('palette') || pinSet.has('palette')) ? { palette: inferred.palette } : {}),
        density: inferred.density,
        seed: ((Number(patchWeights.seed || 4219) * 1664525 + 1013904223) >>> 0) % 990_000 + 1000,
        ...(/\b(landscape|nebula|cosmos|architecture|botanical|ocean|abstract|mosaic|constellation|poster|pixel|logo|cover\s+art)\b/i.test(params.instruction)
          ? { generator: inferred.generator }
          : {}),
      };
    } else if (mediaType === 'audio') {
      const bpmMatch = params.instruction.match(/\b(\d{2,3})\s*bpm\b/i);
      if (bpmMatch) {
        patchWeights.bpm = Number(bpmMatch[1]);
      } else if (!existingPins.has('bpm')) {
        if (/\b(faster|energetic|upbeat)\b/i.test(params.instruction)) {
          patchWeights.bpm = Math.min(170, Number(patchWeights.bpm || 96) + 18);
        } else if (/\b(slower|calmer|ambient|nocturnal)\b/i.test(params.instruction)) {
          patchWeights.bpm = Math.max(60, Number(patchWeights.bpm || 96) - 16);
        }
      }
      const scMatch = params.instruction.match(/\b(pentatonic|major|minor|dorian)\b/i);
      if (scMatch) {
        patchWeights.scale = scMatch[1]!.toLowerCase();
      } else if (!existingPins.has('scale') && /\b(dark|darker|nocturnal|moody|twilight)\b/i.test(params.instruction)) {
        patchWeights.scale = 'minor';
      }
    } else if (mediaType === 'video') {
      const existingScenes = Array.isArray(patchWeights.scenes)
        ? (patchWeights.scenes as Array<Record<string, unknown>>).map((s) => ({ ...s }))
        : [];
      const addSceneMatch = params.instruction.match(/\badd\s+scene\b[:\s-]*(.+)$/i);
      if (addSceneMatch) {
        const title = addSceneMatch[1]!.trim().slice(0, 48) || `Scene ${existingScenes.length + 1}`;
        const palette = Array.isArray(patchWeights.palette) ? (patchWeights.palette as string[]) : ['#0f172a', '#1e293b', '#34d399'];
        existingScenes.push({
          title,
          subtitle: 'Added via natural language',
          durationSec: 4,
          accent: palette[2] ?? '#34d399',
          motion: 'rise',
        });
        patchWeights.scenes = existingScenes;
      }
      if (!existingPins.has('palette')) {
        if (/\b(midnight|dark|nocturnal)\b/i.test(params.instruction)) {
          patchWeights.palette = ['#020617', '#0f172a', '#60a5fa'];
        } else if (/\b(golden|warm|sunset)\b/i.test(params.instruction)) {
          patchWeights.palette = ['#1c1917', '#451a03', '#f59e0b'];
        } else if (/\b(emerald|forest|verdant)\b/i.test(params.instruction)) {
          patchWeights.palette = ['#022c22', '#064e3b', '#34d399'];
        }
      }
      if (/\b(slower|calm|relaxed)\b/i.test(params.instruction) && existingScenes.length > 0) {
        patchWeights.scenes = existingScenes.map((s) => ({
          ...s,
          durationSec: Math.min(10, Number(s.durationSec || 4) + 1.5),
        }));
      } else if (/\b(faster|energetic|snappy)\b/i.test(params.instruction) && existingScenes.length > 0) {
        patchWeights.scenes = existingScenes.map((s) => ({
          ...s,
          durationSec: Math.max(2, Number(s.durationSec || 4) - 1),
        }));
      }
    } else if (mediaType === 'research-program') {
      const hyps = Array.isArray(patchWeights.hypotheses)
        ? (patchWeights.hypotheses as Array<Record<string, unknown>>).map((h) => ({ ...h }))
        : [];
      const miles = Array.isArray(patchWeights.milestones)
        ? (patchWeights.milestones as Array<Record<string, unknown>>).map((m) => ({ ...m }))
        : [];
      const evs = Array.isArray(patchWeights.evidence)
        ? (patchWeights.evidence as Array<Record<string, unknown>>).map((e) => ({ ...e }))
        : [];
      const addH = params.instruction.match(/\badd\s+hypothesis\b(?:\s+that)?[:\s-]*(.+)$/i);
      if (addH) {
        hyps.push({ id: `H${hyps.length + 1}`, statement: addH[1]!.trim().slice(0, 220), status: 'open' });
        patchWeights.hypotheses = hyps;
      }
      const markH = params.instruction.match(/\b(?:mark|set)\s+(h\d+)\s+(?:as\s+)?(open|supported|refuted)\b/i);
      if (markH) {
        const hid = markH[1]!.toUpperCase();
        const st = markH[2]!.toLowerCase();
        patchWeights.hypotheses = hyps.map((h) => (String(h.id).toUpperCase() === hid ? { ...h, status: st } : h));
      }
      const addEv = params.instruction.match(/\b(?:log|add)\s+(?:evidence|finding|observation)\b[:\s-]*(.+)$/i);
      if (addEv) {
        const txt = addEv[1]!.trim().slice(0, 360);
        const verdict = /\b(refut|disprov|fail)/i.test(txt)
          ? 'refuting'
          : /\b(support|confirm|improv|drop|gain|faster|lower)/i.test(txt)
            ? 'supporting'
            : 'observation';
        const refH = txt.match(/\b(H\d+)\b/i)?.[1]?.toUpperCase() ?? String(hyps[0]?.id ?? 'H1');
        evs.unshift({ hypothesisId: refH, verdict, text: txt, date: new Date().toISOString().slice(0, 10) });
        patchWeights.evidence = evs;
      }
      const addM = params.instruction.match(/\badd\s+milestone\b[:\s-]*(.+)$/i);
      if (addM) {
        miles.push({ title: addM[1]!.trim().slice(0, 100), status: 'planned', dueLabel: `Week ${miles.length + 1}` });
        patchWeights.milestones = miles;
      }
    } else if (mediaType === 'tool-organ') {
      const wins = Array.isArray(patchWeights.scheduleWindows)
        ? (patchWeights.scheduleWindows as Array<Record<string, unknown>>).map((w) => ({ ...w }))
        : [];
      const steps = Array.isArray(patchWeights.steps)
        ? (patchWeights.steps as Array<Record<string, unknown>>).map((s) => ({ ...s }))
        : [];
      const tMatch = params.instruction.match(
        /\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:to|-|–|until)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b/i,
      );
      if (tMatch && /\b(protect|block|schedule|focus|quiet|add)\b/i.test(params.instruction)) {
        const toMin = (hRaw: string, mRaw: string | undefined, mer: string | undefined): number => {
          let h = Number(hRaw) % 24;
          const m = mRaw ? Number(mRaw) % 60 : 0;
          const lowMer = mer?.toLowerCase();
          if (lowMer === 'pm' && h < 12) h += 12;
          if (lowMer === 'am' && h === 12) h = 0;
          return h * 60 + m;
        };
        const sMin = toMin(tMatch[1]!, tMatch[2], tMatch[3] || tMatch[6]);
        const eMin = toMin(tMatch[4]!, tMatch[5], tMatch[6] || tMatch[3]);
        if (eMin > sMin) {
          const forMatch = params.instruction.match(/\bfor\s+([a-z0-9][a-z0-9\s-]{1,40})$/i);
          wins.push({
            label: forMatch?.[1]?.trim() || 'Protected Block',
            startMinuteOfDay: sMin,
            endMinuteOfDay: eMin,
            daysOfWeek: [0, 1, 2, 3, 4, 5, 6],
            protectedWindow: true,
          });
          patchWeights.scheduleWindows = wins;
        }
      }
      const addStep = params.instruction.match(/\badd\s+step\b[:\s-]*(.+)$/i);
      if (addStep) {
        steps.push({ id: `s${Date.now()}`, text: addStep[1]!.trim().slice(0, 180), done: false });
        patchWeights.steps = steps;
      }
    } else if (mediaType === 'space') {
      if (/\bboard\b/i.test(params.instruction)) patchWeights.layout = 'board';
      else if (/\bgrid\b/i.test(params.instruction)) patchWeights.layout = 'grid';
      else if (/\bshowcase\b/i.test(params.instruction)) patchWeights.layout = 'showcase';
      else if (/\blist\b|\btimeline\b/i.test(params.instruction)) patchWeights.layout = 'timeline';
    }

    const seedJson =
      rawModelJson ||
      JSON.stringify({
        name: proj.draft.name,
        purpose: decl.concept.purpose || proj.draft.summary,
        mediaType,
        readyToBuild: true,
        weights: patchWeights,
        reply: `Updated ${proj.draft.name}.`,
      });

    const { spec } = parseDiscoveryAgentResponse(seedJson, params.instruction, decl, 2, mediaType);
    const assembled = assembleProceduralPrismOps(decl, { ...spec, name: proj.draft.name, readyToBuild: true }, params.instruction);
    await store.applyStudioOp(
      proj.id,
      {
        op: 'batch',
        ops: [
          ...assembled.ops,
          {
            op: 'set-concept-profile',
            pinnedFields: mergedPins,
            semanticConceptId: refinedSemanticRev.conceptId,
            semanticKappa: refinedSemanticRev.kappa,
            semanticVersion: refinedSemanticRev.version,
          },
        ],
      },
      { pushRevision: true },
    );
    pool.evictPrism(proj.id);
    const published = await store.publish(proj.id);
    const updatedProj = published ?? store.get(proj.id)!;
    this.lastTouchedPrismId = updatedProj.id;

    const refreshedPrisms = store.list();
    observer.syncPublishedPrisms(refreshedPrisms);
    registry.syncPrismsAsExperts(refreshedPrisms);
    galleries.syncFromPrisms(refreshedPrisms);

    const rendered = await materializeInlineArtifact(updatedProj, this.deps);
    if (mediaType !== 'space' && (rendered.thumbnailDataUrl || rendered.previewImageDataUrl)) {
      galleries.setArtifactMeta(updatedProj.id, {
        previewDataUrl: rendered.thumbnailDataUrl ?? rendered.previewImageDataUrl,
      });
      const rec = galleries.get(updatedProj.id);
      for (const sid of rec?.spaceIds ?? []) {
        await syncSpacePrismView(sid, this.deps);
      }
    }

    if (mediaType !== 'space' && this.deps.scheduleProgressiveReconciliation) {
      const refineObs = observeInputSensoryState([], params.instruction, false);
      const refineSnap = observer.getThreadSnapshot('studio');
      const refineBudget = senseUserQualityBudget({
        latestUserText: params.instruction,
        immediateObs: refineObs.vector,
        integratedBelief: refineSnap.belief,
        surpriseQ10: Math.max(420, refineSnap.lastSurpriseQ10),
        retryEscalationLevel: refineSnap.retryEscalationLevel + 1,
        globalUserFacetsQ10: registry.basis.getUserFacetsQ10(),
        hasEnabledContentStreams: Boolean(this.deps.contentStreams?.hasEnabledStreams()),
      });
      this.deps.scheduleProgressiveReconciliation({
        prismId: updatedProj.id,
        prompt: `${updatedProj.draft.name}: ${params.instruction}`,
        initialRevision: refinedSemanticRev,
        startRank: 1,
        maxRank:
          refineBudget.featureRichness === 'minimal'
            ? (Math.max(1, refineBudget.targetLodRank) as PrismLodRank)
            : 3,
      });
    }

    registry.basis.recordAdaptation({
      mediaType,
      expertId: mediaType === 'space' ? `space:${updatedProj.id}` : `content:${updatedProj.id}`,
      kappa: (updatedProj.published ?? updatedProj.draft).digest,
      title: updatedProj.draft.name,
      summary: params.instruction,
    });

    return {
      __prismInvocation: true,
      action: 'update',
      autoCreated: false,
      prismId: updatedProj.id,
      prismName: updatedProj.published?.name ?? updatedProj.draft.name,
      mediaType,
      galleryLabel: mediaType === 'space' ? 'Spaces' : GALLERY_LABEL_BY_TYPE[mediaType],
      digest: (updatedProj.published ?? updatedProj.draft).digest,
      summary: `Updated "${updatedProj.draft.name}": ${params.instruction}`,
      ...rendered,
      output: { status: 'updated', prismName: updatedProj.draft.name, instruction: params.instruction },
    };
  }

  async manageSpacesAndPrisms(params: {
    readonly action: 'list' | 'create' | 'add_item' | 'open' | 'publish' | 'delete' | 'rename';
    readonly name?: string;
    readonly target?: string;
    readonly description?: string;
    readonly artifact?: string;
    readonly space?: string;
    readonly addLatest?: boolean;
    readonly spacesOnly?: boolean;
  }): Promise<StudioChatArtifactResult> {
    const { store, galleries, observer, registry, pool } = this.deps;
    const all = store.list();

    if (params.action === 'list') {
      const filtered = params.spacesOnly
        ? all.filter((p) => (p.declaration?.view?.proceduralOrgan?.mediaType ?? p.declaration?.concept.mediaType) === 'space')
        : all;
      const items = filtered.map((p) => {
        const mt: PrismMediaType =
          p.declaration?.view?.proceduralOrgan?.mediaType ?? p.declaration?.concept.mediaType ?? 'tool-organ';
        return {
          id: p.id,
          name: p.published?.name ?? p.draft.name,
          mediaType: mt,
          published: p.published !== null,
          summary: p.draft.summary || PRISM_MEDIA_CATALOG[mt].shortSummary,
          previewDataUrl: galleries.get(p.id)?.previewDataUrl,
        };
      });
      return {
        __prismInvocation: true,
        action: 'list',
        autoCreated: false,
        prismId: items[0]?.id ?? 'list',
        prismName: params.spacesOnly ? 'Spaces' : 'Your Prisms & Spaces',
        mediaType: 'space',
        digest: 'list',
        summary:
          items.length === 0
            ? 'No Prisms or Spaces created yet. Ask me to generate a picture, audio track, game, document, or Space!'
            : `${items.length} saved ${params.spacesOnly ? 'Space' : 'item'}${items.length === 1 ? '' : 's'}.`,
        prismsList: items,
        output: { count: items.length, items: items.map((i) => ({ name: i.name, type: i.mediaType })) },
      };
    }

    if (params.action === 'create') {
      const name = params.name?.trim() || 'New Space';
      const previousArtifactId = this.lastTouchedPrismId;
      const created = await this.createArtifactFromPrompt({
        prompt: params.description || `Curated Space "${name}" organizing content with a live custom view.`,
        title: name,
        forcedMediaType: 'space',
        weightOverrides: { title: name, description: params.description || `Curated content in ${name}.`, layout: 'grid', items: [] },
      });
      if (params.addLatest && previousArtifactId && previousArtifactId !== created.prismId) {
        galleries.assignToSpace(previousArtifactId, created.prismId);
        await syncSpacePrismView(created.prismId, this.deps);
      }
      return created;
    }

    if (params.action === 'add_item') {
      let spaceProj = findPrismByNameOrId(params.space, all, { mediaType: 'space' });
      if (!spaceProj && params.space?.trim()) {
        const createdSpace = await this.createArtifactFromPrompt({
          prompt: `Curated Space "${params.space.trim()}"`,
          title: params.space.trim(),
          forcedMediaType: 'space',
        });
        spaceProj = store.get(createdSpace.prismId);
      }
      const nonSpaces = all.filter(
        (p) => (p.declaration?.view?.proceduralOrgan?.mediaType ?? p.declaration?.concept.mediaType) !== 'space',
      );
      const artProj =
        !params.artifact || params.artifact.toLowerCase() === 'latest'
          ? (this.lastTouchedPrismId ? store.get(this.lastTouchedPrismId) : null) ?? nonSpaces[0] ?? null
          : findPrismByNameOrId(params.artifact, nonSpaces);

      if (!spaceProj || !artProj) {
        throw new Error('Could not find the Space or artifact to link.');
      }
      galleries.assignToSpace(artProj.id, spaceProj.id);
      const synced = await syncSpacePrismView(spaceProj.id, this.deps);
      return {
        __prismInvocation: true,
        action: 'space',
        autoCreated: false,
        prismId: spaceProj.id,
        prismName: spaceProj.draft.name,
        mediaType: 'space',
        spaceId: spaceProj.id,
        spaceName: spaceProj.draft.name,
        digest: (synced?.published ?? spaceProj.draft).digest,
        summary: `Added "${artProj.draft.name}" to Space "${spaceProj.draft.name}".`,
        output: { added: artProj.draft.name, space: spaceProj.draft.name },
      };
    }

    const found = findPrismByNameOrId(params.target ?? params.name, all);
    if (!found) {
      throw new Error(`No Prism or Space matching "${params.target ?? params.name ?? ''}" was found.`);
    }
    const mt: PrismMediaType =
      found.declaration?.view?.proceduralOrgan?.mediaType ?? found.declaration?.concept.mediaType ?? 'tool-organ';

    if (params.action === 'open') {
      this.deps.onOpenPrism?.(found.id);
      return {
        __prismInvocation: true,
        action: 'open',
        autoCreated: false,
        prismId: found.id,
        prismName: found.published?.name ?? found.draft.name,
        mediaType: mt,
        digest: (found.published ?? found.draft).digest,
        summary: `Opened "${found.draft.name}".`,
        output: { opened: found.draft.name },
      };
    }

    if (params.action === 'publish') {
      const pub = await store.publish(found.id);
      observer.syncPublishedPrisms(store.list());
      registry.syncPrismsAsExperts(store.list());
      galleries.syncFromPrisms(store.list());
      return {
        __prismInvocation: true,
        action: 'publish',
        autoCreated: false,
        prismId: found.id,
        prismName: pub?.published?.name ?? found.draft.name,
        mediaType: mt,
        digest: (pub?.published ?? found.draft).digest,
        summary: `Published "${found.draft.name}".`,
        output: { published: found.draft.name },
      };
    }

    if (params.action === 'delete') {
      const name = found.draft.name;
      pool.evictPrism(found.id);
      await store.delete(found.id);
      const next = store.list();
      observer.syncPublishedPrisms(next);
      registry.syncPrismsAsExperts(next);
      galleries.syncFromPrisms(next);
      return {
        __prismInvocation: true,
        action: 'delete',
        autoCreated: false,
        prismId: found.id,
        prismName: name,
        mediaType: mt,
        digest: 'deleted',
        summary: `Deleted "${name}".`,
        output: { deleted: name },
      };
    }

    throw new Error(`Unsupported Studio action "${params.action}".`);
  }

  browseGallery(mediaType: GalleryMediaType, category?: string): StudioChatArtifactResult {
    const items = this.deps.galleries.listByGallery(mediaType, { category });
    this.deps.onOpenGallery?.(mediaType);
    const label = GALLERY_LABEL_BY_TYPE[mediaType];
    return {
      __prismInvocation: true,
      action: 'gallery',
      autoCreated: false,
      prismId: items[0]?.id ?? `gallery:${mediaType}`,
      prismName: `${label} Gallery`,
      mediaType,
      galleryLabel: label,
      digest: `gallery:${mediaType}`,
      summary:
        items.length === 0
          ? `Your ${label} gallery is empty. Ask me to create your first ${PRISM_MEDIA_CATALOG[mediaType].label.toLowerCase()}!`
          : `${items.length} item${items.length === 1 ? '' : 's'} in ${label}${category ? ` (${category})` : ''}.`,
      prismsList: items.map((it) => ({
        id: it.id,
        name: it.title,
        mediaType: it.mediaType,
        published: it.published,
        summary: it.summary,
        previewDataUrl: it.previewDataUrl,
      })),
      output: { gallery: label, count: items.length },
    };
  }
}
