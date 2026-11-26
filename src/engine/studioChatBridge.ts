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
  synthesizeCompositionStereoPcm,
} from './audioSynthesisEngine';
import type { VerdantCognitiveObserver } from './cognitiveObserver';
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
  buildWeightRoutingDiscoveryPrompt,
  classifyArtifactMediaType,
  extractCleanIdeaTitle,
  parseDiscoveryAgentResponse,
} from './prismStudioAgent';
import {
  inferPictureWeightsFromPrompt,
  renderPictureWithCoherenceCheck,
} from './proceduralPictureRenderer';
import type { UnifiedExpertRegistry } from './unifiedModelBasis';
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
  readonly registry: UnifiedExpertRegistry;
  readonly runLocalStudioPrompt?: (systemPrompt: string, userPrompt: string, maxTokens: number) => Promise<string>;
  readonly synthesizeAudioWav?: (params: {
    readonly title: string;
    readonly style: AudioWeights['style'];
    readonly bpm: number;
    readonly rootMidi: number;
    readonly scale: AudioWeights['scale'];
    readonly durationSec: number;
  }) => Promise<{ readonly wavBytes: Uint8Array; readonly durationSec: number }>;
  readonly onOpenPrism?: (prismId: string) => void;
  readonly onOpenGallery?: (mediaType: GalleryMediaType) => void;
}

function normKey(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, '');
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
    const rendered = await renderPictureWithCoherenceCheck(weights as PictureWeights, {
      maxPreviewDim: 640,
    });
    return {
      previewImageDataUrl: rendered.pngDataUrl,
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
    const pcm = synthesizeCompositionStereoPcm(comp, 22_050);
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

    // Seed picture weights using semantic prompt inference + learned living basis affinities
    const baseWeightsSeed: Record<string, unknown> =
      mediaType === 'picture'
        ? {
            ...inferPictureWeightsFromPrompt(promptText, title, affinities),
            ...(params.weightOverrides ?? {}),
          }
        : mediaType === 'audio'
          ? {
              ...(affinities.preferredAudioScale ? { scale: affinities.preferredAudioScale } : {}),
              ...(affinities.preferredAudioBpm ? { bpm: affinities.preferredAudioBpm } : {}),
              ...(params.weightOverrides ?? {}),
            }
          : { ...(params.weightOverrides ?? {}) };

    // Engage on-device LLM weight router when Local AI is active for richer custom specs
    let rawModelJson = '';
    if (this.deps.runLocalStudioPrompt) {
      try {
        const sysPrompt = buildWeightRoutingDiscoveryPrompt(
          [{ role: 'user', text: promptText }],
          baseDecl,
          2,
          mediaType,
        );
        rawModelJson = await this.deps.runLocalStudioPrompt(sysPrompt, promptText, 512);
      } catch {
        rawModelJson = '';
      }
    }

    const seedJson =
      rawModelJson ||
      JSON.stringify({
        name: title,
        purpose: promptText.slice(0, 220),
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
    await store.applyStudioOp(created.id, { op: 'batch', ops: assembled.ops }, { pushRevision: false });
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
        previewDataUrl: rendered.previewImageDataUrl,
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
      summary: activeProj.draft.summary || promptText.slice(0, 160),
      ...rendered,
      output: {
        status: 'synthesized',
        mediaType,
        gallery: mediaType === 'space' ? 'Spaces' : GALLERY_LABEL_BY_TYPE[mediaType],
        categories,
        ...(resolvedSpaceName ? { space: resolvedSpaceName } : {}),
      },
    };
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

    let rawModelJson = '';
    if (this.deps.runLocalStudioPrompt) {
      try {
        const prompt = buildWeightRefinementPrompt([], decl, params.instruction);
        rawModelJson = await this.deps.runLocalStudioPrompt(prompt, params.instruction, 512);
      } catch {
        rawModelJson = '';
      }
    }

    // Apply deterministic semantic overrides (e.g. "night with stars", "16x16 with 40 mines", "128 BPM")
    const curW = decl.view?.proceduralOrgan?.weights as unknown as Record<string, unknown> | undefined;
    let patchWeights: Record<string, unknown> = { ...(curW ?? {}) };
    if (mediaType === 'picture') {
      const inferred = inferPictureWeightsFromPrompt(params.instruction, proj.draft.name);
      patchWeights = {
        ...patchWeights,
        lighting: inferred.lighting,
        palette: inferred.palette,
        density: inferred.density,
        seed: ((Number(patchWeights.seed || 4219) * 1664525 + 1013904223) >>> 0) % 990_000 + 1000,
        ...(/\b(landscape|nebula|cosmos|architecture|botanical|ocean|abstract|mosaic|constellation|poster|pixel)\b/i.test(params.instruction)
          ? { generator: inferred.generator }
          : {}),
      };
    } else if (mediaType === 'audio') {
      const bpmMatch = params.instruction.match(/\b(\d{2,3})\s*bpm\b/i);
      if (bpmMatch) patchWeights.bpm = Number(bpmMatch[1]);
      const scMatch = params.instruction.match(/\b(pentatonic|major|minor|dorian)\b/i);
      if (scMatch) patchWeights.scale = scMatch[1]!.toLowerCase();
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
    await store.applyStudioOp(proj.id, { op: 'batch', ops: assembled.ops }, { pushRevision: true });
    pool.evictPrism(proj.id);
    const published = await store.publish(proj.id);
    const updatedProj = published ?? store.get(proj.id)!;
    this.lastTouchedPrismId = updatedProj.id;

    const refreshedPrisms = store.list();
    observer.syncPublishedPrisms(refreshedPrisms);
    registry.syncPrismsAsExperts(refreshedPrisms);
    galleries.syncFromPrisms(refreshedPrisms);

    const rendered = await materializeInlineArtifact(updatedProj, this.deps);
    if (mediaType !== 'space' && rendered.previewImageDataUrl) {
      galleries.setArtifactMeta(updatedProj.id, { previewDataUrl: rendered.previewImageDataUrl });
      const rec = galleries.get(updatedProj.id);
      for (const sid of rec?.spaceIds ?? []) {
        await syncSpacePrismView(sid, this.deps);
      }
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
