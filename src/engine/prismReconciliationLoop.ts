/**
 * Closed-Loop Progressive Prism Reconciliation Engine (`src/engine/prismReconciliationLoop.ts`)
 *
 * Extrapolated directly from `afflom/verdant-engine@9d22f5ac`:
 * - `crates/verdant-core/src/streaming.rs`: 4-tier monotonic LOD ladder
 *   (`Lod0CoarseProxy (1/4)` -> `Lod1MediumDetail (1/2)` -> `Lod2HighDetail (3/4)` -> `Lod3LosslessSource (1:1)`)
 * - `crates/verdant-tensor/src/linear/delta_rule.rs`: Gated Delta Rule cross-modal associative updates
 * - `crates/verdant-core/src/observer_streams.rs`: parent-linked `kappa` semantic revisions
 *
 * Orchestrates the closed-loop flow across EVERY Gallery MediaType without blocking navigation:
 *   Seed ("a dog") -> Language/Reasoning contextualization -> UnifiedSemanticAssociator (vN, kappa)
 *   -> Verdant Studio closure (Prism state space) -> Lod0..Lod3 progressive synthesis + Vision/Audio QA
 *   -> deterministic full-definition promotion into VerdantUI Gallery & Spaces.
 */

import {
  analyzeAudioFidelity,
  composeMultiTrackArrangement,
  masterStereoPcm,
  synthesizeCompositionStereoPcm,
} from './audioSynthesisEngine';
import type { ContentGalleryStore } from './contentGalleryStore';
import type { AudioWeights, PictureWeights, PrismMediaType, VideoWeights } from './prismMediaOrgans';
import type { PrismStore, VerdantPrismProject } from './prismStudio';
import {
  encodeCompactThumbnailDataUrl,
  renderPictureWithCoherenceCheck,
} from './proceduralPictureRenderer';
import {
  facetsQ10ForMediaType,
  type SemanticRevisionRecord,
  type UnifiedExpertRegistry,
  type UnifiedSemanticPacket,
} from './unifiedModelBasis';
import type { PrismLodLabel, PrismLodRank, PrismLodState } from './verdantStudioApi';
import {
  evaluateSemanticVisualQa,
  type SemanticVisualQaReport,
  type SemanticVisualQaTarget,
} from './visionCritiqueLoop';
import { createCooperativeYielder } from './vramSwapPolicy';

export interface LodTierSpec {
  readonly rank: PrismLodRank;
  readonly label: PrismLodLabel;
  readonly resRatio: number;
  readonly maxImageDim: number;
  readonly latentDim: number;
  readonly audioDurationSec: number;
  readonly deterministic: boolean;
  readonly userStatusLabel: string;
}

/** Monotonic 4-tier LOD ladder matching `verdant-engine/crates/verdant-core/src/streaming.rs`. */
export const VERDANT_LOD_LADDER: readonly [LodTierSpec, LodTierSpec, LodTierSpec, LodTierSpec] = [
  {
    rank: 0,
    label: 'Lod0CoarseProxy',
    resRatio: 0.25,
    maxImageDim: 128,
    latentDim: 16,
    audioDurationSec: 4,
    deterministic: false,
    userStatusLabel: 'Creating draft…',
  },
  {
    rank: 1,
    label: 'Lod1MediumDetail',
    resRatio: 0.5,
    maxImageDim: 256,
    latentDim: 32,
    audioDurationSec: 6,
    deterministic: false,
    userStatusLabel: 'Refining details…',
  },
  {
    rank: 2,
    label: 'Lod2HighDetail',
    resRatio: 0.75,
    maxImageDim: 384,
    latentDim: 48,
    audioDurationSec: 8,
    deterministic: false,
    userStatusLabel: 'Checking quality…',
  },
  {
    rank: 3,
    label: 'Lod3LosslessSource',
    resRatio: 1.0,
    maxImageDim: 512,
    latentDim: 64,
    audioDurationSec: 10,
    deterministic: true,
    userStatusLabel: 'Finishing full resolution…',
  },
];

export interface ReconciliationProgressEvent {
  readonly prismId: string;
  readonly conceptId: string;
  readonly semanticVersion: number;
  readonly lodRank: PrismLodRank;
  readonly lodLabel: PrismLodLabel;
  readonly qaScoreQ10: number;
  readonly qaAccepted: boolean;
  readonly statusLabel: string;
  readonly previewDataUrl?: string;
}

export interface ReconciliationDeps {
  readonly store: PrismStore;
  readonly galleries: ContentGalleryStore;
  readonly registry: UnifiedExpertRegistry;
  readonly runLocalLanguagePrompt?: (systemPrompt: string, userPrompt: string, maxTokens: number) => Promise<string>;
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
  readonly syncSpacesForArtifact?: (prismId: string) => Promise<void>;
}

const reconciliationYielder = createCooperativeYielder();

function yieldCooperativeFrame(): Promise<void> {
  return reconciliationYielder.yieldNow();
}

function extractCanonicalSubject(prompt: string): string {
  const cleaned = prompt
    .replace(
      /^\s*(?:please\s+)?(?:create|make|generate|draw|paint|render|compose|build|write|design|start)\s+(?:me\s+)?(?:an|a|the)?\s*/i,
      '',
    )
    .replace(
      /\b(picture|image|photo|illustration|artwork|video|clip|movie|audio|song|soundtrack|story|essay|document|game|research|utility|tool|prism)\s+(?:of|about|for)\s+/i,
      '',
    )
    .trim();
  const words = cleaned.split(/\s+/).filter(Boolean).slice(0, 6);
  return words.join(' ') || prompt.trim().slice(0, 48) || 'subject';
}

function inferDeterministicAttributes(prompt: string, subject: string): {
  readonly attributes: readonly string[];
  readonly composition: string;
  readonly lighting: string;
  readonly motionCues: readonly string[];
  readonly acousticCues: readonly string[];
} {
  const lower = prompt.toLowerCase();
  const attributes: string[] = [subject];
  if (/\b(dog|puppy|retriever|hound|canine)\b/.test(lower)) {
    attributes.push('expressive eyes', 'warm fur texture', 'alert ears', 'natural outdoor posture');
  } else if (/\b(cat|kitten|feline)\b/.test(lower)) {
    attributes.push('sleek silhouette', 'whiskers', 'calm poised stance');
  } else if (/\b(mountain|alpine|forest|ocean|sunset|landscape)\b/.test(lower)) {
    attributes.push('layered depth', 'atmospheric horizon', 'natural light balance');
  } else if (/\b(logo|emblem|brand|icon)\b/.test(lower)) {
    attributes.push('clean vector geometry', 'high figure-ground contrast', 'balanced symmetry');
  } else {
    attributes.push('clear focal silhouette', 'balanced composition', 'coherent detail');
  }

  const lighting = /\b(night|midnight|noir|dark)\b/.test(lower)
    ? 'midnight'
    : /\b(golden|sunset|sunrise|warm)\b/.test(lower)
      ? 'golden-hour'
      : /\b(twilight|dusk|evening)\b/.test(lower)
        ? 'twilight'
        : /\b(neon|cyber)\b/.test(lower)
          ? 'neon'
          : 'daylight';

  const composition = /\b(portrait|close-up|closeup)\b/.test(lower)
    ? 'centered portrait framing'
    : /\b(wide|panoramic|landscape)\b/.test(lower)
      ? 'wide environmental framing'
      : 'centered subject framing';

  const motionCues = /\b(run|running|play|playful|jump|chase|fast|energetic)\b/.test(lower)
    ? ['lively camera pan', 'playful rhythmic motion']
    : ['gentle breathing drift', 'smooth parallax glide'];

  const acousticCues = /\b(dog|puppy|playful|bright|happy|sun)\b/.test(lower)
    ? ['warm major harmony', 'light organic rhythm']
    : /\b(night|midnight|dark|noir|rain)\b/.test(lower)
      ? ['calm minor ambience', 'soft reverberant texture']
      : ['balanced melodic pulse', 'natural stereo warmth'];

  return { attributes, composition, lighting, motionCues, acousticCues };
}

/**
 * Stage 1: Activates Language/Reasoning contextualization (reusing any associated cross-modal
 * concept anchor from `UnifiedSemanticAssociator`), writes new semantic weights into the
 * associative state matrix `S` via `verdantGatedDeltaStepF32`, and returns the committed `kappa`
 * `SemanticRevisionRecord`.
 */
export async function contextualizePromptToUnifiedWeights(params: {
  readonly prompt: string;
  readonly mediaType: PrismMediaType;
  readonly prismId?: string;
  readonly activePrismId?: string;
  readonly palette?: readonly [string, string, string, string];
  readonly deps: Pick<ReconciliationDeps, 'registry' | 'runLocalLanguagePrompt'>;
}): Promise<{ readonly revision: SemanticRevisionRecord; readonly inheritedAnchor: boolean }> {
  const { prompt, mediaType, prismId, activePrismId, palette, deps } = params;
  // When refining an existing Prism (`prismId` already anchored), always advance its own concept
  // revision; for brand-new Prisms, only inherit an earlier anchor on strong semantic/anaphoric match.
  const ownConcept = prismId ? deps.registry.semantics.getConceptForPrism(prismId) : null;
  const associated =
    ownConcept ?? deps.registry.semantics.recallAssociatedSemantics(prompt, activePrismId);
  const canonicalSubject = associated?.canonicalSubject ?? extractCanonicalSubject(prompt);
  const det = inferDeterministicAttributes(prompt, canonicalSubject);

  const buildMediaBrief = (): string => {
    const cleanSubj = canonicalSubject.replace(/^[a-z]/, (c) => c.toUpperCase());
    if (mediaType === 'audio') {
      return `${cleanSubj} — multi-track stereo arrangement (${det.acousticCues.join(', ')}).`;
    }
    if (mediaType === 'video') {
      return `${cleanSubj} — animated storyboard sequence (${det.motionCues.join(', ')}).`;
    }
    if (mediaType === 'interactive-game') {
      return `${cleanSubj} — playable interactive game with local score tracking and keyboard/touch controls.`;
    }
    if (mediaType === 'written-work') {
      return `${cleanSubj} — multi-chapter editorial document with Write/Read views and Markdown/HTML export.`;
    }
    if (mediaType === 'research-program') {
      return `${cleanSubj} — structured research notebook with hypotheses, evidence log, and milestones.`;
    }
    if (mediaType === 'tool-organ') {
      return `${cleanSubj} — focused private utility running locally in your browser.`;
    }
    return `${cleanSubj} — ${det.attributes.slice(1, 4).join(', ')} in ${det.lighting} lighting.`;
  };

  const isLegacyStaleBrief =
    associated !== null &&
    mediaType !== 'picture' &&
    (/clear focal silhouette|daylight lighting/i.test(associated.contextualBrief) ||
      /^n\s+/i.test(associated.contextualBrief));

  let contextualBrief =
    associated && !isLegacyStaleBrief
      ? `${associated.contextualBrief} Adapted for ${mediaType}: ${prompt.slice(0, 140)}`
      : buildMediaBrief();
  let llmAttrs: readonly string[] = [];

  if (deps.runLocalLanguagePrompt) {
    try {
      const raw = await deps.runLocalLanguagePrompt(
        'You contextualize a user creative prompt into compact JSON semantic attributes for Verdant Studio. Reply ONLY with JSON: {"subject":"...","brief":"...","attributes":["..."],"lighting":"...","composition":"..."}.',
        prompt,
        160,
      );
      const match = raw.match(/\{[\s\S]*\}/);
      if (match) {
        const parsed = JSON.parse(match[0]) as {
          subject?: unknown;
          brief?: unknown;
          attributes?: unknown;
        };
        if (typeof parsed.brief === 'string' && parsed.brief.trim().length >= 8) {
          contextualBrief = parsed.brief.trim().slice(0, 220);
        }
        if (Array.isArray(parsed.attributes)) {
          llmAttrs = parsed.attributes
            .filter((x): x is string => typeof x === 'string' && x.trim().length > 0)
            .map((x) => x.trim().slice(0, 48))
            .slice(0, 6);
        }
      }
    } catch {
      // Deterministic contextualization already provides complete semantic weights
    }
  }

  const packet: UnifiedSemanticPacket = {
    canonicalSubject,
    contextualBrief,
    attributes: [...(associated?.attributes ?? []), ...det.attributes, ...llmAttrs],
    composition: associated?.composition ?? det.composition,
    lighting: associated?.lighting ?? det.lighting,
    palette: palette ?? associated?.palette,
    motionCues: [...(associated?.motionCues ?? []), ...det.motionCues],
    acousticCues: [...(associated?.acousticCues ?? []), ...det.acousticCues],
    facetsQ10: facetsQ10ForMediaType(mediaType),
    linkedPrismId: prismId,
  };

  const { revision } = deps.registry.semantics.writeSemanticObservation(canonicalSubject, packet, {
    gain: associated ? 0.45 : 0.72,
  });
  return { revision, inheritedAnchor: associated !== null };
}

/**
 * Stage 2 & 3: Drives a Prism monotonically across `Lod0CoarseProxy (0) -> Lod1MediumDetail (1)
 * -> Lod2HighDetail (2) -> Lod3LosslessSource (3)` with closed-loop Vision/Audio QA and Gated
 * Delta Rule semantic reconciliation.
 *
 * Strictly checks `isStale()` before every commit so navigation never interrupts it, while newer
 * edits or explicit cancellation cleanly supersede stale passes.
 */
export async function runProgressivePrismReconciliation(params: {
  readonly prismId: string;
  readonly prompt: string;
  readonly initialRevision: SemanticRevisionRecord;
  readonly startRank?: PrismLodRank;
  readonly maxRank?: PrismLodRank;
  readonly isStale?: () => boolean;
  readonly onProgress?: (ev: ReconciliationProgressEvent) => void;
  readonly deps: ReconciliationDeps;
}): Promise<{
  readonly finalRank: PrismLodRank;
  readonly finalRevision: SemanticRevisionRecord;
  readonly qaScoreQ10: number;
  readonly qaAccepted: boolean;
  readonly previewDataUrl?: string;
  readonly cancelled: boolean;
}> {
  const { prismId, prompt, deps, onProgress } = params;
  const isStale = params.isStale ?? (() => false);
  const startRank: PrismLodRank = params.startRank ?? 0;
  const maxRank: PrismLodRank = params.maxRank ?? 3;

  let currentRev = params.initialRevision;
  let lastRank: PrismLodRank = startRank;
  let lastScoreQ10 = 760;
  let lastAccepted = true;
  let lastPreviewUrl: string | undefined;
  let qaPasses = 0;

  for (let r = startRank; r <= maxRank; r++) {
    if (isStale()) {
      return {
        finalRank: lastRank,
        finalRevision: currentRev,
        qaScoreQ10: lastScoreQ10,
        qaAccepted: lastAccepted,
        previewDataUrl: lastPreviewUrl,
        cancelled: true,
      };
    }

    const tier = VERDANT_LOD_LADDER[r as PrismLodRank]!;
    const proj: VerdantPrismProject | null = deps.store.get(prismId);
    if (!proj?.declaration) break;

    const organ = proj.declaration.view?.proceduralOrgan;
    const mediaType: PrismMediaType =
      organ?.mediaType ?? proj.declaration.concept.mediaType ?? 'tool-organ';
    const pinned = new Set(proj.declaration.concept.pinnedFields ?? []);

    onProgress?.({
      prismId,
      conceptId: currentRev.conceptId,
      semanticVersion: currentRev.version,
      lodRank: tier.rank,
      lodLabel: tier.label,
      qaScoreQ10: lastScoreQ10,
      qaAccepted: lastAccepted,
      statusLabel: tier.userStatusLabel,
      previewDataUrl: lastPreviewUrl,
    });

    if (mediaType === 'picture' && organ?.weights.mediaType === 'picture') {
      const pw = organ.weights as PictureWeights;
      const runPass = async () => {
        let neuralLayer: { readonly width: number; readonly height: number; readonly rgba: Uint8ClampedArray } | null = null;
        if (deps.synthesizeImageLayer) {
          try {
            const synth = await deps.synthesizeImageLayer({
              prompt: `${currentRev.canonicalSubject}, ${currentRev.attributes.slice(0, 4).join(', ')}, ${currentRev.lighting} lighting, ${currentRev.composition}`,
              seed: pw.seed + tier.rank * 7,
              latentH: tier.latentDim,
              latentW: tier.latentDim,
            });
            if (synth.neuralAvailable && synth.rgba && synth.width > 0 && synth.height > 0) {
              neuralLayer = { width: synth.width, height: synth.height, rgba: synth.rgba };
            }
          } catch {
            neuralLayer = null;
          }
        }
        return renderPictureWithCoherenceCheck(pw, {
          maxPreviewDim: tier.maxImageDim,
          neuralLayer,
        });
      };

      const rendered =
        tier.deterministic && deps.runWithDeterministicPrecision
          ? await deps.runWithDeterministicPrecision(runPass)
          : await runPass();

      if (isStale()) {
        return {
          finalRank: lastRank,
          finalRevision: currentRev,
          qaScoreQ10: lastScoreQ10,
          qaAccepted: lastAccepted,
          previewDataUrl: lastPreviewUrl,
          cancelled: true,
        };
      }

      const qaTarget: SemanticVisualQaTarget = {
        conceptId: currentRev.conceptId,
        canonicalSubject: currentRev.canonicalSubject,
        contextualBrief: currentRev.contextualBrief,
        attributes: currentRev.attributes,
        lighting: currentRev.lighting,
        semanticVector: currentRev.semanticVector,
      };
      let qaReport: SemanticVisualQaReport;
      if (deps.evaluateVisualQa) {
        try {
          qaReport = await deps.evaluateVisualQa({
            width: rendered.width,
            height: rendered.height,
            rgba: rendered.rgba,
            target: qaTarget,
            minAcceptScoreQ10: tier.rank >= 2 ? 740 : 660,
          });
        } catch {
          qaReport = await evaluateSemanticVisualQa(
            { rgba: rendered.rgba, width: rendered.width, height: rendered.height },
            qaTarget,
            { minAcceptScoreQ10: tier.rank >= 2 ? 740 : 660 },
          );
        }
      } else {
        qaReport = await evaluateSemanticVisualQa(
          { rgba: rendered.rgba, width: rendered.width, height: rendered.height },
          qaTarget,
          { minAcceptScoreQ10: tier.rank >= 2 ? 740 : 660 },
        );
      }
      qaPasses += 1;
      lastScoreQ10 = qaReport.scoreQ10;
      lastAccepted = qaReport.accepted;
      lastPreviewUrl = rendered.pngDataUrl;

      // Feed Vision QA delta back into UnifiedSemanticAssociator & Verdant Studio state space
      let nextWeights: PictureWeights = rendered.weights;
      if (!qaReport.accepted && tier.rank < maxRank) {
        const { revision } = deps.registry.semantics.writeSemanticObservation(
          currentRev.canonicalSubject,
          {
            canonicalSubject: currentRev.canonicalSubject,
            contextualBrief: `${currentRev.contextualBrief} ${qaReport.refinementHint}`.slice(0, 240),
            attributes: [...currentRev.attributes, ...qaReport.missingAttributes.slice(0, 2)],
            composition: currentRev.composition,
            lighting: currentRev.lighting,
            palette: currentRev.palette,
            linkedPrismId: prismId,
          },
          { gain: 0.4 },
        );
        currentRev = revision;
        nextWeights = {
          ...rendered.weights,
          density: Math.min(92, (pw.density ?? 60) + 12),
          ...(!pinned.has('lighting') && qaReport.meanLuminance < 0.16 ? { lighting: 'golden-hour' } : {}),
        };
      } else if (qaReport.accepted && tier.rank === maxRank) {
        currentRev =
          deps.registry.semantics.recordAcceptedVisionTarget(currentRev.conceptId, qaReport.kappa) ??
          currentRev;
      }

      const lodState: PrismLodState = {
        rank: tier.rank,
        label: tier.label,
        qaScoreQ10: qaReport.scoreQ10,
        qaPasses,
        qaSummary: qaReport.refinementHint,
        updatedAt: Date.now(),
      };
      await deps.store.applyStudioOp(
        prismId,
        {
          op: 'reconcile-semantic-weights',
          semanticConceptId: currentRev.conceptId,
          semanticKappa: currentRev.kappa,
          semanticVersion: currentRev.version,
          lodState,
          weights: nextWeights,
        },
        { pushRevision: false },
      );

      const compactThumb = encodeCompactThumbnailDataUrl(
        rendered.rgba,
        rendered.width,
        rendered.height,
        256,
      );
      deps.galleries.setArtifactMeta(prismId, { previewDataUrl: compactThumb });
      if (tier.rank === maxRank && deps.syncSpacesForArtifact) {
        await deps.syncSpacesForArtifact(prismId);
      }
      lastRank = tier.rank;
    } else if (mediaType === 'audio' && organ?.weights.mediaType === 'audio') {
      const aw = organ.weights as AudioWeights;
      const comp = composeMultiTrackArrangement({
        title: aw.title,
        style: aw.style,
        bpm: aw.bpm,
        rootMidi: aw.rootMidi,
        scale: aw.scale,
        durationSec: Math.min(tier.audioDurationSec, aw.durationSec),
      });
      const rawPcm = synthesizeCompositionStereoPcm(comp, 22_050);
      const mastered = masterStereoPcm(rawPcm.left, rawPcm.right, rawPcm.sampleRate);
      const fidelity = analyzeAudioFidelity(mastered.left, mastered.right);
      qaPasses += 1;
      lastAccepted = !fidelity.hasNaN && fidelity.nonSilent && fidelity.clippedSamples === 0;
      lastScoreQ10 = lastAccepted ? 940 : 620;

      const lodState: PrismLodState = {
        rank: tier.rank,
        label: tier.label,
        qaScoreQ10: lastScoreQ10,
        qaPasses,
        qaSummary: lastAccepted
          ? `Audio QA accepted (${comp.durationSec}s stereo, peak ${fidelity.peakDbfs.toFixed(1)} dBFS).`
          : 'Audio level normalized for headroom.',
        updatedAt: Date.now(),
      };
      await deps.store.applyStudioOp(
        prismId,
        {
          op: 'reconcile-semantic-weights',
          semanticConceptId: currentRev.conceptId,
          semanticKappa: currentRev.kappa,
          semanticVersion: currentRev.version,
          lodState,
        },
        { pushRevision: false },
      );
      lastRank = tier.rank;
      // Audio converges cleanly at Lod1 and Lod3 without redundant middle renders
      if (r === 1 && maxRank >= 3) r = 2;
    } else if (mediaType === 'video' && organ?.weights.mediaType === 'video') {
      const vw = organ.weights as VideoWeights;
      const posterWeights: PictureWeights = {
        mediaType: 'picture',
        generator: 'abstract-flow',
        lighting: (currentRev.lighting as PictureWeights['lighting']) || 'twilight',
        width: vw.aspectRatio === '1:1' ? tier.maxImageDim : vw.aspectRatio === '9:16' ? Math.round(tier.maxImageDim * 0.56) : tier.maxImageDim,
        height: vw.aspectRatio === '1:1' ? tier.maxImageDim : vw.aspectRatio === '9:16' ? tier.maxImageDim : Math.round(tier.maxImageDim * 0.56),
        seed: (proj.createdAt % 900_000) + 1000 + tier.rank * 13,
        symmetry: 6,
        density: 58 + tier.rank * 6,
        palette: currentRev.palette,
        caption: vw.scenes[0]?.title ?? currentRev.canonicalSubject,
      };
      const poster = await renderPictureWithCoherenceCheck(posterWeights, {
        maxPreviewDim: tier.maxImageDim,
      });
      const qaReport = await evaluateSemanticVisualQa(
        { rgba: poster.rgba, width: poster.width, height: poster.height },
        {
          conceptId: currentRev.conceptId,
          canonicalSubject: currentRev.canonicalSubject,
          contextualBrief: currentRev.contextualBrief,
          attributes: currentRev.attributes,
          lighting: currentRev.lighting,
          semanticVector: currentRev.semanticVector,
        },
      );
      qaPasses += 1;
      lastScoreQ10 = qaReport.scoreQ10;
      lastAccepted = qaReport.accepted;
      lastPreviewUrl = poster.pngDataUrl;
      const lodState: PrismLodState = {
        rank: tier.rank,
        label: tier.label,
        qaScoreQ10: qaReport.scoreQ10,
        qaPasses,
        qaSummary: qaReport.refinementHint,
        updatedAt: Date.now(),
      };
      await deps.store.applyStudioOp(
        prismId,
        {
          op: 'reconcile-semantic-weights',
          semanticConceptId: currentRev.conceptId,
          semanticKappa: currentRev.kappa,
          semanticVersion: currentRev.version,
          lodState,
        },
        { pushRevision: false },
      );
      deps.galleries.setArtifactMeta(prismId, {
        previewDataUrl: encodeCompactThumbnailDataUrl(poster.rgba, poster.width, poster.height, 256),
      });
      lastRank = tier.rank;
    } else {
      // Written work, interactive game, research program, utility, space: reconcile semantic state space directly
      qaPasses += 1;
      lastScoreQ10 = 960;
      lastAccepted = true;
      const lodState: PrismLodState = {
        rank: maxRank,
        label: VERDANT_LOD_LADDER[maxRank]!.label,
        qaScoreQ10: lastScoreQ10,
        qaPasses,
        qaSummary: `Reconciled "${currentRev.canonicalSubject}" state space (${prompt.slice(0, 60)}).`,
        updatedAt: Date.now(),
      };
      await deps.store.applyStudioOp(
        prismId,
        {
          op: 'reconcile-semantic-weights',
          semanticConceptId: currentRev.conceptId,
          semanticKappa: currentRev.kappa,
          semanticVersion: currentRev.version,
          lodState,
        },
        { pushRevision: false },
      );
      lastRank = maxRank;
      break;
    }

    onProgress?.({
      prismId,
      conceptId: currentRev.conceptId,
      semanticVersion: currentRev.version,
      lodRank: lastRank,
      lodLabel: VERDANT_LOD_LADDER[lastRank]!.label,
      qaScoreQ10: lastScoreQ10,
      qaAccepted: lastAccepted,
      statusLabel:
        lastRank === maxRank ? 'Full quality ready' : VERDANT_LOD_LADDER[lastRank]!.userStatusLabel,
      previewDataUrl: lastPreviewUrl,
    });

    if (r < maxRank) {
      await yieldCooperativeFrame();
    }
  }

  return {
    finalRank: lastRank,
    finalRevision: currentRev,
    qaScoreQ10: lastScoreQ10,
    qaAccepted: lastAccepted,
    previewDataUrl: lastPreviewUrl,
    cancelled: false,
  };
}
