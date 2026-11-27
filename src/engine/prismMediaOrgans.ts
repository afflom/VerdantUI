/**
 * Procedural MediaType Organ Synthesizers & Sparse Studio Sub-Model Activation
 * (`src/engine/prismMediaOrgans.ts`)
 *
 * Implements DEC-009 (Zero LLM-written JavaScript):
 * - Every Verdant Studio artifact has an explicit `PrismMediaType` (`picture`, `written-work`,
 *   `video`, `interactive-game`, `research-program`, `tool-organ`).
 * - `deriveActiveStudioOrgans(mediaType)` activates ONLY the sparse subset of Verdant Studio
 *   sub-model organs required for that artifact type.
 * - The on-device LLM acts strictly as a Weight Router (`route-prism-weights`), outputting
 *   compact, schema-clamped JSON weights (`ProceduralOrganWeights`).
 * - `synthesizeProceduralOrganView(decl, spec)` compiles those validated JSON weights into
 *   audited, 100% self-contained procedural HTML/CSS/runtime-kernel JS with zero LLM-authored code.
 */

import type { PrismDeclaration } from './verdantStudioApi';
import {
  inferPictureWeightsFromPrompt,
  synthesizeLogoSvg,
  type PictureLightingMood,
} from './proceduralPictureRenderer';
import {
  buildArcadeCanvasRuntimeJs,
  buildCellularSimRuntimeJs,
  buildGridPuzzleRuntimeJs,
  buildGridSweepRuntimeJs,
  buildTurnBoardRuntimeJs,
} from './prismGameRuntimes';
import {
  buildAudioRuntimeJs,
  buildPictureRuntimeJs,
  buildResearchProgramRuntimeJs,
  buildSpaceRuntimeJs,
  buildToolOrganRuntimeJs,
  buildVideoRuntimeJs,
  buildWrittenWorkRuntimeJs,
} from './prismMediaRuntimes';

export type PrismMediaType =
  | 'picture'
  | 'audio'
  | 'written-work'
  | 'video'
  | 'interactive-game'
  | 'research-program'
  | 'tool-organ'
  | 'space';

export type StudioFeatureOrgan =
  | 'visual-canvas'
  | 'audio-synth'
  | 'palette-theme'
  | 'narrative-outline'
  | 'timeline-motion'
  | 'game-simulation'
  | 'rule-invariants'
  | 'evidence-ledger'
  | 'sensory-stream'
  | 'impulse-emitter'
  | 'privacy-compartment';

export type MediaCapabilityKind = 'language' | 'vision' | 'audio' | 'image-creation';

export function requiredCapabilitiesForMediaType(mediaType: PrismMediaType): readonly MediaCapabilityKind[] {
  switch (mediaType) {
    case 'picture':
      return ['image-creation', 'vision'];
    case 'audio':
      return ['audio', 'language'];
    case 'video':
      return ['image-creation', 'vision', 'audio'];
    case 'interactive-game':
      return ['language', 'vision'];
    case 'written-work':
      return ['language'];
    case 'research-program':
      return ['language', 'vision'];
    case 'tool-organ':
      return ['language'];
    case 'space':
      return ['language', 'vision'];
  }
}

export interface MediaTypeDescriptor {
  readonly mediaType: PrismMediaType;
  readonly label: string;
  readonly shortSummary: string;
  readonly activeOrgans: readonly StudioFeatureOrgan[];
  readonly activeWizardQuestions: readonly string[];
}

export const PRISM_MEDIA_CATALOG: Readonly<Record<PrismMediaType, MediaTypeDescriptor>> = {
  picture: {
    mediaType: 'picture',
    label: 'Picture',
    shortSummary: 'Artwork and visual composition with PNG export',
    activeOrgans: ['visual-canvas', 'palette-theme'],
    activeWizardQuestions: ['visual style & mood', 'color palette & composition'],
  },
  audio: {
    mediaType: 'audio',
    label: 'Audio',
    shortSummary: 'Multi-track music and soundscape studio with WAV export',
    activeOrgans: ['audio-synth', 'palette-theme'],
    activeWizardQuestions: ['musical mood & tempo', 'scale & sonic texture'],
  },
  'written-work': {
    mediaType: 'written-work',
    label: 'Writing',
    shortSummary: 'Novel, story, or manuscript workspace with chapter outline & drafting',
    activeOrgans: ['narrative-outline', 'palette-theme'],
    activeWizardQuestions: ['genre & voice', 'chapter structure & target length'],
  },
  video: {
    mediaType: 'video',
    label: 'Video',
    shortSummary: 'Storyboard, motion timeline player, and video export',
    activeOrgans: ['visual-canvas', 'timeline-motion', 'palette-theme'],
    activeWizardQuestions: ['motion style & pacing', 'scenes & visual palette'],
  },
  'interactive-game': {
    mediaType: 'interactive-game',
    label: 'Game',
    shortSummary: 'Playable game synthesized from your rules, board, and difficulty',
    activeOrgans: ['game-simulation', 'rule-invariants', 'palette-theme'],
    activeWizardQuestions: ['board size & difficulty presets', 'special rules & visual feel'],
  },
  'research-program': {
    mediaType: 'research-program',
    label: 'Research',
    shortSummary: 'Hypotheses, experiments, milestones, and structured evidence synthesis',
    activeOrgans: ['narrative-outline', 'rule-invariants', 'evidence-ledger', 'privacy-compartment'],
    activeWizardQuestions: ['research question & hypotheses', 'milestones & privacy space'],
  },
  'tool-organ': {
    mediaType: 'tool-organ',
    label: 'Utility',
    shortSummary: 'Personal utility, habit & schedule guardian, calculator, or workflow helper',
    activeOrgans: ['rule-invariants', 'sensory-stream', 'impulse-emitter', 'privacy-compartment'],
    activeWizardQuestions: ['core routine or rules', 'metrics or schedule preferences'],
  },
  space: {
    mediaType: 'space',
    label: 'Space',
    shortSummary: 'Curated Verdant Prism space organizing arbitrary content with a live custom view',
    activeOrgans: ['palette-theme', 'narrative-outline', 'evidence-ledger'],
    activeWizardQuestions: ['space focus & theme', 'layout style & curated sections'],
  },
};

export function deriveActiveStudioOrgans(mediaType: PrismMediaType): readonly StudioFeatureOrgan[] {
  return PRISM_MEDIA_CATALOG[mediaType].activeOrgans;
}

// ============================================================================
// Typed Declarative Weight Schemas (`ProceduralOrganWeights`)
// ============================================================================

export type PictureGeneratorKind =
  | 'landscape-vista'
  | 'cosmic-nebula'
  | 'architectural-scene'
  | 'botanical-garden'
  | 'ocean-seascape'
  | 'abstract-flow'
  | 'geometric-mosaic'
  | 'constellation-atlas'
  | 'poster-editorial'
  | 'pixel-scene'
  | 'logo-mark'
  | 'cover-art'
  | 'botanical-fractal'
  | 'wave-field'
  | 'constellation'
  | 'pixel-sprite';

export interface PictureWeights {
  readonly mediaType: 'picture';
  readonly generator: PictureGeneratorKind;
  readonly lighting?: PictureLightingMood;
  readonly width: number;
  readonly height: number;
  readonly seed: number;
  readonly symmetry: number;
  readonly density: number;
  readonly palette: readonly [string, string, string, string];
  readonly caption: string;
  readonly subtitle?: string;
  readonly coverFormat?: 'book' | 'album' | 'game';
  readonly svgMarkup?: string;
  /** Compact (`<= 512px`) Vision-QA-accepted neural render persisted for 1:1 live sandbox display. */
  readonly reconciledImageDataUrl?: string;
}

export interface AudioWeights {
  readonly mediaType: 'audio';
  readonly style: 'ambient-pad' | 'melodic-arpeggio' | 'rhythm-pulse' | 'sound-effects';
  readonly bpm: number;
  readonly rootMidi: number;
  readonly scale: 'pentatonic' | 'major' | 'minor' | 'dorian';
  readonly durationSec: number;
  readonly title: string;
}

export interface WrittenChapterWeight {
  readonly title: string;
  readonly beat: string;
  readonly summary: string;
  readonly body: string;
}

export interface WrittenWorkWeights {
  readonly mediaType: 'written-work';
  readonly genre: string;
  readonly title: string;
  readonly premise: string;
  readonly voice: string;
  readonly targetWordCount: number;
  readonly allowLocalAssist: boolean;
  readonly chapters: readonly WrittenChapterWeight[];
}

export interface VideoSceneWeight {
  readonly title: string;
  readonly subtitle: string;
  readonly durationSec: number;
  readonly motion: 'pan' | 'pulse' | 'orbit' | 'rise';
  readonly accent: string;
}

export interface VideoWeights {
  readonly mediaType: 'video';
  readonly aspectRatio: '16:9' | '1:1' | '9:16';
  readonly fps: number;
  readonly style: 'kinetic-type' | 'abstract-motion' | 'storyboard-scenes';
  readonly palette: readonly [string, string, string];
  readonly scenes: readonly VideoSceneWeight[];
}

export type GameArchetype =
  | 'grid-sweep'
  | 'grid-puzzle'
  | 'turn-board'
  | 'arcade-canvas'
  | 'cellular-sim';

export interface GridSweepPreset {
  readonly label: string;
  readonly rows: number;
  readonly cols: number;
  readonly mines: number;
}

export interface GridSweepGameWeights {
  readonly mediaType: 'interactive-game';
  readonly archetype: 'grid-sweep';
  readonly rows: number;
  readonly cols: number;
  readonly mines: number;
  readonly presets: readonly GridSweepPreset[];
  readonly firstClickSafe: boolean;
  readonly floodReveal: boolean;
  readonly chordReveal: boolean;
  readonly showTimer: boolean;
  readonly visualStyle: 'classic' | 'modern' | 'retro';
}

export interface GridPuzzleGameWeights {
  readonly mediaType: 'interactive-game';
  readonly archetype: 'grid-puzzle';
  readonly variant: 'merge-2048' | 'memory-match' | 'lights-out';
  readonly rows: number;
  readonly cols: number;
  readonly targetScore: number;
}

export interface TurnBoardGameWeights {
  readonly mediaType: 'interactive-game';
  readonly archetype: 'turn-board';
  readonly rows: number;
  readonly cols: number;
  readonly winLength: number;
  readonly gravity: boolean;
  readonly vsCpu: boolean;
  readonly p1Label: string;
  readonly p2Label: string;
}

export interface ArcadeCanvasGameWeights {
  readonly mediaType: 'interactive-game';
  readonly archetype: 'arcade-canvas';
  readonly variant: 'snake' | 'breakout' | 'pong';
  readonly speed: number;
  readonly lives: number;
  readonly gridCols: number;
  readonly gridRows: number;
  readonly wrapWalls: boolean;
}

export interface CellularSimGameWeights {
  readonly mediaType: 'interactive-game';
  readonly archetype: 'cellular-sim';
  readonly variant: 'game-of-life' | 'rule-30';
  readonly rows: number;
  readonly cols: number;
  readonly ruleNumber: number;
  readonly tickMs: number;
}

export type InteractiveGameWeights =
  | GridSweepGameWeights
  | GridPuzzleGameWeights
  | TurnBoardGameWeights
  | ArcadeCanvasGameWeights
  | CellularSimGameWeights;

export interface ResearchHypothesisWeight {
  readonly id: string;
  readonly statement: string;
  readonly status: 'open' | 'supported' | 'refuted';
}

export interface ResearchMilestoneWeight {
  readonly title: string;
  readonly status: 'planned' | 'in-progress' | 'done';
  readonly dueLabel: string;
}

export interface ResearchEvidenceWeight {
  readonly hypothesisId: string;
  readonly verdict: 'supporting' | 'refuting' | 'observation';
  readonly text: string;
  readonly date?: string;
}

export interface ResearchProgramWeights {
  readonly mediaType: 'research-program';
  readonly topic: string;
  readonly question: string;
  readonly compartmentId: string;
  readonly tracks: readonly string[];
  readonly hypotheses: readonly ResearchHypothesisWeight[];
  readonly milestones: readonly ResearchMilestoneWeight[];
  readonly evidence?: readonly ResearchEvidenceWeight[];
}

export interface ScheduleWindowWeight {
  readonly label: string;
  readonly startMinuteOfDay: number;
  readonly endMinuteOfDay: number;
  readonly daysOfWeek: readonly number[]; // 0=Sun..6=Sat
  readonly protectedWindow: boolean;
}

export interface ToolOrganMetricWeight {
  readonly label: string;
  readonly target: number;
  readonly unit: string;
}

export interface ToolWorkflowStepWeight {
  readonly id: string;
  readonly text: string;
  readonly done: boolean;
}

export interface ToolOrganWeights {
  readonly mediaType: 'tool-organ';
  readonly role: 'habit-guardian' | 'schedule-guard' | 'calculator' | 'workflow-helper';
  readonly compartmentId: string;
  readonly emitsOpaqueConstraints: boolean;
  readonly rulesSummary: string;
  readonly scheduleWindows: readonly ScheduleWindowWeight[];
  readonly metrics: readonly ToolOrganMetricWeight[];
  readonly steps?: readonly ToolWorkflowStepWeight[];
}

export interface SpaceArtifactEntry {
  readonly id: string;
  readonly title: string;
  readonly mediaType: Exclude<PrismMediaType, 'space'>;
  readonly summary: string;
  readonly categories?: readonly string[];
  readonly previewDataUrl?: string;
}

export interface SpaceWeights {
  readonly mediaType: 'space';
  readonly title: string;
  readonly description: string;
  readonly layout: 'grid' | 'board' | 'showcase' | 'timeline';
  readonly accent: string;
  readonly items: readonly SpaceArtifactEntry[];
}

export type ProceduralOrganWeights =
  | PictureWeights
  | AudioWeights
  | WrittenWorkWeights
  | VideoWeights
  | InteractiveGameWeights
  | ResearchProgramWeights
  | ToolOrganWeights
  | SpaceWeights;

export interface ProceduralOrganSpec {
  readonly mediaType: PrismMediaType;
  readonly activeOrgans: readonly StudioFeatureOrgan[];
  readonly weights: ProceduralOrganWeights;
  readonly summary?: string;
}

// ============================================================================
// Safe Weight Clamping & Normalization
// ============================================================================

function clampInt(val: unknown, min: number, max: number, fallback: number): number {
  const n = typeof val === 'number' ? val : Number(val);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, Math.round(n)));
}

function safeStr(val: unknown, fallback: string, maxLen = 180): string {
  if (typeof val !== 'string') return fallback;
  const t = val.trim();
  return t ? t.slice(0, maxLen) : fallback;
}

function safeHex(val: unknown, fallback: string): string {
  if (typeof val === 'string' && /^#[0-9a-fA-F]{3,8}$/.test(val.trim())) {
    return val.trim();
  }
  return fallback;
}

export function inferGameArchetypeFromText(text: string): {
  readonly archetype: GameArchetype;
  readonly variant?: string;
} {
  const lower = text.toLowerCase();
  if (/\b(minesweeper|mine\s*sweep|sweep|mines|bomb\s*grid|hazard\s*grid)\b/.test(lower)) {
    return { archetype: 'grid-sweep' };
  }
  if (/\b(2048|merge\s*tile|sliding\s*number)\b/.test(lower)) {
    return { archetype: 'grid-puzzle', variant: 'merge-2048' };
  }
  if (/\b(memory\s*(?:match|game|cards?)|matching\s*pairs|concentration)\b/.test(lower)) {
    return { archetype: 'grid-puzzle', variant: 'memory-match' };
  }
  if (/\b(lights?\s*out|toggle\s*puzzle|tile\s*puzzle)\b/.test(lower)) {
    return { archetype: 'grid-puzzle', variant: 'lights-out' };
  }
  if (/\b(connect\s*4|connect\s*four)\b/.test(lower)) {
    return { archetype: 'turn-board', variant: 'connect-4' };
  }
  if (/\b(gomoku|five\s*in\s*a\s*row)\b/.test(lower)) {
    return { archetype: 'turn-board', variant: 'gomoku' };
  }
  if (/\b(tic[\s-]?tac[\s-]?toe|noughts?\s*and\s*crosses)\b/.test(lower)) {
    return { archetype: 'turn-board', variant: 'tic-tac-toe' };
  }
  if (/\b(snake|serpent|nibbles)\b/.test(lower)) {
    return { archetype: 'arcade-canvas', variant: 'snake' };
  }
  if (/\b(breakout|arkanoid|brick\s*breaker)\b/.test(lower)) {
    return { archetype: 'arcade-canvas', variant: 'breakout' };
  }
  if (/\b(pong|paddle)\b/.test(lower)) {
    return { archetype: 'arcade-canvas', variant: 'pong' };
  }
  if (/\b(game\s*of\s*life|conway|cellular|rule\s*30|automaton)\b/.test(lower)) {
    return { archetype: 'cellular-sim', variant: /\brule\s*30\b/.test(lower) ? 'rule-30' : 'game-of-life' };
  }
  return { archetype: 'grid-sweep' };
}

export function normalizeOrganWeights(
  mediaType: PrismMediaType,
  rawPatch: Record<string, unknown> | undefined,
  hintText = '',
  titleHint = 'My Prism',
): ProceduralOrganWeights {
  const p = rawPatch ?? {};
  const lowerHint = `${titleHint} ${hintText}`.toLowerCase();

  if (mediaType === 'space') {
    const allowedLayouts: readonly SpaceWeights['layout'][] = ['grid', 'board', 'showcase', 'timeline'];
    const layout = allowedLayouts.includes(p.layout as SpaceWeights['layout'])
      ? (p.layout as SpaceWeights['layout'])
      : /\b(board|kanban|columns?)\b/.test(lowerHint)
        ? 'board'
        : /\b(showcase|gallery|hero)\b/.test(lowerHint)
          ? 'showcase'
          : /\b(timeline|chronolog|journal)\b/.test(lowerHint)
            ? 'timeline'
            : 'grid';
    const rawItems = Array.isArray(p.items) ? p.items : [];
    const items: SpaceArtifactEntry[] = rawItems
      .filter((x): x is Record<string, unknown> => Boolean(x && typeof x === 'object'))
      .slice(0, 64)
      .map((it, idx) => ({
        id: safeStr(it.id, `item_${idx + 1}`, 80),
        title: safeStr(it.title, `Artifact ${idx + 1}`, 80),
        mediaType: (typeof it.mediaType === 'string' && it.mediaType !== 'space'
          ? it.mediaType
          : 'tool-organ') as SpaceArtifactEntry['mediaType'],
        summary: safeStr(it.summary, '', 200),
        categories: Array.isArray(it.categories)
          ? it.categories.map((c) => safeStr(c, '', 32)).filter(Boolean).slice(0, 4)
          : [],
        previewDataUrl: typeof it.previewDataUrl === 'string' ? it.previewDataUrl : undefined,
      }));
    return {
      mediaType: 'space',
      title: safeStr(p.title, titleHint, 100),
      description: safeStr(p.description, hintText || `Curated Verdant Space "${titleHint}".`, 240),
      layout,
      accent: safeHex(p.accent, '#0f5132'),
      items,
    };
  }

  if (mediaType === 'picture') {
    const inferredPic = inferPictureWeightsFromPrompt(hintText || titleHint, titleHint);
    const allowedGen: readonly PictureGeneratorKind[] = [
      'landscape-vista',
      'cosmic-nebula',
      'architectural-scene',
      'botanical-garden',
      'ocean-seascape',
      'abstract-flow',
      'geometric-mosaic',
      'constellation-atlas',
      'poster-editorial',
      'pixel-scene',
      'logo-mark',
      'cover-art',
      'botanical-fractal',
      'wave-field',
      'constellation',
      'pixel-sprite',
    ];
    const gen = allowedGen.includes(p.generator as PictureGeneratorKind)
      ? (p.generator as PictureGeneratorKind)
      : inferredPic.generator;
    const allowedLights: readonly PictureLightingMood[] = [
      'golden-hour',
      'twilight',
      'midnight',
      'daylight',
      'neon',
      'studio',
    ];
    const lighting = allowedLights.includes(p.lighting as PictureLightingMood)
      ? (p.lighting as PictureLightingMood)
      : inferredPic.lighting;
    const rawPal = Array.isArray(p.palette) ? p.palette : [];
    const palette: [string, string, string, string] = [
      safeHex(rawPal[0], inferredPic.palette[0]),
      safeHex(rawPal[1], inferredPic.palette[1]),
      safeHex(rawPal[2], inferredPic.palette[2]),
      safeHex(rawPal[3], inferredPic.palette[3]),
    ];
    const coverFormat =
      p.coverFormat === 'book' || p.coverFormat === 'album' || p.coverFormat === 'game'
        ? p.coverFormat
        : inferredPic.coverFormat;
    const picWeights: PictureWeights = {
      mediaType: 'picture',
      generator: gen,
      lighting,
      width: clampInt(p.width, 320, 1600, inferredPic.width),
      height: clampInt(p.height, 240, 1200, inferredPic.height),
      seed: clampInt(p.seed, 1, 999_999, inferredPic.seed),
      symmetry: clampInt(p.symmetry, 1, 16, inferredPic.symmetry),
      density: clampInt(p.density, 8, 120, inferredPic.density),
      palette,
      caption: safeStr(p.caption, titleHint, 140),
      ...(typeof p.subtitle === 'string' && p.subtitle.trim() ? { subtitle: safeStr(p.subtitle, '', 160) } : {}),
      ...(coverFormat ? { coverFormat } : {}),
    };
    return gen === 'logo-mark'
      ? { ...picWeights, svgMarkup: typeof p.svgMarkup === 'string' && p.svgMarkup.includes('<svg') ? p.svgMarkup : synthesizeLogoSvg(picWeights) }
      : picWeights;
  }

  if (mediaType === 'audio') {
    const allowedStyles: readonly AudioWeights['style'][] = [
      'ambient-pad',
      'melodic-arpeggio',
      'rhythm-pulse',
      'sound-effects',
    ];
    let defaultStyle: AudioWeights['style'] = 'ambient-pad';
    if (/\b(beat|drum|rhythm|techno|pulse|lo-?fi)\b/.test(lowerHint)) defaultStyle = 'rhythm-pulse';
    else if (/\b(melody|arpeggio|piano|synth|song|tune|chiptune)\b/.test(lowerHint)) defaultStyle = 'melodic-arpeggio';
    else if (/\b(sfx|effect|chime|alert|bell|sound\s+effect)\b/.test(lowerHint)) defaultStyle = 'sound-effects';
    const style = allowedStyles.includes(p.style as AudioWeights['style'])
      ? (p.style as AudioWeights['style'])
      : defaultStyle;
    const allowedScales: readonly AudioWeights['scale'][] = ['pentatonic', 'major', 'minor', 'dorian'];
    const scale = allowedScales.includes(p.scale as AudioWeights['scale'])
      ? (p.scale as AudioWeights['scale'])
      : /\b(dark|minor|moody|noir|nocturnal|melanchol)\b/.test(lowerHint)
        ? 'minor'
        : /\b(dorian|mystic|twilight|alpine)\b/.test(lowerHint)
          ? 'dorian'
          : /\b(major|bright|sunny|uplifting|happy)\b/.test(lowerHint)
            ? 'major'
            : 'pentatonic';
    const bpmMatch = lowerHint.match(/\b(\d{2,3})\s*bpm\b/);
    const inferredBpm = bpmMatch
      ? Number(bpmMatch[1])
      : style === 'ambient-pad'
        ? 76
        : style === 'rhythm-pulse'
          ? 118
          : 104;
    return {
      mediaType: 'audio',
      style,
      bpm: clampInt(p.bpm, 45, 200, inferredBpm),
      rootMidi: clampInt(p.rootMidi, 36, 72, 60),
      scale,
      durationSec: clampInt(p.durationSec, 4, 30, 8),
      title: safeStr(p.title, titleHint, 100),
    };
  }

  if (mediaType === 'written-work') {
    const rawChapters = Array.isArray(p.chapters) ? p.chapters : [];
    const chapters: WrittenChapterWeight[] =
      rawChapters.length > 0
        ? rawChapters.slice(0, 24).map((ch, i) => {
            const c = (ch ?? {}) as Record<string, unknown>;
            return {
              title: safeStr(c.title, `Chapter ${i + 1}`, 80),
              beat: safeStr(c.beat, i === 0 ? 'Opening' : i === rawChapters.length - 1 ? 'Resolution' : 'Development', 40),
              summary: safeStr(c.summary, 'Draft the core moment of this chapter.', 240),
              body: safeStr(c.body, '', 4000),
            };
          })
        : [
            {
              title: 'Chapter 1: The Inciting Spark',
              beat: 'Opening',
              summary: 'Introduce the protagonist, setting, and the central tension.',
              body: '',
            },
            {
              title: 'Chapter 2: Rising Stakes',
              beat: 'Escalation',
              summary: 'Deepen the world and confront the first obstacle.',
              body: '',
            },
            {
              title: 'Chapter 3: The Turning Point',
              beat: 'Climax & Resolution',
              summary: 'Resolve the core conflict and reveal what changed.',
              body: '',
            },
          ];
    return {
      mediaType: 'written-work',
      genre: safeStr(p.genre, /\b(sci-?fi|space)\b/.test(lowerHint) ? 'Science Fiction' : /\b(mystery|detective)\b/.test(lowerHint) ? 'Mystery' : 'Fiction', 48),
      title: safeStr(p.title, titleHint, 100),
      premise: safeStr(p.premise, hintText || `An original written work titled "${titleHint}".`, 280),
      voice: safeStr(p.voice, 'Clear, vivid, and engaging', 80),
      targetWordCount: clampInt(p.targetWordCount, 500, 200_000, 25_000),
      allowLocalAssist: p.allowLocalAssist !== false,
      chapters,
    };
  }

  if (mediaType === 'video') {
    const rawScenes = Array.isArray(p.scenes) ? p.scenes : [];
    const motions = ['pan', 'pulse', 'orbit', 'rise'] as const;
    const scenes: VideoSceneWeight[] =
      rawScenes.length > 0
        ? rawScenes.slice(0, 12).map((sc, i) => {
            const s = (sc ?? {}) as Record<string, unknown>;
            const mot = motions.includes(s.motion as (typeof motions)[number])
              ? (s.motion as VideoSceneWeight['motion'])
              : motions[i % motions.length];
            return {
              title: safeStr(s.title, `Scene ${i + 1}`, 70),
              subtitle: safeStr(s.subtitle, 'Procedural motion beat', 140),
              durationSec: clampInt(s.durationSec, 1, 20, 4),
              motion: mot,
              accent: safeHex(s.accent, i % 2 === 0 ? '#10b981' : '#f59e0b'),
            };
          })
        : [
            { title: titleHint, subtitle: 'Opening visual sequence', durationSec: 4, motion: 'rise', accent: '#10b981' },
            { title: 'Core Theme', subtitle: 'Procedural motion & rhythm', durationSec: 4, motion: 'orbit', accent: '#38bdf8' },
            { title: 'Finale', subtitle: 'Closing frame & cadence', durationSec: 4, motion: 'pulse', accent: '#f59e0b' },
          ];
    const ar = p.aspectRatio === '1:1' || p.aspectRatio === '9:16' ? p.aspectRatio : '16:9';
    const style =
      p.style === 'abstract-motion' || p.style === 'storyboard-scenes' ? p.style : 'kinetic-type';
    const rawPal = Array.isArray(p.palette) ? p.palette : [];
    return {
      mediaType: 'video',
      aspectRatio: ar,
      fps: clampInt(p.fps, 15, 60, 30),
      style,
      palette: [
        safeHex(rawPal[0], '#0b0f14'),
        safeHex(rawPal[1], '#10b981'),
        safeHex(rawPal[2], '#f8fafc'),
      ],
      scenes,
    };
  }

  if (mediaType === 'interactive-game') {
    const inferred = inferGameArchetypeFromText(lowerHint);
    const allowedArch: readonly GameArchetype[] = [
      'grid-sweep',
      'grid-puzzle',
      'turn-board',
      'arcade-canvas',
      'cellular-sim',
    ];
    const archetype: GameArchetype = allowedArch.includes(p.archetype as GameArchetype)
      ? (p.archetype as GameArchetype)
      : inferred.archetype;

    if (archetype === 'grid-sweep') {
      // Parse explicit NxM and mines from hintText if present (e.g. "16x16 with 40 mines")
      const dimMatch = lowerHint.match(/\b(\d{1,2})\s*[x×]\s*(\d{1,2})\b/);
      const mineMatch = lowerHint.match(/\b(\d{1,3})\s*(?:mines?|bombs?|hazards?)\b/);
      const rows = clampInt(p.rows ?? (dimMatch ? Number(dimMatch[1]) : 9), 5, 24, 9);
      const cols = clampInt(p.cols ?? (dimMatch ? Number(dimMatch[2]) : 9), 5, 30, 9);
      const maxMines = Math.max(1, rows * cols - 9);
      const defaultMines = Math.max(1, Math.min(maxMines, Math.round(rows * cols * 0.14)));
      const mines = clampInt(p.mines ?? (mineMatch ? Number(mineMatch[1]) : defaultMines), 1, maxMines, defaultMines);
      const rawPresets = Array.isArray(p.presets) ? p.presets : [];
      const presets: GridSweepPreset[] =
        rawPresets.length > 0
          ? rawPresets.slice(0, 4).map((pr, idx) => {
              const obj = (pr ?? {}) as Record<string, unknown>;
              const r = clampInt(obj.rows, 5, 24, 9);
              const c = clampInt(obj.cols, 5, 30, 9);
              return {
                label: safeStr(obj.label, `Preset ${idx + 1}`, 24),
                rows: r,
                cols: c,
                mines: clampInt(obj.mines, 1, Math.max(1, r * c - 9), Math.round(r * c * 0.14)),
              };
            })
          : [
              { label: 'Beginner', rows: 9, cols: 9, mines: 10 },
              { label: 'Intermediate', rows: 16, cols: 16, mines: 40 },
              { label: 'Expert', rows: 16, cols: 24, mines: 72 },
            ];
      const visualStyle =
        p.visualStyle === 'classic' || p.visualStyle === 'retro' ? p.visualStyle : 'modern';
      return {
        mediaType: 'interactive-game',
        archetype: 'grid-sweep',
        rows,
        cols,
        mines,
        presets,
        firstClickSafe: p.firstClickSafe !== false,
        floodReveal: p.floodReveal !== false,
        chordReveal: p.chordReveal !== false,
        showTimer: p.showTimer !== false,
        visualStyle,
      };
    }

    if (archetype === 'grid-puzzle') {
      const variant =
        p.variant === 'memory-match' || p.variant === 'lights-out' || p.variant === 'merge-2048'
          ? p.variant
          : ((inferred.variant as GridPuzzleGameWeights['variant']) || 'merge-2048');
      return {
        mediaType: 'interactive-game',
        archetype: 'grid-puzzle',
        variant,
        rows: clampInt(p.rows, 3, 6, variant === 'lights-out' ? 5 : 4),
        cols: clampInt(p.cols, 3, 6, variant === 'lights-out' ? 5 : 4),
        targetScore: clampInt(p.targetScore, 16, 65536, 2048),
      };
    }

    if (archetype === 'turn-board') {
      const isConnect4 = inferred.variant === 'connect-4' || p.gravity === true;
      const isGomoku = inferred.variant === 'gomoku';
      return {
        mediaType: 'interactive-game',
        archetype: 'turn-board',
        rows: clampInt(p.rows, 3, 12, isConnect4 ? 6 : isGomoku ? 9 : 3),
        cols: clampInt(p.cols, 3, 12, isConnect4 ? 7 : isGomoku ? 9 : 3),
        winLength: clampInt(p.winLength, 3, 5, isConnect4 ? 4 : isGomoku ? 5 : 3),
        gravity: Boolean(p.gravity ?? isConnect4),
        vsCpu: p.vsCpu !== false,
        p1Label: safeStr(p.p1Label, 'You', 20),
        p2Label: safeStr(p.p2Label, p.vsCpu === false ? 'Player 2' : 'Computer', 20),
      };
    }

    if (archetype === 'arcade-canvas') {
      const variant =
        p.variant === 'breakout' || p.variant === 'pong' || p.variant === 'snake'
          ? p.variant
          : ((inferred.variant as ArcadeCanvasGameWeights['variant']) || 'snake');
      return {
        mediaType: 'interactive-game',
        archetype: 'arcade-canvas',
        variant,
        speed: clampInt(p.speed, 1, 10, 5),
        lives: clampInt(p.lives, 1, 9, 3),
        gridCols: clampInt(p.gridCols, 12, 32, 20),
        gridRows: clampInt(p.gridRows, 12, 24, 16),
        wrapWalls: Boolean(p.wrapWalls),
      };
    }

    return {
      mediaType: 'interactive-game',
      archetype: 'cellular-sim',
      variant: p.variant === 'rule-30' || inferred.variant === 'rule-30' ? 'rule-30' : 'game-of-life',
      rows: clampInt(p.rows, 12, 48, 24),
      cols: clampInt(p.cols, 16, 64, 36),
      ruleNumber: clampInt(p.ruleNumber, 0, 255, 30),
      tickMs: clampInt(p.tickMs, 40, 600, 140),
    };
  }

  if (mediaType === 'research-program') {
    const rawH = Array.isArray(p.hypotheses) ? p.hypotheses : [];
    const hypotheses: ResearchHypothesisWeight[] =
      rawH.length > 0
        ? rawH.slice(0, 12).map((h, i) => {
            const obj = (h ?? {}) as Record<string, unknown>;
            const st = obj.status === 'supported' || obj.status === 'refuted' ? obj.status : 'open';
            return {
              id: safeStr(obj.id, `H${i + 1}`, 16),
              statement: safeStr(obj.statement, `Hypothesis ${i + 1}`, 220),
              status: st,
            };
          })
        : [
            { id: 'H1', statement: `Primary hypothesis for ${titleHint}`, status: 'open' },
            { id: 'H2', statement: 'Falsifiable baseline comparison', status: 'open' },
          ];
    const rawM = Array.isArray(p.milestones) ? p.milestones : [];
    const milestones: ResearchMilestoneWeight[] =
      rawM.length > 0
        ? rawM.slice(0, 12).map((m, i) => {
            const obj = (m ?? {}) as Record<string, unknown>;
            const st = obj.status === 'in-progress' || obj.status === 'done' ? obj.status : 'planned';
            return {
              title: safeStr(obj.title, `Milestone ${i + 1}`, 100),
              status: st,
              dueLabel: safeStr(obj.dueLabel, `Week ${i + 1}`, 32),
            };
          })
        : [
            { title: 'Literature & baseline synthesis', status: 'in-progress', dueLabel: 'Week 1' },
            { title: 'Controlled evaluation & evidence log', status: 'planned', dueLabel: 'Week 2' },
          ];
    const tracks = Array.isArray(p.tracks)
      ? p.tracks.map((t) => safeStr(t, '', 48)).filter(Boolean).slice(0, 8)
      : ['Core Theory', 'Experiments', 'Synthesis'];
    const rawEvidence = Array.isArray(p.evidence) ? p.evidence : [];
    const evidence: ResearchEvidenceWeight[] = rawEvidence.slice(0, 24).map((e) => {
      const obj = (e ?? {}) as Record<string, unknown>;
      const v: ResearchEvidenceWeight['verdict'] =
        obj.verdict === 'supporting' || obj.verdict === 'refuting' ? obj.verdict : 'observation';
      return {
        hypothesisId: safeStr(obj.hypothesisId, hypotheses[0]?.id ?? 'General', 24),
        verdict: v,
        text: safeStr(obj.text, '', 360),
        ...(typeof obj.date === 'string' && obj.date.trim() ? { date: obj.date.trim().slice(0, 32) } : {}),
      };
    }).filter((e) => e.text.length > 0);
    return {
      mediaType: 'research-program',
      topic: safeStr(p.topic, titleHint, 120),
      question: safeStr(p.question, hintText || `Investigate and validate ${titleHint}.`, 260),
      compartmentId: safeStr(p.compartmentId, 'research', 32),
      tracks,
      hypotheses,
      milestones,
      ...(evidence.length > 0 ? { evidence } : {}),
    };
  }

  // tool-organ (habit guardian, schedule guard, calculator, workflow helper)
  const isHabit = /\b(habit|healthy|workout|exercise|meditat|sleep|walk|run|wellness|gym|hydrat)\b/.test(lowerHint);
  const isSchedule = /\b(calendar|schedule|meeting|quiet|focus|pomodoro|block\s*time|protect)\b/.test(lowerHint);
  const role: ToolOrganWeights['role'] =
    p.role === 'habit-guardian' ||
    p.role === 'schedule-guard' ||
    p.role === 'calculator' ||
    p.role === 'workflow-helper'
      ? p.role
      : isHabit
        ? 'habit-guardian'
        : isSchedule
          ? 'schedule-guard'
          : 'workflow-helper';

  // Extract time window like "5pm to 6pm" or "17:00-18:00" if mentioned in natural language
  const timeMatch = lowerHint.match(
    /\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:to|-|–|until)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b/,
  );
  let startMin = 17 * 60;
  let endMin = 18 * 60;
  if (timeMatch) {
    const toMin = (hRaw: string, mRaw: string | undefined, mer: string | undefined): number => {
      let h = Number(hRaw) % 24;
      const m = mRaw ? Number(mRaw) % 60 : 0;
      if (mer === 'pm' && h < 12) h += 12;
      if (mer === 'am' && h === 12) h = 0;
      return h * 60 + m;
    };
    const s = toMin(timeMatch[1], timeMatch[2], timeMatch[3] || timeMatch[6]);
    const e = toMin(timeMatch[4], timeMatch[5], timeMatch[6] || timeMatch[3]);
    if (e > s) {
      startMin = s;
      endMin = e;
    }
  }

  const rawWindows = Array.isArray(p.scheduleWindows) ? p.scheduleWindows : [];
  const scheduleWindows: ScheduleWindowWeight[] =
    rawWindows.length > 0
      ? rawWindows.slice(0, 8).map((w, i) => {
          const obj = (w ?? {}) as Record<string, unknown>;
          const sMin = clampInt(obj.startMinuteOfDay, 0, 1430, startMin);
          const eMin = clampInt(obj.endMinuteOfDay, sMin + 5, 1440, Math.max(sMin + 30, endMin));
          const days = Array.isArray(obj.daysOfWeek)
            ? obj.daysOfWeek.map((d) => clampInt(d, 0, 6, 1))
            : [1, 2, 3, 4, 5];
          return {
            label: safeStr(obj.label, `Protected Window ${i + 1}`, 60),
            startMinuteOfDay: sMin,
            endMinuteOfDay: eMin,
            daysOfWeek: days,
            protectedWindow: obj.protectedWindow !== false,
          };
        })
      : isHabit || isSchedule
        ? [
            {
              label: isHabit ? 'Daily Healthy Habit' : 'Protected Focus Window',
              startMinuteOfDay: startMin,
              endMinuteOfDay: endMin,
              daysOfWeek: [0, 1, 2, 3, 4, 5, 6],
              protectedWindow: true,
            },
          ]
        : [];

  const rawMetrics = Array.isArray(p.metrics) ? p.metrics : [];
  const metrics: ToolOrganMetricWeight[] =
    rawMetrics.length > 0
      ? rawMetrics.slice(0, 6).map((m, i) => {
          const obj = (m ?? {}) as Record<string, unknown>;
          return {
            label: safeStr(obj.label, `Goal ${i + 1}`, 48),
            target: clampInt(obj.target, 1, 100_000, 1),
            unit: safeStr(obj.unit, 'sessions', 20),
          };
        })
      : [{ label: isHabit ? 'Daily streak' : 'Completed items', target: 1, unit: 'today' }];

  const rawSteps = Array.isArray(p.steps) ? p.steps : [];
  const steps: ToolWorkflowStepWeight[] = rawSteps
    .slice(0, 24)
    .map((s, i) => {
      const obj = (s ?? {}) as Record<string, unknown>;
      return {
        id: safeStr(obj.id, `s${i + 1}`, 24),
        text: safeStr(obj.text, '', 180),
        done: Boolean(obj.done),
      };
    })
    .filter((s) => s.text.length > 0);

  return {
    mediaType: 'tool-organ',
    role,
    compartmentId: safeStr(p.compartmentId, isHabit ? 'personal' : 'work', 32),
    emitsOpaqueConstraints: Boolean(p.emitsOpaqueConstraints ?? (isHabit || isSchedule)),
    rulesSummary: safeStr(
      p.rulesSummary,
      hintText || `Local tool organ "${titleHint}" with privacy-preserving defaults.`,
      220,
    ),
    scheduleWindows,
    metrics,
    ...(steps.length > 0 ? { steps } : {}),
  };
}

// ============================================================================
// Audited Procedural Organ View Synthesizers (Zero LLM-Written JavaScript!)
// ============================================================================

function escHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

const PROCEDURAL_BASE_CSS = `
:root {
  --v-bg: #f8f9fa;
  --v-surface: #ffffff;
  --v-surface-alt: #f1f5f9;
  --v-border: #e2e8f0;
  --v-text: #0f172a;
  --v-muted: #64748b;
  --v-accent: #0f5132;
  --v-accent-text: #ffffff;
  color-scheme: light;
}
:root.dark, :root[data-theme="dark"] {
  --v-bg: #0b0f14;
  --v-surface: #0f141c;
  --v-surface-alt: #141b26;
  --v-border: #1e293b;
  --v-text: #f1f5f9;
  --v-muted: #94a3b8;
  --v-accent: #10b981;
  --v-accent-text: #062116;
  color-scheme: dark;
}
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0; min-height: 100%;
  background: var(--v-bg); color: var(--v-text);
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 14px; line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}
.organ-shell { max-width: 960px; margin: 0 auto; padding: 20px 20px 28px; display: flex; flex-direction: column; gap: 14px; }
.organ-shell.organ-editorial { max-width: 780px; padding: 24px 24px 40px; }
.organ-card { background: var(--v-surface); border: 1px solid var(--v-border); border-radius: 16px; padding: 20px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03); }
.organ-card.organ-card-editorial { background: var(--v-surface); border: 1px solid var(--v-border); border-radius: 18px; padding: 28px 36px 24px; min-height: 68vh; display: flex; flex-direction: column; }
@media (max-width: 640px) { .organ-card.organ-card-editorial { padding: 18px 18px 20px; } }
.organ-sub { margin: 0; font-size: 12px; color: var(--v-muted); }
.organ-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.organ-btn { display: inline-flex; align-items: center; justify-content: center; gap: 6px; padding: 6px 12px; border-radius: 8px; border: 1px solid var(--v-border); background: var(--v-surface); color: var(--v-text); font: inherit; font-size: 12px; font-weight: 500; cursor: pointer; transition: background 120ms ease, border-color 120ms ease, opacity 120ms ease; }
.organ-btn:hover { background: var(--v-surface-alt); }
.organ-btn.primary { background: var(--v-accent); color: var(--v-accent-text); border-color: transparent; font-weight: 600; }
.organ-btn.primary:hover { opacity: 0.92; background: var(--v-accent); }
.organ-btn.ghost { border-color: transparent; background: transparent; color: var(--v-muted); padding: 5px 9px; }
.organ-btn.ghost:hover { background: var(--v-surface-alt); color: var(--v-text); }
.organ-pill { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px; border: 1px solid var(--v-border); background: var(--v-surface-alt); color: var(--v-muted); font-size: 11px; font-weight: 500; font-variant-numeric: tabular-nums; }
.organ-input { padding: 7px 11px; border-radius: 8px; border: 1px solid var(--v-border); background: var(--v-surface); color: var(--v-text); font: inherit; font-size: 13px; }
.organ-input:focus { outline: none; border-color: var(--v-accent); }
.organ-seg { display: inline-flex; padding: 2px; border-radius: 9px; border: 1px solid var(--v-border); background: var(--v-surface-alt); gap: 2px; }
.organ-seg button { border: 0; background: transparent; color: var(--v-muted); padding: 4px 10px; border-radius: 7px; font: inherit; font-size: 11px; font-weight: 500; cursor: pointer; transition: background 120ms ease, color 120ms ease; }
.organ-seg button.active { background: var(--v-surface); color: var(--v-text); font-weight: 600; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06); }
.doc-topbar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; padding-bottom: 16px; margin-bottom: 20px; border-bottom: 1px solid var(--v-border); }
.doc-chapters { display: flex; flex-wrap: wrap; align-items: center; gap: 4px; }
.doc-ch-pill { display: inline-flex; align-items: center; gap: 5px; border: 0; background: transparent; color: var(--v-muted); padding: 5px 10px; border-radius: 8px; font: inherit; font-size: 12px; font-weight: 500; cursor: pointer; transition: background 120ms ease, color 120ms ease; }
.doc-ch-pill:hover { background: var(--v-surface-alt); color: var(--v-text); }
.doc-ch-pill.active { background: var(--v-surface-alt); color: var(--v-text); font-weight: 600; }
.doc-title-input { width: 100%; border: 0; background: transparent; color: var(--v-text); font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 28px; font-weight: 700; letter-spacing: -0.025em; line-height: 1.25; padding: 0; margin: 0 0 8px; outline: none; }
.doc-title-input::placeholder { color: var(--v-muted); opacity: 0.55; }
.doc-sub-input { width: 100%; border: 0; background: transparent; color: var(--v-muted); font-family: Georgia, Charter, "Iowan Old Style", ui-serif, serif; font-style: italic; font-size: 14.5px; line-height: 1.5; padding: 0 0 18px; margin: 0 0 18px; border-bottom: 1px solid var(--v-border); outline: none; }
.doc-sub-input::placeholder { color: var(--v-muted); opacity: 0.5; }
.doc-editor { width: 100%; flex: 1; min-height: 340px; border: 0; background: transparent; color: var(--v-text); font-family: Georgia, Charter, "Iowan Old Style", ui-serif, serif; font-size: 16.5px; line-height: 1.85; padding: 0; resize: none; outline: none; }
.doc-editor::placeholder { color: var(--v-muted); opacity: 0.5; }
.doc-prose { flex: 1; min-height: 340px; font-family: Georgia, Charter, "Iowan Old Style", ui-serif, serif; font-size: 16.5px; line-height: 1.85; color: var(--v-text); }
.doc-prose h1 { font-family: Inter, -apple-system, sans-serif; font-size: 24px; font-weight: 700; letter-spacing: -0.02em; margin: 24px 0 10px; }
.doc-prose h2 { font-family: Inter, -apple-system, sans-serif; font-size: 19px; font-weight: 650; letter-spacing: -0.015em; margin: 22px 0 8px; }
.doc-prose h3 { font-family: Inter, -apple-system, sans-serif; font-size: 16px; font-weight: 600; margin: 18px 0 6px; }
.doc-prose p { margin: 0 0 16px; }
.doc-prose blockquote { margin: 16px 0; padding: 2px 0 2px 16px; border-left: 3px solid var(--v-accent); color: var(--v-muted); font-style: italic; }
.doc-prose ul, .doc-prose ol { margin: 0 0 16px; padding-left: 22px; }
.doc-prose li { margin-bottom: 6px; }
.doc-prose code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.88em; padding: 2px 5px; border-radius: 5px; background: var(--v-surface-alt); }
.doc-prose hr { border: 0; border-top: 1px solid var(--v-border); margin: 24px 0; }
.doc-footer { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; margin-top: 24px; padding-top: 14px; border-top: 1px solid var(--v-border); font-size: 11.5px; color: var(--v-muted); }
.doc-progress { width: 88px; height: 4px; border-radius: 999px; background: var(--v-surface-alt); overflow: hidden; display: inline-block; vertical-align: middle; margin-left: 8px; }
.doc-progress > i { display: block; height: 100%; background: var(--v-accent); border-radius: 999px; }
.doc-menu { position: relative; display: inline-flex; }
.doc-popover { position: absolute; right: 0; top: calc(100% + 6px); z-index: 30; min-width: 156px; padding: 5px; border-radius: 10px; border: 1px solid var(--v-border); background: var(--v-surface); box-shadow: 0 10px 28px rgba(15, 23, 42, 0.12); display: flex; flex-direction: column; gap: 2px; }
.doc-popover button { width: 100%; justify-content: flex-start; border: 0; background: transparent; padding: 7px 10px; border-radius: 7px; font-size: 12px; font-weight: 500; color: var(--v-text); cursor: pointer; text-align: left; }
.doc-popover button:hover { background: var(--v-surface-alt); }
.ms-grid { display: grid; gap: 3px; justify-content: center; margin: 0 auto; user-select: none; touch-action: manipulation; }
.ms-cell { width: 30px; height: 30px; border-radius: 6px; border: 1px solid var(--v-border); background: var(--v-surface-alt); color: var(--v-text); font-weight: 700; font-size: 13px; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: transform 60ms ease, background 80ms ease; }
.ms-cell:hover:not(.rev) { filter: brightness(1.06); }
.ms-cell.rev { background: var(--v-bg); border-color: transparent; cursor: default; }
.ms-cell.mine { background: #dc2626; color: #fff; }
.ms-cell.flag { background: rgba(245, 158, 11, 0.22); border-color: #f59e0b; }
.ms-n1 { color: #2563eb; } .ms-n2 { color: #16a34a; } .ms-n3 { color: #dc2626; } .ms-n4 { color: #7c3aed; }
.ms-n5 { color: #b45309; } .ms-n6 { color: #0891b2; } .ms-n7 { color: #475569; } .ms-n8 { color: #94a3b8; }
canvas.organ-canvas { display: block; width: 100%; height: auto; border-radius: 12px; border: 1px solid var(--v-border); background: #0b0f14; }
.organ-textarea { width: 100%; min-height: 220px; padding: 14px; border-radius: 10px; border: 1px solid var(--v-border); background: var(--v-bg); color: var(--v-text); font: inherit; font-size: 14px; line-height: 1.7; resize: vertical; }
`;


export function synthesizeProceduralOrganView(
  decl: PrismDeclaration,
  spec: ProceduralOrganSpec,
): {
  readonly indexHtml: string;
  readonly stylesCss: string;
  readonly appJs: string;
} {
  const title = decl.view?.title || decl.metadata.name || 'Verdant Prism';
  const accent = decl.theme.accent || '#0f5132';
  const isEditorial = spec.weights.mediaType === 'written-work';

  let runtimeJs: string;
  const w = spec.weights;
  if (w.mediaType === 'picture') {
    runtimeJs = buildPictureRuntimeJs(w);
  } else if (w.mediaType === 'audio') {
    runtimeJs = buildAudioRuntimeJs(w);
  } else if (w.mediaType === 'written-work') {
    runtimeJs = buildWrittenWorkRuntimeJs(w);
  } else if (w.mediaType === 'video') {
    runtimeJs = buildVideoRuntimeJs(w);
  } else if (w.mediaType === 'research-program') {
    runtimeJs = buildResearchProgramRuntimeJs(w);
  } else if (w.mediaType === 'tool-organ') {
    runtimeJs = buildToolOrganRuntimeJs(w);
  } else if (w.mediaType === 'space') {
    runtimeJs = buildSpaceRuntimeJs(w);
  } else if (w.archetype === 'grid-sweep') {
    runtimeJs = buildGridSweepRuntimeJs(w);
  } else if (w.archetype === 'turn-board') {
    runtimeJs = buildTurnBoardRuntimeJs(w);
  } else if (w.archetype === 'grid-puzzle') {
    runtimeJs = buildGridPuzzleRuntimeJs(w);
  } else if (w.archetype === 'arcade-canvas') {
    runtimeJs = buildArcadeCanvasRuntimeJs(w);
  } else {
    runtimeJs = buildCellularSimRuntimeJs(w);
  }

  const disclosureHtml = decl.view?.disclosure?.trim()
    ? `\n    <footer class="organ-sub" style="text-align:center">${escHtml(decl.view.disclosure.trim())}</footer>`
    : '';

  const shellClass = isEditorial ? 'organ-shell organ-editorial' : 'organ-shell';
  const cardClass = isEditorial ? 'organ-card organ-card-editorial' : 'organ-card';

  const indexHtml = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>${escHtml(title)}</title>
  <link rel="icon" href="data:,">
  <link rel="stylesheet" href="styles.css">
  <script type="module" src="prism.js"></script>
  <script type="module" src="app.js"></script>
</head>
<body>
  <main class="${shellClass}">
    <section class="${cardClass}" id="organ-root"></section>${disclosureHtml}
  </main>
</body>
</html>
`;

  const stylesCss = `${PROCEDURAL_BASE_CSS}\n:root { --v-accent: ${accent}; }\n`;
  const appJs = `// Procedurally synthesized by Verdant Studio (zero LLM JS).\n${runtimeJs}\n`;
  return { indexHtml, stylesCss, appJs };
}
