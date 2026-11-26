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

export type PrismMediaType =
  | 'picture'
  | 'audio'
  | 'written-work'
  | 'video'
  | 'interactive-game'
  | 'research-program'
  | 'tool-organ';

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

export type MediaModelModality = 'image' | 'audio' | 'video' | 'text' | 'embedding';

/**
 * Declares an imported local or catalog open-weight model routed onto a specific procedural Prism organ
 * (100% browser-local via OPFS / vTensor WASM SIMD — zero external API servers).
 */
export interface MediaModelWeightBinding {
  readonly id: string;
  readonly label: string;
  readonly modality: MediaModelModality;
  readonly targetOrgan: StudioFeatureOrgan;
  readonly source: 'catalog' | 'opfs-local' | 'file-import';
  readonly modelId: string;
  readonly format: 'gguf' | 'safetensors' | 'bin';
  /** Exact Q10 routing influence weight in [0..1024]. */
  readonly routingWeightQ10: number;
  readonly approximateBytes?: number;
  readonly digest?: string;
}

/**
 * Factual registry of compatible open-weight multimodal models users can import into OPFS and route
 * onto Verdant Studio's procedural Picture, Audio, Video, Game, and Written-Work organs.
 */
export const MULTIMODAL_WEIGHT_PRESETS: readonly MediaModelWeightBinding[] = [
  {
    id: 'img-flux-schnell-q4',
    label: 'FLUX.1-schnell (Q4_K_S Image Weights)',
    modality: 'image',
    targetOrgan: 'visual-canvas',
    source: 'catalog',
    modelId: 'flux-1-schnell-q4_k_s',
    format: 'gguf',
    routingWeightQ10: 896,
    approximateBytes: 6_800_000_000,
  },
  {
    id: 'img-sd35-medium-q4',
    label: 'SD 3.5 Medium (Q4_K_M Image Weights)',
    modality: 'image',
    targetOrgan: 'visual-canvas',
    source: 'catalog',
    modelId: 'sd-3.5-medium-q4_k_m',
    format: 'gguf',
    routingWeightQ10: 832,
    approximateBytes: 1_900_000_000,
  },
  {
    id: 'aud-musicgen-small-q8',
    label: 'MusicGen-Small (Q8_0 Audio Weights)',
    modality: 'audio',
    targetOrgan: 'audio-synth',
    source: 'catalog',
    modelId: 'musicgen-small-q8_0',
    format: 'gguf',
    routingWeightQ10: 896,
    approximateBytes: 420_000_000,
  },
  {
    id: 'aud-whisper-turbo-q5',
    label: 'Whisper Large v3 Turbo (Q5_0 Audio Encoder)',
    modality: 'audio',
    targetOrgan: 'audio-synth',
    source: 'catalog',
    modelId: 'whisper-large-v3-turbo-q5_0',
    format: 'gguf',
    routingWeightQ10: 768,
    approximateBytes: 574_000_000,
  },
  {
    id: 'vid-wan21-1.3b-q4',
    label: 'Wan 2.1 T2V 1.3B (Q4_K_M Video Weights)',
    modality: 'video',
    targetOrgan: 'timeline-motion',
    source: 'catalog',
    modelId: 'wan2.1-t2v-1.3b-q4_k_m',
    format: 'gguf',
    routingWeightQ10: 896,
    approximateBytes: 1_150_000_000,
  },
];

export function compatibleModalitiesForMediaType(mediaType: PrismMediaType): readonly MediaModelModality[] {
  switch (mediaType) {
    case 'picture':
      return ['image'];
    case 'audio':
      return ['audio'];
    case 'video':
      return ['video', 'image', 'audio'];
    case 'interactive-game':
      return ['image', 'audio', 'text'];
    case 'written-work':
      return ['text'];
    case 'research-program':
      return ['text', 'embedding'];
    case 'tool-organ':
      return ['text', 'audio', 'embedding'];
  }
}

export function createImportedWeightBinding(params: {
  readonly fileName: string;
  readonly byteLength: number;
  readonly modality: MediaModelModality;
  readonly digest?: string;
}): MediaModelWeightBinding {
  const cleanName = params.fileName.trim() || 'local-weights.gguf';
  const format: MediaModelWeightBinding['format'] = /\.safetensors$/i.test(cleanName)
    ? 'safetensors'
    : /\.gguf$/i.test(cleanName)
      ? 'gguf'
      : 'bin';
  const targetOrgan: StudioFeatureOrgan =
    params.modality === 'image'
      ? 'visual-canvas'
      : params.modality === 'audio'
        ? 'audio-synth'
        : params.modality === 'video'
          ? 'timeline-motion'
          : 'narrative-outline';
  const slug = cleanName.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 48);
  return {
    id: `local-${params.modality}-${slug || 'model'}`,
    label: cleanName,
    modality: params.modality,
    targetOrgan,
    source: 'file-import',
    modelId: slug || 'local-model',
    format,
    routingWeightQ10: 896,
    approximateBytes: Math.max(0, Math.round(params.byteLength)),
    digest: params.digest,
  };
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
    shortSummary: 'Procedural artwork & visual composition with PNG export & optional image weights',
    activeOrgans: ['visual-canvas', 'palette-theme'],
    activeWizardQuestions: ['visual style & mood', 'color palette & symmetry'],
  },
  audio: {
    mediaType: 'audio',
    label: 'Audio',
    shortSummary: 'Procedural Web Audio synthesizer, soundscape & WAV export with optional audio weights',
    activeOrgans: ['audio-synth', 'palette-theme'],
    activeWizardQuestions: ['musical mood & tempo', 'scale & sonic texture'],
  },
  'written-work': {
    mediaType: 'written-work',
    label: 'Written Work',
    shortSummary: 'Novel, story, or manuscript workspace with chapter outline & drafting',
    activeOrgans: ['narrative-outline', 'palette-theme'],
    activeWizardQuestions: ['genre & voice', 'chapter structure & target length'],
  },
  video: {
    mediaType: 'video',
    label: 'Video',
    shortSummary: 'Procedural storyboard & real-time motion timeline player',
    activeOrgans: ['visual-canvas', 'timeline-motion', 'palette-theme'],
    activeWizardQuestions: ['motion style & pacing', 'scenes & visual palette'],
  },
  'interactive-game': {
    mediaType: 'interactive-game',
    label: 'Game',
    shortSummary: 'Playable procedural game synthesized from rules, board & difficulty weights',
    activeOrgans: ['game-simulation', 'rule-invariants', 'palette-theme'],
    activeWizardQuestions: ['board size & difficulty presets', 'special rules & visual feel'],
  },
  'research-program': {
    mediaType: 'research-program',
    label: 'Research Program',
    shortSummary: 'Hypotheses, experiments, milestones, and structured evidence synthesis',
    activeOrgans: ['narrative-outline', 'rule-invariants', 'evidence-ledger', 'privacy-compartment'],
    activeWizardQuestions: ['research question & hypotheses', 'milestones & privacy space'],
  },
  'tool-organ': {
    mediaType: 'tool-organ',
    label: 'Tool',
    shortSummary: 'Personal utility, habit/schedule guardian, or unified-model helper',
    activeOrgans: ['rule-invariants', 'sensory-stream', 'impulse-emitter', 'privacy-compartment'],
    activeWizardQuestions: ['core routine or rules', 'space & privacy sharing preference'],
  },
};

export function deriveActiveStudioOrgans(mediaType: PrismMediaType): readonly StudioFeatureOrgan[] {
  return PRISM_MEDIA_CATALOG[mediaType].activeOrgans;
}

// ============================================================================
// Typed Declarative Weight Schemas (`ProceduralOrganWeights`)
// ============================================================================

export type PictureGeneratorKind =
  | 'geometric-mosaic'
  | 'botanical-fractal'
  | 'wave-field'
  | 'constellation'
  | 'pixel-sprite';

export interface PictureWeights {
  readonly mediaType: 'picture';
  readonly generator: PictureGeneratorKind;
  readonly width: number;
  readonly height: number;
  readonly seed: number;
  readonly symmetry: number;
  readonly density: number;
  readonly palette: readonly [string, string, string, string];
  readonly caption: string;
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

export interface ResearchProgramWeights {
  readonly mediaType: 'research-program';
  readonly topic: string;
  readonly question: string;
  readonly compartmentId: string;
  readonly tracks: readonly string[];
  readonly hypotheses: readonly ResearchHypothesisWeight[];
  readonly milestones: readonly ResearchMilestoneWeight[];
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

export interface ToolOrganWeights {
  readonly mediaType: 'tool-organ';
  readonly role: 'habit-guardian' | 'schedule-guard' | 'calculator' | 'workflow-helper';
  readonly compartmentId: string;
  readonly emitsOpaqueConstraints: boolean;
  readonly rulesSummary: string;
  readonly scheduleWindows: readonly ScheduleWindowWeight[];
  readonly metrics: readonly ToolOrganMetricWeight[];
}

export type ProceduralOrganWeights =
  | PictureWeights
  | AudioWeights
  | WrittenWorkWeights
  | VideoWeights
  | InteractiveGameWeights
  | ResearchProgramWeights
  | ToolOrganWeights;

export interface ProceduralOrganSpec {
  readonly mediaType: PrismMediaType;
  readonly activeOrgans: readonly StudioFeatureOrgan[];
  readonly weights: ProceduralOrganWeights;
  /** Optional imported image/audio/video/text open-weight models routed onto this Prism's organs. */
  readonly attachedWeights?: readonly MediaModelWeightBinding[];
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

  if (mediaType === 'picture') {
    const allowedGen: readonly PictureGeneratorKind[] = [
      'geometric-mosaic',
      'botanical-fractal',
      'wave-field',
      'constellation',
      'pixel-sprite',
    ];
    let defaultGen: PictureGeneratorKind = 'geometric-mosaic';
    if (/\b(tree|plant|forest|flower|leaf|botanical|fractal|fern)\b/.test(lowerHint)) {
      defaultGen = 'botanical-fractal';
    } else if (/\b(wave|ocean|sound|ripple|flow|ribbon|aurora)\b/.test(lowerHint)) {
      defaultGen = 'wave-field';
    } else if (/\b(star|space|galaxy|constellation|night|cosmos)\b/.test(lowerHint)) {
      defaultGen = 'constellation';
    } else if (/\b(pixel|8-?bit|sprite|retro)\b/.test(lowerHint)) {
      defaultGen = 'pixel-sprite';
    }
    const gen = allowedGen.includes(p.generator as PictureGeneratorKind)
      ? (p.generator as PictureGeneratorKind)
      : defaultGen;
    const rawPal = Array.isArray(p.palette) ? p.palette : [];
    const palette: [string, string, string, string] = [
      safeHex(rawPal[0], '#0b1714'),
      safeHex(rawPal[1], '#10b981'),
      safeHex(rawPal[2], '#f59e0b'),
      safeHex(rawPal[3], '#ecfdf5'),
    ];
    return {
      mediaType: 'picture',
      generator: gen,
      width: clampInt(p.width, 320, 1600, 768),
      height: clampInt(p.height, 240, 1200, 512),
      seed: clampInt(p.seed, 1, 999_999, 4219),
      symmetry: clampInt(p.symmetry, 1, 16, gen === 'geometric-mosaic' ? 6 : 4),
      density: clampInt(p.density, 8, 120, 42),
      palette,
      caption: safeStr(p.caption, titleHint, 140),
    };
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
      : /\b(dark|minor|moody|noir)\b/.test(lowerHint)
        ? 'minor'
        : 'pentatonic';
    return {
      mediaType: 'audio',
      style,
      bpm: clampInt(p.bpm, 45, 200, style === 'ambient-pad' ? 76 : style === 'rhythm-pulse' ? 118 : 104),
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
    return {
      mediaType: 'research-program',
      topic: safeStr(p.topic, titleHint, 120),
      question: safeStr(p.question, hintText || `Investigate and validate ${titleHint}.`, 260),
      compartmentId: safeStr(p.compartmentId, 'research', 32),
      tracks,
      hypotheses,
      milestones,
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
  };
}

// ============================================================================
// Audited Procedural Organ View Synthesizers (Zero LLM-Written JavaScript!)
// ============================================================================

function escHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function safeJsonForScript(obj: unknown): string {
  return JSON.stringify(obj).replace(/<\//g, '<\\/');
}

const PROCEDURAL_BASE_CSS = `
:root {
  --v-bg: #f8f9fa;
  --v-surface: #ffffff;
  --v-surface-alt: #f1f5f9;
  --v-border: #d8dee9;
  --v-text: #0f172a;
  --v-muted: #64748b;
  --v-accent: #0f5132;
  --v-accent-text: #ffffff;
  color-scheme: light;
}
:root.dark, :root[data-theme="dark"] {
  --v-bg: #0b0f14;
  --v-surface: #111822;
  --v-surface-alt: #182230;
  --v-border: #263244;
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
  font-size: 14px; line-height: 1.45;
  -webkit-font-smoothing: antialiased;
}
.organ-shell { max-width: 900px; margin: 0 auto; padding: 16px; display: flex; flex-direction: column; gap: 14px; }
.organ-header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; padding: 12px 16px; background: var(--v-surface); border: 1px solid var(--v-border); border-radius: 12px; }
.organ-title { margin: 0; font-size: 16px; font-weight: 700; }
.organ-sub { margin: 2px 0 0; font-size: 12px; color: var(--v-muted); }
.organ-card { background: var(--v-surface); border: 1px solid var(--v-border); border-radius: 12px; padding: 16px; }
.organ-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.organ-btn { display: inline-flex; align-items: center; justify-content: center; gap: 6px; padding: 6px 12px; border-radius: 8px; border: 1px solid var(--v-border); background: var(--v-surface-alt); color: var(--v-text); font: inherit; font-size: 12px; font-weight: 600; cursor: pointer; }
.organ-btn:hover { opacity: 0.9; }
.organ-btn.primary { background: var(--v-accent); color: var(--v-accent-text); border-color: transparent; }
.organ-pill { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 999px; border: 1px solid var(--v-border); background: var(--v-surface-alt); font-size: 12px; font-weight: 600; font-variant-numeric: tabular-nums; }
.organ-input { padding: 7px 10px; border-radius: 8px; border: 1px solid var(--v-border); background: var(--v-surface-alt); color: var(--v-text); font: inherit; font-size: 13px; }
.ms-grid { display: grid; gap: 3px; justify-content: center; margin: 0 auto; user-select: none; touch-action: manipulation; }
.ms-cell { width: 30px; height: 30px; border-radius: 6px; border: 1px solid var(--v-border); background: var(--v-surface-alt); color: var(--v-text); font-weight: 700; font-size: 13px; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: transform 60ms ease, background 80ms ease; }
.ms-cell:hover:not(.rev) { filter: brightness(1.06); }
.ms-cell.rev { background: var(--v-bg); border-color: transparent; cursor: default; }
.ms-cell.mine { background: #dc2626; color: #fff; }
.ms-cell.flag { background: rgba(245, 158, 11, 0.22); border-color: #f59e0b; }
.ms-n1 { color: #2563eb; } .ms-n2 { color: #16a34a; } .ms-n3 { color: #dc2626; } .ms-n4 { color: #7c3aed; }
.ms-n5 { color: #b45309; } .ms-n6 { color: #0891b2; } .ms-n7 { color: #475569; } .ms-n8 { color: #94a3b8; }
canvas.organ-canvas { display: block; width: 100%; height: auto; border-radius: 10px; border: 1px solid var(--v-border); background: #0b0f14; }
.organ-textarea { width: 100%; min-height: 220px; padding: 12px; border-radius: 10px; border: 1px solid var(--v-border); background: var(--v-bg); color: var(--v-text); font: inherit; font-size: 14px; line-height: 1.65; resize: vertical; }
`;

function buildGridSweepRuntimeJs(weights: GridSweepGameWeights): string {
  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
let rows = W.rows, cols = W.cols, mines = W.mines;
let board = [], revealed = [], flagged = [], firstClick = true, over = false, won = false;
let flagMode = false, elapsed = 0, timerId = null;

function stopTimer() { if (timerId) { clearInterval(timerId); timerId = null; } }
function startTimer() {
  if (!W.showTimer || timerId) return;
  timerId = setInterval(() => { elapsed += 1; updateHud(); }, 1000);
}
function syncModel(statusCode) {
  try {
    if (globalThis.prism && typeof globalThis.prism.invoke === 'function') {
      void globalThis.prism.invoke(rows + ' ' + cols + ' ' + mines + ' ' + statusCode).catch(() => {});
    }
  } catch (_) {}
}
function neighbors(r, c) {
  const out = [];
  for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
    if (dr === 0 && dc === 0) continue;
    const nr = r + dr, nc = c + dc;
    if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) out.push([nr, nc]);
  }
  return out;
}
function placeMines(safeR, safeC) {
  const forbidden = new Set();
  if (W.firstClickSafe) {
    forbidden.add(safeR + ',' + safeC);
    for (const [nr, nc] of neighbors(safeR, safeC)) forbidden.add(nr + ',' + nc);
  }
  const pool = [];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    if (!forbidden.has(r + ',' + c)) pool.push([r, c]);
  }
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  const target = Math.min(mines, pool.length);
  for (let i = 0; i < target; i++) board[pool[i][0]][pool[i][1]] = -1;
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    if (board[r][c] === -1) continue;
    let cnt = 0;
    for (const [nr, nc] of neighbors(r, c)) if (board[nr][nc] === -1) cnt += 1;
    board[r][c] = cnt;
  }
}
function initGame(r = rows, c = cols, m = mines) {
  rows = r; cols = c; mines = Math.min(m, Math.max(1, r * c - 9));
  stopTimer(); elapsed = 0; firstClick = true; over = false; won = false;
  board = Array.from({ length: rows }, () => Array(cols).fill(0));
  revealed = Array.from({ length: rows }, () => Array(cols).fill(false));
  flagged = Array.from({ length: rows }, () => Array(cols).fill(false));
  render();
}
function revealCell(r, c) {
  if (over || flagged[r][c]) return;
  if (firstClick) {
    firstClick = false;
    placeMines(r, c);
    startTimer();
  }
  if (revealed[r][c]) {
    if (W.chordReveal && board[r][c] > 0) {
      const adj = neighbors(r, c);
      const adjFlags = adj.filter(([nr, nc]) => flagged[nr][nc]).length;
      if (adjFlags === board[r][c]) {
        for (const [nr, nc] of adj) if (!revealed[nr][nc] && !flagged[nr][nc]) revealCell(nr, nc);
      }
    }
    return;
  }
  if (board[r][c] === -1) {
    over = true; won = false; stopTimer();
    for (let rr = 0; rr < rows; rr++) for (let cc = 0; cc < cols; cc++) {
      if (board[rr][cc] === -1) revealed[rr][cc] = true;
    }
    syncModel(2);
    render();
    return;
  }
  const queue = [[r, c]];
  while (queue.length > 0) {
    const [cr, cc] = queue.shift();
    if (revealed[cr][cc] || flagged[cr][cc]) continue;
    revealed[cr][cc] = true;
    if (W.floodReveal && board[cr][cc] === 0) {
      for (const [nr, nc] of neighbors(cr, cc)) {
        if (!revealed[nr][nc] && !flagged[nr][nc]) queue.push([nr, nc]);
      }
    }
  }
  let safeLeft = 0;
  for (let rr = 0; rr < rows; rr++) for (let cc = 0; cc < cols; cc++) {
    if (board[rr][cc] !== -1 && !revealed[rr][cc]) safeLeft += 1;
  }
  if (safeLeft === 0) {
    over = true; won = true; stopTimer();
    for (let rr = 0; rr < rows; rr++) for (let cc = 0; cc < cols; cc++) {
      if (board[rr][cc] === -1) flagged[rr][cc] = true;
    }
    syncModel(1);
  }
  render();
}
function toggleFlag(r, c) {
  if (over || revealed[r][c]) return;
  flagged[r][c] = !flagged[r][c];
  render();
}
function updateHud() {
  const t = document.getElementById('ms-timer');
  if (t) t.textContent = elapsed + 's';
}
function render() {
  const flagCount = flagged.flat().filter(Boolean).length;
  const remaining = mines - flagCount;
  const statusText = over ? (won ? 'Cleared! You won!' : 'Hit a mine — try again!') : 'Active';
  const cellPx = cols > 18 ? 24 : cols > 12 ? 28 : 32;
  root.innerHTML = '';

  const hud = document.createElement('div');
  hud.className = 'organ-row';
  hud.style.justifyContent = 'space-between';
  hud.style.marginBottom = '12px';

  const stats = document.createElement('div');
  stats.className = 'organ-row';
  stats.innerHTML =
    '<span class="organ-pill">Mines: ' + remaining + '</span>' +
    (W.showTimer ? '<span class="organ-pill">Time: <span id="ms-timer">' + elapsed + 's</span></span>' : '') +
    '<span class="organ-pill" style="' + (over ? (won ? 'border-color:#10b981;color:#10b981' : 'border-color:#dc2626;color:#dc2626') : '') + '">' + statusText + '</span>';
  hud.appendChild(stats);

  const controls = document.createElement('div');
  controls.className = 'organ-row';
  const flagBtn = document.createElement('button');
  flagBtn.type = 'button';
  flagBtn.className = 'organ-btn' + (flagMode ? ' primary' : '');
  flagBtn.textContent = flagMode ? 'Flag Mode: ON' : 'Flag Mode';
  flagBtn.onclick = () => { flagMode = !flagMode; render(); };
  controls.appendChild(flagBtn);

  const newBtn = document.createElement('button');
  newBtn.type = 'button';
  newBtn.className = 'organ-btn primary';
  newBtn.textContent = 'New Game';
  newBtn.onclick = () => initGame();
  controls.appendChild(newBtn);
  hud.appendChild(controls);
  root.appendChild(hud);

  if (W.presets && W.presets.length > 0) {
    const presetRow = document.createElement('div');
    presetRow.className = 'organ-row';
    presetRow.style.marginBottom = '12px';
    for (const pr of W.presets) {
      const b = document.createElement('button');
      b.type = 'button';
      const active = pr.rows === rows && pr.cols === cols && pr.mines === mines;
      b.className = 'organ-btn' + (active ? ' primary' : '');
      b.textContent = pr.label + ' (' + pr.rows + '×' + pr.cols + ' · ' + pr.mines + ')';
      b.onclick = () => initGame(pr.rows, pr.cols, pr.mines);
      presetRow.appendChild(b);
    }
    root.appendChild(presetRow);
  }

  const wrap = document.createElement('div');
  wrap.style.overflowX = 'auto';
  wrap.style.paddingBottom = '6px';
  const grid = document.createElement('div');
  grid.className = 'ms-grid';
  grid.style.gridTemplateColumns = 'repeat(' + cols + ', ' + cellPx + 'px)';
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.style.width = cellPx + 'px';
      btn.style.height = cellPx + 'px';
      const isRev = revealed[r][c];
      const isFlag = flagged[r][c];
      const val = board[r][c];
      btn.className = 'ms-cell' + (isRev ? (val === -1 ? ' rev mine' : ' rev ms-n' + val) : isFlag ? ' flag' : '');
      btn.textContent = isRev ? (val === -1 ? '✹' : val > 0 ? String(val) : '') : (isFlag ? '⚑' : '');
      btn.setAttribute('aria-label', 'Row ' + (r + 1) + ' Col ' + (c + 1));
      btn.onclick = () => (flagMode ? toggleFlag(r, c) : revealCell(r, c));
      btn.oncontextmenu = (ev) => { ev.preventDefault(); toggleFlag(r, c); };
      grid.appendChild(btn);
    }
  }
  wrap.appendChild(grid);
  root.appendChild(wrap);

  const hint = document.createElement('p');
  hint.className = 'organ-sub';
  hint.style.textAlign = 'center';
  hint.style.marginTop = '10px';
  hint.textContent = 'Left-click to reveal · Right-click (or Flag Mode) to mark mines · Click a revealed number to chord';
  root.appendChild(hint);
}
initGame();
`;
}

function buildPictureRuntimeJs(weights: PictureWeights): string {
  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
let seed = W.seed, symmetry = W.symmetry, density = W.density, generator = W.generator;

function mulberry32(a) {
  return function() {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function drawArtwork(canvas) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const rng = mulberry32(seed);
  const [bg, c1, c2, c3] = W.palette;
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  const cx = w / 2, cy = h / 2, rad = Math.min(w, h) * 0.44;
  if (generator === 'wave-field') {
    for (let i = 0; i < density; i++) {
      ctx.beginPath();
      const yBase = (h / (density + 1)) * (i + 1);
      const amp = 14 + rng() * 36;
      const freq = 0.008 + rng() * 0.015;
      const phase = rng() * Math.PI * 2;
      for (let x = 0; x <= w; x += 6) {
        const y = yBase + Math.sin(x * freq + phase) * amp;
        if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = i % 3 === 0 ? c1 : i % 3 === 1 ? c2 : c3;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 2 + (i % 3);
      ctx.stroke();
    }
  } else if (generator === 'constellation') {
    const pts = [];
    for (let i = 0; i < density; i++) pts.push([rng() * w, rng() * h, 1.5 + rng() * 3.5]);
    ctx.strokeStyle = c1; ctx.globalAlpha = 0.28; ctx.lineWidth = 1;
    for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {
      const dx = pts[i][0] - pts[j][0], dy = pts[i][1] - pts[j][1];
      if (dx * dx + dy * dy < 14000) { ctx.beginPath(); ctx.moveTo(pts[i][0], pts[i][1]); ctx.lineTo(pts[j][0], pts[j][1]); ctx.stroke(); }
    }
    for (let i = 0; i < pts.length; i++) {
      ctx.globalAlpha = 0.9;
      ctx.fillStyle = i % 4 === 0 ? c2 : c3;
      ctx.beginPath(); ctx.arc(pts[i][0], pts[i][1], pts[i][2], 0, Math.PI * 2); ctx.fill();
    }
  } else if (generator === 'botanical-fractal') {
    function branch(x, y, len, angle, depth) {
      if (depth <= 0 || len < 4) return;
      const x2 = x + Math.cos(angle) * len, y2 = y + Math.sin(angle) * len;
      ctx.strokeStyle = depth <= 2 ? c2 : c1;
      ctx.lineWidth = Math.max(1, depth * 1.4);
      ctx.globalAlpha = 0.85;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x2, y2); ctx.stroke();
      const spread = 0.35 + rng() * 0.25;
      branch(x2, y2, len * 0.72, angle - spread, depth - 1);
      branch(x2, y2, len * 0.72, angle + spread, depth - 1);
    }
    branch(cx, h * 0.92, h * 0.24, -Math.PI / 2, Math.min(9, Math.max(5, Math.floor(symmetry + 3))));
  } else if (generator === 'pixel-sprite') {
    const grid = Math.max(8, Math.min(20, symmetry * 2));
    const cell = Math.floor(Math.min(w, h) * 0.72 / grid);
    const ox = Math.floor((w - grid * cell) / 2), oy = Math.floor((h - grid * cell) / 2);
    for (let r = 0; r < grid; r++) for (let c = 0; c < Math.ceil(grid / 2); c++) {
      if (rng() > 0.45) {
        ctx.fillStyle = rng() > 0.3 ? c1 : c2;
        ctx.fillRect(ox + c * cell, oy + r * cell, cell - 2, cell - 2);
        ctx.fillRect(ox + (grid - 1 - c) * cell, oy + r * cell, cell - 2, cell - 2);
      }
    }
  } else {
    ctx.save(); ctx.translate(cx, cy);
    for (let i = 0; i < density; i++) {
      const r0 = rng() * rad * 0.85, sz = 10 + rng() * 44;
      const col = i % 3 === 0 ? c1 : i % 3 === 1 ? c2 : c3;
      const a0 = rng() * Math.PI * 2;
      for (let s = 0; s < symmetry; s++) {
        const a = a0 + (s * Math.PI * 2) / symmetry;
        ctx.save(); ctx.rotate(a);
        ctx.fillStyle = col; ctx.globalAlpha = 0.32 + rng() * 0.45;
        ctx.beginPath();
        ctx.moveTo(r0, 0); ctx.lineTo(r0 + sz, sz * 0.42); ctx.lineTo(r0 + sz * 0.6, -sz * 0.42);
        ctx.closePath(); ctx.fill();
        ctx.restore();
      }
    }
    ctx.restore();
  }
  ctx.globalAlpha = 1;
}

function render() {
  root.innerHTML = '';
  const bar = document.createElement('div');
  bar.className = 'organ-row'; bar.style.marginBottom = '12px'; bar.style.justifyContent = 'space-between';

  const left = document.createElement('div'); left.className = 'organ-row';
  const genBtn = document.createElement('button'); genBtn.className = 'organ-btn primary'; genBtn.textContent = 'New Variation';
  genBtn.onclick = () => { seed = (seed * 1664525 + 1013904223) % 999999; drawArtwork(canvas); };
  left.appendChild(genBtn);

  const symBtn = document.createElement('button'); symBtn.className = 'organ-btn'; symBtn.textContent = 'Symmetry: ' + symmetry;
  symBtn.onclick = () => { symmetry = symmetry >= 12 ? 3 : symmetry + 1; symBtn.textContent = 'Symmetry: ' + symmetry; drawArtwork(canvas); };
  left.appendChild(symBtn);
  bar.appendChild(left);

  const expBtn = document.createElement('button'); expBtn.className = 'organ-btn'; expBtn.textContent = 'Download PNG';
  expBtn.onclick = () => {
    const a = document.createElement('a');
    a.href = canvas.toDataURL('image/png');
    a.download = (W.caption || 'artwork').toLowerCase().replace(/[^a-z0-9]+/g, '-') + '.png';
    a.click();
  };
  bar.appendChild(expBtn);
  root.appendChild(bar);

  const canvas = document.createElement('canvas');
  canvas.className = 'organ-canvas';
  canvas.width = W.width; canvas.height = W.height;
  root.appendChild(canvas);
  drawArtwork(canvas);
}
render();
`;
}

function buildWrittenWorkRuntimeJs(weights: WrittenWorkWeights): string {
  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
let chapters = W.chapters.map((c) => ({ ...c }));
let activeIdx = 0;

function totalWords() {
  return chapters.reduce((acc, c) => acc + (c.body.trim() ? c.body.trim().split(/\\s+/).length : 0), 0);
}
function persist() {
  try { window.verdant?.storage?.set('chapters', chapters); } catch (_) {}
}
async function hydrate() {
  try {
    const saved = await window.verdant?.storage?.get('chapters');
    if (Array.isArray(saved) && saved.length > 0) { chapters = saved; render(); }
  } catch (_) {}
}
function render() {
  const ch = chapters[activeIdx] || chapters[0];
  const words = totalWords();
  root.innerHTML = '';

  const top = document.createElement('div');
  top.className = 'organ-row'; top.style.justifyContent = 'space-between'; top.style.marginBottom = '12px';
  top.innerHTML =
    '<div class="organ-row">' +
    '<span class="organ-pill">' + W.genre + '</span>' +
    '<span class="organ-pill">Words: ' + words.toLocaleString() + ' / ' + W.targetWordCount.toLocaleString() + '</span>' +
    '</div>';
  const actions = document.createElement('div'); actions.className = 'organ-row';
  const addCh = document.createElement('button'); addCh.className = 'organ-btn'; addCh.textContent = '+ Chapter';
  addCh.onclick = () => {
    chapters.push({ title: 'Chapter ' + (chapters.length + 1), beat: 'Scene', summary: 'Next beat', body: '' });
    activeIdx = chapters.length - 1; persist(); render();
  };
  actions.appendChild(addCh);
  const exp = document.createElement('button'); exp.className = 'organ-btn primary'; exp.textContent = 'Export Markdown';
  exp.onclick = () => {
    const md = '# ' + W.title + '\\n\\n' + W.premise + '\\n\\n' + chapters.map((c) => '## ' + c.title + '\\n\\n' + (c.body || c.summary)).join('\\n\\n');
    if (window.verdant?.downloadFile) { window.verdant.downloadFile(W.title.toLowerCase().replace(/[^a-z0-9]+/g, '-') + '.md', md, 'text/markdown'); }
  };
  actions.appendChild(exp);
  top.appendChild(actions);
  root.appendChild(top);

  const tabs = document.createElement('div');
  tabs.className = 'organ-row'; tabs.style.marginBottom = '10px';
  chapters.forEach((c, idx) => {
    const b = document.createElement('button');
    b.className = 'organ-btn' + (idx === activeIdx ? ' primary' : '');
    b.textContent = c.title;
    b.onclick = () => { activeIdx = idx; render(); };
    tabs.appendChild(b);
  });
  root.appendChild(tabs);

  const meta = document.createElement('p');
  meta.className = 'organ-sub'; meta.style.marginBottom = '8px';
  meta.textContent = ch.beat + ' · ' + ch.summary;
  root.appendChild(meta);

  const ta = document.createElement('textarea');
  ta.className = 'organ-textarea';
  ta.placeholder = 'Write ' + ch.title + ' here…';
  ta.value = ch.body;
  ta.oninput = () => { ch.body = ta.value; persist(); };
  root.appendChild(ta);

  if (W.allowLocalAssist && window.verdant?.ask) {
    const assistBtn = document.createElement('button');
    assistBtn.className = 'organ-btn'; assistBtn.style.marginTop = '10px';
    assistBtn.textContent = 'Draft Next Paragraph Locally';
    assistBtn.onclick = async () => {
      assistBtn.disabled = true; assistBtn.textContent = 'Drafting…';
      try {
        const prompt = 'Continue this ' + W.genre + ' chapter in voice "' + W.voice + '". Premise: ' + W.premise + '. Chapter summary: ' + ch.summary + '. Existing draft:\\n' + ch.body.slice(-600);
        const reply = await window.verdant.ask(prompt);
        ch.body = (ch.body ? ch.body + '\\n\\n' : '') + String(reply || '').trim();
        persist(); render();
      } catch (_) { assistBtn.disabled = false; assistBtn.textContent = 'Draft Next Paragraph Locally'; }
    };
    root.appendChild(assistBtn);
  }
}
render();
hydrate();
`;
}

function buildVideoRuntimeJs(weights: VideoWeights): string {
  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
const dims = W.aspectRatio === '1:1' ? [540, 540] : W.aspectRatio === '9:16' ? [360, 640] : [640, 360];
const totalDur = W.scenes.reduce((a, s) => a + s.durationSec, 0) || 6;
let playing = true, tSec = 0, lastTs = 0;

root.innerHTML = '';
const bar = document.createElement('div'); bar.className = 'organ-row'; bar.style.marginBottom = '10px'; bar.style.justifyContent = 'space-between';
const playBtn = document.createElement('button'); playBtn.className = 'organ-btn primary'; playBtn.textContent = 'Pause';
const timeEl = document.createElement('span'); timeEl.className = 'organ-pill'; timeEl.textContent = '0.0s / ' + totalDur + 's';
const left = document.createElement('div'); left.className = 'organ-row'; left.append(playBtn, timeEl);
const snapBtn = document.createElement('button'); snapBtn.className = 'organ-btn'; snapBtn.textContent = 'Save Frame PNG';
bar.append(left, snapBtn);
root.appendChild(bar);

const canvas = document.createElement('canvas');
canvas.className = 'organ-canvas'; canvas.width = dims[0]; canvas.height = dims[1];
root.appendChild(canvas);
const ctx = canvas.getContext('2d');

playBtn.onclick = () => { playing = !playing; playBtn.textContent = playing ? 'Pause' : 'Play'; };
snapBtn.onclick = () => { const a = document.createElement('a'); a.href = canvas.toDataURL('image/png'); a.download = 'frame.png'; a.click(); };

function drawFrame(sec) {
  let acc = 0, scene = W.scenes[0], localT = 0;
  for (const s of W.scenes) {
    if (sec <= acc + s.durationSec) { scene = s; localT = (sec - acc) / Math.max(0.1, s.durationSec); break; }
    acc += s.durationSec;
  }
  const [w, h] = dims;
  ctx.fillStyle = W.palette[0]; ctx.fillRect(0, 0, w, h);
  ctx.save();
  const cx = w / 2, cy = h / 2;
  for (let i = 0; i < 6; i++) {
    const r = 36 + i * 26 + Math.sin(localT * Math.PI * 2 + i) * 14;
    ctx.strokeStyle = scene.accent; ctx.globalAlpha = 0.22; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(cx + Math.cos(localT * 6 + i) * 24, cy + Math.sin(localT * 4 + i) * 14, r, 0, Math.PI * 2); ctx.stroke();
  }
  ctx.globalAlpha = 1;
  ctx.fillStyle = W.palette[2]; ctx.textAlign = 'center';
  const yOff = scene.motion === 'rise' ? (1 - Math.min(1, localT * 2)) * 28 : 0;
  ctx.font = '700 28px Inter, sans-serif'; ctx.fillText(scene.title, cx, cy - 6 + yOff);
  ctx.fillStyle = scene.accent; ctx.font = '500 15px Inter, sans-serif'; ctx.fillText(scene.subtitle, cx, cy + 24 + yOff);
  ctx.restore();
  timeEl.textContent = sec.toFixed(1) + 's / ' + totalDur + 's';
}
function loop(ts) {
  if (!lastTs) lastTs = ts;
  const dt = Math.min(0.1, (ts - lastTs) / 1000);
  lastTs = ts;
  if (playing) tSec = (tSec + dt) % totalDur;
  drawFrame(tSec);
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
`;
}

function buildTurnBoardRuntimeJs(weights: TurnBoardGameWeights): string {
  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
let grid = [], turn = 1, winner = 0, draws = 0, score = [0, 0, 0];

function resetBoard() {
  grid = Array.from({ length: W.rows }, () => Array(W.cols).fill(0));
  turn = 1; winner = 0; render();
}
function checkWin(p) {
  const dirs = [[0,1],[1,0],[1,1],[1,-1]];
  for (let r = 0; r < W.rows; r++) for (let c = 0; c < W.cols; c++) {
    if (grid[r][c] !== p) continue;
    for (const [dr, dc] of dirs) {
      let k = 1;
      while (k < W.winLength) {
        const nr = r + dr * k, nc = c + dc * k;
        if (nr < 0 || nr >= W.rows || nc < 0 || nc >= W.cols || grid[nr][nc] !== p) break;
        k += 1;
      }
      if (k >= W.winLength) return true;
    }
  }
  return false;
}
function playAt(r, c) {
  if (winner !== 0) return;
  let tr = r;
  if (W.gravity) {
    tr = -1;
    for (let rr = W.rows - 1; rr >= 0; rr--) { if (grid[rr][c] === 0) { tr = rr; break; } }
    if (tr < 0) return;
  } else if (grid[tr][c] !== 0) return;
  grid[tr][c] = turn;
  if (checkWin(turn)) { winner = turn; score[turn] += 1; render(); return; }
  if (grid.flat().every((v) => v !== 0)) { winner = -1; draws += 1; render(); return; }
  turn = turn === 1 ? 2 : 1;
  render();
  if (W.vsCpu && turn === 2 && winner === 0) {
    setTimeout(() => {
      const moves = [];
      for (let rr = 0; rr < W.rows; rr++) for (let cc = 0; cc < W.cols; cc++) {
        if (W.gravity ? (rr === 0 && grid[0][cc] === 0) : grid[rr][cc] === 0) moves.push([rr, cc]);
      }
      if (moves.length > 0) {
        const pick = moves[Math.floor(Math.random() * moves.length)];
        playAt(pick[0], pick[1]);
      }
    }, 180);
  }
}
function render() {
  root.innerHTML = '';
  const top = document.createElement('div'); top.className = 'organ-row'; top.style.justifyContent = 'space-between'; top.style.marginBottom = '12px';
  const status = winner === -1 ? 'Draw!' : winner === 1 ? W.p1Label + ' wins!' : winner === 2 ? W.p2Label + ' wins!' : (turn === 1 ? W.p1Label : W.p2Label) + "'s turn";
  top.innerHTML = '<div class="organ-row"><span class="organ-pill">' + W.p1Label + ': ' + score[1] + '</span><span class="organ-pill">' + W.p2Label + ': ' + score[2] + '</span><span class="organ-pill">' + status + '</span></div>';
  const btn = document.createElement('button'); btn.className = 'organ-btn primary'; btn.textContent = 'New Round'; btn.onclick = resetBoard;
  top.appendChild(btn); root.appendChild(top);

  const cellPx = W.cols <= 4 ? 64 : 44;
  const g = document.createElement('div'); g.className = 'ms-grid';
  g.style.gridTemplateColumns = 'repeat(' + W.cols + ', ' + cellPx + 'px)';
  for (let r = 0; r < W.rows; r++) for (let c = 0; c < W.cols; c++) {
    const b = document.createElement('button'); b.className = 'ms-cell';
    b.style.width = cellPx + 'px'; b.style.height = cellPx + 'px'; b.style.fontSize = cellPx > 50 ? '24px' : '18px';
    const v = grid[r][c];
    b.textContent = v === 1 ? (W.gravity ? '●' : 'X') : v === 2 ? (W.gravity ? '●' : 'O') : '';
    b.style.color = v === 1 ? '#10b981' : v === 2 ? '#f59e0b' : 'inherit';
    b.onclick = () => playAt(r, c);
    g.appendChild(b);
  }
  root.appendChild(g);
}
resetBoard();
`;
}

function buildGridPuzzleRuntimeJs(weights: GridPuzzleGameWeights): string {
  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
let board = [], moves = 0, score = 0, won = false;

function initLightsOut() {
  board = Array.from({ length: W.rows }, () => Array(W.cols).fill(false));
  for (let i = 0; i < W.rows + W.cols; i++) toggleLight(Math.floor(Math.random() * W.rows), Math.floor(Math.random() * W.cols), true);
  moves = 0; won = false; render();
}
function toggleLight(r, c, silent) {
  for (const [dr, dc] of [[0,0],[1,0],[-1,0],[0,1],[0,-1]]) {
    const nr = r + dr, nc = c + dc;
    if (nr >= 0 && nr < W.rows && nc >= 0 && nc < W.cols) board[nr][nc] = !board[nr][nc];
  }
  if (!silent) { moves += 1; won = board.flat().every((v) => !v); render(); }
}
function init2048() {
  board = Array.from({ length: W.rows }, () => Array(W.cols).fill(0));
  score = 0; moves = 0; won = false; spawn2048(); spawn2048(); render();
}
function spawn2048() {
  const empty = [];
  for (let r = 0; r < W.rows; r++) for (let c = 0; c < W.cols; c++) if (board[r][c] === 0) empty.push([r, c]);
  if (empty.length > 0) { const [r, c] = empty[Math.floor(Math.random() * empty.length)]; board[r][c] = Math.random() < 0.9 ? 2 : 4; }
}
function slideRow(arr) {
  const nums = arr.filter((v) => v > 0);
  const out = [];
  for (let i = 0; i < nums.length; i++) {
    if (nums[i] === nums[i + 1]) { const m = nums[i] * 2; score += m; if (m >= W.targetScore) won = true; out.push(m); i += 1; }
    else out.push(nums[i]);
  }
  while (out.length < arr.length) out.push(0);
  return out;
}
function move2048(dir) {
  const before = JSON.stringify(board);
  if (dir === 'left') for (let r = 0; r < W.rows; r++) board[r] = slideRow(board[r]);
  if (dir === 'right') for (let r = 0; r < W.rows; r++) board[r] = slideRow([...board[r]].reverse()).reverse();
  if (dir === 'up') for (let c = 0; c < W.cols; c++) { const col = slideRow(board.map((r) => r[c])); for (let r = 0; r < W.rows; r++) board[r][c] = col[r]; }
  if (dir === 'down') for (let c = 0; c < W.cols; c++) { const col = slideRow(board.map((r) => r[c]).reverse()).reverse(); for (let r = 0; r < W.rows; r++) board[r][c] = col[r]; }
  if (JSON.stringify(board) !== before) { moves += 1; spawn2048(); render(); }
}
function render() {
  root.innerHTML = '';
  const top = document.createElement('div'); top.className = 'organ-row'; top.style.justifyContent = 'space-between'; top.style.marginBottom = '12px';
  top.innerHTML = '<div class="organ-row"><span class="organ-pill">Moves: ' + moves + '</span>' + (W.variant === 'merge-2048' ? '<span class="organ-pill">Score: ' + score + '</span>' : '') + (won ? '<span class="organ-pill" style="color:#10b981;border-color:#10b981">Solved!</span>' : '') + '</div>';
  const reset = document.createElement('button'); reset.className = 'organ-btn primary'; reset.textContent = 'Reset';
  reset.onclick = () => (W.variant === 'lights-out' ? initLightsOut() : init2048());
  top.appendChild(reset); root.appendChild(top);

  const g = document.createElement('div'); g.className = 'ms-grid';
  g.style.gridTemplateColumns = 'repeat(' + W.cols + ', 58px)';
  for (let r = 0; r < W.rows; r++) for (let c = 0; c < W.cols; c++) {
    const b = document.createElement('button'); b.className = 'ms-cell'; b.style.width = '58px'; b.style.height = '58px';
    if (W.variant === 'lights-out') {
      b.style.background = board[r][c] ? 'var(--v-accent)' : 'var(--v-surface-alt)';
      b.onclick = () => toggleLight(r, c, false);
    } else {
      const v = board[r][c];
      b.textContent = v ? String(v) : '';
      b.style.fontSize = v >= 1024 ? '14px' : '18px';
      if (v > 0) b.style.background = 'rgba(16,185,129,' + Math.min(0.85, 0.14 + Math.log2(v) * 0.08) + ')';
    }
    g.appendChild(b);
  }
  root.appendChild(g);
  if (W.variant !== 'lights-out') {
    const ctrls = document.createElement('div'); ctrls.className = 'organ-row'; ctrls.style.justifyContent = 'center'; ctrls.style.marginTop = '12px';
    for (const d of ['left','up','down','right']) {
      const b = document.createElement('button'); b.className = 'organ-btn'; b.textContent = d.toUpperCase(); b.onclick = () => move2048(d); ctrls.appendChild(b);
    }
    root.appendChild(ctrls);
  }
}
window.addEventListener('keydown', (e) => {
  if (W.variant === 'lights-out') return;
  const map = { ArrowLeft: 'left', ArrowRight: 'right', ArrowUp: 'up', ArrowDown: 'down' };
  if (map[e.key]) { e.preventDefault(); move2048(map[e.key]); }
});
if (W.variant === 'lights-out') initLightsOut(); else init2048();
`;
}

function buildArcadeCanvasRuntimeJs(weights: ArcadeCanvasGameWeights): string {
  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
let score = 0, alive = true, dir = [1, 0], snake = [[6, 8], [5, 8], [4, 8]], food = [12, 8];

root.innerHTML = '';
const bar = document.createElement('div'); bar.className = 'organ-row'; bar.style.justifyContent = 'space-between'; bar.style.marginBottom = '10px';
const scorePill = document.createElement('span'); scorePill.className = 'organ-pill'; scorePill.textContent = 'Score: 0';
const resetBtn = document.createElement('button'); resetBtn.className = 'organ-btn primary'; resetBtn.textContent = 'Restart';
bar.append(scorePill, resetBtn); root.appendChild(bar);

const canvas = document.createElement('canvas'); canvas.className = 'organ-canvas'; canvas.width = 480; canvas.height = 360;
root.appendChild(canvas);
const ctx = canvas.getContext('2d');
const cw = Math.floor(canvas.width / W.gridCols), ch = Math.floor(canvas.height / W.gridRows);

function reset() {
  score = 0; alive = true; dir = [1, 0]; snake = [[6, 8], [5, 8], [4, 8]];
  food = [Math.floor(Math.random() * W.gridCols), Math.floor(Math.random() * W.gridRows)];
  scorePill.textContent = 'Score: 0';
}
resetBtn.onclick = reset;
window.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowUp' && dir[1] !== 1) dir = [0, -1];
  else if (e.key === 'ArrowDown' && dir[1] !== -1) dir = [0, 1];
  else if (e.key === 'ArrowLeft' && dir[0] !== 1) dir = [-1, 0];
  else if (e.key === 'ArrowRight' && dir[0] !== -1) dir = [1, 0];
});
setInterval(() => {
  if (alive) {
    let nx = snake[0][0] + dir[0], ny = snake[0][1] + dir[1];
    if (W.wrapWalls) { nx = (nx + W.gridCols) % W.gridCols; ny = (ny + W.gridRows) % W.gridRows; }
    if (nx < 0 || nx >= W.gridCols || ny < 0 || ny >= W.gridRows || snake.some(([sx, sy]) => sx === nx && sy === ny)) {
      alive = false; scorePill.textContent = 'Game Over · Score: ' + score;
    } else {
      snake.unshift([nx, ny]);
      if (nx === food[0] && ny === food[1]) {
        score += 10; scorePill.textContent = 'Score: ' + score;
        food = [Math.floor(Math.random() * W.gridCols), Math.floor(Math.random() * W.gridRows)];
      } else snake.pop();
    }
  }
  ctx.fillStyle = '#0b0f14'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#f59e0b'; ctx.fillRect(food[0] * cw + 2, food[1] * ch + 2, cw - 4, ch - 4);
  ctx.fillStyle = '#10b981';
  for (const [sx, sy] of snake) ctx.fillRect(sx * cw + 1, sy * ch + 1, cw - 2, ch - 2);
}, Math.max(60, 200 - W.speed * 14));
`;
}

function buildCellularSimRuntimeJs(weights: CellularSimGameWeights): string {
  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
let grid = Array.from({ length: W.rows }, () => Array.from({ length: W.cols }, () => (Math.random() < 0.25 ? 1 : 0)));
let running = true, gen = 0;

root.innerHTML = '';
const bar = document.createElement('div'); bar.className = 'organ-row'; bar.style.justifyContent = 'space-between'; bar.style.marginBottom = '10px';
const genPill = document.createElement('span'); genPill.className = 'organ-pill'; genPill.textContent = 'Gen: 0';
const playBtn = document.createElement('button'); playBtn.className = 'organ-btn primary'; playBtn.textContent = 'Pause';
const randBtn = document.createElement('button'); randBtn.className = 'organ-btn'; randBtn.textContent = 'Randomize';
const left = document.createElement('div'); left.className = 'organ-row'; left.append(playBtn, randBtn);
bar.append(genPill, left); root.appendChild(bar);

const canvas = document.createElement('canvas'); canvas.className = 'organ-canvas'; canvas.width = W.cols * 14; canvas.height = W.rows * 14;
root.appendChild(canvas);
const ctx = canvas.getContext('2d');
playBtn.onclick = () => { running = !running; playBtn.textContent = running ? 'Pause' : 'Play'; };
randBtn.onclick = () => { gen = 0; grid = Array.from({ length: W.rows }, () => Array.from({ length: W.cols }, () => (Math.random() < 0.28 ? 1 : 0))); };
canvas.onclick = (e) => {
  const rect = canvas.getBoundingClientRect();
  const c = Math.floor(((e.clientX - rect.left) / rect.width) * W.cols);
  const r = Math.floor(((e.clientY - rect.top) / rect.height) * W.rows);
  if (r >= 0 && r < W.rows && c >= 0 && c < W.cols) grid[r][c] = grid[r][c] ? 0 : 1;
};
function step() {
  const next = Array.from({ length: W.rows }, () => Array(W.cols).fill(0));
  for (let r = 0; r < W.rows; r++) for (let c = 0; c < W.cols; c++) {
    let n = 0;
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
      if (dr === 0 && dc === 0) continue;
      n += grid[(r + dr + W.rows) % W.rows][(c + dc + W.cols) % W.cols];
    }
    next[r][c] = grid[r][c] ? (n === 2 || n === 3 ? 1 : 0) : (n === 3 ? 1 : 0);
  }
  grid = next; gen += 1; genPill.textContent = 'Gen: ' + gen;
}
setInterval(() => {
  if (running) step();
  ctx.fillStyle = '#0b0f14'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#10b981';
  for (let r = 0; r < W.rows; r++) for (let c = 0; c < W.cols; c++) {
    if (grid[r][c]) ctx.fillRect(c * 14 + 1, r * 14 + 1, 12, 12);
  }
}, W.tickMs);
`;
}

function buildResearchProgramRuntimeJs(weights: ResearchProgramWeights): string {
  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
let hypotheses = W.hypotheses.map((h) => ({ ...h }));
let milestones = W.milestones.map((m) => ({ ...m }));
let notes = [];

function persist() { try { window.verdant?.storage?.set('research', { hypotheses, milestones, notes }); } catch (_) {} }
async function hydrate() {
  try {
    const saved = await window.verdant?.storage?.get('research');
    if (saved && Array.isArray(saved.hypotheses)) { hypotheses = saved.hypotheses; milestones = saved.milestones || milestones; notes = saved.notes || []; render(); }
  } catch (_) {}
}
function render() {
  root.innerHTML = '';
  const head = document.createElement('div'); head.className = 'organ-row'; head.style.justifyContent = 'space-between'; head.style.marginBottom = '12px';
  head.innerHTML = '<div class="organ-row"><span class="organ-pill">Space: ' + W.compartmentId + '</span>' + W.tracks.map((t) => '<span class="organ-pill">' + t + '</span>').join('') + '</div>';
  root.appendChild(head);

  const q = document.createElement('p'); q.style.margin = '0 0 12px'; q.style.fontWeight = '600'; q.textContent = W.question;
  root.appendChild(q);

  const hSec = document.createElement('div'); hSec.style.marginBottom = '14px';
  hSec.innerHTML = '<div class="organ-sub" style="margin-bottom:6px;font-weight:700">HYPOTHESES (click status to cycle)</div>';
  hypotheses.forEach((h) => {
    const row = document.createElement('div'); row.className = 'organ-row'; row.style.justifyContent = 'space-between'; row.style.padding = '8px 10px'; row.style.border = '1px solid var(--v-border)'; row.style.borderRadius = '8px'; row.style.marginBottom = '6px';
    const txt = document.createElement('span'); txt.textContent = h.id + ': ' + h.statement;
    const st = document.createElement('button'); st.className = 'organ-btn'; st.textContent = h.status;
    st.onclick = () => { h.status = h.status === 'open' ? 'supported' : h.status === 'supported' ? 'refuted' : 'open'; persist(); render(); };
    row.append(txt, st); hSec.appendChild(row);
  });
  root.appendChild(hSec);

  const mSec = document.createElement('div');
  mSec.innerHTML = '<div class="organ-sub" style="margin-bottom:6px;font-weight:700">MILESTONES</div>';
  milestones.forEach((m) => {
    const row = document.createElement('div'); row.className = 'organ-row'; row.style.justifyContent = 'space-between'; row.style.padding = '8px 10px'; row.style.border = '1px solid var(--v-border)'; row.style.borderRadius = '8px'; row.style.marginBottom = '6px';
    const txt = document.createElement('span'); txt.textContent = m.title + ' (' + m.dueLabel + ')';
    const st = document.createElement('button'); st.className = 'organ-btn' + (m.status === 'done' ? ' primary' : ''); st.textContent = m.status;
    st.onclick = () => { m.status = m.status === 'planned' ? 'in-progress' : m.status === 'in-progress' ? 'done' : 'planned'; persist(); render(); };
    row.append(txt, st); mSec.appendChild(row);
  });
  root.appendChild(mSec);
}
render();
hydrate();
`;
}

function buildToolOrganRuntimeJs(weights: ToolOrganWeights): string {
  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
let windows = W.scheduleWindows.map((w) => ({ ...w }));
let checkins = 0;

function fmtMin(m) {
  const hh = String(Math.floor(m / 60)).padStart(2, '0');
  const mm = String(m % 60).padStart(2, '0');
  return hh + ':' + mm;
}
function emitConstraints() {
  if (!W.emitsOpaqueConstraints) return;
  try {
    window.verdant?.emit?.('constraint', {
      compartmentId: W.compartmentId,
      opaqueOnly: true,
      windows: windows.filter((w) => w.protectedWindow).map((w) => ({
        startMinuteOfDay: w.startMinuteOfDay,
        endMinuteOfDay: w.endMinuteOfDay,
        daysOfWeek: w.daysOfWeek,
      })),
    });
  } catch (_) {}
}
function render() {
  root.innerHTML = '';
  const top = document.createElement('div'); top.className = 'organ-row'; top.style.justifyContent = 'space-between'; top.style.marginBottom = '12px';
  top.innerHTML =
    '<div class="organ-row">' +
    '<span class="organ-pill">Space: ' + W.compartmentId + '</span>' +
    (W.emitsOpaqueConstraints ? '<span class="organ-pill" style="border-color:#10b981;color:#10b981">Privacy Shield: Availability only (zero details shared)</span>' : '') +
    '</div>';
  root.appendChild(top);

  const summary = document.createElement('p'); summary.style.margin = '0 0 12px'; summary.textContent = W.rulesSummary;
  root.appendChild(summary);

  const metricRow = document.createElement('div'); metricRow.className = 'organ-row'; metricRow.style.marginBottom = '12px';
  const checkBtn = document.createElement('button'); checkBtn.className = 'organ-btn primary'; checkBtn.textContent = 'Log Check-In (' + checkins + ')';
  checkBtn.onclick = () => { checkins += 1; emitConstraints(); render(); };
  metricRow.appendChild(checkBtn);
  root.appendChild(metricRow);

  if (windows.length > 0) {
    const label = document.createElement('div'); label.className = 'organ-sub'; label.style.marginBottom = '6px'; label.style.fontWeight = '700';
    label.textContent = 'PROTECTED SCHEDULE WINDOWS';
    root.appendChild(label);
    windows.forEach((w) => {
      const row = document.createElement('div'); row.className = 'organ-row'; row.style.justifyContent = 'space-between'; row.style.padding = '8px 10px'; row.style.border = '1px solid var(--v-border)'; row.style.borderRadius = '8px'; row.style.marginBottom = '6px';
      const txt = document.createElement('span'); txt.textContent = w.label + ' · ' + fmtMin(w.startMinuteOfDay) + ' – ' + fmtMin(w.endMinuteOfDay);
      const toggle = document.createElement('button'); toggle.className = 'organ-btn' + (w.protectedWindow ? ' primary' : '');
      toggle.textContent = w.protectedWindow ? 'Protected' : 'Paused';
      toggle.onclick = () => { w.protectedWindow = !w.protectedWindow; emitConstraints(); render(); };
      row.append(txt, toggle); root.appendChild(row);
    });
  }
}
render();
emitConstraints();
`;
}

function buildAudioRuntimeJs(weights: AudioWeights): string {
  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
const SCALES = {
  pentatonic: [0, 2, 4, 7, 9, 12],
  major: [0, 2, 4, 5, 7, 9, 11, 12],
  minor: [0, 2, 3, 5, 7, 8, 10, 12],
  dorian: [0, 2, 3, 5, 7, 9, 10, 12],
};
let bpm = W.bpm, playing = false, timer = null, step = 0, ctx = null;
function midiToHz(m) { return 440 * Math.pow(2, (m - 69) / 12); }
function playStep() {
  if (!ctx) { const AC = window.AudioContext || window.webkitAudioContext; if (!AC) return; ctx = new AC(); }
  if (ctx.state === 'suspended') void ctx.resume();
  const scale = SCALES[W.scale] || SCALES.pentatonic;
  const note = W.rootMidi + scale[step % scale.length] + (step % 8 === 0 ? -12 : 0);
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = W.style === 'ambient-pad' ? 'sine' : W.style === 'rhythm-pulse' ? 'triangle' : 'sawtooth';
  osc.frequency.value = midiToHz(note);
  const now = ctx.currentTime;
  const dur = Math.max(0.12, (60 / bpm) * (W.style === 'ambient-pad' ? 1.4 : 0.45));
  gain.gain.setValueAtTime(0.001, now);
  gain.gain.exponentialRampToValueAtTime(0.14, now + 0.03);
  gain.gain.exponentialRampToValueAtTime(0.001, now + dur);
  osc.connect(gain); gain.connect(ctx.destination);
  osc.start(now); osc.stop(now + dur + 0.02);
  step += 1;
  drawWave();
}
function stopPlay() { playing = false; if (timer) { clearInterval(timer); timer = null; } render(); }
function startPlay() {
  playing = true;
  playStep();
  timer = setInterval(playStep, Math.max(120, Math.round((60 / bpm) * 500)));
  render();
}
function exportWav() {
  const sr = 22050, total = sr * Math.min(12, W.durationSec);
  const buf = new ArrayBuffer(44 + total * 2), dv = new DataView(buf);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i)); };
  writeStr(0, 'RIFF'); dv.setUint32(4, 36 + total * 2, true); writeStr(8, 'WAVE');
  writeStr(12, 'fmt '); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
  dv.setUint32(24, sr, true); dv.setUint32(28, sr * 2, true); dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
  writeStr(36, 'data'); dv.setUint32(40, total * 2, true);
  const scale = SCALES[W.scale] || SCALES.pentatonic;
  const stepSamples = Math.max(2048, Math.round(sr * (60 / bpm) * 0.5));
  for (let i = 0; i < total; i++) {
    const idx = Math.floor(i / stepSamples);
    const local = (i % stepSamples) / stepSamples;
    const env = Math.sin(Math.min(1, local * 6) * Math.PI * 0.5) * Math.pow(1 - local, 1.5);
    const hz = midiToHz(W.rootMidi + scale[idx % scale.length]);
    const sample = Math.sin((2 * Math.PI * hz * i) / sr) * env * 0.45;
    dv.setInt16(44 + i * 2, Math.max(-32767, Math.min(32767, Math.round(sample * 32767))), true);
  }
  const blob = new Blob([buf], { type: 'audio/wav' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'verdant-audio.wav'; a.click();
}
let canvasEl = null;
function drawWave() {
  if (!canvasEl) return;
  const c2d = canvasEl.getContext('2d'), w = canvasEl.width, h = canvasEl.height;
  c2d.fillStyle = '#0b0f14'; c2d.fillRect(0, 0, w, h);
  c2d.strokeStyle = '#10b981'; c2d.lineWidth = 2; c2d.beginPath();
  for (let x = 0; x < w; x += 4) {
    const y = h / 2 + Math.sin(x * 0.04 + step * 0.9) * (playing ? 36 : 12) * Math.cos(x * 0.01);
    if (x === 0) c2d.moveTo(x, y); else c2d.lineTo(x, y);
  }
  c2d.stroke();
}
function render() {
  root.innerHTML = '';
  const bar = document.createElement('div'); bar.className = 'organ-row'; bar.style.justifyContent = 'space-between'; bar.style.marginBottom = '12px';
  const left = document.createElement('div'); left.className = 'organ-row';
  const playBtn = document.createElement('button'); playBtn.className = 'organ-btn primary'; playBtn.textContent = playing ? 'Stop Audio' : 'Play Synth';
  playBtn.onclick = () => (playing ? stopPlay() : startPlay());
  const tempoBtn = document.createElement('button'); tempoBtn.className = 'organ-btn'; tempoBtn.textContent = bpm + ' BPM';
  tempoBtn.onclick = () => { bpm = bpm >= 160 ? 72 : bpm + 16; if (playing) { stopPlay(); startPlay(); } else render(); };
  const scalePill = document.createElement('span'); scalePill.className = 'organ-pill'; scalePill.textContent = W.style + ' · ' + W.scale;
  left.append(playBtn, tempoBtn, scalePill);
  const wavBtn = document.createElement('button'); wavBtn.className = 'organ-btn'; wavBtn.textContent = 'Download WAV';
  wavBtn.onclick = exportWav;
  bar.append(left, wavBtn); root.appendChild(bar);
  canvasEl = document.createElement('canvas'); canvasEl.className = 'organ-canvas'; canvasEl.width = 640; canvasEl.height = 140;
  root.appendChild(canvasEl);
  drawWave();
}
render();
`;
}

export function synthesizeProceduralOrganView(
  decl: PrismDeclaration,
  spec: ProceduralOrganSpec,
): {
  readonly indexHtml: string;
  readonly stylesCss: string;
  readonly appJs: string;
} {
  const title = decl.view?.title || decl.metadata.name || 'Verdant Prism';
  const subtitle = decl.view?.subtitle || decl.concept.purpose || PRISM_MEDIA_CATALOG[spec.mediaType].shortSummary;
  const mediaLabel = PRISM_MEDIA_CATALOG[spec.mediaType].label;
  const accent = decl.theme.accent || '#0f5132';

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

  const attachedWeightsHtml =
    spec.attachedWeights && spec.attachedWeights.length > 0
      ? `\n    <div class="organ-row">${spec.attachedWeights
          .map(
            (b) =>
              `<span class="organ-pill" style="border-color:var(--v-accent)">Routed ${escHtml(b.modality)} weights: ${escHtml(b.label)}</span>`,
          )
          .join('')}</div>`
      : '';

  const disclosureHtml = decl.view?.disclosure?.trim()
    ? `\n    <footer class="organ-sub" style="text-align:center">${escHtml(decl.view.disclosure.trim())}</footer>`
    : '';

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
  <main class="organ-shell">
    <header class="organ-header">
      <div>
        <h1 class="organ-title">${escHtml(title)}</h1>
        <p class="organ-sub">${escHtml(subtitle)}</p>
      </div>
      <span class="organ-pill">${escHtml(mediaLabel)}</span>
    </header>${attachedWeightsHtml}
    <section class="organ-card" id="organ-root"></section>${disclosureHtml}
  </main>
</body>
</html>
`;

  const stylesCss = `${PROCEDURAL_BASE_CSS}\n:root { --v-accent: ${accent}; }\n`;
  const appJs = `// Procedurally synthesized by Verdant Studio from declarative weights (DEC-009: zero LLM JS).\nconst PRISM_WEIGHT_BINDINGS = ${safeJsonForScript(spec.attachedWeights ?? [])};\n${runtimeJs}\n`;
  return { indexHtml, stylesCss, appJs };
}
