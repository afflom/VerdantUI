/**
 * Verdant Studio Procedural Weight-Routing Agent (`src/engine/prismStudioAgent.ts`)
 *
 * Implements DEC-009 (Zero LLM-Written JavaScript):
 * VerdantUI's on-device LLM NEVER writes JavaScript, HTML, or CSS for Prisms.
 * Instead, it:
 *   1. Classifies the user's requested artifact across the 6 first-class `PrismMediaType`s
 *      (`picture`, `written-work`, `video`, `interactive-game`, `research-program`, `tool-organ`)
 *      and activates ONLY the sparse subset of Verdant Studio sub-model organs needed.
 *   2. Guides a short plain-English conversation to anchor the Concept, Space/Privacy boundary,
 *      and core rules — outputting ONLY structured JSON weights (`route-prism-weights`).
 *   3. Procedurally synthesizes the entire playable/usable Prism deterministically via
 *      `prismMediaOrgans.ts` + `prismCompiler.ts`, verifiable by Lean 4.32.2 in the browser.
 *   4. Applies natural-language refinements (*"make it 16x16 with 40 mines"*, *"protect 5pm-6pm"*)
 *      purely by routing updated declarative weight patches — instant, deterministic, zero syntax errors.
 */

import {
  checkDeclaration,
  deriveRepairOp,
  dispatchStudioOp,
  evaluateDeclaredModel,
  fieldsToJsonSchema,
  formatRequestLine,
  sanitizeIdentifier,
  type CompareOp,
  type DataClassification,
  type EncodedField,
  type PrismAudience,
  type PrismDeclaration,
  type PrismDomain,
  type PrismJurisdiction,
  type RuleOutcome,
  type StakeholderDecl,
  type StudioApiOp,
} from './verdantStudioApi';
import {
  parseIdeaSentence,
  reconcileAllRemainingFacets,
} from './prismInterviewEngine';
import {
  deriveActiveStudioOrgans,
  inferGameArchetypeFromText,
  normalizeOrganWeights,
  PRISM_MEDIA_CATALOG,
  type GameArchetype,
  type PrismMediaType,
  type ProceduralOrganWeights,
  type StudioFeatureOrgan,
} from './prismMediaOrgans';

export type PrismRealizationMode = 'procedural-organ' | 'structured-model';

export interface ArtifactMediaClassification {
  readonly mediaType: PrismMediaType;
  readonly activeOrgans: readonly StudioFeatureOrgan[];
  readonly mode: PrismRealizationMode;
  readonly gameArchetype?: GameArchetype;
  readonly reason: string;
}

export interface AgentChatTurn {
  readonly role: 'user' | 'studio';
  readonly text: string;
}

export interface AgentModelSpecDraft {
  readonly name: string;
  readonly purpose: string;
  readonly mediaType: PrismMediaType;
  readonly activeOrgans: readonly StudioFeatureOrgan[];
  readonly compartmentId: string;
  readonly domain: PrismDomain;
  readonly audience: PrismAudience;
  readonly dataClassification: DataClassification;
  readonly jurisdictions: readonly PrismJurisdiction[];
  readonly usesLocalAi: boolean;
  readonly involvesPayments: boolean;
  readonly mode: PrismRealizationMode;
  readonly readyToBuild: boolean;
  readonly weights: ProceduralOrganWeights;
  readonly stakeholders: readonly {
    readonly name: string;
    readonly role: StakeholderDecl['provRole'];
    readonly concern: string;
  }[];
  readonly inputs: readonly EncodedField[];
  readonly refusals: readonly {
    readonly statement: string;
    readonly field: string;
    readonly op: CompareOp;
    readonly value: number;
    readonly failureText: string;
  }[];
  readonly rules: readonly {
    readonly statement: string;
    readonly field?: string;
    readonly op?: CompareOp;
    readonly value?: number;
    readonly outcome: string;
    readonly tone: 'default' | 'success' | 'warning' | 'destructive' | 'brand';
  }[];
  readonly reply: string;
  readonly quickReplies: readonly string[];
}

export interface DiscoveryParseResult {
  readonly spec: AgentModelSpecDraft;
  readonly usedFallback: boolean;
}

const EXPLICIT_LOGO_OR_COVER_RE =
  /\b(logo|brand\s*mark|emblem|monogram|wordmark|app\s+icon|crest|cover\s+art|book\s+cover|album\s+cover|album\s+art|box\s+art|game\s+cover|magazine\s+cover)\b/i;
const COMPANION_WITH_ART_RE =
  /\b(?:with|plus|including|and)\s+(?:a\s+|an\s+|its\s+own\s+)?(?:logo|brand\s*mark|cover\s+art|book\s+cover|album\s+cover|soundtrack)\b/i;
const PICTURE_RE =
  /\b(picture|image|artwork|logo|brand\s*mark|emblem|monogram|cover\s+art|book\s+cover|album\s+cover|album\s+art|poster|illustration|painting|portrait|wallpaper|fractal|mosaic|constellation|sprite|drawing|visual\s+art|svg\s+art|landscape|nebula|seascape|sunset|sunrise|scenery)\b/i;
const SPACE_RE =
  /\b(create\s+(?:a\s+)?(?:new\s+)?space|new\s+space|space\s+called|space\s+named|curated\s+space|project\s+space)\b/i;
const AUDIO_RE =
  /\b(audio|music|song|soundtrack|soundscape|synth(?:esizer)?|melody|beat|rhythm|podcast|jingle|chime|lofi|lo-?fi|ambient\s+sound|sound\s+effects?|wav)\b/i;
const WRITTEN_WORK_RE =
  /\b(novel|book|short\s+story|story\s+draft|manuscript|screenplay|essay|memoir|poetry|poem|chapter(?:s)?|written\s+work|fiction|non-?fiction|authoring)\b/i;
const VIDEO_RE =
  /\b(video|animation|animated|motion\s+graphic|storyboard|trailer|clip|short\s+film|reel|intro\s+video)\b/i;
const GAME_RE =
  /\b(video\s+game|game|minesweeper|mine\s*sweep|tetris|snake|chess|checkers|sudoku|2048|wordle|solitaire|pong|breakout|arkanoid|pac-?man|tic[\s-]?tac[\s-]?toe|connect\s*4|connect\s*four|gomoku|lights?\s*out|memory\s+(?:game|match)|puzzle|arcade|platformer|roguelike|game\s+of\s+life|conway|rule\s*30|cellular\s+automaton|playable)\b/i;
const RESEARCH_RE =
  /\b(research\s+program|research\s+project|study\s+plan|hypothesis|hypotheses|literature\s+review|experiment\s+tracker|lab\s+notebook|scientific\s+program)\b/i;
const TOOL_ORGAN_RE =
  /\b(healthy\s+habits?|habit\s+(?:tracker|guardian|coach)|schedule\s+guard|focus\s+timer|pomodoro|quiet\s+hours?|protect\s+(?:my\s+)?(?:time|schedule|workout|exercise)|calendar\s+guard|sensory|tool\s+for|workflow\s+helper|daily\s+routine)\b/i;
const EXPLICIT_NUMERIC_RULE_RE =
  /\b(budget|expense|bill|tip\s*(?:%|percent)|loan|mortgage|eligibility|credit\s+score|discount|pricing|shipping|tax|salary|calorie|bmi|dosage|refuse\s+(?:if|when|zero|under|over)|warn\s+when|approve\s+(?:if|when|scores?))\b/i;
const BUILD_NOW_RE =
  /\b(build\s+it|make\s+it|create\s+it|generate\s+it|go\s+ahead|looks?\s+good|sounds?\s+good|yes|yep|yeah|sure|proceed|do\s+it|let'?s\s+(?:go|build|play)|reconcile\s+everything|finish\s+it|recommended)\b/i;

/**
 * Classifies the requested artifact MediaType and derives the sparse set of Verdant Studio
 * sub-model organs to activate (DEC-009).
 */
export function classifyArtifactMediaType(sentence: string): ArtifactMediaClassification {
  const trimmed = sentence.trim();
  if (!trimmed) {
    return {
      mediaType: 'tool-organ',
      activeOrgans: deriveActiveStudioOrgans('tool-organ'),
      mode: 'structured-model',
      reason: 'Empty input defaults to structured tool model.',
    };
  }
  if (SPACE_RE.test(trimmed)) {
    return {
      mediaType: 'space',
      activeOrgans: deriveActiveStudioOrgans('space'),
      mode: 'procedural-organ',
      reason: 'Creates a Verdant Space Prism organizing curated content with a live custom view.',
    };
  }
  // Explicit single-artifact Logo or Cover Art requests take precedence over "book"/"album"/"game" nouns
  // unless phrased as a companion ("build a game WITH cover art").
  if (EXPLICIT_LOGO_OR_COVER_RE.test(trimmed) && !COMPANION_WITH_ART_RE.test(trimmed)) {
    return {
      mediaType: 'picture',
      activeOrgans: deriveActiveStudioOrgans('picture'),
      mode: 'procedural-organ',
      reason: 'Activates Visual Canvas and Palette organs for procedural logo or cover art synthesis.',
    };
  }
  if (GAME_RE.test(trimmed)) {
    const inferred = inferGameArchetypeFromText(trimmed);
    return {
      mediaType: 'interactive-game',
      activeOrgans: deriveActiveStudioOrgans('interactive-game'),
      mode: 'procedural-organ',
      gameArchetype: inferred.archetype,
      reason: `Activates Game Simulation, Rules, and Palette organs (${inferred.archetype}).`,
    };
  }
  if (PICTURE_RE.test(trimmed)) {
    return {
      mediaType: 'picture',
      activeOrgans: deriveActiveStudioOrgans('picture'),
      mode: 'procedural-organ',
      reason: 'Activates only Visual Canvas and Palette organs for procedural picture generation.',
    };
  }
  if (AUDIO_RE.test(trimmed)) {
    return {
      mediaType: 'audio',
      activeOrgans: deriveActiveStudioOrgans('audio'),
      mode: 'procedural-organ',
      reason: 'Activates only Audio Synth and Palette organs for procedural audio synthesis.',
    };
  }
  if (VIDEO_RE.test(trimmed)) {
    return {
      mediaType: 'video',
      activeOrgans: deriveActiveStudioOrgans('video'),
      mode: 'procedural-organ',
      reason: 'Activates Visual Canvas, Motion Timeline, and Palette organs for procedural video synthesis.',
    };
  }
  if (WRITTEN_WORK_RE.test(trimmed)) {
    return {
      mediaType: 'written-work',
      activeOrgans: deriveActiveStudioOrgans('written-work'),
      mode: 'procedural-organ',
      reason: 'Activates Narrative Outline and Palette organs for written-work authoring.',
    };
  }
  if (RESEARCH_RE.test(trimmed)) {
    return {
      mediaType: 'research-program',
      activeOrgans: deriveActiveStudioOrgans('research-program'),
      mode: 'procedural-organ',
      reason: 'Activates Research Program, Evidence Ledger, Rules, and Privacy Space organs.',
    };
  }
  if (TOOL_ORGAN_RE.test(trimmed)) {
    return {
      mediaType: 'tool-organ',
      activeOrgans: deriveActiveStudioOrgans('tool-organ'),
      mode: 'procedural-organ',
      reason: 'Activates Tool, Sensory Stream, Impulse Emitter, and Privacy Space organs.',
    };
  }
  if (EXPLICIT_NUMERIC_RULE_RE.test(trimmed)) {
    return {
      mediaType: 'tool-organ',
      activeOrgans: deriveActiveStudioOrgans('tool-organ'),
      mode: 'structured-model',
      reason: 'Describes explicit numeric rules, calculations, or policy thresholds.',
    };
  }
  return {
    mediaType: 'tool-organ',
    activeOrgans: deriveActiveStudioOrgans('tool-organ'),
    mode: 'procedural-organ',
    reason: 'General custom Prism routed via procedural Tool organ weights.',
  };
}

export function classifyIdeaRealizationMode(sentence: string): {
  readonly mode: PrismRealizationMode;
  readonly mediaType: PrismMediaType;
  readonly activeOrgans: readonly StudioFeatureOrgan[];
  readonly reason: string;
} {
  const c = classifyArtifactMediaType(sentence);
  return {
    mode: c.mode,
    mediaType: c.mediaType,
    activeOrgans: c.activeOrgans,
    reason: c.reason,
  };
}

export function extractCleanIdeaTitle(sentence: string): string {
  const quoted = sentence.match(/["'“‘]([^"'”’]{2,48})["'”’]/);
  if (quoted && quoted[1].trim().length >= 2) {
    return quoted[1].trim();
  }
  const parsed = parseIdeaSentence(sentence);
  if (parsed.name.value && parsed.name.value !== 'Game Calculator' && parsed.name.value !== 'My Prism') {
    return parsed.name.value;
  }
  const cleaned = sentence
    .replace(
      /\b(this\s+prism\s+(?:will\s+be|is|should\s+be)|please|generate|create|draw|paint|render|compose|make\s+me|build\s+me|make|build|i\s+want\s+(?:a|an|to\s+build)|the\s+video\s+game|video\s+game|the\s+game|a\s+game\s+called|a\s+picture\s+of|an\s+image\s+of|a\s+photo\s+of|a\s+video\s+of|a\s+song\s+about|picture\s+of|image\s+of|picture|image|photo|illustration|artwork|called|named)\b/gi,
      ' ',
    )
    .replace(/\b(a|an|the|of|for|at|in|on|with)\b/gi, ' ')
    .replace(/[^a-zA-Z0-9\s+-]/g, ' ')
    .trim();
  const words = cleaned.split(/\s+/).filter((w) => w.length > 1).slice(0, 4);
  if (words.length > 0) {
    return words.map((w) => w[0].toUpperCase() + w.slice(1)).join(' ');
  }
  return 'Custom Prism';
}

function formatTranscriptForPrompt(history: readonly AgentChatTurn[]): string {
  if (history.length === 0) return '(start of conversation)';
  return history
    .slice(-8)
    .map((t) => `${t.role === 'user' ? 'Creator' : 'Verdant Studio'}: ${t.text}`)
    .join('\n');
}

/**
 * Shared byte-identical system prefix across Discovery and Refinement turns so VerdantUI's
 * KV prefix cache reuses prefilled tokens across multi-turn Studio conversations.
 * Explicitly enforces DEC-009 (ZERO JavaScript/HTML/CSS generation) and DEC-011 (zero jargon).
 */
export const STUDIO_AGENT_SYSTEM_PREFIX = `You are Verdant Studio, a collaborative 100% on-device Prism designer pairing with a human Creator in plain English.
Core principles:
- NEVER write JavaScript, HTML, or CSS code blocks. Verdant Studio procedurally synthesizes the Prism purely from structured JSON weights you route.
- Identify the requested artifact mediaType ("picture", "audio", "written-work", "video", "interactive-game", "research-program", "tool-organ", or "space") so only the relevant Studio features activate.
- Output ONLY one concise JSON object routing weights and asking at most 1-2 brief, plain-English clarifying questions when helpful.
- NEVER mention internal engineering terms in user-facing text (never say Observer, OpenTelemetry, vTensor, LexLean, Holon, FK-07, Q10, or REQ-*).`;

function weightSchemaHintForMediaType(mediaType: PrismMediaType): string {
  switch (mediaType) {
    case 'interactive-game':
      return `weights schema for interactive-game:
  - grid-sweep (Minesweeper): { "mediaType":"interactive-game", "archetype":"grid-sweep", "rows":9, "cols":9, "mines":10, "firstClickSafe":true, "floodReveal":true, "chordReveal":true, "showTimer":true, "visualStyle":"modern" }
  - grid-puzzle: { "mediaType":"interactive-game", "archetype":"grid-puzzle", "variant":"merge-2048|memory-match|lights-out", "rows":4, "cols":4, "targetScore":2048 }
  - turn-board: { "mediaType":"interactive-game", "archetype":"turn-board", "rows":6, "cols":7, "winLength":4, "gravity":true, "vsCpu":true }
  - arcade-canvas: { "mediaType":"interactive-game", "archetype":"arcade-canvas", "variant":"snake|breakout|pong", "speed":5, "gridCols":20, "gridRows":16, "wrapWalls":false }
  - cellular-sim: { "mediaType":"interactive-game", "archetype":"cellular-sim", "variant":"game-of-life|rule-30", "rows":24, "cols":36, "tickMs":140 }`;
    case 'picture':
      return `weights schema for picture: { "mediaType":"picture", "generator":"landscape-vista|cosmic-nebula|architectural-scene|botanical-garden|ocean-seascape|abstract-flow|geometric-mosaic|constellation-atlas|poster-editorial|pixel-scene", "lighting":"golden-hour|twilight|midnight|daylight|neon|studio", "width":800, "height":500, "symmetry":6, "density":58, "seed":4219, "palette":["#0b1714","#10b981","#f59e0b","#ecfdf5"], "caption":"Title" }`;
    case 'audio':
      return `weights schema for audio: { "mediaType":"audio", "style":"ambient-pad|melodic-arpeggio|rhythm-pulse|sound-effects", "bpm":96, "rootMidi":60, "scale":"pentatonic|major|minor|dorian", "durationSec":8, "title":"Title" }`;
    case 'written-work':
      return `weights schema for written-work: { "mediaType":"written-work", "genre":"Fiction", "title":"...", "premise":"...", "voice":"...", "targetWordCount":25000, "allowLocalAssist":true, "chapters":[{"title":"...","beat":"...","summary":"...","body":""}] }`;
    case 'video':
      return `weights schema for video: { "mediaType":"video", "aspectRatio":"16:9|1:1|9:16", "fps":30, "style":"kinetic-type|abstract-motion|storyboard-scenes", "palette":["#0b0f14","#10b981","#f8fafc"], "scenes":[{"title":"...","subtitle":"...","durationSec":4,"motion":"pan|pulse|orbit|rise","accent":"#10b981"}] }`;
    case 'research-program':
      return `weights schema for research-program: { "mediaType":"research-program", "topic":"...", "question":"...", "compartmentId":"research", "tracks":["..."], "hypotheses":[{"id":"H1","statement":"...","status":"open"}], "milestones":[{"title":"...","status":"planned","dueLabel":"Week 1"}] }`;
    case 'tool-organ':
      return `weights schema for tool-organ: { "mediaType":"tool-organ", "role":"habit-guardian|schedule-guard|calculator|workflow-helper", "compartmentId":"personal", "emitsOpaqueConstraints":true, "rulesSummary":"...", "scheduleWindows":[{"label":"...","startMinuteOfDay":1020,"endMinuteOfDay":1080,"daysOfWeek":[0,1,2,3,4,5,6],"protectedWindow":true}], "metrics":[{"label":"...","target":1,"unit":"today"}] }`;
    case 'space':
      return `weights schema for space: { "mediaType":"space", "title":"Space Name", "description":"What this space curates", "layout":"grid|board|showcase|timeline", "accent":"#0f5132", "items":[] }`;
  }
}

export function buildWeightRoutingDiscoveryPrompt(
  history: readonly AgentChatTurn[],
  decl: PrismDeclaration,
  userTurnCount: number,
  mediaTypeHint?: PrismMediaType,
): string {
  const mediaType =
    mediaTypeHint ??
    decl.concept.mediaType ??
    classifyArtifactMediaType(history.map((h) => h.text).join(' ')).mediaType;
  const activeOrgans = deriveActiveStudioOrgans(mediaType);
  const existingWeights = decl.view?.proceduralOrgan?.weights
    ? JSON.stringify(decl.view.proceduralOrgan.weights)
    : '(none yet)';

  return `${STUDIO_AGENT_SYSTEM_PREFIX}

Active Artifact MediaType: ${mediaType} (${PRISM_MEDIA_CATALOG[mediaType].label})
Activated Studio Organs (sparse): ${activeOrgans.join(', ')}
Current Prism: ${decl.metadata.name || 'Untitled'} — ${decl.concept.purpose || '(new)'}
Current Weights: ${existingWeights}

Conversation so far:
${formatTranscriptForPrompt(history)}

Turn guidance:
- On turn 1 (current user turns = ${userTurnCount}), unless the Creator said "build it now" or gave full details, set "readyToBuild": false and ask 1–2 short, concrete questions tailored to ${PRISM_MEDIA_CATALOG[mediaType].activeWizardQuestions.join(' and ')}.
- On turn 2+ or when ready, set "readyToBuild": true and populate complete "weights".
- STRICT: Output ONLY one valid JSON object. NEVER output JavaScript, HTML, or CSS.

${weightSchemaHintForMediaType(mediaType)}

JSON shape:
{
  "name": "Short title",
  "purpose": "One-sentence description",
  "mediaType": "${mediaType}",
  "compartmentId": "personal|work|health|research",
  "readyToBuild": true,
  "weights": { ... },
  "reply": "Warm plain-English reply (<= 3 sentences)",
  "quickReplies": ["Option 1", "Option 2", "Build it now"]
}`;
}

/** Compatibility alias for callers using the previous discovery function name. */
export function buildDiscoveryAgentPrompt(
  history: readonly AgentChatTurn[],
  decl: PrismDeclaration,
  userTurnCount: number,
): string {
  return buildWeightRoutingDiscoveryPrompt(history, decl, userTurnCount);
}

export function buildWeightRefinementPrompt(
  history: readonly AgentChatTurn[],
  decl: PrismDeclaration,
  instruction: string,
): string {
  const mediaType: PrismMediaType =
    decl.view?.proceduralOrgan?.mediaType ??
    decl.concept.mediaType ??
    classifyArtifactMediaType(`${decl.metadata.name} ${instruction}`).mediaType;
  const currentWeights = decl.view?.proceduralOrgan?.weights ?? normalizeOrganWeights(mediaType, undefined, instruction, decl.metadata.name);

  return `${STUDIO_AGENT_SYSTEM_PREFIX}

Task: Route an updated JSON weight patch to refine the existing "${decl.metadata.name}" (${PRISM_MEDIA_CATALOG[mediaType].label}) Prism based on the Creator's request.
STRICT: NEVER write JavaScript, HTML, or CSS. Output ONLY one JSON object containing the updated "weights" and a brief "reply".

Current Weights:
${JSON.stringify(currentWeights)}

${weightSchemaHintForMediaType(mediaType)}

Recent Conversation:
${formatTranscriptForPrompt(history)}

Creator's Refinement Request: ${instruction}

Output ONLY:
{
  "name": "${decl.metadata.name}",
  "purpose": "${decl.concept.purpose || decl.metadata.description}",
  "mediaType": "${mediaType}",
  "compartmentId": "${decl.concept.compartmentId || 'personal'}",
  "readyToBuild": true,
  "weights": { ...updated weights... },
  "reply": "One-sentence summary of the weight change",
  "quickReplies": ["Try another preset", "Adjust visuals"]
}`;
}

function extractFirstJsonObject(raw: string): Record<string, unknown> | null {
  const cleaned = raw
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/^```(?:json)?\s*/gim, '')
    .replace(/```\s*$/gim, '')
    .trim();
  const start = cleaned.indexOf('{');
  if (start < 0) return null;
  let depth = 0;
  let inStr = false;
  let escaped = false;
  for (let i = start; i < cleaned.length; i++) {
    const ch = cleaned[i];
    if (inStr) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inStr = false;
    } else if (ch === '"') {
      inStr = true;
    } else if (ch === '{') {
      depth += 1;
    } else if (ch === '}') {
      depth -= 1;
      if (depth === 0) {
        try {
          const parsed = JSON.parse(cleaned.slice(start, i + 1));
          if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
            return parsed as Record<string, unknown>;
          }
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

function defaultCoreInputsForMediaType(mediaType: PrismMediaType, weights: ProceduralOrganWeights): {
  readonly domain: PrismDomain;
  readonly inputs: readonly EncodedField[];
  readonly refusals: AgentModelSpecDraft['refusals'];
  readonly rules: AgentModelSpecDraft['rules'];
} {
  if (mediaType === 'interactive-game') {
    if (weights.mediaType === 'interactive-game' && weights.archetype === 'grid-sweep') {
      return {
        domain: 'games-media',
        inputs: [
          { name: 'rows', label: 'Board rows', kind: 'nat', min: 5, max: 24, defaultValue: weights.rows, unit: 'rows' },
          { name: 'cols', label: 'Board columns', kind: 'nat', min: 5, max: 30, defaultValue: weights.cols, unit: 'cols' },
          { name: 'mines', label: 'Mine count', kind: 'nat', min: 1, max: 200, defaultValue: weights.mines, unit: 'mines' },
          { name: 'status', label: 'Round state', kind: 'choice', min: 0, max: 2, defaultValue: 0, choices: ['Active', 'Cleared', 'Detonated'] },
        ],
        refusals: [
          {
            statement: 'Refuse boards with zero mines',
            field: 'mines',
            op: 'eq',
            value: 0,
            failureText: 'Board must place at least one mine.',
          },
        ],
        rules: [
          {
            statement: 'Declare victory when all safe cells are cleared',
            field: 'status',
            op: 'eq',
            value: 1,
            outcome: 'Board Cleared — Victory',
            tone: 'success',
          },
          {
            statement: 'End round when a mine is revealed',
            field: 'status',
            op: 'eq',
            value: 2,
            outcome: 'Mine Detonated — Round Over',
            tone: 'destructive',
          },
          {
            statement: 'Continue active round',
            outcome: 'Round In Progress',
            tone: 'brand',
          },
        ],
      };
    }
    return {
      domain: 'games-media',
      inputs: [
        { name: 'score', label: 'Current score', kind: 'nat', min: 0, max: 100000, defaultValue: 0 },
        { name: 'moves', label: 'Moves taken', kind: 'nat', min: 0, max: 10000, defaultValue: 0 },
        { name: 'status', label: 'Round state', kind: 'choice', min: 0, max: 2, defaultValue: 0, choices: ['Active', 'Won', 'Over'] },
      ],
      refusals: [],
      rules: [
        { statement: 'Celebrate win state', field: 'status', op: 'eq', value: 1, outcome: 'Victory', tone: 'success' },
        { statement: 'Record game over state', field: 'status', op: 'eq', value: 2, outcome: 'Game Over', tone: 'destructive' },
        { statement: 'Keep playing', outcome: 'Active', tone: 'brand' },
      ],
    };
  }

  if (mediaType === 'picture') {
    return {
      domain: 'games-media',
      inputs: [
        { name: 'symmetry', label: 'Symmetry order', kind: 'nat', min: 1, max: 16, defaultValue: 6 },
        { name: 'density', label: 'Element density', kind: 'nat', min: 8, max: 120, defaultValue: 42 },
      ],
      refusals: [
        { statement: 'Refuse zero symmetry', field: 'symmetry', op: 'eq', value: 0, failureText: 'Symmetry must be at least 1.' },
      ],
      rules: [{ statement: 'Render balanced procedural composition', outcome: 'Artwork Ready', tone: 'success' }],
    };
  }

  if (mediaType === 'audio') {
    return {
      domain: 'games-media',
      inputs: [
        { name: 'bpm', label: 'Tempo (BPM)', kind: 'nat', min: 45, max: 200, defaultValue: 96, unit: 'bpm' },
        { name: 'duration', label: 'Clip length (sec)', kind: 'nat', min: 4, max: 30, defaultValue: 8, unit: 's' },
      ],
      refusals: [
        { statement: 'Refuse zero tempo', field: 'bpm', op: 'eq', value: 0, failureText: 'Tempo must be greater than zero.' },
      ],
      rules: [{ statement: 'Synthesize audio clip cleanly', outcome: 'Audio Ready', tone: 'success' }],
    };
  }

  if (mediaType === 'written-work') {
    return {
      domain: 'education',
      inputs: [
        { name: 'chapters', label: 'Chapters', kind: 'nat', min: 1, max: 100, defaultValue: 3 },
        { name: 'words', label: 'Current words', kind: 'nat', min: 0, max: 200000, defaultValue: 0 },
      ],
      refusals: [
        { statement: 'Refuse empty manuscript outline', field: 'chapters', op: 'eq', value: 0, failureText: 'At least one chapter is required.' },
      ],
      rules: [{ statement: 'Track manuscript progress', outcome: 'Drafting Active', tone: 'brand' }],
    };
  }

  if (mediaType === 'video') {
    return {
      domain: 'games-media',
      inputs: [
        { name: 'scenes', label: 'Scene count', kind: 'nat', min: 1, max: 20, defaultValue: 3 },
        { name: 'fps', label: 'Frame rate', kind: 'nat', min: 15, max: 60, defaultValue: 30 },
      ],
      refusals: [
        { statement: 'Refuse empty timeline', field: 'scenes', op: 'eq', value: 0, failureText: 'Video timeline requires at least one scene.' },
      ],
      rules: [{ statement: 'Render procedural timeline', outcome: 'Timeline Ready', tone: 'success' }],
    };
  }

  if (mediaType === 'research-program') {
    return {
      domain: 'education',
      inputs: [
        { name: 'hypotheses', label: 'Active hypotheses', kind: 'nat', min: 1, max: 50, defaultValue: 2 },
        { name: 'verified', label: 'Verified milestones', kind: 'nat', min: 0, max: 50, defaultValue: 0 },
      ],
      refusals: [
        { statement: 'Refuse research program with no hypotheses', field: 'hypotheses', op: 'eq', value: 0, failureText: 'At least one hypothesis is required.' },
      ],
      rules: [{ statement: 'Track research milestones', outcome: 'Program Active', tone: 'brand' }],
    };
  }

  if (mediaType === 'space') {
    return {
      domain: 'education',
      inputs: [
        { name: 'items', label: 'Curated items', kind: 'nat', min: 0, max: 500, defaultValue: 0 },
      ],
      refusals: [],
      rules: [{ statement: 'Organize curated Space content cleanly', outcome: 'Space Ready', tone: 'success' }],
    };
  }

  return {
    domain: 'health',
    inputs: [
      { name: 'checkins', label: 'Daily check-ins', kind: 'nat', min: 0, max: 100, defaultValue: 0 },
      { name: 'protected_windows', label: 'Protected windows', kind: 'nat', min: 0, max: 16, defaultValue: 1 },
    ],
    refusals: [],
    rules: [
      { statement: 'Honor protected windows with opaque availability', field: 'protected_windows', op: 'gte', value: 1, outcome: 'Protected & Active', tone: 'success' },
      { statement: 'Ready for check-in', outcome: 'Active', tone: 'brand' },
    ],
  };
}

function defaultQuickRepliesForMediaType(mediaType: PrismMediaType): readonly string[] {
  switch (mediaType) {
    case 'interactive-game':
      return ['Classic 9×9 (10 mines) + presets — build it', '16×16 Intermediate with 40 mines', 'Build it now with recommended defaults'];
    case 'picture':
      return ['Emerald geometric mosaic — build it', 'Botanical fractal in warm gold', 'Build it now'];
    case 'audio':
      return ['Calm ambient pentatonic soundscape — build it', '118 BPM lo-fi rhythm pulse', 'Build it now'];
    case 'written-work':
      return ['3-chapter sci-fi outline — build it', 'Short story with local drafting assist', 'Build it now'];
    case 'video':
      return ['16:9 kinetic title sequence — build it', '1:1 square abstract motion loop', 'Build it now'];
    case 'research-program':
      return ['3 hypotheses + weekly milestones — build it', 'Keep in private Research space', 'Build it now'];
    case 'tool-organ':
      return ['Protect 5pm–6pm daily', 'Daily habit check-in tracker', 'Build it now'];
    case 'space':
      return ['Grid view for mixed content', 'Showcase gallery layout', 'Build it now'];
  }
}

export function parseDiscoveryAgentResponse(
  rawModelOutput: string,
  latestUserMessage: string,
  currentDecl: PrismDeclaration,
  userTurnCount: number,
  forcedMediaType?: PrismMediaType,
): DiscoveryParseResult {
  const parsed = extractFirstJsonObject(rawModelOutput);
  const combinedHint = `${currentDecl.metadata.name !== 'Untitled Prism' ? currentDecl.metadata.name : ''} ${currentDecl.concept.purpose} ${latestUserMessage}`.trim();
  const classified = classifyArtifactMediaType(combinedHint);
  const allowedMedia: readonly PrismMediaType[] = [
    'picture',
    'audio',
    'written-work',
    'video',
    'interactive-game',
    'research-program',
    'tool-organ',
    'space',
  ];
  const parsedMedia =
    parsed && typeof parsed.mediaType === 'string' && allowedMedia.includes(parsed.mediaType as PrismMediaType)
      ? (parsed.mediaType as PrismMediaType)
      : null;
  const mediaType: PrismMediaType =
    forcedMediaType ??
    currentDecl.concept.mediaType ??
    parsedMedia ??
    classified.mediaType;
  const activeOrgans = deriveActiveStudioOrgans(mediaType);

  const fallbackTitle =
    currentDecl.metadata.name && currentDecl.metadata.name !== 'Untitled Prism'
      ? currentDecl.metadata.name
      : extractCleanIdeaTitle(latestUserMessage);
  const name =
    parsed && typeof parsed.name === 'string' && parsed.name.trim().length >= 2
      ? parsed.name.trim().slice(0, 64)
      : fallbackTitle;
  const purpose =
    parsed && typeof parsed.purpose === 'string' && parsed.purpose.trim().length >= 6
      ? parsed.purpose.trim().slice(0, 220)
      : currentDecl.concept.purpose || latestUserMessage.trim().slice(0, 220);

  const existingWeights =
    currentDecl.view?.proceduralOrgan?.mediaType === mediaType
      ? (currentDecl.view.proceduralOrgan.weights as unknown as Record<string, unknown>)
      : {};
  const rawWeightsPatch =
    parsed && parsed.weights && typeof parsed.weights === 'object' && !Array.isArray(parsed.weights)
      ? { ...existingWeights, ...(parsed.weights as Record<string, unknown>) }
      : existingWeights;

  const weights = normalizeOrganWeights(mediaType, rawWeightsPatch, combinedHint, name);
  const core = defaultCoreInputsForMediaType(mediaType, weights);
  const compartmentId =
    (parsed && typeof parsed.compartmentId === 'string' && parsed.compartmentId.trim()) ||
    currentDecl.concept.compartmentId ||
    (mediaType === 'research-program'
      ? 'research'
      : mediaType === 'tool-organ' && weights.mediaType === 'tool-organ'
        ? weights.compartmentId
        : 'personal');

  const userSaidBuild = BUILD_NOW_RE.test(latestUserMessage);
  const readyToBuild =
    userSaidBuild ||
    userTurnCount >= 2 ||
    Boolean(currentDecl.view?.proceduralOrgan) ||
    Boolean(parsed?.readyToBuild === true && userTurnCount >= 1);

  const mediaDesc = PRISM_MEDIA_CATALOG[mediaType];
  const fallbackReply = readyToBuild
    ? `Routing ${mediaDesc.label} weights for "${name}" and synthesizing your Prism now.`
    : `Got it — I've activated Verdant Studio's ${mediaDesc.label} features for "${name}". Before I synthesize it, any preference on ${mediaDesc.activeWizardQuestions.join(' or ')}? (Or say "Build it now" for smart defaults.)`;

  const reply =
    parsed && typeof parsed.reply === 'string' && parsed.reply.trim().length >= 8
      ? parsed.reply.trim().slice(0, 420)
      : fallbackReply;

  const quickReplies =
    parsed && Array.isArray(parsed.quickReplies) && parsed.quickReplies.length > 0
      ? parsed.quickReplies.map((q) => String(q).trim()).filter(Boolean).slice(0, 4)
      : defaultQuickRepliesForMediaType(mediaType);

  return {
    usedFallback: parsed === null,
    spec: {
      name,
      purpose,
      mediaType,
      activeOrgans,
      compartmentId,
      domain: currentDecl.concept.domain ?? core.domain,
      audience: currentDecl.concept.audience ?? 'personal',
      dataClassification: currentDecl.concept.dataClassification ?? 'none',
      jurisdictions: currentDecl.concept.jurisdictions.length > 0 ? currentDecl.concept.jurisdictions : ['none'],
      usesLocalAi: mediaType === 'written-work' && weights.mediaType === 'written-work' ? weights.allowLocalAssist : false,
      involvesPayments: false,
      mode: 'procedural-organ',
      readyToBuild,
      weights,
      stakeholders: [{ name: 'Creator', role: 'consumer', concern: purpose }],
      inputs: core.inputs,
      refusals: core.refusals,
      rules: core.rules,
      reply,
      quickReplies,
    },
  };
}

export function buildAgentCoreModelOps(
  currentDecl: PrismDeclaration,
  spec: AgentModelSpecDraft,
  creatorQuote: string,
): StudioApiOp[] {
  const ops: StudioApiOp[] = [];
  let working = currentDecl;
  const applyLocal = (op: StudioApiOp) => {
    const res = dispatchStudioOp(working, op);
    if (res.ok) {
      working = res.declaration;
      ops.push(op);
    }
  };

  applyLocal({ op: 'set-metadata', name: spec.name, description: spec.purpose });
  applyLocal({
    op: 'set-concept-profile',
    purpose: spec.purpose,
    domain: spec.domain,
    audience: spec.audience,
    dataClassification: spec.dataClassification,
    jurisdictions: spec.jurisdictions,
    usesLocalAi: spec.usesLocalAi,
    involvesPayments: spec.involvesPayments,
  });

  if (working.decisions.length === 0) {
    applyLocal({
      op: 'record-decision',
      stakeholder: 'Creator',
      quote: creatorQuote.slice(0, 280),
      requirementIds: ['REQ-01'],
    });
  }

  for (const s of spec.stakeholders) {
    const already = working.concept.stakeholders.some((ex) => ex.name.toLowerCase() === s.name.toLowerCase());
    if (!already) {
      applyLocal({
        op: 'admit-stakeholder',
        stakeholder: { name: s.name, provRole: s.role, concern: s.concern },
        autoAdmitRequirement: working.requirements.length === 0,
      });
    }
  }

  if (working.requirements.length === 0) {
    applyLocal({
      op: 'admit-requirement',
      requirement: {
        id: 'REQ-01',
        statement: spec.purpose,
        stakeholder: 'Creator',
        origin: 'identity',
        evidence: creatorQuote.slice(0, 200),
      },
    });
  }

  const primaryReqId = working.requirements[0]?.id ?? 'REQ-01';
  const reqCitation = { kind: 'req' as const, requirementId: primaryReqId };

  if (!working.encoding || working.encoding.fields.length === 0) {
    applyLocal({
      op: 'propose-encoding',
      name: `${sanitizeIdentifier(spec.name)}_input`,
      schema: fieldsToJsonSchema(spec.inputs),
      citation: reqCitation,
    });
  }

  const fieldNames = new Set((working.encoding?.fields ?? []).map((f) => f.name));
  if (working.refusals.length === 0) {
    spec.refusals.forEach((r, idx) => {
      const f = sanitizeIdentifier(r.field);
      if (!fieldNames.has(f)) return;
      applyLocal({
        op: 'admit-refusal',
        refusal: {
          id: `REF-${String(idx + 1).padStart(2, '0')}`,
          negativeId: `NEG-${String(idx + 1).padStart(2, '0')}`,
          statement: r.statement,
          condition: { all: [{ field: f, op: r.op, value: r.value }] },
          failureText: r.failureText,
          fieldHint: f,
          citation: reqCitation,
        },
      });
    });
  }

  if (working.features.length === 0) {
    spec.rules.forEach((rule, idx) => {
      const f = rule.field ? sanitizeIdentifier(rule.field) : undefined;
      const cond = f && fieldNames.has(f) && rule.op && rule.value !== undefined
        ? { all: [{ field: f, op: rule.op, value: rule.value }] }
        : { all: [] };
      const outcome: RuleOutcome = { kind: 'label', code: idx + 1, label: rule.outcome, tone: rule.tone };
      applyLocal({
        op: 'admit-feature',
        feature: {
          id: `F-${String(idx + 1).padStart(2, '0')}`,
          statement: rule.statement,
          condition: cond,
          outcome,
          priority: (idx + 1) * 10,
          level: 'build',
          citation: reqCitation,
        },
      });
    });
  }

  return ops;
}

/**
 * Assembles the complete closed Studio API batch that routes declarative weights onto the
 * active procedural MediaType organ and reconciles all Core Models, acceptance vectors, and standards.
 */
export function assembleProceduralPrismOps(
  currentDecl: PrismDeclaration,
  spec: AgentModelSpecDraft,
  creatorQuote: string,
): {
  readonly ops: StudioApiOp[];
  readonly badges: string[];
  readonly finalDecl: PrismDeclaration;
} {
  const ops: StudioApiOp[] = [];
  let working = currentDecl;
  const applyLocal = (op: StudioApiOp) => {
    const res = dispatchStudioOp(working, op);
    if (res.ok) {
      working = res.declaration;
      ops.push(op);
    }
  };

  for (const op of buildAgentCoreModelOps(working, spec, creatorQuote)) {
    applyLocal(op);
  }

  const primaryReqId = working.requirements[0]?.id ?? 'REQ-01';
  applyLocal({
    op: 'route-prism-weights',
    mediaType: spec.mediaType,
    activeOrgans: spec.activeOrgans,
    weights: spec.weights,
    compartmentId: spec.compartmentId,
    title: spec.name,
    eyebrow: PRISM_MEDIA_CATALOG[spec.mediaType].label,
    subtitle: spec.purpose,
    summary: `Procedurally synthesized ${PRISM_MEDIA_CATALOG[spec.mediaType].label} Prism (zero LLM JS)`,
    citation: { kind: 'req', requirementId: primaryReqId },
  });

  // Reconcile remaining Core 3 (Security) & applicable standards cleanly
  const reconciled = reconcileAllRemainingFacets(working);
  for (const op of reconciled.ops) {
    if (op.op === 'synthesize-view-from-system' || op.op === 'declare-view') continue;
    applyLocal(op);
  }

  // Ensure default acceptance vector + refusal vectors replay cleanly
  if (working.encoding && working.encoding.fields.length > 0 && working.vectors.length === 0) {
    const defaults: Record<string, number> = {};
    for (const f of working.encoding.fields) defaults[f.name] = f.defaultValue;
    const reqLine = formatRequestLine(working.encoding, defaults);
    const ev = evaluateDeclaredModel(working, reqLine);
    applyLocal({
      op: 'admit-vector',
      vector: {
        id: 'vec_default',
        inputs: defaults,
        expectedResponse: ev.response,
        forFeatureId: ev.matchedId || working.features[0]?.id || 'F-01',
        isRefusal: ev.isRefusal,
      },
    });
  }

  // Run self-healing vector repairs if any check diagnostic remains
  for (let pass = 0; pass < 3; pass++) {
    const report = checkDeclaration(working);
    if (report.ok) break;
    let repairedAny = false;
    for (const diag of report.diagnostics) {
      if (diag.severity !== 'error') continue;
      const repair = deriveRepairOp(working, diag.code, diag.message);
      if (repair && repair.op !== 'synthesize-view-from-system' && repair.op !== 'declare-view') {
        applyLocal(repair);
        repairedAny = true;
      }
    }
    if (!repairedAny) break;
  }

  const badges = [
    `${PRISM_MEDIA_CATALOG[spec.mediaType].label}`,
    `${spec.activeOrgans.length} active feature${spec.activeOrgans.length === 1 ? '' : 's'}`,
  ];
  return { ops, badges, finalDecl: working };
}
