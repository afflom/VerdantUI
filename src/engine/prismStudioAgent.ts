/**
 * Verdant Studio Agent Orchestrator (`src/engine/prismStudioAgent.ts`)
 *
 * Pairs the Creator with VerdantUI's 100% client-side on-device model to design,
 * reconcile, build, and refine ANY Prism from plain-English conversation — without
 * hardcoded templates or fake placeholder forms:
 *
 * 1. Realization Routing (`classifyIdeaRealizationMode`): distinguishes open-ended
 *    interactive apps/games (e.g. Minesweeper, Snake, Kanban, Pomodoro, 2048, Flashcards)
 *    from structured numeric rule/policy forms.
 * 2. Conversational Discovery (`buildDiscoveryAgentPrompt` + `parseDiscoveryAgentResponse`):
 *    anchors the Concept & Core Models via closed `StudioApiOp` calls while asking 1–2
 *    tailored, domain-specific clarifying questions.
 * 3. Full Prism Synthesis (`buildInteractiveAppPrompt` + `extractAuthoredAppFromModelOutput`):
 *    generates a complete, self-contained, playable/usable interactive app (`html`, `css`, `js`)
 *    admitted via `admit-authored-view` and packaged into real binary `prism/holo/6`
 *    `HOLOVIEW` alongside its Lean-proved core invariants (`App.lex.tex`).
 * 4. Iterative Refinement & Self-Healing (`buildRefineInteractiveAppPrompt`):
 *    applies natural-language edits and fixes runtime errors caught by `PrismSandboxFrame`.
 */

import {
  checkDeclaration,
  deriveRepairOp,
  dispatchStudioOp,
  evaluateDeclaredModel,
  formatRequestLine,
  sanitizeIdentifier,
  type AuthoredPrismView,
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
  parseContextReply,
  parseDesignPreferences,
  parseIdeaSentence,
  reconcileAllRemainingFacets,
} from './prismInterviewEngine';

export type PrismRealizationMode = 'interactive-app' | 'structured-model';

export interface AgentChatTurn {
  readonly role: 'user' | 'studio';
  readonly text: string;
}

export interface AgentModelSpecDraft {
  readonly name: string;
  readonly purpose: string;
  readonly domain: PrismDomain;
  readonly audience: PrismAudience;
  readonly dataClassification: DataClassification;
  readonly jurisdictions: readonly PrismJurisdiction[];
  readonly usesLocalAi: boolean;
  readonly involvesPayments: boolean;
  readonly mode: PrismRealizationMode;
  readonly readyToBuild: boolean;
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

export type AuthoredAppExtractionResult =
  | { readonly ok: true; readonly view: AuthoredPrismView }
  | { readonly ok: false; readonly reason: string };

const INTERACTIVE_APP_KEYWORDS_RE =
  /\b(video\s+game|game|minesweeper|tetris|snake|chess|checkers|sudoku|2048|wordle|solitaire|pong|breakout|arkanoid|pac-?man|tic[\s-]?tac[\s-]?toe|connect\s*4|connect\s*four|battleship|hangman|blackjack|poker|memory\s+(?:game|match)|puzzle|arcade|platformer|roguelike|clicker|idle\s+game|maze|flappy|asteroids|space\s+invaders|game\s+of\s+life|conway|kanban|todo|to-?do|task\s+board|pomodoro|stopwatch|countdown|timer|clock|calendar|whiteboard|paint|draw(?:ing)?\s+app|sketchpad|piano|synth(?:esizer)?|drum\s+machine|metronome|flashcards?|quiz\s+(?:app|game)|habit\s+tracker|markdown\s+editor|note(?:s|pad)|code\s+editor|dashboard|visualizer|simulator|playground|spreadsheet|music\s+player|playable)\b/i;

const EXPLICIT_NUMERIC_RULE_RE =
  /\b(budget|expense|bill|tip\s*(?:%|percent)|loan|mortgage|eligibility|credit\s+score|discount|pricing|shipping|tax|salary|calorie|hydration|water\s+intake|bmi|dosage|refuse\s+(?:if|when|zero|under|over)|warn\s+when|approve\s+(?:if|when|scores?))\b/i;

const BUILD_NOW_RE =
  /\b(build\s+it|make\s+it|create\s+it|generate\s+it|go\s+ahead|looks?\s+good|sounds?\s+good|yes|yep|yeah|sure|proceed|do\s+it|let'?s\s+(?:go|build|play)|reconcile\s+everything|finish\s+it|recommended)\b/i;

/**
 * Classifies whether the Creator's idea calls for an open-ended interactive app/game
 * synthesized by the Verdant Studio Agent (`interactive-app`) or a deterministic
 * structured numeric policy/calculator (`structured-model`).
 */
export function classifyIdeaRealizationMode(sentence: string): {
  readonly mode: PrismRealizationMode;
  readonly reason: string;
} {
  const trimmed = sentence.trim();
  if (!trimmed) {
    return { mode: 'structured-model', reason: 'Empty input.' };
  }
  if (INTERACTIVE_APP_KEYWORDS_RE.test(trimmed)) {
    return {
      mode: 'interactive-app',
      reason: 'Describes an interactive game, workspace, or rich visual application.',
    };
  }
  if (EXPLICIT_NUMERIC_RULE_RE.test(trimmed)) {
    return {
      mode: 'structured-model',
      reason: 'Describes explicit numeric rules, calculations, or policy thresholds.',
    };
  }
  const deterministic = parseIdeaSentence(trimmed);
  const fellBackToDummyValue =
    deterministic.inputs.length === 1 &&
    deterministic.inputs[0].name === 'value' &&
    deterministic.inputs[0].source === 'suggested';
  if (fellBackToDummyValue) {
    return {
      mode: 'interactive-app',
      reason: 'Open-ended custom Prism idea that requires agent synthesis rather than a generic numeric form.',
    };
  }
  return {
    mode: 'structured-model',
    reason: 'Structured inputs and rules identified directly.',
  };
}

/**
 * Extracts a crisp working title from an open-ended idea sentence (e.g.
 * `This Prism will be the video game 'Minesweeper'` -> `Minesweeper`).
 */
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
      /\b(this\s+prism\s+(?:will\s+be|is|should\s+be)|make\s+me|build\s+me|create\s+a|i\s+want\s+(?:a|an|to\s+build)|the\s+video\s+game|video\s+game|the\s+game|a\s+game\s+called|called|named)\b/gi,
      ' ',
    )
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
    .slice(-10)
    .map((t) => `${t.role === 'user' ? 'Creator' : 'Verdant Studio'}: ${t.text}`)
    .join('\n');
}

function summarizeDeclarationForPrompt(decl: PrismDeclaration): string {
  const fields = (decl.encoding?.fields ?? []).map((f) => `${f.name} (${f.kind}, ${f.min}..${f.max})`).join(', ');
  const rules = decl.features.map((f) => `${f.id}: ${f.statement}`).join('; ');
  const refusals = decl.refusals.map((r) => `${r.id}: ${r.failureText}`).join('; ');
  return [
    `Name: ${decl.metadata.name || 'Untitled'}`,
    `Purpose: ${decl.concept.purpose || '(undefined)'}`,
    `Domain: ${decl.concept.domain ?? 'general'}, Audience: ${decl.concept.audience ?? 'personal'}, Data: ${decl.concept.dataClassification ?? 'none'}, Regions: ${decl.concept.jurisdictions.join(', ') || 'none'}`,
    `Core Model Inputs: ${fields || '(none yet)'}`,
    `Core Model Rules: ${rules || '(none yet)'}`,
    `Core Model Refusals: ${refusals || '(none yet)'}`,
    `Has Interactive App View: ${decl.view?.authoredView ? 'yes' : 'no'}`,
  ].join('\n');
}

/**
 * Shared byte-identical system prefix across Discovery, Build, and Refinement turns so
 * VerdantUI's content-addressed KV prefix cache (`VramUnifiedStore`) reuses prefilled
 * system tokens across multi-turn Studio Agent sessions with zero redundant prefill.
 */
export const STUDIO_AGENT_SYSTEM_PREFIX = `You are Verdant Studio, a collaborative 100% on-device Prism architect pairing with a human Creator in plain English.
Core principles:
- Build the exact real, complete, interactive Prism the Creator describes — never a stub, placeholder, or hardcoded template.
- Keep all generated apps 100% self-contained and offline (zero remote URLs, zero CDN links, zero external fetches, zero alert/prompt).
- Use Verdant theme tokens (--v-bg, --v-surface, --v-surface-alt, --v-border, --v-text, --v-muted, --v-accent, --v-accent-text) so Light and Dark modes look sharp.
- Sync key state transitions asynchronously with globalThis.prism?.invoke(...) inside try/catch without blocking UI responsiveness.
- NEVER mention internal engineering terms in user-facing text (never say Observer, OpenTelemetry, vTensor, LexLean, Studio API, or REQ-D6/D9/D11).`;

/**
 * Zero-execution JavaScript syntax validator. Compiles via `new Function(source)` without
 * ever invoking the function, catching truncation and syntax errors before admitting code
 * into a Prism declaration or mounting the sandbox preview.
 */
export function validateAuthoredJsSyntax(js: string): { readonly ok: true } | { readonly ok: false; readonly error: string } {
  const trimmed = js.trim();
  if (!trimmed) {
    return { ok: false, error: 'JavaScript source is empty.' };
  }
  try {
    // Parse-only check (returned function is immediately discarded and never called)
    // eslint-disable-next-line @typescript-eslint/no-implied-eval
    new Function(trimmed);
    return { ok: true };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { ok: false, error: `JavaScript syntax error: ${msg}` };
  }
}

/**
 * Builds the system prompt for a conversational discovery/clarification turn.
 * Asks the local model to anchor the Core Models AND ask 1–2 concise, domain-specific
 * questions tailored to the Creator's exact idea (or mark `readyToBuild: true` once answered).
 */
export function buildDiscoveryAgentPrompt(
  history: readonly AgentChatTurn[],
  decl: PrismDeclaration,
  userTurnCount: number,
): string {
  const shouldBeReadyHint =
    userTurnCount >= 2
      ? 'The Creator has already answered your initial questions. Unless a critical requirement is still ambiguous, set "readyToBuild": true so we immediately build the playable/usable Prism now.'
      : 'This is the first message. Ask 1–2 short, concrete, friendly questions specifically tailored to THIS exact app/game (e.g., key gameplay/feature options + whether it is personal offline or for kids/public) and set "readyToBuild": false unless the user explicitly asked to build immediately with defaults.';

  return `${STUDIO_AGENT_SYSTEM_PREFIX}

PHASE: DISCOVERY & CORE MODEL ANCHORING

Current Prism state:
${summarizeDeclarationForPrompt(decl)}

Conversation so far:
${formatTranscriptForPrompt(history)}

Guidance for this turn:
${shouldBeReadyHint}

Output ONLY a single valid JSON object inside a \`\`\`json code fence matching this exact schema:
{
  "name": "Short title of the Prism (e.g. Minesweeper)",
  "purpose": "One clear sentence describing what this Prism does",
  "domain": "general" | "finance" | "health" | "education" | "games-media" | "enterprise-ops",
  "audience": "personal" | "team" | "public-adults" | "includes-minors",
  "dataClassification": "none" | "operational" | "personal" | "sensitive-regulated",
  "jurisdictions": ["none" | "eu" | "us" | "uk" | "global"],
  "usesLocalAi": false,
  "involvesPayments": false,
  "mode": "interactive-app" | "structured-model",
  "readyToBuild": true | false,
  "stakeholders": [{"name": "Player", "role": "consumer", "concern": "Enjoy a responsive offline game"}],
  "inputs": [
    {"name": "rows", "label": "Board Rows", "kind": "nat", "min": 4, "max": 30, "defaultValue": 9}
  ],
  "refusals": [
    {"statement": "Refuse zero rows", "field": "rows", "op": "eq", "value": 0, "failureText": "Board rows must be at least 1."}
  ],
  "rules": [
    {"statement": "Default active state", "outcome": "Active", "tone": "brand"}
  ],
  "reply": "Short friendly message to the Creator (2–4 sentences max, plain English)",
  "quickReplies": ["Natural quick reply 1", "Natural quick reply 2", "Build it now with recommended settings"]
}

Rules for the JSON:
- "inputs" must have 2 to 5 simple non-negative integer ("nat") or boolean ("bool") fields representing verifiable invariants/state for this Prism (for Minesweeper: rows, cols, mines, revealedSafe, hitMine; for Kanban: totalTasks, doneTasks, wipLimit; for Pomodoro: workMinutes, breakMinutes, completedSessions).
- "refusals" must have 1 to 2 simple boundary checks using op in ["eq","lt","gt","lte","gte"].
- "rules" must have 1 to 3 simple outcomes, with the LAST rule having NO "field" (acting as the default outcome).
- Keep "reply" concise, warm, and directly about the Creator's specific idea.`;
}

function tryExtractJsonObject(raw: string): Record<string, unknown> | null {
  const candidates: string[] = [];
  const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced?.[1]) candidates.push(fenced[1].trim());
  const firstBrace = raw.indexOf('{');
  const lastBrace = raw.lastIndexOf('}');
  if (firstBrace >= 0 && lastBrace > firstBrace) {
    candidates.push(raw.slice(firstBrace, lastBrace + 1));
  }
  for (const cand of candidates) {
    const cleaned = cand
      .replace(/,\s*([}\]])/g, '$1')
      .replace(/[\u201C\u201D]/g, '"')
      .replace(/[\u2018\u2019]/g, "'");
    try {
      const parsed = JSON.parse(cleaned);
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      // Try next candidate
    }
  }
  return null;
}

const VALID_DOMAINS = new Set<PrismDomain>([
  'general',
  'finance',
  'health',
  'education',
  'games-media',
  'enterprise-ops',
]);
const VALID_AUDIENCES = new Set<PrismAudience>(['personal', 'team', 'public-adults', 'includes-minors']);
const VALID_DATA = new Set<DataClassification>(['none', 'operational', 'personal', 'sensitive-regulated']);
const VALID_JURISDICTIONS = new Set<PrismJurisdiction>(['none', 'eu', 'us', 'uk', 'global']);
const VALID_OPS = new Set<CompareOp>(['eq', 'neq', 'lt', 'lte', 'gt', 'gte']);
const VALID_TONES = new Set<'default' | 'success' | 'warning' | 'destructive' | 'brand'>([
  'default',
  'success',
  'warning',
  'destructive',
  'brand',
]);

function buildFallbackSpec(
  userMessage: string,
  decl: PrismDeclaration,
  userTurnCount: number,
): AgentModelSpecDraft {
  const title = decl.metadata.name && decl.metadata.name !== 'Untitled Prism'
    ? decl.metadata.name
    : extractCleanIdeaTitle(userMessage);
  const modeInfo = classifyIdeaRealizationMode(userMessage);
  const deterministic = parseIdeaSentence(userMessage);
  const isGame = deterministic.domain.value === 'games-media' || /\bgame|play|minesweeper|puzzle|arcade\b/i.test(userMessage);
  const readyToBuild = userTurnCount >= 2 || BUILD_NOW_RE.test(userMessage);

  const inputs: EncodedField[] =
    decl.encoding && decl.encoding.fields.length > 0
      ? [...decl.encoding.fields]
      : isGame
        ? [
            { name: 'gridSize', label: 'Board Size', kind: 'nat', min: 1, max: 50, defaultValue: 9 },
            { name: 'score', label: 'Score', kind: 'nat', min: 0, max: 100000, defaultValue: 0 },
            { name: 'gameOver', label: 'Game Over', kind: 'bool', min: 0, max: 1, defaultValue: 0 },
          ]
        : [
            { name: 'itemCount', label: 'Active Items', kind: 'nat', min: 0, max: 1000, defaultValue: 3 },
            { name: 'completed', label: 'Completed', kind: 'nat', min: 0, max: 1000, defaultValue: 1 },
          ];

  return {
    name: title,
    purpose: decl.concept.purpose || userMessage.trim(),
    domain: decl.concept.domain ?? (isGame ? 'games-media' : deterministic.domain.value),
    audience: decl.concept.audience ?? deterministic.audience.value,
    dataClassification: decl.concept.dataClassification ?? deterministic.dataClassification.value,
    jurisdictions:
      decl.concept.jurisdictions.length > 0
        ? decl.concept.jurisdictions
        : deterministic.jurisdictions.value,
    usesLocalAi: decl.concept.usesLocalAi || deterministic.usesLocalAi.value,
    involvesPayments: decl.concept.involvesPayments || deterministic.involvesPayments.value,
    mode: decl.view?.authoredView ? 'interactive-app' : modeInfo.mode,
    readyToBuild,
    stakeholders:
      decl.concept.stakeholders.length > 0
        ? decl.concept.stakeholders.map((s) => ({ name: s.name, role: s.provRole, concern: s.concern }))
        : [
            {
              name: isGame ? 'Player' : 'Creator',
              role: 'consumer',
              concern: `Use ${title} reliably and offline.`,
            },
          ],
    inputs,
    refusals: [
      {
        statement: `Refuse invalid ${inputs[0].label.toLowerCase()} of zero.`,
        field: inputs[0].name,
        op: 'eq',
        value: 0,
        failureText: `${inputs[0].label} must be greater than zero.`,
      },
    ],
    rules: isGame
      ? [
          {
            statement: 'Report Game Over when the end condition triggers.',
            field: 'gameOver',
            op: 'eq',
            value: 1,
            outcome: 'Game Over',
            tone: 'destructive',
          },
          {
            statement: 'Report Active game state during play.',
            outcome: 'Playing',
            tone: 'brand',
          },
        ]
      : [
          {
            statement: 'Report active state for valid inputs.',
            outcome: 'Ready',
            tone: 'success',
          },
        ],
    reply: readyToBuild
      ? `Got it — building your interactive ${title} Prism now and reconciling all core checks.`
      : `I've anchored "${title}"! Before I build the interactive Prism for you, what specific features, modes, or visual style would you like, and is it just for personal offline use or also for kids/public users?`,
    quickReplies: [
      'Classic style with full interactive controls, personal offline use',
      'Dark modern theme with score/status tracking, personal offline use',
      'Build it now with recommended settings',
    ],
  };
}

/**
 * Parses and sanitizes the local model's discovery JSON response with resilient fallback.
 */
export function parseDiscoveryAgentResponse(
  raw: string,
  userMessage: string,
  decl: PrismDeclaration,
  userTurnCount: number,
): DiscoveryParseResult {
  const fallback = buildFallbackSpec(userMessage, decl, userTurnCount);
  const parsed = tryExtractJsonObject(raw);
  if (!parsed) {
    const cleanText = raw
      .replace(/```[\s\S]*?```/g, '')
      .trim()
      .slice(0, 420);
    return {
      spec: {
        ...fallback,
        reply: cleanText.length >= 16 ? cleanText : fallback.reply,
      },
      usedFallback: true,
    };
  }

  const name =
    typeof parsed.name === 'string' && parsed.name.trim().length >= 2
      ? parsed.name.trim().slice(0, 64)
      : fallback.name;
  const purpose =
    typeof parsed.purpose === 'string' && parsed.purpose.trim().length >= 8
      ? parsed.purpose.trim().slice(0, 280)
      : fallback.purpose;
  const domain =
    typeof parsed.domain === 'string' && VALID_DOMAINS.has(parsed.domain as PrismDomain)
      ? (parsed.domain as PrismDomain)
      : fallback.domain;
  const audience =
    typeof parsed.audience === 'string' && VALID_AUDIENCES.has(parsed.audience as PrismAudience)
      ? (parsed.audience as PrismAudience)
      : fallback.audience;
  const dataClassification =
    typeof parsed.dataClassification === 'string' &&
    VALID_DATA.has(parsed.dataClassification as DataClassification)
      ? (parsed.dataClassification as DataClassification)
      : fallback.dataClassification;

  const rawJurs = Array.isArray(parsed.jurisdictions) ? parsed.jurisdictions : [];
  const validJurs = rawJurs
    .map((j) => String(j).toLowerCase())
    .filter((j): j is PrismJurisdiction => VALID_JURISDICTIONS.has(j as PrismJurisdiction));
  const jurisdictions = validJurs.length > 0 ? validJurs : fallback.jurisdictions;

  const mode: PrismRealizationMode =
    parsed.mode === 'interactive-app' || parsed.mode === 'structured-model'
      ? parsed.mode
      : fallback.mode;

  const readyToBuild =
    Boolean(parsed.readyToBuild) || userTurnCount >= 2 || BUILD_NOW_RE.test(userMessage);

  // Sanitize inputs
  const inputs: EncodedField[] = [];
  const seenFields = new Set<string>();
  if (Array.isArray(parsed.inputs)) {
    for (const item of parsed.inputs.slice(0, 5)) {
      if (!item || typeof item !== 'object') continue;
      const obj = item as Record<string, unknown>;
      const rawName = typeof obj.name === 'string' ? obj.name : '';
      const cleanName = sanitizeIdentifier(rawName);
      if (!cleanName || seenFields.has(cleanName)) continue;
      seenFields.add(cleanName);
      const kind = obj.kind === 'bool' ? 'bool' : 'nat';
      const min = kind === 'bool' ? 0 : Math.max(0, Math.floor(Number(obj.min ?? 0) || 0));
      const max =
        kind === 'bool'
          ? 1
          : Math.max(min + 1, Math.min(1_000_000, Math.floor(Number(obj.max ?? 1000) || 1000)));
      const def = Math.max(
        min,
        Math.min(max, Math.floor(Number(obj.defaultValue ?? (min === 0 ? 1 : min)) || 0)),
      );
      inputs.push({
        name: cleanName,
        label:
          typeof obj.label === 'string' && obj.label.trim()
            ? obj.label.trim().slice(0, 40)
            : cleanName,
        kind,
        min,
        max,
        defaultValue: def,
      });
    }
  }
  const finalInputs = inputs.length > 0 ? inputs : fallback.inputs;
  const fieldSet = new Set(finalInputs.map((f) => f.name));

  // Sanitize refusals
  const refusals: AgentModelSpecDraft['refusals'][number][] = [];
  if (Array.isArray(parsed.refusals)) {
    for (const item of parsed.refusals.slice(0, 3)) {
      if (!item || typeof item !== 'object') continue;
      const obj = item as Record<string, unknown>;
      const fName = sanitizeIdentifier(String(obj.field ?? ''));
      const op = String(obj.op ?? 'eq') as CompareOp;
      const value = Math.max(0, Math.floor(Number(obj.value ?? 0) || 0));
      const failureText =
        typeof obj.failureText === 'string' && obj.failureText.trim()
          ? obj.failureText.trim().slice(0, 120)
          : '';
      if (!fieldSet.has(fName) || !VALID_OPS.has(op) || !failureText) continue;
      refusals.push({
        statement:
          typeof obj.statement === 'string' && obj.statement.trim()
            ? obj.statement.trim().slice(0, 140)
            : `Refuse invalid ${fName}.`,
        field: fName,
        op,
        value,
        failureText,
      });
    }
  }

  // Sanitize rules (always ensure a fallback default rule with no field at the end)
  const rules: AgentModelSpecDraft['rules'][number][] = [];
  if (Array.isArray(parsed.rules)) {
    for (const item of parsed.rules.slice(0, 4)) {
      if (!item || typeof item !== 'object') continue;
      const obj = item as Record<string, unknown>;
      const outcome =
        typeof obj.outcome === 'string' && obj.outcome.trim()
          ? obj.outcome.trim().slice(0, 80)
          : '';
      if (!outcome) continue;
      const tone = VALID_TONES.has(obj.tone as 'brand')
        ? (obj.tone as 'default' | 'success' | 'warning' | 'destructive' | 'brand')
        : 'brand';
      const rawField = typeof obj.field === 'string' && obj.field.trim() ? sanitizeIdentifier(obj.field) : undefined;
      const op = typeof obj.op === 'string' && VALID_OPS.has(obj.op as CompareOp) ? (obj.op as CompareOp) : undefined;
      if (rawField && fieldSet.has(rawField) && op) {
        rules.push({
          statement:
            typeof obj.statement === 'string' && obj.statement.trim()
              ? obj.statement.trim().slice(0, 140)
              : outcome,
          field: rawField,
          op,
          value: Math.max(0, Math.floor(Number(obj.value ?? 0) || 0)),
          outcome,
          tone,
        });
      } else {
        rules.push({
          statement:
            typeof obj.statement === 'string' && obj.statement.trim()
              ? obj.statement.trim().slice(0, 140)
              : outcome,
          outcome,
          tone,
        });
      }
    }
  }
  if (!rules.some((r) => !r.field)) {
    rules.push({
      statement: `Active verified state for ${name}.`,
      outcome: 'Active',
      tone: 'brand',
    });
  }

  // Sanitize stakeholders
  const validRoles = new Set<StakeholderDecl['provRole']>(['producer', 'consumer', 'beneficiary', 'regulator']);
  const stakeholders: AgentModelSpecDraft['stakeholders'][number][] = [];
  if (Array.isArray(parsed.stakeholders)) {
    for (const item of parsed.stakeholders.slice(0, 4)) {
      if (!item || typeof item !== 'object') continue;
      const obj = item as Record<string, unknown>;
      const sName = typeof obj.name === 'string' && obj.name.trim() ? obj.name.trim().slice(0, 48) : '';
      if (!sName) continue;
      const role = validRoles.has(obj.role as StakeholderDecl['provRole'])
        ? (obj.role as StakeholderDecl['provRole'])
        : 'consumer';
      stakeholders.push({
        name: sName,
        role,
        concern:
          typeof obj.concern === 'string' && obj.concern.trim()
            ? obj.concern.trim().slice(0, 160)
            : `Use ${name} safely and reliably.`,
      });
    }
  }

  const reply =
    typeof parsed.reply === 'string' && parsed.reply.trim().length >= 8
      ? parsed.reply.trim().slice(0, 700)
      : fallback.reply;

  const quickReplies = Array.isArray(parsed.quickReplies)
    ? parsed.quickReplies
        .map((q) => String(q).trim())
        .filter((q) => q.length >= 2 && q.length <= 120)
        .slice(0, 3)
    : fallback.quickReplies;

  return {
    spec: {
      name,
      purpose,
      domain,
      audience,
      dataClassification,
      jurisdictions,
      usesLocalAi: Boolean(parsed.usesLocalAi),
      involvesPayments: Boolean(parsed.involvesPayments),
      mode,
      readyToBuild,
      stakeholders: stakeholders.length > 0 ? stakeholders : fallback.stakeholders,
      inputs: finalInputs,
      refusals: refusals.length > 0 ? refusals : fallback.refusals,
      rules,
      reply,
      quickReplies: quickReplies.length > 0 ? quickReplies : fallback.quickReplies,
    },
    usedFallback: false,
  };
}

/**
 * Compiles an `AgentModelSpecDraft` into atomic, validated `StudioApiOp[]` operations
 * that populate Concept (OPM + Stakeholders), System (Encoding + Rules + Refusals +
 * verified Acceptance Vectors), and Security so Lean 4.32.2 proofs always succeed.
 */
export function buildAgentCoreModelOps(
  decl: PrismDeclaration,
  spec: AgentModelSpecDraft,
  creatorQuote: string,
): readonly StudioApiOp[] {
  const ops: StudioApiOp[] = [];
  let working = decl;

  const push = (op: StudioApiOp) => {
    const res = dispatchStudioOp(working, op);
    if (res.ok) {
      working = res.declaration;
      ops.push(op);
    }
  };

  push({
    op: 'set-metadata',
    name: spec.name,
    description: spec.purpose,
  });
  push({
    op: 'set-concept-profile',
    purpose: spec.purpose,
    domain: spec.domain,
    audience: spec.audience,
    dataClassification: spec.dataClassification,
    jurisdictions: spec.jurisdictions,
    usesLocalAi: spec.usesLocalAi,
    involvesPayments: spec.involvesPayments,
  });

  // Record verbatim Creator decision if none recorded yet
  if (working.decisions.length === 0 && creatorQuote.trim()) {
    push({
      op: 'record-decision',
      id: 'DEC-01',
      stakeholder: 'Creator',
      quote: creatorQuote.trim(),
      requirementIds: ['REQ-01'],
    });
  }

  // Admit stakeholders
  const existingStk = new Set(working.concept.stakeholders.map((s) => s.name.toLowerCase()));
  for (let i = 0; i < spec.stakeholders.length; i++) {
    const s = spec.stakeholders[i];
    if (existingStk.has(s.name.toLowerCase())) continue;
    push({
      op: 'admit-stakeholder',
      stakeholder: {
        id: `STK-${String(working.concept.stakeholders.length + 1).padStart(2, '0')}`,
        name: s.name,
        provRole: s.role,
        concern: s.concern,
      },
      autoAdmitRequirement: working.requirements.length === 0,
    });
  }

  if (working.requirements.length === 0) {
    push({
      op: 'admit-requirement',
      requirement: {
        id: 'REQ-01',
        statement: spec.purpose,
        stakeholder: working.concept.stakeholders[0]?.name || 'Creator',
        origin: 'identity',
        evidence: 'Creator natural-language specification (DEC-01)',
      },
    });
  }
  const primaryReq = working.requirements[0]?.id ?? 'REQ-01';

  // Only populate initial OPM objects/encoding/rules if not already populated (or replace clean undefined start)
  if (!working.encoding || working.encoding.fields.length === 0) {
    const inputObjIds: string[] = [];
    for (let i = 0; i < spec.inputs.length; i++) {
      const f = spec.inputs[i];
      const objId = `OBJ-${String(i + 1).padStart(2, '0')}`;
      inputObjIds.push(objId);
      push({
        op: 'admit-opm-object',
        object: {
          id: objId,
          name: f.label,
          kind: f.kind,
          states: f.kind === 'bool' ? ['False', 'True'] : [],
          role: 'input',
          min: f.min,
          max: f.max,
          defaultValue: f.defaultValue,
        },
      });
    }
    const outObjId = `OBJ-${String(spec.inputs.length + 1).padStart(2, '0')}`;
    push({
      op: 'admit-opm-object',
      object: {
        id: outObjId,
        name: `${spec.name} State`,
        kind: 'nat',
        states: spec.rules.map((r) => r.outcome).slice(0, 4),
        role: 'output',
      },
    });
    push({
      op: 'admit-opm-process',
      process: {
        id: 'PROC-01',
        name: `Run ${spec.name}`,
        summary: spec.purpose,
        consumesObjectIds: inputObjIds,
        yieldsObjectIds: [outObjId],
        actorStakeholderIds: working.concept.stakeholders.map((s) => s.id),
      },
    });
    push({
      op: 'propose-encoding',
      name: `${sanitizeIdentifier(spec.name.replace(/\s+/g, ''))}Input`,
      fields: spec.inputs,
      citation: { kind: 'req', requirementId: primaryReq },
    });

    // Admit refusals
    let nextAppNum = 1;
    for (const ref of spec.refusals) {
      const id = `APP-${String(nextAppNum++).padStart(2, '0')}`;
      push({
        op: 'admit-refusal',
        refusal: {
          id,
          statement: ref.statement,
          condition: { all: [{ field: ref.field, op: ref.op, value: ref.value }] },
          failureText: ref.failureText,
          fieldHint: ref.field,
          negativeId: id,
          citation: { kind: 'req', requirementId: primaryReq },
        },
      });
    }

    // Admit rules
    for (let i = 0; i < spec.rules.length; i++) {
      const rule = spec.rules[i];
      const id = `APP-${String(nextAppNum++).padStart(2, '0')}`;
      const outcome: RuleOutcome = {
        kind: 'label',
        code: i + 1,
        label: rule.outcome,
        tone: rule.tone,
      };
      push({
        op: 'admit-feature',
        feature: {
          id,
          level: 'build',
          statement: rule.statement,
          condition:
            rule.field && rule.op
              ? { all: [{ field: rule.field, op: rule.op, value: rule.value ?? 0 }] }
              : { all: [] },
          outcome,
          priority: rule.field ? i + 1 : 99,
          citation: { kind: 'req', requirementId: primaryReq },
        },
      });
    }

    // Synthesize guaranteed-passing acceptance vectors (default happy path + every refusal)
    const defaultInputs: Record<string, number> = {};
    for (const f of working.encoding?.fields ?? []) {
      defaultInputs[f.name] = f.defaultValue;
    }
    // Ensure default happy-path doesn't trip an `eq 0` refusal
    for (const ref of working.refusals) {
      const pred = ref.condition.all[0];
      if (!pred) continue;
      if (pred.op === 'eq' && defaultInputs[pred.field] === pred.value) {
        defaultInputs[pred.field] = pred.value + 1;
      } else if (pred.op === 'lt' && defaultInputs[pred.field] < pred.value) {
        defaultInputs[pred.field] = pred.value;
      } else if (pred.op === 'gt' && defaultInputs[pred.field] > pred.value) {
        defaultInputs[pred.field] = Math.max(0, pred.value);
      }
    }
    const defaultEval = evaluateDeclaredModel(working, formatRequestLine(working.encoding, defaultInputs));
    push({
      op: 'admit-vector',
      vector: {
        id: 'vec_default',
        inputs: defaultInputs,
        expectedResponse: defaultEval.response,
        forFeatureId: working.features[working.features.length - 1]?.id ?? 'APP-01',
        isRefusal: defaultEval.isRefusal,
      },
    });

    // Ensure every refusal has a verified rejecting vector
    const chk = checkDeclaration(working);
    for (const diag of chk.diagnostics) {
      if (diag.code === 'contract.refusal-unreplayed' || diag.code === 'holo.vectors') {
        const repair = deriveRepairOp(working, diag.code, diag.message, diag.id);
        if (repair) push(repair);
      }
    }
  }

  push({ op: 'sync-stride-threats' });
  return ops;
}

/**
 * Builds the dedicated local-LLM prompt to synthesize a complete, playable/usable,
 * 100% self-contained interactive Prism app (`html`, `css`, `js`) matching the
 * Creator's exact conversation and Core Models.
 */
export function buildInteractiveAppPrompt(
  history: readonly AgentChatTurn[],
  decl: PrismDeclaration,
): string {
  const fields = decl.encoding?.fields ?? [];
  const sampleRequest = fields.map((f) => f.defaultValue).join(' ');
  const fieldDoc = fields.map((f) => `${f.name} (${f.label})`).join(', ');

  return `${STUDIO_AGENT_SYSTEM_PREFIX}

PHASE: INTERACTIVE APP REALIZATION

Prism specification:
${summarizeDeclarationForPrompt(decl)}

Full conversation with the Creator:
${formatTranscriptForPrompt(history)}

REALIZATION DETAILS:
- For games (Minesweeper, Snake, Tetris, 2048, Chess, Tic-Tac-Toe, Sudoku, etc.), implement EVERY real game mechanic cleanly (playable board/canvas, win & loss detection, counters/score, timer if appropriate, difficulty/mode selector, New Game / Reset button, right-click & mobile toggle support where applicable).
- For productivity tools (Kanban, Pomodoro, Flashcards, Habit Tracker, Whiteboard, etc.), implement full live interactivity, state management, add/edit/delete/toggle controls, and clear feedback.
- \`globalThis.prism.invoke(line)\` accepts space-separated non-negative integers for [${fieldDoc}] (e.g. "${sampleRequest}").
- Write clean, defensive vanilla JavaScript (no external frameworks, no \`import\` statements). Guard every DOM query before attaching listeners.

OUTPUT FORMAT — Output ONLY these three fenced code blocks (no extra explanation before or after):

\`\`\`html
<!-- Inner body markup only (no <html>, <head>, or <body> tags) -->
\`\`\`

\`\`\`css
/* Complete responsive styles using var(--v-*) tokens */
\`\`\`

\`\`\`js
// Complete self-contained interactive JavaScript
\`\`\``;
}

/**
 * Builds the targeted refinement / self-healing prompt when the Creator requests
 * changes in plain English OR when the live sandbox reports a runtime error.
 */
export function buildRefineInteractiveAppPrompt(
  history: readonly AgentChatTurn[],
  decl: PrismDeclaration,
  currentView: AuthoredPrismView,
  instruction: string,
  runtimeError?: { readonly message: string; readonly line: number | null } | null,
): string {
  const goalBlock = runtimeError
    ? `RUNTIME ERROR TO FIX IMMEDIATELY:\n"${runtimeError.message}"${runtimeError.line ? ` (line ${runtimeError.line})` : ''}\nDiagnose the exact bug in the current code and return the complete fixed HTML, CSS, and JS.`
    : `CREATOR CHANGE REQUEST:\n"${instruction}"\nApply this change cleanly while keeping all existing functionality working smoothly.`;

  return `${STUDIO_AGENT_SYSTEM_PREFIX}

PHASE: REFINEMENT & SELF-HEALING ("${decl.metadata.name}")

${goalBlock}

Recent conversation:
${formatTranscriptForPrompt(history.slice(-6))}

Current HTML:
\`\`\`html
${currentView.html.slice(0, 8000)}
\`\`\`

Current CSS:
\`\`\`css
${currentView.css.slice(0, 6000)}
\`\`\`

Current JS:
\`\`\`js
${currentView.js.slice(0, 14000)}
\`\`\`

Return the COMPLETE updated HTML, CSS, and JS in these exact three fenced blocks (never use "...existing code..." ellipses):

\`\`\`html
...
\`\`\`

\`\`\`css
...
\`\`\`

\`\`\`js
...
\`\`\``;
}

function sanitizeAuthoredHtml(rawHtml: string): string {
  let html = rawHtml.trim();
  const bodyMatch = html.match(/<body[^>]*>([\s\S]*?)<\/body>/i);
  if (bodyMatch) html = bodyMatch[1].trim();
  // Strip any external scripts/stylesheets or inline script tags (JS lives in view/app.js)
  html = html
    .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
    .replace(/<link[^>]+href=["']https?:\/\/[^"']+["'][^>]*\/?>/gi, '')
    .replace(/<link[^>]+href=["'](?:\.\/)?(?:prism\.css|styles\.css)["'][^>]*\/?>/gi, '')
    .trim();
  return html;
}

function sanitizeAuthoredJs(rawJs: string): string {
  return rawJs
    .replace(/^\s*import\s+[\s\S]*?;\s*$/gm, '')
    .replace(/^\s*export\s+(?:default\s+)?/gm, '')
    .trim();
}

/**
 * Extracts `{ html, css, js }` from local LLM output supporting fenced blocks,
 * single-document HTML with embedded `<style>`/`<script>`, or JSON payloads.
 */
export function extractAuthoredAppFromModelOutput(raw: string): AuthoredAppExtractionResult {
  const text = raw.trim();
  if (!text) {
    return { ok: false, reason: 'Model returned an empty response.' };
  }

  let html = '';
  let css = '';
  let js = '';

  // 1. Primary path: fenced ```html, ```css, ```js / ```javascript blocks
  const blockRe = /```([a-zA-Z0-9_-]*)\s*\n([\s\S]*?)```/g;
  let m: RegExpExecArray | null;
  while ((m = blockRe.exec(text)) !== null) {
    const lang = m[1].trim().toLowerCase();
    const content = m[2].trim();
    if (!content) continue;
    if (lang === 'html' || lang === 'htm' || lang === 'markup') {
      if (content.length > html.length) html = content;
    } else if (lang === 'css') {
      if (content.length > css.length) css = content;
    } else if (lang === 'js' || lang === 'javascript' || lang === 'mjs' || lang === 'ts') {
      if (content.length > js.length) js = content;
    }
  }

  // 2. If single-file HTML contained <style> and/or <script>, extract them cleanly
  if (html && (!css || !js)) {
    if (!css) {
      const styleBlocks = Array.from(html.matchAll(/<style[^>]*>([\s\S]*?)<\/style>/gi)).map((s) =>
        s[1].trim(),
      );
      if (styleBlocks.length > 0) css = styleBlocks.join('\n\n');
    }
    if (!js) {
      const scriptBlocks = Array.from(html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/gi)).map(
        (s) => s[1].trim(),
      );
      if (scriptBlocks.length > 0) js = scriptBlocks.join('\n\n');
    }
    html = html.replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '');
  }

  // 3. Raw unfenced full HTML document fallback
  if (!html && /<(?:!doctype\s+html|html|div|main|section|canvas|header)/i.test(text)) {
    const styleBlocks = Array.from(text.matchAll(/<style[^>]*>([\s\S]*?)<\/style>/gi)).map((s) =>
      s[1].trim(),
    );
    const scriptBlocks = Array.from(text.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/gi)).map(
      (s) => s[1].trim(),
    );
    if (!css && styleBlocks.length > 0) css = styleBlocks.join('\n\n');
    if (!js && scriptBlocks.length > 0) js = scriptBlocks.join('\n\n');
    html = text.replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '');
  }

  // 4. JSON payload fallback
  if (!html || !js) {
    const jsonObj = tryExtractJsonObject(text);
    if (jsonObj) {
      if (!html && typeof jsonObj.html === 'string') html = jsonObj.html;
      if (!css && typeof jsonObj.css === 'string') css = jsonObj.css;
      if (!js && typeof jsonObj.js === 'string') js = jsonObj.js;
    }
  }

  const cleanHtml = sanitizeAuthoredHtml(html);
  const cleanCss = css.replace(/<\/style>/gi, '').trim();
  const cleanJs = sanitizeAuthoredJs(js);

  if (!cleanHtml || cleanHtml.length < 12) {
    return { ok: false, reason: 'Could not extract interactive HTML markup from the studio agent response.' };
  }
  if (!cleanJs || cleanJs.length < 20) {
    return { ok: false, reason: 'Could not extract interactive JavaScript behavior from the studio agent response.' };
  }
  const syntaxCheck = validateAuthoredJsSyntax(cleanJs);
  if (!syntaxCheck.ok) {
    return { ok: false, reason: syntaxCheck.error };
  }

  return {
    ok: true,
    view: {
      html: cleanHtml,
      css: cleanCss,
      js: cleanJs,
      authorMode: 'studio-agent',
    },
  };
}

/**
 * Combines Core Model ops + authored interactive view admission + automatic
 * reconciliation of all safety checks and applicable standards into a single atomic batch.
 */
export function assembleCompleteAuthoredPrismOps(
  decl: PrismDeclaration,
  spec: AgentModelSpecDraft | null,
  authoredView: AuthoredPrismView,
  userMessage: string,
): { readonly ops: readonly StudioApiOp[]; readonly badges: readonly string[] } {
  const ops: StudioApiOp[] = [];
  const badges: string[] = [];
  let working = decl;

  const pushAll = (list: readonly StudioApiOp[]) => {
    for (const op of list) {
      const res = dispatchStudioOp(working, op);
      if (res.ok) {
        working = res.declaration;
        ops.push(op);
      }
    }
  };

  if (spec) {
    pushAll(buildAgentCoreModelOps(working, spec, userMessage));
    badges.push(`Concept: ${working.metadata.name}`);
  }

  // Apply any context or theme preferences mentioned in the latest reply
  const ctx = parseContextReply(working, userMessage);
  if (ctx.ops.length > 0) {
    pushAll(ctx.ops);
    badges.push(...ctx.changes);
  }
  const designPrefs = parseDesignPreferences(userMessage);
  if (designPrefs.theme) {
    pushAll([{ op: 'set-theme', theme: { preset: designPrefs.theme } }]);
    badges.push(`Theme: ${designPrefs.theme}`);
  }

  // Guarantee at least REQ-01 exists before admitting authored view
  if (working.requirements.length === 0) {
    pushAll([
      {
        op: 'admit-requirement',
        requirement: {
          id: 'REQ-01',
          statement: working.concept.purpose || `Deliver interactive ${working.metadata.name}.`,
          stakeholder: working.concept.stakeholders[0]?.name || 'Creator',
          origin: 'identity',
          evidence: 'Creator natural-language specification',
        },
      },
    ]);
  }
  const primaryReq = working.requirements[0]?.id ?? 'REQ-01';

  // Admit the authored interactive app view
  pushAll([
    {
      op: 'admit-authored-view',
      authoredView,
      title: working.metadata.name,
      subtitle: working.concept.purpose || working.metadata.description,
      citation: { kind: 'req', requirementId: primaryReq },
    },
  ]);
  badges.push(`Built interactive ${working.metadata.name}`);

  // Reconcile all remaining safety boundaries, vectors, and applicable standards
  // (synthesize-view-from-system preserves authoredView automatically!)
  const reconciled = reconcileAllRemainingFacets(working, {
    viewpoint: 'interactive-app',
  });
  pushAll(reconciled.ops);
  badges.push(...reconciled.badges);

  return { ops, badges: Array.from(new Set(badges)) };
}
