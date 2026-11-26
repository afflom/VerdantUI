/**
 * Unified VerdantUI Cognitive Observer & Cross-Classified Prism Route Graph.
 *
 * Replaces manual UI modes (`Fast` / `Thinking` / `Agentic`) and blocking model pickers with
 * ONE unified VerdantUI model grounded in:
 *   1. Second-order cybernetic Predictive Coding & Active Inference (`predict → observe → surprise →
 *      Expected Free Energy minimization → metacognitive feedback`) per isolated conversation thread.
 *   2. Exact integer `Q15` harmonic state-space mixing (orthogonal 8-channel Walsh-Hadamard coupling
 *      + leaky modal integration, zero float drift — extending `verdant-engine` `neural.rs` & `audio_ring.rs`).
 *   3. `verdant-studio` Cross-Classified Model Coherence (`REQ-D6`, `REQ-D11`, `REQ-SETTLEMENT`):
 *      all verified deterministic operators and local GGUF substrates inhabit one 8D cognitive facet
 *      space; relevance and differentiation are computed mechanically via exact integer inner-product
 *      resonance + Prism `CanonicalObjective` (`canonicalObjectiveScore` + `canonicalTieBreak`).
 *   4. `verdant-engine` `ContainerGrantMatrix` (`FK-07` denial-wins browser capability lattice).
 */

import type { ModelContext, ThreadMessage } from '@assistant-ui/react';
import type { Tool } from 'assistant-stream';
import {
  buildModelContextFromHarness,
  type HarnessSuggestionSpec,
  type ImportedHarnessSpec,
} from './harnessConfig';
import type { LoadedModelSummary } from './llamaRuntime';
import {
  MODEL_CATALOG,
  findCatalogModelById,
  isInternalBootstrapModel,
  type CatalogModel,
} from './modelCatalog';
import {
  canonicalObjectiveScore,
  canonicalTieBreak,
  prismContentDigest,
  type PrismArtifactCostVector,
} from './prismKernel';
import type { OpfsQuotaSnapshot } from './vramBackingStore';
import type { VTensorTelemetrySnapshot } from './vtensorProvider';
import type { OtelEvent } from './observerStreamBus';
import {
  prismToolName,
  shortPrismDigest,
  type PublishedPrismSummary,
  type VerdantPrismProject,
} from './prismStudio';
import {
  formatOpaqueConstraintsForContext,
  ObserverPossessionStore,
  type ContextCompartmentId,
  type PublishedPrismOrgan,
} from './unifiedPossessionModel';

export type VerdantThinkingPolicy = 'auto' | 'on' | 'off';

// ============================================================================
// 1. Browser Capability Lattice (`verdant-engine` cybernetics.rs FK-07)
// ============================================================================

export type BrowserCapabilityOrgan =
  | 'wasm-core'
  | 'opfs-storage'
  | 'math-eval'
  | 'runtime-inspector'
  | 'workspace-notes'
  | 'text-analysis'
  | 'network-weights'
  | 'prism-runtime'
  | 'prism-storage'
  | 'prism-network';

export interface OrganGrantRecord {
  readonly explicitlyDenied: boolean;
  readonly explicitlyGranted: boolean;
}

const ALWAYS_SANDBOXED_ORGANS: ReadonlySet<BrowserCapabilityOrgan> = new Set([
  'wasm-core',
  'opfs-storage',
  'math-eval',
  'runtime-inspector',
  'workspace-notes',
  'text-analysis',
  'network-weights',
  'prism-runtime',
  'prism-storage',
]);

/**
 * Denial-wins browser capability lattice (`Foundry FK-07` / `verdant-engine` `ContainerGrantMatrix`).
 */
export class ContainerGrantMatrix {
  private readonly records = new Map<BrowserCapabilityOrgan, OrganGrantRecord>();

  grant(organ: BrowserCapabilityOrgan): void {
    const prev = this.records.get(organ);
    this.records.set(organ, {
      explicitlyDenied: prev?.explicitlyDenied ?? false,
      explicitlyGranted: true,
    });
  }

  /** Explicit denial overrides any prior or future grant until reset. */
  deny(organ: BrowserCapabilityOrgan): void {
    const prev = this.records.get(organ);
    this.records.set(organ, {
      explicitlyDenied: true,
      explicitlyGranted: prev?.explicitlyGranted ?? false,
    });
  }

  reset(organ: BrowserCapabilityOrgan): void {
    this.records.delete(organ);
  }

  isAllowed(organ: BrowserCapabilityOrgan): boolean {
    const always = ALWAYS_SANDBOXED_ORGANS.has(organ);
    const rec = this.records.get(organ);
    if (!rec) return always;
    return !rec.explicitlyDenied && (rec.explicitlyGranted || always);
  }

  snapshot(): Record<BrowserCapabilityOrgan, boolean> {
    return {
      'wasm-core': this.isAllowed('wasm-core'),
      'opfs-storage': this.isAllowed('opfs-storage'),
      'math-eval': this.isAllowed('math-eval'),
      'runtime-inspector': this.isAllowed('runtime-inspector'),
      'workspace-notes': this.isAllowed('workspace-notes'),
      'text-analysis': this.isAllowed('text-analysis'),
      'network-weights': this.isAllowed('network-weights'),
      'prism-runtime': this.isAllowed('prism-runtime'),
      'prism-storage': this.isAllowed('prism-storage'),
      'prism-network': this.isAllowed('prism-network'),
    };
  }
}

// ============================================================================
// 2. 8D Exact Integer Harmonic Cognitive State (Q10 / 0..1024 milli-scale)
// ============================================================================

export const COGNITIVE_DIM = 8;
export const COGNITIVE_UNIT = 1024; // Exact power-of-two fixed-point scale (10-bit fractional)

export const COGNITIVE_AXIS_LABELS: readonly [
  string,
  string,
  string,
  string,
  string,
  string,
  string,
  string,
] = [
  'Verification',
  'Deductive Depth',
  'Code Synthesis',
  'Local Grounding',
  'Epistemic Uncertainty',
  'Continuity',
  'Elaboration',
  'Tool Composition',
];

export type CognitiveVector = readonly [
  number, // 0: verificationDemand (exact math / proof / factual check)
  number, // 1: deductiveDepth (multi-step reasoning / design / debugging)
  number, // 2: codeSynthesis (programming / algorithms / specs)
  number, // 3: groundingNeed (live runtime / OPFS / time / notes / attachments)
  number, // 4: epistemicUncertainty (novelty / ambiguity / retry escalation)
  number, // 5: conversationalContinuity (multi-turn context resonance)
  number, // 6: outputElaboration (concise vs comprehensive)
  number, // 7: toolComposition (multi-step tool + neural synthesis)
];

function clampQ10(v: number): number {
  const n = Math.round(v);
  if (n < 0) return 0;
  if (n > COGNITIVE_UNIT) return COGNITIVE_UNIT;
  return n;
}

/** Exact integer inner product (`uor-matmul` accumulate-in-i32, encode-once). */
export function dotCognitiveQ10(a: CognitiveVector, b: CognitiveVector): number {
  let acc = 0;
  for (let i = 0; i < COGNITIVE_DIM; i++) {
    acc += Math.imul(a[i], b[i]);
  }
  return acc >> 10; // exact dyadic scale back to Q10
}

/**
 * Orthogonal 8-point Walsh-Hadamard harmonic coupling (`H_8 / 8` residual mix) paired with
 * dyadic leaky membrane integration (extending `verdant-engine` `audio_ring.rs` & `neural.rs`).
 * Preserves conservation bounds `[0, 1024]` with zero floating-point drift across architectures.
 */
export function harmonicIntegrateBelief(
  prior: CognitiveVector,
  observation: CognitiveVector,
  surpriseQ10: number,
): CognitiveVector {
  // Adaptive Kalman-like gain: higher surprise increases sensory weight (precision weighting)
  // gain in [512, 896] out of 1024 (50% .. 87.5% sensory weight)
  const sensoryGain = Math.min(896, 512 + ((surpriseQ10 * 384) >> 10));
  const priorWeight = COGNITIVE_UNIT - sensoryGain;

  // Step 1: Leaky integration of prior belief + new sensory observation
  const integrated = new Int32Array(COGNITIVE_DIM);
  for (let i = 0; i < COGNITIVE_DIM; i++) {
    integrated[i] = (prior[i] * priorWeight + observation[i] * sensoryGain) >> 10;
  }

  // Step 2: Fast Walsh-Hadamard Transform (exact integer butterfly) to compute harmonic mean
  // and orthogonal cross-channel modal energy, then gently couple 1/16th harmonic resonance
  // across correlated axes (e.g. deductiveDepth <-> codeSynthesis, verification <-> grounding).
  const h = new Int32Array(integrated);
  for (let step = 1; step < COGNITIVE_DIM; step <<= 1) {
    for (let i = 0; i < COGNITIVE_DIM; i += step << 1) {
      for (let j = 0; j < step; j++) {
        const u = h[i + j];
        const v = h[i + j + step];
        h[i + j] = u + v;
        h[i + j + step] = u - v;
      }
    }
  }
  // Low-pass harmonic damping: retain fundamental + first 4 modes fully, damp highest Nyquist noise by 1/4
  h[6] = (h[6] * 3) >> 2;
  h[7] = (h[7] * 3) >> 2;
  // Inverse FWHT (exact divide by 8 via >> 3)
  for (let step = 1; step < COGNITIVE_DIM; step <<= 1) {
    for (let i = 0; i < COGNITIVE_DIM; i += step << 1) {
      for (let j = 0; j < step; j++) {
        const u = h[i + j];
        const v = h[i + j + step];
        h[i + j] = u + v;
        h[i + j + step] = u - v;
      }
    }
  }

  // Blend 7/8 direct integrated + 1/8 harmonically filtered mode (strictly bounded [0, 1024])
  const next: number[] = new Array(COGNITIVE_DIM);
  for (let i = 0; i < COGNITIVE_DIM; i++) {
    const harmonic = h[i] >> 3;
    next[i] = clampQ10((integrated[i] * 7 + harmonic) >> 3);
  }
  return next as unknown as CognitiveVector;
}

/** Exact L1 Predictive Coding Surprise (`||obs - prior||_1 / 8`, bounded in `[0, 1024]`). */
export function computePredictionSurpriseQ10(
  prior: CognitiveVector,
  observation: CognitiveVector,
): number {
  let diffSum = 0;
  for (let i = 0; i < COGNITIVE_DIM; i++) {
    diffSum += Math.abs(observation[i] - prior[i]);
  }
  return clampQ10(diffSum >> 3);
}

// ============================================================================
// 3. Direct Deterministic Operator Detection & Safe Execution (LOD0)
// ============================================================================

export type BuiltInDirectToolName =
  | 'calculate'
  | 'current_datetime'
  | 'inspect_runtime'
  | 'workspace_notes'
  | 'analyze_text';

export type DirectToolName = BuiltInDirectToolName | `prism_${string}`;

export interface DirectRouteMatch {
  readonly toolName: DirectToolName;
  readonly args: Record<string, unknown>;
  readonly organ: BrowserCapabilityOrgan;
  readonly confidenceQ10: number;
  readonly reason: string;
  readonly prismId?: string;
  readonly prismName?: string;
  readonly prismDigest?: string;
}

/**
 * Predictable explicit addressing (`@PrismName ...` or `/open PrismName ...`) so Threads can
 * invoke Published Prisms deterministically without brittle keyword hijacking.
 */
export function detectExplicitPrismAddress(
  text: string,
  publishedPrisms?: readonly PublishedPrismSummary[],
): DirectRouteMatch | null {
  if (!publishedPrisms || publishedPrisms.length === 0) return null;
  const trimmed = text.trim();
  if (!trimmed) return null;

  const isAtMention = trimmed.startsWith('@');
  const isOpenCmd = /^\/open\s+/i.test(trimmed);
  if (!isAtMention && !isOpenCmd) return null;

  const rest = isAtMention ? trimmed.slice(1).trim() : trimmed.replace(/^\/open\s+/i, '').trim();
  if (!rest) return null;
  const lowerRest = rest.toLowerCase();

  // Match longest Prism name or slug prefix first so "Sensory Monitor" beats "Sensory"
  const sorted = [...publishedPrisms].sort((a, b) => b.name.length - a.name.length);
  for (const p of sorted) {
    const candidates = [p.name.toLowerCase(), p.slug.toLowerCase(), prismToolName(p).toLowerCase()];
    for (const cand of candidates) {
      if (!cand) continue;
      if (lowerRest === cand || lowerRest.startsWith(`${cand} `) || lowerRest.startsWith(`${cand}:`)) {
        const message = rest
          .slice(cand.length)
          .replace(/^[\s:]+/, '')
          .trim();
        return {
          toolName: prismToolName(p) as `prism_${string}`,
          args: message ? { action: 'input', message } : { action: 'status' },
          organ: 'prism-runtime',
          confidenceQ10: 996,
          reason: `Explicitly addressed Published Prism "${p.name}".`,
          prismId: p.id,
          prismName: p.name,
          prismDigest: p.digest,
        };
      }
    }
  }
  return null;
}

export interface SensoryObservation {
  readonly vector: CognitiveVector;
  readonly directMatch: DirectRouteMatch | null;
  readonly inputDigest: string;
  readonly charCount: number;
  readonly hasAttachment: boolean;
}

// Safe recursive-descent math evaluator (Zero eval / Function)
const MATH_CONSTANTS: Record<string, number> = {
  pi: Math.PI,
  e: Math.E,
  tau: Math.PI * 2,
  ln2: Math.LN2,
  ln10: Math.LN10,
};

const MATH_FUNCTIONS: Record<string, (...args: number[]) => number> = {
  sqrt: (x) => Math.sqrt(x),
  cbrt: (x) => Math.cbrt(x),
  abs: (x) => Math.abs(x),
  sin: (x) => Math.sin(x),
  cos: (x) => Math.cos(x),
  tan: (x) => Math.tan(x),
  asin: (x) => Math.asin(x),
  acos: (x) => Math.acos(x),
  atan: (x) => Math.atan(x),
  log: (x, base) => (base !== undefined ? Math.log(x) / Math.log(base) : Math.log10(x)),
  log10: (x) => Math.log10(x),
  log2: (x) => Math.log2(x),
  ln: (x) => Math.log(x),
  exp: (x) => Math.exp(x),
  floor: (x) => Math.floor(x),
  ceil: (x) => Math.ceil(x),
  round: (x) => Math.round(x),
  pow: (a, b) => Math.pow(a, b),
  min: (...xs) => Math.min(...xs),
  max: (...xs) => Math.max(...xs),
};

type MathToken =
  | { kind: 'num'; value: number }
  | { kind: 'id'; value: string }
  | { kind: 'op'; value: string };

function tokenizeMath(input: string): MathToken[] {
  const tokens: MathToken[] = [];
  let i = 0;
  const s = input.replace(/\*\*/g, '^');
  while (i < s.length) {
    const ch = s[i];
    if (/\s/.test(ch)) {
      i++;
      continue;
    }
    if (
      (ch >= '0' && ch <= '9') ||
      (ch === '.' && i + 1 < s.length && s[i + 1] >= '0' && s[i + 1] <= '9')
    ) {
      const start = i;
      while (i < s.length && ((s[i] >= '0' && s[i] <= '9') || s[i] === '.')) i++;
      if (i < s.length && (s[i] === 'e' || s[i] === 'E')) {
        i++;
        if (i < s.length && (s[i] === '+' || s[i] === '-')) i++;
        while (i < s.length && s[i] >= '0' && s[i] <= '9') i++;
      }
      const num = Number(s.slice(start, i));
      if (!Number.isFinite(num)) throw new Error(`Invalid number "${s.slice(start, i)}".`);
      tokens.push({ kind: 'num', value: num });
      continue;
    }
    if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch === '_') {
      const start = i;
      while (i < s.length && /[a-zA-Z0-9_]/.test(s[i])) i++;
      tokens.push({ kind: 'id', value: s.slice(start, i).toLowerCase() });
      continue;
    }
    if ('+-*/^%(),'.includes(ch)) {
      tokens.push({ kind: 'op', value: ch });
      i++;
      continue;
    }
    throw new Error(`Unsupported character "${ch}" in math expression.`);
  }
  return tokens;
}

/** Pure, sandboxed recursive-descent math evaluator with zero `eval`/`Function`. */
export function evaluateMathExpression(expression: string): number {
  const trimmed = expression.trim();
  if (!trimmed || trimmed.length > 512) {
    throw new Error('Math expression must be 1..512 characters.');
  }
  const tokens = tokenizeMath(trimmed);
  let pos = 0;

  const peek = (): MathToken | undefined => tokens[pos];
  const consume = (): MathToken => tokens[pos++];

  const parseExpr = (): number => {
    let left = parseTerm();
    while (peek()?.kind === 'op' && (peek()!.value === '+' || peek()!.value === '-')) {
      const op = consume().value;
      const right = parseTerm();
      left = op === '+' ? left + right : left - right;
    }
    return left;
  };

  const parseTerm = (): number => {
    let left = parseUnary();
    while (
      peek()?.kind === 'op' &&
      (peek()!.value === '*' || peek()!.value === '/' || peek()!.value === '%')
    ) {
      const op = consume().value;
      const right = parseUnary();
      if (op === '*') left *= right;
      else if (op === '/') {
        if (right === 0) throw new Error('Division by zero.');
        left /= right;
      } else {
        if (right === 0) throw new Error('Modulo by zero.');
        left %= right;
      }
    }
    return left;
  };

  const parseUnary = (): number => {
    if (peek()?.kind === 'op' && (peek()!.value === '+' || peek()!.value === '-')) {
      const op = consume().value;
      const val = parseUnary();
      return op === '-' ? -val : val;
    }
    return parsePower();
  };

  const parsePower = (): number => {
    const base = parsePrimary();
    if (peek()?.kind === 'op' && peek()!.value === '^') {
      consume();
      const exp = parseUnary();
      return Math.pow(base, exp);
    }
    return base;
  };

  const parsePrimary = (): number => {
    const tok = peek();
    if (!tok) throw new Error('Unexpected end of expression.');
    if (tok.kind === 'num') {
      consume();
      return tok.value;
    }
    if (tok.kind === 'op' && tok.value === '(') {
      consume();
      const val = parseExpr();
      const close = peek();
      if (!close || close.kind !== 'op' || close.value !== ')') {
        throw new Error('Missing closing parenthesis ")".');
      }
      consume();
      return val;
    }
    if (tok.kind === 'id') {
      consume();
      const name = tok.value;
      if (peek()?.kind === 'op' && peek()!.value === '(') {
        const fn = MATH_FUNCTIONS[name];
        if (!fn) throw new Error(`Unknown math function "${name}".`);
        consume();
        const args: number[] = [];
        if (!(peek()?.kind === 'op' && peek()!.value === ')')) {
          args.push(parseExpr());
          while (peek()?.kind === 'op' && peek()!.value === ',') {
            consume();
            args.push(parseExpr());
          }
        }
        const close = peek();
        if (!close || close.kind !== 'op' || close.value !== ')') {
          throw new Error(`Missing closing parenthesis for "${name}()".`);
        }
        consume();
        if (args.length === 0) throw new Error(`Function "${name}()" requires at least one argument.`);
        return fn(...args);
      }
      if (Object.hasOwn(MATH_CONSTANTS, name)) {
        return MATH_CONSTANTS[name];
      }
      throw new Error(`Unknown identifier "${name}".`);
    }
    throw new Error(`Unexpected token "${tok.value}".`);
  };

  const value = parseExpr();
  if (pos !== tokens.length) {
    throw new Error(`Unexpected trailing token "${tokens[pos].value}".`);
  }
  if (!Number.isFinite(value)) {
    throw new Error('Expression evaluated to a non-finite number.');
  }
  return value;
}

/** Attempts to extract a pure, unambiguous arithmetic expression that settles completely at LOD0. */
function tryExtractDirectMathExpression(text: string): string | null {
  const trimmed = text.trim();
  if (!trimmed || trimmed.length > 240) return null;
  // Don't intercept compound multi-task prompts like "Calculate X and save it to a note"
  if (/\b(and\s+(save|store|write|explain|compare|note)|why|prove|explain)\b/i.test(trimmed)) {
    return null;
  }

  let candidate = trimmed
    .replace(/^(?:please\s+)?(?:calculate|compute|evaluate|solve|what\s+is|what's)\s+/i, '')
    .replace(/[?=.\s]+$/, '')
    .trim();

  if (!candidate) return null;
  // Must contain at least one operator or math function call so a bare number isn't treated as math
  if (!/[+\-*/^%]|\b(?:sqrt|cbrt|abs|sin|cos|tan|log|log2|log10|ln|exp|pow|min|max)\s*\(/i.test(candidate)) {
    return null;
  }
  try {
    evaluateMathExpression(candidate);
    return candidate;
  } catch {
    return null;
  }
}

function detectDirectOperator(text: string, hasAttachment: boolean): DirectRouteMatch | null {
  const trimmed = text.trim();
  if (!trimmed) return null;
  const lower = trimmed.toLowerCase();

  // 1. Unambiguous pure arithmetic expression
  const mathExpr = tryExtractDirectMathExpression(trimmed);
  if (mathExpr) {
    return {
      toolName: 'calculate',
      args: { expression: mathExpr },
      organ: 'math-eval',
      confidenceQ10: 992,
      reason: 'Exact deterministic math expression settled at LOD0.',
    };
  }

  // 2. Live Verdant / vTensor runtime & OPFS inspection
  if (
    !hasAttachment &&
    trimmed.length <= 180 &&
    /\b(inspect|check|show|status|diagnostics?|telemetry)\b/.test(lower) &&
    /\b(runtime|verdant|vtensor|engine|opfs|storage|thread\s*pool|workers?|cached\s+models?)\b/.test(
      lower,
    ) &&
    !/\b(write|code|implement|refactor|design|explain\s+how)\b/.test(lower)
  ) {
    return {
      toolName: 'inspect_runtime',
      args: {},
      organ: 'runtime-inspector',
      confidenceQ10: 960,
      reason: 'Live Verdant runtime & OPFS inspection settled deterministically.',
    };
  }

  // 3. Direct current date/time query
  if (
    !hasAttachment &&
    trimmed.length <= 96 &&
    /^(?:what(?:'s|\s+is)\s+(?:the\s+)?)?(?:current\s+)?(?:date|time|day|today|timezone|local\s+time)(?:\s+(?:and|or)\s+(?:date|time|timezone))?(?:\s+(?:is\s+it|right\s+now))?[?.!]*$/i.test(
      trimmed,
    )
  ) {
    return {
      toolName: 'current_datetime',
      args: {},
      organ: 'runtime-inspector',
      confidenceQ10: 976,
      reason: 'Local clock & timezone query settled deterministically.',
    };
  }

  // 4. Direct workspace notes list/read/delete
  if (
    !hasAttachment &&
    trimmed.length <= 120 &&
    /^(?:list|show|view)\s+(?:my\s+)?(?:local\s+)?(?:workspace\s+)?notes[?.!]*$/i.test(trimmed)
  ) {
    return {
      toolName: 'workspace_notes',
      args: { action: 'list' },
      organ: 'workspace-notes',
      confidenceQ10: 968,
      reason: 'Local workspace notes listing settled deterministically.',
    };
  }

  // 5. Direct JSON validation / formatting when user explicitly pastes raw JSON
  if (
    /^(?:validate|format|parse|pretty[- ]?print|check)\s+(?:this\s+)?json[:\s]+([\s\S]+)$/i.test(
      trimmed,
    )
  ) {
    const m = trimmed.match(
      /^(?:validate|format|parse|pretty[- ]?print|check)\s+(?:this\s+)?json[:\s]+([\s\S]+)$/i,
    );
    const payload = m?.[1]?.trim().replace(/^```(?:json)?\s*|\s*```$/g, '');
    if (payload && (payload.startsWith('{') || payload.startsWith('['))) {
      return {
        toolName: 'analyze_text',
        args: { text: payload, mode: 'json' },
        organ: 'text-analysis',
        confidenceQ10: 960,
        reason: 'Deterministic JSON validation & formatting settled at LOD0.',
      };
    }
  }

  return null;
}

/**
 * Extracts an 8D exact-integer sensory observation vector (`0..1024`) from the user's message
 * and thread history (extending `verdant-engine` `kernel11CodedSlmReasoning` feature projection).
 */
export function observeInputSensoryState(
  messages: readonly ThreadMessage[],
  latestText: string,
  hasAttachment: boolean,
): SensoryObservation {
  const trimmed = latestText.trim();
  const lower = trimmed.toLowerCase();
  const directMatch = detectDirectOperator(trimmed, hasAttachment);

  // Axis 0: Verification demand (exact math, formal proof, JSON validation, factual precision)
  let verification = directMatch ? 960 : 128;
  if (/\b(prove|proof|theorem|verify|exact|invariant|formal|validate|calculate|compute|equation)\b/.test(lower)) {
    verification += 448;
  }
  if (/[0-9]+\s*[+\-*/^]\s*[0-9]+/.test(trimmed)) {
    verification += 320;
  }

  // Axis 1: Deductive depth (multi-step reasoning, system design, trade-off analysis, debugging)
  let deductive = 160;
  if (
    /\b(why|prove|design|architect|compare|trade-?offs?|debug|diagnose|optimize|analyze|complexity|algorithm|deadlock|race\s+condition|derive|step[\s-]by[\s-]step|reason|plan)\b/.test(
      lower,
    )
  ) {
    deductive += 480;
  }
  if (trimmed.length > 260) deductive += 192;
  if (trimmed.includes('?') && trimmed.split(/[?\n]/).filter((s) => s.trim().length > 12).length >= 2) {
    deductive += 192;
  }
  if (directMatch) {
    deductive = Math.min(deductive, 256);
  }

  // Axis 2: Code synthesis & engineering salience
  let code = 64;
  if (
    /```|\b(typescript|javascript|rust|python|wasm|simd|sql|html|css|react|function|class|struct|async|await|regex|compiler|refactor|bug|stack\s*trace|api|git|cli|kernel|shader)\b/.test(
      lower,
    )
  ) {
    code += 576;
  }
  if (hasAttachment) code += 256;

  // Axis 3: Local grounding need (runtime telemetry, OPFS, notes, clock, attached files)
  let grounding = directMatch ? 928 : 64;
  if (
    /\b(runtime|opfs|verdant|vtensor|thread\s*pool|storage|workspace\s+note|save\s+.*note|my\s+notes|current\s+time|today'?s?\s+date|timezone)\b/.test(
      lower,
    )
  ) {
    grounding += 576;
  }
  if (hasAttachment) grounding += 384;

  // Axis 4: Epistemic uncertainty (novelty, open questions, compound constraints)
  let uncertainty = directMatch ? 64 : 256;
  if (/\b(best|should\s+i|how\s+do|what\s+if|subtle|edge\s+case|unknown|unclear|might|weird)\b/.test(lower)) {
    uncertainty += 384;
  }
  if (trimmed.length > 400) uncertainty += 192;

  // Axis 5: Conversational continuity (depth of thread history + referential language)
  let continuity = Math.min(768, Math.max(0, messages.length - 1) * 160);
  if (/\b(that|this|above|previous|earlier|continue|expand|also|instead|follow[\s-]up)\b/.test(lower)) {
    continuity += 256;
  }

  // Axis 6: Output elaboration expected
  let elaboration = 320;
  if (/\b(brief|short|concise|one[\s-]line|tl;?dr|quick|just\s+the)\b/.test(lower)) {
    elaboration = 128;
  } else if (/\b(detailed|comprehensive|thorough|deep\s+dive|full|complete|walkthrough|explain)\b/.test(lower)) {
    elaboration = 832;
  } else if (trimmed.length > 220) {
    elaboration = 640;
  }

  // Axis 7: Multi-step tool composition (requires combining local tools + neural synthesis)
  let toolComp = 64;
  if (!directMatch) {
    const wantsCalc = /\b(calculate|compute|evaluate|sqrt|sum|product)\b/.test(lower);
    const wantsNote = /\b(note|notes|save|remember|bookmark)\b/.test(lower);
    const wantsRuntime = /\b(inspect|runtime|opfs|vtensor|telemetry|workers?)\b/.test(lower);
    const wantsTime = /\b(current\s+time|today'?s?\s+date|timezone)\b/.test(lower);
    const wantsJson = /\b(validate|parse|count\s+words|count\s+lines|search\s+text)\b/.test(lower);
    const hits = [wantsCalc, wantsNote, wantsRuntime, wantsTime, wantsJson].filter(Boolean).length;
    if (hits >= 2) {
      toolComp = 928;
    } else if (hits === 1) {
      toolComp = 672;
    }
  }

  const vector: CognitiveVector = [
    clampQ10(verification),
    clampQ10(deductive),
    clampQ10(code),
    clampQ10(grounding),
    clampQ10(uncertainty),
    clampQ10(continuity),
    clampQ10(elaboration),
    clampQ10(toolComp),
  ];

  return {
    vector,
    directMatch,
    inputDigest: prismContentDigest(`${trimmed.length}:${trimmed.slice(0, 160)}`),
    charCount: trimmed.length,
    hasAttachment,
  };
}

// ============================================================================
// 4. Cross-Classified Model Coherence Matrix (`verdant-studio` REQ-D11 & REQ-D6)
// ============================================================================

export type VerdantRouteKind = 'lod0-verified' | 'neural-resident' | 'neural-on-demand';

export interface CrossClassifiedRouteSpec {
  readonly routeId: string;
  readonly label: string;
  readonly kind: VerdantRouteKind;
  readonly modelId: string | null;
  /** 8D capability facet profile in the exact same Q10 cognitive space (`REQ-D11`). */
  readonly facets: CognitiveVector;
  readonly supportsNativeThinking: boolean;
  readonly approximateBytes: number;
}

const LOD0_VERIFIED_FACETS: CognitiveVector = [1024, 192, 128, 1024, 64, 256, 256, 384];

function buildCatalogModelFacets(model: CatalogModel): CognitiveVector {
  const hay = `${model.id} ${model.name}`.toLowerCase();
  const isCoder = model.category === 'coding' || hay.includes('coder');
  const isThinking = !isCoder && (hay.includes('qwen3') || hay.includes('qwq') || hay.includes('r1'));
  const params = model.geometry.paramsBillions;

  const verification = isThinking ? 768 : isCoder ? 704 : 512;
  const deductive = isThinking
    ? params >= 7
      ? 960
      : 896
    : isCoder
      ? 832
      : params >= 7
        ? 768
        : 512;
  const code = isCoder ? 992 : isThinking ? 832 : params >= 3 ? 640 : 448;
  const grounding = 640;
  const uncertainty = isThinking ? 896 : params >= 7 ? 768 : 576;
  const continuity = model.geometry.nativeContextTokens >= 32_768 ? 896 : 704;
  const elaboration = params >= 4 ? 832 : 640;
  const toolComp = hay.includes('qwen') || hay.includes('llama-3') ? 864 : 640;

  return [
    verification,
    deductive,
    code,
    grounding,
    uncertainty,
    continuity,
    elaboration,
    toolComp,
  ];
}

export function pickDefaultNeuralCatalogModel(cachedModelIds: readonly string[] = []): CatalogModel {
  // Prefer an already-cached general/reasoning Qwen3 checkpoint first
  const preferenceOrder = [
    'qwen3-4b-q4_k_m',
    'qwen3-8b-q4_k_m',
    'qwen2.5-1.5b-instruct-q4_k_m',
    'qwen2.5-3b-instruct-q4_k_m',
    'qwen3-coder-30b-a3b-q4_k_m',
  ];
  for (const id of preferenceOrder) {
    if (cachedModelIds.includes(id)) {
      const m = findCatalogModelById(id);
      if (m) return m;
    }
  }
  for (const id of cachedModelIds) {
    const m = findCatalogModelById(id);
    if (m) return m;
  }
  const nav =
    typeof navigator !== 'undefined' ? (navigator as Navigator & { deviceMemory?: number }) : undefined;
  const memGb = typeof nav?.deviceMemory === 'number' ? nav.deviceMemory : 8;
  const targetId = memGb <= 4 ? 'qwen2.5-1.5b-instruct-q4_k_m' : 'qwen3-4b-q4_k_m';
  return findCatalogModelById(targetId) ?? MODEL_CATALOG[0];
}

// ============================================================================
// 5. Active Inference Expected Free Energy + Prism CostSurface Settlement
// ============================================================================

export interface RouteEnergyCandidate {
  readonly routeId: string;
  readonly label: string;
  readonly kind: VerdantRouteKind;
  readonly modelId: string | null;
  readonly admitted: boolean;
  readonly resonanceQ10: number;
  readonly freeEnergy: number;
  readonly costVector: PrismArtifactCostVector;
  readonly refusalReason?: string;
}

export interface CognitiveRouteCertificate {
  readonly routeId: string;
  readonly label: string;
  readonly kind: VerdantRouteKind;
  readonly modelId: string | null;
  readonly thinkingPolicy: VerdantThinkingPolicy;
  readonly attachTools: boolean;
  readonly recommendedSampling: {
    readonly temperature: number;
    readonly topP: number;
    readonly topK: number;
    readonly repeatPenalty: number;
  };
  readonly directMatch: DirectRouteMatch | null;
  readonly settlementRule: 'REQ-SETTLEMENT-LOD0' | 'REQ-SETTLEMENT-RESIDENT' | 'REQ-SETTLEMENT-PROMOTE';
  readonly freeEnergy: number;
  readonly surpriseQ10: number;
  readonly escalatedFromRetry: boolean;
  readonly digest: string;
  readonly rationale: string;
  readonly candidates: readonly RouteEnergyCandidate[];
}

export interface WorkingMemoryFact {
  readonly key: string;
  readonly summary: string;
  readonly turnIndex: number;
}

export interface CognitiveObserverSnapshot {
  readonly threadId: string;
  readonly turnCount: number;
  readonly belief: CognitiveVector;
  readonly lastSurpriseQ10: number;
  readonly retryEscalationLevel: number;
  readonly workingMemory: readonly WorkingMemoryFact[];
  readonly grants: Record<BrowserCapabilityOrgan, boolean>;
  readonly lastCertificate: CognitiveRouteCertificate | null;
}

interface InternalThreadMemory {
  turnCount: number;
  belief: CognitiveVector;
  lastSurpriseQ10: number;
  lastInputDigest: string;
  retryEscalationLevel: number;
  /** Monotonic per-thread tool latch: prevents toggling `<tools>` mid-thread and wiping the KV prefix! */
  toolsLatched: boolean;
  workingMemory: WorkingMemoryFact[];
  lastCertificate: CognitiveRouteCertificate | null;
}

const NEUTRAL_BELIEF: CognitiveVector = [192, 256, 192, 128, 256, 128, 384, 128];
const MAX_TRACKED_THREADS = 64;

export interface RouteDecisionContext {
  readonly threadId: string;
  readonly messages: readonly ThreadMessage[];
  readonly latestUserText: string;
  readonly hasAttachment: boolean;
  readonly activeModel: LoadedModelSummary | null;
  readonly cachedModelIds: readonly string[];
  readonly isToolContinuationStep: boolean;
  readonly hasImportedHarness: boolean;
  readonly publishedPrisms?: readonly PublishedPrismSummary[];
  /** True once the thread has consolidated turns in its lossless episodic ledger (`recall_thread_memory`). */
  readonly hasConsolidatedHistory?: boolean;
}

/**
 * Unified Per-Thread Cognitive Observer & Cross-Classified Prism Router.
 * Maintains strictly isolated cognitive state per `threadId` so threads never conflate
 * semantically or mechanically.
 */
export class VerdantCognitiveObserver {
  private readonly threads = new Map<string, InternalThreadMemory>();
  readonly grants = new ContainerGrantMatrix();
  /** Unified possession hierarchy (`Observer ⊇ User ⊇ {Threads, Prisms, ContextCompartments}`) & non-leaking boundary. */
  readonly possession = new ObserverPossessionStore('personal');

  /**
   * Registers all Published Prisms as active Observer cognitive weight organs, prunes any
   * unpublished/deleted Prisms cleanly, and syncs opaque schedule constraints declared by
   * Tool Organs (e.g. Healthy Habits in Personal space).
   */
  syncPublishedPrisms(projects: readonly VerdantPrismProject[]): void {
    const activeIds = new Set<string>();
    for (const proj of projects) {
      if (proj.published) activeIds.add(proj.id);
    }
    this.possession.pruneRemovedPrisms(activeIds);

    for (const proj of projects) {
      if (!proj.published) continue;
      const decl = proj.declaration;
      const proc = decl?.view?.proceduralOrgan;
      const mediaType = proc?.mediaType ?? decl?.concept.mediaType ?? 'tool-organ';
      const compartmentId: ContextCompartmentId =
        (decl?.concept.compartmentId as ContextCompartmentId | undefined) ?? 'personal';
      const organ: PublishedPrismOrgan = {
        prismId: proj.id,
        name: proj.published.name,
        slug: proj.id,
        digest: proj.published.digest,
        mediaType,
        compartmentId,
        cognitiveWeightsQ10:
          mediaType === 'interactive-game'
            ? [128, 384, 256, 512, 128, 256, 256, 640]
            : mediaType === 'research-program'
              ? [640, 768, 128, 512, 256, 512, 512, 512]
              : [256, 256, 128, 640, 128, 384, 256, 768],
        sensoryChannels: ['browser', 'storage', 'schedule'],
        impulseChannels: ['render-surface', 'notify', 'schedule-constraint', 'tool-invoke'],
        shareOpaqueConstraints:
          proc?.weights.mediaType === 'tool-organ' ? proc.weights.emitsOpaqueConstraints : false,
      };
      this.possession.registerPublishedPrismOrgan(organ);

      if (proc?.weights.mediaType === 'tool-organ' && proc.weights.scheduleWindows.length > 0) {
        const toolWeights = proc.weights;
        const nextConstraints = toolWeights.scheduleWindows.map((win, idx) => ({
          id: `${proj.id}:win:${idx}`,
          privateLabel: win.label,
          sourcePrismId: proj.id,
          startMinuteOfDay: win.startMinuteOfDay,
          endMinuteOfDay: win.endMinuteOfDay,
          daysOfWeek: win.daysOfWeek,
          weightQ10: 960,
          shareOpaqueAvailability: toolWeights.emitsOpaqueConstraints && win.protectedWindow,
        }));
        this.possession.replacePrismScheduleConstraints(compartmentId, proj.id, nextConstraints);
      }
    }
  }

  /**
   * Filters Published Prisms to ONLY those owned by the active Space (or explicitly `'public'`),
   * ensuring private Personal/Health Prism names and summaries never leak into Work tools or chat.
   */
  getVisiblePublishedPrismsForSpace(
    allPublished: readonly PublishedPrismSummary[],
    spaceId?: ContextCompartmentId,
  ): readonly PublishedPrismSummary[] {
    const active = spaceId ?? this.possession.getActiveCompartmentId();
    const organs = this.possession.listPublishedOrgans();
    if (organs.length === 0) return allPublished;
    const spaceById = new Map(organs.map((o) => [o.prismId, o.compartmentId] as const));
    return allPublished.filter((p) => {
      const ownerSpace = spaceById.get(p.id);
      return !ownerSpace || ownerSpace === active || ownerSpace === 'public';
    });
  }

  /**
   * Returns zero-detail opaque availability constraints projected from other user spaces into the
   * currently active space (zero private labels, zero Prism names, zero source space leakage).
   */
  getOpaqueConstraintsPromptForSpace(spaceId?: ContextCompartmentId): string {
    const target = spaceId ?? this.possession.getActiveCompartmentId();
    const projected = this.possession.getProjectedConstraintsFor(target);
    return formatOpaqueConstraintsForContext(projected);
  }

  private getOrCreateThread(threadId: string): InternalThreadMemory {
    const key = threadId || 'default-thread';
    let mem = this.threads.get(key);
    if (!mem) {
      if (this.threads.size >= MAX_TRACKED_THREADS) {
        const oldestKey = this.threads.keys().next().value;
        if (oldestKey !== undefined) this.threads.delete(oldestKey);
      }
      mem = {
        turnCount: 0,
        belief: NEUTRAL_BELIEF,
        lastSurpriseQ10: 0,
        lastInputDigest: '',
        retryEscalationLevel: 0,
        toolsLatched: false,
        workingMemory: [],
        lastCertificate: null,
      };
      this.threads.set(key, mem);
    }
    return mem;
  }

  /** Returns whether tool schemas are monotonically latched for `threadId` (KV-prefix stability). */
  isToolsLatched(threadId: string): boolean {
    const key = threadId || 'default-thread';
    return this.threads.get(key)?.toolsLatched ?? false;
  }

  getThreadSnapshot(threadId: string): CognitiveObserverSnapshot {
    const key = threadId || 'default-thread';
    const mem = this.getOrCreateThread(key);
    return {
      threadId: key,
      turnCount: mem.turnCount,
      belief: mem.belief,
      lastSurpriseQ10: mem.lastSurpriseQ10,
      retryEscalationLevel: mem.retryEscalationLevel,
      workingMemory: [...mem.workingMemory],
      grants: this.grants.snapshot(),
      lastCertificate: mem.lastCertificate,
    };
  }

  resetThread(threadId: string): void {
    this.threads.delete(threadId || 'default-thread');
  }

  /**
   * Executes one full Active-Inference Cognitive Observer cycle:
   * 1. Extracts 8D sensory observation from the input.
   * 2. Detects regeneration/retry prediction error (automatically escalating cognitive depth).
   * 3. Computes Predictive Coding Surprise and updates the thread's harmonic belief state.
   * 4. Evaluates all cross-classified routes (`LOD0 verified`, `resident neural`, `cached neural`,
   *    `on-demand promotion`) via Expected Free Energy + Prism `CanonicalObjective`.
   * 5. Returns a deterministic, verifiable `CognitiveRouteCertificate`.
   */
  selectRoute(ctx: RouteDecisionContext): {
    certificate: CognitiveRouteCertificate;
    snapshot: CognitiveObserverSnapshot;
  } {
    const activeSpace = this.possession.getActiveCompartmentId();
    this.possession.bindThreadToCompartment(ctx.threadId || 'default-thread', activeSpace);
    const mem = this.getOrCreateThread(ctx.threadId);
    const obs = observeInputSensoryState(ctx.messages, ctx.latestUserText, ctx.hasAttachment);

    // Couple active Space's Published Prism cognitive weights (`PublishedPrismOrgan.cognitiveWeightsQ10`)
    // into the unified model's prior belief (1/8 exact Q10 blend) strictly within the owning Space!
    const spaceOrgans = this.possession.listPublishedOrgans(activeSpace);
    let effectivePrior = mem.belief;
    if (spaceOrgans.length > 0) {
      const blended: number[] = new Array(COGNITIVE_DIM);
      for (let i = 0; i < COGNITIVE_DIM; i++) {
        let sum = 0;
        for (const o of spaceOrgans) sum += o.cognitiveWeightsQ10[i];
        const meanOrganQ10 = Math.round(sum / spaceOrgans.length);
        blended[i] = clampQ10((mem.belief[i] * 7 + meanOrganQ10) >> 3);
      }
      effectivePrior = blended as unknown as CognitiveVector;
    }

    // Metacognitive retry detection: same prompt digest re-submitted without a tool continuation
    // means the user clicked Regenerate — treat as prediction error and escalate deductive depth!
    const isRetry =
      !ctx.isToolContinuationStep &&
      mem.lastInputDigest !== '' &&
      mem.lastInputDigest === obs.inputDigest;
    if (isRetry) {
      mem.retryEscalationLevel = Math.min(2, mem.retryEscalationLevel + 1);
    } else if (!ctx.isToolContinuationStep) {
      mem.retryEscalationLevel = 0;
    }

    const escalatedObs: CognitiveVector = isRetry
      ? [
          clampQ10(obs.vector[0] + 192),
          clampQ10(obs.vector[1] + 384),
          obs.vector[2],
          obs.vector[3],
          clampQ10(obs.vector[4] + 384),
          obs.vector[5],
          clampQ10(obs.vector[6] + 192),
          obs.vector[7],
        ]
      : obs.vector;

    const surpriseQ10 = computePredictionSurpriseQ10(effectivePrior, escalatedObs);
    const nextBelief = harmonicIntegrateBelief(effectivePrior, escalatedObs, surpriseQ10);

    mem.belief = nextBelief;
    mem.lastSurpriseQ10 = surpriseQ10;
    if (!ctx.isToolContinuationStep) {
      mem.lastInputDigest = obs.inputDigest;
      mem.turnCount += 1;
    }

    // Derive cognitive execution posture directly from integrated belief (zero manual modes!)
    const deepReasoningSignal = Math.max(nextBelief[1], escalatedObs[1]);
    const epistemicPressure = Math.max(nextBelief[4], escalatedObs[4]);
    const toolCompSignal = Math.max(nextBelief[7], escalatedObs[7]);
    const needsTools = ctx.isToolContinuationStep || toolCompSignal >= 576;
    const needsThinking =
      !ctx.isToolContinuationStep &&
      (mem.retryEscalationLevel > 0 ||
        deepReasoningSignal >= 544 ||
        (deepReasoningSignal >= 448 && epistemicPressure >= 512));

    const thinkingPolicy: VerdantThinkingPolicy = needsTools
      ? 'auto'
      : needsThinking
        ? 'on'
        : 'off';

    const recommendedSampling = needsTools
      ? { temperature: 0.25, topP: 0.9, topK: 20, repeatPenalty: 1.05 }
      : needsThinking
        ? { temperature: 0.6, topP: 0.95, topK: 20, repeatPenalty: 1.05 }
        : { temperature: 0.65, topP: 0.85, topK: 20, repeatPenalty: 1.05 };

    // Build cross-classified route candidates (`REQ-D11`)
    const candidates: RouteEnergyCandidate[] = [];
    const hasLoadedRealModel =
      ctx.activeModel !== null && !isInternalBootstrapModel(ctx.activeModel.id);

    // Candidate 1: LOD0 Verified Deterministic Route (`REQ-SETTLEMENT`: verified first!)
    const explicitPrismMatch = detectExplicitPrismAddress(ctx.latestUserText, ctx.publishedPrisms);
    const effectiveDirectMatch = explicitPrismMatch ?? obs.directMatch;
    {
      const dm = effectiveDirectMatch;
      const organAllowed = dm ? this.grants.isAllowed(dm.organ) : false;
      const admitted =
        !ctx.isToolContinuationStep &&
        !isRetry &&
        !ctx.hasImportedHarness &&
        dm !== null &&
        organAllowed &&
        dm.confidenceQ10 >= 900;
      const resonance = dm ? dotCognitiveQ10(nextBelief, LOD0_VERIFIED_FACETS) : 0;
      const costVector: PrismArtifactCostVector = {
        dramReadCost: 0,
        computeOpCost: 10, // sub-millisecond deterministic execution
        quantEncodeCost: 0,
        syncBarrierCost: 0,
        hostTransferCost: 0,
      };
      const freeEnergy = admitted
        ? canonicalObjectiveScore(costVector) + Math.max(0, 1024 - (dm?.confidenceQ10 ?? 0))
        : 999_999;
      candidates.push({
        routeId: dm ? `lod0:${dm.toolName}` : 'lod0:deterministic',
        label: dm ? (dm.prismName ? `Prism · ${dm.prismName}` : `Verified · ${dm.toolName}`) : 'Verified Deterministic',
        kind: 'lod0-verified',
        modelId: null,
        admitted,
        resonanceQ10: resonance,
        freeEnergy,
        costVector,
        refusalReason: admitted
          ? undefined
          : dm && !organAllowed
            ? `Capability organ "${dm.organ}" denied by ContainerGrantMatrix (FK-07)`
            : 'Input requires generative neural synthesis',
      });
    }

    // Candidate 2: Currently Resident Neural Route (zero transfer/switch cost!)
    if (hasLoadedRealModel && ctx.activeModel) {
      const cat = findCatalogModelById(ctx.activeModel.id);
      const facets: CognitiveVector = cat
        ? buildCatalogModelFacets(cat)
        : [640, 768, 704, 640, 704, 768, 768, 768];
      const resonance = dotCognitiveQ10(nextBelief, facets);
      // Pragmatic risk = inverse resonance normalized cleanly
      const pragmaticRisk = Math.max(0, 1800 - resonance);
      const computeTicks = cat
        ? Math.round(cat.geometry.activeParamsBillions * 120)
        : 360;
      const costVector: PrismArtifactCostVector = {
        dramReadCost: 60,
        computeOpCost: computeTicks,
        quantEncodeCost: 20,
        syncBarrierCost: 10,
        hostTransferCost: 0, // already resident in vTensor arena!
      };
      candidates.push({
        routeId: `neural:resident:${ctx.activeModel.id}`,
        label: ctx.activeModel.displayName,
        kind: 'neural-resident',
        modelId: ctx.activeModel.id,
        admitted: true,
        resonanceQ10: resonance,
        freeEnergy: canonicalObjectiveScore(costVector) + pragmaticRisk,
        costVector,
      });
    }

    // Candidate 3: Already-cached OPFS neural checkpoints (cross-classified coherence, zero network!)
    for (const cachedId of ctx.cachedModelIds) {
      if (hasLoadedRealModel && ctx.activeModel?.id === cachedId) continue;
      const cat = findCatalogModelById(cachedId);
      if (!cat) continue;
      const facets = buildCatalogModelFacets(cat);
      const resonance = dotCognitiveQ10(nextBelief, facets);
      const pragmaticRisk = Math.max(0, 1800 - resonance);
      // Switching OPFS checkpoints incurs warm-load penalty (480 ticks) so we only switch when
      // a cached specialist genuinely dominates (e.g. heavy code -> cached coder, or nothing loaded yet)
      const switchPenalty = hasLoadedRealModel ? 480 : 40;
      const costVector: PrismArtifactCostVector = {
        dramReadCost: 120,
        computeOpCost: Math.round(cat.geometry.activeParamsBillions * 120),
        quantEncodeCost: 25,
        syncBarrierCost: 15,
        hostTransferCost: switchPenalty,
      };
      candidates.push({
        routeId: `neural:opfs:${cat.id}`,
        label: cat.shortName,
        kind: 'neural-on-demand',
        modelId: cat.id,
        admitted: true,
        resonanceQ10: resonance,
        freeEnergy: canonicalObjectiveScore(costVector) + pragmaticRisk,
        costVector,
      });
    }

    // Candidate 4: Cold-start Seamless Promotion Route (auto-streams default unified neural route)
    if (!hasLoadedRealModel && candidates.every((c) => !c.admitted || c.kind === 'lod0-verified')) {
      const defModel = pickDefaultNeuralCatalogModel(ctx.cachedModelIds);
      const facets = buildCatalogModelFacets(defModel);
      const resonance = dotCognitiveQ10(nextBelief, facets);
      const pragmaticRisk = Math.max(0, 1800 - resonance);
      const networkAllowed = this.grants.isAllowed('network-weights');
      const costVector: PrismArtifactCostVector = {
        dramReadCost: 240,
        computeOpCost: Math.round(defModel.geometry.activeParamsBillions * 120),
        quantEncodeCost: 30,
        syncBarrierCost: 20,
        hostTransferCost: 850,
      };
      candidates.push({
        routeId: `neural:promote:${defModel.id}`,
        label: defModel.shortName,
        kind: 'neural-on-demand',
        modelId: defModel.id,
        admitted: networkAllowed,
        resonanceQ10: resonance,
        freeEnergy: canonicalObjectiveScore(costVector) + pragmaticRisk,
        costVector,
        refusalReason: networkAllowed
          ? undefined
          : 'Network weight streaming denied by ContainerGrantMatrix (FK-07)',
      });
    }

    const admitted = candidates.filter((c) => c.admitted);
    admitted.sort((a, b) =>
      a.freeEnergy !== b.freeEnergy
        ? a.freeEnergy - b.freeEnergy
        : canonicalTieBreak(a.routeId, b.routeId),
    );
    const winner = admitted[0] ?? candidates[candidates.length - 1];

    const settlementRule: CognitiveRouteCertificate['settlementRule'] =
      winner.kind === 'lod0-verified'
        ? 'REQ-SETTLEMENT-LOD0'
        : winner.kind === 'neural-resident'
          ? 'REQ-SETTLEMENT-RESIDENT'
          : 'REQ-SETTLEMENT-PROMOTE';

    const rationale =
      winner.kind === 'lod0-verified'
        ? effectiveDirectMatch?.reason ?? 'Settled deterministically at LOD0.'
        : isRetry
          ? `Regeneration prediction error detected — escalated deductive depth on ${winner.label}.`
          : needsTools
            ? `Grounded tool composition activated on ${winner.label} (surprise ${Math.round((surpriseQ10 / 1024) * 100)}%).`
            : needsThinking
              ? `Deliberative reasoning activated on ${winner.label} (depth ${Math.round((deepReasoningSignal / 1024) * 100)}%).`
              : `Direct unified synthesis on ${winner.label}.`;

    const digest = prismContentDigest(
      `${winner.routeId}:${winner.freeEnergy}:${thinkingPolicy}:${needsTools ? 1 : 0}:${surpriseQ10}:${obs.inputDigest}`,
    );

    const shouldAttachToolsNow =
      winner.kind !== 'lod0-verified' &&
      (mem.toolsLatched ||
        needsTools ||
        Boolean(ctx.hasConsolidatedHistory) ||
        Boolean(ctx.publishedPrisms && ctx.publishedPrisms.length > 0));
    if (shouldAttachToolsNow) {
      mem.toolsLatched = true;
    }

    const certificate: CognitiveRouteCertificate = {
      routeId: winner.routeId,
      label: winner.label,
      kind: winner.kind,
      modelId: winner.modelId,
      thinkingPolicy: winner.kind === 'lod0-verified' ? 'off' : thinkingPolicy,
      attachTools: shouldAttachToolsNow,
      recommendedSampling,
      directMatch: winner.kind === 'lod0-verified' ? effectiveDirectMatch : null,
      settlementRule,
      freeEnergy: winner.freeEnergy,
      surpriseQ10,
      escalatedFromRetry: isRetry,
      digest,
      rationale,
      candidates,
    };

    mem.lastCertificate = certificate;
    return {
      certificate,
      snapshot: this.getThreadSnapshot(ctx.threadId),
    };
  }

  /**
   * Closes the second-order cybernetic feedback loop after a turn settles, recording verified
   * facts into the thread's Global Workspace working memory.
   */
  recordTurnOutcome(
    threadId: string,
    outcome: {
      readonly factKey?: string;
      readonly factSummary?: string;
      readonly hadToolError?: boolean;
    },
  ): CognitiveObserverSnapshot {
    const mem = this.getOrCreateThread(threadId);
    if (outcome.factKey && outcome.factSummary) {
      const existingIdx = mem.workingMemory.findIndex((f) => f.key === outcome.factKey);
      const entry: WorkingMemoryFact = {
        key: outcome.factKey,
        summary: outcome.factSummary.slice(0, 180),
        turnIndex: mem.turnCount,
      };
      if (existingIdx >= 0) {
        mem.workingMemory[existingIdx] = entry;
      } else {
        mem.workingMemory.push(entry);
        if (mem.workingMemory.length > 8) {
          mem.workingMemory.shift();
        }
      }
    }
    if (outcome.hadToolError) {
      // Elevate epistemic uncertainty slightly when a tool errored so next turn verifies carefully
      mem.belief = [
        mem.belief[0],
        mem.belief[1],
        mem.belief[2],
        mem.belief[3],
        clampQ10(mem.belief[4] + 192),
        mem.belief[5],
        mem.belief[6],
        mem.belief[7],
      ];
    }
    return this.getThreadSnapshot(threadId);
  }

  /**
   * Feeds a live OpenTelemetry sensory event (e.g. from a Published Prism) into the Observer's
   * Global Workspace working memory.
   */
  recordSensoryEvent(event: OtelEvent, threadId = 'default-thread'): void {
    const summary =
      event.body !== undefined
        ? typeof event.body === 'string'
          ? event.body
          : JSON.stringify(event.body)
        : event.name;
    this.recordTurnOutcome(threadId, {
      factKey: `${event.endpointKind}:${event.endpointId}:${event.name}`,
      factSummary: summary,
      hadToolError: event.status === 'error',
    });
  }
}

// ============================================================================
// 6. Unified VerdantUI System Prompt, Suggestions & 5 Client-Side Tools
// ============================================================================

export const UNIFIED_VERDANT_SYSTEM_PROMPT = [
  'You are VerdantUI, a unified private AI assistant running 100% locally in the user\'s browser.',
  'Be accurate, clear, and concise. When working through complex problems inside <think>...</think>, verify edge cases before giving a crisp final answer.',
  'When <tools> are provided and a local tool helps answer accurately or take an action, call it with exact JSON inside <tool_call>...</tool_call> and stop immediately after </tool_call> to await <tool_response>. Never fabricate tool results.',
].join('\n');

export const UNIFIED_VERDANT_SUGGESTIONS: readonly HarnessSuggestionSpec[] = [
  {
    label: 'Quick exact math',
    prompt: 'Calculate (144 * 37) + sqrt(625)',
  },
  {
    label: 'Plan my afternoon',
    prompt: 'Help me schedule a focused 45-minute project review without conflicting with my busy times.',
  },
  {
    label: 'Brainstorm a project',
    prompt: 'Give me 4 creative ideas for a relaxing puzzle game I can build in Studio.',
  },
  {
    label: 'Summarize clearly',
    prompt: 'Explain the key benefits of keeping personal AI completely on-device in 4 short bullet points.',
  },
];

// Client-Side Persistent Workspace Notes Store (localStorage + memory fallback)
const WORKSPACE_NOTES_STORAGE_KEY = 'verdant-ui:workspace-notes:v1';

export interface WorkspaceNote {
  readonly title: string;
  readonly content: string;
  readonly updatedAt: string;
}

const memoryNotes = new Map<string, WorkspaceNote>();

function readWorkspaceNotes(): Record<string, WorkspaceNote> {
  if (typeof window !== 'undefined' && typeof window.localStorage !== 'undefined') {
    try {
      const raw = window.localStorage.getItem(WORKSPACE_NOTES_STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as Record<string, WorkspaceNote>;
        if (parsed && typeof parsed === 'object') return parsed;
      }
    } catch {
      // Fall back to memory map
    }
  }
  return Object.fromEntries(memoryNotes.entries());
}

function writeWorkspaceNotes(notes: Record<string, WorkspaceNote>): void {
  memoryNotes.clear();
  for (const [k, v] of Object.entries(notes)) {
    memoryNotes.set(k, v);
  }
  if (typeof window !== 'undefined' && typeof window.localStorage !== 'undefined') {
    try {
      window.localStorage.setItem(WORKSPACE_NOTES_STORAGE_KEY, JSON.stringify(notes));
    } catch {
      // Ignore storage quota errors
    }
  }
}

export interface RuntimeToolSnapshot {
  readonly model: LoadedModelSummary | null;
  readonly telemetry: VTensorTelemetrySnapshot;
  readonly opfsStorage: OpfsQuotaSnapshot;
  readonly lastTokensPerSec: number;
}

export interface PublishedPrismToolBridge {
  readonly prisms: readonly PublishedPrismSummary[];
  dispatchPrism(
    prismId: string,
    payload: unknown,
    signal?: AbortSignal,
  ): Promise<{ readonly result: unknown; readonly prismName: string; readonly digest: string }>;
}

export interface EpisodicRecallToolBridge {
  recallTurns(
    query: {
      readonly query?: string;
      readonly turnIndex?: number;
      readonly kappa?: string;
      readonly limit?: number;
    },
  ): readonly {
    readonly turnIndex: number;
    readonly role: 'user' | 'assistant' | 'system';
    readonly kappa: string;
    readonly salienceQ10: number;
    readonly verbatimText: string;
  }[];
}

/**
 * Creates VerdantUI's 100% client-side `@assistant-ui/react` verified Tool registry,
 * dynamically registering every Published Prism via its JSON Schema 2020-12 input contract
 * and optional bit-exact lossless episodic recall (`recall_thread_memory`).
 */
export function createVerdantClientTools(
  getSnapshot: () => RuntimeToolSnapshot,
  publishedPrismBridge?: PublishedPrismToolBridge,
  episodicRecallBridge?: EpisodicRecallToolBridge,
  getActiveSpace?: () => ContextCompartmentId,
): Record<string, Tool<any, any>> {
  const tools: Record<string, Tool<any, any>> = {
    calculate: {
      description:
        'Evaluate a mathematical expression accurately (supports +, -, *, /, ^, %, parentheses, sqrt, abs, sin, cos, tan, log, ln, exp, floor, ceil, round, min, max, pow, pi, e).',
      parameters: {
        type: 'object',
        properties: {
          expression: {
            type: 'string',
            description: 'Mathematical expression to evaluate, e.g. "(144 * 37) + sqrt(625)"',
          },
        },
        required: ['expression'],
      },
      execute: async (args: { expression?: string }) => {
        const expr = String(args?.expression ?? '').trim();
        const result = evaluateMathExpression(expr);
        return {
          expression: expr,
          result,
          formatted: Number.isInteger(result) ? result.toLocaleString('en-US') : String(result),
        };
      },
    },
    current_datetime: {
      description:
        'Get the user\'s current local date, time, day of week, timezone, and Unix timestamp.',
      parameters: {
        type: 'object',
        properties: {},
      },
      execute: async () => {
        const now = new Date();
        const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || 'Local';
        return {
          iso: now.toISOString(),
          local: now.toLocaleString('en-US', { dateStyle: 'full', timeStyle: 'medium' }),
          timezone: tz,
          unixMs: now.getTime(),
        };
      },
    },
    inspect_runtime: {
      description:
        'Inspect the live local Verdant / vTensor engine state: active model, quantization, context window, multi-threaded WASM SIMD worker count, tokens/sec, Roofline contract status, and OPFS cached models.',
      parameters: {
        type: 'object',
        properties: {},
      },
      execute: async () => {
        const snap = getSnapshot();
        const tp = snap.telemetry.threadPool;
        const pc = snap.telemetry.performanceContract;
        const activeRealModel =
          snap.model && !isInternalBootstrapModel(snap.model.id) ? snap.model : null;
        return {
          provider: snap.telemetry.activeProvider,
          verdantVersion: snap.telemetry.verdantVersion,
          kernelAcceleration: snap.telemetry.kernelAcceleration,
          model: activeRealModel
            ? {
                id: activeRealModel.id,
                name: activeRealModel.displayName,
                quantization: activeRealModel.hparams.primaryQuantization,
                contextWindow: activeRealModel.maxContextWindow,
                backingKind: activeRealModel.backingKind,
              }
            : { status: 'LOD0 Instant (neural weights stream on demand)' },
          threadPool: tp
            ? {
                active: tp.active,
                workers: tp.workerCount,
                barrierLatencyUs: tp.barrierLatencyUs,
              }
            : { active: false, workers: 1 },
          lastTokensPerSec: snap.lastTokensPerSec,
          contractVerdict: pc?.verdict ?? null,
          opfs: {
            supported: snap.opfsStorage.supported,
            cachedModels: snap.opfsStorage.storedModels.map((m) => ({
              modelId: m.modelId,
              bytes: m.byteLength,
              mode: m.storageMode ?? 'full',
            })),
          },
        };
      },
    },
    workspace_notes: {
      description:
        'Read, write, list, or delete persistent local browser workspace notes/snippets stored privately on this device.',
      parameters: {
        type: 'object',
        properties: {
          action: {
            type: 'string',
            enum: ['list', 'read', 'write', 'delete'],
            description: 'Action to perform on local workspace notes.',
          },
          title: {
            type: 'string',
            description: 'Note title (required for read, write, delete).',
          },
          content: {
            type: 'string',
            description: 'Markdown or plain text note body (required for write).',
          },
        },
        required: ['action'],
      },
      execute: async (args: { action?: string; title?: string; content?: string }) => {
        const action = String(args?.action ?? 'list').toLowerCase();
        const space = getActiveSpace ? getActiveSpace() : 'personal';
        const prefix = `space:${space}:`;
        const notes = readWorkspaceNotes();
        if (action === 'list') {
          const list = Object.entries(notes)
            .filter(([k]) => (getActiveSpace ? k.startsWith(prefix) : k.startsWith(prefix) || !k.startsWith('space:')))
            .map(([, n]) => ({
              title: n.title,
              updatedAt: n.updatedAt,
              chars: n.content.length,
            }));
          return { action: 'list', count: list.length, notes: list };
        }
        const title = String(args?.title ?? '').trim();
        if (!title) {
          throw new Error('Note title is required for read/write/delete.');
        }
        const key = `${prefix}${title.toLowerCase()}`;
        const legacyKey = title.toLowerCase();
        if (action === 'read') {
          const found = notes[key] ?? (!getActiveSpace ? notes[legacyKey] : undefined);
          if (!found) return { action: 'read', found: false, title };
          return { action: 'read', found: true, note: found };
        }
        if (action === 'write') {
          const content = String(args?.content ?? '');
          const note: WorkspaceNote = {
            title,
            content,
            updatedAt: new Date().toISOString(),
          };
          notes[key] = note;
          writeWorkspaceNotes(notes);
          return { action: 'write', saved: true, note };
        }
        if (action === 'delete') {
          const existed = Object.hasOwn(notes, key) || (!getActiveSpace && Object.hasOwn(notes, legacyKey));
          delete notes[key];
          if (!getActiveSpace) delete notes[legacyKey];
          writeWorkspaceNotes(notes);
          return { action: 'delete', deleted: existed, title };
        }
        throw new Error(`Unsupported workspace_notes action "${action}".`);
      },
    },
    analyze_text: {
      description:
        'Deterministically inspect a block of text or JSON: counts lines, words, characters, validates/formats JSON, or searches with a case-insensitive pattern.',
      parameters: {
        type: 'object',
        properties: {
          text: { type: 'string', description: 'Text or JSON content to inspect.' },
          mode: {
            type: 'string',
            enum: ['stats', 'json', 'search'],
            description: 'Inspection mode (default: stats).',
          },
          query: {
            type: 'string',
            description: 'Substring pattern when mode is "search".',
          },
        },
        required: ['text'],
      },
      execute: async (args: { text?: string; mode?: string; query?: string }) => {
        const text = String(args?.text ?? '');
        const mode = String(args?.mode ?? 'stats').toLowerCase();
        const lines = text.length === 0 ? [] : text.split(/\r?\n/);
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;
        if (mode === 'json') {
          try {
            const parsed = JSON.parse(text);
            const keys =
              parsed && typeof parsed === 'object' && !Array.isArray(parsed)
                ? Object.keys(parsed)
                : null;
            return {
              validJson: true,
              type: Array.isArray(parsed) ? 'array' : typeof parsed,
              length: Array.isArray(parsed) ? parsed.length : (keys?.length ?? null),
              keys,
              formatted: JSON.stringify(parsed, null, 2).slice(0, 2000),
            };
          } catch (err) {
            return {
              validJson: false,
              error: err instanceof Error ? err.message : String(err),
            };
          }
        }
        if (mode === 'search') {
          const query = String(args?.query ?? '').trim();
          if (!query) return { query: '', matchCount: 0, matches: [] };
          const matches: Array<{ line: number; text: string }> = [];
          const lowerQ = query.toLowerCase();
          for (let idx = 0; idx < lines.length && matches.length < 20; idx++) {
            if (lines[idx].toLowerCase().includes(lowerQ)) {
              matches.push({ line: idx + 1, text: lines[idx].slice(0, 200) });
            }
          }
          return { query, matchCount: matches.length, matches };
        }
        return {
          chars: text.length,
          words,
          lines: lines.length,
        };
      },
    },
  };

  if (publishedPrismBridge) {
    for (const prism of publishedPrismBridge.prisms) {
      const name = prismToolName(prism);
      tools[name] = {
        description:
          prism.summary ||
          `Dispatch an input stream to the Published Prism "${prism.name}" (${shortPrismDigest(prism.digest)}).`,
        parameters: prism.streams.inputs,
        execute: async (args: Record<string, unknown>, ctx?: { abortSignal?: AbortSignal }) => {
          const { result, prismName, digest } = await publishedPrismBridge.dispatchPrism(
            prism.id,
            args ?? {},
            ctx?.abortSignal,
          );
          return {
            __prismInvocation: true,
            prismId: prism.id,
            prismName,
            digest,
            output: result,
          };
        },
      };
    }
  }

  if (episodicRecallBridge) {
    tools.recall_thread_memory = {
      description:
        'Retrieve bit-exact verbatim earlier turns from this conversation’s lossless content-addressed episodic ledger by kappa hash, turn index, or keyword query.',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'Keyword, symbol, or topic to recall verbatim.' },
          kappa: { type: 'string', description: 'Optional 16-hex kappa address from the episodic index.' },
          turnIndex: { type: 'number', description: 'Optional 0-based turn number.' },
          limit: { type: 'number', description: 'Maximum matching turns to return (1..5, default 2).' },
        },
      },
      execute: async (args: { query?: string; kappa?: string; turnIndex?: number; limit?: number }) => {
        const hits = episodicRecallBridge.recallTurns({
          query: typeof args?.query === 'string' ? args.query : undefined,
          kappa: typeof args?.kappa === 'string' ? args.kappa : undefined,
          turnIndex: typeof args?.turnIndex === 'number' ? args.turnIndex : undefined,
          limit: typeof args?.limit === 'number' ? args.limit : 2,
        });
        return {
          lossless: true,
          hitCount: hits.length,
          turns: hits,
        };
      },
    };
  }

  return tools;
}

/** Formats a concise, grounded Markdown response for a settled LOD0 deterministic operator. */
export function formatDirectOperatorSummary(toolName: DirectToolName, result: unknown): string {
  const r = result as Record<string, any>;
  if (toolName.startsWith('prism_') && r && r.__prismInvocation) {
    const out = r.output;
    const body =
      typeof out === 'string'
        ? out
        : '```json\n' + JSON.stringify(out, null, 2).slice(0, 1600) + '\n```';
    return `**${r.prismName}** (\`#${shortPrismDigest(String(r.digest ?? ''))}\`)\n\n${body}`;
  }
  switch (toolName) {
    case 'calculate':
      return `\`${r.expression}\` = **${r.formatted}**`;
    case 'current_datetime':
      return `Local time: **${r.local}** (${r.timezone})`;
    case 'inspect_runtime': {
      const m = r.model;
      const modelLine = m?.name
        ? `**${m.name}** (${m.quantization} · ${Number(m.contextWindow).toLocaleString()} tok · ${m.backingKind})`
        : m?.status ?? 'LOD0 Instant';
      const tp = r.threadPool;
      const cachedCount = Array.isArray(r.opfs?.cachedModels) ? r.opfs.cachedModels.length : 0;
      return [
        `- **Substrate:** \`${r.provider}\` v${r.verdantVersion} (${r.kernelAcceleration})`,
        `- **Neural Route:** ${modelLine}`,
        `- **Thread Pool:** ${tp?.active ? `${tp.workers} WASM SIMD workers` : '1 worker'}`,
        `- **OPFS Storage:** ${r.opfs?.supported ? `${cachedCount} cached model${cachedCount === 1 ? '' : 's'}` : 'Unavailable'}`,
      ].join('\n');
    }
    case 'workspace_notes': {
      if (r.action === 'list') {
        const notes = Array.isArray(r.notes) ? r.notes : [];
        if (notes.length === 0) return 'No local workspace notes saved yet.';
        return notes.map((n: any) => `- **${n.title}** (${n.chars} chars)`).join('\n');
      }
      return '```json\n' + JSON.stringify(r, null, 2) + '\n```';
    }
    case 'analyze_text': {
      if (r.validJson === true) {
        return `Valid JSON (${r.type}${r.length != null ? `, ${r.length} entries` : ''}):\n\n\`\`\`json\n${r.formatted}\n\`\`\``;
      }
      if (r.validJson === false) {
        return `Invalid JSON: ${r.error}`;
      }
      return `Lines: **${r.lines}** · Words: **${r.words}** · Characters: **${r.chars}**`;
    }
    default:
      return '```json\n' + JSON.stringify(r, null, 2).slice(0, 1600) + '\n```';
  }
}

/**
 * Builds the official `@assistant-ui/react` `ModelContext` for the single unified VerdantUI model,
 * exposing verified client tools to `ToolInvocationTracker` and composing any imported custom harness.
 */
export function buildUnifiedVerdantModelContext(
  clientTools: Record<string, Tool<any, any>>,
  importedHarness: ImportedHarnessSpec | null,
  samplingOverride?: { readonly temperature?: number; readonly topP?: number },
  opaqueConstraintsNotice?: string,
): ModelContext {
  const systemText = opaqueConstraintsNotice?.trim()
    ? `${UNIFIED_VERDANT_SYSTEM_PROMPT}\n\n${opaqueConstraintsNotice.trim()}`
    : UNIFIED_VERDANT_SYSTEM_PROMPT;
  const baseContext: ModelContext = {
    system: systemText,
    callSettings: {
      ...(samplingOverride?.temperature !== undefined
        ? { temperature: samplingOverride.temperature }
        : {}),
      ...(samplingOverride?.topP !== undefined ? { topP: samplingOverride.topP } : {}),
    },
    tools: clientTools,
  };

  if (!importedHarness) {
    return baseContext;
  }

  const imported = buildModelContextFromHarness(importedHarness);
  return {
    ...baseContext,
    ...(imported.system !== undefined ? { system: imported.system } : {}),
    callSettings: {
      ...baseContext.callSettings,
      ...(imported.callSettings ?? {}),
    },
    config: {
      ...(imported.config ?? {}),
    },
  };
}
