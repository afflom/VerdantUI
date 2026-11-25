/**
 * Cohesive Verdant + Assistant-UI Harness (`Fast` · `Thinking` · `Agentic`).
 *
 * Orchestrates curated local GGUF runtime tiers and maps each user-facing response mode directly
 * onto `@assistant-ui/react`'s standard `ModelContext` contract (`system`, `callSettings`,
 * `config`, `tools`):
 *   - `fast`: Low-latency direct answers with explicit Qwen3 non-thinking prefix (`<think>\n\n</think>\n\n`).
 *   - `thinking`: Structured chain-of-thought streamed into Assistant-UI `ReasoningMessagePart`
 *     with Qwen3-calibrated sampling (`temp=0.6, topP=0.95`) and explicit `<think>\n` guidance.
 *   - `agentic`: Multi-step tool-calling loop powered by Assistant-UI's `LocalThreadRuntimeCore`
 *     (`maxSteps: 5`) + 5 100%-client-side deterministic browser tools (`calculate`, `current_datetime`,
 *     `inspect_runtime`, `workspace_notes`, `analyze_text`).
 */

import type { ModelContext } from '@assistant-ui/react';
import type { Tool } from 'assistant-stream';
import {
  buildModelContextFromHarness,
  type HarnessSuggestionSpec,
  type ImportedHarnessSpec,
} from './harnessConfig';
import type { LoadedModelSummary } from './llamaRuntime';
import { findCatalogModelById, type CatalogModel } from './modelCatalog';
import type { OpfsQuotaSnapshot } from './vramBackingStore';
import type { VTensorTelemetrySnapshot } from './vtensorProvider';

export type VerdantHarnessMode = 'fast' | 'thinking' | 'agentic';
export type VerdantThinkingPolicy = 'auto' | 'on' | 'off';
export type VerdantRuntimeTierId = 'compact' | 'balanced' | 'pro' | 'expert-moe';

export interface VerdantHarnessModeSpec {
  readonly id: VerdantHarnessMode;
  readonly label: string;
  readonly shortLabel: string;
  readonly description: string;
  readonly thinkingPolicy: VerdantThinkingPolicy;
  readonly system: string;
  readonly callSettings: {
    readonly temperature: number;
    readonly topP: number;
  };
  readonly topK: number;
  readonly repeatPenalty: number;
  readonly reasoningEffort: 'low' | 'high' | 'medium';
  readonly enableTools: boolean;
  readonly suggestions: readonly HarnessSuggestionSpec[];
}

export interface VerdantRuntimeTier {
  readonly id: VerdantRuntimeTierId;
  readonly label: string;
  readonly badge: string;
  readonly tagline: string;
  readonly modelId: string;
  readonly supportsNativeThinking: boolean;
  readonly supportsNativeTools: boolean;
  readonly recommended?: boolean;
}

const HARNESS_MODE_STORAGE_KEY = 'verdant-ui:harness-mode:v1';
const WORKSPACE_NOTES_STORAGE_KEY = 'verdant-ui:workspace-notes:v1';

/**
 * Curated Verdant Runtime Tiers replacing raw arbitrary model browsing in the primary UX.
 * Each tier maps to a verified checkpoint in `MODEL_CATALOG`.
 */
export const VERDANT_RUNTIME_TIERS: readonly VerdantRuntimeTier[] = [
  {
    id: 'balanced',
    label: 'Verdant Core',
    badge: 'Recommended',
    tagline: 'Unified Fast, Thinking, and Agentic tool-use in a single 2.5 GB local model.',
    modelId: 'qwen3-4b-q4_k_m',
    supportsNativeThinking: true,
    supportsNativeTools: true,
    recommended: true,
  },
  {
    id: 'compact',
    label: 'Verdant Lite',
    badge: 'Fastest download',
    tagline: 'Ultra-light 1.0 GB local model for instant startup and lower-memory laptops.',
    modelId: 'qwen2.5-1.5b-instruct-q4_k_m',
    supportsNativeThinking: false,
    supportsNativeTools: true,
  },
  {
    id: 'pro',
    label: 'Verdant Pro',
    badge: 'Stronger reasoning',
    tagline: 'Higher-capacity 5.0 GB Qwen3 model for deeper reasoning and complex coding.',
    modelId: 'qwen3-8b-q4_k_m',
    supportsNativeThinking: true,
    supportsNativeTools: true,
  },
  {
    id: 'expert-moe',
    label: 'Verdant Coder MoE',
    badge: '30B On-Demand',
    tagline: '30.5B Mixture-of-Experts coding model (3.3B active) streamed via OPFS slabs.',
    modelId: 'qwen3-coder-30b-a3b-q4_k_m',
    supportsNativeThinking: false,
    supportsNativeTools: true,
  },
];

export function resolveTierCatalogModel(tierId: VerdantRuntimeTierId): CatalogModel {
  const tier = VERDANT_RUNTIME_TIERS.find((t) => t.id === tierId) ?? VERDANT_RUNTIME_TIERS[0];
  const found = findCatalogModelById(tier.modelId);
  if (!found) {
    throw new Error(`Verdant runtime tier "${tier.id}" references missing catalog model "${tier.modelId}".`);
  }
  return found;
}

export function findTierForModelId(modelId: string | null | undefined): VerdantRuntimeTier | null {
  if (!modelId) return null;
  return VERDANT_RUNTIME_TIERS.find((t) => t.modelId === modelId) ?? null;
}

/**
 * Recommends the best Verdant Runtime Tier based on cached OPFS models and browser hardware.
 */
export function recommendVerdantTier(cachedModelIds: readonly string[] = []): VerdantRuntimeTier {
  // Prefer an already-cached managed tier (highest capability first among balanced/pro/compact)
  for (const preferredId of ['balanced', 'pro', 'compact', 'expert-moe'] as const) {
    const tier = VERDANT_RUNTIME_TIERS.find((t) => t.id === preferredId)!;
    if (cachedModelIds.includes(tier.modelId)) {
      return tier;
    }
  }
  const nav = typeof navigator !== 'undefined' ? (navigator as Navigator & { deviceMemory?: number }) : undefined;
  const memGb = typeof nav?.deviceMemory === 'number' ? nav.deviceMemory : 8;
  if (memGb <= 4) {
    return VERDANT_RUNTIME_TIERS.find((t) => t.id === 'compact')!;
  }
  return VERDANT_RUNTIME_TIERS.find((t) => t.id === 'balanced')!;
}

export const VERDANT_HARNESS_MODES: Readonly<Record<VerdantHarnessMode, VerdantHarnessModeSpec>> = {
  fast: {
    id: 'fast',
    label: 'Fast',
    shortLabel: 'Fast',
    description: 'Direct, low-latency answers without internal thinking monologue.',
    thinkingPolicy: 'off',
    system:
      'You are VerdantUI, a helpful local AI assistant running privately in the user\'s browser. Answer directly, accurately, and concisely without unnecessary preamble.',
    callSettings: {
      temperature: 0.7,
      topP: 0.8,
    },
    topK: 20,
    repeatPenalty: 1.05,
    reasoningEffort: 'low',
    enableTools: false,
    suggestions: [
      {
        label: 'Explain a concept clearly',
        prompt: 'Explain how WebAssembly SIMD speeds up local LLM inference in 4 concise bullet points.',
      },
      {
        label: 'Draft crisp technical copy',
        prompt: 'Write a short, clear README summary for a client-side local AI workbench.',
      },
    ],
  },
  thinking: {
    id: 'thinking',
    label: 'Thinking',
    shortLabel: 'Thinking',
    description: 'Step-by-step reasoning in a collapsible thought trace before the final answer.',
    thinkingPolicy: 'on',
    system:
      'You are VerdantUI in Thinking mode. Work through the problem carefully step by step inside <think>...</think>, verify edge cases, and then provide a clear, concise final answer after </think>.',
    callSettings: {
      temperature: 0.6,
      topP: 0.95,
    },
    topK: 20,
    repeatPenalty: 1.05,
    reasoningEffort: 'high',
    enableTools: false,
    suggestions: [
      {
        label: 'Reason through an algorithm',
        prompt: 'Design an O(1) LRU cache with second-chance frequency protection and prove why eviction stays bounded.',
      },
      {
        label: 'Debug a subtle concurrency bug',
        prompt: 'Compare lock-free ring buffers vs Atomics.wait barriers for a SharedArrayBuffer thread pool.',
      },
    ],
  },
  agentic: {
    id: 'agentic',
    label: 'Agentic',
    shortLabel: 'Agentic',
    description: 'Uses client-side browser tools (math, runtime inspector, local notes, text analysis).',
    thinkingPolicy: 'auto',
    system: [
      'You are VerdantUI in Agentic mode, running 100% locally in the user\'s browser with access to client-side tools.',
      'When a tool helps answer accurately or take an action, call it using exact JSON inside <tool_call> tags:',
      '<tool_call>',
      '{"name": "tool_name", "arguments": {"param": "value"}}',
      '</tool_call>',
      'Stop immediately after closing </tool_call> and wait for the <tool_response>. Never fabricate tool outputs. After receiving the tool result, give a concise, grounded answer.',
    ].join('\n'),
    callSettings: {
      temperature: 0.25,
      topP: 0.9,
    },
    topK: 20,
    repeatPenalty: 1.05,
    reasoningEffort: 'medium',
    enableTools: true,
    suggestions: [
      {
        label: 'Inspect live Verdant engine',
        prompt: 'Inspect my local Verdant runtime and summarize the active model, thread pool, and OPFS storage.',
      },
      {
        label: 'Exact math & local note',
        prompt: 'Calculate (144 * 37) + sqrt(625) and save the result to a local workspace note titled "Quick Calculation".',
      },
    ],
  },
};

export function loadSavedHarnessMode(): VerdantHarnessMode {
  if (typeof window === 'undefined' || typeof window.localStorage === 'undefined') {
    return 'fast';
  }
  try {
    const raw = window.localStorage.getItem(HARNESS_MODE_STORAGE_KEY);
    if (raw === 'fast' || raw === 'thinking' || raw === 'agentic') {
      return raw;
    }
  } catch {
    // Ignore storage error
  }
  return 'fast';
}

export function saveHarnessMode(mode: VerdantHarnessMode): void {
  if (typeof window === 'undefined' || typeof window.localStorage === 'undefined') {
    return;
  }
  try {
    window.localStorage.setItem(HARNESS_MODE_STORAGE_KEY, mode);
  } catch {
    // Ignore storage error
  }
}

// ============================================================================
// Safe Deterministic Recursive-Descent Math Evaluator (Zero eval / Function)
// ============================================================================

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
    if ((ch >= '0' && ch <= '9') || (ch === '.' && i + 1 < s.length && s[i + 1] >= '0' && s[i + 1] <= '9')) {
      let start = i;
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
      let start = i;
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
    while (peek()?.kind === 'op' && (peek()!.value === '*' || peek()!.value === '/' || peek()!.value === '%')) {
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
        consume(); // '('
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

// ============================================================================
// Client-Side Persistent Workspace Notes Store (localStorage + memory fallback)
// ============================================================================

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
  readonly harnessMode: VerdantHarnessMode;
}

/**
 * Creates VerdantUI's 100% client-side `@assistant-ui/react` Tool registry for Agentic mode.
 * Every tool executes strictly inside the browser with zero external server dependencies.
 */
export function createVerdantClientTools(
  getSnapshot: () => RuntimeToolSnapshot,
): Record<string, Tool<any, any>> {
  return {
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
      description: 'Get the user\'s current local date, time, day of week, timezone, and Unix timestamp.',
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
        return {
          harnessMode: snap.harnessMode,
          provider: snap.telemetry.activeProvider,
          verdantVersion: snap.telemetry.verdantVersion,
          kernelAcceleration: snap.telemetry.kernelAcceleration,
          model: snap.model
            ? {
                id: snap.model.id,
                name: snap.model.displayName,
                quantization: snap.model.hparams.primaryQuantization,
                contextWindow: snap.model.maxContextWindow,
                backingKind: snap.model.backingKind,
              }
            : null,
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
        const notes = readWorkspaceNotes();
        if (action === 'list') {
          const list = Object.values(notes).map((n) => ({
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
        const key = title.toLowerCase();
        if (action === 'read') {
          const found = notes[key];
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
          const existed = Object.hasOwn(notes, key);
          delete notes[key];
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
            description: 'Substring or regex pattern when mode is "search".',
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
              length: Array.isArray(parsed) ? parsed.length : keys?.length ?? null,
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
}

/**
 * Builds the official `@assistant-ui/react` `ModelContext` for the active Verdant Harness mode,
 * seamlessly composing an optional user-imported custom harness override when present.
 */
export function buildVerdantModelContext(
  mode: VerdantHarnessMode,
  clientTools: Record<string, Tool<any, any>>,
  importedHarness: ImportedHarnessSpec | null,
): ModelContext {
  const spec = VERDANT_HARNESS_MODES[mode];
  const baseContext: ModelContext = {
    system: spec.system,
    callSettings: {
      temperature: spec.callSettings.temperature,
      topP: spec.callSettings.topP,
    },
    config: {
      reasoningEffort: spec.reasoningEffort,
    },
    ...(spec.enableTools ? { tools: clientTools } : {}),
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
      ...baseContext.config,
      ...(imported.config ?? {}),
    },
  };
}
