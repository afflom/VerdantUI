/**
 * Standards-based Imported Harness Configuration mapped directly to
 * `@assistant-ui/react`'s official `ModelContext` contract (`system`, `callSettings`, `config`).
 *
 * VerdantUI never invents or hardcodes system prompts. When no harness is imported,
 * `ModelContext.system` is `undefined` and the model runs cleanly with zero injected instructions.
 * When a user imports a standard JSON harness / ModelContext specification, its `system` message
 * and model parameters are registered directly via Assistant-UI's `modelContext.register(...)` API.
 */

import type {
  LanguageModelConfig,
  LanguageModelV1CallSettings,
  ModelContext,
} from '@assistant-ui/react';
import type { SamplingConfig } from './llamaRuntime';
import { MODEL_CATALOG, type CatalogModel } from './modelCatalog';
import type { PromptFormatMode } from './workerClient';

export interface HarnessSuggestionSpec {
  readonly prompt: string;
  readonly label?: string;
}

export interface ImportedHarnessSpec {
  readonly id: string;
  readonly name: string;
  readonly description?: string;
  readonly sourceLabel: string;
  readonly importedAt: number;
  /** Standard Assistant-UI ModelContext.system (omitted if harness defines no system message). */
  readonly system?: string;
  /** Standard Assistant-UI ModelContext.callSettings (`temperature`, `topP`, `maxTokens`, etc.). */
  readonly callSettings?: LanguageModelV1CallSettings;
  /** Standard Assistant-UI ModelContext.config (`modelName`, `reasoningEffort`, `baseUrl`). */
  readonly config?: LanguageModelConfig;
  /** Optional direct GGUF URL if the harness specifies an explicit model checkpoint. */
  readonly modelUrl?: string;
  /** Optional GGUF sampler extensions (topK / repeatPenalty / promptMode). */
  readonly topK?: number;
  readonly repeatPenalty?: number;
  readonly promptMode?: PromptFormatMode;
  /** Optional prompt suggestions explicitly provided by the imported harness. */
  readonly suggestions?: readonly HarnessSuggestionSpec[];
}

const HARNESS_STORAGE_KEY = 'verdant-ui:imported-harness:v1';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function readOptionalString(obj: Record<string, unknown>, ...keys: string[]): string | undefined {
  for (const key of keys) {
    const val = obj[key];
    if (typeof val === 'string' && val.trim().length > 0) {
      return val.trim();
    }
  }
  return undefined;
}

function readOptionalNumber(
  obj: Record<string, unknown>,
  min: number,
  max: number,
  ...keys: string[]
): number | undefined {
  for (const key of keys) {
    const val = obj[key];
    if (typeof val === 'number' && Number.isFinite(val) && val >= min && val <= max) {
      return val;
    }
  }
  return undefined;
}

function extractSystemFromMessages(messages: unknown): string | undefined {
  if (!Array.isArray(messages)) return undefined;
  const parts: string[] = [];
  for (const item of messages) {
    if (!isRecord(item)) continue;
    if (item.role !== 'system' && item.role !== 'developer') continue;
    if (typeof item.content === 'string' && item.content.trim().length > 0) {
      parts.push(item.content.trim());
    } else if (Array.isArray(item.content)) {
      const joined = item.content
        .filter((p): p is { type: string; text: string } => isRecord(p) && typeof p.text === 'string')
        .map((p) => p.text.trim())
        .filter(Boolean)
        .join('\n');
      if (joined) parts.push(joined);
    }
  }
  return parts.length > 0 ? parts.join('\n\n') : undefined;
}

function parseSuggestions(raw: unknown): HarnessSuggestionSpec[] | undefined {
  if (!Array.isArray(raw)) return undefined;
  const parsed: HarnessSuggestionSpec[] = [];
  for (const item of raw) {
    if (typeof item === 'string' && item.trim().length > 0) {
      parsed.push({ prompt: item.trim() });
    } else if (isRecord(item)) {
      const prompt = readOptionalString(item, 'prompt', 'text');
      const label = readOptionalString(item, 'label', 'title');
      if (prompt) {
        parsed.push(label ? { prompt, label } : { prompt });
      }
    }
  }
  return parsed.length > 0 ? parsed : undefined;
}

/**
 * Parses and validates an imported JSON harness specification.
 *
 * Supports:
 * - Official `@assistant-ui/react` `ModelContext` JSON (`{ system, callSettings, config }`)
 * - OpenAI Chat Completions / Agent JSON (`{ model, messages: [{ role: 'system', content }], temperature, top_p, max_tokens }`)
 * - llama.cpp / GGUF parameter JSON (`{ system_prompt, top_k, repeat_penalty, modelUrl }`)
 */
export function parseHarnessConfigJson(
  rawJson: string,
  sourceLabel = 'Imported JSON',
): ImportedHarnessSpec {
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawJson);
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    throw new Error(`Invalid JSON harness file: ${detail}`);
  }

  if (!isRecord(parsed)) {
    throw new Error('Harness configuration must be a JSON object.');
  }

  const callSettingsSource = isRecord(parsed.callSettings)
    ? parsed.callSettings
    : isRecord(parsed.parameters)
      ? parsed.parameters
      : isRecord(parsed.sampling)
        ? parsed.sampling
        : parsed;

  const configSource = isRecord(parsed.config) ? parsed.config : parsed;

  const system =
    readOptionalString(parsed, 'system', 'systemPrompt', 'system_prompt', 'instructions') ??
    extractSystemFromMessages(parsed.messages);

  const temperature = readOptionalNumber(callSettingsSource, 0, 2, 'temperature');
  const topP = readOptionalNumber(callSettingsSource, 0.05, 1, 'topP', 'top_p');
  const maxTokensRaw = readOptionalNumber(
    callSettingsSource,
    1,
    1_048_576,
    'maxTokens',
    'max_tokens',
    'max_completion_tokens',
  );
  const maxTokens = maxTokensRaw !== undefined ? Math.round(maxTokensRaw) : undefined;
  const presencePenalty = readOptionalNumber(
    callSettingsSource,
    -2,
    2,
    'presencePenalty',
    'presence_penalty',
  );
  const frequencyPenalty = readOptionalNumber(
    callSettingsSource,
    -2,
    2,
    'frequencyPenalty',
    'frequency_penalty',
  );
  const seedRaw = readOptionalNumber(callSettingsSource, 0, Number.MAX_SAFE_INTEGER, 'seed');
  const seed = seedRaw !== undefined ? Math.round(seedRaw) : undefined;

  const callSettings: LanguageModelV1CallSettings = {
    ...(temperature !== undefined ? { temperature } : {}),
    ...(topP !== undefined ? { topP } : {}),
    ...(maxTokens !== undefined ? { maxTokens } : {}),
    ...(presencePenalty !== undefined ? { presencePenalty } : {}),
    ...(frequencyPenalty !== undefined ? { frequencyPenalty } : {}),
    ...(seed !== undefined ? { seed } : {}),
  };

  const modelName =
    readOptionalString(configSource, 'modelName', 'model_name', 'model') ??
    readOptionalString(parsed, 'modelName', 'model_name', 'model');
  const reasoningEffort =
    readOptionalString(configSource, 'reasoningEffort', 'reasoning_effort') ??
    readOptionalString(parsed, 'reasoningEffort', 'reasoning_effort');
  const baseUrl = readOptionalString(configSource, 'baseUrl', 'base_url');

  const config: LanguageModelConfig = {
    ...(modelName !== undefined ? { modelName } : {}),
    ...(reasoningEffort !== undefined ? { reasoningEffort } : {}),
    ...(baseUrl !== undefined ? { baseUrl } : {}),
  };

  const topKRaw =
    readOptionalNumber(callSettingsSource, 1, 200, 'topK', 'top_k') ??
    readOptionalNumber(parsed, 1, 200, 'topK', 'top_k');
  const topK = topKRaw !== undefined ? Math.round(topKRaw) : undefined;
  const repeatPenalty =
    readOptionalNumber(callSettingsSource, 1, 2, 'repeatPenalty', 'repeat_penalty') ??
    readOptionalNumber(parsed, 1, 2, 'repeatPenalty', 'repeat_penalty');
  const modelUrlRaw = readOptionalString(parsed, 'modelUrl', 'model_url');
  const modelUrl = modelUrlRaw && /^https?:\/\//i.test(modelUrlRaw) ? modelUrlRaw : undefined;

  const rawPromptMode = readOptionalString(parsed, 'promptMode', 'prompt_mode');
  const promptMode: PromptFormatMode | undefined =
    rawPromptMode === 'auto' ||
    rawPromptMode === 'chatml' ||
    rawPromptMode === 'continuation' ||
    rawPromptMode === 'instruct'
      ? rawPromptMode
      : undefined;

  const suggestions = parseSuggestions(parsed.suggestions);

  const hasAnyModelContextField =
    system !== undefined ||
    Object.keys(callSettings).length > 0 ||
    Object.keys(config).length > 0 ||
    topK !== undefined ||
    repeatPenalty !== undefined ||
    modelUrl !== undefined ||
    promptMode !== undefined ||
    suggestions !== undefined;

  if (!hasAnyModelContextField) {
    throw new Error(
      'Harness JSON did not include any recognized ModelContext fields (system, callSettings, config, model, or messages).',
    );
  }

  const defaultName = sourceLabel.replace(/\.json$/i, '') || 'Imported Harness';
  const name = readOptionalString(parsed, 'name', 'title') ?? defaultName;
  const description = readOptionalString(parsed, 'description');

  return {
    id: `harness-${Date.now()}`,
    name,
    ...(description ? { description } : {}),
    sourceLabel,
    importedAt: Date.now(),
    ...(system !== undefined ? { system } : {}),
    ...(Object.keys(callSettings).length > 0 ? { callSettings } : {}),
    ...(Object.keys(config).length > 0 ? { config } : {}),
    ...(modelUrl !== undefined ? { modelUrl } : {}),
    ...(topK !== undefined ? { topK } : {}),
    ...(repeatPenalty !== undefined ? { repeatPenalty } : {}),
    ...(promptMode !== undefined ? { promptMode } : {}),
    ...(suggestions !== undefined ? { suggestions } : {}),
  };
}

/**
 * Converts an active `ImportedHarnessSpec` into an official `@assistant-ui/react` `ModelContext`.
 * When `harness` is `null`, returns `{}` so no system message or speculative defaults are injected.
 */
export function buildModelContextFromHarness(harness: ImportedHarnessSpec | null): ModelContext {
  if (!harness) return {};
  return {
    ...(harness.system !== undefined ? { system: harness.system } : {}),
    ...(harness.callSettings ? { callSettings: { ...harness.callSettings } } : {}),
    ...(harness.config ? { config: { ...harness.config } } : {}),
  };
}

/**
 * Merges standard Assistant-UI `ModelContext.callSettings` with the runtime's sampling config
 * for a single inference run without mutating unrelated parameters.
 */
export function applyCallSettingsToSampling(
  base: SamplingConfig,
  callSettings?: LanguageModelV1CallSettings,
  harness?: ImportedHarnessSpec | null,
): SamplingConfig {
  const temperature =
    typeof callSettings?.temperature === 'number' && Number.isFinite(callSettings.temperature)
      ? callSettings.temperature
      : base.temperature;
  const topP =
    typeof callSettings?.topP === 'number' && Number.isFinite(callSettings.topP)
      ? callSettings.topP
      : base.topP;
  const maxTokens =
    typeof callSettings?.maxTokens === 'number' && Number.isFinite(callSettings.maxTokens)
      ? Math.max(1, Math.round(callSettings.maxTokens))
      : base.maxTokens;
  const fullContextOutput =
    typeof callSettings?.maxTokens === 'number' ? false : base.fullContextOutput;
  const topK =
    typeof harness?.topK === 'number' && Number.isFinite(harness.topK) ? harness.topK : base.topK;
  const repeatPenalty =
    typeof harness?.repeatPenalty === 'number' && Number.isFinite(harness.repeatPenalty)
      ? harness.repeatPenalty
      : base.repeatPenalty;

  return {
    ...base,
    temperature,
    topP,
    topK,
    repeatPenalty,
    maxTokens,
    fullContextOutput,
  };
}

/**
 * Matches an imported harness's requested `config.modelName` against `MODEL_CATALOG`
 * so the user can load the required checkpoint directly.
 */
export function matchCatalogModelForHarness(
  harness: ImportedHarnessSpec | null,
): CatalogModel | null {
  const requested = harness?.config?.modelName?.trim().toLowerCase();
  if (!requested) return null;

  const exact = MODEL_CATALOG.find(
    (m) =>
      m.id.toLowerCase() === requested ||
      m.name.toLowerCase() === requested ||
      m.shortName.toLowerCase() === requested,
  );
  if (exact) return exact;

  const normalizedReq = requested.replace(/[^a-z0-9]+/g, ' ').trim();
  if (!normalizedReq) return null;

  return (
    MODEL_CATALOG.find((m) => {
      const hay = `${m.id} ${m.name} ${m.shortName}`.toLowerCase().replace(/[^a-z0-9]+/g, ' ');
      return hay.includes(normalizedReq) || normalizedReq.includes(m.shortName.toLowerCase().replace(/[^a-z0-9]+/g, ' '));
    }) ?? null
  );
}

/**
 * Serializes the active harness (or current runtime parameters) into a clean,
 * standards-compliant `@assistant-ui/react` `ModelContext` JSON document.
 */
export function serializeHarnessConfig(
  harness: ImportedHarnessSpec | null,
  sampling: SamplingConfig,
  activeModelName?: string,
): string {
  const payload: Record<string, unknown> = {
    name: harness?.name ?? 'VerdantUI Harness',
    ...(harness?.description ? { description: harness.description } : {}),
    ...(harness?.system !== undefined ? { system: harness.system } : {}),
    callSettings: {
      temperature: harness?.callSettings?.temperature ?? sampling.temperature,
      topP: harness?.callSettings?.topP ?? sampling.topP,
      ...(harness?.callSettings?.maxTokens !== undefined
        ? { maxTokens: harness.callSettings.maxTokens }
        : !sampling.fullContextOutput
          ? { maxTokens: sampling.maxTokens }
          : {}),
    },
    config: {
      ...(harness?.config?.modelName || activeModelName
        ? { modelName: harness?.config?.modelName ?? activeModelName }
        : {}),
      ...(harness?.config?.reasoningEffort
        ? { reasoningEffort: harness.config.reasoningEffort }
        : {}),
    },
  };
  return JSON.stringify(payload, null, 2);
}

export function loadSavedHarness(): ImportedHarnessSpec | null {
  if (typeof window === 'undefined' || typeof window.localStorage === 'undefined') {
    return null;
  }
  try {
    const raw = window.localStorage.getItem(HARNESS_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as ImportedHarnessSpec;
    if (!parsed || typeof parsed !== 'object' || typeof parsed.name !== 'string') {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

export function saveHarnessToStorage(harness: ImportedHarnessSpec | null): void {
  if (typeof window === 'undefined' || typeof window.localStorage === 'undefined') {
    return;
  }
  try {
    if (!harness) {
      window.localStorage.removeItem(HARNESS_STORAGE_KEY);
    } else {
      window.localStorage.setItem(HARNESS_STORAGE_KEY, JSON.stringify(harness));
    }
  } catch {
    // Ignore storage error
  }
}
