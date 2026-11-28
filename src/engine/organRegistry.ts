/**
 * Unified Declarative Organ Registry (`src/engine/organRegistry.ts`)
 *
 * Single authoritative manifest for all 8 VerdantUI MediaTypes and all 7 `tool-organ` roles:
 *   - Default compartment (`'personal'`, proved by `VerdantUI.Compartment`)
 *   - Required FK-07 capability grants (proved by `VerdantUI.Grants`)
 *   - Required persisted weight keys that MUST survive `normalizeOrganWeights`
 *   - Canonical structured `@Prism status` serializer (eliminates `"[object Object]"`)
 *   - OTEL-instrumented lifecycle helpers (`organ.status`, `organ.create`, `organ.refine`)
 */

import type { ObserverStreamBus } from './observerStreamBus';
import {
  PRISM_MEDIA_CATALOG,
  type PrismMediaType,
  type ProceduralOrganWeights,
  type ToolOrganWeights,
} from './prismMediaOrgans';
import type { PrismDeclaration } from './verdantStudioApi';

export type OrganGrantRequirement = 'prism-runtime' | 'prism-storage' | 'prism-browser';

export interface OrganStatusPayload {
  readonly status: 'ok';
  readonly prismName: string;
  readonly mediaType: PrismMediaType;
  readonly role: string;
  readonly summary: string;
  readonly activeOrgans: readonly string[];
  readonly metrics: Readonly<Record<string, number | string>>;
  readonly request: string;
  readonly refused: false;
}

export interface OrganDescriptor {
  readonly id: string;
  readonly mediaType: PrismMediaType;
  readonly role: string;
  readonly label: string;
  readonly defaultCompartment: 'personal';
  readonly requiredGrants: readonly OrganGrantRequirement[];
  readonly persistedWeightKeys: readonly string[];
  readonly samplePrompt: string;
  readonly sampleRefinePrompt: string | null;
  readonly suggestionChips: readonly string[];
}

export const ORGAN_REGISTRY: readonly OrganDescriptor[] = Object.freeze([
  {
    id: 'picture',
    mediaType: 'picture',
    role: 'picture',
    label: 'Picture',
    defaultCompartment: 'personal',
    requiredGrants: ['prism-runtime', 'prism-browser'],
    persistedWeightKeys: ['generator', 'palette', 'lightingMood', 'title'],
    samplePrompt: 'Create a minimalist mountain sunrise landscape picture',
    sampleRefinePrompt: 'Shift palette to cobalt twilight',
    suggestionChips: ['Make lighting golden hour', 'Try cobalt twilight palette', 'Export PNG'],
  },
  {
    id: 'audio',
    mediaType: 'audio',
    role: 'audio',
    label: 'Audio',
    defaultCompartment: 'personal',
    requiredGrants: ['prism-runtime', 'prism-browser'],
    persistedWeightKeys: ['style', 'bpm', 'scale', 'rootMidi', 'durationSec'],
    samplePrompt: 'Compose a calm ambient synth soundtrack at 88 BPM',
    sampleRefinePrompt: 'Increase tempo to 104 BPM',
    suggestionChips: ['Raise tempo to 110 BPM', 'Switch scale to dorian', 'Export stereo WAV'],
  },
  {
    id: 'video',
    mediaType: 'video',
    role: 'video',
    label: 'Video',
    defaultCompartment: 'personal',
    requiredGrants: ['prism-runtime', 'prism-browser'],
    persistedWeightKeys: ['style', 'fps', 'durationSec', 'scenes', 'palette'],
    samplePrompt: 'Create a 3-scene ocean sunset motion storyboard video',
    sampleRefinePrompt: 'Add a calm night lantern final scene',
    suggestionChips: ['Add a closing scene', 'Slow down scene transitions', 'Preview storyboard'],
  },
  {
    id: 'written-work',
    mediaType: 'written-work',
    role: 'written-work',
    label: 'Writing',
    defaultCompartment: 'personal',
    requiredGrants: ['prism-runtime', 'prism-storage', 'prism-browser'],
    persistedWeightKeys: ['genre', 'tone', 'chapters', 'targetWords'],
    samplePrompt: 'Create a short sci-fi novella workspace about orbital gardens',
    sampleRefinePrompt: 'Add a chapter about the greenhouse dome',
    suggestionChips: ['Add a new chapter', 'Refine chapter outline', 'Export manuscript'],
  },
  {
    id: 'interactive-game',
    mediaType: 'interactive-game',
    role: 'interactive-game',
    label: 'Game',
    defaultCompartment: 'personal',
    requiredGrants: ['prism-runtime', 'prism-storage', 'prism-browser'],
    persistedWeightKeys: ['template', 'difficulty', 'gridSize', 'palette'],
    samplePrompt: 'Build a playable 4x4 sliding puzzle game',
    sampleRefinePrompt: 'Increase difficulty to hard',
    suggestionChips: ['Switch difficulty', 'Change board theme', 'Reset high score'],
  },
  {
    id: 'research-program',
    mediaType: 'research-program',
    role: 'research-program',
    label: 'Research',
    defaultCompartment: 'personal',
    requiredGrants: ['prism-runtime', 'prism-storage', 'prism-browser'],
    persistedWeightKeys: ['question', 'hypotheses', 'milestones', 'compartment'],
    samplePrompt: 'Create a solar microgrid efficiency research program',
    sampleRefinePrompt: 'Add hypothesis: battery thermal buffer increases winter yield by 12%',
    suggestionChips: ['Add a hypothesis', 'Log an experiment milestone', 'Review evidence'],
  },
  {
    id: 'space',
    mediaType: 'space',
    role: 'space',
    label: 'Space',
    defaultCompartment: 'personal',
    requiredGrants: ['prism-runtime', 'prism-storage', 'prism-browser'],
    persistedWeightKeys: ['spaceTitle', 'layout', 'items', 'palette'],
    samplePrompt: 'Create a Creative Projects space',
    sampleRefinePrompt: 'Add a note card for weekend sprint goals',
    suggestionChips: ['Switch layout style', 'Add a curated item', 'Share view'],
  },
  {
    id: 'tool-organ:budget-tracker',
    mediaType: 'tool-organ',
    role: 'budget-tracker',
    label: 'Budget Tracker',
    defaultCompartment: 'personal',
    requiredGrants: ['prism-runtime', 'prism-storage', 'prism-browser'],
    persistedWeightKeys: ['role', 'monthlyBudget', 'expenses', 'compartment'],
    samplePrompt: 'Build a monthly budget tracker utility',
    sampleRefinePrompt: 'Add expense $18 lunch',
    suggestionChips: ['Add expense $24 groceries', 'Set monthly budget to $2500', 'Check status'],
  },
  {
    id: 'tool-organ:flashcards',
    mediaType: 'tool-organ',
    role: 'flashcards',
    label: 'Flashcards',
    defaultCompartment: 'personal',
    requiredGrants: ['prism-runtime', 'prism-storage', 'prism-browser'],
    persistedWeightKeys: ['role', 'cards', 'compartment'],
    samplePrompt: 'Create a spaced-repetition flashcards study utility',
    sampleRefinePrompt: 'Add flashcard: What is the derivative of sin(x)? -> cos(x)',
    suggestionChips: ['Add a study card', 'Shuffle deck', 'Check status'],
  },
  {
    id: 'tool-organ:focus-timer',
    mediaType: 'tool-organ',
    role: 'focus-timer',
    label: 'Focus Timer',
    defaultCompartment: 'personal',
    requiredGrants: ['prism-runtime', 'prism-storage', 'prism-browser'],
    persistedWeightKeys: ['role', 'workMinutes', 'breakMinutes', 'compartment'],
    samplePrompt: 'Create a Pomodoro focus timer utility',
    sampleRefinePrompt: 'Set focus to 30 minutes and break to 10 minutes',
    suggestionChips: ['Set 25/5 Pomodoro', 'Set 50/10 deep work', 'Check status'],
  },
  {
    id: 'tool-organ:habit-guardian',
    mediaType: 'tool-organ',
    role: 'habit-guardian',
    label: 'Habit Guardian',
    defaultCompartment: 'personal',
    requiredGrants: ['prism-runtime', 'prism-storage', 'prism-browser'],
    persistedWeightKeys: ['role', 'items', 'compartment'],
    samplePrompt: 'Create a daily hydration and stretching habit tracker utility',
    sampleRefinePrompt: 'Add habit: Evening walk 20 minutes',
    suggestionChips: ['Add daily habit', 'Review streak', 'Check status'],
  },
  {
    id: 'tool-organ:schedule-guard',
    mediaType: 'tool-organ',
    role: 'schedule-guard',
    label: 'Schedule Guard',
    defaultCompartment: 'personal',
    requiredGrants: ['prism-runtime', 'prism-storage', 'prism-browser'],
    persistedWeightKeys: ['role', 'items', 'compartment'],
    samplePrompt: 'Create a daily focus schedule guard utility',
    sampleRefinePrompt: 'Add block: 14:00 Deep architecture review',
    suggestionChips: ['Add time block', 'Check schedule', 'Check status'],
  },
  {
    id: 'tool-organ:calculator',
    mediaType: 'tool-organ',
    role: 'calculator',
    label: 'Calculator',
    defaultCompartment: 'personal',
    requiredGrants: ['prism-runtime', 'prism-storage', 'prism-browser'],
    persistedWeightKeys: ['role', 'items', 'compartment'],
    samplePrompt: 'Create a quick unit converter and savings calculator utility',
    sampleRefinePrompt: 'Add preset: 15% tip calculator',
    suggestionChips: ['Add calculation preset', 'Check status'],
  },
  {
    id: 'tool-organ:workflow-helper',
    mediaType: 'tool-organ',
    role: 'workflow-helper',
    label: 'Workflow Helper',
    defaultCompartment: 'personal',
    requiredGrants: ['prism-runtime', 'prism-storage', 'prism-browser'],
    persistedWeightKeys: ['role', 'items', 'compartment'],
    samplePrompt: 'Create a release checklist workflow helper utility',
    sampleRefinePrompt: 'Add step: Verify production health checks',
    suggestionChips: ['Add checklist step', 'Check progress', 'Check status'],
  },
]);

export function lookupOrganDescriptor(
  mediaType: PrismMediaType,
  role?: string,
): OrganDescriptor {
  if (mediaType === 'tool-organ' && role) {
    const exact = ORGAN_REGISTRY.find((d) => d.mediaType === 'tool-organ' && d.role === role);
    if (exact) return exact;
  }
  const byMedia = ORGAN_REGISTRY.find((d) => d.mediaType === mediaType);
  if (!byMedia) {
    throw new Error(`refused: unregistered organ mediaType "${String(mediaType)}"`);
  }
  return byMedia;
}

/**
 * Formats a deterministic, structured status payload for any procedural Prism organ.
 * Never returns an unserialized object string (`"[object Object]"`).
 */
export function formatOrganStatusPayload(params: {
  readonly prismName: string;
  readonly summary: string;
  readonly decl: PrismDeclaration;
  readonly mediaType: PrismMediaType;
  readonly activeOrgans: readonly string[];
  readonly weights: ProceduralOrganWeights;
  readonly requestLine: string;
  readonly bus?: ObserverStreamBus;
}): OrganStatusPayload {
  const w = params.weights as unknown as Record<string, unknown>;
  const role =
    params.mediaType === 'tool-organ'
      ? String((params.weights as ToolOrganWeights).role ?? 'workflow-helper')
      : params.mediaType;

  const span = params.bus?.startSpan('organ.status', {
    endpointKind: 'prism',
    endpointId: params.prismName,
    parentContext: params.bus.getActiveTraceContext(),
    attributes: {
      'verdant.organ.media_type': params.mediaType,
      'verdant.organ.role': role,
    },
  });

  const metrics: Record<string, number | string> = {};
  if (Array.isArray(w.items)) metrics.itemCount = w.items.length;
  if (Array.isArray(w.expenses)) {
    metrics.expenseCount = w.expenses.length;
    metrics.monthlyBudget = typeof w.monthlyBudget === 'number' ? w.monthlyBudget : 2000;
  }
  if (Array.isArray(w.cards)) metrics.cardCount = w.cards.length;
  if (Array.isArray(w.chapters)) metrics.chapterCount = w.chapters.length;
  if (Array.isArray(w.hypotheses)) metrics.hypothesisCount = w.hypotheses.length;
  if (Array.isArray(w.scenes)) metrics.sceneCount = w.scenes.length;
  if (typeof w.workMinutes === 'number') metrics.workMinutes = w.workMinutes;
  if (typeof w.breakMinutes === 'number') metrics.breakMinutes = w.breakMinutes;
  if (typeof w.bpm === 'number') metrics.bpm = w.bpm;

  const payload: OrganStatusPayload = {
    status: 'ok',
    prismName: params.prismName,
    mediaType: params.mediaType,
    role,
    summary: params.summary || params.decl.concept.purpose || PRISM_MEDIA_CATALOG[params.mediaType].shortSummary,
    activeOrgans: params.activeOrgans,
    metrics,
    request: params.requestLine || 'status',
    refused: false,
  };

  span?.setAttributes({
    'verdant.organ.refused': false,
    'verdant.organ.metric_keys': Object.keys(metrics).join(','),
  });
  span?.end('ok');
  return payload;
}
