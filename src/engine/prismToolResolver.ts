/**
 * Autonomous Prism Tool-Call Resolver (`src/engine/prismToolResolver.ts`)
 *
 * Implements DEC-013 (Every Chat Tool-Call Resolves Through a Prism):
 * - When the unified model attempts ANY `<tool_call>` in a chat thread, the Observer resolves
 *   it to a Published Prism inside the active Space (`Observer ⊇ User ⊇ Prisms`).
 * - If a matching Published Prism already exists in the active Space, it is reused directly
 *   (zero duplication, strict cross-space isolation).
 * - If NO matching Prism exists yet, VerdantUI autonomously synthesizes a declarative,
 *   procedurally compiled Prism (zero LLM-written JavaScript!), publishes it into the active
 *   Space, registers it as an active Observer organ, and dispatches the tool-call payload to it.
 */

import type { VerdantCognitiveObserver } from './cognitiveObserver';
import type { ContentGalleryStore } from './contentGalleryStore';
import type { PrismHostPool } from './prismHostPool';
import type { PrismMediaType } from './prismMediaOrgans';
import {
  prismToolName,
  type PrismStore,
  type PublishedPrismSummary,
} from './prismStudio';
import {
  assembleProceduralPrismOps,
  classifyArtifactMediaType,
  parseDiscoveryAgentResponse,
} from './prismStudioAgent';
import {
  StudioChatBridge,
  type StudioChatArtifactResult,
} from './studioChatBridge';
import type { UnifiedExpertRegistry } from './unifiedModelBasis';
import type { ContextCompartmentId } from './unifiedPossessionModel';

export interface PrismToolInvocationResult {
  readonly __prismInvocation: true;
  readonly autoCreated: boolean;
  readonly prismId: string;
  readonly prismName: string;
  readonly mediaType: PrismMediaType;
  readonly compartmentId: ContextCompartmentId;
  readonly digest: string;
  readonly summary?: string;
  readonly galleryLabel?: string;
  readonly categories?: readonly string[];
  readonly previewImageDataUrl?: string;
  readonly previewImageWidth?: number;
  readonly previewImageHeight?: number;
  readonly audioWavDataUrl?: string;
  readonly audioDurationSec?: number;
  readonly writtenExcerpt?: string;
  readonly output: unknown;
}

/** Converts a snake_case / kebab-case / camelCase tool name into a clean human title. */
export function humanizeToolName(rawToolName: string): string {
  const cleaned = rawToolName
    .replace(/^prism[_\-\s]+/i, '')
    .replace(/^(?:create|build|make|run|use|open|launch|invoke|play|compose|generate|start)[_\-\s]+/i, '')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/[_\-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  if (!cleaned) return 'Custom Tool';
  return cleaned
    .split(' ')
    .map((w) => (w.length > 0 ? w[0]!.toUpperCase() + w.slice(1).toLowerCase() : ''))
    .join(' ')
    .slice(0, 64);
}

function normalizeMatchKey(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, '');
}

/**
 * Finds an already-published Prism visible in the active Space that matches `toolName`
 * (by exact `prism_*` tool name, slug, display name, or normalized keyword token).
 */
export function findMatchingPublishedPrism(
  toolName: string,
  visiblePublished: readonly PublishedPrismSummary[],
): PublishedPrismSummary | null {
  if (!toolName.trim() || visiblePublished.length === 0) return null;
  const targetExact = toolName.trim().toLowerCase();
  const targetNorm = normalizeMatchKey(toolName.replace(/^prism[_-]/i, ''));
  const targetHumanNorm = normalizeMatchKey(humanizeToolName(toolName));

  for (const p of visiblePublished) {
    if (prismToolName(p).toLowerCase() === targetExact) return p;
    if (p.slug.toLowerCase() === targetExact || p.name.toLowerCase() === targetExact) return p;
    const pNameNorm = normalizeMatchKey(p.name);
    const pSlugNorm = normalizeMatchKey(p.slug);
    if (
      (targetNorm.length >= 3 && (pNameNorm === targetNorm || pSlugNorm === targetNorm)) ||
      (targetHumanNorm.length >= 3 && (pNameNorm === targetHumanNorm || pSlugNorm === targetHumanNorm))
    ) {
      return p;
    }
  }
  return null;
}

function buildSynthesisDescription(
  toolName: string,
  args: Record<string, unknown>,
  latestUserText: string,
): { readonly title: string; readonly description: string } {
  const explicitName =
    typeof args.name === 'string' && args.name.trim()
      ? args.name.trim().slice(0, 64)
      : typeof args.title === 'string' && args.title.trim()
        ? args.title.trim().slice(0, 64)
        : humanizeToolName(toolName);

  const explicitDesc = [
    typeof args.description === 'string' ? args.description : '',
    typeof args.purpose === 'string' ? args.purpose : '',
    typeof args.prompt === 'string' ? args.prompt : '',
    typeof args.goal === 'string' ? args.goal : '',
    typeof args.query === 'string' ? args.query : '',
    typeof args.message === 'string' ? args.message : '',
  ]
    .map((s) => s.trim())
    .filter(Boolean)
    .join('. ');

  const argPairs = Object.entries(args)
    .filter(([k, v]) => !['name', 'title', 'description', 'purpose', 'prompt', 'goal'].includes(k) && v !== undefined)
    .map(([k, v]) => `${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`)
    .slice(0, 6)
    .join(', ');

  const parts = [
    explicitName,
    explicitDesc || latestUserText.trim().slice(0, 220),
    argPairs ? `Parameters: ${argPairs}` : '',
  ].filter(Boolean);

  return {
    title: explicitName,
    description: parts.join(' — ').slice(0, 480),
  };
}

/**
 * Resolves a model `<tool_call>` through VerdantUI's Observer–Prism relationship:
 * 1. Reuses any matching Published Prism visible in `activeSpace`.
 * 2. Otherwise procedurally synthesizes, publishes, and registers a new Prism in `activeSpace`
 *    (strictly zero LLM-written JS, zero cross-space leakage).
 * 3. Dispatches the tool call's arguments to that Prism and returns a structured result card.
 */
export async function resolveOrCreatePrismForToolCall(params: {
  readonly toolName: string;
  readonly args: Record<string, unknown>;
  readonly latestUserText: string;
  readonly activeSpace: ContextCompartmentId;
  readonly store: PrismStore;
  readonly pool: PrismHostPool;
  readonly observer: VerdantCognitiveObserver;
  readonly galleries?: ContentGalleryStore;
  readonly registry?: UnifiedExpertRegistry;
  readonly studioChatBridge?: StudioChatBridge;
  readonly signal?: AbortSignal;
}): Promise<PrismToolInvocationResult> {
  const { toolName, args, latestUserText, activeSpace, store, pool, observer, studioChatBridge, signal } = params;

  observer.syncPublishedPrisms(store.list());
  const visiblePublished = observer.getVisiblePublishedPrismsForSpace(
    store.listPublishedSummaries(),
    activeSpace,
  );

  // 1. Reuse an existing Space-visible Published Prism if one already fulfills this tool-call
  const existing = findMatchingPublishedPrism(toolName, visiblePublished);
  if (existing) {
    const dispatched = await pool.dispatchPublishedPrism(existing.id, args, { signal });
    const proj = store.get(existing.id);
    const mediaType: PrismMediaType =
      proj?.declaration?.view?.proceduralOrgan?.mediaType ??
      proj?.declaration?.concept.mediaType ??
      'tool-organ';
    return {
      __prismInvocation: true,
      autoCreated: false,
      prismId: existing.id,
      prismName: dispatched.prismName,
      mediaType,
      compartmentId: activeSpace,
      digest: dispatched.digest,
      output: dispatched.result,
    };
  }

  // 2. If full StudioChatBridge is wired, synthesize through Verdant Studio with real inline PNG/WAV/Gallery/Basis integration!
  const { title, description } = buildSynthesisDescription(toolName, args, latestUserText);
  if (studioChatBridge) {
    const res: StudioChatArtifactResult = await studioChatBridge.createArtifactFromPrompt({
      prompt: `${title}. ${description}`,
      title,
      weightOverrides: { ...args, compartmentId: activeSpace },
    });
    return {
      __prismInvocation: true,
      autoCreated: true,
      prismId: res.prismId,
      prismName: res.prismName,
      mediaType: res.mediaType,
      compartmentId: activeSpace,
      digest: res.digest,
      summary: res.summary,
      galleryLabel: res.galleryLabel,
      categories: res.categories,
      previewImageDataUrl: res.previewImageDataUrl,
      previewImageWidth: res.previewImageWidth,
      previewImageHeight: res.previewImageHeight,
      audioWavDataUrl: res.audioWavDataUrl,
      audioDurationSec: res.audioDurationSec,
      writtenExcerpt: res.writtenExcerpt,
      output: res.output,
    };
  }

  // 3. Standalone fallback when called without StudioChatBridge (headless unit tests)
  const classified = classifyArtifactMediaType(`${title}. ${description}`);
  const created = await store.createDeclarative('undefined');
  const baseDecl = created.declaration!;

  const { spec } = parseDiscoveryAgentResponse(
    JSON.stringify({
      name: title,
      purpose: description,
      mediaType: classified.mediaType,
      readyToBuild: true,
      weights: {
        mediaType: classified.mediaType,
        ...args,
        compartmentId: activeSpace,
      },
      reply: `Created ${title} to handle this task.`,
    }),
    description,
    baseDecl,
    2,
    classified.mediaType,
  );

  const assembled = assembleProceduralPrismOps(baseDecl, { ...spec, name: title, readyToBuild: true }, description);
  const ops = [
    ...assembled.ops,
    { op: 'set-concept-profile' as const, compartmentId: activeSpace },
  ];
  await store.applyStudioOp(created.id, { op: 'batch', ops }, { pushRevision: false });
  const published = await store.publish(created.id);
  observer.syncPublishedPrisms(store.list());

  // 3. Execute the tool call through the newly synthesized Published Prism
  const dispatched = await pool.dispatchPublishedPrism(created.id, args, { signal });
  return {
    __prismInvocation: true,
    autoCreated: true,
    prismId: created.id,
    prismName: published?.published?.name ?? title,
    mediaType: spec.mediaType,
    compartmentId: activeSpace,
    digest: dispatched.digest,
    output: dispatched.result,
  };
}
