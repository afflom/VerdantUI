/**
 * Zero-Cutoff Realtime Content Streamer (`src/engine/realtimeWeightStreamer.ts`)
 *
 * Dynamically imported (`0 KiB` main-chunk cost) 100% client-side content ingestion engine:
 *   - Streams ONLY from user-enabled Content Streams in `ContentStreamCatalogStore`
 *     (Wikipedia, Wikidata, Wiktionary, OpenAlex, arXiv, Crossref, GitHub, npm, crates.io,
 *     MDN, Open-Meteo, Wikimedia, OpenStreetMap, plus user-added RSS/Atom, JSON, and web URLs).
 *   - Enforces strict two-stage Content-Addressed Deduplication:
 *       1. Pre-fetch freshness TTL check in `LivingTrainingBasis` -> `0` network calls on fresh hit.
 *       2. Post-fetch `sha256` `kappa` check -> `0` duplicate `UnifiedSemanticAssociator` updates.
 *   - Folds novel canonical chunks directly into `UnifiedSemanticAssociator` (`verdantGatedDeltaStepF32`)
 *     and `LivingTrainingBasis` so VerdantUI has no static training cutoff and shares live grounded
 *     facts seamlessly across Chat turns and Verdant Studio Prisms.
 */

import type {
  ContentStreamCatalogStore,
  ContentStreamSpec,
} from './contentStreamCatalog';
import { sha256HexSync } from './prismStudio';
import type { UnifiedExpertRegistry } from './unifiedModelBasis';

export interface StreamedKnowledgeChunk {
  readonly kappa: string;
  readonly conceptId: string;
  readonly streamId: string;
  readonly catalogId: string;
  readonly provider: string;
  readonly title: string;
  readonly sourceUrl: string;
  readonly canonicalText: string;
  readonly attributes: readonly string[];
  readonly fetchedAt: number;
  readonly ttlMs: number;
  readonly deduplicatedHit: boolean;
}

export interface StreamRealtimeResult {
  readonly chunks: readonly StreamedKnowledgeChunk[];
  readonly networkFetches: number;
  readonly freshnessCacheHits: number;
  readonly kappaDedupHits: number;
  readonly foldedNovelChunks: number;
  readonly promptContextBlock: string | null;
}

interface RawFetchedItem {
  readonly title: string;
  readonly sourceUrl: string;
  readonly rawText: string;
  readonly attributes?: readonly string[];
}

const STREAM_FETCH_TIMEOUT_MS = 2_800;
const MAX_CHUNK_CHARS = 1_100;

/** Strips HTML scripts, styles, tags, and noisy whitespace into compact canonical text. */
export function canonicalizeStreamText(raw: string, maxChars = MAX_CHUNK_CHARS): string {
  return raw
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/gi, ' ')
    .replace(/&amp;/gi, '&')
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>')
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, maxChars);
}

/** Deterministic content-addressed `kappa` for any canonical knowledge chunk. */
export function computeKnowledgeKappa(canonicalText: string, sourceUrl: string): string {
  const norm = canonicalText.toLowerCase().replace(/\s+/g, ' ').trim();
  return `know:${sha256HexSync(`${norm}|${sourceUrl.trim().toLowerCase()}`).slice(0, 24)}`;
}

function extractSearchTopic(prompt: string): string {
  const cleaned = prompt
    .replace(/https?:\/\/\S+/gi, ' ')
    .replace(
      /^\s*(?:please\s+)?(?:tell\s+me\s+about|what\s+is|who\s+is|explain|research|find\s+papers\s+(?:on|about)|look\s+up|search\s+for|create\s+(?:a|an)\s+(?:picture|video|document|research|story|game|utility)\s+(?:of|about|for)|latest\s+news\s+(?:on|about))\s+/i,
      '',
    )
    .replace(/[?!.,;:]+$/g, '')
    .trim();
  return (cleaned || prompt.trim()).slice(0, 120);
}

/** Scores each user-enabled stream by lexical/domain relevance to `query` and picks top `limit`. */
export function selectRelevantEnabledStreams(
  query: string,
  enabledStreams: readonly ContentStreamSpec[],
  limit = 2,
): readonly ContentStreamSpec[] {
  if (enabledStreams.length === 0) return [];
  const lower = query.toLowerCase();
  const urlsInQuery = lower.match(/https?:\/\/\S+/g) ?? [];

  const scored = enabledStreams.map((stream) => {
    let score = 10;
    // Direct URL match on a custom stream wins decisively
    try {
      const host = new URL(stream.endpointUrl).hostname.toLowerCase();
      for (const u of urlsInQuery) {
        if (u.includes(host)) {
          score += 100;
        }
      }
    } catch {
      // Ignore malformed URL safely
    }
    if (!stream.builtIn) score += 22; // User-added custom streams have explicit personal relevance
    for (const tag of stream.tags) {
      if (tag.length >= 2 && lower.includes(tag.toLowerCase())) {
        score += 35;
      }
    }
    if (stream.providerKind === 'open-meteo' && /\b(weather|forecast|temperature|rain|snow|wind|humidity)\b/.test(lower)) {
      score += 90;
    }
    if (
      (stream.providerKind === 'openalex' || stream.providerKind === 'arxiv' || stream.providerKind === 'crossref') &&
      /\b(paper|papers|study|studies|research|arxiv|journal|citation|theorem|algorithm|model|neural|biology|physics|medicine)\b/.test(lower)
    ) {
      score += 65;
    }
    if (
      (stream.providerKind === 'github' || stream.providerKind === 'npm' || stream.providerKind === 'crates' || stream.providerKind === 'mdn') &&
      /\b(code|library|package|npm|rust|crate|github|repo|api|typescript|javascript|react|css|html|wasm|web|release)\b/.test(lower)
    ) {
      score += 65;
    }
    if (stream.providerKind === 'wikipedia' || stream.providerKind === 'wikidata') {
      score += 25;
    }
    return { stream, score };
  });

  scored.sort((a, b) => b.score - a.score || a.stream.id.localeCompare(b.stream.id));
  return scored.slice(0, Math.max(1, limit)).map((x) => x.stream);
}

async function fetchWithTimeout(
  url: string,
  fetchImpl: typeof fetch,
  signal?: AbortSignal,
  timeoutMs = STREAM_FETCH_TIMEOUT_MS,
): Promise<Response> {
  const ctrl = new AbortController();
  const onAbort = () => ctrl.abort();
  signal?.addEventListener('abort', onAbort, { once: true });
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    return await fetchImpl(url, {
      headers: { Accept: 'application/json, application/xml, text/xml, text/plain, text/html;q=0.8' },
      signal: ctrl.signal,
    });
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', onAbort);
  }
}

/** Lightweight zero-dependency RSS 2.0 / Atom XML feed parser. */
export function parseRssOrAtomXml(xml: string, fallbackUrl: string): RawFetchedItem[] {
  const items: RawFetchedItem[] = [];
  const blockRe = /<(?:item|entry)\b[^>]*>([\s\S]*?)<\/(?:item|entry)>/gi;
  let m: RegExpExecArray | null;
  while ((m = blockRe.exec(xml)) !== null && items.length < 4) {
    const body = m[1] ?? '';
    const titleM = body.match(/<title[^>]*>(?:<!\[CDATA\[)?([\s\S]*?)(?:\]\]>)?<\/title>/i);
    const descM = body.match(
      /<(?:description|summary|content)[^>]*>(?:<!\[CDATA\[)?([\s\S]*?)(?:\]\]>)?<\/(?:description|summary|content)>/i,
    );
    const linkM =
      body.match(/<link[^>]+href=["']([^"']+)["']/i) ??
      body.match(/<link[^>]*>([\s\S]*?)<\/link>/i);
    const title = canonicalizeStreamText(titleM?.[1] ?? 'Feed update', 100);
    const rawText = canonicalizeStreamText(descM?.[1] ?? title);
    const sourceUrl = (linkM?.[1] ?? fallbackUrl).trim();
    if (rawText.length >= 12) {
      items.push({ title, sourceUrl, rawText });
    }
  }
  return items;
}

async function fetchFromStreamProvider(
  stream: ContentStreamSpec,
  topic: string,
  fetchImpl: typeof fetch,
  signal?: AbortSignal,
): Promise<RawFetchedItem[]> {
  const q = encodeURIComponent(topic);

  if (stream.providerKind === 'wikipedia' || stream.providerKind === 'wiktionary' || stream.providerKind === 'wikimedia') {
    const url = `${stream.endpointUrl}?action=query&list=search&srsearch=${q}&srlimit=2&utf8=1&format=json&origin=*`;
    const res = await fetchWithTimeout(url, fetchImpl, signal);
    if (!res.ok) return [];
    const data = (await res.json()) as {
      query?: { search?: Array<{ title?: string; snippet?: string; pageid?: number }> };
    };
    const host = new URL(stream.endpointUrl).origin;
    const hits = (data.query?.search ?? []).slice(0, 2);
    if (stream.providerKind === 'wikipedia' && hits.length > 0 && hits[0]?.title) {
      const topTitle = hits[0].title;
      try {
        const sumRes = await fetchWithTimeout(
          `${host}/api/rest_v1/page/summary/${encodeURIComponent(topTitle.replace(/\s+/g, '_'))}`,
          fetchImpl,
          signal,
        );
        if (sumRes.ok) {
          const sum = (await sumRes.json()) as {
            title?: string;
            extract?: string;
            description?: string;
            content_urls?: { desktop?: { page?: string } };
          };
          if (sum.extract && sum.extract.trim().length >= 16) {
            return [
              {
                title: sum.title ?? topTitle,
                sourceUrl:
                  sum.content_urls?.desktop?.page ??
                  `${host}/wiki/${encodeURIComponent(topTitle.replace(/\s+/g, '_'))}`,
                rawText: sum.extract,
                attributes: sum.description ? [sum.description] : undefined,
              },
            ];
          }
        }
      } catch {
        // Fall back seamlessly to search snippets below
      }
    }
    return hits.map((hit) => ({
      title: hit.title ?? topic,
      sourceUrl: `${host}/wiki/${encodeURIComponent((hit.title ?? topic).replace(/\s+/g, '_'))}`,
      rawText: `${hit.title ?? ''}: ${hit.snippet ?? ''}`,
    }));
  }

  if (stream.providerKind === 'wikidata') {
    const url = `${stream.endpointUrl}?action=wbsearchentities&search=${q}&language=en&limit=3&format=json&origin=*`;
    const res = await fetchWithTimeout(url, fetchImpl, signal);
    if (!res.ok) return [];
    const data = (await res.json()) as {
      search?: Array<{ id?: string; label?: string; description?: string; concepturi?: string }>;
    };
    return (data.search ?? []).slice(0, 3).map((hit) => ({
      title: hit.label ?? hit.id ?? topic,
      sourceUrl: hit.concepturi ?? `https://www.wikidata.org/wiki/${hit.id ?? ''}`,
      rawText: `${hit.label ?? ''} (${hit.id ?? ''}): ${hit.description ?? ''}`,
      attributes: hit.description ? [hit.description] : undefined,
    }));
  }

  if (stream.providerKind === 'openalex') {
    const url = `${stream.endpointUrl}?search=${q}&per-page=3`;
    const res = await fetchWithTimeout(url, fetchImpl, signal);
    if (!res.ok) return [];
    const data = (await res.json()) as {
      results?: Array<{
        title?: string;
        publication_year?: number;
        cited_by_count?: number;
        doi?: string;
        id?: string;
        authorships?: Array<{ author?: { display_name?: string } }>;
      }>;
    };
    return (data.results ?? []).slice(0, 3).map((w) => {
      const authors = (w.authorships ?? [])
        .slice(0, 3)
        .map((a) => a.author?.display_name)
        .filter(Boolean)
        .join(', ');
      return {
        title: w.title ?? topic,
        sourceUrl: w.doi ?? w.id ?? stream.endpointUrl,
        rawText: `${w.title ?? 'Work'} (${w.publication_year ?? 'n.d.'}${authors ? ` by ${authors}` : ''}, ${w.cited_by_count ?? 0} citations).`,
        attributes: w.publication_year ? [`published ${w.publication_year}`] : undefined,
      };
    });
  }

  if (stream.providerKind === 'arxiv') {
    const url = `${stream.endpointUrl}?search_query=all:${q}&start=0&max_results=3`;
    const res = await fetchWithTimeout(url, fetchImpl, signal);
    if (!res.ok) return [];
    return parseRssOrAtomXml(await res.text(), url);
  }

  if (stream.providerKind === 'crossref') {
    const url = `${stream.endpointUrl}?query=${q}&rows=3`;
    const res = await fetchWithTimeout(url, fetchImpl, signal);
    if (!res.ok) return [];
    const data = (await res.json()) as {
      message?: { items?: Array<{ title?: string[]; URL?: string; DOI?: string; 'container-title'?: string[] }> };
    };
    return (data.message?.items ?? []).slice(0, 3).map((it) => ({
      title: it.title?.[0] ?? topic,
      sourceUrl: it.URL ?? (it.DOI ? `https://doi.org/${it.DOI}` : stream.endpointUrl),
      rawText: `${it.title?.[0] ?? 'Publication'}${it['container-title']?.[0] ? ` — ${it['container-title'][0]}` : ''}.`,
    }));
  }

  if (stream.providerKind === 'github') {
    const isDirectRepo = /api\.github\.com\/repos\/[^/]+\/[^/]+/i.test(stream.endpointUrl);
    const url = isDirectRepo
      ? stream.endpointUrl
      : `${stream.endpointUrl}?q=${q}&sort=stars&per_page=3`;
    const res = await fetchWithTimeout(url, fetchImpl, signal);
    if (!res.ok) return [];
    if (isDirectRepo) {
      const repo = (await res.json()) as {
        full_name?: string;
        html_url?: string;
        description?: string;
        language?: string;
        stargazers_count?: number;
        topics?: string[];
      };
      return [
        {
          title: repo.full_name ?? stream.name,
          sourceUrl: repo.html_url ?? stream.endpointUrl,
          rawText: `${repo.full_name ?? stream.name}: ${repo.description ?? ''} (${repo.language ?? 'multi-language'}, ${repo.stargazers_count ?? 0} stars).`,
          attributes: (repo.topics ?? []).slice(0, 5),
        },
      ];
    }
    const data = (await res.json()) as {
      items?: Array<{
        full_name?: string;
        html_url?: string;
        description?: string;
        language?: string;
        stargazers_count?: number;
      }>;
    };
    return (data.items ?? []).slice(0, 3).map((r) => ({
      title: r.full_name ?? topic,
      sourceUrl: r.html_url ?? stream.endpointUrl,
      rawText: `${r.full_name ?? ''}: ${r.description ?? ''} (${r.language ?? 'code'}, ${r.stargazers_count ?? 0} stars).`,
    }));
  }

  if (stream.providerKind === 'npm') {
    const pkgSlug = topic.toLowerCase().replace(/[^a-z0-9@/_-]+/g, '-').replace(/^-+|-+$/g, '');
    const url = `${stream.endpointUrl}/-/v1/search?text=${encodeURIComponent(pkgSlug || topic)}&size=3`;
    const res = await fetchWithTimeout(url, fetchImpl, signal);
    if (!res.ok) return [];
    const data = (await res.json()) as {
      objects?: Array<{ package?: { name?: string; version?: string; description?: string; links?: { npm?: string } } }>;
    };
    return (data.objects ?? []).slice(0, 3).map((o) => ({
      title: `${o.package?.name ?? topic}@${o.package?.version ?? 'latest'}`,
      sourceUrl: o.package?.links?.npm ?? `https://www.npmjs.com/package/${o.package?.name ?? ''}`,
      rawText: `${o.package?.name ?? ''} v${o.package?.version ?? ''}: ${o.package?.description ?? ''}`,
    }));
  }

  if (stream.providerKind === 'crates') {
    const url = `${stream.endpointUrl}?q=${q}&per_page=3`;
    const res = await fetchWithTimeout(url, fetchImpl, signal);
    if (!res.ok) return [];
    const data = (await res.json()) as {
      crates?: Array<{ name?: string; max_version?: string; description?: string }>;
    };
    return (data.crates ?? []).slice(0, 3).map((c) => ({
      title: `${c.name ?? topic} ${c.max_version ?? ''}`.trim(),
      sourceUrl: `https://crates.io/crates/${c.name ?? ''}`,
      rawText: `${c.name ?? ''} v${c.max_version ?? ''}: ${c.description ?? ''}`,
    }));
  }

  if (stream.providerKind === 'mdn') {
    const url = `${stream.endpointUrl}?q=${q}&size=3`;
    const res = await fetchWithTimeout(url, fetchImpl, signal);
    if (!res.ok) return [];
    const data = (await res.json()) as {
      documents?: Array<{ title?: string; summary?: string; mdn_url?: string }>;
    };
    return (data.documents ?? []).slice(0, 3).map((d) => ({
      title: d.title ?? topic,
      sourceUrl: d.mdn_url ? `https://developer.mozilla.org${d.mdn_url}` : stream.endpointUrl,
      rawText: `${d.title ?? ''}: ${d.summary ?? ''}`,
    }));
  }

  if (stream.providerKind === 'open-meteo') {
    const locMatch = topic.match(
      /\b(?:in|for|at|near)\s+([a-z][a-z\s,.-]{1,48}?)(?:\s+(?:today|tomorrow|right\s+now|now|this\s+week|tonight|currently)\b|[?!.]|$)/i,
    );
    const cleanLoc = (
      locMatch?.[1] ??
      topic.replace(/\b(weather|forecast|temperature|rain|snow|wind|humidity|current|live|today|tomorrow|right\s+now|now|the)\b/gi, ' ')
    )
      .replace(/\s+/g, ' ')
      .trim();
    const geoQuery = encodeURIComponent(cleanLoc || topic);
    const geoRes = await fetchWithTimeout(
      `https://geocoding-api.open-meteo.com/v1/search?name=${geoQuery}&count=1&language=en&format=json`,
      fetchImpl,
      signal,
    );
    if (!geoRes.ok) return [];
    const geo = (await geoRes.json()) as {
      results?: Array<{ name?: string; country?: string; latitude?: number; longitude?: number }>;
    };
    const loc = geo.results?.[0];
    if (!loc || typeof loc.latitude !== 'number' || typeof loc.longitude !== 'number') return [];
    const wxUrl = `${stream.endpointUrl}?latitude=${loc.latitude}&longitude=${loc.longitude}&current=temperature_2m,relative_humidity_2m,wind_speed_10m`;
    const wxRes = await fetchWithTimeout(wxUrl, fetchImpl, signal);
    if (!wxRes.ok) return [];
    const wx = (await wxRes.json()) as {
      current?: { temperature_2m?: number; relative_humidity_2m?: number; wind_speed_10m?: number };
    };
    const cur = wx.current;
    if (!cur) return [];
    return [
      {
        title: `Live weather in ${loc.name ?? topic}${loc.country ? `, ${loc.country}` : ''}`,
        sourceUrl: 'https://open-meteo.com/',
        rawText: `Current conditions in ${loc.name ?? topic}: ${cur.temperature_2m ?? '?'}°C, humidity ${cur.relative_humidity_2m ?? '?'}%, wind ${cur.wind_speed_10m ?? '?'} km/h.`,
      },
    ];
  }

  if (stream.providerKind === 'osm') {
    const url = `${stream.endpointUrl}?q=${q}&format=json&limit=2`;
    const res = await fetchWithTimeout(url, fetchImpl, signal);
    if (!res.ok) return [];
    const places = (await res.json()) as Array<{ display_name?: string; lat?: string; lon?: string; type?: string }>;
    return (places ?? []).slice(0, 2).map((pl) => ({
      title: (pl.display_name ?? topic).split(',')[0] ?? topic,
      sourceUrl: `https://www.openstreetmap.org/?mlat=${pl.lat ?? 0}&mlon=${pl.lon ?? 0}`,
      rawText: `${pl.display_name ?? ''} (${pl.type ?? 'place'}, lat ${pl.lat ?? '?'}, lon ${pl.lon ?? '?'}).`,
    }));
  }

  // Custom Streams: RSS/Atom, JSON API, or public web URL
  const res = await fetchWithTimeout(stream.endpointUrl, fetchImpl, signal);
  if (!res.ok) return [];
  const text = await res.text();
  if (stream.providerKind === 'rss' || /<(?:rss|feed)\b/i.test(text.slice(0, 400))) {
    return parseRssOrAtomXml(text, stream.endpointUrl);
  }
  if (stream.providerKind === 'json-api') {
    try {
      const parsed = JSON.parse(text);
      const flat = JSON.stringify(parsed).slice(0, MAX_CHUNK_CHARS);
      return [{ title: stream.name, sourceUrl: stream.endpointUrl, rawText: flat }];
    } catch {
      // Fall through to plain text canonicalization
    }
  }
  const titleMatch = text.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
  const title = canonicalizeStreamText(titleMatch?.[1] ?? stream.name, 96);
  const bodyText = canonicalizeStreamText(text, MAX_CHUNK_CHARS);
  return bodyText.length >= 16 ? [{ title, sourceUrl: stream.endpointUrl, rawText: bodyText }] : [];
}

/**
 * Streams deduplicated realtime content ONLY from user-enabled Content Streams,
 * folds novel chunks into `UnifiedSemanticAssociator` + `LivingTrainingBasis`, and
 * returns a compact cited grounding block for tail KV injection (0 prefix wipe).
 */
export async function streamRealtimeWeightsToPrism(params: {
  readonly query: string;
  readonly catalogStore: ContentStreamCatalogStore;
  readonly registry: UnifiedExpertRegistry;
  readonly fetchFn?: typeof fetch;
  readonly signal?: AbortSignal;
  readonly maxStreams?: number;
  readonly nowMs?: number;
  /** Optional adaptive wall-clock deadline (ms) derived by ObserverChronology. */
  readonly timeoutMs?: number;
}): Promise<StreamRealtimeResult> {
  const enabled = params.catalogStore.getEnabledStreams();
  if (enabled.length === 0) {
    return {
      chunks: [],
      networkFetches: 0,
      freshnessCacheHits: 0,
      kappaDedupHits: 0,
      foldedNovelChunks: 0,
      promptContextBlock: null,
    };
  }

  const now = params.nowMs ?? Date.now();
  const topic = extractSearchTopic(params.query);
  const enabledIds = new Set(enabled.map((s) => s.id));

  // Stage 1 Dedup: Pre-fetch freshness TTL check in LivingTrainingBasis -> 0 network calls!
  const freshHits = params.registry.basis.findFreshStreamedKnowledge(topic, enabledIds, now, 3);
  if (freshHits.length > 0) {
    const chunks: StreamedKnowledgeChunk[] = freshHits.map((entry) => ({
      kappa: entry.kappa,
      conceptId: entry.conceptId ?? topic.toLowerCase(),
      streamId: entry.streamId ?? 'cached',
      catalogId: entry.catalogId ?? 'reference',
      provider: entry.provider ?? 'cached',
      title: entry.title,
      sourceUrl: entry.sourceUrl ?? '',
      canonicalText: entry.summary,
      attributes: entry.categories,
      fetchedAt: entry.fetchedAt ?? entry.updatedAt,
      ttlMs: entry.ttlMs ?? 6 * 3600 * 1000,
      deduplicatedHit: true,
    }));
    const lines = chunks.map(
      (c) => `- **${c.title}**${c.sourceUrl ? ` (${c.sourceUrl})` : ''}: ${c.canonicalText}`,
    );
    return {
      chunks,
      networkFetches: 0,
      freshnessCacheHits: chunks.length,
      kappaDedupHits: 0,
      foldedNovelChunks: 0,
      promptContextBlock: `[Live Grounded Content Streams]\n${lines.join('\n')}`,
    };
  }

  const selectedStreams = selectRelevantEnabledStreams(params.query, enabled, params.maxStreams ?? 2);
  const fetchImpl = params.fetchFn ?? fetch;
  const deadlineCtrl = new AbortController();
  const onParentAbort = () => deadlineCtrl.abort();
  if (params.signal?.aborted) {
    deadlineCtrl.abort();
  } else {
    params.signal?.addEventListener('abort', onParentAbort, { once: true });
  }
  const overallTimeoutMs =
    typeof params.timeoutMs === 'number' && params.timeoutMs >= 400
      ? params.timeoutMs
      : STREAM_FETCH_TIMEOUT_MS + 600;
  const deadlineTimer = setTimeout(() => deadlineCtrl.abort(), overallTimeoutMs);

  let networkFetches = 0;
  let kappaDedupHits = 0;
  let foldedNovelChunks = 0;
  const collected: StreamedKnowledgeChunk[] = [];

  try {
    await Promise.all(
      selectedStreams.map(async (stream) => {
        if (deadlineCtrl.signal.aborted) return;
        networkFetches += 1;
        try {
          const rawItems = await fetchFromStreamProvider(
            stream,
            topic,
            fetchImpl,
            deadlineCtrl.signal,
          );
        let addedForStream = 0;
        for (const item of rawItems) {
          const canonicalText = canonicalizeStreamText(item.rawText, MAX_CHUNK_CHARS);
          if (canonicalText.length < 12) continue;
          const kappa = computeKnowledgeKappa(canonicalText, item.sourceUrl);
          // Stage 2 Dedup: Exact content `kappa` check -> 0 duplicate delta updates!
          const alreadyKnown = params.registry.basis.hasKnowledgeKappa(kappa);
          if (alreadyKnown) {
            kappaDedupHits += 1;
            params.registry.basis.refreshKnowledgeTimestamp(kappa, now);
            collected.push({
              kappa,
              conceptId: topic.toLowerCase(),
              streamId: stream.id,
              catalogId: stream.catalogId,
              provider: stream.providerKind,
              title: item.title,
              sourceUrl: item.sourceUrl,
              canonicalText,
              attributes: item.attributes ?? [],
              fetchedAt: now,
              ttlMs: stream.freshnessTtlMs,
              deduplicatedHit: true,
            });
            continue;
          }

          // Novel content -> fold into UnifiedSemanticAssociator (verdantGatedDeltaStepF32) + LivingTrainingBasis!
          const { revision } = params.registry.semantics.writeSemanticObservation(
            item.title || topic,
            {
              canonicalSubject: item.title || topic,
              contextualBrief: canonicalText.slice(0, 240),
              attributes: [
                ...(item.attributes ?? []),
                ...stream.tags.slice(0, 3),
              ],
            },
            { gain: 0.55 },
          );
          params.registry.basis.recordStreamedKnowledge({
            kappa,
            streamId: stream.id,
            catalogId: stream.catalogId,
            provider: stream.providerKind,
            conceptId: revision.conceptId,
            title: item.title,
            sourceUrl: item.sourceUrl,
            summary: canonicalText,
            categories: [...stream.tags.slice(0, 4), stream.catalogId],
            fetchedAt: now,
            ttlMs: stream.freshnessTtlMs,
          });
          foldedNovelChunks += 1;
          addedForStream += 1;
          collected.push({
            kappa,
            conceptId: revision.conceptId,
            streamId: stream.id,
            catalogId: stream.catalogId,
            provider: stream.providerKind,
            title: item.title,
            sourceUrl: item.sourceUrl,
            canonicalText,
            attributes: item.attributes ?? [],
            fetchedAt: now,
            ttlMs: stream.freshnessTtlMs,
            deduplicatedHit: false,
          });
        }
        params.catalogStore.recordStreamSync(stream.id, addedForStream, null);
        } catch (err) {
          const msg = err instanceof Error ? err.message : 'Unable to reach stream';
          params.catalogStore.recordStreamSync(stream.id, 0, msg);
        }
      }),
    );
  } finally {
    clearTimeout(deadlineTimer);
    params.signal?.removeEventListener('abort', onParentAbort);
  }

  const bounded = collected.slice(0, 4);
  const promptContextBlock =
    bounded.length > 0
      ? `[Live Grounded Content Streams]\n${bounded
          .map((c) => `- **${c.title}**${c.sourceUrl ? ` (${c.sourceUrl})` : ''}: ${c.canonicalText.slice(0, 480)}`)
          .join('\n')}`
      : null;

  return {
    chunks: bounded,
    networkFetches,
    freshnessCacheHits: 0,
    kappaDedupHits,
    foldedNovelChunks,
    promptContextBlock,
  };
}
