import test from 'node:test';
import assert from 'node:assert/strict';
import { ContentStreamCatalogStore } from './contentStreamCatalog';
import {
  canonicalizeStreamText,
  computeKnowledgeKappa,
  parseRssOrAtomXml,
  streamRealtimeWeightsToPrism,
} from './realtimeWeightStreamer';
import { UnifiedExpertRegistry } from './unifiedModelBasis';

function makeJsonResponse(body: unknown, contentType = 'application/json'): Response {
  return new Response(typeof body === 'string' ? body : JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': contentType },
  });
}

test('streamRealtimeWeightsToPrism performs ZERO network fetches when no Content Streams are enabled', async () => {
  const catalogStore = new ContentStreamCatalogStore();
  const registry = new UnifiedExpertRegistry();
  let calls = 0;

  const res = await streamRealtimeWeightsToPrism({
    query: 'Explain latest quantum error correction research',
    catalogStore,
    registry,
    fetchFn: async () => {
      calls += 1;
      return makeJsonResponse({});
    },
  });

  assert.equal(calls, 0);
  assert.equal(res.networkFetches, 0);
  assert.equal(res.chunks.length, 0);
  assert.equal(res.promptContextBlock, null);
});

test('streamRealtimeWeightsToPrism folds novel live chunks into UnifiedSemanticAssociator & LivingTrainingBasis with 2-stage TTL + kappa dedup', async () => {
  const catalogStore = new ContentStreamCatalogStore();
  catalogStore.toggleStream('stream:wikipedia', true);
  const registry = new UnifiedExpertRegistry();

  let fetchCalls = 0;
  const mockFetch: typeof fetch = async (input) => {
    fetchCalls += 1;
    const url = String(input);
    if (url.includes('action=query') && url.includes('list=search')) {
      return makeJsonResponse({
        query: {
          search: [{ title: 'James Webb Space Telescope' }],
        },
      });
    }
    if (url.includes('/api/rest_v1/page/summary/')) {
      return makeJsonResponse({
        title: 'James Webb Space Telescope',
        extract:
          'The James Webb Space Telescope is a space telescope designed to conduct infrared astronomy and observe high-redshift galaxies.',
        content_urls: {
          desktop: { page: 'https://en.wikipedia.org/wiki/James_Webb_Space_Telescope' },
        },
      });
    }
    return new Response('Not found', { status: 404 });
  };

  const t0 = 1_700_000_000_000;
  // Pass 1: Cold miss -> fetches from Wikipedia, folds into UnifiedSemanticAssociator & LivingTrainingBasis
  const first = await streamRealtimeWeightsToPrism({
    query: 'What is the James Webb Space Telescope?',
    catalogStore,
    registry,
    fetchFn: mockFetch,
    nowMs: t0,
  });

  assert.ok(fetchCalls >= 1);
  assert.equal(first.networkFetches, 1);
  assert.equal(first.foldedNovelChunks, 1);
  assert.equal(first.kappaDedupHits, 0);
  assert.ok(first.promptContextBlock?.includes('James Webb Space Telescope'));
  assert.ok(registry.basis.hasKnowledgeKappa(first.chunks[0]!.kappa));
  assert.ok(registry.semantics.listConcepts().length >= 1);

  // Pass 2: Same query within freshness TTL -> Stage-1 pre-fetch cache hit with ZERO network calls!
  const callsBeforePass2 = fetchCalls;
  const second = await streamRealtimeWeightsToPrism({
    query: 'Tell me about the James Webb Space Telescope',
    catalogStore,
    registry,
    fetchFn: mockFetch,
    nowMs: t0 + 60_000,
  });
  assert.equal(fetchCalls, callsBeforePass2, 'Fresh TTL hit must incur 0 network calls');
  assert.equal(second.networkFetches, 0);
  assert.equal(second.freshnessCacheHits, 1);
  assert.equal(second.foldedNovelChunks, 0);
  assert.ok(second.promptContextBlock?.includes('James Webb Space Telescope'));

  // Pass 3: After TTL expiry, same unchanged Wikipedia content -> Stage-2 sha256 kappa dedup -> 0 duplicate delta updates!
  const conceptCountBeforePass3 = registry.semantics.listConcepts().length;
  const third = await streamRealtimeWeightsToPrism({
    query: 'What is the James Webb Space Telescope?',
    catalogStore,
    registry,
    fetchFn: mockFetch,
    nowMs: t0 + 48 * 60 * 60 * 1000,
  });
  assert.equal(third.networkFetches, 1);
  assert.equal(third.kappaDedupHits, 1);
  assert.equal(third.foldedNovelChunks, 0, 'Identical post-fetch kappa must NOT re-fold duplicate weights');
  assert.equal(registry.semantics.listConcepts().length, conceptCountBeforePass3);
});

test('parseRssOrAtomXml, canonicalizeStreamText, and custom RSS stream ingestion work cleanly offline with mock fetch', async () => {
  const sampleXml = `<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
      <item>
        <title>Verdant WASM SIMD Release</title>
        <link>https://example.org/posts/wasm-simd</link>
        <description><![CDATA[<p>Multi-threaded <b>WASM SIMD</b> kernels ship in browsers.</p><script>alert(1)</script>]]></description>
      </item>
    </channel></rss>`;

  const parsed = parseRssOrAtomXml(sampleXml, 'https://example.org/feed.xml');
  assert.equal(parsed.length, 1);
  assert.equal(parsed[0]!.title, 'Verdant WASM SIMD Release');
  const clean = canonicalizeStreamText(parsed[0]!.rawText);
  assert.equal(clean.includes('alert'), false);
  assert.ok(clean.includes('Multi-threaded WASM SIMD kernels'));
  assert.equal(
    computeKnowledgeKappa(clean, parsed[0]!.sourceUrl),
    computeKnowledgeKappa(clean, parsed[0]!.sourceUrl),
  );

  const catalogStore = new ContentStreamCatalogStore();
  const added = catalogStore.addCustomStream({
    name: 'Example Engineering Blog',
    url: 'https://example.org/feed.xml',
  });
  assert.equal(added.ok, true);
  const registry = new UnifiedExpertRegistry();

  const res = await streamRealtimeWeightsToPrism({
    query: 'Latest WASM SIMD updates from example.org',
    catalogStore,
    registry,
    fetchFn: async () => makeJsonResponse(sampleXml, 'application/xml'),
  });
  assert.equal(res.foldedNovelChunks, 1);
  assert.ok(res.promptContextBlock?.includes('Verdant WASM SIMD Release'));
});

test('Open-Meteo weather stream extracts clean city name from conversational queries and Wikipedia prefers rich REST summaries', async () => {
  const catalogStore = new ContentStreamCatalogStore();
  catalogStore.toggleStream('stream:open-meteo', true);
  const registry = new UnifiedExpertRegistry();

  const requestedUrls: string[] = [];
  const res = await streamRealtimeWeightsToPrism({
    query: 'What is the weather in Tokyo right now?',
    catalogStore,
    registry,
    fetchFn: async (input) => {
      const url = String(input);
      requestedUrls.push(url);
      if (url.includes('geocoding-api.open-meteo.com')) {
        assert.ok(url.includes('name=Tokyo'), `Expected clean city name=Tokyo, got ${url}`);
        return makeJsonResponse({
          results: [{ name: 'Tokyo', country: 'Japan', latitude: 35.6895, longitude: 139.6917 }],
        });
      }
      if (url.includes('api.open-meteo.com')) {
        return makeJsonResponse({
          current: { temperature_2m: 18.4, relative_humidity_2m: 54, wind_speed_10m: 11.2 },
        });
      }
      return new Response('Not found', { status: 404 });
    },
  });

  assert.equal(requestedUrls.length, 2);
  assert.equal(res.foldedNovelChunks, 1);
  assert.ok(res.promptContextBlock?.includes('Tokyo, Japan'));
  assert.ok(res.promptContextBlock?.includes('18.4°C'));
});
