import test from 'node:test';
import assert from 'node:assert/strict';
import {
  BUILTIN_CONTENT_STREAMS,
  ContentStreamCatalogStore,
  STREAM_CATALOGS,
  normalizeCustomStreamInput,
} from './contentStreamCatalog';

test('ContentStreamCatalogStore starts with ALL built-in streams disabled (strict opt-in) and exposes ZERO System Stream entries', () => {
  const store = new ContentStreamCatalogStore();
  const snap = store.getSnapshot();

  assert.equal(STREAM_CATALOGS.length, 5);
  assert.ok(BUILTIN_CONTENT_STREAMS.length >= 13);
  assert.equal(snap.enabledCount, 0, 'Every built-in Content Stream must start OFF by default');
  assert.equal(store.hasEnabledStreams(), false);

  for (const s of snap.streams) {
    assert.equal(s.enabled, false);
    assert.equal(/\b(gguf|qwen|huggingface\.co\/.*gguf|system\s*stream|sdxs|whisper|siglip)\b/i.test(`${s.name} ${s.endpointUrl}`), false);
  }
});

test('normalizeCustomStreamInput enforces public HTTPS, rejects private/local hosts, and auto-detects GitHub, RSS, JSON, and Web sources', () => {
  assert.equal(normalizeCustomStreamInput({ url: '' }).ok, false);
  assert.equal(normalizeCustomStreamInput({ url: 'http://example.org/feed.xml' }).ok, false);
  assert.equal(normalizeCustomStreamInput({ url: 'https://localhost:8080/api' }).ok, false);
  assert.equal(normalizeCustomStreamInput({ url: 'https://192.168.1.5/feed.xml' }).ok, false);
  assert.equal(normalizeCustomStreamInput({ url: 'https://10.0.0.1/data.json' }).ok, false);

  const gh = normalizeCustomStreamInput({ url: 'https://github.com/afflom/VerdantUI' });
  assert.equal(gh.ok, true);
  if (gh.ok) {
    assert.equal(gh.spec.providerKind, 'github');
    assert.equal(gh.spec.endpointUrl, 'https://api.github.com/repos/afflom/VerdantUI');
    assert.equal(gh.spec.name, 'afflom/VerdantUI');
    assert.equal(gh.spec.enabled, true);
  }

  const rss = normalizeCustomStreamInput({ url: 'https://blog.rust-lang.org/feed.xml' });
  assert.equal(rss.ok, true);
  if (rss.ok) {
    assert.equal(rss.spec.providerKind, 'rss');
  }

  const json = normalizeCustomStreamInput({ url: 'https://api.example.org/v1/data.json', name: 'Climate Data' });
  assert.equal(json.ok, true);
  if (json.ok) {
    assert.equal(json.spec.providerKind, 'json-api');
    assert.equal(json.spec.name, 'Climate Data');
  }
});

test('ContentStreamCatalogStore supports toggling streams/catalogs, custom stream lifecycle, deduplication, and search/filtering', () => {
  const store = new ContentStreamCatalogStore();
  let notifications = 0;
  const unsub = store.subscribe(() => {
    notifications += 1;
  });

  // Toggle individual built-in stream
  assert.equal(store.toggleStream('stream:wikipedia', true), true);
  assert.equal(store.getSnapshot().enabledCount, 1);
  assert.equal(store.hasEnabledStreams(), true);

  // Toggle entire catalog
  const sciChanged = store.toggleCatalog('science', true);
  assert.ok(sciChanged >= 3);
  assert.equal(store.getSnapshot().enabledCount, 1 + sciChanged);

  // Search and active filter
  const activeGroups = store.searchCatalogs('', 'active');
  assert.ok(activeGroups.some((g) => g.catalog.id === 'reference' && g.streams.length === 1));
  assert.ok(activeGroups.some((g) => g.catalog.id === 'science' && g.streams.length === sciChanged));

  const wikiSearch = store.searchCatalogs('encyclopedia', 'all');
  assert.ok(wikiSearch.some((g) => g.streams.some((s) => s.id === 'stream:wikipedia')));

  // Add custom stream + duplicate URL deduplication
  const added1 = store.addCustomStream({ url: 'https://feeds.npr.org/1001/rss.xml', name: 'NPR News' });
  assert.equal(added1.ok, true);
  if (!added1.ok) return;
  const added2 = store.addCustomStream({ url: 'https://feeds.npr.org/1001/rss.xml', name: 'Duplicate NPR' });
  assert.equal(added2.ok, true);
  if (added2.ok) {
    assert.equal(added2.stream.id, added1.stream.id, 'Duplicate endpoint URL must reuse existing stream');
  }
  assert.equal(store.getSnapshot().customCount, 1);

  // Built-in streams cannot be removed; custom streams can
  assert.equal(store.removeCustomStream('stream:wikipedia'), false);
  assert.equal(store.removeCustomStream(added1.stream.id), true);
  assert.equal(store.getSnapshot().customCount, 0);
  assert.ok(notifications >= 4);
  unsub();
});
