import assert from 'node:assert/strict';
import test from 'node:test';
import {
  ContainerGrantMatrix,
  VerdantCognitiveObserver,
  createVerdantClientTools,
  detectExplicitPrismAddress,
} from './cognitiveObserver';
import {
  ObserverStreamBus,
  formatTraceparent,
} from './observerStreamBus';
import { PrismHostPool } from './prismHostPool';
import {
  VERDANT_SANDBOX_CSP,
  buildPrismSandboxDocument,
} from './prismSandboxHtml';
import {
  PrismStore,
  buildSnapshot,
  canonicalizeJson,
  computePrismDigest,
  createStarterSnapshot,
  extractPrismPatchFromModelOutput,
  parseHoloArchive,
  prismToolName,
  serializeHoloArchive,
  sha256HexSync,
} from './prismStudio';

test('ObserverStreamBus propagates W3C Trace Context across causal spans, exports OTLP/JSON, and routes endpoint RPC', async () => {
  const bus = new ObserverStreamBus();
  const observedEvents: string[] = [];
  const unsub = bus.subscribe((ev) => observedEvents.push(ev.name));

  const root = bus.startSpan('thread.turn', {
    endpointKind: 'thread',
    endpointId: 'thread-1',
    kind: 'SERVER',
  });

  assert.equal(root.context.traceId.length, 32);
  assert.equal(root.context.spanId.length, 16);
  assert.match(formatTraceparent(root.context), /^00-[0-9a-f]{32}-[0-9a-f]{16}-01$/);

  let receivedTraceId = '';
  let receivedParentSpanId: string | undefined;
  const unreg = bus.registerEndpoint({
    endpointId: 'prism-calc',
    endpointKind: 'prism',
    dispatch: async (payload, ctx) => {
      receivedTraceId = ctx.traceId;
      receivedParentSpanId = ctx.parentSpanId;
      return { echoed: payload, ok: true };
    },
  });

  const { result, span } = await bus.dispatchToEndpoint(
    'prism-calc',
    { action: 'ping' },
    { parentContext: root.context },
  );
  root.end('ok');

  assert.deepEqual(result, { echoed: { action: 'ping' }, ok: true });
  assert.equal(receivedTraceId, root.context.traceId);
  assert.equal(receivedParentSpanId, root.context.spanId);
  assert.equal(span.status, 'ok');
  assert.ok(observedEvents.includes('endpoint.response'));

  const otlp = bus.toOtlpJson();
  assert.equal(otlp.resourceSpans.length, 1);
  assert.equal(otlp.resourceLogs.length, 1);

  unreg();
  unsub();
});

test('Canonical SHA-256 .holo archive round-trips deterministically and rejects tampered payloads', () => {
  // Known FIPS 180-4 SHA-256 test vector for "abc"
  assert.equal(
    sha256HexSync('abc'),
    'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
  );
  assert.equal(canonicalizeJson({ b: 2, a: 1 }), '{"a":1,"b":2}');

  const starter = createStarterSnapshot('kanban');
  const serialized = serializeHoloArchive(starter);
  const parsed = parseHoloArchive(serialized);

  assert.equal(parsed.digest, starter.digest);
  assert.equal(parsed.name, 'Task Board');
  assert.equal(parsed.app.js, starter.app.js);

  // Tamper with a single character in app HTML while keeping original digest
  const tampered = serialized.replace('Task Board', 'Tampered Board');
  assert.throws(() => parseHoloArchive(tampered), /Integrity check failed/);

  // Detect external binary HOLO magic cleanly
  const binaryMagic = new Uint8Array([0x48, 0x4f, 0x4c, 0x4f, 0x00, 0x04]);
  assert.throws(() => parseHoloArchive(binaryMagic), /Binary \.holo container detected/);
});

test('PrismStore isolates draft vs immutable published snapshot, supports Undo, and isolates per-Prism KV', async () => {
  const store = new PrismStore();
  const created = await store.create('focus', 'Deep Focus');
  const initialDigest = created.draft.digest;
  assert.equal(created.published, null);

  // Publish freezes snapshot
  const published = await store.publish(created.id);
  assert.ok(published?.published);
  const publishedDigest = published!.published!.digest;
  assert.equal(publishedDigest, initialDigest);

  // Editing draft MUST NOT mutate the published snapshot
  const edited = await store.updateDraft(created.id, (prev) => ({
    ...prev,
    app: { ...prev.app, css: `${prev.app.css}\nbody { opacity: 0.99; }` },
  }));
  assert.ok(edited);
  assert.notEqual(edited!.draft.digest, publishedDigest);
  assert.equal(edited!.published!.digest, publishedDigest);
  assert.equal(edited!.revisions.length, 1);

  // Undo restores previous draft cleanly
  const undone = await store.undoDraft(created.id);
  assert.ok(undone);
  assert.equal(undone!.draft.digest, initialDigest);
  assert.equal(undone!.revisions.length, 0);

  // Scoped per-Prism KV isolation
  const second = await store.create('budget', 'Monthly Budget');
  await store.setKv(created.id, 'sessions', 4);
  await store.setKv(second.id, 'sessions', 1);
  assert.equal(await store.getKv(created.id, 'sessions'), 4);
  assert.equal(await store.getKv(second.id, 'sessions'), 1);
});

test('extractPrismPatchFromModelOutput applies targeted CSS/JS patches without clobbering untouched sections', () => {
  const base = createStarterSnapshot('blank');
  const modelOutput = [
    'Updated the card border and renamed the Prism.',
    '```css',
    '.empty-card { border-width: 2px; }',
    '```',
    '```json',
    '{"name": "Scratchpad"}',
    '```',
  ].join('\n');

  const patch = extractPrismPatchFromModelOutput(modelOutput, base);
  assert.ok(patch);
  assert.deepEqual([...patch!.updatedSections].sort(), ['css', 'meta']);
  assert.equal(patch!.snapshot.name, 'Scratchpad');
  assert.equal(patch!.snapshot.app.css, '.empty-card { border-width: 2px; }');
  // Untouched HTML & JS remain byte-identical
  assert.equal(patch!.snapshot.app.html, base.app.html);
  assert.equal(patch!.snapshot.app.js, base.app.js);
  assert.equal(patch!.snapshot.digest, computePrismDigest(patch!.snapshot));
});

test('Sandbox HTML enforces strict CSP, hot-CSS hook, browser/ui bridge, and FK-07 capability grants', async () => {
  const snap = buildSnapshot({
    ...createStarterSnapshot('writer'),
    grants: { observer: true, storage: true, browser: true, network: false },
  });
  const html = buildPrismSandboxDocument({
    prismId: 'prism-sec',
    channelId: 'chan-1',
    snapshot: snap,
    darkMode: true,
  });

  assert.ok(html.includes(VERDANT_SANDBOX_CSP));
  assert.ok(html.includes("connect-src 'none'"));
  assert.ok(html.includes('id="verdant-user-css"'));
  assert.ok(html.includes("msg.type === 'hot-css'"));
  assert.ok(html.includes('browser: Object.freeze'));
  assert.ok(html.includes('ui: Object.freeze'));
  assert.ok(html.includes('class="dark"'));

  const bus = new ObserverStreamBus();
  const store = new PrismStore();
  const grants = new ContainerGrantMatrix();
  let notifiedTitle = '';
  const pool = new PrismHostPool(bus, store, grants, null, {
    onNotify: ({ title }) => {
      notifiedTitle = title;
    },
  });

  // Network grant is false by default -> refused before fetch
  await assert.rejects(
    () => pool.proxiedFetch('prism-sec', snap.grants, 'https://example.com/data.json'),
    /Network grant is Off/,
  );

  // Browser notify succeeds when browser grant is enabled, refused when disabled
  await pool.notify('prism-sec', snap.grants, 'Saved', 'Export complete');
  assert.equal(notifiedTitle, 'Saved');
  await assert.rejects(
    () =>
      pool.copyText(
        'prism-sec',
        { ...snap.grants, browser: false },
        'secret',
      ),
    /Browser grant is Off/,
  );

  // Prism-to-Prism composition & self-recursion guard
  const p1 = await store.create('kanban', 'Task Board');
  const p2 = await store.create('budget', 'Expense Tracker');
  await store.publish(p1.id);
  await store.publish(p2.id);

  const composed = (await pool.callPrism(
    p1.id,
    p1.draft.grants,
    'Expense Tracker',
    { action: 'summary' },
  )) as Record<string, unknown>;
  assert.equal(composed.ok, true);
  assert.equal(composed.name, 'Expense Tracker');

  await assert.rejects(
    () => pool.callPrism(p1.id, p1.draft.grants, 'Task Board', {}),
    /Self-recursive Prism call refused/,
  );

  // Explicit FK-07 denial on prism-storage overrides Prism's own storage=true grant
  grants.deny('prism-storage');
  await assert.rejects(
    () => pool.getKv('prism-sec', snap.grants, 'k'),
    /Storage grant denied/,
  );
  pool.dispose();
});

test('Published Prisms register JSON Schema 2020-12 tools and route predictably via @Prism without hijacking normal chat', async () => {
  const store = new PrismStore();
  const bus = new ObserverStreamBus();
  const observer = new VerdantCognitiveObserver();
  const pool = new PrismHostPool(bus, store, observer.grants, null);

  const proj = await store.create('kanban', 'Task Board');
  await store.publish(proj.id);
  const published = store.listPublishedSummaries();
  assert.equal(published.length, 1);

  // Normal conversation mentioning "task board" MUST NOT be hijacked
  assert.equal(
    detectExplicitPrismAddress('Can you explain how a kanban task board works?', published),
    null,
  );

  // Explicit @Task Board routes deterministically at LOD0 even with Model Stream Off
  const route = observer.selectRoute({
    threadId: 'thread-a',
    messages: [],
    latestUserText: '@Task Board add Ship v2',
    hasAttachment: false,
    activeModel: null,
    cachedModelIds: [],
    isToolContinuationStep: false,
    hasImportedHarness: false,
    publishedPrisms: published,
  });
  assert.equal(route.certificate.kind, 'lod0-verified');
  assert.equal(route.certificate.directMatch?.prismId, proj.id);
  assert.equal(route.certificate.directMatch?.toolName, prismToolName(published[0]));

  // Dynamic tool registration uses JSON Schema 2020-12 and dispatches cleanly through PrismHostPool
  const tools = createVerdantClientTools(
    () => ({
      model: null,
      telemetry: {} as never,
      opfsStorage: { supported: false, usageBytes: 0, quotaBytes: 0, storedModels: [] },
      lastTokensPerSec: 0,
    }),
    {
      prisms: published,
      dispatchPrism: (id, payload, signal) => pool.dispatchPublishedPrism(id, payload, { signal }),
    },
  );

  const toolKey = prismToolName(published[0]);
  assert.ok(tools[toolKey]);
  assert.equal((tools[toolKey].parameters as { type?: string }).type, 'object');

  const execResult = (await tools[toolKey].execute!(
    { action: 'list' },
    { toolCallId: 'tc_1', abortSignal: new AbortController().signal, human: async () => undefined } as never,
  )) as Record<string, unknown>;
  assert.equal(execResult.__prismInvocation, true);
  assert.equal(execResult.prismId, proj.id);
  assert.equal(execResult.digest, published[0].digest);

  pool.dispose();
});

test('Reliability hardening: unclosed code-fence salvage, KV cleanup on delete, indirect cycle guard, and aborted ask bailout', async () => {
  // 1. Salvage truncated/unclosed trailing code fence + CRLF line endings
  const base = createStarterSnapshot('blank');
  const truncatedModelOutput =
    'Sure, adding dark card styling:\r\n```css\r\n.empty-card { border-radius: 18px; padding: 32px; }';
  const salvaged = extractPrismPatchFromModelOutput(truncatedModelOutput, base);
  assert.ok(salvaged);
  assert.deepEqual(salvaged!.updatedSections, ['css']);
  assert.ok(salvaged!.snapshot.app.css.includes('border-radius: 18px'));

  // 2. Deleting a Prism purges its scoped KV entries with zero leaks
  const store = new PrismStore();
  const bus = new ObserverStreamBus();
  const grants = new ContainerGrantMatrix();
  const pool = new PrismHostPool(bus, store, grants, null);

  const a = await store.create('kanban', 'Board A');
  const b = await store.create('budget', 'Budget B');
  await store.publish(a.id);
  await store.publish(b.id);
  await store.setKv(a.id, 'tasks', ['t1']);
  await store.setKv(b.id, 'expenses', [10]);

  await store.delete(a.id);
  assert.equal(await store.getKv(a.id, 'tasks'), null);
  assert.deepEqual(await store.getKv(b.id, 'expenses'), [10]);

  // 3. Indirect cycle detection (B -> C -> B) via callChain
  const c = await store.create('focus', 'Timer C');
  await store.publish(c.id);
  await assert.rejects(
    () => pool.callPrism(c.id, c.draft.grants, 'Budget B', {}, undefined, [b.id, c.id]),
    /Cyclic Prism-to-Prism call refused/,
  );

  // 4. Aborted verdant.ask() bails immediately without invoking worker
  const aborted = new AbortController();
  aborted.abort();
  await assert.rejects(
    () => pool.ask({ prismId: c.id, prompt: 'hello', signal: aborted.signal }),
    /cancelled/i,
  );

  pool.dispose();
});
