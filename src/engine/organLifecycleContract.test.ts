import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { VerdantHoloHost } from '../host/verdantHoloHost';
import { ObserverStreamBus, formatTraceparent, isValidW3cTraceparent } from './observerStreamBus';
import { ORGAN_REGISTRY, lookupOrganDescriptor, formatOrganStatusPayload } from './organRegistry';
import { VerdantCognitiveObserver } from './cognitiveObserver';
import { PrismStore } from './prismStudio';
import { PrismHostPool } from './prismHostPool';
import { ContentGalleryStore } from './contentGalleryStore';
import { UnifiedExpertRegistry } from './unifiedModelBasis';
import { StudioChatBridge } from './studioChatBridge';
import { normalizeOrganWeights, synthesizeProceduralOrganView } from './prismMediaOrgans';

test('VerdantHoloHost loads, verifies, traces via OTEL, and dispatches all 7 Lean-proved domains in verdant-ui.holo', async () => {
  const bus = new ObserverStreamBus();
  const host = new VerdantHoloHost(bus);
  const holoBytes = new Uint8Array(
    readFileSync(join(process.cwd(), 'app/.prism/build/verdant-ui.holo')),
  );

  const snap = await host.loadFromBytes(holoBytes);
  assert.equal(snap.loaded, true);
  assert.equal(snap.verified, true);
  assert.ok(snap.vectorCount >= 24, `Expected >=24 acceptance vectors, got ${snap.vectorCount}`);
  assert.equal(snap.vectorsPassed, snap.vectorCount);

  // 1. Consent (paused -> accepted with loaded model NEVER locks)
  const cAccepted = await host.dispatchConsent('accepted', true, true, true);
  assert.equal(cAccepted.phase, 'ready');
  assert.equal(cAccepted.unlocked, true);
  assert.equal(cAccepted.refused, false);

  const cPaused = await host.dispatchConsent('paused', true, true, true);
  assert.equal(cPaused.phase, 'ready');
  assert.equal(cPaused.unlocked, true);

  const cPending = await host.dispatchConsent('pending', false, true, false);
  assert.equal(cPending.phase, 'consent');
  assert.equal(cPending.unlocked, false);

  const cBad = await host.dispatchConsent('bogus', true, true, true);
  assert.equal(cBad.refused, true);
  assert.equal(cBad.raw, 'refused: unknown consent state');

  // 2. Grants (FK-07 denial wins, offline denies network, stateful organs admit storage)
  const gStorage = await host.dispatchGrant('tool-organ', 'storage', true, false);
  assert.equal(gStorage.allowed, true);
  assert.equal(gStorage.refused, false);

  const gDenied = await host.dispatchGrant('tool-organ', 'storage', false, true);
  assert.equal(gDenied.allowed, false);
  assert.equal(gDenied.refused, true);
  assert.equal(gDenied.raw, 'refused: denied by FK-07');

  const gOfflineNet = await host.dispatchGrant('tool-organ', 'network', true, false);
  assert.equal(gOfflineNet.allowed, false);
  assert.equal(gOfflineNet.refused, true);
  assert.equal(gOfflineNet.raw, 'refused: offline mode denies network');

  // 3. Compartment visibility (personal visible in personal; private work hidden; unknown refused)
  const compOk = await host.dispatchCompartment('personal', 'personal', 'private');
  assert.equal(compOk.visible, true);
  assert.equal(compOk.refused, false);

  const compHidden = await host.dispatchCompartment('work', 'personal', 'private');
  assert.equal(compHidden.visible, false);
  assert.equal(compHidden.refused, false);

  const compBad = await host.dispatchCompartment('bogus', 'personal', 'private');
  assert.equal(compBad.refused, true);
  assert.equal(compBad.raw, 'refused: unknown compartment');

  // 4. Parts (every MediaType answers status; unknown part/verb refused)
  for (const desc of ORGAN_REGISTRY) {
    const pRes = await host.dispatchPart(desc.mediaType, 'status');
    assert.equal(pRes.ok, true, `Part ${desc.mediaType} status failed: ${pRes.raw}`);
    assert.equal(pRes.refused, false);
  }
  const pBad = await host.dispatchPart('nonexistent', 'status');
  assert.equal(pBad.refused, true);
  assert.equal(pBad.raw, 'refused: unknown part');

  // 5. GUI / Observer parity (an_agent_gets_the_tree_a_person_gets)
  const guiTree = await host.dispatchGuiTree();
  const obsTree = await host.dispatchObserverPerceive();
  assert.deepEqual(obsTree.parts, guiTree.parts);
  assert.ok(guiTree.parts.includes('home') && guiTree.parts.includes('gallery'));

  const obsMutate = await host.dispatchRaw('verdant\tobserver\tmutate');
  assert.equal(obsMutate.refused, true);
  assert.equal(obsMutate.raw, 'refused: observer read-only');

  // 6. Cells tune (Studio.Gui.Generation parity)
  const cellOk = await host.dispatchCellTune('budget.limit');
  assert.equal(cellOk.routed, true);
  assert.equal(cellOk.target, 'budget.limit');

  const cellEmpty = await host.dispatchCellTune('');
  assert.equal(cellEmpty.refused, true);
  assert.equal(cellEmpty.raw, 'refused: empty cell target');

  // 6b. W3C Trace Context & OTEL span timing (afflom/verdant-studio Studio.Telemetry parity)
  const liveSpan = bus.startSpan('telemetry.check', {
    endpointKind: 'observer',
    endpointId: 'verdant-ui-root',
  });
  const liveTp = formatTraceparent(liveSpan.context);
  liveSpan.end('ok');
  assert.equal(isValidW3cTraceparent(liveTp), true);
  const tpOk = await host.dispatchTelemetryTraceparent(liveTp);
  assert.equal(tpOk.ok, true);
  assert.equal(tpOk.refused, false);

  const tpFf = await host.dispatchTelemetryTraceparent(
    'ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01',
  );
  assert.equal(tpFf.refused, true);
  assert.equal(tpFf.raw, 'refused: traceparent version ff invalid');

  const tpZero = await host.dispatchTelemetryTraceparent(
    '00-00000000000000000000000000000000-00f067aa0ba902b7-01',
  );
  assert.equal(tpZero.refused, true);
  assert.equal(tpZero.raw, 'refused: traceparent zero trace or parent id');

  const timingOk = await host.dispatchTelemetryTiming(1000, 1250);
  assert.equal(timingOk.ok, true);
  const timingBad = await host.dispatchTelemetryTiming(1250, 1000);
  assert.equal(timingBad.refused, true);
  assert.equal(timingBad.raw, 'refused: span end before start');

  // 7. W3C OTEL observability: holo.load, holo.verify, holo.dispatch, and prism.dispatch recorded in OTLP JSON
  const spans = bus.getRecentSpans();
  assert.ok(spans.some((s) => s.name === 'holo.load' && s.status === 'ok'));
  assert.ok(spans.some((s) => s.name === 'holo.verify' && s.status === 'ok'));
  assert.ok(spans.some((s) => s.name === 'holo.dispatch' && s.status === 'ok'));
  assert.ok(spans.some((s) => s.name === 'prism.dispatch' && s.status === 'ok'));

  const otlp = bus.toOtlpJson();
  assert.equal(otlp.resourceSpans.length, 1);

  // 8. Tampered archive bytes are refused explicitly (zero silent fallback)
  const tampered = new Uint8Array(holoBytes);
  tampered[tampered.length - 1] ^= 0xff;
  const badHost = new VerdantHoloHost(bus);
  await assert.rejects(() => badHost.loadFromBytes(tampered));
  assert.equal(badHost.isVerified(), false);
});

test('Universal Organ Lifecycle Contract proves all 7 invariants across every registered MediaType and utility role', async () => {
  const holoBytes = new Uint8Array(
    readFileSync(join(process.cwd(), 'app/.prism/build/verdant-ui.holo')),
  );
  const bus = new ObserverStreamBus();
  const holo = new VerdantHoloHost(bus);
  await holo.loadFromBytes(holoBytes);

  const observer = new VerdantCognitiveObserver();
  const store = new PrismStore();
  const galleries = new ContentGalleryStore();
  const registry = new UnifiedExpertRegistry();
  const pool = new PrismHostPool(bus, store, observer.grants, null);
  let neuralCalls = 0;
  const bridge = new StudioChatBridge({
    store,
    pool,
    observer,
    galleries,
    registry,
    runLocalStudioPrompt: async () => {
      neuralCalls += 1;
      return '{}';
    },
  });

  try {
    for (const desc of ORGAN_REGISTRY) {
      const created = await bridge.createArtifactFromPrompt({
        prompt: desc.samplePrompt,
        forcedMediaType: desc.mediaType,
      });
      const proj = store.get(created.prismId);
      assert.ok(proj, `Organ ${desc.id}: project missing in store`);
      assert.ok(proj.declaration?.view?.proceduralOrgan, `Organ ${desc.id}: missing proceduralOrgan`);
      const proc = proj.declaration!.view!.proceduralOrgan!;

      // Invariant 1: normalizeOrganWeights round-trips every persisted field cleanly
      const renormalized = normalizeOrganWeights(
        proc.mediaType,
        proc.weights as unknown as Record<string, unknown>,
        desc.samplePrompt,
      );
      assert.equal(renormalized.mediaType, desc.mediaType, `Organ ${desc.id}: mediaType drifted`);
      const renormRecord = renormalized as unknown as Record<string, unknown>;
      for (const key of desc.persistedWeightKeys) {
        if (key in (proc.weights as unknown as Record<string, unknown>)) {
          assert.notEqual(
            renormRecord[key],
            undefined,
            `Organ ${desc.id}: normalizeOrganWeights dropped required field "${key}"`,
          );
        }
      }

      // Invariant 2: Default FK-07 grants satisfy storage/browser requirements via verdant-ui.holo
      const needsStorage = desc.requiredGrants.includes('prism-storage');
      if (needsStorage) {
        assert.equal(proj.draft.grants.storage, true, `Organ ${desc.id}: storage grant must be enabled`);
        const holoGrant = await holo.dispatchGrant(desc.mediaType, 'storage', true, false);
        assert.equal(holoGrant.allowed, true, `Organ ${desc.id}: holo refused offline storage`);
      }

      // Invariant 3: Visible in active 'personal' Published Prism registry (verified by both host & holo)
      const comp = String((proc.weights as unknown as Record<string, unknown>).compartment ?? 'personal');
      const holoComp = await holo.dispatchCompartment(comp, 'personal', 'private');
      assert.equal(
        holoComp.visible,
        true,
        `Organ ${desc.id}: compartment "${comp}" is hidden from active "personal" space`,
      );
      const visibleList = observer.getVisiblePublishedPrismsForSpace(
        store.listPublishedSummaries(),
        'personal',
      );
      assert.ok(
        visibleList.some((p) => p.id === proj.id),
        `Organ ${desc.id}: not visible in getVisiblePublishedPrismsForSpace("personal")`,
      );

      // Invariant 4: @Prism status via PrismHostPool returns structured { status: 'ok', refused: false }
      const statusDispatch = await pool.dispatchPublishedPrism(proj.id, 'status');
      const statusObj = statusDispatch.result as Record<string, unknown>;
      assert.equal(statusObj.status, 'ok', `Organ ${desc.id}: status dispatch did not return ok`);
      assert.equal(statusObj.refused, false, `Organ ${desc.id}: status dispatch refused`);
      assert.equal(statusObj.mediaType, desc.mediaType);
      assert.ok(
        typeof statusObj.summary === 'string' && !statusObj.summary.includes('[object Object]'),
        `Organ ${desc.id}: status summary corrupted`,
      );

      // Invariant 5: Refine succeeds cleanly across all organs; utility/space/research fast-path uses 0 neural calls
      if (desc.sampleRefinePrompt) {
        const beforeCalls = neuralCalls;
        const refined = await bridge.refinePrismFromPrompt({
          target: proj.id,
          instruction: desc.sampleRefinePrompt,
        });
        assert.ok(refined.prismId === proj.id, `Organ ${desc.id}: refine did not target same prism`);
        if (
          desc.mediaType === 'tool-organ' ||
          desc.mediaType === 'space' ||
          desc.mediaType === 'research-program'
        ) {
          assert.equal(
            neuralCalls,
            beforeCalls,
            `Organ ${desc.id}: deterministic fast-path unexpectedly invoked neural model`,
          );
        }
      }

      // Invariant 6: Synthesized procedural view compiles clean HTML/JS with zero syntax errors
      const view = synthesizeProceduralOrganView(proj.declaration!, proj.declaration!.view!.proceduralOrgan!);
      assert.ok(view.indexHtml.length > 40, `Organ ${desc.id}: empty HTML view`);
      assert.ok(view.appJs.length > 40, `Organ ${desc.id}: empty JS runtime`);
      assert.doesNotThrow(() => new Function(view.appJs), `Organ ${desc.id}: generated runtime JS has syntax error`);

      // Invariant 7: Descriptor lookup & status formatter never fail on valid organs, refuse unknown
      const looked = lookupOrganDescriptor(desc.mediaType, desc.role);
      assert.equal(looked.mediaType, desc.mediaType);
      const formatted = formatOrganStatusPayload({
        prismName: proj.draft.name,
        summary: proj.draft.summary,
        decl: proj.declaration!,
        mediaType: desc.mediaType,
        activeOrgans: proc.activeOrgans,
        weights: proc.weights,
        requestLine: 'status',
        bus,
      });
      assert.equal(formatted.refused, false);
    }

    assert.throws(() => lookupOrganDescriptor('bogus' as never), /refused:/);
  } finally {
    pool.dispose();
  }
});
