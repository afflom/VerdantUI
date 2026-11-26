import test from 'node:test';
import assert from 'node:assert/strict';
import {
  checkStudioLifecycle,
  deriveStudioAgenda,
  loadStudioLifecycleModel,
  planSemanticRelease,
  renderGeneratedDocs,
  type StudioLifecycleModel,
} from '../../scripts/studio-lifecycle';
import {
  evaluateVerdantRatchet,
  loadRatchetSnapshot,
  verifyPlantedDefects,
} from '../../scripts/ratchet-gate';

test('Verdant Studio lifecycle check passes on live model/ registers with well-founded citations', () => {
  const model = loadStudioLifecycleModel(process.cwd());
  const report = checkStudioLifecycle(model, { verifyGeneratedDocs: true, verifyFilesExist: true });
  assert.equal(report.ok, true, `Unexpected lifecycle errors: ${JSON.stringify(report.errors)}`);
  assert.ok(report.admissionsChecked >= 16);
  assert.ok(report.activeFeatures >= 6);
  assert.ok(report.elapsedMs < 1000, `studio:check took too long: ${report.elapsedMs}ms`);
});

test('Verdant Studio lifecycle refuses dropped locked IDs, unresolved citations, cyclic Der, and regressed budgets', () => {
  const base = loadStudioLifecycleModel(process.cwd());

  // 1. Monotone lock violation (dropped locked ID)
  const droppedLocked: StudioLifecycleModel = {
    ...base,
    ids: base.ids.filter((id) => id.code !== 'VUI-PUBLIC-BENEFIT-01'),
  };
  const r1 = checkStudioLifecycle(droppedLocked, { verifyGeneratedDocs: false, verifyFilesExist: false });
  assert.equal(r1.ok, false);
  assert.ok(r1.errors.some((e) => e.code === 'MONOTONE-LOCK-BROKEN'));

  // 2. Unresolved Req citation
  const badReq: StudioLifecycleModel = {
    ...base,
    admissions: [
      ...base.admissions,
      { construct: 'VUI-FAKE-01', citation: { kind: 'req', decisionId: 'DEC-999' } },
    ],
  };
  const r2 = checkStudioLifecycle(badReq, { verifyGeneratedDocs: false, verifyFilesExist: false });
  assert.equal(r2.ok, false);
  assert.ok(r2.errors.some((e) => e.code === 'UNRESOLVED-CITATION'));

  // 3. Well-foundedness violation (Der citing an unadmitted / forward construct)
  const badDer: StudioLifecycleModel = {
    ...base,
    admissions: [
      ...base.admissions,
      {
        construct: 'VUI-DER-CYCLE',
        citation: {
          kind: 'der',
          rule: 'extrapolated',
          inputs: ['VUI-NOT-YET-ADMITTED'],
          checkId: 'VUI-CLIENT-01',
        },
      },
    ],
  };
  const r3 = checkStudioLifecycle(badDer, { verifyGeneratedDocs: false, verifyFilesExist: false });
  assert.equal(r3.ok, false);
  assert.ok(r3.errors.some((e) => e.message.includes('well-foundedness')));

  // 4. Performance budget regression
  const badBudget: StudioLifecycleModel = {
    ...base,
    performanceBudgets: {
      ...base.performanceBudgets,
      budgets: {
        ...base.performanceBudgets.budgets,
        turnCompletionOpfsStallMsMax: 25,
        frameYieldBudgetUsMax: 25_000,
      },
    },
  };
  const r4 = checkStudioLifecycle(badBudget, { verifyGeneratedDocs: false, verifyFilesExist: false });
  assert.equal(r4.ok, false);
  assert.ok(r4.errors.filter((e) => e.code === 'BUDGET-REGRESSION').length === 2);
});

test('Semantic release planner derives exact minor/patch/none bumps and deterministic docs from model/', () => {
  const model = loadStudioLifecycleModel(process.cwd());
  const latest = model.releaseLedger.releases[model.releaseLedger.releases.length - 1]!;
  const [maj, min, pat] = latest.version.split('.').map(Number);

  // 1. Newly active feature beyond latest release ledger triggers a minor bump
  const withNewFeat: StudioLifecycleModel = {
    ...model,
    features: [
      ...model.features,
      {
        id: 'VUI-FEAT-TEST-NEW',
        title: 'Synthetic New Active Feature',
        stage: 'active',
        introducedIn: `${maj}.${min + 1}.0`,
        ownerFile: 'src/engine/vtensorProvider.ts',
        testFile: 'src/engine/vtensorProvider.test.ts',
        invariantId: 'VUI-ENGINE-01',
        citation: { kind: 'req', decisionId: 'DEC-001' },
        publicBenefitImpact: 'Test minor bump',
      },
    ],
  };
  const minorPlan = planSemanticRelease(withNewFeat, 'auto');
  assert.equal(minorPlan.bump, 'minor');
  assert.equal(minorPlan.nextVersion, `${maj}.${min + 1}.0`);
  assert.ok(minorPlan.releaseNotesMarkdown.includes(`VerdantUI v${maj}.${min + 1}.0`));

  // 2. Engine pin bump with all active features already released triggers a patch bump
  const allActiveIds = model.features.filter((f) => f.stage === 'active').map((f) => f.id);
  const syncedLedger: StudioLifecycleModel = {
    ...model,
    releaseLedger: {
      schema: 'verdant-ui/release-ledger/1',
      releases: [{ ...latest, activeFeatureIds: allActiveIds }],
    },
  };
  assert.equal(planSemanticRelease(syncedLedger, 'auto').bump, 'none');

  const engineBumpModel: StudioLifecycleModel = {
    ...syncedLedger,
    enginePin: { ...model.enginePin, tag: 'v2.99.0' },
  };
  const patchPlan = planSemanticRelease(engineBumpModel, 'auto');
  assert.equal(patchPlan.bump, 'patch');
  assert.equal(patchPlan.nextVersion, `${maj}.${min}.${pat + 1}`);

  const agenda = deriveStudioAgenda(withNewFeat);
  assert.ok(agenda.some((item) => item.kind === 'release-ready'));

  const docs = renderGeneratedDocs(model);
  assert.ok(docs['CONFORMANCE.md'].includes('Free Public Benefit AI Service'));
  assert.ok(docs['docs/owner-decisions.md'].includes('DEC-011'));
  assert.ok(docs['docs/product-lifecycle.md'].includes('VUI-FEAT-005'));
});

test('Permanent ratchet enforces 11/11 planted-defect gates including Free Public Benefit & zero UI jargon', () => {
  const snap = loadRatchetSnapshot(process.cwd());
  const clean = evaluateVerdantRatchet(snap);
  assert.equal(clean.ok, true, `Ratchet violations: ${JSON.stringify(clean.violations)}`);

  const planted = verifyPlantedDefects(snap);
  assert.equal(planted.allCaught, true, `Missed planted defects: ${planted.failures.join(', ')}`);
  assert.equal(planted.casesVerified, 11);
});
