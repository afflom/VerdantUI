import assert from 'node:assert/strict';
import test from 'node:test';
import {
  createDeclarativeStarter,
  createUndefinedDeclaration,
  dispatchStudioOp,
  upgradeDeclarationV1ToV2,
  type PrismDeclaration,
  type StudioApiOp,
} from './verdantStudioApi';
import { evaluateVerdantRouting, wcagContrastRatio } from './verdantStandardsRouter';
import { compilePrismDeclaration } from './prismCompiler';

function applyOps(decl: PrismDeclaration, ops: readonly StudioApiOp[]): PrismDeclaration {
  const res = dispatchStudioOp(decl, { op: 'batch', ops });
  assert.equal(res.ok, true, res.ok ? '' : `Op batch failed [${res.code}]: ${res.message}`);
  return res.declaration;
}

test('new undefined Verdant Prism has zero fields, rules, or view and refuses premature compilation', () => {
  const fresh = createUndefinedDeclaration();
  const routing = evaluateVerdantRouting(fresh);

  assert.equal(routing.isUndefined, true);
  assert.equal(fresh.encoding, null);
  assert.equal(fresh.view, null);
  assert.equal(fresh.features.length, 0);
  assert.equal(fresh.refusals.length, 0);
  assert.equal(routing.readyToPreview, false);
  assert.equal(routing.readyToSeal, false);
  assert.equal(routing.recommendedStage, 'concept');
  assert.equal(compilePrismDeclaration(fresh), null);
});

test('guided cross-facet flow derives encoding, STRIDE threats, and traceable view from Concept/OPM', () => {
  const fresh = createUndefinedDeclaration('Shipping Fee Guard');
  const complete = applyOps(fresh, [
    {
      op: 'set-concept-profile',
      purpose: 'Compute deterministic shipping fee with free-shipping threshold and zero-weight refusal.',
      domain: 'general',
      audience: 'personal',
      dataClassification: 'none',
      jurisdictions: ['none'],
      usesLocalAi: false,
      involvesPayments: false,
    },
    {
      op: 'admit-stakeholder',
      stakeholder: { name: 'Shopper', provRole: 'consumer', concern: 'Know exact shipping fee upfront.' },
      autoAdmitRequirement: true,
    },
    {
      op: 'admit-opm-object',
      object: { name: 'Weight', role: 'input', kind: 'nat', states: [], unit: 'oz', min: 0, max: 500, defaultValue: 16 },
    },
    {
      op: 'admit-opm-object',
      object: { name: 'Shipping Verdict', role: 'output', kind: 'choice', states: ['Standard', 'Heavy'] },
    },
    {
      op: 'admit-opm-process',
      process: {
        name: 'Estimate Shipping',
        summary: 'Classify package weight and refuse zero-weight requests.',
        consumesObjectIds: ['OBJ-01'],
        yieldsObjectIds: ['OBJ-02'],
        actorStakeholderIds: ['STK-01'],
      },
    },
    { op: 'derive-encoding-from-opm' },
    {
      op: 'admit-refusal',
      refusal: {
        id: 'APP-01',
        statement: 'Refuse zero package weight.',
        condition: { all: [{ field: 'weight', op: 'eq', value: 0 }] },
        failureText: 'Weight must be greater than zero.',
        fieldHint: 'weight',
        negativeId: 'APP-01',
        citation: { kind: 'req', requirementId: 'REQ-01' },
      },
    },
    {
      op: 'admit-feature',
      feature: {
        id: 'APP-02',
        level: 'build',
        statement: 'Flag heavy parcels over 100 oz.',
        condition: { all: [{ field: 'weight', op: 'gt', value: 100 }] },
        outcome: { kind: 'label', code: 1, label: 'Heavy', tone: 'warning' },
        priority: 1,
        citation: { kind: 'req', requirementId: 'REQ-01' },
      },
    },
    {
      op: 'admit-feature',
      feature: {
        id: 'APP-03',
        level: 'build',
        statement: 'Classify standard parcels.',
        condition: { all: [] },
        outcome: { kind: 'label', code: 2, label: 'Standard', tone: 'success' },
        priority: 2,
        citation: { kind: 'req', requirementId: 'REQ-01' },
      },
    },
    {
      op: 'admit-vector',
      vector: {
        id: 'vec_std',
        inputs: { weight: 16 },
        expectedResponse: 'Standard',
        forFeatureId: 'APP-03',
        isRefusal: false,
      },
    },
    {
      op: 'admit-vector',
      vector: {
        id: 'vec_zero',
        inputs: { weight: 0 },
        expectedResponse: 'Weight must be greater than zero.',
        forFeatureId: 'APP-01',
        isRefusal: true,
      },
    },
    { op: 'sync-stride-threats' },
    { op: 'synthesize-view-from-system' },
  ]);

  const routing = evaluateVerdantRouting(complete);
  assert.equal(routing.isUndefined, false);
  assert.equal(routing.readyToPreview, true);
  assert.equal(routing.readyToSeal, true, `Expected readyToSeal, stages: ${JSON.stringify(routing.stages)}`);
  // Personal offline tool must route ZERO extra regulatory layers
  assert.equal(routing.activeLayeredStandards.length, 0);
  assert.ok(routing.excludedStandards.some((e) => e.authority.id === 'EU-REG-2016-679'));
  assert.ok(routing.excludedStandards.some((e) => e.authority.id === 'EU-REG-2024-1689'));
  assert.ok(routing.excludedStandards.some((e) => e.authority.id === 'US-CFR-16-312-2026'));

  const compiled = compilePrismDeclaration(complete);
  assert.ok(compiled !== null);
  assert.ok(compiled.files['model/concept-opm.json']);
  assert.ok(compiled.files['model/security-stride.json']);
  assert.ok(compiled.files['model/standards-routing.json']);
});

test('deterministic applicability router activates GDPR, EU AI Act, CRA, COPPA, and Games pack only when triggered', () => {
  // Start from complete personal baseline, then layer regulated EU + Personal Data + Local AI weights
  const base = createDeclarativeStarter('budget-guard');
  const regulated = applyOps(base, [
    {
      op: 'set-concept-profile',
      audience: 'public-adults',
      dataClassification: 'personal',
      jurisdictions: ['eu'],
      usesLocalAi: true,
    },
    { op: 'sync-stride-threats' },
  ]);

  let routing = evaluateVerdantRouting(regulated);
  const activeIds = new Set(routing.activeLayeredStandards.map((s) => s.authority.id));
  assert.ok(activeIds.has('EU-REG-2016-679'), 'GDPR should route for personal data + EU');
  assert.ok(activeIds.has('EU-REG-2024-1689'), 'EU AI Act should route for usesLocalAi + EU');
  assert.ok(activeIds.has('NIST-AI-RMF-PLAYBOOK'), 'NIST AI RMF should route for usesLocalAi');
  assert.ok(activeIds.has('EU-REG-2024-2847'), 'CRA should route for public + EU');
  assert.ok(activeIds.has('EU-DIR-2019-770'), 'Digital Content Directive should route for public + EU');
  assert.ok(activeIds.has('ETSI-EN-301-549-3.2.1'), 'EN 301 549 should route for public + EU');
  assert.ok(!activeIds.has('US-CFR-16-312-2026'), 'COPPA must NOT route when audience is adults and jurisdiction is EU');
  assert.equal(routing.readyToSeal, false, 'Unsettled threats & layered obligations must block sealing');

  // Settle any open STRIDE threats and every unsettled routed obligation via closed API ops
  const settleOps: StudioApiOp[] = [];
  for (const thr of regulated.security.threats) {
    if (thr.status === 'open') {
      settleOps.push({
        op: 'settle-stride-threat',
        id: thr.id,
        status: 'mitigated-by-sandbox',
        rationale: 'Verified BLAKE3 .holo provenance and on-device zero-egress sandbox isolation.',
      });
    }
  }
  for (const std of routing.activeLayeredStandards) {
    for (const ob of std.obligations) {
      if (!ob.settled && ob.suggestedOp) {
        settleOps.push(ob.suggestedOp);
      }
    }
  }
  const settled = applyOps(regulated, settleOps);
  routing = evaluateVerdantRouting(settled);
  assert.equal(
    routing.readyToSeal,
    true,
    `Expected readyToSeal after settling all threats & obligations: ${JSON.stringify(routing.stages)}`,
  );

  // Check minor-facing US + Games/Media routing
  const minorGame = applyOps(base, [
    {
      op: 'set-concept-profile',
      domain: 'games-media',
      audience: 'includes-minors',
      jurisdictions: ['us'],
    },
  ]);
  const minorRouting = evaluateVerdantRouting(minorGame);
  const minorIds = new Set(minorRouting.activeLayeredStandards.map((s) => s.authority.id));
  assert.ok(minorIds.has('US-CFR-16-312-2026'), 'COPPA should route for minors + US');
  assert.ok(minorIds.has('HUNICKE-MDA-2004'), 'Games domain pack should route for domain=games-media');
  assert.ok(!minorIds.has('EU-REG-2016-679'), 'GDPR should remain excluded for US non-personal game');
});

test('all curated reference starters pass full Verdant Prism completeness and WCAG AA contrast', () => {
  assert.ok(wcagContrastRatio('#f7f4ec', '#1f5e44') >= 4.5);
  assert.ok(wcagContrastRatio('#ffffff', '#ffffaa') < 4.5);

  for (const kind of ['budget-guard', 'pricing-rules', 'eligibility-gate', 'split-calc', 'blank'] as const) {
    const decl = createDeclarativeStarter(kind);
    const routing = evaluateVerdantRouting(decl);
    assert.equal(
      routing.readyToSeal,
      true,
      `Starter "${kind}" should be readyToSeal: ${JSON.stringify(routing.stages)}`,
    );
  }
});

test('upgradeDeclarationV1ToV2 preserves legacy v1 declarative fields, rules, and views cleanly', () => {
  const modern = createDeclarativeStarter('split-calc');
  const fakeLegacyV1 = {
    schema: 'verdant-studio/declaration/1',
    metadata: modern.metadata,
    decisions: modern.decisions,
    authorities: modern.authorities.slice(0, 2),
    requirements: modern.requirements,
    encoding: modern.encoding,
    features: modern.features,
    refusals: modern.refusals,
    vectors: modern.vectors,
    view: modern.view,
    theme: modern.theme,
    capabilities: modern.capabilities,
    admissions: modern.admissions,
    lockedIds: modern.lockedIds,
    errors: modern.errors,
    auditLog: modern.auditLog,
  };
  const upgraded = upgradeDeclarationV1ToV2(fakeLegacyV1);
  assert.equal(upgraded.schema, 'verdant-studio/declaration/2');
  assert.equal(upgraded.encoding?.fields.length, modern.encoding?.fields.length);
  assert.equal(upgraded.features.length, modern.features.length);
  assert.ok(upgraded.concept.opmObjects.length > 0);
  assert.ok(upgraded.security.threats.length > 0);
  assert.equal(evaluateVerdantRouting(upgraded).readyToPreview, true);
});
