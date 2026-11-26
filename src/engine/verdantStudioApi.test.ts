import assert from 'node:assert/strict';
import test from 'node:test';
import {
  checkDeclaration,
  computeAgenda,
  createDeclarativeStarter,
  createEmptyDeclaration,
  deriveRepairOp,
  dispatchStudioOp,
  evaluateDeclaredModel,
  type DeclarativeStarterId,
} from './verdantStudioApi';

test('all curated declarative starters pass admission, traceability, and vector replay', () => {
  const starters: DeclarativeStarterId[] = [
    'budget-guard',
    'pricing-rules',
    'eligibility-gate',
    'split-calc',
    'blank',
  ];
  for (const kind of starters) {
    const decl = createDeclarativeStarter(kind);
    const report = checkDeclaration(decl);
    assert.equal(report.ok, true, `Starter "${kind}" diagnostics: ${JSON.stringify(report.diagnostics)}`);
    assert.equal(report.vectorsPassed, report.vectorsTotal);
    assert.ok(report.vectorsTotal >= 2, `Starter "${kind}" should have >= 2 acceptance vectors`);
  }
});

test('Studio.Admission refuses uncited constructs, duplicate IDs, and dropped locked registers', () => {
  const base = createDeclarativeStarter('blank');

  // Uncited requirement reference is refused
  const badFeature = dispatchStudioOp(base, {
    op: 'admit-feature',
    feature: {
      id: 'APP-99',
      level: 'build',
      statement: 'Uncited feature',
      condition: { all: [] },
      outcome: { kind: 'label', code: 1, label: 'OK', tone: 'success' },
      priority: 99,
      citation: { kind: 'req', requirementId: 'REQ-MISSING' },
    },
  });
  assert.equal(badFeature.ok, false);
  assert.equal(badFeature.code, 'admission.unresolved');

  // Duplicate ID is refused
  const dup = dispatchStudioOp(base, {
    op: 'admit-feature',
    feature: {
      id: 'APP-02',
      level: 'build',
      statement: 'Duplicate',
      condition: { all: [] },
      outcome: { kind: 'label', code: 1, label: 'OK', tone: 'success' },
      priority: 5,
      citation: { kind: 'req', requirementId: 'REQ-01' },
    },
  });
  assert.equal(dup.ok, false);
  assert.equal(dup.code, 'admission.duplicate');

  // Locking registers enforces monotonicity (P10)
  const lockedRes = dispatchStudioOp(base, { op: 'lock-registers' });
  assert.equal(lockedRes.ok, true);
  const dropLocked = dispatchStudioOp(lockedRes.declaration, {
    op: 'remove-feature',
    id: 'APP-02',
  });
  assert.equal(dropLocked.ok, false);
  assert.equal(dropLocked.code, 'admission.monotone');
});

test('REQ-D9 refuses uncited view elements and R5 detects unreplayed refusals with self-healing repair', () => {
  const base = createDeclarativeStarter('blank');

  // Uncited UI element is refused at API boundary
  const badView = dispatchStudioOp(base, {
    op: 'declare-view',
    view: {
      ...base.view!,
      elements: [
        ...base.view!.elements,
        {
          id: 'el_orphan',
          slot: 'stat',
          label: 'Orphan Widget',
          binding: 'outcome',
          requirementId: 'REQ-NOPE',
        },
      ],
    },
  });
  assert.equal(badView.ok, false);
  assert.equal(badView.code, 'view.uncited-element');

  // Add a refusal without an example vector -> check reports contract.refusal-unreplayed -> self-healing repairs it!
  const withRefusal = dispatchStudioOp(base, {
    op: 'admit-refusal',
    refusal: {
      id: 'APP-09',
      statement: 'Refuse values over 500.',
      condition: { all: [{ field: 'amount', op: 'gt', value: 500 }] },
      failureText: 'Amount exceeds 500.',
      fieldHint: 'amount',
      negativeId: 'APP-09',
      citation: { kind: 'req', requirementId: 'REQ-01' },
    },
  });
  assert.equal(withRefusal.ok, true);
  const report = checkDeclaration(withRefusal.declaration);
  assert.equal(report.ok, false);
  assert.ok(report.diagnostics.some((d) => d.code === 'contract.refusal-unreplayed' && d.id === 'APP-09'));

  const agenda = computeAgenda(withRefusal.declaration);
  const repair = deriveRepairOp(withRefusal.declaration, 'contract.refusal-unreplayed', '', 'APP-09');
  assert.ok(repair, 'Should derive a concrete repair operation for unreplayed refusal');
  assert.ok(agenda.length > 0);

  const repaired = dispatchStudioOp(withRefusal.declaration, repair!);
  assert.equal(repaired.ok, true);
  const afterReport = checkDeclaration(repaired.declaration);
  assert.equal(afterReport.ok, true, `Expected clean pass after self-healing repair: ${JSON.stringify(afterReport.diagnostics)}`);
});

test('evaluateDeclaredModel computes deterministic rules and exact refusals', () => {
  const pricing = createDeclarativeStarter('pricing-rules');
  assert.equal(evaluateDeclaredModel(pricing, '50 4 1').response, 'VIP Net Total: 160 USD');
  assert.equal(evaluateDeclaredModel(pricing, '50 10 0').response, 'Bulk Net Total: 450 USD');
  assert.equal(evaluateDeclaredModel(pricing, '50 3 0').response, 'Standard Total: 150 USD');
  assert.equal(evaluateDeclaredModel(pricing, '50 0 0').response, 'Quantity must be at least 1.');
  assert.equal(evaluateDeclaredModel(pricing, 'bad input').isRefusal, true);
});

test('propose-encoding accepts raw JSON Schema 2020-12 and normalizes typed fields', () => {
  const base = createEmptyDeclaration('Schema Test');
  const res = dispatchStudioOp(base, {
    op: 'propose-encoding',
    name: 'LoanRequest',
    schema: {
      type: 'object',
      properties: {
        principal: { type: 'number', description: 'Principal amount' },
        plan: { type: 'string', enum: ['Fixed', 'Variable'] },
        verified: { type: 'boolean' },
      },
      required: ['principal', 'plan', 'verified'],
    },
    citation: { kind: 'req', requirementId: 'REQ-01' },
  });
  assert.equal(res.ok, true);
  assert.equal(res.declaration.encoding?.fields.length, 3);
  assert.equal(res.declaration.encoding?.fields[1].kind, 'choice');
  assert.deepEqual(res.declaration.encoding?.fields[1].choices, ['Fixed', 'Variable']);
  assert.equal(res.declaration.encoding?.fields[2].kind, 'bool');
});

test('field-vs-field comparisons, reorder-features, and update-refusal work deterministically', () => {
  const budget = createDeclarativeStarter('budget-guard');
  assert.equal(evaluateDeclaredModel(budget, '500 100 650 2').response, 'Over Budget');
  assert.equal(evaluateDeclaredModel(budget, '2000 850 120 0').response, 'Remaining Budget: 1030 USD');

  const reordered = dispatchStudioOp(budget, {
    op: 'reorder-features',
    orderedIds: ['APP-04', 'APP-03'],
  });
  assert.equal(reordered.ok, true);
  assert.equal(reordered.declaration.features[0].id, 'APP-04');

  const updatedRefusal = dispatchStudioOp(budget, {
    op: 'update-refusal',
    id: 'APP-01',
    patch: { failureText: 'Expense cannot be zero.' },
  });
  assert.equal(updatedRefusal.ok, true);
  assert.equal(evaluateDeclaredModel(updatedRefusal.declaration, '2000 100 0 0').response, 'Expense cannot be zero.');
});
