import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { gunzipSync } from 'node:zlib';
import {
  checkDeclaration,
  createUndefinedDeclaration,
  dispatchStudioOp,
  type PrismDeclaration,
  type StudioApiOp,
} from './verdantStudioApi';
import { evaluateVerdantRouting } from './verdantStandardsRouter';
import {
  buildStudioPromptForStep,
  computeGuidedWizardState,
  intentToAnchorOps,
  mergeLocalModelExtraction,
  parseFollowUpInstruction,
  parseIdeaSentence,
  reconcileWizardTurn,
  type GuidedStepId,
} from './prismInterviewEngine';
import { compilePrismDeclaration, instantiateAppFoundation } from './prismCompiler';

const PRISM_CODE_ROOT = path.resolve(process.cwd(), 'node_modules/prism-code');

function applyOps(decl: PrismDeclaration, ops: readonly StudioApiOp[]): PrismDeclaration {
  const res = dispatchStudioOp(decl, { op: 'batch', ops });
  assert.equal(res.ok, true, res.ok ? '' : `Batch failed [${res.code}]: ${res.message}`);
  return res.declaration;
}

function loadFoundationFiles(): Record<string, string> {
  const canonicalDir = path.join(PRISM_CODE_ROOT, 'templates/canonical');
  const index = JSON.parse(fs.readFileSync(path.join(canonicalDir, 'index.json'), 'utf8')) as {
    files: Record<string, string>;
  };
  const out: Record<string, string> = {};
  for (const [relPath, label] of Object.entries(index.files)) {
    if (!relPath.startsWith('app/') || relPath === 'app/src/App.lex.tex') continue;
    const objPath = path.join(canonicalDir, 'objects', label.replace(/^blake3:/, ''));
    out[relPath.slice(4)] = fs.readFileSync(objPath, 'utf8');
  }
  return out;
}

test('parseIdeaSentence synthesizes valid, provable Prisms and routes exact standards across 5 real-world prompts', () => {
  const cases = [
    {
      sentence: 'Check if a new expense fits my monthly budget and warn me if it goes over, refusing zero amounts.',
      expectLayered: [] as string[],
      forbidLayered: ['EU-REG-2016-679', 'US-CFR-16-312-2026', 'EU-REG-2024-1689'],
    },
    {
      sentence: 'Let EU customers check loan eligibility by age and credit score, refusing applicants under 18 and approving scores 75 or higher.',
      expectLayered: ['EU-REG-2016-679', 'EU-REG-2024-2847', 'EU-DIR-2019-770', 'ETSI-EN-301-549-3.2.1'],
      forbidLayered: ['US-CFR-16-312-2026'],
    },
    {
      sentence: 'Split a restaurant bill with tip percent across diners and refuse zero diners or a zero bill.',
      expectLayered: [] as string[],
      forbidLayered: ['EU-REG-2016-679', 'US-CFR-16-312-2026'],
    },
    {
      sentence: 'An interactive game damage calculator with local AI hints for kids in the US.',
      expectLayered: ['US-CFR-16-312-2026', 'NIST-AI-RMF-PLAYBOOK', 'HUNICKE-MDA-2004'],
      forbidLayered: ['EU-REG-2016-679'],
    },
    {
      sentence: 'Alert me when daily water intake is under 8 cups.',
      expectLayered: [] as string[],
      forbidLayered: ['EU-REG-2016-679', 'US-CFR-16-312-2026'],
    },
  ];

  for (const tc of cases) {
    const fresh = createUndefinedDeclaration();
    const intent = parseIdeaSentence(tc.sentence);
    const ops = intentToAnchorOps(fresh, intent);
    const anchored = applyOps(fresh, ops);

    // Verbatim owner decision provenance (DEC-01)
    assert.equal(anchored.decisions[0]?.id, 'DEC-01');
    assert.equal(anchored.decisions[0]?.quote, tc.sentence);
    assert.ok(anchored.encoding && anchored.encoding.fields.length >= 1);
    assert.ok(anchored.features.length >= 1);
    assert.ok(anchored.refusals.length >= 1);
    // Progressive settlement: view stays null until Step 5!
    assert.equal(anchored.view, null);

    // Now simulate Steps 4 & 5: sync safety & synthesize design view
    const withView = applyOps(anchored, [
      { op: 'sync-stride-threats' },
      { op: 'synthesize-view-from-system', viewpoint: intent.recommendedViewpoint },
    ]);

    const check = checkDeclaration(withView);
    assert.equal(check.ok, true, `"${tc.sentence}" failed checkDeclaration: ${JSON.stringify(check.diagnostics)}`);
    assert.equal(check.vectorsPassed, check.vectorsTotal);

    const routing = evaluateVerdantRouting(withView);
    const activeIds = new Set(routing.activeLayeredStandards.map((s) => s.authority.id));
    for (const id of tc.expectLayered) {
      assert.ok(activeIds.has(id), `Expected standard ${id} for "${tc.sentence}", got ${[...activeIds].join(', ')}`);
    }
    for (const id of tc.forbidLayered) {
      assert.ok(!activeIds.has(id), `Unexpected standard ${id} for "${tc.sentence}"`);
    }

    // Never expose Observer / OTel in routed standards or guided safety copy
    assert.ok(!activeIds.has('OTEL-SPEC-1.40.0-TRACE-API'));
    const guided = computeGuidedWizardState(withView, routing, check);
    for (const item of guided.openSafetyItems) {
      assert.ok(!/observer|vtensor|lexlean|opentelemetry/i.test(item.plainPrompt));
    }

    // Must compile with 0 prism.design/1 errors
    const compiled = compilePrismDeclaration(withView);
    assert.ok(compiled !== null);
    assert.equal(compiled.designDiagnostics.filter((d) => d.severity === 'error').length, 0);
  }
});

test('plain-English synthesized Prisms pass real WASM LexLean + Lean 4.32.2 kernel proof', async () => {
  const buildCore = await import(path.join(PRISM_CODE_ROOT, 'src/journey/build-core.mjs'));
  const modelProve = await import(path.join(PRISM_CODE_ROOT, 'src/journey/model-prove.mjs'));
  const proveCore = await import(path.join(PRISM_CODE_ROOT, 'src/journey/prove-core.mjs'));

  const lexleanGz = fs.readFileSync(
    path.join(PRISM_CODE_ROOT, 'wasm/pinned/4f/302c18e02a46b0d015c57bd389a0754b301ec876aa7a637f786c4562274ecb'),
  );
  const lexleanModule = await WebAssembly.compile(gunzipSync(lexleanGz));
  const evaluatorBytes = new Uint8Array(
    fs.readFileSync(path.join(PRISM_CODE_ROOT, 'src/journey/guest/evaluator.wasm')),
  );
  const kernel = await modelProve.openModelKernel();
  const foundation = loadFoundationFiles();

  const sentences = [
    'Check if a new expense fits my monthly budget and warn me if it goes over, refusing zero amounts.',
    'Let EU customers check loan eligibility by age and credit score, refusing applicants under 18 and approving scores 75 or higher.',
    'Split a restaurant bill with tip percent across diners and refuse zero diners or a zero bill.',
  ];

  for (const sentence of sentences) {
    const fresh = createUndefinedDeclaration();
    const intent = parseIdeaSentence(sentence);
    const ready = applyOps(fresh, [
      ...intentToAnchorOps(fresh, intent),
      { op: 'sync-stride-threats' },
      { op: 'synthesize-view-from-system', viewpoint: intent.recommendedViewpoint },
    ]);
    const compiled = compilePrismDeclaration(ready);
    assert.ok(compiled !== null);
    const mergedApp = {
      ...instantiateAppFoundation(foundation, ready.metadata.slug),
      ...compiled.appFiles,
    };
    const result = await buildCore.buildFromFiles({
      files: mergedApp,
      lexlean: lexleanModule,
      evaluator: evaluatorBytes,
      prove: proveCore.kernelProver(kernel, { record: true }),
    });
    assert.equal(
      result.ok,
      true,
      `Synthesized Prism for "${sentence}" failed WASM proof at ${result.stage}: ${result.message}`,
    );
    assert.equal(result.proved, true);
  }
});

test('parseFollowUpInstruction and mergeLocalModelExtraction handle incremental refinements and malformed JSON safely', () => {
  const fresh = createUndefinedDeclaration();
  const intent = parseIdeaSentence('Alert me when daily water intake is under 8 cups.');
  let decl = applyOps(fresh, intentToAnchorOps(fresh, intent));

  // Add a choice field
  const f1 = parseFollowUpInstruction(decl, 'Add a field for activity level (Low, Moderate, High)');
  assert.equal(f1.ok, true);
  decl = applyOps(decl, f1.ops);
  assert.ok(decl.encoding?.fields.some((f) => f.kind === 'choice' && f.choices?.length === 3));

  // Add a refusal with automatic passing R5 vector
  const f2 = parseFollowUpInstruction(decl, 'Refuse when cups is over 30');
  assert.equal(f2.ok, true);
  decl = applyOps(decl, f2.ops);
  assert.ok(decl.refusals.some((r) => r.failureText.includes('30')));

  // Add a threshold rule
  const f3 = parseFollowUpInstruction(decl, 'Approve when cups is at least 10');
  assert.equal(f3.ok, true);
  decl = applyOps(decl, f3.ops);
  assert.ok(decl.features.some((f) => f.outcome.kind === 'label' && f.outcome.label === 'Approved'));

  // Malformed local model output falls back cleanly without throwing
  const mergedBad = mergeLocalModelExtraction('Sorry I cannot output JSON right now', intent);
  assert.equal(mergedBad.name.value, intent.name.value);

  // Valid local model JSON enhances title & domain safely
  const mergedGood = mergeLocalModelExtraction(
    '```json\n{"name":"Hydration Coach","domain":"health","outputName":"Hydration Status"}\n```',
    intent,
  );
  assert.equal(mergedGood.name.value, 'Hydration Coach');
  assert.equal(mergedGood.domain.value, 'health');
  assert.equal(mergedGood.outputName, 'Hydration Status');
});

test('full multi-turn natural-language conversation reconciles a regulated EU Prism from undefined to readyToSeal + Lean WASM proof with ZERO manual form edits', async () => {
  let decl = createUndefinedDeclaration();
  let step: GuidedStepId = 'anchor-idea';

  const turns = [
    'Check loan eligibility by age and credit score, refusing applicants under 18 and approving scores 75 or higher.',
    'This is for EU customers and handles personal info, strictly offline.',
    'Also refuse if credit score is over 850.',
    'Looks good, check safety next.',
    'Yes, keep strictly on-device and offline.',
    'Use a crisp paper decision form.',
    'Yes, apply all required protections.',
  ];

  for (const userMsg of turns) {
    const res = reconcileWizardTurn(decl, step, userMsg);
    if (res.ops.length > 0) {
      decl = applyOps(decl, res.ops);
    }
    step = res.nextStep;
  }

  assert.equal(step, 'ready-to-seal');
  const routing = evaluateVerdantRouting(decl);
  assert.equal(routing.readyToPreview, true);
  assert.equal(
    routing.readyToSeal,
    true,
    `Expected full NL conversation to reach readyToSeal: ${JSON.stringify(routing.stages)}`,
  );
  // EU + personal info -> GDPR + CRA + Digital Content + EN 301 549 routed AND settled purely via NL!
  const activeIds = new Set(routing.activeLayeredStandards.map((s) => s.authority.id));
  assert.ok(activeIds.has('EU-REG-2016-679'));
  assert.ok(activeIds.has('EU-REG-2024-2847'));
  assert.equal(routing.summary.obligationsSettled, routing.summary.obligationsTotal);
  assert.equal(decl.theme.preset, 'paper');
  assert.equal(decl.view?.viewpoint, 'decision-form');

  // Post-seal natural-language tweak automatically re-reconciles everything and stays readyToSeal!
  const tweak = reconcileWizardTurn(decl, step, 'Switch to dark midnight live calculator');
  decl = applyOps(decl, tweak.ops);
  assert.equal(decl.theme.preset, 'midnight');
  assert.equal(decl.view?.viewpoint, 'calculator');
  assert.equal(evaluateVerdantRouting(decl).readyToSeal, true);

  // Verify the NL-reconciled Prism compiles & proves cleanly in real WASM Lean 4.32.2!
  const buildCore = await import(path.join(PRISM_CODE_ROOT, 'src/journey/build-core.mjs'));
  const modelProve = await import(path.join(PRISM_CODE_ROOT, 'src/journey/model-prove.mjs'));
  const proveCore = await import(path.join(PRISM_CODE_ROOT, 'src/journey/prove-core.mjs'));
  const lexleanGz = fs.readFileSync(
    path.join(PRISM_CODE_ROOT, 'wasm/pinned/4f/302c18e02a46b0d015c57bd389a0754b301ec876aa7a637f786c4562274ecb'),
  );
  const lexleanModule = await WebAssembly.compile(gunzipSync(lexleanGz));
  const evaluatorBytes = new Uint8Array(
    fs.readFileSync(path.join(PRISM_CODE_ROOT, 'src/journey/guest/evaluator.wasm')),
  );
  const kernel = await modelProve.openModelKernel();
  const foundation = loadFoundationFiles();
  const compiled = compilePrismDeclaration(decl);
  assert.ok(compiled !== null);

  const built = await buildCore.buildFromFiles({
    files: { ...instantiateAppFoundation(foundation, decl.metadata.slug), ...compiled.appFiles },
    lexlean: lexleanModule,
    evaluator: evaluatorBytes,
    prove: proveCore.kernelProver(kernel, { record: true }),
  });
  assert.equal(built.ok, true, `NL-reconciled Prism failed WASM proof at ${built.stage}: ${built.message}`);
  assert.equal(built.proved, true);
});

test('2-turn autonomous shortcut ("Reconcile everything & finish") settles all remaining facets cleanly', () => {
  let decl = createUndefinedDeclaration();
  const t1 = reconcileWizardTurn(
    decl,
    'anchor-idea',
    'Split a restaurant bill with tip percent across diners and refuse zero diners.',
  );
  decl = applyOps(decl, t1.ops);

  const prompt2 = buildStudioPromptForStep(decl, evaluateVerdantRouting(decl), checkDeclaration(decl), t1.nextStep);
  assert.ok(!/observer|vtensor|lexlean|opentelemetry/i.test(prompt2.promptText));

  const t2 = reconcileWizardTurn(decl, t1.nextStep, 'Reconcile everything & finish with terracotta theme');
  decl = applyOps(decl, t2.ops);

  assert.equal(t2.nextStep, 'ready-to-seal');
  assert.equal(t2.openPreview, true);
  assert.equal(decl.theme.preset, 'terracotta');
  assert.equal(evaluateVerdantRouting(decl).readyToSeal, true);
});
