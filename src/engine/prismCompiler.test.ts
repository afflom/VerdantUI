import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { gunzipSync } from 'node:zlib';
import { loadArchive } from 'prism-code';
import {
  createDeclarativeStarter,
  evaluateDeclaredModel,
  type DeclarativeStarterId,
} from './verdantStudioApi';
import {
  compilePrismDeclaration,
  instantiateAppFoundation,
} from './prismCompiler';

const PRISM_CODE_ROOT = path.resolve(process.cwd(), 'node_modules/prism-code');

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

test('all declarative starters compile with zero prism.design/1 violations', () => {
  const starters: DeclarativeStarterId[] = [
    'budget-guard',
    'pricing-rules',
    'eligibility-gate',
    'split-calc',
    'blank',
  ];
  for (const kind of starters) {
    const decl = createDeclarativeStarter(kind);
    const compiled = compilePrismDeclaration(decl);
    const errors = compiled.designDiagnostics.filter((d) => d.severity === 'error');
    assert.equal(
      errors.length,
      0,
      `Starter "${kind}" has prism.design/1 violations: ${JSON.stringify(errors)}`,
    );
    assert.ok(compiled.compiledCss.length > 500);
    assert.ok(compiled.compiledUiJs.length > 100);
    assert.ok(compiled.files['app/src/App.lex.tex'].includes('theorem{'));
  }
});

test('end-to-end real WASM LexLean check + Lean 4.32.2 kernel proof + binary prism/holo/6 seal', async () => {
  // Import prism-code's browser-safe build core + Lean kernel prover directly from node_modules
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

  // Prove and seal two representative multi-rule starters (formula + refusal + decision table)
  for (const kind of ['budget-guard', 'eligibility-gate'] as const) {
    const decl = createDeclarativeStarter(kind);
    const compiled = compilePrismDeclaration(decl);
    const mergedApp = {
      ...instantiateAppFoundation(foundation, decl.metadata.slug),
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
      `Starter "${kind}" failed real WASM build at stage "${result.stage}": ${result.message} :: ${JSON.stringify(result.diagnostics)} :: ${JSON.stringify(result.theorems)}`,
    );
    assert.equal(result.proved, true, `Starter "${kind}" theorems were not proved by Lean kernel`);
    assert.ok(result.holo?.bytes instanceof Uint8Array, `Starter "${kind}" produced no binary .holo`);
    assert.ok(result.holo.label.startsWith('blake3:'));
    assert.equal(result.holo.carries.proof, true);
    assert.equal(result.holo.carries.snapshot, true);
    assert.equal(result.holo.carries.sources, true);

    // Load the sealed binary prism/holo/6 archive and verify guest WASM matches evaluateDeclaredModel 1-for-1!
    const app = await loadArchive(result.holo.bytes, {});
    assert.equal(app.identities.verified, true);
    const acceptance = await app.acceptance();
    assert.equal(acceptance.failed.length, 0);
    assert.equal(acceptance.passed, decl.vectors.length);

    for (const vec of decl.vectors) {
      const wasmAnswer = new TextDecoder().decode(
        await app.dispatch(new TextEncoder().encode(vec.request)),
      );
      const fastAnswer = evaluateDeclaredModel(decl, vec.request).response;
      assert.equal(wasmAnswer, vec.expectedResponse);
      assert.equal(wasmAnswer, fastAnswer);
    }
  }
});
