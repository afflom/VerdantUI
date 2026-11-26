import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { gunzipSync } from 'node:zlib';
import { loadArchive } from 'prism-code';
import {
  deriveActiveStudioOrgans,
  normalizeOrganWeights,
  requiredCapabilitiesForMediaType,
  synthesizeProceduralOrganView,
  type PrismMediaType,
} from './prismMediaOrgans';
import {
  assembleProceduralPrismOps,
  buildWeightRefinementPrompt,
  buildWeightRoutingDiscoveryPrompt,
  classifyArtifactMediaType,
  parseDiscoveryAgentResponse,
  STUDIO_AGENT_SYSTEM_PREFIX,
} from './prismStudioAgent';
import {
  checkDeclaration,
  createEmptyDeclaration,
  dispatchStudioOp,
} from './verdantStudioApi';
import { compilePrismDeclaration, instantiateAppFoundation } from './prismCompiler';

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

test('Sparse Studio sub-model activation activates ONLY relevant organs for each MediaType', () => {
  const picOrgans = deriveActiveStudioOrgans('picture');
  assert.deepEqual(picOrgans, ['visual-canvas', 'palette-theme']);
  assert.equal(picOrgans.includes('game-simulation'), false);
  assert.equal(picOrgans.includes('narrative-outline'), false);

  const audioOrgans = deriveActiveStudioOrgans('audio');
  assert.deepEqual(audioOrgans, ['audio-synth', 'palette-theme']);

  const novelOrgans = deriveActiveStudioOrgans('written-work');
  assert.deepEqual(novelOrgans, ['narrative-outline', 'palette-theme']);

  const videoOrgans = deriveActiveStudioOrgans('video');
  assert.deepEqual(videoOrgans, ['visual-canvas', 'timeline-motion', 'palette-theme']);

  const gameOrgans = deriveActiveStudioOrgans('interactive-game');
  assert.deepEqual(gameOrgans, ['game-simulation', 'rule-invariants', 'palette-theme']);

  const researchOrgans = deriveActiveStudioOrgans('research-program');
  assert.ok(researchOrgans.includes('evidence-ledger'));
  assert.ok(researchOrgans.includes('privacy-compartment'));

  const toolOrgans = deriveActiveStudioOrgans('tool-organ');
  assert.ok(toolOrgans.includes('sensory-stream'));
  assert.ok(toolOrgans.includes('impulse-emitter'));
});

test('Weight-routing prompts strictly forbid JavaScript/HTML/CSS codegen and classify all 7 MediaTypes', () => {
  assert.ok(STUDIO_AGENT_SYSTEM_PREFIX.includes('NEVER write JavaScript, HTML, or CSS'));
  assert.equal(classifyArtifactMediaType("This Prism will be the video game 'Minesweeper'").mediaType, 'interactive-game');
  assert.equal(classifyArtifactMediaType('Generate a geometric emerald poster picture').mediaType, 'picture');
  assert.equal(classifyArtifactMediaType('Create an ambient pentatonic audio soundscape').mediaType, 'audio');
  assert.equal(classifyArtifactMediaType('Write a sci-fi novel with 5 chapters').mediaType, 'written-work');
  assert.equal(classifyArtifactMediaType('Build a kinetic intro video sequence').mediaType, 'video');
  assert.equal(classifyArtifactMediaType('Manage my climate research program and hypotheses').mediaType, 'research-program');
  assert.equal(classifyArtifactMediaType('Healthy Habits: protect 5pm to 6pm daily for exercise').mediaType, 'tool-organ');

  const empty = createEmptyDeclaration('Minesweeper');
  const discPrompt = buildWeightRoutingDiscoveryPrompt(
    [{ role: 'user', text: "This Prism will be the video game 'Minesweeper'" }],
    empty,
    1,
    'interactive-game',
  );
  assert.ok(discPrompt.includes('NEVER output JavaScript, HTML, or CSS'));
  assert.ok(discPrompt.includes('grid-sweep'));
});

test('Minesweeper end-to-end procedural synthesis routes JSON weights, refines board size, and passes real WASM Lean 4.32.2 proof', async () => {
  const empty = createEmptyDeclaration('Untitled Prism');

  // Turn 1: User describes Minesweeper -> Studio classifies interactive-game & asks clarifying question
  const t1 = parseDiscoveryAgentResponse(
    JSON.stringify({
      name: 'Minesweeper',
      purpose: 'Classic Minesweeper puzzle game with presets and safe first click.',
      mediaType: 'interactive-game',
      readyToBuild: false,
      weights: { mediaType: 'interactive-game', archetype: 'grid-sweep', rows: 9, cols: 9, mines: 10 },
      reply: 'Activated Game features for Minesweeper. Want classic 9x9 or a larger board?',
      quickReplies: ['Classic 9x9 — build it', '16x16 with 40 mines'],
    }),
    "This Prism will be the video game 'Minesweeper'",
    empty,
    1,
  );
  assert.equal(t1.spec.mediaType, 'interactive-game');
  assert.equal(t1.spec.readyToBuild, false);

  // Turn 2: User says "16x16 with 40 mines, build it" -> routes weights & synthesizes full Minesweeper Prism
  const t2 = parseDiscoveryAgentResponse(
    '', // even with empty/fallback model output, deterministic normalizer extracts 16x16 & 40 mines!
    'Make it 16x16 with 40 mines, build it now',
    empty,
    2,
    'interactive-game',
  );
  assert.equal(t2.spec.readyToBuild, true);
  assert.equal(t2.spec.weights.mediaType, 'interactive-game');
  if (t2.spec.weights.mediaType === 'interactive-game' && t2.spec.weights.archetype === 'grid-sweep') {
    assert.equal(t2.spec.weights.rows, 16);
    assert.equal(t2.spec.weights.cols, 16);
    assert.equal(t2.spec.weights.mines, 40);
  } else {
    assert.fail('Expected grid-sweep weights for Minesweeper');
  }

  const assembled = assembleProceduralPrismOps(empty, t2.spec, 'Make it 16x16 with 40 mines');
  const decl = assembled.finalDecl;
  assert.equal(decl.view?.authoredView, null);
  assert.ok(decl.view?.proceduralOrgan);
  assert.equal(checkDeclaration(decl).ok, true);

  // Compile & verify procedural Minesweeper view contains routed weights and zero external URLs
  const compiled = compilePrismDeclaration(decl);
  assert.ok(compiled !== null, 'Minesweeper declaration must compile cleanly');
  assert.ok(compiled.files['app/view/app.js'].includes('"rows":16'));
  assert.ok(compiled.files['app/view/app.js'].includes('"mines":40'));
  assert.equal(/https?:\/\//.test(compiled.files['app/view/index.html']), false);

  // Refinement prompt keeps pure weight routing (zero JS codegen)
  const refPrompt = buildWeightRefinementPrompt([], decl, 'Switch to 9x9 with 10 mines');
  assert.ok(refPrompt.includes('NEVER write JavaScript, HTML, or CSS'));

  // Prove & seal real binary prism/holo/6 archive with Lean 4.32.2 WASM kernel!
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
  const appFiles = {
    ...instantiateAppFoundation(foundation, decl.metadata.slug),
    ...compiled.appFiles,
  };
  const built = await buildCore.buildFromFiles({
    files: appFiles,
    lexlean: lexleanModule,
    evaluator: evaluatorBytes,
    prove: proveCore.kernelProver(kernel, { record: true }),
  });
  assert.equal(built.ok, true, `buildFromFiles failed: ${built.message} :: ${JSON.stringify(built.diagnostics)}`);
  assert.equal(built.proved, true);
  assert.ok(built.holo?.bytes instanceof Uint8Array);
  const loaded = await loadArchive(built.holo.bytes, {});
  assert.equal(loaded.identities.verified, true);
});

test('All 7 MediaTypes declare real capability requirements and synthesize cleanly without LLM JS', () => {
  assert.deepEqual(requiredCapabilitiesForMediaType('picture'), ['image-creation', 'vision']);
  assert.deepEqual(requiredCapabilitiesForMediaType('audio'), ['audio', 'language']);
  assert.deepEqual(requiredCapabilitiesForMediaType('video'), ['image-creation', 'vision', 'audio']);
  assert.deepEqual(requiredCapabilitiesForMediaType('interactive-game'), ['language', 'vision']);

  const mediaTypes: readonly PrismMediaType[] = [
    'picture',
    'audio',
    'written-work',
    'video',
    'interactive-game',
    'research-program',
    'tool-organ',
  ];
  for (const mt of mediaTypes) {
    const empty = createEmptyDeclaration(`Test ${mt}`);
    const spec = parseDiscoveryAgentResponse('', `Build ${mt} Prism now`, empty, 2, mt).spec;
    const { finalDecl } = assembleProceduralPrismOps(empty, spec, `Build ${mt} Prism now`);
    const primaryReq = finalDecl.requirements[0]!.id;
    const weights = normalizeOrganWeights(mt, undefined, `Sample ${mt}`, `Test ${mt}`);
    const routed = dispatchStudioOp(finalDecl, {
      op: 'route-prism-weights',
      mediaType: mt,
      weights,
      citation: { kind: 'req', requirementId: primaryReq },
    });
    assert.equal(routed.ok, true, `Failed routing ${mt}: ${routed.ok ? '' : routed.message}`);
    const synth = synthesizeProceduralOrganView(routed.declaration, routed.declaration.view!.proceduralOrgan!);
    assert.ok(synth.indexHtml.includes('id="organ-root"'));
    assert.ok(synth.appJs.includes('zero LLM JS'));
  }
});
