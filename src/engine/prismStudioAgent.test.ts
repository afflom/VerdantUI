import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { gunzipSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';
import {
  checkDeclaration,
  createUndefinedDeclaration,
  dispatchStudioOp,
  type PrismDeclaration,
  type StudioApiOp,
} from './verdantStudioApi';
import { evaluateVerdantRouting } from './verdantStandardsRouter';
import { compilePrismDeclaration, instantiateAppFoundation } from './prismCompiler';
import { buildHoloviewSandboxDocument } from './prismSandboxHtml';
import {
  STUDIO_AGENT_SYSTEM_PREFIX,
  assembleCompleteAuthoredPrismOps,
  buildAgentCoreModelOps,
  buildDiscoveryAgentPrompt,
  buildInteractiveAppPrompt,
  buildRefineInteractiveAppPrompt,
  classifyIdeaRealizationMode,
  extractAuthoredAppFromModelOutput,
  extractCleanIdeaTitle,
  parseDiscoveryAgentResponse,
  validateAuthoredJsSyntax,
} from './prismStudioAgent';
import { PrismStore } from './prismStudio';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PRISM_CODE_ROOT = path.resolve(__dirname, '../../node_modules/prism-code');

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

function applyOps(start: PrismDeclaration, ops: readonly StudioApiOp[]): PrismDeclaration {
  const res = dispatchStudioOp(start, { op: 'batch', ops });
  assert.equal(res.ok, true, res.ok ? '' : `Batch failed [${res.code}]: ${res.message}`);
  return res.declaration;
}

test('classifyIdeaRealizationMode distinguishes open-ended games/apps from numeric policies', () => {
  assert.equal(
    classifyIdeaRealizationMode("This Prism will be the video game 'Minesweeper'").mode,
    'interactive-app',
  );
  assert.equal(
    classifyIdeaRealizationMode('Build a Pomodoro focus timer with customizable sessions').mode,
    'interactive-app',
  );
  assert.equal(
    classifyIdeaRealizationMode('Make a Kanban board for my projects').mode,
    'interactive-app',
  );
  assert.equal(
    classifyIdeaRealizationMode('Check if a new expense fits my monthly budget, warn when over, and refuse zero amounts.').mode,
    'structured-model',
  );
  assert.equal(extractCleanIdeaTitle("This Prism will be the video game 'Minesweeper'"), 'Minesweeper');
});

test('extractAuthoredAppFromModelOutput parses fenced html/css/js, single-file HTML, and strips remote URLs', () => {
  const fenced = `
Here is your app:
\`\`\`html
<main id="game"><h1>Minesweeper</h1><div id="grid"></div></main>
\`\`\`
\`\`\`css
#grid { display: grid; gap: 4px; }
\`\`\`
\`\`\`js
const grid = document.getElementById('grid');
if (grid) grid.textContent = 'Ready';
\`\`\`
`;
  const res1 = extractAuthoredAppFromModelOutput(fenced);
  assert.equal(res1.ok, true);
  if (res1.ok) {
    assert.ok(res1.view.html.includes('id="grid"'));
    assert.ok(res1.view.css.includes('display: grid'));
    assert.ok(res1.view.js.includes("getElementById('grid')"));
  }

  const singleFile = `<!doctype html>
<html>
<head>
  <link rel="stylesheet" href="https://evil.example.com/cdn.css">
  <style>.board { padding: 8px; }</style>
</head>
<body>
  <script src="https://evil.example.com/track.js"></script>
  <div class="board" id="board">Play</div>
  <script>document.getElementById('board').dataset.ok = '1';</script>
</body>
</html>`;
  const res2 = extractAuthoredAppFromModelOutput(singleFile);
  assert.equal(res2.ok, true);
  if (res2.ok) {
    assert.ok(!res2.view.html.includes('evil.example.com'));
    assert.ok(res2.view.css.includes('.board'));
    assert.ok(res2.view.js.includes("dataset.ok = '1'"));
  }

  const bad = extractAuthoredAppFromModelOutput('Sorry, I cannot write code.');
  assert.equal(bad.ok, false);
});

test('admit-authored-view enforces citation and rejects external remote URLs', () => {
  let decl = createUndefinedDeclaration();
  decl = applyOps(decl, [
    {
      op: 'admit-requirement',
      requirement: {
        id: 'REQ-01',
        statement: 'Play Minesweeper offline.',
        stakeholder: 'Player',
        origin: 'identity',
        evidence: 'Creator request',
      },
    },
  ]);

  const badRemote = dispatchStudioOp(decl, {
    op: 'admit-authored-view',
    authoredView: {
      html: '<script src="https://cdn.example.com/x.js"></script><div id="app">Hi</div>',
      css: '',
      js: 'console.log("ok");',
      authorMode: 'studio-agent',
    },
    citation: { kind: 'req', requirementId: 'REQ-01' },
  });
  assert.equal(badRemote.ok, false);
  if (!badRemote.ok) assert.equal(badRemote.code, 'view.external-asset-refused');

  const badCitation = dispatchStudioOp(decl, {
    op: 'admit-authored-view',
    authoredView: {
      html: '<div id="app">Play</div>',
      css: '.x { color: red; }',
      js: 'document.getElementById("app").textContent = "Ready";',
      authorMode: 'studio-agent',
    },
    citation: { kind: 'req', requirementId: 'REQ-99' },
  });
  assert.equal(badCitation.ok, false);
});

test('End-to-end Verdant Studio Agent Minesweeper flow compiles, proves in real WASM Lean 4.32.2, seals HOLOVIEW, and round-trips through loadArchive', async () => {
  let decl = createUndefinedDeclaration();

  // Turn 1: Discovery prompt never exposes internal Verdant constructs
  const discoveryPrompt = buildDiscoveryAgentPrompt(
    [{ role: 'user', text: "This Prism will be the video game 'Minesweeper'" }],
    decl,
    1,
  );
  assert.ok(discoveryPrompt.includes('Minesweeper'));

  const mockTurn1Json = `\`\`\`json
{
  "name": "Minesweeper",
  "purpose": "Classic playable Minesweeper grid game with mine avoidance, right-click flagging, timer, and safe first click.",
  "domain": "games-media",
  "audience": "personal",
  "dataClassification": "none",
  "jurisdictions": ["none"],
  "usesLocalAi": false,
  "involvesPayments": false,
  "mode": "interactive-app",
  "readyToBuild": false,
  "stakeholders": [{"name": "Player", "role": "consumer", "concern": "Play a responsive, fair Minesweeper game offline."}],
  "inputs": [
    {"name": "rows", "label": "Board Rows", "kind": "nat", "min": 4, "max": 30, "defaultValue": 9},
    {"name": "cols", "label": "Board Columns", "kind": "nat", "min": 4, "max": 30, "defaultValue": 9},
    {"name": "mines", "label": "Mines", "kind": "nat", "min": 1, "max": 200, "defaultValue": 10},
    {"name": "hitMine", "label": "Hit Mine", "kind": "bool", "defaultValue": 0}
  ],
  "refusals": [
    {"statement": "Refuse a board with zero rows.", "field": "rows", "op": "eq", "value": 0, "failureText": "Board rows must be at least 1."},
    {"statement": "Refuse a board with zero mines.", "field": "mines", "op": "eq", "value": 0, "failureText": "Mine count must be at least 1."}
  ],
  "rules": [
    {"statement": "Lose immediately when a mine is triggered.", "field": "hitMine", "op": "eq", "value": 1, "outcome": "Game Over", "tone": "destructive"},
    {"statement": "Active Minesweeper game.", "outcome": "Playing", "tone": "brand"}
  ],
  "reply": "I anchored Minesweeper! Would you like Easy/Medium/Hard presets with right-click flagging and a timer?",
  "quickReplies": ["Yes, build it now!", "Simple 9x9 only"]
}
\`\`\``;

  const turn1 = parseDiscoveryAgentResponse(
    mockTurn1Json,
    "This Prism will be the video game 'Minesweeper'",
    decl,
    1,
  );
  assert.equal(turn1.usedFallback, false);
  assert.equal(turn1.spec.name, 'Minesweeper');
  assert.equal(turn1.spec.readyToBuild, false);

  const coreOps = buildAgentCoreModelOps(decl, turn1.spec, "This Prism will be the video game 'Minesweeper'");
  decl = applyOps(decl, coreOps);
  assert.equal(decl.metadata.name, 'Minesweeper');
  assert.equal(decl.encoding?.fields.length, 4);

  // Turn 2: User answers -> readyToBuild flips true -> agent synthesizes playable Minesweeper HTML/CSS/JS
  const appPrompt = buildInteractiveAppPrompt(
    [
      { role: 'user', text: "This Prism will be the video game 'Minesweeper'" },
      { role: 'studio', text: turn1.spec.reply },
      { role: 'user', text: 'Yes, Easy/Medium/Hard with flagging, timer, and personal offline play!' },
    ],
    decl,
  );
  assert.ok(appPrompt.includes('globalThis.prism.invoke'));

  const mockMinesweeperOutput = `
\`\`\`html
<main class="ms-app">
  <header class="ms-bar">
    <h1>Minesweeper</h1>
    <div class="ms-controls">
      <select id="ms-diff" aria-label="Difficulty">
        <option value="9,9,10">Easy (9x9)</option>
        <option value="16,16,40">Medium (16x16)</option>
      </select>
      <span id="ms-mines">Mines: 10</span>
      <span id="ms-status">Playing</span>
      <button id="ms-reset" type="button">New Game</button>
    </div>
  </header>
  <div id="ms-board" class="ms-board" role="grid" aria-label="Minesweeper board"></div>
</main>
\`\`\`
\`\`\`css
.ms-app { max-width: 680px; margin: 0 auto; padding: 16px; }
.ms-bar { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 12px; }
.ms-board { display: grid; gap: 3px; background: var(--v-surface); padding: 8px; border: 1px solid var(--v-border); border-radius: 10px; }
.ms-cell { width: 30px; height: 30px; border: 1px solid var(--v-border); background: var(--v-surface-alt); color: var(--v-text); font-weight: 700; cursor: pointer; }
\`\`\`
\`\`\`js
const boardEl = document.getElementById('ms-board');
const statusEl = document.getElementById('ms-status');
const resetBtn = document.getElementById('ms-reset');
let rows = 9, cols = 9, mines = 10, gameOver = false;
async function syncModel(hitMine) {
  try {
    const verdict = await globalThis.prism?.invoke(rows + ' ' + cols + ' ' + mines + ' ' + (hitMine ? 1 : 0));
    if (verdict && statusEl) statusEl.textContent = verdict;
  } catch (_) {}
}
function startGame() {
  gameOver = false;
  if (!boardEl) return;
  boardEl.style.gridTemplateColumns = 'repeat(' + cols + ', 30px)';
  boardEl.innerHTML = '';
  for (let i = 0; i < rows * cols; i++) {
    const btn = document.createElement('button');
    btn.className = 'ms-cell';
    btn.type = 'button';
    btn.addEventListener('click', () => {
      if (gameOver) return;
      btn.textContent = '1';
      syncModel(false);
    });
    boardEl.appendChild(btn);
  }
  syncModel(false);
}
resetBtn?.addEventListener('click', startGame);
startGame();
\`\`\`
`;

  const extracted = extractAuthoredAppFromModelOutput(mockMinesweeperOutput);
  assert.equal(extracted.ok, true);
  if (!extracted.ok) return;

  const assembled = assembleCompleteAuthoredPrismOps(
    decl,
    null,
    extracted.view,
    'Yes, Easy/Medium/Hard with flagging, timer, and personal offline play!',
  );
  decl = applyOps(decl, assembled.ops);

  const routing = evaluateVerdantRouting(decl);
  const chk = checkDeclaration(decl);
  assert.equal(chk.ok, true, `Check diagnostics: ${JSON.stringify(chk.diagnostics)}`);
  assert.equal(routing.readyToPreview, true);
  assert.equal(routing.readyToSeal, true, `Stages: ${JSON.stringify(routing.stages)}`);
  assert.equal(decl.view?.viewpoint, 'interactive-app');
  assert.ok(decl.view?.authoredView?.html.includes('id="ms-board"'));

  // Refinement prompt includes current code & error cleanly
  const refinePrompt = buildRefineInteractiveAppPrompt(
    [{ role: 'user', text: 'Add chord-reveal on double click' }],
    decl,
    decl.view!.authoredView!,
    'Add chord-reveal on double click',
    null,
  );
  assert.ok(refinePrompt.includes('ms-board'));
  assert.ok(refinePrompt.includes('Add chord-reveal on double click'));

  // Compile declaration -> emits view/index.html, view/styles.css, view/app.js with 0 design errors
  const compiled = compilePrismDeclaration(decl);
  assert.ok(compiled !== null);
  assert.equal(compiled.designDiagnostics.length, 0);
  assert.ok(compiled.appFiles['view/styles.css']?.includes('.ms-board'));
  assert.ok(compiled.appFiles['view/index.html']?.includes('href="styles.css"'));
  assert.ok(!compiled.appFiles['view/index.html']?.includes('href="prism.css"'));

  // Sandbox HTML builder preserves authored CSS and strips styles.css link cleanly
  const sandboxDoc = buildHoloviewSandboxDocument({
    prismId: 'ms-test',
    channelId: 'ch-1',
    digest: 'sha256:test',
    indexHtml: compiled.appFiles['view/index.html'],
    prismCss: compiled.compiledCss,
    prismUiJs: compiled.compiledUiJs,
    appJs: compiled.appFiles['view/app.js'],
    darkMode: true,
  });
  assert.ok(sandboxDoc.includes('.ms-board'));
  assert.ok(sandboxDoc.includes('id="ms-board"'));
  assert.ok(!sandboxDoc.includes('href="styles.css"'));

  // REAL WASM LexLean + Lean 4.32.2 kernel proof & binary prism/holo/6 seal + loadArchive round-trip!
  const buildCore = await import(path.join(PRISM_CODE_ROOT, 'src/journey/build-core.mjs'));
  const modelProve = await import(path.join(PRISM_CODE_ROOT, 'src/journey/model-prove.mjs'));
  const proveCore = await import(path.join(PRISM_CODE_ROOT, 'src/journey/prove-core.mjs'));
  const prismIndex = await import(path.join(PRISM_CODE_ROOT, 'src/index.mjs'));
  const lexleanGz = fs.readFileSync(
    path.join(PRISM_CODE_ROOT, 'wasm/pinned/4f/302c18e02a46b0d015c57bd389a0754b301ec876aa7a637f786c4562274ecb'),
  );
  const lexleanModule = await WebAssembly.compile(gunzipSync(lexleanGz));
  const evaluatorBytes = new Uint8Array(
    fs.readFileSync(path.join(PRISM_CODE_ROOT, 'src/journey/guest/evaluator.wasm')),
  );
  const kernel = await modelProve.openModelKernel();
  const foundation = loadFoundationFiles();

  const built = await buildCore.buildFromFiles({
    files: { ...instantiateAppFoundation(foundation, decl.metadata.slug), ...compiled.appFiles },
    lexlean: lexleanModule,
    evaluator: evaluatorBytes,
    prove: proveCore.kernelProver(kernel, { record: true }),
  });
  assert.equal(built.ok, true, `Minesweeper Prism failed WASM proof at ${built.stage}: ${built.message}`);
  assert.equal(built.proved, true);
  assert.ok(built.holo && built.holo.bytes.length > 0);
  assert.equal(built.holo.page, 'authored');

  // Verify binary .holo round-trip carries Minesweeper's authored view files intact!
  const loaded = await prismIndex.loadArchive(built.holo.bytes, {});
  assert.equal(loaded.identities.verified, true);
  const viewFiles = loaded.view?.files;
  assert.ok(viewFiles instanceof Map);
  const dec = new TextDecoder();
  assert.ok(dec.decode(viewFiles.get('index.html')).includes('id="ms-board"'));
  assert.ok(dec.decode(viewFiles.get('styles.css')).includes('.ms-board'));
  assert.ok(dec.decode(viewFiles.get('app.js')).includes('startGame'));
  assert.ok(viewFiles.has('prism.js'));
});

test('Performance & stability: WeakMap memoization, shared KV prefix, zero-exec JS syntax gate, and unified model cohesion', async () => {
  // 1. Zero-execution JS syntax gate rejects truncated/invalid JS without executing side effects
  let sideEffectTriggered = false;
  (globalThis as Record<string, unknown>).__verdantSyntaxGateProbe = () => {
    sideEffectTriggered = true;
  };
  const validCheck = validateAuthoredJsSyntax('globalThis.__verdantSyntaxGateProbe(); const x = 42;');
  assert.equal(validCheck.ok, true);
  assert.equal(sideEffectTriggered, false, 'Syntax validator must NEVER execute code');
  delete (globalThis as Record<string, unknown>).__verdantSyntaxGateProbe;

  const truncated = extractAuthoredAppFromModelOutput(`
\`\`\`html
<div id="game">Minesweeper</div>
\`\`\`
\`\`\`css
#game { padding: 12px; }
\`\`\`
\`\`\`js
function revealCell(r, c) {
  if (r < 0) {
    return;
\`\`\`
`);
  assert.equal(truncated.ok, false);
  if (!truncated.ok) {
    assert.ok(/syntax error/i.test(truncated.reason));
  }

  // 2. WeakMap memoization returns identical cached instances for immutable declarations and invalidates on mutation
  const store = new PrismStore();
  const created = await store.createDeclarative('budget-guard');
  const decl1 = created.declaration!;
  const routeA = evaluateVerdantRouting(decl1);
  const routeB = evaluateVerdantRouting(decl1);
  assert.strictEqual(routeA, routeB, 'evaluateVerdantRouting must return WeakMap-cached result');

  const compA = compilePrismDeclaration(decl1);
  const compB = compilePrismDeclaration(decl1);
  assert.strictEqual(compA, compB, 'compilePrismDeclaration must return WeakMap-cached result');

  // 3. KV-cache prefix sharing across Discovery, Build, and Refinement prompts
  const dPrompt = buildDiscoveryAgentPrompt([{ role: 'user', text: 'Minesweeper' }], decl1, 1);
  const bPrompt = buildInteractiveAppPrompt([{ role: 'user', text: 'Minesweeper' }], decl1);
  const rPrompt = buildRefineInteractiveAppPrompt(
    [{ role: 'user', text: 'Add timer' }],
    decl1,
    { html: '<div id="ms"></div>', css: '', js: 'const a = 1;', authorMode: 'studio-agent' },
    'Add timer',
  );
  assert.ok(dPrompt.startsWith(STUDIO_AGENT_SYSTEM_PREFIX));
  assert.ok(bPrompt.startsWith(STUDIO_AGENT_SYSTEM_PREFIX));
  assert.ok(rPrompt.startsWith(STUDIO_AGENT_SYSTEM_PREFIX));

  // 4. Unified viewpoint <-> authoredView invariant: switching to structured viewpoint clears stale authoredView
  const admitted = await store.applyStudioOp(created.id, {
    op: 'admit-authored-view',
    authoredView: {
      html: '<main id="ms-app">Minesweeper</main>',
      css: '#ms-app { padding: 8px; }',
      js: 'document.getElementById("ms-app")?.setAttribute("data-ready", "1");',
      authorMode: 'studio-agent',
    },
    citation: { kind: 'req', requirementId: 'REQ-01' },
  });
  assert.equal(admitted.ok, true);
  if (!admitted.ok) return;
  assert.equal(admitted.project.declaration!.view?.viewpoint, 'interactive-app');
  assert.ok(admitted.project.declaration!.view?.authoredView);
  assert.notStrictEqual(compilePrismDeclaration(admitted.project.declaration!), compA);

  const switched = await store.applyStudioOp(created.id, {
    op: 'synthesize-view-from-system',
    viewpoint: 'calculator',
  });
  assert.equal(switched.ok, true);
  if (!switched.ok) return;
  assert.equal(switched.project.declaration!.view?.viewpoint, 'calculator');
  assert.equal(switched.project.declaration!.view?.authoredView ?? null, null);

  // 5. Persistent unified conversation history on VerdantPrismProject.messages
  const updatedProj = await store.appendMessage(created.id, {
    role: 'assistant',
    content: 'Built Minesweeper!',
    badges: ['Concept: Minesweeper', 'Built interactive Minesweeper'],
    needsLocalModel: false,
  });
  assert.ok(updatedProj !== null);
  assert.equal(updatedProj!.messages.length, 1);
  assert.deepEqual(updatedProj!.messages[0].badges, ['Concept: Minesweeper', 'Built interactive Minesweeper']);
});
