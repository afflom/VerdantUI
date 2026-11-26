import test from 'node:test';
import assert from 'node:assert/strict';
import {
  assembleProceduralPrismOps,
  buildAgentCoreModelOps,
  buildDiscoveryAgentPrompt,
  buildWeightRefinementPrompt,
  classifyIdeaRealizationMode,
  extractCleanIdeaTitle,
  parseDiscoveryAgentResponse,
  STUDIO_AGENT_SYSTEM_PREFIX,
} from './prismStudioAgent';
import {
  checkDeclaration,
  createEmptyDeclaration,
  dispatchStudioOp,
} from './verdantStudioApi';
import { PrismStore } from './prismStudio';

test('classifyIdeaRealizationMode routes games, media, and tools to procedural organs without hardcoded templates', () => {
  const ms = classifyIdeaRealizationMode("This Prism will be the video game 'Minesweeper'");
  assert.equal(ms.mode, 'procedural-organ');
  assert.equal(ms.mediaType, 'interactive-game');
  assert.equal(extractCleanIdeaTitle("This Prism will be the video game 'Minesweeper'"), 'Minesweeper');

  const budget = classifyIdeaRealizationMode('Check if a new expense fits my monthly budget and refuse zero amounts');
  assert.equal(budget.mode, 'structured-model');
});

test('STUDIO_AGENT_SYSTEM_PREFIX is shared byte-exact across discovery and refinement turns for KV reuse', () => {
  const decl = createEmptyDeclaration('Minesweeper');
  const p1 = buildDiscoveryAgentPrompt([{ role: 'user', text: 'Minesweeper' }], decl, 1);
  const p2 = buildWeightRefinementPrompt([{ role: 'user', text: 'Minesweeper' }], decl, 'Make it 16x16');
  assert.ok(p1.startsWith(STUDIO_AGENT_SYSTEM_PREFIX));
  assert.ok(p2.startsWith(STUDIO_AGENT_SYSTEM_PREFIX));
});

test('Persistent chat transcript & viewpoint cohesion work cleanly with procedural organs in PrismStore', async () => {
  const store = new PrismStore();
  const created = await store.createDeclarative('blank');
  await store.appendMessage(created.id, {
    role: 'user',
    content: "This Prism will be the video game 'Minesweeper'",
  });

  const decl = created.declaration!;
  const { spec } = parseDiscoveryAgentResponse('', 'Build classic Minesweeper now', decl, 2, 'interactive-game');
  const { ops } = assembleProceduralPrismOps(decl, spec, 'Build classic Minesweeper now');
  const res = await store.applyStudioOp(created.id, { op: 'batch', ops });
  assert.equal(res.ok, true);
  assert.ok(res.project.declaration!.view?.proceduralOrgan);
  assert.equal(res.project.declaration!.view?.authoredView, null);
  assert.equal(checkDeclaration(res.project.declaration!).ok, true);
  assert.equal(res.project.messages.length, 1);

  // Switching to a structured viewpoint clears proceduralOrgan cleanly
  const switched = await store.applyStudioOp(created.id, {
    op: 'synthesize-view-from-system',
    viewpoint: 'decision-form',
  });
  assert.equal(switched.ok, true);
  assert.equal(switched.project.declaration!.view?.proceduralOrgan ?? null, null);
  assert.equal(switched.project.declaration!.view?.authoredView ?? null, null);

  // Direct core-model ops populate clean requirements & encoding
  const fresh = createEmptyDeclaration('Habit Guard');
  const habitSpec = parseDiscoveryAgentResponse('', 'Healthy Habits: protect 5pm to 6pm daily', fresh, 1, 'tool-organ').spec;
  const coreOps = buildAgentCoreModelOps(fresh, habitSpec, 'Healthy Habits');
  let working = fresh;
  for (const op of coreOps) {
    const r = dispatchStudioOp(working, op);
    assert.equal(r.ok, true);
    working = r.declaration;
  }
  assert.ok(working.requirements.length >= 1);
  assert.ok(working.encoding && working.encoding.fields.length >= 1);
});
