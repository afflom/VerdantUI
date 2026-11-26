import test from 'node:test';
import assert from 'node:assert/strict';
import { VerdantCognitiveObserver } from './cognitiveObserver';
import { ObserverStreamBus } from './observerStreamBus';
import { PrismHostPool } from './prismHostPool';
import { PrismStore } from './prismStudio';
import {
  findMatchingPublishedPrism,
  humanizeToolName,
  resolveOrCreatePrismForToolCall,
} from './prismToolResolver';
import type { LlamaVTensorWorkerClient } from './workerClient';

test('humanizeToolName normalizes arbitrary LLM tool names into clean human Prism titles', () => {
  assert.equal(humanizeToolName('play_minesweeper'), 'Minesweeper');
  assert.equal(humanizeToolName('create_ambient_soundscape'), 'Ambient Soundscape');
  assert.equal(humanizeToolName('prism_healthy_habits'), 'Healthy Habits');
  assert.equal(humanizeToolName('unitConverter'), 'Unit Converter');
});

test('resolveOrCreatePrismForToolCall auto-synthesizes a missing Prism, publishes it into the active Space, and reuses it on repeat calls', async () => {
  const bus = new ObserverStreamBus();
  const store = new PrismStore();
  const observer = new VerdantCognitiveObserver();
  const fakeWorker = { cognitiveObserver: observer } as unknown as LlamaVTensorWorkerClient;
  const pool = new PrismHostPool(bus, store, observer.grants, fakeWorker);

  // 1. Model attempts a tool-call for "play_minesweeper" when NO Minesweeper Prism exists yet
  const first = await resolveOrCreatePrismForToolCall({
    toolName: 'play_minesweeper',
    args: { rows: 9, cols: 9, mines: 10 },
    latestUserText: 'Can you make a quick Minesweeper game for me?',
    activeSpace: 'personal',
    store,
    pool,
    observer,
  });

  assert.equal(first.__prismInvocation, true);
  assert.equal(first.autoCreated, true);
  assert.equal(first.prismName, 'Minesweeper');
  assert.equal(first.mediaType, 'interactive-game');
  assert.equal(first.compartmentId, 'personal');

  // Verify the auto-created Prism is genuinely published & registered in Observer ⊇ User ⊇ Prisms
  const published = store.listPublishedSummaries();
  assert.equal(published.length, 1);
  assert.equal(observer.possession.listPublishedOrgans('personal').length, 1);
  assert.ok(findMatchingPublishedPrism('play_minesweeper', published));

  // 2. Subsequent call to "minesweeper" in the same Space reuses the existing Published Prism (zero duplication!)
  const second = await resolveOrCreatePrismForToolCall({
    toolName: 'minesweeper',
    args: { action: 'status' },
    latestUserText: 'Open Minesweeper again',
    activeSpace: 'personal',
    store,
    pool,
    observer,
  });

  assert.equal(second.autoCreated, false);
  assert.equal(second.prismId, first.prismId);
  assert.equal(store.listPublishedSummaries().length, 1);

  // 3. Model attempts an audio tool-call -> automatically selects 'audio' mediaType with zero UI buttons!
  const music = await resolveOrCreatePrismForToolCall({
    toolName: 'compose_ambient_music',
    args: { bpm: 84, scale: 'pentatonic' },
    latestUserText: 'Create a relaxing ambient music soundscape',
    activeSpace: 'personal',
    store,
    pool,
    observer,
  });
  assert.equal(music.autoCreated, true);
  assert.equal(music.mediaType, 'audio');

  // 4. Strict Space isolation: a Personal Prism is NOT reused by a Work tool-call (never leaks across Spaces!)
  const workTool = await resolveOrCreatePrismForToolCall({
    toolName: 'minesweeper',
    args: { rows: 9, cols: 9, mines: 10 },
    latestUserText: 'Make a break-time Minesweeper in Work space',
    activeSpace: 'work',
    store,
    pool,
    observer,
  });
  assert.equal(workTool.autoCreated, true);
  assert.notEqual(workTool.prismId, first.prismId);
  assert.equal(workTool.compartmentId, 'work');

  pool.dispose();
});
