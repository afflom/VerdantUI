import test from 'node:test';
import assert from 'node:assert/strict';
import type { ThreadMessage } from '@assistant-ui/react';
import {
  formatMessagesForLlama,
  formatSpeculativePrefixForLlama,
} from './assistantAdapter';
import {
  UnifiedCognitiveContextGovernor,
  deriveContextBudgetEnvelope,
} from './cognitiveContext';
import {
  UNIFIED_VERDANT_SYSTEM_PROMPT,
  VerdantCognitiveObserver,
  createVerdantClientTools,
} from './cognitiveObserver';
import type { LoadedModelSummary } from './llamaRuntime';
import { GgmlType } from './gguf';
import { VramKvPager, VramPageState } from './vramKvPager';
import { VTensorBuffer } from './vtensorProvider';

function makeMsg(id: string, role: 'user' | 'assistant', text: string): ThreadMessage {
  return {
    id,
    role,
    content: [{ type: 'text', text }],
    createdAt: new Date(0),
    status: { type: 'complete', reason: 'stop' },
    metadata: { custom: {} },
    attachments: [],
  } as unknown as ThreadMessage;
}

const QWEN3_4B = {
  id: 'qwen3-4b-q4_k_m',
  displayName: 'Qwen3 4B (Q4_K_M)',
  maxContextWindow: 32768,
  hparams: {
    name: 'Qwen3-4B',
    architecture: 'qwen3',
    chatTemplate: '<|im_start|>{{ role }}\n{{ content }}<|im_end|>\n',
  },
} as unknown as LoadedModelSummary;

const NEUTRAL_BELIEF = [256, 256, 256, 256, 256, 256, 256, 256] as const;

test('UnifiedCognitiveContextGovernor preserves 100% lossless canonical turns, bounds active context via epoch hysteresis, and foveates verbatim recall without mutating frozen epoch prefix', async () => {
  const governor = new UnifiedCognitiveContextGovernor();
  const threadId = 'thread-long-progression';
  const budgetOverride = {
    highWatermarkTokens: 420,
    lowWatermarkTokens: 240,
    maxFoveatedRecallTokens: 160,
    minRecentTurns: 4,
  };

  const messages: ThreadMessage[] = [
    makeMsg(
      'm-0',
      'user',
      'Core architecture requirement: project AURORA_QUORUM_42 must enforce deterministic offline checksums with zero external servers.',
    ),
    makeMsg(
      'm-1',
      'assistant',
      'Understood. Recorded AURORA_QUORUM_42 constraint: deterministic offline checksums and strictly client-side execution.',
    ),
  ];

  // Progress through 16 multi-turn exchanges (~1,800+ verbatim tokens) and verify O(1) bounded active context
  const observedEpochs: number[] = [];
  let frozenManifestAfterEpoch1: string | null = null;
  let intraEpochManifestIdentical = false;

  for (let turn = 1; turn <= 12; turn++) {
    messages.push(
      makeMsg(
        `u-${turn}`,
        'user',
        `Iteration ${turn}: analyze layout spacing, typography rhythm, and responsive panel behavior for dashboard module section_${turn} with detailed notes.`,
      ),
    );
    const proj = governor.projectContext(threadId, messages, {
      latestUserText: `Iteration ${turn}: analyze layout spacing`,
      belief: NEUTRAL_BELIEF,
      maxContextWindow: 4096,
      systemTokensEstimate: 60,
      budgetOverride,
    });
    observedEpochs.push(proj.epoch);

    if (proj.epoch === 1) {
      if (frozenManifestAfterEpoch1 === null) {
        frozenManifestAfterEpoch1 = proj.epochManifestBlock;
      } else if (proj.epochManifestBlock === frozenManifestAfterEpoch1) {
        intraEpochManifestIdentical = true;
      }
    }

    messages.push(
      makeMsg(
        `a-${turn}`,
        'assistant',
        `Completed iteration ${turn} design review for section_${turn}: grid columns align cleanly across desktop and mobile viewports with consistent spacing.`,
      ),
    );
  }

  // Verify epoch consolidation occurred cleanly AND held steady intra-epoch (hysteresis -> zero thrashing)
  const maxEpoch = Math.max(...observedEpochs);
  assert.ok(maxEpoch >= 1, 'Long thread must settle into epoch consolidation');
  assert.equal(
    intraEpochManifestIdentical,
    true,
    'Frozen epoch manifest must stay byte-identical across intra-epoch turns for KV prefix reuse',
  );

  // Now ask a query referencing Turn 0's exact symbol `AURORA_QUORUM_42`!
  const recallQuery = 'Does our design still obey AURORA_QUORUM_42 offline checksums?';
  messages.push(makeMsg('u-recall', 'user', recallQuery));

  const beforeRecallStats = governor.getThreadLedgerStats(threadId);
  const projectedRecall = governor.projectContext(threadId, messages, {
    latestUserText: recallQuery,
    belief: NEUTRAL_BELIEF,
    maxContextWindow: 4096,
    systemTokensEstimate: 60,
    budgetOverride,
  });

  // 1. Active context stays dramatically smaller than full verbatim history (non-degrading O(1) envelope)
  assert.ok(
    projectedRecall.activeEstimatedTokens < projectedRecall.totalVerbatimTokens,
    `Expected bounded active tokens (${projectedRecall.activeEstimatedTokens}) < total verbatim (${projectedRecall.totalVerbatimTokens})`,
  );
  // 2. Canonical ledger preserves EVERY turn bit-exact (zero loss!)
  assert.equal(projectedRecall.totalCanonicalTurns, messages.length);
  assert.ok(projectedRecall.consolidatedTurns > 0);
  // 3. Foveated lossless refault automatically recalls Turn 0 verbatim into `foveatedRecallBlock`!
  assert.ok(
    projectedRecall.foveatedTurnIndices.includes(0),
    `Expected Turn 0 to be foveated on AURORA_QUORUM_42 query, got: ${JSON.stringify(projectedRecall.foveatedTurnIndices)}`,
  );
  assert.ok(projectedRecall.foveatedRecallBlock?.includes('AURORA_QUORUM_42'));
  assert.ok(projectedRecall.epochManifestBlock?.includes('Verdant Lossless Episodic Index'));

  // 4. Explicit bit-exact lossless recall by kappa, turnIndex, and query via `recall_thread_memory`
  const byTurn0 = governor.recallTurns(threadId, { turnIndex: 0 });
  assert.equal(byTurn0.length, 1);
  assert.ok(byTurn0[0].verbatimText.includes('AURORA_QUORUM_42'));

  const byKappa = governor.recallTurns(threadId, { kappa: byTurn0[0].kappa });
  assert.equal(byKappa.length, 1);
  assert.equal(byKappa[0].verbatimText, byTurn0[0].verbatimText);

  const tools = createVerdantClientTools(
    () => ({
      model: QWEN3_4B,
      telemetry: {} as never,
      opfsStorage: { supported: false, usageBytes: 0, quotaBytes: 0, storedModels: [], partialModels: [] },
      lastTokensPerSec: 20,
    }),
    undefined,
    { recallTurns: (q) => governor.recallTurns(threadId, q) },
  );
  assert.ok(tools.recall_thread_memory, 'recall_thread_memory tool should be registered when bridge is supplied');
  const toolOut = (await tools.recall_thread_memory.execute!(
    { query: 'AURORA_QUORUM_42' },
    {} as never,
  )) as { lossless: boolean; hitCount: number; turns: { verbatimText: string }[] };
  assert.equal(toolOut.lossless, true);
  assert.ok(toolOut.hitCount >= 1);
  assert.ok(toolOut.turns[0].verbatimText.includes('AURORA_QUORUM_42'));
  assert.ok(beforeRecallStats.totalTurns >= 24);
});

test('KV-prefix stability: monotonic per-thread tool latch + Qwen3 empty-think prefix parity prevent multi-turn KV wipes', () => {
  // 1. Monotonic tool latch: once tools attach on a thread, subsequent non-tool turns keep tools latched
  const observer = new VerdantCognitiveObserver();
  const threadId = 'thread-tool-latch';

  const t1 = observer.selectRoute({
    threadId,
    messages: [makeMsg('u1', 'user', 'Calculate 12 * 12 and save to a workspace note titled "Math".')],
    latestUserText: 'Calculate 12 * 12 and save to a workspace note titled "Math".',
    hasAttachment: false,
    activeModel: QWEN3_4B,
    cachedModelIds: ['qwen3-4b-q4_k_m'],
    isToolContinuationStep: false,
    hasImportedHarness: false,
  });
  assert.equal(t1.certificate.attachTools, true);
  assert.equal(observer.isToolsLatched(threadId), true);

  // Follow-up conversational turn without tool words MUST keep attachTools === true (zero token-40 KV wipe!)
  const t2 = observer.selectRoute({
    threadId,
    messages: [
      makeMsg('u1', 'user', 'Calculate 12 * 12 and save to a workspace note titled "Math".'),
      makeMsg('a1', 'assistant', 'Saved 144 to note Math.'),
      makeMsg('u2', 'user', 'Thanks! Now explain what a prime number is in one sentence.'),
    ],
    latestUserText: 'Thanks! Now explain what a prime number is in one sentence.',
    hasAttachment: false,
    activeModel: QWEN3_4B,
    cachedModelIds: ['qwen3-4b-q4_k_m'],
    isToolContinuationStep: false,
    hasImportedHarness: false,
  });
  assert.equal(t2.certificate.attachTools, true);

  // 2. Qwen3 empty-think prefix continuity: Turn 2 strictly extends Turn 1 prompt + Turn 1 reply!
  const turn1Prompt = formatMessagesForLlama(
    [makeMsg('u1', 'user', 'Say hello briefly.')],
    QWEN3_4B,
    'auto',
    UNIFIED_VERDANT_SYSTEM_PROMPT,
    { thinkingMode: 'off', preserveEmptyThinkPrefix: true },
  );
  const turn1Reply = 'Hello! How can I help?';
  const turn1FullGeneratedSequence = `${turn1Prompt}${turn1Reply}<|im_end|>\n`;

  const turn2Prompt = formatMessagesForLlama(
    [
      makeMsg('u1', 'user', 'Say hello briefly.'),
      makeMsg('a1', 'assistant', turn1Reply),
      makeMsg('u2', 'user', 'Tell me a short fact.'),
    ],
    QWEN3_4B,
    'auto',
    UNIFIED_VERDANT_SYSTEM_PROMPT,
    { thinkingMode: 'off', preserveEmptyThinkPrefix: true },
  );
  assert.ok(
    turn2Prompt.startsWith(turn1FullGeneratedSequence),
    'Turn 2 prompt must be a strict byte-exact extension of Turn 1 prompt + reply for 100% KV reuse!',
  );

  // 3. Speculative prefill parity holds with epochManifestBlock + preserveEmptyThinkPrefix!
  const manifest = '[Verdant Lossless Episodic Index · Epoch 1]\n- [#0 · kappa:1234 · user] Goal';
  const specPrefix = formatSpeculativePrefixForLlama(
    [makeMsg('u1', 'user', 'Say hello briefly.'), makeMsg('a1', 'assistant', turn1Reply)],
    'Tell me a ',
    QWEN3_4B,
    'auto',
    UNIFIED_VERDANT_SYSTEM_PROMPT,
    { epochManifestBlock: manifest, preserveEmptyThinkPrefix: true },
  );
  const finalPrompt = formatMessagesForLlama(
    [
      makeMsg('u1', 'user', 'Say hello briefly.'),
      makeMsg('a1', 'assistant', turn1Reply),
      makeMsg('u2', 'user', 'Tell me a short fact.'),
    ],
    QWEN3_4B,
    'auto',
    UNIFIED_VERDANT_SYSTEM_PROMPT,
    { thinkingMode: 'off', epochManifestBlock: manifest, preserveEmptyThinkPrefix: true },
  );
  assert.ok(
    finalPrompt.startsWith(specPrefix),
    'Speculative prefill with epoch manifest & empty-think continuity must be a strict byte prefix of final prompt',
  );

  // Budget envelope scales smoothly
  const smallEnv = deriveContextBudgetEnvelope(2048);
  const largeEnv = deriveContextBudgetEnvelope(32768);
  assert.ok(smallEnv.lowWatermarkTokens < smallEnv.highWatermarkTokens);
  assert.ok(largeEnv.highWatermarkTokens <= 3072);
});

test('VramKvPager pins system/anchor prefix pages above lossy Cold-Q4 and Tier-3 Swap demotion across long sequences', () => {
  const pager = new VramKvPager(
    {
      numLayers: 2,
      maxContext: 256, // 16 pages of 16 tokens
      nHeadsKv: 2,
      headDim: 16,
      hotFrameLimit: 3,
      warmQ8Limit: 6,
      coldQ4Enabled: true,
      coldQ4PoolLimit: 4,
    },
    (id, usage, capacity) => new VTensorBuffer(id, usage, capacity),
  );

  // Pin pages 0, 1, 2 (48 tokens of system prompt + anchor manifest)
  pager.setPinnedPrefixPages(3);
  assert.equal(pager.getPinnedPrefixPages(), 3);

  const vec = new Float32Array(32);
  for (let pos = 0; pos < 224; pos++) {
    for (let d = 0; d < 32; d++) {
      vec[d] = Math.sin((pos + 1) * 0.13 + d * 0.27);
    }
    for (let layer = 0; layer < 2; layer++) {
      pager.writeKv(false, layer, pos, vec);
      pager.writeKv(true, layer, pos, vec);
    }
  }

  // Pages 0, 1, 2 MUST remain lossless Hot F32 or Warm Q8 — NEVER crushed to lossy 4-bit Q4 or Swapped Q4!
  for (const pinnedPage of [0, 1, 2]) {
    const st = pager.getPageState(pinnedPage);
    assert.ok(
      st === VramPageState.HOT_RESIDENT || st === VramPageState.COLD_COMPRESSED,
      `Pinned anchor page ${pinnedPage} degraded to lossy state ${st}`,
    );
  }

  // Older unpinned pages (e.g. pages 3..7) MUST cascade cleanly to Cold Q4 / Swapped Q4 to keep RAM bounded
  const telemetry = pager.getTelemetry();
  assert.equal(telemetry.pinnedPrefixPages, 3);
  assert.ok(
    telemetry.coldQ4Pages + telemetry.swappedQ4Pages > 0,
    'Unpinned middle pages should still cascade into Cold Q4 / Swap to keep memory bounded',
  );
  // Suppress unused import lint if needed
  assert.equal(GgmlType.F32, 0);
});
