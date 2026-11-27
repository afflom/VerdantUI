import test from 'node:test';
import assert from 'node:assert/strict';
import {
  deriveAppInitializationState,
  formatInitStatusLine,
  isVerdantUsableForConsent,
} from './appInitialization';

test('First-time visitors start directly on the Splash Initialization Screen in consent phase (returning users bypass consent)', () => {
  const firstVisit = deriveAppInitializationState({
    streamingConsent: 'pending',
    engineStatus: 'ready',
    progressPercent: 100,
    initStatusDetail: '',
    modelId: 'stories260k-f32',
    errorMessage: null,
    storesHydrated: true,
    sessionUnlocked: false,
  });
  assert.equal(firstVisit.visible, true);
  assert.equal(firstVisit.phase, 'consent');
  assert.equal(firstVisit.canSwitchOffline, true);

  // Returning user with saved accepted consent NEVER sees consent phase
  const returningBoot = deriveAppInitializationState({
    streamingConsent: 'accepted',
    engineStatus: 'downloading',
    progressPercent: 10,
    initStatusDetail: 'Checking local cache',
    modelId: null,
    errorMessage: null,
    storesHydrated: false,
    sessionUnlocked: false,
  });
  assert.equal(returningBoot.visible, true);
  assert.equal(returningBoot.phase, 'initializing');
});

test('Accepting consent enters the Application Initializing Screen and completes full OPFS + context warming before unlocking for instant TTFT', () => {
  // Stage 2a: Connecting header/manifest
  const connecting = deriveAppInitializationState({
    streamingConsent: 'accepted',
    engineStatus: 'downloading',
    progressPercent: 25,
    initStatusDetail: 'Preparing on-demand model cache',
    modelId: 'stories260k-f32',
    errorMessage: null,
    storesHydrated: true,
    sessionUnlocked: false,
  });
  assert.equal(connecting.visible, true);
  assert.equal(connecting.phase, 'initializing');
  assert.equal(connecting.canSwitchOffline, true);
  assert.equal(connecting.steps[0]?.status, 'done');
  assert.equal(connecting.steps[1]?.status, 'active');
  assert.equal(connecting.steps[2]?.status, 'pending');

  // Stage 2b: Streaming core capabilities with live speed & ETA (MUST stay on splash until warm!)
  const streaming = deriveAppInitializationState({
    streamingConsent: 'accepted',
    engineStatus: 'ready',
    progressPercent: 100,
    initStatusDetail: '',
    modelId: 'qwen3-4b-q4_k_m',
    errorMessage: null,
    storesHydrated: true,
    prewarmActive: true,
    prewarmPercent: 50,
    prewarmPhase: 'streaming',
    prewarmBytesPerSec: 45 * 1024 * 1024,
    prewarmEtaSeconds: 28,
    contextWarmed: false,
    sessionUnlocked: false,
  });
  assert.equal(streaming.visible, true);
  assert.equal(streaming.steps[1]?.status, 'active');
  assert.ok(streaming.statusLine.includes('45.0 MB/s'));
  assert.ok(streaming.overallPercent > connecting.overallPercent);

  // Stage 3: Warming WASM memory & system-prompt KV context
  const warmingCtx = deriveAppInitializationState({
    streamingConsent: 'accepted',
    engineStatus: 'ready',
    progressPercent: 100,
    initStatusDetail: '',
    modelId: 'qwen3-4b-q4_k_m',
    errorMessage: null,
    storesHydrated: true,
    prewarmActive: true,
    prewarmPercent: 100,
    prewarmPhase: 'warming-context',
    contextWarmed: false,
    sessionUnlocked: false,
  });
  assert.equal(warmingCtx.visible, true);
  assert.equal(warmingCtx.steps[1]?.status, 'done');
  assert.equal(warmingCtx.steps[2]?.status, 'active');
  assert.ok(warmingCtx.overallPercent > streaming.overallPercent);

  // Ready: OPFS + WASM layers + LM-head + system-prompt KV prefix are ALL warm -> instant Turn 1 TTFT!
  const ready = deriveAppInitializationState({
    streamingConsent: 'accepted',
    engineStatus: 'ready',
    progressPercent: 100,
    initStatusDetail: '',
    modelId: 'qwen3-4b-q4_k_m',
    errorMessage: null,
    storesHydrated: true,
    prewarmActive: false,
    prewarmPercent: 100,
    prewarmPhase: 'ready',
    contextWarmed: true,
    sessionUnlocked: false,
  });
  assert.equal(ready.visible, false);
  assert.equal(ready.phase, 'ready');
  assert.equal(ready.overallPercent, 100);
});

test('Initializing screen surfaces clean error recovery with Retry and Offline Only options when network setup fails', () => {
  const errView = deriveAppInitializationState({
    streamingConsent: 'accepted',
    engineStatus: 'ready',
    progressPercent: 0,
    initStatusDetail: 'Model error',
    modelId: 'stories260k-f32',
    errorMessage: 'Failed to fetch remote header.',
    storesHydrated: true,
    sessionUnlocked: false,
  });
  assert.equal(errView.visible, true);
  assert.equal(errView.phase, 'error');
  assert.equal(errView.canRetry, true);
  assert.equal(errView.canSwitchOffline, true);
  assert.equal(errView.steps[1]?.status, 'error');
});

test('Once sessionUnlocked is true, mid-session capability switches do NOT re-open the full-screen initializing overlay', () => {
  const midSession = deriveAppInitializationState({
    streamingConsent: 'accepted',
    engineStatus: 'downloading',
    progressPercent: 40,
    initStatusDetail: 'Connecting Qwen3 8B',
    modelId: 'qwen3-4b-q4_k_m',
    errorMessage: null,
    storesHydrated: true,
    sessionUnlocked: true,
  });
  assert.equal(midSession.visible, false);
  assert.equal(midSession.phase, 'ready');
});

test('Offline-only mode becomes usable as soon as workspace stores and local bootstrap runtime are ready', () => {
  assert.equal(
    isVerdantUsableForConsent({
      streamingConsent: 'paused',
      engineStatus: 'ready',
      modelId: 'stories260k-f32',
      storesHydrated: false,
    }),
    false,
  );
  assert.equal(
    isVerdantUsableForConsent({
      streamingConsent: 'paused',
      engineStatus: 'ready',
      modelId: 'stories260k-f32',
      storesHydrated: true,
    }),
    true,
  );
  const statusLine = formatInitStatusLine('Preparing on-demand model cache', 'downloading', 45);
  assert.equal(statusLine, 'Connecting capabilities to your browser…');
});
