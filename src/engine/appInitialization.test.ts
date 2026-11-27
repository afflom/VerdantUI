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

test('Accepting consent enters the Application Initializing Screen and progresses monotonically until real runtime is ready', () => {
  // Immediately after clicking Continue (bootstrap model still active, connecting capabilities)
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
  assert.ok(connecting.overallPercent >= 15 && connecting.overallPercent < 82);

  // Loading runtime stage
  const loading = deriveAppInitializationState({
    streamingConsent: 'accepted',
    engineStatus: 'loading',
    progressPercent: 98,
    initStatusDetail: 'Initializing Qwen3 4B',
    modelId: 'stories260k-f32',
    errorMessage: null,
    storesHydrated: true,
    sessionUnlocked: false,
  });
  assert.equal(loading.visible, true);
  assert.equal(loading.steps[1]?.status, 'done');
  assert.equal(loading.steps[2]?.status, 'active');
  assert.ok(loading.overallPercent > connecting.overallPercent);

  // Real demand-first runtime mounted -> VerdantUI is usable and screen clears immediately
  const ready = deriveAppInitializationState({
    streamingConsent: 'accepted',
    engineStatus: 'ready',
    progressPercent: 100,
    initStatusDetail: '',
    modelId: 'qwen3-4b-q4_k_m',
    errorMessage: null,
    storesHydrated: true,
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
