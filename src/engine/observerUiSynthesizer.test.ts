import assert from 'node:assert/strict';
import test from 'node:test';
import {
  buildCanonicalWorkspaceUiTree,
  buildRootWorkspacePrism,
  buildSystemPartPrisms,
  galleryPrismId,
  solveAndVerifyUiTree,
  stepObserverSurfaceAttention,
  synthesizeObserverPrismFrame,
  tropicalFocusReachabilityClosure,
  tropicalUiBoxExtents,
  VERDANT_HOME_PRISM_ID,
  VERDANT_ROOT_PRISM_ID,
  type UiWidgetSpec,
} from './observerUiSynthesizer';

test('Observer procedural UI synthesizer solves Tropical (max,+) extents and (min,+) focus closure', () => {
  const horiz = tropicalUiBoxExtents(
    [
      [260, 540],
      [680, 500],
    ],
    8,
    true,
  );
  assert.deepEqual(horiz, [948, 540]);

  const vert = tropicalUiBoxExtents(
    [
      [680, 48],
      [680, 400],
      [680, 64],
    ],
    4,
    false,
  );
  assert.deepEqual(vert, [680, 520]);

  const closure = tropicalFocusReachabilityClosure(4, [
    [0, 1],
    [1, 2],
    [2, 3],
  ]);
  assert.equal(closure[0 * 4 + 3], 3);
  assert.equal(closure[3 * 4 + 0], 3);
});

test('Canonical VerdantUI workspace UiTree passes WCAG, touch, non-overlap, and focus compliance on desktop and mobile', () => {
  for (const [vw, vh, sidebar] of [
    [1280, 800, true],
    [390, 844, false],
  ] as const) {
    const tree = buildCanonicalWorkspaceUiTree(VERDANT_HOME_PRISM_ID, 'preview', sidebar);
    const solved = solveAndVerifyUiTree(tree, vw, vh);
    assert.equal(solved.certificate.ok, true, `Violations: ${solved.certificate.violations.join('; ')}`);
    assert.equal(solved.certificate.wcagContrastPassed, true);
    assert.equal(solved.certificate.focusGraphReachable, true);
    assert.equal(solved.certificate.siblingNonOverlapping, true);
    assert.equal(solved.certificate.touchTargetsCompliant, true);
    assert.ok(solved.layoutDigest.length >= 8);
  }
});

test('solveAndVerifyUiTree rejects contrast and unreachable-focus violations cleanly', () => {
  const broken: UiWidgetSpec[] = [
    {
      index: 0,
      id: 'root',
      slot: 'root',
      label: 'Root',
      minWidthPx: 320,
      minHeightPx: 200,
      flexWeight: 1,
      gapPx: 0,
      horizontal: false,
      contrastMilli: 4800,
      isText: false,
      interactive: false,
      focusNeighbors: [],
      children: [1, 2],
      prismId: VERDANT_ROOT_PRISM_ID,
    },
    {
      index: 1,
      id: 'low-contrast-btn',
      slot: 'prism-header',
      label: 'Low Contrast',
      minWidthPx: 100,
      minHeightPx: 48,
      flexWeight: 0,
      gapPx: 0,
      horizontal: true,
      contrastMilli: 2100,
      isText: true,
      interactive: true,
      focusNeighbors: [],
      children: [],
      prismId: VERDANT_HOME_PRISM_ID,
    },
    {
      index: 2,
      id: 'isolated-btn',
      slot: 'prism-nl-bar',
      label: 'Isolated',
      minWidthPx: 100,
      minHeightPx: 48,
      flexWeight: 1,
      gapPx: 0,
      horizontal: true,
      contrastMilli: 9000,
      isText: true,
      interactive: true,
      focusNeighbors: [],
      children: [],
      prismId: VERDANT_HOME_PRISM_ID,
    },
  ];
  const solved = solveAndVerifyUiTree(broken, 640, 480);
  assert.equal(solved.certificate.ok, false);
  assert.equal(solved.certificate.wcagContrastPassed, false);
  assert.equal(solved.certificate.focusGraphReachable, false);
});

test('VerdantUI root Prism and every system part Prism synthesize deterministically', () => {
  const parts = buildSystemPartPrisms();
  assert.ok(parts.some((p) => p.id === VERDANT_HOME_PRISM_ID));
  assert.ok(parts.some((p) => p.id === galleryPrismId('written-work')));
  assert.ok(parts.some((p) => p.id === galleryPrismId('picture')));
  const root = buildRootWorkspacePrism(parts, []);
  assert.equal(root.id, VERDANT_ROOT_PRISM_ID);
  assert.equal(root.childPrismIds.length, parts.length);

  const frame = synthesizeObserverPrismFrame({
    activePrism: parts.find((p) => p.id === galleryPrismId('written-work'))!,
    partPrisms: parts,
    userPrisms: [],
    surfaceMode: 'preview',
    viewportWidthPx: 1280,
    viewportHeightPx: 800,
    darkMode: false,
    userTyping: true,
    dwellMillis: 2400,
  });
  assert.equal(frame.layout.certificate.ok, true);
  assert.equal(frame.directive.calmPeripheralUi, true);
  assert.equal(frame.directive.tokens['--v-accent'], '#0F5132');

  const darkAttention = stepObserverSurfaceAttention({
    activePrismId: VERDANT_HOME_PRISM_ID,
    surfaceMode: 'details',
    userTyping: false,
    dwellMillis: 200,
    darkMode: true,
  });
  assert.equal(darkAttention.calmPeripheralUi, false);
  assert.equal(darkAttention.tokens['--v-accent'], '#10B981');
});
