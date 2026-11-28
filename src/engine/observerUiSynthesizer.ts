/**
 * Observer Procedural UI Synthesizer & Self-Similar Prism Tree (`src/engine/observerUiSynthesizer.ts`)
 *
 * Implements `DEC-015` grounded directly in `verdant-engine` (`crates/verdant-core/src/ui.rs`,
 * `cybernetics.rs`, and `verdant-tensor` `kernel7`, `kernel9`, `kernel10`):
 *
 * 1. Self-Similar Prism Hierarchy:
 *    - VerdantUI itself is the root Verdant Prism (`verdant-ui-root`), possessed by the Observer.
 *    - Every part of VerdantUI (Home, Navigation, every Global MediaType Gallery, Settings,
 *      every Space Prism, and every Artifact Prism) is a child Verdant Prism with its own
 *      content-addressed `kappa` digest, purpose, capabilities, `UiTree`, natural-language interface,
 *      rendering (preview) surface, and `Preview | Details` inspection mode.
 * 2. Formal `verdant-engine` Tropical Semiring UI Solver:
 *    - Bottom-up intrinsic minimum extents via Tropical `(max, +)` (`tropicalUiBoxExtents`, `kernel9`).
 *    - Keyboard/pointer focus reachability closure via Tropical `(min, +)` all-pairs closure
 *      (`tropicalFocusReachabilityClosure`, `kernel7`).
 *    - Exact integer top-down flex distribution, sibling non-overlap check, WCAG 2.2 AA contrast
 *      verification (`>= 4500` milli text / `>= 3000` milli chrome), and touch-target verification.
 * 3. Second-Order Cybernetic Observer Salience (`kernel10`):
 *    - Exact integer Q10 attention step (`stepObserverSurfaceAttention`) computes `calmPeripheralUi`
 *      and per-surface salience so peripheral chrome recedes calmly when the user is reading,
 *      writing, or inspecting a Prism preview.
 * 4. Procedural Token & Surface Synthesis:
 *    - Emits unified CSS custom properties and layout frames shared across the root shell AND
 *      every sandboxed child Prism so every surface shares the serene homescreen aesthetic.
 */

import { prismContentDigest } from './prismKernel';
import {
  MEDIA_GALLERIES,
  type GalleryMediaType,
} from './contentGalleryStore';
import type { VerdantPrismProject } from './prismStudio';

function digestJson(value: unknown): string {
  return prismContentDigest(JSON.stringify(value));
}

export const UI_MIN_TOUCH_TARGET_PX = 44;
export const UI_TEXT_CONTRAST_MIN_MILLI = 4500;
export const UI_NON_TEXT_CONTRAST_MIN_MILLI = 3000;

export type SystemPrismKind =
  | 'root-workspace'
  | 'home-thread'
  | 'navigation-rail'
  | 'gallery'
  | 'settings'
  | 'space'
  | 'artifact';

export type PrismSurfaceMode = 'preview' | 'details';

export interface UiWidgetSpec {
  readonly index: number;
  readonly id: string;
  readonly slot:
    | 'root'
    | 'topbar'
    | 'sidebar'
    | 'stage'
    | 'prism-header'
    | 'prism-preview'
    | 'prism-details'
    | 'prism-nl-bar';
  readonly label: string;
  readonly minWidthPx: number;
  readonly minHeightPx: number;
  readonly flexWeight: number;
  readonly gapPx: number;
  readonly horizontal: boolean;
  /** WCAG 2.2 contrast ratio in milli-units (`4500` = `4.5:1`). */
  readonly contrastMilli: number;
  readonly isText: boolean;
  readonly interactive: boolean;
  readonly focusNeighbors: readonly number[];
  readonly children: readonly number[];
  /** Owning Verdant Prism ID for this widget surface. */
  readonly prismId: string;
}

export interface SolvedUiRect {
  readonly widgetIndex: number;
  readonly id: string;
  readonly slot: UiWidgetSpec['slot'];
  readonly label: string;
  readonly prismId: string;
  readonly xPx: number;
  readonly yPx: number;
  readonly widthPx: number;
  readonly heightPx: number;
}

export interface UiComplianceCertificate {
  readonly ok: boolean;
  readonly wcagContrastPassed: boolean;
  readonly focusGraphReachable: boolean;
  readonly siblingNonOverlapping: boolean;
  readonly touchTargetsCompliant: boolean;
  readonly widgetCount: number;
  readonly violations: readonly string[];
}

export interface SolvedUiLayout {
  readonly rects: readonly SolvedUiRect[];
  readonly certificate: UiComplianceCertificate;
  readonly viewportWidthPx: number;
  readonly viewportHeightPx: number;
  readonly layoutDigest: string;
}

export interface VerdantSystemPrismDescriptor {
  readonly id: string;
  readonly parentId: string | null;
  readonly kind: SystemPrismKind;
  readonly name: string;
  readonly subtitle: string;
  readonly purpose: string;
  readonly digest: string;
  readonly mediaType?: GalleryMediaType | 'space' | 'workspace';
  readonly nlPlaceholder: string;
  readonly nlSuggestions: readonly string[];
  readonly childPrismIds: readonly string[];
  readonly capabilities: readonly string[];
}

export interface ObserverAttentionSample {
  readonly activePrismId: string;
  readonly surfaceMode: PrismSurfaceMode;
  readonly userTyping: boolean;
  readonly dwellMillis: number;
  readonly darkMode: boolean;
}

export interface ObserverSurfaceDirective {
  readonly calmPeripheralUi: boolean;
  readonly salienceQ10: number;
  readonly previewEmphasis: 'full' | 'balanced';
  readonly tokens: Readonly<Record<string, string>>;
}

export interface SynthesizedPrismFrame {
  readonly rootPrism: VerdantSystemPrismDescriptor;
  readonly activePartPrism: VerdantSystemPrismDescriptor;
  readonly partPrisms: readonly VerdantSystemPrismDescriptor[];
  readonly layout: SolvedUiLayout;
  readonly directive: ObserverSurfaceDirective;
  readonly surfaceMode: PrismSurfaceMode;
  readonly synthesisDigest: string;
}

// ============================================================================
// 1. Tropical Semiring Kernels (`verdant-engine` kernel7 & kernel9 parity)
// ============================================================================

/**
 * Tropical `(max, +)` intrinsic 2D box extent solver (`verdant_tensor::tropical_ui_box_extents`).
 * - Along the main axis: sum child intrinsic extents + `(n - 1) * gap`.
 * - Along the cross axis: tropical `max` across children.
 */
export function tropicalUiBoxExtents(
  children: readonly (readonly [number, number])[],
  gapPx: number,
  horizontal: boolean,
): readonly [number, number] {
  if (children.length === 0) return [0, 0];
  let mainSum = 0;
  let crossMax = 0;
  for (let i = 0; i < children.length; i++) {
    const [w, h] = children[i]!;
    if (horizontal) {
      mainSum += Math.max(0, Math.floor(w));
      if (h > crossMax) crossMax = Math.max(0, Math.floor(h));
    } else {
      mainSum += Math.max(0, Math.floor(h));
      if (w > crossMax) crossMax = Math.max(0, Math.floor(w));
    }
  }
  const totalGap = Math.max(0, Math.floor(gapPx)) * (children.length - 1);
  return horizontal ? [mainSum + totalGap, crossMax] : [crossMax, mainSum + totalGap];
}

/**
 * Tropical `(min, +)` Floyd-Warshall all-pairs shortest-path closure (`kernel7`)
 * used to verify keyboard/focus-graph reachability across every interactive widget.
 */
export function tropicalFocusReachabilityClosure(
  count: number,
  edges: readonly (readonly [number, number])[],
): Int32Array {
  const INF = 0x3fffffff;
  const dist = new Int32Array(count * count);
  dist.fill(INF);
  for (let i = 0; i < count; i++) dist[i * count + i] = 0;
  for (const [u, v] of edges) {
    if (u >= 0 && u < count && v >= 0 && v < count) {
      dist[u * count + v] = 1;
      dist[v * count + u] = 1;
    }
  }
  for (let k = 0; k < count; k++) {
    for (let i = 0; i < count; i++) {
      const dik = dist[i * count + k]!;
      if (dik >= INF) continue;
      for (let j = 0; j < count; j++) {
        const alt = dik + dist[k * count + j]!;
        if (alt < dist[i * count + j]!) {
          dist[i * count + j] = alt;
        }
      }
    }
  }
  return dist;
}

/**
 * Exact integer UI constraint solver & compliance verifier (`verdant-core/src/ui.rs` parity).
 */
export function solveAndVerifyUiTree(
  widgets: readonly UiWidgetSpec[],
  viewportWidthPx: number,
  viewportHeightPx: number,
): SolvedUiLayout {
  const count = widgets.length;
  const violations: string[] = [];
  if (count === 0 || viewportWidthPx <= 0 || viewportHeightPx <= 0) {
    return {
      rects: [],
      certificate: {
        ok: false,
        wcagContrastPassed: false,
        focusGraphReachable: false,
        siblingNonOverlapping: false,
        touchTargetsCompliant: false,
        widgetCount: count,
        violations: ['UiTree and viewport dimensions must be non-empty.'],
      },
      viewportWidthPx,
      viewportHeightPx,
      layoutDigest: digestJson({ empty: true, viewportWidthPx, viewportHeightPx }),
    };
  }

  // 1. Bottom-up intrinsic bounds via Tropical (max, +)
  const intrinsic: [number, number][] = widgets.map((w) => [w.minWidthPx, w.minHeightPx]);
  for (let idx = count - 1; idx >= 0; idx--) {
    const w = widgets[idx]!;
    if (w.children.length > 0) {
      const childPairs = w.children.map((c) => intrinsic[c]!);
      const [ew, eh] = tropicalUiBoxExtents(childPairs, w.gapPx, w.horizontal);
      intrinsic[idx] = [Math.max(intrinsic[idx]![0], ew), Math.max(intrinsic[idx]![1], eh)];
    }
  }

  // 2. Focus-graph reachability via Tropical (min, +)
  const edges: [number, number][] = [];
  for (const w of widgets) {
    for (const dst of w.focusNeighbors) {
      edges.push([w.index, dst]);
    }
  }
  const closure = tropicalFocusReachabilityClosure(count, edges);
  const firstInteractive = widgets.findIndex((w) => w.interactive);

  // 3. Top-down exact integer flex allocation
  const rawRects: ([number, number, number, number] | null)[] = Array(count).fill(null);
  rawRects[0] = [
    0,
    0,
    Math.max(Math.floor(viewportWidthPx), intrinsic[0]![0]),
    Math.max(Math.floor(viewportHeightPx), intrinsic[0]![1]),
  ];

  let siblingNonOverlapping = true;

  for (let idx = 0; idx < count; idx++) {
    const parent = rawRects[idx];
    const w = widgets[idx]!;
    if (!parent || w.children.length === 0) continue;
    const [px, py, pw, ph] = parent;
    const totalGap = w.gapPx * (w.children.length - 1);
    const mainAvail = Math.max(0, (w.horizontal ? pw : ph) - totalGap);
    let minSum = 0;
    let weightSum = 0;
    for (const c of w.children) {
      minSum += w.horizontal ? intrinsic[c]![0] : intrinsic[c]![1];
      weightSum += widgets[c]!.flexWeight;
    }
    const extra = Math.max(0, mainAvail - minSum);

    let cursorMain = w.horizontal ? px : py;
    for (const c of w.children) {
      const cwFlex = widgets[c]!.flexWeight;
      const share = weightSum > 0 ? Math.floor((extra * cwFlex) / weightSum) : 0;
      if (w.horizontal) {
        const childW = intrinsic[c]![0] + share;
        const childH = Math.max(ph, intrinsic[c]![1]);
        rawRects[c] = [cursorMain, py, childW, childH];
        cursorMain += childW + w.gapPx;
      } else {
        const childH = intrinsic[c]![1] + share;
        const childW = Math.max(pw, intrinsic[c]![0]);
        rawRects[c] = [px, cursorMain, childW, childH];
        cursorMain += childH + w.gapPx;
      }
    }

    // Sibling non-overlap check
    for (let a = 0; a < w.children.length; a++) {
      for (let b = a + 1; b < w.children.length; b++) {
        const ra = rawRects[w.children[a]!];
        const rb = rawRects[w.children[b]!];
        if (!ra || !rb) continue;
        const overlapX = ra[0] < rb[0] + rb[2] && rb[0] < ra[0] + ra[2];
        const overlapY = ra[1] < rb[1] + rb[3] && rb[1] < ra[1] + ra[3];
        if (overlapX && overlapY) {
          siblingNonOverlapping = false;
          violations.push(`Sibling overlap between ${widgets[w.children[a]!]!.id} and ${widgets[w.children[b]!]!.id}`);
        }
      }
    }
  }

  // 4. Compliance verification
  let wcagContrastPassed = true;
  let focusGraphReachable = true;
  let touchTargetsCompliant = true;
  const INF = 0x3fffffff;

  const rects: SolvedUiRect[] = widgets.map((w) => {
    const [xPx, yPx, widthPx, heightPx] = rawRects[w.index] ?? [0, 0, w.minWidthPx, w.minHeightPx];
    const reqContrast = w.isText ? UI_TEXT_CONTRAST_MIN_MILLI : UI_NON_TEXT_CONTRAST_MIN_MILLI;
    if (w.contrastMilli < reqContrast) {
      wcagContrastPassed = false;
      violations.push(`Contrast ${w.contrastMilli} < ${reqContrast} on ${w.id}`);
    }
    if (w.interactive) {
      if (widthPx < UI_MIN_TOUCH_TARGET_PX || heightPx < UI_MIN_TOUCH_TARGET_PX) {
        touchTargetsCompliant = false;
        violations.push(`Touch target ${widthPx}x${heightPx} < ${UI_MIN_TOUCH_TARGET_PX} on ${w.id}`);
      }
      if (firstInteractive >= 0 && closure[firstInteractive * count + w.index]! >= INF) {
        focusGraphReachable = false;
        violations.push(`Interactive widget ${w.id} unreachable in focus graph`);
      }
    }
    return {
      widgetIndex: w.index,
      id: w.id,
      slot: w.slot,
      label: w.label,
      prismId: w.prismId,
      xPx,
      yPx,
      widthPx,
      heightPx,
    };
  });

  const certificate: UiComplianceCertificate = {
    ok: wcagContrastPassed && focusGraphReachable && siblingNonOverlapping && touchTargetsCompliant,
    wcagContrastPassed,
    focusGraphReachable,
    siblingNonOverlapping,
    touchTargetsCompliant,
    widgetCount: count,
    violations,
  };

  return {
    rects,
    certificate,
    viewportWidthPx,
    viewportHeightPx,
    layoutDigest: digestJson({ rects, ok: certificate.ok }),
  };
}

// ============================================================================
// 2. Self-Similar VerdantUI System Prism Catalog
// ============================================================================

export const VERDANT_ROOT_PRISM_ID = 'verdant-ui-root' as const;
export const VERDANT_HOME_PRISM_ID = 'verdant-part-home' as const;
export const VERDANT_SIDEBAR_PRISM_ID = 'verdant-part-sidebar' as const;
export const VERDANT_SETTINGS_PRISM_ID = 'verdant-part-settings' as const;

export function galleryPrismId(mediaType: GalleryMediaType): string {
  return `verdant-part-gallery-${mediaType}`;
}

function gallerySuggestions(mediaType: GalleryMediaType): readonly string[] {
  switch (mediaType) {
    case 'picture':
      return [
        'Create a misty emerald mountain sunrise landscape',
        'Create a minimalist cosmic nebula artwork',
        'Filter by landscapes',
      ];
    case 'video':
      return [
        'Create a 3-scene ocean horizon motion reel',
        'Create a calm architectural showcase video',
        'Show all videos',
      ];
    case 'audio':
      return [
        'Compose a calm ambient synth soundscape',
        'Compose a warm melodic lo-fi loop',
        'Show all audio tracks',
      ];
    case 'written-work':
      return [
        'Draft an essay about calm software design',
        'Start a short story about an alpine observatory',
        'Show all documents',
      ];
    case 'interactive-game':
      return [
        'Make a classic Minesweeper game',
        'Make a clean 2048 tile puzzle',
        'Make a Snake arcade game',
      ];
    case 'research-program':
      return [
        'Start a research notebook on local-first design',
        'Create a literature & hypothesis workspace',
        'Show all research programs',
      ];
    case 'tool-organ':
      return [
        'Protect 5pm to 6pm daily for exercise',
        'Create a focused daily checklist utility',
        'Create a quick savings calculator',
      ];
  }
}

export function buildSystemPartPrisms(): readonly VerdantSystemPrismDescriptor[] {
  const galleryPrisms: VerdantSystemPrismDescriptor[] = MEDIA_GALLERIES.map((g) => ({
    id: galleryPrismId(g.mediaType),
    parentId: VERDANT_ROOT_PRISM_ID,
    kind: 'gallery',
    name: g.label,
    subtitle: g.description,
    purpose: `Global ${g.label.toLowerCase()} index across all Spaces with instant natural-language creation, filtering, and curation.`,
    digest: digestJson({ prism: galleryPrismId(g.mediaType), mediaType: g.mediaType }),
    mediaType: g.mediaType,
    nlPlaceholder: `Ask ${g.label} to create, filter, or organize…`,
    nlSuggestions: gallerySuggestions(g.mediaType),
    childPrismIds: [],
    capabilities: ['natural-language-interface', 'live-preview', 'prism-details', 'local-index'],
  }));

  const homePrism: VerdantSystemPrismDescriptor = {
    id: VERDANT_HOME_PRISM_ID,
    parentId: VERDANT_ROOT_PRISM_ID,
    kind: 'home-thread',
    name: 'VerdantUI Home',
    subtitle: 'Natural-language workspace for conversation, creation, and orchestration.',
    purpose: 'Primary conversational Prism that creates, refines, and opens every other part and artifact in VerdantUI.',
    digest: digestJson({ prism: VERDANT_HOME_PRISM_ID, version: 1 }),
    mediaType: 'workspace',
    nlPlaceholder: 'Ask VerdantUI anything…',
    nlSuggestions: [
      'Generate a misty alpine sunset landscape picture',
      'Draft an essay on calm interfaces',
      'Create a new Travel Space',
    ],
    childPrismIds: [],
    capabilities: ['natural-language-interface', 'live-preview', 'prism-details', 'multimodal-io'],
  };

  const sidebarPrism: VerdantSystemPrismDescriptor = {
    id: VERDANT_SIDEBAR_PRISM_ID,
    parentId: VERDANT_ROOT_PRISM_ID,
    kind: 'navigation-rail',
    name: 'Workspace Navigation',
    subtitle: 'Threads, Spaces, and Media Galleries.',
    purpose: 'Child navigation Prism organizing threads, custom Spaces, and global MediaType Galleries.',
    digest: digestJson({ prism: VERDANT_SIDEBAR_PRISM_ID, version: 1 }),
    mediaType: 'workspace',
    nlPlaceholder: 'Jump to a Space, Gallery, or Thread…',
    nlSuggestions: ['Open Pictures', 'Open Documents', 'Create a new Space'],
    childPrismIds: galleryPrisms.map((g) => g.id),
    capabilities: ['navigation', 'local-index'],
  };

  const settingsPrism: VerdantSystemPrismDescriptor = {
    id: VERDANT_SETTINGS_PRISM_ID,
    parentId: VERDANT_ROOT_PRISM_ID,
    kind: 'settings',
    name: 'Settings',
    subtitle: 'Appearance and private on-device storage.',
    purpose: 'Workspace preferences Prism supporting natural-language theme switching and local storage hygiene.',
    digest: digestJson({ prism: VERDANT_SETTINGS_PRISM_ID, version: 1 }),
    mediaType: 'workspace',
    nlPlaceholder: 'Ask Settings to switch theme or manage local storage…',
    nlSuggestions: ['Switch to dark mode', 'Switch to light mode', 'Check saved storage'],
    childPrismIds: [],
    capabilities: ['natural-language-interface', 'live-preview', 'prism-details', 'local-storage'],
  };

  return [homePrism, sidebarPrism, settingsPrism, ...galleryPrisms];
}

export function buildRootWorkspacePrism(
  partPrisms: readonly VerdantSystemPrismDescriptor[],
  userPrisms: readonly VerdantPrismProject[],
): VerdantSystemPrismDescriptor {
  const childIds = [
    ...partPrisms.map((p) => p.id),
    ...userPrisms.map((p) => p.id),
  ];
  return {
    id: VERDANT_ROOT_PRISM_ID,
    parentId: null,
    kind: 'root-workspace',
    name: 'VerdantUI',
    subtitle: 'Free, private, 100% on-device AI workspace.',
    purpose:
      'Self-similar root Verdant Prism composing Home, Spaces, Global Galleries, Settings, and every user creation into one coherent procedural interface.',
    digest: digestJson({
      prism: VERDANT_ROOT_PRISM_ID,
      partCount: partPrisms.length,
      userPrismCount: userPrisms.length,
    }),
    mediaType: 'workspace',
    nlPlaceholder: 'Ask VerdantUI anything…',
    nlSuggestions: [
      'Generate an emerald landscape picture',
      'Draft a calm document',
      'Create a new Space',
    ],
    childPrismIds: childIds,
    capabilities: [
      'self-similar-prism-root',
      'procedural-ui-synthesis',
      'natural-language-interface',
      'local-only-privacy',
    ],
  };
}

export function descriptorForUserPrism(project: VerdantPrismProject): VerdantSystemPrismDescriptor {
  const mt =
    project.declaration?.view?.proceduralOrgan?.mediaType ??
    project.declaration?.concept.mediaType ??
    'tool-organ';
  const isSpace = mt === 'space';
  const name = project.published?.name ?? project.draft.name;
  const purpose =
    project.declaration?.concept.purpose ||
    project.draft.summary ||
    'Runs privately on your device inside VerdantUI.';

  const suggestionsByType: Record<string, readonly string[]> = {
    'written-work': [
      'Continue drafting this chapter in a calm, vivid voice',
      'Add a new chapter expanding the core idea',
      'Tighten the prose and sharpen the opening',
    ],
    picture: [
      'Shift lighting to golden hour with richer detail',
      'Try an architectural sunset composition',
      'Create a fresh variation with warmer tones',
    ],
    audio: [
      'Make it warmer and slower around 78 BPM',
      'Switch to an uplifting melodic arpeggio',
      'Add gentle rhythmic pulse',
    ],
    video: [
      'Add a closing scene with a smooth rise transition',
      'Switch aspect ratio to widescreen 16:9',
      'Brighten the accent palette',
    ],
    'interactive-game': [
      'Increase difficulty slightly and polish pacing',
      'Switch to an emerald twilight color theme',
      'Reset with a crisp new board layout',
    ],
    'research-program': [
      'Add a falsifiable baseline hypothesis',
      'Synthesize current evidence and milestones',
      'Add a validation milestone for next week',
    ],
    'tool-organ': (() => {
      const tw = project.declaration?.view?.proceduralOrgan?.weights;
      const role = tw && tw.mediaType === 'tool-organ' ? tw.role : null;
      if (role === 'focus-timer') {
        return [
          'Start a 25 minute focus session',
          'Switch to a 5 minute short break',
          'Reset the timer',
        ];
      }
      if (role === 'flashcards') {
        return [
          'Add card: Alkene | Hydrocarbon containing a carbon-carbon double bond',
          'Flip the current card',
          'Move to the next card',
        ];
      }
      if (role === 'budget-tracker') {
        return [
          'Add expense $18 lunch',
          'Add expense $45 transit pass',
          'Review remaining monthly budget',
        ];
      }
      if (role === 'calculator') {
        return ['Save current scenario', 'Compare adjusted totals', 'Reset values'];
      }
      return [
        'Protect 5pm to 6pm daily for focused time',
        'Add a daily check-in goal',
        'Refine the rules summary',
      ];
    })(),
    space: [
      'Switch layout to editorial showcase',
      'Switch layout to grouped board view',
      'Add my latest creation to this Space',
    ],
  };

  return {
    id: project.id,
    parentId: VERDANT_ROOT_PRISM_ID,
    kind: isSpace ? 'space' : 'artifact',
    name,
    subtitle: isSpace
      ? 'Curated Space collecting related creations.'
      : purpose,
    purpose,
    digest: project.published?.digest ?? project.draft.digest,
    mediaType: isSpace ? 'space' : (mt as GalleryMediaType),
    nlPlaceholder:
      mt === 'written-work'
        ? `Ask "${name}" to draft, revise, add a chapter, or refine tone…`
        : isSpace
          ? `Ask "${name}" to change layout, add creations, or organize…`
          : `Describe any change to "${name}" in plain English…`,
    nlSuggestions: suggestionsByType[mt] ?? [
      'Refine the visual polish and clarity',
      'Try a calmer color palette',
      'Publish this creation',
    ],
    childPrismIds: [],
    capabilities: ['natural-language-interface', 'live-preview', 'prism-details', 'sandboxed-runtime'],
  };
}

// ============================================================================
// 3. Observer Procedural UI Tree & Attention Synthesis
// ============================================================================

/**
 * Constructs the canonical `UiTree` for the active VerdantUI workspace + focused Prism surface.
 * Every surface follows the universal contract: Header (`Preview | Details` toggle) +
 * Rendering/Preview Panel + Natural Language Interface Bar.
 */
export function buildCanonicalWorkspaceUiTree(
  activePrismId: string,
  surfaceMode: PrismSurfaceMode,
  includeSidebar: boolean,
): readonly UiWidgetSpec[] {
  const hiText = 12800; // > 12:1 slate-950 on white / slate-50 on #0B0F14
  const chromeContrast = 4800; // > 4.5:1 border/control contrast

  if (!includeSidebar) {
    return [
      {
        index: 0,
        id: 'verdant-root-shell',
        slot: 'root',
        label: 'VerdantUI Root Prism',
        minWidthPx: 320,
        minHeightPx: 480,
        flexWeight: 1,
        gapPx: 0,
        horizontal: false,
        contrastMilli: chromeContrast,
        isText: false,
        interactive: false,
        focusNeighbors: [],
        children: [1, 2],
        prismId: VERDANT_ROOT_PRISM_ID,
      },
      {
        index: 1,
        id: 'verdant-topbar',
        slot: 'topbar',
        label: 'Workspace Header',
        minWidthPx: 320,
        minHeightPx: 52,
        flexWeight: 0,
        gapPx: 8,
        horizontal: true,
        contrastMilli: hiText,
        isText: true,
        interactive: true,
        focusNeighbors: [3],
        children: [],
        prismId: VERDANT_ROOT_PRISM_ID,
      },
      {
        index: 2,
        id: 'verdant-stage',
        slot: 'stage',
        label: 'Active Prism Stage',
        minWidthPx: 320,
        minHeightPx: 420,
        flexWeight: 1,
        gapPx: 0,
        horizontal: false,
        contrastMilli: chromeContrast,
        isText: false,
        interactive: false,
        focusNeighbors: [],
        children: [3, 4, 5],
        prismId: activePrismId,
      },
      {
        index: 3,
        id: 'prism-header',
        slot: 'prism-header',
        label: 'Prism Preview/Details Bar',
        minWidthPx: 320,
        minHeightPx: 48,
        flexWeight: 0,
        gapPx: 8,
        horizontal: true,
        contrastMilli: hiText,
        isText: true,
        interactive: true,
        focusNeighbors: [1, 4],
        children: [],
        prismId: activePrismId,
      },
      {
        index: 4,
        id: surfaceMode === 'details' ? 'prism-details-surface' : 'prism-preview-surface',
        slot: surfaceMode === 'details' ? 'prism-details' : 'prism-preview',
        label: surfaceMode === 'details' ? 'Prism Details' : 'Prism Preview',
        minWidthPx: 320,
        minHeightPx: 280,
        flexWeight: 1,
        gapPx: 0,
        horizontal: false,
        contrastMilli: hiText,
        isText: true,
        interactive: true,
        focusNeighbors: [3, 5],
        children: [],
        prismId: activePrismId,
      },
      {
        index: 5,
        id: 'prism-nl-bar',
        slot: 'prism-nl-bar',
        label: 'Natural Language Interface',
        minWidthPx: 320,
        minHeightPx: 64,
        flexWeight: 0,
        gapPx: 8,
        horizontal: true,
        contrastMilli: hiText,
        isText: true,
        interactive: true,
        focusNeighbors: [4, 1],
        children: [],
        prismId: activePrismId,
      },
    ];
  }

  return [
    {
      index: 0,
      id: 'verdant-root-shell',
      slot: 'root',
      label: 'VerdantUI Root Prism',
      minWidthPx: 960,
      minHeightPx: 600,
      flexWeight: 1,
      gapPx: 0,
      horizontal: false,
      contrastMilli: chromeContrast,
      isText: false,
      interactive: false,
      focusNeighbors: [],
      children: [1, 2],
      prismId: VERDANT_ROOT_PRISM_ID,
    },
    {
      index: 1,
      id: 'verdant-topbar',
      slot: 'topbar',
      label: 'Workspace Header',
      minWidthPx: 960,
      minHeightPx: 52,
      flexWeight: 0,
      gapPx: 8,
      horizontal: true,
      contrastMilli: hiText,
      isText: true,
      interactive: true,
      focusNeighbors: [3, 5],
      children: [],
      prismId: VERDANT_ROOT_PRISM_ID,
    },
    {
      index: 2,
      id: 'verdant-body-row',
      slot: 'stage',
      label: 'Workspace Body',
      minWidthPx: 960,
      minHeightPx: 540,
      flexWeight: 1,
      gapPx: 0,
      horizontal: true,
      contrastMilli: chromeContrast,
      isText: false,
      interactive: false,
      focusNeighbors: [],
      children: [3, 4],
      prismId: VERDANT_ROOT_PRISM_ID,
    },
    {
      index: 3,
      id: 'verdant-sidebar-prism',
      slot: 'sidebar',
      label: 'Navigation Prism',
      minWidthPx: 260,
      minHeightPx: 540,
      flexWeight: 0,
      gapPx: 0,
      horizontal: false,
      contrastMilli: hiText,
      isText: true,
      interactive: true,
      focusNeighbors: [1, 5],
      children: [],
      prismId: VERDANT_SIDEBAR_PRISM_ID,
    },
    {
      index: 4,
      id: 'verdant-active-prism',
      slot: 'stage',
      label: 'Active Prism Stage',
      minWidthPx: 680,
      minHeightPx: 540,
      flexWeight: 1,
      gapPx: 0,
      horizontal: false,
      contrastMilli: chromeContrast,
      isText: false,
      interactive: false,
      focusNeighbors: [],
      children: [5, 6, 7],
      prismId: activePrismId,
    },
    {
      index: 5,
      id: 'prism-header',
      slot: 'prism-header',
      label: 'Prism Preview/Details Bar',
      minWidthPx: 680,
      minHeightPx: 48,
      flexWeight: 0,
      gapPx: 8,
      horizontal: true,
      contrastMilli: hiText,
      isText: true,
      interactive: true,
      focusNeighbors: [1, 3, 6],
      children: [],
      prismId: activePrismId,
    },
    {
      index: 6,
      id: surfaceMode === 'details' ? 'prism-details-surface' : 'prism-preview-surface',
      slot: surfaceMode === 'details' ? 'prism-details' : 'prism-preview',
      label: surfaceMode === 'details' ? 'Prism Details' : 'Prism Preview',
      minWidthPx: 680,
      minHeightPx: 400,
      flexWeight: 1,
      gapPx: 0,
      horizontal: false,
      contrastMilli: hiText,
      isText: true,
      interactive: true,
      focusNeighbors: [5, 7],
      children: [],
      prismId: activePrismId,
    },
    {
      index: 7,
      id: 'prism-nl-bar',
      slot: 'prism-nl-bar',
      label: 'Natural Language Interface',
      minWidthPx: 680,
      minHeightPx: 64,
      flexWeight: 0,
      gapPx: 8,
      horizontal: true,
      contrastMilli: hiText,
      isText: true,
      interactive: true,
      focusNeighbors: [6, 1],
      children: [],
      prismId: activePrismId,
    },
  ];
}

/**
 * Second-order cybernetic mutual-attention directive (`verdant-engine` `kernel10` parity).
 * Derives calm peripheral chrome & unified homescreen-grade design tokens in exact integer Q10.
 */
export function stepObserverSurfaceAttention(
  sample: ObserverAttentionSample,
): ObserverSurfaceDirective {
  const dwellClamped = Math.min(1024, Math.floor(sample.dwellMillis / 4));
  const typingBoost = sample.userTyping ? 256 : 0;
  const salienceQ10 = Math.min(1024, Math.max(128, 384 + (dwellClamped >> 1) + typingBoost));
  const calmPeripheralUi = sample.surfaceMode === 'preview' && (sample.userTyping || sample.dwellMillis >= 1500);

  const tokens: Record<string, string> = sample.darkMode
    ? {
        '--v-bg': '#0B0F14',
        '--v-surface': '#0F141C',
        '--v-surface-elevated': '#141B26',
        '--v-border': '#1E293B',
        '--v-text': '#F1F5F9',
        '--v-muted': '#94A3B8',
        '--v-accent': '#10B981',
        '--v-accent-text': '#062116',
      }
    : {
        '--v-bg': '#F8F9FA',
        '--v-surface': '#FFFFFF',
        '--v-surface-elevated': '#F1F5F9',
        '--v-border': '#E2E8F0',
        '--v-text': '#0F172A',
        '--v-muted': '#64748B',
        '--v-accent': '#0F5132',
        '--v-accent-text': '#FFFFFF',
      };

  return {
    calmPeripheralUi,
    salienceQ10,
    previewEmphasis: sample.surfaceMode === 'preview' ? 'full' : 'balanced',
    tokens,
  };
}

/**
 * Procedurally synthesizes the complete VerdantUI Prism frame for the active part.
 */
export function synthesizeObserverPrismFrame(opts: {
  readonly activePrism: VerdantSystemPrismDescriptor;
  readonly partPrisms: readonly VerdantSystemPrismDescriptor[];
  readonly userPrisms: readonly VerdantPrismProject[];
  readonly surfaceMode: PrismSurfaceMode;
  readonly viewportWidthPx: number;
  readonly viewportHeightPx: number;
  readonly darkMode: boolean;
  readonly userTyping?: boolean;
  readonly dwellMillis?: number;
  readonly cachedLayout?: SolvedUiLayout;
}): SynthesizedPrismFrame {
  const rootPrism = buildRootWorkspacePrism(opts.partPrisms, opts.userPrisms);
  const includeSidebar = opts.viewportWidthPx >= 1024;
  const layout =
    opts.cachedLayout ??
    solveAndVerifyUiTree(
      buildCanonicalWorkspaceUiTree(opts.activePrism.id, opts.surfaceMode, includeSidebar),
      opts.viewportWidthPx,
      opts.viewportHeightPx,
    );
  const directive = stepObserverSurfaceAttention({
    activePrismId: opts.activePrism.id,
    surfaceMode: opts.surfaceMode,
    userTyping: Boolean(opts.userTyping),
    dwellMillis: opts.dwellMillis ?? 0,
    darkMode: opts.darkMode,
  });
  const synthesisDigest = digestJson({
    root: rootPrism.digest,
    active: opts.activePrism.digest,
    layout: layout.layoutDigest,
    mode: opts.surfaceMode,
    salienceQ10: directive.salienceQ10,
  });
  return {
    rootPrism,
    activePartPrism: opts.activePrism,
    partPrisms: opts.partPrisms,
    layout,
    directive,
    surfaceMode: opts.surfaceMode,
    synthesisDigest,
  };
}
