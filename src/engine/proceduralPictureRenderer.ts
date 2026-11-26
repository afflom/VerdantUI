/**
 * High-Spec 10-Scene Layered Procedural Picture Synthesizer
 * (`src/engine/proceduralPictureRenderer.ts`)
 *
 * Shared deterministic pixel/RGBA & PNG synthesis engine used by:
 * 1. Chat Threads (inline high-spec PNG images rendered directly inside assistant messages)
 * 2. Verdant Studio Picture Prisms (interactive HiDPI canvas + variation/aspect/lighting controls)
 * 3. Global Pictures Gallery (cached thumbnails & deterministic regeneration from weights)
 * 4. Closed-Loop Vision Self-Critique (`evaluateRenderedFrameCoherence`)
 *
 * Supports 10 rich procedural scene generators:
 *   landscape-vista | cosmic-nebula | architectural-scene | botanical-garden | ocean-seascape |
 *   abstract-flow | geometric-mosaic | constellation-atlas | poster-editorial | pixel-scene
 * (plus seamless backward compatibility for legacy generator aliases:
 *   botanical-fractal -> botanical-garden, wave-field -> ocean-seascape,
 *   constellation -> constellation-atlas, pixel-sprite -> pixel-scene).
 */

import type { PictureGeneratorKind, PictureWeights } from './prismMediaOrgans';
import type { UserStyleAffinities } from './unifiedModelBasis';
import { evaluateRenderedFrameCoherence, type VisualCoherenceReport } from './visionCritiqueLoop';

export type PictureLightingMood =
  | 'golden-hour'
  | 'twilight'
  | 'midnight'
  | 'daylight'
  | 'neon'
  | 'studio';

export interface RenderedPictureResult {
  readonly width: number;
  readonly height: number;
  readonly rgba: Uint8ClampedArray;
  readonly pngDataUrl: string;
  readonly coherence: VisualCoherenceReport;
  readonly weights: PictureWeights;
}

type Rgb = readonly [number, number, number];

function parseHexRgb(hex: string, fallback: Rgb = [16, 185, 129]): Rgb {
  const clean = (hex || '').trim().replace(/^#/, '');
  if (clean.length === 3) {
    return [
      parseInt(clean[0]! + clean[0]!, 16) || fallback[0],
      parseInt(clean[1]! + clean[1]!, 16) || fallback[1],
      parseInt(clean[2]! + clean[2]!, 16) || fallback[2],
    ];
  }
  if (clean.length >= 6) {
    return [
      parseInt(clean.slice(0, 2), 16) || fallback[0],
      parseInt(clean.slice(2, 4), 16) || fallback[1],
      parseInt(clean.slice(4, 6), 16) || fallback[2],
    ];
  }
  return fallback;
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function clamp255(x: number): number {
  return x < 0 ? 0 : x > 255 ? 255 : Math.round(x);
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function smoothstep(t: number): number {
  const c = clamp01(t);
  return c * c * (3 - 2 * c);
}

function mixRgb(a: Rgb, b: Rgb, t: number): Rgb {
  const u = clamp01(t);
  return [lerp(a[0], b[0], u), lerp(a[1], b[1], u), lerp(a[2], b[2], u)];
}

function addGlow(base: Rgb, glow: Rgb, amount: number): Rgb {
  const a = Math.max(0, amount);
  return [
    Math.min(255, base[0] + glow[0] * a),
    Math.min(255, base[1] + glow[1] * a),
    Math.min(255, base[2] + glow[2] * a),
  ];
}

function hash2d(ix: number, iy: number, seed: number): number {
  let h = Math.imul(ix | 0, 374761393) + Math.imul(iy | 0, 668265263) + Math.imul(seed | 0, 1442695041);
  h = (h ^ (h >>> 13)) | 0;
  h = Math.imul(h, 1274126177);
  h = (h ^ (h >>> 16)) >>> 0;
  return h / 4294967296;
}

function valueNoise2d(x: number, y: number, seed: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = smoothstep(x - ix);
  const fy = smoothstep(y - iy);
  const a = hash2d(ix, iy, seed);
  const b = hash2d(ix + 1, iy, seed);
  const c = hash2d(ix, iy + 1, seed);
  const d = hash2d(ix + 1, iy + 1, seed);
  return lerp(lerp(a, b, fx), lerp(c, d, fx), fy);
}

function fbm2d(x: number, y: number, seed: number, octaves = 4): number {
  let sum = 0;
  let amp = 0.5;
  let freq = 1.0;
  let norm = 0;
  for (let o = 0; o < octaves; o++) {
    sum += amp * valueNoise2d(x * freq, y * freq, seed + o * 97);
    norm += amp;
    amp *= 0.5;
    freq *= 2.04;
  }
  return norm > 0 ? sum / norm : 0;
}

function fbm1d(x: number, seed: number, octaves = 4): number {
  return fbm2d(x, 0.37, seed, octaves);
}

function mulberry32(a: number): () => number {
  let s = a | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function blendPixel(
  rgba: Uint8ClampedArray,
  w: number,
  h: number,
  x: number,
  y: number,
  col: Rgb,
  alpha: number,
): void {
  const xi = Math.round(x);
  const yi = Math.round(y);
  if (xi < 0 || xi >= w || yi < 0 || yi >= h || alpha <= 0) return;
  const idx = (yi * w + xi) * 4;
  const a = clamp01(alpha);
  const inv = 1 - a;
  rgba[idx] = clamp255(rgba[idx]! * inv + col[0] * a);
  rgba[idx + 1] = clamp255(rgba[idx + 1]! * inv + col[1] * a);
  rgba[idx + 2] = clamp255(rgba[idx + 2]! * inv + col[2] * a);
  rgba[idx + 3] = 255;
}

function drawDisc(
  rgba: Uint8ClampedArray,
  w: number,
  h: number,
  cx: number,
  cy: number,
  radius: number,
  col: Rgb,
  alpha = 1,
  softEdge = 0.25,
): void {
  const r = Math.max(0.5, radius);
  const x0 = Math.max(0, Math.floor(cx - r - 1));
  const x1 = Math.min(w - 1, Math.ceil(cx + r + 1));
  const y0 = Math.max(0, Math.floor(cy - r - 1));
  const y1 = Math.min(h - 1, Math.ceil(cy + r + 1));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const d = Math.hypot(x - cx, y - cy) / r;
      if (d <= 1) {
        const edge = softEdge > 0 ? smoothstep(clamp01((1 - d) / softEdge)) : 1;
        blendPixel(rgba, w, h, x, y, col, alpha * edge);
      }
    }
  }
}

function drawLine(
  rgba: Uint8ClampedArray,
  w: number,
  h: number,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  col: Rgb,
  alpha = 1,
  thickness = 1,
): void {
  const dist = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0)));
  const r = Math.max(0.6, thickness * 0.5);
  for (let i = 0; i <= dist; i++) {
    const t = i / dist;
    const px = lerp(x0, x1, t);
    const py = lerp(y0, y1, t);
    if (r <= 1) {
      blendPixel(rgba, w, h, px, py, col, alpha);
    } else {
      drawDisc(rgba, w, h, px, py, r, col, alpha, 0.35);
    }
  }
}

/** Canonicalizes legacy generator names into the upgraded 10-scene set. */
export function resolveCanonicalGenerator(gen: PictureGeneratorKind): PictureGeneratorKind {
  if (gen === 'botanical-fractal') return 'botanical-garden';
  if (gen === 'wave-field') return 'ocean-seascape';
  if (gen === 'constellation') return 'constellation-atlas';
  if (gen === 'pixel-sprite') return 'pixel-scene';
  return gen;
}

/**
 * Infers rich, semantic `PictureWeights` (scene generator, palette, lighting, aspect ratio,
 * density, symmetry, and deterministic seed) directly from natural language and learned user basis.
 */
export function inferPictureWeightsFromPrompt(
  prompt: string,
  titleHint = 'Artwork',
  basisAffinities?: UserStyleAffinities,
): PictureWeights {
  const lower = `${titleHint} ${prompt}`.toLowerCase();

  let generator: PictureGeneratorKind = 'landscape-vista';
  if (/\b(nebula|galaxy|planet|saturn|cosmos|deep\s+space|supernova|black\s+hole)\b/.test(lower)) {
    generator = 'cosmic-nebula';
  } else if (/\b(constellation|star\s*map|celestial|astrology|night\s+sky\s+chart|zodiac)\b/.test(lower)) {
    generator = 'constellation-atlas';
  } else if (/\b(city|skyline|architecture|building|greenhouse|cathedral|temple|bridge|arch|monolith|cyberpunk\s+street)\b/.test(lower)) {
    generator = 'architectural-scene';
  } else if (/\b(flower|garden|botanical|tree|fern|blossom|orchard|leaf|foliage|bonsai|ivy)\b/.test(lower)) {
    generator = 'botanical-garden';
  } else if (/\b(ocean|sea|seascape|waves?|coast|beach|lighthouse|surf|aurora\s+water)\b/.test(lower)) {
    generator = 'ocean-seascape';
  } else if (/\b(mountain|lake|valley|forest|alpine|meadow|canyon|desert|sunset|sunrise|landscape|vista|horizon|hills?)\b/.test(lower)) {
    generator = 'landscape-vista';
  } else if (/\b(poster|editorial|bauhaus|swiss|cover|flyer|banner|minimalist\s+graphic)\b/.test(lower)) {
    generator = 'poster-editorial';
  } else if (/\b(pixel|8-?bit|16-?bit|retro\s+art|sprite)\b/.test(lower)) {
    generator = 'pixel-scene';
  } else if (/\b(mosaic|mandala|stained\s+glass|geometric|sunburst|kaleidoscope|tessellat)\b/.test(lower)) {
    generator = 'geometric-mosaic';
  } else if (/\b(abstract|flow|ribbon|fluid|silk|swirl|marble|aura)\b/.test(lower)) {
    generator = 'abstract-flow';
  } else if (basisAffinities?.preferredPictureStyle) {
    generator = basisAffinities.preferredPictureStyle as PictureGeneratorKind;
  }

  let lighting: PictureLightingMood = 'golden-hour';
  if (/\b(night|midnight|starry|nocturne|moon|dark|noir|space|cosmos)\b/.test(lower)) {
    lighting = 'midnight';
  } else if (/\b(twilight|dusk|evening|violet|aurora|mystic)\b/.test(lower)) {
    lighting = 'twilight';
  } else if (/\b(neon|cyber|synthwave|vaporwave|electric|glow)\b/.test(lower)) {
    lighting = 'neon';
  } else if (/\b(day|morning|noon|bright|sunlit|spring|pastel|fresh)\b/.test(lower)) {
    lighting = 'daylight';
  } else if (/\b(sunset|golden|amber|autumn|warm|dawn|sunrise)\b/.test(lower)) {
    lighting = 'golden-hour';
  } else if (basisAffinities?.preferredPictureLighting) {
    lighting = basisAffinities.preferredPictureLighting as PictureLightingMood;
  }

  // Cohesive, high-contrast 4-color palettes tuned to prompt semantics
  let palette: [string, string, string, string];
  if (/\b(sunset|golden|amber|autumn|fire|desert|mars|crimson|warm)\b/.test(lower)) {
    palette = ['#1b1126', '#d95d39', '#f59e0b', '#fef3c7'];
  } else if (/\b(emerald|forest|pine|jade|verdant|moss|botanical|green)\b/.test(lower)) {
    palette = ['#071916', '#0f766e', '#34d399', '#ecfdf5'];
  } else if (/\b(ocean|cobalt|sapphire|ice|arctic|glacier|blue|marine)\b/.test(lower)) {
    palette = ['#071324', '#0284c7', '#38bdf8', '#e0f2fe'];
  } else if (/\b(neon|cyber|violet|magenta|synthwave|nebula|cosmic|purple)\b/.test(lower)) {
    palette = ['#0a0618', '#7c3aed', '#ec4899', '#fde68a'];
  } else if (/\b(monochrome|ink|noir|charcoal|minimal|black\s+and\s+white)\b/.test(lower)) {
    palette = ['#0b0d10', '#334155', '#94a3b8', '#f8fafc'];
  } else if (/\b(pastel|sakura|cherry|blossom|rose|dawn|peach)\b/.test(lower)) {
    palette = ['#1f1629', '#e11d48', '#fb923c', '#fff1f2'];
  } else if (basisAffinities?.preferredPalette) {
    palette = [...basisAffinities.preferredPalette] as [string, string, string, string];
  } else if (lighting === 'midnight') {
    palette = ['#070b18', '#1d4ed8', '#38bdf8', '#f8fafc'];
  } else if (lighting === 'twilight') {
    palette = ['#110e24', '#6366f1', '#f472b6', '#fef9c3'];
  } else {
    palette = ['#0b1918', '#0f766e', '#f59e0b', '#fefce8'];
  }

  // Aspect ratio inference
  let width = 800;
  let height = 500;
  if (/\b(square|1:1|avatar|icon|album\s+cover)\b/.test(lower)) {
    width = 640;
    height = 640;
  } else if (/\b(portrait|vertical|9:16|phone|poster|tall)\b/.test(lower)) {
    width = 512;
    height = 768;
  } else if (/\b(4:3|classic)\b/.test(lower)) {
    width = 768;
    height = 576;
  } else if (/\b(16:9|widescreen|cinematic|wallpaper|banner)\b/.test(lower)) {
    width = 896;
    height = 504;
  }

  // Deterministic hash seed from prompt text so every unique description produces a unique artwork
  let seed = 4219;
  for (let i = 0; i < prompt.length; i++) {
    seed = (Math.imul(seed, 31) + prompt.charCodeAt(i)) | 0;
  }
  seed = (Math.abs(seed) % 990_000) + 1000;

  const symmetry = /\b(mandala|kaleidoscope|sunburst|radial)\b/.test(lower) ? 10 : generator === 'geometric-mosaic' ? 8 : 6;
  const density = /\b(detailed|intricate|rich|dense|high\s+detail|lush)\b/.test(lower)
    ? 82
    : /\b(minimal|simple|clean|sparse)\b/.test(lower)
      ? 32
      : 58;

  return {
    mediaType: 'picture',
    generator,
    lighting,
    width,
    height,
    seed,
    symmetry,
    density,
    palette,
    caption: titleHint.slice(0, 120),
  };
}

/**
 * Deterministically renders any of the 10 procedural picture scene generators into an RGBA8 buffer.
 * Runs identically in browser, Worker, and headless Node tests (zero DOM dependency).
 */
export function renderPictureToRgba(
  weights: PictureWeights,
  targetWidth?: number,
  targetHeight?: number,
): { readonly width: number; readonly height: number; readonly rgba: Uint8ClampedArray } {
  const w = Math.max(64, Math.min(1600, Math.round(targetWidth ?? weights.width)));
  const h = Math.max(64, Math.min(1200, Math.round(targetHeight ?? weights.height)));
  const rgba = new Uint8ClampedArray(w * h * 4);
  const seed = weights.seed || 4219;
  const rng = mulberry32(seed);
  const gen = resolveCanonicalGenerator(weights.generator);

  const cBg = parseHexRgb(weights.palette[0], [11, 23, 20]);
  const cMid = parseHexRgb(weights.palette[1], [16, 185, 129]);
  const cAcc = parseHexRgb(weights.palette[2], [245, 158, 11]);
  const cHi = parseHexRgb(weights.palette[3], [236, 253, 245]);

  if (gen === 'landscape-vista') {
    const horizonY = Math.round(h * 0.62);
    const sunX = w * (0.28 + rng() * 0.44);
    const sunY = h * (0.22 + rng() * 0.22);
    const sunR = Math.min(w, h) * 0.085;
    const isNight = weights.lighting === 'midnight';

    // Sky + volumetric clouds + sun/moon glow
    for (let y = 0; y < horizonY; y++) {
      const ty = y / Math.max(1, horizonY);
      const skyBase = mixRgb(cBg, mixRgb(cMid, cAcc, smoothstep(ty * 1.15)), Math.pow(ty, 0.85));
      for (let x = 0; x < w; x++) {
        const u = x / w;
        const v = y / h;
        const cloudN = fbm2d(u * 4.5 + 1.2, v * 7.5 + 0.5, seed + 17, 4);
        const cloudMask = smoothstep((cloudN - 0.48) * 3.1) * (1 - ty * 0.45);
        let col = mixRgb(skyBase, mixRgb(cAcc, cHi, 0.55), cloudMask * (isNight ? 0.22 : 0.48));
        const dSun = Math.hypot(x - sunX, y - sunY);
        const halo = Math.exp(-dSun / (sunR * 3.2)) * (isNight ? 0.55 : 0.85);
        col = addGlow(col, mixRgb(cAcc, cHi, 0.65), halo);
        if (dSun < sunR) {
          const core = smoothstep((sunR - dSun) / (sunR * 0.22));
          col = mixRgb(col, cHi, core);
        }
        const idx = (y * w + x) * 4;
        rgba[idx] = clamp255(col[0]);
        rgba[idx + 1] = clamp255(col[1]);
        rgba[idx + 2] = clamp255(col[2]);
        rgba[idx + 3] = 255;
      }
    }

    // Subtle stars for night/twilight sky
    if (isNight || weights.lighting === 'twilight') {
      const starCount = Math.max(40, Math.round(weights.density * 2.5));
      for (let i = 0; i < starCount; i++) {
        const sx = rng() * w;
        const sy = rng() * horizonY * 0.72;
        const sr = 0.6 + rng() * 1.4;
        drawDisc(rgba, w, h, sx, sy, sr, cHi, 0.45 + rng() * 0.5, 0.4);
      }
    }

    // 4 layered fBm mountain ridges with atmospheric haze
    const layers = 4;
    for (let L = 0; L < layers; L++) {
      const depth = L / (layers - 1); // 0 = far, 1 = near
      const baseH = horizonY - (1 - depth) * h * 0.22 + depth * h * 0.03;
      const amp = h * (0.21 - depth * 0.04);
      const ridgeCol = mixRgb(mixRgb(cMid, cAcc, 0.22), cBg, 0.22 + depth * 0.68);
      const mistCol = mixRgb(cAcc, cHi, 0.4);
      for (let x = 0; x < w; x++) {
        const nx = (x / w) * (2.6 + L * 1.4) + L * 7.3;
        const ridgeY = Math.round(baseH - (fbm1d(nx, seed + L * 211, 5) - 0.35) * amp);
        for (let y = Math.max(0, ridgeY); y < horizonY; y++) {
          const distFromCrest = (y - ridgeY) / Math.max(1, horizonY - ridgeY);
          const rimHighlight = y - ridgeY < 3 ? 0.22 * (1 - depth * 0.5) : 0;
          const fog = (1 - depth) * 0.35 * distFromCrest;
          let col = mixRgb(ridgeCol, mistCol, fog);
          if (rimHighlight > 0) col = addGlow(col, cHi, rimHighlight);
          const idx = (y * w + x) * 4;
          rgba[idx] = clamp255(col[0]);
          rgba[idx + 1] = clamp255(col[1]);
          rgba[idx + 2] = clamp255(col[2]);
          rgba[idx + 3] = 255;
        }
      }
    }

    // Foreground pine treeline silhouette on left/right shores
    const treeCount = Math.max(24, Math.round(weights.density * 0.9));
    for (let i = 0; i < treeCount; i++) {
      const side = i % 2 === 0 ? rng() * 0.34 : 0.66 + rng() * 0.34;
      const tx = side * w;
      const th = h * (0.04 + rng() * 0.085);
      const baseY = horizonY + 1;
      for (let dy = 0; dy < th; dy++) {
        const t = dy / th;
        const halfW = t * th * 0.28 * (0.7 + 0.3 * Math.sin(t * 16));
        const y = Math.round(baseY - th + dy);
        if (y < 0 || y >= horizonY) continue;
        for (let x = Math.max(0, Math.round(tx - halfW)); x <= Math.min(w - 1, Math.round(tx + halfW)); x++) {
          const idx = (y * w + x) * 4;
          rgba[idx] = clamp255(cBg[0] * 0.45);
          rgba[idx + 1] = clamp255(cBg[1] * 0.55);
          rgba[idx + 2] = clamp255(cBg[2] * 0.55);
          rgba[idx + 3] = 255;
        }
      }
    }

    // Mirrored lake reflection with water ripples & sun/moon glitter
    for (let y = horizonY; y < h; y++) {
      const depth = (y - horizonY) / Math.max(1, h - horizonY);
      const srcY = Math.max(0, Math.min(horizonY - 1, Math.round(horizonY - (y - horizonY) * 0.92)));
      for (let x = 0; x < w; x++) {
        const ripple = Math.sin(y * 0.55 + fbm2d(x / 36, y / 10, seed + 901, 2) * 8) * (2 + depth * 12);
        const sx = Math.max(0, Math.min(w - 1, Math.round(x + ripple)));
        const sIdx = (srcY * w + sx) * 4;
        let col: Rgb = [rgba[sIdx]!, rgba[sIdx + 1]!, rgba[sIdx + 2]!];
        col = mixRgb(col, cBg, 0.24 + depth * 0.32);
        const sunGlint = Math.exp(-Math.abs(x - sunX) / (18 + depth * 64)) * Math.max(0, Math.sin(y * 0.9 + ripple));
        col = addGlow(col, mixRgb(cAcc, cHi, 0.65), sunGlint * 0.55);
        const idx = (y * w + x) * 4;
        rgba[idx] = clamp255(col[0]);
        rgba[idx + 1] = clamp255(col[1]);
        rgba[idx + 2] = clamp255(col[2]);
        rgba[idx + 3] = 255;
      }
    }
  } else if (gen === 'cosmic-nebula') {
    const px = w * (0.56 + (rng() - 0.5) * 0.16);
    const py = h * (0.48 + (rng() - 0.5) * 0.12);
    const pr = Math.min(w, h) * 0.21;
    // Deep turbulent nebula field
    for (let y = 0; y < h; y++) {
      const v = y / h;
      for (let x = 0; x < w; x++) {
        const u = x / w;
        const warpX = fbm2d(u * 3.1, v * 3.1, seed + 13, 4);
        const warpY = fbm2d(u * 3.1 + 4.2, v * 3.1 + 1.9, seed + 41, 4);
        const n1 = fbm2d(u * 3.6 + warpX * 1.4, v * 3.6 + warpY * 1.4, seed + 77, 5);
        const n2 = fbm2d(u * 5.2 - warpY, v * 5.2 + warpX, seed + 131, 4);
        let col = mixRgb(cBg, cMid, smoothstep((n1 - 0.34) * 2.4) * 0.85);
        col = addGlow(col, cAcc, smoothstep((n2 - 0.48) * 3.0) * 0.62);
        col = addGlow(col, cHi, smoothstep((n1 * n2 - 0.34) * 5.0) * 0.42);
        const idx = (y * w + x) * 4;
        rgba[idx] = clamp255(col[0]);
        rgba[idx + 1] = clamp255(col[1]);
        rgba[idx + 2] = clamp255(col[2]);
        rgba[idx + 3] = 255;
      }
    }
    // Starfield with bright diffraction spikes
    const stars = Math.max(90, weights.density * 4);
    for (let i = 0; i < stars; i++) {
      const sx = rng() * w;
      const sy = rng() * h;
      const bright = rng() > 0.92;
      const sr = bright ? 1.8 + rng() * 1.8 : 0.6 + rng() * 1.1;
      const scol = i % 5 === 0 ? cAcc : cHi;
      drawDisc(rgba, w, h, sx, sy, sr, scol, 0.55 + rng() * 0.45, 0.45);
      if (bright) {
        drawLine(rgba, w, h, sx - sr * 6, sy, sx + sr * 6, sy, scol, 0.55, 1);
        drawLine(rgba, w, h, sx, sy - sr * 6, sx, sy + sr * 6, scol, 0.55, 1);
      }
    }
    // Back half of planetary rings + shaded ringed planet + front ring arc
    const tilt = -0.32;
    const cosT = Math.cos(tilt);
    const sinT = Math.sin(tilt);
    const drawRingHalf = (front: boolean) => {
      const rIn = pr * 1.32;
      const rOut = pr * 2.18;
      for (let y = Math.max(0, Math.floor(py - rOut)); y <= Math.min(h - 1, Math.ceil(py + rOut)); y++) {
        for (let x = Math.max(0, Math.floor(px - rOut)); x <= Math.min(w - 1, Math.ceil(px + rOut)); x++) {
          const dx = x - px;
          const dy = y - py;
          const rx = dx * cosT - dy * sinT;
          const ry = (dx * sinT + dy * cosT) * 3.1;
          if ((front && ry < 0) || (!front && ry >= 0)) continue;
          const dist = Math.hypot(rx, ry);
          if (dist >= rIn && dist <= rOut) {
            const band = 0.5 + 0.5 * Math.sin((dist - rIn) * 0.42);
            const alpha = (0.25 + band * 0.42) * smoothstep(Math.min((dist - rIn) / 8, (rOut - dist) / 10));
            blendPixel(rgba, w, h, x, y, mixRgb(cAcc, cHi, band * 0.6), alpha);
          }
        }
      }
    };
    drawRingHalf(false);
    for (let y = Math.max(0, Math.floor(py - pr - 2)); y <= Math.min(h - 1, Math.ceil(py + pr + 2)); y++) {
      for (let x = Math.max(0, Math.floor(px - pr - 2)); x <= Math.min(w - 1, Math.ceil(px + pr + 2)); x++) {
        const dx = (x - px) / pr;
        const dy = (y - py) / pr;
        const r2 = dx * dx + dy * dy;
        if (r2 > 1) continue;
        const nz = Math.sqrt(Math.max(0, 1 - r2));
        const light = clamp01(dx * -0.55 + dy * -0.35 + nz * 0.75);
        const band = fbm2d(dx * 2.2, dy * 6.5, seed + 512, 4);
        let col = mixRgb(cBg, mixRgb(cMid, cAcc, band), Math.pow(light, 0.72));
        const rim = Math.pow(1 - nz, 3.0) * clamp01(light + 0.2);
        col = addGlow(col, cHi, rim * 0.85);
        blendPixel(rgba, w, h, x, y, col, 1);
      }
    }
    drawRingHalf(true);
  } else if (gen === 'architectural-scene') {
    const groundY = Math.round(h * 0.7);
    // Atmospheric sky + dramatic volumetric light beams
    for (let y = 0; y < groundY; y++) {
      const ty = y / Math.max(1, groundY);
      const base = mixRgb(cBg, mixRgb(cMid, cAcc, 0.45), Math.pow(ty, 0.9));
      for (let x = 0; x < w; x++) {
        const beam = Math.pow(Math.max(0, Math.sin((x / w) * 5.2 - ty * 2.4 + 0.8)), 6);
        const col = addGlow(base, mixRgb(cAcc, cHi, 0.5), beam * 0.24);
        const idx = (y * w + x) * 4;
        rgba[idx] = clamp255(col[0]);
        rgba[idx + 1] = clamp255(col[1]);
        rgba[idx + 2] = clamp255(col[2]);
        rgba[idx + 3] = 255;
      }
    }
    // Layered architectural towers, glass domes & arches
    for (let layer = 0; layer < 3; layer++) {
      const count = 9 + layer * 4;
      const bCol = mixRgb(cMid, cBg, 0.35 + layer * 0.28);
      for (let i = 0; i < count; i++) {
        const bw = Math.round(w * (0.045 + rng() * 0.085));
        const bh = Math.round(h * (0.16 + rng() * (0.42 - layer * 0.07)));
        const bx = Math.round((i / count) * w + (rng() - 0.5) * 28);
        const by = groundY - bh;
        for (let y = Math.max(0, by); y < groundY; y++) {
          for (let x = Math.max(0, bx); x < Math.min(w, bx + bw); x++) {
            const sideShade = x - bx < bw * 0.22 ? 0.14 : 0;
            const col = addGlow(bCol, cHi, sideShade);
            const idx = (y * w + x) * 4;
            rgba[idx] = clamp255(col[0]);
            rgba[idx + 1] = clamp255(col[1]);
            rgba[idx + 2] = clamp255(col[2]);
            rgba[idx + 3] = 255;
          }
        }
        // Illuminated architectural glass windows
        if (layer >= 1) {
          for (let wy = by + 8; wy < groundY - 8; wy += 10) {
            for (let wx = bx + 5; wx < bx + bw - 6; wx += 8) {
              if (hash2d(wx, wy, seed + layer * 31) > 0.42) {
                const winCol = hash2d(wx, wy, seed + 77) > 0.3 ? cAcc : cHi;
                for (let dy = 0; dy < 5; dy++) {
                  for (let dx = 0; dx < 4; dx++) {
                    blendPixel(rgba, w, h, wx + dx, wy + dy, winCol, 0.72);
                  }
                }
              }
            }
          }
        }
      }
    }
    // Wet-stone plaza reflection below horizon
    for (let y = groundY; y < h; y++) {
      const depth = (y - groundY) / Math.max(1, h - groundY);
      const srcY = Math.max(0, Math.min(groundY - 1, Math.round(groundY - (y - groundY) * 1.15)));
      for (let x = 0; x < w; x++) {
        const sIdx = (srcY * w + x) * 4;
        const reflected: Rgb = [rgba[sIdx]!, rgba[sIdx + 1]!, rgba[sIdx + 2]!];
        const col = mixRgb(reflected, cBg, 0.35 + depth * 0.45);
        const idx = (y * w + x) * 4;
        rgba[idx] = clamp255(col[0]);
        rgba[idx + 1] = clamp255(col[1]);
        rgba[idx + 2] = clamp255(col[2]);
        rgba[idx + 3] = 255;
      }
    }
  } else if (gen === 'botanical-garden') {
    // Warm dappled bokeh garden backdrop
    for (let y = 0; y < h; y++) {
      const ty = y / h;
      const base = mixRgb(cBg, cMid, 0.25 + ty * 0.3);
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;
        rgba[idx] = clamp255(base[0]);
        rgba[idx + 1] = clamp255(base[1]);
        rgba[idx + 2] = clamp255(base[2]);
        rgba[idx + 3] = 255;
      }
    }
    for (let i = 0; i < 28; i++) {
      drawDisc(rgba, w, h, rng() * w, rng() * h * 0.85, 14 + rng() * 44, i % 2 === 0 ? cAcc : cHi, 0.14 + rng() * 0.16, 0.85);
    }
    const branch = (x: number, y: number, len: number, angle: number, depth: number) => {
      if (depth <= 0 || len < 5) return;
      const x2 = x + Math.cos(angle) * len;
      const y2 = y + Math.sin(angle) * len;
      const stemCol = mixRgb(cBg, cMid, 0.5 + (7 - depth) * 0.08);
      drawLine(rgba, w, h, x, y, x2, y2, stemCol, 0.92, Math.max(1.2, depth * 1.35));
      if (depth <= 3) {
        // Organic leaf & blossom cluster
        for (let k = 0; k < 3; k++) {
          const la = angle + (k - 1) * 0.65;
          const lx = x2 + Math.cos(la) * (6 + rng() * 9);
          const ly = y2 + Math.sin(la) * (6 + rng() * 9);
          drawDisc(rgba, w, h, lx, ly, 4 + rng() * 6, depth === 1 ? cAcc : cMid, 0.78, 0.4);
        }
        if (depth === 1 && rng() > 0.35) {
          drawDisc(rgba, w, h, x2, y2, 5 + rng() * 5, cHi, 0.9, 0.55);
          drawDisc(rgba, w, h, x2, y2, 2.5, cAcc, 0.95, 0.3);
        }
      }
      const spread = 0.34 + rng() * 0.24;
      branch(x2, y2, len * 0.74, angle - spread, depth - 1);
      branch(x2, y2, len * 0.74, angle + spread, depth - 1);
    };
    branch(w * 0.5, h * 0.96, h * 0.25, -Math.PI / 2 + (rng() - 0.5) * 0.12, 7);
    branch(w * 0.24, h * 0.98, h * 0.17, -Math.PI / 2 + 0.18, 5);
    branch(w * 0.76, h * 0.98, h * 0.18, -Math.PI / 2 - 0.18, 5);
  } else if (gen === 'ocean-seascape') {
    const horizonY = Math.round(h * 0.44);
    const sunX = w * (0.35 + rng() * 0.3);
    const sunY = horizonY - h * 0.12;
    const sunR = Math.min(w, h) * 0.08;
    for (let y = 0; y < horizonY; y++) {
      const ty = y / Math.max(1, horizonY);
      const base = mixRgb(cBg, mixRgb(cMid, cAcc, ty * 0.75), ty);
      for (let x = 0; x < w; x++) {
        const dSun = Math.hypot(x - sunX, y - sunY);
        let col = addGlow(base, mixRgb(cAcc, cHi, 0.6), Math.exp(-dSun / (sunR * 3.5)) * 0.85);
        if (dSun < sunR) col = mixRgb(col, cHi, smoothstep((sunR - dSun) / (sunR * 0.2)));
        const idx = (y * w + x) * 4;
        rgba[idx] = clamp255(col[0]);
        rgba[idx + 1] = clamp255(col[1]);
        rgba[idx + 2] = clamp255(col[2]);
        rgba[idx + 3] = 255;
      }
    }
    // Layered cresting ocean waves with foam & glitter path
    for (let y = horizonY; y < h; y++) {
      const depth = (y - horizonY) / Math.max(1, h - horizonY);
      const waveBase = mixRgb(cMid, cBg, 0.2 + depth * 0.65);
      for (let x = 0; x < w; x++) {
        const u = x / w;
        const w1 = Math.sin(u * 18 + depth * 26 + fbm2d(u * 5, depth * 6, seed + 19, 3) * 5);
        const crest = smoothstep((w1 - 0.62) * 3.2);
        const glint = Math.exp(-Math.abs(x - sunX) / (16 + depth * 85)) * Math.max(0, w1);
        let col = mixRgb(waveBase, cHi, crest * (0.32 + depth * 0.38));
        col = addGlow(col, mixRgb(cAcc, cHi, 0.55), glint * 0.7);
        const idx = (y * w + x) * 4;
        rgba[idx] = clamp255(col[0]);
        rgba[idx + 1] = clamp255(col[1]);
        rgba[idx + 2] = clamp255(col[2]);
        rgba[idx + 3] = 255;
      }
    }
  } else if (gen === 'abstract-flow') {
    for (let y = 0; y < h; y++) {
      const v = y / h;
      for (let x = 0; x < w; x++) {
        const u = x / w;
        const n = fbm2d(u * 2.8, v * 2.8, seed + 7, 4);
        const n2 = fbm2d(u * 3.6 + n * 2.2, v * 3.6 - n * 1.8, seed + 67, 4);
        const ribbon = Math.pow(0.5 + 0.5 * Math.sin((u * 6 + v * 4 + n2 * 7.5) * Math.PI), 2.4);
        let col = mixRgb(cBg, cMid, clamp01(n * 1.15));
        col = mixRgb(col, cAcc, ribbon * 0.68);
        col = addGlow(col, cHi, Math.pow(ribbon, 4) * 0.45);
        const idx = (y * w + x) * 4;
        rgba[idx] = clamp255(col[0]);
        rgba[idx + 1] = clamp255(col[1]);
        rgba[idx + 2] = clamp255(col[2]);
        rgba[idx + 3] = 255;
      }
    }
  } else if (gen === 'constellation-atlas') {
    for (let y = 0; y < h; y++) {
      const v = y / h;
      for (let x = 0; x < w; x++) {
        const u = x / w;
        const wash = fbm2d(u * 3.2, v * 3.2, seed + 44, 4);
        const col = mixRgb(cBg, cMid, smoothstep((wash - 0.42) * 2.2) * 0.42);
        const idx = (y * w + x) * 4;
        rgba[idx] = clamp255(col[0]);
        rgba[idx + 1] = clamp255(col[1]);
        rgba[idx + 2] = clamp255(col[2]);
        rgba[idx + 3] = 255;
      }
    }
    // Celestial coordinate graticule
    const cx = w / 2;
    const cy = h / 2;
    for (let r = 60; r < Math.max(w, h) * 0.62; r += 70) {
      for (let a = 0; a < 360; a += 1) {
        const rad = (a * Math.PI) / 180;
        blendPixel(rgba, w, h, cx + Math.cos(rad) * r, cy + Math.sin(rad) * r, cHi, 0.14);
      }
    }
    const pts: Array<[number, number, number]> = [];
    const nPts = Math.max(36, Math.min(90, weights.density));
    for (let i = 0; i < nPts; i++) {
      pts.push([w * (0.08 + rng() * 0.84), h * (0.08 + rng() * 0.84), 1.4 + rng() * 3.2]);
    }
    const maxD2 = Math.pow(Math.min(w, h) * 0.24, 2);
    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        const dx = pts[i]![0] - pts[j]![0];
        const dy = pts[i]![1] - pts[j]![1];
        if (dx * dx + dy * dy < maxD2) {
          drawLine(rgba, w, h, pts[i]![0], pts[i]![1], pts[j]![0], pts[j]![1], cMid, 0.36, 1);
        }
      }
    }
    for (let i = 0; i < pts.length; i++) {
      const [sx, sy, sr] = pts[i]!;
      drawDisc(rgba, w, h, sx, sy, sr * 2.4, i % 4 === 0 ? cAcc : cMid, 0.25, 0.85);
      drawDisc(rgba, w, h, sx, sy, sr, i % 4 === 0 ? cAcc : cHi, 0.95, 0.35);
    }
  } else if (gen === 'poster-editorial') {
    for (let y = 0; y < h; y++) {
      const col = mixRgb(cHi, cBg, 0.08 + (y / h) * 0.12);
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;
        rgba[idx] = clamp255(col[0]);
        rgba[idx + 1] = clamp255(col[1]);
        rgba[idx + 2] = clamp255(col[2]);
        rgba[idx + 3] = 255;
      }
    }
    // Bold Bauhaus/Swiss geometric composition
    const cx = w * 0.5;
    const cy = h * 0.46;
    const rad = Math.min(w, h) * 0.32;
    drawDisc(rgba, w, h, cx - rad * 0.22, cy, rad, cBg, 0.96, 0.02);
    drawDisc(rgba, w, h, cx + rad * 0.28, cy - rad * 0.14, rad * 0.72, cMid, 0.9, 0.02);
    drawDisc(rgba, w, h, cx + rad * 0.05, cy + rad * 0.22, rad * 0.42, cAcc, 0.94, 0.02);
    // Crisp editorial margin rules & grid bars
    const m = Math.round(Math.min(w, h) * 0.06);
    drawLine(rgba, w, h, m, m, w - m, m, cBg, 0.85, 2.5);
    drawLine(rgba, w, h, m, h - m * 2.2, w - m, h - m * 2.2, cBg, 0.85, 2);
    drawLine(rgba, w, h, m, h - m, w - m, h - m, cAcc, 0.9, 4);
  } else if (gen === 'pixel-scene') {
    const gridW = 48;
    const gridH = Math.max(24, Math.round((h / w) * gridW));
    const cellW = w / gridW;
    const cellH = h / gridH;
    const horizon = Math.round(gridH * 0.62);
    for (let gy = 0; gy < gridH; gy++) {
      for (let gx = 0; gx < gridW; gx++) {
        const bayer = ((gx & 1) * 2 + (gy & 1)) / 4 - 0.375;
        let col: Rgb;
        if (gy < horizon) {
          const t = gy / horizon + bayer * 0.14;
          col = t < 0.38 ? cBg : t < 0.72 ? cMid : cAcc;
          const sunD = Math.hypot(gx - gridW * 0.68, gy - horizon * 0.45);
          if (sunD < 4.5) col = cHi;
          else if (sunD < 7) col = cAcc;
        } else {
          const ridge = Math.round(horizon + Math.sin(gx * 0.32 + seed) * 3 + fbm1d(gx * 0.18, seed, 2) * 5);
          col = gy < ridge ? cMid : (gx + gy) % 3 === 0 ? cBg : mixRgb(cBg, cMid, 0.35);
        }
        const x0 = Math.floor(gx * cellW);
        const y0 = Math.floor(gy * cellH);
        const x1 = Math.min(w, Math.ceil((gx + 1) * cellW));
        const y1 = Math.min(h, Math.ceil((gy + 1) * cellH));
        for (let y = y0; y < y1; y++) {
          for (let x = x0; x < x1; x++) {
            const idx = (y * w + x) * 4;
            rgba[idx] = clamp255(col[0]);
            rgba[idx + 1] = clamp255(col[1]);
            rgba[idx + 2] = clamp255(col[2]);
            rgba[idx + 3] = 255;
          }
        }
      }
    }
  } else {
    // geometric-mosaic: high-precision radial mandala & stained-glass tessellation
    const cx = w / 2;
    const cy = h / 2;
    const maxR = Math.hypot(cx, cy);
    const sym = Math.max(4, Math.min(16, weights.symmetry || 8));
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const dx = x - cx;
        const dy = y - cy;
        const r = Math.hypot(dx, dy) / maxR;
        const theta = Math.atan2(dy, dx);
        const fold = Math.abs(((theta * sym) / Math.PI) % 2 - 1);
        const wave = 0.5 + 0.5 * Math.sin(r * 22 - fold * Math.PI * 2 + (seed % 17));
        const ring = 0.5 + 0.5 * Math.cos(r * 14 + Math.cos(theta * sym) * 1.4);
        const facet = smoothstep(wave * 0.6 + ring * 0.4);
        let col = mixRgb(cBg, cMid, clamp01((1 - r * 0.65) * facet));
        col = mixRgb(col, cAcc, smoothstep((ring - 0.58) * 3.0) * (1 - r * 0.4));
        const border = Math.abs(Math.sin(r * 28)) < 0.05 || Math.abs(Math.sin(theta * sym)) < 0.04;
        if (border) col = mixRgb(col, cHi, 0.35);
        const idx = (y * w + x) * 4;
        rgba[idx] = clamp255(col[0]);
        rgba[idx + 1] = clamp255(col[1]);
        rgba[idx + 2] = clamp255(col[2]);
        rgba[idx + 3] = 255;
      }
    }
  }

  // Photographic post-process: gentle vignette + micro-grain
  const cx = w * 0.5;
  const cy = h * 0.5;
  const invDiag = 1 / Math.hypot(cx, cy);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4;
      const vDist = Math.hypot(x - cx, y - cy) * invDiag;
      const vig = 1 - Math.pow(vDist, 2.2) * 0.24;
      const grain = (hash2d(x, y, seed + 4099) - 0.5) * 6;
      rgba[idx] = clamp255(rgba[idx]! * vig + grain);
      rgba[idx + 1] = clamp255(rgba[idx + 1]! * vig + grain);
      rgba[idx + 2] = clamp255(rgba[idx + 2]! * vig + grain);
      rgba[idx + 3] = 255;
    }
  }

  return { width: w, height: h, rgba };
}

// ============================================================================
// Pure Zero-Dependency PNG Encoder (Works identically in Browser, Worker & Node)
// ============================================================================

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    }
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    c = CRC_TABLE[(c ^ bytes[i]!) & 0xff]! ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

function adler32(bytes: Uint8Array): number {
  let a = 1;
  let b = 0;
  for (let i = 0; i < bytes.length; i++) {
    a = (a + bytes[i]!) % 65521;
    b = (b + a) % 65521;
  }
  return ((b << 16) | a) >>> 0;
}

function makePngChunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length, false);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  const crc = crc32(out.subarray(4, 8 + data.length));
  dv.setUint32(8 + data.length, crc, false);
  return out;
}

function uint8ToBase64(bytes: Uint8Array): string {
  if (typeof Buffer !== 'undefined') {
    return Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString('base64');
  }
  let bin = '';
  const step = 0x8000;
  for (let i = 0; i < bytes.length; i += step) {
    bin += String.fromCharCode(...bytes.subarray(i, Math.min(bytes.length, i + step)));
  }
  return btoa(bin);
}

/** Encodes an RGBA8 image into a standards-compliant `data:image/png;base64,...` data URL. */
export function encodeRgbaToPngDataUrl(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
): string {
  // Fast browser canvas path when available for compact compressed PNG
  if (typeof document !== 'undefined') {
    try {
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        const imgData = new ImageData(new Uint8ClampedArray(rgba), width, height);
        ctx.putImageData(imgData, 0, 0);
        return canvas.toDataURL('image/png');
      }
    } catch {
      // Fall back to pure JS PNG encoder
    }
  }

  // Pure JS standards-compliant PNG with uncompressed zlib blocks (works in Node & Workers)
  const stride = width * 4;
  const raw = new Uint8Array(height * (1 + stride));
  for (let y = 0; y < height; y++) {
    raw[y * (1 + stride)] = 0; // filter type 0 (None)
    raw.set(rgba.subarray(y * stride, (y + 1) * stride), y * (1 + stride) + 1);
  }

  const maxBlock = 65535;
  const numBlocks = Math.max(1, Math.ceil(raw.length / maxBlock));
  const zlib = new Uint8Array(2 + raw.length + numBlocks * 5 + 4);
  zlib[0] = 0x78;
  zlib[1] = 0x01;
  let pos = 2;
  for (let off = 0; off < raw.length; off += maxBlock) {
    const len = Math.min(maxBlock, raw.length - off);
    const isFinal = off + len >= raw.length ? 1 : 0;
    zlib[pos++] = isFinal;
    zlib[pos++] = len & 0xff;
    zlib[pos++] = (len >>> 8) & 0xff;
    const nlen = (~len) & 0xffff;
    zlib[pos++] = nlen & 0xff;
    zlib[pos++] = (nlen >>> 8) & 0xff;
    zlib.set(raw.subarray(off, off + len), pos);
    pos += len;
  }
  const adler = adler32(raw);
  const zDv = new DataView(zlib.buffer);
  zDv.setUint32(pos, adler, false);

  const sig = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
  const ihdrData = new Uint8Array(13);
  const ihdrDv = new DataView(ihdrData.buffer);
  ihdrDv.setUint32(0, width, false);
  ihdrDv.setUint32(4, height, false);
  ihdrData[8] = 8; // bit depth
  ihdrData[9] = 6; // RGBA
  ihdrData[10] = 0;
  ihdrData[11] = 0;
  ihdrData[12] = 0;

  const ihdr = makePngChunk('IHDR', ihdrData);
  const idat = makePngChunk('IDAT', zlib);
  const iend = makePngChunk('IEND', new Uint8Array(0));

  const full = new Uint8Array(sig.length + ihdr.length + idat.length + iend.length);
  let cursor = 0;
  full.set(sig, cursor); cursor += sig.length;
  full.set(ihdr, cursor); cursor += ihdr.length;
  full.set(idat, cursor); cursor += idat.length;
  full.set(iend, cursor);

  return `data:image/png;base64,${uint8ToBase64(full)}`;
}

/**
 * Renders a high-spec procedural picture and runs a closed-loop visual coherence check
 * (`evaluateRenderedFrameCoherence`), automatically retrying with a boosted-contrast seed
 * if the first render fails coherence.
 */
export async function renderPictureWithCoherenceCheck(
  weights: PictureWeights,
  opts?: { readonly maxPreviewDim?: number },
): Promise<RenderedPictureResult> {
  const maxDim = opts?.maxPreviewDim ?? 640;
  const scale = Math.min(1, maxDim / Math.max(weights.width, weights.height));
  const w = Math.max(160, Math.round(weights.width * scale));
  const h = Math.max(120, Math.round(weights.height * scale));

  let activeWeights = weights;
  let rendered = renderPictureToRgba(activeWeights, w, h);
  let coherence = await evaluateRenderedFrameCoherence({
    rgba: rendered.rgba,
    width: rendered.width,
    height: rendered.height,
  });

  if (!coherence.coherent) {
    activeWeights = {
      ...activeWeights,
      seed: ((activeWeights.seed * 1664525 + 1013904223) >>> 0) % 990_000 + 1000,
      density: Math.min(100, activeWeights.density + 18),
    };
    rendered = renderPictureToRgba(activeWeights, w, h);
    coherence = await evaluateRenderedFrameCoherence({
      rgba: rendered.rgba,
      width: rendered.width,
      height: rendered.height,
    });
  }

  return {
    width: rendered.width,
    height: rendered.height,
    rgba: rendered.rgba,
    pngDataUrl: encodeRgbaToPngDataUrl(rendered.rgba, rendered.width, rendered.height),
    coherence,
    weights: activeWeights,
  };
}
