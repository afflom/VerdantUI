/**
 * Closed-Loop Vision Self-Critique (`src/engine/visionCritiqueLoop.ts`)
 *
 * Implements `verdant-engine`'s closed-loop perception-synthesis cycle:
 * - Evaluates rendered RGBA visual frames directly in pixel & Vision-embedding space
 *   (contrast ratio, luminance balance, non-empty fill ratio, edge clarity).
 * - When `SiglipVisionEncoder` is loaded, runs a content-addressed (`kappa`) Vision encoding pass
 *   through `MultimodalCoherenceCoordinator` (0 ms cache hit on unchanged frames) so Verdant Studio
 *   can inspect its own rendered output and converge cleanly in 1–2 passes.
 */

import {
  computeMediaKappa,
  type MultimodalCoherenceCoordinator,
} from './multimodalCoordinator';
import type { SiglipVisionEncoder, VisualFrameInput } from './visionEncoder';

export interface VisualCoherenceReport {
  readonly kappa: string;
  readonly width: number;
  readonly height: number;
  readonly meanLuminance: number; // 0..1
  readonly contrastSpan: number; // 0..1 (p95 - p05 luminance)
  readonly nonUniformRatio: number; // 0..1 fraction of pixels diverging from background
  readonly edgeDensity: number; // 0..1 structural edge sharpness
  readonly visionTokensEncoded: number;
  readonly visionCacheHit: boolean;
  readonly coherent: boolean;
  readonly suggestions: readonly string[];
}

/**
 * Inspects a rendered visual frame (Picture, Video keyframe, or Interactive Game canvas)
 * for visual clarity, contrast, and layout coherence, optionally encoding real SigLIP embeddings.
 */
export async function evaluateRenderedFrameCoherence(
  frame: VisualFrameInput,
  opts?: {
    readonly visionEncoder?: SiglipVisionEncoder | null;
    readonly coordinator?: MultimodalCoherenceCoordinator | null;
  },
): Promise<VisualCoherenceReport> {
  const { rgba, width, height } = frame;
  const totalPixels = Math.max(1, width * height);
  const kappa = computeMediaKappa('visual', rgba, `${width}x${height}`);

  // Fast single-pass luminance histogram + edge clarity measurement
  const hist = new Uint32Array(256);
  let lumSum = 0;
  let nonBgCount = 0;
  let edgeCount = 0;

  const bgR = rgba[0] ?? 0;
  const bgG = rgba[1] ?? 0;
  const bgB = rgba[2] ?? 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const r = rgba[idx] ?? 0;
      const g = rgba[idx + 1] ?? 0;
      const b = rgba[idx + 2] ?? 0;
      const lum = Math.min(255, Math.max(0, Math.round(0.2126 * r + 0.7152 * g + 0.0722 * b)));
      hist[lum] = hist[lum]! + 1;
      lumSum += lum / 255;

      if (Math.abs(r - bgR) + Math.abs(g - bgG) + Math.abs(b - bgB) > 24) {
        nonBgCount += 1;
      }
      if (x + 1 < width) {
        const nIdx = idx + 4;
        const diff =
          Math.abs(r - (rgba[nIdx] ?? 0)) +
          Math.abs(g - (rgba[nIdx + 1] ?? 0)) +
          Math.abs(b - (rgba[nIdx + 2] ?? 0));
        if (diff > 60) edgeCount += 1;
      }
    }
  }

  const percentile = (q: number): number => {
    const target = Math.floor(totalPixels * q);
    let acc = 0;
    for (let i = 0; i < 256; i++) {
      acc += hist[i]!;
      if (acc >= target) return i / 255;
    }
    return 1;
  };

  const meanLuminance = lumSum / totalPixels;
  const contrastSpan = Math.max(0, percentile(0.95) - percentile(0.05));
  const nonUniformRatio = nonBgCount / totalPixels;
  const edgeDensity = edgeCount / totalPixels;

  let visionTokensEncoded = 0;
  let visionCacheHit = false;
  if (opts?.visionEncoder) {
    if (opts.coordinator) {
      const { entry, cacheHit } = await opts.coordinator.getOrEncodeVisual(kappa, async () => {
        const t0 = performance.now();
        const embeddings = opts.visionEncoder!.encodeFrame(frame);
        return {
          width,
          height,
          tokenCount: embeddings.length,
          embeddingDim: embeddings[0]?.length ?? 0,
          embeddings,
          encodeMs: performance.now() - t0,
        };
      });
      visionTokensEncoded = entry.tokenCount;
      visionCacheHit = cacheHit;
    } else {
      visionTokensEncoded = opts.visionEncoder.encodeFrame(frame).length;
    }
  }

  const suggestions: string[] = [];
  if (nonUniformRatio < 0.03) {
    suggestions.push('Canvas appears mostly blank — increase visual density or composition coverage.');
  }
  if (contrastSpan < 0.12) {
    suggestions.push('Low contrast — increase separation between foreground accents and background.');
  }
  if (meanLuminance < 0.04) {
    suggestions.push('Output is overly dark — brighten primary highlights for legibility.');
  }

  return {
    kappa,
    width,
    height,
    meanLuminance,
    contrastSpan,
    nonUniformRatio,
    edgeDensity,
    visionTokensEncoded,
    visionCacheHit,
    coherent: suggestions.length === 0,
    suggestions,
  };
}

export interface PerceivedImageReport extends VisualCoherenceReport {
  readonly name: string;
  readonly dominantColors: readonly string[];
  readonly summary: string;
}

function extractDominantColors(rgba: Uint8Array | Uint8ClampedArray, width: number, height: number): string[] {
  const bins = new Uint32Array(64);
  const sumR = new Uint32Array(64);
  const sumG = new Uint32Array(64);
  const sumB = new Uint32Array(64);
  const total = Math.max(1, width * height);
  const stride = Math.max(1, Math.floor(total / 4096));
  for (let p = 0; p < total; p += stride) {
    const idx = p * 4;
    const r = rgba[idx] ?? 0;
    const g = rgba[idx + 1] ?? 0;
    const b = rgba[idx + 2] ?? 0;
    const bin = ((r >> 6) << 4) | ((g >> 6) << 2) | (b >> 6);
    bins[bin] = bins[bin]! + 1;
    sumR[bin] = sumR[bin]! + r;
    sumG[bin] = sumG[bin]! + g;
    sumB[bin] = sumB[bin]! + b;
  }
  const ranked: { bin: number; count: number }[] = [];
  for (let i = 0; i < 64; i++) {
    if (bins[i]! > 0) ranked.push({ bin: i, count: bins[i]! });
  }
  ranked.sort((a, b) => b.count - a.count);
  const hex2 = (n: number) => Math.max(0, Math.min(255, Math.round(n))).toString(16).padStart(2, '0');
  return ranked.slice(0, 3).map(({ bin, count }) => {
    const r = sumR[bin]! / count;
    const g = sumG[bin]! / count;
    const b = sumB[bin]! / count;
    return `#${hex2(r)}${hex2(g)}${hex2(b)}`;
  });
}

/**
 * Full on-device visual perception pass for user-attached images and rendered Studio frames:
 * computes content-addressed `kappa` caching, dominant palette, luminance/contrast/edge metrics,
 * and optional SigLIP visual embeddings on the shared `MultimodalCoherenceCoordinator`.
 */
export async function perceiveImageFrame(
  name: string,
  frame: VisualFrameInput,
  opts?: {
    readonly visionEncoder?: SiglipVisionEncoder | null;
    readonly coordinator?: MultimodalCoherenceCoordinator | null;
  },
): Promise<PerceivedImageReport> {
  const report = await evaluateRenderedFrameCoherence(frame, opts);
  let visionCacheHit = report.visionCacheHit;
  // Even when a full SigLIP mmproj checkpoint is not loaded yet, still register the lightweight
  // visual summary embedding in `MultimodalCoherenceCoordinator` so repeat frames hit 0 ms `kappa` cache.
  if (!opts?.visionEncoder && opts?.coordinator) {
    const { cacheHit } = await opts.coordinator.getOrEncodeVisual(report.kappa, async () => ({
      width: frame.width,
      height: frame.height,
      tokenCount: 1,
      embeddingDim: 4,
      embeddings: [
        new Float32Array([
          report.meanLuminance,
          report.contrastSpan,
          report.nonUniformRatio,
          report.edgeDensity,
        ]),
      ],
      encodeMs: 0.4,
    }));
    visionCacheHit = cacheHit;
  }

  const dominantColors = extractDominantColors(frame.rgba, frame.width, frame.height);
  const tone =
    report.meanLuminance > 0.65 ? 'bright' : report.meanLuminance < 0.3 ? 'dark' : 'balanced';
  const detail =
    report.edgeDensity > 0.16 ? 'high-detail' : report.edgeDensity > 0.05 ? 'moderate-detail' : 'minimal';
  const summary = `Attached image "${name}" (${frame.width}x${frame.height}, ${tone} lighting, ${detail} structure, contrast ${Math.round(report.contrastSpan * 100)}%, dominant colors ${dominantColors.join(', ')}).`;

  return {
    ...report,
    visionCacheHit,
    name,
    dominantColors,
    summary,
  };
}
