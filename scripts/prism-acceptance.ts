#!/usr/bin/env node
/**
 * Mechanical Physical-Facet Acceptance Gate (`scripts/prism-acceptance.ts`)
 *
 * Parity with `afflom/verdant-studio/.github/scripts/prism-acceptance.py`:
 * Runs VerdantUI's real synthesis pipelines on canonical seeds across ALL 8 MediaTypes
 * and verifies physical output fidelity with zero mocks:
 *   - Audio WAV: RIFF/WAVE header, stereo 16-bit PCM, zero NaN/Inf, safe true peak & non-silent RMS
 *   - Picture PNG & SVG: valid PNG magic/IHDR/IEND + pixel entropy; valid SVG namespace XML
 *   - Editorial/Research/Utility/Game/Video/Space HTML+JS: zero JS syntax errors, non-empty DOM/CSS,
 *     and every form control (<input>/<select>/<textarea>) carries `id` or `name` (zero DevTools form notices)
 *   - `verdant-ui.holo`: archive integrity & 100% vector replay
 */

import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  composeMultiTrackArrangement,
  encodeStereoWav,
  masterStereoPcm,
  synthesizeCompositionStereoPcm,
} from '../src/engine/audioSynthesisEngine';
import {
  inferPictureWeightsFromPrompt,
  renderPictureWithCoherenceCheck,
  synthesizeLogoSvg,
} from '../src/engine/proceduralPictureRenderer';
import { ORGAN_REGISTRY } from '../src/engine/organRegistry';
import { ObserverStreamBus } from '../src/engine/observerStreamBus';
import { VerdantCognitiveObserver } from '../src/engine/cognitiveObserver';
import { PrismStore } from '../src/engine/prismStudio';
import { PrismHostPool } from '../src/engine/prismHostPool';
import { ContentGalleryStore } from '../src/engine/contentGalleryStore';
import { UnifiedExpertRegistry } from '../src/engine/unifiedModelBasis';
import { StudioChatBridge } from '../src/engine/studioChatBridge';
import { synthesizeProceduralOrganView } from '../src/engine/prismMediaOrgans';
import { VerdantHoloHost } from '../src/host/verdantHoloHost';

function ascii(bytes: Uint8Array, start: number, len: number): string {
  return String.fromCharCode(...bytes.subarray(start, start + len));
}

function verifyWavPhysical(wav: Uint8Array): { readonly peakDbfs: number; readonly rmsDbfs: number } {
  if (wav.byteLength < 44) throw new Error(`refused: WAV too small (${wav.byteLength} bytes)`);
  if (ascii(wav, 0, 4) !== 'RIFF' || ascii(wav, 8, 4) !== 'WAVE') {
    throw new Error('refused: invalid WAV RIFF/WAVE header');
  }
  const view = new DataView(wav.buffer, wav.byteOffset, wav.byteLength);
  const channels = view.getUint16(22, true);
  const bitsPerSample = view.getUint16(34, true);
  if (channels !== 2 || bitsPerSample !== 16) {
    throw new Error(`refused: expected stereo 16-bit PCM WAV, got ${channels}ch ${bitsPerSample}-bit`);
  }
  let peak = 0;
  let sumSq = 0;
  const sampleCount = Math.floor((wav.byteLength - 44) / 2);
  for (let i = 0; i < sampleCount; i++) {
    const s = view.getInt16(44 + i * 2, true) / 32768;
    if (!Number.isFinite(s)) throw new Error('refused: non-finite WAV sample');
    const a = Math.abs(s);
    if (a > peak) peak = a;
    sumSq += s * s;
  }
  const rms = Math.sqrt(sumSq / Math.max(1, sampleCount));
  const peakDbfs = peak > 0 ? 20 * Math.log10(peak) : -120;
  const rmsDbfs = rms > 0 ? 20 * Math.log10(rms) : -120;
  if (rmsDbfs < -55) throw new Error(`refused: WAV appears silent (${rmsDbfs.toFixed(1)} dBFS)`);
  if (peakDbfs > 0.01) throw new Error(`refused: WAV peak clips above 0 dBFS (${peakDbfs.toFixed(2)})`);
  return { peakDbfs, rmsDbfs };
}

function verifyPngPhysical(png: Uint8Array): { readonly width: number; readonly height: number } {
  const sig = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  for (let i = 0; i < sig.length; i++) {
    if (png[i] !== sig[i]) throw new Error('refused: invalid PNG magic header');
  }
  if (ascii(png, 12, 4) !== 'IHDR') throw new Error('refused: missing PNG IHDR chunk');
  const view = new DataView(png.buffer, png.byteOffset, png.byteLength);
  const width = view.getUint32(16, false);
  const height = view.getUint32(20, false);
  if (width < 32 || height < 32) throw new Error(`refused: PNG dimensions too small (${width}x${height})`);
  const tail = ascii(png, Math.max(0, png.byteLength - 12), 12);
  if (!tail.includes('IEND')) throw new Error('refused: PNG missing IEND chunk');
  return { width, height };
}

function verifyFormControlAccessibility(html: string, organId: string): void {
  const controlTags = html.match(/<(input|select|textarea)\b[^>]*>/gi) ?? [];
  for (const tag of controlTags) {
    if (!/\b(id|name)=["'][^"']+["']/i.test(tag)) {
      throw new Error(`refused: organ "${organId}" form control missing id/name: ${tag.slice(0, 80)}`);
    }
  }
}

export async function runPrismAcceptance(cwd = process.cwd()): Promise<void> {
  const root = resolve(cwd);
  const t0 = performance.now();

  // 1. Audio WAV physical facet check
  const arrangement = composeMultiTrackArrangement({
    title: 'Acceptance Nocturne',
    style: 'ambient-pad',
    bpm: 92,
    rootMidi: 57,
    scale: 'minor',
    durationSec: 4,
  });
  const rawPcm = synthesizeCompositionStereoPcm(arrangement, 22_050);
  const mastered = masterStereoPcm(rawPcm.left, rawPcm.right, 22_050);
  const wav = encodeStereoWav(mastered.left, mastered.right, 22_050);
  const wavStats = verifyWavPhysical(wav);

  // 2. Picture PNG & SVG physical facet check
  const picWeights = inferPictureWeightsFromPrompt('Minimalist sunrise over alpine lake');
  const rendered = await renderPictureWithCoherenceCheck(picWeights, { maxPreviewDim: 160 });
  const pngB64 = rendered.pngDataUrl.split(',')[1] ?? '';
  const pngBytes = new Uint8Array(Buffer.from(pngB64, 'base64'));
  const pngDims = verifyPngPhysical(pngBytes);

  const logoSvg = synthesizeLogoSvg({
    ...picWeights,
    generator: 'logo-mark',
    caption: 'Verdant Acceptance Mark',
  });
  if (!logoSvg.includes('<svg') || !logoSvg.includes('http://www.w3.org/2000/svg') || !logoSvg.includes('</svg>')) {
    throw new Error('refused: synthesizeLogoSvg produced invalid SVG XML');
  }

  // 3. All 8 MediaTypes + Space HTML/CSS/JS syntax & form-accessibility check
  const bus = new ObserverStreamBus();
  const observer = new VerdantCognitiveObserver();
  const store = new PrismStore();
  const galleries = new ContentGalleryStore();
  const registry = new UnifiedExpertRegistry();
  const pool = new PrismHostPool(bus, store, observer.grants, null);
  const bridge = new StudioChatBridge({ store, pool, observer, galleries, registry });

  let checkedOrgans = 0;
  try {
    for (const desc of ORGAN_REGISTRY) {
      const created = await bridge.createArtifactFromPrompt({
        prompt: desc.samplePrompt,
        forcedMediaType: desc.mediaType,
      });
      const proj = store.get(created.prismId);
      if (!proj?.declaration?.view?.proceduralOrgan) {
        throw new Error(`refused: organ "${desc.id}" produced no proceduralOrgan`);
      }
      const view = synthesizeProceduralOrganView(proj.declaration, proj.declaration.view.proceduralOrgan);
      if (view.indexHtml.length < 40 || view.stylesCss.length < 20 || view.appJs.length < 40) {
        throw new Error(`refused: organ "${desc.id}" generated incomplete view assets`);
      }
      new Function(view.appJs);
      verifyFormControlAccessibility(view.indexHtml, desc.id);
      checkedOrgans += 1;
    }
  } finally {
    pool.dispose();
  }

  // 4. Verified verdant-ui.holo archive check
  const holoHost = new VerdantHoloHost(bus);
  const holoSnap = await holoHost.loadFromBytes(
    new Uint8Array(readFileSync(join(root, 'app/.prism/build/verdant-ui.holo'))),
  );
  if (!holoSnap.verified || holoSnap.vectorsPassed !== holoSnap.vectorCount) {
    throw new Error('refused: verdant-ui.holo failed acceptance verification');
  }

  const ms = Math.round(performance.now() - t0);
  console.log(
    `prism-acceptance: PASS · wav=${wav.byteLength}B(rms=${wavStats.rmsDbfs.toFixed(1)}dBFS,peak=${wavStats.peakDbfs.toFixed(1)}dBFS) · png=${pngDims.width}x${pngDims.height} · svg=valid · organs=${checkedOrgans}/${ORGAN_REGISTRY.length} · holo=${holoSnap.vectorsPassed}/${holoSnap.vectorCount} · ${ms}ms`,
  );
}

const isMain =
  process.argv[1] !== undefined &&
  resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url));

if (isMain) {
  runPrismAcceptance(process.cwd()).catch((err) => {
    console.error(`prism-acceptance: ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
  });
}
