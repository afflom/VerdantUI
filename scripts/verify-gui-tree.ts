#!/usr/bin/env node
/**
 * Mechanical GUI View-Tree & WCAG 2.2 Parity Verifier (`scripts/verify-gui-tree.ts`)
 *
 * Parity with `afflom/verdant-studio/.github/scripts/gui-tree.py`:
 * Loads the built `app/.prism/build/verdant-ui.holo` archive through `VerdantHoloHost`,
 * verifies:
 *   1. `verdant\tgui\ttree` === `verdant\tobserver\tperceive` (live agent/human parity)
 *   2. `verdant\tgui\twcag` === `pass\t2.4.2,2.4.6,3.3.1,4.1.2` (WCAG 2.2 AA profile)
 *   3. Every registered organ in `ORGAN_REGISTRY` answers `verdant\tpart\t<mediaType>\tstatus` cleanly
 *   4. Writes/checks deterministic `surface/gui-resources.json`
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { VerdantHoloHost } from '../src/host/verdantHoloHost';
import { ObserverStreamBus } from '../src/engine/observerStreamBus';
import { ORGAN_REGISTRY } from '../src/engine/organRegistry';

export async function verifyGuiTree(cwd = process.cwd(), writeMode = false): Promise<void> {
  const root = resolve(cwd);
  const holoPath = join(root, 'app/.prism/build/verdant-ui.holo');
  if (!existsSync(holoPath)) {
    throw new Error(`refused: missing built archive ${holoPath}; run \`npm run prism:build\` first`);
  }

  const bus = new ObserverStreamBus();
  const host = new VerdantHoloHost(bus);
  const snap = await host.loadFromBytes(new Uint8Array(readFileSync(holoPath)));

  const guiTree = await host.dispatchGuiTree();
  const obsTree = await host.dispatchObserverPerceive();
  if (guiTree.refused || obsTree.refused) {
    throw new Error(`refused: gui/observer tree dispatch failed (${guiTree.raw} / ${obsTree.raw})`);
  }
  if (JSON.stringify(guiTree.parts) !== JSON.stringify(obsTree.parts)) {
    throw new Error(
      `refused: agent/human view tree mismatch (gui=${guiTree.raw} vs observer=${obsTree.raw})`,
    );
  }

  const wcag = await host.dispatchRaw('verdant\tgui\twcag');
  if (wcag.refused || !wcag.raw.startsWith('pass\t')) {
    throw new Error(`refused: WCAG 2.2 audit failed (${wcag.raw})`);
  }
  const wcagCriteria = wcag.raw.slice(5).split(',').filter(Boolean);

  const uniqueMediaTypes = Array.from(new Set(ORGAN_REGISTRY.map((o) => o.mediaType)));
  const verifiedParts: string[] = [];
  for (const mt of uniqueMediaTypes) {
    const res = await host.dispatchPart(mt, 'status');
    if (!res.ok || res.refused) {
      throw new Error(`refused: registered part "${mt}" failed status dispatch (${res.raw})`);
    }
    verifiedParts.push(mt);
  }

  const resources = {
    schema: 'verdant-ui/gui-resources/1',
    holoAddress: snap.address,
    vectorsVerified: `${snap.vectorsPassed}/${snap.vectorCount}`,
    surfaces: guiTree.parts,
    observerParity: true,
    wcag22Criteria: wcagCriteria,
    registeredMediaTypes: verifiedParts,
    organDescriptors: ORGAN_REGISTRY.map((o) => ({
      id: o.id,
      mediaType: o.mediaType,
      role: o.role,
      defaultCompartment: o.defaultCompartment,
      requiredGrants: o.requiredGrants,
    })),
  };

  const outPath = join(root, 'surface/gui-resources.json');
  const serialized = `${JSON.stringify(resources, null, 2)}\n`;
  if (writeMode || !existsSync(outPath)) {
    mkdirSync(dirname(outPath), { recursive: true });
    writeFileSync(outPath, serialized, 'utf8');
  } else {
    const existing = readFileSync(outPath, 'utf8');
    const existingObj = JSON.parse(existing) as Record<string, unknown>;
    // Compare semantic structure cleanly (surfaces, wcag22Criteria, registeredMediaTypes, organDescriptors)
    if (
      JSON.stringify(existingObj.surfaces) !== JSON.stringify(resources.surfaces) ||
      JSON.stringify(existingObj.wcag22Criteria) !== JSON.stringify(resources.wcag22Criteria) ||
      JSON.stringify(existingObj.registeredMediaTypes) !== JSON.stringify(resources.registeredMediaTypes) ||
      JSON.stringify(existingObj.organDescriptors) !== JSON.stringify(resources.organDescriptors)
    ) {
      throw new Error('refused: surface/gui-resources.json drifted from verdant-ui.holo; run with --write');
    }
  }

  console.log(
    `verify-gui-tree: PASS · surfaces=${guiTree.parts.length} · parity=true · wcag=${wcagCriteria.join(',')} · mediaTypes=${verifiedParts.length}`,
  );
}

const isMain =
  process.argv[1] !== undefined &&
  resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url));

if (isMain) {
  verifyGuiTree(process.cwd(), process.argv.includes('--write')).catch((err) => {
    console.error(`verify-gui-tree: ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
  });
}
