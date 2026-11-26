#!/usr/bin/env node
/**
 * GitHub Pages Static Site Assembler (`scripts/assemble-pages.ts`)
 *
 * Finalizes `dist/` for deterministic GitHub Pages static deployment:
 *   1. Writes `dist/.nojekyll` and `dist/404.html` (SPA fallback).
 *   2. Writes `dist/manifest.webmanifest` (W3C Web App Manifest).
 *   3. Writes `dist/app.json` with repository, live Pages URL, commit SHA,
 *      compute policy (client-only, zero WebGPU, multi-threaded Verdant WASM SIMD),
 *      and workflow provenance metadata.
 */

import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { VERDANT_TENSOR_VERSION } from '../src/engine/verdantKernels';
import { DISABLED_DEFAULT_GGML_PROVIDERS, VTENSOR_CONTRACT } from '../src/engine/vtensorProvider';

export function assemblePagesSite(distArg = 'dist', cwd = process.cwd()): void {
  const distDir = resolve(cwd, distArg);
  mkdirSync(distDir, { recursive: true });

  const pkg = JSON.parse(readFileSync(join(cwd, 'package.json'), 'utf8')) as {
    name?: string;
    version?: string;
    description?: string;
  };
  const enginePin = JSON.parse(readFileSync(join(cwd, 'model/engine-pin.json'), 'utf8')) as {
    repository: string;
    tag: string;
    version: string;
    commit: string;
    registers: ReadonlyArray<{ idPrefix: string; path: string; sha256: string }>;
  };

  const studioPinPath = join(cwd, 'model/studio-pin.json');
  const studioPin = existsSync(studioPinPath)
    ? (JSON.parse(readFileSync(studioPinPath, 'utf8')) as {
        repository: string;
        commit: string;
      })
    : null;

  const ghRepo = process.env.GITHUB_REPOSITORY || 'afflom/VerdantUI';
  const [owner, repo] = ghRepo.split('/');
  const repoUrl = `https://github.com/${ghRepo}`;
  const liveUrl =
    process.env.PAGES_BASE_URL ||
    `https://${(owner || 'afflom').toLowerCase()}.github.io/${repo || 'VerdantUI'}/`;
  const commitSha = process.env.GITHUB_SHA || 'local';

  const appManifest = {
    schema: 'verdant-ui/pages-app/1',
    name: 'VerdantUI',
    version: pkg.version ?? '0.1.0',
    description: pkg.description ?? '',
    repository: repoUrl,
    live: liveUrl,
    commit: commitSha,
    publicBenefit: {
      freePublicBenefitService: true,
      license: 'Apache-2.0',
      zeroTracking: true,
      zeroPaywalls: true,
      zeroServers: true,
    },
    ...(studioPin
      ? {
          studioPin: {
            repository: studioPin.repository,
            commit: studioPin.commit,
          },
        }
      : {}),
    enginePin: {
      repository: enginePin.repository,
      tag: enginePin.tag,
      commit: enginePin.commit,
      registers: enginePin.registers.map((r) => ({
        idPrefix: r.idPrefix,
        path: r.path,
        sha256: r.sha256,
      })),
    },
    runtime: {
      deploymentModel: 'client-only',
      serverRuntime: false,
      computeSubstrate: 'verdant-engine/vTensor WASM SIMD (multi-threaded SharedArrayBuffer)',
      verdantTensorVersion: VERDANT_TENSOR_VERSION,
      verdantTensorCommit: enginePin.commit,
      providerName: VTENSOR_CONTRACT.providerName,
      ggmlAbiVersion: VTENSOR_CONTRACT.ggmlAbiVersion,
      webgpuAllowed: false,
      disabledProviders: [...DISABLED_DEFAULT_GGML_PROVIDERS],
      crossOriginIsolationWorker: './coi-serviceworker.js',
    },
    workflow: `${repoUrl}/actions/workflows/pages.yml`,
  };
  writeFileSync(join(distDir, 'app.json'), `${JSON.stringify(appManifest, null, 2)}\n`);

  const webManifest = {
    name: 'VerdantUI',
    short_name: 'VerdantUI',
    description: pkg.description ?? 'Client-side local AI workbench powered by Verdant Tensor.',
    start_url: './',
    scope: './',
    display: 'standalone',
    background_color: '#0B0F14',
    theme_color: '#0F5132',
  };
  writeFileSync(
    join(distDir, 'manifest.webmanifest'),
    `${JSON.stringify(webManifest, null, 2)}\n`,
  );

  writeFileSync(join(distDir, '.nojekyll'), '');
  const indexHtmlPath = join(distDir, 'index.html');
  if (existsSync(indexHtmlPath)) {
    copyFileSync(indexHtmlPath, join(distDir, '404.html'));
  }

  console.log(
    `assemble-pages: ${distDir} -> app.json, manifest.webmanifest, .nojekyll, 404.html (live=${liveUrl})`,
  );
}

const isMain =
  process.argv[1] !== undefined &&
  resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url));

if (isMain) {
  assemblePagesSite(process.argv[2] ?? 'dist');
}
