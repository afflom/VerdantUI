#!/usr/bin/env node
/**
 * GitHub Pages Build Integrity Verifier (`scripts/verify-pages.ts`)
 *
 * Fast static verification of the assembled `dist/` bundle before deployment:
 *   1. Confirms all required static outputs exist (`index.html`, `404.html`, `.nojekyll`,
 *      `coi-serviceworker.js`, `models/stories260K.gguf`, `app.json`, `manifest.webmanifest`).
 *   2. Verifies `index.html` uses relative paths (`./`) safe for GitHub Pages subpath hosting,
 *      registers `./coi-serviceworker.js`, and every local `<script>` / `<link>` resolves on disk.
 *   3. Verifies `coi-serviceworker.js` sets COOP (`same-origin`) and COEP headers for
 *      `SharedArrayBuffer` multi-threaded WASM SIMD.
 *   4. Confirms built JS chunks include the multi-core compute worker and zero `navigator.gpu`
 *      or server-runtime references.
 */

import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

function fail(message: string): never {
  console.error(`verify-pages: REFUSED — ${message}`);
  process.exit(1);
}

export function verifyPagesBundle(distArg = 'dist', cwd = process.cwd()): void {
  const distDir = resolve(cwd, distArg);
  if (!existsSync(distDir)) {
    fail(`Build directory not found: ${distDir}`);
  }

  const requiredFiles = [
    'index.html',
    '404.html',
    '.nojekyll',
    'coi-serviceworker.js',
    'app.json',
    'manifest.webmanifest',
    'models/stories260K.gguf',
  ];
  for (const rel of requiredFiles) {
    const full = join(distDir, rel);
    if (!existsSync(full)) {
      fail(`Missing required Pages output: ${rel}`);
    }
    if (rel !== '.nojekyll' && statSync(full).size === 0) {
      fail(`Required Pages output is empty: ${rel}`);
    }
  }

  const indexHtml = readFileSync(join(distDir, 'index.html'), 'utf8');
  if (!/src=["']\.\/coi-serviceworker\.js["']/.test(indexHtml)) {
    fail('dist/index.html does not load ./coi-serviceworker.js');
  }

  // Ensure no broken root-absolute asset URLs sneak into index.html
  const attrRegex = /(?:src|href)=["']([^"']+)["']/g;
  let match: RegExpExecArray | null;
  let checkedRefs = 0;
  while ((match = attrRegex.exec(indexHtml)) !== null) {
    const ref = match[1];
    if (/^https?:\/\//i.test(ref) || ref.startsWith('data:') || ref.startsWith('#')) {
      continue;
    }
    if (ref.startsWith('/')) {
      fail(`Root-absolute reference "${ref}" in dist/index.html breaks GitHub Pages subpath hosting`);
    }
    const cleanRel = ref.replace(/^\.\//, '').split(/[?#]/)[0];
    if (!existsSync(join(distDir, cleanRel))) {
      fail(`Referenced asset "${ref}" does not exist in dist/`);
    }
    checkedRefs += 1;
  }

  const coiSw = readFileSync(join(distDir, 'coi-serviceworker.js'), 'utf8');
  if (
    !coiSw.includes('Cross-Origin-Opener-Policy') ||
    !coiSw.includes('Cross-Origin-Embedder-Policy')
  ) {
    fail('dist/coi-serviceworker.js is missing COOP/COEP headers');
  }

  const assetsDir = join(distDir, 'assets');
  if (!existsSync(assetsDir)) {
    fail('Missing dist/assets directory');
  }
  const jsFiles = readdirSync(assetsDir).filter((f) => f.endsWith('.js'));
  if (jsFiles.length === 0) {
    fail('No bundled JavaScript assets found in dist/assets');
  }

  let hasComputeWorker = false;
  let hasInferenceWorker = false;
  let totalJsBytes = 0;
  for (const file of jsFiles) {
    const full = join(assetsDir, file);
    totalJsBytes += statSync(full).size;
    if (/vtensorCompute\.worker/i.test(file)) hasComputeWorker = true;
    if (/inference\.worker/i.test(file)) hasInferenceWorker = true;
    const body = readFileSync(full, 'utf8');
    if (/\bnavigator\.gpu\b/.test(body)) {
      fail(`Built asset ${file} contains forbidden navigator.gpu reference`);
    }
    if (/MAJOR_CAPABILITY_SERVER_SIDE/.test(body)) {
      fail(`Built asset ${file} contains forbidden server capability reference`);
    }
  }

  if (!hasInferenceWorker || !hasComputeWorker) {
    fail(
      `Expected both inference.worker and vtensorCompute.worker bundles in dist/assets (found: ${jsFiles.join(', ')})`,
    );
  }

  console.log(
    `verify-pages: PASS · refs=${checkedRefs} · js_chunks=${jsFiles.length} (${(totalJsBytes / 1024).toFixed(1)} KiB) · workers=inference+compute · coi=COOP/COEP`,
  );
}

const isMain =
  process.argv[1] !== undefined &&
  resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url));

if (isMain) {
  verifyPagesBundle(process.argv[2] ?? 'dist');
}
