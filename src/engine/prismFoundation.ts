/**
 * Lightweight LexLean template foundation instantiator (`src/engine/prismFoundation.ts`).
 *
 * Kept dependency-free from UI/media organ synthesizers so `prismBuild.worker.ts`
 * never duplicates `verdant-studio-organs` (~390 kB) inside the worker bundle.
 */

import { sha256HexSync as prismSha256HexSync } from '@prism-code-internal/sha256.mjs';

const enc = new TextEncoder();

export function pascalModuleName(slug: string): string {
  const parts = slug
    .split(/[^a-zA-Z0-9]+/)
    .filter(Boolean)
    .map((p) => p[0].toUpperCase() + p.slice(1).toLowerCase());
  return `Prism${parts.join('') || 'App'}`;
}

export function instantiateAppFoundation(
  foundationFiles: Readonly<Record<string, string>>,
  slug: string,
): Record<string, string> {
  const mod = pascalModuleName(slug);
  const crate = `prism-${slug}`;
  const forms: readonly [string, string][] = [
    ['prism-template', crate],
    ['PrismTemplate', mod],
    ['"template"', JSON.stringify(slug)],
  ];
  const named = new Set(['lexlean.toml', 'lakefile.toml', 'lake-manifest.json', 'prismpm.toml']);
  const out: Record<string, string> = {};

  for (const [rel, content] of Object.entries(foundationFiles)) {
    let text = content;
    if (named.has(rel)) {
      for (const [from, to] of forms) {
        text = text.split(from).join(to);
      }
    }
    out[rel] = text;
  }

  if (out['lexlean.lock']) {
    const digest = (rel: string) => prismSha256HexSync(enc.encode(out[rel] ?? ''));
    let lock = out['lexlean.lock'];
    lock = lock.replace(
      /^project_config_sha256 = "[0-9a-f]{64}"$/m,
      () => `project_config_sha256 = "${digest('lexlean.toml')}"`,
    );
    lock = lock.replace(
      /(\[\[workspace_file\]\]\npath = "([^"\n]+)"\nsha256 = ")[0-9a-f]{64}"/g,
      (all, head: string, file: string) => (out[file] !== undefined ? `${head}${digest(file)}"` : all),
    );
    out['lexlean.lock'] = lock;
  }

  return out;
}
