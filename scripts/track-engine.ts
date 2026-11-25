#!/usr/bin/env node
/**
 * Verdant Engine Release Tracker (`scripts/track-engine.ts`)
 *
 * Implements `verdant-studio`'s `REQ-ENGINE-TRACKING` (`Studio/Engine.lex.tex`) for VerdantUI:
 * checks the latest `afflom/verdant-engine` GitHub release and hashes its four authoritative
 * registers (`model/authorities.toml`, `model/ids.toml`, `site/generated/release-ledger.json`,
 * `standards.lock`) against `model/engine-pin.json`.
 *
 * Usage:
 *   npm run track:engine           # Check whether model/engine-pin.json matches latest release
 *   npm run track:engine -- --write # Update model/engine-pin.json to latest release
 */

import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

interface EngineRegisterSpec {
  readonly idPrefix: string;
  readonly path: string;
  readonly mediaType: string;
  readonly sha256: string;
}

interface EnginePinFile {
  readonly schema: 'verdant-ui/engine-pin/1';
  readonly requirement: 'REQ-ENGINE-TRACKING';
  readonly repository: string;
  readonly tag: string;
  readonly version: string;
  readonly commit: string;
  readonly publishedAt: string;
  readonly registers: readonly EngineRegisterSpec[];
}

const REGISTERS: ReadonlyArray<Omit<EngineRegisterSpec, 'sha256'>> = [
  {
    idPrefix: 'VERDANT-ENGINE-AUTHORITIES-',
    path: 'model/authorities.toml',
    mediaType: 'application/toml',
  },
  {
    idPrefix: 'VERDANT-ENGINE-IDS-',
    path: 'model/ids.toml',
    mediaType: 'application/toml',
  },
  {
    idPrefix: 'VERDANT-ENGINE-RELEASE-LEDGER-',
    path: 'site/generated/release-ledger.json',
    mediaType: 'application/json',
  },
  {
    idPrefix: 'VERDANT-ENGINE-STANDARDS-LOCK-',
    path: 'standards.lock',
    mediaType: 'application/json',
  },
];

async function fetchJson<T>(url: string): Promise<T> {
  const headers: Record<string, string> = {
    Accept: 'application/vnd.github+json',
    'User-Agent': 'VerdantUI-track-engine',
  };
  if (process.env.GITHUB_TOKEN) {
    headers.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
  }
  const res = await fetch(url, { headers });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} fetching ${url}`);
  }
  return (await res.json()) as T;
}

async function fetchSha256(url: string): Promise<string> {
  const res = await fetch(url, { headers: { 'User-Agent': 'VerdantUI-track-engine' } });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} fetching ${url}`);
  }
  const buf = new Uint8Array(await res.arrayBuffer());
  return createHash('sha256').update(buf).digest('hex');
}

async function main(): Promise<void> {
  const writeMode = process.argv.includes('--write');
  const pinPath = join(resolve(process.cwd()), 'model/engine-pin.json');
  const current = JSON.parse(readFileSync(pinPath, 'utf8')) as EnginePinFile;

  const release = await fetchJson<{
    tag_name: string;
    published_at: string;
  }>(`https://api.github.com/repos/${current.repository}/releases/latest`);

  const tagRef = await fetchJson<{
    object: { sha: string; type: string; url: string };
  }>(`https://api.github.com/repos/${current.repository}/git/ref/tags/${release.tag_name}`);

  let commitSha = tagRef.object.sha;
  if (tagRef.object.type === 'tag') {
    const annotated = await fetchJson<{ object: { sha: string } }>(tagRef.object.url);
    commitSha = annotated.object.sha;
  }

  const registers: EngineRegisterSpec[] = await Promise.all(
    REGISTERS.map(async (r) => ({
      ...r,
      sha256: await fetchSha256(
        `https://raw.githubusercontent.com/${current.repository}/${commitSha}/${r.path}`,
      ),
    })),
  );

  const next: EnginePinFile = {
    schema: 'verdant-ui/engine-pin/1',
    requirement: 'REQ-ENGINE-TRACKING',
    repository: current.repository,
    tag: release.tag_name,
    version: release.tag_name.replace(/^v/, ''),
    commit: commitSha,
    publishedAt: release.published_at,
    registers,
  };

  const isSame =
    current.tag === next.tag &&
    current.commit === next.commit &&
    JSON.stringify(current.registers) === JSON.stringify(next.registers);

  if (isSame) {
    console.log(
      `track-engine: UP TO DATE · ${next.repository}@${next.tag} (${next.commit.slice(0, 12)}) · 4/4 registers verified`,
    );
    return;
  }

  if (writeMode) {
    writeFileSync(pinPath, `${JSON.stringify(next, null, 2)}\n`);
    console.log(
      `track-engine: UPDATED model/engine-pin.json -> ${next.repository}@${next.tag} (${next.commit.slice(0, 12)})`,
    );
    return;
  }

  console.log(
    `track-engine: UPDATE AVAILABLE · pinned=${current.tag} (${current.commit.slice(0, 12)}) -> latest=${next.tag} (${next.commit.slice(0, 12)}). Run with --write to stage.`,
  );
}

const isMain =
  process.argv[1] !== undefined &&
  resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url));

if (isMain) {
  main().catch((err) => {
    console.error(`track-engine: ERROR — ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
  });
}
