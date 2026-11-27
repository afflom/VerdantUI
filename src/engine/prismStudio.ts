/**
 * Verdant Studio & Prism (.holo) Model, Canonical SHA-256 Addressing, and IndexedDB Store.
 *
 * Standards-first client-side Prism representation:
 * - Structured `{ html, css, js }` realization for fast targeted local-model iteration
 * - Separate mutable `draft` vs. immutable `published` snapshot + 20-step Undo history
 * - JSON Schema 2020-12 input stream contract (reused directly by Assistant-UI `ModelContext.tools`)
 * - Canonical sorted-key JSON + FIPS 180-4 SHA-256 `.holo` archive (`verdant-ui/holo/1`)
 * - Single async IndexedDB store (`verdant-ui-studio-v1`) with Node in-memory fallback
 */

import {
  createDeclarativeStarter,
  dispatchStudioOp,
  fieldsToJsonSchema,
  upgradeDeclarationV1ToV2,
  type DeclarativeStarterId,
  type PrismDeclaration,
  type StudioApiOp,
} from './verdantStudioApi';
import { compilePrismDeclaration } from './prismCompiler';
import type { WorkerVerifyPayload } from './prismBuildClient';

export const HOLO_ARCHIVE_SCHEMA = 'verdant-ui/holo/1' as const;
const MAX_REVISIONS = 20;
const IDB_NAME = 'verdant-ui-studio-v1';
const IDB_VERSION = 1;
const STORE_PROJECTS = 'projects';
const STORE_PRISM_KV = 'prism_kv';

export type PrismProjectKind = 'declarative' | 'external-holo' | 'legacy-v1';

export interface SealedHoloRecord {
  readonly bytes: Uint8Array;
  readonly address: string;
  readonly schema: string;
  readonly semanticId: string | null;
  readonly proved: boolean;
  readonly theorems: readonly {
    readonly theorem: string;
    readonly status: string;
    readonly axioms?: readonly string[];
  }[];
  readonly vectorsPassed: number;
  readonly vectorsTotal: number;
  readonly carries: {
    readonly snapshot: boolean;
    readonly sources: boolean;
    readonly proof: boolean;
  };
  readonly viewFiles?: Readonly<Record<string, string>> | null;
  readonly viewEntry?: string | null;
  readonly sealedAt: number;
}

export interface JsonSchema2020Object {
  readonly type: 'object';
  readonly properties: Record<
    string,
    {
      readonly type: 'string' | 'number' | 'boolean' | 'object' | 'array';
      readonly description?: string;
      readonly enum?: string[];
    }
  >;
  readonly required?: string[];
}

export interface PrismStreamContract {
  readonly inputs: JsonSchema2020Object;
  readonly outputs: readonly string[];
}

export interface PrismGrants {
  readonly observer: boolean;
  readonly storage: boolean;
  readonly browser: boolean;
  /** Denied by default per FK-07 until explicitly granted by the user. */
  readonly network: boolean;
}

export interface PrismAppBundle {
  readonly html: string;
  readonly css: string;
  readonly js: string;
}

export interface VerdantPrismSnapshot {
  readonly name: string;
  readonly summary: string;
  readonly streams: PrismStreamContract;
  readonly grants: PrismGrants;
  readonly app: PrismAppBundle;
  readonly digest: string;
}

export interface StudioChatMessage {
  readonly id: string;
  readonly role: 'user' | 'assistant';
  readonly content: string;
  readonly createdAt: number;
  readonly updatedSections?: readonly ('html' | 'css' | 'js' | 'meta')[];
  readonly badges?: readonly string[];
  readonly needsLocalModel?: boolean;
}

export interface VerdantPrismProject {
  readonly id: string;
  readonly kind?: PrismProjectKind;
  readonly draft: VerdantPrismSnapshot;
  readonly published: VerdantPrismSnapshot | null;
  readonly publishedAt: number | null;
  readonly revisions: readonly VerdantPrismSnapshot[];
  readonly messages: readonly StudioChatMessage[];
  /** Declarative registers (authoritative single source for declarative Prisms). */
  readonly declaration?: PrismDeclaration;
  readonly declarationRevisions?: readonly PrismDeclaration[];
  /** Latest real binary `prism/holo/6` archive sealed from the current draft. */
  readonly sealedDraft?: SealedHoloRecord | null;
  /** Immutable published binary `prism/holo/6` archive (also used for imported external `.holo`). */
  readonly publishedHolo?: SealedHoloRecord | null;
  readonly createdAt: number;
  readonly updatedAt: number;
}

export interface PublishedPrismSummary {
  readonly id: string;
  readonly slug: string;
  readonly name: string;
  readonly summary: string;
  readonly digest: string;
  readonly streams: PrismStreamContract;
  readonly grants: PrismGrants;
}

export type PrismStarterId = 'blank' | 'kanban' | 'focus' | 'budget' | 'writer';

/** Matches an explicit request for a built-in starter app when Model Stream is Off. */
export function matchStarterFromPrompt(prompt: string): PrismStarterId | null {
  const lower = prompt.trim().toLowerCase();
  if (!lower) return null;
  if (/\b(kanban|task\s*board|todo\s*board|to-do\s*board)\b/.test(lower)) return 'kanban';
  if (/\b(pomodoro|focus\s*timer|countdown\s*timer|study\s*timer)\b/.test(lower)) return 'focus';
  if (/\b(expense|budget\s*tracker|spending\s*tracker|expense\s*sheet)\b/.test(lower)) return 'budget';
  if (/\b(writing\s*assistant|text\s*summarizer|draft\s*refiner|grammar\s*polish)\b/.test(lower)) {
    return 'writer';
  }
  return null;
}

export const DEFAULT_PRISM_GRANTS: PrismGrants = Object.freeze({
  observer: true,
  storage: true,
  browser: true,
  network: false,
});

export const DEFAULT_STREAM_CONTRACT: PrismStreamContract = {
  inputs: {
    type: 'object',
    properties: {
      action: { type: 'string', description: 'Action or command to dispatch to this Prism.' },
      message: { type: 'string', description: 'Freeform user or Observer input text.' },
    },
  },
  outputs: ['prism.ready', 'prism.output'],
};

// ============================================================================
// 1. Canonical JSON (RFC 8785-style) & FIPS 180-4 SHA-256 Digest
// ============================================================================

export function canonicalizeJson(value: unknown): string {
  if (value === null || typeof value !== 'object') {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => canonicalizeJson(item)).join(',')}]`;
  }
  const obj = value as Record<string, unknown>;
  const keys = Object.keys(obj)
    .filter((k) => obj[k] !== undefined)
    .sort();
  return `{${keys.map((k) => `${JSON.stringify(k)}:${canonicalizeJson(obj[k])}`).join(',')}}`;
}

const SHA256_K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

/** Synchronous FIPS 180-4 SHA-256 returning lowercase 64-char hex (identical to Web Crypto SHA-256). */
export function sha256HexSync(text: string): string {
  const utf8 = new TextEncoder().encode(text);
  const bitLen = utf8.length * 8;
  const paddedLen = Math.ceil((utf8.length + 9) / 64) * 64;
  const padded = new Uint8Array(paddedLen);
  padded.set(utf8);
  padded[utf8.length] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(paddedLen - 8, Math.floor(bitLen / 0x100000000), false);
  view.setUint32(paddedLen - 4, bitLen >>> 0, false);

  let h0 = 0x6a09e667;
  let h1 = 0xbb67ae85;
  let h2 = 0x3c6ef372;
  let h3 = 0xa54ff53a;
  let h4 = 0x510e527f;
  let h5 = 0x9b05688c;
  let h6 = 0x1f83d9ab;
  let h7 = 0x5be0cd19;

  const w = new Uint32Array(64);
  const rotr = (x: number, n: number) => (x >>> n) | (x << (32 - n));

  for (let offset = 0; offset < paddedLen; offset += 64) {
    for (let i = 0; i < 16; i++) {
      w[i] = view.getUint32(offset + i * 4, false);
    }
    for (let i = 16; i < 64; i++) {
      const s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3);
      const s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }

    let a = h0;
    let b = h1;
    let c = h2;
    let d = h3;
    let e = h4;
    let f = h5;
    let g = h6;
    let h = h7;

    for (let i = 0; i < 64; i++) {
      const S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const ch = (e & f) ^ (~e & g);
      const temp1 = (h + S1 + ch + SHA256_K[i] + w[i]) >>> 0;
      const S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (S0 + maj) >>> 0;

      h = g;
      g = f;
      f = e;
      e = (d + temp1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temp1 + temp2) >>> 0;
    }

    h0 = (h0 + a) >>> 0;
    h1 = (h1 + b) >>> 0;
    h2 = (h2 + c) >>> 0;
    h3 = (h3 + d) >>> 0;
    h4 = (h4 + e) >>> 0;
    h5 = (h5 + f) >>> 0;
    h6 = (h6 + g) >>> 0;
    h7 = (h7 + h) >>> 0;
  }

  return [h0, h1, h2, h3, h4, h5, h6, h7]
    .map((n) => n.toString(16).padStart(8, '0'))
    .join('');
}

export function computePrismDigest(
  snapshot: Omit<VerdantPrismSnapshot, 'digest'>,
): string {
  const payload = canonicalizeJson({
    schema: HOLO_ARCHIVE_SCHEMA,
    name: snapshot.name,
    summary: snapshot.summary,
    streams: snapshot.streams,
    grants: snapshot.grants,
    app: snapshot.app,
  });
  return `sha256:${sha256HexSync(payload)}`;
}

export function shortPrismDigest(digest: string): string {
  const raw = digest.replace(/^sha256:/, '');
  return raw.slice(0, 10);
}

export function slugifyPrismName(name: string, idFallback = 'prism'): string {
  const clean = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 32);
  return clean || idFallback.replace(/[^a-z0-9_]/gi, '_').slice(0, 24);
}

export function prismToolName(summary: Pick<PublishedPrismSummary, 'slug'>): string {
  return `prism_${summary.slug}`;
}

export function buildSnapshot(
  fields: Omit<VerdantPrismSnapshot, 'digest'>,
): VerdantPrismSnapshot {
  const normalized: Omit<VerdantPrismSnapshot, 'digest'> = {
    name: fields.name.trim() || 'Untitled Prism',
    summary: fields.summary.trim(),
    streams: fields.streams,
    grants: {
      observer: Boolean(fields.grants.observer),
      storage: Boolean(fields.grants.storage),
      browser: fields.grants.browser !== false,
      network: Boolean(fields.grants.network),
    },
    app: {
      html: fields.app.html,
      css: fields.app.css,
      js: fields.app.js,
    },
  };
  return {
    ...normalized,
    digest: computePrismDigest(normalized),
  };
}

// ============================================================================
// 2. `.holo` Archive Serialization & Verification
// ============================================================================

export interface HoloArchiveEnvelope {
  readonly schema: typeof HOLO_ARCHIVE_SCHEMA;
  readonly digest: string;
  readonly name: string;
  readonly summary: string;
  readonly streams: PrismStreamContract;
  readonly grants: PrismGrants;
  readonly app: PrismAppBundle;
}

export function serializeHoloArchive(snapshot: VerdantPrismSnapshot): string {
  const verifiedDigest = computePrismDigest(snapshot);
  const envelope: HoloArchiveEnvelope = {
    schema: HOLO_ARCHIVE_SCHEMA,
    digest: verifiedDigest,
    name: snapshot.name,
    summary: snapshot.summary,
    streams: snapshot.streams,
    grants: snapshot.grants,
    app: snapshot.app,
  };
  return JSON.stringify(envelope, null, 2);
}

export function parseHoloArchive(raw: string | Uint8Array): VerdantPrismSnapshot {
  if (raw instanceof Uint8Array) {
    // Detect binary HOLO archive magic gracefully
    if (raw.length >= 4 && raw[0] === 0x48 && raw[1] === 0x4f && raw[2] === 0x4c && raw[3] === 0x4f) {
      throw new Error(
        'Binary .holo container detected. Export a VerdantUI .holo package or provide a JSON-encoded verdant-ui/holo/1 archive.',
      );
    }
    raw = new TextDecoder().decode(raw);
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    throw new Error(
      `Invalid .holo file: ${err instanceof Error ? err.message : 'malformed JSON'}`,
    );
  }

  if (!parsed || typeof parsed !== 'object') {
    throw new Error('Invalid .holo file: expected JSON object.');
  }

  const obj = parsed as Record<string, unknown>;
  if (obj.schema !== HOLO_ARCHIVE_SCHEMA) {
    throw new Error(
      `Unsupported .holo schema "${String(obj.schema ?? 'missing')}" (expected "${HOLO_ARCHIVE_SCHEMA}").`,
    );
  }

  if (typeof obj.name !== 'string' || !obj.app || typeof obj.app !== 'object') {
    throw new Error('Invalid .holo archive: missing required name or app bundle.');
  }

  const appObj = obj.app as Record<string, unknown>;
  const grantsObj = (obj.grants ?? {}) as Record<string, unknown>;
  const streamsObj = (obj.streams ?? {}) as Record<string, unknown>;
  const inputsObj =
    streamsObj.inputs && typeof streamsObj.inputs === 'object'
      ? (streamsObj.inputs as JsonSchema2020Object)
      : DEFAULT_STREAM_CONTRACT.inputs;
  const outputsArr = Array.isArray(streamsObj.outputs)
    ? streamsObj.outputs.filter((x): x is string => typeof x === 'string')
    : [...DEFAULT_STREAM_CONTRACT.outputs];

  const candidate: Omit<VerdantPrismSnapshot, 'digest'> = {
    name: obj.name,
    summary: typeof obj.summary === 'string' ? obj.summary : '',
    streams: {
      inputs: inputsObj,
      outputs: outputsArr,
    },
    grants: {
      observer: grantsObj.observer !== false,
      storage: grantsObj.storage !== false,
      browser: grantsObj.browser !== false,
      // Imported Prisms always preserve declared network flag, but UI still gates via FK-07
      network: grantsObj.network === true,
    },
    app: {
      html: typeof appObj.html === 'string' ? appObj.html : '',
      css: typeof appObj.css === 'string' ? appObj.css : '',
      js: typeof appObj.js === 'string' ? appObj.js : '',
    },
  };

  const expectedDigest = computePrismDigest(candidate);
  if (typeof obj.digest === 'string' && obj.digest !== expectedDigest) {
    throw new Error(
      `Integrity check failed: .holo digest mismatch (declared ${shortPrismDigest(obj.digest)}, computed ${shortPrismDigest(expectedDigest)}).`,
    );
  }

  return { ...candidate, digest: expectedDigest };
}

// ============================================================================
// 3. Built-in High-Craft Interactive Starters
// ============================================================================

export function createStarterSnapshot(starterId: PrismStarterId): VerdantPrismSnapshot {
  switch (starterId) {
    case 'kanban':
      return buildSnapshot({
        name: 'Task Board',
        summary: 'Three-column Kanban board with local persistence, Markdown export, and Thread task commands.',
        streams: {
          inputs: {
            type: 'object',
            properties: {
              action: {
                type: 'string',
                enum: ['list', 'add', 'complete', 'status'],
                description: 'List tasks, add a new task, or mark a task done.',
              },
              message: { type: 'string', description: 'Task title when adding or completing.' },
            },
          },
          outputs: ['task.added', 'task.moved', 'task.completed'],
        },
        grants: { ...DEFAULT_PRISM_GRANTS },
        app: {
          html: `<div class="kb-app">
  <header class="kb-head">
    <div>
      <h1>Task Board</h1>
      <p id="counts" class="v-muted">0 tasks</p>
    </div>
    <button id="exportBtn" class="v-btn">Export Markdown</button>
  </header>
  <form id="addForm" class="kb-form">
    <input id="taskInput" class="v-input" placeholder="Add a new task..." />
    <button type="submit" class="v-btn v-btn-primary">Add</button>
  </form>
  <div class="kb-cols">
    <section class="kb-col" data-col="todo"><h2>To Do</h2><div class="kb-list" id="col-todo"></div></section>
    <section class="kb-col" data-col="doing"><h2>In Progress</h2><div class="kb-list" id="col-doing"></div></section>
    <section class="kb-col" data-col="done"><h2>Done</h2><div class="kb-list" id="col-done"></div></section>
  </div>
</div>`,
          css: `.kb-app { max-width: 920px; margin: 0 auto; display: flex; flex-direction: column; gap: 14px; }
.kb-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.kb-head h1 { margin: 0; font-size: 18px; font-weight: 700; }
.kb-head p { margin: 2px 0 0; font-size: 12px; }
.kb-form { display: flex; gap: 8px; }
.kb-cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.kb-col { background: var(--v-surface); border: 1px solid var(--v-border); border-radius: 12px; padding: 12px; display: flex; flex-direction: column; gap: 8px; min-height: 220px; }
.kb-col h2 { margin: 0; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--v-muted); }
.kb-list { display: flex; flex-direction: column; gap: 8px; }
.kb-card { border: 1px solid var(--v-border); border-radius: 8px; padding: 10px; background: var(--v-bg); display: flex; flex-direction: column; gap: 8px; }
.kb-title { font-size: 13px; font-weight: 500; word-break: break-word; }
.kb-actions { display: flex; gap: 6px; justify-content: flex-end; }
.kb-mini { height: 26px; padding: 0 8px; font-size: 11px; border-radius: 6px; }`,
          js: `let tasks = [
  { id: 't1', title: 'Draft architecture notes', col: 'todo' },
  { id: 't2', title: 'Review open tasks', col: 'doing' },
];
const cols = ['todo', 'doing', 'done'];
const countsEl = document.getElementById('counts');
const inputEl = document.getElementById('taskInput');

async function save() {
  await window.verdant.storage.set('tasks', tasks);
}

function render() {
  const doneCount = tasks.filter((t) => t.col === 'done').length;
  countsEl.textContent = tasks.length + ' total · ' + doneCount + ' completed';
  for (const col of cols) {
    const container = document.getElementById('col-' + col);
    container.innerHTML = '';
    const items = tasks.filter((t) => t.col === col);
    for (const item of items) {
      const card = document.createElement('div');
      card.className = 'kb-card';
      const title = document.createElement('div');
      title.className = 'kb-title';
      title.textContent = item.title;
      const actions = document.createElement('div');
      actions.className = 'kb-actions';
      const idx = cols.indexOf(col);
      if (idx > 0) {
        const back = document.createElement('button');
        back.className = 'v-btn kb-mini';
        back.textContent = '←';
        back.onclick = async () => { item.col = cols[idx - 1]; await save(); render(); };
        actions.appendChild(back);
      }
      if (idx < cols.length - 1) {
        const next = document.createElement('button');
        next.className = 'v-btn kb-mini';
        next.textContent = idx === 1 ? 'Done ✓' : '→';
        next.onclick = async () => {
          item.col = cols[idx + 1];
          await save();
          window.verdant.emit(item.col === 'done' ? 'task.completed' : 'task.moved', { title: item.title, col: item.col });
          render();
        };
        actions.appendChild(next);
      }
      const del = document.createElement('button');
      del.className = 'v-btn kb-mini';
      del.textContent = '×';
      del.onclick = async () => { tasks = tasks.filter((t) => t.id !== item.id); await save(); render(); };
      actions.appendChild(del);
      card.append(title, actions);
      container.appendChild(card);
    }
  }
}

document.getElementById('addForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const title = inputEl.value.trim();
  if (!title) return;
  tasks.unshift({ id: 't_' + Date.now().toString(36), title, col: 'todo' });
  inputEl.value = '';
  await save();
  window.verdant.emit('task.added', { title });
  render();
});

document.getElementById('exportBtn').addEventListener('click', async () => {
  const md = cols.map((c) => '## ' + c.toUpperCase() + '\\n' + tasks.filter((t) => t.col === c).map((t) => '- ' + t.title).join('\\n')).join('\\n\\n');
  await window.verdant.browser.downloadFile('task-board.md', md, 'text/markdown');
  await window.verdant.browser.notify('Exported', 'Saved task-board.md');
});

window.verdant.onInput(async (payload) => {
  const raw = String(payload?.message || '').trim();
  const action = String(payload?.action || '').toLowerCase();
  if (action === 'add' || /^add\\s+/i.test(raw)) {
    const title = raw.replace(/^add\\s+/i, '').replace(/^["']|["']$/g, '').trim();
    if (title) {
      tasks.unshift({ id: 't_' + Date.now().toString(36), title, col: 'todo' });
      await save();
      render();
      return { added: title, total: tasks.length };
    }
  }
  return {
    total: tasks.length,
    todo: tasks.filter((t) => t.col === 'todo').map((t) => t.title),
    inProgress: tasks.filter((t) => t.col === 'doing').map((t) => t.title),
    done: tasks.filter((t) => t.col === 'done').map((t) => t.title),
  };
});

window.verdant.storage.get('tasks').then((saved) => {
  if (Array.isArray(saved) && saved.length > 0) tasks = saved;
  render();
});`,
        },
      });

    case 'focus':
      return buildSnapshot({
        name: 'Focus Timer',
        summary: 'Circular Pomodoro focus timer with audio chime, daily session tracking, and Thread status.',
        streams: {
          inputs: {
            type: 'object',
            properties: {
              action: {
                type: 'string',
                enum: ['status', 'start', 'pause', 'reset'],
                description: 'Query or control the focus timer.',
              },
            },
          },
          outputs: ['focus.started', 'focus.completed'],
        },
        grants: { ...DEFAULT_PRISM_GRANTS },
        app: {
          html: `<div class="ft-wrap">
  <div class="ft-modes">
    <button class="v-btn ft-tab active" data-mins="25" data-label="Focus">Focus · 25m</button>
    <button class="v-btn ft-tab" data-mins="5" data-label="Short Break">Break · 5m</button>
    <button class="v-btn ft-tab" data-mins="15" data-label="Long Break">Long · 15m</button>
  </div>
  <div class="ft-ring-box">
    <svg viewBox="0 0 180 180" class="ft-svg">
      <circle cx="90" cy="90" r="78" class="ft-track" />
      <circle id="arc" cx="90" cy="90" r="78" class="ft-progress" stroke-dasharray="490" stroke-dashoffset="0" />
    </svg>
    <div class="ft-center">
      <div id="clock" class="ft-clock">25:00</div>
      <div id="modeLabel" class="v-muted">Focus</div>
    </div>
  </div>
  <div class="ft-controls">
    <button id="toggleBtn" class="v-btn v-btn-primary">Start</button>
    <button id="resetBtn" class="v-btn">Reset</button>
  </div>
  <div id="stats" class="v-muted ft-stats">Completed sessions today: 0</div>
</div>`,
          css: `.ft-wrap { max-width: 420px; margin: 12px auto; display: flex; flex-direction: column; align-items: center; gap: 18px; }
.ft-modes { display: flex; gap: 6px; }
.ft-tab { height: 32px; font-size: 12px; }
.ft-tab.active { background: var(--v-accent); color: var(--v-accent-text); border-color: transparent; }
.ft-ring-box { position: relative; width: 210px; height: 210px; display: grid; place-items: center; }
.ft-svg { width: 100%; height: 100%; transform: rotate(-90deg); }
.ft-track { fill: none; stroke: var(--v-border); stroke-width: 8; }
.ft-progress { fill: none; stroke: var(--v-accent); stroke-width: 8; stroke-linecap: round; transition: stroke-dashoffset 250ms linear; }
.ft-center { position: absolute; text-align: center; }
.ft-clock { font-size: 40px; font-weight: 700; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
.ft-controls { display: flex; gap: 10px; }
.ft-stats { font-size: 12px; }`,
          js: `const CIRC = 2 * Math.PI * 78;
let totalSec = 25 * 60;
let remaining = totalSec;
let modeLabel = 'Focus';
let running = false;
let sessions = 0;
let timer = null;

const clockEl = document.getElementById('clock');
const arcEl = document.getElementById('arc');
const labelEl = document.getElementById('modeLabel');
const toggleBtn = document.getElementById('toggleBtn');
const statsEl = document.getElementById('stats');

function fmt(s) {
  const m = Math.floor(s / 60).toString().padStart(2, '0');
  const sec = (s % 60).toString().padStart(2, '0');
  return m + ':' + sec;
}

function chime() {
  try {
    const ctx = new AudioContext();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.value = 660;
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.5);
  } catch (_) {}
}

function render() {
  clockEl.textContent = fmt(remaining);
  labelEl.textContent = modeLabel;
  arcEl.style.strokeDasharray = String(CIRC);
  arcEl.style.strokeDashoffset = String(CIRC * (1 - remaining / Math.max(1, totalSec)));
  toggleBtn.textContent = running ? 'Pause' : 'Start';
  statsEl.textContent = 'Completed sessions: ' + sessions;
}

function start() {
  if (running) return;
  running = true;
  window.verdant.emit('focus.started', { mode: modeLabel, remaining });
  timer = setInterval(async () => {
    remaining -= 1;
    if (remaining <= 0) {
      clearInterval(timer);
      running = false;
      remaining = 0;
      if (modeLabel === 'Focus') {
        sessions += 1;
        await window.verdant.storage.set('sessions', sessions);
      }
      chime();
      window.verdant.emit('focus.completed', { mode: modeLabel, sessions });
      await window.verdant.browser.notify('Timer complete', modeLabel + ' session finished.');
    }
    render();
  }, 1000);
  render();
}

function pause() {
  running = false;
  if (timer) clearInterval(timer);
  render();
}

toggleBtn.addEventListener('click', () => (running ? pause() : start()));
document.getElementById('resetBtn').addEventListener('click', () => { pause(); remaining = totalSec; render(); });

document.querySelectorAll('.ft-tab').forEach((btn) => {
  btn.addEventListener('click', () => {
    pause();
    document.querySelectorAll('.ft-tab').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    totalSec = Number(btn.dataset.mins) * 60;
    remaining = totalSec;
    modeLabel = btn.dataset.label;
    render();
  });
});

window.verdant.onInput(async (payload) => {
  const act = String(payload?.action || payload?.message || 'status').toLowerCase();
  if (act.includes('start')) start();
  else if (act.includes('pause') || act.includes('stop')) pause();
  else if (act.includes('reset')) { pause(); remaining = totalSec; render(); }
  return { mode: modeLabel, remaining: fmt(remaining), running, completedSessions: sessions };
});

window.verdant.storage.get('sessions').then((v) => {
  if (typeof v === 'number') sessions = v;
  render();
});`,
        },
      });

    case 'budget':
      return buildSnapshot({
        name: 'Expense Tracker',
        summary: 'Interactive expense tracker with live category bars, CSV export, and Thread summaries.',
        streams: {
          inputs: {
            type: 'object',
            properties: {
              action: {
                type: 'string',
                enum: ['summary', 'add'],
                description: 'Summarize spending or add an expense.',
              },
              message: { type: 'string', description: 'Expense description and amount (e.g. "Coffee 5.50").' },
            },
          },
          outputs: ['expense.added'],
        },
        grants: { ...DEFAULT_PRISM_GRANTS },
        app: {
          html: `<div class="ex-app">
  <header class="ex-head">
    <div>
      <h1>Expense Tracker</h1>
      <div id="total" class="ex-total">$0.00</div>
    </div>
    <button id="csvBtn" class="v-btn">Export CSV</button>
  </header>
  <form id="exForm" class="ex-form">
    <input id="desc" class="v-input" placeholder="Description (e.g. Coffee)" required />
    <input id="amt" class="v-input" type="number" step="0.01" min="0" placeholder="Amount" style="max-width:110px" required />
    <select id="cat" class="v-input" style="max-width:130px">
      <option>Food</option><option>Tools</option><option>Travel</option><option>Other</option>
    </select>
    <button type="submit" class="v-btn v-btn-primary">Add</button>
  </form>
  <div id="bars" class="v-card ex-bars"></div>
  <div id="rows" class="ex-rows"></div>
</div>`,
          css: `.ex-app { max-width: 720px; margin: 0 auto; display: flex; flex-direction: column; gap: 14px; }
.ex-head { display: flex; align-items: center; justify-content: space-between; }
.ex-head h1 { margin: 0; font-size: 16px; color: var(--v-muted); font-weight: 600; }
.ex-total { font-size: 28px; font-weight: 700; font-variant-numeric: tabular-nums; margin-top: 2px; }
.ex-form { display: flex; flex-wrap: wrap; gap: 8px; }
.ex-bars { display: flex; flex-direction: column; gap: 8px; }
.ex-bar-row { display: grid; grid-template-columns: 72px 1fr 72px; align-items: center; gap: 10px; font-size: 12px; }
.ex-bar-track { height: 8px; border-radius: 999px; background: var(--v-border); overflow: hidden; }
.ex-bar-fill { height: 100%; background: var(--v-accent); border-radius: 999px; }
.ex-rows { display: flex; flex-direction: column; gap: 6px; }
.ex-item { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 8px 12px; border: 1px solid var(--v-border); border-radius: 8px; background: var(--v-surface); font-size: 13px; }`,
          js: `let items = [
  { id: 'e1', desc: 'Domain renewal', amount: 14.0, category: 'Tools' },
  { id: 'e2', desc: 'Espresso', amount: 4.5, category: 'Food' },
];
const totalEl = document.getElementById('total');
const barsEl = document.getElementById('bars');
const rowsEl = document.getElementById('rows');

async function save() { await window.verdant.storage.set('expenses', items); }

function summarize() {
  const total = items.reduce((s, x) => s + x.amount, 0);
  const byCategory = {};
  for (const x of items) byCategory[x.category] = (byCategory[x.category] || 0) + x.amount;
  return { total: Number(total.toFixed(2)), count: items.length, byCategory };
}

function render() {
  const sum = summarize();
  totalEl.textContent = '$' + sum.total.toFixed(2);
  barsEl.innerHTML = '';
  const cats = Object.entries(sum.byCategory).sort((a, b) => b[1] - a[1]);
  if (cats.length === 0) {
    barsEl.innerHTML = '<span class="v-muted" style="font-size:12px">No expenses yet.</span>';
  } else {
    for (const [cat, val] of cats) {
      const pct = sum.total > 0 ? Math.round((val / sum.total) * 100) : 0;
      const row = document.createElement('div');
      row.className = 'ex-bar-row';
      row.innerHTML = '<span>' + cat + '</span><div class="ex-bar-track"><div class="ex-bar-fill" style="width:' + pct + '%"></div></div><span style="text-align:right;font-variant-numeric:tabular-nums">$' + val.toFixed(2) + '</span>';
      barsEl.appendChild(row);
    }
  }
  rowsEl.innerHTML = '';
  for (const it of items) {
    const el = document.createElement('div');
    el.className = 'ex-item';
    const left = document.createElement('div');
    left.innerHTML = '<strong>' + it.desc + '</strong> <span class="v-pill" style="margin-left:6px">' + it.category + '</span>';
    const right = document.createElement('div');
    right.style.display = 'flex';
    right.style.alignItems = 'center';
    right.style.gap = '8px';
    const amt = document.createElement('span');
    amt.style.fontVariantNumeric = 'tabular-nums';
    amt.textContent = '$' + it.amount.toFixed(2);
    const del = document.createElement('button');
    del.className = 'v-btn';
    del.style.height = '26px';
    del.style.padding = '0 8px';
    del.textContent = '×';
    del.onclick = async () => { items = items.filter((x) => x.id !== it.id); await save(); render(); };
    right.append(amt, del);
    el.append(left, right);
    rowsEl.appendChild(el);
  }
}

document.getElementById('exForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const desc = document.getElementById('desc').value.trim();
  const amount = Number(document.getElementById('amt').value);
  const category = document.getElementById('cat').value;
  if (!desc || !Number.isFinite(amount) || amount <= 0) return;
  items.unshift({ id: 'e_' + Date.now().toString(36), desc, amount, category });
  document.getElementById('desc').value = '';
  document.getElementById('amt').value = '';
  await save();
  window.verdant.emit('expense.added', { desc, amount, category });
  render();
});

document.getElementById('csvBtn').addEventListener('click', async () => {
  const csv = 'Description,Category,Amount\\n' + items.map((x) => JSON.stringify(x.desc) + ',' + x.category + ',' + x.amount.toFixed(2)).join('\\n');
  await window.verdant.browser.downloadFile('expenses.csv', csv, 'text/csv');
  await window.verdant.browser.notify('CSV Exported', 'Saved expenses.csv');
});

window.verdant.onInput(async (payload) => {
  const raw = String(payload?.message || '').trim();
  const m = raw.match(/^(?:add\\s+)?(.+?)\\s+\\$?([0-9]+(?:\\.[0-9]{1,2})?)$/i);
  if (m) {
    const desc = m[1].trim();
    const amount = Number(m[2]);
    items.unshift({ id: 'e_' + Date.now().toString(36), desc, amount, category: 'Other' });
    await save();
    render();
    return { added: { desc, amount }, ...summarize() };
  }
  return summarize();
});

window.verdant.storage.get('expenses').then((saved) => {
  if (Array.isArray(saved) && saved.length > 0) items = saved;
  render();
});`,
        },
      });

    case 'writer':
      return buildSnapshot({
        name: 'AI Writing Assistant',
        summary: 'Local AI writing tool to summarize, tighten, or extract action items using VerdantUI.',
        streams: {
          inputs: {
            type: 'object',
            properties: {
              text: { type: 'string', description: 'Source text to transform.' },
              mode: {
                type: 'string',
                enum: ['summarize', 'tighten', 'actions', 'polish'],
                description: 'Editing transformation.',
              },
            },
            required: ['text'],
          },
          outputs: ['writer.completed'],
        },
        grants: { ...DEFAULT_PRISM_GRANTS },
        app: {
          html: `<div class="wr-app">
  <header class="wr-head">
    <h1>AI Writing Assistant</h1>
    <select id="mode" class="v-input" style="width:auto">
      <option value="tighten">Tighten &amp; Clarify</option>
      <option value="summarize">Summarize (3 bullets)</option>
      <option value="actions">Extract Action Items</option>
      <option value="polish">Polish Grammar</option>
    </select>
  </header>
  <textarea id="source" class="v-input" rows="5" placeholder="Paste draft text, meeting notes, or an email..."></textarea>
  <div class="wr-actions">
    <button id="runBtn" class="v-btn v-btn-primary">Run Locally</button>
    <button id="copyBtn" class="v-btn" disabled>Copy Output</button>
    <span id="status" class="v-muted"></span>
  </div>
  <div id="output" class="v-card wr-out">Result will stream here...</div>
</div>`,
          css: `.wr-app { max-width: 700px; margin: 0 auto; display: flex; flex-direction: column; gap: 12px; }
.wr-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.wr-head h1 { margin: 0; font-size: 18px; font-weight: 700; }
.wr-actions { display: flex; align-items: center; gap: 8px; font-size: 12px; }
.wr-out { min-height: 110px; white-space: pre-wrap; line-height: 1.6; font-size: 13.5px; }`,
          js: `const srcEl = document.getElementById('source');
const modeEl = document.getElementById('mode');
const outEl = document.getElementById('output');
const statusEl = document.getElementById('status');
const runBtn = document.getElementById('runBtn');
const copyBtn = document.getElementById('copyBtn');
let lastOutput = '';

const INSTRUCTIONS = {
  tighten: 'Rewrite the following text to be clearer and more concise. Return only the improved text.',
  summarize: 'Summarize the following text in 3 concise bullet points.',
  actions: 'Extract concrete action items as a clean Markdown checklist.',
  polish: 'Fix grammar and flow while preserving tone and meaning. Return only the edited text.',
};

async function transform(text, mode) {
  statusEl.textContent = 'Streaming locally...';
  outEl.textContent = '';
  runBtn.disabled = true;
  copyBtn.disabled = true;
  try {
    const prompt = (INSTRUCTIONS[mode] || INSTRUCTIONS.tighten) + '\\n\\n' + text;
    const final = await window.verdant.ask(prompt, {
      onToken: (partial) => { outEl.textContent = partial; },
    });
    lastOutput = final;
    outEl.textContent = final;
    copyBtn.disabled = !final;
    statusEl.textContent = 'Ready';
    window.verdant.emit('writer.completed', { mode, chars: final.length });
    return { mode, output: final };
  } catch (err) {
    const msg = err && err.message ? err.message : String(err);
    outEl.textContent = msg;
    statusEl.textContent = 'Model Stream required';
    return { error: msg };
  } finally {
    runBtn.disabled = false;
  }
}

runBtn.addEventListener('click', () => {
  const txt = srcEl.value.trim();
  if (!txt) { statusEl.textContent = 'Paste some text first.'; return; }
  void transform(txt, modeEl.value);
});

copyBtn.addEventListener('click', async () => {
  if (!lastOutput) return;
  await window.verdant.browser.copyText(lastOutput);
  await window.verdant.browser.notify('Copied', 'Output copied to clipboard.');
});

window.verdant.onInput(async (payload) => {
  const txt = String(payload?.text || payload?.message || '').trim();
  const mode = String(payload?.mode || 'tighten');
  if (!txt) return { error: 'Provide text to transform.' };
  srcEl.value = txt;
  return transform(txt, mode);
});`,
        },
      });

    case 'blank':
    default:
      return buildSnapshot({
        name: 'Untitled Prism',
        summary: 'Client-side app built in Verdant Studio.',
        streams: { ...DEFAULT_STREAM_CONTRACT },
        grants: { ...DEFAULT_PRISM_GRANTS },
        app: {
          html: `<main class="empty-stage">
  <div class="empty-card">
    <h1>Untitled Prism</h1>
    <p class="v-muted">Describe the app you want to build in the chat on the left, pick a starter, or switch to <strong>Prism</strong> in the toolbar above to edit HTML, CSS, and JS directly.</p>
  </div>
</main>`,
          css: `.empty-stage { min-height: calc(100vh - 40px); display: grid; place-items: center; }
.empty-card { max-width: 440px; text-align: center; padding: 28px 24px; border-radius: 14px; border: 1px dashed var(--v-border); background: var(--v-surface); }
.empty-card h1 { margin: 0 0 8px; font-size: 18px; font-weight: 700; }
.empty-card p { margin: 0; font-size: 13px; line-height: 1.6; }`,
          js: `window.verdant.onInput(async (payload) => ({ ok: true, received: payload }));`,
        },
      });
  }
}

// ============================================================================
// 4. Targeted Studio Patch Extraction
// ============================================================================

export interface ExtractedPrismPatch {
  readonly snapshot: VerdantPrismSnapshot;
  readonly updatedSections: readonly ('html' | 'css' | 'js' | 'meta')[];
}

export function extractPrismPatchFromModelOutput(
  rawText: string,
  currentDraft: VerdantPrismSnapshot,
): ExtractedPrismPatch | null {
  const normalizedText = rawText.replace(/\r\n/g, '\n');
  const blockRegex = /```([a-zA-Z0-9_-]*)[ \t]*\n([\s\S]*?)```/g;
  let nextHtml = currentDraft.app.html;
  let nextCss = currentDraft.app.css;
  let nextJs = currentDraft.app.js;
  let nextName = currentDraft.name;
  let nextSummary = currentDraft.summary;
  let nextStreams = currentDraft.streams;

  const updated = new Set<'html' | 'css' | 'js' | 'meta'>();
  const blocks: Array<{ lang: string; body: string }> = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = blockRegex.exec(normalizedText)) !== null) {
    blocks.push({ lang: match[1].trim().toLowerCase(), body: match[2].trim() });
    lastIndex = blockRegex.lastIndex;
  }

  // Salvage an unclosed trailing code fence if generation hit maxTokens mid-block
  const tail = normalizedText.slice(lastIndex);
  const unclosedMatch = tail.match(/```(html|htm|css|js|javascript)[ \t]*\n([\s\S]+)$/i);
  if (unclosedMatch && unclosedMatch[2].trim().length >= 16) {
    blocks.push({
      lang: unclosedMatch[1].trim().toLowerCase(),
      body: unclosedMatch[2].trim(),
    });
  }

  for (const { lang, body } of blocks) {
    if (!body) continue;

    if (lang === 'html' || lang === 'htm') {
      // If model returned a full <!doctype html> doc, cleanly split style/script/body
      const hasFullDoc = /<body[\s>]/i.test(body) || /<style[\s>]/i.test(body) || /<script[\s>]/i.test(body);
      if (hasFullDoc) {
        const styleMatch = body.match(/<style[^>]*>([\s\S]*?)<\/style>/i);
        const scriptMatch = body.match(/<script[^>]*>([\s\S]*?)<\/script>/i);
        const bodyMatch = body.match(/<body[^>]*>([\s\S]*?)<\/body>/i);
        if (styleMatch?.[1]?.trim()) {
          nextCss = styleMatch[1].trim();
          updated.add('css');
        }
        if (scriptMatch?.[1]?.trim()) {
          nextJs = scriptMatch[1].trim();
          updated.add('js');
        }
        const strippedBody = (bodyMatch ? bodyMatch[1] : body)
          .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
          .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
          .trim();
        if (strippedBody) {
          nextHtml = strippedBody;
          updated.add('html');
        }
      } else {
        nextHtml = body;
        updated.add('html');
      }
    } else if (lang === 'css') {
      nextCss = body;
      updated.add('css');
    } else if (lang === 'js' || lang === 'javascript') {
      nextJs = body;
      updated.add('js');
    } else if (lang === 'json') {
      try {
        const meta = JSON.parse(body) as Record<string, unknown>;
        if (typeof meta.name === 'string' && meta.name.trim()) {
          nextName = meta.name.trim();
          updated.add('meta');
        }
        if (typeof meta.summary === 'string') {
          nextSummary = meta.summary.trim();
          updated.add('meta');
        }
        if (meta.streams && typeof meta.streams === 'object') {
          const s = meta.streams as Record<string, unknown>;
          if (s.inputs && typeof s.inputs === 'object') {
            nextStreams = {
              inputs: s.inputs as JsonSchema2020Object,
              outputs: Array.isArray(s.outputs)
                ? s.outputs.filter((x): x is string => typeof x === 'string')
                : nextStreams.outputs,
            };
            updated.add('meta');
          }
        }
      } catch {
        // Ignore non-metadata JSON snippets in prose
      }
    }
  }

  if (updated.size === 0) return null;

  return {
    snapshot: buildSnapshot({
      name: nextName,
      summary: nextSummary,
      streams: nextStreams,
      grants: currentDraft.grants,
      app: { html: nextHtml, css: nextCss, js: nextJs },
    }),
    updatedSections: [...updated],
  };
}

// ============================================================================
// 5. Single Async IndexedDB Store (with In-Memory Fallback for Node Tests)
// ============================================================================

function openStudioDb(): Promise<IDBDatabase | null> {
  if (typeof globalThis.indexedDB === 'undefined') {
    return Promise.resolve(null);
  }
  return new Promise((resolve) => {
    try {
      const req = globalThis.indexedDB.open(IDB_NAME, IDB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE_PROJECTS)) {
          db.createObjectStore(STORE_PROJECTS, { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains(STORE_PRISM_KV)) {
          db.createObjectStore(STORE_PRISM_KV, { keyPath: 'compositeKey' });
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

function idbReq<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error('IndexedDB request failed'));
  });
}

type PrismStoreListener = (projects: readonly VerdantPrismProject[]) => void;

export class PrismStore {
  private readonly projects = new Map<string, VerdantPrismProject>();
  private readonly memoryKv = new Map<string, unknown>();
  private readonly listeners = new Set<PrismStoreListener>();
  private dbPromise: Promise<IDBDatabase | null> | null = null;
  private hydrated = false;

  private getDb(): Promise<IDBDatabase | null> {
    if (!this.dbPromise) {
      this.dbPromise = openStudioDb().then((db) => {
        if (db) {
          db.onclose = () => {
            this.dbPromise = null;
          };
          db.onversionchange = () => {
            db.close();
            this.dbPromise = null;
          };
        } else {
          this.dbPromise = null;
        }
        return db;
      });
    }
    return this.dbPromise;
  }

  async hydrate(): Promise<readonly VerdantPrismProject[]> {
    if (this.hydrated) return this.list();
    this.hydrated = true;
    const db = await this.getDb();
    if (!db) return this.list();
    try {
      const tx = db.transaction(STORE_PROJECTS, 'readonly');
      const all = await idbReq<VerdantPrismProject[]>(tx.objectStore(STORE_PROJECTS).getAll());
      for (const proj of all) {
        if (proj && typeof proj.id === 'string' && proj.draft && !this.projects.has(proj.id)) {
          const normalized = proj.declaration
            ? {
                ...proj,
                declaration: upgradeDeclarationV1ToV2(proj.declaration),
                declarationRevisions: (proj.declarationRevisions ?? []).map((rev) =>
                  upgradeDeclarationV1ToV2(rev),
                ),
              }
            : proj;
          this.projects.set(proj.id, normalized);
        }
      }
      this.notify();
    } catch {
      // Keep memory state on IDB error
    }
    return this.list();
  }

  subscribe(listener: PrismStoreListener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private notify(): void {
    const snap = this.list();
    for (const listener of this.listeners) {
      try {
        listener(snap);
      } catch {
        // Isolate subscriber errors
      }
    }
  }

  private async persistProject(project: VerdantPrismProject): Promise<void> {
    const db = await this.getDb();
    if (!db) return;
    try {
      const tx = db.transaction(STORE_PROJECTS, 'readwrite');
      await idbReq(tx.objectStore(STORE_PROJECTS).put(project));
    } catch {
      // Non-fatal persistence error
    }
  }

  private async removeProjectFromDb(id: string): Promise<void> {
    // Also clean up scoped per-Prism KV keys so deleted Prisms leave zero orphaned storage
    const prefix = `${id}:`;
    for (const k of [...this.memoryKv.keys()]) {
      if (k.startsWith(prefix)) this.memoryKv.delete(k);
    }
    const db = await this.getDb();
    if (!db) return;
    try {
      const tx = db.transaction([STORE_PROJECTS, STORE_PRISM_KV], 'readwrite');
      await idbReq(tx.objectStore(STORE_PROJECTS).delete(id));
      const kvStore = tx.objectStore(STORE_PRISM_KV);
      const allKeys = await idbReq<IDBValidKey[]>(kvStore.getAllKeys());
      for (const key of allKeys) {
        if (typeof key === 'string' && key.startsWith(prefix)) {
          kvStore.delete(key);
        }
      }
    } catch {
      // Non-fatal persistence error
    }
  }

  list(): readonly VerdantPrismProject[] {
    return [...this.projects.values()].sort((a, b) => b.updatedAt - a.updatedAt);
  }

  get(id: string): VerdantPrismProject | null {
    return this.projects.get(id) ?? null;
  }

  listPublishedSummaries(): readonly PublishedPrismSummary[] {
    const usedSlugs = new Set<string>();
    const out: PublishedPrismSummary[] = [];
    for (const proj of this.list()) {
      if (!proj.published) continue;
      let slug = slugifyPrismName(proj.published.name, proj.id);
      if (usedSlugs.has(slug)) {
        slug = `${slug}_${proj.id.slice(-4)}`;
      }
      usedSlugs.add(slug);
      out.push({
        id: proj.id,
        slug,
        name: proj.published.name,
        summary: proj.published.summary,
        digest: proj.publishedHolo?.address ?? proj.published.digest,
        streams: proj.published.streams,
        grants: proj.published.grants,
      });
    }
    return out;
  }

  async create(starterId: PrismStarterId = 'blank', customName?: string): Promise<VerdantPrismProject> {
    const now = Date.now();
    const base = createStarterSnapshot(starterId);
    const draft = customName?.trim() ? buildSnapshot({ ...base, name: customName.trim() }) : base;
    const id = `prism_${now.toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
    const project: VerdantPrismProject = {
      id,
      draft,
      published: null,
      publishedAt: null,
      revisions: [],
      messages: [],
      createdAt: now,
      updatedAt: now,
    };
    this.projects.set(id, project);
    this.notify();
    await this.persistProject(project);
    return project;
  }

  async updateDraft(
    id: string,
    updater: (prev: VerdantPrismSnapshot) => Omit<VerdantPrismSnapshot, 'digest'>,
    opts?: { readonly pushRevision?: boolean },
  ): Promise<VerdantPrismProject | null> {
    const existing = this.projects.get(id);
    if (!existing) return null;
    const nextDraft = buildSnapshot(updater(existing.draft));
    if (nextDraft.digest === existing.draft.digest) return existing;

    const pushRev = opts?.pushRevision !== false;
    const revisions = pushRev
      ? [...existing.revisions, existing.draft].slice(-MAX_REVISIONS)
      : existing.revisions;

    const updated: VerdantPrismProject = {
      ...existing,
      draft: nextDraft,
      revisions,
      updatedAt: Date.now(),
    };
    this.projects.set(id, updated);
    this.notify();
    await this.persistProject(updated);
    return updated;
  }

  async undoDraft(id: string): Promise<VerdantPrismProject | null> {
    const existing = this.projects.get(id);
    if (!existing || existing.revisions.length === 0) return null;
    const previous = existing.revisions[existing.revisions.length - 1];
    const updated: VerdantPrismProject = {
      ...existing,
      draft: previous,
      revisions: existing.revisions.slice(0, -1),
      updatedAt: Date.now(),
    };
    this.projects.set(id, updated);
    this.notify();
    await this.persistProject(updated);
    return updated;
  }

  async appendMessage(
    id: string,
    msg: Omit<StudioChatMessage, 'id' | 'createdAt'>,
  ): Promise<VerdantPrismProject | null> {
    const existing = this.projects.get(id);
    if (!existing) return null;
    const entry: StudioChatMessage = {
      id: `smsg_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 5)}`,
      role: msg.role,
      content: msg.content,
      createdAt: Date.now(),
      ...(msg.updatedSections ? { updatedSections: msg.updatedSections } : {}),
      ...(msg.badges && msg.badges.length > 0 ? { badges: msg.badges } : {}),
      ...(msg.needsLocalModel ? { needsLocalModel: true } : {}),
    };
    const updated: VerdantPrismProject = {
      ...existing,
      messages: [...existing.messages, entry].slice(-80),
      updatedAt: Date.now(),
    };
    this.projects.set(id, updated);
    this.notify();
    await this.persistProject(updated);
    return updated;
  }

  /** Freezes `draft` into an immutable `published` snapshot by its SHA-256 digest. */
  async publish(id: string): Promise<VerdantPrismProject | null> {
    const existing = this.projects.get(id);
    if (!existing) return null;
    const frozen = buildSnapshot(existing.draft);
    const now = Date.now();
    const updated: VerdantPrismProject = {
      ...existing,
      draft: frozen,
      published: frozen,
      publishedAt: now,
      updatedAt: now,
    };
    this.projects.set(id, updated);
    this.notify();
    await this.persistProject(updated);
    return updated;
  }

  async unpublish(id: string): Promise<VerdantPrismProject | null> {
    const existing = this.projects.get(id);
    if (!existing) return null;
    const updated: VerdantPrismProject = {
      ...existing,
      published: null,
      publishedAt: null,
      updatedAt: Date.now(),
    };
    this.projects.set(id, updated);
    this.notify();
    await this.persistProject(updated);
    return updated;
  }

  /**
   * Updates permissions on the live published snapshot (and mirrors them onto draft)
   * WITHOUT publishing unrelated unpublished draft code edits!
   */
  async updatePublishedGrants(
    id: string,
    updater: (prev: PrismGrants) => PrismGrants,
  ): Promise<VerdantPrismProject | null> {
    const existing = this.projects.get(id);
    if (!existing) return null;
    const nextDraft = buildSnapshot({
      ...existing.draft,
      grants: updater(existing.draft.grants),
    });
    const nextPublished = existing.published
      ? buildSnapshot({
          ...existing.published,
          grants: updater(existing.published.grants),
        })
      : null;
    const updated: VerdantPrismProject = {
      ...existing,
      draft: nextDraft,
      published: nextPublished,
      updatedAt: Date.now(),
    };
    this.projects.set(id, updated);
    this.notify();
    await this.persistProject(updated);
    return updated;
  }

  async clearMessages(id: string): Promise<VerdantPrismProject | null> {
    const existing = this.projects.get(id);
    if (!existing || existing.messages.length === 0) return existing ?? null;
    const updated: VerdantPrismProject = {
      ...existing,
      messages: [],
      updatedAt: Date.now(),
    };
    this.projects.set(id, updated);
    this.notify();
    await this.persistProject(updated);
    return updated;
  }

  async delete(id: string): Promise<boolean> {
    const existed = this.projects.delete(id);
    if (!existed) return false;
    this.notify();
    await this.removeProjectFromDb(id);
    return true;
  }

  async importHolo(raw: string | Uint8Array, publishImmediately = false): Promise<VerdantPrismProject> {
    const snapshot = parseHoloArchive(raw);
    const now = Date.now();
    const id = `prism_${now.toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
    const project: VerdantPrismProject = {
      id,
      draft: snapshot,
      published: publishImmediately ? snapshot : null,
      publishedAt: publishImmediately ? now : null,
      revisions: [],
      messages: [],
      createdAt: now,
      updatedAt: now,
    };
    this.projects.set(id, project);
    this.notify();
    await this.persistProject(project);
    return project;
  }

  exportHolo(id: string, usePublished = false): string {
    const proj = this.projects.get(id);
    if (!proj) throw new Error(`Prism "${id}" not found.`);
    const snap = usePublished && proj.published ? proj.published : proj.draft;
    return serializeHoloArchive(snap);
  }

  /** Scoped per-Prism KV storage isolated by `${prismId}:${key}`. */
  async getKv(prismId: string, key: string): Promise<unknown> {
    const compositeKey = `${prismId}:${key}`;
    const db = await this.getDb();
    if (!db) {
      return this.memoryKv.has(compositeKey) ? this.memoryKv.get(compositeKey) : null;
    }
    try {
      const tx = db.transaction(STORE_PRISM_KV, 'readonly');
      const rec = await idbReq<{ compositeKey: string; value: unknown } | undefined>(
        tx.objectStore(STORE_PRISM_KV).get(compositeKey),
      );
      return rec ? rec.value : null;
    } catch {
      return this.memoryKv.has(compositeKey) ? this.memoryKv.get(compositeKey) : null;
    }
  }

  async setKv(prismId: string, key: string, value: unknown): Promise<void> {
    const compositeKey = `${prismId}:${key}`;
    this.memoryKv.set(compositeKey, value);
    const db = await this.getDb();
    if (!db) return;
    try {
      const tx = db.transaction(STORE_PRISM_KV, 'readwrite');
      await idbReq(tx.objectStore(STORE_PRISM_KV).put({ compositeKey, prismId, key, value }));
    } catch {
      // Keep memory fallback
    }
  }

  // ============================================================================
  // Declarative Verdant Studio & Real Binary `prism/holo/6` Operations
  // ============================================================================

  async createDeclarative(
    starterId: DeclarativeStarterId = 'undefined',
    customName?: string,
  ): Promise<VerdantPrismProject> {
    let decl = createDeclarativeStarter(starterId);
    if (customName?.trim()) {
      const renamed = dispatchStudioOp(decl, { op: 'set-metadata', name: customName.trim() });
      if (renamed.ok) decl = renamed.declaration;
    }
    const now = Date.now();
    const id = `prism_${now.toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
    const draft = snapshotFromDeclaration(decl);
    const project: VerdantPrismProject = {
      id,
      kind: 'declarative',
      draft,
      published: null,
      publishedAt: null,
      revisions: [],
      messages: [],
      declaration: decl,
      declarationRevisions: [],
      sealedDraft: null,
      publishedHolo: null,
      createdAt: now,
      updatedAt: now,
    };
    this.projects.set(id, project);
    this.notify();
    await this.persistProject(project);
    return project;
  }

  async applyStudioOp(
    id: string,
    op: StudioApiOp,
    opts?: { readonly pushRevision?: boolean },
  ): Promise<
    | { readonly ok: true; readonly project: VerdantPrismProject; readonly message: string }
    | { readonly ok: false; readonly code: string; readonly message: string; readonly remediation: string }
  > {
    const existing = this.projects.get(id);
    if (!existing || !existing.declaration) {
      return {
        ok: false,
        code: 'project.not-declarative',
        message: 'This project does not hold a declarative Prism model.',
        remediation: 'Create a new declarative Prism in Verdant Studio.',
      };
    }
    const res = dispatchStudioOp(existing.declaration, op);
    if (!res.ok) {
      return {
        ok: false,
        code: res.code,
        message: res.message,
        remediation: res.remediation,
      };
    }
    const nextDraft = snapshotFromDeclaration(res.declaration);
    const pushRev = opts?.pushRevision !== false;
    const declarationRevisions = pushRev
      ? [
          ...(existing.declarationRevisions ?? []),
          stripHeavyRevisionFields(existing.declaration),
        ].slice(-MAX_REVISIONS)
      : (existing.declarationRevisions ?? []);

    // Invalidate sealedDraft whenever declaration mutates (except lock-registers / resolve-error)
    const keepSealed = op.op === 'lock-registers' || op.op === 'resolve-error';
    const updated: VerdantPrismProject = {
      ...existing,
      kind: 'declarative',
      draft: nextDraft,
      declaration: res.declaration,
      declarationRevisions,
      sealedDraft: keepSealed ? existing.sealedDraft : null,
      updatedAt: Date.now(),
    };
    this.projects.set(id, updated);
    this.notify();
    await this.persistProject(updated);
    return { ok: true, project: updated, message: res.message };
  }

  async undoDeclaration(id: string): Promise<VerdantPrismProject | null> {
    const existing = this.projects.get(id);
    const revs = existing?.declarationRevisions ?? [];
    if (!existing || revs.length === 0) return null;
    const prevDecl = revs[revs.length - 1];
    const updated: VerdantPrismProject = {
      ...existing,
      draft: snapshotFromDeclaration(prevDecl),
      declaration: prevDecl,
      declarationRevisions: revs.slice(0, -1),
      sealedDraft: null,
      updatedAt: Date.now(),
    };
    this.projects.set(id, updated);
    this.notify();
    await this.persistProject(updated);
    return updated;
  }

  async saveSealedDraft(id: string, sealed: SealedHoloRecord): Promise<VerdantPrismProject | null> {
    const existing = this.projects.get(id);
    if (!existing) return null;
    const updated: VerdantPrismProject = {
      ...existing,
      sealedDraft: sealed,
      updatedAt: Date.now(),
    };
    this.projects.set(id, updated);
    this.notify();
    await this.persistProject(updated);
    return updated;
  }

  async publishSealedHolo(id: string, sealed?: SealedHoloRecord): Promise<VerdantPrismProject | null> {
    const existing = this.projects.get(id);
    if (!existing) return null;
    const holoToPublish = sealed ?? existing.sealedDraft ?? existing.publishedHolo ?? null;
    let decl = existing.declaration;
    if (decl) {
      const locked = dispatchStudioOp(decl, { op: 'lock-registers' });
      if (locked.ok) decl = locked.declaration;
    }
    const frozenDraft = decl ? snapshotFromDeclaration(decl) : buildSnapshot(existing.draft);
    const now = Date.now();
    const updated: VerdantPrismProject = {
      ...existing,
      draft: frozenDraft,
      published: frozenDraft,
      publishedAt: now,
      declaration: decl,
      sealedDraft: holoToPublish,
      publishedHolo: holoToPublish,
      updatedAt: now,
    };
    this.projects.set(id, updated);
    this.notify();
    await this.persistProject(updated);
    return updated;
  }

  async importBinaryHolo(
    bytes: Uint8Array,
    verified: WorkerVerifyPayload,
    publishImmediately = true,
  ): Promise<VerdantPrismProject> {
    const now = Date.now();
    const id = `prism_${now.toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
    const viewFiles = verified.viewFiles ?? null;
    const sealed: SealedHoloRecord = {
      bytes,
      address: verified.address,
      schema: verified.schema,
      semanticId: verified.semanticId,
      proved: verified.carries.proof && verified.ok,
      theorems: [],
      vectorsPassed: verified.vectorsPassed,
      vectorsTotal: verified.vectorsTotal,
      carries: verified.carries,
      viewFiles,
      viewEntry: verified.viewEntry ?? 'index.html',
      sealedAt: now,
    };
    const snap = buildSnapshot({
      name: verified.name || 'Imported Prism',
      summary: `Verified ${verified.schema} archive (${verified.vectorsPassed}/${verified.vectorsTotal} vectors passed).`,
      streams: {
        inputs: {
          type: 'object',
          properties: {
            request: { type: 'string', description: 'Line of text sent to the proved Prism WASM guest.' },
          },
          required: ['request'],
        },
        outputs: ['prism.ready', 'prism.output'],
      },
      grants: { observer: true, storage: false, browser: false, network: false },
      app: {
        html: viewFiles?.['index.html'] ?? '',
        css: viewFiles?.['prism.css'] ?? viewFiles?.['styles.css'] ?? '',
        js: viewFiles?.['app.js'] ?? '',
      },
    });
    const project: VerdantPrismProject = {
      id,
      kind: 'external-holo',
      draft: snap,
      published: publishImmediately ? snap : null,
      publishedAt: publishImmediately ? now : null,
      revisions: [],
      messages: [],
      sealedDraft: sealed,
      publishedHolo: publishImmediately ? sealed : null,
      createdAt: now,
      updatedAt: now,
    };
    this.projects.set(id, project);
    this.notify();
    await this.persistProject(project);
    return project;
  }
}

function stripHeavyRevisionFields(decl: PrismDeclaration): PrismDeclaration {
  const organ = decl.view?.proceduralOrgan;
  if (!organ || organ.weights.mediaType !== 'picture' || !organ.weights.reconciledImageDataUrl) {
    return decl;
  }
  const { reconciledImageDataUrl: _stripped, ...leanWeights } = organ.weights;
  return {
    ...decl,
    view: {
      ...decl.view,
      proceduralOrgan: {
        ...organ,
        weights: leanWeights,
      },
    },
  };
}

export function snapshotFromDeclaration(decl: PrismDeclaration): VerdantPrismSnapshot {
  const compiled = compilePrismDeclaration(decl);
  const inputs: JsonSchema2020Object = decl.encoding
    ? fieldsToJsonSchema(decl.encoding.fields)
    : DEFAULT_STREAM_CONTRACT.inputs;
  return buildSnapshot({
    name: decl.metadata.name,
    summary: decl.metadata.description || decl.concept.purpose || 'Undefined Prism draft',
    streams: {
      inputs,
      outputs: ['prism.ready', 'prism.output'],
    },
    grants: decl.capabilities,
    app: compiled
      ? {
          html: compiled.appFiles['view/index.html'] ?? '',
          css: compiled.compiledCss,
          js: compiled.appFiles['view/app.js'] ?? '',
        }
      : {
          html: '',
          css: '',
          js: '',
        },
  });
}
