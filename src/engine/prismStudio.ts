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

export const HOLO_ARCHIVE_SCHEMA = 'verdant-ui/holo/1' as const;
const MAX_REVISIONS = 20;
const IDB_NAME = 'verdant-ui-studio-v1';
const IDB_VERSION = 1;
const STORE_PROJECTS = 'projects';
const STORE_PRISM_KV = 'prism_kv';

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
}

export interface VerdantPrismProject {
  readonly id: string;
  readonly draft: VerdantPrismSnapshot;
  readonly published: VerdantPrismSnapshot | null;
  readonly publishedAt: number | null;
  readonly revisions: readonly VerdantPrismSnapshot[];
  readonly messages: readonly StudioChatMessage[];
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

export type PrismStarterId = 'blank' | 'dashboard' | 'game' | 'ai-helper';

export interface PrismStarterSpec {
  readonly id: PrismStarterId;
  readonly label: string;
  readonly description: string;
}

export const PRISM_STARTERS: readonly PrismStarterSpec[] = [
  {
    id: 'dashboard',
    label: 'Live Telemetry Dashboard',
    description: 'Interactive monitor wired to Observer sensory streams and theme tokens.',
  },
  {
    id: 'game',
    label: 'Arcade Reflex Canvas',
    description: '60fps interactive Canvas mini-game with scoped high-score storage.',
  },
  {
    id: 'ai-helper',
    label: 'AI Writing Studio',
    description: 'AI-enabled app that streams prompts through VerdantUI via verdant.ask().',
  },
  {
    id: 'blank',
    label: 'Blank Prism App',
    description: 'Minimal reactive HTML/CSS/JS starter ready to shape in chat.',
  },
];

export const DEFAULT_PRISM_GRANTS: PrismGrants = Object.freeze({
  observer: true,
  storage: true,
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
    case 'dashboard':
      return buildSnapshot({
        name: 'Sensory Monitor',
        summary: 'Live interactive telemetry and stream control panel wired to VerdantUI Observer.',
        streams: {
          inputs: {
            type: 'object',
            properties: {
              action: {
                type: 'string',
                enum: ['ping', 'note', 'status'],
                description: 'Monitor command to execute.',
              },
              message: { type: 'string', description: 'Optional status note or payload.' },
            },
          },
          outputs: ['monitor.ping', 'monitor.note'],
        },
        grants: { ...DEFAULT_PRISM_GRANTS },
        app: {
          html: `<div class="v-shell">
  <header class="v-header">
    <div>
      <h1>Sensory Monitor</h1>
      <p class="v-muted">Live Observer stream endpoint</p>
    </div>
    <span id="badge" class="v-pill">Ready</span>
  </header>
  <section class="v-grid">
    <div class="v-card"><div class="v-label">Events Emitted</div><div id="count" class="v-metric">0</div></div>
    <div class="v-card"><div class="v-label">Last Latency</div><div id="latency" class="v-metric">0 ms</div></div>
  </section>
  <form id="noteForm" class="v-row">
    <input id="noteInput" class="v-input" placeholder="Send a sensory note to the Observer..." />
    <button type="submit" class="v-btn v-btn-primary">Emit</button>
  </form>
  <ul id="feed" class="v-list"></ul>
</div>`,
          css: `.v-shell { max-width: 680px; margin: 0 auto; display: flex; flex-direction: column; gap: 16px; }
.v-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.v-header h1 { margin: 0; font-size: 18px; font-weight: 700; }
.v-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.v-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--v-muted); }
.v-metric { font-size: 24px; font-weight: 700; margin-top: 4px; font-variant-numeric: tabular-nums; }
.v-row { display: flex; gap: 8px; }
.v-list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 6px; }
.v-list li { padding: 8px 12px; border: 1px solid var(--v-border); border-radius: 8px; background: var(--v-surface); font-size: 13px; display: flex; justify-content: space-between; gap: 8px; }`,
          js: `const countEl = document.getElementById('count');
const latencyEl = document.getElementById('latency');
const feedEl = document.getElementById('feed');
const badgeEl = document.getElementById('badge');
const inputEl = document.getElementById('noteInput');
let count = 0;

function appendLog(label, detail) {
  const li = document.createElement('li');
  li.innerHTML = '<strong>' + label + '</strong><span class="v-muted">' + detail + '</span>';
  feedEl.prepend(li);
  while (feedEl.children.length > 8) feedEl.lastChild.remove();
}

async function init() {
  const saved = await window.verdant.storage.get('count');
  if (typeof saved === 'number') {
    count = saved;
    countEl.textContent = String(count);
  }
}

document.getElementById('noteForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const msg = inputEl.value.trim() || 'Pulse check';
  const t0 = performance.now();
  count += 1;
  countEl.textContent = String(count);
  window.verdant.emit('monitor.note', { message: msg, count });
  await window.verdant.storage.set('count', count);
  latencyEl.textContent = Math.max(1, Math.round(performance.now() - t0)) + ' ms';
  appendLog('Emitted', msg);
  inputEl.value = '';
});

window.verdant.onInput(async (payload) => {
  const msg = (payload && (payload.message || payload.action)) || 'status';
  badgeEl.textContent = 'Synced';
  count += 1;
  countEl.textContent = String(count);
  appendLog('Observer -> Prism', String(msg));
  return { status: 'ok', count, received: msg, timestamp: new Date().toISOString() };
});

init();`,
        },
      });

    case 'game':
      return buildSnapshot({
        name: 'Pulse Catcher',
        summary: '60fps Canvas reflex mini-game with persistent best score and Observer telemetry.',
        streams: {
          inputs: {
            type: 'object',
            properties: {
              action: {
                type: 'string',
                enum: ['status', 'reset'],
                description: 'Inspect current high score or reset the game.',
              },
            },
          },
          outputs: ['game.score', 'game.best'],
        },
        grants: { ...DEFAULT_PRISM_GRANTS },
        app: {
          html: `<div class="game-wrap">
  <header class="game-bar">
    <div><strong>Pulse Catcher</strong> <span class="v-muted">Click rings before they fade</span></div>
    <div class="scores">Score <b id="score">0</b> · Best <b id="best">0</b></div>
  </header>
  <canvas id="stage" width="600" height="340"></canvas>
  <div class="game-actions">
    <button id="resetBtn" class="v-btn">Reset Round</button>
  </div>
</div>`,
          css: `.game-wrap { max-width: 640px; margin: 0 auto; display: flex; flex-direction: column; gap: 12px; }
.game-bar { display: flex; align-items: center; justify-content: space-between; font-size: 13px; }
.scores { font-variant-numeric: tabular-nums; }
#stage { width: 100%; height: auto; border-radius: 12px; border: 1px solid var(--v-border); background: var(--v-surface); cursor: crosshair; display: block; }
.game-actions { display: flex; justify-content: flex-end; }`,
          js: `const canvas = document.getElementById('stage');
const ctx = canvas.getContext('2d');
const scoreEl = document.getElementById('score');
const bestEl = document.getElementById('best');
let score = 0;
let best = 0;
let target = { x: 300, y: 170, r: 28, ttl: 1 };

function spawnTarget() {
  target = {
    x: 48 + Math.random() * (canvas.width - 96),
    y: 48 + Math.random() * (canvas.height - 96),
    r: 22 + Math.random() * 14,
    ttl: 1,
  };
}

canvas.addEventListener('pointerdown', async (e) => {
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (canvas.width / rect.width);
  const y = (e.clientY - rect.top) * (canvas.height / rect.height);
  if (Math.hypot(x - target.x, y - target.y) <= target.r + 8) {
    score += 1;
    scoreEl.textContent = String(score);
    if (score > best) {
      best = score;
      bestEl.textContent = String(best);
      await window.verdant.storage.set('best', best);
      window.verdant.emit('game.best', { best });
    }
    spawnTarget();
  }
});

document.getElementById('resetBtn').addEventListener('click', () => {
  score = 0;
  scoreEl.textContent = '0';
  spawnTarget();
});

window.verdant.onInput(async (payload) => {
  if (payload && payload.action === 'reset') {
    score = 0;
    scoreEl.textContent = '0';
    spawnTarget();
  }
  return { game: 'Pulse Catcher', score, best };
});

function tick() {
  target.ttl -= 0.008;
  if (target.ttl <= 0) spawnTarget();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  ctx.arc(target.x, target.y, target.r * (0.65 + 0.35 * target.ttl), 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(15, 81, 50, ' + (0.25 + 0.65 * target.ttl) + ')';
  ctx.fill();
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = '#10b981';
  ctx.stroke();
  requestAnimationFrame(tick);
}

window.verdant.storage.get('best').then((v) => {
  if (typeof v === 'number') {
    best = v;
    bestEl.textContent = String(best);
  }
});
tick();`,
        },
      });

    case 'ai-helper':
      return buildSnapshot({
        name: 'Draft Refiner',
        summary: 'AI-enabled text refiner that streams local cognition via verdant.ask().',
        streams: {
          inputs: {
            type: 'object',
            properties: {
              text: { type: 'string', description: 'Text to refine or summarize.' },
              tone: {
                type: 'string',
                enum: ['concise', 'technical', 'friendly'],
                description: 'Target tone.',
              },
            },
            required: ['text'],
          },
          outputs: ['refiner.completed'],
        },
        grants: { ...DEFAULT_PRISM_GRANTS },
        app: {
          html: `<div class="ai-app">
  <header class="ai-head">
    <h1>Draft Refiner</h1>
    <select id="tone" class="v-input" style="width:auto">
      <option value="concise">Concise</option>
      <option value="technical">Technical</option>
      <option value="friendly">Friendly</option>
    </select>
  </header>
  <textarea id="source" class="v-input" rows="5" placeholder="Paste rough notes or a paragraph to refine locally..."></textarea>
  <div class="ai-actions">
    <button id="runBtn" class="v-btn v-btn-primary">Refine with VerdantUI</button>
    <span id="status" class="v-muted"></span>
  </div>
  <div id="output" class="v-card ai-out">Output will stream here...</div>
</div>`,
          css: `.ai-app { max-width: 680px; margin: 0 auto; display: flex; flex-direction: column; gap: 12px; }
.ai-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.ai-head h1 { margin: 0; font-size: 18px; font-weight: 700; }
.ai-actions { display: flex; align-items: center; gap: 12px; font-size: 12px; }
.ai-out { min-height: 96px; white-space: pre-wrap; line-height: 1.55; font-size: 13.5px; }`,
          js: `const srcEl = document.getElementById('source');
const toneEl = document.getElementById('tone');
const outEl = document.getElementById('output');
const statusEl = document.getElementById('status');
const runBtn = document.getElementById('runBtn');

async function refine(text, tone) {
  statusEl.textContent = 'Streaming from Observer...';
  outEl.textContent = '';
  runBtn.disabled = true;
  try {
    const prompt = 'Rewrite the following text in a ' + tone + ' tone. Return only the improved text:\\n\\n' + text;
    const final = await window.verdant.ask(prompt, {
      onToken: (partial) => { outEl.textContent = partial; },
    });
    outEl.textContent = final;
    statusEl.textContent = 'Done';
    window.verdant.emit('refiner.completed', { tone, chars: final.length });
    return { refined: final, tone };
  } catch (err) {
    const msg = err && err.message ? err.message : String(err);
    outEl.textContent = msg;
    statusEl.textContent = 'Notice';
    return { error: msg };
  } finally {
    runBtn.disabled = false;
  }
}

runBtn.addEventListener('click', () => {
  const txt = srcEl.value.trim();
  if (!txt) { statusEl.textContent = 'Enter text first.'; return; }
  void refine(txt, toneEl.value);
});

window.verdant.onInput(async (payload) => {
  const txt = payload && typeof payload.text === 'string' ? payload.text : String(payload?.message || '');
  const tone = payload && typeof payload.tone === 'string' ? payload.tone : 'concise';
  srcEl.value = txt;
  toneEl.value = tone;
  if (!txt.trim()) return { error: 'Provide text to refine.' };
  return refine(txt, tone);
});`,
        },
      });

    case 'blank':
    default:
      return buildSnapshot({
        name: 'Untitled Prism',
        summary: 'Interactive client-side Prism app connected to VerdantUI.',
        streams: { ...DEFAULT_STREAM_CONTRACT },
        grants: { ...DEFAULT_PRISM_GRANTS },
        app: {
          html: `<div class="v-card" style="max-width:560px;margin:24px auto;">
  <h1 style="margin:0 0 6px;font-size:18px;">New Prism</h1>
  <p class="v-muted" style="margin:0 0 14px;">Describe what you want to build in Verdant Studio, or edit HTML/CSS/JS directly.</p>
  <button id="helloBtn" class="v-btn v-btn-primary">Send Sensory Pulse</button>
  <p id="status" class="v-muted" style="margin:12px 0 0;font-size:12px;"></p>
</div>`,
          css: ``,
          js: `const statusEl = document.getElementById('status');
document.getElementById('helloBtn').addEventListener('click', () => {
  const at = new Date().toLocaleTimeString();
  window.verdant.emit('prism.output', { message: 'Pulse at ' + at });
  statusEl.textContent = 'Sent sensory pulse to Observer at ' + at;
});
window.verdant.onInput(async (payload) => {
  statusEl.textContent = 'Received input: ' + JSON.stringify(payload);
  return { ok: true, echo: payload };
});`,
        },
      });
  }
}

// ============================================================================
// 4. Targeted Studio Prompting & Patch Extraction (Fast Local Iteration)
// ============================================================================

export function buildStudioSystemPrompt(draft: VerdantPrismSnapshot): string {
  return [
    'You are Verdant Studio, building a live client-side Prism web app inside VerdantUI.',
    'The app runs in a sandboxed iframe with built-in Verdant theme CSS variables (--v-bg, --v-surface, --v-border, --v-text, --v-muted, --v-accent) and utility classes (.v-card, .v-btn, .v-btn-primary, .v-input, .v-pill, .v-muted).',
    'The sandbox exposes window.verdant: emit(name, body), onInput(async (payload) => result), ask(prompt, { onToken }), storage.get(k)/set(k, v), and fetch(url, init).',
    'Respond concisely and emit ONLY the sections that need to change using fenced code blocks:',
    '- ```html ... ``` for body markup',
    '- ```css ... ``` for styles',
    '- ```js ... ``` for script logic',
    '- Optional ```json ... ``` with {"name","summary"} when renaming or updating metadata.',
    'Keep code clean, self-contained, accessible, and bug-free (no external CDN scripts).',
    '',
    `Current Prism: "${draft.name}" — ${draft.summary}`,
    'Current HTML:',
    '```html',
    draft.app.html,
    '```',
    'Current CSS:',
    '```css',
    draft.app.css,
    '```',
    'Current JS:',
    '```js',
    draft.app.js,
    '```',
  ].join('\n');
}

export interface ExtractedPrismPatch {
  readonly snapshot: VerdantPrismSnapshot;
  readonly updatedSections: readonly ('html' | 'css' | 'js' | 'meta')[];
}

export function extractPrismPatchFromModelOutput(
  rawText: string,
  currentDraft: VerdantPrismSnapshot,
): ExtractedPrismPatch | null {
  const blockRegex = /```([a-zA-Z0-9_-]*)\s*\n([\s\S]*?)```/g;
  let nextHtml = currentDraft.app.html;
  let nextCss = currentDraft.app.css;
  let nextJs = currentDraft.app.js;
  let nextName = currentDraft.name;
  let nextSummary = currentDraft.summary;
  let nextStreams = currentDraft.streams;

  const updated = new Set<'html' | 'css' | 'js' | 'meta'>();
  let match: RegExpExecArray | null;

  while ((match = blockRegex.exec(rawText)) !== null) {
    const lang = match[1].trim().toLowerCase();
    const body = match[2].trim();
    if (!body) continue;

    if (lang === 'html') {
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
      this.dbPromise = openStudioDb();
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
        if (proj && typeof proj.id === 'string' && proj.draft) {
          this.projects.set(proj.id, proj);
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
    const db = await this.getDb();
    if (!db) return;
    try {
      const tx = db.transaction(STORE_PROJECTS, 'readwrite');
      await idbReq(tx.objectStore(STORE_PROJECTS).delete(id));
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
        digest: proj.published.digest,
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
}
