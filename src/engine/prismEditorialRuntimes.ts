/**
 * Editorial, Research, Utility & Space Procedural Runtimes (`src/engine/prismEditorialRuntimes.ts`)
 *
 * Supplies serene, distraction-free editorial surfaces (zero raw inline form/dropdown strips)
 * wired cleanly to `window.verdant.onInput` for Documents, Research notebooks, Utilities, and Spaces.
 */

import type {
  ResearchProgramWeights,
  SpaceWeights,
  ToolOrganWeights,
  WrittenWorkWeights,
} from './prismMediaOrgans';

function safeJsonForScript(obj: unknown): string {
  return JSON.stringify(obj).replace(/<\//g, '<\\/');
}

export function buildWrittenWorkRuntimeJs(weights: WrittenWorkWeights): string {
  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
let chapters = W.chapters.map((c) => ({ ...c }));
let activeIdx = 0;
let viewMode = 'write';
let exportOpen = false;
let saveStatus = 'Saved locally';

function totalWords() {
  return chapters.reduce((acc, c) => acc + (c.body && c.body.trim() ? c.body.trim().split(/\\s+/).length : 0), 0);
}
function chapterWords(c) {
  return c && c.body && c.body.trim() ? c.body.trim().split(/\\s+/).length : 0;
}
function persist() {
  saveStatus = 'Saved locally';
  try { window.verdant?.storage?.set('chapters', chapters); } catch (_) {}
  updateFooter();
}
async function hydrate() {
  try {
    const saved = await window.verdant?.storage?.get('chapters');
    if (Array.isArray(saved) && saved.length > 0) {
      chapters = saved;
      render();
    }
  } catch (_) {}
}
function esc(s) {
  return String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
function renderMarkdownProse(md) {
  const lines = String(md || '').split('\\n');
  const out = [];
  let inList = false;
  let para = [];
  const flushPara = () => {
    if (!para.length) return;
    const inline = esc(para.join(' '))
      .replace(/\`([^\`]+)\`/g, '<code>$1</code>')
      .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
      .replace(/\\*(.+?)\\*/g, '<em>$1</em>');
    out.push('<p>' + inline + '</p>');
    para = [];
  };
  const closeList = () => { if (inList) { out.push('</ul>'); inList = false; } };
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (!line.trim()) { flushPara(); closeList(); continue; }
    if (/^---+$/.test(line.trim())) { flushPara(); closeList(); out.push('<hr>'); continue; }
    const h3 = line.match(/^###\\s+(.+)/); if (h3) { flushPara(); closeList(); out.push('<h3>' + esc(h3[1]) + '</h3>'); continue; }
    const h2 = line.match(/^##\\s+(.+)/); if (h2) { flushPara(); closeList(); out.push('<h2>' + esc(h2[1]) + '</h2>'); continue; }
    const h1 = line.match(/^#\\s+(.+)/); if (h1) { flushPara(); closeList(); out.push('<h1>' + esc(h1[1]) + '</h1>'); continue; }
    const bq = line.match(/^>\\s+(.+)/); if (bq) { flushPara(); closeList(); out.push('<blockquote>' + esc(bq[1]) + '</blockquote>'); continue; }
    const li = line.match(/^[-*]\\s+(.+)/);
    if (li) {
      flushPara();
      if (!inList) { out.push('<ul>'); inList = true; }
      out.push('<li>' + esc(li[1]).replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>').replace(/\\*(.+?)\\*/g, '<em>$1</em>') + '</li>');
      continue;
    }
    closeList();
    para.push(line.trim());
  }
  flushPara(); closeList();
  return out.length ? out.join('') : '<p style="color:var(--v-muted);font-style:italic">Start writing or ask the prompt bar below to draft this section.</p>';
}
function fullMarkdown() {
  return '# ' + W.title + '\\n\\n' + (W.premise ? '_' + W.premise + '_\\n\\n' : '') +
    chapters.map((c) => '## ' + c.title + '\\n\\n' + (c.body || c.summary || '')).join('\\n\\n');
}
function updateFooter() {
  const el = document.getElementById('doc-footer-meta');
  if (!el) return;
  const words = totalWords();
  const readMin = Math.max(1, Math.round(words / 220));
  const pct = Math.min(100, Math.round((words / Math.max(1, W.targetWordCount)) * 100));
  el.innerHTML = '<span>' + words.toLocaleString() + ' words · ' + readMin + ' min read</span><span class="doc-progress"><i style="width:' + pct + '%"></i></span>';
  const st = document.getElementById('doc-save-status');
  if (st) st.textContent = saveStatus;
}
function autoSize(ta) {
  if (!ta) return;
  ta.style.height = 'auto';
  ta.style.height = Math.max(340, ta.scrollHeight) + 'px';
}

function render() {
  if (activeIdx >= chapters.length) activeIdx = Math.max(0, chapters.length - 1);
  const ch = chapters[activeIdx] || chapters[0];
  root.innerHTML = '';

  const bar = document.createElement('div'); bar.className = 'doc-topbar';
  const chRow = document.createElement('div'); chRow.className = 'doc-chapters';
  chapters.forEach((c, idx) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'doc-ch-pill' + (idx === activeIdx ? ' active' : '');
    b.textContent = c.title || ('Section ' + (idx + 1));
    b.onclick = () => { activeIdx = idx; exportOpen = false; render(); };
    chRow.appendChild(b);
  });
  const addCh = document.createElement('button');
  addCh.type = 'button'; addCh.className = 'doc-ch-pill'; addCh.title = 'Add section'; addCh.textContent = '+ Section';
  addCh.onclick = () => {
    chapters.push({ title: 'Section ' + (chapters.length + 1), beat: 'Section', summary: '', body: '' });
    activeIdx = chapters.length - 1; viewMode = 'write'; exportOpen = false; persist(); render();
  };
  chRow.appendChild(addCh);

  const right = document.createElement('div'); right.className = 'organ-row';
  const seg = document.createElement('div'); seg.className = 'organ-seg';
  for (const [m, lbl] of [['write', 'Write'], ['read', 'Read']]) {
    const sb = document.createElement('button'); sb.type = 'button';
    sb.className = viewMode === m ? 'active' : '';
    sb.textContent = lbl;
    sb.onclick = () => { viewMode = m; exportOpen = false; render(); };
    seg.appendChild(sb);
  }
  right.appendChild(seg);

  const menuWrap = document.createElement('div'); menuWrap.className = 'doc-menu';
  const expTrigger = document.createElement('button'); expTrigger.type = 'button'; expTrigger.className = 'organ-btn ghost'; expTrigger.textContent = 'Export ▾';
  expTrigger.onclick = (e) => { e.stopPropagation(); exportOpen = !exportOpen; render(); };
  menuWrap.appendChild(expTrigger);
  if (exportOpen) {
    const pop = document.createElement('div'); pop.className = 'doc-popover';
    const bCopy = document.createElement('button'); bCopy.type = 'button'; bCopy.textContent = 'Copy Markdown';
    bCopy.onclick = () => { (window.verdant?.browser?.copyText || window.verdant?.copyText)?.(fullMarkdown()); exportOpen = false; saveStatus = 'Copied to clipboard'; render(); };
    const bMd = document.createElement('button'); bMd.type = 'button'; bMd.textContent = 'Download .md';
    bMd.onclick = () => { (window.verdant?.browser?.downloadFile || window.verdant?.downloadFile)?.(W.title.toLowerCase().replace(/[^a-z0-9]+/g, '-') + '.md', fullMarkdown(), 'text/markdown'); exportOpen = false; render(); };
    const bHtml = document.createElement('button'); bHtml.type = 'button'; bHtml.textContent = 'Download .html';
    bHtml.onclick = () => {
      const html = '<!doctype html><html><head><meta charset="utf-8"><title>' + esc(W.title) + '</title><style>body{max-width:680px;margin:56px auto;font-family:Georgia,serif;font-size:17px;line-height:1.85;padding:0 20px;color:#0f172a}</style></head><body>' + renderMarkdownProse(fullMarkdown()) + '</body></html>';
      (window.verdant?.browser?.downloadFile || window.verdant?.downloadFile)?.(W.title.toLowerCase().replace(/[^a-z0-9]+/g, '-') + '.html', html, 'text/html');
      exportOpen = false; render();
    };
    pop.append(bCopy, bMd, bHtml);
    if (chapters.length > 1) {
      const bDel = document.createElement('button'); bDel.type = 'button'; bDel.style.color = '#dc2626'; bDel.textContent = 'Remove current section';
      bDel.onclick = () => { chapters.splice(activeIdx, 1); activeIdx = Math.max(0, activeIdx - 1); exportOpen = false; persist(); render(); };
      pop.appendChild(bDel);
    }
    menuWrap.appendChild(pop);
  }
  right.appendChild(menuWrap);
  bar.append(chRow, right);
  root.appendChild(bar);

  if (viewMode === 'write') {
    const titleIn = document.createElement('input');
    titleIn.type = 'text'; titleIn.className = 'doc-title-input';
    titleIn.value = ch.title || ''; titleIn.placeholder = 'Untitled section';
    titleIn.oninput = () => {
      ch.title = titleIn.value;
      const pills = root.querySelectorAll('.doc-ch-pill');
      if (pills[activeIdx]) pills[activeIdx].textContent = ch.title || ('Section ' + (activeIdx + 1));
      persist();
    };
    const subIn = document.createElement('input');
    subIn.type = 'text'; subIn.className = 'doc-sub-input';
    subIn.value = ch.summary || ''; subIn.placeholder = 'Optional subtitle or focus note…';
    subIn.oninput = () => { ch.summary = subIn.value; persist(); };

    const ta = document.createElement('textarea');
    ta.className = 'doc-editor';
    ta.placeholder = 'Write freely… Markdown formatting works when you switch to Read.';
    ta.value = ch.body || '';
    ta.oninput = () => { ch.body = ta.value; autoSize(ta); persist(); };
    root.append(titleIn, subIn, ta);
    requestAnimationFrame(() => autoSize(ta));
  } else {
    const h = document.createElement('h1'); h.className = 'doc-title-input'; h.textContent = ch.title || 'Untitled section';
    root.appendChild(h);
    if (ch.summary) {
      const sub = document.createElement('p'); sub.className = 'doc-sub-input'; sub.textContent = ch.summary;
      root.appendChild(sub);
    }
    const prose = document.createElement('article'); prose.className = 'doc-prose';
    prose.innerHTML = renderMarkdownProse(ch.body || ch.summary || '');
    root.appendChild(prose);
  }

  const foot = document.createElement('footer'); foot.className = 'doc-footer';
  const meta = document.createElement('div'); meta.id = 'doc-footer-meta';
  const status = document.createElement('span'); status.id = 'doc-save-status'; status.textContent = saveStatus;
  foot.append(meta, status);
  root.appendChild(foot);
  updateFooter();
}

if (window.verdant?.onInput) {
  window.verdant.onInput(async (input) => {
    const text = typeof input === 'string' ? input : String((input && (input.prompt || input.text)) || '');
    const lower = text.toLowerCase();
    if (/\\b(read|reading|preview)\\s*(mode|view)?\\b/.test(lower)) { viewMode = 'read'; render(); return { ok: true, mode: 'read' }; }
    if (/\\b(write|edit|writing)\\s*(mode|view)?\\b/.test(lower)) { viewMode = 'write'; render(); return { ok: true, mode: 'write' }; }
    if (/\\b(add|new)\\s+(chapter|section)\\b/.test(lower)) {
      const nameMatch = text.match(/(?:about|titled|called|named)\\s+["']?([^"'.!?]{2,60})/i);
      chapters.push({ title: nameMatch ? nameMatch[1].trim() : 'Section ' + (chapters.length + 1), beat: 'Section', summary: '', body: '' });
      activeIdx = chapters.length - 1; viewMode = 'write'; persist(); render();
      return { ok: true, added: chapters[activeIdx].title };
    }
    if (window.verdant?.ask && text.trim()) {
      const ch = chapters[activeIdx] || chapters[0];
      saveStatus = 'Drafting…'; updateFooter();
      try {
        const reply = await window.verdant.ask('Document "' + W.title + '" (' + W.genre + ', voice: ' + W.voice + '). Current section "' + ch.title + '":\\n' + (ch.body || ch.summary || '').slice(-900) + '\\n\\nUser request: ' + text + '\\nWrite only the prose to append or revise.');
        const clean = String(reply || '').trim();
        if (clean) {
          ch.body = (ch.body ? ch.body + '\\n\\n' : '') + clean;
          persist(); render();
          return { ok: true, words: chapterWords(ch) };
        }
      } catch (_) {}
      saveStatus = 'Saved locally'; updateFooter();
    }
    return { ok: true };
  });
}

document.addEventListener('click', () => { if (exportOpen) { exportOpen = false; render(); } });
render();
hydrate();
`;
}

export function buildResearchProgramRuntimeJs(weights: ResearchProgramWeights): string {
  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
let hypotheses = (W.hypotheses || []).map((h) => ({ ...h }));
let milestones = (W.milestones || []).map((m) => ({ ...m }));
let notes = Array.isArray(W.evidence) ? W.evidence.map((n) => ({ ...n })) : [];

function persist() { try { window.verdant?.storage?.set('research', { hypotheses, milestones, notes }); } catch (_) {} }
async function hydrate() {
  try {
    const saved = await window.verdant?.storage?.get('research');
    if (saved && Array.isArray(saved.hypotheses)) {
      if (saved.hypotheses.length > hypotheses.length) hypotheses = saved.hypotheses;
      if (Array.isArray(saved.milestones) && saved.milestones.length > milestones.length) milestones = saved.milestones;
      if (Array.isArray(saved.notes) && saved.notes.length > notes.length) notes = saved.notes;
      render();
    }
  } catch (_) {}
}
function exportMarkdown() {
  const md = [
    '# ' + W.topic, '', '> ' + W.question, '',
    '## Hypotheses', ...hypotheses.map((h) => '- **' + h.id + '** [' + h.status + ']: ' + h.statement), '',
    '## Evidence & Findings', ...(notes.length ? notes.map((n) => '- [' + n.verdict + '] (' + n.hypothesisId + ') ' + n.text + ' — _' + n.date + '_') : ['- No entries logged yet.']), '',
    '## Milestones', ...milestones.map((m) => '- [' + (m.status === 'done' ? 'x' : ' ') + '] ' + m.title + ' (' + m.dueLabel + ' · ' + m.status + ')')
  ].join('\\n');
  const dl = window.verdant?.browser?.downloadFile || window.verdant?.downloadFile;
  dl?.((W.topic || 'research').toLowerCase().replace(/[^a-z0-9]+/g, '-') + '-report.md', md, 'text/markdown');
}

function sectionHeader(label) {
  const d = document.createElement('div');
  d.style.cssText = 'font-size:11px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:var(--v-muted);margin:20px 0 8px';
  d.textContent = label;
  return d;
}

function render() {
  root.innerHTML = '';
  const doneCount = milestones.filter((m) => m.status === 'done').length;

  const top = document.createElement('div'); top.className = 'doc-topbar';
  const tracks = document.createElement('div'); tracks.className = 'doc-chapters';
  for (const t of (W.tracks || [])) {
    const p = document.createElement('span'); p.className = 'organ-pill'; p.textContent = t; tracks.appendChild(p);
  }
  const prog = document.createElement('span'); prog.className = 'organ-pill';
  prog.textContent = doneCount + ' of ' + milestones.length + ' milestones done';
  tracks.appendChild(prog);
  top.appendChild(tracks);
  root.appendChild(top);

  const q = document.createElement('h2');
  q.style.cssText = 'margin:4px 0 12px;font-size:20px;font-weight:700;line-height:1.4;letter-spacing:-0.015em';
  q.textContent = W.question || W.topic;
  root.appendChild(q);

  root.appendChild(sectionHeader('Hypotheses'));
  hypotheses.forEach((h) => {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:center;justify-content:space-between;gap:12px;padding:10px 12px;border:1px solid var(--v-border);border-radius:10px;margin-bottom:8px;background:var(--v-bg)';
    const txt = document.createElement('span'); txt.style.lineHeight = '1.5'; txt.textContent = h.id + ' · ' + h.statement;
    const st = document.createElement('button'); st.type = 'button';
    st.className = 'organ-btn' + (h.status === 'supported' ? ' primary' : '');
    st.textContent = h.status;
    st.onclick = () => { h.status = h.status === 'open' ? 'supported' : h.status === 'supported' ? 'refuted' : 'open'; persist(); render(); };
    row.append(txt, st); root.appendChild(row);
  });

  root.appendChild(sectionHeader('Evidence & Findings'));
  if (notes.length === 0) {
    const empty = document.createElement('p');
    empty.style.cssText = 'margin:0 0 8px;font-size:13px;color:var(--v-muted);font-style:italic';
    empty.textContent = 'No evidence logged yet. Use the prompt bar below (e.g. "Log evidence: ...") to record observations.';
    root.appendChild(empty);
  } else {
    notes.forEach((n) => {
      const row = document.createElement('div');
      row.style.cssText = 'padding:10px 12px;border-left:3px solid var(--v-accent);border-radius:8px;background:var(--v-surface-alt);margin-bottom:8px;font-size:13.5px;line-height:1.55';
      const meta = document.createElement('div'); meta.style.cssText = 'font-size:11px;font-weight:600;color:var(--v-muted);margin-bottom:3px';
      meta.textContent = n.hypothesisId + ' · ' + n.verdict + (n.date ? ' · ' + n.date : '');
      const body = document.createElement('div'); body.textContent = n.text;
      row.append(meta, body); root.appendChild(row);
    });
  }

  root.appendChild(sectionHeader('Milestones'));
  milestones.forEach((m) => {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:center;justify-content:space-between;gap:12px;padding:10px 12px;border:1px solid var(--v-border);border-radius:10px;margin-bottom:8px;background:var(--v-bg)';
    const txt = document.createElement('span'); txt.textContent = m.title + ' · ' + m.dueLabel;
    const st = document.createElement('button'); st.type = 'button';
    st.className = 'organ-btn' + (m.status === 'done' ? ' primary' : '');
    st.textContent = m.status;
    st.onclick = () => { m.status = m.status === 'planned' ? 'in-progress' : m.status === 'in-progress' ? 'done' : 'planned'; persist(); render(); };
    row.append(txt, st); root.appendChild(row);
  });

  const foot = document.createElement('footer'); foot.className = 'doc-footer'; foot.style.marginTop = '18px';
  const left = document.createElement('span'); left.textContent = hypotheses.length + ' hypotheses · ' + notes.length + ' findings';
  const actions = document.createElement('div'); actions.className = 'organ-row';
  if (window.verdant?.ask) {
    const synthBtn = document.createElement('button'); synthBtn.type = 'button'; synthBtn.className = 'organ-btn ghost'; synthBtn.textContent = 'Synthesize Findings';
    synthBtn.onclick = async () => {
      synthBtn.disabled = true; synthBtn.textContent = 'Synthesizing…';
      try {
        const prompt = 'Synthesize concise research findings for "' + W.topic + '" (' + W.question + '). Hypotheses: ' + hypotheses.map((h) => h.id + ' [' + h.status + ']: ' + h.statement).join('; ') + '. Evidence: ' + notes.map((n) => n.verdict + ': ' + n.text).join('; ');
        const reply = await window.verdant.ask(prompt);
        notes.unshift({ hypothesisId: 'Synthesis', verdict: 'observation', text: String(reply || '').trim(), date: new Date().toLocaleDateString() });
        persist(); render();
      } catch (_) { synthBtn.disabled = false; synthBtn.textContent = 'Synthesize Findings'; }
    };
    actions.appendChild(synthBtn);
  }
  const mdBtn = document.createElement('button'); mdBtn.type = 'button'; mdBtn.className = 'organ-btn primary'; mdBtn.textContent = 'Export Markdown'; mdBtn.onclick = exportMarkdown;
  actions.appendChild(mdBtn);
  foot.append(left, actions); root.appendChild(foot);
}

if (window.verdant?.onInput) {
  window.verdant.onInput((input) => {
    const text = String(input?.prompt || input?.text || input || '').trim();
    const low = text.toLowerCase();
    const hm = text.match(/\\badd\\s+hypothesis\\b(?:\\s+that)?[:\\s-]*(.+)$/i);
    if (hm) { hypotheses.push({ id: 'H' + (hypotheses.length + 1), statement: hm[1].trim(), status: 'open' }); persist(); render(); return { ok: true, hypotheses: hypotheses.length }; }
    const em = text.match(/\\b(?:log|add)\\s+(?:evidence|finding|observation)\\b[:\\s-]*(.+)$/i);
    if (em) {
      const verdict = /\\brefut|disprov|fail/i.test(low) ? 'refuting' : /\\bsupport|confirm|improv|drop|gain/i.test(low) ? 'supporting' : 'observation';
      notes.unshift({ hypothesisId: hypotheses[0]?.id || 'General', verdict, text: em[1].trim(), date: new Date().toLocaleDateString() });
      persist(); render(); return { ok: true, notes: notes.length };
    }
    const mm = text.match(/\\badd\\s+milestone\\b[:\\s-]*(.+)$/i);
    if (mm) { milestones.push({ title: mm[1].trim(), status: 'planned', dueLabel: 'Week ' + (milestones.length + 1) }); persist(); render(); return { ok: true, milestones: milestones.length }; }
    return { ok: true };
  });
}

render();
hydrate();
`;
}

export function buildToolOrganRuntimeJs(weights: ToolOrganWeights): string {
  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
let state = {
  windows: (W.scheduleWindows || []).map((w) => ({ ...w })),
  checkins: 0,
  streakDays: Array(14).fill(false),
  metricValues: (W.metrics || []).map(() => 0),
  steps: Array.isArray(W.steps) && W.steps.length > 0
    ? W.steps.map((s) => ({ ...s }))
    : [
        { id: 's1', text: W.rulesSummary || 'Review objectives', done: false },
        { id: 's2', text: 'Complete primary action', done: false },
        { id: 's3', text: 'Verify & log completion', done: false },
      ],
  calcInputs: (W.metrics || []).map((m) => m.target || 10),
  scenarios: [],
};

function persist() { try { window.verdant?.storage?.set('utilityState', state); } catch (_) {} }
async function hydrate() {
  try {
    const s = await window.verdant?.storage?.get('utilityState');
    if (s && typeof s === 'object') {
      state = {
        ...state,
        checkins: typeof s.checkins === 'number' ? s.checkins : state.checkins,
        streakDays: Array.isArray(s.streakDays) ? s.streakDays : state.streakDays,
        scenarios: Array.isArray(s.scenarios) ? s.scenarios : state.scenarios,
        windows: Array.isArray(s.windows) && s.windows.length > state.windows.length ? s.windows : state.windows,
        steps: Array.isArray(s.steps) && s.steps.length > state.steps.length ? s.steps : state.steps,
      };
      render();
    }
  } catch (_) {}
}
function fmtMin(m) { return String(Math.floor(m / 60)).padStart(2, '0') + ':' + String(m % 60).padStart(2, '0'); }
function emitConstraints() {
  if (!W.emitsOpaqueConstraints) return;
  try {
    window.verdant?.emit?.('constraint', {
      compartmentId: W.compartmentId, opaqueOnly: true,
      windows: state.windows.filter((w) => w.protectedWindow).map((w) => ({ startMinuteOfDay: w.startMinuteOfDay, endMinuteOfDay: w.endMinuteOfDay, daysOfWeek: w.daysOfWeek })),
    });
  } catch (_) {}
}

function sectionHeader(label) {
  const d = document.createElement('div');
  d.style.cssText = 'font-size:11px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:var(--v-muted);margin:18px 0 8px';
  d.textContent = label;
  return d;
}

function render() {
  root.innerHTML = '';
  const summary = document.createElement('p');
  summary.style.cssText = 'margin:0 0 16px;font-size:15px;line-height:1.55;color:var(--v-muted)';
  summary.textContent = W.rulesSummary;
  root.appendChild(summary);

  if (W.role === 'focus-timer') {
    const workMin = Number(W.workMinutes) || 25;
    const shortMin = Number(W.shortBreakMinutes) || 5;
    const longMin = Number(W.longBreakMinutes) || 15;
    const modes = { focus: workMin * 60, short: shortMin * 60, long: longMin * 60 };
    if (!window.__verdantTimer) {
      window.__verdantTimer = { mode: 'focus', remaining: modes.focus, running: false, sessions: state.checkins || 0, interval: null, audioCtx: null };
      const closeTimerAudio = () => {
        const tm = window.__verdantTimer;
        if (tm?.interval) { clearInterval(tm.interval); tm.interval = null; }
        if (tm?.audioCtx && tm.audioCtx.state !== 'closed') { try { void tm.audioCtx.close(); } catch (_) {} }
        if (tm) tm.audioCtx = null;
      };
      window.addEventListener('pagehide', closeTimerAudio);
      window.addEventListener('beforeunload', closeTimerAudio);
    }
    const tm = window.__verdantTimer;
    const chime = () => {
      try {
        const AC = window.AudioContext || window.webkitAudioContext;
        if (!AC) return;
        if (!tm.audioCtx || tm.audioCtx.state === 'closed') tm.audioCtx = new AC();
        if (tm.audioCtx.state === 'suspended') void tm.audioCtx.resume();
        const now = tm.audioCtx.currentTime;
        for (const [off, hz] of [[0, 523.25], [0.16, 659.25], [0.32, 783.99]]) {
          const osc = tm.audioCtx.createOscillator(), g = tm.audioCtx.createGain();
          osc.type = 'sine'; osc.frequency.setValueAtTime(hz, now + off);
          g.gain.setValueAtTime(0.001, now + off); g.gain.exponentialRampToValueAtTime(0.11, now + off + 0.02); g.gain.exponentialRampToValueAtTime(0.0005, now + off + 0.35);
          osc.connect(g); g.connect(tm.audioCtx.destination); osc.start(now + off); osc.stop(now + off + 0.38);
        }
      } catch (_) {}
    };
    const fmtClock = (sec) => String(Math.floor(sec / 60)).padStart(2, '0') + ':' + String(sec % 60).padStart(2, '0');
    const stopTick = () => { if (tm.interval) { clearInterval(tm.interval); tm.interval = null; } tm.running = false; };
    const startTick = () => {
      if (tm.interval) return;
      tm.running = true;
      tm.interval = setInterval(() => {
        if (tm.remaining > 0) {
          tm.remaining -= 1;
          const clk = document.getElementById('timer-clock');
          const bar = document.getElementById('timer-bar-fill');
          if (clk) clk.textContent = fmtClock(tm.remaining);
          if (bar) bar.style.width = Math.round(((modes[tm.mode] - tm.remaining) / Math.max(1, modes[tm.mode])) * 100) + '%';
        } else {
          stopTick(); chime();
          if (tm.mode === 'focus') { tm.sessions += 1; state.checkins = tm.sessions; persist(); }
          tm.mode = tm.mode === 'focus' ? (tm.sessions % 4 === 0 ? 'long' : 'short') : 'focus';
          tm.remaining = modes[tm.mode];
          render();
        }
      }, 1000);
    };

    const modeRow = document.createElement('div'); modeRow.className = 'organ-seg'; modeRow.style.marginBottom = '18px';
    for (const [mKey, mLbl] of [['focus', 'Focus (' + workMin + 'm)'], ['short', 'Short Break (' + shortMin + 'm)'], ['long', 'Long Break (' + longMin + 'm)']]) {
      const b = document.createElement('button'); b.type = 'button';
      b.className = tm.mode === mKey ? 'active' : '';
      b.textContent = mLbl;
      b.onclick = () => { stopTick(); tm.mode = mKey; tm.remaining = modes[mKey]; render(); };
      modeRow.appendChild(b);
    }
    root.appendChild(modeRow);

    const card = document.createElement('div');
    card.style.cssText = 'text-align:center;padding:28px 20px;border:1px solid var(--v-border);border-radius:16px;background:var(--v-bg);margin-bottom:16px';
    const clk = document.createElement('div'); clk.id = 'timer-clock';
    clk.style.cssText = 'font-size:64px;font-weight:700;font-variant-numeric:tabular-nums;letter-spacing:-0.03em;line-height:1;margin-bottom:16px';
    clk.textContent = fmtClock(tm.remaining);
    const barWrap = document.createElement('div');
    barWrap.style.cssText = 'height:8px;border-radius:999px;background:var(--v-surface-alt);overflow:hidden;max-width:340px;margin:0 auto 20px';
    const barFill = document.createElement('div'); barFill.id = 'timer-bar-fill';
    const pct = Math.round(((modes[tm.mode] - tm.remaining) / Math.max(1, modes[tm.mode])) * 100);
    barFill.style.cssText = 'height:100%;background:var(--v-accent);transition:width 0.3s ease;width:' + pct + '%';
    barWrap.appendChild(barFill);

    const controls = document.createElement('div'); controls.className = 'organ-row'; controls.style.justifyContent = 'center'; controls.style.gap = '10px';
    const mainBtn = document.createElement('button'); mainBtn.type = 'button'; mainBtn.className = 'organ-btn primary';
    mainBtn.style.padding = '10px 26px'; mainBtn.textContent = tm.running ? 'Pause' : 'Start';
    mainBtn.onclick = () => { if (tm.running) stopTick(); else startTick(); render(); };
    const resetBtn = document.createElement('button'); resetBtn.type = 'button'; resetBtn.className = 'organ-btn';
    resetBtn.textContent = 'Reset';
    resetBtn.onclick = () => { stopTick(); tm.remaining = modes[tm.mode]; render(); };
    const skipBtn = document.createElement('button'); skipBtn.type = 'button'; skipBtn.className = 'organ-btn ghost';
    skipBtn.textContent = tm.mode === 'focus' ? 'Take Break' : 'Back to Focus';
    skipBtn.onclick = () => { stopTick(); tm.mode = tm.mode === 'focus' ? 'short' : 'focus'; tm.remaining = modes[tm.mode]; render(); };
    controls.append(mainBtn, resetBtn, skipBtn);
    card.append(clk, barWrap, controls);
    root.appendChild(card);

    const foot = document.createElement('footer'); foot.className = 'doc-footer';
    const sessEl = document.createElement('span'); sessEl.textContent = 'Completed focus sessions: ' + tm.sessions;
    const statusEl = document.createElement('span'); statusEl.textContent = tm.running ? 'Running' : 'Ready';
    foot.append(sessEl, statusEl);
    root.appendChild(foot);
    return;
  }

  if (W.role === 'calculator') {
    const grid = document.createElement('div'); grid.style.cssText = 'display:grid;gap:10px;margin-bottom:14px';
    (W.metrics || []).forEach((m, i) => {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;align-items:center;justify-content:space-between;gap:12px;padding:10px 12px;border:1px solid var(--v-border);border-radius:10px;background:var(--v-bg)';
      const lbl = document.createElement('label'); lbl.textContent = m.label + ' (' + m.unit + ')';
      const inp = document.createElement('input'); inp.type = 'number'; inp.className = 'organ-input'; inp.style.width = '110px';
      inp.value = String(state.calcInputs[i] ?? m.target);
      inp.oninput = () => { state.calcInputs[i] = Number(inp.value) || 0; persist(); updateCalc(); };
      row.append(lbl, inp); grid.appendChild(row);
    });
    root.appendChild(grid);
    const outPill = document.createElement('div'); outPill.id = 'calc-out'; outPill.className = 'organ-pill'; outPill.style.marginBottom = '12px'; root.appendChild(outPill);
    function updateCalc() {
      const sum = state.calcInputs.reduce((a, b) => a + Number(b || 0), 0);
      const avg = state.calcInputs.length ? (sum / state.calcInputs.length).toFixed(2) : '0';
      outPill.textContent = 'Total: ' + sum.toLocaleString() + ' · Average: ' + avg;
    }
    updateCalc();
    const foot = document.createElement('footer'); foot.className = 'doc-footer';
    const saveBtn = document.createElement('button'); saveBtn.type = 'button'; saveBtn.className = 'organ-btn primary'; saveBtn.textContent = 'Save Scenario';
    saveBtn.onclick = () => {
      const sum = state.calcInputs.reduce((a, b) => a + Number(b || 0), 0);
      state.scenarios.unshift({ values: [...state.calcInputs], total: sum, at: new Date().toLocaleTimeString() });
      state.scenarios = state.scenarios.slice(0, 8); persist(); render();
    };
    foot.append(document.createElement('span'), saveBtn); root.appendChild(foot);
    if (state.scenarios.length > 0) {
      root.appendChild(sectionHeader('Saved Scenarios'));
      state.scenarios.forEach((sc) => {
        const r = document.createElement('div'); r.className = 'organ-sub'; r.style.padding = '5px 0';
        r.textContent = sc.at + ' · [' + sc.values.join(', ') + '] → Total ' + sc.total.toLocaleString();
        root.appendChild(r);
      });
    }
    return;
  }

  if (W.role === 'workflow-helper') {
    const done = state.steps.filter((s) => s.done).length;
    state.steps.forEach((st) => {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;align-items:center;gap:10px;padding:10px 12px;border:1px solid var(--v-border);border-radius:10px;margin-bottom:8px;background:var(--v-bg)';
      const chk = document.createElement('button'); chk.type = 'button'; chk.className = 'organ-btn' + (st.done ? ' primary' : ''); chk.textContent = st.done ? '✓' : '○';
      chk.onclick = () => { st.done = !st.done; persist(); render(); };
      const lbl = document.createElement('span'); lbl.style.flex = '1'; lbl.style.textDecoration = st.done ? 'line-through' : 'none'; lbl.textContent = st.text;
      row.append(chk, lbl); root.appendChild(row);
    });
    const foot = document.createElement('footer'); foot.className = 'doc-footer'; foot.style.marginTop = '16px';
    const meta = document.createElement('span'); meta.textContent = done + ' of ' + state.steps.length + ' completed';
    const exp = document.createElement('button'); exp.type = 'button'; exp.className = 'organ-btn ghost'; exp.textContent = 'Export Markdown';
    exp.onclick = () => (window.verdant?.browser?.downloadFile || window.verdant?.downloadFile)?.('workflow-checklist.md', state.steps.map((s) => '- [' + (s.done ? 'x' : ' ') + '] ' + s.text).join('\\n'), 'text/markdown');
    foot.append(meta, exp); root.appendChild(foot);
    return;
  }

  root.appendChild(sectionHeader('14-Day Streak'));
  const heat = document.createElement('div'); heat.style.cssText = 'display:grid;grid-template-columns:repeat(14,1fr);gap:5px;margin-bottom:16px';
  state.streakDays.forEach((on, i) => {
    const cell = document.createElement('button'); cell.type = 'button';
    cell.style.cssText = 'height:24px;border-radius:6px;border:1px solid var(--v-border);cursor:pointer;background:' + (on ? 'var(--v-accent)' : 'var(--v-surface-alt)');
    cell.onclick = () => { state.streakDays[i] = !state.streakDays[i]; persist(); render(); };
    heat.appendChild(cell);
  });
  root.appendChild(heat);

  root.appendChild(sectionHeader('Protected Schedule Windows'));
  const bar24 = document.createElement('div');
  bar24.style.cssText = 'position:relative;height:14px;border-radius:999px;background:var(--v-surface-alt);border:1px solid var(--v-border);margin-bottom:12px;overflow:hidden';
  state.windows.filter((w) => w.protectedWindow).forEach((w) => {
    const seg = document.createElement('div');
    seg.style.cssText = 'position:absolute;top:0;bottom:0;left:' + (w.startMinuteOfDay / 1440 * 100) + '%;width:' + (Math.max(30, w.endMinuteOfDay - w.startMinuteOfDay) / 1440 * 100) + '%;background:var(--v-accent)';
    bar24.appendChild(seg);
  });
  root.appendChild(bar24);

  state.windows.forEach((w) => {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:center;justify-content:space-between;gap:12px;padding:10px 12px;border:1px solid var(--v-border);border-radius:10px;margin-bottom:8px;background:var(--v-bg)';
    const txt = document.createElement('span'); txt.textContent = w.label + ' · ' + fmtMin(w.startMinuteOfDay) + ' – ' + fmtMin(w.endMinuteOfDay);
    const toggle = document.createElement('button'); toggle.type = 'button'; toggle.className = 'organ-btn' + (w.protectedWindow ? ' primary' : '');
    toggle.textContent = w.protectedWindow ? 'Protected' : 'Paused';
    toggle.onclick = () => { w.protectedWindow = !w.protectedWindow; emitConstraints(); persist(); render(); };
    row.append(txt, toggle); root.appendChild(row);
  });

  const foot = document.createElement('footer'); foot.className = 'doc-footer'; foot.style.marginTop = '16px';
  const checkBtn = document.createElement('button'); checkBtn.type = 'button'; checkBtn.className = 'organ-btn primary'; checkBtn.textContent = 'Log Today (' + state.checkins + ')';
  checkBtn.onclick = () => { state.checkins += 1; state.streakDays[state.streakDays.length - 1] = true; emitConstraints(); persist(); render(); };
  const expBtn = document.createElement('button'); expBtn.type = 'button'; expBtn.className = 'organ-btn ghost'; expBtn.textContent = 'Export JSON';
  expBtn.onclick = () => (window.verdant?.browser?.downloadFile || window.verdant?.downloadFile)?.('habit-schedule-log.json', JSON.stringify(state, null, 2), 'application/json');
  foot.append(checkBtn, expBtn); root.appendChild(foot);
}

if (window.verdant?.onInput) {
  window.verdant.onInput((input) => {
    const text = String(input?.prompt || input?.text || input || '').trim();
    const low = text.toLowerCase();
    if (/\\blog\\s+today\\b|\\bcheck\\s*in\\b/.test(low)) {
      state.checkins += 1; state.streakDays[state.streakDays.length - 1] = true; emitConstraints(); persist(); render();
      return { ok: true, checkins: state.checkins };
    }
    const sm = text.match(/\\badd\\s+step\\b[:\\s-]*(.+)$/i);
    if (sm) {
      state.steps.push({ id: 's' + Date.now(), text: sm[1].trim(), done: false }); persist(); render();
      return { ok: true, steps: state.steps.length };
    }
    return { ok: true };
  });
}

render();
hydrate();
emitConstraints();
`;
}

export function buildSpaceRuntimeJs(weights: SpaceWeights): string {
  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
let layout = W.layout || 'grid', filterType = 'all';

function labelForType(mt) {
  return ({ picture: 'Picture', audio: 'Audio', video: 'Video', 'written-work': 'Document', 'interactive-game': 'Game', 'research-program': 'Research', 'tool-organ': 'Utility' })[mt] || 'Item';
}

function render() {
  root.innerHTML = '';
  const top = document.createElement('div'); top.className = 'organ-row'; top.style.justifyContent = 'space-between'; top.style.marginBottom = '14px';
  const filters = document.createElement('div'); filters.className = 'organ-row';
  const typesPresent = ['all', ...new Set(W.items.map((it) => it.mediaType))];
  for (const t of typesPresent) {
    const b = document.createElement('button'); b.className = 'organ-btn' + (filterType === t ? ' primary' : '');
    b.textContent = t === 'all' ? 'All (' + W.items.length + ')' : labelForType(t);
    b.onclick = () => { filterType = t; render(); };
    filters.appendChild(b);
  }
  const layouts = document.createElement('div'); layouts.className = 'organ-row';
  for (const [id, lbl] of [['grid','Grid'],['board','Board'],['showcase','Showcase'],['timeline','List']]) {
    const b = document.createElement('button'); b.className = 'organ-btn' + (layout === id ? ' primary' : '');
    b.textContent = lbl; b.onclick = () => { layout = id; render(); };
    layouts.appendChild(b);
  }
  top.append(filters, layouts); root.appendChild(top);

  const visible = filterType === 'all' ? W.items : W.items.filter((it) => it.mediaType === filterType);
  if (visible.length === 0) {
    const empty = document.createElement('div');
    empty.style.padding = '36px 16px'; empty.style.textAlign = 'center'; empty.className = 'organ-sub';
    empty.textContent = 'This Space is ready for content. Ask VerdantUI in chat to create pictures, audio, documents, games, or videos for "' + W.title + '", or add items from any Gallery.';
    root.appendChild(empty);
    return;
  }

  function makeCard(item) {
    const card = document.createElement('div');
    card.style.border = '1px solid var(--v-border)'; card.style.borderRadius = '12px'; card.style.overflow = 'hidden'; card.style.background = 'var(--v-bg)';
    card.style.display = 'flex'; card.style.flexDirection = 'column';
    if (item.previewDataUrl) {
      const img = document.createElement('img'); img.src = item.previewDataUrl; img.alt = item.title;
      img.style.width = '100%'; img.style.height = layout === 'showcase' ? '200px' : '140px'; img.style.objectFit = 'cover';
      card.appendChild(img);
    }
    const body = document.createElement('div'); body.style.padding = '12px'; body.style.display = 'flex'; body.style.flexDirection = 'column'; body.style.gap = '6px'; body.style.flex = '1';
    const meta = document.createElement('div'); meta.className = 'organ-row'; meta.style.justifyContent = 'space-between';
    const badge = document.createElement('span'); badge.className = 'organ-pill'; badge.style.fontSize = '10px'; badge.textContent = labelForType(item.mediaType);
    meta.appendChild(badge);
    const title = document.createElement('div'); title.style.fontWeight = '700'; title.style.fontSize = '14px'; title.textContent = item.title;
    const sum = document.createElement('div'); sum.className = 'organ-sub'; sum.textContent = item.summary || '';
    const openBtn = document.createElement('button'); openBtn.className = 'organ-btn primary'; openBtn.style.marginTop = 'auto'; openBtn.textContent = 'Open';
    openBtn.onclick = () => { try { (window.verdant?.ui?.openPrism || window.verdant?.openPrism)?.(item.id); } catch (_) {} };
    body.append(meta, title, sum, openBtn); card.appendChild(body);
    return card;
  }

  if (layout === 'board') {
    const groups = new Map();
    for (const it of visible) {
      const key = labelForType(it.mediaType);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(it);
    }
    const board = document.createElement('div');
    board.style.display = 'grid'; board.style.gap = '12px'; board.style.gridTemplateColumns = 'repeat(auto-fit, minmax(230px, 1fr))';
    for (const [colName, colItems] of groups.entries()) {
      const col = document.createElement('div');
      col.style.border = '1px solid var(--v-border)'; col.style.borderRadius = '12px'; col.style.padding = '10px'; col.style.background = 'var(--v-surface-alt)';
      col.style.display = 'flex'; col.style.flexDirection = 'column'; col.style.gap = '10px';
      const hdr = document.createElement('div'); hdr.style.fontWeight = '700'; hdr.style.fontSize = '12px'; hdr.textContent = colName + ' (' + colItems.length + ')';
      col.appendChild(hdr);
      for (const it of colItems) col.appendChild(makeCard(it));
      board.appendChild(col);
    }
    root.appendChild(board);
    return;
  }

  const container = document.createElement('div');
  container.style.display = 'grid'; container.style.gap = '12px';
  container.style.gridTemplateColumns = layout === 'timeline' ? '1fr' : layout === 'showcase' ? 'repeat(auto-fill, minmax(280px, 1fr))' : 'repeat(auto-fill, minmax(210px, 1fr))';
  for (const item of visible) container.appendChild(makeCard(item));
  root.appendChild(container);
}

if (window.verdant?.onInput) {
  window.verdant.onInput((input) => {
    const low = String(input?.prompt || input?.text || input || '').trim().toLowerCase();
    if (/\\bboard\\b/.test(low)) layout = 'board';
    else if (/\\bgrid\\b/.test(low)) layout = 'grid';
    else if (/\\bshowcase\\b/.test(low)) layout = 'showcase';
    else if (/\\blist\\b|\\btimeline\\b/.test(low)) layout = 'timeline';
    if (/\\bshow\\s+all\\b/.test(low)) filterType = 'all';
    else if (/\\bpictures?|images?\\b/.test(low)) filterType = 'picture';
    else if (/\\baudio|music|songs?\\b/.test(low)) filterType = 'audio';
    else if (/\\bvideos?\\b/.test(low)) filterType = 'video';
    else if (/\\bdocuments?|stories\\b/.test(low)) filterType = 'written-work';
    render();
    return { ok: true, layout, filterType };
  });
}

render();
`;
}
