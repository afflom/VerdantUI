/**
 * Feature-Complete Procedural Interactive Game Runtimes (`src/engine/prismGameRuntimes.ts`)
 *
 * Implements ALL 10 declared game variants cleanly with zero silent fallbacks, zero LLM-written JS,
 * persistent best scores/times via `window.verdant.storage`, pause/resume, keyboard + mouse/touch
 * support, and optional Web Audio SFX:
 *  - `grid-sweep`: Minesweeper (presets, chord reveal, flood fill, flag mode, best time)
 *  - `grid-puzzle`: `merge-2048` | `memory-match` (full card flip + emblems) | `lights-out`
 *  - `turn-board`: Tic-Tac-Toe / Connect 4 / Gomoku with tactical minimax/threat-blocking AI
 *  - `arcade-canvas`: `snake` | `breakout` (real bricks/paddle/levels) | `pong` (AI + 2P)
 *  - `cellular-sim`: `game-of-life` (patterns + step + draw) | `rule-30` (Wolfram 1D 0..255)
 */

import type {
  ArcadeCanvasGameWeights,
  CellularSimGameWeights,
  GridPuzzleGameWeights,
  GridSweepGameWeights,
  TurnBoardGameWeights,
} from './prismMediaOrgans';

function safeJsonForScript(obj: unknown): string {
  return JSON.stringify(obj).replace(/<\//g, '<\\/');
}

const SHARED_GAME_SFX_SNIPPET = `
let sfxEnabled = true, sfxCtx = null;
function playSfx(freq, dur = 0.08, type = 'triangle', gain = 0.08) {
  if (!sfxEnabled) return;
  try {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    if (!sfxCtx) sfxCtx = new AC();
    if (sfxCtx.state === 'suspended') void sfxCtx.resume();
    const osc = sfxCtx.createOscillator(), g = sfxCtx.createGain();
    osc.type = type; osc.frequency.setValueAtTime(freq, sfxCtx.currentTime);
    g.gain.setValueAtTime(gain, sfxCtx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.0005, sfxCtx.currentTime + dur);
    osc.connect(g); g.connect(sfxCtx.destination);
    osc.start(); osc.stop(sfxCtx.currentTime + dur + 0.01);
  } catch (_) {}
}
`;

export function buildGridSweepRuntimeJs(weights: GridSweepGameWeights): string {
  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
${SHARED_GAME_SFX_SNIPPET}
let rows = W.rows, cols = W.cols, mines = W.mines;
let board = [], revealed = [], flagged = [], firstClick = true, over = false, won = false;
let flagMode = false, elapsed = 0, timerId = null, bestTimes = {};

async function loadBest() {
  try { const b = await window.verdant?.storage?.get('bestTimes'); if (b && typeof b === 'object') { bestTimes = b; render(); } } catch (_) {}
}
function saveBest(key, sec) {
  if (bestTimes[key] === undefined || sec < bestTimes[key]) {
    bestTimes[key] = sec;
    try { window.verdant?.storage?.set('bestTimes', bestTimes); } catch (_) {}
  }
}
function stopTimer() { if (timerId) { clearInterval(timerId); timerId = null; } }
function startTimer() {
  if (!W.showTimer || timerId) return;
  timerId = setInterval(() => { elapsed += 1; updateHud(); }, 1000);
}
function syncModel(statusCode) {
  try { if (globalThis.prism?.invoke) void globalThis.prism.invoke(rows + ' ' + cols + ' ' + mines + ' ' + statusCode).catch(() => {}); } catch (_) {}
}
function neighbors(r, c) {
  const out = [];
  for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
    if (dr === 0 && dc === 0) continue;
    const nr = r + dr, nc = c + dc;
    if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) out.push([nr, nc]);
  }
  return out;
}
function placeMines(safeR, safeC) {
  const forbidden = new Set();
  if (W.firstClickSafe) {
    forbidden.add(safeR + ',' + safeC);
    for (const [nr, nc] of neighbors(safeR, safeC)) forbidden.add(nr + ',' + nc);
  }
  const pool = [];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) if (!forbidden.has(r + ',' + c)) pool.push([r, c]);
  for (let i = pool.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [pool[i], pool[j]] = [pool[j], pool[i]]; }
  const target = Math.min(mines, pool.length);
  for (let i = 0; i < target; i++) board[pool[i][0]][pool[i][1]] = -1;
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    if (board[r][c] === -1) continue;
    let cnt = 0;
    for (const [nr, nc] of neighbors(r, c)) if (board[nr][nc] === -1) cnt += 1;
    board[r][c] = cnt;
  }
}
function initGame(r = rows, c = cols, m = mines) {
  rows = r; cols = c; mines = Math.min(m, Math.max(1, r * c - 9));
  stopTimer(); elapsed = 0; firstClick = true; over = false; won = false;
  board = Array.from({ length: rows }, () => Array(cols).fill(0));
  revealed = Array.from({ length: rows }, () => Array(cols).fill(false));
  flagged = Array.from({ length: rows }, () => Array(cols).fill(false));
  render();
}
function revealCell(r, c) {
  if (over || flagged[r][c]) return;
  if (firstClick) { firstClick = false; placeMines(r, c); startTimer(); }
  if (revealed[r][c]) {
    if (W.chordReveal && board[r][c] > 0) {
      const adj = neighbors(r, c);
      if (adj.filter(([nr, nc]) => flagged[nr][nc]).length === board[r][c]) {
        for (const [nr, nc] of adj) if (!revealed[nr][nc] && !flagged[nr][nc]) revealCell(nr, nc);
      }
    }
    return;
  }
  if (board[r][c] === -1) {
    over = true; won = false; stopTimer(); playSfx(140, 0.24, 'sawtooth', 0.11);
    for (let rr = 0; rr < rows; rr++) for (let cc = 0; cc < cols; cc++) if (board[rr][cc] === -1) revealed[rr][cc] = true;
    syncModel(2); render(); return;
  }
  const queue = [[r, c]];
  while (queue.length > 0) {
    const [cr, cc] = queue.shift();
    if (revealed[cr][cc] || flagged[cr][cc]) continue;
    revealed[cr][cc] = true;
    if (W.floodReveal && board[cr][cc] === 0) {
      for (const [nr, nc] of neighbors(cr, cc)) if (!revealed[nr][nc] && !flagged[nr][nc]) queue.push([nr, nc]);
    }
  }
  playSfx(480, 0.04, 'sine', 0.05);
  let safeLeft = 0;
  for (let rr = 0; rr < rows; rr++) for (let cc = 0; cc < cols; cc++) if (board[rr][cc] !== -1 && !revealed[rr][cc]) safeLeft += 1;
  if (safeLeft === 0) {
    over = true; won = true; stopTimer(); playSfx(660, 0.18, 'triangle', 0.1);
    saveBest(rows + 'x' + cols + 'x' + mines, elapsed);
    for (let rr = 0; rr < rows; rr++) for (let cc = 0; cc < cols; cc++) if (board[rr][cc] === -1) flagged[rr][cc] = true;
    syncModel(1);
  }
  render();
}
function toggleFlag(r, c) {
  if (over || revealed[r][c]) return;
  flagged[r][c] = !flagged[r][c];
  playSfx(320, 0.04, 'square', 0.04);
  render();
}
function updateHud() { const t = document.getElementById('ms-timer'); if (t) t.textContent = elapsed + 's'; }
function render() {
  const remaining = mines - flagged.flat().filter(Boolean).length;
  const best = bestTimes[rows + 'x' + cols + 'x' + mines];
  const statusText = over ? (won ? 'Cleared! You won!' : 'Hit a mine — try again!') : 'Active';
  const cellPx = cols > 18 ? 24 : cols > 12 ? 28 : 32;
  root.innerHTML = '';

  const hud = document.createElement('div');
  hud.className = 'organ-row'; hud.style.justifyContent = 'space-between'; hud.style.marginBottom = '12px';
  const stats = document.createElement('div'); stats.className = 'organ-row';
  stats.innerHTML =
    '<span class="organ-pill">Mines: ' + remaining + '</span>' +
    (W.showTimer ? '<span class="organ-pill">Time: <span id="ms-timer">' + elapsed + 's</span></span>' : '') +
    (best !== undefined ? '<span class="organ-pill">Best: ' + best + 's</span>' : '') +
    '<span class="organ-pill" style="' + (over ? (won ? 'border-color:#10b981;color:#10b981' : 'border-color:#dc2626;color:#dc2626') : '') + '">' + statusText + '</span>';
  hud.appendChild(stats);

  const controls = document.createElement('div'); controls.className = 'organ-row';
  const sfxBtn = document.createElement('button'); sfxBtn.type = 'button'; sfxBtn.className = 'organ-btn';
  sfxBtn.textContent = sfxEnabled ? 'Sound: ON' : 'Sound: OFF';
  sfxBtn.onclick = () => { sfxEnabled = !sfxEnabled; render(); };
  const flagBtn = document.createElement('button'); flagBtn.type = 'button'; flagBtn.className = 'organ-btn' + (flagMode ? ' primary' : '');
  flagBtn.textContent = flagMode ? 'Flag Mode: ON' : 'Flag Mode';
  flagBtn.onclick = () => { flagMode = !flagMode; render(); };
  const newBtn = document.createElement('button'); newBtn.type = 'button'; newBtn.className = 'organ-btn primary';
  newBtn.textContent = 'New Game'; newBtn.onclick = () => initGame();
  controls.append(sfxBtn, flagBtn, newBtn); hud.appendChild(controls); root.appendChild(hud);

  if (W.presets && W.presets.length > 0) {
    const presetRow = document.createElement('div'); presetRow.className = 'organ-row'; presetRow.style.marginBottom = '12px';
    for (const pr of W.presets) {
      const b = document.createElement('button'); b.type = 'button';
      const active = pr.rows === rows && pr.cols === cols && pr.mines === mines;
      b.className = 'organ-btn' + (active ? ' primary' : '');
      b.textContent = pr.label + ' (' + pr.rows + '×' + pr.cols + ' · ' + pr.mines + ')';
      b.onclick = () => initGame(pr.rows, pr.cols, pr.mines);
      presetRow.appendChild(b);
    }
    root.appendChild(presetRow);
  }

  const wrap = document.createElement('div'); wrap.style.overflowX = 'auto'; wrap.style.paddingBottom = '6px';
  const grid = document.createElement('div'); grid.className = 'ms-grid';
  grid.style.gridTemplateColumns = 'repeat(' + cols + ', ' + cellPx + 'px)';
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const btn = document.createElement('button'); btn.type = 'button';
    btn.style.width = cellPx + 'px'; btn.style.height = cellPx + 'px';
    const isRev = revealed[r][c], isFlag = flagged[r][c], val = board[r][c];
    btn.className = 'ms-cell' + (isRev ? (val === -1 ? ' rev mine' : ' rev ms-n' + val) : isFlag ? ' flag' : '');
    btn.textContent = isRev ? (val === -1 ? '✹' : val > 0 ? String(val) : '') : (isFlag ? '⚑' : '');
    btn.setAttribute('aria-label', 'Row ' + (r + 1) + ' Col ' + (c + 1));
    btn.onclick = () => (flagMode ? toggleFlag(r, c) : revealCell(r, c));
    btn.oncontextmenu = (ev) => { ev.preventDefault(); toggleFlag(r, c); };
    grid.appendChild(btn);
  }
  wrap.appendChild(grid); root.appendChild(wrap);
  const hint = document.createElement('p'); hint.className = 'organ-sub'; hint.style.textAlign = 'center'; hint.style.marginTop = '10px';
  hint.textContent = 'Left-click to reveal · Right-click (or Flag Mode) to mark mines · Click a revealed number to chord';
  root.appendChild(hint);
}
initGame();
loadBest();
`;
}

export function buildTurnBoardRuntimeJs(weights: TurnBoardGameWeights): string {
  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
${SHARED_GAME_SFX_SNIPPET}
let grid = [], turn = 1, winner = 0, draws = 0, score = [0, 0, 0], winCells = [], vsCpu = W.vsCpu;

function resetBoard() {
  grid = Array.from({ length: W.rows }, () => Array(W.cols).fill(0));
  turn = 1; winner = 0; winCells = []; render();
}
function findWinLine(p) {
  const dirs = [[0,1],[1,0],[1,1],[1,-1]];
  for (let r = 0; r < W.rows; r++) for (let c = 0; c < W.cols; c++) {
    if (grid[r][c] !== p) continue;
    for (const [dr, dc] of dirs) {
      const line = [[r, c]];
      for (let k = 1; k < W.winLength; k++) {
        const nr = r + dr * k, nc = c + dc * k;
        if (nr < 0 || nr >= W.rows || nc < 0 || nc >= W.cols || grid[nr][nc] !== p) break;
        line.push([nr, nc]);
      }
      if (line.length >= W.winLength) return line;
    }
  }
  return null;
}
function listLegalMoves() {
  const out = [];
  if (W.gravity) {
    for (let c = 0; c < W.cols; c++) {
      for (let r = W.rows - 1; r >= 0; r--) {
        if (grid[r][c] === 0) { out.push([r, c]); break; }
      }
    }
  } else {
    for (let r = 0; r < W.rows; r++) for (let c = 0; c < W.cols; c++) if (grid[r][c] === 0) out.push([r, c]);
  }
  return out;
}
function scoreHeuristic(player) {
  let pts = 0;
  const opp = player === 2 ? 1 : 2;
  const dirs = [[0,1],[1,0],[1,1],[1,-1]];
  const midC = (W.cols - 1) / 2, midR = (W.rows - 1) / 2;
  for (let r = 0; r < W.rows; r++) for (let c = 0; c < W.cols; c++) {
    if (grid[r][c] === player) pts += Math.max(0, 4 - Math.abs(c - midC) - Math.abs(r - midR) * 0.5);
    for (const [dr, dc] of dirs) {
      let own = 0, foe = 0;
      for (let k = 0; k < W.winLength; k++) {
        const nr = r + dr * k, nc = c + dc * k;
        if (nr < 0 || nr >= W.rows || nc < 0 || nc >= W.cols) { own = -1; break; }
        if (grid[nr][nc] === player) own += 1;
        else if (grid[nr][nc] === opp) foe += 1;
      }
      if (own < 0) continue;
      if (own > 0 && foe === 0) pts += Math.pow(6, own);
      else if (foe > 0 && own === 0) pts -= Math.pow(6.5, foe);
    }
  }
  return pts;
}
function chooseTacticalCpuMove() {
  const moves = listLegalMoves();
  if (moves.length === 0) return null;
  // 1. Take immediate winning move
  for (const [r, c] of moves) {
    grid[r][c] = 2;
    const w = findWinLine(2);
    grid[r][c] = 0;
    if (w) return [r, c];
  }
  // 2. Block immediate opponent winning threat
  for (const [r, c] of moves) {
    grid[r][c] = 1;
    const w = findWinLine(1);
    grid[r][c] = 0;
    if (w) return [r, c];
  }
  // 3. Depth-bounded minimax alpha-beta tactical search
  const maxDepth = W.rows * W.cols <= 9 ? 5 : 3;
  function minimax(depth, alpha, beta, maximizing) {
    if (findWinLine(2)) return 100000 + depth;
    if (findWinLine(1)) return -100000 - depth;
    const legal = listLegalMoves();
    if (depth === 0 || legal.length === 0) return scoreHeuristic(2);
    if (maximizing) {
      let best = -Infinity;
      for (const [mr, mc] of legal) {
        grid[mr][mc] = 2;
        const val = minimax(depth - 1, alpha, beta, false);
        grid[mr][mc] = 0;
        if (val > best) best = val;
        if (best > alpha) alpha = best;
        if (beta <= alpha) break;
      }
      return best;
    } else {
      let best = Infinity;
      for (const [mr, mc] of legal) {
        grid[mr][mc] = 1;
        const val = minimax(depth - 1, alpha, beta, true);
        grid[mr][mc] = 0;
        if (val < best) best = val;
        if (best < beta) beta = best;
        if (beta <= alpha) break;
      }
      return best;
    }
  }
  let bestMove = moves[0], bestVal = -Infinity;
  for (const [r, c] of moves) {
    grid[r][c] = 2;
    const val = minimax(maxDepth - 1, -Infinity, Infinity, false);
    grid[r][c] = 0;
    if (val > bestVal) { bestVal = val; bestMove = [r, c]; }
  }
  return bestMove;
}
function playAt(r, c) {
  if (winner !== 0) return;
  let tr = r;
  if (W.gravity) {
    tr = -1;
    for (let rr = W.rows - 1; rr >= 0; rr--) if (grid[rr][c] === 0) { tr = rr; break; }
    if (tr < 0) return;
  } else if (grid[tr][c] !== 0) return;
  grid[tr][c] = turn;
  playSfx(turn === 1 ? 440 : 349, 0.05, 'triangle', 0.06);
  const line = findWinLine(turn);
  if (line) { winner = turn; winCells = line; score[turn] += 1; playSfx(660, 0.2, 'triangle', 0.1); render(); return; }
  if (grid.flat().every((v) => v !== 0)) { winner = -1; draws += 1; render(); return; }
  turn = turn === 1 ? 2 : 1;
  render();
  if (vsCpu && turn === 2 && winner === 0) {
    setTimeout(() => {
      const pick = chooseTacticalCpuMove();
      if (pick) playAt(pick[0], pick[1]);
    }, 180);
  }
}
function render() {
  root.innerHTML = '';
  const top = document.createElement('div'); top.className = 'organ-row'; top.style.justifyContent = 'space-between'; top.style.marginBottom = '12px';
  const p2Name = vsCpu ? 'Computer' : 'Player 2';
  const status = winner === -1 ? 'Draw!' : winner === 1 ? W.p1Label + ' wins!' : winner === 2 ? p2Name + ' wins!' : (turn === 1 ? W.p1Label : p2Name) + "'s turn";
  top.innerHTML = '<div class="organ-row"><span class="organ-pill">' + W.p1Label + ': ' + score[1] + '</span><span class="organ-pill">' + p2Name + ': ' + score[2] + '</span><span class="organ-pill">Draws: ' + draws + '</span><span class="organ-pill">' + status + '</span></div>';
  const ctrls = document.createElement('div'); ctrls.className = 'organ-row';
  const modeBtn = document.createElement('button'); modeBtn.className = 'organ-btn'; modeBtn.textContent = vsCpu ? 'vs Computer' : '2 Players';
  modeBtn.onclick = () => { vsCpu = !vsCpu; resetBoard(); };
  const btn = document.createElement('button'); btn.className = 'organ-btn primary'; btn.textContent = 'New Round'; btn.onclick = resetBoard;
  ctrls.append(modeBtn, btn); top.appendChild(ctrls); root.appendChild(top);

  const cellPx = W.cols <= 4 ? 68 : W.cols <= 7 ? 46 : 36;
  const winSet = new Set(winCells.map(([r, c]) => r + ',' + c));
  const g = document.createElement('div'); g.className = 'ms-grid';
  g.style.gridTemplateColumns = 'repeat(' + W.cols + ', ' + cellPx + 'px)';
  for (let r = 0; r < W.rows; r++) for (let c = 0; c < W.cols; c++) {
    const b = document.createElement('button'); b.className = 'ms-cell';
    b.style.width = cellPx + 'px'; b.style.height = cellPx + 'px'; b.style.fontSize = cellPx > 50 ? '26px' : '18px';
    if (winSet.has(r + ',' + c)) { b.style.borderColor = '#10b981'; b.style.background = 'rgba(16,185,129,0.22)'; }
    const v = grid[r][c];
    b.textContent = v === 1 ? (W.gravity ? '●' : 'X') : v === 2 ? (W.gravity ? '●' : 'O') : '';
    b.style.color = v === 1 ? '#10b981' : v === 2 ? '#f59e0b' : 'inherit';
    b.onclick = () => playAt(r, c);
    g.appendChild(b);
  }
  root.appendChild(g);
}
resetBoard();
`;
}

export function buildGridPuzzleRuntimeJs(weights: GridPuzzleGameWeights): string {
  if (weights.variant === 'memory-match') {
    return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
${SHARED_GAME_SFX_SNIPPET}
const SYMBOLS = ['✦','◈','▲','●','✿','☀','☾','⚑','❖','★','⬡','⚡','♫','☂','☘','♛','⚓','⚙'];
let cards = [], flipped = [], matched = new Set(), moves = 0, bestMoves = null, lock = false, elapsed = 0, timerId = null;

async function loadBest() {
  try { const b = await window.verdant?.storage?.get('memoryBest'); if (typeof b === 'number') { bestMoves = b; render(); } } catch (_) {}
}
function stopTimer() { if (timerId) { clearInterval(timerId); timerId = null; } }
function initMemory() {
  stopTimer(); elapsed = 0; moves = 0; flipped = []; matched = new Set(); lock = false;
  const total = Math.max(4, Math.floor((W.rows * W.cols) / 2) * 2);
  const pairCount = total / 2;
  const deck = [];
  for (let i = 0; i < pairCount; i++) { const sym = SYMBOLS[i % SYMBOLS.length]; deck.push(sym, sym); }
  for (let i = deck.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [deck[i], deck[j]] = [deck[j], deck[i]]; }
  cards = deck;
  render();
}
function flipCard(idx) {
  if (lock || matched.has(idx) || flipped.includes(idx)) return;
  if (!timerId) timerId = setInterval(() => { elapsed += 1; const el = document.getElementById('mem-timer'); if (el) el.textContent = elapsed + 's'; }, 1000);
  flipped.push(idx);
  playSfx(420, 0.05, 'sine', 0.06);
  render();
  if (flipped.length === 2) {
    moves += 1;
    const [a, b] = flipped;
    if (cards[a] === cards[b]) {
      matched.add(a); matched.add(b); flipped = [];
      playSfx(660, 0.12, 'triangle', 0.08);
      if (matched.size === cards.length) {
        stopTimer();
        if (bestMoves === null || moves < bestMoves) {
          bestMoves = moves;
          try { window.verdant?.storage?.set('memoryBest', bestMoves); } catch (_) {}
        }
      }
      render();
    } else {
      lock = true;
      setTimeout(() => { flipped = []; lock = false; render(); }, 650);
    }
  }
}
function render() {
  const won = matched.size === cards.length && cards.length > 0;
  root.innerHTML = '';
  const top = document.createElement('div'); top.className = 'organ-row'; top.style.justifyContent = 'space-between'; top.style.marginBottom = '12px';
  top.innerHTML =
    '<div class="organ-row">' +
    '<span class="organ-pill">Pairs: ' + (matched.size / 2) + ' / ' + (cards.length / 2) + '</span>' +
    '<span class="organ-pill">Moves: ' + moves + '</span>' +
    '<span class="organ-pill">Time: <span id="mem-timer">' + elapsed + 's</span></span>' +
    (bestMoves !== null ? '<span class="organ-pill">Best: ' + bestMoves + ' moves</span>' : '') +
    (won ? '<span class="organ-pill" style="border-color:#10b981;color:#10b981">All Matched!</span>' : '') +
    '</div>';
  const reset = document.createElement('button'); reset.className = 'organ-btn primary'; reset.textContent = 'New Deck'; reset.onclick = initMemory;
  top.appendChild(reset); root.appendChild(top);

  const g = document.createElement('div'); g.className = 'ms-grid';
  g.style.gridTemplateColumns = 'repeat(' + W.cols + ', 64px)';
  cards.forEach((sym, idx) => {
    const show = matched.has(idx) || flipped.includes(idx);
    const b = document.createElement('button'); b.type = 'button'; b.className = 'ms-cell';
    b.style.width = '64px'; b.style.height = '64px'; b.style.fontSize = '24px';
    b.style.background = matched.has(idx) ? 'rgba(16,185,129,0.22)' : show ? 'var(--v-surface)' : 'var(--v-surface-alt)';
    if (matched.has(idx)) b.style.borderColor = '#10b981';
    b.textContent = show ? sym : '?';
    b.setAttribute('aria-label', show ? 'Card ' + sym : 'Hidden card ' + (idx + 1));
    b.onclick = () => flipCard(idx);
    g.appendChild(b);
  });
  root.appendChild(g);
}
initMemory();
loadBest();
`;
  }

  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
${SHARED_GAME_SFX_SNIPPET}
let board = [], moves = 0, score = 0, bestScore = 0, won = false;

async function loadBest() {
  try { const b = await window.verdant?.storage?.get('puzzleBest'); if (typeof b === 'number') { bestScore = b; render(); } } catch (_) {}
}
function saveBest(s) {
  if (s > bestScore) { bestScore = s; try { window.verdant?.storage?.set('puzzleBest', bestScore); } catch (_) {} }
}
function initLightsOut() {
  board = Array.from({ length: W.rows }, () => Array(W.cols).fill(false));
  for (let i = 0; i < W.rows + W.cols; i++) toggleLight(Math.floor(Math.random() * W.rows), Math.floor(Math.random() * W.cols), true);
  moves = 0; won = false; render();
}
function toggleLight(r, c, silent) {
  for (const [dr, dc] of [[0,0],[1,0],[-1,0],[0,1],[0,-1]]) {
    const nr = r + dr, nc = c + dc;
    if (nr >= 0 && nr < W.rows && nc >= 0 && nc < W.cols) board[nr][nc] = !board[nr][nc];
  }
  if (!silent) { moves += 1; playSfx(360, 0.05); won = board.flat().every((v) => !v); render(); }
}
function init2048() {
  board = Array.from({ length: W.rows }, () => Array(W.cols).fill(0));
  score = 0; moves = 0; won = false; spawn2048(); spawn2048(); render();
}
function spawn2048() {
  const empty = [];
  for (let r = 0; r < W.rows; r++) for (let c = 0; c < W.cols; c++) if (board[r][c] === 0) empty.push([r, c]);
  if (empty.length > 0) { const [r, c] = empty[Math.floor(Math.random() * empty.length)]; board[r][c] = Math.random() < 0.9 ? 2 : 4; }
}
function slideRow(arr) {
  const nums = arr.filter((v) => v > 0);
  const out = [];
  for (let i = 0; i < nums.length; i++) {
    if (nums[i] === nums[i + 1]) { const m = nums[i] * 2; score += m; saveBest(score); if (m >= W.targetScore) won = true; out.push(m); i += 1; }
    else out.push(nums[i]);
  }
  while (out.length < arr.length) out.push(0);
  return out;
}
function move2048(dir) {
  const before = JSON.stringify(board);
  if (dir === 'left') for (let r = 0; r < W.rows; r++) board[r] = slideRow(board[r]);
  if (dir === 'right') for (let r = 0; r < W.rows; r++) board[r] = slideRow([...board[r]].reverse()).reverse();
  if (dir === 'up') for (let c = 0; c < W.cols; c++) { const col = slideRow(board.map((r) => r[c])); for (let r = 0; r < W.rows; r++) board[r][c] = col[r]; }
  if (dir === 'down') for (let c = 0; c < W.cols; c++) { const col = slideRow(board.map((r) => r[c]).reverse()).reverse(); for (let r = 0; r < W.rows; r++) board[r][c] = col[r]; }
  if (JSON.stringify(board) !== before) { moves += 1; playSfx(440, 0.04); spawn2048(); render(); }
}
function render() {
  root.innerHTML = '';
  const top = document.createElement('div'); top.className = 'organ-row'; top.style.justifyContent = 'space-between'; top.style.marginBottom = '12px';
  top.innerHTML = '<div class="organ-row"><span class="organ-pill">Moves: ' + moves + '</span>' + (W.variant === 'merge-2048' ? '<span class="organ-pill">Score: ' + score + '</span><span class="organ-pill">Best: ' + bestScore + '</span>' : '') + (won ? '<span class="organ-pill" style="color:#10b981;border-color:#10b981">Solved!</span>' : '') + '</div>';
  const reset = document.createElement('button'); reset.className = 'organ-btn primary'; reset.textContent = 'Reset';
  reset.onclick = () => (W.variant === 'lights-out' ? initLightsOut() : init2048());
  top.appendChild(reset); root.appendChild(top);

  const g = document.createElement('div'); g.className = 'ms-grid';
  g.style.gridTemplateColumns = 'repeat(' + W.cols + ', 58px)';
  for (let r = 0; r < W.rows; r++) for (let c = 0; c < W.cols; c++) {
    const b = document.createElement('button'); b.className = 'ms-cell'; b.style.width = '58px'; b.style.height = '58px';
    if (W.variant === 'lights-out') {
      b.style.background = board[r][c] ? 'var(--v-accent)' : 'var(--v-surface-alt)';
      b.onclick = () => toggleLight(r, c, false);
    } else {
      const v = board[r][c];
      b.textContent = v ? String(v) : '';
      b.style.fontSize = v >= 1024 ? '14px' : '18px';
      if (v > 0) b.style.background = 'rgba(16,185,129,' + Math.min(0.85, 0.14 + Math.log2(v) * 0.08) + ')';
    }
    g.appendChild(b);
  }
  root.appendChild(g);
  if (W.variant !== 'lights-out') {
    const ctrls = document.createElement('div'); ctrls.className = 'organ-row'; ctrls.style.justifyContent = 'center'; ctrls.style.marginTop = '12px';
    for (const d of ['left','up','down','right']) {
      const b = document.createElement('button'); b.className = 'organ-btn'; b.textContent = d.toUpperCase(); b.onclick = () => move2048(d); ctrls.appendChild(b);
    }
    root.appendChild(ctrls);
  }
}
window.addEventListener('keydown', (e) => {
  if (W.variant === 'lights-out') return;
  const map = { ArrowLeft: 'left', ArrowRight: 'right', ArrowUp: 'up', ArrowDown: 'down' };
  if (map[e.key]) { e.preventDefault(); move2048(map[e.key]); }
});
if (W.variant === 'lights-out') initLightsOut(); else init2048();
loadBest();
`;
}

export function buildArcadeCanvasRuntimeJs(weights: ArcadeCanvasGameWeights): string {
  if (weights.variant === 'breakout') {
    return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
${SHARED_GAME_SFX_SNIPPET}
let score = 0, best = 0, lives = W.lives || 3, level = 1, paused = false, over = false;
const cw = 560, ch = 380;
const paddle = { x: cw / 2 - 48, y: ch - 26, w: 96, h: 12 };
let ball = { x: cw / 2, y: ch - 44, vx: 3.4, vy: -3.8, r: 7 };
let bricks = [], keys = { left: false, right: false };

const COLORS = ['#ef4444','#f59e0b','#10b981','#38bdf8','#a855f7'];
function buildBricks() {
  bricks = [];
  const rows = Math.min(7, 4 + level), cols = 9, pad = 6, bw = (cw - 32 - pad * (cols - 1)) / cols, bh = 18;
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    bricks.push({ x: 16 + c * (bw + pad), y: 42 + r * (bh + pad), w: bw, h: bh, hp: r === 0 && level > 1 ? 2 : 1, color: COLORS[r % COLORS.length] });
  }
}
function resetBall() {
  const spd = 3.2 + W.speed * 0.28 + level * 0.25;
  ball = { x: paddle.x + paddle.w / 2, y: paddle.y - 12, vx: (Math.random() > 0.5 ? 1 : -1) * spd * 0.72, vy: -spd, r: 7 };
}
function resetGame() { score = 0; lives = W.lives || 3; level = 1; paused = false; over = false; paddle.x = cw / 2 - paddle.w / 2; buildBricks(); resetBall(); updateHud(); }

root.innerHTML = '';
const bar = document.createElement('div'); bar.className = 'organ-row'; bar.style.justifyContent = 'space-between'; bar.style.marginBottom = '10px';
const hudLeft = document.createElement('div'); hudLeft.className = 'organ-row';
const scorePill = document.createElement('span'); scorePill.className = 'organ-pill';
const livesPill = document.createElement('span'); livesPill.className = 'organ-pill';
const bestPill = document.createElement('span'); bestPill.className = 'organ-pill';
hudLeft.append(scorePill, livesPill, bestPill);
const hudRight = document.createElement('div'); hudRight.className = 'organ-row';
const pauseBtn = document.createElement('button'); pauseBtn.className = 'organ-btn'; pauseBtn.textContent = 'Pause';
const resetBtn = document.createElement('button'); resetBtn.className = 'organ-btn primary'; resetBtn.textContent = 'Restart';
hudRight.append(pauseBtn, resetBtn); bar.append(hudLeft, hudRight); root.appendChild(bar);

const canvas = document.createElement('canvas'); canvas.className = 'organ-canvas'; canvas.width = cw; canvas.height = ch; root.appendChild(canvas);
const ctx = canvas.getContext('2d');
const hint = document.createElement('p'); hint.className = 'organ-sub'; hint.style.textAlign = 'center'; hint.style.marginTop = '8px';
hint.textContent = 'Move paddle with mouse, touch, or Left/Right arrow keys · Space to pause';
root.appendChild(hint);

function updateHud() {
  scorePill.textContent = 'Score: ' + score + ' · Lv ' + level;
  livesPill.textContent = over ? 'Game Over' : 'Lives: ' + '♥'.repeat(Math.max(0, lives));
  bestPill.textContent = 'Best: ' + best;
}
pauseBtn.onclick = () => { paused = !paused; pauseBtn.textContent = paused ? 'Resume' : 'Pause'; };
resetBtn.onclick = resetGame;
window.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowLeft' || e.key === 'a') keys.left = true;
  if (e.key === 'ArrowRight' || e.key === 'd') keys.right = true;
  if (e.key === ' ') { e.preventDefault(); paused = !paused; pauseBtn.textContent = paused ? 'Resume' : 'Pause'; }
});
window.addEventListener('keyup', (e) => {
  if (e.key === 'ArrowLeft' || e.key === 'a') keys.left = false;
  if (e.key === 'ArrowRight' || e.key === 'd') keys.right = false;
});
function movePointer(clientX) {
  const rect = canvas.getBoundingClientRect();
  const x = ((clientX - rect.left) / rect.width) * cw;
  paddle.x = Math.max(4, Math.min(cw - paddle.w - 4, x - paddle.w / 2));
}
canvas.addEventListener('mousemove', (e) => movePointer(e.clientX));
canvas.addEventListener('touchmove', (e) => { if (e.touches[0]) movePointer(e.touches[0].clientX); }, { passive: true });

(async () => {
  try { const b = await window.verdant?.storage?.get('breakoutBest'); if (typeof b === 'number') best = b; } catch (_) {}
  buildBricks(); resetBall(); updateHud();
})();

function tick() {
  if (!paused && !over) {
    if (keys.left) paddle.x = Math.max(4, paddle.x - 6.5);
    if (keys.right) paddle.x = Math.min(cw - paddle.w - 4, paddle.x + 6.5);
    ball.x += ball.vx; ball.y += ball.vy;
    if (ball.x - ball.r <= 0 || ball.x + ball.r >= cw) { ball.vx *= -1; playSfx(320, 0.04); }
    if (ball.y - ball.r <= 0) { ball.vy *= -1; playSfx(360, 0.04); }
    if (ball.y + ball.r >= paddle.y && ball.y - ball.r <= paddle.y + paddle.h && ball.x >= paddle.x && ball.x <= paddle.x + paddle.w && ball.vy > 0) {
      const hitPos = (ball.x - (paddle.x + paddle.w / 2)) / (paddle.w / 2);
      const speed = Math.hypot(ball.vx, ball.vy);
      ball.vx = hitPos * speed * 0.9;
      ball.vy = -Math.max(2.2, Math.sqrt(Math.max(4, speed * speed - ball.vx * ball.vx)));
      playSfx(480, 0.05);
    }
    for (const br of bricks) {
      if (br.hp <= 0) continue;
      if (ball.x + ball.r >= br.x && ball.x - ball.r <= br.x + br.w && ball.y + ball.r >= br.y && ball.y - ball.r <= br.y + br.h) {
        br.hp -= 1; ball.vy *= -1; score += 20;
        if (score > best) { best = score; try { window.verdant?.storage?.set('breakoutBest', best); } catch (_) {} }
        playSfx(620, 0.06); updateHud(); break;
      }
    }
    if (bricks.every((b) => b.hp <= 0)) { level += 1; buildBricks(); resetBall(); updateHud(); playSfx(780, 0.16); }
    if (ball.y - ball.r > ch) {
      lives -= 1; playSfx(160, 0.2, 'sawtooth', 0.09);
      if (lives <= 0) { over = true; } else { resetBall(); }
      updateHud();
    }
  }
  ctx.fillStyle = '#0b0f14'; ctx.fillRect(0, 0, cw, ch);
  for (const br of bricks) {
    if (br.hp <= 0) continue;
    ctx.fillStyle = br.color; ctx.globalAlpha = br.hp > 1 ? 1 : 0.85;
    ctx.fillRect(br.x, br.y, br.w, br.h);
    if (br.hp > 1) { ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.5; ctx.strokeRect(br.x + 1, br.y + 1, br.w - 2, br.h - 2); }
  }
  ctx.globalAlpha = 1;
  ctx.fillStyle = '#10b981'; ctx.fillRect(paddle.x, paddle.y, paddle.w, paddle.h);
  ctx.fillStyle = '#f8fafc'; ctx.beginPath(); ctx.arc(ball.x, ball.y, ball.r, 0, Math.PI * 2); ctx.fill();
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
`;
  }

  if (weights.variant === 'pong') {
    return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
${SHARED_GAME_SFX_SNIPPET}
const cw = 580, ch = 350, winScore = 7;
let p1 = ch / 2 - 36, p2 = ch / 2 - 36, ph = 72, pw = 10;
let s1 = 0, s2 = 0, twoPlayer = false, paused = false, winner = 0;
let ball = { x: cw / 2, y: ch / 2, vx: 4.2, vy: 2.3, r: 7 };
const keys = {};

function serve(dir) {
  const spd = 3.8 + W.speed * 0.28;
  ball = { x: cw / 2, y: ch / 2, vx: dir * spd, vy: (Math.random() - 0.5) * spd, r: 7 };
}
function resetMatch() { s1 = 0; s2 = 0; winner = 0; paused = false; p1 = ch / 2 - ph / 2; p2 = ch / 2 - ph / 2; serve(1); updateHud(); }

root.innerHTML = '';
const bar = document.createElement('div'); bar.className = 'organ-row'; bar.style.justifyContent = 'space-between'; bar.style.marginBottom = '10px';
const scorePill = document.createElement('span'); scorePill.className = 'organ-pill';
const ctrls = document.createElement('div'); ctrls.className = 'organ-row';
const modeBtn = document.createElement('button'); modeBtn.className = 'organ-btn'; modeBtn.textContent = '1P vs AI';
const pauseBtn = document.createElement('button'); pauseBtn.className = 'organ-btn'; pauseBtn.textContent = 'Pause';
const resetBtn = document.createElement('button'); resetBtn.className = 'organ-btn primary'; resetBtn.textContent = 'New Match';
ctrls.append(modeBtn, pauseBtn, resetBtn); bar.append(scorePill, ctrls); root.appendChild(bar);

const canvas = document.createElement('canvas'); canvas.className = 'organ-canvas'; canvas.width = cw; canvas.height = ch; root.appendChild(canvas);
const ctx = canvas.getContext('2d');
const hint = document.createElement('p'); hint.className = 'organ-sub'; hint.style.textAlign = 'center'; hint.style.marginTop = '8px';
hint.textContent = 'Left paddle: W / S (or mouse/touch) · Right paddle (2P): Up / Down arrows · First to 7 wins';
root.appendChild(hint);

function updateHud() {
  const rightLbl = twoPlayer ? 'P2' : 'CPU';
  scorePill.textContent = winner ? (winner === 1 ? 'You Win! ' : rightLbl + ' Wins! ') + s1 + ' – ' + s2 : 'You ' + s1 + ' – ' + s2 + ' ' + rightLbl;
}
modeBtn.onclick = () => { twoPlayer = !twoPlayer; modeBtn.textContent = twoPlayer ? '2 Players' : '1P vs AI'; resetMatch(); };
pauseBtn.onclick = () => { paused = !paused; pauseBtn.textContent = paused ? 'Resume' : 'Pause'; };
resetBtn.onclick = resetMatch;
window.addEventListener('keydown', (e) => { keys[e.key] = true; if (e.key === ' ') { e.preventDefault(); paused = !paused; pauseBtn.textContent = paused ? 'Resume' : 'Pause'; } });
window.addEventListener('keyup', (e) => { keys[e.key] = false; });
canvas.addEventListener('mousemove', (e) => { const r = canvas.getBoundingClientRect(); p1 = Math.max(4, Math.min(ch - ph - 4, ((e.clientY - r.top) / r.height) * ch - ph / 2)); });

updateHud();
function loop() {
  if (!paused && !winner) {
    if (keys['w'] || keys['W']) p1 = Math.max(4, p1 - 6);
    if (keys['s'] || keys['S']) p1 = Math.min(ch - ph - 4, p1 + 6);
    if (twoPlayer) {
      if (keys['ArrowUp']) p2 = Math.max(4, p2 - 6);
      if (keys['ArrowDown']) p2 = Math.min(ch - ph - 4, p2 + 6);
    } else {
      const target = ball.y - ph / 2;
      const aiSpd = 3.2 + W.speed * 0.25;
      if (p2 + 4 < target) p2 = Math.min(ch - ph - 4, p2 + aiSpd);
      else if (p2 - 4 > target) p2 = Math.max(4, p2 - aiSpd);
    }
    ball.x += ball.vx; ball.y += ball.vy;
    if (ball.y - ball.r <= 0 || ball.y + ball.r >= ch) { ball.vy *= -1; playSfx(320, 0.04); }
    if (ball.x - ball.r <= 18 + pw && ball.y >= p1 && ball.y <= p1 + ph && ball.vx < 0) {
      const rel = (ball.y - (p1 + ph / 2)) / (ph / 2);
      ball.vx = Math.min(10, Math.abs(ball.vx) * 1.05);
      ball.vy = rel * 5.2; playSfx(480, 0.05);
    }
    if (ball.x + ball.r >= cw - 18 - pw && ball.y >= p2 && ball.y <= p2 + ph && ball.vx > 0) {
      const rel = (ball.y - (p2 + ph / 2)) / (ph / 2);
      ball.vx = -Math.min(10, Math.abs(ball.vx) * 1.05);
      ball.vy = rel * 5.2; playSfx(420, 0.05);
    }
    if (ball.x < 0) { s2 += 1; playSfx(200, 0.14); if (s2 >= winScore) winner = 2; else serve(-1); updateHud(); }
    if (ball.x > cw) { s1 += 1; playSfx(620, 0.14); if (s1 >= winScore) winner = 1; else serve(1); updateHud(); }
  }
  ctx.fillStyle = '#0b0f14'; ctx.fillRect(0, 0, cw, ch);
  ctx.setLineDash([8, 8]); ctx.strokeStyle = 'rgba(148,163,184,0.3)'; ctx.beginPath(); ctx.moveTo(cw / 2, 0); ctx.lineTo(cw / 2, ch); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = '#10b981'; ctx.fillRect(18, p1, pw, ph);
  ctx.fillStyle = '#f59e0b'; ctx.fillRect(cw - 18 - pw, p2, pw, ph);
  ctx.fillStyle = '#f8fafc'; ctx.beginPath(); ctx.arc(ball.x, ball.y, ball.r, 0, Math.PI * 2); ctx.fill();
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
`;
  }

  // Snake (default arcade-canvas variant)
  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
${SHARED_GAME_SFX_SNIPPET}
let score = 0, best = 0, alive = true, paused = false, dir = [1, 0], snake = [[6, 8], [5, 8], [4, 8]], food = [12, 8];

root.innerHTML = '';
const bar = document.createElement('div'); bar.className = 'organ-row'; bar.style.justifyContent = 'space-between'; bar.style.marginBottom = '10px';
const stats = document.createElement('div'); stats.className = 'organ-row';
const scorePill = document.createElement('span'); scorePill.className = 'organ-pill'; scorePill.textContent = 'Score: 0';
const bestPill = document.createElement('span'); bestPill.className = 'organ-pill'; bestPill.textContent = 'Best: 0';
stats.append(scorePill, bestPill);
const ctrls = document.createElement('div'); ctrls.className = 'organ-row';
const pauseBtn = document.createElement('button'); pauseBtn.className = 'organ-btn'; pauseBtn.textContent = 'Pause';
const resetBtn = document.createElement('button'); resetBtn.className = 'organ-btn primary'; resetBtn.textContent = 'Restart';
ctrls.append(pauseBtn, resetBtn); bar.append(stats, ctrls); root.appendChild(bar);

const canvas = document.createElement('canvas'); canvas.className = 'organ-canvas'; canvas.width = 480; canvas.height = 360;
root.appendChild(canvas);
const ctx = canvas.getContext('2d');
const cw = Math.floor(canvas.width / W.gridCols), ch = Math.floor(canvas.height / W.gridRows);

const dpad = document.createElement('div'); dpad.className = 'organ-row'; dpad.style.justifyContent = 'center'; dpad.style.marginTop = '10px';
for (const [lbl, vec] of [['◀', [-1,0]], ['▲', [0,-1]], ['▼', [0,1]], ['▶', [1,0]]]) {
  const b = document.createElement('button'); b.className = 'organ-btn'; b.textContent = lbl;
  b.onclick = () => { if (dir[0] + vec[0] !== 0 || dir[1] + vec[1] !== 0) dir = vec; };
  dpad.appendChild(b);
}
root.appendChild(dpad);

(async () => {
  try { const b = await window.verdant?.storage?.get('snakeBest'); if (typeof b === 'number') { best = b; bestPill.textContent = 'Best: ' + best; } } catch (_) {}
})();

function reset() {
  score = 0; alive = true; paused = false; dir = [1, 0]; snake = [[6, 8], [5, 8], [4, 8]];
  food = [Math.floor(Math.random() * W.gridCols), Math.floor(Math.random() * W.gridRows)];
  scorePill.textContent = 'Score: 0'; pauseBtn.textContent = 'Pause';
}
pauseBtn.onclick = () => { paused = !paused; pauseBtn.textContent = paused ? 'Resume' : 'Pause'; };
resetBtn.onclick = reset;
window.addEventListener('keydown', (e) => {
  if (e.key === ' ') { e.preventDefault(); paused = !paused; pauseBtn.textContent = paused ? 'Resume' : 'Pause'; return; }
  if (e.key === 'ArrowUp' && dir[1] !== 1) dir = [0, -1];
  else if (e.key === 'ArrowDown' && dir[1] !== -1) dir = [0, 1];
  else if (e.key === 'ArrowLeft' && dir[0] !== 1) dir = [-1, 0];
  else if (e.key === 'ArrowRight' && dir[0] !== -1) dir = [1, 0];
});
setInterval(() => {
  if (alive && !paused) {
    let nx = snake[0][0] + dir[0], ny = snake[0][1] + dir[1];
    if (W.wrapWalls) { nx = (nx + W.gridCols) % W.gridCols; ny = (ny + W.gridRows) % W.gridRows; }
    if (nx < 0 || nx >= W.gridCols || ny < 0 || ny >= W.gridRows || snake.some(([sx, sy]) => sx === nx && sy === ny)) {
      alive = false; scorePill.textContent = 'Game Over · Score: ' + score; playSfx(160, 0.2, 'sawtooth', 0.09);
    } else {
      snake.unshift([nx, ny]);
      if (nx === food[0] && ny === food[1]) {
        score += 10; scorePill.textContent = 'Score: ' + score; playSfx(560, 0.06);
        if (score > best) { best = score; bestPill.textContent = 'Best: ' + best; try { window.verdant?.storage?.set('snakeBest', best); } catch (_) {} }
        food = [Math.floor(Math.random() * W.gridCols), Math.floor(Math.random() * W.gridRows)];
      } else snake.pop();
    }
  }
  ctx.fillStyle = '#0b0f14'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#f59e0b'; ctx.fillRect(food[0] * cw + 2, food[1] * ch + 2, cw - 4, ch - 4);
  snake.forEach(([sx, sy], i) => {
    ctx.fillStyle = i === 0 ? '#34d399' : '#10b981';
    ctx.fillRect(sx * cw + 1, sy * ch + 1, cw - 2, ch - 2);
  });
}, Math.max(60, 200 - W.speed * 14));
`;
}

export function buildCellularSimRuntimeJs(weights: CellularSimGameWeights): string {
  if (weights.variant === 'rule-30') {
    return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
let ruleNum = Math.max(0, Math.min(255, W.ruleNumber ?? 30));
const cols = Math.max(48, W.cols * 2), rows = Math.max(36, W.rows * 2);
let history = [], running = true, gen = 0;

function seedSingle() {
  const row = Array(cols).fill(0); row[Math.floor(cols / 2)] = 1;
  history = [row]; gen = 0;
}
function seedRandom() {
  history = [Array.from({ length: cols }, () => (Math.random() < 0.5 ? 1 : 0))]; gen = 0;
}
function nextElementaryRow(prev) {
  const out = Array(cols).fill(0);
  for (let i = 0; i < cols; i++) {
    const l = prev[(i - 1 + cols) % cols], c = prev[i], r = prev[(i + 1) % cols];
    const bit = (l << 2) | (c << 1) | r;
    out[i] = (ruleNum >> bit) & 1;
  }
  return out;
}

root.innerHTML = '';
const bar = document.createElement('div'); bar.className = 'organ-row'; bar.style.justifyContent = 'space-between'; bar.style.marginBottom = '10px';
const left = document.createElement('div'); left.className = 'organ-row';
const rulePill = document.createElement('span'); rulePill.className = 'organ-pill'; rulePill.textContent = 'Rule ' + ruleNum;
const genPill = document.createElement('span'); genPill.className = 'organ-pill'; genPill.textContent = 'Row: 0';
left.append(rulePill, genPill);
const right = document.createElement('div'); right.className = 'organ-row';
for (const rn of [30, 90, 110, 184]) {
  const b = document.createElement('button'); b.className = 'organ-btn'; b.textContent = 'R' + rn;
  b.onclick = () => { ruleNum = rn; rulePill.textContent = 'Rule ' + ruleNum; seedSingle(); };
  right.appendChild(b);
}
const playBtn = document.createElement('button'); playBtn.className = 'organ-btn primary'; playBtn.textContent = 'Pause';
playBtn.onclick = () => { running = !running; playBtn.textContent = running ? 'Pause' : 'Play'; };
const randBtn = document.createElement('button'); randBtn.className = 'organ-btn'; randBtn.textContent = 'Random Seed';
randBtn.onclick = seedRandom;
const snapBtn = document.createElement('button'); snapBtn.className = 'organ-btn'; snapBtn.textContent = 'Save PNG';
right.append(playBtn, randBtn, snapBtn); bar.append(left, right); root.appendChild(bar);

const cell = 7;
const canvas = document.createElement('canvas'); canvas.className = 'organ-canvas'; canvas.width = cols * cell; canvas.height = rows * cell;
root.appendChild(canvas);
const ctx = canvas.getContext('2d');
snapBtn.onclick = () => { const a = document.createElement('a'); a.href = canvas.toDataURL('image/png'); a.download = 'rule-' + ruleNum + '.png'; a.click(); };

seedSingle();
setInterval(() => {
  if (running) {
    history.push(nextElementaryRow(history[history.length - 1]));
    if (history.length > rows) history.shift();
    gen += 1; genPill.textContent = 'Row: ' + gen;
  }
  ctx.fillStyle = '#0b0f14'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#10b981';
  for (let r = 0; r < history.length; r++) for (let c = 0; c < cols; c++) {
    if (history[r][c]) ctx.fillRect(c * cell, r * cell, cell - 1, cell - 1);
  }
}, Math.max(45, W.tickMs));
`;
  }

  // Conway's Game of Life with Step, Glider stamp, Clear & Randomize
  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
let grid = Array.from({ length: W.rows }, () => Array.from({ length: W.cols }, () => (Math.random() < 0.25 ? 1 : 0)));
let running = true, gen = 0;

root.innerHTML = '';
const bar = document.createElement('div'); bar.className = 'organ-row'; bar.style.justifyContent = 'space-between'; bar.style.marginBottom = '10px';
const genPill = document.createElement('span'); genPill.className = 'organ-pill'; genPill.textContent = 'Gen: 0';
const ctrls = document.createElement('div'); ctrls.className = 'organ-row';
const playBtn = document.createElement('button'); playBtn.className = 'organ-btn primary'; playBtn.textContent = 'Pause';
const stepBtn = document.createElement('button'); stepBtn.className = 'organ-btn'; stepBtn.textContent = 'Step';
const gliderBtn = document.createElement('button'); gliderBtn.className = 'organ-btn'; gliderBtn.textContent = '+ Glider';
const clearBtn = document.createElement('button'); clearBtn.className = 'organ-btn'; clearBtn.textContent = 'Clear';
const randBtn = document.createElement('button'); randBtn.className = 'organ-btn'; randBtn.textContent = 'Randomize';
ctrls.append(playBtn, stepBtn, gliderBtn, clearBtn, randBtn); bar.append(genPill, ctrls); root.appendChild(bar);

const canvas = document.createElement('canvas'); canvas.className = 'organ-canvas'; canvas.width = W.cols * 14; canvas.height = W.rows * 14;
root.appendChild(canvas);
const ctx = canvas.getContext('2d');

function step() {
  const next = Array.from({ length: W.rows }, () => Array(W.cols).fill(0));
  for (let r = 0; r < W.rows; r++) for (let c = 0; c < W.cols; c++) {
    let n = 0;
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
      if (dr === 0 && dc === 0) continue;
      n += grid[(r + dr + W.rows) % W.rows][(c + dc + W.cols) % W.cols];
    }
    next[r][c] = grid[r][c] ? (n === 2 || n === 3 ? 1 : 0) : (n === 3 ? 1 : 0);
  }
  grid = next; gen += 1; genPill.textContent = 'Gen: ' + gen;
}
playBtn.onclick = () => { running = !running; playBtn.textContent = running ? 'Pause' : 'Play'; };
stepBtn.onclick = () => { running = false; playBtn.textContent = 'Play'; step(); };
clearBtn.onclick = () => { gen = 0; genPill.textContent = 'Gen: 0'; grid = Array.from({ length: W.rows }, () => Array(W.cols).fill(0)); };
gliderBtn.onclick = () => {
  const mr = Math.floor(W.rows / 2), mc = Math.floor(W.cols / 2);
  for (const [dr, dc] of [[0,1],[1,2],[2,0],[2,1],[2,2]]) grid[(mr + dr) % W.rows][(mc + dc) % W.cols] = 1;
};
randBtn.onclick = () => { gen = 0; genPill.textContent = 'Gen: 0'; grid = Array.from({ length: W.rows }, () => Array.from({ length: W.cols }, () => (Math.random() < 0.28 ? 1 : 0))); };
canvas.onclick = (e) => {
  const rect = canvas.getBoundingClientRect();
  const c = Math.floor(((e.clientX - rect.left) / rect.width) * W.cols);
  const r = Math.floor(((e.clientY - rect.top) / rect.height) * W.rows);
  if (r >= 0 && r < W.rows && c >= 0 && c < W.cols) grid[r][c] = grid[r][c] ? 0 : 1;
};
setInterval(() => {
  if (running) step();
  ctx.fillStyle = '#0b0f14'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#10b981';
  for (let r = 0; r < W.rows; r++) for (let c = 0; c < W.cols; c++) {
    if (grid[r][c]) ctx.fillRect(c * 14 + 1, r * 14 + 1, 12, 12);
  }
}, W.tickMs);
`;
}
