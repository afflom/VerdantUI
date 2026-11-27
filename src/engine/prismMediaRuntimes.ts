/**
 * Feature-Complete Procedural Media Runtimes (`src/engine/prismMediaRuntimes.ts`)
 *
 * Supplies high-spec, audited procedural runtimes (zero LLM-written JS · DEC-009) for:
 *  - Picture (10-scene layered procedural renderer, Aspect/Lighting/HiDPI/Variation/PNG/JSON)
 *  - Audio (live AnalyserNode FFT/oscilloscope, 4-track mixer, kick/snare/hat percussion, 16-step grid, Stereo WAV)
 *  - Video (layered motion scenes, transitions, interactive scrubber, scene editing, audio-muxed WebM + PNG)
 *  - Written Work / Document (Write/Preview/Split Markdown, chapter CRUD, reading time, AI assist, HTML/MD/Copy)
 *  - Research Program (Hypotheses + wired Evidence Log CRUD + Milestones + AI synthesis + Markdown/JSON report)
 *  - Utilities / Tool Organ (persisted across all 4 roles: habit-guardian, schedule-guard, calculator, workflow-helper)
 *  - Space Prism (curated multi-artifact Space View with grid/board/showcase/timeline layouts & one-click openPrism)
 */

import type {
  AudioWeights,
  PictureWeights,
  ResearchProgramWeights,
  SpaceWeights,
  ToolOrganWeights,
  VideoWeights,
  WrittenWorkWeights,
} from './prismMediaOrgans';

function safeJsonForScript(obj: unknown): string {
  return JSON.stringify(obj).replace(/<\//g, '<\\/');
}

export function buildPictureRuntimeJs(weights: PictureWeights): string {
  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
const SCENES = [
  ['landscape-vista','Landscape'],
  ['cosmic-nebula','Cosmic Nebula'],
  ['architectural-scene','Architecture'],
  ['botanical-garden','Botanical'],
  ['ocean-seascape','Ocean'],
  ['abstract-flow','Abstract Flow'],
  ['geometric-mosaic','Mosaic'],
  ['constellation-atlas','Star Atlas'],
  ['poster-editorial','Poster'],
  ['pixel-scene','Pixel Scene']
];
const ASPECTS = [['16:9', 896, 504], ['4:3', 800, 600], ['1:1', 680, 680], ['9:16', 504, 896]];
let seed = W.seed || 4219, symmetry = W.symmetry || 6, density = W.density || 58;
let generator = W.generator === 'botanical-fractal' ? 'botanical-garden'
  : W.generator === 'wave-field' ? 'ocean-seascape'
  : W.generator === 'constellation' ? 'constellation-atlas'
  : W.generator === 'pixel-sprite' ? 'pixel-scene'
  : W.generator || 'landscape-vista';
let lighting = W.lighting || 'golden-hour';
let curW = W.width || 800, curH = W.height || 500, hiDpi = 1;

function mulberry32(a) {
  return function() {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function hash2(ix, iy, s) {
  let h = Math.imul(ix | 0, 374761393) + Math.imul(iy | 0, 668265263) + Math.imul(s | 0, 1442695041);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
function noise2(x, y, s) {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
  const a = hash2(ix, iy, s), b = hash2(ix + 1, iy, s), c = hash2(ix, iy + 1, s), d = hash2(ix + 1, iy + 1, s);
  return (a + (b - a) * sx) + ((c + (d - c) * sx) - (a + (b - a) * sx)) * sy;
}
function fbm(x, y, s, oct = 4) {
  let sum = 0, amp = 0.5, freq = 1, norm = 0;
  for (let o = 0; o < oct; o++) { sum += amp * noise2(x * freq, y * freq, s + o * 97); norm += amp; amp *= 0.5; freq *= 2.04; }
  return sum / norm;
}

function drawArtwork(canvas) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const rng = mulberry32(seed);
  const [cBg, cMid, cAcc, cHi] = W.palette;
  ctx.clearRect(0, 0, w, h);

  if (generator === 'landscape-vista') {
    const horizon = h * 0.62, sunX = w * (0.28 + rng() * 0.44), sunY = h * (0.22 + rng() * 0.2), sunR = Math.min(w, h) * 0.085;
    const sky = ctx.createLinearGradient(0, 0, 0, horizon);
    sky.addColorStop(0, cBg); sky.addColorStop(0.55, cMid); sky.addColorStop(1, cAcc);
    ctx.fillStyle = sky; ctx.fillRect(0, 0, w, horizon);
    if (lighting === 'midnight' || lighting === 'twilight') {
      for (let i = 0; i < density * 2; i++) {
        ctx.fillStyle = cHi; ctx.globalAlpha = 0.35 + rng() * 0.65;
        ctx.beginPath(); ctx.arc(rng() * w, rng() * horizon * 0.72, 0.8 + rng() * 1.6, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
    const halo = ctx.createRadialGradient(sunX, sunY, sunR * 0.2, sunX, sunY, sunR * 4.2);
    halo.addColorStop(0, cHi); halo.addColorStop(0.28, cAcc); halo.addColorStop(1, 'transparent');
    ctx.fillStyle = halo; ctx.fillRect(0, 0, w, horizon);
    for (let c = 0; c < 14; c++) {
      ctx.fillStyle = cHi; ctx.globalAlpha = 0.08 + rng() * 0.14;
      ctx.beginPath(); ctx.ellipse(rng() * w, horizon * (0.15 + rng() * 0.45), 60 + rng() * 130, 12 + rng() * 24, 0, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
    for (let L = 0; L < 4; L++) {
      const depth = L / 3;
      const baseH = horizon - (1 - depth) * h * 0.22 + depth * h * 0.02;
      const amp = h * (0.22 - depth * 0.04);
      ctx.beginPath(); ctx.moveTo(0, horizon);
      for (let x = 0; x <= w; x += 4) {
        const y = baseH - (fbm((x / w) * (2.6 + L * 1.4) + L * 7.1, 0.37, seed + L * 211, 5) - 0.36) * amp;
        ctx.lineTo(x, y);
      }
      ctx.lineTo(w, horizon); ctx.closePath();
      ctx.fillStyle = L < 2 ? cMid : cBg;
      ctx.globalAlpha = 0.55 + depth * 0.42;
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    // Pine treeline silhouettes
    ctx.fillStyle = cBg;
    for (let i = 0; i < Math.max(28, density); i++) {
      const side = i % 2 === 0 ? rng() * 0.34 : 0.66 + rng() * 0.34;
      const tx = side * w, th = h * (0.04 + rng() * 0.09);
      ctx.beginPath(); ctx.moveTo(tx, horizon - th); ctx.lineTo(tx - th * 0.26, horizon); ctx.lineTo(tx + th * 0.26, horizon); ctx.closePath(); ctx.fill();
    }
    // Mirrored lake with ripples & sun glitter
    ctx.save(); ctx.translate(0, horizon * 2); ctx.scale(1, -1); ctx.globalAlpha = 0.68;
    ctx.drawImage(canvas, 0, 0, w, horizon, 0, 0, w, horizon);
    ctx.restore();
    const water = ctx.createLinearGradient(0, horizon, 0, h);
    water.addColorStop(0, 'rgba(0,0,0,0.12)'); water.addColorStop(1, cBg);
    ctx.fillStyle = water; ctx.globalAlpha = 0.62; ctx.fillRect(0, horizon, w, h - horizon);
    ctx.globalAlpha = 0.38; ctx.strokeStyle = cHi;
    for (let y = horizon + 4; y < h; y += 6) {
      const span = 22 + ((y - horizon) / (h - horizon)) * 120;
      ctx.lineWidth = 1.2; ctx.beginPath();
      ctx.moveTo(sunX - span * (0.4 + rng() * 0.6), y);
      ctx.lineTo(sunX + span * (0.4 + rng() * 0.6), y); ctx.stroke();
    }
    ctx.globalAlpha = 1;
  } else if (generator === 'cosmic-nebula') {
    ctx.fillStyle = cBg; ctx.fillRect(0, 0, w, h);
    for (let i = 0; i < 24; i++) {
      const gx = rng() * w, gy = rng() * h, gr = Math.min(w, h) * (0.14 + rng() * 0.36);
      const g = ctx.createRadialGradient(gx, gy, 0, gx, gy, gr);
      g.addColorStop(0, i % 3 === 0 ? cAcc : cMid); g.addColorStop(1, 'transparent');
      ctx.globalAlpha = 0.24; ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
    }
    ctx.globalAlpha = 1;
    for (let i = 0; i < density * 4; i++) {
      const sx = rng() * w, sy = rng() * h, bright = rng() > 0.93, sr = bright ? 2.2 : 0.7 + rng() * 1.2;
      ctx.fillStyle = i % 5 === 0 ? cAcc : cHi; ctx.globalAlpha = 0.5 + rng() * 0.5;
      ctx.beginPath(); ctx.arc(sx, sy, sr, 0, Math.PI * 2); ctx.fill();
      if (bright) {
        ctx.strokeStyle = cHi; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(sx - 12, sy); ctx.lineTo(sx + 12, sy); ctx.moveTo(sx, sy - 12); ctx.lineTo(sx, sy + 12); ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    const px = w * 0.56, py = h * 0.48, pr = Math.min(w, h) * 0.2;
    ctx.save(); ctx.translate(px, py); ctx.rotate(-0.3);
    ctx.strokeStyle = cAcc; ctx.lineWidth = pr * 0.22; ctx.globalAlpha = 0.35;
    ctx.beginPath(); ctx.ellipse(0, 0, pr * 1.9, pr * 0.5, 0, Math.PI, Math.PI * 2); ctx.stroke();
    ctx.restore(); ctx.globalAlpha = 1;
    const pg = ctx.createRadialGradient(px - pr * 0.35, py - pr * 0.35, pr * 0.1, px, py, pr);
    pg.addColorStop(0, cHi); pg.addColorStop(0.35, cMid); pg.addColorStop(1, cBg);
    ctx.fillStyle = pg; ctx.beginPath(); ctx.arc(px, py, pr, 0, Math.PI * 2); ctx.fill();
    ctx.save(); ctx.translate(px, py); ctx.rotate(-0.3);
    ctx.strokeStyle = cHi; ctx.lineWidth = pr * 0.16; ctx.globalAlpha = 0.52;
    ctx.beginPath(); ctx.ellipse(0, 0, pr * 1.9, pr * 0.5, 0, 0, Math.PI); ctx.stroke();
    ctx.restore(); ctx.globalAlpha = 1;
  } else if (generator === 'architectural-scene') {
    const gY = h * 0.72;
    const bg = ctx.createLinearGradient(0, 0, 0, gY); bg.addColorStop(0, cBg); bg.addColorStop(0.6, cMid); bg.addColorStop(1, cAcc);
    ctx.fillStyle = bg; ctx.fillRect(0, 0, w, gY);
    for (let layer = 0; layer < 3; layer++) {
      const cnt = 10 + layer * 4;
      for (let i = 0; i < cnt; i++) {
        const bw = w * (0.045 + rng() * 0.085), bh = h * (0.16 + rng() * (0.44 - layer * 0.06));
        const bx = (i / cnt) * w + (rng() - 0.5) * 26, by = gY - bh;
        ctx.fillStyle = layer === 0 ? cMid : cBg; ctx.globalAlpha = 0.55 + layer * 0.22;
        ctx.fillRect(bx, by, bw, bh);
        if (layer >= 1) {
          ctx.fillStyle = cAcc; ctx.globalAlpha = 0.72;
          for (let wy = by + 10; wy < gY - 10; wy += 12) for (let wx = bx + 6; wx < bx + bw - 8; wx += 10) {
            if (rng() > 0.45) ctx.fillRect(wx, wy, 4, 6);
          }
        }
      }
    }
    ctx.globalAlpha = 1;
    ctx.save(); ctx.translate(0, gY * 2); ctx.scale(1, -1); ctx.globalAlpha = 0.45;
    ctx.drawImage(canvas, 0, 0, w, gY, 0, 0, w, gY); ctx.restore();
    ctx.fillStyle = cBg; ctx.globalAlpha = 0.5; ctx.fillRect(0, gY, w, h - gY); ctx.globalAlpha = 1;
  } else if (generator === 'botanical-garden') {
    const bg = ctx.createLinearGradient(0, 0, 0, h); bg.addColorStop(0, cBg); bg.addColorStop(1, cMid);
    ctx.fillStyle = bg; ctx.fillRect(0, 0, w, h);
    for (let i = 0; i < 24; i++) {
      ctx.fillStyle = i % 2 === 0 ? cAcc : cHi; ctx.globalAlpha = 0.14;
      ctx.beginPath(); ctx.arc(rng() * w, rng() * h * 0.85, 16 + rng() * 48, 0, Math.PI * 2); ctx.fill();
    }
    function branch(x, y, len, ang, d) {
      if (d <= 0 || len < 5) return;
      const x2 = x + Math.cos(ang) * len, y2 = y + Math.sin(ang) * len;
      ctx.strokeStyle = d <= 2 ? cAcc : cMid; ctx.lineWidth = Math.max(1.2, d * 1.5); ctx.globalAlpha = 0.9;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x2, y2); ctx.stroke();
      if (d <= 2) {
        ctx.fillStyle = d === 1 ? cHi : cAcc; ctx.globalAlpha = 0.85;
        ctx.beginPath(); ctx.arc(x2, y2, 4 + rng() * 6, 0, Math.PI * 2); ctx.fill();
      }
      const sp = 0.34 + rng() * 0.25;
      branch(x2, y2, len * 0.74, ang - sp, d - 1); branch(x2, y2, len * 0.74, ang + sp, d - 1);
    }
    branch(w * 0.5, h * 0.96, h * 0.24, -Math.PI / 2, 7);
    branch(w * 0.24, h * 0.98, h * 0.16, -Math.PI / 2 + 0.16, 5);
    branch(w * 0.76, h * 0.98, h * 0.17, -Math.PI / 2 - 0.16, 5);
    ctx.globalAlpha = 1;
  } else if (generator === 'ocean-seascape') {
    const hz = h * 0.44, sunX = w * 0.52, sunY = hz - h * 0.12, sr = Math.min(w, h) * 0.08;
    const sky = ctx.createLinearGradient(0, 0, 0, hz); sky.addColorStop(0, cBg); sky.addColorStop(0.6, cMid); sky.addColorStop(1, cAcc);
    ctx.fillStyle = sky; ctx.fillRect(0, 0, w, hz);
    const sg = ctx.createRadialGradient(sunX, sunY, 2, sunX, sunY, sr * 4);
    sg.addColorStop(0, cHi); sg.addColorStop(0.3, cAcc); sg.addColorStop(1, 'transparent');
    ctx.fillStyle = sg; ctx.fillRect(0, 0, w, hz);
    const sea = ctx.createLinearGradient(0, hz, 0, h); sea.addColorStop(0, cMid); sea.addColorStop(1, cBg);
    ctx.fillStyle = sea; ctx.fillRect(0, hz, w, h - hz);
    for (let i = 0; i < density; i++) {
      const yB = hz + ((h - hz) / (density + 1)) * (i + 1), amp = 4 + (i / density) * 16;
      ctx.beginPath();
      for (let x = 0; x <= w; x += 6) {
        const y = yB + Math.sin(x * 0.018 + i * 0.8) * amp;
        if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = i % 3 === 0 ? cHi : cAcc; ctx.globalAlpha = 0.28 + (i / density) * 0.3; ctx.lineWidth = 1.6; ctx.stroke();
    }
    ctx.globalAlpha = 1;
  } else if (generator === 'constellation-atlas') {
    ctx.fillStyle = cBg; ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = cHi; ctx.globalAlpha = 0.14;
    for (let r = 60; r < Math.max(w, h) * 0.6; r += 75) { ctx.beginPath(); ctx.arc(w / 2, h / 2, r, 0, Math.PI * 2); ctx.stroke(); }
    const pts = [];
    for (let i = 0; i < density; i++) pts.push([w * (0.08 + rng() * 0.84), h * (0.08 + rng() * 0.84), 1.5 + rng() * 3.2]);
    ctx.strokeStyle = cMid; ctx.globalAlpha = 0.35; ctx.lineWidth = 1;
    for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {
      const dx = pts[i][0] - pts[j][0], dy = pts[i][1] - pts[j][1];
      if (dx * dx + dy * dy < Math.pow(Math.min(w, h) * 0.24, 2)) { ctx.beginPath(); ctx.moveTo(pts[i][0], pts[i][1]); ctx.lineTo(pts[j][0], pts[j][1]); ctx.stroke(); }
    }
    for (let i = 0; i < pts.length; i++) {
      ctx.globalAlpha = 0.92; ctx.fillStyle = i % 4 === 0 ? cAcc : cHi;
      ctx.beginPath(); ctx.arc(pts[i][0], pts[i][1], pts[i][2], 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
  } else if (generator === 'logo-mark') {
    ctx.fillStyle = cBg; ctx.fillRect(0, 0, w, h);
    const cx = w * 0.5, cy = h * 0.45, r = Math.min(w, h) * 0.28, variant = Math.abs(seed) % 5;
    if (variant === 0) {
      ctx.strokeStyle = cAcc; ctx.lineWidth = Math.max(5, r * 0.09); ctx.lineJoin = 'round';
      ctx.beginPath();
      for (let i = 0; i < 6; i++) {
        const a = -Math.PI / 2 + (i * Math.PI) / 3, px = cx + Math.cos(a) * r, py = cy + Math.sin(a) * r;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.closePath(); ctx.fillStyle = cMid; ctx.globalAlpha = 0.34; ctx.fill(); ctx.globalAlpha = 1; ctx.stroke();
      ctx.fillStyle = cHi; ctx.beginPath(); ctx.arc(cx, cy, r * 0.22, 0, Math.PI * 2); ctx.fill();
    } else if (variant === 1) {
      ctx.strokeStyle = cMid; ctx.lineWidth = Math.max(5, r * 0.09);
      ctx.beginPath(); ctx.arc(cx, cy, r * 0.9, 0, Math.PI * 2); ctx.stroke();
      for (const ang of [-0.55, 0.55]) {
        ctx.save(); ctx.translate(cx, cy); ctx.rotate(ang);
        ctx.strokeStyle = ang < 0 ? cAcc : cHi; ctx.lineWidth = Math.max(4, r * 0.07);
        ctx.beginPath(); ctx.ellipse(0, 0, r * 1.05, r * 0.44, 0, 0, Math.PI * 2); ctx.stroke();
        ctx.restore();
      }
      ctx.fillStyle = cAcc; ctx.beginPath(); ctx.arc(cx, cy, r * 0.26, 0, Math.PI * 2); ctx.fill();
    } else if (variant === 2) {
      ctx.fillStyle = cMid; ctx.globalAlpha = 0.22; ctx.beginPath(); ctx.arc(cx, cy, r * 1.02, 0, Math.PI * 2); ctx.fill();
      ctx.globalAlpha = 1; ctx.strokeStyle = cAcc; ctx.lineWidth = Math.max(4, r * 0.07); ctx.stroke();
      ctx.fillStyle = cAcc; ctx.beginPath(); ctx.moveTo(cx, cy - r * 0.85);
      ctx.bezierCurveTo(cx + r * 0.65, cy - r * 0.4, cx + r * 0.65, cy + r * 0.45, cx, cy + r * 0.85);
      ctx.bezierCurveTo(cx - r * 0.65, cy + r * 0.45, cx - r * 0.65, cy - r * 0.4, cx, cy - r * 0.85);
      ctx.fill();
      ctx.strokeStyle = cBg; ctx.lineWidth = Math.max(4, r * 0.06); ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(cx, cy - r * 0.62); ctx.lineTo(cx, cy + r * 0.72); ctx.stroke();
    } else if (variant === 3) {
      ctx.save(); ctx.translate(cx, cy); ctx.rotate(Math.PI / 4);
      ctx.fillStyle = cMid; ctx.globalAlpha = 0.32; ctx.fillRect(-r * 0.76, -r * 0.76, r * 1.52, r * 1.52);
      ctx.globalAlpha = 1; ctx.strokeStyle = cAcc; ctx.lineWidth = Math.max(5, r * 0.09);
      ctx.strokeRect(-r * 0.76, -r * 0.76, r * 1.52, r * 1.52);
      ctx.strokeStyle = cHi; ctx.lineWidth = Math.max(3, r * 0.05);
      ctx.strokeRect(-r * 0.5, -r * 0.5, r, r);
      ctx.restore();
      ctx.fillStyle = cHi; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.font = '800 ' + Math.round(r * 0.68) + 'px Inter, sans-serif';
      ctx.fillText((W.caption || 'V').trim().charAt(0).toUpperCase() || 'V', cx, cy + 2);
    } else {
      ctx.fillStyle = cMid; ctx.globalAlpha = 0.9; ctx.beginPath(); ctx.arc(cx - r * 0.28, cy, r * 0.78, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = cAcc; ctx.globalAlpha = 0.82; ctx.beginPath(); ctx.arc(cx + r * 0.28, cy, r * 0.78, 0, Math.PI * 2); ctx.fill();
      ctx.globalAlpha = 0.94; ctx.fillStyle = cHi; ctx.beginPath(); ctx.arc(cx, cy, r * 0.3, 0, Math.PI * 2); ctx.fill();
      ctx.globalAlpha = 1;
    }
    ctx.fillStyle = cHi; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
    ctx.font = '700 ' + Math.round(Math.min(w, h) * 0.052) + 'px Inter, sans-serif';
    ctx.fillText((W.caption || 'VERDANT').toUpperCase().slice(0, 28), cx, h * 0.88);
  } else if (generator === 'cover-art') {
    const prevGen = generator;
    generator = Math.abs(seed) % 3 === 0 ? 'landscape-vista' : Math.abs(seed) % 3 === 1 ? 'cosmic-nebula' : 'ocean-seascape';
    drawArtwork(canvas);
    generator = prevGen;
    const topG = ctx.createLinearGradient(0, 0, 0, h * 0.32);
    topG.addColorStop(0, cBg); topG.addColorStop(1, 'transparent');
    ctx.globalAlpha = 0.78; ctx.fillStyle = topG; ctx.fillRect(0, 0, w, h * 0.32);
    const botG = ctx.createLinearGradient(0, h * 0.68, 0, h);
    botG.addColorStop(0, 'transparent'); botG.addColorStop(1, cBg);
    ctx.fillStyle = botG; ctx.fillRect(0, h * 0.68, w, h * 0.32);
    ctx.globalAlpha = 1;
    const m = Math.round(Math.min(w, h) * 0.07);
    ctx.strokeStyle = cHi; ctx.globalAlpha = 0.45; ctx.lineWidth = 1.5; ctx.strokeRect(m, m, w - m * 2, h - m * 2);
    ctx.globalAlpha = 1; ctx.fillStyle = cAcc; ctx.fillRect(m, h - m - 3, w - m * 2, 4);
    ctx.fillStyle = cHi; ctx.textAlign = 'center';
    ctx.font = '700 ' + Math.round(Math.min(w, h) * 0.068) + 'px Georgia, serif';
    ctx.fillText((W.caption || 'Untitled').slice(0, 28), w * 0.5, m + Math.round(Math.min(w, h) * 0.13));
    if (W.subtitle) {
      ctx.globalAlpha = 0.82; ctx.font = '500 ' + Math.round(Math.min(w, h) * 0.028) + 'px Inter, sans-serif';
      ctx.fillText(W.subtitle.slice(0, 48), w * 0.5, h - m - 22);
      ctx.globalAlpha = 1;
    }
  } else if (generator === 'poster-editorial') {
    ctx.fillStyle = cHi; ctx.fillRect(0, 0, w, h);
    const cx = w * 0.5, cy = h * 0.45, rad = Math.min(w, h) * 0.3;
    ctx.fillStyle = cBg; ctx.beginPath(); ctx.arc(cx - rad * 0.22, cy, rad, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = cMid; ctx.globalAlpha = 0.88; ctx.beginPath(); ctx.arc(cx + rad * 0.28, cy - rad * 0.12, rad * 0.72, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = cAcc; ctx.globalAlpha = 0.92; ctx.beginPath(); ctx.arc(cx + rad * 0.04, cy + rad * 0.22, rad * 0.42, 0, Math.PI * 2); ctx.fill();
    ctx.globalAlpha = 1; const m = Math.round(Math.min(w, h) * 0.06);
    ctx.fillStyle = cBg; ctx.fillRect(m, m, w - m * 2, 3); ctx.fillRect(m, h - m * 2.4, w - m * 2, 2);
    ctx.fillStyle = cAcc; ctx.fillRect(m, h - m, w - m * 2, 5);
    ctx.fillStyle = cBg; ctx.font = '700 ' + Math.round(h * 0.05) + 'px Inter, sans-serif';
    ctx.fillText((W.caption || 'VERDANT').toUpperCase().slice(0, 28), m, h - m * 1.45);
  } else if (generator === 'pixel-scene') {
    const gw = 48, gh = Math.max(24, Math.round((h / w) * gw)), cw = w / gw, ch = h / gh, hz = Math.round(gh * 0.62);
    for (let gy = 0; gy < gh; gy++) for (let gx = 0; gx < gw; gx++) {
      let col = cBg;
      if (gy < hz) {
        const d = Math.hypot(gx - gw * 0.68, gy - hz * 0.42);
        col = d < 4.5 ? cHi : d < 7 ? cAcc : gy / hz < 0.45 ? cBg : cMid;
      } else {
        const ridge = hz + Math.sin(gx * 0.32 + seed) * 3 + fbm(gx * 0.18, 0.4, seed, 2) * 5;
        col = gy < ridge ? cMid : (gx + gy) % 3 === 0 ? cBg : cMid;
      }
      ctx.fillStyle = col; ctx.fillRect(Math.floor(gx * cw), Math.floor(gy * ch), Math.ceil(cw), Math.ceil(ch));
    }
  } else if (generator === 'abstract-flow') {
    ctx.fillStyle = cBg; ctx.fillRect(0, 0, w, h);
    for (let i = 0; i < density * 1.5; i++) {
      ctx.beginPath();
      let x = rng() * w, y = rng() * h; ctx.moveTo(x, y);
      for (let s = 0; s < 36; s++) {
        const ang = fbm(x / w * 3.2, y / h * 3.2, seed + 19, 3) * Math.PI * 4;
        x += Math.cos(ang) * 14; y += Math.sin(ang) * 14; ctx.lineTo(x, y);
      }
      ctx.strokeStyle = i % 3 === 0 ? cMid : i % 3 === 1 ? cAcc : cHi;
      ctx.globalAlpha = 0.42; ctx.lineWidth = 2 + (i % 4); ctx.stroke();
    }
    ctx.globalAlpha = 1;
  } else {
    // geometric-mosaic radial mandala
    ctx.fillStyle = cBg; ctx.fillRect(0, 0, w, h);
    ctx.save(); ctx.translate(w / 2, h / 2);
    const rad = Math.min(w, h) * 0.45;
    for (let i = 0; i < density; i++) {
      const r0 = rng() * rad * 0.85, sz = 14 + rng() * 48, col = i % 3 === 0 ? cMid : i % 3 === 1 ? cAcc : cHi, a0 = rng() * Math.PI * 2;
      for (let s = 0; s < symmetry; s++) {
        ctx.save(); ctx.rotate(a0 + (s * Math.PI * 2) / symmetry);
        ctx.fillStyle = col; ctx.globalAlpha = 0.34 + rng() * 0.44;
        ctx.beginPath(); ctx.moveTo(r0, 0); ctx.lineTo(r0 + sz, sz * 0.45); ctx.lineTo(r0 + sz * 0.65, -sz * 0.45); ctx.closePath(); ctx.fill();
        ctx.strokeStyle = cHi; ctx.globalAlpha = 0.2; ctx.lineWidth = 1; ctx.stroke();
        ctx.restore();
      }
    }
    ctx.restore(); ctx.globalAlpha = 1;
  }

  // Soft vignette
  const vig = ctx.createRadialGradient(w / 2, h / 2, Math.min(w, h) * 0.35, w / 2, h / 2, Math.hypot(w / 2, h / 2));
  vig.addColorStop(0, 'transparent'); vig.addColorStop(1, 'rgba(0,0,0,0.32)');
  ctx.fillStyle = vig; ctx.fillRect(0, 0, w, h);

  if (window.verdant?.sense?.inspectCanvas) {
    try { window.verdant.sense.inspectCanvas(canvas); } catch (_) {}
  }
}

function render() {
  root.innerHTML = '';
  const canvas = document.createElement('canvas');
  canvas.className = 'organ-canvas';
  canvas.width = curW * hiDpi; canvas.height = curH * hiDpi;
  root.appendChild(canvas);
  drawArtwork(canvas);

  const foot = document.createElement('div');
  foot.className = 'doc-footer';
  foot.style.marginTop = '12px';
  const meta = document.createElement('span');
  meta.textContent = curW + '×' + curH + ' · ' + lighting.replace('-', ' ');
  const actions = document.createElement('div'); actions.className = 'organ-row';
  const genBtn = document.createElement('button'); genBtn.className = 'organ-btn ghost'; genBtn.textContent = 'New Variation';
  genBtn.onclick = () => { seed = (seed * 1664525 + 1013904223) % 999999; drawArtwork(canvas); };
  const expPng = document.createElement('button'); expPng.className = 'organ-btn primary'; expPng.textContent = 'Download PNG';
  expPng.onclick = () => { const a = document.createElement('a'); a.href = canvas.toDataURL('image/png'); a.download = (W.caption || 'artwork').toLowerCase().replace(/[^a-z0-9]+/g, '-') + '.png'; a.click(); };
  actions.append(genBtn, expPng);
  if (generator === 'logo-mark' && W.svgMarkup) {
    const expSvg = document.createElement('button'); expSvg.className = 'organ-btn'; expSvg.textContent = 'Download SVG';
    expSvg.onclick = () => { (window.verdant?.browser?.downloadFile || window.verdant?.downloadFile)?.((W.caption || 'logo').toLowerCase().replace(/[^a-z0-9]+/g, '-') + '.svg', W.svgMarkup, 'image/svg+xml'); };
    actions.appendChild(expSvg);
  }
  foot.append(meta, actions);
  root.appendChild(foot);
}
if (window.verdant?.onInput) {
  window.verdant.onInput((input) => {
    const text = (typeof input === 'string' ? input : String((input && (input.prompt || input.text)) || '')).toLowerCase();
    if (/\\b(variation|remix|regenerate)\\b/.test(text)) seed = (seed * 1664525 + 1013904223) % 999999;
    if (/\\b(golden|sunset|warm)\\b/.test(text)) lighting = 'golden-hour';
    else if (/\\b(twilight|dusk|evening)\\b/.test(text)) lighting = 'twilight';
    else if (/\\b(midnight|night|dark|noir)\\b/.test(text)) lighting = 'midnight';
    else if (/\\b(daylight|bright|morning)\\b/.test(text)) lighting = 'daylight';
    else if (/\\b(neon|cyber)\\b/.test(text)) lighting = 'neon';
    if (/\\b(16:9|widescreen)\\b/.test(text)) { curW = 896; curH = 504; }
    else if (/\\b(square|1:1)\\b/.test(text)) { curW = 640; curH = 640; }
    else if (/\\b(portrait|9:16|vertical)\\b/.test(text)) { curW = 512; curH = 768; }
    if (/\\b(high|rich|more)\\s+detail\\b/.test(text)) density = 84;
    else if (/\\b(minimal|simple|less\\s+detail)\\b/.test(text)) density = 34;
    render();
    return { ok: true, generator, lighting, width: curW, height: curH };
  });
}
render();
`;
}

export function buildAudioRuntimeJs(weights: AudioWeights): string {
  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
const SCALES = {
  pentatonic: [0, 2, 4, 7, 9, 12],
  major: [0, 2, 4, 5, 7, 9, 11, 12],
  minor: [0, 2, 3, 5, 7, 8, 10, 12],
  dorian: [0, 2, 3, 5, 7, 9, 10, 12],
};
const PROG = [0, 3, 4, 2];
let bpm = W.bpm, scaleName = W.scale || 'pentatonic', playing = false, timer = null, step = 0;
let ctx = null, masterFilter = null, analyser = null, rafId = null;
const tracks = { pad: true, bass: true, lead: true, rhythm: W.style === 'rhythm-pulse' };
function midiToHz(m) { return 440 * Math.pow(2, (m - 69) / 12); }

function ensureAudio() {
  if (!ctx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return false;
    ctx = new AC();
    masterFilter = ctx.createBiquadFilter();
    masterFilter.type = 'lowpass'; masterFilter.frequency.value = 3800;
    analyser = ctx.createAnalyser(); analyser.fftSize = 256;
    masterFilter.connect(analyser); analyser.connect(ctx.destination);
  }
  if (ctx.state === 'suspended') void ctx.resume();
  return true;
}

function triggerModalVoice(hz, dur, vel, pan, partials) {
  if (!ctx || !masterFilter) return;
  const now = ctx.currentTime;
  const outNode = typeof ctx.createStereoPanner === 'function' ? ctx.createStereoPanner() : ctx.createGain();
  if ('pan' in outNode && outNode.pan) outNode.pan.setValueAtTime(pan, now);
  outNode.connect(masterFilter);
  for (const p of partials) {
    const osc = ctx.createOscillator(), g = ctx.createGain();
    osc.type = 'sine'; osc.frequency.setValueAtTime(hz * p.ratio, now);
    g.gain.setValueAtTime(0.0008, now);
    g.gain.exponentialRampToValueAtTime(Math.max(0.001, vel * p.amp), now + p.atk);
    g.gain.exponentialRampToValueAtTime(0.0005, now + Math.max(p.atk + 0.05, dur * p.decayScale));
    osc.connect(g); g.connect(outNode); osc.start(now); osc.stop(now + dur + 0.08);
  }
}

function triggerPercussion(kind) {
  if (!ctx || !masterFilter) return;
  const now = ctx.currentTime;
  if (kind === 'kick') {
    const osc = ctx.createOscillator(), g = ctx.createGain();
    osc.frequency.setValueAtTime(145, now); osc.frequency.exponentialRampToValueAtTime(42, now + 0.14);
    g.gain.setValueAtTime(0.28, now); g.gain.exponentialRampToValueAtTime(0.001, now + 0.18);
    osc.connect(g); g.connect(masterFilter); osc.start(now); osc.stop(now + 0.2);
  } else {
    const len = Math.floor(ctx.sampleRate * (kind === 'snare' ? 0.12 : 0.045));
    const buf = ctx.createBuffer(1, len, ctx.sampleRate), data = buf.getChannelData(0);
    for (let i = 0; i < len; i++) data[i] = (Math.random() * 2 - 1) * Math.exp(-i / (len * 0.28));
    const src = ctx.createBufferSource(), f = ctx.createBiquadFilter(), g = ctx.createGain();
    src.buffer = buf; f.type = kind === 'snare' ? 'bandpass' : 'highpass'; f.frequency.value = kind === 'snare' ? 1600 : 6200;
    g.gain.setValueAtTime(kind === 'snare' ? 0.16 : 0.08, now);
    src.connect(f); f.connect(g); g.connect(masterFilter); src.start(now);
  }
}

function playStep() {
  if (!ensureAudio()) return;
  const scale = SCALES[scaleName] || SCALES.pentatonic;
  const chordDeg = PROG[Math.floor(step / 8) % PROG.length];
  const stepDur = Math.max(0.14, (60 / bpm) * 0.5);

  if (tracks.pad && step % 8 === 0) {
    for (const [k, off] of [0, 2, 4].entries()) {
      const deg = (chordDeg + off) % scale.length, oct = Math.floor((chordDeg + off) / scale.length) * 12;
      triggerModalVoice(midiToHz(W.rootMidi - 12 + scale[deg] + oct), stepDur * 7.5, 0.075, (k - 1) * 0.35, [{ ratio: 1, amp: 1, atk: 0.14, decayScale: 1 }, { ratio: 2, amp: 0.3, atk: 0.18, decayScale: 0.85 }]);
    }
  }
  if (tracks.bass && step % 2 === 0) {
    const bassNote = W.rootMidi - 24 + scale[chordDeg % scale.length] + (step % 4 === 2 ? 7 : 0);
    triggerModalVoice(midiToHz(bassNote), stepDur * 1.8, 0.11, -0.12, [{ ratio: 1, amp: 1, atk: 0.01, decayScale: 0.9 }, { ratio: 2, amp: 0.42, atk: 0.01, decayScale: 0.55 }]);
  }
  if (tracks.lead && (W.style !== 'ambient-pad' || step % 2 === 0)) {
    const arpIdx = (chordDeg + (step % 5) * 2) % scale.length;
    triggerModalVoice(midiToHz(W.rootMidi + scale[arpIdx] + (step % 6 === 5 ? 12 : 0)), stepDur * (W.style === 'ambient-pad' ? 2.8 : 1.3), 0.09, Math.sin(step * 0.9) * 0.48, [{ ratio: 1, amp: 1, atk: 0.008, decayScale: 1 }, { ratio: 2, amp: 0.38, atk: 0.008, decayScale: 0.65 }, { ratio: 3.01, amp: 0.18, atk: 0.008, decayScale: 0.4 }]);
  }
  if (tracks.rhythm) {
    if (step % 4 === 0) triggerPercussion('kick');
    if (step % 4 === 2) triggerPercussion('snare');
    triggerPercussion('hat');
  }
  step += 1;
  updateStepDots();
}

function updateStepDots() {
  const active = (step - 1) & 15;
  for (let i = 0; i < 16; i++) {
    const el = document.getElementById('aud-step-' + i);
    if (el) el.style.background = playing && i === active ? 'var(--v-accent)' : 'var(--v-surface-alt)';
  }
}

let canvasEl = null;
function drawLiveAnalyser() {
  if (canvasEl) {
    const c2d = canvasEl.getContext('2d'), w = canvasEl.width, h = canvasEl.height;
    c2d.fillStyle = '#0b0f14'; c2d.fillRect(0, 0, w, h);
    if (analyser && playing) {
      const freq = new Uint8Array(analyser.frequencyBinCount);
      analyser.getByteFrequencyData(freq);
      const bw = w / freq.length;
      for (let i = 0; i < freq.length; i++) {
        const bh = (freq[i] / 255) * (h * 0.86);
        c2d.fillStyle = i % 2 === 0 ? '#10b981' : '#38bdf8';
        c2d.fillRect(i * bw, h - bh, Math.max(2, bw - 1), bh);
      }
      const wave = new Uint8Array(analyser.fftSize);
      analyser.getByteTimeDomainData(wave);
      c2d.strokeStyle = '#f8fafc'; c2d.lineWidth = 1.6; c2d.beginPath();
      for (let i = 0; i < wave.length; i++) {
        const x = (i / wave.length) * w, y = (wave[i] / 255) * h;
        if (i === 0) c2d.moveTo(x, y); else c2d.lineTo(x, y);
      }
      c2d.stroke();
    } else {
      c2d.strokeStyle = 'rgba(16,185,129,0.45)'; c2d.lineWidth = 1.5; c2d.beginPath();
      c2d.moveTo(0, h / 2); c2d.lineTo(w, h / 2); c2d.stroke();
    }
  }
  if (playing) rafId = requestAnimationFrame(drawLiveAnalyser);
}

function stopPlay() { playing = false; if (timer) { clearInterval(timer); timer = null; } if (rafId) { cancelAnimationFrame(rafId); rafId = null; } render(); }
function startPlay() {
  if (!ensureAudio()) return;
  playing = true; playStep();
  timer = setInterval(playStep, Math.max(115, Math.round((60 / bpm) * 500)));
  render(); drawLiveAnalyser();
}

function exportWav() {
  const sr = 22050, durSec = Math.min(14, Math.max(4, W.durationSec)), total = sr * durSec;
  const left = new Float32Array(total), right = new Float32Array(total);
  const scale = SCALES[scaleName] || SCALES.pentatonic, stepSec = (60 / bpm) * 0.5, numSteps = Math.floor(durSec / stepSec);
  function addNote(t0, dur, hz, vel, pan, modes) {
    const s0 = Math.max(0, Math.floor(t0 * sr)), s1 = Math.min(total, Math.floor((t0 + dur) * sr));
    const ang = (pan + 1) * Math.PI * 0.25, gL = Math.cos(ang) * vel, gR = Math.sin(ang) * vel;
    for (let i = s0; i < s1; i++) {
      const t = (i - s0) / sr, atk = Math.min(1, t / 0.015);
      let v = 0; for (const m of modes) v += m.amp * Math.exp(-m.damp * t) * Math.sin(2 * Math.PI * hz * m.ratio * t);
      left[i] += v * atk * gL; right[i] += v * atk * gR;
    }
  }
  for (let st = 0; st < numSteps; st++) {
    const t0 = st * stepSec, chordDeg = PROG[Math.floor(st / 8) % PROG.length];
    if (tracks.pad && st % 8 === 0) for (const [k, off] of [0, 2, 4].entries()) {
      const deg = (chordDeg + off) % scale.length, oct = Math.floor((chordDeg + off) / scale.length) * 12;
      addNote(t0, stepSec * 7.8, midiToHz(W.rootMidi - 12 + scale[deg] + oct), 0.14, (k - 1) * 0.35, [{ ratio: 1, amp: 1, damp: 0.45 }, { ratio: 2, amp: 0.28, damp: 0.9 }]);
    }
    if (tracks.bass && st % 2 === 0) addNote(t0, stepSec * 1.8, midiToHz(W.rootMidi - 24 + scale[chordDeg % scale.length]), 0.22, -0.12, [{ ratio: 1, amp: 1, damp: 2.8 }, { ratio: 2, amp: 0.4, damp: 4.5 }]);
    if (tracks.lead && (W.style !== 'ambient-pad' || st % 2 === 0)) {
      const arpIdx = (chordDeg + (st % 5) * 2) % scale.length;
      addNote(t0, stepSec * 1.6, midiToHz(W.rootMidi + scale[arpIdx]), 0.18, Math.sin(st * 0.9) * 0.48, [{ ratio: 1, amp: 1, damp: 2.2 }, { ratio: 2, amp: 0.36, damp: 4.2 }]);
    }
    if (tracks.rhythm && st % 4 === 0) addNote(t0, 0.2, 62, 0.32, 0, [{ ratio: 1, amp: 1, damp: 12 }]);
  }
  const dataBytes = total * 4, buf = new ArrayBuffer(44 + dataBytes), dv = new DataView(buf);
  const ws = (off, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i)); };
  ws(0, 'RIFF'); dv.setUint32(4, 36 + dataBytes, true); ws(8, 'WAVE'); ws(12, 'fmt '); dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true); dv.setUint16(22, 2, true); dv.setUint32(24, sr, true); dv.setUint32(28, sr * 4, true);
  dv.setUint16(32, 4, true); dv.setUint16(34, 16, true); ws(36, 'data'); dv.setUint32(40, dataBytes, true);
  let p = 44;
  for (let i = 0; i < total; i++) {
    dv.setInt16(p, Math.max(-32767, Math.min(32767, Math.round(Math.tanh(left[i]) * 32767))), true); p += 2;
    dv.setInt16(p, Math.max(-32767, Math.min(32767, Math.round(Math.tanh(right[i]) * 32767))), true); p += 2;
  }
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([buf], { type: 'audio/wav' })); a.download = (W.title || 'verdant-audio').toLowerCase().replace(/[^a-z0-9]+/g, '-') + '.wav'; a.click();
}

function exportMidi() {
  const ppqn = 480, usPerBeat = Math.round(60000000 / Math.max(45, bpm));
  const scale = SCALES[scaleName] || SCALES.pentatonic, durSec = Math.min(14, Math.max(4, W.durationSec));
  const numSteps = Math.floor(durSec / ((60 / bpm) * 0.5)), msgs = [];
  msgs.push([0, [0xff, 0x51, 0x03, (usPerBeat >>> 16) & 255, (usPerBeat >>> 8) & 255, usPerBeat & 255]]);
  for (let st = 0; st < numSteps; st++) {
    const t = st * (ppqn >> 1), chordDeg = PROG[Math.floor(st / 8) % PROG.length];
    if (tracks.pad && st % 8 === 0) for (const off of [0, 2, 4]) {
      const n = Math.max(0, Math.min(127, W.rootMidi - 12 + scale[(chordDeg + off) % scale.length]));
      msgs.push([t, [0x90, n, 64]], [t + ppqn * 3, [0x80, n, 0]]);
    }
    if (tracks.lead && (W.style !== 'ambient-pad' || st % 2 === 0)) {
      const n = Math.max(0, Math.min(127, W.rootMidi + scale[(chordDeg + (st % 5) * 2) % scale.length]));
      msgs.push([t, [0x91, n, 82]], [t + (ppqn >> 1), [0x81, n, 0]]);
    }
  }
  msgs.sort((a, b) => a[0] - b[0]);
  const trk = []; let prev = 0;
  const vlq = (v) => { const s = [v & 127]; v >>>= 7; while (v > 0) { s.push((v & 127) | 128); v >>>= 7; } for (let i = s.length - 1; i >= 0; i--) trk.push(s[i]); };
  for (const [tk, bs] of msgs) { vlq(Math.max(0, tk - prev)); for (const b of bs) trk.push(b); prev = tk; }
  trk.push(0, 0xff, 0x2f, 0);
  const out = new Uint8Array(22 + trk.length), dv = new DataView(out.buffer);
  out.set([0x4d, 0x54, 0x68, 0x64], 0); dv.setUint32(4, 6, false); dv.setUint16(8, 1, false); dv.setUint16(10, 1, false); dv.setUint16(12, ppqn, false);
  out.set([0x4d, 0x54, 0x72, 0x6b], 14); dv.setUint32(18, trk.length, false); out.set(trk, 22);
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([out], { type: 'audio/midi' })); a.download = (W.title || 'verdant-audio').toLowerCase().replace(/[^a-z0-9]+/g, '-') + '.mid'; a.click();
}

function render() {
  root.innerHTML = '';
  canvasEl = document.createElement('canvas'); canvasEl.className = 'organ-canvas'; canvasEl.width = 640; canvasEl.height = 170;
  root.appendChild(canvasEl);
  drawLiveAnalyser();

  const seq = document.createElement('div'); seq.style.display = 'grid'; seq.style.gridTemplateColumns = 'repeat(16, 1fr)'; seq.style.gap = '4px'; seq.style.marginTop = '10px';
  for (let i = 0; i < 16; i++) {
    const d = document.createElement('div'); d.id = 'aud-step-' + i;
    d.style.height = '6px'; d.style.borderRadius = '999px'; d.style.background = 'var(--v-surface-alt)';
    seq.appendChild(d);
  }
  root.appendChild(seq);

  const foot = document.createElement('div'); foot.className = 'doc-footer'; foot.style.marginTop = '14px';
  const left = document.createElement('div'); left.className = 'organ-row';
  const playBtn = document.createElement('button'); playBtn.className = 'organ-btn primary'; playBtn.textContent = playing ? 'Stop' : 'Play Stereo';
  playBtn.onclick = () => (playing ? stopPlay() : startPlay());
  const meta = document.createElement('span'); meta.textContent = bpm + ' BPM · ' + scaleName[0].toUpperCase() + scaleName.slice(1);
  left.append(playBtn, meta);
  const right = document.createElement('div'); right.className = 'organ-row';
  const wavBtn = document.createElement('button'); wavBtn.className = 'organ-btn'; wavBtn.textContent = 'Download Stereo WAV'; wavBtn.onclick = exportWav;
  const midBtn = document.createElement('button'); midBtn.className = 'organ-btn ghost'; midBtn.textContent = 'MIDI'; midBtn.onclick = exportMidi;
  right.append(wavBtn, midBtn);
  foot.append(left, right); root.appendChild(foot);
}
if (window.verdant?.onInput) {
  window.verdant.onInput((input) => {
    const text = (typeof input === 'string' ? input : String((input && (input.prompt || input.text)) || '')).toLowerCase();
    const bm = text.match(/\\b(\\d{2,3})\\s*bpm\\b/);
    if (bm) bpm = Math.max(45, Math.min(200, Number(bm[1])));
    else if (/\\b(faster|upbeat|energetic)\\b/.test(text)) bpm = Math.min(170, bpm + 16);
    else if (/\\b(slower|calmer|ambient)\\b/.test(text)) bpm = Math.max(60, bpm - 14);
    for (const sc of ['pentatonic','major','minor','dorian']) if (text.includes(sc)) scaleName = sc;
    if (/\\b(drum|beat|rhythm|percussion)\\b/.test(text)) tracks.rhythm = !/\\b(no|remove|without)\\s+(drum|beat|rhythm)/.test(text);
    if (/\\bplay\\b/.test(text) && !playing) startPlay();
    else if (/\\bstop|pause\\b/.test(text) && playing) stopPlay();
    else render();
    return { ok: true, bpm, scale: scaleName };
  });
}
render();
`;
}

export function buildVideoRuntimeJs(weights: VideoWeights): string {
  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
const dims = W.aspectRatio === '1:1' ? [540, 540] : W.aspectRatio === '9:16' ? [360, 640] : [640, 360];
let scenes = W.scenes.map((s) => ({ ...s }));
function getTotalDur() { return scenes.reduce((a, s) => a + Math.max(1, s.durationSec), 0) || 6; }
let playing = true, tSec = 0, lastTs = 0, selectedIdx = 0;
let recorder = null, recChunks = [], recording = false, recTimer = null;

function renderChrome() {
  root.innerHTML = '';
  const canvas = document.createElement('canvas'); canvas.id = 'vid-canvas';
  canvas.className = 'organ-canvas'; canvas.width = dims[0]; canvas.height = dims[1];
  root.appendChild(canvas);

  const scrub = document.createElement('input'); scrub.id = 'vid-scrub'; scrub.type = 'range';
  scrub.min = '0'; scrub.max = String(getTotalDur()); scrub.step = '0.05'; scrub.value = String(tSec);
  scrub.style.width = '100%'; scrub.style.margin = '12px 0 6px';
  scrub.oninput = () => { tSec = Number(scrub.value); drawFrame(tSec); };
  root.appendChild(scrub);

  const sceneRow = document.createElement('div'); sceneRow.className = 'doc-chapters'; sceneRow.style.marginBottom = '4px';
  let accStart = 0;
  scenes.forEach((sc, idx) => {
    const startAt = accStart; accStart += sc.durationSec;
    const chip = document.createElement('button'); chip.type = 'button';
    chip.className = 'doc-ch-pill' + (idx === selectedIdx ? ' active' : '');
    chip.textContent = (idx + 1) + '. ' + sc.title;
    chip.onclick = () => { selectedIdx = idx; tSec = startAt + 0.05; renderChrome(); };
    sceneRow.appendChild(chip);
  });
  root.appendChild(sceneRow);

  const foot = document.createElement('div'); foot.className = 'doc-footer'; foot.style.marginTop = '10px';
  const left = document.createElement('div'); left.className = 'organ-row';
  const playBtn = document.createElement('button'); playBtn.className = 'organ-btn primary'; playBtn.textContent = playing ? 'Pause' : 'Play';
  playBtn.onclick = () => { playing = !playing; playBtn.textContent = playing ? 'Pause' : 'Play'; };
  const timeEl = document.createElement('span'); timeEl.id = 'vid-time'; timeEl.textContent = tSec.toFixed(1) + 's / ' + getTotalDur() + 's';
  left.append(playBtn, timeEl);

  const right = document.createElement('div'); right.className = 'organ-row';
  const snapBtn = document.createElement('button'); snapBtn.className = 'organ-btn ghost'; snapBtn.textContent = 'Save Frame PNG';
  const recBtn = document.createElement('button'); recBtn.id = 'vid-rec'; recBtn.className = 'organ-btn' + (recording ? ' primary' : '');
  recBtn.textContent = recording ? 'Stop Recording…' : 'Record WebM';
  right.append(snapBtn, recBtn);
  foot.append(left, right); root.appendChild(foot);

  snapBtn.onclick = () => { const a = document.createElement('a'); a.href = canvas.toDataURL('image/png'); a.download = 'verdant-frame.png'; a.click(); };
  recBtn.onclick = () => toggleRecord(canvas, recBtn);
}

function stopRecording(recBtn) {
  if (recTimer) { clearTimeout(recTimer); recTimer = null; }
  if (recorder && recorder.state !== 'inactive') { try { recorder.stop(); } catch (_) {} }
  recording = false;
  if (recBtn) { recBtn.textContent = 'Record WebM'; recBtn.className = 'organ-btn'; }
}

function toggleRecord(canvas, recBtn) {
  if (recording) { stopRecording(recBtn); return; }
  if (typeof canvas.captureStream !== 'function' || typeof window.MediaRecorder !== 'function') { recBtn.textContent = 'WebM Unavailable'; return; }
  try {
    const vStream = canvas.captureStream(Math.max(15, Math.min(60, W.fps || 30)));
    const tracks = [...vStream.getVideoTracks()];
    // Synthesize a coherent harmonic modal chord & melody progression into MediaRecorder mux
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (AC) {
        const ac = new AC(), dest = ac.createMediaStreamDestination(), filter = ac.createBiquadFilter();
        filter.type = 'lowpass'; filter.frequency.value = 2600; filter.connect(dest);
        const totalDur = getTotalDur(), notes = [261.63, 293.66, 329.63, 392.0, 440.0, 523.25];
        const t0 = ac.currentTime;
        for (let step = 0; step < Math.ceil(totalDur * 2); step++) {
          const at = t0 + step * 0.5, hz = notes[(step * 2 + Math.floor(step / 4)) % notes.length];
          const osc = ac.createOscillator(), g = ac.createGain();
          osc.type = 'sine'; osc.frequency.setValueAtTime(hz, at);
          g.gain.setValueAtTime(0.0008, at); g.gain.exponentialRampToValueAtTime(0.045, at + 0.04); g.gain.exponentialRampToValueAtTime(0.0006, at + 0.48);
          osc.connect(g); g.connect(filter); osc.start(at); osc.stop(at + 0.5);
          if (step % 4 === 0) {
            const pad = ac.createOscillator(), pg = ac.createGain();
            pad.type = 'triangle'; pad.frequency.setValueAtTime(hz * 0.5, at);
            pg.gain.setValueAtTime(0.0008, at); pg.gain.exponentialRampToValueAtTime(0.03, at + 0.2); pg.gain.exponentialRampToValueAtTime(0.0006, at + 1.9);
            pad.connect(pg); pg.connect(filter); pad.start(at); pad.stop(at + 2.0);
          }
        }
        setTimeout(() => { try { void ac.close(); } catch (_) {} }, Math.round(totalDur * 1000) + 350);
        tracks.push(...dest.stream.getAudioTracks());
      }
    } catch (_) {}
    const muxStream = new MediaStream(tracks);
    const mime = MediaRecorder.isTypeSupported('video/webm;codecs=vp9,opus') ? 'video/webm;codecs=vp9,opus' : MediaRecorder.isTypeSupported('video/webm') ? 'video/webm' : '';
    recorder = mime ? new MediaRecorder(muxStream, { mimeType: mime }) : new MediaRecorder(muxStream);
    recChunks = [];
    recorder.ondataavailable = (ev) => { if (ev.data && ev.data.size > 0) recChunks.push(ev.data); };
    recorder.onstop = () => {
      if (recChunks.length === 0) return;
      const url = URL.createObjectURL(new Blob(recChunks, { type: mime || 'video/webm' }));
      const a = document.createElement('a'); a.href = url; a.download = 'verdant-video.webm'; a.click();
      setTimeout(() => URL.revokeObjectURL(url), 4000);
    };
    tSec = 0; playing = true; recording = true;
    recorder.start(250); recBtn.textContent = 'Stop Recording…'; recBtn.className = 'organ-btn primary';
    recTimer = setTimeout(() => stopRecording(recBtn), Math.round(getTotalDur() * 1000) + 180);
  } catch (_) { recBtn.textContent = 'WebM Unavailable'; }
}

let sensedOnce = false;
function drawFrame(sec) {
  const canvas = document.getElementById('vid-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d'), [w, h] = dims, totalDur = getTotalDur();
  let acc = 0, scene = scenes[0], localT = 0, sIdx = 0;
  for (let i = 0; i < scenes.length; i++) {
    const s = scenes[i];
    if (sec <= acc + s.durationSec) { scene = s; localT = (sec - acc) / Math.max(0.1, s.durationSec); sIdx = i; break; }
    acc += s.durationSec;
  }
  const fade = Math.min(1, localT * 4) * Math.min(1, (1 - localT) * 5 + 0.15);
  const bg = ctx.createLinearGradient(0, 0, w, h);
  bg.addColorStop(0, W.palette[0]); bg.addColorStop(1, scene.accent);
  ctx.fillStyle = bg; ctx.fillRect(0, 0, w, h);

  // Layered parallax geometric ribbons & drifting particles
  ctx.save();
  for (let k = 0; k < 5; k++) {
    ctx.beginPath();
    const yBase = h * (0.2 + k * 0.15);
    for (let x = 0; x <= w; x += 16) {
      const y = yBase + Math.sin(x * 0.01 + localT * Math.PI * 2 + k + sIdx) * (20 + k * 6);
      if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = k % 2 === 0 ? scene.accent : W.palette[2]; ctx.globalAlpha = 0.18 * fade; ctx.lineWidth = 2; ctx.stroke();
  }
  for (let p = 0; p < 18; p++) {
    const px = ((p * 97 + sec * (24 + (p % 4) * 10)) % (w + 40)) - 20;
    const py = ((p * 53 + Math.sin(sec + p) * 28) % h + h) % h;
    ctx.fillStyle = W.palette[2]; ctx.globalAlpha = 0.25 * fade;
    ctx.beginPath(); ctx.arc(px, py, 2 + (p % 3), 0, Math.PI * 2); ctx.fill();
  }
  // Eased kinetic typography
  const ease = 1 - Math.pow(1 - Math.min(1, localT * 2.2), 3);
  const xOff = scene.motion === 'pan' ? (1 - ease) * 44 : 0;
  const yOff = scene.motion === 'rise' ? (1 - ease) * 32 : 0;
  const sc = scene.motion === 'pulse' ? 0.94 + ease * 0.06 : 1;
  ctx.translate(w / 2 + xOff, h / 2 + yOff); ctx.scale(sc, sc);
  ctx.globalAlpha = fade; ctx.textAlign = 'center';
  ctx.fillStyle = W.palette[2]; ctx.font = '700 30px Inter, sans-serif'; ctx.fillText(scene.title, 0, -6);
  ctx.fillStyle = '#e2e8f0'; ctx.font = '500 15px Inter, sans-serif'; ctx.fillText(scene.subtitle, 0, 24);
  ctx.restore();

  const timeEl = document.getElementById('vid-time'); if (timeEl) timeEl.textContent = sec.toFixed(1) + 's / ' + totalDur + 's';
  const scrub = document.getElementById('vid-scrub'); if (scrub && document.activeElement !== scrub) scrub.value = String(sec);
  if (!sensedOnce && window.verdant?.sense?.inspectCanvas) { sensedOnce = true; try { window.verdant.sense.inspectCanvas(canvas); } catch (_) {} }
}

renderChrome();
function loop(ts) {
  if (!lastTs) lastTs = ts;
  const dt = Math.min(0.1, (ts - lastTs) / 1000); lastTs = ts;
  if (playing) tSec = (tSec + dt) % getTotalDur();
  drawFrame(tSec);
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
`;
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

  // Single serene top bar: chapter pills on left, Write/Read + Export menu on right
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

  // Borderless on-page editorial heading + subtle subtitle/premise
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

  // Quiet editorial footer
  const foot = document.createElement('footer'); foot.className = 'doc-footer';
  const meta = document.createElement('div'); meta.id = 'doc-footer-meta';
  const status = document.createElement('span'); status.id = 'doc-save-status'; status.textContent = saveStatus;
  foot.append(meta, status);
  root.appendChild(foot);
  updateFooter();
}

// Accept live natural-language directives from the Prism's universal Natural Language bar
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
let hypotheses = W.hypotheses.map((h) => ({ ...h }));
let milestones = W.milestones.map((m) => ({ ...m }));
let notes = [];

function persist() { try { window.verdant?.storage?.set('research', { hypotheses, milestones, notes }); } catch (_) {} }
async function hydrate() {
  try {
    const saved = await window.verdant?.storage?.get('research');
    if (saved && Array.isArray(saved.hypotheses)) { hypotheses = saved.hypotheses; milestones = saved.milestones || milestones; notes = saved.notes || []; render(); }
  } catch (_) {}
}
function exportMarkdown() {
  const md = [
    '# ' + W.topic, '', '> ' + W.question, '',
    '## Hypotheses', ...hypotheses.map((h) => '- **' + h.id + '** [' + h.status + ']: ' + h.statement), '',
    '## Evidence & Experiment Log', ...(notes.length ? notes.map((n) => '- [' + n.verdict + '] (' + n.hypothesisId + ') ' + n.text + ' — _' + n.date + '_') : ['- No entries logged yet.']), '',
    '## Milestones', ...milestones.map((m) => '- [' + (m.status === 'done' ? 'x' : ' ') + '] ' + m.title + ' (' + m.dueLabel + ' · ' + m.status + ')')
  ].join('\\n');
  const dl = window.verdant?.browser?.downloadFile || window.verdant?.downloadFile;
  dl?.((W.topic || 'research').toLowerCase().replace(/[^a-z0-9]+/g, '-') + '-report.md', md, 'text/markdown');
}

function render() {
  root.innerHTML = '';
  const doneCount = milestones.filter((m) => m.status === 'done').length;
  const head = document.createElement('div'); head.className = 'organ-row'; head.style.justifyContent = 'space-between'; head.style.marginBottom = '12px';
  head.innerHTML = '<div class="organ-row">' + W.tracks.map((t) => '<span class="organ-pill">' + t + '</span>').join('') + '<span class="organ-pill">Progress: ' + doneCount + '/' + milestones.length + '</span></div>';
  const expRow = document.createElement('div'); expRow.className = 'organ-row';
  if (window.verdant?.ask) {
    const synthBtn = document.createElement('button'); synthBtn.className = 'organ-btn'; synthBtn.textContent = 'Synthesize Findings';
    synthBtn.onclick = async () => {
      synthBtn.disabled = true; synthBtn.textContent = 'Synthesizing…';
      try {
        const prompt = 'Synthesize concise research findings for "' + W.topic + '" (' + W.question + '). Hypotheses: ' + hypotheses.map((h) => h.id + ' [' + h.status + ']: ' + h.statement).join('; ') + '. Evidence: ' + notes.map((n) => n.verdict + ': ' + n.text).join('; ');
        const reply = await window.verdant.ask(prompt);
        notes.unshift({ hypothesisId: 'Synthesis', verdict: 'observation', text: String(reply || '').trim(), date: new Date().toLocaleDateString() });
        persist(); render();
      } catch (_) { synthBtn.disabled = false; synthBtn.textContent = 'Synthesize Findings'; }
    };
    expRow.appendChild(synthBtn);
  }
  const mdBtn = document.createElement('button'); mdBtn.className = 'organ-btn primary'; mdBtn.textContent = 'Export Report MD'; mdBtn.onclick = exportMarkdown;
  const jsonBtn = document.createElement('button'); jsonBtn.className = 'organ-btn'; jsonBtn.textContent = 'Export JSON';
  jsonBtn.onclick = () => (window.verdant?.browser?.downloadFile || window.verdant?.downloadFile)?.('research-data.json', JSON.stringify({ topic: W.topic, question: W.question, hypotheses, notes, milestones }, null, 2), 'application/json');
  expRow.append(mdBtn, jsonBtn); head.appendChild(expRow); root.appendChild(head);

  const q = document.createElement('p'); q.style.margin = '0 0 14px'; q.style.fontWeight = '600'; q.textContent = W.question; root.appendChild(q);

  // Hypotheses + Add Form
  const hSec = document.createElement('div'); hSec.style.marginBottom = '16px';
  hSec.innerHTML = '<div class="organ-sub" style="margin-bottom:6px;font-weight:700">HYPOTHESES (click status to cycle)</div>';
  hypotheses.forEach((h, idx) => {
    const row = document.createElement('div'); row.className = 'organ-row'; row.style.justifyContent = 'space-between'; row.style.padding = '8px 10px'; row.style.border = '1px solid var(--v-border)'; row.style.borderRadius = '8px'; row.style.marginBottom = '6px';
    const txt = document.createElement('span'); txt.textContent = h.id + ': ' + h.statement;
    const ctrls = document.createElement('div'); ctrls.className = 'organ-row';
    const st = document.createElement('button'); st.className = 'organ-btn' + (h.status === 'supported' ? ' primary' : ''); st.textContent = h.status;
    st.onclick = () => { h.status = h.status === 'open' ? 'supported' : h.status === 'supported' ? 'refuted' : 'open'; persist(); render(); };
    const rm = document.createElement('button'); rm.className = 'organ-btn'; rm.textContent = '×'; rm.onclick = () => { hypotheses.splice(idx, 1); persist(); render(); };
    ctrls.append(st, rm); row.append(txt, ctrls); hSec.appendChild(row);
  });
  const addHRow = document.createElement('div'); addHRow.className = 'organ-row';
  const hIn = document.createElement('input'); hIn.className = 'organ-input'; hIn.style.flex = '1'; hIn.placeholder = 'Add a testable hypothesis…';
  const hBtn = document.createElement('button'); hBtn.className = 'organ-btn'; hBtn.textContent = '+ Hypothesis';
  hBtn.onclick = () => { if (!hIn.value.trim()) return; hypotheses.push({ id: 'H' + (hypotheses.length + 1), statement: hIn.value.trim(), status: 'open' }); persist(); render(); };
  addHRow.append(hIn, hBtn); hSec.appendChild(addHRow); root.appendChild(hSec);

  // Evidence & Experiment Log (wired!)
  const eSec = document.createElement('div'); eSec.style.marginBottom = '16px';
  eSec.innerHTML = '<div class="organ-sub" style="margin-bottom:6px;font-weight:700">EVIDENCE & EXPERIMENT LOG</div>';
  notes.forEach((n, idx) => {
    const row = document.createElement('div'); row.className = 'organ-row'; row.style.justifyContent = 'space-between'; row.style.padding = '8px 10px'; row.style.border = '1px solid var(--v-border)'; row.style.borderRadius = '8px'; row.style.marginBottom = '6px';
    const txt = document.createElement('span'); txt.textContent = '[' + n.verdict + ' · ' + n.hypothesisId + '] ' + n.text + ' (' + n.date + ')';
    const rm = document.createElement('button'); rm.className = 'organ-btn'; rm.textContent = '×'; rm.onclick = () => { notes.splice(idx, 1); persist(); render(); };
    row.append(txt, rm); eSec.appendChild(row);
  });
  const addERow = document.createElement('div'); addERow.className = 'organ-row';
  const eHyp = document.createElement('select'); eHyp.className = 'organ-input';
  (hypotheses.length ? hypotheses : [{ id: 'General' }]).forEach((h) => { const o = document.createElement('option'); o.value = h.id; o.textContent = h.id; eHyp.appendChild(o); });
  const eVerd = document.createElement('select'); eVerd.className = 'organ-input';
  for (const v of ['supporting','refuting','observation']) { const o = document.createElement('option'); o.value = v; o.textContent = v; eVerd.appendChild(o); }
  const eIn = document.createElement('input'); eIn.className = 'organ-input'; eIn.style.flex = '1'; eIn.placeholder = 'Log an experimental observation or citation…';
  const eBtn = document.createElement('button'); eBtn.className = 'organ-btn primary'; eBtn.textContent = '+ Log Evidence';
  eBtn.onclick = () => { if (!eIn.value.trim()) return; notes.push({ hypothesisId: eHyp.value, verdict: eVerd.value, text: eIn.value.trim(), date: new Date().toLocaleDateString() }); persist(); render(); };
  addERow.append(eHyp, eVerd, eIn, eBtn); eSec.appendChild(addERow); root.appendChild(eSec);

  // Milestones + Add Form
  const mSec = document.createElement('div');
  mSec.innerHTML = '<div class="organ-sub" style="margin-bottom:6px;font-weight:700">MILESTONES</div>';
  milestones.forEach((m) => {
    const row = document.createElement('div'); row.className = 'organ-row'; row.style.justifyContent = 'space-between'; row.style.padding = '8px 10px'; row.style.border = '1px solid var(--v-border)'; row.style.borderRadius = '8px'; row.style.marginBottom = '6px';
    const txt = document.createElement('span'); txt.textContent = m.title + ' (' + m.dueLabel + ')';
    const st = document.createElement('button'); st.className = 'organ-btn' + (m.status === 'done' ? ' primary' : ''); st.textContent = m.status;
    st.onclick = () => { m.status = m.status === 'planned' ? 'in-progress' : m.status === 'in-progress' ? 'done' : 'planned'; persist(); render(); };
    row.append(txt, st); mSec.appendChild(row);
  });
  const addMRow = document.createElement('div'); addMRow.className = 'organ-row';
  const mIn = document.createElement('input'); mIn.className = 'organ-input'; mIn.style.flex = '1'; mIn.placeholder = 'Add milestone…';
  const mBtn = document.createElement('button'); mBtn.className = 'organ-btn'; mBtn.textContent = '+ Milestone';
  mBtn.onclick = () => { if (!mIn.value.trim()) return; milestones.push({ title: mIn.value.trim(), status: 'planned', dueLabel: 'Week ' + (milestones.length + 1) }); persist(); render(); };
  addMRow.append(mIn, mBtn); mSec.appendChild(addMRow); root.appendChild(mSec);
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
  windows: W.scheduleWindows.map((w) => ({ ...w })),
  checkins: 0,
  streakDays: Array(14).fill(false),
  metricValues: W.metrics.map(() => 0),
  steps: [
    { id: 's1', text: W.rulesSummary || 'Review objectives', done: false },
    { id: 's2', text: 'Complete primary action', done: false },
    { id: 's3', text: 'Verify & log completion', done: false },
  ],
  calcInputs: W.metrics.map((m) => m.target || 10),
  scenarios: [],
};

function persist() { try { window.verdant?.storage?.set('utilityState', state); } catch (_) {} }
async function hydrate() {
  try { const s = await window.verdant?.storage?.get('utilityState'); if (s && typeof s === 'object') { state = { ...state, ...s }; render(); } } catch (_) {}
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

function render() {
  root.innerHTML = '';
  const summary = document.createElement('p'); summary.style.margin = '0 0 12px'; summary.textContent = W.rulesSummary; root.appendChild(summary);

  if (W.role === 'calculator') {
    const grid = document.createElement('div'); grid.style.display = 'grid'; grid.style.gap = '10px'; grid.style.marginBottom = '12px';
    W.metrics.forEach((m, i) => {
      const row = document.createElement('div'); row.className = 'organ-row'; row.style.justifyContent = 'space-between';
      const lbl = document.createElement('label'); lbl.textContent = m.label + ' (' + m.unit + ')';
      const inp = document.createElement('input'); inp.type = 'number'; inp.className = 'organ-input'; inp.value = String(state.calcInputs[i] ?? m.target);
      inp.oninput = () => { state.calcInputs[i] = Number(inp.value) || 0; persist(); updateCalc(); };
      row.append(lbl, inp); grid.appendChild(row);
    });
    root.appendChild(grid);
    const outPill = document.createElement('div'); outPill.id = 'calc-out'; outPill.className = 'organ-pill'; outPill.style.marginBottom = '10px'; root.appendChild(outPill);
    function updateCalc() {
      const sum = state.calcInputs.reduce((a, b) => a + Number(b || 0), 0);
      const avg = state.calcInputs.length ? (sum / state.calcInputs.length).toFixed(2) : '0';
      outPill.textContent = 'Total: ' + sum.toLocaleString() + ' · Average: ' + avg;
    }
    updateCalc();
    const saveBtn = document.createElement('button'); saveBtn.className = 'organ-btn primary'; saveBtn.textContent = 'Save Scenario';
    saveBtn.onclick = () => {
      const sum = state.calcInputs.reduce((a, b) => a + Number(b || 0), 0);
      state.scenarios.unshift({ values: [...state.calcInputs], total: sum, at: new Date().toLocaleTimeString() });
      state.scenarios = state.scenarios.slice(0, 8); persist(); render();
    };
    root.appendChild(saveBtn);
    if (state.scenarios.length > 0) {
      const list = document.createElement('div'); list.style.marginTop = '12px';
      state.scenarios.forEach((sc) => {
        const r = document.createElement('div'); r.className = 'organ-sub'; r.style.padding = '4px 0';
        r.textContent = sc.at + ' · [' + sc.values.join(', ') + '] → Total ' + sc.total.toLocaleString();
        list.appendChild(r);
      });
      root.appendChild(list);
    }
    return;
  }

  if (W.role === 'workflow-helper') {
    const done = state.steps.filter((s) => s.done).length;
    const top = document.createElement('div'); top.className = 'organ-row'; top.style.justifyContent = 'space-between'; top.style.marginBottom = '12px';
    top.innerHTML = '<span class="organ-pill">Completed: ' + done + ' / ' + state.steps.length + '</span>';
    const exp = document.createElement('button'); exp.className = 'organ-btn'; exp.textContent = 'Export Markdown';
    exp.onclick = () => (window.verdant?.browser?.downloadFile || window.verdant?.downloadFile)?.('workflow-checklist.md', state.steps.map((s) => '- [' + (s.done ? 'x' : ' ') + '] ' + s.text).join('\\n'), 'text/markdown');
    top.appendChild(exp); root.appendChild(top);

    state.steps.forEach((st, idx) => {
      const row = document.createElement('div'); row.className = 'organ-row'; row.style.justifyContent = 'space-between'; row.style.padding = '8px 10px'; row.style.border = '1px solid var(--v-border)'; row.style.borderRadius = '8px'; row.style.marginBottom = '6px';
      const chk = document.createElement('button'); chk.className = 'organ-btn' + (st.done ? ' primary' : ''); chk.textContent = st.done ? '✓' : '○';
      chk.onclick = () => { st.done = !st.done; persist(); render(); };
      const lbl = document.createElement('span'); lbl.style.flex = '1'; lbl.style.textDecoration = st.done ? 'line-through' : 'none'; lbl.textContent = st.text;
      const rm = document.createElement('button'); rm.className = 'organ-btn'; rm.textContent = '×'; rm.onclick = () => { state.steps.splice(idx, 1); persist(); render(); };
      row.append(chk, lbl, rm); root.appendChild(row);
    });
    const addRow = document.createElement('div'); addRow.className = 'organ-row'; addRow.style.marginTop = '8px';
    const inp = document.createElement('input'); inp.className = 'organ-input'; inp.style.flex = '1'; inp.placeholder = 'Add workflow step…';
    const btn = document.createElement('button'); btn.className = 'organ-btn primary'; btn.textContent = '+ Add Step';
    btn.onclick = () => { if (!inp.value.trim()) return; state.steps.push({ id: 's' + Date.now(), text: inp.value.trim(), done: false }); persist(); render(); };
    addRow.append(inp, btn); root.appendChild(addRow);
    return;
  }

  // habit-guardian & schedule-guard: persistent check-ins, 14-day streak grid, 24h timeline + editable windows
  const metricRow = document.createElement('div'); metricRow.className = 'organ-row'; metricRow.style.justifyContent = 'space-between'; metricRow.style.marginBottom = '12px';
  const checkBtn = document.createElement('button'); checkBtn.className = 'organ-btn primary'; checkBtn.textContent = 'Log Today (' + state.checkins + ')';
  checkBtn.onclick = () => { state.checkins += 1; state.streakDays[state.streakDays.length - 1] = true; emitConstraints(); persist(); render(); };
  const expBtn = document.createElement('button'); expBtn.className = 'organ-btn'; expBtn.textContent = 'Export JSON';
  expBtn.onclick = () => (window.verdant?.browser?.downloadFile || window.verdant?.downloadFile)?.('habit-schedule-log.json', JSON.stringify(state, null, 2), 'application/json');
  metricRow.append(checkBtn, expBtn); root.appendChild(metricRow);

  // 14-day streak heatmap
  const heatLbl = document.createElement('div'); heatLbl.className = 'organ-sub'; heatLbl.style.marginBottom = '6px'; heatLbl.style.fontWeight = '700'; heatLbl.textContent = '14-DAY STREAK (click to toggle)';
  root.appendChild(heatLbl);
  const heat = document.createElement('div'); heat.style.display = 'grid'; heat.style.gridTemplateColumns = 'repeat(14, 1fr)'; heat.style.gap = '4px'; heat.style.marginBottom = '14px';
  state.streakDays.forEach((on, i) => {
    const cell = document.createElement('button'); cell.type = 'button'; cell.style.height = '24px'; cell.style.borderRadius = '6px'; cell.style.border = '1px solid var(--v-border)'; cell.style.cursor = 'pointer';
    cell.style.background = on ? 'var(--v-accent)' : 'var(--v-surface-alt)';
    cell.onclick = () => { state.streakDays[i] = !state.streakDays[i]; persist(); render(); };
    heat.appendChild(cell);
  });
  root.appendChild(heat);

  // Protected schedule windows + 24h timeline bar
  const label = document.createElement('div'); label.className = 'organ-sub'; label.style.marginBottom = '6px'; label.style.fontWeight = '700'; label.textContent = 'PROTECTED SCHEDULE WINDOWS';
  root.appendChild(label);
  const bar24 = document.createElement('div'); bar24.style.position = 'relative'; bar24.style.height = '18px'; bar24.style.borderRadius = '999px'; bar24.style.background = 'var(--v-surface-alt)'; bar24.style.border = '1px solid var(--v-border)'; bar24.style.marginBottom = '10px'; bar24.style.overflow = 'hidden';
  state.windows.filter((w) => w.protectedWindow).forEach((w) => {
    const seg = document.createElement('div'); seg.style.position = 'absolute'; seg.style.top = '0'; seg.style.bottom = '0';
    seg.style.left = (w.startMinuteOfDay / 1440 * 100) + '%'; seg.style.width = (Math.max(30, w.endMinuteOfDay - w.startMinuteOfDay) / 1440 * 100) + '%';
    seg.style.background = 'var(--v-accent)'; bar24.appendChild(seg);
  });
  root.appendChild(bar24);

  state.windows.forEach((w, idx) => {
    const row = document.createElement('div'); row.className = 'organ-row'; row.style.justifyContent = 'space-between'; row.style.padding = '8px 10px'; row.style.border = '1px solid var(--v-border)'; row.style.borderRadius = '8px'; row.style.marginBottom = '6px';
    const txt = document.createElement('span'); txt.textContent = w.label + ' · ' + fmtMin(w.startMinuteOfDay) + ' – ' + fmtMin(w.endMinuteOfDay);
    const ctrls = document.createElement('div'); ctrls.className = 'organ-row';
    const toggle = document.createElement('button'); toggle.className = 'organ-btn' + (w.protectedWindow ? ' primary' : ''); toggle.textContent = w.protectedWindow ? 'Protected' : 'Paused';
    toggle.onclick = () => { w.protectedWindow = !w.protectedWindow; emitConstraints(); persist(); render(); };
    const rm = document.createElement('button'); rm.className = 'organ-btn'; rm.textContent = '×'; rm.onclick = () => { state.windows.splice(idx, 1); emitConstraints(); persist(); render(); };
    ctrls.append(toggle, rm); row.append(txt, ctrls); root.appendChild(row);
  });

  // Add Window + Live Slot Conflict Checker
  const addWinRow = document.createElement('div'); addWinRow.className = 'organ-row'; addWinRow.style.marginTop = '8px';
  const wLbl = document.createElement('input'); wLbl.className = 'organ-input'; wLbl.style.flex = '1'; wLbl.placeholder = 'New protected block (e.g. Deep Focus)';
  const wStart = document.createElement('input'); wStart.type = 'time'; wStart.className = 'organ-input'; wStart.value = '09:00';
  const wEnd = document.createElement('input'); wEnd.type = 'time'; wEnd.className = 'organ-input'; wEnd.value = '10:30';
  const toMin = (hhmm) => { const [h, m] = String(hhmm || '09:00').split(':').map(Number); return (h || 0) * 60 + (m || 0); };
  const addWinBtn = document.createElement('button'); addWinBtn.className = 'organ-btn primary'; addWinBtn.textContent = '+ Add Block';
  addWinBtn.onclick = () => {
    const s = toMin(wStart.value), e = toMin(wEnd.value);
    if (e <= s) return;
    state.windows.push({ label: wLbl.value.trim() || 'Protected Block', startMinuteOfDay: s, endMinuteOfDay: e, daysOfWeek: [0,1,2,3,4,5,6], protectedWindow: true });
    emitConstraints(); persist(); render();
  };
  const chkBtn = document.createElement('button'); chkBtn.className = 'organ-btn'; chkBtn.textContent = 'Check Conflict';
  const chkOut = document.createElement('span'); chkOut.className = 'organ-sub';
  chkBtn.onclick = () => {
    const s = toMin(wStart.value), e = toMin(wEnd.value);
    const hits = state.windows.filter((w) => w.protectedWindow && s < w.endMinuteOfDay && e > w.startMinuteOfDay);
    chkOut.textContent = hits.length > 0 ? 'Conflicts with ' + hits.length + ' protected block(s)' : 'Slot is open ✓';
  };
  addWinRow.append(wLbl, wStart, wEnd, addWinBtn, chkBtn, chkOut);
  root.appendChild(addWinRow);
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
render();
`;
}
