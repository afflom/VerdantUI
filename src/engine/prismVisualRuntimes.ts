/**
 * Visual Procedural Media Runtimes (`src/engine/prismVisualRuntimes.ts`)
 *
 * Supplies audited procedural runtimes for Picture and Video Prisms:
 *  - Zero-idle-CPU visibility-aware RAF loop on Video (cancels RAF when paused or hidden)
 *  - Deterministic Web Audio teardown on WebM recording stop & pagehide
 *  - Uncluttered full-bleed stages with natural-language `window.verdant.onInput`
 */

import type { PictureWeights, VideoWeights } from './prismMediaOrgans';

function safeJsonForScript(obj: unknown): string {
  return JSON.stringify(obj).replace(/<\//g, '<\\/');
}

export function buildPictureRuntimeJs(weights: PictureWeights): string {
  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
let seed = W.seed || 4219, symmetry = W.symmetry || 6, density = W.density || 58;
let generator = W.generator === 'botanical-fractal' ? 'botanical-garden'
  : W.generator === 'wave-field' ? 'ocean-seascape'
  : W.generator === 'constellation' ? 'constellation-atlas'
  : W.generator === 'pixel-sprite' ? 'pixel-scene'
  : W.generator || 'landscape-vista';
let lighting = W.lighting || 'golden-hour';
let curW = W.width || 800, curH = W.height || 500, hiDpi = 1;
let useReconciledImage = typeof W.reconciledImageDataUrl === 'string' && W.reconciledImageDataUrl.startsWith('data:image/');
let drawGeneration = 0;

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
  const genToken = ++drawGeneration;
  ctx.clearRect(0, 0, w, h);

  if (useReconciledImage && generator !== 'logo-mark' && W.reconciledImageDataUrl) {
    ctx.fillStyle = cBg;
    ctx.fillRect(0, 0, w, h);
    const img = new Image();
    img.onload = () => {
      if (genToken !== drawGeneration || !useReconciledImage) return;
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(img, 0, 0, w, h);
      if (window.verdant?.sense?.inspectCanvas) {
        try { window.verdant.sense.inspectCanvas(canvas); } catch (_) {}
      }
    };
    img.onerror = () => {
      if (genToken !== drawGeneration) return;
      useReconciledImage = false;
      drawArtwork(canvas);
    };
    img.src = W.reconciledImageDataUrl;
    return;
  }

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
    ctx.fillStyle = cBg;
    for (let i = 0; i < Math.max(28, density); i++) {
      const side = i % 2 === 0 ? rng() * 0.34 : 0.66 + rng() * 0.34;
      const tx = side * w, th = h * (0.04 + rng() * 0.09);
      ctx.beginPath(); ctx.moveTo(tx, horizon - th); ctx.lineTo(tx - th * 0.26, horizon); ctx.lineTo(tx + th * 0.26, horizon); ctx.closePath(); ctx.fill();
    }
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
  genBtn.onclick = () => { useReconciledImage = false; seed = (seed * 1664525 + 1013904223) % 999999; drawArtwork(canvas); };
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
    if (/\\b(variation|remix|regenerate)\\b/.test(text)) { useReconciledImage = false; seed = (seed * 1664525 + 1013904223) % 999999; }
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

export function buildVideoRuntimeJs(weights: VideoWeights): string {
  return `
const W = ${safeJsonForScript(weights)};
const root = document.getElementById('organ-root');
const dims = W.aspectRatio === '1:1' ? [540, 540] : W.aspectRatio === '9:16' ? [360, 640] : [640, 360];
let scenes = W.scenes.map((s) => ({ ...s }));
function getTotalDur() { return scenes.reduce((a, s) => a + Math.max(1, s.durationSec), 0) || 6; }
let playing = true, tSec = 0, lastTs = 0, selectedIdx = 0, rafId = null;
let recorder = null, recChunks = [], recording = false, recTimer = null, recAudioCtx = null;
window.__verdantVideoRafCount = 0;

function stopLoop() {
  if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
  lastTs = 0;
}
function startLoop() {
  if (!playing || document.hidden || rafId !== null) return;
  lastTs = 0;
  rafId = requestAnimationFrame(loop);
}

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
  playBtn.onclick = () => {
    playing = !playing;
    playBtn.textContent = playing ? 'Pause' : 'Play';
    if (playing) startLoop(); else stopLoop();
  };
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
  drawFrame(tSec);
}

function closeRecAudio() {
  if (recAudioCtx) {
    try { void recAudioCtx.close(); } catch (_) {}
    recAudioCtx = null;
  }
}

function stopRecording(recBtn) {
  if (recTimer) { clearTimeout(recTimer); recTimer = null; }
  if (recorder && recorder.state !== 'inactive') { try { recorder.stop(); } catch (_) {} }
  closeRecAudio();
  recording = false;
  if (recBtn) { recBtn.textContent = 'Record WebM'; recBtn.className = 'organ-btn'; }
}

function toggleRecord(canvas, recBtn) {
  if (recording) { stopRecording(recBtn); return; }
  if (typeof canvas.captureStream !== 'function' || typeof window.MediaRecorder !== 'function') { recBtn.textContent = 'WebM Unavailable'; return; }
  try {
    const vStream = canvas.captureStream(Math.max(15, Math.min(60, W.fps || 30)));
    const tracks = [...vStream.getVideoTracks()];
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (AC) {
        closeRecAudio();
        const ac = new AC();
        recAudioCtx = ac;
        const dest = ac.createMediaStreamDestination(), filter = ac.createBiquadFilter();
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
        tracks.push(...dest.stream.getAudioTracks());
      }
    } catch (_) {}
    const muxStream = new MediaStream(tracks);
    const mime = MediaRecorder.isTypeSupported('video/webm;codecs=vp9,opus') ? 'video/webm;codecs=vp9,opus' : MediaRecorder.isTypeSupported('video/webm') ? 'video/webm' : '';
    recorder = mime ? new MediaRecorder(muxStream, { mimeType: mime }) : new MediaRecorder(muxStream);
    recChunks = [];
    recorder.ondataavailable = (ev) => { if (ev.data && ev.data.size > 0) recChunks.push(ev.data); };
    recorder.onstop = () => {
      closeRecAudio();
      if (recChunks.length === 0) return;
      const url = URL.createObjectURL(new Blob(recChunks, { type: mime || 'video/webm' }));
      const a = document.createElement('a'); a.href = url; a.download = 'verdant-video.webm'; a.click();
      setTimeout(() => URL.revokeObjectURL(url), 4000);
    };
    tSec = 0; playing = true; recording = true; startLoop();
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

function loop(ts) {
  rafId = null;
  if (!playing || document.hidden) return;
  window.__verdantVideoRafCount += 1;
  if (!lastTs) lastTs = ts;
  const dt = Math.min(0.1, (ts - lastTs) / 1000); lastTs = ts;
  tSec = (tSec + dt) % getTotalDur();
  drawFrame(tSec);
  rafId = requestAnimationFrame(loop);
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden) stopLoop();
  else if (playing) startLoop();
});
window.addEventListener('pagehide', () => { stopLoop(); stopRecording(null); });

renderChrome();
if (window.verdant?.onInput) {
  window.verdant.onInput((input) => {
    const prompt = String(input?.prompt || input?.text || input || '').trim();
    const low = prompt.toLowerCase();
    if (/\\bpause\\b|\\bstop\\b/.test(low)) { playing = false; stopLoop(); renderChrome(); return { playing, scenes: scenes.length }; }
    if (/\\bplay\\b|\\bresume\\b/.test(low)) { playing = true; renderChrome(); startLoop(); return { playing, scenes: scenes.length }; }
    const scMatch = low.match(/\\bscene\\s+(\\d+)\\b/);
    if (scMatch) {
      const idx = Math.max(0, Math.min(scenes.length - 1, Number(scMatch[1]) - 1));
      selectedIdx = idx;
      let startAt = 0;
      for (let i = 0; i < idx; i++) startAt += scenes[i].durationSec;
      tSec = startAt + 0.05;
      renderChrome();
      return { playing, selectedScene: idx + 1 };
    }
    if (/\\badd\\s+scene\\b/.test(low)) {
      const title = prompt.replace(/^.*\\badd\\s+scene\\b[:\\s-]*/i, '').trim() || ('Scene ' + (scenes.length + 1));
      scenes.push({ title: title.slice(0, 42), subtitle: 'Added via natural language', durationSec: 4, accent: W.palette[2] || '#34d399', motion: 'rise' });
      selectedIdx = scenes.length - 1;
      renderChrome();
      return { playing, scenes: scenes.length };
    }
    if (/\\bmidnight\\b|\\bdark\\b|\\bnocturnal\\b/.test(low)) { W.palette = ['#020617', '#0f172a', '#60a5fa']; }
    else if (/\\bgolden\\b|\\bwarm\\b|\\bsunset\\b/.test(low)) { W.palette = ['#1c1917', '#451a03', '#f59e0b']; }
    else if (/\\bemerald\\b|\\bforest\\b|\\bverdant\\b/.test(low)) { W.palette = ['#022c22', '#064e3b', '#34d399']; }
    if (/\\bslower\\b|\\bcalm\\b/.test(low)) { scenes = scenes.map((s) => ({ ...s, durationSec: Math.min(10, s.durationSec + 1.5) })); }
    else if (/\\bfaster\\b|\\benergetic\\b/.test(low)) { scenes = scenes.map((s) => ({ ...s, durationSec: Math.max(2, s.durationSec - 1) })); }
    renderChrome();
    return { playing, scenes: scenes.length, palette: W.palette };
  });
}
startLoop();
`;
}
