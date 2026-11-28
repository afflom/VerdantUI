/**
 * Audio Procedural Media Runtime (`src/engine/prismAudioRuntimes.ts`)
 *
 * Supplies the audited stereo Web Audio + WAV + MIDI 1.0 Audio Prism runtime with
 * clean AudioContext suspend/close teardown so repeated sessions never leak handles.
 */

import type { AudioWeights } from './prismMediaOrgans';

function safeJsonForScript(obj: unknown): string {
  return JSON.stringify(obj).replace(/<\//g, '<\\/');
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
  if (!ctx || ctx.state === 'closed') {
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

function disposeAudioContext() {
  if (ctx && ctx.state !== 'closed') {
    try { void ctx.close(); } catch (_) {}
  }
  ctx = null;
  masterFilter = null;
  analyser = null;
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
  if (playing && !document.hidden) rafId = requestAnimationFrame(drawLiveAnalyser);
  else rafId = null;
}

function stopPlay() {
  playing = false;
  if (timer) { clearInterval(timer); timer = null; }
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  if (ctx && ctx.state === 'running') { try { void ctx.suspend(); } catch (_) {} }
  render();
}
function startPlay() {
  if (!ensureAudio()) return;
  playing = true; playStep();
  timer = setInterval(playStep, Math.max(115, Math.round((60 / bpm) * 500)));
  render(); drawLiveAnalyser();
}

window.addEventListener('pagehide', () => { stopPlay(); disposeAudioContext(); });
window.addEventListener('beforeunload', () => { disposeAudioContext(); });

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
    if (tracks.rhythm) {
      if (st % 4 === 0) addNote(t0, 0.2, 62, 0.34, 0, [{ ratio: 1, amp: 1, damp: 12 }]);
      if (st % 4 === 2) {
        const s0 = Math.max(0, Math.floor(t0 * sr)), s1 = Math.min(total, Math.floor((t0 + 0.12) * sr));
        let seedN = (st * 2654435761) >>> 0;
        for (let i = s0; i < s1; i++) {
          seedN = (Math.imul(seedN, 1664525) + 1013904223) >>> 0;
          const n = ((seedN / 4294967296) * 2 - 1) * Math.exp(-(i - s0) / (sr * 0.032)) * 0.16;
          left[i] += n; right[i] += n * 0.94;
        }
      }
      const h0 = Math.max(0, Math.floor(t0 * sr)), h1 = Math.min(total, Math.floor((t0 + 0.04) * sr));
      let seedH = ((st + 17) * 2246822519) >>> 0;
      for (let i = h0; i < h1; i++) {
        seedH = (Math.imul(seedH, 1664525) + 1013904223) >>> 0;
        const n = ((seedH / 4294967296) * 2 - 1) * Math.exp(-(i - h0) / (sr * 0.012)) * 0.065;
        left[i] += n * 0.85; right[i] += n * 1.05;
      }
    }
  }
  // Gentle cross-channel stereo reverb + EBU R128 / peak-safe mastering
  const delayL = Math.floor(sr * 0.071), delayR = Math.floor(sr * 0.097);
  for (let i = Math.max(delayL, delayR); i < total; i++) {
    left[i] += right[i - delayL] * 0.16;
    right[i] += left[i - delayR] * 0.16;
  }
  let sumSq = 0, peak = 0;
  for (let i = 0; i < total; i++) {
    const al = Math.abs(left[i]), ar = Math.abs(right[i]);
    if (al > peak) peak = al; if (ar > peak) peak = ar;
    sumSq += left[i] * left[i] + right[i] * right[i];
  }
  const rms = Math.sqrt(sumSq / Math.max(1, total * 2));
  const targetGain = rms > 1e-4 ? Math.min(2.2, 0.18 / rms) : 1;
  const gain = peak * targetGain > 0.92 ? 0.92 / Math.max(1e-4, peak) : targetGain;
  const dataBytes = total * 4, buf = new ArrayBuffer(44 + dataBytes), dv = new DataView(buf);
  const ws = (off, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i)); };
  ws(0, 'RIFF'); dv.setUint32(4, 36 + dataBytes, true); ws(8, 'WAVE'); ws(12, 'fmt '); dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true); dv.setUint16(22, 2, true); dv.setUint32(24, sr, true); dv.setUint32(28, sr * 4, true);
  dv.setUint16(32, 4, true); dv.setUint16(34, 16, true); ws(36, 'data'); dv.setUint32(40, dataBytes, true);
  let p = 44;
  for (let i = 0; i < total; i++) {
    dv.setInt16(p, Math.max(-32767, Math.min(32767, Math.round(Math.tanh(left[i] * gain) * 32767))), true); p += 2;
    dv.setInt16(p, Math.max(-32767, Math.min(32767, Math.round(Math.tanh(right[i] * gain) * 32767))), true); p += 2;
  }
  const url = URL.createObjectURL(new Blob([buf], { type: 'audio/wav' }));
  const a = document.createElement('a'); a.href = url; a.download = (W.title || 'verdant-audio').toLowerCase().replace(/[^a-z0-9]+/g, '-') + '.wav'; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
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
    if (tracks.bass && st % 2 === 0) {
      const bn = Math.max(0, Math.min(127, W.rootMidi - 24 + scale[chordDeg % scale.length] + (st % 4 === 2 ? 7 : 0)));
      msgs.push([t, [0x92, bn, 76]], [t + ppqn, [0x82, bn, 0]]);
    }
    if (tracks.lead && (W.style !== 'ambient-pad' || st % 2 === 0)) {
      const n = Math.max(0, Math.min(127, W.rootMidi + scale[(chordDeg + (st % 5) * 2) % scale.length]));
      msgs.push([t, [0x91, n, 82]], [t + (ppqn >> 1), [0x81, n, 0]]);
    }
    if (tracks.rhythm) {
      if (st % 4 === 0) msgs.push([t, [0x99, 36, 96]], [t + 120, [0x89, 36, 0]]);
      if (st % 4 === 2) msgs.push([t, [0x99, 38, 86]], [t + 120, [0x89, 38, 0]]);
      msgs.push([t, [0x99, 42, 62]], [t + 80, [0x89, 42, 0]]);
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
  const url = URL.createObjectURL(new Blob([out], { type: 'audio/midi' }));
  const a = document.createElement('a'); a.href = url; a.download = (W.title || 'verdant-audio').toLowerCase().replace(/[^a-z0-9]+/g, '-') + '.mid'; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
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

  const stems = document.createElement('div'); stems.className = 'organ-row'; stems.style.marginTop = '10px'; stems.style.gap = '6px';
  for (const [key, label] of [['pad', 'Pad'], ['bass', 'Bass'], ['lead', 'Lead'], ['rhythm', 'Drums']]) {
    const tb = document.createElement('button'); tb.type = 'button';
    tb.className = 'organ-btn ' + (tracks[key] ? 'primary' : 'ghost');
    tb.style.fontSize = '11px'; tb.style.padding = '4px 10px';
    tb.textContent = label;
    tb.onclick = () => { tracks[key] = !tracks[key]; render(); };
    stems.appendChild(tb);
  }
  root.appendChild(stems);

  const foot = document.createElement('div'); foot.className = 'doc-footer'; foot.style.marginTop = '12px';
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
