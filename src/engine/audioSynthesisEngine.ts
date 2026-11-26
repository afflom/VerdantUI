/**
 * Multi-Track Modal Audio Synthesis Engine (`src/engine/audioSynthesisEngine.ts`)
 *
 * Grounded in `verdant-engine`'s `ModalAudioMixMatrix` (damped harmonic modes, stereo panning,
 * and 4-channel Hadamard Feedback Delay Network reverberation):
 * - Synthesizes coherent polyphonic multi-track compositions (melody, chord pad, bass, pulse)
 *   into non-clipping stereo PCM and valid 16-bit stereo `RIFF`/`WAVE` buffers.
 */

export type MusicalScaleName = 'pentatonic' | 'major' | 'minor' | 'dorian';
export type AudioMoodStyle = 'ambient-pad' | 'melodic-arpeggio' | 'rhythm-pulse' | 'sound-effects';

export interface MusicalNoteEvent {
  readonly midi: number;
  readonly startBeat: number;
  readonly durationBeats: number;
  readonly velocity: number; // 0..1
  readonly pan: number; // -1 (left) .. +1 (right)
  readonly timbre: 'modal-bell' | 'warm-pad' | 'plucked-bass' | 'soft-pulse';
}

export interface MusicalComposition {
  readonly title: string;
  readonly bpm: number;
  readonly rootMidi: number;
  readonly scale: MusicalScaleName;
  readonly durationSec: number;
  readonly events: readonly MusicalNoteEvent[];
}

const SCALE_INTERVALS: Readonly<Record<MusicalScaleName, readonly number[]>> = {
  pentatonic: [0, 2, 4, 7, 9, 12],
  major: [0, 2, 4, 5, 7, 9, 11, 12],
  minor: [0, 2, 3, 5, 7, 8, 10, 12],
  dorian: [0, 2, 3, 5, 7, 9, 10, 12],
};

function midiToHz(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

/**
 * Generates a coherent multi-track polyphonic arrangement (harmony pad + bass + melody + pulse).
 */
export function composeMultiTrackArrangement(params: {
  readonly title: string;
  readonly style: AudioMoodStyle;
  readonly bpm: number;
  readonly rootMidi: number;
  readonly scale: MusicalScaleName;
  readonly durationSec: number;
}): MusicalComposition {
  const bpm = Math.max(50, Math.min(180, Math.round(params.bpm)));
  const durationSec = Math.max(2, Math.min(20, Math.round(params.durationSec)));
  const totalBeats = Math.max(4, Math.floor((durationSec * bpm) / 60));
  const scale = SCALE_INTERVALS[params.scale] ?? SCALE_INTERVALS.pentatonic;
  const events: MusicalNoteEvent[] = [];

  // Track 1: Warm harmonic triad chords every 4 beats
  const chordRoots = [0, 3, 4, 2];
  for (let beat = 0; beat < totalBeats; beat += 4) {
    const barIdx = Math.floor(beat / 4) % chordRoots.length;
    const deg = chordRoots[barIdx]!;
    const triad = [
      params.rootMidi + scale[deg % scale.length]!,
      params.rootMidi + scale[(deg + 2) % scale.length]!,
      params.rootMidi + scale[(deg + 4) % scale.length]!,
    ];
    triad.forEach((midi, vi) => {
      events.push({
        midi,
        startBeat: beat,
        durationBeats: Math.min(3.8, totalBeats - beat),
        velocity: params.style === 'ambient-pad' ? 0.34 : 0.24,
        pan: vi === 0 ? -0.35 : vi === 2 ? 0.35 : 0,
        timbre: 'warm-pad',
      });
    });

    // Track 2: Grounded bass root on beat 0 & beat 2
    events.push({
      midi: params.rootMidi - 12 + scale[deg % scale.length]!,
      startBeat: beat,
      durationBeats: 1.9,
      velocity: 0.42,
      pan: 0,
      timbre: 'plucked-bass',
    });
    if (beat + 2 < totalBeats) {
      events.push({
        midi: params.rootMidi - 12 + scale[(deg + 2) % scale.length]!,
        startBeat: beat + 2,
        durationBeats: 1.8,
        velocity: 0.32,
        pan: 0,
        timbre: 'plucked-bass',
      });
    }
  }

  // Track 3: Melodic modal voice
  const stepBeats = params.style === 'ambient-pad' ? 1.0 : 0.5;
  const melodyPattern = [0, 2, 4, 3, 5, 4, 2, 1];
  let stepIdx = 0;
  for (let beat = 0; beat < totalBeats; beat += stepBeats) {
    const deg = melodyPattern[stepIdx % melodyPattern.length]!;
    const midi = params.rootMidi + 12 + scale[deg % scale.length]!;
    events.push({
      midi,
      startBeat: beat,
      durationBeats: stepBeats * (params.style === 'ambient-pad' ? 1.6 : 0.9),
      velocity: params.style === 'ambient-pad' ? 0.26 : 0.36,
      pan: Math.sin(stepIdx * 0.7) * 0.45,
      timbre: 'modal-bell',
    });
    if (params.style === 'rhythm-pulse' && stepIdx % 2 === 0) {
      events.push({
        midi: params.rootMidi - 24,
        startBeat: beat,
        durationBeats: 0.2,
        velocity: 0.35,
        pan: 0,
        timbre: 'soft-pulse',
      });
    }
    stepIdx += 1;
  }

  return {
    title: params.title,
    bpm,
    rootMidi: params.rootMidi,
    scale: params.scale,
    durationSec,
    events,
  };
}

/**
 * Synthesizes a `MusicalComposition` into stereo PCM using damped modal harmonic modes,
 * equal-power stereo panning, and a 4-channel Hadamard Feedback Delay Network reverb.
 */
export function synthesizeCompositionStereoPcm(
  comp: MusicalComposition,
  sampleRate = 22_050,
): { readonly left: Float32Array; readonly right: Float32Array; readonly sampleRate: number } {
  const totalSamples = Math.max(1024, Math.round(comp.durationSec * sampleRate));
  const dryL = new Float32Array(totalSamples);
  const dryR = new Float32Array(totalSamples);
  const secPerBeat = 60 / comp.bpm;

  // Modal harmonic ratios & decay factors per instrument timbre
  const TIMBRE_MODES: Readonly<
    Record<MusicalNoteEvent['timbre'], readonly { readonly ratio: number; readonly amp: number; readonly decay: number }[]>
  > = {
    'modal-bell': [
      { ratio: 1.0, amp: 1.0, decay: 2.2 },
      { ratio: 2.0, amp: 0.42, decay: 3.4 },
      { ratio: 2.76, amp: 0.18, decay: 4.8 },
    ],
    'warm-pad': [
      { ratio: 1.0, amp: 1.0, decay: 0.6 },
      { ratio: 2.0, amp: 0.28, decay: 1.1 },
      { ratio: 1.5, amp: 0.16, decay: 1.3 },
    ],
    'plucked-bass': [
      { ratio: 1.0, amp: 1.0, decay: 3.0 },
      { ratio: 2.0, amp: 0.48, decay: 4.5 },
      { ratio: 3.0, amp: 0.16, decay: 6.0 },
    ],
    'soft-pulse': [
      { ratio: 1.0, amp: 1.0, decay: 12.0 },
      { ratio: 1.41, amp: 0.3, decay: 18.0 },
    ],
  };

  for (const ev of comp.events) {
    const startIdx = Math.max(0, Math.round(ev.startBeat * secPerBeat * sampleRate));
    const durSamples = Math.max(64, Math.round(ev.durationBeats * secPerBeat * sampleRate));
    const endIdx = Math.min(totalSamples, startIdx + durSamples);
    if (startIdx >= totalSamples) continue;

    const baseHz = midiToHz(ev.midi);
    const modes = TIMBRE_MODES[ev.timbre];
    const angle = ((Math.max(-1, Math.min(1, ev.pan)) + 1) * Math.PI) / 4;
    const gainL = Math.cos(angle) * ev.velocity * 0.26;
    const gainR = Math.sin(angle) * ev.velocity * 0.26;
    const attackSamples = Math.max(16, Math.round((ev.timbre === 'warm-pad' ? 0.08 : 0.012) * sampleRate));
    const releaseSamples = Math.max(32, Math.round(0.06 * sampleRate));

    for (let i = startIdx; i < endIdx; i++) {
      const local = i - startIdx;
      const tSec = local / sampleRate;
      const atk = local < attackSamples ? local / attackSamples : 1.0;
      const rem = endIdx - i;
      const rel = rem < releaseSamples ? rem / releaseSamples : 1.0;
      const gate = atk * rel;

      let sample = 0.0;
      for (const m of modes) {
        sample += Math.sin(2.0 * Math.PI * baseHz * m.ratio * tSec) * m.amp * Math.exp(-m.decay * tSec);
      }
      const v = sample * gate;
      dryL[i] = dryL[i]! + v * gainL;
      dryR[i] = dryR[i]! + v * gainR;
    }
  }

  // 4-channel Hadamard FDN Reverb (matching verdant-engine ModalAudioMixMatrix)
  const delays = [
    Math.round(sampleRate * 0.0297),
    Math.round(sampleRate * 0.0371),
    Math.round(sampleRate * 0.0411),
    Math.round(sampleRate * 0.0437),
  ];
  const lines = delays.map((len) => new Float32Array(Math.max(16, len)));
  const cursors = [0, 0, 0, 0];
  const fb = 0.42;
  const wetMix = 0.22;

  const left = new Float32Array(totalSamples);
  const right = new Float32Array(totalSamples);
  for (let i = 0; i < totalSamples; i++) {
    const inL = dryL[i]!;
    const inR = dryR[i]!;
    const d0 = lines[0]![cursors[0]!]!;
    const d1 = lines[1]![cursors[1]!]!;
    const d2 = lines[2]![cursors[2]!]!;
    const d3 = lines[3]![cursors[3]!]!;

    // Normalized 4x4 Hadamard orthogonal mixing (* 0.5)
    const h0 = (d0 + d1 + d2 + d3) * 0.5;
    const h1 = (d0 - d1 + d2 - d3) * 0.5;
    const h2 = (d0 + d1 - d2 - d3) * 0.5;
    const h3 = (d0 - d1 - d2 + d3) * 0.5;

    lines[0]![cursors[0]!] = inL + h0 * fb;
    lines[1]![cursors[1]!] = inR + h1 * fb;
    lines[2]![cursors[2]!] = inL + h2 * fb;
    lines[3]![cursors[3]!] = inR + h3 * fb;

    for (let c = 0; c < 4; c++) {
      cursors[c] = (cursors[c]! + 1) % lines[c]!.length;
    }

    // Soft-clip limiter guarantees zero harsh digital clipping
    left[i] = Math.tanh(inL + (h0 + h2) * wetMix);
    right[i] = Math.tanh(inR + (h1 + h3) * wetMix);
  }

  return { left, right, sampleRate };
}

/** Encodes stereo F32 PCM into a standard 16-bit stereo `RIFF`/`WAVE` buffer. */
export function encodeStereoWav(
  left: Float32Array,
  right: Float32Array,
  sampleRate: number,
): Uint8Array {
  const frames = Math.min(left.length, right.length);
  const dataBytes = frames * 4; // 2 channels * 2 bytes
  const buf = new ArrayBuffer(44 + dataBytes);
  const dv = new DataView(buf);
  const writeStr = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i));
  };
  writeStr(0, 'RIFF');
  dv.setUint32(4, 36 + dataBytes, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true); // PCM
  dv.setUint16(22, 2, true); // stereo
  dv.setUint32(24, sampleRate, true);
  dv.setUint32(28, sampleRate * 4, true);
  dv.setUint16(32, 4, true);
  dv.setUint16(34, 16, true);
  writeStr(36, 'data');
  dv.setUint32(40, dataBytes, true);

  let off = 44;
  for (let i = 0; i < frames; i++) {
    const l = Math.max(-1, Math.min(1, left[i]!));
    const r = Math.max(-1, Math.min(1, right[i]!));
    dv.setInt16(off, Math.round(l * 32767), true);
    dv.setInt16(off + 2, Math.round(r * 32767), true);
    off += 4;
  }
  return new Uint8Array(buf);
}
