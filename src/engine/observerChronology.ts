/**
 * Unified Observer Chronological & Periodicity Model (`src/engine/observerChronology.ts`)
 *
 * Gives the VerdantUI Observer a first-class sensory model of wall-clock time, monotonic elapsed
 * duration, object timeliness, and natural periodicity across ALL modalities:
 *   1. Dual-Clock Spine: binds Unix wall-clock time (`wallClockMs`, local circadian minute/day)
 *      with jitter-free monotonic elapsed time (`monoMs`) so every object in VerdantUI shares
 *      one coherent temporal anchor.
 *   2. Object Timeliness (`timelinessQ10`, `freshnessQ10`): tracks every active and periodic object
 *      (startup, TTFT, decode, Content Streams, knowledge facts, Studio LOD0..LOD3 parts, audio/
 *      vision/video frames, and schedule windows) in exact integer Q10 fixed-point `[0..1024]`.
 *   3. Periodicity & Rhythm (`periodMs`, `observedPeriodMs`, `phaseQ10`): senses natural cycles
 *      — 60 Hz frames (`16.67 ms`), audio BPM beats, stream refresh TTLs, user interaction cadence
 *      (`rapid` / `interactive` / `deliberative`), and 24-hour / 7-day user schedules.
 *   4. Closed-Loop Wall-Clock Throughput Projection: continuously calibrates real host network
 *      (`MB/s`), OPFS read (`MB/s`), prefill (`tok/s`), and decode (`tok/s`) speeds so the Observer
 *      projects real wall-clock milliseconds BEFORE committing any route, stream, or part stage!
 */

import type { OpaqueConstraintImpulse } from './unifiedPossessionModel';

export type ChronologicalModality =
  | 'system'
  | 'language'
  | 'vision'
  | 'audio'
  | 'video'
  | 'prism'
  | 'stream'
  | 'schedule'
  | 'storage';

export type ChronologicalObjectKind =
  | 'app-init'
  | 'turn-ttft'
  | 'turn-decode'
  | 'stream-fetch'
  | 'knowledge-fact'
  | 'prism-part'
  | 'modal-frame'
  | 'schedule-window'
  | 'expert-prewarm';

export type TimelinessState = 'fresh' | 'on-time' | 'tight' | 'overdue' | 'stale';

export type UserInteractionCadence = 'rapid' | 'interactive' | 'deliberative';

export interface ChronologyClockSource {
  wallNow(): number;
  monoNow(): number;
}

export interface ChronologyInstant {
  readonly wallClockMs: number;
  readonly monoMs: number;
  readonly localMinuteOfDay: number; // 0..1439
  readonly localDayOfWeek: number; // 0..6 (Sun..Sat)
  /** Exact integer Q10 phase within the 24-hour circadian cycle (0..1024). */
  readonly circadianPhaseQ10: number;
}

export interface TemporalObjectRecord {
  readonly objectId: string;
  readonly modality: ChronologicalModality;
  readonly kind: ChronologicalObjectKind;
  readonly label: string;
  readonly bornAtWallMs: number;
  readonly startedMonoMs: number;
  readonly budgetMs: number;
  readonly deadlineMonoMs: number;
  /** Natural periodicity in milliseconds (e.g., 16.67 ms frame, 60000/bpm beat, stream TTL). */
  readonly periodMs: number;
  readonly lastPulseWallMs: number;
  readonly lastPulseMonoMs: number;
  readonly pulseCount: number;
  /** EMA-smoothed observed inter-pulse period in milliseconds (0 until >= 2 pulses). */
  readonly observedPeriodMs: number;
  readonly completed: boolean;
  readonly completedDurationMs?: number;
}

export interface TemporalObjectObservation {
  readonly objectId: string;
  readonly modality: ChronologicalModality;
  readonly kind: ChronologicalObjectKind;
  readonly label: string;
  readonly elapsedMs: number;
  readonly remainingMs: number;
  readonly budgetMs: number;
  readonly periodMs: number;
  readonly observedPeriodMs: number;
  /** Exact integer Q10 timeliness score (1024 = crisp/on-time, 0 = severely overdue). */
  readonly timelinessQ10: number;
  /** Exact integer Q10 timeliness pressure (0 = relaxed, 512 = tight, 1024 = overdue). */
  readonly pressureQ10: number;
  /** Exact integer Q10 phase within the object's natural period (0..1024). */
  readonly phaseQ10: number;
  readonly state: TimelinessState;
}

export interface TemporalThroughputModel {
  readonly networkBytesPerSec: number;
  readonly opfsReadBytesPerSec: number;
  readonly prefillTokensPerSec: number;
  readonly decodeTokensPerSec: number;
  readonly userTurnPeriodMs: number;
  readonly networkSamples: number;
  readonly turnSamples: number;
}

export interface TemporalCoherenceSnapshot {
  readonly instant: ChronologyInstant;
  /** Global exact-integer Q10 timeliness coherence across all tracked modalities (0..1024). */
  readonly coherenceQ10: number;
  readonly activeObjectCount: number;
  readonly tightCount: number;
  readonly overdueCount: number;
  readonly userCadence: UserInteractionCadence;
  readonly throughput: TemporalThroughputModel;
  readonly modalityCoherenceQ10: Readonly<Partial<Record<ChronologicalModality, number>>>;
  readonly activeObjects: readonly TemporalObjectObservation[];
}

export interface ChronologicalSlas {
  readonly appInitTargetMs: number;
  readonly turnTtftTargetMs: number;
  readonly streamFetchMaxMs: number;
  readonly prismLod0Ms: number;
  readonly prismLod1Ms: number;
  readonly prismLod2Ms: number;
  readonly prismLod3Ms: number;
  readonly framePeriodMs: number;
}

export const DEFAULT_CHRONOLOGICAL_SLAS: ChronologicalSlas = {
  appInitTargetMs: 25_000,
  turnTtftTargetMs: 2_500,
  streamFetchMaxMs: 2_600,
  prismLod0Ms: 600,
  prismLod1Ms: 3_000,
  prismLod2Ms: 8_000,
  prismLod3Ms: 16_000,
  framePeriodMs: 16.667,
};

const THROUGHPUT_STORAGE_KEY = 'verdant-ui:chronology-throughput:v1';
const MAX_TRACKED_OBJECTS = 64;

function clampQ10(v: number): number {
  return Math.max(0, Math.min(1024, Math.round(v)));
}

const DEFAULT_CLOCK: ChronologyClockSource = {
  wallNow: () => Date.now(),
  monoNow: () =>
    typeof performance !== 'undefined' && typeof performance.now === 'function'
      ? performance.now()
      : Date.now(),
};

/**
 * Computes exact-integer Q10 timeliness pressure (`0..1024`) from elapsed vs budget duration:
 *   - `elapsed <= 50% budget` -> low pressure (`0..256`, `'on-time'`)
 *   - `50% < elapsed <= 100% budget` -> rising pressure (`256..768`, `'tight'` past 75%)
 *   - `elapsed > budget` -> overdue (`768..1024`, `'overdue'`)
 */
export function computeTimelinessPressureQ10(elapsedMs: number, budgetMs: number): number {
  if (budgetMs <= 0) return 0;
  const ratio = Math.max(0, elapsedMs) / budgetMs;
  if (ratio <= 0.5) {
    return clampQ10(ratio * 512);
  }
  if (ratio <= 1.0) {
    return clampQ10(256 + (ratio - 0.5) * 1024);
  }
  return clampQ10(768 + Math.min(1, ratio - 1.0) * 256);
}

/**
 * Computes exact-integer Q10 freshness (`0..1024`) for a periodic knowledge/stream object:
 *   - `ageMs <= 0` -> `1024` (brand new)
 *   - `ageMs === ttlMs` -> `512` (half-life boundary -> ready for refresh)
 *   - `ageMs >= 3 * ttlMs` -> `64` (stale floor, never zero so historical recall stays valid offline)
 */
export function computeKnowledgeFreshnessQ10(ageMs: number, periodTtlMs: number): number {
  if (periodTtlMs <= 0) return 1024;
  if (ageMs <= 0) return 1024;
  const ratio = ageMs / periodTtlMs;
  if (ratio <= 1.0) {
    return clampQ10(1024 - ratio * 512);
  }
  const tailDecay = Math.min(1, (ratio - 1.0) / 2.0);
  return clampQ10(Math.max(64, 512 - tailDecay * 448));
}

/** Computes exact-integer Q10 phase (`0..1024`) within a natural periodic cycle `periodMs`. */
export function computePeriodicPhaseQ10(elapsedMs: number, periodMs: number): number {
  if (periodMs <= 0) return 0;
  const mod = ((elapsedMs % periodMs) + periodMs) % periodMs;
  return clampQ10((mod / periodMs) * 1024);
}

/** Classifies user interaction cadence from observed inter-turn wall-clock period. */
export function classifyUserCadence(userTurnPeriodMs: number): UserInteractionCadence {
  if (userTurnPeriodMs > 0 && userTurnPeriodMs <= 12_000) return 'rapid';
  if (userTurnPeriodMs <= 60_000) return 'interactive';
  return 'deliberative';
}

function loadSavedThroughput(): TemporalThroughputModel {
  const defaults: TemporalThroughputModel = {
    networkBytesPerSec: 25 * 1024 * 1024, // 25 MB/s neutral broadband prior until measured
    opfsReadBytesPerSec: 1_200 * 1024 * 1024, // 1.2 GB/s neutral local OPFS prior
    prefillTokensPerSec: 38,
    decodeTokensPerSec: 14,
    userTurnPeriodMs: 25_000,
    networkSamples: 0,
    turnSamples: 0,
  };
  if (typeof window === 'undefined' || typeof window.localStorage === 'undefined') {
    return defaults;
  }
  try {
    const raw = window.localStorage.getItem(THROUGHPUT_STORAGE_KEY);
    if (!raw) return defaults;
    const p = JSON.parse(raw) as Partial<TemporalThroughputModel>;
    return {
      networkBytesPerSec:
        typeof p.networkBytesPerSec === 'number' && p.networkBytesPerSec >= 64 * 1024
          ? p.networkBytesPerSec
          : defaults.networkBytesPerSec,
      opfsReadBytesPerSec:
        typeof p.opfsReadBytesPerSec === 'number' && p.opfsReadBytesPerSec >= 10 * 1024 * 1024
          ? p.opfsReadBytesPerSec
          : defaults.opfsReadBytesPerSec,
      prefillTokensPerSec:
        typeof p.prefillTokensPerSec === 'number' && p.prefillTokensPerSec >= 1
          ? p.prefillTokensPerSec
          : defaults.prefillTokensPerSec,
      decodeTokensPerSec:
        typeof p.decodeTokensPerSec === 'number' && p.decodeTokensPerSec >= 0.5
          ? p.decodeTokensPerSec
          : defaults.decodeTokensPerSec,
      userTurnPeriodMs:
        typeof p.userTurnPeriodMs === 'number' && p.userTurnPeriodMs >= 500
          ? p.userTurnPeriodMs
          : defaults.userTurnPeriodMs,
      networkSamples: typeof p.networkSamples === 'number' ? p.networkSamples : 0,
      turnSamples: typeof p.turnSamples === 'number' ? p.turnSamples : 0,
    };
  } catch {
    return defaults;
  }
}

function saveThroughput(model: TemporalThroughputModel): void {
  if (typeof window === 'undefined' || typeof window.localStorage === 'undefined') return;
  try {
    window.localStorage.setItem(THROUGHPUT_STORAGE_KEY, JSON.stringify(model));
  } catch {
    // Ignore quota/storage errors
  }
}

/**
 * Authoritative Chronological & Periodicity Engine owned by the VerdantUI Observer.
 * Anchors all modalities, parts, streams, and memory lifetimes onto ONE wall-clock timeline.
 */
export class ObserverChronology {
  private readonly clock: ChronologyClockSource;
  readonly slas: ChronologicalSlas;
  private readonly objects = new Map<string, TemporalObjectRecord>();
  private throughput: TemporalThroughputModel;
  private lastUserTurnWallMs = 0;

  constructor(opts?: {
    readonly clock?: ChronologyClockSource;
    readonly slas?: Partial<ChronologicalSlas>;
    readonly initialThroughput?: Partial<TemporalThroughputModel>;
  }) {
    this.clock = opts?.clock ?? DEFAULT_CLOCK;
    this.slas = { ...DEFAULT_CHRONOLOGICAL_SLAS, ...(opts?.slas ?? {}) };
    this.throughput = {
      ...loadSavedThroughput(),
      ...(opts?.initialThroughput ?? {}),
    };
  }

  /** Reads the current coherent dual-clock instant (wall clock + monotonic + circadian phase). */
  now(): ChronologyInstant {
    const wallClockMs = this.clock.wallNow();
    const monoMs = this.clock.monoNow();
    const d = new Date(wallClockMs);
    const localMinuteOfDay = d.getHours() * 60 + d.getMinutes();
    const localDayOfWeek = d.getDay();
    const circadianPhaseQ10 = clampQ10((localMinuteOfDay / 1440) * 1024);
    return {
      wallClockMs,
      monoMs,
      localMinuteOfDay,
      localDayOfWeek,
      circadianPhaseQ10,
    };
  }

  getThroughput(): TemporalThroughputModel {
    return this.throughput;
  }

  getUserCadence(): UserInteractionCadence {
    return classifyUserCadence(this.throughput.userTurnPeriodMs);
  }

  /** Records a user turn arrival on the wall clock to sense interaction periodicity & rhythm. */
  observeUserTurn(): { readonly cadence: UserInteractionCadence; readonly turnPeriodMs: number } {
    const wall = this.clock.wallNow();
    if (this.lastUserTurnWallMs > 0) {
      const delta = wall - this.lastUserTurnWallMs;
      if (delta >= 500 && delta <= 30 * 60_000) {
        const nextPeriod =
          this.throughput.turnSamples === 0
            ? delta
            : Math.round(this.throughput.userTurnPeriodMs * 0.65 + delta * 0.35);
        this.throughput = {
          ...this.throughput,
          userTurnPeriodMs: nextPeriod,
          turnSamples: this.throughput.turnSamples + 1,
        };
      }
    }
    this.lastUserTurnWallMs = wall;
    return {
      cadence: this.getUserCadence(),
      turnPeriodMs: this.throughput.userTurnPeriodMs,
    };
  }

  /** Calibrates live network transfer speed (`bytes/sec`) from actual Range/stream measurements. */
  recordNetworkThroughput(bytesPerSec: number): void {
    if (!Number.isFinite(bytesPerSec) || bytesPerSec < 32 * 1024) return;
    const clamped = Math.min(2 * 1024 * 1024 * 1024, bytesPerSec);
    const alpha = this.throughput.networkSamples === 0 ? 0.85 : 0.35;
    this.throughput = {
      ...this.throughput,
      networkBytesPerSec: Math.round(
        this.throughput.networkBytesPerSec * (1 - alpha) + clamped * alpha,
      ),
      networkSamples: this.throughput.networkSamples + 1,
    };
    saveThroughput(this.throughput);
  }

  /** Calibrates live inference prefill & decode speeds from actual completed turns. */
  recordInferenceThroughput(params: {
    readonly prefillTokSec?: number;
    readonly decodeTokSec?: number;
    readonly opfsReadBytesPerSec?: number;
  }): void {
    let changed = false;
    let next = this.throughput;
    if (
      typeof params.prefillTokSec === 'number' &&
      Number.isFinite(params.prefillTokSec) &&
      params.prefillTokSec >= 0.5
    ) {
      const alpha = next.turnSamples === 0 ? 0.75 : 0.3;
      next = {
        ...next,
        prefillTokensPerSec: Number(
          (next.prefillTokensPerSec * (1 - alpha) + params.prefillTokSec * alpha).toFixed(2),
        ),
      };
      changed = true;
    }
    if (
      typeof params.decodeTokSec === 'number' &&
      Number.isFinite(params.decodeTokSec) &&
      params.decodeTokSec >= 0.2
    ) {
      const alpha = next.turnSamples === 0 ? 0.75 : 0.3;
      next = {
        ...next,
        decodeTokensPerSec: Number(
          (next.decodeTokensPerSec * (1 - alpha) + params.decodeTokSec * alpha).toFixed(2),
        ),
      };
      changed = true;
    }
    if (
      typeof params.opfsReadBytesPerSec === 'number' &&
      Number.isFinite(params.opfsReadBytesPerSec) &&
      params.opfsReadBytesPerSec >= 10 * 1024 * 1024
    ) {
      next = {
        ...next,
        opfsReadBytesPerSec: Math.round(
          next.opfsReadBytesPerSec * 0.7 + params.opfsReadBytesPerSec * 0.3,
        ),
      };
      changed = true;
    }
    if (changed) {
      this.throughput = { ...next, turnSamples: next.turnSamples + 1 };
      saveThroughput(this.throughput);
    }
  }

  /** Projects real wall-clock milliseconds for a network byte transfer on the host's measured link. */
  projectTransferMs(bytes: number): number {
    if (bytes <= 0) return 0;
    return Math.round((bytes / Math.max(64 * 1024, this.throughput.networkBytesPerSec)) * 1000);
  }

  /**
   * Projects real wall-clock Time-to-First-Token (ms) for a route candidate using the Observer's
   * live measured network, OPFS, prefill, and decode rates.
   */
  projectTurnTtftMs(params: {
    readonly missingTransferBytes: number;
    readonly opfsReadBytes?: number;
    readonly uncachedPrefillTokens: number;
    readonly thinkingTokensEstimate?: number;
  }): number {
    const netMs = this.projectTransferMs(params.missingTransferBytes);
    const opfsMs =
      params.opfsReadBytes && params.opfsReadBytes > 0
        ? Math.round((params.opfsReadBytes / Math.max(10 * 1024 * 1024, this.throughput.opfsReadBytesPerSec)) * 1000)
        : 0;
    const prefillMs =
      params.uncachedPrefillTokens > 0
        ? Math.round((params.uncachedPrefillTokens / Math.max(1, this.throughput.prefillTokensPerSec)) * 1000)
        : 0;
    const thinkMs =
      params.thinkingTokensEstimate && params.thinkingTokensEstimate > 0
        ? Math.round((params.thinkingTokensEstimate / Math.max(0.5, this.throughput.decodeTokensPerSec)) * 1000)
        : 0;
    return netMs + opfsMs + prefillMs + thinkMs;
  }

  /**
   * Derives an adaptive wall-clock timeout (ms) for external Realtime Content Streams based on
   * user cadence (`rapid` vs `deliberative`) and the turn's TTFT budget.
   */
  deriveAdaptiveStreamTimeoutMs(featureRichness: 'minimal' | 'balanced' | 'full'): number {
    const cadence = this.getUserCadence();
    const base =
      featureRichness === 'full'
        ? 3_200
        : featureRichness === 'minimal'
          ? 1_600
          : this.slas.streamFetchMaxMs;
    if (cadence === 'rapid') {
      return Math.max(1_200, Math.round(base * 0.72));
    }
    if (cadence === 'deliberative') {
      return Math.min(4_000, Math.round(base * 1.15));
    }
    return base;
  }

  /** Derives wall-clock stage budget (ms) for a Verdant Studio LOD reconciliation stage. */
  derivePrismStageBudgetMs(lodRank: 0 | 1 | 2 | 3): number {
    if (lodRank === 0) return this.slas.prismLod0Ms;
    if (lodRank === 1) return this.slas.prismLod1Ms;
    if (lodRank === 2) return this.slas.prismLod2Ms;
    return this.slas.prismLod3Ms;
  }

  /** Registers or restarts a tracked chronological object on the Observer's timeline. */
  beginObject(params: {
    readonly objectId: string;
    readonly modality: ChronologicalModality;
    readonly kind: ChronologicalObjectKind;
    readonly label: string;
    readonly budgetMs: number;
    readonly periodMs?: number;
  }): TemporalObjectRecord {
    const wall = this.clock.wallNow();
    const mono = this.clock.monoNow();
    const existing = this.objects.get(params.objectId);
    const prevPulseDelta =
      existing && existing.lastPulseMonoMs > 0 ? Math.max(0, mono - existing.lastPulseMonoMs) : 0;
    const observedPeriodMs =
      prevPulseDelta > 0
        ? existing && existing.observedPeriodMs > 0
          ? Math.round(existing.observedPeriodMs * 0.6 + prevPulseDelta * 0.4)
          : Math.round(prevPulseDelta)
        : existing?.observedPeriodMs ?? 0;

    const record: TemporalObjectRecord = {
      objectId: params.objectId,
      modality: params.modality,
      kind: params.kind,
      label: params.label,
      bornAtWallMs: existing?.bornAtWallMs ?? wall,
      startedMonoMs: mono,
      budgetMs: Math.max(1, params.budgetMs),
      deadlineMonoMs: mono + Math.max(1, params.budgetMs),
      periodMs: Math.max(0, params.periodMs ?? 0),
      lastPulseWallMs: wall,
      lastPulseMonoMs: mono,
      pulseCount: (existing?.pulseCount ?? 0) + 1,
      observedPeriodMs,
      completed: false,
    };
    this.objects.set(params.objectId, record);
    this.pruneIfNeeded();
    return record;
  }

  /**
   * Pulses a periodic or ongoing object (e.g., audio beat, video frame, stream refresh, token)
   * to track its true wall-clock periodicity and reset freshness.
   */
  pulseObject(params: {
    readonly objectId: string;
    readonly modality: ChronologicalModality;
    readonly kind: ChronologicalObjectKind;
    readonly label: string;
    readonly periodMs: number;
    readonly budgetMs?: number;
  }): TemporalObjectObservation {
    const wall = this.clock.wallNow();
    const mono = this.clock.monoNow();
    const prev = this.objects.get(params.objectId);
    const delta = prev ? Math.max(0, mono - prev.lastPulseMonoMs) : 0;
    const observedPeriodMs =
      prev && delta > 0
        ? prev.observedPeriodMs > 0
          ? Math.round(prev.observedPeriodMs * 0.65 + delta * 0.35)
          : Math.round(delta)
        : prev?.observedPeriodMs ?? 0;

    const budget = Math.max(1, params.budgetMs ?? params.periodMs);
    const rec: TemporalObjectRecord = {
      objectId: params.objectId,
      modality: params.modality,
      kind: params.kind,
      label: params.label,
      bornAtWallMs: prev?.bornAtWallMs ?? wall,
      startedMonoMs: mono,
      budgetMs: budget,
      deadlineMonoMs: mono + budget,
      periodMs: Math.max(1, params.periodMs),
      lastPulseWallMs: wall,
      lastPulseMonoMs: mono,
      pulseCount: (prev?.pulseCount ?? 0) + 1,
      observedPeriodMs,
      completed: false,
    };
    this.objects.set(params.objectId, rec);
    this.pruneIfNeeded();
    return this.observeObject(params.objectId)!;
  }

  /** Marks an active chronological object complete and records its exact elapsed wall duration. */
  completeObject(objectId: string): TemporalObjectObservation | null {
    const prev = this.objects.get(objectId);
    if (!prev) return null;
    const mono = this.clock.monoNow();
    const wall = this.clock.wallNow();
    const durationMs = Math.max(0, mono - prev.startedMonoMs);
    const next: TemporalObjectRecord = {
      ...prev,
      lastPulseWallMs: wall,
      lastPulseMonoMs: mono,
      completed: true,
      completedDurationMs: durationMs,
    };
    this.objects.set(objectId, next);
    return this.toObservation(next, mono);
  }

  /** Returns the remaining monotonic budget (ms) for an active object (0 when overdue). */
  getRemainingBudgetMs(objectId: string): number {
    const rec = this.objects.get(objectId);
    if (!rec || rec.completed) return 0;
    return Math.max(0, Math.round(rec.deadlineMonoMs - this.clock.monoNow()));
  }

  /** Observes a single registered object's live timeliness, phase, and state. */
  observeObject(objectId: string): TemporalObjectObservation | null {
    const rec = this.objects.get(objectId);
    if (!rec) return null;
    return this.toObservation(rec, this.clock.monoNow());
  }

  /** Filters opaque schedule constraints to those currently active on the local wall clock. */
  getActiveScheduleWindows(
    constraints: readonly OpaqueConstraintImpulse[],
  ): readonly OpaqueConstraintImpulse[] {
    if (constraints.length === 0) return [];
    const inst = this.now();
    return constraints.filter(
      (c) =>
        c.daysOfWeek.includes(inst.localDayOfWeek) &&
        inst.localMinuteOfDay >= c.startMinuteOfDay &&
        inst.localMinuteOfDay < c.endMinuteOfDay,
    );
  }

  /** Returns a coherent snapshot of timeliness across all modalities for the Observer & telemetry. */
  snapshot(): TemporalCoherenceSnapshot {
    const instant = this.now();
    const activeObjects: TemporalObjectObservation[] = [];
    const modalitySums = new Map<ChronologicalModality, { sum: number; count: number }>();
    let tightCount = 0;
    let overdueCount = 0;
    let totalTimeliness = 0;

    for (const rec of this.objects.values()) {
      const obs = this.toObservation(rec, instant.monoMs);
      if (!rec.completed) {
        activeObjects.push(obs);
        if (obs.state === 'tight') tightCount += 1;
        if (obs.state === 'overdue' || obs.state === 'stale') overdueCount += 1;
      }
      const bucket = modalitySums.get(rec.modality) ?? { sum: 0, count: 0 };
      bucket.sum += obs.timelinessQ10;
      bucket.count += 1;
      modalitySums.set(rec.modality, bucket);
      totalTimeliness += obs.timelinessQ10;
    }

    const modalityCoherenceQ10: Partial<Record<ChronologicalModality, number>> = {};
    for (const [mod, { sum, count }] of modalitySums) {
      modalityCoherenceQ10[mod] = clampQ10(sum / Math.max(1, count));
    }

    const coherenceQ10 =
      this.objects.size === 0 ? 1024 : clampQ10(totalTimeliness / this.objects.size);

    return {
      instant,
      coherenceQ10,
      activeObjectCount: activeObjects.length,
      tightCount,
      overdueCount,
      userCadence: this.getUserCadence(),
      throughput: this.throughput,
      modalityCoherenceQ10,
      activeObjects,
    };
  }

  private toObservation(rec: TemporalObjectRecord, nowMonoMs: number): TemporalObjectObservation {
    const elapsedMs = rec.completed
      ? (rec.completedDurationMs ?? 0)
      : Math.max(0, nowMonoMs - rec.startedMonoMs);
    const remainingMs = rec.completed ? 0 : Math.max(0, rec.deadlineMonoMs - nowMonoMs);
    const pressureQ10 = computeTimelinessPressureQ10(elapsedMs, rec.budgetMs);
    const timelinessQ10 = clampQ10(1024 - pressureQ10);
    const phaseQ10 =
      rec.periodMs > 0 ? computePeriodicPhaseQ10(elapsedMs, rec.periodMs) : 0;

    let state: TimelinessState;
    if (rec.kind === 'knowledge-fact' && rec.periodMs > 0) {
      const fresh = computeKnowledgeFreshnessQ10(elapsedMs, rec.periodMs);
      state = fresh >= 768 ? 'fresh' : fresh >= 512 ? 'on-time' : 'stale';
    } else if (elapsedMs > rec.budgetMs) {
      state = 'overdue';
    } else if (pressureQ10 >= 512) {
      state = 'tight';
    } else {
      state = 'on-time';
    }

    return {
      objectId: rec.objectId,
      modality: rec.modality,
      kind: rec.kind,
      label: rec.label,
      elapsedMs: Math.round(elapsedMs),
      remainingMs: Math.round(remainingMs),
      budgetMs: rec.budgetMs,
      periodMs: rec.periodMs,
      observedPeriodMs: rec.observedPeriodMs,
      timelinessQ10,
      pressureQ10,
      phaseQ10,
      state,
    };
  }

  private pruneIfNeeded(): void {
    if (this.objects.size <= MAX_TRACKED_OBJECTS) return;
    // Evict oldest completed objects first, preserving active in-flight objects
    const completed = [...this.objects.values()]
      .filter((o) => o.completed)
      .sort((a, b) => a.lastPulseMonoMs - b.lastPulseMonoMs);
    for (const c of completed) {
      if (this.objects.size <= MAX_TRACKED_OBJECTS) return;
      this.objects.delete(c.objectId);
    }
    while (this.objects.size > MAX_TRACKED_OBJECTS) {
      const oldest = this.objects.keys().next().value;
      if (oldest === undefined) break;
      this.objects.delete(oldest);
    }
  }
}
