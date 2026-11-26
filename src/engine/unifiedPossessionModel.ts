/**
 * Unified Possession Hierarchy (`Observer ⊇ User ⊇ Parts`) & Non-Leaking Cross-Context Boundary
 * (`src/engine/unifiedPossessionModel.ts`)
 *
 * Implements DEC-010 (extrapolating clean core invariants from `afflom/me` `PrismBoundary`,
 * `PrismDest`, `PrismHolon`, and `Foundation.Me` without importing schema bloat):
 *
 * 1. Explicit Possession Hierarchy:
 *    - `UserModel` possesses its VerdantUI parts: owned conversation threads/messages,
 *      owned draft/published Prisms, and isolated `ContextCompartment`s (`personal`, `work`,
 *      `health`, `family`, `research`, or custom user-defined spaces).
 *    - `ObserverRootModel` possesses the `UserModel` and all of its parts, plus active
 *      Published Prism cognitive organs, sensory frames, and action impulses.
 * 2. Zero-Detail Opaque Constraint Projection (`projectOpaqueConstraints`):
 *    - Raw facts, messages, labels, and Prism payloads NEVER cross between compartments
 *      (e.g. `personal` *Healthy Habits* -> `work` *Calendar*).
 *    - Instead, a foreign compartment projects strictly zero-detail `OpaqueConstraintImpulse`
 *      weights (`{ startMinuteOfDay, endMinuteOfDay, daysOfWeek, weightQ10, opaqueSourceKappa, detail: null }`)
 *      so `work` schedules around a protected personal window with ZERO personal details exposed.
 * 3. Deterministic Context Boundary Gate (`evaluateContextBoundary`):
 *    - Kill switch (`halted = true`) denies all cross-context & outbound transfers immediately.
 *    - Regulated detectors (`pci`, `phi`, `credential`, `special`) and `restricted` labels are
 *      forced to `hashonly` locally and `deny` outbound.
 *    - Outbound delivery (`calendar`, `mail`, `platform`, `network`) verifies zero cross-context
 *      taint (`verifyZeroCrossContextLeak`) and requires owner approval on the exact seen bytes.
 */

import { prismContentDigest } from './prismKernel';
import type { PrismMediaType } from './prismMediaOrgans';

export type ContextCompartmentId =
  | 'personal'
  | 'work'
  | 'health'
  | 'family'
  | 'social'
  | 'research'
  | 'public'
  | (string & {});

export type DataSensitivityLabel =
  | 'public'
  | 'internal'
  | 'confidential'
  | 'restricted'
  | 'unlabeled';

export type RegulatedDetectorKind = 'none' | 'pci' | 'phi' | 'credential' | 'special';

export type EgressDestination =
  | 'browser-local'
  | 'same-compartment'
  | 'foreign-compartment'
  | 'calendar'
  | 'mail'
  | 'platform'
  | 'network';

export type BoundaryVerdict = 'allow' | 'hashonly' | 'ask' | 'deny';

export interface CompartmentScheduleConstraint {
  readonly id: string;
  /** Private human label stored strictly inside the owning compartment (NEVER projected!). */
  readonly privateLabel: string;
  readonly sourcePrismId?: string;
  readonly startMinuteOfDay: number;
  readonly endMinuteOfDay: number;
  readonly daysOfWeek: readonly number[]; // 0=Sun..6=Sat
  readonly weightQ10: number; // 0..1024 exact integer priority
  readonly shareOpaqueAvailability: boolean;
}

export interface ContextCompartment {
  readonly compartmentId: ContextCompartmentId;
  readonly label: string;
  readonly sensitivityFloor: DataSensitivityLabel;
  readonly ownedThreadIds: readonly string[];
  readonly attachedPrismIds: readonly string[];
  /** Private compartment tokens/facts used by zero-leak taint detection (never leave compartment). */
  readonly privateFacts: readonly string[];
  readonly scheduleConstraints: readonly CompartmentScheduleConstraint[];
}

export interface OpaqueConstraintImpulse {
  readonly kind: 'availability-block';
  readonly targetCompartmentId: ContextCompartmentId;
  /** Deterministic opaque hash proving origin without revealing source compartment or Prism name. */
  readonly opaqueSourceKappa: string;
  readonly startMinuteOfDay: number;
  readonly endMinuteOfDay: number;
  readonly daysOfWeek: readonly number[];
  readonly weightQ10: number;
  /** Structurally guaranteed null across compartment boundaries (zero detail leakage). */
  readonly detail: null;
}

export interface UserModel {
  readonly userId: 'owner';
  readonly ownedThreadIds: readonly string[];
  readonly ownedPrismIds: readonly string[];
  readonly compartments: Readonly<Record<string, ContextCompartment>>;
  readonly activeCompartmentId: ContextCompartmentId;
  readonly quietMode: boolean;
  /** Global owner kill switch: when true, all cross-context & outbound movement is refused. */
  readonly halted: boolean;
}

export interface PublishedPrismOrgan {
  readonly prismId: string;
  readonly name: string;
  readonly slug: string;
  readonly digest: string;
  readonly mediaType: PrismMediaType;
  readonly compartmentId: ContextCompartmentId;
  /** 8D Q10 cognitive facet affinity weights contributed into the unified model. */
  readonly cognitiveWeightsQ10: readonly [number, number, number, number, number, number, number, number];
  readonly sensoryChannels: readonly ('browser' | 'storage' | 'schedule' | 'network')[];
  readonly impulseChannels: readonly ('render-surface' | 'notify' | 'schedule-constraint' | 'stream-emit' | 'tool-invoke')[];
  readonly shareOpaqueConstraints: boolean;
}

export interface PrismSensoryFrame {
  readonly id: string;
  readonly prismId: string;
  readonly compartmentId: ContextCompartmentId;
  readonly channel: 'browser' | 'storage' | 'schedule' | 'network';
  readonly sensitivity: DataSensitivityLabel;
  readonly payloadText: string;
  readonly receivedAt: number;
}

export interface ObserverActionImpulse {
  readonly id: string;
  readonly targetPrismId: string;
  readonly sourceCompartmentId: ContextCompartmentId;
  readonly targetCompartmentId: ContextCompartmentId;
  readonly kind: 'render-surface' | 'notify' | 'schedule-constraint' | 'stream-emit' | 'tool-invoke';
  readonly payload: unknown;
  readonly opaqueOnly: boolean;
  readonly ownerApproved: boolean;
  readonly emittedAt: number;
}

export interface BoundaryEvaluationRequest {
  readonly sourceCompartmentId: ContextCompartmentId;
  readonly targetCompartmentId?: ContextCompartmentId;
  readonly destination: EgressDestination;
  readonly label: DataSensitivityLabel;
  readonly payloadText: string;
  readonly isOpaqueProjection?: boolean;
  readonly ownerApproved?: boolean;
  readonly bytesMatchSeen?: boolean;
  readonly halted: boolean;
  readonly foreignCompartments?: readonly ContextCompartment[];
}

export interface BoundaryEvaluationResult {
  readonly verdict: BoundaryVerdict;
  readonly detector: RegulatedDetectorKind;
  readonly containsIdentifier: boolean;
  readonly leakedForeignTerms: readonly string[];
  readonly reason: string;
}

export const DEFAULT_USER_SPACES: readonly {
  readonly id: ContextCompartmentId;
  readonly label: string;
  readonly floor: DataSensitivityLabel;
}[] = [
  { id: 'personal', label: 'Personal', floor: 'confidential' },
  { id: 'work', label: 'Work', floor: 'internal' },
  { id: 'health', label: 'Health', floor: 'restricted' },
  { id: 'research', label: 'Research', floor: 'internal' },
];

// ============================================================================
// 1. Regulated Data & Personal Identifier Detection (`afflom/me` parity)
// ============================================================================

const PCI_RE = /\b(?:\d[ -]*?){13,19}\b/;
const CREDENTIAL_RE = /\b(?:ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|Bearer\s+[A-Za-z0-9._-]{20,}|-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----)\b/;
const PHI_RE = /\b(?:diagnosis|prescription|medical\s+record|blood\s+pressure|hemoglobin|therapy\s+session|patient\s+id|ssn|social\s+security)\b/i;
const IDENTIFIER_RE = /\b(?:[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4})\b/;

export function detectRegulatedContent(text: string): {
  readonly detector: RegulatedDetectorKind;
  readonly containsIdentifier: boolean;
} {
  const s = text || '';
  let detector: RegulatedDetectorKind = 'none';
  if (CREDENTIAL_RE.test(s)) detector = 'credential';
  else if (PCI_RE.test(s)) detector = 'pci';
  else if (PHI_RE.test(s)) detector = 'phi';
  const containsIdentifier = IDENTIFIER_RE.test(s);
  return { detector, containsIdentifier };
}

/**
 * Verifies that `outboundText` contains ZERO private facts or private schedule labels from
 * any foreign compartment (non-interference check across spaces).
 */
export function verifyZeroCrossContextLeak(
  outboundText: string,
  foreignCompartments: readonly ContextCompartment[],
): readonly string[] {
  const lower = (outboundText || '').toLowerCase();
  if (!lower) return [];
  const leaked: string[] = [];
  for (const comp of foreignCompartments) {
    for (const fact of comp.privateFacts) {
      const token = fact.trim().toLowerCase();
      if (token.length >= 4 && lower.includes(token)) {
        leaked.push(fact);
      }
    }
    for (const sc of comp.scheduleConstraints) {
      const lbl = sc.privateLabel.trim().toLowerCase();
      if (lbl.length >= 4 && lower.includes(lbl)) {
        leaked.push(sc.privateLabel);
      }
    }
  }
  return leaked;
}

/**
 * Pure deterministic Context Boundary Gate (`afflom/me` `PrismBoundary` + `PrismDest` parity).
 */
export function evaluateContextBoundary(req: BoundaryEvaluationRequest): BoundaryEvaluationResult {
  const { detector, containsIdentifier } = detectRegulatedContent(req.payloadText);
  const foreigns = (req.foreignCompartments ?? []).filter(
    (c) => c.compartmentId !== req.sourceCompartmentId,
  );
  const leakedForeignTerms = verifyZeroCrossContextLeak(req.payloadText, foreigns);

  // 1. Global kill switch: stop means stop
  if (req.halted) {
    return {
      verdict: 'deny',
      detector,
      containsIdentifier,
      leakedForeignTerms,
      reason: 'Sharing is paused by owner kill switch.',
    };
  }

  // 2. Cross-context taint detection: NEVER leak foreign compartment private facts
  if (leakedForeignTerms.length > 0 && req.destination !== 'browser-local') {
    return {
      verdict: 'deny',
      detector,
      containsIdentifier,
      leakedForeignTerms,
      reason: 'Blocked potential cross-space private detail leakage.',
    };
  }

  // 3. Regulated detectors or restricted label: hashonly locally, deny outbound
  const isOutbound =
    req.destination === 'calendar' ||
    req.destination === 'mail' ||
    req.destination === 'platform' ||
    req.destination === 'network';

  if (detector !== 'none' || req.label === 'restricted') {
    if (req.destination === 'browser-local' || req.destination === 'same-compartment') {
      return {
        verdict: 'hashonly',
        detector,
        containsIdentifier,
        leakedForeignTerms,
        reason: 'Sensitive/regulated data may only be stored as a local hash pointer.',
      };
    }
    return {
      verdict: 'deny',
      detector,
      containsIdentifier,
      leakedForeignTerms,
      reason: 'Sensitive/regulated data may never leave its local compartment.',
    };
  }

  // 4. Cross-compartment transfer: only opaque projections allowed across spaces
  if (
    req.destination === 'foreign-compartment' ||
    (req.targetCompartmentId && req.targetCompartmentId !== req.sourceCompartmentId)
  ) {
    if (req.isOpaqueProjection) {
      return {
        verdict: 'allow',
        detector,
        containsIdentifier,
        leakedForeignTerms,
        reason: 'Zero-detail opaque constraint projection permitted across spaces.',
      };
    }
    return {
      verdict: 'deny',
      detector,
      containsIdentifier,
      leakedForeignTerms,
      reason: 'Raw payloads cannot cross spaces; use opaque constraint projection instead.',
    };
  }

  // 5. Outbound external delivery: requires explicit owner approval on exact seen bytes
  if (isOutbound) {
    if (!req.ownerApproved || req.bytesMatchSeen === false) {
      return {
        verdict: 'ask',
        detector,
        containsIdentifier,
        leakedForeignTerms,
        reason: 'External delivery requires owner approval on the exact bytes to send.',
      };
    }
    if (containsIdentifier && req.label !== 'public' && !req.ownerApproved) {
      return {
        verdict: 'ask',
        detector,
        containsIdentifier,
        leakedForeignTerms,
        reason: 'Personal identifier detected; owner approval required.',
      };
    }
    return {
      verdict: 'allow',
      detector,
      containsIdentifier,
      leakedForeignTerms,
      reason: 'Owner-approved outbound delivery verified leak-free.',
    };
  }

  return {
    verdict: 'allow',
    detector,
    containsIdentifier,
    leakedForeignTerms,
    reason: 'Permitted within owning local compartment.',
  };
}

// ============================================================================
// 2. Zero-Detail Opaque Constraint Projection (`Healthy Habits -> Work`)
// ============================================================================

export function formatMinuteOfDay(min: number): string {
  const clamped = Math.max(0, Math.min(1440, Math.round(min)));
  const hh = String(Math.floor(clamped / 60)).padStart(2, '0');
  const mm = String(clamped % 60).padStart(2, '0');
  return `${hh}:${mm}`;
}

/**
 * Projects strictly zero-detail `OpaqueConstraintImpulse` availability blocks from all foreign
 * compartments owned by `user` into `targetCompartmentId`.
 * Structurally incapable of leaking source compartment names, Prism names, or private labels.
 */
export function projectOpaqueConstraints(
  user: UserModel,
  targetCompartmentId: ContextCompartmentId,
): readonly OpaqueConstraintImpulse[] {
  if (user.halted) return [];
  const projected: OpaqueConstraintImpulse[] = [];

  for (const comp of Object.values(user.compartments)) {
    if (comp.compartmentId === targetCompartmentId) continue;
    for (const sc of comp.scheduleConstraints) {
      if (!sc.shareOpaqueAvailability) continue;
      // Hash origin deterministically so the target never learns which space or Prism emitted it
      const opaqueSourceKappa = prismContentDigest(
        `opaque:${comp.compartmentId}:${sc.id}:${sc.startMinuteOfDay}:${sc.endMinuteOfDay}`,
      ).slice(0, 16);
      projected.push({
        kind: 'availability-block',
        targetCompartmentId,
        opaqueSourceKappa,
        startMinuteOfDay: Math.max(0, Math.min(1439, Math.round(sc.startMinuteOfDay))),
        endMinuteOfDay: Math.max(1, Math.min(1440, Math.round(sc.endMinuteOfDay))),
        daysOfWeek: [...sc.daysOfWeek],
        weightQ10: Math.max(0, Math.min(1024, Math.round(sc.weightQ10))),
        detail: null,
      });
    }
  }

  return projected;
}

const DAY_NAMES = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'] as const;

export function formatOpaqueConstraintsForContext(
  impulses: readonly OpaqueConstraintImpulse[],
): string {
  if (impulses.length === 0) return '';
  const lines = impulses.map((imp) => {
    const days =
      imp.daysOfWeek.length === 7
        ? 'Daily'
        : imp.daysOfWeek.map((d) => DAY_NAMES[d] ?? '').filter(Boolean).join('/');
    return `- ${days} ${formatMinuteOfDay(imp.startMinuteOfDay)}–${formatMinuteOfDay(imp.endMinuteOfDay)}: Unavailable (protected schedule constraint #${imp.opaqueSourceKappa.slice(0, 6)})`;
  });
  return [
    'Protected availability constraints (honor these when scheduling; do NOT ask about, guess, or disclose any private reasons):',
    ...lines,
  ].join('\n');
}

// ============================================================================
// 3. ObserverRootModel (Observer ⊇ User ⊇ {Threads, Prisms, Compartments})
// ============================================================================

export class ObserverPossessionStore {
  private user: UserModel;
  private readonly publishedOrgans = new Map<string, PublishedPrismOrgan>();
  private readonly sensoryFrames: PrismSensoryFrame[] = [];
  private readonly impulseHistory: ObserverActionImpulse[] = [];

  constructor(initialActiveSpace: ContextCompartmentId = 'personal') {
    const compartments: Record<string, ContextCompartment> = {};
    for (const sp of DEFAULT_USER_SPACES) {
      compartments[sp.id] = {
        compartmentId: sp.id,
        label: sp.label,
        sensitivityFloor: sp.floor,
        ownedThreadIds: [],
        attachedPrismIds: [],
        privateFacts: [],
        scheduleConstraints: [],
      };
    }
    this.user = {
      userId: 'owner',
      ownedThreadIds: [],
      ownedPrismIds: [],
      compartments,
      activeCompartmentId: compartments[initialActiveSpace] ? initialActiveSpace : 'personal',
      quietMode: false,
      halted: false,
    };
  }

  getUser(): UserModel {
    return this.user;
  }

  getActiveCompartmentId(): ContextCompartmentId {
    return this.user.activeCompartmentId;
  }

  setActiveCompartment(compartmentId: ContextCompartmentId): void {
    this.ensureCompartment(compartmentId);
    this.user = { ...this.user, activeCompartmentId: compartmentId };
  }

  setHalted(halted: boolean): void {
    this.user = { ...this.user, halted };
  }

  isHalted(): boolean {
    return this.user.halted;
  }

  setQuietMode(quietMode: boolean): void {
    this.user = { ...this.user, quietMode };
  }

  ensureCompartment(
    compartmentId: ContextCompartmentId,
    label?: string,
    sensitivityFloor: DataSensitivityLabel = 'confidential',
  ): ContextCompartment {
    const existing = this.user.compartments[compartmentId];
    if (existing) return existing;
    const created: ContextCompartment = {
      compartmentId,
      label: label || compartmentId.charAt(0).toUpperCase() + compartmentId.slice(1),
      sensitivityFloor,
      ownedThreadIds: [],
      attachedPrismIds: [],
      privateFacts: [],
      scheduleConstraints: [],
    };
    this.user = {
      ...this.user,
      compartments: { ...this.user.compartments, [compartmentId]: created },
    };
    return created;
  }

  /** Claims ownership of a thread inside a specific compartment (`User ⊇ Thread`). */
  bindThreadToCompartment(threadId: string, compartmentId: ContextCompartmentId): void {
    const comp = this.ensureCompartment(compartmentId);
    const ownedThreads = this.user.ownedThreadIds.includes(threadId)
      ? this.user.ownedThreadIds
      : [...this.user.ownedThreadIds, threadId];
    const compThreads = comp.ownedThreadIds.includes(threadId)
      ? comp.ownedThreadIds
      : [...comp.ownedThreadIds, threadId];
    this.user = {
      ...this.user,
      ownedThreadIds: ownedThreads,
      compartments: {
        ...this.user.compartments,
        [compartmentId]: { ...comp, ownedThreadIds: compThreads },
      },
    };
  }

  /** Claims ownership of a Prism inside a compartment (`Observer ⊇ User ⊇ Prism`). */
  bindPrismToCompartment(
    prismId: string,
    compartmentId: ContextCompartmentId,
    privateLabels: readonly string[] = [],
  ): void {
    const comp = this.ensureCompartment(compartmentId);
    const ownedPrisms = this.user.ownedPrismIds.includes(prismId)
      ? this.user.ownedPrismIds
      : [...this.user.ownedPrismIds, prismId];
    const attached = comp.attachedPrismIds.includes(prismId)
      ? comp.attachedPrismIds
      : [...comp.attachedPrismIds, prismId];
    const nextFacts = Array.from(new Set([...comp.privateFacts, ...privateLabels.filter(Boolean)]));
    this.user = {
      ...this.user,
      ownedPrismIds: ownedPrisms,
      compartments: {
        ...this.user.compartments,
        [compartmentId]: { ...comp, attachedPrismIds: attached, privateFacts: nextFacts },
      },
    };
  }

  upsertScheduleConstraint(
    compartmentId: ContextCompartmentId,
    constraint: CompartmentScheduleConstraint,
  ): void {
    const comp = this.ensureCompartment(compartmentId);
    const filtered = comp.scheduleConstraints.filter((c) => c.id !== constraint.id);
    this.user = {
      ...this.user,
      compartments: {
        ...this.user.compartments,
        [compartmentId]: {
          ...comp,
          scheduleConstraints: [...filtered, constraint],
        },
      },
    };
  }

  registerPublishedPrismOrgan(organ: PublishedPrismOrgan): void {
    this.publishedOrgans.set(organ.prismId, organ);
    this.bindPrismToCompartment(organ.prismId, organ.compartmentId, [organ.name]);
  }

  listPublishedOrgans(compartmentId?: ContextCompartmentId): readonly PublishedPrismOrgan[] {
    const all = Array.from(this.publishedOrgans.values());
    if (!compartmentId) return all;
    return all.filter((o) => o.compartmentId === compartmentId);
  }

  getProjectedConstraintsFor(targetCompartmentId: ContextCompartmentId): readonly OpaqueConstraintImpulse[] {
    return projectOpaqueConstraints(this.user, targetCompartmentId);
  }

  /** Ingests a sensory signal from a Published Prism, enforcing the non-leaking boundary gate. */
  ingestSensoryFrame(frame: PrismSensoryFrame): BoundaryEvaluationResult {
    const verdict = evaluateContextBoundary({
      sourceCompartmentId: frame.compartmentId,
      destination: 'browser-local',
      label: frame.sensitivity,
      payloadText: frame.payloadText,
      halted: this.user.halted,
    });
    if (verdict.verdict === 'allow' || verdict.verdict === 'hashonly') {
      const storedPayload =
        verdict.verdict === 'hashonly'
          ? `sha256:${prismContentDigest(frame.payloadText).slice(0, 16)}`
          : frame.payloadText.slice(0, 512);
      this.sensoryFrames.push({ ...frame, payloadText: storedPayload });
      if (this.sensoryFrames.length > 64) this.sensoryFrames.shift();
    }
    return verdict;
  }

  /** Dispatches an Observer action impulse to a Published Prism, enforcing boundary & ownership. */
  dispatchActionImpulse(impulse: ObserverActionImpulse): BoundaryEvaluationResult {
    const foreigns = Object.values(this.user.compartments).filter(
      (c) => c.compartmentId !== impulse.targetCompartmentId,
    );
    const verdict = evaluateContextBoundary({
      sourceCompartmentId: impulse.sourceCompartmentId,
      targetCompartmentId: impulse.targetCompartmentId,
      destination:
        impulse.sourceCompartmentId === impulse.targetCompartmentId
          ? 'same-compartment'
          : 'foreign-compartment',
      label: 'internal',
      payloadText: typeof impulse.payload === 'string' ? impulse.payload : JSON.stringify(impulse.payload ?? ''),
      isOpaqueProjection: impulse.opaqueOnly,
      ownerApproved: impulse.ownerApproved,
      bytesMatchSeen: true,
      halted: this.user.halted,
      foreignCompartments: foreigns,
    });
    if (verdict.verdict === 'allow') {
      this.impulseHistory.push(impulse);
      if (this.impulseHistory.length > 64) this.impulseHistory.shift();
    }
    return verdict;
  }

  getRecentSensoryFrames(): readonly PrismSensoryFrame[] {
    return [...this.sensoryFrames];
  }

  getRecentImpulses(): readonly ObserverActionImpulse[] {
    return [...this.impulseHistory];
  }
}
