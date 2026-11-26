/**
 * Deterministic Verdant Standards Applicability Router (`src/engine/verdantStandardsRouter.ts`)
 *
 * Implements `REQ-D6` ("a part is served only when it is relevant and verified") and
 * `REQ-D11` ("relevance and differentiation are computed from cross-classification"):
 *
 * - Always routes every Verdant Prism through the 4 mandatory Core Models:
 *   1. Concept (ISO/PAS 19450 OPM + W3C PROV-O Stakeholders + ISO 29148)
 *   2. Architecture & System (ISO 42010 + UML Behavior + JSON Schema 2020-12)
 *   3. Security & Threat Model (STRIDE + OWASP ASVS 5.0.0)
 *   4. Design & Experience (ISO 9241-210 + W3C WCAG 2.2 + prism.design/1)
 * - Deterministically evaluates populated Core weights (domain, audience, data classification,
 *   jurisdictions, local AI, payments, capabilities, assurance level) to activate ONLY
 *   applicable Layered Regulatory & Domain Standards.
 * - Records every non-applicable standard cleanly in `excludedStandards` with its derived
 *   exclusion rationale so irrelevant compliance forms never clutter the Creator while audits
 *   remain 100% transparent.
 */

import {
  PRISM_THEME_PRESETS,
  checkDeclaration,
  type PrismDeclaration,
  type StudioApiOp,
  type StudioAuthority,
  type StudioWizardStage,
} from './verdantStudioApi';

export type StageProgressStatus = 'empty' | 'in-progress' | 'complete';

export type StandardFacet =
  | 'concept'
  | 'architecture'
  | 'security'
  | 'design'
  | 'privacy'
  | 'ai-governance'
  | 'child-safety'
  | 'compliance'
  | 'operations'
  | 'domain';

export interface TrustBoundaryDecl {
  readonly id: string;
  readonly name: string;
  readonly kind: 'input' | 'sandbox' | 'storage' | 'network' | 'ai-stream';
  readonly active: boolean;
  readonly summary: string;
}

export interface DesignAccessibilityCheck {
  readonly id: string;
  readonly label: string;
  readonly wcagRef: string;
  readonly passed: boolean;
  readonly detail: string;
}

export interface StandardObligation {
  readonly id: string;
  readonly title: string;
  readonly detail: string;
  readonly settled: boolean;
  readonly autoSatisfied: boolean;
  readonly evidence?: string;
  readonly suggestedOp?: StudioApiOp;
}

export interface RoutedStandardEntry {
  readonly authority: StudioAuthority;
  readonly tier: 'core' | 'layered' | 'domain-pack';
  readonly facet: StandardFacet;
  readonly applicable: true;
  readonly triggeredBy: readonly string[];
  readonly reason: string;
  readonly obligations: readonly StandardObligation[];
  readonly settledCount: number;
  readonly totalCount: number;
}

export interface ExcludedStandardEntry {
  readonly authority: StudioAuthority;
  readonly tier: 'layered' | 'domain-pack';
  readonly facet: StandardFacet;
  readonly applicable: false;
  readonly exclusionRationale: string;
}

export interface WizardStageReport {
  readonly stage: StudioWizardStage;
  readonly number: number;
  readonly label: string;
  readonly shortLabel: string;
  readonly standardsLabel: string;
  readonly status: StageProgressStatus;
  readonly completedItems: number;
  readonly totalItems: number;
  readonly missing: readonly string[];
  readonly visible: boolean;
}

export interface VerdantRoutingReport {
  readonly isUndefined: boolean;
  readonly coreSpine: readonly RoutedStandardEntry[];
  readonly activeLayeredStandards: readonly RoutedStandardEntry[];
  readonly excludedStandards: readonly ExcludedStandardEntry[];
  readonly trustBoundaries: readonly TrustBoundaryDecl[];
  readonly designChecks: readonly DesignAccessibilityCheck[];
  readonly stages: readonly WizardStageReport[];
  readonly readyToPreview: boolean;
  readonly readyToSeal: boolean;
  readonly recommendedStage: StudioWizardStage;
  readonly summary: {
    readonly coreComplete: number;
    readonly coreTotal: 4;
    readonly activeLayeredCount: number;
    readonly obligationsSettled: number;
    readonly obligationsTotal: number;
    readonly excludedCount: number;
  };
}

// ============================================================================
// WCAG 2.2 Relative Luminance & Contrast Calculation (Deterministic, Zero Deps)
// ============================================================================

function parseHexRgb(hex: string): [number, number, number] | null {
  const clean = hex.trim().replace(/^#/, '');
  if (/^[0-9a-f]{3}$/i.test(clean)) {
    return [
      Number.parseInt(clean[0] + clean[0], 16),
      Number.parseInt(clean[1] + clean[1], 16),
      Number.parseInt(clean[2] + clean[2], 16),
    ];
  }
  if (/^[0-9a-f]{6}$/i.test(clean)) {
    return [
      Number.parseInt(clean.slice(0, 2), 16),
      Number.parseInt(clean.slice(2, 4), 16),
      Number.parseInt(clean.slice(4, 6), 16),
    ];
  }
  return null;
}

function relativeLuminance(rgb: [number, number, number]): number {
  const lin = rgb.map((c) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
}

export function wcagContrastRatio(hexA: string, hexB: string): number {
  const a = parseHexRgb(hexA);
  const b = parseHexRgb(hexB);
  if (!a || !b) return 1;
  const la = relativeLuminance(a);
  const lb = relativeLuminance(b);
  const lighter = Math.max(la, lb);
  const darker = Math.min(la, lb);
  return (lighter + 0.05) / (darker + 0.05);
}

export function evaluateDesignAccessibility(decl: PrismDeclaration): readonly DesignAccessibilityCheck[] {
  const preset = PRISM_THEME_PRESETS.find((p) => p.id === decl.theme.preset) ?? PRISM_THEME_PRESETS[0];
  const surface = preset.swatches[0];
  const foreground = decl.theme.accent ?? preset.swatches[1];
  const ratio = wcagContrastRatio(surface, foreground);
  const contrastPassed = ratio >= 4.5;

  const hasView = decl.view !== null;
  const fields = decl.encoding?.fields ?? [];
  const boundSet = new Set((decl.view?.elements ?? []).map((e) => e.binding));
  const allFieldsBound = hasView && fields.length > 0 && fields.every((f) => boundSet.has(f.name));
  const reqIds = new Set(decl.requirements.map((r) => r.id));
  const allElementsTraced =
    hasView &&
    (decl.view?.elements.length ?? 0) > 0 &&
    (decl.view?.elements.every((e) => reqIds.has(e.requirementId)) ?? false);
  const refusalsTextual = decl.refusals.every((r) => r.failureText.trim().length > 0);

  return [
    {
      id: 'wcag-contrast',
      label: 'WCAG AA Contrast Ratio (≥ 4.5:1)',
      wcagRef: 'WCAG 2.2 SC 1.4.3',
      passed: contrastPassed,
      detail: `${ratio.toFixed(2)}:1 between ${surface} surface and ${foreground} primary ink/accent.`,
    },
    {
      id: 'wcag-labels',
      label: 'All System Inputs Bound & Labeled',
      wcagRef: 'WCAG 2.2 SC 1.3.1 / 3.3.2',
      passed: allFieldsBound,
      detail: allFieldsBound
        ? `All ${fields.length} encoded input field(s) have labeled view controls.`
        : 'Synthesize or bind every System input field to a labeled UI control.',
    },
    {
      id: 'req-d9-trace',
      label: 'Bidirectional Requirement Traceability',
      wcagRef: 'ISO 42010 / REQ-D9',
      passed: allElementsTraced,
      detail: allElementsTraced
        ? `All ${decl.view?.elements.length ?? 0} UI element(s) trace to admitted requirements.`
        : 'Every UI element must cite an admitted Requirement.',
    },
    {
      id: 'wcag-refusals',
      label: 'Explicit Plain-Text Refusal Messages',
      wcagRef: 'WCAG 2.2 SC 1.4.1 / 3.3.1',
      passed: refusalsTextual,
      detail:
        decl.refusals.length > 0
          ? `${decl.refusals.length} typed refusal(s) convey errors in explicit text (never color alone).`
          : 'No refusals declared yet (valid refusals render explicit inline & alert text).',
    },
  ];
}

export function deriveTrustBoundaries(decl: PrismDeclaration): readonly TrustBoundaryDecl[] {
  return [
    {
      id: 'TB-INPUT',
      name: 'User Input -> WASM Guest',
      kind: 'input',
      active: (decl.encoding?.fields.length ?? 0) > 0 || decl.concept.opmObjects.some((o) => o.role === 'input'),
      summary: 'All user values cross a typed JSON Schema 2020-12 boundary into total Nat LexLean rules.',
    },
    {
      id: 'TB-SANDBOX',
      name: 'Zero-Origin View Sandbox',
      kind: 'sandbox',
      active: true,
      summary: 'Authored prism.design/1 view executes inside an isolated zero-origin iframe with zero rule JS.',
    },
    {
      id: 'TB-STORAGE',
      name: 'Scoped Local Storage',
      kind: 'storage',
      active: decl.capabilities.storage,
      summary: decl.capabilities.storage
        ? 'Local IndexedDB KV persistence enabled (scoped strictly to this Prism ID).'
        : 'Disabled (ephemeral in-memory evaluation only).',
    },
    {
      id: 'TB-NETWORK',
      name: 'Outbound Network Egress',
      kind: 'network',
      active: decl.capabilities.network,
      summary: decl.capabilities.network
        ? 'Explicitly granted — requires spoofing & exfiltration threat review.'
        : 'Denied by default (100% offline execution).',
    },
    {
      id: 'TB-AI',
      name: 'Local vTensor Model Stream',
      kind: 'ai-stream',
      active: decl.concept.usesLocalAi,
      summary: decl.concept.usesLocalAi
        ? 'Uses local WASM SIMD inference; deterministic rules remain strictly proved in LexLean.'
        : 'Not used — purely deterministic model.',
    },
  ];
}

// ============================================================================
// Layered Standards Catalog (Grounded in afflom/verdant-studio Authorities)
// ============================================================================

interface LayeredStandardSpec {
  readonly authority: StudioAuthority;
  readonly tier: 'layered' | 'domain-pack';
  readonly facet: StandardFacet;
  readonly evaluateTrigger: (decl: PrismDeclaration) =>
    | { readonly applicable: true; readonly triggeredBy: readonly string[]; readonly reason: string }
    | { readonly applicable: false; readonly exclusionRationale: string };
  readonly buildObligations: (decl: PrismDeclaration) => readonly StandardObligation[];
}

function hasJurisdiction(decl: PrismDeclaration, targets: readonly string[]): boolean {
  return decl.concept.jurisdictions.some((j) => targets.includes(j));
}

function isSettled(decl: PrismDeclaration, obligationId: string): boolean {
  return decl.settledObligations[obligationId]?.status === 'settled';
}

function nextReqId(decl: PrismDeclaration, preferred: string): string {
  if (!decl.requirements.some((r) => r.id === preferred)) return preferred;
  return `REQ-${String(decl.requirements.length + 1).padStart(2, '0')}`;
}

const LAYERED_STANDARD_SPECS: readonly LayeredStandardSpec[] = [
  // 1. GDPR (EU-REG-2016-679)
  {
    authority: {
      id: 'EU-REG-2016-679',
      name: 'GDPR (EU 2016/679)',
      edition: '2016',
      statement: 'Data minimization, lawful basis, purpose limitation, and data-subject transparency.',
    },
    tier: 'layered',
    facet: 'privacy',
    evaluateTrigger: (decl) => {
      const personal =
        decl.concept.dataClassification === 'personal' ||
        decl.concept.dataClassification === 'sensitive-regulated';
      const eu = hasJurisdiction(decl, ['eu', 'global']);
      if (personal && eu) {
        return {
          applicable: true,
          triggeredBy: [`dataClassification=${decl.concept.dataClassification}`, 'jurisdiction=EU/Global'],
          reason: 'Processes personal or regulated data for users in EU/Global jurisdictions.',
        };
      }
      return {
        applicable: false,
        exclusionRationale: !personal
          ? `Excluded: data classification is "${decl.concept.dataClassification ?? 'unset'}" (no personal data processed).`
          : 'Excluded: EU/Global jurisdiction is not selected in the Conceptual Model.',
      };
    },
    buildObligations: (decl) => {
      const auth: StudioAuthority = {
        id: 'EU-REG-2016-679',
        name: 'GDPR (EU 2016/679)',
        edition: '2016',
        statement: 'Data minimization, lawful basis, purpose limitation, and data-subject transparency.',
      };
      const reqId = nextReqId(decl, 'REQ-GDPR-01');
      const minSettled = isSettled(decl, 'GDPR-MINIMIZATION');
      const discSettled =
        isSettled(decl, 'GDPR-DISCLOSURE') ||
        Boolean(decl.view?.disclosure && decl.view.disclosure.toLowerCase().includes('local'));
      return [
        {
          id: 'GDPR-MINIMIZATION',
          title: 'Admit Data Minimization & Local-Processing Constraint',
          detail: 'Register a constraint requirement documenting minimal input fields and local-only evaluation.',
          settled: minSettled,
          autoSatisfied: false,
          evidence: decl.settledObligations['GDPR-MINIMIZATION']?.evidence,
          suggestedOp: {
            op: 'settle-standard-obligation',
            obligationId: 'GDPR-MINIMIZATION',
            authority: auth,
            evidence: `Registered ${reqId} restricting data processing to declared fields with client-only execution.`,
            linkedId: reqId,
            followUpOps: [
              {
                op: 'admit-requirement',
                requirement: {
                  id: reqId,
                  statement: 'Process only declared input fields strictly on-device with zero external data transfer.',
                  stakeholder: 'Data Subject (EU)',
                  origin: 'constraint',
                  evidence: 'EU-REG-2016-679 Art. 5(1)(c) data minimization',
                },
              },
            ],
          },
        },
        {
          id: 'GDPR-DISCLOSURE',
          title: 'Attach Privacy Notice Disclosure to View',
          detail: 'Display a clear data-handling disclosure in the compiled Prism view footer.',
          settled: discSettled,
          autoSatisfied: !isSettled(decl, 'GDPR-DISCLOSURE') && discSettled,
          evidence:
            decl.settledObligations['GDPR-DISCLOSURE']?.evidence ??
            (discSettled ? 'View disclosure text states local processing.' : undefined),
          suggestedOp: decl.view
            ? {
                op: 'settle-standard-obligation',
                obligationId: 'GDPR-DISCLOSURE',
                authority: auth,
                evidence: 'Attached GDPR Art. 13 on-device privacy disclosure to compiled View.',
                followUpOps: [
                  {
                    op: 'declare-view',
                    view: {
                      ...decl.view,
                      disclosure:
                        'Privacy Notice (GDPR): Inputs are evaluated locally on your device by a verified WASM model and are not transmitted to external servers.',
                    },
                  },
                ],
              }
            : {
                op: 'settle-standard-obligation',
                obligationId: 'GDPR-DISCLOSURE',
                authority: auth,
                evidence: 'Recorded GDPR Art. 13 disclosure requirement for View synthesis.',
              },
        },
      ];
    },
  },

  // 2. COPPA (US-CFR-16-312-2026)
  {
    authority: {
      id: 'US-CFR-16-312-2026',
      name: 'COPPA Rule (16 CFR Part 312)',
      edition: '2026',
      statement: 'Child data collection limits, notice, and verifiable parental consent.',
    },
    tier: 'layered',
    facet: 'child-safety',
    evaluateTrigger: (decl) => {
      const minors = decl.concept.audience === 'includes-minors';
      const us = hasJurisdiction(decl, ['us', 'global']);
      if (minors && us) {
        return {
          applicable: true,
          triggeredBy: ['audience=includes-minors', 'jurisdiction=US/Global'],
          reason: 'Audience includes minors within US/Global jurisdiction.',
        };
      }
      return {
        applicable: false,
        exclusionRationale: !minors
          ? `Excluded: audience is "${decl.concept.audience ?? 'unset'}" (not directed to minors).`
          : 'Excluded: US/Global jurisdiction is not selected.',
      };
    },
    buildObligations: (decl) => {
      const auth: StudioAuthority = {
        id: 'US-CFR-16-312-2026',
        name: 'COPPA Rule (16 CFR Part 312)',
        edition: '2026',
        statement: 'Child data collection limits, notice, and verifiable parental consent.',
      };
      const reqId = nextReqId(decl, 'REQ-COPPA-01');
      const noNetwork = !decl.capabilities.network;
      const settled = isSettled(decl, 'COPPA-ZERO-COLLECTION');
      return [
        {
          id: 'COPPA-ZERO-COLLECTION',
          title: 'Enforce Zero Remote Child Data Collection',
          detail: 'Keep outbound network denied and admit a COPPA constraint requirement prohibiting remote PII collection.',
          settled: settled && noNetwork,
          autoSatisfied: false,
          evidence: decl.settledObligations['COPPA-ZERO-COLLECTION']?.evidence,
          suggestedOp: {
            op: 'settle-standard-obligation',
            obligationId: 'COPPA-ZERO-COLLECTION',
            authority: auth,
            evidence: `Network denied and ${reqId} admitted under 16 CFR 312.`,
            linkedId: reqId,
            followUpOps: [
              { op: 'set-capabilities', capabilities: { network: false } },
              {
                op: 'admit-requirement',
                requirement: {
                  id: reqId,
                  statement: 'Prohibit remote collection or transmission of personal information from minors.',
                  stakeholder: 'Parent / Guardian',
                  origin: 'constraint',
                  evidence: 'US-CFR-16-312-2026 COPPA compliance boundary',
                },
              },
            ],
          },
        },
      ];
    },
  },

  // 3. UK Children's Code (ICO-AADC-STANDARDS)
  {
    authority: {
      id: 'ICO-AADC-STANDARDS',
      name: "UK Children's Code (ICO AADC)",
      edition: '2021',
      statement: 'High-privacy defaults, data minimization, and age-appropriate transparency for minors.',
    },
    tier: 'layered',
    facet: 'child-safety',
    evaluateTrigger: (decl) => {
      const minors = decl.concept.audience === 'includes-minors';
      const uk = hasJurisdiction(decl, ['uk', 'global']);
      if (minors && uk) {
        return {
          applicable: true,
          triggeredBy: ['audience=includes-minors', 'jurisdiction=UK/Global'],
          reason: 'Audience includes minors within UK/Global jurisdiction.',
        };
      }
      return {
        applicable: false,
        exclusionRationale: !minors
          ? 'Excluded: audience does not include minors.'
          : 'Excluded: UK/Global jurisdiction is not selected.',
      };
    },
    buildObligations: (decl) => {
      const auth: StudioAuthority = {
        id: 'ICO-AADC-STANDARDS',
        name: "UK Children's Code (ICO AADC)",
        edition: '2021',
        statement: 'High-privacy defaults, data minimization, and age-appropriate transparency for minors.',
      };
      const highPrivacy = !decl.capabilities.network && !decl.capabilities.storage;
      const settled = isSettled(decl, 'AADC-HIGH-PRIVACY') || highPrivacy;
      return [
        {
          id: 'AADC-HIGH-PRIVACY',
          title: 'High-Privacy Default Posture',
          detail: 'Keep network and persistent storage off by default unless explicitly justified.',
          settled,
          autoSatisfied: !isSettled(decl, 'AADC-HIGH-PRIVACY') && highPrivacy,
          evidence:
            decl.settledObligations['AADC-HIGH-PRIVACY']?.evidence ??
            (highPrivacy ? 'Satisfied structurally: network and storage are denied by default.' : undefined),
          suggestedOp: {
            op: 'settle-standard-obligation',
            obligationId: 'AADC-HIGH-PRIVACY',
            authority: auth,
            evidence: 'Set network=false and storage=false for high-privacy default.',
            followUpOps: [{ op: 'set-capabilities', capabilities: { network: false, storage: false } }],
          },
        },
      ];
    },
  },

  // 4. EU AI Act (EU-REG-2024-1689)
  {
    authority: {
      id: 'EU-REG-2024-1689',
      name: 'EU AI Act (EU 2024/1689)',
      edition: '2024',
      statement: 'AI system transparency, human oversight, and deterministic safety boundaries.',
    },
    tier: 'layered',
    facet: 'ai-governance',
    evaluateTrigger: (decl) => {
      const ai = decl.concept.usesLocalAi;
      const eu = hasJurisdiction(decl, ['eu', 'global']);
      if (ai && eu) {
        return {
          applicable: true,
          triggeredBy: ['usesLocalAi=true', 'jurisdiction=EU/Global'],
          reason: 'Uses local AI inference for users in EU/Global jurisdictions.',
        };
      }
      return {
        applicable: false,
        exclusionRationale: !ai
          ? 'Excluded: Prism does not use AI inference (pure deterministic rules).'
          : 'Excluded: EU/Global jurisdiction is not selected.',
      };
    },
    buildObligations: (decl) => {
      const auth: StudioAuthority = {
        id: 'EU-REG-2024-1689',
        name: 'EU AI Act (EU 2024/1689)',
        edition: '2024',
        statement: 'AI system transparency, human oversight, and deterministic safety boundaries.',
      };
      const reqId = nextReqId(decl, 'REQ-AI-01');
      const settled = isSettled(decl, 'EU-AI-TRANSPARENCY');
      return [
        {
          id: 'EU-AI-TRANSPARENCY',
          title: 'AI Transparency & Proved Guardrail Boundary',
          detail: 'Document AI assistance clearly and guarantee deterministic LexLean rules enforce final safety bounds.',
          settled,
          autoSatisfied: false,
          evidence: decl.settledObligations['EU-AI-TRANSPARENCY']?.evidence,
          suggestedOp: {
            op: 'settle-standard-obligation',
            obligationId: 'EU-AI-TRANSPARENCY',
            authority: auth,
            evidence: `Admitted ${reqId} enforcing EU AI Act Art. 50 transparency & deterministic boundary.`,
            linkedId: reqId,
            followUpOps: [
              {
                op: 'admit-requirement',
                requirement: {
                  id: reqId,
                  statement: 'Disclose local AI assistance clearly while keeping all enforceable decisions inside proved rules.',
                  stakeholder: 'Deployer / End User',
                  origin: 'constraint',
                  evidence: 'EU-REG-2024-1689 transparency & human oversight',
                },
              },
              ...(decl.view
                ? [
                    {
                      op: 'declare-view' as const,
                      view: {
                        ...decl.view,
                        disclosure:
                          'AI Transparency Notice (EU AI Act): Optional local AI assistance is bounded by proved deterministic rules.',
                      },
                    },
                  ]
                : []),
            ],
          },
        },
      ];
    },
  },

  // 5. NIST AI RMF 1.0 (NIST-AI-RMF-PLAYBOOK)
  {
    authority: {
      id: 'NIST-AI-RMF-PLAYBOOK',
      name: 'NIST AI RMF 1.0',
      edition: '1.0',
      statement: 'Govern, Map, Measure, and Manage AI risk with deterministic verification boundaries.',
    },
    tier: 'layered',
    facet: 'ai-governance',
    evaluateTrigger: (decl) => {
      if (decl.concept.usesLocalAi) {
        return {
          applicable: true,
          triggeredBy: ['usesLocalAi=true'],
          reason: 'Local AI model stream is enabled in the Conceptual Model.',
        };
      }
      return {
        applicable: false,
        exclusionRationale: 'Excluded: usesLocalAi is false (deterministic rule evaluation only).',
      };
    },
    buildObligations: (decl) => {
      const auth: StudioAuthority = {
        id: 'NIST-AI-RMF-PLAYBOOK',
        name: 'NIST AI RMF 1.0',
        edition: '1.0',
        statement: 'Govern, Map, Measure, and Manage AI risk with deterministic verification boundaries.',
      };
      const settled = isSettled(decl, 'NIST-AI-RMF-BOUNDARY');
      return [
        {
          id: 'NIST-AI-RMF-BOUNDARY',
          title: 'Map & Manage Deterministic Fallback',
          detail: 'Confirm that all critical outputs and refusals are governed by proved LexLean rules rather than probabilistic generation.',
          settled,
          autoSatisfied: false,
          evidence: decl.settledObligations['NIST-AI-RMF-BOUNDARY']?.evidence,
          suggestedOp: {
            op: 'settle-standard-obligation',
            obligationId: 'NIST-AI-RMF-BOUNDARY',
            authority: auth,
            evidence: 'Confirmed all normative verdicts & refusals compile to Lean-proved WASM guest rules.',
          },
        },
      ];
    },
  },

  // 6. Cyber Resilience Act (EU-REG-2024-2847)
  {
    authority: {
      id: 'EU-REG-2024-2847',
      name: 'EU Cyber Resilience Act (CRA)',
      edition: '2024',
      statement: 'Security-by-default digital products, vulnerability handling, and reproducible integrity.',
    },
    tier: 'layered',
    facet: 'compliance',
    evaluateTrigger: (decl) => {
      const pub = decl.concept.audience === 'public-adults' || decl.concept.audience === 'includes-minors';
      const eu = hasJurisdiction(decl, ['eu', 'global']);
      if (pub && eu) {
        return {
          applicable: true,
          triggeredBy: [`audience=${decl.concept.audience}`, 'jurisdiction=EU/Global'],
          reason: 'Publicly distributed digital product targeting EU/Global users.',
        };
      }
      return {
        applicable: false,
        exclusionRationale: !pub
          ? `Excluded: audience is "${decl.concept.audience ?? 'unset'}" (internal/personal use, not public market).`
          : 'Excluded: EU/Global jurisdiction is not selected.',
      };
    },
    buildObligations: (decl) => {
      const auth: StudioAuthority = {
        id: 'EU-REG-2024-2847',
        name: 'EU Cyber Resilience Act (CRA)',
        edition: '2024',
        statement: 'Security-by-default digital products, vulnerability handling, and reproducible integrity.',
      };
      const allThreatsSettled =
        decl.security.threats.length > 0 && decl.security.threats.every((t) => t.status !== 'open');
      const settled = isSettled(decl, 'CRA-SECURE-DEFAULT') || (allThreatsSettled && !decl.capabilities.network);
      return [
        {
          id: 'CRA-SECURE-DEFAULT',
          title: 'Secure-by-Default Posture & Settled Threat Model',
          detail: 'All STRIDE threats settled and sealed into a BLAKE3-verified .holo archive.',
          settled,
          autoSatisfied: !isSettled(decl, 'CRA-SECURE-DEFAULT') && settled,
          evidence:
            decl.settledObligations['CRA-SECURE-DEFAULT']?.evidence ??
            (settled ? 'All STRIDE threats settled with deny-by-default network isolation.' : undefined),
          suggestedOp: {
            op: 'settle-standard-obligation',
            obligationId: 'CRA-SECURE-DEFAULT',
            authority: auth,
            evidence: 'Verified secure-by-default capabilities and BLAKE3 .holo provenance.',
          },
        },
      ];
    },
  },

  // 7. EU Digital Content Directive (EU-DIR-2019-770)
  {
    authority: {
      id: 'EU-DIR-2019-770',
      name: 'EU Digital Content Directive',
      edition: '2019',
      statement: 'Objective & subjective conformity of supplied digital content and services.',
    },
    tier: 'layered',
    facet: 'compliance',
    evaluateTrigger: (decl) => {
      const pub = decl.concept.audience === 'public-adults' || decl.concept.audience === 'includes-minors';
      const eu = hasJurisdiction(decl, ['eu', 'global']);
      if (pub && eu) {
        return {
          applicable: true,
          triggeredBy: [`audience=${decl.concept.audience}`, 'jurisdiction=EU/Global'],
          reason: 'Public digital content supplied to EU/Global consumers.',
        };
      }
      return {
        applicable: false,
        exclusionRationale: 'Excluded: not a public consumer digital service in EU/Global jurisdiction.',
      };
    },
    buildObligations: (decl) => {
      const check = checkDeclaration(decl);
      const conforms = check.vectorsTotal >= 2 && check.vectorsPassed === check.vectorsTotal;
      return [
        {
          id: 'DCD-CONFORMITY-VECTORS',
          title: 'Executable Conformity Examples',
          detail: 'All declared acceptance vectors pass against the compiled model (auto-verified).',
          settled: conforms,
          autoSatisfied: conforms,
          evidence: conforms
            ? `${check.vectorsPassed}/${check.vectorsTotal} acceptance vectors verified.`
            : undefined,
        },
      ];
    },
  },

  // 8. PSD2 (EU-DIR-2015-2366)
  {
    authority: {
      id: 'EU-DIR-2015-2366',
      name: 'PSD2 Payment Services Directive',
      edition: '2015',
      statement: 'Payment validation, explicit amount/payer guardrails, and refusal of invalid transactions.',
    },
    tier: 'layered',
    facet: 'compliance',
    evaluateTrigger: (decl) => {
      if (decl.concept.involvesPayments) {
        return {
          applicable: true,
          triggeredBy: ['involvesPayments=true'],
          reason: 'Conceptual Model declares payment or monetary settlement flows.',
        };
      }
      return {
        applicable: false,
        exclusionRationale: 'Excluded: involvesPayments is false.',
      };
    },
    buildObligations: (decl) => {
      const auth: StudioAuthority = {
        id: 'EU-DIR-2015-2366',
        name: 'PSD2 Payment Services Directive',
        edition: '2015',
        statement: 'Payment validation, explicit amount/payer guardrails, and refusal of invalid transactions.',
      };
      const hasRefusal = decl.refusals.length > 0;
      const settled = isSettled(decl, 'PSD2-REFUSAL-GUARD');
      const reqId = nextReqId(decl, 'REQ-PAY-01');
      return [
        {
          id: 'PSD2-REFUSAL-GUARD',
          title: 'Payment Boundary Validation & Explicit Refusal',
          detail: 'Admit payment validation constraint and ensure at least one typed refusal guards invalid payment amounts.',
          settled: settled && hasRefusal,
          autoSatisfied: false,
          evidence: decl.settledObligations['PSD2-REFUSAL-GUARD']?.evidence,
          suggestedOp: {
            op: 'settle-standard-obligation',
            obligationId: 'PSD2-REFUSAL-GUARD',
            authority: auth,
            evidence: `Registered ${reqId} for deterministic payment input refusal guard.`,
            linkedId: reqId,
            followUpOps: [
              {
                op: 'admit-requirement',
                requirement: {
                  id: reqId,
                  statement: 'Reject zero, malformed, or out-of-policy payment requests with an explicit typed refusal.',
                  stakeholder: 'Payment Service User',
                  origin: 'constraint',
                  evidence: 'EU-DIR-2015-2366 payment integrity guardrail',
                },
              },
            ],
          },
        },
      ];
    },
  },

  // 9. European Accessibility Standard (ETSI-EN-301-549-3.2.1)
  {
    authority: {
      id: 'ETSI-EN-301-549-3.2.1',
      name: 'EN 301 549 Accessibility Standard',
      edition: '3.2.1',
      statement: 'Functional accessibility performance aligned with WCAG 2.2 AA for public ICT products.',
    },
    tier: 'layered',
    facet: 'design',
    evaluateTrigger: (decl) => {
      const pub = decl.concept.audience === 'public-adults' || decl.concept.audience === 'includes-minors';
      const eu = hasJurisdiction(decl, ['eu', 'global']);
      if (pub && eu) {
        return {
          applicable: true,
          triggeredBy: [`audience=${decl.concept.audience}`, 'jurisdiction=EU/Global'],
          reason: 'Public-facing UI targeting EU/Global users.',
        };
      }
      return {
        applicable: false,
        exclusionRationale: 'Excluded: internal/personal Prism or non-EU/Global scope (baseline WCAG 2.2 still checked in Core Design).',
      };
    },
    buildObligations: (decl) => {
      const checks = evaluateDesignAccessibility(decl);
      const allPass = checks.every((c) => c.passed);
      return [
        {
          id: 'EN301549-WCAG-AA',
          title: 'WCAG 2.2 AA Automated View Baseline',
          detail: 'Contrast ≥ 4.5:1, labeled controls, requirement traceability, and text refusals (auto-checked).',
          settled: allPass,
          autoSatisfied: allPass,
          evidence: allPass ? 'All 4 WCAG 2.2 AA baseline checks passing.' : undefined,
        },
      ];
    },
  },

  // 10. CycloneDX Supply Chain Provenance (ECMA-424-CYCLONEDX-1.6)
  {
    authority: {
      id: 'ECMA-424-CYCLONEDX-1.6',
      name: 'CycloneDX 1.6 Provenance / BOM',
      edition: '1.6',
      statement: 'Component, authority, and provenance attestation for elevated security assurance tiers.',
    },
    tier: 'layered',
    facet: 'security',
    evaluateTrigger: (decl) => {
      const elevated = decl.security.assuranceLevel === 'L2' || decl.security.assuranceLevel === 'L3';
      if (elevated) {
        return {
          applicable: true,
          triggeredBy: [`assuranceLevel=${decl.security.assuranceLevel}`],
          reason: `Elevated OWASP ASVS ${decl.security.assuranceLevel} assurance tier requires explicit authority & archive provenance.`,
        };
      }
      return {
        applicable: false,
        exclusionRationale: 'Excluded: OWASP ASVS L1 baseline does not require CycloneDX BOM attestation.',
      };
    },
    buildObligations: (decl) => {
      const hasAuthorities = decl.authorities.length > 0 && decl.admissions.length > 0;
      return [
        {
          id: 'CYCLONEDX-ATTESTATION',
          title: 'Locked Authority & Archive Manifest',
          detail: 'Compiled .holo embeds cited authorities, BLAKE3 address, and model/standards-routing.json (auto-generated).',
          settled: hasAuthorities,
          autoSatisfied: hasAuthorities,
          evidence: hasAuthorities
            ? `${decl.authorities.length} authorities & ${decl.admissions.length} admitted constructs tracked.`
            : undefined,
        },
      ];
    },
  },

  // 11. OpenTelemetry Trace API (OTEL-SPEC-1.40.0-TRACE-API)
  {
    authority: {
      id: 'OTEL-SPEC-1.40.0-TRACE-API',
      name: 'OpenTelemetry Trace API 1.40',
      edition: '1.40.0',
      statement: 'Causal W3C Trace Context spans for live model invocations and studio mutations.',
    },
    tier: 'layered',
    facet: 'operations',
    evaluateTrigger: (decl) => {
      const multiUser =
        decl.concept.audience === 'team' ||
        decl.concept.audience === 'public-adults' ||
        decl.concept.audience === 'includes-minors';
      if (decl.capabilities.observer && multiUser && decl.encoding !== null) {
        return {
          applicable: true,
          triggeredBy: ['capabilities.observer=true', `audience=${decl.concept.audience}`],
          reason: 'Observer Stream tracing is enabled for a team or public Prism.',
        };
      }
      return {
        applicable: false,
        exclusionRationale: !multiUser
          ? 'Excluded: personal offline Prism uses built-in local trace view without team/public telemetry obligations.'
          : 'Excluded: Observer tracing is disabled or System Model is not yet defined.',
      };
    },
    buildObligations: () => [
      {
        id: 'OTEL-SPAN-BINDING',
        title: 'W3C Trace Context Observer Spans',
        detail: 'VerdantUI ObserverStreamBus automatically binds causal trace spans to every Prism invoke() call.',
        settled: true,
        autoSatisfied: true,
        evidence: 'Automatically satisfied by VerdantUI ObserverStreamBus.',
      },
    ],
  },

  // 12. Domain Pack: Games & Interactive Media (HUNICKE-MDA-2004 & ESRB-RATINGS-GUIDE)
  {
    authority: {
      id: 'HUNICKE-MDA-2004',
      name: 'MDA Framework & ESRB Content Disclosure',
      edition: '2004',
      statement: 'Mechanics–Dynamics–Aesthetics alignment and age/content interactivity disclosure for games & media.',
    },
    tier: 'domain-pack',
    facet: 'domain',
    evaluateTrigger: (decl) => {
      if (decl.concept.domain === 'games-media') {
        return {
          applicable: true,
          triggeredBy: ['domain=games-media'],
          reason: 'Conceptual Model domain is Games & Interactive Media.',
        };
      }
      return {
        applicable: false,
        exclusionRationale: `Excluded: domain is "${decl.concept.domain ?? 'unset'}" (Games & Media domain pack not triggered).`,
      };
    },
    buildObligations: (decl) => {
      const auth: StudioAuthority = {
        id: 'HUNICKE-MDA-2004',
        name: 'MDA Framework & ESRB Content Disclosure',
        edition: '2004',
        statement: 'Mechanics–Dynamics–Aesthetics alignment and age/content interactivity disclosure for games & media.',
      };
      const reqId = nextReqId(decl, 'REQ-MDA-01');
      const settled = isSettled(decl, 'GAMES-MDA-AESTHETIC');
      return [
        {
          id: 'GAMES-MDA-AESTHETIC',
          title: 'Declare Intended Player Aesthetic & Content Rating Scope',
          detail: 'Link the Prism mechanics to an explicit player experience goal and age-appropriateness statement.',
          settled,
          autoSatisfied: false,
          evidence: decl.settledObligations['GAMES-MDA-AESTHETIC']?.evidence,
          suggestedOp: {
            op: 'settle-standard-obligation',
            obligationId: 'GAMES-MDA-AESTHETIC',
            authority: auth,
            evidence: `Admitted ${reqId} linking declared mechanics to player aesthetic & content scope.`,
            linkedId: reqId,
            followUpOps: [
              {
                op: 'admit-requirement',
                requirement: {
                  id: reqId,
                  statement: 'Align deterministic game mechanics to clear player feedback and declared age/content scope.',
                  stakeholder: 'Player',
                  origin: 'identity',
                  evidence: 'HUNICKE-MDA-2004 / ESRB-RATINGS-GUIDE domain pack',
                },
              },
            ],
          },
        },
      ];
    },
  },
];

// ============================================================================
// Main Deterministic Routing & Completeness Evaluator
// ============================================================================

export function isPrismSynthesizable(decl: PrismDeclaration): boolean {
  return (
    decl.encoding !== null &&
    decl.encoding.fields.length > 0 &&
    decl.features.length > 0 &&
    decl.view !== null &&
    decl.view.elements.length > 0
  );
}

export function evaluateVerdantRouting(decl: PrismDeclaration): VerdantRoutingReport {
  const check = checkDeclaration(decl);
  const trustBoundaries = deriveTrustBoundaries(decl);
  const designChecks = evaluateDesignAccessibility(decl);

  // Determine whether literally nothing is defined yet ("+ New Prism" initial state)
  const isUndefined =
    !decl.concept.purpose.trim() &&
    decl.concept.domain === null &&
    decl.concept.audience === null &&
    decl.concept.stakeholders.length === 0 &&
    decl.concept.opmObjects.length === 0 &&
    decl.concept.opmProcesses.length === 0 &&
    decl.encoding === null &&
    decl.features.length === 0 &&
    decl.refusals.length === 0 &&
    decl.view === null;

  // --- Stage 1: Concept (OPM + Stakeholders + Context Weights) ---
  const conceptMissing: string[] = [];
  if (!decl.concept.purpose.trim()) conceptMissing.push('Define the Prism purpose / problem statement');
  if (!decl.concept.domain) conceptMissing.push('Select a domain');
  if (!decl.concept.audience) conceptMissing.push('Select the target audience');
  if (!decl.concept.dataClassification) conceptMissing.push('Select data classification');
  if (decl.concept.stakeholders.length === 0) conceptMissing.push('Add at least one PROV-O Stakeholder');
  if (decl.concept.opmObjects.length === 0) conceptMissing.push('Add at least one OPM Object (ISO 19450)');
  if (decl.concept.opmProcesses.length === 0) conceptMissing.push('Add at least one OPM Process (ISO 19450)');
  const conceptTotal = 7;
  const conceptDone = conceptTotal - conceptMissing.length;
  const conceptStatus: StageProgressStatus =
    conceptDone === 0 ? 'empty' : conceptMissing.length === 0 ? 'complete' : 'in-progress';

  // --- Stage 2: Architecture & System (ISO 42010 / UML / JSON Schema / Behavior) ---
  const archMissing: string[] = [];
  if (!decl.encoding || decl.encoding.fields.length === 0) archMissing.push('Declare or derive typed input fields');
  if (decl.features.length === 0) archMissing.push('Add at least one deterministic decision rule or formula');
  if (decl.vectors.length === 0) archMissing.push('Add at least one acceptance example');
  if (decl.vectors.length > 0 && check.vectorsPassed < check.vectorsTotal) {
    archMissing.push(`Fix ${check.vectorsTotal - check.vectorsPassed} failing acceptance example(s)`);
  }
  const unreplayedRefusals = check.diagnostics.filter((d) => d.code === 'contract.refusal-unreplayed');
  if (unreplayedRefusals.length > 0) {
    archMissing.push(`Add rejecting example(s) for ${unreplayedRefusals.length} refusal(s) (R5)`);
  }
  const archTotal = 5;
  const archDone = Math.max(0, archTotal - archMissing.length);
  const archEmpty =
    decl.encoding === null && decl.features.length === 0 && decl.refusals.length === 0 && decl.vectors.length === 0;
  const archStatus: StageProgressStatus = archEmpty
    ? 'empty'
    : archMissing.length === 0
      ? 'complete'
      : 'in-progress';

  // --- Stage 3: Security & Threat Model (STRIDE + OWASP ASVS) ---
  const secMissing: string[] = [];
  if (decl.security.threats.length === 0) {
    secMissing.push('Sync STRIDE threats from the System Boundary');
  } else {
    const openThreats = decl.security.threats.filter((t) => t.status === 'open');
    if (openThreats.length > 0) {
      secMissing.push(`Settle ${openThreats.length} open STRIDE threat(s)`);
    }
  }
  const secTotal = Math.max(1, decl.security.threats.length);
  const secDone =
    decl.security.threats.length === 0
      ? 0
      : decl.security.threats.filter((t) => t.status !== 'open').length;
  const secStatus: StageProgressStatus =
    decl.security.threats.length === 0
      ? 'empty'
      : secMissing.length === 0
        ? 'complete'
        : 'in-progress';

  // --- Stage 4: Design & Experience (ISO 9241 / WCAG 2.2 / prism.design/1) ---
  const designMissing: string[] = [];
  if (!decl.view) {
    designMissing.push('Synthesize or declare the prism.design/1 View');
  } else {
    for (const c of designChecks) {
      if (!c.passed) designMissing.push(`${c.label} (${c.wcagRef})`);
    }
  }
  const designTotal = designChecks.length;
  const designDone = !decl.view ? 0 : designChecks.filter((c) => c.passed).length;
  const designStatus: StageProgressStatus =
    !decl.view ? 'empty' : designMissing.length === 0 ? 'complete' : 'in-progress';

  // --- Core Spine Summary ---
  const coreSpine: RoutedStandardEntry[] = [
    {
      authority: {
        id: 'ISO-PAS-19450',
        name: 'ISO 19450 OPM + PROV-O + ISO 29148',
        edition: 'Core 1',
        statement: 'Conceptual Objects, Processes, States, Stakeholders, and Context weights.',
      },
      tier: 'core',
      facet: 'concept',
      applicable: true,
      triggeredBy: ['Mandatory Verdant Prism Core Spine'],
      reason: 'Every Verdant Prism requires a complete Conceptual & Stakeholder model.',
      obligations: [],
      settledCount: conceptDone,
      totalCount: conceptTotal,
    },
    {
      authority: {
        id: 'ISO-IEC-IEEE-42010-2022',
        name: 'ISO 42010 + UML + JSON Schema 2020-12',
        edition: 'Core 2',
        statement: 'System boundary, typed encodings, pure deterministic rules, refusals, and vectors.',
      },
      tier: 'core',
      facet: 'architecture',
      applicable: true,
      triggeredBy: ['Mandatory Verdant Prism Core Spine'],
      reason: 'Every Verdant Prism compiles its Architecture & Behavior into proved LexLean.',
      obligations: [],
      settledCount: archDone,
      totalCount: archTotal,
    },
    {
      authority: {
        id: 'MICROSOFT-TMT-STRIDE',
        name: `STRIDE + OWASP ASVS ${decl.security.assuranceLevel}`,
        edition: 'Core 3',
        statement: 'Explicit trust boundaries and settled threat dispositions.',
      },
      tier: 'core',
      facet: 'security',
      applicable: true,
      triggeredBy: ['Mandatory Verdant Prism Core Spine'],
      reason: 'Every Verdant Prism documents and mitigates its active threat surface.',
      obligations: [],
      settledCount: secDone,
      totalCount: secTotal,
    },
    {
      authority: {
        id: 'W3C-WCAG22-2024',
        name: 'ISO 9241-210 + WCAG 2.2 + prism.design/1',
        edition: 'Core 4',
        statement: 'Requirement-traced views, accessible contrast, and zero-rule page compilation.',
      },
      tier: 'core',
      facet: 'design',
      applicable: true,
      triggeredBy: ['Mandatory Verdant Prism Core Spine'],
      reason: 'Every Verdant Prism compiles a verified, accessible, traceable interface.',
      obligations: [],
      settledCount: designDone,
      totalCount: designTotal,
    },
  ];

  // --- Stage 5: Deterministic Applicability Routing over Layered Standards ---
  const activeLayeredStandards: RoutedStandardEntry[] = [];
  const excludedStandards: ExcludedStandardEntry[] = [];

  for (const spec of LAYERED_STANDARD_SPECS) {
    const res = spec.evaluateTrigger(decl);
    if (res.applicable) {
      const obligations = spec.buildObligations(decl);
      const settledCount = obligations.filter((o) => o.settled).length;
      activeLayeredStandards.push({
        authority: spec.authority,
        tier: spec.tier,
        facet: spec.facet,
        applicable: true,
        triggeredBy: res.triggeredBy,
        reason: res.reason,
        obligations,
        settledCount,
        totalCount: obligations.length,
      });
    } else {
      excludedStandards.push({
        authority: spec.authority,
        tier: spec.tier,
        facet: spec.facet,
        applicable: false,
        exclusionRationale: res.exclusionRationale,
      });
    }
  }

  const obligationsTotal = activeLayeredStandards.reduce((acc, s) => acc + s.totalCount, 0);
  const obligationsSettled = activeLayeredStandards.reduce((acc, s) => acc + s.settledCount, 0);
  const standardsMissing: string[] = [];
  for (const std of activeLayeredStandards) {
    for (const ob of std.obligations) {
      if (!ob.settled) {
        standardsMissing.push(`${std.authority.name}: ${ob.title}`);
      }
    }
  }
  const hasLayered = activeLayeredStandards.length > 0;
  const standardsStatus: StageProgressStatus = !hasLayered
    ? 'complete'
    : standardsMissing.length === 0
      ? 'complete'
      : obligationsSettled > 0
        ? 'in-progress'
        : 'empty';

  const coreCompleteCount = [conceptStatus, archStatus, secStatus, designStatus].filter(
    (s) => s === 'complete',
  ).length;
  const readyToPreview = isPrismSynthesizable(decl);
  const readyToSeal =
    coreCompleteCount === 4 && standardsMissing.length === 0 && check.ok;

  const sealMissing: string[] = [];
  if (conceptStatus !== 'complete') sealMissing.push('Complete Stage 1: Concept (OPM & Stakeholders)');
  if (archStatus !== 'complete') sealMissing.push('Complete Stage 2: System & Behavior');
  if (secStatus !== 'complete') sealMissing.push('Complete Stage 3: Security & Threat Model');
  if (designStatus !== 'complete') sealMissing.push('Complete Stage 4: Design & Accessibility');
  if (standardsMissing.length > 0) sealMissing.push(`Settle ${standardsMissing.length} routed standard obligation(s)`);
  if (!check.ok) {
    for (const diag of check.diagnostics.slice(0, 3)) {
      if (!sealMissing.includes(diag.message)) sealMissing.push(diag.message);
    }
  }

  const stages: WizardStageReport[] = [
    {
      stage: 'concept',
      number: 1,
      label: 'Concept & OPM',
      shortLabel: 'Concept',
      standardsLabel: 'ISO 19450 · PROV-O · ISO 29148',
      status: conceptStatus,
      completedItems: conceptDone,
      totalItems: conceptTotal,
      missing: conceptMissing,
      visible: true,
    },
    {
      stage: 'architecture',
      number: 2,
      label: 'System & Rules',
      shortLabel: 'System',
      standardsLabel: 'ISO 42010 · UML · JSON Schema',
      status: archStatus,
      completedItems: archDone,
      totalItems: archTotal,
      missing: archMissing,
      visible: true,
    },
    {
      stage: 'security',
      number: 3,
      label: 'Security & Threats',
      shortLabel: 'Security',
      standardsLabel: `STRIDE · OWASP ASVS ${decl.security.assuranceLevel}`,
      status: secStatus,
      completedItems: secDone,
      totalItems: secTotal,
      missing: secMissing,
      visible: true,
    },
    {
      stage: 'design',
      number: 4,
      label: 'Design & UX',
      shortLabel: 'Design',
      standardsLabel: 'ISO 9241-210 · WCAG 2.2 · prism.design/1',
      status: designStatus,
      completedItems: designDone,
      totalItems: designTotal,
      missing: designMissing,
      visible: true,
    },
    {
      stage: 'standards',
      number: 5,
      label: `Routed Standards (${activeLayeredStandards.length})`,
      shortLabel: `Standards (${activeLayeredStandards.length})`,
      standardsLabel: 'REQ-D6 · REQ-D11 Applicable Layers',
      status: standardsStatus,
      completedItems: obligationsSettled,
      totalItems: Math.max(1, obligationsTotal),
      missing: standardsMissing,
      visible: hasLayered,
    },
    {
      stage: 'seal',
      number: hasLayered ? 6 : 5,
      label: 'Verify & Seal',
      shortLabel: 'Seal',
      standardsLabel: 'prism-code · Lean 4.32.2 · .holo v6',
      status: readyToSeal ? 'complete' : readyToPreview ? 'in-progress' : 'empty',
      completedItems: coreCompleteCount + (standardsMissing.length === 0 ? 1 : 0),
      totalItems: 5,
      missing: sealMissing,
      visible: true,
    },
  ];

  const recommendedStage: StudioWizardStage =
    conceptStatus !== 'complete'
      ? 'concept'
      : archStatus !== 'complete'
        ? 'architecture'
        : secStatus !== 'complete'
          ? 'security'
          : designStatus !== 'complete'
            ? 'design'
            : hasLayered && standardsStatus !== 'complete'
              ? 'standards'
              : 'seal';

  return {
    isUndefined,
    coreSpine,
    activeLayeredStandards,
    excludedStandards,
    trustBoundaries,
    designChecks,
    stages,
    readyToPreview,
    readyToSeal,
    recommendedStage,
    summary: {
      coreComplete: coreCompleteCount,
      coreTotal: 4,
      activeLayeredCount: activeLayeredStandards.length,
      obligationsSettled,
      obligationsTotal,
      excludedCount: excludedStandards.length,
    },
  };
}
