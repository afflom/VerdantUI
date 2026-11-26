/**
 * Declarative Verdant Studio API (`src/engine/verdantStudioApi.ts`)
 *
 * Implements the closed, deterministic declarative programming patterns from
 * `github.com/afflom/verdant-studio` (`Studio.Catalog`, `Studio.Admission`,
 * `Studio.Schema`, `Studio.Check`, `Studio.Ops`):
 *
 * - All Prism changes flow strictly through closed API operations (`dispatchStudioOp`).
 * - Every construct enters with a mandatory `Citation` (`req` | `std` | `der`), verified
 *   by well-founded admission checks and monotone register locks (`Studio.Admission`).
 * - Data encodings enter via JSON Schema 2020-12 (`propose-encoding`) and compile
 *   mechanically into typed bounded `Nat` fields for LexLean + Lean 4.32.2 proofs.
 * - UI `View` elements trace back to admitted requirements (`REQ-D9` / ISO 42010).
 * - `computeAgenda` / `nextOperation` derive open work directly from the registers.
 * - `deriveRepairOp` implements the `REQ-SELF-HEALING` error-to-repair loop.
 */

import type { JsonSchema2020Object, PrismGrants } from './prismStudio';

export const DECLARATION_SCHEMA = 'verdant-studio/declaration/1' as const;

export type PrismThemePreset =
  | 'paper'
  | 'evergreen'
  | 'terracotta'
  | 'noir'
  | 'midnight'
  | 'hologram';

export const PRISM_THEME_PRESETS: readonly {
  readonly id: PrismThemePreset;
  readonly name: string;
  readonly summary: string;
  readonly swatches: readonly [string, string, string];
}[] = [
  { id: 'evergreen', name: 'Evergreen', summary: 'Warm paper & deep forest green', swatches: ['#f7f4ec', '#1f5e44', '#e2dccd'] },
  { id: 'paper', name: 'Paper', summary: 'Crisp editorial ink & warm ivory', swatches: ['#fcfbf8', '#1c1c1c', '#a8763e'] },
  { id: 'terracotta', name: 'Terracotta', summary: 'Sun-baked clay & warm cream', swatches: ['#faf4ee', '#a2462a', '#d79a7e'] },
  { id: 'noir', name: 'Noir', summary: 'Studio dark with brass highlight', swatches: ['#0b0b0a', '#f1ede4', '#e3b77a'] },
  { id: 'midnight', name: 'Midnight', summary: 'Deep navy & amber gold', swatches: ['#0b0f1a', '#e8b04b', '#6ea8ff'] },
  { id: 'hologram', name: 'Hologram', summary: 'Canonical UOR neutral system', swatches: ['#ffffff', '#18181b', '#e4e4e7'] },
];

export type RequirementOrigin = 'identity' | 'refinement' | 'constraint';
export type ClaimLevel = 'build' | 'some-true' | 'open';
export type SettlementRule = 'standard' | 'pattern' | 'extrapolated';

export type StudioCitation =
  | { readonly kind: 'req'; readonly requirementId: string }
  | { readonly kind: 'std'; readonly authorityId: string; readonly row?: string }
  | {
      readonly kind: 'der';
      readonly rule: SettlementRule;
      readonly inputs: readonly string[];
      readonly checkId: string;
    };

export interface OwnerDecision {
  readonly id: string;
  readonly stakeholder: string;
  readonly at: string;
  readonly quote: string;
  readonly requirementIds: readonly string[];
}

export interface StudioAuthority {
  readonly id: string;
  readonly name: string;
  readonly edition: string;
  readonly statement: string;
  readonly sha256?: string;
}

export interface StudioRequirement {
  readonly id: string;
  readonly statement: string;
  readonly stakeholder: string;
  readonly origin: RequirementOrigin;
  readonly evidence: string;
}

export type FieldKind = 'nat' | 'choice' | 'bool';

export interface EncodedField {
  readonly name: string;
  readonly label: string;
  readonly kind: FieldKind;
  readonly min: number;
  readonly max: number;
  readonly choices?: readonly string[];
  readonly unit?: string;
  readonly defaultValue: number;
  readonly description?: string;
}

export interface StudioEncoding {
  readonly name: string;
  readonly fields: readonly EncodedField[];
  readonly schema: JsonSchema2020Object;
  readonly citation: StudioCitation;
}

export type CompareOp = 'eq' | 'neq' | 'lt' | 'lte' | 'gt' | 'gte';

export interface FieldPredicate {
  readonly field: string;
  readonly op: CompareOp;
  readonly value: number;
  /** Optional second field to compare against instead of a constant `value`. */
  readonly rhsField?: string;
}

export interface RuleCondition {
  readonly all: readonly FieldPredicate[];
}

export type ArithmeticExpr =
  | { readonly op: 'const'; readonly value: number }
  | { readonly op: 'field'; readonly field: string }
  | {
      readonly op: 'add' | 'sub' | 'mul' | 'div' | 'mod' | 'pct';
      readonly left: ArithmeticExpr;
      readonly right: ArithmeticExpr;
    };

export type RuleOutcome =
  | {
      readonly kind: 'label';
      readonly code: number;
      readonly label: string;
      readonly tone: 'default' | 'success' | 'warning' | 'destructive' | 'brand';
    }
  | {
      readonly kind: 'formula';
      readonly label: string;
      readonly unit?: string;
      readonly expr: ArithmeticExpr;
    };

export interface StudioFeature {
  readonly id: string;
  readonly level: ClaimLevel;
  readonly statement: string;
  readonly condition: RuleCondition;
  readonly outcome: RuleOutcome;
  readonly priority: number;
  readonly citation: StudioCitation;
}

export interface StudioRefusal {
  readonly id: string;
  readonly statement: string;
  readonly condition: RuleCondition;
  readonly failureText: string;
  readonly fieldHint?: string;
  readonly negativeId: string;
  readonly citation: StudioCitation;
}

export interface AcceptanceVectorDecl {
  readonly id: string;
  readonly inputs: Readonly<Record<string, number>>;
  readonly request: string;
  readonly expectedResponse: string;
  readonly forFeatureId: string;
  readonly isRefusal: boolean;
}

export type PrismViewpoint = 'decision-form' | 'calculator' | 'ledger-tracker' | 'request-reply';

export type ViewElementSlot =
  | 'field-input'
  | 'choice-group'
  | 'stat'
  | 'result-card'
  | 'history-list'
  | 'refusal-alert';

export interface ViewElementDecl {
  readonly id: string;
  readonly slot: ViewElementSlot;
  readonly label: string;
  readonly binding: string;
  readonly requirementId: string;
}

export interface StudioViewDecl {
  readonly viewpoint: PrismViewpoint;
  readonly title: string;
  readonly eyebrow: string;
  readonly subtitle: string;
  readonly submitLabel: string;
  readonly elements: readonly ViewElementDecl[];
  readonly citation: StudioCitation;
}

export interface StudioThemeDecl {
  readonly preset: PrismThemePreset;
  readonly accent?: string;
}

export interface AdmissionDecl {
  readonly construct: string;
  readonly citation: StudioCitation;
}

export interface StudioErrorRecord {
  readonly id: string;
  readonly at: string;
  readonly stage: 'api' | 'check' | 'prove' | 'build' | 'verify' | 'runtime';
  readonly code: string;
  readonly refusal: string;
  readonly brokenRequirementId?: string;
  readonly remediation?: string;
  readonly resolved: boolean;
  readonly suggestedRepair?: StudioApiOp;
}

export interface StudioAuditEntry {
  readonly id: string;
  readonly at: string;
  readonly op: StudioApiOp['op'];
  readonly summary: string;
}

export interface PrismDeclaration {
  readonly schema: typeof DECLARATION_SCHEMA;
  readonly metadata: {
    readonly name: string;
    readonly slug: string;
    readonly version: string;
    readonly description: string;
  };
  readonly decisions: readonly OwnerDecision[];
  readonly authorities: readonly StudioAuthority[];
  readonly requirements: readonly StudioRequirement[];
  readonly encoding: StudioEncoding | null;
  readonly features: readonly StudioFeature[];
  readonly refusals: readonly StudioRefusal[];
  readonly vectors: readonly AcceptanceVectorDecl[];
  readonly view: StudioViewDecl;
  readonly theme: StudioThemeDecl;
  readonly capabilities: PrismGrants;
  readonly admissions: readonly AdmissionDecl[];
  readonly lockedIds: readonly string[];
  readonly errors: readonly StudioErrorRecord[];
  readonly auditLog: readonly StudioAuditEntry[];
}

export type StudioApiOp =
  | {
      readonly op: 'set-metadata';
      readonly name?: string;
      readonly version?: string;
      readonly description?: string;
    }
  | {
      readonly op: 'record-decision';
      readonly id?: string;
      readonly stakeholder: string;
      readonly quote: string;
      readonly requirementIds: readonly string[];
    }
  | {
      readonly op: 'stage-authority';
      readonly authority: StudioAuthority;
    }
  | {
      readonly op: 'admit-requirement';
      readonly requirement: StudioRequirement;
    }
  | {
      readonly op: 'propose-encoding';
      readonly name: string;
      readonly fields?: readonly EncodedField[];
      readonly schema?: JsonSchema2020Object;
      readonly citation: StudioCitation;
    }
  | {
      readonly op: 'admit-feature';
      readonly feature: StudioFeature;
    }
  | {
      readonly op: 'update-feature';
      readonly id: string;
      readonly patch: Partial<Omit<StudioFeature, 'id' | 'citation'>>;
    }
  | {
      readonly op: 'reorder-features';
      readonly orderedIds: readonly string[];
    }
  | {
      readonly op: 'remove-feature';
      readonly id: string;
    }
  | {
      readonly op: 'admit-refusal';
      readonly refusal: StudioRefusal;
    }
  | {
      readonly op: 'update-refusal';
      readonly id: string;
      readonly patch: Partial<Omit<StudioRefusal, 'id' | 'citation'>>;
    }
  | {
      readonly op: 'remove-refusal';
      readonly id: string;
    }
  | {
      readonly op: 'admit-vector';
      readonly vector: Omit<AcceptanceVectorDecl, 'request'> & { readonly request?: string };
    }
  | {
      readonly op: 'remove-vector';
      readonly id: string;
    }
  | {
      readonly op: 'declare-view';
      readonly view: StudioViewDecl;
    }
  | {
      readonly op: 'set-theme';
      readonly theme: StudioThemeDecl;
    }
  | {
      readonly op: 'set-capabilities';
      readonly capabilities: Partial<PrismGrants>;
    }
  | {
      readonly op: 'lock-registers';
    }
  | {
      readonly op: 'record-error';
      readonly error: Omit<StudioErrorRecord, 'id' | 'at' | 'resolved' | 'suggestedRepair'>;
    }
  | {
      readonly op: 'resolve-error';
      readonly id: string;
    }
  | {
      readonly op: 'batch';
      readonly ops: readonly StudioApiOp[];
    };

export type StudioApiResult =
  | {
      readonly ok: true;
      readonly declaration: PrismDeclaration;
      readonly message: string;
    }
  | {
      readonly ok: false;
      readonly code: string;
      readonly message: string;
      readonly remediation: string;
      readonly brokenRequirementId?: string;
    };

export interface DeclarationCheckDiagnostic {
  readonly code: string;
  readonly severity: 'error' | 'warning';
  readonly message: string;
  readonly remediation: string;
  readonly id?: string;
}

export interface DeclarationCheckReport {
  readonly ok: boolean;
  readonly diagnostics: readonly DeclarationCheckDiagnostic[];
  readonly vectorsPassed: number;
  readonly vectorsTotal: number;
  readonly vectorResults: readonly {
    readonly id: string;
    readonly request: string;
    readonly expected: string;
    readonly actual: string;
    readonly passed: boolean;
  }[];
  readonly traceability: readonly {
    readonly requirementId: string;
    readonly featureIds: readonly string[];
    readonly refusalIds: readonly string[];
    readonly viewElementIds: readonly string[];
    readonly covered: boolean;
  }[];
}

export interface StudioAgendaItem {
  readonly id: string;
  readonly priority: number;
  readonly title: string;
  readonly detail: string;
  readonly actionLabel: string;
  readonly targetTab: 'features' | 'data' | 'design' | 'capabilities';
  readonly suggestedOp?: StudioApiOp;
}

export type DeclarativeStarterId =
  | 'budget-guard'
  | 'pricing-rules'
  | 'eligibility-gate'
  | 'split-calc'
  | 'blank';

export const DECLARATIVE_STARTERS: readonly {
  readonly id: DeclarativeStarterId;
  readonly label: string;
  readonly description: string;
  readonly preset: PrismThemePreset;
  readonly viewpoint: PrismViewpoint;
}[] = [
  {
    id: 'budget-guard',
    label: 'Monthly Budget Guard',
    description: 'Proved spending limit evaluator with category choices, remaining balance math, and zero/cap refusals.',
    preset: 'evergreen',
    viewpoint: 'ledger-tracker',
  },
  {
    id: 'pricing-rules',
    label: 'Tiered Discount Engine',
    description: 'Volume & member-tier pricing rules with proved net totals and quantity/price refusals.',
    preset: 'paper',
    viewpoint: 'calculator',
  },
  {
    id: 'eligibility-gate',
    label: 'Access & Plan Gate',
    description: 'Policy decision table classifying accounts into Approved, Manual Review, or Denied with proved invariants.',
    preset: 'midnight',
    viewpoint: 'decision-form',
  },
  {
    id: 'split-calc',
    label: 'Bill & Tip Splitter',
    description: 'Per-person total calculator with tip percentage and proved zero-diner / zero-bill refusals.',
    preset: 'terracotta',
    viewpoint: 'calculator',
  },
];

// ============================================================================
// Helpers & Pure Rule Evaluation (1-to-1 with Compiled App.lex.tex)
// ============================================================================

export function slugifyPrismName(name: string): string {
  const cleaned = name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return cleaned || 'prism-app';
}

export function sanitizeIdentifier(name: string): string {
  const cleaned = name
    .trim()
    .replace(/[^a-zA-Z0-9_]/g, '_')
    .replace(/^[^a-zA-Z]+/, '');
  return cleaned || 'field';
}

const VALID_ID = /^[A-Z][A-Z0-9-]{1,31}$/;
const HEX_COLOR = /^#(?:[0-9a-f]{3}|[0-9a-f]{6})$/i;

export function fieldsToJsonSchema(fields: readonly EncodedField[]): JsonSchema2020Object {
  const properties: JsonSchema2020Object['properties'] = {};
  const required: string[] = [];
  for (const field of fields) {
    if (field.kind === 'choice' && field.choices && field.choices.length > 0) {
      properties[field.name] = {
        type: 'string',
        description: field.description ?? `${field.label} (${field.choices.join(' | ')})`,
        enum: [...field.choices],
      };
    } else if (field.kind === 'bool') {
      properties[field.name] = {
        type: 'boolean',
        description: field.description ?? field.label,
      };
    } else {
      properties[field.name] = {
        type: 'number',
        description:
          field.description ??
          `${field.label} (${field.min}..${field.max}${field.unit ? ` ${field.unit}` : ''})`,
      };
    }
    required.push(field.name);
  }
  return { type: 'object', properties, required };
}

export function jsonSchemaToFields(schema: JsonSchema2020Object): EncodedField[] {
  const fields: EncodedField[] = [];
  for (const [rawKey, prop] of Object.entries(schema.properties ?? {})) {
    const name = sanitizeIdentifier(rawKey);
    const label = name
      .replace(/([a-z])([A-Z])/g, '$1 $2')
      .replace(/[_-]+/g, ' ')
      .replace(/\b\w/g, (c) => c.toUpperCase());
    if (prop.type === 'boolean') {
      fields.push({
        name,
        label,
        kind: 'bool',
        min: 0,
        max: 1,
        defaultValue: 0,
        description: prop.description,
      });
    } else if (prop.type === 'string' && Array.isArray(prop.enum) && prop.enum.length > 0) {
      fields.push({
        name,
        label,
        kind: 'choice',
        min: 0,
        max: prop.enum.length - 1,
        choices: [...prop.enum],
        defaultValue: 0,
        description: prop.description,
      });
    } else {
      fields.push({
        name,
        label,
        kind: 'nat',
        min: 0,
        max: 1_000_000,
        defaultValue: 0,
        description: prop.description,
      });
    }
  }
  return fields;
}

/**
 * Canonical space-separated non-negative integer request line for a declared encoding.
 * Choices map to their 0-based index, booleans map to 0/1, nats map to non-negative integers.
 */
export function formatRequestLine(
  encoding: StudioEncoding | null,
  inputs: Readonly<Record<string, number>>,
): string {
  if (!encoding || encoding.fields.length === 0) return '';
  return encoding.fields
    .map((f) => {
      const raw = inputs[f.name] ?? f.defaultValue ?? 0;
      return String(Math.max(0, Math.floor(Number.isFinite(raw) ? raw : 0)));
    })
    .join(' ');
}

export function parseRequestLine(
  encoding: StudioEncoding | null,
  request: string,
): { readonly ok: true; readonly values: Record<string, number> } | { readonly ok: false; readonly reason: string } {
  if (!encoding || encoding.fields.length === 0) {
    return { ok: true, values: {} };
  }
  const trimmed = request.trim();
  if (!trimmed) {
    return { ok: false, reason: 'Invalid request format.' };
  }
  const parts = trimmed.split(/\s+/);
  if (parts.length !== encoding.fields.length) {
    return { ok: false, reason: 'Invalid request format.' };
  }
  const values: Record<string, number> = {};
  for (let i = 0; i < encoding.fields.length; i++) {
    const field = encoding.fields[i];
    const token = parts[i];
    if (/^\d+$/.test(token)) {
      values[field.name] = Number.parseInt(token, 10);
      continue;
    }
    if (field.kind === 'choice' && field.choices) {
      const idx = field.choices.findIndex((c) => c.toLowerCase() === token.toLowerCase());
      if (idx >= 0) {
        values[field.name] = idx;
        continue;
      }
    }
    if (field.kind === 'bool') {
      if (token.toLowerCase() === 'true' || token.toLowerCase() === 'yes') {
        values[field.name] = 1;
        continue;
      }
      if (token.toLowerCase() === 'false' || token.toLowerCase() === 'no') {
        values[field.name] = 0;
        continue;
      }
    }
    return { ok: false, reason: 'Invalid request format.' };
  }
  return { ok: true, values };
}

export function evalPredicate(pred: FieldPredicate, values: Readonly<Record<string, number>>): boolean {
  const lhs = Math.max(0, Math.floor(values[pred.field] ?? 0));
  const rhs = pred.rhsField
    ? Math.max(0, Math.floor(values[pred.rhsField] ?? 0))
    : Math.max(0, Math.floor(pred.value));
  switch (pred.op) {
    case 'eq': return lhs === rhs;
    case 'neq': return lhs !== rhs;
    case 'lt': return lhs < rhs;
    case 'lte': return lhs <= rhs;
    case 'gt': return lhs > rhs;
    case 'gte': return lhs >= rhs;
  }
}

export function evalCondition(cond: RuleCondition, values: Readonly<Record<string, number>>): boolean {
  return cond.all.every((p) => evalPredicate(p, values));
}

/** Exact Nat arithmetic semantics matching LexLean (`sub` truncates at 0, `div`/`mod` by 0 return 0). */
export function evalArithmetic(expr: ArithmeticExpr, values: Readonly<Record<string, number>>): number {
  switch (expr.op) {
    case 'const':
      return Math.max(0, Math.floor(expr.value));
    case 'field':
      return Math.max(0, Math.floor(values[expr.field] ?? 0));
    case 'add':
      return evalArithmetic(expr.left, values) + evalArithmetic(expr.right, values);
    case 'sub':
      return Math.max(0, evalArithmetic(expr.left, values) - evalArithmetic(expr.right, values));
    case 'mul':
      return evalArithmetic(expr.left, values) * evalArithmetic(expr.right, values);
    case 'div': {
      const denom = evalArithmetic(expr.right, values);
      return denom === 0 ? 0 : Math.floor(evalArithmetic(expr.left, values) / denom);
    }
    case 'mod': {
      const denom = evalArithmetic(expr.right, values);
      return denom === 0 ? 0 : evalArithmetic(expr.left, values) % denom;
    }
    case 'pct': {
      return Math.floor((evalArithmetic(expr.left, values) * evalArithmetic(expr.right, values)) / 100);
    }
  }
}

export interface EvaluatedModelAnswer {
  readonly response: string;
  readonly isRefusal: boolean;
  readonly matchedId: string | null;
  readonly numericValue?: number;
  readonly tone?: 'default' | 'success' | 'warning' | 'destructive' | 'brand';
}

/**
 * Deterministic instant evaluator whose output matches the compiled LexLean `dispatch`
 * byte-for-byte on every valid and invalid request line.
 */
export function evaluateDeclaredModel(
  decl: PrismDeclaration,
  requestLine: string,
): EvaluatedModelAnswer {
  if (!decl.encoding || decl.encoding.fields.length === 0) {
    if (requestLine.trim().length === 0) {
      const firstRefusal = decl.refusals[0];
      if (firstRefusal) {
        return {
          response: firstRefusal.failureText,
          isRefusal: true,
          matchedId: firstRefusal.id,
          tone: 'destructive',
        };
      }
    }
    return {
      response: requestLine,
      isRefusal: false,
      matchedId: decl.features[0]?.id ?? null,
      tone: 'default',
    };
  }

  const parsed = parseRequestLine(decl.encoding, requestLine);
  if (!parsed.ok) {
    return {
      response: parsed.reason,
      isRefusal: true,
      matchedId: null,
      tone: 'destructive',
    };
  }
  const values = parsed.values;

  // 1. Check declared refusals first (in declaration order)
  for (const refusal of decl.refusals) {
    if (evalCondition(refusal.condition, values)) {
      return {
        response: refusal.failureText,
        isRefusal: true,
        matchedId: refusal.id,
        tone: 'destructive',
      };
    }
  }

  // 2. Check choice bounds if not already caught by an explicit refusal
  for (const field of decl.encoding.fields) {
    const val = values[field.name] ?? 0;
    if (field.kind === 'choice' && field.choices && val >= field.choices.length) {
      return {
        response: `Unknown ${field.label.toLowerCase()}.`,
        isRefusal: true,
        matchedId: null,
        tone: 'destructive',
      };
    }
  }

  // 3. Evaluate features sorted by priority ascending
  const ordered = [...decl.features].sort((a, b) => a.priority - b.priority || a.id.localeCompare(b.id));
  for (const feat of ordered) {
    if (evalCondition(feat.condition, values)) {
      if (feat.outcome.kind === 'label') {
        return {
          response: feat.outcome.label,
          isRefusal: false,
          matchedId: feat.id,
          tone: feat.outcome.tone,
        };
      }
      const computed = evalArithmetic(feat.outcome.expr, values);
      const unitSuffix = feat.outcome.unit ? ` ${feat.outcome.unit}` : '';
      return {
        response: `${feat.outcome.label}: ${computed}${unitSuffix}`,
        isRefusal: false,
        matchedId: feat.id,
        numericValue: computed,
        tone: 'success',
      };
    }
  }

  return {
    response: 'No matching rule.',
    isRefusal: true,
    matchedId: null,
    tone: 'warning',
  };
}

// ============================================================================
// Studio.Admission Gate (Exact Parity with afflom/verdant-studio)
// ============================================================================

function citationResolves(
  decl: PrismDeclaration,
  admittedConstructs: ReadonlySet<string>,
  citation: StudioCitation,
): { readonly ok: true } | { readonly ok: false; readonly detail: string } {
  if (citation.kind === 'req') {
    const exists = decl.requirements.some((r) => r.id === citation.requirementId);
    return exists
      ? { ok: true }
      : { ok: false, detail: `Requirement "${citation.requirementId}" is not registered.` };
  }
  if (citation.kind === 'std') {
    const exists = decl.authorities.some((a) => a.id === citation.authorityId);
    return exists
      ? { ok: true }
      : { ok: false, detail: `Authority "${citation.authorityId}" is not staged.` };
  }
  const missingInput = citation.inputs.find((inp) => !admittedConstructs.has(inp));
  if (missingInput) {
    return {
      ok: false,
      detail: `Derived citation input "${missingInput}" is not admitted yet (well-foundedness violation).`,
    };
  }
  const checkExists =
    decl.features.some((f) => f.id === citation.checkId) ||
    decl.refusals.some((r) => r.id === citation.checkId) ||
    decl.requirements.some((r) => r.id === citation.checkId);
  if (!checkExists) {
    return {
      ok: false,
      detail: `Derived citation check "${citation.checkId}" is not registered.`,
    };
  }
  return { ok: true };
}

function allRegisteredIds(decl: PrismDeclaration): Set<string> {
  const ids = new Set<string>();
  for (const r of decl.requirements) ids.add(r.id);
  for (const f of decl.features) ids.add(f.id);
  for (const r of decl.refusals) ids.add(r.id);
  for (const a of decl.authorities) ids.add(a.id);
  return ids;
}

export function checkDeclaration(decl: PrismDeclaration): DeclarationCheckReport {
  const diagnostics: DeclarationCheckDiagnostic[] = [];

  // 1. Monotonicity check (`Studio.Admission.monotone`): no locked ID dropped
  const currentIds = allRegisteredIds(decl);
  for (const locked of decl.lockedIds) {
    if (!currentIds.has(locked)) {
      diagnostics.push({
        code: 'admission.monotone',
        severity: 'error',
        id: locked,
        message: `Locked identifier "${locked}" was dropped from the declaration registers.`,
        remediation: `Restore "${locked}" through the Studio API; locked registers are monotone (P10).`,
      });
    }
  }

  // 2. Well-founded admission chain (`Studio.Admission.admitsAll`)
  const admitted = new Set<string>();
  for (const entry of decl.admissions) {
    const res = citationResolves(decl, admitted, entry.citation);
    if (!res.ok) {
      diagnostics.push({
        code: 'admission.unresolved',
        severity: 'error',
        id: entry.construct,
        message: `Construct "${entry.construct}" failed admission: ${res.detail}`,
        remediation: 'Cite an admitted Requirement, staged Authority, or previously admitted construct (D12).',
      });
    } else {
      admitted.add(entry.construct);
    }
  }

  // 3. Encoding & field predicate validation
  const fieldNames = new Set(decl.encoding?.fields.map((f) => f.name) ?? []);
  if (!decl.encoding || decl.encoding.fields.length === 0) {
    diagnostics.push({
      code: 'encoding.missing',
      severity: 'error',
      message: 'No input data encoding is declared yet.',
      remediation: 'Declare at least one typed input field in the Data tab (propose-encoding).',
    });
  }

  const checkConditionFields = (cond: RuleCondition, ownerId: string) => {
    for (const pred of cond.all) {
      if (!fieldNames.has(pred.field)) {
        diagnostics.push({
          code: 'rule.unknown-field',
          severity: 'error',
          id: ownerId,
          message: `Rule "${ownerId}" references undeclared field "${pred.field}".`,
          remediation: `Add "${pred.field}" via propose-encoding or update rule "${ownerId}".`,
        });
      }
      if (pred.rhsField && !fieldNames.has(pred.rhsField)) {
        diagnostics.push({
          code: 'rule.unknown-field',
          severity: 'error',
          id: ownerId,
          message: `Rule "${ownerId}" compares against undeclared field "${pred.rhsField}".`,
          remediation: `Add "${pred.rhsField}" via propose-encoding or update rule "${ownerId}".`,
        });
      }
    }
  };

  for (const feat of decl.features) {
    checkConditionFields(feat.condition, feat.id);
  }
  for (const ref of decl.refusals) {
    checkConditionFields(ref.condition, ref.id);
    if (!ref.failureText.trim()) {
      diagnostics.push({
        code: 'contract.negative',
        severity: 'error',
        id: ref.id,
        message: `Refusal "${ref.id}" has an empty refusal message.`,
        remediation: 'Provide the exact refusal text the model answers (R5).',
      });
    }
  }

  // 4. Acceptance vector replay & R5 refusal coverage
  if (decl.vectors.length === 0) {
    diagnostics.push({
      code: 'holo.no-vectors',
      severity: 'error',
      message: 'No acceptance examples are declared yet.',
      remediation: 'Add at least one input → expected-output example before building.',
    });
  }

  const vectorResults = decl.vectors.map((v) => {
    const evaluated = evaluateDeclaredModel(decl, v.request);
    return {
      id: v.id,
      request: v.request,
      expected: v.expectedResponse,
      actual: evaluated.response,
      passed: evaluated.response === v.expectedResponse,
    };
  });

  for (const vr of vectorResults) {
    if (!vr.passed) {
      diagnostics.push({
        code: 'holo.vectors',
        severity: 'error',
        id: vr.id,
        message: `Example "${vr.id}" expected "${vr.expected}", but model answered "${vr.actual}".`,
        remediation: 'Update the rule or the example expectation so the model agrees with every example.',
      });
    }
  }

  // R5: Every registered refusal must have at least one passing replayed vector answering exact failureText
  for (const ref of decl.refusals) {
    const replayed = vectorResults.some((vr) => vr.passed && vr.expected === ref.failureText);
    if (!replayed) {
      diagnostics.push({
        code: 'contract.refusal-unreplayed',
        severity: 'error',
        id: ref.id,
        message: `Refusal "${ref.id}" ("${ref.failureText}") has no passing example vector.`,
        remediation: `Add a concrete example input that triggers "${ref.failureText}" (R5).`,
      });
    }
  }

  // 5. Bidirectional View Traceability (`REQ-D9` / ISO 29148)
  const reqIds = new Set(decl.requirements.map((r) => r.id));
  for (const el of decl.view.elements) {
    if (!reqIds.has(el.requirementId)) {
      diagnostics.push({
        code: 'view.uncited-element',
        severity: 'error',
        id: el.id,
        message: `View element "${el.label}" cites unknown requirement "${el.requirementId}".`,
        remediation: 'Bind every UI element to an admitted Requirement (REQ-D9).',
      });
    }
    if ((el.slot === 'field-input' || el.slot === 'choice-group') && !fieldNames.has(el.binding)) {
      diagnostics.push({
        code: 'view.unbound-field',
        severity: 'error',
        id: el.id,
        message: `View control "${el.label}" binds to undeclared field "${el.binding}".`,
        remediation: `Declare "${el.binding}" in Data or rebind the control in Design.`,
      });
    }
  }

  const traceability = decl.requirements.map((req) => {
    const featureIds = decl.features
      .filter((f) => f.citation.kind === 'req' && f.citation.requirementId === req.id)
      .map((f) => f.id);
    const refusalIds = decl.refusals
      .filter((r) => r.citation.kind === 'req' && r.citation.requirementId === req.id)
      .map((r) => r.id);
    const viewElementIds = decl.view.elements
      .filter((e) => e.requirementId === req.id)
      .map((e) => e.id);
    const covered = featureIds.length > 0 || refusalIds.length > 0 || viewElementIds.length > 0;
    return { requirementId: req.id, featureIds, refusalIds, viewElementIds, covered };
  });

  const vectorsPassed = vectorResults.filter((v) => v.passed).length;
  return {
    ok: diagnostics.every((d) => d.severity !== 'error'),
    diagnostics,
    vectorsPassed,
    vectorsTotal: vectorResults.length,
    vectorResults,
    traceability,
  };
}

// ============================================================================
// Derived Agenda & Self-Healing Repair Engine (`next` / `agenda` / REQ-SELF-HEALING)
// ============================================================================

export function deriveRepairOp(
  decl: PrismDeclaration,
  code: string,
  refusalMessage: string,
  targetId?: string,
): StudioApiOp | undefined {
  if (code === 'contract.refusal-unreplayed' && targetId) {
    const ref = decl.refusals.find((r) => r.id === targetId);
    if (ref && decl.encoding) {
      const inputs: Record<string, number> = {};
      for (const f of decl.encoding.fields) inputs[f.name] = f.defaultValue;
      for (const pred of ref.condition.all) {
        if (pred.op === 'eq' || pred.op === 'lte' || pred.op === 'gte') inputs[pred.field] = pred.value;
        else if (pred.op === 'lt') inputs[pred.field] = Math.max(0, pred.value - 1);
        else if (pred.op === 'gt') inputs[pred.field] = pred.value + 1;
        else if (pred.op === 'neq') inputs[pred.field] = pred.value + 1;
      }
      return {
        op: 'admit-vector',
        vector: {
          id: `vec_${ref.id.toLowerCase().replace(/[^a-z0-9]+/g, '_')}`,
          inputs,
          expectedResponse: ref.failureText,
          forFeatureId: ref.id,
          isRefusal: true,
        },
      };
    }
  }
  if (code === 'holo.vectors' && targetId) {
    const vec = decl.vectors.find((v) => v.id === targetId);
    if (vec) {
      const actual = evaluateDeclaredModel(decl, vec.request);
      return {
        op: 'admit-vector',
        vector: {
          ...vec,
          expectedResponse: actual.response,
          isRefusal: actual.isRefusal,
        },
      };
    }
  }
  if (code === 'view.uncited-element' && targetId && decl.requirements.length > 0) {
    const fallbackReq = decl.requirements[0].id;
    return {
      op: 'declare-view',
      view: {
        ...decl.view,
        elements: decl.view.elements.map((el) =>
          el.id === targetId ? { ...el, requirementId: fallbackReq } : el,
        ),
      },
    };
  }
  if (code === 'view.unbound-field' && targetId) {
    return {
      op: 'declare-view',
      view: {
        ...decl.view,
        elements: decl.view.elements.filter((el) => el.id !== targetId),
      },
    };
  }
  void refusalMessage;
  return undefined;
}

export function computeAgenda(decl: PrismDeclaration): readonly StudioAgendaItem[] {
  const items: StudioAgendaItem[] = [];
  const check = checkDeclaration(decl);

  for (const err of decl.errors.filter((e) => !e.resolved)) {
    items.push({
      id: `repair_${err.id}`,
      priority: 0,
      title: `Repair ${err.code}`,
      detail: err.refusal,
      actionLabel: err.suggestedRepair ? 'Apply API Repair' : 'Inspect',
      targetTab: 'capabilities',
      suggestedOp: err.suggestedRepair,
    });
  }

  for (const diag of check.diagnostics) {
    const repair = deriveRepairOp(decl, diag.code, diag.message, diag.id);
    items.push({
      id: `diag_${diag.code}_${diag.id ?? 'global'}`,
      priority: 1,
      title: diag.message,
      detail: diag.remediation,
      actionLabel: repair ? 'Apply Fix' : 'Open Tab',
      targetTab:
        diag.code.startsWith('encoding') || diag.code.startsWith('rule.')
          ? 'data'
          : diag.code.startsWith('view.')
            ? 'design'
            : 'features',
      suggestedOp: repair,
    });
  }

  // Check for declared input fields not yet bound in the View
  if (decl.encoding) {
    const boundFields = new Set(decl.view.elements.map((e) => e.binding));
    const reqId = decl.requirements[0]?.id;
    for (const f of decl.encoding.fields) {
      if (!boundFields.has(f.name) && reqId) {
        items.push({
          id: `bind_field_${f.name}`,
          priority: 2,
          title: `Bind "${f.label}" field to interface`,
          detail: `Field "${f.name}" is declared in Data but not yet shown in the Design view.`,
          actionLabel: 'Bind Control',
          targetTab: 'design',
          suggestedOp: {
            op: 'declare-view',
            view: {
              ...decl.view,
              elements: [
                ...decl.view.elements,
                {
                  id: `el_${f.name}`,
                  slot: f.kind === 'choice' ? 'choice-group' : 'field-input',
                  label: f.label,
                  binding: f.name,
                  requirementId: reqId,
                },
              ],
            },
          },
        });
      }
    }
  }

  // Check for requirements with no rule/refusal yet
  for (const trace of check.traceability) {
    if (!trace.covered) {
      items.push({
        id: `cover_req_${trace.requirementId}`,
        priority: 3,
        title: `Add rule for ${trace.requirementId}`,
        detail: `Requirement ${trace.requirementId} has no feature rule or refusal attached yet.`,
        actionLabel: 'Add Rule',
        targetTab: 'features',
      });
    }
  }

  return items.sort((a, b) => a.priority - b.priority);
}

// ============================================================================
// Closed Declarative API Dispatcher (`dispatchStudioOp`)
// ============================================================================

function appendAudit(
  decl: PrismDeclaration,
  op: StudioApiOp['op'],
  summary: string,
): readonly StudioAuditEntry[] {
  const entry: StudioAuditEntry = {
    id: `aud_${Date.now().toString(36)}_${decl.auditLog.length + 1}`,
    at: new Date().toISOString(),
    op,
    summary,
  };
  return [...decl.auditLog.slice(-49), entry];
}

function upsertAdmission(
  admissions: readonly AdmissionDecl[],
  construct: string,
  citation: StudioCitation,
): readonly AdmissionDecl[] {
  const idx = admissions.findIndex((a) => a.construct === construct);
  if (idx >= 0) {
    const next = [...admissions];
    next[idx] = { construct, citation };
    return next;
  }
  return [...admissions, { construct, citation }];
}

export function createEmptyDeclaration(name = 'Untitled Prism'): PrismDeclaration {
  const slug = slugifyPrismName(name);
  return {
    schema: DECLARATION_SCHEMA,
    metadata: {
      name: name.trim() || 'Untitled Prism',
      slug,
      version: '0.1.0',
      description: 'A declarative Prism application compiled into a verified .holo archive.',
    },
    decisions: [],
    authorities: [
      {
        id: 'PRISM-TEMPLATE-CONTRACT-2',
        name: 'UOR Prism Template Contract 2',
        edition: '2.0',
        statement: 'Authoritative single-source model, typed refusals (R5), and Lean kernel proof boundary.',
      },
      {
        id: 'PRISM-DESIGN-1',
        name: 'Prism Design System (prism.design/1)',
        edition: '1.0',
        statement: 'Compiled shadcn/ui data-slot components and WCAG-checked token themes.',
      },
    ],
    requirements: [
      {
        id: 'REQ-01',
        statement: 'Provide a clear, deterministic answer for every valid user input.',
        stakeholder: 'creator',
        origin: 'identity',
        evidence: 'Initial Prism declaration',
      },
    ],
    encoding: null,
    features: [],
    refusals: [],
    vectors: [],
    view: {
      viewpoint: 'decision-form',
      title: name.trim() || 'Untitled Prism',
      eyebrow: 'Verified Prism',
      subtitle: 'Every answer is computed by the declared, proved model.',
      submitLabel: 'Evaluate',
      elements: [
        {
          id: 'el_result',
          slot: 'result-card',
          label: 'Model Answer',
          binding: 'outcome',
          requirementId: 'REQ-01',
        },
      ],
      citation: { kind: 'req', requirementId: 'REQ-01' },
    },
    theme: { preset: 'evergreen' },
    capabilities: {
      observer: true,
      storage: false,
      browser: false,
      network: false,
    },
    admissions: [
      { construct: 'View.Default', citation: { kind: 'req', requirementId: 'REQ-01' } },
    ],
    lockedIds: [],
    errors: [],
    auditLog: [],
  };
}

export function dispatchStudioOp(decl: PrismDeclaration, op: StudioApiOp): StudioApiResult {
  if (op.op === 'batch') {
    let current = decl;
    for (const sub of op.ops) {
      const res = dispatchStudioOp(current, sub);
      if (!res.ok) return res;
      current = res.declaration;
    }
    return {
      ok: true,
      declaration: current,
      message: `Applied ${op.ops.length} declarative API operations.`,
    };
  }

  switch (op.op) {
    case 'set-metadata': {
      const nextName = op.name !== undefined ? op.name.trim() : decl.metadata.name;
      if (!nextName) {
        return {
          ok: false,
          code: 'metadata.empty-name',
          message: 'Prism name cannot be empty.',
          remediation: 'Provide a non-empty application name.',
        };
      }
      const next: PrismDeclaration = {
        ...decl,
        metadata: {
          name: nextName,
          slug: slugifyPrismName(nextName),
          version: op.version?.trim() || decl.metadata.version,
          description: op.description !== undefined ? op.description.trim() : decl.metadata.description,
        },
        view: {
          ...decl.view,
          title: decl.view.title === decl.metadata.name ? nextName : decl.view.title,
        },
        auditLog: appendAudit(decl, op.op, `Updated metadata (${nextName})`),
      };
      return { ok: true, declaration: next, message: `Updated metadata for "${nextName}".` };
    }

    case 'record-decision': {
      const quote = op.quote.trim();
      if (!quote) {
        return {
          ok: false,
          code: 'decision.empty-quote',
          message: 'Owner decision quote cannot be empty.',
          remediation: 'Provide the verbatim decision text.',
        };
      }
      const decision: OwnerDecision = {
        id: op.id?.trim() || `DEC-${String(decl.decisions.length + 1).padStart(2, '0')}`,
        stakeholder: op.stakeholder.trim() || 'creator',
        at: new Date().toISOString(),
        quote,
        requirementIds: [...op.requirementIds],
      };
      const next: PrismDeclaration = {
        ...decl,
        decisions: [...decl.decisions, decision],
        auditLog: appendAudit(decl, op.op, `Recorded decision ${decision.id}`),
      };
      return { ok: true, declaration: next, message: `Recorded decision ${decision.id}.` };
    }

    case 'stage-authority': {
      const id = op.authority.id.trim().toUpperCase();
      if (!VALID_ID.test(id)) {
        return {
          ok: false,
          code: 'authority.invalid-id',
          message: `Authority ID "${op.authority.id}" must be uppercase alphanumeric with hyphens (e.g. ISO-25010).`,
          remediation: 'Use an uppercase ID like "ISO-25010".',
        };
      }
      const exists = decl.authorities.some((a) => a.id === id);
      const nextAuths = exists
        ? decl.authorities.map((a) => (a.id === id ? { ...op.authority, id } : a))
        : [...decl.authorities, { ...op.authority, id }];
      return {
        ok: true,
        declaration: {
          ...decl,
          authorities: nextAuths,
          auditLog: appendAudit(decl, op.op, `Staged authority ${id}`),
        },
        message: `Staged authority ${id}.`,
      };
    }

    case 'admit-requirement': {
      const id = op.requirement.id.trim().toUpperCase();
      if (!VALID_ID.test(id)) {
        return {
          ok: false,
          code: 'requirement.invalid-id',
          message: `Requirement ID "${op.requirement.id}" is invalid (expected e.g. REQ-02).`,
          remediation: 'Use an uppercase ID such as REQ-02.',
        };
      }
      if (!op.requirement.statement.trim() || !op.requirement.evidence.trim()) {
        return {
          ok: false,
          code: 'requirement.missing-evidence',
          message: `Requirement ${id} requires both a statement and origin evidence (REQ-D10).`,
          remediation: 'Provide the requirement statement and cite its origin evidence.',
        };
      }
      if (decl.requirements.some((r) => r.id === id)) {
        return {
          ok: false,
          code: 'admission.duplicate',
          message: `Requirement "${id}" is already registered.`,
          remediation: 'Choose a distinct requirement ID.',
        };
      }
      const next: PrismDeclaration = {
        ...decl,
        requirements: [
          ...decl.requirements,
          {
            ...op.requirement,
            id,
            statement: op.requirement.statement.trim(),
            evidence: op.requirement.evidence.trim(),
            stakeholder: op.requirement.stakeholder.trim() || 'creator',
          },
        ],
        auditLog: appendAudit(decl, op.op, `Admitted requirement ${id}`),
      };
      return { ok: true, declaration: next, message: `Admitted requirement ${id}.` };
    }

    case 'propose-encoding': {
      const admittedSet = new Set(decl.admissions.map((a) => a.construct));
      const checkCit = citationResolves(decl, admittedSet, op.citation);
      if (!checkCit.ok) {
        return {
          ok: false,
          code: 'admission.unresolved',
          message: `Encoding "${op.name}" refused: ${checkCit.detail}`,
          remediation: 'Cite an admitted Requirement or staged Authority (D12).',
        };
      }
      const rawFields = op.fields ? [...op.fields] : op.schema ? jsonSchemaToFields(op.schema) : [];
      if (rawFields.length === 0) {
        return {
          ok: false,
          code: 'encoding.empty',
          message: 'An encoding must declare at least one input field.',
          remediation: 'Add at least one typed field to the encoding.',
        };
      }
      const seen = new Set<string>();
      const normalized: EncodedField[] = [];
      for (const f of rawFields) {
        const cleanName = sanitizeIdentifier(f.name);
        if (seen.has(cleanName)) {
          return {
            ok: false,
            code: 'encoding.duplicate-field',
            message: `Field "${cleanName}" is declared twice in encoding "${op.name}".`,
            remediation: 'Give every input field a unique identifier.',
          };
        }
        seen.add(cleanName);
        const min = Math.max(0, Math.floor(f.min));
        const max = Math.max(min, Math.floor(f.max));
        const choices = f.kind === 'choice'
          ? (f.choices && f.choices.length > 0 ? f.choices.map((c) => c.trim()).filter(Boolean) : ['Option A', 'Option B'])
          : undefined;
        normalized.push({
          ...f,
          name: cleanName,
          label: f.label.trim() || cleanName,
          min: f.kind === 'choice' ? 0 : f.kind === 'bool' ? 0 : min,
          max: f.kind === 'choice' ? Math.max(0, (choices?.length ?? 1) - 1) : f.kind === 'bool' ? 1 : max,
          choices,
          defaultValue: Math.max(min, Math.min(f.kind === 'choice' ? Math.max(0, (choices?.length ?? 1) - 1) : max, Math.floor(f.defaultValue ?? min))),
        });
      }
      const encoding: StudioEncoding = {
        name: sanitizeIdentifier(op.name),
        fields: normalized,
        schema: fieldsToJsonSchema(normalized),
        citation: op.citation,
      };
      // Re-serialize existing vectors so request lines stay aligned with field order
      const vectors = decl.vectors.map((v) => ({
        ...v,
        request: formatRequestLine(encoding, v.inputs),
      }));
      const next: PrismDeclaration = {
        ...decl,
        encoding,
        vectors,
        admissions: upsertAdmission(decl.admissions, `Encoding.${encoding.name}`, op.citation),
        auditLog: appendAudit(decl, op.op, `Proposed encoding ${encoding.name} (${normalized.length} fields)`),
      };
      return { ok: true, declaration: next, message: `Admitted encoding "${encoding.name}" (${normalized.length} fields).` };
    }

    case 'admit-feature': {
      const id = op.feature.id.trim().toUpperCase();
      if (!VALID_ID.test(id)) {
        return {
          ok: false,
          code: 'feature.invalid-id',
          message: `Feature ID "${op.feature.id}" is invalid (expected e.g. APP-01).`,
          remediation: 'Use an uppercase ID such as APP-01.',
        };
      }
      if (decl.features.some((f) => f.id === id) || decl.refusals.some((r) => r.id === id)) {
        return {
          ok: false,
          code: 'admission.duplicate',
          message: `Capability ID "${id}" is already registered.`,
          remediation: 'Choose a distinct ID or use update-feature.',
        };
      }
      const admittedSet = new Set(decl.admissions.map((a) => a.construct));
      const citCheck = citationResolves(decl, admittedSet, op.feature.citation);
      if (!citCheck.ok) {
        return {
          ok: false,
          code: 'admission.unresolved',
          message: `Feature "${id}" refused: ${citCheck.detail}`,
          remediation: 'Cite an admitted Requirement, staged Authority, or admitted construct (D12).',
        };
      }
      const feature: StudioFeature = {
        ...op.feature,
        id,
        statement: op.feature.statement.trim() || `Feature ${id}`,
      };
      const next: PrismDeclaration = {
        ...decl,
        features: [...decl.features, feature],
        admissions: upsertAdmission(decl.admissions, `Feature.${id}`, feature.citation),
        auditLog: appendAudit(decl, op.op, `Admitted feature ${id}`),
      };
      return { ok: true, declaration: next, message: `Admitted feature ${id}.` };
    }

    case 'update-feature': {
      const existing = decl.features.find((f) => f.id === op.id);
      if (!existing) {
        return {
          ok: false,
          code: 'feature.not-found',
          message: `Feature "${op.id}" does not exist.`,
          remediation: 'Admit the feature first.',
        };
      }
      const next: PrismDeclaration = {
        ...decl,
        features: decl.features.map((f) => (f.id === op.id ? { ...f, ...op.patch } : f)),
        auditLog: appendAudit(decl, op.op, `Updated feature ${op.id}`),
      };
      return { ok: true, declaration: next, message: `Updated feature ${op.id}.` };
    }

    case 'reorder-features': {
      const rank = new Map(op.orderedIds.map((id, idx) => [id, idx + 1]));
      const nextFeatures = [...decl.features]
        .map((f) => ({ ...f, priority: rank.get(f.id) ?? f.priority + 100 }))
        .sort((a, b) => a.priority - b.priority || a.id.localeCompare(b.id));
      const next: PrismDeclaration = {
        ...decl,
        features: nextFeatures,
        auditLog: appendAudit(decl, op.op, `Reordered ${nextFeatures.length} feature priorities`),
      };
      return { ok: true, declaration: next, message: 'Reordered feature evaluation priority.' };
    }

    case 'remove-feature': {
      if (decl.lockedIds.includes(op.id)) {
        return {
          ok: false,
          code: 'admission.monotone',
          message: `Cannot remove locked feature "${op.id}" (registers are monotone per P10).`,
          remediation: 'Refine the feature in place via update-feature instead of removing it.',
        };
      }
      const next: PrismDeclaration = {
        ...decl,
        features: decl.features.filter((f) => f.id !== op.id),
        admissions: decl.admissions.filter((a) => a.construct !== `Feature.${op.id}`),
        auditLog: appendAudit(decl, op.op, `Removed draft feature ${op.id}`),
      };
      return { ok: true, declaration: next, message: `Removed feature ${op.id}.` };
    }

    case 'admit-refusal': {
      const id = op.refusal.id.trim().toUpperCase();
      if (!VALID_ID.test(id)) {
        return {
          ok: false,
          code: 'refusal.invalid-id',
          message: `Refusal ID "${op.refusal.id}" is invalid (expected e.g. APP-02).`,
          remediation: 'Use an uppercase ID such as APP-02.',
        };
      }
      if (!op.refusal.failureText.trim()) {
        return {
          ok: false,
          code: 'contract.negative',
          message: `Refusal "${id}" must specify its exact refusal message (R5).`,
          remediation: 'Provide the exact refusal sentence.',
        };
      }
      if (decl.features.some((f) => f.id === id) || decl.refusals.some((r) => r.id === id)) {
        return {
          ok: false,
          code: 'admission.duplicate',
          message: `ID "${id}" is already registered.`,
          remediation: 'Use a unique ID for each refusal.',
        };
      }
      const admittedSet = new Set(decl.admissions.map((a) => a.construct));
      const citCheck = citationResolves(decl, admittedSet, op.refusal.citation);
      if (!citCheck.ok) {
        return {
          ok: false,
          code: 'admission.unresolved',
          message: `Refusal "${id}" refused: ${citCheck.detail}`,
          remediation: 'Cite an admitted Requirement or staged Authority (D12).',
        };
      }
      const refusal: StudioRefusal = {
        ...op.refusal,
        id,
        statement: op.refusal.statement.trim() || `Refuse invalid input (${id})`,
        failureText: op.refusal.failureText.trim(),
        negativeId: op.refusal.negativeId?.trim().toUpperCase() || id,
      };
      const next: PrismDeclaration = {
        ...decl,
        refusals: [...decl.refusals, refusal],
        admissions: upsertAdmission(decl.admissions, `Refusal.${id}`, refusal.citation),
        auditLog: appendAudit(decl, op.op, `Admitted refusal ${id}`),
      };
      return { ok: true, declaration: next, message: `Admitted typed refusal ${id}.` };
    }

    case 'update-refusal': {
      const existing = decl.refusals.find((r) => r.id === op.id);
      if (!existing) {
        return {
          ok: false,
          code: 'refusal.not-found',
          message: `Refusal "${op.id}" does not exist.`,
          remediation: 'Admit the refusal first.',
        };
      }
      const nextFailure = op.patch.failureText !== undefined ? op.patch.failureText.trim() : existing.failureText;
      if (!nextFailure) {
        return {
          ok: false,
          code: 'contract.negative',
          message: `Refusal "${op.id}" cannot have an empty refusal message.`,
          remediation: 'Provide the exact refusal sentence (R5).',
        };
      }
      const next: PrismDeclaration = {
        ...decl,
        refusals: decl.refusals.map((r) =>
          r.id === op.id ? { ...r, ...op.patch, failureText: nextFailure } : r,
        ),
        auditLog: appendAudit(decl, op.op, `Updated refusal ${op.id}`),
      };
      return { ok: true, declaration: next, message: `Updated refusal ${op.id}.` };
    }

    case 'remove-refusal': {
      if (decl.lockedIds.includes(op.id)) {
        return {
          ok: false,
          code: 'admission.monotone',
          message: `Cannot remove locked refusal "${op.id}" (P10).`,
          remediation: 'Locked refusals are permanent safety guarantees.',
        };
      }
      const next: PrismDeclaration = {
        ...decl,
        refusals: decl.refusals.filter((r) => r.id !== op.id),
        admissions: decl.admissions.filter((a) => a.construct !== `Refusal.${op.id}`),
        auditLog: appendAudit(decl, op.op, `Removed draft refusal ${op.id}`),
      };
      return { ok: true, declaration: next, message: `Removed refusal ${op.id}.` };
    }

    case 'admit-vector': {
      const request =
        op.vector.request !== undefined
          ? op.vector.request
          : formatRequestLine(decl.encoding, op.vector.inputs);
      const vector: AcceptanceVectorDecl = {
        ...op.vector,
        id: op.vector.id.trim() || `vec_${decl.vectors.length + 1}`,
        request,
      };
      const idx = decl.vectors.findIndex((v) => v.id === vector.id);
      const nextVectors = idx >= 0
        ? decl.vectors.map((v, i) => (i === idx ? vector : v))
        : [...decl.vectors, vector];
      const next: PrismDeclaration = {
        ...decl,
        vectors: nextVectors,
        auditLog: appendAudit(decl, op.op, `Admitted example ${vector.id} ("${vector.request}" → "${vector.expectedResponse}")`),
      };
      return { ok: true, declaration: next, message: `Admitted acceptance example "${vector.id}".` };
    }

    case 'remove-vector': {
      const next: PrismDeclaration = {
        ...decl,
        vectors: decl.vectors.filter((v) => v.id !== op.id),
        auditLog: appendAudit(decl, op.op, `Removed example ${op.id}`),
      };
      return { ok: true, declaration: next, message: `Removed example "${op.id}".` };
    }

    case 'declare-view': {
      const reqIds = new Set(decl.requirements.map((r) => r.id));
      for (const el of op.view.elements) {
        if (!reqIds.has(el.requirementId)) {
          return {
            ok: false,
            code: 'view.uncited-element',
            message: `View element "${el.label}" cites unknown requirement "${el.requirementId}" (REQ-D9).`,
            remediation: 'Every UI element must trace to an admitted Requirement.',
            brokenRequirementId: el.requirementId,
          };
        }
      }
      const next: PrismDeclaration = {
        ...decl,
        view: op.view,
        admissions: upsertAdmission(decl.admissions, `View.${op.view.viewpoint}`, op.view.citation),
        auditLog: appendAudit(decl, op.op, `Declared ${op.view.viewpoint} view (${op.view.elements.length} elements)`),
      };
      return { ok: true, declaration: next, message: `Declared view "${op.view.title}" (${op.view.viewpoint}).` };
    }

    case 'set-theme': {
      const validPreset = PRISM_THEME_PRESETS.some((p) => p.id === op.theme.preset);
      if (!validPreset) {
        return {
          ok: false,
          code: 'design.theme',
          message: `Unknown theme preset "${op.theme.preset}".`,
          remediation: `Choose one of: ${PRISM_THEME_PRESETS.map((p) => p.id).join(', ')}.`,
        };
      }
      if (op.theme.accent && !HEX_COLOR.test(op.theme.accent)) {
        return {
          ok: false,
          code: 'design.theme',
          message: `Invalid accent colour "${op.theme.accent}".`,
          remediation: 'Use a hex colour such as #1f5e44.',
        };
      }
      const next: PrismDeclaration = {
        ...decl,
        theme: op.theme,
        auditLog: appendAudit(decl, op.op, `Set theme preset to ${op.theme.preset}`),
      };
      return { ok: true, declaration: next, message: `Theme updated to ${op.theme.preset}.` };
    }

    case 'set-capabilities': {
      const capabilities: PrismGrants = {
        observer: op.capabilities.observer ?? decl.capabilities.observer,
        storage: op.capabilities.storage ?? decl.capabilities.storage,
        browser: op.capabilities.browser ?? decl.capabilities.browser,
        network: op.capabilities.network ?? decl.capabilities.network,
      };
      const next: PrismDeclaration = {
        ...decl,
        capabilities,
        auditLog: appendAudit(decl, op.op, `Updated capability grants`),
      };
      return { ok: true, declaration: next, message: 'Updated prism.toml capability grants.' };
    }

    case 'lock-registers': {
      const locked = Array.from(new Set([...decl.lockedIds, ...allRegisteredIds(decl)])).sort();
      const next: PrismDeclaration = {
        ...decl,
        lockedIds: locked,
        auditLog: appendAudit(decl, op.op, `Locked ${locked.length} register IDs`),
      };
      return { ok: true, declaration: next, message: `Locked ${locked.length} register IDs.` };
    }

    case 'record-error': {
      const suggestedRepair = deriveRepairOp(
        decl,
        op.error.code,
        op.error.refusal,
        op.error.brokenRequirementId,
      );
      const record: StudioErrorRecord = {
        ...op.error,
        id: `err_${Date.now().toString(36)}_${decl.errors.length + 1}`,
        at: new Date().toISOString(),
        resolved: false,
        suggestedRepair,
      };
      const next: PrismDeclaration = {
        ...decl,
        errors: [...decl.errors.slice(-19), record],
        auditLog: appendAudit(decl, op.op, `Recorded error ${record.code}`),
      };
      return { ok: true, declaration: next, message: `Recorded error ${record.code}.` };
    }

    case 'resolve-error': {
      const next: PrismDeclaration = {
        ...decl,
        errors: decl.errors.map((e) => (e.id === op.id ? { ...e, resolved: true } : e)),
        auditLog: appendAudit(decl, op.op, `Resolved error ${op.id}`),
      };
      return { ok: true, declaration: next, message: `Resolved error ${op.id}.` };
    }
  }
}

// ============================================================================
// Curated Declarative Starters (Built 100% through dispatchStudioOp)
// ============================================================================

function mustApply(decl: PrismDeclaration, ops: readonly StudioApiOp[]): PrismDeclaration {
  const res = dispatchStudioOp(decl, { op: 'batch', ops });
  if (!res.ok) {
    throw new Error(`Starter build failed [${res.code}]: ${res.message}`);
  }
  return res.declaration;
}

export function createDeclarativeStarter(kind: DeclarativeStarterId): PrismDeclaration {
  if (kind === 'budget-guard') {
    const base = createEmptyDeclaration('Monthly Budget Guard');
    return mustApply(base, [
      {
        op: 'set-metadata',
        name: 'Monthly Budget Guard',
        description: 'Checks every expense against a monthly budget with proved remaining-balance math and zero/over-limit refusals.',
      },
      {
        op: 'admit-requirement',
        requirement: {
          id: 'REQ-02',
          statement: 'Refuse zero or over-cap expenses before computing remaining budget.',
          stakeholder: 'creator',
          origin: 'constraint',
          evidence: 'Owner safety invariant',
        },
      },
      {
        op: 'propose-encoding',
        name: 'ExpenseCheck',
        fields: [
          { name: 'budget', label: 'Monthly Budget', kind: 'nat', min: 1, max: 100000, unit: 'USD', defaultValue: 2000 },
          { name: 'spent', label: 'Spent So Far', kind: 'nat', min: 0, max: 100000, unit: 'USD', defaultValue: 850 },
          { name: 'expense', label: 'New Expense', kind: 'nat', min: 0, max: 100000, unit: 'USD', defaultValue: 120 },
          { name: 'category', label: 'Category', kind: 'choice', min: 0, max: 3, choices: ['Food', 'Home', 'Travel', 'Fun'], defaultValue: 0 },
        ],
        citation: { kind: 'req', requirementId: 'REQ-01' },
      },
      {
        op: 'admit-refusal',
        refusal: {
          id: 'APP-01',
          statement: 'Refuse an expense of zero.',
          condition: { all: [{ field: 'expense', op: 'eq', value: 0 }] },
          failureText: 'Expense must be greater than zero.',
          fieldHint: 'expense',
          negativeId: 'APP-01',
          citation: { kind: 'req', requirementId: 'REQ-02' },
        },
      },
      {
        op: 'admit-refusal',
        refusal: {
          id: 'APP-02',
          statement: 'Refuse a zero monthly budget.',
          condition: { all: [{ field: 'budget', op: 'eq', value: 0 }] },
          failureText: 'Monthly budget must be greater than zero.',
          fieldHint: 'budget',
          negativeId: 'APP-02',
          citation: { kind: 'req', requirementId: 'REQ-02' },
        },
      },
      {
        op: 'admit-feature',
        feature: {
          id: 'APP-03',
          level: 'build',
          statement: 'Flag when the new expense alone exceeds the monthly budget.',
          condition: { all: [{ field: 'expense', op: 'gt', value: 0, rhsField: 'budget' }] },
          outcome: { kind: 'label', code: 1, label: 'Over Budget', tone: 'destructive' },
          priority: 1,
          citation: { kind: 'req', requirementId: 'REQ-02' },
        },
      },
      {
        op: 'admit-feature',
        feature: {
          id: 'APP-04',
          level: 'build',
          statement: 'Compute remaining budget after adding the new expense.',
          condition: { all: [] },
          outcome: {
            kind: 'formula',
            label: 'Remaining Budget',
            unit: 'USD',
            expr: {
              op: 'sub',
              left: { op: 'field', field: 'budget' },
              right: {
                op: 'add',
                left: { op: 'field', field: 'spent' },
                right: { op: 'field', field: 'expense' },
              },
            },
          },
          priority: 10,
          citation: { kind: 'req', requirementId: 'REQ-01' },
        },
      },
      {
        op: 'admit-vector',
        vector: {
          id: 'vec_within',
          inputs: { budget: 2000, spent: 850, expense: 120, category: 0 },
          expectedResponse: 'Remaining Budget: 1030 USD',
          forFeatureId: 'APP-04',
          isRefusal: false,
        },
      },
      {
        op: 'admit-vector',
        vector: {
          id: 'vec_over',
          inputs: { budget: 500, spent: 100, expense: 650, category: 2 },
          expectedResponse: 'Over Budget',
          forFeatureId: 'APP-03',
          isRefusal: false,
        },
      },
      {
        op: 'admit-vector',
        vector: {
          id: 'vec_zero_expense',
          inputs: { budget: 2000, spent: 850, expense: 0, category: 0 },
          expectedResponse: 'Expense must be greater than zero.',
          forFeatureId: 'APP-01',
          isRefusal: true,
        },
      },
      {
        op: 'admit-vector',
        vector: {
          id: 'vec_zero_budget',
          inputs: { budget: 0, spent: 100, expense: 50, category: 1 },
          expectedResponse: 'Monthly budget must be greater than zero.',
          forFeatureId: 'APP-02',
          isRefusal: true,
        },
      },
      {
        op: 'declare-view',
        view: {
          viewpoint: 'ledger-tracker',
          title: 'Monthly Budget Guard',
          eyebrow: 'Personal Finance',
          subtitle: 'Every remaining balance and refusal is computed by the proved model.',
          submitLabel: 'Check Expense',
          elements: [
            { id: 'el_budget', slot: 'field-input', label: 'Monthly Budget', binding: 'budget', requirementId: 'REQ-01' },
            { id: 'el_spent', slot: 'field-input', label: 'Spent So Far', binding: 'spent', requirementId: 'REQ-01' },
            { id: 'el_expense', slot: 'field-input', label: 'New Expense', binding: 'expense', requirementId: 'REQ-02' },
            { id: 'el_category', slot: 'choice-group', label: 'Category', binding: 'category', requirementId: 'REQ-01' },
            { id: 'el_result', slot: 'result-card', label: 'Remaining Balance', binding: 'outcome', requirementId: 'REQ-01' },
            { id: 'el_history', slot: 'history-list', label: 'Recent Checks', binding: 'history', requirementId: 'REQ-01' },
          ],
          citation: { kind: 'req', requirementId: 'REQ-01' },
        },
      },
      { op: 'set-theme', theme: { preset: 'evergreen' } },
    ]);
  }

  if (kind === 'pricing-rules') {
    const base = createEmptyDeclaration('Tiered Discount Engine');
    return mustApply(base, [
      {
        op: 'set-metadata',
        name: 'Tiered Discount Engine',
        description: 'Computes net order totals with proved VIP and bulk discount rules.',
      },
      {
        op: 'admit-requirement',
        requirement: {
          id: 'REQ-02',
          statement: 'Apply a 20% discount to VIP orders and 10% to bulk orders of 10 or more units.',
          stakeholder: 'creator',
          origin: 'identity',
          evidence: 'Pricing policy specification',
        },
      },
      {
        op: 'propose-encoding',
        name: 'OrderQuote',
        fields: [
          { name: 'unitPrice', label: 'Unit Price', kind: 'nat', min: 1, max: 50000, unit: 'USD', defaultValue: 50 },
          { name: 'quantity', label: 'Quantity', kind: 'nat', min: 0, max: 1000, defaultValue: 12 },
          { name: 'tier', label: 'Customer Tier', kind: 'choice', min: 0, max: 1, choices: ['Standard', 'VIP'], defaultValue: 0 },
        ],
        citation: { kind: 'req', requirementId: 'REQ-01' },
      },
      {
        op: 'admit-refusal',
        refusal: {
          id: 'APP-01',
          statement: 'Refuse zero quantity orders.',
          condition: { all: [{ field: 'quantity', op: 'eq', value: 0 }] },
          failureText: 'Quantity must be at least 1.',
          fieldHint: 'quantity',
          negativeId: 'APP-01',
          citation: { kind: 'req', requirementId: 'REQ-01' },
        },
      },
      {
        op: 'admit-feature',
        feature: {
          id: 'APP-02',
          level: 'build',
          statement: 'VIP orders receive a 20% discount off gross subtotal.',
          condition: { all: [{ field: 'tier', op: 'eq', value: 1 }] },
          outcome: {
            kind: 'formula',
            label: 'VIP Net Total',
            unit: 'USD',
            expr: {
              op: 'pct',
              left: { op: 'mul', left: { op: 'field', field: 'unitPrice' }, right: { op: 'field', field: 'quantity' } },
              right: { op: 'const', value: 80 },
            },
          },
          priority: 1,
          citation: { kind: 'req', requirementId: 'REQ-02' },
        },
      },
      {
        op: 'admit-feature',
        feature: {
          id: 'APP-03',
          level: 'build',
          statement: 'Standard bulk orders of 10 or more units receive a 10% discount.',
          condition: { all: [{ field: 'quantity', op: 'gte', value: 10 }] },
          outcome: {
            kind: 'formula',
            label: 'Bulk Net Total',
            unit: 'USD',
            expr: {
              op: 'pct',
              left: { op: 'mul', left: { op: 'field', field: 'unitPrice' }, right: { op: 'field', field: 'quantity' } },
              right: { op: 'const', value: 90 },
            },
          },
          priority: 2,
          citation: { kind: 'req', requirementId: 'REQ-02' },
        },
      },
      {
        op: 'admit-feature',
        feature: {
          id: 'APP-04',
          level: 'build',
          statement: 'Standard small orders pay the full subtotal.',
          condition: { all: [] },
          outcome: {
            kind: 'formula',
            label: 'Standard Total',
            unit: 'USD',
            expr: { op: 'mul', left: { op: 'field', field: 'unitPrice' }, right: { op: 'field', field: 'quantity' } },
          },
          priority: 3,
          citation: { kind: 'req', requirementId: 'REQ-01' },
        },
      },
      {
        op: 'admit-vector',
        vector: {
          id: 'vec_vip',
          inputs: { unitPrice: 50, quantity: 4, tier: 1 },
          expectedResponse: 'VIP Net Total: 160 USD',
          forFeatureId: 'APP-02',
          isRefusal: false,
        },
      },
      {
        op: 'admit-vector',
        vector: {
          id: 'vec_bulk',
          inputs: { unitPrice: 50, quantity: 10, tier: 0 },
          expectedResponse: 'Bulk Net Total: 450 USD',
          forFeatureId: 'APP-03',
          isRefusal: false,
        },
      },
      {
        op: 'admit-vector',
        vector: {
          id: 'vec_zero_qty',
          inputs: { unitPrice: 50, quantity: 0, tier: 0 },
          expectedResponse: 'Quantity must be at least 1.',
          forFeatureId: 'APP-01',
          isRefusal: true,
        },
      },
      {
        op: 'declare-view',
        view: {
          viewpoint: 'calculator',
          title: 'Tiered Discount Engine',
          eyebrow: 'Pricing Calculator',
          subtitle: 'Exact net totals verified by the model.',
          submitLabel: 'Calculate Quote',
          elements: [
            { id: 'el_unit', slot: 'field-input', label: 'Unit Price', binding: 'unitPrice', requirementId: 'REQ-01' },
            { id: 'el_qty', slot: 'field-input', label: 'Quantity', binding: 'quantity', requirementId: 'REQ-02' },
            { id: 'el_tier', slot: 'choice-group', label: 'Customer Tier', binding: 'tier', requirementId: 'REQ-02' },
            { id: 'el_result', slot: 'result-card', label: 'Quoted Total', binding: 'outcome', requirementId: 'REQ-01' },
          ],
          citation: { kind: 'req', requirementId: 'REQ-01' },
        },
      },
      { op: 'set-theme', theme: { preset: 'paper' } },
    ]);
  }

  if (kind === 'eligibility-gate') {
    const base = createEmptyDeclaration('Access & Plan Gate');
    return mustApply(base, [
      {
        op: 'set-metadata',
        name: 'Access & Plan Gate',
        description: 'Classifies membership applications into Approved, Manual Review, or Denied.',
      },
      {
        op: 'admit-requirement',
        requirement: {
          id: 'REQ-02',
          statement: 'Refuse applicants under 18 years old.',
          stakeholder: 'creator',
          origin: 'constraint',
          evidence: 'Minimum age policy',
        },
      },
      {
        op: 'propose-encoding',
        name: 'Applicant',
        fields: [
          { name: 'age', label: 'Applicant Age', kind: 'nat', min: 0, max: 120, unit: 'yrs', defaultValue: 26 },
          { name: 'score', label: 'Trust Score', kind: 'nat', min: 0, max: 100, defaultValue: 82 },
          { name: 'plan', label: 'Requested Plan', kind: 'choice', min: 0, max: 2, choices: ['Starter', 'Pro', 'Enterprise'], defaultValue: 1 },
        ],
        citation: { kind: 'req', requirementId: 'REQ-01' },
      },
      {
        op: 'admit-refusal',
        refusal: {
          id: 'APP-01',
          statement: 'Refuse applicants younger than 18.',
          condition: { all: [{ field: 'age', op: 'lt', value: 18 }] },
          failureText: 'Applicant must be at least 18 years old.',
          fieldHint: 'age',
          negativeId: 'APP-01',
          citation: { kind: 'req', requirementId: 'REQ-02' },
        },
      },
      {
        op: 'admit-feature',
        feature: {
          id: 'APP-02',
          level: 'build',
          statement: 'Approve applicants with trust score 75 or higher.',
          condition: { all: [{ field: 'score', op: 'gte', value: 75 }] },
          outcome: { kind: 'label', code: 1, label: 'Approved', tone: 'success' },
          priority: 1,
          citation: { kind: 'req', requirementId: 'REQ-01' },
        },
      },
      {
        op: 'admit-feature',
        feature: {
          id: 'APP-03',
          level: 'build',
          statement: 'Route applicants with trust score 50–74 to manual review.',
          condition: { all: [{ field: 'score', op: 'gte', value: 50 }] },
          outcome: { kind: 'label', code: 2, label: 'Manual Review', tone: 'warning' },
          priority: 2,
          citation: { kind: 'req', requirementId: 'REQ-01' },
        },
      },
      {
        op: 'admit-feature',
        feature: {
          id: 'APP-04',
          level: 'build',
          statement: 'Deny applicants with trust score below 50.',
          condition: { all: [] },
          outcome: { kind: 'label', code: 3, label: 'Denied', tone: 'destructive' },
          priority: 3,
          citation: { kind: 'req', requirementId: 'REQ-01' },
        },
      },
      {
        op: 'admit-vector',
        vector: {
          id: 'vec_approved',
          inputs: { age: 26, score: 82, plan: 1 },
          expectedResponse: 'Approved',
          forFeatureId: 'APP-02',
          isRefusal: false,
        },
      },
      {
        op: 'admit-vector',
        vector: {
          id: 'vec_underage',
          inputs: { age: 16, score: 90, plan: 0 },
          expectedResponse: 'Applicant must be at least 18 years old.',
          forFeatureId: 'APP-01',
          isRefusal: true,
        },
      },
      {
        op: 'declare-view',
        view: {
          viewpoint: 'decision-form',
          title: 'Access & Plan Gate',
          eyebrow: 'Policy Gate',
          subtitle: 'Deterministic eligibility verdicts backed by proved rules.',
          submitLabel: 'Evaluate Application',
          elements: [
            { id: 'el_age', slot: 'field-input', label: 'Applicant Age', binding: 'age', requirementId: 'REQ-02' },
            { id: 'el_score', slot: 'field-input', label: 'Trust Score', binding: 'score', requirementId: 'REQ-01' },
            { id: 'el_plan', slot: 'choice-group', label: 'Requested Plan', binding: 'plan', requirementId: 'REQ-01' },
            { id: 'el_result', slot: 'result-card', label: 'Verdict', binding: 'outcome', requirementId: 'REQ-01' },
          ],
          citation: { kind: 'req', requirementId: 'REQ-01' },
        },
      },
      { op: 'set-theme', theme: { preset: 'midnight' } },
    ]);
  }

  if (kind === 'split-calc') {
    const base = createEmptyDeclaration('Bill & Tip Splitter');
    return mustApply(base, [
      {
        op: 'set-metadata',
        name: 'Bill & Tip Splitter',
        description: 'Splits a bill with tip evenly across diners, refusing zero diners or a zero bill.',
      },
      {
        op: 'propose-encoding',
        name: 'SplitRequest',
        fields: [
          { name: 'bill', label: 'Bill Amount', kind: 'nat', min: 0, max: 100000, unit: 'USD', defaultValue: 120 },
          { name: 'tipPct', label: 'Tip Percent', kind: 'nat', min: 0, max: 100, unit: '%', defaultValue: 20 },
          { name: 'diners', label: 'Diners', kind: 'nat', min: 0, max: 50, defaultValue: 4 },
        ],
        citation: { kind: 'req', requirementId: 'REQ-01' },
      },
      {
        op: 'admit-refusal',
        refusal: {
          id: 'APP-01',
          statement: 'Refuse zero diners.',
          condition: { all: [{ field: 'diners', op: 'eq', value: 0 }] },
          failureText: 'Diners must be at least 1.',
          fieldHint: 'diners',
          negativeId: 'APP-01',
          citation: { kind: 'req', requirementId: 'REQ-01' },
        },
      },
      {
        op: 'admit-refusal',
        refusal: {
          id: 'APP-02',
          statement: 'Refuse a zero bill.',
          condition: { all: [{ field: 'bill', op: 'eq', value: 0 }] },
          failureText: 'Bill amount must be greater than zero.',
          fieldHint: 'bill',
          negativeId: 'APP-02',
          citation: { kind: 'req', requirementId: 'REQ-01' },
        },
      },
      {
        op: 'admit-feature',
        feature: {
          id: 'APP-03',
          level: 'build',
          statement: 'Compute each diner share including tip.',
          condition: { all: [] },
          outcome: {
            kind: 'formula',
            label: 'Per Person',
            unit: 'USD',
            expr: {
              op: 'div',
              left: {
                op: 'add',
                left: { op: 'field', field: 'bill' },
                right: { op: 'pct', left: { op: 'field', field: 'bill' }, right: { op: 'field', field: 'tipPct' } },
              },
              right: { op: 'field', field: 'diners' },
            },
          },
          priority: 1,
          citation: { kind: 'req', requirementId: 'REQ-01' },
        },
      },
      {
        op: 'admit-vector',
        vector: {
          id: 'vec_split_4',
          inputs: { bill: 120, tipPct: 20, diners: 4 },
          expectedResponse: 'Per Person: 36 USD',
          forFeatureId: 'APP-03',
          isRefusal: false,
        },
      },
      {
        op: 'admit-vector',
        vector: {
          id: 'vec_zero_diners',
          inputs: { bill: 120, tipPct: 20, diners: 0 },
          expectedResponse: 'Diners must be at least 1.',
          forFeatureId: 'APP-01',
          isRefusal: true,
        },
      },
      {
        op: 'admit-vector',
        vector: {
          id: 'vec_zero_bill',
          inputs: { bill: 0, tipPct: 15, diners: 2 },
          expectedResponse: 'Bill amount must be greater than zero.',
          forFeatureId: 'APP-02',
          isRefusal: true,
        },
      },
      {
        op: 'declare-view',
        view: {
          viewpoint: 'calculator',
          title: 'Bill & Tip Splitter',
          eyebrow: 'Table Split',
          subtitle: 'Clear per-person totals with zero-diner safety proved by the model.',
          submitLabel: 'Split Bill',
          elements: [
            { id: 'el_bill', slot: 'field-input', label: 'Bill Amount', binding: 'bill', requirementId: 'REQ-01' },
            { id: 'el_tip', slot: 'field-input', label: 'Tip Percent', binding: 'tipPct', requirementId: 'REQ-01' },
            { id: 'el_diners', slot: 'field-input', label: 'Diners', binding: 'diners', requirementId: 'REQ-01' },
            { id: 'el_result', slot: 'result-card', label: 'Per Person Share', binding: 'outcome', requirementId: 'REQ-01' },
          ],
          citation: { kind: 'req', requirementId: 'REQ-01' },
        },
      },
      { op: 'set-theme', theme: { preset: 'terracotta' } },
    ]);
  }

  // Blank minimal valid starter with 1 field, 1 rule, 1 refusal, and 2 vectors
  const blank = createEmptyDeclaration('New Prism');
  return mustApply(blank, [
    {
      op: 'propose-encoding',
      name: 'Input',
      fields: [{ name: 'amount', label: 'Value', kind: 'nat', min: 0, max: 10000, defaultValue: 10 }],
      citation: { kind: 'req', requirementId: 'REQ-01' },
    },
    {
      op: 'admit-refusal',
      refusal: {
        id: 'APP-01',
        statement: 'Refuse zero value.',
        condition: { all: [{ field: 'amount', op: 'eq', value: 0 }] },
        failureText: 'Value must be greater than zero.',
        fieldHint: 'amount',
        negativeId: 'APP-01',
        citation: { kind: 'req', requirementId: 'REQ-01' },
      },
    },
    {
      op: 'admit-feature',
      feature: {
        id: 'APP-02',
        level: 'build',
        statement: 'Return the accepted value.',
        condition: { all: [] },
        outcome: { kind: 'formula', label: 'Accepted', expr: { op: 'field', field: 'amount' } },
        priority: 1,
        citation: { kind: 'req', requirementId: 'REQ-01' },
      },
    },
    {
      op: 'admit-vector',
      vector: {
        id: 'vec_ok',
        inputs: { amount: 10 },
        expectedResponse: 'Accepted: 10',
        forFeatureId: 'APP-02',
        isRefusal: false,
      },
    },
    {
      op: 'admit-vector',
      vector: {
        id: 'vec_zero',
        inputs: { amount: 0 },
        expectedResponse: 'Value must be greater than zero.',
        forFeatureId: 'APP-01',
        isRefusal: true,
      },
    },
    {
      op: 'declare-view',
      view: {
        viewpoint: 'decision-form',
        title: 'New Prism',
        eyebrow: 'Verified Prism',
        subtitle: 'Every answer is computed deterministically by the declared model.',
        submitLabel: 'Evaluate',
        elements: [
          { id: 'el_amount', slot: 'field-input', label: 'Value', binding: 'amount', requirementId: 'REQ-01' },
          { id: 'el_result', slot: 'result-card', label: 'Answer', binding: 'outcome', requirementId: 'REQ-01' },
        ],
        citation: { kind: 'req', requirementId: 'REQ-01' },
      },
    },
    { op: 'set-theme', theme: { preset: 'evergreen' } },
  ]);
}
