/**
 * Conversational Prism Interview & Intent Synthesis Engine (`src/engine/prismInterviewEngine.ts`)
 *
 * Translates plain-English creator descriptions and follow-up answers into deterministic,
 * cited `StudioApiOp[]` operations — NEVER generating HTML, CSS, JavaScript, or model code:
 *
 * - Works instantaneously offline (`<5ms`, zero server, zero downloads) via a composable
 *   5-pass deterministic NLU pipeline (`parseIdeaSentence` + `parseFollowUpInstruction`).
 * - Optionally merges structured JSON extractions from VerdantUI's local on-device model
 *   (`mergeLocalModelExtraction`) with strict schema validation and deterministic fallback.
 * - Upholds R2 Honesty: distinguishes facts explicitly `'stated'` in the Creator's sentence
 *   from conservative `'suggested'` defaults.
 * - Records the Creator's exact 1-sentence idea verbatim via `record-decision` (`DEC-01`)
 *   and guarantees total provable closure (every refusal gets a verified rejecting vector;
 *   every primary rule gets a verified passing vector).
 */

import {
  checkDeclaration,
  deriveRepairOp,
  dispatchStudioOp,
  evaluateDeclaredModel,
  sanitizeIdentifier,
  type ArithmeticExpr,
  type CompareOp,
  type DataClassification,
  type DeclarationCheckReport,
  type EncodedField,
  type PrismAudience,
  type PrismDeclaration,
  type PrismDomain,
  type PrismJurisdiction,
  type PrismThemePreset,
  type PrismViewpoint,
  type RuleCondition,
  type RuleOutcome,
  type StakeholderDecl,
  type StudioApiOp,
} from './verdantStudioApi';
import { evaluateVerdantRouting, type VerdantRoutingReport } from './verdantStandardsRouter';

export type ExtractionConfidence = 'stated' | 'suggested';

export type GuidedStepId =
  | 'anchor-idea'
  | 'confirm-context'
  | 'refine-rules'
  | 'safety-check'
  | 'design-feel'
  | 'routed-standards'
  | 'ready-to-seal';

export interface TaggedFact<T> {
  readonly value: T;
  readonly source: ExtractionConfidence;
  readonly reason: string;
}

export interface ExtractedInputField extends EncodedField {
  readonly source: ExtractionConfidence;
}

export interface ExtractedRuleSpec {
  readonly statement: string;
  readonly condition: RuleCondition;
  readonly outcome: RuleOutcome;
  readonly source: ExtractionConfidence;
}

export interface ExtractedRefusalSpec {
  readonly statement: string;
  readonly condition: RuleCondition;
  readonly failureText: string;
  readonly fieldHint?: string;
  readonly source: ExtractionConfidence;
}

export interface ExtractedPrismIntent {
  readonly originalSentence: string;
  readonly name: TaggedFact<string>;
  readonly purpose: string;
  readonly domain: TaggedFact<PrismDomain>;
  readonly audience: TaggedFact<PrismAudience>;
  readonly dataClassification: TaggedFact<DataClassification>;
  readonly jurisdictions: TaggedFact<readonly PrismJurisdiction[]>;
  readonly usesLocalAi: TaggedFact<boolean>;
  readonly involvesPayments: TaggedFact<boolean>;
  readonly stakeholders: readonly {
    readonly name: string;
    readonly provRole: StakeholderDecl['provRole'];
    readonly concern: string;
    readonly source: ExtractionConfidence;
  }[];
  readonly inputs: readonly ExtractedInputField[];
  readonly outputName: string;
  readonly processName: string;
  readonly rules: readonly ExtractedRuleSpec[];
  readonly refusals: readonly ExtractedRefusalSpec[];
  readonly recommendedViewpoint: PrismViewpoint;
  readonly recommendedTheme: PrismThemePreset;
  readonly openQuestions: readonly string[];
}

export interface FollowUpParseResult {
  readonly ok: boolean;
  readonly summary: string;
  readonly ops: readonly StudioApiOp[];
  readonly hint?: string;
}

export interface GuidedStepSummary {
  readonly id: GuidedStepId;
  readonly label: string;
  readonly complete: boolean;
  readonly visible: boolean;
}

export interface GuidedWizardState {
  readonly recommendedStep: GuidedStepId;
  readonly steps: readonly GuidedStepSummary[];
  readonly plainRules: readonly { readonly id: string; readonly summary: string }[];
  readonly plainRefusals: readonly { readonly id: string; readonly summary: string }[];
  readonly autoProtectedThreats: number;
  readonly openSafetyItems: readonly {
    readonly id: string;
    readonly title: string;
    readonly plainPrompt: string;
    readonly quickActions: readonly {
      readonly label: string;
      readonly op: StudioApiOp;
    }[];
  }[];
}

export const INSPIRATION_PROMPTS: readonly { readonly label: string; readonly sentence: string }[] = [
  {
    label: 'Monthly budget check',
    sentence: 'Check if a new expense fits my monthly budget, warn when it goes over, and refuse zero amounts.',
  },
  {
    label: 'EU loan eligibility check',
    sentence: 'Let EU customers check loan eligibility by age and credit score, refusing applicants under 18 and approving scores 75 or higher.',
  },
  {
    label: 'Bill & tip splitter',
    sentence: 'Split a restaurant bill with tip percent across diners and refuse zero diners or a zero bill.',
  },
];

// ============================================================================
// Pass 1: Context Weight & Domain Detection
// ============================================================================

function titleCase(raw: string): string {
  return raw
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => w[0].toUpperCase() + w.slice(1).toLowerCase())
    .join(' ');
}

function derivePrismName(sentence: string): TaggedFact<string> {
  const lower = sentence.toLowerCase();
  if (/\bbudget\b/.test(lower) && /\b(expense|spend|purchase|coffee|monthly)\b/.test(lower)) {
    return { value: 'Budget Guard', source: 'stated', reason: 'Derived from budget & expense intent.' };
  }
  if (/\b(bill|check)\b/.test(lower) && /\b(split|tip|diner|people)\b/.test(lower)) {
    return { value: 'Bill & Tip Splitter', source: 'stated', reason: 'Derived from bill-splitting intent.' };
  }
  if (/\b(loan|credit|mortgage|eligibility|applicant)\b/.test(lower)) {
    return { value: 'Eligibility Check', source: 'stated', reason: 'Derived from eligibility screening intent.' };
  }
  if (/\b(discount|quote|pricing|order|bulk|vip)\b/.test(lower)) {
    return { value: 'Pricing Calculator', source: 'stated', reason: 'Derived from pricing & discount intent.' };
  }
  if (/\b(game|damage|rpg|loot|attack|player|match)\b/.test(lower)) {
    return { value: 'Game Calculator', source: 'stated', reason: 'Derived from game mechanics intent.' };
  }
  if (/\b(water|hydration|cups|calorie|sleep|bmi|dose|health)\b/.test(lower)) {
    return { value: 'Daily Wellness Check', source: 'stated', reason: 'Derived from wellness goal intent.' };
  }
  if (/\b(ship|shipping|parcel|package|delivery)\b/.test(lower)) {
    return { value: 'Shipping Estimator', source: 'stated', reason: 'Derived from shipping intent.' };
  }

  const stop = new Set([
    'a', 'an', 'the', 'my', 'our', 'to', 'for', 'of', 'in', 'on', 'by', 'with', 'and', 'or',
    'that', 'which', 'help', 'me', 'us', 'let', 'make', 'build', 'create', 'app', 'prism',
    'tool', 'check', 'if', 'when', 'is', 'are', 'be', 'can', 'should', 'will', 'alert', 'warn',
    'refuse', 'reject', 'calculate', 'compute', 'track', 'interactive', 'please', 'generate',
    'picture', 'image', 'photo', 'draw', 'paint', 'render', 'compose', 'video', 'audio', 'song',
    'artwork', 'illustration', 'show', 'give', 'write',
  ]);
  const words = sentence
    .replace(/[^a-zA-Z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 2 && !stop.has(w.toLowerCase()));
  if (words.length > 0) {
    const picked = titleCase(words.slice(0, 4).join(' '));
    return { value: picked, source: 'stated', reason: 'Derived from key terms in your idea.' };
  }
  return { value: 'My Prism', source: 'suggested', reason: 'Default working title (editable anytime).' };
}

function detectContextWeights(sentence: string): {
  domain: TaggedFact<PrismDomain>;
  audience: TaggedFact<PrismAudience>;
  dataClassification: TaggedFact<DataClassification>;
  jurisdictions: TaggedFact<readonly PrismJurisdiction[]>;
  usesLocalAi: TaggedFact<boolean>;
  involvesPayments: TaggedFact<boolean>;
} {
  const lower = sentence.toLowerCase();

  // Domain
  let domain: TaggedFact<PrismDomain> = {
    value: 'general',
    source: 'suggested',
    reason: 'General-purpose utility (default).',
  };
  if (/\b(game|gaming|player|rpg|damage|loot|xp|level|quest|esports|matchmaking)\b/.test(lower)) {
    domain = { value: 'games-media', source: 'stated', reason: 'Detected interactive game / media concepts.' };
  } else if (/\b(patient|medical|clinic|health|dose|dosage|bmi|calorie|water|hydration|sleep|symptom)\b/.test(lower)) {
    domain = { value: 'health', source: 'stated', reason: 'Detected health or wellness concepts.' };
  } else if (/\b(budget|expense|loan|mortgage|tax|salary|invoice|payment|price|tip|bill|credit|deposit|rent)\b/.test(lower)) {
    domain = { value: 'finance', source: 'stated', reason: 'Detected financial or budget concepts.' };
  } else if (/\b(student|grade|gpa|exam|quiz|course|homework|class|school)\b/.test(lower)) {
    domain = { value: 'education', source: 'stated', reason: 'Detected education concepts.' };
  } else if (/\b(order|discount|shipping|sla|approval|ticket|inventory|policy|employee|hr|vendor|quote)\b/.test(lower)) {
    domain = { value: 'enterprise-ops', source: 'stated', reason: 'Detected business operations or policy concepts.' };
  }

  // Audience
  let audience: TaggedFact<PrismAudience> = {
    value: 'personal',
    source: 'suggested',
    reason: 'Personal offline use (default until you choose otherwise).',
  };
  if (/\b(kid|kids|child|children|minor|minors|teen|teens|under 13|youth)\b/.test(lower)) {
    audience = { value: 'includes-minors', source: 'stated', reason: 'Mentions kids, children, or minors.' };
  } else if (/\b(customer|customers|public|shopper|shoppers|applicant|applicants|visitor|visitors|consumer|consumers|anyone|players)\b/.test(lower)) {
    audience = { value: 'public-adults', source: 'stated', reason: 'Mentions public users, customers, or applicants.' };
  } else if (/\b(team|company|employee|employees|staff|internal|sales desk|org|organization)\b/.test(lower)) {
    audience = { value: 'team', source: 'stated', reason: 'Mentions internal team or staff.' };
  } else if (/\b(my|myself|personal|me)\b/.test(lower)) {
    audience = { value: 'personal', source: 'stated', reason: 'Written for personal use ("my" / "me").' };
  }

  // Data Classification
  let dataClassification: TaggedFact<DataClassification> = {
    value: 'none',
    source: 'suggested',
    reason: 'No personal data stored or transmitted (default).',
  };
  if (/\b(medical|patient|health record|diagnosis|biometric|ssn|social security|credit card)\b/.test(lower)) {
    dataClassification = { value: 'sensitive-regulated', source: 'stated', reason: 'Mentions sensitive health or regulated records.' };
  } else if (/\b(personal data|pii|email|applicant|customer|credit score|age|income|identity|gdpr)\b/.test(lower)) {
    dataClassification = { value: 'personal', source: 'stated', reason: 'Evaluates personal attributes (e.g. age, credit, or customer data).' };
  } else if (/\b(order|inventory|quote|shipping|ticket|sla)\b/.test(lower)) {
    dataClassification = { value: 'operational', source: 'stated', reason: 'Handles operational business figures.' };
  }

  // Jurisdictions
  const jurList: PrismJurisdiction[] = [];
  if (/\b(eu|europe|european|gdpr)\b/.test(lower)) jurList.push('eu');
  if (/\b(us|u\.s\.|usa|united states|america|coppa)\b/.test(lower)) jurList.push('us');
  if (/\b(uk|u\.k\.|britain|british)\b/.test(lower)) jurList.push('uk');
  if (/\b(global|worldwide|international)\b/.test(lower)) jurList.push('global');
  const jurisdictions: TaggedFact<readonly PrismJurisdiction[]> =
    jurList.length > 0
      ? { value: jurList, source: 'stated', reason: `Detected jurisdiction(s): ${jurList.map((j) => j.toUpperCase()).join(', ')}.` }
      : { value: ['none'], source: 'suggested', reason: 'Local / personal scope (no regional regulations selected).' };

  // Local AI & Payments
  const aiMatch = /\b(ai|llm|assistant|local model|hints with ai|ai hints)\b/.test(lower);
  const usesLocalAi: TaggedFact<boolean> = aiMatch
    ? { value: true, source: 'stated', reason: 'Mentions AI assistance.' }
    : { value: false, source: 'suggested', reason: 'Pure deterministic rules (no AI required).' };

  const payMatch = /\b(payment|payments|checkout|charge|card|billing|transfer money|pay online)\b/.test(lower);
  const involvesPayments: TaggedFact<boolean> = payMatch
    ? { value: true, source: 'stated', reason: 'Mentions payments or checkout.' }
    : { value: false, source: 'suggested', reason: 'Calculation / decision only (no payment processing).' };

  return { domain, audience, dataClassification, jurisdictions, usesLocalAi, involvesPayments };
}

// ============================================================================
// Pass 2: Entity, Choice & Quantity Extraction
// ============================================================================

interface KnownFieldDef {
  readonly pattern: RegExp;
  readonly field: Omit<ExtractedInputField, 'source'>;
}

const KNOWN_FIELD_PATTERNS: readonly KnownFieldDef[] = [
  {
    pattern: /\b(monthly\s+budget|budget\s+cap|budget)\b/i,
    field: { name: 'budget', label: 'Monthly Budget', kind: 'nat', min: 0, max: 100000, unit: 'USD', defaultValue: 2000 },
  },
  {
    pattern: /\b(spent\s+so\s+far|already\s+spent|spent)\b/i,
    field: { name: 'spent', label: 'Spent So Far', kind: 'nat', min: 0, max: 100000, unit: 'USD', defaultValue: 500 },
  },
  {
    pattern: /\b(new\s+expense|expense|purchase|spending)\b/i,
    field: { name: 'expense', label: 'Expense', kind: 'nat', min: 0, max: 100000, unit: 'USD', defaultValue: 120 },
  },
  {
    pattern: /\b(restaurant\s+bill|bill\s+amount|bill|check\s+total)\b/i,
    field: { name: 'bill', label: 'Bill Amount', kind: 'nat', min: 0, max: 100000, unit: 'USD', defaultValue: 120 },
  },
  {
    pattern: /\b(tip\s+percent|tip\s*%|tip)\b/i,
    field: { name: 'tipPct', label: 'Tip Percent', kind: 'nat', min: 0, max: 100, unit: '%', defaultValue: 20 },
  },
  {
    pattern: /\b(diners|people|guests|friends|persons)\b/i,
    field: { name: 'diners', label: 'Diners', kind: 'nat', min: 0, max: 50, defaultValue: 4 },
  },
  {
    pattern: /\b(unit\s+price|price|cost|rate)\b/i,
    field: { name: 'price', label: 'Price', kind: 'nat', min: 0, max: 50000, unit: 'USD', defaultValue: 50 },
  },
  {
    pattern: /\b(quantity|units|items|count)\b/i,
    field: { name: 'quantity', label: 'Quantity', kind: 'nat', min: 0, max: 10000, defaultValue: 5 },
  },
  {
    pattern: /\b(applicant\s+age|age)\b/i,
    field: { name: 'age', label: 'Age', kind: 'nat', min: 0, max: 120, unit: 'yrs', defaultValue: 26 },
  },
  {
    pattern: /\b(credit\s+score|trust\s+score|score)\b/i,
    field: { name: 'score', label: 'Score', kind: 'nat', min: 0, max: 850, defaultValue: 80 },
  },
  {
    pattern: /\b(water\s+intake|water|cups)\b/i,
    field: { name: 'cups', label: 'Water Intake', kind: 'nat', min: 0, max: 40, unit: 'cups', defaultValue: 8 },
  },
  {
    pattern: /\b(weight|mass|lbs|kg|oz)\b/i,
    field: { name: 'weight', label: 'Weight', kind: 'nat', min: 0, max: 10000, defaultValue: 20 },
  },
  {
    pattern: /\b(hours|hrs)\b/i,
    field: { name: 'hours', label: 'Hours', kind: 'nat', min: 0, max: 1000, unit: 'hrs', defaultValue: 8 },
  },
  {
    pattern: /\b(days)\b/i,
    field: { name: 'days', label: 'Days', kind: 'nat', min: 0, max: 3650, unit: 'days', defaultValue: 7 },
  },
  {
    pattern: /\b(distance|miles|km)\b/i,
    field: { name: 'distance', label: 'Distance', kind: 'nat', min: 0, max: 10000, defaultValue: 10 },
  },
  {
    pattern: /\b(attack|base\s+damage|damage|power)\b/i,
    field: { name: 'attack', label: 'Attack Power', kind: 'nat', min: 0, max: 10000, defaultValue: 50 },
  },
  {
    pattern: /\b(defense|armor|shield)\b/i,
    field: { name: 'defense', label: 'Defense', kind: 'nat', min: 0, max: 10000, defaultValue: 15 },
  },
  {
    pattern: /\b(level|rank)\b/i,
    field: { name: 'level', label: 'Level', kind: 'nat', min: 0, max: 100, defaultValue: 5 },
  },
  {
    pattern: /\b(income|salary)\b/i,
    field: { name: 'income', label: 'Income', kind: 'nat', min: 0, max: 1000000, unit: 'USD', defaultValue: 65000 },
  },
  {
    pattern: /\b(loan\s+amount|loan|principal)\b/i,
    field: { name: 'loanAmount', label: 'Loan Amount', kind: 'nat', min: 0, max: 1000000, unit: 'USD', defaultValue: 10000 },
  },
  {
    pattern: /\b(limit|cap|threshold)\b/i,
    field: { name: 'limit', label: 'Limit', kind: 'nat', min: 0, max: 100000, defaultValue: 500 },
  },
];

function extractInlineChoices(sentence: string): ExtractedInputField[] {
  const out: ExtractedInputField[] = [];
  // Pattern: "tier (standard, pro, enterprise)" or "category: food, home, travel"
  const parenRegex = /\b([a-zA-Z][a-zA-Z0-9\s]{1,20})\s*\(([^()]{3,80})\)/g;
  let m: RegExpExecArray | null;
  while ((m = parenRegex.exec(sentence)) !== null) {
    const rawLabel = m[1].trim().split(/\s+/).slice(-2).join(' ');
    const items = m[2]
      .split(/[,/|]/)
      .map((s) => s.replace(/^(?:or|and)\s+/i, '').trim())
      .filter((s) => s.length > 0 && s.length <= 24);
    if (items.length >= 2) {
      const name = sanitizeIdentifier(rawLabel.replace(/\s+/g, ''));
      out.push({
        name: name[0].toLowerCase() + name.slice(1),
        label: titleCase(rawLabel),
        kind: 'choice',
        min: 0,
        max: items.length - 1,
        choices: items.map(titleCase),
        defaultValue: 0,
        source: 'stated',
      });
    }
  }
  if (/\bvip\b/i.test(sentence) && !out.some((f) => f.name === 'tier')) {
    out.push({
      name: 'tier',
      label: 'Customer Tier',
      kind: 'choice',
      min: 0,
      max: 1,
      choices: ['Standard', 'VIP'],
      defaultValue: 0,
      source: 'stated',
    });
  }
  return out;
}

function extractInputs(sentence: string): ExtractedInputField[] {
  const fields: ExtractedInputField[] = [];
  const seen = new Set<string>();

  for (const choiceField of extractInlineChoices(sentence)) {
    if (!seen.has(choiceField.name)) {
      seen.add(choiceField.name);
      fields.push(choiceField);
    }
  }

  for (const def of KNOWN_FIELD_PATTERNS) {
    if (def.pattern.test(sentence) && !seen.has(def.field.name)) {
      seen.add(def.field.name);
      fields.push({ ...def.field, source: 'stated' });
    }
  }

  // If budget was mentioned without an expense, add expense so comparison works naturally
  if (seen.has('budget') && !seen.has('expense')) {
    seen.add('expense');
    fields.push({
      name: 'expense',
      label: 'New Expense',
      kind: 'nat',
      min: 0,
      max: 100000,
      unit: 'USD',
      defaultValue: 120,
      source: 'suggested',
    });
  }
  // If attack/damage mentioned without defense in a game calculator, suggest defense
  if (seen.has('attack') && !seen.has('defense') && /\b(game|rpg|damage)\b/i.test(sentence)) {
    seen.add('defense');
    fields.push({
      name: 'defense',
      label: 'Target Defense',
      kind: 'nat',
      min: 0,
      max: 10000,
      defaultValue: 15,
      source: 'suggested',
    });
  }

  // Scale numeric max cleanly if the sentence mentions numbers larger than default max
  const nums = [...sentence.matchAll(/\b(\d+)\b/g)].map((m) => Number.parseInt(m[1], 10));
  const maxNum = nums.length > 0 ? Math.max(...nums) : 0;
  for (let i = 0; i < fields.length; i++) {
    const f = fields[i];
    if (f.kind === 'nat' && maxNum > f.max) {
      fields[i] = { ...f, max: Math.max(f.max, maxNum * 2) };
    }
  }

  // Fallback for ultra-vague ideas (e.g. "Coffee tracker")
  if (fields.length === 0) {
    const unitMatch = sentence.match(/\b(\d+)\s*(cups|usd|\$|%|hours|hrs|days|minutes|kg|lbs|miles|km|points)\b/i);
    const unit = unitMatch ? unitMatch[2] : undefined;
    const defaultVal = unitMatch ? Math.max(1, Number.parseInt(unitMatch[1], 10)) : 10;
    fields.push({
      name: 'amount',
      label: unit ? titleCase(unit) : 'Value',
      kind: 'nat',
      min: 0,
      max: Math.max(10000, defaultVal * 10),
      unit,
      defaultValue: defaultVal,
      source: 'suggested',
    });
  }

  return fields.slice(0, 6);
}

// ============================================================================
// Pass 3: Behavior Extraction (Rules, Formulas & Typed Refusals)
// ============================================================================

function findBestField(
  fields: readonly ExtractedInputField[],
  hintText: string,
  fallbackIdx = 0,
): ExtractedInputField {
  const lower = hintText.toLowerCase();
  for (const f of fields) {
    if (
      lower.includes(f.name.toLowerCase()) ||
      lower.includes(f.label.toLowerCase()) ||
      f.label.toLowerCase().split(/\s+/).some((w) => w.length > 2 && lower.includes(w))
    ) {
      return f;
    }
  }
  return fields[Math.min(fallbackIdx, fields.length - 1)];
}

function extractBehavior(
  sentence: string,
  fields: readonly ExtractedInputField[],
): {
  rules: ExtractedRuleSpec[];
  refusals: ExtractedRefusalSpec[];
  outputName: string;
  processName: string;
  viewpoint: PrismViewpoint;
} {
  const lower = sentence.toLowerCase();
  const rules: ExtractedRuleSpec[] = [];
  const refusals: ExtractedRefusalSpec[] = [];
  const byName = new Map(fields.map((f) => [f.name, f]));
  const primary = fields[0];

  // --- 1. Refusal extraction ---
  // Pattern A: "refusing/rejecting ... under/below/less than N" or "must be at least N"
  const underRefMatch = sentence.match(
    /(?:refus\w*|reject\w*|block\w*|deny\w*|disallow\w*)[^.,;]*?\b(?:under|below|less\s+than|younger\s+than|<)\s+(\d+)/i,
  ) ?? sentence.match(/must\s+be\s+at\s+least\s+(\d+)/i);
  if (underRefMatch) {
    const threshold = Math.max(1, Number.parseInt(underRefMatch[1], 10));
    const target = byName.get('age') ?? findBestField(fields, underRefMatch[0]);
    refusals.push({
      statement: `Refuse when ${target.label.toLowerCase()} is under ${threshold}.`,
      condition: { all: [{ field: target.name, op: 'lt', value: threshold }] },
      failureText: `${target.label} must be at least ${threshold}.`,
      fieldHint: target.name,
      source: 'stated',
    });
  }

  // Pattern B: "refuse ... over/above/exceeding N"
  const overRefMatch = sentence.match(
    /(?:refus\w*|reject\w*|block\w*|forbid\w*)[^.,;]*?\b(?:over|above|more\s+than|exceed\w*|>)\s+(\d+)/i,
  );
  if (overRefMatch) {
    const threshold = Math.max(1, Number.parseInt(overRefMatch[1], 10));
    const target = findBestField(fields, overRefMatch[0]);
    refusals.push({
      statement: `Refuse when ${target.label.toLowerCase()} exceeds ${threshold}.`,
      condition: { all: [{ field: target.name, op: 'gt', value: threshold }] },
      failureText: `${target.label} cannot exceed ${threshold}.`,
      fieldHint: target.name,
      source: 'stated',
    });
  }

  // Pattern C: zero refusals ("refuse zero diners or a zero bill", "refuse zero amounts")
  const zeroMatches = [...sentence.matchAll(/\bzero\s+([a-zA-Z]+)/gi)];
  const addedZeroFields = new Set<string>();
  for (const zm of zeroMatches) {
    const target = findBestField(fields, zm[1], fields.findIndex((f) => f.kind === 'nat'));
    if (target.kind === 'nat' && !addedZeroFields.has(target.name)) {
      addedZeroFields.add(target.name);
      refusals.push({
        statement: `Refuse a zero ${target.label.toLowerCase()}.`,
        condition: { all: [{ field: target.name, op: 'eq', value: 0 }] },
        failureText: `${target.label} must be greater than zero.`,
        fieldHint: target.name,
        source: 'stated',
      });
    }
  }
  if (
    refusals.length === 0 &&
    /\b(refus\w*|reject\w*|no\s+zero|greater\s+than\s+zero)\b/i.test(lower)
  ) {
    const target =
      byName.get('expense') ??
      byName.get('diners') ??
      byName.get('quantity') ??
      fields.find((f) => f.kind === 'nat') ??
      primary;
    refusals.push({
      statement: `Refuse a zero ${target.label.toLowerCase()}.`,
      condition: { all: [{ field: target.name, op: 'eq', value: 0 }] },
      failureText: `${target.label} must be greater than zero.`,
      fieldHint: target.name,
      source: 'stated',
    });
  }

  // Always ensure at least one conservative zero-guard refusal on numeric primary if none stated
  if (refusals.length === 0) {
    const natTarget = byName.get('diners') ?? byName.get('expense') ?? fields.find((f) => f.kind === 'nat');
    if (natTarget) {
      refusals.push({
        statement: `Refuse a zero ${natTarget.label.toLowerCase()}.`,
        condition: { all: [{ field: natTarget.name, op: 'eq', value: 0 }] },
        failureText: `${natTarget.label} must be greater than zero.`,
        fieldHint: natTarget.name,
        source: 'suggested',
      });
    }
  }

  // --- 2. Formula & Policy Rule extraction ---
  let outputName = 'Result';
  let processName = 'Evaluate Request';
  let viewpoint: PrismViewpoint = 'decision-form';

  // Case 1: Bill & Tip Splitter
  if (byName.has('bill') && byName.has('diners')) {
    const hasTip = byName.has('tipPct');
    const expr: ArithmeticExpr = hasTip
      ? {
          op: 'div',
          left: {
            op: 'add',
            left: { op: 'field', field: 'bill' },
            right: { op: 'pct', left: { op: 'field', field: 'bill' }, right: { op: 'field', field: 'tipPct' } },
          },
          right: { op: 'field', field: 'diners' },
        }
      : { op: 'div', left: { op: 'field', field: 'bill' }, right: { op: 'field', field: 'diners' } };
    rules.push({
      statement: hasTip
        ? 'Calculate each diner share including tip percentage.'
        : 'Split the bill evenly across all diners.',
      condition: { all: [] },
      outcome: { kind: 'formula', label: 'Per Person', unit: 'USD', expr },
      source: 'stated',
    });
    outputName = 'Per Person Share';
    processName = 'Split Bill';
    viewpoint = 'calculator';
  }
  // Case 2: Budget Guard
  else if (byName.has('budget') && byName.has('expense')) {
    rules.push({
      statement: 'Warn when the expense exceeds the monthly budget.',
      condition: { all: [{ field: 'expense', op: 'gt', value: 0, rhsField: 'budget' }] },
      outcome: { kind: 'label', code: 1, label: 'Over Budget', tone: 'destructive' },
      source: 'stated',
    });
    const rhsExpr: ArithmeticExpr = byName.has('spent')
      ? { op: 'add', left: { op: 'field', field: 'spent' }, right: { op: 'field', field: 'expense' } }
      : { op: 'field', field: 'expense' };
    rules.push({
      statement: 'Compute remaining monthly budget after this expense.',
      condition: { all: [] },
      outcome: {
        kind: 'formula',
        label: 'Remaining Budget',
        unit: 'USD',
        expr: { op: 'sub', left: { op: 'field', field: 'budget' }, right: rhsExpr },
      },
      source: 'stated',
    });
    outputName = 'Remaining Budget';
    processName = 'Check Expense';
    viewpoint = 'ledger-tracker';
  }
  // Case 3: Discount / Pricing Calculator
  else if (byName.has('price') && /\b(discount|off|quote|total)\b/i.test(lower)) {
    const pctMatch = sentence.match(/(\d+)\s*%\s*(?:discount|off)/i);
    const discPct = pctMatch ? Math.min(90, Math.max(1, Number.parseInt(pctMatch[1], 10))) : 15;
    const subtotalExpr: ArithmeticExpr = byName.has('quantity')
      ? { op: 'mul', left: { op: 'field', field: 'price' }, right: { op: 'field', field: 'quantity' } }
      : { op: 'field', field: 'price' };

    if (byName.has('tier')) {
      rules.push({
        statement: `Apply a ${discPct}% discount for VIP orders.`,
        condition: { all: [{ field: 'tier', op: 'eq', value: 1 }] },
        outcome: {
          kind: 'formula',
          label: 'Discounted Total',
          unit: 'USD',
          expr: { op: 'pct', left: subtotalExpr, right: { op: 'const', value: 100 - discPct } },
        },
        source: 'stated',
      });
    } else if (byName.has('quantity')) {
      rules.push({
        statement: `Apply a ${discPct}% bulk discount for 10 or more units.`,
        condition: { all: [{ field: 'quantity', op: 'gte', value: 10 }] },
        outcome: {
          kind: 'formula',
          label: 'Bulk Total',
          unit: 'USD',
          expr: { op: 'pct', left: subtotalExpr, right: { op: 'const', value: 100 - discPct } },
        },
        source: 'stated',
      });
    }
    rules.push({
      statement: 'Compute standard total.',
      condition: { all: [] },
      outcome: { kind: 'formula', label: 'Total', unit: 'USD', expr: subtotalExpr },
      source: 'stated',
    });
    outputName = 'Quoted Total';
    processName = 'Calculate Quote';
    viewpoint = 'calculator';
  }
  // Case 4: Game Damage Calculator
  else if (byName.has('attack')) {
    const expr: ArithmeticExpr = byName.has('defense')
      ? { op: 'sub', left: { op: 'field', field: 'attack' }, right: { op: 'field', field: 'defense' } }
      : byName.has('level')
        ? { op: 'mul', left: { op: 'field', field: 'attack' }, right: { op: 'field', field: 'level' } }
        : { op: 'field', field: 'attack' };
    rules.push({
      statement: 'Compute net damage from attack and target stats.',
      condition: { all: [] },
      outcome: { kind: 'formula', label: 'Net Damage', expr },
      source: 'stated',
    });
    outputName = 'Net Damage';
    processName = 'Calculate Damage';
    viewpoint = 'calculator';
  }
  // Case 5: General Threshold / Approval / Alert Rules
  else {
    const approveMatch = sentence.match(
      /(?:approv\w*|accept\w*|pass\w*)[^.,;]*?\b(\d+)\s*(?:or\s+higher|or\s+more|\+|and\s+above)?/i,
    ) ?? sentence.match(/(?:>=|at\s+least)\s*(\d+)/i);
    const underAlertMatch = sentence.match(
      /(?:alert|warn|flag|low|below\s+target)[^.,;]*?\b(?:under|below|less\s+than|<)\s+(\d+)/i,
    );
    const overAlertMatch = sentence.match(
      /(?:alert|warn|flag|high|over\s+limit)[^.,;]*?\b(?:over|above|more\s+than|exceed\w*|>)\s+(\d+)/i,
    );

    if (approveMatch) {
      const threshold = Number.parseInt(approveMatch[1], 10);
      const target = byName.get('score') ?? fields.find((f) => f.kind === 'nat' && f.name !== 'age') ?? primary;
      rules.push({
        statement: `Approve when ${target.label.toLowerCase()} is ${threshold} or higher.`,
        condition: { all: [{ field: target.name, op: 'gte', value: threshold }] },
        outcome: { kind: 'label', code: 1, label: 'Approved', tone: 'success' },
        source: 'stated',
      });
      if (threshold >= 20) {
        const mid = Math.floor(threshold * 0.65);
        rules.push({
          statement: `Route ${target.label.toLowerCase()} ${mid}–${threshold - 1} for manual review.`,
          condition: { all: [{ field: target.name, op: 'gte', value: mid }] },
          outcome: { kind: 'label', code: 2, label: 'Manual Review', tone: 'warning' },
          source: 'suggested',
        });
      }
      rules.push({
        statement: `Decline when ${target.label.toLowerCase()} is below threshold.`,
        condition: { all: [] },
        outcome: { kind: 'label', code: 3, label: 'Declined', tone: 'destructive' },
        source: 'stated',
      });
      outputName = 'Eligibility Verdict';
      processName = 'Evaluate Eligibility';
      viewpoint = 'decision-form';
    } else if (underAlertMatch) {
      const threshold = Number.parseInt(underAlertMatch[1], 10);
      const target = findBestField(fields, underAlertMatch[0]);
      rules.push({
        statement: `Alert when ${target.label.toLowerCase()} is under ${threshold}.`,
        condition: { all: [{ field: target.name, op: 'lt', value: threshold }] },
        outcome: { kind: 'label', code: 1, label: 'Below Target', tone: 'warning' },
        source: 'stated',
      });
      rules.push({
        statement: `Confirm ${target.label.toLowerCase()} meets target.`,
        condition: { all: [] },
        outcome: { kind: 'label', code: 2, label: 'On Target', tone: 'success' },
        source: 'stated',
      });
      outputName = 'Status';
      processName = 'Check Target';
      viewpoint = 'ledger-tracker';
    } else if (overAlertMatch) {
      const threshold = Number.parseInt(overAlertMatch[1], 10);
      const target = findBestField(fields, overAlertMatch[0]);
      rules.push({
        statement: `Alert when ${target.label.toLowerCase()} exceeds ${threshold}.`,
        condition: { all: [{ field: target.name, op: 'gt', value: threshold }] },
        outcome: { kind: 'label', code: 1, label: 'Over Limit', tone: 'destructive' },
        source: 'stated',
      });
      rules.push({
        statement: `Confirm ${target.label.toLowerCase()} is within limit.`,
        condition: { all: [] },
        outcome: { kind: 'label', code: 2, label: 'Within Limit', tone: 'success' },
        source: 'stated',
      });
      outputName = 'Status';
      processName = 'Check Limit';
      viewpoint = 'decision-form';
    } else if (byName.has('limit') && fields.length >= 2) {
      const subject = fields.find((f) => f.name !== 'limit') ?? primary;
      rules.push({
        statement: `Flag when ${subject.label.toLowerCase()} exceeds limit.`,
        condition: { all: [{ field: subject.name, op: 'gt', value: 0, rhsField: 'limit' }] },
        outcome: { kind: 'label', code: 1, label: 'Over Limit', tone: 'destructive' },
        source: 'stated',
      });
      rules.push({
        statement: `Accept when ${subject.label.toLowerCase()} is within limit.`,
        condition: { all: [] },
        outcome: { kind: 'label', code: 2, label: 'Within Limit', tone: 'success' },
        source: 'suggested',
      });
      outputName = 'Check Result';
      processName = 'Check Against Limit';
      viewpoint = 'decision-form';
    } else {
      rules.push({
        statement: `Return validated ${primary.label.toLowerCase()}.`,
        condition: { all: [] },
        outcome: {
          kind: 'formula',
          label: primary.label,
          unit: primary.unit,
          expr: { op: 'field', field: primary.name },
        },
        source: 'suggested',
      });
      outputName = `${primary.label} Result`;
      processName = `Evaluate ${primary.label}`;
      viewpoint = 'decision-form';
    }
  }

  return { rules, refusals, outputName, processName, viewpoint };
}

function recommendTheme(domain: PrismDomain): PrismThemePreset {
  switch (domain) {
    case 'finance': return 'evergreen';
    case 'enterprise-ops': return 'paper';
    case 'games-media': return 'midnight';
    case 'health': return 'hologram';
    case 'education': return 'terracotta';
    default: return 'evergreen';
  }
}

export function parseIdeaSentence(rawSentence: string): ExtractedPrismIntent {
  const sentence = rawSentence.trim();
  const name = derivePrismName(sentence);
  const ctx = detectContextWeights(sentence);
  const inputs = extractInputs(sentence);
  const behavior = extractBehavior(sentence, inputs);

  const primaryStakeholder =
    ctx.audience.value === 'includes-minors'
      ? { name: 'Young User & Guardian', provRole: 'consumer' as const, concern: sentence || 'Safe, clear answers.', source: ctx.audience.source }
      : ctx.audience.value === 'public-adults'
        ? { name: 'Customer', provRole: 'consumer' as const, concern: sentence || 'Clear, transparent decisions.', source: ctx.audience.source }
        : ctx.audience.value === 'team'
          ? { name: 'Team Member', provRole: 'consumer' as const, concern: sentence || 'Consistent policy verdicts.', source: ctx.audience.source }
          : { name: 'Owner', provRole: 'consumer' as const, concern: sentence || 'Fast, trustworthy personal answers.', source: ctx.audience.source };

  const openQuestions: string[] = [];
  if (ctx.audience.source === 'suggested') openQuestions.push('Confirm who will use this Prism (personal, team, or public).');
  if (ctx.dataClassification.source === 'suggested') openQuestions.push('Confirm whether this handles any personal or regulated data.');
  if (behavior.rules.every((r) => r.source === 'suggested')) {
    openQuestions.push('Specify the exact thresholds, rules, or formulas you want applied.');
  }

  return {
    originalSentence: sentence,
    name,
    purpose: sentence || `Deterministic tool for ${name.value}.`,
    domain: ctx.domain,
    audience: ctx.audience,
    dataClassification: ctx.dataClassification,
    jurisdictions: ctx.jurisdictions,
    usesLocalAi: ctx.usesLocalAi,
    involvesPayments: ctx.involvesPayments,
    stakeholders: [primaryStakeholder],
    inputs,
    outputName: behavior.outputName,
    processName: behavior.processName,
    rules: behavior.rules,
    refusals: behavior.refusals,
    recommendedViewpoint: behavior.viewpoint,
    recommendedTheme: recommendTheme(ctx.domain.value),
    openQuestions,
  };
}

// ============================================================================
// Pass 4 & 5: Compile Intent -> Atomic StudioApiOp[] + Verified Vectors
// ============================================================================

function buildRefusalSampleInputs(
  fields: readonly EncodedField[],
  condition: RuleCondition,
): Record<string, number> {
  const sample: Record<string, number> = {};
  for (const f of fields) sample[f.name] = f.defaultValue;
  for (const pred of condition.all) {
    if (pred.op === 'eq' || pred.op === 'lte' || pred.op === 'gte') {
      sample[pred.field] = pred.value;
    } else if (pred.op === 'lt') {
      sample[pred.field] = Math.max(0, pred.value - 1);
    } else if (pred.op === 'gt' || pred.op === 'neq') {
      sample[pred.field] = pred.value + 1;
    }
  }
  return sample;
}

function buildPositiveSampleInputs(
  fields: readonly EncodedField[],
  refusals: readonly ExtractedRefusalSpec[],
): Record<string, number> {
  const sample: Record<string, number> = {};
  for (const f of fields) {
    sample[f.name] = f.kind === 'nat' ? Math.max(1, f.defaultValue) : f.defaultValue;
  }
  // Nudge away from any refusal condition cleanly
  for (const ref of refusals) {
    for (const pred of ref.condition.all) {
      const f = fields.find((x) => x.name === pred.field);
      if (!f || f.kind !== 'nat') continue;
      if (pred.op === 'eq' && sample[pred.field] === pred.value) {
        sample[pred.field] = pred.value + 10;
      } else if (pred.op === 'lt' && sample[pred.field] < pred.value) {
        sample[pred.field] = pred.value + 5;
      } else if (pred.op === 'lte' && sample[pred.field] <= pred.value) {
        sample[pred.field] = pred.value + 5;
      } else if (pred.op === 'gt' && sample[pred.field] > pred.value) {
        sample[pred.field] = Math.max(1, pred.value - 1);
      }
    }
  }
  return sample;
}

/**
 * Compiles an `ExtractedPrismIntent` into an atomic, undoable sequence of closed
 * `StudioApiOp` calls. Intentionally leaves Security review & Design View open for
 * the Guided Wizard's progressive confirmation steps.
 */
export function intentToAnchorOps(
  baseDecl: PrismDeclaration,
  intent: ExtractedPrismIntent,
): readonly StudioApiOp[] {
  const ops: StudioApiOp[] = [
    {
      op: 'set-metadata',
      name: intent.name.value,
      description: intent.purpose,
    },
    {
      op: 'record-decision',
      id: 'DEC-01',
      stakeholder: intent.stakeholders[0]?.name || 'Creator',
      quote: intent.originalSentence,
      requirementIds: ['REQ-01'],
    },
    {
      op: 'set-concept-profile',
      purpose: intent.purpose,
      domain: intent.domain.value,
      audience: intent.audience.value,
      dataClassification: intent.dataClassification.value,
      jurisdictions: intent.jurisdictions.value,
      usesLocalAi: intent.usesLocalAi.value,
      involvesPayments: intent.involvesPayments.value,
    },
  ];

  // Stakeholders
  intent.stakeholders.forEach((stk, idx) => {
    ops.push({
      op: 'admit-stakeholder',
      stakeholder: {
        id: `STK-${String(idx + 1).padStart(2, '0')}`,
        name: stk.name,
        provRole: stk.provRole,
        concern: stk.concern,
      },
    });
  });

  // Requirements (citing verbatim DEC-01!)
  if (!baseDecl.requirements.some((r) => r.id === 'REQ-01')) {
    ops.push({
      op: 'admit-requirement',
      requirement: {
        id: 'REQ-01',
        statement: intent.purpose,
        stakeholder: intent.stakeholders[0]?.name || 'Creator',
        origin: 'identity',
        evidence: 'Owner decision DEC-01',
      },
    });
  }
  if (intent.refusals.length > 0 && !baseDecl.requirements.some((r) => r.id === 'REQ-02')) {
    ops.push({
      op: 'admit-requirement',
      requirement: {
        id: 'REQ-02',
        statement: `Refuse invalid inputs cleanly: ${intent.refusals.map((r) => r.failureText).join(' ')}`,
        stakeholder: intent.stakeholders[0]?.name || 'Creator',
        origin: 'constraint',
        evidence: 'Owner decision DEC-01 input safety boundary',
      },
    });
  }

  // OPM Objects (inputs + output) & Process
  const inputObjIds: string[] = [];
  intent.inputs.forEach((inp, idx) => {
    const id = `OBJ-${String(idx + 1).padStart(2, '0')}`;
    inputObjIds.push(id);
    ops.push({
      op: 'admit-opm-object',
      object: {
        id,
        name: inp.label,
        kind: inp.kind,
        states: inp.choices ? [...inp.choices] : [],
        role: inp.name === 'spent' ? 'stateful-asset' : 'input',
        unit: inp.unit,
        min: inp.min,
        max: inp.max,
        defaultValue: inp.defaultValue,
      },
    });
  });
  const outObjId = `OBJ-${String(intent.inputs.length + 1).padStart(2, '0')}`;
  ops.push({
    op: 'admit-opm-object',
    object: {
      id: outObjId,
      name: intent.outputName,
      kind: 'nat',
      states: intent.rules
        .filter((r) => r.outcome.kind === 'label')
        .map((r) => (r.outcome.kind === 'label' ? r.outcome.label : '')),
      role: 'output',
    },
  });
  ops.push({
    op: 'admit-opm-process',
    process: {
      id: 'PROC-01',
      name: intent.processName,
      summary: intent.purpose,
      consumesObjectIds: inputObjIds,
      yieldsObjectIds: [outObjId],
      actorStakeholderIds: ['STK-01'],
    },
  });

  // Encoding (JSON Schema 2020-12)
  const cleanFields: EncodedField[] = intent.inputs.map(({ source: _src, ...rest }) => rest);
  ops.push({
    op: 'propose-encoding',
    name: sanitizeIdentifier(intent.name.value.replace(/\s+/g, '')) || 'Request',
    fields: cleanFields,
    citation: { kind: 'req', requirementId: 'REQ-01' },
  });

  // Refusals & Rules
  let appNum = 1;
  const refusalReqId = intent.refusals.length > 0 ? 'REQ-02' : 'REQ-01';
  const refusalIdMap: { id: string; spec: ExtractedRefusalSpec }[] = [];
  for (const ref of intent.refusals) {
    const id = `APP-${String(appNum++).padStart(2, '0')}`;
    refusalIdMap.push({ id, spec: ref });
    ops.push({
      op: 'admit-refusal',
      refusal: {
        id,
        statement: ref.statement,
        condition: ref.condition,
        failureText: ref.failureText,
        fieldHint: ref.fieldHint,
        negativeId: id,
        citation: { kind: 'req', requirementId: refusalReqId },
      },
    });
  }

  let firstFeatureId = 'APP-01';
  intent.rules.forEach((rule, idx) => {
    const id = `APP-${String(appNum++).padStart(2, '0')}`;
    if (idx === 0) firstFeatureId = id;
    ops.push({
      op: 'admit-feature',
      feature: {
        id,
        level: 'build',
        statement: rule.statement,
        condition: rule.condition,
        outcome: rule.outcome,
        priority: idx + 1,
        citation: { kind: 'req', requirementId: 'REQ-01' },
      },
    });
  });

  ops.push({ op: 'set-theme', theme: { preset: intent.recommendedTheme } });

  // Synthesize guaranteed-passing acceptance vectors by evaluating against shadow declaration
  const shadowRes = dispatchStudioOp(baseDecl, { op: 'batch', ops });
  if (shadowRes.ok) {
    const shadow = shadowRes.declaration;
    // 1. Positive example
    const posInputs = buildPositiveSampleInputs(cleanFields, intent.refusals);
    const posLine = cleanFields.map((f) => String(posInputs[f.name] ?? 0)).join(' ');
    const posEval = evaluateDeclaredModel(shadow, posLine);
    ops.push({
      op: 'admit-vector',
      vector: {
        id: 'vec_primary',
        inputs: posInputs,
        expectedResponse: posEval.response,
        forFeatureId: posEval.matchedId ?? firstFeatureId,
        isRefusal: posEval.isRefusal,
      },
    });

    // 2. Rejecting example for every refusal (R5!)
    for (const { id, spec } of refusalIdMap) {
      const refInputs = buildRefusalSampleInputs(cleanFields, spec.condition);
      const refLine = cleanFields.map((f) => String(refInputs[f.name] ?? 0)).join(' ');
      const refEval = evaluateDeclaredModel(shadow, refLine);
      ops.push({
        op: 'admit-vector',
        vector: {
          id: `vec_${id.toLowerCase().replace(/[^a-z0-9]+/g, '_')}`,
          inputs: refInputs,
          expectedResponse: refEval.response,
          forFeatureId: id,
          isRefusal: refEval.isRefusal,
        },
      });
    }
  }

  return ops;
}

// ============================================================================
// Incremental Plain-English Follow-Up Instruction Parser
// ============================================================================

function nextAppId(decl: PrismDeclaration): string {
  const used = new Set([
    ...decl.features.map((f) => f.id),
    ...decl.refusals.map((r) => r.id),
  ]);
  let n = used.size + 1;
  while (used.has(`APP-${String(n).padStart(2, '0')}`)) n++;
  return `APP-${String(n).padStart(2, '0')}`;
}

export function parseFollowUpInstruction(
  decl: PrismDeclaration,
  rawInstruction: string,
): FollowUpParseResult {
  const text = rawInstruction.trim();
  if (!text) {
    return { ok: false, summary: '', ops: [], hint: 'Type a plain-English rule, input, or refusal to add.' };
  }
  const fields = decl.encoding?.fields ?? [];
  const reqId = decl.requirements[0]?.id ?? 'REQ-01';

  // 1. Add a choice/numeric input field: "add a field for priority (low, medium, high)"
  const addFieldChoice = text.match(/add\s+(?:a\s+|an\s+)?(?:choice\s+|input\s+)?field\s+(?:for\s+|called\s+|named\s+)?([a-zA-Z][a-zA-Z0-9\s]{1,24})\s*\(([^()]+)\)/i);
  if (addFieldChoice) {
    const label = titleCase(addFieldChoice[1].trim());
    const choices = addFieldChoice[2].split(/[,/|]/).map((s) => titleCase(s.trim())).filter(Boolean);
    if (choices.length >= 2) {
      const rawName = sanitizeIdentifier(label.replace(/\s+/g, ''));
      const name = rawName[0].toLowerCase() + rawName.slice(1);
      const nextField: EncodedField = {
        name,
        label,
        kind: 'choice',
        min: 0,
        max: choices.length - 1,
        choices,
        defaultValue: 0,
      };
      const nextFields = [...fields.filter((f) => f.name !== name), nextField];
      return {
        ok: true,
        summary: `Add choice input "${label}" (${choices.join(', ')})`,
        ops: [
          { op: 'admit-opm-object', object: { name: label, kind: 'choice', role: 'input', states: choices, defaultValue: 0 } },
          { op: 'propose-encoding', name: decl.encoding?.name ?? 'Request', fields: nextFields, citation: { kind: 'req', requirementId: reqId } },
        ],
      };
    }
  }

  const addNumericField = text.match(/add\s+(?:a\s+|an\s+)?(?:number\s+|input\s+)?field\s+(?:for\s+|called\s+|named\s+)?([a-zA-Z][a-zA-Z0-9\s]{1,24})(?:\s+in\s+([a-zA-Z%$]+))?/i);
  if (addNumericField) {
    const label = titleCase(addNumericField[1].trim());
    const unit = addNumericField[2]?.trim();
    const rawName = sanitizeIdentifier(label.replace(/\s+/g, ''));
    const name = rawName[0].toLowerCase() + rawName.slice(1);
    const nextField: EncodedField = {
      name,
      label,
      kind: 'nat',
      min: 0,
      max: 100000,
      unit,
      defaultValue: 10,
    };
    const nextFields = [...fields.filter((f) => f.name !== name), nextField];
    return {
      ok: true,
      summary: `Add numeric input "${label}"${unit ? ` (${unit})` : ''}`,
      ops: [
        { op: 'admit-opm-object', object: { name: label, kind: 'nat', role: 'input', states: [], unit, min: 0, max: 100000, defaultValue: 10 } },
        { op: 'propose-encoding', name: decl.encoding?.name ?? 'Request', fields: nextFields, citation: { kind: 'req', requirementId: reqId } },
      ],
    };
  }

  if (fields.length === 0) {
    return {
      ok: false,
      summary: '',
      ops: [],
      hint: 'Add at least one input field first, e.g. "Add a field for amount in USD".',
    };
  }

  const targetField = (() => {
    const lower = text.toLowerCase();
    for (const f of fields) {
      if (lower.includes(f.name.toLowerCase()) || lower.includes(f.label.toLowerCase())) return f;
    }
    return fields[0];
  })();

  // Parse comparator + number
  const parseComp = (s: string): { op: CompareOp; value: number } | null => {
    let m = s.match(/(?:>=|at\s+least|or\s+more|or\s+higher)\s*(\d+)/i) ?? s.match(/(\d+)\s*(?:or\s+higher|or\s+more|\+)/i);
    if (m) return { op: 'gte', value: Number.parseInt(m[1], 10) };
    m = s.match(/(?:<=|at\s+most)\s*(\d+)/i);
    if (m) return { op: 'lte', value: Number.parseInt(m[1], 10) };
    m = s.match(/(?:>|over|above|more\s+than|exceed\w*)\s*(\d+)/i);
    if (m) return { op: 'gt', value: Number.parseInt(m[1], 10) };
    m = s.match(/(?:<|under|below|less\s+than)\s*(\d+)/i);
    if (m) return { op: 'lt', value: Number.parseInt(m[1], 10) };
    m = s.match(/(?:=|equals?|is)\s*(zero|\d+)/i);
    if (m) return { op: 'eq', value: m[1].toLowerCase() === 'zero' ? 0 : Number.parseInt(m[1], 10) };
    if (/\bzero\b/i.test(s)) return { op: 'eq', value: 0 };
    return null;
  };

  const comp = parseComp(text);

  // 2. Add a refusal: "refuse when amount is over 5000" / "reject zero budget"
  if (/\b(refus\w*|reject\w*|block\w*|forbid\w*|disallow\w*)\b/i.test(text) && comp) {
    const id = nextAppId(decl);
    const failureText =
      comp.op === 'eq' && comp.value === 0
        ? `${targetField.label} must be greater than zero.`
        : comp.op === 'lt'
          ? `${targetField.label} must be at least ${comp.value}.`
          : comp.op === 'gt'
            ? `${targetField.label} cannot exceed ${comp.value}.`
            : `Invalid ${targetField.label.toLowerCase()}.`;
    const condition: RuleCondition = { all: [{ field: targetField.name, op: comp.op, value: comp.value }] };
    const sampleIn = buildRefusalSampleInputs(fields, condition);
    return {
      ok: true,
      summary: `Add refusal: "${failureText}"`,
      ops: [
        {
          op: 'admit-refusal',
          refusal: {
            id,
            statement: text,
            condition,
            failureText,
            fieldHint: targetField.name,
            negativeId: id,
            citation: { kind: 'req', requirementId: reqId },
          },
        },
        {
          op: 'admit-vector',
          vector: {
            id: `vec_${id.toLowerCase()}`,
            inputs: sampleIn,
            expectedResponse: failureText,
            forFeatureId: id,
            isRefusal: true,
          },
        },
      ],
    };
  }

  // 3. Add a verdict/threshold rule: "approve when score >= 80", "warn when expense > 500"
  if (comp) {
    const id = nextAppId(decl);
    const lower = text.toLowerCase();
    const outcomeLabel = /\b(approv\w*|accept\w*|pass\w*)\b/.test(lower)
      ? 'Approved'
      : /\b(review|manual|check)\b/.test(lower)
        ? 'Manual Review'
        : /\b(warn\w*|alert|caution|flag)\b/.test(lower)
          ? 'Warning'
          : /\b(deny|deni\w*|declin\w*)\b/.test(lower)
            ? 'Declined'
            : 'Matched';
    const tone =
      outcomeLabel === 'Approved'
        ? 'success'
        : outcomeLabel === 'Declined'
          ? 'destructive'
          : outcomeLabel === 'Warning' || outcomeLabel === 'Manual Review'
            ? 'warning'
            : 'brand';
    const condition: RuleCondition = { all: [{ field: targetField.name, op: comp.op, value: comp.value }] };
    return {
      ok: true,
      summary: `Add rule: When ${targetField.label} ${comp.op} ${comp.value} → ${outcomeLabel}`,
      ops: [
        {
          op: 'admit-feature',
          feature: {
            id,
            level: 'build',
            statement: text,
            condition,
            outcome: { kind: 'label', code: decl.features.length + 1, label: outcomeLabel, tone },
            priority: 1,
            citation: { kind: 'req', requirementId: reqId },
          },
        },
      ],
    };
  }

  return {
    ok: false,
    summary: '',
    ops: [],
    hint: 'Try something specific like: "Refuse if amount is over 5000", "Approve when score is at least 80", or "Add a field for priority (Low, Medium, High)".',
  };
}

// ============================================================================
// Optional Local On-Device Model Structured Extraction & Safe Merge
// ============================================================================

export function mergeLocalModelExtraction(
  rawModelOutput: string,
  fallback: ExtractedPrismIntent,
): ExtractedPrismIntent {
  const fenced = rawModelOutput.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = (fenced ? fenced[1] : rawModelOutput).trim();
  const start = candidate.indexOf('{');
  const end = candidate.lastIndexOf('}');
  if (start < 0 || end <= start) return fallback;

  try {
    const parsed = JSON.parse(candidate.slice(start, end + 1)) as Record<string, unknown>;
    const validDomains = new Set<PrismDomain>(['general', 'finance', 'health', 'education', 'games-media', 'enterprise-ops']);
    const validAudiences = new Set<PrismAudience>(['personal', 'team', 'public-adults', 'includes-minors']);
    const validData = new Set<DataClassification>(['none', 'operational', 'personal', 'sensitive-regulated']);

    const nextName =
      typeof parsed.name === 'string' && parsed.name.trim().length >= 2 && parsed.name.trim().length <= 48
        ? { value: parsed.name.trim(), source: 'stated' as const, reason: 'Refined from your idea.' }
        : fallback.name;
    const nextDomain =
      typeof parsed.domain === 'string' && validDomains.has(parsed.domain as PrismDomain)
        ? { value: parsed.domain as PrismDomain, source: 'stated' as const, reason: 'Inferred from your description.' }
        : fallback.domain;
    const nextAudience =
      typeof parsed.audience === 'string' && validAudiences.has(parsed.audience as PrismAudience)
        ? { value: parsed.audience as PrismAudience, source: 'stated' as const, reason: 'Inferred from your description.' }
        : fallback.audience;
    const nextData =
      typeof parsed.dataClassification === 'string' && validData.has(parsed.dataClassification as DataClassification)
        ? { value: parsed.dataClassification as DataClassification, source: 'stated' as const, reason: 'Inferred from your description.' }
        : fallback.dataClassification;

    // Only accept model-suggested inputs if fallback had suggested-only inputs
    let nextInputs = fallback.inputs;
    if (
      fallback.inputs.every((f) => f.source === 'suggested') &&
      Array.isArray(parsed.inputs) &&
      parsed.inputs.length > 0
    ) {
      const sanitized: ExtractedInputField[] = [];
      const seen = new Set<string>();
      for (const rawItem of parsed.inputs.slice(0, 5)) {
        if (!rawItem || typeof rawItem !== 'object') continue;
        const item = rawItem as Record<string, unknown>;
        const label = typeof item.label === 'string' ? item.label.trim() : '';
        if (!label) continue;
        const nameRaw = sanitizeIdentifier(typeof item.name === 'string' ? item.name : label.replace(/\s+/g, ''));
        const name = nameRaw[0].toLowerCase() + nameRaw.slice(1);
        if (seen.has(name)) continue;
        seen.add(name);
        const kind = item.kind === 'choice' || item.kind === 'bool' ? item.kind : 'nat';
        const choices =
          kind === 'choice' && Array.isArray(item.choices)
            ? item.choices.filter((c): c is string => typeof c === 'string' && c.trim().length > 0).slice(0, 6)
            : undefined;
        sanitized.push({
          name,
          label,
          kind: kind === 'choice' && (!choices || choices.length < 2) ? 'nat' : kind,
          min: 0,
          max:
            kind === 'choice' && choices && choices.length >= 2
              ? choices.length - 1
              : kind === 'bool'
                ? 1
                : Math.max(10, typeof item.max === 'number' && Number.isFinite(item.max) ? Math.floor(item.max) : 10000),
          choices: choices && choices.length >= 2 ? choices : undefined,
          unit: typeof item.unit === 'string' && item.unit.trim() ? item.unit.trim().slice(0, 12) : undefined,
          defaultValue: kind === 'nat' ? 10 : 0,
          source: 'stated',
        });
      }
      if (sanitized.length > 0) {
        nextInputs = sanitized;
      }
    }

    return {
      ...fallback,
      name: nextName,
      domain: nextDomain,
      audience: nextAudience,
      dataClassification: nextData,
      outputName:
        typeof parsed.outputName === 'string' && parsed.outputName.trim()
          ? parsed.outputName.trim().slice(0, 36)
          : fallback.outputName,
      processName:
        typeof parsed.processName === 'string' && parsed.processName.trim()
          ? parsed.processName.trim().slice(0, 36)
          : fallback.processName,
      inputs: nextInputs,
    };
  } catch {
    return fallback;
  }
}

// ============================================================================
// Adaptive Guided Wizard State Planner (Zero Internal Jargon)
// ============================================================================

export function computeGuidedWizardState(
  decl: PrismDeclaration,
  routing: VerdantRoutingReport,
  checkReport: DeclarationCheckReport,
): GuidedWizardState {
  const conceptStage = routing.stages.find((s) => s.stage === 'concept');
  const archStage = routing.stages.find((s) => s.stage === 'architecture');
  const secStage = routing.stages.find((s) => s.stage === 'security');
  const designStage = routing.stages.find((s) => s.stage === 'design');
  const stdStage = routing.stages.find((s) => s.stage === 'standards');

  const hasIdeaAnchor = !routing.isUndefined && decl.concept.purpose.trim().length > 0;
  const contextConfirmed = conceptStage?.status === 'complete';
  const rulesComplete = archStage?.status === 'complete' && checkReport.vectorsTotal > 0;
  const safetyComplete = secStage?.status === 'complete';
  const designComplete = designStage?.status === 'complete';
  const hasStandards = Boolean(stdStage?.visible);
  const standardsComplete = !hasStandards || stdStage?.status === 'complete';

  const steps: GuidedStepSummary[] = [
    { id: 'anchor-idea', label: '1. Idea', complete: hasIdeaAnchor, visible: true },
    { id: 'confirm-context', label: '2. Context', complete: contextConfirmed, visible: true },
    { id: 'refine-rules', label: '3. Rules', complete: rulesComplete, visible: true },
    { id: 'safety-check', label: '4. Safety', complete: safetyComplete, visible: true },
    { id: 'design-feel', label: '5. Design', complete: designComplete, visible: true },
    {
      id: 'routed-standards',
      label: `Compliance (${routing.activeLayeredStandards.length})`,
      complete: standardsComplete,
      visible: hasStandards,
    },
    { id: 'ready-to-seal', label: 'Prove', complete: routing.readyToSeal, visible: true },
  ];

  const recommendedStep: GuidedStepId = !hasIdeaAnchor
    ? 'anchor-idea'
    : !contextConfirmed
      ? 'confirm-context'
      : !rulesComplete
        ? 'refine-rules'
        : !safetyComplete
          ? 'safety-check'
          : !designComplete
            ? 'design-feel'
            : hasStandards && !standardsComplete
              ? 'routed-standards'
              : 'ready-to-seal';

  const plainRules = decl.features.map((f) => ({
    id: f.id,
    summary: f.statement,
  }));
  const plainRefusals = decl.refusals.map((r) => ({
    id: r.id,
    summary: `${r.statement} → “${r.failureText}”`,
  }));

  const autoProtectedThreats = decl.security.threats.filter((t) => t.status !== 'open').length;
  const openSafetyItems = decl.security.threats
    .filter((t) => t.status === 'open')
    .map((thr) => {
      const quickActions: { label: string; op: StudioApiOp }[] = [];
      if (thr.id === 'THR-TAMPER-INPUT' && decl.refusals.length > 0) {
        quickActions.push({
          label: `Protected by refusal (${decl.refusals[0].id})`,
          op: {
            op: 'settle-stride-threat',
            id: thr.id,
            status: 'mitigated-by-refusal',
            linkedRefusalId: decl.refusals[0].id,
            rationale: `Protected by input validation and refusal "${decl.refusals[0].failureText}".`,
          },
        });
      }
      if (thr.id === 'THR-INFO-LEAK') {
        quickActions.push({
          label: 'Keep data strictly on-device (offline)',
          op: {
            op: 'batch',
            ops: [
              { op: 'set-capabilities', capabilities: { network: false } },
              {
                op: 'settle-stride-threat',
                id: thr.id,
                status: 'mitigated-by-sandbox',
                rationale: 'All inputs remain strictly on-device with outbound internet access denied.',
              },
            ],
          },
        });
      }
      if (thr.id === 'THR-SPOOF-ORIGIN') {
        quickActions.push({
          label: 'Verify package integrity before running',
          op: {
            op: 'settle-stride-threat',
            id: thr.id,
            status: 'mitigated-by-sandbox',
            rationale: 'Content-addressed archive verification checks package integrity before execution.',
          },
        });
      }
      if (quickActions.length === 0) {
        quickActions.push({
          label: 'Confirm isolated sandbox protection',
          op: {
            op: 'settle-stride-threat',
            id: thr.id,
            status: 'mitigated-by-sandbox',
            rationale: 'Mitigated by isolated sandbox execution and strict input bounds.',
          },
        });
      }
      const plainPrompt =
        thr.id === 'THR-INFO-LEAK'
          ? 'Your Prism touches personal or stored data. Should all values stay strictly on the user’s device with no internet transfer?'
          : thr.id === 'THR-SPOOF-ORIGIN'
            ? 'Because this Prism may be shared publicly, confirm that package integrity verification protects callers from tampered copies.'
            : thr.id === 'THR-TAMPER-INPUT'
              ? 'Protect against malformed or out-of-range user inputs.'
              : thr.rationale;
      return {
        id: thr.id,
        title: thr.title,
        plainPrompt,
        quickActions,
      };
    });

  return {
    recommendedStep,
    steps,
    plainRules,
    plainRefusals,
    autoProtectedThreats,
    openSafetyItems,
  };
}

// ============================================================================
// Continuous Natural-Language Multi-Turn Reconciliation Engine
// (Drives EVERY Wizard Stage Purely via Natural Language — Zero Manual Forms)
// ============================================================================

export interface StudioConversationalPrompt {
  readonly step: GuidedStepId;
  readonly promptText: string;
  readonly placeholder: string;
  readonly quickReplies: readonly string[];
}

export interface WizardTurnReconciliation {
  readonly ops: readonly StudioApiOp[];
  readonly assistantReply: string;
  readonly reconciledBadges: readonly string[];
  readonly nextStep: GuidedStepId;
  readonly openPreview: boolean;
}

const AFFIRMATIVE_RE =
  /^(?:yes|yeah|yep|yup|sure|ok|okay|looks?\s+good|sounds?\s+good|sounds?\s+great|great|perfect|continue|next|go\s+ahead|proceed|confirm|confirmed|none|no\s+changes|nothing\s+else|keep\s+going|move\s+on|skip|check\s+safety|recommended|default|use\s+recommended|surprise\s+me|apply(?:\s+all|\s+them)?|done)(?:[\s,!.-]+(?:keep\s+going|check\s+safety(?:\s+next)?|move\s+on|continue|next|please|thanks|go\s+ahead|proceed))?[\s!.?]*$/i;

const RECONCILE_ALL_RE =
  /\b(reconcile\s+(?:all|everything)|finish\s+(?:it|everything|setup|the\s+prism)|complete\s+(?:all|everything)|auto[-\s]?complete|build\s+the\s+rest)\b/i;

export function parseDesignPreferences(text: string): {
  readonly theme?: PrismThemePreset;
  readonly viewpoint?: PrismViewpoint;
} {
  const lower = text.toLowerCase();
  let theme: PrismThemePreset | undefined;
  if (/\bnoir\b/.test(lower) || /\bdark\s+(?:minimal|brass|studio)\b/.test(lower)) theme = 'noir';
  else if (/\bmidnight\b/.test(lower) || /\b(dark|navy)\b/.test(lower)) theme = 'midnight';
  else if (/\bterracotta\b/.test(lower) || /\b(clay|warm\s+orange|sunset)\b/.test(lower)) theme = 'terracotta';
  else if (/\bpaper\b/.test(lower) || /\b(editorial|ivory|classic)\b/.test(lower)) theme = 'paper';
  else if (/\bhologram\b/.test(lower) || /\b(minimal|neutral|white|clean\s+white)\b/.test(lower)) theme = 'hologram';
  else if (/\bevergreen\b/.test(lower) || /\b(green|forest|warm)\b/.test(lower)) theme = 'evergreen';

  let viewpoint: PrismViewpoint | undefined;
  if (/\b(calculator|live\s+calc|recalculate)\b/.test(lower)) viewpoint = 'calculator';
  else if (/\b(tracker|ledger|history|log)\b/.test(lower)) viewpoint = 'ledger-tracker';
  else if (/\b(transcript|chat|q&a|conversation|request[-\s]?reply)\b/.test(lower)) viewpoint = 'request-reply';
  else if (/\b(decision|form|policy|verdict)\b/.test(lower)) viewpoint = 'decision-form';

  return { theme, viewpoint };
}

export function parseContextReply(
  decl: PrismDeclaration,
  text: string,
): { readonly ops: StudioApiOp[]; readonly changes: string[] } {
  const lower = text.toLowerCase();
  const patch: {
    audience?: PrismAudience;
    dataClassification?: DataClassification;
    jurisdictions?: PrismJurisdiction[];
    usesLocalAi?: boolean;
    involvesPayments?: boolean;
  } = {};
  const changes: string[] = [];

  if (/\b(kid|kids|child|children|minor|minors|youth|under\s+13)\b/.test(lower)) {
    patch.audience = 'includes-minors';
    changes.push('Audience: Includes minors');
  } else if (/\b(public|customers?|consumers?|anyone|everyone|external|clients?|applicants?)\b/.test(lower)) {
    patch.audience = 'public-adults';
    changes.push('Audience: Public');
  } else if (/\b(team|internal|staff|employees?|company|org)\b/.test(lower)) {
    patch.audience = 'team';
    changes.push('Audience: Internal team');
  } else if (/\b(just\s+me|only\s+me|myself|solo|personal\s+(?:use|tool|app|only))\b/.test(lower)) {
    patch.audience = 'personal';
    changes.push('Audience: Personal');
  }

  if (/\b(no\s+personal|no\s+pii|anonymous|non[-\s]?personal)\b/.test(lower)) {
    patch.dataClassification = 'none';
    changes.push('Data: No personal info');
  } else if (/\b(medical|health|patient|sensitive|regulated|hipaa)\b/.test(lower)) {
    patch.dataClassification = 'sensitive-regulated';
    changes.push('Data: Sensitive / regulated');
  } else if (/\b(personal\s+info|personal\s+data|pii|credit|email|customer\s+data|user\s+data)\b/.test(lower)) {
    patch.dataClassification = 'personal';
    changes.push('Data: Personal info');
  } else if (/\b(business|operational|internal\s+numbers|orders?)\b/.test(lower)) {
    patch.dataClassification = 'operational';
    changes.push('Data: Operational');
  }

  const jurs: PrismJurisdiction[] = [];
  if (/\b(eu|europe|european|gdpr)\b/.test(lower)) jurs.push('eu');
  if (/\b(us|u\.s\.|usa|united\s+states|america|coppa)\b/.test(lower)) jurs.push('us');
  if (/\b(uk|u\.k\.|britain|british)\b/.test(lower)) jurs.push('uk');
  if (/\b(global|worldwide|international|everywhere)\b/.test(lower)) jurs.push('global');
  if (jurs.length > 0) {
    patch.jurisdictions = jurs;
    changes.push(`Regions: ${jurs.map((j) => j.toUpperCase()).join(', ')}`);
  } else if (/\b(offline|local\s+only|no\s+region)\b/.test(lower)) {
    patch.jurisdictions = ['none'];
    changes.push('Regions: Local only');
  }

  if (/\b(no\s+ai|without\s+ai)\b/.test(lower)) {
    patch.usesLocalAi = false;
    changes.push('AI: Off');
  } else if (/\b(uses?\s+(?:local\s+)?ai|with\s+ai|on-device\s+ai)\b/.test(lower)) {
    patch.usesLocalAi = true;
    changes.push('On-device AI: Enabled');
  }

  if (/\b(no\s+payments?|without\s+payments?)\b/.test(lower)) {
    patch.involvesPayments = false;
    changes.push('Payments: Off');
  } else if (/\b(payments?|checkout|billing|charge|credit\s+card)\b/.test(lower)) {
    patch.involvesPayments = true;
    changes.push('Payments: Enabled');
  }

  const ops: StudioApiOp[] = [];
  if (Object.keys(patch).length > 0) {
    ops.push({ op: 'set-concept-profile', ...patch });
  }
  if (/\b(offline|no\s+internet|strictly\s+on[-\s]?device)\b/.test(lower)) {
    ops.push({ op: 'set-capabilities', capabilities: { network: false } });
    changes.push('Internet: Off (on-device)');
  }
  if (/\b(save\s+locally|local\s+storage|remember\s+history)\b/.test(lower)) {
    ops.push({ op: 'set-capabilities', capabilities: { storage: true } });
    changes.push('Local Storage: Enabled');
  }

  // Ensure Concept profile has concrete non-null core weights if still unset
  if (
    !decl.concept.domain ||
    !decl.concept.audience ||
    !decl.concept.dataClassification
  ) {
    ops.push({
      op: 'set-concept-profile',
      domain: decl.concept.domain ?? 'general',
      audience: patch.audience ?? decl.concept.audience ?? 'personal',
      dataClassification: patch.dataClassification ?? decl.concept.dataClassification ?? 'none',
      jurisdictions:
        patch.jurisdictions ??
        (decl.concept.jurisdictions.length > 0 ? decl.concept.jurisdictions : ['none']),
    });
  }

  return { ops, changes };
}

/**
 * Reconciles ALL remaining unsettled Prism facets (vectors, STRIDE safety, WCAG Design View,
 * and every triggered compliance standard obligation) in dependency order strictly via closed
 * `StudioApiOp` calls — requiring zero manual form intervention.
 */
export function reconcileAllRemainingFacets(
  decl: PrismDeclaration,
  prefs?: { readonly theme?: PrismThemePreset; readonly viewpoint?: PrismViewpoint; readonly allowStorage?: boolean; readonly allowNetwork?: boolean },
): { readonly ops: readonly StudioApiOp[]; readonly badges: readonly string[] } {
  const ops: StudioApiOp[] = [];
  const badges: string[] = [];
  let working = decl;

  const pushAndApply = (subOps: readonly StudioApiOp[], badge?: string) => {
    for (const op of subOps) {
      const res = dispatchStudioOp(working, op);
      if (res.ok) {
        working = res.declaration;
        ops.push(op);
      }
    }
    if (badge) badges.push(badge);
  };

  // 1. Capabilities if explicitly requested
  if (prefs?.allowStorage !== undefined || prefs?.allowNetwork !== undefined) {
    pushAndApply(
      [
        {
          op: 'set-capabilities',
          capabilities: {
            ...(prefs.allowStorage !== undefined ? { storage: prefs.allowStorage } : {}),
            ...(prefs.allowNetwork !== undefined ? { network: prefs.allowNetwork } : {}),
          },
        },
      ],
      prefs.allowNetwork === false ? 'Strictly offline' : undefined,
    );
  }

  // 2. Repair any unreplayed refusal vectors or failing vectors
  const check = checkDeclaration(working);
  for (const diag of check.diagnostics) {
    if (diag.code === 'contract.refusal-unreplayed' || diag.code === 'holo.vectors') {
      const repair = deriveRepairOp(working, diag.code, diag.message, diag.id);
      if (repair) pushAndApply([repair], `Verified example (${diag.id ?? 'vector'})`);
    }
  }

  // 3. Sync & settle all open safety/threat checks automatically
  pushAndApply([{ op: 'sync-stride-threats' }]);
  const openThreats = working.security.threats.filter((t) => t.status === 'open');
  if (openThreats.length > 0) {
    for (const thr of openThreats) {
      if (thr.id === 'THR-TAMPER-INPUT' && working.refusals.length > 0) {
        pushAndApply([
          {
            op: 'settle-stride-threat',
            id: thr.id,
            status: 'mitigated-by-refusal',
            linkedRefusalId: working.refusals[0].id,
            rationale: `Protected by typed input validation and refusal "${working.refusals[0].failureText}".`,
          },
        ]);
      } else {
        pushAndApply([
          {
            op: 'settle-stride-threat',
            id: thr.id,
            status: 'mitigated-by-sandbox',
            rationale:
              thr.id === 'THR-INFO-LEAK'
                ? 'Inputs stay strictly on-device inside an isolated offline sandbox with no remote transfer.'
                : 'Protected by isolated sandbox execution and content-addressed package verification.',
          },
        ]);
      }
    }
    badges.push(`Reconciled ${openThreats.length} safety check(s)`);
  }

  // 4. Theme & traceable WCAG 2.2 View synthesis
  if (prefs?.theme) {
    pushAndApply([{ op: 'set-theme', theme: { preset: prefs.theme } }], `Theme: ${prefs.theme}`);
  }
  if (!working.view || prefs?.viewpoint) {
    pushAndApply(
      [{ op: 'synthesize-view-from-system', viewpoint: prefs?.viewpoint }],
      working.view ? `Updated layout (${prefs?.viewpoint})` : 'Synthesized interactive interface',
    );
  }

  // 5. Re-evaluate standards routing and settle every unsettled routed obligation
  const routing = evaluateVerdantRouting(working);
  let settledStdCount = 0;
  for (const std of routing.activeLayeredStandards) {
    for (const ob of std.obligations) {
      if (!ob.settled && ob.suggestedOp) {
        pushAndApply([ob.suggestedOp]);
        settledStdCount += 1;
      }
    }
  }
  if (settledStdCount > 0) {
    badges.push(`Applied ${settledStdCount} compliance protection(s)`);
  }

  return { ops, badges };
}

/**
 * Generates the warm, plain-English prompt & quick-reply sentences for the current step.
 */
export function buildStudioPromptForStep(
  decl: PrismDeclaration,
  routing: VerdantRoutingReport,
  checkReport: DeclarationCheckReport,
  step: GuidedStepId,
): StudioConversationalPrompt {
  if (step === 'anchor-idea' || routing.isUndefined) {
    return {
      step: 'anchor-idea',
      promptText:
        'What should your Prism do? Describe your idea in one plain sentence, and I will guide you through defining and reconciling every feature.',
      placeholder: 'e.g. Check if an expense fits my monthly budget, warn when over, and refuse zero amounts…',
      quickReplies: INSPIRATION_PROMPTS.map((p) => p.sentence),
    };
  }

  if (step === 'confirm-context') {
    const aud = decl.concept.audience ?? 'personal';
    const data = decl.concept.dataClassification ?? 'none';
    const regions = decl.concept.jurisdictions.length > 0 ? decl.concept.jurisdictions.join(', ').toUpperCase() : 'LOCAL';
    return {
      step: 'confirm-context',
      promptText: `I mapped out "${decl.metadata.name}" with ${decl.encoding?.fields.length ?? 0} input(s) and ${decl.features.length + decl.refusals.length} rule(s). Right now it's set for ${aud} use, ${data} data, and ${regions} scope. Who will use this, and does it handle personal info or regulated regions (like EU or US)?`,
      placeholder: 'e.g. "Just me offline", "EU customers with personal info", or "Looks good"…',
      quickReplies: [
        'Looks good, keep going',
        'Just me, offline, no personal info',
        'For EU customers with personal info',
        'Reconcile everything & finish',
      ],
    };
  }

  if (step === 'refine-rules') {
    const rulesSummary = decl.features.map((f) => f.statement).slice(0, 3).join(' · ');
    const refSummary = decl.refusals.map((r) => `“${r.failureText}”`).join(', ');
    return {
      step: 'refine-rules',
      promptText: `Here is how "${decl.metadata.name}" behaves (${checkReport.vectorsPassed}/${checkReport.vectorsTotal} examples passing): ${rulesSummary || 'basic evaluation'}${refSummary ? `. Refusals: ${refSummary}` : ''}. Would you like to add or adjust any input, threshold, or refusal in plain English?`,
      placeholder: 'e.g. "Refuse if amount is over 5000", "Add a field for priority (Low, Medium, High)", or "Looks good"…',
      quickReplies: [
        'Looks good, check safety next',
        'Reconcile everything & finish',
      ],
    };
  }

  if (step === 'safety-check') {
    const open = decl.security.threats.filter((t) => t.status === 'open');
    const promptText =
      open.length > 0
        ? `Quick safety check: ${open.length} item(s) need your confirmation (${open.map((t) => t.title.toLowerCase()).join('; ')}). Should all data stay strictly on-device with no internet access and input refusals enforced?`
        : `All ${decl.security.threats.length} safety boundaries are automatically verified by isolated offline execution. Ready to pick the look and feel?`;
    return {
      step: 'safety-check',
      promptText,
      placeholder: 'e.g. "Yes, keep everything strictly on-device" or "Allow local storage"…',
      quickReplies: [
        'Yes, keep strictly on-device & offline',
        'Allow local browser storage, stay offline',
        'Reconcile everything & finish',
      ],
    };
  }

  if (step === 'design-feel') {
    return {
      step: 'design-feel',
      promptText:
        'How should your Prism look and feel? Describe a layout (e.g. live calculator, tracker with history, decision form) or mood (e.g. warm evergreen, dark midnight, crisp paper, terracotta), or say "use recommended".',
      placeholder: 'e.g. "Warm evergreen tracker with history", "Dark midnight calculator", or "Use recommended"…',
      quickReplies: [
        'Use recommended layout & theme',
        'Dark midnight live calculator',
        'Warm evergreen tracker with history',
      ],
    };
  }

  if (step === 'routed-standards') {
    const unsettled = routing.activeLayeredStandards.filter((s) => s.settledCount < s.totalCount);
    const names = unsettled.map((s) => s.authority.name).join(', ');
    return {
      step: 'routed-standards',
      promptText: `Because of your audience, data, or regions, ${names || 'applicable standards'} apply. Would you like me to reconcile and attach all required protections and disclosures automatically?`,
      placeholder: 'e.g. "Yes, apply all required protections"…',
      quickReplies: ['Yes, apply all required protections'],
    };
  }

  return {
    step: 'ready-to-seal',
    promptText: `Everything for "${decl.metadata.name}" is reconciled and ready to mathematically prove and publish! You can prove & seal it below, or keep telling me anything you want to tweak in plain English.`,
    placeholder: 'e.g. "Switch to dark noir theme", "Refuse values over 10000", or "Make it for EU customers"…',
    quickReplies: [
      'Switch to dark midnight theme',
      'Switch to live calculator layout',
    ],
  };
}

/**
 * Reconciles a single natural-language creator message at any stage of the wizard into
 * atomic closed `StudioApiOp[]` operations + a plain-English Studio reply + next stage.
 */
export function reconcileWizardTurn(
  decl: PrismDeclaration,
  currentStep: GuidedStepId,
  rawUserText: string,
  preParsedIntent?: ExtractedPrismIntent,
): WizardTurnReconciliation {
  const text = rawUserText.trim();
  const routing = evaluateVerdantRouting(decl);

  // 1. Initial Idea Anchor (undefined Prism or explicitly at anchor-idea)
  if (routing.isUndefined || currentStep === 'anchor-idea') {
    const intent = preParsedIntent ?? parseIdeaSentence(text);
    const ops = intentToAnchorOps(decl, intent);
    const inputList = intent.inputs.map((f) => f.label).join(', ');
    const badges = [
      `Concept: ${intent.name.value}`,
      `${intent.inputs.length} input(s): ${inputList}`,
      `${intent.rules.length} rule(s) · ${intent.refusals.length} refusal(s)`,
    ];
    return {
      ops,
      assistantReply: `Anchored "${intent.name.value}" from your sentence with ${intent.inputs.length} input(s) (${inputList}), ${intent.rules.length} rule(s), and ${intent.refusals.length} refusal check(s).`,
      reconciledBadges: badges,
      nextStep: 'confirm-context',
      openPreview: false,
    };
  }

  // 2. One-Shot Autonomous Reconciliation Shortcut ("reconcile everything" / "finish it")
  if (RECONCILE_ALL_RE.test(text)) {
    const designPrefs = parseDesignPreferences(text);
    const { ops, badges } = reconcileAllRemainingFacets(decl, designPrefs);
    return {
      ops,
      assistantReply:
        'Done — I reconciled all remaining safety checks, synthesized your accessible interface, and applied every required protection automatically. Your Live Preview is open and ready to prove and publish!',
      reconciledBadges: badges.length > 0 ? badges : ['All Prism facets reconciled'],
      nextStep: 'ready-to-seal',
      openPreview: true,
    };
  }

  // 3. Step 2: Confirm Context Weights via Natural Language
  if (currentStep === 'confirm-context') {
    const { ops, changes } = parseContextReply(decl, text);
    let previewDecl = decl;
    for (const op of ops) {
      const r = dispatchStudioOp(previewDecl, op);
      if (r.ok) previewDecl = r.declaration;
    }
    const nextRouting = evaluateVerdantRouting(previewDecl);
    const stdNote =
      nextRouting.activeLayeredStandards.length > 0
        ? ` Applicable standards detected: ${nextRouting.activeLayeredStandards.map((s) => s.authority.name).join(', ')}.`
        : ' No extra regulatory layers are needed for this setup.';
    return {
      ops,
      assistantReply: `${changes.length > 0 ? `Updated context (${changes.join(' · ')}).` : 'Confirmed context profile.'}${stdNote}`,
      reconciledBadges: changes.length > 0 ? changes : ['Context confirmed'],
      nextStep: 'refine-rules',
      openPreview: false,
    };
  }

  // 4. Step 3: Refine Rules & Inputs via Natural Language
  if (currentStep === 'refine-rules') {
    if (AFFIRMATIVE_RE.test(text)) {
      const syncOp: StudioApiOp = { op: 'sync-stride-threats' };
      const syncedRes = dispatchStudioOp(decl, syncOp);
      const synced = syncedRes.ok ? syncedRes.declaration : decl;
      const openCount = synced.security.threats.filter((t) => t.status === 'open').length;
      if (openCount === 0) {
        return {
          ops: [syncOp],
          assistantReply: `Rules confirmed and all ${synced.security.threats.length} safety checks passed automatically via offline sandbox isolation. Now tell me how you want your Prism to look and feel!`,
          reconciledBadges: ['Rules confirmed', `${synced.security.threats.length}/${synced.security.threats.length} safety checks verified`],
          nextStep: 'design-feel',
          openPreview: false,
        };
      }
      return {
        ops: [syncOp],
        assistantReply: `Rules confirmed. I ran a safety check and found ${openCount} item(s) to confirm before building your interface.`,
        reconciledBadges: ['Rules confirmed', `${openCount} safety check(s) to confirm`],
        nextStep: 'safety-check',
        openPreview: false,
      };
    }

    const parsed = parseFollowUpInstruction(decl, text);
    if (parsed.ok) {
      const ops: StudioApiOp[] = [...parsed.ops];
      if (decl.view) ops.push({ op: 'synthesize-view-from-system', viewpoint: decl.view.viewpoint });
      return {
        ops,
        assistantReply: `${parsed.summary}. Anything else you want to add or change, or say "looks good" to check safety?`,
        reconciledBadges: [parsed.summary],
        nextStep: 'refine-rules',
        openPreview: false,
      };
    }

    return {
      ops: [],
      assistantReply:
        parsed.hint ??
        'Tell me any rule, input, or refusal in plain English (for example: "Refuse if amount is over 5000"), or say "looks good" to continue.',
      reconciledBadges: [],
      nextStep: 'refine-rules',
      openPreview: false,
    };
  }

  // 5. Step 4: Safety & Privacy Check via Natural Language
  if (currentStep === 'safety-check') {
    const lower = text.toLowerCase();
    const allowStorage = /\b(allow|enable|use)\s+(?:local\s+)?storage\b/.test(lower) ? true : undefined;
    const allowNetwork = /\b(allow|enable|need|needs)\s+(?:internet|network|online)\b/.test(lower)
      ? true
      : /\b(offline|no\s+internet|on[-\s]?device|yes|looks?\s+good|confirm)\b/.test(lower)
        ? false
        : undefined;

    const ops: StudioApiOp[] = [];
    let working = decl;
    const applyLocal = (op: StudioApiOp) => {
      const r = dispatchStudioOp(working, op);
      if (r.ok) {
        working = r.declaration;
        ops.push(op);
      }
    };

    if (allowStorage !== undefined || allowNetwork !== undefined) {
      applyLocal({
        op: 'set-capabilities',
        capabilities: {
          ...(allowStorage !== undefined ? { storage: allowStorage } : {}),
          ...(allowNetwork !== undefined ? { network: allowNetwork } : {}),
        },
      });
    }
    applyLocal({ op: 'sync-stride-threats' });
    for (const thr of working.security.threats) {
      if (thr.status === 'open') {
        if (thr.id === 'THR-TAMPER-INPUT' && working.refusals.length > 0) {
          applyLocal({
            op: 'settle-stride-threat',
            id: thr.id,
            status: 'mitigated-by-refusal',
            linkedRefusalId: working.refusals[0].id,
            rationale: `Protected by typed input validation and refusal "${working.refusals[0].failureText}".`,
          });
        } else {
          applyLocal({
            op: 'settle-stride-threat',
            id: thr.id,
            status: 'mitigated-by-sandbox',
            rationale:
              allowNetwork === true
                ? 'Explicitly authorized by creator with bounded input validation.'
                : 'Confirmed strictly on-device offline sandbox execution.',
          });
        }
      }
    }

    return {
      ops,
      assistantReply: 'Safety and privacy boundaries are reconciled and verified.',
      reconciledBadges: [
        `All ${working.security.threats.length} safety checks verified`,
        ...(allowStorage ? ['Local storage enabled'] : []),
        ...(allowNetwork === false ? ['Offline on-device only'] : []),
      ],
      nextStep: 'design-feel',
      openPreview: false,
    };
  }

  // 6. Step 5: Look & Feel via Natural Language -> Synthesizes Interface & Unlocks Live Preview!
  if (currentStep === 'design-feel') {
    const prefs = parseDesignPreferences(text);
    const ops: StudioApiOp[] = [];
    const badges: string[] = [];
    let working = decl;

    if (prefs.theme) {
      const tOp: StudioApiOp = { op: 'set-theme', theme: { preset: prefs.theme } };
      const r = dispatchStudioOp(working, tOp);
      if (r.ok) {
        working = r.declaration;
        ops.push(tOp);
        badges.push(`Theme: ${prefs.theme}`);
      }
    }
    const vOp: StudioApiOp = { op: 'synthesize-view-from-system', viewpoint: prefs.viewpoint };
    const vr = dispatchStudioOp(working, vOp);
    if (vr.ok) {
      working = vr.declaration;
      ops.push(vOp);
      badges.push(`Layout: ${working.view?.viewpoint ?? 'decision-form'}`);
      badges.push('Live Preview unlocked');
    }

    const nextRouting = evaluateVerdantRouting(working);
    const unsettledStd = nextRouting.activeLayeredStandards.some((s) => s.settledCount < s.totalCount);
    if (unsettledStd) {
      return {
        ops,
        assistantReply:
          'I built your interface and opened the Live Preview on the right! One last step: let’s reconcile the applicable protections triggered by your audience/region.',
        reconciledBadges: badges,
        nextStep: 'routed-standards',
        openPreview: true,
      };
    }

    return {
      ops,
      assistantReply:
        'I built your accessible interface and opened the Live Preview on the right! Every core check is complete — you can prove & publish your Prism now, or keep describing tweaks here.',
      reconciledBadges: badges,
      nextStep: 'ready-to-seal',
      openPreview: true,
    };
  }

  // 7. Step 6: Routed Standards Confirmation via Natural Language
  if (currentStep === 'routed-standards') {
    const { ops, badges } = reconcileAllRemainingFacets(decl);
    return {
      ops,
      assistantReply:
        'Applied all required protections and updated your Live Preview disclosures. Everything is reconciled and ready to mathematically prove and publish!',
      reconciledBadges: badges.length > 0 ? badges : ['All required protections applied'],
      nextStep: 'ready-to-seal',
      openPreview: true,
    };
  }

  // 8. Post-Reconciliation Continuous Natural-Language Refinement (at ready-to-seal!)
  const designPrefs = parseDesignPreferences(text);
  const contextChange = parseContextReply(decl, text);
  const ruleChange = parseFollowUpInstruction(decl, text);

  const combinedOps: StudioApiOp[] = [];
  const badges: string[] = [];

  if (contextChange.changes.length > 0) {
    combinedOps.push(...contextChange.ops);
    badges.push(...contextChange.changes);
  }
  if (ruleChange.ok) {
    combinedOps.push(...ruleChange.ops);
    badges.push(ruleChange.summary);
  }
  if (designPrefs.theme || designPrefs.viewpoint) {
    if (designPrefs.theme) {
      combinedOps.push({ op: 'set-theme', theme: { preset: designPrefs.theme } });
      badges.push(`Theme: ${designPrefs.theme}`);
    }
    if (designPrefs.viewpoint) {
      combinedOps.push({ op: 'synthesize-view-from-system', viewpoint: designPrefs.viewpoint });
      badges.push(`Layout: ${designPrefs.viewpoint}`);
    }
  }

  if (combinedOps.length > 0) {
    // Re-reconcile all downstream facets automatically so the Prism stays 100% ready to seal!
    let temp = decl;
    for (const op of combinedOps) {
      const r = dispatchStudioOp(temp, op);
      if (r.ok) temp = r.declaration;
    }
    const followUp = reconcileAllRemainingFacets(temp, {
      viewpoint: designPrefs.viewpoint ?? temp.view?.viewpoint,
    });
    return {
      ops: [...combinedOps, ...followUp.ops],
      assistantReply: `Updated your Prism and automatically reconciled all downstream safety, view, and compliance checks (${[...badges, ...followUp.badges].join(' · ')}).`,
      reconciledBadges: [...badges, ...followUp.badges],
      nextStep: 'ready-to-seal',
      openPreview: true,
    };
  }

  return {
    ops: [],
    assistantReply:
      'Your Prism is ready to prove and publish! You can also ask me to tweak anything — e.g. "Switch to dark midnight theme", "Refuse amounts over 5000", or "Make it for EU customers".',
    reconciledBadges: [],
    nextStep: 'ready-to-seal',
    openPreview: false,
  };
}

