/**
 * Mechanical Prism Compiler (`src/engine/prismCompiler.ts`)
 *
 * Pure deterministic compiler `compilePrismDeclaration(decl)` → canonical `uor.foundation`
 * Prism workspace files (`app/src/App.lex.tex`, `app/view/index.html`, `app/view/app.js`,
 * `app/design/theme.json`, `prism.toml`, `model/ids.toml`, `features/suites/app.feature`,
 * `CONFORMANCE.md`, `model/requirements.json`, `model/admissions.json`, `model/owner-decisions.json`).
 *
 * Zero hand-authored code, zero LLM code generation:
 * - Every rule and refusal compiles into pure total `Nat`-domain LexLean functions + Lean-kernel
 *   provable `linear_arithmetic` theorems (`breaks*` pattern per `lexlean-1.2.md`).
 * - Every UI control compiles into `prism.design/1` semantic HTML (`data-slot`) and thin
 *   `globalThis.prism.invoke(line)` client JS with zero business logic (`R1`).
 * - Runs `prism-code`'s zero-Node `compileDesign` + `checkDesign` in-process (<10ms) for
 *   instant live preview and zero design violations during `prism-code build`.
 */

import { compileDesign } from '@prism-code-internal/design/compile.mjs';
import { checkDesign, type DesignDiagnostic } from '@prism-code-internal/design/gate.mjs';
import { resolveTheme } from '@prism-code-internal/design/system.mjs';
import {
  type ArithmeticExpr,
  type EncodedField,
  type FieldPredicate,
  type PrismDeclaration,
  type RuleCondition,
  type StudioFeature,
  type StudioRefusal,
  sanitizeIdentifier,
} from './verdantStudioApi';
import { evaluateVerdantRouting, isPrismSynthesizable } from './verdantStandardsRouter';
import { synthesizeProceduralOrganView } from './prismMediaOrgans';
import { sha256HexSync } from './prismStudio';

const enc = new TextEncoder();

export interface CompiledPrismWorkspace {
  /** All generated workspace files (paths relative to project root, e.g. `app/src/App.lex.tex`). */
  readonly files: Readonly<Record<string, string>>;
  /** Files relative to the LexLean model root (`app/`) ready to merge with canonical template foundation. */
  readonly appFiles: Readonly<Record<string, string>>;
  /** Compiled `prism.design/1` stylesheet (`prism.css`) for instant live preview. */
  readonly compiledCss: string;
  /** Compiled `prism.design/1` runtime script (`prism-ui.js`) for instant live preview. */
  readonly compiledUiJs: string;
  /** Component slots compiled into the design sheet. */
  readonly designSlots: readonly string[];
  /** `prism.design/1` gate diagnostics (0 errors on valid mechanical output). */
  readonly designDiagnostics: readonly DesignDiagnostic[];
  /** Supported theme modes from `resolveTheme` (e.g. `['dark', 'immersive']` for `noir` / `midnight`). */
  readonly themeModes: readonly string[];
  /** Theorem names generated in `App.lex.tex` mapped by feature/refusal ID. */
  readonly theoremMap: Readonly<Record<string, string>>;
}

function escapeLexString(text: string): string {
  return text.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function pascalModuleName(slug: string): string {
  const parts = slug
    .split(/[^a-zA-Z0-9]+/)
    .filter(Boolean)
    .map((p) => p[0].toUpperCase() + p.slice(1).toLowerCase());
  return `Prism${parts.join('') || 'App'}`;
}

function theoremIdent(id: string): string {
  return `theorem_${id.toLowerCase().replace(/[^a-z0-9]+/g, '_')}`;
}

function breaksIdent(id: string): string {
  return `breaks_${id.toLowerCase().replace(/[^a-z0-9]+/g, '_')}`;
}

function formulaIdent(id: string): string {
  return `compute_${id.toLowerCase().replace(/[^a-z0-9]+/g, '_')}`;
}

function compilePredicateLex(pred: FieldPredicate): string {
  const fieldExpr = `(project input ${sanitizeIdentifier(pred.field)})`;
  const rhsExpr = pred.rhsField
    ? `(project input ${sanitizeIdentifier(pred.rhsField)})`
    : String(Math.max(0, Math.floor(pred.value)));
  switch (pred.op) {
    case 'eq': return `(${fieldExpr} =? ${rhsExpr})`;
    case 'neq': return `(not (${fieldExpr} =? ${rhsExpr}))`;
    case 'lt': return `(${fieldExpr} <? ${rhsExpr})`;
    case 'lte': return `(${fieldExpr} <=? ${rhsExpr})`;
    case 'gt': return `(${rhsExpr} <? ${fieldExpr})`;
    case 'gte': return `(${rhsExpr} <=? ${fieldExpr})`;
  }
}

function compileConditionLex(cond: RuleCondition): string {
  if (cond.all.length === 0) return 'true';
  let expr = compilePredicateLex(cond.all[0]);
  for (let i = 1; i < cond.all.length; i++) {
    expr = `(${expr} && ${compilePredicateLex(cond.all[i])})`;
  }
  return expr;
}

function conjoinLex(parts: readonly string[]): string {
  const filtered = parts.filter((p) => p !== 'true');
  if (filtered.length === 0) return 'true';
  let expr = filtered[0];
  for (let i = 1; i < filtered.length; i++) {
    expr = `(${expr} && ${filtered[i]})`;
  }
  return expr;
}

function compileArithLex(expr: ArithmeticExpr): string {
  switch (expr.op) {
    case 'const':
      return String(Math.max(0, Math.floor(expr.value)));
    case 'field':
      return `(project input ${sanitizeIdentifier(expr.field)})`;
    case 'add':
      return `(${compileArithLex(expr.left)} + ${compileArithLex(expr.right)})`;
    case 'sub':
      return `(prim subtract ${compileArithLex(expr.left)} ${compileArithLex(expr.right)} : Nat)`;
    case 'mul':
      return `(prim multiply ${compileArithLex(expr.left)} ${compileArithLex(expr.right)} : Nat)`;
    case 'div':
      return `(prim quotient ${compileArithLex(expr.left)} ${compileArithLex(expr.right)} 0 : Nat)`;
    case 'mod':
      return `(prim remainder ${compileArithLex(expr.left)} ${compileArithLex(expr.right)} 0 : Nat)`;
    case 'pct':
      return `(prim quotient (prim multiply ${compileArithLex(expr.left)} ${compileArithLex(expr.right)} : Nat) 100 0 : Nat)`;
  }
}

/**
 * Compiles the LexLean 1.2 model (`app/src/App.lex.tex`) from a `PrismDeclaration`.
 * Every refusal & feature compiles into pure `Nat` rules + Lean-kernel-provable theorems.
 */
export function emitLexLeanApp(decl: PrismDeclaration): {
  readonly source: string;
  readonly theoremMap: Record<string, string>;
} {
  if (!decl.encoding || decl.encoding.fields.length === 0) {
    throw new Error('Cannot emit LexLean model: no System input encoding is declared yet.');
  }
  const fields: readonly EncodedField[] = decl.encoding.fields;
  const orderedFeatures: readonly StudioFeature[] = [...decl.features].sort(
    (a, b) => a.priority - b.priority || a.id.localeCompare(b.id),
  );
  const refusals: readonly StudioRefusal[] = decl.refusals;
  const choiceFields = fields.filter((f) => f.kind === 'choice' && f.choices && f.choices.length > 0);
  const modPrefix = pascalModuleName(decl.metadata.slug);
  const crateName = `prism-${decl.metadata.slug}`;

  const theoremMap: Record<string, string> = {};
  const defs: string[] = [];

  // 1. Input structure
  const fieldDecls = fields.map((f) => `{name: ${sanitizeIdentifier(f.name)}, type: Nat}`).join(', ');
  defs.push(`structure{
  name: Input,
  type_parameters: [],
  parameters: [],
  fields: [${fieldDecls}]
}`);

  // 2. Outcome code constants
  // 0 = fallback, 1..R = refusals, R+1..R+C = choice out-of-range, 100+i = features
  const choiceBase = refusals.length + 1;
  const featureBase = 100;

  // 3. decideOutcome(input : Input) : Nat
  // Build nested if-then-else bottom-up
  let decideBody = '0';
  for (let i = orderedFeatures.length - 1; i >= 0; i--) {
    const feat = orderedFeatures[i];
    const code = String(featureBase + i);
    if (feat.condition.all.length === 0) {
      decideBody = code;
    } else {
      decideBody = `(if ${compileConditionLex(feat.condition)} then ${code} else ${decideBody})`;
    }
  }
  for (let i = choiceFields.length - 1; i >= 0; i--) {
    const cf = choiceFields[i];
    const code = String(choiceBase + i);
    const limit = cf.choices!.length;
    decideBody = `(if (${limit} <=? (project input ${sanitizeIdentifier(cf.name)})) then ${code} else ${decideBody})`;
  }
  for (let i = refusals.length - 1; i >= 0; i--) {
    const ref = refusals[i];
    const code = String(i + 1);
    decideBody = `(if ${compileConditionLex(ref.condition)} then ${code} else ${decideBody})`;
  }

  defs.push(`definition{
  name: decideOutcome,
  parameters: [input : Input],
  result: Nat,
  body: ${decideBody}
}`);

  // 4. Pure formula computation functions (for formula features)
  for (const feat of orderedFeatures) {
    if (feat.outcome.kind === 'formula') {
      defs.push(`definition{
  name: ${formulaIdent(feat.id)},
  parameters: [input : Input],
  result: Nat,
  body: ${compileArithLex(feat.outcome.expr)}
}`);
    }
  }

  // 5. Lean-kernel-provable theorems (one per refusal & build feature)
  const priorNegations: string[] = [];

  for (let i = 0; i < refusals.length; i++) {
    const ref = refusals[i];
    const code = i + 1;
    const condExpr = compileConditionLex(ref.condition);
    const guard = conjoinLex([...priorNegations, condExpr]);
    const bName = breaksIdent(ref.id);
    const tName = theoremIdent(ref.id);
    theoremMap[ref.id] = `App.${tName}`;
    const breaksBody = guard === 'true'
      ? `(not ((decideOutcome input) =? ${code}))`
      : `(${guard} && (not ((decideOutcome input) =? ${code})))`;
    const unfold = [bName, 'decideOutcome'].sort().join(', ');

    defs.push(`definition{
  name: ${bName},
  parameters: [input : Input],
  result: Bool,
  body: ${breaksBody}
}`);
    defs.push(`theorem{
  name: ${tName},
  parameters: [input : Input],
  statement: ((${bName} input) = false),
  proof: linear_arithmetic{
    definitions: [${unfold}]
  },
  axioms: [Quot.sound, propext]
}`);
    if (condExpr !== 'true') priorNegations.push(`(not ${condExpr})`);
  }

  for (const cf of choiceFields) {
    const limit = cf.choices!.length;
    priorNegations.push(`(not (${limit} <=? (project input ${sanitizeIdentifier(cf.name)})))`);
  }

  for (let i = 0; i < orderedFeatures.length; i++) {
    const feat = orderedFeatures[i];
    const code = featureBase + i;
    const condExpr = compileConditionLex(feat.condition);
    if (feat.level === 'build') {
      const guard = conjoinLex([...priorNegations, condExpr]);
      const bName = breaksIdent(feat.id);
      const tName = theoremIdent(feat.id);
      theoremMap[feat.id] = `App.${tName}`;
      const breaksBody = guard === 'true'
        ? `(not ((decideOutcome input) =? ${code}))`
        : `(${guard} && (not ((decideOutcome input) =? ${code})))`;
      const unfold = [bName, 'decideOutcome'].sort().join(', ');

      defs.push(`definition{
  name: ${bName},
  parameters: [input : Input],
  result: Bool,
  body: ${breaksBody}
}`);
      defs.push(`theorem{
  name: ${tName},
  parameters: [input : Input],
  statement: ((${bName} input) = false),
  proof: linear_arithmetic{
    definitions: [${unfold}]
  },
  axioms: [Quot.sound, propext]
}`);
    }
    if (condExpr !== 'true') priorNegations.push(`(not ${condExpr})`);
  }

  // 6. Pure-LexLean Bytes parser & Nat-to-ASCII-decimal formatter
  defs.push(`definition{
  name: digitGlyphs,
  parameters: [],
  result: Bytes,
  body: (prim utf8_encode "0123456789" : Bytes)
}`);

  defs.push(`definition{
  name: digitByte,
  parameters: [d : Nat],
  result: Bytes,
  body: (match (prim slice (digitGlyphs) d 1 : (Option Bytes))
    | Option.none => (prim utf8_encode "0" : Bytes)
    | Option.some b => b)
}`);

  defs.push(`definition{
  name: natToBytesLoop,
  parameters: [n : Nat, fuel : Nat],
  result: Bytes,
  body: (match fuel
    | Nat.zero => (prim utf8_encode "" : Bytes)
    | Nat.succ rest => (if (n =? 0) then (prim utf8_encode "" : Bytes) else (prim append (natToBytesLoop (prim quotient n 10 0 : Nat) rest) (digitByte (prim remainder n 10 0 : Nat)) : Bytes))),
  recursive_argument: fuel
}`);

  defs.push(`definition{
  name: natToBytes,
  parameters: [n : Nat],
  result: Bytes,
  body: (if (n =? 0) then (prim utf8_encode "0" : Bytes) else (natToBytesLoop n 20))
}`);

  defs.push(`definition{
  name: skipSpaces,
  parameters: [bs : Bytes, pos : Nat, fuel : Nat],
  result: Nat,
  body: (match fuel
    | Nat.zero => pos
    | Nat.succ rest => (match (prim index bs pos : (Option UInt8))
      | Option.none => pos
      | Option.some b => (let c : Nat := (prim to_nat b : Nat); (if ((c =? 32) || ((c =? 9) || ((c =? 10) || (c =? 13)))) then (skipSpaces bs (pos + 1) rest) else pos)))),
  recursive_argument: fuel
}`);

  defs.push(`definition{
  name: tokenEnd,
  parameters: [bs : Bytes, pos : Nat, fuel : Nat],
  result: Nat,
  body: (match fuel
    | Nat.zero => pos
    | Nat.succ rest => (match (prim index bs pos : (Option UInt8))
      | Option.none => pos
      | Option.some b => (let c : Nat := (prim to_nat b : Nat); (if ((48 <=? c) && (c <=? 57)) then (tokenEnd bs (pos + 1) rest) else pos)))),
  recursive_argument: fuel
}`);

  defs.push(`definition{
  name: parseDigits,
  parameters: [bs : Bytes, pos : Nat, endPos : Nat, acc : Nat, fuel : Nat],
  result: (Option Nat),
  body: (match fuel
    | Nat.zero => (if (pos =? endPos) then (ctor Option.some [Nat] acc) else (ctor Option.none [Nat]))
    | Nat.succ rest => (if (endPos <=? pos) then (ctor Option.some [Nat] acc) else (match (prim index bs pos : (Option UInt8))
      | Option.none => (ctor Option.none [Nat])
      | Option.some b => (let c : Nat := (prim to_nat b : Nat); (if ((48 <=? c) && (c <=? 57)) then (parseDigits bs (pos + 1) endPos ((prim multiply acc 10 : Nat) + (prim subtract c 48 : Nat)) rest) else (ctor Option.none [Nat])))))),
  recursive_argument: fuel
}`);

  defs.push(`definition{
  name: parseTokenAt,
  parameters: [bs : Bytes, startPos : Nat],
  result: (Option Nat),
  body: (let endPos : Nat := (tokenEnd bs startPos 16); (if (endPos <=? startPos) then (ctor Option.none [Nat]) else (parseDigits bs startPos endPos 0 16)))
}`);

  // 7. Format output from decided outcome code
  let formatBody = `(prim utf8_encode "No matching rule." : Bytes)`;
  for (let i = orderedFeatures.length - 1; i >= 0; i--) {
    const feat = orderedFeatures[i];
    const code = featureBase + i;
    let outExpr: string;
    if (feat.outcome.kind === 'label') {
      outExpr = `(prim utf8_encode "${escapeLexString(feat.outcome.label)}" : Bytes)`;
    } else {
      const prefix = `${feat.outcome.label}: `;
      const suffix = feat.outcome.unit ? ` ${feat.outcome.unit}` : '';
      const numPart = `(natToBytes (${formulaIdent(feat.id)} input))`;
      if (suffix) {
        outExpr = `(prim append (prim append (prim utf8_encode "${escapeLexString(prefix)}" : Bytes) ${numPart} : Bytes) (prim utf8_encode "${escapeLexString(suffix)}" : Bytes) : Bytes)`;
      } else {
        outExpr = `(prim append (prim utf8_encode "${escapeLexString(prefix)}" : Bytes) ${numPart} : Bytes)`;
      }
    }
    formatBody = `(if (code =? ${code}) then ${outExpr} else ${formatBody})`;
  }
  for (let i = choiceFields.length - 1; i >= 0; i--) {
    const cf = choiceFields[i];
    const code = choiceBase + i;
    const msg = `Unknown ${cf.label.toLowerCase()}.`;
    formatBody = `(if (code =? ${code}) then (prim utf8_encode "${escapeLexString(msg)}" : Bytes) else ${formatBody})`;
  }
  for (let i = refusals.length - 1; i >= 0; i--) {
    const ref = refusals[i];
    const code = i + 1;
    formatBody = `(if (code =? ${code}) then (prim utf8_encode "${escapeLexString(ref.failureText)}" : Bytes) else ${formatBody})`;
  }

  defs.push(`definition{
  name: formatAnswer,
  parameters: [input : Input],
  result: Bytes,
  body: (let code : Nat := (decideOutcome input); ${formatBody})
}`);

  // 8. Parse N space-separated fields from request bytes and call formatAnswer
  const invalidExpr = `(prim utf8_encode "Invalid request format." : Bytes)`;
  const recordFields = fields.map((f, i) => `${sanitizeIdentifier(f.name)} := v${i}`).join(', ');
  let parseChain = `(let pEnd : Nat := (skipSpaces request e${fields.length - 1} 64); (if (pEnd =? (prim length request : Nat)) then (formatAnswer (record Input {${recordFields}})) else ${invalidExpr}))`;

  for (let i = fields.length - 1; i >= 0; i--) {
    const prevEnd = i === 0 ? '0' : `e${i - 1}`;
    const separatorGuard =
      i === 0
        ? `(let s${i} : Nat := (skipSpaces request ${prevEnd} 64); (let e${i} : Nat := (tokenEnd request s${i} 16); (match (parseTokenAt request s${i}) | Option.none => ${invalidExpr} | Option.some v${i} => ${parseChain})))`
        : `(let s${i} : Nat := (skipSpaces request ${prevEnd} 64); (if (s${i} <=? ${prevEnd}) then ${invalidExpr} else (let e${i} : Nat := (tokenEnd request s${i} 16); (match (parseTokenAt request s${i}) | Option.none => ${invalidExpr} | Option.some v${i} => ${parseChain}))))`;
    parseChain = separatorGuard;
  }

  defs.push(`definition{
  name: dispatch,
  parameters: [request : Bytes],
  result: Bytes,
  body: ${parseChain}
}`);

  // 9. Acceptance vectors list
  let vectorsExpr = `(nil Foundation.View.Text.V1.Model::AcceptanceVector)`;
  for (let i = decl.vectors.length - 1; i >= 0; i--) {
    const vec = decl.vectors[i];
    vectorsExpr = `(cons (record Foundation.View.Text.V1.Model::AcceptanceVector {request := (prim utf8_encode "${escapeLexString(vec.request)}" : Bytes), response := (prim utf8_encode "${escapeLexString(vec.expectedResponse)}" : Bytes)}) ${vectorsExpr})`;
  }

  // 10. TextApplication record
  const viewTitle = decl.view?.title ?? decl.metadata.name;
  const viewSubmit = decl.view?.submitLabel ?? 'Evaluate';
  defs.push(`definition{
  name: application,
  parameters: [],
  result: Foundation.View.Text.V1.Model::TextApplication,
  body: (record Foundation.View.Text.V1.Model::TextApplication {profile := "prismpm/text-application/1", name := "${escapeLexString(decl.metadata.slug)}", cargoName := "${escapeLexString(crateName)}", cargoVersion := "${escapeLexString(decl.metadata.version)}", cargoDescription := "${escapeLexString(decl.metadata.description)}", cargoRepository := "", cargoHomepage := "", libraryRoots := (cons "${modPrefix}.App.dispatch" (nil String)), acceptanceVectors := ${vectorsExpr}, entryRoot := "${modPrefix}.App.dispatch", coreContract := "hologram:guest/core-wasm@1", requestMaximum := (128 : UInt32), responseMaximum := (256 : UInt32), guestAllocationMaximum := (512 : UInt32), capabilitiesEmpty := true, fatArchive := true, primaryLayer := (0 : UInt8), viewLayer := (1 : UInt8), view := (record Foundation.View.Text.V1.Model::TextView {title := "${escapeLexString(viewTitle)}", heading := "${escapeLexString(viewTitle)}", inputLabel := "Request", submitLabel := "${escapeLexString(viewSubmit)}", outputLabel := "Answer", inputError := "Invalid request", responseError := "Invalid response"})})
}`);

  const source = `\\begin{lexlean}{App}
\\useglossary{lexlean.std.bool@1.2.0}
\\useglossary{lexlean.std.nat@1.2.0}
\\useglossary{prism.arch@1.0.0}
\\importmodule{Foundation.View.Text.V1.Model}
\\title{Text application model}
\\begin{semanticmodule}
${defs.join('\n\n')}
\\end{semanticmodule}
\\end{lexlean}
`;

  return { source, theoremMap };
}

/**
 * Emits `prism.toml` with deny-by-default `[capabilities]`.
 */
export function emitPrismToml(decl: PrismDeclaration): string {
  const capLines: string[] = [];
  if (decl.capabilities.storage) capLines.push('storage = true');
  if (decl.capabilities.network) capLines.push('network = true');
  return `[application]
name = "${escapeLexString(decl.metadata.slug)}"
version = "${escapeLexString(decl.metadata.version)}"
description = "${escapeLexString(decl.metadata.description)}"

[model]
source = "app/src"
watch = ["app/src/**/*.lex.tex", "app/language/**/*.toml", "app/lexlean.toml", "app/view/**", "model/**/*.toml", "features/**/*.feature"]

# Deny by default: an empty set grants the holo nothing.
[capabilities]
${capLines.join('\n')}${capLines.length ? '\n' : ''}`;
}

/**
 * Emits `model/ids.toml` (`spec = "template/2"`), `features/suites/app.feature`, and `CONFORMANCE.md`.
 */
export function emitConformanceRegisters(
  decl: PrismDeclaration,
  theoremMap: Readonly<Record<string, string>>,
): {
  readonly idsToml: string;
  readonly featureGherkin: string;
  readonly conformanceMd: string;
} {
  const idBlocks: string[] = ['spec = "template/2"'];
  const scenarios: string[] = [
    'Feature: app',
    '',
    `  Declarative model for ${decl.metadata.name}.`,
  ];
  const mdRows: string[] = [
    '# Conformance',
    '',
    '| ID | Level | Suite | Statement | Evidence |',
    '| --- | --- | --- | --- | --- |',
  ];

  for (const ref of decl.refusals) {
    const thm = theoremMap[ref.id];
    idBlocks.push(`
[[id]]
id = "${ref.id}"
level = "build"
suite = "app"
statement = "${escapeLexString(ref.statement)}"
theorems = ["${thm}"]
failure = "${escapeLexString(ref.failureText)}"
negative = "${ref.negativeId}"`);

    scenarios.push(
      '',
      `  @${ref.id} @build @rejects`,
      `  Scenario: ${ref.statement}`,
      '    Given the declared model in app/src/App.lex.tex',
      `    When prism-code check proves ${thm}`,
      `    Then invalid input is refused with ${ref.failureText}`,
    );

    mdRows.push(`| ${ref.id} | build | app | ${ref.statement} | \`${thm}\` (refusal: "${ref.failureText}") |`);
  }

  for (const feat of decl.features) {
    const thm = theoremMap[feat.id];
    if (feat.level === 'build' && thm) {
      idBlocks.push(`
[[id]]
id = "${feat.id}"
level = "build"
suite = "app"
statement = "${escapeLexString(feat.statement)}"
theorems = ["${thm}"]`);
      mdRows.push(`| ${feat.id} | build | app | ${feat.statement} | \`${thm}\` |`);
    } else {
      idBlocks.push(`
[[id]]
id = "${feat.id}"
level = "${feat.level}"
suite = "app"
statement = "${escapeLexString(feat.statement)}"`);
      mdRows.push(`| ${feat.id} | ${feat.level} | app | ${feat.statement} | measured |`);
    }

    scenarios.push(
      '',
      `  @${feat.id} @${feat.level}`,
      `  Scenario: ${feat.statement}`,
      '    Given the declared model in app/src/App.lex.tex',
      `    When the model evaluates a request for ${feat.id}`,
      '    Then the answer matches the declared rule and acceptance vectors',
    );
  }

  const routing = evaluateVerdantRouting(decl);
  mdRows.push(
    '',
    '## Verdant Prism Core Spine & Standards Applicability',
    '',
    '| Standard | Tier | Status | Trigger / Rationale |',
    '| --- | --- | --- | --- |',
  );
  for (const core of routing.coreSpine) {
    mdRows.push(
      `| \`${core.authority.id}\` (${core.authority.name}) | Core | ${core.settledCount}/${core.totalCount} | ${core.reason} |`,
    );
  }
  for (const layered of routing.activeLayeredStandards) {
    mdRows.push(
      `| \`${layered.authority.id}\` (${layered.authority.name}) | Applicable | ${layered.settledCount}/${layered.totalCount} settled | ${layered.reason} |`,
    );
  }
  for (const excl of routing.excludedStandards) {
    mdRows.push(
      `| \`${excl.authority.id}\` (${excl.authority.name}) | Not Applicable | Excluded | ${excl.exclusionRationale} |`,
    );
  }

  return {
    idsToml: `${idBlocks.join('\n')}\n`,
    featureGherkin: `${scenarios.join('\n')}\n`,
    conformanceMd: `${mdRows.join('\n')}\n`,
  };
}

/**
 * Emits `app/design/theme.json` (`prism/theme/1`).
 */
export function emitThemeJson(decl: PrismDeclaration): string {
  const obj: Record<string, unknown> = {
    schema: 'prism/theme/1',
    preset: decl.theme.preset,
  };
  if (decl.theme.accent) {
    obj.light = { primary: decl.theme.accent, brand: decl.theme.accent, ring: decl.theme.accent };
  }
  return `${JSON.stringify(obj, null, 2)}\n`;
}

/**
 * Emits `app/view/index.html`, optional `app/view/styles.css`, and `app/view/app.js`.
 *
 * - For authored interactive apps (`decl.view.authoredView`, `prism.app-view/1`):
 *   emits the complete author/agent-synthesized interactive page with `styles.css`,
 *   `prism.js` (`globalThis.prism.invoke`), and `app.js` — omitting `prism.css` so
 *   `prism-code` packs the authored app cleanly into `HOLOVIEW` alongside its Lean-proved
 *   model (`app/src/App.lex.tex`).
 * - For structured declarative forms (`decision-form`, `calculator`, `ledger-tracker`,
 *   `request-reply`): mechanically emits `prism.design/1` semantic HTML (`data-slot`)
 *   and thin `globalThis.prism.invoke(line)` client JS.
 */
export function emitDesignView(decl: PrismDeclaration): {
  readonly indexHtml: string;
  readonly stylesCss: string | null;
  readonly appJs: string;
  readonly isAuthoredApp: boolean;
} {
  if (!decl.view) {
    throw new Error('Cannot emit prism.design/1 view: no Design View is declared yet.');
  }
  const view = decl.view;

  // Procedural MediaType Organ view (DEC-009: 100% procedurally synthesized from declarative weights, zero LLM JS)
  if (view.proceduralOrgan) {
    const synth = synthesizeProceduralOrganView(decl, view.proceduralOrgan);
    return {
      indexHtml: synth.indexHtml,
      stylesCss: synth.stylesCss,
      appJs: synth.appJs,
      isAuthoredApp: true,
    };
  }

  // Legacy read-only custom authored interactive app (`prism.app-view/1`)
  if (view.authoredView) {
    const disclosureHtml = view.disclosure?.trim()
      ? `\n  <footer class="v-prism-disclosure" role="note">${escapeHtml(view.disclosure.trim())}</footer>`
      : '';
    const authoredBody = view.authoredView.html.trim();
    const indexHtml = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>${escapeHtml(view.title)}</title>
  <link rel="icon" href="data:,">
  <link rel="stylesheet" href="styles.css">
  <script type="module" src="prism.js"></script>
  <script type="module" src="app.js"></script>
</head>
<body>
${authoredBody}${disclosureHtml}
</body>
</html>
`;
    const baseTokens = `:root {
  --v-bg: #f8f9fa;
  --v-surface: #ffffff;
  --v-surface-alt: #f1f5f9;
  --v-border: #d8dee9;
  --v-text: #0f172a;
  --v-muted: #64748b;
  --v-accent: ${decl.theme.accent || '#0f5132'};
  --v-accent-text: #ffffff;
  color-scheme: light;
}
:root.dark, :root[data-theme="dark"] {
  --v-bg: #0b0f14;
  --v-surface: #111822;
  --v-surface-alt: #182230;
  --v-border: #263244;
  --v-text: #f1f5f9;
  --v-muted: #94a3b8;
  --v-accent: ${decl.theme.accent || '#10b981'};
  --v-accent-text: #062116;
  color-scheme: dark;
}
* { box-sizing: border-box; }
html, body {
  margin: 0;
  padding: 0;
  min-height: 100%;
  background: var(--v-bg);
  color: var(--v-text);
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.v-prism-disclosure {
  margin: 12px auto 16px;
  max-width: 680px;
  padding: 0 16px;
  text-align: center;
  font-size: 11px;
  color: var(--v-muted);
}
`;
    const stylesCss = `${baseTokens}\n${view.authoredView.css.trim()}\n`;
    const appJs = `${view.authoredView.js.trim()}\n`;
    return { indexHtml, stylesCss, appJs, isAuthoredApp: true };
  }
  const fields = decl.encoding?.fields ?? [];
  const fieldByName = new Map(fields.map((f) => [f.name, f]));
  const hasHistory = view.elements.some((e) => e.slot === 'history-list');

  // Render declared input controls in declared View order
  const controlBlocks: string[] = [];
  for (const el of view.elements) {
    if (el.slot === 'field-input') {
      const f = fieldByName.get(el.binding);
      if (!f) continue;
      const fid = `field-${f.name}`;
      const unitHint = f.unit ? ` (${escapeHtml(f.unit)})` : '';
      if (f.kind === 'bool') {
        controlBlocks.push(`          <fieldset data-slot="choice-group" id="${fid}-field" class="grid gap-2">
            <legend data-slot="label">${escapeHtml(el.label)}</legend>
            <div class="grid grid-cols-2 gap-3">
              <label data-slot="choice">
                <input type="radio" name="${escapeHtml(f.name)}" value="1"${f.defaultValue === 1 ? ' checked' : ''}>
                <span class="font-medium">Yes</span>
              </label>
              <label data-slot="choice">
                <input type="radio" name="${escapeHtml(f.name)}" value="0"${f.defaultValue !== 1 ? ' checked' : ''}>
                <span class="font-medium">No</span>
              </label>
            </div>
            <p id="${fid}-error" class="text-sm text-destructive" hidden></p>
          </fieldset>`);
      } else {
        controlBlocks.push(`          <div data-slot="field" id="${fid}-field">
            <label data-slot="label" for="${fid}">${escapeHtml(el.label)}${unitHint}</label>
            <input data-slot="input" id="${fid}" name="${escapeHtml(f.name)}" type="number" inputmode="numeric" min="${f.min}" max="${f.max}" value="${f.defaultValue}" required>
            <p id="${fid}-error" class="text-sm text-destructive" hidden></p>
          </div>`);
      }
    } else if (el.slot === 'choice-group') {
      const f = fieldByName.get(el.binding);
      if (!f || !f.choices?.length) continue;
      const fid = `field-${f.name}`;
      const cols = f.choices.length <= 2 ? 'grid-cols-2' : 'grid-cols-2 md:grid-cols-4';
      const radios = f.choices
        .map(
          (choice, idx) => `              <label data-slot="choice">
                <input type="radio" name="${escapeHtml(f.name)}" value="${idx}"${f.defaultValue === idx ? ' checked' : ''}>
                <span class="font-medium">${escapeHtml(choice)}</span>
              </label>`,
        )
        .join('\n');
      controlBlocks.push(`          <fieldset data-slot="choice-group" id="${fid}-field" class="grid gap-2">
            <legend data-slot="label">${escapeHtml(el.label)}</legend>
            <div class="grid ${cols} gap-3">
${radios}
            </div>
            <p id="${fid}-error" class="text-sm text-destructive" hidden></p>
          </fieldset>`);
    }
  }

  const resultLabel =
    view.elements.find((e) => e.slot === 'result-card')?.label ?? 'Model Answer';

  const historySection = hasHistory
    ? `
      <section data-slot="card" aria-labelledby="history-heading">
        <div data-slot="card-header">
          <h2 data-slot="card-title" id="history-heading">Recent Evaluations</h2>
          <p data-slot="card-description" id="history-count">0 evaluated</p>
        </div>
        <div data-slot="card-content">
          <div data-slot="empty" id="history-empty">
            <h3>No evaluations yet</h3>
            <p>Submit values above and the proved model computes each result.</p>
          </div>
          <ul data-slot="list" id="history-list" hidden></ul>
        </div>
      </section>`
    : '';

  const iconName =
    view.viewpoint === 'ledger-tracker'
      ? 'wallet'
      : view.viewpoint === 'calculator'
        ? 'calculator'
        : view.viewpoint === 'request-reply'
          ? 'message-square'
          : 'sparkles';

  const isCalculator = view.viewpoint === 'calculator';
  const isLedger = view.viewpoint === 'ledger-tracker';
  const isRequestReply = view.viewpoint === 'request-reply';

  const ledgerSummaryRow = isLedger
    ? `
      <section class="grid gap-4 sm:grid-cols-3" aria-label="Summary figures">
        <div data-slot="card">
          <div data-slot="card-content" class="pt-4">
            <div data-slot="stat">
              <p data-slot="stat-label">${escapeHtml(resultLabel)}</p>
              <p data-slot="stat-value" id="summary-latest">—</p>
            </div>
          </div>
        </div>
        <div data-slot="card">
          <div data-slot="card-content" class="pt-4">
            <div data-slot="stat">
              <p data-slot="stat-label">Checks Run</p>
              <p data-slot="stat-value" id="summary-count">0</p>
            </div>
          </div>
        </div>
        <div data-slot="card">
          <div data-slot="card-content" class="pt-4">
            <div data-slot="stat">
              <p data-slot="stat-label">Refusals Caught</p>
              <p data-slot="stat-value" id="summary-refusals">0</p>
            </div>
          </div>
        </div>
      </section>`
    : '';

  const mainStage = isRequestReply
    ? `      <section data-slot="card" aria-labelledby="thread-heading">
        <div data-slot="card-header">
          <h2 data-slot="card-title" id="thread-heading">Model Conversation</h2>
          <p data-slot="card-description">Each line is evaluated directly by the proved model.</p>
        </div>
        <div data-slot="card-content" class="grid gap-4">
          <div data-slot="stat" id="result-stat">
            <p data-slot="stat-label">${escapeHtml(resultLabel)}</p>
            <p data-slot="stat-value" id="result-value">—</p>
          </div>
          <div data-slot="alert" data-variant="destructive" id="refusal-alert" role="alert" hidden>
            <p data-slot="alert-title">Model refused request</p>
            <p data-slot="alert-description" id="refusal-text"></p>
          </div>
          <ol data-slot="thread" id="reply-thread" aria-live="polite" aria-label="Requests and answers"></ol>
          <p class="text-xs text-muted-foreground" id="protocol-trace">Ready.</p>
        </div>
      </section>
      <form data-slot="card" id="model-form" autocomplete="off">
        <div data-slot="card-content" class="grid gap-4 pt-4">
${controlBlocks.join('\n')}
        </div>
        <div data-slot="card-footer" class="flex items-center justify-between gap-3">
          <button data-slot="button" id="submit-btn" type="submit">${escapeHtml(view.submitLabel)}</button>
          <button data-slot="button" data-variant="ghost" id="reset-btn" type="button">Reset</button>
        </div>
      </form>`
    : `${ledgerSummaryRow}
      <section class="grid gap-6 md:grid-cols-2 items-start">
        <form data-slot="card" id="model-form" autocomplete="off">
          <div data-slot="card-header">
            <h2 data-slot="card-title">${isCalculator ? 'Parameters' : 'Inputs'}</h2>
            <p data-slot="card-description">${isCalculator ? 'Adjust any value to recalculate live through the proved model.' : 'Values are sent directly to the proved model.'}</p>
          </div>
          <div data-slot="card-content" class="grid gap-4">
${controlBlocks.join('\n')}
          </div>
          <div data-slot="card-footer" class="flex items-center justify-between gap-3">
            <button data-slot="button" id="submit-btn" type="submit">${escapeHtml(view.submitLabel)}</button>
            <button data-slot="button" data-variant="ghost" id="reset-btn" type="button">Reset</button>
          </div>
        </form>

        <section data-slot="card" aria-labelledby="result-heading">
          <div data-slot="card-header">
            <h2 data-slot="card-title" id="result-heading">${escapeHtml(resultLabel)}</h2>
            <p data-slot="card-description">Computed deterministically by the model</p>
          </div>
          <div data-slot="card-content" class="grid gap-4">
            <div data-slot="stat" id="result-stat">
              <p data-slot="stat-label">${escapeHtml(resultLabel)}</p>
              <p data-slot="stat-value" id="result-value">—</p>
            </div>
            <div data-slot="alert" data-variant="destructive" id="refusal-alert" role="alert" hidden>
              <p data-slot="alert-title">Model refused request</p>
              <p data-slot="alert-description" id="refusal-text"></p>
            </div>
            <p class="text-xs text-muted-foreground" id="protocol-trace">Ready.</p>
          </div>
        </section>
      </section>`;

  const disclosureBlock = view.disclosure?.trim()
    ? `\n      <p class="text-center text-xs text-muted-foreground">${escapeHtml(view.disclosure.trim())}</p>`
    : '';

  const indexHtml = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>${escapeHtml(view.title)}</title>
  <link rel="icon" href="data:,">
  <link rel="stylesheet" href="prism.css">
  <script src="prism-ui.js"></script>
  <script type="module" src="prism.js"></script>
  <script type="module" src="app.js"></script>
</head>
<body>
  <div data-slot="app">
    <header data-slot="toolbar">
      <div data-slot="brand" class="min-w-0">
        <span data-slot="brand-mark" aria-hidden="true"><span data-slot="icon" data-icon="${iconName}"></span></span>
        <h1 data-slot="brand-name" class="truncate">${escapeHtml(view.title)}</h1>
      </div>
      <span data-slot="badge" data-variant="success" data-tooltip="Every answer is computed by the app's proved model">Proved model</span>
    </header>

    <main data-slot="page" class="grid gap-6 py-8">
      <header data-slot="page-header">
        <p data-slot="eyebrow">${escapeHtml(view.eyebrow)}</p>
        <h1>${escapeHtml(view.title)}</h1>
        <p>${escapeHtml(view.subtitle)}</p>
      </header>

${mainStage}
${historySection}${disclosureBlock}
      <p class="text-center text-xs text-muted-foreground">The model's rules are proved. This page is compiled from the declaration and only displays what the model answers.</p>
    </main>
  </div>
</body>
</html>
`;

  const fieldSpecJson = JSON.stringify(
    fields.map((f) => ({
      name: f.name,
      label: f.label,
      kind: f.kind,
      defaultValue: f.defaultValue,
      choices: f.choices ?? null,
    })),
  );
  const refusalMapJson = JSON.stringify(
    Object.fromEntries(decl.refusals.map((r) => [r.failureText, r.fieldHint ?? null])),
  );
  const toneMapJson = JSON.stringify(
    Object.fromEntries(
      decl.features
        .filter((f) => f.outcome.kind === 'label')
        .map((f) => [f.outcome.label, f.outcome.kind === 'label' ? f.outcome.tone : 'default']),
    ),
  );

  const appJs = `// Mechanically compiled by Verdant Studio. The proved model decides every answer;
// this script only serializes inputs, calls globalThis.prism.invoke(), and displays the answer (R1).
const { invoke } = globalThis.prism;
const { h } = globalThis.ui;

const FIELDS = ${fieldSpecJson};
const REFUSALS = ${refusalMapJson};
const TONES = ${toneMapJson};
const form = document.getElementById('model-form');
const submitBtn = document.getElementById('submit-btn');
const resetBtn = document.getElementById('reset-btn');
const resultStat = document.getElementById('result-stat');
const resultValue = document.getElementById('result-value');
const refusalAlert = document.getElementById('refusal-alert');
const refusalText = document.getElementById('refusal-text');
const protocolTrace = document.getElementById('protocol-trace');
const historyList = document.getElementById('history-list');
const historyEmpty = document.getElementById('history-empty');
const historyCount = document.getElementById('history-count');
const summaryLatest = document.getElementById('summary-latest');
const summaryCount = document.getElementById('summary-count');
const summaryRefusals = document.getElementById('summary-refusals');
const replyThread = document.getElementById('reply-thread');
const LIVE_CALC = ${isCalculator ? 'true' : 'false'};

const historyEntries = [];
let refusalTotal = 0;

function clearErrors() {
  refusalAlert.hidden = true;
  refusalText.textContent = '';
  for (const f of FIELDS) {
    const err = document.getElementById('field-' + f.name + '-error');
    const wrap = document.getElementById('field-' + f.name + '-field');
    if (err) { err.hidden = true; err.textContent = ''; }
    if (wrap) wrap.removeAttribute('data-invalid');
  }
}

function readRequestLine() {
  const parts = [];
  for (const f of FIELDS) {
    if (f.kind === 'choice' || f.kind === 'bool') {
      const checked = form.querySelector('input[name="' + f.name + '"]:checked');
      parts.push(checked ? String(checked.value) : String(f.defaultValue));
    } else {
      const el = document.getElementById('field-' + f.name);
      const raw = el ? Number.parseInt(el.value, 10) : f.defaultValue;
      parts.push(String(Number.isFinite(raw) && raw >= 0 ? raw : 0));
    }
  }
  return parts.join(' ');
}

function renderHistory() {
  if (!historyList || !historyEmpty || !historyCount) return;
  historyCount.textContent = historyEntries.length === 1 ? '1 evaluated' : historyEntries.length + ' evaluated';
  historyEmpty.hidden = historyEntries.length > 0;
  historyList.hidden = historyEntries.length === 0;
  historyList.replaceChildren(
    ...historyEntries.slice(0, 8).map((entry) =>
      h('li', { slot: 'list-item' },
        h('span', { class: 'font-medium truncate' }, entry.answer),
        h('span', { slot: 'badge', variant: entry.refused ? 'destructive' : 'secondary' }, entry.request),
      ),
    ),
  );
}

async function evaluateCurrent() {
  clearErrors();
  const line = readRequestLine();
  submitBtn.disabled = true;
  try {
    const answer = await invoke(line);
    const isRefusal = Object.prototype.hasOwnProperty.call(REFUSALS, answer) || answer === 'Invalid request format.';
    if (isRefusal) {
      resultValue.textContent = 'Refused';
      if (resultStat) resultStat.setAttribute('data-tone', 'destructive');
      refusalAlert.hidden = false;
      refusalText.textContent = answer;
      const hint = REFUSALS[answer];
      if (hint) {
        const err = document.getElementById('field-' + hint + '-error');
        const wrap = document.getElementById('field-' + hint + '-field');
        if (err) { err.textContent = answer; err.hidden = false; }
        if (wrap) wrap.setAttribute('data-invalid', 'true');
      }
    } else {
      resultValue.textContent = answer;
      const tone = TONES[answer] || 'brand';
      if (resultStat) resultStat.setAttribute('data-tone', tone);
    }
    protocolTrace.textContent = 'Request "' + line + '" -> "' + answer + '"';
    if (isRefusal) refusalTotal += 1;
    historyEntries.unshift({ request: line, answer, refused: isRefusal });
    if (summaryLatest) summaryLatest.textContent = isRefusal ? 'Refused' : answer;
    if (summaryCount) summaryCount.textContent = String(historyEntries.length);
    if (summaryRefusals) summaryRefusals.textContent = String(refusalTotal);
    if (replyThread) {
      replyThread.append(h('li', { slot: 'bubble', from: 'me' }, line));
      replyThread.append(h('li', { slot: 'bubble', from: 'them', ...(isRefusal ? { 'data-failed': true } : {}) }, answer));
    }
    renderHistory();
  } catch (error) {
    refusalAlert.hidden = false;
    refusalText.textContent = error instanceof Error ? error.message : String(error);
  } finally {
    submitBtn.disabled = false;
  }
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  evaluateCurrent();
});

if (LIVE_CALC) {
  let calcTimer = null;
  form.addEventListener('input', () => {
    if (calcTimer) clearTimeout(calcTimer);
    calcTimer = setTimeout(() => evaluateCurrent(), 80);
  });
  form.addEventListener('change', () => evaluateCurrent());
}

resetBtn.addEventListener('click', () => {
  form.reset();
  clearErrors();
  evaluateCurrent();
});

evaluateCurrent();
`;

  return { indexHtml, stylesCss: null, appJs, isAuthoredApp: false };
}

/**
 * Renames canonical template foundation files (`app/lexlean.toml`, `app/lakefile.toml`, etc.)
 * to the Prism's own slug/module name and re-stamps `lexlean.lock` SHA-256 digests so
 * `buildFromFiles` accepts the workspace identically to `prism-code init --name <slug>`.
 */
export function instantiateAppFoundation(
  foundationFiles: Readonly<Record<string, string>>,
  slug: string,
): Record<string, string> {
  const mod = pascalModuleName(slug);
  const crate = `prism-${slug}`;
  const forms: readonly [string, string][] = [
    ['prism-template', crate],
    ['PrismTemplate', mod],
    ['"template"', JSON.stringify(slug)],
  ];
  const named = new Set(['lexlean.toml', 'lakefile.toml', 'lake-manifest.json', 'prismpm.toml']);
  const out: Record<string, string> = {};

  for (const [rel, content] of Object.entries(foundationFiles)) {
    let text = content;
    if (named.has(rel)) {
      for (const [from, to] of forms) {
        text = text.split(from).join(to);
      }
    }
    out[rel] = text;
  }

  if (out['lexlean.lock']) {
    const digest = (rel: string) => sha256HexSync(out[rel] ?? '');
    let lock = out['lexlean.lock'];
    lock = lock.replace(
      /^project_config_sha256 = "[0-9a-f]{64}"$/m,
      () => `project_config_sha256 = "${digest('lexlean.toml')}"`,
    );
    lock = lock.replace(
      /(\[\[workspace_file\]\]\npath = "([^"\n]+)"\nsha256 = ")[0-9a-f]{64}"/g,
      (all, head: string, file: string) => (out[file] !== undefined ? `${head}${digest(file)}"` : all),
    );
    out['lexlean.lock'] = lock;
  }

  return out;
}

const COMPILE_CACHE = new WeakMap<PrismDeclaration, CompiledPrismWorkspace | null>();

/**
 * Full mechanical compilation of a `PrismDeclaration` into canonical Prism files +
 * compiled `prism.design/1` stylesheet/runtime for instant preview & WASM sealing.
 *
 * Memoized per immutable `PrismDeclaration` reference via WeakMap so repeated reads
 * across Store snapshots, Studio Preview, and Inspector cost O(1) with zero recomputation.
 *
 * Returns `null` when the Prism is still undefined / not yet synthesizable (zero fake placeholders).
 */
export function compilePrismDeclaration(decl: PrismDeclaration): CompiledPrismWorkspace | null {
  if (COMPILE_CACHE.has(decl)) {
    return COMPILE_CACHE.get(decl) ?? null;
  }
  const result = compilePrismDeclarationUncached(decl);
  COMPILE_CACHE.set(decl, result);
  return result;
}

function compilePrismDeclarationUncached(decl: PrismDeclaration): CompiledPrismWorkspace | null {
  if (!isPrismSynthesizable(decl)) {
    return null;
  }
  const { source: appLexTex, theoremMap } = emitLexLeanApp(decl);
  const prismToml = emitPrismToml(decl);
  const { idsToml, featureGherkin, conformanceMd } = emitConformanceRegisters(decl, theoremMap);
  const themeJson = emitThemeJson(decl);
  const { indexHtml, stylesCss, appJs, isAuthoredApp } = emitDesignView(decl);
  const routing = evaluateVerdantRouting(decl);

  const pageMap = new Map<string, Uint8Array>([
    ['index.html', enc.encode(indexHtml)],
    ['app.js', enc.encode(appJs)],
    ...(stylesCss !== null ? [['styles.css', enc.encode(stylesCss)] as const] : []),
  ]);
  const themeBytes = enc.encode(themeJson);
  const compiled = isAuthoredApp ? null : compileDesign(pageMap, themeBytes);
  const designDiagnostics = isAuthoredApp ? [] : checkDesign(pageMap, themeBytes);
  const resolvedTheme = resolveTheme(themeBytes);

  const appFiles: Record<string, string> = {
    'src/App.lex.tex': appLexTex,
    'design/theme.json': themeJson,
    'view/index.html': indexHtml,
    ...(stylesCss !== null ? { 'view/styles.css': stylesCss } : {}),
    'view/app.js': appJs,
  };

  const files: Record<string, string> = {
    'prism.toml': prismToml,
    'app/src/App.lex.tex': appLexTex,
    'app/design/theme.json': themeJson,
    'app/view/index.html': indexHtml,
    ...(stylesCss !== null ? { 'app/view/styles.css': stylesCss } : {}),
    'app/view/app.js': appJs,
    'model/ids.toml': idsToml,
    'features/suites/app.feature': featureGherkin,
    'CONFORMANCE.md': conformanceMd,
    'model/concept-opm.json': `${JSON.stringify(decl.concept, null, 2)}\n`,
    'model/security-stride.json': `${JSON.stringify(
      {
        assuranceLevel: decl.security.assuranceLevel,
        trustBoundaries: routing.trustBoundaries,
        threats: decl.security.threats,
      },
      null,
      2,
    )}\n`,
    'model/standards-routing.json': `${JSON.stringify(
      {
        summary: routing.summary,
        coreSpine: routing.coreSpine.map((s) => ({
          id: s.authority.id,
          name: s.authority.name,
          settled: s.settledCount,
          total: s.totalCount,
        })),
        applicableLayered: routing.activeLayeredStandards.map((s) => ({
          id: s.authority.id,
          name: s.authority.name,
          triggeredBy: s.triggeredBy,
          reason: s.reason,
          settled: s.settledCount,
          total: s.totalCount,
        })),
        excluded: routing.excludedStandards.map((e) => ({
          id: e.authority.id,
          name: e.authority.name,
          rationale: e.exclusionRationale,
        })),
      },
      null,
      2,
    )}\n`,
    'model/requirements.json': `${JSON.stringify(decl.requirements, null, 2)}\n`,
    'model/admissions.json': `${JSON.stringify(decl.admissions, null, 2)}\n`,
    'model/owner-decisions.json': `${JSON.stringify(decl.decisions, null, 2)}\n`,
  };

  return {
    files,
    appFiles,
    compiledCss: stylesCss ?? compiled?.css ?? '',
    compiledUiJs: compiled?.js ?? '',
    designSlots: compiled?.slots ?? [],
    designDiagnostics,
    themeModes: resolvedTheme.modes,
    theoremMap,
  };
}
