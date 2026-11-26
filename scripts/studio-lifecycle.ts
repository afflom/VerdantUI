#!/usr/bin/env node
/**
 * VerdantUI Lifecycle Governance CLI (`scripts/studio-lifecycle.ts`)
 *
 * Implements `afflom/verdant-studio` product & feature lifecycle governance for VerdantUI:
 *   - `check`        Verifies well-founded `Req | Std | Der` citations, monotone locked IDs,
 *                    file existence for features/invariants, performance budgets, upstream pins,
 *                    and zero drift in generated docs (`CONFORMANCE.md`, `docs/owner-decisions.md`,
 *                    `docs/product-lifecycle.md`).
 *   - `agenda`       Derives open lifecycle work directly from `model/` registers.
 *   - `publish`      Deterministically generates `CONFORMANCE.md`, `docs/owner-decisions.md`,
 *                    `docs/product-lifecycle.md`, and `CHANGELOG.md` from `model/` (Rule R1).
 *   - `release`      Derives semantic version bump from newly active features / upstream pins,
 *                    updates `package.json` + `model/release-ledger.json` + generated docs, and
 *                    optionally writes GitHub Release notes.
 *   - `track-studio` Checks `afflom/verdant-studio` HEAD registers against `model/studio-pin.json`.
 */

import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export type LifecycleCitation =
  | { readonly kind: 'req'; readonly decisionId: string }
  | { readonly kind: 'std'; readonly authorityId: string }
  | {
      readonly kind: 'der';
      readonly rule: 'standard' | 'pattern' | 'extrapolated';
      readonly inputs: readonly string[];
      readonly checkId: string;
    };

export type FeatureStage = 'proposed' | 'admitted' | 'active' | 'deprecated' | 'retired';

export interface LifecycleFeature {
  readonly id: string;
  readonly title: string;
  readonly stage: FeatureStage;
  readonly introducedIn: string;
  readonly ownerFile: string;
  readonly testFile: string;
  readonly invariantId: string;
  readonly citation: LifecycleCitation;
  readonly publicBenefitImpact: string;
}

export interface LifecycleAdmission {
  readonly construct: string;
  readonly citation: LifecycleCitation;
}

export interface OwnerDecisionEntry {
  readonly id: string;
  readonly at: string;
  readonly requirements: readonly string[];
  readonly quote: string;
}

export interface InvariantIdEntry {
  readonly code: string;
  readonly title: string;
  readonly owner: string;
  readonly evidence: string;
}

export interface AuthorityEntry {
  readonly id: string;
  readonly name: string;
  readonly edition: string;
  readonly source: string;
  readonly scope: string;
}

export interface PerformanceBudgetsFile {
  readonly schema: string;
  readonly authority: string;
  readonly budgets: {
    readonly turnCompletionOpfsStallMsMax: number;
    readonly frameYieldBudgetUsMax: number;
    readonly totalDistJsKiBMax: number;
    readonly mainIndexChunkGzipKiBMax: number;
    readonly studioCheckLatencyMsMax: number;
    readonly requireMultiCoreParityBench: boolean;
    readonly requireCoopCoepIsolation: boolean;
    readonly requireZeroAllocationHotViews: boolean;
  };
}

export interface ReleaseLedgerEntry {
  readonly version: string;
  readonly tag: string;
  readonly releasedAt: string;
  readonly commit: string;
  readonly engineTag: string;
  readonly studioCommit: string;
  readonly activeFeatureIds: readonly string[];
  readonly summary: string;
}

export interface StudioLifecycleModel {
  readonly cwd: string;
  readonly packageVersion: string;
  readonly authorities: readonly AuthorityEntry[];
  readonly decisions: readonly OwnerDecisionEntry[];
  readonly ids: readonly InvariantIdEntry[];
  readonly lockedIds: readonly string[];
  readonly admissions: readonly LifecycleAdmission[];
  readonly features: readonly LifecycleFeature[];
  readonly performanceBudgets: PerformanceBudgetsFile;
  readonly enginePin: {
    readonly repository: string;
    readonly tag: string;
    readonly version: string;
    readonly commit: string;
  };
  readonly studioPin: {
    readonly repository: string;
    readonly branch: string;
    readonly commit: string;
    readonly registers: ReadonlyArray<{ readonly idPrefix: string; readonly path: string; readonly sha256: string }>;
  };
  readonly releaseLedger: {
    readonly schema: string;
    readonly releases: readonly ReleaseLedgerEntry[];
  };
}

export interface LifecycleCheckError {
  readonly code: string;
  readonly path: string;
  readonly message: string;
}

export interface LifecycleCheckReport {
  readonly ok: boolean;
  readonly elapsedMs: number;
  readonly errors: readonly LifecycleCheckError[];
  readonly activeFeatures: number;
  readonly proposedFeatures: number;
  readonly admissionsChecked: number;
}

export interface LifecycleAgendaItem {
  readonly priority: 'high' | 'medium' | 'info';
  readonly kind: 'feature-promotion' | 'release-ready' | 'doc-sync';
  readonly id: string;
  readonly detail: string;
}

export interface SemanticReleasePlan {
  readonly currentVersion: string;
  readonly nextVersion: string;
  readonly bump: 'none' | 'patch' | 'minor' | 'major';
  readonly newlyActiveFeatures: readonly LifecycleFeature[];
  readonly engineUpgraded: boolean;
  readonly studioUpgraded: boolean;
  readonly releaseNotesMarkdown: string;
}

function parseTomlBlocks(raw: string, header: string): Array<Record<string, string>> {
  const blocks: Array<Record<string, string>> = [];
  let current: Record<string, string> | null = null;
  for (const rawLine of raw.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    if (line === `[[${header}]]`) {
      current = {};
      blocks.push(current);
      continue;
    }
    if (!current) continue;
    const m = line.match(/^([A-Za-z0-9_]+)\s*=\s*"((?:[^"\\]|\\.)*)"$/);
    if (m) {
      current[m[1]] = m[2].replace(/\\"/g, '"').replace(/\\\\/g, '\\');
    }
  }
  return blocks;
}

export function loadStudioLifecycleModel(cwd = process.cwd()): StudioLifecycleModel {
  const root = resolve(cwd);
  const readJson = <T>(rel: string): T => JSON.parse(readFileSync(join(root, rel), 'utf8')) as T;
  const readText = (rel: string): string => readFileSync(join(root, rel), 'utf8');

  const pkg = readJson<{ version?: string }>('package.json');
  const authoritiesRaw = readText('model/authorities.toml');
  const idsRaw = readText('model/ids.toml');
  const decisionsFile = readJson<{ decisions: OwnerDecisionEntry[] }>('model/owner-decisions.json');
  const admissionsFile = readJson<{
    lockedIds: string[];
    admissions: LifecycleAdmission[];
  }>('model/admissions.json');
  const featuresFile = readJson<{ features: LifecycleFeature[] }>('model/features.json');
  const performanceBudgets = readJson<PerformanceBudgetsFile>('model/performance-budgets.json');
  const enginePin = readJson<StudioLifecycleModel['enginePin']>('model/engine-pin.json');
  const studioPin = readJson<StudioLifecycleModel['studioPin']>('model/studio-pin.json');
  const releaseLedger = readJson<StudioLifecycleModel['releaseLedger']>('model/release-ledger.json');

  const authorities: AuthorityEntry[] = parseTomlBlocks(authoritiesRaw, 'authority').map((b) => ({
    id: b.id ?? '',
    name: b.name ?? '',
    edition: b.edition ?? '',
    source: b.source ?? '',
    scope: b.scope ?? '',
  }));

  const ids: InvariantIdEntry[] = parseTomlBlocks(idsRaw, 'id').map((b) => ({
    code: b.code ?? '',
    title: b.title ?? '',
    owner: b.owner ?? '',
    evidence: b.evidence ?? '',
  }));

  return {
    cwd: root,
    packageVersion: pkg.version ?? '0.1.0',
    authorities,
    decisions: decisionsFile.decisions ?? [],
    ids,
    lockedIds: admissionsFile.lockedIds ?? [],
    admissions: admissionsFile.admissions ?? [],
    features: featuresFile.features ?? [],
    performanceBudgets,
    enginePin,
    studioPin,
    releaseLedger,
  };
}

function validateCitation(
  citation: LifecycleCitation,
  decisionIds: ReadonlySet<string>,
  authorityIds: ReadonlySet<string>,
  idCodes: ReadonlySet<string>,
  admittedSoFar: ReadonlySet<string>,
): string | null {
  if (citation.kind === 'req') {
    return decisionIds.has(citation.decisionId)
      ? null
      : `Unresolved Req decisionId "${citation.decisionId}"`;
  }
  if (citation.kind === 'std') {
    return authorityIds.has(citation.authorityId)
      ? null
      : `Unresolved Std authorityId "${citation.authorityId}"`;
  }
  if (citation.kind === 'der') {
    if (!idCodes.has(citation.checkId)) {
      return `Unresolved Der checkId "${citation.checkId}"`;
    }
    for (const input of citation.inputs) {
      if (!admittedSoFar.has(input)) {
        return `Der input "${input}" is not previously admitted (well-foundedness violated)`;
      }
    }
    return null;
  }
  return 'Unknown citation kind';
}

export function renderGeneratedDocs(model: StudioLifecycleModel): Readonly<Record<string, string>> {
  const activeFeatures = model.features.filter((f) => f.stage === 'active');
  const proposedFeatures = model.features.filter((f) => f.stage === 'proposed');

  const conformance = [
    '# VerdantUI Conformance & Standards Register',
    '',
    '> Generated mechanically by `npm run studio:publish` from `model/` (Rule R1). Do not edit by hand.',
    '',
    `- **Service Charter:** Free Public Benefit AI Service (Apache-2.0, 100% client-side, zero servers, zero tracking, zero paywalls)`,
    `- **Compute Substrate:** \`${model.enginePin.repository}@${model.enginePin.tag}\` (\`${model.enginePin.commit.slice(0, 12)}\`)`,
    `- **Lifecycle Authority:** \`${model.studioPin.repository}@${model.studioPin.commit.slice(0, 12)}\``,
    `- **Performance Budgets:** Turn OPFS stall \`<= ${model.performanceBudgets.budgets.turnCompletionOpfsStallMsMax} ms\`, Frame yield KPI \`<= ${model.performanceBudgets.budgets.frameYieldBudgetUsMax} us\`, JS bundle \`<= ${model.performanceBudgets.budgets.totalDistJsKiBMax} KiB\``,
    '',
    '## Locked Architectural & Performance Invariants',
    '',
    '| ID | Owner | Evidence | Statement |',
    '| --- | --- | --- | --- |',
    ...model.ids.map((id) => `| \`${id.code}\` | \`${id.owner}\` | \`${id.evidence}\` | ${id.title} |`),
    '',
    '## Pinned Authorities',
    '',
    '| Authority | Edition | Scope |',
    '| --- | --- | --- |',
    ...model.authorities.map((a) => `| \`${a.id}\` | ${a.edition} | ${a.scope} |`),
    '',
  ].join('\n');

  const ownerDecisionsDoc = [
    '# VerdantUI Owner Decisions',
    '',
    '> Generated mechanically by `npm run studio:publish` from `model/owner-decisions.json`. Do not edit by hand.',
    '',
    ...model.decisions.flatMap((d) => [
      `## ${d.id} (${d.at})`,
      '',
      `- **Requirements:** ${d.requirements.map((r) => `\`${r}\``).join(', ')}`,
      `- **Decision:** "${d.quote}"`,
      '',
    ]),
  ].join('\n');

  const productLifecycleDoc = [
    '# VerdantUI Product & Feature Lifecycle',
    '',
    '> Generated mechanically by `npm run studio:publish` from `model/features.json` and `model/release-ledger.json`.',
    '',
    `- **Active Features:** ${activeFeatures.length}`,
    `- **Proposed / Planned Features:** ${proposedFeatures.length}`,
    `- **Recorded Releases:** ${model.releaseLedger.releases.length}`,
    '',
    '## Feature Register',
    '',
    '| Feature ID | Stage | Target | Invariant | Title |',
    '| --- | --- | --- | --- | --- |',
    ...model.features.map(
      (f) => `| \`${f.id}\` | **${f.stage}** | \`v${f.introducedIn}\` | \`${f.invariantId}\` | ${f.title} |`,
    ),
    '',
    '## Release Ledger',
    '',
    '| Version | Engine | Studio Pin | Active Features | Summary |',
    '| --- | --- | --- | --- | --- |',
    ...model.releaseLedger.releases.map(
      (r) =>
        `| \`${r.tag}\` | \`${r.engineTag}\` | \`${r.studioCommit.slice(0, 10)}\` | ${r.activeFeatureIds.length} | ${r.summary} |`,
    ),
    '',
  ].join('\n');

  const changelog = [
    '# Changelog',
    '',
    '> Generated mechanically by `npm run studio:publish` from `model/release-ledger.json` and `model/features.json`.',
    '',
    ...[...model.releaseLedger.releases].reverse().flatMap((r) => [
      `## ${r.tag} (${r.releasedAt.slice(0, 10)})`,
      '',
      `${r.summary}`,
      '',
      `- **Engine Pin:** \`afflom/verdant-engine@${r.engineTag}\``,
      `- **Studio Pin:** \`afflom/verdant-studio@${r.studioCommit.slice(0, 12)}\``,
      `- **Active Features:** ${r.activeFeatureIds.map((id) => `\`${id}\``).join(', ')}`,
      '',
    ]),
  ].join('\n');

  return {
    'CONFORMANCE.md': conformance,
    'docs/owner-decisions.md': ownerDecisionsDoc,
    'docs/product-lifecycle.md': productLifecycleDoc,
    'CHANGELOG.md': changelog,
  };
}

export function checkStudioLifecycle(
  model: StudioLifecycleModel,
  options?: { readonly verifyGeneratedDocs?: boolean; readonly verifyFilesExist?: boolean },
): LifecycleCheckReport {
  const t0 = performance.now();
  const errors: LifecycleCheckError[] = [];
  const verifyFiles = options?.verifyFilesExist ?? true;
  const verifyDocs = options?.verifyGeneratedDocs ?? true;

  const decisionIds = new Set(model.decisions.map((d) => d.id));
  const authorityIds = new Set(model.authorities.map((a) => a.id));
  const idCodes = new Set(model.ids.map((i) => i.code));

  // 1. Monotone lock: every locked ID must exist in model/ids.toml and be admitted
  const admittedConstructs = new Set(model.admissions.map((a) => a.construct));
  for (const locked of model.lockedIds) {
    if (!idCodes.has(locked)) {
      errors.push({
        code: 'MONOTONE-LOCK-BROKEN',
        path: 'model/ids.toml',
        message: `Locked invariant "${locked}" was dropped from model/ids.toml`,
      });
    }
    if (!admittedConstructs.has(locked)) {
      errors.push({
        code: 'LOCKED-ID-UNADMITTED',
        path: 'model/admissions.json',
        message: `Locked invariant "${locked}" has no citation in model/admissions.json`,
      });
    }
  }

  // 2. Well-founded admission check (Studio.Admission parity: acyclic Req / Std / Der)
  const admittedSoFar = new Set<string>();
  const seenAdmissions = new Set<string>();
  for (const adm of model.admissions) {
    if (seenAdmissions.has(adm.construct)) {
      errors.push({
        code: 'DUPLICATE-ADMISSION',
        path: 'model/admissions.json',
        message: `Duplicate admission entry for "${adm.construct}"`,
      });
      continue;
    }
    seenAdmissions.add(adm.construct);
    const err = validateCitation(adm.citation, decisionIds, authorityIds, idCodes, admittedSoFar);
    if (err) {
      errors.push({
        code: 'UNRESOLVED-CITATION',
        path: 'model/admissions.json',
        message: `${adm.construct}: ${err}`,
      });
    } else {
      admittedSoFar.add(adm.construct);
    }
  }

  // 3. Feature lifecycle validation
  const seenFeatures = new Set<string>();
  const validStages = new Set<FeatureStage>(['proposed', 'admitted', 'active', 'deprecated', 'retired']);
  for (const feat of model.features) {
    if (seenFeatures.has(feat.id)) {
      errors.push({
        code: 'DUPLICATE-FEATURE',
        path: 'model/features.json',
        message: `Duplicate feature ID "${feat.id}"`,
      });
    }
    seenFeatures.add(feat.id);
    if (!validStages.has(feat.stage)) {
      errors.push({
        code: 'INVALID-FEATURE-STAGE',
        path: 'model/features.json',
        message: `${feat.id}: invalid stage "${String(feat.stage)}"`,
      });
    }
    if (!idCodes.has(feat.invariantId)) {
      errors.push({
        code: 'UNKNOWN-FEATURE-INVARIANT',
        path: 'model/features.json',
        message: `${feat.id}: invariantId "${feat.invariantId}" is not in model/ids.toml`,
      });
    }
    const citErr = validateCitation(feat.citation, decisionIds, authorityIds, idCodes, admittedSoFar);
    if (citErr) {
      errors.push({
        code: 'UNRESOLVED-FEATURE-CITATION',
        path: 'model/features.json',
        message: `${feat.id}: ${citErr}`,
      });
    }
    if (verifyFiles && feat.stage === 'active') {
      if (!existsSync(join(model.cwd, feat.ownerFile))) {
        errors.push({
          code: 'MISSING-OWNER-FILE',
          path: feat.ownerFile,
          message: `Active feature ${feat.id} ownerFile does not exist: ${feat.ownerFile}`,
        });
      }
      if (!existsSync(join(model.cwd, feat.testFile))) {
        errors.push({
          code: 'MISSING-TEST-FILE',
          path: feat.testFile,
          message: `Active feature ${feat.id} testFile does not exist: ${feat.testFile}`,
        });
      }
    }
  }

  // 4. Performance budgets & Studio pin sanity
  const b = model.performanceBudgets.budgets;
  if (b.turnCompletionOpfsStallMsMax > 0) {
    errors.push({
      code: 'BUDGET-REGRESSION',
      path: 'model/performance-budgets.json',
      message: 'turnCompletionOpfsStallMsMax must remain 0 ms',
    });
  }
  if (b.frameYieldBudgetUsMax > 16_666) {
    errors.push({
      code: 'BUDGET-REGRESSION',
      path: 'model/performance-budgets.json',
      message: `frameYieldBudgetUsMax (${b.frameYieldBudgetUsMax}) exceeds 60fps KPI (16666 us)`,
    });
  }
  if (!/^[0-9a-f]{40}$/.test(model.studioPin.commit)) {
    errors.push({
      code: 'INVALID-STUDIO-PIN',
      path: 'model/studio-pin.json',
      message: 'studio-pin.json commit must be a 40-char hex SHA',
    });
  }

  // 5. Generated documentation parity (Rule R1: generated from model/, never hand-drifted)
  if (verifyDocs) {
    const expectedDocs = renderGeneratedDocs(model);
    for (const [relPath, expected] of Object.entries(expectedDocs)) {
      const absPath = join(model.cwd, relPath);
      if (!existsSync(absPath)) {
        errors.push({
          code: 'MISSING-GENERATED-DOC',
          path: relPath,
          message: `Generated file ${relPath} is missing; run \`npm run studio:publish\``,
        });
      } else {
        const actual = readFileSync(absPath, 'utf8');
        if (actual !== expected) {
          errors.push({
            code: 'GENERATED-DOC-DRIFT',
            path: relPath,
            message: `${relPath} drifted from model/; run \`npm run studio:publish\``,
          });
        }
      }
    }
  }

  return {
    ok: errors.length === 0,
    elapsedMs: Math.round((performance.now() - t0) * 10) / 10,
    errors,
    activeFeatures: model.features.filter((f) => f.stage === 'active').length,
    proposedFeatures: model.features.filter((f) => f.stage === 'proposed').length,
    admissionsChecked: model.admissions.length,
  };
}

export function publishStudioDocs(model: StudioLifecycleModel): readonly string[] {
  const docs = renderGeneratedDocs(model);
  const written: string[] = [];
  for (const [relPath, content] of Object.entries(docs)) {
    const absPath = join(model.cwd, relPath);
    mkdirSync(dirname(absPath), { recursive: true });
    writeFileSync(absPath, content, 'utf8');
    written.push(relPath);
  }
  return written;
}

function parseSemver(v: string): [number, number, number] {
  const m = v.replace(/^v/, '').match(/^(\d+)\.(\d+)\.(\d+)/);
  if (!m) return [0, 1, 0];
  return [Number(m[1]), Number(m[2]), Number(m[3])];
}

export function planSemanticRelease(
  model: StudioLifecycleModel,
  forcedBump: 'auto' | 'patch' | 'minor' | 'major' = 'auto',
): SemanticReleasePlan {
  const latest = model.releaseLedger.releases[model.releaseLedger.releases.length - 1];
  const currentVersion = latest?.version ?? model.packageVersion;
  const releasedFeatureSet = new Set(latest?.activeFeatureIds ?? []);
  const newlyActiveFeatures = model.features.filter(
    (f) => f.stage === 'active' && !releasedFeatureSet.has(f.id),
  );
  const engineUpgraded = Boolean(latest && latest.engineTag !== model.enginePin.tag);
  const studioUpgraded = Boolean(latest && latest.studioCommit !== model.studioPin.commit);

  let bump: SemanticReleasePlan['bump'] = 'none';
  if (forcedBump !== 'auto') {
    bump = forcedBump;
  } else if (newlyActiveFeatures.length > 0) {
    bump = 'minor';
  } else if (engineUpgraded || studioUpgraded) {
    bump = 'patch';
  }

  const [maj, min, pat] = parseSemver(currentVersion);
  const nextVersion =
    bump === 'major'
      ? `${maj + 1}.0.0`
      : bump === 'minor'
        ? `${maj}.${min + 1}.0`
        : bump === 'patch'
          ? `${maj}.${min}.${pat + 1}`
          : currentVersion;

  const lines: string[] = [
    `# VerdantUI v${nextVersion}`,
    '',
    'Free Public Benefit AI Service — 100% client-side multi-threaded Verdant WASM SIMD.',
    '',
  ];
  if (newlyActiveFeatures.length > 0) {
    lines.push('## Newly Active Features', '');
    for (const f of newlyActiveFeatures) {
      lines.push(`- **${f.id} — ${f.title}**: ${f.publicBenefitImpact}`);
    }
    lines.push('');
  }
  if (engineUpgraded) {
    lines.push(`- Upgraded Verdant Engine pin to \`${model.enginePin.repository}@${model.enginePin.tag}\``);
  }
  if (studioUpgraded) {
    lines.push(`- Updated Verdant Studio lifecycle pin to \`${model.studioPin.commit.slice(0, 12)}\``);
  }
  lines.push(
    '',
    '## Provenance',
    `- Engine: \`${model.enginePin.repository}@${model.enginePin.tag}\` (\`${model.enginePin.commit.slice(0, 12)}\`)`,
    `- Studio Pin: \`${model.studioPin.repository}@${model.studioPin.commit.slice(0, 12)}\``,
    '',
  );

  return {
    currentVersion,
    nextVersion,
    bump,
    newlyActiveFeatures,
    engineUpgraded,
    studioUpgraded,
    releaseNotesMarkdown: lines.join('\n'),
  };
}

export function deriveStudioAgenda(model: StudioLifecycleModel): readonly LifecycleAgendaItem[] {
  const items: LifecycleAgendaItem[] = [];
  for (const feat of model.features) {
    if (feat.stage === 'proposed' || feat.stage === 'admitted') {
      items.push({
        priority: 'medium',
        kind: 'feature-promotion',
        id: feat.id,
        detail: `Promote ${feat.id} (${feat.title}) from ${feat.stage} -> active for v${feat.introducedIn}`,
      });
    }
  }
  const rel = planSemanticRelease(model);
  if (rel.bump !== 'none') {
    items.push({
      priority: 'high',
      kind: 'release-ready',
      id: `v${rel.nextVersion}`,
      detail: `Release v${rel.nextVersion} (${rel.bump}) ready: ${rel.newlyActiveFeatures.length} newly active feature(s)`,
    });
  }
  const check = checkStudioLifecycle(model, { verifyGeneratedDocs: true, verifyFilesExist: false });
  for (const err of check.errors) {
    if (err.code === 'GENERATED-DOC-DRIFT' || err.code === 'MISSING-GENERATED-DOC') {
      items.push({
        priority: 'high',
        kind: 'doc-sync',
        id: err.path,
        detail: err.message,
      });
    }
  }
  return items;
}

export function applySemanticRelease(
  model: StudioLifecycleModel,
  plan: SemanticReleasePlan,
  commitSha = process.env.GITHUB_SHA || 'local',
): ReleaseLedgerEntry {
  const pkgPath = join(model.cwd, 'package.json');
  const pkg = JSON.parse(readFileSync(pkgPath, 'utf8')) as Record<string, unknown>;
  pkg.version = plan.nextVersion;
  writeFileSync(pkgPath, `${JSON.stringify(pkg, null, 2)}\n`, 'utf8');

  const activeFeatureIds = model.features.filter((f) => f.stage === 'active').map((f) => f.id);
  const summary =
    plan.newlyActiveFeatures.length > 0
      ? plan.newlyActiveFeatures.map((f) => f.title).join('; ')
      : `Maintenance & upstream pin release (${model.enginePin.tag})`;

  const entry: ReleaseLedgerEntry = {
    version: plan.nextVersion,
    tag: `v${plan.nextVersion}`,
    releasedAt: new Date().toISOString(),
    commit: commitSha,
    engineTag: model.enginePin.tag,
    studioCommit: model.studioPin.commit,
    activeFeatureIds,
    summary,
  };

  const ledgerPath = join(model.cwd, 'model/release-ledger.json');
  const updatedLedger = {
    schema: 'verdant-ui/release-ledger/1',
    releases: [...model.releaseLedger.releases, entry],
  };
  writeFileSync(ledgerPath, `${JSON.stringify(updatedLedger, null, 2)}\n`, 'utf8');
  return entry;
}

async function trackStudioUpstream(cwd: string, writeMode: boolean): Promise<void> {
  const pinPath = join(resolve(cwd), 'model/studio-pin.json');
  const current = JSON.parse(readFileSync(pinPath, 'utf8')) as StudioLifecycleModel['studioPin'] & {
    schema: string;
    requirement: string;
    pinnedAt: string;
  };
  const headers: Record<string, string> = {
    Accept: 'application/vnd.github+json',
    'User-Agent': 'VerdantUI-track-studio',
  };
  if (process.env.GITHUB_TOKEN) {
    headers.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
  }
  const commitRes = await fetch(
    `https://api.github.com/repos/${current.repository}/commits/${current.branch}`,
    { headers },
  );
  if (!commitRes.ok) {
    throw new Error(`HTTP ${commitRes.status} fetching ${current.repository} HEAD`);
  }
  const commitJson = (await commitRes.json()) as { sha: string };
  const nextSha = commitJson.sha;

  const registers = await Promise.all(
    current.registers.map(async (r) => {
      const res = await fetch(
        `https://raw.githubusercontent.com/${current.repository}/${nextSha}/${r.path}`,
        { headers: { 'User-Agent': 'VerdantUI-track-studio' } },
      );
      if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${r.path}`);
      const buf = new Uint8Array(await res.arrayBuffer());
      return { ...r, sha256: createHash('sha256').update(buf).digest('hex') };
    }),
  );

  const unchanged =
    current.commit === nextSha && JSON.stringify(current.registers) === JSON.stringify(registers);
  if (unchanged) {
    console.log(
      `track-studio: UP TO DATE · ${current.repository}@${nextSha.slice(0, 12)} · ${registers.length}/${registers.length} registers verified`,
    );
    return;
  }
  if (writeMode) {
    const next = {
      ...current,
      commit: nextSha,
      pinnedAt: new Date().toISOString(),
      registers,
    };
    writeFileSync(pinPath, `${JSON.stringify(next, null, 2)}\n`, 'utf8');
    console.log(`track-studio: UPDATED model/studio-pin.json -> ${current.repository}@${nextSha.slice(0, 12)}`);
  } else {
    console.log(
      `track-studio: UPDATE AVAILABLE · pinned=${current.commit.slice(0, 12)} -> latest=${nextSha.slice(0, 12)}. Run with --write to stage.`,
    );
  }
}

async function main(): Promise<void> {
  const cmd = process.argv[2] ?? 'check';
  const cwd = process.cwd();

  if (cmd === 'track-studio') {
    await trackStudioUpstream(cwd, process.argv.includes('--write'));
    return;
  }

  const model = loadStudioLifecycleModel(cwd);

  if (cmd === 'publish') {
    const written = publishStudioDocs(model);
    console.log(`studio-publish: PASS · wrote ${written.join(', ')}`);
    return;
  }

  if (cmd === 'check') {
    const report = checkStudioLifecycle(model);
    if (!report.ok) {
      console.error(`studio-check: REFUSED (${report.errors.length} error(s))`);
      for (const e of report.errors) {
        console.error(`  - [${e.code}] ${e.path}: ${e.message}`);
      }
      process.exit(1);
    }
    console.log(
      `studio-check: PASS · admissions=${report.admissionsChecked} · features=${report.activeFeatures} active/${report.proposedFeatures} proposed · ${report.elapsedMs}ms`,
    );
    return;
  }

  if (cmd === 'agenda') {
    const items = deriveStudioAgenda(model);
    if (items.length === 0) {
      console.log('studio-agenda: CLEAN · 0 open lifecycle items');
      return;
    }
    console.log(`studio-agenda: ${items.length} open item(s)`);
    for (const item of items) {
      console.log(`  - [${item.priority}/${item.kind}] ${item.id}: ${item.detail}`);
    }
    return;
  }

  if (cmd === 'release') {
    const bumpArg = process.argv.find((a) => a.startsWith('--bump='))?.split('=')[1] as
      | 'auto'
      | 'patch'
      | 'minor'
      | 'major'
      | undefined;
    const writeMode = process.argv.includes('--write');
    const notesIdx = process.argv.indexOf('--notes-out');
    const notesOut = notesIdx >= 0 ? process.argv[notesIdx + 1] : undefined;

    const plan = planSemanticRelease(model, bumpArg ?? 'auto');
    if (notesOut) {
      mkdirSync(dirname(resolve(cwd, notesOut)), { recursive: true });
      writeFileSync(resolve(cwd, notesOut), plan.releaseNotesMarkdown, 'utf8');
    }
    if (plan.bump === 'none') {
      console.log(`studio-release: NOOP · already at v${plan.currentVersion}`);
      return;
    }
    if (writeMode) {
      const entry = applySemanticRelease(model, plan);
      const refreshed = loadStudioLifecycleModel(cwd);
      publishStudioDocs(refreshed);
      console.log(`studio-release: RELEASED ${entry.tag} · ${entry.activeFeatureIds.length} active features`);
    } else {
      console.log(
        `studio-release: READY · v${plan.currentVersion} -> v${plan.nextVersion} (${plan.bump}) · ${plan.newlyActiveFeatures.length} new feature(s)`,
      );
    }
    return;
  }

  console.error(`Unknown studio-lifecycle command: "${cmd}". Use check|agenda|publish|release|track-studio.`);
  process.exit(1);
}

const isMain =
  process.argv[1] !== undefined &&
  resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url));

if (isMain) {
  main().catch((err) => {
    console.error(`studio-lifecycle: ERROR — ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
  });
}
