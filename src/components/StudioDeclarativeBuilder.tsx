/**
 * Guided Verdant Prism Wizard Builder (`src/components/StudioDeclarativeBuilder.tsx`)
 *
 * Opinionated, calm creator workflow over the closed `VerdantStudioApi`:
 * - Stage 1: Concept & OPM (`ISO/PAS 19450`, `W3C PROV-O`, `ISO 29148`, Context Routing Weights)
 * - Stage 2: System & Rules (`ISO 42010`, UML behavior, JSON Schema 2020-12, Capabilities, Rules, Refusals, Examples)
 * - Stage 3: Security & Threats (`MICROSOFT-TMT-STRIDE`, `OWASP ASVS 5.0.0`, Trust Boundaries)
 * - Stage 4: Design & UX (`ISO 9241-210`, `W3C WCAG 2.2`, `prism.design/1` View & Theme)
 * - Stage 5: Routed Standards (dynamically shown ONLY when Core weights activate layered standards)
 * - Stage 6: Verify & Seal (Completeness Gate, WASM Lean proof & `.holo` v6 seal, Decisions, Audit Log)
 *
 * Zero raw code editing; every mutation flows strictly through `dispatchStudioOp`.
 * Optional reference examples stay collapsed in the quiet `Examples` popover (`REQ-CONTENT-BOUNDARY`).
 */

import { useEffect, useMemo, useState, type FC, type FormEvent } from 'react';
import {
  AlertTriangle,
  ArrowDown,
  ArrowRight,
  ArrowUp,
  BookOpen,
  Check,
  CheckCircle2,
  Circle,
  Compass,
  Cpu,
  Globe,
  Layers,
  Loader2,
  Palette,
  Pencil,
  Plus,
  Scale,
  Shield,
  ShieldCheck,
  Sparkles,
  Trash2,
  Upload,
  Wrench,
  XCircle,
} from 'lucide-react';
import {
  DECLARATIVE_STARTERS,
  PRISM_THEME_PRESETS,
  evaluateDeclaredModel,
  formatRequestLine,
  recommendedAssuranceLevel,
  type AssuranceLevel,
  type CompareOp,
  type DataClassification,
  type DeclarationCheckReport,
  type DeclarativeStarterId,
  type EncodedField,
  type FieldKind,
  type PrismAudience,
  type PrismDeclaration,
  type PrismDomain,
  type PrismJurisdiction,
  type PrismViewpoint,
  type RequirementOrigin,
  type StudioAgendaItem,
  type StudioApiOp,
  type StudioWizardStage,
  type ThreatDisposition,
} from '../engine/verdantStudioApi';
import type { VerdantRoutingReport } from '../engine/verdantStandardsRouter';
import type { JsonSchema2020Object } from '../engine/prismStudio';

interface StudioDeclarativeBuilderProps {
  declaration: PrismDeclaration;
  checkReport: DeclarationCheckReport;
  routing: VerdantRoutingReport;
  agenda: readonly StudioAgendaItem[];
  activeStage: StudioWizardStage;
  onSelectStage: (stage: StudioWizardStage) => void;
  onDispatchOp: (op: StudioApiOp, opts?: { readonly pushRevision?: boolean }) => Promise<boolean>;
  onLoadStarter: (starter: DeclarativeStarterId) => void;
  onImportHoloFile: () => void;
  onCheckAndBuild: () => void;
  onPublish: () => void;
  building: boolean;
  apiError: { code: string; message: string; remediation: string } | null;
  onClearApiError: () => void;
  /** Optional local-AI NL -> StudioApiOp[] proposal helper */
  modelStreamEnabled: boolean;
  onEnableModelStream: () => void;
  proposingOps: boolean;
  proposedOps: readonly StudioApiOp[] | null;
  onProposeOpsFromPrompt: (prompt: string) => void;
  onApplyProposedOps: () => void;
  onDismissProposedOps: () => void;
}

const COMPARE_LABELS: Record<CompareOp, string> = {
  eq: '=',
  neq: '≠',
  lt: '<',
  lte: '≤',
  gt: '>',
  gte: '≥',
};

const DOMAIN_OPTIONS: readonly { readonly id: PrismDomain; readonly label: string }[] = [
  { id: 'general', label: 'General Utility' },
  { id: 'finance', label: 'Finance & Budgeting' },
  { id: 'enterprise-ops', label: 'Operations & Policy' },
  { id: 'education', label: 'Education' },
  { id: 'health', label: 'Health & Wellness' },
  { id: 'games-media', label: 'Games & Media' },
];

const AUDIENCE_OPTIONS: readonly { readonly id: PrismAudience; readonly label: string }[] = [
  { id: 'personal', label: 'Personal (Solo)' },
  { id: 'team', label: 'Internal Team' },
  { id: 'public-adults', label: 'Public (Adults)' },
  { id: 'includes-minors', label: 'Includes Minors' },
];

const DATA_OPTIONS: readonly { readonly id: DataClassification; readonly label: string }[] = [
  { id: 'none', label: 'No Personal Data' },
  { id: 'operational', label: 'Operational / Business' },
  { id: 'personal', label: 'Personal Data (PII)' },
  { id: 'sensitive-regulated', label: 'Sensitive / Regulated' },
];

const JURISDICTION_OPTIONS: readonly { readonly id: PrismJurisdiction; readonly label: string }[] = [
  { id: 'none', label: 'Local / Personal' },
  { id: 'us', label: 'US' },
  { id: 'eu', label: 'EU' },
  { id: 'uk', label: 'UK' },
  { id: 'global', label: 'Global' },
];

export const StudioDeclarativeBuilder: FC<StudioDeclarativeBuilderProps> = ({
  declaration,
  checkReport,
  routing,
  agenda,
  activeStage,
  onSelectStage,
  onDispatchOp,
  onLoadStarter,
  onImportHoloFile,
  onCheckAndBuild,
  onPublish,
  building,
  apiError,
  onClearApiError,
  modelStreamEnabled,
  onEnableModelStream,
  proposingOps,
  proposedOps,
  onProposeOpsFromPrompt,
  onApplyProposedOps,
  onDismissProposedOps,
}) => {
  const [examplesOpen, setExamplesOpen] = useState(false);

  // Stage 1: Concept / OPM local form state
  const [nameInput, setNameInput] = useState(declaration.metadata.name);
  const [purposeInput, setPurposeInput] = useState(declaration.concept.purpose);
  const [stkName, setStkName] = useState('');
  const [stkRole, setStkRole] = useState<'producer' | 'consumer' | 'beneficiary' | 'regulator'>('consumer');
  const [stkConcern, setStkConcern] = useState('');
  const [objName, setObjName] = useState('');
  const [objRole, setObjRole] = useState<'input' | 'output' | 'stateful-asset'>('input');
  const [objKind, setObjKind] = useState<FieldKind>('nat');
  const [objStates, setObjStates] = useState('');
  const [objUnit, setObjUnit] = useState('');
  const [objMax, setObjMax] = useState('10000');
  const [procName, setProcName] = useState('');
  const [procSummary, setProcSummary] = useState('');

  // Stage 2: System (Fields, Reqs, Rules, Refusals, Examples) state
  const [addingField, setAddingField] = useState(false);
  const [editingFieldName, setEditingFieldName] = useState<string | null>(null);
  const [newFieldName, setNewFieldName] = useState('');
  const [newFieldLabel, setNewFieldLabel] = useState('');
  const [newFieldKind, setNewFieldKind] = useState<FieldKind>('nat');
  const [newFieldMax, setNewFieldMax] = useState('10000');
  const [newFieldDefault, setNewFieldDefault] = useState('0');
  const [newFieldUnit, setNewFieldUnit] = useState('');
  const [newFieldChoices, setNewFieldChoices] = useState('Standard, Pro, Enterprise');
  const [showRawSchema, setShowRawSchema] = useState(false);
  const [rawSchemaDraft, setRawSchemaDraft] = useState('');

  const [addingReq, setAddingReq] = useState(false);
  const [reqStatement, setReqStatement] = useState('');
  const [reqOrigin, setReqOrigin] = useState<RequirementOrigin>('identity');

  const [addingRule, setAddingRule] = useState(false);
  const [editingRuleId, setEditingRuleId] = useState<string | null>(null);
  const [ruleStatement, setRuleStatement] = useState('');
  const [ruleField, setRuleField] = useState(() => declaration.encoding?.fields[0]?.name ?? '');
  const [ruleOp, setRuleOp] = useState<CompareOp>('gte');
  const [ruleRhsMode, setRuleRhsMode] = useState<'const' | 'field'>('const');
  const [ruleRhsField, setRuleRhsField] = useState(
    () => declaration.encoding?.fields[1]?.name ?? declaration.encoding?.fields[0]?.name ?? '',
  );
  const [ruleThreshold, setRuleThreshold] = useState('100');
  const [ruleHasCondition, setRuleHasCondition] = useState(true);
  const [ruleOutcomeKind, setRuleOutcomeKind] = useState<'label' | 'formula'>('label');
  const [ruleOutcomeLabel, setRuleOutcomeLabel] = useState('Approved');
  const [ruleOutcomeTone, setRuleOutcomeTone] = useState<'default' | 'success' | 'warning' | 'destructive' | 'brand'>('success');
  const [ruleFormulaOp, setRuleFormulaOp] = useState<'field' | 'add' | 'sub' | 'mul' | 'div' | 'pct'>('pct');
  const [ruleFormulaRhsMode, setRuleFormulaRhsMode] = useState<'const' | 'field'>('const');
  const [ruleFormulaRight, setRuleFormulaRight] = useState('90');
  const [ruleFormulaRightField, setRuleFormulaRightField] = useState(
    () => declaration.encoding?.fields[1]?.name ?? declaration.encoding?.fields[0]?.name ?? '',
  );
  const [ruleUnit, setRuleUnit] = useState('');

  const [addingRefusal, setAddingRefusal] = useState(false);
  const [editingRefusalId, setEditingRefusalId] = useState<string | null>(null);
  const [refusalField, setRefusalField] = useState(() => declaration.encoding?.fields[0]?.name ?? '');
  const [refusalOp, setRefusalOp] = useState<CompareOp>('eq');
  const [refusalValue, setRefusalValue] = useState('0');
  const [refusalText, setRefusalText] = useState('');

  const [addingVector, setAddingVector] = useState(false);
  const [vecInputs, setVecInputs] = useState<Record<string, number>>({});

  // Stage 3: Security threat edit draft
  const [editingThreatId, setEditingThreatId] = useState<string | null>(null);
  const [threatStatus, setThreatStatus] = useState<Exclude<ThreatDisposition, 'open'>>('mitigated-by-sandbox');
  const [threatRefusalId, setThreatRefusalId] = useState('');
  const [threatRationale, setThreatRationale] = useState('');

  // Stage 6: Owner Decision & NL Proposer
  const [decisionQuote, setDecisionQuote] = useState('');
  const [nlPrompt, setNlPrompt] = useState('');

  const fields = declaration.encoding?.fields ?? [];
  const nextAgenda = agenda[0] ?? null;
  const firstReqId = declaration.requirements[0]?.id ?? 'REQ-01';
  const recAssurance = recommendedAssuranceLevel(declaration.concept);

  useEffect(() => {
    setNameInput(declaration.metadata.name);
  }, [declaration.metadata.name]);

  useEffect(() => {
    setPurposeInput(declaration.concept.purpose);
  }, [declaration.concept.purpose]);

  // Keep field dropdown selections aligned whenever encoding fields change
  useEffect(() => {
    if (fields.length === 0) return;
    const names = new Set(fields.map((f) => f.name));
    const first = fields[0].name;
    const second = fields[1]?.name ?? first;
    if (!names.has(ruleField)) setRuleField(first);
    if (!names.has(ruleRhsField)) setRuleRhsField(second);
    if (!names.has(ruleFormulaRightField)) setRuleFormulaRightField(second);
    if (!names.has(refusalField)) setRefusalField(first);
  }, [fields, refusalField, ruleField, ruleFormulaRightField, ruleRhsField]);

  const previewVecAnswer = useMemo(() => {
    if (!declaration.encoding || !addingVector) return null;
    const line = formatRequestLine(declaration.encoding, vecInputs);
    return { line, answer: evaluateDeclaredModel(declaration, line) };
  }, [addingVector, declaration, vecInputs]);

  const visibleStages = routing.stages.filter((s) => s.visible);
  const stageIcons: Record<StudioWizardStage, typeof Compass> = {
    concept: Compass,
    architecture: Cpu,
    security: Shield,
    design: Palette,
    standards: Scale,
    seal: ShieldCheck,
  };

  // --- Stage 1 Handlers ---
  const handleCommitConceptIdentity = async () => {
    const ops: StudioApiOp[] = [];
    if (nameInput.trim() && nameInput.trim() !== declaration.metadata.name) {
      ops.push({ op: 'set-metadata', name: nameInput.trim() });
    }
    if (purposeInput.trim() !== declaration.concept.purpose) {
      ops.push({ op: 'set-concept-profile', purpose: purposeInput.trim() });
    }
    if (ops.length > 0) {
      await onDispatchOp(ops.length === 1 ? ops[0] : { op: 'batch', ops });
    }
  };

  const handleToggleJurisdiction = async (jur: PrismJurisdiction) => {
    const cur = declaration.concept.jurisdictions;
    let next: PrismJurisdiction[];
    if (jur === 'none') {
      next = ['none'];
    } else if (cur.includes(jur)) {
      next = cur.filter((j) => j !== jur);
    } else {
      next = [...cur.filter((j) => j !== 'none'), jur];
    }
    await onDispatchOp({ op: 'set-concept-profile', jurisdictions: next });
  };

  const handleAddStakeholder = async (e: FormEvent) => {
    e.preventDefault();
    if (!stkName.trim() || !stkConcern.trim()) return;
    const ok = await onDispatchOp({
      op: 'admit-stakeholder',
      stakeholder: { name: stkName.trim(), provRole: stkRole, concern: stkConcern.trim() },
      autoAdmitRequirement: true,
    });
    if (ok) {
      setStkName('');
      setStkConcern('');
    }
  };

  const handleAddOpmObject = async (e: FormEvent) => {
    e.preventDefault();
    if (!objName.trim()) return;
    const states = objStates
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
    const maxVal = Math.max(1, Number.parseInt(objMax, 10) || 10000);
    const ok = await onDispatchOp({
      op: 'admit-opm-object',
      object: {
        name: objName.trim(),
        role: objRole,
        kind: objKind,
        states,
        unit: objUnit.trim() || undefined,
        min: 0,
        max: maxVal,
        defaultValue: objKind === 'nat' ? Math.min(10, maxVal) : 0,
      },
    });
    if (ok) {
      setObjName('');
      setObjStates('');
      setObjUnit('');
    }
  };

  const handleAddOpmProcess = async (e: FormEvent) => {
    e.preventDefault();
    if (!procName.trim()) return;
    const consumes = declaration.concept.opmObjects
      .filter((o) => o.role === 'input' || o.role === 'stateful-asset')
      .map((o) => o.id);
    const yields = declaration.concept.opmObjects
      .filter((o) => o.role === 'output')
      .map((o) => o.id);
    const actors = declaration.concept.stakeholders.map((s) => s.id);
    const ok = await onDispatchOp({
      op: 'admit-opm-process',
      process: {
        name: procName.trim(),
        summary: procSummary.trim() || procName.trim(),
        consumesObjectIds: consumes,
        yieldsObjectIds: yields,
        actorStakeholderIds: actors,
      },
    });
    if (ok) {
      setProcName('');
      setProcSummary('');
    }
  };

  const handleDeriveEncodingAndAdvance = async () => {
    const ok = await onDispatchOp({ op: 'derive-encoding-from-opm' });
    if (ok) onSelectStage('architecture');
  };

  // --- Stage 2 Handlers ---
  const ensureRequirementExists = async (): Promise<string> => {
    if (declaration.requirements.length > 0) return declaration.requirements[0].id;
    await onDispatchOp({
      op: 'admit-requirement',
      requirement: {
        id: 'REQ-01',
        statement:
          declaration.concept.purpose ||
          `Evaluate ${declaration.metadata.name} inputs deterministically.`,
        stakeholder: declaration.concept.stakeholders[0]?.name || 'Creator',
        origin: 'identity',
        evidence: 'System Model declaration',
      },
    });
    return 'REQ-01';
  };

  const handleAddRequirement = async (e: FormEvent) => {
    e.preventDefault();
    if (!reqStatement.trim()) return;
    const nextNum = declaration.requirements.length + 1;
    const id = `REQ-${String(nextNum).padStart(2, '0')}`;
    const ok = await onDispatchOp({
      op: 'admit-requirement',
      requirement: {
        id,
        statement: reqStatement.trim(),
        stakeholder: declaration.concept.stakeholders[0]?.name || 'Creator',
        origin: reqOrigin,
        evidence: `Creator declaration in Verdant Studio`,
      },
    });
    if (ok) {
      setReqStatement('');
      setAddingReq(false);
    }
  };

  const startEditField = (field: EncodedField) => {
    setEditingFieldName(field.name);
    setAddingField(true);
    setNewFieldName(field.name);
    setNewFieldLabel(field.label);
    setNewFieldKind(field.kind);
    setNewFieldMax(String(field.max));
    setNewFieldDefault(String(field.defaultValue));
    setNewFieldUnit(field.unit ?? '');
    setNewFieldChoices(field.choices?.join(', ') ?? 'Standard, Pro, Enterprise');
  };

  const handleSaveField = async (e: FormEvent) => {
    e.preventDefault();
    const label = newFieldLabel.trim() || newFieldName.trim();
    if (!label) return;
    const reqId = await ensureRequirementExists();
    const rawName = (editingFieldName ?? (newFieldName.trim() || label))
      .replace(/[^a-zA-Z0-9]+/g, ' ')
      .trim()
      .split(/\s+/)
      .map((w, i) => (i === 0 ? w.toLowerCase() : w[0].toUpperCase() + w.slice(1).toLowerCase()))
      .join('');
    const choices =
      newFieldKind === 'choice'
        ? newFieldChoices.split(',').map((s) => s.trim()).filter(Boolean)
        : undefined;
    const maxVal =
      newFieldKind === 'choice'
        ? Math.max(0, (choices?.length ?? 2) - 1)
        : newFieldKind === 'bool'
          ? 1
          : Math.max(1, Number.parseInt(newFieldMax, 10) || 10000);
    const defaultVal = Math.max(0, Math.min(maxVal, Number.parseInt(newFieldDefault, 10) || 0));
    const nextField: EncodedField = {
      name: rawName || `field${fields.length + 1}`,
      label,
      kind: newFieldKind,
      min: 0,
      max: maxVal,
      choices,
      unit: newFieldUnit.trim() || undefined,
      defaultValue: defaultVal,
    };
    const nextFields = editingFieldName
      ? fields.map((f) => (f.name === editingFieldName ? nextField : f))
      : [...fields, nextField];
    const ok = await onDispatchOp({
      op: 'propose-encoding',
      name: declaration.encoding?.name ?? 'Input',
      fields: nextFields,
      citation: { kind: 'req', requirementId: reqId },
    });
    if (ok) {
      setNewFieldName('');
      setNewFieldLabel('');
      setNewFieldUnit('');
      setNewFieldDefault('0');
      setEditingFieldName(null);
      setAddingField(false);
    }
  };

  const handleRemoveField = async (fieldName: string) => {
    if (fields.length <= 1) return;
    const reqId = await ensureRequirementExists();
    const remaining = fields.filter((f) => f.name !== fieldName);
    const ops: StudioApiOp[] = [
      {
        op: 'propose-encoding',
        name: declaration.encoding?.name ?? 'Input',
        fields: remaining,
        citation: { kind: 'req', requirementId: reqId },
      },
    ];
    if (declaration.view) {
      ops.push({
        op: 'declare-view',
        view: {
          ...declaration.view,
          elements: declaration.view.elements.filter((el) => el.binding !== fieldName),
        },
      });
    }
    await onDispatchOp({ op: 'batch', ops });
  };

  const handleApplyRawSchema = async () => {
    try {
      const parsed = JSON.parse(rawSchemaDraft) as JsonSchema2020Object;
      const reqId = await ensureRequirementExists();
      await onDispatchOp({
        op: 'propose-encoding',
        name: declaration.encoding?.name ?? 'Input',
        schema: parsed,
        citation: { kind: 'req', requirementId: reqId },
      });
    } catch {
      // Invalid JSON ignored safely
    }
  };

  const nextAppId = () => {
    const used = new Set([
      ...declaration.features.map((f) => f.id),
      ...declaration.refusals.map((r) => r.id),
    ]);
    let n = used.size + 1;
    while (used.has(`APP-${String(n).padStart(2, '0')}`)) n++;
    return `APP-${String(n).padStart(2, '0')}`;
  };

  const startEditRule = (feat: PrismDeclaration['features'][number]) => {
    setEditingRuleId(feat.id);
    setAddingRule(true);
    setRuleStatement(feat.statement);
    if (feat.condition.all.length > 0) {
      const pred = feat.condition.all[0];
      setRuleHasCondition(true);
      setRuleField(pred.field);
      setRuleOp(pred.op);
      if (pred.rhsField) {
        setRuleRhsMode('field');
        setRuleRhsField(pred.rhsField);
      } else {
        setRuleRhsMode('const');
        setRuleThreshold(String(pred.value));
      }
    } else {
      setRuleHasCondition(false);
    }
    if (feat.outcome.kind === 'label') {
      setRuleOutcomeKind('label');
      setRuleOutcomeLabel(feat.outcome.label);
      setRuleOutcomeTone(feat.outcome.tone);
    } else {
      setRuleOutcomeKind('formula');
      setRuleOutcomeLabel(feat.outcome.label);
      setRuleUnit(feat.outcome.unit ?? '');
      const ex = feat.outcome.expr;
      if (ex.op === 'field') {
        setRuleFormulaOp('field');
      } else if (ex.op !== 'const') {
        setRuleFormulaOp(ex.op === 'mod' ? 'div' : ex.op);
        if (ex.right.op === 'const') {
          setRuleFormulaRhsMode('const');
          setRuleFormulaRight(String(ex.right.value));
        } else if (ex.right.op === 'field') {
          setRuleFormulaRhsMode('field');
          setRuleFormulaRightField(ex.right.field);
        }
      }
    }
  };

  const handleSaveRule = async (e: FormEvent) => {
    e.preventDefault();
    const activeField = ruleField || fields[0]?.name;
    if (!activeField) return;
    const reqId = await ensureRequirementExists();
    const threshold = Math.max(0, Number.parseInt(ruleThreshold, 10) || 0);
    const rightVal = Math.max(0, Number.parseInt(ruleFormulaRight, 10) || 0);
    const rhsFieldName = ruleRhsField || fields[0]?.name || activeField;
    const formulaRightField = ruleFormulaRightField || fields[0]?.name || activeField;

    const condition = ruleHasCondition
      ? {
          all: [
            {
              field: activeField,
              op: ruleOp,
              value: threshold,
              ...(ruleRhsMode === 'field' ? { rhsField: rhsFieldName } : {}),
            },
          ],
        }
      : { all: [] };

    const outcome =
      ruleOutcomeKind === 'label'
        ? {
            kind: 'label' as const,
            code: declaration.features.length + 1,
            label: ruleOutcomeLabel.trim() || 'Matched',
            tone: ruleOutcomeTone,
          }
        : {
            kind: 'formula' as const,
            label: ruleOutcomeLabel.trim() || 'Result',
            unit: ruleUnit.trim() || undefined,
            expr:
              ruleFormulaOp === 'field'
                ? ({ op: 'field', field: activeField } as const)
                : ({
                    op: ruleFormulaOp,
                    left: { op: 'field', field: activeField },
                    right:
                      ruleFormulaRhsMode === 'field'
                        ? ({ op: 'field', field: formulaRightField } as const)
                        : ({ op: 'const', value: rightVal } as const),
                  } as const),
          };

    const statement =
      ruleStatement.trim() ||
      (ruleHasCondition
        ? `When ${activeField} ${COMPARE_LABELS[ruleOp]} ${ruleRhsMode === 'field' ? rhsFieldName : threshold}, return ${ruleOutcomeLabel || 'result'}.`
        : `Default rule returning ${ruleOutcomeLabel || 'result'}.`);

    const ok = editingRuleId
      ? await onDispatchOp({
          op: 'update-feature',
          id: editingRuleId,
          patch: { statement, condition, outcome },
        })
      : await onDispatchOp({
          op: 'admit-feature',
          feature: {
            id: nextAppId(),
            level: 'build',
            statement,
            condition,
            outcome,
            priority: declaration.features.length + 1,
            citation: { kind: 'req', requirementId: reqId },
          },
        });
    if (ok) {
      setRuleStatement('');
      setEditingRuleId(null);
      setAddingRule(false);
    }
  };

  const handleMoveFeature = async (id: string, delta: -1 | 1) => {
    const ordered = [...declaration.features].sort(
      (a, b) => a.priority - b.priority || a.id.localeCompare(b.id),
    );
    const idx = ordered.findIndex((f) => f.id === id);
    const target = idx + delta;
    if (idx < 0 || target < 0 || target >= ordered.length) return;
    const ids = ordered.map((f) => f.id);
    [ids[idx], ids[target]] = [ids[target], ids[idx]];
    await onDispatchOp({ op: 'reorder-features', orderedIds: ids });
  };

  const startEditRefusal = (ref: PrismDeclaration['refusals'][number]) => {
    setEditingRefusalId(ref.id);
    setAddingRefusal(true);
    const pred = ref.condition.all[0];
    if (pred) {
      setRefusalField(pred.field);
      setRefusalOp(pred.op);
      setRefusalValue(String(pred.value));
    }
    setRefusalText(ref.failureText);
  };

  const handleSaveRefusal = async (e: FormEvent) => {
    e.preventDefault();
    const activeField = refusalField || fields[0]?.name;
    if (!activeField || !refusalText.trim()) return;
    const reqId = await ensureRequirementExists();
    const val = Math.max(0, Number.parseInt(refusalValue, 10) || 0);
    const condition = { all: [{ field: activeField, op: refusalOp, value: val }] };
    const statement = `Refuse when ${activeField} ${COMPARE_LABELS[refusalOp]} ${val}.`;

    if (editingRefusalId) {
      const ok = await onDispatchOp({
        op: 'update-refusal',
        id: editingRefusalId,
        patch: { statement, condition, failureText: refusalText.trim(), fieldHint: activeField },
      });
      if (ok) {
        setRefusalText('');
        setEditingRefusalId(null);
        setAddingRefusal(false);
      }
      return;
    }

    const id = nextAppId();
    const vecIn: Record<string, number> = {};
    for (const f of fields) vecIn[f.name] = f.defaultValue;
    vecIn[activeField] = refusalOp === 'lt' ? Math.max(0, val - 1) : refusalOp === 'gt' ? val + 1 : val;

    const ok = await onDispatchOp({
      op: 'batch',
      ops: [
        {
          op: 'admit-refusal',
          refusal: {
            id,
            statement,
            condition,
            failureText: refusalText.trim(),
            fieldHint: activeField,
            negativeId: id,
            citation: { kind: 'req', requirementId: reqId },
          },
        },
        {
          op: 'admit-vector',
          vector: {
            id: `vec_${id.toLowerCase().replace(/[^a-z0-9]+/g, '_')}`,
            inputs: vecIn,
            expectedResponse: refusalText.trim(),
            forFeatureId: id,
            isRefusal: true,
          },
        },
      ],
    });
    if (ok) {
      setRefusalText('');
      setAddingRefusal(false);
    }
  };

  const handleAddVector = async (e: FormEvent) => {
    e.preventDefault();
    if (!declaration.encoding || !previewVecAnswer) return;
    const ok = await onDispatchOp({
      op: 'admit-vector',
      vector: {
        id: `vec_${declaration.vectors.length + 1}`,
        inputs: { ...vecInputs },
        expectedResponse: previewVecAnswer.answer.response,
        forFeatureId: previewVecAnswer.answer.matchedId ?? declaration.features[0]?.id ?? 'APP-01',
        isRefusal: previewVecAnswer.answer.isRefusal,
      },
    });
    if (ok) setAddingVector(false);
  };

  const handleSettleThreatSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!editingThreatId || !threatRationale.trim()) return;
    const ok = await onDispatchOp({
      op: 'settle-stride-threat',
      id: editingThreatId,
      status: threatStatus,
      linkedRefusalId: threatStatus === 'mitigated-by-refusal' ? threatRefusalId || declaration.refusals[0]?.id : undefined,
      rationale: threatRationale.trim(),
    });
    if (ok) {
      setEditingThreatId(null);
      setThreatRationale('');
    }
  };

  const handleUpdateViewField = async (patch: Partial<NonNullable<PrismDeclaration['view']>>) => {
    if (!declaration.view) return;
    await onDispatchOp(
      { op: 'declare-view', view: { ...declaration.view, ...patch } },
      { pushRevision: false },
    );
  };

  const handleRecordDecision = async (e: FormEvent) => {
    e.preventDefault();
    if (!decisionQuote.trim()) return;
    const ok = await onDispatchOp({
      op: 'record-decision',
      stakeholder: declaration.concept.stakeholders[0]?.name || 'Creator',
      quote: decisionQuote.trim(),
      requirementIds: declaration.requirements.map((r) => r.id),
    });
    if (ok) setDecisionQuote('');
  };

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-col bg-white dark:bg-[#0E131B]">
      {/* Compact Wizard Stage Rail + Quiet Examples Popover (REQ-CONTENT-BOUNDARY) */}
      <div className="relative flex shrink-0 items-center justify-between gap-1 border-b border-zinc-200 px-2 py-1.5 dark:border-zinc-800">
        <div className="flex min-w-0 flex-1 flex-wrap items-center gap-1">
          {visibleStages.map((st) => {
            const Icon = stageIcons[st.stage];
            const active = activeStage === st.stage;
            return (
              <button
                key={st.stage}
                type="button"
                onClick={() => onSelectStage(st.stage)}
                className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium transition-colors ${
                  active
                    ? 'bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900'
                    : 'text-zinc-600 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-800'
                }`}
                title={`${st.number}. ${st.label} (${st.standardsLabel})`}
              >
                <Icon className="h-3 w-3 shrink-0" />
                <span>{st.shortLabel}</span>
                {st.status === 'complete' ? (
                  <CheckCircle2
                    className={`h-2.5 w-2.5 shrink-0 ${active ? 'text-emerald-300 dark:text-emerald-700' : 'text-emerald-600'}`}
                  />
                ) : st.status === 'in-progress' ? (
                  <Circle
                    className={`h-2 w-2 shrink-0 fill-amber-500 text-amber-500`}
                  />
                ) : null}
              </button>
            );
          })}
        </div>

        {/* Quiet Examples Popover (`REQ-CONTENT-BOUNDARY`: never takes over the workspace) */}
        <div className="relative shrink-0">
          <button
            type="button"
            onClick={() => setExamplesOpen((v) => !v)}
            className="inline-flex items-center gap-1 rounded-md border border-zinc-200 px-2 py-1 text-[11px] font-medium text-zinc-600 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
            title="Load an optional reference example or import a .holo file"
          >
            <BookOpen className="h-3 w-3" />
            <span>Examples</span>
          </button>
          {examplesOpen && (
            <div className="absolute right-0 top-full z-30 mt-1 w-72 rounded-xl border border-zinc-200 bg-white p-2 shadow-xl dark:border-zinc-700 dark:bg-[#11161F]">
              <p className="px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
                Optional Reference Examples
              </p>
              <div className="space-y-1">
                {DECLARATIVE_STARTERS.map((s) => (
                  <button
                    key={s.id}
                    type="button"
                    onClick={() => {
                      setExamplesOpen(false);
                      onLoadStarter(s.id);
                    }}
                    className="flex w-full flex-col rounded-lg px-2 py-1.5 text-left hover:bg-zinc-100 dark:hover:bg-zinc-800"
                  >
                    <span className="text-xs font-medium text-zinc-900 dark:text-zinc-100">{s.label}</span>
                    <span className="line-clamp-2 text-[10px] text-zinc-500">{s.description}</span>
                  </button>
                ))}
              </div>
              <div className="mt-1.5 border-t border-zinc-200 pt-1.5 dark:border-zinc-800">
                <button
                  type="button"
                  onClick={() => {
                    setExamplesOpen(false);
                    onImportHoloFile();
                  }}
                  className="flex w-full items-center gap-1.5 rounded-lg px-2 py-1.5 text-xs font-medium text-emerald-700 hover:bg-emerald-50 dark:text-emerald-300 dark:hover:bg-emerald-950/40"
                >
                  <Upload className="h-3.5 w-3.5" />
                  Import verified .holo archive…
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* API Error / Refusal Banner */}
      {apiError && (
        <div className="flex shrink-0 items-start justify-between gap-2 border-b border-red-200 bg-red-50 px-3 py-2 text-xs text-red-900 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-200">
          <div className="min-w-0">
            <p className="font-semibold">{apiError.message}</p>
            <p className="mt-0.5 text-[11px] opacity-85">{apiError.remediation}</p>
          </div>
          <button type="button" onClick={onClearApiError} className="shrink-0 rounded p-0.5 hover:bg-black/5">
            <XCircle className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      {/* Derived Agenda / Self-Healing Repair Strip */}
      {nextAgenda && !routing.isUndefined && (
        <div className="flex shrink-0 items-center justify-between gap-2 border-b border-amber-200/80 bg-amber-50/70 px-3 py-1.5 text-[11px] text-amber-950 dark:border-amber-900/50 dark:bg-amber-950/30 dark:text-amber-200">
          <div className="flex min-w-0 items-center gap-1.5">
            <Wrench className="h-3.5 w-3.5 shrink-0 text-amber-600 dark:text-amber-400" />
            <span className="truncate font-medium">{nextAgenda.title}</span>
          </div>
          <button
            type="button"
            onClick={() => {
              if (nextAgenda.suggestedOp) {
                void onDispatchOp(nextAgenda.suggestedOp);
              } else {
                const mapStage: Record<string, StudioWizardStage> = {
                  concept: 'concept',
                  architecture: 'architecture',
                  security: 'security',
                  design: 'design',
                  standards: 'standards',
                  seal: 'seal',
                  features: 'architecture',
                  data: 'architecture',
                  capabilities: 'seal',
                };
                onSelectStage(mapStage[nextAgenda.targetTab] ?? 'architecture');
              }
            }}
            className="shrink-0 rounded bg-amber-900 px-2 py-0.5 text-[10px] font-semibold text-white hover:bg-amber-800 dark:bg-amber-200 dark:text-amber-950"
          >
            {nextAgenda.actionLabel}
          </button>
        </div>
      )}

      {/* Scrollable Active Wizard Stage Body */}
      <div className="min-h-0 flex-1 overflow-y-auto p-3 text-xs">
        {/* ================================================================ */}
        {/* STAGE 1: CONCEPT & OPM (ISO/PAS 19450 + PROV-O + Context Weights) */}
        {/* ================================================================ */}
        {activeStage === 'concept' && (
          <div className="space-y-4">
            <div className="rounded-lg border border-emerald-200/80 bg-emerald-50/50 p-3 dark:border-emerald-900/50 dark:bg-emerald-950/20">
              <p className="text-[11px] font-semibold text-emerald-900 dark:text-emerald-200">
                Stage 1 · Conceptual Model (ISO/PAS 19450 OPM &amp; W3C PROV-O)
              </p>
              <p className="mt-0.5 text-[11px] text-emerald-800/80 dark:text-emerald-300/80">
                Define what this Prism does, who it serves, and its core Objects &amp; Processes. Populating these weights routes applicable standards and seeds Stage 2 automatically.
              </p>
            </div>

            {/* Identity & Purpose */}
            <section className="space-y-2 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                1. Identity &amp; Purpose
              </h4>
              <div className="space-y-2">
                <label className="block">
                  <span className="text-[11px] font-medium text-zinc-600 dark:text-zinc-300">Prism Name</span>
                  <input
                    type="text"
                    value={nameInput}
                    onChange={(e) => setNameInput(e.target.value)}
                    onBlur={() => void handleCommitConceptIdentity()}
                    placeholder="e.g. Monthly Budget Guard"
                    className="mt-0.5 w-full rounded-md border border-zinc-300 bg-transparent px-2.5 py-1.5 text-xs dark:border-zinc-700"
                  />
                </label>
                <label className="block">
                  <span className="text-[11px] font-medium text-zinc-600 dark:text-zinc-300">
                    Problem Statement / Purpose
                  </span>
                  <textarea
                    rows={2}
                    value={purposeInput}
                    onChange={(e) => setPurposeInput(e.target.value)}
                    onBlur={() => void handleCommitConceptIdentity()}
                    placeholder="What decision, calculation, or invariant does this Prism guarantee?"
                    className="mt-0.5 w-full rounded-md border border-zinc-300 bg-transparent px-2.5 py-1.5 text-xs dark:border-zinc-700"
                  />
                </label>
              </div>
            </section>

            {/* Context & Standards Routing Weights */}
            <section className="space-y-3 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <div>
                <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                  2. Context Weights (Routes Applicable Standards)
                </h4>
                <p className="mt-0.5 text-[10px] text-zinc-500">
                  Only standards triggered by these facts enter your workflow; irrelevant compliance layers stay excluded.
                </p>
              </div>

              <div>
                <p className="mb-1 text-[11px] font-medium text-zinc-600 dark:text-zinc-300">Domain</p>
                <div className="flex flex-wrap gap-1.5">
                  {DOMAIN_OPTIONS.map((opt) => (
                    <button
                      key={opt.id}
                      type="button"
                      onClick={() => void onDispatchOp({ op: 'set-concept-profile', domain: opt.id })}
                      className={`rounded-md border px-2 py-1 text-[11px] font-medium transition-colors ${
                        declaration.concept.domain === opt.id
                          ? 'border-emerald-600 bg-emerald-600 text-white'
                          : 'border-zinc-200 text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800'
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <p className="mb-1 text-[11px] font-medium text-zinc-600 dark:text-zinc-300">Target Audience</p>
                <div className="flex flex-wrap gap-1.5">
                  {AUDIENCE_OPTIONS.map((opt) => (
                    <button
                      key={opt.id}
                      type="button"
                      onClick={() => void onDispatchOp({ op: 'set-concept-profile', audience: opt.id })}
                      className={`rounded-md border px-2 py-1 text-[11px] font-medium transition-colors ${
                        declaration.concept.audience === opt.id
                          ? 'border-emerald-600 bg-emerald-600 text-white'
                          : 'border-zinc-200 text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800'
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <p className="mb-1 text-[11px] font-medium text-zinc-600 dark:text-zinc-300">Data Classification</p>
                <div className="flex flex-wrap gap-1.5">
                  {DATA_OPTIONS.map((opt) => (
                    <button
                      key={opt.id}
                      type="button"
                      onClick={() => void onDispatchOp({ op: 'set-concept-profile', dataClassification: opt.id })}
                      className={`rounded-md border px-2 py-1 text-[11px] font-medium transition-colors ${
                        declaration.concept.dataClassification === opt.id
                          ? 'border-emerald-600 bg-emerald-600 text-white'
                          : 'border-zinc-200 text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800'
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <p className="mb-1 text-[11px] font-medium text-zinc-600 dark:text-zinc-300">Jurisdiction Scope</p>
                <div className="flex flex-wrap gap-1.5">
                  {JURISDICTION_OPTIONS.map((opt) => {
                    const selected = declaration.concept.jurisdictions.includes(opt.id);
                    return (
                      <button
                        key={opt.id}
                        type="button"
                        onClick={() => void handleToggleJurisdiction(opt.id)}
                        className={`rounded-md border px-2 py-1 text-[11px] font-medium transition-colors ${
                          selected
                            ? 'border-zinc-900 bg-zinc-900 text-white dark:border-zinc-100 dark:bg-zinc-100 dark:text-zinc-900'
                            : 'border-zinc-200 text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800'
                        }`}
                      >
                        {opt.label}
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 pt-1">
                <label className="flex cursor-pointer items-center justify-between rounded-md border border-zinc-200 px-2.5 py-1.5 dark:border-zinc-800">
                  <span className="text-[11px] font-medium">Uses Local AI</span>
                  <input
                    type="checkbox"
                    checked={declaration.concept.usesLocalAi}
                    onChange={(e) =>
                      void onDispatchOp({ op: 'set-concept-profile', usesLocalAi: e.target.checked })
                    }
                    className="h-3.5 w-3.5 accent-emerald-600"
                  />
                </label>
                <label className="flex cursor-pointer items-center justify-between rounded-md border border-zinc-200 px-2.5 py-1.5 dark:border-zinc-800">
                  <span className="text-[11px] font-medium">Involves Payments</span>
                  <input
                    type="checkbox"
                    checked={declaration.concept.involvesPayments}
                    onChange={(e) =>
                      void onDispatchOp({ op: 'set-concept-profile', involvesPayments: e.target.checked })
                    }
                    className="h-3.5 w-3.5 accent-emerald-600"
                  />
                </label>
              </div>
            </section>

            {/* PROV-O Stakeholders */}
            <section className="space-y-2 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                3. Stakeholders (W3C PROV-O) · {declaration.concept.stakeholders.length}
              </h4>
              {declaration.concept.stakeholders.map((stk) => (
                <div
                  key={stk.id}
                  className="flex items-start justify-between gap-2 rounded-md bg-zinc-50 px-2.5 py-1.5 dark:bg-zinc-900/60"
                >
                  <div className="min-w-0">
                    <span className="font-semibold text-zinc-900 dark:text-zinc-100">{stk.name}</span>{' '}
                    <span className="rounded bg-zinc-200/80 px-1 py-0.5 text-[9px] uppercase text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400">
                      {stk.provRole}
                    </span>
                    <p className="text-[11px] text-zinc-500">{stk.concern}</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => void onDispatchOp({ op: 'remove-stakeholder', id: stk.id })}
                    className="shrink-0 text-zinc-400 hover:text-red-500"
                    title="Remove stakeholder"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))}
              <form onSubmit={(e) => void handleAddStakeholder(e)} className="space-y-1.5 pt-1">
                <div className="grid grid-cols-2 gap-1.5">
                  <input
                    type="text"
                    value={stkName}
                    onChange={(e) => setStkName(e.target.value)}
                    placeholder="Role (e.g. Budget Owner)"
                    className="rounded border border-zinc-300 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                  />
                  <select
                    value={stkRole}
                    onChange={(e) => setStkRole(e.target.value as typeof stkRole)}
                    className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                  >
                    <option value="consumer">Consumer (Uses Prism)</option>
                    <option value="producer">Producer (Authors Rules)</option>
                    <option value="beneficiary">Beneficiary</option>
                    <option value="regulator">Regulator / Authority</option>
                  </select>
                </div>
                <div className="flex gap-1.5">
                  <input
                    type="text"
                    value={stkConcern}
                    onChange={(e) => setStkConcern(e.target.value)}
                    placeholder="Primary concern / requirement…"
                    className="flex-1 rounded border border-zinc-300 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                  />
                  <button
                    type="submit"
                    className="inline-flex items-center gap-1 rounded bg-zinc-900 px-2.5 py-1 text-xs font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900"
                  >
                    <Plus className="h-3 w-3" />
                    Add
                  </button>
                </div>
              </form>
            </section>

            {/* OPM Objects */}
            <section className="space-y-2 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                4. OPM Objects &amp; States (ISO 19450) · {declaration.concept.opmObjects.length}
              </h4>
              {declaration.concept.opmObjects.map((obj) => (
                <div
                  key={obj.id}
                  className="flex items-start justify-between gap-2 rounded-md bg-zinc-50 px-2.5 py-1.5 dark:bg-zinc-900/60"
                >
                  <div className="min-w-0">
                    <span className="font-semibold text-zinc-900 dark:text-zinc-100">{obj.name}</span>{' '}
                    <span className="rounded bg-emerald-100 px-1 py-0.5 text-[9px] font-semibold uppercase text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300">
                      {obj.role} · {obj.kind}
                    </span>
                    {obj.states.length > 0 && (
                      <p className="text-[10px] text-zinc-500">States: {obj.states.join(', ')}</p>
                    )}
                  </div>
                  <button
                    type="button"
                    onClick={() => void onDispatchOp({ op: 'remove-opm-object', id: obj.id })}
                    className="shrink-0 text-zinc-400 hover:text-red-500"
                    title="Remove OPM object"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))}
              <form onSubmit={(e) => void handleAddOpmObject(e)} className="space-y-1.5 pt-1">
                <div className="grid grid-cols-3 gap-1.5">
                  <input
                    type="text"
                    value={objName}
                    onChange={(e) => setObjName(e.target.value)}
                    placeholder="Object name"
                    className="col-span-2 rounded border border-zinc-300 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                  />
                  <select
                    value={objRole}
                    onChange={(e) => setObjRole(e.target.value as typeof objRole)}
                    className="rounded border border-zinc-300 bg-white px-1.5 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                  >
                    <option value="input">Input</option>
                    <option value="output">Output</option>
                    <option value="stateful-asset">Stateful</option>
                  </select>
                </div>
                <div className="grid grid-cols-3 gap-1.5">
                  <select
                    value={objKind}
                    onChange={(e) => setObjKind(e.target.value as FieldKind)}
                    className="rounded border border-zinc-300 bg-white px-1.5 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                  >
                    <option value="nat">Number</option>
                    <option value="choice">Choice / States</option>
                    <option value="bool">Yes / No</option>
                  </select>
                  {objKind === 'choice' ? (
                    <input
                      type="text"
                      value={objStates}
                      onChange={(e) => setObjStates(e.target.value)}
                      placeholder="States: A, B, C"
                      className="col-span-2 rounded border border-zinc-300 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                    />
                  ) : (
                    <>
                      <input
                        type="text"
                        value={objUnit}
                        onChange={(e) => setObjUnit(e.target.value)}
                        placeholder="Unit (e.g. USD)"
                        className="rounded border border-zinc-300 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                      />
                      <input
                        type="number"
                        value={objMax}
                        onChange={(e) => setObjMax(e.target.value)}
                        placeholder="Max"
                        className="rounded border border-zinc-300 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                      />
                    </>
                  )}
                </div>
                <button
                  type="submit"
                  className="inline-flex items-center gap-1 rounded bg-zinc-900 px-2.5 py-1 text-xs font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900"
                >
                  <Plus className="h-3 w-3" />
                  Add OPM Object
                </button>
              </form>
            </section>

            {/* OPM Processes */}
            <section className="space-y-2 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                5. OPM Processes (ISO 19450) · {declaration.concept.opmProcesses.length}
              </h4>
              {declaration.concept.opmProcesses.map((proc) => (
                <div
                  key={proc.id}
                  className="flex items-start justify-between gap-2 rounded-md bg-zinc-50 px-2.5 py-1.5 dark:bg-zinc-900/60"
                >
                  <div className="min-w-0">
                    <span className="font-semibold text-zinc-900 dark:text-zinc-100">{proc.name}</span>
                    <p className="text-[11px] text-zinc-500">{proc.summary}</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => void onDispatchOp({ op: 'remove-opm-process', id: proc.id })}
                    className="shrink-0 text-zinc-400 hover:text-red-500"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))}
              <form onSubmit={(e) => void handleAddOpmProcess(e)} className="space-y-1.5 pt-1">
                <input
                  type="text"
                  value={procName}
                  onChange={(e) => setProcName(e.target.value)}
                  placeholder="Process name (e.g. Evaluate Budget)"
                  className="w-full rounded border border-zinc-300 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                />
                <div className="flex gap-1.5">
                  <input
                    type="text"
                    value={procSummary}
                    onChange={(e) => setProcSummary(e.target.value)}
                    placeholder="How inputs transform into outputs…"
                    className="flex-1 rounded border border-zinc-300 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                  />
                  <button
                    type="submit"
                    className="inline-flex items-center gap-1 rounded bg-zinc-900 px-2.5 py-1 text-xs font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900"
                  >
                    <Plus className="h-3 w-3" />
                    Add
                  </button>
                </div>
              </form>
            </section>

            {/* Stage 1 -> Stage 2 Cross-Facet Propagation Footer */}
            <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
              <button
                type="button"
                disabled={
                  !declaration.concept.opmObjects.some(
                    (o) => o.role === 'input' || o.role === 'stateful-asset',
                  )
                }
                onClick={() => void handleDeriveEncodingAndAdvance()}
                className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-700 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-800 disabled:opacity-45"
              >
                <Sparkles className="h-3.5 w-3.5" />
                Derive System Fields from OPM &amp; Continue
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
              <button
                type="button"
                onClick={() => onSelectStage('architecture')}
                className="inline-flex items-center gap-1 rounded-lg border border-zinc-200 px-2.5 py-1.5 text-xs font-medium text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
              >
                Next: System
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        )}

        {/* ================================================================ */}
        {/* STAGE 2: ARCHITECTURE & SYSTEM (ISO 42010 + UML + JSON Schema)    */}
        {/* ================================================================ */}
        {activeStage === 'architecture' && (
          <div className="space-y-4">
            {/* System Boundary & Deny-by-Default Capabilities */}
            <section className="space-y-2 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <div className="flex items-center justify-between">
                <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                  Runtime System Boundary (Deny by Default)
                </h4>
              </div>
              <div className="grid grid-cols-3 gap-1.5">
                {(
                  [
                    ['storage', 'Local Storage'],
                    ['browser', 'Browser Utilities'],
                    ['network', 'Internet Access'],
                  ] as const
                ).map(([key, label]) => (
                  <label
                    key={key}
                    className="flex cursor-pointer items-center justify-between rounded border border-zinc-200 px-2 py-1.5 text-[11px] dark:border-zinc-800"
                  >
                    <span>{label}</span>
                    <input
                      type="checkbox"
                      checked={declaration.capabilities[key]}
                      onChange={(e) =>
                        void onDispatchOp({
                          op: 'set-capabilities',
                          capabilities: { [key]: e.target.checked },
                        })
                      }
                      className="h-3.5 w-3.5 accent-emerald-600"
                    />
                  </label>
                ))}
              </div>
            </section>

            {/* Typed Input Fields (JSON Schema 2020-12) */}
            <section className="space-y-2 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <div className="flex flex-wrap items-center justify-between gap-1">
                <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                  Input Encoding ({fields.length} fields)
                </h4>
                <div className="flex items-center gap-1">
                  {declaration.concept.opmObjects.some(
                    (o) => o.role === 'input' || o.role === 'stateful-asset',
                  ) && (
                    <button
                      type="button"
                      onClick={() => void onDispatchOp({ op: 'derive-encoding-from-opm' })}
                      className="rounded border border-emerald-300 bg-emerald-50 px-2 py-0.5 text-[10px] font-medium text-emerald-800 hover:bg-emerald-100 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300"
                    >
                      Sync from OPM
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => {
                      setEditingFieldName(null);
                      setAddingField((v) => !v);
                    }}
                    className="inline-flex items-center gap-1 rounded bg-zinc-900 px-2 py-0.5 text-[10px] font-medium text-white dark:bg-zinc-100 dark:text-zinc-900"
                  >
                    <Plus className="h-3 w-3" />
                    Field
                  </button>
                </div>
              </div>

              {fields.length === 0 ? (
                <p className="rounded border border-dashed border-zinc-300 p-2.5 text-[11px] text-zinc-500 dark:border-zinc-700">
                  No input fields declared yet. Click <strong>Sync from OPM</strong> or <strong>+ Field</strong> to define your JSON Schema 2020-12 input boundary.
                </p>
              ) : (
                <div className="space-y-1.5">
                  {fields.map((f) => (
                    <div
                      key={f.name}
                      className="flex items-center justify-between gap-2 rounded-md bg-zinc-50 px-2.5 py-1.5 dark:bg-zinc-900/60"
                    >
                      <div className="min-w-0">
                        <span className="font-semibold text-zinc-900 dark:text-zinc-100">{f.label}</span>{' '}
                        <code className="text-[10px] text-zinc-500">{f.name}</code>
                        <p className="text-[10px] text-zinc-500">
                          {f.kind === 'choice'
                            ? `Choices: ${f.choices?.join(', ')}`
                            : f.kind === 'bool'
                              ? 'Toggle (Yes/No)'
                              : `Integer ${f.min}..${f.max}${f.unit ? ` ${f.unit}` : ''} (default ${f.defaultValue})`}
                        </p>
                      </div>
                      <div className="flex shrink-0 items-center gap-1">
                        <button
                          type="button"
                          onClick={() => startEditField(f)}
                          className="rounded p-1 text-zinc-400 hover:bg-zinc-200/70 hover:text-zinc-800 dark:hover:bg-zinc-800"
                          title="Edit field"
                        >
                          <Pencil className="h-3 w-3" />
                        </button>
                        {fields.length > 1 && (
                          <button
                            type="button"
                            onClick={() => void handleRemoveField(f.name)}
                            className="rounded p-1 text-zinc-400 hover:text-red-500"
                            title="Remove field"
                          >
                            <Trash2 className="h-3 w-3" />
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {addingField && (
                <form onSubmit={(e) => void handleSaveField(e)} className="space-y-2 rounded-lg border border-zinc-200 p-2.5 dark:border-zinc-700">
                  <div className="grid grid-cols-2 gap-1.5">
                    <input
                      type="text"
                      value={newFieldLabel}
                      onChange={(e) => setNewFieldLabel(e.target.value)}
                      placeholder="Field label (e.g. Amount)"
                      className="rounded border border-zinc-300 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                    />
                    <select
                      value={newFieldKind}
                      onChange={(e) => setNewFieldKind(e.target.value as FieldKind)}
                      className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                    >
                      <option value="nat">Whole Number</option>
                      <option value="choice">Choice Options</option>
                      <option value="bool">Yes / No Toggle</option>
                    </select>
                  </div>
                  {newFieldKind === 'choice' ? (
                    <input
                      type="text"
                      value={newFieldChoices}
                      onChange={(e) => setNewFieldChoices(e.target.value)}
                      placeholder="Comma-separated options"
                      className="w-full rounded border border-zinc-300 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                    />
                  ) : newFieldKind === 'nat' ? (
                    <div className="grid grid-cols-3 gap-1.5">
                      <input
                        type="number"
                        value={newFieldMax}
                        onChange={(e) => setNewFieldMax(e.target.value)}
                        placeholder="Max"
                        className="rounded border border-zinc-300 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                      />
                      <input
                        type="number"
                        value={newFieldDefault}
                        onChange={(e) => setNewFieldDefault(e.target.value)}
                        placeholder="Default"
                        className="rounded border border-zinc-300 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                      />
                      <input
                        type="text"
                        value={newFieldUnit}
                        onChange={(e) => setNewFieldUnit(e.target.value)}
                        placeholder="Unit"
                        className="rounded border border-zinc-300 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                      />
                    </div>
                  ) : null}
                  <div className="flex justify-end gap-1.5">
                    <button
                      type="button"
                      onClick={() => {
                        setAddingField(false);
                        setEditingFieldName(null);
                      }}
                      className="rounded px-2 py-1 text-xs text-zinc-500 hover:bg-zinc-100 dark:hover:bg-zinc-800"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      className="rounded bg-emerald-700 px-2.5 py-1 text-xs font-medium text-white hover:bg-emerald-800"
                    >
                      {editingFieldName ? 'Save Field' : 'Admit Field'}
                    </button>
                  </div>
                </form>
              )}

              <div>
                <button
                  type="button"
                  onClick={() => {
                    setRawSchemaDraft(JSON.stringify(declaration.encoding?.schema ?? { type: 'object', properties: {} }, null, 2));
                    setShowRawSchema((v) => !v);
                  }}
                  className="text-[10px] font-medium text-zinc-500 underline-offset-2 hover:underline"
                >
                  {showRawSchema ? 'Hide JSON Schema 2020-12' : 'Inspect / Edit Raw JSON Schema 2020-12'}
                </button>
                {showRawSchema && (
                  <div className="mt-1.5 space-y-1.5">
                    <textarea
                      rows={6}
                      value={rawSchemaDraft}
                      onChange={(e) => setRawSchemaDraft(e.target.value)}
                      className="w-full rounded border border-zinc-300 bg-zinc-50 p-2 font-mono text-[10px] dark:border-zinc-700 dark:bg-zinc-950"
                    />
                    <button
                      type="button"
                      onClick={() => void handleApplyRawSchema()}
                      className="rounded bg-zinc-900 px-2.5 py-1 text-[11px] font-medium text-white dark:bg-zinc-100 dark:text-zinc-900"
                    >
                      Apply Schema
                    </button>
                  </div>
                )}
              </div>
            </section>

            {/* Requirements (ISO 29148) */}
            <section className="space-y-2 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <div className="flex items-center justify-between">
                <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                  Requirements ({declaration.requirements.length})
                </h4>
                <button
                  type="button"
                  onClick={() => setAddingReq((v) => !v)}
                  className="inline-flex items-center gap-1 rounded bg-zinc-900 px-2 py-0.5 text-[10px] font-medium text-white dark:bg-zinc-100 dark:text-zinc-900"
                >
                  <Plus className="h-3 w-3" />
                  Requirement
                </button>
              </div>
              {declaration.requirements.map((r) => (
                <div key={r.id} className="rounded bg-zinc-50 px-2.5 py-1.5 dark:bg-zinc-900/60">
                  <div className="flex items-center justify-between gap-1">
                    <code className="font-mono text-[10px] font-semibold text-emerald-700 dark:text-emerald-400">
                      {r.id}
                    </code>
                    <span className="rounded bg-zinc-200/70 px-1.5 py-0.5 text-[9px] uppercase text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400">
                      {r.origin}
                    </span>
                  </div>
                  <p className="mt-0.5 text-[11px] text-zinc-800 dark:text-zinc-200">{r.statement}</p>
                </div>
              ))}
              {addingReq && (
                <form onSubmit={(e) => void handleAddRequirement(e)} className="space-y-1.5 pt-1">
                  <input
                    type="text"
                    value={reqStatement}
                    onChange={(e) => setReqStatement(e.target.value)}
                    placeholder="Requirement statement…"
                    className="w-full rounded border border-zinc-300 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                  />
                  <div className="flex items-center justify-between gap-1.5">
                    <select
                      value={reqOrigin}
                      onChange={(e) => setReqOrigin(e.target.value as RequirementOrigin)}
                      className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                    >
                      <option value="identity">Identity (Creator)</option>
                      <option value="constraint">Constraint (Policy / Standard)</option>
                      <option value="refinement">Refinement (Feedback)</option>
                    </select>
                    <button
                      type="submit"
                      className="rounded bg-emerald-700 px-2.5 py-1 text-xs font-medium text-white"
                    >
                      Admit
                    </button>
                  </div>
                </form>
              )}
            </section>

            {/* Deterministic Rules (UML Behavior -> LexLean) */}
            <section className="space-y-2 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <div className="flex items-center justify-between">
                <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                  Decision Rules &amp; Formulas ({declaration.features.length})
                </h4>
                <button
                  type="button"
                  disabled={fields.length === 0}
                  onClick={() => {
                    setEditingRuleId(null);
                    setAddingRule((v) => !v);
                  }}
                  className="inline-flex items-center gap-1 rounded bg-zinc-900 px-2 py-0.5 text-[10px] font-medium text-white disabled:opacity-40 dark:bg-zinc-100 dark:text-zinc-900"
                >
                  <Plus className="h-3 w-3" />
                  Rule
                </button>
              </div>

              {declaration.features.map((feat, idx) => (
                <div
                  key={feat.id}
                  className="flex items-start justify-between gap-2 rounded-md bg-zinc-50 px-2.5 py-1.5 dark:bg-zinc-900/60"
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-1.5">
                      <code className="font-mono text-[10px] font-semibold text-emerald-700 dark:text-emerald-400">
                        {feat.id}
                      </code>
                      <span className="text-[11px] font-medium text-zinc-900 dark:text-zinc-100">
                        {feat.outcome.kind === 'label' ? feat.outcome.label : `${feat.outcome.label} (formula)`}
                      </span>
                    </div>
                    <p className="text-[10px] text-zinc-500">{feat.statement}</p>
                  </div>
                  <div className="flex shrink-0 items-center gap-0.5">
                    <button
                      type="button"
                      disabled={idx === 0}
                      onClick={() => void handleMoveFeature(feat.id, -1)}
                      className="rounded p-1 text-zinc-400 hover:text-zinc-800 disabled:opacity-30"
                      title="Higher priority"
                    >
                      <ArrowUp className="h-3 w-3" />
                    </button>
                    <button
                      type="button"
                      disabled={idx === declaration.features.length - 1}
                      onClick={() => void handleMoveFeature(feat.id, 1)}
                      className="rounded p-1 text-zinc-400 hover:text-zinc-800 disabled:opacity-30"
                      title="Lower priority"
                    >
                      <ArrowDown className="h-3 w-3" />
                    </button>
                    <button
                      type="button"
                      onClick={() => startEditRule(feat)}
                      className="rounded p-1 text-zinc-400 hover:text-zinc-800"
                      title="Edit rule"
                    >
                      <Pencil className="h-3 w-3" />
                    </button>
                    <button
                      type="button"
                      onClick={() => void onDispatchOp({ op: 'remove-feature', id: feat.id })}
                      className="rounded p-1 text-zinc-400 hover:text-red-500"
                      title="Remove rule"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                </div>
              ))}

              {addingRule && fields.length > 0 && (
                <form onSubmit={(e) => void handleSaveRule(e)} className="space-y-2 rounded-lg border border-zinc-200 p-2.5 dark:border-zinc-700">
                  <label className="flex items-center gap-2 text-[11px]">
                    <input
                      type="checkbox"
                      checked={ruleHasCondition}
                      onChange={(e) => setRuleHasCondition(e.target.checked)}
                    />
                    <span>Apply condition (uncheck for default fallback rule)</span>
                  </label>
                  {ruleHasCondition && (
                    <div className="grid grid-cols-4 gap-1">
                      <select
                        value={ruleField}
                        onChange={(e) => setRuleField(e.target.value)}
                        className="rounded border border-zinc-300 bg-white px-1.5 py-1 text-[11px] dark:border-zinc-700 dark:bg-zinc-900"
                      >
                        {fields.map((f) => (
                          <option key={f.name} value={f.name}>{f.label}</option>
                        ))}
                      </select>
                      <select
                        value={ruleOp}
                        onChange={(e) => setRuleOp(e.target.value as CompareOp)}
                        className="rounded border border-zinc-300 bg-white px-1.5 py-1 text-[11px] dark:border-zinc-700 dark:bg-zinc-900"
                      >
                        {(Object.keys(COMPARE_LABELS) as CompareOp[]).map((op) => (
                          <option key={op} value={op}>{COMPARE_LABELS[op]}</option>
                        ))}
                      </select>
                      <select
                        value={ruleRhsMode}
                        onChange={(e) => setRuleRhsMode(e.target.value as 'const' | 'field')}
                        className="rounded border border-zinc-300 bg-white px-1 py-1 text-[11px] dark:border-zinc-700 dark:bg-zinc-900"
                      >
                        <option value="const">Value</option>
                        <option value="field">Field</option>
                      </select>
                      {ruleRhsMode === 'field' ? (
                        <select
                          value={ruleRhsField}
                          onChange={(e) => setRuleRhsField(e.target.value)}
                          className="rounded border border-zinc-300 bg-white px-1 py-1 text-[11px] dark:border-zinc-700 dark:bg-zinc-900"
                        >
                          {fields.map((f) => (
                            <option key={f.name} value={f.name}>{f.label}</option>
                          ))}
                        </select>
                      ) : (
                        <input
                          type="number"
                          value={ruleThreshold}
                          onChange={(e) => setRuleThreshold(e.target.value)}
                          className="rounded border border-zinc-300 bg-transparent px-1.5 py-1 text-[11px] dark:border-zinc-700"
                        />
                      )}
                    </div>
                  )}
                  <div className="grid grid-cols-2 gap-1.5">
                    <select
                      value={ruleOutcomeKind}
                      onChange={(e) => setRuleOutcomeKind(e.target.value as 'label' | 'formula')}
                      className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                    >
                      <option value="label">Return Verdict Label</option>
                      <option value="formula">Return Formula Result</option>
                    </select>
                    <input
                      type="text"
                      value={ruleOutcomeLabel}
                      onChange={(e) => setRuleOutcomeLabel(e.target.value)}
                      placeholder="Outcome label"
                      className="rounded border border-zinc-300 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                    />
                  </div>
                  {ruleOutcomeKind === 'label' ? (
                    <select
                      value={ruleOutcomeTone}
                      onChange={(e) => setRuleOutcomeTone(e.target.value as typeof ruleOutcomeTone)}
                      className="w-full rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                    >
                      <option value="success">Tone: Positive / Approved</option>
                      <option value="warning">Tone: Review / Caution</option>
                      <option value="destructive">Tone: Denied / Alert</option>
                      <option value="brand">Tone: Brand Highlight</option>
                      <option value="default">Tone: Neutral</option>
                    </select>
                  ) : (
                    <div className="grid grid-cols-3 gap-1">
                      <select
                        value={ruleFormulaOp}
                        onChange={(e) => setRuleFormulaOp(e.target.value as typeof ruleFormulaOp)}
                        className="rounded border border-zinc-300 bg-white px-1.5 py-1 text-[11px] dark:border-zinc-700 dark:bg-zinc-900"
                      >
                        <option value="field">Exact field</option>
                        <option value="add">Add (+)</option>
                        <option value="sub">Subtract (-)</option>
                        <option value="mul">Multiply (×)</option>
                        <option value="div">Divide (÷)</option>
                        <option value="pct">Percent (%)</option>
                      </select>
                      {ruleFormulaOp !== 'field' && (
                        <>
                          <select
                            value={ruleFormulaRhsMode}
                            onChange={(e) => setRuleFormulaRhsMode(e.target.value as 'const' | 'field')}
                            className="rounded border border-zinc-300 bg-white px-1 py-1 text-[11px] dark:border-zinc-700 dark:bg-zinc-900"
                          >
                            <option value="const">Constant</option>
                            <option value="field">Field</option>
                          </select>
                          {ruleFormulaRhsMode === 'field' ? (
                            <select
                              value={ruleFormulaRightField}
                              onChange={(e) => setRuleFormulaRightField(e.target.value)}
                              className="rounded border border-zinc-300 bg-white px-1 py-1 text-[11px] dark:border-zinc-700 dark:bg-zinc-900"
                            >
                              {fields.map((f) => (
                                <option key={f.name} value={f.name}>{f.label}</option>
                              ))}
                            </select>
                          ) : (
                            <input
                              type="number"
                              value={ruleFormulaRight}
                              onChange={(e) => setRuleFormulaRight(e.target.value)}
                              className="rounded border border-zinc-300 bg-transparent px-1.5 py-1 text-[11px] dark:border-zinc-700"
                            />
                          )}
                        </>
                      )}
                    </div>
                  )}
                  <input
                    type="text"
                    value={ruleStatement}
                    onChange={(e) => setRuleStatement(e.target.value)}
                    placeholder="Optional plain-English rule statement…"
                    className="w-full rounded border border-zinc-300 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                  />
                  <div className="flex justify-end gap-1.5">
                    <button
                      type="button"
                      onClick={() => {
                        setAddingRule(false);
                        setEditingRuleId(null);
                      }}
                      className="rounded px-2 py-1 text-xs text-zinc-500"
                    >
                      Cancel
                    </button>
                    <button type="submit" className="rounded bg-emerald-700 px-2.5 py-1 text-xs font-medium text-white">
                      {editingRuleId ? 'Save Rule' : 'Admit Rule'}
                    </button>
                  </div>
                </form>
              )}
            </section>

            {/* Typed Refusals (R5) */}
            <section className="space-y-2 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <div className="flex items-center justify-between">
                <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                  Typed Refusals ({declaration.refusals.length})
                </h4>
                <button
                  type="button"
                  disabled={fields.length === 0}
                  onClick={() => {
                    setEditingRefusalId(null);
                    setAddingRefusal((v) => !v);
                  }}
                  className="inline-flex items-center gap-1 rounded bg-zinc-900 px-2 py-0.5 text-[10px] font-medium text-white disabled:opacity-40 dark:bg-zinc-100 dark:text-zinc-900"
                >
                  <Plus className="h-3 w-3" />
                  Refusal
                </button>
              </div>
              {declaration.refusals.map((ref) => (
                <div
                  key={ref.id}
                  className="flex items-start justify-between gap-2 rounded-md bg-red-50/60 px-2.5 py-1.5 dark:bg-red-950/20"
                >
                  <div className="min-w-0">
                    <code className="font-mono text-[10px] font-semibold text-red-700 dark:text-red-400">{ref.id}</code>{' '}
                    <span className="text-[11px] font-medium text-zinc-900 dark:text-zinc-100">“{ref.failureText}”</span>
                    <p className="text-[10px] text-zinc-500">{ref.statement}</p>
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    <button type="button" onClick={() => startEditRefusal(ref)} className="text-zinc-400 hover:text-zinc-800">
                      <Pencil className="h-3 w-3" />
                    </button>
                    <button
                      type="button"
                      onClick={() => void onDispatchOp({ op: 'remove-refusal', id: ref.id })}
                      className="text-zinc-400 hover:text-red-500"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                </div>
              ))}
              {addingRefusal && fields.length > 0 && (
                <form onSubmit={(e) => void handleSaveRefusal(e)} className="space-y-2 rounded-lg border border-zinc-200 p-2.5 dark:border-zinc-700">
                  <div className="grid grid-cols-3 gap-1">
                    <select
                      value={refusalField}
                      onChange={(e) => setRefusalField(e.target.value)}
                      className="rounded border border-zinc-300 bg-white px-1.5 py-1 text-[11px] dark:border-zinc-700 dark:bg-zinc-900"
                    >
                      {fields.map((f) => (
                        <option key={f.name} value={f.name}>{f.label}</option>
                      ))}
                    </select>
                    <select
                      value={refusalOp}
                      onChange={(e) => setRefusalOp(e.target.value as CompareOp)}
                      className="rounded border border-zinc-300 bg-white px-1.5 py-1 text-[11px] dark:border-zinc-700 dark:bg-zinc-900"
                    >
                      {(Object.keys(COMPARE_LABELS) as CompareOp[]).map((op) => (
                        <option key={op} value={op}>{COMPARE_LABELS[op]}</option>
                      ))}
                    </select>
                    <input
                      type="number"
                      value={refusalValue}
                      onChange={(e) => setRefusalValue(e.target.value)}
                      className="rounded border border-zinc-300 bg-transparent px-1.5 py-1 text-[11px] dark:border-zinc-700"
                    />
                  </div>
                  <input
                    type="text"
                    value={refusalText}
                    onChange={(e) => setRefusalText(e.target.value)}
                    placeholder="Exact refusal message (e.g. Amount must be greater than zero.)"
                    className="w-full rounded border border-zinc-300 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                  />
                  <div className="flex justify-end gap-1.5">
                    <button
                      type="button"
                      onClick={() => {
                        setAddingRefusal(false);
                        setEditingRefusalId(null);
                      }}
                      className="rounded px-2 py-1 text-xs text-zinc-500"
                    >
                      Cancel
                    </button>
                    <button type="submit" className="rounded bg-red-700 px-2.5 py-1 text-xs font-medium text-white">
                      {editingRefusalId ? 'Save Refusal' : 'Admit Refusal'}
                    </button>
                  </div>
                </form>
              )}
            </section>

            {/* Acceptance Examples */}
            <section className="space-y-2 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <div className="flex items-center justify-between">
                <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                  Acceptance Examples ({checkReport.vectorsPassed}/{checkReport.vectorsTotal} passing)
                </h4>
                <button
                  type="button"
                  disabled={fields.length === 0}
                  onClick={() => {
                    const init: Record<string, number> = {};
                    for (const f of fields) init[f.name] = f.defaultValue;
                    setVecInputs(init);
                    setAddingVector((v) => !v);
                  }}
                  className="inline-flex items-center gap-1 rounded bg-zinc-900 px-2 py-0.5 text-[10px] font-medium text-white disabled:opacity-40 dark:bg-zinc-100 dark:text-zinc-900"
                >
                  <Plus className="h-3 w-3" />
                  Example
                </button>
              </div>
              {checkReport.vectorResults.map((vr) => (
                <div
                  key={vr.id}
                  className="flex items-center justify-between gap-2 rounded bg-zinc-50 px-2 py-1.5 text-[11px] dark:bg-zinc-900/60"
                >
                  <div className="min-w-0">
                    <span className={vr.passed ? 'text-emerald-600' : 'text-red-600'}>
                      {vr.passed ? '✓' : '✗'}
                    </span>{' '}
                    <code className="font-mono text-[10px]">{vr.request}</code> →{' '}
                    <span className="font-medium">{vr.expected}</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => void onDispatchOp({ op: 'remove-vector', id: vr.id })}
                    className="text-zinc-400 hover:text-red-500"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </div>
              ))}
              {addingVector && fields.length > 0 && (
                <form onSubmit={(e) => void handleAddVector(e)} className="space-y-2 rounded-lg border border-zinc-200 p-2.5 dark:border-zinc-700">
                  <div className="grid grid-cols-2 gap-1.5">
                    {fields.map((f) => (
                      <label key={f.name} className="block text-[10px]">
                        <span className="text-zinc-500">{f.label}</span>
                        {f.kind === 'choice' && f.choices ? (
                          <select
                            value={vecInputs[f.name] ?? f.defaultValue}
                            onChange={(e) =>
                              setVecInputs((p) => ({ ...p, [f.name]: Number.parseInt(e.target.value, 10) }))
                            }
                            className="mt-0.5 w-full rounded border border-zinc-300 bg-white px-1.5 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                          >
                            {f.choices.map((c, ci) => (
                              <option key={c} value={ci}>{c}</option>
                            ))}
                          </select>
                        ) : (
                          <input
                            type="number"
                            value={vecInputs[f.name] ?? f.defaultValue}
                            onChange={(e) =>
                              setVecInputs((p) => ({ ...p, [f.name]: Number.parseInt(e.target.value, 10) || 0 }))
                            }
                            className="mt-0.5 w-full rounded border border-zinc-300 bg-transparent px-1.5 py-1 text-xs dark:border-zinc-700"
                          />
                        )}
                      </label>
                    ))}
                  </div>
                  {previewVecAnswer && (
                    <p className="rounded bg-zinc-100 px-2 py-1 font-mono text-[10px] dark:bg-zinc-900">
                      Model answer: <strong>{previewVecAnswer.answer.response}</strong>
                    </p>
                  )}
                  <div className="flex justify-end gap-1.5">
                    <button type="button" onClick={() => setAddingVector(false)} className="px-2 py-1 text-xs text-zinc-500">
                      Cancel
                    </button>
                    <button type="submit" className="rounded bg-emerald-700 px-2.5 py-1 text-xs font-medium text-white">
                      Admit Example
                    </button>
                  </div>
                </form>
              )}
            </section>

            {/* Stage 2 -> Stage 3 Footer */}
            <div className="flex items-center justify-between gap-2 pt-1">
              <button
                type="button"
                onClick={async () => {
                  await onDispatchOp({ op: 'sync-stride-threats' });
                  onSelectStage('security');
                }}
                className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-700 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-800"
              >
                <Shield className="h-3.5 w-3.5" />
                Derive STRIDE Threats &amp; Continue
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        )}

        {/* ================================================================ */}
        {/* STAGE 3: SECURITY & THREAT MODEL (STRIDE + OWASP ASVS 5.0.0)      */}
        {/* ================================================================ */}
        {activeStage === 'security' && (
          <div className="space-y-4">
            <div className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <div className="flex items-center justify-between gap-2">
                <div>
                  <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                    OWASP ASVS 5.0.0 Assurance Tier
                  </h4>
                  <p className="mt-0.5 text-[10px] text-zinc-500">
                    Recommended from Concept weights: <strong>{recAssurance}</strong>
                  </p>
                </div>
                <div className="inline-flex rounded-lg border border-zinc-200 bg-zinc-100 p-0.5 dark:border-zinc-700 dark:bg-zinc-900">
                  {(['L1', 'L2', 'L3'] as const satisfies readonly AssuranceLevel[]).map((lvl) => (
                    <button
                      key={lvl}
                      type="button"
                      onClick={() => void onDispatchOp({ op: 'set-assurance-level', level: lvl })}
                      className={`rounded-md px-2.5 py-1 text-[11px] font-semibold ${
                        declaration.security.assuranceLevel === lvl
                          ? 'bg-white text-zinc-900 shadow-xs dark:bg-zinc-800 dark:text-zinc-100'
                          : 'text-zinc-500 hover:text-zinc-800 dark:text-zinc-400'
                      }`}
                    >
                      {lvl}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Trust Boundaries */}
            <div className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                Derived Trust Boundaries
              </h4>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {routing.trustBoundaries.map((tb) => (
                  <span
                    key={tb.id}
                    title={tb.summary}
                    className={`rounded-md border px-2 py-0.5 text-[10px] font-medium ${
                      tb.active
                        ? 'border-emerald-300 bg-emerald-50 text-emerald-900 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-200'
                        : 'border-zinc-200 bg-zinc-50 text-zinc-400 line-through dark:border-zinc-800 dark:bg-zinc-900/40'
                    }`}
                  >
                    {tb.name}
                  </span>
                ))}
              </div>
            </div>

            {/* STRIDE Threat Register */}
            <div className="space-y-2 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <div className="flex items-center justify-between gap-2">
                <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                  STRIDE Threat Register ({declaration.security.threats.filter((t) => t.status !== 'open').length}/{declaration.security.threats.length})
                </h4>
                <button
                  type="button"
                  onClick={() => void onDispatchOp({ op: 'sync-stride-threats' })}
                  className="rounded bg-zinc-900 px-2 py-0.5 text-[10px] font-medium text-white dark:bg-zinc-100 dark:text-zinc-900"
                >
                  Sync from System
                </button>
              </div>

              {declaration.security.threats.length === 0 ? (
                <div className="rounded border border-dashed border-zinc-300 p-3 text-center dark:border-zinc-700">
                  <p className="text-[11px] text-zinc-500">
                    Click <strong>Sync from System</strong> to derive the STRIDE threat matrix from your OPM assets and runtime capabilities.
                  </p>
                </div>
              ) : (
                <div className="space-y-2">
                  {declaration.security.threats.map((thr) => (
                    <div
                      key={thr.id}
                      className={`rounded-lg border p-2.5 ${
                        thr.status === 'open'
                          ? 'border-amber-300 bg-amber-50/50 dark:border-amber-900/60 dark:bg-amber-950/20'
                          : 'border-zinc-200 bg-zinc-50/60 dark:border-zinc-800 dark:bg-zinc-900/40'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <div className="flex flex-wrap items-center gap-1.5">
                            <span className="rounded bg-zinc-200/80 px-1.5 py-0.5 font-mono text-[9px] font-semibold uppercase text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
                              {thr.category}
                            </span>
                            <span className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">
                              {thr.title}
                            </span>
                          </div>
                          <p className="mt-0.5 text-[10px] text-zinc-500">
                            Asset: {thr.asset} · Boundary: {thr.boundary}
                          </p>
                          <p className="mt-1 text-[11px] text-zinc-700 dark:text-zinc-300">{thr.rationale}</p>
                        </div>
                        <button
                          type="button"
                          onClick={() => {
                            setEditingThreatId(thr.id);
                            setThreatStatus(
                              thr.status === 'open' ? 'mitigated-by-sandbox' : thr.status,
                            );
                            setThreatRefusalId(thr.linkedRefusalId ?? declaration.refusals[0]?.id ?? '');
                            setThreatRationale(thr.rationale);
                          }}
                          className={`shrink-0 rounded px-2 py-0.5 text-[10px] font-semibold ${
                            thr.status === 'open'
                              ? 'bg-amber-600 text-white hover:bg-amber-700'
                              : 'border border-zinc-200 text-zinc-600 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300'
                          }`}
                        >
                          {thr.status === 'open' ? 'Settle' : thr.status}
                        </button>
                      </div>

                      {editingThreatId === thr.id && (
                        <form onSubmit={(e) => void handleSettleThreatSubmit(e)} className="mt-2 space-y-1.5 border-t border-zinc-200 pt-2 dark:border-zinc-700">
                          <select
                            value={threatStatus}
                            onChange={(e) => setThreatStatus(e.target.value as Exclude<ThreatDisposition, 'open'>)}
                            className="w-full rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                          >
                            <option value="mitigated-by-sandbox">Mitigated by WASM Sandbox / Deny-by-Default</option>
                            {declaration.refusals.length > 0 && (
                              <option value="mitigated-by-refusal">Mitigated by Registered Refusal</option>
                            )}
                            <option value="not-applicable">Not Applicable (with rationale)</option>
                          </select>
                          {threatStatus === 'mitigated-by-refusal' && declaration.refusals.length > 0 && (
                            <select
                              value={threatRefusalId}
                              onChange={(e) => setThreatRefusalId(e.target.value)}
                              className="w-full rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                            >
                              {declaration.refusals.map((r) => (
                                <option key={r.id} value={r.id}>
                                  {r.id}: {r.failureText}
                                </option>
                              ))}
                            </select>
                          )}
                          <input
                            type="text"
                            value={threatRationale}
                            onChange={(e) => setThreatRationale(e.target.value)}
                            placeholder="Document mitigation or exclusion rationale…"
                            className="w-full rounded border border-zinc-300 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                          />
                          <div className="flex justify-end gap-1.5">
                            <button
                              type="button"
                              onClick={() => setEditingThreatId(null)}
                              className="px-2 py-1 text-xs text-zinc-500"
                            >
                              Cancel
                            </button>
                            <button
                              type="submit"
                              className="rounded bg-emerald-700 px-2.5 py-1 text-xs font-medium text-white"
                            >
                              Save Disposition
                            </button>
                          </div>
                        </form>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="flex justify-end">
              <button
                type="button"
                onClick={() => onSelectStage('design')}
                className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-700 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-800"
              >
                Next: Design &amp; UX
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        )}

        {/* ================================================================ */}
        {/* STAGE 4: DESIGN & EXPERIENCE (ISO 9241 / WCAG 2.2 / prism.design) */}
        {/* ================================================================ */}
        {activeStage === 'design' && (
          <div className="space-y-4">
            {!declaration.view ? (
              <div className="rounded-xl border border-dashed border-emerald-400/70 bg-emerald-50/40 p-4 text-center dark:border-emerald-800 dark:bg-emerald-950/20">
                <Palette className="mx-auto h-6 w-6 text-emerald-600" />
                <p className="mt-2 text-xs font-semibold text-zinc-900 dark:text-zinc-100">
                  No View Synthesized Yet
                </p>
                <p className="mt-1 text-[11px] text-zinc-500">
                  Synthesize a requirement-traced <code className="font-mono">prism.design/1</code> interface directly from your System inputs, outcomes, and refusals.
                </p>
                <button
                  type="button"
                  disabled={fields.length === 0}
                  onClick={() => void onDispatchOp({ op: 'synthesize-view-from-system' })}
                  className="mt-3 inline-flex items-center gap-1.5 rounded-lg bg-emerald-700 px-3.5 py-1.5 text-xs font-medium text-white hover:bg-emerald-800 disabled:opacity-45"
                >
                  <Sparkles className="h-3.5 w-3.5" />
                  Synthesize Traceable View from System
                </button>
              </div>
            ) : (
              <>
                {/* Viewpoint Selector */}
                <section className="space-y-2 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
                  <div className="flex items-center justify-between">
                    <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                      Interface Realization
                    </h4>
                    <button
                      type="button"
                      onClick={() => void onDispatchOp({ op: 'synthesize-view-from-system' })}
                      className="text-[10px] font-medium text-emerald-700 hover:underline dark:text-emerald-400"
                    >
                      Re-sync Controls
                    </button>
                  </div>
                  {declaration.view.authoredView && (
                    <div className="rounded-lg border border-emerald-500/70 bg-emerald-50/70 p-2.5 text-xs dark:border-emerald-700 dark:bg-emerald-950/30">
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-semibold text-emerald-900 dark:text-emerald-200">
                          Interactive App (Studio Agent)
                        </span>
                        <span className="font-mono text-[10px] text-emerald-800/80 dark:text-emerald-300/80">
                          {(declaration.view.authoredView.html.length / 1024).toFixed(1)} KB HTML ·{' '}
                          {(declaration.view.authoredView.js.length / 1024).toFixed(1)} KB JS
                        </span>
                      </div>
                      <p className="mt-1 text-[10px] text-emerald-800/90 dark:text-emerald-300/90">
                        Custom interactive view bound to your proved core model. Refine it in plain English in Guided mode, or pick a structured layout below to switch to a form.
                      </p>
                    </div>
                  )}
                  <div className="grid grid-cols-2 gap-1.5">
                    {(
                      [
                        ['decision-form', 'Decision Form', 'Structured policy form + verdict card'],
                        ['calculator', 'Live Calculator', 'Instant recalculation on input'],
                        ['ledger-tracker', 'Ledger Tracker', 'Summary stats + evaluation history'],
                        ['request-reply', 'Request & Reply', 'Transcript of requests and answers'],
                      ] as readonly [PrismViewpoint, string, string][]
                    ).map(([vp, label, desc]) => (
                      <button
                        key={vp}
                        type="button"
                        onClick={() => void onDispatchOp({ op: 'synthesize-view-from-system', viewpoint: vp })}
                        className={`rounded-lg border p-2 text-left transition-colors ${
                          declaration.view?.viewpoint === vp && !declaration.view.authoredView
                            ? 'border-emerald-600 bg-emerald-50/70 dark:border-emerald-500 dark:bg-emerald-950/30'
                            : 'border-zinc-200 hover:bg-zinc-50 dark:border-zinc-800 dark:hover:bg-zinc-900'
                        }`}
                      >
                        <p className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">{label}</p>
                        <p className="text-[10px] text-zinc-500">{desc}</p>
                      </button>
                    ))}
                  </div>
                </section>

                {/* Copy & Disclosure */}
                <section className="space-y-2 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
                  <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                    Header Copy &amp; Compliance Disclosure
                  </h4>
                  <div className="grid grid-cols-2 gap-1.5">
                    <input
                      type="text"
                      value={declaration.view.eyebrow}
                      onChange={(e) => void handleUpdateViewField({ eyebrow: e.target.value })}
                      placeholder="Eyebrow"
                      className="rounded border border-zinc-300 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                    />
                    <input
                      type="text"
                      value={declaration.view.submitLabel}
                      onChange={(e) => void handleUpdateViewField({ submitLabel: e.target.value })}
                      placeholder="Submit button label"
                      className="rounded border border-zinc-300 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                    />
                  </div>
                  <input
                    type="text"
                    value={declaration.view.subtitle}
                    onChange={(e) => void handleUpdateViewField({ subtitle: e.target.value })}
                    placeholder="Subtitle"
                    className="w-full rounded border border-zinc-300 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                  />
                  <input
                    type="text"
                    value={declaration.view.disclosure ?? ''}
                    onChange={(e) => void handleUpdateViewField({ disclosure: e.target.value || undefined })}
                    placeholder="Optional footer privacy / AI disclosure…"
                    className="w-full rounded border border-zinc-300 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                  />
                </section>

                {/* Theme Presets & Custom Accent */}
                <section className="space-y-2 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
                  <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                    W3C DTCG Theme Preset (prism.design/1)
                  </h4>
                  <div className="grid grid-cols-2 gap-1.5">
                    {PRISM_THEME_PRESETS.map((preset) => (
                      <button
                        key={preset.id}
                        type="button"
                        onClick={() =>
                          void onDispatchOp({
                            op: 'set-theme',
                            theme: { preset: preset.id, accent: declaration.theme.accent },
                          })
                        }
                        className={`flex items-center justify-between gap-2 rounded-lg border px-2.5 py-1.5 text-left ${
                          declaration.theme.preset === preset.id
                            ? 'border-emerald-600 bg-emerald-50/50 dark:border-emerald-500 dark:bg-emerald-950/30'
                            : 'border-zinc-200 hover:bg-zinc-50 dark:border-zinc-800'
                        }`}
                      >
                        <div className="min-w-0">
                          <p className="text-xs font-semibold">{preset.name}</p>
                          <p className="truncate text-[10px] text-zinc-500">{preset.summary}</p>
                        </div>
                        <div className="flex shrink-0 gap-0.5">
                          {preset.swatches.map((hex) => (
                            <span
                              key={hex}
                              className="h-3.5 w-3.5 rounded-full border border-black/15"
                              style={{ backgroundColor: hex }}
                            />
                          ))}
                        </div>
                      </button>
                    ))}
                  </div>
                  <div className="flex items-center justify-between gap-2 pt-1">
                    <label className="flex items-center gap-2 text-[11px]">
                      <span className="text-zinc-500">Custom Accent</span>
                      <input
                        type="color"
                        value={declaration.theme.accent ?? '#1f5e44'}
                        onChange={(e) =>
                          void onDispatchOp(
                            { op: 'set-theme', theme: { ...declaration.theme, accent: e.target.value } },
                            { pushRevision: false },
                          )
                        }
                        className="h-6 w-8 cursor-pointer rounded border border-zinc-300 bg-transparent"
                      />
                    </label>
                    {declaration.theme.accent && (
                      <button
                        type="button"
                        onClick={() =>
                          void onDispatchOp({
                            op: 'set-theme',
                            theme: { preset: declaration.theme.preset },
                          })
                        }
                        className="text-[11px] text-zinc-500 hover:underline"
                      >
                        Reset Accent
                      </button>
                    )}
                  </div>
                </section>
              </>
            )}

            {/* WCAG 2.2 Automated Baseline Checks */}
            <section className="space-y-1.5 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                WCAG 2.2 &amp; Traceability Baseline
              </h4>
              {routing.designChecks.map((chk) => (
                <div key={chk.id} className="flex items-start gap-2 rounded bg-zinc-50 px-2.5 py-1.5 dark:bg-zinc-900/60">
                  {chk.passed ? (
                    <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-600" />
                  ) : (
                    <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-500" />
                  )}
                  <div className="min-w-0">
                    <p className="text-[11px] font-medium text-zinc-900 dark:text-zinc-100">
                      {chk.label} <span className="font-mono text-[9px] text-zinc-400">({chk.wcagRef})</span>
                    </p>
                    <p className="text-[10px] text-zinc-500">{chk.detail}</p>
                  </div>
                </div>
              ))}
            </section>

            <div className="flex justify-end">
              <button
                type="button"
                onClick={() =>
                  onSelectStage(routing.activeLayeredStandards.length > 0 ? 'standards' : 'seal')
                }
                className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-700 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-800"
              >
                Next: {routing.activeLayeredStandards.length > 0 ? 'Routed Standards' : 'Verify & Seal'}
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        )}

        {/* ================================================================ */}
        {/* STAGE 5: CONDITIONAL ROUTED LAYERED STANDARDS (REQ-D6 / REQ-D11)  */}
        {/* ================================================================ */}
        {activeStage === 'standards' && (
          <div className="space-y-3">
            <div className="rounded-lg border border-emerald-200/80 bg-emerald-50/50 p-3 dark:border-emerald-900/50 dark:bg-emerald-950/20">
              <p className="text-[11px] font-semibold text-emerald-900 dark:text-emerald-200">
                Routed Layered Standards ({routing.summary.obligationsSettled}/{routing.summary.obligationsTotal} obligations settled)
              </p>
              <p className="mt-0.5 text-[11px] text-emerald-800/80 dark:text-emerald-300/80">
                Activated deterministically from your Stage 1–3 Core weights. Each settled obligation enters with a formal <code className="font-mono">Std</code> citation.
              </p>
            </div>

            {routing.activeLayeredStandards.map((std) => (
              <div key={std.authority.id} className="space-y-2 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <h4 className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">
                      {std.authority.name}
                    </h4>
                    <p className="text-[10px] text-zinc-500">{std.reason}</p>
                  </div>
                  <code className="shrink-0 rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-[9px] text-zinc-500 dark:bg-zinc-800">
                    {std.authority.id}
                  </code>
                </div>

                <div className="space-y-1.5">
                  {std.obligations.map((ob) => (
                    <div
                      key={ob.id}
                      className={`flex items-start justify-between gap-2 rounded-md border p-2 ${
                        ob.settled
                          ? 'border-emerald-200 bg-emerald-50/50 dark:border-emerald-900/50 dark:bg-emerald-950/20'
                          : 'border-amber-200 bg-amber-50/50 dark:border-amber-900/50 dark:bg-amber-950/20'
                      }`}
                    >
                      <div className="min-w-0">
                        <p className="text-[11px] font-semibold text-zinc-900 dark:text-zinc-100">{ob.title}</p>
                        <p className="text-[10px] text-zinc-500">{ob.evidence ?? ob.detail}</p>
                      </div>
                      {ob.settled ? (
                        <span className="inline-flex shrink-0 items-center gap-1 rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-semibold text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300">
                          <Check className="h-3 w-3" />
                          {ob.autoSatisfied ? 'Verified' : 'Settled'}
                        </span>
                      ) : ob.suggestedOp ? (
                        <button
                          type="button"
                          onClick={() => void onDispatchOp(ob.suggestedOp!)}
                          className="shrink-0 rounded bg-emerald-700 px-2.5 py-1 text-[10px] font-semibold text-white hover:bg-emerald-800"
                        >
                          Apply &amp; Settle
                        </button>
                      ) : (
                        <span className="shrink-0 text-[10px] font-medium text-amber-700 dark:text-amber-300">
                          Resolve in Core
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}

            <div className="flex justify-end">
              <button
                type="button"
                onClick={() => onSelectStage('seal')}
                className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-700 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-800"
              >
                Next: Verify &amp; Seal
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        )}

        {/* ================================================================ */}
        {/* STAGE 6: VERIFY & SEAL (Completeness Gate + Decisions + Audit)    */}
        {/* ================================================================ */}
        {activeStage === 'seal' && (
          <div className="space-y-4">
            <section className="space-y-2 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <div className="flex items-center justify-between gap-2">
                <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                  Verdant Prism Completeness Gate
                </h4>
                <span
                  className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                    routing.readyToSeal
                      ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300'
                      : 'bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300'
                  }`}
                >
                  {routing.readyToSeal ? 'Ready to Prove & Seal' : 'Incomplete'}
                </span>
              </div>
              <div className="space-y-1.5">
                {routing.stages
                  .filter((s) => s.stage !== 'seal' && s.visible)
                  .map((st) => (
                    <button
                      key={st.stage}
                      type="button"
                      onClick={() => onSelectStage(st.stage)}
                      className="flex w-full items-center justify-between rounded-md bg-zinc-50 px-2.5 py-1.5 text-left hover:bg-zinc-100 dark:bg-zinc-900/60 dark:hover:bg-zinc-800"
                    >
                      <span className="flex items-center gap-2">
                        {st.status === 'complete' ? (
                          <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
                        ) : (
                          <AlertTriangle className="h-3.5 w-3.5 text-amber-500" />
                        )}
                        <span className="font-medium text-zinc-900 dark:text-zinc-100">{st.label}</span>
                      </span>
                      <span className="text-[10px] text-zinc-500">
                        {st.missing.length === 0 ? 'Complete' : st.missing[0]}
                      </span>
                    </button>
                  ))}
              </div>
              <div className="flex flex-wrap gap-2 pt-2">
                <button
                  type="button"
                  disabled={!routing.readyToSeal || building}
                  onClick={onCheckAndBuild}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-700 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-800 disabled:opacity-45"
                >
                  {building ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ShieldCheck className="h-3.5 w-3.5" />}
                  Check &amp; Build (.holo v6)
                </button>
                <button
                  type="button"
                  disabled={!routing.readyToSeal || building}
                  onClick={onPublish}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-600 px-3 py-1.5 text-xs font-medium text-emerald-800 hover:bg-emerald-50 disabled:opacity-45 dark:border-emerald-700 dark:text-emerald-200 dark:hover:bg-emerald-950/40"
                >
                  <Globe className="h-3.5 w-3.5" />
                  Publish to App
                </button>
              </div>
            </section>

            {/* Owner Decisions Ledger */}
            <section className="space-y-2 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                Owner Decisions ({declaration.decisions.length})
              </h4>
              {declaration.decisions.map((d) => (
                <div key={d.id} className="rounded bg-zinc-50 px-2.5 py-1.5 text-[11px] dark:bg-zinc-900/60">
                  <code className="font-mono text-[10px] text-zinc-400">{d.id}</code> · “{d.quote}”
                </div>
              ))}
              <form onSubmit={(e) => void handleRecordDecision(e)} className="flex gap-1.5">
                <input
                  type="text"
                  value={decisionQuote}
                  onChange={(e) => setDecisionQuote(e.target.value)}
                  placeholder="Record verbatim owner decision…"
                  className="flex-1 rounded border border-zinc-300 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                />
                <button type="submit" className="rounded bg-zinc-900 px-2.5 py-1 text-xs font-medium text-white dark:bg-zinc-100 dark:text-zinc-900">
                  Record
                </button>
              </form>
            </section>

            {/* API Audit Log */}
            <section className="space-y-1 rounded-lg border border-zinc-200 p-3 dark:border-zinc-800">
              <h4 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
                Declarative API Audit Log ({declaration.auditLog.length})
              </h4>
              <div className="max-h-36 space-y-1 overflow-y-auto font-mono text-[10px]">
                {[...declaration.auditLog].reverse().slice(0, 15).map((entry) => (
                  <div key={entry.id} className="flex items-center justify-between gap-2 text-zinc-600 dark:text-zinc-400">
                    <span className="truncate">
                      <span className="text-emerald-700 dark:text-emerald-400">{entry.op}</span> · {entry.summary}
                    </span>
                  </div>
                ))}
              </div>
            </section>
          </div>
        )}
      </div>

      {/* Optional NL -> Structured Studio API Operations Bar */}
      <div className="shrink-0 border-t border-zinc-200 bg-zinc-50/70 p-2.5 dark:border-zinc-800 dark:bg-[#0B0F14]">
        {proposedOps && (
          <div className="mb-2 rounded-lg border border-emerald-300 bg-emerald-50 p-2 text-[11px] dark:border-emerald-800 dark:bg-emerald-950/40">
            <div className="flex items-center justify-between gap-2">
              <span className="font-semibold text-emerald-900 dark:text-emerald-200">
                Proposed {proposedOps.length} Studio API op(s)
              </span>
              <div className="flex gap-1">
                <button
                  type="button"
                  onClick={onApplyProposedOps}
                  className="rounded bg-emerald-700 px-2 py-0.5 text-[10px] font-semibold text-white"
                >
                  Apply
                </button>
                <button
                  type="button"
                  onClick={onDismissProposedOps}
                  className="rounded px-2 py-0.5 text-[10px] text-zinc-600 dark:text-zinc-300"
                >
                  Dismiss
                </button>
              </div>
            </div>
          </div>
        )}
        {modelStreamEnabled ? (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              if (!nlPrompt.trim() || proposingOps) return;
              onProposeOpsFromPrompt(nlPrompt.trim());
              setNlPrompt('');
            }}
            className="flex items-center gap-1.5"
          >
            <input
              type="text"
              value={nlPrompt}
              onChange={(e) => setNlPrompt(e.target.value)}
              placeholder="Optional: describe a rule to propose API ops…"
              className="min-w-0 flex-1 rounded-md border border-zinc-300 bg-white px-2.5 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
            />
            <button
              type="submit"
              disabled={proposingOps || !nlPrompt.trim()}
              className="inline-flex shrink-0 items-center gap-1 rounded-md bg-emerald-700 px-2.5 py-1 text-xs font-medium text-white hover:bg-emerald-800 disabled:opacity-45"
            >
              {proposingOps ? <Loader2 className="h-3 w-3 animate-spin" /> : <Sparkles className="h-3 w-3" />}
              Propose
            </button>
          </form>
        ) : (
          <div className="flex items-center justify-between text-[11px] text-zinc-500">
            <span className="inline-flex items-center gap-1">
              <Layers className="h-3 w-3 text-emerald-600" />
              Every change updates your Prism model deterministically
            </span>
            <button
              type="button"
              onClick={onEnableModelStream}
              className="font-medium text-emerald-700 hover:underline dark:text-emerald-400"
            >
              Enable Local AI Helper
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
