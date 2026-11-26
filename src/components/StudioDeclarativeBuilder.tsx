/**
 * Declarative Verdant Studio Left-Pane Builder (`src/components/StudioDeclarativeBuilder.tsx`)
 *
 * Calm, jargon-free GUI over the closed `VerdantStudioApi`:
 * - Zero raw code editing; every button dispatches a deterministic `StudioApiOp`.
 * - 4 clean tabs: Features (Requirements, Rules, Refusals, Examples), Data (Fields & JSON Schema),
 *   Design (Viewpoint, Traceable UI Elements, `prism.design/1` Theme Presets), Capabilities (Grants,
 *   Owner Decisions, Audit Log).
 * - Live Derived Agenda (`next`) & Self-Healing 1-Click API Repairs (`REQ-SELF-HEALING`).
 * - Optional NL -> API Operations proposer (proposes structured `StudioApiOp[]` for user approval,
 *   never raw HTML/CSS/JS).
 */

import { useMemo, useState, type FC, type FormEvent } from 'react';
import {
  AlertTriangle,
  ArrowDown,
  ArrowRight,
  ArrowUp,
  BookOpen,
  Check,
  CheckCircle2,
  Database,
  Layers,
  Loader2,
  Palette,
  Pencil,
  Plus,
  Shield,
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
  type CompareOp,
  type DeclarationCheckReport,
  type DeclarativeStarterId,
  type EncodedField,
  type FieldKind,
  type PrismDeclaration,
  type PrismViewpoint,
  type RequirementOrigin,
  type StudioAgendaItem,
  type StudioApiOp,
} from '../engine/verdantStudioApi';
import type { JsonSchema2020Object } from '../engine/prismStudio';

export type BuilderTab = 'features' | 'data' | 'design' | 'capabilities';

interface StudioDeclarativeBuilderProps {
  declaration: PrismDeclaration;
  checkReport: DeclarationCheckReport;
  agenda: readonly StudioAgendaItem[];
  onDispatchOp: (op: StudioApiOp) => Promise<boolean>;
  onLoadStarter: (starter: DeclarativeStarterId) => void;
  onImportHoloFile: () => void;
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

export const StudioDeclarativeBuilder: FC<StudioDeclarativeBuilderProps> = ({
  declaration,
  checkReport,
  agenda,
  onDispatchOp,
  onLoadStarter,
  onImportHoloFile,
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
  const [activeTab, setActiveTab] = useState<BuilderTab>('features');
  const [examplesOpen, setExamplesOpen] = useState(false);

  // Inline forms state
  const [addingReq, setAddingReq] = useState(false);
  const [reqStatement, setReqStatement] = useState('');
  const [reqOrigin, setReqOrigin] = useState<RequirementOrigin>('identity');

  const [addingRule, setAddingRule] = useState(false);
  const [editingRuleId, setEditingRuleId] = useState<string | null>(null);
  const [ruleStatement, setRuleStatement] = useState('');
  const [ruleField, setRuleField] = useState(() => declaration.encoding?.fields[0]?.name ?? 'amount');
  const [ruleOp, setRuleOp] = useState<CompareOp>('gte');
  const [ruleRhsMode, setRuleRhsMode] = useState<'const' | 'field'>('const');
  const [ruleRhsField, setRuleRhsField] = useState(() => declaration.encoding?.fields[1]?.name ?? declaration.encoding?.fields[0]?.name ?? 'amount');
  const [ruleThreshold, setRuleThreshold] = useState('100');
  const [ruleHasCondition, setRuleHasCondition] = useState(true);
  const [ruleOutcomeKind, setRuleOutcomeKind] = useState<'label' | 'formula'>('label');
  const [ruleOutcomeLabel, setRuleOutcomeLabel] = useState('Approved');
  const [ruleFormulaOp, setRuleFormulaOp] = useState<'field' | 'add' | 'sub' | 'mul' | 'pct'>('pct');
  const [ruleFormulaRight, setRuleFormulaRight] = useState('90');
  const [ruleUnit, setRuleUnit] = useState('');

  const [addingRefusal, setAddingRefusal] = useState(false);
  const [refusalField, setRefusalField] = useState(() => declaration.encoding?.fields[0]?.name ?? 'amount');
  const [refusalOp, setRefusalOp] = useState<CompareOp>('eq');
  const [refusalValue, setRefusalValue] = useState('0');
  const [refusalText, setRefusalText] = useState('');

  const [addingVector, setAddingVector] = useState(false);
  const [vecInputs, setVecInputs] = useState<Record<string, number>>({});

  // Data tab add-field state
  const [addingField, setAddingField] = useState(false);
  const [newFieldName, setNewFieldName] = useState('');
  const [newFieldLabel, setNewFieldLabel] = useState('');
  const [newFieldKind, setNewFieldKind] = useState<FieldKind>('nat');
  const [newFieldMax, setNewFieldMax] = useState('10000');
  const [newFieldUnit, setNewFieldUnit] = useState('');
  const [newFieldChoices, setNewFieldChoices] = useState('Standard, Pro, Enterprise');
  const [showRawSchema, setShowRawSchema] = useState(false);
  const [rawSchemaDraft, setRawSchemaDraft] = useState('');

  // Capabilities / Decision form
  const [decisionQuote, setDecisionQuote] = useState('');

  // Optional NL -> API Ops input
  const [nlPrompt, setNlPrompt] = useState('');

  const fields = declaration.encoding?.fields ?? [];
  const nextAgenda = agenda[0] ?? null;
  const firstReqId = declaration.requirements[0]?.id ?? 'REQ-01';

  const nextAppId = useMemo(() => {
    const existing = new Set([
      ...declaration.features.map((f) => f.id),
      ...declaration.refusals.map((r) => r.id),
    ]);
    for (let n = 1; n < 999; n++) {
      const candidate = `APP-${String(n).padStart(2, '0')}`;
      if (!existing.has(candidate)) return candidate;
    }
    return `APP-${Date.now().toString().slice(-3)}`;
  }, [declaration.features, declaration.refusals]);

  const nextReqId = useMemo(() => {
    const existing = new Set(declaration.requirements.map((r) => r.id));
    for (let n = 1; n < 999; n++) {
      const candidate = `REQ-${String(n).padStart(2, '0')}`;
      if (!existing.has(candidate)) return candidate;
    }
    return `REQ-${Date.now().toString().slice(-3)}`;
  }, [declaration.requirements]);

  const handleAddRequirement = async (e: FormEvent) => {
    e.preventDefault();
    if (!reqStatement.trim()) return;
    const ok = await onDispatchOp({
      op: 'admit-requirement',
      requirement: {
        id: nextReqId,
        statement: reqStatement.trim(),
        stakeholder: 'creator',
        origin: reqOrigin,
        evidence: 'Declared in Verdant Studio',
      },
    });
    if (ok) {
      setReqStatement('');
      setAddingReq(false);
    }
  };

  const handleAddRule = async (e: FormEvent) => {
    e.preventDefault();
    const targetField = fields.some((f) => f.name === ruleField)
      ? ruleField
      : (fields[0]?.name ?? 'amount');
    const numericThresh = Math.max(0, Number.parseInt(ruleThreshold, 10) || 0);
    const numericRight = Math.max(0, Number.parseInt(ruleFormulaRight, 10) || 0);
    const label = ruleOutcomeLabel.trim() || 'Result';
    const rhsFieldName =
      ruleRhsMode === 'field' && fields.some((f) => f.name === ruleRhsField)
        ? ruleRhsField
        : undefined;
    const rhsDisplay = rhsFieldName ?? String(numericThresh);
    const condition = {
      all: ruleHasCondition
        ? [{ field: targetField, op: ruleOp, value: numericThresh, ...(rhsFieldName ? { rhsField: rhsFieldName } : {}) }]
        : [],
    };
    const outcome =
      ruleOutcomeKind === 'label'
        ? {
            kind: 'label' as const,
            code: declaration.features.length + 1,
            label,
            tone: 'success' as const,
          }
        : {
            kind: 'formula' as const,
            label,
            unit: ruleUnit.trim() || undefined,
            expr:
              ruleFormulaOp === 'field'
                ? ({ op: 'field', field: targetField } as const)
                : ({
                    op: ruleFormulaOp,
                    left: { op: 'field', field: targetField },
                    right: { op: 'const', value: numericRight },
                  } as const),
          };
    const statement =
      ruleStatement.trim() ||
      (ruleHasCondition
        ? `When ${targetField} ${COMPARE_LABELS[ruleOp]} ${rhsDisplay}, answer ${label}.`
        : `Default outcome: ${label}.`);

    const ok = editingRuleId
      ? await onDispatchOp({
          op: 'update-feature',
          id: editingRuleId,
          patch: { statement, condition, outcome },
        })
      : await onDispatchOp({
          op: 'admit-feature',
          feature: {
            id: nextAppId,
            level: 'build',
            statement,
            condition,
            outcome,
            priority: declaration.features.length + 1,
            citation: { kind: 'req', requirementId: firstReqId },
          },
        });
    if (ok) {
      setRuleStatement('');
      setEditingRuleId(null);
      setAddingRule(false);
    }
  };

  const handleStartEditRule = (featId: string) => {
    const feat = declaration.features.find((f) => f.id === featId);
    if (!feat) return;
    setEditingRuleId(feat.id);
    setRuleStatement(feat.statement);
    const firstPred = feat.condition.all[0];
    if (firstPred) {
      setRuleHasCondition(true);
      setRuleField(firstPred.field);
      setRuleOp(firstPred.op);
      if (firstPred.rhsField) {
        setRuleRhsMode('field');
        setRuleRhsField(firstPred.rhsField);
      } else {
        setRuleRhsMode('const');
        setRuleThreshold(String(firstPred.value));
      }
    } else {
      setRuleHasCondition(false);
    }
    setRuleOutcomeKind(feat.outcome.kind);
    setRuleOutcomeLabel(feat.outcome.label);
    if (feat.outcome.kind === 'formula') {
      setRuleUnit(feat.outcome.unit ?? '');
      if (feat.outcome.expr.op === 'field') {
        setRuleFormulaOp('field');
      } else if (
        feat.outcome.expr.op === 'pct' ||
        feat.outcome.expr.op === 'add' ||
        feat.outcome.expr.op === 'sub' ||
        feat.outcome.expr.op === 'mul'
      ) {
        setRuleFormulaOp(feat.outcome.expr.op);
        if (feat.outcome.expr.right.op === 'const') {
          setRuleFormulaRight(String(feat.outcome.expr.right.value));
        }
      }
    }
    setAddingRule(true);
  };

  const handleMoveFeature = async (featId: string, dir: -1 | 1) => {
    const ordered = [...declaration.features].sort(
      (a, b) => a.priority - b.priority || a.id.localeCompare(b.id),
    );
    const idx = ordered.findIndex((f) => f.id === featId);
    const target = idx + dir;
    if (idx < 0 || target < 0 || target >= ordered.length) return;
    const ids = ordered.map((f) => f.id);
    [ids[idx], ids[target]] = [ids[target], ids[idx]];
    await onDispatchOp({ op: 'reorder-features', orderedIds: ids });
  };

  const handleAddRefusal = async (e: FormEvent) => {
    e.preventDefault();
    if (!refusalText.trim()) return;
    const targetField = fields.some((f) => f.name === refusalField)
      ? refusalField
      : (fields[0]?.name ?? 'amount');
    const numericVal = Math.max(0, Number.parseInt(refusalValue, 10) || 0);
    const newId = nextAppId;

    // Build an automatic matching refusal example vector so R5 passes right away!
    const sampleInputs: Record<string, number> = {};
    for (const f of fields) sampleInputs[f.name] = f.defaultValue;
    if (refusalOp === 'eq' || refusalOp === 'lte' || refusalOp === 'gte') sampleInputs[targetField] = numericVal;
    else if (refusalOp === 'lt') sampleInputs[targetField] = Math.max(0, numericVal - 1);
    else sampleInputs[targetField] = numericVal + 1;

    const ok = await onDispatchOp({
      op: 'batch',
      ops: [
        {
          op: 'admit-refusal',
          refusal: {
            id: newId,
            statement: `Refuse when ${targetField} ${COMPARE_LABELS[refusalOp]} ${numericVal}.`,
            condition: { all: [{ field: targetField, op: refusalOp, value: numericVal }] },
            failureText: refusalText.trim(),
            fieldHint: targetField,
            negativeId: newId,
            citation: { kind: 'req', requirementId: firstReqId },
          },
        },
        {
          op: 'admit-vector',
          vector: {
            id: `vec_${newId.toLowerCase().replace(/[^a-z0-9]+/g, '_')}`,
            inputs: sampleInputs,
            expectedResponse: refusalText.trim(),
            forFeatureId: newId,
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
    const merged: Record<string, number> = {};
    for (const f of fields) {
      merged[f.name] = vecInputs[f.name] ?? f.defaultValue;
    }
    const line = formatRequestLine(declaration.encoding, merged);
    const evaluated = evaluateDeclaredModel(declaration, line);
    const ok = await onDispatchOp({
      op: 'admit-vector',
      vector: {
        id: `vec_${declaration.vectors.length + 1}`,
        inputs: merged,
        request: line,
        expectedResponse: evaluated.response,
        forFeatureId: evaluated.matchedId ?? declaration.features[0]?.id ?? 'APP-01',
        isRefusal: evaluated.isRefusal,
      },
    });
    if (ok) setAddingVector(false);
  };

  const handleAddField = async (e: FormEvent) => {
    e.preventDefault();
    const label = newFieldLabel.trim() || newFieldName.trim();
    if (!label) return;
    const rawName = (newFieldName.trim() || label).toLowerCase().replace(/[^a-z0-9]+/g, '_');
    const choices =
      newFieldKind === 'choice'
        ? newFieldChoices
            .split(',')
            .map((s) => s.trim())
            .filter(Boolean)
        : undefined;
    const nextField: EncodedField = {
      name: rawName,
      label,
      kind: newFieldKind,
      min: 0,
      max:
        newFieldKind === 'choice'
          ? Math.max(0, (choices?.length ?? 2) - 1)
          : newFieldKind === 'bool'
            ? 1
            : Math.max(1, Number.parseInt(newFieldMax, 10) || 10000),
      choices,
      unit: newFieldUnit.trim() || undefined,
      defaultValue: 0,
    };
    const nextFields = [...fields, nextField];
    const ok = await onDispatchOp({
      op: 'batch',
      ops: [
        {
          op: 'propose-encoding',
          name: declaration.encoding?.name ?? 'Input',
          fields: nextFields,
          citation: { kind: 'req', requirementId: firstReqId },
        },
        {
          op: 'declare-view',
          view: {
            ...declaration.view,
            elements: [
              ...declaration.view.elements,
              {
                id: `el_${rawName}`,
                slot: newFieldKind === 'choice' ? 'choice-group' : 'field-input',
                label,
                binding: rawName,
                requirementId: firstReqId,
              },
            ],
          },
        },
      ],
    });
    if (ok) {
      setNewFieldName('');
      setNewFieldLabel('');
      setNewFieldUnit('');
      setAddingField(false);
    }
  };

  const handleRemoveField = async (fieldName: string) => {
    if (fields.length <= 1) return;
    const nextFields = fields.filter((f) => f.name !== fieldName);
    await onDispatchOp({
      op: 'batch',
      ops: [
        {
          op: 'propose-encoding',
          name: declaration.encoding?.name ?? 'Input',
          fields: nextFields,
          citation: { kind: 'req', requirementId: firstReqId },
        },
        {
          op: 'declare-view',
          view: {
            ...declaration.view,
            elements: declaration.view.elements.filter((el) => el.binding !== fieldName),
          },
        },
      ],
    });
  };

  const handleApplyRawSchema = async () => {
    try {
      const parsed = JSON.parse(rawSchemaDraft) as JsonSchema2020Object;
      const ok = await onDispatchOp({
        op: 'propose-encoding',
        name: declaration.encoding?.name ?? 'Input',
        schema: parsed,
        citation: { kind: 'req', requirementId: firstReqId },
      });
      if (ok) setShowRawSchema(false);
    } catch {
      // Invalid JSON ignored until valid
    }
  };

  const handleRecordDecision = async (e: FormEvent) => {
    e.preventDefault();
    if (!decisionQuote.trim()) return;
    const ok = await onDispatchOp({
      op: 'record-decision',
      stakeholder: 'owner',
      quote: decisionQuote.trim(),
      requirementIds: [firstReqId],
    });
    if (ok) setDecisionQuote('');
  };

  const handleSubmitNl = (e: FormEvent) => {
    e.preventDefault();
    if (!nlPrompt.trim() || proposingOps) return;
    onProposeOpsFromPrompt(nlPrompt.trim());
    setNlPrompt('');
  };

  const tabs: readonly { id: BuilderTab; label: string; icon: typeof Layers; badge?: number }[] = [
    { id: 'features', label: 'Features', icon: Layers, badge: declaration.features.length + declaration.refusals.length },
    { id: 'data', label: 'Data', icon: Database, badge: fields.length },
    { id: 'design', label: 'Design', icon: Palette },
    { id: 'capabilities', label: 'Policy', icon: Shield },
  ];

  return (
    <div className="flex h-full min-h-0 flex-col bg-[#FBFBF9] dark:bg-[#0D1117]">
      {/* Builder Navigation Tabs */}
      <div className="flex shrink-0 items-center justify-between gap-2 border-b border-zinc-200 px-3 py-2 dark:border-zinc-800">
        <div className="flex items-center gap-1">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium transition-colors ${
                  active
                    ? 'bg-emerald-700 text-white shadow-xs dark:bg-emerald-600'
                    : 'text-zinc-600 hover:bg-zinc-200/70 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-100'
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                <span>{tab.label}</span>
                {tab.badge !== undefined && (
                  <span
                    className={`rounded-full px-1.5 py-0.2 text-[10px] ${
                      active
                        ? 'bg-white/20 text-white'
                        : 'bg-zinc-200 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-300'
                    }`}
                  >
                    {tab.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>
        <div className="relative flex items-center gap-1">
          <button
            type="button"
            onClick={() => setExamplesOpen((v) => !v)}
            title="Optional reference examples (opens in a separate Prism)"
            className="inline-flex items-center gap-1 rounded-md border border-zinc-200 bg-white px-2 py-1 text-[11px] font-medium text-zinc-600 hover:bg-zinc-100 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300 dark:hover:bg-zinc-800"
          >
            <BookOpen className="h-3 w-3" />
            <span className="hidden sm:inline">Examples</span>
          </button>
          <button
            type="button"
            onClick={onImportHoloFile}
            title="Import any binary .holo archive to verify and run offline"
            className="inline-flex items-center gap-1 rounded-md border border-zinc-200 bg-white px-2 py-1 text-[11px] font-medium text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300 dark:hover:bg-zinc-800"
          >
            <Upload className="h-3 w-3" />
            <span className="hidden sm:inline">Import</span>
          </button>
          {examplesOpen && (
            <div className="absolute right-0 top-full z-30 mt-1.5 w-64 rounded-lg border border-zinc-200 bg-white p-2 shadow-lg dark:border-zinc-700 dark:bg-zinc-900">
              <div className="mb-1.5 flex items-center justify-between px-1">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
                  Reference Examples
                </span>
                <button
                  type="button"
                  onClick={() => setExamplesOpen(false)}
                  className="text-[10px] text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200"
                >
                  Close
                </button>
              </div>
              <div className="space-y-1">
                {DECLARATIVE_STARTERS.map((s) => (
                  <button
                    key={s.id}
                    type="button"
                    onClick={() => {
                      setExamplesOpen(false);
                      onLoadStarter(s.id);
                    }}
                    className="w-full rounded-md px-2 py-1.5 text-left transition-colors hover:bg-zinc-100 dark:hover:bg-zinc-800"
                  >
                    <div className="flex items-center justify-between gap-1">
                      <span className="text-xs font-medium text-zinc-800 dark:text-zinc-100">
                        {s.label}
                      </span>
                      <span className="rounded bg-zinc-100 px-1 py-0.5 text-[10px] text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400">
                        {s.preset}
                      </span>
                    </div>
                    <p className="mt-0.5 line-clamp-1 text-[10px] text-zinc-500 dark:text-zinc-400">
                      {s.description}
                    </p>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* API Refusal Banner */}
      {apiError && (
        <div className="mx-3 mt-2 flex items-start justify-between gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-200">
          <div className="min-w-0">
            <p className="font-semibold">
              API Refused <code className="font-mono text-[11px]">{apiError.code}</code>
            </p>
            <p className="mt-0.5">{apiError.message}</p>
            <p className="mt-0.5 text-[11px] opacity-80">{apiError.remediation}</p>
          </div>
          <button
            type="button"
            onClick={onClearApiError}
            className="shrink-0 rounded px-1.5 py-0.5 text-[11px] hover:bg-red-100 dark:hover:bg-red-900/60"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Live Derived Agenda (`next`) Strip */}
      {nextAgenda && (
        <div className="mx-3 mt-2 flex items-center justify-between gap-2 rounded-lg border border-amber-200/90 bg-amber-50/80 px-3 py-2 text-xs text-amber-900 dark:border-amber-900/50 dark:bg-amber-950/30 dark:text-amber-200">
          <div className="flex min-w-0 items-center gap-2">
            <Wrench className="h-3.5 w-3.5 shrink-0 text-amber-600 dark:text-amber-400" />
            <div className="min-w-0">
              <p className="truncate font-medium">Next: {nextAgenda.title}</p>
              <p className="truncate text-[11px] opacity-80">{nextAgenda.detail}</p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => {
              if (nextAgenda.suggestedOp) {
                void onDispatchOp(nextAgenda.suggestedOp);
              } else {
                setActiveTab(nextAgenda.targetTab);
              }
            }}
            className="inline-flex shrink-0 items-center gap-1 rounded-md bg-amber-700 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-amber-800 dark:bg-amber-600 dark:hover:bg-amber-500"
          >
            <span>{nextAgenda.actionLabel}</span>
            <ArrowRight className="h-3 w-3" />
          </button>
        </div>
      )}

      {/* Tab Body */}
      <div className="flex-1 space-y-4 overflow-y-auto px-3 py-3">
        {activeTab === 'features' && (
          <>
            {/* Rules & Outcomes */}
            <section className="space-y-2">
              <div className="flex items-center justify-between">
                <h3 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                  Declared Rules ({declaration.features.length})
                </h3>
                <button
                  type="button"
                  onClick={() => setAddingRule((v) => !v)}
                  className="inline-flex items-center gap-1 rounded-md border border-zinc-200 bg-white px-2 py-1 text-[11px] font-medium text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200 dark:hover:bg-zinc-800"
                >
                  <Plus className="h-3 w-3" />
                  Add Rule
                </button>
              </div>

              {addingRule && (
                <form
                  onSubmit={handleAddRule}
                  className="space-y-2 rounded-lg border border-emerald-200 bg-emerald-50/50 p-2.5 text-xs dark:border-emerald-900/50 dark:bg-emerald-950/20"
                >
                  <div className="flex items-center gap-2">
                    <label className="inline-flex items-center gap-1.5 text-[11px] font-medium text-zinc-700 dark:text-zinc-300">
                      <input
                        type="checkbox"
                        checked={ruleHasCondition}
                        onChange={(e) => setRuleHasCondition(e.target.checked)}
                      />
                      When condition holds
                    </label>
                  </div>
                  {ruleHasCondition && fields.length > 0 && (() => {
                    const selectedField = fields.find((f) => f.name === ruleField) ?? fields[0];
                    return (
                      <div className="space-y-1.5">
                        <div className="grid grid-cols-3 gap-1.5">
                          <select
                            value={ruleField}
                            onChange={(e) => setRuleField(e.target.value)}
                            className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                          >
                            {fields.map((f) => (
                              <option key={f.name} value={f.name}>
                                {f.label}
                              </option>
                            ))}
                          </select>
                          <select
                            value={ruleOp}
                            onChange={(e) => setRuleOp(e.target.value as CompareOp)}
                            className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                          >
                            {(['gte', 'gt', 'lte', 'lt', 'eq', 'neq'] as CompareOp[]).map((op) => (
                              <option key={op} value={op}>
                                {COMPARE_LABELS[op]}
                              </option>
                            ))}
                          </select>
                          {ruleRhsMode === 'field' ? (
                            <select
                              value={ruleRhsField}
                              onChange={(e) => setRuleRhsField(e.target.value)}
                              className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                            >
                              {fields.map((f) => (
                                <option key={f.name} value={f.name}>
                                  {f.label}
                                </option>
                              ))}
                            </select>
                          ) : selectedField?.kind === 'choice' && selectedField.choices ? (
                            <select
                              value={ruleThreshold}
                              onChange={(e) => setRuleThreshold(e.target.value)}
                              className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                            >
                              {selectedField.choices.map((c, idx) => (
                                <option key={idx} value={String(idx)}>
                                  {c}
                                </option>
                              ))}
                            </select>
                          ) : selectedField?.kind === 'bool' ? (
                            <select
                              value={ruleThreshold}
                              onChange={(e) => setRuleThreshold(e.target.value)}
                              className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                            >
                              <option value="1">Yes</option>
                              <option value="0">No</option>
                            </select>
                          ) : (
                            <input
                              type="number"
                              min={0}
                              value={ruleThreshold}
                              onChange={(e) => setRuleThreshold(e.target.value)}
                              className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                            />
                          )}
                        </div>
                        {fields.length > 1 && (
                          <div className="flex items-center gap-2 text-[11px] text-zinc-500">
                            <span>Compare to:</span>
                            <button
                              type="button"
                              onClick={() => setRuleRhsMode('const')}
                              className={`rounded px-1.5 py-0.5 ${ruleRhsMode === 'const' ? 'bg-emerald-700 text-white' : 'bg-zinc-200/70 dark:bg-zinc-800'}`}
                            >
                              Value
                            </button>
                            <button
                              type="button"
                              onClick={() => setRuleRhsMode('field')}
                              className={`rounded px-1.5 py-0.5 ${ruleRhsMode === 'field' ? 'bg-emerald-700 text-white' : 'bg-zinc-200/70 dark:bg-zinc-800'}`}
                            >
                              Another Field
                            </button>
                          </div>
                        )}
                      </div>
                    );
                  })()}
                  <div className="grid grid-cols-2 gap-1.5">
                    <select
                      value={ruleOutcomeKind}
                      onChange={(e) => setRuleOutcomeKind(e.target.value as 'label' | 'formula')}
                      className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                    >
                      <option value="label">Return Verdict Label</option>
                      <option value="formula">Compute Formula</option>
                    </select>
                    <input
                      type="text"
                      value={ruleOutcomeLabel}
                      onChange={(e) => setRuleOutcomeLabel(e.target.value)}
                      placeholder="Outcome label (e.g. Approved)"
                      className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                    />
                  </div>
                  {ruleOutcomeKind === 'formula' && (
                    <div className="grid grid-cols-3 gap-1.5">
                      <select
                        value={ruleFormulaOp}
                        onChange={(e) => setRuleFormulaOp(e.target.value as typeof ruleFormulaOp)}
                        className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                      >
                        <option value="pct">% of {ruleField}</option>
                        <option value="add">{ruleField} + N</option>
                        <option value="sub">{ruleField} - N</option>
                        <option value="mul">{ruleField} × N</option>
                        <option value="field">Echo {ruleField}</option>
                      </select>
                      <input
                        type="number"
                        min={0}
                        value={ruleFormulaRight}
                        onChange={(e) => setRuleFormulaRight(e.target.value)}
                        disabled={ruleFormulaOp === 'field'}
                        className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs disabled:opacity-40 dark:border-zinc-700 dark:bg-zinc-900"
                      />
                      <input
                        type="text"
                        value={ruleUnit}
                        onChange={(e) => setRuleUnit(e.target.value)}
                        placeholder="Unit (e.g. USD)"
                        className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                      />
                    </div>
                  )}
                  <input
                    type="text"
                    value={ruleStatement}
                    onChange={(e) => setRuleStatement(e.target.value)}
                    placeholder="Optional plain-English rule statement"
                    className="w-full rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                  />
                  <div className="flex justify-end gap-1.5">
                    <button
                      type="button"
                      onClick={() => {
                        setEditingRuleId(null);
                        setAddingRule(false);
                      }}
                      className="rounded px-2 py-1 text-[11px] text-zinc-600 hover:bg-zinc-200/70 dark:text-zinc-400 dark:hover:bg-zinc-800"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      className="rounded bg-emerald-700 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-emerald-800"
                    >
                      {editingRuleId ? `Save ${editingRuleId}` : `Admit Rule (${nextAppId})`}
                    </button>
                  </div>
                </form>
              )}

              <div className="space-y-1.5">
                {[...declaration.features]
                  .sort((a, b) => a.priority - b.priority || a.id.localeCompare(b.id))
                  .map((feat, idx, arr) => {
                  const locked = declaration.lockedIds.includes(feat.id);
                  const condText =
                    feat.condition.all.length === 0
                      ? 'Default fallback'
                      : feat.condition.all
                          .map((p) => {
                            const f = fields.find((x) => x.name === p.field);
                            const rhs = p.rhsField
                              ? p.rhsField
                              : f?.kind === 'choice' && f.choices?.[p.value]
                                ? f.choices[p.value]
                                : f?.kind === 'bool'
                                  ? p.value === 1 ? 'Yes' : 'No'
                                  : String(p.value);
                            return `${p.field} ${COMPARE_LABELS[p.op]} ${rhs}`;
                          })
                          .join(' ∧ ');
                  return (
                    <div
                      key={feat.id}
                      className="flex items-start justify-between gap-2 rounded-lg border border-zinc-200 bg-white px-2.5 py-2 text-xs dark:border-zinc-800 dark:bg-zinc-900/70"
                    >
                      <div className="min-w-0">
                        <div className="flex flex-wrap items-center gap-1.5">
                          <span className="rounded bg-emerald-100 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300">
                            {feat.id}
                          </span>
                          <span className="text-[11px] font-medium text-zinc-700 dark:text-zinc-200">
                            {condText}
                          </span>
                          <span className="text-zinc-400">→</span>
                          <span className="font-semibold text-zinc-900 dark:text-zinc-100">
                            {feat.outcome.label}
                          </span>
                        </div>
                        <p className="mt-0.5 text-[11px] text-zinc-500 dark:text-zinc-400">
                          {feat.statement}
                        </p>
                      </div>
                      <div className="flex shrink-0 items-center gap-0.5">
                        {arr.length > 1 && (
                          <>
                            <button
                              type="button"
                              disabled={idx === 0}
                              onClick={() => void handleMoveFeature(feat.id, -1)}
                              title="Evaluate earlier"
                              className="rounded p-1 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 disabled:opacity-30 dark:hover:bg-zinc-800"
                            >
                              <ArrowUp className="h-3 w-3" />
                            </button>
                            <button
                              type="button"
                              disabled={idx === arr.length - 1}
                              onClick={() => void handleMoveFeature(feat.id, 1)}
                              title="Evaluate later"
                              className="rounded p-1 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 disabled:opacity-30 dark:hover:bg-zinc-800"
                            >
                              <ArrowDown className="h-3 w-3" />
                            </button>
                          </>
                        )}
                        <button
                          type="button"
                          onClick={() => handleStartEditRule(feat.id)}
                          title="Refine rule via update-feature"
                          className="rounded p-1 text-zinc-400 hover:bg-zinc-100 hover:text-emerald-700 dark:hover:bg-zinc-800"
                        >
                          <Pencil className="h-3 w-3" />
                        </button>
                        {!locked && (
                          <button
                            type="button"
                            onClick={() => void onDispatchOp({ op: 'remove-feature', id: feat.id })}
                            title="Remove draft rule"
                            className="rounded p-1 text-zinc-400 hover:bg-zinc-100 hover:text-red-600 dark:hover:bg-zinc-800"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>

            {/* Typed Refusals (R5) */}
            <section className="space-y-2">
              <div className="flex items-center justify-between">
                <h3 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                  Safety Refusals ({declaration.refusals.length})
                </h3>
                <button
                  type="button"
                  onClick={() => setAddingRefusal((v) => !v)}
                  className="inline-flex items-center gap-1 rounded-md border border-zinc-200 bg-white px-2 py-1 text-[11px] font-medium text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200 dark:hover:bg-zinc-800"
                >
                  <Plus className="h-3 w-3" />
                  Add Refusal
                </button>
              </div>

              {addingRefusal && fields.length > 0 && (
                <form
                  onSubmit={handleAddRefusal}
                  className="space-y-2 rounded-lg border border-red-200 bg-red-50/40 p-2.5 text-xs dark:border-red-900/50 dark:bg-red-950/20"
                >
                  <div className="grid grid-cols-3 gap-1.5">
                    <select
                      value={refusalField}
                      onChange={(e) => setRefusalField(e.target.value)}
                      className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                    >
                      {fields.map((f) => (
                        <option key={f.name} value={f.name}>
                          {f.label}
                        </option>
                      ))}
                    </select>
                    <select
                      value={refusalOp}
                      onChange={(e) => setRefusalOp(e.target.value as CompareOp)}
                      className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                    >
                      {(['eq', 'lt', 'lte', 'gt', 'gte'] as CompareOp[]).map((op) => (
                        <option key={op} value={op}>
                          {COMPARE_LABELS[op]}
                        </option>
                      ))}
                    </select>
                    <input
                      type="number"
                      min={0}
                      value={refusalValue}
                      onChange={(e) => setRefusalValue(e.target.value)}
                      className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                    />
                  </div>
                  <input
                    type="text"
                    required
                    value={refusalText}
                    onChange={(e) => setRefusalText(e.target.value)}
                    placeholder="Exact refusal message (e.g. Amount must be positive.)"
                    className="w-full rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                  />
                  <div className="flex justify-end gap-1.5">
                    <button
                      type="button"
                      onClick={() => setAddingRefusal(false)}
                      className="rounded px-2 py-1 text-[11px] text-zinc-600 dark:text-zinc-400"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      className="rounded bg-red-700 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-red-800"
                    >
                      Admit Refusal ({nextAppId})
                    </button>
                  </div>
                </form>
              )}

              <div className="space-y-1.5">
                {declaration.refusals.map((ref) => {
                  const locked = declaration.lockedIds.includes(ref.id);
                  const condText = ref.condition.all
                    .map((p) => `${p.field} ${COMPARE_LABELS[p.op]} ${p.value}`)
                    .join(' ∧ ');
                  return (
                    <div
                      key={ref.id}
                      className="flex items-start justify-between gap-2 rounded-lg border border-red-200/70 bg-white px-2.5 py-2 text-xs dark:border-red-950/60 dark:bg-zinc-900/70"
                    >
                      <div className="min-w-0">
                        <div className="flex items-center gap-1.5">
                          <span className="rounded bg-red-100 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-red-700 dark:bg-red-950/70 dark:text-red-300">
                            {ref.id}
                          </span>
                          <span className="text-[11px] font-medium text-zinc-600 dark:text-zinc-300">
                            When {condText}
                          </span>
                        </div>
                        <p className="mt-0.5 text-[11px] font-medium text-red-700 dark:text-red-300">
                          “{ref.failureText}”
                        </p>
                      </div>
                      {!locked && (
                        <button
                          type="button"
                          onClick={() => void onDispatchOp({ op: 'remove-refusal', id: ref.id })}
                          className="shrink-0 rounded p-1 text-zinc-400 hover:bg-zinc-100 hover:text-red-600 dark:hover:bg-zinc-800"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
            </section>

            {/* Acceptance Examples (Live <1ms Check) */}
            <section className="space-y-2">
              <div className="flex items-center justify-between">
                <h3 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                  Examples ({checkReport.vectorsPassed}/{checkReport.vectorsTotal} passing)
                </h3>
                <button
                  type="button"
                  onClick={() => {
                    const init: Record<string, number> = {};
                    for (const f of fields) init[f.name] = f.defaultValue;
                    setVecInputs(init);
                    setAddingVector((v) => !v);
                  }}
                  className="inline-flex items-center gap-1 rounded-md border border-zinc-200 bg-white px-2 py-1 text-[11px] font-medium text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200 dark:hover:bg-zinc-800"
                >
                  <Plus className="h-3 w-3" />
                  Add Example
                </button>
              </div>

              {addingVector && fields.length > 0 && (
                <form
                  onSubmit={handleAddVector}
                  className="space-y-2 rounded-lg border border-zinc-200 bg-white p-2.5 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                >
                  <div className="grid grid-cols-2 gap-1.5">
                    {fields.map((f) => (
                      <label key={f.name} className="space-y-0.5 text-[11px] text-zinc-600 dark:text-zinc-300">
                        <span>{f.label}</span>
                        {f.kind === 'choice' && f.choices ? (
                          <select
                            value={vecInputs[f.name] ?? f.defaultValue}
                            onChange={(e) =>
                              setVecInputs((prev) => ({
                                ...prev,
                                [f.name]: Number.parseInt(e.target.value, 10) || 0,
                              }))
                            }
                            className="w-full rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-950"
                          >
                            {f.choices.map((c, idx) => (
                              <option key={idx} value={idx}>
                                {c}
                              </option>
                            ))}
                          </select>
                        ) : f.kind === 'bool' ? (
                          <select
                            value={vecInputs[f.name] ?? f.defaultValue}
                            onChange={(e) =>
                              setVecInputs((prev) => ({
                                ...prev,
                                [f.name]: Number.parseInt(e.target.value, 10) || 0,
                              }))
                            }
                            className="w-full rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-950"
                          >
                            <option value={1}>Yes</option>
                            <option value={0}>No</option>
                          </select>
                        ) : (
                          <input
                            type="number"
                            min={0}
                            value={vecInputs[f.name] ?? f.defaultValue}
                            onChange={(e) =>
                              setVecInputs((prev) => ({
                                ...prev,
                                [f.name]: Math.max(0, Number.parseInt(e.target.value, 10) || 0),
                              }))
                            }
                            className="w-full rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-950"
                          />
                        )}
                      </label>
                    ))}
                  </div>
                  <p className="text-[11px] text-zinc-500">
                    Model answers:{' '}
                    <strong className="text-zinc-800 dark:text-zinc-100">
                      {evaluateDeclaredModel(declaration, formatRequestLine(declaration.encoding, vecInputs)).response}
                    </strong>
                  </p>
                  <div className="flex justify-end gap-1.5">
                    <button
                      type="button"
                      onClick={() => setAddingVector(false)}
                      className="rounded px-2 py-1 text-[11px] text-zinc-500"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      className="rounded bg-emerald-700 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-emerald-800"
                    >
                      Save Example
                    </button>
                  </div>
                </form>
              )}

              <div className="space-y-1">
                {checkReport.vectorResults.map((vr) => (
                  <div
                    key={vr.id}
                    className="flex items-center justify-between gap-2 rounded-lg border border-zinc-200 bg-white px-2.5 py-1.5 text-[11px] dark:border-zinc-800 dark:bg-zinc-900/70"
                  >
                    <div className="flex min-w-0 items-center gap-1.5">
                      {vr.passed ? (
                        <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-emerald-600 dark:text-emerald-400" />
                      ) : (
                        <XCircle className="h-3.5 w-3.5 shrink-0 text-red-600 dark:text-red-400" />
                      )}
                      <code className="rounded bg-zinc-100 px-1 py-0.5 font-mono text-[10px] dark:bg-zinc-800">
                        {vr.request}
                      </code>
                      <span className="text-zinc-400">→</span>
                      <span className="truncate font-medium text-zinc-800 dark:text-zinc-200">
                        {vr.expected}
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={() => void onDispatchOp({ op: 'remove-vector', id: vr.id })}
                      className="shrink-0 rounded p-0.5 text-zinc-400 hover:text-red-600"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                ))}
              </div>
            </section>

            {/* Requirements */}
            <section className="space-y-2">
              <div className="flex items-center justify-between">
                <h3 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                  Requirements ({declaration.requirements.length})
                </h3>
                <button
                  type="button"
                  onClick={() => setAddingReq((v) => !v)}
                  className="inline-flex items-center gap-1 rounded-md border border-zinc-200 bg-white px-2 py-1 text-[11px] font-medium text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200 dark:hover:bg-zinc-800"
                >
                  <Plus className="h-3 w-3" />
                  Add Requirement
                </button>
              </div>
              {addingReq && (
                <form
                  onSubmit={handleAddRequirement}
                  className="space-y-2 rounded-lg border border-zinc-200 bg-white p-2.5 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                >
                  <input
                    type="text"
                    required
                    value={reqStatement}
                    onChange={(e) => setReqStatement(e.target.value)}
                    placeholder="What must this Prism guarantee?"
                    className="w-full rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-950"
                  />
                  <div className="flex items-center justify-between gap-2">
                    <select
                      value={reqOrigin}
                      onChange={(e) => setReqOrigin(e.target.value as RequirementOrigin)}
                      className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-950"
                    >
                      <option value="identity">Origin: Creator Identity</option>
                      <option value="refinement">Origin: User Refinement</option>
                      <option value="constraint">Origin: Policy Constraint</option>
                    </select>
                    <button
                      type="submit"
                      className="rounded bg-emerald-700 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-emerald-800"
                    >
                      Admit {nextReqId}
                    </button>
                  </div>
                </form>
              )}
              <div className="space-y-1">
                {declaration.requirements.map((req) => (
                  <div
                    key={req.id}
                    className="rounded-lg border border-zinc-200 bg-white px-2.5 py-1.5 text-[11px] dark:border-zinc-800 dark:bg-zinc-900/70"
                  >
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono text-[10px] font-semibold text-zinc-700 dark:text-zinc-300">
                        {req.id}
                      </span>
                      <span className="rounded bg-zinc-100 px-1.5 py-0.5 text-[10px] text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400">
                        {req.origin}
                      </span>
                    </div>
                    <p className="mt-0.5 text-zinc-600 dark:text-zinc-300">{req.statement}</p>
                  </div>
                ))}
              </div>
            </section>
          </>
        )}

        {activeTab === 'data' && (
          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-xs font-semibold text-zinc-800 dark:text-zinc-100">
                  Input Fields ({fields.length})
                </h3>
                <p className="text-[11px] text-zinc-500">
                  Declared once via <code className="font-mono">propose-encoding</code>; compiled into model bounds &amp; form inputs.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setAddingField((v) => !v)}
                className="inline-flex items-center gap-1 rounded-md border border-zinc-200 bg-white px-2 py-1 text-[11px] font-medium text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200"
              >
                <Plus className="h-3 w-3" />
                Add Field
              </button>
            </div>

            {addingField && (
              <form
                onSubmit={handleAddField}
                className="space-y-2 rounded-lg border border-emerald-200 bg-emerald-50/40 p-2.5 text-xs dark:border-emerald-900/50 dark:bg-emerald-950/20"
              >
                <div className="grid grid-cols-2 gap-1.5">
                  <input
                    type="text"
                    required
                    value={newFieldLabel}
                    onChange={(e) => setNewFieldLabel(e.target.value)}
                    placeholder="Label (e.g. Discount Rate)"
                    className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                  />
                  <select
                    value={newFieldKind}
                    onChange={(e) => setNewFieldKind(e.target.value as FieldKind)}
                    className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                  >
                    <option value="nat">Whole Number</option>
                    <option value="choice">Choice List</option>
                    <option value="bool">Yes / No Toggle</option>
                  </select>
                </div>
                {newFieldKind === 'nat' && (
                  <div className="grid grid-cols-2 gap-1.5">
                    <input
                      type="number"
                      min={1}
                      value={newFieldMax}
                      onChange={(e) => setNewFieldMax(e.target.value)}
                      placeholder="Max bound"
                      className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                    />
                    <input
                      type="text"
                      value={newFieldUnit}
                      onChange={(e) => setNewFieldUnit(e.target.value)}
                      placeholder="Unit (optional, e.g. USD)"
                      className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                    />
                  </div>
                )}
                {newFieldKind === 'choice' && (
                  <input
                    type="text"
                    value={newFieldChoices}
                    onChange={(e) => setNewFieldChoices(e.target.value)}
                    placeholder="Comma-separated choices (e.g. Food, Home, Travel)"
                    className="w-full rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                  />
                )}
                <div className="flex justify-end gap-1.5">
                  <button
                    type="button"
                    onClick={() => setAddingField(false)}
                    className="rounded px-2 py-1 text-[11px] text-zinc-500"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="rounded bg-emerald-700 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-emerald-800"
                  >
                    Propose Encoding
                  </button>
                </div>
              </form>
            )}

            <div className="space-y-1.5">
              {fields.map((f) => (
                <div
                  key={f.name}
                  className="flex items-center justify-between gap-2 rounded-lg border border-zinc-200 bg-white px-2.5 py-2 text-xs dark:border-zinc-800 dark:bg-zinc-900/70"
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="font-semibold text-zinc-800 dark:text-zinc-100">{f.label}</span>
                      <code className="rounded bg-zinc-100 px-1 py-0.5 font-mono text-[10px] text-zinc-500 dark:bg-zinc-800">
                        {f.name}
                      </code>
                      <span className="rounded bg-emerald-50 px-1.5 py-0.5 text-[10px] font-medium text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300">
                        {f.kind === 'nat' ? `${f.min}..${f.max}${f.unit ? ` ${f.unit}` : ''}` : f.kind === 'choice' ? `${f.choices?.length ?? 0} choices` : 'yes/no'}
                      </span>
                    </div>
                    {f.choices && (
                      <p className="mt-0.5 text-[11px] text-zinc-500">{f.choices.join(' · ')}</p>
                    )}
                  </div>
                  {fields.length > 1 && (
                    <button
                      type="button"
                      onClick={() => void handleRemoveField(f.name)}
                      className="rounded p-1 text-zinc-400 hover:bg-zinc-100 hover:text-red-600 dark:hover:bg-zinc-800"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
              ))}
            </div>

            {/* Collapsible JSON Schema 2020-12 Editor */}
            <div className="rounded-lg border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900/60">
              <button
                type="button"
                onClick={() => {
                  setRawSchemaDraft(JSON.stringify(declaration.encoding?.schema ?? {}, null, 2));
                  setShowRawSchema((v) => !v);
                }}
                className="flex w-full items-center justify-between px-3 py-2 text-xs font-medium text-zinc-700 dark:text-zinc-300"
              >
                <span>JSON Schema 2020-12 Contract</span>
                <span className="text-[11px] text-zinc-400">{showRawSchema ? 'Hide' : 'Inspect / Edit'}</span>
              </button>
              {showRawSchema && (
                <div className="space-y-2 border-t border-zinc-200 p-2.5 dark:border-zinc-800">
                  <textarea
                    rows={8}
                    value={rawSchemaDraft}
                    onChange={(e) => setRawSchemaDraft(e.target.value)}
                    spellCheck={false}
                    className="w-full rounded border border-zinc-300 bg-zinc-50 p-2 font-mono text-[11px] dark:border-zinc-700 dark:bg-zinc-950"
                  />
                  <div className="flex justify-end">
                    <button
                      type="button"
                      onClick={() => void handleApplyRawSchema()}
                      className="rounded bg-emerald-700 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-emerald-800"
                    >
                      Apply Schema
                    </button>
                  </div>
                </div>
              )}
            </div>
          </section>
        )}

        {activeTab === 'design' && (
          <div className="space-y-4">
            {/* Theme Presets */}
            <section className="space-y-2">
              <h3 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                Theme Preset (prism.design/1)
              </h3>
              <div className="grid grid-cols-2 gap-1.5">
                {PRISM_THEME_PRESETS.map((preset) => {
                  const selected = declaration.theme.preset === preset.id;
                  return (
                    <button
                      key={preset.id}
                      type="button"
                      onClick={() =>
                        void onDispatchOp({
                          op: 'set-theme',
                          theme: { preset: preset.id },
                        })
                      }
                      className={`flex items-center justify-between rounded-lg border p-2 text-left transition-colors ${
                        selected
                          ? 'border-emerald-600 bg-emerald-50/60 dark:border-emerald-500 dark:bg-emerald-950/30'
                          : 'border-zinc-200 bg-white hover:border-zinc-300 dark:border-zinc-800 dark:bg-zinc-900/70'
                      }`}
                    >
                      <div className="min-w-0">
                        <div className="flex items-center gap-1 text-xs font-semibold text-zinc-800 dark:text-zinc-100">
                          <span>{preset.name}</span>
                          {selected && <Check className="h-3 w-3 text-emerald-600" />}
                        </div>
                        <p className="truncate text-[10px] text-zinc-500">{preset.summary}</p>
                      </div>
                      <div className="flex shrink-0 -space-x-1">
                        {preset.swatches.map((hex, i) => (
                          <span
                            key={i}
                            className="h-4 w-4 rounded-full border border-black/15"
                            style={{ backgroundColor: hex }}
                          />
                        ))}
                      </div>
                    </button>
                  );
                })}
              </div>
            </section>

            {/* Viewpoint & Header Copy */}
            <section className="space-y-2">
              <h3 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                Viewpoint &amp; Page Copy
              </h3>
              <div className="grid grid-cols-2 gap-1.5">
                {(
                  [
                    ['decision-form', 'Decision Form'],
                    ['calculator', 'Calculator'],
                    ['ledger-tracker', 'Ledger / Tracker'],
                    ['request-reply', 'Request & Reply'],
                  ] as readonly [PrismViewpoint, string][]
                ).map(([vp, label]) => (
                  <button
                    key={vp}
                    type="button"
                    onClick={() =>
                      void onDispatchOp({
                        op: 'declare-view',
                        view: { ...declaration.view, viewpoint: vp },
                      })
                    }
                    className={`rounded-lg border px-2.5 py-1.5 text-left text-xs font-medium ${
                      declaration.view.viewpoint === vp
                        ? 'border-emerald-600 bg-emerald-50/60 text-emerald-900 dark:border-emerald-500 dark:bg-emerald-950/30 dark:text-emerald-200'
                        : 'border-zinc-200 bg-white text-zinc-700 dark:border-zinc-800 dark:bg-zinc-900/70 dark:text-zinc-300'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <div className="grid gap-2 rounded-lg border border-zinc-200 bg-white p-2.5 text-xs dark:border-zinc-800 dark:bg-zinc-900/70">
                <label className="space-y-0.5">
                  <span className="text-[11px] text-zinc-500">Eyebrow</span>
                  <input
                    type="text"
                    value={declaration.view.eyebrow}
                    onChange={(e) =>
                      void onDispatchOp({
                        op: 'declare-view',
                        view: { ...declaration.view, eyebrow: e.target.value },
                      })
                    }
                    className="w-full rounded border border-zinc-200 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                  />
                </label>
                <label className="space-y-0.5">
                  <span className="text-[11px] text-zinc-500">Subtitle</span>
                  <input
                    type="text"
                    value={declaration.view.subtitle}
                    onChange={(e) =>
                      void onDispatchOp({
                        op: 'declare-view',
                        view: { ...declaration.view, subtitle: e.target.value },
                      })
                    }
                    className="w-full rounded border border-zinc-200 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                  />
                </label>
                <label className="space-y-0.5">
                  <span className="text-[11px] text-zinc-500">Primary Action Label</span>
                  <input
                    type="text"
                    value={declaration.view.submitLabel}
                    onChange={(e) =>
                      void onDispatchOp({
                        op: 'declare-view',
                        view: { ...declaration.view, submitLabel: e.target.value },
                      })
                    }
                    className="w-full rounded border border-zinc-200 bg-transparent px-2 py-1 text-xs dark:border-zinc-700"
                  />
                </label>
              </div>
            </section>

            {/* Traceable View Elements (REQ-D9) */}
            <section className="space-y-1.5">
              <div className="flex items-center justify-between">
                <h3 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                  Bound UI Elements ({declaration.view.elements.length})
                </h3>
                <button
                  type="button"
                  onClick={() => {
                    const hasHist = declaration.view.elements.some((e) => e.slot === 'history-list');
                    void onDispatchOp({
                      op: 'declare-view',
                      view: {
                        ...declaration.view,
                        elements: hasHist
                          ? declaration.view.elements.filter((e) => e.slot !== 'history-list')
                          : [
                              ...declaration.view.elements,
                              {
                                id: 'el_history',
                                slot: 'history-list',
                                label: 'Recent Evaluations',
                                binding: 'history',
                                requirementId: firstReqId,
                              },
                            ],
                      },
                    });
                  }}
                  className="text-[11px] font-medium text-emerald-700 hover:underline dark:text-emerald-400"
                >
                  {declaration.view.elements.some((e) => e.slot === 'history-list')
                    ? 'Hide History Card'
                    : '+ Add History Card'}
                </button>
              </div>
              <div className="space-y-1">
                {declaration.view.elements.map((el, idx) => (
                  <div
                    key={el.id}
                    className="flex items-center justify-between gap-2 rounded-lg border border-zinc-200 bg-white px-2.5 py-1.5 text-[11px] dark:border-zinc-800 dark:bg-zinc-900/70"
                  >
                    <div className="flex min-w-0 items-center gap-1.5">
                      <span className="truncate font-medium text-zinc-800 dark:text-zinc-100">{el.label}</span>
                      <span className="shrink-0 rounded bg-zinc-100 px-1.5 py-0.5 text-[10px] text-zinc-500 dark:bg-zinc-800">
                        {el.slot}
                      </span>
                    </div>
                    <div className="flex shrink-0 items-center gap-1">
                      <select
                        value={el.requirementId}
                        onChange={(e) =>
                          void onDispatchOp({
                            op: 'declare-view',
                            view: {
                              ...declaration.view,
                              elements: declaration.view.elements.map((item) =>
                                item.id === el.id ? { ...item, requirementId: e.target.value } : item,
                              ),
                            },
                          })
                        }
                        aria-label={`Requirement for ${el.label}`}
                        className="rounded border border-zinc-200 bg-transparent px-1 py-0.5 font-mono text-[10px] text-emerald-700 dark:border-zinc-700 dark:text-emerald-400"
                      >
                        {declaration.requirements.map((r) => (
                          <option key={r.id} value={r.id}>
                            {r.id}
                          </option>
                        ))}
                      </select>
                      <button
                        type="button"
                        disabled={idx === 0}
                        onClick={() => {
                          const next = [...declaration.view.elements];
                          [next[idx - 1], next[idx]] = [next[idx], next[idx - 1]];
                          void onDispatchOp({ op: 'declare-view', view: { ...declaration.view, elements: next } });
                        }}
                        className="rounded p-0.5 text-zinc-400 hover:bg-zinc-100 disabled:opacity-30 dark:hover:bg-zinc-800"
                        title="Move up"
                      >
                        <ArrowUp className="h-3 w-3" />
                      </button>
                      <button
                        type="button"
                        disabled={idx === declaration.view.elements.length - 1}
                        onClick={() => {
                          const next = [...declaration.view.elements];
                          [next[idx + 1], next[idx]] = [next[idx], next[idx + 1]];
                          void onDispatchOp({ op: 'declare-view', view: { ...declaration.view, elements: next } });
                        }}
                        className="rounded p-0.5 text-zinc-400 hover:bg-zinc-100 disabled:opacity-30 dark:hover:bg-zinc-800"
                        title="Move down"
                      >
                        <ArrowDown className="h-3 w-3" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          </div>
        )}

        {activeTab === 'capabilities' && (
          <div className="space-y-4">
            {/* Deny-by-default prism.toml Capabilities */}
            <section className="space-y-2">
              <div>
                <h3 className="text-xs font-semibold text-zinc-800 dark:text-zinc-100">
                  Capabilities (Deny by Default)
                </h3>
                <p className="text-[11px] text-zinc-500">
                  Declared in <code className="font-mono">prism.toml</code> and enforced by FK-07.
                </p>
              </div>
              <div className="space-y-1.5">
                {(
                  [
                    ['observer', 'Observer Stream', 'Emit OpenTelemetry spans & receive @Prism inputs'],
                    ['storage', 'Local Storage', 'Persist scoped per-Prism state in IndexedDB'],
                    ['browser', 'Browser Utilities', 'Clipboard copy & local file export'],
                    ['network', 'Internet Fetch', 'Denied by default; opt-in outbound requests'],
                  ] as const
                ).map(([key, title, desc]) => (
                  <label
                    key={key}
                    className="flex cursor-pointer items-center justify-between gap-2 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-xs dark:border-zinc-800 dark:bg-zinc-900/70"
                  >
                    <div>
                      <p className="font-medium text-zinc-800 dark:text-zinc-100">{title}</p>
                      <p className="text-[11px] text-zinc-500">{desc}</p>
                    </div>
                    <input
                      type="checkbox"
                      checked={declaration.capabilities[key]}
                      onChange={(e) =>
                        void onDispatchOp({
                          op: 'set-capabilities',
                          capabilities: { [key]: e.target.checked },
                        })
                      }
                      className="h-4 w-4 accent-emerald-600"
                    />
                  </label>
                ))}
              </div>
            </section>

            {/* Owner Decisions (`record-decision`) */}
            <section className="space-y-2">
              <h3 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                Owner Decisions ({declaration.decisions.length})
              </h3>
              <form onSubmit={handleRecordDecision} className="flex gap-1.5">
                <input
                  type="text"
                  value={decisionQuote}
                  onChange={(e) => setDecisionQuote(e.target.value)}
                  placeholder="Record a verbatim product decision…"
                  className="flex-1 rounded-lg border border-zinc-200 bg-white px-2.5 py-1.5 text-xs dark:border-zinc-700 dark:bg-zinc-900"
                />
                <button
                  type="submit"
                  className="rounded-lg bg-zinc-900 px-2.5 py-1.5 text-xs font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900"
                >
                  Record
                </button>
              </form>
              {declaration.decisions.map((dec) => (
                <div
                  key={dec.id}
                  className="rounded-lg border border-zinc-200 bg-white px-2.5 py-1.5 text-[11px] dark:border-zinc-800 dark:bg-zinc-900/70"
                >
                  <span className="font-mono text-[10px] font-semibold text-zinc-500">{dec.id}</span>
                  <p className="mt-0.5 italic text-zinc-700 dark:text-zinc-300">“{dec.quote}”</p>
                </div>
              ))}
            </section>

            {/* Self-Healing Error Ledger */}
            {declaration.errors.some((e) => !e.resolved) && (
              <section className="space-y-1.5">
                <h3 className="text-[11px] font-semibold uppercase tracking-wider text-red-600 dark:text-red-400">
                  Recorded Errors (Self-Healing)
                </h3>
                {declaration.errors
                  .filter((e) => !e.resolved)
                  .map((err) => (
                    <div
                      key={err.id}
                      className="rounded-lg border border-red-200 bg-red-50/60 p-2.5 text-xs dark:border-red-900/60 dark:bg-red-950/30"
                    >
                      <div className="flex items-center gap-1.5 font-semibold text-red-800 dark:text-red-200">
                        <AlertTriangle className="h-3.5 w-3.5" />
                        <code>{err.code}</code>
                      </div>
                      <p className="mt-1 text-[11px] text-red-700 dark:text-red-300">{err.refusal}</p>
                      <div className="mt-2 flex gap-1.5">
                        {err.suggestedRepair && (
                          <button
                            type="button"
                            onClick={async () => {
                              const ok = await onDispatchOp(err.suggestedRepair!);
                              if (ok) await onDispatchOp({ op: 'resolve-error', id: err.id });
                            }}
                            className="rounded bg-red-700 px-2 py-1 text-[11px] font-medium text-white hover:bg-red-800"
                          >
                            Apply Derived Repair
                          </button>
                        )}
                        <button
                          type="button"
                          onClick={() => void onDispatchOp({ op: 'resolve-error', id: err.id })}
                          className="rounded border border-red-300 px-2 py-1 text-[11px] text-red-700 dark:border-red-800 dark:text-red-300"
                        >
                          Mark Resolved
                        </button>
                      </div>
                    </div>
                  ))}
              </section>
            )}

            {/* API Audit Log */}
            <section className="space-y-1">
              <h3 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">
                API Operation Log ({declaration.auditLog.length})
              </h3>
              <div className="max-h-40 space-y-1 overflow-y-auto">
                {[...declaration.auditLog].reverse().slice(0, 12).map((entry) => (
                  <div
                    key={entry.id}
                    className="flex items-center justify-between gap-2 rounded border border-zinc-200/70 bg-white px-2 py-1 text-[11px] dark:border-zinc-800 dark:bg-zinc-900/50"
                  >
                    <span className="truncate text-zinc-700 dark:text-zinc-300">{entry.summary}</span>
                    <code className="shrink-0 font-mono text-[10px] text-zinc-400">{entry.op}</code>
                  </div>
                ))}
              </div>
            </section>
          </div>
        )}
      </div>

      {/* Proposed API Ops Confirmation Drawer */}
      {proposedOps && proposedOps.length > 0 && (
        <div className="mx-3 mb-2 rounded-lg border border-emerald-300 bg-emerald-50 p-2.5 text-xs dark:border-emerald-800 dark:bg-emerald-950/40">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-emerald-900 dark:text-emerald-200">
              Proposed {proposedOps.length} Declarative API Operation{proposedOps.length === 1 ? '' : 's'}
            </span>
            <div className="flex gap-1.5">
              <button
                type="button"
                onClick={onDismissProposedOps}
                className="rounded px-2 py-0.5 text-[11px] text-zinc-600 dark:text-zinc-400"
              >
                Discard
              </button>
              <button
                type="button"
                onClick={onApplyProposedOps}
                className="rounded bg-emerald-700 px-2.5 py-0.5 text-[11px] font-medium text-white hover:bg-emerald-800"
              >
                Apply via API
              </button>
            </div>
          </div>
          <ul className="mt-1.5 space-y-0.5 text-[11px] text-emerald-800 dark:text-emerald-300">
            {proposedOps.map((op, i) => (
              <li key={i} className="truncate font-mono">
                • {op.op}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Optional Compact NL -> Declarative API Helper */}
      <form
        onSubmit={handleSubmitNl}
        className="shrink-0 border-t border-zinc-200 bg-white px-3 py-2 dark:border-zinc-800 dark:bg-[#0B0F14]"
      >
        <div className="flex items-center gap-1.5">
          <Sparkles className="h-3.5 w-3.5 shrink-0 text-emerald-600 dark:text-emerald-400" />
          <input
            type="text"
            value={nlPrompt}
            onChange={(e) => setNlPrompt(e.target.value)}
            placeholder={
              modelStreamEnabled
                ? 'Optional: describe a rule to propose Studio API operations…'
                : 'Turn on Model Stream to translate English → Studio API ops'
            }
            className="min-w-0 flex-1 bg-transparent text-xs text-zinc-800 placeholder:text-zinc-400 focus:outline-none dark:text-zinc-100"
          />
          {modelStreamEnabled ? (
            <button
              type="submit"
              disabled={proposingOps || !nlPrompt.trim()}
              className="inline-flex shrink-0 items-center gap-1 rounded-md bg-emerald-700 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-emerald-800 disabled:opacity-40"
            >
              {proposingOps ? <Loader2 className="h-3 w-3 animate-spin" /> : 'Propose'}
            </button>
          ) : (
            <button
              type="button"
              onClick={onEnableModelStream}
              className="shrink-0 rounded-md border border-zinc-200 px-2 py-1 text-[11px] font-medium text-zinc-600 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
            >
              Enable AI
            </button>
          )}
        </div>
      </form>
    </div>
  );
};
