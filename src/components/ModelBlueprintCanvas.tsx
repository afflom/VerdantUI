/**
 * Live Verdant Prism Model Blueprint Canvas (`src/components/ModelBlueprintCanvas.tsx`)
 *
 * Shown in the Studio right pane while a Prism is undefined or being defined (and accessible
 * anytime via the `Blueprint | Preview | Prism` switcher):
 * - Clearly shows an empty, calm starting state on "+ New Prism" (zero placeholder app UI).
 * - Renders a live ISO/PAS 19450 OPM Object–Process–Stakeholder SVG diagram as the Creator
 *   populates Stage 1.
 * - Visualizes the 4 mandatory Core Models, Trust Boundaries, and Deterministic Standards
 *   Applicability Routing (`REQ-D6` / `REQ-D11`), including an auditable "Not Applicable"
 *   drawer for excluded standards.
 */

import { useState, type FC } from 'react';
import {
  ArrowRight,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Circle,
  Compass,
  Cpu,
  Eye,
  Layers,
  Palette,
  Scale,
  Shield,
  Sparkles,
} from 'lucide-react';
import type { PrismDeclaration, StudioWizardStage } from '../engine/verdantStudioApi';
import type { StageProgressStatus, VerdantRoutingReport } from '../engine/verdantStandardsRouter';
import { PRISM_MEDIA_CATALOG } from '../engine/prismMediaOrgans';

interface ModelBlueprintCanvasProps {
  declaration: PrismDeclaration;
  routing: VerdantRoutingReport;
  activeStage: StudioWizardStage;
  onSelectStage: (stage: StudioWizardStage) => void;
  onOpenPreview: () => void;
}

function stageBadgeClass(status: StageProgressStatus): string {
  if (status === 'complete') {
    return 'border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-900/60 dark:bg-emerald-950/40 dark:text-emerald-300';
  }
  if (status === 'in-progress') {
    return 'border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-900/60 dark:bg-amber-950/40 dark:text-amber-300';
  }
  return 'border-zinc-200 bg-zinc-50 text-zinc-500 dark:border-zinc-800 dark:bg-zinc-900/60 dark:text-zinc-400';
}

export const ModelBlueprintCanvas: FC<ModelBlueprintCanvasProps> = ({
  declaration,
  routing,
  activeStage,
  onSelectStage,
  onOpenPreview,
}) => {
  const [excludedOpen, setExcludedOpen] = useState(false);

  const coreStages = routing.stages.filter((s) =>
    s.stage === 'concept' || s.stage === 'architecture' || s.stage === 'security' || s.stage === 'design',
  );
  const inputObjs = declaration.concept.opmObjects.filter(
    (o) => o.role === 'input' || o.role === 'stateful-asset',
  );
  const outputObjs = declaration.concept.opmObjects.filter((o) => o.role === 'output');
  const processes = declaration.concept.opmProcesses;
  const stakeholders = declaration.concept.stakeholders;

  const coreIcons: Record<string, typeof Compass> = {
    concept: Compass,
    architecture: Cpu,
    security: Shield,
    design: Palette,
  };

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-col overflow-y-auto bg-[#F8F9FA] p-5 dark:bg-[#090D12]">
      <div className="mx-auto w-full max-w-3xl space-y-5">
        {/* Top Status Header */}
        <div className="rounded-xl border border-zinc-200 bg-white p-4 shadow-2xs dark:border-zinc-800 dark:bg-[#0E131B]">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0 space-y-1">
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center gap-1 rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300">
                  <Layers className="h-3 w-3" />
                  Verdant Prism Blueprint
                </span>
                <span className="text-[11px] text-zinc-500">
                  Core Models: {routing.summary.coreComplete}/{routing.summary.coreTotal} complete
                </span>
              </div>
              <h2 className="text-base font-semibold text-zinc-900 dark:text-zinc-100">
                {routing.isUndefined
                  ? 'No Prism Defined Yet'
                  : declaration.metadata.name}
              </h2>
              <p className="text-xs leading-relaxed text-zinc-500 dark:text-zinc-400">
                {routing.isUndefined
                  ? 'Start in Stage 1 (Concept) to define your stakeholders, OPM objects & processes, and context weights. Verdant Studio will mechanically derive downstream facets and route only applicable standards.'
                  : declaration.concept.purpose ||
                    declaration.metadata.description ||
                    'Complete the 4 Core Models to synthesize the live preview and seal a proved .holo archive.'}
              </p>
            </div>
            <div className="flex shrink-0 items-center gap-2">
              {routing.readyToPreview ? (
                <button
                  type="button"
                  onClick={onOpenPreview}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-700 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-800"
                >
                  <Eye className="h-3.5 w-3.5" />
                  Open Live Preview
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => onSelectStage(routing.recommendedStage)}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-zinc-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900"
                >
                  <span>
                    {routing.isUndefined
                      ? 'Begin Stage 1: Concept'
                      : `Continue: ${routing.stages.find((s) => s.stage === routing.recommendedStage)?.shortLabel ?? 'Next'}`}
                  </span>
                  <ArrowRight className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
          </div>

          {/* 4 Core Model Spine Cards */}
          <div className="mt-4 grid gap-2 sm:grid-cols-2">
            {coreStages.map((st) => {
              const Icon = coreIcons[st.stage] ?? Compass;
              const isSelected = activeStage === st.stage;
              return (
                <button
                  key={st.stage}
                  type="button"
                  onClick={() => onSelectStage(st.stage)}
                  className={`flex flex-col items-start rounded-lg border p-3 text-left transition-colors ${
                    isSelected
                      ? 'border-emerald-500 bg-emerald-50/40 dark:border-emerald-600 dark:bg-emerald-950/20'
                      : 'border-zinc-200 bg-zinc-50/50 hover:border-zinc-300 dark:border-zinc-800 dark:bg-zinc-900/40 dark:hover:border-zinc-700'
                  }`}
                >
                  <div className="flex w-full items-center justify-between gap-2">
                    <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-zinc-900 dark:text-zinc-100">
                      <Icon className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
                      {st.number}. {st.label}
                    </span>
                    <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium ${stageBadgeClass(st.status)}`}>
                      {st.status === 'complete' ? (
                        <CheckCircle2 className="h-2.5 w-2.5" />
                      ) : (
                        <Circle className="h-2.5 w-2.5" />
                      )}
                      {st.completedItems}/{st.totalItems}
                    </span>
                  </div>
                  <span className="mt-1 text-[10px] text-zinc-500">{st.standardsLabel}</span>
                  {st.missing.length > 0 ? (
                    <span className="mt-1.5 line-clamp-1 text-[11px] text-zinc-600 dark:text-zinc-400">
                      Next: {st.missing[0]}
                    </span>
                  ) : (
                    <span className="mt-1.5 text-[11px] text-emerald-700 dark:text-emerald-400">
                      Core facet settled
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* Live ISO/PAS 19450 OPM Conceptual Diagram */}
        <div className="rounded-xl border border-zinc-200 bg-white p-4 shadow-2xs dark:border-zinc-800 dark:bg-[#0E131B]">
          <div className="flex items-center justify-between gap-2">
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
                Conceptual Model · ISO/PAS 19450 OPM &amp; PROV-O
              </h3>
              <p className="mt-0.5 text-[11px] text-zinc-500">
                Objects (rectangles), Processes (ellipses), and Stakeholders feed System Encodings &amp; Security Assets.
              </p>
            </div>
            <button
              type="button"
              onClick={() => onSelectStage('concept')}
              className="rounded-md border border-zinc-200 px-2 py-1 text-[11px] font-medium text-zinc-600 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
            >
              Edit Concept
            </button>
          </div>

          {stakeholders.length === 0 && inputObjs.length === 0 && processes.length === 0 && outputObjs.length === 0 ? (
            <div className="mt-3 flex flex-col items-center justify-center rounded-lg border border-dashed border-zinc-300 px-4 py-8 text-center dark:border-zinc-700">
              <Sparkles className="h-5 w-5 text-zinc-400" />
              <p className="mt-2 text-xs font-medium text-zinc-700 dark:text-zinc-300">
                OPM canvas is empty
              </p>
              <p className="mt-1 max-w-md text-[11px] text-zinc-500">
                Add Stakeholders, OPM Objects (inputs/outputs), and OPM Processes in Stage 1 to watch the conceptual model take shape.
              </p>
            </div>
          ) : (
            <div className="mt-3 grid gap-3 md:grid-cols-3">
              {/* Column 1: Inputs & Stakeholders */}
              <div className="space-y-2 rounded-lg border border-zinc-200/80 bg-zinc-50/70 p-2.5 dark:border-zinc-800 dark:bg-zinc-900/40">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
                  Input Objects ({inputObjs.length})
                </p>
                {inputObjs.length === 0 ? (
                  <p className="text-[11px] italic text-zinc-400">None declared yet</p>
                ) : (
                  inputObjs.map((obj) => (
                    <div
                      key={obj.id}
                      className="rounded border-2 border-emerald-600/60 bg-white px-2.5 py-1.5 dark:bg-zinc-950"
                    >
                      <div className="flex items-center justify-between gap-1">
                        <span className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">{obj.name}</span>
                        <span className="font-mono text-[9px] text-zinc-400">{obj.id}</span>
                      </div>
                      {obj.states.length > 0 && (
                        <div className="mt-1 flex flex-wrap gap-1">
                          {obj.states.map((st) => (
                            <span
                              key={st}
                              className="rounded-full border border-amber-400/60 bg-amber-50 px-1.5 py-0.5 text-[9px] text-amber-900 dark:bg-amber-950/40 dark:text-amber-200"
                            >
                              {st}
                            </span>
                          ))}
                        </div>
                      )}
                      {obj.unit && (
                        <p className="mt-0.5 text-[10px] text-zinc-500">
                          {obj.min ?? 0}..{obj.max ?? 10000} {obj.unit}
                        </p>
                      )}
                    </div>
                  ))
                )}
                {stakeholders.length > 0 && (
                  <div className="pt-1">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
                      Stakeholders ({stakeholders.length})
                    </p>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {stakeholders.map((s) => (
                        <span
                          key={s.id}
                          className="rounded bg-zinc-200/80 px-1.5 py-0.5 text-[10px] font-medium text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300"
                        >
                          {s.name} ({s.provRole})
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Column 2: Processes (OPM Ellipses) */}
              <div className="flex flex-col justify-center space-y-2 rounded-lg border border-zinc-200/80 bg-zinc-50/70 p-2.5 dark:border-zinc-800 dark:bg-zinc-900/40">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
                  OPM Processes ({processes.length})
                </p>
                {processes.length === 0 ? (
                  <p className="text-[11px] italic text-zinc-400">None declared yet</p>
                ) : (
                  processes.map((proc) => (
                    <div
                      key={proc.id}
                      className="rounded-full border-2 border-sky-600/70 bg-white px-3 py-2 text-center dark:bg-zinc-950"
                    >
                      <p className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">{proc.name}</p>
                      <p className="mt-0.5 line-clamp-2 text-[10px] text-zinc-500">{proc.summary}</p>
                    </div>
                  ))
                )}
                <div className="pt-1 text-center font-mono text-[10px] text-zinc-400">
                  {declaration.features.length} rule(s) · {declaration.refusals.length} refusal(s) · {declaration.vectors.length} vector(s)
                </div>
              </div>

              {/* Column 3: Output Objects & Bound View */}
              <div className="space-y-2 rounded-lg border border-zinc-200/80 bg-zinc-50/70 p-2.5 dark:border-zinc-800 dark:bg-zinc-900/40">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
                  Yielded Outputs ({outputObjs.length})
                </p>
                {outputObjs.length === 0 ? (
                  <p className="text-[11px] italic text-zinc-400">No output object yet</p>
                ) : (
                  outputObjs.map((obj) => (
                    <div
                      key={obj.id}
                      className="rounded border-2 border-emerald-600/60 bg-white px-2.5 py-1.5 dark:bg-zinc-950"
                    >
                      <span className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">{obj.name}</span>
                      {obj.states.length > 0 && (
                        <div className="mt-1 flex flex-wrap gap-1">
                          {obj.states.map((st) => (
                            <span
                              key={st}
                              className="rounded-full border border-emerald-400/60 bg-emerald-50 px-1.5 py-0.5 text-[9px] text-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-200"
                            >
                              {st}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))
                )}
                <div className="pt-1">
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
                    Compiled Interface
                  </p>
                  <p className="mt-0.5 text-[11px] text-zinc-600 dark:text-zinc-300">
                    {declaration.view?.proceduralOrgan
                      ? `${PRISM_MEDIA_CATALOG[declaration.view.proceduralOrgan.mediaType].label} · ${declaration.view.proceduralOrgan.activeOrgans.length} active studio feature(s) · Space: ${declaration.concept.compartmentId ?? 'personal'}`
                      : declaration.view?.authoredView
                        ? `Interactive App · bound to ${declaration.view.elements.length} model signal(s)`
                        : declaration.view
                          ? `${declaration.view.viewpoint} · ${declaration.view.elements.length} bound element(s)`
                          : 'Not synthesized yet (Design)'}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Deterministic Standards Applicability Routing Matrix */}
        <div className="rounded-xl border border-zinc-200 bg-white p-4 shadow-2xs dark:border-zinc-800 dark:bg-[#0E131B]">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                <Scale className="h-3.5 w-3.5 text-emerald-600" />
                Applicable Standards &amp; Compliance
              </h3>
              <p className="mt-0.5 text-[11px] text-zinc-500">
                Core weights route to 4 mandatory models + {routing.summary.activeLayeredCount} applicable layered standard(s).
              </p>
            </div>
            {routing.activeLayeredStandards.length > 0 && (
              <button
                type="button"
                onClick={() => onSelectStage('standards')}
                className="rounded-md border border-emerald-300 bg-emerald-50 px-2.5 py-1 text-[11px] font-medium text-emerald-800 hover:bg-emerald-100 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-200"
              >
                Review Routed Standards ({routing.summary.obligationsSettled}/{routing.summary.obligationsTotal})
              </button>
            )}
          </div>

          {routing.activeLayeredStandards.length === 0 ? (
            <div className="mt-3 rounded-lg border border-zinc-200 bg-zinc-50 px-3 py-2.5 text-[11px] text-zinc-600 dark:border-zinc-800 dark:bg-zinc-900/50 dark:text-zinc-400">
              No additional regulatory or domain standards are triggered by the current Core Model profile. Only the 4 mandatory Core Models apply.
            </div>
          ) : (
            <div className="mt-3 space-y-2">
              {routing.activeLayeredStandards.map((std) => (
                <div
                  key={std.authority.id}
                  className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-zinc-200 px-3 py-2 text-xs dark:border-zinc-800"
                >
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-zinc-900 dark:text-zinc-100">{std.authority.name}</span>
                      <code className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-[10px] text-zinc-500 dark:bg-zinc-800">
                        {std.authority.id}
                      </code>
                    </div>
                    <p className="mt-0.5 text-[11px] text-zinc-500">{std.reason}</p>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {std.triggeredBy.map((t) => (
                        <span
                          key={t}
                          className="rounded bg-emerald-50 px-1.5 py-0.5 font-mono text-[9px] text-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-300"
                        >
                          {t}
                        </span>
                      ))}
                    </div>
                  </div>
                  <span
                    className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                      std.settledCount === std.totalCount
                        ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300'
                        : 'bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300'
                    }`}
                  >
                    {std.settledCount}/{std.totalCount} settled
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Auditable Not-Applicable Ledger */}
          <div className="mt-3 border-t border-zinc-200 pt-2.5 dark:border-zinc-800">
            <button
              type="button"
              onClick={() => setExcludedOpen((v) => !v)}
              className="inline-flex items-center gap-1.5 text-[11px] font-medium text-zinc-500 hover:text-zinc-800 dark:text-zinc-400 dark:hover:text-zinc-200"
            >
              {excludedOpen ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
              Not Applicable ({routing.excludedStandards.length} standards excluded by Core Model)
            </button>
            {excludedOpen && (
              <div className="mt-2 space-y-1">
                {routing.excludedStandards.map((ex) => (
                  <div
                    key={ex.authority.id}
                    className="flex flex-col gap-0.5 rounded bg-zinc-50 px-2.5 py-1.5 text-[11px] dark:bg-zinc-900/60"
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-zinc-700 dark:text-zinc-300">{ex.authority.name}</span>
                      <code className="font-mono text-[10px] text-zinc-400">{ex.authority.id}</code>
                    </div>
                    <span className="text-zinc-500">{ex.exclusionRationale}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
