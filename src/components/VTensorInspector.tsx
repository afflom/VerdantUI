import { useState, type FC } from 'react';
import { CheckCircle2, Play, RotateCcw, ShieldCheck, X, XCircle } from 'lucide-react';
import { formatBytes } from '../engine/modelCatalog';
import type { SamplingConfig } from '../engine/llamaRuntime';
import type { PrecisionContractMode } from '../engine/performanceContract';
import type { WeightResidencyPolicy } from '../engine/vramWeightPager';
import type { EngineState, PromptFormatMode } from '../engine/workerClient';
import { VTENSOR_CONTRACT } from '../engine/vtensorProvider';

interface VTensorInspectorProps {
  open: boolean;
  onClose: () => void;
  engineState: EngineState;
  onUpdateSampling: (patch: Partial<SamplingConfig>) => void;
  onUpdatePromptMode: (mode: PromptFormatMode) => void;
  onUpdatePrecisionMode: (mode: PrecisionContractMode) => void;
  onUpdateWeightResidencyPolicy: (policy: WeightResidencyPolicy) => void;
  onResetSampling: () => void;
  onRunConformance: () => Promise<void>;
}

type InspectorTab = 'backend' | 'graph' | 'sampling';

export const VTensorInspector: FC<VTensorInspectorProps> = ({
  open,
  onClose,
  engineState,
  onUpdateSampling,
  onUpdatePromptMode,
  onUpdatePrecisionMode,
  onUpdateWeightResidencyPolicy,
  onResetSampling,
  onRunConformance,
}) => {
  const [tab, setTab] = useState<InspectorTab>('backend');

  if (!open) return null;

  const {
    telemetry,
    model,
    sampling,
    promptMode,
    precisionMode,
    weightResidencyPolicy,
    conformanceReport,
    conformanceRunning,
  } = engineState;
  const contract = telemetry.performanceContract;
  const vramWeights = telemetry.vramWeights;
  const unifiedStore = telemetry.unifiedStore;
  const frameBudget = telemetry.frameBudget;
  const webgpu = telemetry.webgpu;
  const threadPool = telemetry.threadPool;
  const prism = telemetry.prism;
  const residencyToggleDisabled =
    engineState.status === 'generating' ||
    engineState.status === 'downloading' ||
    engineState.status === 'loading';

  return (
    <>
      <div
        className="fixed inset-0 z-20 bg-slate-950/45 xl:hidden"
        onClick={onClose}
        aria-hidden="true"
      />
      <aside
        aria-label="vTensor GGML provider inspector"
        className="fixed inset-y-0 right-0 z-30 flex w-[348px] max-w-[92vw] flex-col border-l border-slate-200 bg-white shadow-xl xl:static xl:z-auto xl:shadow-none dark:border-slate-800 dark:bg-[#0F141C]"
      >
      <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3.5 dark:border-slate-800">
        <div>
          <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            vTensor Provider
          </h2>
          <p className="text-[11px] text-slate-500 dark:text-slate-400">
            GGML v{telemetry.ggmlAbiVersion} · Verdant Tensor {telemetry.verdantVersion}
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close vTensor inspector"
          className="inline-flex h-8 w-8 items-center justify-center rounded-md text-slate-500 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-100"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="border-b border-slate-200 px-3 py-2 dark:border-slate-800">
        <div
          role="tablist"
          aria-label="Inspector sections"
          className="grid grid-cols-3 gap-1 rounded-lg bg-slate-100 p-1 dark:bg-slate-900"
        >
          {(
            [
              ['backend', 'Backend'],
              ['graph', 'Graph'],
              ['sampling', 'Sampling'],
            ] as const
          ).map(([id, label]) => (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={tab === id}
              onClick={() => setTab(id)}
              className={`h-8 rounded-md text-xs font-medium transition-colors whitespace-nowrap ${
                tab === id
                  ? 'bg-white text-slate-900 shadow-xs dark:bg-[#1B2432] dark:text-white'
                  : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 text-xs">
        {tab === 'backend' && (
          <div className="space-y-5">
            <section>
              <div className="flex items-center gap-1.5 text-slate-900 dark:text-slate-100">
                <ShieldCheck className="h-4 w-4 text-emerald-700 dark:text-emerald-400" />
                <h3 className="font-semibold">Exclusive GGML Backend Lock</h3>
              </div>
              <p className="mt-1.5 leading-relaxed text-slate-600 dark:text-slate-400">
                llama.cpp graph execution and buffer allocation are bound exclusively to{' '}
                <span className="font-mono text-slate-900 dark:text-slate-100">
                  {VTENSOR_CONTRACT.providerRepo}
                </span>
                . Default GGML providers are unregistered.
              </p>

              <dl className="mt-3 divide-y divide-slate-100 rounded-lg border border-slate-200 dark:divide-slate-800 dark:border-slate-800">
                <div className="flex items-center justify-between px-3 py-2">
                  <dt className="text-slate-500 dark:text-slate-400">Active GGML device</dt>
                  <dd className="font-mono font-medium text-emerald-800 dark:text-emerald-400">
                    {telemetry.activeProvider}
                  </dd>
                </div>
                <div className="flex items-center justify-between gap-2 px-3 py-2">
                  <dt className="whitespace-nowrap text-slate-500 dark:text-slate-400">
                    Compute substrate
                  </dt>
                  <dd className="whitespace-nowrap text-right font-mono text-[11px] text-slate-900 dark:text-slate-100">
                    {telemetry.kernelAcceleration}
                  </dd>
                </div>
                <div className="flex items-center justify-between px-3 py-2">
                  <dt className="text-slate-500 dark:text-slate-400">Pinned llama.cpp</dt>
                  <dd className="font-mono text-slate-900 dark:text-slate-100">
                    {telemetry.llamaCppPinnedCommit.slice(0, 8)}
                  </dd>
                </div>
                <div className="flex items-center justify-between px-3 py-2">
                  <dt className="text-slate-500 dark:text-slate-400">CPU fallback ops</dt>
                  <dd className="font-mono font-semibold text-emerald-800 dark:text-emerald-400 tabular-nums">
                    {telemetry.cpuFallbackOps}
                  </dd>
                </div>
                <div className="flex items-center justify-between px-3 py-2">
                  <dt className="text-slate-500 dark:text-slate-400">Graph dispatches</dt>
                  <dd className="font-mono text-slate-900 dark:text-slate-100 tabular-nums">
                    {telemetry.totalGraphRuns.toLocaleString()} runs /{' '}
                    {telemetry.totalOpsExecuted.toLocaleString()} ops
                  </dd>
                </div>
                <div className="flex items-center justify-between px-3 py-2">
                  <dt className="text-slate-500 dark:text-slate-400">Compiled plan reuses</dt>
                  <dd className="font-mono text-slate-900 dark:text-slate-100 tabular-nums">
                    {telemetry.compiledGraphReuses.toLocaleString()}
                  </dd>
                </div>
                {frameBudget && (
                  <div className="flex items-center justify-between gap-2 px-3 py-2">
                    <dt className="whitespace-nowrap text-slate-500 dark:text-slate-400">
                      T7 Frame KPI (16.7ms)
                    </dt>
                    <dd
                      className={`whitespace-nowrap text-right font-mono text-[11px] tabular-nums ${
                        frameBudget.withinFrameBudget
                          ? 'text-emerald-800 dark:text-emerald-400'
                          : 'text-amber-700 dark:text-amber-400'
                      }`}
                    >
                      {frameBudget.medianStepUs > 0
                        ? `${(frameBudget.medianStepUs / 1000).toFixed(2)}ms · ${
                            frameBudget.yieldTransport === 'message-channel' ? 'MsgCh' : 'Timer'
                          } (${frameBudget.yieldsAvoided} skipped)`
                        : frameBudget.yieldTransport === 'message-channel'
                          ? 'MsgCh zero-clamp ready'
                          : 'Timer fallback'}
                    </dd>
                  </div>
                )}
                {threadPool && (
                  <div className="flex items-center justify-between gap-2 px-3 py-2">
                    <dt className="whitespace-nowrap text-slate-500 dark:text-slate-400">
                      T3 Multi-Core Pool
                    </dt>
                    <dd
                      title={threadPool.reason}
                      className={`truncate text-right font-mono text-[11px] tabular-nums ${
                        threadPool.active
                          ? 'text-emerald-800 dark:text-emerald-400'
                          : 'text-slate-600 dark:text-slate-400'
                      }`}
                    >
                      {threadPool.active
                        ? `${threadPool.workerCount}T SIMD · ${threadPool.parallelDispatches.toLocaleString()} par`
                        : '1T SIMD (single)'}
                    </dd>
                  </div>
                )}
                {webgpu && (
                  <div className="flex items-center justify-between gap-2 px-3 py-2">
                    <dt className="whitespace-nowrap text-slate-500 dark:text-slate-400">
                      vTensor WebGPU
                    </dt>
                    <dd
                      title={webgpu.reason}
                      className={`truncate text-right font-mono text-[11px] ${
                        webgpu.status === 'hardware-active'
                          ? 'text-emerald-800 dark:text-emerald-400'
                          : 'text-slate-600 dark:text-slate-400'
                      }`}
                    >
                      {webgpu.status === 'hardware-active'
                        ? `HW GPU · ${webgpu.adapterLabel}`
                        : webgpu.status === 'software-rejected'
                          ? 'SW GPU rejected (SIMD)'
                          : webgpu.status === 'pending'
                            ? 'Probing…'
                            : 'Unavailable (SIMD)'}
                    </dd>
                  </div>
                )}
              </dl>
            </section>

            {contract && (
              <section>
                <div className="flex items-center justify-between gap-2">
                  <h3 className="font-semibold text-slate-900 dark:text-slate-100">
                    Prism Performance Contract
                  </h3>
                  <span
                    className={`font-mono text-[10px] font-semibold tracking-wide ${
                      contract.verdict === 'UPHELD' || contract.verdict === 'EXCEEDS_ESTIMATE'
                        ? 'text-emerald-800 dark:text-emerald-400'
                        : contract.verdict === 'BELOW_CONTRACT'
                          ? 'text-amber-700 dark:text-amber-400'
                          : 'text-slate-500 dark:text-slate-400'
                    }`}
                  >
                    {contract.verdict === 'EXCEEDS_ESTIMATE' ? 'UPHELD+' : contract.verdict}
                  </span>
                </div>
                <dl className="mt-2 divide-y divide-slate-100 rounded-lg border border-slate-200 dark:divide-slate-800 dark:border-slate-800">
                  <div className="flex items-center justify-between gap-2 px-3 py-1.5">
                    <dt className="whitespace-nowrap text-slate-500 dark:text-slate-400">
                      Calibrated host
                    </dt>
                    <dd className="whitespace-nowrap font-mono text-[11px] tabular-nums text-slate-900 dark:text-slate-100">
                      {contract.env.memoryBandwidthGBs.toFixed(1)} GB/s ·{' '}
                      {contract.env.computeGops.toFixed(1)} GOPS
                    </dd>
                  </div>
                  <div className="flex items-center justify-between gap-2 px-3 py-1.5">
                    <dt className="whitespace-nowrap text-slate-500 dark:text-slate-400">
                      T1 Roofline bound
                    </dt>
                    <dd className="whitespace-nowrap font-mono text-[11px] tabular-nums text-slate-900 dark:text-slate-100">
                      {contract.rooflineDecodeTokSec.toFixed(1)} tok/s* ({contract.decodeRegime === 'memory-bound' ? 'mem' : 'cmp'})
                    </dd>
                  </div>
                  <div className="flex items-center justify-between gap-2 px-3 py-1.5">
                    <dt className="whitespace-nowrap text-slate-500 dark:text-slate-400">
                      T5 Observed / eff
                    </dt>
                    <dd className="whitespace-nowrap font-mono text-[11px] tabular-nums text-emerald-800 dark:text-emerald-400">
                      {contract.observedDecodeTokSec > 0
                        ? `${contract.observedDecodeTokSec.toFixed(1)} tok/s (${contract.rooflineEfficiencyPct}%)`
                        : 'Awaiting first turn'}
                    </dd>
                  </div>
                  <div className="flex items-center justify-between gap-2 px-3 py-1.5">
                    <dt className="whitespace-nowrap text-slate-500 dark:text-slate-400">
                      T2 Kernel / batch
                    </dt>
                    <dd className="whitespace-nowrap font-mono text-[11px] text-slate-900 dark:text-slate-100">
                      B*={contract.optimalPrefillBatchSize} · {contract.activationQuantLabel}
                    </dd>
                  </div>
                  {prism && (
                    <div className="flex items-center justify-between gap-2 px-3 py-1.5">
                      <dt className="whitespace-nowrap text-slate-500 dark:text-slate-400">
                        Prism route / cert
                      </dt>
                      <dd className="whitespace-nowrap font-mono text-[11px] tabular-nums text-emerald-800 dark:text-emerald-400">
                        {prism.activePrimaryRoute} · #{prism.scheduleDigest}
                      </dd>
                    </div>
                  )}
                  <div className="flex items-center justify-between gap-2 px-3 py-1.5">
                    <dt className="whitespace-nowrap text-slate-500 dark:text-slate-400">
                      T4 Capacity law
                    </dt>
                    <dd className="whitespace-nowrap font-mono text-[11px] tabular-nums text-slate-900 dark:text-slate-100">
                      {contract.hotFrameLimit}H/{contract.warmQ8Limit}W
                      {contract.coldQ4Enabled ? '/Q4' : ''}
                      {contract.swapQ4Enabled ? '+Sw' : ''} @{' '}
                      {contract.effectiveMaxContext.toLocaleString()} ctx
                    </dd>
                  </div>
                  <div className="flex items-center justify-between gap-2 px-3 py-1.5">
                    <dt className="whitespace-nowrap text-slate-500 dark:text-slate-400">
                      Precision policy
                    </dt>
                    <dd>
                      <div
                        role="group"
                        aria-label="Precision contract policy"
                        className="inline-grid grid-cols-2 gap-0.5 rounded-md bg-slate-100 p-0.5 dark:bg-slate-900"
                      >
                        {(
                          [
                            ['q8-vnni', 'Q8-VNNI'],
                            ['strict-f64', 'Strict F64'],
                          ] as const
                        ).map(([mode, label]) => (
                          <button
                            key={mode}
                            type="button"
                            aria-pressed={precisionMode === mode}
                            disabled={engineState.status === 'generating'}
                            onClick={() => onUpdatePrecisionMode(mode)}
                            className={`rounded px-1.5 py-0.5 font-mono text-[10px] font-medium transition-colors whitespace-nowrap disabled:opacity-50 ${
                              precisionMode === mode
                                ? 'bg-white text-slate-900 shadow-xs dark:bg-[#1B2432] dark:text-white'
                                : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200'
                            }`}
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                    </dd>
                  </div>
                </dl>
              </section>
            )}

            {vramWeights && (
              <section>
                <div className="flex items-center justify-between gap-2">
                  <h3 className="font-semibold text-slate-900 dark:text-slate-100">
                    OPFS-Backed vRAM Weights
                  </h3>
                  <span className="font-mono text-[11px] font-medium text-emerald-800 dark:text-emerald-400 tabular-nums">
                    {vramWeights.residencyMode === 'opfs-paged'
                      ? `${vramWeights.virtualToResidentRatio.toFixed(2)}× virtual`
                      : '1.00× resident'}
                  </span>
                </div>
                <dl className="mt-2 divide-y divide-slate-100 rounded-lg border border-slate-200 dark:divide-slate-800 dark:border-slate-800">
                  <div className="flex items-center justify-between gap-2 px-3 py-1.5">
                    <dt className="whitespace-nowrap text-slate-500 dark:text-slate-400">
                      Backing / mode
                    </dt>
                    <dd className="whitespace-nowrap font-mono text-[11px] text-slate-900 dark:text-slate-100">
                      {vramWeights.backingKind === 'opfs-sync' ? 'OPFS Sync' : 'Memory'} ·{' '}
                      {vramWeights.residencyMode === 'opfs-paged' ? 'OPFS Paged' : 'Resident'}
                    </dd>
                  </div>
                  <div className="flex items-center justify-between gap-2 px-3 py-1.5">
                    <dt className="whitespace-nowrap text-slate-500 dark:text-slate-400">
                      Logical / resident
                    </dt>
                    <dd className="whitespace-nowrap font-mono text-[11px] tabular-nums text-slate-900 dark:text-slate-100">
                      {formatBytes(vramWeights.logicalWeightBytes)} /{' '}
                      {formatBytes(vramWeights.residentWeightBytes)}
                    </dd>
                  </div>
                  <div className="flex items-center justify-between gap-2 px-3 py-1.5">
                    <dt className="whitespace-nowrap text-slate-500 dark:text-slate-400">
                      Layer frames / pin
                    </dt>
                    <dd className="whitespace-nowrap font-mono text-[11px] tabular-nums text-slate-900 dark:text-slate-100">
                      {vramWeights.layerFrameSlots}/{vramWeights.totalLayers}L ·{' '}
                      {vramWeights.evictionPolicy === 'belady-cyclic'
                        ? `${vramWeights.pinnedLayerCount}P Belady`
                        : `${vramWeights.avgCoalescedExtentsPerLayer} ext/L`}
                    </dd>
                  </div>
                  <div className="flex items-center justify-between gap-2 px-3 py-1.5">
                    <dt className="whitespace-nowrap text-slate-500 dark:text-slate-400">
                      Faults / reclaims
                    </dt>
                    <dd className="whitespace-nowrap font-mono text-[11px] tabular-nums text-slate-900 dark:text-slate-100">
                      {vramWeights.layerFaults}F / {vramWeights.slotReclaims}R
                      {vramWeights.avgBackingReadMBs > 0
                        ? ` · ${vramWeights.avgBackingReadMBs.toFixed(0)} MB/s`
                        : ''}
                    </dd>
                  </div>
                  <div className="flex items-center justify-between gap-2 px-3 py-1.5">
                    <dt className="whitespace-nowrap text-slate-500 dark:text-slate-400">
                      Embd / LM-head
                    </dt>
                    <dd className="whitespace-nowrap font-mono text-[11px] text-slate-900 dark:text-slate-100">
                      {vramWeights.streamEmbeddings ? 'OPFS row' : 'Pinned'} /{' '}
                      {vramWeights.streamLmHead ? 'OPFS stripe' : 'Pinned'}
                    </dd>
                  </div>
                  <div className="flex items-center justify-between gap-2 px-3 py-1.5">
                    <dt className="whitespace-nowrap text-slate-500 dark:text-slate-400">
                      Weight residency
                    </dt>
                    <dd>
                      <div
                        role="group"
                        aria-label="vRAM weight residency policy"
                        className="inline-grid grid-cols-2 gap-0.5 rounded-md bg-slate-100 p-0.5 dark:bg-slate-900"
                      >
                        {(
                          [
                            ['auto', 'Auto (T4)'],
                            ['paged', 'OPFS Paged'],
                          ] as const
                        ).map(([policy, label]) => (
                          <button
                            key={policy}
                            type="button"
                            aria-pressed={weightResidencyPolicy === policy}
                            disabled={residencyToggleDisabled}
                            onClick={() => onUpdateWeightResidencyPolicy(policy)}
                            className={`rounded px-1.5 py-0.5 font-mono text-[10px] font-medium transition-colors whitespace-nowrap disabled:opacity-50 ${
                              weightResidencyPolicy === policy
                                ? 'bg-white text-slate-900 shadow-xs dark:bg-[#1B2432] dark:text-white'
                                : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200'
                            }`}
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                    </dd>
                  </div>
                </dl>
              </section>
            )}

            {unifiedStore && (
              <section>
                <div className="flex items-center justify-between gap-2">
                  <h3 className="font-semibold text-slate-900 dark:text-slate-100">
                    Unified vRAM Store (T6)
                  </h3>
                  <span className="font-mono text-[11px] font-medium text-emerald-800 dark:text-emerald-400 tabular-nums">
                    {unifiedStore.speculativeActive
                      ? 'Prefilling…'
                      : unifiedStore.lastTurnPrefillHitRatePct > 0
                        ? `${unifiedStore.lastTurnPrefillHitRatePct}% hit`
                        : `${unifiedStore.kvPrefixBlocks} blk${unifiedStore.kvPrefixBlocks === 1 ? '' : 's'}`}
                  </span>
                </div>
                <dl className="mt-2 divide-y divide-slate-100 rounded-lg border border-slate-200 dark:divide-slate-800 dark:border-slate-800">
                  <div className="flex items-center justify-between gap-2 px-3 py-1.5">
                    <dt className="whitespace-nowrap text-slate-500 dark:text-slate-400">
                      Unified / budget
                    </dt>
                    <dd className="whitespace-nowrap font-mono text-[11px] tabular-nums text-slate-900 dark:text-slate-100">
                      {formatBytes(unifiedStore.totalUnifiedResidentBytes)} /{' '}
                      {formatBytes(unifiedStore.globalBudgetBytes)}
                    </dd>
                  </div>
                  <div className="flex items-center justify-between gap-2 px-3 py-1.5">
                    <dt className="whitespace-nowrap text-slate-500 dark:text-slate-400">
                      Hot embd slab
                    </dt>
                    <dd className="whitespace-nowrap font-mono text-[11px] tabular-nums text-slate-900 dark:text-slate-100">
                      {unifiedStore.embdCacheSlots > 0
                        ? `${unifiedStore.embdCacheResidentCount}/${unifiedStore.embdCacheSlots} · ${unifiedStore.embdCacheHitRatePct}% hit`
                        : 'Pinned in RAM'}
                    </dd>
                  </div>
                  <div className="flex items-center justify-between gap-2 px-3 py-1.5">
                    <dt className="whitespace-nowrap text-slate-500 dark:text-slate-400">
                      KV prefix blocks
                    </dt>
                    <dd className="whitespace-nowrap font-mono text-[11px] tabular-nums text-slate-900 dark:text-slate-100">
                      {unifiedStore.kvPrefixBlocks}/{unifiedStore.kvPrefixMaxBlocks} ·{' '}
                      {unifiedStore.kvPrefixHits}H · {unifiedStore.kvPrefixDeduplicated}D
                      {unifiedStore.kvPrefixPersistedOpfs ? ' · OPFS' : ''}
                    </dd>
                  </div>
                  <div className="flex items-center justify-between gap-2 px-3 py-1.5">
                    <dt className="whitespace-nowrap text-slate-500 dark:text-slate-400">
                      Continuous prefill
                    </dt>
                    <dd className="whitespace-nowrap font-mono text-[11px] tabular-nums text-emerald-800 dark:text-emerald-400">
                      {unifiedStore.speculativeReadyTokens} ready ·{' '}
                      {unifiedStore.speculativePrefilledTokens} spec
                      {unifiedStore.estimatedTtftSavedMs > 0
                        ? ` · −${unifiedStore.estimatedTtftSavedMs}ms`
                        : ''}
                    </dd>
                  </div>
                  {unifiedStore.factoredSavedBytes > 0 && (
                    <div className="flex items-center justify-between gap-2 px-3 py-1.5">
                      <dt className="whitespace-nowrap text-slate-500 dark:text-slate-400">
                        T7 lossless factored
                      </dt>
                      <dd className="whitespace-nowrap font-mono text-[11px] tabular-nums text-emerald-800 dark:text-emerald-400">
                        {unifiedStore.factoredBlocks} blk · −{formatBytes(unifiedStore.factoredSavedBytes)}
                      </dd>
                    </div>
                  )}
                </dl>
              </section>
            )}

            {telemetry.vramKv && (
              <section>
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-slate-900 dark:text-slate-100">
                    vRAM Tiered KV Pager
                  </h3>
                  <span className="font-mono text-[11px] font-medium text-emerald-800 dark:text-emerald-400 tabular-nums">
                    {telemetry.vramKv.compressionRatio.toFixed(2)}× ratio
                  </span>
                </div>
                <dl className="mt-2 divide-y divide-slate-100 rounded-lg border border-slate-200 dark:divide-slate-800 dark:border-slate-800">
                  <div className="flex items-center justify-between px-3 py-1.5">
                    <dt className="text-slate-500 dark:text-slate-400">
                      {telemetry.vramKv.swapQ4Enabled
                        ? 'Hot / Warm / Q4 / Swap'
                        : telemetry.vramKv.coldQ4Enabled
                          ? 'Hot / Warm / Cold Q4'
                          : 'Hot F32 / Warm Q8'}
                    </dt>
                    <dd className="font-mono tabular-nums text-slate-900 dark:text-slate-100">
                      {telemetry.vramKv.hotResidentPages}/{telemetry.vramKv.hotFrameLimit}H ·{' '}
                      {telemetry.vramKv.warmQ8Pages}W
                      {telemetry.vramKv.coldQ4Enabled ? ` · ${telemetry.vramKv.coldQ4Pages}Q4` : ''}
                      {telemetry.vramKv.swapQ4Enabled ? ` · ${telemetry.vramKv.swappedQ4Pages}Sw` : ''}
                    </dd>
                  </div>
                  <div className="flex items-center justify-between px-3 py-1.5">
                    <dt className="text-slate-500 dark:text-slate-400">KV footprint / saved</dt>
                    <dd className="font-mono tabular-nums text-slate-900 dark:text-slate-100">
                      {formatBytes(telemetry.vramKv.allocatedBytes)} (−
                      {formatBytes(telemetry.vramKv.savedBytes)})
                    </dd>
                  </div>
                  <div className="flex items-center justify-between px-3 py-1.5">
                    <dt className="text-slate-500 dark:text-slate-400">
                      {telemetry.vramKv.coldQ4Enabled
                        ? 'Reclaims / Q4 cascades'
                        : 'Reclaims / refaults'}
                    </dt>
                    <dd className="font-mono tabular-nums text-slate-900 dark:text-slate-100">
                      {telemetry.vramKv.reclaimCycles} /{' '}
                      {telemetry.vramKv.coldQ4Enabled
                        ? telemetry.vramKv.tierCascadeCycles
                        : telemetry.vramKv.refaultCount}
                    </dd>
                  </div>
                  <div className="flex items-center justify-between px-3 py-1.5">
                    <dt className="text-slate-500 dark:text-slate-400">Residency bitset</dt>
                    <dd className="font-mono text-[10px] text-slate-700 dark:text-slate-300">
                      {telemetry.vramKv.residencyBitsetHex}
                    </dd>
                  </div>
                </dl>
              </section>
            )}

            <section>
              <h3 className="font-semibold text-slate-900 dark:text-slate-100">
                Disabled Default GGML Providers
              </h3>
              <ul className="mt-2 divide-y divide-slate-100 rounded-lg border border-slate-200 dark:divide-slate-800 dark:border-slate-800">
                {telemetry.disabledProviders.map((provider) => (
                  <li
                    key={provider}
                    className="flex items-center justify-between px-3 py-2 font-mono text-[11px]"
                  >
                    <span className="text-slate-600 dark:text-slate-400">{provider}</span>
                    <span className="text-slate-400 dark:text-slate-500">Blocked</span>
                  </li>
                ))}
              </ul>
            </section>

            <section>
              <h3 className="font-semibold text-slate-900 dark:text-slate-100">
                vTensor Aligned Buffers (32B)
              </h3>
              {telemetry.buffers.length === 0 ? (
                <p className="mt-2 text-slate-500 dark:text-slate-400">
                  No buffers allocated yet.
                </p>
              ) : (
                <ul className="mt-2 divide-y divide-slate-100 rounded-lg border border-slate-200 dark:divide-slate-800 dark:border-slate-800">
                  {telemetry.buffers.map((buf) => (
                    <li key={buf.id} className="px-3 py-2">
                      <div className="flex items-center justify-between">
                        <span className="font-mono font-medium text-slate-800 dark:text-slate-200">
                          {buf.id}
                        </span>
                        <span className="font-mono text-slate-700 dark:text-slate-300 tabular-nums">
                          {formatBytes(buf.usedBytes)}
                        </span>
                      </div>
                      <div className="mt-0.5 flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400 tabular-nums">
                        <span>{buf.tensorCount} tensors</span>
                        <span>cap {formatBytes(buf.capacityBytes)}</span>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </section>

            {model && (
              <section>
                <h3 className="font-semibold text-slate-900 dark:text-slate-100">
                  Loaded GGUF Architecture
                </h3>
                <dl className="mt-2 divide-y divide-slate-100 rounded-lg border border-slate-200 dark:divide-slate-800 dark:border-slate-800">
                  <div className="flex justify-between px-3 py-1.5">
                    <dt className="text-slate-500 dark:text-slate-400">Arch / RoPE</dt>
                    <dd className="font-mono text-slate-900 dark:text-slate-100">
                      {model.hparams.architecture} ({model.hparams.ropeType})
                    </dd>
                  </div>
                  <div className="flex justify-between px-3 py-1.5">
                    <dt className="text-slate-500 dark:text-slate-400">Tokenizer</dt>
                    <dd className="font-mono text-slate-900 dark:text-slate-100">
                      {model.hparams.tokenizerModel} ({model.hparams.preTokenizer})
                    </dd>
                  </div>
                  <div className="flex justify-between px-3 py-1.5">
                    <dt className="text-slate-500 dark:text-slate-400">Layers / Heads (KV)</dt>
                    <dd className="font-mono tabular-nums text-slate-900 dark:text-slate-100">
                      {model.hparams.blockCount}L · {model.hparams.headCount}/
                      {model.hparams.headCountKv}H
                    </dd>
                  </div>
                  <div className="flex justify-between px-3 py-1.5">
                    <dt className="text-slate-500 dark:text-slate-400">Embed / FFN dim</dt>
                    <dd className="font-mono tabular-nums text-slate-900 dark:text-slate-100">
                      {model.hparams.embeddingLength} / {model.hparams.feedForwardLength}
                    </dd>
                  </div>
                  <div className="flex justify-between px-3 py-1.5">
                    <dt className="text-slate-500 dark:text-slate-400">Vocab / Context</dt>
                    <dd className="font-mono tabular-nums text-slate-900 dark:text-slate-100">
                      {model.hparams.vocabSize.toLocaleString()} /{' '}
                      {model.maxContextWindow.toLocaleString()} tok
                    </dd>
                  </div>
                </dl>
              </section>
            )}

            <section>
              <div className="flex items-center justify-between gap-2">
                <h3 className="font-semibold text-slate-900 dark:text-slate-100">
                  Live Conformance Suite
                </h3>
                <button
                  type="button"
                  disabled={conformanceRunning}
                  onClick={() => void onRunConformance()}
                  className="inline-flex h-7 items-center gap-1 rounded-md border border-slate-200 bg-white px-2.5 text-[11px] font-medium text-slate-800 hover:bg-slate-50 disabled:opacity-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-200 dark:hover:bg-slate-800 transition-colors whitespace-nowrap"
                >
                  <Play className="h-3 w-3" />
                  <span>{conformanceRunning ? 'Running...' : 'Run Checks'}</span>
                </button>
              </div>
              {conformanceReport ? (
                <div className="mt-2 space-y-2">
                  <div className="flex items-center justify-between text-[11px] text-slate-600 dark:text-slate-400 font-mono tabular-nums">
                    <span>
                      {conformanceReport.passedCount}/{conformanceReport.totalCount} passed
                    </span>
                    <span>{conformanceReport.totalDurationMs.toFixed(1)} ms</span>
                  </div>
                  <ul className="divide-y divide-slate-100 rounded-lg border border-slate-200 dark:divide-slate-800 dark:border-slate-800">
                    {conformanceReport.checks.map((c) => (
                      <li key={c.id} className="px-2.5 py-2">
                        <div className="flex items-center justify-between gap-2">
                          <span className="inline-flex items-center gap-1.5 font-mono text-[11px] font-medium text-slate-900 dark:text-slate-100">
                            {c.passed ? (
                              <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-emerald-700 dark:text-emerald-400" />
                            ) : (
                              <XCircle className="h-3.5 w-3.5 shrink-0 text-red-600 dark:text-red-400" />
                            )}
                            <span>{c.id}</span>
                          </span>
                          <span className="font-mono text-[10px] text-slate-500 dark:text-slate-400 tabular-nums">
                            {c.passed ? 'PASS' : 'FAIL'} · {c.durationMs.toFixed(2)} ms
                          </span>
                        </div>
                        <p className="mt-0.5 pl-5 text-[11px] text-slate-500 dark:text-slate-400">
                          {c.detail}
                        </p>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : (
                <p className="mt-1.5 text-[11px] leading-relaxed text-slate-500 dark:text-slate-400">
                  Run VT-02 through VT-67 &amp; PRISM-01..05 checks directly inside the browser Web
                  Worker to verify exclusive provider lock, Prism NativeBackend &amp; CostSurface
                  certificates, Q8_K narrow-lane (Q4_K/Q6_K) &amp; uor-matmul tabulated kernels,
                  3-tier vRAM KV paging, OPFS Belady-cyclic weight paging, Unified Store (T6), and
                  hardware-only WebGPU admission.
                </p>
              )}
            </section>
          </div>
        )}

        {tab === 'graph' && (
          <div className="space-y-5">
            <section>
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-slate-900 dark:text-slate-100">
                  Verdant Kernel Dispatches
                </h3>
                <span className="font-mono text-[11px] text-slate-500 dark:text-slate-400 tabular-nums">
                  {telemetry.totalOpsExecuted.toLocaleString()} total
                </span>
              </div>

              {telemetry.opMetrics.length === 0 ? (
                <p className="mt-2 rounded-lg border border-slate-200 p-3 text-slate-500 dark:border-slate-800 dark:text-slate-400">
                  Send a prompt to inspect real per-operator Verdant Tensor dispatches from
                  llama.cpp&apos;s computation graph.
                </p>
              ) : (
                <div className="mt-2 overflow-hidden rounded-lg border border-slate-200 dark:border-slate-800">
                  <table className="w-full text-left font-mono text-[11px] tabular-nums">
                    <thead className="border-b border-slate-200 bg-slate-50 text-slate-500 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-400">
                      <tr>
                        <th className="px-2.5 py-1.5 font-medium">GGML Op</th>
                        <th className="px-2.5 py-1.5 text-right font-medium">Calls</th>
                        <th className="px-2.5 py-1.5 text-right font-medium">Time</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                      {telemetry.opMetrics.map((m) => (
                        <tr key={m.op}>
                          <td className="px-2.5 py-1.5 text-slate-800 dark:text-slate-200">
                            {m.op}
                          </td>
                          <td className="px-2.5 py-1.5 text-right text-slate-700 dark:text-slate-300">
                            {m.calls.toLocaleString()}
                          </td>
                          <td className="px-2.5 py-1.5 text-right text-slate-500 dark:text-slate-400">
                            {m.totalMs.toFixed(1)} ms
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>

            <section>
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-slate-900 dark:text-slate-100">
                  Last Token ggml_cgraph
                </h3>
                <span className="font-mono text-[11px] text-slate-500 dark:text-slate-400 tabular-nums">
                  {telemetry.lastGraphNodeCount} nodes
                </span>
              </div>
              {telemetry.lastGraphTrace.length === 0 ? (
                <p className="mt-2 text-slate-500 dark:text-slate-400">
                  No decode graph captured yet.
                </p>
              ) : (
                <ol className="mt-2 max-h-72 divide-y divide-slate-100 overflow-y-auto rounded-lg border border-slate-200 font-mono text-[11px] dark:divide-slate-800 dark:border-slate-800">
                  {telemetry.lastGraphTrace.map((node) => (
                    <li key={node.index} className="px-2.5 py-1.5">
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate font-medium text-slate-800 dark:text-slate-200">
                          {node.index.toString().padStart(2, '0')} {node.op}
                        </span>
                        <span className="shrink-0 text-[10px] text-emerald-800 dark:text-emerald-400">
                          {node.srcType}
                        </span>
                      </div>
                      <div className="mt-0.5 truncate text-[10px] text-slate-500 dark:text-slate-400">
                        {node.label} · {node.shapeDesc}
                      </div>
                    </li>
                  ))}
                </ol>
              )}
            </section>
          </div>
        )}

        {tab === 'sampling' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-slate-900 dark:text-slate-100">
                llama.cpp Sampler Parameters
              </h3>
              <button
                type="button"
                onClick={onResetSampling}
                className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium text-slate-600 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-100"
              >
                <RotateCcw className="h-3 w-3" />
                <span>Reset</span>
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <div className="flex justify-between">
                  <label
                    htmlFor="sampling-temperature"
                    className="font-medium text-slate-700 dark:text-slate-300"
                  >
                    Temperature
                  </label>
                  <span className="font-mono tabular-nums text-slate-900 dark:text-slate-100">
                    {sampling.temperature.toFixed(2)}
                  </span>
                </div>
                <input
                  id="sampling-temperature"
                  type="range"
                  min={0}
                  max={1.5}
                  step={0.05}
                  value={sampling.temperature}
                  onChange={(e) => onUpdateSampling({ temperature: Number(e.target.value) })}
                  className="mt-1.5 w-full accent-[#0F5132]"
                />
                <p className="mt-1 text-[11px] text-slate-500 dark:text-slate-400">
                  Set to 0.00 for deterministic greedy decoding.
                </p>
              </div>

              <div>
                <div className="flex justify-between">
                  <label
                    htmlFor="sampling-top-p"
                    className="font-medium text-slate-700 dark:text-slate-300"
                  >
                    Top-P (Nucleus)
                  </label>
                  <span className="font-mono tabular-nums text-slate-900 dark:text-slate-100">
                    {sampling.topP.toFixed(2)}
                  </span>
                </div>
                <input
                  id="sampling-top-p"
                  type="range"
                  min={0.1}
                  max={1}
                  step={0.05}
                  value={sampling.topP}
                  onChange={(e) => onUpdateSampling({ topP: Number(e.target.value) })}
                  className="mt-1.5 w-full accent-[#0F5132]"
                />
              </div>

              <div>
                <div className="flex justify-between">
                  <label
                    htmlFor="sampling-top-k"
                    className="font-medium text-slate-700 dark:text-slate-300"
                  >
                    Top-K
                  </label>
                  <span className="font-mono tabular-nums text-slate-900 dark:text-slate-100">
                    {sampling.topK}
                  </span>
                </div>
                <input
                  id="sampling-top-k"
                  type="range"
                  min={1}
                  max={100}
                  step={1}
                  value={sampling.topK}
                  onChange={(e) => onUpdateSampling({ topK: Number(e.target.value) })}
                  className="mt-1.5 w-full accent-[#0F5132]"
                />
              </div>

              <div>
                <div className="flex justify-between">
                  <label
                    htmlFor="sampling-repeat"
                    className="font-medium text-slate-700 dark:text-slate-300"
                  >
                    Repeat Penalty
                  </label>
                  <span className="font-mono tabular-nums text-slate-900 dark:text-slate-100">
                    {sampling.repeatPenalty.toFixed(2)}
                  </span>
                </div>
                <input
                  id="sampling-repeat"
                  type="range"
                  min={1}
                  max={1.5}
                  step={0.05}
                  value={sampling.repeatPenalty}
                  onChange={(e) => onUpdateSampling({ repeatPenalty: Number(e.target.value) })}
                  className="mt-1.5 w-full accent-[#0F5132]"
                />
              </div>

              <div>
                {(() => {
                  const maxCtx = Math.max(16, model?.maxContextWindow ?? 2048);
                  const isFullContext = Boolean(sampling.fullContextOutput);
                  const sliderStep = maxCtx > 8192 ? 128 : maxCtx > 2048 ? 64 : 16;
                  const clampedCustom = Math.min(Math.max(1, sampling.maxTokens), maxCtx);
                  return (
                    <>
                      <div className="flex items-center justify-between gap-2">
                        <label
                          htmlFor="sampling-max-tokens"
                          className="font-medium text-slate-700 dark:text-slate-300"
                        >
                          Max Output Tokens
                        </label>
                        <div className="flex items-center gap-1.5">
                          <button
                            type="button"
                            aria-pressed={isFullContext}
                            onClick={() =>
                              onUpdateSampling({
                                fullContextOutput: !isFullContext,
                                maxTokens: !isFullContext ? maxCtx : clampedCustom,
                              })
                            }
                            className={`rounded-md border px-2 py-0.5 font-mono text-[10px] font-medium transition-colors whitespace-nowrap ${
                              isFullContext
                                ? 'border-emerald-700/30 bg-emerald-50 text-emerald-800 dark:border-emerald-500/40 dark:bg-emerald-950/50 dark:text-emerald-300'
                                : 'border-slate-200 bg-white text-slate-600 hover:text-slate-900 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-400 dark:hover:text-slate-200'
                            }`}
                          >
                            Full context
                          </button>
                          <span className="font-mono tabular-nums text-slate-900 dark:text-slate-100">
                            {isFullContext ? maxCtx.toLocaleString() : clampedCustom.toLocaleString()}
                          </span>
                        </div>
                      </div>
                      <div className="mt-1.5 flex items-center gap-2">
                        <input
                          id="sampling-max-tokens"
                          type="range"
                          min={16}
                          max={maxCtx}
                          step={sliderStep}
                          value={isFullContext ? maxCtx : Math.max(16, clampedCustom)}
                          onChange={(e) => {
                            const nextVal = Number(e.target.value);
                            onUpdateSampling({
                              maxTokens: nextVal,
                              fullContextOutput: nextVal >= maxCtx,
                            });
                          }}
                          className="w-full accent-[#0F5132]"
                        />
                        <input
                          aria-label="Exact max output tokens"
                          type="number"
                          min={1}
                          max={maxCtx}
                          value={isFullContext ? maxCtx : clampedCustom}
                          onChange={(e) => {
                            const raw = Number(e.target.value);
                            if (!Number.isFinite(raw) || raw < 1) return;
                            const nextVal = Math.min(maxCtx, Math.round(raw));
                            onUpdateSampling({
                              maxTokens: nextVal,
                              fullContextOutput: nextVal >= maxCtx,
                            });
                          }}
                          className="h-7 w-20 shrink-0 rounded-md border border-slate-200 bg-white px-2 text-right font-mono text-[11px] tabular-nums text-slate-900 outline-none focus:border-emerald-700 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-100"
                        />
                      </div>
                      <p className="mt-1 text-[11px] text-slate-500 dark:text-slate-400">
                        {isFullContext
                          ? `Full-context mode: streams up to the model's entire ${maxCtx.toLocaleString()}-token window (stops naturally on EOS).`
                          : `Custom cap: up to ${clampedCustom.toLocaleString()} of ${maxCtx.toLocaleString()} context tokens.`}
                      </p>
                    </>
                  );
                })()}
              </div>

              <div className="border-t border-slate-100 pt-4 dark:border-slate-800">
                <span className="block font-medium text-slate-700 dark:text-slate-300">
                  Prompt Format
                </span>
                <div
                  role="group"
                  aria-label="Prompt format mode"
                  className="mt-2 grid grid-cols-4 gap-1 rounded-lg bg-slate-100 p-1 dark:bg-slate-900"
                >
                  {(
                    [
                      ['auto', 'Auto'],
                      ['chatml', 'ChatML'],
                      ['continuation', 'Story'],
                      ['instruct', 'Plain'],
                    ] as const
                  ).map(([id, label]) => (
                    <button
                      key={id}
                      type="button"
                      aria-pressed={promptMode === id}
                      onClick={() => onUpdatePromptMode(id)}
                      className={`h-7 rounded-md text-[11px] font-medium transition-colors whitespace-nowrap ${
                        promptMode === id
                          ? 'bg-white text-slate-900 shadow-xs dark:bg-[#1B2432] dark:text-white'
                          : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200'
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <p className="mt-1.5 text-[11px] text-slate-500 dark:text-slate-400">
                  Auto reads the GGUF chat_template (ChatML for SmolLM2 &amp; Qwen2.5, Llama-3,
                  Zephyr, or Story continuation).
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </aside>
    </>
  );
};
