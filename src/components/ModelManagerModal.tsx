import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type FC,
  type FormEvent,
} from 'react';
import {
  Check,
  Code2,
  Download,
  FileJson,
  FolderUp,
  HardDrive,
  Layers,
  MessageSquare,
  RotateCw,
  Trash2,
  Upload,
  X,
  Zap,
} from 'lucide-react';
import {
  matchCatalogModelForHarness,
  parseHarnessConfigJson,
  serializeHarnessConfig,
  type ImportedHarnessSpec,
} from '../engine/harnessConfig';
import {
  formatBytes,
  type CatalogModel,
  type ModelCategory,
} from '../engine/modelCatalog';
import {
  observeCatalogScaling,
  type CatalogModelAssessment,
} from '../engine/verdantObserver';
import type { EngineState } from '../engine/workerClient';

export type ModalViewTab = ModelCategory | 'custom-ingest' | 'harness';

interface ModelManagerModalProps {
  open: boolean;
  initialTab?: ModalViewTab;
  onClose: () => void;
  engineState: EngineState;
  onSelectCatalogModel: (model: CatalogModel) => Promise<void>;
  onUploadLocalFile: (file: File) => Promise<void>;
  onLoadCustomUrl: (url: string) => Promise<void>;
  onImportHarness: (harness: ImportedHarnessSpec) => void;
  onClearHarness: () => void;
  onCancelDownload: () => void;
  onDeleteCachedModel: (modelId: string) => Promise<void>;
  onClearCache: () => Promise<void>;
}

function renderCategoryIcon(id: ModelCategory) {
  switch (id) {
    case 'all':
      return <Layers className="h-4 w-4 shrink-0" />;
    case 'coding':
      return <Code2 className="h-4 w-4 shrink-0" />;
    case 'general':
      return <MessageSquare className="h-4 w-4 shrink-0" />;
    case 'compact':
      return <Zap className="h-4 w-4 shrink-0" />;
  }
}

function formatContextLength(tokens: number): string {
  if (tokens >= 1024) {
    return `${Math.round(tokens / 1024)}K`;
  }
  return tokens.toLocaleString();
}

export const ModelManagerModal: FC<ModelManagerModalProps> = ({
  open,
  initialTab = 'all',
  onClose,
  engineState,
  onSelectCatalogModel,
  onUploadLocalFile,
  onLoadCustomUrl,
  onImportHarness,
  onClearHarness,
  onCancelDownload,
  onDeleteCachedModel,
  onClearCache,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const harnessFileInputRef = useRef<HTMLInputElement>(null);
  const [activeTab, setActiveTab] = useState<ModalViewTab>(initialTab);
  const [customUrl, setCustomUrl] = useState('');
  const [harnessJsonDraft, setHarnessJsonDraft] = useState('');
  const [localError, setLocalError] = useState<string | null>(null);
  const [localSuccess, setLocalSuccess] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);

  useEffect(() => {
    if (open) {
      setActiveTab(initialTab);
      setLocalError(null);
      setLocalSuccess(null);
    }
  }, [open, initialTab]);

  const inspectedCategory: ModelCategory =
    activeTab === 'custom-ingest' || activeTab === 'harness' ? 'all' : activeTab;

  const catalogView = useMemo(
    () =>
      observeCatalogScaling(
        {
          activeModelId: engineState.model?.id ?? null,
          activeModelName: engineState.model?.displayName ?? null,
          env: engineState.telemetry.performanceContract?.env ?? null,
          sampling: engineState.sampling,
          precisionMode: engineState.precisionMode,
          weightResidencyPolicy: engineState.weightResidencyPolicy,
          opfsStorage: engineState.opfsStorage,
        },
        inspectedCategory,
      ),
    [
      engineState.model?.id,
      engineState.model?.displayName,
      engineState.telemetry.performanceContract?.env,
      engineState.sampling,
      engineState.precisionMode,
      engineState.weightResidencyPolicy,
      engineState.opfsStorage,
      inspectedCategory,
    ],
  );

  const importedHarness = engineState.importedHarness;
  const matchedHarnessModel = useMemo(
    () => matchCatalogModelForHarness(importedHarness),
    [importedHarness],
  );

  const isDownloading = engineState.status === 'downloading';
  const isBusy = busyId !== null || isDownloading || engineState.status === 'loading';

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !isBusy) {
        onClose();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [open, isBusy, onClose]);

  if (!open) return null;

  const handleCatalogSelect = async (model: CatalogModel) => {
    setLocalError(null);
    setLocalSuccess(null);
    setBusyId(model.id);
    try {
      await onSelectCatalogModel(model);
      onClose();
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg !== 'Download cancelled.') {
        setLocalError(msg);
      }
    } finally {
      setBusyId(null);
    }
  };

  const uploadFile = async (file: File) => {
    setLocalError(null);
    setLocalSuccess(null);
    setBusyId('local-file');
    try {
      await onUploadLocalFile(file);
      onClose();
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyId(null);
    }
  };

  const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    await uploadFile(file);
    e.target.value = '';
  };

  const handleDrop = async (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    if (isBusy) return;
    const file = e.dataTransfer.files?.[0];
    if (file) {
      await uploadFile(file);
    }
  };

  const handleUrlSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const trimmed = customUrl.trim();
    if (!trimmed) {
      setLocalError('Enter a direct HTTPS URL to a supported GGUF model file.');
      return;
    }
    if (!/^https?:\/\//i.test(trimmed)) {
      setLocalError('URL must start with http:// or https://.');
      return;
    }
    setLocalError(null);
    setLocalSuccess(null);
    setBusyId('custom-url');
    try {
      await onLoadCustomUrl(trimmed);
      setCustomUrl('');
      onClose();
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg !== 'Download cancelled.') {
        setLocalError(msg);
      }
    } finally {
      setBusyId(null);
    }
  };

  const handleHarnessFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLocalError(null);
    setLocalSuccess(null);
    try {
      const text = await file.text();
      const parsed = parseHarnessConfigJson(text, file.name);
      onImportHarness(parsed);
      setHarnessJsonDraft('');
      setLocalSuccess(`Loaded harness "${parsed.name}".`);
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : String(err));
    } finally {
      e.target.value = '';
    }
  };

  const handleHarnessJsonSubmit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = harnessJsonDraft.trim();
    if (!trimmed) {
      setLocalError('Paste a JSON harness configuration or choose a .json file.');
      return;
    }
    setLocalError(null);
    setLocalSuccess(null);
    try {
      const parsed = parseHarnessConfigJson(trimmed, 'Pasted JSON');
      onImportHarness(parsed);
      setHarnessJsonDraft('');
      setLocalSuccess(`Loaded harness "${parsed.name}".`);
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleExportHarnessJson = () => {
    const json = serializeHarnessConfig(
      importedHarness,
      engineState.sampling,
      engineState.model?.displayName,
    );
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${(importedHarness?.name ?? 'verdant-harness')
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const { envelope, activeCategory } = catalogView;
  const totalStoredItems = envelope.cachedModelCount + envelope.partialModelCount;

  const renderModelRow = (assessment: CatalogModelAssessment) => {
    const { model } = assessment;
    const isActive = assessment.readiness === 'active';
    const isCached = assessment.readiness === 'ready-cached';
    const isPaused = assessment.readiness === 'paused-resumable';
    const isDemandCache = assessment.readiness === 'ready-demand-cache';
    const hasStoredData = isCached || isPaused;
    const isHarnessTarget = matchedHarnessModel?.id === model.id;

    const paramSummary = model.geometry.isMoe
      ? `${model.geometry.paramsBillions}B (${model.geometry.activeParamsBillions}B active)`
      : `${model.geometry.paramsBillions}B`;

    const statusLabel = isActive
      ? 'Active'
      : isCached
        ? assessment.storageMode === 'demand-cache'
          ? 'Cached on demand'
          : 'Downloaded'
        : isPaused
          ? `Paused (${formatBytes(assessment.partialReceivedBytes)} / ${formatBytes(assessment.partialTotalBytes)})`
          : isDemandCache
            ? 'Streams on demand'
            : 'Available';

    return (
      <div
        key={model.id}
        data-testid={`model-row-${model.id}`}
        className="flex flex-col gap-3 px-4 py-3.5 sm:flex-row sm:items-center sm:justify-between"
      >
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
            <span className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              {model.name}
            </span>
            <span className="font-mono text-xs text-slate-500 dark:text-slate-400 tabular-nums">
              {model.sizeLabel}
            </span>
            {isHarnessTarget && (
              <span className="text-[11px] font-semibold text-emerald-800 dark:text-emerald-400">
                · Requested by harness
              </span>
            )}
          </div>

          <p className="mt-1 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
            {model.description}
          </p>

          <div className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-slate-500 dark:text-slate-400 tabular-nums">
            <span className="font-mono">{paramSummary}</span>
            <span aria-hidden="true">·</span>
            <span className="font-mono">{model.quantization}</span>
            <span aria-hidden="true">·</span>
            <span>{formatContextLength(model.geometry.nativeContextTokens)} context</span>
            <span aria-hidden="true">·</span>
            <span
              className={
                isCached || isActive
                  ? 'font-medium text-emerald-800 dark:text-emerald-400'
                  : 'text-slate-500 dark:text-slate-400'
              }
            >
              {statusLabel}
            </span>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-1.5 sm:pl-3">
          {hasStoredData && !isActive && (
            <button
              type="button"
              disabled={isBusy}
              onClick={() => void onDeleteCachedModel(model.id)}
              aria-label={
                isPaused
                  ? `Discard partial ${model.name} download`
                  : `Remove ${model.name} from local storage`
              }
              title={isPaused ? 'Discard partial download' : 'Remove from local storage'}
              className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-slate-200 text-slate-500 hover:bg-slate-100 hover:text-red-600 disabled:opacity-45 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-red-400 transition-colors"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          )}

          <button
            type="button"
            disabled={isBusy || isActive}
            onClick={() => void handleCatalogSelect(model)}
            className={`inline-flex h-9 shrink-0 items-center justify-center gap-1.5 rounded-lg px-3.5 text-xs font-medium transition-colors whitespace-nowrap ${
              isActive
                ? 'border border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-900/60 dark:bg-emerald-950/40 dark:text-emerald-300'
                : 'bg-[#0F5132] text-white hover:bg-[#0B3D25] disabled:opacity-45'
            }`}
          >
            {isActive ? (
              <>
                <Check className="h-3.5 w-3.5" />
                <span>Active</span>
              </>
            ) : isCached || isDemandCache ? (
              <>
                <HardDrive className="h-3.5 w-3.5" />
                <span>Load</span>
              </>
            ) : isPaused ? (
              <>
                <RotateCw className="h-3.5 w-3.5" />
                <span>
                  {assessment.partialPercent !== null
                    ? `Resume (${assessment.partialPercent}%)`
                    : 'Resume'}
                </span>
              </>
            ) : (
              <>
                <Download className="h-3.5 w-3.5" />
                <span>Download</span>
              </>
            )}
          </button>
        </div>
      </div>
    );
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="model-manager-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-5"
    >
      <div
        className="fixed inset-0 bg-slate-950/50"
        onClick={() => !isBusy && onClose()}
        aria-hidden="true"
      />
      <div className="relative z-10 flex max-h-[88vh] w-full max-w-4xl flex-col overflow-hidden rounded-xl border border-slate-200 bg-white shadow-2xl dark:border-slate-800 dark:bg-[#0F141C]">
        {/* Modal Header */}
        <div className="flex items-center justify-between gap-4 border-b border-slate-200 px-5 py-4 dark:border-slate-800">
          <div className="min-w-0">
            <h2
              id="model-manager-title"
              className="text-base font-semibold text-slate-900 dark:text-slate-100"
            >
              Local Storage &amp; Checkpoints
            </h2>
            <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
              Manage OPFS-cached GGUF substrates, open a local file from disk, or import a custom
              JSON harness override.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={isBusy}
            aria-label="Close model manager"
            className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md text-slate-500 hover:bg-slate-100 hover:text-slate-900 disabled:opacity-40 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-100"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Split Body */}
        <div className="flex min-h-0 flex-1 flex-col md:flex-row">
          {/* Left Rail: Checkpoints, Local/URL, Harness, Browser Storage */}
          <aside className="flex shrink-0 flex-col border-b border-slate-200 bg-[#F8F9FA] md:w-60 md:border-r md:border-b-0 dark:border-slate-800 dark:bg-[#0B0F14]">
            <nav
              aria-label="Model manager sections"
              className="flex gap-1.5 overflow-x-auto p-3 md:flex-1 md:flex-col md:overflow-x-visible md:overflow-y-auto"
            >
              <div className="hidden px-2 pb-1 text-[11px] font-semibold tracking-wide text-slate-500 md:block dark:text-slate-400">
                Substrates
              </div>

              {catalogView.categorySummaries.map((summary) => {
                const selected = activeTab === summary.category.id;
                return (
                  <button
                    key={summary.category.id}
                    type="button"
                    data-testid={`category-tab-${summary.category.id}`}
                    onClick={() => {
                      setActiveTab(summary.category.id);
                      setLocalError(null);
                      setLocalSuccess(null);
                    }}
                    className={`flex shrink-0 items-center justify-between gap-2 rounded-lg border px-3 py-2 text-left text-xs font-semibold transition-colors whitespace-nowrap md:w-full ${
                      selected
                        ? 'border-emerald-700/40 bg-white text-slate-950 dark:border-emerald-500/40 dark:bg-[#141B26] dark:text-white'
                        : 'border-transparent text-slate-700 hover:bg-slate-200/60 dark:text-slate-300 dark:hover:bg-slate-800/60'
                    }`}
                  >
                    <span className="inline-flex items-center gap-2">
                      <span
                        className={
                          selected
                            ? 'text-emerald-800 dark:text-emerald-400'
                            : 'text-slate-500 dark:text-slate-400'
                        }
                      >
                        {renderCategoryIcon(summary.category.id)}
                      </span>
                      <span className="md:hidden">{summary.category.shortLabel}</span>
                      <span className="hidden md:inline">{summary.category.label}</span>
                    </span>
                    <span className="hidden font-mono text-[11px] font-normal text-slate-500 tabular-nums md:inline dark:text-slate-400">
                      {summary.totalCount}
                    </span>
                  </button>
                );
              })}

              <div className="hidden my-2 border-t border-slate-200 md:block dark:border-slate-800" />

              <button
                type="button"
                data-testid="tab-custom-ingest"
                onClick={() => {
                  setActiveTab('custom-ingest');
                  setLocalError(null);
                  setLocalSuccess(null);
                }}
                className={`flex shrink-0 items-center gap-2 rounded-lg border px-3 py-2 text-left text-xs font-semibold transition-colors whitespace-nowrap md:w-full ${
                  activeTab === 'custom-ingest'
                    ? 'border-emerald-700/40 bg-white text-slate-950 dark:border-emerald-500/40 dark:bg-[#141B26] dark:text-white'
                    : 'border-transparent text-slate-700 hover:bg-slate-200/60 dark:text-slate-300 dark:hover:bg-slate-800/60'
                }`}
              >
                <FolderUp
                  className={`h-4 w-4 shrink-0 ${
                    activeTab === 'custom-ingest'
                      ? 'text-emerald-800 dark:text-emerald-400'
                      : 'text-slate-500 dark:text-slate-400'
                  }`}
                />
                <span>Local &amp; URL</span>
              </button>

              <button
                type="button"
                data-testid="tab-harness"
                onClick={() => {
                  setActiveTab('harness');
                  setLocalError(null);
                  setLocalSuccess(null);
                }}
                className={`flex shrink-0 items-center justify-between gap-2 rounded-lg border px-3 py-2 text-left text-xs font-semibold transition-colors whitespace-nowrap md:w-full ${
                  activeTab === 'harness'
                    ? 'border-emerald-700/40 bg-white text-slate-950 dark:border-emerald-500/40 dark:bg-[#141B26] dark:text-white'
                    : 'border-transparent text-slate-700 hover:bg-slate-200/60 dark:text-slate-300 dark:hover:bg-slate-800/60'
                }`}
              >
                <span className="inline-flex items-center gap-2">
                  <FileJson
                    className={`h-4 w-4 shrink-0 ${
                      activeTab === 'harness'
                        ? 'text-emerald-800 dark:text-emerald-400'
                        : 'text-slate-500 dark:text-slate-400'
                    }`}
                  />
                  <span>Harness</span>
                </span>
                {importedHarness && (
                  <span className="text-[10px] font-semibold text-emerald-800 dark:text-emerald-400">
                    Loaded
                  </span>
                )}
              </button>
            </nav>

            {/* Minimal Browser Storage Summary */}
            <div className="hidden border-t border-slate-200 p-3.5 text-[11px] md:block dark:border-slate-800">
              <div className="flex items-center justify-between text-slate-700 dark:text-slate-300">
                <span className="inline-flex items-center gap-1.5 font-medium">
                  <HardDrive className="h-3.5 w-3.5 text-slate-500 dark:text-slate-400" />
                  <span>Browser Storage</span>
                </span>
              </div>
              <p className="mt-1.5 font-mono text-slate-600 dark:text-slate-400 tabular-nums">
                {envelope.opfsSupported
                  ? `${formatBytes(envelope.opfsUsedBytes)}${
                      envelope.opfsQuotaBytes > 0
                        ? ` / ${formatBytes(envelope.opfsQuotaBytes)}`
                        : ' used'
                    }`
                  : 'Temporary memory'}
              </p>
              {totalStoredItems > 0 && (
                <button
                  type="button"
                  disabled={isBusy}
                  onClick={() => void onClearCache()}
                  className="mt-2.5 inline-flex w-full items-center justify-center gap-1.5 rounded-md border border-slate-200 bg-white px-2.5 py-1.5 text-[11px] font-medium text-slate-600 hover:bg-slate-100 hover:text-slate-900 disabled:opacity-45 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-white transition-colors"
                >
                  <Trash2 className="h-3 w-3" />
                  <span>Clear Storage ({totalStoredItems})</span>
                </button>
              )}
            </div>
          </aside>

          {/* Right Pane */}
          <div className="min-w-0 flex-1 space-y-5 overflow-y-auto p-5">
            {(localError || engineState.errorMessage) && (
              <div
                role="alert"
                className="rounded-lg border border-red-200 bg-red-50 px-3.5 py-2.5 text-xs text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-300"
              >
                {localError ?? engineState.errorMessage}
              </div>
            )}

            {localSuccess && (
              <div
                role="status"
                className="rounded-lg border border-emerald-200 bg-emerald-50 px-3.5 py-2.5 text-xs text-emerald-900 dark:border-emerald-900/60 dark:bg-emerald-950/40 dark:text-emerald-300"
              >
                {localSuccess}
              </div>
            )}

            {isBusy && (
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-3.5 dark:border-slate-800 dark:bg-[#141B26]">
                <div className="flex items-center justify-between gap-3 text-xs text-slate-700 dark:text-slate-300">
                  <span className="truncate">{engineState.statusDetail}</span>
                  <div className="flex shrink-0 items-center gap-2">
                    <span className="font-mono tabular-nums">{engineState.progressPercent}%</span>
                    {isBusy && (
                      <button
                        type="button"
                        onClick={onCancelDownload}
                        className="rounded-md border border-slate-300 bg-white px-2 py-0.5 text-[11px] font-medium text-slate-700 hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
                      >
                        Cancel
                      </button>
                    )}
                  </div>
                </div>
                <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
                  <div
                    className="h-full bg-[#0F5132] dark:bg-emerald-500"
                    style={{
                      width: '100%',
                      transform: `scaleX(${Math.max(0.04, engineState.progressPercent / 100)})`,
                      transformOrigin: 'left',
                    }}
                  />
                </div>
              </div>
            )}

            {/* Mobile compact storage bar */}
            <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-slate-200 bg-slate-50/80 px-3.5 py-2 text-[11px] md:hidden dark:border-slate-800 dark:bg-[#141B26]">
              <span className="font-mono text-slate-600 dark:text-slate-300 tabular-nums">
                {envelope.opfsSupported
                  ? `${formatBytes(envelope.opfsUsedBytes)} stored`
                  : 'Temporary memory'}
              </span>
              {totalStoredItems > 0 && (
                <button
                  type="button"
                  disabled={isBusy}
                  onClick={() => void onClearCache()}
                  className="text-[11px] font-medium text-slate-600 underline hover:text-slate-900 dark:text-slate-300 dark:hover:text-white"
                >
                  Clear Storage ({totalStoredItems})
                </button>
              )}
            </div>

            {activeTab === 'custom-ingest' ? (
              <div className="space-y-4">
                <div>
                  <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                    Local File &amp; Direct URL
                  </h3>
                  <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                    Load any local or direct-link GGUF model (LLaMA, Qwen, Mistral, Mixtral, or
                    Granite). Local files read directly from disk without consuming browser storage.
                  </p>
                </div>

                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <div
                    onDragOver={(e) => {
                      e.preventDefault();
                      setDragActive(true);
                    }}
                    onDragLeave={() => setDragActive(false)}
                    onDrop={(e) => void handleDrop(e)}
                    className={`rounded-lg border p-4 transition-colors ${
                      dragActive
                        ? 'border-emerald-700 bg-emerald-50/50 dark:border-emerald-500 dark:bg-emerald-950/20'
                        : 'border-slate-200 dark:border-slate-800'
                    }`}
                  >
                    <h4 className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                      Open Local GGUF File
                    </h4>
                    <p className="mt-1 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
                      Drop or select a <code className="font-mono">.gguf</code> file from your
                      device. Opens directly from disk with zero duplicate copy—even for 30B+
                      models.
                    </p>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".gguf"
                      onChange={(e) => void handleFileChange(e)}
                      className="sr-only"
                      aria-label="Upload local GGUF model file"
                    />
                    <button
                      type="button"
                      disabled={isBusy}
                      onClick={() => fileInputRef.current?.click()}
                      className="mt-3.5 inline-flex h-9 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3.5 text-xs font-medium text-slate-800 hover:bg-slate-50 disabled:opacity-45 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-200 dark:hover:bg-slate-800 transition-colors whitespace-nowrap"
                    >
                      <Upload className="h-3.5 w-3.5" />
                      <span>Choose .gguf File</span>
                    </button>
                  </div>

                  <form
                    onSubmit={(e) => void handleUrlSubmit(e)}
                    className="rounded-lg border border-slate-200 p-4 dark:border-slate-800"
                  >
                    <label
                      htmlFor="custom-gguf-url"
                      className="block text-xs font-semibold text-slate-900 dark:text-slate-100"
                    >
                      Load from Direct URL
                    </label>
                    <p className="mt-1 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
                      Paste a direct Hugging Face or CORS-enabled GGUF link. Supports automatic
                      resume and on-demand caching for large files.
                    </p>
                    <div className="mt-3.5 flex gap-2">
                      <input
                        id="custom-gguf-url"
                        type="url"
                        value={customUrl}
                        onChange={(e) => setCustomUrl(e.target.value)}
                        placeholder="https://huggingface.co/.../model.gguf"
                        className="h-9 min-w-0 flex-1 rounded-lg border border-slate-200 bg-white px-2.5 text-xs text-slate-900 outline-none focus:border-emerald-700 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-100"
                      />
                      <button
                        type="submit"
                        disabled={isBusy}
                        className="inline-flex h-9 shrink-0 items-center rounded-lg bg-[#0F5132] px-3.5 text-xs font-medium text-white hover:bg-[#0B3D25] disabled:opacity-45 transition-colors whitespace-nowrap"
                      >
                        Load
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            ) : activeTab === 'harness' ? (
              <div className="space-y-5">
                <div className="flex flex-col gap-3 border-b border-slate-200 pb-4 sm:flex-row sm:items-start sm:justify-between dark:border-slate-800">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                      Imported Harness Configuration
                    </h3>
                    <p className="mt-1 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
                      Import a standard JSON harness (Assistant-UI{' '}
                      <code className="font-mono">ModelContext</code> or OpenAI Chat format) to set
                      the system message and model parameters. Without an imported harness, models
                      run unmodified with no injected system prompt.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={handleExportHarnessJson}
                    className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-200 dark:hover:bg-slate-800 transition-colors whitespace-nowrap"
                  >
                    <Download className="h-3.5 w-3.5" />
                    <span>Export JSON</span>
                  </button>
                </div>

                {/* Active Harness Summary */}
                {importedHarness ? (
                  <div
                    data-testid="active-harness-summary"
                    className="rounded-lg border border-slate-200 p-4 dark:border-slate-800"
                  >
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                            {importedHarness.name}
                          </span>
                          <span className="text-[11px] font-medium text-emerald-800 dark:text-emerald-400">
                            · Active
                          </span>
                        </div>
                        <p className="mt-0.5 text-[11px] text-slate-500 dark:text-slate-400">
                          Source: {importedHarness.sourceLabel}
                        </p>
                      </div>
                      <button
                        type="button"
                        onClick={() => {
                          onClearHarness();
                          setLocalSuccess('Harness unloaded. Model will run without a system prompt.');
                        }}
                        className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-slate-200 px-3 text-xs font-medium text-slate-700 hover:bg-slate-100 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-800 transition-colors whitespace-nowrap"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                        <span>Unload Harness</span>
                      </button>
                    </div>

                    <dl className="mt-3 divide-y divide-slate-100 rounded-lg border border-slate-200 text-xs dark:divide-slate-800 dark:border-slate-800">
                      <div className="px-3 py-2">
                        <dt className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
                          System message
                        </dt>
                        <dd className="mt-1 whitespace-pre-wrap text-slate-800 dark:text-slate-200">
                          {importedHarness.system ?? 'None (unprompted)'}
                        </dd>
                      </div>

                      {importedHarness.callSettings &&
                        Object.keys(importedHarness.callSettings).length > 0 && (
                          <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2">
                            <dt className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
                              Call settings
                            </dt>
                            <dd className="font-mono text-[11px] text-slate-800 tabular-nums dark:text-slate-200">
                              {Object.entries(importedHarness.callSettings)
                                .map(([k, v]) => `${k}: ${String(v)}`)
                                .join(' · ')}
                            </dd>
                          </div>
                        )}

                      {importedHarness.config?.modelName && (
                        <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2">
                          <dt className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
                            Requested model
                          </dt>
                          <dd className="flex items-center gap-2">
                            <span className="font-mono text-[11px] text-slate-800 dark:text-slate-200">
                              {importedHarness.config.modelName}
                            </span>
                            {matchedHarnessModel &&
                              engineState.model?.id !== matchedHarnessModel.id && (
                                <button
                                  type="button"
                                  disabled={isBusy}
                                  onClick={() => void handleCatalogSelect(matchedHarnessModel)}
                                  className="inline-flex h-7 items-center rounded-md bg-[#0F5132] px-2.5 text-[11px] font-medium text-white hover:bg-[#0B3D25] disabled:opacity-45 whitespace-nowrap"
                                >
                                  Load {matchedHarnessModel.shortName}
                                </button>
                              )}
                          </dd>
                        </div>
                      )}

                      {importedHarness.modelUrl && (
                        <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2">
                          <dt className="text-[11px] font-medium text-slate-500 dark:text-slate-400">
                            Model URL
                          </dt>
                          <dd className="flex items-center gap-2">
                            <span className="max-w-[260px] truncate font-mono text-[11px] text-slate-600 dark:text-slate-400">
                              {importedHarness.modelUrl}
                            </span>
                            <button
                              type="button"
                              disabled={isBusy}
                              onClick={() => void onLoadCustomUrl(importedHarness.modelUrl!)}
                              className="inline-flex h-7 items-center rounded-md bg-[#0F5132] px-2.5 text-[11px] font-medium text-white hover:bg-[#0B3D25] disabled:opacity-45 whitespace-nowrap"
                            >
                              Load URL
                            </button>
                          </dd>
                        </div>
                      )}
                    </dl>
                  </div>
                ) : (
                  <div
                    data-testid="no-harness-banner"
                    className="rounded-lg border border-slate-200 bg-slate-50/70 px-4 py-3 text-xs text-slate-600 dark:border-slate-800 dark:bg-[#141B26] dark:text-slate-400"
                  >
                    No harness imported. Models run cleanly with no system message injected.
                  </div>
                )}

                {/* Import From File or JSON */}
                <div className="space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-slate-200 p-4 dark:border-slate-800">
                    <div>
                      <h4 className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                        Import Harness File (.json)
                      </h4>
                      <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                        Load a JSON file containing <code className="font-mono">system</code>,{' '}
                        <code className="font-mono">callSettings</code>,{' '}
                        <code className="font-mono">config</code>, or OpenAI{' '}
                        <code className="font-mono">messages</code>.
                      </p>
                    </div>
                    <input
                      ref={harnessFileInputRef}
                      type="file"
                      accept=".json,application/json"
                      onChange={(e) => void handleHarnessFileChange(e)}
                      className="sr-only"
                      aria-label="Import JSON harness file"
                    />
                    <button
                      type="button"
                      onClick={() => harnessFileInputRef.current?.click()}
                      className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3.5 text-xs font-medium text-slate-800 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-200 dark:hover:bg-slate-800 transition-colors whitespace-nowrap"
                    >
                      <Upload className="h-3.5 w-3.5" />
                      <span>Choose .json File</span>
                    </button>
                  </div>

                  <form
                    onSubmit={handleHarnessJsonSubmit}
                    className="rounded-lg border border-slate-200 p-4 dark:border-slate-800"
                  >
                    <label
                      htmlFor="harness-json-input"
                      className="block text-xs font-semibold text-slate-900 dark:text-slate-100"
                    >
                      Paste Harness JSON
                    </label>
                    <textarea
                      id="harness-json-input"
                      value={harnessJsonDraft}
                      onChange={(e) => setHarnessJsonDraft(e.target.value)}
                      rows={6}
                      placeholder={
                        '{\n  "name": "My Harness",\n  "system": "...",\n  "callSettings": { "temperature": 0.2, "topP": 0.9 },\n  "config": { "modelName": "Qwen3-Coder 30B-A3B" }\n}'
                      }
                      className="mt-2 w-full rounded-lg border border-slate-200 bg-white p-3 font-mono text-xs text-slate-900 outline-none focus:border-emerald-700 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-100"
                    />
                    <div className="mt-3 flex justify-end">
                      <button
                        type="submit"
                        className="inline-flex h-9 items-center rounded-lg bg-[#0F5132] px-4 text-xs font-medium text-white hover:bg-[#0B3D25] transition-colors whitespace-nowrap"
                      >
                        Apply Harness
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            ) : (
              <>
                <section className="border-b border-slate-200 pb-4 dark:border-slate-800">
                  <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                    {activeCategory.label}
                  </h3>
                  <p className="mt-1 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
                    {activeCategory.description}
                  </p>
                </section>

                <section aria-label={`${activeCategory.label} list`}>
                  <div className="flex items-baseline justify-between gap-2">
                    <h4 className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                      Checkpoints ({catalogView.models.length})
                    </h4>
                  </div>

                  <div className="mt-2 divide-y divide-slate-200 rounded-lg border border-slate-200 dark:divide-slate-800 dark:border-slate-800">
                    {catalogView.models.map(renderModelRow)}
                  </div>
                </section>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
