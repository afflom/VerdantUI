import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type FC,
  type FormEvent,
} from 'react';
import { Check, Database, Download, HardDrive, RotateCw, Trash2, Upload, X } from 'lucide-react';
import { formatBytes, MODEL_CATALOG, type CatalogModel } from '../engine/modelCatalog';
import type { EngineState } from '../engine/workerClient';

interface ModelManagerModalProps {
  open: boolean;
  onClose: () => void;
  engineState: EngineState;
  onSelectCatalogModel: (model: CatalogModel) => Promise<void>;
  onUploadLocalFile: (file: File) => Promise<void>;
  onLoadCustomUrl: (url: string) => Promise<void>;
  onCancelDownload: () => void;
  onDeleteCachedModel: (modelId: string) => Promise<void>;
  onClearCache: () => Promise<void>;
}

export const ModelManagerModal: FC<ModelManagerModalProps> = ({
  open,
  onClose,
  engineState,
  onSelectCatalogModel,
  onUploadLocalFile,
  onLoadCustomUrl,
  onCancelDownload,
  onDeleteCachedModel,
  onClearCache,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [customUrl, setCustomUrl] = useState('');
  const [localError, setLocalError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);

  const isDownloading = engineState.status === 'downloading';
  const isBusy =
    busyId !== null || isDownloading || engineState.status === 'loading';

  const opfsStored = engineState.opfsStorage.storedModels;
  const opfsPartials = engineState.opfsStorage.partialModels ?? [];
  const totalOpfsBytes =
    opfsStored.reduce((acc, m) => acc + m.byteLength, 0) +
    opfsPartials.reduce((acc, m) => acc + m.receivedBytes, 0);
  const remoteCachedCount = opfsStored.length;
  const partialCount = opfsPartials.length;

  const instructModels = MODEL_CATALOG.filter((m) => m.category === 'instruct');
  const referenceModels = MODEL_CATALOG.filter((m) => m.category === 'reference');

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

  const renderModelList = (models: readonly CatalogModel[]) => (
    <div className="mt-2.5 divide-y divide-slate-200 rounded-lg border border-slate-200 dark:divide-slate-800 dark:border-slate-800">
      {models.map((item) => {
        const isActive = engineState.model?.id === item.id;
        const isPersistedOpfs = opfsStored.some((m) => m.modelId === item.id);
        const partial = !isPersistedOpfs
          ? opfsPartials.find((p) => p.modelId === item.id)
          : undefined;
        const partialPct =
          partial && partial.totalBytes > 0
            ? Math.min(99, Math.max(1, Math.round((partial.receivedBytes / partial.totalBytes) * 100)))
            : null;
        const isCached = item.bundled || isPersistedOpfs;

        return (
          <div
            key={item.id}
            className="flex flex-col gap-3 p-4 sm:flex-row sm:items-center sm:justify-between"
          >
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
                <span className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                  {item.name}
                </span>
                <span className="font-mono text-xs text-slate-500 dark:text-slate-400 tabular-nums">
                  {item.sizeLabel}
                </span>
                {item.recommended && (
                  <span className="text-[11px] font-semibold text-emerald-800 dark:text-emerald-400">
                    · Recommended
                  </span>
                )}
              </div>
              <p className="mt-1 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
                {item.description}
              </p>
              <div className="mt-1.5 flex flex-wrap items-center gap-x-2 text-[11px] text-slate-500 dark:text-slate-400">
                <span>{item.architecture}</span>
                <span aria-hidden="true">·</span>
                <span className="font-mono">{item.quantization}</span>
                <span aria-hidden="true">·</span>
                <span>
                  {item.bundled
                    ? 'Bundled offline'
                    : isPersistedOpfs
                      ? 'Persisted in OPFS vRAM'
                      : partial
                        ? `Paused in OPFS (${formatBytes(partial.receivedBytes)}${
                            partial.totalBytes > 0 ? ` / ${formatBytes(partial.totalBytes)}` : ''
                          })`
                        : 'Resumable OPFS vRAM stream'}
                </span>
              </div>
            </div>

            <div className="flex shrink-0 items-center gap-1.5">
              {(isPersistedOpfs || partial) && !isActive && (
                <button
                  type="button"
                  disabled={isBusy}
                  onClick={() => void onDeleteCachedModel(item.id)}
                  aria-label={
                    partial
                      ? `Discard partial ${item.name} download`
                      : `Remove ${item.name} from OPFS storage`
                  }
                  title={partial ? 'Discard partial download' : 'Remove from OPFS storage'}
                  className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-slate-200 text-slate-500 hover:bg-slate-100 hover:text-red-600 disabled:opacity-45 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-red-400 transition-colors"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              )}

              <button
                type="button"
                disabled={isBusy || isActive}
                onClick={() => void handleCatalogSelect(item)}
                className={`inline-flex h-9 shrink-0 items-center justify-center gap-1.5 rounded-lg px-3.5 text-xs font-medium transition-colors whitespace-nowrap ${
                  isActive
                    ? 'border border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-900/60 dark:bg-emerald-950/40 dark:text-emerald-300'
                    : 'bg-[#0F5132] text-white hover:bg-[#0B3D25] disabled:opacity-45'
                }`}
              >
                {isActive ? (
                  <>
                    <Check className="h-3.5 w-3.5" />
                    <span>Loaded</span>
                  </>
                ) : isCached ? (
                  <>
                    <HardDrive className="h-3.5 w-3.5" />
                    <span>Load Model</span>
                  </>
                ) : partial ? (
                  <>
                    <RotateCw className="h-3.5 w-3.5" />
                    <span>{partialPct !== null ? `Resume (${partialPct}%)` : 'Resume'}</span>
                  </>
                ) : (
                  <>
                    <Download className="h-3.5 w-3.5" />
                    <span>Stream &amp; Load</span>
                  </>
                )}
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="model-manager-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
    >
      <div
        className="fixed inset-0 bg-slate-950/50"
        onClick={() => !isBusy && onClose()}
        aria-hidden="true"
      />
      <div className="relative z-10 flex max-h-[88vh] w-full max-w-2xl flex-col overflow-hidden rounded-xl border border-slate-200 bg-white shadow-2xl dark:border-slate-800 dark:bg-[#0F141C]">
        <div className="flex items-center justify-between border-b border-slate-200 px-5 py-4 dark:border-slate-800">
          <div>
            <h2
              id="model-manager-title"
              className="text-base font-semibold text-slate-900 dark:text-slate-100"
            >
              GGUF Model Manager
            </h2>
            <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
              Resumable HTTP Range streaming into OPFS-backed vRAM and vTensor WASM arenas for
              LLaMA, Qwen2/3, Mistral, and Granite GGUFs (F32, F16, BF16, Q2_K–Q6_K, Q4_0–Q8_0).
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={isBusy}
            aria-label="Close model manager"
            className="inline-flex h-8 w-8 items-center justify-center rounded-md text-slate-500 hover:bg-slate-100 hover:text-slate-900 disabled:opacity-40 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-100"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex-1 space-y-5 overflow-y-auto p-5">
          {(localError || engineState.errorMessage) && (
            <div
              role="alert"
              className="rounded-lg border border-red-200 bg-red-50 px-3.5 py-2.5 text-xs text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-300"
            >
              {localError ?? engineState.errorMessage}
            </div>
          )}

          {isBusy && (
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3.5 dark:border-slate-800 dark:bg-[#141B26]">
              <div className="flex items-center justify-between gap-3 text-xs text-slate-700 dark:text-slate-300">
                <span className="truncate">{engineState.statusDetail}</span>
                <div className="flex items-center gap-2 shrink-0">
                  <span className="font-mono tabular-nums">{engineState.progressPercent}%</span>
                  {isDownloading && (
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

          <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-slate-200 bg-slate-50/80 px-3.5 py-2.5 text-xs dark:border-slate-800 dark:bg-[#141B26]">
            <div className="flex items-center gap-2 text-slate-700 dark:text-slate-300">
              <Database className="h-3.5 w-3.5 shrink-0 text-emerald-700 dark:text-emerald-400" />
              <span className="font-medium">OPFS-Backed vRAM Storage</span>
              <span className="font-mono text-[11px] text-slate-500 dark:text-slate-400 tabular-nums">
                {engineState.opfsStorage.supported
                  ? `${remoteCachedCount} cached${
                      partialCount > 0 ? ` · ${partialCount} paused` : ''
                    } · ${formatBytes(totalOpfsBytes)}${
                      engineState.opfsStorage.quotaBytes > 0
                        ? ` / ${formatBytes(engineState.opfsStorage.quotaBytes)} quota`
                        : ''
                    }`
                  : 'Memory fallback'}
              </span>
            </div>
            {(remoteCachedCount > 0 || partialCount > 0) && (
              <button
                type="button"
                disabled={isBusy}
                onClick={() => void onClearCache()}
                className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium text-slate-600 hover:bg-slate-200/70 hover:text-slate-900 disabled:opacity-45 dark:text-slate-300 dark:hover:bg-slate-800 dark:hover:text-white transition-colors whitespace-nowrap"
              >
                <Trash2 className="h-3 w-3" />
                <span>Clear OPFS ({remoteCachedCount + partialCount})</span>
              </button>
            )}
          </div>

          <section>
            <div className="flex items-center justify-between gap-2">
              <h3 className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                Instruction-Tuned Chat Models (135M–3B · OPFS vRAM Streaming)
              </h3>
            </div>
            {renderModelList(instructModels)}
          </section>

          <section>
            <h3 className="text-xs font-semibold text-slate-900 dark:text-slate-100">
              Lightweight &amp; vTensor Conformance Reference Models
            </h3>
            {renderModelList(referenceModels)}
          </section>

          <section className="grid grid-cols-1 gap-4 sm:grid-cols-2">
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
              <h3 className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                Load Local GGUF File
              </h3>
              <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                Drop or select any LLaMA, Qwen2/3, Mistral, or Granite{' '}
                <code className="font-mono">.gguf</code> file (including 2–8 GB+ checkpoints).
                Streams in 4 MiB chunks into OPFS vRAM without full JS RAM copies.
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
                className="mt-3 inline-flex h-9 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3.5 text-xs font-medium text-slate-800 hover:bg-slate-50 disabled:opacity-45 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-200 dark:hover:bg-slate-800 transition-colors whitespace-nowrap"
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
                Stream from Direct GGUF URL
              </label>
              <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                Stream any CORS-enabled Hugging Face or HTTP{' '}
                <code className="font-mono">.gguf</code> directly into OPFS vRAM backing storage.
              </p>
              <div className="mt-3 flex gap-2">
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
                  className="inline-flex h-9 shrink-0 items-center rounded-lg bg-slate-900 px-3 text-xs font-medium text-white hover:bg-slate-800 disabled:opacity-45 dark:bg-slate-100 dark:text-slate-950 dark:hover:bg-slate-200 transition-colors whitespace-nowrap"
                >
                  Fetch
                </button>
              </div>
            </form>
          </section>
        </div>
      </div>
    </div>
  );
};
