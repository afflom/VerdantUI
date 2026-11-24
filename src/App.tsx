/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  AssistantRuntimeProvider,
  useLocalRuntime,
  useRemoteThreadListRuntime,
} from '@assistant-ui/react';
import { Cpu, Layers, Menu, Moon, Sun } from 'lucide-react';
import { ModelManagerModal } from './components/ModelManagerModal';
import { ThreadSidebar } from './components/ThreadSidebar';
import { ThreadView } from './components/ThreadView';
import { VTensorInspector } from './components/VTensorInspector';
import { createLlamaVTensorAdapter } from './engine/assistantAdapter';
import { RECOMMENDED_INSTRUCT_MODEL, type CatalogModel } from './engine/modelCatalog';
import { createVTensorThreadListAdapter } from './engine/threadStorage';
import { LlamaVTensorWorkerClient, type EngineState } from './engine/workerClient';

const THEME_STORAGE_KEY = 'vtensor-assistant:theme:v1';

function readInitialDarkMode(): boolean {
  if (typeof window === 'undefined' || typeof window.localStorage === 'undefined') {
    return false;
  }
  try {
    return window.localStorage.getItem(THEME_STORAGE_KEY) === 'dark';
  } catch {
    return false;
  }
}

export default function App() {
  const workerClient = useMemo(() => new LlamaVTensorWorkerClient(), []);
  const modelAdapter = useMemo(() => createLlamaVTensorAdapter(workerClient), [workerClient]);
  const threadListAdapter = useMemo(() => createVTensorThreadListAdapter(), []);

  const useThreadRuntime = useCallback(
    () => useLocalRuntime(modelAdapter),
    [modelAdapter],
  );

  const runtime = useRemoteThreadListRuntime({
    runtimeHook: useThreadRuntime,
    adapter: threadListAdapter,
  });

  const [engineState, setEngineState] = useState<EngineState>(() => workerClient.getState());
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [modelsModalOpen, setModelsModalOpen] = useState(false);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [darkMode, setDarkMode] = useState<boolean>(readInitialDarkMode);

  useEffect(() => {
    const unsubscribe = workerClient.subscribe(setEngineState);

    if (!workerClient.getState().model) {
      void workerClient.bootInitialModel().catch(() => {
        // Error state is surfaced via engineState.errorMessage
      });
    }

    return () => {
      unsubscribe();
    };
  }, [workerClient]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const mq = window.matchMedia('(max-width: 1279px)');
    const sync = (matches: boolean) => {
      if (matches) setInspectorOpen(false);
    };
    sync(mq.matches);
    const handler = (e: MediaQueryListEvent) => sync(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (darkMode) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, darkMode ? 'dark' : 'light');
    } catch {
      // Ignore storage error
    }
  }, [darkMode]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      if (mobileSidebarOpen) {
        setMobileSidebarOpen(false);
        return;
      }
      if (inspectorOpen && typeof window !== 'undefined' && window.innerWidth < 1280) {
        setInspectorOpen(false);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [mobileSidebarOpen, inspectorOpen]);

  const handleSelectCatalogModel = async (model: CatalogModel) => {
    await workerClient.loadCatalogModel(model);
  };

  const handleDownloadRecommended = () => {
    void workerClient.loadCatalogModel(RECOMMENDED_INSTRUCT_MODEL).catch(() => {
      // Error or cancel state is surfaced cleanly via engineState
    });
  };

  const handleUploadLocalFile = async (file: File) => {
    await workerClient.loadFromFile(file);
  };

  const handleLoadCustomUrl = async (url: string) => {
    const fileName = url.split('/').pop()?.replace(/\.gguf$/i, '') || 'Custom GGUF';
    await workerClient.loadFromUrl(`custom-${Date.now()}`, fileName, url, false);
  };

  const handleCancelDownload = () => {
    workerClient.cancelDownload();
  };

  const handleDeleteCachedModel = async (modelId: string) => {
    await workerClient.deleteCachedModel(modelId);
  };

  const handleClearCache = async () => {
    await workerClient.clearModelCache();
  };

  const handleRunConformance = async () => {
    await workerClient.runConformanceCheck();
  };

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="flex h-screen w-screen flex-col overflow-hidden bg-[#F8F9FA] text-slate-900 dark:bg-[#0B0F14] dark:text-slate-100">
        {/* Top Bar Contract: Brand on left, single-line workspace actions on right */}
        <header className="flex h-14 shrink-0 items-center justify-between gap-3 border-b border-slate-200 bg-white px-3.5 sm:px-5 dark:border-slate-800 dark:bg-[#0F141C]">
          <div className="flex min-w-0 items-center gap-2.5">
            <button
              type="button"
              onClick={() => setMobileSidebarOpen(true)}
              aria-label="Open conversation threads"
              className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-50 lg:hidden dark:border-slate-800 dark:text-slate-300 dark:hover:bg-slate-800"
            >
              <Menu className="h-4 w-4" />
            </button>
            <span className="font-display text-[15px] font-bold tracking-tight text-slate-950 whitespace-nowrap truncate dark:text-white">
              <span className="hidden sm:inline">vTensor Assistant</span>
              <span className="sm:hidden">vTensor</span>
            </span>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={() => setModelsModalOpen(true)}
              className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-2.5 sm:px-3 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-200 dark:hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0"
            >
              <Layers className="h-3.5 w-3.5" />
              <span>Models</span>
            </button>

            <button
              type="button"
              onClick={() => setInspectorOpen((prev) => !prev)}
              aria-pressed={inspectorOpen}
              className={`inline-flex h-9 items-center gap-1.5 rounded-lg px-2.5 sm:px-3 text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                inspectorOpen
                  ? 'bg-[#0F5132] text-white hover:bg-[#0B3D25]'
                  : 'border border-slate-200 bg-white text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-200 dark:hover:bg-slate-800'
              }`}
            >
              <Cpu className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">vTensor Inspector</span>
              <span className="sm:hidden">Inspector</span>
            </button>

            <button
              type="button"
              onClick={() => setDarkMode((d) => !d)}
              aria-label={darkMode ? 'Switch to light theme' : 'Switch to dark theme'}
              className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-300 dark:hover:bg-slate-800 transition-colors shrink-0"
            >
              {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
          </div>
        </header>

        {/* Main 3-zone workbench */}
        <div className="flex min-h-0 flex-1 overflow-hidden">
          <ThreadSidebar
            engineState={engineState}
            mobileOpen={mobileSidebarOpen}
            onCloseMobile={() => setMobileSidebarOpen(false)}
            onOpenModels={() => setModelsModalOpen(true)}
            onOpenInspector={() => setInspectorOpen(true)}
          />

          <main className="min-w-0 flex-1 overflow-hidden">
            <ThreadView
              engineState={engineState}
              onOpenModels={() => setModelsModalOpen(true)}
              onDownloadRecommended={handleDownloadRecommended}
              onCancelDownload={handleCancelDownload}
              onToggleInspector={() => setInspectorOpen((prev) => !prev)}
              onSpeculativeDraftChange={(messages, draft) =>
                workerClient.scheduleSpeculativePrefill(messages, draft)
              }
            />
          </main>

          <VTensorInspector
            open={inspectorOpen}
            onClose={() => setInspectorOpen(false)}
            engineState={engineState}
            onUpdateSampling={(patch) => workerClient.updateSampling(patch)}
            onUpdatePromptMode={(mode) => workerClient.updatePromptMode(mode)}
            onUpdatePrecisionMode={(mode) => void workerClient.updatePrecisionMode(mode)}
            onUpdateWeightResidencyPolicy={(policy) =>
              void workerClient.updateWeightResidencyPolicy(policy)
            }
            onResetSampling={() => workerClient.resetSampling()}
            onRunConformance={handleRunConformance}
          />
        </div>

        <ModelManagerModal
          open={modelsModalOpen}
          onClose={() => setModelsModalOpen(false)}
          engineState={engineState}
          onSelectCatalogModel={handleSelectCatalogModel}
          onUploadLocalFile={handleUploadLocalFile}
          onLoadCustomUrl={handleLoadCustomUrl}
          onCancelDownload={handleCancelDownload}
          onDeleteCachedModel={handleDeleteCachedModel}
          onClearCache={handleClearCache}
        />
      </div>
    </AssistantRuntimeProvider>
  );
}
