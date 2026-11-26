/**
 * Calm, minimal user Settings drawer (`src/components/SettingsDrawer.tsx`).
 *
 * Implements DEC-011 (Zero Technical Jargon) & DEC-012 (Zero User Tunables & Zero Manual Uploads):
 * - No temperature/top-p/max-tokens sliders, no quantization pickers, no manual model uploads.
 * - Only human-relevant choices: Appearance and On-Device Storage cleanup.
 */

import { useState, type FC } from 'react';
import { Moon, Sun, Trash2, X } from 'lucide-react';
import { formatBytes } from '../engine/modelCatalog';

interface SettingsDrawerProps {
  open: boolean;
  onClose: () => void;
  darkMode: boolean;
  onToggleDarkMode: () => void;
  cachedBytes: number;
  onClearCache?: () => Promise<void>;
  debugMode?: boolean;
  onOpenDebugPanel?: () => void;
}

export const SettingsDrawer: FC<SettingsDrawerProps> = ({
  open,
  onClose,
  darkMode,
  onToggleDarkMode,
  cachedBytes,
  onClearCache,
  debugMode = false,
  onOpenDebugPanel,
}) => {
  const [clearing, setClearing] = useState(false);

  if (!open) return null;

  return (
    <>
      <div
        className="fixed inset-0 z-20 bg-slate-950/45 xl:hidden"
        onClick={onClose}
        aria-hidden="true"
      />
      <aside
        aria-label="Settings"
        className="fixed inset-y-0 right-0 z-30 flex w-[320px] max-w-[92vw] flex-col border-l border-slate-200 bg-white shadow-xl xl:static xl:z-auto xl:shadow-none dark:border-slate-800 dark:bg-[#0F141C]"
      >
        <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3.5 dark:border-slate-800">
          <div>
            <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">Settings</h2>
            <p className="text-[11px] text-slate-500 dark:text-slate-400">
              Appearance and local storage
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close settings"
            className="inline-flex h-8 w-8 items-center justify-center rounded-md text-slate-500 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-100"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="flex-1 space-y-6 overflow-y-auto p-4 text-xs">
          {/* Appearance */}
          <section className="space-y-2.5">
            <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              Appearance
            </h3>
            <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-slate-50/70 px-3 py-2.5 dark:border-slate-800 dark:bg-[#141B26]">
              <div>
                <div className="font-medium text-slate-900 dark:text-slate-100">Theme</div>
                <div className="text-[11px] text-slate-500 dark:text-slate-400">
                  {darkMode ? 'Dark mode' : 'Light mode'}
                </div>
              </div>
              <button
                type="button"
                onClick={onToggleDarkMode}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
              >
                {darkMode ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
                <span>{darkMode ? 'Light' : 'Dark'}</span>
              </button>
            </div>
          </section>

          {/* Local Storage */}
          <section className="space-y-2.5">
            <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              On-Device Storage
            </h3>
            <div className="space-y-2.5 rounded-xl border border-slate-200 bg-slate-50/70 p-3 dark:border-slate-800 dark:bg-[#141B26]">
              <div className="flex items-center justify-between">
                <span className="text-slate-600 dark:text-slate-300">Saved locally</span>
                <span className="font-medium text-slate-900 dark:text-slate-100 tabular-nums">
                  {cachedBytes > 0 ? formatBytes(cachedBytes) : 'Empty'}
                </span>
              </div>
              <p className="text-[11px] leading-relaxed text-slate-500 dark:text-slate-400">
                Everything runs directly in your browser with zero accounts and zero tracking.
              </p>
              {onClearCache && cachedBytes > 0 && (
                <button
                  type="button"
                  disabled={clearing}
                  onClick={() => {
                    setClearing(true);
                    void onClearCache().finally(() => setClearing(false));
                  }}
                  className="inline-flex w-full items-center justify-center gap-1.5 rounded-lg border border-red-200 bg-white px-3 py-1.5 text-xs font-medium text-red-700 hover:bg-red-50 disabled:opacity-50 dark:border-red-900/60 dark:bg-slate-900 dark:text-red-300 dark:hover:bg-red-950/40"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  <span>{clearing ? 'Clearing…' : 'Clear Cached Data'}</span>
                </button>
              )}
            </div>
          </section>

          {debugMode && onOpenDebugPanel && (
            <section className="border-t border-slate-200 pt-4 dark:border-slate-800">
              <button
                type="button"
                onClick={onOpenDebugPanel}
                className="w-full rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
              >
                Open Developer Diagnostics
              </button>
            </section>
          )}
        </div>
      </aside>
    </>
  );
};
