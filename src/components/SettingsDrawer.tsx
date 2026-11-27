/**
 * Settings Prism (`src/components/SettingsDrawer.tsx`)
 *
 * First-class Verdant Prism rendered through the universal `PrismSurfaceShell` (`DEC-011`, `DEC-012`, `DEC-015`):
 * - Clean Preview panel for Appearance and On-Device Storage
 * - Universal `Preview | Details` toggle
 * - Natural Language Interface bar ("Switch to dark mode", "Clear cached data")
 */

import { useState, type FC } from 'react';
import { Moon, Sun, Trash2 } from 'lucide-react';
import { formatBytes } from '../engine/modelCatalog';
import type {
  PrismSurfaceMode,
  SynthesizedPrismFrame,
} from '../engine/observerUiSynthesizer';
import { PrismSurfaceShell, type PrismSurfaceMessage } from './PrismSurfaceShell';

interface SettingsDrawerProps {
  frame: SynthesizedPrismFrame;
  surfaceMode: PrismSurfaceMode;
  onChangeSurfaceMode: (mode: PrismSurfaceMode) => void;
  onSelectSystemPrism: (prismId: string) => void;
  onTypingChange?: (typing: boolean) => void;
  darkMode: boolean;
  onToggleDarkMode: () => void;
  onSetDarkMode?: (dark: boolean) => void;
  cachedBytes: number;
  onClearCache?: () => Promise<void>;
  debugMode?: boolean;
  onOpenDebugPanel?: () => void;
}

export const SettingsDrawer: FC<SettingsDrawerProps> = ({
  frame,
  surfaceMode,
  onChangeSurfaceMode,
  onSelectSystemPrism,
  onTypingChange,
  darkMode,
  onToggleDarkMode,
  onSetDarkMode,
  cachedBytes,
  onClearCache,
  debugMode = false,
  onOpenDebugPanel,
}) => {
  const [clearing, setClearing] = useState(false);
  const [messages, setMessages] = useState<PrismSurfaceMessage[]>([]);

  const handleSubmitNaturalLanguage = async (raw: string) => {
    const prompt = raw.trim();
    if (!prompt) return;
    const lower = prompt.toLowerCase();
    setMessages((prev) => [...prev.slice(-12), { id: `u_${Date.now()}`, role: 'user', content: prompt }]);

    if (/\bdark\b/.test(lower)) {
      if (onSetDarkMode) onSetDarkMode(true);
      else if (!darkMode) onToggleDarkMode();
      setMessages((prev) => [...prev, { id: `a_${Date.now()}`, role: 'assistant', content: 'Switched to dark mode.' }]);
      return;
    }
    if (/\blight\b/.test(lower)) {
      if (onSetDarkMode) onSetDarkMode(false);
      else if (darkMode) onToggleDarkMode();
      setMessages((prev) => [...prev, { id: `a_${Date.now()}`, role: 'assistant', content: 'Switched to light mode.' }]);
      return;
    }
    if (/\b(clear|delete|empty|reset)\b.*\b(cache|storage|data)\b/.test(lower)) {
      if (onClearCache && cachedBytes > 0) {
        setClearing(true);
        try {
          await onClearCache();
          setMessages((prev) => [...prev, { id: `a_${Date.now()}`, role: 'assistant', content: 'Cleared locally cached data.' }]);
        } finally {
          setClearing(false);
        }
      } else {
        setMessages((prev) => [...prev, { id: `a_${Date.now()}`, role: 'assistant', content: 'Local cache is already empty.' }]);
      }
      return;
    }
    if (/\b(storage|space|size|used|saved)\b/.test(lower)) {
      setMessages((prev) => [
        ...prev,
        {
          id: `a_${Date.now()}`,
          role: 'assistant',
          content: cachedBytes > 0 ? `Currently using ${formatBytes(cachedBytes)} locally on your device.` : 'Local storage cache is currently empty.',
        },
      ]);
      return;
    }
    if (/^(?:show|open|view)\s+(?:prism\s+)?details\b/.test(lower)) {
      onChangeSurfaceMode('details');
      return;
    }
    setMessages((prev) => [
      ...prev,
      {
        id: `a_${Date.now()}`,
        role: 'assistant',
        content: 'Try "Switch to dark mode", "Switch to light mode", or "Clear cached data".',
      },
    ]);
  };

  const previewContent = (
    <div className="h-full overflow-y-auto bg-[#F8F9FA] dark:bg-[#0B0F14]">
      <div className="mx-auto max-w-2xl space-y-6 px-5 py-8">
        <div>
          <h2 className="text-xl font-bold tracking-tight text-slate-950 dark:text-white">
            Settings
          </h2>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Appearance and private on-device storage.
          </p>
        </div>

        {/* Appearance */}
        <section className="space-y-2.5">
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
            Appearance
          </h3>
          <div className="flex items-center justify-between rounded-2xl border border-slate-200 bg-white px-4 py-3.5 shadow-xs dark:border-slate-800 dark:bg-[#0F141C]">
            <div>
              <div className="text-sm font-medium text-slate-900 dark:text-slate-100">Theme</div>
              <div className="text-xs text-slate-500 dark:text-slate-400">
                {darkMode ? 'Dark mode' : 'Light mode'}
              </div>
            </div>
            <button
              type="button"
              onClick={onToggleDarkMode}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-100 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-200 dark:hover:bg-slate-800"
            >
              {darkMode ? <Sun className="h-3.5 w-3.5" /> : <Moon className="h-3.5 w-3.5" />}
              <span>{darkMode ? 'Switch to Light' : 'Switch to Dark'}</span>
            </button>
          </div>
        </section>

        {/* Local Storage */}
        <section className="space-y-2.5">
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
            On-Device Storage
          </h3>
          <div className="space-y-3 rounded-2xl border border-slate-200 bg-white p-4 shadow-xs dark:border-slate-800 dark:bg-[#0F141C]">
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-600 dark:text-slate-300">Saved locally</span>
              <span className="font-semibold text-slate-900 dark:text-slate-100 tabular-nums">
                {cachedBytes > 0 ? formatBytes(cachedBytes) : 'Empty'}
              </span>
            </div>
            <p className="text-xs leading-relaxed text-slate-500 dark:text-slate-400">
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
                className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-red-200 bg-white px-3.5 py-2 text-xs font-medium text-red-700 hover:bg-red-50 disabled:opacity-50 dark:border-red-900/60 dark:bg-[#141B26] dark:text-red-300 dark:hover:bg-red-950/40"
              >
                <Trash2 className="h-3.5 w-3.5" />
                <span>{clearing ? 'Clearing…' : 'Clear Cached Data'}</span>
              </button>
            )}
          </div>
        </section>

        {debugMode && onOpenDebugPanel && (
          <section className="border-t border-slate-200 pt-5 dark:border-slate-800">
            <button
              type="button"
              onClick={onOpenDebugPanel}
              className="rounded-lg border border-slate-200 bg-white px-3.5 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#0F141C] dark:text-slate-200 dark:hover:bg-slate-800"
            >
              Open Developer Diagnostics
            </button>
          </section>
        )}
      </div>
    </div>
  );

  return (
    <PrismSurfaceShell
      frame={frame}
      mode={surfaceMode}
      onChangeMode={onChangeSurfaceMode}
      previewContent={previewContent}
      onSelectChildPrism={onSelectSystemPrism}
      onSubmitNaturalLanguage={handleSubmitNaturalLanguage}
      nlMessages={messages}
      onTypingChange={onTypingChange}
    />
  );
};
