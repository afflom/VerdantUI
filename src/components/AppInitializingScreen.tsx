import { memo, type FC } from 'react';
import { AlertCircle, CheckCircle2, Circle, Loader2, RefreshCw, WifiOff } from 'lucide-react';
import type { AppInitializationView, InitStepItem } from '../engine/appInitialization';

interface AppInitializingScreenProps {
  readonly view: AppInitializationView;
  readonly onRetry: () => void;
  readonly onUseOfflineOnly: () => void;
}

const StepIcon: FC<{ readonly status: InitStepItem['status'] }> = ({ status }) => {
  if (status === 'done') {
    return <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600 dark:text-emerald-400" />;
  }
  if (status === 'active') {
    return <Loader2 className="h-4 w-4 shrink-0 animate-spin text-emerald-700 dark:text-emerald-400" />;
  }
  if (status === 'error') {
    return <AlertCircle className="h-4 w-4 shrink-0 text-red-600 dark:text-red-400" />;
  }
  return <Circle className="h-4 w-4 shrink-0 text-slate-300 dark:text-slate-700" />;
};

/**
 * Full-screen Application Initializing Screen displayed from streaming consent (and on returning
 * startup) until VerdantUI's workspace and browser capabilities are ready to use.
 */
export const AppInitializingScreen: FC<AppInitializingScreenProps> = memo(
  ({ view, onRetry, onUseOfflineOnly }) => {
    if (!view.visible) return null;

    const isError = view.phase === 'error';

    return (
      <div
        role="status"
        aria-live="polite"
        aria-busy={!isError}
        data-testid="app-initializing-screen"
        data-init-phase={view.phase}
        className="fixed inset-0 z-50 flex items-center justify-center bg-[#F8F9FA] px-4 py-8 text-slate-900 dark:bg-[#0B0F14] dark:text-slate-100"
      >
        <div className="w-full max-w-md rounded-2xl border border-slate-200/90 bg-white p-6 sm:p-7 shadow-xl shadow-slate-900/5 dark:border-slate-800 dark:bg-[#0F141C] dark:shadow-black/40">
          {/* Brand header */}
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2.5">
              <span
                className={`inline-flex h-2.5 w-2.5 rounded-full ${
                  isError ? 'bg-red-500' : 'animate-pulse bg-emerald-600 dark:bg-emerald-400'
                }`}
              />
              <span className="font-display text-sm font-bold tracking-tight text-slate-950 dark:text-white">
                VerdantUI
              </span>
            </div>
            {!isError && (
              <span
                data-testid="app-init-percent"
                className="font-mono text-xs font-medium tabular-nums text-emerald-800 dark:text-emerald-300"
              >
                {view.overallPercent}%
              </span>
            )}
          </div>

          <h1 className="mt-4 font-display text-lg font-semibold tracking-tight text-slate-950 dark:text-white">
            {view.headline}
          </h1>
          <p
            data-testid="app-init-status-line"
            className={`mt-1.5 text-xs leading-relaxed ${
              isError ? 'text-red-700 dark:text-red-300' : 'text-slate-600 dark:text-slate-400'
            }`}
          >
            {view.statusLine}
          </p>

          {/* Progress track */}
          <div className="mt-5 h-2 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800/90">
            <div
              className={`h-full rounded-full transition-all duration-300 ease-out ${
                isError ? 'bg-red-500' : 'bg-[#0F5132] dark:bg-emerald-500'
              }`}
              style={{ width: `${Math.max(4, Math.min(100, view.overallPercent))}%` }}
            />
          </div>

          {/* Step checklist */}
          {view.steps.length > 0 && (
            <ul className="mt-5 space-y-2.5" aria-label="Initialization progress">
              {view.steps.map((step) => (
                <li
                  key={step.id}
                  data-testid={`app-init-step-${step.id}`}
                  data-step-status={step.status}
                  className="flex items-center justify-between gap-3 rounded-xl border border-slate-100 bg-slate-50/70 px-3 py-2.5 dark:border-slate-800/80 dark:bg-[#141B26]/70"
                >
                  <div className="flex min-w-0 items-center gap-2.5">
                    <StepIcon status={step.status} />
                    <span
                      className={`truncate text-xs font-medium ${
                        step.status === 'pending'
                          ? 'text-slate-500 dark:text-slate-400'
                          : 'text-slate-900 dark:text-slate-100'
                      }`}
                    >
                      {step.label}
                    </span>
                  </div>
                  <span
                    className={`shrink-0 text-[11px] tabular-nums ${
                      step.status === 'error'
                        ? 'text-red-600 dark:text-red-400'
                        : step.status === 'done'
                          ? 'text-emerald-700 dark:text-emerald-400'
                          : 'text-slate-500 dark:text-slate-400'
                    }`}
                  >
                    {step.detail}
                  </span>
                </li>
              ))}
            </ul>
          )}

          {/* Actions */}
          {(view.canRetry || view.canSwitchOffline) && (
            <div className="mt-6 flex flex-wrap items-center justify-end gap-2">
              {view.canSwitchOffline && (
                <button
                  type="button"
                  data-testid="app-init-offline"
                  onClick={onUseOfflineOnly}
                  className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-200 dark:hover:bg-slate-800 transition-colors"
                >
                  <WifiOff className="h-3.5 w-3.5" />
                  <span>Use Offline Only</span>
                </button>
              )}
              {view.canRetry && (
                <button
                  type="button"
                  data-testid="app-init-retry"
                  onClick={onRetry}
                  className="inline-flex h-9 items-center gap-1.5 rounded-lg bg-[#0F5132] px-3.5 text-xs font-medium text-white hover:bg-[#0B3D25] transition-colors"
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                  <span>Retry</span>
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    );
  },
);
