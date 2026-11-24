import { useState, type FC } from 'react';
import {
  ThreadListItemPrimitive,
  ThreadListPrimitive,
  useAuiState,
} from '@assistant-ui/react';
import { Archive, ArchiveRestore, ChevronDown, ChevronRight, Cpu, Plus, Trash2, X } from 'lucide-react';
import { formatBytes } from '../engine/modelCatalog';
import type { EngineState } from '../engine/workerClient';

interface ThreadSidebarProps {
  engineState: EngineState;
  mobileOpen: boolean;
  onCloseMobile: () => void;
  onOpenModels: () => void;
  onOpenInspector: () => void;
}

const ThreadListItem: FC = () => (
  <ThreadListItemPrimitive.Root className="group flex items-center gap-1 rounded-lg px-2 py-1.5 text-sm text-slate-700 hover:bg-slate-100 data-[active]:bg-slate-200/75 data-[active]:font-medium data-[active]:text-slate-950 dark:text-slate-300 dark:hover:bg-slate-800/70 dark:data-[active]:bg-slate-800 dark:data-[active]:text-white transition-colors">
    <ThreadListItemPrimitive.Trigger className="flex-1 truncate py-1 text-left text-[13px] focus-visible:outline-2 focus-visible:outline-emerald-700">
      <ThreadListItemPrimitive.Title fallback="New conversation" />
    </ThreadListItemPrimitive.Trigger>
    <ThreadListItemPrimitive.Archive
      aria-label="Archive thread"
      className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-slate-400 opacity-0 hover:bg-slate-200 hover:text-slate-700 group-hover:opacity-100 focus-visible:opacity-100 dark:hover:bg-slate-700 dark:hover:text-slate-200 transition-opacity"
    >
      <Archive className="h-3.5 w-3.5" />
    </ThreadListItemPrimitive.Archive>
    <ThreadListItemPrimitive.Delete
      aria-label="Delete thread"
      className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-slate-400 opacity-0 hover:bg-red-100 hover:text-red-700 group-hover:opacity-100 focus-visible:opacity-100 dark:hover:bg-red-950/60 dark:hover:text-red-300 transition-opacity"
    >
      <Trash2 className="h-3.5 w-3.5" />
    </ThreadListItemPrimitive.Delete>
  </ThreadListItemPrimitive.Root>
);

const ArchivedThreadListItem: FC = () => (
  <ThreadListItemPrimitive.Root className="group flex items-center gap-1 rounded-lg px-2 py-1.5 text-sm text-slate-500 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800/70 transition-colors">
    <ThreadListItemPrimitive.Trigger className="flex-1 truncate py-1 text-left text-[13px] focus-visible:outline-2 focus-visible:outline-emerald-700">
      <ThreadListItemPrimitive.Title fallback="Archived conversation" />
    </ThreadListItemPrimitive.Trigger>
    <ThreadListItemPrimitive.Unarchive
      aria-label="Restore archived thread"
      className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-slate-400 hover:bg-slate-200 hover:text-slate-800 dark:hover:bg-slate-700 dark:hover:text-slate-100 transition-colors"
    >
      <ArchiveRestore className="h-3.5 w-3.5" />
    </ThreadListItemPrimitive.Unarchive>
    <ThreadListItemPrimitive.Delete
      aria-label="Delete archived thread"
      className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-slate-400 hover:bg-red-100 hover:text-red-700 dark:hover:bg-red-950/60 dark:hover:text-red-300 transition-colors"
    >
      <Trash2 className="h-3.5 w-3.5" />
    </ThreadListItemPrimitive.Delete>
  </ThreadListItemPrimitive.Root>
);

export const ThreadSidebar: FC<ThreadSidebarProps> = ({
  engineState,
  mobileOpen,
  onCloseMobile,
  onOpenModels,
  onOpenInspector,
}) => {
  const [showArchived, setShowArchived] = useState(false);
  const archivedCount = useAuiState((s) => s.threads.archivedThreadIds.length);

  const totalBufferBytes = engineState.telemetry.buffers.reduce(
    (sum, b) => sum + b.usedBytes,
    0,
  );

  const sidebarContent = (
    <div className="flex h-full w-full flex-col bg-white dark:bg-[#0F141C]">
      <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3.5 dark:border-slate-800">
        <span className="font-display text-sm font-semibold tracking-tight text-slate-900 dark:text-slate-100">
          Threads
        </span>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={onCloseMobile}
            aria-label="Close sidebar"
            className="inline-flex h-8 w-8 items-center justify-center rounded-md text-slate-500 hover:bg-slate-100 lg:hidden dark:text-slate-400 dark:hover:bg-slate-800"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>

      <ThreadListPrimitive.Root className="flex flex-1 flex-col overflow-hidden p-3">
        <ThreadListPrimitive.New className="flex h-10 w-full items-center justify-center gap-2 rounded-lg border border-slate-200 bg-[#F8F9FA] px-3 text-xs font-semibold text-slate-900 hover:border-slate-300 hover:bg-slate-100 dark:border-slate-800 dark:bg-[#141B26] dark:text-slate-100 dark:hover:bg-slate-800 transition-colors whitespace-nowrap">
          <Plus className="h-4 w-4" />
          <span>New Thread</span>
        </ThreadListPrimitive.New>

        <nav aria-label="Conversation threads" className="mt-3 flex-1 space-y-1 overflow-y-auto">
          <ThreadListPrimitive.Items components={{ ThreadListItem }} />

          {archivedCount > 0 && (
            <div className="mt-4 border-t border-slate-100 pt-3 dark:border-slate-800/80">
              <button
                type="button"
                aria-expanded={showArchived}
                onClick={() => setShowArchived((prev) => !prev)}
                className="flex w-full items-center justify-between rounded-md px-2 py-1 text-xs font-medium text-slate-500 hover:bg-slate-100 hover:text-slate-800 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-200 transition-colors"
              >
                <span className="inline-flex items-center gap-1.5">
                  {showArchived ? (
                    <ChevronDown className="h-3.5 w-3.5" />
                  ) : (
                    <ChevronRight className="h-3.5 w-3.5" />
                  )}
                  <span>Archived</span>
                </span>
                <span className="font-mono text-[11px] tabular-nums">{archivedCount}</span>
              </button>
              {showArchived && (
                <div className="mt-1 space-y-1">
                  <ThreadListPrimitive.Items
                    archived
                    components={{ ThreadListItem: ArchivedThreadListItem }}
                  />
                </div>
              )}
            </div>
          )}
        </nav>
      </ThreadListPrimitive.Root>

      <div className="border-t border-slate-200 p-3.5 dark:border-slate-800">
        <div className="flex items-center justify-between text-xs font-medium text-slate-900 dark:text-slate-100">
          <span className="truncate">{engineState.model?.displayName ?? 'Loading model'}</span>
          <span className="font-mono text-[11px] text-emerald-800 dark:text-emerald-400 shrink-0 ml-2">
            vTensor
          </span>
        </div>
        <div className="mt-1 flex items-center gap-1.5 text-[11px] text-slate-500 dark:text-slate-400 font-mono tabular-nums">
          <span>{engineState.model?.hparams.primaryQuantization ?? 'F32'}</span>
          <span aria-hidden="true">·</span>
          <span>{totalBufferBytes > 0 ? formatBytes(totalBufferBytes) : '0 B'}</span>
          <span aria-hidden="true">·</span>
          <span>{engineState.telemetry.totalOpsExecuted.toLocaleString()} ops</span>
        </div>
        <div className="mt-3 grid grid-cols-2 gap-2">
          <button
            type="button"
            onClick={() => {
              onCloseMobile();
              onOpenModels();
            }}
            className="inline-flex h-8 items-center justify-center rounded-md border border-slate-200 bg-white px-2.5 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-200 dark:hover:bg-slate-800 transition-colors whitespace-nowrap"
          >
            Switch GGUF
          </button>
          <button
            type="button"
            onClick={() => {
              onCloseMobile();
              onOpenInspector();
            }}
            className="inline-flex h-8 items-center justify-center gap-1 rounded-md border border-slate-200 bg-white px-2.5 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-200 dark:hover:bg-slate-800 transition-colors whitespace-nowrap"
          >
            <Cpu className="h-3.5 w-3.5" />
            <span>Provider</span>
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <>
      <aside className="hidden w-[264px] shrink-0 border-r border-slate-200 lg:block dark:border-slate-800">
        {sidebarContent}
      </aside>

      {mobileOpen && (
        <div className="fixed inset-0 z-40 flex lg:hidden">
          <div
            className="fixed inset-0 bg-slate-950/40"
            onClick={onCloseMobile}
            aria-hidden="true"
          />
          <aside className="relative z-10 flex h-full w-72 max-w-[82vw] border-r border-slate-200 shadow-xl dark:border-slate-800">
            {sidebarContent}
          </aside>
        </div>
      )}
    </>
  );
};
