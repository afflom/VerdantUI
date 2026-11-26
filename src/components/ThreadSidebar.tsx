import { useRef, useState, type ChangeEvent, type FC } from 'react';
import {
  ThreadListItemPrimitive,
  ThreadListPrimitive,
  useAuiState,
} from '@assistant-ui/react';
import {
  Archive,
  ArchiveRestore,
  ChevronDown,
  ChevronRight,
  FileText,
  FlaskConical,
  Gamepad2,
  Image as ImageIcon,
  Layers,
  Music,
  Plus,
  ShieldCheck,
  SlidersHorizontal,
  Trash2,
  Upload,
  Video,
  Wrench,
  X,
} from 'lucide-react';
import {
  MEDIA_GALLERIES,
  type GalleryMediaType,
} from '../engine/contentGalleryStore';
import { isInternalBootstrapModel } from '../engine/modelCatalog';
import type { VerdantPrismProject } from '../engine/prismStudio';
import type { EngineState } from '../engine/workerClient';

interface ThreadSidebarProps {
  engineState: EngineState;
  mobileOpen: boolean;
  onCloseMobile: () => void;
  onOpenInspector: () => void;
  activeView:
    | { readonly kind: 'thread' }
    | { readonly kind: 'prism'; readonly prismId: string; readonly mode: 'studio' | 'app' }
    | { readonly kind: 'gallery'; readonly mediaType: GalleryMediaType };
  onSelectThreadView: () => void;
  spaces: readonly VerdantPrismProject[];
  galleryCounts: Readonly<Record<GalleryMediaType, number>>;
  onCreateSpace: () => void;
  onSelectGallery: (mediaType: GalleryMediaType) => void;
  onImportHoloFile?: (file: File) => void;
  onSelectPrism: (prismId: string, mode: 'studio' | 'app') => void;
  onDeletePrism: (prismId: string) => void;
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
  onOpenInspector,
  activeView,
  onSelectThreadView,
  spaces,
  galleryCounts,
  onCreateSpace,
  onSelectGallery,
  onImportHoloFile,
  onSelectPrism,
  onDeletePrism,
}) => {
  const [showArchived, setShowArchived] = useState(false);
  const [threadsExpanded, setThreadsExpanded] = useState(true);
  const [spacesExpanded, setSpacesExpanded] = useState(true);
  const [galleriesExpanded, setGalleriesExpanded] = useState(true);
  const holoFileRef = useRef<HTMLInputElement | null>(null);
  const handleHoloPick = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (file && onImportHoloFile) {
      onImportHoloFile(file);
      onCloseMobile();
    }
  };
  const archivedCount = useAuiState((s) => s.threads.archivedThreadIds.length);
  const isBootstrapUnbound = isInternalBootstrapModel(engineState.model?.id);

  const sidebarContent = (
    <div className="flex h-full w-full flex-col bg-white dark:bg-[#0F141C]">
      <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3 lg:hidden dark:border-slate-800">
        <span className="font-display text-sm font-semibold tracking-tight text-slate-900 dark:text-slate-100">
          Workspace
        </span>
        <button
          type="button"
          onClick={onCloseMobile}
          aria-label="Close sidebar"
          className="inline-flex h-8 w-8 items-center justify-center rounded-md text-slate-500 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="flex min-h-0 flex-1 flex-col divide-y divide-slate-200 overflow-hidden dark:divide-slate-800">
        {/* Section 1: Threads */}
        <ThreadListPrimitive.Root
          className={`flex flex-col overflow-hidden p-3 ${threadsExpanded ? 'min-h-0 flex-1' : 'shrink-0'}`}
        >
          <div className="mb-2 flex items-center justify-between">
            <button
              type="button"
              aria-expanded={threadsExpanded}
              onClick={() => setThreadsExpanded((prev) => !prev)}
              className="inline-flex items-center gap-1 text-xs font-semibold tracking-tight text-slate-700 hover:text-slate-950 dark:text-slate-300 dark:hover:text-white"
            >
              {threadsExpanded ? (
                <ChevronDown className="h-3.5 w-3.5 text-slate-400" />
              ) : (
                <ChevronRight className="h-3.5 w-3.5 text-slate-400" />
              )}
              <span>Threads</span>
            </button>
          </div>

          <div
            onClick={() => {
              onSelectThreadView();
              onCloseMobile();
            }}
          >
            <ThreadListPrimitive.New className="flex h-9 w-full items-center justify-center gap-2 rounded-lg border border-slate-200 bg-[#F8F9FA] px-3 text-xs font-semibold text-slate-900 hover:border-slate-300 hover:bg-slate-100 dark:border-slate-800 dark:bg-[#141B26] dark:text-slate-100 dark:hover:bg-slate-800 transition-colors whitespace-nowrap">
              <Plus className="h-3.5 w-3.5" />
              <span>New Thread</span>
            </ThreadListPrimitive.New>
          </div>

          {threadsExpanded && (
            <nav
              aria-label="Conversation threads"
              onClick={() => {
                onSelectThreadView();
                onCloseMobile();
              }}
              className="mt-2.5 flex-1 space-y-1 overflow-y-auto"
            >
              <ThreadListPrimitive.Items components={{ ThreadListItem }} />

              {archivedCount > 0 && (
                <div className="mt-3 border-t border-slate-100 pt-2.5 dark:border-slate-800/80">
                  <button
                    type="button"
                    aria-expanded={showArchived}
                    onClick={(e) => {
                      e.stopPropagation();
                      setShowArchived((prev) => !prev);
                    }}
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
          )}
        </ThreadListPrimitive.Root>

        {/* Section 2: Spaces (each Space is a Verdant Prism) */}
        <section
          aria-label="Spaces"
          className={`flex flex-col overflow-hidden p-3 ${spacesExpanded ? 'min-h-0 max-h-[34%] shrink-0' : 'shrink-0'}`}
        >
          <div className="mb-2 flex items-center justify-between">
            <button
              type="button"
              aria-expanded={spacesExpanded}
              onClick={() => setSpacesExpanded((prev) => !prev)}
              className="inline-flex items-center gap-1 text-xs font-semibold tracking-tight text-slate-700 hover:text-slate-950 dark:text-slate-300 dark:hover:text-white"
            >
              {spacesExpanded ? (
                <ChevronDown className="h-3.5 w-3.5 text-slate-400" />
              ) : (
                <ChevronRight className="h-3.5 w-3.5 text-slate-400" />
              )}
              <span>Spaces</span>
            </button>
            {spaces.length > 0 && (
              <span className="font-mono text-[10px] text-slate-400 tabular-nums">
                {spaces.length}
              </span>
            )}
          </div>

          <div className="flex items-center gap-1.5">
            <button
              type="button"
              data-testid="new-prism-btn"
              onClick={() => {
                onCreateSpace();
                onCloseMobile();
              }}
              className="flex h-8 min-w-0 flex-1 items-center justify-center gap-1.5 rounded-lg border border-emerald-700/30 bg-emerald-50/70 px-3 text-xs font-semibold text-emerald-950 hover:border-emerald-700/50 hover:bg-emerald-100/70 dark:border-emerald-500/30 dark:bg-emerald-950/40 dark:text-emerald-200 dark:hover:bg-emerald-950/70 transition-colors whitespace-nowrap"
            >
              <Layers className="h-3.5 w-3.5" />
              <span>New Space</span>
            </button>
            {onImportHoloFile && (
              <>
                <input
                  ref={holoFileRef}
                  type="file"
                  accept=".holo,.json,application/octet-stream,application/json"
                  className="hidden"
                  onChange={handleHoloPick}
                />
                <button
                  type="button"
                  onClick={() => holoFileRef.current?.click()}
                  title="Import a saved creation"
                  aria-label="Import creation"
                  className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-slate-200 bg-[#F8F9FA] text-slate-700 hover:border-slate-300 hover:bg-slate-100 dark:border-slate-800 dark:bg-[#141B26] dark:text-slate-300 dark:hover:bg-slate-800 transition-colors"
                >
                  <Upload className="h-3.5 w-3.5" />
                </button>
              </>
            )}
          </div>

          {spacesExpanded && (
            <div className="mt-2 space-y-1 overflow-y-auto">
              {spaces.length === 0 ? (
                <p className="px-2 py-1.5 text-[11px] leading-relaxed text-slate-400 dark:text-slate-500">
                  Group pictures, audio, documents, games, and utilities into custom Spaces.
                </p>
              ) : (
                spaces.map((p) => {
                  const isActive = activeView.kind === 'prism' && activeView.prismId === p.id;
                  const isPub = p.published !== null;
                  return (
                    <div
                      key={p.id}
                      className={`group flex items-center gap-1 rounded-lg px-2 py-1 text-sm transition-colors ${
                        isActive
                          ? 'bg-slate-200/75 font-medium text-slate-950 dark:bg-slate-800 dark:text-white'
                          : 'text-slate-700 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800/70'
                      }`}
                    >
                      <button
                        type="button"
                        onClick={() => {
                          onSelectPrism(p.id, isPub ? 'app' : 'studio');
                          onCloseMobile();
                        }}
                        className="flex min-w-0 flex-1 items-center gap-2 py-0.5 text-left text-[13px]"
                        title={p.draft.name}
                      >
                        <Layers className="h-3.5 w-3.5 shrink-0 text-emerald-700 dark:text-emerald-400" />
                        <span className="truncate">{p.draft.name}</span>
                        {(p.publishedHolo?.proved || p.sealedDraft?.proved) && (
                          <ShieldCheck
                            className="h-3.5 w-3.5 shrink-0 text-emerald-600 dark:text-emerald-400"
                            aria-label="Verified"
                          />
                        )}
                      </button>
                      <button
                        type="button"
                        aria-label={`Delete Space ${p.draft.name}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeletePrism(p.id);
                        }}
                        className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-slate-400 opacity-0 hover:bg-red-100 hover:text-red-700 group-hover:opacity-100 focus-visible:opacity-100 dark:hover:bg-red-950/60 dark:hover:text-red-300 transition-opacity"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  );
                })
              )}
            </div>
          )}
        </section>

        {/* Section 3: Global MediaType Galleries */}
        <section
          aria-label="Galleries"
          className="flex shrink-0 flex-col overflow-hidden p-3"
        >
          <div className="mb-1.5 flex items-center justify-between">
            <button
              type="button"
              aria-expanded={galleriesExpanded}
              onClick={() => setGalleriesExpanded((prev) => !prev)}
              className="inline-flex items-center gap-1 text-xs font-semibold tracking-tight text-slate-700 hover:text-slate-950 dark:text-slate-300 dark:hover:text-white"
            >
              {galleriesExpanded ? (
                <ChevronDown className="h-3.5 w-3.5 text-slate-400" />
              ) : (
                <ChevronRight className="h-3.5 w-3.5 text-slate-400" />
              )}
              <span>Galleries</span>
            </button>
          </div>

          {galleriesExpanded && (
            <nav aria-label="Media Galleries" className="space-y-0.5">
              {MEDIA_GALLERIES.map((g) => {
                const isActive = activeView.kind === 'gallery' && activeView.mediaType === g.mediaType;
                const count = galleryCounts[g.mediaType] ?? 0;
                const Icon =
                  g.mediaType === 'picture'
                    ? ImageIcon
                    : g.mediaType === 'video'
                      ? Video
                      : g.mediaType === 'audio'
                        ? Music
                        : g.mediaType === 'written-work'
                          ? FileText
                          : g.mediaType === 'interactive-game'
                            ? Gamepad2
                            : g.mediaType === 'research-program'
                              ? FlaskConical
                              : Wrench;
                return (
                  <button
                    key={g.mediaType}
                    type="button"
                    onClick={() => {
                      onSelectGallery(g.mediaType);
                      onCloseMobile();
                    }}
                    className={`flex w-full items-center justify-between rounded-lg px-2 py-1.5 text-left text-[13px] transition-colors ${
                      isActive
                        ? 'bg-emerald-50 font-medium text-emerald-950 dark:bg-emerald-950/50 dark:text-emerald-200'
                        : 'text-slate-700 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800/70'
                    }`}
                  >
                    <span className="inline-flex items-center gap-2 truncate">
                      <Icon className="h-3.5 w-3.5 shrink-0 text-slate-500 dark:text-slate-400" />
                      <span className="truncate">{g.label}</span>
                    </span>
                    {count > 0 && (
                      <span className="font-mono text-[10px] text-slate-400 tabular-nums">
                        {count}
                      </span>
                    )}
                  </button>
                );
              })}
            </nav>
          )}
        </section>
      </div>

      <div className="border-t border-slate-200 p-3.5 dark:border-slate-800">
        <div className="truncate text-xs font-semibold text-slate-900 dark:text-slate-100">
          VerdantUI
        </div>
        <div className="mt-0.5 truncate text-[11px] text-slate-500 dark:text-slate-400 tabular-nums">
          {isBootstrapUnbound
            ? engineState.modelStreamEnabled
              ? `Preparing AI · ${engineState.progressPercent}%`
              : 'On-device · Private'
            : 'Local AI ready · Private'}
        </div>

        <div className="mt-3">
          <button
            type="button"
            onClick={() => {
              onCloseMobile();
              onOpenInspector();
            }}
            className="inline-flex h-8 w-full items-center justify-center gap-1.5 rounded-md border border-slate-200 bg-white px-2.5 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-200 dark:hover:bg-slate-800 transition-colors whitespace-nowrap"
          >
            <SlidersHorizontal className="h-3.5 w-3.5" />
            <span>Settings</span>
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
