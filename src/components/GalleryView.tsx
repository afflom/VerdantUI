/**
 * Global MediaType Gallery View (`src/components/GalleryView.tsx`)
 *
 * Displays all user artifacts of a given top-level MediaType across all Spaces:
 *   Pictures | Videos | Audio | Documents | Games | Research | Utilities
 * with organic sub-category filtering, Space filtering, Space assignment,
 * instant preview thumbnails, and zero technical jargon (`DEC-011`, `DEC-014`).
 */

import { useMemo, useState, type FC } from 'react';
import {
  ExternalLink,
  FolderPlus,
  Image as ImageIcon,
  Layers,
  Music,
  Plus,
  Search,
  Sparkles,
  Trash2,
  Video,
  FileText,
  Gamepad2,
  FlaskConical,
  Wrench,
} from 'lucide-react';
import {
  MEDIA_GALLERIES,
  type ContentArtifactRecord,
  type GalleryMediaType,
} from '../engine/contentGalleryStore';
import type { VerdantPrismProject } from '../engine/prismStudio';

interface GalleryViewProps {
  mediaType: GalleryMediaType;
  items: readonly ContentArtifactRecord[];
  categories: readonly string[];
  spaces: readonly VerdantPrismProject[];
  onCreateInGallery: (mediaType: GalleryMediaType) => void;
  onOpenArtifact: (id: string, mode: 'app' | 'studio') => void;
  onDeleteArtifact: (id: string) => void;
  onAssignToSpace: (artifactId: string, spaceId: string) => void;
  onOpenSpace: (spaceId: string) => void;
}

function galleryIcon(mt: GalleryMediaType) {
  switch (mt) {
    case 'picture':
      return ImageIcon;
    case 'video':
      return Video;
    case 'audio':
      return Music;
    case 'written-work':
      return FileText;
    case 'interactive-game':
      return Gamepad2;
    case 'research-program':
      return FlaskConical;
    case 'tool-organ':
      return Wrench;
  }
}

export const GalleryView: FC<GalleryViewProps> = ({
  mediaType,
  items,
  categories,
  spaces,
  onCreateInGallery,
  onOpenArtifact,
  onDeleteArtifact,
  onAssignToSpace,
  onOpenSpace,
}) => {
  const [activeCategory, setActiveCategory] = useState<string>('all');
  const [spaceFilter, setSpaceFilter] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');

  const descriptor = useMemo(
    () => MEDIA_GALLERIES.find((g) => g.mediaType === mediaType) ?? MEDIA_GALLERIES[0]!,
    [mediaType],
  );
  const Icon = galleryIcon(mediaType);

  const spaceNameById = useMemo(() => {
    const m = new Map<string, string>();
    for (const sp of spaces) m.set(sp.id, sp.published?.name ?? sp.draft.name);
    return m;
  }, [spaces]);

  const filtered = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return items.filter((it) => {
      if (activeCategory !== 'all' && !it.categories.includes(activeCategory)) return false;
      if (spaceFilter !== 'all' && !it.spaceIds.includes(spaceFilter)) return false;
      if (q) {
        const hay = `${it.title} ${it.summary} ${it.categories.join(' ')}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }, [items, activeCategory, spaceFilter, searchQuery]);

  return (
    <div className="flex h-full flex-col overflow-hidden bg-[#F8F9FA] dark:bg-[#0B0F14]">
      <div className="shrink-0 border-b border-slate-200 bg-white px-5 py-4 dark:border-slate-800 dark:bg-[#0F141C]">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="inline-flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-50 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-400">
              <Icon className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-base font-semibold text-slate-950 dark:text-white">
                {descriptor.label}
              </h1>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                {descriptor.description}
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={() => onCreateInGallery(mediaType)}
            className="inline-flex h-9 items-center gap-1.5 rounded-lg bg-[#0F5132] px-3.5 text-xs font-semibold text-white hover:bg-[#0B3D25] transition-colors"
          >
            <Plus className="h-3.5 w-3.5" />
            <span>New {descriptor.singularLabel}</span>
          </button>
        </div>

        <div className="mx-auto mt-3 flex max-w-6xl flex-wrap items-center justify-between gap-2.5">
          <div className="flex flex-wrap items-center gap-1.5">
            <button
              type="button"
              onClick={() => setActiveCategory('all')}
              className={`rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
                activeCategory === 'all'
                  ? 'bg-emerald-700 text-white'
                  : 'border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-300'
              }`}
            >
              All ({items.length})
            </button>
            {categories.map((cat) => (
              <button
                key={cat}
                type="button"
                onClick={() => setActiveCategory(cat)}
                className={`rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
                  activeCategory === cat
                    ? 'bg-emerald-700 text-white'
                    : 'border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-300'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2">
            {spaces.length > 0 && (
              <select
                aria-label="Filter by Space"
                value={spaceFilter}
                onChange={(e) => setSpaceFilter(e.target.value)}
                className="h-8 rounded-lg border border-slate-200 bg-white px-2.5 text-xs text-slate-700 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-200"
              >
                <option value="all">All Spaces</option>
                {spaces.map((sp) => (
                  <option key={sp.id} value={sp.id}>
                    {sp.published?.name ?? sp.draft.name}
                  </option>
                ))}
              </select>
            )}
            <div className="relative">
              <Search className="pointer-events-none absolute top-2 left-2.5 h-3.5 w-3.5 text-slate-400" />
              <input
                type="search"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder={`Search ${descriptor.label.toLowerCase()}…`}
                className="h-8 w-44 rounded-lg border border-slate-200 bg-white pr-2.5 pl-8 text-xs text-slate-800 placeholder:text-slate-400 focus:border-emerald-600 focus:outline-none dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-100"
              />
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-5">
        <div className="mx-auto max-w-6xl">
          {filtered.length === 0 ? (
            <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-slate-300 bg-white px-6 py-16 text-center dark:border-slate-800 dark:bg-[#0F141C]">
              <Icon className="h-9 w-9 text-emerald-700/70 dark:text-emerald-400/70" />
              <h2 className="mt-3 text-sm font-semibold text-slate-900 dark:text-slate-100">
                No {descriptor.label.toLowerCase()} here yet
              </h2>
              <p className="mt-1 max-w-md text-xs text-slate-500 dark:text-slate-400">
                Ask VerdantUI in any chat thread to create a {descriptor.singularLabel.toLowerCase()}, or start one right here. Everything you create is indexed automatically.
              </p>
              <button
                type="button"
                onClick={() => onCreateInGallery(mediaType)}
                className="mt-4 inline-flex items-center gap-1.5 rounded-lg bg-[#0F5132] px-3.5 py-2 text-xs font-semibold text-white hover:bg-[#0B3D25]"
              >
                <Sparkles className="h-3.5 w-3.5" />
                <span>Create {descriptor.singularLabel}</span>
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {filtered.map((item) => (
                <article
                  key={item.id}
                  className="group flex flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-xs transition-shadow hover:shadow-md dark:border-slate-800 dark:bg-[#0F141C]"
                >
                  {item.previewDataUrl ? (
                    <button
                      type="button"
                      onClick={() => onOpenArtifact(item.id, item.published ? 'app' : 'studio')}
                      className="relative aspect-[16/10] w-full overflow-hidden bg-slate-950 text-left"
                    >
                      <img
                        src={item.previewDataUrl}
                        alt={item.title}
                        className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.02]"
                      />
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={() => onOpenArtifact(item.id, item.published ? 'app' : 'studio')}
                      className="flex aspect-[16/8] w-full items-center justify-center bg-gradient-to-br from-slate-100 to-emerald-50/70 text-emerald-700/70 dark:from-[#121923] dark:to-emerald-950/30 dark:text-emerald-400/70"
                    >
                      <Icon className="h-10 w-10" />
                    </button>
                  )}

                  <div className="flex flex-1 flex-col p-3.5">
                    <div className="flex items-start justify-between gap-2">
                      <h3 className="truncate text-sm font-semibold text-slate-950 dark:text-white">
                        {item.title}
                      </h3>
                      <button
                        type="button"
                        aria-label={`Delete ${item.title}`}
                        onClick={() => onDeleteArtifact(item.id)}
                        className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-slate-400 opacity-0 hover:bg-red-50 hover:text-red-700 group-hover:opacity-100 dark:hover:bg-red-950/40 dark:hover:text-red-300"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                    {item.summary && (
                      <p className="mt-1 line-clamp-2 text-xs text-slate-500 dark:text-slate-400">
                        {item.summary}
                      </p>
                    )}

                    <div className="mt-2.5 flex flex-wrap gap-1">
                      {item.categories.map((cat) => (
                        <span
                          key={cat}
                          className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-medium text-slate-600 dark:bg-slate-800 dark:text-slate-300"
                        >
                          {cat}
                        </span>
                      ))}
                      {item.spaceIds.map((sid) => {
                        const sName = spaceNameById.get(sid);
                        if (!sName) return null;
                        return (
                          <button
                            key={sid}
                            type="button"
                            onClick={() => onOpenSpace(sid)}
                            className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-0.5 text-[10px] font-medium text-emerald-800 hover:bg-emerald-100 dark:bg-emerald-950/60 dark:text-emerald-300"
                          >
                            <Layers className="h-2.5 w-2.5" />
                            <span>{sName}</span>
                          </button>
                        );
                      })}
                    </div>

                    <div className="mt-auto flex items-center justify-between gap-2 pt-3.5">
                      <div className="flex items-center gap-1.5">
                        <button
                          type="button"
                          onClick={() => onOpenArtifact(item.id, item.published ? 'app' : 'studio')}
                          className="inline-flex items-center gap-1 rounded-md bg-[#0F5132] px-2.5 py-1 text-[11px] font-medium text-white hover:bg-[#0B3D25]"
                        >
                          <ExternalLink className="h-3 w-3" />
                          <span>Open</span>
                        </button>
                        <button
                          type="button"
                          onClick={() => onOpenArtifact(item.id, 'studio')}
                          className="rounded-md border border-slate-200 bg-white px-2 py-1 text-[11px] font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-200"
                        >
                          Studio
                        </button>
                      </div>

                      {spaces.length > 0 && (
                        <div className="relative inline-flex items-center">
                          <FolderPlus className="pointer-events-none absolute left-2 h-3 w-3 text-slate-400" />
                          <select
                            aria-label={`Add ${item.title} to Space`}
                            defaultValue=""
                            onChange={(e) => {
                              if (e.target.value) {
                                onAssignToSpace(item.id, e.target.value);
                                e.target.value = '';
                              }
                            }}
                            className="h-7 rounded-md border border-slate-200 bg-white pr-2 pl-6 text-[11px] text-slate-600 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-300"
                          >
                            <option value="">Add to Space</option>
                            {spaces.map((sp) => (
                              <option key={sp.id} value={sp.id}>
                                {sp.published?.name ?? sp.draft.name}
                              </option>
                            ))}
                          </select>
                        </div>
                      )}
                    </div>
                  </div>
                </article>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
