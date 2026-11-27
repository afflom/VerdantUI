/**
 * Global MediaType Gallery Prism (`src/components/GalleryView.tsx`)
 *
 * Every MediaType Gallery (Pictures, Videos, Audio, Documents, Games, Research, Utilities)
 * is itself a Verdant Prism rendered through the universal `PrismSurfaceShell`:
 *   - Clean live Preview grid matching the serene homescreen aesthetic
 *   - `Preview | Details` toggle
 *   - Natural Language Interface bar to create, filter, or search in plain English (`DEC-015`).
 */

import { useMemo, useState, type FC } from 'react';
import {
  ExternalLink,
  FileText,
  FlaskConical,
  FolderPlus,
  Gamepad2,
  Image as ImageIcon,
  Layers,
  Music,
  Play,
  Plus,
  Search,
  Sparkles,
  Trash2,
  Video,
  Wrench,
} from 'lucide-react';
import {
  MEDIA_GALLERIES,
  type ContentArtifactRecord,
  type GalleryMediaType,
} from '../engine/contentGalleryStore';
import type {
  PrismSurfaceMode,
  SynthesizedPrismFrame,
} from '../engine/observerUiSynthesizer';
import type { VerdantPrismProject } from '../engine/prismStudio';
import { PrismSurfaceShell, type PrismSurfaceMessage } from './PrismSurfaceShell';

interface GalleryViewProps {
  mediaType: GalleryMediaType;
  frame: SynthesizedPrismFrame;
  surfaceMode: PrismSurfaceMode;
  onChangeSurfaceMode: (mode: PrismSurfaceMode) => void;
  onSelectSystemPrism: (prismId: string) => void;
  onTypingChange?: (typing: boolean) => void;
  items: readonly ContentArtifactRecord[];
  categories: readonly string[];
  spaces: readonly VerdantPrismProject[];
  onCreateInGallery: (mediaType: GalleryMediaType) => void;
  onCreateFromPrompt: (mediaType: GalleryMediaType, prompt: string) => Promise<string>;
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
  frame,
  surfaceMode,
  onChangeSurfaceMode,
  onSelectSystemPrism,
  onTypingChange,
  items,
  categories,
  spaces,
  onCreateInGallery,
  onCreateFromPrompt,
  onOpenArtifact,
  onDeleteArtifact,
  onAssignToSpace,
  onOpenSpace,
}) => {
  const [activeCategory, setActiveCategory] = useState<string>('all');
  const [spaceFilter, setSpaceFilter] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [playingAudioUrl, setPlayingAudioUrl] = useState<Record<string, string>>({});
  const [nlBusy, setNlBusy] = useState(false);
  const [nlMessages, setNlMessages] = useState<PrismSurfaceMessage[]>([]);

  const handleQuickPlayAudio = async (item: ContentArtifactRecord) => {
    if (playingAudioUrl[item.id]) return;
    const { composeMultiTrackArrangement, synthesizeCompositionStereoPcm, encodeStereoWav, encodeStereoWavDataUrl } =
      await import('../engine/audioSynthesisEngine');
    const comp = composeMultiTrackArrangement({
      title: item.title,
      style: item.categories.includes('Beats & Rhythm')
        ? 'rhythm-pulse'
        : item.categories.includes('Melodic')
          ? 'melodic-arpeggio'
          : 'ambient-pad',
      bpm: 92,
      rootMidi: 60,
      scale: 'pentatonic',
      durationSec: Math.min(8, item.audioDurationSec ?? 6),
    });
    const pcm = synthesizeCompositionStereoPcm(comp, 22_050);
    const wav = encodeStereoWav(pcm.left, pcm.right, pcm.sampleRate);
    setPlayingAudioUrl((prev) => ({ ...prev, [item.id]: encodeStereoWavDataUrl(wav) }));
  };

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

  const handleSubmitNaturalLanguage = async (raw: string) => {
    const prompt = raw.trim();
    if (!prompt) return;
    const lower = prompt.toLowerCase();
    const userMsg: PrismSurfaceMessage = { id: `u_${Date.now()}`, role: 'user', content: prompt };
    setNlMessages((prev) => [...prev.slice(-14), userMsg]);

    if (/^(?:show|view|open)\s+(?:prism\s+)?details\b/.test(lower)) {
      onChangeSurfaceMode('details');
      setNlMessages((prev) => [...prev, { id: `a_${Date.now()}`, role: 'assistant', content: `Showing ${descriptor.label} Prism details.` }]);
      return;
    }
    if (/^(?:show\s+all|clear|reset)\b/.test(lower)) {
      setActiveCategory('all');
      setSpaceFilter('all');
      setSearchQuery('');
      setNlMessages((prev) => [...prev, { id: `a_${Date.now()}`, role: 'assistant', content: `Showing all ${items.length} ${descriptor.label.toLowerCase()}.` }]);
      return;
    }
    const filterMatch = lower.match(/^(?:filter|search|find|show\s+only)\s+(?:by\s+|for\s+)?(.+)$/);
    if (filterMatch) {
      const term = filterMatch[1]!.trim();
      const matchedCat = categories.find((c) => c.toLowerCase().includes(term));
      if (matchedCat) {
        setActiveCategory(matchedCat);
        setSearchQuery('');
        setNlMessages((prev) => [...prev, { id: `a_${Date.now()}`, role: 'assistant', content: `Filtered ${descriptor.label} to "${matchedCat}".` }]);
        return;
      }
      setActiveCategory('all');
      setSearchQuery(term);
      setNlMessages((prev) => [...prev, { id: `a_${Date.now()}`, role: 'assistant', content: `Filtered ${descriptor.label} by "${term}".` }]);
      return;
    }

    // Default: synthesize a new artifact in this Gallery's MediaType directly!
    setNlBusy(true);
    try {
      const summary = await onCreateFromPrompt(mediaType, prompt);
      setNlMessages((prev) => [...prev, { id: `a_${Date.now()}`, role: 'assistant', content: summary }]);
    } finally {
      setNlBusy(false);
    }
  };

  const headerActions = (
    <button
      type="button"
      onClick={() => onCreateInGallery(mediaType)}
      className="inline-flex h-7 items-center gap-1 rounded-md bg-[#0F5132] px-2.5 text-xs font-medium text-white hover:bg-[#0B3D25] transition-colors"
    >
      <Plus className="h-3.5 w-3.5" />
      <span>New {descriptor.singularLabel}</span>
    </button>
  );

  const previewContent = (
    <div className="flex h-full flex-col overflow-hidden bg-[#F8F9FA] dark:bg-[#0B0F14]">
      {/* Compact filter strip */}
      {(items.length > 0 || categories.length > 0) && (
        <div className="shrink-0 border-b border-slate-200/70 bg-white/70 px-5 py-2.5 dark:border-slate-800/70 dark:bg-[#0F141C]/70">
          <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-2.5">
            <div className="flex flex-wrap items-center gap-1.5">
              <button
                type="button"
                onClick={() => setActiveCategory('all')}
                className={`rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
                  activeCategory === 'all'
                    ? 'bg-[#0F5132] text-white'
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
                      ? 'bg-[#0F5132] text-white'
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
      )}

      <div className="flex-1 overflow-y-auto p-5">
        <div className="mx-auto max-w-6xl">
          {filtered.length === 0 ? (
            <div className="mx-auto mt-8 flex max-w-lg flex-col items-center justify-center rounded-2xl border border-slate-200 bg-white px-8 py-14 text-center shadow-xs dark:border-slate-800 dark:bg-[#0F141C]">
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-emerald-50 text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-400">
                <Icon className="h-6 w-6" />
              </div>
              <h2 className="mt-4 text-base font-semibold text-slate-900 dark:text-slate-100">
                No {descriptor.label.toLowerCase()} here yet
              </h2>
              <p className="mt-1.5 text-sm leading-relaxed text-slate-500 dark:text-slate-400">
                Describe what you want in the prompt bar below to create a {descriptor.singularLabel.toLowerCase()}, or start a blank one right away.
              </p>
              <button
                type="button"
                onClick={() => onCreateInGallery(mediaType)}
                className="mt-5 inline-flex items-center gap-1.5 rounded-lg bg-[#0F5132] px-4 py-2 text-xs font-semibold text-white hover:bg-[#0B3D25]"
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

                    {item.mediaType === 'audio' && (
                      <div className="mt-2.5">
                        {playingAudioUrl[item.id] ? (
                          <audio controls autoPlay src={playingAudioUrl[item.id]} className="h-8 w-full" />
                        ) : (
                          <button
                            type="button"
                            onClick={() => void handleQuickPlayAudio(item)}
                            className="inline-flex items-center gap-1.5 rounded-md border border-emerald-700/30 bg-emerald-50/70 px-2.5 py-1 text-[11px] font-medium text-emerald-900 hover:bg-emerald-100 dark:border-emerald-500/30 dark:bg-emerald-950/40 dark:text-emerald-200"
                          >
                            <Play className="h-3 w-3" />
                            <span>Quick Preview</span>
                          </button>
                        )}
                      </div>
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
                      <button
                        type="button"
                        onClick={() => onOpenArtifact(item.id, item.published ? 'app' : 'studio')}
                        className="inline-flex items-center gap-1 rounded-md bg-[#0F5132] px-2.5 py-1 text-[11px] font-medium text-white hover:bg-[#0B3D25]"
                      >
                        <ExternalLink className="h-3 w-3" />
                        <span>Open</span>
                      </button>

                      {spaces.length > 0 && (
                        <label className="inline-flex items-center gap-1 text-[11px] text-slate-500">
                          <FolderPlus className="h-3 w-3 shrink-0" />
                          <select
                            aria-label={`Add ${item.title} to Space`}
                            value=""
                            onChange={(e) => {
                              if (e.target.value) onAssignToSpace(item.id, e.target.value);
                            }}
                            className="rounded border border-slate-200 bg-transparent px-1.5 py-0.5 text-[11px] text-slate-600 dark:border-slate-700 dark:text-slate-300"
                          >
                            <option value="">+ Space</option>
                            {spaces.map((sp) => (
                              <option key={sp.id} value={sp.id}>
                                {sp.published?.name ?? sp.draft.name}
                              </option>
                            ))}
                          </select>
                        </label>
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

  const detailsExtra = (
    <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-[#0F141C]">
      <div className="text-xs font-semibold text-slate-900 dark:text-slate-100">
        Gallery Index
      </div>
      <div className="mt-2 grid grid-cols-2 gap-3 text-xs">
        <div>
          <div className="text-[11px] text-slate-500 dark:text-slate-400">Indexed Items</div>
          <div className="mt-0.5 font-semibold text-slate-900 dark:text-slate-100 tabular-nums">
            {items.length}
          </div>
        </div>
        <div>
          <div className="text-[11px] text-slate-500 dark:text-slate-400">Categories</div>
          <div className="mt-0.5 font-semibold text-slate-900 dark:text-slate-100 tabular-nums">
            {categories.length}
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <PrismSurfaceShell
      frame={frame}
      mode={surfaceMode}
      onChangeMode={onChangeSurfaceMode}
      headerActions={headerActions}
      previewContent={previewContent}
      detailsExtra={detailsExtra}
      onSelectChildPrism={onSelectSystemPrism}
      onSubmitNaturalLanguage={handleSubmitNaturalLanguage}
      nlBusy={nlBusy}
      nlBusyLabel={`Creating ${descriptor.singularLabel.toLowerCase()}…`}
      nlMessages={nlMessages}
      onTypingChange={onTypingChange}
    />
  );
};
