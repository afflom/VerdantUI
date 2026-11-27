/**
 * Content Streams Settings Section (`src/components/ContentStreamsSection.tsx`)
 *
 * Lazy-loaded inside the Settings view so the main bundle stays compact.
 * Provides an uncluttered, scalable UI for:
 * - Offline-Only Mode toggle (controls all browser streaming globally)
 * - Live search + filter pills (All · Active · Custom)
 * - Inline collapsible "+ Add Stream" drawer with HTTPS validation & CORS reachability check
 * - Collapsible Catalog accordions (all built-in streams opt-in / off by default)
 */

import { useCallback, useMemo, useState, useSyncExternalStore, type FC } from 'react';
import {
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Globe,
  Loader2,
  Plus,
  Search,
  Trash2,
  Wifi,
  WifiOff,
  X,
} from 'lucide-react';
import {
  normalizeCustomStreamInput,
  type ContentStreamCatalogStore,
  type ContentStreamProviderKind,
  type ContentStreamSpec,
  type StreamCatalogId,
} from '../engine/contentStreamCatalog';
import type { StreamingConsentState } from '../engine/workerClient';

export interface ContentStreamsSectionProps {
  readonly contentStreams: ContentStreamCatalogStore;
  readonly streamingConsent: StreamingConsentState;
  readonly onSetStreamingConsent: (next: 'accepted' | 'paused') => void;
}

type StreamFilterMode = 'all' | 'active' | 'custom';
type CustomFormatHint = 'auto' | 'rss' | 'json-api' | 'github' | 'web-url';

const PROVIDER_LABELS: Readonly<Record<ContentStreamProviderKind, string>> = {
  wikipedia: 'Encyclopedia',
  wikidata: 'Knowledge Graph',
  wiktionary: 'Dictionary',
  openalex: 'Research',
  arxiv: 'Preprints',
  crossref: 'Citations',
  github: 'GitHub',
  npm: 'Packages',
  crates: 'Packages',
  mdn: 'Docs',
  'open-meteo': 'Weather',
  wikimedia: 'Media',
  osm: 'Places',
  rss: 'RSS / Atom',
  'json-api': 'JSON Feed',
  'web-url': 'Web Page',
};

function formatRelativeSync(ts: number | null): string | null {
  if (!ts || !Number.isFinite(ts)) return null;
  const diffSec = Math.max(0, Math.floor((Date.now() - ts) / 1000));
  if (diffSec < 60) return 'Updated just now';
  const mins = Math.floor(diffSec / 60);
  if (mins < 60) return `Updated ${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `Updated ${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `Updated ${days}d ago`;
}

export const ContentStreamsSection: FC<ContentStreamsSectionProps> = ({
  contentStreams,
  streamingConsent,
  onSetStreamingConsent,
}) => {
  const snapshot = useSyncExternalStore(
    useCallback((cb) => contentStreams.subscribe(cb), [contentStreams]),
    useCallback(() => contentStreams.getSnapshot(), [contentStreams]),
  );

  const [searchQuery, setSearchQuery] = useState('');
  const [filterMode, setFilterMode] = useState<StreamFilterMode>('all');
  const [expandedCatalogs, setExpandedCatalogs] = useState<ReadonlySet<StreamCatalogId>>(
    () => new Set<StreamCatalogId>(),
  );
  const [addDrawerOpen, setAddDrawerOpen] = useState(false);
  const [customName, setCustomName] = useState('');
  const [customUrl, setCustomUrl] = useState('');
  const [customFormat, setCustomFormat] = useState<CustomFormatHint>('auto');
  const [customError, setCustomError] = useState<string | null>(null);
  const [testingUrl, setTestingUrl] = useState(false);
  const [testNotice, setTestNotice] = useState<{ ok: boolean; text: string } | null>(null);

  const isOfflineOnly = streamingConsent === 'paused';
  const enabledCount = snapshot.enabledCount;
  const customCount = snapshot.customCount;

  const filteredGroups = useMemo(() => {
    const groups = contentStreams.searchCatalogs(searchQuery, filterMode);
    if (filterMode === 'custom' && groups.length === 0) {
      const customMeta = snapshot.catalogs.find((c) => c.id === 'custom');
      return customMeta ? [{ catalog: customMeta, streams: [] as readonly ContentStreamSpec[] }] : [];
    }
    return groups;
  }, [contentStreams, snapshot, searchQuery, filterMode]);

  const forceExpand = searchQuery.trim().length > 0 || filterMode !== 'all';

  const toggleCatalogExpanded = (catalogId: StreamCatalogId) => {
    setExpandedCatalogs((prev) => {
      const next = new Set(prev);
      if (next.has(catalogId)) next.delete(catalogId);
      else next.add(catalogId);
      return next;
    });
  };

  const handleTestCustomUrl = async () => {
    setCustomError(null);
    setTestNotice(null);
    const check = normalizeCustomStreamInput({
      name: customName,
      url: customUrl,
      formatHint: customFormat,
    });
    if (!check.ok) {
      setCustomError(check.error);
      return;
    }
    setTestingUrl(true);
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 5000);
      try {
        const resp = await fetch(check.spec.endpointUrl, {
          method: 'GET',
          signal: ctrl.signal,
        });
        if (resp.ok) {
          setTestNotice({
            ok: true,
            text: `Reachable (${PROVIDER_LABELS[check.spec.providerKind] ?? 'Source'}) and ready to add.`,
          });
        } else {
          setTestNotice({
            ok: false,
            text: `Source returned status ${resp.status}. You can still save it if it accepts queries.`,
          });
        }
      } finally {
        clearTimeout(timer);
      }
    } catch {
      setTestNotice({
        ok: false,
        text: 'Could not reach this address from the browser. Some sites block cross-origin browser requests.',
      });
    } finally {
      setTestingUrl(false);
    }
  };

  const handleAddCustomStream = () => {
    setCustomError(null);
    setTestNotice(null);
    const res = contentStreams.addCustomStream({
      name: customName,
      url: customUrl,
      formatHint: customFormat,
    });
    if (!res.ok) {
      setCustomError(res.error);
      return;
    }
    setCustomName('');
    setCustomUrl('');
    setCustomFormat('auto');
    setAddDrawerOpen(false);
    setExpandedCatalogs((prev) => {
      const next = new Set(prev);
      next.add('custom');
      return next;
    });
  };

  return (
    <section className="space-y-3" aria-label="Content Streams">
      <div className="flex items-center justify-between">
        <h3 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
          Content Streams
        </h3>
        <span className="text-[11px] text-slate-500 dark:text-slate-400 tabular-nums">
          {enabledCount} enabled
        </span>
      </div>

      {/* Offline-Only Mode control */}
      <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-xs dark:border-slate-800 dark:bg-[#0F141C]">
        <div className="flex items-center justify-between gap-3">
          <div className="flex min-w-0 items-start gap-2.5">
            <div
              className={`mt-0.5 inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${
                isOfflineOnly
                  ? 'bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400'
                  : 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-400'
              }`}
            >
              {isOfflineOnly ? <WifiOff className="h-4 w-4" /> : <Wifi className="h-4 w-4" />}
            </div>
            <div className="min-w-0">
              <div className="text-sm font-medium text-slate-900 dark:text-slate-100">
                Offline-Only Mode
              </div>
              <div className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                {isOfflineOnly
                  ? 'Browser streaming is paused. Only cached capabilities and saved creations run.'
                  : streamingConsent === 'pending'
                    ? 'Browser streaming has not been confirmed yet.'
                    : enabledCount > 0
                      ? `Streaming active · ${enabledCount} content stream${enabledCount === 1 ? '' : 's'} enabled.`
                      : 'Streaming active · Enable any catalog sources below to include live knowledge.'}
              </div>
            </div>
          </div>

          <button
            type="button"
            role="switch"
            aria-checked={isOfflineOnly}
            aria-label="Offline-Only Mode"
            data-testid="offline-only-toggle"
            onClick={() => onSetStreamingConsent(isOfflineOnly ? 'accepted' : 'paused')}
            className={`relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-700 ${
              isOfflineOnly
                ? 'bg-slate-700 dark:bg-emerald-600'
                : 'bg-slate-200 dark:bg-slate-700'
            }`}
          >
            <span
              className={`inline-block h-4.5 w-4.5 transform rounded-full bg-white shadow-xs transition-transform ${
                isOfflineOnly ? 'translate-x-5.5' : 'translate-x-1'
              }`}
            />
          </button>
        </div>
      </div>

      {/* Search, filter pills, and inline + Add Stream action */}
      <div className="rounded-2xl border border-slate-200 bg-white p-3.5 shadow-xs dark:border-slate-800 dark:bg-[#0F141C]">
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative min-w-[180px] flex-1">
            <Search className="pointer-events-none absolute top-1/2 left-2.5 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search catalogs and streams…"
              aria-label="Search catalogs and streams"
              className="h-8.5 w-full rounded-lg border border-slate-200 bg-slate-50 pr-7 pl-8 text-xs text-slate-900 outline-none placeholder:text-slate-400 focus:border-emerald-700 focus:bg-white dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-100 dark:focus:border-emerald-500"
            />
            {searchQuery && (
              <button
                type="button"
                aria-label="Clear stream search"
                onClick={() => setSearchQuery('')}
                className="absolute top-1/2 right-2 -translate-y-1/2 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </div>

          <div className="inline-flex rounded-lg border border-slate-200 bg-slate-50 p-0.5 text-[11px] font-medium dark:border-slate-700 dark:bg-[#141B26]">
            {(
              [
                { id: 'all', label: 'All' },
                { id: 'active', label: `Active${enabledCount > 0 ? ` (${enabledCount})` : ''}` },
                { id: 'custom', label: `Custom${customCount > 0 ? ` (${customCount})` : ''}` },
              ] as const
            ).map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setFilterMode(tab.id)}
                className={`rounded-md px-2.5 py-1 transition-colors ${
                  filterMode === tab.id
                    ? 'bg-white text-slate-900 shadow-2xs dark:bg-slate-800 dark:text-white'
                    : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <button
            type="button"
            data-testid="add-stream-toggle"
            aria-expanded={addDrawerOpen}
            onClick={() => {
              setAddDrawerOpen((prev) => !prev);
              setCustomError(null);
              setTestNotice(null);
            }}
            className="inline-flex h-8.5 items-center gap-1 rounded-lg bg-[#0F5132] px-3 text-xs font-medium text-white hover:bg-[#0B3D25] transition-colors whitespace-nowrap"
          >
            <Plus className="h-3.5 w-3.5" />
            <span>Add Stream</span>
          </button>
        </div>

        {/* Inline collapsible "+ Add Stream" drawer */}
        {addDrawerOpen && (
          <div
            data-testid="add-stream-drawer"
            className="mt-3 space-y-2.5 rounded-xl border border-emerald-700/25 bg-emerald-50/40 p-3 dark:border-emerald-500/25 dark:bg-emerald-950/20"
          >
            <div className="flex items-center justify-between">
              <div className="text-xs font-semibold text-slate-900 dark:text-slate-100">
                Add a Custom Content Stream
              </div>
              <button
                type="button"
                aria-label="Close add stream form"
                onClick={() => setAddDrawerOpen(false)}
                className="text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>

            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              <input
                type="text"
                value={customName}
                onChange={(e) => setCustomName(e.target.value)}
                placeholder="Name (optional)"
                aria-label="Custom stream name"
                className="h-8.5 rounded-lg border border-slate-200 bg-white px-2.5 text-xs text-slate-900 outline-none focus:border-emerald-700 dark:border-slate-700 dark:bg-[#0F141C] dark:text-slate-100"
              />
              <select
                value={customFormat}
                onChange={(e) => setCustomFormat(e.target.value as CustomFormatHint)}
                aria-label="Stream format"
                className="h-8.5 rounded-lg border border-slate-200 bg-white px-2.5 text-xs text-slate-900 outline-none focus:border-emerald-700 dark:border-slate-700 dark:bg-[#0F141C] dark:text-slate-100"
              >
                <option value="auto">Auto-detect format</option>
                <option value="rss">RSS / Atom Feed</option>
                <option value="json-api">Public JSON API</option>
                <option value="github">GitHub Repository</option>
                <option value="web-url">Public Web Page</option>
              </select>
            </div>

            <input
              type="url"
              value={customUrl}
              onChange={(e) => {
                setCustomUrl(e.target.value);
                setCustomError(null);
              }}
              placeholder="https://example.org/feed.xml or https://github.com/owner/repo"
              aria-label="Stream HTTPS URL"
              className="h-8.5 w-full rounded-lg border border-slate-200 bg-white px-2.5 text-xs text-slate-900 outline-none focus:border-emerald-700 dark:border-slate-700 dark:bg-[#0F141C] dark:text-slate-100"
            />

            {customError && (
              <p role="alert" className="text-[11px] text-red-600 dark:text-red-400">
                {customError}
              </p>
            )}

            {testNotice && (
              <p
                className={`flex items-center gap-1.5 text-[11px] ${
                  testNotice.ok
                    ? 'text-emerald-700 dark:text-emerald-400'
                    : 'text-amber-700 dark:text-amber-300'
                }`}
              >
                {testNotice.ok && <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />}
                <span>{testNotice.text}</span>
              </p>
            )}

            <div className="flex flex-wrap items-center justify-end gap-2 pt-0.5">
              <button
                type="button"
                disabled={testingUrl || !customUrl.trim()}
                onClick={() => void handleTestCustomUrl()}
                className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-2.5 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-200 dark:hover:bg-slate-800"
              >
                {testingUrl ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Globe className="h-3.5 w-3.5" />
                )}
                <span>Test</span>
              </button>
              <button
                type="button"
                disabled={!customUrl.trim()}
                onClick={handleAddCustomStream}
                data-testid="submit-custom-stream"
                className="inline-flex h-8 items-center gap-1 rounded-lg bg-[#0F5132] px-3 text-xs font-medium text-white hover:bg-[#0B3D25] disabled:opacity-50"
              >
                <span>Add Stream</span>
              </button>
            </div>
          </div>
        )}

        {isOfflineOnly && (
          <p className="mt-2.5 text-[11px] text-slate-500 dark:text-slate-400">
            You can choose your sources now; they will sync automatically once Offline-Only Mode is turned off.
          </p>
        )}

        {/* Collapsible Catalog Accordions */}
        <div className="mt-3 divide-y divide-slate-100 rounded-xl border border-slate-200/80 dark:divide-slate-800 dark:border-slate-800">
          {filteredGroups.length === 0 ? (
            <div className="px-4 py-6 text-center text-xs text-slate-500 dark:text-slate-400">
              {filterMode === 'active'
                ? 'No content streams are enabled yet. Switch to All to browse catalogs.'
                : 'No streams match your search.'}
            </div>
          ) : (
            filteredGroups.map(({ catalog, streams }) => {
              const enabledInCatalog = streams.filter((s) => s.enabled).length;
              const isOpen = forceExpand || expandedCatalogs.has(catalog.id);
              return (
                <div key={catalog.id}>
                  <button
                    type="button"
                    aria-expanded={isOpen}
                    data-testid={`catalog-accordion-${catalog.id}`}
                    onClick={() => toggleCatalogExpanded(catalog.id)}
                    className="flex w-full items-center justify-between gap-3 px-3.5 py-2.5 text-left hover:bg-slate-50/80 dark:hover:bg-slate-800/40 transition-colors"
                  >
                    <div className="flex min-w-0 items-center gap-2">
                      {isOpen ? (
                        <ChevronDown className="h-3.5 w-3.5 shrink-0 text-slate-400" />
                      ) : (
                        <ChevronRight className="h-3.5 w-3.5 shrink-0 text-slate-400" />
                      )}
                      <div className="min-w-0">
                        <div className="truncate text-xs font-semibold text-slate-900 dark:text-slate-100">
                          {catalog.name}
                        </div>
                        <div className="truncate text-[11px] text-slate-500 dark:text-slate-400">
                          {catalog.description}
                        </div>
                      </div>
                    </div>
                    <span
                      className={`shrink-0 rounded-full px-2 py-0.5 font-mono text-[10px] font-medium tabular-nums ${
                        enabledInCatalog > 0
                          ? 'bg-emerald-50 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300'
                          : 'bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400'
                      }`}
                    >
                      {enabledInCatalog} / {streams.length}
                    </span>
                  </button>

                  {isOpen && (
                    <div className="space-y-1 border-t border-slate-100 bg-slate-50/40 px-3 py-2 dark:border-slate-800/80 dark:bg-[#0B0F14]/40">
                      {streams.length === 0 ? (
                        <div className="flex items-center justify-between gap-2 px-1 py-1.5 text-xs text-slate-500 dark:text-slate-400">
                          <span>No custom streams added yet.</span>
                          <button
                            type="button"
                            onClick={() => setAddDrawerOpen(true)}
                            className="font-medium text-emerald-700 hover:underline dark:text-emerald-400"
                          >
                            + Add one
                          </button>
                        </div>
                      ) : (
                        streams.map((stream: ContentStreamSpec) => {
                          const syncedLabel = formatRelativeSync(stream.lastSyncedAt);
                          return (
                            <div
                              key={stream.id}
                              data-testid={`stream-row-${stream.id}`}
                              className="flex items-center justify-between gap-3 rounded-lg bg-white px-3 py-2 dark:bg-[#0F141C]"
                            >
                              <div className="min-w-0 flex-1">
                                <div className="flex flex-wrap items-center gap-1.5">
                                  <span className="truncate text-xs font-medium text-slate-900 dark:text-slate-100">
                                    {stream.name}
                                  </span>
                                  <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-500 dark:bg-slate-800 dark:text-slate-400">
                                    {PROVIDER_LABELS[stream.providerKind] ?? 'Source'}
                                  </span>
                                  {syncedLabel && (
                                    <span className="text-[10px] text-emerald-700 dark:text-emerald-400">
                                      {syncedLabel}
                                    </span>
                                  )}
                                </div>
                                <div className="mt-0.5 truncate text-[11px] text-slate-500 dark:text-slate-400">
                                  {stream.description}
                                </div>
                              </div>

                              <div className="flex shrink-0 items-center gap-2">
                                {!stream.builtIn && (
                                  <button
                                    type="button"
                                    aria-label={`Remove ${stream.name}`}
                                    onClick={() => contentStreams.removeCustomStream(stream.id)}
                                    className="inline-flex h-7 w-7 items-center justify-center rounded-md text-slate-400 hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-950/50 dark:hover:text-red-300"
                                  >
                                    <Trash2 className="h-3.5 w-3.5" />
                                  </button>
                                )}
                                <button
                                  type="button"
                                  role="switch"
                                  aria-checked={stream.enabled}
                                  aria-label={`Toggle ${stream.name}`}
                                  data-testid={`stream-toggle-${stream.id}`}
                                  onClick={() => contentStreams.toggleStream(stream.id)}
                                  className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-700 ${
                                    stream.enabled
                                      ? 'bg-[#0F5132] dark:bg-emerald-500'
                                      : 'bg-slate-200 dark:bg-slate-700'
                                  }`}
                                >
                                  <span
                                    className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow-2xs transition-transform ${
                                      stream.enabled ? 'translate-x-4.5' : 'translate-x-0.5'
                                    }`}
                                  />
                                </button>
                              </div>
                            </div>
                          );
                        })
                      )}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      </div>
    </section>
  );
};
