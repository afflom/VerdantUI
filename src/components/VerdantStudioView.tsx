/**
 * Verdant Studio & Published Prism App Workspace.
 *
 * Clean, zero-bloat Studio core inspired by Foundry's studio ergonomics:
 * - Collapsed ~40px Prism Focus Bar (editable title, Draft/Published/Modified dot, short SHA-256
 *   digest, Undo, Studio|App toggle, Publish, and expandable Prism Model drawer)
 * - Studio Mode: Split Chat + Real-Time Sandboxed Preview (targeted {html, css, js} updates,
 *   inherited light/dark theme, runtime error banner with one-click "Fix with VerdantUI",
 *   and collapsible OpenTelemetry Stream Trace strip)
 * - App Mode: Full-stage live Published Prism app with instant Edit-in-Studio switch
 */

import { useCallback, useEffect, useRef, useState, type FC, type FormEvent } from 'react';
import type { ThreadMessage } from '@assistant-ui/react';
import {
  Activity,
  AlertTriangle,
  ArrowUp,
  Check,
  ChevronDown,
  ChevronRight,
  Code2,
  Download,
  ExternalLink,
  Globe,
  Loader2,
  Monitor,
  Play,
  RotateCcw,
  Send,
  Smartphone,
  Sparkles,
  Square,
  Undo2,
  Upload,
  Wrench,
} from 'lucide-react';
import {
  formatMessagesForLlama,
  stripReasoningTags,
} from '../engine/assistantAdapter';
import type { ObserverStreamBus, OtelEvent } from '../engine/observerStreamBus';
import type { PrismBridgeRuntime } from '../engine/prismHostPool';
import {
  PRISM_STARTERS,
  buildStudioSystemPrompt,
  createStarterSnapshot,
  extractPrismPatchFromModelOutput,
  shortPrismDigest,
  type JsonSchema2020Object,
  type PrismStarterId,
  type PrismStore,
  type VerdantPrismProject,
} from '../engine/prismStudio';
import { isInternalBootstrapModel } from '../engine/modelCatalog';
import type { EngineState, LlamaVTensorWorkerClient } from '../engine/workerClient';
import { PrismSandboxFrame, type PrismRuntimeError } from './PrismSandboxFrame';

interface VerdantStudioViewProps {
  project: VerdantPrismProject;
  mode: 'studio' | 'app';
  onChangeMode: (mode: 'studio' | 'app') => void;
  store: PrismStore;
  bus: ObserverStreamBus;
  bridgeRuntime: PrismBridgeRuntime;
  workerClient: LlamaVTensorWorkerClient;
  engineState: EngineState;
  darkMode: boolean;
  onEnableModelStream: () => void;
  onSelectProject: (prismId: string, mode: 'studio' | 'app') => void;
}

type FocusDrawerTab = 'streams' | 'source' | 'package';
type SourceFileTab = 'html' | 'css' | 'js';

export const VerdantStudioView: FC<VerdantStudioViewProps> = ({
  project,
  mode,
  onChangeMode,
  store,
  bus,
  bridgeRuntime,
  workerClient,
  engineState,
  darkMode,
  onEnableModelStream,
  onSelectProject,
}) => {
  const [focusExpanded, setFocusExpanded] = useState(false);
  const [drawerTab, setDrawerTab] = useState<FocusDrawerTab>('streams');
  const [sourceTab, setSourceTab] = useState<SourceFileTab>('html');
  const [viewportMode, setViewportMode] = useState<'desktop' | 'mobile'>('desktop');
  const [reloadNonce, setReloadNonce] = useState(0);
  const [traceOpen, setTraceOpen] = useState(false);
  const [runtimeError, setRuntimeError] = useState<PrismRuntimeError | null>(null);
  const [recentEvents, setRecentEvents] = useState<readonly OtelEvent[]>(() =>
    bus.getRecentEvents({ endpointId: project.id }).slice(-12),
  );
  const [chatInput, setChatInput] = useState('');
  const [generating, setGenerating] = useState(false);
  const [streamingReply, setStreamingReply] = useState('');
  const [importError, setImportError] = useState<string | null>(null);
  const [justPublished, setJustPublished] = useState(false);
  const [editingTitle, setEditingTitle] = useState(false);
  const [titleDraft, setTitleDraft] = useState(project.draft.name);
  const [testInputText, setTestInputText] = useState('{"action": "status"}');
  const [testResultText, setTestResultText] = useState<string | null>(null);
  const [testRunning, setTestRunning] = useState(false);
  const [schemaDraft, setSchemaDraft] = useState(() =>
    JSON.stringify(project.draft.streams.inputs, null, 2),
  );
  const [schemaError, setSchemaError] = useState<string | null>(null);
  const [summaryDraft, setSummaryDraft] = useState(project.draft.summary);
  const [sourceDraft, setSourceDraft] = useState(project.draft.app[sourceTab]);
  const sourceSnapshotPushedRef = useRef(false);
  const summaryDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sourceDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const publishBadgeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const activeSnapshot =
    mode === 'app' && project.published ? project.published : project.draft;
  const isPublished = project.published !== null;
  const hasUnpublishedChanges =
    project.published !== null && project.published.digest !== project.draft.digest;
  const hasRealModel =
    engineState.model !== null && !isInternalBootstrapModel(engineState.model.id);

  useEffect(() => {
    setTitleDraft(project.draft.name);
    setSummaryDraft(project.draft.summary);
    setSchemaDraft(JSON.stringify(project.draft.streams.inputs, null, 2));
    setSchemaError(null);
    setTestResultText(null);
    setRuntimeError(null);
    setImportError(null);
    sourceSnapshotPushedRef.current = false;
    setRecentEvents(bus.getRecentEvents({ endpointId: project.id }).slice(-12));
  }, [project.id, project.draft.name, project.draft.summary, project.draft.streams.inputs, bus]);

  useEffect(() => {
    setSourceDraft(project.draft.app[sourceTab]);
  }, [project.id, project.draft.digest, sourceTab, project.draft.app]);

  useEffect(() => {
    const unsub = bus.subscribe(
      (ev) => {
        setRecentEvents((prev) => [...prev.slice(-19), ev]);
      },
      { endpointId: project.id },
    );
    return unsub;
  }, [bus, project.id]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [project.messages.length, streamingReply]);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (summaryDebounceRef.current) clearTimeout(summaryDebounceRef.current);
      if (sourceDebounceRef.current) clearTimeout(sourceDebounceRef.current);
      if (publishBadgeTimerRef.current) clearTimeout(publishBadgeTimerRef.current);
    };
  }, []);

  const flushSummaryDraft = useCallback(
    async (val: string) => {
      if (summaryDebounceRef.current) {
        clearTimeout(summaryDebounceRef.current);
        summaryDebounceRef.current = null;
      }
      if (val === project.draft.summary) return;
      await store.updateDraft(
        project.id,
        (prev) => ({ ...prev, summary: val }),
        { pushRevision: false },
      );
    },
    [project.id, project.draft.summary, store],
  );

  const flushSourceDraft = useCallback(
    async (tab: SourceFileTab, val: string) => {
      if (sourceDebounceRef.current) {
        clearTimeout(sourceDebounceRef.current);
        sourceDebounceRef.current = null;
      }
      if (val === project.draft.app[tab]) return;
      const shouldPushRevision = !sourceSnapshotPushedRef.current;
      sourceSnapshotPushedRef.current = true;
      await store.updateDraft(
        project.id,
        (prev) => ({
          ...prev,
          app: { ...prev.app, [tab]: val },
        }),
        { pushRevision: shouldPushRevision },
      );
    },
    [project.id, project.draft.app, store],
  );

  const handleCommitTitle = useCallback(async () => {
    setEditingTitle(false);
    const trimmed = titleDraft.trim();
    if (!trimmed || trimmed === project.draft.name) {
      setTitleDraft(project.draft.name);
      return;
    }
    await store.updateDraft(project.id, (prev) => ({ ...prev, name: trimmed }));
  }, [titleDraft, project.id, project.draft.name, store]);

  const handlePublish = useCallback(async () => {
    await flushSummaryDraft(summaryDraft);
    await flushSourceDraft(sourceTab, sourceDraft);
    await store.publish(project.id);
    const updated = store.get(project.id);
    bus.emit({
      name: 'prism.published',
      endpointKind: 'studio',
      endpointId: project.id,
      attributes: {
        'verdant.prism.name': updated?.draft.name ?? project.draft.name,
        'verdant.prism.digest': updated?.draft.digest ?? project.draft.digest,
      },
    });
    setJustPublished(true);
    if (publishBadgeTimerRef.current) clearTimeout(publishBadgeTimerRef.current);
    publishBadgeTimerRef.current = setTimeout(() => setJustPublished(false), 1800);
  }, [
    flushSummaryDraft,
    summaryDraft,
    flushSourceDraft,
    sourceTab,
    sourceDraft,
    store,
    bus,
    project.id,
    project.draft.name,
    project.draft.digest,
  ]);

  const handleUndo = useCallback(async () => {
    setRuntimeError(null);
    await store.undoDraft(project.id);
  }, [store, project.id]);

  const handleApplyStarter = useCallback(
    async (starterId: PrismStarterId) => {
      setRuntimeError(null);
      const snap = createStarterSnapshot(starterId);
      await store.updateDraft(project.id, () => ({
        name: snap.name,
        summary: snap.summary,
        streams: snap.streams,
        grants: snap.grants,
        app: snap.app,
      }));
      await store.appendMessage(project.id, {
        role: 'assistant',
        content: `Loaded the **${snap.name}** starter into your live preview. Describe any changes you want, or edit HTML/CSS/JS directly in the Prism drawer.`,
        updatedSections: ['html', 'css', 'js', 'meta'],
      });
    },
    [store, project.id],
  );

  const handleExportHolo = useCallback(() => {
    const content = store.exportHolo(project.id, mode === 'app');
    const blob = new Blob([content], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const slug = activeSnapshot.name.toLowerCase().replace(/[^a-z0-9]+/g, '-') || 'prism';
    a.href = url;
    a.download = `${slug}.holo`;
    a.click();
    URL.revokeObjectURL(url);
  }, [store, project.id, mode, activeSnapshot.name]);

  const handleImportHoloFile = useCallback(
    async (file: File) => {
      setImportError(null);
      try {
        const bytes = new Uint8Array(await file.arrayBuffer());
        const imported = await store.importHolo(bytes, false);
        onSelectProject(imported.id, 'studio');
      } catch (err) {
        setImportError(err instanceof Error ? err.message : String(err));
      }
    },
    [store, onSelectProject],
  );

  const handleExportOtlp = useCallback(() => {
    const payload = JSON.stringify(bus.toOtlpJson(project.id), null, 2);
    const blob = new Blob([payload], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `verdant-otel-${project.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [bus, project.id]);

  const runStudioPrompt = useCallback(
    async (userPrompt: string) => {
      const trimmed = userPrompt.trim();
      if (!trimmed || generating) return;
      setChatInput('');
      setRuntimeError(null);

      await store.appendMessage(project.id, { role: 'user', content: trimmed });

      const span = bus.startSpan('studio.author', {
        endpointKind: 'studio',
        endpointId: project.id,
        kind: 'SERVER',
        attributes: { 'verdant.prompt.chars': trimmed.length },
      });

      if (!hasRealModel) {
        const msg = engineState.modelStreamEnabled
          ? `Model Stream is initializing (${engineState.progressPercent}%). Wait for it to finish, or edit the Prism directly in the expanded Prism drawer.`
          : 'Model Stream is currently **Off**. Toggle **Model Stream** on in the top bar for freeform AI app generation—or pick a starter below / expand the **Prism** bar to edit HTML, CSS, and JS directly.';
        await store.appendMessage(project.id, { role: 'assistant', content: msg });
        span.end('ok');
        return;
      }

      setGenerating(true);
      setStreamingReply('');
      const ctrl = new AbortController();
      abortRef.current = ctrl;

      try {
        const currentProj = store.get(project.id) ?? project;
        const sysPrompt = buildStudioSystemPrompt(currentProj.draft);
        const historyMessages: ThreadMessage[] = currentProj.messages.slice(-6).map((m) => ({
          id: m.id,
          role: m.role,
          createdAt: new Date(m.createdAt),
          content: [{ type: 'text', text: m.content }],
          attachments: [],
          metadata: { custom: {} },
        })) as ThreadMessage[];
        const fullPrompt = formatMessagesForLlama(
          historyMessages,
          engineState.model!,
          engineState.promptMode,
          sysPrompt,
          { thinkingMode: 'auto' },
        );

        let accumulated = '';
        let lastUiFlushAt = 0;
        for await (const ev of workerClient.streamGenerate(
          fullPrompt,
          ctrl.signal,
          { ...engineState.sampling, temperature: 0.35, maxTokens: 2048 },
          `studio:${project.id}`,
        )) {
          accumulated = ev.accumulatedText;
          const now = Date.now();
          if (now - lastUiFlushAt >= 100) {
            lastUiFlushAt = now;
            setStreamingReply(accumulated);
          }
        }
        setStreamingReply(accumulated);

        const cleanedOutput = stripReasoningTags(accumulated);
        const latestProj = store.get(project.id) ?? project;
        const extracted = extractPrismPatchFromModelOutput(cleanedOutput, latestProj.draft);
        if (extracted) {
          await store.updateDraft(project.id, () => ({
            name: extracted.snapshot.name,
            summary: extracted.snapshot.summary,
            streams: extracted.snapshot.streams,
            grants: extracted.snapshot.grants,
            app: extracted.snapshot.app,
          }));
          const summaryNote =
            cleanedOutput
              .replace(/```[\s\S]*?```/g, '')
              .trim()
              .slice(0, 360) ||
            `Updated ${extracted.updatedSections.join(', ').toUpperCase()} in the live preview.`;
          await store.appendMessage(project.id, {
            role: 'assistant',
            content: summaryNote,
            updatedSections: extracted.updatedSections,
          });
          span.setAttributes({
            'verdant.studio.sections': extracted.updatedSections.join(','),
          });
        } else {
          await store.appendMessage(project.id, {
            role: 'assistant',
            content: cleanedOutput.trim() || 'No code changes emitted.',
          });
        }
        span.end('ok');
      } catch (err) {
        if (!ctrl.signal.aborted) {
          const msg = err instanceof Error ? err.message : String(err);
          await store.appendMessage(project.id, {
            role: 'assistant',
            content: `Unable to update Prism: ${msg}`,
          });
          span.end('error', msg);
        } else {
          span.end('ok');
        }
      } finally {
        setGenerating(false);
        setStreamingReply('');
        abortRef.current = null;
      }
    },
    [generating, store, project, bus, hasRealModel, engineState, workerClient],
  );

  const handleSubmitChat = (e: FormEvent) => {
    e.preventDefault();
    void runStudioPrompt(chatInput);
  };

  const handleFixRuntimeError = () => {
    if (!runtimeError) return;
    const loc = runtimeError.line ? ` (line ${runtimeError.line})` : '';
    void runStudioPrompt(
      `Fix this runtime error in the app preview${loc}: ${runtimeError.message}`,
    );
  };

  const statusBadge = !isPublished
    ? { label: 'Draft', dot: 'bg-slate-400 dark:bg-slate-500' }
    : hasUnpublishedChanges
      ? { label: 'Modified', dot: 'bg-amber-500' }
      : { label: 'Published', dot: 'bg-emerald-600 dark:bg-emerald-400' };

  return (
    <div className="flex h-full flex-col overflow-hidden bg-[#F8F9FA] dark:bg-[#0B0F14]">
      {/* Collapsed ~40px Prism Focus Bar */}
      <div className="flex min-h-[44px] shrink-0 flex-wrap items-center justify-between gap-2 border-b border-slate-200 bg-white px-3.5 py-1.5 dark:border-slate-800 dark:bg-[#0F141C]">
        <div className="flex min-w-0 items-center gap-2">
          <button
            type="button"
            aria-expanded={focusExpanded}
            data-testid="prism-focus-toggle"
            onClick={() => setFocusExpanded((prev) => !prev)}
            className="inline-flex h-7 items-center gap-1 rounded-md px-1.5 text-xs font-medium text-slate-600 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-300 dark:hover:bg-slate-800"
            title="Inspect or edit Prism model, grants, source, and .holo package"
          >
            {focusExpanded ? (
              <ChevronDown className="h-3.5 w-3.5" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5" />
            )}
            <span className="hidden sm:inline">Prism</span>
          </button>

          {editingTitle && mode === 'studio' ? (
            <input
              type="text"
              autoFocus
              value={titleDraft}
              onChange={(e) => setTitleDraft(e.target.value)}
              onBlur={() => void handleCommitTitle()}
              onKeyDown={(e) => {
                if (e.key === 'Enter') void handleCommitTitle();
                if (e.key === 'Escape') {
                  setTitleDraft(project.draft.name);
                  setEditingTitle(false);
                }
              }}
              className="h-7 w-44 rounded-md border border-emerald-600 bg-white px-2 text-xs font-semibold text-slate-900 focus:outline-none dark:bg-[#141B26] dark:text-white"
            />
          ) : (
            <button
              type="button"
              onClick={() => {
                if (mode === 'studio') setEditingTitle(true);
              }}
              className="truncate text-xs font-semibold text-slate-900 hover:text-emerald-800 dark:text-slate-100 dark:hover:text-emerald-400"
              title={mode === 'studio' ? 'Click to rename Prism' : activeSnapshot.name}
            >
              {activeSnapshot.name}
            </button>
          )}

          <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 px-2 py-0.5 text-[11px] font-medium text-slate-600 dark:border-slate-700 dark:text-slate-300">
            <span className={`h-1.5 w-1.5 rounded-full ${statusBadge.dot}`} />
            {statusBadge.label}
          </span>

          <span
            className="hidden font-mono text-[11px] text-slate-400 md:inline dark:text-slate-500"
            title={activeSnapshot.digest}
          >
            #{shortPrismDigest(activeSnapshot.digest)}
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          {mode === 'studio' && project.revisions.length > 0 && (
            <button
              type="button"
              onClick={() => void handleUndo()}
              title={`Undo last change (${project.revisions.length} saved)`}
              className="inline-flex h-7 items-center gap-1 rounded-md border border-slate-200 bg-white px-2 text-xs font-medium text-slate-600 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-300 dark:hover:bg-slate-800"
            >
              <Undo2 className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Undo</span>
            </button>
          )}

          {isPublished && (
            <div
              role="group"
              aria-label="Prism view mode"
              className="inline-flex h-7 rounded-md border border-slate-200 bg-slate-100/80 p-0.5 text-[11px] font-medium dark:border-slate-700 dark:bg-[#141B26]"
            >
              <button
                type="button"
                onClick={() => onChangeMode('studio')}
                className={`rounded px-2 transition-colors ${
                  mode === 'studio'
                    ? 'bg-white text-slate-900 shadow-xs dark:bg-slate-800 dark:text-white'
                    : 'text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
                }`}
              >
                Studio
              </button>
              <button
                type="button"
                onClick={() => onChangeMode('app')}
                className={`rounded px-2 transition-colors ${
                  mode === 'app'
                    ? 'bg-white text-slate-900 shadow-xs dark:bg-slate-800 dark:text-white'
                    : 'text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
                }`}
              >
                App
              </button>
            </div>
          )}

          {mode === 'studio' && (
            <button
              type="button"
              data-testid="prism-publish-btn"
              onClick={() => void handlePublish()}
              className="inline-flex h-7 items-center gap-1.5 rounded-md bg-[#0F5132] px-2.5 text-xs font-medium text-white hover:bg-[#0B3D25] transition-colors"
            >
              {justPublished ? (
                <>
                  <Check className="h-3.5 w-3.5" />
                  <span>Published</span>
                </>
              ) : (
                <>
                  <Play className="h-3 w-3" />
                  <span>{hasUnpublishedChanges ? 'Republish' : isPublished ? 'Publish' : 'Publish'}</span>
                </>
              )}
            </button>
          )}
        </div>
      </div>

      {/* Expandable Prism Focus Drawer (collapsed by default) */}
      {focusExpanded && (
        <div
          data-testid="prism-focus-drawer"
          className="shrink-0 border-b border-slate-200 bg-slate-50/80 px-4 py-3 text-xs dark:border-slate-800 dark:bg-[#111722]"
        >
          <div className="mb-2.5 flex items-center justify-between gap-2">
            <div className="flex items-center gap-1">
              {(['streams', 'source', 'package'] as const).map((t) => (
                <button
                  key={t}
                  type="button"
                  onClick={() => setDrawerTab(t)}
                  className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                    drawerTab === t
                      ? 'bg-white text-slate-900 shadow-xs ring-1 ring-slate-200 dark:bg-slate-800 dark:text-white dark:ring-slate-700'
                      : 'text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
                  }`}
                >
                  {t === 'streams' ? 'Streams & Grants' : t === 'source' ? 'Source' : 'Package (.holo)'}
                </button>
              ))}
            </div>
            {isPublished && mode === 'studio' && (
              <button
                type="button"
                onClick={() => void store.unpublish(project.id)}
                className="text-[11px] font-medium text-slate-500 hover:text-red-600 dark:text-slate-400 dark:hover:text-red-400"
              >
                Unpublish
              </button>
            )}
          </div>

          {drawerTab === 'streams' && (
            <div className="grid gap-3 lg:grid-cols-2">
              <div className="space-y-2.5">
                <label className="block">
                  <span className="text-[11px] font-medium text-slate-600 dark:text-slate-400">
                    Summary (used by VerdantUI to route requests)
                  </span>
                  <input
                    type="text"
                    value={summaryDraft}
                    onBlur={() => void flushSummaryDraft(summaryDraft)}
                    onChange={(e) => {
                      const val = e.target.value;
                      setSummaryDraft(val);
                      if (summaryDebounceRef.current) clearTimeout(summaryDebounceRef.current);
                      summaryDebounceRef.current = setTimeout(() => {
                        void flushSummaryDraft(val);
                      }, 180);
                    }}
                    className="mt-1 w-full rounded-md border border-slate-200 bg-white px-2.5 py-1.5 text-xs text-slate-900 dark:border-slate-700 dark:bg-[#0F141C] dark:text-slate-100"
                  />
                </label>

                <div className="space-y-1.5">
                  <div className="text-[11px] font-medium text-slate-600 dark:text-slate-400">
                    Capability Grants (FK-07 Denial-Wins)
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {(
                      [
                        { key: 'observer', label: 'Observer' },
                        { key: 'storage', label: 'Storage' },
                        { key: 'browser', label: 'Browser' },
                        { key: 'network', label: 'Network' },
                      ] as const
                    ).map(({ key, label }) => {
                      const enabled = project.draft.grants[key];
                      return (
                        <button
                          key={key}
                          type="button"
                          role="switch"
                          aria-checked={enabled}
                          onClick={() =>
                            void store.updateDraft(project.id, (prev) => ({
                              ...prev,
                              grants: { ...prev.grants, [key]: !prev.grants[key] },
                            }))
                          }
                          className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] font-medium transition-colors ${
                            enabled
                              ? 'border-emerald-700/40 bg-emerald-50 text-emerald-900 dark:border-emerald-500/40 dark:bg-emerald-950/50 dark:text-emerald-200'
                              : 'border-slate-200 bg-white text-slate-600 dark:border-slate-700 dark:bg-[#0F141C] dark:text-slate-400'
                          }`}
                        >
                          {key === 'network' && <Globe className="h-3 w-3" />}
                          <span>{label}</span>
                          <span className="font-mono text-[10px] opacity-80">
                            {enabled ? 'On' : 'Off'}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] font-medium text-slate-600 dark:text-slate-400">
                      Test Stream Dispatch (calls window.verdant.onInput)
                    </span>
                    <span className="text-[10px] text-slate-400">
                      Thread syntax: <code>@{project.draft.name}</code>
                    </span>
                  </div>
                  <div className="flex gap-1.5">
                    <input
                      type="text"
                      value={testInputText}
                      onChange={(e) => setTestInputText(e.target.value)}
                      className="flex-1 rounded-md border border-slate-200 bg-white px-2 py-1 font-mono text-[11px] text-slate-900 dark:border-slate-700 dark:bg-[#0F141C] dark:text-slate-100"
                    />
                    <button
                      type="button"
                      disabled={testRunning}
                      onClick={async () => {
                        setTestRunning(true);
                        setTestResultText(null);
                        try {
                          let parsed: unknown = testInputText;
                          try {
                            parsed = JSON.parse(testInputText);
                          } catch {
                            parsed = { message: testInputText };
                          }
                          const t0 = performance.now();
                          const { result } = await bus.dispatchToEndpoint(project.id, parsed);
                          const ms = Math.max(1, Math.round(performance.now() - t0));
                          setTestResultText(
                            `${ms}ms · ${typeof result === 'string' ? result : JSON.stringify(result, null, 2)}`,
                          );
                        } catch (err) {
                          setTestResultText(
                            `Error: ${err instanceof Error ? err.message : String(err)}`,
                          );
                        } finally {
                          setTestRunning(false);
                        }
                      }}
                      className="inline-flex items-center gap-1 rounded-md bg-slate-900 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-slate-700 disabled:opacity-50 dark:bg-emerald-700 dark:hover:bg-emerald-600"
                    >
                      <Send className="h-3 w-3" />
                      <span>{testRunning ? 'Sending…' : 'Dispatch'}</span>
                    </button>
                  </div>
                  {testResultText && (
                    <pre className="max-h-24 overflow-auto rounded-md border border-slate-200 bg-white p-2 font-mono text-[11px] text-slate-700 dark:border-slate-700 dark:bg-[#0B0F14] dark:text-slate-300">
                      {testResultText}
                    </pre>
                  )}
                </div>
              </div>

              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-medium text-slate-600 dark:text-slate-400">
                    Input Contract (JSON Schema 2020-12)
                  </span>
                  <button
                    type="button"
                    onClick={() => {
                      try {
                        const parsed = JSON.parse(schemaDraft) as JsonSchema2020Object;
                        if (!parsed || parsed.type !== 'object' || typeof parsed.properties !== 'object') {
                          throw new Error('Schema must be an object with "type": "object" and "properties".');
                        }
                        setSchemaError(null);
                        void store.updateDraft(project.id, (prev) => ({
                          ...prev,
                          streams: { ...prev.streams, inputs: parsed },
                        }));
                      } catch (err) {
                        setSchemaError(err instanceof Error ? err.message : String(err));
                      }
                    }}
                    className="rounded border border-slate-200 bg-white px-2 py-0.5 text-[10px] font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#0F141C] dark:text-slate-200"
                  >
                    Apply Schema
                  </button>
                </div>
                <textarea
                  rows={6}
                  spellCheck={false}
                  value={schemaDraft}
                  onChange={(e) => setSchemaDraft(e.target.value)}
                  className="w-full rounded-md border border-slate-200 bg-white p-2 font-mono text-[11px] text-slate-900 focus:border-emerald-700 focus:outline-none dark:border-slate-700 dark:bg-[#0B0F14] dark:text-slate-100"
                />
                {schemaError && (
                  <div className="text-[11px] text-red-600 dark:text-red-400">{schemaError}</div>
                )}
              </div>
            </div>
          )}

          {drawerTab === 'source' && (
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-1">
                  {(['html', 'css', 'js'] as const).map((f) => (
                    <button
                      key={f}
                      type="button"
                      onClick={() => {
                        void flushSourceDraft(sourceTab, sourceDraft);
                        setSourceTab(f);
                        setSourceDraft(project.draft.app[f]);
                        sourceSnapshotPushedRef.current = false;
                      }}
                      className={`inline-flex items-center gap-1 rounded px-2 py-0.5 font-mono text-[11px] uppercase ${
                        sourceTab === f
                          ? 'bg-slate-900 text-white dark:bg-emerald-700'
                          : 'bg-white text-slate-600 ring-1 ring-slate-200 dark:bg-[#0F141C] dark:text-slate-300 dark:ring-slate-700'
                      }`}
                    >
                      <Code2 className="h-3 w-3" />
                      {f}
                    </button>
                  ))}
                </div>
                <span className="text-[10px] text-slate-400">
                  {sourceTab === 'css' ? 'Hot-swaps live without reloading app state' : 'Auto-updates live preview'}
                </span>
              </div>
              <textarea
                value={sourceDraft}
                onBlur={() => {
                  void flushSourceDraft(sourceTab, sourceDraft);
                  sourceSnapshotPushedRef.current = false;
                }}
                onChange={(e) => {
                  const nextVal = e.target.value;
                  setSourceDraft(nextVal);
                  setRuntimeError(null);
                  if (sourceDebounceRef.current) clearTimeout(sourceDebounceRef.current);
                  sourceDebounceRef.current = setTimeout(() => {
                    void flushSourceDraft(sourceTab, nextVal);
                  }, 180);
                }}
                rows={7}
                spellCheck={false}
                className="w-full rounded-lg border border-slate-200 bg-white p-2.5 font-mono text-[12px] leading-relaxed text-slate-900 focus:border-emerald-700 focus:outline-none dark:border-slate-700 dark:bg-[#0B0F14] dark:text-slate-100"
              />
            </div>
          )}

          {drawerTab === 'package' && (
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="space-y-0.5">
                <div className="font-mono text-[11px] text-slate-700 dark:text-slate-300">
                  {activeSnapshot.digest}
                </div>
                <div className="text-[11px] text-slate-500 dark:text-slate-400">
                  Self-contained <code className="font-mono">.holo</code> archive (JSON Schema 2020-12 streams + SHA-256 integrity).
                </div>
                {importError && (
                  <div className="text-[11px] text-red-600 dark:text-red-400">{importError}</div>
                )}
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleExportHolo}
                  className="inline-flex h-8 items-center gap-1.5 rounded-md border border-slate-200 bg-white px-2.5 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#0F141C] dark:text-slate-200 dark:hover:bg-slate-800"
                >
                  <Download className="h-3.5 w-3.5" />
                  <span>Export .holo</span>
                </button>
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="inline-flex h-8 items-center gap-1.5 rounded-md border border-slate-200 bg-white px-2.5 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#0F141C] dark:text-slate-200 dark:hover:bg-slate-800"
                >
                  <Upload className="h-3.5 w-3.5" />
                  <span>Import .holo</span>
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".holo,.json"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) void handleImportHoloFile(file);
                    e.target.value = '';
                  }}
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Main Content: App Mode vs Studio Split Mode */}
      {mode === 'app' && project.published ? (
        <div className="relative flex min-h-0 flex-1 flex-col">
          <div className="min-h-0 flex-1">
            <PrismSandboxFrame
              prismId={project.id}
              snapshot={project.published}
              darkMode={darkMode}
              bus={bus}
              bridgeRuntime={bridgeRuntime}
              reloadNonce={reloadNonce}
              registerAsEndpoint={true}
              onRuntimeError={setRuntimeError}
            />
          </div>
          {runtimeError && (
            <div className="flex items-center justify-between gap-3 border-t border-red-200 bg-red-50 px-4 py-2 text-xs text-red-800 dark:border-red-900/60 dark:bg-red-950/60 dark:text-red-200">
              <span className="truncate font-mono">{runtimeError.message}</span>
              <button
                type="button"
                onClick={() => onChangeMode('studio')}
                className="shrink-0 rounded-md bg-red-700 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-red-800"
              >
                Open in Studio
              </button>
            </div>
          )}
        </div>
      ) : (
        <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[minmax(320px,420px)_1fr]">
          {/* Left Pane: Studio Chat */}
          <div className="flex min-h-0 flex-col border-b border-slate-200 bg-white lg:border-r lg:border-b-0 dark:border-slate-800 dark:bg-[#0F141C]">
            <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3.5">
              {project.messages.length === 0 && !streamingReply && (
                <div className="space-y-3 pt-2">
                  <div>
                    <h2 className="font-display text-sm font-semibold text-slate-900 dark:text-white">
                      Verdant Studio
                    </h2>
                    <p className="mt-1 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
                      Describe an interactive app, game, or dashboard to build and preview live. Publish when ready so both you and VerdantUI can run it.
                    </p>
                  </div>

                  {!hasRealModel && !engineState.modelStreamEnabled && (
                    <div className="rounded-lg border border-slate-200 bg-slate-50 p-2.5 text-xs text-slate-600 dark:border-slate-800 dark:bg-[#141B26] dark:text-slate-300">
                      <span>Model Stream is off. Pick a starter below or </span>
                      <button
                        type="button"
                        onClick={onEnableModelStream}
                        className="font-semibold text-emerald-800 underline underline-offset-2 hover:text-emerald-700 dark:text-emerald-400"
                      >
                        enable Model Stream
                      </button>
                      <span> for freeform AI authoring.</span>
                    </div>
                  )}

                  <div className="grid gap-1.5">
                    {PRISM_STARTERS.map((starter) => (
                      <button
                        key={starter.id}
                        type="button"
                        onClick={() => void handleApplyStarter(starter.id)}
                        className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-left hover:border-emerald-700/40 hover:bg-emerald-50/40 dark:border-slate-800 dark:bg-[#141B26] dark:hover:border-emerald-500/40 dark:hover:bg-emerald-950/30 transition-colors"
                      >
                        <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-900 dark:text-slate-100">
                          <Sparkles className="h-3.5 w-3.5 text-emerald-700 dark:text-emerald-400" />
                          <span>{starter.label}</span>
                        </div>
                        <p className="mt-0.5 text-[11px] text-slate-500 dark:text-slate-400">
                          {starter.description}
                        </p>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {project.messages.map((m) => (
                <div
                  key={m.id}
                  className={
                    m.role === 'user'
                      ? 'ml-6 rounded-xl bg-slate-900 px-3 py-2 text-xs leading-relaxed text-white dark:bg-emerald-900/60'
                      : 'rounded-xl border border-slate-200 bg-slate-50/80 px-3 py-2 text-xs leading-relaxed text-slate-700 dark:border-slate-800 dark:bg-[#141B26] dark:text-slate-200'
                  }
                >
                  <div className="whitespace-pre-wrap">{m.content}</div>
                  {m.updatedSections && m.updatedSections.length > 0 && (
                    <div className="mt-1.5 flex flex-wrap gap-1">
                      {m.updatedSections.map((sec) => (
                        <span
                          key={sec}
                          className="rounded bg-emerald-100 px-1.5 py-0.5 font-mono text-[10px] font-medium uppercase text-emerald-900 dark:bg-emerald-950 dark:text-emerald-300"
                        >
                          {sec}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}

              {streamingReply && (
                <div className="rounded-xl border border-emerald-700/30 bg-emerald-50/40 px-3 py-2 text-xs leading-relaxed text-slate-700 dark:border-emerald-500/30 dark:bg-emerald-950/30 dark:text-slate-200">
                  <div className="mb-1 flex items-center gap-1.5 text-[11px] font-medium text-emerald-800 dark:text-emerald-400">
                    <Loader2 className="h-3 w-3 animate-spin" />
                    <span>
                      {streamingReply.includes('<think>') && !streamingReply.includes('</think>')
                        ? 'Planning app changes…'
                        : 'Writing Prism update…'}
                    </span>
                  </div>
                  {(() => {
                    const visible = stripReasoningTags(streamingReply).trim();
                    if (!visible) return null;
                    return (
                      <div className="max-h-36 overflow-y-auto font-mono text-[11px] whitespace-pre-wrap opacity-80">
                        {visible.slice(-500)}
                      </div>
                    );
                  })()}
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <form
              onSubmit={handleSubmitChat}
              className="shrink-0 border-t border-slate-200 p-3 dark:border-slate-800"
            >
              <div className="flex items-end gap-2 rounded-xl border border-slate-200 bg-[#F8F9FA] p-1.5 focus-within:border-emerald-700 dark:border-slate-700 dark:bg-[#141B26]">
                <textarea
                  rows={2}
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      void runStudioPrompt(chatInput);
                    }
                  }}
                  placeholder="Describe a change or feature for this Prism…"
                  className="max-h-28 flex-1 resize-none bg-transparent px-2 py-1 text-xs text-slate-900 placeholder:text-slate-400 focus:outline-none dark:text-slate-100"
                />
                {generating ? (
                  <button
                    type="button"
                    onClick={() => abortRef.current?.abort()}
                    aria-label="Stop generation"
                    className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-slate-900 text-white hover:bg-slate-700 dark:bg-slate-200 dark:text-slate-900"
                  >
                    <Square className="h-3.5 w-3.5" />
                  </button>
                ) : (
                  <button
                    type="submit"
                    disabled={!chatInput.trim()}
                    aria-label="Send Studio prompt"
                    className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#0F5132] text-white hover:bg-[#0B3D25] disabled:opacity-40"
                  >
                    <ArrowUp className="h-4 w-4" />
                  </button>
                )}
              </div>
            </form>
          </div>

          {/* Right Pane: Real-Time Preview + Collapsible OTel Stream Strip */}
          <div className="flex min-h-0 flex-col bg-[#F8F9FA] dark:bg-[#0B0F14]">
            <div className="flex h-9 shrink-0 items-center justify-between border-b border-slate-200 bg-white/80 px-3 text-xs dark:border-slate-800 dark:bg-[#0F141C]/80">
              <span className="font-medium text-slate-600 dark:text-slate-400">Live Preview</span>
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => setViewportMode('desktop')}
                  aria-label="Desktop preview"
                  className={`inline-flex h-6 w-6 items-center justify-center rounded ${
                    viewportMode === 'desktop'
                      ? 'bg-slate-200 text-slate-900 dark:bg-slate-800 dark:text-white'
                      : 'text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
                  }`}
                >
                  <Monitor className="h-3.5 w-3.5" />
                </button>
                <button
                  type="button"
                  onClick={() => setViewportMode('mobile')}
                  aria-label="Mobile preview"
                  className={`inline-flex h-6 w-6 items-center justify-center rounded ${
                    viewportMode === 'mobile'
                      ? 'bg-slate-200 text-slate-900 dark:bg-slate-800 dark:text-white'
                      : 'text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
                  }`}
                >
                  <Smartphone className="h-3.5 w-3.5" />
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setRuntimeError(null);
                    setReloadNonce((n) => n + 1);
                  }}
                  aria-label="Reload preview"
                  className="inline-flex h-6 w-6 items-center justify-center rounded text-slate-500 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800"
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                </button>
                {isPublished && (
                  <button
                    type="button"
                    onClick={() => onChangeMode('app')}
                    title="Open Published App full-screen"
                    className="ml-1 inline-flex h-6 items-center gap-1 rounded px-1.5 text-[11px] font-medium text-emerald-800 hover:bg-emerald-50 dark:text-emerald-400 dark:hover:bg-emerald-950/40"
                  >
                    <ExternalLink className="h-3 w-3" />
                    <span>Open App</span>
                  </button>
                )}
              </div>
            </div>

            {runtimeError && (
              <div className="flex shrink-0 items-center justify-between gap-2 border-b border-red-200 bg-red-50 px-3 py-1.5 text-xs text-red-800 dark:border-red-900/50 dark:bg-red-950/50 dark:text-red-200">
                <div className="flex min-w-0 items-center gap-1.5">
                  <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
                  <span className="truncate font-mono text-[11px]">{runtimeError.message}</span>
                </div>
                <button
                  type="button"
                  onClick={handleFixRuntimeError}
                  className="inline-flex shrink-0 items-center gap-1 rounded-md bg-red-700 px-2 py-0.5 text-[11px] font-medium text-white hover:bg-red-800"
                >
                  <Wrench className="h-3 w-3" />
                  <span>Fix with VerdantUI</span>
                </button>
              </div>
            )}

            <div className="flex min-h-0 flex-1 items-stretch justify-center overflow-auto p-0">
              <div
                className={
                  viewportMode === 'mobile'
                    ? 'my-4 w-[390px] max-w-full overflow-hidden rounded-2xl border border-slate-200 shadow-sm dark:border-slate-800'
                    : 'h-full w-full'
                }
              >
                <PrismSandboxFrame
                  prismId={project.id}
                  snapshot={project.draft}
                  darkMode={darkMode}
                  bus={bus}
                  bridgeRuntime={bridgeRuntime}
                  reloadNonce={reloadNonce}
                  registerAsEndpoint={true}
                  onRuntimeError={setRuntimeError}
                />
              </div>
            </div>

            {/* Collapsible OpenTelemetry Stream Activity Strip */}
            <div className="shrink-0 border-t border-slate-200 bg-white dark:border-slate-800 dark:bg-[#0F141C]">
              <div className="flex h-8 items-center justify-between px-3 text-[11px]">
                <button
                  type="button"
                  aria-expanded={traceOpen}
                  onClick={() => setTraceOpen((prev) => !prev)}
                  className="inline-flex items-center gap-1.5 font-medium text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-200"
                >
                  <Activity className="h-3.5 w-3.5 text-emerald-700 dark:text-emerald-400" />
                  <span>OTel Stream</span>
                  <span className="font-mono text-[10px] text-slate-400 tabular-nums">
                    ({recentEvents.length})
                  </span>
                </button>
                {traceOpen && recentEvents.length > 0 && (
                  <button
                    type="button"
                    onClick={handleExportOtlp}
                    className="font-mono text-[10px] text-emerald-800 hover:underline dark:text-emerald-400"
                  >
                    Export OTLP JSON
                  </button>
                )}
              </div>
              {traceOpen && (
                <div className="max-h-36 overflow-y-auto border-t border-slate-100 px-3 py-1.5 font-mono text-[11px] dark:border-slate-800/80">
                  {recentEvents.length === 0 ? (
                    <div className="py-1 text-slate-400">
                      No sensory events yet. Interact with the preview to stream OTel events.
                    </div>
                  ) : (
                    <ul className="space-y-1">
                      {[...recentEvents].reverse().map((ev) => (
                        <li
                          key={ev.id}
                          className="flex items-center justify-between gap-2 text-slate-600 dark:text-slate-300"
                        >
                          <span className="truncate">
                            <span
                              className={
                                ev.status === 'error'
                                  ? 'mr-1.5 text-red-600 dark:text-red-400'
                                  : 'mr-1.5 text-emerald-700 dark:text-emerald-400'
                              }
                            >
                              ●
                            </span>
                            {ev.name}
                          </span>
                          <span className="shrink-0 text-[10px] text-slate-400">
                            {ev.traceId.slice(0, 8)}
                          </span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
