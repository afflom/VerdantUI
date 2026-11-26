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
  Copy,
  Download,
  ExternalLink,
  Eye,
  Globe,
  Layers,
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
  X,
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
  matchStarterFromPrompt,
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

type PrismPanelTab = 'overview' | 'code' | 'package';
type SourceFileTab = 'html' | 'css' | 'js';
type PreviewPaneMode = 'preview' | 'prism';

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
  const [previewPaneMode, setPreviewPaneMode] = useState<PreviewPaneMode>('preview');
  const [appPrismModalOpen, setAppPrismModalOpen] = useState(false);
  const [panelTab, setPanelTab] = useState<PrismPanelTab>('overview');
  const [schemaExpanded, setSchemaExpanded] = useState(false);
  const [digestCopied, setDigestCopied] = useState(false);
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
    setAppPrismModalOpen(false);
  }, [project.id, mode]);

  useEffect(() => {
    if (!appPrismModalOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setAppPrismModalOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [appPrismModalOpen]);

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
    bridgeRuntime.evictPrism(project.id);
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
    bridgeRuntime,
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
        const matchedStarter = matchStarterFromPrompt(trimmed);
        if (matchedStarter) {
          const snap = createStarterSnapshot(matchedStarter);
          await store.updateDraft(project.id, () => ({
            name: snap.name,
            summary: snap.summary,
            streams: snap.streams,
            grants: snap.grants,
            app: snap.app,
          }));
          await store.appendMessage(project.id, {
            role: 'assistant',
            content: `Scaffolded **${snap.name}** in your live preview. Toggle **Model Stream** on for custom AI edits, or switch to **Prism → Code** to customize directly.`,
            updatedSections: ['html', 'css', 'js', 'meta'],
          });
          span.setAttributes({ 'verdant.studio.starter': matchedStarter });
          span.end('ok');
          return;
        }
        const msg = engineState.modelStreamEnabled
          ? `Model Stream is initializing (${engineState.progressPercent}%). Wait for it to finish, or switch the preview to Prism to edit code directly.`
          : 'Model Stream is currently **Off**. Toggle **Model Stream** on in the top bar for freeform AI app generation—or pick a starter / toggle **Prism** in the preview bar to edit HTML, CSS, and JS directly.';
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
        // Exclude transient Model Stream Off notices from model context
        const historyMessages: ThreadMessage[] = currentProj.messages
          .filter(
            (m) =>
              !(
                m.role === 'assistant' &&
                (m.content.startsWith('Model Stream is currently') ||
                  m.content.startsWith('Model Stream is initializing'))
              ),
          )
          .slice(-6)
          .map((m) => ({
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

  const isReadOnlyAppInspect = mode === 'app';

  const handleCopyDigest = () => {
    if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
      void navigator.clipboard.writeText(activeSnapshot.digest);
    }
    setDigestCopied(true);
    setTimeout(() => setDigestCopied(false), 1500);
  };

  const renderPrismPanelContent = () => (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden bg-white dark:bg-[#0F141C]">
      {/* Refined Prism Tab Strip */}
      <div className="flex shrink-0 items-center justify-between gap-2 border-b border-slate-200 px-4 py-2 dark:border-slate-800">
        <div className="flex items-center gap-1">
          {(
            [
              { id: 'overview', label: 'Overview' },
              { id: 'code', label: 'Code' },
              { id: 'package', label: 'Package (.holo)' },
            ] as const
          ).map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setPanelTab(t.id)}
              className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                panelTab === t.id
                  ? 'bg-slate-900 text-white dark:bg-emerald-700 dark:text-white'
                  : 'text-slate-600 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-200'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
        {isReadOnlyAppInspect && (
          <button
            type="button"
            onClick={() => {
              setAppPrismModalOpen(false);
              onChangeMode('studio');
            }}
            className="text-xs font-medium text-emerald-800 hover:underline dark:text-emerald-400"
          >
            Edit in Studio
          </button>
        )}
      </div>

      {/* Tab Body */}
      {panelTab === 'overview' && (
        <div className="min-h-0 flex-1 space-y-5 overflow-y-auto p-4 text-xs">
          {/* Description */}
          <section className="space-y-1.5">
            <label className="block text-[11px] font-semibold tracking-wide text-slate-500 uppercase dark:text-slate-400">
              Description
            </label>
            {isReadOnlyAppInspect ? (
              <p className="rounded-lg border border-slate-200 bg-slate-50/70 px-3 py-2 text-xs leading-relaxed text-slate-700 dark:border-slate-800 dark:bg-[#141B26] dark:text-slate-300">
                {activeSnapshot.summary || 'No description provided.'}
              </p>
            ) : (
              <input
                type="text"
                value={summaryDraft}
                placeholder="Describe what this Prism does so VerdantUI knows when to use it…"
                onBlur={() => void flushSummaryDraft(summaryDraft)}
                onChange={(e) => {
                  const val = e.target.value;
                  setSummaryDraft(val);
                  if (summaryDebounceRef.current) clearTimeout(summaryDebounceRef.current);
                  summaryDebounceRef.current = setTimeout(() => {
                    void flushSummaryDraft(val);
                  }, 180);
                }}
                className="w-full rounded-lg border border-slate-200 bg-[#F8F9FA] px-3 py-2 text-xs text-slate-900 focus:border-emerald-700 focus:bg-white focus:outline-none dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-100"
              />
            )}
          </section>

          {/* Permissions */}
          <section className="space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="text-[11px] font-semibold tracking-wide text-slate-500 uppercase dark:text-slate-400">
                Permissions
              </h3>
              <span className="text-[11px] text-slate-400">Sandboxed · Denial-wins</span>
            </div>
            <div className="divide-y divide-slate-100 rounded-xl border border-slate-200 bg-[#F8F9FA]/60 dark:divide-slate-800 dark:border-slate-800 dark:bg-[#141B26]/60">
              {(
                [
                  {
                    key: 'observer',
                    label: 'VerdantUI & Threads',
                    desc: 'Let Threads run this app and let this app call local AI.',
                  },
                  {
                    key: 'storage',
                    label: 'Local Storage',
                    desc: 'Save app state privately inside this browser.',
                  },
                  {
                    key: 'browser',
                    label: 'Browser Actions',
                    desc: 'Clipboard copy, file downloads, and notifications.',
                  },
                  {
                    key: 'network',
                    label: 'Internet Access',
                    desc: 'Allow outbound HTTP fetch requests (off by default).',
                  },
                ] as const
              ).map(({ key, label, desc }) => {
                const enabled = activeSnapshot.grants[key];
                return (
                  <div
                    key={key}
                    className="flex items-center justify-between gap-3 px-3.5 py-2.5"
                  >
                    <div className="min-w-0">
                      <div className="flex items-center gap-1.5 font-medium text-slate-900 dark:text-slate-100">
                        {key === 'network' && (
                          <Globe className="h-3.5 w-3.5 text-slate-400" />
                        )}
                        <span>{label}</span>
                      </div>
                      <p className="mt-0.5 text-[11px] text-slate-500 dark:text-slate-400">
                        {desc}
                      </p>
                    </div>
                    <button
                      type="button"
                      role="switch"
                      aria-checked={enabled}
                      onClick={async () => {
                        if (isReadOnlyAppInspect) {
                          // Update published & draft grants WITHOUT accidentally publishing unfinished draft code!
                          await store.updatePublishedGrants(project.id, (prev) => ({
                            ...prev,
                            [key]: !prev[key],
                          }));
                          bridgeRuntime.evictPrism(project.id);
                        } else {
                          await store.updateDraft(project.id, (prev) => ({
                            ...prev,
                            grants: { ...prev.grants, [key]: !prev.grants[key] },
                          }));
                        }
                      }}
                      className={`inline-flex h-6 w-11 shrink-0 items-center rounded-full p-0.5 transition-colors ${
                        enabled
                          ? 'bg-[#0F5132] dark:bg-emerald-600'
                          : 'bg-slate-300 dark:bg-slate-700'
                      }`}
                    >
                      <span
                        className={`h-5 w-5 rounded-full bg-white shadow-xs transition-transform ${
                          enabled ? 'translate-x-5' : 'translate-x-0'
                        }`}
                      />
                    </button>
                  </div>
                );
              })}
            </div>
          </section>

          {/* Quick Test */}
          <section className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <h3 className="text-[11px] font-semibold tracking-wide text-slate-500 uppercase dark:text-slate-400">
                Quick Test
              </h3>
              <span className="text-[11px] text-slate-500 dark:text-slate-400">
                In any Thread:{' '}
                <code className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-slate-800 dark:bg-slate-800 dark:text-slate-200">
                  @{activeSnapshot.name}
                </code>
              </span>
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                value={testInputText}
                onChange={(e) => setTestInputText(e.target.value)}
                placeholder='{"action": "status"} or plain text'
                className="flex-1 rounded-lg border border-slate-200 bg-[#F8F9FA] px-3 py-1.5 font-mono text-[11px] text-slate-900 focus:border-emerald-700 focus:bg-white focus:outline-none dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-100"
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
                className="inline-flex items-center gap-1.5 rounded-lg bg-slate-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-slate-700 disabled:opacity-50 dark:bg-emerald-700 dark:hover:bg-emerald-600"
              >
                <Send className="h-3 w-3" />
                <span>{testRunning ? 'Running…' : 'Run'}</span>
              </button>
            </div>
            {testResultText && (
              <pre className="max-h-36 overflow-auto rounded-lg border border-slate-200 bg-[#F8F9FA] p-2.5 font-mono text-[11px] text-slate-700 dark:border-slate-800 dark:bg-[#0B0F14] dark:text-slate-300">
                {testResultText}
              </pre>
            )}
          </section>
        </div>
      )}

      {panelTab === 'code' && (
        <div className="flex min-h-0 flex-1 flex-col p-4">
          <div className="mb-2.5 flex items-center justify-between gap-2">
            <div className="flex items-center gap-1">
              {(['html', 'css', 'js'] as const).map((f) => (
                <button
                  key={f}
                  type="button"
                  onClick={() => {
                    if (!isReadOnlyAppInspect) {
                      void flushSourceDraft(sourceTab, sourceDraft);
                    }
                    setSourceTab(f);
                    setSourceDraft(activeSnapshot.app[f]);
                    sourceSnapshotPushedRef.current = false;
                  }}
                  className={`inline-flex items-center gap-1 rounded-md px-2.5 py-1 font-mono text-[11px] font-medium uppercase transition-colors ${
                    sourceTab === f
                      ? 'bg-slate-900 text-white dark:bg-emerald-700'
                      : 'bg-slate-100 text-slate-600 hover:text-slate-900 dark:bg-[#141B26] dark:text-slate-300 dark:hover:text-white'
                  }`}
                >
                  <Code2 className="h-3 w-3" />
                  {f}
                </button>
              ))}
            </div>
            <span className="text-[11px] text-slate-400">
              {isReadOnlyAppInspect
                ? 'Published snapshot (read-only)'
                : sourceTab === 'css'
                  ? 'Hot-swaps live without reloading state'
                  : 'Updates Preview automatically'}
            </span>
          </div>
          <textarea
            readOnly={isReadOnlyAppInspect}
            value={isReadOnlyAppInspect ? activeSnapshot.app[sourceTab] : sourceDraft}
            onBlur={() => {
              if (!isReadOnlyAppInspect) {
                void flushSourceDraft(sourceTab, sourceDraft);
                sourceSnapshotPushedRef.current = false;
              }
            }}
            onChange={(e) => {
              if (isReadOnlyAppInspect) return;
              const nextVal = e.target.value;
              setSourceDraft(nextVal);
              setRuntimeError(null);
              if (sourceDebounceRef.current) clearTimeout(sourceDebounceRef.current);
              sourceDebounceRef.current = setTimeout(() => {
                void flushSourceDraft(sourceTab, nextVal);
              }, 180);
            }}
            spellCheck={false}
            className="min-h-[260px] flex-1 resize-none rounded-xl border border-slate-200 bg-[#F8F9FA] p-3 font-mono text-[12px] leading-relaxed text-slate-900 focus:border-emerald-700 focus:bg-white focus:outline-none dark:border-slate-700 dark:bg-[#0B0F14] dark:text-slate-100"
          />
        </div>
      )}

      {panelTab === 'package' && (
        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4 text-xs">
          <div className="rounded-xl border border-slate-200 bg-[#F8F9FA]/70 p-4 dark:border-slate-800 dark:bg-[#141B26]/70">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0 space-y-1">
                <div className="font-semibold text-slate-900 dark:text-white">
                  {activeSnapshot.name}.holo
                </div>
                <div className="flex items-center gap-1.5 font-mono text-[11px] text-slate-600 dark:text-slate-400">
                  <span className="truncate">{activeSnapshot.digest}</span>
                  <button
                    type="button"
                    onClick={handleCopyDigest}
                    className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded hover:bg-slate-200 dark:hover:bg-slate-700"
                    title="Copy SHA-256 digest"
                  >
                    {digestCopied ? (
                      <Check className="h-3 w-3 text-emerald-700 dark:text-emerald-400" />
                    ) : (
                      <Copy className="h-3 w-3" />
                    )}
                  </button>
                </div>
                <p className="text-[11px] text-slate-500 dark:text-slate-400">
                  Self-contained, content-addressed Prism package verified by SHA-256.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={handleExportHolo}
                  className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#0F141C] dark:text-slate-200 dark:hover:bg-slate-800"
                >
                  <Download className="h-3.5 w-3.5" />
                  <span>Export .holo</span>
                </button>
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#0F141C] dark:text-slate-200 dark:hover:bg-slate-800"
                >
                  <Upload className="h-3.5 w-3.5" />
                  <span>Import .holo</span>
                </button>
              </div>
            </div>

            {importError && (
              <div className="mt-2.5 rounded-lg border border-red-200 bg-red-50 px-3 py-1.5 text-[11px] text-red-700 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-300">
                {importError}
              </div>
            )}

            {isPublished && (
              <div className="mt-3 flex items-center justify-between border-t border-slate-200/80 pt-3 dark:border-slate-800">
                <span className="text-[11px] text-slate-500 dark:text-slate-400">
                  Published Prisms are callable from any Thread and other Prisms.
                </span>
                <button
                  type="button"
                  onClick={() => {
                    setAppPrismModalOpen(false);
                    bridgeRuntime.evictPrism(project.id);
                    onChangeMode('studio');
                    void store.unpublish(project.id);
                  }}
                  className="text-[11px] font-medium text-red-600 hover:underline dark:text-red-400"
                >
                  Unpublish
                </button>
              </div>
            )}
          </div>

          {/* Collapsible Advanced Input Contract (JSON Schema 2020-12) */}
          <div className="rounded-xl border border-slate-200 dark:border-slate-800">
            <button
              type="button"
              aria-expanded={schemaExpanded}
              onClick={() => setSchemaExpanded((prev) => !prev)}
              className="flex w-full items-center justify-between px-3.5 py-2.5 text-left font-medium text-slate-700 hover:bg-slate-50 dark:text-slate-300 dark:hover:bg-slate-800/60"
            >
              <span className="inline-flex items-center gap-1.5">
                {schemaExpanded ? (
                  <ChevronDown className="h-3.5 w-3.5 text-slate-400" />
                ) : (
                  <ChevronRight className="h-3.5 w-3.5 text-slate-400" />
                )}
                <span>Input Contract (JSON Schema 2020-12)</span>
              </span>
              <span className="text-[11px] font-normal text-slate-400">Advanced</span>
            </button>
            {schemaExpanded && (
              <div className="space-y-2 border-t border-slate-200 p-3.5 dark:border-slate-800">
                <textarea
                  readOnly={isReadOnlyAppInspect}
                  rows={7}
                  spellCheck={false}
                  value={
                    isReadOnlyAppInspect
                      ? JSON.stringify(activeSnapshot.streams.inputs, null, 2)
                      : schemaDraft
                  }
                  onChange={(e) => setSchemaDraft(e.target.value)}
                  className="w-full rounded-lg border border-slate-200 bg-[#F8F9FA] p-2.5 font-mono text-[11px] text-slate-900 focus:border-emerald-700 focus:bg-white focus:outline-none dark:border-slate-700 dark:bg-[#0B0F14] dark:text-slate-100"
                />
                {schemaError && (
                  <div className="text-[11px] text-red-600 dark:text-red-400">{schemaError}</div>
                )}
                {!isReadOnlyAppInspect && (
                  <div className="flex justify-end">
                    <button
                      type="button"
                      onClick={() => {
                        try {
                          const parsed = JSON.parse(schemaDraft) as JsonSchema2020Object;
                          if (
                            !parsed ||
                            parsed.type !== 'object' ||
                            typeof parsed.properties !== 'object'
                          ) {
                            throw new Error(
                              'Schema must be an object with "type": "object" and "properties".',
                            );
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
                      className="rounded-md bg-slate-900 px-3 py-1 text-[11px] font-medium text-white hover:bg-slate-700 dark:bg-emerald-700 dark:hover:bg-emerald-600"
                    >
                      Save Schema
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className="flex h-full flex-col overflow-hidden bg-[#F8F9FA] dark:bg-[#0B0F14]">
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
      {/* Calm Single-Line Workspace Header */}
      <div className="flex h-11 shrink-0 items-center justify-between gap-2 border-b border-slate-200 bg-white px-4 dark:border-slate-800 dark:bg-[#0F141C]">
        <div className="flex min-w-0 items-center gap-2.5">
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
              className="h-7 w-48 rounded-md border border-emerald-600 bg-white px-2 text-xs font-semibold text-slate-900 focus:outline-none dark:bg-[#141B26] dark:text-white"
            />
          ) : (
            <button
              type="button"
              onClick={() => {
                if (mode === 'studio') setEditingTitle(true);
              }}
              className="truncate text-[13px] font-semibold text-slate-900 hover:text-emerald-800 dark:text-slate-100 dark:hover:text-emerald-400"
              title={mode === 'studio' ? 'Click to rename Prism' : activeSnapshot.name}
            >
              {activeSnapshot.name}
            </button>
          )}

          <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 px-2 py-0.5 text-[11px] font-medium text-slate-600 dark:border-slate-700 dark:text-slate-300">
            <span className={`h-1.5 w-1.5 rounded-full ${statusBadge.dot}`} />
            {statusBadge.label}
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          {mode === 'studio' && project.revisions.length > 0 && (
            <button
              type="button"
              onClick={() => void handleUndo()}
              title={`Undo last change (${project.revisions.length} saved)`}
              className="inline-flex h-7 items-center gap-1 rounded-md border border-slate-200 bg-white px-2.5 text-xs font-medium text-slate-600 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-300 dark:hover:bg-slate-800"
            >
              <Undo2 className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Undo</span>
            </button>
          )}

          {mode === 'app' ? (
            <>
              <button
                type="button"
                data-testid="app-prism-modal-toggle"
                onClick={() => setAppPrismModalOpen(true)}
                className="inline-flex h-7 items-center gap-1.5 rounded-md border border-slate-200 bg-white px-2.5 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-200 dark:hover:bg-slate-800"
              >
                <Layers className="h-3.5 w-3.5" />
                <span>Prism</span>
              </button>
              <button
                type="button"
                onClick={() => onChangeMode('studio')}
                className="inline-flex h-7 items-center gap-1.5 rounded-md bg-slate-900 px-2.5 text-xs font-medium text-white hover:bg-slate-700 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-slate-200"
              >
                <span>Edit in Studio</span>
              </button>
            </>
          ) : (
            <>
              {isPublished && (
                <button
                  type="button"
                  onClick={() => onChangeMode('app')}
                  className="inline-flex h-7 items-center gap-1.5 rounded-md border border-slate-200 bg-white px-2.5 text-xs font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-[#141B26] dark:text-slate-200 dark:hover:bg-slate-800"
                >
                  <ExternalLink className="h-3 w-3" />
                  <span>Open App</span>
                </button>
              )}
              <button
                type="button"
                data-testid="prism-publish-btn"
                onClick={() => void handlePublish()}
                className="inline-flex h-7 items-center gap-1.5 rounded-md bg-[#0F5132] px-3 text-xs font-medium text-white hover:bg-[#0B3D25] transition-colors"
              >
                {justPublished ? (
                  <>
                    <Check className="h-3.5 w-3.5" />
                    <span>Published</span>
                  </>
                ) : (
                  <>
                    <Play className="h-3 w-3" />
                    <span>{hasUnpublishedChanges ? 'Republish' : 'Publish'}</span>
                  </>
                )}
              </button>
            </>
          )}
        </div>
      </div>

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
                Edit in Studio
              </button>
            </div>
          )}

          {/* Refined Centered Prism Modal (App Mode) */}
          {appPrismModalOpen && (
            <div
              role="dialog"
              aria-modal="true"
              aria-label={`${activeSnapshot.name} Prism settings`}
              className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/50 p-4 backdrop-blur-[2px]"
              onClick={() => setAppPrismModalOpen(false)}
            >
              <div
                className="flex max-h-[82vh] w-full max-w-2xl flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-800 dark:bg-[#0F141C]"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-800">
                  <div className="flex min-w-0 items-center gap-2">
                    <Layers className="h-4 w-4 shrink-0 text-emerald-700 dark:text-emerald-400" />
                    <span className="truncate text-sm font-semibold text-slate-900 dark:text-white">
                      {activeSnapshot.name}
                    </span>
                    <span className="font-mono text-[11px] text-slate-400">
                      #{shortPrismDigest(activeSnapshot.digest)}
                    </span>
                  </div>
                  <button
                    type="button"
                    aria-label="Close Prism modal"
                    onClick={() => setAppPrismModalOpen(false)}
                    className="inline-flex h-7 w-7 items-center justify-center rounded-md text-slate-500 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-white"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
                {renderPrismPanelContent()}
              </div>
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

                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="flex w-full items-center justify-center gap-1.5 rounded-lg border border-dashed border-slate-300 bg-white/60 px-3 py-2 text-xs font-medium text-slate-600 hover:border-emerald-700/50 hover:text-emerald-800 dark:border-slate-700 dark:bg-[#141B26]/60 dark:text-slate-300 dark:hover:border-emerald-500/50 dark:hover:text-emerald-400 transition-colors"
                  >
                    <Upload className="h-3.5 w-3.5" />
                    <span>Import existing .holo package</span>
                  </button>
                  {importError && (
                    <div className="rounded-lg border border-red-200 bg-red-50 px-2.5 py-1.5 text-[11px] text-red-700 dark:border-red-900/50 dark:bg-red-950/40 dark:text-red-300">
                      {importError}
                    </div>
                  )}
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
              {project.messages.length > 0 && !generating && (
                <div className="mb-1.5 flex justify-end">
                  <button
                    type="button"
                    onClick={() => void store.clearMessages(project.id)}
                    className="text-[10px] font-medium text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
                  >
                    Clear chat
                  </button>
                </div>
              )}
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

          {/* Right Pane: Real-Time Preview / Prism Toggle + Collapsible OTel Stream Strip */}
          <div className="flex min-h-0 flex-col bg-[#F8F9FA] dark:bg-[#0B0F14]">
            <div className="flex h-10 shrink-0 items-center justify-between border-b border-slate-200 bg-white/90 px-3 text-xs dark:border-slate-800 dark:bg-[#0F141C]/90">
              <div
                role="tablist"
                aria-label="Preview or Prism view"
                data-testid="prism-view-toggle"
                className="inline-flex h-7 items-center rounded-lg border border-slate-200 bg-slate-100/80 p-0.5 text-[11px] font-medium dark:border-slate-700 dark:bg-[#141B26]"
              >
                <button
                  type="button"
                  role="tab"
                  aria-selected={previewPaneMode === 'preview'}
                  onClick={() => setPreviewPaneMode('preview')}
                  className={`inline-flex h-6 items-center gap-1.5 rounded-md px-2.5 transition-colors ${
                    previewPaneMode === 'preview'
                      ? 'bg-white text-slate-900 shadow-xs dark:bg-slate-800 dark:text-white'
                      : 'text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
                  }`}
                >
                  <Eye className="h-3.5 w-3.5" />
                  <span>Preview</span>
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={previewPaneMode === 'prism'}
                  data-testid="prism-focus-toggle"
                  onClick={() => setPreviewPaneMode('prism')}
                  className={`inline-flex h-6 items-center gap-1.5 rounded-md px-2.5 transition-colors ${
                    previewPaneMode === 'prism'
                      ? 'bg-white text-slate-900 shadow-xs dark:bg-slate-800 dark:text-white'
                      : 'text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
                  }`}
                >
                  <Layers className="h-3.5 w-3.5" />
                  <span>Prism</span>
                </button>
              </div>

              <div className="flex items-center gap-1">
                {previewPaneMode === 'preview' ? (
                  <>
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
                  </>
                ) : (
                  <span
                    className="font-mono text-[11px] text-slate-400 dark:text-slate-500"
                    title={project.draft.digest}
                  >
                    #{shortPrismDigest(project.draft.digest)}
                  </span>
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

            {/* Keep sandbox mounted when toggling to Prism so app state & Quick Test stay live */}
            <div
              className={`min-h-0 flex-1 items-stretch justify-center overflow-auto p-0 ${
                previewPaneMode === 'preview' ? 'flex' : 'hidden'
              }`}
            >
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

            {previewPaneMode === 'prism' && (
              <div
                data-testid="prism-focus-panel"
                className="flex min-h-0 flex-1 flex-col overflow-hidden"
              >
                {renderPrismPanelContent()}
              </div>
            )}

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
