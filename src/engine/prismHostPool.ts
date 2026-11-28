/**
 * Prism Host Runtime Pool & FIFO Queued `verdant.ask()` Coordinator.
 *
 * Solves two real browser runtime problems cleanly:
 * 1. Serialized FIFO `verdant.ask()` queue with isolated `threadId = "prism:<id>"` KV namespaces
 *    so concurrent AI-enabled Prism requests never preempt each other or pollute Thread KV state.
 * 2. Warm LRU(2) off-screen sandbox pool (60s idle eviction, 3s ready handshake) so Threads can
 *    invoke Published Prisms deterministically via typed `onInput -> return` RPC even when the
 *    Prism is not mounted full-screen.
 */

import type { ThreadMessage } from '@assistant-ui/react';
import { formatMessagesForLlama, stripReasoningTags } from './assistantAdapter';
import type { ContainerGrantMatrix } from './cognitiveObserver';
import type { ObserverStreamBus, OtelTraceContext } from './observerStreamBus';
import type { PrismBuildWorkerClient } from './prismBuildClient';
import { buildPrismSandboxDocument } from './prismSandboxHtml';
import type { PrismStore, VerdantPrismSnapshot } from './prismStudio';
import { evaluateDeclaredModel, formatRequestLine, type PrismDeclaration } from './verdantStudioApi';
import { evaluateContextBoundary, type ContextCompartmentId } from './unifiedPossessionModel';
import { isInternalBootstrapModel } from './modelCatalog';
import type { LlamaVTensorWorkerClient } from './workerClient';

export interface PrismHostCallbacks {
  readonly onNotify?: (notice: { readonly prismId: string; readonly title: string; readonly body: string }) => void;
  readonly onOpenPrism?: (prismId: string) => void;
}

export interface PrismAskRequest {
  readonly prismId: string;
  readonly prompt: string;
  readonly onToken?: (accumulatedText: string) => void;
  readonly signal?: AbortSignal;
  readonly parentContext?: OtelTraceContext;
}

export interface PrismBridgeRuntime {
  ask(req: PrismAskRequest): Promise<string>;
  getKv(prismId: string, grants: VerdantPrismSnapshot['grants'], key: string): Promise<unknown>;
  setKv(
    prismId: string,
    grants: VerdantPrismSnapshot['grants'],
    key: string,
    value: unknown,
  ): Promise<void>;
  copyText(prismId: string, grants: VerdantPrismSnapshot['grants'], text: string): Promise<boolean>;
  downloadFile(
    prismId: string,
    grants: VerdantPrismSnapshot['grants'],
    filename: string,
    content: string,
    mimeType: string,
  ): Promise<boolean>;
  notify(
    prismId: string,
    grants: VerdantPrismSnapshot['grants'],
    title: string,
    body: string,
  ): Promise<boolean>;
  getUiState(prismId: string, grants: VerdantPrismSnapshot['grants']): Promise<Record<string, unknown>>;
  callPrism(
    callerPrismId: string,
    grants: VerdantPrismSnapshot['grants'],
    targetNameOrId: string,
    payload: unknown,
    parentContext?: OtelTraceContext,
    callChain?: readonly string[],
  ): Promise<unknown>;
  openPrism(
    callerPrismId: string,
    grants: VerdantPrismSnapshot['grants'],
    targetNameOrId: string,
  ): Promise<boolean>;
  proxiedFetch(
    prismId: string,
    grants: VerdantPrismSnapshot['grants'],
    url: string,
    init?: { readonly method?: string; readonly headers?: Record<string, string>; readonly body?: string },
    parentContext?: OtelTraceContext,
    onChunk?: (accumulatedText: string) => void,
    signal?: AbortSignal,
  ): Promise<{ readonly status: number; readonly ok: boolean; readonly text: string }>;
  handlePrismEmit(
    prismId: string,
    grants: VerdantPrismSnapshot['grants'],
    name: string,
    body: unknown,
    attributes?: Record<string, string | number | boolean>,
    traceContext?: OtelTraceContext,
  ): { readonly accepted: boolean; readonly reason?: string };
  evictPrism(prismId: string): void;
}

interface PooledPrismInstance {
  readonly prismId: string;
  readonly digest: string;
  readonly channelId: string;
  readonly iframe: HTMLIFrameElement;
  readonly readyPromise: Promise<void>;
  readonly pendingDispatches: Map<
    string,
    { resolve: (v: unknown) => void; reject: (e: Error) => void }
  >;
  lastUsedAt: number;
  cleanup: () => void;
}

const MAX_POOLED_PRISMS = 2;
const IDLE_EVICT_MS = 60_000;
const READY_TIMEOUT_MS = 3_000;
const FETCH_TIMEOUT_MS = 20_000;
const MAX_PRISM_CALL_DEPTH = 4;

export class PrismHostPool implements PrismBridgeRuntime {
  private askTail: Promise<unknown> = Promise.resolve();
  private readonly pool = new Map<string, PooledPrismInstance>();
  private dispatchSeq = 1;
  private evictTimer: ReturnType<typeof setInterval> | null = null;
  private callbacks: PrismHostCallbacks = {};
  private buildClient: PrismBuildWorkerClient | null = null;
  private getBuildClientFn: (() => Promise<PrismBuildWorkerClient>) | null = null;

  constructor(
    private readonly bus: ObserverStreamBus,
    private readonly store: PrismStore,
    private readonly grantsMatrix: ContainerGrantMatrix,
    private readonly workerClient: LlamaVTensorWorkerClient | null,
    callbacks?: PrismHostCallbacks,
    buildClient?: PrismBuildWorkerClient | (() => Promise<PrismBuildWorkerClient>) | null,
  ) {
    if (callbacks) this.callbacks = callbacks;
    if (typeof buildClient === 'function') this.getBuildClientFn = buildClient;
    else if (buildClient) this.buildClient = buildClient;
    if (typeof window !== 'undefined') {
      this.evictTimer = setInterval(() => this.evictIdle(), 15_000);
    }
  }

  private async resolveBuildClient(): Promise<PrismBuildWorkerClient | null> {
    if (this.buildClient) return this.buildClient;
    if (!this.getBuildClientFn) return null;
    this.buildClient = await this.getBuildClientFn();
    return this.buildClient;
  }

  setBuildClient(client: PrismBuildWorkerClient | null): void {
    this.buildClient = client;
  }

  setCallbacks(callbacks: PrismHostCallbacks): void {
    this.callbacks = callbacks;
  }

  evictPrism(prismId: string): void {
    const inst = this.pool.get(prismId);
    if (!inst) return;
    inst.cleanup();
    this.pool.delete(prismId);
  }

  dispose(): void {
    if (this.evictTimer !== null) {
      clearInterval(this.evictTimer);
      this.evictTimer = null;
    }
    for (const inst of this.pool.values()) {
      inst.cleanup();
    }
    this.pool.clear();
  }

  /**
   * Serializes `verdant.ask()` calls through a FIFO queue with isolated `prism:<id>` KV state.
   */
  ask(req: PrismAskRequest): Promise<string> {
    const run = async (): Promise<string> => {
      if (req.signal?.aborted) {
        throw new Error('Prism AI request cancelled.');
      }
      const span = this.bus.startSpan(`prism.ask:${req.prismId}`, {
        endpointKind: 'prism',
        endpointId: req.prismId,
        kind: 'CLIENT',
        parentContext: req.parentContext,
      });

      try {
        if (!this.workerClient) {
          throw new Error('VerdantUI inference worker is unavailable.');
        }
        const state = this.workerClient.getState();
        const hasRealModel =
          state.model !== null && !isInternalBootstrapModel(state.model.id);

        if (!hasRealModel) {
          const notice =
            state.streamingConsent === 'accepted'
              ? state.status === 'downloading' || state.status === 'loading'
                ? `Streaming capabilities to your browser (${state.progressPercent}%). Try again in a moment.`
                : 'Send a prompt in Chat or wait a moment for browser capabilities to initialize.'
              : state.streamingConsent === 'pending'
                ? 'Enable browser streaming in Settings → Content Streams to enable AI capabilities inside apps.'
                : 'Offline-only mode is active. Turn off Offline-Only Mode in Settings → Content Streams to enable AI capabilities inside apps.';
          span.addEvent('prism.ask.deferred', { reason: notice }, undefined, 'ok');
          span.end('ok');
          throw new Error(notice);
        }

        const syntheticUser: ThreadMessage = {
          id: `prism_ask_${Date.now()}`,
          role: 'user',
          createdAt: new Date(),
          content: [{ type: 'text', text: req.prompt }],
          attachments: [],
          metadata: { custom: {} },
        } as ThreadMessage;
        const prompt = formatMessagesForLlama(
          [syntheticUser],
          state.model!,
          state.promptMode,
          'You are VerdantUI assisting a live local Prism app. Be concise, direct, and accurate.',
          { thinkingMode: 'off' },
        );
        const abortCtrl = new AbortController();
        const onOuterAbort = () => abortCtrl.abort();
        if (req.signal) {
          if (req.signal.aborted) abortCtrl.abort();
          else req.signal.addEventListener('abort', onOuterAbort, { once: true });
        }

        let finalText = '';
        try {
          for await (const ev of this.workerClient.streamGenerate(
            prompt,
            abortCtrl.signal,
            { ...state.sampling, temperature: 0.4, maxTokens: 768 },
            `prism:${req.prismId}`,
          )) {
            finalText = stripReasoningTags(ev.accumulatedText);
            req.onToken?.(finalText);
          }
        } finally {
          req.signal?.removeEventListener('abort', onOuterAbort);
        }

        const cleaned = finalText.trim();
        span.setAttributes({ 'verdant.output.chars': cleaned.length });
        span.end('ok');
        return cleaned;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        span.end('error', msg);
        throw err;
      }
    };

    const next = this.askTail.then(run, run);
    this.askTail = next.then(
      () => undefined,
      () => undefined,
    );
    return next;
  }

  private resolvePrismCompartmentId(prismId: string): ContextCompartmentId {
    const proj = this.store.get(prismId);
    const declared = proj?.declaration?.concept.compartmentId;
    if (declared && declared.trim()) return declared.trim();
    return this.workerClient?.cognitiveObserver?.possession.getActiveCompartmentId() ?? 'personal';
  }

  async getKv(
    prismId: string,
    grants: VerdantPrismSnapshot['grants'],
    key: string,
  ): Promise<unknown> {
    if (!grants.storage || !this.grantsMatrix.isAllowed('prism-storage')) {
      throw new Error(`Storage grant denied for Prism "${prismId}" (FK-07).`);
    }
    const space = this.resolvePrismCompartmentId(prismId);
    return this.store.getKv(prismId, `space:${space}:${key}`);
  }

  async setKv(
    prismId: string,
    grants: VerdantPrismSnapshot['grants'],
    key: string,
    value: unknown,
  ): Promise<void> {
    if (!grants.storage || !this.grantsMatrix.isAllowed('prism-storage')) {
      throw new Error(`Storage grant denied for Prism "${prismId}" (FK-07).`);
    }
    const space = this.resolvePrismCompartmentId(prismId);
    await this.store.setKv(prismId, `space:${space}:${key}`, value);
  }

  /**
   * Mediates `window.verdant.emit('constraint' | 'sensory' | 'impulse' | custom, payload)`
   * through `ContainerGrantMatrix` (FK-07) and `ContextBoundaryGate` (DEC-010).
   */
  handlePrismEmit(
    prismId: string,
    grants: VerdantPrismSnapshot['grants'],
    name: string,
    body: unknown,
    attributes?: Record<string, string | number | boolean>,
    traceContext?: OtelTraceContext,
  ): { readonly accepted: boolean; readonly reason?: string } {
    if (!grants.observer) {
      return { accepted: false, reason: 'Observer stream grant is Off for this Prism.' };
    }
    const possession = this.workerClient?.cognitiveObserver?.possession;
    const sourceSpace = this.resolvePrismCompartmentId(prismId);

    if (possession && possession.isHalted() && (name === 'constraint' || name === 'impulse')) {
      return { accepted: false, reason: 'Sharing is paused by owner kill switch.' };
    }

    if (name === 'constraint' && possession && body && typeof body === 'object') {
      const payload = body as {
        readonly opaqueOnly?: boolean;
        readonly windows?: readonly {
          readonly startMinuteOfDay?: number;
          readonly endMinuteOfDay?: number;
          readonly daysOfWeek?: readonly number[];
        }[];
      };
      const proj = this.store.get(prismId);
      const privLabel = proj?.draft.name || `prism:${prismId}`;
      const windows = Array.isArray(payload.windows) ? payload.windows : [];
      const nextConstraints = windows
        .filter(
          (w): w is { readonly startMinuteOfDay: number; readonly endMinuteOfDay: number; readonly daysOfWeek?: readonly number[] } =>
            typeof w.startMinuteOfDay === 'number' && typeof w.endMinuteOfDay === 'number',
        )
        .map((w, idx) => ({
          id: `${prismId}:live-win-${idx}`,
          privateLabel: privLabel,
          sourcePrismId: prismId,
          startMinuteOfDay: w.startMinuteOfDay,
          endMinuteOfDay: w.endMinuteOfDay,
          daysOfWeek: Array.isArray(w.daysOfWeek) ? w.daysOfWeek : [0, 1, 2, 3, 4, 5, 6],
          weightQ10: 960,
          shareOpaqueAvailability: payload.opaqueOnly !== false,
        }));
      possession.replacePrismScheduleConstraints(sourceSpace, prismId, nextConstraints);
      this.bus.emit({
        name: 'prism.constraint.updated',
        endpointKind: 'prism',
        endpointId: prismId,
        traceContext,
        attributes: {
          ...attributes,
          'verdant.constraint.windows': windows.length,
          'verdant.constraint.opaque': true,
        },
      });
      return { accepted: true };
    }

    if (name === 'sensory' && possession) {
      const payloadText = typeof body === 'string' ? body : JSON.stringify(body ?? '');
      const res = possession.ingestSensoryFrame({
        id: `sf_${prismId}_${Date.now().toString(36)}`,
        prismId,
        compartmentId: sourceSpace,
        channel: 'browser',
        sensitivity: 'internal',
        payloadText,
        receivedAt: Date.now(),
      });
      if (res.verdict === 'deny') {
        return { accepted: false, reason: res.reason };
      }
      this.bus.emit({
        name: 'prism.sensory',
        endpointKind: 'prism',
        endpointId: prismId,
        traceContext,
        attributes: { ...attributes, 'verdant.boundary.verdict': res.verdict },
      });
      return { accepted: true };
    }

    if (name === 'impulse' && possession && body && typeof body === 'object') {
      const imp = body as {
        readonly targetPrismId?: string;
        readonly targetCompartmentId?: ContextCompartmentId;
        readonly kind?: 'render-surface' | 'notify' | 'schedule-constraint' | 'stream-emit' | 'tool-invoke';
        readonly payload?: unknown;
        readonly opaqueOnly?: boolean;
      };
      const res = possession.dispatchActionImpulse({
        id: `imp_${prismId}_${Date.now().toString(36)}`,
        targetPrismId: String(imp.targetPrismId || prismId),
        sourceCompartmentId: sourceSpace,
        targetCompartmentId: imp.targetCompartmentId ?? sourceSpace,
        kind: imp.kind ?? 'stream-emit',
        payload: imp.payload ?? {},
        opaqueOnly: Boolean(imp.opaqueOnly),
        ownerApproved: grants.observer,
        emittedAt: Date.now(),
      });
      if (res.verdict !== 'allow') {
        return { accepted: false, reason: res.reason };
      }
      this.bus.emit({
        name: 'prism.impulse',
        endpointKind: 'prism',
        endpointId: prismId,
        traceContext,
        attributes: { ...attributes, 'verdant.impulse.kind': imp.kind ?? 'stream-emit' },
      });
      return { accepted: true };
    }

    this.bus.emit({
      name: String(name || 'prism.event'),
      endpointKind: 'prism',
      endpointId: prismId,
      traceContext,
      attributes,
      body,
    });
    return { accepted: true };
  }

  async copyText(
    prismId: string,
    grants: VerdantPrismSnapshot['grants'],
    text: string,
  ): Promise<boolean> {
    if (!grants.browser) {
      throw new Error(`Browser grant is Off for Prism "${prismId}".`);
    }
    if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    }
    this.bus.emit({
      name: 'prism.browser.copy',
      endpointKind: 'browser',
      endpointId: prismId,
      attributes: { 'verdant.chars': text.length },
    });
    return true;
  }

  async downloadFile(
    prismId: string,
    grants: VerdantPrismSnapshot['grants'],
    filename: string,
    content: string,
    mimeType: string,
  ): Promise<boolean> {
    if (!grants.browser) {
      throw new Error(`Browser grant is Off for Prism "${prismId}".`);
    }
    const safeName = filename.replace(/[/\\]/g, '_').trim() || 'export.txt';
    if (typeof document !== 'undefined' && typeof URL !== 'undefined') {
      const blob = new Blob([content], { type: mimeType || 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = safeName;
      a.click();
      URL.revokeObjectURL(url);
    }
    this.bus.emit({
      name: 'prism.browser.download',
      endpointKind: 'browser',
      endpointId: prismId,
      attributes: { 'file.name': safeName, 'file.size': content.length },
    });
    return true;
  }

  async notify(
    prismId: string,
    grants: VerdantPrismSnapshot['grants'],
    title: string,
    body: string,
  ): Promise<boolean> {
    if (!grants.browser) {
      throw new Error(`Browser grant is Off for Prism "${prismId}".`);
    }
    this.callbacks.onNotify?.({ prismId, title, body });
    this.bus.emit({
      name: 'prism.browser.notify',
      endpointKind: 'browser',
      endpointId: prismId,
      attributes: { title },
      body,
    });
    return true;
  }

  async getUiState(
    prismId: string,
    grants: VerdantPrismSnapshot['grants'],
  ): Promise<Record<string, unknown>> {
    if (!grants.observer) {
      throw new Error(`Observer grant is Off for Prism "${prismId}".`);
    }
    const engine = this.workerClient?.getState();
    const published = this.store.listPublishedSummaries();
    return {
      darkMode:
        typeof document !== 'undefined'
          ? document.documentElement.classList.contains('dark')
          : false,
      modelStreamEnabled: engine?.modelStreamEnabled ?? false,
      modelReady:
        Boolean(engine?.model && !isInternalBootstrapModel(engine.model.id)),
      publishedPrisms: published.map((p) => ({
        id: p.id,
        slug: p.slug,
        name: p.name,
        summary: p.summary,
      })),
    };
  }

  private resolveTargetPrismId(targetNameOrId: string): string | null {
    const clean = targetNameOrId.trim();
    if (!clean) return null;
    const direct = this.store.get(clean);
    if (direct?.published) return direct.id;
    const lower = clean.toLowerCase();
    const match = this.store
      .listPublishedSummaries()
      .find((p) => p.name.toLowerCase() === lower || p.slug.toLowerCase() === lower);
    return match ? match.id : null;
  }

  async callPrism(
    callerPrismId: string,
    grants: VerdantPrismSnapshot['grants'],
    targetNameOrId: string,
    payload: unknown,
    parentContext?: OtelTraceContext,
    callChain: readonly string[] = [],
  ): Promise<unknown> {
    if (!grants.observer) {
      throw new Error(`Observer grant is Off for Prism "${callerPrismId}".`);
    }
    const chain = callChain.includes(callerPrismId) ? callChain : [...callChain, callerPrismId];
    if (chain.length >= MAX_PRISM_CALL_DEPTH) {
      throw new Error(`Prism-to-Prism call depth exceeded limit (${MAX_PRISM_CALL_DEPTH}).`);
    }
    const targetId = this.resolveTargetPrismId(targetNameOrId);
    if (!targetId) {
      throw new Error(`Published Prism "${targetNameOrId}" not found.`);
    }
    if (targetId === callerPrismId) {
      throw new Error('Self-recursive Prism call refused.');
    }
    if (chain.includes(targetId)) {
      throw new Error(`Cyclic Prism-to-Prism call refused (${[...chain, targetId].join(' -> ')}).`);
    }
    const nextChain = [...chain, targetId];
    const { result } = await this.dispatchPublishedPrism(targetId, payload, {
      parentContext,
      callChain: nextChain,
    });
    return result;
  }

  async openPrism(
    callerPrismId: string,
    grants: VerdantPrismSnapshot['grants'],
    targetNameOrId: string,
  ): Promise<boolean> {
    if (!grants.observer) {
      throw new Error(`Observer grant is Off for Prism "${callerPrismId}".`);
    }
    const clean = targetNameOrId.trim();
    const direct = clean ? this.store.get(clean) : null;
    const targetId = direct ? direct.id : this.resolveTargetPrismId(targetNameOrId);
    if (!targetId) {
      throw new Error(`Prism "${targetNameOrId}" not found.`);
    }
    this.callbacks.onOpenPrism?.(targetId);
    return true;
  }

  async proxiedFetch(
    prismId: string,
    grants: VerdantPrismSnapshot['grants'],
    url: string,
    init?: { readonly method?: string; readonly headers?: Record<string, string>; readonly body?: string },
    parentContext?: OtelTraceContext,
    onChunk?: (accumulatedText: string) => void,
    signal?: AbortSignal,
  ): Promise<{ readonly status: number; readonly ok: boolean; readonly text: string }> {
    const span = this.bus.startSpan(`prism.fetch:${prismId}`, {
      endpointKind: 'network',
      endpointId: prismId,
      kind: 'CLIENT',
      parentContext,
      attributes: { 'url.full': url, 'http.request.method': init?.method ?? 'GET' },
    });

    if (!grants.network) {
      const msg = `Network grant is Off for this Prism. Enable Network in the Prism Focus drawer to allow fetching ${url}.`;
      span.end('error', msg);
      throw new Error(msg);
    }

    // Enforce DEC-010 non-leaking cross-context boundary & global owner kill switch on outbound fetch
    const possession = this.workerClient?.cognitiveObserver?.possession;
    if (possession) {
      const proj = this.store.get(prismId);
      const sourceSpace: ContextCompartmentId =
        (proj?.declaration?.concept.compartmentId as ContextCompartmentId | undefined) ?? 'personal';
      const user = possession.getUser();
      const boundary = evaluateContextBoundary({
        sourceCompartmentId: sourceSpace,
        destination: 'network',
        label: 'internal',
        payloadText: `${url}\n${init?.body ?? ''}`,
        ownerApproved: grants.network,
        bytesMatchSeen: true,
        halted: user.halted,
        foreignCompartments: Object.values(user.compartments),
      });
      if (boundary.verdict === 'deny') {
        const msg = `Outbound request blocked by privacy boundary: ${boundary.reason}`;
        span.end('error', msg);
        throw new Error(msg);
      }
    }

    let parsedUrl: URL;
    try {
      parsedUrl = new URL(url);
    } catch {
      const msg = `Invalid fetch URL "${url}".`;
      span.end('error', msg);
      throw new Error(msg);
    }
    if (parsedUrl.protocol !== 'https:' && parsedUrl.protocol !== 'http:') {
      const msg = `Unsupported URL protocol "${parsedUrl.protocol}".`;
      span.end('error', msg);
      throw new Error(msg);
    }

    const fetchCtrl = new AbortController();
    const timeoutId = setTimeout(() => fetchCtrl.abort(), FETCH_TIMEOUT_MS);
    const onOuterAbort = () => fetchCtrl.abort();
    if (signal) {
      if (signal.aborted) fetchCtrl.abort();
      else signal.addEventListener('abort', onOuterAbort, { once: true });
    }

    try {
      const res = await fetch(parsedUrl.toString(), {
        method: init?.method ?? 'GET',
        headers: init?.headers,
        body: init?.body,
        signal: fetchCtrl.signal,
      });
      let text = '';
      if (onChunk && res.body && typeof res.body.getReader === 'function') {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        while (text.length < 128_000) {
          const { done, value } = await reader.read();
          if (done) break;
          text += decoder.decode(value, { stream: true });
          onChunk(text.slice(0, 128_000));
        }
        text += decoder.decode();
      } else {
        text = await res.text();
      }
      span.setAttributes({ 'http.response.status_code': res.status });
      span.end(res.ok ? 'ok' : 'error');
      return { status: res.status, ok: res.ok, text: text.slice(0, 128_000) };
    } catch (err) {
      const msg =
        fetchCtrl.signal.aborted && !signal?.aborted
          ? `Fetch timed out after ${FETCH_TIMEOUT_MS}ms (${url}).`
          : err instanceof Error
            ? err.message
            : String(err);
      span.end('error', msg);
      throw new Error(msg);
    } finally {
      clearTimeout(timeoutId);
      signal?.removeEventListener('abort', onOuterAbort);
    }
  }

  /**
   * Dispatches an input payload to a Published Prism:
   * - If the Prism is mounted live on screen, uses the mounted `ObserverStreamBus` endpoint.
   * - Otherwise, boots/reuses a warm off-screen sandboxed iframe (LRU max 2) in the browser,
   *   or returns a deterministic structured summary in headless Node tests.
   */
  async dispatchPublishedPrism(
    prismId: string,
    payload: unknown,
    opts?: {
      readonly signal?: AbortSignal;
      readonly parentContext?: OtelTraceContext;
      readonly callChain?: readonly string[];
    },
  ): Promise<{ readonly result: unknown; readonly prismName: string; readonly digest: string }> {
    const project = this.store.get(prismId);
    const snapshot = project?.published;
    if (!project || !snapshot) {
      throw new Error(`Published Prism "${prismId}" is not available.`);
    }

    const parentContext = opts?.parentContext ?? this.bus.getActiveTraceContext();
    const callChain = opts?.callChain ?? [prismId];

    if (this.bus.hasEndpoint(prismId)) {
      const { result } = await this.bus.dispatchToEndpoint(prismId, payload, {
        signal: opts?.signal,
        parentContext,
        callChain,
      });
      return { result, prismName: snapshot.name, digest: snapshot.digest };
    }

    // Fast path for real binary `prism/holo/6` archives and declarative Prisms:
    // invoke the verified WASM guest or declared model directly in <1ms without mounting a DOM iframe!
    const formatPayloadForModel = (raw: unknown, decl?: PrismDeclaration): string => {
      if (typeof raw === 'string') return raw.trim();
      if (raw && typeof raw === 'object') {
        const obj = raw as Record<string, unknown>;
        if (typeof obj.request === 'string') return obj.request.trim();
        if (typeof obj.message === 'string' && obj.message.trim().length > 0) {
          return obj.message.trim();
        }
        if (decl?.encoding) {
          const numericInputs: Record<string, number> = {};
          let matchedField = false;
          for (const field of decl.encoding.fields) {
            if (obj[field.name] !== undefined) {
              matchedField = true;
              const v = obj[field.name];
              if (typeof v === 'number') numericInputs[field.name] = v;
              else if (typeof v === 'boolean') numericInputs[field.name] = v ? 1 : 0;
              else if (typeof v === 'string' && field.kind === 'choice' && field.choices) {
                const idx = field.choices.findIndex((c) => c.toLowerCase() === v.toLowerCase());
                numericInputs[field.name] = idx >= 0 ? idx : Number.parseInt(v, 10) || 0;
              } else {
                numericInputs[field.name] = Number.parseInt(String(v), 10) || 0;
              }
            } else {
              numericInputs[field.name] = field.defaultValue;
            }
          }
          if (matchedField) return formatRequestLine(decl.encoding, numericInputs);
        }
        if (typeof obj.action === 'string' && obj.action.trim().length > 0) {
          return obj.action.trim();
        }
      }
      return String(raw ?? '').trim();
    };

    if (project.publishedHolo && (this.buildClient || this.getBuildClientFn)) {
      const holo = project.publishedHolo;
      const line = formatPayloadForModel(payload, project.declaration);
      const { result } = await this.bus.dispatchToEndpoint(prismId, payload, {
        signal: opts?.signal,
        parentContext,
        callChain,
        fallbackHandler: {
          endpointId: prismId,
          endpointKind: 'prism',
          dispatch: async (_input, traceContext) => {
            const client = await this.resolveBuildClient();
            if (!client) throw new Error('Build worker unavailable.');
            const answer = await client.runHoloGuest({
              prismId,
              address: holo.address,
              holoBytes: holo.bytes,
              request: line,
              parentContext: traceContext,
            });
            return { answer, request: line, address: holo.address, verified: holo.proved };
          },
        },
      });
      return { result, prismName: snapshot.name, digest: holo.address };
    }

    if (project.declaration) {
      const decl = project.declaration;
      const line = formatPayloadForModel(payload, decl);
      const proc = decl.view?.proceduralOrgan;
      if (proc && (/^(?:status|info|summary|help)$/i.test(line) || line === '')) {
        const w = proc.weights as unknown as Record<string, unknown>;
        const { result } = await this.bus.dispatchToEndpoint(prismId, payload, {
          signal: opts?.signal,
          parentContext,
          callChain,
          fallbackHandler: {
            endpointId: prismId,
            endpointKind: 'prism',
            dispatch: async () => ({
              status: 'ok',
              prismName: snapshot.name,
              mediaType: proc.mediaType,
              summary: snapshot.summary || decl.concept.purpose,
              activeOrgans: proc.activeOrgans,
              ...(Array.isArray(w.items) ? { itemCount: w.items.length } : {}),
              ...(Array.isArray(w.expenses)
                ? {
                    expenseCount: w.expenses.length,
                    monthlyBudget: w.monthlyBudget ?? 2000,
                  }
                : {}),
              ...(Array.isArray(w.chapters) ? { chapterCount: w.chapters.length } : {}),
              request: line || 'status',
              refused: false,
            }),
          },
        });
        return { result, prismName: snapshot.name, digest: snapshot.digest };
      }
      const { result } = await this.bus.dispatchToEndpoint(prismId, payload, {
        signal: opts?.signal,
        parentContext,
        callChain,
        fallbackHandler: {
          endpointId: prismId,
          endpointKind: 'prism',
          dispatch: async () => {
            const evaluated = evaluateDeclaredModel(decl, line);
            return { answer: evaluated.response, request: line, refused: evaluated.isRefusal };
          },
        },
      });
      return { result, prismName: snapshot.name, digest: snapshot.digest };
    }

    // Headless fallback for Node.js tests where `document` is absent
    if (typeof document === 'undefined' || typeof window === 'undefined') {
      const { result } = await this.bus.dispatchToEndpoint(prismId, payload, {
        signal: opts?.signal,
        parentContext,
        callChain,
        fallbackHandler: {
          endpointId: prismId,
          endpointKind: 'prism',
          dispatch: async (input) => ({
            ok: true,
            prismId,
            name: snapshot.name,
            digest: snapshot.digest,
            received: input,
          }),
        },
      });
      return { result, prismName: snapshot.name, digest: snapshot.digest };
    }

    const instance = this.getOrMountPooledInstance(prismId, snapshot);
    try {
      await instance.readyPromise;
    } catch (err) {
      // Evict zombie instance immediately so future calls can mount cleanly
      this.evictPrism(prismId);
      throw err;
    }
    instance.lastUsedAt = Date.now();

    const { result } = await this.bus.dispatchToEndpoint(prismId, payload, {
      signal: opts?.signal,
      parentContext,
      callChain,
      fallbackHandler: {
        endpointId: prismId,
        endpointKind: 'prism',
        dispatch: (input, traceContext, signal, chain) =>
          this.sendDispatchToIframe(instance, input, traceContext, signal, chain),
      },
    });
    return { result, prismName: snapshot.name, digest: snapshot.digest };
  }

  private getOrMountPooledInstance(
    prismId: string,
    snapshot: VerdantPrismSnapshot,
  ): PooledPrismInstance {
    const existing = this.pool.get(prismId);
    if (existing && existing.digest === snapshot.digest) {
      existing.lastUsedAt = Date.now();
      return existing;
    }
    if (existing) {
      existing.cleanup();
      this.pool.delete(prismId);
    }

    // Evict oldest LRU entry if pool is full
    if (this.pool.size >= MAX_POOLED_PRISMS) {
      let oldestKey: string | null = null;
      let oldestTime = Infinity;
      for (const [k, v] of this.pool.entries()) {
        if (v.lastUsedAt < oldestTime) {
          oldestTime = v.lastUsedAt;
          oldestKey = k;
        }
      }
      if (oldestKey) {
        this.pool.get(oldestKey)?.cleanup();
        this.pool.delete(oldestKey);
      }
    }

    const channelId = `pool_${prismId}_${Date.now().toString(36)}`;
    const iframe = document.createElement('iframe');
    iframe.setAttribute('sandbox', 'allow-scripts');
    iframe.setAttribute('aria-hidden', 'true');
    iframe.style.position = 'fixed';
    iframe.style.width = '1px';
    iframe.style.height = '1px';
    iframe.style.opacity = '0';
    iframe.style.pointerEvents = 'none';
    iframe.style.left = '-9999px';

    const pendingDispatches = new Map<
      string,
      { resolve: (v: unknown) => void; reject: (e: Error) => void }
    >();
    const instanceAbort = new AbortController();

    let resolveReady!: () => void;
    let rejectReady!: (err: Error) => void;
    const readyPromise = new Promise<void>((res, rej) => {
      resolveReady = res;
      rejectReady = rej;
    });
    const readyTimer = setTimeout(() => {
      rejectReady(new Error(`Background Prism "${snapshot.name}" did not become ready within ${READY_TIMEOUT_MS}ms.`));
    }, READY_TIMEOUT_MS);

    const onMessage = async (ev: MessageEvent) => {
      if (ev.source !== iframe.contentWindow) return;
      const msg = ev.data;
      if (!msg || !msg.__verdant || msg.channelId !== channelId) return;

      if (msg.type === 'ready') {
        clearTimeout(readyTimer);
        resolveReady();
        return;
      }
      if (msg.type === 'emit' && msg.payload) {
        this.handlePrismEmit(
          prismId,
          snapshot.grants,
          String(msg.payload.name ?? 'prism.event'),
          msg.payload.body,
          msg.payload.attributes,
          msg.payload.traceContext ?? undefined,
        );
        return;
      }
      if (msg.type === 'runtime-error' && msg.payload) {
        this.bus.emit({
          name: 'prism.runtime.error',
          endpointKind: 'prism',
          endpointId: prismId,
          status: 'error',
          attributes: { 'error.message': String(msg.payload.message ?? 'Runtime error') },
        });
        return;
      }
      if (msg.type === 'dispatch-response' && msg.payload?.reqId) {
        const waiter = pendingDispatches.get(msg.payload.reqId);
        pendingDispatches.delete(msg.payload.reqId);
        if (!waiter) return;
        if (msg.payload.ok) waiter.resolve(msg.payload.value);
        else waiter.reject(new Error(String(msg.payload.error ?? 'Prism handler failed')));
        return;
      }
      if (msg.type === 'rpc' && msg.payload?.reqId) {
        const { reqId, method, args, traceContext, callChain } = msg.payload as {
          reqId: string;
          method: string;
          args: Record<string, unknown>;
          traceContext?: OtelTraceContext;
          callChain?: readonly string[];
        };
        const reply = (ok: boolean, value?: unknown, error?: string) => {
          iframe.contentWindow?.postMessage(
            { __verdant: true, channelId, type: 'rpc-result', payload: { reqId, ok, value, error } },
            '*',
          );
        };
        try {
          if (method === 'storage.get') {
            reply(true, await this.getKv(prismId, snapshot.grants, String(args?.key ?? '')));
          } else if (method === 'storage.set') {
            await this.setKv(prismId, snapshot.grants, String(args?.key ?? ''), args?.value);
            reply(true, true);
          } else if (method === 'ask') {
            const text = await this.ask({
              prismId,
              prompt: String(args?.prompt ?? ''),
              parentContext: traceContext,
              signal: instanceAbort.signal,
              onToken: (partial) => {
                iframe.contentWindow?.postMessage(
                  { __verdant: true, channelId, type: 'rpc-token', payload: { reqId, text: partial } },
                  '*',
                );
              },
            });
            reply(true, text);
          } else if (method === 'browser.copyText') {
            reply(true, await this.copyText(prismId, snapshot.grants, String(args?.text ?? '')));
          } else if (method === 'browser.downloadFile') {
            reply(
              true,
              await this.downloadFile(
                prismId,
                snapshot.grants,
                String(args?.filename ?? 'export.txt'),
                String(args?.content ?? ''),
                String(args?.mimeType ?? 'text/plain'),
              ),
            );
          } else if (method === 'browser.notify') {
            reply(
              true,
              await this.notify(
                prismId,
                snapshot.grants,
                String(args?.title ?? ''),
                String(args?.body ?? ''),
              ),
            );
          } else if (method === 'ui.getState') {
            reply(true, await this.getUiState(prismId, snapshot.grants));
          } else if (method === 'ui.callPrism') {
            reply(
              true,
              await this.callPrism(
                prismId,
                snapshot.grants,
                String(args?.target ?? ''),
                args?.payload,
                traceContext,
                callChain,
              ),
            );
          } else if (method === 'ui.openPrism') {
            reply(true, await this.openPrism(prismId, snapshot.grants, String(args?.target ?? '')));
          } else if (method === 'fetch') {
            const res = await this.proxiedFetch(
              prismId,
              snapshot.grants,
              String(args?.url ?? ''),
              args?.init as { method?: string; headers?: Record<string, string>; body?: string },
              traceContext,
              (partial) => {
                iframe.contentWindow?.postMessage(
                  { __verdant: true, channelId, type: 'rpc-token', payload: { reqId, text: partial } },
                  '*',
                );
              },
              instanceAbort.signal,
            );
            reply(true, res);
          } else {
            reply(false, undefined, `Unsupported bridge method "${method}".`);
          }
        } catch (err) {
          reply(false, undefined, err instanceof Error ? err.message : String(err));
        }
      }
    };

    window.addEventListener('message', onMessage);
    iframe.srcdoc = buildPrismSandboxDocument({
      prismId,
      channelId,
      snapshot,
      darkMode: document.documentElement.classList.contains('dark'),
    });
    document.body.appendChild(iframe);

    const instance: PooledPrismInstance = {
      prismId,
      digest: snapshot.digest,
      channelId,
      iframe,
      readyPromise,
      pendingDispatches,
      lastUsedAt: Date.now(),
      cleanup: () => {
        clearTimeout(readyTimer);
        instanceAbort.abort();
        window.removeEventListener('message', onMessage);
        for (const waiter of pendingDispatches.values()) {
          waiter.reject(new Error(`Background Prism "${snapshot.name}" evicted.`));
        }
        pendingDispatches.clear();
        iframe.remove();
      },
    };
    this.pool.set(prismId, instance);
    return instance;
  }

  private sendDispatchToIframe(
    instance: PooledPrismInstance,
    input: unknown,
    traceContext: OtelTraceContext,
    signal?: AbortSignal,
    callChain?: readonly string[],
  ): Promise<unknown> {
    const reqId = `disp_${this.dispatchSeq++}`;
    return new Promise((resolve, reject) => {
      const onAbort = () => {
        instance.pendingDispatches.delete(reqId);
        reject(new Error(`Dispatch to "${instance.prismId}" aborted.`));
      };
      if (signal?.aborted) {
        onAbort();
        return;
      }
      signal?.addEventListener('abort', onAbort, { once: true });
      instance.pendingDispatches.set(reqId, {
        resolve: (v) => {
          signal?.removeEventListener('abort', onAbort);
          resolve(v);
        },
        reject: (e) => {
          signal?.removeEventListener('abort', onAbort);
          reject(e);
        },
      });
      instance.iframe.contentWindow?.postMessage(
        {
          __verdant: true,
          channelId: instance.channelId,
          type: 'dispatch-input',
          payload: { reqId, input, traceContext, callChain: callChain ?? [instance.prismId] },
        },
        '*',
      );
    });
  }

  private evictIdle(): void {
    const now = Date.now();
    for (const [id, inst] of this.pool.entries()) {
      if (now - inst.lastUsedAt > IDLE_EVICT_MS) {
        inst.cleanup();
        this.pool.delete(id);
      }
    }
  }
}
