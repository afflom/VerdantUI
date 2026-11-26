/**
 * Lean OpenTelemetry (OTLP/JSON + W3C Trace Context) Observer Sensory & Output Stream Bus.
 *
 * Unifies every sensory input (Threads, Verdant Studio, Published Prisms, Browser runtime,
 * and granted Internet streams) and cognitive output into one causal trace model:
 * - W3C Trace Context (`traceId` 128-bit hex, `spanId` 64-bit hex, `parentSpanId`, `traceparent`)
 * - Lean OTel `Span` and `Event` records with bounded ring buffers (zero ceremonial SDK bloat)
 * - Standard `toOtlpJson()` projection (`resourceSpans` + `resourceLogs`) for inspection/export
 * - Typed request/response endpoint dispatch for routing cognitive outputs to live/pooled Prisms
 */

export type StreamEndpointKind =
  | 'observer'
  | 'thread'
  | 'studio'
  | 'prism'
  | 'browser'
  | 'network';

export interface OtelTraceContext {
  readonly traceId: string;
  readonly spanId: string;
  readonly parentSpanId?: string;
  readonly traceFlags: number;
}

export type OtelSpanKind = 'INTERNAL' | 'CLIENT' | 'SERVER' | 'PRODUCER' | 'CONSUMER';
export type OtelStatusCode = 'unset' | 'ok' | 'error';
export type OtelAttributes = Readonly<Record<string, string | number | boolean | null | undefined>>;

export interface OtelEvent {
  readonly id: string;
  readonly name: string;
  readonly timestampMs: number;
  readonly traceId: string;
  readonly spanId: string;
  readonly endpointKind: StreamEndpointKind;
  readonly endpointId: string;
  readonly status: 'ok' | 'error';
  readonly attributes: OtelAttributes;
  readonly body?: unknown;
}

export interface OtelSpan {
  readonly traceId: string;
  readonly spanId: string;
  readonly parentSpanId?: string;
  readonly name: string;
  readonly kind: OtelSpanKind;
  readonly endpointKind: StreamEndpointKind;
  readonly endpointId: string;
  readonly startTimeMs: number;
  readonly endTimeMs?: number;
  readonly durationMs?: number;
  readonly status: OtelStatusCode;
  readonly errorMessage?: string;
  readonly attributes: Record<string, string | number | boolean>;
  readonly events: readonly OtelEvent[];
}

export interface OtelSpanHandle {
  readonly context: OtelTraceContext;
  addEvent(name: string, attributes?: OtelAttributes, body?: unknown, status?: 'ok' | 'error'): OtelEvent;
  setAttributes(attrs: OtelAttributes): void;
  end(status?: OtelStatusCode, errorMessage?: string): OtelSpan;
}

export interface StreamEndpointHandler {
  readonly endpointId: string;
  readonly endpointKind: StreamEndpointKind;
  readonly priority?: number;
  dispatch(
    payload: unknown,
    context: OtelTraceContext,
    signal?: AbortSignal,
    callChain?: readonly string[],
  ): Promise<unknown>;
}

export interface StreamFilter {
  readonly endpointKind?: StreamEndpointKind;
  readonly endpointId?: string;
  readonly namePrefix?: string;
}

export type StreamListener = (event: OtelEvent) => void;
export type SpanListener = (span: OtelSpan) => void;

const MAX_EVENTS = 128;
const MAX_SPANS = 64;

let fallbackCounter = 1;

function randomHex(byteCount: number): string {
  const bytes = new Uint8Array(byteCount);
  if (typeof globalThis.crypto !== 'undefined' && typeof globalThis.crypto.getRandomValues === 'function') {
    globalThis.crypto.getRandomValues(bytes);
  } else {
    let seed = (Date.now() ^ (fallbackCounter++ * 0x9e3779b1)) >>> 0;
    for (let i = 0; i < byteCount; i++) {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      bytes[i] = (seed >>> 24) & 0xff;
    }
  }
  let hex = '';
  let allZero = true;
  for (let i = 0; i < bytes.length; i++) {
    if (bytes[i] !== 0) allZero = false;
    hex += bytes[i].toString(16).padStart(2, '0');
  }
  if (allZero) {
    return `${hex.slice(0, -1)}1`;
  }
  return hex;
}

export function createTraceId(): string {
  return randomHex(16); // 32 hex chars (128-bit W3C Trace Context)
}

export function createSpanId(): string {
  return randomHex(8); // 16 hex chars (64-bit W3C Trace Context)
}

export function formatTraceparent(ctx: OtelTraceContext): string {
  const flags = (ctx.traceFlags & 0xff).toString(16).padStart(2, '0');
  return `00-${ctx.traceId}-${ctx.spanId}-${flags}`;
}

export function createTraceContext(parent?: OtelTraceContext): OtelTraceContext {
  return {
    traceId: parent?.traceId ?? createTraceId(),
    spanId: createSpanId(),
    parentSpanId: parent?.spanId,
    traceFlags: parent?.traceFlags ?? 1,
  };
}

function sanitizeAttributes(attrs?: OtelAttributes): Record<string, string | number | boolean> {
  const out: Record<string, string | number | boolean> = {};
  if (!attrs) return out;
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null) continue;
    if (typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') {
      out[k] = v;
    }
  }
  return out;
}

function toOtlpAttributes(
  attrs: Readonly<Record<string, string | number | boolean | null | undefined>>,
): Array<{ key: string; value: Record<string, unknown> }> {
  const out: Array<{ key: string; value: Record<string, unknown> }> = [];
  for (const [key, val] of Object.entries(attrs)) {
    if (val === undefined || val === null) continue;
    if (typeof val === 'string') {
      out.push({ key, value: { stringValue: val } });
    } else if (typeof val === 'boolean') {
      out.push({ key, value: { boolValue: val } });
    } else if (typeof val === 'number' && Number.isFinite(val)) {
      out.push({
        key,
        value: Number.isInteger(val) ? { intValue: String(val) } : { doubleValue: val },
      });
    }
  }
  return out;
}

function msToUnixNano(ms: number): string {
  return `${Math.max(0, Math.round(ms))}000000`;
}

function spanKindCode(kind: OtelSpanKind): number {
  switch (kind) {
    case 'INTERNAL':
      return 1;
    case 'SERVER':
      return 2;
    case 'CLIENT':
      return 3;
    case 'PRODUCER':
      return 4;
    case 'CONSUMER':
      return 5;
  }
}

export class ObserverStreamBus {
  private readonly events: OtelEvent[] = [];
  private readonly spans: OtelSpan[] = [];
  private readonly listeners = new Set<{ filter?: StreamFilter; fn: StreamListener }>();
  private readonly spanListeners = new Set<SpanListener>();
  private readonly endpoints = new Map<string, StreamEndpointHandler[]>();
  private activeTraceContext: OtelTraceContext | undefined;
  private eventSeq = 1;

  setActiveTraceContext(ctx: OtelTraceContext | undefined): void {
    this.activeTraceContext = ctx;
  }

  getActiveTraceContext(): OtelTraceContext | undefined {
    return this.activeTraceContext;
  }

  async runWithTraceContext<T>(ctx: OtelTraceContext | undefined, fn: () => Promise<T>): Promise<T> {
    const prev = this.activeTraceContext;
    this.activeTraceContext = ctx;
    try {
      return await fn();
    } finally {
      this.activeTraceContext = prev;
    }
  }

  emit(input: {
    readonly name: string;
    readonly endpointKind: StreamEndpointKind;
    readonly endpointId: string;
    readonly traceContext?: OtelTraceContext;
    readonly status?: 'ok' | 'error';
    readonly attributes?: OtelAttributes;
    readonly body?: unknown;
  }): OtelEvent {
    const ctx = input.traceContext ?? createTraceContext();
    const event: OtelEvent = {
      id: `evt_${this.eventSeq++}`,
      name: input.name,
      timestampMs: Date.now(),
      traceId: ctx.traceId,
      spanId: ctx.spanId,
      endpointKind: input.endpointKind,
      endpointId: input.endpointId,
      status: input.status ?? 'ok',
      attributes: sanitizeAttributes(input.attributes),
      body: input.body,
    };

    this.events.push(event);
    if (this.events.length > MAX_EVENTS) {
      this.events.shift();
    }

    for (const sub of this.listeners) {
      if (sub.filter) {
        if (sub.filter.endpointKind && sub.filter.endpointKind !== event.endpointKind) continue;
        if (sub.filter.endpointId && sub.filter.endpointId !== event.endpointId) continue;
        if (sub.filter.namePrefix && !event.name.startsWith(sub.filter.namePrefix)) continue;
      }
      try {
        sub.fn(event);
      } catch {
        // Isolate subscriber errors from the stream bus
      }
    }

    return event;
  }

  startSpan(
    name: string,
    opts: {
      readonly endpointKind: StreamEndpointKind;
      readonly endpointId: string;
      readonly kind?: OtelSpanKind;
      readonly parentContext?: OtelTraceContext;
      readonly attributes?: OtelAttributes;
    },
  ): OtelSpanHandle {
    const context = createTraceContext(opts.parentContext);
    const startTimeMs = Date.now();
    const mutableAttrs = sanitizeAttributes(opts.attributes);
    const spanEvents: OtelEvent[] = [];
    let ended = false;

    const recordSpan = (span: OtelSpan) => {
      const idx = this.spans.findIndex((s) => s.spanId === span.spanId);
      if (idx >= 0) {
        this.spans[idx] = span;
      } else {
        this.spans.push(span);
        if (this.spans.length > MAX_SPANS) {
          this.spans.shift();
        }
      }
      for (const listener of this.spanListeners) {
        try {
          listener(span);
        } catch {
          // Isolate subscriber errors
        }
      }
    };

    const initialSpan: OtelSpan = {
      traceId: context.traceId,
      spanId: context.spanId,
      parentSpanId: context.parentSpanId,
      name,
      kind: opts.kind ?? 'INTERNAL',
      endpointKind: opts.endpointKind,
      endpointId: opts.endpointId,
      startTimeMs,
      status: 'unset',
      attributes: { ...mutableAttrs },
      events: [],
    };
    recordSpan(initialSpan);

    return {
      context,
      addEvent: (eventName, attributes, body, status = 'ok') => {
        const ev = this.emit({
          name: eventName,
          endpointKind: opts.endpointKind,
          endpointId: opts.endpointId,
          traceContext: context,
          status,
          attributes,
          body,
        });
        spanEvents.push(ev);
        return ev;
      },
      setAttributes: (attrs) => {
        Object.assign(mutableAttrs, sanitizeAttributes(attrs));
      },
      end: (status = 'ok', errorMessage) => {
        const endTimeMs = Date.now();
        const finalSpan: OtelSpan = {
          traceId: context.traceId,
          spanId: context.spanId,
          parentSpanId: context.parentSpanId,
          name,
          kind: opts.kind ?? 'INTERNAL',
          endpointKind: opts.endpointKind,
          endpointId: opts.endpointId,
          startTimeMs,
          endTimeMs,
          durationMs: Math.max(0, endTimeMs - startTimeMs),
          status,
          ...(errorMessage ? { errorMessage } : {}),
          attributes: { ...mutableAttrs },
          events: [...spanEvents],
        };
        if (!ended) {
          ended = true;
          recordSpan(finalSpan);
        }
        return finalSpan;
      },
    };
  }

  subscribe(listener: StreamListener, filter?: StreamFilter): () => void {
    const entry = { filter, fn: listener };
    this.listeners.add(entry);
    return () => {
      this.listeners.delete(entry);
    };
  }

  subscribeSpans(listener: SpanListener): () => void {
    this.spanListeners.add(listener);
    return () => {
      this.spanListeners.delete(listener);
    };
  }

  getRecentEvents(filter?: StreamFilter): readonly OtelEvent[] {
    if (!filter) return [...this.events];
    return this.events.filter((event) => {
      if (filter.endpointKind && filter.endpointKind !== event.endpointKind) return false;
      if (filter.endpointId && filter.endpointId !== event.endpointId) return false;
      if (filter.namePrefix && !event.name.startsWith(filter.namePrefix)) return false;
      return true;
    });
  }

  getRecentSpans(endpointId?: string): readonly OtelSpan[] {
    if (!endpointId) return [...this.spans];
    return this.spans.filter((s) => s.endpointId === endpointId);
  }

  registerEndpoint(handler: StreamEndpointHandler): () => void {
    const list = this.endpoints.get(handler.endpointId) ?? [];
    list.push(handler);
    list.sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0));
    this.endpoints.set(handler.endpointId, list);
    return () => {
      const curr = this.endpoints.get(handler.endpointId);
      if (!curr) return;
      const next = curr.filter((h) => h !== handler);
      if (next.length === 0) {
        this.endpoints.delete(handler.endpointId);
      } else {
        this.endpoints.set(handler.endpointId, next);
      }
    };
  }

  hasEndpoint(endpointId: string): boolean {
    const list = this.endpoints.get(endpointId);
    return Boolean(list && list.length > 0);
  }

  async dispatchToEndpoint(
    endpointId: string,
    payload: unknown,
    opts?: {
      readonly signal?: AbortSignal;
      readonly parentContext?: OtelTraceContext;
      readonly timeoutMs?: number;
      readonly callChain?: readonly string[];
      readonly fallbackHandler?: StreamEndpointHandler;
    },
  ): Promise<{ readonly result: unknown; readonly span: OtelSpan }> {
    const registered = this.endpoints.get(endpointId)?.[0] ?? opts?.fallbackHandler;
    const parentContext = opts?.parentContext ?? this.activeTraceContext;
    const spanHandle = this.startSpan(`endpoint.dispatch:${endpointId}`, {
      endpointKind: registered?.endpointKind ?? 'prism',
      endpointId,
      kind: 'CLIENT',
      parentContext,
      attributes: {
        'verdant.endpoint.id': endpointId,
      },
    });

    if (!registered) {
      const errMsg = `No active stream endpoint registered for "${endpointId}".`;
      spanHandle.addEvent('endpoint.missing', { 'error.message': errMsg }, undefined, 'error');
      spanHandle.end('error', errMsg);
      throw new Error(errMsg);
    }

    const timeoutMs = opts?.timeoutMs ?? 15_000;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const abortCtrl = new AbortController();

    const onOuterAbort = () => abortCtrl.abort();
    if (opts?.signal) {
      if (opts.signal.aborted) {
        abortCtrl.abort();
      } else {
        opts.signal.addEventListener('abort', onOuterAbort, { once: true });
      }
    }

    try {
      const dispatchPromise = registered.dispatch(
        payload,
        spanHandle.context,
        abortCtrl.signal,
        opts?.callChain,
      );
      const result = await new Promise<unknown>((resolve, reject) => {
        if (abortCtrl.signal.aborted) {
          reject(new Error(`Stream dispatch to "${endpointId}" aborted.`));
          return;
        }
        timer = setTimeout(() => {
          abortCtrl.abort();
          reject(new Error(`Stream dispatch to "${endpointId}" timed out after ${timeoutMs}ms.`));
        }, timeoutMs);
        abortCtrl.signal.addEventListener(
          'abort',
          () => reject(new Error(`Stream dispatch to "${endpointId}" aborted.`)),
          { once: true },
        );
        dispatchPromise.then(resolve, reject);
      });

      spanHandle.addEvent('endpoint.response', { 'verdant.response.ok': true }, result, 'ok');
      const finalSpan = spanHandle.end('ok');
      return { result, span: finalSpan };
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      spanHandle.addEvent('endpoint.error', { 'error.message': errMsg }, undefined, 'error');
      spanHandle.end('error', errMsg);
      throw err;
    } finally {
      if (timer !== null) clearTimeout(timer);
      opts?.signal?.removeEventListener('abort', onOuterAbort);
    }
  }

  /** Exports recent spans and events in standard OTLP/JSON 1.x format. */
  toOtlpJson(endpointId?: string): {
    readonly resourceSpans: readonly unknown[];
    readonly resourceLogs: readonly unknown[];
  } {
    const resource = {
      attributes: toOtlpAttributes({
        'service.name': 'verdant-ui',
        'service.version': '0.1.0',
      }),
    };
    const scope = { name: 'verdant-ui.observer-bus', version: '1.0.0' };
    const spans = this.getRecentSpans(endpointId).map((s) => ({
      traceId: s.traceId,
      spanId: s.spanId,
      ...(s.parentSpanId ? { parentSpanId: s.parentSpanId } : {}),
      name: s.name,
      kind: spanKindCode(s.kind),
      startTimeUnixNano: msToUnixNano(s.startTimeMs),
      endTimeUnixNano: msToUnixNano(s.endTimeMs ?? s.startTimeMs),
      attributes: toOtlpAttributes({
        'verdant.endpoint.kind': s.endpointKind,
        'verdant.endpoint.id': s.endpointId,
        ...s.attributes,
      }),
      events: s.events.map((ev) => ({
        timeUnixNano: msToUnixNano(ev.timestampMs),
        name: ev.name,
        attributes: toOtlpAttributes(ev.attributes),
      })),
      status: {
        code: s.status === 'ok' ? 1 : s.status === 'error' ? 2 : 0,
        ...(s.errorMessage ? { message: s.errorMessage } : {}),
      },
    }));

    const logRecords = this.getRecentEvents(endpointId ? { endpointId } : undefined).map((ev) => ({
      timeUnixNano: msToUnixNano(ev.timestampMs),
      severityNumber: ev.status === 'error' ? 17 : 9,
      severityText: ev.status === 'error' ? 'ERROR' : 'INFO',
      eventName: ev.name,
      traceId: ev.traceId,
      spanId: ev.spanId,
      body:
        ev.body !== undefined
          ? { stringValue: typeof ev.body === 'string' ? ev.body : JSON.stringify(ev.body) }
          : undefined,
      attributes: toOtlpAttributes({
        'verdant.endpoint.kind': ev.endpointKind,
        'verdant.endpoint.id': ev.endpointId,
        ...ev.attributes,
      }),
    }));

    return {
      resourceSpans: [{ resource, scopeSpans: [{ scope, spans }] }],
      resourceLogs: [{ resource, scopeLogs: [{ scope, logRecords }] }],
    };
  }
}
