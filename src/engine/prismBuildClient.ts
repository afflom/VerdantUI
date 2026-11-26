/**
 * Main-Thread Client for the Dedicated Prism WASM Build & Verification Worker.
 *
 * Lazily spawns `prismBuild.worker.ts` only when Studio checks, proves, builds, verifies,
 * or executes a real binary `prism/holo/6` archive, emitting causal OpenTelemetry spans
 * on `ObserverStreamBus`.
 */

import type { ObserverStreamBus, OtelTraceContext } from './observerStreamBus';
import type {
  PrismBuildStage,
  PrismWorkerRequest,
  PrismWorkerResponse,
  WorkerBuildPayload,
  WorkerVerifyPayload,
} from './prismBuild.worker';

export type { PrismBuildStage, WorkerBuildPayload, WorkerVerifyPayload };

type PrismWorkerPayload =
  PrismWorkerRequest extends infer U
    ? U extends PrismWorkerRequest
      ? Omit<U, 'id'>
      : never
    : never;

interface PendingCall {
  resolve: (value: unknown) => void;
  reject: (err: Error) => void;
  onStage?: (stage: PrismBuildStage, detail: string) => void;
}

export class PrismBuildWorkerClient {
  private worker: Worker | null = null;
  private readonly pending = new Map<string, PendingCall>();
  private seq = 1;

  constructor(private readonly bus: ObserverStreamBus) {}

  private ensureWorker(): Worker {
    if (this.worker) return this.worker;
    const worker = new Worker(new URL('./prismBuild.worker.ts', import.meta.url), {
      type: 'module',
    });
    worker.onmessage = (event: MessageEvent<PrismWorkerResponse>) => {
      const msg = event.data;
      if (!msg || !msg.id) return;
      const waiter = this.pending.get(msg.id);
      if (!waiter) return;
      if (msg.event === 'stage') {
        waiter.onStage?.(msg.stage, msg.detail);
        return;
      }
      this.pending.delete(msg.id);
      if (msg.event === 'error') {
        waiter.reject(new Error(msg.message));
      } else if (msg.event === 'build-result') {
        waiter.resolve(msg.result);
      } else if (msg.event === 'verify-result') {
        waiter.resolve(msg.result);
      } else if (msg.event === 'run-result') {
        waiter.resolve(msg.answer);
      }
    };
    worker.onerror = (err) => {
      const message = err.message || 'Prism build worker crashed.';
      for (const [, waiter] of this.pending) {
        waiter.reject(new Error(message));
      }
      this.pending.clear();
      this.worker?.terminate();
      this.worker = null;
    };
    this.worker = worker;
    return worker;
  }

  private call<T>(
    req: PrismWorkerPayload,
    onStage?: (stage: PrismBuildStage, detail: string) => void,
  ): Promise<T> {
    const id = `pw_${this.seq++}`;
    const worker = this.ensureWorker();
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, {
        resolve: resolve as (v: unknown) => void,
        reject,
        onStage,
      });
      worker.postMessage({ ...req, id });
    });
  }

  private resolveBaseUrl(): string {
    if (typeof document !== 'undefined' && document.baseURI) {
      return new URL('./', document.baseURI).href;
    }
    if (typeof location !== 'undefined') {
      return new URL('./', location.href).href;
    }
    return 'http://localhost/';
  }

  async buildAndSeal(opts: {
    readonly prismId: string;
    readonly slug: string;
    readonly appFiles: Readonly<Record<string, string>>;
    readonly onStage?: (stage: PrismBuildStage, detail: string) => void;
    readonly parentContext?: OtelTraceContext;
  }): Promise<WorkerBuildPayload> {
    const span = this.bus.startSpan(`studio.build:${opts.slug}`, {
      endpointKind: 'studio',
      endpointId: opts.prismId,
      kind: 'INTERNAL',
      parentContext: opts.parentContext,
    });
    try {
      const result = await this.call<WorkerBuildPayload>(
        {
          type: 'build',
          baseUrl: this.resolveBaseUrl(),
          slug: opts.slug,
          appFiles: opts.appFiles,
        },
        (stage, detail) => {
          span.addEvent(`studio.build.${stage}`, { detail });
          opts.onStage?.(stage, detail);
        },
      );
      span.setAttributes({
        'verdant.prism.ok': result.ok,
        'verdant.prism.proved': result.proved,
        'verdant.prism.vectors_passed': result.vectorsPassed,
        ...(result.holo ? { 'verdant.prism.address': result.holo.address } : {}),
      });
      span.end(result.ok ? 'ok' : 'error', result.ok ? undefined : result.message);
      return result;
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      span.end('error', msg);
      throw error;
    }
  }

  async verifyArchive(opts: {
    readonly prismId?: string;
    readonly holoBytes: Uint8Array;
    readonly onStage?: (stage: PrismBuildStage, detail: string) => void;
    readonly parentContext?: OtelTraceContext;
  }): Promise<WorkerVerifyPayload> {
    const span = this.bus.startSpan('studio.verify.holo', {
      endpointKind: 'studio',
      endpointId: opts.prismId ?? 'studio',
      kind: 'INTERNAL',
      parentContext: opts.parentContext,
    });
    try {
      const result = await this.call<WorkerVerifyPayload>(
        { type: 'verify', holoBytes: opts.holoBytes },
        opts.onStage,
      );
      span.setAttributes({
        'verdant.prism.address': result.address,
        'verdant.prism.verified': result.ok,
        'verdant.prism.vectors': `${result.vectorsPassed}/${result.vectorsTotal}`,
      });
      span.end(result.ok ? 'ok' : 'error', result.ok ? undefined : 'Archive verification failed');
      return result;
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      span.end('error', msg);
      throw error;
    }
  }

  async runHoloGuest(opts: {
    readonly prismId: string;
    readonly address: string;
    readonly holoBytes: Uint8Array;
    readonly request: string;
    readonly parentContext?: OtelTraceContext;
  }): Promise<string> {
    const span = this.bus.startSpan(`prism.wasm.invoke:${opts.prismId}`, {
      endpointKind: 'prism',
      endpointId: opts.prismId,
      kind: 'INTERNAL',
      parentContext: opts.parentContext,
      attributes: {
        'verdant.prism.address': opts.address,
        'verdant.input.bytes': opts.request.length,
      },
    });
    try {
      const answer = await this.call<string>({
        type: 'run',
        address: opts.address,
        holoBytes: opts.holoBytes,
        request: opts.request,
      });
      span.setAttributes({ 'verdant.output.bytes': answer.length });
      span.end('ok');
      return answer;
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      span.end('error', msg);
      throw error;
    }
  }

  dispose(): void {
    for (const [, waiter] of this.pending) {
      waiter.reject(new Error('Prism build worker disposed.'));
    }
    this.pending.clear();
    this.worker?.terminate();
    this.worker = null;
  }
}
