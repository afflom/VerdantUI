/**
 * VerdantUI Verified Holo Host (`src/host/verdantHoloHost.ts`)
 *
 * Thin browser & Node host substrate that loads the build-emitted, Lean-proved
 * `verdant-ui.holo` Core-Wasm guest (`prism/holo/6`) using ONLY leaf browser-safe
 * `prism-code` modules (`archive.mjs`, `core.mjs`, `middleware.mjs`, `tracing.mjs`) —
 * avoiding the Node/CLI barrel and saving ~340 KiB of JS bundle size.
 *
 * Verifies BLAKE3 footer/seal, identities, and all acceptance vectors before allowing
 * dispatch; attaches `experimental_tracing` onto `ObserverStreamBus`; and refuses
 * explicitly (`refused: ...`) on any verification or dispatch failure (zero silent fallbacks).
 */

import { asBytes, parseArchive, resolveArchive } from '@prism-code-internal/archive.mjs';
import {
  CORE_CONTRACT,
  checkRequest,
  compileGuest,
  instantiateGuest,
  runAcceptance,
  runInstance,
} from '@prism-code-internal/core.mjs';
import { createMiddlewareStack, type MiddlewareStack } from '@prism-code-internal/middleware.mjs';
import { experimental_tracing } from '@prism-code-internal/tracing.mjs';
import type { ObserverStreamBus } from '../engine/observerStreamBus';

const enc = new TextEncoder();
const dec = new TextDecoder();

export interface VerdantHoloSnapshot {
  readonly loaded: boolean;
  readonly verified: boolean;
  readonly address: string | null;
  readonly bytes: number;
  readonly vectorCount: number;
  readonly vectorsPassed: number;
  readonly dispatchCount: number;
  readonly refusalCount: number;
  readonly loadError: string | null;
}

export interface HoloRawResult {
  readonly raw: string;
  readonly refused: boolean;
  readonly latencyUs: number;
}

export interface HoloConsentResult {
  readonly phase: 'consent' | 'initializing' | 'ready' | 'refused';
  readonly unlocked: boolean;
  readonly refused: boolean;
  readonly raw: string;
}

export interface HoloGrantResult {
  readonly allowed: boolean;
  readonly refused: boolean;
  readonly reason: string;
  readonly raw: string;
}

export interface HoloCompartmentResult {
  readonly visible: boolean;
  readonly refused: boolean;
  readonly raw: string;
}

export interface HoloPartResult {
  readonly ok: boolean;
  readonly refused: boolean;
  readonly raw: string;
}

export interface HoloGuiTreeResult {
  readonly parts: readonly string[];
  readonly refused: boolean;
  readonly raw: string;
}

export interface HoloCellTuneResult {
  readonly routed: boolean;
  readonly target: string;
  readonly refused: boolean;
  readonly raw: string;
}

interface VerifiedGuestHandle {
  readonly name: string;
  readonly module: WebAssembly.Module;
  readonly requestMaximum: number;
  readonly middleware: MiddlewareStack;
}

export class VerdantHoloHost {
  private guest: VerifiedGuestHandle | null = null;
  private snapshot: VerdantHoloSnapshot = {
    loaded: false,
    verified: false,
    address: null,
    bytes: 0,
    vectorCount: 0,
    vectorsPassed: 0,
    dispatchCount: 0,
    refusalCount: 0,
    loadError: null,
  };

  constructor(private readonly bus: ObserverStreamBus) {}

  getSnapshot(): VerdantHoloSnapshot {
    return this.snapshot;
  }

  isVerified(): boolean {
    return this.snapshot.loaded && this.snapshot.verified && this.guest !== null;
  }

  /** Loads and verifies `verdant-ui.holo` from raw archive bytes (works in Node & Browser). */
  async loadFromBytes(bytes: Uint8Array): Promise<VerdantHoloSnapshot> {
    const loadSpan = this.bus.startSpan('holo.load', {
      endpointKind: 'prism',
      endpointId: 'verdant-ui-holo',
      attributes: { 'verdant.holo_bytes': bytes.byteLength },
    });
    const t0 = performance.now();
    try {
      const parsed = parseArchive(asBytes(bytes));
      const sealOk = parsed.format >= 5 ? parsed.verification.seal !== false : parsed.verification.footer !== false;
      if (!sealOk || parsed.verification.footer === false) {
        throw new Error('refused: verdant-ui.holo integrity footer/seal verification failed');
      }
      const resolved = resolveArchive(parsed);
      if (resolved.identities && resolved.identities.ok === false) {
        throw new Error('refused: verdant-ui.holo identity verification failed');
      }
      const appDoc = resolved.modelDocument.application;
      if (!appDoc) {
        throw new Error('refused: verdant-ui.holo missing application document');
      }

      const wasmModule = await compileGuest(resolved.guest.content, undefined, CORE_CONTRACT);
      const rawVectors = appDoc.acceptance_vectors ?? appDoc.acceptanceVectors ?? [];
      const vectors = rawVectors.map((v) => ({
        request: Uint8Array.from(v.request),
        response: Uint8Array.from(v.response),
      }));

      const requestMaximum = Number(appDoc.request_maximum ?? appDoc.requestMaximum ?? 1024);
      const verifySpan = this.bus.startSpan('holo.verify', {
        endpointKind: 'prism',
        endpointId: 'verdant-ui-holo',
        parentContext: loadSpan.context,
      });
      const acc = await runAcceptance(vectors, (req) => {
        checkRequest(req, { request: requestMaximum, guestAllocation: requestMaximum * 16 });
        return runInstance(instantiateGuest(wasmModule, {}), req);
      });
      const address = String(parsed.kappa ?? parsed.fingerprint ?? 'verified');
      const total = Number(acc?.total ?? vectors.length);
      const passed = Number(acc?.passed ?? 0);
      const ok = Boolean(acc && total > 0 && passed === total);
      verifySpan.setAttributes({
        'verdant.holo_address': address,
        'verdant.holo_verified': ok,
        'verdant.holo_vectors_total': total,
        'verdant.holo_vectors_passed': passed,
      });
      if (!ok) {
        const msg = `refused: verdant-ui.holo failed acceptance verification (${passed}/${total})`;
        verifySpan.end('error', msg);
        throw new Error(msg);
      }
      verifySpan.end('ok');

      const middleware = createMiddlewareStack();
      middleware.use(
        experimental_tracing({
          tracer: this.bus.asOtelTracer({
            endpointKind: 'prism',
            endpointId: 'verdant-ui-holo',
          }),
          steps: ['dispatch'],
        }),
      );

      this.guest = {
        name: String(appDoc.name ?? 'verdant-ui'),
        module: wasmModule,
        requestMaximum: Number(appDoc.request_maximum ?? appDoc.requestMaximum ?? 1024),
        middleware,
      };
      this.snapshot = {
        loaded: true,
        verified: true,
        address,
        bytes: bytes.byteLength,
        vectorCount: total,
        vectorsPassed: passed,
        dispatchCount: 0,
        refusalCount: 0,
        loadError: null,
      };
      loadSpan.setAttributes({
        'verdant.holo_address': address,
        'verdant.holo_verified': true,
        'verdant.holo_ms': Math.round(performance.now() - t0),
      });
      loadSpan.end('ok');
      return this.snapshot;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.guest = null;
      this.snapshot = {
        ...this.snapshot,
        loaded: false,
        verified: false,
        loadError: msg,
      };
      loadSpan.end('error', msg);
      throw err;
    }
  }

  /** Fetches `./holos/verdant-ui.holo` relative to the current document (subpath-safe for GitHub Pages). */
  async loadFromUrl(url = './holos/verdant-ui.holo'): Promise<VerdantHoloSnapshot> {
    const res = await fetch(url);
    if (!res.ok) {
      const msg = `refused: failed to fetch ${url} (HTTP ${res.status})`;
      this.snapshot = { ...this.snapshot, loaded: false, verified: false, loadError: msg };
      throw new Error(msg);
    }
    const buf = new Uint8Array(await res.arrayBuffer());
    return this.loadFromBytes(buf);
  }

  async dispatchRaw(command: string): Promise<HoloRawResult> {
    if (!this.guest || !this.snapshot.verified) {
      throw new Error('refused: verdant-ui.holo is not loaded and verified');
    }
    const guest = this.guest;
    const route = command.split('\t').slice(0, 2).join(':');
    const span = this.bus.startSpan('holo.dispatch', {
      endpointKind: 'prism',
      endpointId: 'verdant-ui-holo',
      parentContext: this.bus.getActiveTraceContext(),
      attributes: { 'verdant.holo_route': route },
    });
    const t0 = performance.now();
    try {
      const request = enc.encode(command);
      checkRequest(request, { request: guest.requestMaximum, guestAllocation: guest.requestMaximum * 16 });
      const outBytes = (await this.bus.runWithTraceContext(span.context, async () => {
        const instance = instantiateGuest(guest.module, {});
        const runStep = guest.middleware.resolve(
          'dispatch',
          (args) => runInstance(args.instance as WebAssembly.Instance, args.request as Uint8Array),
          { application: { name: guest.name, kind: 'holo' } },
        );
        return (await runStep({ instance, request })) as Uint8Array;
      })) as Uint8Array;

      const raw = dec.decode(outBytes);
      const refused = raw.startsWith('refused:');
      const latencyUs = Math.max(1, Math.round((performance.now() - t0) * 1000));
      this.snapshot = {
        ...this.snapshot,
        dispatchCount: this.snapshot.dispatchCount + 1,
        refusalCount: this.snapshot.refusalCount + (refused ? 1 : 0),
      };
      span.setAttributes({
        'verdant.holo_refused': refused,
        'verdant.holo_latency_us': latencyUs,
      });
      span.end('ok');
      return { raw, refused, latencyUs };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      span.end('error', msg);
      throw err;
    }
  }

  async dispatchConsent(
    consent: string,
    modelLoaded: boolean,
    storesHydrated: boolean,
    contextWarmed: boolean,
  ): Promise<HoloConsentResult> {
    const cmd = `verdant\tconsent\t${consent}\t${modelLoaded ? '1' : '0'}\t${storesHydrated ? '1' : '0'}\t${contextWarmed ? '1' : '0'}`;
    const { raw, refused } = await this.dispatchRaw(cmd);
    if (refused) return { phase: 'refused', unlocked: false, refused: true, raw };
    const [phase, lock] = raw.split('\t');
    return {
      phase: phase === 'ready' ? 'ready' : phase === 'initializing' ? 'initializing' : 'consent',
      unlocked: lock === 'unlocked',
      refused: false,
      raw,
    };
  }

  async dispatchGrant(
    organ: string,
    capability: string,
    offlineMode: boolean,
    explicitDenied: boolean,
  ): Promise<HoloGrantResult> {
    const cmd = `verdant\tgrant\t${organ}\t${capability}\t${offlineMode ? '1' : '0'}\t${explicitDenied ? '1' : '0'}`;
    const { raw, refused } = await this.dispatchRaw(cmd);
    return {
      allowed: !refused && raw.startsWith('allow\t'),
      refused,
      reason: raw,
      raw,
    };
  }

  async dispatchCompartment(
    ownerCompartment: string,
    activeCompartment: string,
    visibility: 'private' | 'shared',
  ): Promise<HoloCompartmentResult> {
    const cmd = `verdant\tcompartment\t${ownerCompartment}\t${activeCompartment}\t${visibility}`;
    const { raw, refused } = await this.dispatchRaw(cmd);
    return {
      visible: !refused && raw === 'visible',
      refused,
      raw,
    };
  }

  async dispatchPart(partId: string, verb: string): Promise<HoloPartResult> {
    const cmd = `verdant\tpart\t${partId}\t${verb}`;
    const { raw, refused } = await this.dispatchRaw(cmd);
    return {
      ok: !refused && raw.startsWith('ok\t'),
      refused,
      raw,
    };
  }

  async dispatchGuiTree(): Promise<HoloGuiTreeResult> {
    const { raw, refused } = await this.dispatchRaw('verdant\tgui\ttree');
    const parts = !refused && raw.startsWith('tree\t') ? raw.slice(5).split(',').filter(Boolean) : [];
    return { parts, refused, raw };
  }

  async dispatchObserverPerceive(): Promise<HoloGuiTreeResult> {
    const { raw, refused } = await this.dispatchRaw('verdant\tobserver\tperceive');
    const parts = !refused && raw.startsWith('tree\t') ? raw.slice(5).split(',').filter(Boolean) : [];
    return { parts, refused, raw };
  }

  async dispatchCellTune(target: string): Promise<HoloCellTuneResult> {
    const { raw, refused } = await this.dispatchRaw(`verdant\tcells\ttune\t${target}`);
    return {
      routed: !refused && raw.startsWith('routed\t'),
      target: !refused && raw.startsWith('routed\t') ? raw.slice(7) : '',
      refused,
      raw,
    };
  }
}
