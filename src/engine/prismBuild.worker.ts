/**
 * Dedicated Browser Web Worker for Prism WASM Check, Lean Proof, `.holo` v6 Seal,
 * Offline Archive Verification, and Portable Core-Wasm Guest Execution.
 */

import { inspect, loadArchive, parseArchive, resolveArchive } from 'prism-code';
import { relex, reprove, type KernelProofTheorem, type OpenKernel } from 'prism-code/reprove';
import { kappa } from '@prism-code-internal/blake3.mjs';
import { buildFromFiles, type BuildDiagnostic, type BuildTheoremResult } from '@prism-code-internal/journey/build-core.mjs';
import { kernelProver } from '@prism-code-internal/journey/prove-core.mjs';
import { decodeHoloview } from '@prism-code-internal/view.mjs';
import { instantiateAppFoundation } from './prismCompiler';
import { LEXLEAN_PIN, loadPrismToolchain } from './prismToolchain';

export type PrismBuildStage =
  | 'loading-toolchain'
  | 'checking'
  | 'proving'
  | 'sealing'
  | 'verifying';

export type PrismWorkerRequest =
  | {
      readonly id: string;
      readonly type: 'build';
      readonly baseUrl: string;
      readonly slug: string;
      readonly appFiles: Readonly<Record<string, string>>;
    }
  | {
      readonly id: string;
      readonly type: 'verify';
      readonly holoBytes: Uint8Array;
      readonly baseUrl?: string;
      readonly reprove?: boolean;
    }
  | {
      readonly id: string;
      readonly type: 'run';
      readonly address: string;
      readonly holoBytes: Uint8Array;
      readonly request: string;
    };

export interface WorkerBuildPayload {
  readonly ok: boolean;
  readonly stage: string;
  readonly message: string;
  readonly diagnostics: readonly BuildDiagnostic[];
  readonly semanticId: string | null;
  readonly proved: boolean;
  readonly theorems: readonly BuildTheoremResult[];
  readonly vectorsPassed: number;
  readonly vectorsTotal: number;
  readonly ms: number;
  readonly holo: {
    readonly bytes: Uint8Array;
    readonly address: string;
    readonly name: string;
    readonly semanticId: string;
    readonly carries: { readonly snapshot: boolean; readonly sources: boolean; readonly proof: boolean };
    readonly page: string;
  } | null;
}

export interface WorkerVerifyPayload {
  readonly ok: boolean;
  readonly address: string;
  readonly name: string;
  readonly version: string;
  readonly schema: string;
  readonly semanticId: string | null;
  readonly identitiesVerified: boolean;
  readonly vectorsPassed: number;
  readonly vectorsTotal: number;
  readonly vectorDetails: readonly {
    readonly index: number;
    readonly request: string;
    readonly expected: string;
    readonly actual: string;
    readonly passed: boolean;
  }[];
  readonly carries: { readonly snapshot: boolean; readonly sources: boolean; readonly proof: boolean };
  readonly reproved?: {
    readonly passed: boolean;
    readonly theorems: readonly KernelProofTheorem[];
    readonly reason: string | null;
  } | null;
  readonly sourcesMatch?: boolean | null;
  readonly viewFiles: Readonly<Record<string, string>> | null;
  readonly viewEntry: string | null;
  readonly error?: string;
}

export type PrismWorkerResponse =
  | { readonly id: string; readonly event: 'stage'; readonly stage: PrismBuildStage; readonly detail: string }
  | { readonly id: string; readonly event: 'build-result'; readonly result: WorkerBuildPayload }
  | { readonly id: string; readonly event: 'verify-result'; readonly result: WorkerVerifyPayload }
  | { readonly id: string; readonly event: 'run-result'; readonly ok: true; readonly answer: string }
  | { readonly id: string; readonly event: 'error'; readonly message: string };

interface LoadedGuestEntry {
  readonly dispatch: (bytes: Uint8Array) => Promise<Uint8Array>;
}

const guestCache = new Map<string, Promise<LoadedGuestEntry>>();
const enc = new TextEncoder();
const dec = new TextDecoder();

function post(msg: PrismWorkerResponse, transfer: Transferable[] = []): void {
  (self as unknown as Worker).postMessage(msg, transfer);
}

async function getLoadedGuest(address: string, holoBytes: Uint8Array): Promise<LoadedGuestEntry> {
  let pending = guestCache.get(address);
  if (!pending) {
    pending = (async () => {
      const app = await loadArchive(holoBytes, {});
      return { dispatch: (b: Uint8Array) => app.dispatch(b) };
    })();
    pending.catch(() => guestCache.delete(address));
    if (guestCache.size >= 8) {
      const oldest = guestCache.keys().next().value;
      if (oldest) guestCache.delete(oldest);
    }
    guestCache.set(address, pending);
  }
  return pending;
}

self.onmessage = async (event: MessageEvent<PrismWorkerRequest>) => {
  const req = event.data;
  if (!req || !req.id) return;

  try {
    if (req.type === 'build') {
      post({ id: req.id, event: 'stage', stage: 'loading-toolchain', detail: 'Loading BLAKE3-verified WASM toolchain…' });
      const tools = await loadPrismToolchain(req.baseUrl, (detail) => {
        post({ id: req.id, event: 'stage', stage: 'loading-toolchain', detail });
      });

      const mergedFiles = {
        ...instantiateAppFoundation(tools.foundationFiles, req.slug),
        ...req.appFiles,
      };

      const built = await buildFromFiles({
        files: mergedFiles,
        lexlean: tools.lexlean,
        evaluator: tools.evaluator,
        prove: kernelProver(tools.kernel, { record: true }),
        sources: true,
        onStage: (stage) => {
          const mapped: PrismBuildStage =
            stage === 'check' ? 'checking' : stage === 'prove' ? 'proving' : 'sealing';
          const detail =
            mapped === 'checking'
              ? 'Checking model & semantic snapshot…'
              : mapped === 'proving'
                ? 'Proving guarantees with Lean 4.32.2 WASM kernel…'
                : 'Sealing binary prism/holo/6 archive & replaying vectors…';
          post({ id: req.id, event: 'stage', stage: mapped, detail });
        },
      });

      const payload: WorkerBuildPayload = {
        ok: built.ok,
        stage: built.stage,
        message: built.message,
        diagnostics: built.diagnostics ?? [],
        semanticId: built.semanticId,
        proved: built.proved,
        theorems: built.theorems ?? [],
        vectorsPassed: built.vectors?.passed ?? 0,
        vectorsTotal: built.vectors?.total ?? 0,
        ms: built.ms,
        holo: built.holo
          ? {
              bytes: built.holo.bytes,
              address: built.holo.label,
              name: built.holo.name,
              semanticId: built.holo.semanticId,
              carries: built.holo.carries,
              page: built.holo.page,
            }
          : null,
      };

      const transfer: Transferable[] = payload.holo ? [payload.holo.bytes.buffer] : [];
      post({ id: req.id, event: 'build-result', result: payload }, transfer);
      return;
    }

    if (req.type === 'verify') {
      post({ id: req.id, event: 'stage', stage: 'verifying', detail: 'Verifying BLAKE3 identities & replaying acceptance vectors…' });
      const bytes = req.holoBytes instanceof Uint8Array ? req.holoBytes : new Uint8Array(req.holoBytes);
      const address = kappa(bytes);
      const inspected = inspect(bytes) as {
        name?: string;
        version?: string;
        schema?: string;
        semanticId?: string | null;
        carries?: { snapshot?: boolean; sources?: boolean; proof?: boolean };
      };
      const parsed = parseArchive(bytes);
      const resolved = resolveArchive(parsed);

      let viewFiles: Record<string, string> | null = null;
      let viewEntry: string | null = null;
      const viewBytes = resolved.view?.content;
      if (viewBytes instanceof Uint8Array && viewBytes.length > 0) {
        try {
          const decoded = decodeHoloview(viewBytes);
          viewEntry = decoded.entry;
          viewFiles = {};
          for (const [filePath, fileBytes] of decoded.files) {
            viewFiles[filePath] = dec.decode(fileBytes);
          }
        } catch {
          viewFiles = null;
        }
      }

      const app = await loadArchive(bytes, {});
      const acceptance = await app.acceptance();
      const failedSet = new Set(acceptance.failed.map((f: { index: number }) => f.index));
      const vectorDetails: {
        index: number;
        request: string;
        expected: string;
        actual: string;
        passed: boolean;
      }[] = [];
      for (const f of acceptance.failed as Array<{ index: number; request: Uint8Array; expected: Uint8Array; actual: Uint8Array }>) {
        vectorDetails.push({
          index: f.index,
          request: dec.decode(f.request),
          expected: dec.decode(f.expected),
          actual: dec.decode(f.actual),
          passed: false,
        });
      }

      let reprovedSummary: WorkerVerifyPayload['reproved'] = null;
      let sourcesMatch: boolean | null = null;
      if (req.reprove && req.baseUrl && inspected.carries?.snapshot) {
        post({ id: req.id, event: 'stage', stage: 'proving', detail: 'Re-proving embedded model snapshot with Lean 4.32.2 WASM kernel…' });
        const tools = await loadPrismToolchain(req.baseUrl, (detail) => {
          post({ id: req.id, event: 'stage', stage: 'loading-toolchain', detail });
        });
        const rp = await reprove(bytes, { kernel: tools.kernel as OpenKernel });
        reprovedSummary = {
          passed: rp.passed,
          theorems: rp.theorems,
          reason: rp.reason,
        };
        if (inspected.carries?.sources) {
          const rl = await relex(bytes, {
            lexlean: { module: tools.lexlean, kappa: LEXLEAN_PIN.kappa },
          });
          sourcesMatch = rl.matchesSnapshot;
        }
      }

      const ok =
        app.identities.verified === true &&
        failedSet.size === 0 &&
        (reprovedSummary === null || reprovedSummary.passed) &&
        (sourcesMatch === null || sourcesMatch);
      const result: WorkerVerifyPayload = {
        ok,
        address,
        name: inspected.name ?? app.name ?? 'Prism App',
        version: inspected.version ?? '0.1.0',
        schema: inspected.schema ?? 'prism/holo/6',
        semanticId:
          inspected.semanticId ??
          (typeof resolved.provenance?.semantic_id === 'string'
            ? resolved.provenance.semantic_id
            : null),
        identitiesVerified: app.identities.verified === true,
        vectorsPassed: acceptance.passed,
        vectorsTotal: acceptance.total,
        vectorDetails,
        carries: {
          snapshot: Boolean(inspected.carries?.snapshot),
          sources: Boolean(inspected.carries?.sources),
          proof: Boolean(inspected.carries?.proof),
        },
        reproved: reprovedSummary,
        sourcesMatch,
        viewFiles,
        viewEntry,
      };
      post({ id: req.id, event: 'verify-result', result });
      return;
    }

    if (req.type === 'run') {
      const guest = await getLoadedGuest(req.address, req.holoBytes);
      const outBytes = await guest.dispatch(enc.encode(req.request));
      post({ id: req.id, event: 'run-result', ok: true, answer: dec.decode(outBytes) });
      return;
    }
  } catch (error) {
    post({
      id: req.id,
      event: 'error',
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
