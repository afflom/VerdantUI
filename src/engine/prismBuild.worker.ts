/**
 * Dedicated Browser Web Worker for Prism WASM Check, Lean Proof, `.holo` v6 Seal,
 * Offline Archive Verification, and Portable Core-Wasm Guest Execution.
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
import { relex, reprove, type KernelProofTheorem, type OpenKernel } from 'prism-code/reprove';
import { kappa } from '@prism-code-internal/blake3.mjs';
import type { BuildDiagnostic, BuildTheoremResult } from '@prism-code-internal/journey/build-core.mjs';
import { lexleanCheck, lexleanSnapshot } from '@prism-code-internal/journey/lexlean-core.mjs';
import { buildHolo } from '@prism-code-internal/journey/model-holo.mjs';
import { kernelProver } from '@prism-code-internal/journey/prove-core.mjs';
import { decodeHoloview } from '@prism-code-internal/view.mjs';
import { instantiateAppFoundation } from './prismFoundation';
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

async function openVerifiedGuest(holoBytes: Uint8Array): Promise<{
  readonly name: string;
  readonly identitiesVerified: boolean;
  readonly dispatch: (bytes: Uint8Array) => Promise<Uint8Array>;
  readonly acceptance: () => Promise<{
    readonly passed: number;
    readonly total: number;
    readonly failed: readonly {
      readonly index: number;
      readonly request: Uint8Array;
      readonly expected: Uint8Array;
      readonly actual?: Uint8Array;
    }[];
  }>;
}> {
  const parsed = parseArchive(asBytes(holoBytes));
  const sealOk = parsed.format >= 5 ? parsed.verification.seal !== false : parsed.verification.footer !== false;
  if (!sealOk || parsed.verification.footer === false) {
    throw new Error('refused: holo integrity footer/seal verification failed');
  }
  const resolved = resolveArchive(parsed);
  const appDoc = resolved.modelDocument.application;
  if (!appDoc) throw new Error('refused: holo archive missing application document');
  const wasmModule = await compileGuest(resolved.guest.content, undefined, CORE_CONTRACT);
  const requestMaximum = Number(appDoc.request_maximum ?? appDoc.requestMaximum ?? 1024);
  const rawVectors = appDoc.acceptance_vectors ?? appDoc.acceptanceVectors ?? [];
  const vectors = rawVectors.map((v) => ({
    request: Uint8Array.from(v.request),
    response: Uint8Array.from(v.response),
  }));
  const dispatch = async (input: Uint8Array): Promise<Uint8Array> => {
    const reqBytes = asBytes(input);
    checkRequest(reqBytes, { request: requestMaximum, guestAllocation: requestMaximum * 16 });
    return runInstance(instantiateGuest(wasmModule, {}), reqBytes);
  };
  return {
    name: String(appDoc.name ?? 'Prism App'),
    identitiesVerified: resolved.identities?.verified !== false && resolved.identities?.ok !== false,
    dispatch,
    acceptance: async () => {
      const res = await runAcceptance(vectors, dispatch);
      return {
        passed: res.passed,
        total: res.total,
        failed: res.failed as readonly {
          readonly index: number;
          readonly request: Uint8Array;
          readonly expected: Uint8Array;
          readonly actual?: Uint8Array;
        }[],
      };
    },
  };
}

async function getLoadedGuest(address: string, holoBytes: Uint8Array): Promise<LoadedGuestEntry> {
  let pending = guestCache.get(address);
  if (!pending) {
    pending = (async () => {
      const app = await openVerifiedGuest(holoBytes);
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

      const started = performance.now();
      const elapsed = () => Math.round(performance.now() - started);
      const mergedFiles = {
        ...instantiateAppFoundation(tools.foundationFiles, req.slug),
        ...req.appFiles,
      };
      const fileMap = new Map<string, Uint8Array>(
        Object.entries(mergedFiles).map(([p, text]) => [
          p.startsWith('/') ? p : `/${p}`,
          enc.encode(text),
        ]),
      );
      const toDiag = (
        list?: readonly {
          readonly severity?: string;
          readonly code?: string;
          readonly message?: string;
          readonly primary?: { readonly path?: string; readonly line_start?: number; readonly column_start?: number } | null;
        }[],
      ): BuildDiagnostic[] =>
        (list ?? []).map((d) => ({
          severity: d.severity ?? 'error',
          code: d.code ?? 'lexlean.error',
          message: String(d.message ?? ''),
          file: d.primary?.path ?? null,
          line: d.primary?.line_start ?? null,
          column: d.primary?.column_start ?? null,
        }));

      post({ id: req.id, event: 'stage', stage: 'checking', detail: 'Checking model & semantic snapshot…' });
      const checkRes = await lexleanCheck(tools.lexlean, fileMap);
      if (!checkRes.success) {
        post({
          id: req.id,
          event: 'build-result',
          result: {
            ok: false,
            stage: 'check',
            message: 'LexLean refused the model.',
            diagnostics: toDiag(checkRes.diagnostics),
            semanticId: null,
            proved: false,
            theorems: [],
            vectorsPassed: 0,
            vectorsTotal: 0,
            ms: elapsed(),
            holo: null,
          },
        });
        return;
      }
      const snapRes = await lexleanSnapshot(tools.lexlean, fileMap);
      if (!snapRes.ok || !snapRes.snapshot) {
        post({
          id: req.id,
          event: 'build-result',
          result: {
            ok: false,
            stage: 'check',
            message: 'LexLean could not take semantic snapshot.',
            diagnostics: toDiag(snapRes.check?.diagnostics),
            semanticId: null,
            proved: false,
            theorems: [],
            vectorsPassed: 0,
            vectorsTotal: 0,
            ms: elapsed(),
            holo: null,
          },
        });
        return;
      }
      const diagnostics = toDiag(checkRes.diagnostics);
      const semanticId = `sha256:${snapRes.snapshot.semantic_id}`;

      post({ id: req.id, event: 'stage', stage: 'proving', detail: 'Proving guarantees with Lean 4.32.2 WASM kernel…' });
      const prove = kernelProver(tools.kernel, { record: true });
      const judged = await prove({ files: fileMap, snapshot: snapRes.snapshot, lexlean: tools.lexlean });
      const theorems: BuildTheoremResult[] = (judged.theorems ?? []).map((t) => ({
        theorem: t.theorem,
        status: t.status,
        ms: t.ms,
        axioms: t.axioms,
      }));
      if (!judged.ok) {
        post({
          id: req.id,
          event: 'build-result',
          result: {
            ok: false,
            stage: 'prove',
            message: judged.message ?? 'Lean kernel rejected one or more theorems.',
            diagnostics,
            semanticId,
            proved: false,
            theorems,
            vectorsPassed: 0,
            vectorsTotal: 0,
            ms: elapsed(),
            holo: null,
          },
        });
        return;
      }

      post({ id: req.id, event: 'stage', stage: 'sealing', detail: 'Sealing binary prism/holo/6 archive & replaying vectors…' });
      const carriedSources = Array.from(fileMap.entries()).map(([p, b]) => ({
        path: p.slice(1),
        bytes: b,
      }));
      const sealed = await buildHolo(snapRes.snapshot, {
        evaluator: tools.evaluator,
        snapshotBytes: snapRes.bytes ?? null,
        sources: carriedSources,
        proof: judged.record ?? null,
        page: fileMap,
      });
      const readBackGuest = await openVerifiedGuest(sealed.bytes);
      const vecs = await readBackGuest.acceptance();
      const buildOk = readBackGuest.identitiesVerified && vecs.failed.length === 0;

      const payload: WorkerBuildPayload = {
        ok: buildOk,
        stage: 'build',
        message: buildOk
          ? `Built verified prism/holo/6 (${sealed.bytes.byteLength} bytes, ${vecs.passed}/${vecs.total} vectors, ${theorems.length} proved).`
          : `Built archive failed readback verification (${vecs.passed}/${vecs.total} vectors).`,
        diagnostics,
        semanticId,
        proved: true,
        theorems,
        vectorsPassed: vecs.passed,
        vectorsTotal: vecs.total,
        ms: elapsed(),
        holo: buildOk
          ? {
              bytes: sealed.bytes,
              address: sealed.label,
              name: sealed.name,
              semanticId,
              carries: sealed.carries,
              page: sealed.page,
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
      const parsed = parseArchive(bytes);
      const resolved = resolveArchive(parsed);
      const extNames = new Set((parsed.extensions ?? []).map((e) => e.name));
      const carries = {
        snapshot: Array.from(extNames).some((n) => n.includes('snapshot')),
        sources: Array.from(extNames).some((n) => n.includes('source')),
        proof: Array.from(extNames).some((n) => n.includes('proof')),
      };
      const appDoc = resolved.modelDocument.application;

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

      const app = await openVerifiedGuest(bytes);
      const acceptance = await app.acceptance();
      const failedSet = new Set(acceptance.failed.map((f) => f.index));
      const vectorDetails: {
        index: number;
        request: string;
        expected: string;
        actual: string;
        passed: boolean;
      }[] = [];
      for (const f of acceptance.failed) {
        vectorDetails.push({
          index: f.index,
          request: dec.decode(f.request),
          expected: dec.decode(f.expected),
          actual: f.actual ? dec.decode(f.actual) : '',
          passed: false,
        });
      }

      let reprovedSummary: WorkerVerifyPayload['reproved'] = null;
      let sourcesMatch: boolean | null = null;
      if (req.reprove && req.baseUrl && carries.snapshot) {
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
        if (carries.sources) {
          const rl = await relex(bytes, {
            lexlean: { module: tools.lexlean, kappa: LEXLEAN_PIN.kappa },
          });
          sourcesMatch = rl.matchesSnapshot;
        }
      }

      const ok =
        app.identitiesVerified === true &&
        failedSet.size === 0 &&
        (reprovedSummary === null || reprovedSummary.passed) &&
        (sourcesMatch === null || sourcesMatch);
      const result: WorkerVerifyPayload = {
        ok,
        address,
        name: appDoc?.name ?? app.name ?? 'Prism App',
        version: appDoc?.cargo_version ?? appDoc?.version ?? '0.1.0',
        schema: `prism/holo/${parsed.format ?? 6}`,
        semanticId:
          typeof resolved.provenance?.semantic_id === 'string'
            ? resolved.provenance.semantic_id
            : null,
        identitiesVerified: app.identitiesVerified === true,
        vectorsPassed: acceptance.passed,
        vectorsTotal: acceptance.total,
        vectorDetails,
        carries,
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
