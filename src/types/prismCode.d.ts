/**
 * Typed declarations for browser-safe internal subpaths of `prism-code@0.11.5`.
 * These modules import zero Node APIs and zero WebGPU APIs.
 */

declare module '@prism-code-internal/design/compile.mjs' {
  export const DESIGN_SHEET: string;
  export const DESIGN_RUNTIME: string;
  export const DESIGN_SCHEMA: string;
  export interface CompiledDesign {
    readonly css: string;
    readonly js: string;
    readonly slots: readonly string[];
    readonly parts: readonly string[];
  }
  export function compileDesign(
    pageFiles: Map<string, Uint8Array>,
    themeBytes?: Uint8Array | null,
  ): CompiledDesign;
  export function usesDesign(pageFiles: Map<string, Uint8Array>): boolean;
}

declare module '@prism-code-internal/design/gate.mjs' {
  export interface DesignDiagnostic {
    readonly code: string;
    readonly severity: string;
    readonly file: string | null;
    readonly line: number | null;
    readonly message: string;
    readonly remediation?: string | null;
  }
  export function checkDesign(
    pageFiles: Map<string, Uint8Array>,
    themeBytes?: Uint8Array | null,
  ): readonly DesignDiagnostic[];
}

declare module '@prism-code-internal/design/system.mjs' {
  export const THEME_FILE: string;
  export const THEME_SCHEMA: string;
  export const PRESETS: Readonly<Record<string, { readonly name: string }>>;
  export function resolveTheme(bytes?: Uint8Array | null): {
    readonly name: string;
    readonly light: Record<string, string>;
    readonly dark: Record<string, string>;
    readonly modes: readonly string[];
  };
}

declare module '@prism-code-internal/journey/build-core.mjs' {
  export interface BuildDiagnostic {
    readonly severity: string;
    readonly code: string;
    readonly message: string;
    readonly file: string | null;
    readonly line: number | null;
    readonly column: number | null;
  }
  export interface BuildTheoremResult {
    readonly theorem: string;
    readonly status: 'proved' | 'not proved' | 'not settled' | string;
    readonly ms?: number;
    readonly axioms?: readonly string[];
  }
  export interface BuildFromFilesResult {
    readonly ok: boolean;
    readonly stage: 'check' | 'prove' | 'build' | string;
    readonly code?: string | null;
    readonly diagnostics: readonly BuildDiagnostic[];
    readonly semanticId: string | null;
    readonly proved: boolean;
    readonly theorems: readonly BuildTheoremResult[] | null;
    readonly holo: {
      readonly bytes: Uint8Array;
      readonly label: string;
      readonly name: string;
      readonly semanticId: string;
      readonly carries: { readonly snapshot: boolean; readonly sources: boolean; readonly proof: boolean };
      readonly page: 'authored' | 'generated' | string;
    } | null;
    readonly vectors?: { readonly passed: number; readonly total: number };
    readonly ms: number;
    readonly message: string;
  }
  export function buildFromFiles(opts: {
    files: Map<string, Uint8Array> | Record<string, string | Uint8Array>;
    lexlean: WebAssembly.Module;
    evaluator: Uint8Array;
    prove?: ((input: {
      files: Map<string, Uint8Array>;
      snapshot: unknown;
      lexlean: WebAssembly.Module;
    }) => Promise<{
      ok: boolean;
      theorems: readonly BuildTheoremResult[];
      message?: string;
      code?: string;
      record?: { kernel: string; library: string; [k: string]: unknown };
    }>) | null;
    sources?: boolean;
    log?: (line: string) => void;
    onStage?: (stage: 'check' | 'prove' | 'build') => void;
  }): Promise<BuildFromFilesResult>;
  export function readBack(bytes: Uint8Array): Promise<{
    readonly verified: boolean;
    readonly passed: number;
    readonly total: number;
    readonly failed: number;
  }>;
}

declare module '@prism-code-internal/journey/toolchain.mjs' {
  export interface BlobStorePort {
    get(label: string): Promise<Uint8Array | null>;
    put(bytes: Uint8Array): Promise<string>;
  }
  export function memoryStore(): BlobStorePort;
  export function admitModule(
    pin: { readonly kappa: string; readonly url: string; readonly encoding?: 'gzip' | 'identity' },
    opts?: { store?: BlobStorePort | null; fetch?: typeof globalThis.fetch | null; log?: (msg: string) => void; what?: string },
  ): Promise<{ readonly module: WebAssembly.Module; readonly from: 'store' | 'fetch'; readonly ms: number }>;
  export function admitBytes(
    pin: { readonly kappa: string; readonly url: string; readonly encoding?: 'gzip' | 'identity' },
    opts?: { store?: BlobStorePort | null; fetch?: typeof globalThis.fetch | null; what?: string },
  ): Promise<Uint8Array>;
}

declare module '@prism-code-internal/journey/kappa-opfs.mjs' {
  export function opfsStore(opts?: { root?: FileSystemDirectoryHandle | null }): Promise<{
    readonly off: boolean;
    put(bytes: Uint8Array): Promise<string>;
    get(name: string): Promise<Uint8Array | null>;
    bind(key: string, name: string): Promise<void>;
    bound(key: string): Promise<string | null>;
  }>;
}

declare module '@prism-code-internal/journey/prove-core.mjs' {
  export function openKernel(opts: {
    factory: (cfg: Record<string, unknown>) => Promise<unknown>;
    wasm: Uint8Array;
    baseText: string;
    image?: Uint8Array | null;
    keepImage?: ((image: Uint8Array) => void) | null;
    resettable?: boolean;
    labels?: { readonly kernel: string; readonly library: string } | null;
  }): Promise<unknown>;
  export function kernelProver(
    kernel: unknown,
    opts?: {
      known?: Map<string, unknown>;
      keep?: unknown;
      record?: boolean;
      proofs?: unknown;
    },
  ): (input: {
    files: Map<string, Uint8Array>;
    snapshot: unknown;
    lexlean: WebAssembly.Module;
  }) => Promise<{
    ok: boolean;
    theorems: readonly { readonly theorem: string; readonly name?: string; readonly status: string; readonly ms?: number; readonly axioms?: readonly string[] }[];
    message?: string;
    code?: string;
    record?: { kernel: string; library: string; [k: string]: unknown };
  }>;
  export function openKernelFrom(
    pins: {
      readonly loader: { readonly kappa: string; readonly url: string; readonly encoding?: 'identity' | 'gzip' };
      readonly kernel: { readonly kappa: string; readonly url: string; readonly encoding?: 'identity' | 'gzip' };
      readonly library: { readonly kappa: string; readonly url: string; readonly encoding?: 'identity' | 'gzip' };
    },
    opts?: {
      store?: { get(label: string): Promise<Uint8Array | null>; put(bytes: Uint8Array): Promise<string> } | null;
      fetch?: typeof globalThis.fetch | null;
      resettable?: boolean;
    },
  ): Promise<unknown>;
}

declare module '@prism-code-internal/journey/model-kernel-pins.mjs' {
  export const MODEL_KERNEL_FILES: Readonly<Record<
    'lean-kernel.mjs' | 'lean-kernel.wasm.gz' | 'base.ndjson.gz',
    { readonly path: string; readonly kappa: string; readonly encoding: 'identity' | 'gzip'; readonly bytes: number }
  >>;
}

declare module '@prism-code-internal/journey/lexlean-core.mjs' {
  export function lexleanCheck(module: WebAssembly.Module, files: Map<string, Uint8Array>): Promise<{
    readonly success: boolean;
    readonly diagnostics?: readonly {
      readonly severity?: string;
      readonly code?: string;
      readonly message?: string;
      readonly primary?: { readonly path?: string; readonly line_start?: number; readonly column_start?: number } | null;
    }[];
    readonly ms: number;
  }>;
  export function lexleanSnapshot(module: WebAssembly.Module, files: Map<string, Uint8Array>): Promise<{
    readonly ok: boolean;
    readonly bytes?: Uint8Array;
    readonly snapshot?: { readonly semantic_id: string; [k: string]: unknown };
    readonly check?: {
      readonly diagnostics?: readonly {
        readonly severity?: string;
        readonly code?: string;
        readonly message?: string;
        readonly primary?: { readonly path?: string; readonly line_start?: number; readonly column_start?: number } | null;
      }[];
    };
    readonly ms: number;
  }>;
}

declare module '@prism-code-internal/journey/model-holo.mjs' {
  export function buildHolo(
    snapshot: unknown,
    opts: {
      readonly evaluator: Uint8Array;
      readonly snapshotBytes?: Uint8Array | null;
      readonly sources?: ReadonlyArray<{ readonly path: string; readonly bytes: Uint8Array }> | null;
      readonly proof?: { readonly kernel: string; readonly library: string; [k: string]: unknown } | null;
      readonly page?: Map<string, Uint8Array> | null;
    },
  ): Promise<{
    readonly bytes: Uint8Array;
    readonly label: string;
    readonly name: string;
    readonly carries: { readonly snapshot: boolean; readonly sources: boolean; readonly proof: boolean };
    readonly page: string;
  }>;
}

declare module '@prism-code-internal/blake3.mjs' {
  export function blake3Hex(bytes: Uint8Array): string;
  export function kappa(bytes: Uint8Array): string;
}

declare module '@prism-code-internal/sha256.mjs' {
  export function sha256HexSync(bytes: Uint8Array): string;
}

declare module '@prism-code-internal/view.mjs' {
  export function decodeHoloview(bytes: Uint8Array): {
    readonly entry: string;
    readonly files: Map<string, Uint8Array>;
  };
}

declare module '@prism-code-internal/archive.mjs' {
  export function asBytes(value: Uint8Array | ArrayBuffer | string): Uint8Array;
  export function parseArchive(bytes: Uint8Array): {
    readonly format: number;
    readonly kappa?: string;
    readonly fingerprint?: string;
    readonly extensions?: ReadonlyArray<{ readonly name: string }>;
    readonly verification: { readonly footer?: boolean; readonly seal?: boolean };
  };
  export function resolveArchive(parsed: ReturnType<typeof parseArchive>): {
    readonly modelDocument: {
      readonly application?: {
        readonly name?: string;
        readonly version?: string;
        readonly cargo_version?: string;
        readonly request_maximum?: number;
        readonly requestMaximum?: number;
        readonly acceptance_vectors?: ReadonlyArray<{ readonly request: ArrayLike<number>; readonly response: ArrayLike<number> }>;
        readonly acceptanceVectors?: ReadonlyArray<{ readonly request: ArrayLike<number>; readonly response: ArrayLike<number> }>;
      };
    };
    readonly guest: { readonly content: Uint8Array };
    readonly view?: { readonly content?: Uint8Array };
    readonly provenance?: { readonly semantic_id?: string };
    readonly identities: { readonly ok?: boolean; readonly verified?: boolean; readonly problems?: readonly string[] };
  };
}

declare module '@prism-code-internal/core.mjs' {
  export const CORE_CONTRACT: string;
  export function compileGuest(
    wasmBytes: Uint8Array,
    grant?: unknown,
    contract?: string,
  ): Promise<WebAssembly.Module>;
  export function checkRequest(request: Uint8Array, limits?: { readonly request?: number; readonly guestAllocation?: number }): void;
  export function instantiateGuest(
    module: WebAssembly.Module,
    imports?: Record<string, Record<string, unknown>>,
  ): WebAssembly.Instance;
  export function runInstance(instance: WebAssembly.Instance, request: Uint8Array): Uint8Array;
  export function runAcceptance(
    vectors: ReadonlyArray<{ readonly request: Uint8Array; readonly response: Uint8Array }>,
    dispatch: (request: Uint8Array) => Promise<Uint8Array> | Uint8Array,
  ): Promise<{ readonly passed: number; readonly total: number; readonly failed: readonly unknown[] }>;
}

declare module '@prism-code-internal/middleware.mjs' {
  export interface MiddlewareStack {
    use(pluggable: unknown): unknown;
    resolve(
      step: string,
      terminal: (args: Record<string, unknown>) => Promise<unknown> | unknown,
      context: Record<string, unknown>,
    ): (args: Record<string, unknown>) => Promise<unknown>;
  }
  export function createMiddlewareStack(): MiddlewareStack;
}

declare module '@prism-code-internal/tracing.mjs' {
  export function experimental_tracing(options: {
    readonly tracer: unknown;
    readonly steps?: readonly string[];
  }): unknown;
}
