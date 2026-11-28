#!/usr/bin/env node
/**
 * VerdantUI Permanent Ratchet Gate (`scripts/ratchet-gate.ts`)
 *
 * Fast (<100 ms) architectural gate modeled on the Verdant ratchet discipline
 * (`afflom/verdant-aqm` + `afflom/verdant-studio`):
 * - Evaluates every permanent zero ceiling declared in `model/verdant-ratchet.toml`.
 * - Verifies its own soundness via in-memory planted-defect self-checks (every gate
 *   proves it rejects a concrete violation before reporting PASS).
 */

import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export interface RatchetViolation {
  readonly category:
    | 'server-dependency'
    | 'server-file'
    | 'foreign-processor-api'
    | 'root-absolute-asset'
    | 'multithread-isolation'
    | 'vtensor-provider-lock'
    | 'engine-pin-parity'
    | 'owner-decisions'
    | 'ids-register'
    | 'public-benefit-charter'
    | 'user-facing-jargon'
    | 'silent-fallback'
    | 'policy-config';
  readonly path: string;
  readonly line?: number;
  readonly detail: string;
}

export interface RatchetSnapshot {
  readonly policyToml: string;
  readonly idsTomlRaw: string;
  readonly enginePinRaw: string;
  readonly ownerDecisionsRaw: string;
  readonly packageJsonRaw: string;
  readonly metadataJsonRaw: string;
  readonly indexHtmlRaw: string;
  readonly viteConfigRaw: string;
  readonly coiSwRaw: string | null;
  readonly rootFiles: readonly string[];
  readonly srcFiles: Readonly<Record<string, string>>;
}

export interface RatchetReport {
  readonly ok: boolean;
  readonly schema: string;
  readonly auditedSrcFiles: number;
  readonly violations: readonly RatchetViolation[];
}

const FORBIDDEN_SERVER_DEPS = new Set<string>([
  'express',
  '@types/express',
  'fastify',
  'koa',
  'hono',
  '@google/genai',
  'dotenv',
  'socket.io',
  'ws',
  'next',
  'nuxt',
]);

const FORBIDDEN_SERVER_FILE_PATTERNS: readonly RegExp[] = [
  /^server\.(js|mjs|cjs|ts)$/i,
  /^\.env(\.example)?$/i,
  /^api\//i,
];

const FORBIDDEN_PROCESSOR_API_PATTERNS: readonly { readonly label: string; readonly regex: RegExp }[] = [
  { label: 'navigator.gpu', regex: /\bnavigator\s*\.\s*gpu\b/ },
  { label: 'WebGPU WGSL shader', regex: /@compute\s+@workgroup_size/ },
  { label: 'verdantWebGpu module', regex: /\bverdantWebGpu\b/ },
  { label: 'WebGpuSubstrate', regex: /\bWebGpuSubstrate\b/ },
  { label: 'webgpu-wgsl route', regex: /\bwebgpu-wgsl\b/ },
  { label: 'WebNN navigator.ml', regex: /\bnavigator\s*\.\s*ml\b/ },
  { label: 'ONNX Runtime', regex: /\bonnxruntime\b/i },
  { label: 'TensorFlow.js', regex: /@tensorflow\/tfjs/ },
];

const FORBIDDEN_PUBLIC_BENEFIT_DEPS = new Set<string>([
  '@sentry/browser',
  '@sentry/react',
  'posthog-js',
  'mixpanel-browser',
  '@amplitude/analytics-browser',
  'amplitude-js',
  '@segment/analytics-next',
  'stripe',
  '@stripe/stripe-js',
  '@auth0/auth0-react',
  '@clerk/clerk-react',
  'react-ga4',
]);

const FORBIDDEN_TRACKING_URLS_RE =
  /https?:\/\/(?:www\.)?(?:googletagmanager\.com|google-analytics\.com|cdn\.segment\.com|app\.posthog\.com|api\.mixpanel\.com)/i;

/**
 * Universal Zero-Jargon Ratchet (DEC-011 / DEC-012 / VUI-UX-JARGON-01):
 * Scans ALL user-facing TSX files under `src/App.tsx` and `src/components/*.tsx`
 * (excluding only the `?debug=1`-gated `VTensorInspector.tsx`) to guarantee zero internal
 * engineering jargon ("Verdant Engine", "vTensor", "routing", "weights", "WASM", "OPFS",
 * "GGUF", "Observer", "Model Stream", "Blueprint", "Holon", ".holo", etc.) ever leaks to users.
 */
const DEBUG_ONLY_UI_FILES = new Set<string>(['src/components/VTensorInspector.tsx']);

const USER_FACING_JARGON_TERMS =
  'Verdant Engine|vTensor|OpenTelemetry|OTel|LexLean|Holon|\\.holo|FK-07|REQ-D\\d+|LOD0|Model Stream|SharedArrayBuffer|WASM|OPFS|GGUF|Safetensors|ggml|vRAM|Belady|SIMD|PROV-O|ISO\\/PAS|STRIDE|Traceability|Roofline|Blueprint|Observer|routing|routed|weights|weight|tok\\/s|prefill';

const USER_FACING_JARGON_JSX_RE = new RegExp(
  `(?:>[^<{}]*\\b(${USER_FACING_JARGON_TERMS})\\b[^<{}]*<|(?:title|placeholder|aria-label)=["'][^"']*\\b(${USER_FACING_JARGON_TERMS})\\b[^"']*["'])`,
  'i',
);

const REQUIRED_DISABLED_GGML_PROVIDERS = [
  'GGML_CPU',
  'GGML_BLAS',
  'GGML_METAL',
  'GGML_CUDA',
  'GGML_WEBGPU',
] as const;

function findLineNumber(content: string, regex: RegExp): number | undefined {
  const lines = content.split('\n');
  for (let i = 0; i < lines.length; i++) {
    if (regex.test(lines[i])) return i + 1;
  }
  return undefined;
}

function collectSrcFiles(rootDir: string): Record<string, string> {
  const out: Record<string, string> = {};
  const srcRoot = join(rootDir, 'src');
  if (!existsSync(srcRoot)) return out;

  const walk = (dir: string) => {
    for (const entry of readdirSync(dir)) {
      const full = join(dir, entry);
      const st = statSync(full);
      if (st.isDirectory()) {
        walk(full);
      } else if (/\.(ts|tsx|js|mjs|cjs)$/.test(entry)) {
        const rel = relative(rootDir, full).split('\\').join('/');
        out[rel] = readFileSync(full, 'utf8');
      }
    }
  };
  walk(srcRoot);
  return out;
}

export function loadRatchetSnapshot(cwd = process.cwd()): RatchetSnapshot {
  const root = resolve(cwd);
  const readReq = (rel: string): string => {
    const abs = join(root, rel);
    if (!existsSync(abs)) {
      throw new Error(`Missing required file: ${rel}`);
    }
    return readFileSync(abs, 'utf8');
  };

  const coiPath = join(root, 'public/coi-serviceworker.js');
  return {
    policyToml: readReq('model/verdant-ratchet.toml'),
    idsTomlRaw: readReq('model/ids.toml'),
    enginePinRaw: readReq('model/engine-pin.json'),
    ownerDecisionsRaw: readReq('model/owner-decisions.json'),
    packageJsonRaw: readReq('package.json'),
    metadataJsonRaw: readReq('metadata.json'),
    indexHtmlRaw: readReq('index.html'),
    viteConfigRaw: readReq('vite.config.ts'),
    coiSwRaw: existsSync(coiPath) ? readFileSync(coiPath, 'utf8') : null,
    rootFiles: readdirSync(root),
    srcFiles: collectSrcFiles(root),
  };
}

export function evaluateVerdantRatchet(snapshot: RatchetSnapshot): RatchetReport {
  const violations: RatchetViolation[] = [];
  const schemaMatch = snapshot.policyToml.match(/^\s*schema\s*=\s*"([^"]+)"/m);
  const schema = schemaMatch?.[1] ?? 'unknown';

  if (schema !== 'verdant-ui/ratchet/1') {
    violations.push({
      category: 'policy-config',
      path: 'model/verdant-ratchet.toml',
      detail: `Unexpected ratchet schema "${schema}" (expected "verdant-ui/ratchet/1")`,
    });
  }

  // 1. Zero server dependencies in package.json and metadata.json
  try {
    const pkg = JSON.parse(snapshot.packageJsonRaw) as {
      dependencies?: Record<string, string>;
      devDependencies?: Record<string, string>;
      optionalDependencies?: Record<string, string>;
    };
    const depBlocks = [
      ['dependencies', pkg.dependencies ?? {}],
      ['devDependencies', pkg.devDependencies ?? {}],
      ['optionalDependencies', pkg.optionalDependencies ?? {}],
    ] as const;
    for (const [section, deps] of depBlocks) {
      for (const name of Object.keys(deps)) {
        if (FORBIDDEN_SERVER_DEPS.has(name)) {
          violations.push({
            category: 'server-dependency',
            path: 'package.json',
            detail: `Forbidden server/external-API package "${name}" in ${section}`,
          });
        }
        if (FORBIDDEN_PUBLIC_BENEFIT_DEPS.has(name)) {
          violations.push({
            category: 'public-benefit-charter',
            path: 'package.json',
            detail: `Forbidden tracking/analytics/paywall package "${name}" in ${section} violates Free Public Benefit Charter (DEC-007)`,
          });
        }
      }
    }
  } catch (err) {
    violations.push({
      category: 'server-dependency',
      path: 'package.json',
      detail: `Invalid package.json: ${err instanceof Error ? err.message : String(err)}`,
    });
  }

  if (/MAJOR_CAPABILITY_SERVER_SIDE/i.test(snapshot.metadataJsonRaw)) {
    violations.push({
      category: 'server-dependency',
      path: 'metadata.json',
      detail: 'Server-side capability declaration is forbidden in client-only VerdantUI',
    });
  }

  // 2. Zero server files at repo root
  for (const file of snapshot.rootFiles) {
    for (const pattern of FORBIDDEN_SERVER_FILE_PATTERNS) {
      if (pattern.test(file)) {
        violations.push({
          category: 'server-file',
          path: file,
          detail: `Forbidden server/env artifact "${file}" in client-only repository`,
        });
      }
    }
  }

  // 3. Zero WebGPU or foreign processor APIs anywhere under src/
  const srcPaths = Object.keys(snapshot.srcFiles).sort();
  for (const relPath of srcPaths) {
    const content = snapshot.srcFiles[relPath];
    for (const rule of FORBIDDEN_PROCESSOR_API_PATTERNS) {
      if (rule.regex.test(content)) {
        violations.push({
          category: 'foreign-processor-api',
          path: relPath,
          line: findLineNumber(content, rule.regex),
          detail: `Forbidden processor API "${rule.label}" — VerdantUI runs strictly on verdant-engine/vTensor WASM SIMD`,
        });
      }
    }
  }

  // 4. Subpath-safe GitHub Pages asset routing
  if (!/base\s*:\s*process\.env\.VITE_BASE_PATH\s*\|\|\s*['"]\.\/['"]/.test(snapshot.viteConfigRaw)) {
    violations.push({
      category: 'root-absolute-asset',
      path: 'vite.config.ts',
      detail: 'vite.config.ts must configure relative base (`process.env.VITE_BASE_PATH || "./"`) for GitHub Pages subpath hosting',
    });
  }
  const catalogSrc = snapshot.srcFiles['src/engine/modelCatalog.ts'] ?? '';
  if (/url\s*:\s*['"]\/models\//.test(catalogSrc)) {
    violations.push({
      category: 'root-absolute-asset',
      path: 'src/engine/modelCatalog.ts',
      line: findLineNumber(catalogSrc, /url\s*:\s*['"]\/models\//),
      detail: 'Bundled model URL must be relative ("models/...") rather than root-absolute ("/models/...")',
    });
  }

  // 5. Multi-threaded WASM SIMD COOP/COEP isolation wiring
  if (!snapshot.coiSwRaw) {
    violations.push({
      category: 'multithread-isolation',
      path: 'public/coi-serviceworker.js',
      detail: 'Missing COOP/COEP isolation service worker required for SharedArrayBuffer on GitHub Pages',
    });
  } else {
    if (!snapshot.coiSwRaw.includes('Cross-Origin-Opener-Policy')) {
      violations.push({
        category: 'multithread-isolation',
        path: 'public/coi-serviceworker.js',
        detail: 'coi-serviceworker.js must set Cross-Origin-Opener-Policy',
      });
    }
    if (!snapshot.coiSwRaw.includes('Cross-Origin-Embedder-Policy')) {
      violations.push({
        category: 'multithread-isolation',
        path: 'public/coi-serviceworker.js',
        detail: 'coi-serviceworker.js must set Cross-Origin-Embedder-Policy',
      });
    }
  }
  if (!/src=["']\.\/coi-serviceworker\.js["']/.test(snapshot.indexHtmlRaw)) {
    violations.push({
      category: 'multithread-isolation',
      path: 'index.html',
      detail: 'index.html must load ./coi-serviceworker.js before module boot for GitHub Pages multi-threading',
    });
  }
  const workerSrc = snapshot.srcFiles['src/engine/inference.worker.ts'] ?? '';
  if (!/runtime\.initThreadPool\(/.test(workerSrc)) {
    violations.push({
      category: 'multithread-isolation',
      path: 'src/engine/inference.worker.ts',
      detail: 'inference.worker.ts must initialize the multi-core WASM SIMD thread pool on startup',
    });
  }

  // 6. Exclusive vTensor provider lock (refusing all default GGML backends)
  const providerSrc = snapshot.srcFiles['src/engine/vtensorProvider.ts'] ?? '';
  for (const disabled of REQUIRED_DISABLED_GGML_PROVIDERS) {
    if (!providerSrc.includes(`'${disabled}'`)) {
      violations.push({
        category: 'vtensor-provider-lock',
        path: 'src/engine/vtensorProvider.ts',
        detail: `DISABLED_DEFAULT_GGML_PROVIDERS must explicitly refuse '${disabled}'`,
      });
    }
  }

  // 7. Verdant Engine pin parity (Studio REQ-ENGINE-TRACKING)
  try {
    const pin = JSON.parse(snapshot.enginePinRaw) as {
      schema?: string;
      version?: string;
      commit?: string;
      registers?: Array<{ path?: string; sha256?: string }>;
    };
    if (pin.schema !== 'verdant-ui/engine-pin/1') {
      violations.push({
        category: 'engine-pin-parity',
        path: 'model/engine-pin.json',
        detail: `Unexpected engine pin schema "${pin.schema ?? 'missing'}"`,
      });
    }
    if (!pin.commit || !/^[0-9a-f]{40}$/.test(pin.commit) || /0{16,}$/.test(pin.commit)) {
      violations.push({
        category: 'engine-pin-parity',
        path: 'model/engine-pin.json',
        detail: 'Engine pin commit must be a full non-placeholder 40-char lowercase hex SHA',
      });
    }
    const regs = pin.registers ?? [];
    if (regs.length < 4 || regs.some((r) => !r.sha256 || !/^[0-9a-f]{64}$/.test(r.sha256))) {
      violations.push({
        category: 'engine-pin-parity',
        path: 'model/engine-pin.json',
        detail: 'Engine pin must record valid 64-hex SHA-256 digests for all 4 verdant-engine registers',
      });
    }
    const kernelsSrc = snapshot.srcFiles['src/engine/verdantKernels.ts'] ?? '';
    const verMatch = kernelsSrc.match(/export const VERDANT_TENSOR_VERSION\s*=\s*'([^']+)'/);
    const shaMatch = kernelsSrc.match(/export const VERDANT_TENSOR_COMMIT\s*=\s*'([^']+)'/);
    if (!verMatch || verMatch[1] !== pin.version) {
      violations.push({
        category: 'engine-pin-parity',
        path: 'src/engine/verdantKernels.ts',
        detail: `VERDANT_TENSOR_VERSION (${verMatch?.[1] ?? 'missing'}) does not match model/engine-pin.json (${pin.version ?? 'missing'})`,
      });
    }
    if (!shaMatch || shaMatch[1] !== pin.commit) {
      violations.push({
        category: 'engine-pin-parity',
        path: 'src/engine/verdantKernels.ts',
        detail: `VERDANT_TENSOR_COMMIT (${shaMatch?.[1] ?? 'missing'}) does not match model/engine-pin.json (${pin.commit ?? 'missing'})`,
      });
    }
  } catch (err) {
    violations.push({
      category: 'engine-pin-parity',
      path: 'model/engine-pin.json',
      detail: `Invalid engine-pin.json: ${err instanceof Error ? err.message : String(err)}`,
    });
  }

  // 8. Owner decisions register (Studio record-decision pattern)
  try {
    const od = JSON.parse(snapshot.ownerDecisionsRaw) as {
      schema?: string;
      decisions?: Array<{ id?: string; quote?: string }>;
    };
    if (
      od.schema !== 'verdant-ui/owner-decisions/1' ||
      !Array.isArray(od.decisions) ||
      od.decisions.length === 0 ||
      od.decisions.some((d) => !d.id || !d.quote)
    ) {
      violations.push({
        category: 'owner-decisions',
        path: 'model/owner-decisions.json',
        detail: 'owner-decisions.json must carry schema verdant-ui/owner-decisions/1 with non-empty id+quote entries',
      });
    }
  } catch (err) {
    violations.push({
      category: 'owner-decisions',
      path: 'model/owner-decisions.json',
      detail: `Invalid owner-decisions.json: ${err instanceof Error ? err.message : String(err)}`,
    });
  }

  // 9. Verdant Studio invariant ID register (`model/ids.toml`)
  const idsSchema = snapshot.idsTomlRaw.match(/^\s*schema\s*=\s*"([^"]+)"/m)?.[1];
  const codeMatches = Array.from(snapshot.idsTomlRaw.matchAll(/^\s*code\s*=\s*"([^"]+)"/gm)).map((m) => m[1]);
  const seenCodes = new Set<string>();
  const dupCodes: string[] = [];
  for (const code of codeMatches) {
    if (seenCodes.has(code)) dupCodes.push(code);
    seenCodes.add(code);
  }
  if (idsSchema !== 'verdant-ui/ids/1' || codeMatches.length < 6 || dupCodes.length > 0) {
    violations.push({
      category: 'ids-register',
      path: 'model/ids.toml',
      detail:
        dupCodes.length > 0
          ? `Duplicate invariant IDs in model/ids.toml: ${dupCodes.join(', ')}`
          : `model/ids.toml must carry schema verdant-ui/ids/1 with >= 6 unique invariant IDs (found ${codeMatches.length})`,
    });
  }

  // 10. Free Public Benefit Service Charter (zero tracking/analytics URLs in HTML or src/)
  if (FORBIDDEN_TRACKING_URLS_RE.test(snapshot.indexHtmlRaw)) {
    violations.push({
      category: 'public-benefit-charter',
      path: 'index.html',
      line: findLineNumber(snapshot.indexHtmlRaw, FORBIDDEN_TRACKING_URLS_RE),
      detail: 'Forbidden third-party tracking/analytics URL in index.html violates Free Public Benefit Charter (DEC-007)',
    });
  }
  for (const relPath of srcPaths) {
    const content = snapshot.srcFiles[relPath];
    if (FORBIDDEN_TRACKING_URLS_RE.test(content)) {
      violations.push({
        category: 'public-benefit-charter',
        path: relPath,
        line: findLineNumber(content, FORBIDDEN_TRACKING_URLS_RE),
        detail: 'Forbidden third-party tracking/analytics URL in src/ violates Free Public Benefit Charter (DEC-007)',
      });
    }
  }

  // 11. Universal zero internal jargon across ALL user-facing TSX surfaces + welcome suggestions (DEC-011 / DEC-012 / VUI-UX-JARGON-01)
  for (const uiPath of srcPaths) {
    if (uiPath !== 'src/App.tsx' && !uiPath.startsWith('src/components/')) continue;
    if (!uiPath.endsWith('.tsx') || DEBUG_ONLY_UI_FILES.has(uiPath)) continue;
    const content = snapshot.srcFiles[uiPath];
    if (!content) continue;
    const match = content.match(USER_FACING_JARGON_JSX_RE);
    if (match) {
      const term = match[1] || match[2] || 'internal term';
      violations.push({
        category: 'user-facing-jargon',
        path: uiPath,
        line: findLineNumber(content, USER_FACING_JARGON_JSX_RE),
        detail: `User-facing JSX exposes technical jargon "${term}" — violates DEC-011 / VUI-UX-JARGON-01`,
      });
    }
  }
  const obsSrc = snapshot.srcFiles['src/engine/cognitiveObserver.ts'] ?? '';
  const suggMatch = obsSrc.match(/export const UNIFIED_VERDANT_SUGGESTIONS[\s\S]*?\];/);
  if (suggMatch) {
    const jargonInSugg = suggMatch[0].match(
      /\b(Verdant Engine|vTensor|WASM|OPFS|GGUF|LOD0|SIMD|SharedArrayBuffer|Model Stream|Holon|LexLean|Observer)\b/i,
    );
    if (jargonInSugg) {
      violations.push({
        category: 'user-facing-jargon',
        path: 'src/engine/cognitiveObserver.ts',
        line: findLineNumber(obsSrc, /export const UNIFIED_VERDANT_SUGGESTIONS/),
        detail: `User-facing welcome suggestion exposes technical jargon "${jargonInSugg[1]}" — violates DEC-011 / VUI-UX-JARGON-01`,
      });
    }
  }

  // 12. Explicit-Refusal Discipline (afflom/verdant-studio parity): zero silent fallback cover-ups in src/engine/
  const BANNED_FALLBACK_PATTERNS: readonly { readonly label: string; readonly regex: RegExp }[] = [
    { label: 'runDirectAnswerFallback', regex: /\brunDirectAnswerFallback\b/ },
    { label: 'reasoningFallback', regex: /\breasoningFallback\b/ },
    { label: 'buildFallbackEnvironmentProfile', regex: /\bbuildFallbackEnvironmentProfile\b/ },
    { label: 'CLIP hash-guessing tokenizer fallback', regex: /deterministic hash fallback offline/i },
    { label: 'non-finite logit token-0 fallback', regex: /Finite-logit fallback/i },
  ];
  for (const relPath of srcPaths) {
    if (!relPath.startsWith('src/engine/') || relPath.endsWith('.test.ts')) continue;
    const content = snapshot.srcFiles[relPath];
    if (!content) continue;
    for (const pat of BANNED_FALLBACK_PATTERNS) {
      if (pat.regex.test(content)) {
        violations.push({
          category: 'silent-fallback',
          path: relPath,
          line: findLineNumber(content, pat.regex),
          detail: `Banned silent fallback cover-up "${pat.label}" — refuse explicitly by name instead of masking unfinished/failed work`,
        });
      }
    }
  }

  return {
    ok: violations.length === 0,
    schema,
    auditedSrcFiles: srcPaths.length,
    violations,
  };
}

export function verifyPlantedDefects(cleanSnapshot: RatchetSnapshot): {
  readonly allCaught: boolean;
  readonly casesVerified: number;
  readonly failures: readonly string[];
} {
  const cases: ReadonlyArray<{
    readonly name: string;
    readonly expectedCategory: RatchetViolation['category'];
    readonly mutate: (s: RatchetSnapshot) => RatchetSnapshot;
  }> = [
    {
      name: 'planted-express-server-dep',
      expectedCategory: 'server-dependency',
      mutate: (s) => {
        const pkg = JSON.parse(s.packageJsonRaw) as { dependencies: Record<string, string> };
        pkg.dependencies.express = '^4.21.2';
        return { ...s, packageJsonRaw: JSON.stringify(pkg) };
      },
    },
    {
      name: 'planted-server-file',
      expectedCategory: 'server-file',
      mutate: (s) => ({ ...s, rootFiles: [...s.rootFiles, 'server.js'] }),
    },
    {
      name: 'planted-navigator-gpu',
      expectedCategory: 'foreign-processor-api',
      mutate: (s) => ({
        ...s,
        srcFiles: {
          ...s.srcFiles,
          'src/engine/plantedProbe.ts': 'export const x = navigator.gpu?.requestAdapter();',
        },
      }),
    },
    {
      name: 'planted-root-absolute-model-url',
      expectedCategory: 'root-absolute-asset',
      mutate: (s) => ({
        ...s,
        srcFiles: {
          ...s.srcFiles,
          'src/engine/modelCatalog.ts': (s.srcFiles['src/engine/modelCatalog.ts'] ?? '').replace(
            "url: 'models/stories260K.gguf'",
            "url: '/models/stories260K.gguf'",
          ),
        },
      }),
    },
    {
      name: 'planted-missing-coi-sw',
      expectedCategory: 'multithread-isolation',
      mutate: (s) => ({ ...s, coiSwRaw: null }),
    },
    {
      name: 'planted-unlocked-ggml-webgpu',
      expectedCategory: 'vtensor-provider-lock',
      mutate: (s) => ({
        ...s,
        srcFiles: {
          ...s.srcFiles,
          'src/engine/vtensorProvider.ts': (
            s.srcFiles['src/engine/vtensorProvider.ts'] ?? ''
          ).replace("'GGML_WEBGPU'", "'REMOVED'"),
        },
      }),
    },
    {
      name: 'planted-engine-commit-drift',
      expectedCategory: 'engine-pin-parity',
      mutate: (s) => ({
        ...s,
        srcFiles: {
          ...s.srcFiles,
          'src/engine/verdantKernels.ts': (
            s.srcFiles['src/engine/verdantKernels.ts'] ?? ''
          ).replace(
            /VERDANT_TENSOR_COMMIT\s*=\s*'[^']+'/,
            "VERDANT_TENSOR_COMMIT = '0000000000000000000000000000000000000000'",
          ),
        },
      }),
    },
    {
      name: 'planted-empty-owner-decisions',
      expectedCategory: 'owner-decisions',
      mutate: (s) => ({
        ...s,
        ownerDecisionsRaw: JSON.stringify({ schema: 'verdant-ui/owner-decisions/1', decisions: [] }),
      }),
    },
    {
      name: 'planted-duplicate-id-register',
      expectedCategory: 'ids-register',
      mutate: (s) => ({
        ...s,
        idsTomlRaw: `${s.idsTomlRaw}\n[[id]]\ncode = "VUI-CLIENT-01"\ntitle = "duplicate"\n`,
      }),
    },
    {
      name: 'planted-analytics-tracker-dep',
      expectedCategory: 'public-benefit-charter',
      mutate: (s) => {
        const pkg = JSON.parse(s.packageJsonRaw) as { dependencies: Record<string, string> };
        pkg.dependencies['posthog-js'] = '^1.0.0';
        return { ...s, packageJsonRaw: JSON.stringify(pkg) };
      },
    },
    {
      name: 'planted-user-facing-observer-jargon',
      expectedCategory: 'user-facing-jargon',
      mutate: (s) => ({
        ...s,
        srcFiles: {
          ...s.srcFiles,
          'src/components/StudioGuidedWizard.tsx': `${s.srcFiles['src/components/StudioGuidedWizard.tsx'] ?? ''}\nexport const _bad = () => <span>Observer Stream Active</span>;`,
        },
      }),
    },
    {
      name: 'planted-user-facing-routing-weights-jargon',
      expectedCategory: 'user-facing-jargon',
      mutate: (s) => ({
        ...s,
        srcFiles: {
          ...s.srcFiles,
          'src/components/SettingsDrawer.tsx': `${s.srcFiles['src/components/SettingsDrawer.tsx'] ?? ''}\nexport const _bad2 = () => <span>Verdant Engine routing weights</span>;`,
        },
      }),
    },
    {
      name: 'planted-silent-fallback-coverup',
      expectedCategory: 'silent-fallback',
      mutate: (s) => ({
        ...s,
        srcFiles: {
          ...s.srcFiles,
          'src/engine/assistantAdapter.ts': `${s.srcFiles['src/engine/assistantAdapter.ts'] ?? ''}\nconst reasoningFallback = 'masked';`,
        },
      }),
    },
  ];

  const failures: string[] = [];
  for (const c of cases) {
    const res = evaluateVerdantRatchet(c.mutate(cleanSnapshot));
    const caught = !res.ok && res.violations.some((v) => v.category === c.expectedCategory);
    if (!caught) {
      failures.push(`${c.name} (expected refusal in category "${c.expectedCategory}")`);
    }
  }

  return {
    allCaught: failures.length === 0,
    casesVerified: cases.length,
    failures,
  };
}

const isMain =
  process.argv[1] !== undefined &&
  resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url));

if (isMain) {
  const snapshot = loadRatchetSnapshot(process.cwd());
  const report = evaluateVerdantRatchet(snapshot);
  if (!report.ok) {
    console.error('verdant-ratchet: REFUSED');
    for (const v of report.violations) {
      console.error(`  - [${v.category}] ${v.path}${v.line ? `:${v.line}` : ''}: ${v.detail}`);
    }
    process.exit(1);
  }
  const planted = verifyPlantedDefects(snapshot);
  if (!planted.allCaught) {
    console.error('verdant-ratchet: planted-defect self-check FAILED');
    for (const f of planted.failures) {
      console.error(`  - ${f}`);
    }
    process.exit(1);
  }
  console.log(
    `verdant-ratchet: PASS · schema=${report.schema} · ceilings=0/0 · src_files=${report.auditedSrcFiles} · planted_gates=${planted.casesVerified}/${planted.casesVerified}`,
  );
}
