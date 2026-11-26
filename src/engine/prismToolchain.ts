/**
 * Browser-Native Prism Toolchain Loader (`src/engine/prismToolchain.ts`)
 *
 * Lazy-loads the BLAKE3-pinned `prism-code@0.11.5` WASM toolchain (LexLean front end,
 * Lean 4.32.2 kernel + base library, `evaluator.wasm`, and canonical template foundation)
 * from relative `./prism-toolchain/...` URLs and caches them permanently in OPFS via
 * `kappa-opfs.mjs`. Every byte is verified against its pinned BLAKE3 κ-label before use.
 */

import { opfsStore } from '@prism-code-internal/journey/kappa-opfs.mjs';
import { MODEL_KERNEL_FILES } from '@prism-code-internal/journey/model-kernel-pins.mjs';
import { openKernelFrom } from '@prism-code-internal/journey/prove-core.mjs';
import { admitBytes, admitModule, memoryStore, type BlobStorePort } from '@prism-code-internal/journey/toolchain.mjs';

export const LEXLEAN_PIN = Object.freeze({
  kappa: 'blake3:4f302c18e02a46b0d015c57bd389a0754b301ec876aa7a637f786c4562274ecb',
  relPath: 'prism-toolchain/4f/302c18e02a46b0d015c57bd389a0754b301ec876aa7a637f786c4562274ecb',
  encoding: 'gzip' as const,
});

function pinnedRelPath(kappaLabel: string): string {
  const hex = kappaLabel.replace(/^blake3:/, '');
  return `prism-toolchain/${hex.slice(0, 2)}/${hex.slice(2)}`;
}

export interface LoadedPrismToolchain {
  readonly lexlean: WebAssembly.Module;
  readonly evaluator: Uint8Array;
  readonly kernel: unknown;
  readonly foundationFiles: Readonly<Record<string, string>>;
}

let cachedToolchain: Promise<LoadedPrismToolchain> | null = null;

async function openBlobStore(): Promise<BlobStorePort> {
  try {
    const opfs = await opfsStore();
    if (!opfs.off) {
      return {
        get: (label: string) => opfs.get(label),
        put: (bytes: Uint8Array) => opfs.put(bytes),
      };
    }
  } catch {
    // Fallback to in-memory store in private browsing or restricted contexts
  }
  return memoryStore();
}

/**
 * Loads and verifies the complete browser-native Prism WASM toolchain once per worker,
 * using OPFS content-addressed BLAKE3 caching across sessions.
 */
export function loadPrismToolchain(
  baseUrl: string,
  onProgress?: (detail: string) => void,
): Promise<LoadedPrismToolchain> {
  if (cachedToolchain) return cachedToolchain;

  const urlOf = (rel: string) => new URL(rel, baseUrl).href;

  cachedToolchain = (async () => {
    onProgress?.('Opening OPFS content-addressed cache…');
    const store = await openBlobStore();

    onProgress?.('Loading LexLean WASM & evaluator guest…');
    const [lexleanRes, evaluatorResp, foundationResp] = await Promise.all([
      admitModule(
        { kappa: LEXLEAN_PIN.kappa, url: urlOf(LEXLEAN_PIN.relPath), encoding: 'gzip' },
        { store, fetch: globalThis.fetch.bind(globalThis), what: 'LexLean WASM' },
      ),
      globalThis.fetch(urlOf('prism-toolchain/evaluator.wasm')),
      globalThis.fetch(urlOf('prism-toolchain/template-foundation.json')),
    ]);

    if (!evaluatorResp.ok) {
      throw new Error(`Failed to fetch evaluator.wasm (HTTP ${evaluatorResp.status}).`);
    }
    if (!foundationResp.ok) {
      throw new Error(`Failed to fetch template-foundation.json (HTTP ${foundationResp.status}).`);
    }

    const evaluator = new Uint8Array(await evaluatorResp.arrayBuffer());
    const foundationJson = (await foundationResp.json()) as {
      files: Record<string, string>;
    };

    onProgress?.('Initializing Lean 4.32.2 WASM kernel…');
    const loaderPin = MODEL_KERNEL_FILES['lean-kernel.mjs'];
    const kernelPin = MODEL_KERNEL_FILES['lean-kernel.wasm.gz'];
    const libraryPin = MODEL_KERNEL_FILES['base.ndjson.gz'];

    const kernel = await openKernelFrom(
      {
        loader: {
          kappa: loaderPin.kappa,
          url: urlOf(pinnedRelPath(loaderPin.kappa)),
          encoding: loaderPin.encoding,
        },
        kernel: {
          kappa: kernelPin.kappa,
          url: urlOf(pinnedRelPath(kernelPin.kappa)),
          encoding: kernelPin.encoding,
        },
        library: {
          kappa: libraryPin.kappa,
          url: urlOf(pinnedRelPath(libraryPin.kappa)),
          encoding: libraryPin.encoding,
        },
      },
      { store, fetch: globalThis.fetch.bind(globalThis), resettable: true },
    );

    return {
      lexlean: lexleanRes.module,
      evaluator,
      kernel,
      foundationFiles: foundationJson.files,
    };
  })();

  cachedToolchain.catch(() => {
    cachedToolchain = null;
  });
  return cachedToolchain;
}

export { admitBytes };
