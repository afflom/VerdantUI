/**
 * VerdantUI Cross-Origin Isolation Service Worker (COOP/COEP).
 *
 * Enables `window.crossOriginIsolated === true` and `SharedArrayBuffer` on static hosts
 * without custom HTTP response headers (such as GitHub Pages), unlocking VerdantUI's
 * zero-copy multi-threaded WASM Relaxed-SIMD VNNI thread pool (`VTensorThreadPool`).
 *
 * Design guarantees:
 * - Zero overhead in dev/preview or hosts already sending COOP/COEP headers (complete no-op).
 * - Only intercepts same-origin requests so multi-GB Hugging Face GGUF downloads and HTTP
 *   Range streams bypass the service worker completely.
 * - Uses `Cross-Origin-Embedder-Policy: credentialless` by default with automatic one-shot
 *   fallback to `require-corp` for browsers that only recognize `require-corp`, bounded by a
 *   strict `sessionStorage` reload guard so it can never loop.
 */
(() => {
  const COEP_MODE_CREDENTIALLESS = 'credentialless';
  const COEP_MODE_REQUIRE_CORP = 'require-corp';

  if (typeof window === 'undefined') {
    let coepMode = COEP_MODE_CREDENTIALLESS;

    self.addEventListener('install', () => {
      self.skipWaiting();
    });

    self.addEventListener('activate', (event) => {
      event.waitUntil(self.clients.claim());
    });

    self.addEventListener('message', (event) => {
      if (!event.data || typeof event.data !== 'object') return;
      if (event.data.type === 'verdant-coi-set-mode') {
        coepMode =
          event.data.mode === COEP_MODE_REQUIRE_CORP
            ? COEP_MODE_REQUIRE_CORP
            : COEP_MODE_CREDENTIALLESS;
      }
    });

    self.addEventListener('fetch', (event) => {
      const request = event.request;
      if (request.cache === 'only-if-cached' && request.mode !== 'same-origin') {
        return;
      }

      const url = new URL(request.url);
      // Never proxy cross-origin requests (multi-GB Hugging Face GGUF streams, CDN fonts, etc.).
      if (url.origin !== self.location.origin) {
        return;
      }

      const fetchRequest =
        coepMode === COEP_MODE_CREDENTIALLESS && request.mode === 'no-cors'
          ? new Request(request, { credentials: 'omit' })
          : request;

      event.respondWith(
        fetch(fetchRequest).then((response) => {
          if (response.status === 0) {
            return response;
          }
          const headers = new Headers(response.headers);
          headers.set('Cross-Origin-Opener-Policy', 'same-origin');
          headers.set('Cross-Origin-Embedder-Policy', coepMode);
          headers.set('Cross-Origin-Resource-Policy', 'cross-origin');
          return new Response(response.body, {
            status: response.status,
            statusText: response.statusText,
            headers,
          });
        }),
      );
    });
    return;
  }

  // Window bootstrap path
  const STORAGE_KEY = 'verdant-ui:coi-attempts:v1';

  if (window.crossOriginIsolated) {
    try {
      window.sessionStorage.removeItem(STORAGE_KEY);
    } catch {
      // Ignore storage access errors in restricted contexts
    }
    return;
  }

  if (!window.isSecureContext || !('serviceWorker' in navigator)) {
    return;
  }

  let attempts = 0;
  try {
    attempts = Number(window.sessionStorage.getItem(STORAGE_KEY) || '0') || 0;
  } catch {
    attempts = 0;
  }

  // Strict reload cap: attempt 0 -> credentialless, attempt 1 -> require-corp fallback, >= 2 -> stop.
  if (attempts >= 2) {
    return;
  }

  const currentScript = document.currentScript;
  const swUrl =
    currentScript && 'src' in currentScript && currentScript.src
      ? currentScript.src
      : new URL('./coi-serviceworker.js', document.baseURI || window.location.href).href;

  const triggerReload = (nextAttempt) => {
    window.__VERDANT_COI_RELOADING = true;
    window.__VERDANT_COI_PENDING = false;
    try {
      if (typeof BroadcastChannel === 'function') {
        const ch = new BroadcastChannel('verdant-ui-opfs-v1');
        ch.postMessage({ type: 'release-opfs', senderId: 'coi-bootstrap', modelId: null, ts: Date.now() });
        ch.close();
      }
    } catch {
      // Ignore restricted BroadcastChannel
    }
    try {
      window.sessionStorage.setItem(STORAGE_KEY, String(nextAttempt));
    } catch {
      // Ignore storage errors
    }
    window.location.reload();
  };

  window.__VERDANT_COI_PENDING = true;

  navigator.serviceWorker
    .register(swUrl)
    .then((registration) => {
      const targetMode =
        attempts === 1 ? COEP_MODE_REQUIRE_CORP : COEP_MODE_CREDENTIALLESS;

      const activateAndReload = (worker) => {
        if (!worker) {
          window.__VERDANT_COI_PENDING = false;
          return;
        }
        worker.postMessage({ type: 'verdant-coi-set-mode', mode: targetMode });
        triggerReload(attempts + 1);
      };

      if (registration.active && navigator.serviceWorker.controller) {
        // Already controlled, yet crossOriginIsolated is still false -> try fallback COEP mode once.
        activateAndReload(registration.active);
        return;
      }

      if (registration.active) {
        activateAndReload(registration.active);
        return;
      }

      const installing = registration.installing || registration.waiting;
      if (installing) {
        installing.addEventListener('statechange', () => {
          if (installing.state === 'activated') {
            activateAndReload(installing);
          } else if (installing.state === 'redundant') {
            window.__VERDANT_COI_PENDING = false;
          }
        });
      } else {
        window.__VERDANT_COI_PENDING = false;
      }
    })
    .catch(() => {
      // Non-fatal: runtime degrades cleanly if browser policy forbids service workers.
      window.__VERDANT_COI_PENDING = false;
    });
})();
