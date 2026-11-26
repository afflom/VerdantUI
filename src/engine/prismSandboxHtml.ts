/**
 * Pure HTML document composer for VerdantUI's sandboxed Prism runtime (`<iframe sandbox="allow-scripts" srcdoc>`).
 *
 * Security & UX guarantees:
 * - Strict CSP (`connect-src 'none'`, zero external script/network exfiltration)
 * - Inherited Verdant light/dark base theme tokens (`--v-bg`, `--v-surface`, `--v-border`, `--v-text`,
 *   `--v-muted`, `--v-accent`) + live theme updates over `postMessage` without reloading the iframe
 * - Typed `window.verdant` stream bridge (`emit`, `onInput`, `ask`, `storage.get/set`, `fetch`)
 * - Automatic `window.onerror` & `unhandledrejection` capture routed as OTel error events to the host
 */

import type { VerdantPrismSnapshot } from './prismStudio';

export const VERDANT_SANDBOX_CSP =
  "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; media-src data: blob:; font-src data:; connect-src 'none';";

function escapeClosingTag(src: string, tagName: 'script' | 'style'): string {
  return src.replace(new RegExp(`</${tagName}`, 'gi'), `<\\/${tagName}`);
}

const VERDANT_BASE_STYLES = `
:root {
  --v-bg: #f8f9fa;
  --v-surface: #ffffff;
  --v-border: #e2e8f0;
  --v-text: #0f172a;
  --v-muted: #64748b;
  --v-accent: #0f5132;
  --v-accent-text: #ffffff;
  color-scheme: light;
}
:root.dark {
  --v-bg: #0b0f14;
  --v-surface: #111822;
  --v-border: #1e293b;
  --v-text: #f1f5f9;
  --v-muted: #94a3b8;
  --v-accent: #10b981;
  --v-accent-text: #062116;
  color-scheme: dark;
}
* { box-sizing: border-box; }
html, body {
  margin: 0;
  padding: 0;
  min-height: 100%;
  background: var(--v-bg);
  color: var(--v-text);
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 14px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}
body { padding: 16px; }
.v-card {
  background: var(--v-surface);
  border: 1px solid var(--v-border);
  border-radius: 12px;
  padding: 16px;
}
.v-muted { color: var(--v-muted); }
.v-pill {
  display: inline-flex;
  align-items: center;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  border: 1px solid var(--v-border);
  background: var(--v-surface);
  color: var(--v-text);
}
.v-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  height: 36px;
  padding: 0 14px;
  border-radius: 8px;
  border: 1px solid var(--v-border);
  background: var(--v-surface);
  color: var(--v-text);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 120ms ease, background 120ms ease;
}
.v-btn:hover { opacity: 0.9; }
.v-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.v-btn-primary {
  background: var(--v-accent);
  color: var(--v-accent-text);
  border-color: transparent;
}
.v-input {
  width: 100%;
  padding: 8px 11px;
  border-radius: 8px;
  border: 1px solid var(--v-border);
  background: var(--v-surface);
  color: var(--v-text);
  font: inherit;
  font-size: 13px;
}
.v-input:focus {
  outline: 2px solid var(--v-accent);
  outline-offset: 1px;
}
`;

export function buildPrismSandboxDocument(opts: {
  readonly prismId: string;
  readonly channelId: string;
  readonly snapshot: VerdantPrismSnapshot;
  readonly darkMode: boolean;
}): string {
  const safePrismId = JSON.stringify(opts.prismId);
  const safeChannelId = JSON.stringify(opts.channelId);
  const safeDigest = JSON.stringify(opts.snapshot.digest);
  const safeCss = escapeClosingTag(opts.snapshot.app.css, 'style');
  const safeJs = escapeClosingTag(opts.snapshot.app.js, 'script');
  const rootClass = opts.darkMode ? ' class="dark"' : '';

  const bridgeScript = `
(function() {
  const CHANNEL = ${safeChannelId};
  const PRISM_ID = ${safePrismId};
  const DIGEST = ${safeDigest};
  let rpcSeq = 1;
  const pending = new Map();
  const tokenCallbacks = new Map();
  const inputHandlers = new Set();
  let activeTraceContext = null;
  let activeCallChain = [];

  function post(type, payload) {
    try {
      window.parent.postMessage({ __verdant: true, channelId: CHANNEL, prismId: PRISM_ID, type, payload }, '*');
    } catch (_) {}
  }

  function callHost(method, args, onToken) {
    const reqId = 'rpc_' + (rpcSeq++);
    if (typeof onToken === 'function') tokenCallbacks.set(reqId, onToken);
    return new Promise((resolve, reject) => {
      pending.set(reqId, { resolve, reject });
      post('rpc', {
        reqId,
        method,
        args,
        traceContext: activeTraceContext,
        callChain: activeCallChain,
      });
    });
  }

  window.addEventListener('message', async (ev) => {
    const msg = ev && ev.data;
    if (!msg || !msg.__verdant || msg.channelId !== CHANNEL) return;
    if (msg.type === 'theme') {
      document.documentElement.classList.toggle('dark', Boolean(msg.payload && msg.payload.dark));
      return;
    }
    if (msg.type === 'hot-css') {
      const styleEl = document.getElementById('verdant-user-css');
      if (styleEl) styleEl.textContent = String((msg.payload && msg.payload.css) || '');
      return;
    }
    if (msg.type === 'rpc-token' && msg.payload && msg.payload.reqId) {
      const cb = tokenCallbacks.get(msg.payload.reqId);
      if (cb) { try { cb(msg.payload.text); } catch (_) {} }
      return;
    }
    if (msg.type === 'rpc-result' && msg.payload && msg.payload.reqId) {
      const entry = pending.get(msg.payload.reqId);
      pending.delete(msg.payload.reqId);
      tokenCallbacks.delete(msg.payload.reqId);
      if (!entry) return;
      if (msg.payload.ok) entry.resolve(msg.payload.value);
      else entry.reject(new Error(msg.payload.error || 'Verdant bridge request failed'));
      return;
    }
    if (msg.type === 'dispatch-input' && msg.payload) {
      const { reqId, input, traceContext, callChain } = msg.payload;
      const prevTrace = activeTraceContext;
      const prevChain = activeCallChain;
      activeTraceContext = traceContext || null;
      activeCallChain = Array.isArray(callChain) ? callChain.slice() : [];
      try {
        let lastResult = { ok: true, prismId: PRISM_ID, received: input };
        for (const handler of inputHandlers) {
          const r = await handler(input, traceContext);
          if (r !== undefined) lastResult = r;
        }
        post('dispatch-response', { reqId, ok: true, value: lastResult });
      } catch (err) {
        const message = err && err.message ? err.message : String(err);
        post('dispatch-response', { reqId, ok: false, error: message });
      } finally {
        activeTraceContext = prevTrace;
        activeCallChain = prevChain;
      }
    }
  });

  window.addEventListener('error', (ev) => {
    post('runtime-error', {
      message: (ev && ev.message) || 'Uncaught script error',
      line: ev && ev.lineno ? ev.lineno : null,
      col: ev && ev.colno ? ev.colno : null,
    });
  });

  window.addEventListener('unhandledrejection', (ev) => {
    const r = ev && ev.reason;
    post('runtime-error', {
      message: r && r.message ? r.message : String(r || 'Unhandled promise rejection'),
      line: null,
      col: null,
    });
  });

  window.verdant = Object.freeze({
    prismId: PRISM_ID,
    digest: DIGEST,
    emit(name, body, attributes) {
      post('emit', {
        name: String(name || 'prism.event'),
        body,
        attributes,
        traceContext: activeTraceContext,
      });
    },
    onInput(handler) {
      if (typeof handler !== 'function') return () => {};
      inputHandlers.add(handler);
      return () => { inputHandlers.delete(handler); };
    },
    ask(prompt, opts) {
      return callHost('ask', { prompt: String(prompt || '') }, opts && opts.onToken);
    },
    storage: Object.freeze({
      get(key) { return callHost('storage.get', { key: String(key || '') }); },
      set(key, value) { return callHost('storage.set', { key: String(key || ''), value }); },
    }),
    browser: Object.freeze({
      copyText(text) { return callHost('browser.copyText', { text: String(text || '') }); },
      downloadFile(filename, content, mimeType) {
        return callHost('browser.downloadFile', {
          filename: String(filename || 'export.txt'),
          content: String(content || ''),
          mimeType: String(mimeType || 'text/plain'),
        });
      },
      notify(title, body) {
        return callHost('browser.notify', { title: String(title || ''), body: String(body || '') });
      },
    }),
    ui: Object.freeze({
      getState() { return callHost('ui.getState', {}); },
      callPrism(target, payload) { return callHost('ui.callPrism', { target: String(target || ''), payload }); },
      openPrism(target) { return callHost('ui.openPrism', { target: String(target || '') }); },
    }),
    fetch(url, init, opts) {
      return callHost('fetch', { url: String(url || ''), init: init || {} }, opts && opts.onChunk);
    },
  });

  window.addEventListener('DOMContentLoaded', () => {
    post('ready', { digest: DIGEST });
  }, { once: true });
})();
`;

  return `<!doctype html>
<html lang="en"${rootClass}>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta http-equiv="Content-Security-Policy" content="${VERDANT_SANDBOX_CSP}" />
<style>${VERDANT_BASE_STYLES}</style>
<style id="verdant-user-css">${safeCss}</style>
<script>${bridgeScript}</script>
</head>
<body>
${opts.snapshot.app.html}
<script>
try {
${safeJs}
} catch (err) {
  window.dispatchEvent(new ErrorEvent('error', { message: err && err.message ? err.message : String(err) }));
}
</script>
</body>
</html>`;
}

/**
 * Composes a strictly sandboxed HTML document for a real `prism/holo/6` `HOLOVIEW`
 * (`prism.design/1` `index.html`, `prism.css`, `prism-ui.js`, `app.js`), replacing the
 * network-bound `prism.js` fetch shim with a zero-origin `postMessage` bridge exposing
 * `globalThis.prism.invoke(line)` backed by the verified WASM guest or instant draft evaluator.
 */
export function buildHoloviewSandboxDocument(opts: {
  readonly prismId: string;
  readonly channelId: string;
  readonly digest: string;
  readonly indexHtml: string;
  readonly prismCss: string;
  readonly prismUiJs: string;
  readonly appJs: string;
  readonly darkMode: boolean;
}): string {
  const safePrismId = JSON.stringify(opts.prismId);
  const safeChannelId = JSON.stringify(opts.channelId);
  const safeDigest = JSON.stringify(opts.digest);
  const safeCss = escapeClosingTag(opts.prismCss || '', 'style');
  const safeUiJs = escapeClosingTag(opts.prismUiJs || '', 'script');
  const safeAppJs = escapeClosingTag(opts.appJs || '', 'script');

  const prismBridgeScript = `
(function () {
  const CHANNEL_ID = ${safeChannelId};
  const PRISM_ID = ${safePrismId};
  const DIGEST = ${safeDigest};
  const pendingInvoke = new Map();
  const pendingRpc = new Map();
  const tokenCallbacks = new Map();
  const inputHandlers = new Set();
  let activeTraceContext = null;
  let activeCallChain = [];
  let seq = 1;

  function post(type, payload) {
    window.parent.postMessage({ __verdant: true, channelId: CHANNEL_ID, prismId: PRISM_ID, type, payload }, '*');
  }

  function callHost(method, args, onToken) {
    const reqId = 'rpc_' + (seq++);
    if (typeof onToken === 'function') tokenCallbacks.set(reqId, onToken);
    return new Promise((resolve, reject) => {
      pendingRpc.set(reqId, { resolve, reject });
      post('rpc', {
        reqId,
        method,
        args,
        traceContext: activeTraceContext,
        callChain: activeCallChain,
      });
    });
  }

  function applyTheme(dark) {
    document.documentElement.classList.toggle('dark', !!dark);
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
  }

  window.addEventListener('error', (ev) => {
    post('runtime-error', {
      message: ev.message || 'Uncaught runtime error',
      line: ev.lineno || null,
      col: ev.colno || null,
    });
  });

  window.addEventListener('unhandledrejection', (ev) => {
    const r = ev.reason;
    post('runtime-error', {
      message: (r && r.message) ? String(r.message) : String(r || 'Unhandled rejection'),
      line: null,
      col: null,
    });
  });

  window.addEventListener('message', async (ev) => {
    const msg = ev.data;
    if (!msg || !msg.__verdant || msg.channelId !== CHANNEL_ID) return;
    if (msg.type === 'theme' && msg.payload) {
      applyTheme(!!msg.payload.dark);
    } else if (msg.type === 'hot-css' && msg.payload && typeof msg.payload.css === 'string') {
      const styleEl = document.getElementById('verdant-user-css');
      if (styleEl) styleEl.textContent = msg.payload.css;
    } else if (msg.type === 'prism-invoke-result' && msg.payload) {
      const w = pendingInvoke.get(msg.payload.reqId);
      if (!w) return;
      pendingInvoke.delete(msg.payload.reqId);
      if (msg.payload.ok) w.resolve(String(msg.payload.value ?? ''));
      else w.reject(new Error(String(msg.payload.error || 'Model invocation failed')));
    } else if (msg.type === 'rpc-token' && msg.payload && msg.payload.reqId) {
      const cb = tokenCallbacks.get(msg.payload.reqId);
      if (cb) { try { cb(msg.payload.text); } catch (_) {} }
    } else if (msg.type === 'rpc-result' && msg.payload && msg.payload.reqId) {
      const entry = pendingRpc.get(msg.payload.reqId);
      pendingRpc.delete(msg.payload.reqId);
      tokenCallbacks.delete(msg.payload.reqId);
      if (!entry) return;
      if (msg.payload.ok) entry.resolve(msg.payload.value);
      else entry.reject(new Error(msg.payload.error || 'Verdant bridge request failed'));
    } else if (msg.type === 'dispatch-input' && msg.payload) {
      const { reqId, input, traceContext, callChain } = msg.payload;
      const prevTrace = activeTraceContext;
      const prevChain = activeCallChain;
      activeTraceContext = traceContext || null;
      activeCallChain = Array.isArray(callChain) ? callChain.slice() : [];
      try {
        let lastResult = { ok: true, prismId: PRISM_ID, received: input };
        for (const handler of inputHandlers) {
          const r = await handler(input, traceContext);
          if (r !== undefined) lastResult = r;
        }
        post('dispatch-response', { reqId, ok: true, value: lastResult });
      } catch (err) {
        post('dispatch-response', { reqId, ok: false, error: err && err.message ? err.message : String(err) });
      } finally {
        activeTraceContext = prevTrace;
        activeCallChain = prevChain;
      }
    }
  });

  const invoke = (request) => new Promise((resolve, reject) => {
    const reqId = 'inv_' + (seq++);
    pendingInvoke.set(reqId, { resolve, reject });
    post('prism-invoke', { reqId, request: String(request ?? '') });
  });

  globalThis.prism = Object.freeze({
    invoke,
    requestMaximum: 4096,
    responseMaximum: 8192,
  });

  window.verdant = Object.freeze({
    prismId: PRISM_ID,
    digest: DIGEST,
    emit(name, body, attributes) {
      post('emit', {
        name: String(name || 'prism.event'),
        body,
        attributes,
        traceContext: activeTraceContext,
      });
    },
    onInput(handler) {
      if (typeof handler !== 'function') return () => {};
      inputHandlers.add(handler);
      return () => { inputHandlers.delete(handler); };
    },
    ask(prompt, opts) {
      return callHost('ask', { prompt: String(prompt || '') }, opts && opts.onToken);
    },
    storage: Object.freeze({
      get(key) { return callHost('storage.get', { key: String(key || '') }); },
      set(key, value) { return callHost('storage.set', { key: String(key || ''), value }); },
    }),
    browser: Object.freeze({
      copyText(text) { return callHost('browser.copyText', { text: String(text || '') }); },
      downloadFile(filename, content, mimeType) {
        return callHost('browser.downloadFile', {
          filename: String(filename || 'export.txt'),
          content: String(content || ''),
          mimeType: String(mimeType || 'text/plain'),
        });
      },
      notify(title, body) {
        return callHost('browser.notify', { title: String(title || ''), body: String(body || '') });
      },
    }),
    downloadFile(filename, content, mimeType) {
      return callHost('browser.downloadFile', {
        filename: String(filename || 'export.txt'),
        content: String(content || ''),
        mimeType: String(mimeType || 'text/plain'),
      });
    },
    ui: Object.freeze({
      getState() { return callHost('ui.getState', {}); },
      callPrism(target, payload) { return callHost('ui.callPrism', { target: String(target || ''), payload }); },
      openPrism(target) { return callHost('ui.openPrism', { target: String(target || '') }); },
    }),
    fetch(url, init, opts) {
      return callHost('fetch', { url: String(url || ''), init: init || {} }, opts && opts.onChunk);
    },
  });

  window.addEventListener('DOMContentLoaded', () => {
    post('ready', { digest: DIGEST });
  }, { once: true });
})();
`;

  // Preserve any inline <style> blocks in <head> alongside compiled/authored CSS
  const headMatch = opts.indexHtml.match(/<head[^>]*>([\s\S]*?)<\/head>/i);
  const inlineHeadStyles = headMatch
    ? Array.from(headMatch[1].matchAll(/<style[^>]*>([\s\S]*?)<\/style>/gi))
        .map((m) => escapeClosingTag(m[1], 'style'))
        .join('\n')
    : '';

  // Extract body inner HTML if full HTML doc was provided, stripping external prism/authored asset tags
  const bodyMatch = opts.indexHtml.match(/<body[^>]*>([\s\S]*?)<\/body>/i);
  const rawBody = bodyMatch ? bodyMatch[1] : opts.indexHtml;
  const cleanedBody = rawBody
    .replace(/<script[^>]+src=["'](?:\.\/)?(?:prism\.js|prism-ui\.js|app\.js)["'][^>]*><\/script>/gi, '')
    .replace(/<link[^>]+href=["'](?:\.\/)?(?:prism\.css|styles\.css)["'][^>]*\/?>/gi, '');

  const htmlAttr = opts.darkMode ? ' class="dark" data-theme="dark"' : ' data-theme="light"';

  return `<!doctype html>
<html lang="en"${htmlAttr}>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta http-equiv="Content-Security-Policy" content="${VERDANT_SANDBOX_CSP}" />
<style id="verdant-user-css">${safeCss}${inlineHeadStyles ? `\n${inlineHeadStyles}` : ''}</style>
<script>${prismBridgeScript}</script>
${safeUiJs ? `<script>${safeUiJs}</script>` : ''}
</head>
<body>
${cleanedBody}
<script>
(function () {
  try {
${safeAppJs}
  } catch (err) {
    window.dispatchEvent(new ErrorEvent('error', { message: err && err.message ? err.message : String(err) }));
  }
})();
</script>
</body>
</html>`;
}
