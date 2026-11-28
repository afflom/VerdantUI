#!/usr/bin/env node
/**
 * Zero-Dependency Automated Chrome CDP Browser Evaluation Harness (`scripts/eval-browser.ts`)
 *
 * Honesty level: `open` (R2 — measured and reported honestly, never falsely asserted).
 * Dependencies: ZERO external packages (`R6`) — uses Node 22 built-in `fetch` and `WebSocket`
 * directly against Chrome DevTools Protocol (`--cdp http://127.0.0.1:9222`).
 *
 * Usage:
 *   npm run eval:browser -- --url http://127.0.0.1:4173/ --cdp http://127.0.0.1:9222
 *   npm run eval:browser -- --url https://afflom.github.io/VerdantUI/ --cdp http://127.0.0.1:9222
 *   npm run eval:browser -- --validate-suite   (offline validation of held-out-prompts.json)
 */

import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

interface HeldOutSuite {
  readonly schema: string;
  readonly honestyLevel: string;
  readonly holoRefusalProbes: ReadonlyArray<{
    readonly id: string;
    readonly command: string;
    readonly expectedResponse: string;
  }>;
  readonly chatPrompts: ReadonlyArray<{
    readonly id: string;
    readonly category: string;
    readonly prompt: string;
    readonly mediaType?: string;
    readonly refinePrompt?: string;
  }>;
}

interface CdpMessage {
  readonly id?: number;
  readonly method?: string;
  readonly params?: Record<string, unknown>;
  readonly result?: Record<string, unknown>;
  readonly error?: { readonly code: number; readonly message: string };
}

class MinimalCdpClient {
  private ws: WebSocket | null = null;
  private seq = 1;
  private readonly pending = new Map<
    number,
    { resolve: (v: Record<string, unknown>) => void; reject: (e: Error) => void }
  >();
  private readonly eventListeners = new Map<string, Set<(params: Record<string, unknown>) => void>>();

  async connect(wsUrl: string): Promise<void> {
    await new Promise<void>((res, rej) => {
      const ws = new WebSocket(wsUrl);
      this.ws = ws;
      ws.addEventListener('open', () => res(), { once: true });
      ws.addEventListener('error', (ev) => rej(new Error(`CDP WebSocket error: ${String(ev)}`)), { once: true });
      ws.addEventListener('message', (ev) => {
        const msg = JSON.parse(String(ev.data)) as CdpMessage;
        if (typeof msg.id === 'number') {
          const waiter = this.pending.get(msg.id);
          if (waiter) {
            this.pending.delete(msg.id);
            if (msg.error) waiter.reject(new Error(`CDP ${msg.error.code}: ${msg.error.message}`));
            else waiter.resolve(msg.result ?? {});
          }
        } else if (msg.method) {
          const listeners = this.eventListeners.get(msg.method);
          if (listeners) {
            for (const fn of listeners) fn(msg.params ?? {});
          }
        }
      });
    });
  }

  on(method: string, fn: (params: Record<string, unknown>) => void): void {
    const set = this.eventListeners.get(method) ?? new Set();
    set.add(fn);
    this.eventListeners.set(method, set);
  }

  async send(method: string, params: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
    if (!this.ws) throw new Error('refused: CDP socket not connected');
    const id = this.seq++;
    return new Promise((res, rej) => {
      this.pending.set(id, { resolve: res, reject: rej });
      this.ws!.send(JSON.stringify({ id, method, params }));
    });
  }

  async evalJson<T>(expression: string): Promise<T> {
    const out = await this.send('Runtime.evaluate', {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    const ex = out.exceptionDetails as { text?: string; exception?: { description?: string } } | undefined;
    if (ex) {
      throw new Error(`Browser eval error: ${ex.exception?.description ?? ex.text ?? 'unknown'}`);
    }
    const result = out.result as { value?: T } | undefined;
    return result?.value as T;
  }

  close(): void {
    this.ws?.close();
    this.ws = null;
  }
}

function parseArg(name: string, fallback: string): string {
  const idx = process.argv.indexOf(name);
  return idx >= 0 && process.argv[idx + 1] ? process.argv[idx + 1]! : fallback;
}

function measureLocalDistJsKiB(cwd: string): number | null {
  const assetsDir = join(resolve(cwd), 'dist/assets');
  if (!existsSync(assetsDir)) return null;
  let total = 0;
  for (const f of readdirSync(assetsDir)) {
    if (f.endsWith('.js')) total += statSync(join(assetsDir, f)).size;
  }
  return Math.round((total / 1024) * 10) / 10;
}

export function loadHeldOutSuite(cwd = process.cwd()): HeldOutSuite {
  const suite = JSON.parse(
    readFileSync(join(resolve(cwd), 'evals/held-out-prompts.json'), 'utf8'),
  ) as HeldOutSuite;
  if (suite.schema !== 'verdant-ui/held-out-evals/1' || suite.honestyLevel !== 'open') {
    throw new Error('refused: invalid evals/held-out-prompts.json schema or honestyLevel');
  }
  if (!Array.isArray(suite.holoRefusalProbes) || suite.holoRefusalProbes.length < 4) {
    throw new Error('refused: evals/held-out-prompts.json missing holoRefusalProbes');
  }
  return suite;
}

async function main(): Promise<void> {
  const cwd = process.cwd();
  const suite = loadHeldOutSuite(cwd);
  if (process.argv.includes('--validate-suite')) {
    console.log(
      `eval-browser: SUITE VALID · schema=${suite.schema} · refusalProbes=${suite.holoRefusalProbes.length} · chatPrompts=${suite.chatPrompts.length}`,
    );
    return;
  }

  const targetUrl = parseArg('--url', 'https://afflom.github.io/VerdantUI/');
  const cdpBase = parseArg('--cdp', 'http://127.0.0.1:9222').replace(/\/+$/, '');

  let targets: Array<{ id: string; type: string; url: string; webSocketDebuggerUrl?: string }>;
  try {
    const res = await fetch(`${cdpBase}/json/list`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    targets = (await res.json()) as typeof targets;
  } catch (err) {
    console.error(
      `eval-browser: REFUSED — cannot reach Chrome DevTools Protocol at ${cdpBase} (${err instanceof Error ? err.message : String(err)}). Launch Chrome with --remote-debugging-port=9222 and retry.`,
    );
    process.exit(2);
  }

  // Open a dedicated clean page target (or reuse an existing matching page)
  let wsUrl = targets.find((t) => t.type === 'page' && t.url.startsWith(targetUrl) && t.webSocketDebuggerUrl)
    ?.webSocketDebuggerUrl;
  if (!wsUrl) {
    const created = await fetch(`${cdpBase}/json/new?${encodeURIComponent(targetUrl)}`, { method: 'PUT' });
    if (created.ok) {
      const info = (await created.json()) as { webSocketDebuggerUrl?: string };
      wsUrl = info.webSocketDebuggerUrl;
    } else {
      wsUrl = targets.find((t) => t.type === 'page' && t.webSocketDebuggerUrl)?.webSocketDebuggerUrl;
    }
  }
  if (!wsUrl) {
    console.error('eval-browser: REFUSED — no inspectable Chrome page WebSocket target available');
    process.exit(2);
  }

  const cdp = new MinimalCdpClient();
  const jsErrors: string[] = [];
  const httpErrors: string[] = [];
  const targetOrigin = new URL(targetUrl).origin;

  await cdp.connect(wsUrl);
  try {
    await cdp.send('Runtime.enable');
    await cdp.send('Network.enable');
    await cdp.send('Page.enable');

    cdp.on('Runtime.exceptionThrown', (params) => {
      const details = params.exceptionDetails as { text?: string; exception?: { description?: string } } | undefined;
      jsErrors.push(details?.exception?.description ?? details?.text ?? 'uncaught exception');
    });
    cdp.on('Network.responseReceived', (params) => {
      const resp = params.response as { url?: string; status?: number } | undefined;
      if (resp?.url && typeof resp.status === 'number' && resp.status >= 400 && resp.url.startsWith(targetOrigin)) {
        httpErrors.push(`${resp.status} ${resp.url}`);
      }
    });

    await cdp.send('Page.navigate', { url: targetUrl });

    // Wait up to 20s for window.__VERDANT_OBSERVER__ and verified verdant-ui.holo
    const bootStart = performance.now();
    let ready = false;
    while (performance.now() - bootStart < 20_000) {
      ready = await cdp.evalJson<boolean>(
        `Boolean(window.__VERDANT_OBSERVER__ && window.__VERDANT_OBSERVER__.holoStatus?.()?.verified === true)`,
      );
      if (ready) break;
      await new Promise((r) => setTimeout(r, 250));
    }
    const bootToHoloVerifiedMs = Math.round(performance.now() - bootStart);

    // 1. Live holo status & refusal probe suite
    const holoStatus = await cdp.evalJson<Record<string, unknown>>(
      `window.__VERDANT_OBSERVER__ ? window.__VERDANT_OBSERVER__.holoStatus() : null`,
    );

    const refusalResults: Array<{ id: string; passed: boolean; actual: string; latencyUs: number }> = [];
    for (const probe of suite.holoRefusalProbes) {
      const res = await cdp.evalJson<{ raw: string; refused: boolean; latencyUs: number }>(
        `window.__VERDANT_OBSERVER__.holoDispatch(${JSON.stringify(probe.command)})`,
      );
      refusalResults.push({
        id: probe.id,
        passed: res.refused === true && res.raw === probe.expectedResponse,
        actual: res.raw,
        latencyUs: res.latencyUs,
      });
    }

    // 2. Live GUI vs Observer tree parity
    const parity = await cdp.evalJson<{ equal: boolean; gui: string; obs: string }>(`(async () => {
      const g = await window.__VERDANT_OBSERVER__.holoDispatch("verdant\\tgui\\ttree");
      const o = await window.__VERDANT_OBSERVER__.holoDispatch("verdant\\tobserver\\tperceive");
      return { equal: !g.refused && !o.refused && g.raw === o.raw, gui: g.raw, obs: o.raw };
    })()`);

    // 3. Live consent toggle paused -> accepted (zero Splash deadlock)
    const consentCheck = await cdp.evalJson<{ ok: boolean; afterPaused: string; afterAccepted: string }>(`(async () => {
      const orig = window.__VERDANT_OBSERVER__.perceive().ui.streamingConsent;
      await window.__VERDANT_OBSERVER__.enact({ type: 'set_streaming_consent', consent: 'paused' });
      const afterPaused = window.__VERDANT_OBSERVER__.perceive().ui.streamingConsent;
      await window.__VERDANT_OBSERVER__.enact({ type: 'set_streaming_consent', consent: 'accepted' });
      const afterAccepted = window.__VERDANT_OBSERVER__.perceive().ui.streamingConsent;
      if (orig === 'paused') {
        await window.__VERDANT_OBSERVER__.enact({ type: 'set_streaming_consent', consent: 'paused' });
      }
      return { ok: afterPaused === 'paused' && afterAccepted === 'accepted', afterPaused, afterAccepted };
    })()`);

    // 4. Live Studio create -> refine -> perceive -> delete cleanup
    const studioRoundTrip = await cdp.evalJson<{ ok: boolean; refineMs: number; detail: string }>(`(async () => {
      const c = await window.__VERDANT_OBSERVER__.enact({
        type: 'create_prism',
        prompt: 'Build a monthly budget tracker utility',
        mediaType: 'tool-organ'
      });
      if (!c.ok || !c.result?.prismId) return { ok: false, refineMs: 0, detail: 'create failed: ' + (c.error || '') };
      const pid = c.result.prismId;
      const t0 = performance.now();
      const r = await window.__VERDANT_OBSERVER__.enact({
        type: 'refine_prism',
        prismId: pid,
        instruction: 'Add expense $19 coffee'
      });
      const refineMs = Math.round(performance.now() - t0);
      await window.__VERDANT_OBSERVER__.enact({ type: 'delete_prism', prismId: pid });
      return { ok: Boolean(r.ok), refineMs, detail: r.ok ? 'ok' : String(r.error || 'refine failed') };
    })()`);

    // 5. Observer diagnose & full W3C OTLP JSON export
    const diagnosis = await cdp.evalJson<{ status: string; overallHealthQ10: number; findings: unknown[] }>(
      `window.__VERDANT_OBSERVER__.diagnose()`,
    );
    const spans = await cdp.evalJson<Array<{ name: string; status: string; durationMs?: number }>>(
      `window.__VERDANT_OBSERVER__.recentSpans()`,
    );
    const otlp = await cdp.evalJson<Record<string, unknown>>(`window.__VERDANT_OBSERVER__.otlpJson()`);
    const chronology = await cdp.evalJson<{ coherenceQ10: number }>(`window.__VERDANT_OBSERVER__.chronology()`);

    const otelErrorSpans = spans.filter((s) => s.status === 'error');
    const avgHoloUs =
      refusalResults.length > 0
        ? Math.round(refusalResults.reduce((a, b) => a + b.latencyUs, 0) / refusalResults.length)
        : 0;

    const hardGates = {
      zero_console_js_errors: jsErrors.length === 0,
      zero_http_4xx_5xx: httpErrors.length === 0,
      holo_verified: Boolean(holoStatus && holoStatus.verified === true),
      holo_refusals_live: refusalResults.every((r) => r.passed),
      gui_observer_parity_live: parity.equal,
      consent_toggle_no_deadlock: consentCheck.ok,
      studio_create_refine_cleanup: studioRoundTrip.ok,
      observer_diagnose_not_degraded: diagnosis.status !== 'degraded',
    };

    const allHardPassed = Object.values(hardGates).every(Boolean);

    const report = {
      schema: 'verdant-ui/browser-eval-report/1',
      honestyLevel: 'open',
      evaluatedAt: new Date().toISOString(),
      targetUrl,
      allHardPassed,
      hardGates,
      openMeasurements: {
        bootToHoloVerifiedMs,
        holoDispatchUsAvg: avgHoloUs,
        studioDeterministicRefineMs: studioRoundTrip.refineMs,
        otelSpanCount: spans.length,
        otelErrorSpanCount: otelErrorSpans.length,
        coherenceQ10: chronology.coherenceQ10,
        overallHealthQ10: diagnosis.overallHealthQ10,
        distJsKiB: measureLocalDistJsKiB(cwd),
      },
      refusalResults,
      jsErrors,
      httpErrors,
      diagnosis,
    };

    const reportPath = join(resolve(cwd), 'evals/last-report.json');
    const otlpPath = join(resolve(cwd), 'evals/last-otlp.json');
    mkdirSync(dirname(reportPath), { recursive: true });
    writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
    writeFileSync(otlpPath, `${JSON.stringify(otlp, null, 2)}\n`, 'utf8');

    console.log(`\n| Gate / Metric | Value |`);
    console.log(`| --- | --- |`);
    for (const [k, v] of Object.entries(hardGates)) {
      console.log(`| \`${k}\` | ${v ? 'PASS' : 'FAIL'} |`);
    }
    for (const [k, v] of Object.entries(report.openMeasurements)) {
      console.log(`| \`open:${k}\` | ${String(v)} |`);
    }
    console.log(`\nWrote ${reportPath} and ${otlpPath}`);

    if (!allHardPassed) process.exit(1);
  } finally {
    cdp.close();
  }
}

const isMain =
  process.argv[1] !== undefined &&
  resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url));

if (isMain) {
  main().catch((err) => {
    console.error(`eval-browser: ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
  });
}
