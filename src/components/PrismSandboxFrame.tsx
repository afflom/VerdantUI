/**
 * Strictly sandboxed Prism runtime host (`<iframe sandbox="allow-scripts" srcdoc>`).
 *
 * Shared by both Verdant Studio's Real-Time Preview and Published App view:
 * - Zero origin / zero parent DOM access (COOP/COEP-safe `srcdoc`)
 * - Live light/dark theme sync over `postMessage` without reloading the iframe
 * - Mediates `window.verdant` (`emit`, `onInput`, `ask`, `storage`, `fetch`) through
 *   `ObserverStreamBus` + `PrismBridgeRuntime` (`ContainerGrantMatrix` FK-07)
 * - Registers itself as the high-priority live endpoint on `ObserverStreamBus` while mounted
 * - Surfaces sandbox runtime errors directly for Studio's one-click "Fix with VerdantUI" loop
 */

import { useEffect, useMemo, useRef, type FC } from 'react';
import type { ObserverStreamBus, OtelTraceContext } from '../engine/observerStreamBus';
import type { PrismBridgeRuntime } from '../engine/prismHostPool';
import { buildHoloviewSandboxDocument, buildPrismSandboxDocument } from '../engine/prismSandboxHtml';
import { sha256HexSync, type VerdantPrismSnapshot } from '../engine/prismStudio';

export interface PrismRuntimeError {
  readonly message: string;
  readonly line: number | null;
  readonly col: number | null;
  readonly timestampMs: number;
}

export interface HoloviewMountConfig {
  readonly indexHtml: string;
  readonly prismCss: string;
  readonly prismUiJs: string;
  readonly appJs: string;
  readonly onInvoke: (requestLine: string, traceContext?: OtelTraceContext) => Promise<string>;
}

interface PrismSandboxFrameProps {
  prismId: string;
  snapshot: VerdantPrismSnapshot;
  /** When set, mounts a compiled `prism/holo/6` HOLOVIEW with `globalThis.prism.invoke`. */
  holoview?: HoloviewMountConfig | null;
  darkMode: boolean;
  bus: ObserverStreamBus;
  bridgeRuntime: PrismBridgeRuntime;
  /** Increment to force a clean iframe remount (e.g. Reload button). */
  reloadNonce?: number;
  /** Register as a routable Observer endpoint (enabled for Published App mode & Studio preview). */
  registerAsEndpoint?: boolean;
  onRuntimeError?: (error: PrismRuntimeError) => void;
  onReady?: () => void;
  className?: string;
}

export const PrismSandboxFrame: FC<PrismSandboxFrameProps> = ({
  prismId,
  snapshot,
  holoview = null,
  darkMode,
  bus,
  bridgeRuntime,
  reloadNonce = 0,
  registerAsEndpoint = true,
  onRuntimeError,
  onReady,
  className,
}) => {
  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  const grantsRef = useRef(snapshot.grants);
  grantsRef.current = snapshot.grants;

  const digestRef = useRef(snapshot.digest);
  digestRef.current = snapshot.digest;

  const holoviewRef = useRef(holoview);
  holoviewRef.current = holoview;

  const activeCss = holoview ? holoview.prismCss : snapshot.app.css;
  const latestCssRef = useRef(activeCss);
  latestCssRef.current = activeCss;

  const latestDarkRef = useRef(darkMode);
  latestDarkRef.current = darkMode;

  const onRuntimeErrorRef = useRef(onRuntimeError);
  onRuntimeErrorRef.current = onRuntimeError;

  const onReadyRef = useRef(onReady);
  onReadyRef.current = onReady;

  // Structural key excludes CSS so CSS/theme edits hot-swap without reloading iframe state!
  const structuralKey = useMemo(
    () =>
      holoview
        ? sha256HexSync(`holoview\0${holoview.indexHtml}\0${holoview.appJs}\0${holoview.prismUiJs}`).slice(0, 12)
        : sha256HexSync(`${snapshot.app.html}\0${snapshot.app.js}`).slice(0, 12),
    [holoview?.indexHtml, holoview?.appJs, holoview?.prismUiJs, holoview !== null, snapshot.app.html, snapshot.app.js],
  );

  const channelId = useMemo(
    () => `vframe_${prismId}_${structuralKey}_${reloadNonce}`,
    [prismId, structuralKey, reloadNonce],
  );

  // Build srcDoc only when HTML/JS structure or reloadNonce changes — NOT on theme or CSS-only edits!
  const initialDarkRef = useRef(darkMode);
  const srcDoc = useMemo(() => {
    initialDarkRef.current = darkMode;
    if (holoview) {
      return buildHoloviewSandboxDocument({
        prismId,
        channelId,
        digest: snapshot.digest,
        indexHtml: holoview.indexHtml,
        prismCss: holoview.prismCss,
        prismUiJs: holoview.prismUiJs,
        appJs: holoview.appJs,
        darkMode: initialDarkRef.current,
      });
    }
    return buildPrismSandboxDocument({
      prismId,
      channelId,
      snapshot,
      darkMode: initialDarkRef.current,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prismId, channelId, structuralKey]);

  // Live hot-swap CSS without reloading the sandbox iframe
  useEffect(() => {
    const win = iframeRef.current?.contentWindow;
    if (!win) return;
    win.postMessage(
      { __verdant: true, channelId, type: 'hot-css', payload: { css: activeCss } },
      '*',
    );
  }, [activeCss, channelId]);

  // Sync light/dark theme live without reloading the sandbox iframe
  useEffect(() => {
    const win = iframeRef.current?.contentWindow;
    if (!win) return;
    win.postMessage(
      { __verdant: true, channelId, type: 'theme', payload: { dark: darkMode } },
      '*',
    );
  }, [darkMode, channelId]);

  // Wire postMessage bridge + live ObserverStreamBus endpoint registration
  useEffect(() => {
    // Evict any stale background pooled iframe for this Prism now that the live frame is active
    bridgeRuntime.evictPrism(prismId);

    const pendingDispatches = new Map<
      string,
      { resolve: (v: unknown) => void; reject: (e: Error) => void }
    >();
    const frameAbort = new AbortController();
    let dispSeq = 1;
    let isReady = false;
    let resolveReady!: () => void;
    let rejectReady!: (err: Error) => void;
    const readyPromise = new Promise<void>((res, rej) => {
      resolveReady = res;
      rejectReady = rej;
    });
    const readyTimer = setTimeout(() => {
      if (!isReady) {
        rejectReady(new Error(`Prism frame "${prismId}" did not become ready in time.`));
      }
    }, 3_000);

    const onMessage = async (ev: MessageEvent) => {
      const iframeWin = iframeRef.current?.contentWindow;
      if (!iframeWin || ev.source !== iframeWin) return;
      const msg = ev.data;
      if (!msg || !msg.__verdant || msg.channelId !== channelId) return;

      if (msg.type === 'ready') {
        isReady = true;
        clearTimeout(readyTimer);
        resolveReady();
        // Re-sync latest theme & hot CSS in case either changed before DOMContentLoaded
        iframeWin.postMessage(
          { __verdant: true, channelId, type: 'theme', payload: { dark: latestDarkRef.current } },
          '*',
        );
        iframeWin.postMessage(
          { __verdant: true, channelId, type: 'hot-css', payload: { css: latestCssRef.current } },
          '*',
        );
        bus.emit({
          name: 'prism.ready',
          endpointKind: 'prism',
          endpointId: prismId,
          attributes: { 'verdant.prism.digest': digestRef.current },
        });
        onReadyRef.current?.();
        return;
      }

      if (msg.type === 'emit' && msg.payload) {
        bridgeRuntime.handlePrismEmit(
          prismId,
          grantsRef.current,
          String(msg.payload.name ?? 'prism.event'),
          msg.payload.body,
          msg.payload.attributes,
          msg.payload.traceContext ?? undefined,
        );
        return;
      }

      if (msg.type === 'runtime-error' && msg.payload) {
        const errInfo: PrismRuntimeError = {
          message: String(msg.payload.message ?? 'Uncaught runtime error'),
          line: typeof msg.payload.line === 'number' ? msg.payload.line : null,
          col: typeof msg.payload.col === 'number' ? msg.payload.col : null,
          timestampMs: Date.now(),
        };
        bus.emit({
          name: 'prism.runtime.error',
          endpointKind: 'prism',
          endpointId: prismId,
          status: 'error',
          attributes: {
            'error.message': errInfo.message,
            ...(errInfo.line !== null ? { 'code.lineno': errInfo.line } : {}),
          },
        });
        onRuntimeErrorRef.current?.(errInfo);
        return;
      }

      if (msg.type === 'prism-invoke' && msg.payload?.reqId) {
        const hv = holoviewRef.current;
        const reqId = String(msg.payload.reqId);
        const requestLine = String(msg.payload.request ?? '');
        const reply = (ok: boolean, value?: string, error?: string) => {
          iframeRef.current?.contentWindow?.postMessage(
            { __verdant: true, channelId, type: 'prism-invoke-result', payload: { reqId, ok, value, error } },
            '*',
          );
        };
        if (!hv) {
          reply(false, undefined, 'No Prism model handler is attached.');
          return;
        }
        try {
          const answer = await hv.onInvoke(requestLine);
          bus.emit({
            name: 'prism.model.invoke',
            endpointKind: 'prism',
            endpointId: prismId,
            attributes: {
              'verdant.request.bytes': requestLine.length,
              'verdant.response.bytes': answer.length,
            },
            body: { request: requestLine, answer },
          });
          reply(true, answer);
        } catch (err) {
          reply(false, undefined, err instanceof Error ? err.message : String(err));
        }
        return;
      }

      if (msg.type === 'dispatch-response' && msg.payload?.reqId) {
        const waiter = pendingDispatches.get(msg.payload.reqId);
        pendingDispatches.delete(msg.payload.reqId);
        if (!waiter) return;
        if (msg.payload.ok) waiter.resolve(msg.payload.value);
        else waiter.reject(new Error(String(msg.payload.error ?? 'Prism handler failed')));
        return;
      }

      if (msg.type === 'rpc' && msg.payload?.reqId) {
        const { reqId, method, args, traceContext, callChain } = msg.payload as {
          reqId: string;
          method: string;
          args: Record<string, unknown>;
          traceContext?: OtelTraceContext;
          callChain?: readonly string[];
        };
        const reply = (ok: boolean, value?: unknown, error?: string) => {
          iframeRef.current?.contentWindow?.postMessage(
            { __verdant: true, channelId, type: 'rpc-result', payload: { reqId, ok, value, error } },
            '*',
          );
        };
        try {
          const grants = grantsRef.current;
          if (method === 'storage.get') {
            reply(true, await bridgeRuntime.getKv(prismId, grants, String(args?.key ?? '')));
          } else if (method === 'storage.set') {
            await bridgeRuntime.setKv(prismId, grants, String(args?.key ?? ''), args?.value);
            bridgeRuntime.evictPrism(prismId);
            reply(true, true);
          } else if (method === 'ask') {
            if (!grants.observer) {
              throw new Error('Observer Stream grant is disabled for this Prism.');
            }
            const text = await bridgeRuntime.ask({
              prismId,
              prompt: String(args?.prompt ?? ''),
              parentContext: traceContext,
              signal: frameAbort.signal,
              onToken: (partial) => {
                iframeRef.current?.contentWindow?.postMessage(
                  { __verdant: true, channelId, type: 'rpc-token', payload: { reqId, text: partial } },
                  '*',
                );
              },
            });
            reply(true, text);
          } else if (method === 'browser.copyText') {
            reply(true, await bridgeRuntime.copyText(prismId, grants, String(args?.text ?? '')));
          } else if (method === 'browser.downloadFile') {
            reply(
              true,
              await bridgeRuntime.downloadFile(
                prismId,
                grants,
                String(args?.filename ?? 'export.txt'),
                String(args?.content ?? ''),
                String(args?.mimeType ?? 'text/plain'),
              ),
            );
          } else if (method === 'browser.notify') {
            reply(
              true,
              await bridgeRuntime.notify(
                prismId,
                grants,
                String(args?.title ?? ''),
                String(args?.body ?? ''),
              ),
            );
          } else if (method === 'ui.getState') {
            reply(true, await bridgeRuntime.getUiState(prismId, grants));
          } else if (method === 'ui.callPrism') {
            reply(
              true,
              await bridgeRuntime.callPrism(
                prismId,
                grants,
                String(args?.target ?? ''),
                args?.payload,
                traceContext,
                callChain,
              ),
            );
          } else if (method === 'ui.openPrism') {
            reply(true, await bridgeRuntime.openPrism(prismId, grants, String(args?.target ?? '')));
          } else if (method === 'fetch') {
            const res = await bridgeRuntime.proxiedFetch(
              prismId,
              grants,
              String(args?.url ?? ''),
              args?.init as { method?: string; headers?: Record<string, string>; body?: string },
              traceContext,
              (partial) => {
                iframeRef.current?.contentWindow?.postMessage(
                  { __verdant: true, channelId, type: 'rpc-token', payload: { reqId, text: partial } },
                  '*',
                );
              },
              frameAbort.signal,
            );
            reply(true, res);
          } else {
            reply(false, undefined, `Unsupported bridge method "${method}".`);
          }
        } catch (err) {
          reply(false, undefined, err instanceof Error ? err.message : String(err));
        }
      }
    };

    window.addEventListener('message', onMessage);

    let unregisterEndpoint: (() => void) | null = null;
    if (registerAsEndpoint) {
      unregisterEndpoint = bus.registerEndpoint({
        endpointId: prismId,
        endpointKind: 'prism',
        priority: 10, // Mounted live frame takes priority over warm background pool
        dispatch: async (
          input: unknown,
          traceContext: OtelTraceContext,
          signal?: AbortSignal,
          callChain?: readonly string[],
        ) => {
          // For HOLOVIEW / declarative Prisms, invoke the proved model directly with causal trace context!
          const hv = holoviewRef.current;
          if (hv) {
            const line =
              typeof input === 'string'
                ? input
                : input && typeof input === 'object' && typeof (input as { request?: unknown }).request === 'string'
                  ? String((input as { request: string }).request)
                  : input && typeof input === 'object' && typeof (input as { message?: unknown }).message === 'string'
                    ? String((input as { message: string }).message)
                    : JSON.stringify(input ?? '');
            const answer = await hv.onInvoke(line, traceContext);
            return { answer, request: line };
          }
          if (!isReady) {
            await readyPromise;
          }
          const win = iframeRef.current?.contentWindow;
          if (!win) {
            throw new Error('Prism frame is not mounted.');
          }
          const reqId = `live_${dispSeq++}`;
          return new Promise((resolve, reject) => {
            const onAbort = () => {
              pendingDispatches.delete(reqId);
              reject(new Error(`Dispatch to "${prismId}" aborted.`));
            };
            if (signal?.aborted) {
              onAbort();
              return;
            }
            signal?.addEventListener('abort', onAbort, { once: true });
            pendingDispatches.set(reqId, {
              resolve: (v) => {
                signal?.removeEventListener('abort', onAbort);
                resolve(v);
              },
              reject: (e) => {
                signal?.removeEventListener('abort', onAbort);
                reject(e);
              },
            });
            win.postMessage(
              {
                __verdant: true,
                channelId,
                type: 'dispatch-input',
                payload: { reqId, input, traceContext, callChain: callChain ?? [prismId] },
              },
              '*',
            );
          });
        },
      });
    }

    return () => {
      clearTimeout(readyTimer);
      frameAbort.abort();
      if (!isReady) {
        rejectReady(new Error('Prism frame unmounted before ready.'));
      }
      window.removeEventListener('message', onMessage);
      unregisterEndpoint?.();
      for (const waiter of pendingDispatches.values()) {
        waiter.reject(new Error('Prism frame unmounted.'));
      }
      pendingDispatches.clear();
    };
  }, [prismId, channelId, bus, bridgeRuntime, registerAsEndpoint]);

  return (
    <iframe
      ref={iframeRef}
      key={channelId}
      title={`Prism: ${snapshot.name}`}
      sandbox="allow-scripts"
      srcDoc={srcDoc}
      className={className ?? 'h-full w-full border-0 bg-[#F8F9FA] dark:bg-[#0B0F14]'}
    />
  );
};
