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
import { buildPrismSandboxDocument } from '../engine/prismSandboxHtml';
import type { VerdantPrismSnapshot } from '../engine/prismStudio';

export interface PrismRuntimeError {
  readonly message: string;
  readonly line: number | null;
  readonly col: number | null;
  readonly timestampMs: number;
}

interface PrismSandboxFrameProps {
  prismId: string;
  snapshot: VerdantPrismSnapshot;
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

  const onRuntimeErrorRef = useRef(onRuntimeError);
  onRuntimeErrorRef.current = onRuntimeError;

  const onReadyRef = useRef(onReady);
  onReadyRef.current = onReady;

  const channelId = useMemo(
    () => `vframe_${prismId}_${snapshot.digest.slice(-10)}_${reloadNonce}`,
    [prismId, snapshot.digest, reloadNonce],
  );

  // Build srcDoc only when Prism code/digest or reloadNonce changes — NOT on theme toggles!
  const initialDarkRef = useRef(darkMode);
  const srcDoc = useMemo(() => {
    initialDarkRef.current = darkMode;
    return buildPrismSandboxDocument({
      prismId,
      channelId,
      snapshot,
      darkMode: initialDarkRef.current,
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prismId, channelId, snapshot.digest]);

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
    const pendingDispatches = new Map<
      string,
      { resolve: (v: unknown) => void; reject: (e: Error) => void }
    >();
    let dispSeq = 1;

    const onMessage = async (ev: MessageEvent) => {
      const iframeWin = iframeRef.current?.contentWindow;
      if (!iframeWin || ev.source !== iframeWin) return;
      const msg = ev.data;
      if (!msg || !msg.__verdant || msg.channelId !== channelId) return;

      if (msg.type === 'ready') {
        bus.emit({
          name: 'prism.ready',
          endpointKind: 'prism',
          endpointId: prismId,
          attributes: { 'verdant.prism.digest': snapshot.digest },
        });
        onReadyRef.current?.();
        return;
      }

      if (msg.type === 'emit' && msg.payload) {
        if (!grantsRef.current.observer) return;
        bus.emit({
          name: String(msg.payload.name ?? 'prism.event'),
          endpointKind: 'prism',
          endpointId: prismId,
          attributes: msg.payload.attributes,
          body: msg.payload.body,
        });
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

      if (msg.type === 'dispatch-response' && msg.payload?.reqId) {
        const waiter = pendingDispatches.get(msg.payload.reqId);
        pendingDispatches.delete(msg.payload.reqId);
        if (!waiter) return;
        if (msg.payload.ok) waiter.resolve(msg.payload.value);
        else waiter.reject(new Error(String(msg.payload.error ?? 'Prism handler failed')));
        return;
      }

      if (msg.type === 'rpc' && msg.payload?.reqId) {
        const { reqId, method, args } = msg.payload as {
          reqId: string;
          method: string;
          args: Record<string, unknown>;
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
            reply(true, true);
          } else if (method === 'ask') {
            if (!grants.observer) {
              throw new Error('Observer Stream grant is disabled for this Prism.');
            }
            const text = await bridgeRuntime.ask({
              prismId,
              prompt: String(args?.prompt ?? ''),
              onToken: (partial) => {
                iframeRef.current?.contentWindow?.postMessage(
                  { __verdant: true, channelId, type: 'rpc-token', payload: { reqId, text: partial } },
                  '*',
                );
              },
            });
            reply(true, text);
          } else if (method === 'fetch') {
            const res = await bridgeRuntime.proxiedFetch(
              prismId,
              grants,
              String(args?.url ?? ''),
              args?.init as { method?: string; headers?: Record<string, string>; body?: string },
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
        dispatch: (input: unknown, traceContext: OtelTraceContext) => {
          const win = iframeRef.current?.contentWindow;
          if (!win) {
            return Promise.reject(new Error('Prism frame is not mounted.'));
          }
          const reqId = `live_${dispSeq++}`;
          return new Promise((resolve, reject) => {
            pendingDispatches.set(reqId, { resolve, reject });
            win.postMessage(
              {
                __verdant: true,
                channelId,
                type: 'dispatch-input',
                payload: { reqId, input, traceContext },
              },
              '*',
            );
          });
        },
      });
    }

    return () => {
      window.removeEventListener('message', onMessage);
      unregisterEndpoint?.();
      for (const waiter of pendingDispatches.values()) {
        waiter.reject(new Error('Prism frame unmounted.'));
      }
      pendingDispatches.clear();
    };
  }, [prismId, channelId, snapshot.digest, bus, bridgeRuntime, registerAsEndpoint]);

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
