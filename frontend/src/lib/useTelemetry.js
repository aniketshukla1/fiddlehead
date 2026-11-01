import { useCallback, useEffect, useRef, useState } from "react";
import { mockFrame, mockMeta } from "./mockStream.js";

const HISTORY_LIMIT = 400;
const RECONNECT_MS = 1500;

/**
 * Connects to a compute target and exposes the training stream + a control
 * channel. A "target" is { mode, url }:
 *   - "local": a training engine on the user's machine  (ws://localhost:8765)
 *   - "cloud": a remote training engine you've connected  (wss://…)
 *   - "demo":  no backend — the synthetic induction stream (mockStream.js)
 *
 * Local and cloud are identical to the UI: both speak the same telemetry+control
 * protocol, so the only difference is the URL. Control messages (create a
 * project, start/pause, generate, ablate, save/load) are queued until the socket
 * is open, then flushed — so you can "create & train" in one click. The socket
 * auto-reconnects if the engine restarts, and `reset()` clears the studio state
 * when you start a new project on the same target.
 */
export function useTelemetry(target) {
  const mode = target?.mode ?? "demo";
  const url = target?.url ?? "ws://localhost:8765";

  const [connected, setConnected] = useState(false);
  const [meta, setMeta] = useState(null);
  const [latest, setLatest] = useState(null);
  const [history, setHistory] = useState([]);
  const [status, setStatus] = useState(null);
  const [sample, setSample] = useState(null);
  const [error, setError] = useState(null);

  const wsRef = useRef(null);
  const outbox = useRef([]);          // control messages queued until open
  const mockTimer = useRef(null);
  const mockAblated = useRef(new Set());
  const stepRef = useRef(0);

  const clearStudioState = useCallback(() => {
    setMeta(null); setLatest(null); setHistory([]);
    setStatus(null); setSample(null); setError(null);
    stepRef.current = 0;
  }, []);

  // clears the previous project's data when a new one starts on the same target
  const reset = useCallback(() => clearStudioState(), [clearStudioState]);
  const clearError = useCallback(() => setError(null), []);

  const send = useCallback((msg) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg));
    } else if (mode === "demo") {
      if (msg.type === "ablate") {
        const key = `${msg.layer}-${msg.head}`;
        if (msg.on) mockAblated.current.add(key);
        else mockAblated.current.delete(key);
      }
    } else {
      outbox.current.push(msg); // flush on open
    }
  }, [mode]);

  useEffect(() => {
    setConnected(false);
    clearStudioState();

    if (mode === "demo") {
      setMeta(mockMeta());
      mockTimer.current = setInterval(() => {
        const frame = mockFrame(stepRef.current++, mockAblated.current);
        setLatest(frame);
        setHistory((h) => [...h.slice(-HISTORY_LIMIT + 1), frame]);
      }, 250);
      return () => { clearInterval(mockTimer.current); mockTimer.current = null; };
    }

    if (mode !== "local" && mode !== "cloud") return; // idle: no target chosen yet

    let cancelled = false;
    let retry = null;

    function connect() {
      let ws;
      try {
        ws = new WebSocket(url);
      } catch (e) {
        setError(String(e));
        retry = setTimeout(connect, RECONNECT_MS);
        return;
      }
      wsRef.current = ws;
      ws.onopen = () => {
        if (cancelled) return;
        setConnected(true);
        setError(null);
        outbox.current.forEach((m) => ws.send(JSON.stringify(m)));
        outbox.current = [];
        ws.send(JSON.stringify({ type: "resync" })); // recover meta/status if a project is live
      };
      ws.onmessage = (evt) => {
        const frame = JSON.parse(evt.data);
        switch (frame.type) {
          case "meta": setMeta(frame); break;
          case "status":
            setStatus(frame);
            // rewind / resume-from-step: discard everything after the checkpoint
            // so the curves stay honest and the step counter reflects the rewind
            if (frame.rewound_to != null) {
              setHistory((h) => {
                const trimmed = h.filter((f) => f.step <= frame.rewound_to);
                setLatest(trimmed.length ? { ...trimmed[trimmed.length - 1], checkpoints: frame.checkpoints } : null);
                return trimmed;
              });
            }
            break;
          case "sample": setSample(frame); break;
          case "saved": setStatus((s) => ({ ...(s || {}), saved: frame.name })); break;
          case "capabilities": case "prepared": case "models": case "exported":
          case "datasets": case "dataset_saved": break; // setup-only frames, not studio telemetry
          case "error": setError(frame.message); break;
          default:
            setLatest(frame);
            setHistory((h) => [...h.slice(-HISTORY_LIMIT + 1), frame]);
        }
      };
      ws.onerror = () => {};
      ws.onclose = () => {
        if (cancelled) return;
        setConnected(false);
        retry = setTimeout(connect, RECONNECT_MS); // engine restarted? keep trying
      };
    }
    connect();

    return () => {
      cancelled = true;
      clearTimeout(retry);
      if (wsRef.current) wsRef.current.close();
      wsRef.current = null;
    };
  }, [mode, url, clearStudioState]);

  return { connected, mode, meta, latest, history, status, sample, error, send, reset, clearError };
}
