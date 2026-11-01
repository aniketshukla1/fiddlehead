import { useCallback, useEffect, useState } from "react";
import { probeLocalCapabilities } from "./capabilities.js";

// Is the local training engine running, and what can this machine do?
// Probed on mount and whenever the window regains focus — so starting the engine
// in a terminal and switching back is noticed — rather than polled, which would
// log a failed-socket error every few seconds while it's down.
export function useEngine(liveConnected) {
  const [state, setState] = useState({ online: null, caps: null });

  const check = useCallback(async () => {
    const caps = await probeLocalCapabilities();
    setState({ online: !!caps, caps });
    return !!caps;
  }, []);

  useEffect(() => {
    check();
    const onFocus = () => check();
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [check]);

  // a live studio socket is proof enough
  const online = liveConnected ? true : state.online;
  return { online, caps: state.caps, check };
}
