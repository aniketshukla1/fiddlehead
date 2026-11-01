import { useEffect, useState } from "react";

// Hash routes (#/models, #/settings/security, …): deep links and a working Back
// button on any static host and inside the desktop shell, with no server config.
// Anything that isn't a route (e.g. an OAuth "#token=…" return) reads as "/".
const read = () => {
  const h = window.location.hash.replace(/^#/, "");
  return h.startsWith("/") ? h.split("?")[0] : "/";
};

export function useRoute() {
  const [path, setPath] = useState(read);
  useEffect(() => {
    const on = () => setPath(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return path;
}

export function navigate(path) {
  if (read() !== path) window.location.hash = path;
}
