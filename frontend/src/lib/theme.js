// Light/dark theme: respects the OS preference until the user picks one,
// then persists. Applied via [data-theme] on <html> (see ui.css tokens).
const KEY = "fiddlehead.theme";

export function initTheme() {
  let t;
  try { t = localStorage.getItem(KEY); } catch {}
  if (t !== "light" && t !== "dark") {
    t = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }
  document.documentElement.setAttribute("data-theme", t);
  return t;
}

export function getTheme() {
  return document.documentElement.getAttribute("data-theme") || "dark";
}

export function setTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  try { localStorage.setItem(KEY, t); } catch {}
}

export function toggleTheme() {
  const next = getTheme() === "dark" ? "light" : "dark";
  setTheme(next);
  return next;
}
