// Client for the Fiddlehead control-plane API (accounts, projects, models,
// keys, usage, billing). The training engine is still a separate WebSocket.
const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const TOKEN_KEY = "fiddlehead.token";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}
export function setToken(t) {
  if (t) localStorage.setItem(TOKEN_KEY, t);
  else localStorage.removeItem(TOKEN_KEY);
}

export async function api(path, { method = "GET", body } = {}) {
  const token = getToken();
  const res = await fetch(API + path, {
    method,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: body != null ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try { const j = await res.json(); msg = j.detail || j.message || msg; } catch {}
    // we sent a token and it was rejected → the session was revoked or expired;
    // drop it so the app falls back to signed-out instead of looping on errors
    if (res.status === 401 && token) {
      setToken(null);
      localStorage.removeItem("fiddlehead.user");
    }
    const err = new Error(msg);
    err.status = res.status;
    throw err;
  }
  return res.status === 204 ? null : res.json();
}

export const apiBase = API;
