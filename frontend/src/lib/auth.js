// Real account auth against the control-plane API: signup/login return a JWT
// (stored) + the user's org. The user object is cached in localStorage for a
// fast first paint; refresh() revalidates it against the server on load.
import { api, setToken } from "./api.js";

const USER_KEY = "fiddlehead.user";

function cache(res) {
  setToken(res.token);
  const u = { ...res.user, org: res.org };
  localStorage.setItem(USER_KEY, JSON.stringify(u));
  return u;
}

export function currentUser() {
  try { return JSON.parse(localStorage.getItem(USER_KEY) || "null"); } catch { return null; }
}

export async function signUp({ name, email, password }) {
  return cache(await api("/auth/signup", { method: "POST", body: { name, email, password } }));
}

// Returns either the signed-in user, or {mfaRequired, mfaToken} when the
// account has two-factor auth on — no session exists until the code is verified.
export async function signIn({ email, password }) {
  const res = await api("/auth/login", { method: "POST", body: { email, password } });
  if (res.mfa_required) return { mfaRequired: true, mfaToken: res.mfa_token };
  return cache(res);
}

export async function completeMfa({ mfaToken, code }) {
  return cache(await api("/auth/mfa", { method: "POST", body: { mfa_token: mfaToken, code } }));
}

export async function signOut() {
  try { await api("/auth/logout", { method: "POST" }); } catch {}  // revoke server-side
  setToken(null);
  localStorage.removeItem(USER_KEY);
}

// revalidate the stored session against the server (call on app load)
export async function refresh() {
  try { return cache(await api("/auth/me")); }
  catch { signOut(); return null; }
}
