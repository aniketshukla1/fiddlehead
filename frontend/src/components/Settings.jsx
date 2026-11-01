import React, { useEffect, useState } from "react";
import { User, Lock, Building2, Users, CreditCard, ScrollText, AlertTriangle, Monitor, Mail, ShieldCheck } from "lucide-react";
import {
  getAccount, updateProfile, changePassword, renameOrg,
  listMembers, setMemberRole, removeMember,
  listInvites, inviteMember, revokeInvite,
  getUsage, deleteAccount,
  listSessions, revokeSession, revokeOtherSessions,
  startTotp, enableTotp, disableTotp,
  listWorkspaces, switchWorkspace, listAudit,
  billingStatus, startCheckout, billingPortal,
} from "../lib/account.js";
import { refresh as refreshSession } from "../lib/auth.js";
import { BRAND } from "../lib/brand.js";

// Account & workspace settings. Everything here is backed by the control-plane
// API — profile, credentials, workspace, members/invites, plan, audit log.
// API keys live on the API page.

const SECTIONS = [
  { id: "profile", group: "Account", label: "Profile", icon: User },
  { id: "security", group: "Account", label: "Security", icon: Lock },
  { id: "workspace", group: "Workspace", label: "General", icon: Building2 },
  { id: "members", group: "Workspace", label: "Members", icon: Users },
  { id: "billing", group: "Workspace", label: "Plan & billing", icon: CreditCard, billing: true },
  { id: "activity", group: "Workspace", label: "Activity", icon: ScrollText },
  { id: "danger", group: "Danger zone", label: "Delete account", icon: AlertTriangle, danger: true },
];

const PLANS = [
  { id: "free", name: "Free", price: "$0", blurb: "Kick the tyres", feats: ["5 compute credits", "1 workspace", "Community support"] },
  { id: "pro", name: "Pro", price: "$29/mo", blurb: "Ship real models", feats: ["500 credits/mo", "Cloud GPU training", "Hosted endpoints", "Email support"] },
  { id: "team", name: "Team", price: "$99/mo", blurb: "Build together", feats: ["2,000 credits/mo", "Unlimited seats", "Roles & audit", "Priority support"] },
];

const initials = (n = "") => n.trim().split(/\s+/).slice(0, 2).map((w) => w[0]).join("").toUpperCase() || "?";
const fmtDate = (d) => (d ? new Date(d).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) : "—");

function Row({ label, children, hint }) {
  return (
    <div className="set-row">
      <div className="set-row-l">
        <div className="set-row-label">{label}</div>
        {hint && <div className="set-row-hint">{hint}</div>}
      </div>
      <div className="set-row-c">{children}</div>
    </div>
  );
}

function Copyable({ value }) {
  const [done, setDone] = useState(false);
  return (
    <button className="set-copy" title="copy" onClick={() => {
      navigator.clipboard?.writeText(value); setDone(true); setTimeout(() => setDone(false), 1200);
    }}>
      <code>{value}</code><span>{done ? "copied" : "copy"}</span>
    </button>
  );
}

function Note({ kind, children }) {
  if (!children) return null;
  return <div className={`set-note ${kind}`}>{children}</div>;
}

export default function Settings({ user, features = {}, section = "profile", onSignOut, onUserChange }) {
  const [acct, setAcct] = useState(null);
  const [members, setMembers] = useState([]);
  const [invites, setInvites] = useState([]);
  const [usage, setUsage] = useState(null);
  const [err, setErr] = useState(null);

  const load = () => {
    getAccount().then(setAcct).catch((e) => setErr(e.message));
    listMembers().then(setMembers).catch(() => {});
    listInvites().then(setInvites).catch(() => {});
    getUsage().then(setUsage).catch(() => {});
  };
  useEffect(() => { load(); }, []);

  const sections = SECTIONS.filter((s) => !s.billing || features.billing);
  const tab = sections.some((s) => s.id === section) ? section : "profile";
  const canManage = acct && (acct.role === "owner" || acct.role === "admin");
  const isOwner = acct && acct.role === "owner";
  const groups = [...new Set(sections.map((s) => s.group))];

  return (
    <div className="page settings-v2">
      <header className="ph">
        <div>
          <h1>Settings</h1>
          <p>Your account, and the people and plan of {acct?.org?.name || user?.org?.name || "this workspace"}.</p>
        </div>
      </header>

      <div className="set-layout">
        <nav className="set-subnav" aria-label="Settings sections">
          {groups.map((g) => (
            <div key={g} className="set-subnav-g">
              <div className="set-subnav-h">{g}</div>
              {sections.filter((s) => s.group === g).map((s) => {
                const Icon = s.icon;
                return (
                  <a key={s.id} href={`#/settings/${s.id}`} aria-current={tab === s.id ? "page" : undefined}
                     className={`set-subnav-i${s.danger ? " danger" : ""}`}>
                    <Icon size={15} strokeWidth={1.75} aria-hidden="true" />{s.label}
                  </a>
                );
              })}
            </div>
          ))}
        </nav>

        <div className="set-main">
          <Note kind="err">{err}</Note>
          {!acct ? (
            <div className="set-sec" aria-busy="true">
              <div className="skel" style={{ width: 180, height: 22 }} />
              <div className="skel" style={{ width: 280, marginTop: 10 }} />
              <div className="ui-card" style={{ marginTop: 22, height: 220 }} />
            </div>
          ) : (
            <>
              {tab === "profile" && <Profile acct={acct} reload={load} onUserChange={onUserChange} />}
              {tab === "security" && <Security />}
              {tab === "workspace" && <Workspace acct={acct} canManage={canManage} reload={load} billing={features.billing} />}
              {tab === "members" && (
                <Members acct={acct} members={members} invites={invites} canManage={canManage}
                         isOwner={isOwner} reload={load} />
              )}
              {tab === "billing" && <Billing acct={acct} usage={usage} />}
              {tab === "activity" && <Activity />}
              {tab === "danger" && <Danger acct={acct} onSignOut={onSignOut} />}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ---- profile ---------------------------------------------------------- //
function Profile({ acct, reload, onUserChange }) {
  const [name, setName] = useState(acct.user.name);
  const [email, setEmail] = useState(acct.user.email);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState(null);
  const [err, setErr] = useState(null);
  const dirty = name !== acct.user.name || email !== acct.user.email;

  async function save() {
    setBusy(true); setErr(null); setMsg(null);
    try {
      await updateProfile({ name, email });
      await refreshSession().catch(() => {});
      setMsg("Profile updated");
      onUserChange?.();
      reload();
    } catch (e) { setErr(e.message); } finally { setBusy(false); }
  }

  return (
    <section className="set-sec">
      <h2 className="set-h">Profile</h2>
      <p className="set-sub">How you appear across {BRAND}.</p>

      <div className="set-card">
        <div className="set-identity">
          <div className="set-avatar">{initials(name)}</div>
          <div>
            <div className="set-identity-n">{name || "—"}</div>
            <div className="set-identity-e">{acct.user.email}</div>
            <div className="set-identity-m">Member since {fmtDate(acct.created_at)}</div>
          </div>
        </div>

        <Row label="Display name" hint="Shown to teammates and on your models">
          <input className="set-input" value={name} onChange={(e) => setName(e.target.value)} placeholder="Your name" />
        </Row>
        <Row label="Email address" hint="Used to sign in and for account notices">
          <input className="set-input" type="email" value={email} onChange={(e) => setEmail(e.target.value)} />
        </Row>
        <Row label="User ID" hint="Handy for bug reports">
          <Copyable value={acct.user.id} />
        </Row>

        <Note kind="err">{err}</Note>
        <Note kind="ok">{msg}</Note>
        <div className="set-actions">
          <button className="set-btn primary" disabled={!dirty || busy} onClick={save}>
            {busy ? "Saving…" : "Save changes"}
          </button>
          {dirty && <button className="set-btn" onClick={() => { setName(acct.user.name); setEmail(acct.user.email); }}>Cancel</button>}
        </div>
      </div>
    </section>
  );
}

// ---- security --------------------------------------------------------- //
function Security() {
  const [cur, setCur] = useState("");
  const [next, setNext] = useState("");
  const [conf, setConf] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState(null);
  const [err, setErr] = useState(null);
  const mismatch = conf.length > 0 && next !== conf;
  const ok = cur && next.length >= 6 && next === conf;

  async function submit(e) {
    e.preventDefault();
    setBusy(true); setErr(null); setMsg(null);
    try {
      await changePassword(cur, next);
      setMsg("Password updated — your other devices stay signed in until their session expires.");
      setCur(""); setNext(""); setConf("");
    } catch (e2) { setErr(e2.message); } finally { setBusy(false); }
  }

  return (
    <section className="set-sec">
      <h2 className="set-h">Security</h2>
      <p className="set-sub">Keep your account and your models safe.</p>

      <form className="set-card" onSubmit={submit}>
        <div className="set-card-h">Change password</div>
        <Row label="Current password">
          <input className="set-input" type="password" value={cur} onChange={(e) => setCur(e.target.value)} autoComplete="current-password" />
        </Row>
        <Row label="New password" hint="At least 6 characters">
          <input className="set-input" type="password" value={next} onChange={(e) => setNext(e.target.value)} autoComplete="new-password" />
        </Row>
        <Row label="Confirm new password">
          <input className={`set-input ${mismatch ? "bad" : ""}`} type="password" value={conf} onChange={(e) => setConf(e.target.value)} autoComplete="new-password" />
        </Row>
        {mismatch && <Note kind="err">Passwords don't match</Note>}
        <Note kind="err">{err}</Note>
        <Note kind="ok">{msg}</Note>
        <div className="set-actions">
          <button className="set-btn primary" type="submit" disabled={!ok || busy}>{busy ? "Updating…" : "Update password"}</button>
        </div>
      </form>

      <TwoFactor />
      <Devices onSignedOutHere={() => window.location.reload()} />
    </section>
  );
}

// ---- two-factor auth --------------------------------------------------- //
function TwoFactor() {
  const [on, setOn] = useState(null);          // null until account loads
  const [setup, setSetup] = useState(null);    // {secret, uri}
  const [code, setCode] = useState("");
  const [codes, setCodes] = useState(null);    // recovery codes, shown once
  const [pw, setPw] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  useEffect(() => { getAccount().then((a) => setOn(a.two_factor)).catch(() => {}); }, []);

  async function begin() {
    setBusy(true); setErr(null);
    try { setSetup(await startTotp()); } catch (e) { setErr(e.message); } finally { setBusy(false); }
  }
  async function confirm() {
    setBusy(true); setErr(null);
    try { const r = await enableTotp(code.trim()); setCodes(r.recovery_codes); setOn(true); setSetup(null); setCode(""); }
    catch (e) { setErr(e.message); } finally { setBusy(false); }
  }
  async function turnOff() {
    setBusy(true); setErr(null);
    try { await disableTotp(pw); setOn(false); setPw(""); setCodes(null); }
    catch (e) { setErr(e.message); } finally { setBusy(false); }
  }

  return (
    <div className="set-card">
      <div className="set-card-h">Two-factor authentication</div>
      {on === null ? <div className="set-static">Checking…</div> : on ? (
        <>
          <Row label="Status" hint="A code from your authenticator app is required at sign-in">
            <span className="set-badge role-owner">Enabled</span>
          </Row>
          {codes && (
            <>
              <Note kind="info">Save these recovery codes somewhere safe — each works once if you lose your device.</Note>
              <div className="set-codes">{codes.map((c) => <code key={c}>{c}</code>)}</div>
            </>
          )}
          <Row label="Turn off" hint="Confirm with your password">
            <input className="set-input" type="password" value={pw} placeholder="Your password"
                   onChange={(e) => setPw(e.target.value)} autoComplete="current-password" />
          </Row>
          <Note kind="err">{err}</Note>
          <div className="set-actions">
            <button className="set-btn danger" disabled={!pw || busy} onClick={turnOff}>
              {busy ? "…" : "Disable two-factor"}
            </button>
          </div>
        </>
      ) : setup ? (
        <>
          <Note kind="info">
            Add this secret to your authenticator app (1Password, Authy, Google Authenticator),
            then enter the 6-digit code it shows.
          </Note>
          <Row label="Setup key" hint="Type it into your app, or paste the otpauth link">
            <Copyable value={setup.secret} />
          </Row>
          <Row label="otpauth link">
            <Copyable value={setup.uri} />
          </Row>
          <Row label="Verification code">
            <input className="set-input" value={code} onChange={(e) => setCode(e.target.value)}
                   placeholder="123456" inputMode="numeric" style={{ letterSpacing: "0.3em" }} />
          </Row>
          <Note kind="err">{err}</Note>
          <div className="set-actions">
            <button className="set-btn primary" disabled={code.trim().length < 6 || busy} onClick={confirm}>
              {busy ? "Verifying…" : "Verify & enable"}
            </button>
            <button className="set-btn" onClick={() => { setSetup(null); setErr(null); }}>Cancel</button>
          </div>
        </>
      ) : (
        <>
          <Row label="Status" hint="Protects your account even if your password leaks">
            <span className="set-badge muted">Off</span>
          </Row>
          <Note kind="err">{err}</Note>
          <div className="set-actions">
            <button className="set-btn primary" disabled={busy} onClick={begin}>
              {busy ? "…" : "Set up two-factor"}
            </button>
          </div>
        </>
      )}
    </div>
  );
}

// ---- signed-in devices ------------------------------------------------- //
function deviceName(ua = "") {
  const os = /Mac/i.test(ua) ? "macOS" : /Windows/i.test(ua) ? "Windows" : /Linux/i.test(ua) ? "Linux"
    : /iPhone|iPad/i.test(ua) ? "iOS" : /Android/i.test(ua) ? "Android" : "Unknown device";
  const br = /Edg/i.test(ua) ? "Edge" : /Chrome/i.test(ua) ? "Chrome" : /Safari/i.test(ua) ? "Safari"
    : /Firefox/i.test(ua) ? "Firefox" : null;
  return br ? `${br} on ${os}` : os;
}

function Devices({ onSignedOutHere }) {
  const [rows, setRows] = useState([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const load = () => { listSessions().then(setRows).catch((e) => setErr(e.message)); };
  useEffect(() => { load(); }, []);

  async function kill(s) {
    if (!window.confirm(s.current ? "Sign out of this device?" : "Sign out that device?")) return;
    try {
      const r = await revokeSession(s.id);
      if (r.was_current) return onSignedOutHere?.();
      load();
    } catch (e) { setErr(e.message); }
  }
  async function killOthers() {
    setBusy(true);
    try { await revokeOtherSessions(); load(); } catch (e) { setErr(e.message); } finally { setBusy(false); }
  }

  return (
    <div className="set-card">
      <div className="set-card-h">Where you're signed in</div>
      <Note kind="err">{err}</Note>
      <div className="set-list">
        {rows.map((s) => (
          <div className="set-member" key={s.id}>
            <div className="set-avatar sm pending"><Monitor size={14} /></div>
            <div className="set-member-id">
              <div className="set-member-n">
                {deviceName(s.user_agent)}
                {s.current && <span className="set-you">this device</span>}
              </div>
              <div className="set-member-e">
                {s.ip || "unknown IP"} · started {fmtDate(s.created_at)} · last active {fmtDate(s.last_seen_at)}
              </div>
            </div>
            <button className="set-btn danger sm" onClick={() => kill(s)}>Sign out</button>
          </div>
        ))}
        {rows.length === 0 && <div className="set-empty">No active sessions.</div>}
      </div>
      {rows.length > 1 && (
        <div className="set-actions">
          <button className="set-btn" disabled={busy} onClick={killOthers}>
            {busy ? "…" : "Sign out all other devices"}
          </button>
        </div>
      )}
    </div>
  );
}

// ---- audit log --------------------------------------------------------- //
const ACTION_LABEL = {
  "auth.login": "Signed in", "auth.logout": "Signed out", "auth.signup": "Account created",
  "auth.login_failed": "Failed sign-in", "auth.mfa_failed": "Failed 2FA code",
  "password.changed": "Password changed", "profile.updated": "Profile updated",
  "2fa.enabled": "Two-factor enabled", "2fa.disabled": "Two-factor disabled",
  "session.revoked": "Device signed out", "session.revoked_others": "Other devices signed out",
  "workspace.renamed": "Workspace renamed", "member.invited": "Teammate invited",
  "member.added": "Teammate added", "member.joined": "Teammate joined",
  "member.removed": "Teammate removed", "member.left": "Left workspace",
  "member.role_changed": "Role changed",
};

function Activity() {
  const [rows, setRows] = useState([]);
  const [err, setErr] = useState(null);
  useEffect(() => { listAudit().then(setRows).catch((e) => setErr(e.message)); }, []);
  const risky = (a) => a.startsWith("auth.login_failed") || a.endsWith("_failed") || a === "2fa.disabled";

  return (
    <section className="set-sec">
      <h2 className="set-h">Activity</h2>
      <p className="set-sub">Security and workspace events, newest first.</p>
      <div className="set-card">
        <Note kind="err">{err}</Note>
        <div className="set-list">
          {rows.length === 0 && <div className="set-empty">Nothing recorded yet.</div>}
          {rows.map((a) => (
            <div className="set-member" key={a.id}>
              <div className={`set-avatar sm ${risky(a.action) ? "warn" : "pending"}`}>
                {risky(a.action) ? <AlertTriangle size={13} /> : <ShieldCheck size={13} />}
              </div>
              <div className="set-member-id">
                <div className="set-member-n">{ACTION_LABEL[a.action] || a.action}</div>
                <div className="set-member-e">
                  {[a.actor, a.detail, a.ip].filter(Boolean).join(" · ") || "—"}
                </div>
              </div>
              <div className="set-static" style={{ fontSize: 11.5 }}>
                {a.created_at ? new Date(a.created_at).toLocaleString() : ""}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

// ---- workspace -------------------------------------------------------- //
function Workspace({ acct, canManage, reload, billing }) {
  const [name, setName] = useState(acct.org.name);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState(null);
  const [err, setErr] = useState(null);
  const dirty = name !== acct.org.name;

  async function save() {
    setBusy(true); setErr(null); setMsg(null);
    try { await renameOrg(name); setMsg("Workspace renamed"); reload(); }
    catch (e) { setErr(e.message); } finally { setBusy(false); }
  }

  return (
    <section className="set-sec">
      <h2 className="set-h">Workspace</h2>
      <p className="set-sub">Everything you build — models, datasets, keys — belongs to this workspace.</p>

      <div className="set-card">
        <Row label="Workspace name" hint={canManage ? "Visible to everyone in the workspace" : "Only owners and admins can rename"}>
          <input className="set-input" value={name} disabled={!canManage} onChange={(e) => setName(e.target.value)} />
        </Row>
        <Row label="Your role">
          <span className={`set-badge role-${acct.role}`}>{acct.role}</span>
        </Row>
        <Row label="Type">
          <div className="set-static">{acct.org_personal ? "Personal workspace" : "Team workspace"}</div>
        </Row>
        {billing && (
          <Row label="Plan">
            <span className={`set-badge plan-${acct.org.plan}`}>{acct.org.plan}</span>
          </Row>
        )}
        <Row label="Workspace ID" hint="Use it in API calls and bug reports">
          <Copyable value={acct.org.id} />
        </Row>
        <Row label="Created">
          <div className="set-static">{fmtDate(acct.org_created_at)}</div>
        </Row>
        <div className="set-stats">
          <div><b>{acct.member_count}</b><span>members</span></div>
          <div><b>{acct.model_count}</b><span>models</span></div>
          <div><b>{acct.key_count}</b><span>active keys</span></div>
        </div>
        <Note kind="err">{err}</Note>
        <Note kind="ok">{msg}</Note>
        {canManage && (
          <div className="set-actions">
            <button className="set-btn primary" disabled={!dirty || busy} onClick={save}>{busy ? "Saving…" : "Save changes"}</button>
            {dirty && <button className="set-btn" onClick={() => setName(acct.org.name)}>Cancel</button>}
          </div>
        )}
      </div>

      <WorkspaceSwitcher />
    </section>
  );
}

// ---- switch between workspaces you belong to --------------------------- //
function WorkspaceSwitcher() {
  const [rows, setRows] = useState([]);
  const [err, setErr] = useState(null);
  useEffect(() => { listWorkspaces().then(setRows).catch((e) => setErr(e.message)); }, []);

  async function go(w) {
    try { await switchWorkspace(w.id); window.location.reload(); }
    catch (e) { setErr(e.message); }
  }
  if (rows.length <= 1) return null;   // nothing to switch between yet

  return (
    <div className="set-card">
      <div className="set-card-h">Your workspaces</div>
      <Note kind="err">{err}</Note>
      <div className="set-list">
        {rows.map((w) => (
          <div className="set-member" key={w.id}>
            <div className="set-avatar sm">{initials(w.name)}</div>
            <div className="set-member-id">
              <div className="set-member-n">
                {w.name}{w.current && <span className="set-you">current</span>}
              </div>
              <div className="set-member-e">
                {w.personal ? "personal" : "team"} · {w.members} member{w.members === 1 ? "" : "s"} · {w.plan}
              </div>
            </div>
            <span className={`set-badge role-${w.role}`}>{w.role}</span>
            {!w.current && <button className="set-btn sm" onClick={() => go(w)}>Switch</button>}
          </div>
        ))}
      </div>
    </div>
  );
}

// ---- members ---------------------------------------------------------- //
function Members({ acct, members, invites, canManage, isOwner, reload }) {
  const [email, setEmail] = useState("");
  const [role, setRole] = useState("member");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState(null);
  const [err, setErr] = useState(null);

  async function send(e) {
    e.preventDefault();
    setBusy(true); setErr(null); setMsg(null);
    try {
      const r = await inviteMember(email.trim(), role);
      setMsg(r.added ? `${r.email} was added to the workspace` : `Invitation created for ${r.email} — they'll join automatically when they sign up`);
      setEmail("");
      reload();
    } catch (e2) { setErr(e2.message); } finally { setBusy(false); }
  }

  async function changeRole(m, r) {
    try { await setMemberRole(m.id, r); reload(); } catch (e) { setErr(e.message); }
  }
  async function kick(m) {
    const self = m.is_you;
    if (!window.confirm(self ? "Leave this workspace?" : `Remove ${m.name} from the workspace?`)) return;
    try { await removeMember(m.id); reload(); } catch (e) { setErr(e.message); }
  }
  async function unInvite(i) {
    try { await revokeInvite(i.id); reload(); } catch (e) { setErr(e.message); }
  }

  return (
    <section className="set-sec">
      <h2 className="set-h">Members</h2>
      <p className="set-sub">Who can build, train, and deploy in this workspace.</p>

      {canManage && (
        <form className="set-card" onSubmit={send}>
          <div className="set-card-h">Invite a teammate</div>
          <div className="set-invite">
            <input className="set-input" type="email" required placeholder="teammate@company.com"
                   value={email} onChange={(e) => setEmail(e.target.value)} />
            <select className="set-input" value={role} onChange={(e) => setRole(e.target.value)}>
              <option value="member">Member — build & train</option>
              <option value="admin">Admin — manage workspace</option>
            </select>
            <button className="set-btn primary" type="submit" disabled={busy || !email.trim()}>{busy ? "Sending…" : "Send invite"}</button>
          </div>
          <Note kind="err">{err}</Note>
          <Note kind="ok">{msg}</Note>
        </form>
      )}

      <div className="set-card">
        <div className="set-card-h">{members.length} member{members.length === 1 ? "" : "s"}</div>
        <div className="set-list">
          {members.map((m) => (
            <div className="set-member" key={m.id}>
              <div className="set-avatar sm">{initials(m.name)}</div>
              <div className="set-member-id">
                <div className="set-member-n">{m.name}{m.is_you && <span className="set-you">you</span>}</div>
                <div className="set-member-e">{m.email}</div>
              </div>
              {isOwner && !m.is_you ? (
                <select className="set-mini" value={m.role} onChange={(e) => changeRole(m, e.target.value)}>
                  <option value="owner">owner</option>
                  <option value="admin">admin</option>
                  <option value="member">member</option>
                </select>
              ) : (
                <span className={`set-badge role-${m.role}`}>{m.role}</span>
              )}
              {(canManage || m.is_you) && (
                <button className="set-btn danger sm" onClick={() => kick(m)}>{m.is_you ? "Leave" : "Remove"}</button>
              )}
            </div>
          ))}
        </div>
      </div>

      {invites.length > 0 && (
        <div className="set-card">
          <div className="set-card-h">Pending invitations</div>
          <div className="set-list">
            {invites.map((i) => (
              <div className="set-member" key={i.id}>
                <div className="set-avatar sm pending"><Mail size={14} /></div>
                <div className="set-member-id">
                  <div className="set-member-n">{i.email}</div>
                  <div className="set-member-e">Invited {fmtDate(i.created_at)} · joins as {i.role}</div>
                </div>
                <span className="set-badge muted">pending</span>
                {canManage && <button className="set-btn danger sm" onClick={() => unInvite(i)}>Revoke</button>}
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

// ---- plan & usage ----------------------------------------------------- //
function Billing({ acct, usage }) {
  const credits = acct.org.credits ?? 0;
  const train = usage?.train_units ?? 0;
  const infer = usage?.infer_units ?? 0;
  const grant = { free: 5, pro: 500, team: 2000 }[acct.org.plan] || 5;
  const pct = Math.max(0, Math.min(100, (credits / grant) * 100));
  const [bill, setBill] = useState(null);
  const [busy, setBusy] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => { billingStatus().then(setBill).catch(() => {}); }, []);

  async function upgrade(plan) {
    setBusy(plan); setErr(null);
    try { const { url } = await startCheckout(plan); window.location.href = url; }
    catch (e) { setErr(e.message); setBusy(null); }
  }
  async function manage() {
    setBusy("portal"); setErr(null);
    try { const { url } = await billingPortal(); window.location.href = url; }
    catch (e) { setErr(e.message); setBusy(null); }
  }

  return (
    <section className="set-sec">
      <h2 className="set-h">Plan &amp; usage</h2>
      <p className="set-sub">What you're on, what you've used, and what's metered.</p>

      <div className="set-card">
        <div className="set-plan-now">
          <div>
            <div className="set-card-h">Current plan</div>
            <div className="set-plan-name">{acct.org.plan}</div>
          </div>
          <div className="set-credits">
            <div className="set-credits-v">{credits.toFixed(1)}</div>
            <div className="set-credits-l">credits left</div>
          </div>
        </div>
        <div className="set-meter"><span style={{ width: `${pct}%` }} /></div>
        <div className="set-usage">
          <div><b>{train.toFixed(0)}</b><span>training units</span></div>
          <div><b>{infer.toFixed(0)}</b><span>inference calls</span></div>
          <div><b>{(usage?.events || []).length}</b><span>metered events</span></div>
        </div>
      </div>

      <div className="set-card">
        <div className="set-card-h">Plans</div>
        <div className="set-plans">
          {PLANS.map((p) => (
            <div className={`set-plan ${acct.org.plan === p.id ? "on" : ""}`} key={p.id}>
              <div className="set-plan-t">{p.name}</div>
              <div className="set-plan-p">{p.price}</div>
              <div className="set-plan-b">{p.blurb}</div>
              <ul>{p.feats.map((f) => <li key={f}>{f}</li>)}</ul>
              {acct.org.plan === p.id
                ? <div className="set-plan-cur">Current plan</div>
                : p.id === "free"
                  ? <div className="set-plan-cur" />
                  : (
                    <button className="set-btn" disabled={!bill?.configured || busy === p.id}
                            title={bill?.configured ? `Subscribe to ${p.name}` : "Billing isn't configured on this deployment"}
                            onClick={() => upgrade(p.id)}>
                      {busy === p.id ? "Opening checkout…" : "Upgrade"}
                    </button>
                  )}
            </div>
          ))}
        </div>
        <Note kind="err">{err}</Note>
        {bill?.configured ? (
          <>
            {bill.plan_status && (
              <div className="adv-hint" style={{ marginTop: 12 }}>
                Subscription <b>{bill.plan_status}</b>
                {bill.renews_at ? ` · renews ${fmtDate(bill.renews_at)}` : ""}
              </div>
            )}
            {bill.has_subscription && (
              <div className="set-actions">
                <button className="set-btn" disabled={busy === "portal"} onClick={manage}>
                  {busy === "portal" ? "Opening…" : "Manage billing & invoices"}
                </button>
              </div>
            )}
          </>
        ) : (
          <Note kind="info">
            Checkout isn't switched on for this deployment (no Stripe key configured). Usage is
            metered accurately today and billing reads from these same counters, so turning it on
            is a config change — not a migration.
          </Note>
        )}
      </div>
    </section>
  );
}

// ---- danger zone ------------------------------------------------------ //
function Danger({ acct, onSignOut }) {
  const [pw, setPw] = useState("");
  const [conf, setConf] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const ready = pw && conf.toLowerCase().trim() === acct.user.email.toLowerCase();

  async function nuke() {
    if (!window.confirm("This permanently deletes your account, workspace, models and keys. Continue?")) return;
    setBusy(true); setErr(null);
    try { await deleteAccount(pw, conf); onSignOut(); }
    catch (e) { setErr(e.message); setBusy(false); }
  }

  return (
    <section className="set-sec">
      <h2 className="set-h">Delete account</h2>
      <p className="set-sub">Irreversible. Please be certain.</p>

      <div className="set-card danger">
        <div className="set-card-h">Delete account</div>
        <p className="set-danger-copy">
          This permanently removes your account and any workspace where you're the only member —
          including its models, datasets, API keys and usage history. Workspaces you share with
          others are left intact; you'll simply be removed from them.
        </p>
        <Row label="Confirm password">
          <input className="set-input" type="password" value={pw} onChange={(e) => setPw(e.target.value)} autoComplete="current-password" />
        </Row>
        <Row label="Type your email to confirm" hint={acct.user.email}>
          <input className="set-input" value={conf} onChange={(e) => setConf(e.target.value)} placeholder={acct.user.email} />
        </Row>
        <Note kind="err">{err}</Note>
        <div className="set-actions">
          <button className="set-btn danger" disabled={!ready || busy} onClick={nuke}>
            {busy ? "Deleting…" : "Delete my account"}
          </button>
        </div>
      </div>
    </section>
  );
}
