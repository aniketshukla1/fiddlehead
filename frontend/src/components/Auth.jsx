import React, { useEffect, useState } from "react";
import { Eye, EyeOff, ArrowLeft, Github, AlertTriangle, ShieldCheck } from "lucide-react";
import { signIn, signUp, completeMfa } from "../lib/auth.js";
import { oauthProviders, oauthStart } from "../lib/account.js";
import { navigate } from "../lib/router.js";
import Logo from "./Logo.jsx";
import HeroScope from "./HeroScope.jsx";
import ThemeToggle from "./ThemeToggle.jsx";

// Sign in / sign up — backed by the control-plane API (see lib/auth.js).
// Social buttons appear only for providers the deployment has configured.

export default function Auth({ initialMode = "signup", onAuthed }) {
  const mode = initialMode;
  const isSignup = mode === "signup";
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [show, setShow] = useState(false);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [mfaToken, setMfaToken] = useState(null);   // set when 2FA is required
  const [code, setCode] = useState("");
  const [providers, setProviders] = useState([]);

  useEffect(() => { oauthProviders().then(setProviders).catch(() => setProviders([])); }, []);

  async function social(p) {
    try { const { url } = await oauthStart(p.id); window.location.href = url; }
    catch (e) { setError(e.message); }
  }

  async function submit(e) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      if (mfaToken) { onAuthed(await completeMfa({ mfaToken, code: code.trim() })); return; }
      const u = isSignup ? await signUp({ name, email, password }) : await signIn({ email, password });
      if (u?.mfaRequired) { setMfaToken(u.mfaToken); setBusy(false); return; }
      onAuthed(u);
    } catch (err) {
      setError(err.message || "Something went wrong");
      setBusy(false);
    }
  }

  const form = mfaToken ? (
    <form className="auth2-form" onSubmit={submit}>
      <span className="auth2-badge"><ShieldCheck size={18} aria-hidden="true" /></span>
      <h1>Two-factor verification</h1>
      <p className="auth2-sub">Enter the 6-digit code from your authenticator app, or one of your recovery codes.</p>
      <label className="field">
        <span className="field-l">Code</span>
        <input className="input auth2-code" value={code} onChange={(e) => setCode(e.target.value)} placeholder="123 456"
               inputMode="numeric" autoComplete="one-time-code" autoFocus />
      </label>
      {error && <div className="note err" role="alert"><AlertTriangle size={15} /><span>{error}</span></div>}
      <button className="btn btn-primary btn-lg btn-block" type="submit" disabled={busy || code.trim().length < 6}>
        {busy ? "Verifying…" : "Verify"}
      </button>
      <button type="button" className="btn btn-ghost btn-block" onClick={() => { setMfaToken(null); setCode(""); setError(null); }}>
        <ArrowLeft size={15} />Use a different account
      </button>
    </form>
  ) : (
    <form className="auth2-form" onSubmit={submit}>
      <h1>{isSignup ? "Create your account" : "Welcome back"}</h1>
      <p className="auth2-sub">{isSignup ? "Your workspace is ready in seconds — models, keys and endpoints included." : "Sign in to your workspace."}</p>

      {providers.length > 0 && (
        <>
          <div className="auth2-social">
            {providers.map((p) => (
              <button key={p.id} type="button" className="btn btn-lg btn-block" onClick={() => social(p)}>
                {p.id === "github" && <Github size={16} aria-hidden="true" />}Continue with {p.label}
              </button>
            ))}
          </div>
          <div className="auth2-or"><span>or with email</span></div>
        </>
      )}

      {isSignup && (
        <label className="field">
          <span className="field-l">Name</span>
          <input className="input input-lg" value={name} onChange={(e) => setName(e.target.value)} placeholder="Ada Lovelace" autoComplete="name" autoFocus />
        </label>
      )}
      <label className="field">
        <span className="field-l">Email</span>
        <input className="input input-lg" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com"
               autoComplete="email" required autoFocus={!isSignup} />
      </label>
      <label className="field">
        <span className="field-l">Password</span>
        <span className="input-wrap">
          <input className="input input-lg" type={show ? "text" : "password"} value={password} onChange={(e) => setPassword(e.target.value)}
                 placeholder={isSignup ? "At least 6 characters" : "Your password"} minLength={isSignup ? 6 : undefined}
                 autoComplete={isSignup ? "new-password" : "current-password"} required />
          <button type="button" className="input-eye" onClick={() => setShow((s) => !s)} aria-label={show ? "Hide password" : "Show password"}>
            {show ? <EyeOff size={16} /> : <Eye size={16} />}
          </button>
        </span>
      </label>

      {error && <div className="note err" role="alert"><AlertTriangle size={15} /><span>{error}</span></div>}

      <button className="btn btn-primary btn-lg btn-block" type="submit" disabled={busy}>
        {busy ? (isSignup ? "Creating your workspace…" : "Signing in…") : isSignup ? "Create account" : "Sign in"}
      </button>

      <p className="auth2-switch">
        {isSignup ? "Already have an account?" : "New to Fiddlehead?"}{" "}
        <a className="btn-link" href={isSignup ? "#/signin" : "#/signup"}>{isSignup ? "Sign in" : "Create an account"}</a>
      </p>
    </form>
  );

  return (
    <div className="auth2">
      <div className="auth2-left">
        <div className="auth2-top">
          <Logo href="#/" />
          <div className="row">
            <ThemeToggle />
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => navigate("/")}><ArrowLeft size={14} />Home</button>
          </div>
        </div>
        <div className="auth2-center">{form}</div>
        <p className="auth2-legal">Self-hosted — your account lives in this deployment's own database.</p>
      </div>
      <aside className="auth2-right" aria-hidden="true">
        <div className="auth2-art">
          <div className="auth2-quote">
            <div className="lp2-eyebrow">What you'll see</div>
            <h2>Watch a model learn — the loss falling, attention heads forming, its first coherent words.</h2>
          </div>
          <HeroScope />
        </div>
      </aside>
    </div>
  );
}
