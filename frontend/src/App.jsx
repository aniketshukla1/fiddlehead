import React, { useEffect, useMemo, useRef, useState } from "react";
import { CheckCircle2 } from "lucide-react";
import { useTelemetry } from "./lib/useTelemetry.js";
import { useRoute, navigate } from "./lib/router.js";
import { useEngine } from "./lib/useEngine.js";
import { toggleTheme } from "./lib/theme.js";
import Shell from "./components/Shell.jsx";
import Overview from "./components/Overview.jsx";
import Models from "./components/Models.jsx";
import ApiPage from "./components/ApiPage.jsx";
import Studio from "./components/Studio.jsx";
import Setup, { SAMPLE_SPEC } from "./components/Setup.jsx";
import Landing from "./components/Landing.jsx";
import Settings from "./components/Settings.jsx";
import Docs from "./components/Docs.jsx";
import Compute from "./components/Compute.jsx";
import Auth from "./components/Auth.jsx";
import ModelCard from "./components/ModelCard.jsx";
import CommandPalette from "./components/CommandPalette.jsx";
import CoachMarks from "./components/CoachMarks.jsx";
import EngineDialog from "./components/EngineDialog.jsx";
import { registerModel, verifyEmail, serverMeta } from "./lib/account.js";
import { setToken } from "./lib/api.js";
import { currentUser, signOut, refresh } from "./lib/auth.js";

const SESSION_KEY = "fiddlehead.session";
const COACH_KEY = "fiddlehead.coach.seen";
const LOCAL = { mode: "local", url: "ws://localhost:8765" };

// remember the active run so a refresh (on any page) reconnects to it
function loadSession() {
  try {
    const s = JSON.parse(localStorage.getItem(SESSION_KEY) || "null");
    if (s && (s.spec || s.loadName) && s.target) return s;
  } catch {}
  return null;
}

export default function App() {
  const route = useRoute();
  const [target, setTarget] = useState(() => loadSession()?.target ?? { mode: "none" });
  const [spec, setSpec] = useState(() => loadSession()?.spec ?? null);
  const [loadName, setLoadName] = useState(() => loadSession()?.loadName ?? null);
  const [tab, setTab] = useState(() => loadSession()?.tab ?? "monitor");
  const [user, setUser] = useState(currentUser);
  const [features, setFeatures] = useState({ billing: false, email: false, oauth: [] });
  const [showCard, setShowCard] = useState(false);
  const [palette, setPalette] = useState(false);
  const [coach, setCoach] = useState(false);
  const [engineHelp, setEngineHelp] = useState(false);
  const [notice, setNotice] = useState(null);
  const [prefill, setPrefill] = useState(null);     // wizard prefill when retraining a saved model
  const createdKey = useRef(null);    // the job (spec object or model name) the engine holds
  const forceRun = useRef(false);     // an explicit "do this now" (a fresh launch/load)

  const t = useTelemetry(target);
  const { connected, mode, meta, latest, history, status, send, reset } = t;
  const engine = useEngine(connected && target.mode === "local");

  const job = spec || loadName;
  const isDemo = mode === "demo";
  const training = status?.state === "training" || isDemo;
  const hasRun = !!job && (target.mode === "local" || target.mode === "cloud");
  const runName = meta?.model || meta?.name || spec?.name || loadName || "model";
  const page = route.split("/")[1] || "";

  // if the socket drops, the engine may have lost our project — forget we built it
  useEffect(() => { if (!connected) createdKey.current = null; }, [connected]);
  // meta arriving means the engine already has our job (re-attach, don't rebuild)
  useEffect(() => { if (meta && job) createdKey.current = job; }, [meta, job]);

  // Create/load when we explicitly launched, or when the engine is idle and
  // doesn't have it (fresh start, engine restart, or a refresh with a dead engine).
  // Never rebuild a project the engine is already running — that's how a refresh
  // re-attaches instead of restarting training.
  useEffect(() => {
    if (!connected || !job || createdKey.current === job) {
      if (createdKey.current === job) forceRun.current = false;
      return;
    }
    if (forceRun.current || status?.state === "idle") {
      createdKey.current = job;
      forceRun.current = false;
      send(loadName ? { type: "load", name: loadName } : { type: "create", spec });
    }
  }, [connected, job, status, send]); // eslint-disable-line

  // persist the active run (local/cloud only; never uploaded file blobs)
  useEffect(() => {
    try {
      const blobs = spec?.data?.source === "upload" || spec?.data?.pairs;
      if (hasRun && !blobs) localStorage.setItem(SESSION_KEY, JSON.stringify({ target, spec, loadName, tab }));
      else localStorage.removeItem(SESSION_KEY);
    } catch {}
  }, [target, spec, loadName, tab, hasRun]);

  // which optional integrations this deployment has on (billing, email, social sign-in)
  useEffect(() => { serverMeta().then(setFeatures).catch(() => {}); }, []);

  // revalidate the stored session (fresh org, or sign out if expired) and consume
  // the links that land back here: email confirmation ?verify=… and OAuth #token=…
  useEffect(() => {
    const url = new URL(window.location.href);
    const hashToken = new URLSearchParams(url.hash.slice(1)).get("token");
    const verify = url.searchParams.get("verify");
    const clean = () => {
      url.searchParams.delete("verify");
      url.searchParams.delete("billing");
      window.history.replaceState({}, "", url.pathname + url.search + (url.hash.startsWith("#/") ? url.hash : ""));
    };
    if (hashToken) {
      setToken(hashToken);
      window.history.replaceState({}, "", url.pathname + url.search + "#/");
      refresh().then(setUser);
      return;
    }
    if (verify) {
      verifyEmail(verify)
        .then(() => { setNotice("Email confirmed — thanks!"); return refresh().then(setUser); })
        .catch((e) => setNotice(e.message))
        .finally(clean);
      return;
    }
    if (url.searchParams.get("billing") === "success") {
      setNotice("Subscription active — your new plan is live.");
      clean();
    }
    if (currentUser()) refresh().then(setUser);
  }, []);

  useEffect(() => {
    if (!notice) return;
    const id = setTimeout(() => setNotice(null), 6000);
    return () => clearTimeout(id);
  }, [notice]);

  // ⌘/Ctrl+K opens the command palette anywhere
  useEffect(() => {
    function onKey(e) {
      if ((e.metaKey || e.ctrlKey) && (e.key === "k" || e.key === "K")) {
        e.preventDefault();
        setPalette((p) => !p);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // the first time someone reaches a live studio, walk them through the panels
  useEffect(() => {
    if (page === "studio" && meta && !isDemo) {
      try { if (!localStorage.getItem(COACH_KEY)) setCoach(true); } catch {}
    }
  }, [page, meta, isDemo]);
  function closeCoach() {
    setCoach(false);
    try { localStorage.setItem(COACH_KEY, "1"); } catch {}
  }

  function startJob(nextTarget, nextSpec, nextLoadName) {
    reset();
    createdKey.current = null;
    forceRun.current = true;
    setTab("monitor");
    setSpec(nextSpec);
    setLoadName(nextLoadName);
    setTarget(nextTarget);
    navigate("/studio");
  }
  const launch = (tg, s) => startJob(tg, s, null);
  const openSaved = (name) => startJob(LOCAL, null, name);
  const trySample = () => startJob(LOCAL, SAMPLE_SPEC, null);
  const openDemo = () => { reset(); setSpec(null); setLoadName(null); setTarget({ mode: "demo" }); navigate("/studio"); };
  const newModel = () => { setPrefill(null); navigate("/new"); };
  const retrainFrom = (m) => { setPrefill(m); navigate("/new"); };

  function endSession() {
    if (training && !isDemo && !status?.saved &&
        !window.confirm(`End this session? Training on “${runName}” stops, and unsaved progress is lost.`)) return;
    if (training && !isDemo) send({ type: "pause" });
    reset();
    createdKey.current = null;
    setSpec(null);
    setLoadName(null);
    setTarget({ mode: "none" });
  }

  // retrain the current project from scratch: fresh weights, same recipe, and go
  function retrain() {
    if (!window.confirm("Restart training from scratch? This discards the current run.")) return;
    reset();
    send({ type: "reset" });
    send({ type: "start" });
    setTab("monitor");
  }

  function handleSignOut() {
    signOut();
    setUser(null);
    navigate("/");
  }

  const quality = useMemo(() => {
    for (let i = history.length - 1; i >= 0; i--) if (history[i].eval) return history[i].eval;
    return null;
  }, [history]);

  // a model-card object assembled from the live studio state
  const cardModel = useMemo(() => {
    if (!meta) return null;
    const kind = (meta.base_model || meta.template === "finetune" || spec?.template === "finetune") ? "finetune" : "scratch";
    return {
      name: runName, kind,
      template: meta.template, base_model: meta.base_model || spec?.base_model,
      method: meta.method || spec?.method, size: meta.size,
      params_m: meta.params_m, block_size: meta.block_size,
      n_experts: meta.n_experts, moe_top_k: meta.moe_top_k,
      tokenizer: meta.tokenizer, vocab_size: meta.vocab_size,
      step: latest?.step ?? status?.step, created: meta.created,
      data: meta.data, final_loss: latest?.loss, eval: quality,
      samples: latest?.generation ? [latest.generation] : [],
    };
  }, [meta, latest, quality, status, spec, runName]);

  // save on the engine AND register in the workspace (so it can be deployed)
  function saveModel(which) {
    const name = which === "best" ? `${runName}-best` : runName;
    send({ type: "save", name, ...(which === "best" ? { which: "best" } : {}) });
    if (cardModel) {
      registerModel({
        name, kind: cardModel.kind, modality: meta?.modality || "text",
        base_model: cardModel.base_model || null, params_m: cardModel.params_m ?? null,
        metrics: quality || (cardModel.final_loss != null ? { final_loss: cardModel.final_loss } : null),
      }).then(() => setNotice(`Saved “${name}” — find it in Models.`)).catch(() => setNotice(`Saved “${name}” on the engine.`));
    }
  }

  const toast = notice && (
    <div className="toast-v2" role="status" aria-live="polite" onClick={() => setNotice(null)}>
      <CheckCircle2 size={16} aria-hidden="true" /><span>{notice}</span>
    </div>
  );

  // ---------------- signed out: landing, auth, public docs ----------------
  if (!user) {
    if (route === "/signin" || route === "/signup") return (
      <Auth key={route} initialMode={route === "/signin" ? "signin" : "signup"}
            onAuthed={(u) => { setUser(u); navigate("/"); }} />
    );
    if (page === "docs") return <Docs onStart={() => navigate("/signup")} />;
    return <>{<Landing features={features} />}{toast}</>;
  }

  // ---------------- signed in: the app shell ----------------
  const run = hasRun || isDemo ? {
    name: isDemo ? "Demo run" : runName,
    step: latest?.step ?? status?.step ?? 0,
    training: isDemo || (training && connected),
    label: isDemo ? "Demo" : !connected ? "Connecting" : status?.early_stopped ? "Stopped early"
      : meta?.trainable === false ? "Inference only" : "Paused",
    losses: history.slice(-90).map((f) => f.loss).filter(Number.isFinite),
    canToggle: !isDemo && !!meta && meta.trainable !== false,
  } : null;

  let content;
  if (page === "models") content = <Models engine={engine} onNewModel={newModel} onOpen={openSaved} onRetrain={retrainFrom} onNotice={setNotice} />;
  else if (page === "new") content = <Setup key={prefill?.name || "new"} prefill={prefill} engine={engine} onLaunch={launch} onEngineHelp={() => setEngineHelp(true)} />;
  else if (page === "studio") content = (
    <Studio t={t} target={target} spec={spec} runName={runName} isDemo={isDemo} training={training} hasRun={hasRun}
            tab={tab} setTab={setTab} quality={quality} onSave={saveModel} onRetrain={retrain}
            onShowCard={() => setShowCard(true)} onCoach={() => setCoach(true)} onNewModel={newModel}
            onEnd={endSession} onDemo={openDemo} />
  );
  else if (page === "compute") content = <Compute onNotice={setNotice} />;
  else if (page === "api") content = <ApiPage onNotice={setNotice} />;
  else if (page === "docs") content = <Docs user={user} embedded />;
  else if (page === "settings") content = (
    <Settings user={user} features={features} section={route.split("/")[2] || "profile"}
              onSignOut={handleSignOut} onUserChange={() => refresh().then((u) => u && setUser(u))} />
  );
  else content = (
    <Overview user={user} features={features} engine={engine} run={run} onNewModel={newModel}
              onTrySample={trySample} onEngineHelp={() => setEngineHelp(true)} onNotice={setNotice} />
  );

  const go = (p) => () => navigate(p);
  const commands = [
    { id: "new", group: "Actions", label: "New model", hint: "N", run: newModel },
    { id: "sample", group: "Actions", label: "Train the 30-second sample", run: trySample },
    ...(run && run.canToggle ? [{ id: "toggle", group: "Run", label: run.training ? "Pause training" : "Resume training", run: () => send({ type: run.training ? "pause" : "start" }) }] : []),
    ...(hasRun && meta ? [
      { id: "save", group: "Run", label: `Save “${runName}”`, run: () => saveModel() },
      { id: "card", group: "Run", label: "Model card", run: () => { navigate("/studio"); setShowCard(true); } },
    ] : []),
    { id: "overview", group: "Go to", label: "Overview", run: go("/") },
    { id: "models", group: "Go to", label: "Models", run: go("/models") },
    { id: "studio", group: "Go to", label: "Studio", hint: run ? "live" : "", run: go("/studio") },
    { id: "compute", group: "Go to", label: "Compute", hint: "multi-GPU planner", run: go("/compute") },
    { id: "api", group: "Go to", label: "API & keys", run: go("/api") },
    { id: "docs", group: "Go to", label: "API reference", run: go("/docs") },
    { id: "settings", group: "Go to", label: "Settings", run: go("/settings/profile") },
    { id: "engine", group: "Actions", label: "Training engine status", run: () => setEngineHelp(true) },
    { id: "theme", group: "Actions", label: "Toggle light / dark theme", run: () => toggleTheme() },
    { id: "signout", group: "Actions", label: "Sign out", run: handleSignOut },
  ];

  return (
    <>
      <Shell route={route} user={user} engine={engine} run={run} onNewModel={newModel}
             onSearch={() => setPalette(true)} onSignOut={handleSignOut}
             onEngineHelp={() => setEngineHelp(true)}
             onRunToggle={() => send({ type: training ? "pause" : "start" })}>
        {content}
      </Shell>
      {showCard && <ModelCard model={cardModel} onClose={() => setShowCard(false)} />}
      {coach && <CoachMarks onClose={closeCoach} />}
      {engineHelp && <EngineDialog engine={engine} onClose={() => setEngineHelp(false)} />}
      {palette && <CommandPalette commands={commands} onClose={() => setPalette(false)} />}
      {toast}
    </>
  );
}
