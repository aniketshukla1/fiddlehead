import React, { useEffect, useState } from "react";
import {
  Plus, BookOpen, Box, Radio, Zap, Gauge, Coins, Check, ArrowRight, Play, Cpu, X, Mail,
  MessageSquareText, AudioLines, Image as ImageIcon, Volume2, ChevronRight, Network,
} from "lucide-react";
import { listModels, listKeys, getUsage, listJobs, resendVerification } from "../lib/account.js";
import { probeModels } from "../lib/capabilities.js";
import { navigate } from "../lib/router.js";
import { useCountUp } from "../lib/motion.js";
import { describeModel } from "../lib/modelCard.js";

const HIDE_KEY = "fiddlehead.onboarding.hidden";
const ENGINE_URL = "ws://localhost:8765";

export const MODALITY_ICON = { text: MessageSquareText, speech: AudioLines, asr: AudioLines, vision: ImageIcon, vlm: ImageIcon, tts: Volume2 };

function greeting() {
  const h = new Date().getHours();
  return h < 5 ? "Good evening" : h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening";
}
export const gpuTime = (s) => (!s ? "0 s" : s < 120 ? `${Math.round(s)} s` : s < 7200 ? `${(s / 60).toFixed(1)} min` : `${(s / 3600).toFixed(1)} h`);

function Tile({ icon: Icon, label, value, sub, ready }) {
  const n = useCountUp(Number(value) || 0, { start: ready, duration: 800 });
  return (
    <div className="ui-card tile">
      <div className="tile-l"><Icon size={14} aria-hidden="true" />{label}</div>
      <div className="tile-v">{ready ? (typeof value === "number" ? Math.round(n).toLocaleString() : value) : "—"}</div>
      {sub && <div className="tile-s">{sub}</div>}
    </div>
  );
}

export default function Overview({ user, features, engine, run, onNewModel, onTrySample, onEngineHelp, onNotice }) {
  const [models, setModels] = useState(null);       // registered in this workspace
  const [artifacts, setArtifacts] = useState(null); // saved on the training engine
  const [keys, setKeys] = useState(null);
  const [usage, setUsage] = useState(null);
  const [jobs, setJobs] = useState(null);
  const [hidden, setHidden] = useState(() => { try { return localStorage.getItem(HIDE_KEY) === "1"; } catch { return false; } });
  const [sending, setSending] = useState(false);
  const org = user?.org || {};

  useEffect(() => {
    listModels().then(setModels).catch(() => setModels([]));
    listKeys().then(setKeys).catch(() => setKeys([]));
    getUsage().then(setUsage).catch(() => setUsage({}));
    listJobs().then(setJobs).catch(() => setJobs([]));
  }, []);
  useEffect(() => {
    if (engine.online) probeModels(ENGINE_URL).then((m) => setArtifacts(m || []));
    else if (engine.online === false) setArtifacts([]);
  }, [engine.online]);

  const deployed = (models || []).filter((m) => m.deployed);
  const names = new Set([...(models || []).map((m) => m.name), ...(artifacts || []).map((a) => a.name)]);
  const trained = names.size > 0 || !!run;
  const caps = engine.caps;

  const steps = [
    { t: "Start the training engine", done: !!engine.online,
      d: engine.online ? `Running on this machine${caps ? ` · ${caps.cpu_cores} cores · ${caps.accelerator}` : ""}` : "A small Python process that trains models right here on your machine.",
      action: { label: "How to start", onClick: onEngineHelp } },
    { t: "Train your first model", done: trained,
      d: "A tiny model learns Shakespeare in about 30 seconds — watch its attention heads form.",
      action: { label: "Try the sample", icon: Play, primary: true, onClick: onTrySample, disabled: !engine.online } },
    { t: "Save it to your workspace", done: (models?.length || 0) > 0,
      d: "Saving in the studio registers the model here, ready to version and deploy.",
      action: run ? { label: "Open studio", onClick: () => navigate("/studio") } : { label: "New model", onClick: onNewModel } },
    { t: "Deploy an endpoint", done: deployed.length > 0,
      d: "Serve any saved text model as an HTTP API.",
      action: { label: "Open models", onClick: () => navigate("/models") } },
    { t: "Create an API key", done: (keys?.length || 0) > 0,
      d: "Keys authenticate calls to your endpoints.",
      action: { label: "Create a key", onClick: () => navigate("/api") } },
  ];
  const loaded = models !== null && keys !== null && engine.online !== null;
  const doneCount = steps.filter((s) => s.done).length;
  const current = steps.findIndex((s) => !s.done);
  const showGuide = loaded && !hidden && doneCount < steps.length;

  function hideGuide() {
    setHidden(true);
    try { localStorage.setItem(HIDE_KEY, "1"); } catch {}
  }

  // newest first: registry rows (with deploy state) plus engine-only checkpoints
  const recent = [
    ...(models || []).map((m) => ({ ...m, status: m.deployed ? "live" : "saved", headline: m.metrics?.headline })),
    ...(artifacts || []).filter((a) => !(models || []).some((m) => m.name === a.name))
      .map((a) => ({ ...a, modality: a.modality || a.kind, status: "engine" })),
  ].slice(0, 6);

  return (
    <div className="page">
      <header className="ph">
        <div>
          <div className="ph-eyebrow">{org.name || "Your workspace"}</div>
          <h1>{greeting()}{user?.name ? `, ${user.name.split(" ")[0]}` : ""}</h1>
          <p>Build, train and serve your own models — and watch them learn.</p>
        </div>
        <div className="ph-actions">
          <a className="btn" href="#/docs"><BookOpen size={15} aria-hidden="true" />Docs</a>
          <button type="button" className="btn btn-primary" onClick={onNewModel}><Plus size={16} aria-hidden="true" />New model</button>
        </div>
      </header>

      {features.email && user && user.email_verified === false && (
        <div className="note warn ov-verify">
          <Mail size={16} aria-hidden="true" />
          <span className="grow">Confirm <b>{user.email}</b> to secure your account and invite teammates.</span>
          <button type="button" className="btn btn-sm" disabled={sending} onClick={async () => {
            setSending(true);
            try { const r = await resendVerification(); onNotice?.(`Confirmation link sent to ${r.sent_to || user.email}`); }
            catch (e) { onNotice?.(e.message); }
            finally { setSending(false); }
          }}>{sending ? "Sending…" : "Resend link"}</button>
        </div>
      )}

      {showGuide && (
        <section className="ui-card guide" aria-labelledby="guide-t">
          <div className="guide-h">
            <div>
              <h2 id="guide-t" className="ui-card-t">Get started</h2>
              <p className="ui-card-d">From nothing to a model behind an API in about five minutes.</p>
            </div>
            <div className="guide-prog">
              <span className="num">{doneCount} of {steps.length}</span>
              <div className="progress" aria-hidden="true"><span style={{ width: `${(doneCount / steps.length) * 100}%` }} /></div>
              <button type="button" className="btn btn-ghost btn-icon btn-sm" onClick={hideGuide} aria-label="Hide getting started"><X size={15} /></button>
            </div>
          </div>
          <ol className="guide-steps">
            {steps.map((s, i) => {
              const A = s.action;
              return (
                <li key={s.t} className={`guide-step${s.done ? " done" : ""}${i === current ? " current" : ""}`}>
                  <span className="guide-n" aria-hidden="true">{s.done ? <Check size={13} strokeWidth={2.5} /> : i + 1}</span>
                  <div className="grow">
                    <div className="guide-t">{s.t}<span className="sr-only">{s.done ? " (done)" : ""}</span></div>
                    <div className="guide-d">{s.d}</div>
                  </div>
                  {!s.done && A && (
                    <button type="button" className={`btn btn-sm${A.primary && i === current ? " btn-primary" : ""}`}
                            onClick={A.onClick} disabled={A.disabled}
                            title={A.disabled ? "Start the engine first" : undefined}>
                      {A.icon && <A.icon size={13} aria-hidden="true" />}{A.label}
                    </button>
                  )}
                </li>
              );
            })}
          </ol>
        </section>
      )}

      <div className="tiles ov-tiles">
        <Tile icon={Box} label="Models" value={names.size} ready={models !== null && artifacts !== null}
              sub={artifacts?.length ? `${artifacts.length} saved on this engine` : "nothing saved yet"} />
        <Tile icon={Radio} label="Live endpoints" value={deployed.length} ready={models !== null}
              sub={deployed.length ? deployed.slice(0, 2).map((m) => m.name).join(", ") : "deploy from Models"} />
        <Tile icon={Zap} label="API calls" value={Math.round(usage?.infer_units || 0)} ready={usage !== null} sub="all time" />
        {features.billing
          ? <Tile icon={Coins} label="Credits" value={Math.round(org.credits ?? 0)} ready sub={`${org.plan || "free"} plan`} />
          : <Tile icon={Gauge} label="Distributed training" value={gpuTime(usage?.train_units)} ready={usage !== null} sub="GPU time metered" />}
      </div>

      <div className="ov-cols">
        <section className="ui-card" aria-labelledby="recent-t">
          <div className="ui-card-h">
            <h2 id="recent-t" className="ui-card-t">Recent models</h2>
            <a className="btn btn-ghost btn-sm" href="#/models">View all <ArrowRight size={13} aria-hidden="true" /></a>
          </div>
          {models === null ? (
            <div className="list" aria-busy="true">{[0, 1, 2].map((i) => <div key={i} className="list-row"><div className="skel" style={{ width: "60%" }} /></div>)}</div>
          ) : recent.length === 0 ? (
            <div className="empty">
              <div className="empty-i"><Box size={20} /></div>
              <h3>No models yet</h3>
              <p>Train the 30-second sample, or start from a recipe — chat assistant, code completion, speech, vision.</p>
              <div className="row" style={{ justifyContent: "center" }}>
                <button type="button" className="btn btn-primary" onClick={onNewModel}><Plus size={15} />New model</button>
              </div>
            </div>
          ) : (
            <div className="list">
              {recent.map((m) => {
                const Icon = MODALITY_ICON[m.modality] || MODALITY_ICON[m.kind] || MessageSquareText;
                return (
                  <a key={m.name} className="list-row" href="#/models">
                    <span className="cell-icon"><Icon size={16} aria-hidden="true" /></span>
                    <div className="grow">
                      <div className="cell-t">{m.name}</div>
                      <div className="cell-s">
                        {describeModel(m)}{m.params_m ? ` · ${m.params_m}M params` : ""}{m.headline ? ` · ${m.headline}` : ""}
                      </div>
                    </div>
                    {m.status === "live" ? <span className="badge badge-ok"><span className="live-dot" />Live</span>
                      : m.status === "saved" ? <span className="badge">Saved</span>
                      : <span className="badge">On engine</span>}
                    <ChevronRight size={15} className="subtle" aria-hidden="true" />
                  </a>
                );
              })}
            </div>
          )}
        </section>

        <div className="stack">
          <section className="ui-card" aria-labelledby="jobs-t">
            <div className="ui-card-h">
              <h2 id="jobs-t" className="ui-card-t">Distributed runs</h2>
              <a className="btn btn-ghost btn-sm" href="#/compute">Compute <ArrowRight size={13} aria-hidden="true" /></a>
            </div>
            {jobs === null ? <div className="list-row"><div className="skel" style={{ width: "70%" }} /></div>
              : jobs.length === 0 ? (
                <div className="ov-mini-empty">
                  <Network size={16} aria-hidden="true" />
                  <span>Plan a multi-GPU run and launch it from Compute.</span>
                </div>
              ) : (
                <div className="list">
                  {jobs.slice(0, 4).map((j) => {
                    const p = j.progress || {};
                    const pct = p.max_steps ? Math.round(((p.step || 0) / p.max_steps) * 100) : j.status === "done" ? 100 : 0;
                    return (
                      <a key={j.id} className="list-row ov-job" href="#/compute">
                        <div className="grow">
                          <div className="row" style={{ justifyContent: "space-between" }}>
                            <span className="cell-t">{j.model_name}</span>
                            <span className={`badge ${j.status === "done" ? "badge-ok" : j.status === "failed" ? "badge-danger" : j.status === "running" ? "badge-ok" : ""}`}>{j.status}</span>
                          </div>
                          <div className="cell-s">{j.strategy} · {j.world_size} GPU{j.world_size === 1 ? "" : "s"}{j.gpu_seconds ? ` · ${gpuTime(j.gpu_seconds)}` : ""}</div>
                          <div className="progress" style={{ marginTop: 8 }}><span style={{ width: `${pct}%` }} /></div>
                        </div>
                      </a>
                    );
                  })}
                </div>
              )}
          </section>

          <section className="ui-card ov-engine" aria-labelledby="eng-t">
            <div className="ui-card-h">
              <h2 id="eng-t" className="ui-card-t">This machine</h2>
              <span className={`badge ${engine.online ? "badge-ok" : engine.online === false ? "badge-warn" : ""}`}>
                {engine.online ? "Engine online" : engine.online === false ? "Engine offline" : "Checking…"}
              </span>
            </div>
            {engine.online && caps ? (
              <dl className="ov-specs">
                <div><dt>CPU</dt><dd>{caps.cpu_cores} cores</dd></div>
                <div><dt>Memory</dt><dd>{caps.ram_gb} GB</dd></div>
                <div><dt>Accelerator</dt><dd>{caps.accelerator}</dd></div>
              </dl>
            ) : (
              <div className="ov-mini-empty">
                <Cpu size={16} aria-hidden="true" />
                <span>{engine.online === false ? <>The engine isn't running. <button type="button" className="btn-link" onClick={onEngineHelp}>How to start it</button></> : "Looking for the local engine…"}</span>
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
