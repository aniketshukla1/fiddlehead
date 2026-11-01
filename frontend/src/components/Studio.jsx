import React from "react";
import {
  Pause, Play, Save, ChevronDown, MoreHorizontal, Star, Undo2, History, RotateCcw, FileText,
  CircleHelp, Power, Plus, Box, Sparkles, AlertTriangle, X,
} from "lucide-react";
import Menu from "./Menu.jsx";
import { Mark } from "./Logo.jsx";
import CodeBlock from "./CodeBlock.jsx";
import LossScope from "./LossScope.jsx";
import CircuitMap from "./CircuitMap.jsx";
import EmbeddingSpace from "./EmbeddingSpace.jsx";
import ArchGraph from "./ArchGraph.jsx";
import GenerationTicker from "./GenerationTicker.jsx";
import TestPanel from "./TestPanel.jsx";
import Chat from "./Chat.jsx";
import EvalPrompts from "./EvalPrompts.jsx";
import ExpertRouting from "./ExpertRouting.jsx";

const fmt = (n) => (n == null ? "—" : Number(n).toLocaleString());

// The live instrument panel for one training run: a toolbar (status, views,
// controls) over the loss scope, attention circuitry, architecture and samples.
export default function Studio({
  t, target, spec, runName, isDemo, training, hasRun, tab, setTab, quality,
  onSave, onRetrain, onShowCard, onCoach, onNewModel, onEnd, onDemo,
}) {
  const { connected, mode, meta, latest, status, sample, error, send, clearError } = t;

  if (!hasRun && !isDemo) {
    return (
      <div className="studio-empty">
        <div className="empty">
          <div className="empty-i"><Sparkles size={20} /></div>
          <h3>No run in progress</h3>
          <p>The studio shows a model while it trains: the loss falling, attention heads forming, samples improving. Start a run, open a saved model, or explore a recorded demo.</p>
          <div className="row" style={{ justifyContent: "center" }}>
            <button type="button" className="btn btn-primary" onClick={onNewModel}><Plus size={15} />New model</button>
            <a className="btn" href="#/models"><Box size={15} />Open a saved model</a>
            <button type="button" className="btn btn-ghost" onClick={onDemo}>Explore the demo</button>
          </div>
        </div>
      </div>
    );
  }

  const waiting = !isDemo && (!connected || (!meta && !latest));
  const trainable = !isDemo && meta && meta.trainable !== false;
  const step = latest?.step ?? status?.step ?? 0;
  const checkpoints = latest?.checkpoints || [];
  const canBest = !isDemo && meta && spec && spec.template !== "finetune" && latest?.best_step != null;

  const state = isDemo ? ["Demo", "idle"]
    : !connected ? ["Connecting", "warn"]
    : status?.early_stopped ? ["Stopped early", "warn"]
    : meta?.trainable === false ? ["Inference only", "idle"]
    : training ? ["Training", "ok"]
    : meta ? ["Paused", "idle"] : ["Starting", "idle"];

  const views = [["monitor", "Monitor"], ...(meta?.n_experts > 1 ? [["experts", "Experts"]] : []), ["eval", "Eval"], ["chat", "Chat"]];

  return (
    <div className="studio">
      <div className="sbar">
        <div className="sbar-l">
          <div className="sbar-title">
            <h1 className="sbar-name" title={runName}>{runName}</h1>
            {meta && <span className="sbar-meta">{[meta.template, meta.size, meta.params_m != null && `${meta.params_m}M params`].filter(Boolean).join(" · ")}</span>}
          </div>
          <span className={`run-chip ${state[1]}`}>
            <span className={`status-dot ${state[1] === "ok" ? "ok pulse" : state[1] === "warn" ? "off" : "idle"}`} />
            {state[0]}
          </span>
        </div>

        {!isDemo && meta && (
          <div className="seg-ctl sbar-tabs" role="tablist" aria-label="Studio view">
            {views.map(([id, label]) => (
              <button key={id} type="button" role="tab" aria-selected={tab === id} className={tab === id ? "on" : ""} onClick={() => setTab(id)}>{label}</button>
            ))}
          </div>
        )}

        <div className="sbar-r">
          <dl className="sbar-stats">
            <div><dt>step</dt><dd className="num">{fmt(step)}</dd></div>
            {latest?.loss != null && <div><dt>loss</dt><dd className="num">{Number(latest.loss).toFixed(3)}</dd></div>}
            {latest?.best_step != null && <div title={latest.best_headline || "best evaluation so far"}><dt>best</dt><dd className="num">@{fmt(latest.best_step)}</dd></div>}
          </dl>
          {latest?.induction_step != null && <span className="badge badge-ok" title={`induction circuit formed at step ${latest.induction_step}`}>induction</span>}
          {latest?.frozen && <span className="badge badge-warn">frozen · probe</span>}

          {trainable && (
            <button type="button" className={`btn btn-sm ${training ? "" : "btn-primary"}`} onClick={() => send({ type: training ? "pause" : "start" })}>
              {training ? <Pause size={14} /> : <Play size={14} />}{training ? "Pause" : "Resume"}
            </button>
          )}
          {!isDemo && meta && (
            <Menu label="Save" trigger={({ props }) => (
              <button type="button" className="btn btn-sm" {...props}>
                <Save size={14} />{status?.saved ? "Saved" : "Save"}<ChevronDown size={13} className="subtle" />
              </button>
            )} items={[
              { label: status?.saved ? `Save again as “${runName}”` : `Save as “${runName}”`, icon: Save, onSelect: () => onSave() },
              canBest && { label: `Save best checkpoint (step ${fmt(latest.best_step)})`, icon: Star, onSelect: () => onSave("best") },
              "-",
              { label: "Model card", icon: FileText, onSelect: onShowCard },
            ]} />
          )}
          {!isDemo && meta && (
            <Menu label="Run actions" trigger={({ props }) => (
              <button type="button" className="btn btn-ghost btn-icon btn-sm" aria-label="More run actions" {...props}><MoreHorizontal size={16} /></button>
            )} items={[
              canBest && { label: "Roll back to best checkpoint", icon: Undo2, onSelect: () => send({ type: "use_best" }) },
              ...(checkpoints.length ? [{ heading: "Rewind to a checkpoint" },
                ...checkpoints.slice(-6).reverse().map((c) => ({
                  label: `Step ${fmt(c.step)}`, hint: c.loss != null ? `loss ${c.loss}` : undefined, icon: History,
                  onSelect: () => send({ type: "rewind", step: Number(c.step) }),
                })), "-"] : []),
              trainable && spec && { label: "Retrain from scratch", icon: RotateCcw, onSelect: onRetrain },
              { label: "Walkthrough", icon: CircleHelp, onSelect: onCoach },
              "-",
              { label: "End session", icon: Power, danger: true, onSelect: onEnd },
            ]} />
          )}
          {isDemo && <button type="button" className="btn btn-sm" onClick={onEnd}><X size={14} />Close demo</button>}
        </div>
      </div>

      {error && !waiting && (
        <div className="studio-error" role="alert">
          <AlertTriangle size={15} aria-hidden="true" />
          <span className="grow">{error}</span>
          <button type="button" className="btn btn-ghost btn-icon btn-sm" onClick={clearError} aria-label="Dismiss"><X size={14} /></button>
        </div>
      )}

      {waiting ? (
        <div className="studio-wait">
          {!connected ? (
            <div className="studio-wait-box">
              <div className="unfurl"><Mark size={44} /></div>
              <h2>Waiting for the {mode === "cloud" ? "remote" : "training"} engine…</h2>
              {mode === "cloud" ? (
                <p>Couldn't reach <code>{target.url}</code> yet. Check the URL and that the engine is running there — this page reconnects on its own.</p>
              ) : (
                <>
                  <p>Start it in a terminal from the project folder. This page connects the moment it's up.</p>
                  <CodeBlock code={"cd backend\npython engine.py"} lang="shell" />
                </>
              )}
              {error && <div className="note err"><AlertTriangle size={15} /><span>{error}</span></div>}
            </div>
          ) : (
            <div className="studio-wait-box">
              <div className="unfurl"><Mark size={44} /></div>
              <h2>Building {runName}…</h2>
              <p>Gathering data, fitting the tokenizer and laying out the network. Training starts right after.</p>
            </div>
          )}
        </div>
      ) : tab === "chat" && !isDemo ? (
        <Chat meta={meta} sample={sample} error={error} send={send} />
      ) : tab === "eval" && !isDemo ? (
        <EvalPrompts meta={meta} sample={sample} send={send} step={step} training={training} />
      ) : tab === "experts" && !isDemo ? (
        <ExpertRouting latest={latest} meta={meta} />
      ) : (
        <>
          <div className="grid stagger in">
            <LossScope history={t.history} latest={latest} />
            {isDemo ? <EmbeddingSpace latest={latest} /> : (
              <TestPanel meta={meta} sample={sample} error={error} send={send} quality={quality} />
            )}
            <CircuitMap latest={latest} meta={meta} send={send} />
            <ArchGraph meta={meta} latest={latest} />
          </div>
          <GenerationTicker latest={latest} history={t.history} />
        </>
      )}
    </div>
  );
}
