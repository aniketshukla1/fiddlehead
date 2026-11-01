import React, { useEffect, useMemo, useState } from "react";
import {
  Plus, Search, MoreHorizontal, Play, FileText, RotateCcw, Download, Trash2, GitCompare, Box,
  MessageSquareText, Link2,
} from "lucide-react";
import { listModels, registerModel, deployModel, undeployModel, deleteModelApi } from "../lib/account.js";
import { probeModels, deleteModel as deleteArtifact, exportModel } from "../lib/capabilities.js";
import { apiBase } from "../lib/api.js";
import { describeModel } from "../lib/modelCard.js";
import Menu from "./Menu.jsx";
import ModelCard from "./ModelCard.jsx";
import ModelCompare from "./ModelCompare.jsx";
import { CopyField } from "./CodeBlock.jsx";
import { MODALITY_ICON } from "./Overview.jsx";

const ENGINE = "ws://localhost:8765";
const SERVABLE = (k) => !["asr", "vlm", "tts"].includes(k);   // hosted inference is text-first
const endpoint = (name) => `${apiBase}/v1/${encodeURIComponent(name)}/generate`;

// the model's own training curve, drawn small
function Curve({ values }) {
  // saved curves are [step, loss] pairs; accept plain numbers too
  const v = (values || []).map((x) => (Array.isArray(x) ? x[1] : x)).filter((x) => Number.isFinite(x));
  if (v.length < 3) return null;
  const pts = v.length > 60 ? v.filter((_, i) => i % Math.ceil(v.length / 60) === 0) : v;
  const lo = Math.min(...pts), hi = Math.max(...pts), span = hi - lo || 1;
  const d = pts.map((y, i) => `${((i / (pts.length - 1)) * 100).toFixed(1)},${(2 + (1 - (y - lo) / span) * 20).toFixed(1)}`).join(" ");
  return (
    <svg className="spark mdl-curve" viewBox="0 0 100 24" preserveAspectRatio="none" role="img"
         aria-label={`loss ${v[0].toFixed(2)} → ${v[v.length - 1].toFixed(2)}`}>
      <polyline points={d} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

const quality = (i) => i.eval?.headline || i.metrics?.headline
  || (i.final_loss != null ? `loss ${Number(i.final_loss).toFixed(3)}` : i.metrics?.final_loss != null ? `loss ${Number(i.metrics.final_loss).toFixed(3)}` : null);

export default function Models({ engine, onNewModel, onOpen, onRetrain, onNotice }) {
  const [reg, setReg] = useState(null);     // registered in the workspace (deployable)
  const [art, setArt] = useState(null);     // checkpoints saved on the engine
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(null);
  const [card, setCard] = useState(null);
  const [comparing, setComparing] = useState(false);

  useEffect(() => { listModels().then(setReg).catch(() => setReg([])); }, []);
  useEffect(() => {
    if (engine.online) probeModels(ENGINE).then((m) => setArt(m || []));
    else if (engine.online === false) setArt([]);
  }, [engine.online]);

  const rows = useMemo(() => {
    const by = new Map();
    (art || []).forEach((a) => by.set(a.name, { name: a.name, artifact: a }));
    (reg || []).forEach((r) => by.set(r.name, { ...(by.get(r.name) || { name: r.name }), record: r }));
    return [...by.values()]
      .map((r) => ({
        ...r,
        info: { ...(r.record || {}), ...(r.artifact || {}) },
        ts: r.artifact?.created ? r.artifact.created * 1000 : Date.parse(r.record?.created_at) || 0,
      }))
      .sort((a, b) => b.ts - a.ts);
  }, [art, reg]);

  const shown = rows.filter((r) => !q.trim() || `${r.name} ${describeModel(r.info)}`.toLowerCase().includes(q.trim().toLowerCase()));
  const loading = reg === null || art === null;

  async function setLive(row, on) {
    setBusy(row.name);
    try {
      let rec = row.record;
      if (!rec) {                       // engine-only checkpoint: register it, then deploy
        const a = row.artifact;
        rec = await registerModel({
          name: a.name, kind: a.kind || "scratch", modality: a.modality || "text",
          base_model: a.base_model || null, params_m: a.params_m ?? null,
          metrics: a.eval || (a.final_loss != null ? { final_loss: a.final_loss } : null),
        });
      }
      const up = on ? await deployModel(rec.id) : await undeployModel(rec.id);
      setReg((rs) => [...(rs || []).filter((r) => r.id !== up.id), up]);
      onNotice?.(on ? `${row.name} is live at /v1/${row.name}/generate` : `${row.name} is no longer served`);
    } catch (e) { onNotice?.(e.message); } finally { setBusy(null); }
  }

  async function remove(row) {
    const where = [row.artifact && "its files on the engine", row.record && "its workspace entry and endpoint"].filter(Boolean).join(" and ");
    if (!window.confirm(`Delete “${row.name}”? This removes ${where}.`)) return;
    try {
      if (row.artifact) { const l = await deleteArtifact(ENGINE, row.name); if (l) setArt(l); }
      if (row.record) { await deleteModelApi(row.record.id); setReg((rs) => rs.filter((r) => r.id !== row.record.id)); }
      onNotice?.(`Deleted ${row.name}`);
    } catch (e) { onNotice?.(e.message); }
  }

  return (
    <div className="page">
      <header className="ph">
        <div>
          <h1>Models</h1>
          <p>Everything you've trained — saved checkpoints on your engine and models in this workspace. Switch an endpoint on to serve it over HTTP.</p>
        </div>
        <div className="ph-actions">
          {(art?.length || 0) >= 2 && (
            <button type="button" className="btn" onClick={() => setComparing(true)}><GitCompare size={15} />Compare runs</button>
          )}
          <button type="button" className="btn btn-primary" onClick={onNewModel}><Plus size={16} />New model</button>
        </div>
      </header>

      {engine.online === false && (
        <div className="note warn" style={{ marginBottom: 16 }}>
          <Box size={16} aria-hidden="true" />
          <span>The training engine isn't running, so checkpoints saved on it aren't listed. Workspace models still show below.</span>
        </div>
      )}

      <section className="ui-card">
        <div className="ui-card-h mdl-toolbar">
          <label className="mdl-search">
            <Search size={15} aria-hidden="true" />
            <input className="input" placeholder="Filter models" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Filter models" />
          </label>
          <span className="subtle num" style={{ fontSize: 12.5 }}>{loading ? "Loading…" : `${shown.length} model${shown.length === 1 ? "" : "s"}`}</span>
        </div>

        {loading ? (
          <div className="list" aria-busy="true">{[0, 1, 2].map((i) => <div key={i} className="list-row"><div className="skel" style={{ width: `${70 - i * 12}%` }} /></div>)}</div>
        ) : rows.length === 0 ? (
          <div className="empty">
            <div className="empty-i"><Box size={20} /></div>
            <h3>No models yet</h3>
            <p>Train one from a recipe — or the 30-second sample — and save it from the studio. It shows up here, ready to deploy.</p>
            <button type="button" className="btn btn-primary" onClick={onNewModel}><Plus size={15} />New model</button>
          </div>
        ) : shown.length === 0 ? (
          <div className="empty"><h3>No matches</h3><p>Nothing matches “{q}”.</p></div>
        ) : (
          <div className="tbl-wrap">
            <table className="tbl mdl-tbl">
              <thead>
                <tr><th>Model</th><th>Training</th><th>Quality</th><th>Endpoint</th><th className="ta-r"><span className="sr-only">Actions</span></th></tr>
              </thead>
              <tbody>
                {shown.map((row) => {
                  const i = row.info;
                  const Icon = MODALITY_ICON[i.modality] || MODALITY_ICON[i.kind] || MessageSquareText;
                  const live = !!row.record?.deployed;
                  const servable = SERVABLE(i.kind);
                  return (
                    <tr key={row.name}>
                      <td>
                        <div className="cell-main">
                          <span className="cell-icon"><Icon size={16} aria-hidden="true" /></span>
                          <div style={{ minWidth: 0 }}>
                            <div className="cell-t">{row.name}</div>
                            <div className="cell-s">{describeModel(i)}{i.params_m ? ` · ${i.params_m}M` : ""}{!row.artifact ? " · workspace only" : ""}</div>
                          </div>
                        </div>
                      </td>
                      <td className="mdl-train">
                        <Curve values={i.loss_curve} />
                        {i.step ? <div className="cell-s num">{Number(i.step).toLocaleString()} steps</div> : <span className="subtle">—</span>}
                      </td>
                      <td><span className="mdl-q">{quality(i) || <span className="subtle">—</span>}</span></td>
                      <td>
                        <div className="mdl-ep">
                          <button type="button" role="switch" aria-checked={live} className="switch"
                                  disabled={!servable || busy === row.name}
                                  aria-label={`${live ? "Stop serving" : "Serve"} ${row.name}`}
                                  title={servable ? (live ? "Serving — click to stop" : "Serve over HTTP") : "Hosted inference is text-only for now"}
                                  onClick={() => setLive(row, !live)} />
                          {live ? <CopyField value={`/v1/${row.name}/generate`} label="Copy endpoint path" />
                            : <span className="subtle mdl-ep-l">{servable ? "Off" : "Text models only"}</span>}
                        </div>
                      </td>
                      <td>
                        <div className="cell-actions">
                          {row.artifact && (
                            <button type="button" className="btn btn-sm" onClick={() => onOpen(row.name)} title="Open in the studio to chat, test and inspect">
                              <Play size={13} />Open
                            </button>
                          )}
                          <Menu label={`${row.name} actions`} trigger={({ props }) => (
                            <button type="button" className="btn btn-ghost btn-icon btn-sm" aria-label={`More actions for ${row.name}`} {...props}>
                              <MoreHorizontal size={16} />
                            </button>
                          )} items={[
                            { label: "Model card", icon: FileText, onSelect: () => setCard(i) },
                            row.artifact && { label: "Retrain with new data", icon: RotateCcw, onSelect: () => onRetrain(i) },
                            row.artifact && { label: "Download .zip", icon: Download, onSelect: () => exportModel(ENGINE, row.name) },
                            live && { label: "Copy full endpoint URL", icon: Link2, onSelect: () => { navigator.clipboard?.writeText(endpoint(row.name)); onNotice?.("Endpoint URL copied"); } },
                            "-",
                            { label: "Delete", icon: Trash2, danger: true, onSelect: () => remove(row) },
                          ]} />
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <ModelCard model={card} onClose={() => setCard(null)} />
      {comparing && <ModelCompare models={art || []} onClose={() => setComparing(false)} />}
    </div>
  );
}
