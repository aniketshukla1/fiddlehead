import React, { useState } from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from "recharts";
import { kindLine } from "../lib/modelCard.js";

// Compare two saved runs (the payoff of retrain → v2/v3): overlaid loss curves,
// metrics side by side, a recipe diff, and each model's sample output.

function accPct(ev) {
  if (!ev) return null;
  const a = ev.next_char_accuracy ?? ev.copy_accuracy ?? ev.sort_accuracy;
  return a != null ? Math.round(a * 100) : null;
}
function mergeCurves(a, b) {
  const map = new Map();
  (a || []).forEach(([s, l]) => map.set(s, { step: s, a: l }));
  (b || []).forEach(([s, l]) => { const e = map.get(s) || { step: s }; e.b = l; map.set(s, e); });
  return [...map.values()].sort((x, y) => x.step - y.step);
}
const num = (v) => (typeof v === "number" ? v : null);

export default function ModelCompare({ models, onClose }) {
  const [aName, setAName] = useState(models[0]?.name || "");
  const [bName, setBName] = useState(models[1]?.name || models[0]?.name || "");
  const a = models.find((m) => m.name === aName);
  const b = models.find((m) => m.name === bName);

  const curve = mergeCurves(a?.loss_curve, b?.loss_curve);
  const hasCurve = curve.some((p) => p.a != null || p.b != null);

  // [label, aValue, bValue, better] — better: "lower" | "higher" | null
  const metrics = [
    ["Parameters", a ? `${a.params_m}M` : "—", b ? `${b.params_m}M` : "—", null, null, null],
    ["Steps", a?.step ?? "—", b?.step ?? "—", null, num(a?.step), num(b?.step)],
    ["Final loss", a?.final_loss?.toFixed?.(3) ?? "—", b?.final_loss?.toFixed?.(3) ?? "—", "lower", num(a?.final_loss), num(b?.final_loss)],
    ["Accuracy", accPct(a?.eval) != null ? `${accPct(a.eval)}%` : "—", accPct(b?.eval) != null ? `${accPct(b.eval)}%` : "—", "higher", accPct(a?.eval), accPct(b?.eval)],
  ];
  const recipe = [
    ["Kind", a && kindLine(a), b && kindLine(b)],
    ["Template", a?.template || (a?.base_model ? "finetune" : "—"), b?.template || (b?.base_model ? "finetune" : "—")],
    ["Base / size", a?.base_model || a?.size || "—", b?.base_model || b?.size || "—"],
    ["Tokenizer", a ? `${a.tokenizer || "—"}${a.vocab_size ? ` · ${a.vocab_size}` : ""}` : "—", b ? `${b.tokenizer || "—"}${b.vocab_size ? ` · ${b.vocab_size}` : ""}` : "—"],
    ["Context", a?.block_size || "—", b?.block_size || "—"],
  ];
  const sampleText = (m) => {
    const s = (m?.samples || []).find(Boolean);
    if (!s) return null;
    return typeof s === "string" ? s : `${s.prompt || ""}${s.cont && s.prompt && s.cont.startsWith(s.prompt) ? s.cont.slice(s.prompt.length) : (s.cont || "")}`;
  };

  function winner(better, av, bv) {
    if (!better || av == null || bv == null || av === bv) return "";
    const aWins = better === "lower" ? av < bv : av > bv;
    return aWins ? "a" : "b";
  }

  return (
    <div className="mc-overlay" onClick={onClose}>
      <div className="mc-card cmp-card" onClick={(e) => e.stopPropagation()}>
        <div className="mc-head">
          <div>
            <div className="mc-name">Compare runs</div>
            <div className="mc-kind">Overlay two saved models — loss, metrics, recipe & output</div>
          </div>
          <button className="mc-x" onClick={onClose} aria-label="close">✕</button>
        </div>

        <div className="cmp-pickers">
          <div className="cmp-pick a">
            <span className="cmp-dot" />
            <select value={aName} onChange={(e) => setAName(e.target.value)}>
              {models.map((m) => <option key={m.name} value={m.name}>{m.name}</option>)}
            </select>
          </div>
          <span className="cmp-vs">vs</span>
          <div className="cmp-pick b">
            <span className="cmp-dot" />
            <select value={bName} onChange={(e) => setBName(e.target.value)}>
              {models.map((m) => <option key={m.name} value={m.name}>{m.name}</option>)}
            </select>
          </div>
        </div>

        <div className="mc-body">
          <div className="mc-section">Loss</div>
          {hasCurve ? (
            <div className="cmp-chart">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={curve} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                  <CartesianGrid stroke="var(--line)" strokeDasharray="2 4" vertical={false} />
                  <XAxis dataKey="step" stroke="var(--muted)" tick={{ fontSize: 10, fontFamily: "IBM Plex Mono" }} />
                  <YAxis stroke="var(--muted)" width={34} tick={{ fontSize: 10, fontFamily: "IBM Plex Mono" }} />
                  <Tooltip contentStyle={{ background: "var(--surface-2)", border: "1px solid var(--line)", fontFamily: "IBM Plex Mono", fontSize: 12 }} labelStyle={{ color: "var(--muted)" }} />
                  <Legend wrapperStyle={{ fontFamily: "IBM Plex Mono", fontSize: 11 }} />
                  <Line name={aName} type="monotone" dataKey="a" stroke="var(--accent)" dot={false} strokeWidth={1.75} isAnimationActive={false} connectNulls />
                  <Line name={bName} type="monotone" dataKey="b" stroke="var(--accent-blue)" dot={false} strokeWidth={1.75} isAnimationActive={false} connectNulls />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="cmp-empty">No saved loss curve yet — train &amp; save a model to record one, then compare.</div>
          )}

          <div className="mc-section">Metrics</div>
          <table className="cmp-table">
            <thead><tr><th></th><th className="cmp-col-a">{aName}</th><th className="cmp-col-b">{bName}</th></tr></thead>
            <tbody>
              {metrics.map(([label, av, bv, better, an, bn]) => {
                const w = winner(better, an, bn);
                return (
                  <tr key={label}>
                    <td className="cmp-lab">{label}</td>
                    <td className={w === "a" ? "cmp-win" : ""}>{av}</td>
                    <td className={w === "b" ? "cmp-win" : ""}>{bv}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <div className="mc-section">Recipe</div>
          <table className="cmp-table">
            <tbody>
              {recipe.map(([label, av, bv]) => (
                <tr key={label} className={av !== bv ? "cmp-diff" : ""}>
                  <td className="cmp-lab">{label}</td>
                  <td>{av || "—"}</td>
                  <td>{bv || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="mc-section">Sample output</div>
          <div className="cmp-samples">
            <div className="cmp-sample">
              <div className="cmp-sample-h a">{aName}</div>
              <pre>{sampleText(a) || "—"}</pre>
            </div>
            <div className="cmp-sample">
              <div className="cmp-sample-h b">{bName}</div>
              <pre>{sampleText(b) || "—"}</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
