import React, { useState } from "react";
import { modelCardRows, dataSummary, sampleCont, kindLine, modelCardMarkdown, usageSnippet } from "../lib/modelCard.js";

// A shareable card for a model — recipe, metrics, data, and sample output —
// with copy-to-Markdown and download. Rendered as a modal overlay.
export default function ModelCard({ model, onClose }) {
  const [copied, setCopied] = useState(false);
  const [copiedUsage, setCopiedUsage] = useState(false);
  if (!model) return null;

  const rows = modelCardRows(model);
  const ds = dataSummary(model.data);
  const samples = (model.samples || []).filter(Boolean);
  const usage = usageSnippet(model);
  const md = modelCardMarkdown(model);

  function copyUsage() {
    navigator.clipboard?.writeText(usage).then(() => {
      setCopiedUsage(true);
      setTimeout(() => setCopiedUsage(false), 1500);
    });
  }

  function copy() {
    navigator.clipboard?.writeText(md).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }
  function download() {
    const blob = new Blob([md], { type: "text/markdown" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${model.name}.md`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  return (
    <div className="mc-overlay" onClick={onClose}>
      <div className="mc-card" onClick={(e) => e.stopPropagation()}>
        <div className="mc-head">
          <div>
            <div className="mc-name">{model.name}</div>
            <div className="mc-kind">{kindLine(model)}{model.params_m ? ` · ${model.params_m}M params` : ""}</div>
          </div>
          <button className="mc-x" onClick={onClose} aria-label="close">✕</button>
        </div>

        <div className="mc-body">
          <div className="mc-section">Recipe</div>
          <div className="mc-grid">
            {rows.map(([k, v]) => (
              <React.Fragment key={k}>
                <span className="mc-k">{k}</span>
                <span className="mc-v">{v}</span>
              </React.Fragment>
            ))}
          </div>

          {(model.final_loss != null || model.eval?.headline) && (
            <>
              <div className="mc-section">Metrics</div>
              <div className="mc-metrics">
                {model.final_loss != null && (
                  <span className="mc-metric"><b>{Number(model.final_loss).toFixed(3)}</b> final loss</span>
                )}
                {model.eval?.headline && <span className="mc-metric mc-quality">{model.eval.headline}</span>}
              </div>
            </>
          )}

          {ds && (
            <>
              <div className="mc-section">Data</div>
              <div className="mc-dataline">{ds}</div>
            </>
          )}

          {samples.length > 0 && (
            <>
              <div className="mc-section">Sample output</div>
              {samples.map((s, i) => (
                <pre className="mc-sample" key={i}>
                  <span className="mc-seed">{typeof s === "string" ? "" : s.prompt}</span>
                  {typeof s === "string" ? s : sampleCont(s)}
                </pre>
              ))}
            </>
          )}

          <div className="mc-section mc-section-row">
            <span>Usage — run it</span>
            <button className="mc-inline-btn" onClick={copyUsage}>{copiedUsage ? "✓ copied" : "⧉ copy code"}</button>
          </div>
          <pre className="mc-sample mc-usage">{usage}</pre>
        </div>

        <div className="mc-foot">
          <button className="mc-btn" onClick={copy}>{copied ? "✓ copied" : "⧉ Copy Markdown"}</button>
          <button className="mc-btn" onClick={download}>↓ Download .md</button>
        </div>
      </div>
    </div>
  );
}
