import React from "react";

// Live mixture-of-experts routing — per-layer, per-expert share of tokens, so
// you can watch experts specialize (or collapse) as the model trains. Shown as
// the "Experts" tab, only for MoE models.
export default function ExpertRouting({ latest, meta }) {
  const ex = latest?.experts;
  const n = meta?.n_experts || ex?.n || 0;
  const topK = meta?.moe_top_k || ex?.top_k;
  const layers = ex?.load || [];

  return (
    <div className="moeview">
      <div className="moe-head">
        <div>
          <div className="eyebrow">Mixture of Experts</div>
          <div className="moe-title">Expert routing</div>
        </div>
        <div className="moe-stat">
          <span><b>{n}</b> experts</span>
          <span>top-<b>{topK}</b> / token</span>
          {ex?.aux != null && <span>load-balance loss <b>{ex.aux.toFixed(3)}</b></span>}
        </div>
      </div>

      {layers.length === 0 ? (
        <div className="moe-empty">Waiting for a routing signal — start training to watch experts specialize.</div>
      ) : (
        <div className="moe-grid">
          {layers.map((load, li) => {
            const max = Math.max(...load, 1e-6);
            return (
              <div className="moe-layer" key={li}>
                <div className="moe-layer-h">Layer {li}</div>
                <div className="moe-bars">
                  {load.map((f, ei) => (
                    <div className="moe-col" key={ei} title={`expert ${ei}: ${(f * 100).toFixed(1)}% of tokens`}>
                      <div className="moe-bar"><div className={`moe-fill ${f < 0.02 ? "dead" : ""}`} style={{ height: `${(f / max) * 100}%` }} /></div>
                      <div className="moe-eval">{Math.round(f * 100)}%</div>
                      <div className="moe-elabel">E{ei}</div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}

      <div className="moe-note">
        Each bar is the share of tokens routed to that expert this step. Even bars mean balanced
        routing; a dominant bar or dim (dead) experts show specialization — or collapse.
      </div>
    </div>
  );
}
