import React, { useEffect, useMemo, useRef } from "react";

// The hero panel: the model's entire attention circuitry at a glance. Every
// layer/head is a live attention heatmap, tinted and labeled by the behavior
// the backend detected it specializing into. Induction heads — the in-context
// copy circuit — glow. Click any head to ablate it and watch the damage ripple
// into the loss and generation streams (a live causal experiment).

const LABEL_COLORS = {
  induction: "var(--accent)", // the star: prefix-matching / copy circuit
  prev: "var(--accent-blue)",      // previous-token head
  pos: "var(--accent-violet)",       // positional / self
  sink: "var(--warn)",      // attention sink (dumps onto token 0)
  "": "var(--muted)",        // still forming / mixed
};
const LABEL_TEXT = {
  induction: "induction",
  prev: "prev-token",
  pos: "positional",
  sink: "attn sink",
  "": "forming…",
};

function HeadCell({ layerIdx, headIdx, head, ablated, onToggle }) {
  const canvasRef = useRef(null);
  const color = LABEL_COLORS[head.label] ?? LABEL_COLORS[""];

  useEffect(() => {
    const canvas = canvasRef.current;
    const mat = head.matrix;
    if (!canvas || !mat) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width) return;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    // a canvas can't read CSS variables, so resolve the head's tint (--hc) here
    const css = getComputedStyle(canvas);
    ctx.fillStyle = css.getPropertyValue("--hc").trim();
    const n = mat.length;
    const px = rect.width / n;
    const py = rect.height / n;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        const v = mat[i][j];
        if (v <= 0.002) continue;
        ctx.globalAlpha = 0.06 + v * 0.94;
        ctx.fillRect(j * px, i * py, px + 0.6, py + 0.6);
      }
    }
    if (ablated) {
      ctx.globalAlpha = 0.72;
      ctx.fillStyle = css.getPropertyValue("--panel").trim();
      ctx.fillRect(0, 0, rect.width, rect.height);
    }
  }, [head.matrix, color, ablated]);

  const cls = [
    "headcell",
    head.label ? "labeled" : "forming",
    head.label === "induction" ? "glow" : "",
    ablated ? "ablated" : "",
  ].join(" ");

  return (
    <button
      className={cls}
      style={{ "--hc": color }}
      onClick={() => onToggle(layerIdx, headIdx, !ablated)}
      title={`layer ${layerIdx} · head ${headIdx}\ninduction ${head.induction} · prev ${head.prev} · pos ${head.pos} · sink ${head.sink}\nclick to ${ablated ? "restore" : "ablate"}`}
    >
      <div className="hc-canvas-wrap">
        <canvas ref={canvasRef} />
        {ablated && <span className="hc-off">ablated</span>}
      </div>
      <div className="hc-foot">
        <span className="hc-tag">L{layerIdx}·H{headIdx}</span>
        <span className="hc-label">{LABEL_TEXT[head.label] ?? head.label}</span>
      </div>
    </button>
  );
}

export default function CircuitMap({ latest, meta, send }) {
  // Fine-tuned models only ship the (expensive) attention map every few steps;
  // keep showing the last one we got so the grid doesn't blink between frames.
  const attnRef = useRef(null);
  const live = latest?.attention;
  if (live && live.layers && live.layers.length) attnRef.current = live;
  const attention = (live && live.layers && live.layers.length) ? live : attnRef.current;
  const layers = attention?.layers;

  const ablatedSet = useMemo(
    () => new Set((latest?.ablated ?? []).map(([l, h]) => `${l}-${h}`)),
    [latest?.ablated]
  );

  const inductionCount = useMemo(() => {
    let c = 0;
    for (const L of layers ?? []) for (const h of L.heads) if (h.label === "induction") c += 1;
    return c;
  }, [layers]);

  function onToggle(layer, head, on) {
    send?.({ type: "ablate", layer, head, on });
  }

  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <div className="eyebrow">Signal / 02 — head circuits</div>
          <div className="title">Attention Circuitry</div>
        </div>
        <div className="readout">
          {latest?.induction_step != null ? (
            <>
              <span style={{ color: "var(--signal)" }}>◉ induction</span> @ step {latest.induction_step}
              <br />
              <span style={{ color: "var(--muted)" }}>{inductionCount} head{inductionCount === 1 ? "" : "s"} · click to ablate</span>
            </>
          ) : (
            <span style={{ color: "var(--muted)" }}>heads forming…</span>
          )}
        </div>
      </div>
      <div className="panel-body padded" style={{ overflow: "auto" }}>
        {!layers ? (
          <div className="empty-hint">no attention samples yet…</div>
        ) : (
          <div className="circuit-grid" style={{ gridTemplateColumns: `repeat(${layers.length}, 1fr)` }}>
            {layers.map((L) => (
              <div className="circuit-col" key={L.idx}>
                <div className="col-head">layer {L.idx}</div>
                {L.heads.map((h, hi) => (
                  <HeadCell
                    key={hi}
                    layerIdx={L.idx}
                    headIdx={hi}
                    head={h}
                    ablated={ablatedSet.has(`${L.idx}-${hi}`)}
                    onToggle={onToggle}
                  />
                ))}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
