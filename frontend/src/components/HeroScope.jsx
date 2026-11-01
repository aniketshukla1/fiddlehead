import React, { useEffect, useId, useMemo, useState } from "react";
import { useCountUp } from "../lib/motion.js";

// A miniature of the studio that plays one short training run on load: the loss
// trace draws itself, attention heads resolve from noise into circuits (the
// induction heads light up), and the model types its first coherent line.
// Purely decorative; everything renders in its final state under reduced motion.

const W = 400, H = 150;
const rand = (seed) => () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;

function lossPath() {
  const r = rand(7), ys = [];
  for (let i = 0; i <= 120; i++) {
    const x = i / 120;
    const phase = x > 0.52 ? 1 - Math.exp(-(x - 0.52) * 18) : 0;   // the induction "phase change"
    ys.push(0.86 * Math.exp(-x * 3.1) + 0.24 - phase * 0.16 + (r() - 0.5) * 0.035 * (1.2 - x));
  }
  const hi = Math.max(...ys), lo = Math.min(...ys);
  // high loss sits near the top of the plot: SVG y grows downward
  return ys.map((y, i) => `${i ? "L" : "M"}${((i / 120) * W).toFixed(1)} ${(10 + ((hi - y) / (hi - lo)) * (H - 24)).toFixed(1)}`).join("");
}
function gradPath() {
  const r = rand(19), pts = [];
  for (let i = 0; i <= 120; i++) {
    const x = i / 120;
    const spike = Math.exp(-((x - 0.53) ** 2) / 0.0012) * 0.34;
    const y = 0.5 + 0.18 * Math.exp(-x * 2) + (r() - 0.5) * 0.12 + spike;
    pts.push([x * W, H - 14 - y * 70]);
  }
  return pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join("");
}

// 8×8 causal attention, as the lower triangle of cells: [row, col, weight]
function headCells(kind, seed) {
  const r = rand(seed), cells = [];
  for (let i = 0; i < 8; i++) for (let j = 0; j <= i; j++) {
    let w;
    if (kind === "prev") w = j === i - 1 ? 0.95 : j === i ? 0.25 : 0.05;
    else if (kind === "induction") w = j === i - 3 || (i < 3 && j === 0) ? 1 : 0.06;
    else if (kind === "first") w = j === 0 ? 0.9 : 0.08;
    else if (kind === "self") w = j === i ? 0.9 : 0.06;
    else w = 0.15 + r() * 0.35;
    cells.push([i, j, w, 0.12 + r() * 0.6]);
  }
  return cells;
}
const KINDS = ["self", "prev", "first", "mix", "prev", "induction", "mix", "first", "induction", "self", "mix", "prev"];

function Head({ kind, seed, delay }) {
  const cells = useMemo(() => headCells(kind, seed), [kind, seed]);
  return (
    <div className={`hs-head${kind === "induction" ? " ind" : ""}`} style={{ "--d": `${delay}s` }}>
      <svg viewBox="0 0 8 8" preserveAspectRatio="none">
        <g className="hs-noise">{cells.map(([i, j, , n]) => <rect key={`n${i}-${j}`} x={j} y={i} width="1" height="1" opacity={n} />)}</g>
        <g className="hs-learned">{cells.map(([i, j, w]) => <rect key={`l${i}-${j}`} x={j} y={i} width="1" height="1" opacity={w} />)}</g>
      </svg>
    </div>
  );
}

const SAMPLE = "to be, or not to be, that is the question:";

export default function HeroScope({ compact = false }) {
  const [go, setGo] = useState(false);
  useEffect(() => { const t = setTimeout(() => setGo(true), 250); return () => clearTimeout(t); }, []);
  const step = useCountUp(1284, { start: go, duration: 5200 });
  const loss = 2.31 - (2.31 - 0.41) * (step / 1284);
  const loss_d = useMemo(lossPath, []);
  const grad_d = useMemo(gradPath, []);
  const clip = `hs-sweep-${useId().replace(/:/g, "")}`;

  return (
    <div className={`hs${compact ? " compact" : ""}${go ? " go" : ""}`} aria-hidden="true">
      <div className="hs-bar">
        <span className="hs-dots"><i /><i /><i /></span>
        <span className="hs-name">sample-shakespeare</span>
        <span className="hs-chip"><span className="live-dot" />Training</span>
        {!compact && <span className="hs-tabs"><b>Monitor</b><span>Eval</span><span>Chat</span></span>}
        <span className="hs-stat">step <b className="num">{Math.round(step).toLocaleString()}</b></span>
        <span className="hs-stat">loss <b className="num">{loss.toFixed(3)}</b></span>
      </div>
      <div className="hs-body">
        <div className="hs-panel hs-loss">
          <div className="hs-h"><span className="hs-ch">CH1</span>Loss &amp; gradient norm</div>
          <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="hs-plot">
            {[1, 2, 3].map((k) => <line key={`h${k}`} x1="0" x2={W} y1={(H / 4) * k} y2={(H / 4) * k} className="hs-grid" />)}
            {[1, 2, 3, 4, 5, 6, 7].map((k) => <line key={`v${k}`} y1="0" y2={H} x1={(W / 8) * k} x2={(W / 8) * k} className="hs-grid" />)}
            <clipPath id={clip}><rect className="hs-sweep" x="0" y="0" width={W} height={H} /></clipPath>
            <g clipPath={`url(#${clip})`}>
              <path d={grad_d} className="hs-grad" />
              <path d={loss_d} className="hs-curve" />
            </g>
            <line className="hs-beam" x1="0" x2="0" y1="0" y2={H} />
          </svg>
          <div className="hs-legend"><span className="lg-a">loss</span><span className="lg-b">grad norm</span><span className="hs-mark">induction forms ↓</span></div>
        </div>
        <div className="hs-panel hs-heads">
          <div className="hs-h"><span className="hs-ch">CH2</span>Attention circuitry</div>
          <div className="hs-grid-heads">
            {KINDS.map((k, i) => <Head key={i} kind={k} seed={i * 31 + 3} delay={1.1 + (i % 4) * 0.35 + Math.floor(i / 4) * 0.25} />)}
          </div>
          <div className="hs-legend"><span>L0–L2 · 4 heads each</span><span className="hs-ind">2 induction heads</span></div>
        </div>
        {!compact && (
          <div className="hs-panel hs-sample">
            <span className="hs-ch">CH3</span>
            <span className="hs-prompt">live sample ›</span>
            <span className="hs-typed"><span style={{ "--n": SAMPLE.length }}>{SAMPLE}</span><i className="hs-caret" /></span>
          </div>
        )}
      </div>
    </div>
  );
}
