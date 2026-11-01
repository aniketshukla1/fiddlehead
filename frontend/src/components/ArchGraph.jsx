import React from "react";

// Renders the transformer stack as an SVG block diagram. Each layer's
// residual block is tinted by its current gradient norm, so architecture
// and optimization dynamics show up in the same view.
export default function ArchGraph({ meta, latest }) {
  const layers = meta?.layers ?? 6;
  const gradPerLayer = latest?.grad_norm_per_layer ?? [];
  const maxGrad = Math.max(0.5, ...gradPerLayer);

  const blockH = 46;
  const gap = 10;
  const width = 260;
  const height = layers * (blockH + gap) + 40;

  // the theme's accent (calm) blending into warn (hot), so it reads in light and dark
  function heat(v) {
    return `color-mix(in srgb, var(--warn) ${Math.round(Math.min(1, v / maxGrad) * 100)}%, var(--accent))`;
  }

  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <div className="eyebrow">Signal / 04</div>
          <div className="title">Architecture — {meta?.model ?? "model"}</div>
        </div>
        <div className="readout">{meta ? `${meta.params_m}M params` : "—"}</div>
      </div>
      <div className="panel-body padded" style={{ overflow: "auto" }}>
        <svg width={width} height={height} style={{ display: "block", margin: "0 auto" }}>
          {/* input */}
          <text x={width / 2} y={16} textAnchor="middle" fill="var(--muted)" fontSize="10" fontFamily="IBM Plex Mono">
            token embed + pos
          </text>
          {Array.from({ length: layers }).map((_, i) => {
            const y = 26 + i * (blockH + gap);
            const grad = gradPerLayer[i] ?? 0;
            const color = heat(grad);
            return (
              <g key={i}>
                <rect
                  x={20}
                  y={y}
                  width={width - 40}
                  height={blockH}
                  rx={4}
                  fill="var(--surface-2)"
                  stroke={color}
                  strokeWidth={1.5}
                />
                <text x={32} y={y + 17} fill="var(--text)" fontSize="11" fontFamily="IBM Plex Mono">
                  block {i}
                </text>
                <text x={32} y={y + 32} fill="var(--muted)" fontSize="9" fontFamily="IBM Plex Mono">
                  self-attn ({meta?.heads ?? "–"}h) → mlp
                </text>
                <circle cx={width - 34} cy={y + blockH / 2} r={4} fill={color} />
                <text
                  x={width - 44}
                  y={y + blockH / 2 + 3}
                  textAnchor="end"
                  fill={color}
                  fontSize="9"
                  fontFamily="IBM Plex Mono"
                >
                  {grad ? grad.toFixed(2) : "–"}
                </text>
                {i < layers - 1 && (
                  <line
                    x1={width / 2}
                    y1={y + blockH}
                    x2={width / 2}
                    y2={y + blockH + gap}
                    stroke="var(--line)"
                  />
                )}
              </g>
            );
          })}
          <text
            x={width / 2}
            y={height - 6}
            textAnchor="middle"
            fill="var(--muted)"
            fontSize="10"
            fontFamily="IBM Plex Mono"
          >
            output head
          </text>
        </svg>
      </div>
    </div>
  );
}
