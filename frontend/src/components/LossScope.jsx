import React, { useMemo } from "react";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine,
} from "recharts";

export default function LossScope({ history, latest }) {
  const data = useMemo(
    () => history.map((f) => ({ step: f.step, loss: f.loss, grad: f.grad_norm, lr: f.lr })),
    [history]
  );
  const hasLr = data.some((d) => d.lr != null);
  const fmtLr = (v) => (v == null ? "" : Number(v).toExponential(1).replace("e-", "e−"));

  // the step an induction circuit emerged — pinned on the curve as the
  // "induction bump", the moment the model learned to copy from context
  const inductionStep = latest?.induction_step ?? null;
  const inWindow = inductionStep != null && data.length > 1 &&
    inductionStep >= data[0].step && inductionStep <= data[data.length - 1].step;

  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <div className="eyebrow">Signal / 01</div>
          <div className="title">Loss &amp; Gradient Norm</div>
        </div>
        <div className="readout">
          {latest?.loss != null ? `loss ${latest.loss.toFixed(3)}` : "—"}
          <br />
          <span style={{ color: "var(--warn)" }}>
            {latest?.grad_norm != null ? `‖∇‖ ${latest.grad_norm.toFixed(2)}` : ""}
          </span>
          {latest?.lr != null && (
            <>
              <br />
              <span style={{ color: "var(--accent-blue)" }}>lr {fmtLr(latest.lr)}</span>
            </>
          )}
        </div>
      </div>
      <div className="panel-body">
        <div className="scanline" />
        {data.length < 2 ? (
          <div className="empty-hint">waiting for training signal…</div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 12, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="var(--line)" strokeDasharray="2 4" vertical={false} />
              <XAxis dataKey="step" stroke="var(--muted)" tick={{ fontSize: 10, fontFamily: "IBM Plex Mono" }} />
              <YAxis
                yAxisId="loss"
                stroke="var(--muted)"
                tick={{ fontSize: 10, fontFamily: "IBM Plex Mono" }}
                width={34}
              />
              <YAxis yAxisId="grad" orientation="right" hide />
              <YAxis yAxisId="lr" orientation="right" hide domain={[0, "dataMax"]} />
              {inWindow && (
                <ReferenceLine
                  yAxisId="loss"
                  x={inductionStep}
                  stroke="var(--signal)"
                  strokeDasharray="3 3"
                  strokeOpacity={0.9}
                  label={{
                    value: "◉ induction emerges",
                    position: "insideTopRight",
                    fill: "var(--accent)",
                    fontSize: 10,
                    fontFamily: "IBM Plex Mono",
                  }}
                />
              )}
              <Tooltip
                contentStyle={{
                  background: "var(--surface-2)",
                  border: "1px solid var(--line)",
                  fontFamily: "IBM Plex Mono",
                  fontSize: 12,
                }}
                labelStyle={{ color: "var(--muted)" }}
              />
              <Line
                yAxisId="loss"
                type="monotone"
                dataKey="loss"
                stroke="var(--accent)"
                dot={false}
                strokeWidth={1.75}
                isAnimationActive={false}
              />
              <Line
                yAxisId="grad"
                type="monotone"
                dataKey="grad"
                stroke="var(--warn)"
                dot={false}
                strokeWidth={1}
                strokeOpacity={0.75}
                isAnimationActive={false}
              />
              {hasLr && (
                <Line
                  yAxisId="lr"
                  type="monotone"
                  dataKey="lr"
                  stroke="var(--accent-blue)"
                  dot={false}
                  strokeWidth={1}
                  strokeDasharray="4 3"
                  strokeOpacity={0.85}
                  isAnimationActive={false}
                  connectNulls
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
