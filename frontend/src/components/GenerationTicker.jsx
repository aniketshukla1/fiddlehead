import React, { useMemo } from "react";

// Full-width bottom strip: the model's live quality, in plain language. Carries
// the latest evaluation headline ("continues your text · 82% next-char
// accuracy") plus a rolling sample and a quality sparkline. Template-agnostic —
// copy shows per-character hits, text/sort just stream the sample.

function qualityOf(f) {
  const g = f.generation, e = f.eval;
  if (g && g.of) return g.match / g.of;
  if (e) return e.next_char_accuracy ?? e.copy_accuracy ?? e.sort_accuracy ?? null;
  return null;
}

export default function GenerationTicker({ latest, history }) {
  const gen = latest?.generation;

  const lastEval = useMemo(() => {
    for (let i = history.length - 1; i >= 0; i--) if (history[i].eval) return history[i].eval;
    return null;
  }, [history]);

  const spark = useMemo(() => {
    const out = [];
    let q = 0;
    for (const f of history.slice(-90)) {
      const v = qualityOf(f);
      if (v != null) q = v;
      out.push(q);
    }
    return out;
  }, [history]);

  const q = spark.length ? spark[spark.length - 1] : 0;
  const online = q > 0.6;
  const headline = lastEval?.headline
    ?? (gen?.of ? `copying ${gen.match}/${gen.of} from context` : "warming up…");

  return (
    <div className="ticker">
      <div className="tk-status">
        <div className="tk-eyebrow">live quality</div>
        <div className={`tk-state ${online ? "on" : ""}`}>
          <span className="dot" style={{ background: online ? "var(--signal)" : "var(--muted)" }} />
          {headline}
        </div>
      </div>

      <div className="tk-strip">
        {gen ? (
          <>
            <span className="tk-prompt">{gen.prompt}</span>
            <span className="tk-arrow">▸</span>
            <span className="tk-cont">
              {gen.cont.split("").map((c, i) =>
                gen.of != null ? (
                  <span key={i} className={c === gen.prompt[i] ? "hit" : "miss"}>{c}</span>
                ) : (
                  <span key={i} className="hit">{c}</span>
                )
              )}
            </span>
          </>
        ) : (
          <span className="empty-hint">warming up…</span>
        )}
      </div>

      <div className="tk-spark" aria-hidden>
        {spark.map((v, i) => (
          <span
            key={i}
            className="bar"
            style={{
              height: `${6 + v * 26}px`,
              background: v > 0.6 ? "var(--signal)" : "var(--muted)",
              opacity: 0.3 + (i / spark.length) * 0.7,
            }}
          />
        ))}
      </div>
    </div>
  );
}
