import React, { useEffect, useRef, useState } from "react";

// A small suite of test prompts that re-run as the model trains, so you can
// watch output quality improve — not just the loss drop. Frontend-driven: it
// sends tagged generate requests (id "eval:<i>") and routes the sample frames
// back by that id. Generations are serialized by the engine, so they don't
// overlap each other or the user's manual test box.

const KEY = "fiddlehead.evalprompts";
const DEFAULTS = ["to be, or not", "The meaning of life is", "Once upon a time"];
const MAX_PROMPTS = 6;

function loadPrompts() {
  try { const v = JSON.parse(localStorage.getItem(KEY)); return Array.isArray(v) && v.length ? v : DEFAULTS; }
  catch { return DEFAULTS; }
}

export default function EvalPrompts({ meta, sample, send, step, training }) {
  const [prompts, setPrompts] = useState(loadPrompts);
  const [outputs, setOutputs] = useState({}); // index -> { text, done, step }
  const [draft, setDraft] = useState("");
  const [auto, setAuto] = useState(false);
  const running = useRef(false);
  const stepRef = useRef(step);
  stepRef.current = step;

  useEffect(() => { try { localStorage.setItem(KEY, JSON.stringify(prompts)); } catch {} }, [prompts]);

  // route eval sample frames back to the prompt that produced them
  useEffect(() => {
    if (!sample || typeof sample.id !== "string" || !sample.id.startsWith("eval:")) return;
    const i = Number(sample.id.slice(5));
    setOutputs((o) => ({ ...o, [i]: { text: sample.text, done: sample.done, step: stepRef.current } }));
    if (sample.done && i === prompts.length - 1) running.current = false;
  }, [sample]); // eslint-disable-line

  function runAll() {
    if (!meta || !prompts.length) return;
    running.current = true;
    prompts.forEach((p, i) =>
      send({ type: "generate", id: `eval:${i}`, prompt: p, max_new_tokens: 40, temperature: 0.8, top_k: 40 }));
  }

  // auto-run on a gentle cadence while training (won't stack runs)
  useEffect(() => {
    if (!auto || !training) return;
    if (!running.current) runAll();
    const t = setInterval(() => { if (!running.current) runAll(); }, 12000);
    return () => clearInterval(t);
  }, [auto, training]); // eslint-disable-line

  function addPrompt() {
    const p = draft.trim();
    if (!p || prompts.length >= MAX_PROMPTS) return;
    setPrompts((ps) => [...ps, p]);
    setOutputs({});
    setDraft("");
  }
  function removePrompt(i) {
    setPrompts((ps) => ps.filter((_, j) => j !== i));
    setOutputs({});
  }

  return (
    <div className="evalpanel">
      <div className="eval-head">
        <div>
          <div className="eyebrow">Eval prompts</div>
          <div className="eval-title">Watch quality evolve</div>
        </div>
        <div className="eval-actions">
          <button className="ctrl" onClick={runAll} disabled={!meta}>▶ run now</button>
          <label className={`eval-auto ${auto ? "on" : ""}`}>
            <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} disabled={!training} />
            auto {training ? "" : "(train to enable)"}
          </label>
        </div>
      </div>

      <div className="eval-add">
        <input value={draft} onChange={(e) => setDraft(e.target.value)}
               onKeyDown={(e) => { if (e.key === "Enter") addPrompt(); }}
               placeholder={prompts.length >= MAX_PROMPTS ? "max prompts reached" : "Add a test prompt…"}
               disabled={prompts.length >= MAX_PROMPTS} />
        <button onClick={addPrompt} disabled={!draft.trim() || prompts.length >= MAX_PROMPTS}>+ add</button>
      </div>

      <div className="eval-list">
        {prompts.map((p, i) => {
          const o = outputs[i];
          const cont = o?.text ? (o.text.startsWith(p) ? o.text.slice(p.length) : o.text) : "";
          return (
            <div className="eval-row" key={i}>
              <div className="eval-prompt">
                <span className="eval-p">{p}</span>
                <button className="eval-x" onClick={() => removePrompt(i)} aria-label="remove">✕</button>
              </div>
              <div className="eval-out">
                {o ? (
                  <>
                    <span className="eval-seed">{p}</span>
                    <span className="eval-gen">{cont}</span>
                    {!o.done && <span className="eval-caret">▋</span>}
                    {o.done && o.step != null && <span className="eval-step">@ step {o.step}</span>}
                  </>
                ) : (
                  <span className="eval-empty">— run to see output —</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
