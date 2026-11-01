import React, { useEffect, useRef, useState } from "react";

// "Test how it's working." Type a prompt, the trained model continues it. This is
// the payoff of the whole loop — the user prompts their own model and reads the
// output, no notebook required. Works on any template (text/copy/sort).

export default function TestPanel({ meta, sample, error, send, quality }) {
  const [prompt, setPrompt] = useState("");
  const [pending, setPending] = useState(false);
  const idRef = useRef(0);
  const timer = useRef(null);

  const stopPending = () => { setPending(false); clearTimeout(timer.current); };
  useEffect(() => { if (sample && sample.done !== false) stopPending(); }, [sample]); // final frame only
  useEffect(() => { if (error) stopPending(); }, [error]); // never get stuck on an error

  function run() {
    const p = prompt.trim();
    if (!p || pending) return; // don't send an empty prompt
    idRef.current += 1;
    setPending(true);
    send?.({ type: "generate", prompt: p, max_new_tokens: meta?.template === "text" ? 96 : 24, id: `t${idRef.current}` });
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setPending(false), 20000); // safety net
  }

  const placeholder =
    meta?.template === "text" ? "Type a few words and let your model continue them…"
    : meta?.template === "sort" ? "e.g. 41302>  (digits then '>')"
    : "e.g. abfkqrst  (a block to be repeated)";

  return (
    <div className="panel testpanel">
      <div className="panel-head">
        <div>
          <div className="eyebrow">Playground</div>
          <div className="title">Prompt your model</div>
        </div>
        <div className="readout">{quality?.headline ? quality.headline : "warming up…"}</div>
      </div>
      <div className="panel-body padded testpanel-body">
        <div className="test-io">
          <textarea
            className="test-prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder={placeholder}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run();
            }}
            spellCheck={false}
          />
          <button className="test-run" onClick={run} disabled={pending}>
            {pending ? "generating…" : "Generate ⏎"}
          </button>
        </div>
        <div className="test-output">
          {sample ? (
            <>
              <span className="test-seed">{sample.prompt}</span>
              <span className="test-gen">{sample.text}</span>
            </>
          ) : (
            <span className="empty-hint">Your model's output will appear here. ⌘/Ctrl+Enter to run.</span>
          )}
        </div>
      </div>
    </div>
  );
}
