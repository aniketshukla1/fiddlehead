import React, { useState } from "react";
import { Check, Copy } from "lucide-react";

// A code sample with a copy button. `lang` shows as a small label.
export default function CodeBlock({ code, lang, className = "" }) {
  const [done, setDone] = useState(false);
  function copy() {
    navigator.clipboard?.writeText(code);
    setDone(true);
    setTimeout(() => setDone(false), 1400);
  }
  return (
    <div className={`code ${className}`}>
      <div className="code-h">
        <span className="code-lang">{lang || ""}</span>
        <button type="button" className="code-copy" onClick={copy} aria-label={done ? "Copied" : "Copy code"}>
          {done ? <Check size={13} /> : <Copy size={13} />}
          <span>{done ? "Copied" : "Copy"}</span>
        </button>
      </div>
      <pre><code>{code}</code></pre>
    </div>
  );
}

// An inline value (key, id, URL) that copies on click.
export function CopyField({ value, label = "Copy" }) {
  const [done, setDone] = useState(false);
  return (
    <button type="button" className="copyfield" title={label} onClick={() => {
      navigator.clipboard?.writeText(value);
      setDone(true);
      setTimeout(() => setDone(false), 1400);
    }}>
      <code>{value}</code>
      <span className="copyfield-i">{done ? <Check size={13} /> : <Copy size={13} />}</span>
    </button>
  );
}
