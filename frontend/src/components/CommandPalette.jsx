import React, { useEffect, useMemo, useRef, useState } from "react";
import { Search, CornerDownLeft } from "lucide-react";

// ⌘/Ctrl+K command palette. Takes { id, label, hint, group, run } commands,
// filters by substring, groups the results, and runs the chosen one.
export default function CommandPalette({ commands, onClose }) {
  const [q, setQ] = useState("");
  const [i, setI] = useState(0);
  const inputRef = useRef(null);
  const listRef = useRef(null);

  useEffect(() => { inputRef.current?.focus(); }, []);

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    const list = s ? commands.filter((c) => `${c.label} ${c.hint || ""} ${c.group || ""}`.toLowerCase().includes(s)) : commands;
    // keep groups together, in first-seen order
    const order = [...new Set(list.map((c) => c.group || ""))];
    return order.flatMap((g) => list.filter((c) => (c.group || "") === g));
  }, [q, commands]);

  useEffect(() => { setI(0); }, [q]);
  useEffect(() => {
    listRef.current?.querySelector(`[data-idx="${i}"]`)?.scrollIntoView({ block: "nearest" });
  }, [i]);

  function onKey(e) {
    if (e.key === "ArrowDown") { e.preventDefault(); setI((x) => Math.min(filtered.length - 1, x + 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setI((x) => Math.max(0, x - 1)); }
    else if (e.key === "Enter") { e.preventDefault(); run(filtered[i]); }
    else if (e.key === "Escape") { e.preventDefault(); onClose(); }
  }
  function run(c) { if (!c) return; onClose(); c.run(); }

  let lastGroup = null;
  return (
    <div className="cmdk-overlay" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="cmdk-v2" role="dialog" aria-modal="true" aria-label="Command palette">
        <div className="cmdk-search">
          <Search size={16} aria-hidden="true" />
          <input ref={inputRef} placeholder="Search commands and pages…" value={q}
                 onChange={(e) => setQ(e.target.value)} onKeyDown={onKey}
                 role="combobox" aria-expanded="true" aria-controls="cmdk-list"
                 aria-activedescendant={filtered[i] ? `cmd-${filtered[i].id}` : undefined} />
          <kbd className="kbd">esc</kbd>
        </div>
        <div className="cmdk-items" id="cmdk-list" role="listbox" ref={listRef}>
          {filtered.length === 0 && <div className="cmdk-none">No matching commands</div>}
          {filtered.map((c, idx) => {
            const head = (c.group || "") !== lastGroup ? (lastGroup = c.group || "") : null;
            return (
              <React.Fragment key={c.id}>
                {head && <div className="cmdk-group">{head}</div>}
                <button type="button" id={`cmd-${c.id}`} data-idx={idx} role="option" aria-selected={idx === i}
                        className={`cmdk-row${idx === i ? " on" : ""}`} onMouseMove={() => setI(idx)} onClick={() => run(c)}>
                  <span className="cmdk-l">{c.label}</span>
                  {c.hint && <span className="cmdk-h">{c.hint}</span>}
                  {idx === i && <CornerDownLeft size={13} className="cmdk-enter" aria-hidden="true" />}
                </button>
              </React.Fragment>
            );
          })}
        </div>
        <div className="cmdk-foot-v2"><span><kbd className="kbd">↑</kbd><kbd className="kbd">↓</kbd> move</span><span><kbd className="kbd">↵</kbd> run</span></div>
      </div>
    </div>
  );
}
