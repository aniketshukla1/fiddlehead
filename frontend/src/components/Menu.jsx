import React, { useEffect, useRef, useState } from "react";

// A small accessible dropdown. `trigger` renders the button and receives the
// props to spread on it; `items` are { label, icon, onSelect, hint, danger,
// disabled }, "-" for a separator, or { heading } for a group label.
// Closes on outside click / Escape (focus returns to the trigger); arrows move.
export default function Menu({ trigger, items, align = "end", side = "bottom", label, className = "" }) {
  const [open, setOpen] = useState(false);
  const wrap = useRef(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e) => { if (!wrap.current?.contains(e.target)) setOpen(false); };
    const onKey = (e) => {
      if (e.key !== "Escape") return;
      setOpen(false);
      wrap.current?.querySelector("[aria-haspopup]")?.focus();
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    const r = requestAnimationFrame(() =>
      wrap.current?.querySelector('[role="menuitem"]:not([disabled])')?.focus());
    return () => {
      cancelAnimationFrame(r);
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  function onListKey(e) {
    const els = [...wrap.current.querySelectorAll('[role="menuitem"]:not([disabled])')];
    const i = els.indexOf(document.activeElement);
    if (e.key === "ArrowDown") { e.preventDefault(); els[(i + 1) % els.length]?.focus(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); els[(i - 1 + els.length) % els.length]?.focus(); }
    else if (e.key === "Tab") setOpen(false);
  }

  return (
    <div className={`menu-wrap ${className}`} ref={wrap}>
      {trigger({
        open,
        props: { "aria-haspopup": "menu", "aria-expanded": open, onClick: () => setOpen((o) => !o) },
      })}
      {open && (
        <div className={`menu menu-${align} menu-${side}`} role="menu" aria-label={label} onKeyDown={onListKey}>
          {items.filter(Boolean).map((it, i) => {
            if (it === "-") return <div key={i} className="menu-sep" role="separator" />;
            if (it.heading) return <div key={i} className="menu-label">{it.heading}</div>;
            const Icon = it.icon;
            return (
              <button key={i} type="button" role="menuitem" disabled={it.disabled}
                      className={`menu-item${it.danger ? " danger" : ""}`}
                      onClick={() => { setOpen(false); it.onSelect?.(); }}>
                {Icon ? <Icon size={15} strokeWidth={1.75} aria-hidden="true" /> : <span className="menu-gap" />}
                <span className="menu-text">{it.label}</span>
                {it.hint && <span className="menu-hint">{it.hint}</span>}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
