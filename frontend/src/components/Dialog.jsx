import React, { useEffect, useRef } from "react";
import { X } from "lucide-react";

// Modal dialog: dims the page, closes on Escape or a backdrop click, moves focus
// inside on open and back to whatever had it on close.
export default function Dialog({ title, onClose, children, footer, width = 520, className = "" }) {
  const box = useRef(null);
  useEffect(() => {
    const before = document.activeElement;
    const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); onClose(); } };
    document.addEventListener("keydown", onKey);
    box.current?.querySelector("input, button:not(.dlg-x), [href], select, textarea")?.focus();
    return () => { document.removeEventListener("keydown", onKey); before?.focus?.(); };
  }, []); // eslint-disable-line

  return (
    <div className="dlg-overlay" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className={`dlg ${className}`} role="dialog" aria-modal="true" aria-label={title}
           ref={box} style={{ maxWidth: width }}>
        <div className="dlg-head">
          <h2 className="dlg-title">{title}</h2>
          <button type="button" className="btn btn-ghost btn-icon btn-sm dlg-x" onClick={onClose} aria-label="Close">
            <X size={16} />
          </button>
        </div>
        <div className="dlg-body">{children}</div>
        {footer && <div className="dlg-foot">{footer}</div>}
      </div>
    </div>
  );
}
