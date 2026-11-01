import React, { useState } from "react";
import { RefreshCw } from "lucide-react";
import Dialog from "./Dialog.jsx";
import CodeBlock from "./CodeBlock.jsx";

const START = `cd backend
pip install -r requirements.txt   # first time only
python engine.py                  # serves ws://localhost:8765`;

// Where the training engine stands, and exactly how to start it when it's down.
export default function EngineDialog({ engine, onClose }) {
  const [checking, setChecking] = useState(false);
  const caps = engine.caps;

  async function again() {
    setChecking(true);
    await engine.check();
    setChecking(false);
  }

  return (
    <Dialog title="Training engine" onClose={onClose} footer={
      <>
        <button type="button" className="btn" onClick={again} disabled={checking}>
          <RefreshCw size={14} className={checking ? "spin" : ""} /> {checking ? "Checking…" : "Check again"}
        </button>
        <button type="button" className="btn btn-primary" onClick={onClose}>Done</button>
      </>
    }>
      <div className={`eng-state ${engine.online ? "ok" : "off"}`}>
        <span className="status-dot" />
        <div>
          <div className="eng-state-t">{engine.online ? "Running on this machine" : "Not running"}</div>
          <div className="eng-state-s">
            {engine.online
              ? caps ? `${caps.cpu_cores} CPU cores · ${caps.ram_gb} GB RAM · ${caps.accelerator}` : "ws://localhost:8765"
              : "Models train inside the engine, a small Python process next to this app."}
          </div>
        </div>
      </div>
      {!engine.online && (
        <>
          <p className="dlg-p">Start it from the project folder in a terminal:</p>
          <CodeBlock code={START} lang="shell" />
          <p className="dlg-p muted">Leave it running, then come back to this tab — it's detected automatically.
            Training on a GPU server instead? Choose <b>Remote engine</b> when you create a model.</p>
        </>
      )}
    </Dialog>
  );
}
