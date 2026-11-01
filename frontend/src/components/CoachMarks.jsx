import React, { useState } from "react";

// A short, dismissible walkthrough of the studio instrument panel — shown once
// on a user's first arrival, and replayable from the "?" button.
const STEPS = [
  { t: "Welcome to the Studio",
    d: "This is a live instrument panel for your model. As it trains, every panel updates in real time — you're watching it actually learn." },
  { t: "Loss & Gradient Norm",
    d: "The green trace is loss falling as the model improves; orange is gradient magnitude. A sudden bump often marks a new capability forming." },
  { t: "Attention Circuitry",
    d: "Each tile is one attention head, by layer. Watch them specialize — induction heads (copying from context) light up as in-context learning emerges." },
  { t: "Architecture",
    d: "The model's blocks and the live path through the forward pass, with parameter counts. Click a head to ablate it and see what breaks." },
  { t: "Test, Eval & Chat",
    d: "Switch tabs up top to prompt your model in the Test box, run a suite of Eval prompts as it trains, or hold a full streaming Chat." },
];

export default function CoachMarks({ onClose }) {
  const [i, setI] = useState(0);
  const s = STEPS[i];
  const last = i === STEPS.length - 1;
  return (
    <div className="coach-overlay" onClick={onClose}>
      <div className="coach" onClick={(e) => e.stopPropagation()}>
        <div className="coach-step">{i + 1} / {STEPS.length}</div>
        <h3 className="coach-t">{s.t}</h3>
        <p className="coach-d">{s.d}</p>
        <div className="coach-dots">
          {STEPS.map((_, j) => <span key={j} className={`coach-dot ${j === i ? "on" : ""}`} />)}
        </div>
        <div className="coach-actions">
          <button className="coach-skip" onClick={onClose}>Skip</button>
          <div className="coach-nav">
            {i > 0 && <button className="coach-back" onClick={() => setI(i - 1)}>← Back</button>}
            <button className="coach-next" onClick={() => (last ? onClose() : setI(i + 1))}>{last ? "Got it" : "Next →"}</button>
          </div>
        </div>
      </div>
    </div>
  );
}
