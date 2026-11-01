// Generates plausible-looking training telemetry so the dashboard is fully
// demoable without a live PyTorch backend. Shapes match hooks.py exactly:
// a per-step frame carrying the full circuit map (every layer/head's attention
// matrix + behavior label), a loss trace with an induction "bump", live
// generation copy-accuracy, and an emergence step.
//
// The story it stages: for the first ~40 steps every head is "forming"
// (near-uniform attention); then layer-2's heads specialize into an induction
// circuit, attention sinks appear in the last layer, the loss drops sharply,
// and the model starts copying its prompt verbatim.

const LAYERS = 4;
const HEADS = 2;
const CORE = 8;
// a random block immediately repeated — same task the real backend trains on
const CORE_TOKENS = ["r", "o", "s", "e", "b", "u", "d", "x"];
const TOKENS = [...CORE_TOKENS, ...CORE_TOKENS.slice(0, CORE - 1)]; // length 15
const SEQ = TOKENS.length;

// each head's eventual specialty; "" means it never fully specializes (stays
// mixed/forming). Mirrors what the real model tends to learn.
const DESTINY = [
  ["prev", ""],                 // layer 0 — an early previous-token head
  ["", "induction"],            // layer 1 — induction starts to appear
  ["induction", "induction"],   // layer 2 — the copy circuit
  ["sink", "sink"],             // layer 3 — attention sinks
];

const EMERGE = 55;              // step the induction marker latches
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const sigmoid = (x) => 1 / (1 + Math.exp(-x));

export function mockMeta() {
  return {
    type: "meta",
    model: "tiny-gpt-induction",
    layers: LAYERS,
    heads: HEADS,
    d_model: 64,
    params_m: 0.2,
    dataset: "repeated-random (demo)",
  };
}

// ---- the same behavior scorer the backend uses, ported to JS so mock labels
//      emerge from the generated matrices exactly as they would live ----
function headScores(a, tokens) {
  const T = a.length;
  let prev = 0, pos = 0, sink = 0;
  for (let i = 1; i < T; i++) { prev += a[i][i - 1]; sink += a[i][0]; }
  for (let i = 0; i < T; i++) pos += a[i][i];
  prev /= T - 1; sink /= T - 1; pos /= T;
  let ind = 0, cnt = 0;
  for (let i = 1; i < T; i++) {
    let s = 0, any = false;
    for (let p = 0; p < i; p++) {
      if (tokens[p] === tokens[i] && p + 1 < i) { s += a[i][p + 1]; any = true; }
    }
    if (any) { ind += s; cnt += 1; }
  }
  ind = cnt ? ind / cnt : 0;
  return { prev, pos, sink, induction: ind };
}

function labelOf(s) {
  let top = "prev";
  for (const k of ["pos", "sink", "induction"]) if (s[k] > s[top]) top = k;
  const bar = top === "induction" ? 0.25 : 0.3;
  return s[top] >= bar ? top : "";
}

// build a causal (lower-triangular) attention matrix that leans toward the
// given behavior with intensity `strength` in [0,1]; strength 0 => uniform.
function buildMatrix(behavior, strength, tokens) {
  const T = tokens.length;
  const K = 14 * strength;
  const mat = [];
  for (let i = 0; i < T; i++) {
    const row = new Array(T).fill(0);
    let targets = [];
    if (behavior === "prev" && i >= 1) targets = [i - 1];
    else if (behavior === "pos") targets = [i];
    else if (behavior === "sink") targets = [0];
    else if (behavior === "induction") {
      for (let p = 0; p < i; p++) if (tokens[p] === tokens[i] && p + 1 < i) targets.push(p + 1);
    }
    let sum = 0;
    for (let j = 0; j <= i; j++) {
      let w = 1 + (Math.random() - 0.5) * 0.3;      // uniform base + jitter
      if (targets.includes(j)) w += K;
      row[j] = Math.exp(w * 0.35);
      sum += row[j];
    }
    for (let j = 0; j <= i; j++) row[j] = +(row[j] / sum).toFixed(3);
    mat.push(row);
  }
  return mat;
}

let t = 0;
export function mockFrame(step, ablated = new Set()) {
  t += 1;
  const s = clamp((t - 40) / 30, 0, 1); // specialization ramp around EMERGE

  // is the layer-2 copy circuit intact? ablating its heads breaks copying.
  const indHeadsAblated = [ "2-0", "2-1" ].filter((k) => ablated.has(k)).length;
  const circuitHealth = t < EMERGE ? 0 : (2 - indHeadsAblated) / 2;

  const layers = DESTINY.map((heads, li) => ({
    layer: `blocks.${li}.attn`,
    idx: li,
    heads: heads.map((behavior, hi) => {
      const forming = behavior === "";
      const strength = forming ? 0.12 : s;
      const matrix = buildMatrix(forming ? "pos" : behavior, strength, TOKENS);
      const scores = headScores(matrix, TOKENS);
      return {
        matrix,
        label: labelOf(scores),
        prev: +scores.prev.toFixed(3),
        pos: +scores.pos.toFixed(3),
        sink: +scores.sink.toFixed(3),
        induction: +scores.induction.toFixed(3),
      };
    }),
  }));

  // loss: a plateau that breaks into the induction bump around EMERGE
  const noise = (Math.random() - 0.5) * 0.05;
  const loss = 1.5 + 1.85 * (1 - sigmoid((t - EMERGE) / 6)) + 0.18 * (1 - circuitHealth) + noise;

  const spike = Math.random() < 0.015;
  const gradNorm = (spike ? 3.4 : 0.55) + Math.random() * 0.35;
  const gradPerLayer = Array.from({ length: LAYERS }, (_, i) =>
    Math.max(0.02, gradNorm * (1 - i / (LAYERS + 2)) * (0.7 + Math.random() * 0.6))
  );

  // generation: copy accuracy tracks the circuit's health
  const match = Math.round(clamp(circuitHealth, 0, 1) * CORE - (1 - circuitHealth) * Math.random());
  const prompt = CORE_TOKENS.join("");
  const cont = CORE_TOKENS.map((c, i) =>
    i < match ? c : "abcdefghijklmnopqrstuvwxyz"[Math.floor(Math.random() * 26)]
  ).join("");

  const embeddings = TOKENS.map((tok, i) => {
    const angle = (i / TOKENS.length) * Math.PI * 2 + t * 0.01;
    const r = 2 + Math.sin(t * 0.02 + i) * 0.4 * s;
    return { token: tok, x: Math.cos(angle) * r, y: Math.sin(angle * 1.3) * r * 0.6, z: Math.sin(angle) * r };
  });

  return {
    type: "step",
    step,
    loss,
    lr: 1e-3,
    grad_norm: gradNorm,
    grad_norm_per_layer: gradPerLayer,
    attention: { tokens: TOKENS, layers },
    induction_step: t >= EMERGE ? EMERGE : null,
    embeddings,
    generation: { prompt, cont, match: Math.max(0, match), of: CORE },
    ablated: [...ablated].map((k) => k.split("-").map(Number)),
    frozen: ablated.size > 0,
    anomaly: spike,
  };
}
