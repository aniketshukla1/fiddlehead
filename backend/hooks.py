"""
hooks.py — attach telemetry hooks to a PyTorch transformer.

Usage:
    from hooks import TelemetryHooks
    telem = TelemetryHooks(model, layer_names=["blocks.0.attn", ...])
    ...
    loss.backward()
    frame = telem.collect(step=step, loss=loss.item(), lr=lr, tokens=token_strs)
    await broadcast(frame)

Design: rather than assume a specific HF/nanoGPT/custom class hierarchy,
this looks for two conventions on your attention module, either of which
is a one-line addition to most implementations:

  1. `module.last_attn_weights` — a tensor of shape [batch, heads, seq, seq]
     set at the end of your attention forward().
  2. A forward hook that captures the softmax output directly, if your
     attention module returns weights as part of its output tuple.

It also hooks every parameter with `requires_grad` to accumulate a
per-layer gradient norm, and grabs the final hidden state for a 3D PCA
projection of token embeddings.
"""

from __future__ import annotations
import numpy as np
import torch

# A head gets a behavior label once its top score clears this bar; below it,
# the head is still "mixed" (forming) and stays unlabeled.
LABEL_THRESHOLD = 0.30
# Induction is a spread-out signal (its mass is split across every prior
# occurrence of the current token), so it earns a *label* at a slightly lower
# bar than the point-like behaviors — heads flicker "induction" as they form.
INDUCTION_LABEL_THRESHOLD = 0.25
# But declaring induction "emerged" model-wide (the marker pinned on the loss
# curve) is a stronger claim: a real induction head clears ~0.4 and stays,
# while random-init noise only ever grazes ~0.2. Requiring it to hold for a
# couple of steps rejects transient spikes so the marker lands on the bump.
EMERGENCE_THRESHOLD = 0.40
EMERGENCE_SUSTAIN = 2


class TelemetryHooks:
    def __init__(self, model: torch.nn.Module, attn_module_names: list[str] | None = None):
        self.model = model
        self._attn_cache: dict[str, torch.Tensor] = {}
        self._hidden_cache: torch.Tensor | None = None
        self._grad_norms: dict[str, float] = {}
        self._handles = []
        self._induction_step: int | None = None  # step an induction head first emerged
        self._emerge_count = 0  # consecutive steps above the emergence bar

        # Discover attention submodules: anything that stashes attention weights
        # (our `last_attn_weights` convention) or whose class name says so.
        if attn_module_names is None:
            attn_module_names = [
                name for name, m in model.named_modules()
                if hasattr(m, "last_attn_weights") or "attention" in type(m).__name__.lower()
            ]
        self.attn_module_names = attn_module_names

        for name, module in model.named_modules():
            if name in self.attn_module_names:
                self._handles.append(module.register_forward_hook(self._make_attn_hook(name)))

        # capture the last transformer block's output as "hidden states"
        blocks = [n for n, _ in model.named_modules() if n.count(".") == 0 and n != ""]
        if blocks:
            last = dict(model.named_modules())[blocks[-1]]
            self._handles.append(last.register_forward_hook(self._hidden_hook))

        # per-layer grad norm: group parameters by their top-level layer index
        self._param_groups: dict[str, list[torch.nn.Parameter]] = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            key = name.split(".")[1] if name.count(".") >= 1 else name
            self._param_groups.setdefault(key, []).append(p)

    def _make_attn_hook(self, name):
        def hook(module, inputs, output):
            weights = getattr(module, "last_attn_weights", None)
            if weights is None and isinstance(output, tuple) and len(output) > 1:
                weights = output[1]
            if weights is not None:
                self._attn_cache[name] = weights.detach()
        return hook

    def _hidden_hook(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        self._hidden_cache = h.detach()

    def _compute_grad_norms(self):
        norms = []
        for key, params in self._param_groups.items():
            total = 0.0
            for p in params:
                if p.grad is not None:
                    total += float(p.grad.data.norm(2).item() ** 2)
            norms.append(total ** 0.5)
        return norms

    def _project_embeddings_3d(self, tokens: list[str]) -> list[dict]:
        if self._hidden_cache is None:
            return []
        h = self._hidden_cache[0]  # first batch item: [seq, d_model]
        h = h.detach().float().cpu().numpy()
        h = h - h.mean(axis=0, keepdims=True)
        # SVD-based PCA, cheap for small seq lengths
        try:
            u, s, vt = np.linalg.svd(h, full_matrices=False)
            proj = u[:, :3] * s[:3]
        except np.linalg.LinAlgError:
            proj = np.zeros((h.shape[0], 3))
        out = []
        for i, tok in enumerate(tokens[: proj.shape[0]]):
            out.append({"token": tok, "x": float(proj[i, 0]), "y": float(proj[i, 1]), "z": float(proj[i, 2])})
        return out

    @staticmethod
    def _head_scores(a: np.ndarray, tokens: list[str]) -> dict[str, float]:
        """Score one head's [T, T] (causal) attention matrix for four canonical
        behaviors. Each score is roughly "fraction of attention mass spent
        doing X", averaged over query positions.

          prev      — attends to the immediately preceding token (i -> i-1)
          pos       — attends to itself / stays on the diagonal (i -> i)
          sink      — dumps attention on token 0 (the classic attention sink)
          induction — attends to the token *after* the previous occurrence of
                      the current token: the copy circuit [A][B]…[A] -> [B]
        """
        t = a.shape[0]
        if t < 2:
            return {"prev": 0.0, "pos": 0.0, "sink": 0.0, "induction": 0.0}

        prev = float(np.mean([a[i, i - 1] for i in range(1, t)]))
        pos = float(np.mean([a[i, i] for i in range(t)]))
        sink = float(np.mean([a[i, 0] for i in range(1, t)]))

        # induction: for each query, sum mass landing on {p+1 : tokens[p] == tokens[i]}
        ind_vals = []
        for i in range(1, t):
            targets = [p + 1 for p in range(i) if tokens[p] == tokens[i] and p + 1 < i]
            if targets:
                ind_vals.append(float(sum(a[i, tgt] for tgt in targets)))
        induction = float(np.mean(ind_vals)) if ind_vals else 0.0

        return {"prev": prev, "pos": pos, "sink": sink, "induction": induction}

    @staticmethod
    def _label(scores: dict[str, float]) -> str:
        """Name a head after its dominant behavior, or '' while still forming."""
        top = max(scores, key=scores.get)
        bar = INDUCTION_LABEL_THRESHOLD if top == "induction" else LABEL_THRESHOLD
        return top if scores[top] >= bar else ""

    def _classify_heads(self, present: list[str], tokens: list[str]) -> tuple[list[dict], float]:
        """Build the full circuit map — every captured layer, every head, with
        its attention matrix, behavior scores, and label. Returns
        (layers, strongest_induction_score_this_step)."""
        layers = []
        max_induction = 0.0
        for idx, name in enumerate(present):
            w = self._attn_cache[name][0].float().cpu().numpy()  # [heads, T, T]
            heads = []
            for h in range(w.shape[0]):
                scores = self._head_scores(w[h], tokens)
                max_induction = max(max_induction, scores["induction"])
                heads.append({
                    "matrix": np.round(w[h], 3).tolist(),
                    "label": self._label(scores),
                    **{k: round(v, 3) for k, v in scores.items()},
                })
            layers.append({"layer": name, "idx": idx, "heads": heads})
        return layers, max_induction

    def meta(self, model_name: str, dataset: str) -> dict:
        n_params = sum(p.numel() for p in self.model.parameters())
        return {
            "type": "meta",
            "model": model_name,
            "layers": len(self.attn_module_names),
            "heads": getattr(self.model, "n_heads", None) or getattr(self.model, "num_heads", None),
            "d_model": getattr(self.model, "d_model", None) or getattr(self.model, "n_embd", None),
            "params_m": round(n_params / 1e6, 2),
            "dataset": dataset,
        }

    def collect(self, step: int, loss: float, lr: float, tokens: list[str]) -> dict:
        grad_norms = self._compute_grad_norms()

        # classify every attention layer captured on this forward pass; the
        # full circuit map (matrices + behavior labels for all layers/heads)
        # is what the frontend renders as the model's live circuitry.
        present = [n for n in self.attn_module_names if n in self._attn_cache]
        layers, max_induction = self._classify_heads(present, tokens)
        seq = layers[0]["heads"][0]["matrix"] if layers else []
        n_tok = len(seq) if seq else len(tokens)

        if self._induction_step is None:
            self._emerge_count = self._emerge_count + 1 if max_induction >= EMERGENCE_THRESHOLD else 0
            if self._emerge_count >= EMERGENCE_SUSTAIN:
                self._induction_step = step

        frame = {
            "type": "step",
            "step": step,
            "loss": loss,
            "lr": lr,
            "grad_norm": float(np.sqrt(sum(n ** 2 for n in grad_norms))) if grad_norms else 0.0,
            "grad_norm_per_layer": grad_norms,
            "attention": {"tokens": tokens[:n_tok], "layers": layers} if layers else None,
            "induction_step": self._induction_step,
            "embeddings": self._project_embeddings_3d(tokens),
            "anomaly": bool(grad_norms and max(grad_norms) > 5.0),
        }
        self._attn_cache.clear()
        return frame

    def remove(self):
        for h in self._handles:
            h.remove()
