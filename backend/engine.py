"""
engine.py — a controllable training engine the UI drives over the socket.

This is what turns Fiddlehead from "a script that trains one hardcoded task"
into a product: the frontend sends control messages (create a project from data
+ template + size, start/pause, generate from a prompt, ablate a head, save/load)
and the engine runs the training loop, streaming the same telemetry frames the
visualizer already understands.

Local and cloud compute are the same thing behind this: whoever runs the engine
(the user's machine or a remote worker) speaks this one protocol.
"""

from __future__ import annotations
import json
import math
import os
import time

import numpy as np
import torch

from artifacts import META, SAVE_DIR, meta_path
from model import GPT, GPTConfig
from data import build_task, TEMPLATES
from tokenizer import load_tokenizer
from hooks import TelemetryHooks
from server import TelemetryServer
import capabilities


# ---- production training configuration --------------------------------- #
# Every field is optional; the defaults reproduce the original behaviour, so
# existing specs and old checkpoints keep training exactly as before.
DEFAULT_TRAIN = dict(
    lr=1e-3, min_lr=0.0, weight_decay=0.01, beta1=0.9, beta2=0.999,
    grad_clip=1.0, warmup_steps=0, schedule="constant",
    batch_size=32, grad_accum=1, max_steps=0, label_smoothing=0.0,
    dropout=0.0, precision="fp32", seed=1337, checkpoint_every=0,
)
CHECKPOINT_KEEP = 8   # rolling in-memory checkpoints retained for rewind


def parse_train_cfg(spec: dict) -> dict:
    """Merge legacy top-level keys, a nested `train` block, and defaults into a
    fully-typed training config."""
    t = dict(DEFAULT_TRAIN)
    for legacy in ("lr", "batch_size"):          # back-compat with old specs
        if spec.get(legacy) is not None:
            t[legacy] = spec[legacy]
    t.update({k: v for k, v in (spec.get("train") or {}).items() if v is not None})
    fl = lambda k: float(t[k])
    it = lambda k: int(t[k])
    return dict(
        lr=fl("lr"), min_lr=fl("min_lr"), weight_decay=fl("weight_decay"),
        betas=(fl("beta1"), fl("beta2")), grad_clip=fl("grad_clip"),
        warmup_steps=max(0, it("warmup_steps")), schedule=str(t["schedule"]),
        batch_size=max(1, it("batch_size")), grad_accum=max(1, it("grad_accum")),
        max_steps=max(0, it("max_steps")), label_smoothing=fl("label_smoothing"),
        dropout=min(0.9, max(0.0, fl("dropout"))), precision=str(t["precision"]),
        seed=it("seed"), checkpoint_every=max(0, it("checkpoint_every")),
    )


def scheduled_lr(cfg: dict, step: int) -> float:
    """Warmup then constant / cosine / linear decay toward min_lr."""
    lr, min_lr, warm, total = cfg["lr"], cfg["min_lr"], cfg["warmup_steps"], cfg["max_steps"]
    if warm and step < warm:
        return lr * (step + 1) / warm
    if cfg["schedule"] == "constant" or total <= 0:
        return lr
    prog = min(1.0, max(0.0, (step - warm) / max(1, total - warm)))
    if cfg["schedule"] == "cosine":
        return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * prog))
    if cfg["schedule"] == "linear":
        return lr - (lr - min_lr) * prog
    return lr


class InferenceTask:
    """A loaded model is inference-only until re-created: it carries its
    tokenizer so you can generate/test, but has no training data attached."""
    trainable = False

    def __init__(self, tok, block_size: int, seed: str, kind: str):
        self.tok = tok
        self.block_size = block_size
        self._seed = seed
        self.kind = kind
        self.info = {"loaded": True}

    def default_prompt(self) -> str:
        return self._seed

    def evaluate(self, model) -> dict:
        return {}

    def get_batch(self, *a, **k):
        raise RuntimeError("loaded model is inference-only — create a project to train")

    def describe(self) -> dict:
        return {"kind": self.kind, "vocab_size": self.tok.vocab_size, "block_size": self.block_size, "loaded": True}


class TrainingEngine:
    def __init__(self, telemetry: TelemetryServer):
        self.telemetry = telemetry
        self.task = None
        self.model: GPT | None = None
        self.hooks: TelemetryHooks | None = None
        self.finetuner = None      # set instead of model on the fine-tune path
        self.opt = None
        self.device = capabilities.train_device()  # cuda on a cloud GPU, else cpu
        self.spec: dict = {}
        self.step = 0
        self.running = False
        self._last_gen = None
        self._last_loss = None
        self._last_eval = None
        self._reset_best()

    def _reset_best(self):
        """Forget best-checkpoint / early-stopping state for a fresh run."""
        self._best_score = None
        self._best_step = None
        self._best_eval = None
        self._best_state = None   # a CPU copy of the best from-scratch weights
        self._no_improve = 0
        self._loss_curve = []     # downsampled [step, loss] for run comparison
        self._checkpoints = []    # rolling [{step, loss, state}] for rewind/resume

    def _record_curve(self):
        """Append a downsampled loss point, halving the series if it grows large."""
        if self._last_loss is None:
            return
        self._loss_curve.append([self.step, round(float(self._last_loss), 4)])
        if len(self._loss_curve) > 250:
            self._loss_curve = self._loss_curve[::2]

    @staticmethod
    def _eval_score(ev):
        """A single 'lower is better' score for best-checkpoint selection and
        early stopping — val loss if present, else (1 - accuracy)."""
        if not ev:
            return None
        if ev.get("val_loss") is not None:
            return float(ev["val_loss"])
        for k in ("next_char_accuracy", "copy_accuracy", "sort_accuracy"):
            if ev.get(k) is not None:
                return 1.0 - float(ev[k])
        return None

    def _update_best(self, ev, model):
        """Track the best eval so far; snapshot weights (from-scratch) and honour
        an optional early-stop patience window."""
        score = self._eval_score(ev)
        if score is None:
            return
        if self._best_score is None or score < self._best_score - 1e-6:
            self._best_score, self._best_step, self._best_eval = score, self.step, ev
            if model is not None:
                self._best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self._no_improve = 0
        else:
            self._no_improve += 1
            es = self.spec.get("early_stop") or {}
            patience = int(es.get("patience", 5) or 5)
            if es.get("enabled") and self._no_improve >= patience:
                self.running = False
                self.telemetry.publish({"type": "status", "state": "ready", "step": self.step,
                                        "has_model": True, "early_stopped": True,
                                        "best_step": self._best_step})

    # ---- lifecycle ------------------------------------------------------- #
    def create(self, spec: dict):
        """Build a fresh project: task from data/template, model from a size
        preset. The user chose a word ('small'); we pick the architecture."""
        # gathering/scraping/BPE-training can take a moment — tell the UI
        self.telemetry.publish({"type": "status", "state": "preparing"})
        self._reset_best()

        if spec.get("template") in ("finetune", "asr", "vlm", "tts"):
            try:
                tmpl = spec["template"]
                if tmpl == "asr":
                    from asr import AsrTuner
                    self.finetuner = AsrTuner(spec, device=self.device)
                elif tmpl == "vlm":
                    from vlm import VlmTuner
                    self.finetuner = VlmTuner(spec, device=self.device)
                elif tmpl == "tts":
                    from tts import TtsTuner
                    self.finetuner = TtsTuner(spec, device=self.device)
                else:
                    from finetune import FineTuner
                    self.finetuner = FineTuner(spec, device=self.device)
            except ImportError:
                need = {"asr": "transformers + peft + soundfile + librosa",
                        "vlm": "transformers + peft + pillow",
                        "tts": "transformers + peft + soundfile"}.get(spec["template"], "transformers + peft + accelerate")
                self.telemetry.publish({"type": "error", "message": f"This model type needs: {need}"})
                self.publish_status()
                return
            except Exception as e:   # bad/missing data (e.g. no audio pairs) → report, don't crash
                self.telemetry.publish({"type": "error", "message": f"Couldn't build the model: {e}"})
                self.publish_status()
                return
            self.model, self.hooks, self.task = None, None, None
            self.spec = {**spec, "size": spec.get("method", "lora")}
            self.step = 0
            self.running = bool(spec.get("autostart", True))
            self.telemetry.publish(self.finetuner.meta())
            self.publish_status()
            return
        self.finetuner = None

        self.task = build_task(spec)
        size = spec.get("size", "small")
        moe = spec.get("moe") or {}
        cfg = parse_train_cfg(spec)
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"] & 0xFFFFFFFF)
        config = GPTConfig.from_preset(size, self.task.tok.vocab_size, self.task.block_size,
                                       n_experts=int(moe.get("experts", 0) or 0),
                                       moe_top_k=int(moe.get("top_k", 2) or 2),
                                       dropout=cfg["dropout"])
        self.model = GPT(config).to(self.device)
        self.hooks = TelemetryHooks(self.model)
        self.train_cfg = cfg
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg["lr"],
                                     betas=cfg["betas"], weight_decay=cfg["weight_decay"])
        self._scaler = torch.amp.GradScaler("cuda", enabled=(cfg["precision"] == "fp16" and self.device == "cuda"))
        self.spec = {**spec, "size": size}
        self.step = 0
        self.running = bool(spec.get("autostart", True))
        self._last_gen = self._sample()
        self.publish_meta()
        self.publish_status()

    def _prepare(self, msg: dict):
        """Gather the configured data and report stats + a sample WITHOUT training,
        so the user can sanity-check what they're about to train on (did the PDF
        extract cleanly? did the scrape find anything?)."""
        import sources
        spec = msg.get("spec", msg)
        mid = msg.get("id")
        self.telemetry.publish({"type": "status", "state": "preparing"})
        try:
            data_spec = spec.get("data") or {"source": "paste", "text": spec.get("text", "")}
            docs = sources.gather(data_spec)
            info = sources.corpus_stats(docs)
            kind = (spec.get("tokenizer") or {}).get("kind", "char")
            joined = "".join(d.text for d in docs if d.text)
            # char count is exact for char tokens; BPE typically compresses ~4x
            est = info["chars"] if kind == "char" else info["chars"] // 4
            vocab = len(set(joined)) if kind == "char" else int((spec.get("tokenizer") or {}).get("vocab_size", 512))
            self.telemetry.publish({"type": "prepared", "id": mid, **info,
                                    "est_tokens": est, "tokenizer": kind, "vocab_size": vocab})
        except Exception as e:
            self.telemetry.publish({"type": "error", "id": mid, "message": f"preview failed: {e}"})
        self.publish_status()

    def publish_meta(self):
        if not self.model:
            return
        meta = self.hooks.meta(model_name=self.spec.get("name", f"{self.spec.get('size','small')}-model"),
                               dataset=self.task.kind)
        meta.update({
            "template": self.task.kind,
            "template_desc": TEMPLATES.get(self.task.kind, ""),
            "size": self.spec.get("size", "small"),
            "block_size": self.task.block_size,
            "vocab_size": self.task.tok.vocab_size,
            "params_m": round(self.model.num_params() / 1e6, 3),
            "tokenizer": getattr(self.task.tok, "kind", "char"),
            "data": getattr(self.task, "info", None),
            "trainable": getattr(self.task, "trainable", True),
            "n_experts": self.model.config.n_experts,
            "moe_top_k": self.model.config.moe_top_k,
            "dropout": getattr(self.model.config, "dropout", 0.0),
            "train": getattr(self, "train_cfg", None),
            "device": self.device,
        })
        self.telemetry.publish(meta)

    def publish_status(self):
        has = self.model is not None or self.finetuner is not None
        self.telemetry.publish({
            "type": "status",
            "state": "training" if self.running else ("ready" if has else "idle"),
            "step": self.step,
            "has_model": has,
        })

    # ---- one training step ---------------------------------------------- #
    def training_step(self):
        if self.finetuner is not None:
            frame = self.finetuner.train_step()
            self.step = self.finetuner.step
            if frame.get("loss") is not None:
                self._last_loss = frame["loss"]
            if frame.get("eval"):
                self._last_eval = frame["eval"]
                self._update_best(frame["eval"], None)
                self._record_curve()
            if self._best_step is not None:
                frame["best_step"] = self._best_step
                frame["best_headline"] = (self._best_eval or {}).get("headline")
            self.telemetry.publish(frame)
            return

        model, task, hooks, opt = self.model, self.task, self.hooks, self.opt
        cfg = getattr(self, "train_cfg", None) or parse_train_cfg(self.spec)
        model.train()

        frozen = any(blk.attn.ablated for blk in model.blocks)

        # learning-rate schedule: warmup, then constant / cosine / linear decay
        lr_now = scheduled_lr(cfg, self.step)
        for g in opt.param_groups:
            g["lr"] = lr_now

        # mixed precision — bf16 autocast works on CPU too; fp16 uses the CUDA scaler
        on_cuda = self.device == "cuda"
        want_amp = cfg["precision"] in ("fp16", "bf16")
        amp_dtype = torch.float16 if (cfg["precision"] == "fp16" and on_cuda) else torch.bfloat16

        # gradient accumulation: average `grad_accum` micro-batches into one step
        accum = cfg["grad_accum"]
        opt.zero_grad(set_to_none=True)
        task_loss = 0.0
        last_x = last_hidden = None
        for _ in range(accum):
            x, y = task.get_batch(cfg["batch_size"])
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast(device_type="cuda" if on_cuda else "cpu", dtype=amp_dtype, enabled=want_amp):
                logits, hidden = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                    label_smoothing=cfg["label_smoothing"])
            task_loss += loss.item()
            if getattr(model, "last_aux_loss", None) is not None:
                loss = loss + 0.01 * model.last_aux_loss   # MoE load-balancing pressure
            self._scaler.scale(loss / accum).backward()
            last_x, last_hidden = x, hidden
        task_loss /= accum
        hooks._hidden_cache = last_hidden.detach().float()   # fp32 for PCA under bf16/fp16
        if not frozen:
            self._scaler.unscale_(opt)

        # Telemetry comes before clipping, so the gradient norm is the real one
        # rather than pinned at the clip value, and before sampling, whose
        # forward passes would overwrite the captured attention and routing.
        self._last_loss = task_loss
        # per-position display labels for the attention map (works for char & BPE)
        tokens = [task.tok.decode([int(i)]) for i in last_x[0].tolist()]
        frame = hooks.collect(step=self.step, loss=self._last_loss, lr=lr_now, tokens=tokens)
        frame["ablated"] = model.ablation_state()
        frame["frozen"] = frozen
        if model.config.n_experts > 1:
            frame["experts"] = {"n": model.config.n_experts, "top_k": model.config.moe_top_k,
                                "load": model.expert_load(),
                                "aux": float(model.last_aux_loss.detach()) if model.last_aux_loss is not None else None}

        # gradient clipping (skipped during ablation freezes)
        if not frozen:
            if cfg["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            self._scaler.step(opt)
            self._scaler.update()

        if self.step % 8 == 0:
            self._last_gen = self._sample()
        frame["generation"] = self._last_gen
        if self.step % 20 == 0:
            model.eval()
            frame["eval"] = task.evaluate(model)
            model.train()
            self._last_eval = frame["eval"]
            self._update_best(frame["eval"], model)
            self._record_curve()
        if self._best_step is not None:
            frame["best_step"] = self._best_step
            frame["best_headline"] = (self._best_eval or {}).get("headline")

        # periodic checkpoints for rewind / resume-from-step (in-memory ring)
        ce = cfg["checkpoint_every"]
        if ce and self.step > 0 and self.step % ce == 0 and not (
                self._checkpoints and self._checkpoints[-1]["step"] == self.step):
            self._checkpoints.append({
                "step": self.step,
                "loss": round(float(self._last_loss), 4) if self._last_loss is not None else None,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            })
            if len(self._checkpoints) > CHECKPOINT_KEEP:
                self._checkpoints.pop(0)
        if self._checkpoints:
            frame["checkpoints"] = [{"step": c["step"], "loss": c["loss"]} for c in self._checkpoints]

        self.telemetry.publish(frame)
        self.step += 1

        # honor an explicit training budget (max_steps): auto-pause when reached
        if cfg["max_steps"] and self.step >= cfg["max_steps"] and self.running:
            self.running = False
            self.telemetry.publish({"type": "status", "state": "ready", "step": self.step,
                                    "has_model": True, "reason": "max_steps"})

    # ---- generation (the test box) -------------------------------------- #
    def _gen_params(self):
        deterministic = self.task.kind in ("copy", "sort")
        return (0.1 if deterministic else float(self.spec.get("temperature", 0.8)),
                1 if deterministic else int(self.spec.get("top_k", 40)))

    def _preview_len(self) -> int:
        # kept short for the always-on training ticker so it doesn't slow the
        # loop; the on-demand test box generates full-length on request.
        return (getattr(self.task, "core_len", None) or getattr(self.task, "seq_len", None)
                or min(32, self.task.block_size))

    def _sample(self):
        if self.model is not None:
            self.model.eval()   # generation should ignore dropout
        prompt = self.task.default_prompt()
        ids = torch.tensor([self.task.tok.encode(prompt)], dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return None
        temp, top_k = self._gen_params()
        out = self.model.generate(ids, self._preview_len(), temperature=temp, top_k=top_k)
        cont = self.task.tok.decode(out[0])
        g = {"prompt": prompt, "cont": cont}
        if self.task.kind == "copy":
            g["match"] = sum(a == b for a, b in zip(prompt, cont))
            g["of"] = len(prompt)
        return g

    @staticmethod
    def _trim_chat(text: str, is_chat: bool, stops) -> str:
        """Keep only this turn's reply (cut at the next role marker)."""
        if not is_chat:
            return text
        cut = min([i for i in (text.find(s) for s in stops) if i != -1], default=-1)
        return (text[:cut] if cut != -1 else text).strip()

    def _emit(self, mid, prompt, is_chat, params, text, done):
        self.telemetry.publish({"type": "sample", "id": mid, "chat": is_chat, "prompt": prompt,
                                "text": text, "params": params, "done": done})

    def _generate_ft(self, msg: dict):
        ft = self.finetuner
        messages = msg.get("messages")
        is_chat = bool(messages)
        prompt = ft.format_chat(messages, msg.get("system")) if is_chat else str(msg.get("prompt", ft._seed))
        params = {"temperature": float(msg.get("temperature", 0.8)),
                  "top_k": int(msg.get("top_k", 40) or 0), "top_p": float(msg.get("top_p", 1.0)),
                  "max_new_tokens": int(msg.get("max_new_tokens", 96 if is_chat else 48))}
        mid = msg.get("id")
        on_token = lambda t: self._emit(mid, prompt, is_chat, params, self._trim_chat(t, is_chat, ft.chat_stops), False)
        text = ft.generate_text(prompt, params, on_token=on_token)
        self._emit(mid, prompt, is_chat, params, self._trim_chat(text, is_chat, ft.chat_stops), True)

    def generate(self, msg: dict):
        """On-demand generation — powers both the Test box (a raw prompt) and the
        Chat window. Streams partial `sample` frames (done=False) as tokens arrive,
        then a final one (done=True)."""
        if self.finetuner is not None:
            return self._generate_ft(msg)
        if not self.model:
            return
        messages = msg.get("messages")
        is_chat = bool(messages)
        prompt = (self.task.format_chat(messages, msg.get("system")) if is_chat
                  else str(msg.get("prompt", self.task.default_prompt())))

        d_temp, d_top_k = self._gen_params()
        temperature = float(msg.get("temperature", d_temp))
        top_k = int(msg.get("top_k", d_top_k) or 0) or None
        top_p = float(msg.get("top_p", 1.0))
        n = int(msg.get("max_new_tokens", 96 if is_chat else self._preview_len() * 2))
        eos = getattr(self.task, "eos_token_id", None)
        params = {"temperature": temperature, "top_k": top_k, "top_p": top_p, "max_new_tokens": n}
        mid = msg.get("id")
        stops = getattr(self.task, "chat_stops", ("\nUser:", "\nSystem:", "\nAssistant:"))

        on_token = lambda new_ids: self._emit(
            mid, prompt, is_chat, params, self._trim_chat(self.task.tok.decode(new_ids[0]), is_chat, stops), False)
        ids = torch.tensor([self.task.tok.encode(prompt) or [0]], dtype=torch.long, device=self.device)
        out = self.model.generate(ids, n, temperature=temperature, top_k=top_k, top_p=top_p,
                                  eos_token_id=eos, on_token=on_token)
        self._emit(mid, prompt, is_chat, params, self._trim_chat(self.task.tok.decode(out[0]), is_chat, stops), True)

    # ---- persistence (a saved-models library) ---------------------------- #
    def save(self, msg: dict):
        """Save the current model into saved/<name>/ with a fiddlehead.json so
        it can be listed and loaded back later (from-scratch or fine-tune)."""
        name = str(msg.get("name", "model")).replace("/", "_").strip() or "model"
        # "best" saves the best-eval checkpoint (from-scratch) instead of the latest
        use_best = msg.get("which") == "best" and self._best_state is not None
        d = os.path.join(SAVE_DIR, name)
        os.makedirs(d, exist_ok=True)
        meta = {"name": name, "created": int(time.time())}
        if self.finetuner is not None:
            ft = self.finetuner
            ft.save(d)
            is_asr = hasattr(ft, "transcribe")
            is_vlm = hasattr(ft, "caption")
            is_tts = hasattr(ft, "synthesize")
            meta.update({"kind": "tts" if is_tts else "vlm" if is_vlm else "asr" if is_asr else "finetune",
                         "modality": "speech-out" if is_tts else "vision" if is_vlm else "speech" if is_asr else "text",
                         "base_model": ft.base, "method": ft.method,
                         "params_m": round(ft.total / 1e6, 2), "block_size": ft.block_size,
                         "step": self.step, "seed": getattr(ft, "_seed", ""),
                         "final_loss": getattr(self, "_last_loss", None),
                         "eval": getattr(self, "_last_eval", None),
                         "loss_curve": self._loss_curve,
                         "samples": [self._last_gen] if getattr(self, "_last_gen", None) else []})
        elif self.model is not None:
            state = self._best_state if use_best else self.model.state_dict()
            save_step = self._best_step if use_best else self.step
            save_eval = self._best_eval if use_best else getattr(self, "_last_eval", None)
            save_loss = (save_eval or {}).get("val_loss") if use_best else getattr(self, "_last_loss", None)
            torch.save({"config": self.model.config.__dict__, "state_dict": state,
                        "spec": self.spec, "step": save_step, "tokenizer": self.task.tok.state(),
                        "seed": self.task.default_prompt(), "block_size": self.task.block_size,
                        "kind": self.task.kind}, os.path.join(d, "model.pt"))
            meta.update({"kind": "scratch", "template": self.task.kind, "size": self.spec.get("size", "small"),
                         "params_m": round(self.model.num_params() / 1e6, 3), "block_size": self.task.block_size,
                         "step": save_step, "checkpoint": "best" if use_best else "latest",
                         "n_experts": self.model.config.n_experts, "moe_top_k": self.model.config.moe_top_k,
                         "tokenizer": getattr(self.task.tok, "kind", "char"),
                         "vocab_size": self.task.tok.vocab_size,
                         "data": getattr(self.task, "info", None),
                         "final_loss": save_loss,
                         "eval": save_eval,
                         "loss_curve": self._loss_curve,
                         "samples": [self._last_gen] if getattr(self, "_last_gen", None) else []})
        else:
            return
        with open(os.path.join(d, META), "w") as f:
            json.dump(meta, f)
        self.telemetry.publish({"type": "saved", "name": name, "path": d})
        self.publish_models()

    def use_best(self):
        """Roll the live from-scratch model back to its best-eval weights and
        pause, so the user can inspect or save the peak instead of the latest."""
        if self.model is None or self._best_state is None:
            return
        self.model.load_state_dict({k: v.to(self.device) for k, v in self._best_state.items()})
        self.step = self._best_step or self.step
        self.running = False
        self._last_gen = self._sample()
        self.telemetry.publish({"type": "status", "state": "ready", "step": self.step,
                                "has_model": True, "rolled_back": True, "best_step": self._best_step})

    def rewind(self, msg: dict):
        """Resume-from-step: restore a periodic checkpoint's weights, set the
        step counter back to it, and keep training from there (honest curves —
        anything after the checkpoint is discarded)."""
        if self.model is None or not self._checkpoints:
            return
        target = msg.get("step")
        ckpt = next((c for c in reversed(self._checkpoints) if c["step"] == target), self._checkpoints[-1])
        self.model.load_state_dict({k: v.to(self.device) for k, v in ckpt["state"].items()})
        self.step = int(ckpt["step"])
        self._loss_curve = [p for p in self._loss_curve if p[0] <= self.step]
        self._checkpoints = [c for c in self._checkpoints if c["step"] <= self.step]
        self._no_improve = 0
        self._last_gen = self._sample()
        self.telemetry.publish({"type": "status",
                                "state": "training" if self.running else "ready",
                                "step": self.step, "has_model": True, "rewound_to": self.step,
                                "checkpoints": [{"step": c["step"], "loss": c["loss"]} for c in self._checkpoints]})

    def load(self, msg: dict):
        name = str(msg.get("name", "model")).replace("/", "_")
        d = os.path.join(SAVE_DIR, name)
        meta_file = meta_path(d)
        if not os.path.isfile(meta_file):
            self.telemetry.publish({"type": "error", "message": f"no saved model '{name}'"})
            return
        self.telemetry.publish({"type": "status", "state": "preparing"})
        meta = json.load(open(meta_file))

        if meta.get("kind") in ("finetune", "asr", "vlm", "tts"):
            k = meta.get("kind")
            if k == "asr":
                from asr import AsrTuner
                self.finetuner = AsrTuner.load(d, device=self.device)
                self._last_gen = None
            elif k == "vlm":
                from vlm import VlmTuner
                self.finetuner = VlmTuner.load(d, device=self.device)
                self._last_gen = None
            elif k == "tts":
                from tts import TtsTuner
                self.finetuner = TtsTuner.load(d, device=self.device)
                self._last_gen = None
            else:
                from finetune import FineTuner
                self.finetuner = FineTuner.load(d, device=self.device)
                self._last_gen = self.finetuner._sample()
            self.model, self.hooks, self.task = None, None, None
            self.spec = {"template": meta.get("kind"), "base_model": meta.get("base_model"),
                         "method": meta.get("method"), "size": meta.get("method", "lora"), "name": name}
            self.step = meta.get("step", 0)
            self.running = False
            self.telemetry.publish(self.finetuner.meta())
            self.publish_status()
            return

        self.finetuner = None
        ckpt = torch.load(os.path.join(d, "model.pt"), weights_only=False)
        self.spec = ckpt["spec"]
        self.model = GPT(GPTConfig(**ckpt["config"])).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.hooks = TelemetryHooks(self.model)
        cfg = parse_train_cfg(self.spec)
        self.train_cfg = cfg
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg["lr"],
                                     betas=cfg["betas"], weight_decay=cfg["weight_decay"])
        self._scaler = torch.amp.GradScaler("cuda", enabled=(cfg["precision"] == "fp16" and self.device == "cuda"))
        tok = load_tokenizer(ckpt["tokenizer"])  # rebuild tokenizer; never re-gather
        self.task = InferenceTask(tok, ckpt.get("block_size", self.model.block_size),
                                  ckpt.get("seed", ""), ckpt.get("kind", "text"))
        self.step = ckpt.get("step", 0)
        self.running = False
        self._last_gen = self._sample()
        self.publish_meta()
        self.publish_status()

    def list_models(self) -> list[dict]:
        out = []
        if os.path.isdir(SAVE_DIR):
            for name in sorted(os.listdir(SAVE_DIR)):
                mp = meta_path(os.path.join(SAVE_DIR, name))
                if os.path.isfile(mp):
                    try:
                        out.append(json.load(open(mp)))
                    except Exception:
                        pass
        return out

    def publish_models(self):
        self.telemetry.publish({"type": "models", "models": self.list_models()})

    def publish_datasets(self):
        import sources
        self.telemetry.publish({"type": "datasets", "datasets": sources.list_datasets()})

    def save_dataset(self, msg: dict):
        """Gather the configured data once and store it under a name for reuse."""
        import sources
        spec = msg.get("spec", msg)
        try:
            docs = sources.gather(spec.get("data") or {})
            meta = sources.save_dataset(spec.get("name") or "dataset", docs)
            self.telemetry.publish({"type": "dataset_saved", "name": meta["name"]})
            self.publish_datasets()
        except Exception as e:
            self.telemetry.publish({"type": "error", "message": f"save dataset failed: {e}"})

    def delete_model(self, msg: dict):
        name = str(msg.get("name", "")).replace("/", "_")
        d = os.path.join(SAVE_DIR, name)
        if os.path.isdir(d) and os.path.isfile(meta_path(d)):
            import shutil
            shutil.rmtree(d, ignore_errors=True)
        self.publish_models()

    def export_model(self, msg: dict):
        """Zip a saved model and send it back (base64) so the browser can download
        it — a portable artifact you can load with serve.py or your own code."""
        import base64
        import io
        import zipfile
        name = str(msg.get("name", "")).replace("/", "_")
        d = os.path.join(SAVE_DIR, name)
        if not os.path.isfile(meta_path(d)):
            self.telemetry.publish({"type": "error", "message": f"no saved model '{name}'"})
            return
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(d):
                for fn in files:
                    fp = os.path.join(root, fn)
                    z.write(fp, os.path.join(name, os.path.relpath(fp, d)))
        data = buf.getvalue()
        if len(data) > 48 * 1024 * 1024:
            self.telemetry.publish({"type": "error",
                "message": f"'{name}' is too big to download in-app — it's on disk at {d}"})
            return
        self.telemetry.publish({"type": "exported", "name": name, "filename": f"{name}.zip",
                                "b64": base64.b64encode(data).decode(), "path": d})

    # ---- control dispatch ------------------------------------------------ #
    def apply_control(self, msg: dict):
        t = msg.get("type")
        if t == "create":
            self.create(msg.get("spec", msg))
        elif t == "start":
            has = self.model is not None or self.finetuner is not None
            if self.task is not None:
                trainable = getattr(self.task, "trainable", True)
            elif self.finetuner is not None:
                trainable = getattr(self.finetuner, "is_trainable", True)
            else:
                trainable = True
            self.running = has and trainable
            self.publish_status()
        elif t == "pause":
            self.running = False
            self.publish_status()
        elif t == "reset":
            if self.spec:
                self.create({**self.spec, "autostart": False})
        elif t == "use_best":
            self.use_best()
        elif t == "rewind":
            self.rewind(msg)
        elif t == "capabilities":
            self.telemetry.publish(capabilities.report())
        elif t == "prepare":
            self._prepare(msg)
        elif t in ("resync", "status"):
            # a (re)connecting client asks for the current project + state
            if self.finetuner is not None:
                self.telemetry.publish(self.finetuner.meta())
            elif self.model is not None:
                self.publish_meta()
            self.publish_status()
        elif t == "generate":
            self.generate(msg)
        elif t == "transcribe":
            ft = self.finetuner
            if ft is not None and hasattr(ft, "transcribe"):
                try:
                    text = ft.transcribe(audio=msg.get("audio"))
                    self.telemetry.publish({"type": "sample", "id": msg.get("id"), "chat": False,
                                            "prompt": "[audio]", "text": text, "done": True})
                except Exception as e:
                    self.telemetry.publish({"type": "error", "id": msg.get("id"), "message": f"transcribe failed: {e}"})
        elif t == "caption":
            ft = self.finetuner
            if ft is not None and hasattr(ft, "caption"):
                try:
                    text = ft.caption(image=msg.get("image"))
                    self.telemetry.publish({"type": "sample", "id": msg.get("id"), "chat": False,
                                            "prompt": "[image]", "text": text, "done": True})
                except Exception as e:
                    self.telemetry.publish({"type": "error", "id": msg.get("id"), "message": f"caption failed: {e}"})
        elif t == "speak":
            ft = self.finetuner
            if ft is not None and hasattr(ft, "synthesize"):
                try:
                    b64 = ft.synthesize(msg.get("text"))
                    self.telemetry.publish({"type": "sample", "id": msg.get("id"), "chat": False,
                                            "prompt": msg.get("text", ""), "text": "", "audio": b64, "done": True})
                except Exception as e:
                    self.telemetry.publish({"type": "error", "id": msg.get("id"), "message": f"synthesis failed: {e}"})
        elif t == "save":
            self.save(msg)
        elif t == "load":
            self.load(msg)
        elif t == "list_models":
            self.publish_models()
        elif t == "delete_model":
            self.delete_model(msg)
        elif t == "save_dataset":
            self.save_dataset(msg)
        elif t == "list_datasets":
            self.publish_datasets()
        elif t == "delete_dataset":
            import sources
            self.telemetry.publish({"type": "datasets", "datasets": sources.delete_dataset(msg.get("name", ""))})
        elif t == "export":
            self.export_model(msg)
        elif t == "ablate":
            if self.model and 0 <= msg.get("layer", -1) < len(self.model.blocks):
                ablated = self.model.blocks[msg["layer"]].attn.ablated
                (ablated.add if msg.get("on", True) else ablated.discard)(msg.get("head"))


def main():
    telemetry = TelemetryServer()
    telemetry.start()
    time.sleep(0.3)
    engine = TrainingEngine(telemetry)
    telemetry.publish(capabilities.report())  # announce what this machine can train
    engine.publish_status()  # announce "idle" so a connecting UI knows we're here

    while True:
        for msg in telemetry.poll_control():
            try:
                engine.apply_control(msg)
            except Exception as e:  # never let one bad message kill the loop
                telemetry.publish({"type": "error", "message": str(e)})
        if engine.running and (engine.model is not None or engine.finetuner is not None):
            try:
                engine.training_step()
            except Exception as e:
                engine.running = False
                telemetry.publish({"type": "error", "message": f"training stopped: {e}"})
                engine.publish_status()
            time.sleep(0.04)
        else:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
