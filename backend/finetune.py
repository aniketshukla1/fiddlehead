"""
finetune.py — fine-tune a real open-source model on the user's data.

Loads a pretrained HuggingFace causal LM, optionally wraps it with LoRA (few
trainable params → fast + low memory), and trains it on the corpus produced by
the same data pipeline (sources -> chunk). It streams the same telemetry frames
the visualizer already understands: loss, a plain-language quality readout, live
generations, and a *downsampled* attention circuit map (a few layers/heads run
through the same behavior classifier).

Guarded behind `transformers`/`peft` — the engine only offers this template when
they're installed. CPU-realistic on small models (distilgpt2, SmolLM-135M,
pythia-160m) with LoRA; the bigger ones are a cloud/GPU job.
"""

from __future__ import annotations
import json
import os
import numpy as np
import torch

import sources
from artifacts import meta_path
from hooks import TelemetryHooks

# curated small models that are realistic to fine-tune locally
SUGGESTED_MODELS = [
    "distilgpt2", "gpt2", "HuggingFaceTB/SmolLM-135M", "EleutherAI/pythia-160m",
    "Qwen/Qwen2.5-0.5B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]


class FineTuner:
    ATTN_EVERY = 12
    GEN_EVERY = 8
    EVAL_EVERY = 24

    def __init__(self, spec: dict, device: str = "cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.base = spec.get("base_model", "distilgpt2")
        self.method = spec.get("method", "lora")
        self.device = device
        self.step = 0
        self._last_gen = None

        self.tokenizer = AutoTokenizer.from_pretrained(self.base)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        self.model = AutoModelForCausalLM.from_pretrained(self.base, attn_implementation="eager")
        cfg = self.model.config
        self.n_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0))
        self.n_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0))
        max_pos = getattr(cfg, "max_position_embeddings", getattr(cfg, "n_positions", 1024)) or 1024
        self.block_size = min(int((spec.get("chunking") or {}).get("block_size", 128)), max_pos, 256)

        lora = spec.get("lora") or {}
        lr_cfg = spec.get("train") or {}
        r = int(lora.get("r", 8)); alpha = int(lora.get("alpha", 2 * r)); ldrop = float(lora.get("dropout", 0.05))
        if self.method == "lora":
            from peft import LoraConfig, get_peft_model
            try:
                lcfg = LoraConfig(task_type="CAUSAL_LM", r=r, lora_alpha=alpha, lora_dropout=ldrop,
                                  target_modules="all-linear")
                self.model = get_peft_model(self.model, lcfg)
            except Exception:
                lcfg = LoraConfig(task_type="CAUSAL_LM", r=r, lora_alpha=alpha, lora_dropout=ldrop)
                self.model = get_peft_model(self.model, lcfg)

        self.model.to(self.device)
        self.model.train()
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.trainable_params = sum(p.numel() for p in params)
        self.total = sum(p.numel() for p in self.model.parameters())
        self.is_trainable = True
        self.opt = torch.optim.AdamW(params, lr=float(lr_cfg.get("lr", spec.get("lr", 2e-4))),
                                     weight_decay=float(lr_cfg.get("weight_decay", 0.0)))
        self.lora_meta = {"r": r, "alpha": alpha, "dropout": ldrop} if self.method == "lora" else None
        self.grad_clip = float(lr_cfg.get("grad_clip", 1.0))

        self._build_data(spec)

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        """Reconstruct a saved fine-tune for inference (chat/test) — loads the
        base model + LoRA adapter (or full weights) + tokenizer, no training data."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        meta = json.load(open(meta_path(path)))
        self = cls.__new__(cls)
        self.base = meta.get("base_model", "distilgpt2")
        self.method = meta.get("method", "lora")
        self.device = device
        self.step = meta.get("step", 0)
        self._last_gen = None
        self.is_trainable = False
        self._seed = meta.get("seed", "")
        self.info = {"loaded": True}
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        if self.method == "lora":
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(self.base, attn_implementation="eager")
            self.model = PeftModel.from_pretrained(base, path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path, attn_implementation="eager")
        self.model.to(device).eval()
        cfg = self.model.config
        self.n_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0))
        self.n_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0))
        max_pos = getattr(cfg, "max_position_embeddings", getattr(cfg, "n_positions", 1024)) or 1024
        self.block_size = min(int(meta.get("block_size", 128)), max_pos, 256)
        self.total = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = 0
        return self

    # ---- data ------------------------------------------------------------ #
    def _build_data(self, spec):
        data_spec = spec.get("data") or {"source": "paste", "text": spec.get("text", "")}
        docs = sources.gather(data_spec)
        self.info = sources.corpus_stats(docs)
        text = "\n\n".join(d.text for d in docs if d.text) or "hello world.\n"
        ids = self.tokenizer(text, return_tensors="pt").input_ids[0]
        n = max(1, int(len(ids) * 0.9))
        self.train_ids, self.val_ids = ids[:n], (ids[n:] if len(ids) - n > self.block_size else ids[:n])
        self.info.update({"train_tokens": int(len(self.train_ids)), "val_tokens": int(len(self.val_ids)),
                          "tokenizer": "hf"})
        self._seed = text[: self.block_size // 2].strip() or text[:1]

    def _batch(self, data, bs=4):
        blk = min(self.block_size, max(2, len(data) - 1))
        hi = max(1, len(data) - blk - 1)
        ix = torch.randint(0, hi, (bs,))
        return torch.stack([data[i: i + blk] for i in ix]).to(self.device)

    # ---- meta ------------------------------------------------------------ #
    def meta(self) -> dict:
        return {
            "type": "meta", "model": self.base, "template": "finetune",
            "base_model": self.base, "method": self.method,
            "size": self.method, "layers": self.n_layers, "heads": self.n_heads, "device": self.device,
            "block_size": self.block_size, "vocab_size": len(self.tokenizer), "tokenizer": "hf",
            "params_m": round(self.trainable_params / 1e6, 3), "total_params_m": round(self.total / 1e6, 2),
            "data": self.info, "trainable": self.is_trainable,
            "template_desc": (f"loaded {self.base} ({self.method.upper()})" if not self.is_trainable else
                              f"fine-tuning {self.base} ({self.method.upper()}) — "
                              f"{self.trainable_params/1e6:.2f}M of {self.total/1e6:.1f}M params trainable"),
        }

    # ---- one step -> a telemetry frame ----------------------------------- #
    def train_step(self) -> dict:
        x = self._batch(self.train_ids)
        out = self.model(input_ids=x, labels=x)
        loss = out.loss
        self.opt.zero_grad()
        loss.backward()
        clip = self.grad_clip if getattr(self, "grad_clip", 0) > 0 else 1e9
        gnorm = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], clip)
        self.opt.step()

        if self.step % self.GEN_EVERY == 0:
            self._last_gen = self._sample()

        frame = {
            "type": "step", "step": self.step, "loss": float(loss.item()),
            "lr": self.opt.param_groups[0]["lr"], "grad_norm": float(gnorm),
            "grad_norm_per_layer": [], "attention": None, "embeddings": [],
            "induction_step": None, "anomaly": False,
            "generation": self._last_gen, "ablated": [], "frozen": False,
        }
        if self.step % self.ATTN_EVERY == 0:
            frame["attention"] = self._attention_map()
        if self.step % self.EVAL_EVERY == 0:
            frame["eval"] = self.evaluate()
        self.step += 1
        return frame

    # ---- attention circuit map (downsampled) ----------------------------- #
    @torch.no_grad()
    def _attention_map(self):
        try:
            x = self._batch(self.val_ids, bs=1)[:, : min(24, self.block_size)]
            out = self.model(input_ids=x, output_attentions=True)
            attns = out.attentions  # tuple(L) of [1, heads, T, T]
            if not attns:
                return None
            toks = [self.tokenizer.decode([int(t)]) for t in x[0].tolist()]
            L = len(attns)
            layer_ids = sorted(set(int(i) for i in np.linspace(0, L - 1, min(4, L))))
            layers = []
            for li in layer_ids:
                a = attns[li][0].float().cpu().numpy()  # [heads, T, T]
                hsel = sorted(set(int(i) for i in np.linspace(0, a.shape[0] - 1, min(4, a.shape[0]))))
                heads = []
                for h in hsel:
                    s = TelemetryHooks._head_scores(a[h], toks)
                    heads.append({"matrix": np.round(a[h], 3).tolist(),
                                  "label": TelemetryHooks._label(s),
                                  **{k: round(v, 3) for k, v in s.items()}})
                layers.append({"layer": f"layer.{li}", "idx": li, "heads": heads})
            return {"tokens": toks, "layers": layers}
        except Exception:
            return None

    # ---- generation ------------------------------------------------------ #
    @torch.no_grad()
    def generate_text(self, prompt: str, params: dict, on_token=None) -> str:
        """Generate a continuation. If `on_token` is given, stream it the growing
        text token-by-token (via a background generate thread) for live output."""
        import threading
        from transformers import TextIteratorStreamer

        was = self.model.training
        self.model.eval()
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.block_size)
        input_ids = enc["input_ids"]
        if input_ids.shape[1] == 0:  # empty prompt -> seed with a start token, don't crash
            seed = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id or 0
            input_ids = torch.tensor([[seed]], dtype=torch.long)
        input_ids = input_ids.to(self.device)
        gen_kwargs = dict(
            input_ids=input_ids, do_sample=True,
            temperature=max(0.05, float(params.get("temperature", 0.8))),
            top_k=int(params.get("top_k") or 0) or None,
            top_p=float(params.get("top_p", 1.0)),
            max_new_tokens=int(params.get("max_new_tokens", 96)),
            no_repeat_ngram_size=3,  # kill degenerate loops ("data-driven data-driven…")
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if on_token is None:
            gen = self.model.generate(**gen_kwargs)
            text = self.tokenizer.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True)
        else:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs["streamer"] = streamer
            thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()
            text = ""
            for chunk in streamer:
                text += chunk
                on_token(text)
            thread.join()
        if was:
            self.model.train()
        return text

    def _sample(self):
        prompt = self._seed
        cont = self.generate_text(prompt, {"temperature": 0.8, "top_k": 40, "max_new_tokens": 24})
        return {"prompt": prompt[:40], "cont": cont}

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        x = self._batch(self.val_ids)
        loss = float(self.model(input_ids=x, labels=x).loss.item())
        self.model.train()
        ppl = float(np.exp(min(20, loss)))
        return {"headline": f"fine-tuned {self.base} · perplexity {ppl:.1f}",
                "val_loss": round(loss, 3), "perplexity": round(ppl, 1)}

    # ---- chat template --------------------------------------------------- #
    def format_chat(self, messages, system=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        if getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        lines = ([f"System: {system}"] if system else [])
        lines += [f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in messages]
        return "\n".join(lines + ["Assistant:"])

    chat_stops = ("\nUser:", "\nSystem:", "\nAssistant:")

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
