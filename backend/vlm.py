"""
vlm.py — fine-tune a small vision-language model (image→text) on the user's
image/caption pairs.

Mirrors the FineTuner / AsrTuner interface so the engine drives it identically,
but the modality is image→text: input is an image (ViT pixel values via a
processor + PIL — no torchvision needed), the target is a caption, and quality
is word overlap. Uses a VisionEncoderDecoder captioner (ViT encoder + GPT-2
decoder) with LoRA; CPU-realistic for a handful of images.
"""

from __future__ import annotations
import base64
import io
import json
import os

from artifacts import meta_path
import random

import numpy as np
import torch

SUGGESTED_VLM = ["nlpconnect/vit-gpt2-image-captioning"]


def load_image(path: str | None = None, b64: str | None = None):
    from PIL import Image
    src = io.BytesIO(base64.b64decode(b64)) if b64 is not None else path
    return Image.open(src).convert("RGB")


def _overlap(refs: list[str], hyps: list[str]) -> float:
    """Mean word-overlap (|ref ∩ hyp| / |ref|) — a simple captioning score."""
    accs = []
    for r, h in zip(refs, hyps):
        rw, hw = set(r.lower().split()), set(h.lower().split())
        accs.append(len(rw & hw) / max(1, len(rw)))
    return float(np.mean(accs)) if accs else 0.0


class VlmTuner:
    GEN_EVERY = 12
    EVAL_EVERY = 24

    def __init__(self, spec: dict, device: str = "cpu"):
        from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

        self.base = spec.get("base_model", "nlpconnect/vit-gpt2-image-captioning")
        self.method = spec.get("method", "lora")
        self.device = device
        self.step = 0
        self._last_gen = None

        self.image_processor = ViTImageProcessor.from_pretrained(self.base)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = VisionEncoderDecoderModel.from_pretrained(self.base)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if self.model.config.decoder_start_token_id is None:
            self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id
        dcfg = getattr(self.model.config, "decoder", None)
        self.n_layers = getattr(dcfg, "n_layer", getattr(dcfg, "num_hidden_layers", 0)) if dcfg else 0
        self.n_heads = getattr(dcfg, "n_head", getattr(dcfg, "num_attention_heads", 0)) if dcfg else 0
        self.block_size = 64

        if self.method == "lora":
            from peft import LoraConfig, get_peft_model
            try:
                lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                  target_modules=["c_attn", "query", "value"])
                self.model = get_peft_model(self.model, lcfg)
            except Exception:
                lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules="all-linear")
                self.model = get_peft_model(self.model, lcfg)

        self.model.to(device)
        self.model.train()
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.trainable_params = sum(p.numel() for p in params)
        self.total = sum(p.numel() for p in self.model.parameters())
        self.is_trainable = True
        self.opt = torch.optim.AdamW(params, lr=float(spec.get("lr", 2e-4)))

        self._build_data(spec)

    # ---- data: (image, caption) pairs -> pixel values + label ids -------- #
    def _build_data(self, spec):
        data = spec.get("data") or {}
        pairs = data.get("pairs") or []
        pix, labels, texts, errors = [], [], [], []
        for p in pairs:
            try:
                im = p.get("image") or {}
                img = load_image(path=im.get("path"), b64=im.get("b64"))
                pv = self.image_processor(images=img, return_tensors="pt").pixel_values[0]
                txt = (p.get("text") or "").strip()
                pix.append(pv)
                labels.append(torch.tensor(
                    self.tokenizer(txt, max_length=self.block_size, truncation=True).input_ids, dtype=torch.long))
                texts.append(txt)
            except Exception as e:
                errors.append(str(e))
        if not pix:
            raise RuntimeError("No usable image+caption pairs — add image files with captions.")
        self.pix, self.labels, self.texts = pix, labels, texts
        n = max(1, int(len(pix) * 0.9))
        self.train_idx = list(range(n))
        self.val_idx = list(range(n, len(pix))) or list(range(n))
        self.info = {"documents": len(pix), "images": len(pix), "chars": sum(len(t) for t in texts),
                     "errors": errors[:5], "tokenizer": "hf", "preview": (texts[0][:200] if texts else "")}
        self._seed = texts[0] if texts else ""

    def _batch(self, idxs):
        pv = torch.stack([self.pix[i] for i in idxs]).to(self.device)
        labs = [self.labels[i] for i in idxs]
        maxlen = max(int(l.size(0)) for l in labs)
        padded = torch.full((len(labs), maxlen), -100, dtype=torch.long)
        for j, l in enumerate(labs):
            padded[j, : l.size(0)] = l
        return pv, padded.to(self.device)

    def train_step(self) -> dict:
        bs = min(2, len(self.train_idx))
        idxs = random.sample(self.train_idx, bs)
        pv, labs = self._batch(idxs)
        out = self.model(pixel_values=pv, labels=labs)
        loss = out.loss
        self.opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1e9)
        self.opt.step()

        if self.step % self.GEN_EVERY == 0:
            self._last_gen = self._sample()

        frame = {
            "type": "step", "step": self.step, "loss": float(loss.item()),
            "lr": self.opt.param_groups[0]["lr"], "grad_norm": float(gnorm),
            "grad_norm_per_layer": [], "attention": None, "embeddings": [],
            "induction_step": None, "anomaly": False, "generation": self._last_gen,
            "ablated": [], "frozen": False,
        }
        if self.step % self.EVAL_EVERY == 0:
            frame["eval"] = self.evaluate()
        self.step += 1
        return frame

    # ---- captioning ------------------------------------------------------ #
    @torch.no_grad()
    def caption(self, idx: int | None = None, image: dict | None = None) -> str:
        self.model.eval()
        if image is not None:
            img = load_image(path=image.get("path"), b64=image.get("b64"))
            pv = self.image_processor(images=img, return_tensors="pt").pixel_values.to(self.device)
        else:
            i = self.val_idx[0] if idx is None else idx
            pv = self.pix[i].unsqueeze(0).to(self.device)
        gen = self.model.generate(pixel_values=pv, max_new_tokens=32)
        text = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
        if self.is_trainable:
            self.model.train()
        return text

    def _sample(self):
        try:
            i = self.val_idx[0]
            return {"prompt": f"caption: {self.texts[i][:44]}", "cont": "  ·  got: " + self.caption(idx=i)}
        except Exception:
            return None

    @torch.no_grad()
    def evaluate(self) -> dict:
        refs = [self.texts[i] for i in self.val_idx[:4]]
        hyps = [self.caption(idx=i) for i in self.val_idx[:4]]
        acc = _overlap(refs, hyps)
        return {"headline": f"captions images · {acc * 100:.0f}% word overlap",
                "word_overlap": round(acc, 3), "val_loss": round(1 - acc, 3)}

    # ---- test box: caption a provided image (or a held-out one) ---------- #
    def generate_text(self, prompt: str, params: dict, on_token=None) -> str:
        text = self.caption(image=params.get("image"))
        if on_token:
            on_token(text)
        return text

    def format_chat(self, messages, system=None):
        return ""

    chat_stops = ()

    def meta(self) -> dict:
        return {
            "type": "meta", "model": self.base, "template": "vlm", "modality": "vision",
            "base_model": self.base, "method": self.method, "size": self.method,
            "layers": self.n_layers, "heads": self.n_heads, "device": self.device,
            "block_size": self.block_size, "vocab_size": len(self.tokenizer), "tokenizer": "hf",
            "params_m": round(self.trainable_params / 1e6, 3), "total_params_m": round(self.total / 1e6, 2),
            "data": self.info, "trainable": self.is_trainable,
            "template_desc": (f"loaded {self.base} (image→text)" if not self.is_trainable else
                              f"fine-tuning {self.base} for image captioning ({self.method.upper()})"),
        }

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.image_processor.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
        meta = json.load(open(meta_path(path)))
        self = cls.__new__(cls)
        self.base = meta.get("base_model", "nlpconnect/vit-gpt2-image-captioning")
        self.method = meta.get("method", "lora")
        self.device = device
        self.step = meta.get("step", 0)
        self._last_gen = None
        self.is_trainable = False
        self.info = {"loaded": True}
        self._seed = ""
        self.image_processor = ViTImageProcessor.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if self.method == "lora":
            from peft import PeftModel
            base = VisionEncoderDecoderModel.from_pretrained(self.base)
            self.model = PeftModel.from_pretrained(base, path)
        else:
            self.model = VisionEncoderDecoderModel.from_pretrained(path)
        self.model.to(device).eval()
        self.n_layers = 0
        self.n_heads = 0
        self.block_size = 64
        self.total = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = 0
        self.pix, self.labels, self.texts, self.val_idx = [], [], [], []
        return self
