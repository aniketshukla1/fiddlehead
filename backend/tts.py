"""
tts.py — fine-tune a SpeechT5 text-to-speech model on the user's (text, audio).

The hardest modality: text -> mel spectrogram (SpeechT5) -> waveform (HiFi-GAN
vocoder). Mirrors the FineTuner/AsrTuner/VlmTuner interface so the engine drives
it identically, but the *output* is audio. Single-speaker fine-tune: a fixed
x-vector speaker embedding conditions the voice, so the user's clips teach it a
voice/style. Reuses the ASR 16 kHz mono loader. CPU-realistic to fine-tune on a
handful of short clips; synthesis is slower, so it's on-demand only.

NOTE: written to the canonical SpeechT5 recipe but pending an end-to-end smoke
test (needs the ~600MB speecht5_tts + hifigan download + a real run).
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

from asr import load_audio  # 16 kHz mono float32 loader (soundfile + librosa)

SUGGESTED_TTS = ["microsoft/speecht5_tts"]
VOCODER = "microsoft/speecht5_hifigan"


class TtsTuner:
    GEN_EVERY = 10 ** 9   # synthesis is slow — only on the explicit "speak" control
    EVAL_EVERY = 24

    def __init__(self, spec: dict, device: str = "cpu"):
        from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan

        self.base = spec.get("base_model", "microsoft/speecht5_tts")
        self.method = spec.get("method", "lora")
        self.device = device
        self.step = 0
        self._last_gen = None

        self.processor = SpeechT5Processor.from_pretrained(self.base)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(self.base)
        # the guided-attention aux loss needs padded input/output masks; skip it for
        # a simple single-example fine-tune (the mel-spectrogram loss still trains)
        self.model.config.use_guided_attention_loss = False
        self.vocoder = SpeechT5HifiGan.from_pretrained(VOCODER)
        self.reduction = int(getattr(self.model.config, "reduction_factor", 2) or 2)
        self.n_layers = getattr(self.model.config, "decoder_layers", 0)
        self.n_heads = getattr(self.model.config, "decoder_attention_heads", 0)
        self.block_size = 600

        # a fixed, consistent speaker embedding (single-speaker fine-tune)
        g = torch.Generator().manual_seed(0)
        spk = torch.randn(1, 512, generator=g)
        self.speaker = (spk / spk.norm()).to(device)

        if self.method == "lora":
            from peft import LoraConfig, get_peft_model
            try:
                lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
                self.model = get_peft_model(self.model, lcfg)
            except Exception:
                lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules="all-linear")
                self.model = get_peft_model(self.model, lcfg)

        self.model.to(device)
        self.model.train()
        self.vocoder.to(device).eval()
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.trainable_params = sum(p.numel() for p in params)
        self.total = sum(p.numel() for p in self.model.parameters())
        self.is_trainable = True
        self.opt = torch.optim.AdamW(params, lr=float(spec.get("lr", 1e-4)))

        self._build_data(spec)

    # ---- data: (text, audio) -> input_ids + mel labels ------------------- #
    def _build_data(self, spec):
        data = spec.get("data") or {}
        pairs = data.get("pairs") or []
        examples, texts, errors = [], [], []
        for p in pairs:
            try:
                a = p.get("audio") or {}
                wav = load_audio(path=a.get("path"), b64=a.get("b64"))
                txt = (p.get("text") or "").strip()
                proc = self.processor(text=txt, audio_target=wav, sampling_rate=16000, return_tensors="pt")
                labels = proc["labels"][0]
                T = (labels.size(0) // self.reduction) * self.reduction   # multiple of reduction factor
                if T < self.reduction:
                    continue
                examples.append({"input_ids": proc["input_ids"][0], "labels": labels[:T]})
                texts.append(txt)
            except Exception as e:
                errors.append(str(e))
        if not examples:
            raise RuntimeError("No usable text+audio pairs — add short .wav/.flac clips with their text.")
        self.examples, self.texts = examples, texts
        n = max(1, int(len(examples) * 0.9))
        self.train_idx = list(range(n))
        self.val_idx = list(range(n, len(examples))) or list(range(n))
        self.info = {"documents": len(examples), "clips": len(examples), "chars": sum(len(t) for t in texts),
                     "errors": errors[:5], "tokenizer": "speecht5", "preview": (texts[0][:200] if texts else "")}
        self._seed = texts[0] if texts else "hello world"

    def _forward(self, i):
        ex = self.examples[i]
        ids = ex["input_ids"].unsqueeze(0).to(self.device)
        return self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                          labels=ex["labels"].unsqueeze(0).to(self.device),
                          speaker_embeddings=self.speaker)

    def train_step(self) -> dict:
        i = random.choice(self.train_idx)
        loss = self._forward(i).loss
        self.opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1e9)
        self.opt.step()
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

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        losses = [float(self._forward(i).loss.item()) for i in self.val_idx[:4]]
        if self.is_trainable:
            self.model.train()
        vl = round(float(np.mean(losses)), 3) if losses else None
        return {"headline": f"synthesizes speech from text · mel loss {vl}", "val_loss": vl}

    # ---- synthesis: text -> base64 WAV ----------------------------------- #
    @torch.no_grad()
    def synthesize(self, text: str | None = None) -> str:
        import soundfile as sf
        self.model.eval()
        inp = self.processor(text=(text or self._seed), return_tensors="pt").input_ids.to(self.device)
        speech = self.model.generate_speech(inp, self.speaker, vocoder=self.vocoder)  # [samples] @16kHz
        wav = speech.detach().cpu().numpy().astype(np.float32)
        if self.is_trainable:
            self.model.train()
        buf = io.BytesIO()
        sf.write(buf, wav, 16000, format="WAV")
        return base64.b64encode(buf.getvalue()).decode()

    def generate_text(self, prompt: str, params: dict, on_token=None) -> str:
        return ""  # audio out isn't text; the studio uses the "speak" control

    def format_chat(self, messages, system=None):
        return ""

    chat_stops = ()

    def meta(self) -> dict:
        return {
            "type": "meta", "model": self.base, "template": "tts", "modality": "speech-out",
            "base_model": self.base, "method": self.method, "size": self.method,
            "layers": self.n_layers, "heads": self.n_heads, "device": self.device,
            "block_size": self.block_size, "vocab_size": len(self.processor.tokenizer),
            "tokenizer": "speecht5", "params_m": round(self.trainable_params / 1e6, 3),
            "total_params_m": round(self.total / 1e6, 2), "data": self.info,
            "trainable": self.is_trainable,
            "template_desc": (f"loaded {self.base} (text-to-speech)" if not self.is_trainable else
                              f"fine-tuning {self.base} for text-to-speech ({self.method.upper()})"),
        }

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan
        meta = json.load(open(meta_path(path)))
        self = cls.__new__(cls)
        self.base = meta.get("base_model", "microsoft/speecht5_tts")
        self.method = meta.get("method", "lora")
        self.device = device
        self.step = meta.get("step", 0)
        self._last_gen = None
        self.is_trainable = False
        self.info = {"loaded": True}
        self._seed = "hello world"
        self.processor = SpeechT5Processor.from_pretrained(path)
        self.vocoder = SpeechT5HifiGan.from_pretrained(VOCODER).to(device).eval()
        self.reduction = 2
        if self.method == "lora":
            from peft import PeftModel
            base = SpeechT5ForTextToSpeech.from_pretrained(self.base)
            self.model = PeftModel.from_pretrained(base, path)
        else:
            self.model = SpeechT5ForTextToSpeech.from_pretrained(path)
        self.model.to(device).eval()
        g = torch.Generator().manual_seed(0)
        spk = torch.randn(1, 512, generator=g)
        self.speaker = (spk / spk.norm()).to(device)
        self.n_layers = getattr(self.model.config, "decoder_layers", 0)
        self.n_heads = getattr(self.model.config, "decoder_attention_heads", 0)
        self.block_size = 600
        self.total = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = 0
        self.examples, self.texts, self.val_idx = [], [], []
        return self
