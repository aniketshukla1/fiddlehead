"""
asr.py — fine-tune a Whisper speech-to-text model on the user's audio.

Mirrors the FineTuner interface (meta / train_step / evaluate / save / load) so
the engine drives it the same way, but the modality is audio→text: input is a
log-mel spectrogram (Whisper's numpy feature extractor — no torchaudio needed),
the target is the transcript, and quality is word error rate. Audio is decoded
with soundfile + resampled with librosa to 16 kHz.

Guarded behind transformers/peft/soundfile — the engine only offers the "asr"
template when they import. whisper-tiny/base are CPU-realistic with LoRA.
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

SUGGESTED_ASR = ["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small"]
TARGET_SR = 16000


def load_audio(path: str | None = None, b64: str | None = None) -> np.ndarray:
    """Decode an audio file to a mono float32 array at 16 kHz."""
    import soundfile as sf
    if b64 is not None:
        data, sr = sf.read(io.BytesIO(base64.b64decode(b64)), dtype="float32")
    else:
        data, sr = sf.read(path, dtype="float32")
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)  # downmix to mono
    if sr != TARGET_SR:
        import librosa
        data = librosa.resample(np.asarray(data, dtype=np.float32), orig_sr=sr, target_sr=TARGET_SR)
    return np.asarray(data, dtype=np.float32)


def _wer(refs: list[str], hyps: list[str]) -> float:
    """Mean word error rate (word-level Levenshtein / reference length)."""
    def lev(a, b):
        dp = list(range(len(b) + 1))
        for i in range(1, len(a) + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, len(b) + 1):
                cur = dp[j]
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
                prev = cur
        return dp[-1]
    err = tot = 0
    for r, h in zip(refs, hyps):
        rw, hw = r.lower().split(), h.lower().split()
        err += lev(rw, hw)
        tot += max(1, len(rw))
    return err / max(1, tot)


class AsrTuner:
    GEN_EVERY = 12
    EVAL_EVERY = 24

    def __init__(self, spec: dict, device: str = "cpu"):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.base = spec.get("base_model", "openai/whisper-tiny")
        self.method = spec.get("method", "lora")
        self.device = device
        self.step = 0
        self._last_gen = None
        self.language = spec.get("language", "english")

        self.processor = WhisperProcessor.from_pretrained(self.base)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.base)
        cfg = self.model.config
        self.n_layers = getattr(cfg, "encoder_layers", 0)
        self.n_heads = getattr(cfg, "encoder_attention_heads", 0)
        # let the model choose decoder ids from language/task at generate time
        try:
            self.model.generation_config.language = self.language
            self.model.generation_config.task = "transcribe"
            self.model.config.forced_decoder_ids = None
        except Exception:
            pass

        if self.method == "lora":
            from peft import LoraConfig, get_peft_model
            lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                              target_modules=["q_proj", "v_proj"])
            self.model = get_peft_model(self.model, lcfg)

        self.model.to(device)
        self.model.train()
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.trainable_params = sum(p.numel() for p in params)
        self.total = sum(p.numel() for p in self.model.parameters())
        self.is_trainable = True
        self.block_size = 448  # whisper max target length (unused for input)
        self.opt = torch.optim.AdamW(params, lr=float(spec.get("lr", 1e-4)))

        self._build_data(spec)

    # ---- data: (audio, transcript) pairs -> features + label ids --------- #
    def _build_data(self, spec):
        data = spec.get("data") or {}
        pairs = data.get("pairs") or []
        feats, labels, texts, errors = [], [], [], []
        fe = self.processor.feature_extractor
        for p in pairs:
            try:
                a = p.get("audio") or {}
                wav = load_audio(path=a.get("path"), b64=a.get("b64"))
                inp = fe(wav, sampling_rate=TARGET_SR, return_tensors="pt").input_features[0]
                txt = (p.get("text") or "").strip()
                feats.append(inp)
                labels.append(torch.tensor(self.processor.tokenizer(txt).input_ids, dtype=torch.long))
                texts.append(txt)
            except Exception as e:  # a bad clip shouldn't kill the run
                errors.append(str(e))
        if not feats:
            raise RuntimeError("No usable audio+transcript pairs — add .wav/.flac clips with transcripts.")
        self.feats, self.labels, self.texts = feats, labels, texts
        n = max(1, int(len(feats) * 0.9))
        self.train_idx = list(range(n))
        self.val_idx = list(range(n, len(feats))) or list(range(n))
        self.info = {"documents": len(feats), "clips": len(feats),
                     "chars": sum(len(t) for t in texts), "errors": errors[:5],
                     "tokenizer": "whisper", "preview": (texts[0][:200] if texts else "")}
        self._seed = texts[0] if texts else ""

    def _batch(self, idxs):
        feats = torch.stack([self.feats[i] for i in idxs]).to(self.device)
        labs = [self.labels[i] for i in idxs]
        maxlen = max(int(l.size(0)) for l in labs)
        padded = torch.full((len(labs), maxlen), -100, dtype=torch.long)  # -100 = ignore in loss
        for j, l in enumerate(labs):
            padded[j, : l.size(0)] = l
        return feats, padded.to(self.device)

    # ---- one step -> a telemetry frame ----------------------------------- #
    def train_step(self) -> dict:
        bs = min(2, len(self.train_idx))
        idxs = random.sample(self.train_idx, bs)
        feats, labs = self._batch(idxs)
        out = self.model(input_features=feats, labels=labs)
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

    # ---- transcription --------------------------------------------------- #
    @torch.no_grad()
    def transcribe(self, idx: int | None = None, audio: dict | None = None) -> str:
        self.model.eval()
        if audio is not None:
            wav = load_audio(path=audio.get("path"), b64=audio.get("b64"))
            feats = self.processor.feature_extractor(wav, sampling_rate=TARGET_SR,
                                                     return_tensors="pt").input_features.to(self.device)
        else:
            i = self.val_idx[0] if idx is None else idx
            feats = self.feats[i].unsqueeze(0).to(self.device)
        gen = self.model.generate(input_features=feats, max_new_tokens=128)
        text = self.processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        if self.is_trainable:
            self.model.train()
        return text

    def _sample(self):
        try:
            i = self.val_idx[0]
            return {"prompt": f"ref: {self.texts[i][:48]}", "cont": "  ·  hyp: " + self.transcribe(idx=i)}
        except Exception:
            return None

    @torch.no_grad()
    def evaluate(self) -> dict:
        refs, hyps = [], []
        for i in self.val_idx[:4]:
            refs.append(self.texts[i])
            hyps.append(self.transcribe(idx=i))
        wer = _wer(refs, hyps)
        return {"headline": f"transcribes audio · {(1 - wer) * 100:.0f}% word accuracy (WER {wer * 100:.0f}%)",
                "wer": round(wer, 3), "word_accuracy": round(1 - wer, 3), "val_loss": round(wer, 3)}

    # ---- test box: transcribe provided audio (or a held-out clip) --------- #
    def generate_text(self, prompt: str, params: dict, on_token=None) -> str:
        text = self.transcribe(audio=params.get("audio"))
        if on_token:
            on_token(text)
        return text

    def format_chat(self, messages, system=None):
        return ""

    chat_stops = ()

    def meta(self) -> dict:
        return {
            "type": "meta", "model": self.base, "template": "asr", "modality": "speech",
            "base_model": self.base, "method": self.method, "size": self.method,
            "layers": self.n_layers, "heads": self.n_heads, "device": self.device,
            "block_size": self.block_size, "vocab_size": len(self.processor.tokenizer),
            "tokenizer": "whisper", "params_m": round(self.trainable_params / 1e6, 3),
            "total_params_m": round(self.total / 1e6, 2), "data": self.info,
            "trainable": self.is_trainable,
            "template_desc": (f"loaded {self.base} (speech-to-text)" if not self.is_trainable else
                              f"fine-tuning {self.base} for speech-to-text ({self.method.upper()})"),
        }

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)

    @classmethod
    def load(cls, path: str, device: str = "cpu"):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        meta = json.load(open(meta_path(path)))
        self = cls.__new__(cls)
        self.base = meta.get("base_model", "openai/whisper-tiny")
        self.method = meta.get("method", "lora")
        self.device = device
        self.step = meta.get("step", 0)
        self._last_gen = None
        self.is_trainable = False
        self.info = {"loaded": True}
        self._seed = ""
        self.processor = WhisperProcessor.from_pretrained(path)
        if self.method == "lora":
            from peft import PeftModel
            base = WhisperForConditionalGeneration.from_pretrained(self.base)
            self.model = PeftModel.from_pretrained(base, path)
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(path)
        self.model.to(device).eval()
        cfg = self.model.config
        self.n_layers = getattr(cfg, "encoder_layers", 0)
        self.n_heads = getattr(cfg, "encoder_attention_heads", 0)
        self.block_size = 448
        self.total = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = 0
        self.feats, self.labels, self.texts, self.val_idx = [], [], [], []
        return self
