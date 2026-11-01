"""
data.py — templates the user picks from, behind one `Task` interface.

A Task owns everything data-related so the engine stays generic: the tokenizer,
how to draw a batch, how to turn a prompt into tokens and back, and — for a
*usable* product — an `evaluate()` that reports quality in plain language.

The "text" template runs the full data pipeline: gather (sources.py) -> tokenize
(tokenizer.py) -> chunk (dataset.py). The structured templates (copy, sort)
generate their own synthetic data.
"""

from __future__ import annotations
import random
import torch

import sources
from tokenizer import CharTokenizer, build_tokenizer
from dataset import PackedDataset


class Task:
    kind = "base"
    trainable = True
    tok: CharTokenizer
    block_size: int
    info: dict = {}

    def get_batch(self, batch_size: int, split: str = "train"):
        raise NotImplementedError

    def default_prompt(self) -> str:
        return ""

    def describe(self) -> dict:
        return {"kind": self.kind, "vocab_size": self.tok.vocab_size, "block_size": self.block_size}

    def evaluate(self, model) -> dict:
        return {}

    def format_chat(self, messages: list[dict], system: str | None = None) -> str:
        """Turn a conversation into a prompt. Base (non-instruct) models have no
        chat template, so we use a simple, learnable transcript convention; the
        fine-tune task overrides this to use the real model's chat template."""
        lines = []
        if system:
            lines.append(f"System: {system}")
        for m in messages:
            role = "User" if m.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {m.get('content', '')}")
        lines.append("Assistant:")
        return "\n".join(lines)

    # role markers to cut a chat continuation at (so we return only one reply)
    chat_stops = ("\nUser:", "\nSystem:", "\nAssistant:")


# --------------------------------------------------------------------------- #
# text — a real language model on the user's gathered corpus
# --------------------------------------------------------------------------- #
class CorpusTask(Task):
    kind = "text"

    def __init__(self, tok, dataset: PackedDataset, seed_text: str, info: dict):
        self.tok = tok
        self.ds = dataset
        self.block_size = dataset.block_size
        self._seed = seed_text
        self.info = info

    def get_batch(self, batch_size: int, split: str = "train"):
        return self.ds.get_batch(batch_size, split)

    def default_prompt(self) -> str:
        return self._seed

    def describe(self) -> dict:
        return {"kind": self.kind, **self.ds.stats(), **self.info}

    @torch.no_grad()
    def evaluate(self, model) -> dict:
        import torch.nn.functional as F
        x, y = self.ds.get_batch(64, "val")
        logits, _ = model(x)
        acc = (logits.argmax(-1) == y).float().mean().item()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        return {
            "headline": f"continues your text · {acc*100:.0f}% next-token accuracy",
            "next_char_accuracy": round(acc, 3),
            "val_loss": round(loss, 3),
        }


def _build_text_task(spec: dict) -> CorpusTask:
    data_spec = spec.get("data") or {"source": "paste", "text": spec.get("text", "")}
    docs = sources.gather(data_spec)
    info = sources.corpus_stats(docs)

    joined = "\n\n".join(d.text for d in docs if d.text) or "hello world.\n"
    tok = build_tokenizer(spec.get("tokenizer"), joined)

    ch = spec.get("chunking") or {}
    block_size = int(ch.get("block_size", spec.get("block_size", 128)))
    ds = PackedDataset(docs, tok, block_size=block_size,
                       strategy=ch.get("strategy", "window"),
                       stride=ch.get("stride"), val_frac=float(ch.get("val_frac", 0.1)))

    seed = ""
    for d in docs:
        if d.text.strip():
            seed = d.text.strip()[: max(1, block_size // 4)]
            break
    return CorpusTask(tok, ds, seed, {**info, **ds.stats(), "tokenizer": tok.kind})


# --------------------------------------------------------------------------- #
# copy — repeat a random block (the induction task)
# --------------------------------------------------------------------------- #
class CopyTask(Task):
    kind = "copy"

    def __init__(self, core_len: int = 8, alphabet: str = "abcdefghijklmnopqrstuvwxyz"):
        self.core_len = core_len
        self.tok = CharTokenizer(sorted(set(alphabet)))
        self.block_size = 2 * core_len - 1
        self.info = {"kind": "copy", "synthetic": True}

    def get_batch(self, batch_size: int, split: str = "train"):
        core = torch.randint(0, self.tok.vocab_size, (batch_size, self.core_len))
        seq = torch.cat([core, core], dim=1)
        return seq[:, :-1], seq[:, 1:]

    def _random_core(self) -> str:
        return "".join(random.choice(self.tok.itos) for _ in range(self.core_len))

    def default_prompt(self) -> str:
        return self._random_core()

    @torch.no_grad()
    def evaluate(self, model) -> dict:
        matches, total = 0, 0
        for _ in range(16):
            core = self._random_core()
            ids = torch.tensor([self.tok.encode(core)], dtype=torch.long)
            out = model.generate(ids, self.core_len, temperature=0.1, top_k=1)
            matches += sum(a == b for a, b in zip(core, self.tok.decode(out[0])))
            total += self.core_len
        acc = matches / total
        return {"headline": f"copies patterns from context · {acc*100:.0f}% accuracy", "copy_accuracy": round(acc, 3)}


# --------------------------------------------------------------------------- #
# sort — output the sorted version of a random sequence
# --------------------------------------------------------------------------- #
class SortTask(Task):
    kind = "sort"

    def __init__(self, seq_len: int = 6, n_symbols: int = 10):
        self.seq_len = seq_len
        self.symbols = [str(i) for i in range(n_symbols)]
        self.tok = CharTokenizer(self.symbols + [">"])
        self.sep = self.tok.stoi[">"]
        self.block_size = 2 * seq_len
        self.info = {"kind": "sort", "synthetic": True}

    def _make(self, batch_size: int):
        rows = []
        for _ in range(batch_size):
            seq = [random.randrange(len(self.symbols)) for _ in range(self.seq_len)]
            rows.append(seq + [self.sep] + sorted(seq))
        return torch.tensor(rows, dtype=torch.long)

    def get_batch(self, batch_size: int, split: str = "train"):
        full = self._make(batch_size)
        return full[:, :-1], full[:, 1:]

    def default_prompt(self) -> str:
        seq = [random.randrange(len(self.symbols)) for _ in range(self.seq_len)]
        return "".join(self.symbols[i] for i in seq) + ">"

    @torch.no_grad()
    def evaluate(self, model) -> dict:
        correct, total = 0, 0
        for _ in range(16):
            seq = [random.randrange(len(self.symbols)) for _ in range(self.seq_len)]
            ids = torch.tensor([[i for i in seq] + [self.sep]], dtype=torch.long)
            out = model.generate(ids, self.seq_len, temperature=0.1, top_k=1)
            got = [int(self.tok.itos[int(i)]) if self.tok.itos[int(i)] != ">" else -1 for i in out[0]]
            correct += sum(a == b for a, b in zip(got, sorted(seq)))
            total += self.seq_len
        acc = correct / total
        return {"headline": f"sorts sequences · {acc*100:.0f}% of positions correct", "sort_accuracy": round(acc, 3)}


TEMPLATES = {
    "text": "Autocomplete on your text — paste, point at a file/URL, or scrape the web; it learns to continue it.",
    "copy": "Copy patterns — learns to repeat sequences from context (forms an induction circuit).",
    "sort": "Sort sequences — learns to output a sorted version of its input.",
}


def build_task(spec: dict) -> Task:
    template = spec.get("template", "text")
    if template == "text":
        return _build_text_task(spec)
    if template == "copy":
        return CopyTask(core_len=int(spec.get("core_len", 8)))
    if template == "sort":
        return SortTask(seq_len=int(spec.get("seq_len", 6)))
    raise ValueError(f"unknown template: {template}")
