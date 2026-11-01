"""
tokenizer.py — two tokenizers behind one interface.

  • CharTokenizer     — one id per character. Zero-config, instant, tiny vocab.
                        Great for small/quirky corpora and the toy templates.
  • ByteBPETokenizer  — byte-level Byte-Pair Encoding, trained on your corpus.
                        Learns subword merges so sequences are far shorter, which
                        lets a small model "see" more context per step — the main
                        lever for making a genuinely usable model on real text.

Both expose encode/decode/vocab_size and a JSON-serializable state() so the
tokenizer travels with the model on save/load. BPE is implemented from scratch
(minbpe-style) so there are no heavy dependencies.
"""

from __future__ import annotations


class CharTokenizer:
    kind = "char"

    def __init__(self, vocab: list[str]):
        self.itos = list(vocab)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def train(cls, text: str, **_):
        return cls(sorted(set(text)) or list(" "))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids if 0 <= int(i) < len(self.itos))

    def state(self) -> dict:
        return {"kind": "char", "vocab": self.itos}

    @classmethod
    def from_state(cls, s: dict):
        return cls(s["vocab"])


def _get_stats(ids: list[int]) -> dict:
    counts = {}
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def _merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class ByteBPETokenizer:
    kind = "bpe"
    # cap the text used to *learn* merges so preparing stays responsive; the
    # learned merges still apply to the whole corpus afterwards.
    TRAIN_CAP = 400_000

    def __init__(self, merges: dict | None = None, vocab: dict | None = None):
        self.merges = merges or {}
        self.vocab = vocab or {i: bytes([i]) for i in range(256)}

    @classmethod
    def train(cls, text: str, vocab_size: int = 512, **_):
        vocab_size = max(256, int(vocab_size))
        data = text[: cls.TRAIN_CAP].encode("utf-8", errors="ignore")
        ids = list(data)
        merges, vocab = {}, {i: bytes([i]) for i in range(256)}
        for i in range(vocab_size - 256):
            stats = _get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            if stats[pair] < 2:
                break  # nothing repeats — done
            idx = 256 + i
            ids = _merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        return cls(merges, vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, s: str) -> list[int]:
        ids = list(s.encode("utf-8", errors="ignore"))
        while len(ids) >= 2:
            stats = _get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids) -> str:
        parts = [self.vocab[int(i)] for i in ids if int(i) in self.vocab]
        return b"".join(parts).decode("utf-8", errors="replace")

    def state(self) -> dict:
        return {"kind": "bpe", "merges": [[a, b, idx] for (a, b), idx in self.merges.items()]}

    @classmethod
    def from_state(cls, s: dict):
        merges = {(a, b): idx for a, b, idx in s["merges"]}
        vocab = {i: bytes([i]) for i in range(256)}
        # rebuild merged tokens in id order so each depends only on earlier ones
        for (a, b), idx in sorted(merges.items(), key=lambda kv: kv[1]):
            vocab[idx] = vocab[a] + vocab[b]
        return cls(merges, vocab)


def _chunks(text: str, n: int = 50_000):
    for i in range(0, len(text), n):
        yield text[i: i + n]


class FastBPETokenizer:
    """Byte-level BPE backed by HuggingFace `tokenizers` (Rust) — trains and
    encodes orders of magnitude faster than the pure-Python path, so it scales
    to large corpora. Same interface; used automatically when the library is
    available. State is the tokenizer's own JSON, so it round-trips exactly."""
    kind = "bpe"

    def __init__(self, tk):
        self._tk = tk

    @classmethod
    def train(cls, text: str, vocab_size: int = 512, **_):
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        tk = Tokenizer(models.BPE(unk_token=None))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=max(256, int(vocab_size)), show_progress=False,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=[])
        tk.train_from_iterator(_chunks(text), trainer)
        return cls(tk)

    @property
    def vocab_size(self) -> int:
        return self._tk.get_vocab_size()

    def encode(self, s: str) -> list[int]:
        return self._tk.encode(s).ids

    def decode(self, ids) -> str:
        return self._tk.decode([int(i) for i in ids])

    def state(self) -> dict:
        return {"kind": "bpe_fast", "json": self._tk.to_str()}

    @classmethod
    def from_state(cls, s: dict):
        from tokenizers import Tokenizer
        return cls(Tokenizer.from_str(s["json"]))


def build_tokenizer(spec: dict, text: str):
    """spec = {kind: 'char'|'bpe', vocab_size?} -> a trained tokenizer. For BPE
    we prefer the fast Rust path and transparently fall back to pure Python."""
    spec = spec or {}
    kind = spec.get("kind", "char")
    if kind == "bpe":
        vs = int(spec.get("vocab_size", 512))
        try:
            return FastBPETokenizer.train(text, vocab_size=vs)
        except ImportError:
            return ByteBPETokenizer.train(text, vocab_size=vs)
    return CharTokenizer.train(text)


def load_tokenizer(state: dict):
    kind = state.get("kind")
    if kind == "bpe_fast":
        return FastBPETokenizer.from_state(state)
    if kind == "bpe":
        return ByteBPETokenizer.from_state(state)
    return CharTokenizer.from_state(state)
