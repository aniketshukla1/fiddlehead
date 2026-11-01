"""
dataset.py — turn a pile of documents into training batches.

Given tokenized documents, a block size (context length) and a strategy, this
produces the (x, y) next-token batches the model trains on. Two strategies:

  • "window"      — concatenate documents (with a separator) into one token
                    stream and slide a fixed window over it. Best for free-form
                    text; every position is a training signal.
  • "by_document" — treat each document as its own example, left-padded/truncated
                    to the block. Best when document boundaries matter (records,
                    short prompts) so the model never predicts across them.

The train/val split is by *document*, not by window — so validation text never
appears (even partially) in training, and the val loss means something.
"""

from __future__ import annotations
import random
import torch


class PackedDataset:
    def __init__(self, docs, tokenizer, block_size: int = 128, strategy: str = "window",
                 stride: int | None = None, val_frac: float = 0.1, seed: int = 0):
        self.tok = tokenizer
        self.block_size = block_size
        self.strategy = strategy
        self.stride = stride or max(1, block_size // 2)

        texts = [d.text for d in docs if getattr(d, "text", d) and (getattr(d, "text", d)).strip()]
        if not texts:
            texts = ["hello world.\n"]
        rng = random.Random(seed)
        rng.shuffle(texts)
        n_val = max(1, int(len(texts) * val_frac)) if len(texts) > 1 else 0
        val_texts = texts[:n_val]
        train_texts = texts[n_val:] or texts

        self.train = self._encode_split(train_texts)
        self.val = self._encode_split(val_texts) if val_texts else self.train
        # if a single document had to serve both, carve a tail off for val
        if self.val is self.train and self.strategy == "window" and len(self.train) > block_size * 3:
            cut = int(len(self.train) * (1 - val_frac))
            self.val = self.train[cut:]
            self.train = self.train[:cut]

    def _sep_id(self):
        # a stable separator token so the model can learn document boundaries
        nl = self.tok.encode("\n")
        return nl[0] if nl else 0

    def _encode_split(self, texts):
        if self.strategy == "by_document":
            seqs = []
            for t in texts:
                ids = self.tok.encode(t)[: self.block_size + 1]
                if len(ids) >= 2:
                    seqs.append(ids)
            return seqs  # list of variable-length sequences
        # window: one long stream
        stream = []
        sep = self._sep_id()
        for t in texts:
            stream.extend(self.tok.encode(t))
            stream.append(sep)
        return torch.tensor(stream, dtype=torch.long)

    # ---- batching -------------------------------------------------------- #
    def get_batch(self, batch_size: int, split: str = "train"):
        data = self.train if split == "train" else self.val
        if self.strategy == "by_document":
            return self._batch_docs(data, batch_size)
        return self._batch_window(data, batch_size)

    def _batch_window(self, data, batch_size):
        # clamp the window to what's available so x and y always line up, even
        # when a split has fewer tokens than the block size (small corpora)
        blk = min(self.block_size, max(1, len(data) - 1))
        hi = max(1, len(data) - blk - 1)
        ix = torch.randint(0, hi, (batch_size,))
        x = torch.stack([data[i: i + blk] for i in ix])
        y = torch.stack([data[i + 1: i + 1 + blk] for i in ix])
        return x, y

    def _batch_docs(self, seqs, batch_size):
        if not seqs:
            return torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 2, dtype=torch.long)
        chosen = [random.choice(seqs) for _ in range(batch_size)]
        maxlen = max(len(s) for s in chosen)
        pad = self._sep_id()
        xs, ys = [], []
        for s in chosen:
            s = s + [pad] * (maxlen - len(s))
            xs.append(torch.tensor(s[:-1], dtype=torch.long))
            ys.append(torch.tensor(s[1:], dtype=torch.long))
        return torch.stack(xs), torch.stack(ys)

    # ---- stats for the UI ------------------------------------------------ #
    def stats(self) -> dict:
        def count(d):
            return len(d) if torch.is_tensor(d) else sum(len(s) for s in d)
        return {
            "strategy": self.strategy,
            "block_size": self.block_size,
            "train_tokens": count(self.train),
            "val_tokens": count(self.val),
            "vocab_size": self.tok.vocab_size,
        }
