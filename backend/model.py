"""
model.py — a small, config-driven GPT.

This is the one model definition the whole product uses. A `GPTConfig` (chosen
for the user by a size preset — they never see these numbers) fully specifies it.
The module layout (`blocks.N.attn`, plus `n_heads`/`d_model` attributes) is what
hooks.py keys off for telemetry, and each attention block carries an `ablated`
set so heads can be knocked out live.
"""

from __future__ import annotations
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128     # context length
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    n_experts: int = 0        # 0/1 = dense FFN; >1 = mixture-of-experts
    moe_top_k: int = 2        # experts activated per token
    dropout: float = 0.0      # regularization; nn.Dropout adds no params (ckpt-safe)

    # Human-facing size presets. The user picks a word; we pick the numbers.
    # (block_size is set from the data/task, so it's filled in per project.)
    PRESETS = {
        "tiny": dict(n_layers=3, n_heads=4, d_model=64),
        "small": dict(n_layers=4, n_heads=4, d_model=128),
        "medium": dict(n_layers=6, n_heads=8, d_model=256),
        "large": dict(n_layers=8, n_heads=8, d_model=512),
        "xl": dict(n_layers=12, n_heads=12, d_model=768),
    }

    @classmethod
    def from_preset(cls, preset: str, vocab_size: int, block_size: int,
                    n_experts: int = 0, moe_top_k: int = 2, dropout: float = 0.0) -> "GPTConfig":
        p = cls.PRESETS.get(preset, cls.PRESETS["small"])
        return cls(vocab_size=vocab_size, block_size=block_size,
                   n_experts=int(n_experts), moe_top_k=int(moe_top_k),
                   dropout=float(dropout), **p)


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)   # no params → checkpoint-safe
        self.last_attn_weights = None   # convention hooks.py looks for
        self.layer_idx = None           # set by GPT; addresses ablations
        self.ablated: set[int] = set()  # head indices to knock out (live experiment)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        probs = F.softmax(attn, dim=-1)
        self.last_attn_weights = probs  # [B, heads, T, T] — pre-dropout, for viz
        out = self.attn_drop(probs) @ v
        if self.ablated:
            keep = torch.ones(self.n_heads, device=out.device)
            for h in self.ablated:
                keep[h] = 0.0
            out = out * keep.view(1, self.n_heads, 1, 1)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


def _ffn(d_model: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))


class MoE(nn.Module):
    """A mixture-of-experts FFN: a router scores N expert MLPs per token and
    routes each token to its top-k. Records per-expert load and a load-balancing
    auxiliary loss (Switch-Transformer style) so routing can be watched live."""
    def __init__(self, d_model: int, n_experts: int, top_k: int):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = max(1, min(top_k, n_experts))
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([_ffn(d_model) for _ in range(n_experts)])
        self.last_load = None   # per-expert fraction of tokens routed (telemetry)
        self.last_aux = None    # load-balance auxiliary loss (a tensor)

    def forward(self, x):
        B, T, C = x.shape
        xf = x.reshape(B * T, C)
        probs = F.softmax(self.router(xf), dim=-1)          # [N, n_experts]
        topv, topi = probs.topk(self.top_k, dim=-1)         # [N, k]
        topv = topv / topv.sum(dim=-1, keepdim=True)        # renormalize kept weights
        out = xf.new_zeros(xf.shape)
        for e in range(self.n_experts):
            sel = (topi == e)                               # [N, k]
            idx = sel.any(dim=-1).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            w = (topv * sel).sum(dim=-1)[idx].unsqueeze(-1)  # this expert's weight per token
            out = out.index_add(0, idx, w * self.experts[e](xf[idx]))
        # aux loss: n_experts * Σ (importance_e · load_e) — minimized when balanced
        importance = probs.mean(dim=0)                      # differentiable
        counts = torch.zeros(self.n_experts, device=x.device)
        counts.scatter_add_(0, topi.reshape(-1), torch.ones(topi.numel(), device=x.device))
        load = counts / counts.sum().clamp(min=1)
        self.last_load = load.detach().cpu().tolist()
        self.last_aux = self.n_experts * (importance * load.detach()).sum()
        return out.reshape(B, T, C)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_experts: int = 0, moe_top_k: int = 2, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)   # residual dropout; no params → checkpoint-safe
        self.is_moe = n_experts and n_experts > 1
        if self.is_moe:
            self.moe = MoE(d_model, n_experts, moe_top_k)
        else:
            self.mlp = _ffn(d_model)   # keep the `mlp` name so old checkpoints still load

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop((self.moe if self.is_moe else self.mlp)(self.ln2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.block_size = config.block_size
        self.is_moe = config.n_experts and config.n_experts > 1
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.d_model))
        self.drop = nn.Dropout(getattr(config, "dropout", 0.0))   # embedding dropout
        self.blocks = nn.ModuleList([
            Block(config.d_model, config.n_heads, config.n_experts, config.moe_top_k,
                  getattr(config, "dropout", 0.0))
            for _ in range(config.n_layers)
        ])
        for i, block in enumerate(self.blocks):
            block.attn.layer_idx = i
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)
        self.last_aux_loss = None   # summed MoE load-balance loss for the last forward

    def forward(self, idx):
        h = self.drop(self.tok_emb(idx) + self.pos_emb[:, : idx.size(1)])
        aux = None
        for block in self.blocks:
            h = block(h)
            if block.is_moe:
                aux = block.moe.last_aux if aux is None else aux + block.moe.last_aux
        self.last_aux_loss = aux
        h = self.ln_f(h)
        return self.head(h), h

    def expert_load(self) -> list[list[float]]:
        """Per-MoE-layer list of per-expert token-load fractions (for telemetry)."""
        return [blk.moe.last_load for blk in self.blocks if blk.is_moe and blk.moe.last_load]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int | None = 40, top_p: float = 1.0, eos_token_id: int | None = None,
                 on_token=None):
        """Autoregressive sampling with temperature / top-k / top-p (nucleus).
        `idx` is [1, T] of token ids; returns the newly generated ids, stopping
        early at eos_token_id if given. `on_token`, if given, is called with the
        growing generated ids after each step (for live/streaming output)."""
        was_training = self.training
        self.eval()
        start = idx.size(1)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(1e-6, temperature)
            if top_k:
                k = min(int(top_k), logits.size(-1))
                vals, _ = torch.topk(logits, k)
                logits[logits < vals[:, [-1]]] = float("-inf")
            if top_p and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                logits[0, sorted_idx[0, remove[0]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if on_token is not None:
                on_token(idx[:, start:])
            if eos_token_id is not None and int(nxt) == eos_token_id:
                break
        if was_training:
            self.train()
        return idx[:, start:]

    def ablation_state(self) -> list[list[int]]:
        return [[blk.attn.layer_idx, h] for blk in self.blocks for h in sorted(blk.attn.ablated)]
