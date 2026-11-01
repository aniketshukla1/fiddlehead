"""
train_demo.py — a tiny GPT-style transformer trained on a task that *forces*
induction circuits to form, wired to hooks.py + server.py so the frontend can
watch attention heads specialize in real time.

The data is the classic induction setup: each sequence is a block of random
tokens immediately repeated. The first copy is unpredictable, but the second
copy can only be predicted by looking back at what followed the same token
earlier in the context — i.e. by learning the induction pattern
[A][B]…[A] -> [B]. When that circuit clicks into place, loss on the repeated
half collapses (the famous "induction bump") and the frontend lights up the
head that did it.

Run: python train_demo.py
Then open the frontend (npm run dev) — it connects automatically.

Swap `TinyGPT` for your own model: just make sure your attention module sets
`self.last_attn_weights = attn_probs` before returning, and the hook
auto-discovery in hooks.py will find it.
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from hooks import TelemetryHooks
from server import TelemetryServer

torch.manual_seed(0)

VOCAB = "abcdefghijklmnopqrstuvwxyz"
ITOS = list(VOCAB)
CORE_LEN = 8                     # length of the random block that gets repeated
SEQ_LEN = 2 * CORE_LEN - 1       # model sees the block + its repeat, shifted for LM


def sample_batch(batch_size=16):
    """A random block of tokens, immediately repeated. Predicting the second
    copy is only possible via induction — copy from the first occurrence."""
    core = torch.randint(0, len(VOCAB), (batch_size, CORE_LEN))
    seq = torch.cat([core, core], dim=1)          # [B, 2*CORE_LEN]
    x = seq[:, :-1]                                # [B, SEQ_LEN]
    y = seq[:, 1:]                                 # [B, SEQ_LEN] (next-token targets)
    return x, y


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.last_attn_weights = None  # <- convention hooks.py looks for
        self.layer_idx = None          # set by TinyGPT; used to address ablations
        self.ablated: set[int] = set()  # head indices to knock out (live experiment)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        probs = F.softmax(attn, dim=-1)
        self.last_attn_weights = probs  # [B, heads, T, T]
        out = probs @ v
        if self.ablated:
            # zero the ablated heads' contributions before the output projection
            keep = torch.ones(self.n_heads, device=out.device)
            for h in self.ablated:
                keep[h] = 0.0
            out = out * keep.view(1, self.n_heads, 1, 1)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=4, seq_len=SEQ_LEN):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        for i, block in enumerate(self.blocks):
            block.attn.layer_idx = i
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.tok_emb(x) + self.pos_emb[:, : x.size(1)]
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.head(h), h


def apply_control(model: TinyGPT, msg: dict):
    """Handle a control message from the frontend. Currently: toggle the
    ablation of a single (layer, head) so the operator can knock out a circuit
    and watch the damage in the loss + generation streams."""
    if msg.get("type") != "ablate":
        return
    layer, head = msg.get("layer"), msg.get("head")
    if layer is None or head is None or not (0 <= layer < len(model.blocks)):
        return
    ablated = model.blocks[layer].attn.ablated
    if msg.get("on", True):
        ablated.add(head)
    else:
        ablated.discard(head)


def ablated_state(model: TinyGPT) -> list[list[int]]:
    return [[blk.attn.layer_idx, h] for blk in model.blocks for h in sorted(blk.attn.ablated)]


@torch.no_grad()
def sample_generation(model: TinyGPT) -> dict:
    """Prompt the model with a fresh random block and let it continue. A model
    that has learned induction will reproduce the block verbatim — so `match`
    is a direct, human-readable readout of copy accuracy."""
    was_training = model.training
    model.eval()
    core = torch.randint(0, len(VOCAB), (1, CORE_LEN))
    ctx = core.clone()
    for _ in range(CORE_LEN):
        logits, _ = model(ctx[:, -SEQ_LEN:])
        nxt = logits[0, -1].argmax().view(1, 1)
        ctx = torch.cat([ctx, nxt], dim=1)
    if was_training:
        model.train()

    prompt = "".join(ITOS[i] for i in core[0].tolist())
    cont = "".join(ITOS[i] for i in ctx[0, CORE_LEN:].tolist())
    match = sum(a == b for a, b in zip(prompt, cont))
    return {"prompt": prompt, "cont": cont, "match": match, "of": CORE_LEN}


def main():
    telemetry = TelemetryServer()
    telemetry.start()
    time.sleep(0.3)  # let the socket bind before training starts

    # 4 layers × 2 heads: enough depth for a real induction circuit to form,
    # but few enough heads that the circuit is concentrated — so ablating an
    # induction head visibly breaks copying (the causal demo).
    model = TinyGPT(vocab_size=len(VOCAB), n_layers=4, n_heads=2)
    hooks = TelemetryHooks(model)
    telemetry.publish(hooks.meta(model_name="tiny-gpt-induction", dataset="repeated-random"))

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    step = 0
    generation = sample_generation(model)
    while True:
        for msg in telemetry.poll_control():
            apply_control(model, msg)

        # Freeze the weights whenever a head is ablated: an ablation study probes
        # a *fixed* model. If we kept training, the model would just re-route the
        # induction circuit through other heads within a few steps (it heals the
        # lesion), hiding the causal effect. Frozen, the damage persists until the
        # head is restored — so copying visibly collapses and comes back.
        frozen = any(blk.attn.ablated for blk in model.blocks)

        x, y = sample_batch()
        logits, hidden = model(x)
        hooks._hidden_cache = hidden.detach()  # latest hidden state for the embedding projection
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        opt.zero_grad()
        loss.backward()
        if not frozen:
            opt.step()

        if step % 5 == 0:
            generation = sample_generation(model)

        tokens = [ITOS[i] for i in x[0].tolist()]
        frame = hooks.collect(step=step, loss=loss.item(), lr=opt.param_groups[0]["lr"], tokens=tokens)
        frame["generation"] = generation
        frame["ablated"] = ablated_state(model)
        frame["frozen"] = frozen
        telemetry.publish(frame)

        if step % 20 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  copy {generation['match']}/{generation['of']}")

        step += 1
        time.sleep(0.05)  # slow down so the UI can actually watch it


if __name__ == "__main__":
    main()
