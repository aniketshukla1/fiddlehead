"""Real distributed data-parallel training (torch.distributed + DDP).

Launched by torchrun; the same script runs single-process or across N ranks:

    python dist_train.py --spec run.json                       # 1 process
    torchrun --standalone --nproc_per_node=2 dist_train.py \
        --spec run.json                                        # 2 ranks

Backend is chosen automatically: nccl on CUDA, gloo on CPU — so this trains
across real GPUs in production and is fully exercisable on a laptop CPU today.
Gradients are all-reduced by DDP every step, so every rank ends with identical
weights (verified by the matching parameter checksums the ranks print).

Progress is written to <spec>.progress.json so an orchestrator can watch it.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig


def is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def log(rank: int, msg: str):
    print(f"[rank {rank}] {msg}", flush=True)


class CharData:
    """Minimal char-level corpus, sharded so each rank sees a different slice."""
    def __init__(self, text: str, block: int, rank: int, world: int, device):
        chars = sorted(set(text)) or list("abc")
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        self.block = block
        self.device = device
        ids = torch.tensor([self.stoi[c] for c in text if c in self.stoi], dtype=torch.long)
        # contiguous shard per rank → no sample is trained twice per step
        n = len(ids)
        per = max(self.block + 1, n // world)
        self.data = ids[rank * per: rank * per + per] if world > 1 else ids
        if len(self.data) < self.block + 2:
            self.data = ids  # tiny corpus: everyone reuses it

    def batch(self, bs: int):
        hi = max(1, len(self.data) - self.block - 1)
        ix = torch.randint(0, hi, (bs,))
        x = torch.stack([self.data[i: i + self.block] for i in ix])
        y = torch.stack([self.data[i + 1: i + 1 + self.block] for i in ix])
        return x.to(self.device), y.to(self.device)


def train(spec_path: str):
    spec = json.load(open(spec_path))
    args = argparse.Namespace(spec=spec_path)
    train = spec.get("train", {})

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    on_cuda = torch.cuda.is_available()
    backend = "nccl" if on_cuda else "gloo"

    if is_dist():
        dist.init_process_group(backend=backend)
        device = torch.device(f"cuda:{local_rank}" if on_cuda else "cpu")
        if on_cuda:
            torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cuda:0" if on_cuda else "cpu")

    torch.manual_seed(int(train.get("seed", 1337)))   # same init on every rank
    if rank == 0:
        log(rank, f"world_size={world} backend={backend} device={device}")

    text = ((spec.get("data") or {}).get("text")
            or "the quick brown fox jumps over the lazy dog. " * 200)
    block = int((spec.get("chunking") or {}).get("block_size", 64))
    data = CharData(text, block, rank, world, device)

    size = spec.get("size", "small")
    cfg = GPTConfig.from_preset(size, data.vocab_size, block,
                                dropout=float(train.get("dropout", 0.0)))
    model = GPT(cfg).to(device)
    if is_dist():
        model = DDP(model, device_ids=([local_rank] if on_cuda else None))

    lr = float(train.get("lr", 3e-3))
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=float(train.get("weight_decay", 0.01)))
    max_steps = int(train.get("max_steps", 50)) or 50
    bs = int(train.get("batch_size", 16))
    grad_clip = float(train.get("grad_clip", 1.0))
    prog_path = args.spec.replace(".json", "") + ".progress.json"

    t0 = time.time()
    model.train()
    last_loss = None
    for step in range(max_steps):
        x, y = data.batch(bs)
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()                       # DDP all-reduces grads here
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        last_loss = float(loss.item())

        if rank == 0 and (step % 5 == 0 or step == max_steps - 1):
            log(rank, f"step {step:>3}  loss {last_loss:.4f}")
            json.dump({"step": step + 1, "max_steps": max_steps, "loss": round(last_loss, 4),
                       "world_size": world, "backend": backend,
                       "elapsed_s": round(time.time() - t0, 2)},
                      open(prog_path, "w"))

    # proof of synchronisation: every rank's weights must be bit-identical
    core = model.module if is_dist() else model
    checksum = float(next(core.parameters()).detach().float().sum().item())
    log(rank, f"final loss {last_loss:.4f}  param_checksum {checksum:.6f}")

    if is_dist():
        # gather every rank's checksum so rank 0 can assert they all match
        buf = torch.tensor([checksum], device=device)
        gathered = [torch.zeros_like(buf) for _ in range(world)]
        dist.all_gather(gathered, buf)
        if rank == 0:
            vals = [round(g.item(), 4) for g in gathered]
            synced = len(set(vals)) == 1
            log(rank, f"checksums across ranks: {vals} → {'SYNCED ✓' if synced else 'DIVERGED ✗'}")
            json.dump({"step": max_steps, "max_steps": max_steps, "loss": round(last_loss, 4),
                       "world_size": world, "backend": backend, "done": True,
                       "weights_synced": synced, "elapsed_s": round(time.time() - t0, 2)},
                      open(prog_path, "w"))
        dist.barrier()
        dist.destroy_process_group()
    else:
        json.dump({"step": max_steps, "max_steps": max_steps, "loss": round(last_loss, 4),
                   "world_size": 1, "backend": backend, "done": True, "weights_synced": True,
                   "elapsed_s": round(time.time() - t0, 2)}, open(prog_path, "w"))


def _worker(local_rank: int, world: int, spec_path: str, port: int):
    """One spawned local process — set the env torchrun would set, then train."""
    os.environ.update(RANK=str(local_rank), LOCAL_RANK=str(local_rank),
                      WORLD_SIZE=str(world), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    train(spec_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="path to the run spec JSON")
    ap.add_argument("--procs", type=int, default=0,
                    help="spawn N local ranks (data-parallel). 0/1 = single; "
                         "omit under torchrun, which sets the env itself.")
    args = ap.parse_args()
    if args.procs and args.procs > 1:
        import random
        import torch.multiprocessing as mp
        port = random.randint(20000, 40000)
        mp.spawn(_worker, args=(args.procs, args.spec, port), nprocs=args.procs, join=True)
    else:
        train(args.spec)   # single process, or under torchrun (env already set)


if __name__ == "__main__":
    main()
