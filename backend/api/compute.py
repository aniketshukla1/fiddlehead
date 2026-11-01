"""Compute planning for training runs — how to split a model across GPUs.

The product's paid tier trains models too big for one machine. Given a model's
size and an available GPU pool, this picks the parallelism strategy and the
launch command. It's the real decision logic; the actual GPU execution is done
by a worker (torchrun/FSDP single-node, Ray/SkyPilot/K8s multi-node) that this
plan configures.

Ladder of strategies (cheapest that fits, in order):
  single   — model + optimizer fit on one GPU
  ddp      — fits on one GPU, replicated across N for throughput (all-reduce)
  fsdp     — too big for one GPU; shard params/grads/optimizer across the pool
  tp+pp    — too big even sharded on a node; tensor + pipeline across nodes
"""

# rough training memory model: bytes/param including grads + Adam(m,v) master state
_BYTES_PER_PARAM = {"adamw": 16, "adam": 16, "sgd": 8}
_ACTIVATION_OVERHEAD = 0.5  # crude activations term as a fraction of weights+opt

# a small catalogue of GPU memory (GB) — extend as providers are added
GPU_MEM_GB = {
    "t4": 16, "l4": 24, "a10g": 24, "l40s": 48,
    "a100-40": 40, "a100-80": 80, "h100": 80, "h200": 141,
}

# on-demand $/GPU-hour (rough cloud rates; override per provider)
GPU_COST_HR = {
    "t4": 0.35, "l4": 0.70, "a10g": 1.00, "l40s": 1.80,
    "a100-40": 2.50, "a100-80": 3.50, "h100": 5.00, "h200": 6.50,
}


def training_mem_gb(params_m: float, precision: str = "bf16", optimizer: str = "adamw") -> float:
    p = float(params_m) * 1e6
    per = _BYTES_PER_PARAM.get(optimizer, 16)
    weights_grads_opt = p * per
    return (weights_grads_opt * (1 + _ACTIVATION_OVERHEAD)) / 1e9


def throughput_tok_s(params_m: float, world: int) -> float:
    """Very rough training tokens/sec — falls with model size, scales with the
    pool at ~90% efficiency. For estimates only; the worker reports the truth."""
    per_gpu = 45_000.0 / max(1.0, float(params_m)) ** 0.5
    return per_gpu * max(1, world) * 0.9


def plan(params_m: float, gpu: str = "a100-40", gpus_per_node: int = 1, nodes: int = 1,
         precision: str = "bf16", optimizer: str = "adamw", tokens: float = 0.0) -> dict:
    mem = GPU_MEM_GB.get(gpu, 40)
    usable = mem * 0.8  # leave headroom for fragmentation/kernels
    need = training_mem_gb(params_m, precision, optimizer)
    world = max(1, int(gpus_per_node) * int(nodes))

    if need <= usable and world == 1:
        strat = "single"
    elif need <= usable:
        strat = "ddp"
    elif need <= usable * world:
        strat = "fsdp"
    else:
        strat = "tp+pp"

    if world == 1:
        launcher = "python dist_train.py --spec run.json"
    elif nodes == 1:
        launcher = f"torchrun --standalone --nproc_per_node={gpus_per_node} dist_train.py --spec run.json"
    else:
        launcher = (f"torchrun --nnodes={nodes} --nproc_per_node={gpus_per_node} "
                    f"--rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR dist_train.py --spec run.json")

    notes = {
        "single": "Fits on one GPU — no distribution needed.",
        "ddp": "Model replicated on each GPU; gradients all-reduced every step (throughput).",
        "fsdp": "Params, gradients, and optimizer state sharded across the whole GPU pool.",
        "tp+pp": "Layers split (tensor-parallel) and staged (pipeline-parallel) across nodes — for the largest models.",
    }

    cost_per_hr = round(GPU_COST_HR.get(gpu, 2.5) * world, 2)   # whole-pool rate
    out = {
        "strategy": strat,
        "est_train_mem_gb": round(need, 1),
        "gpu": gpu, "gpu_mem_gb": mem, "usable_gb": round(usable, 1),
        "gpus_per_node": int(gpus_per_node), "nodes": int(nodes), "world_size": world,
        "precision": precision, "optimizer": optimizer,
        "fits": strat != "tp+pp" or need <= usable * world * 4,  # tp+pp can go further
        "launcher": launcher,
        "note": notes[strat],
        "cost_per_hr": cost_per_hr,
        "throughput_tok_s": round(throughput_tok_s(params_m, world)),
    }
    if tokens and tokens > 0:
        secs = float(tokens) / max(1.0, throughput_tok_s(params_m, world))
        out["est_seconds"] = round(secs)
        out["est_gpu_seconds"] = round(secs * world)
        out["est_cost"] = round((secs / 3600.0) * cost_per_hr, 2)
    return out
