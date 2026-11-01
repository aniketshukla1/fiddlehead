"""
capabilities.py — what can this machine realistically train?

The engine reports its hardware (cores, RAM, accelerator) and turns it into
plain recommendations per model size and per fine-tune base: which run
comfortably here, which are slow-but-possible, and which really want a cloud
GPU. The UI uses this to label the size cards honestly instead of pretending
every option is equal.

Note on Apple Silicon: MPS is reported as *present*, but for the small
batches/models this product trains it's slower than CPU (kernel-launch overhead
dominates), so local training uses CPU. The real speedup for big models is a
cloud CUDA GPU — which is exactly what the guidance nudges toward.
"""

from __future__ import annotations
import os
import platform

# rough parameter counts (millions) for the from-scratch size presets
SIZE_PARAMS_M = {"tiny": 0.1, "small": 0.8, "medium": 5, "large": 25, "xl": 85}

# fine-tune bases: hand-set tiers (LoRA changes the naive param math)
BASE_TIERS = {
    "distilgpt2": "local", "gpt2": "local",
    "HuggingFaceTB/SmolLM-135M": "local", "EleutherAI/pythia-160m": "local",
    "Qwen/Qwen2.5-0.5B": "heavy", "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "cloud",
}

TIER_NOTE = {
    "local": "trains on your machine",
    "heavy": "works locally, but slow — a cloud GPU is much faster",
    "cloud": "recommend a cloud GPU",
}


def train_device() -> str:
    """The device training actually runs on: CUDA when present (the cloud win),
    otherwise CPU. MPS is intentionally not used (slower at this scale)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _accelerator() -> tuple[bool, str]:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return True, f"NVIDIA GPU ({name})"
        if torch.backends.mps.is_available():
            return False, "Apple GPU (MPS) present — CPU is faster at these sizes"
    except Exception:
        pass
    return False, "CPU"


def _ram_gb():
    # POSIX (macOS / Linux)
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, AttributeError, OSError):
        pass
    # Windows via the Win32 API (stdlib ctypes, no dependency)
    try:
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        stat = _MemStatus()
        stat.dwLength = ctypes.sizeof(_MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return round(stat.ullTotalPhys / 1e9, 1)
    except Exception:
        return None


def detect() -> dict:
    has_cuda, accel = _accelerator()
    return {
        "cpu_cores": os.cpu_count() or 1,
        "ram_gb": _ram_gb(),
        "cuda": has_cuda,
        "accelerator": accel,
        "device": train_device(),
        "platform": f"{platform.system()} {platform.machine()}",
    }


def _size_tier(params_m: float, cuda: bool, ram_gb) -> str:
    if cuda:
        return "local"                      # a real GPU handles all these sizes
    ram = ram_gb or 8
    if params_m <= 6:
        return "local"
    if params_m <= 30 and ram >= 12:
        return "heavy"
    return "cloud"


def guidance(caps: dict | None = None) -> dict:
    caps = caps or detect()
    cuda, ram = caps.get("cuda", False), caps.get("ram_gb")
    sizes = {s: {"params_m": p, "tier": _size_tier(p, cuda, ram), "note": TIER_NOTE[_size_tier(p, cuda, ram)]}
             for s, p in SIZE_PARAMS_M.items()}
    bases = {b: {"tier": ("local" if cuda else t), "note": TIER_NOTE["local" if cuda else t]}
             for b, t in BASE_TIERS.items()}
    return {"sizes": sizes, "bases": bases}


def report() -> dict:
    """The full frame the engine sends the UI."""
    caps = detect()
    return {"type": "capabilities", **caps, **guidance(caps)}
