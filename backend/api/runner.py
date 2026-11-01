"""Training-job executor.

Runs a submitted job on the compute we actually have — the local machine —
by launching `dist_train.py` (single- or multi-rank), tracking its lifecycle
(queued → running → done/failed), metering GPU-seconds, and billing the
workspace. When a real GPU worker fleet is wired up (HAS_GPU_WORKER), cloud/
cluster targets would dispatch there instead using the plan's `launcher`; the
lifecycle + metering here stay the same.
"""
import json
import os
import subprocess
import sys
import threading
import time

from . import models
from .config import settings
from .db import SessionLocal
from .notify import record

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
RUNS_DIR = os.path.join(settings.SAVE_DIR, "_jobs")
CPU_RANK_CAP = 4          # never spawn more local ranks than this on a laptop
CREDITS_PER_GPU_MIN = 1.0  # 1 credit ≈ 1 GPU-minute of training

HAS_GPU_WORKER = os.environ.get("GPU_WORKER_URL", "") != ""

_procs: dict[str, subprocess.Popen] = {}   # job_id → process, for cancellation


def local_ranks(world: int) -> int:
    return max(1, min(int(world), CPU_RANK_CAP))


def run_job(job_id: str, spec: dict, procs: int):
    db = SessionLocal()
    try:
        job = db.get(models.TrainingJob, job_id)
        if not job:
            return
        os.makedirs(RUNS_DIR, exist_ok=True)
        spec_path = os.path.join(RUNS_DIR, f"{job_id}.json")
        prog_path = spec_path[:-5] + ".progress.json"
        json.dump(spec or {}, open(spec_path, "w"))
        if os.path.exists(prog_path):
            os.remove(prog_path)

        job.status = "running"
        db.commit()

        env = {**os.environ, "OMP_NUM_THREADS": "1", "GLOO_SOCKET_IFNAME": "lo0"}
        cmd = [sys.executable, "dist_train.py", "--spec", spec_path, "--procs", str(procs)]
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd=BACKEND_DIR, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        _procs[job_id] = p

        # stream progress into the job record so the UI can watch it live
        while p.poll() is None:
            time.sleep(0.5)
            prog = _read(prog_path)
            if prog:
                job.plan = {**(job.plan or {}), "progress": prog}
                db.commit()

        rc = p.returncode
        elapsed = time.time() - t0
        prog = _read(prog_path)
        world = int(prog.get("world_size", procs)) if prog else procs
        gpu_seconds = round(elapsed * world, 2)

        canceled = job.status == "canceled" or getattr(p, "_canceled", False)
        ok = rc == 0 and bool(prog.get("done"))
        job.status = "canceled" if canceled else ("done" if ok else "failed")
        job.gpu_seconds = gpu_seconds
        job.plan = {**(job.plan or {}), "progress": prog, "final_loss": prog.get("loss"),
                    "weights_synced": prog.get("weights_synced")}

        # meter → usage event + credit burn (billing reads the same counters)
        db.add(models.UsageEvent(org_id=job.org_id, kind="train", units=gpu_seconds,
                                 meta={"job_id": job_id, "strategy": job.strategy, "world": world}))
        org = db.get(models.Org, job.org_id)
        if org and gpu_seconds > 0:
            org.credits = round(max(0.0, (org.credits or 0.0) - gpu_seconds / 60.0 * CREDITS_PER_GPU_MIN), 4)
        record(db, f"job.{job.status}", org_id=job.org_id,
               detail=f"{job.model_name} · {job.strategy} · world {world} · {gpu_seconds}s")
        db.commit()
    except Exception as e:                          # never leave a job stuck "running"
        try:
            job = db.get(models.TrainingJob, job_id)
            if job and job.status == "running":
                job.status = "failed"
                job.plan = {**(job.plan or {}), "error": str(e)[:200]}
                db.commit()
        except Exception:
            pass
    finally:
        _procs.pop(job_id, None)
        db.close()


def _read(path: str) -> dict:
    try:
        return json.load(open(path))
    except Exception:
        return {}


def dispatch(job_id: str, spec: dict, world: int):
    """Kick off a job in the background. Returns the number of local ranks used."""
    procs = local_ranks(world)
    threading.Thread(target=run_job, args=(job_id, spec, procs), daemon=True).start()
    return procs


def cancel(job_id: str) -> bool:
    p = _procs.get(job_id)
    if p and p.poll() is None:
        p._canceled = True
        p.terminate()
        return True
    return False
