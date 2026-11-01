"""Compute planning + training-job orchestration.

/compute/plan is pure decision logic (how to split a model across GPUs).
/jobs submits a run: it's recorded, then executed by the runner — which launches
the real distributed trainer (dist_train.py) and meters GPU-seconds into billing.
Cloud/cluster targets would dispatch to a GPU worker fleet when one is wired up;
the lifecycle and metering are identical either way.
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .. import models, runner
from . import billing
from ..compute import GPU_COST_HR, GPU_MEM_GB, plan as plan_fn
from ..db import get_db
from ..deps import current_org, get_current_user
from ..notify import client_ip, record

router = APIRouter(tags=["compute"])


class PlanIn(BaseModel):
    params_m: float
    gpu: str = "a100-40"
    gpus_per_node: int = 1
    nodes: int = 1
    precision: str = "bf16"
    tokens: float = 0.0        # optional token budget → cost/ETA estimate


@router.post("/compute/plan")
def compute_plan(body: PlanIn):
    return plan_fn(body.params_m, body.gpu, body.gpus_per_node, body.nodes,
                   body.precision, tokens=body.tokens)


@router.get("/compute/gpus")
def list_gpus():
    return [{"id": k, "mem_gb": v, "cost_hr": GPU_COST_HR.get(k)} for k, v in GPU_MEM_GB.items()]


class JobIn(BaseModel):
    model_name: str
    target: str = "cloud"           # local | cloud | cluster
    gpu: str = "a100-40"
    gpus_per_node: int = 1
    nodes: int = 1
    params_m: float = 1.0
    spec: dict | None = None


def _view(r: models.TrainingJob) -> dict:
    plan = r.plan or {}
    return {
        "id": r.id, "model_name": r.model_name, "target": r.target, "gpu": r.gpu,
        "gpus_per_node": r.gpus_per_node, "nodes": r.nodes,
        "world_size": (r.gpus_per_node or 1) * (r.nodes or 1),
        "strategy": r.strategy, "status": r.status, "gpu_seconds": r.gpu_seconds,
        "created_at": r.created_at,
        "progress": plan.get("progress"), "final_loss": plan.get("final_loss"),
        "weights_synced": plan.get("weights_synced"),
        "cost_per_hr": plan.get("cost_per_hr"), "note": plan.get("note"),
    }


@router.post("/jobs")
def submit_job(body: JobIn, request: Request, user: models.User = Depends(get_current_user),
               org: models.Org = Depends(current_org), db: Session = Depends(get_db)):
    # credits only gate jobs when this deployment sells them; self-hosted runs are unmetered
    if billing.enabled() and (org.credits or 0) <= 0:
        raise HTTPException(402, "Out of compute credits — top up or upgrade to run more jobs.")
    p = plan_fn(body.params_m, body.gpu, body.gpus_per_node, body.nodes)
    world = p["world_size"]
    job = models.TrainingJob(
        org_id=org.id, model_name=body.model_name, target=body.target,
        gpu=body.gpu, gpus_per_node=body.gpus_per_node, nodes=body.nodes,
        strategy=p["strategy"], status="queued", plan=p, spec=body.spec,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # execute it (locally today; a GPU worker fleet later — see runner.HAS_GPU_WORKER)
    ranks = runner.dispatch(job.id, body.spec or {}, world)
    record(db, "job.submitted", user=user, org_id=org.id,
           detail=f"{body.model_name} · {p['strategy']} · world {world}", ip=client_ip(request))
    db.commit()
    return {"id": job.id, "status": "running", "target": job.target,
            "strategy": job.strategy, "world_size": world, "local_ranks": ranks, "plan": p}


@router.get("/jobs")
def list_jobs(org: models.Org = Depends(current_org), db: Session = Depends(get_db)):
    rows = (db.query(models.TrainingJob).filter_by(org_id=org.id)
              .order_by(models.TrainingJob.created_at.desc()).limit(50).all())
    return [_view(r) for r in rows]


@router.get("/jobs/{jid}")
def get_job(jid: str, org: models.Org = Depends(current_org), db: Session = Depends(get_db)):
    r = db.get(models.TrainingJob, jid)
    if not r or r.org_id != org.id:
        raise HTTPException(404, "Job not found")
    return _view(r)


@router.post("/jobs/{jid}/cancel")
def cancel_job(jid: str, org: models.Org = Depends(current_org), db: Session = Depends(get_db)):
    r = db.get(models.TrainingJob, jid)
    if not r or r.org_id != org.id:
        raise HTTPException(404, "Job not found")
    if r.status in ("done", "failed", "canceled"):
        return {"ok": True, "status": r.status}
    r.status = "canceled"
    db.commit()
    runner.cancel(jid)
    return {"ok": True, "status": "canceled"}
