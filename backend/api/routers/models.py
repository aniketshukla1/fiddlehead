from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models as m, schemas
from ..db import get_db
from ..deps import current_org

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[schemas.ModelOut])
def list_models(org: m.Org = Depends(current_org), db: Session = Depends(get_db)):
    return db.query(m.Model).filter_by(org_id=org.id).order_by(m.Model.created_at.desc()).all()


@router.post("", response_model=schemas.ModelOut)
def register_model(body: schemas.ModelIn, org: m.Org = Depends(current_org), db: Session = Depends(get_db)):
    """Register (or update) a trained model's metadata under the org — called by
    the studio when a model is saved. Upserts by (org, name)."""
    existing = db.query(m.Model).filter_by(org_id=org.id, name=body.name).first()
    data = body.model_dump(exclude_none=True)
    if existing:
        for k, v in data.items():
            setattr(existing, k, v)
        db.commit()
        db.refresh(existing)
        return existing
    row = m.Model(org_id=org.id, **data)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.post("/{mid}/deploy", response_model=schemas.ModelOut)
def deploy(mid: str, org: m.Org = Depends(current_org), db: Session = Depends(get_db)):
    row = db.get(m.Model, mid)
    if not row or row.org_id != org.id:
        raise HTTPException(404, "Model not found")
    row.deployed = True
    db.commit()
    db.refresh(row)
    return row


@router.post("/{mid}/undeploy", response_model=schemas.ModelOut)
def undeploy(mid: str, org: m.Org = Depends(current_org), db: Session = Depends(get_db)):
    row = db.get(m.Model, mid)
    if not row or row.org_id != org.id:
        raise HTTPException(404, "Model not found")
    row.deployed = False
    db.commit()
    db.refresh(row)
    return row


@router.delete("/{mid}")
def delete_model(mid: str, org: m.Org = Depends(current_org), db: Session = Depends(get_db)):
    row = db.get(m.Model, mid)
    if not row or row.org_id != org.id:
        raise HTTPException(404, "Model not found")
    db.delete(row)
    db.commit()
    return {"ok": True}
