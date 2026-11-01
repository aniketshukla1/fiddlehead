from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas
from ..db import get_db
from ..deps import current_org, get_current_user

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=list[schemas.ProjectOut])
def list_projects(org: models.Org = Depends(current_org), db: Session = Depends(get_db)):
    return (db.query(models.Project).filter_by(org_id=org.id)
              .order_by(models.Project.created_at.desc()).all())


@router.post("", response_model=schemas.ProjectOut)
def create_project(body: schemas.ProjectIn, org: models.Org = Depends(current_org),
                   user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    p = models.Project(org_id=org.id, name=body.name, template=body.template, created_by=user.id)
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


@router.delete("/{pid}")
def delete_project(pid: str, org: models.Org = Depends(current_org), db: Session = Depends(get_db)):
    p = db.get(models.Project, pid)
    if not p or p.org_id != org.id:
        raise HTTPException(404, "Project not found")
    db.delete(p)
    db.commit()
    return {"ok": True}
