from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import models, schemas
from ..db import get_db
from ..deps import current_org
from ..security import new_api_key

router = APIRouter(prefix="/keys", tags=["api-keys"])


@router.get("", response_model=list[schemas.ApiKeyOut])
def list_keys(org: models.Org = Depends(current_org), db: Session = Depends(get_db)):
    return (db.query(models.ApiKey).filter_by(org_id=org.id, revoked=False)
              .order_by(models.ApiKey.created_at.desc()).all())


@router.post("", response_model=schemas.ApiKeyCreated)
def create_key(body: schemas.ApiKeyIn, org: models.Org = Depends(current_org), db: Session = Depends(get_db)):
    raw, prefix, key_hash = new_api_key()
    k = models.ApiKey(org_id=org.id, name=body.name or "default", prefix=prefix, key_hash=key_hash)
    db.add(k)
    db.commit()
    db.refresh(k)
    return schemas.ApiKeyCreated(id=k.id, name=k.name, prefix=k.prefix, created_at=k.created_at,
                                 last_used_at=k.last_used_at, revoked=k.revoked, key=raw)


@router.delete("/{kid}")
def revoke_key(kid: str, org: models.Org = Depends(current_org), db: Session = Depends(get_db)):
    k = db.get(models.ApiKey, kid)
    if not k or k.org_id != org.id:
        raise HTTPException(404, "Key not found")
    k.revoked = True
    db.commit()
    return {"ok": True}
