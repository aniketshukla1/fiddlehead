from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from .. import models, schemas
from ..db import get_db
from ..deps import current_org

router = APIRouter(prefix="/usage", tags=["usage"])


@router.get("", response_model=schemas.UsageSummary)
def usage_summary(org: models.Org = Depends(current_org), db: Session = Depends(get_db)):
    def total(kind):
        return db.query(func.coalesce(func.sum(models.UsageEvent.units), 0.0)).filter_by(
            org_id=org.id, kind=kind).scalar() or 0.0
    events = (db.query(models.UsageEvent).filter_by(org_id=org.id)
                .order_by(models.UsageEvent.created_at.desc()).limit(50).all())
    return schemas.UsageSummary(train_units=float(total("train")), infer_units=float(total("infer")),
                                credits=org.credits, events=events)
