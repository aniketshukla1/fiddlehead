"""Request dependencies: the signed-in device (session), its user, and the
workspace that device is currently acting in."""
from datetime import datetime, timedelta, timezone

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session as DbSession

from . import models
from .db import get_db
from .security import decode_claims

TOUCH_AFTER = timedelta(seconds=60)   # don't write last_seen on every request


def current_session(authorization: str = Header(None),
                    db: DbSession = Depends(get_db)) -> models.Session:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Not authenticated")
    claims = decode_claims(authorization.split(" ", 1)[1])
    if not claims or not claims.get("jti"):
        # tokens issued before server-side sessions existed — force a fresh sign-in
        raise HTTPException(401, "Invalid or expired session")
    s = db.get(models.Session, claims["jti"])
    if not s or s.revoked or s.user_id != claims.get("sub"):
        raise HTTPException(401, "This session has been signed out")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if not s.last_seen_at or now - s.last_seen_at > TOUCH_AFTER:
        s.last_seen_at = now
        db.commit()
    return s


def get_current_user(s: models.Session = Depends(current_session),
                     db: DbSession = Depends(get_db)) -> models.User:
    user = db.get(models.User, s.user_id)
    if not user:
        raise HTTPException(401, "Invalid or expired session")
    return user


def current_org(s: models.Session = Depends(current_session),
                user: models.User = Depends(get_current_user),
                db: DbSession = Depends(get_db)) -> models.Org:
    """The workspace this device is in — the session's active org when the user
    is still a member of it, otherwise their highest-role membership."""
    if s.active_org_id:
        m = db.query(models.Membership).filter_by(user_id=user.id, org_id=s.active_org_id).first()
        if m:
            org = db.get(models.Org, s.active_org_id)
            if org:
                return org
    m = (db.query(models.Membership)
           .filter_by(user_id=user.id)
           .order_by(models.Membership.role.desc())
           .first())
    org = db.get(models.Org, m.org_id) if m else None
    if not org:
        raise HTTPException(404, "No organization for this user")
    return org
