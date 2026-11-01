"""Account & workspace settings: profile, password, organization, members,
invitations, and account deletion. Role-guarded (owner > admin > member)."""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from .. import models, schemas
from ..db import get_db
from ..deps import current_org, current_session, get_current_user
from ..notify import client_ip, record, send_invite
from ..security import (hash_password, new_recovery_codes, new_totp_secret, totp_uri,
                        verify_password, verify_totp)

router = APIRouter(prefix="/account", tags=["account"])

ROLES = ("owner", "admin", "member")
ORG_SCOPED = (models.ApiKey, models.Model, models.Dataset, models.Project,
              models.UsageEvent, models.TrainingJob, models.Invite)


def role_of(db: Session, user: models.User, org: models.Org) -> str:
    m = db.query(models.Membership).filter_by(org_id=org.id, user_id=user.id).first()
    if not m:
        raise HTTPException(403, "You are not a member of this workspace")
    return m.role


def require(role: str, *allowed: str):
    if role not in allowed:
        raise HTTPException(403, f"Requires {' or '.join(allowed)} role")


def _purge_org(db: Session, org: models.Org):
    """Delete an org and everything scoped to it."""
    for table in ORG_SCOPED:
        db.query(table).filter_by(org_id=org.id).delete(synchronize_session=False)
    db.query(models.Membership).filter_by(org_id=org.id).delete(synchronize_session=False)
    db.delete(org)


# ---- overview ---------------------------------------------------------- #
@router.get("", response_model=schemas.AccountOut)
def overview(user: models.User = Depends(get_current_user),
             org: models.Org = Depends(current_org), db: Session = Depends(get_db)):
    return schemas.AccountOut(
        user=user, org=org, role=role_of(db, user, org),
        member_count=db.query(models.Membership).filter_by(org_id=org.id).count(),
        model_count=db.query(models.Model).filter_by(org_id=org.id).count(),
        key_count=db.query(models.ApiKey).filter_by(org_id=org.id, revoked=False).count(),
        created_at=user.created_at, org_created_at=org.created_at, org_personal=bool(org.personal),
        two_factor=bool(user.totp_enabled),
        session_count=db.query(models.Session).filter_by(user_id=user.id, revoked=False).count(),
        workspace_count=db.query(models.Membership).filter_by(user_id=user.id).count(),
    )


# ---- signed-in devices ------------------------------------------------- #
@router.get("/sessions", response_model=list[schemas.SessionOut])
def list_sessions(s: models.Session = Depends(current_session),
                  user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = (db.query(models.Session).filter_by(user_id=user.id, revoked=False)
              .order_by(models.Session.last_seen_at.desc()).all())
    return [schemas.SessionOut(id=r.id, user_agent=r.user_agent, ip=r.ip, created_at=r.created_at,
                               last_seen_at=r.last_seen_at, current=(r.id == s.id)) for r in rows]


@router.delete("/sessions/{sid}")
def revoke_session(sid: str, request: Request, s: models.Session = Depends(current_session),
                   user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    target = db.get(models.Session, sid)
    if not target or target.user_id != user.id:
        raise HTTPException(404, "Session not found")
    target.revoked = True
    record(db, "session.revoked", user=user, detail=("this device" if sid == s.id else target.user_agent),
           ip=client_ip(request))
    db.commit()
    return {"ok": True, "was_current": sid == s.id}


@router.post("/sessions/revoke-others")
def revoke_other_sessions(request: Request, s: models.Session = Depends(current_session),
                          user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    n = (db.query(models.Session)
           .filter(models.Session.user_id == user.id, models.Session.id != s.id,
                   models.Session.revoked == False)          # noqa: E712
           .update({"revoked": True}, synchronize_session=False))
    record(db, "session.revoked_others", user=user, detail=f"{n} signed out", ip=client_ip(request))
    db.commit()
    return {"ok": True, "revoked": n}


# ---- two-factor authentication ----------------------------------------- #
@router.post("/2fa/setup", response_model=schemas.TotpSetupOut)
def totp_setup(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if user.totp_enabled:
        raise HTTPException(409, "Two-factor authentication is already on")
    user.totp_secret = new_totp_secret()      # stored but inert until verified
    db.commit()
    return schemas.TotpSetupOut(secret=user.totp_secret, uri=totp_uri(user.totp_secret, user.email))


@router.post("/2fa/enable", response_model=schemas.TotpEnabledOut)
def totp_enable(body: schemas.TotpEnableIn, request: Request,
                user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if user.totp_enabled:
        raise HTTPException(409, "Two-factor authentication is already on")
    if not user.totp_secret:
        raise HTTPException(422, "Start the setup step first")
    if not verify_totp(user.totp_secret, body.code):
        raise HTTPException(401, "That code isn't right — check your authenticator app")
    plain, hashes = new_recovery_codes()
    user.totp_enabled = True
    user.recovery_codes = hashes
    record(db, "2fa.enabled", user=user, ip=client_ip(request))
    db.commit()
    return schemas.TotpEnabledOut(recovery_codes=plain)


@router.post("/2fa/disable")
def totp_disable(body: schemas.TotpDisableIn, request: Request,
                 user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Password is incorrect")
    user.totp_enabled = False
    user.totp_secret = None
    user.recovery_codes = None
    record(db, "2fa.disabled", user=user, ip=client_ip(request))
    db.commit()
    return {"ok": True}


# ---- workspaces this user belongs to ----------------------------------- #
@router.get("/workspaces", response_model=list[schemas.WorkspaceOut])
def list_workspaces(org: models.Org = Depends(current_org),
                    user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    out = []
    for m in db.query(models.Membership).filter_by(user_id=user.id).all():
        o = db.get(models.Org, m.org_id)
        if not o:
            continue
        out.append(schemas.WorkspaceOut(
            id=o.id, name=o.name, plan=o.plan, role=m.role, personal=bool(o.personal),
            members=db.query(models.Membership).filter_by(org_id=o.id).count(),
            current=(o.id == org.id)))
    return sorted(out, key=lambda w: (not w.current, w.name.lower()))


@router.post("/workspaces/{oid}/switch")
def switch_workspace(oid: str, s: models.Session = Depends(current_session),
                     user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not db.query(models.Membership).filter_by(user_id=user.id, org_id=oid).first():
        raise HTTPException(403, "You're not a member of that workspace")
    s.active_org_id = oid
    db.commit()
    return {"ok": True, "org_id": oid}


# ---- audit trail -------------------------------------------------------- #
@router.get("/audit", response_model=list[schemas.AuditOut])
def audit(user: models.User = Depends(get_current_user),
          org: models.Org = Depends(current_org), db: Session = Depends(get_db)):
    role_of(db, user, org)
    q = (db.query(models.AuditEvent)
           .filter((models.AuditEvent.org_id == org.id) | (models.AuditEvent.user_id == user.id))
           .order_by(models.AuditEvent.created_at.desc()).limit(60))
    return q.all()


# ---- profile ----------------------------------------------------------- #
@router.patch("/profile", response_model=schemas.UserOut)
def update_profile(body: schemas.ProfileIn, request: Request,
                   user: models.User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    if body.name is not None:
        name = body.name.strip()
        if not name:
            raise HTTPException(422, "Name can't be empty")
        user.name = name
    if body.email is not None:
        email = body.email.lower().strip()
        if email != user.email:
            if db.query(models.User).filter_by(email=email).first():
                raise HTTPException(409, "That email is already in use")
            user.email = email
    record(db, "profile.updated", user=user, ip=client_ip(request))
    db.commit()
    db.refresh(user)
    return user


@router.post("/password")
def change_password(body: schemas.PasswordIn, request: Request,
                    s: models.Session = Depends(current_session),
                    user: models.User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    if not verify_password(body.current_password, user.password_hash):
        raise HTTPException(401, "Current password is incorrect")
    if len(body.new_password) < 6:
        raise HTTPException(422, "New password must be at least 6 characters")
    if verify_password(body.new_password, user.password_hash):
        raise HTTPException(422, "New password must be different from the current one")
    user.password_hash = hash_password(body.new_password)
    # changing a password signs out every *other* device
    n = (db.query(models.Session)
           .filter(models.Session.user_id == user.id, models.Session.id != s.id,
                   models.Session.revoked == False)          # noqa: E712
           .update({"revoked": True}, synchronize_session=False))
    record(db, "password.changed", user=user, detail=f"{n} other device(s) signed out",
           ip=client_ip(request))
    db.commit()
    return {"ok": True, "other_sessions_revoked": n}


# ---- organization ------------------------------------------------------ #
@router.patch("/org", response_model=schemas.OrgOut)
def rename_org(body: schemas.OrgIn, request: Request, user: models.User = Depends(get_current_user),
               org: models.Org = Depends(current_org), db: Session = Depends(get_db)):
    require(role_of(db, user, org), "owner", "admin")
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "Workspace name can't be empty")
    record(db, "workspace.renamed", user=user, org_id=org.id, detail=f"{org.name} → {name}",
           ip=client_ip(request))
    org.name = name
    db.commit()
    db.refresh(org)
    return org


# ---- members ----------------------------------------------------------- #
@router.get("/members", response_model=list[schemas.MemberOut])
def list_members(user: models.User = Depends(get_current_user),
                 org: models.Org = Depends(current_org), db: Session = Depends(get_db)):
    role_of(db, user, org)
    rows = (db.query(models.Membership, models.User)
              .join(models.User, models.User.id == models.Membership.user_id)
              .filter(models.Membership.org_id == org.id).all())
    order = {"owner": 0, "admin": 1, "member": 2}
    out = [schemas.MemberOut(id=m.id, user_id=u.id, name=u.name, email=u.email,
                             role=m.role, is_you=(u.id == user.id)) for m, u in rows]
    return sorted(out, key=lambda x: (order.get(x.role, 9), x.name.lower()))


@router.patch("/members/{mid}", response_model=schemas.MemberOut)
def set_role(mid: str, body: schemas.RoleIn, request: Request,
             user: models.User = Depends(get_current_user),
             org: models.Org = Depends(current_org), db: Session = Depends(get_db)):
    require(role_of(db, user, org), "owner")
    if body.role not in ROLES:
        raise HTTPException(422, f"Role must be one of {', '.join(ROLES)}")
    m = db.get(models.Membership, mid)
    if not m or m.org_id != org.id:
        raise HTTPException(404, "Member not found")
    if m.role == "owner" and body.role != "owner" and _owner_count(db, org) <= 1:
        raise HTTPException(422, "A workspace must keep at least one owner")
    u = db.get(models.User, m.user_id)
    record(db, "member.role_changed", user=user, org_id=org.id,
           detail=f"{u.email}: {m.role} → {body.role}", ip=client_ip(request))
    m.role = body.role
    db.commit()
    return schemas.MemberOut(id=m.id, user_id=u.id, name=u.name, email=u.email,
                             role=m.role, is_you=(u.id == user.id))


@router.delete("/members/{mid}")
def remove_member(mid: str, request: Request, user: models.User = Depends(get_current_user),
                  org: models.Org = Depends(current_org), db: Session = Depends(get_db)):
    my_role = role_of(db, user, org)
    m = db.get(models.Membership, mid)
    if not m or m.org_id != org.id:
        raise HTTPException(404, "Member not found")
    leaving_self = m.user_id == user.id
    if not leaving_self:
        require(my_role, "owner", "admin")
    if m.role == "owner" and _owner_count(db, org) <= 1:
        raise HTTPException(422, "A workspace must keep at least one owner")
    gone = db.get(models.User, m.user_id)
    record(db, "member.left" if leaving_self else "member.removed", user=user, org_id=org.id,
           detail=(gone.email if gone else m.user_id), ip=client_ip(request))
    db.delete(m)
    db.commit()
    return {"ok": True, "left": leaving_self}


def _owner_count(db: Session, org: models.Org) -> int:
    return db.query(models.Membership).filter_by(org_id=org.id, role="owner").count()


# ---- invitations ------------------------------------------------------- #
@router.get("/invites", response_model=list[schemas.InviteOut])
def list_invites(user: models.User = Depends(get_current_user),
                 org: models.Org = Depends(current_org), db: Session = Depends(get_db)):
    role_of(db, user, org)
    return (db.query(models.Invite).filter_by(org_id=org.id, accepted=False)
              .order_by(models.Invite.created_at.desc()).all())


@router.post("/invites")
def invite(body: schemas.InviteIn, request: Request, user: models.User = Depends(get_current_user),
           org: models.Org = Depends(current_org), db: Session = Depends(get_db)):
    require(role_of(db, user, org), "owner", "admin")
    if body.role not in ("admin", "member"):
        raise HTTPException(422, "Invite role must be admin or member")
    email = body.email.lower().strip()
    existing = db.query(models.User).filter_by(email=email).first()
    if existing:
        if db.query(models.Membership).filter_by(org_id=org.id, user_id=existing.id).first():
            raise HTTPException(409, "That person is already in this workspace")
        db.add(models.Membership(org_id=org.id, user_id=existing.id, role=body.role))
        record(db, "member.added", user=user, org_id=org.id, detail=f"{email} as {body.role}",
               ip=client_ip(request))
        db.commit()
        sent = send_invite(email, user.name, org.name, body.role)
        return {"ok": True, "added": True, "email": email, "emailed": sent}
    if db.query(models.Invite).filter_by(org_id=org.id, email=email, accepted=False).first():
        raise HTTPException(409, "That email already has a pending invite")
    inv = models.Invite(org_id=org.id, email=email, role=body.role, invited_by=user.id)
    db.add(inv)
    record(db, "member.invited", user=user, org_id=org.id, detail=f"{email} as {body.role}",
           ip=client_ip(request))
    db.commit()
    sent = send_invite(email, user.name, org.name, body.role)
    return {"ok": True, "added": False, "email": email, "pending": True, "emailed": sent}


@router.delete("/invites/{iid}")
def revoke_invite(iid: str, user: models.User = Depends(get_current_user),
                  org: models.Org = Depends(current_org), db: Session = Depends(get_db)):
    require(role_of(db, user, org), "owner", "admin")
    inv = db.get(models.Invite, iid)
    if not inv or inv.org_id != org.id:
        raise HTTPException(404, "Invite not found")
    db.delete(inv)
    db.commit()
    return {"ok": True}


# ---- danger zone ------------------------------------------------------- #
@router.delete("")
def delete_account(body: schemas.DeleteAccountIn, user: models.User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    if not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Password is incorrect")
    if body.confirm.lower().strip() != user.email:
        raise HTTPException(422, "Type your email exactly to confirm")
    # any org where this user is the only member is deleted outright;
    # otherwise just drop their membership (never orphan a shared workspace).
    for m in list(db.query(models.Membership).filter_by(user_id=user.id).all()):
        org = db.get(models.Org, m.org_id)
        others = db.query(models.Membership).filter(
            models.Membership.org_id == m.org_id, models.Membership.user_id != user.id).count()
        if org and others == 0:
            _purge_org(db, org)
        else:
            if m.role == "owner" and _owner_count(db, org) <= 1:
                raise HTTPException(422,
                                    f"Transfer ownership of “{org.name}” before deleting your account")
            db.delete(m)
    db.query(models.Session).filter_by(user_id=user.id).delete(synchronize_session=False)
    db.delete(user)
    db.commit()
    return {"ok": True}
