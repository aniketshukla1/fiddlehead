"""Real accounts: signup/login open a server-side Session and issue a JWT bound
to it (so it can be revoked); two-factor auth adds a second step."""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session as DbSession

from .. import models, ratelimit, schemas
from ..config import settings
from ..db import get_db
from ..deps import current_org, current_session, get_current_user
from ..notify import client_ip, record, send_verification
from ..security import (create_token, decode_claims, hash_password, hash_recovery_code,
                        verify_password, verify_totp)

router = APIRouter(prefix="/auth", tags=["auth"])

MFA_TOKEN_MIN = 10          # the between-steps token is short-lived
VERIFY_TOKEN_MIN = 60 * 24  # email-confirmation links last a day


def issue_verification(user: models.User) -> str:
    token = create_token(user.id, minutes=VERIFY_TOKEN_MIN, purpose="verify")
    send_verification(user.email, user.name, token)
    return token


def default_org(db: DbSession, user: models.User) -> models.Org | None:
    m = (db.query(models.Membership).filter_by(user_id=user.id)
           .order_by(models.Membership.role.desc()).first())
    return db.get(models.Org, m.org_id) if m else None


def start_session(db: DbSession, user: models.User, request: Request, org_id: str | None = None) -> str:
    s = models.Session(
        user_id=user.id, active_org_id=org_id,
        user_agent=(request.headers.get("user-agent") if request else None),
        ip=client_ip(request),
    )
    db.add(s)
    db.flush()
    return create_token(user.id, jti=s.id)


@router.post("/signup", response_model=schemas.AuthOut)
def signup(body: schemas.SignupIn, request: Request, db: DbSession = Depends(get_db)):
    ratelimit.hit("signup", client_ip(request) or "unknown")
    email = body.email.lower().strip()
    if len(body.password) < 6:
        raise HTTPException(422, "Password must be at least 6 characters")
    if db.query(models.User).filter_by(email=email).first():
        raise HTTPException(409, "An account with this email already exists")
    user = models.User(email=email, name=(body.name.strip() or email.split("@")[0]),
                       password_hash=hash_password(body.password))
    db.add(user)
    db.flush()
    org = models.Org(name=f"{user.name}'s workspace", personal=True, plan="free")
    db.add(org)
    db.flush()
    db.add(models.Membership(org_id=org.id, user_id=user.id, role="owner"))
    # honour any pending workspace invitations for this email
    for inv in db.query(models.Invite).filter_by(email=email, accepted=False).all():
        db.add(models.Membership(org_id=inv.org_id, user_id=user.id, role=inv.role))
        inv.accepted = True
        record(db, "member.joined", user=user, org_id=inv.org_id, detail="accepted invitation",
               ip=client_ip(request))
    token = start_session(db, user, request, org_id=org.id)
    record(db, "auth.signup", user=user, org_id=org.id, ip=client_ip(request))
    db.commit()
    db.refresh(user)
    db.refresh(org)
    issue_verification(user)      # confirmation link (logged in dev, mailed in prod)
    return schemas.AuthOut(token=token, user=user, org=org)


@router.post("/verify-email")
def verify_email(body: schemas.VerifyIn, request: Request, db: DbSession = Depends(get_db)):
    claims = decode_claims(body.token, purpose="verify")
    user = db.get(models.User, claims["sub"]) if claims else None
    if not user:
        raise HTTPException(400, "That confirmation link is invalid or has expired")
    if not user.email_verified:
        user.email_verified = True
        record(db, "email.verified", user=user, ip=client_ip(request))
        db.commit()
    return {"ok": True, "email": user.email}


@router.post("/resend-verification")
def resend_verification(request: Request, user: models.User = Depends(get_current_user)):
    if user.email_verified:
        return {"ok": True, "already": True}
    ratelimit.hit("verify_send", user.id)
    issue_verification(user)
    return {"ok": True, "sent_to": user.email}


@router.post("/login", response_model=schemas.LoginOut)
def login(body: schemas.LoginIn, request: Request, db: DbSession = Depends(get_db)):
    email = body.email.lower().strip()
    ip = client_ip(request) or "unknown"
    ratelimit.hit("login_ip", ip)               # blunt stop for credential stuffing
    ratelimit.hit("login", f"{email}|{ip}")     # per-account throttle
    user = db.query(models.User).filter_by(email=email).first()
    if not user or not verify_password(body.password, user.password_hash):
        record(db, "auth.login_failed", org_id=None, detail=email, ip=ip, commit=True)
        raise HTTPException(401, "Wrong email or password")
    ratelimit.reset("login", f"{email}|{ip}")   # a good password clears the streak
    if user.totp_enabled:
        # password was right, but we don't open a session until the code checks out
        return schemas.LoginOut(mfa_required=True,
                                mfa_token=create_token(user.id, minutes=MFA_TOKEN_MIN, purpose="mfa"))
    org = default_org(db, user)
    token = start_session(db, user, request, org_id=(org.id if org else None))
    record(db, "auth.login", user=user, org_id=(org.id if org else None), ip=client_ip(request))
    db.commit()
    return schemas.LoginOut(token=token, user=user, org=org)


@router.post("/mfa", response_model=schemas.AuthOut)
def login_mfa(body: schemas.MfaIn, request: Request, db: DbSession = Depends(get_db)):
    claims = decode_claims(body.mfa_token, purpose="mfa")
    user = db.get(models.User, claims["sub"]) if claims else None
    if not user or not user.totp_enabled:
        raise HTTPException(401, "That verification step expired — sign in again")
    ratelimit.hit("mfa", user.id)
    code = (body.code or "").strip()
    ok = verify_totp(user.totp_secret or "", code)
    used_recovery = False
    if not ok:
        codes = list(user.recovery_codes or [])
        h = hash_recovery_code(code)
        if h in codes:
            codes.remove(h)                       # single use
            user.recovery_codes = codes
            ok = used_recovery = True
    if not ok:
        record(db, "auth.mfa_failed", user=user, ip=client_ip(request), commit=True)
        raise HTTPException(401, "That code isn't right")
    org = default_org(db, user)
    token = start_session(db, user, request, org_id=(org.id if org else None))
    record(db, "auth.login", user=user, org_id=(org.id if org else None),
           detail="recovery code used" if used_recovery else "2FA verified", ip=client_ip(request))
    db.commit()
    return schemas.AuthOut(token=token, user=user, org=org)


@router.post("/logout")
def logout(request: Request, s: models.Session = Depends(current_session),
           user: models.User = Depends(get_current_user), db: DbSession = Depends(get_db)):
    s.revoked = True
    record(db, "auth.logout", user=user, org_id=s.active_org_id, ip=client_ip(request))
    db.commit()
    return {"ok": True}


@router.get("/me", response_model=schemas.AuthOut)
def me(s: models.Session = Depends(current_session),
       user: models.User = Depends(get_current_user),
       org: models.Org = Depends(current_org)):
    # re-issue against the same session so the token keeps its jti
    return schemas.AuthOut(token=create_token(user.id, jti=s.id), user=user, org=org)
