"""Social sign-in (Google, GitHub) over the OAuth 2.0 authorization-code flow.

A provider only appears once both its CLIENT_ID and CLIENT_SECRET are set, so
this is inert — not broken — until you configure credentials.

Flow: /auth/oauth/{provider}/start issues a signed state and returns the
provider's consent URL; the provider redirects back to /callback, we swap the
code for an access token, read the profile, find-or-create the user, open a
session and bounce to the app with the token in the URL fragment.
"""
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session as DbSession

from .. import models, ratelimit
from ..config import settings
from ..db import get_db
from ..notify import client_ip, record
from ..security import create_token, decode_claims, hash_password
from .auth import start_session

router = APIRouter(prefix="/auth/oauth", tags=["oauth"])

PROVIDERS = {
    "google": {
        "label": "Google",
        "authorize": "https://accounts.google.com/o/oauth2/v2/auth",
        "token": "https://oauth2.googleapis.com/token",
        "userinfo": "https://openidconnect.googleapis.com/v1/userinfo",
        "scope": "openid email profile",
    },
    "github": {
        "label": "GitHub",
        "authorize": "https://github.com/login/oauth/authorize",
        "token": "https://github.com/login/oauth/access_token",
        "userinfo": "https://api.github.com/user",
        "scope": "read:user user:email",
    },
}


def creds(provider: str) -> tuple[str, str]:
    cid = getattr(settings, f"{provider.upper()}_CLIENT_ID", "")
    secret = getattr(settings, f"{provider.upper()}_CLIENT_SECRET", "")
    return cid, secret


def configured() -> list[str]:
    return [p for p in PROVIDERS if all(creds(p))]


def redirect_uri(provider: str) -> str:
    return f"{settings.OAUTH_CALLBACK_BASE}/auth/oauth/{provider}/callback"


@router.get("/providers")
def providers():
    """Which social buttons the sign-in screen should show."""
    return [{"id": p, "label": PROVIDERS[p]["label"]} for p in configured()]


@router.get("/{provider}/start")
def start(provider: str, request: Request):
    if provider not in PROVIDERS:
        raise HTTPException(404, "Unknown provider")
    cid, _ = creds(provider)
    if not cid:
        raise HTTPException(503, f"{PROVIDERS[provider]['label']} sign-in isn't configured")
    ratelimit.hit("oauth", client_ip(request) or "unknown")
    # the state is a short-lived signed token — it comes back untampered or not at all
    state = create_token(provider, minutes=10, purpose="oauth")
    url = PROVIDERS[provider]["authorize"] + "?" + urlencode({
        "client_id": cid,
        "redirect_uri": redirect_uri(provider),
        "response_type": "code",
        "scope": PROVIDERS[provider]["scope"],
        "state": state,
        **({"access_type": "offline", "prompt": "select_account"} if provider == "google" else {}),
    })
    return {"url": url}


async def fetch_profile(provider: str, code: str) -> dict:
    """Exchange the code and return {provider_user_id, email, name}."""
    cid, secret = creds(provider)
    p = PROVIDERS[provider]
    async with httpx.AsyncClient(timeout=15) as http:
        tok = await http.post(p["token"], data={
            "client_id": cid, "client_secret": secret, "code": code,
            "redirect_uri": redirect_uri(provider), "grant_type": "authorization_code",
        }, headers={"Accept": "application/json"})
        tok.raise_for_status()
        access = tok.json().get("access_token")
        if not access:
            raise HTTPException(401, "The provider didn't return an access token")
        me = await http.get(p["userinfo"], headers={
            "Authorization": f"Bearer {access}", "Accept": "application/json",
        })
        me.raise_for_status()
        info = me.json()

        if provider == "google":
            return {"id": str(info.get("sub")), "email": (info.get("email") or "").lower(),
                    "name": info.get("name") or "", "verified": bool(info.get("email_verified"))}

        email = (info.get("email") or "").lower()
        if not email:                       # GitHub hides private addresses
            r = await http.get("https://api.github.com/user/emails",
                               headers={"Authorization": f"Bearer {access}",
                                        "Accept": "application/json"})
            if r.status_code == 200:
                primary = next((e for e in r.json() if e.get("primary") and e.get("verified")), None)
                email = (primary or {}).get("email", "").lower()
        return {"id": str(info.get("id")), "email": email,
                "name": info.get("name") or info.get("login") or "", "verified": bool(email)}


@router.get("/{provider}/callback")
async def callback(provider: str, request: Request, code: str = "", state: str = "",
                   db: DbSession = Depends(get_db)):
    if provider not in PROVIDERS or not all(creds(provider)):
        raise HTTPException(404, "Unknown or unconfigured provider")
    if not code or not decode_claims(state, purpose="oauth"):
        raise HTTPException(400, "Sign-in was cancelled or the request expired")

    profile = await fetch_profile(provider, code)
    if not profile["email"]:
        raise HTTPException(400, "That account has no usable email address")

    link = (db.query(models.OAuthAccount)
              .filter_by(provider=provider, provider_user_id=profile["id"]).first())
    user = db.get(models.User, link.user_id) if link else None

    if not user:
        user = db.query(models.User).filter_by(email=profile["email"]).first()
        if not user:
            # brand new account — no password; they can set one later if they want
            user = models.User(email=profile["email"], name=(profile["name"] or profile["email"].split("@")[0]),
                               password_hash=hash_password(create_token(profile["id"], minutes=1)),
                               email_verified=profile["verified"])
            db.add(user)
            db.flush()
            org = models.Org(name=f"{user.name}'s workspace", personal=True, plan="free")
            db.add(org)
            db.flush()
            db.add(models.Membership(org_id=org.id, user_id=user.id, role="owner"))
            for inv in db.query(models.Invite).filter_by(email=user.email, accepted=False).all():
                db.add(models.Membership(org_id=inv.org_id, user_id=user.id, role=inv.role))
                inv.accepted = True
            record(db, "auth.signup", user=user, org_id=org.id, detail=f"via {provider}",
                   ip=client_ip(request))
        elif profile["verified"]:
            user.email_verified = True
        db.add(models.OAuthAccount(user_id=user.id, provider=provider,
                                   provider_user_id=profile["id"], email=profile["email"]))

    m = (db.query(models.Membership).filter_by(user_id=user.id)
           .order_by(models.Membership.role.desc()).first())
    token = start_session(db, user, request, org_id=(m.org_id if m else None))
    record(db, "auth.login", user=user, org_id=(m.org_id if m else None),
           detail=f"via {provider}", ip=client_ip(request))
    db.commit()
    # hand the session to the SPA in the fragment (never logged by servers/proxies)
    return RedirectResponse(f"{settings.APP_URL}/#token={token}", status_code=302)
