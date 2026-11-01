"""Subscription billing on Stripe, over its REST API (no SDK dependency).

Inert until STRIPE_SECRET is set: /status reports `configured: false` and the UI
keeps checkout disabled rather than pretending. Usage metering (usage_events)
already works regardless — this only adds the paid plans on top.
"""
import hashlib
import hmac
import json
import time
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session as DbSession

from .. import models
from ..config import settings
from ..db import get_db
from ..deps import current_org, get_current_user
from ..notify import client_ip, record

router = APIRouter(prefix="/billing", tags=["billing"])

STRIPE_API = "https://api.stripe.com/v1"
PRICES = {"pro": lambda: settings.STRIPE_PRICE_PRO, "team": lambda: settings.STRIPE_PRICE_TEAM}
WEBHOOK_TOLERANCE = 300   # seconds of clock skew allowed on a webhook


def enabled() -> bool:
    return bool(settings.STRIPE_SECRET)


def require_enabled():
    if not enabled():
        raise HTTPException(503, "Billing isn't configured on this deployment")


async def stripe_post(path: str, data: dict) -> dict:
    """Stripe takes form-encoded bodies; nested keys use a[b] syntax."""
    async with httpx.AsyncClient(timeout=20) as http:
        r = await http.post(f"{STRIPE_API}{path}", data=data,
                            auth=(settings.STRIPE_SECRET, ""))
    if r.status_code >= 400:
        detail = r.json().get("error", {}).get("message", "Stripe rejected the request")
        raise HTTPException(502, f"Stripe: {detail}")
    return r.json()


def role_of(db: DbSession, user: models.User, org: models.Org) -> str:
    m = db.query(models.Membership).filter_by(org_id=org.id, user_id=user.id).first()
    if not m:
        raise HTTPException(403, "You are not a member of this workspace")
    return m.role


@router.get("/status")
def status(org: models.Org = Depends(current_org)):
    return {
        "configured": enabled(),
        "plan": org.plan,
        "plan_status": org.plan_status,
        "renews_at": org.plan_renews_at,
        "has_subscription": bool(org.stripe_subscription_id),
        "credits": org.credits,
        "available_plans": [p for p, price in PRICES.items() if price()] if enabled() else [],
    }


@router.post("/checkout")
async def checkout(body: dict, request: Request, user: models.User = Depends(get_current_user),
                   org: models.Org = Depends(current_org), db: DbSession = Depends(get_db)):
    require_enabled()
    if role_of(db, user, org) not in ("owner", "admin"):
        raise HTTPException(403, "Only owners and admins can change the plan")
    plan = str(body.get("plan", "")).lower()
    price = PRICES.get(plan, lambda: "")()
    if not price:
        raise HTTPException(422, f"No Stripe price configured for the {plan or '—'} plan")

    data = {
        "mode": "subscription",
        "line_items[0][price]": price,
        "line_items[0][quantity]": 1,
        "client_reference_id": org.id,
        "success_url": f"{settings.APP_URL}/?billing=success",
        "cancel_url": f"{settings.APP_URL}/?billing=cancelled",
        "subscription_data[metadata][org_id]": org.id,
        "metadata[org_id]": org.id,
    }
    if org.stripe_customer_id:
        data["customer"] = org.stripe_customer_id
    else:
        data["customer_email"] = user.email

    session = await stripe_post("/checkout/sessions", data)
    record(db, "billing.checkout_started", user=user, org_id=org.id, detail=plan,
           ip=client_ip(request), commit=True)
    return {"url": session["url"], "id": session["id"]}


@router.post("/portal")
async def portal(user: models.User = Depends(get_current_user),
                 org: models.Org = Depends(current_org), db: DbSession = Depends(get_db)):
    require_enabled()
    if role_of(db, user, org) not in ("owner", "admin"):
        raise HTTPException(403, "Only owners and admins can manage billing")
    if not org.stripe_customer_id:
        raise HTTPException(422, "This workspace has no billing account yet")
    s = await stripe_post("/billing_portal/sessions", {
        "customer": org.stripe_customer_id,
        "return_url": f"{settings.APP_URL}/",
    })
    return {"url": s["url"]}


# ---- webhook ----------------------------------------------------------- #
def verify_signature(payload: bytes, header: str, secret: str) -> bool:
    """Stripe's scheme: sha256 HMAC over "{timestamp}.{raw body}"."""
    if not header or not secret:
        return False
    parts = dict(p.split("=", 1) for p in header.split(",") if "=" in p)
    ts, sent = parts.get("t"), parts.get("v1")
    if not ts or not sent:
        return False
    try:
        if abs(time.time() - int(ts)) > WEBHOOK_TOLERANCE:
            return False
    except ValueError:
        return False
    expected = hmac.new(secret.encode(), f"{ts}.".encode() + payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sent)


def apply_plan(db: DbSession, org: models.Org, plan: str, *, status: str | None = None,
               sub_id: str | None = None, customer: str | None = None, renews: int | None = None):
    """Move a workspace onto a plan and top its credits up to that plan's grant."""
    org.plan = plan
    if status is not None:
        org.plan_status = status
    if sub_id is not None:
        org.stripe_subscription_id = sub_id
    if customer:
        org.stripe_customer_id = customer
    if renews:
        org.plan_renews_at = datetime.fromtimestamp(renews, tz=timezone.utc).replace(tzinfo=None)
    grant = settings.PLAN_CREDITS.get(plan)
    if grant is not None and (org.credits or 0) < grant:
        org.credits = grant


def plan_for_price(price_id: str) -> str | None:
    for plan, get in PRICES.items():
        if price_id and get() == price_id:
            return plan
    return None


@router.post("/webhook")
async def webhook(request: Request, db: DbSession = Depends(get_db)):
    raw = await request.body()
    if not verify_signature(raw, request.headers.get("stripe-signature", ""),
                            settings.STRIPE_WEBHOOK_SECRET):
        raise HTTPException(400, "Bad signature")
    event = json.loads(raw or b"{}")
    obj = (event.get("data") or {}).get("object") or {}
    kind = event.get("type")

    org = None
    if obj.get("client_reference_id"):
        org = db.get(models.Org, obj["client_reference_id"])
    if not org and (obj.get("metadata") or {}).get("org_id"):
        org = db.get(models.Org, obj["metadata"]["org_id"])
    if not org and obj.get("customer"):
        org = db.query(models.Org).filter_by(stripe_customer_id=obj["customer"]).first()
    if not org:
        return {"ok": True, "ignored": "no matching workspace"}

    if kind == "checkout.session.completed":
        plan = (obj.get("metadata") or {}).get("plan")
        apply_plan(db, org, plan or org.plan or "pro", status="active",
                   sub_id=obj.get("subscription"), customer=obj.get("customer"))
    elif kind in ("customer.subscription.created", "customer.subscription.updated"):
        item = ((obj.get("items") or {}).get("data") or [{}])[0]
        plan = plan_for_price((item.get("price") or {}).get("id", "")) or org.plan
        apply_plan(db, org, plan, status=obj.get("status"), sub_id=obj.get("id"),
                   customer=obj.get("customer"), renews=obj.get("current_period_end"))
    elif kind == "customer.subscription.deleted":
        org.plan = "free"
        org.plan_status = "canceled"
        org.stripe_subscription_id = None
    else:
        return {"ok": True, "ignored": kind}

    record(db, "billing.updated", org_id=org.id, detail=f"{kind} → {org.plan}", commit=True)
    return {"ok": True, "plan": org.plan}
