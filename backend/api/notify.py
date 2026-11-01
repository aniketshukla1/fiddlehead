"""Outbound email + the audit trail.

Email sends over SMTP when SMTP_HOST is configured; otherwise it logs the
message (so local dev never silently pretends to have delivered mail).
"""
import logging
import smtplib
from email.message import EmailMessage

from . import models
from .config import settings

log = logging.getLogger("fiddlehead.notify")


def send_email(to: str, subject: str, body: str) -> bool:
    """Returns True only if the message was actually handed to an SMTP server."""
    if not settings.SMTP_HOST:
        log.info("[email:not-configured] to=%s | %s\n%s", to, subject, body)
        return False
    try:
        msg = EmailMessage()
        msg["From"] = settings.EMAIL_FROM
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=15) as s:
            if settings.SMTP_TLS:
                s.starttls()
            if settings.SMTP_USER:
                s.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            s.send_message(msg)
        return True
    except Exception as e:                                   # never break the request
        log.warning("email send failed to=%s: %s", to, e)
        return False


def send_invite(email: str, inviter: str, org_name: str, role: str) -> bool:
    return send_email(
        email,
        f"{inviter} invited you to {org_name} on {settings.PRODUCT_NAME}",
        f"""{inviter} has invited you to join “{org_name}” on {settings.PRODUCT_NAME} as {role}.

Create your account with this email address and you'll join the workspace automatically:

  {settings.APP_URL}

If you weren't expecting this, you can ignore it.""",
    )


def send_verification(email: str, name: str, token: str) -> bool:
    link = f"{settings.APP_URL}/?verify={token}"
    return send_email(
        email,
        f"Confirm your email for {settings.PRODUCT_NAME}",
        f"""Hi {name},

Confirm this address to finish setting up your {settings.PRODUCT_NAME} account:

  {link}

The link is good for 24 hours. If you didn't create an account, ignore this email.""",
    )


def record(db, action: str, *, user=None, org_id=None, detail=None, ip=None, commit=False):
    """Append a security-relevant event to the audit trail."""
    db.add(models.AuditEvent(
        org_id=org_id,
        user_id=getattr(user, "id", None),
        actor=(f"{user.name} <{user.email}>" if user is not None else None),
        action=action, detail=detail, ip=ip,
    ))
    if commit:
        db.commit()


def client_ip(request) -> str | None:
    if request is None:
        return None
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return getattr(getattr(request, "client", None), "host", None)
