#!/usr/bin/env python3
"""Report which integrations are configured, and print the exact values to paste
into the Google / GitHub / Stripe consoles.

    python3 check_config.py

Never prints a secret — only whether one is present.
"""
import os
import sys

sys.path.insert(0, ".")
from api.config import settings  # noqa: E402

OK, NO, WARN = "\033[32m✔\033[0m", "\033[90m—\033[0m", "\033[33m!\033[0m"


def mark(cond):
    return OK if cond else NO


def main():
    print("\n\033[1mFiddlehead configuration\033[0m")
    print(f"  app url            {settings.APP_URL}")
    print(f"  api callback base  {settings.OAUTH_CALLBACK_BASE}")

    from_env = bool(os.environ.get("JWT_SECRET"))
    print(f"\n  {OK if from_env else WARN} JWT_SECRET "
          f"{'set' if from_env else 'generated and kept in backend/.jwt_secret — set JWT_SECRET when running more than one API server'}")
    print(f"  {mark(not settings.DATABASE_URL.startswith('sqlite'))} database  "
          f"{'sqlite (fine for dev)' if settings.DATABASE_URL.startswith('sqlite') else 'external'}")

    print("\n\033[1mSocial sign-in\033[0m")
    any_oauth = False
    for name, cid, secret in (
        ("Google", settings.GOOGLE_CLIENT_ID, settings.GOOGLE_CLIENT_SECRET),
        ("GitHub", settings.GITHUB_CLIENT_ID, settings.GITHUB_CLIENT_SECRET),
    ):
        ready = bool(cid and secret)
        any_oauth = any_oauth or ready
        print(f"  {mark(ready)} {name:<7} client id {'set' if cid else 'missing':<7} "
              f" secret {'set' if secret else 'missing'}")
        print(f"      redirect URI → {settings.OAUTH_CALLBACK_BASE}/auth/oauth/{name.lower()}/callback")
    if not any_oauth:
        print("      (buttons stay hidden until a provider has BOTH values)")

    print("\n\033[1mBilling\033[0m")
    print(f"  {mark(bool(settings.STRIPE_SECRET))} STRIPE_SECRET")
    print(f"  {mark(bool(settings.STRIPE_PRICE_PRO))} price: pro   "
          f"{settings.STRIPE_PRICE_PRO or ''}")
    print(f"  {mark(bool(settings.STRIPE_PRICE_TEAM))} price: team  "
          f"{settings.STRIPE_PRICE_TEAM or ''}")
    print(f"  {mark(bool(settings.STRIPE_WEBHOOK_SECRET))} webhook secret")
    print(f"      webhook URL  → {settings.OAUTH_CALLBACK_BASE}/billing/webhook")
    if settings.STRIPE_SECRET.startswith("sk_live"):
        print(f"  {WARN} using a LIVE Stripe key")

    print("\n\033[1mEmail\033[0m")
    if settings.SMTP_HOST:
        print(f"  {OK} SMTP {settings.SMTP_HOST}:{settings.SMTP_PORT} "
              f"tls={'on' if settings.SMTP_TLS else 'off'} from={settings.EMAIL_FROM}")
    else:
        print(f"  {NO} SMTP not set — confirmation and invite links are printed to the API log")
    print()


if __name__ == "__main__":
    main()
