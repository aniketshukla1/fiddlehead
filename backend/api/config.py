"""Runtime config for the Fiddlehead control-plane API.

Everything is env-overridable so the same code runs local (SQLite, dev secret)
and in production (Postgres via DATABASE_URL, real JWT secret, Stripe keys)."""
import os


def _load_dotenv():
    """Read backend/.env into the environment (real env vars always win).

    Keeps secrets out of the shell history and out of git — no dependency."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.isfile(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)


_load_dotenv()

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _jwt_secret() -> str:
    """JWT_SECRET from the environment, else a random one generated on first run and
    kept in backend/.jwt_secret (git-ignored). A fresh self-host is never left signing
    sessions with a key that is published in this repository."""
    if os.environ.get("JWT_SECRET"):
        return os.environ["JWT_SECRET"]
    path = os.path.join(_BACKEND_DIR, ".jwt_secret")
    try:
        with open(path) as fh:
            kept = fh.read().strip()
        if kept:
            return kept
    except FileNotFoundError:
        pass
    import secrets
    kept = secrets.token_urlsafe(48)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(kept)
    return kept


class Settings:
    # SQLite for local dev; set DATABASE_URL=postgresql+psycopg://… in prod
    DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./fiddlehead.db")
    JWT_SECRET = _jwt_secret()
    JWT_ALG = "HS256"
    JWT_EXPIRE_MIN = int(os.environ.get("JWT_EXPIRE_MIN", str(60 * 24 * 7)))  # 7 days
    CORS_ORIGINS = os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:5199,http://localhost:5173,http://localhost:1420,tauri://localhost",
    ).split(",")
    # where trained model artifacts live (the training engine writes here)
    SAVE_DIR = (os.environ.get("FIDDLEHEAD_SAVE_DIR") or os.environ.get("SAVE_DIR")
                or os.path.join(_BACKEND_DIR, "saved"))
    # ---- billing (Stripe). Empty STRIPE_SECRET → checkout is simply unavailable.
    STRIPE_SECRET = os.environ.get("STRIPE_SECRET", "")
    STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
    STRIPE_PRICE_PRO = os.environ.get("STRIPE_PRICE_PRO", "")
    STRIPE_PRICE_TEAM = os.environ.get("STRIPE_PRICE_TEAM", "")
    PLAN_CREDITS = {"free": 5.0, "pro": 500.0, "team": 2000.0}

    # ---- social sign-in. A provider appears in the UI only once its pair is set.
    GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
    GITHUB_CLIENT_ID = os.environ.get("GITHUB_CLIENT_ID", "")
    GITHUB_CLIENT_SECRET = os.environ.get("GITHUB_CLIENT_SECRET", "")
    # where providers send the browser back (this API), e.g. https://api.example.com
    OAUTH_CALLBACK_BASE = os.environ.get("OAUTH_CALLBACK_BASE", "http://localhost:8010")

    REQUIRE_EMAIL_VERIFICATION = os.environ.get("REQUIRE_EMAIL_VERIFICATION", "1") not in ("0", "false", "False")

    # outbound email — unset SMTP_HOST logs messages instead of sending (dev)
    SMTP_HOST = os.environ.get("SMTP_HOST", "")
    SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
    SMTP_USER = os.environ.get("SMTP_USER", "")
    SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
    SMTP_TLS = os.environ.get("SMTP_TLS", "1") not in ("0", "false", "False")
    EMAIL_FROM = os.environ.get("EMAIL_FROM", "Fiddlehead <no-reply@localhost>")
    APP_URL = os.environ.get("APP_URL", "http://localhost:5173")

    PRODUCT_NAME = "Fiddlehead"


settings = Settings()
