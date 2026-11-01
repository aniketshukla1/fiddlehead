"""Fiddlehead control-plane API.

Run (from backend/):  uvicorn api.app:app --port 8000 --reload
Interactive docs:     http://localhost:8000/docs
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# our own loggers at INFO so dev can see things like the "email not configured"
# fallback that prints the confirmation/invite link instead of sending it
logging.getLogger("fiddlehead").setLevel(logging.INFO)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")

from . import models  # noqa: F401 — registers ORM tables on Base.metadata
from .config import settings
from .db import Base, engine, ensure_columns
from .routers import account, auth, billing, infer, jobs, keys, oauth, projects, usage
from .routers import models as models_router

# dev convenience: auto-create tables, then add any newly-declared columns to
# tables that already exist. Prod uses Alembic migrations.
Base.metadata.create_all(bind=engine)
_migrated = ensure_columns()
if _migrated:
    print(f"[schema] added columns: {', '.join(_migrated)}")

app = FastAPI(
    title="Fiddlehead API",
    version="1.0.0",
    description="Accounts, models, keys and hosted inference for Fiddlehead.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for r in (auth.router, oauth.router, account.router, billing.router, projects.router,
          models_router.router, keys.router, usage.router, infer.router, jobs.router):
    app.include_router(r)


@app.get("/health")
def health():
    return {"ok": True, "service": "fiddlehead-api", "version": "1.0.0"}


@app.get("/meta")
def meta():
    """Public: which optional integrations this deployment has switched on, so the
    UI can hide what isn't configured (e.g. plans & credits on a self-hosted install)."""
    return {
        "version": "1.0.0",
        "billing": billing.enabled(),
        "email": bool(settings.SMTP_HOST),
        "oauth": oauth.configured(),
    }
