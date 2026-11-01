"""Hosted inference — call a deployed model with an API key, metered per call.
This is the recurring-revenue surface: every request records a usage_event.

A real deployment runs a dedicated inference worker with model caching; here the
control-plane process loads + caches the model in-process, which is honest and
fast enough for the local product and small models.
"""
import os
import sys
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .. import models
from ..config import settings
from ..db import get_db
from ..security import hash_api_key

router = APIRouter(prefix="/v1", tags=["inference"])

_cache: dict = {}  # model_name -> (generate_fn, meta)


def _load(name: str):
    if name in _cache:
        return _cache[name]
    backend_dir = os.path.dirname(settings.SAVE_DIR)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)  # so `import serve` / artifacts / model resolve
    from artifacts import meta_path
    if not os.path.isfile(meta_path(os.path.join(settings.SAVE_DIR, name))):
        raise HTTPException(404, f"model artifact for '{name}' not found on disk")
    from serve import load_model
    _cache[name] = load_model(name)
    return _cache[name]


def api_key_org(authorization: str = Header(None), db: Session = Depends(get_db)):
    """Authenticate by API key (not JWT). Returns (org, api_key)."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Provide your API key: Authorization: Bearer sk-…")
    raw = authorization.split(" ", 1)[1].strip()
    key = db.query(models.ApiKey).filter_by(key_hash=hash_api_key(raw), revoked=False).first()
    if not key:
        raise HTTPException(401, "Invalid or revoked API key")
    key.last_used_at = datetime.now(timezone.utc)
    db.commit()
    return db.get(models.Org, key.org_id), key


class GenerateIn(BaseModel):
    prompt: str = ""
    max_new_tokens: int = 80
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 1.0


@router.post("/{model_name}/generate")
def generate(model_name: str, body: GenerateIn, ctx=Depends(api_key_org), db: Session = Depends(get_db)):
    org, key = ctx
    row = db.query(models.Model).filter_by(org_id=org.id, name=model_name).first()
    if not row:
        raise HTTPException(404, f"'{model_name}' is not a model in your workspace")
    if not row.deployed:
        raise HTTPException(409, f"'{model_name}' is not deployed — deploy it from your dashboard first")
    if row.kind in ("asr", "vlm", "tts"):
        raise HTTPException(501, f"hosted inference for {row.modality} models is coming soon")

    gen, _ = _load(model_name)
    t0 = time.time()
    text = gen(body.prompt, {"max_new_tokens": body.max_new_tokens, "temperature": body.temperature,
                             "top_k": body.top_k, "top_p": body.top_p})
    dt_ms = round((time.time() - t0) * 1000)

    db.add(models.UsageEvent(org_id=org.id, api_key_id=key.id, kind="infer", units=1.0,
                             model_id=row.id, meta={"ms": dt_ms}))
    db.commit()
    return {"model": model_name, "prompt": body.prompt, "text": text, "latency_ms": dt_ms}
