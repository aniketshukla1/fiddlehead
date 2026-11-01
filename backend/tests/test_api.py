"""API smoke test — the contract the web app relies on, against a throwaway SQLite database.

    cd backend && python -m pytest -q
"""
import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_tmp, "test.db")
os.environ["FIDDLEHEAD_SAVE_DIR"] = os.path.join(_tmp, "saved")
os.environ["JWT_SECRET"] = "test-secret"

from fastapi.testclient import TestClient  # noqa: E402

from api.app import app  # noqa: E402

client = TestClient(app)


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def test_health_and_public_meta():
    assert client.get("/health").json()["ok"] is True
    meta = client.get("/meta").json()
    assert meta["billing"] is False          # self-hosted default: no plans, no credit limits
    assert meta["oauth"] == []


def test_account_models_keys_and_inference_auth():
    r = client.post("/auth/signup", json={"name": "Ada Lovelace", "email": "ada@example.com", "password": "correct-horse-9"})
    assert r.status_code == 200, r.text
    token = r.json()["token"]
    assert client.get("/auth/me", headers=bearer(token)).json()["user"]["email"] == "ada@example.com"
    assert client.get("/models").status_code == 401                       # no token, no data

    m = client.post("/models", json={"name": "sonnets", "params_m": 0.8}, headers=bearer(token)).json()
    assert m["deployed"] is False
    assert client.post(f"/models/{m['id']}/deploy", headers=bearer(token)).json()["deployed"] is True
    assert [x["name"] for x in client.get("/models", headers=bearer(token)).json()] == ["sonnets"]

    key = client.post("/keys", json={"name": "ci"}, headers=bearer(token)).json()["key"]
    assert key.startswith("sk-")
    assert client.post("/v1/sonnets/generate", json={"prompt": "hi"}).status_code == 401
    # authenticated and deployed, but no weights saved in this throwaway workspace → 404
    assert client.post("/v1/sonnets/generate", json={"prompt": "hi"}, headers=bearer(key)).status_code == 404


def test_parallelism_planner():
    small = client.post("/compute/plan", json={"params_m": 20, "gpu": "a100-80", "gpus_per_node": 1}).json()
    big = client.post("/compute/plan", json={"params_m": 70000, "gpu": "a100-80", "gpus_per_node": 8}).json()
    assert small["strategy"] == "single" and small["fits"]
    assert big["strategy"] in ("fsdp", "tp+pp")
