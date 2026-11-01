"""Sliding-window rate limiting for abuse-prone endpoints.

In-process by design: correct and dependency-free for a single API process.
Behind multiple workers or hosts, swap `_store` for Redis — the surface is just
`hit()` / `reset()`.
"""
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException

_store: dict[str, deque] = defaultdict(deque)
_lock = Lock()

# (attempts, seconds) per bucket — tuned for humans, hostile to scripts
LIMITS = {
    "login": (8, 300),          # 8 tries / 5 min per email+IP
    "login_ip": (30, 300),      # 30 tries / 5 min from one IP (credential stuffing)
    "signup": (5, 3600),        # 5 accounts / hour per IP
    "mfa": (10, 300),           # 10 codes / 5 min
    "password": (5, 900),       # 5 password changes / 15 min
    "verify_send": (4, 3600),   # 4 verification emails / hour
    "invite": (30, 3600),       # 30 invites / hour per workspace
    "oauth": (20, 600),
}


def hit(bucket: str, key: str) -> None:
    """Count one attempt; raise 429 (with Retry-After) once over the limit."""
    times, seconds = LIMITS.get(bucket, (20, 300))
    now = time.time()
    k = f"{bucket}:{key}"
    with _lock:
        dq = _store[k]
        while dq and now - dq[0] > seconds:
            dq.popleft()
        if len(dq) >= times:
            retry = max(1, int(seconds - (now - dq[0])) + 1)
            raise HTTPException(
                429,
                f"Too many attempts. Try again in {retry} second{'s' if retry != 1 else ''}.",
                headers={"Retry-After": str(retry)},
            )
        dq.append(now)


def reset(bucket: str, key: str) -> None:
    """Clear a counter — call after a success so one good login wipes the streak."""
    with _lock:
        _store.pop(f"{bucket}:{key}", None)


def remaining(bucket: str, key: str) -> int:
    times, seconds = LIMITS.get(bucket, (20, 300))
    now = time.time()
    with _lock:
        dq = _store.get(f"{bucket}:{key}", deque())
        fresh = sum(1 for t in dq if now - t <= seconds)
    return max(0, times - fresh)
