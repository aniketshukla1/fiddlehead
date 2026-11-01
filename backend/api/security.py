"""Password hashing (bcrypt), JWT session tokens, and API-key generation."""
import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt

from .config import settings


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def verify_password(pw: str, h: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), h.encode())
    except Exception:
        return False


def create_token(sub: str, jti: str | None = None, minutes: int | None = None,
                 purpose: str = "session") -> str:
    """Issue a JWT. `jti` binds it to a Session row so it can be revoked."""
    exp = datetime.now(timezone.utc) + timedelta(minutes=minutes or settings.JWT_EXPIRE_MIN)
    claims = {"sub": sub, "exp": exp, "purpose": purpose}
    if jti:
        claims["jti"] = jti
    return jwt.encode(claims, settings.JWT_SECRET, algorithm=settings.JWT_ALG)


def decode_token(token: str) -> str | None:
    """Back-compat: just the subject."""
    c = decode_claims(token)
    return c.get("sub") if c else None


def decode_claims(token: str, purpose: str = "session") -> dict | None:
    try:
        c = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALG])
    except JWTError:
        return None
    if c.get("purpose", "session") != purpose:
        return None
    return c


# ---- two-factor auth: TOTP (RFC 6238) on the standard library --------- #
_B32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"


def new_totp_secret(length: int = 32) -> str:
    return "".join(secrets.choice(_B32) for _ in range(length))


def _b32decode(secret: str) -> bytes:
    import base64
    pad = "=" * (-len(secret) % 8)
    return base64.b32decode(secret.upper() + pad)


def totp_at(secret: str, counter: int, digits: int = 6) -> str:
    import hmac
    import struct
    mac = hmac.new(_b32decode(secret), struct.pack(">Q", counter), hashlib.sha1).digest()
    off = mac[-1] & 0x0F
    code = (struct.unpack(">I", mac[off:off + 4])[0] & 0x7FFFFFFF) % (10 ** digits)
    return str(code).zfill(digits)


def verify_totp(secret: str, code: str, window: int = 1, step: int = 30) -> bool:
    """Accept the current code plus one step either side (clock drift)."""
    code = (code or "").strip().replace(" ", "")
    if not code.isdigit():
        return False
    counter = int(datetime.now(timezone.utc).timestamp()) // step
    return any(secrets.compare_digest(totp_at(secret, counter + d), code)
               for d in range(-window, window + 1))


def totp_uri(secret: str, email: str, issuer: str = "Fiddlehead") -> str:
    from urllib.parse import quote
    return (f"otpauth://totp/{quote(issuer)}:{quote(email)}"
            f"?secret={secret}&issuer={quote(issuer)}&algorithm=SHA1&digits=6&period=30")


def new_recovery_codes(n: int = 8) -> tuple[list[str], list[str]]:
    """Return (plain_codes_shown_once, sha256_hashes_to_store)."""
    plain = ["-".join(secrets.token_hex(2) for _ in range(2)) for _ in range(n)]
    return plain, [hashlib.sha256(c.encode()).hexdigest() for c in plain]


def hash_recovery_code(code: str) -> str:
    return hashlib.sha256(code.strip().lower().encode()).hexdigest()


def new_api_key() -> tuple[str, str, str]:
    """Return (full_key_shown_once, prefix, sha256_hash). Full key is never stored."""
    raw = "sk-" + secrets.token_urlsafe(32)
    return raw, raw[:11], hashlib.sha256(raw.encode()).hexdigest()


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()
