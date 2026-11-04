import hmac

from fastapi import Header, HTTPException

from app.core import config


async def require_internal_secret(x_internal_secret: str | None = Header(default=None)) -> None:
    if not config.is_production():
        return
    secret = config.internal_secret()
    if not secret or not x_internal_secret:
        raise HTTPException(status_code=401)
    if not hmac.compare_digest(secret.encode(), x_internal_secret.encode()):
        raise HTTPException(status_code=401)
