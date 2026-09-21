"""Request identity: the REST API serves the local owner, and no one else.

There is no account auth to run: a self-hosted install is one loopback-bound
operator, so a request with **no** Authorization header IS the local owner
(:mod:`app.services.local_owner`) and must NOT 401 — that is the whole point.

An Authorization header here is NOT a credential. Agent tokens belong to the
MCP surface and to ``POST /api/v1/imports``, which resolve them themselves and
enforce their scopes (:mod:`app.mcp_hub.identity`, ``app/api/v1/imports.py``).
Letting ``get_current_user`` answer with the token's *owner* would hand a
``memory:read`` client the entire REST surface — including minting itself a
write token, or revoking a sibling's — and answering with the owner for a
revoked token would make revocation meaningless. Both fail closed.
"""
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.user import User
from app.services.local_owner import ensure_local_owner

# auto_error=False is load-bearing: a missing header is not an error any more.
bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """The local owner — the REST surface's only identity.

    Any Bearer value is refused: an agent token belongs to the MCP endpoint and
    ``POST /api/v1/imports`` (where its scopes are enforced), and account JWTs
    no longer exist, so a client sending one is stale config that should hear
    about it rather than silently act as the owner.
    """
    if credentials is not None and (credentials.credentials or "").strip():
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail=(
                "The REST API serves the local owner; agent tokens are used with "
                "the MCP endpoint and POST /api/v1/imports."
            ),
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await ensure_local_owner(db)


async def enforce_llm_quota(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Rate limit + quota guard for endpoints that trigger LLM/embedding spend.

    Without this, anything could loop /memories/recall and run up unbounded
    LLM cost — the quota system was only enforced on the chat path. The owner
    is the only principal, so the guard is keyed on it.
    """
    from app.middleware.rate_limiter import check_rate_limit
    from app.services.quota_service import check_and_increment

    await check_rate_limit(
        str(current_user.id), window_seconds=60, limit=getattr(settings, "RATE_LIMIT_PER_MINUTE", 60)
    )
    await check_and_increment(current_user.id, db)
