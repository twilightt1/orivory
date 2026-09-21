"""The local owner — the one human identity of a self-hosted install.

Account authentication is gone: register / email verification / login / OAuth /
password reset / JWT all protected nothing on a single-operator deployment
whose API binds loopback. What replaces them is not a new auth scheme but the
absence of one — every local request IS the owner.

The owner is still a real ``users`` row, because every per-user path (agent
token ownership, the access ledger, quota, retention, erasure) is keyed on
``user_id``. ``ensure_local_owner`` get-or-creates that row on
``settings.LOCAL_OWNER_EMAIL`` and leaves an existing one exactly as it is: a
row that already exists is an operator's data, not a default to rewrite.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.user import User


def local_owner_email() -> str:
    """The owner's key, normalized the way the API normalizes addresses."""
    return settings.LOCAL_OWNER_EMAIL.strip().lower()


async def ensure_local_owner(db: AsyncSession) -> User:
    """Get-or-create THE local owner row (verified, onboarded, admin, active).

    No password path exists any more, so ``hashed_password`` stays NULL — the
    column is nullable and nothing reads it.
    """
    email = local_owner_email()
    existing = (
        await db.execute(select(User).where(User.email == email))
    ).scalars().first()
    if existing is not None:
        return existing

    user = User(
        email=email,
        display_name="Owner",
        auth_provider="local",
        onboarding_done=True,
        role="admin",
        is_verified=True,
        is_active=True,
    )
    db.add(user)
    try:
        await db.commit()
    except IntegrityError:
        # Two concurrent first requests both found no owner; the unique email
        # index picked a winner. Read theirs instead of 500ing the loser.
        await db.rollback()
        winner = (
            await db.execute(select(User).where(User.email == email))
        ).scalars().first()
        if winner is None:
            raise
        return winner
    await db.refresh(user)
    return user


__all__ = ["ensure_local_owner", "local_owner_email"]
