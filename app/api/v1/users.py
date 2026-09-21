"""User profile endpoints — the local owner's settings surface.

Account auth is gone, so the addressed user is always the local owner and
there is no token to present. What stays is real product surface: the profile
read (``GET /me``, which carries the opt-in retention pair) and the retention
settings write. ``PATCH /me`` and ``POST /me/change-password`` went with the
account surface.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.users import (
    RetentionSettingsRequest,
    RetentionSettingsResponse,
    UserResponse,
)
from app.utils.dependencies import get_current_user

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserResponse)
async def get_me(current_user=Depends(get_current_user)):
    return UserResponse.model_validate(current_user)


@router.patch("/me/settings", response_model=RetentionSettingsResponse)
async def update_retention_settings(
    body: RetentionSettingsRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Turn opt-in retention on/off and choose the window (P4b/T6, spec §8.1).

    A dedicated endpoint rather than a wider ``PATCH /me``: the retention pair
    is a settings write with its own validation (enabled requires a window).
    The read side is ``GET /me`` — ``UserResponse`` carries both fields.
    """
    enabled, days = body.retention_enabled, body.retention_days
    current_user.retention_enabled = enabled
    current_user.retention_days = days
    await db.commit()
    return RetentionSettingsResponse(retention_enabled=enabled, retention_days=days)
