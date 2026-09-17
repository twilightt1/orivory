"""User profile endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.auth import (
    ChangePasswordRequest,
    ChangePasswordResponse,
    RetentionSettingsRequest,
    RetentionSettingsResponse,
    UpdateProfileRequest,
    UserResponse,
)
from app.services import auth_service
from app.utils.dependencies import get_current_verified_user

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserResponse)
async def get_me(current_user=Depends(get_current_verified_user)):
    return UserResponse.model_validate(current_user)


@router.patch("/me", response_model=UserResponse)
async def update_profile(
    body: UpdateProfileRequest,
    current_user=Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db),
):
    user = await auth_service.update_display_name(db, current_user, body.display_name)
    return UserResponse.model_validate(user)


@router.patch("/me/settings", response_model=RetentionSettingsResponse)
async def update_retention_settings(
    body: RetentionSettingsRequest,
    current_user=Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db),
):
    """Turn opt-in retention on/off and choose the window (P4b/T6, spec §8.1).

    A dedicated endpoint rather than a wider ``PATCH /me``: the retention pair
    is a settings write with its own validation (enabled requires a window),
    while ``PATCH /me`` stays the profile write it has always been. The read
    side is ``GET /me`` — ``UserResponse`` carries both fields.
    """
    user = await auth_service.set_retention_settings(
        db, current_user, enabled=body.retention_enabled, days=body.retention_days)
    return RetentionSettingsResponse(
        retention_enabled=user.retention_enabled,
        retention_days=user.retention_days,
    )


@router.post("/me/change-password", response_model=ChangePasswordResponse)
async def change_password(
    body: ChangePasswordRequest,
    current_user=Depends(get_current_verified_user),
    db: AsyncSession = Depends(get_db),
):
    await auth_service.change_password(db, current_user, body.current_password, body.new_password)
    return ChangePasswordResponse(message="Password changed. Please log in again.")
