"""Profile and settings schemas for the local owner (``/users/me``).

Account auth is gone, so there is no register/login/OTP/OAuth request shape
left: what the API exposes about the user is the profile read and the opt-in
retention window it writes back.
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class UserResponse(BaseModel):
    id:              UUID
    email:           str
    display_name:    str | None
    avatar_url:      str | None
    auth_provider:   str
    role:            str
    is_active:       bool = True
    is_deleted:      bool = False
    is_verified:     bool
    onboarding_done: bool
    # Opt-in retention (P4b/T6): read back on GET /me, written by
    # PATCH /users/me/settings. Defaults mirror the model's schema default —
    # OFF until the user turns it on (spec §8.1).
    retention_enabled: bool = False
    retention_days:    int | None = None
    created_at:      datetime

    model_config = ConfigDict(from_attributes=True)


class RetentionSettingsRequest(BaseModel):
    """PATCH /users/me/settings body: the opt-in retention window (P4b/T6).

    Enabling retention requires a window: an enabled user with no
    ``retention_days`` is a setting that says nothing about WHEN to expire, so
    the service would never run for them — refusing at the boundary is the
    honest version of that (instead of accepting a config that silently does
    nothing). Disabled is always allowed, window or not.
    """

    retention_enabled: bool = False
    retention_days:    int | None = Field(default=None, gt=0, le=36_500)

    @model_validator(mode="after")
    def _enabling_needs_a_window(self) -> RetentionSettingsRequest:
        if self.retention_enabled and self.retention_days is None:
            raise ValueError("retention_days is required to enable retention.")
        return self


class RetentionSettingsResponse(BaseModel):
    retention_enabled: bool
    retention_days:    int | None
