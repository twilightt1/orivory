from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    field_validator,
    model_validator,
)


class _EmailNormalizingModel(BaseModel):
    """Base for requests carrying an ``email`` field.

    ``EmailStr`` only normalizes the domain part, so ``User@x.com`` and
    ``user@x.com`` would be treated as different accounts (the users table has
    a unique index on the raw string). We lowercase the whole address at the
    boundary so storage and lookups are consistent everywhere.
    """

    @field_validator("email", check_fields=False)
    @classmethod
    def _normalize_email(cls, v: str) -> str:
        return v.strip().lower() if isinstance(v, str) else v


class RegisterRequest(_EmailNormalizingModel):
    email:    EmailStr
    password: str = Field(min_length=8, max_length=128)


class RegisterResponse(BaseModel):
    message: str


class OTPVerifyRequest(_EmailNormalizingModel):
    email:    EmailStr
    otp_code: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")


class OTPVerifyResponse(BaseModel):
    message:      str
    access_token: str
    next:         str = "onboarding"


class ResendVerificationRequest(_EmailNormalizingModel):
    email: EmailStr


class OnboardingRequest(BaseModel):
    display_name: str = Field(min_length=2, max_length=50)


class OnboardingResponse(BaseModel):
    access_token:  str
    # httpOnly cookie carries the refresh token for browser clients.
    refresh_token: str | None = None
    user:          UserResponse


class LoginRequest(_EmailNormalizingModel):
    email:    EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token:  str
    # Kept for backward compatibility with non-cookie clients; browser
    # clients receive the refresh token in an httpOnly cookie instead.
    refresh_token: str | None = None
    token_type:    str = "bearer"
    user:          UserResponse


class ForgotPasswordRequest(_EmailNormalizingModel):
    email: EmailStr


class ForgotPasswordResponse(BaseModel):
    message: str


class ForgotPasswordOTPVerifyRequest(_EmailNormalizingModel):
    email:    EmailStr
    otp_code: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")


class ForgotPasswordOTPVerifyResponse(BaseModel):
    reset_token: str
    message:     str


class ResetPasswordRequest(BaseModel):
    token:        str
    new_password: str = Field(min_length=8, max_length=128)


class ResetPasswordResponse(BaseModel):
    message: str


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


class UpdateProfileRequest(BaseModel):
    display_name: str = Field(min_length=2, max_length=50)


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


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1)
    new_password:     str = Field(min_length=8, max_length=128)

    @model_validator(mode="after")
    def passwords_differ(self) -> ChangePasswordRequest:
        if self.current_password == self.new_password:
            raise ValueError("New password must differ from current password.")
        return self


class ChangePasswordResponse(BaseModel):
    message: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str = Field(min_length=32)


class LogoutRequest(BaseModel):
    refresh_token: str | None = Field(default=None, min_length=32)


class AuthRedirectExchangeRequest(BaseModel):
    code: str = Field(min_length=32)
