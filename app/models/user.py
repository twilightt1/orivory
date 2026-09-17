import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import TIMESTAMP, Boolean, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models._datetime_helpers import utc_now
from app.models.types import GUID

if TYPE_CHECKING:
    from app.models.conversation import Conversation
    from app.models.email_verification import EmailVerification
    from app.models.entity import Entity
    from app.models.memory import Memory
    from app.models.password_reset_session import PasswordResetSession
    from app.models.source import Source
    from app.models.user_quota import UserQuota


class User(Base):
    __tablename__ = "users"

    id:              Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    email:           Mapped[str]       = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str|None]  = mapped_column(String(255), nullable=True)
    display_name:    Mapped[str|None]  = mapped_column(String(100), nullable=True)
    auth_provider:   Mapped[str]       = mapped_column(String(20), server_default="email")
    google_id:       Mapped[str|None]  = mapped_column(String(128), unique=True, nullable=True, index=True)
    avatar_url:      Mapped[str|None]  = mapped_column(String(500), nullable=True)
    onboarding_done: Mapped[bool]      = mapped_column(Boolean(), default=False)
    role:            Mapped[str]       = mapped_column(String(20), server_default="user")
    is_verified:     Mapped[bool]      = mapped_column(Boolean(), default=False)
    is_active:       Mapped[bool]      = mapped_column(Boolean(), default=True)
    is_deleted:      Mapped[bool]      = mapped_column(Boolean(), default=False)
    # Opt-in retention (P4b/T6, spec §8.1): auto expiration is OFF until the
    # user turns it on — and only then does ``run_retention`` expire memories
    # the system has held for more than ``retention_days``. ``server_default``
    # is deliberate: the ladder's ADD COLUMN spells the same default, so every
    # pre-existing user is OFF without a backfill (the column IS the backfill).
    # ``retention_days`` NULL means "no window chosen" — an enabled user
    # without one is a setting that says nothing, so nothing runs.
    retention_enabled: Mapped[bool]     = mapped_column(Boolean(), nullable=False, server_default="0", default=False)
    retention_days:    Mapped[int | None] = mapped_column(Integer(), nullable=True)
    created_at:      Mapped[datetime]  = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at:      Mapped[datetime]  = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=utc_now)

    email_verifications:     Mapped[list["EmailVerification"]]     = relationship(back_populates="user", cascade="all, delete-orphan")
    password_reset_sessions: Mapped[list["PasswordResetSession"]]  = relationship(back_populates="user", cascade="all, delete-orphan")
    conversations:           Mapped[list["Conversation"]]          = relationship(back_populates="user", cascade="all, delete-orphan")
    quota:                   Mapped["UserQuota"]                   = relationship(back_populates="user", uselist=False, cascade="all, delete-orphan")
    memories:                Mapped[list["Memory"]]                = relationship(back_populates="user", cascade="all, delete-orphan")
    entities:                Mapped[list["Entity"]]                = relationship(back_populates="user", cascade="all, delete-orphan")
    sources:                 Mapped[list["Source"]]                = relationship(back_populates="user", cascade="all, delete-orphan")
