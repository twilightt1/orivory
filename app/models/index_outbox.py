"""Durable index intent + index generation manifest (spec §4.1, §5.1).

``index_outbox`` records every index mutation intent in the SAME SQL commit
as the canonical write, so a crash between SQL and the vector backend is
recoverable (replay, never lost). ``index_generations`` names the active
physical collection per kind, so a cutover is a pointer swap.

P1a keeps Chroma as the vector backend: nothing here talks to a vector store.
"""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class IndexOutbox(Base):
    __tablename__ = "index_outbox"
    __table_args__ = (
        UniqueConstraint("kind", "entity_id", "revision", "target_generation", "operation",
                         name="uq_index_outbox_intent"),
        Index("ix_index_outbox_pending", "status", "next_attempt_at"),
    )

    seq: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"),
                                     primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(16))            # memory | chunk (chunk: P1b)
    entity_id: Mapped[str] = mapped_column(String(32))       # UUID string; no FK (target may be deleted)
    tenant_id: Mapped[str] = mapped_column(String(32))       # user_id
    revision: Mapped[int] = mapped_column(Integer)
    operation: Mapped[str] = mapped_column(String(8))        # upsert | delete
    target_generation: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|done|blocked
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)  # sanitized only
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC))


class IndexGeneration(Base):
    __tablename__ = "index_generations"
    __table_args__ = (
        UniqueConstraint("kind", "generation", name="uq_index_generation"),
        Index("ix_index_generation_active", "kind", "is_active"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)   # uuid4 hex
    kind: Mapped[str] = mapped_column(String(16))
    generation: Mapped[str] = mapped_column(String(64))             # physical collection name
    fingerprint: Mapped[str] = mapped_column(String(128))  # fingerprint_generation() token, 64 hex
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC))


__all__ = ["IndexGeneration", "IndexOutbox"]
