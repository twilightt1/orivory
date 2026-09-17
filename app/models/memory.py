"""
Memory model for Orivory.

A "memory" is a single piece of knowledge captured into the user's
second brain. Memories can come from many sources (file upload, web
clip, Gmail, Google Drive, manual note) and live independently of any
single conversation.

Design notes:
    - `source_type` describes the origin family (file, drive, notion,
      gmail, web_clip, manual_note, conversation_excerpt).
    - `parent_id` allows a memory to be a sub-chunk of a larger memory
      (e.g. an extracted passage from a document).
    - `salience` is a float in [0, 1] that the system can update over
      time based on usage / recency. Used for ranking.
    - `captured_at` is the original event time (e.g. the email date,
      the file mtime). `indexed_at` is when Orivory first stored it.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    JSON,
    TIMESTAMP,
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models._datetime_helpers import utc_now
from app.models.types import GUID

if TYPE_CHECKING:
    from app.models.entity import MemoryEntity
    from app.models.source import MemorySource
    from app.models.user import User


class Memory(Base):
    __tablename__ = "memories"

    id:            Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id:       Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    parent_id:     Mapped[uuid.UUID | None] = mapped_column(GUID(), ForeignKey("memories.id", ondelete="CASCADE"), nullable=True, index=True)

    # Origin description
    source_type:   Mapped[str]       = mapped_column(String(32), nullable=False, server_default="manual_note")
    source_ref:    Mapped[str | None] = mapped_column(String(500), nullable=True)  # e.g. drive file id, url, message id
    source_url:    Mapped[str | None] = mapped_column(String(1000), nullable=True)

    # Content
    title:         Mapped[str | None] = mapped_column(String(500), nullable=True)
    content:       Mapped[str]        = mapped_column(Text, nullable=False)
    summary:       Mapped[str | None] = mapped_column(Text, nullable=True)
    tags:          Mapped[list[str]]  = mapped_column(JSON, nullable=False)

    # Scoring
    salience:      Mapped[float]      = mapped_column(Float, server_default="0.5", nullable=False)
    pinned:        Mapped[bool]        = mapped_column(Boolean(), default=False, nullable=False)
    is_shared:     Mapped[bool]        = mapped_column(Boolean(), default=False, nullable=False)  # Public share
    # Monotonic per-entity write counter driving index-intent idempotency
    # (spec §4.1 — never `updated_at`).
    revision:      Mapped[int]        = mapped_column(Integer, default=1, server_default="1", nullable=False)

    # Authorization boundary (P4a): every pre-P4 row was backfilled to
    # 'personal' by the SQLite ladder's v4 -> v5 step (or by this column's
    # server default on a fresh install). Never set from client input in this
    # phase — the only value that exists is `namespaces.PERSONAL`. `String(32)`
    # is deliberate: the ladder's ADD COLUMN spells the same type
    # (`VARCHAR(32) NOT NULL DEFAULT 'personal'`), so an upgraded file and a
    # fresh one are indistinguishable.
    namespace:     Mapped[str]        = mapped_column(String(32), nullable=False, server_default="personal")

    # Usage feedback (P2.1): bumped when a memory is recalled & used in an
    # answer; decayed periodically when untouched. Drives the salience loop.
    recall_count:  Mapped[int]        = mapped_column(Integer(), default=0, nullable=False)
    last_used_at:  Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)

    # Time
    captured_at:   Mapped[datetime]   = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    indexed_at:    Mapped[datetime]   = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at:    Mapped[datetime]   = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=utc_now, nullable=False)

    # Free-form metadata
    extra_metadata: Mapped[dict]      = mapped_column("metadata", JSON, server_default="{}", nullable=False)

    user:    Mapped[User]              = relationship(back_populates="memories")
    parent:  Mapped[Memory | None]     = relationship("Memory", remote_side="Memory.id", back_populates="children")
    children: Mapped[list[Memory]]    = relationship("Memory", back_populates="parent", cascade="all, delete-orphan")
    entity_links: Mapped[list[MemoryEntity]] = relationship(back_populates="memory", cascade="all, delete-orphan")
    source_links: Mapped[list[MemorySource]] = relationship(back_populates="memory", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_memories_user_captured", "user_id", "captured_at"),
        Index("ix_memories_user_salience", "user_id", "salience"),
        Index("ix_memories_source", "user_id", "source_type"),
        Index("ix_memories_user_last_used", "user_id", "last_used_at"),
        # The namespace boundary is queried WITH the owner: `namespace = ? AND
        # user_id = ?` is the shape every reader is about to take (P4a Task 2+).
        Index("ix_memories_namespace_user", "namespace", "user_id"),
    )


class MemorySuppression(Base):
    """Suppression ledger: a forgotten source-derived identity (spec §5.4/§12.3).

    Reingesting the same source must not resurrect a memory the user chose to
    forget; the ledger records that decision per (user, source identity).

    ``namespace`` records the boundary the suppression was written in (R38);
    ``content_hash`` is the sha256 of the source's BYTES — computed at UPLOAD
    (``document_service.upload_document``), carried onto the projection rows at
    INGEST, and copied into this column only when a forget is recorded (soft
    forget and the hard erase both pass ``projection_content_hash(row)``). The
    import/reindex/drain guards only READ this ledger; they never mint a hash,
    and the column is never backfilled for rows that predate it. Both are
    nullable on purpose: a pre-P4b row has no such value, and a NULL is the
    honest "unknown" instead of an invented one.
    """
    __tablename__ = "memory_suppressions"
    __table_args__ = (
        UniqueConstraint("user_id", "source_ref", name="uq_memory_suppression"),
    )

    id:         Mapped[str]       = mapped_column(String(32), primary_key=True,
                                                  default=lambda: uuid.uuid4().hex)
    user_id:    Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("users.id", ondelete="CASCADE"),
                                                  nullable=False)
    source_ref: Mapped[str]       = mapped_column(String(500), nullable=False)
    reason:     Mapped[str]       = mapped_column(String(64), nullable=False)
    namespace:  Mapped[str | None] = mapped_column(String(32), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime]  = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(),
                                                  nullable=False)


__all__ = ["Memory", "MemorySuppression"]
