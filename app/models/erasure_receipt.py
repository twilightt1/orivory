"""Erasure receipts — verifiable proof of what a memory erasure removed.

One row per erasure call (MCP ``forget_memory`` or the REST erasure API).
``detail`` carries the per-target results: which rows/links/vectors were
deleted and what the post-deletion verification pass (re-query Qdrant +
residual DB row counts) found. Unlike the append-only access ledger,
receipts carry a FK to ``users`` with CASCADE: they quote personal memory
ids, so they must not outlive the user (open-memory-hub.md MVP item 5).
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, TIMESTAMP, ForeignKey, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.types import GUID


class ErasureReceipt(Base):
    __tablename__ = "erasure_receipts"

    id:                   Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id:              Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    requested_memory_ids: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    # String(32): the honest statuses ("completed_with_residual" = 23 chars)
    # do not fit String(16) — the INSERT failed exactly when the receipt mattered.
    status:               Mapped[str]       = mapped_column(String(32), default="completed", server_default="completed", nullable=False)
    detail:               Mapped[dict]      = mapped_column(JSON, default=dict, nullable=False)
    created_at:           Mapped[datetime]  = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_erasure_receipts_user_time", "user_id", "created_at"),
    )

    def __init__(self, **kwargs: Any) -> None:
        # SQLAlchemy applies column defaults at flush time only; populate the
        # defaults here as well so transient instances (and unit tests) see them.
        kwargs.setdefault("requested_memory_ids", [])
        kwargs.setdefault("status", "completed")
        kwargs.setdefault("detail", {})
        super().__init__(**kwargs)


__all__ = ["ErasureReceipt"]
