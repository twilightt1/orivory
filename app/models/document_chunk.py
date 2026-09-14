import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, TIMESTAMP, ForeignKey, Integer, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.types import GUID

if TYPE_CHECKING:
    from app.models.document import Document


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id:          Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    content:     Mapped[str]       = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int]       = mapped_column(Integer(), nullable=False)
    # Monotonic per-chunk write counter for index-intent idempotency (spec §4.1).
    revision:    Mapped[int]       = mapped_column(Integer(), default=1, server_default="1", nullable=False)
    chunk_metadata: Mapped[dict]      = mapped_column(JSON, server_default="{}")
    created_at:  Mapped[datetime]  = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

    document: Mapped["Document"] = relationship(back_populates="chunks")
