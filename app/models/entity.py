"""
Entity / Relation models for Orivory's knowledge graph.

An "entity" is a person, project, topic, concept, or date that
appears across the user's memories. Examples: "Mom", "Project Atlas",
"Python asyncio", "2026-05-12".

A "relation" is a typed edge between two entities (e.g. "Mom — likes
— Project Atlas", weight 0.7). Relations are derived from memory
co-occurrences and can be updated as new evidence arrives.

`MemoryEntity` is the many-to-many join between memories and entities,
with a `salience` score capturing how central the entity is to that
memory. This is the table the agent uses to look up all memories
about a given entity.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    JSON,
    TIMESTAMP,
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
    from app.models.memory import Memory
    from app.models.user import User


# Allowed entity_type values. Kept as constants so the agent picks from a
# known vocabulary instead of inventing one.
ENTITY_TYPES = (
    "person",
    "project",
    "topic",
    "concept",
    "organization",
    "place",
    "date",
    "event",
    "media",   # book, movie, podcast, etc.
    "other",
)

RELATION_TYPES = (
    "related_to",
    "works_on",
    "knows",
    "owns",
    "part_of",
    "mentioned_in",
    "references",
    "contradicts",
    "follows",
    "precedes",
    "summarizes",
)


class Entity(Base):
    __tablename__ = "entities"

    id:            Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id:       Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name:          Mapped[str]        = mapped_column(String(255), nullable=False)
    entity_type:   Mapped[str]        = mapped_column(String(32),  nullable=False, server_default="other")
    aliases:       Mapped[list[str]]  = mapped_column(JSON, nullable=False)
    description:   Mapped[str | None] = mapped_column(Text, nullable=True)

    first_seen_at: Mapped[datetime]   = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    last_seen_at:  Mapped[datetime]   = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    mention_count: Mapped[int]        = mapped_column(Integer(), default=0, nullable=False)

    extra_metadata: Mapped[dict]      = mapped_column("metadata", JSON, server_default="{}", nullable=False)
    created_at:    Mapped[datetime]   = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at:    Mapped[datetime]   = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=utc_now, nullable=False)

    user:           Mapped[User]                  = relationship(back_populates="entities")
    memory_links:   Mapped[list[MemoryEntity]]    = relationship(back_populates="entity", cascade="all, delete-orphan")

    # Source relations (this entity → other entities)
    outgoing_relations: Mapped[list[Relation]] = relationship(
        "Relation",
        foreign_keys="Relation.source_entity_id",
        back_populates="source",
        cascade="all, delete-orphan",
    )
    incoming_relations: Mapped[list[Relation]] = relationship(
        "Relation",
        foreign_keys="Relation.target_entity_id",
        back_populates="target",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("user_id", "name", "entity_type", name="uq_entities_user_name_type"),
        Index("ix_entities_user_type", "user_id", "entity_type"),
        Index("ix_entities_user_last_seen", "user_id", "last_seen_at"),
    )


class Relation(Base):
    __tablename__ = "relations"

    id:                Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id:           Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    source_entity_id:  Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True)
    target_entity_id:  Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True)
    relation:          Mapped[str]       = mapped_column(String(64), nullable=False, server_default="related_to")
    weight:            Mapped[float]     = mapped_column(Float, server_default="0.5", nullable=False)
    evidence_count:    Mapped[int]       = mapped_column(Integer(), default=1, nullable=False)
    last_evidence_at:  Mapped[datetime]  = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    extra_metadata:    Mapped[dict]      = mapped_column("metadata", JSON, server_default="{}", nullable=False)
    created_at:        Mapped[datetime]  = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at:        Mapped[datetime]  = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=utc_now, nullable=False)

    source: Mapped[Entity] = relationship("Entity", foreign_keys=[source_entity_id], back_populates="outgoing_relations")
    target: Mapped[Entity] = relationship("Entity", foreign_keys=[target_entity_id], back_populates="incoming_relations")

    __table_args__ = (
        UniqueConstraint("user_id", "source_entity_id", "target_entity_id", "relation", name="uq_relations_user_pair_type"),
        Index("ix_relations_user_weight", "user_id", "weight"),
    )


class MemoryEntity(Base):
    """Many-to-many link between a memory and an entity, with salience."""

    __tablename__ = "memory_entities"

    id:        Mapped[uuid.UUID] = mapped_column(GUID(), primary_key=True, default=uuid.uuid4)
    memory_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("memories.id", ondelete="CASCADE"), nullable=False, index=True)
    entity_id: Mapped[uuid.UUID] = mapped_column(GUID(), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True)
    salience:  Mapped[float]     = mapped_column(Float, server_default="0.5", nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    memory: Mapped[Memory] = relationship(back_populates="entity_links")
    entity: Mapped[Entity] = relationship(back_populates="memory_links")

    __table_args__ = (
        UniqueConstraint("memory_id", "entity_id", name="uq_memory_entity"),
    )


__all__ = ["Entity", "Relation", "MemoryEntity", "ENTITY_TYPES", "RELATION_TYPES"]
