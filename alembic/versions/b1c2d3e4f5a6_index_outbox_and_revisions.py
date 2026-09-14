"""schema v2: revision counters + durable index-intent tables

P1a makes SQLite the canonical store with revisions and a durable
index-intent outbox before the Qdrant cutover (P1b). This is the Postgres
mirror of the SQLite v1 -> v2 ladder in ``app/database.py``:

- ``memories.revision`` / ``document_chunks.revision``: monotonic per-entity
  write counter driving index-intent idempotency (never ``updated_at``).
- ``index_outbox``: one row per index mutation intent, written in the SAME
  commit as the canonical write; unique per
  (kind, entity_id, revision, target_generation, operation).
- ``index_generations``: active physical collection name + embedding
  fingerprint per kind (the cutover in P1b becomes a pointer swap).
- ``memory_suppressions``: ledger blocking reingest from resurrecting a
  forgotten source-derived identity (spec §5.4/§12.3).

tests/migrations diffs this chain against Base.metadata — keep the columns
typed exactly like the models.

Revision ID: b1c2d3e4f5a6
Revises: a9b8c7d6e5f4
Create Date: 2026-09-14 00:00:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b1c2d3e4f5a6"
down_revision: str | None = "a9b8c7d6e5f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("memories", sa.Column("revision", sa.Integer(), nullable=False, server_default="1"))
    op.add_column("document_chunks", sa.Column("revision", sa.Integer(), nullable=False, server_default="1"))

    op.create_table(
        "index_outbox",
        sa.Column("seq", sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
                  primary_key=True, autoincrement=True),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("entity_id", sa.String(32), nullable=False),
        sa.Column("tenant_id", sa.String(32), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("operation", sa.String(8), nullable=False),
        sa.Column("target_generation", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("kind", "entity_id", "revision", "target_generation", "operation",
                            name="uq_index_outbox_intent"),
    )
    op.create_index("ix_index_outbox_pending", "index_outbox", ["status", "next_attempt_at"])

    op.create_table(
        "index_generations",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("generation", sa.String(64), nullable=False),
        sa.Column("fingerprint", sa.String(128), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("kind", "generation", name="uq_index_generation"),
    )
    op.create_index("ix_index_generation_active", "index_generations", ["kind", "is_active"])

    op.create_table(
        "memory_suppressions",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_ref", sa.String(500), nullable=False),
        sa.Column("reason", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("user_id", "source_ref", name="uq_memory_suppression"),
    )


def downgrade() -> None:
    op.drop_table("memory_suppressions")
    op.drop_index("ix_index_generation_active", table_name="index_generations")
    op.drop_table("index_generations")
    op.drop_index("ix_index_outbox_pending", table_name="index_outbox")
    op.drop_table("index_outbox")
    op.drop_column("document_chunks", "revision")
    op.drop_column("memories", "revision")
