"""memories.namespace — the P4a authorization boundary (personal-only)

P4a makes namespace a real column: a memory belongs to exactly one namespace,
and namespace — not ``cm_scope``, not a tag, not a workspace name — is what an
authorization predicate filters on. This phase ships the column with a constant
default and no way for a client to set it (team sharing is deferred), so every
existing row is backfilled ``personal`` by the server default and a deployment
with ONE namespace answers exactly as it did before the column existed.

The SQLite ladder's v4 -> v5 step (``app.database._upgrade_v4_to_v5``) mirrors
this revision: the same column, the same NOT NULL, the same ``'personal'``
default, the same ``(namespace, user_id)`` index — so a Postgres database and a
SQLite file upgraded in place end up with the same shape.

Revision ID: c2d3e4f5a6b7
Revises: b1c2d3e4f5a6
Create Date: 2026-09-16 00:00:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c2d3e4f5a6b7"
down_revision: str | None = "b1c2d3e4f5a6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "memories",
        sa.Column("namespace", sa.String(32), nullable=False, server_default="personal"),
    )
    op.create_index("ix_memories_namespace_user", "memories", ["namespace", "user_id"])


def downgrade() -> None:
    op.drop_index("ix_memories_namespace_user", table_name="memories")
    op.drop_column("memories", "namespace")
