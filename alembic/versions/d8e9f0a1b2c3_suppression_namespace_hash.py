"""memory_suppressions.namespace + content_hash — the P4b soft-forget ledger

Soft forget (P4b/T3) keeps the row and pins the SOURCE: every affected
``source_ref`` gets a suppression row that the re-import/reindex/outbox guards
(T4) read before writing. Two nullable columns make that ledger answerable:

- ``namespace`` — the boundary the suppression was written in (R38).
- ``content_hash`` — the forgotten source's content hash, computed at UPLOAD
  time by the T4 guards. Deliberately NOT backfilled: a row that predates the
  hash has no value to record, and NULL is the honest "unknown".

The SQLite ladder's v5 -> v6 step (``app.database._upgrade_v5_to_v6``) mirrors
this revision: the same columns, the same ``VARCHAR(32)`` / ``VARCHAR(64)``
types, the same NULLABILITY — so a Postgres database and a SQLite file upgraded
in place end up with the same shape.

Revision ID: d8e9f0a1b2c3
Revises: c2d3e4f5a6b7
Create Date: 2026-09-17 00:00:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "d8e9f0a1b2c3"
down_revision: str | None = "c2d3e4f5a6b7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "memory_suppressions",
        sa.Column("namespace", sa.String(32), nullable=True),
    )
    op.add_column(
        "memory_suppressions",
        sa.Column("content_hash", sa.String(64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("memory_suppressions", "content_hash")
    op.drop_column("memory_suppressions", "namespace")
