"""users retention settings — the P4b/T6 opt-in auto-expiration window

Retention is OPT-IN per user (spec §8.1: "Auto expiration/archival phải opt-in
và có reason/audit; initial default không tự xóa memory"):

- ``retention_enabled`` — NOT NULL with the constant default. The ADD COLUMN
  default IS the backfill: every pre-existing user is OFF, because no migration
  may decide for a user that their memories should start expiring. The Python
  model spells the same ``server_default="0"`` so a fresh install and an
  upgraded one end up with the same shape.
- ``retention_days`` — the user's window, NULLABLE with no default: "no window
  chosen yet" is a real state, and 0 would be a window that expires everything.

The SQLite ladder's v6 -> v7 step (``app.database._upgrade_v6_to_v7``) mirrors
this revision: the same columns, the same ``BOOLEAN`` / ``INTEGER`` types, the
same defaults — so a Postgres database and a SQLite file upgraded in place end
up with the same shape.

Revision ID: e7f8a9b0c1d2
Revises: d8e9f0a1b2c3
Create Date: 2026-09-17 18:00:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "e7f8a9b0c1d2"
down_revision: str | None = "d8e9f0a1b2c3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("retention_enabled", sa.Boolean(), nullable=False, server_default="0"),
    )
    op.add_column(
        "users",
        sa.Column("retention_days", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("users", "retention_days")
    op.drop_column("users", "retention_enabled")
