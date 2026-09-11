"""Model/migration reconciliation: JSON list columns + referral tables

Two drift classes this revision fixes (found by tests/migrations):

1. ARRAY -> JSON: the lite-mode cross-dialect model change (commit b7eecb6)
   switched tags/scopes/aliases from ARRAY(String) to JSON so SQLite could
   work, but the migration chain still created those columns as varchar[].
   A fresh alembic-provisioned Postgres failed on the very first insert of a
   tagged Memory or an AgentClient with scopes (asyncpg DatatypeMismatchError:
   "column \"tags\" is of type character varying[] but expression is of type
   json"). SQLite (lite mode) never saw the bug because bootstrap_sqlite uses
   create_all from the models. This revision converts the three columns to
   JSONB, carrying existing values over with to_jsonb().

2. Referral tables were never migrated: ReferralCode/Referral/ReferralReward
   had models and a mounted router (app/api/v1/router.py) but no CREATE TABLE
   in the chain — every /referral endpoint 500'd on an alembic-provisioned
   database. This revision creates all three tables.

Both steps are guarded so the revision is idempotent on re-run and a no-op
on databases already provisioned by a newer chain.

Revision ID: a9b8c7d6e5f4
Revises: c7d8e9f0a1b2
Create Date: 2026-09-10 20:00:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a9b8c7d6e5f4"
down_revision: str | None = "c7d8e9f0a1b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (table, column, jsonb default) — ARRAY(String) columns the models write as JSON
_ARRAY_TO_JSON = [
    ("memories", "tags", "[]"),
    ("agent_clients", "scopes", '["memory:read"]'),
    ("entities", "aliases", "[]"),
]


def _column_is_array(table: str, column: str) -> bool:
    """True when the column still exists as an ARRAY (needs conversion)."""
    bind = op.get_bind()
    row = bind.execute(
        sa.text(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = :t AND column_name = :c"
        ),
        {"t": table, "c": column},
    ).fetchone()
    return bool(row) and row[0] == "ARRAY"


def _column_is_nullable(table: str, column: str) -> bool:
    bind = op.get_bind()
    row = bind.execute(
        sa.text(
            "SELECT is_nullable FROM information_schema.columns "
            "WHERE table_name = :t AND column_name = :c"
        ),
        {"t": table, "c": column},
    ).fetchone()
    return bool(row) and row[0] == "YES"


def upgrade() -> None:
    # ── 1. ARRAY -> JSONB conversions (guarded: no-op when already JSON) ────
    for table, column, default in _ARRAY_TO_JSON:
        if not _column_is_array(table, column):
            continue  # already converted, or never was ARRAY on this DB
        # The legacy ARRAY columns carry a varchar[] server default
        # ('{}'::varchar[]) which Postgres refuses to auto-cast to jsonb —
        # drop it before the TYPE change, then install a jsonb default.
        op.execute(f"ALTER TABLE {table} ALTER COLUMN {column} DROP DEFAULT")
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN {column} "
            f"TYPE JSONB USING to_jsonb({column})"
        )
        if not _column_is_nullable(table, column):
            op.execute(
                f"ALTER TABLE {table} ALTER COLUMN {column} "
                f"SET DEFAULT '{default}'::jsonb"
            )

    # ── 2. Referral tables (never created by any prior revision) ─────────────
    op.execute("""
        CREATE TABLE IF NOT EXISTS referral_codes (
            id UUID PRIMARY KEY,
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            code VARCHAR(12) NOT NULL UNIQUE,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            max_uses INTEGER NOT NULL DEFAULT 10,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_referral_codes_user_id ON referral_codes (user_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_referral_codes_code ON referral_codes (code)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_referral_codes_user_active "
        "ON referral_codes (user_id, is_active)"
    )

    op.execute("""
        CREATE TABLE IF NOT EXISTS referrals (
            id UUID PRIMARY KEY,
            referral_code_id UUID NOT NULL REFERENCES referral_codes(id) ON DELETE CASCADE,
            referrer_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            referee_id UUID REFERENCES users(id) ON DELETE CASCADE,
            referee_email VARCHAR(255) NOT NULL,
            status VARCHAR(20) NOT NULL DEFAULT 'pending',
            reward_tier INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            completed_at TIMESTAMP WITH TIME ZONE,
            CONSTRAINT uq_referrer_email UNIQUE (referrer_id, referee_email)
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_referrals_referrer_id ON referrals (referrer_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_referrals_referee_id ON referrals (referee_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_referrals_status ON referrals (status)"
    )

    op.execute("""
        CREATE TABLE IF NOT EXISTS referral_rewards (
            id UUID PRIMARY KEY,
            referral_id UUID NOT NULL REFERENCES referrals(id) ON DELETE CASCADE,
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            reward_type VARCHAR(20) NOT NULL,
            reward_value INTEGER NOT NULL DEFAULT 1,
            is_claimed BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
            claimed_at TIMESTAMP WITH TIME ZONE
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_referral_rewards_user_id ON referral_rewards (user_id)"
    )
    # Defensive cleanup: a user may have accumulated several active codes via
    # the old check-then-insert race (no unique constraint existed). Keep the
    # oldest; younger duplicates fall away with their referrals via CASCADE.
    op.execute("""
        DELETE FROM referral_codes
        WHERE is_active
          AND created_at > (
              SELECT MIN(older.created_at) FROM referral_codes older
              WHERE older.user_id = referral_codes.user_id AND older.is_active
          )
    """)
    # One active code per user, enforced by the DB from now on — also closes
    # the check-then-insert race in get_or_create_referral_code at the DB layer.
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_referral_codes_user_active
            ON referral_codes (user_id) WHERE is_active
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_referral_codes_user_active")
    op.execute("DROP TABLE IF EXISTS referral_rewards")
    op.execute("DROP TABLE IF EXISTS referrals")
    op.execute("DROP TABLE IF EXISTS referral_codes")
    for table, column, _default in reversed(_ARRAY_TO_JSON):
        # Postgres forbids subqueries in a USING transform expression, so
        # convert via a temp column: add ARRAY col -> per-row UPDATE from the
        # JSONB values -> drop JSONB col -> rename. NULL/empty arrays survive
        # the round trip; non-string JSON scalars would fail loudly here.
        tmp = f"{column}_arr"
        op.execute(f"ALTER TABLE {table} ADD COLUMN {tmp} VARCHAR[]")
        op.execute(
            f"UPDATE {table} SET {tmp} = ARRAY("
            f"SELECT jsonb_array_elements_text({column}))"
        )
        op.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
        op.execute(f"ALTER TABLE {table} RENAME COLUMN {tmp} TO {column}")
