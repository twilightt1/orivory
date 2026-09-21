from __future__ import annotations

import pytest
from sqlalchemy import text

from app.database import engine

pytestmark = [pytest.mark.integration, pytest.mark.requires_infra]


@pytest.mark.asyncio
async def test_live_database_select_one():
    """The live store the app serves from answers a query (SQLite on the lite
    stack — the Postgres branch went with the full-stack surface)."""
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT 1"))

    assert result.scalar_one() == 1
