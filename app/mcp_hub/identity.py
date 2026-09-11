"""Agent identity for MCP tool calls.

Every MCP request carries a per-client token (registered via the agents
API). ``resolve_principal`` maps it to an active AgentClient row and returns
a lightweight principal the tools can enforce scopes against. The MCP SDK
does not pass caller identity inside the protocol — so the hub is where
identity, scopes and the access ledger live.
"""
from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_client import AgentClient
from app.services.agent_token_service import hash_token

ACTION_SEARCH = "mcp_search"
ACTION_GET = "mcp_get"
ACTION_LIST = "mcp_list"
ACTION_ADD = "mcp_add"
ACTION_CORRECT = "mcp_correct"
ACTION_DELETE = "mcp_delete"
ACTION_FORGET = "mcp_forget"


@dataclass(frozen=True)
class AgentPrincipal:
    user_id: uuid.UUID
    agent_client_id: uuid.UUID
    name: str
    scopes: frozenset[str]

    def can_read(self) -> bool:
        return "memory:read" in self.scopes

    def can_write(self) -> bool:
        return "memory:write" in self.scopes


def extract_token(headers: Mapping[str, str]) -> str | None:
    auth = headers.get("Authorization") or headers.get("authorization")
    if auth:
        scheme, _, value = auth.partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            return value.strip()
    return headers.get("X-Orivory-Agent-Token") or headers.get("x-orivory-agent-token")


async def resolve_principal(db: AsyncSession, token: str | None) -> AgentPrincipal | None:
    if not token:
        return None
    row = (
        await db.execute(select(AgentClient).where(AgentClient.token_hash == hash_token(token)))
    ).scalars().first()
    if row is None or not row.is_active:
        return None
    now = datetime.now(UTC)
    # Touch the ORM instance as well: a bulk UPDATE alone does not refresh
    # attributes already loaded into the session (and unit-test fakes never
    # see it at all).
    row.last_used_at = now
    await db.execute(
        update(AgentClient).where(AgentClient.id == row.id).values(last_used_at=now)
    )
    await db.commit()
    return AgentPrincipal(
        user_id=row.user_id,
        agent_client_id=row.id,
        name=row.name,
        scopes=frozenset(row.scopes or []),
    )


__all__ = [
    "ACTION_ADD", "ACTION_CORRECT", "ACTION_DELETE", "ACTION_FORGET", "ACTION_GET", "ACTION_LIST", "ACTION_SEARCH",
    "AgentPrincipal", "extract_token", "resolve_principal",
]
