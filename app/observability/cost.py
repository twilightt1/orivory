"""
Per-call LLM cost tracking.

Provides:
  - Provider pricing tables (configurable)
  - Per-call cost calculation
  - In-memory + SQLite aggregation by user, conversation, agent
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path("eval/costs.db")


# Pricing in USD per 1M tokens (input, output).
# These are sample numbers — override via env or DB in production.
PRICING: dict[str, dict[str, float]] = {
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/gpt-4o": {"input": 5.00, "output": 15.00},
    "openai/gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "openai/gpt-4.1": {"input": 10.00, "output": 30.00},
    "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
    "anthropic/claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "google/gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "meta-llama/llama-3.1-70b-instruct": {"input": 0.59, "output": 0.79},
}


@dataclass
class CallCost:
    timestamp: str
    agent: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    user_id: str | None = None
    conversation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Return USD cost for a call. Falls back to $0 if model is unknown."""
    pricing = PRICING.get(model)
    if pricing is None:
        return 0.0
    in_cost = (tokens_in / 1_000_000) * pricing["input"]
    out_cost = (tokens_out / 1_000_000) * pricing["output"]
    return round(in_cost + out_cost, 6)


class CostTracker:
    """SQLite-backed LLM cost ledger."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # threading.Lock only serializes threads in ONE process. Under
        # multi-worker servers concurrent writers live in other processes,
        # so wait instead of failing fast with 'database is locked'.
        conn.execute("PRAGMA busy_timeout = 5000;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            # WAL allows one writer + many readers concurrently; persists
            # per-database, so set once here rather than per connection.
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS llm_costs (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts              TEXT NOT NULL,
                    agent           TEXT NOT NULL,
                    model           TEXT NOT NULL,
                    tokens_in       INTEGER NOT NULL,
                    tokens_out      INTEGER NOT NULL,
                    cost_usd        REAL NOT NULL,
                    user_id         TEXT,
                    conversation_id TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_costs_ts ON llm_costs(ts);
                CREATE INDEX IF NOT EXISTS idx_costs_agent ON llm_costs(agent);
                """
            )

    def record(
        self,
        agent: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        user_id: str | None = None,
        conversation_id: str | None = None,
    ) -> CallCost:
        cost = calculate_cost(model, tokens_in, tokens_out)
        record = CallCost(
            timestamp=datetime.now(UTC).isoformat(),
            agent=agent,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO llm_costs
                      (ts, agent, model, tokens_in, tokens_out, cost_usd, user_id, conversation_id)
                    VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (
                        record.timestamp,
                        record.agent,
                        record.model,
                        record.tokens_in,
                        record.tokens_out,
                        record.cost_usd,
                        record.user_id,
                        record.conversation_id,
                    ),
                )
                conn.commit()
        return record

    def total(self, since_iso: str | None = None) -> float:
        with self._connect() as conn:
            if since_iso:
                row = conn.execute(
                    "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_costs WHERE ts >= ?",
                    (since_iso,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_costs"
                ).fetchone()
        return float(row[0]) if row else 0.0

    def breakdown_by_agent(self, since_iso: str | None = None) -> dict[str, dict[str, float]]:
        sql = (
            "SELECT agent, COUNT(*) AS calls, "
            "SUM(tokens_in) AS total_in, SUM(tokens_out) AS total_out, "
            "SUM(cost_usd) AS total_cost "
            "FROM llm_costs "
        )
        params: tuple[Any, ...] = ()
        if since_iso:
            sql += "WHERE ts >= ? "
            params = (since_iso,)
        sql += "GROUP BY agent ORDER BY total_cost DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return {
            r["agent"]: {
                "calls": int(r["calls"]),
                "tokens_in": int(r["total_in"] or 0),
                "tokens_out": int(r["total_out"] or 0),
                "total_cost_usd": float(r["total_cost"] or 0.0),
            }
            for r in rows
        }

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM llm_costs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]


def estimate_max_budget_alert(current_spend: float, budget: float) -> str | None:
    """Return a warning message if spend is high. None if ok."""
    if budget <= 0:
        return None
    ratio = current_spend / budget
    if ratio >= 1.0:
        return f"Budget exceeded: ${current_spend:.4f} of ${budget:.2f}"
    if ratio >= 0.8:
        return f"80% of budget used: ${current_spend:.4f} of ${budget:.2f}"
    return None


def budget_window_iso(hours: int = 24) -> str:
    from datetime import timedelta

    return (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
