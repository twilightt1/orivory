"""Fallback / degradation counters (error-budget observability).

Every fail-open path in the retrieval/RAG pipeline must count its
activations here. Rationale: fallbacks are correct behavior, but a rising
fallback rate means the system is bleeding silently (Jina down for a week
looks identical to Jina healthy when every failure falls back to vector
order). Observe first, operate second: alert on rate, not on occurrence.

Process-local by design (same scope as the SQLite cost ledger). Inspect via
fallback_counts() in a shell, or grep the "Fallback activated" debug logs.
A multi-process deployment should aggregate the debug logs centrally.

Canonical path names (keep stable; dashboards alert on them):
  retrieval.vector_unavailable   vector store down, BM25-only answers
  retrieval.rerank_failed        Jina/reranker error, vector order kept
  retrieval.bm25_rebuild_failed  BM25 lazy rebuild failed, stale index used
  mcp.search_sql_fallback        MCP search answered from the SQL ordering
                                 (freshness barrier timed out / vector outage /
                                 a degraded leg served empty — R25(p2))
  crag.grading_failed            Doc grading error, defaulted IRRELEVANT
  index.outbox_drain_failed      drain round raised; intents stay pending for the retry
"""
from __future__ import annotations

import logging
from collections import Counter

log = logging.getLogger(__name__)

_counts: Counter[str] = Counter()


def count_fallback(path: str) -> int:
    """Record one fallback activation; return the new total for path."""
    _counts[path] += 1
    total = _counts[path]
    log.debug("Fallback activated", extra={"path": path, "total": total})
    return total


def fallback_counts() -> dict[str, int]:
    """Snapshot of all fallback counters (for /ready, tests, debugging)."""
    return dict(_counts)


def reset_fallback_counts() -> None:
    """Zero all counters. Tests only — never call in production code."""
    _counts.clear()
