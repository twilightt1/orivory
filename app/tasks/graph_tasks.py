"""Celery tasks for Orivory knowledge-graph extraction."""
from __future__ import annotations

import logging

from app.graph.builder import GraphBuildResult, build_memory_graph_sync
from app.tasks.celery_app import celery_app

log = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="tasks.build_memory_graph",
    max_retries=3,
    default_retry_delay=30,
    retry_backoff=True,
    retry_jitter=True,
    time_limit=600,
    soft_time_limit=540,
    queue="default",
)
def build_memory_graph_task(self, memory_id: str, force: bool = False) -> dict:
    """Build graph entities/relations for one memory.

    Uses a synchronous SQLAlchemy session because Celery workers in this
    codebase already follow that pattern for ingestion tasks.
    """
    from celery.exceptions import MaxRetriesExceededError, Retry

    from app.tasks.db import sync_session

    with sync_session() as db:
        try:
            result = build_memory_graph_sync(db, memory_id, force=force)
            log.info("Graph build complete", extra=result.to_dict())
            return result.to_dict()
        except Exception as exc:
            db.rollback()
            log.warning(
                "Graph build failed",
                extra={"memory_id": memory_id, "force": force, "error": str(exc)},
            )
            try:
                raise self.retry(exc=exc)
            except Retry:
                raise
            except MaxRetriesExceededError:
                result = GraphBuildResult(
                    memory_id=memory_id,
                    user_id=None,
                    skipped=True,
                    error=str(exc),
                )
                log.error("Graph build failed permanently", extra=result.to_dict())
                return result.to_dict()
