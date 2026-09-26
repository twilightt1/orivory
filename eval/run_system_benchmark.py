#!/usr/bin/env python3
"""Orivory-stack benchmark run: ingest haystack → query through the stack → judge.

Same dataset, same seeded sample (seed 20260906), same judge as the
committed full-context baseline (0.600) — the ONLY difference is where the
answer comes from:

    baseline : model reads the ENTIRE haystack transcript
    system   : haystack is INGESTED into the Orivory memory hub (SQLite +
               Qdrant local mode), then each question is answered from what
               the stack RECALLS (MemoryRetriever: vector + salience +
               entity boosts + rerank), capped at RECALL_TOP_K memories.

Honest protocol: every answer + verdict is a real LLM call through the
gateway in .env. Retrieval is real and local (embeddings + reranking); nothing
is fabricated. This is the FIRST system-vs-baseline comparison — same seed,
same judge, same n.

Usage (from a checkout with .env, dataset under eval/benchmarks/data/):
    QDRANT_MODE=local python3 eval/run_system_benchmark.py \
        --n 20 [--seed 20260906] [--concurrency 4]
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENV_FILE = ROOT / ".env"
if not ENV_FILE.exists():  # worktree fallback
    ENV_FILE = ROOT.parent.parent / ".env"
# No .env anywhere — an archive checkout, a bare container, a worktree whose
# parent has none: run on the ambient environment instead of dying in a
# FileNotFoundError the caller cannot act on.
lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
for line in lines:
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

# The shipped stack: SQLite + in-process Qdrant (no external services).
# NOTE: hard overrides, not setdefault — pydantic Settings reads .env (which
# may carry a DATABASE_URL), and os.environ BEATS env_file, so these must
# land in os.environ unconditionally.
os.environ["QDRANT_MODE"] = "local"
_RESULTS_DIR = ROOT / "eval/benchmarks/results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_RESULTS_DIR}/.system_run.db"
# The stack's LLM client (rewriter) reads OPENROUTER_* — point it at the
# same gateway the judge/answerer use. `or` (not setdefault): an exported
# but EMPTY key must not block the fallback.
os.environ["OPENROUTER_API_KEY"] = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
if os.environ.get("OPENAI_BASE_URL"):
    os.environ["OPENROUTER_BASE_URL"] = os.environ["OPENAI_BASE_URL"]
# Qdrant local path: default /data/qdrant is a Docker volume; on a dev box
# point it inside the results dir.
os.environ["QDRANT_LOCAL_PATH"] = str(_RESULTS_DIR / "qdrant")
# The benchmark is a reproducible self-host lane regardless of stale .env
# provider flags/keys.
os.environ["USE_LOCAL_EMBEDDINGS"] = "1"
# Benchmark answers from top-15: the reranker's own top_n must not truncate
# the pool below that — pin the cap here, independent of the shipped default
# (20 since R13(p2)).
os.environ["RERANK_TOP_N"] = "15"
os.environ["RETRIEVAL_SEMANTIC_RERANK"] = "1"

# The per-memory entity graph is best-effort derived data. During benchmark
# ingest its LLM calls share the answer/rewrite gate and can starve the scored
# queries; keep it off unless --with-graph is explicitly requested.
GRAPH_BUILDS_ENABLED = False
_GRAPH_BUILD_ORIGINAL = None


def _skip_graph_build(*args, **kwargs) -> None:
    return None


def _apply_graph_build_switch(enabled: bool) -> None:
    global GRAPH_BUILDS_ENABLED, _GRAPH_BUILD_ORIGINAL
    from app.retrieval.memory import write_back

    if _GRAPH_BUILD_ORIGINAL is None:
        _GRAPH_BUILD_ORIGINAL = write_back.safe_enqueue_graph_build
    write_back.safe_enqueue_graph_build = (
        _GRAPH_BUILD_ORIGINAL if enabled else _skip_graph_build
    )
    GRAPH_BUILDS_ENABLED = enabled


from eval.benchmarks.llm_judge import JUDGE_PROMPT_VERSION, build_judge_messages  # noqa: E402
from eval.benchmarks.longmemeval_s import load_instances  # noqa: E402

DATASET = ROOT / "eval/benchmarks/data/longmemeval_s_cleaned.json"
if not DATASET.exists():  # worktree: ignored benchmark data lives on main checkout
    DATASET = ROOT.parent.parent / "eval/benchmarks/data/longmemeval_s_cleaned.json"
MODEL = os.environ.get("BENCHMARK_JUDGE_MODEL", os.environ["LLM_MODEL"])
BASELINE = ROOT / "eval/benchmarks/results/longmemeval_s_baseline.json"
if not BASELINE.exists():  # worktree: committed baseline lives on main checkout
    BASELINE = ROOT.parent.parent / "eval/benchmarks/results/longmemeval_s_baseline.json"

# Measured on the 38 wrong questions of the 0.62 run (wrong-only probe,
# artifact eval/benchmarks/results/longmemeval_s_system_n100_20260925T174857.json):
# of 21 with the gold already inside the served top-15, 5 returned an EMPTY
# answer and 16 hit the refusal branch. Both prompt and token cap changed
# here; the pair is the measured effect, not either half alone.
ANSWER_SYSTEM = (
    "You are answering questions about a user from their stored memories. "
    "Use ONLY the numbered MEMORIES below, but ALWAYS give your best specific "
    "answer from them — never reply that you have no information. "
    "Rules: quote numbers, dates, names and durations exactly as they appear; "
    "if a memory is dated, use those dates to compute durations and 'how long "
    "ago' questions; when memories disagree, prefer the most recent one; "
    "for preference questions, answer with the concrete choice the user made. "
    "Reply with the answer only, no preamble, in one short sentence."
)

# ponytail: 300 was a guess. Reasoning-channel models burn the whole budget
# before emitting `content`, and the completion then returns content=None
# (finish_reason=length) — 5 empty answers in that run. Raise when a probe
# shows a truncation again; there is no upper pressure from the prompt, which
# asks for one sentence.
ANSWER_MAX_TOKENS = 2048
ANSWER_PROMPT_VERSION = "no-refusal-v1+date-hint"

# 'How long ago' / 'since when' questions are unanswerable without a reference
# date: the model anchors them to the transcript instead. Measured on the 9
# temporal questions of that same probe: 1/9 correct without it, 6/9 with it
# (discordant 5-0, exact binomial p=0.0625). The date rides on the user turn
# only where the question asks for it — passing it unconditionally measured
# 9/21 vs 12/21 on non-temporal questions.
RELATIVE_TIME_CUE = re.compile(
    r"\b(how long|how many (?:days|weeks|months|years|hours|minutes)"
    r"|ago|since|before|after|last (?:week|month|year|night)"
    r"|this (?:week|month|year)|yesterday|tomorrow|recently|earlier"
    r"|when did|what date|what day)\b",
    re.IGNORECASE,
)


def build_stack_metadata(
    *,
    top_k: int,
    selected_question_ids: list[str] | None = None,
    sample_seed: int | None = None,
    concurrency: int | None = None,
    session_level: bool | None = None,
    chunk_chars: int | None = None,
    fuse: bool | None = None,
    write_index_costs: dict | None = None,
) -> dict:
    """Build reproducibility metadata from the active benchmark stack.

    The benchmark remains sequential in this P0 script, so requested and
    actual concurrency are recorded separately rather than implying parallel
    work that did not run.
    """
    import importlib.metadata
    import platform
    import subprocess

    from app.config import settings
    from app.retrieval.embedder import active_backend_name
    from app.retrieval.embedding_fingerprint import current_fingerprint

    fingerprint = current_fingerprint()
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        git_dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    except Exception:
        git_head = "unknown"
        git_dirty = None

    def package_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    dataset_path = DATASET.resolve()
    try:
        dataset_source = "worktree" if dataset_path.is_relative_to(ROOT) else "external-checkout-fallback"
    except AttributeError:  # pragma: no cover - Python 3.8 compatibility
        dataset_source = "worktree" if str(dataset_path).startswith(str(ROOT)) else "external-checkout-fallback"
    thread_limits = {
        name: os.environ.get(name)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "ORT_NUM_THREADS")
    }
    return {
        "database": "sqlite (lite mode)",
        "vector_store": "qdrant local (in-process)",
        # Keep the legacy scalar key for existing result consumers.
        "embeddings": fingerprint["model_id"],
        "embeddings_actual": {
            "model_id": fingerprint["model_id"],
            "pooling": fingerprint["pooling"],
            "dim": fingerprint["dim"],
            "provider": fingerprint.get("provider"),
            "model_revision": fingerprint.get("model_revision"),
            "artifact_digest": fingerprint.get("artifact_digest"),
            "tokenizer_digest": fingerprint.get("tokenizer_digest"),
            "fingerprint": fingerprint,
        },
        "embedding_backend": active_backend_name(),
        "query_prefix": fingerprint["query_prefix"],
        "passage_prefix": fingerprint["passage_prefix"],
        "retriever": "MemoryRetriever (vector + salience + entity boost + rerank)",
        "graph_builds": "on" if GRAPH_BUILDS_ENABLED else "off (bench ingest)",
        "recall_top_k": top_k,
        "retrieval": {
            "hybrid_enabled": bool(settings.RETRIEVAL_HYBRID_ENABLED),
            "rerank_pool_multiplier": settings.RETRIEVAL_RERANK_POOL_MULTIPLIER,
            "rrf_k": settings.RETRIEVAL_RRF_K,
        },
        "git_head": git_head,
        "git_dirty": git_dirty,
        "dataset_path": str(dataset_path),
        "dataset_source": dataset_source,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "selected_question_ids": selected_question_ids,
        "sample_seed": sample_seed,
        "runtime": {
            "python": platform.python_version(),
            "os": platform.platform(),
            "architecture": platform.machine(),
            "cpu_count": os.cpu_count(),
            "packages": {
                name: package_version(name)
                for name in ("aiosqlite", "qdrant_client", "onnxruntime", "sqlalchemy", "tokenizers")
            },
        },
        "rerank": {
            "enabled": bool(settings.RETRIEVAL_SEMANTIC_RERANK),
            "model": "gte-multilingual-reranker-base (local ONNX, int8)",
            "top_n": settings.RERANK_TOP_N,
        },
        "answer": {
            "model": MODEL,
            "temperature": 0.0,
            "max_tokens": ANSWER_MAX_TOKENS,
            "prompt_version": ANSWER_PROMPT_VERSION,
            "passes_reference_date": True,
        },
        "judge": {
            "model": MODEL,
            "prompt_version": JUDGE_PROMPT_VERSION,
            "temperature": 0.0,
            "max_tokens": 8,
        },
        "timeouts_seconds": {
            "answer_and_judge_client": 240,
        },
        "execution": {
            "requested_concurrency": concurrency if concurrency is not None else 1,
            "actual_concurrency": 1,
            "thread_limits": thread_limits,
            "warmup_performed": False,
            "cache_state": "not_reset; process/model/filesystem cache may be warm",
            "rewrite_policy": "personal-context plus pronoun policy; fast-path when eligible",
            "context_policy": {
                "session_level": session_level,
                "chunk_chars": chunk_chars,
                "fuse": fuse,
            },
        },
        "write_index_costs": write_index_costs or {},
    }


def _parse_session_date(raw: str) -> datetime | None:
    """LongMemEval date format: '2023/05/20 (Sat) 01:42'."""
    import re

    match = re.match(r"(\d{4})/(\d{2})/(\d{2}).*?(\d{2}):(\d{2})", raw or "")
    if not match:
        return None
    year, month, day, hour, minute = (int(g) for g in match.groups())

    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def _session_title(session, instance, max_chars: int = 120) -> str:
    """Title a session memory from its first substantive user turn.

    The embedder concatenates title + content (vector_store helper), so a
    topical title anchors the embedding on what the session is ABOUT —
    per-turn fragments ("user turn — <qid>") gave the embedder nothing to
    match a query against. This was the primary retrieval failure mode in
    the per-turn run (multi-session 2/8 vs baseline 5/8).
    """
    for turn in session.turns:
        text = " ".join((turn.content or "").split())
        if len(text) >= 24:
            return text[:max_chars]
    return f"Conversation {session.session_id} ({instance.question_id})"


async def ingest_instance(user_id, instance, run_dir: Path, session_level: bool = True,
                          chunk_chars: int = 0) -> int:
    """Ingest one instance's haystack as memories (real embed + index).

    Returns the number of memories whose vector write actually LANDED —
    ``index_new_memory`` reports a failed upsert as ``False``/an exception and
    those are never counted as created (the row is still in SQLite with the
    durable outbox intent, but the index does not have it).

    Session-level strategy (v2): ONE memory per haystack session — the full
    turn-by-turn transcript as content, a topical title derived from the
    first substantive user turn, captured_at from the session date. A
    session is the unit a LongMemEval answer lives in: per-turn fragments
    scattered the answer's context across 500+ low-context rows and
    salience/recency ranking surfaced the wrong ones (PR #14: 0.450).
    """
    from uuid import uuid4

    from app.database import AsyncSessionLocal
    from app.models.memory import Memory
    from app.retrieval.memory.write_back import index_new_memory

    created = 0
    index_failed = 0
    memories: list[Memory] = []
    for session in instance.sessions:
        session_dt = _parse_session_date(session.date) or datetime(
            2023, 1, 1, tzinfo=UTC
        )
        if session_level:
            turn_lines = [
                f"{turn.role}: {turn.content}"
                for turn in session.turns
                if turn.content.strip()
            ]
            if not turn_lines:
                continue
            full_text = "\n\n".join(turn_lines)
            if chunk_chars and len(full_text) > chunk_chars:
                # Turn-aligned chunking: pack whole turns into ≤chunk_chars
                # blocks. A 17k-char session as ONE memory dilutes the
                # embedding (the per-session run missed its answer session
                # in the top-25 vector hits); per-turn fragments lose local
                # context. ~4k-char chunks keep both.
                chunks: list[list[str]] = []
                current: list[str] = []
                current_len = 0
                for line in turn_lines:
                    if current and current_len + len(line) > chunk_chars:
                        chunks.append(current)
                        # 1-turn overlap: a fact whose statement straddles
                        # the boundary ("...ordered it on the 15th." | next
                        # chunk starts with the reply) lives in BOTH chunks.
                        current, current_len = [current[-1]], len(current[-1])
                    current.append(line)
                    current_len += len(line)
                if current:
                    chunks.append(current)
                for chunk_no, chunk in enumerate(chunks, 1):
                    memories.append(
                        Memory(
                            id=uuid4(),
                            user_id=user_id,
                            title=(
                                f"{_session_title(session, instance)} "
                                f"[{chunk_no}/{len(chunks)}]"
                            )[:500],
                            content="\n\n".join(chunk),
                            summary=None,
                            tags=["longmemeval", instance.question_type],
                            source_type="other",
                            source_ref=(
                                f"bench:{instance.question_id}:"
                                f"{session.session_id}:{chunk_no}"
                            ),
                            captured_at=session_dt,
                        )
                    )
            else:
                memories.append(
                    Memory(
                        id=uuid4(),
                        user_id=user_id,
                        title=_session_title(session, instance)[:500],
                        content=full_text[:20_000],
                        summary=None,
                        tags=["longmemeval", instance.question_type],
                        source_type="other",
                        source_ref=(
                            f"bench:{instance.question_id}:{session.session_id}"
                        ),
                        captured_at=session_dt,
                    )
                )
            continue
        # per-turn strategy (v1, kept for A/B reproduction)
        for turn in session.turns:
            content = turn.content[:8000]
            if not content.strip():
                continue
            memories.append(
                Memory(
                    id=uuid4(),
                    user_id=user_id,
                    title=f"{turn.role} turn — {instance.question_id}"[:500],
                    content=content,
                    summary=None,
                    tags=["longmemeval", instance.question_type],
                    source_type="other",
                    source_ref=(
                        f"bench:{instance.question_id}:{session.session_id}"
                    ),
                    captured_at=session_dt,
                )
            )
    async with AsyncSessionLocal() as db:
        db.add_all(memories)
        await db.commit()
    for memory in memories:  # real embed + qdrant upsert (graph skipped: off)
        try:
            landed = await index_new_memory(memory)
        except Exception as exc:
            landed = False
            print(f"    index warning: {exc}")
        if landed:
            created += 1
        else:
            index_failed += 1
    if index_failed:
        print(
            f"    index: {created}/{len(memories)} memories landed, "
            f"{index_failed} FAILED (not counted as ingested)"
        )
    (run_dir / f"ingested_{instance.question_id}.json").write_text(
        json.dumps(
            {
                "question_id": instance.question_id,
                "memories": created,
                "index_failed": index_failed,
            }
        )
    )
    if index_failed:
        # A haystack missing memories is not the haystack the benchmark means to
        # measure: the instance is scored against whatever landed. Raising hands
        # it to the caller's error path, which keeps the run off the committed
        # artifact and out of the mean.
        raise RuntimeError(
            f"instance {instance.question_id}: {index_failed} of {len(memories)} "
            "memories never reached the index, so its haystack is incomplete"
        )
    return created


async def purge_instance_memories(user_id, question_id: str) -> int | None:
    """Delete one scored instance's memories (DB rows + vectors).

    Returns the number of rows purged, or None when the vector delete did not
    confirm. On None the SQL rows are deliberately left in place: dropping them
    would strand vectors that nothing can find again, and the caller has to
    treat the run as partial — the next instance shares the user and would
    otherwise recall this haystack.

    The run uses ONE benchmark user for the whole sample (the pinned
    single-user protocol), and the retriever is user-scoped: without this,
    a later instance's recall can return an earlier instance's haystack and
    the scores become order-dependent. Cleaning up between instances keeps
    that protocol intact — a user id per instance would break segment parity.
    """
    from sqlalchemy import delete, select

    from app.database import AsyncSessionLocal
    from app.models.memory import Memory
    from app.retrieval.memory.vector_store import delete_memories as _delete_vectors

    prefix = f"bench:{question_id}:"
    async with AsyncSessionLocal() as db:
        stale = (
            await db.execute(
                select(Memory.id).where(
                    Memory.user_id == user_id,
                    Memory.source_ref.startswith(prefix),
                )
            )
        ).scalars().all()
        if stale:
            if not await _delete_vectors([str(i) for i in stale]):
                return None
            await db.execute(
                delete(Memory).where(
                    Memory.user_id == user_id,
                    Memory.source_ref.startswith(prefix),
                )
            )
            await db.commit()
    return len(stale)


async def stack_recall(user_id, query: str, top_k: int,
                       fuse: bool = False) -> list[dict]:
    """Real retrieval: MemoryRetriever over the ingested memories.

    Two-pass fusion: pass 1 recalls with the raw question; pass 2 recalls
    with the retriever's own LLM-rewritten query (a differently-phrased
    angle). Union by memory id, keeping each memory's best rank — multi-hop
    facts phrased differently in different sessions surface from one pass
    even when the other misses them.
    """
    from app.database import AsyncSessionLocal
    from app.retrieval.memory.retriever import MemoryRetriever

    async with AsyncSessionLocal() as db:
        retriever = MemoryRetriever(db, user_id)
        response = await retriever.recall(query, top_k=top_k)
        results = list(getattr(response, "results", []) or [])
        rewritten = None
        trace = getattr(response, "trace", None)
        if trace is not None:
            rewritten = getattr(trace, "rewritten_query", None)
        if fuse and rewritten and rewritten.strip().lower() != query.strip().lower():
            second = await retriever.recall(rewritten, top_k=top_k)
            results.extend(getattr(second, "results", []) or [])

        # Union by id, best (lowest) rank wins
        seen: dict[str, dict] = {}
        for rank, m in enumerate(results):
            mid = str(m.id)
            if mid not in seen:
                seen[mid] = {
                    "title": m.title,
                    # 4000 chars: matches the ingest chunk size — a
                    # 1200-char excerpt truncated 4k chunks mid-context
                    # (PR #15 finding).
                    "content": (m.content or "")[:4000],
                    "captured_at": (
                        m.captured_at.isoformat() if m.captured_at else None
                    ),
                    "_rank": rank,
                }
        fused = sorted(seen.values(), key=lambda r: r["_rank"])
        for row in fused:
            del row["_rank"]
        return fused[: top_k * 2]


async def answer_from_stack(
    user_id, instance, top_k: int, fuse: bool = False
) -> tuple[str, int]:
    """Answer the question from what the stack recalls (single pass).

    NOTE: a two-phase map-reduce variant was tried twice and LOST both times.
    PR #20: 0.486 vs 0.570 hosted. Re-measured on the 30 recoverable questions
    of the 0.62 run (wrong-only probe, same recall window, same judge): 15/30
    with per-memory MAP vs 13/30 here — discordant 3-4, exact binomial p=1.0,
    for 15 extra LLM calls per question. A single-call MAP was worse still: it
    truncated to content=None at both 2048 and 8192 tokens. Do not re-add it.
    """
    from openai import AsyncOpenAI

    recalled = await stack_recall(user_id, instance.question, top_k, fuse=fuse)
    if not recalled:
        return "I have no information about that.", 0
    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
        timeout=240.0,
    )

    excerpts = "\n\n".join(
        f"[{i + 1}] ({r['captured_at'] or 'undated'}) {r['content']}"
        for i, r in enumerate(recalled)
    )
    # The reference date goes on the user turn, and only when the question is
    # about relative time — unconditional measured worse (see RELATIVE_TIME_CUE).
    preamble = ""
    if instance.question_date and RELATIVE_TIME_CUE.search(instance.question):
        preamble = f"TODAY'S DATE: {instance.question_date}\n\n"
    completion = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"{preamble}MEMORIES:\n{excerpts}\n\nQUESTION: {instance.question}"
                ),
            },
        ],
        temperature=0.0,
        max_tokens=ANSWER_MAX_TOKENS,
    )
    return (completion.choices[0].message.content or "").strip(), len(recalled)


async def judge_one(client, instance, response: str) -> bool:
    completion = await client.chat.completions.create(
        model=MODEL,
        messages=build_judge_messages(instance.question, instance.answer, response),
        temperature=0.0,
        max_tokens=8,
    )
    verdict = (completion.choices[0].message.content or "").strip().lower().strip("`.*!\n ")
    return verdict == "correct"


class QuotaExhausted(Exception):
    """Raised when the provider daily quota dies mid-run.

    Continuing would mark every remaining question error after burning
    retries on each — abort loudly, keep the partial file, rerun after
    reset instead (learned after two 68-error runs).
    """


async def run_instance(client, user_id, instance, top_k, run_dir: Path,
                      session_level: bool = True, chunk_chars: int = 0,
                      fuse: bool = False) -> dict:
    t0 = time.time()
    ingest_seconds = recall_seconds = 0.0
    try:
        ingest_t0 = time.perf_counter()
        ingested = await ingest_instance(
            user_id, instance, run_dir, session_level=session_level,
            chunk_chars=chunk_chars,
        )
        ingest_seconds = time.perf_counter() - ingest_t0
        recall_t0 = time.perf_counter()
        response, recalled = await answer_from_stack(
            user_id, instance, top_k, fuse=fuse
        )
        recall_seconds = time.perf_counter() - recall_t0
        correct = await judge_one(client, instance, response)
        error = None
    except Exception as exc:
        import traceback

        from app.retrieval.embedder import EmbeddingDimensionMismatch

        traceback.print_exc()
        if isinstance(exc, EmbeddingDimensionMismatch):
            raise
        if type(exc).__name__ == "RateLimitError":
            raise QuotaExhausted(f"provider quota exhausted at {instance.question_id}") from exc
        response, recalled, ingested, correct, error = "", 0, 0, False, f"{type(exc).__name__}: {exc}"
    seconds = round(time.time() - t0, 1)
    status = "error" if error else ("correct" if correct else "incorrect")
    print(f"  {instance.question_id} [{instance.question_type}]: {status} "
          f"(ingested={ingested}, recalled={recalled}, {seconds}s)")
    return {
        "question_id": instance.question_id,
        "question_type": instance.question_type,
        "correct": correct,
        "response": response[:500],
        "memories_ingested": ingested,
        "memories_recalled": recalled,
        "ingest_seconds": round(ingest_seconds, 3),
        "recall_seconds": round(recall_seconds, 3),
        "seconds": seconds,
        "error": error,
    }


def result_artifact_path(n: int, chunking: str, *, partial: bool) -> Path:
    """Path for a run's results artifact.

    ``partial=True`` (instances errored, or the run quit early) appends
    ``_partial``: a partial run's mean covers fewer questions and must never
    write the name a clean run commits under.
    """
    if n >= 100:
        out = ROOT / "eval/benchmarks/results/longmemeval_s_system_n100.json"
    else:
        out = ROOT / (
            "eval/benchmarks/results/longmemeval_s_system.json"
            if chunking == "per_turn"
            else "eval/benchmarks/results/longmemeval_s_system_session.json"
        )
    if partial:
        out = out.with_name(f"{out.stem}_partial{out.suffix}")
    return out


def run_exit_code(records: list[dict], base_code: int = 0) -> int:
    """Exit code for a finished run — 0 only when every instance completed.

    ``base_code`` carries an earlier abort (3 = quota exhausted); otherwise an
    instance that errored makes the run non-zero (2): the mean covers fewer
    questions than the sample, which is not a clean success.
    """
    if base_code:
        return base_code
    return 2 if any(r.get("error") for r in records) else 0


async def main_async(args) -> int:
    from uuid import uuid4

    from openai import AsyncOpenAI

    from app.database import bootstrap_sqlite

    await bootstrap_sqlite()

    from app.database import AsyncSessionLocal
    from app.models.user import User

    benchmark_user_id = uuid4()
    async with AsyncSessionLocal() as db:
        from sqlalchemy import delete, select

        from app.models.memory import Memory
        from app.retrieval.memory.vector_store import (
            delete_memories as _delete_bench_vectors,
        )

        existing = (await db.execute(
            select(User).where(User.email == "benchmark@orivory.local")
        )).scalars().first()
        if existing is None:
            db.add(
                User(
                    id=benchmark_user_id,
                    email="benchmark@orivory.local",
                    hashed_password="x",
                    is_verified=True,
                    is_active=True,
                )
            )
            await db.commit()
        else:
            # Reruns start clean: a prior run's bench memories (DB + vectors)
            # would otherwise pollute recall under the same user. The resume
            # flow re-ingests per-instance itself and never enters here.
            benchmark_user_id = existing.id
            stale = (await db.execute(
                select(Memory.id).where(
                    Memory.user_id == existing.id,
                    Memory.source_ref.like("bench:%"),
                )
            )).scalars().all()
            if stale:
                await _delete_bench_vectors([str(i) for i in stale])
                await db.execute(
                    delete(Memory).where(
                        Memory.user_id == existing.id,
                        Memory.source_ref.like("bench:%"),
                    )
                )
                await db.commit()
                print(f"cleared {len(stale)} stale bench memories from a prior run")

    all_instances = load_instances(DATASET)
    rng = random.Random(args.seed)
    indices = list(range(len(all_instances)))
    rng.shuffle(indices)
    selected = [all_instances[i] for i in sorted(indices[: args.n])]

    run_dir = ROOT / "eval/benchmarks/results/system_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"dataset: {DATASET.name} | sample n={len(selected)} seed={args.seed}")
    print(f"model: {MODEL} | judge: {JUDGE_PROMPT_VERSION} | top_k={args.top_k}")
    print(f"user: {benchmark_user_id} (fresh, sqlite at results/.system_run.db)")


    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
        timeout=240.0,
    )

    records: list[dict] = []
    purge_failed: list[str] = []
    t0 = time.time()
    exit_code = 0
    try:
        for inst in selected:  # sequential: one user, deterministic ingest order
            records.append(
                await run_instance(
                    client, benchmark_user_id, inst, args.top_k, run_dir,
                    session_level=args.session, chunk_chars=args.chunk_chars,
                    fuse=args.fuse,
                )
            )
            # Scored — drop this instance's haystack before the next one. One
            # user for the whole sample (pinned single-user protocol) plus a
            # user-scoped retriever means a later instance could otherwise
            # recall an earlier instance's memories (order-dependent scores).
            purged = await purge_instance_memories(benchmark_user_id, inst.question_id)
            if purged:
                print(f"    purged {purged} memories for {inst.question_id}")
            elif purged is None:
                # The next instance shares this user, so anything left behind is
                # recallable by it: every score after this point would be measured
                # against a contaminated haystack. Stop, and report failure.
                purge_failed.append(inst.question_id)
                exit_code = 2
                print(
                    f"    PURGE FAILED for {inst.question_id}: its memories may still be "
                    "recallable, so the run stops here rather than score past it"
                )
                break
    except QuotaExhausted as exc:
        exit_code = 3
        print(f"\nQUOTA EXHAUSTED — aborting run early ({exc}). Partial results "
              f"below cover {len(records)}/{len(selected)}; rerun after reset.")
    total = round(time.time() - t0, 1)

    errors = [r for r in records if r["error"]]
    scored = [r for r in records if not r["error"]]
    correct = sum(1 for r in scored if r["correct"])
    mean = round(correct / len(scored), 3) if scored else 0.0

    by_type: dict[str, dict] = {}
    for r in scored:
        slot = by_type.setdefault(r["question_type"], {"n": 0, "correct": 0})
        slot["n"] += 1
        slot["correct"] += 1 if r["correct"] else 0

    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    comparison = None
    if baseline.get("mean") is not None:
        comparison = {
            "baseline_run_kind": baseline.get("run_kind"),
            "baseline_mean": baseline.get("mean"),
            "baseline_sample": baseline.get("sample"),
            "same_seed": baseline.get("sample", {}).get("seed") == args.seed,
            "delta": round(mean - baseline["mean"], 3) if scored else None,
        }

    # Wilson 95% score interval — the statistics-grade run reports it
    wilson = None
    if scored:
        import math as _math

        n_s, p = len(scored), correct / len(scored)
        z = 1.96
        denom = 1 + z * z / n_s
        center = (p + z * z / (2 * n_s)) / denom
        half = z * _math.sqrt(p * (1 - p) / n_s + z * z / (4 * n_s * n_s)) / denom
        wilson = [round(center - half, 3), round(center + half, 3)]

    write_index_costs = {
        "memories_ingested": sum(r["memories_ingested"] for r in records),
        "ingest_seconds": round(sum(r["ingest_seconds"] for r in records), 3),
        "recall_seconds": round(sum(r["recall_seconds"] for r in records), 3),
        "questions_completed": len(records),
    }
    stack = build_stack_metadata(
        top_k=args.top_k,
        selected_question_ids=[instance.question_id for instance in selected],
        sample_seed=args.seed,
        concurrency=getattr(args, "concurrency", 1),
        session_level=args.session,
        chunk_chars=args.chunk_chars,
        fuse=args.fuse,
        write_index_costs=write_index_costs,
    )
    payload = {
        "benchmark": "longmemeval_s",
        "run_kind": "orivory_stack",
        "chunking": "session_level" if args.session else "per_turn",
        "answer_mode": "single_pass",
        "note": (
            "REAL dataset, REAL Orivory stack (SQLite + local Qdrant + "
            f"{stack['embeddings']} embeddings + MemoryRetriever salience/rerank), "
            "REAL judge. Same seed and n as the committed full-context baseline. "
            "First system-vs-baseline comparison — small n, treat as directional."
        ),
        "dataset_path": str(DATASET.resolve()),
        "dataset_source": stack["dataset_source"],
        "dataset_sha256": stack["dataset_sha256"],
        "dataset_instances": len(all_instances),
        "sample": {
            "n": len(selected),
            "seed": args.seed,
            "question_ids": [instance.question_id for instance in selected],
        },
        "model": MODEL,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "stack": stack,
        "mean": mean,
        "questions": len(scored),
        "correct": correct,
        "errors": len(errors),
        "wilson_95": wilson,
        "by_type": by_type,
        "comparison_to_baseline": comparison,
        "total_seconds": total,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "per_question": records,
    }
    chunking = "session_level" if args.session else "per_turn"
    # A partial run is one that errored instances or quit early: its mean
    # covers fewer questions than the sample.
    partial = bool(errors) or bool(purge_failed) or len(records) < len(selected)
    out = result_artifact_path(n=args.n, chunking=chunking, partial=partial)
    if out.exists():
        # Never overwrite a committed/frozen results file: new runs get a
        # timestamped sibling (learned the hard way — an n=100 rerun once
        # clobbered the frozen 0.570 baseline; restored via git).
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        out = out.with_name(f"{out.stem}_{stamp}{out.suffix}")
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSYSTEM mean: {mean:.3f} ({correct}/{len(scored)}, errors={len(errors)})")
    if partial:
        print(
            f"PARTIAL RUN — {len(errors)} instance(s) errored, {len(purge_failed)} "
            f"unpurged, {len(records)}/"
            f"{len(selected)} completed: the mean covers {len(scored)} question(s) "
            f"and this run did NOT write the committed result artifact. "
            f"Output: {out}"
        )
    if comparison:
        print(f"baseline {comparison['baseline_mean']} → system {mean} "
              f"(delta {comparison['delta']:+.3f}, same_seed={comparison['same_seed']})")
    type_summary = {k: "{}/{}".format(v["correct"], v["n"]) for k, v in by_type.items()}
    print(f"by type: {type_summary}")
    print(f"total: {total}s → {out}")
    return run_exit_code(records, base_code=exit_code)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="requested concurrency (this benchmark currently executes sequentially)",
    )
    parser.add_argument("--session", action="store_true",
                        help="session-level memories (v2 strategy) — "
                             "one memory per haystack session")
    parser.add_argument("--fuse", action="store_true",
                        help="two-pass recall fusion (raw + rewritten "
                             "query, union by id) — 0.650 on the seed "
                             "sample vs 0.700 single-pass; multi-hop "
                             "experiments only")
    parser.add_argument(
        "--with-graph",
        action="store_true",
        help="keep per-memory entity-graph builds enabled during benchmark ingest",
    )
    parser.add_argument("--chunk-chars", type=int, default=0,
                        help="split each session into turn-aligned chunks of "
                             "at most this many chars (session-level only; "
                             "0 = one memory per session, the diluting "
                             "extreme) — the RAG sweet spot is ~4000")
    args = parser.parse_args()
    _apply_graph_build_switch(args.with_graph)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
