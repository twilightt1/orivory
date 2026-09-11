#!/usr/bin/env python3
"""Orivory-stack benchmark run: ingest haystack → query through the stack → judge.

Same dataset, same seeded sample (seed 20260906), same judge as the
committed full-context baseline (0.600) — the ONLY difference is where the
answer comes from:

    baseline : model reads the ENTIRE haystack transcript
    system   : haystack is INGESTED into the Orivory memory hub (SQLite +
               Chroma local mode), then each question is answered from what
               the stack RECALLS (MemoryRetriever: vector + salience +
               entity boosts + rerank), capped at RECALL_TOP_K memories.

Honest protocol: every answer + verdict is a real LLM call through the
gateway in .env. The stack's retrieval is real (embeddings via Jina,
ranking via the retriever). Nothing fabricated. This is the FIRST
system-vs-baseline comparison — same seed, same judge, same n.

Usage (from a checkout with .env, dataset under eval/benchmarks/data/):
    LITE_MODE=1 CHROMA_MODE=local python3 eval/run_system_benchmark.py \
        --n 20 [--seed 20260906] [--concurrency 4]
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENV_FILE = ROOT / ".env"
if not ENV_FILE.exists():  # worktree fallback
    ENV_FILE = ROOT.parent.parent / ".env"
for line in ENV_FILE.open():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())

# Lite-mode stack: SQLite + in-process Chroma (no external services).
# NOTE: hard overrides, not setdefault — pydantic Settings reads .env
# (which carries a postgres DATABASE_URL), and os.environ BEATS env_file,
# so these must land in os.environ unconditionally.
os.environ["LITE_MODE"] = "1"
os.environ["CHROMA_MODE"] = "local"
os.environ["JWT_SECRET_KEY"] = "benchmark-run-secret-key-not-for-prod"
_RESULTS_DIR = ROOT / "eval/benchmarks/results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_RESULTS_DIR}/.system_run.db"
# The stack's LLM client (rewriter) reads OPENROUTER_* — point it at the
# same gateway the judge/answerer use.
os.environ.setdefault("OPENROUTER_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
if os.environ.get("OPENAI_BASE_URL"):
    os.environ["OPENROUTER_BASE_URL"] = os.environ["OPENAI_BASE_URL"]
# Chroma local path: default /data/chroma is a Docker volume; on a dev box
# point it inside the results dir.
os.environ["CHROMA_LOCAL_PATH"] = str(_RESULTS_DIR / "chroma")
# Benchmark answers from top-15: the reranker's own top_n must not truncate
# the pool below that (default JINA_RERANKER_TOP_N=5 is the API default).
os.environ["JINA_RERANKER_TOP_N"] = "15"
os.environ["RETRIEVAL_SEMANTIC_RERANK"] = "1"

from eval.benchmarks.llm_judge import JUDGE_PROMPT_VERSION, build_judge_messages  # noqa: E402
from eval.benchmarks.longmemeval_s import load_instances  # noqa: E402

DATASET = ROOT / "eval/benchmarks/data/longmemeval_s_cleaned.json"
MODEL = os.environ.get("BENCHMARK_JUDGE_MODEL", os.environ["LLM_MODEL"])
BASELINE = ROOT / "eval/benchmarks/results/longmemeval_s_baseline.json"
if not BASELINE.exists():  # worktree: committed baseline lives on main checkout
    BASELINE = ROOT.parent.parent / "eval/benchmarks/results/longmemeval_s_baseline.json"

ANSWER_SYSTEM = (
    "You are answering questions using ONLY the memory excerpts provided. "
    "Answer in at most two sentences. If the memories do not contain the "
    "answer, reply exactly: I have no information about that."
)


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
    for memory in memories:  # real embed + chroma upsert (graph skipped: off)
        try:
            await index_new_memory(memory)
            created += 1
        except Exception as exc:
            print(f"    index warning: {exc}")
    (run_dir / f"ingested_{instance.question_id}.json").write_text(
        json.dumps({"question_id": instance.question_id, "memories": created})
    )
    return created


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

    NOTE: a two-phase map-reduce variant was tried (PR #20) and LOST to
    single-pass (0.486 vs 0.570 clean) — it lives on only as a frozen copy
    inside eval/complete_mapreduce_run.py for reproducibility of the
    recorded negative result. Do not re-add it here.
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
    completion = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"MEMORIES:\n{excerpts}\n\nQUESTION: {instance.question}"
                ),
            },
        ],
        temperature=0.0,
        max_tokens=300,
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


async def run_instance(client, user_id, instance, top_k, run_dir: Path,
                      session_level: bool = True, chunk_chars: int = 0,
                      fuse: bool = False) -> dict:
    t0 = time.time()
    try:
        ingested = await ingest_instance(
            user_id, instance, run_dir, session_level=session_level,
            chunk_chars=chunk_chars,
        )
        response, recalled = await answer_from_stack(
            user_id, instance, top_k, fuse=fuse
        )
        correct = await judge_one(client, instance, response)
        error = None
    except Exception as exc:
        import traceback

        traceback.print_exc()
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
        "seconds": seconds,
        "error": error,
    }


async def main_async(args) -> int:
    from uuid import uuid4

    from openai import AsyncOpenAI

    from app.database import bootstrap_sqlite

    await bootstrap_sqlite()

    from app.database import AsyncSessionLocal
    from app.models.user import User

    benchmark_user_id = uuid4()
    async with AsyncSessionLocal() as db:
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
    t0 = time.time()
    for inst in selected:  # sequential: one user, deterministic ingest order
        records.append(
            await run_instance(
                client, benchmark_user_id, inst, args.top_k, run_dir,
                session_level=args.session, chunk_chars=args.chunk_chars,
                fuse=args.fuse,
            )
        )
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

    payload = {
        "benchmark": "longmemeval_s",
        "run_kind": "orivory_stack",
        "chunking": "session_level" if args.session else "per_turn",
        "answer_mode": "single_pass",
        "note": (
            "REAL dataset, REAL Orivory stack (SQLite + local Chroma + Jina "
            "embeddings + MemoryRetriever salience/rerank), REAL judge. Same "
            "seed and n as the committed full-context baseline. First "
            "system-vs-baseline comparison — small n, treat as directional."
        ),
        "dataset_path": "eval/benchmarks/data/longmemeval_s_cleaned.json",
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "dataset_instances": len(all_instances),
        "sample": {"n": len(selected), "seed": args.seed},
        "model": MODEL,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "stack": {
            "database": "sqlite (lite mode)",
            "vector_store": "chroma local (in-process)",
            "embeddings": "jina-embeddings-v5-text-small",
            "retriever": "MemoryRetriever (vector + salience + entity boost + rerank)",
            "recall_top_k": args.top_k,
        },
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
    suffix = ""
    if args.n >= 100:
        out = ROOT / f"eval/benchmarks/results/longmemeval_s_system_n100{suffix}.json"
    else:
        out = ROOT / (
            "eval/benchmarks/results/longmemeval_s_system.json"
            if chunking == "per_turn"
            else "eval/benchmarks/results/longmemeval_s_system_session.json"
        )
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSYSTEM mean: {mean:.3f} ({correct}/{len(scored)}, errors={len(errors)})")
    if comparison:
        print(f"baseline {comparison['baseline_mean']} → system {mean} "
              f"(delta {comparison['delta']:+.3f}, same_seed={comparison['same_seed']})")
    type_summary = {k: "{}/{}".format(v["correct"], v["n"]) for k, v in by_type.items()}
    print(f"by type: {type_summary}")
    print(f"total: {total}s → {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--session", action="store_true",
                        help="session-level memories (v2 strategy) — "
                             "one memory per haystack session")
    parser.add_argument("--fuse", action="store_true",
                        help="two-pass recall fusion (raw + rewritten "
                             "query, union by id) — 0.650 on the seed "
                             "sample vs 0.700 single-pass; multi-hop "
                             "experiments only")
    parser.add_argument("--chunk-chars", type=int, default=0,
                        help="split each session into turn-aligned chunks of "
                             "at most this many chars (session-level only; "
                             "0 = one memory per session, the diluting "
                             "extreme) — the RAG sweet spot is ~4000")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
