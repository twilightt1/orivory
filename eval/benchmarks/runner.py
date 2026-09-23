"""Benchmark runner core: config, dataset hygiene, aggregation, honest result writing.

The runner never touches a live stack in v0 — ingest/query callables are
injected by the caller (live wiring via the API/MCP is a pinned follow-up),
and no score is ever fabricated: a missing dataset is a hard error, and
results only ever aggregate per-question booleans that were actually
produced by an adapter's judge guard.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from eval.benchmarks.longmemeval_s import BenchmarkInstance, BenchmarkSession
from eval.benchmarks.memoryagentbench import Scenario

# Per-question result record shape shared by both adapters' score paths:
# {"question_id": str, "correct": bool} for LongMemEval-S style judging, and
# {"question_id": str, "recalled_after_forget": bool,
#  "stale_fact_still_present": bool, "competency": str} for MemoryAgentBench
# (interpreted via interpret_result before it becomes "correct").

PHASES = ("plan", "ingest", "query", "score")

_DATASET_MISSING = "dataset not found — download per eval/benchmarks/README.md"


@dataclass(frozen=True)
class RunnerConfig:
    """One benchmark run's configuration.

    ``benchmark`` is one of ``longmemeval_s`` | ``memoryagentbench``;
    ``dataset_path`` must exist (checked at run time, never silently skipped);
    ``limit`` caps instance count when set; ``output_dir`` receives results.
    """

    benchmark: str
    dataset_path: Path
    output_dir: Path
    limit: int | None = None


def interpret_result(
    competency: str, recalled_after_forget: bool, *, stale_fact_still_present: bool = False
) -> str:
    """Interpret a MemoryAgentBench per-question recall flag per competency.

    POLARITY (fixture-specific, pinned by tests): the committed
    ``selective_forgetting`` fixture's gold answer is the SURVIVING updated
    fact (10:45 AM / gate C3), so ``recalled_after_forget=True`` there means
    the memory kept the update while the stale fact was forgotten — i.e.
    correct memory → ``"pass"``. A dedicated forgetting-question dataset (whose
    gold is the to-be-forgotten fact itself) would INVERT this mapping; never
    hardcode ``True=BAD`` across the board.

    Under that polarity the forgotten fact leaking back into the SAME response
    is not a pass: a response carrying both the update and the verbatim stale
    fact means the forget failed, so ``stale_fact_still_present=True`` forces
    ``"fail"`` (the adapter already flags the response — see
    :func:`eval.benchmarks.memoryagentbench.run_forgetting_scenario`).

    All other competencies (and unknown future ones) return
    ``"pending_interpretation"`` until the LLM-judge follow-up lands — the
    runner then reports them honestly as not yet scoreable rather than
    guessing.
    """
    if competency == "selective_forgetting":
        if recalled_after_forget and not stale_fact_still_present:
            return "pass"
        return "fail"
    return "pending_interpretation"


def _require_dataset(path: Path) -> Path:
    if not Path(path).is_file():
        raise RuntimeError(_DATASET_MISSING)
    return Path(path)


def record_sha256(path: Path) -> str:
    """SHA-256 hex digest of the dataset file (hygiene rule 3: dataset digest travels with every score).

    Raises ``RuntimeError`` pointing at the README when the dataset is absent —
    the runner never proceeds against a dataset it cannot verify.
    """
    return hashlib.sha256(_require_dataset(path).read_bytes()).hexdigest()


def aggregate(results: list[dict]) -> dict:
    """Aggregate ``[{"question_id", "correct"}]`` into ``{"mean", "questions"}``.

    Only actual ``True`` counts as correct — ``None``/missing flags count as
    failures rather than being dropped or fabricated into passes. Empty
    input yields ``{"mean": 0.0, "questions": 0}`` (never a division by zero).
    """
    questions = len(results)
    if questions == 0:
        return {"mean": 0.0, "questions": 0}
    correct = sum(1 for r in results if r.get("correct") is True)
    return {"mean": correct / questions, "questions": questions}


async def run_ingest(
    config: RunnerConfig,
    instances: list[BenchmarkInstance] | list[Scenario],
    ingest_history: Callable[[BenchmarkSession], Awaitable[int]],
) -> int:
    """Drive the injected session-ingest callable over every instance; return the ingested-turn count.

    v0 honesty: ``ingest_history`` is injected by the caller (live wiring via
    the API/MCP is a pinned follow-up) — it receives each
    :class:`BenchmarkSession` and returns the number of turns it actually
    ingested, and this function simply sums those returns. Nothing is counted
    that the callable did not do; nothing talks to a live stack from here.

    For MemoryAgentBench scenarios, the scenario's bound question sessions are
    ingested (the scenario's ``history`` and its question's sessions are the
    same turns by the pinned grouping contract).
    """
    _require_dataset(config.dataset_path)
    count = 0
    limit = config.limit
    selected = instances[: limit] if limit is not None else instances
    for instance in selected:
        sessions: tuple[BenchmarkSession, ...]
        if isinstance(instance, Scenario):
            sessions = instance.questions[0].sessions
        else:
            sessions = instance.sessions
        for session in sessions:
            count += await ingest_history(session)
    return count


async def run_score(config: RunnerConfig, results: list[dict]) -> dict:
    """Aggregate judged per-question results into a hygiene-compliant summary.

    The returned dict carries the aggregated ``{"mean", "questions"}`` plus the
    dataset digest (``dataset_sha256``) and benchmark name, ready for
    ``write_results``. MemoryAgentBench records are interpreted per
    competency via :func:`interpret_result` (pending interpretations are
    excluded from the mean and reported separately — never silently scored).
    The remaining hygiene rules (committed judge prompt, full-context
    baseline, protocol deviations) are surfaced as explicit null/empty
    placeholders until real runs fill them — never silently omitted.
    """
    sha = record_sha256(config.dataset_path)
    interpreted: list[dict] = []
    pending: list[dict] = []
    for record in results:
        if "recalled_after_forget" in record:
            verdict = interpret_result(
                record.get("competency", ""),
                record["recalled_after_forget"],
                stale_fact_still_present=bool(record.get("stale_fact_still_present", False)),
            )
            if verdict == "pending_interpretation":
                pending.append({"question_id": record["question_id"], "competency": record.get("competency", "")})
            else:
                interpreted.append({"question_id": record["question_id"], "correct": verdict == "pass"})
        else:
            interpreted.append({"question_id": record["question_id"], "correct": record.get("correct") is True})
    summary = aggregate(interpreted)
    summary.update(
        {
            "benchmark": config.benchmark,
            "dataset_sha256": sha,
            "pending_interpretation": pending,
            "judge_prompt_version": None,
            "full_context_baseline": None,
            "deviations": [],
        }
    )
    return summary


def write_results(out: Path, payload: dict) -> None:
    """Write the results payload as JSON, creating parent directories."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
