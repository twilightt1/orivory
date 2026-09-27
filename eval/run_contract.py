"""The run-artifact seam: what a results file means, decided in ONE place.

Two commands write the same artifact — ``run_system_benchmark.py`` produces it
and ``resume_system_run.py`` merges into it — so every rule that decides
"may this run be resumed / what does it add up to" lives here instead of being
re-derived on both sides:

* **Compatibility** is default-deny. Every key of the recorded stack is
  compared against the current one EXCEPT the explicit provenance paths
  (``PROVENANCE_KEYS``), which record where/when a run happened, not what it
  measured. A newly added score-affecting key therefore blocks a resume on its
  own: nobody has to remember to add it to a whitelist (the defect behind
  #94, where a new answer/judge contract had to be taught to two commands).
* **Totals** (``mean``, Wilson 95%, per-type counts, the baseline delta) come
  from one function, so a resumed artifact cannot disagree with a fresh run
  over the same records.
* **Completeness** is a verdict, not a filename convention: the run writes it,
  the resume path reads it, and the ``_partial`` suffix stops being a signal.
"""
from __future__ import annotations

import math
from typing import Any

#: ``stack`` paths that are provenance, not measurement. Default-deny: every
#: OTHER key must match before a resume is allowed, so a new score-affecting
#: key fails closed without anyone remembering to list it.
#:
#: ``git_*``/``runtime``/``dataset_path``/``dataset_source``/``sample_seed``/
#: ``write_index_costs`` say where and when the run happened; ``timeouts_seconds``
#: and the non-policy ``execution`` fields describe the harness, not the
#: retrieval result. The score-bearing fields deliberately are NOT here:
#: ``recall_top_k``, ``rerank``, ``retrieval``, ``answer``, ``judge``,
#: ``graph_builds``, ``execution.context_policy``, ``embedding*``,
#: ``query/passage_prefix``.
#:
#: ``dataset_sha256`` is also absent on purpose: a DIFFERENT dataset is a
#: different measurement, so it must block a resume even though the path it
#: came from is provenance.
PROVENANCE_KEYS = frozenset({
    "git_head",
    "git_dirty",
    "runtime",
    "dataset_path",
    "dataset_source",
    "sample_seed",
    "write_index_costs",
    "timeouts_seconds",
    "execution.requested_concurrency",
    "execution.actual_concurrency",
    "execution.thread_limits",
    "execution.warmup_performed",
    "execution.cache_state",
    "execution.rewrite_policy",
})

#: The graph-build policies this harness knows how to APPLY, not just compare.
#: A recorded value outside this set is refused: resuming under a policy the
#: resume path cannot restore would silently re-answer a different run.
GRAPH_BUILD_POLICIES = frozenset({"on", "off (bench ingest)"})

_CONTEXT_POLICY_KEYS = ("session_level", "chunk_chars", "fuse")


class _Missing:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<missing>"


_MISSING = _Missing()


def _value_at(stack: object, path: str) -> Any:
    """Read a dotted path out of nested dicts, else the ``_MISSING`` sentinel."""
    current: Any = stack
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _graph_builds(stack: object) -> Any:
    return stack.get("graph_builds") if isinstance(stack, dict) else None


def _measured(stack: dict[str, Any]) -> dict[str, Any]:
    """The stack minus its provenance paths.

    A flat key (``runtime``) is dropped whole; a container that holds both
    provenance and measurement (``execution``) is kept with just its measured
    children, so a NEW key inside it is still compared — default-deny survives
    the nesting.
    """
    measured: dict[str, Any] = {}
    for key, value in stack.items():
        if key in PROVENANCE_KEYS:
            continue
        nested = [path for path in PROVENANCE_KEYS if path.startswith(f"{key}.")]
        if nested and isinstance(value, dict):
            stripped = {
                sub: sub_value
                for sub, sub_value in value.items()
                if f"{key}.{sub}" not in PROVENANCE_KEYS
            }
            if stripped:
                measured[key] = stripped
            continue
        measured[key] = value
    return measured


def _is_graph_build_policy(policy: Any) -> bool:
    """A policy we can both compare AND re-apply. A list/dict recorded where a
    policy belongs is a corrupt artifact, not a match: ``in`` on a frozenset
    would raise on an unhashable value."""
    return isinstance(policy, str) and policy in GRAPH_BUILD_POLICIES


def run_contracts_match(recorded: object, current: object) -> bool:
    """True when a run recorded under ``recorded`` may be resumed under ``current``.

    Default-deny over the whole recorded ``stack``: every key is compared
    unless it is an explicit provenance path. A key present on one side only is
    a mismatch (an older artifact missing a new field is a different
    measurement), and a non-dict on either side is never compatible.
    """
    if not isinstance(recorded, dict) or not isinstance(current, dict):
        return False
    if not _is_graph_build_policy(_graph_builds(recorded)):
        return False
    if not _is_graph_build_policy(_graph_builds(current)):
        return False
    if _value_at(recorded, "execution.context_policy") is _MISSING:
        return False
    if _value_at(current, "execution.context_policy") is _MISSING:
        return False
    return _measured(recorded) == _measured(current)


def recorded_graph_builds(stack: object) -> str | None:
    """The recorded graph-build policy, or None when it is absent/malformed."""
    policy = _graph_builds(stack)
    return policy if _is_graph_build_policy(policy) else None


def recorded_context_policy(stack: object) -> dict[str, Any] | None:
    """The recorded context policy as a complete typed triple, else None.

    ``chunk_chars`` must be a real int (``True`` is not one) and the two flags
    real bools: a policy the resume path would re-run under as a string or a
    truthy int is one nobody measured with.
    """
    policy = _value_at(stack, "execution.context_policy")
    if not isinstance(policy, dict):
        return None
    session_level, chunk_chars, fuse = (policy.get(key) for key in _CONTEXT_POLICY_KEYS)
    if not isinstance(session_level, bool) or not isinstance(fuse, bool):
        return None
    if isinstance(chunk_chars, bool) or not isinstance(chunk_chars, int):
        return None
    return {"session_level": session_level, "chunk_chars": chunk_chars, "fuse": fuse}


def recorded_recall_top_k(stack: object) -> int | None:
    """The recorded ``recall_top_k``, or None when it is not a positive int."""
    top_k = _value_at(stack, "recall_top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        return None
    return top_k


def wilson_95(correct: int, total: int) -> list[float] | None:
    """Wilson score interval at 95% — one formula, both commands."""
    if total <= 0:
        return None
    z = 1.96
    p = correct / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [round(center - half, 3), round(center + half, 3)]


def run_verdict(
    records: list[dict],
    *,
    expected_count: int | None = None,
    recorded_partial: bool | None = None,
    purge_failed: bool = False,
    from_index: int | None = None,
    forced: list[int] | None = None,
    baseline_mean: Any = None,
) -> dict[str, Any]:
    """What a record set adds up to, and whether it still owes a re-run.

    ``recorded_partial`` is the flag the artifact carried: ``True``/``False``
    when it was written, ``None`` for an artifact that predates the flag — where
    the run died mid-flight is then unknowable, so the LAST record is treated as
    unverified rather than trusted.

    Returns ``failed`` (indices to re-run), ``complete`` (whether this is a
    clean, quotable measurement) and ``totals`` — the same numbers a fresh run
    computes from the same records.
    """
    total = len(records)
    failed: list[int] = [
        i
        for i, record in enumerate(records)
        if isinstance(record, dict)
        and (record.get("memories_recalled") == 0 or bool(record.get("error")))
    ]
    if from_index is not None:
        # ``--from-index`` is "everything from here on is unverified", so it
        # ADDS to the records that already look broken — never replaces it. A
        # dropped row before the cursor still owes a re-run.
        failed = sorted({*failed, *(i for i in range(total) if i >= from_index)})
    for index in forced or ():
        if index not in failed:
            failed.append(index)
    if recorded_partial is None and records and total - 1 not in failed:
        # An artifact that never recorded its own state: the last row may never
        # have been written, so it is re-run rather than quoted.
        failed.append(total - 1)
    failed.sort()

    scored = [r for r in records if isinstance(r, dict) and not r.get("error")]
    correct = sum(1 for r in scored if r["correct"])
    by_type: dict[str, dict[str, int]] = {}
    for record in scored:
        slot = by_type.setdefault(record["question_type"], {"n": 0, "correct": 0})
        slot["n"] += 1
        slot["correct"] += 1 if record["correct"] else 0

    mean = round(correct / len(scored), 3) if scored else 0.0
    complete = not purge_failed and not failed and (
        expected_count is None or total >= expected_count
    )
    delta: float | None = None
    if (
        isinstance(baseline_mean, (int, float))
        and not isinstance(baseline_mean, bool)
        and math.isfinite(baseline_mean)
        and scored
    ):
        delta = round(mean - baseline_mean, 3)
    return {
        "complete": complete,
        "failed": failed,
        "totals": {
            "mean": mean,
            "questions": len(scored),
            "correct": correct,
            "errors": len(records) - len(scored),
            "wilson_95": wilson_95(correct, len(scored)),
            "by_type": by_type,
        },
        "comparison_delta": delta,
    }


__all__ = [
    "GRAPH_BUILD_POLICIES",
    "PROVENANCE_KEYS",
    "recorded_context_policy",
    "recorded_graph_builds",
    "recorded_recall_top_k",
    "run_contracts_match",
    "run_verdict",
    "wilson_95",
]
