"""The graph-build switch must not outlive the run that set it.

`_apply_graph_build_switch` rebinds `write_back.safe_enqueue_graph_build` on a
module the whole process shares. A run that turns the switch off and exits
without turning it back left the no-op installed: every later test in the same
pytest process saw a write path that silently never built a graph.
`pytest tests/eval tests/retrieval/test_graph_build_offload.py` failed 5 that way.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.run_system_benchmark import graph_builds  # noqa: E402


def test_the_switch_restores_the_original_callable():
    from app.retrieval.memory import write_back
    from eval import run_system_benchmark

    original = write_back.safe_enqueue_graph_build
    with graph_builds(False):
        assert write_back.safe_enqueue_graph_build is run_system_benchmark._skip_graph_build
    assert write_back.safe_enqueue_graph_build is original


def test_the_switch_restores_even_when_the_run_raises():
    from app.retrieval.memory import write_back
    from eval import run_system_benchmark

    original = write_back.safe_enqueue_graph_build
    with pytest.raises(RuntimeError, match="provider exploded"):
        with graph_builds(False):
            raise RuntimeError("provider exploded")
    assert write_back.safe_enqueue_graph_build is original
    assert run_system_benchmark.GRAPH_BUILDS_ENABLED is False


def test_graph_builds_true_keeps_the_real_callable():
    from app.retrieval.memory import write_back

    original = write_back.safe_enqueue_graph_build
    with graph_builds(True):
        assert write_back.safe_enqueue_graph_build is original
    assert write_back.safe_enqueue_graph_build is original


def test_graph_builds_reports_the_policy_it_applied():
    with graph_builds(True) as enabled:
        assert enabled is True
    with graph_builds(False) as enabled:
        assert enabled is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
