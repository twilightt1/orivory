"""The benchmark entry points must import in a checkout with no ``.env``.

``.env`` is gitignored, so a fresh clone, a CI runner and a bare container all
lack it. Each run script loads ``.env`` when it can and falls through to the
ambient environment when it cannot — but they then read the model name at
module level, so a missing key raised ``KeyError`` during import and the caller
got a traceback instead of a run or a clear message.

The subprocess runs with CWD in an empty directory *and* the model keys removed
from the environment: the scripts locate ``.env`` relative to their own path
(one level above the repo root, for a worktree), so clearing os.environ alone
would still find the developer's file and the bug would stay hidden.

Where this bites: on CI, in a container, and in a fresh clone — anywhere without
a ``.env``. A developer machine that HAS one will see these pass either way,
which is the point: the script finds its config, so the check is silent.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENTRY_POINTS = [
    "eval.run_system_benchmark",
    "eval.run_real_sample",
    "eval.pilot_judged_fixture",
]

_MODEL_KEYS = ("LLM_MODEL", "BENCHMARK_JUDGE_MODEL")


def _import_in_a_bare_checkout(module: str, cwd: Path, extra_env: dict[str, str]) -> str:
    env = {k: v for k, v in os.environ.items() if k not in _MODEL_KEYS}
    env["PYTHONPATH"] = str(ROOT)
    env.update(extra_env)
    result = subprocess.run(
        [sys.executable, "-c", f"import {module} as m; print(m.MODEL)"],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd,
        check=False,
    )
    assert result.returncode == 0, f"{module} failed to import: {result.stderr.strip()}"
    return result.stdout.strip()


@pytest.mark.parametrize("module", ENTRY_POINTS)
def test_an_eval_entry_point_imports_with_no_env(module, tmp_path):
    """Resolves a non-empty model with no ``.env`` anywhere: an empty one would
    fail later, at the API call, instead of naming the problem."""
    assert _import_in_a_bare_checkout(module, tmp_path, {})


@pytest.mark.parametrize("module", ENTRY_POINTS)
def test_an_eval_entry_point_honours_an_explicit_model_override(module, tmp_path):
    model = _import_in_a_bare_checkout(module, tmp_path, {"LLM_MODEL": "sentinel/from-env"})

    assert "sentinel/from-env" in model, f"{module} ignored LLM_MODEL: got {model}"


def test_the_judge_override_still_wins(tmp_path):
    """BENCHMARK_JUDGE_MODEL must keep precedence, or a judge-only run cannot
    be aimed at a different model from the answerer."""
    model = _import_in_a_bare_checkout(
        "eval.run_system_benchmark",
        tmp_path,
        {"LLM_MODEL": "from-env", "BENCHMARK_JUDGE_MODEL": "sentinel/judge"},
    )

    assert "sentinel/judge" in model
