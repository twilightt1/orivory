"""The committed "current" artifact must still say what the docs say it says.

README and eval/README.md quote a headline score and a retrieval config. A
later run can drift either; this test compares the committed artifact to the
numbers written next to it, so the prose cannot quietly start lying.

Skipped when the artifact is absent (a fresh checkout without the dataset).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "eval/benchmarks/results"
README = ROOT / "eval/README.md"
# The DEFAULT lane's artifact. This moves when the default embedder changes —
# e5 replaced arctic as the default in PR #107, and the doc-claim gate has to
# follow it, or it would police a comparison row instead of the headline.
ARTIFACT = "longmemeval_s_system_n100_20260928T085334.json"


def _current_section() -> str:
    """The prose that claims to describe the current lane, and nothing else."""
    return README.read_text().split("### Current", 1)[-1].split("### Historical", 1)[0]


@pytest.mark.skipif(not (RESULTS / ARTIFACT).is_file(), reason="artifact not present")
def test_the_headline_score_matches_the_committed_artifact():
    payload = json.loads((RESULTS / ARTIFACT).read_text())
    mean, questions, correct = payload["mean"], payload["questions"], payload["correct"]
    lo, hi = payload["wilson_95"]

    section = _current_section()
    assert ARTIFACT in section, f"eval/README.md no longer lists {ARTIFACT}"
    for number in (f"{mean:.3f}", f"[{lo}, {hi}]", f"{correct}/{questions}"):
        assert number in section, (
            f"eval/README.md's current section is missing {number!r} "
            f"that the artifact reports"
        )


@pytest.mark.skipif(not (RESULTS / ARTIFACT).is_file(), reason="artifact not present")
def test_the_quoted_retrieval_config_matches_the_committed_artifact():
    """The stack block is the source of truth; the prose must quote it, not guess."""
    stack = json.loads((RESULTS / ARTIFACT).read_text())["stack"]
    section = _current_section()

    assert f"`recall_top_k {stack['recall_top_k']}`" in section, (
        f"the artifact ran recall_top_k {stack['recall_top_k']}; the prose does not say so"
    )
    policy = stack["execution"]["context_policy"]
    chunking = "per-turn chunking" if not policy["session_level"] else "session-level"
    assert chunking in section, (
        f"the artifact ran {policy}; the prose does not describe that chunking"
    )


def test_no_doc_quotes_a_score_the_results_dir_does_not_hold():
    """A quoted artifact filename must exist."""
    text = README.read_text()
    quoted = set(re.findall(r"benchmarks/results/([\w.\-]+\.json)", text))
    missing = sorted(name for name in quoted if not (RESULTS / name).is_file())
    assert not missing, f"eval/README.md cites results that do not exist: {missing}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
