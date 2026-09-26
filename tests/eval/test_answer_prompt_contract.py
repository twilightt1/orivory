"""The answerer contract, pinned by the probe that produced it.

Each test guards a measured finding; none of them guard a preference.
The evidence is the wrong-only probe over the 38 failed questions of the
0.62 run (artifact longmemeval_s_system_n100_20260925T174857.json).
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The harness resolves model config at import time and reads the key when it
# builds its client; CI has neither. The stub client below means these are
# placeholders, never a live call.
os.environ.setdefault("LLM_MODEL", "test-model")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from eval import run_system_benchmark as run  # noqa: E402


# eval/benchmarks/data/ is gitignored, so CI has no dataset at all. The checks
# that read it assert properties of the real question distribution, so they run
# where a loadable dataset exists and skip where one does not. Existence alone
# is not enough — an empty or placeholder file loads to zero instances, which
# would fail the assertions instead of skipping them.
def _dataset_is_loadable() -> bool:
    try:
        from eval.benchmarks.longmemeval_s import load_instances

        return bool(load_instances(run.DATASET))
    except Exception:
        return False


needs_dataset = pytest.mark.skipif(
    not _dataset_is_loadable(), reason="benchmark dataset not loadable"
)


def _instance(question: str, question_date: str = "2023/08/25 (Fri) 12:00"):
    return types.SimpleNamespace(question=question, question_date=question_date)


class _Client:
    """Records the single completion request it is handed."""

    def __init__(self):
        self.kwargs: dict | None = None
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    async def _create(self, **kwargs):
        self.kwargs = kwargs
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]
        )


@pytest.fixture
def answering(monkeypatch):
    """Run answer_from_stack against a stub client; return the recorded request."""
    client = _Client()

    async def recall(_user_id, _question, _top_k, fuse=False):
        assert fuse is False
        return [{"captured_at": "2023-05-01T00:00:00+00:00", "content": "filler"}]

    monkeypatch.setattr(run, "stack_recall", recall)
    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(AsyncOpenAI=lambda **_kwargs: client),
    )

    def ask(question: str) -> dict:
        asyncio.run(run.answer_from_stack("u", _instance(question), top_k=15))
        assert client.kwargs is not None
        return client.kwargs

    return ask


def test_answer_call_uses_the_measured_token_cap(answering):
    """300 truncated reasoning models to content=None — 5 empty answers."""
    assert answering("What did I buy?")["max_tokens"] == run.ANSWER_MAX_TOKENS == 2048


def test_answer_prompt_never_teaches_refusal(answering):
    """16 of 21 answerable questions hit the old 'I have no information' branch."""
    system = answering("What did I buy?")["messages"][0]["content"]
    assert "I have no information about that" not in system
    assert "never reply that you have no information" in system


def test_relative_time_question_receives_the_reference_date(answering):
    """Temporal subset: 1/9 without the date, 6/9 with it (discordant 5-0)."""
    user_turn = answering("How long ago did I buy the car?")["messages"][1]["content"]
    assert "TODAY'S DATE: 2023/08/25" in user_turn


@pytest.mark.parametrize(
    "question",
    ["What did I buy?", "Which gym am I a member of?", "How much did the handbag cost?"],
)
def test_plain_question_gets_no_reference_date(answering, question):
    """Unconditional date measured 9/21 vs 12/21 on non-temporal questions."""
    assert "TODAY'S DATE" not in answering(question)["messages"][1]["content"]


@needs_dataset
def test_date_cue_matches_every_question_the_probe_showed_it_helping():
    """5 of the 6 date-hint wins were 'how many days/weeks ago' questions.

    A cue that stopped covering them would silently lose the only effect in
    this change that has a p-value.
    """
    from eval.benchmarks.longmemeval_s import load_instances

    helped = [
        i
        for i in load_instances(run.DATASET)
        if run.RELATIVE_TIME_CUE.search(i.question)
        and i.question.lower().startswith(("how many days ago", "how many weeks ago"))
    ]
    assert len(helped) >= 5, f"dataset no longer contains the helped shape: {len(helped)}"


def test_date_cue_skips_ordering_questions_that_need_no_reference_date():
    """'Which happened first' is decided by memory dates, not by today."""
    assert not run.RELATIVE_TIME_CUE.search(
        "Which device did I set up first, the thermostat or the mesh network?"
    )


@needs_dataset
def test_stack_metadata_reports_the_answerer_it_actually_used():
    """A run must not record max_tokens=300 while calling with 2048."""
    meta = run.build_stack_metadata(top_k=15, sample_seed=20260906)
    assert meta["answer"]["max_tokens"] == run.ANSWER_MAX_TOKENS
    assert meta["answer"]["prompt_version"] == run.ANSWER_PROMPT_VERSION
    assert meta["answer"]["passes_reference_date"] is True
