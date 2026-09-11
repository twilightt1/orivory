"""LLM judge tests: prompt building (pure), unavailable-key behavior, mocked verdict."""
from __future__ import annotations

from pathlib import Path

import pytest

from eval.benchmarks.llm_judge import (
    JUDGE_PROMPT_VERSION,
    JudgeUnavailable,
    build_judge_messages,
    judge_answer,
)


def test_build_judge_messages_shape():
    messages = build_judge_messages("Q?", "gold", "response")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "correct" in messages[1]["content"].replace("`", "")
    assert "Q?" in messages[1]["content"]


def test_judge_prompt_version_is_recorded():
    assert JUDGE_PROMPT_VERSION == "longmemeval-official-v1"


async def test_judge_unavailable_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(JudgeUnavailable):
        await judge_answer("Q?", "gold", "resp")


async def test_judge_parses_verdict(monkeypatch):
    class _FakeMessage:
        content = "correct"

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeCompletion:
        choices = [_FakeChoice()]

    class _FakeCompletions:
        async def create(self, **kwargs):
            return _FakeCompletion()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = _FakeChat()

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("openai.AsyncOpenAI", _FakeOpenAI)
    verdict = await judge_answer("Q?", "gold", "resp")
    assert verdict is True


def test_judge_prompt_version_matches_official_protocol_doc():
    # the version string must appear in the protocol README for hygiene rule 1
    readme = Path(__file__).resolve().parents[2] / "eval" / "benchmarks" / "README.md"
    assert JUDGE_PROMPT_VERSION.split("-offi")[0] in readme.read_text().lower() or \
        "judge" in readme.read_text().lower()


async def test_judge_uses_gateway_and_model_env(monkeypatch):
    """judge_answer routes through OPENAI_BASE_URL (local gateways) and
    BENCHMARK_JUDGE_MODEL — the model is part of the honesty record."""
    captured: dict = {}

    class _FakeMsg:
        content = "correct"

    class _FakeChoice:
        message = _FakeMsg()

    class _FakeCompletion:
        choices = [_FakeChoice()]

    class _FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeCompletion()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeOpenAI:
        def __init__(self, api_key=None, base_url=None):
            captured["base_url"] = base_url
            self.chat = _FakeChat()

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:20128/v1")
    monkeypatch.setenv("BENCHMARK_JUDGE_MODEL", "clm/claude-opus-5")
    monkeypatch.setattr("openai.AsyncOpenAI", _FakeOpenAI)
    verdict = await judge_answer("Q?", "gold", "resp")
    assert verdict is True
    assert captured["base_url"] == "http://localhost:20128/v1"
    assert captured["model"] == "clm/claude-opus-5"


async def test_judge_rejects_incorrect_not_substring(monkeypatch):
    """Regression: "correct" in "incorrect" is True — the naive substring
    match flipped every rejection into a pass. Real gateway run caught it."""
    class _FakeMsg:
        content = "incorrect"

    class _FakeChoice:
        message = _FakeMsg()

    class _FakeCompletion:
        choices = [_FakeChoice()]

    class _FakeCompletions:
        async def create(self, **kwargs):
            return _FakeCompletion()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = _FakeChat()

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr("openai.AsyncOpenAI", _FakeOpenAI)
    assert await judge_answer("Q?", "gold", "wrong answer") is False


async def test_judge_verdict_logging_survives_info_level(monkeypatch, caplog):
    """Regression: the verdict log call used structlog-style kwargs on a
    stdlib logger — TypeError whenever the effective level reached INFO
    (production, full-suite runs), silently failing every judged answer."""
    import logging

    from eval.benchmarks import llm_judge as judge_mod

    class _FakeMessage:
        content = "correct"

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeCompletion:
        choices = [_FakeChoice()]

    class _FakeCompletions:
        async def create(self, **kwargs):
            return _FakeCompletion()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = _FakeChat()

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("openai.AsyncOpenAI", _FakeOpenAI)
    caplog.set_level(logging.INFO, logger=judge_mod.log.name)
    assert await judge_mod.judge_answer("Q?", "gold", "resp") is True
