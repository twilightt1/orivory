"""Shared LLM client factory for all agents.

One place for: client construction (api key, base URL, timeout, retries),
a request-scope cost hook that feeds `cost_helpers.record_cost`, and a
uniform `complete()` wrapper that records usage automatically.

Every agent used to hand-roll its own module-level `AsyncOpenAI` singleton
(12 copies) with no timeout and no cost recording — a hung OpenRouter call
stalled an SSE stream for the SDK default 600s, and a single turn firing
up to ~40 LLM calls had zero spend observability.
"""
from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from app.config import settings

log = logging.getLogger(__name__)

# Per-attempt LLM call timeout. Deliberately tighter than the SDK default of
# 600s: the answer path chains up to ~10 serial stages, and one hung call
# must not stall an SSE stream for 10 minutes.
DEFAULT_LLM_TIMEOUT_SECONDS = 60.0
# Free-tier OpenRouter models 429 under burst; the SDK backs off per retry.
DEFAULT_LLM_MAX_RETRIES = 3

_client: AsyncOpenAI | None = None


def _is_unsupported_feature_error(exc: Exception) -> bool:
    """Detect provider 400s that mean 'this model lacks structured outputs'."""
    text = str(exc).lower()
    markers = (
        "does not support feature",
        "structured-outputs",
        "response_format",
        "invalid_request_body",
    )
    return "400" in text and any(m in text for m in markers)


class _ResilientCompletions:
    """Wrapper around ``chat.completions`` adding provider-error fallbacks.

    Every agent shares one AsyncOpenAI client, so wrapping here fixes all
    ~20 call sites at once. Handles:
      * structured-outputs 400s → retry without ``response_format`` (with an
        app-default token budget, since reasoning-style models would truncate
        mid-CoT under a small per-agent cap)
    Concurrency gating and 429 retries live in the SDK + semaphore below.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def _strip_rf_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs["response_format"] = None
        # Reasoning models burn tokens on CoT before content; a small
        # per-agent cap (grader: 500) truncates to empty text on fallback.
        kwargs["max_tokens"] = max(int(kwargs.get("max_tokens") or 0), settings.LLM_MAX_TOKENS)
        return kwargs

    async def create(self, **kwargs: Any) -> Any:
        try:
            return await self._inner.create(**kwargs)
        except Exception as exc:
            if kwargs.get("response_format") and _is_unsupported_feature_error(exc):
                kwargs = self._strip_rf_kwargs(kwargs)
                return await self._inner.create(**kwargs)
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class ResilientAsyncOpenAI:
    """Duck-typed AsyncOpenAI whose ``chat.completions`` is resilient."""

    def __init__(self, inner: AsyncOpenAI) -> None:
        self._inner = inner
        self.chat = type("Chat", (), {"completions": _ResilientCompletions(inner.chat.completions)})()
        self._llm_gate = _get_llm_semaphore()

    async def __aenter__(self) -> ResilientAsyncOpenAI:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


_client_loop: asyncio.AbstractEventLoop | None = None


def get_llm_client() -> AsyncOpenAI:
    """Return the shared configured client (OpenRouter by default).

    Returns a ResilientAsyncOpenAI duck-type: transparent to callers, but
    every ``chat.completions.create`` gains the structured-outputs fallback.

    The cached httpx pool is bound to the event loop that created it, so the
    client is rebuilt whenever the running loop changes. Without this, sync
    contexts that drive one coroutine per ``asyncio.run()`` (Celery graph
    tasks) reuse a pool bound to an already-closed loop from the second task
    on — every call raises and extraction silently degrades to fallback.
    """
    global _client, _client_loop
    try:
        running_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    if _client is None or (
        running_loop is not None
        and _client_loop is not None
        and running_loop is not _client_loop
    ):
        # Drop the stale pool without awaiting its close: its loop is either
        # closed already or belongs to another task — either way we cannot
        # cleanly shut it down here, and abandoning beats reusing poison.
        # Empty-string keys count as missing: an exported-but-empty
        # OPENROUTER_API_KEY must not shadow a working OPENAI_* fallback
        # (this silently killed query-rewrite in every run that sourced
        # a .env with the OpenRouter line left blank).
        api_key = settings.OPENROUTER_API_KEY or settings.OPENAI_API_KEY
        base_url = settings.OPENROUTER_BASE_URL
        if not settings.OPENROUTER_API_KEY and settings.OPENAI_API_KEY:
            base_url = os.environ.get("OPENAI_BASE_URL") or base_url
        _client = ResilientAsyncOpenAI(
            AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=DEFAULT_LLM_TIMEOUT_SECONDS,
                max_retries=DEFAULT_LLM_MAX_RETRIES,
                default_headers={
                    "HTTP-Referer": settings.FRONTEND_URL,
                    "X-Title": "Orivory",
                },
            )
        )
        _client_loop = running_loop
    return _client


# App-wide gate on concurrent provider calls. The RAG pipeline fans out
# (router + rewriter + N parallel graders + answer), and free-tier models
# reject bursts with 429 — serializing through a small semaphore trades a
# little latency for a much higher success rate.
_llm_semaphore: asyncio.Semaphore | None = None


def _get_llm_semaphore() -> asyncio.Semaphore:
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(max(1, settings.LLM_MAX_CONCURRENCY))
    return _llm_semaphore


def _is_unsupported_feature_error(exc: Exception) -> bool:
    """Detect provider 400s that mean 'this model lacks structured outputs'."""
    text = str(exc).lower()
    markers = (
        "does not support feature",
        "structured-outputs",
        "response_format",
        "invalid_request_body",
    )
    return any(m in text for m in markers) and "400" in text


def _usage_to_tokens(usage: Any) -> tuple[int, int]:
    """Best-effort extraction of (tokens_in, tokens_out) from a usage object."""
    if usage is None:
        return 0, 0
    tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
    tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
    if not tokens_in and not tokens_out:
        total = int(getattr(usage, "total_tokens", 0) or 0)
        tokens_in = total
    return tokens_in, tokens_out


def record_usage(
    state: dict[str, Any] | None,
    agent: str,
    model: str,
    usage: Any,
) -> None:
    """Record token usage/cost for a completion into AgentState + the ledger.

    Fire-and-forget by design: cost tracking must never break a request.
    """
    if state is None:
        return
    tokens_in, tokens_out = _usage_to_tokens(usage)
    if not tokens_in and not tokens_out:
        return
    try:
        from app.agents.cost_helpers import record_cost

        record_cost(
            agent=agent,
            state=state,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
    except Exception:  # pragma: no cover — observability must not break requests
        log.debug("Cost recording failed", exc_info=True)


async def complete(
    *,
    agent: str,
    state: dict[str, Any] | None = None,
    model: str | None = None,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int | None = None,
    response_format: dict[str, str] | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: float | None = None,
) -> Any:
    """Run a chat completion with usage/cost recording built in.

    Returns the raw completion object (callers read `.choices[0].message`).
    """
    client = get_llm_client()
    kwargs: dict[str, Any] = dict(
        model=model or settings.LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
        response_format=response_format,
        extra_headers=extra_headers,
        timeout=timeout or DEFAULT_LLM_TIMEOUT_SECONDS,
    )
    async with _get_llm_semaphore():
        try:
            response = await client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Some providers/models (e.g. ling, deepseek-reasoner variants)
            # reject `response_format` with a 400 "does not support feature:
            # structured-outputs". Fall back to a plain call — the agents'
            # prompts already demand JSON, and parse_llm_json_object is
            # tolerant of fenced/messy output.
            if response_format and _is_unsupported_feature_error(exc):
                kwargs["response_format"] = None
                # Reasoning-style models burn the token budget on CoT before
                # emitting content; a small per-agent cap (e.g. 500 for the
                # grader) would truncate mid-reasoning and yield empty text.
                # Give the fallback the app-default budget instead.
                kwargs["max_tokens"] = max(
                    int(kwargs["max_tokens"] or 0), settings.LLM_MAX_TOKENS
                )
                async with _get_llm_semaphore():
                    response = await client.chat.completions.create(**kwargs)
            else:
                raise
    record_usage(state, agent, model or settings.LLM_MODEL, getattr(response, "usage", None))
    return response


async def complete_stream(
    *,
    agent: str,
    state: dict[str, Any] | None = None,
    model: str | None = None,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: float | None = None,
) -> AsyncIterator[Any]:
    """Stream a chat completion, recording usage at the end of the stream.

    Usage chunks are only emitted when the request asks for them; the final
    chunk (if any) carries `usage`.
    """
    client = get_llm_client()
    async with _get_llm_semaphore():
        stream = await client.chat.completions.create(
            model=model or settings.LLM_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
            extra_headers=extra_headers,
            timeout=timeout or DEFAULT_LLM_TIMEOUT_SECONDS,
            stream=True,
        )
    last_usage: Any = None
    try:
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                last_usage = chunk.usage
            yield chunk
    finally:
        if last_usage is not None:
            record_usage(state, agent, model or settings.LLM_MODEL, last_usage)
        close = getattr(stream, "close", None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                try:
                    await result
                except Exception:  # pragma: no cover
                    pass
