from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx

from eval.metrics import (
    calculate_fallback_accuracy,
    calculate_keyword_coverage,
    calculate_source_hit,
    has_citation,
    summarize_results,
)
from eval.reporting import build_report, write_json_report, write_markdown_report


def load_dataset(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


class LiveApiEvalError(RuntimeError):
    """Raised when the live API evaluation cannot continue safely."""


@dataclass(frozen=True)
class LiveApiEvalConfig:
    api_base_url: str
    dataset_path: Path
    sample_docs_dir: Path
    output_dir: Path
    email: str | None = None
    password: str | None = None
    access_token: str | None = None
    document_poll_timeout_seconds: float = 120.0
    document_poll_interval_seconds: float = 2.0


@dataclass(frozen=True)
class SseEvent:
    event: str | None
    data: dict[str, Any]


@dataclass(frozen=True)
class CollectedSseResponse:
    answer: str
    sources: list[dict[str, Any]]
    trace: dict[str, Any]
    statuses: list[dict[str, Any]]
    done: dict[str, Any]
    raw_events: list[SseEvent]


def parse_sse_stream(stream_text: str) -> list[SseEvent]:
    """Parse a text/event-stream payload into typed SSE events."""
    events: list[SseEvent] = []
    normalized = stream_text.replace("\r\n", "\n")
    for raw_frame in normalized.strip().split("\n\n"):
        if not raw_frame.strip():
            continue

        event_name: str | None = None
        data_lines: list[str] = []
        for line in raw_frame.splitlines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())

        if not data_lines:
            continue
        try:
            payload = json.loads("\n".join(data_lines))
        except json.JSONDecodeError as exc:
            raise LiveApiEvalError(f"Invalid SSE JSON payload: {data_lines!r}") from exc
        if not isinstance(payload, dict):
            raise LiveApiEvalError("SSE payload must be a JSON object.")
        events.append(SseEvent(event=event_name, data=payload))
    return events


def collect_sse_response(events: Iterable[SseEvent]) -> CollectedSseResponse:
    """Collect token, source, trace, status, and done events from SSE frames."""
    raw_events = list(events)
    tokens: list[str] = []
    statuses: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    trace: dict[str, Any] = {}
    done: dict[str, Any] = {}

    for item in raw_events:
        event_type = item.event or item.data.get("type")
        data = item.data
        if event_type == "error" or data.get("type") == "error":
            raise LiveApiEvalError(str(data.get("message") or "Live chat stream failed."))
        if event_type == "token":
            tokens.append(str(data.get("content", "")))
        elif event_type == "status":
            statuses.append(data)
        elif event_type == "sources":
            sources = list(data.get("sources") or [])
        elif event_type == "trace":
            trace = dict(data.get("agent_trace") or {})
        elif event_type == "done":
            done = dict(data)
            if not sources:
                sources = list(data.get("sources") or [])

    return CollectedSseResponse(
        answer="".join(tokens),
        sources=sources,
        trace=trace,
        statuses=statuses,
        done=done,
        raw_events=raw_events,
    )


def normalize_source_filenames(sources: Iterable[dict[str, Any]]) -> list[str]:
    filenames: list[str] = []
    for source in sources:
        filename = str(source.get("filename") or source.get("source") or "").strip()
        if filename:
            filenames.append(Path(filename).name)
    return filenames


def _recursive_contains_true(value: Any, needle: str) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if needle in str(key).casefold() and bool(nested):
                return True
            if _recursive_contains_true(nested, needle):
                return True
    elif isinstance(value, list):
        return any(_recursive_contains_true(item, needle) for item in value)
    return False


def infer_correction_count(trace: dict[str, Any], done: dict[str, Any]) -> int:
    retry_count = int(done.get("retry_count") or 0)
    correction = trace.get("correction")
    if isinstance(correction, list):
        return max(retry_count, len(correction))
    if isinstance(correction, dict) and correction:
        return max(retry_count, 1)
    if correction:
        return max(retry_count, 1)
    return retry_count


def score_live_response(
    item: dict[str, Any],
    response: CollectedSseResponse,
    latency_ms: float,
) -> dict[str, Any]:
    returned_sources = normalize_source_filenames(response.sources)
    source_text = "\n\n".join(str(source.get("content", "")) for source in response.sources)
    scored_text = f"{response.answer}\n\n{source_text}"
    expected_sources = item.get("expected_sources", [])
    source_hit = calculate_source_hit(returned_sources, expected_sources)
    keyword_coverage = calculate_keyword_coverage(scored_text, item.get("expected_keywords", []))
    fallback_accuracy = calculate_fallback_accuracy(item.get("should_fallback", False), response.answer)
    citation_present = has_citation(response.answer, returned_sources)
    correction_count = infer_correction_count(response.trace, response.done)
    hallucination_flagged = _recursive_contains_true(response.trace, "hallucination")

    passed = (
        source_hit >= 1.0
        and keyword_coverage >= 0.75
        and fallback_accuracy >= 1.0
        and (citation_present or item.get("should_fallback", False))
    )

    return {
        "id": item["id"],
        "query": item["query"],
        "category": item["category"],
        "expected_sources": expected_sources,
        "returned_sources": returned_sources,
        "expected_keywords": item.get("expected_keywords", []),
        "should_fallback": item.get("should_fallback", False),
        "answer": response.answer,
        "source_hit": source_hit,
        "keyword_coverage": keyword_coverage,
        "has_citation": citation_present,
        "fallback_accuracy": fallback_accuracy,
        "latency_ms": latency_ms,
        "hallucination_flagged": hallucination_flagged,
        "correction_count": correction_count,
        "passed": passed,
    }


class LiveApiEvaluator:
    def __init__(self, config: LiveApiEvalConfig):
        self.config = config
        self.token = config.access_token

    async def run(self) -> dict[str, Any]:
        async with httpx.AsyncClient(
            base_url=self.config.api_base_url.rstrip("/"),
            timeout=httpx.Timeout(60.0, connect=10.0),
        ) as client:
            if not self.token:
                self.token = await self._login(client)
            conversation_id = await self._create_conversation(client)
            document_ids = await self._upload_documents(client, conversation_id)
            await self._wait_for_documents(client, conversation_id, document_ids)
            results = await self._evaluate_dataset(client, conversation_id)

        summary = summarize_results(results)
        report = build_report(
            results=results,
            summary=summary,
            metadata={
                "mode": "live-api",
                "api_base_url": self.config.api_base_url,
                "dataset": str(self.config.dataset_path),
                "sample_docs": str(self.config.sample_docs_dir),
            },
        )
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        write_json_report(report, self.config.output_dir / "live_api_report.json")
        write_markdown_report(report, self.config.output_dir / "live_api_report.md")
        return report

    def _headers(self) -> dict[str, str]:
        if not self.token:
            raise LiveApiEvalError("Missing access token for authenticated API request.")
        return {"Authorization": f"Bearer {self.token}"}

    async def _login(self, client: httpx.AsyncClient) -> str:
        # Account auth is gone: there is no login endpoint to call, and a
        # request with no token is already the local owner. A live run only
        # needs a token when it must be scoped/audited as an agent client.
        raise LiveApiEvalError(
            "This install has no account login any more. Pass --access-token "
            "with an agent token (POST /api/v1/agents), or omit it to run as "
            "the local owner."
        )

    async def _create_conversation(self, client: httpx.AsyncClient) -> str:
        response = await client.post(
            "/api/v1/chat/conversations",
            headers=self._headers(),
            json={"title": "Live API Evaluation"},
        )
        response.raise_for_status()
        return str(response.json()["id"])

    async def _upload_documents(self, client: httpx.AsyncClient, conversation_id: str) -> list[str]:
        document_ids: list[str] = []
        for path in sorted(self.config.sample_docs_dir.glob("*.md")):
            with path.open("rb") as file_handle:
                response = await client.post(
                    f"/api/v1/chat/conversations/{conversation_id}/documents",
                    headers=self._headers(),
                    files={"file": (path.name, file_handle, "text/plain")},
                )
            response.raise_for_status()
            document_ids.append(str(response.json()["id"]))
        if not document_ids:
            raise LiveApiEvalError(f"No markdown files found in {self.config.sample_docs_dir}")
        return document_ids

    async def _wait_for_documents(
        self,
        client: httpx.AsyncClient,
        conversation_id: str,
        document_ids: list[str],
    ) -> None:
        deadline = perf_counter() + self.config.document_poll_timeout_seconds
        pending = set(document_ids)
        while pending and perf_counter() < deadline:
            for document_id in list(pending):
                response = await client.get(
                    f"/api/v1/chat/conversations/{conversation_id}/documents/{document_id}",
                    headers=self._headers(),
                )
                response.raise_for_status()
                payload = response.json()
                status = payload.get("status")
                if status == "ready":
                    pending.remove(document_id)
                elif status == "failed":
                    raise LiveApiEvalError(
                        f"Document {document_id} ingestion failed: {payload.get('error_msg')}"
                    )
            if pending:
                await asyncio.sleep(self.config.document_poll_interval_seconds)
        if pending:
            raise LiveApiEvalError(f"Timed out waiting for documents: {sorted(pending)}")

    async def _evaluate_dataset(self, client: httpx.AsyncClient, conversation_id: str) -> list[dict[str, Any]]:
        dataset = load_dataset(self.config.dataset_path)
        results: list[dict[str, Any]] = []
        for item in dataset:
            started = perf_counter()
            response = await client.post(
                f"/api/v1/chat/conversations/{conversation_id}/message",
                headers=self._headers(),
                json={"query": item["query"]},
            )
            response.raise_for_status()
            collected = collect_sse_response(parse_sse_stream(response.text))
            latency_ms = (perf_counter() - started) * 1000
            results.append(score_live_response(item, collected, latency_ms))
        return results


def run_live_api_evaluation(config: LiveApiEvalConfig) -> dict[str, Any]:
    report = asyncio.run(LiveApiEvaluator(config).run())
    summary = report["summary"]
    print("Orivory Live API RAG Evaluation")
    print("=" * 36)
    print(f"Cases:             {summary['total_cases']}")
    print(f"Source hit rate:   {summary['source_hit_rate']:.1%}")
    print(f"Keyword coverage:  {summary['keyword_coverage']:.1%}")
    print(f"Citation rate:     {summary['citation_rate']:.1%}")
    print(f"Fallback accuracy: {summary['fallback_accuracy']:.1%}")
    print(f"Average latency:   {summary['avg_latency_ms']:.1f} ms")
    print(f"Report:            {config.output_dir / 'live_api_report.md'}")
    return report
