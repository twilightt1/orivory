from types import SimpleNamespace

import pytest

from app.agents.llm_parsing import parse_llm_json_object
from app.retrieval import embedder, retrieval_cache
from app.retrieval.bm25_retriever import BM25Retriever

pytestmark = pytest.mark.rag

@pytest.mark.asyncio
async def test_bm25_lazy_ensure_rebuilds_missing_index(monkeypatch):
    retriever = BM25Retriever()

    async def fake_rebuild_async(db, conversation_id: str) -> None:
        retriever.build_from_parents(
            conversation_id,
            [
                {
                    "id": "parent-1",
                    "content": "API keys can be rotated from account settings.",
                    "metadata": {"filename": "api.md"},
                },
                {
                    "id": "parent-2",
                    "content": "Invoices are available from the billing dashboard.",
                    "metadata": {"filename": "billing.md"},
                },
                {
                    "id": "parent-3",
                    "content": "Webhook retries use exponential backoff.",
                    "metadata": {"filename": "webhooks.md"},
                }
            ],
        )

    monkeypatch.setattr(retriever, "rebuild_async", fake_rebuild_async)

    # Hermetic: pin the shared generation to "unavailable" so the result
    # doesn't depend on leftover Redis state from other tests/runs.
    async def no_remote_gen(conversation_id: str) -> int | None:
        return None

    monkeypatch.setattr(retriever, "_read_generation_async", no_remote_gen)

    result = await retriever.ensure_async(db=object(), conversation_id="conv-1")

    assert result == {
        "had_index": False,
        "rebuilt": True,
        "stale": False,
        "has_index": True,
    }
    hits = await retriever.search("rotated API keys", top_k=3, conversation_id="conv-1")
    assert hits
    assert hits[0]["parent_id"] == "parent-1"

@pytest.mark.asyncio
async def test_retrieval_cache_invalidation_removes_conversation_keys(monkeypatch):
    class FakeRedis:
        def __init__(self):
            self.keys = {
                "rag:query:conv:conv-1:a": "cached-a",
                "rag:query:conv:conv-1:b": "cached-b",
                "rag:query:conv:conv-2:c": "cached-c",
            }

        async def scan(self, cursor=0, match=None, count=100):
            prefix = match.removesuffix("*")
            keys = [key for key in self.keys if key.startswith(prefix)]
            return 0, keys

        async def delete(self, *keys):
            deleted = 0
            for key in keys:
                if key in self.keys:
                    deleted += 1
                    del self.keys[key]
            return deleted

    fake_redis = FakeRedis()

    async def fake_get_redis():
        return fake_redis

    monkeypatch.setattr(retrieval_cache, "get_redis", fake_get_redis)

    deleted = await retrieval_cache.invalidate_query_cache("conv-1")

    assert deleted == 2
    assert set(fake_redis.keys) == {"rag:query:conv:conv-2:c"}

@pytest.mark.asyncio
async def test_embed_texts_batches_and_preserves_order(monkeypatch):
    calls: list[list[str]] = []

    async def fake_create(model, input, encoding_format, timeout):
        assert encoding_format == "float"
        calls.append(list(input))
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(text[-1])]) for text in input]
        )

    class FakeEmbeddings:
        create = staticmethod(fake_create)

    class FakeAsyncClient:
        embeddings = FakeEmbeddings()

    monkeypatch.setattr(embedder.settings, "EMBED_BATCH_SIZE", 2)
    monkeypatch.setattr(embedder.settings, "USE_LOCAL_EMBEDDINGS", False)
    monkeypatch.setattr(embedder, "_get_async_client", lambda: FakeAsyncClient())

    embeddings = await embedder.embed_texts(["text-1", "text-2", "text-3"])

    assert calls == [["text-1", "text-2"], ["text-3"]]
    assert embeddings == [[1.0], [2.0], [3.0]]

def test_llm_json_parser_handles_fenced_json_and_none():
    parsed = parse_llm_json_object('```json\n{"score": "yes"}\n```')

    assert parsed.ok is True
    assert parsed.data == {"score": "yes"}

    empty = parse_llm_json_object(None)

    assert empty.ok is False
    assert empty.error == "empty_response"
    assert empty.raw_preview is None


@pytest.mark.parametrize(
    "raw",
    [
        '```text\nhello\n```\n{"a": 1}',        # an earlier NON-JSON fence
        '```\ncode { }\n```\n{"a": 1}',          # an earlier EMPTY-object fence
        'Sure!\n{"a": 1}',                       # plain prose
    ],
)
def test_llm_json_parser_does_not_commit_to_the_first_fence(raw):
    """The prompt asks for ONE object; an early fence that is not the payload
    (a transcript, a shell sample, an empty `{}`) must not shadow the real
    JSON that follows it — the graph builder's entity list lives in there."""
    parsed = parse_llm_json_object(raw)

    assert parsed.ok is True
    assert parsed.data == {"a": 1}


def test_llm_json_parser_prefers_the_labelled_json_fence():
    parsed = parse_llm_json_object('intro {"note": "not the payload"}\n```json\n{"a": 1}\n```')

    assert parsed.data == {"a": 1}


def test_llm_json_parser_still_reports_unparseable_responses():
    parsed = parse_llm_json_object("```json\n{not json at all}\n```")

    assert parsed.ok is False
    assert (parsed.error or "").startswith("invalid_json")


def test_rrf_output_has_stable_id_and_best_child():
    """Fused chunks must carry a stable 'id' — CRAG indexes doc['id'] directly
    (regression: retrieval chunks had parent_id but no id -> KeyError)."""
    from app.retrieval.hybrid_retriever import reciprocal_rank_fusion

    bm25 = [
        {"content": "alpha long content", "parent_id": "p1", "score": 0.9},
        {"content": "beta long content", "parent_id": "p2", "score": 0.5},
    ]
    vector = [
        {"content": "beta long content", "parent_id": "p2", "score": 0.95},
        {"content": "alpha long content", "parent_id": "p1", "score": 0.4},
    ]
    fused = reciprocal_rank_fusion([bm25, vector])
    by_id = {d["id"]: d for d in fused}
    assert {"p1", "p2"} <= set(by_id)
    # Best-ranked child kept per parent: p1 ranks better in bm25 (rank 0),
    # p2 ranks better in vector (rank 0)
    assert by_id["p1"]["score"] == 0.9
    assert by_id["p2"]["score"] == 0.95
    # Chunks without a parent_id fall back to a deterministic content hash id
    orphan = [{"content": "orphan text", "score": 1.0}]
    fused2 = reciprocal_rank_fusion([orphan])
    assert fused2[0]["id"] and len(fused2[0]["id"]) == 32
