from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.graph import builder, extraction
from app.graph.extraction import ExtractedEntity, ExtractedRelation


def _memory() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        user_id=uuid4(),
        extra_metadata={},
        captured_at=datetime.now(UTC),
    )


def test_sync_builder_uses_sync_extraction_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    memory = _memory()
    calls: list[str] = []

    def extract_entities_sync(value):
        calls.append("entities")
        assert value is memory
        return SimpleNamespace(entities=[], fallback_used=False, error=None)

    def extract_relations_sync(value, entities):
        calls.append("relations")
        assert value is memory
        assert entities == []
        return SimpleNamespace(relations=[], fallback_used=False, error=None)

    class FakeSession:
        def get(self, model, memory_id):
            assert memory_id == memory.id
            return memory

        def flush(self):
            pass

        def commit(self):
            pass

    monkeypatch.setattr(builder, "extract_entities_sync", extract_entities_sync, raising=False)
    monkeypatch.setattr(builder, "extract_relations_sync", extract_relations_sync, raising=False)

    result = builder.build_memory_graph_sync(FakeSession(), memory.id)

    assert calls == ["entities", "relations"]
    assert result.memory_id == str(memory.id)
    assert result.fallback_used is False


def test_sync_entity_extraction_uses_sync_client(monkeypatch: pytest.MonkeyPatch) -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"entities": [{"name": "Atlas"}]}'))]
    )

    class Completions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            return response

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(extraction, "_get_sync_client", lambda: client, raising=False)

    memory = SimpleNamespace(title="Notes", summary="", tags=[], content="Project Atlas")
    result = extraction.extract_entities_sync(memory)

    assert result.fallback_used is False
    assert [entity.name for entity in result.entities] == ["Atlas"]
    assert completions.calls == 1


def test_sync_relation_extraction_uses_sync_client(monkeypatch: pytest.MonkeyPatch) -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        '{"relations":[{"source":"Atlas","target":"Beta",'
                        '"relation":"related_to","weight":0.8}]}'
                    )
                )
            )
        ]
    )

    class Completions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            return response

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(extraction, "_get_sync_client", lambda: client)

    memory = SimpleNamespace(title="Notes", summary="", tags=[], content="Atlas knows Beta")
    entities = [
        ExtractedEntity(name="Atlas"),
        ExtractedEntity(name="Beta"),
    ]
    result = extraction.extract_relations_sync(memory, entities)

    assert result.fallback_used is False
    assert result.relations[0] == ExtractedRelation(
        source="Atlas",
        target="Beta",
        relation="related_to",
        weight=0.8,
    )
    assert completions.calls == 1
