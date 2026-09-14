from eval.run_system_benchmark import build_stack_metadata


def test_stack_metadata_reflects_actual_backend(monkeypatch):
    from app.retrieval import embedder

    monkeypatch.setattr(embedder.settings, "USE_LOCAL_EMBEDDINGS", True)
    monkeypatch.setattr(embedder.settings, "LOCAL_EMBED_MODEL", "arctic")
    meta = build_stack_metadata(top_k=10)
    assert meta["embeddings_actual"]["pooling"] == "legacy-mean"
    assert "jina" not in meta["embeddings_actual"]["model_id"].lower()
    assert meta["recall_top_k"] == 10 and "git_head" in meta and "dataset_sha256" in meta
