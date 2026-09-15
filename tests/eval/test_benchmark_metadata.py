from eval.run_system_benchmark import build_stack_metadata


def test_stack_metadata_reflects_actual_backend(monkeypatch):
    from app.retrieval import embedder

    monkeypatch.setattr(embedder.settings, "USE_LOCAL_EMBEDDINGS", True)
    monkeypatch.setattr(embedder.settings, "LOCAL_EMBED_MODEL", "arctic")
    meta = build_stack_metadata(
        top_k=10,
        selected_question_ids=["q-1"],
        sample_seed=7,
        concurrency=4,
    )
    assert meta["embeddings_actual"]["pooling"] == "cls"
    assert "jina" not in meta["embeddings_actual"]["model_id"].lower()
    assert meta["recall_top_k"] == 10
    assert meta["git_head"] and "git_dirty" in meta
    assert meta["dataset_sha256"] and meta["dataset_path"]
    assert meta["dataset_source"] in {"worktree", "external-checkout-fallback"}
    assert meta["selected_question_ids"] == ["q-1"]
    assert meta["sample_seed"] == 7
    assert meta["runtime"]["python"]
    assert "onnxruntime" in meta["runtime"]["packages"]
    assert meta["rerank"]["top_n"] >= 1
    assert meta["execution"]["requested_concurrency"] == 4
    assert meta["execution"]["actual_concurrency"] == 1
