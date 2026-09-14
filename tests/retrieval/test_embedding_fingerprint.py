from app.retrieval import embedding_fingerprint as fp


def test_current_fingerprint_distinguishes_arctic_mean_from_cls():
    legacy = fp.LEGACY_MEAN_FINGERPRINT
    assert legacy["pooling"] == "legacy-mean" and legacy["dim"] == 384


def test_cache_key_changes_with_pooling():
    a = {
        "model_id": "Snowflake/snowflake-arctic-embed-xs",
        "pooling": "legacy-mean",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
        "max_tokens": 512,
        "normalize": True,
        "dim": 384,
        "doc_format": "title-content-v1",
    }
    b = dict(a, pooling="cls")
    assert fp.cache_key(a, "query", "hello") != fp.cache_key(b, "query", "hello")
    assert fp.cache_key(a, "query", "hello") == fp.cache_key(a, "query", "hello")
    assert fp.cache_key(a, "query", "hello") != fp.cache_key(a, "passage", "hello")
