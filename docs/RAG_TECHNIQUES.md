# RAG Techniques Deep-Dive

Implementation notes for the techniques used in Orivory. Each section
links to the source file, explains the trade-off, and points at the tests
that exercise it.

---

## 1. Hybrid retrieval (dense + FTS5 lexical + RRF) — opt-in, SQLite-only

**Why.** A lexical leg excels at exact keywords, code identifiers and Vietnamese
diacritics (`"ORIVORY-4417"`, `"hồ sơ dự án"` typed either way), where a local
embedding model has no idea what the token *means*. Dense retrieval carries the
semantic paraphrases. Fusing the two by Reciprocal Rank Fusion keeps both.

**Status: ships OFF.** `RETRIEVAL_HYBRID_ENABLED=false` by default; only the
T7 ablation artifact passing the signed non-inferiority gate
(`eval/ablation_retrieval_p2.json`; no slice losing > 0.02 recall@5 vs
dense-only, overall gain >= +0.02) may enable it, as a separate documented
decision. On the frozen fixture the flag's arm measured **+0.1042 overall
recall@5** with the entire gain in the `exact_id` slice — evidence at FIXTURE
scale, with a stand-in reranker and index-global BM25 statistics, never a
production claim (read the artifact's `limitations`).

**Where.**
- Lexical leg (memory): [`app/retrieval/memory/lexical_index.py`](../app/retrieval/memory/lexical_index.py) — SQLite FTS5 (`memory_fts`, created by the schema ladder's v4 step; triggers maintain it in the writing transaction). Where no lexical leg exists, a vector outage keeps the typed 503.
- Dense leg: [`app/retrieval/memory/vector_store.py`](../app/retrieval/memory/vector_store.py)
- Fusion: [`app/retrieval/hybrid_retriever.py`](../app/retrieval/hybrid_retriever.py) — `fuse_by_uuid` (UUID-keyed, zero-based ranks); the legacy `reciprocal_rank_fusion` dedupes by parent/content and is **not** used for memory.
- The document/BM25 path ([`app/retrieval/bm25_retriever.py`](../app/retrieval/bm25_retriever.py), Redis parent cache) is a separate legacy leg, not the memory recall's.

**Trade-off.** A second query per recall (the FTS5 page) plus a fusion pass —
in-Python, and only paid when the flag is on. When the vector store is down the
lexical leg answers ALONE (SQLite), which is the one case where the size of
that cost is irrelevant.

**Test.** [`tests/retrieval/test_hybrid_recall.py`](../tests/retrieval/test_hybrid_recall.py)
(fusion order, UUID dedupe, the outage fallback, the OFF path),
[`tests/retrieval/test_p2_ablation_contract.py`](../tests/retrieval/test_p2_ablation_contract.py)
(the artifact's shape), and the §9 gate [`tests/retrieval/test_p2_gate.py`](../tests/retrieval/test_p2_gate.py).

---

## 2. Reciprocal Rank Fusion

```
score(d) = Σ_i  1 / (k + rank_i(d) + 1)     # k = RETRIEVAL_RRF_K (default 60)
```

**Why RRF instead of linear combination.** RRF needs no score
calibration between retrievers (global BM25 and cosine are not comparable). It
also handles "the result only appears in one list" gracefully.

**Memory-specific rules.** Fusion is keyed by the canonical memory UUID
(never content or `parent_id`: two memories with identical text stay two
memories), ranks are ZERO-based, `k >= 1` is validated at config load, and the
legs' own scores travel separately on each candidate — the fused order is the
sum, never a bare cosine (a refill row appended with its raw cosine would
outrank the whole fused head).

**Where.** [`app/retrieval/hybrid_retriever.py`](../app/retrieval/hybrid_retriever.py) — `fuse_by_uuid`.

**Test.** [`tests/retrieval/test_hybrid_recall.py`](../tests/retrieval/test_hybrid_recall.py)
covers single-list inputs, equal rank, the k seam, duplicate-text rows and the
refill re-fusion.

---

## 3. Parent-child chunking (the document path)

```
[Parent (~1500 chars)]  ← stored in DB + Redis, returned to the LLM as context
   ├── [Child (~400 chars)] ← embedded into the vector store
   ├── [Child (~400 chars)]
   └── [Child (~400 chars)]
```

**Why.** Small children → precise embedding (better recall).
Large parents → readable context for the answer agent.

**Where.** [`app/utils/chunker.py`](../app/utils/chunker.py) —
`build_parent_child_chunks` (`PARENT_SIZE` 1500 / `PARENT_OVERLAP` 150,
`CHILD_SIZE` 400 / `CHILD_OVERLAP` 50): each child carries its `parent_id` in
metadata and only children are embedded. Markdown/DOCX headings split first,
a recursive character split is the fallback.

**Trade-off.** Storage is duplicated by design — the parent is one row in
`document_chunks` (plus the Redis cache, §8) and its children are the points
in the vector store.

**Test.** [`tests/rag/test_chunker.py`](../tests/rag/test_chunker.py).

---

## 4. Query rewrite (recall) + HyDE (document path)

Two transformations ship, and neither fans out into query variants:

1. **Conversation rewrite — the recall path.** ONE LLM call
   ([`app/retrieval/memory/query_rewriter.py`](../app/retrieval/memory/query_rewriter.py) —
   `rewrite_query`) resolves pronouns, abbreviations and implicit references
   against the user's recent + pinned memories, and returns a self-contained
   query plus the entities it resolved. It is best-effort — any LLM error keeps
   the original query (`_fallback_used` on the result) — and the recall path
   SKIPS the call entirely for queries with no pronouns
   ([`app/retrieval/memory/correction.py`](../app/retrieval/memory/correction.py) —
   `needs_rewrite`, `rewrite_skipped` on the trace).
2. **HyDE — the document path.** [`app/retrieval/hyde_agent.py`](../app/retrieval/hyde_agent.py)
   generates hypothetical passages (Gao et al., arXiv 2309.08830), and
   [`app/retrieval/vector_retriever.py`](../app/retrieval/vector_retriever.py) —
   `search(..., hyde_text=...)` embeds the hypothetical text instead of the
   query when handed one. `HYDE_ENABLED` (default true) gates the generator,
   but no shipped caller passes `hyde_text`: the HyDE node belonged to the
   LangGraph workflow (§6/§7).

**Not implemented: "multi-query".** There is no code path that issues 3 lexical
variants or fuses their results (`grep -r multi_query app/` finds nothing).

**Trade-off.** The rewrite adds one LLM call to recall; the pronoun fast-path
keeps it off the common query. HyDE trades an LLM call for a denser query in
the document path.

**Test.** [`tests/test_hyde_agent.py`](../tests/test_hyde_agent.py) covers the
HyDE generator; the rewriter is exercised through the recall suites, which
patch its client seam (e.g. `_seams` in
[`tests/retrieval/test_p2_gate.py`](../tests/retrieval/test_p2_gate.py)).

---

## 5. Cross-encoder reranking (opt-in)

A cross-encoder reads `(query, chunk)` pairs and re-orders the candidate pool.
Cheaper than re-embedding the whole corpus, and the signal is much stronger
than cosine similarity alone.

**Status: opt-in per deployment** (`RETRIEVAL_SEMANTIC_RERANK=false` by
default). It is an outbound call carrying memory text, which is why the
pipeline SQL-authorizes and re-reads every candidate BEFORE the transport sees
it (spec §7.5/§14: no pre-ACL outbound text).

**Where.** [`app/retrieval/reranker.py`](../app/retrieval/reranker.py) — the
Jina rerank API (`jina-reranker-v2-base-multilingual`), one HTTP call bounded by
`JINA_RERANKER_TIMEOUT_SECONDS` (default 10).

**Semantics (P2).** The window is the retrieval pool
`top_k x RETRIEVAL_RERANK_POOL_MULTIPLIER` (signed default 2.0);
`JINA_RERANKER_TOP_N` (default 20) is only the per-call CAP on the transport's
answer, and the reranked head is MERGED into dense order — the served count is
`min(top_k, eligible)` and never shrinks because rerank ran. A timeout, a
non-2xx or a malformed body is typed (`RerankUnavailable` /
`RerankInvalidResponse`): the answer continues in dense order, counted as
`retrieval.rerank_failed`. A `0.0` relevance is DATA, never absence.

**Test.** [`tests/retrieval/test_rerank_pool.py`](../tests/retrieval/test_rerank_pool.py)
(pool/top_k/merge/typed failures/zero score) and the §9 gate
[`tests/retrieval/test_p2_gate.py`](../tests/retrieval/test_p2_gate.py)
(over real stores). The ablation's rerank arm (`eval/ablation_retrieval_p2.py`)
measures the STAGE with a local stand-in scorer, never Jina's quality.

---

## 6. LLM-as-judge hallucination detection — removed with the LangGraph agents

The `hallucination_agent` and the graph node that re-entered the answer node on
a flagged answer are gone with the LangGraph chat workflow (`app/api/v1/chat.py`
went with the full-stack surface; the deleted modules are in git history):
`app/agents/` now holds only `app/agents/llm_client.py`,
`app/agents/llm_parsing.py` and `app/agents/state.py`.
What survives:

- the OFFLINE judge — [`eval/llm_judge.py`](../eval/llm_judge.py) scores
  answers (faithfulness, relevancy, context precision) in eval runs.

---

## 7. Self-correction loops — removed with the LangGraph agents

The graph that wired `grade_docs` → re-retrieve and `grade_gen` → re-generate
(each bounded at 3 retries) was removed with the LangGraph agents. The pure
decision helpers it used (`MAX_RETRIES = 3`, `route_after_grade_docs`,
`route_after_grade_gen`, the retry / `record_*_retry_limit` edge names) lived in
`app/agents/routing.py`, which nothing imported and which went with the
lite-only consolidation (git history keeps it). `AgentState`
([`app/agents/state.py`](../app/agents/state.py)) still carries the fields and
appears only as a `TYPE_CHECKING` annotation
([`app/retrieval/hyde_agent.py`](../app/retrieval/hyde_agent.py)).

---

## 8. Parent-chunk cache (Redis)

The ingestion pipeline caches a document's parent chunks in Redis when it
finishes ([`app/ingestion/pipeline.py`](../app/ingestion/pipeline.py) →
[`app/retrieval/parent_store.py`](../app/retrieval/parent_store.py)): key
`parent_chunk:{conversation_id}:{parent_id}`, TTL 7200 s. The read helpers
(`get_parent`, `get_parents_batch`) serve from Redis, fall back to
`document_chunks` and repopulate the cache, and `invalidate_conversation`
drops a conversation's entries. The pipeline's synchronous write
(`store_parents_sync`) is skipped — a sync caller cannot reach the async
in-memory store — so reads go straight to the DB after synchronous ingestion.

**Trade-off.** The cache is process-local: `get_redis()` always hands out the
in-memory stand-in ([`app/redis_client.py`](../app/redis_client.py)), so it is
lost on restart and never shared across processes.

**Legacy.** [`app/retrieval/retrieval_cache.py`](../app/retrieval/retrieval_cache.py)
holds a per-conversation query-result cache (`rag:query:conv:...`, TTL 300 s)
whose get/set helpers have no call site in the shipped app — only the
invalidation hooks run, when documents change. Nothing reads the parent cache
back yet either: the BM25/document leg rebuilds from `document_chunks` and
keys its per-process indexes off a Redis generation counter
([`app/retrieval/bm25_retriever.py`](../app/retrieval/bm25_retriever.py)).

---

## 9. Vietnamese handling

- **Diacritic folding is the tokenizer's job.** `memory_fts` is created with
  `unicode61 remove_diacritics 2`
  ([`app/retrieval/memory/lexical_index.py`](../app/retrieval/memory/lexical_index.py)),
  and the index and the query go through the SAME folding tokenizer: a query
  typed WITH diacritics ("hồ sơ dự án") and one typed without them
  ("ho so du an") match and rank the same rows — no query-side stripping
  happens in Python.
- **The rewrite is Vietnamese-friendly.** The rewriter's system prompt is
  written for a Vietnamese second-brain
  ([`app/retrieval/memory/query_rewriter.py`](../app/retrieval/memory/query_rewriter.py)),
  and the pronoun heuristic that gates it is bilingual — "it/this/that" plus
  "nó/chúng/đó" ([`app/retrieval/memory/correction.py`](../app/retrieval/memory/correction.py)).
- **The dense leg is the configured embedder's.** `LOCAL_EMBED_MODEL="e5"` is
  the multilingual, opt-in model; the default `arctic` is English-first, and
  the dim guard refuses to mix the two on one store
  ([`app/config.py`](../app/config.py)).
- **e5 is opt-in for a measured reason.** On LongMemEval-S n=100 with a fresh
  store, the two local backends' own runs came out at **0.490** (default
  `arctic`, fp32) and **0.430** (quantized e5, int8) — separate runs, with
  their confidence intervals in the CHANGELOG. Both are ~384-dim, with
  incompatible vectors, so switching either way reindexes. Vietnamese-heavy
  corpora are the case for `LOCAL_EMBED_MODEL=e5`; the English bench is not.
  fp32 e5 was never tried: ~470 MB is over the lite budget.

**Not in this repo:** NFC normalization, `underthesea` syllable segmentation
and Vietnamese/English stopword lists — no such dependency and no such code
path.

**Test.** The §9 gate's slice test
([`tests/retrieval/test_p2_gate.py`](../tests/retrieval/test_p2_gate.py) —
`test_exact_id_vi_and_en_slices_rank_their_gold`) asserts the diacritic pair
above, plus the exact-ID and English slices, over the real FTS5 leg.

---

## 10. Prompt management

Prompts are module-level constants inside the module that uses them, not a
registry: `REWRITER_SYSTEM`
([`app/retrieval/memory/query_rewriter.py`](../app/retrieval/memory/query_rewriter.py)),
`HYDE_GENERATION_PROMPT` / `HYDE_REFINEMENT_PROMPT`
([`app/retrieval/hyde_agent.py`](../app/retrieval/hyde_agent.py)),
`ENTITY_EXTRACTION_PROMPT` / `RELATION_EXTRACTION_PROMPT`
([`app/graph/extraction.py`](../app/graph/extraction.py)) and `_SYSTEM_PROMPT`
([`app/services/compression_service.py`](../app/services/compression_service.py)).
There is no versioned-prompt registry and no A/B sweeper script in this repo —
a prompt variant is a code change, and evaluating it is an eval run
([EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)).
