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
- Lexical leg (memory): [`app/retrieval/memory/lexical_index.py`](../app/retrieval/memory/lexical_index.py) — SQLite FTS5 (`memory_fts`, created by the schema ladder's v4 step; triggers maintain it in the writing transaction). SQLite-only by ruling R3: a Postgres deployment has NO lexical leg, and a vector outage there keeps the typed 503.
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

## 3. Parent-child chunking

```
[Parent (1024 tok)]     ← returned to the LLM for context
   ├── [Child (256 tok)] ← embedded into the vector store
   ├── [Child (256 tok)]
   └── [Child (256 tok)]
```

**Why.** Small children → precise embedding (better recall).
Large parents → readable context for the answer agent.

**Where.** [`app/retrieval/hybrid_retriever.py`](../app/retrieval/hybrid_retriever.py)

**Trade-off.** 4-6× more storage, but the LLM's "answer quality" metrics
improve noticeably because the answer agent has surrounding context for
each child.

---

## 4. Multi-query + HyDE + conversation rewrite

Three query transformations run in parallel (`asyncio.gather`):

1. **Conversation rewrite** — resolves pronouns using the last 3 turns
2. **Multi-query** — 3 lexical variants of the question
3. **HyDE** — generate a hypothetical answer, embed that instead of the question

**Why.** Different retrievers respond to different query phrasings. Throwing
3-5 reformulations at the retriever and fusing the results is a cheap win.

**Trade-off.** LLM call for the router → +50-200 ms p50. HyDE is
opt-in (gated by config) because not all domains benefit from it.

**Test.** `tests/rag/test_query_transforms.py`

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

## 6. LLM-as-judge hallucination detection

After the answer agent runs, a second LLM call asks:

> "Given the context and the question, is the answer grounded?
> Does it actually answer the question? Cite a [Source N] marker?"

If `is_hallucination` is True, the graph re-enters the answer node with
the same context but a stricter prompt hint — up to 3 times.

**Where.** [`app/agents/hallucination_agent.py`](../app/agents/hallucination_agent.py)
(used by [`app/agents/graph.py`](../app/agents/graph.py))

**Test.** `tests/rag/test_hallucination_retry.py`

---

## 7. Self-correction loops (LangGraph)

Two correction edges in the graph:
- `grade_docs` returns `context_relevant=False` → re-retrieve
- `grade_gen` returns `is_hallucination=True` → re-generate

Each is bounded at 3 retries to prevent infinite loops.

**Where.** [`app/agents/graph.py`](../app/agents/graph.py) — see the `add_conditional_edges` calls.

**Metric.** "Correction rate" in the eval report measures how often
retries actually fixed the issue.

---

## 8. Parent-chunk cache (Redis)

Successful parent retrievals are cached in Redis (`parent:<doc_id>:<chunk_id>`)
for 2 hours. Hot docs become sub-millisecond to retrieve.

**Where.** [`app/retrieval/retrieval_cache.py`](../app/retrieval/retrieval_cache.py)

**Trade-off.** Redis becomes a hard dependency in production. For local
dev, the cache layer transparently falls back to a no-op.

---

## 9. Vietnamese-specific preprocessing

- **NFC normalization** — `"café"` and `"café"` collapse to the same string
- **Syllable segmentation** — `underthesea` (configurable)
- **Stopword filter** — Vietnamese + English stopword lists
- **Diacritic-safe BM25** — keep diacritics for the index (Vietnamese users
  type with diacritics), strip only for the query (some keyboards lose them)

**Where.** [`app/retrieval/hyde_agent.py`](../app/retrieval/hyde_agent.py)

---

## 10. Prompt management

Agent prompts live as module-level constants inside each agent file
(e.g. `GRADE_DOCS_PROMPT` and `QUERY_EXPANSION_PROMPT` in
[`app/agents/crag_agent.py`](../app/agents/crag_agent.py)). The original
versioned-prompt registry was removed as dead weight during the P4
hardening pass; A/B experimentation now runs through the eval experiment
sweeper (`scripts/eval_experiments.py` — see
[EVALUATION_GUIDE.md](EVALUATION_GUIDE.md#prompt-ab-testing)).
