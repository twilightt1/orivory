# RAG Pipeline Architecture & Optimization Notes

## Overview

This document describes the current Retrieval-Augmented Generation (RAG)
pipeline after the AI architecture hardening work. The design goal is to balance
latency, retrieval quality, and runtime safety for a portfolio-grade but
production-oriented personal second-brain backend.

The current architecture is not a minimal linear "chat with PDF" flow. It uses a
bounded corrective LangGraph workflow, hybrid retrieval, parent-child chunking,
reranking, streaming answers, and traceable guardrails.

## 1. Combined Routing and Query Processing

### Problem

A naive RAG pipeline often performs separate LLM calls for:

1. intent classification
2. query rewriting
3. multi-query generation
4. answer generation
5. post-generation checks

If each step blocks the next, first-token latency becomes unacceptable.

### Current Solution

The router consolidates intent classification, query rewriting, and search
variant generation into a single structured LLM call. It returns JSON like:

```json
{
  "intent": "rag",
  "confidence": 0.95,
  "reasoning": "User asks about product docs",
  "rewritten_query": "How can a user rotate an API key?",
  "search_variants": [
    "API key rotation steps",
    "regenerate API token",
    "replace compromised API key"
  ]
}
```

A regex fast path handles simple chitchat greetings without an LLM call.

### Trade-off

Combining router duties reduces network latency and prompt overhead. The trade-off
is that router quality depends on one structured prompt, so the output is traced
with confidence, reasoning, rewritten query, and variants for debugging.

## 2. Bounded Corrective RAG, Not an Unbounded Cycle

### Problem

Fully cyclic agent graphs can create unpredictable latency and cost if they keep
retrieving or regenerating without hard limits.

### Current Solution

The LangGraph workflow keeps corrective behavior but bounds it with
`MAX_RETRIES = 3` and records correction reasons in `agent_trace`.

Current high-level flow:

```text
router
→ memory
→ retrieval
→ grade_docs
→ answer
→ grade_gen
→ save
```

Correction paths are bounded:

```text
irrelevant context → retry retrieval
hallucination detected → retry answer
answer does not resolve question → retry retrieval
retry limit reached → record limit and continue safely
```

### Trade-off

A pure DAG would have lower and more predictable latency. The current bounded
corrective graph preserves safety and debuggability while preventing infinite
loops. Trace metadata makes retry behavior visible in SSE `trace` events and
saved messages.

## 3. Hybrid Retrieval with Parent-Child Context

### Current Retrieval Flow

```text
conversation-scoped query cache lookup
→ API-process BM25 lazy rebuild if needed
→ BM25 parent search
→ multi-query Qdrant child vector search
→ Reciprocal Rank Fusion
→ parent expansion from Redis or PostgreSQL fallback
→ Jina reranking
→ final top context chunks
```

### Why Parent-Child Chunking

Child chunks are embedded for precise vector recall. Parent chunks are expanded
as LLM context so the answer model receives readable, coherent evidence instead
of tiny fragments.

### BM25 Runtime Consistency

BM25 indexes are in-memory per process. Ingestion may build BM25 inside a Celery
worker, while live chat runs in a separate FastAPI worker. To avoid losing hybrid
retrieval at runtime, the API process lazily rebuilds missing BM25 indexes from
PostgreSQL before lexical search.

This keeps the current simple in-memory implementation viable for a local or
single-node deployment while documenting a clear migration path to durable
keyword search later.

## 4. Vietnamese-Friendly BM25 Tokenization

The BM25 tokenizer uses both unigrams and adjacent bigrams:

```python
words = re.findall(r"\w+", text.lower())
bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
return words + bigrams
```

This lightweight heuristic improves matching for Vietnamese compound terms
without adding heavier NLP dependencies such as `pyvi` or `underthesea`.

## 5. Query Cache and Invalidation

Retrieval results are cached per conversation and query/history hash to reduce
repeat retrieval cost. Cache entries are invalidated when document state changes:

- document upload
- document delete
- ingestion success
- ingestion permanent failure

This prevents stale context from being reused after a knowledge base update.

## 6. Guardrails and Failure Modes

The pipeline still includes:

- pre-answer relevance grading via `evaluator_agent`
- post-answer groundedness / answer-completeness checks via `hallucination_agent`
- citation trace from `answer_agent`

Evaluator behavior is configurable with `EVALUATOR_FAILURE_MODE`:

- `warn_only` / `fail_open`: preserve demo continuity and trace failures
- `fail_closed`: stricter behavior for safer production-style operation

The default favors availability for local demos while making stricter operation a
configuration choice.

## 7. Streaming and Observability

The API streams tokens over SSE and emits final sources, trace, and done events.
`agent_trace` includes runtime metadata such as:

- retrieval cache hit/miss
- BM25 rebuild status
- BM25/vector result counts
- parent expansion counts
- retrieval timing breakdown
- answer latency / first-token latency
- citation status
- evaluator failure mode
- correction reasons and retry counts

This makes the RAG system easier to debug during demos and live smoke tests.

## Limitations & Future Work

1. **Durable keyword search:** BM25 is still in-memory per API worker. Lazy rebuild
   fixes runtime consistency, but horizontally scaled production should migrate to
   Elasticsearch, OpenSearch, PostgreSQL full-text search, or a managed hybrid
   retrieval platform.
2. **Provider resilience:** LLM, embedding, and reranker calls should eventually
   share explicit retry, timeout, and circuit-breaker policies.
3. **Retrieval ablation:** Add retrieval-only evaluation for BM25 vs vector vs
   hybrid vs reranked performance.
4. **Strict citation enforcement:** The current runtime records citation status.
   A future strict mode can reject or regenerate uncited factual answers.

## Conclusion

The current architecture balances latency and reliability by combining router
work into one structured call, using hybrid retrieval with reranking, preserving
bounded corrective loops, and exposing detailed trace metadata. Phase 12A fixed
the major runtime consistency issue by adding API-process BM25 lazy rebuild and
cache invalidation, making the system more credible for production-like live
smoke validation.
