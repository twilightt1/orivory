from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    user_id: str
    conversation_id: str
    query: str
    rewritten_query: str

    query_type: str
    router_confidence: float
    router_reasoning: str
    search_variants: list[str]

    history: list[dict[str, Any]]

    bm25_results: list[dict[str, Any]]
    vector_results: list[dict[str, Any]]
    vector_unavailable: bool
    fused_chunks: list[dict[str, Any]]

    reranked_chunks: list[dict[str, Any]]
    doc_context_chunks: list[dict[str, Any]]
    personal_memory_chunks: list[dict[str, Any]]
    graph_context_chunks: list[dict[str, Any]]
    grounding_context_chunks: list[dict[str, Any]]
    personal_recall_trace: dict[str, Any]
    graph_context_trace: dict[str, Any]

    response: str
    token_count: int
    agent_trace: dict[str, Any]

    error: str | None
    should_stream: bool
    has_documents: bool
    document_count: int
    personal_memory_enabled: bool
    graph_context_enabled: bool
    personal_memory_top_k: int

    context_relevant: bool
    is_hallucination: bool
    answers_question: bool
    retry_count: int

    # AI/ML observability (cost & latency tracking)
    cumulative_cost_usd: float
    total_tokens_in: int
    total_tokens_out: int
    agent_costs: dict[str, float]
    agent_latency_ms: dict[str, float]

    # ── Corrective-RAG (CRAG) State ─────────────────────────────────────────────
    # Reference: Yan et al., arXiv 2401.15884
    crag_trace: dict[str, Any]  # Trace of CRAG operations
    crag_enabled: bool  # Whether CRAG is enabled for this query
    crag_web_fallback_triggered: bool  # Whether web fallback was triggered

    # ── HyDE (Hypothetical Document Embeddings) State ──────────────────────────
    # Reference: Gao et al., arXiv 2309.08830
    hyde_result: dict[str, Any] | None  # Generated hypothetical document
    hyde_trace: dict[str, Any]  # Trace of HyDE operations

    # ── Multi-hop Reasoning State ──────────────────────────────────────────────
    multihop_subqueries: list[dict[str, Any]] | None  # Subqueries for each hop
    multihop_trace: dict[str, Any]  # Trace of multi-hop operations
