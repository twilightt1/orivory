"""Orivory personal-memory retrieval package.

Re-ranks and retrieves from the user's personal `Memory` store (as
opposed to the RAG chatbot's per-conversation document chunks).

Public surface:
    - ``scoring``      : time-decay + entity-boost math
    - ``vector_store`` : Qdrant ops (the active memory generation)
    - ``query_rewriter``: LLM-based query rewrite + entity extraction
    - ``context``      : personal context (recent + pinned memories)
    - ``lexical_index``: the SQLite FTS5 lexical leg (SQLite only — ruling R3)
    - ``retriever``    : orchestrator that combines all the above
"""
