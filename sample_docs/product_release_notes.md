# Product Release Notes

These release notes describe recent Orivory product changes that support
agents may need to reference.

## Version 2.4.0 — SSO and Audit Logs

Released on 2026-04-15.

New features:

- Enterprise SAML and OIDC Single Sign-On.
- Admin audit logs for user, quota, and document changes.
- OAuth redirect exchange codes to avoid exposing tokens in URLs.
- Configurable CORS allowed origins.

## Version 2.3.0 — Hybrid Retrieval

Released on 2026-03-20.

New features:

- ChromaDB vector retrieval for semantic search.
- BM25 lexical retrieval for exact product and API terms.
- Reciprocal Rank Fusion for combining retriever results.
- Local ONNX cross-encoder reranking for final context selection.

## Version 2.2.0 — Async Ingestion

Released on 2026-02-12.

New features:

- Celery-based document ingestion workers.
- MinIO storage for uploaded files.
- Document status transitions: `pending`, `processing`, `ready`, `failed`.
- Admin retry support for failed ingestion jobs.

## Version 2.1.0 — Agent Trace

Released on 2026-01-10.

New features:

- Agent trace metadata stored on assistant messages.
- Retrieval diagnostics for BM25, vector search, fusion, and reranking.
- Source snippets returned with streaming responses.
