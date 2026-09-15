from urllib.parse import urlsplit

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _is_local_host(url: str) -> bool:
    return (urlsplit(url).hostname or "") in _LOCAL_HOSTS


class Settings(BaseSettings):

    # ── Lite mode ──────────────────────────────────────────────────────────────
    # LITE_MODE=1 gives a zero-external-services deployment: SQLite storage,
    # in-process Qdrant, in-memory caches (no Redis), synchronous in-process
    # background work (no worker), filesystem uploads (no MinIO). Default
    # DATABASE_URL / REDIS_URL point at the lite defaults; full-stack compose
    # overrides them.
    LITE_MODE: bool = False

    DATABASE_URL: str = "sqlite+aiosqlite:////data/orivory.db"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20

    # ── Outbox drain (P3 background indexing, both dialects) ──────────────────
    # Every deployment drains its index outbox from a background task in the
    # app's lifespan (app/retrieval/memory/drain_loop.py); the boot replays one
    # bounded batch on top of that. Disable to quiesce a deployment (e.g. while
    # a cutover owns the vector store): intents stay pending, nothing is lost.
    OUTBOX_DRAIN_ENABLED: bool = True
    OUTBOX_DRAIN_INTERVAL_SECONDS: float = 5.0
    OUTBOX_DRAIN_BATCH_SIZE: int = 50

    # ── Recall freshness barrier (P3) ─────────────────────────────────────────
    # A recall waits up to this long for its OWN tenant's pending index intents
    # to land before it searches (app/retrieval/memory/freshness.py): a memory
    # written moments ago must never read as a no-match. Past the budget the
    # recall answers 503 `index_freshness_timeout` — deliberately, since an
    # empty 200 would be that false no-match.
    RECALL_FRESHNESS_BUDGET_SECONDS: float = 2.0

    # The port the API listens on. `scripts/migrate_qdrant.py` probes it (plus
    # its `migrate.lock`) to refuse to run while the app is alive — the P1b
    # migration needs a quiesced store (spec §6.2 step 2, ruling R25). The
    # operator flow is documented in docs/OPERATIONS_RUNBOOK.md ("P1b cutover").
    APP_PORT: int = 8000


    REDIS_URL: str = ""
    REDIS_POOL_MAX: int = 20


    JWT_SECRET_KEY: str = ""  # auto-generated (ephemeral) when unset in lite mode
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Symmetric key for encrypting connector secrets (Source.config) at rest.
    # Must be a urlsafe-base64 32-byte Fernet key. When empty in non-production
    # a key is derived from JWT_SECRET_KEY so local dev works out of the box;
    # production requires an explicit value (see _validate_production_settings).
    CONFIG_ENCRYPTION_KEY: str = ""


    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    API_BASE_URL: str = "http://localhost:8000"

    # ── App identity ──────────────────────────────────────────────────────────────
    # Orivory — Personal AI Second Brain
    APP_NAME: str = "Orivory"
    APP_TAGLINE: str = "Personal AI Second Brain"
    CONTACT_EMAIL: str = "hello@orivory.local"

    # ── Email ─────────────────────────────────────────────────────────────────────
    SENDGRID_API_KEY: str = ""
    EMAIL_FROM: str = "noreply@orivory.local"
    EMAIL_FROM_NAME: str = "Orivory"
    # When SendGrid is not configured, emails are mocked. By default we log
    # only metadata (recipient, subject, body length) to avoid leaking OTP /
    # reset tokens into stdout. Set to True in development to log the full
    # body at DEBUG level.
    EMAIL_MOCK_VERBOSE: bool = False


    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str | None = None
    MINIO_SECRET_KEY: str | None = None
    MINIO_BUCKET: str = "rag-docs"
    MINIO_SECURE: bool = False


    LEDGER_RETENTION_DAYS: int = 90
    # Compression-before-storage (claude-mem adopt-learn): off by default —
    # opt-in per deployment; failures degrade to storing raw content.
    COMPRESSION_ENABLED: bool = False
    COMPRESSION_THRESHOLD_CHARS: int = 2000
    COMPRESSION_MODEL: str = "gpt-4o-mini"

    # ── Qdrant vector backend ─────────────────────────────────────────────────
    # "local" runs Qdrant embedded in-process against QDRANT_LOCAL_PATH — one
    # process owns that folder (qdrant-client locks it; a second client on the
    # same folder refuses); "server" talks to QDRANT_URL.
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_MODE: str = "server"  # server | local
    QDRANT_LOCAL_PATH: str = "/data/qdrant"

    # The RETIRED pre-P1b vector store's directory. Nothing in the app serves
    # from it: the only reader is the P1b migration CLI's backup source
    # (`scripts/migrate_qdrant.py::_backup_sources`); the one-release rollback
    # tool takes the directory as its `--chroma-path` argument instead. A
    # deployment that never ran Chroma leaves it at the default and the backup
    # simply reports it "missing".
    LEGACY_CHROMA_PATH: str = "/data/chroma"

    # lite: "fs" stores uploads on the local filesystem instead of MinIO.
    STORAGE_BACKEND: str = "minio"  # minio | fs
    FS_STORAGE_PATH: str = "/data/uploads"


    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_MODEL: str = "openai/gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.7
    # OpenRouter free-tier models share a congested pool and 429 constantly
    # under burst. Cap how many agent LLM calls hit the provider at once;
    # the SDK retries with backoff (see llm_client.DEFAULT_LLM_MAX_RETRIES).
    LLM_MAX_CONCURRENCY: int = 3
    # Factual RAG answers should be near-deterministic; the global 0.7 is for
    # other/creative uses. The answer agent uses this lower value to reduce
    # hallucination and verbosity.
    ANSWER_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 2048
    # Approx character budget for the assembled LLM context (~4 chars/token).
    # Guards against silently overflowing the model context window.
    CONTEXT_CHAR_BUDGET: int = 24000



    OPENAI_API_KEY: str = ""
    EMBED_MODEL: str = "text-embedding-3-small"
    EMBED_DIMENSIONS: int = 1536
    EMBED_BATCH_SIZE: int = 64
    # ── Bounded local execution (P2/T1) ──────────────────────────────────────
    # Every ASYNC embedding call runs on ONE dedicated executor of this width
    # (thread-name prefix `orivory-embed`): it keeps the synchronous ONNX call
    # off the event loop — the 50-document drain batch the recall barrier runs
    # on the request path was ~674 ms of unbroken loop stall — while a bound
    # (not the loop's default pool) keeps ORT's own threads accountable. The
    # `*_sync` faces stay caller-threaded: their callers are already off-loop.
    EMBED_EXECUTOR_WORKERS: int = 2
    # ONNX Runtime intra-op threads per embedding session (T1 report C1; ruling
    # R8(p2)). 0 = ORT's own default (one thread per core) — the value that meets
    # the signed budgets here: a 50-intent drain ~1.1 s inside the 2.0 s RYW
    # budget, and the recall's own query embed well inside the p95 <= 150 ms.
    # The loop-lag guarantee comes from the OFFLOAD, not from starving intra-op:
    # at intra_op=1 every call is ~4x slower on an M-series box (54 ms vs 13 ms
    # for one 864-char document) and that same drain took 3.07 s — past the
    # signed RYW budget. Set it to 1 (or fewer) only to cap oversubscription on
    # a small machine (EMBED_EXECUTOR_WORKERS=2 concurrent embeds, each free to
    # use every core), and re-measure RYW + recall p95 for that deploy.
    EMBED_ORT_INTRA_OP_THREADS: int = 0
    # Build the local embedding session during the lifespan, BEFORE the boot
    # drain and before anything is served: a cold InferenceSession is 610-685 ms
    # and even inside a thread it leaves C-level parse lag.
    EMBED_WARMUP_ON_BOOT: bool = True


    JINA_API_KEY: str = ""
    JINA_EMBED_MODEL: str = "jina-embeddings-v3"
    JINA_EMBED_DIMENSIONS: int = 1024
    # Cross-encoder rerank inside MemoryRetriever: reorder the vector
    # candidate pool by true query-document relevance (Jina reranker)
    # before salience/decay modifiers. Off by default — per-deployment.
    RETRIEVAL_SEMANTIC_RERANK: bool = False
    JINA_RERANKER_MODEL: str = "jina-reranker-v2-base-multilingual"
    # Per-call CAP on the reranker's own answer, never the rerank window: the
    # per-call `top_n` is the request's own top_k clamped to this value
    # (ruling R4(p2)). The returned result count does NOT depend on it — a
    # rerank that answers with fewer rows than it was handed is merged back
    # into dense order (`retriever.recall`), so raising it only widens the
    # reranked HEAD of a large top_k. Ruling R13(p2): the default (20) covers
    # the default `top_k=10` x RETRIEVAL_RERANK_POOL_MULTIPLIER=2.0 window, so
    # one served window is never half reranked and half dense x boost x decay.
    # Serving a larger window, raise it to `top_k x pool multiplier`; it stays
    # a cap, and the extra ranks cost only what the opt-in rerank flag spends.
    JINA_RERANKER_TOP_N: int = 20
    # Rerank pool: dense candidates fetched per requested result (ruling
    # R4(p2), signed default 2.0). One pool feeds the eligibility filter, the
    # reranker and scoring; a pool smaller than top_k cannot satisfy the count
    # invariant, so the effective fetch is `max(top_k, ceil(top_k * this))`.
    RETRIEVAL_RERANK_POOL_MULTIPLIER: float = 2.0
    # Hybrid recall (P2/T5): when ON, recall runs the SQLite FTS5 lexical leg
    # over the same query next to the dense one and fuses the two by RRF
    # (app/retrieval/hybrid_retriever.fuse_by_uuid). Ships OFF — only the T7
    # ablation artifact may flip it (ruling R2(p2)) — and the OFF path stays
    # byte-equivalent to the dense-only recall: no lexical query is issued and
    # the trace carries no `lexical`/`fused` counters. The vector-outage
    # fallback to the lexical leg is NOT gated by this flag (ruling R19): it
    # only ever replaces the typed 503.
    RETRIEVAL_HYBRID_ENABLED: bool = False
    # The RRF constant `sum(1 / (k + rank + 1))` over ZERO-BASED ranks (ruling
    # R11b(p2)). 60 is both the spec's starting value and the old document
    # helper's default; the fused pool is ordered by this sum, never by the
    # legs' scores (dense cosine and global BM25 are not comparable).
    RETRIEVAL_RRF_K: int = 60
    # Bound on ONE rerank HTTP call (ruling R11(p2)): a hung transport must not
    # hold the recall path for the client's own 30 s default. Overrunning it is
    # a `RerankUnavailable` — dense order continues, counted.
    JINA_RERANKER_TIMEOUT_SECONDS: float = 10.0
    # ── Embedding backend support matrix (frozen v1.1.0) ──
    #   jina  (USE_JINA_EMBEDDINGS=true + JINA_API_KEY): SUPPORTED default
    #           for full-stack. Matches the frozen benchmark baseline.
    #   local (USE_LOCAL_EMBEDDINGS=true): SUPPORTED for lite/self-contained
    #           mode only (384-dim, no API key). Do not mix with Jina/OpenAI
    #           in one store — the dim guard will refuse.
    #   openai (fallback when neither above applies): LEGACY, unbenchmarked,
    #           kept so old deployments boot. Not supported for recall quality.
    # Use jina for embeddings instead of OpenAI
    USE_JINA_EMBEDDINGS: bool = True
    # Local ONNX embeddings (384-dim, no API key). Takes precedence over
    # Jina/OpenAI when true — keeps lite mode and benchmarks self-contained.
    # Do not mix backends in one store.
    USE_LOCAL_EMBEDDINGS: bool = False
    # Which local model backs USE_LOCAL_EMBEDDINGS: "arctic"
    # (snowflake-arctic-embed-xs, default — best English bench, CLS pooling) or
    # "e5" (multilingual, opt-in Vietnamese, mean pooling). Both 384-dim but
    # semantically incompatible — the dim guard records them as different
    # backends and refuses to mix them; switching on an existing store
    # requires reindexing into a fresh collection. Anything else (the old
    # chroma-bundled "minilm", a typo) is refused at load.
    LOCAL_EMBED_MODEL: str = "arctic"
    # Where the e5 onnx/tokenizer files live (downloaded once on first use).
    # Empty = ~/.cache/orivory/e5; the lite image sets /data/models/e5 so the
    # download persists on the data volume.
    LOCAL_E5_DIR: str = ""

    # ── Corrective-RAG (CRAG) ────────────────────────────────────────────────────
    # CRAG self-critiques retrieval quality and falls back to web search when needed.
    # Reference: Yan et al., arXiv 2401.15884
    CRAG_ENABLED: bool = True
    CRAG_GRADING_MODEL: str = "openai/gpt-4o-mini"  # Model for grading (smaller = faster)
    CRAG_RELEVANCE_THRESHOLD: float = 0.7  # Score >= this = RELEVANT
    CRAG_PARTIAL_THRESHOLD: float = 0.4  # Score >= this = PARTIAL
    CRAG_FALLBACK_THRESHOLD: float = 0.5  # % of docs needed to avoid web fallback
    CRAG_MAX_WEB_RESULTS: int = 10  # Max web search results to include
    CRAG_MAX_WEB_CHARS: int = 4000  # Per-doc cap on raw web page text kept in context
    TAVILY_API_KEY: str = ""  # Tavily API key for web search fallback

    # ── HyDE (Hypothetical Document Embeddings) ────────────────────────────────
    # HyDE generates hypothetical documents for better retrieval.
    # Reference: Gao et al., arXiv 2309.08830
    HYDE_ENABLED: bool = True
    HYDE_MODEL: str = "openai/gpt-4o-mini"  # Model for generating hypothetical docs
    HYDE_PASSAGE_COUNT: int = 3  # Number of hypothetical passages to generate
    HYDE_USE_IN_RETRIEVAL: bool = True  # Use HyDE embeddings in retrieval

    # ── Multi-hop Reasoning (EfficientRAG) ─────────────────────────────────────
    # Multi-hop query decomposition and reasoning.
    # Reference: EfficientRAG - EMNLP 2024
    MULTIHOP_ENABLED: bool = True
    MULTIHOP_MODEL: str = "openai/gpt-4o-mini"  # Model for multi-hop reasoning
    MULTIHOP_MAX_HOPS: int = 3  # Maximum number of reasoning hops
    FEEDBACK_MAX_WEIGHT: float = 2.0  # Max document weight
    FEEDBACK_MIN_WEIGHT: float = 0.5  # Min document weight


    EVALUATOR_FAILURE_MODE: str = "warn_only"


    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_DAY: int = 1000

    # ── MCP memory hub ───────────────────────────────────────────────────────────
    # When true, the Open Memory Hub MCP server (stateless streamable HTTP) is
    # mounted at /mcp for registered agent clients.
    MCP_HUB_ENABLED: bool = True
    # Comma-separated list of Host header values / hostnames allowed to reach
    # /mcp (e.g. "api.orivory.io, mcp.orivory.io"). Empty keeps FastMCP's
    # default behaviour: because the app binds host 127.0.0.1, the SDK
    # auto-enables localhost-only DNS-rebind protection, which 421s any
    # non-localhost Host (i.e. the endpoint is localhost-only until this is
    # set — required behind a reverse proxy that forwards a public Host).
    MCP_HUB_ALLOWED_HOSTS: str = ""


    FRONTEND_URL: str = "http://localhost:3000"
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:5173"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    @model_validator(mode="after")
    def _apply_lite_mode(self) -> "Settings":
        if self.LITE_MODE:
            # Ephemeral JWT secret when unset: lite is a single-user personal
            # deployment; tokens survive only until the container restarts.
            if not self.JWT_SECRET_KEY:
                import secrets
                self.JWT_SECRET_KEY = secrets.token_urlsafe(48)
            # In-process background work, embedded Qdrant, filesystem storage
            # unless overridden.
            # Same flip for Qdrant: no API key + a localhost URL means there is
            # no server to talk to, so own a local folder instead.
            if (
                self.QDRANT_MODE == "server"
                and not self.QDRANT_API_KEY
                and _is_local_host(self.QDRANT_URL)
            ):
                self.QDRANT_MODE = "local"
            if self.STORAGE_BACKEND == "minio" and not self.MINIO_ACCESS_KEY:
                self.STORAGE_BACKEND = "fs"
            # Zero-key lite must still remember: no embedding API key means
            # the bundled local model (384-dim, no download beyond ONNX).
            # ponytail: keyed backends win whenever a key exists; the dim
            # guard refuses mixing backends in one store.
            if not self.USE_LOCAL_EMBEDDINGS and not self.JINA_API_KEY and not self.OPENAI_API_KEY:
                self.USE_LOCAL_EMBEDDINGS = True
        return self

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.casefold() == "production"

    @model_validator(mode="after")
    def validate_environment_settings(self):
        self.ENVIRONMENT = self.ENVIRONMENT.casefold()
        self.EVALUATOR_FAILURE_MODE = self.EVALUATOR_FAILURE_MODE.casefold()
        self._validate_ai_runtime_settings()
        if self.is_production:
            self._validate_production_settings()
        else:
            if not self.MINIO_ACCESS_KEY:
                self.MINIO_ACCESS_KEY = "minioadmin"
            if not self.MINIO_SECRET_KEY:
                self.MINIO_SECRET_KEY = "minioadmin"
        return self

    def _validate_ai_runtime_settings(self) -> None:
        if self.EMBED_BATCH_SIZE < 1 or self.EMBED_BATCH_SIZE > 2048:
            raise ValueError("EMBED_BATCH_SIZE must be between 1 and 2048")
        if self.LOCAL_EMBED_MODEL not in {"arctic", "e5"}:
            # The chromadb-bundled MiniLM branch was removed in P1b: a stale
            # "minilm" (or any typo) must fail at load instead of silently
            # embedding with a contract nothing can name or verify.
            raise ValueError("LOCAL_EMBED_MODEL must be one of: arctic, e5")
        if self.QDRANT_MODE not in {"server", "local"}:
            # A typo'd mode would silently boot a server client against a folder
            # path (or the reverse); refuse instead.
            raise ValueError("QDRANT_MODE must be one of: server, local")
        allowed_modes = {"warn_only", "fail_open", "fail_closed"}
        if self.EVALUATOR_FAILURE_MODE not in allowed_modes:
            raise ValueError(
                "EVALUATOR_FAILURE_MODE must be one of: warn_only, fail_open, fail_closed"
            )

    def _validate_production_settings(self) -> None:
        self._require_strong_jwt_secret()
        self._require_explicit_cors_origins()
        self._require_provider_keys()
        self._require_secure_minio_credentials()
        self._require_config_encryption_key()

    def _require_config_encryption_key(self) -> None:
        if not self.CONFIG_ENCRYPTION_KEY.strip():
            raise ValueError(
                "CONFIG_ENCRYPTION_KEY must be set in production "
                "(generate with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\")"
            )

    def _require_strong_jwt_secret(self) -> None:
        placeholders = {
            "change-me",
            "change-me-to-a-random-256-bit-secret",
            "test-secret-key-change-in-production",
            "secret",
            "your-secret-key",
        }
        normalized_secret = self.JWT_SECRET_KEY.strip().casefold()
        if normalized_secret in placeholders or "change-me" in normalized_secret:
            raise ValueError("JWT_SECRET_KEY must not use a placeholder value in production")
        if len(self.JWT_SECRET_KEY) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters in production")

    def _require_explicit_cors_origins(self) -> None:
        origins = [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]
        if not origins:
            raise ValueError("ALLOWED_ORIGINS must define at least one origin in production")
        for origin in origins:
            if origin == "*":
                raise ValueError("ALLOWED_ORIGINS cannot contain '*' in production")
            if not origin.startswith(("https://", "http://")):
                raise ValueError("ALLOWED_ORIGINS must contain explicit HTTP(S) origins in production")

    def _require_provider_keys(self) -> None:
        required_keys = {
            "OPENROUTER_API_KEY": self.OPENROUTER_API_KEY,
            "OPENAI_API_KEY": self.OPENAI_API_KEY,
            "JINA_API_KEY": self.JINA_API_KEY,
        }
        missing = [name for name, value in required_keys.items() if not value.strip()]
        if missing:
            raise ValueError(f"Missing provider keys in production: {', '.join(missing)}")

    def _require_secure_minio_credentials(self) -> None:
        if not self.MINIO_ACCESS_KEY or not self.MINIO_SECRET_KEY:
            raise ValueError("MINIO_ACCESS_KEY and MINIO_SECRET_KEY must be set in production")
        if self.MINIO_ACCESS_KEY == "minioadmin" or self.MINIO_SECRET_KEY == "minioadmin":
            raise ValueError("Default MinIO credentials are not allowed in production")


settings = Settings()
