from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # ── Lite mode ──────────────────────────────────────────────────────────────
    # LITE_MODE=1 gives a zero-external-services deployment: SQLite storage,
    # in-process Chroma, in-memory caches (no Redis), synchronous in-process
    # background work (no worker), filesystem uploads (no MinIO). Default
    # DATABASE_URL / REDIS_URL point at the lite defaults; full-stack compose
    # overrides them.
    LITE_MODE: bool = False

    DATABASE_URL: str = "sqlite+aiosqlite:////data/orivory.db"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20


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


    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001
    # lite: "local" runs ChromaDB in-process (PersistentClient) against
    # CHROMA_LOCAL_PATH — no chroma container needed.
    CHROMA_MODE: str = "http"  # http | local
    LEDGER_RETENTION_DAYS: int = 90
    # Compression-before-storage (claude-mem adopt-learn): off by default —
    # opt-in per deployment; failures degrade to storing raw content.
    COMPRESSION_ENABLED: bool = False
    COMPRESSION_THRESHOLD_CHARS: int = 2000
    COMPRESSION_MODEL: str = "gpt-4o-mini"
    CHROMA_LOCAL_PATH: str = "/data/chroma"

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


    JINA_API_KEY: str = ""
    JINA_EMBED_MODEL: str = "jina-embeddings-v3"
    JINA_EMBED_DIMENSIONS: int = 1024
    # Cross-encoder rerank inside MemoryRetriever: reorder the vector
    # candidate pool by true query-document relevance (Jina reranker)
    # before salience/decay modifiers. Off by default — per-deployment.
    RETRIEVAL_SEMANTIC_RERANK: bool = False
    JINA_RERANKER_MODEL: str = "jina-reranker-v2-base-multilingual"
    JINA_RERANKER_TOP_N: int = 5
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
    # Local ONNX MiniLM embeddings (chromadb-bundled, 384-dim, no API key).
    # Takes precedence over Jina/OpenAI when true — keeps lite mode and
    # benchmarks fully self-contained. Do not mix backends in one store.
    USE_LOCAL_EMBEDDINGS: bool = False
    # Which local model backs USE_LOCAL_EMBEDDINGS: "arctic"
    # (snowflake-arctic-embed-xs, default — best English bench) or "minilm"
    # (chroma-bundled, legacy) or "e5" (multilingual, opt-in Vietnamese).
    # Both 384-dim but semantically incompatible — the dim guard records them
    # as different backends and refuses to mix them; switching on an
    # existing store requires reindexing into a fresh collection.
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
            # In-process background work, local chroma, filesystem storage unless overridden.
            if self.CHROMA_MODE == "http" and self.CHROMA_HOST == "localhost":
                self.CHROMA_MODE = "local"
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
