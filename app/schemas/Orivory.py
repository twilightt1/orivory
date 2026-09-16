"""Pydantic schemas for Orivory Memory, Entity, Relation, Source."""
from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# ─── Memory ─────────────────────────────────────────────────────────────────

class MemoryCreate(BaseModel):
    title:         str | None        = Field(default=None, max_length=500)
    content:       str               = Field(min_length=1, max_length=100_000)
    summary:       str | None        = Field(default=None, max_length=4000)
    source_type:   Literal["manual_note", "file_upload", "google_drive", "notion",
                            "gmail", "web_clipper", "rss", "conversation_excerpt",
                            "chatgpt_import", "claude_import", "gemini_import", "copilot_import", "openclaw_import", "generic_import", "other"] = "manual_note"
    source_ref:    str | None        = Field(default=None, max_length=500)
    source_url:    str | None        = Field(default=None, max_length=1000)
    tags:          list[str]         = Field(default_factory=list, max_length=50)
    captured_at:   datetime | None   = None
    parent_id:     UUID | None       = None
    pinned:        bool              = False
    metadata:      dict              = Field(default_factory=dict)
    # Opt-in AI compression of long bodies before persisting (server flag
    # COMPRESSION_ENABLED must also be on; otherwise this is a no-op).
    auto_compress: bool              = False


class MemoryUpdate(BaseModel):
    title:         str | None        = Field(default=None, max_length=500)
    summary:       str | None        = Field(default=None, max_length=4000)
    tags:          list[str] | None  = Field(default=None, max_length=50)
    salience:      float | None      = Field(default=None, ge=0.0, le=1.0)
    pinned:        bool | None       = None
    metadata:      dict | None       = None


class ImportSummary(BaseModel):
    """Result of one import run (POST /api/v1/imports)."""
    parsed:              int
    created:             int
    skipped_duplicates:  int
    failed:              int
    index_failures:      int


class MemoryEntityLink(BaseModel):
    id:        UUID
    entity_id: UUID
    salience:  float
    model_config = ConfigDict(from_attributes=True)


class MemoryResponse(BaseModel):
    id:          UUID
    user_id:     UUID
    parent_id:   UUID | None
    source_type: str
    source_ref:  str | None
    source_url:  str | None
    title:       str | None
    content:     str
    summary:     str | None
    tags:        list[str]
    salience:    float
    pinned:      bool
    recall_count: int
    last_used_at: datetime | None
    captured_at: datetime
    indexed_at:  datetime
    updated_at:  datetime
    # Monotonic per-entity write counter (spec §4.1): bumped with every
    # content/metadata write that is enqueued to the index outbox.
    revision:    int
    # Write responses only (POST/PATCH): "pending" = the write's durable index
    # intent is still queued (the immediate best-effort embed did not land);
    # "ready" = it landed. None on read paths — a response that did not
    # observe an index state makes no claim about one.
    indexing:    Literal["ready", "pending"] | None = None
    # Lifecycle state, mirroring correction.state_of (spec §4.2): a superseded
    # row stays readable but never reads as current; dirty rows are not served
    # by list views at all.
    state:       Literal["current", "superseded", "dirty", "needs-check", "invalidated"]
    metadata:    dict

    model_config = ConfigDict(from_attributes=True)


class MemoryListResponse(BaseModel):
    items:  list[MemoryResponse]
    total:  int
    limit:  int
    offset: int


# ─── Digest (proactive surfacing, P2.2) ──────────────────────────────────────

class DigestThemeCount(BaseModel):
    """A theme (tag) and how many recent memories carry it."""
    theme: str
    count: int


class DigestResurfacedMemory(BaseModel):
    """A memory resurfaced from the past ('N months/years ago today')."""
    memory:    MemoryResponse
    age_label: str   # e.g. "1 year ago", "6 months ago"
    age_days:  int


class DigestResponse(BaseModel):
    generated_at:     datetime
    window_days:      int
    recent_count:     int                      # memories captured in the window
    top_themes:       list[DigestThemeCount]   # most common tags in the window
    recent_memories:  list[MemoryResponse]     # newest captures in the window
    resurfaced:       list[DigestResurfacedMemory]  # 'on this day' from the past


# ─── Entity ─────────────────────────────────────────────────────────────────

class EntityResponse(BaseModel):
    id:            UUID
    name:          str
    entity_type:   str
    aliases:       list[str]
    description:   str | None
    first_seen_at: datetime
    last_seen_at:  datetime
    mention_count: int
    metadata:      dict
    created_at:    datetime
    updated_at:    datetime

    model_config = ConfigDict(from_attributes=True)


class EntityListResponse(BaseModel):
    items:  list[EntityResponse]
    total:  int
    limit:  int
    offset: int


class EntityCreate(BaseModel):
    name:        str       = Field(min_length=1, max_length=255)
    entity_type: str       = Field(default="other", max_length=32)
    aliases:     list[str] = Field(default_factory=list, max_length=50)
    description: str | None = Field(default=None, max_length=4000)
    metadata:    dict      = Field(default_factory=dict)


class EntityUpdate(BaseModel):
    name:        str | None       = Field(default=None, min_length=1, max_length=255)
    entity_type: str | None       = Field(default=None, max_length=32)
    aliases:     list[str] | None = Field(default=None, max_length=50)
    description: str | None       = Field(default=None, max_length=4000)
    metadata:    dict | None      = None


# ─── Relation ───────────────────────────────────────────────────────────────

class RelationResponse(BaseModel):
    id:                UUID
    user_id:           UUID
    source_entity_id:  UUID
    target_entity_id:  UUID
    relation:          str
    weight:            float
    evidence_count:    int
    last_evidence_at:  datetime
    metadata:          dict
    created_at:        datetime
    updated_at:        datetime

    model_config = ConfigDict(from_attributes=True)


class RelationCreate(BaseModel):
    source_entity_id: UUID
    target_entity_id: UUID
    relation:         str   = Field(default="related_to", max_length=64)
    weight:           float = Field(default=0.5, ge=0.0, le=1.0)
    metadata:         dict  = Field(default_factory=dict)


class RelationUpdate(BaseModel):
    relation: str | None = Field(default=None, max_length=64)
    weight:   float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict | None = None


class GraphEdge(BaseModel):
    source: str           # entity name
    target: str           # entity name
    relation: str
    weight: float


class GraphNode(BaseModel):
    id:   str             # entity name
    type: str
    mentions: int


class GraphSnapshot(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    generated_at: datetime


class GraphCluster(BaseModel):
    id:    str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    score: float


class GraphClustersResponse(BaseModel):
    clusters:     list[GraphCluster]
    generated_at: datetime


# ─── Recall (Phase 3) ───────────────────────────────────────────────────────


class RecallRequest(BaseModel):
    """Request body for ``POST /api/v1/memories/recall``."""
    query:                   str   = Field(min_length=1, max_length=2000)
    top_k:                   int   = Field(default=10, ge=1, le=50)
    include_personal_context: bool = True


class MemoryWithScore(MemoryResponse):
    """A memory plus its retrieval score and the reasons it ranked."""
    score:         float
    match_reasons: list[str] = Field(default_factory=list)


RECALL_TRACE_STAGE_KEYS = (
    "context",
    "queue_wait",
    "rewrite_ms",
    "embed_ms",
    "embed_compute",
    "search_ms",
    "hydrate_ms",
    "refill",
    "lexical",
    "rerank",
    "eligibility",
    "score",
    "serialization",
    "total",
)

# Every declared key except `refill` starts at 0.0 (P0 design: the reserved
# keys stay diffable across runs even when their stage was skipped). `refill`
# is written only when the bounded refill really ran (T2/R12) — a pre-filled
# 0.0 would claim it ran in zero time, so an absent key is the honest
# "not needed". The four `_ms` keys are the legacy names the path has always
# written; declaring them here is what makes them versioned, not leaked extras.
RECALL_TRACE_ZERO_KEYS = tuple(
    key for key in RECALL_TRACE_STAGE_KEYS if key != "refill"
)

# The per-request candidate counters (spec §7.4): how many candidates each
# pipeline leg produced. The trace carries ONLY the legs that ran — an absent
# key means "this leg did not run", never a fabricated zero (`0` is a measured,
# real zero). `lexical` and `fused` are reserved for the hybrid legs (P2/T5,
# default OFF) and stay absent until then. Declared for the contract and the
# tests that pin it: no runtime enforcement of the key set is implemented
# beyond the writers themselves.
RECALL_TRACE_COUNTER_KEYS = (
    "dense",
    "lexical",
    "fused",
    "refill",
    "eligible",
    "reranked",
    "hydrated",
    "returned",
)


class RecallTrace(BaseModel):
    """Debug info returned alongside recall results."""
    rewritten_query:    str
    entities:           list[dict[str, str]]  # [{name, type}]
    latency_ms:         float
    num_candidates:     int
    num_results:        int
    used_personal_context: bool
    llm_fallback:       bool
    llm_reasoning:      str | None = None
    half_life_days:     float = 30.0
    rewrite_skipped:    bool = False
    stage_ms:           dict[str, float] = Field(
        default_factory=lambda: dict.fromkeys(RECALL_TRACE_ZERO_KEYS, 0.0)
    )
    # Candidate counts for the legs that ran (T3): the dense pre-filter fetch,
    # what the bounded refill added, what entered scoring, the reranked head,
    # the rows hydrated at scoring and the served count.
    counts:             dict[str, int] = Field(default_factory=dict)


class RecallResponse(BaseModel):
    """Response body for ``POST /api/v1/memories/recall``."""
    results:          list[MemoryWithScore]
    personal_context: list[MemoryResponse] | None = None
    trace:            RecallTrace


# ─── Agent clients (Open Memory Hub) ────────────────────────────────────────

class AgentClientCreate(BaseModel):
    """Request body for ``POST /api/v1/agents`` — register a hub client."""
    name:   str       = Field(min_length=1, max_length=100)
    scopes: list[str] = Field(default_factory=list, max_length=16)


class AgentClientCreated(BaseModel):
    """Registration response — carries the plaintext token, shown exactly once."""
    id:         UUID
    name:       str
    scopes:     list[str]
    status:     str
    created_at: datetime
    token:      str


class AgentClientResponse(BaseModel):
    """Client record without any token material (list/detail views)."""
    id:           UUID
    name:         str
    scopes:       list[str]
    status:       str
    created_at:   datetime
    last_used_at: datetime | None
    revoked_at:   datetime | None

    model_config = ConfigDict(from_attributes=True)


class AgentClientListResponse(BaseModel):
    items: list[AgentClientResponse]
    total: int


class AccessLogItem(BaseModel):
    """One row of the access ledger ("which AI saw what, and when")."""
    id:              UUID
    agent_client_id: UUID | None
    action:          str
    memory_id:       UUID | None
    detail:          dict
    created_at:      datetime

    model_config = ConfigDict(from_attributes=True)


class AccessLogListResponse(BaseModel):
    items: list[AccessLogItem]
    total: int


# ─── Erasure receipts (Open Memory Hub MVP 5) ───────────────────────────────


class ErasureReceiptCreate(BaseModel):
    """Request body for ``POST /api/v1/erasure-receipts`` — erase + verify."""
    memory_ids: list[UUID] = Field(min_length=1, max_length=100)


class ErasureReceiptItem(BaseModel):
    """One erasure receipt: what was requested and what verification found."""
    id:                   UUID
    user_id:              UUID
    requested_memory_ids: list[UUID]
    status:               str
    detail:               dict
    created_at:           datetime

    model_config = ConfigDict(from_attributes=True)


class ErasureReceiptListResponse(BaseModel):
    items: list[ErasureReceiptItem]
    total: int
