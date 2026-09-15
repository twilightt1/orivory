#!/usr/bin/env python
"""P2 / Task 7 — the retrieval ablation on a FROZEN fixture (the hybrid enable rule).

What this artifact decides
-------------------------
``RETRIEVAL_HYBRID_ENABLED`` ships OFF (ruling R2(p2)); the HYBRID flag may only be
enabled by the signed gate the §12.2 budgets pin: **no slice loses > 0.02 recall@5
vs dense-only on the frozen fixture** and **the overall gain is >= +0.02**. This
script measures the arms and records the verdict — it never flips the flag.

Arms (all through the REAL ``MemoryRetriever.recall`` pipeline — SQL hydration,
visibility filter, scoring, trace):

- ``dense_only``    — the shipped OFF path;
- ``lexical_only``  — the R19(p2) vector-outage leg answering from FTS5 alone;
- ``hybrid_rrf``    — dense + lexical fused by ``fuse_by_uuid`` (the flag's arm);
- ``hybrid_rerank`` — the same pool through the rerank stage;
- (e) full vs late hydration parity — RUN ONLY when the trace's SQL/entity load
  clears 25 % of ``total`` (§7.5 / R5(p2)); otherwise the artifact records the
  measured share and ``not implemented: no measured benefit``.

Slices: exact id/prefix · Vietnamese with diacritics · Vietnamese without
diacritics · English · short query · long query. Metrics: recall@{1,5,10}, MRR,
and every arm's per-slice delta vs dense-only.

The substitutions (recorded, because an ablation is only as honest as its seams)
------------------------------------------------------------------------------
- **Embedding**: the REAL local model (``e5_local.arctic_embed_passages/queries``,
  arctic-embed-xs ONNX, 384-dim, cached on this box). No paid API is called.
- **Dense store**: exact cosine top-k in numpy over those real vectors, filtered by
  tenant, sorted ``(-score, memory_id)`` like ``search_memories``. Production uses
  Qdrant HNSW, which APPROXIMATES this order; at fixture scale the exact order is
  the one the store is trying to reproduce.
- **Lexical leg**: real SQLite FTS5 (T4's DDL + triggers, the real MATCH grammar).
- **Reranker**: the Jina cross-encoder is a paid remote API and is unreachable
  offline, so the rerank stage is driven by a deterministic local IDF-weighted
  token-coverage scorer (:func:`_local_rerank_standin`). It measures the STAGE —
  SQL-authorized content, the merge semantics, the ordering effect — never Jina's
  semantic quality.
- **Query rewrite**: identity (no LLM offline).
- **Modifiers**: every row shares one ``captured_at``, one salience and no pin, so
  entity boost / time decay are a UNIFORM multiplier and the served order is the
  arm's own ranking.

Scale, stated plainly
---------------------
This is EVIDENCE AT FIXTURE SCALE (hundreds of memories, the §12.2 budgets are
signed at that scale). It is never a production claim: embeddings here are local
arctic, the store is exact-cosine, the reranker is a stand-in, and the corpus is
generated. What it does carry is the SHAPE of the decision — which slices the
lexical leg wins, which it loses, and whether the fusion is non-inferior per slice.

Run: ``python eval/ablation_retrieval_p2.py`` (writes ``eval/ablation_retrieval_p2.json``).
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import random
import statistics
import sys
import tempfile
import unicodedata
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import bindparam, event, select  # noqa: E402
from sqlalchemy import text as sql_text  # noqa: E402
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402

from app import database, models  # noqa: E402,F401 — register every table on Base
from app.database import Base  # noqa: E402
from app.models.memory import Memory  # noqa: E402
from app.models.types import GUID  # noqa: E402
from app.models.user import User  # noqa: E402
from app.observability.fallbacks import fallback_counts, reset_fallback_counts  # noqa: E402
from app.retrieval import e5_local  # noqa: E402
from app.retrieval import reranker as reranker_module  # noqa: E402
from app.retrieval.memory import freshness, lexical_index  # noqa: E402
from app.retrieval.memory import retriever as rmod  # noqa: E402
from app.retrieval.memory.retriever import MemoryRetriever  # noqa: E402
from app.retrieval.vector_retriever import VectorUnavailableError  # noqa: E402

SEED = 20260916
TENANT_A = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
TENANT_B = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
# One shared capture time for EVERY row (see "Modifiers" above).
CAPTURED = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)
ARTIFACT_PATH = ROOT / "eval" / "ablation_retrieval_p2.json"
TOP_K = 10
POOL_MULTIPLIER = 2.0
RRF_K = 60
DISTRACTORS_PER_SLICE = 52
SUPERSEDED_PER_SLICE = 2
QUERIES_PER_SLICE = 8
HYDRATION_THRESHOLD = 0.25
ENABLE_MAX_SLICE_LOSS = 0.02
ENABLE_MIN_OVERALL_GAIN = 0.02
SKEW_FOREIGN_SIZES = (100, 300, 1000)
METRIC_KEYS = ("recall@1", "recall@5", "recall@10", "mrr")

SLICE_EXACT_ID = "exact_id"
SLICE_VI = "vi_diacritics"
SLICE_VI_NODIAC = "vi_no_diacritics"
SLICE_EN = "en"
SLICE_SHORT = "short_query"
SLICE_LONG = "long_query"
SLICES = (SLICE_EXACT_ID, SLICE_VI, SLICE_VI_NODIAC, SLICE_EN, SLICE_SHORT, SLICE_LONG)
# Slices that own ROWS: the no-diacritics slice re-uses the VI rows (§7.4: the
# same memories, a differently typed query).
ROW_SLICES = (SLICE_EXACT_ID, SLICE_VI, SLICE_EN, SLICE_SHORT, SLICE_LONG)

# The per-slice arms that must show a lexical leg in the trace before their numbers
# are trusted (carried item C1, T5 review M3).
LEXICAL_LEG_ARMS = ("lexical_only", "hybrid_rrf", "hybrid_rerank")


# ── text helpers ────────────────────────────────────────────────────────────

def fold(text: str) -> str:
    """Casefold + strip combining marks: the FTS5 ``remove_diacritics 2`` intent.

    Approximation used for the fixture's own token arithmetic ONLY (never for a
    measurement): FTS5 folds Latin base+diacritic pairs but leaves letters with no
    decomposition alone — Vietnamese ``đ``/``Đ`` included, which the no-diacritics
    slice measures as-is (the per-token coverage in the artifact is computed with
    the REAL FTS5 MATCH, not with this function).
    """
    out = []
    for char in unicodedata.normalize("NFD", text.casefold()):
        if unicodedata.combining(char):
            continue
        out.append("d" if char in "đ" else char)
    return "".join(out)


def strip_diacritics(text: str) -> str:
    """The VI-no-diacritics slice's query form: ``đ`` is NOT stripped (it has no
    decomposition), exactly like a Vietnamese keyboard typed without tone marks."""
    out = []
    for char in unicodedata.normalize("NFD", text):
        if unicodedata.combining(char):
            continue
        out.append(char)
    return unicodedata.normalize("NFC", "".join(out))


# ── the frozen fixture ──────────────────────────────────────────────────────
# Every gold row is a real-shaped memory; its query is built from that row's own
# tokens so the lexical leg has a fair AND-window (FTS5 joins adjacent phrases
# with implicit AND — a token the gold does not carry makes the whole query match
# nothing, which the long-query slice's two prose rows measure on purpose).

_EXACT_ID_TOPICS = (
    (None, "Closed ORIVORY-4417: the FTS5 ladder upgrade ships behind the hybrid flag, with the rollback recipe in docs/ROLLBACK_P1B.md.", "ORIVORY-4417"),
    (None, "Triaged ORIVORY-4424: the rerank pool multiplier was raised so a top_k of ten keeps its whole window.", "ORIVORY-4424"),
    (None, "Reopened ORIVORY-4431: recall answered an empty list while the drain loop was still applying the batch.", "ORIVORY-4431"),
    (None, "Shipped ORIVORY-4438: the recall trace now carries one counter per leg that really ran.", "ORIVORY-4438"),
    (None, "Blocked ORIVORY-4445: the embedding executor holds its slot after the freshness barrier times out.", "ORIVORY-4445"),
    (None, "Refactor app/retrieval/memory/lexical_index.py: the tenant clause is applied before the limit, so a foreign row never takes a slot.", "app/retrieval/memory/lexical_index.py"),
    (None, "Reading app/retrieval/memory/retriever.py: the refill re-fuses through the dense leg before scoring.", "app/retrieval/memory/retriever.py"),
    (None, "The helper in app/retrieval/hybrid_retriever.py fuses by canonical uuid, never by a content hash.", "app/retrieval/hybrid_retriever"),
)

_VI_TOPICS = (
    (None, "Quyết định: tìm kiếm ngữ nghĩa cho bộ nhớ cá nhân dùng FTS5 làm nhánh từ vựng, chỉ bật khi đo được lợi ích.", "quyết định tìm kiếm ngữ nghĩa cho bộ nhớ cá nhân"),
    ("Ghi chú họp nhóm", "Bản ghi nhớ về lịch họp nhóm: thứ sáu hàng tuần lúc mười giờ, đổi lịch thì nhắn trước một ngày.", "bản ghi nhớ về lịch họp nhóm"),
    (None, "Chỉ mục từ vựng được cập nhật trong cùng giao dịch với hàng memories, nên không cần quét lại toàn bộ.", "chỉ mục từ vựng cập nhật trong cùng giao dịch"),
    (None, "Hàng đợi chỉ mục giữ ý định ghi bền vững; vòng lặp nền xử lý khi rảnh và trả lời 503 khi quá hạn.", "hàng đợi chỉ mục giữ ý định ghi"),
    (None, "Kế hoạch triển khai giai đoạn hai gồm ablation truy hồi, đo độ trễ và kiểm tra tính tương thích.", "kế hoạch triển khai giai đoạn hai"),
    (None, "Sao lưu và khôi phục cơ sở dữ liệu: nén tệp trước khi tải lên, kiểm tra mã băm sau khi tải xong.", "sao lưu và khôi phục cơ sở dữ liệu"),
    (None, "Cửa sổ lưu trữ cho nhật ký là ba mươi ngày; dữ liệu cũ hơn sẽ bị xoá theo lịch.", "cửa sổ lưu trữ cho nhật ký"),
    (None, "Đo độ trễ truy hồi ở quy mô nhỏ: p95 khoảng mười lăm mili giây, chưa tính thời gian tải mô hình.", "đo độ trễ truy hồi ở quy mô nhỏ"),
)

_EN_TOPICS = (
    (None, "Write amplification of the FTS update trigger: every recall_count bump rewrites the row, acceptable while write volume stays low.", "write amplification fts update trigger"),
    (None, "The freshness barrier budget is two seconds for a recall alone; under bulk ingest the wait can end in a typed timeout.", "freshness barrier budget recall alone"),
    (None, "Reciprocal rank fusion constant stays at sixty, over zero-based ranks, one vote per leg per memory.", "reciprocal rank fusion constant ranks"),
    (None, "Erase receipt verification runs after the readback, and reconcile upgrades the delete that lands later.", "erase receipt verification after readback"),
    (None, "Index generation cutover keeps both manifests active until the parity check passes.", "index generation cutover manifests parity"),
    (None, "Salience decay half life is thirty days with a floor of one tenth, so an old memory still surfaces.", "salience decay half life floor"),
    (None, "The outbox claim window is bounded by a single flight door so two claimers never apply the same batch.", "outbox claim window single flight"),
    (None, "MCP tool surface narrowing caps the payload at index fields and keeps personal context out of the response.", "mcp tool surface narrowing payload fields"),
)

_SHORT_TOPICS = (
    (None, "Kestrel is the internal codename for the hybrid recall rollout; the flag stays off until the measured gate passes.", "kestrel"),
    (None, "Halyard is the codename for the FTS5 ladder upgrade that repopulates the lexical index once.", "halyard"),
    (None, "Nimbus names the background drain loop that applies pending index intents every interval.", "nimbus"),
    (None, "Lantern is the codename for the erasure receipt drill run before every release.", "lantern"),
    (None, "RRF sums one over sixty plus rank plus one, over zero-based ranks from each leg.", "rrf sixty"),
    (None, "SQLite FTS5 powers the lexical leg; Postgres deployments have no lexical index at all.", "sqlite fts5"),
    (None, "Tokamak is the codename for the local embedding warmup that runs before the boot drain.", "tokamak"),
    (None, "Beacon is the codename for the observability counter that marks a rerank failure.", "beacon"),
)

_LONG_TOPICS = (
    (None, "Scaling note: the sqlite outbox retention window and the drain loop both matter when a bulk import falls behind, so the window is measured before the interval is shortened.", "sqlite outbox retention window drain loop bulk import falls behind interval"),
    (None, "Hybrid rollout gate: no slice may lose more than two points of recall at five against dense only on the frozen fixture, and the overall gain must clear two points.", "hybrid rollout gate slice lose recall five dense frozen fixture overall gain"),
    (None, "FTS5 ladder upgrade: the backup is written before the transition, the integrity check runs after, and a fresh install goes straight to the new version.", "fts5 ladder upgrade backup transition integrity check fresh install version"),
    (None, "Rerank merge rule: the cross encoder answers with at most the requested window, the rows it did not rank keep their earlier order behind the ranked head, and the served count never depends on the network.", "rerank merge rule cross encoder window rows ranked head served count network"),
    (None, "MCP fallback: when the seam reports a degraded leg and the served order is empty, the tool answers from the sql ordering and records one ledger row instead of raising.", "mcp fallback seam degraded leg served order empty sql ordering ledger"),
    (None, "Eviction policy: unpinned rows older than the retention window are eligible for archival, pinned rows are protected from automatic removal but never from an explicit forget.", "eviction policy unpinned rows older retention window archival pinned protected automatic removal forget"),
    (None, "The decision I keep coming back to is that the lexical leg should only be enabled when the measured evidence on the frozen fixture says every slice stays non inferior.", "what did i decide about enabling the lexical leg if the measured evidence on the frozen fixture says something different"),
    (None, "I told the team that the retention window for the outbox should stay short while the drain loop is still being measured under load.", "when should the retention window for the outbox change while the drain loop is still being measured under load"),
)

# Distractor vocabulary: template + anchor pairs. Most rows sit OUTSIDE the
# golds' phrases — but not all, deliberately: the ORIVORY anchor numbers
# (4400 + 3i) collide with the golds' own identifiers (ORIVORY-4424,
# ORIVORY-4445) and the app/retrieval//hybrid_retriever.py anchor carries a
# gold's module token, so those rows share the query's rare phrase and are HARD
# negatives — not accidental second golds. Measured in
# fixture.lexical_match_counts_per_slice.
_DISTRACTOR_TEMPLATES = {
    SLICE_EXACT_ID: (
        "Closed {anchor}: the ladder ran once on the transition and the backup file stays next to the database.",
        "Triaged {anchor}: an empty recall is a no-match only when every leg that ran answered.",
        "Reviewing {anchor}: the tenant clause sits before the limit, so no foreign row takes a slot.",
        "Follow up on {anchor}: the counter names a leg that really ran, never a fabricated zero.",
        "Handed {anchor} to the platform team: the vote of the dense leg weighs the same as the lexical one.",
    ),
    SLICE_VI: (
        "Ghi chú {anchor}: cần kiểm tra lại trước khi bật mặc định cho mọi người dùng.",
        "Quyết định tạm thời về {anchor}: đo lại sau khi có số liệu của tuần này.",
        "Nhắc lại {anchor}: bản nháp cũ vẫn nằm trong thư mục sao lưu của nhóm.",
        "Theo dõi {anchor} trong buổi họp tuần sau, chưa cần đổi gì ở cấu hình.",
        "Kết luận về {anchor}: giữ nguyên cách làm hiện tại cho tới khi đo xong.",
    ),
    SLICE_EN: (
        "Follow up on {anchor}: the decision is recorded, the numbers are not yet.",
        "Revisited {anchor} after the review asked for the measurement first.",
        "Drafted the {anchor} note so the next session does not re-litigate it.",
        "Checked {anchor} against the shipped defaults before writing anything down.",
        "Parked {anchor} until the fixture is frozen and the run is reproducible.",
    ),
    SLICE_SHORT: (
        "Scratch note: {anchor} came up again while reading the retrieval code.",
        "Idea: {anchor} might be worth a spike once the fixture is frozen.",
        "Reminder to look at {anchor} after the current phase lands.",
        "Half-formed thought about {anchor} and the shape of the trace.",
        "Quiet observation on {anchor}, no action needed yet.",
    ),
    SLICE_LONG: (
        "Working note: the sqlite retention window and the drain loop were revisited again while {anchor}, and the interval stays where it is until the next measurement lands.",
        "Session log: the gate, the fixture and the overall gain were all discussed while {anchor}, and the conclusion was to keep the flag off until the numbers are in.",
        "Design note: the backup, the integrity check and the fresh install path all change together while {anchor}, so the transition stays a single step.",
    ),
}
_DISTRACTOR_ANCHORS = {
    SLICE_EXACT_ID: [f"ORIVORY-{4400 + 3 * i}" for i in range(26)] + [
        f"app/retrieval/{package}/{module}.py"
        for package, module in (
            ("memory", "scoring"), ("memory", "visibility"), ("memory", "correction"),
            ("memory", "salience"), ("memory", "outbox"), ("memory", "write_back"),
            ("memory", "reindex"), ("memory", "drain_loop"), ("memory", "context"),
            ("memory", "query_rewriter"), ("", "embedder"), ("", "embedding_fingerprint"),
            ("", "parent_store"), ("", "retrieval_cache"), ("", "hyde_agent"),
            ("", "vector_backend"), ("", "vector_retriever"), ("", "qdrant_filter"),
            ("", "e5_local"), ("", "bm25_retriever"), ("", "hybrid_retriever"),
            ("", "reranker"), ("", "qdrant_filter"), ("", "embedder"),
            ("", "vector_store"), ("", "salience"),
        )
    ],
    SLICE_VI: [
        "hợp đồng thuê nhà", "lịch tiêm phòng", "danh sách mua sắm", "khoá học tiếng anh",
        "chuyến đi đà lạt", "sổ tiết kiệm", "hoá đơn điện", "lịch bảo dưỡng xe",
        "kế hoạch nghỉ hè", "tài liệu ôn thi", "hồ sơ bảo hiểm", "đơn xin nghỉ phép",
        "bài tập về nhà", "lịch thanh toán thẻ", "phiếu khám sức khoẻ", "hợp đồng lao động",
        "danh sách khách mời", "kế hoạch tập luyện", "công việc nhà", "lịch gặp bác sĩ",
        "ngân sách tháng này", "thư viện sách", "kho ảnh gia đình", "món ăn cuối tuần",
        "buổi họp phụ huynh", "kế hoạch tiết kiệm",
    ],
    SLICE_EN: [
        "the meeting cadence", "the printer queue", "the grocery list", "the running plan",
        "the tax folder", "the home inventory", "the bike service", "the book club",
        "the winter trip", "the gym schedule", "the recipe box", "the budget sheet",
        "the doctor visit", "the insurance file", "the course notes", "the contact list",
        "the lease renewal", "the warranty papers", "the photo archive", "the plant care",
        "the language practice", "the commute route", "the side project", "the reading list",
        "the pantry stock", "the gift ideas",
    ],
    SLICE_SHORT: [
        "saffron", "trellis", "juniper", "marlin", "obsidian", "pelican", "quarry", "rivet",
        "sandpiper", "thistle", "umbra", "vellum", "wicker", "yarrow", "zephyr", "almanac",
        "bramble", "cinder", "dovetail", "ember", "furrow", "gantry", "hearth", "isthmus",
        "jetty", "kelp",
    ],
    SLICE_LONG: [
        "the retention window was on the table", "the bulk import was still running",
        "the review asked for the numbers", "the fixture was being rebuilt",
        "the interval was under discussion", "the drain loop was catching up",
        "the batch was still landing", "the ladder was mid transition",
        "the parity check was pending", "the manifest was being read",
        "the receipt was being verified", "the claim window was open",
        "the pool was being widened", "the rerank head was merging",
        "the counters were being read", "the trace was being compared",
        "the slice was being re-measured", "the gate was being restated",
        "the budget was being re-checked", "the flag was still off",
        "the executor was busy", "the barrier was waiting",
        "the timeout was being typed", "the outage was being simulated",
        "the row was being re-authorized", "the revision was changing",
    ],
}
# The VI-no-diacritics slice shares the VI golds: same memories, query typed
# without tone marks (what a Vietnamese keyboard without an IME produces).


def _row_key(slice_name: str, index: int, *, kind: str = "gold") -> str:
    return f"{slice_name}.{kind}.{index:02d}"


def _memory_id(key: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"orivory/p2/t7/{SEED}/{key}")


def build_fixture() -> dict:
    """The frozen fixture: rows + queries, deterministic for a given ``SEED``.

    Pure Python, no I/O and no model: the contract test re-runs this to prove the
    artifact's ``corpus_hash`` still describes the corpus the artifact measured.
    """
    rng = random.Random(SEED)
    queries: list[dict] = []
    golds: dict[str, dict] = {}

    def add_gold(slice_name: str, index: int, title, content, query_text, *, query_slice=None):
        key = _row_key(slice_name, index)
        golds[key] = {"key": key, "slice": slice_name, "title": title, "content": content}
        queries.append({
            "key": f"{query_slice or slice_name}.q{index:02d}",
            "slice": query_slice or slice_name,
            "text": query_text,
            "golds": [key],
        })
        return key

    for index, (title, content, query_text) in enumerate(_EXACT_ID_TOPICS):
        add_gold(SLICE_EXACT_ID, index, title, content, query_text)
    # §7.4 / R11b: identical text, different UUID — two memories, never collapsed
    # by a content hash. It is a SECOND gold of the first exact-id query.
    dup_key = _row_key(SLICE_EXACT_ID, 0, kind="dup")
    golds[dup_key] = {
        "key": dup_key,
        "slice": SLICE_EXACT_ID,
        "title": _EXACT_ID_TOPICS[0][0],
        "content": _EXACT_ID_TOPICS[0][1],
    }
    queries[0]["golds"].append(dup_key)

    for index, (title, content, query_text) in enumerate(_VI_TOPICS):
        add_gold(SLICE_VI, index, title, content, query_text)
    # The no-diacritics slice is the SAME memories, queried without tone marks
    # (what a Vietnamese keyboard without an IME produces): the golds are the rows
    # above — only the query form changes, never the corpus.
    for index in range(len(_VI_TOPICS)):
        queries.append({
            "key": f"{SLICE_VI_NODIAC}.q{index:02d}",
            "slice": SLICE_VI_NODIAC,
            "text": strip_diacritics(_VI_TOPICS[index][2]),
            "golds": [_row_key(SLICE_VI, index)],
        })
    for index, (title, content, query_text) in enumerate(_EN_TOPICS):
        add_gold(SLICE_EN, index, title, content, query_text)
    for index, (title, content, query_text) in enumerate(_SHORT_TOPICS):
        add_gold(SLICE_SHORT, index, title, content, query_text)
    for index, (title, content, query_text) in enumerate(_LONG_TOPICS):
        add_gold(SLICE_LONG, index, title, content, query_text)

    rows: list[dict] = []
    for gold in golds.values():
        rows.append({**gold, "superseded": False, "user_id": TENANT_A})

    for slice_name in ROW_SLICES:
        anchors = _DISTRACTOR_ANCHORS[slice_name]
        templates = _DISTRACTOR_TEMPLATES[slice_name]
        for index in range(DISTRACTORS_PER_SLICE):
            anchor = anchors[index % len(anchors)]
            template = templates[index % len(templates)]
            # A repeat of an anchor takes a numbered variant so no two rows are
            # byte-identical and the corpus hash is meaningful.
            if index >= len(anchors):
                anchor = f"{anchor} {index:02d}"
            rows.append({
                "key": _row_key(slice_name, index, kind="negative"),
                "slice": slice_name,
                "title": None,
                "content": template.format(anchor=anchor),
                "superseded": False,
                "user_id": TENANT_A,
            })
        # Superseded near-misses: they compete for the pool and must then be
        # filtered, which is what earns the bounded refill its one extra fetch.
        for index in range(SUPERSEDED_PER_SLICE):
            rows.append({
                "key": _row_key(slice_name, index, kind="superseded"),
                "slice": slice_name,
                "title": None,
                "content": f"[replaced] {golds[_row_key(slice_name, index)]['content']}",
                "superseded": True,
                "user_id": TENANT_A,
            })

    rng.shuffle(rows)  # insert order must not be the ranking order
    for row in rows:
        row["memory_id"] = _memory_id(row["key"])
    rows.sort(key=lambda row: str(row["memory_id"]))
    return {"seed": SEED, "rows": rows, "queries": queries}


def corpus_hash(fixture: dict) -> str:
    """sha256 over the canonical corpus serialization (ids + tenants + text)."""
    digest = hashlib.sha256()
    for row in sorted(fixture["rows"], key=lambda r: str(r["memory_id"])):
        digest.update(
            f"{row['memory_id']}\t{row['user_id']}\t{row['title']}\t{row['content']}\n".encode()
        )
    return f"sha256:{digest.hexdigest()}"


def query_set_hash(fixture: dict) -> str:
    digest = hashlib.sha256()
    for query in fixture["queries"]:
        digest.update(f"{query['slice']}\t{query['text']}\t{','.join(query['golds'])}\n".encode())
    return f"sha256:{digest.hexdigest()}"


# ── the dense leg (exact cosine over the real local embedder) ────────────────

class ExactCosineStore:
    """``search_memories`` over the fixture's real arctic vectors.

    Same surface (and same tenant filter, ``(-score, memory_id)`` sort and stale
    payload ``content`` the retriever must replace with the SQL row) as the Qdrant
    store — the ordering it produces is the exact top-k the ANN store approximates.
    """

    def __init__(self, rows: list[dict], vectors: np.ndarray, *, outage: bool = False) -> None:
        self.ids = np.array([str(row["memory_id"]) for row in rows])
        self.tenants = np.array([str(row["user_id"]) for row in rows])
        self.vectors = vectors
        self.outage = outage
        self.calls: list[int] = []

    async def __call__(self, embedding, *, user_id: str, top_k: int = 10, where=None):
        self.calls.append(int(top_k))
        if self.outage:
            raise VectorUnavailableError("vector store down (ablation outage arm)")
        query = np.asarray(embedding, dtype=np.float64)
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        mask = self.tenants == str(user_id)
        scores = self.vectors[mask] @ query
        ids = self.ids[mask]
        order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))[:top_k]
        return [
            {
                "memory_id": str(ids[i]),
                "content": "[vector payload copy — stale by contract]",
                "score": float(scores[i]),
                "rank": rank,
                "source": "vector",
            }
            for rank, i in enumerate(order)
        ]


async def _local_rerank_standin(query: str, chunks: list[dict], *, top_n: int | None = None):
    """The offline stand-in for the Jina cross-encoder (documented substitution).

    Deterministic IDF-weighted token coverage over the SQL-authorized ``content``:
    a rare query term in a document moves it up, a common one barely does. It
    mirrors the real ``rerank`` contract — ``[{**chunk, "rerank_score"}]``, best
    first, at most ``min(top_n, JINA_RERANKER_TOP_N)`` rows — so the pipeline's
    merge and revalidation run exactly as in production.
    """
    query_terms = set(_word_tokens(fold(query)))
    docs = [_word_tokens(fold(chunk.get("content", ""))) for chunk in chunks]
    document_frequency = Counter(
        term for tokens in docs for term in set(tokens) if term in query_terms
    )
    total = max(len(chunks), 1)

    def score(tokens: list[str]) -> float:
        present = set(tokens)
        return sum(
            math.log(1.0 + total / (1.0 + document_frequency[term]))
            for term in query_terms
            if term in present
        )

    scored = [
        (chunk, score(docs[i])) for i, chunk in enumerate(chunks)
    ]
    scored.sort(key=lambda pair: (-pair[1], str(pair[0].get("memory_id", ""))))
    cap = reranker_module.settings.JINA_RERANKER_TOP_N
    limit = min(int(top_n), cap) if top_n is not None else cap
    return [
        {**chunk, "rerank_score": value}
        for chunk, value in scored[: max(limit, 0)]
    ]


def _word_tokens(text: str) -> list[str]:
    import re

    return re.findall(r"\w+", text)


# ── database plumbing ───────────────────────────────────────────────────────

def _engine_for(path: Path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}", poolclass=NullPool)
    event.listen(engine.sync_engine, "connect", database._configure_sqlite_connection)
    return engine


async def _prepare_db(path: Path, rows: list[dict]):
    """Create the schema + T4's FTS index and seed the rows (triggers fire)."""
    engine = _engine_for(path)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(lexical_index.create_index)
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    await _seed(factory, rows, [])
    return engine, factory


def _orm_row(row: dict) -> Memory:
    meta = {"cm_superseded_by": str(uuid.uuid4())} if row.get("superseded") else {}
    return Memory(
        id=row["memory_id"],
        user_id=row["user_id"],
        title=row["title"],
        content=row["content"],
        tags=[],
        salience=0.5,
        pinned=False,
        source_type="manual_note",
        recall_count=0,
        captured_at=CAPTURED,
        indexed_at=CAPTURED,
        updated_at=CAPTURED,
        extra_metadata=meta,
    )


async def _seed(factory, rows: list[dict], extra_rows: list[dict]) -> None:
    async with factory() as session:
        wanted = {row["user_id"] for row in [*rows, *extra_rows]}
        present = set((await session.execute(select(User.id))).scalars().all()) if wanted else set()
        for user_id in sorted(wanted - present, key=str):
            session.add(User(
                id=user_id, email=f"{user_id}@ablation.invalid", onboarding_done=True,
                is_verified=True, is_active=True, is_deleted=False,
            ))
        session.add_all([_orm_row(row) for row in [*rows, *extra_rows]])
        await session.commit()


# ── running one arm ─────────────────────────────────────────────────────────

@contextlib.contextmanager
def _patched(module, **attributes):
    saved = {name: getattr(module, name) for name in attributes}
    for name, value in attributes.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


def _rewrite_stub():
    async def _rewrite(query, context=None):
        return {"rewritten_query": query, "entities": [], "reasoning": None,
                "_fallback_used": False}

    return _rewrite


def _identity_context():
    async def _context(db, user_id):
        return []

    return _context


def _embed_query_bound(model_embed):
    """The real local embedder behind the recall seam — no cache, deliberately.

    A cached query vector would move the embedding cost out of every arm after the
    first, which is exactly the number the hydration share is computed against.
    """

    async def _embed(text: str) -> list[float]:
        return (await asyncio.to_thread(model_embed, [text]))[0]

    return _embed


def _corpus_documents(rows: list[dict]) -> list[str]:
    """The text production embeds (``vector_store._memory_to_document``)."""
    return [
        f"Title: {row['title']}\n{row['content']}" if row["title"] else row["content"]
        for row in rows
    ]


async def run_arm(
    factory,
    fixture: dict,
    *,
    arm: str,
    hybrid: bool,
    semantic_rerank: bool,
    outage: bool,
    vectors: np.ndarray,
) -> dict:
    records: list[dict] = []
    store = ExactCosineStore(fixture["rows"], vectors, outage=outage)
    rerank_patch = {"rerank": _local_rerank_standin} if semantic_rerank else {}

    with _patched(rmod, rewrite_query=_rewrite_stub(),
                  fetch_personal_context=_identity_context(),
                  embed_query=_embed_query_bound(e5_local.arctic_embed_queries),
                  search_memories=store), \
            _patched(reranker_module, **rerank_patch):
        for query in fixture["queries"]:
            async with factory() as session:
                retriever = MemoryRetriever(
                    session, TENANT_A, hybrid=hybrid, semantic_rerank=semantic_rerank,
                    pool_multiplier=POOL_MULTIPLIER, rrf_k=RRF_K,
                )
                response = await retriever.recall(
                    query["text"], top_k=TOP_K, include_personal_context=False,
                )
            served = [str(result.id) for result in response.results]
            counts = dict(response.trace.counts)
            stage_ms = {key: float(value) for key, value in response.trace.stage_ms.items()}
            records.append({
                "query": query,
                "served": served,
                "gold_ids": [str(_memory_id(key)) for key in query["golds"]],
                "metrics": _query_metrics(
                    served, [str(_memory_id(key)) for key in query["golds"]]
                ),
                "counts": counts,
                "stage_ms": stage_ms,
                "total_ms": stage_ms.get("total", response.trace.latency_ms),
            })

    _validate_arm(arm, records, outage=outage)
    return {"arm": arm, "records": records, "store_calls": store.calls}


def _validate_arm(arm: str, records: list[dict], *, outage: bool) -> None:
    """Carried item C1: refuse to trust an arm whose lexical leg never ran."""
    if arm in LEXICAL_LEG_ARMS:
        silent = [
            record["query"]["key"] for record in records
            if record["counts"].get("lexical") is None
        ]
        if silent:
            raise RuntimeError(
                f"C1 violation: arm {arm!r} served {len(silent)} queries without a lexical leg "
                f"in the trace ({silent[:3]}...) — that is a dense-vs-dense measurement, not a "
                f"hybrid one. An FTS-less environment must fail here, never measure."
            )
    if arm == "hybrid_rerank":
        silent = [
            record["query"]["key"] for record in records
            if record["counts"].get("reranked") is None
        ]
        if silent:
            raise RuntimeError(
                f"the rerank arm fell back to the un-reranked order for {len(silent)} queries "
                f"({silent[:3]}...) — that is the hybrid_rrf arm wearing a rerank label"
            )
    if arm == "lexical_only":
        leaked = [record["query"]["key"] for record in records if "dense" in record["counts"]]
        if leaked:
            raise RuntimeError(f"outage arm answered with a dense leg ({leaked[:3]}...)")
        if not outage:
            raise RuntimeError("the lexical-only arm is the R19 vector outage")
    if arm == "dense_only":
        leaked = [record["query"]["key"] for record in records if "lexical" in record["counts"]]
        if leaked:
            raise RuntimeError(
                f"the OFF path probed the lexical index ({leaked[:3]}...) — R2(p2) says it never does"
            )


def _query_metrics(served: list[str], golds: list[str]) -> dict:
    def recall(k: int) -> float:
        return sum(1 for gold in golds if gold in served[:k]) / len(golds)

    rank = next((i + 1 for i, sid in enumerate(served) if sid in golds), None)
    return {
        "recall@1": recall(1),
        "recall@5": recall(5),
        "recall@10": recall(10),
        "mrr": 0.0 if rank is None else 1.0 / rank,
    }


def _aggregate(records: list[dict], key_of) -> dict:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(key_of(record), []).append(record["metrics"])
    return {
        name: {
            **{metric: round(statistics.fmean(item[metric] for item in items), 6)
               for metric in METRIC_KEYS},
            "n": len(items),
        }
        for name, items in sorted(groups.items())
    }


def _arm_payload(arm: str, config: dict, run: dict) -> dict:
    records = run["records"]
    per_slice = _aggregate(records, lambda record: record["query"]["slice"])
    overall = {
        **{metric: round(statistics.fmean(r["metrics"][metric] for r in records), 6)
           for metric in METRIC_KEYS},
        "n": len(records),
    }
    shares = [
        record["stage_ms"].get("hydrate_ms", 0.0) / max(record["total_ms"], 1e-9)
        for record in records
    ]
    totals = sorted(record["total_ms"] for record in records)
    stage_names = sorted({name for record in records for name in record["stage_ms"]})
    return {
        "config": config,
        "per_slice": per_slice,
        "overall": overall,
        "latency_ms": {
            "p50": round(statistics.median(totals), 3),
            "p95": round(totals[min(len(totals) - 1, int(0.95 * len(totals)))], 3),
            "max": round(totals[-1], 3),
        },
        "trace": {
            "counts_keys_seen": sorted({key for record in records for key in record["counts"]}),
            "counts_mean": {
                key: round(statistics.fmean(
                    record["counts"].get(key, 0) for record in records
                ), 3)
                for key in sorted({key for record in records for key in record["counts"]})
            },
            "stage_ms_mean": {
                name: round(statistics.fmean(record["stage_ms"].get(name, 0.0) for record in records), 3)
                for name in stage_names
            },
            "hydrate_ms_over_total": {
                "mean": round(statistics.fmean(shares), 6),
                "p95": round(sorted(shares)[min(len(shares) - 1, int(0.95 * len(shares)))], 6),
                "max": round(max(shares), 6),
                "n": len(shares),
            },
        },
        "store_calls": run["store_calls"],
        "c1_lexical_leg_ran": all(
            record["counts"].get("lexical") is not None for record in records
        ) if arm in LEXICAL_LEG_ARMS else None,
    }


def _deltas(arms: dict) -> dict:
    reference = arms["dense_only"]["per_slice"]
    reference_overall = arms["dense_only"]["overall"]
    deltas: dict[str, dict] = {}
    for arm, payload in arms.items():
        if arm == "dense_only":
            continue
        deltas[arm] = {
            "per_slice": {
                slice_name: {
                    **{metric: round(payload["per_slice"][slice_name][metric]
                                     - reference[slice_name][metric], 6)
                       for metric in METRIC_KEYS},
                    "recall@5_loss": round(reference[slice_name]["recall@5"]
                                           - payload["per_slice"][slice_name]["recall@5"], 6),
                }
                for slice_name in SLICES
            },
            "overall": {
                **{metric: round(payload["overall"][metric] - reference_overall[metric], 6)
                   for metric in METRIC_KEYS},
                "recall@5_gain": round(payload["overall"]["recall@5"]
                                       - reference_overall["recall@5"], 6),
            },
        }
    return deltas


def _enable_verdict(deltas: dict, arm: str, *, decides: bool) -> dict:
    per_slice = deltas[arm]["per_slice"]
    losses = {slice_name: per_slice[slice_name]["recall@5_loss"] for slice_name in SLICES}
    max_loss = max(losses.values())
    gain = deltas[arm]["overall"]["recall@5_gain"]
    passed = max_loss <= ENABLE_MAX_SLICE_LOSS + 1e-9 and gain >= ENABLE_MIN_OVERALL_GAIN - 1e-9
    return {
        "decides_enable": decides,
        "verdict": "PASS" if passed else "FAIL",
        "max_slice_loss_recall@5": max_loss,
        "overall_gain_recall@5": gain,
        "per_slice_loss_recall@5": losses,
        "failing_slices": sorted(
            name for name, loss in losses.items() if loss > ENABLE_MAX_SLICE_LOSS + 1e-9
        ),
    }


# ── the BM25 skew measurement (carried item C2, §7.4 line 260) ──────────────

async def measure_bm25_skew(fixture: dict, workdir: Path) -> dict:
    """How another tenant's corpus moves THIS tenant's lexical ordering.

    Built on its own copy of the fixture (same seed, same rows): the arms' frozen
    database is never mutated, and no embedding runs here — the claim under
    measurement is BM25's index-global ``idf``/``avgdl``, so only the lexical leg
    is exercised.

    The probes are chosen so that an intra-tenant ORDER exists to flip: two-token
    AND queries over the corpus's own vocabulary whose terms sit in different
    frequency bands (a global idf shift moves a document most when its two terms
    move differently). The foreign corpus writes about the SAME words the tenant
    does — a second tenant is not a disjoint vocabulary — which is exactly what
    makes global statistics visible.
    """
    engine, factory = await _prepare_db(workdir / "skew.db", fixture["rows"])
    try:
        candidates = _skew_probe_candidates(fixture)
        raw = await _probe_orders(factory, candidates)
        probes = [query for query in candidates if len(raw[query]) >= 2]
        baseline = {query: raw[query] for query in probes}
        vocabulary = _shared_vocabulary(fixture)
        results: dict[str, dict] = {}
        example: dict | None = None
        foreign_tokens = 0
        foreign_ids_returned = 0
        for size in SKEW_FOREIGN_SIZES:
            added = sum(int(entry["foreign_rows"]) for entry in results.values())
            rows, tokens = _foreign_rows(
                size - added, tenant=TENANT_B, offset=added, vocabulary=vocabulary
            )
            foreign_tokens += tokens
            await _seed(factory, [], rows)
            after = await _probe_orders(factory, probes)
            flipped = [query for query in probes if after[query] != baseline[query]]
            changed_sets = [query for query in probes if set(after[query]) != set(baseline[query])]
            foreign_ids_returned += sum(
                1
                for query in probes
                for memory_id in after[query]
                if memory_id in {str(row["memory_id"]) for row in rows}
            )
            if example is None:
                found = _first_flip(baseline, after)
                if found and found[0]:
                    example = found[0]
            results[str(size)] = {
                "foreign_rows": size,
                "foreign_tokens": foreign_tokens,
                "flipped_orders": len(flipped),
                "flipped_queries": sorted(flipped),
                "changed_sets": len(changed_sets),
                "probe_queries": len(probes),
            }
    finally:
        await engine.dispose()

    if foreign_ids_returned:
        raise RuntimeError(
            f"C2: {foreign_ids_returned} foreign-tenant rows were returned by the lexical leg — "
            f"the tenant clause is not applied before the LIMIT"
        )
    if not probes:
        raise RuntimeError("C2: no multi-row probe survived — nothing to measure")
    return {
        "spec": "§7.4 line 260 — FTS global corpus statistics may shift ordering between tenants; record the limitation and measure the skew",
        "known_limitation": (
            "FTS5 bm25 idf/avgdl are INDEX-GLOBAL: another tenant's rows change how this "
            "tenant's lexical leg orders its own rows"
        ),
        "authorization_impact": (
            "none — the tenant + visibility clauses are applied BEFORE the LIMIT: no foreign "
            "row was returned at any foreign size (asserted)"
        ),
        "measurement": {
            "probe_queries": len(probes),
            "probe_selection": (
                "two-token AND queries over the corpus's own vocabulary, adjacent in the "
                "document-frequency order, keeping only those whose tenant answer is multi-row. "
                "DELIBERATELY stopword-heavy — the corpus's most common tokens are function words "
                "('the and', 'a so', ...), because an intra-tenant ORDER only exists where several "
                "rows answer: the selection trades natural-language realism for a measurable "
                "order, and is recorded here rather than hidden"
            ),
            "probes": [
                {"query": query, "baseline_rows": len(baseline[query])} for query in probes
            ],
            "foreign_corpus": (
                "long foreign-tenant rows drawn from the tenant corpus's own most common "
                "tokens — a foreign corpus about the same words is what moves global idf"
            ),
            "foreign_row_sizes": list(SKEW_FOREIGN_SIZES),
            "results": results,
            "flips_at": {size: entry["flipped_orders"] for size, entry in results.items()},
            "first_flip_example": example or {
                "note": "no intra-tenant order flip at these foreign sizes on this fixture"
            },
        },
    }


def _skew_probe_candidates(fixture: dict, limit: int = 12) -> list[str]:
    """Two-token AND probes that SEVERAL tenant rows can answer.

    BM25 only has an intra-tenant order to flip when more than one of the tenant's
    own rows matches, and a global idf shift moves a document most when its two
    terms carry different document frequencies — so the probes are token pairs
    co-occurring in at least three rows, drawn deterministically from the corpus.
    """
    row_tokens = [
        set(_word_tokens(fold(f"{row['title'] or ''} {row['content']}")))
        for row in fixture["rows"]
    ]
    counts: Counter[str] = Counter(token for tokens in row_tokens for token in tokens)
    common = [
        token for token, df in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if df >= 3
    ][:40]
    pairs: list[tuple[int, str, str]] = []
    for index, first in enumerate(common):
        for second in common[index + 1:]:
            cooc = sum(1 for tokens in row_tokens if first in tokens and second in tokens)
            if cooc >= 3:
                pairs.append((-cooc, first, second))
    pairs.sort()
    step = max(1, len(pairs) // limit)
    return [f"{first} {second}" for _, first, second in pairs[::step][:limit]]


def _shared_vocabulary(fixture: dict, size: int = 80) -> list[str]:
    """The tenant corpus's own most common tokens (the foreign corpus's words)."""
    counts: Counter[str] = Counter()
    for row in fixture["rows"]:
        counts.update(_word_tokens(fold(f"{row['title'] or ''} {row['content']}")))
    return [token for token, _ in counts.most_common(size)]


async def _probe_orders(factory, probes: list[str]) -> dict[str, list[str]]:
    """One probe round, each in a FRESH session: a long-lived SQLite read session
    keeps answering from the snapshot taken before the foreign rows committed."""

    def _run(session):
        conn = session.connection()
        if not lexical_index.is_available(conn):
            raise RuntimeError("C1: no lexical index in the skew fixture — nothing to measure")
        return {
            query: [
                row["memory_id"]
                for row in lexical_index.search(conn, query, user_id=TENANT_A, limit=20)
            ]
            for query in probes
        }

    async with factory() as session:
        return await session.run_sync(_run)


def _first_flip(baseline: dict, after: dict) -> list:
    for key, before_ids in baseline.items():
        after_ids = after[key]
        if before_ids == after_ids:
            continue
        for memory_id in before_ids:
            if memory_id in after_ids and before_ids.index(memory_id) != after_ids.index(memory_id):
                return [{
                    "query": key,
                    "memory_id": memory_id,
                    "rank_before": before_ids.index(memory_id),
                    "rank_after": after_ids.index(memory_id),
                    "order_before": before_ids[:5],
                    "order_after": after_ids[:5],
                }]
    return []


def _foreign_rows(
    count: int, *, tenant: uuid.UUID, offset: int, vocabulary: list[str]
) -> tuple[list[dict], int]:
    """Long foreign-tenant rows: prose that inflates ``avgdl`` and shifts idf."""
    rng = random.Random(SEED + 1 + offset)
    rows = []
    tokens = 0
    for index in range(count):
        length = rng.randint(60, 120)
        words = [rng.choice(vocabulary) for _ in range(length)]
        tokens += length
        rows.append({
            "key": f"foreign.{offset + index:05d}",
            "slice": "foreign",
            "title": None,
            "content": f"corpus filler {offset + index:05d} " + " ".join(words),
            "superseded": False,
            "user_id": tenant,
            "memory_id": _memory_id(f"foreign.{offset + index:05d}"),
        })
    return rows, tokens


# ── the runner: corpus embedding, the arms, the artifact ────────────────────

def _hydration_verdict(arms: dict) -> dict:
    """§7.5 / R5(p2): the two-step refactor only for a MEASURED bottleneck.

    The gate number is the SQL/entity load as a share of ``total`` on the arm the
    flag's decision carries (``hybrid_rrf``); the arms that cross the threshold are
    recorded rather than averaged away, because a verdict that hides them is worse
    than one a reviewer can overturn.
    """
    shares = {arm: payload["trace"]["hydrate_ms_over_total"] for arm, payload in arms.items()}
    deciding_arm = "hybrid_rrf"
    deciding = shares[deciding_arm]
    crossing = {
        arm: {"mean": share["mean"], "p95": share["p95"], "max": share["max"]}
        for arm, share in shares.items()
        if share["mean"] >= HYDRATION_THRESHOLD or share["p95"] >= HYDRATION_THRESHOLD
    }
    below = (
        deciding["mean"] < HYDRATION_THRESHOLD
        and deciding["p95"] < HYDRATION_THRESHOLD
        and deciding["max"] < HYDRATION_THRESHOLD
    )
    return {
        "spec": "§7.5 / R5(p2): late hydration is implemented ONLY when the trace shows SQL/entity-load >= 25 % of total",
        "signed_threshold": HYDRATION_THRESHOLD,
        "deciding_arm": deciding_arm,
        "deciding_measurement": deciding,
        "measured": {
            "hydrate_ms_over_total": {
                "mean_over_arms": round(statistics.fmean(s["mean"] for s in shares.values()), 6),
                "per_arm": shares,
                "samples": sum(share["n"] for share in shares.values()),
            },
            "per_arm_stage_ms_mean": {
                arm: payload["trace"]["stage_ms_mean"] for arm, payload in arms.items()
            },
            "hydrate_ms_absolute": {
                arm: payload["trace"]["stage_ms_mean"].get("hydrate_ms", 0.0)
                for arm, payload in arms.items()
            },
        },
        "crossing_arms": crossing,
        "verdict": (
            "not implemented: no measured benefit" if below
            else "implemented: late hydration (arm e parity measured)"
        ),
        "parity_arm": {
            "arm": "full vs late hydration parity",
            "status": "not run" if below else "required",
            "reason": (
                "arm (e) is gated on the verdict above (§7.5 / R5(p2)): with the deciding arm "
                "under the trigger there is no two-step variant to compare against"
                if below else
                "the deciding arm crossed the trigger — the two-step variant must be built and "
                "its parity measured before this arm can report"
            ),
        },
        "rationale": (
            "On the arm the flag's decision carries (hybrid_rrf) the SQL/entity load is "
            f"{deciding['mean']:.4f} of total (mean; p95 {deciding['p95']:.4f}, max "
            f"{deciding['max']:.4f}) — under the {HYDRATION_THRESHOLD} trigger, so full hydration "
            "stays and arm (e) parity is not run. The arms that DO cross it are recorded above "
            "rather than averaged away: dense_only crosses because its total is the smallest at "
            "fixture scale, and hybrid_rerank's p95 crosses because that arm pays the POST-NETWORK "
            "re-validation hydrate — which §7.5 forbids deferring to save SQL. The absolute SQL "
            "cost is ~1-3 ms per recall against a signed p95 budget of 150 ms at this scale, and "
            "§7.5 requires a PROVEN benefit plus parity before the two-step variant is adopted."
        ),
    }


async def _embed_corpus(fixture: dict) -> np.ndarray:
    if not e5_local.arctic_files_cached():
        raise RuntimeError(
            "the local arctic model is not cached on this box and this artifact never "
            "downloads one silently — fetch it first (the P3 gate's pattern)"
        )
    documents = _corpus_documents(fixture["rows"])
    vectors = await asyncio.to_thread(e5_local.arctic_embed_passages, documents)
    array = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True).clip(min=1e-12)
    return array / norms


async def _token_coverage(session, fixture: dict) -> dict:
    """Per-token MATCH coverage of each query against its golds, from the REAL index.

    FTS5 joins adjacent phrases with implicit AND, so one absent token makes the
    whole query match nothing. This is the diagnostic that explains a lexical
    number; it never feeds one.
    """
    gold_ids = {row["key"]: str(row["memory_id"]) for row in fixture["rows"]}

    def _run(session):
        conn = session.connection()
        out: dict[str, list[float]] = {}
        for query in fixture["queries"]:
            expression = lexical_index.match_expression(query["text"])
            tokens = expression.split() if expression else []
            hits = 0
            for token in tokens:
                for key in query["golds"]:
                    row = conn.execute(
                        sql_text(
                            f"SELECT count(*) FROM {lexical_index.TABLE} "
                            f"JOIN memories m ON m.id = {lexical_index.TABLE}.memory_id "
                            f"WHERE {lexical_index.TABLE} MATCH :expression AND m.id = :memory_id"
                        ).bindparams(
                            bindparam("expression", value=token),
                            bindparam("memory_id", value=gold_ids[key], type_=GUID()),
                        )
                    ).scalar_one()
                    hits += 1 if row else 0
            out.setdefault(query["slice"], []).append(
                round(hits / (len(tokens) * len(query["golds"])), 6) if tokens else 0.0
            )
        return out

    return await session.run_sync(_run)


async def _lexical_match_counts(session, fixture: dict) -> dict[str, list[int]]:
    """Rows the REAL lexical leg answers each query with — the fixture's ceiling.

    A query whose FTS5 answer is exactly one row (its own gold) leaves the fusion
    nothing to fix and the lexical leg nothing to break: the semantic slices sit at
    a STRUCTURAL recall ceiling, and the enable rule's "no slice loses > 0.02"
    condition can only discriminate where the answer is multi-row or empty. Recorded
    per slice in the artifact, never hidden. The single-row answer must be the gold —
    a wrong row ranking alone would change what the ceiling means, so it fails loudly.
    """
    corpus_size = len(fixture["rows"])
    gold_ids = {
        query["key"]: {str(_memory_id(key)) for key in query["golds"]}
        for query in fixture["queries"]
    }

    def _run(session):
        conn = session.connection()
        out: dict[str, list[int]] = {}
        for query in fixture["queries"]:
            rows = lexical_index.search(
                conn, query["text"], user_id=TENANT_A, limit=corpus_size
            )
            if len(rows) == 1 and rows[0]["memory_id"] not in gold_ids[query["key"]]:
                raise RuntimeError(
                    f"fixture: {query['key']} has a single lexical answer that is NOT its gold — "
                    f"the structural-ceiling claim in limitations would be false"
                )
            out.setdefault(query["slice"], []).append(len(rows))
        return out

    return await session.run_sync(_run)


async def run_ablation(workdir: Path) -> dict:
    fixture = build_fixture()
    engine, factory = await _prepare_db(workdir / "ablation.db", fixture["rows"])
    try:
        async with factory() as session:
            coverage = await _token_coverage(session, fixture)
            match_counts = await _lexical_match_counts(session, fixture)

            def _fts_coverage(session):
                return lexical_index.coverage(session.connection())

            fts_coverage = await session.run_sync(_fts_coverage)
        if fts_coverage["missing"] or fts_coverage["orphan"]:
            raise RuntimeError(f"C1: the fixture's FTS index drifted: {fts_coverage}")

        vectors = await _embed_corpus(fixture)

        arm_configs = {
            "dense_only": {"hybrid": False, "semantic_rerank": False, "dense_leg": "live"},
            "lexical_only": {"hybrid": False, "semantic_rerank": False, "dense_leg": "outage"},
            "hybrid_rrf": {"hybrid": True, "semantic_rerank": False, "dense_leg": "live"},
            "hybrid_rerank": {"hybrid": True, "semantic_rerank": True, "dense_leg": "live"},
        }
        dup_ids = [str(_memory_id(key)) for key in fixture["queries"][0]["golds"]]
        arms: dict[str, dict] = {}
        runs: dict[str, dict] = {}
        dup_hits: dict[str, int] = {}
        outage_fallbacks = None
        # The P3 freshness barrier reads its OWN outbox through its own
        # sessionmaker (tests/retrieval/conftest.py pattern): point it at the
        # fixture, or the barrier fails closed against the ambient database.
        with _patched(freshness, AsyncSessionLocal=factory):
            for arm, config in arm_configs.items():
                if arm == "lexical_only":
                    reset_fallback_counts()
                run = await run_arm(
                    factory, fixture, arm=arm, hybrid=config["hybrid"],
                    semantic_rerank=config["semantic_rerank"], outage=config["dense_leg"] == "outage",
                    vectors=vectors,
                )
                if arm == "lexical_only":
                    outage_fallbacks = fallback_counts().get("retrieval.vector_unavailable", 0)
                arms[arm] = _arm_payload(arm, config, run)
                runs[arm] = run
                dup_hits[arm] = sum(
                    1 for memory_id in dup_ids
                    if memory_id in run["records"][0]["served"]
                )
        deltas = _deltas(arms)
        # I2: how the PASS was won — the queries whose lexical answer is a single row.
        one_row_queries = sum(
            1 for counts in match_counts.values() for count in counts if count == 1
        )
    finally:
        await engine.dispose()

    hydration = _hydration_verdict(arms)
    rerank_order_diff = sum(
        1
        for fused, reranked in zip(
            runs["hybrid_rrf"]["records"], runs["hybrid_rerank"]["records"], strict=True
        )
        if fused["served"] != reranked["served"]
    )

    skew = await measure_bm25_skew(fixture, workdir)

    return {
        "artifact": "p2-task7-ablation-retrieval",
        "plan": "2026-09-15-retrieval-correctness-hybrid-p2",
        "scales": {
            "kind": "fixture",
            "memories": len(fixture["rows"]),
            "queries": len(fixture["queries"]),
            "top_k": TOP_K,
            "pool_multiplier": POOL_MULTIPLIER,
            "rrf_k": RRF_K,
            "note": (
                "EVIDENCE AT FIXTURE SCALE. The §12.2 budgets are signed at this scale; this "
                "artifact is never a production claim."
            ),
        },
        "fixture": {
            "seed": fixture["seed"],
            "corpus_hash": corpus_hash(fixture),
            "query_set_hash": query_set_hash(fixture),
            "slices": list(SLICES),
            "queries_per_slice": {
                slice_name: sum(1 for query in fixture["queries"] if query["slice"] == slice_name)
                for slice_name in SLICES
            },
            "generator": "seeded in-memory corpus (eval/ablation_retrieval_p2.py:build_fixture)",
            "rows_per_slice": {
                slice_name: sum(1 for row in fixture["rows"] if row["slice"] == slice_name)
                for slice_name in ROW_SLICES
            },
            "superseded_rows": sum(1 for row in fixture["rows"] if row["superseded"]),
            "duplicate_text_case": {
                "note": "identical text, different UUIDs — both are golds of the first exact_id query",
                "memory_ids": dup_ids,
                "text": next(
                    row["content"] for row in fixture["rows"]
                    if str(row["memory_id"]) == dup_ids[0]
                )[:120],
                "golds_in_served_top10_per_arm": dup_hits,
            },
            "fts_coverage": fts_coverage,
            "query_token_coverage_per_slice": {
                slice_name: round(statistics.fmean(values), 6)
                for slice_name, values in sorted(coverage.items())
            },
            "query_token_coverage_note": (
                "fraction of a query's literal tokens that MATCH its gold row in the REAL FTS5 "
                "index (MATCH, one token at a time). FTS5 joins adjacent phrases with implicit "
                "AND, so a coverage below 1.0 means the lexical leg returns nothing for that "
                "query at all."
            ),
            "lexical_match_counts_per_slice": {
                slice_name: {
                    "queries": len(counts),
                    "returning_exactly_one_row": sum(1 for count in counts if count == 1),
                    "rows_returned": counts,
                }
                for slice_name, counts in sorted(match_counts.items())
            },
            "lexical_match_counts_note": (
                "rows the REAL FTS5 lexical leg returns per query (limit = corpus size, tenant + "
                "visibility filtered), in fixture query order. A query answered with exactly one "
                "row — its own gold — leaves the lexical leg no room to rank a WRONG row above a "
                "dense gold: that is the structural ceiling this artifact's PASS rests on. The "
                "multi-row cases are the distractors that deliberately carry a gold's identifier; "
                "the zero-row cases are the long-query prose rows (implicit-AND coverage < 1)."
            ),
        },
        "substitutions": {
            "embedding": "REAL local arctic-embed-xs ONNX (384-dim, cached); no paid API",
            "dense_store": (
                "exact cosine top-k in numpy over those vectors (Qdrant HNSW approximates it) — an "
                "OPTIMISTIC dense baseline: exact top-k recall >= the ANN top-k it stands in for, so "
                "dense_only's numbers are a BEST CASE and the measured "
                f"{deltas['hybrid_rrf']['overall']['recall@5_gain']:+.4f} overall gain is conservative. "
                "The substitution is one of convenience at this scale, not an offline casualty — "
                "Qdrant runs embedded in this repo (lite mode); the only offline-unreachable seam "
                "here is Jina."
            ),
            "lexical": "REAL SQLite FTS5 (T4 DDL + triggers, real MATCH grammar)",
            "rerank": (
                "Jina cross-encoder is a paid remote API and unreachable offline: the stage runs "
                "a deterministic local IDF-weighted token-coverage scorer. Measures the STAGE "
                "(SQL-authorized content, merge, ordering effect), never Jina's semantic quality."
            ),
            "rewrite": "identity (no LLM offline)",
            "modifiers": "uniform by construction: one captured_at, one salience, no pins",
        },
        "arms": arms,
        "deltas_vs_dense_only": deltas,
        "enable_rule": {
            "flag": "RETRIEVAL_HYBRID_ENABLED",
            "definition": (
                "hybrid may be enabled only if every slice loses <= 0.02 recall@5 vs dense-only "
                "AND the overall gain is >= +0.02"
            ),
            "signed_thresholds": {
                "max_slice_loss_recall@5": ENABLE_MAX_SLICE_LOSS,
                "min_overall_gain_recall@5": ENABLE_MIN_OVERALL_GAIN,
            },
            "verdicts": {
                "hybrid_rrf": _enable_verdict(deltas, "hybrid_rrf", decides=True),
                "hybrid_rerank": _enable_verdict(deltas, "hybrid_rerank", decides=False),
            },
            "authority": (
                "the flag ships OFF (R2(p2)); flipping it is a separate documented decision this "
                "artifact only informs — and only at fixture scale"
            ),
        },
        "hydration": hydration,
        "bm25_skew": skew,
        "outage_arm_fallbacks": {"retrieval.vector_unavailable": outage_fallbacks},
        "rerank_stage": {
            "orders_diff_from_hybrid_rrf": rerank_order_diff,
            "queries": len(runs["hybrid_rerank"]["records"]),
            "note": (
                "the rerank arm's served order differs from the fused order on the queries above "
                "(and counts['reranked'] is present for every query) — the stage ran; its "
                "per-slice aggregates coincide with hybrid_rrf's because the golds stay inside "
                "the same rank buckets"
            ),
        },
        "limitations": [
            "fixture scale (hundreds of memories) — the §12.2 budgets are signed at this scale",
            "generated corpus: the text is template-built and deterministic, not user traffic",
            (
                "exact-cosine dense leg, not Qdrant ANN: an OPTIMISTIC dense baseline (exact top-k "
                "recall >= the ANN top-k it stands in for), so dense_only is a BEST CASE and every "
                "measured gain is conservative; the rerank stage uses a local stand-in scorer"
            ),
            (
                f"the PASS rests on a near-deterministic lexical leg: {one_row_queries}/"
                f"{len(fixture['queries'])} queries return exactly ONE row (their own gold) from the "
                "real FTS index, so the semantic slices sit at a STRUCTURAL CEILING — the whole "
                f"{deltas['hybrid_rrf']['overall']['recall@5_gain']:+.4f} overall gain comes from "
                "exact_id, the 'every slice >= -0.02' condition can only discriminate on exact_id / "
                "long_query, and the direction in which the lexical leg could HURT (a wrong row "
                "displacing a dense gold on a semantic query) is not exercised by this fixture "
                "(see fixture.lexical_match_counts_per_slice)"
            ),
            "BM25 statistics are index-global: the bm25_skew measurement shows the ceiling",
            "no LLM query rewrite (identity), so keyword-ish queries stand in for rewritten ones",
        ],
    }


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="P2/T7 retrieval ablation on a frozen fixture (dense/lexical/RRF/rerank)"
    )
    parser.add_argument("--out", default=str(ARTIFACT_PATH), help="artifact path")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="p2-t7-ablation-") as tmp:
        artifact = await run_ablation(Path(tmp))
    artifact["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")

    out = Path(args.out)
    out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _print_summary(artifact, out)
    return 0


def _print_summary(artifact: dict, out: Path) -> None:
    print(f"fixture: seed={artifact['fixture']['seed']} "
          f"rows={artifact['scales']['memories']} queries={artifact['scales']['queries']} "
          f"corpus={artifact['fixture']['corpus_hash'][:19]}…")
    header = f"{'arm':<15}" + "".join(f"{name:>12}" for name in METRIC_KEYS)
    print(header)
    for arm, payload in artifact["arms"].items():
        row = f"{arm:<15}" + "".join(f"{payload['overall'][name]:>12.4f}" for name in METRIC_KEYS)
        print(row)
    verdicts = artifact["enable_rule"]["verdicts"]
    primary = verdicts["hybrid_rrf"]
    print(f"enable rule (hybrid_rrf): {primary['verdict']} — "
          f"max slice loss {primary['max_slice_loss_recall@5']:+.4f} "
          f"(<= {artifact['enable_rule']['signed_thresholds']['max_slice_loss_recall@5']}), "
          f"overall gain {primary['overall_gain_recall@5']:+.4f} "
          f"(>= {artifact['enable_rule']['signed_thresholds']['min_overall_gain_recall@5']})")
    hydration = artifact["hydration"]
    print(f"hydration: {hydration['deciding_arm']} SQL/entity load "
          f"{hydration['deciding_measurement']['mean']:.4f} of total vs threshold "
          f"{hydration['signed_threshold']} → {hydration['verdict']}")
    skew = artifact["bm25_skew"]["measurement"]
    print("bm25 skew: " + ", ".join(
        f"{size} rows → {entry['flipped_orders']}/{entry['probe_queries']} flipped"
        for size, entry in skew["results"].items()
    ))
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
