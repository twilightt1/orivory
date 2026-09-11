# SPEC — Orivory Memory Next-Gen V1: Evidence-first Correctable Memory (Lite-only)

**Ngày chốt:** 2026-09-12 · **Trạng thái:** APPROVED scope (founder) — chờ implement
**Tiền đề:** self-host 1 container trên máy user · agent ngoài là orchestrator · server là lớp nhớ chung
**Bằng chứng nền:** `docs/research/memory-papers.md` (9 paper) · review: `docs/research/memory-nextgen-review.html`

---

## 1. Problem

Dev dùng 2+ agent (Claude + Cursor) trên cùng repo phải sửa cùng 1 fact N lần.
Append-only + top-k similarity khiến bản cũ sống mãi: agent sau vẫn trả lời bằng thông tin đã sửa.

## 2. Goal (đo được)

Sửa 1 lần → mọi agent sau trả đúng bản hiệu lực.
- Correct-answer hiện tại ↑ · đúng-lịch-sử không rớt
- Stale-answer rate ↓ · false-supersession ≈ 0
- Write+read cost công khai cả vòng đời · p50/p95 latency có trace theo stage
- Guardrail: leak scope = 0 · residual sau xóa = 0 · receipt verify pass

Không phải goal: recall@k cao hơn Supermemory mọi giá · tuyên bố thắng ai.

## 3. Non-goals (cấm ở V1)

Bảng/DB mới · graph engine/PPR mới · microservice mới · fine-tune · 4-network Hindsight ·
recursive consolidation · self-optimizer · dashboard/settings/graph-viz mới · team sharing ·
SSO/SLA · multimodal · frontend Next.js · LangGraph agents/chat-RAG · connectors ngoài
manual/file_upload/mcp_agent · SaaS growth (referral/experiments/analytics/quota/digest-email/demo-seed).

## 4. Users

- **Chính:** solo dev, 1 máy, 2+ agent qua MCP. Lát cắt: fact cấu hình dự án có hiệu lực theo thời gian.
- **Sau V1:** team nhỏ share memory (workspace + scoped token + ledger đã có).
- **Không phục vụ V1:** enterprise, end-user phổ thông, multimodal.

## 5. UX contract (4 câu)

1. **Nhớ:** nói bình thường → auto-retain. Không lệnh `save`.
2. **Sửa:** "sai rồi, production vẫn Postgres" → supersede có evidence. Mơ hồ → giữ cả hai + hỏi 1 câu.
3. **Hỏi:** "sao nhớ vậy?" → nguồn + thời gian + scope. Không có bằng chứng → nói không có, không bịa.
4. **Quên:** "quên X đi" → erasure receipt, dọn cả view dẫn xuất.
- Badge duy nhất user/agent thấy: `current / superseded / needs-check`. Không lộ half-life/top-k/rerank.

## 6. Architecture — approach A (metadata-only, không migration)

```
[Agent/Connector] → Retain → Resolve → Index → Recall → Derive → Correct/Invalidate/Erase (scope + ledger)
```

### 6.1 Data model (reuse `Memory`, thêm metadata trong `extra_metadata`)

| Key | Kiểu | Ý nghĩa |
|---|---|---|
| `assertion` | `fact\|plan\|inference` | phân biệt sự thật / dự định / suy luận |
| `subject` / `attribute` / `scope` | string | khóa resolve trùng (vd: `proj-x/db/prod`) |
| `valid_from` / `valid_to` | ISO datetime \| null | hiệu lực (khác `captured_at` = biết khi nào) |
| `supersedes` / `superseded_by` | uuid \| null | chain thay thế |
| `evidence_ids` | uuid[] | source memory IDs chứng minh |
| `derived_from` | uuid[] | view dẫn xuất dùng nguồn nào (`parent_id` 1-parent không đủ) |
| `derived_dirty` | bool | true → chặn phục vụ, chờ rebuild |

Rule: `content`/`source_ref` raw bất biến — LLM không ghi đè. Import cũ đến muộn không tự thắng fact mới.
Không migration: mọi key vắng mặt = legacy (xem như `fact`, scope `default`, đang hiệu lực).
`valid_from` vắng mặt = `captured_at`. Thời gian luôn ISO-8601 UTC.

### 6.2 Write path — `resolve` (1 hàm duy nhất ở đường ghi)

- Cùng `(subject, attribute, scope)` + evidence mới hợp lệ → tạo Memory mới, set `supersedes`/`superseded_by`, đánh dirty mọi derived có `derived_from` chứa bản cũ.
- Resolve chạy trong cùng transaction với INSERT bản mới; `superseded_by` của bản cũ + `derived_dirty` set cùng commit (không dirty-then-crash nửa chừng).
- `subject/attribute/scope` chuẩn hóa: trim + lowercase + collapse whitespace; rỗng → `needs-check`, không tự đoán.
- Thiếu evidence / khác scope / dự định-vs-sự thật mơ hồ → **giữ cả hai + trả `needs-check`** (default an toàn).
- Cấm hard-delete qua correction. Xóa pháp lý đi flow `erasure_service` riêng.
- Reuse `write_back.index_new_memory` làm owner indexing duy nhất (embed + graph best-effort).

### 6.3 Read path — `MemoryRetriever` (giữ pipeline, thêm 2 filter)

Mọi filter validity/dirty chạy ở tầng hydrate (SQL/ORM sau khi có candidate IDs),
không nhét vào Chroma metadata — tránh 2 nguồn sự thật lệch nhau.

1. Personal context (giữ) → 2. rewrite (**fast-path:** query rõ → skip LLM, embed trực tiếp) →
3. embed → 4. Chroma search → 5. hydrate PG → 6. **filter: ẩn `superseded_by != null`
(trừ `include_history=true`) + chặn `derived_dirty=true`** → 7. score hiện có → 8. trace.
- Rerank/graph chỉ bật khi lỗi chứng minh cần, có cap token/time. Không biết → rỗng + trace.

### 6.4 MCP contract (6 tool cũ + 1 mới)

| Tool | Thay đổi V1 |
|---|---|
| `search_memory` | thêm `state` mỗi hit, mặc định ẩn superseded |
| `get_memory` | thêm provenance: `supersedes/superseded_by/evidence_ids/valid_from/scope/state` |
| `timeline` | reuse + **fix O(n): 2 query `captured_at </>` + limit window** (`tools.py:212-223`) |
| `correct_memory` (**mới**) | input: `memory_id?, subject, attribute, scope, content, valid_from?, evidence?` → output: bản mới + chain; mơ hồ → `needs-check`. **Ownership: `memory_id` (nếu có) phải thuộc caller, ngược lại `memory not found`. Ledger ghi cả resolve-noop.** |
| `add_memory` | giữ, đi qua resolve trùng. **`add` không nhận `supersedes` từ caller — chain chỉ resolve server tạo.** |
| `list_recent`, `delete_memory`, `forget_memory` | giữ; forget mở rộng tới derived dirty + receipt |

### 6.5 Latency (0 đổi stack)

- Stack V1: **Lite-only** — SQLite + Chroma in-process + task eager + FS uploads (`config.py:8-12`, `Dockerfile.lite`, `install.sh`).
- 3 patch: fast-path skip-rewrite · fix `timeline` load-all · tách `RecallTrace.latency_ms` theo stage (rewrite/embed/search/hydrate).
- Đổi stack (Postgres/pgvector/Redis/service mới) chỉ khi trace chỉ đúng thủ phạm + eval chứng minh.

## 7. Deletion list (tối giản để nhanh)

V1 không build / không ship: `frontend/` · `app/agents/` (15 LangGraph agents) · `api/v1/chat.py` (chat-RAG) ·
connectors ngoài 3 nguồn giữ (`manual`, `file_upload`, `mcp_agent`) · routers SaaS
(referral/experiments/analytics/quota/demo/system_settings/workspaces) · Celery/Redis/worker/beat/flower
khỏi path V1 · full-stack compose khỏi scope next-gen.
Giữ: Memory+metadata · retriever · write_back · erasure+receipt · mcp_hub · ledger+token ·
import ChatGPT/Claude JSON · harness eval.
(Xóa vật lý theo nhát riêng, mỗi nhát chạy `pytest` đường memory/MCP — ngoài scope spec này.)

## 8. Eval acceptance (reuse harness, theo `memory-papers.md §6`)

- Subset LongMemEval knowledge-update + temporal + abstention · FactConsolidation-SH/MH.
- 9 case nội bộ: đổi nơi làm việc · hồi tố hiệu lực · import muộn bản cũ · dự định-vs-sự thật ·
2 người trùng tên · sửa rồi hỏi lịch sử · sửa nguồn nhiều summary dùng · xóa nguồn · revoke trước rebuild.
- Metric §2. Pin dataset/reader/judge/budget · giữ question khỏi write phase · báo per-category + failures.
- Ngưỡng merge tối thiểu: stale ↓ có ý nghĩa trên subset KU + 9/9 case nội bộ pass + false-supersession = 0 + leak = 0 + residual = 0. Chưa đặt % cứng khi chưa có baseline subset.
- `needs-check` là output hợp lệ (đếm riêng, không tính là fail) nhưng tỉ lệ > 30% trên eval = spec fail (resolve quá nhát).
- **Đạt mới merge. Không gọi smoke test là full S. Không công bố thắng đối thủ.**

## 9. Rollout

eval subset + case nội bộ → đạt metric → implement lát cắt → demo Claude + Cursor chung MCP →
merge. Mở rộng (graph bounded, consolidation, team) chỉ bằng ablation riêng sau V1.

## 9b. Self-check (ponytail: 1 check duy nhất, chạy trước merge)

`eval` phải có 1 test chạy được chứng minh resolve đúng root-cause: cùng `(subject, attribute, scope)`
ghi 2 bản → bản cũ bị ẩn ở `search` mặc định, hiện ở `include_history=true`, `get` bản mới
thấy chain; bản khác scope không bị đụng. Fail test này = V1 chưa xong, mọi metric khác vô nghĩa.

## 10. Risks

- Resolve sai scope/thời gian tệ hơn không sửa → default giữ-cả-hai + `needs-check`.
- Agent ngoài đã đọc/copy không thu hồi được → nói rõ trong skill, không hứa.
- Write cost tăng (resolve + dirty + rebuild) → đo cả vòng đời, cấm báo rẻ bằng read latency.

## 11. Quyết định đã chốt (founder)

1. Approach A metadata-only + lát cắt fact cấu hình dự án — YES
2. Thêm đúng 1 tool `correct_memory` — YES
3. V1 Lite-only self-host, full-stack ngoài scope next-gen — YES
4. Viết spec này — YES (done)
