# Orivory open-source positioning (quyết định)

**Ngày:** 2026-09-10 · **Trạng thái:** đề xuất chờ founder chốt (§5 câu hỏi mở)
**Nền:** [open-memory-hub.md](open-memory-hub.md) (quyết định D1, 02/09) +
toàn bộ [../research/](../research/) · **Bằng chứng kỹ thuật mới:**
chuỗi benchmark PR #13–#20 + đợt review/remediation 10/09 (xem CHANGELOG).

## 1. Định vị một câu

> **Orivory = memory hub tự chủ cho mọi AI agent — có phân quyền, có receipts, có benchmark.**

Ba trụ, mỗi trụ có bằng chứng kèm theo (không claim suông):

| Trụ | Bằng chứng trong repo |
|---|---|
| **Mọi agent đọc/ghi** | MCP server `/mcp` (7 tools), skill ClawHub `skills/orivory/`, one-command `install.sh` |
| **Có phân quyền + receipts** | Per-token scopes, append-only ledger, erasure receipts có adversarial verification (đã qua review độc lập) |
| **Có benchmark** | LongMemEval-S n=100 (0.570, CI tách biệt), negative results công khai, judge prompt versioned |

## 2. Ma trận khác biệt (mỗi đối thủ một câu)

- **Mem0 core** (61–65K★): chuyển sang dev-infra cho teams; bỏ hub cá nhân (OpenMemory sunset) — niche trống.
- **Zep/Graphiti** (~30K★): KG mạnh nhất nhưng là SDK/framework; Zep CE deprecated; không governance user-facing.
- **Cognee** (~30K★): pipeline ingestion tốt, nhưng SDK/API không app, không proactive, governance dev-facing.
- **Letta**: stateful-agent platform, memory là blocks cho agent — không phải memory home cho user.
- **memU** (~14K★): session-log coding agents, scope hẹp, license mập mờ ("Other").
- **claude-mem** (~93K★): chứng minh demand trần cao, nhưng chỉ là session-memory — không curation, KG, governance.
- **ChatGPT/Claude memory**: walled garden — không export, không audit xuyên app.
- **LangMem**: thư viện, không sản phẩm.

**Khoảng trống duy nhất còn lại:** life-scope ingestion + curation UI + KG + proactive + per-agent permissions/audit + open license. Không ai gom đủ sáu.

## 3. ICP theo thứ tự (từ ICP_AND_TIMING.md, giữ nguyên)

1. **OpenClaw users** — mật độ + reachability cao nhất; skill là đơn vị adoption.
2. **Coding-agent power users** — reach tốt nhưng đông đúc nhất (~20 backend giành cùng posts).
3. **Privacy knowledge workers** — lối phụ (import Rewind/Limitless/OpenRecall).

## 4. Narrative chuẩn (áp cho README, landing, Show HN)

Thứ tự kể — hub trước, app sau (hiện README đang kể ngược):

1. **30s pitch**: câu định vị + ai dùng (mọi agent) + khác gì (receipts).
2. **Cài trong 1 lệnh** → config agent copy-paste → ảnh ledger ("AI nào đã đọc gì").
3. **Số benchmark** + link results (kể cả negative — đó là trust asset).
4. **Kiến trúc** (bảng ngắn) → **Docs/Contributing**.

Quy tắc: claim nào không có link thì xóa.

## 5. Câu hỏi mở cần founder chốt

1. **Frontend 17 routes giữ hay cắt?** D1 nói app chỉ là "client đầu tiên" — memories/chat/dashboard đủ demo; analytics/workspaces/referral là gánh solo-maintainer.
2. **MIT thuần hay chừa đường enterprise?** Ảnh hưởng cách kể governance story (phòng thủ vs compliance).
3. **Release v1.1.0?** Từ v1.0.0 đã có ~20 PR + fix critical (full-stack Postgres deploy là gãy trước migration `a9b8c7d6e5f4`). Không release = user cài bản gãy.
4. **Skill publish ClawHub** — bước founder-side tồn từ 04/09, là cửa ngõ kênh A1.

## 6. Dọn repo đi kèm (Pha 0, đang làm)

- [x] Xóa 12 branch merged · untrack `eval/experiments.db` · move diagram HTML vào `docs/architecture/`
- [x] CHANGELOG + ROADMAP cập nhật tới 10/09
- [ ] Gom 8 script `run_*.py` về 2 entrypoint (cần LLM keys để verify — hoãn, tạm document)
- [ ] Quyết frontend scope (chờ §5.1)
