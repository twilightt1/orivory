# Memory thế hệ tiếp theo — bằng chứng từ paper và hướng tối thiểu cho Orivory

**Ngày nghiên cứu:** 2026-09-11 · **Phương pháp:** đọc văn bản sơ cấp trên arXiv, cố định phiên bản `v1`; khám phá thêm qua README của tác giả. Đã xem convention trong `docs/research/CLAUDE_MEM_ANALYSIS.md` và `docs/EVALUATION_GUIDE.md` trước khi viết.

**Phạm vi bằng chứng:** mọi kết quả benchmark dưới đây là **tác giả báo cáo, chưa được tái lập tại Orivory**. “Hạn chế triển khai” và “ý tưởng tối thiểu” là phân tích/đề xuất của tài liệu này, không phải kết luận thực nghiệm của paper. Đây không phải systematic review hay bảng xếp hạng mới nhất tháng 9/2026. Search API hết quota; các paper bổ sung được tìm qua repo tác giả rồi tải trực tiếp HTML arXiv bằng HTTP. Một số bảng của năm paper đầu mất công thức/số trong bản trích xuất; không điền lại bằng suy đoán.

## 1. Kết luận để ra quyết định

1. **Ưu tiên sửa bộ nhớ có bằng chứng và thời gian, không ưu tiên thêm graph.** LongMemEval kiểm tra knowledge update/temporal reasoning; MemoryAgentBench cho thấy conflict resolution vẫn khó. Đây là cơ sở để thử cơ chế thay thế fact có nguồn và vô hiệu hóa context dẫn xuất. Không có bằng chứng rằng tổ hợp này độc nhất thị trường. [P4, P5]
2. **Giữ nguồn gốc; tóm tắt là index/view, không phải sự thật cuối cùng.** A-MEM làm giàu note; SimpleMem chuẩn hóa đại từ và thời gian; HippoRAG 2 khôi phục passage context vào graph. Có thể học phần này mà không cài nguyên framework. MemoryAgentBench cảnh báo mất thông tin khi chỉ giữ fact được trích xuất. [P1–P3, P5, P6]
3. **Đo cả write cost và read cost.** Mem0 báo cáo giảm latency/token ở query so với full context, trong khi MemoryAgentBench đo thấy xây dựng memory đắt trong thiết lập của họ. Hai kết quả không mâu thuẫn: khác workload và ranh giới hạch toán. [P3, P5]
4. **Không dùng headline score để chọn kiến trúc.** A-MEM, Mem0 và SimpleMem khác giao thức LoCoMo; Hindsight dùng baseline công bố với judge khác; MemoryAgentBench có LongMemEval(S*) đã biến đổi, không phải LongMemEval-S nguyên bản. [P1, P3–P7]
5. **Học từ paper mới: sửa chất lượng extraction trước khi làm “self-evolving”.** Ablation EvolveMem báo cáo loại bỏ extraction guards gây hại lớn hơn loại bỏ self-evolution. Bước rẻ là kiểm tra ingest/retry/coverage và đọc failure log, không phải cho agent tự chỉnh production. [P9, §4.5]

## 2. Danh tính paper đã xác minh bằng văn bản sơ cấp

Các link HTML dưới đây chính là phiên bản đã tải và đọc; link abstract dùng cùng hậu tố để tránh citation drift.

| Mã | Paper / tên gọi thường dùng | arXiv đã xác minh | Văn bản đã đọc |
|---|---|---|---|
| P1 | **A-Mem: Agentic Memory for LLM Agents** — A-MEM | [2502.12110v1](https://arxiv.org/abs/2502.12110v1) | [HTML](https://arxiv.org/html/2502.12110v1), §3–4, §6 |
| P2 | **From RAG to Memory: Non-Parametric Continual Learning for Large Language Models** — HippoRAG 2 | [2502.14802v1](https://arxiv.org/abs/2502.14802v1) | [HTML](https://arxiv.org/html/2502.14802v1), §3–6 |
| P3 | **Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory** | [2504.19413v1](https://arxiv.org/abs/2504.19413v1) | [HTML](https://arxiv.org/html/2504.19413v1), abstract, §2–3 |
| P4 | **LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory** | [2410.10813v1](https://arxiv.org/abs/2410.10813v1) | [HTML](https://arxiv.org/html/2410.10813v1), §3–4; tiêu đề HTML có ngắt dòng “Assist-ants” |
| P5 | **Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions** — MemoryAgentBench | [2507.05257v1](https://arxiv.org/abs/2507.05257v1) | [HTML](https://arxiv.org/html/2507.05257v1), §3–5 |
| P6 | **SimpleMem: Efficient Lifelong Memory for LLM Agents** | [2601.02553v1](https://arxiv.org/abs/2601.02553v1) | [HTML](https://arxiv.org/html/2601.02553v1), §2–3 |
| P7 | **Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects** | [2512.12818v1](https://arxiv.org/abs/2512.12818v1) | [HTML](https://arxiv.org/html/2512.12818v1), §3–4, §7 |
| P8 | **OmniMem: Autoresearch-Guided Discovery of Lifelong Multimodal Agent Memory** | [2604.01007v1](https://arxiv.org/abs/2604.01007v1) | [HTML](https://arxiv.org/html/2604.01007v1), abstract và cấu trúc phương pháp; chỉ sàng lọc |
| P9 | **EvolveMem: Self-Evolving Memory Architecture via AutoResearch for LLM Agents** | [2605.13941v1](https://arxiv.org/abs/2605.13941v1) | [HTML](https://arxiv.org/html/2605.13941v1), §3–4, phụ lục D–E |

Repo dùng để tìm P6/P8/P9: [aiming-lab/SimpleMem](https://github.com/aiming-lab/SimpleMem); P7: [vectorize-io/hindsight](https://github.com/vectorize-io/hindsight). README hiện tại có thể chứa tính năng xuất hiện sau paper; không xem README là chứng minh rằng `v1` đã triển khai chúng.

## 3. Năm paper bắt buộc

### P1 — A-MEM: note tự chứa và enrichment có giới hạn

- **Cơ chế:** note gồm nội dung tương tác, timestamp, keywords, tags, contextual description và links. Dùng embedding tìm các note gần nhất, LLM quyết định liên kết, rồi cập nhật context/keywords/tags của các note liên quan. Retrieval ở §3.4 vẫn lấy top-k bằng similarity; không nên diễn giải thành graph traversal bắt buộc ở query time. [P1, §3]
- **Bằng chứng:** thử trên LoCoMo với sáu foundation models; F1/BLEU-1, các metric bổ sung, token length và ablation link generation/memory evolution. Ví dụ bảng §4.2 với GPT-4o-mini báo cáo multi-hop F1 45.85 cho A-Mem và 25.52 cho MemGPT. Đây không phải mức tăng đã tái lập, cũng không phải overall accuracy. [P1, §4]
- **Hạn chế:** tác giả thừa nhận phụ thuộc năng lực LLM và mới xét text. Với Orivory, enrichment có thể phát sinh suy diễn sai và write amplification; update metadata không tự chứng minh khả năng sửa mâu thuẫn, xóa lan truyền hay audit. [P1, §6; phân tích triển khai]
- **Ý tưởng tối thiểu:** tái dùng `summary`/`tags`, giữ `content` và nguồn không bị LLM ghi đè; chỉ enrichment nhóm ứng viên nhỏ do retriever hiện có tìm. Chưa thêm autonomous evolution toàn kho.
- **Đo công bằng:** raw-note hybrid → thêm summary → thêm links → thêm evolution; cố định reader, embeddings và ngân sách context. Đo multi-hop F1 cùng số LLM calls/token khi ghi, tỉ lệ fact suy diễn sai, recall nguồn gốc. Không chỉnh k bằng nhãn loại câu hỏi mà production không có.

### P2 — HippoRAG 2: graph phải giữ passage context

- **Cơ chế:** OpenIE tạo triples; nối synonym giữa phrase nodes; thêm passage nodes và cạnh `contains`. Query được match với cả triples và passages; LLM lọc triples (“recognition memory”), chọn seed rồi chạy Personalized PageRank để xếp hạng passage. Không có triple phù hợp thì fallback dense passage retrieval. [P2, §3.1–3.5]
- **Bằng chứng:** Simple QA (NQ, PopQA), multi-hop (MuSiQue, 2Wiki, HotpotQA, LV-Eval), discourse understanding (NarrativeQA); passage recall@5 và answer F1. Setup chính dùng Llama-3.3-70B-Instruct, NV-Embed-v2 và tối ưu prompt filter bằng DSPy. Abstract báo cáo cải thiện 7% trên associative-memory tasks; không chuyển số này thành cam kết cho Orivory. [P2, abstract, §4]
- **Hạn chế:** corpus QA không tương đương personal-memory cập nhật liên tục. OpenIE, embeddings, graph và online filter có chi phí; entity/triple sai có thể lan truyền trong graph. Những thí nghiệm này không chứng minh tenant isolation, erasure hoặc temporal correction. [P2, §3–4; phân tích phạm vi]
- **Ý tưởng tối thiểu:** trước hết bảo đảm kết quả graph luôn trả về passage/source tương ứng và có hybrid fallback. Chỉ thử query-to-triple hoặc mở rộng lân cận hạn chế nếu error log cho thấy thiếu evidence multi-hop; chưa thêm graph DB hay PPR engine mới.
- **Đo công bằng:** cùng reader/embedding/corpus, so hybrid hiện có với graph hiện có, rồi bật từng thay đổi. Báo cáo recall@5, QA F1, p95 query, indexing cost; một cải thiện multi-hop không được che hồi quy single-hop.

### P3 — Mem0: quyết định ghi rõ ràng, không chỉ append

- **Cơ chế:** extraction đọc message pair mới cộng summary lịch sử và recent messages; update so candidate fact với top-k memory tương tự rồi chọn `ADD`, `UPDATE`, `DELETE`, `NOOP`. Summary làm bất đồng bộ. Biến thể graph dùng entity/relation, đánh dấu relation lỗi thời là invalid thay vì xóa vật lý để hỗ trợ temporal reasoning. [P3, §2.1–2.2]
- **Bằng chứng:** abstract báo cáo 26% cải thiện **tương đối** ở LLM-as-a-Judge so OpenAI memory; p95 latency thấp hơn 91% và tiết kiệm hơn 90% token so full-context. Bản graph chỉ tăng khoảng 2% overall theo cách diễn đạt của tác giả. Comparator của accuracy và efficiency khác nhau. [P3, abstract]
- **Hạn chế:** benchmark loại adversarial/unanswerable questions; không suy ra abstention tốt. Metric token §3.2 là context được retrieve cho QA, không phải tất cả token extraction/update. OpenAI baseline dùng giao diện ChatGPT, không phải API so sánh đồng nhất. Không lấy headline này làm bằng chứng giảm tổng chi phí vận hành. [P3, §3.1–3.3]
- **Ý tưởng tối thiểu:** tận dụng dedup nguồn hiện có; thêm quyết định “giữ nguyên / thêm / thay thế có bằng chứng” chỉ cho nhóm cùng chủ thể/thuộc tính. Correction phải lưu nguồn và quan hệ supersession, không cho LLM tùy ý hard-delete. Xóa theo quyền riêng tư là flow riêng.
- **Đo công bằng:** LoCoMo cùng subset/judge/prompts; bổ sung LongMemEval knowledge-update và abstention, FactConsolidation. Đo false merge, false deletion, stale-answer rate và write+read cost; tách graph khỏi base bằng ablation.

### P4 — LongMemEval: benchmark ưu tiên cho personal memory

- **Cơ chế benchmark:** 500 câu hỏi; năm năng lực: information extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention. Histories có timestamp; có evidence annotation. `S` khoảng 115k tokens/câu, `M` khoảng 1.5 triệu tokens với 500 sessions. Năm năng lực không đồng nghĩa chỉ năm nhãn câu hỏi. [P4, §3.1–3.3]
- **Bằng chứng thiết kế:** chia nhỏ session, fact-augmented index keys và time-aware query expansion là các cải tiến được paper đề xuất. QA được chấm bằng LLM judge; recall@k/NDCG@k giúp tách lỗi retrieval khỏi reader. [P4, abstract, §3.4, §4]
- **Hạn chế:** history tổng hợp và các fact/evidence được xây có kiểm soát; không đại diện đầy đủ dữ liệu coding thực hay governance. So sánh commercial ở §3.5 dùng 97 câu với history rút gọn, thực hiện năm 2024; không phải review ChatGPT hiện tại. [P4, §3]
- **Ý tưởng tối thiểu:** dùng loader/runner hiện có; giữ session timestamps và source/evidence ID xuyên ingest→retrieval→answer. Dùng summary/facts làm search key nhưng luôn có đường đọc raw evidence. Không suy valid-time trực tiếp từ ingestion order.
- **Đo công bằng:** pin dataset hash, split, question IDs, reader, judge version/prompt, token budget. Báo cáo theo từng năng lực và abstention; oracle-evidence baseline giúp xác định lỗi do memory hay do reader. Không gọi subset smoke test là full LongMemEval-S.

### P5 — MemoryAgentBench: kiểm tra nhớ, học và giải quyết mâu thuẫn riêng

- **Cơ chế benchmark:** chia input thành chunks và nạp tăng dần qua nhiều lượt. Bốn nhóm: Accurate Retrieval, Test-Time Learning, Long-Range Understanding, Conflict Resolution. EventQA kiểm tra chuỗi sự kiện; FactConsolidation dùng counterfactual edits và yêu cầu ưu tiên thông tin đến sau theo giao thức benchmark. [P5, §3]
- **Bằng chứng:** RAG thường mạnh ở retrieval; long-context mạnh hơn ở TTL/LRU trong thiết lập tác giả. FactConsolidation multi-hop rất khó. Bảng 2 ghi Contriever 7.0 trong cột này trong khi phần văn xuôi nói tối đa 6%; tài liệu này không lặp lại con số tổng quát thiếu nhất quán ấy. [P5, §4.2, Table 2]
- **Hạn chế:** tác giả thừa nhận dữ liệu phần lớn synthetic. Chunk size không đồng nhất cho mọi phương pháp (Mem0/Cognee dùng 4096 vì cost), nên đây không phải phép đo cô lập chất lượng từng thuật toán. `LongMemEval(S*)` gồm history được reformulate thành năm dialogue/300 câu, không thể so trực tiếp với 500 câu LongMemEval-S gốc. [P5, §3.2, §4.1, §5]
- **Ý tưởng tối thiểu:** chạy FactConsolidation-SH/MH cùng một tập retrieval và một tập LRU nhỏ; giữ cả evidence chi tiết và summary để tránh tối ưu fact recall làm hỏng tổng hợp. “Later wins” chỉ dùng khi đề bài/nguồn xác định rõ; không biến thành chính sách tin mọi message mới hơn.
- **Đo công bằng:** metric nguyên bản cho từng task, không gộp F1/classification/recommendation/summarization thành “accuracy” chung. Nạp tuần tự, không rò câu hỏi/đáp án vào consolidation; log memory-construction và query-execution riêng. [P5, §3–4]

## 4. Paper bổ sung: đáng học gì, bỏ gì

### P6 — SimpleMem: chuẩn hóa đầu vào trước khi tối ưu retrieval

**Cơ chế:** lọc window theo novelty/information score; biến nội dung thành memory units tự chứa, giải đại từ và neo biểu thức thời gian; index semantic + lexical + metadata; recursive consolidation bất đồng bộ; retrieval scope theo độ phức tạp query. [P6, §2]

**Bằng chứng:** trên setup GPT-4.1-mini/LoCoMo, tác giả báo cáo average F1 43.24 so Mem0 34.20; ablation bỏ atomization làm temporal F1 từ 58.62 xuống 25.40. Đây là bằng chứng định hướng cho chuẩn hóa ngữ cảnh, không chứng minh từng thao tác riêng lẻ tạo toàn bộ mức tăng. [P6, §3.2–3.4]

**Hạn chế:** “semantic lossless compression” là tên/mô tả phương pháp, không phải bảo đảm không mất dữ liệu; gating thực sự loại window khỏi memory construction (§2.1). Con số giảm token inference không thay thế accounting cả vòng đời; số giây “Retrieve Time” trong Table 3 không phải p95 của một query.

**Tối thiểu + đánh giá:** thêm subject/date/source rõ ràng vào summary hiện có; không đoán ngày nếu thiếu timezone hoặc reference date. Giữ raw evidence theo retention policy. A/B temporal accuracy và extraction coverage dưới cùng ngân sách token trước khi thử recursive consolidation.

### P7 — Hindsight: evidence khác inference; summary phải cập nhật theo fact

**Cơ chế:** bốn logical networks phân biệt world facts, experiences, observations và opinions; `retain / recall / reflect`. Recall kết hợp semantic, keyword, graph, temporal qua RRF/rerank và token budget. Observation là summary entity được tái tạo bất đồng bộ khi fact nền thay đổi. [P7, §3, §4.1.5–4.2]

**Bằng chứng và caveat quan trọng:** Table 3 báo cáo LongMemEval-S accuracy 83.6% (OSS-20B), 89.0% (OSS-120B), 91.4% (Gemini-3 answer generator). Nhưng §7.3 dùng baseline Supermemory công bố với GPT-4o judge, còn Hindsight dùng GPT-OSS-120B judge; baseline LoCoMo cũng lấy từ báo cáo bên khác. Cùng section còn để placeholder `<add>` ở token budget. Vì vậy không dùng bảng này để khẳng định thắng Supermemory ở điều kiện tương đương. [P7, §7.3–7.5]

**Tối thiểu + đánh giá:** học phân biệt nguồn với suy diễn và đánh dấu summary cần rebuild khi evidence đổi. Không cần bốn network hay personality/opinion engine. Đo stale-context rate ngay sau correction và sau khi background job hoàn tất, thêm chi phí rebuild. P7 là tiền lệ rõ cho cập nhật derived observations; không tuyên bố Orivory phát minh ý tưởng này.

### P8 — OmniMem: ghi nhận, chưa đưa vào phạm vi xây

**Cơ chế sàng lọc:** autonomous research tìm cấu hình memory multimodal qua thử nghiệm; phương pháp gồm selective ingestion, progressive hybrid retrieval và graph augmentation. Abstract báo cáo cải thiện so **baseline khởi đầu của chính hệ thống**, không phải phần trăm thắng hệ thống tốt nhất. [P8, abstract, §3]

**Hạn chế/tối thiểu:** nghiên cứu này mới được sàng lọc, chưa kiểm toán protocol/split. Multimodal và autonomous architecture search chưa có nhu cầu được xác lập cho Orivory. Chỉ lấy bài học kiểm tra data-pipeline bugs trước khi thêm thuật toán; nếu sau này thử phải tách development/test và dùng đúng LoCoMo/Mem-Gallery evaluator. Chưa dùng headline SOTA làm tiêu chí lựa chọn.

### P9 — EvolveMem: failure logs có giá trị hơn nhãn “self-evolving”

**Cơ chế:** đưa retrieval configuration thành action space; evaluate→diagnose→propose→guard; LLM đọc lỗi từng câu rồi đề xuất thay đổi, rollback nếu regression. Dùng SQLite/FTS5 và multi-view retrieval. [P9, §3–4]

**Bằng chứng:** §4.5 báo cáo bỏ extraction quality control giảm 23.22 F1 points; bỏ self-evolution giảm 2.03 points trong ablation của tác giả. Không coi đó là chứng minh nhân quả phổ quát, nhưng đủ để ưu tiên extraction correctness hơn một vòng optimizer mới.

**Hạn chế:** MemBench chỉ 28 samples trong setup này; LoCoMo evaluation và evolution dùng benchmark feedback nên cần kiểm toán ranh giới tuning/held-out. Phụ lục E nói có hai LoCoMo samples validation; chưa đủ để tự suy ra mọi score là test chưa từng dùng. Không đồng nhất MemBench với MemoryAgentBench. Chi phí optimizer/answer verification phải tính riêng. [P9, §4.1, §4.4, Appendix D–E]

**Tối thiểu + đánh giá:** lưu config, retrieved IDs, failure reason và per-question result trong eval hiện có; người chọn một thay đổi nhỏ rồi chạy lại held-out. Chưa cho LLM tự chỉnh production hay tạo thêm cấu hình tùy ý. Đo accuracy/cost Pareto với reader cố định và tính cả chi phí tìm cấu hình.

## 5. Áp dụng vào Orivory mà không dựng lại cái đã có

**Kiểm tra tại repo:** [`Memory`](../../app/models/memory.py) đã có `content`, `summary`, `tags`, `parent_id`, `source_ref`, `captured_at`, `indexed_at`, `extra_metadata`; [`import_service.py`](../../app/services/import_service.py) đã dedup theo nguồn. `captured_at` được model mô tả là original event time; `indexed_at` là lần đầu lưu. Hai field này **chưa tự biểu diễn đầy đủ khoảng hiệu lực của một fact**.

Theo bối cảnh tích hợp đã được xác minh trong luồng nghiên cứu chính, Orivory đã có search snippets→get, pinned/recent context cap và một owner indexing ở `write_back.py`; không đề xuất thêm progressive disclosure, context manager hay write pipeline thứ hai. Tài liệu này không sửa code và không audit toàn bộ các flow đó.

### Giả thuyết sản phẩm cần chứng minh

> Khi người dùng sửa một thông tin, mọi lần recall sau đó phải chỉ ra nguồn của thông tin có hiệu lực, không phát lại summary/context cũ, và vẫn giải thích được lịch sử thay đổi trong quyền truy cập cho phép.

Một lát cắt thử nghiệm đủ nhỏ:

1. **Evidence-backed correction:** lưu source memory ID và vị trí bằng chứng; phân biệt phát biểu chắc chắn, dự định, suy luận. Chỉ đánh dấu superseded khi có bằng chứng phù hợp cùng scope/chủ thể/thuộc tính; trường hợp mơ hồ giữ cả hai và yêu cầu làm rõ.
2. **Temporal semantics:** tách thời điểm biết thông tin và thời điểm thông tin có hiệu lực. Dùng metadata sẵn có để thử một kiểu correction trước; chưa xây framework bitemporal toàn cục. Import trễ một bản cũ không được tự thắng fact mới.
3. **Derived-context invalidation:** theo dõi các source IDs của summary/context dẫn xuất và đánh dấu dirty khi nguồn sửa/xóa. Chặn phục vụ view đã biết là stale; rebuild qua owner indexing hiện có. `parent_id` chỉ mô tả một parent, không thay thế được dependency nhiều nguồn.
4. **Bounded retrieval:** raw evidence là fallback; graph expansion/LLM reranking chỉ bật khi đo thấy cần, với cap token/time rõ ràng. Không thêm planner tự trị trên đường nóng.
5. **Governance không được giản lược:** provenance, candidate retrieval, dependency rebuild và cache invalidation đều phải theo scope. Correction không phải legal erasure. Khi xóa theo quyền riêng tư, xóa cả nội dung dẫn xuất liên quan và xác minh receipt; không giữ bản cũ dưới danh nghĩa “audit”.

Đây là **đề xuất**, không phải mô tả tính năng đã có hay claim vượt đối thủ. Tài liệu phân tích Claude-Mem cũ là nguồn convention, không phải nguồn xác nhận khoảng trống hiện tại. Không kế thừa các khẳng định “Supermemory local là closed binary” hoặc “không có governance”: chúng không được chứng minh. Thông tin self-host phải đối chiếu [docs Supermemory](https://supermemory.ai/docs/self-hosting/overview); kiến trúc Claude-Mem phải đối chiếu [repo đúng](https://github.com/thedotmack/claude-mem) và [overview](https://docs.claude-mem.ai/architecture/overview).

## 6. Giao thức đánh giá tối thiểu, có thể so sánh

| Câu hỏi cần trả lời | Bộ đo | Baseline / ablation | Phải log |
|---|---|---|---|
| Chuẩn hóa summary có giúp không? | LongMemEval-S temporal + extraction, sau đó full S | Raw hybrid hiện tại / thêm normalized summary | Evidence recall, QA judge, write token, query token |
| Sửa fact có lan đến câu trả lời không? | LongMemEval knowledge-update + FactConsolidation-SH/MH | Append-only / supersession / supersession+invalidation | Current-answer accuracy, historical accuracy, stale-context rate, false supersession |
| Nén có làm mất khả năng tổng hợp không? | MAB retrieval + TTL + LRU | Facts-only / facts+source fallback / thêm summary | Metric riêng từng task, coverage, context tokens |
| Graph có đáng chi phí không? | MuSiQue/2Wiki hoặc lỗi multi-hop nội bộ đã gắn evidence | Hybrid / graph hiện tại / bounded expansion | Recall@5, F1, p95, build cost |
| Flow có an toàn không? | Cases nội bộ cross-scope, correction→delete, cache stale | Trước/sau từng bước | Unauthorized exposure, residual derived content, receipt verification |

**Các case nội bộ nên có:** đổi nơi làm việc; hồi tố ngày hiệu lực; import muộn nguồn cũ; dự định chưa thành sự thật; hai người trùng tên; sửa rồi hỏi lịch sử; sửa nguồn được nhiều summary dùng; xóa nguồn; token bị revoke trước rebuild. Governance suite là kiểm thử bổ sung của Orivory, không được gắn nhãn là benchmark học thuật.

**Quy tắc so sánh:** pin dataset hash/split/IDs và commit, ingest order, source timestamp, embedding/reader/judge model, prompt, temperature, context budget, chunk size và k. Giữ question/reference khỏi write phase. Reset memory giữa các instance độc lập, nhưng không reset giữa chunks của cùng sequence. Báo cáo per-category, số lỗi/timeout, số câu thực sự chấm; không âm thầm loại failures. Chạy nhiều lần hoặc paired bootstrap khi kết luận từ chênh lệch nhỏ.

**Cost accounting:** tách extraction, embedding, consolidation/index rebuild, retrieval/rerank và answer generation; báo cáo p50/p95, token/call count, storage và cost theo workload với tỉ lệ reads:writes công bố. Không gọi token context là tổng token; không gọi giảm read latency là giảm tổng chi phí.

**Tận dụng scaffold đúng mức:** [`eval/run_benchmark.py`](../../eval/run_benchmark.py) thực sự có `plan` và `score`; `ingest/query` ở CLI này trả lỗi live-wiring, không chạy benchmark thật. Repo cũng có [`eval/run_system_benchmark.py`](../../eval/run_system_benchmark.py), nhưng tài liệu này chưa chạy/kiểm toán end-to-end runner đó. Cần kiểm tra đường chạy thực trước khi công bố kết quả. Hướng dẫn dùng chung: [`EVALUATION_GUIDE.md`](../EVALUATION_GUIDE.md).

**Điều kiện nâng độ phức tạp:** chỉ thêm consolidation, graph expansion hoặc optimizer khi một ablation giữ nguyên điều kiện cho thấy cải thiện tập lỗi mục tiêu mà không gây hồi quy nghiêm trọng về accuracy, stale context, cost hoặc quyền truy cập. Chưa cần database mới, memory microservice mới, agent tự trị hay fine-tuning chỉ để chứng minh lát cắt đầu tiên.
