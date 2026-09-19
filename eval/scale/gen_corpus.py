#!/usr/bin/env python
"""Deterministic scale-workload generator: a 10K-memory corpus, in two parts.

What it produces
----------------
One JSONL file (one memory per line, plain data, **no model calls**, no
embeddings) plus a manifest with the sha256 digests, the per-part counts and
the seed. Two parts, counted SEPARATELY because they are not the same claim:

- ``real``      — user turns sampled from LongMemEval-S (``haystack_sessions``,
  role ``user``), the real human-conversational-text part. If the dataset file
  is absent this fails loudly: the generator never fabricates "real" text;
- ``synthetic`` — template-built capacity fill (fixed seed), varied lengths,
  Vietnamese + English. It is what makes the corpus reach the target size, and
  it is a GENERATED corpus like every other scale benchmark in this repo.

Determinism is the contract: the same ``(seed, rows, dataset)`` must produce
byte-identical corpus and manifest bytes — the sha256 in the manifest is what
the 10K artifact pins, so a re-run that does not reproduce it is not a re-run.

Run: ``python eval/scale/gen_corpus.py --rows 10000 --out /tmp/corpus.jsonl``
(``--self-check`` regenerates into a temp dir and compares digests).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 20260919
ROWS = 10_000
REAL_SHARE = 0.6
# Turns outside this window are dropped (recorded, never silently): a 76K-char
# turn is not a memory-sized document, and sub-20-char turns carry no text.
MIN_TURN_CHARS = 20
MAX_TURN_CHARS = 4000
DEFAULT_DATASET = ROOT / "eval" / "benchmarks" / "data" / "longmemeval_s_cleaned.json"
DATASET_HINT = (
    "download it per eval/benchmarks/README.md (LongMemEval-S, ~264MB) — the "
    "generator never fabricates real text"
)


class CorpusError(RuntimeError):
    """A missing/invalid dataset, or a corpus that cannot be written."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_id(seed: int, part: str, index: int) -> str:
    """Deterministic memory id — stable across runs, unique per corpus row."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"orivory/scale/{seed}/{part}/{index:05d}"))


# ── part (i): real user turns from LongMemEval-S ────────────────────────────


def real_pool(dataset: Path) -> tuple[list[dict], dict]:
    """Every usable user turn, in file order, plus the filter statistics."""
    dataset = Path(dataset)
    if not dataset.is_file():
        raise CorpusError(
            f"real-text dataset not found: {dataset} — expected "
            f"{DEFAULT_DATASET.name}; {DATASET_HINT}"
        )
    try:
        instances = json.loads(dataset.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CorpusError(f"real-text dataset unreadable: {dataset} ({exc})") from exc
    if not isinstance(instances, list) or not instances:
        raise CorpusError(f"real-text dataset is not a non-empty JSON array: {dataset}")

    pool: list[dict] = []
    dropped = 0
    for instance in instances:
        question_id = instance.get("question_id")
        for session_index, session in enumerate(instance.get("haystack_sessions") or []):
            for turn_index, turn in enumerate(session or []):
                if (turn or {}).get("role") != "user":
                    continue
                text = (turn.get("content") or "").strip()
                if MIN_TURN_CHARS <= len(text) <= MAX_TURN_CHARS:
                    pool.append({
                        "question_id": str(question_id),
                        "session_index": session_index,
                        "turn_index": turn_index,
                        "text": text,
                    })
                else:
                    dropped += 1
    if not pool:
        raise CorpusError(
            f"real-text dataset yielded no usable user turns (filters "
            f"{MIN_TURN_CHARS}-{MAX_TURN_CHARS} chars): {dataset}"
        )
    return pool, {
        "instances": len(instances),
        "pool_user_turns": len(pool),
        "dropped_filtered": dropped,
        "filter_min_chars": MIN_TURN_CHARS,
        "filter_max_chars": MAX_TURN_CHARS,
    }


def _real_rows(dataset: Path, seed: int, target: int) -> tuple[list[dict], dict]:
    """Sample ``target`` turns off the file-ordered pool, seeded and stable.

    The sample is drawn with ``random.Random(seed).sample`` over the pool and
    then put back in file order, so the corpus row order does not depend on the
    RNG's call pattern. ``requested_real`` vs the selected count is recorded:
    a small pool cannot fill a large share, and that is not a silent change.
    """
    pool, stats = real_pool(dataset)
    selected = min(max(target, 0), len(pool))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(pool)), selected)) if selected else []
    rows = [
        {
            "memory_id": _row_id(seed, "real", index),
            "part": "real",
            "lang": "en",  # LongMemEval-S is English
            "text": pool[i]["text"],
            "source": {
                "dataset": Path(dataset).name,
                "question_id": pool[i]["question_id"],
                "session_index": pool[i]["session_index"],
                "turn_index": pool[i]["turn_index"],
            },
        }
        for index, i in enumerate(indices)
    ]
    stats["requested_real"] = target
    return rows, stats


# ── part (ii): synthetic capacity fill (fixed seed, VI + EN) ────────────────

_VI_TOPICS = (
    "hợp đồng thuê nhà", "lịch tiêm phòng cho mèo", "danh sách mua sắm tuần này",
    "khoá học tiếng anh buổi tối", "chuyến đi đà lạt tháng sau", "sổ tiết kiệm ngân hàng",
    "hoá đơn điện tháng chín", "lịch bảo dưỡng xe máy", "kế hoạch nghỉ hè của nhóm",
    "tài liệu ôn thi chứng chỉ", "hồ sơ bảo hiểm sức khoẻ", "đơn xin nghỉ phép năm",
    "lịch thanh toán thẻ tín dụng", "phiếu khám sức khoẻ định kỳ", "danh sách khách mời",
    "kế hoạch tập luyện buổi sáng", "ngân sách gia đình tháng này", "thư viện sách cũ",
    "kho ảnh gia đình lưu trữ", "buổi họp phụ huynh cuối kỳ", "kế hoạch tiết kiệm mua nhà",
    "lịch cắt tóc của cả nhà", "bản nháp báo cáo công việc", "lịch gặp bác sĩ tim mạch",
)
_EN_TOPICS = (
    "the lease renewal", "the cat's vaccination schedule", "this week's grocery list",
    "the evening English class", "next month's trip to Da Lat", "the savings account",
    "the September electricity bill", "the motorbike service", "the team's summer leave",
    "the certification study notes", "the health insurance file", "the annual leave request",
    "the credit card payment", "the yearly health check", "the guest list",
    "the morning workout plan", "the household budget", "the second-hand book shelf",
    "the family photo archive", "the end-of-term parents meeting", "the house deposit plan",
    "the family haircut rota", "the work report draft", "the cardiology appointment",
)
_VI_SENTENCES = (
    "Nhắc lại: {topic} phải xong trước {date}, cả nhóm thống nhất dời lại thêm một tuần.",
    "Đã kiểm tra {topic} với {doc} và ghi lại kết luận, chưa cần đổi gì thêm.",
    "Buổi họp hôm nay chỉ bàn về {topic}: giữ nguyên cách làm cũ cho tới khi có số mới.",
    "Tôi gửi {doc} cho nhóm rồi, phần {topic} để tuần sau xử lý tiếp.",
    "{topic} đã được xác nhận qua {doc}; mã theo dõi là {code}.",
    "Chị Lan nói {topic} không gấp, nhưng {date} là hạn cuối của đơn vị.",
    "Ghi chú nhanh: {topic} cần thêm {doc} trước khi trình ký.",
    "Số {code} là mã hồ sơ cho {topic}, lưu vào thư mục sao lưu của nhóm.",
)
_EN_SENTENCES = (
    "Reminder: {topic} is due before {date}; the group agreed to push it one more week.",
    "Checked {topic} against {doc} and wrote the conclusion down — nothing else to change yet.",
    "Today's meeting was only about {topic}: keep the current approach until new numbers land.",
    "Sent {doc} to the group; the {topic} part waits until next week.",
    "{topic} is confirmed through {doc}; the tracking code is {code}.",
    "Lan said {topic} is not urgent, but {date} is the department's deadline.",
    "Quick note: {topic} still needs {doc} before it can be signed off.",
    "Reference {code} is the case number for {topic}, filed in the group's backup folder.",
)
_VI_LONG = (
    "Ghi chú cuộc họp: nhóm dành phần lớn thời gian cho {topic}. "
    "Các việc đã chốt gồm: cập nhật {doc}, đối chiếu số liệu với tuần trước, "
    "và chuẩn bị bản tóm tắt cho buổi sau. Chưa ai nhận phần khó nhất là {topic_extra}. "
    "Mã tham chiếu {code}, hạn nội bộ là {date}.",
    "Nhật ký công việc: buổi sáng xử lý {topic}, buổi chiều rà lại {doc} cùng chị Lan. "
    "Kết luận tạm thời là giữ nguyên cấu hình hiện tại, đo lại sau {date}. "
    "Nếu số không đổi thì chuyển {topic_extra} sang quý sau. Mã {code}.",
)
_EN_LONG = (
    "Meeting notes: most of the session went to {topic}. Agreed items: refresh {doc}, "
    "reconcile the numbers with last week, and draft a summary for the next meeting. "
    "Nobody picked up the hardest part, which is {topic_extra}. "
    "Reference {code}, internal deadline {date}.",
    "Work log: the morning went to {topic} and the afternoon to re-checking {doc} with Lan. "
    "Working conclusion is to keep the current configuration and measure again after {date}. "
    "If nothing moves, {topic_extra} goes to next quarter. Code {code}.",
)
_MONTHS = ("tháng một", "tháng ba", "tháng sáu", "tháng chín", "tháng mười hai")
_DAYS = ("03", "07", "12", "17", "21", "26", "30")


def _synthetic_row(seed: int, index: int, rng: random.Random) -> dict:
    vi = rng.random() < 0.5
    topic = rng.choice(_VI_TOPICS if vi else _EN_TOPICS)
    topic_extra = rng.choice(_VI_TOPICS if vi else _EN_TOPICS)
    doc = ("bản nháp " if vi else "draft ") + topic
    date = (
        f"{rng.choice(_DAYS)}/{rng.choice(('03', '06', '09', '12'))}/2026"
        if vi else
        f"{rng.choice(_MONTHS_OF_YEAR)} {rng.choice(_DAYS)}, 2026"
    )
    code = f"{(rng.choice(('HD', 'LICH', 'REF', 'NOTE', 'HS')))}-{rng.randrange(1000, 9999)}"
    sentences = _VI_SENTENCES if vi else _EN_SENTENCES
    long_sentences = _VI_LONG if vi else _EN_LONG
    # Length buckets: 1 sentence (~90-190 chars) up to 12 (~900-1800 chars).
    count = rng.choice((1, 1, 1, 2, 3, 4, 5, 6, 8, 10, 12))
    parts: list[str] = []
    for _ in range(count):
        template = rng.choice(long_sentences if rng.random() < 0.25 else sentences)
        parts.append(template.format(topic=topic, topic_extra=topic_extra, doc=doc,
                                     date=date, code=code))
    return {
        "memory_id": _row_id(seed, "synthetic", index),
        "part": "synthetic",
        "lang": "vi" if vi else "en",
        "text": " ".join(parts),
        "source": {"generator": "templates.v1", "template_index": index},
    }


_MONTHS_OF_YEAR = (
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
)


def _synthetic_rows(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed ^ 0x5CA1E)  # a stream of its own, never the real one
    rows = [_synthetic_row(seed, index, rng) for index in range(count)]
    # Guarantee the mix the artifact claims: VI and EN both present whenever
    # there is room for them (a 1-row corpus cannot carry both).
    if count >= 2:
        langs = {row["lang"] for row in rows}
        for missing, index in (("vi", 0), ("en", 1)):
            if missing not in langs:
                row = _synthetic_row(seed, count + index, random.Random(seed + index))
                row["lang"], row["memory_id"] = missing, _row_id(seed, "synthetic", count + index)
                rows[index] = row
    return rows


# ── the artifact pair ───────────────────────────────────────────────────────


def generate(*, rows: int = ROWS, seed: int = SEED, dataset: Path = DEFAULT_DATASET,
             out: Path = Path("corpus.jsonl"), real_share: float = REAL_SHARE) -> dict:
    """Write ``out`` (JSONL) + ``<out>.manifest.json``; return the manifest.

    Raises :class:`CorpusError` (writing nothing) when the real dataset is
    missing — a corpus whose "real" half could not be sourced is not written.
    """
    if rows <= 0:
        raise CorpusError(f"rows must be positive, got {rows}")
    if not 0.0 <= real_share <= 1.0:
        raise CorpusError(f"real_share must be within [0, 1], got {real_share}")

    real_target = round(rows * real_share)
    real_rows, stats = _real_rows(Path(dataset), seed, real_target)
    synthetic_rows = _synthetic_rows(rows - len(real_rows), seed)
    corpus = real_rows + synthetic_rows
    if len(corpus) != rows:
        raise CorpusError(f"generated {len(corpus)} rows for a {rows}-row request")

    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in corpus
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(payload)

    synthetic_langs: dict[str, int] = {}
    for row in synthetic_rows:
        synthetic_langs[row["lang"]] = synthetic_langs.get(row["lang"], 0) + 1
    manifest = {
        "corpus_file": out.name,  # a name, not a path: two runs in different dirs must be byte-identical
        "corpus_sha256": digest,
        "rows": rows,
        "seed": seed,
        "real_share": real_share,
        "dataset": {
            "file": Path(dataset).name,
            "sha256": sha256_file(dataset),
            "instances": stats["instances"],
            "pool_user_turns": stats["pool_user_turns"],
            "dropped_filtered": stats["dropped_filtered"],
            "filter_chars": [MIN_TURN_CHARS, MAX_TURN_CHARS],
            "requested_real": stats["requested_real"],
        },
        "parts": {
            "real": {
                "count": len(real_rows),
                "language": "en",
                "source": "LongMemEval-S user turns (haystack_sessions)",
            },
            "synthetic": {
                "count": len(synthetic_rows),
                "languages": dict(sorted(synthetic_langs.items())),
                "source": "templates.v1 (fixed seed)",
            },
        },
        "row_schema": {
            "memory_id": "uuid5(NAMESPACE_URL, 'orivory/scale/<seed>/<part>/<index>')",
            "part": "real | synthetic",
            "lang": "en | vi",
            "text": "memory content, plain data (no embeddings)",
            "source": "real: dataset/question/session/turn | synthetic: generator tag",
        },
    }
    manifest_path = out.parent / (out.name + ".manifest.json")
    manifest_path.write_bytes(
        (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    )
    return manifest


def load_corpus(path: Path) -> tuple[list[dict], dict]:
    """Read a corpus + its manifest back, verifying the digest."""
    path = Path(path)
    manifest_path = path.parent / (path.name + ".manifest.json")
    if not manifest_path.is_file():
        raise CorpusError(f"corpus manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != manifest.get("corpus_sha256"):
        raise CorpusError(
            f"corpus digest mismatch for {path}: {actual} != {manifest.get('corpus_sha256')}"
        )
    # Split on "\\n" ONLY: ``str.splitlines`` also breaks on U+2028/U+2029/NEL,
    # which json.dumps does not escape — real turns contain U+2028 (7 of them in
    # the LongMemEval-S sample), and a splitline there shreds a valid JSONL row.
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]
    return rows, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="deterministic scale-corpus generator")
    parser.add_argument("--rows", type=int, default=ROWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--out", default="corpus.jsonl")
    parser.add_argument("--real-share", type=float, default=REAL_SHARE)
    parser.add_argument("--self-check", action="store_true",
                        help="regenerate into a temp dir and compare digests")
    args = parser.parse_args(argv)

    manifest = generate(rows=args.rows, seed=args.seed, dataset=Path(args.dataset),
                        out=Path(args.out), real_share=args.real_share)
    print(f"corpus: {args.out} rows={manifest['rows']} sha256={manifest['corpus_sha256']}")
    parts = manifest["parts"]
    print(f"  real={parts['real']['count']} (of {manifest['dataset']['requested_real']} requested) "
          f"synthetic={parts['synthetic']['count']} {parts['synthetic']['languages']}")

    if args.self_check:
        with tempfile.TemporaryDirectory(prefix="scale-corpus-selfcheck-") as tmp:
            again = generate(rows=args.rows, seed=args.seed, dataset=Path(args.dataset),
                             out=Path(tmp) / "corpus.jsonl", real_share=args.real_share)
        same = again["corpus_sha256"] == manifest["corpus_sha256"]
        print(f"self-check: same seed -> {'byte-identical' if same else 'DIFFERENT'} "
              f"({again['corpus_sha256']})")
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
