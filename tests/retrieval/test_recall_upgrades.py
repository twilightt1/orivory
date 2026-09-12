"""Recall upgrades: lexical refinement, query routing, same-slot closure."""
from __future__ import annotations

# ── IDEA 1: lexical_bonus ─────────────────────────────────────────────

def test_lexical_bonus_exact_match():
    from app.retrieval.memory.scoring import lexical_bonus
    bonus, reasons = lexical_bonus('Hợp đồng với "Anh Tuấn" 2024', 'anh tuấn ký hợp đồng năm 2024')
    assert bonus == 0.15
    assert any("tuấn" in r.lower() for r in reasons)


def test_lexical_bonus_no_match_zero():
    from app.retrieval.memory.scoring import lexical_bonus
    bonus, reasons = lexical_bonus('"Anh Tuấn" Hà Nội', 'thời tiết hôm nay đẹp')
    assert bonus == 0.0
    assert reasons == []


def test_lexical_bonus_empty_query():
    from app.retrieval.memory.scoring import lexical_bonus
    assert lexical_bonus('', 'anything here') == (0.0, [])
