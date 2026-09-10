from __future__ import annotations

import pytest

from support_agent.eval.label_cli import Labeller, parse_compact_code
from support_agent.llm.batch import read_jsonl

CANDS = [
    {"item_id": "G0001", "root_id": 11, "root_text": "@SpotifyCares app crashes when I open a playlist", "root_created_at": "2017-11-20T10:00:00+00:00"},
    {"item_id": "G0002", "root_id": 12, "root_text": "@SpotifyCares charged twice, refund NOW this is ridiculous", "root_created_at": "2017-11-20T11:00:00+00:00"},
    {"item_id": "G0003", "root_id": 13, "root_text": "@SpotifyCares @115712 same here", "root_created_at": "2017-11-20T12:00:00+00:00"},
    {"item_id": "G0004", "root_id": 14, "root_text": "@SpotifyCares 😩", "root_created_at": "2017-11-20T13:00:00+00:00"},
]


def test_parse_compact_code_grammar():
    assert parse_compact_code("1", 10) == {"intent": 1, "secondary": None, "reason": None, "anger": 0, "confidence": 3, "flags": [], "note": None}
    p = parse_compact_code("5e2", 10)
    assert p["intent"] == 5 and p["reason"] == "payment_refund" and p["anger"] == 0
    p = parse_compact_code("3/9a1c2", 10)
    assert p["secondary"] == 9 and p["anger"] == 1 and p["confidence"] == 2
    p = parse_compact_code("10fni", 10)
    assert p["intent"] == 10 and p["flags"] == ["noise_or_spam", "image_only"]
    p = parse_compact_code("2E1 a1  n: not sure hacked vs forgot password", 10)
    assert p["reason"] == "account_or_pii" and p["anger"] == 1 and p["note"] == "not sure hacked vs forgot password"
    p = parse_compact_code("1 fam", 10)
    assert p["flags"] == ["ambiguous", "multi_intent"]
    assert parse_compact_code("4e7", 10)["reason"] == "no_actionable_content"


@pytest.mark.parametrize("bad", ["", "abc", "0", "11", "3/3", "3/12", "1e9", "1a3", "1c0", "1fz", "1q", "e2"])
def test_parse_compact_code_rejects(bad):
    with pytest.raises(ValueError):
        parse_compact_code(bad, 10)


def scripted(answers):
    it = iter(answers)
    return lambda prompt: next(it)


def test_compact_session(taxonomy, tmp_path):
    out = tmp_path / "labels.jsonl"
    printed = []
    answers = [
        "1",                                  # G0001: playback, auto
        "9",                                  # invalid (4 intents) -> re-prompt
        "2e2a2c2fam n:double issue",          # G0002: payment, escalate refund, anger 2, conf 2, flags, note
        "x", "fragment addressed to another customer",   # G0003 excluded
        "4e7fi c1",                           # G0004: other, no_actionable_content, image flag, confidence 1 (spaces are ignored)
    ]
    lab = Labeller(taxonomy, CANDS, out, round_no=1, input_fn=scripted(answers), print_fn=printed.append)
    assert lab.run() == 4
    rows = read_jsonl(out)
    r1, r2, r3, r4 = rows
    assert r1["intent_primary"] == "playback_issue" and r1["escalate"] is False and r1["reason_code"] == "none"
    assert r1["anger_0_2"] == 0 and r1["labeller_confidence_1_3"] == 3 and r1["quality_flags"] == [] and r1["needs_reply"] is True
    assert r2["intent_primary"] == "payment_billing" and r2["escalate"] is True and r2["reason_code"] == "payment_refund"
    assert r2["anger_0_2"] == 2 and r2["labeller_confidence_1_3"] == 2 and r2["quality_flags"] == ["ambiguous", "multi_intent"] and r2["note"] == "double issue"
    assert r3["excluded"] is True and r3["exclusion_reason"].startswith("fragment")
    assert r4["intent_primary"] == "other_unclear" and r4["reason_code"] == "no_actionable_content" and r4["quality_flags"] == ["image_only"]
    assert r4["labeller_confidence_1_3"] == 1
    assert any("intent must be 1-4" in p for p in printed)      # the '9' was rejected with guidance
    assert any("CODE:" in p for p in printed)                   # compact legend is on the cheat sheet
