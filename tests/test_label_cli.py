from __future__ import annotations

import itertools

from support_agent.eval.label_cli import Labeller, select_for_relabel, self_agreement
from support_agent.llm.batch import read_jsonl

CANDS = [
    {"item_id": "G0001", "root_id": 11, "root_text": "@SpotifyCares app crashes when I open a playlist", "root_created_at": "2017-11-20T10:00:00+00:00",
     "first_brand_reply": "SECRET-BRAND-REPLY try a clean reinstall", "part": "A"},
    {"item_id": "G0002", "root_id": 12, "root_text": "@SpotifyCares charged twice, refund NOW this is ridiculous", "root_created_at": "2017-11-20T11:00:00+00:00"},
    {"item_id": "G0003", "root_id": 13, "root_text": "@SpotifyCares @115712 same here", "root_created_at": "2017-11-20T12:00:00+00:00"},
    {"item_id": "G0004", "root_id": 14, "root_text": "@SpotifyCares 😩", "root_created_at": "2017-11-20T13:00:00+00:00"},
]


def scripted(answers):
    it = iter(answers)
    return lambda prompt: next(it)


def test_labelling_session_resume_exclusion_and_quit(taxonomy, tmp_path):
    out = tmp_path / "labels.jsonl"
    printed = []
    ticks = itertools.chain([0.0, 12.5, 100.0, 130.0, 200.0, 205.0], itertools.count(300.0))
    answers = [
        # item 1: playback, no secondary, auto, anger 0, confidence 3, no flags, no note
        "1", "", "n", "", "", "", "",
        # item 2: bad intent then payment, secondary 'none', escalate, reason 2 (payment_refund), anger 2, conf 2, flag a+m, note
        "9", "2", "", "y", "2", "2", "2", "am", "double issue",
        # item 3: exclude with reason
        "x", "fragment addressed to another customer",
        # item 4: quit
        "q",
    ]
    lab = Labeller(taxonomy, CANDS, out, round_no=1, input_fn=scripted(answers), print_fn=printed.append, clock=lambda: next(ticks), compact=False)
    n = lab.run()
    assert n == 3
    rows = read_jsonl(out)
    assert [r["item_id"] for r in rows] == ["G0001", "G0002", "G0003"]
    r1, r2, r3 = rows
    assert r1["intent_primary"] == "playback_issue" and r1["intent_secondary"] is None and r1["escalate"] is False
    assert r1["reason_code"] == "none" and r1["anger_0_2"] == 0 and r1["labeller_confidence_1_3"] == 3
    assert r1["quality_flags"] == [] and r1["needs_reply"] is True and r1["label_seconds"] == 12.5
    assert r2["intent_primary"] == "payment_billing" and r2["escalate"] is True and r2["reason_code"] == "payment_refund"
    assert r2["anger_0_2"] == 2 and r2["quality_flags"] == ["ambiguous", "multi_intent"] and r2["note"] == "double issue"
    assert r3["excluded"] is True and r3["exclusion_reason"].startswith("fragment")
    assert any("enter 1-4" in p for p in printed)          # the invalid '9' was rejected with guidance
    assert any("stopped; progress saved" in p for p in printed)
    assert "SpotifyCares" in printed[0] and "HARD" in printed[0]   # cheat sheet + policy shown first
    # the labeller never shows the brand's reply or the sampling part
    assert not any("SECRET-BRAND-REPLY" in p for p in printed)
    assert not any("part" in p.split("]")[-1] and "G0001" in p for p in printed)

    # resume: only the unlabelled item is offered; label it fully
    lab2 = Labeller(taxonomy, CANDS, out, round_no=1, input_fn=scripted(["4", "", "y", "7", "0", "1", "i", ""]), print_fn=printed.append, compact=False)
    assert lab2.run() == 1
    rows = read_jsonl(out)
    assert len(rows) == 4 and rows[-1]["item_id"] == "G0004" and rows[-1]["reason_code"] == "no_actionable_content"
    assert rows[-1]["quality_flags"] == ["image_only"] and rows[-1]["labeller_confidence_1_3"] == 1

    # round 2 is independent of round 1 progress
    lab3 = Labeller(taxonomy, CANDS[:1], out, round_no=2, input_fn=scripted(["2", "", "n", "", "", "", ""]), print_fn=printed.append, compact=False)
    assert lab3.run() == 1
    rows = read_jsonl(out)
    assert sum(r["round"] == 2 for r in rows) == 1


def test_relabel_selection_and_self_agreement():
    round1 = [
        {"item_id": "a", "round": 1, "intent_primary": "x", "escalate": True, "reason_code": "payment_refund"},
        {"item_id": "b", "round": 1, "intent_primary": "y", "escalate": False, "reason_code": "none"},
        {"item_id": "c", "round": 1, "intent_primary": "x", "escalate": True, "reason_code": "anger_churn"},
        {"item_id": "d", "round": 1, "excluded": True},
    ]
    picked = select_for_relabel(round1, n=2, seed=1)
    assert len(picked) == 2 and "d" not in picked
    round2 = [
        {"item_id": "a", "round": 2, "intent_primary": "x", "escalate": True, "reason_code": "payment_refund"},
        {"item_id": "b", "round": 2, "intent_primary": "x", "escalate": False, "reason_code": "none"},
        {"item_id": "c", "round": 2, "intent_primary": "x", "escalate": False, "reason_code": "none"},
    ]
    agg = self_agreement(round1, round2)
    assert agg["n"] == 3
    assert agg["intent"]["percent_agreement"] == 2 / 3
    assert agg["escalate"]["percent_agreement"] == 2 / 3
    assert agg["escalate_hard_items"] == {"n": 1, "percent_agreement": 1.0}
    assert agg["escalate_soft_items"]["n"] == 1 and agg["escalate_soft_items"]["percent_agreement"] == 0.0
