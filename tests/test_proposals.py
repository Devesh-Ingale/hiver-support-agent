from __future__ import annotations

import json

from support_agent.eval.label_cli import Labeller
from support_agent.eval.proposals import merge_proposals, proposal_accuracy, propose_label, proposal_schema
from support_agent.llm.base import LLMResponse
from support_agent.llm.batch import read_jsonl


class ScriptedLLM:
    provider = "fake"

    def __init__(self, model, *texts):
        self.model = model
        self.texts = list(texts)
        self.calls = []

    def complete(self, system, user, *, schema=None, temperature=0.0, seed=42, max_tokens=1024):
        self.calls.append((system, user, schema))
        return LLMResponse(text=self.texts.pop(0), provider=self.provider, model=self.model, latency_ms=1)


def test_propose_label_parses_and_sanitises(taxonomy):
    good = json.dumps({"intent": "payment_billing", "intent_secondary": "none", "escalate": True, "reason_code": "payment_refund", "anger": 1})
    llm = ScriptedLLM("m1", good, json.dumps({"intent": "made_up", "intent_secondary": "none", "escalate": True, "reason_code": "none", "anger": 7}), "garbage")
    p = propose_label(llm, taxonomy, "charged twice, refund please")
    assert p == {"model": "m1", "intent": "payment_billing", "intent_secondary": None, "escalate": True, "reason_code": "payment_refund", "anger": 1, "parse_failed": False}
    system, user, schema = llm.calls[0]
    assert "# Escalation checklist" in system and "charged twice" in user and schema == proposal_schema(taxonomy.intent_ids)
    bad = propose_label(llm, taxonomy, "x")
    assert bad["intent"] is None and bad["reason_code"] is None and bad["anger"] == 0 and bad["parse_failed"] is True
    garbage = propose_label(llm, taxonomy, "x")
    assert garbage["parse_failed"] is True


def test_merge_and_accuracy(taxonomy):
    a = {"model": "A", "intent": "playback_issue", "intent_secondary": None, "escalate": False, "reason_code": "none", "anger": 0, "parse_failed": False}
    b = {"model": "B", "intent": "playback_issue", "intent_secondary": None, "escalate": True, "reason_code": "anger_churn", "anger": 2, "parse_failed": False}
    m = merge_proposals("G1", a, b)
    assert m["agree_intent"] is True and m["agree_escalate"] is False and m["agree_reason"] is False
    proposals = {"G1": m, "G2": merge_proposals("G2", {**a, "intent": "other_unclear"}, {**b, "intent": "payment_billing", "escalate": False, "reason_code": "none"})}
    blind = [{"item_id": "G1", "intent_primary": "playback_issue", "escalate": False, "reason_code": "none"},
             {"item_id": "G2", "intent_primary": "payment_billing", "escalate": False, "reason_code": "none"},
             {"item_id": "G3", "intent_primary": "payment_billing", "escalate": False, "reason_code": "none", "assisted": True}]  # assisted rows are excluded
    acc = proposal_accuracy(proposals, blind)
    assert acc["n"] == 2
    assert acc["a"]["intent_accuracy"] == 0.5 and acc["b"]["intent_accuracy"] == 1.0
    assert acc["a"]["escalate_accuracy"] == 1.0 and acc["b"]["escalate_accuracy"] == 0.5
    assert acc["models_agree_intent"] == 0.5 and acc["intent_accuracy_when_models_agree"] == 1.0


def test_assisted_labelling_accept_and_override(taxonomy, tmp_path):
    cands = [{"item_id": "G1", "root_id": 1, "root_text": "app crashes", "root_created_at": "2017-11-20T10:00:00+00:00"},
             {"item_id": "G2", "root_id": 2, "root_text": "charged twice", "root_created_at": "2017-11-20T11:00:00+00:00"},
             {"item_id": "G3", "root_id": 3, "root_text": "no proposal here", "root_created_at": "2017-11-20T12:00:00+00:00"}]
    agree = {"model": "A", "intent": "playback_issue", "intent_secondary": None, "escalate": False, "reason_code": "none", "anger": 0, "parse_failed": False}
    a2 = {"model": "A", "intent": "payment_billing", "intent_secondary": None, "escalate": True, "reason_code": "payment_refund", "anger": 1, "parse_failed": False}
    b2 = {"model": "B", "intent": "account_access", "intent_secondary": None, "escalate": True, "reason_code": "account_or_pii", "anger": 0, "parse_failed": False}
    proposals = {"G1": merge_proposals("G1", agree, {**agree, "model": "B"}), "G2": merge_proposals("G2", a2, b2)}
    printed = []
    answers = iter(["", "2e2a2 n:took A but angrier", "4"])
    lab = Labeller(taxonomy, cands, tmp_path / "l.jsonl", input_fn=lambda p: next(answers), print_fn=printed.append, proposals=proposals)
    assert lab.run() == 3
    r1, r2, r3 = read_jsonl(tmp_path / "l.jsonl")
    assert r1["assisted"] is True and r1["accepted_proposal"] is True and r1["intent_primary"] == "playback_issue" and r1["escalate"] is False
    assert r1["models_agree_intent"] is True
    assert r2["assisted"] is True and r2["accepted_proposal"] is False and r2["intent_primary"] == "payment_billing" and r2["anger_0_2"] == 2
    assert r2["models_agree_intent"] is False and r2["note"] == "took A but angrier"
    assert "assisted" not in r3 and r3["intent_primary"] == "other_unclear"
    assert any("both models agree" in p for p in printed) and any("⚠ models disagree on intent" in p for p in printed)
    assert lab.proposal_code(a2) == "2e2a1"   # test taxonomy: payment_billing is intent 2
    assert lab.proposal_code(agree) == "1"
    assert lab.proposal_code({**a2, "reason_code": None}) is None