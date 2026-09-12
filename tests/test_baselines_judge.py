from __future__ import annotations

import json

import pandas as pd

from support_agent.baselines.simple import cv_intent_predictions, human_reference_rows, rule_decision, simple_rows, verbatim_reply
from support_agent.baselines.trivial import TEMPLATE_REPLY, majority_intent, trivial_rows
from support_agent.eval.judge import (VALIDATION_CASES, Judge, pairwise_both_orders, perturb_with_fabricated_step,
                                      validation_item)
from support_agent.llm.base import LLMResponse
from support_agent.retrieval.index import TfidfIndex

ITEMS = [
    {"item_id": "G1", "root_id": 1, "root_text": "@SpotifyCares app crashes when I open a playlist", "lang": "en", "first_brand_reply": "@1 Sorry! Try a clean reinstall /JR"},
    {"item_id": "G2", "root_id": 2, "root_text": "@SpotifyCares charged twice for premium, refund please", "lang": "en", "first_brand_reply": "@2 Please DM us your account email /AB"},
    {"item_id": "G3", "root_id": 3, "root_text": "@SpotifyCares 😩", "lang": "unk", "first_brand_reply": ""},
    {"item_id": "G4", "root_id": 4, "root_text": "@SpotifyCares la aplicación se cierra", "lang": "es", "first_brand_reply": "@4 Hola! Envíanos un DM"},
]


def index() -> TfidfIndex:
    docs = pd.DataFrame([
        ("d1", "app crashes when I open a playlist", "@9 Sorry! Try a clean reinstall of the app /JR", True),
        ("d2", "charged twice for premium", "@8 Please DM us your account email /AB", False),
    ], columns=["doc_id", "message", "brand_turns", "substantive"]).assign(first_reply=lambda d: d.brand_turns, created_at="2017-11-01")
    return TfidfIndex(min_df=1).fit(docs)


def test_trivial_baselines():
    labels = [{"intent_primary": "a"}, {"intent_primary": "b"}, {"intent_primary": "a"}, {"intent_primary": "c", "excluded": True}]
    assert majority_intent(labels) == "a"
    rows = trivial_rows(ITEMS, "a", always_escalate=True)
    assert {r["decision"] for r in rows} == {"escalate"} and {r["system"] for r in rows} == {"trivial_escalate"}
    assert rows[0]["reply"] == TEMPLATE_REPLY and rows[0]["violations"] == [] and rows[0]["intent"] == "a"
    assert {r["decision"] for r in trivial_rows(ITEMS, "a", always_escalate=False)} == {"auto"}


def test_cv_intent_predictions_are_out_of_fold_and_complete():
    texts = ["app crashes playlist", "app keeps crashing", "crash on open", "playlist crash again", "crashing app",
             "charged twice", "double charge refund", "billing wrong", "refund my payment", "charged again"]
    labels = ["play"] * 5 + ["pay"] * 5
    preds, confs = cv_intent_predictions(texts, labels, extra_texts=["payment failed"], extra_labels=["pay"], n_splits=5)
    assert len(preds) == 10 and set(preds) <= {"play", "pay"} and all(0 < c <= 1 for c in confs)
    assert sum(p == l for p, l in zip(preds, labels)) >= 8   # trivially separable vocabulary


def test_rule_decision_and_verbatim_reply(taxonomy):
    rules = taxonomy.hard_rules()
    assert rule_decision("charged twice, refund please", "en", rules)[:2] == ("escalate", "payment_refund")
    assert rule_decision("app crashes when opening a playlist", "en", rules)[:2] == ("auto", "none")
    assert rule_decision("app crashes", "en", rules)[1] == "no_actionable_content"   # two content tokens is too little
    assert rule_decision("😩", "unk", rules)[1] == "no_actionable_content"
    assert rule_decision("la aplicación se cierra", "es", rules)[1] == "non_english"
    reply, top, sim = verbatim_reply("my app crashes when opening a playlist", index(), "SpotifyCares")
    assert reply == "Sorry! Try a clean reinstall of the app" and top.doc_id == "d1" and 0 < sim <= 1
    assert verbatim_reply("zzzz", index(), "SpotifyCares") == ("", None, None)


def test_simple_rows_and_human_reference(taxonomy):
    rows = simple_rows(ITEMS, ["playback_issue", "payment_billing", "other_unclear", "other_unclear"], [0.9, 0.8, 0.3, 0.3],
                       index(), taxonomy.hard_rules(), "SpotifyCares")
    by = {r["item_id"]: r for r in rows}
    assert by["G1"]["decision"] == "auto" and by["G1"]["reply"].startswith("Sorry! Try a clean reinstall") and by["G1"]["violations"] == []
    assert by["G2"]["decision"] == "escalate" and by["G2"]["reason_code"] == "payment_refund" and by["G2"]["evidence"][0]["doc_id"] == "d2"
    # a verbatim "DM us" neighbour reply with no hard-rule keyword in the tweet is still an escalation
    handoff_rows = simple_rows([{"item_id": "G9", "root_id": 9, "root_text": "@SpotifyCares something is off with my premium account, can you check?", "lang": "en"}],
                               ["payment_billing"], [0.7], index(), taxonomy.hard_rules(), "SpotifyCares")
    assert handoff_rows[0]["reply"].startswith("Please DM us") and handoff_rows[0]["decision"] == "escalate" and handoff_rows[0]["reason_code"] == "payment_refund"
    assert by["G3"]["reason_code"] == "no_actionable_content" and by["G4"]["reason_code"] == "non_english"
    assert all(r["system"] == "simple" and r["intent"] for r in rows)
    ref = human_reference_rows(ITEMS)
    assert [r["item_id"] for r in ref] == ["G1", "G2", "G4"]           # G3 had no brand reply
    assert ref[0]["reply"] == "Sorry! Try a clean reinstall" and ref[0]["violations"] == [] and ref[0]["system"] == "human_ref"


class ScriptedJudgeLLM:
    provider, model = "fake", "judge-1"

    def __init__(self, *texts):
        self.texts = list(texts)
        self.calls = []

    def complete(self, system, user, *, schema=None, temperature=0.0, seed=42, max_tokens=1024):
        self.calls.append((system, user, schema))
        return LLMResponse(text=self.texts.pop(0), provider=self.provider, model=self.model, latency_ms=2)


def test_judge_absolute_and_pairwise(taxonomy):
    item, evidence = validation_item("SpotifyCares")
    good = json.dumps({"rationale": "grounded in E1", "unsupported_content": False, "addresses_problem": True,
                       "unsafe_or_overpromising": False, "tweet_valid": True, "decision_reason_consistent": True,
                       "overall_1_5": 4, "sendable": True})
    llm = ScriptedJudgeLLM(good, "garbage")
    judge = Judge(llm, taxonomy)
    row = {"system": "main", "reply": VALIDATION_CASES[0][1], "decision": "auto", "reason": "known fix"}
    v = judge.absolute(item, row, evidence)
    assert v["sendable"] is True and v["overall_1_5"] == 4 and v["judge_parse_failed"] is False and v["system"] == "main"
    system_prompt, user_prompt, schema = llm.calls[0]
    assert "SpotifyCares" in system_prompt and "ALWAYS escalate" in system_prompt
    assert "[E1]" in user_prompt and "clean reinstall" in user_prompt and list(schema["properties"])[0] == "rationale"
    assert "main" not in user_prompt.split("# Agent's draft reply")[1]   # system identity is never shown
    bad = judge.absolute(item, row, evidence)
    assert bad["judge_parse_failed"] is True and bad["sendable"] is None

    pref_a = json.dumps({"rationale": "A is grounded", "preferred": "A"})
    pref_b = json.dumps({"rationale": "B is grounded", "preferred": "B"})
    llm2 = ScriptedJudgeLLM(pref_a, pref_b)  # consistent: prefers the same content in both orders
    both = pairwise_both_orders(Judge(llm2, taxonomy), item, {"system": "main", "reply": "x"}, {"system": "simple", "reply": "y"}, evidence)
    assert both["combined"] == "main" and both["flipped"] is False and both["weak"] is False
    llm3 = ScriptedJudgeLLM(pref_a, pref_a)  # position bias: always slot A
    both = pairwise_both_orders(Judge(llm3, taxonomy), item, {"system": "main", "reply": "x"}, {"system": "simple", "reply": "y"}, evidence)
    assert both["combined"] == "flip" and both["flipped"] is True
    tie = json.dumps({"rationale": "same", "preferred": "tie"})
    both = pairwise_both_orders(Judge(ScriptedJudgeLLM(tie, pref_b), taxonomy), item, {"system": "main", "reply": "x"}, {"system": "simple", "reply": "y"}, evidence)
    assert both["combined"] == "main" and both["weak"] is True   # order 2 preferred slot B == main


def test_validation_cases_and_perturbation():
    labels = [c[0] for c in VALIDATION_CASES]
    assert len(labels) == len(set(labels)) and sum(c[2] for c in VALIDATION_CASES) == 2
    perturbed = perturb_with_fabricated_step(VALIDATION_CASES[0][1])
    assert "Hardware Acceleration" in perturbed and perturbed.startswith(VALIDATION_CASES[0][1][:20])
