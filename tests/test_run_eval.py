"""End-to-end check of the offline aggregator on a tiny synthetic result set (no models, no network)."""
from __future__ import annotations

import json
from pathlib import Path

from support_agent.config import Paths, Settings
from support_agent.eval.figures import render_all
from support_agent.eval.run_eval import evaluate
from support_agent.llm.batch import write_jsonl

INTENTS = ["playback_issue", "payment_billing", "account_access", "other_unclear"]


def tmp_settings(tmp: Path) -> Settings:
    paths = Paths(root=tmp, raw=tmp / "raw", interim=tmp / "interim", processed=tmp / "processed", golden=tmp / "golden",
                  outputs=tmp / "outputs", runs=tmp / "outputs/runs", judge=tmp / "outputs/judge", human=tmp / "outputs/human",
                  results=tmp / "outputs/results", report=tmp / "report", figures=tmp / "report/figures",
                  taxonomy_dir=tmp / "tax", cache_db=tmp / "outputs/cache.sqlite")
    return Settings(brand="SpotifyCares", paths=paths)


def build_fixture(tmp: Path) -> Settings:
    s = tmp_settings(tmp)
    test = []
    for i in range(12):
        intent = INTENTS[i % 4]
        escalate = intent in ("payment_billing", "account_access")
        reason = {"payment_billing": "payment_refund", "account_access": "account_or_pii"}.get(intent, "none")
        test.append({"item_id": f"G{i:04d}", "root_id": i, "root_text": f"tweet {i} about {intent}", "part": "A" if i < 8 else "B",
                     "intent_primary": intent, "intent_secondary": None, "escalate": escalate, "reason_code": reason,
                     "labeller_confidence_1_3": 3 if i % 3 else 2, "n_content_tokens": 5 + i, "quality_flags": [], "lang": "en",
                     "first_brand_reply": "@1 Sorry! Try a clean reinstall /JR"})
    write_jsonl(s.paths.golden / "test.jsonl", test)

    def row(t, system, intent, decision, reason_code, reply, sim=0.6, model_decision=None, forced=None, violations=None, provider="ollama"):
        return {"item_id": t["item_id"], "system": system, "provider": provider, "model": "m", "intent": intent, "intent_secondary": None,
                "intent_confidence": 0.95 if intent == t["intent_primary"] else 0.5, "reply": reply, "decision": decision,
                "reason_code": reason_code, "reason": "r", "evidence_ids": [], "model_decision": model_decision or decision,
                "model_reason_code": reason_code, "forced_reason": forced, "violations": violations or [], "retrieval_max_sim": sim,
                "evidence": [], "parse_failed": False, "repaired": False, "latency_ms": 100, "input_tokens": 500, "output_tokens": 80}

    main, simple, triv, no_rag, ref, seed7 = [], [], [], [], [], []
    for i, t in enumerate(test):
        correct = i % 6 != 5                       # main gets 10/12 intents right (items 5 and 11 wrong)
        intent = t["intent_primary"] if correct else INTENTS[(i + 1) % 4]
        esc = t["escalate"] if i != 9 else False   # one missed hard escalation (item 9 = payment_billing)
        main.append(row(t, "main", intent, "escalate" if esc else "auto", t["reason_code"] if esc else "none", f"Sorry! grounded reply {i}",
                        sim=0.2 if i == 9 else 0.6, forced="payment_refund" if (esc and t["reason_code"] == "payment_refund") else None))
        seed7.append({**main[-1], "system": "main_seed7", "intent": intent if i != 2 else "other_unclear"})
        simple.append(row(t, "simple", INTENTS[(i + 1) % 4], "escalate" if t["reason_code"] == "payment_refund" else "auto",
                          "payment_refund" if t["reason_code"] == "payment_refund" else "none", f"verbatim reply {i}", provider="rule"))
        triv.append(row(t, "trivial_escalate", "playback_issue", "escalate", "none", "Sorry! Please DM us the details.", sim=None, provider="rule"))
        no_rag.append(row(t, "no_rag", intent, "auto", "none", f"ungrounded reply {i} https://fake.url", sim=None, violations=["url_not_in_evidence"]))
        ref.append(row(t, "human_ref", None, None, None, "Sorry! Try a clean reinstall", sim=None, provider="human"))
    for name, rows in [("main", main), ("simple", simple), ("trivial_escalate", triv), ("no_rag", no_rag), ("human_ref", ref), ("main_seed7", seed7)]:
        write_jsonl(s.paths.runs / f"{name}.jsonl", rows)
    write_jsonl(s.paths.runs / "main_dev.jsonl", main[:2])   # must be ignored by the aggregator

    def judged(r, sendable, overall, unsupported=False):
        return {"item_id": r["item_id"], "system": r["system"], "reply": r["reply"], "reply_chars": len(r["reply"] or ""), "rationale": "x",
                "unsupported_content": unsupported, "addresses_problem": True, "unsafe_or_overpromising": False, "tweet_valid": True,
                "decision_reason_consistent": True, "overall_1_5": overall, "sendable": sendable, "judge_parse_failed": False}

    write_jsonl(s.paths.judge / "absolute_main.jsonl", [judged(r, i % 4 != 0, 4 if i % 4 else 2) for i, r in enumerate(main)])
    write_jsonl(s.paths.judge / "absolute_simple.jsonl", [judged(r, i % 2 == 0, 3) for i, r in enumerate(simple)])
    write_jsonl(s.paths.judge / "absolute_human_ref.jsonl", [judged(r, True, 4) for r in ref])
    write_jsonl(s.paths.judge / "pairwise_main_vs_simple.jsonl",
                [{"item_id": r["item_id"], "system_x": "main", "system_y": "simple", "combined": ["main", "main", "simple", "tie", "flip", "main"][i % 6],
                  "flipped": i % 6 == 4, "weak": False, "judge_parse_failed": False} for i, r in enumerate(main)])
    write_jsonl(s.paths.judge / "validation.jsonl", [{"item_id": "VAL-a", "label": "good", "expected_sendable": True, "sendable": True, "overall_1_5": 4, "unsupported_content": False, "judge_parse_failed": False},
                                                     {"item_id": "VAL-b", "label": "bad", "expected_sendable": False, "sendable": False, "overall_1_5": 1, "unsupported_content": True, "judge_parse_failed": False}])
    write_jsonl(s.paths.judge / "perturbation.jsonl", [
        {"pair_key": "G0000:original", "item_id": "G0000", "variant": "original", "unsupported_content": False, "sendable": True, "overall_1_5": 4, "judge_parse_failed": False},
        {"pair_key": "G0000:perturbed", "item_id": "G0000", "variant": "perturbed", "unsupported_content": True, "sendable": False, "overall_1_5": 2, "judge_parse_failed": False}])
    write_jsonl(s.paths.judge / "consistency.jsonl", [judged(r, i % 4 != 0, 4 if i % 4 else 3) for i, r in enumerate(main[:8])])
    write_jsonl(s.paths.human / "absolute.jsonl", [{**judged(r, i % 4 != 0 if i != 1 else False, 4 if i % 4 else 2), "rating_id": f"R{i}", "seconds": 40} for i, r in enumerate(main[:8])])
    write_jsonl(s.paths.human / "pairs.jsonl", [{"item_id": r["item_id"], "preferred_system": ["main", "simple", "tie", "main"][i % 4], "seconds": 20} for i, r in enumerate(main)])
    (s.paths.golden / "self_agreement.json").write_text(json.dumps({"n": 40, "intent": {"kappa": 0.8}}), encoding="utf-8")
    return s


def test_evaluate_end_to_end(tmp_path):
    s = build_fixture(tmp_path)
    m = evaluate(s)
    assert (s.paths.results / "metrics.json").exists() and (s.paths.results / "tables.md").exists()
    assert set(m["systems"]) == {"main", "simple", "trivial_escalate", "no_rag"}   # human_ref and seed re-run are not scored systems
    main = m["systems"]["main"]
    assert main["n"] == 12 and main["intent"]["accuracy"]["value"] == 10 / 12
    esc = main["escalation"]
    assert esc["n_hard"] == 6 and esc["hard_recall"]["value"] == 5 / 6 and esc["missed_hard_rate"] == 1 / 12
    assert main["forced_by_rule"] == {"payment_refund": 2}   # items 1 and 5; item 9 is the missed one
    assert m["systems"]["no_rag"]["violations_raw_draft"]["any"] == 1.0 and m["systems"]["no_rag"]["auto_sent_violation_count"] == 12
    assert m["systems_part_a"]["main"]["n"] == 8
    assert m["operating_point"]["best_at_budget"]["achievable"] is True   # floor at 0.2 catches item 9 (sim 0.2 < 0.21)
    assert m["variance"]["n"] == 12 and m["variance"]["intent_agreement"] == 11 / 12
    assert m["paired"]["main_vs_simple"]["n"] == 12
    j = m["judge"]
    assert j["absolute"]["main"]["n"] == 12 and j["absolute"]["main"]["pass_rate"]["value"] == 9 / 12
    assert j["absolute"]["main"]["n_auto_handled"] == 7 and j["absolute"]["human_ref"]["pass_rate"]["value"] == 1.0
    pw = j["pairwise"]["main_vs_simple"]
    assert pw["n"] == 12 and pw["flip_rate"] == 2 / 12 and pw["win_rate_x"]["value"] == 6 / 12
    assert j["validation"]["accuracy"] == 1.0 and j["perturbation"]["unsupported_flag_rate_perturbed"] == 1.0
    assert j["consistency"]["n"] == 8
    ha = m["human_agreement"]
    assert ha["absolute"]["n"] == 8 and 0 < ha["absolute"]["sendable"]["percent_agreement"] <= 1
    assert ha["pairwise"]["n"] == 12 and "percent_agreement" in ha["pairwise"]
    assert m["self_agreement"]["n"] == 40
    gate = m["gate"]
    assert gate["conditions"]["1_hard_recall_ge_0.95_lowerCI_ge_0.85"]["passed"] is False   # 5/6 recall fails the gate, as it should
    assert gate["conditions"]["2_zero_validity_violations_on_auto_sent"]["passed"] is True
    tables = (s.paths.results / "tables.md").read_text(encoding="utf-8")
    assert "| main |" in tables and "Acceptance gate" in tables and "human_ref" in tables
    assert (s.paths.results / "items_main.csv").exists()
    # metrics.json must be plain JSON (no NaN / numpy leftovers)
    json.loads((s.paths.results / "metrics.json").read_text(encoding="utf-8"))

    figs = render_all(s)
    names = {p.name for p in figs}
    assert {"confusion_main.png", "cost_sensitivity.png", "operating_curve.png", "judge_pass_rates.png", "pairwise.png", "human_vs_judge.png"} <= names
    assert all(p.stat().st_size > 5_000 for p in figs)
