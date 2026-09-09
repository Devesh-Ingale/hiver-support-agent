from __future__ import annotations

import numpy as np
import pytest

from support_agent.eval.agreement import binary_agreement, length_bias, ordinal_agreement, pairwise_agreement
from support_agent.eval.metrics import (CostMatrix, automation_at_budget, cost_sensitivity, escalation_metrics,
                                        intent_metrics, item_costs, operating_curve, reason_code_agreement,
                                        violation_rates)
from support_agent.eval.stats import bootstrap_ci, mcnemar, paired_bootstrap_diff


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    vals = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    lo, hi = bootstrap_ci(vals)
    assert lo <= 0.7 <= hi and 0.3 < lo and hi <= 1.0
    assert bootstrap_ci(vals) == (lo, hi)
    assert all(np.isnan(bootstrap_ci([])))


def test_paired_diff_and_mcnemar():
    a = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1] * 5
    b = [1, 0, 1, 0, 1, 0, 1, 0, 0, 1] * 5
    d = paired_bootstrap_diff(a, b)
    assert d["diff"] == pytest.approx(0.4) and d["significant"] is True and d["ci_low"] > 0
    m = mcnemar(a, b)
    assert m["a_only_correct"] == 20 and m["b_only_correct"] == 0 and m["p_value"] < 0.001
    assert mcnemar([1, 1], [1, 1])["p_value"] == 1.0


def test_intent_metrics_strict_and_lenient():
    labels = ["a", "b", "c"]
    y_true = ["a", "a", "b", "b", "c", "c"]
    y_pred = ["a", "b", "b", "c", "c", "c"]
    y_sec = [None, "b", None, "none", "", None]
    m = intent_metrics(y_true, y_pred, labels, y_true_secondary=y_sec)
    assert m["accuracy"]["value"] == pytest.approx(4 / 6)
    assert m["accuracy_lenient"]["value"] == pytest.approx(5 / 6)  # item 2's secondary 'b' rescues it
    assert m["confusion"].loc["b", "c"] == 1 and m["confusion"].values.sum() == 6
    support = {row["intent"]: row["support"] for row in m["per_class"]}
    assert support == {"a": 2, "b": 2, "c": 2}
    assert 0 < m["macro_f1"]["value"] < 1 and m["macro_f1"]["ci_low"] <= m["macro_f1"]["value"] <= m["macro_f1"]["ci_high"]


def _esc_fixture():
    #            0      1      2      3       4      5      6      7
    true = [True, True, True, False, False, True, True, False]
    reason = ["payment_refund", "account_or_pii", "anger_churn", "none", "none", "non_english", "legal_safety_threat", "none"]
    pred = [True, False, True, True, False, True, True, False]
    return true, pred, reason


def test_escalation_metrics_and_costs():
    true, pred, reason = _esc_fixture()
    m = escalation_metrics(true, pred, reason)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (4, 1, 1, 2)
    assert m["precision"] == pytest.approx(4 / 5) and m["recall"] == pytest.approx(4 / 5)
    assert m["hard_recall"]["value"] == pytest.approx(2 / 3) and m["n_hard"] == 3   # missed the account_or_pii item
    assert m["soft_recall"]["value"] == 1.0
    assert m["recall_excluding_trivial"] == pytest.approx(3 / 4)                     # drop the non_english item
    assert m["automation_rate"]["value"] == pytest.approx(3 / 8)
    assert m["missed_hard_rate"] == pytest.approx(1 / 8)
    costs = item_costs(true, pred, reason, CostMatrix())
    assert costs.tolist() == [0, 10, 0, 1, 0, 0, 0, 0]
    assert m["expected_cost_per_100"]["value"] == pytest.approx(11 / 8 * 100)
    curve = cost_sensitivity(true, pred, reason, ratios=(1, 10))
    assert curve["cost_per_100"].tolist() == pytest.approx([2 / 8 * 100, 11 / 8 * 100])


def test_reason_code_agreement():
    true, pred, reason = _esc_fixture()
    pred_reason = ["payment_refund", "none", "explicit_human_request", "anger_churn", "none", "non_english", "legal_safety_threat", "none"]
    r = reason_code_agreement(reason, pred_reason, true, pred)
    assert r["n_both_escalated"] == 4
    assert r["exact_match"] == pytest.approx(3 / 4)
    assert r["tier_match"] == pytest.approx(3 / 4)  # anger_churn (soft) vs explicit_human_request (hard) differ in tier


def test_operating_curve_and_budget():
    true, model_pred, reason = _esc_fixture()
    sims = [0.9, 0.1, 0.5, 0.9, 0.9, None, 0.7, 0.8]
    curve = operating_curve(model_pred, sims, true, reason, thresholds=[0.0, 0.2, 0.95])
    # threshold 0.0: nothing forced; automation 3/8, one missed hard (item 1)
    assert curve.iloc[0]["automation_rate"] == pytest.approx(3 / 8) and curve.iloc[0]["missed_hard_rate"] == pytest.approx(1 / 8)
    # threshold 0.2: item 1 (sim 0.1) is now escalated -> no missed hard, automation 2/8
    assert curve.iloc[1]["missed_hard_rate"] == 0.0 and curve.iloc[1]["automation_rate"] == pytest.approx(2 / 8)
    # threshold 0.95: everything with a similarity is escalated; only the None-sim item keeps the model's call
    assert curve.iloc[2]["automation_rate"] == 0.0
    best = automation_at_budget(curve, max_missed_hard_rate=0.02)
    assert best["achievable"] and best["threshold"] == pytest.approx(0.2) and best["automation_rate"] == pytest.approx(2 / 8)
    assert automation_at_budget(curve.iloc[:1], max_missed_hard_rate=0.0)["achievable"] is False


def test_violation_rates():
    rates = violation_rates([[], ["too_long"], ["too_long", "promise"], []])
    assert rates["too_long"] == 0.5 and rates["promise"] == 0.25 and rates["any"] == 0.5


def test_agreement_stats():
    b = binary_agreement(["pass", "fail", "pass", "pass", "fail", "pass"], ["pass", "fail", "pass", "fail", "fail", "pass"])
    assert b["percent_agreement"] == pytest.approx(5 / 6) and 0 < b["kappa"] < 1
    o = ordinal_agreement([5, 4, 3, 2, 1, 4, 4], [5, 4, 3, 1, 1, 3, 4])
    assert o["spearman"] > 0.9 and o["exact_agreement"] == pytest.approx(5 / 7) and o["within_one"] == 1.0
    assert o["crosstab"].loc[2, 1] == 1 and o["crosstab"].values.sum() == 7
    p = pairwise_agreement(["A", "B", "tie", "A", "B"], ["A", "B", "A", "B", "B"])
    assert p["percent_agreement"] == pytest.approx(3 / 5) and p["n_both_decisive"] == 4
    assert p["agreement_when_both_decisive"] == pytest.approx(3 / 4)
    lb = length_bias([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
    assert lb["spearman"] == pytest.approx(1.0)
    assert np.isnan(length_bias([3, 3, 3], [1, 2, 3])["spearman"])
