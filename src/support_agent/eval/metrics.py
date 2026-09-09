"""Intent, escalation and cost metrics. Pure functions over aligned per-item arrays; no I/O."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from ..agent.schemas import HARD_REASONS, ROUTING_REASONS, SOFT_REASONS
from .stats import bootstrap_ci, mean_with_ci

# --- intents --------------------------------------------------------------------------------------------


def intent_metrics(y_true, y_pred, labels: list[str], y_true_secondary=None) -> dict:
    """Accuracy (headline) with CI, macro-F1 (secondary, with per-class support), strict vs lenient scoring.

    Lenient counts a prediction as correct when it matches the primary OR the labelled secondary intent.
    """
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    strict = (y_true == y_pred).astype(float)
    result = {"n": int(len(y_true)), "accuracy": mean_with_ci(strict)}
    if y_true_secondary is not None:
        sec = np.asarray([s if s not in (None, "", "none") else None for s in y_true_secondary], dtype=object)
        lenient = ((y_true == y_pred) | (sec == y_pred)).astype(float)
        result["accuracy_lenient"] = mean_with_ci(lenient)
    macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    lo, hi = bootstrap_ci(np.arange(len(y_true)), n_boot=500,
                          stat=lambda idx: f1_score(y_true[idx.astype(int)], y_pred[idx.astype(int)],
                                                    labels=labels, average="macro", zero_division=0))
    result["macro_f1"] = {"value": float(macro), "ci_low": lo, "ci_high": hi}
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    result["per_class"] = [
        {"intent": lab, "precision": float(pi), "recall": float(ri), "f1": float(fi), "support": int(si)}
        for lab, pi, ri, fi, si in zip(labels, p, r, f, s)
    ]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    result["confusion"] = pd.DataFrame(cm, index=labels, columns=labels)
    result["per_item_correct"] = strict.tolist()
    return result


# --- escalation -----------------------------------------------------------------------------------------

TRIVIAL_REASONS = ("non_english", "no_actionable_content")


@dataclass
class CostMatrix:
    missed_hard: float = 10.0
    missed_soft: float = 3.0
    unnecessary: float = 1.0


def _as_bool(x) -> np.ndarray:
    return np.asarray(x, dtype=bool)


def escalation_metrics(true_escalate, pred_escalate, true_reason, costs: CostMatrix | None = None) -> dict:
    """Precision/recall/F1 on 'escalate', hard- and soft-category recall, automation rate, expected cost.

    `true_reason` is the labeller's reason code (one of the schema codes, 'none' when not escalated).
    Items whose only reason is trivially detectable (non-English, empty) are also reported excluded, so a
    chunk of recall from regex-level cases cannot flatter the model.
    """
    costs = costs or CostMatrix()
    t, p = _as_bool(true_escalate), _as_bool(pred_escalate)
    reason = np.asarray(true_reason, dtype=object)
    hard = np.isin(reason, HARD_REASONS)
    soft = np.isin(reason, SOFT_REASONS)
    trivial = np.isin(reason, TRIVIAL_REASONS)

    tp = int(np.sum(t & p)); fp = int(np.sum(~t & p)); fn = int(np.sum(t & ~p)); tn = int(np.sum(~t & ~p))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp + fp and tp + fn and (precision + recall) else float("nan")

    def recall_on(mask) -> dict:
        if not mask.any():
            return {"value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
        return mean_with_ci(p[mask].astype(float))

    non_trivial = ~trivial
    tp_nt = int(np.sum(t & p & non_trivial)); fn_nt = int(np.sum(t & ~p & non_trivial))
    per_item_cost = item_costs(t, p, reason, costs)
    return {
        "n": int(len(t)), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "recall_ci": bootstrap_ci(p[t].astype(float)) if t.any() else (float("nan"), float("nan")),
        "hard_recall": recall_on(hard),
        "soft_recall": recall_on(soft),
        "recall_excluding_trivial": tp_nt / (tp_nt + fn_nt) if tp_nt + fn_nt else float("nan"),
        "automation_rate": mean_with_ci((~p).astype(float)),
        "missed_hard_rate": float(np.sum(hard & ~p) / len(t)) if len(t) else float("nan"),
        "n_hard": int(hard.sum()), "n_soft": int(soft.sum()), "n_trivial": int(trivial.sum()),
        "expected_cost_per_100": mean_with_ci(per_item_cost * 100),
        "per_item_correct": (t == p).astype(float).tolist(),
    }


def item_costs(true_escalate, pred_escalate, true_reason, costs: CostMatrix) -> np.ndarray:
    """Per-item cost: a missed hard escalation is the expensive error; needless escalation is cheap but not free."""
    t, p = _as_bool(true_escalate), _as_bool(pred_escalate)
    reason = np.asarray(true_reason, dtype=object)
    hard = np.isin(reason, HARD_REASONS)
    cost = np.zeros(len(t), dtype=float)
    missed = t & ~p
    cost[missed & hard] = costs.missed_hard
    cost[missed & ~hard] = costs.missed_soft
    cost[~t & p] = costs.unnecessary
    return cost


def cost_sensitivity(true_escalate, pred_escalate, true_reason, ratios=(1, 2, 3, 5, 10, 20),
                     soft_over_unnecessary: float = 3.0) -> pd.DataFrame:
    """Expected cost per 100 messages as the missed-hard : unnecessary-escalation ratio varies."""
    rows = []
    for ratio in ratios:
        c = CostMatrix(missed_hard=float(ratio), missed_soft=min(float(ratio), soft_over_unnecessary), unnecessary=1.0)
        rows.append({"ratio": ratio, "cost_per_100": float(item_costs(true_escalate, pred_escalate, true_reason, c).mean() * 100)})
    return pd.DataFrame(rows)


def reason_code_agreement(true_reason, pred_reason, true_escalate, pred_escalate) -> dict:
    """Among items both sides escalate: exact code match, and match at the hard/soft/routing tier."""
    t, p = _as_bool(true_escalate), _as_bool(pred_escalate)
    both = t & p
    tr = np.asarray(true_reason, dtype=object)[both]
    pr = np.asarray(pred_reason, dtype=object)[both]

    def tier(code):
        if code in HARD_REASONS:
            return "hard"
        if code in SOFT_REASONS:
            return "soft"
        if code in ROUTING_REASONS:
            return "routing"
        return "none"

    tiers_match = np.array([tier(a) == tier(b) for a, b in zip(tr, pr)], dtype=float)
    return {"n_both_escalated": int(both.sum()),
            "exact_match": float(np.mean(tr == pr)) if both.any() else float("nan"),
            "tier_match": float(tiers_match.mean()) if both.any() else float("nan")}


def operating_curve(model_escalate, retrieval_sim, true_escalate, true_reason,
                    thresholds=np.linspace(0.0, 1.0, 101), forced_escalate=None) -> pd.DataFrame:
    """Sweep the 'no relevant resolution' similarity floor: items below it are escalated regardless of the model.

    Lets the report state automation rate at a fixed missed-hard budget without any new model calls.
    `forced_escalate` marks items other post-processing rules already escalate (language, hard rules...).
    """
    m = _as_bool(model_escalate)
    sim = np.asarray([np.nan if s is None else s for s in retrieval_sim], dtype=float)
    forced = _as_bool(forced_escalate) if forced_escalate is not None else np.zeros(len(m), dtype=bool)
    t = _as_bool(true_escalate)
    reason = np.asarray(true_reason, dtype=object)
    hard = np.isin(reason, HARD_REASONS)
    rows = []
    for th in thresholds:
        below = np.nan_to_num(sim, nan=np.inf) < th
        pred = m | forced | below
        rows.append({
            "threshold": float(th),
            "automation_rate": float(np.mean(~pred)),
            "missed_hard_rate": float(np.sum(hard & ~pred) / len(t)),
            "escalation_recall": float(np.sum(t & pred) / t.sum()) if t.any() else float("nan"),
            "escalation_precision": float(np.sum(t & pred) / pred.sum()) if pred.any() else float("nan"),
        })
    return pd.DataFrame(rows)


def automation_at_budget(curve: pd.DataFrame, max_missed_hard_rate: float = 0.02) -> dict:
    ok = curve[curve["missed_hard_rate"] <= max_missed_hard_rate]
    if ok.empty:
        return {"threshold": float("nan"), "automation_rate": 0.0, "missed_hard_rate": float("nan"), "achievable": False}
    best = ok.sort_values(["automation_rate", "threshold"], ascending=[False, True]).iloc[0]
    return {"threshold": float(best["threshold"]), "automation_rate": float(best["automation_rate"]),
            "missed_hard_rate": float(best["missed_hard_rate"]), "achievable": True}


# --- reply validity -------------------------------------------------------------------------------------


def violation_rates(violations_per_item: list[list[str]], codes: tuple[str, ...] | None = None) -> dict:
    n = len(violations_per_item)
    all_codes = codes or tuple(sorted({c for v in violations_per_item for c in v}))
    rates = {c: sum(c in v for v in violations_per_item) / n if n else float("nan") for c in all_codes}
    rates["any"] = sum(bool(v) for v in violations_per_item) / n if n else float("nan")
    return rates
