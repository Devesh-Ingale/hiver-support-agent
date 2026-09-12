"""Recompute every number in the report from files on disk. No model calls, no network.

Inputs: data/golden/{test,dev}.jsonl, outputs/runs/*.jsonl, outputs/judge/*.jsonl, outputs/human/*.jsonl,
plus the small JSON side files. Outputs: outputs/results/metrics.json, outputs/results/tables.md and one
per-item CSV per system for failure analysis.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..agent.schemas import HARD_REASONS
from ..config import Settings
from ..llm.batch import read_jsonl
from .agreement import binary_agreement, length_bias, ordinal_agreement, pairwise_agreement
from .metrics import (CostMatrix, automation_at_budget, cost_sensitivity, escalation_metrics, intent_metrics,
                      operating_curve, reason_code_agreement, violation_rates)
from .stats import mcnemar, mean_with_ci, paired_bootstrap_diff

SYSTEMS_WITH_DECISIONS = ("main", "main_gemini", "no_rag", "simple", "trivial_escalate", "trivial_auto")
JUDGED_SYSTEMS = ("main", "main_gemini", "no_rag", "simple", "trivial_escalate", "human_ref")


def _clean(obj: Any) -> Any:
    """Make nested results JSON-serialisable (DataFrames -> records, numpy -> python, NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return {"index": [str(i) for i in obj.index], "columns": [str(c) for c in obj.columns], "data": _clean(obj.values.tolist())}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if (isinstance(obj, float) or isinstance(obj, np.floating)) and math.isnan(float(obj)) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def load_runs(runs_dir: Path) -> dict[str, dict[str, dict]]:
    """system -> item_id -> row (test-set runs only: files without a _dev suffix)."""
    runs: dict[str, dict[str, dict]] = {}
    for path in sorted(runs_dir.glob("*.jsonl")):
        if path.stem.endswith("_dev"):
            continue
        rows = [r for r in read_jsonl(path) if "error" not in r]
        if rows:
            runs[path.stem] = {r["item_id"]: r for r in rows}
    return runs


def _aligned(test: list[dict], rows: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    items = [t for t in test if t["item_id"] in rows]
    return items, [rows[t["item_id"]] for t in items]


def system_block(test: list[dict], rows: dict[str, dict], labels: list[str], costs: CostMatrix, part_filter: str | None = None) -> dict:
    """All label-based metrics for one system on the test items (optionally only Part A)."""
    if part_filter:
        test = [t for t in test if t.get("part") == part_filter]
    items, preds = _aligned(test, rows)
    if not items:
        return {"n": 0}
    y_true = [t["intent_primary"] for t in items]
    y_sec = [t.get("intent_secondary") for t in items]
    y_pred = [p["intent"] for p in preds]
    true_esc = [bool(t["escalate"]) for t in items]
    pred_esc = [p["decision"] == "escalate" for p in preds]
    true_reason = [t["reason_code"] for t in items]
    pred_reason = [p.get("reason_code") or "none" for p in preds]
    block: dict[str, Any] = {"n": len(items), "coverage": len(items) / len(test)}
    block["intent"] = intent_metrics(y_true, y_pred, labels, y_true_secondary=y_sec) if all(p["intent"] for p in preds) else None
    block["escalation"] = escalation_metrics(true_esc, pred_esc, true_reason, costs)
    block["escalation"]["reason_agreement"] = reason_code_agreement(true_reason, pred_reason, true_esc, pred_esc)
    block["cost_sensitivity"] = cost_sensitivity(true_esc, pred_esc, true_reason)
    block["violations_raw_draft"] = violation_rates([p.get("violations") or [] for p in preds])
    block["parse_failure_rate"] = float(np.mean([bool(p.get("parse_failed")) for p in preds]))
    forced = pd.Series([p.get("forced_reason") for p in preds]).value_counts(dropna=True)
    block["forced_by_rule"] = {str(k): int(v) for k, v in forced.items()}
    block["model_vs_final_escalation"] = {
        "model_escalate_rate": float(np.mean([p.get("model_decision") == "escalate" for p in preds])),
        "final_escalate_rate": float(np.mean(pred_esc)),
    }
    lat = [p.get("latency_ms") for p in preds if p.get("latency_ms")]
    block["latency_ms"] = {"mean": float(np.mean(lat)) if lat else None, "p90": float(np.percentile(lat, 90)) if lat else None}
    toks_in = [p.get("input_tokens") for p in preds if p.get("input_tokens")]
    toks_out = [p.get("output_tokens") for p in preds if p.get("output_tokens")]
    block["tokens"] = {"mean_input": float(np.mean(toks_in)) if toks_in else None, "mean_output": float(np.mean(toks_out)) if toks_out else None}
    return block


def operating_block(test: list[dict], rows: dict[str, dict], max_missed_hard: float = 0.02) -> dict:
    items, preds = _aligned(test, rows)
    if not items:
        return {}
    other_forced = [p.get("forced_reason") not in (None, "no_relevant_resolution") for p in preds]
    curve = operating_curve([p.get("model_decision") == "escalate" for p in preds],
                            [p.get("retrieval_max_sim") for p in preds],
                            [bool(t["escalate"]) for t in items], [t["reason_code"] for t in items],
                            forced_escalate=other_forced)
    return {"curve": curve, "best_at_budget": automation_at_budget(curve, max_missed_hard)}


def confidence_reliability(test: list[dict], rows: dict[str, dict]) -> pd.DataFrame:
    items, preds = _aligned(test, rows)
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame({"conf": [p.get("intent_confidence") or 0 for p in preds],
                       "correct": [p["intent"] == t["intent_primary"] for p, t in zip(preds, items)]})
    df["bucket"] = pd.cut(df["conf"], bins=[-0.01, 0.6, 0.9, 1.0], labels=["<=0.6", "0.6-0.9", ">0.9"])
    return df.groupby("bucket", observed=False).agg(n=("correct", "size"), accuracy=("correct", "mean"), mean_conf=("conf", "mean")).reset_index()


def subgroup_block(test: list[dict], rows: dict[str, dict]) -> dict:
    items, preds = _aligned(test, rows)
    if not items:
        return {}
    df = pd.DataFrame({
        "part": [t.get("part") for t in items],
        "correct": [p["intent"] == t["intent_primary"] for p, t in zip(preds, items)],
        "esc_correct": [(p["decision"] == "escalate") == bool(t["escalate"]) for p, t in zip(preds, items)],
        "missed_hard": [(t["reason_code"] in HARD_REASONS) and p["decision"] != "escalate" for p, t in zip(preds, items)],
        "sim": [p.get("retrieval_max_sim") if p.get("retrieval_max_sim") is not None else np.nan for p in preds],
        "tokens": [t.get("n_content_tokens") or 0 for t in items],
        "conf": [t.get("labeller_confidence_1_3") for t in items],
        "intent": [t["intent_primary"] for t in items],
        "trivial": [t["reason_code"] in ("non_english", "no_actionable_content") for t in items],
        # who wrote the label: blind author, agreed model proposals, or the assistant's adjudication
        "label_source": [("author_blind" if t.get("labeller", "author") == "author" else t.get("label_source", "assistant")) for t in items],
    })
    median_tokens = df["tokens"].median()
    df["recurring"] = np.where(df["sim"].isna(), "n/a", np.where(df["sim"] >= 0.5, "recurring (sim>=0.5)", "novel (sim<0.5)"))
    df["length"] = np.where(df["tokens"] > median_tokens, f"long (>{int(median_tokens)} tokens)", f"short (<={int(median_tokens)} tokens)")

    def agg(col):
        g = df.groupby(col, observed=False).agg(n=("correct", "size"), intent_accuracy=("correct", "mean"),
                                                escalation_accuracy=("esc_correct", "mean"), missed_hard=("missed_hard", "sum"))
        return g.reset_index()

    return {"by_part": agg("part"), "recurring_vs_novel": agg("recurring"), "short_vs_long": agg("length"),
            "by_labeller_confidence": agg("conf"), "trivial_in_out": agg("trivial"), "by_intent": agg("intent"),
            "by_label_source": agg("label_source")}


def judge_block(judge_dir: Path, runs: dict[str, dict[str, dict]], test_ids: set[str]) -> dict:
    out: dict[str, Any] = {"absolute": {}, "pairwise": {}, "validation": None, "perturbation": None, "consistency": None}
    for path in sorted(judge_dir.glob("absolute_*.jsonl")):
        system = path.stem[len("absolute_"):]
        rows = [r for r in read_jsonl(path) if not r.get("judge_parse_failed") and r["item_id"] in test_ids]
        if not rows:
            continue
        run_rows = runs.get(system, {})
        auto_rows = [r for r in rows if run_rows.get(r["item_id"], {}).get("decision") == "auto"]
        block = {
            "n": len(rows),
            "pass_rate": mean_with_ci([bool(r["sendable"]) for r in rows]),
            "overall_mean": mean_with_ci([r["overall_1_5"] for r in rows]),
            "checks": {k: float(np.mean([bool(r[k]) for r in rows])) for k in ("unsupported_content", "addresses_problem", "unsafe_or_overpromising", "tweet_valid", "decision_reason_consistent")},
            "pass_rate_auto_handled": mean_with_ci([bool(r["sendable"]) for r in auto_rows]) if auto_rows else None,
            "n_auto_handled": len(auto_rows),
            "length_bias": length_bias([r["overall_1_5"] for r in rows], [r.get("reply_chars") or 0 for r in rows]),
            "judge_parse_failures": sum(r.get("judge_parse_failed", False) for r in read_jsonl(path)),
        }
        out["absolute"][system] = block
    for path in sorted(judge_dir.glob("pairwise_*.jsonl")):
        rows = [r for r in read_jsonl(path) if not r.get("judge_parse_failed") and r["item_id"] in test_ids]
        if not rows:
            continue
        x, y = rows[0]["system_x"], rows[0]["system_y"]
        combined = pd.Series([r["combined"] for r in rows])
        out["pairwise"][path.stem[len("pairwise_"):]] = {
            "n": len(rows), "system_x": x, "system_y": y,
            "win_rate_x": mean_with_ci([c == x for c in combined]),
            "win_rate_y": mean_with_ci([c == y for c in combined]),
            "tie_rate": float(np.mean(combined == "tie")), "flip_rate": float(np.mean(combined == "flip")),
            "weak_rate": float(np.mean([bool(r.get("weak")) for r in rows])),
            "win_rate_x_excluding_flips": float(np.mean([c == x for c in combined if c != "flip"])) if (combined != "flip").any() else None,
        }
    val = judge_dir / "validation.jsonl"
    if val.exists():
        rows = [r for r in read_jsonl(val) if not r.get("judge_parse_failed")]
        if rows:
            correct = [bool(r["sendable"]) == bool(r["expected_sendable"]) for r in rows]
            out["validation"] = {"n": len(rows), "accuracy": float(np.mean(correct)),
                                 "cases": [{"label": r["label"], "expected_sendable": r["expected_sendable"], "judge_sendable": r["sendable"],
                                            "overall": r["overall_1_5"], "unsupported": r["unsupported_content"]} for r in rows]}
    pert = judge_dir / "perturbation.jsonl"
    if pert.exists():
        rows = [r for r in read_jsonl(pert) if not r.get("judge_parse_failed")]
        orig = {r["item_id"]: r for r in rows if r["variant"] == "original"}
        pert_rows = [r for r in rows if r["variant"] == "perturbed" and r["item_id"] in orig]
        if pert_rows:
            out["perturbation"] = {
                "n": len(pert_rows),
                "unsupported_flag_rate_original": float(np.mean([bool(orig[r["item_id"]]["unsupported_content"]) for r in pert_rows])),
                "unsupported_flag_rate_perturbed": float(np.mean([bool(r["unsupported_content"]) for r in pert_rows])),
                "pass_rate_original": float(np.mean([bool(orig[r["item_id"]]["sendable"]) for r in pert_rows])),
                "pass_rate_perturbed": float(np.mean([bool(r["sendable"]) for r in pert_rows])),
                "overall_drop": float(np.mean([orig[r["item_id"]]["overall_1_5"] - r["overall_1_5"] for r in pert_rows])),
            }
    cons = judge_dir / "consistency.jsonl"
    if cons.exists():
        rows = [r for r in read_jsonl(cons) if not r.get("judge_parse_failed")]
        base = {}
        for r in read_jsonl(judge_dir / "absolute_main.jsonl") if (judge_dir / "absolute_main.jsonl").exists() else []:
            base[r["item_id"]] = r
        pairs = [(base[r["item_id"]], r) for r in rows if r["item_id"] in base and not base[r["item_id"]].get("judge_parse_failed")]
        if pairs:
            out["consistency"] = {
                "n": len(pairs),
                "sendable": binary_agreement([bool(a["sendable"]) for a, _ in pairs], [bool(b["sendable"]) for _, b in pairs]),
                "overall": {k: v for k, v in ordinal_agreement([a["overall_1_5"] for a, _ in pairs], [b["overall_1_5"] for _, b in pairs]).items() if k != "crosstab"},
            }
    return out


def human_block(human_dir: Path, judge_dir: Path) -> dict:
    out: dict[str, Any] = {}
    abs_path, pairs_path = human_dir / "absolute.jsonl", human_dir / "pairs.jsonl"
    if abs_path.exists():
        human = read_jsonl(abs_path)
        judge_by = {}
        for path in judge_dir.glob("absolute_*.jsonl"):
            system = path.stem[len("absolute_"):]
            for r in read_jsonl(path):
                if not r.get("judge_parse_failed"):
                    judge_by[(system, r["item_id"])] = r
        pairs = [(h, judge_by[(h["system"], h["item_id"])]) for h in human if (h["system"], h["item_id"]) in judge_by]
        if pairs:
            per_system = {}
            for system in sorted({h["system"] for h, _ in pairs}):
                sub = [(h, j) for h, j in pairs if h["system"] == system]
                per_system[system] = {"n": len(sub), "sendable_agreement": float(np.mean([bool(h["sendable"]) == bool(j["sendable"]) for h, j in sub]))}
            disagreements = [{"item_id": h["item_id"], "system": h["system"], "human_overall": h["overall_1_5"], "judge_overall": j["overall_1_5"],
                              "human_sendable": h["sendable"], "judge_sendable": j["sendable"], "reply": h["reply"], "judge_rationale": j.get("rationale")}
                             for h, j in pairs if bool(h["sendable"]) != bool(j["sendable"]) or abs(h["overall_1_5"] - j["overall_1_5"]) >= 2]
            out["absolute"] = {
                "n": len(pairs), "n_human_rated": len(human),
                "sendable": binary_agreement([bool(h["sendable"]) for h, _ in pairs], [bool(j["sendable"]) for _, j in pairs]),
                "overall": ordinal_agreement([h["overall_1_5"] for h, _ in pairs], [j["overall_1_5"] for _, j in pairs]),
                "checks_agreement": {k: float(np.mean([bool(h[k]) == bool(j[k]) for h, j in pairs])) for k in ("unsupported_content", "addresses_problem", "unsafe_or_overpromising", "tweet_valid", "decision_reason_consistent")},
                "per_system": per_system, "disagreements": disagreements,
                "human_seconds_mean": float(np.mean([h.get("seconds") or 0 for h in human])),
            }
    if pairs_path.exists():
        human = read_jsonl(pairs_path)
        judge_pairs = {}
        for path in judge_dir.glob("pairwise_*.jsonl"):
            for r in read_jsonl(path):
                if not r.get("judge_parse_failed"):
                    judge_pairs[r["item_id"]] = r
        both = [(h, judge_pairs[h["item_id"]]) for h in human if h["item_id"] in judge_pairs]
        if both:
            def norm(v: str) -> str:
                return "tie" if v in ("tie", "flip") else v
            out["pairwise"] = {
                "n": len(both), "n_human_rated": len(human),
                **pairwise_agreement([norm(h["preferred_system"]) for h, _ in both], [norm(j["combined"]) for _, j in both]),
                "human_prefers_main_rate": float(np.mean([h["preferred_system"] == "main" for h, _ in both])),
                "judge_prefers_main_rate": float(np.mean([j["combined"] == "main" for _, j in both])),
                "human_tie_rate": float(np.mean([h["preferred_system"] == "tie" for h, _ in both])),
            }
    return out


def variance_block(runs: dict[str, dict[str, dict]]) -> dict | None:
    base = runs.get("main")
    rerun = next((runs[k] for k in runs if k.startswith("main_seed")), None)
    if not base or not rerun:
        return None
    common = sorted(set(base) & set(rerun))
    if not common:
        return None
    return {"n": len(common), "rerun_file": next(k for k in runs if k.startswith("main_seed")),
            "intent_agreement": float(np.mean([base[i]["intent"] == rerun[i]["intent"] for i in common])),
            "decision_agreement": float(np.mean([base[i]["decision"] == rerun[i]["decision"] for i in common])),
            "reply_identical": float(np.mean([base[i]["reply"] == rerun[i]["reply"] for i in common]))}


def gate_verdict(metrics: dict, settings: Settings) -> dict:
    main = metrics["systems"].get("main", {})
    esc = main.get("escalation", {})
    trivial = metrics["systems"].get("trivial_escalate", {}).get("escalation", {})
    judge_main = metrics.get("judge", {}).get("absolute", {}).get("main", {})
    hard = esc.get("hard_recall", {})
    conds = {
        "1_hard_recall_ge_0.95_lowerCI_ge_0.85": (hard.get("value") is not None and hard.get("value") >= 0.95 and hard.get("ci_low", 0) >= 0.85, hard),
        "2_zero_validity_violations_on_auto_sent": (main.get("auto_sent_violation_count") == 0, main.get("auto_sent_violation_count")),
        "3_judge_pass_rate_auto_handled_ge_0.70": ((judge_main.get("pass_rate_auto_handled") or {}).get("value", 0) >= 0.70 if judge_main else None, judge_main.get("pass_rate_auto_handled") if judge_main else None),
        "4_cost_below_always_escalate": (esc.get("expected_cost_per_100", {}).get("value", np.inf) < trivial.get("expected_cost_per_100", {}).get("value", -np.inf) if esc and trivial else None,
                                         {"main": esc.get("expected_cost_per_100"), "always_escalate": trivial.get("expected_cost_per_100")}),
        "5_automation_at_2pct_missed_hard": (True, metrics.get("operating_point", {}).get("best_at_budget")),
    }
    passed = all(v[0] for v in conds.values() if v[0] is not None)
    return {"conditions": {k: {"passed": v[0], "evidence": v[1]} for k, v in conds.items()}, "all_passed": passed,
            "untested": [k for k, v in conds.items() if v[0] is None]}


def paired_comparisons(test: list[dict], runs: dict[str, dict[str, dict]]) -> dict:
    out = {}
    main = runs.get("main")
    if not main:
        return out
    for other in ("simple", "no_rag", "main_gemini", "trivial_escalate", "trivial_auto"):
        if other not in runs:
            continue
        common = [t for t in test if t["item_id"] in main and t["item_id"] in runs[other]]
        if len(common) < 5:
            continue
        a = [main[t["item_id"]]["intent"] == t["intent_primary"] for t in common]
        b = [runs[other][t["item_id"]]["intent"] == t["intent_primary"] for t in common]
        esc_a = [(main[t["item_id"]]["decision"] == "escalate") == bool(t["escalate"]) for t in common]
        esc_b = [(runs[other][t["item_id"]]["decision"] == "escalate") == bool(t["escalate"]) for t in common]
        out[f"main_vs_{other}"] = {"n": len(common), "intent_accuracy_diff": paired_bootstrap_diff(a, b), "intent_mcnemar": mcnemar(a, b),
                                  "escalation_accuracy_diff": paired_bootstrap_diff(esc_a, esc_b)}
    return out


def evaluate_dev(settings: Settings) -> str:
    """Dev-set numbers for prompt/threshold iteration: label-based metrics only, printed, never the headline.

    Reads data/golden/dev.jsonl and outputs/runs/*_dev.jsonl. Also sweeps the 'no relevant resolution'
    similarity floor for `main` so the value frozen in settings.yaml is chosen on dev, not test.
    """
    paths = settings.paths
    dev = read_jsonl(paths.golden / "dev.jsonl")
    if not dev:
        raise FileNotFoundError("data/golden/dev.jsonl is missing — run `label --finalize` first")
    labels = sorted({t["intent_primary"] for t in dev})
    costs = CostMatrix(settings.cost_missed_hard, settings.cost_missed_soft, settings.cost_unnecessary_escalation)
    lines = [f"# Dev set (n={len(dev)}) — iteration numbers, not the headline", ""]
    lines.append("label distribution: " + ", ".join(f"{k}={v}" for k, v in pd.Series([t["intent_primary"] for t in dev]).value_counts().items()))
    lines.append("escalation labels: " + ", ".join(f"{k}={v}" for k, v in pd.Series([t["reason_code"] for t in dev]).value_counts().items()))
    lines.append("")
    for path in sorted(paths.runs.glob("*_dev.jsonl")):
        rows = {r["item_id"]: r for r in read_jsonl(path) if "error" not in r}
        system = path.stem[: -len("_dev")]
        b = system_block(dev, rows, labels, costs)
        if not b.get("n"):
            continue
        it, esc = b.get("intent") or {}, b["escalation"]
        versions = sorted({(r.get("prompt_version"), r.get("taxonomy_version")) for r in rows.values()})
        lines.append(f"## {system}  (n={b['n']}, versions {versions})")
        lines.append(f"- intent accuracy {_pct(it.get('accuracy'))} · lenient {_pct(it.get('accuracy_lenient'))} · macro-F1 {_num((it.get('macro_f1') or {}).get('value'))}")
        lines.append(f"- escalation P/R/F1 {_num(esc.get('precision'), '{:.2f}')}/{_num(esc.get('recall'), '{:.2f}')}/{_num(esc.get('f1'), '{:.2f}')} · hard recall {_pct(esc.get('hard_recall'))} (n_hard={esc.get('n_hard')}) · "
                     f"automation {_pct(esc.get('automation_rate'))} · cost/100 {_num(esc.get('expected_cost_per_100', {}).get('value'), '{:.0f}')} · missed hard {esc.get('missed_hard_rate', 0):.1%}")
        lines.append(f"- reason-code agreement (both escalated): exact {_num(esc['reason_agreement'].get('exact_match'), '{:.2f}')} on n={esc['reason_agreement'].get('n_both_escalated')}")
        lines.append(f"- raw-draft violations {b['violations_raw_draft']} · forced by rule {b['forced_by_rule']} · parse failures {b['parse_failure_rate']:.1%}")
        if it.get("per_class"):
            worst = sorted(it["per_class"], key=lambda r: r["f1"])[:3]
            lines.append("- weakest intents: " + "; ".join(f"{w['intent']} F1 {w['f1']:.2f} (n={w['support']})" for w in worst))
            cm = it["confusion"]
            off = [(cm.index[i], cm.columns[j], int(cm.values[i, j])) for i in range(len(cm)) for j in range(len(cm)) if i != j and cm.values[i, j] >= 2]
            if off:
                lines.append("- confusions ≥2: " + "; ".join(f"{a}→{p} ×{n}" for a, p, n in sorted(off, key=lambda x: -x[2])))
        if system == "main":
            op = operating_block(dev, rows)
            curve = op["curve"]
            lines.append("- similarity-floor sweep (threshold → automation, missed-hard): " + "; ".join(
                f"{r.threshold:.2f}→{r.automation_rate:.0%},{r.missed_hard_rate:.1%}" for r in curve.iloc[::10].itertuples()))
            lines.append(f"- best floor at ≤2 % missed hard: {op['best_at_budget']}")
        lines.append("")
    text = "\n".join(lines)
    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / "dev_metrics.md").write_text(text, encoding="utf-8")
    return text


def auto_sent_violations(test: list[dict], rows: dict[str, dict]) -> int:
    items, preds = _aligned(test, rows)
    return int(sum(1 for p in preds if p["decision"] == "auto" and p.get("violations")))


def evaluate(settings: Settings) -> dict:
    paths = settings.paths
    test = read_jsonl(paths.golden / "test.jsonl")
    if not test:
        raise FileNotFoundError("data/golden/test.jsonl is missing — label and finalize the golden set first")
    labels = sorted({t["intent_primary"] for t in test})
    runs = load_runs(paths.runs)
    costs = CostMatrix(settings.cost_missed_hard, settings.cost_missed_soft, settings.cost_unnecessary_escalation)
    test_ids = {t["item_id"] for t in test}

    metrics: dict[str, Any] = {
        "brand": settings.brand, "n_test": len(test), "n_part_a": sum(t.get("part") == "A" for t in test),
        "intent_labels": labels, "label_distribution": {k: int(v) for k, v in pd.Series([t["intent_primary"] for t in test]).value_counts().items()},
        "escalation_label_distribution": {k: int(v) for k, v in pd.Series([t["reason_code"] for t in test]).value_counts().items()},
        "systems": {}, "systems_part_a": {},
    }
    for system, rows in runs.items():
        if system.startswith("main_seed") or system == "human_ref":
            continue
        metrics["systems"][system] = system_block(test, rows, labels, costs)
        metrics["systems"][system]["auto_sent_violation_count"] = auto_sent_violations(test, rows)
        metrics["systems_part_a"][system] = system_block(test, rows, labels, costs, part_filter="A")
    if "main" in runs:
        metrics["operating_point"] = operating_block(test, runs["main"])
        metrics["confidence_reliability"] = confidence_reliability(test, runs["main"])
        metrics["subgroups"] = subgroup_block(test, runs["main"])
    metrics["paired"] = paired_comparisons(test, runs)
    metrics["variance"] = variance_block(runs)
    metrics["judge"] = judge_block(paths.judge, runs, test_ids)
    metrics["human_agreement"] = human_block(paths.human, paths.judge)
    for side in ("self_agreement.json", "sampling_stats.json"):
        p = paths.golden / side
        if p.exists():
            metrics[side.replace(".json", "")] = json.loads(p.read_text(encoding="utf-8"))
    split_stats = paths.processed / f"{settings.brand}_split_stats.json"
    if split_stats.exists():
        metrics["split_stats"] = json.loads(split_stats.read_text(encoding="utf-8"))
    metrics["gate"] = gate_verdict(metrics, settings)

    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / "metrics.json").write_text(json.dumps(_clean(metrics), indent=2, ensure_ascii=False), encoding="utf-8")
    (paths.results / "tables.md").write_text(render_tables(metrics), encoding="utf-8")
    for system, rows in runs.items():
        items, preds = _aligned(test, rows)
        if items:
            pd.DataFrame([{**{k: t.get(k) for k in ("item_id", "part", "root_text", "intent_primary", "intent_secondary", "escalate", "reason_code", "labeller_confidence_1_3", "quality_flags")},
                            **{f"pred_{k}": p.get(k) for k in ("intent", "decision", "reason_code", "reason", "reply", "forced_reason", "violations", "retrieval_max_sim", "model_decision")}}
                           for t, p in zip(items, preds)]).to_csv(paths.results / f"items_{system}.csv", index=False)
    return metrics


# ----------------------------------------------------------------------------------------------------------
# markdown tables
# ----------------------------------------------------------------------------------------------------------
def _pct(m: dict | None, key: str = "value") -> str:
    if not m or m.get(key) is None or (isinstance(m.get(key), float) and math.isnan(m[key])):
        return "—"
    v = m[key]
    lo, hi = m.get("ci_low"), m.get("ci_high")
    if lo is None or hi is None or (isinstance(lo, float) and math.isnan(lo)):
        return f"{v:.1%}"
    return f"{v:.1%} [{lo:.0%}, {hi:.0%}]"


def _num(v, fmt: str = "{:.2f}") -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else fmt.format(v)


def render_tables(m: dict) -> str:
    lines = [f"# Results tables — {m['brand']} (test set, n={m['n_test']}; Part A n={m['n_part_a']})", ""]
    lines += ["## Headline (all test items, A+B)", "",
              "| system | intent acc [95% CI] | macro-F1 | esc. recall | hard recall | automation | cost/100 | raw-draft violations | judge pass | judge pass (auto only) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    judge_abs = m.get("judge", {}).get("absolute", {})
    for system in [s for s in ("main", "main_gemini", "no_rag", "simple", "trivial_escalate", "trivial_auto") if s in m["systems"]]:
        b = m["systems"][system]
        if not b.get("n"):
            continue
        it, esc = b.get("intent") or {}, b["escalation"]
        j = judge_abs.get(system, {})
        lines.append(f"| {system} | {_pct(it.get('accuracy'))} | {_num((it.get('macro_f1') or {}).get('value'))} | {_num(esc.get('recall'), '{:.1%}')} | "
                     f"{_pct(esc.get('hard_recall'))} | {_pct(esc.get('automation_rate'))} | {_num(esc.get('expected_cost_per_100', {}).get('value'), '{:.0f}')} | "
                     f"{_num(b['violations_raw_draft'].get('any'), '{:.1%}')} | {_pct(j.get('pass_rate'))} | {_pct(j.get('pass_rate_auto_handled'))} |")
    if "human_ref" in judge_abs:
        j = judge_abs["human_ref"]
        lines.append(f"| human_ref (brand's real replies) | — | — | — | — | — | — | — | {_pct(j.get('pass_rate'))} | — |")
    lines.append("")
    lines += ["## Headline on Part A only (uniform random sample)", "", "| system | intent acc [95% CI] | esc. recall | hard recall | automation | cost/100 |", "|---|---|---|---|---|---|"]
    for system, b in m.get("systems_part_a", {}).items():
        if b.get("n"):
            it, esc = b.get("intent") or {}, b["escalation"]
            lines.append(f"| {system} | {_pct(it.get('accuracy'))} | {_num(esc.get('recall'), '{:.1%}')} | {_pct(esc.get('hard_recall'))} | {_pct(esc.get('automation_rate'))} | {_num(esc.get('expected_cost_per_100', {}).get('value'), '{:.0f}')} |")
    lines.append("")
    pw = m.get("judge", {}).get("pairwise", {})
    if pw:
        lines += ["## Pairwise judge (both orders)", "", "| comparison | n | X wins | Y wins | tie | flip (position bias) |", "|---|---|---|---|---|---|"]
        for name, b in pw.items():
            lines.append(f"| {b['system_x']} vs {b['system_y']} | {b['n']} | {_pct(b['win_rate_x'])} | {_pct(b['win_rate_y'])} | {_num(b['tie_rate'], '{:.1%}')} | {_num(b['flip_rate'], '{:.1%}')} |")
        lines.append("")
    ha = m.get("human_agreement", {})
    if ha:
        lines += ["## Human vs judge agreement", ""]
        if "absolute" in ha:
            a = ha["absolute"]
            lines.append(f"- absolute ratings: n={a['n']}; sendable agreement {a['sendable']['percent_agreement']:.1%}, κ={a['sendable']['kappa']:.2f} "
                         f"[{a['sendable']['ci_low']:.2f}, {a['sendable']['ci_high']:.2f}]; overall Spearman ρ={a['overall'].get('spearman', float('nan')):.2f}, "
                         f"weighted κ={a['overall'].get('weighted_kappa', {}).get('kappa', float('nan')):.2f}, within-1 {a['overall'].get('within_one', float('nan')):.1%}")
        if "pairwise" in ha:
            p = ha["pairwise"]
            lines.append(f"- pairwise preferences: n={p['n']}; agreement {p['percent_agreement']:.1%} (κ={p['kappa_kappa']:.2f}); when both decisive {_num(p['agreement_when_both_decisive'], '{:.1%}')}; "
                         f"human prefers main {p['human_prefers_main_rate']:.1%} vs judge {p['judge_prefers_main_rate']:.1%}")
        lines.append("")
    if m.get("gate"):
        lines += ["## Acceptance gate", ""]
        for k, v in m["gate"]["conditions"].items():
            status = "PASS" if v["passed"] else ("untested" if v["passed"] is None else "FAIL")
            lines.append(f"- {k}: **{status}**")
        lines.append(f"\nAll tested conditions passed: **{m['gate']['all_passed']}**" + (f" (untested: {m['gate']['untested']})" if m['gate']['untested'] else ""))
    return "\n".join(lines) + "\n"
