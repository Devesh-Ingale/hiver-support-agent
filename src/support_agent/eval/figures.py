"""Report figures (static PNG for the .docx), rendered from outputs/results/*.json|csv.

Palette and mark conventions follow one validated system: categorical hues assigned in fixed order
(blue, orange, aqua, yellow), sequential = one blue ramp, thin marks, hairline solid grid, text in ink
tokens (never the series colour), a legend whenever there are two or more series.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from ..config import Settings  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # categorical slots 1-4 (validated adjacent order)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"], "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "savefig.dpi": 200,
    "legend.frameon": False, "legend.fontsize": 8.5,
})


def _style(ax, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=0, labelsize=8.5)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def _df(obj) -> pd.DataFrame:
    """Rebuild a DataFrame serialised by run_eval._clean."""
    if isinstance(obj, pd.DataFrame):
        return obj
    return pd.DataFrame(obj["data"], index=obj["index"], columns=obj["columns"])


def _pct_axis(ax, axis: str = "x", decimals: int = 0) -> None:
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path


# --- 1. brand profile -------------------------------------------------------------------------------------
def fig_brand_profile(profile: pd.DataFrame, path: Path, chosen: str | None = None) -> Path:
    df = profile.sort_values("customer_threads", ascending=True)
    n = len(df)
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * n + 1.2))
    y = np.arange(n)
    h = 0.34
    ax.barh(y + h / 2, df["substantive_first_reply"], height=h, color=SERIES[0], label="first reply gives steps or a link")
    ax.barh(y - h / 2, df["first_reply_dm_redirect"], height=h, color=SERIES[1], label='first reply is a "DM us" redirect')
    labels = [f"{b}  ({int(v):,} threads)" for b, v in zip(df["brand"], df["customer_threads"])]
    ax.set_yticks(y, labels)
    for lab, brand in zip(ax.get_yticklabels(), df["brand"]):
        lab.set_color(INK if brand == chosen else INK2)
        if brand == chosen:
            lab.set_fontweight("semibold")
    _style(ax, grid_axis="x")
    _pct_axis(ax, "x")
    ax.set_xlim(0, 1)
    ax.set_title("How the 15 largest brands answer in public: substance vs boilerplate (share of first replies)")
    ax.legend(loc="lower right")
    return _save(fig, path)


# --- 2. confusion matrix ----------------------------------------------------------------------------------
def fig_confusion(cm: pd.DataFrame, path: Path, title: str = "Intent confusion matrix (rows = human label, columns = prediction)") -> Path:
    labels = list(cm.index)
    m = cm.values.astype(float)
    fig, ax = plt.subplots(figsize=(0.62 * len(labels) + 2.2, 0.55 * len(labels) + 1.6))
    ax.imshow(m, cmap=SEQ_CMAP, vmin=0, vmax=max(m.max(), 1))
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = int(m[i, j])
            if v == 0:
                continue
            color = "white" if m[i, j] > 0.55 * m.max() else INK
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8.5, color=color, fontweight="semibold" if i == j else "normal")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(length=0, labelsize=8)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title)
    return _save(fig, path)


# --- 3. cost sensitivity ----------------------------------------------------------------------------------
def fig_cost_sensitivity(curves: dict[str, pd.DataFrame], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    order = [s for s in ("main", "simple", "trivial_escalate", "trivial_auto", "no_rag", "main_gemini") if s in curves][:4]
    for color, system in zip(SERIES, order):
        c = curves[system]
        ax.plot(c["ratio"], c["cost_per_100"], color=color, linewidth=2, solid_capstyle="round", label=system)
        ax.scatter(c["ratio"].iloc[-1], c["cost_per_100"].iloc[-1], s=36, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(system, (c["ratio"].iloc[-1], c["cost_per_100"].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
    _style(ax)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20], ["1×", "2×", "3×", "5×", "10×", "20×"])
    ax.set_xlabel("cost of a missed hard escalation relative to one unnecessary escalation")
    ax.set_ylabel("expected cost per 100 messages")
    ax.set_title("Which policy is cheapest depends on how much a missed escalation hurts")
    ax.legend(loc="upper left")
    ax.margins(x=0.18)
    return _save(fig, path)


# --- 4. operating curve -----------------------------------------------------------------------------------
def fig_operating_curve(curve: pd.DataFrame, best: dict, path: Path, budget: float = 0.02) -> Path:
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    c = curve.sort_values("threshold")
    ax.plot(c["missed_hard_rate"], c["automation_rate"], color=SERIES[0], linewidth=2, solid_capstyle="round")
    ax.axvline(budget, color=AXIS, linewidth=0.8)
    ax.text(budget, ax.get_ylim()[1] if False else c["automation_rate"].max(), f"  budget: ≤{budget:.0%} missed hard escalations",
            color=INK2, fontsize=8.5, va="top")
    if best.get("achievable"):
        ax.scatter(best["missed_hard_rate"], best["automation_rate"], s=48, color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"operating point: {best['automation_rate']:.0%} automated\n(similarity floor {best['threshold']:.2f})",
                    (best["missed_hard_rate"], best["automation_rate"]), xytext=(10, -22), textcoords="offset points",
                    fontsize=8.5, color=INK, arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.8})
    _style(ax, grid_axis="both")
    _pct_axis(ax, "x", decimals=1)
    _pct_axis(ax, "y")
    ax.set_xlabel("missed hard escalations (share of all test messages)")
    ax.set_ylabel("messages auto-handled")
    ax.set_title("Automation vs safety as the 'no relevant resolution' floor is swept")
    return _save(fig, path)


# --- 5. judge pass rates ----------------------------------------------------------------------------------
def fig_pass_rates(judge_abs: dict, path: Path) -> Path:
    order = [s for s in ("main", "main_gemini", "no_rag", "simple", "trivial_escalate", "human_ref") if s in judge_abs]
    names = {"human_ref": "brand's real replies (reference)", "trivial_escalate": "trivial (template)", "no_rag": "main without retrieval",
             "main_gemini": "main prompts on gemini-3.8-flash", "simple": "simple (kNN verbatim)", "main": "main (local Qwen3-4B + retrieval)"}
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(order) + 1.2))
    y = np.arange(len(order))[::-1]
    for yi, system in zip(y, order):
        b = judge_abs[system]["pass_rate"]
        color = MUTED if system == "human_ref" else SERIES[0]
        ax.barh(yi, b["value"], height=0.5, color=color)
        ax.errorbar(b["value"], yi, xerr=[[b["value"] - b["ci_low"]], [b["ci_high"] - b["value"]]], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
        ax.text(min(b["ci_high"] + 0.02, 1.0), yi, f"{b['value']:.0%}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, [names.get(s, s) for s in order])
    _style(ax, grid_axis="x")
    _pct_axis(ax, "x")
    ax.set_xlim(0, 1.08)
    ax.set_title("Judge pass rate: “a brand agent would send this with at most a light edit” (95 % CI)")
    return _save(fig, path)


# --- 6. human vs judge cross-tab --------------------------------------------------------------------------
def fig_human_vs_judge(crosstab: pd.DataFrame, stats: dict, path: Path) -> Path:
    m = crosstab.values.astype(float)
    fig, ax = plt.subplots(figsize=(4.2, 3.9))
    ax.imshow(m, cmap=SEQ_CMAP, vmin=0, vmax=max(m.max(), 1))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j] > 0:
                ax.text(j, i, str(int(m[i, j])), ha="center", va="center", fontsize=9, color="white" if m[i, j] > 0.55 * m.max() else INK)
    ax.set_xticks(range(m.shape[1]), list(crosstab.columns))
    ax.set_yticks(range(m.shape[0]), list(crosstab.index))
    ax.set_xlabel("judge overall score (1–5)")
    ax.set_ylabel("human overall score (1–5)")
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    sub = f"Spearman ρ = {stats.get('spearman', float('nan')):.2f} · weighted κ = {stats.get('weighted_kappa', {}).get('kappa', float('nan')):.2f} · within-1 = {stats.get('within_one', float('nan')):.0%} · n = {stats.get('n', 0)}"
    ax.set_title("Human vs LLM-judge overall scores\n" + sub, fontsize=9.5)
    return _save(fig, path)


# --- 7. pairwise -------------------------------------------------------------------------------------------
def fig_pairwise(pairwise: dict, path: Path) -> Path:
    rows = list(pairwise.values())
    fig, ax = plt.subplots(figsize=(6.6, 0.6 * len(rows) + 1.3))
    for yi, b in enumerate(rows):
        parts = [(b["win_rate_x"]["value"], SERIES[0], f"{b['system_x']} preferred"), (b["tie_rate"], MUTED, "tie"),
                 (b["win_rate_y"]["value"], SERIES[1], f"{b['system_y']} preferred"), (b["flip_rate"], AXIS, "flip (position bias)")]
        left = 0.0
        for val, color, label in parts:
            ax.barh(yi, val, left=left, height=0.5, color=color, edgecolor=SURFACE, linewidth=2, label=label if yi == 0 else None)
            if val >= 0.08:
                ax.text(left + val / 2, yi, f"{val:.0%}", ha="center", va="center", fontsize=8.5,
                        color="white" if color in (SERIES[0], SERIES[1]) else INK)
            left += val
    ax.set_yticks(range(len(rows)), [f"{b['system_x']} vs {b['system_y']}  (n={b['n']})" for b in rows])
    _style(ax, grid_axis="x")
    _pct_axis(ax, "x")
    ax.set_xlim(0, 1)
    ax.set_title("Pairwise judge verdicts, both presentation orders combined")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4)
    return _save(fig, path)


# --- driver ------------------------------------------------------------------------------------------------
def render_all(settings: Settings) -> list[Path]:
    out: list[Path] = []
    fig_dir = settings.paths.figures
    profile_csv = settings.paths.results / "brand_profile.csv"
    if profile_csv.exists():
        out.append(fig_brand_profile(pd.read_csv(profile_csv), fig_dir / "brand_profile.png", chosen=settings.brand or None))
    metrics_path = settings.paths.results / "metrics.json"
    if not metrics_path.exists():
        return out
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    main = m.get("systems", {}).get("main", {})
    if main.get("intent") and main["intent"].get("confusion"):
        out.append(fig_confusion(_df(main["intent"]["confusion"]), fig_dir / "confusion_main.png"))
    curves = {s: _df(b["cost_sensitivity"]) for s, b in m.get("systems", {}).items() if b.get("cost_sensitivity")}
    if curves:
        out.append(fig_cost_sensitivity(curves, fig_dir / "cost_sensitivity.png"))
    op = m.get("operating_point") or {}
    if op.get("curve"):
        out.append(fig_operating_curve(_df(op["curve"]), op.get("best_at_budget", {}), fig_dir / "operating_curve.png"))
    judge_abs = m.get("judge", {}).get("absolute", {})
    if judge_abs:
        out.append(fig_pass_rates(judge_abs, fig_dir / "judge_pass_rates.png"))
    pairwise = m.get("judge", {}).get("pairwise", {})
    if pairwise:
        out.append(fig_pairwise(pairwise, fig_dir / "pairwise.png"))
    ha = m.get("human_agreement", {}).get("absolute")
    if ha and ha.get("overall", {}).get("crosstab"):
        out.append(fig_human_vs_judge(_df(ha["overall"]["crosstab"]), ha["overall"], fig_dir / "human_vs_judge.png"))
    return out
