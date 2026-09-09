"""Human-vs-judge (and self-vs-self) agreement statistics, reported with intervals and never tuned."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from .stats import bootstrap_ci


def kappa(a, b, weights: str | None = None) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or (len(set(a)) == 1 and len(set(b)) == 1 and a[0] == b[0]):
        return float("nan") if len(a) == 0 else 1.0
    return float(cohen_kappa_score(a, b, weights=weights))


def kappa_with_ci(a, b, weights: str | None = None, n_boot: int = 1000, seed: int = 42) -> dict:
    a, b = np.asarray(a), np.asarray(b)
    idx = np.arange(len(a))
    lo, hi = bootstrap_ci(idx, n_boot=n_boot, seed=seed,
                          stat=lambda i: kappa(a[i.astype(int)], b[i.astype(int)], weights=weights))
    return {"kappa": kappa(a, b, weights=weights), "ci_low": lo, "ci_high": hi, "n": int(len(a))}


def binary_agreement(human, judge) -> dict:
    """For pass/fail or A/B style labels: percent agreement and Cohen's kappa with CI."""
    h, j = np.asarray(human), np.asarray(judge)
    return {"percent_agreement": float(np.mean(h == j)) if len(h) else float("nan"), **kappa_with_ci(h, j)}


def ordinal_agreement(human, judge, scale_max: int = 5) -> dict:
    """For 1–5 overall scores: Spearman, quadratic-weighted kappa, exact and within-1 agreement, MAE."""
    h, j = np.asarray(human, dtype=float), np.asarray(judge, dtype=float)
    if len(h) < 3:
        return {"n": int(len(h))}
    rho = stats.spearmanr(h, j)
    return {
        "n": int(len(h)),
        "spearman": float(rho.statistic), "spearman_p": float(rho.pvalue),
        "weighted_kappa": kappa_with_ci(h.astype(int), j.astype(int), weights="quadratic"),
        "exact_agreement": float(np.mean(h == j)),
        "within_one": float(np.mean(np.abs(h - j) <= 1)),
        "mae": float(np.mean(np.abs(h - j))),
        "human_mean": float(h.mean()), "judge_mean": float(j.mean()),
        "crosstab": pd.crosstab(pd.Series(h.astype(int), name="human"), pd.Series(j.astype(int), name="judge"))
        .reindex(index=range(1, scale_max + 1), columns=range(1, scale_max + 1), fill_value=0),
    }


def pairwise_agreement(human_pref, judge_pref) -> dict:
    """A/B/tie preferences. Reports agreement with and without ties (ties are the noisy class)."""
    h, j = np.asarray(human_pref), np.asarray(judge_pref)
    decisive = (h != "tie") & (j != "tie")
    out = {"n": int(len(h)), "percent_agreement": float(np.mean(h == j)) if len(h) else float("nan"),
           **{f"kappa_{k}": v for k, v in kappa_with_ci(h, j).items()}}
    out["n_both_decisive"] = int(decisive.sum())
    out["agreement_when_both_decisive"] = float(np.mean(h[decisive] == j[decisive])) if decisive.any() else float("nan")
    return out


def length_bias(scores, lengths) -> dict:
    """Does the judge reward longer replies? Spearman between score and character length."""
    s, l = np.asarray(scores, dtype=float), np.asarray(lengths, dtype=float)
    if len(s) < 3 or np.all(s == s[0]) or np.all(l == l[0]):
        return {"spearman": float("nan"), "p": float("nan"), "n": int(len(s))}
    r = stats.spearmanr(s, l)
    return {"spearman": float(r.statistic), "p": float(r.pvalue), "n": int(len(s))}
