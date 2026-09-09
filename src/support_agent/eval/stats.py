"""Uncertainty helpers: bootstrap confidence intervals and paired comparisons.

Every number in the report carries a 95 % interval from these functions; 'main beats baseline' is only
claimed when the paired interval excludes zero.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(values, stat=np.mean, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of `stat` over the items in `values`."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = np.array([stat(arr[i]) for i in idx])
    return (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def mean_with_ci(values, **kw) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    lo, hi = bootstrap_ci(arr, **kw)
    return {"value": float(arr.mean()) if arr.size else float("nan"), "ci_low": lo, "ci_high": hi, "n": int(arr.size)}


def paired_bootstrap_diff(a, b, n_boot: int = 1000, seed: int = 42) -> dict[str, float]:
    """CI of mean(a - b) over items, resampling items (not a and b independently)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired comparison needs equal-length arrays")
    diff = a - b
    lo, hi = bootstrap_ci(diff, n_boot=n_boot, seed=seed)
    return {"diff": float(diff.mean()), "ci_low": lo, "ci_high": hi, "n": int(diff.size),
            "significant": bool(lo > 0 or hi < 0)}


def mcnemar(a_correct, b_correct) -> dict[str, float]:
    """Exact McNemar test on discordant pairs for two systems' per-item correctness."""
    a, b = np.asarray(a_correct, dtype=bool), np.asarray(b_correct, dtype=bool)
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    n = only_a + only_b
    p = 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(only_a, only_b), n, 0.5)))
    return {"a_only_correct": only_a, "b_only_correct": only_b, "p_value": p}
