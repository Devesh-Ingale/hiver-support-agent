"""Profile the largest brands so the brand choice is data-driven (decision-log entry)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .clean import detect_lang, normalize_for_dedup
from .tags import primary_reply_type

REPLY_TYPES = ("dm_redirect", "steps", "link", "clarifying_question", "apology_only", "other")


def customer_initiated(threads: pd.DataFrame) -> pd.DataFrame:
    """Threads that start with a customer tweet whose parent is known to be absent (true roots)."""
    return threads[threads["root_inbound"] & ~threads["orphan_root"] & (threads["brand"] != "")]


def profile_brands(threads: pd.DataFrame, top_n: int = 15, lang_sample: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cand = customer_initiated(threads)
    volume = cand.groupby("brand").size().sort_values(ascending=False)
    rows = []
    for brand in volume.index[:top_n]:
        t = cand[cand["brand"] == brand]
        replied = t[t["has_brand_reply"]]
        types = replied["first_brand_reply"].map(primary_reply_type)
        type_share = types.value_counts(normalize=True).reindex(REPLY_TYPES).fillna(0.0)
        sample_idx = rng.choice(len(t), size=min(lang_sample, len(t)), replace=False)
        langs = t.iloc[sample_idx]["root_text"].map(detect_lang)
        rows.append(
            {
                "brand": brand,
                "customer_threads": len(t),
                "reply_rate": float(t["has_brand_reply"].mean()),
                "first_reply_dm_redirect": float(type_share["dm_redirect"]),
                "first_reply_steps": float(type_share["steps"]),
                "first_reply_link": float(type_share["link"]),
                "first_reply_question": float(type_share["clarifying_question"]),
                "first_reply_apology_only": float(type_share["apology_only"]),
                "substantive_first_reply": float(type_share["steps"] + type_share["link"]),
                "distinct_reply_ratio": float(replied["first_brand_reply"].map(normalize_for_dedup).nunique() / max(len(replied), 1)),
                "median_reply_chars": float(replied["first_brand_reply"].str.len().median()) if len(replied) else np.nan,
                "median_brand_turns": float(replied["n_brand_turns"].median()) if len(replied) else np.nan,
                "median_reply_lag_min": float(replied["first_reply_lag_min"].median()) if len(replied) else np.nan,
                "english_share": float((langs == "en").mean()),
                "unknown_lang_share": float((langs == "unk").mean()),
                "mentions_numeric_handle": float(t["mentions_numeric_handle"].mean()),
                "first_day": t["root_created_at"].min().date().isoformat(),
                "last_day": t["root_created_at"].max().date().isoformat(),
            }
        )
    return pd.DataFrame(rows)


def to_markdown(profile: pd.DataFrame) -> str:
    cols = [
        "brand", "customer_threads", "reply_rate", "first_reply_dm_redirect", "substantive_first_reply",
        "first_reply_question", "distinct_reply_ratio", "median_reply_chars", "english_share",
    ]
    view = profile[cols].copy()
    for c in cols[2:]:
        if c == "median_reply_chars":
            view[c] = view[c].map(lambda v: f"{v:.0f}")
        else:
            view[c] = view[c].map(lambda v: f"{v:.2f}")
    header = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in view.itertuples(index=False))
    return header + body + "\n"
