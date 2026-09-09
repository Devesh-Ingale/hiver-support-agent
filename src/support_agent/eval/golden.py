"""Golden-set sampling: Part A (uniform random), Part B (targeted), dev, plus spares; and the sampling note.

Every candidate carries `part` and `sampling_reason` so each table in the report can say which set it
uses, and every exclusion from the pool is counted so the sampling note writes itself.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

ESCALATION_KEYWORDS = re.compile(
    r"\b(?:refund|charged|charge|billing|bill|payment|paid|money|hacked|hack|stolen|unauthori[sz]ed|password|"
    r"lawyer|legal|sue|fraud|scam|cancel(?:l?ing|led)?|unsubscribe|again|third time|3rd time|still|"
    r"ridiculous|disgusting|worst|hate|never again|speak to|human|real person|manager)\b",
    re.I,
)


@dataclass
class SampleConfig:
    n_random: int = 120           # Part A
    n_targeted: int = 80          # Part B: cluster floors + escalation-keyword oversample
    n_dev: int = 50
    spare_frac: float = 0.3
    non_english_quota: int = 6    # kept in Part A so routing behaviour is tested
    near_empty_quota: int = 6
    min_content_tokens: int = 3
    n_clusters: int = 20
    cluster_floor: int = 2
    seed: int = 42


@dataclass
class SampleStats:
    pool_size: int
    excluded: dict[str, int] = field(default_factory=dict)
    eligible: int = 0
    counts: dict[str, int] = field(default_factory=dict)


def _take(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.iloc[0:0]
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    return df.iloc[np.sort(idx)]


def sample_golden(pool: pd.DataFrame, cfg: SampleConfig | None = None) -> tuple[pd.DataFrame, SampleStats]:
    """Draw test (Part A + Part B), dev and spare candidates from the annotated pool.

    Pool columns required: root_id, root_text, message, root_created_at, lang, n_content_tokens,
    near_dup_cluster, is_cluster_representative, has_brand_reply, first_brand_reply, orphan_root,
    mentions_other_customer.
    """
    cfg = cfg or SampleConfig()
    rng = np.random.default_rng(cfg.seed)
    stats = SampleStats(pool_size=len(pool))

    # --- exclusions (counted, not silent) ------------------------------------------------------------
    df = pool.copy()
    stats.excluded["orphan_root_fragment"] = int(df["orphan_root"].sum())
    df = df[~df["orphan_root"]]
    dup = ~df["is_cluster_representative"]
    stats.excluded["near_duplicate_of_kept_item"] = int(dup.sum())
    df = df[~dup]

    near_empty = df["n_content_tokens"] < cfg.min_content_tokens
    non_english = (df["lang"] != "en") & (df["lang"] != "unk") & ~near_empty
    core = df[~near_empty & ~non_english]
    stats.excluded["near_empty_beyond_quota"] = max(0, int(near_empty.sum()) - cfg.near_empty_quota)
    stats.excluded["non_english_beyond_quota"] = max(0, int(non_english.sum()) - cfg.non_english_quota)
    stats.eligible = int(len(core))

    parts: list[pd.DataFrame] = []

    def add(rows: pd.DataFrame, part: str, reason: str) -> None:
        if rows.empty:
            return
        rows = rows.copy()
        rows["part"] = part
        rows["sampling_reason"] = reason
        parts.append(rows)

    # --- Part A: uniform random core + small quotas of the awkward stuff ---------------------------------
    quota_ne = _take(df[near_empty], cfg.near_empty_quota, rng)
    quota_nen = _take(df[non_english], cfg.non_english_quota, rng)
    n_core_a = cfg.n_random - len(quota_ne) - len(quota_nen)
    part_a = _take(core, n_core_a, rng)
    add(part_a, "A", "uniform_random")
    add(quota_ne, "A", "quota_near_empty")
    add(quota_nen, "A", "quota_non_english")
    remaining = core.drop(part_a.index)

    # --- Part B: cluster floors then escalation-keyword oversample -------------------------------------
    n_clusters = min(cfg.n_clusters, max(2, len(remaining) // 10))
    if len(remaining) >= n_clusters:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
        X = vec.fit_transform(remaining["message"].tolist())
        km = KMeans(n_clusters=n_clusters, n_init=5, random_state=cfg.seed)
        remaining = remaining.assign(topic_cluster=km.fit_predict(X))
        floors = []
        for _, grp in remaining.groupby("topic_cluster"):
            floors.append(_take(grp, cfg.cluster_floor, rng))
        part_b_floor = pd.concat(floors) if floors else remaining.iloc[0:0]
    else:
        remaining = remaining.assign(topic_cluster=-1)
        part_b_floor = remaining.iloc[0:0]
    add(part_b_floor, "B", "topic_cluster_floor")
    remaining = remaining.drop(part_b_floor.index)

    n_keyword = cfg.n_targeted - len(part_b_floor)
    keyword_hits = remaining[remaining["root_text"].str.contains(ESCALATION_KEYWORDS, regex=True)]
    part_b_kw = _take(keyword_hits, n_keyword, rng)
    add(part_b_kw, "B", "escalation_keyword_oversample")
    remaining = remaining.drop(part_b_kw.index)
    if len(part_b_kw) < n_keyword:  # not enough keyword hits: top Part B up at random so n_targeted holds
        top_up = _take(remaining, n_keyword - len(part_b_kw), rng)
        add(top_up, "B", "random_top_up")
        remaining = remaining.drop(top_up.index)

    # --- dev and spares -------------------------------------------------------------------------------
    dev = _take(remaining, cfg.n_dev, rng)
    add(dev, "dev", "uniform_random")
    remaining = remaining.drop(dev.index)
    n_spare = int(round((cfg.n_random + cfg.n_targeted + cfg.n_dev) * cfg.spare_frac))
    spare = _take(remaining, n_spare, rng)
    add(spare, "spare", "uniform_random")

    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)   # labelling order is random
    out.insert(0, "item_id", [f"G{i:04d}" for i in range(1, len(out) + 1)])
    stats.counts = out["part"].value_counts().to_dict()
    stats.counts.update({f"reason:{k}": int(v) for k, v in out["sampling_reason"].value_counts().items()})
    return out, stats


def candidate_records(sample: pd.DataFrame) -> list[dict]:
    """The JSONL rows the labeller reads. The brand's real reply is carried but never shown in round 1."""
    cols = ["item_id", "root_id", "root_text", "message", "root_created_at", "part", "sampling_reason", "lang",
            "n_content_tokens", "near_dup_cluster_size", "has_brand_reply", "first_brand_reply", "brand_turns",
            "mentions_other_customer"]
    cols += [c for c in ("topic_cluster",) if c in sample.columns]
    rows = []
    for rec in sample[cols].to_dict(orient="records"):
        rec["root_id"] = int(rec["root_id"])
        rec["root_created_at"] = pd.Timestamp(rec["root_created_at"]).isoformat()
        for k, v in list(rec.items()):
            if isinstance(v, (np.integer,)):
                rec[k] = int(v)
            elif isinstance(v, (np.bool_,)):
                rec[k] = bool(v)
            elif isinstance(v, float) and np.isnan(v):
                rec[k] = None
        rows.append(rec)
    return rows


def sampling_note(stats: SampleStats, split_stats: dict, cfg: SampleConfig) -> str:
    lines = [
        "# Golden set — how it was sampled",
        "",
        f"Brand: **{split_stats['brand']}**. Golden window: {split_stats['cutoff'][:10]} to {split_stats['tail_end'][:10]} "
        f"(retrieval corpus and taxonomy induction use only threads before {split_stats['cutoff'][:10]}; "
        f"a {2}-day buffer before the dump's last day avoids truncated threads).",
        "",
        f"Population: {stats.pool_size} customer-initiated threads in the window.",
        "",
        "Excluded before sampling (counted, never silent):",
    ]
    lines += [f"- {k.replace('_', ' ')}: {v}" for k, v in stats.excluded.items()]
    lines += [
        "",
        f"Eligible core (English, ≥{cfg.min_content_tokens} content tokens, one representative per near-duplicate cluster): {stats.eligible}.",
        "",
        "Draws (seed {}):".format(cfg.seed),
        f"- **Part A, {cfg.n_random} items** — uniform random from the core, plus quotas of up to {cfg.near_empty_quota} near-empty/image-only "
        f"and {cfg.non_english_quota} non-English tweets so routing behaviour is exercised. Headline numbers use Part A.",
        f"- **Part B, {cfg.n_targeted} items** — {cfg.cluster_floor} per TF-IDF topic cluster ({cfg.n_clusters} clusters) for per-intent support, "
        "then an oversample of tweets matching escalation keywords (refund, charged, hacked, lawyer, cancel, 'again', 'third time', ...). "
        "Per-intent and escalation-recall tables use A+B and say so.",
        f"- **Dev, {cfg.n_dev} items** — uniform random, disjoint; the only set prompts and thresholds were tuned on.",
        f"- **Spares** — {int(cfg.spare_frac * 100)} % extra, used only to replace items excluded during labelling (each exclusion is logged with its reason).",
        "",
        "Resulting counts: " + ", ".join(f"{k}={v}" for k, v in stats.counts.items() if not k.startswith("reason:")) + ".",
        "",
        "Labelling protocol: taxonomy and escalation policy frozen and committed first; a 30-item pilot (merges only) preceded the main pass; "
        "items were shown in random order with no model output, no cluster id and no brand reply visible; 40 items were re-labelled blind "
        "on a later day to estimate label noise (Cohen's κ).",
    ]
    return "\n".join(lines) + "\n"
