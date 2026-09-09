"""Brand slice -> cleaned customer roots -> time split into retrieval corpus and golden-candidate pool.

Time is the split key: everything the model may learn from (retrieval corpus, taxonomy induction) is
strictly before the cutoff; golden candidates come after it, with a buffer before the dump's tail where
threads are truncated (brand replies missing because collection stopped).
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

from .clean import CUSTOMER_MENTION_RE, clean_text, content_tokens, detect_lang, normalize_for_dedup
from .tags import primary_reply_type
from .threads import TURN_SEP, brand_alias_ids

log = logging.getLogger(__name__)


@dataclass
class SplitStats:
    brand: str
    brand_alias_ids: list[str]
    threads_total: int
    customer_roots: int
    brand_initiated_roots: int
    orphan_roots: int
    cutoff: str
    tail_end: str
    corpus_threads: int
    corpus_with_substantive_reply: int
    pool_candidates: int
    pool_english: int
    pool_near_dup_clusters: int
    pool_near_dup_collapsed: int

    def to_dict(self) -> dict:
        return asdict(self)


def near_duplicate_clusters(texts: list[str], threshold: float = 0.9) -> np.ndarray:
    """Cluster id per text; texts whose char-n-gram TF-IDF cosine exceeds `threshold` share a cluster.

    Texts are normalised first (mentions, URLs, punctuation, case, sign-offs removed) so "Spotify is
    DOWN!!!" and "spotify is down" are one item. Transitive (connected components), so an outage burst
    with small wording drift still collapses to one representative.
    """
    if not texts:
        return np.array([], dtype=int)
    normalised = [normalize_for_dedup(t) or "empty" for t in texts]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
    X = vec.fit_transform(normalised)
    sims = (X @ X.T).tocsr()
    sims.data[sims.data < threshold] = 0.0
    sims.eliminate_zeros()
    _, labels = connected_components(sims, directed=False)
    return labels


def substantive_flag(brand_turns: str) -> bool:
    return any(primary_reply_type(turn) in ("steps", "link") for turn in str(brand_turns).split(TURN_SEP) if turn)


def prepare_brand(threads_all: pd.DataFrame, brand: str, golden_window_days: int = 14, tail_buffer_days: int = 2,
                  near_dup_threshold: float = 0.9) -> tuple[pd.DataFrame, pd.DataFrame, SplitStats]:
    """Returns (corpus, pool, stats).

    corpus: one row per pre-cutoff customer-initiated thread the brand replied to (retrieval documents).
    pool:   every customer root in the golden window, cleaned and annotated (language, near-dup cluster).
    """
    t = threads_all[threads_all["brand"] == brand].copy()
    if t.empty:
        raise ValueError(f"no threads for brand {brand!r}")
    customer = t[t["root_inbound"]].copy()
    aliases = brand_alias_ids(customer)
    alias_re = re.compile(r"@(" + "|".join(map(re.escape, aliases)) + r")\b") if aliases else None
    customer["message"] = customer["root_text"].map(lambda s: clean_text(s, brand=brand, aliases=aliases))
    customer["n_content_tokens"] = customer["root_text"].map(lambda s: len(content_tokens(s)))
    customer["dedup_key"] = customer["root_text"].map(normalize_for_dedup)
    # a tweet mentions *another customer* only if it carries a numeric handle that is not a brand alias
    stripped = customer["root_text"].map(lambda s: alias_re.sub("", s) if alias_re else s)
    customer["mentions_other_customer"] = stripped.str.contains(CUSTOMER_MENTION_RE, regex=True).astype(bool)

    tail_end = customer["root_created_at"].max() - pd.Timedelta(days=tail_buffer_days)
    cutoff = tail_end - pd.Timedelta(days=golden_window_days)

    corpus_mask = (customer["root_created_at"] < cutoff) & customer["has_brand_reply"] & ~customer["orphan_root"]
    corpus = customer[corpus_mask].copy()
    corpus["doc_id"] = corpus["root_id"].astype(str)
    corpus["substantive"] = corpus["brand_turns"].map(substantive_flag)
    corpus["first_reply_type"] = corpus["first_brand_reply"].map(primary_reply_type)
    corpus = corpus[["doc_id", "message", "root_text", "brand_turns", "first_brand_reply", "first_reply_type",
                     "substantive", "n_brand_turns", "n_content_tokens", "dedup_key", "root_created_at"]]
    corpus = corpus.rename(columns={"first_brand_reply": "first_reply", "root_created_at": "created_at"})
    corpus = corpus.sort_values("created_at").reset_index(drop=True)

    pool_mask = (customer["root_created_at"] >= cutoff) & (customer["root_created_at"] <= tail_end)
    pool = customer[pool_mask].copy().sort_values("root_created_at").reset_index(drop=True)
    log.info("language detection on %d pool tweets", len(pool))
    pool["lang"] = pool["root_text"].map(detect_lang)
    pool["near_dup_cluster"] = near_duplicate_clusters(pool["message"].tolist(), threshold=near_dup_threshold)
    cluster_sizes = pool["near_dup_cluster"].map(pool["near_dup_cluster"].value_counts())
    pool["near_dup_cluster_size"] = cluster_sizes.astype(int)
    pool["is_cluster_representative"] = ~pool.duplicated("near_dup_cluster", keep="first")

    stats = SplitStats(
        brand=brand, brand_alias_ids=aliases, threads_total=int(len(t)), customer_roots=int(len(customer)),
        brand_initiated_roots=int((~t["root_inbound"]).sum()), orphan_roots=int(customer["orphan_root"].sum()),
        cutoff=cutoff.isoformat(), tail_end=tail_end.isoformat(),
        corpus_threads=int(len(corpus)), corpus_with_substantive_reply=int(corpus["substantive"].sum()),
        pool_candidates=int(len(pool)), pool_english=int((pool["lang"] == "en").sum()),
        pool_near_dup_clusters=int(pool["near_dup_cluster"].nunique()),
        pool_near_dup_collapsed=int(len(pool) - pool["near_dup_cluster"].nunique()),
    )
    return corpus, pool, stats
