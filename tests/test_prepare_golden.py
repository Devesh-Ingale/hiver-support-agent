from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from support_agent.data.prepare import near_duplicate_clusters, prepare_brand, substantive_flag
from support_agent.eval.golden import SampleConfig, candidate_records, sample_golden, sampling_note


def make_threads(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Synthetic thread table: 40 days of Spotify-ish traffic — templated complaints with wording drift,
    exact copy-paste duplicates (outage bursts), near-empty and non-English tweets, and unique oddities."""
    rng = np.random.default_rng(seed)
    templates = [
        "@SpotifyCares my app crashes when I open a playlist", "@SpotifyCares songs keep skipping in offline mode",
        "@SpotifyCares charged twice for premium this month, refund please", "@SpotifyCares cant log in, password reset not working",
        "@SpotifyCares the new update is ugly and slow", "@SpotifyCares podcast episodes wont download",
        "@SpotifyCares shuffle is not random at all", "@SpotifyCares family plan invite link broken",
    ]
    tails = ["on my iphone since yesterday", "after the latest update on android", "for the third time this week",
             "and support chat is not answering", "even after reinstalling twice", "on desktop and mobile both",
             "please fix this asap", "what is going on with your servers", "using the web player at work",
             "while driving which is dangerous", "on my samsung tv app", "since I upgraded to premium"]
    rows = []
    start = pd.Timestamp("2017-10-20", tz="UTC")
    for i in range(n):
        day = int(rng.integers(0, 40))
        created = start + pd.Timedelta(days=day, hours=int(rng.integers(0, 24)))
        kind = rng.random()
        if kind < 0.60:   # templated complaint with wording drift -> distinct items
            text = templates[int(rng.integers(0, len(templates)))] + " " + " ".join(
                rng.choice(tails, size=int(rng.integers(1, 3)), replace=False))
        elif kind < 0.75:  # exact copy-paste template -> near-duplicate cluster
            text = templates[int(rng.integers(0, len(templates)))]
        elif kind < 0.82:
            text = "@SpotifyCares 😩😩"                                   # near-empty
        elif kind < 0.90:
            text = "@SpotifyCares la aplicación se cierra cuando abro una lista de reproducción"   # non-English
        else:
            text = f"@SpotifyCares issue number {i} with something quite specific and unusual happening"
        replied = rng.random() < 0.85
        substantive = rng.random() < 0.5
        turns = ("@1 Sorry! Try a clean reinstall of the app /JR" if substantive else "@1 Please DM us your account email /AB") if replied else ""
        rows.append({
            "root_id": 1000 + i, "root_author": str(500000 + i), "root_inbound": True, "root_text": text,
            "root_created_at": created, "orphan_root": rng.random() < 0.05, "brand": "SpotifyCares",
            "has_brand_reply": replied, "n_turns": 2 if replied else 1, "n_brand_turns": int(replied),
            "n_customer_turns": 1, "first_brand_reply": turns, "brand_turns": turns, "max_depth": int(replied),
            "mentions_other_customer": False, "first_reply_lag_min": 12.0 if replied else np.nan,
        })
    # a brand-initiated root and another brand's thread must be ignored
    rows.append({**rows[0], "root_id": 1, "root_inbound": False, "root_text": "@123 Hey! all sorted?", "orphan_root": False})
    rows.append({**rows[1], "root_id": 2, "brand": "AppleSupport", "orphan_root": False})
    return pd.DataFrame(rows)


def test_near_duplicate_clusters_are_transitive_and_strict():
    labels = near_duplicate_clusters([
        "spotify is down again", "spotify is down again!!", "Spotify is down again...",
        "my playlist disappeared", "totally different complaint about podcasts",
    ], threshold=0.9)
    assert labels[0] == labels[1] == labels[2]
    assert len({labels[3], labels[4], labels[0]}) == 3
    assert near_duplicate_clusters([]).size == 0


def test_substantive_flag_uses_any_turn():
    assert substantive_flag("@1 Sorry to hear that! ||| @1 Try logging out and back in") is True
    assert substantive_flag("@1 Sorry! Please DM us your account email") is False


@pytest.fixture()
def prepared():
    return prepare_brand(make_threads(), "SpotifyCares", golden_window_days=14, tail_buffer_days=2)


def test_prepare_brand_time_split_and_columns(prepared):
    corpus, pool, stats = prepared
    cutoff = pd.Timestamp(stats.cutoff)
    assert (corpus["created_at"] < cutoff).all()
    assert (pool["root_created_at"] >= cutoff).all() and (pool["root_created_at"] <= pd.Timestamp(stats.tail_end)).all()
    assert set(corpus["doc_id"]).isdisjoint(set(pool["root_id"].astype(str)))
    assert {"doc_id", "message", "brand_turns", "first_reply", "substantive", "created_at"} <= set(corpus.columns)
    assert corpus["substantive"].dtype == bool and 0 < corpus["substantive"].mean() < 1
    assert {"lang", "near_dup_cluster", "is_cluster_representative", "n_content_tokens"} <= set(pool.columns)
    assert stats.brand_initiated_roots == 1 and stats.threads_total == 601
    assert stats.pool_near_dup_collapsed > 0 and stats.pool_english > 0
    assert (pool["message"].str.contains("<brand>")).all()   # brand handle normalised, customer text kept


def test_sample_golden_parts_and_disjointness(prepared):
    _, pool, split_stats = prepared
    cfg = SampleConfig(n_random=30, n_targeted=20, n_dev=10, spare_frac=0.2, non_english_quota=2, near_empty_quota=2,
                       n_clusters=4, cluster_floor=2, seed=1)
    sample, stats = sample_golden(pool, cfg)
    counts = sample["part"].value_counts()
    assert counts["A"] == 30 and counts["B"] == 20 and counts["dev"] == 10 and counts["spare"] == 12
    assert sample["item_id"].is_unique and sample["root_id"].is_unique
    assert sample["item_id"].iloc[0].startswith("G") and len(sample["item_id"].iloc[0]) == 5
    reasons = sample.groupby("part")["sampling_reason"].unique()
    assert set(reasons["A"]) <= {"uniform_random", "quota_near_empty", "quota_non_english"}
    assert "topic_cluster_floor" in reasons["B"]
    assert (sample[sample["sampling_reason"] == "quota_non_english"]["lang"] != "en").all()
    assert sample[sample["part"] != "A"]["lang"].isin(["en", "unk"]).all()   # non-English lives only in Part A quotas
    assert not sample["orphan_root"].any() and sample["is_cluster_representative"].all()
    assert stats.excluded["near_duplicate_of_kept_item"] > 0 and stats.eligible > 0
    # the order is shuffled, not chronological
    assert not sample["root_created_at"].is_monotonic_increasing


def test_candidate_records_and_note(prepared):
    _, pool, split_stats = prepared
    cfg = SampleConfig(n_random=10, n_targeted=6, n_dev=4, spare_frac=0.0, non_english_quota=1, near_empty_quota=1, n_clusters=3, seed=2)
    sample, stats = sample_golden(pool, cfg)
    records = candidate_records(sample)
    assert len(records) == len(sample)
    r = records[0]
    assert isinstance(r["root_id"], int) and isinstance(r["has_brand_reply"], bool) and "T" in r["root_created_at"]
    assert r["part"] in {"A", "B", "dev"} and "first_brand_reply" in r
    note = sampling_note(stats, split_stats.to_dict(), cfg)
    assert "Part A" in note and "Part B" in note and "Excluded before sampling" in note and "κ" in note
