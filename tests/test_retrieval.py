from __future__ import annotations

import pandas as pd
import pytest

from support_agent.retrieval.index import TfidfIndex


def corpus() -> pd.DataFrame:
    rows = [
        # d1/d2: same resolution text, different customer mention, sign-off and punctuation -> must dedup
        ("d1", "app crashes when I open a playlist on iphone", "@1 Sorry! Try a clean reinstall: https://t.co/x /JR", True),
        ("d2", "app keeps crashing every time i open my playlists", "@2 Sorry, try a clean reinstall https://t.co/y /AB", True),
        ("d3", "charged twice for premium this month", "@3 Sorry! Please DM us your account email and we'll look /JR", False),
        ("d4", "cant log in to my account password not working", "@4 DM us your account email so we can check /CD", False),
        ("d5", "songs skip and stop playing offline mode", "@5 Toggle offline mode off and on, then restart the app /EF", True),
        # d6: lexically closest to the android query but pure boilerplate
        ("d6", "app crashes when I open a playlist on android", "@6 Sorry! Please DM us the details /GH", False),
    ]
    return pd.DataFrame(rows, columns=["doc_id", "message", "brand_turns", "substantive"]).assign(
        first_reply=lambda d: d["brand_turns"], created_at="2017-11-01"
    )


@pytest.fixture()
def index() -> TfidfIndex:
    return TfidfIndex(min_df=1).fit(corpus())


def test_search_returns_most_similar_first(index):
    res = index.search("my app crashes whenever i open a playlist", k=3)
    assert res[0].doc_id in {"d1", "d2", "d6"}
    assert res[0].similarity >= res[-1].similarity
    assert all(0 <= r.similarity <= 1.0001 for r in res)


def test_dedup_collapses_near_identical_brand_replies(index):
    res = index.search("app crashing when opening playlist", k=6)
    ids = {r.doc_id for r in res}
    assert len(ids & {"d1", "d2"}) == 1          # one resolution, not the same text twice
    assert len(index.search("app crashing when opening playlist", k=6, dedup=False)) > len(res)


def test_boilerplate_penalty_flips_only_close_calls():
    idx = TfidfIndex(min_df=1, boilerplate_penalty=0.3).fit(corpus())
    query = "app crashes when I open a playlist on android"
    res = idx.search(query, k=3)
    raw_best = max(res, key=lambda r: r.similarity)
    assert raw_best.doc_id == "d6"                # raw similarity is reported untouched...
    assert res[0].doc_id == "d1" and res[0].substantive   # ...but the near-tie is won by the substantive thread
    assert res[0].similarity < raw_best.similarity and res[0].score > raw_best.score
    # a boilerplate thread that is far ahead on similarity still ranks first
    far = idx.search("charged twice for premium this month", k=2)
    assert far[0].doc_id == "d3" and far[0].substantive is False


def test_no_penalty_index_ranks_by_raw_similarity():
    idx = TfidfIndex(min_df=1, boilerplate_penalty=0.0).fit(corpus())
    res = idx.search("app crashes when I open a playlist on android", k=1)
    assert res[0].doc_id == "d6" and res[0].score == pytest.approx(res[0].similarity)


def test_customer_handles_and_signoffs_are_stripped_from_evidence(index):
    res = index.search("charged twice premium", k=1)
    assert res[0].doc_id == "d3" and "@3" not in res[0].brand_turns and res[0].first_reply.startswith("Sorry!")
    assert "/JR" not in res[0].brand_turns and res[0].brand_turns.endswith("we'll look")
    res = index.search("app crashes when I open a playlist on iphone", k=1)
    assert "/JR" not in res[0].first_reply and "https://t.co/x" in res[0].first_reply   # links stay, initials go


def test_exclude_and_max_similarity(index):
    res = index.search("charged twice for premium and now i cant log in to my account", k=3, exclude_ids={"d3"})
    assert res and "d3" not in {r.doc_id for r in res}
    assert res[0].doc_id == "d4"
    assert TfidfIndex.max_similarity(res) == pytest.approx(max(r.similarity for r in res))
    assert TfidfIndex.max_similarity([]) is None
    assert index.search("zzzz qqqq", k=3) == []   # nothing in common with the corpus -> no evidence


def test_save_load_roundtrip(index, tmp_path):
    path = index.save(tmp_path / "index.joblib")
    loaded = TfidfIndex.load(path)
    assert len(loaded) == 6
    a = [r.doc_id for r in index.search("offline songs skip", k=2)]
    b = [r.doc_id for r in loaded.search("offline songs skip", k=2)]
    assert a == b and a[0] == "d5"
