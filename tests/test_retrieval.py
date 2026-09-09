from __future__ import annotations

import pandas as pd
import pytest

from support_agent.retrieval.index import TfidfIndex


def corpus() -> pd.DataFrame:
    rows = [
        ("d1", "app crashes when I open a playlist on iphone", "@1 Sorry! Try a clean reinstall: https://t.co/x /JR", True),
        ("d2", "app keeps crashing every time i open my playlists", "@2 Try a clean reinstall of the app: https://t.co/x /AB", True),
        ("d3", "charged twice for premium this month", "@3 Sorry! Please DM us your account email and we'll look /JR", False),
        ("d4", "cant log in to my account password not working", "@4 DM us your account email so we can check /CD", False),
        ("d5", "songs skip and stop playing offline mode", "@5 Toggle offline mode off and on, then restart the app /EF", True),
        ("d6", "playlist crashes the app on android", "@6 Sorry! Please DM us the details /GH", False),
    ]
    return pd.DataFrame(rows, columns=["doc_id", "message", "brand_turns", "substantive"]).assign(
        first_reply=lambda d: d["brand_turns"], created_at="2017-11-01"
    )


@pytest.fixture()
def index() -> TfidfIndex:
    return TfidfIndex(min_df=1).fit(corpus())


def test_search_returns_most_similar_first(index):
    res = index.search("my app crashes whenever i open a playlist", k=3)
    assert res[0].doc_id in {"d1", "d2"}
    assert res[0].similarity > res[-1].similarity
    assert all(0 <= r.similarity <= 1.0001 for r in res)


def test_dedup_collapses_near_identical_brand_replies(index):
    res = index.search("app crashing when opening playlist", k=6)
    keys = [r.brand_turns.lower().replace("/ab", "").replace("/jr", "") for r in res]
    # d1 and d2 carry the same resolution text apart from the sign-off -> only one of them may appear
    assert len({r.doc_id for r in res} & {"d1", "d2"}) == 1
    assert len(keys) == len(set(keys))


def test_boilerplate_penalty_prefers_substantive_when_close(index):
    # d6 (DM-redirect) is lexically closest to this query; the substantive d1/d2 must be ranked above it
    res = index.search("playlist crashes the app", k=3)
    assert res[0].substantive is True
    assert "d6" in {r.doc_id for r in res}
    raw_best = max(res, key=lambda r: r.similarity)
    assert raw_best.doc_id == "d6"  # raw similarity is reported untouched


def test_no_penalty_index_ranks_by_raw_similarity():
    idx = TfidfIndex(min_df=1, boilerplate_penalty=0.0).fit(corpus())
    res = idx.search("playlist crashes the app", k=1)
    assert res[0].doc_id == "d6" and res[0].score == pytest.approx(res[0].similarity)


def test_customer_handles_are_stripped_from_evidence(index):
    res = index.search("charged twice premium", k=1)
    assert res[0].doc_id == "d3" and "@3" not in res[0].brand_turns and res[0].first_reply.startswith("Sorry!")


def test_exclude_and_max_similarity(index):
    res = index.search("charged twice premium", k=2, exclude_ids={"d3"})
    assert "d3" not in {r.doc_id for r in res}
    assert TfidfIndex.max_similarity(res) == pytest.approx(res[0].similarity)
    assert TfidfIndex.max_similarity([]) is None


def test_save_load_roundtrip(index, tmp_path):
    path = index.save(tmp_path / "index.joblib")
    loaded = TfidfIndex.load(path)
    assert len(loaded) == 6
    a = [r.doc_id for r in index.search("offline songs skip", k=2)]
    b = [r.doc_id for r in loaded.search("offline songs skip", k=2)]
    assert a == b and a[0] == "d5"
