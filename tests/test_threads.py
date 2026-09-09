"""Thread reconstruction on a synthetic mini-dump that has every awkward shape we care about."""
from __future__ import annotations

import io

import pandas as pd
import pytest

from support_agent.data.threads import assign_roots, build_threads, load_twcs

# Shapes covered:
#  - thread 1: customer root (1) -> brand reply (2) -> customer (3) -> brand (4); root has two direct
#    replies (2 and 5) so response_tweet_id is comma-separated; 5 is another customer chiming in
#  - thread 10: customer root (10) with NO brand reply, brand identified from the @mention
#  - thread 20: orphan root (20) — its parent 19 is not in the dump — followed by a brand reply (21)
#  - thread 30: brand-initiated root (30) with a customer reply (31)
#  - tweet 40: customer root with no mention of any known brand -> brand ""
CSV = """tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id
1,115712,True,Tue Oct 31 22:10:47 +0000 2017,@SpotifyCares my app crashes when I open a playlist,"2,5",
2,SpotifyCares,False,Tue Oct 31 22:12:00 +0000 2017,@115712 Sorry about that! Which device and OS version are you on? /JR,3,1.0
3,115712,True,Tue Oct 31 22:15:00 +0000 2017,@SpotifyCares iPhone 8 on iOS 11,4,2.0
4,SpotifyCares,False,Tue Oct 31 22:20:00 +0000 2017,@115712 Thanks! Try a clean reinstall: https://t.co/abc and let us know /JR,,3.0
5,222222,True,Tue Oct 31 22:30:00 +0000 2017,@115712 @SpotifyCares same here!!,,1.0
10,333333,True,Wed Nov 01 09:00:00 +0000 2017,@SpotifyCares premium charged twice this month??,,
20,444444,True,Wed Nov 01 10:00:00 +0000 2017,@SpotifyCares still nothing from you,21,19.0
21,SpotifyCares,False,Wed Nov 01 10:05:00 +0000 2017,@444444 Please DM us your account email and we'll take a look /AB,,20.0
30,SpotifyCares,False,Wed Nov 01 11:00:00 +0000 2017,@555555 Hey! Following up on your earlier note - all sorted now? /CD,31,
31,555555,True,Wed Nov 01 11:30:00 +0000 2017,@SpotifyCares yes thank you!,,30.0
40,666666,True,Wed Nov 01 12:00:00 +0000 2017,anyone else's music app broken today,,
"""


@pytest.fixture()
def tweets() -> pd.DataFrame:
    return assign_roots(load_twcs(io.StringIO(CSV)))


def test_load_parses_types(tweets):
    assert tweets["inbound"].dtype == bool
    assert str(tweets["parent_id"].dtype) == "Int64"
    assert tweets["created_at"].dt.tz is not None
    assert tweets.loc[tweets.tweet_id == 1, "parent_id"].isna().all()


def test_roots_and_depth(tweets):
    by_id = tweets.set_index("tweet_id")
    assert by_id.loc[[1, 2, 3, 4, 5], "root_id"].tolist() == [1, 1, 1, 1, 1]
    assert by_id.loc[[1, 2, 3, 4, 5], "depth"].tolist() == [0, 1, 2, 3, 1]
    assert by_id.loc[21, "root_id"] == 20  # climbs to the orphan, not beyond
    assert bool(by_id.loc[20, "orphan_root"]) is True
    assert bool(by_id.loc[1, "orphan_root"]) is False
    assert by_id["is_root"].sum() == 5  # tweets 1, 10, 20, 30, 40


def test_build_threads_shapes(tweets):
    threads = build_threads(tweets).set_index("root_id")
    assert len(threads) == 5

    t1 = threads.loc[1]
    assert t1["brand"] == "SpotifyCares"
    assert t1["n_turns"] == 5 and t1["n_brand_turns"] == 2 and t1["n_customer_turns"] == 3
    assert t1["first_brand_reply"].startswith("@115712 Sorry about that!")
    assert t1["brand_turns"].split(" ||| ")[1].startswith("@115712 Thanks! Try a clean reinstall")
    assert t1["max_depth"] == 3
    assert t1["first_reply_lag_min"] == pytest.approx(73 / 60, abs=1e-6)  # 22:10:47 -> 22:12:00
    assert bool(t1["mentions_other_customer"]) is False

    t10 = threads.loc[10]
    assert t10["brand"] == "SpotifyCares" and not t10["has_brand_reply"] and t10["first_brand_reply"] == ""

    t20 = threads.loc[20]
    assert bool(t20["orphan_root"]) is True and t20["n_brand_turns"] == 1

    t30 = threads.loc[30]
    assert bool(t30["root_inbound"]) is False and t30["brand"] == "SpotifyCares"

    t40 = threads.loc[40]
    assert t40["brand"] == "" and t40["n_turns"] == 1


def test_customer_mention_flag(tweets):
    # tweet 5 is not a root, so the flag must come from the root text only
    threads = build_threads(tweets).set_index("root_id")
    assert not threads["mentions_other_customer"].any()
    extra = tweets.copy()
    extra.loc[extra.tweet_id == 40, "text"] = "@777777 @SpotifyCares same problem as this person"
    flagged = build_threads(extra).set_index("root_id")
    assert bool(flagged.loc[40, "mentions_other_customer"]) is True
    assert flagged.loc[40, "brand"] == "SpotifyCares"
