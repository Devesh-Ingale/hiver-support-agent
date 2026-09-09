"""Thread reconstruction for the Kaggle 'Customer Support on Twitter' dump (twcs.csv).

Columns in the dump: tweet_id, author_id, inbound, created_at, text, response_tweet_id
(comma-separated ids of replies *to* this tweet), in_response_to_tweet_id (the parent, NaN for roots).
We only follow parent links; a tweet whose parent is referenced but absent from the dump is an
"orphan root" — kept, flagged and counted, because its text can be a mid-conversation fragment.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CREATED_AT_FMT = "%a %b %d %H:%M:%S %z %Y"
MAX_DEPTH = 500
TURN_SEP = " ||| "
MENTION_RE = re.compile(r"@(\w+)")
CUSTOMER_MENTION_RE = re.compile(r"@\d+\b")


def load_twcs(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read the dump with stable dtypes; parses timestamps and normalises the parent id column."""
    df = pd.read_csv(
        path,
        nrows=nrows,
        dtype={
            "tweet_id": "int64",
            "author_id": "string",
            "inbound": "string",
            "created_at": "string",
            "text": "string",
            "response_tweet_id": "string",
            "in_response_to_tweet_id": "float64",
        },
    )
    df = df.drop_duplicates("tweet_id").reset_index(drop=True)
    df["inbound"] = df["inbound"].str.strip().str.lower().eq("true").fillna(False).astype(bool)
    df["parent_id"] = df["in_response_to_tweet_id"].round().astype("Int64")
    df["created_at"] = pd.to_datetime(df["created_at"], format=CREATED_AT_FMT, errors="coerce", utc=True)
    df["text"] = df["text"].fillna("").astype(str)
    df["author_id"] = df["author_id"].fillna("").astype(str)
    return df.drop(columns=["in_response_to_tweet_id"])


def assign_roots(df: pd.DataFrame) -> pd.DataFrame:
    """Add root_id, depth, is_root and orphan_root by following parent links that exist in the dump."""
    n = len(df)
    ids = df["tweet_id"].to_numpy(dtype=np.int64)
    pos_of = pd.Series(np.arange(n), index=ids)
    parent_filled = df["parent_id"].fillna(-1).astype("int64").to_numpy()
    parent_pos = pos_of.reindex(parent_filled).to_numpy(dtype=float)
    parent_pos = np.where(np.isnan(parent_pos), -1, parent_pos).astype(np.int64)

    root_pos = np.arange(n)
    depth = np.zeros(n, dtype=np.int32)
    active = parent_pos[root_pos] >= 0
    for _ in range(MAX_DEPTH):
        if not active.any():
            break
        root_pos[active] = parent_pos[root_pos[active]]
        depth[active] += 1
        active = parent_pos[root_pos] >= 0
    else:
        log.warning("%d tweets still climbing after %d hops (cycle?) — left at current ancestor", active.sum(), MAX_DEPTH)

    df = df.copy()
    df["root_id"] = ids[root_pos]
    df["depth"] = depth
    df["is_root"] = root_pos == np.arange(n)
    df["orphan_root"] = df["parent_id"].notna().to_numpy() & (parent_pos == -1)
    return df


def _brand_from_mentions(text: str, brand_handles_lower: dict[str, str]) -> str:
    for handle in MENTION_RE.findall(text):
        if handle.lower() in brand_handles_lower:
            return brand_handles_lower[handle.lower()]
    return ""


def build_threads(df: pd.DataFrame) -> pd.DataFrame:
    """One row per thread root with the customer's root tweet and everything the brand said in reply.

    Brand = author of the first outbound tweet in the thread; when the brand never replied, the
    brand handle @-mentioned in the root text (if it is a known brand account).
    """
    if "root_id" not in df.columns:
        df = assign_roots(df)
    df = df.sort_values(["root_id", "created_at", "tweet_id"], kind="mergesort")
    brand_handles = sorted(set(df.loc[~df["inbound"], "author_id"]) - {""})
    handles_lower = {h.lower(): h for h in brand_handles}

    roots = df[df["is_root"]].set_index("root_id", drop=False)
    per_thread = df.groupby("root_id", sort=False)
    outbound = df[~df["inbound"]].groupby("root_id", sort=False)

    threads = pd.DataFrame(index=roots.index)
    threads["root_id"] = roots["tweet_id"]
    threads["root_author"] = roots["author_id"]
    threads["root_inbound"] = roots["inbound"]
    threads["root_text"] = roots["text"]
    threads["root_created_at"] = roots["created_at"]
    threads["orphan_root"] = roots["orphan_root"]
    threads["n_turns"] = per_thread.size().reindex(roots.index).fillna(0).astype(int)
    threads["max_depth"] = per_thread["depth"].max().reindex(roots.index).fillna(0).astype(int)
    threads["n_brand_turns"] = outbound.size().reindex(roots.index).fillna(0).astype(int)
    threads["n_customer_turns"] = threads["n_turns"] - threads["n_brand_turns"]
    threads["has_brand_reply"] = threads["n_brand_turns"] > 0
    threads["brand"] = outbound["author_id"].first().reindex(roots.index)
    threads["first_brand_reply"] = outbound["text"].first().reindex(roots.index)
    threads["first_brand_reply_at"] = outbound["created_at"].first().reindex(roots.index)
    threads["brand_turns"] = outbound["text"].agg(TURN_SEP.join).reindex(roots.index)

    no_reply = threads["brand"].isna()
    threads.loc[no_reply, "brand"] = [
        _brand_from_mentions(t, handles_lower) for t in threads.loc[no_reply, "root_text"]
    ]
    threads["brand"] = threads["brand"].fillna("").astype(str)
    threads["first_brand_reply"] = threads["first_brand_reply"].fillna("").astype(str)
    threads["brand_turns"] = threads["brand_turns"].fillna("").astype(str)
    # only meaningful for customer roots: "@SpotifyCares @123 same here" is a reply to another customer
    threads["mentions_other_customer"] = (
        threads["root_inbound"] & threads["root_text"].str.contains(CUSTOMER_MENTION_RE, regex=True)
    ).astype(bool)
    lag = (threads["first_brand_reply_at"] - threads["root_created_at"]).dt.total_seconds() / 60
    threads["first_reply_lag_min"] = lag
    return threads.reset_index(drop=True)


def build_and_save(csv_path: Path, out_path: Path, nrows: int | None = None) -> pd.DataFrame:
    log.info("loading %s", csv_path)
    df = load_twcs(csv_path, nrows=nrows)
    log.info("%d tweets; assigning roots", len(df))
    df = assign_roots(df)
    log.info("%d threads; building thread table", df["is_root"].sum())
    threads = build_threads(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    threads.to_parquet(out_path, index=False)
    log.info("wrote %s (%d threads)", out_path, len(threads))
    return threads
