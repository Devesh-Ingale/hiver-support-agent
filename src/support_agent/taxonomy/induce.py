"""Taxonomy induction support: cluster pre-cutoff customer messages so a human can write ≤10 intents.

The clusters are an aid, not the taxonomy. The human reads top terms + examples per cluster, merges and
names them, writes definitions and edge rules into taxonomy.yaml, and freezes it before labelling.
Optionally a local LLM drafts names/definitions from the cluster sheet — a starting point the human edits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from ..llm.base import LLMClient


def cluster_messages(messages: list[str], k: int = 20, seed: int = 42, top_terms: int = 12, n_examples: int = 8) -> list[dict]:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True, stop_words="english",
                          strip_accents="unicode", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b")
    X = vec.fit_transform(messages)
    k = min(k, max(2, X.shape[0] // 20))
    km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(X)
    terms = np.array(vec.get_feature_names_out())
    clusters = []
    for c in range(k):
        members = np.where(km.labels_ == c)[0]
        if members.size == 0:
            continue
        centroid = km.cluster_centers_[c]
        top = terms[np.argsort(-centroid)[:top_terms]].tolist()
        dists = np.asarray(X[members] @ centroid).ravel()
        closest = members[np.argsort(-dists)[:n_examples]]
        clusters.append({"cluster": c, "size": int(members.size), "share": float(members.size / X.shape[0]),
                         "top_terms": top, "examples": [messages[i] for i in closest]})
    clusters.sort(key=lambda d: -d["size"])
    return clusters


def clusters_markdown(clusters: list[dict], brand: str, n_messages: int) -> str:
    lines = [f"# {brand}: {len(clusters)} TF-IDF clusters over {n_messages} pre-cutoff customer messages", "",
             "Read each cluster, then merge/split into at most 10 intents (incl. other/unclear). Sizes are shares of the sample.", ""]
    for c in clusters:
        lines.append(f"## Cluster {c['cluster']} — {c['size']} msgs ({c['share']:.1%})")
        lines.append("Top terms: " + ", ".join(c["top_terms"]))
        for ex in c["examples"]:
            lines.append(f"- {ex}")
        lines.append("")
    return "\n".join(lines)


PROPOSAL_SYSTEM = """You help a support-analytics engineer design an intent taxonomy for {brand}'s incoming customer tweets.
You will be shown clusters of real customer messages with top terms and examples. Propose at most 9 intents plus one `other_unclear`.
Rules: intents must be mutually exclusive and cover the clusters; name them in snake_case; give a one-sentence definition, two short example paraphrases, and one edge rule that resolves a likely confusion with another intent; say which clusters map to each intent.
Prefer fewer, broader intents when clusters differ only by device or wording. Do not invent intents the clusters do not support.
Answer in Markdown, one section per intent."""


def propose_intents(llm: LLMClient, clusters_md: str, brand: str) -> str:
    resp = llm.complete(PROPOSAL_SYSTEM.format(brand=brand), clusters_md, schema=None, temperature=0.2, max_tokens=2500)
    return resp.text


def sample_for_induction(corpus: pd.DataFrame, n: int = 3000, seed: int = 42, min_tokens: int = 3) -> list[str]:
    eligible = corpus[corpus["n_content_tokens"] >= min_tokens]
    eligible = eligible.drop_duplicates("dedup_key")
    take = eligible.sample(n=min(n, len(eligible)), random_state=seed)
    return take["message"].str.replace("<brand>", "", regex=False).str.strip().tolist()
