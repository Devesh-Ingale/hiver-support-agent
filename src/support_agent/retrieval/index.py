"""TF-IDF retrieval over the brand's historical threads: customer message -> what the brand did about it.

Word 1–2-grams catch topic words ("premium", "offline", "charged"); character 3–5-grams absorb tweet
spelling ("cant", "wont", "loggin"). Cosine similarity on the L2-normalised concatenation. No embeddings,
no torch: transparent, seconds to build, and the reproduction path stays light (decision log #5).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ..data.clean import CUSTOMER_MENTION_RE, normalize_for_dedup, strip_signoffs
from ..data.threads import TURN_SEP

REQUIRED_COLUMNS = ("doc_id", "message", "brand_turns", "first_reply", "substantive", "created_at")


@dataclass
class Evidence:
    doc_id: str
    message: str            # the historical customer message (cleaned)
    brand_turns: str        # everything the brand said, turns separated by " ||| "
    first_reply: str
    substantive: bool       # at least one brand turn gives steps or a link (vs pure DM-redirect/apology)
    created_at: str
    similarity: float       # raw cosine similarity to the query
    score: float            # similarity after the boilerplate penalty; used for ranking only

    def to_dict(self) -> dict:
        return asdict(self)


class TfidfIndex:
    def __init__(self, word_ngrams: tuple[int, int] = (1, 2), char_ngrams: tuple[int, int] = (3, 5),
                 min_df: int = 2, boilerplate_penalty: float = 0.2):
        self.word_vec = TfidfVectorizer(ngram_range=word_ngrams, min_df=min_df, sublinear_tf=True,
                                        strip_accents="unicode", lowercase=True)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngrams, min_df=min_df,
                                        sublinear_tf=True, lowercase=True)
        self.boilerplate_penalty = boilerplate_penalty
        self.docs: pd.DataFrame | None = None
        self._matrix: sparse.csr_matrix | None = None

    # -- build ---------------------------------------------------------------------------------------
    def fit(self, docs: pd.DataFrame) -> "TfidfIndex":
        missing = set(REQUIRED_COLUMNS) - set(docs.columns)
        if missing:
            raise ValueError(f"index docs missing columns {sorted(missing)}")
        self.docs = docs.reset_index(drop=True).copy()
        self.docs["doc_id"] = self.docs["doc_id"].astype(str)
        self.docs["_dedup_key"] = self.docs["brand_turns"].map(normalize_for_dedup)
        self._matrix = self._transform(self.docs["message"].tolist())
        return self

    def _transform(self, texts: list[str], fit: bool = False) -> sparse.csr_matrix:
        if fit or not hasattr(self.word_vec, "vocabulary_"):
            w = self.word_vec.fit_transform(texts)
            c = self.char_vec.fit_transform(texts)
        else:
            w = self.word_vec.transform(texts)
            c = self.char_vec.transform(texts)
        return normalize(sparse.hstack([w, c]).tocsr())

    # -- query ---------------------------------------------------------------------------------------
    def search(self, query: str, k: int = 5, exclude_ids: set[str] | None = None,
               dedup: bool = True) -> list[Evidence]:
        """Top-k historical threads for a customer message.

        Ranking = cosine similarity, discounted by `boilerplate_penalty` for threads where the brand never
        said anything substantive, so "DM us" boilerplate only wins when nothing better is close. Threads
        whose brand replies are near-identical are collapsed to one so the k slots carry k different
        resolutions. `similarity` on each result is the raw, undiscounted value.
        """
        if self.docs is None or self._matrix is None:
            raise RuntimeError("index is not built; call fit() or load()")
        q = self._transform([query])
        sims = (self._matrix @ q.T).toarray().ravel()
        penalty = np.where(self.docs["substantive"].to_numpy(dtype=bool), 1.0, 1.0 - self.boilerplate_penalty)
        scores = sims * penalty
        order = np.argsort(-scores, kind="stable")
        results: list[Evidence] = []
        seen_keys: set[str] = set()
        exclude = exclude_ids or set()
        for i in order:
            if scores[i] <= 0:
                break
            row = self.docs.iloc[i]
            if row["doc_id"] in exclude:
                continue
            if dedup:
                if row["_dedup_key"] in seen_keys:
                    continue
                seen_keys.add(row["_dedup_key"])
            # evidence shows what the brand *did*, not who signed it: customer handles and agent initials go
            results.append(Evidence(
                doc_id=row["doc_id"], message=row["message"],
                brand_turns=strip_signoffs(CUSTOMER_MENTION_RE.sub("", row["brand_turns"])),
                first_reply=strip_signoffs(CUSTOMER_MENTION_RE.sub("", row["first_reply"])),
                substantive=bool(row["substantive"]), created_at=str(row["created_at"]),
                similarity=float(sims[i]), score=float(scores[i]),
            ))
            if len(results) >= k:
                break
        return results

    @staticmethod
    def max_similarity(results: list[Evidence]) -> float | None:
        return max((r.similarity for r in results), default=None)

    # -- persistence ---------------------------------------------------------------------------------
    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"word_vec": self.word_vec, "char_vec": self.char_vec, "docs": self.docs,
                     "matrix": self._matrix, "boilerplate_penalty": self.boilerplate_penalty}, path, compress=3)
        return path

    @classmethod
    def load(cls, path: Path) -> "TfidfIndex":
        state = joblib.load(path)
        idx = cls(boilerplate_penalty=state["boilerplate_penalty"])
        idx.word_vec, idx.char_vec = state["word_vec"], state["char_vec"]
        idx.docs, idx._matrix = state["docs"], state["matrix"]
        return idx

    def __len__(self) -> int:
        return 0 if self.docs is None else len(self.docs)
