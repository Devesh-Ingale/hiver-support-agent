"""Simple baseline: TF-IDF + logistic regression intents, nearest-neighbour verbatim reply, policy keyword rules.

- Intent: stratified k-fold cross-validation over the labelled test items, with the dev labels added to
  every training fold. Every test prediction is out-of-fold, so nothing is scored on its own label.
- Reply: the brand's real first reply to the most similar historical customer message (customer handles
  and agent sign-offs stripped). Strong, cheap, and exactly what "grounded in history" means at its simplest.
- Escalation: the policy's hard-rule regexes plus the non-content routing rules, applied to the tweet.
"""
from __future__ import annotations

import logging
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline

from ..agent.checks import HardRules
from ..data.clean import SIGNOFF_RE, clean_text, content_tokens
from ..retrieval.index import TfidfIndex
from .rows import make_row

log = logging.getLogger(__name__)


def cv_intent_predictions(texts: list[str], labels: list[str], extra_texts: list[str] | None = None,
                          extra_labels: list[str] | None = None, n_splits: int = 5, seed: int = 42,
                          brand: str | None = None) -> tuple[list[str], list[float]]:
    """Out-of-fold intent predictions (and max class probability) for every labelled text."""
    X = np.array([clean_text(t, brand=brand) for t in texts], dtype=object)
    y = np.array(labels, dtype=object)
    Xe = np.array([clean_text(t, brand=brand) for t in (extra_texts or [])], dtype=object)
    ye = np.array(extra_labels or [], dtype=object)
    min_count = min(Counter(labels).values())
    if min_count >= 2:
        splits = min(n_splits, min_count)
        splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
        folds = splitter.split(X, y)
    else:
        log.warning("a class has a single example; falling back to unstratified %d-fold", n_splits)
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X)
    preds = np.empty(len(X), dtype=object)
    confs = np.zeros(len(X), dtype=float)
    for train_idx, test_idx in folds:
        clf = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, strip_accents="unicode"),
            LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced"),
        )
        clf.fit(np.concatenate([X[train_idx], Xe]), np.concatenate([y[train_idx], ye]))
        proba = clf.predict_proba(X[test_idx])
        preds[test_idx] = clf.classes_[proba.argmax(axis=1)]
        confs[test_idx] = proba.max(axis=1)
    return preds.tolist(), confs.tolist()


def rule_decision(text: str, lang: str | None, hard_rules: HardRules, min_content_tokens: int = 3) -> tuple[str, str, str]:
    """(decision, reason_code, reason) from the policy rules alone."""
    if lang not in (None, "en", "unk"):
        return "escalate", "non_english", "rule: message is not in English"
    if len(content_tokens(text)) < min_content_tokens:
        return "escalate", "no_actionable_content", "rule: no actionable text"
    hit = hard_rules.match(text)
    if hit:
        return "escalate", hit, f"rule: policy pattern '{hit}' matched"
    return "auto", "none", "rule: no escalation pattern matched"


def verbatim_reply(text: str, index: TfidfIndex, brand: str, exclude_ids: set[str] | None = None):
    """The brand's real first reply to the nearest historical message, cleaned for public sending."""
    results = index.search(clean_text(text, brand=brand), k=1, exclude_ids=exclude_ids, dedup=False)
    if not results:
        return "", None, None
    top = results[0]
    reply = SIGNOFF_RE.sub("", top.first_reply).strip()
    return reply, top, top.similarity


def simple_rows(items: list[dict], intents: list[str], confidences: list[float], index: TfidfIndex,
                hard_rules: HardRules, brand: str) -> list[dict]:
    rows = []
    for item, intent, conf in zip(items, intents, confidences):
        text = item["root_text"]
        reply, top, sim = verbatim_reply(text, index, brand, exclude_ids={str(item.get("root_id", ""))})
        decision, code, reason = rule_decision(text, item.get("lang"), hard_rules)
        evidence = [top.to_dict()] if top else []
        rows.append(make_row(
            item, system="simple", provider="rule", model="tfidf-logreg+knn", intent=intent, reply=reply,
            decision=decision, reason_code=code, reason=reason, evidence=evidence,
            evidence_texts=[top.brand_turns] if top else [], retrieval_max_sim=sim, intent_confidence=conf,
        ))
    return rows


def human_reference_rows(items: list[dict]) -> list[dict]:
    """The brand's actual first reply to each golden tweet — the human reference level for the judge."""
    rows = []
    for item in items:
        real = item.get("first_brand_reply") or ""
        if not real:
            continue
        reply = SIGNOFF_RE.sub("", clean_text(real)).replace("<url>", "").strip()
        rows.append(make_row(item, system="human_ref", provider="human", model="brand agent", intent=None, reply=reply,
                             decision=None, reason_code=None, reason="the brand's real first reply",
                             evidence_texts=[real]))
    return rows
