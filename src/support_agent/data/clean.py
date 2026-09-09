"""Text normalisation shared by retrieval, sampling and the validity checks.

Tweets in this dump are anonymised: customer handles appear as `@115712`-style numeric mentions,
brand handles keep their names (`@SpotifyCares`). Keep that distinction in mind below.
"""
from __future__ import annotations

import html
import re
from collections.abc import Iterable

MENTION_RE = re.compile(r"@\w+")
CUSTOMER_MENTION_RE = re.compile(r"@\d+\b")
URL_RE = re.compile(r"https?://\S+|\bwww\.\S+|\bt\.co/\S+", re.I)
WS_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")
# "... let us know! /JR" — agents sign public replies with initials; brand convention, never content
SIGNOFF_RE = re.compile(r"(?:^|\s)[/^~-]\s?[A-Z]{2,3}[.!]?\s*$")
PLACEHOLDERS = {"url", "brand"}


def clean_text(text: str, brand: str | None = None, aliases: Iterable[str] | None = None) -> str:
    """HTML-unescape, replace URLs with <url>, the brand handle (and its anonymised aliases, e.g. the
    brand's main account rewritten as `@115888`) with <brand>, and drop every other @mention."""
    t = html.unescape(str(text))
    t = URL_RE.sub(" <url> ", t)
    if brand:
        t = re.sub(rf"@{re.escape(brand)}\b", " <brand> ", t, flags=re.I)
    for alias in aliases or ():
        t = re.sub(rf"@{re.escape(str(alias))}\b", " <brand> ", t)
    t = MENTION_RE.sub(" ", t)
    return WS_RE.sub(" ", t).strip()


def content_tokens(text: str) -> list[str]:
    """Alphabetic tokens excluding placeholders — used for 'is there anything to classify here?'."""
    return [w for w in WORD_RE.findall(clean_text(text)) if w.lower() not in PLACEHOLDERS]


def normalize_for_dedup(text: str) -> str:
    """Aggressive normalisation so near-identical tweets collapse (outage bursts, copy-paste templates)."""
    t = clean_text(SIGNOFF_RE.sub("", str(text))).lower()
    t = re.sub(r"<url>|<brand>", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return WS_RE.sub(" ", t).strip()


def detect_lang(text: str, min_tokens: int = 3, confident_tokens: int = 5, min_prob: float = 0.85) -> str:
    """'en', a confident non-English ISO-639-1 code, or 'unk'.

    langdetect is unreliable on short tweets ("songs skip in offline mode" can come back Danish), and a
    wrong non-English verdict would route an English customer to a human for no reason. So a non-English
    code is only returned when the text has at least `confident_tokens` words, English is not a plausible
    candidate, and the top language has probability >= `min_prob`. Everything ambiguous is 'unk', which the
    policy layer treats like English.
    """
    tokens = content_tokens(text)
    if len(tokens) < min_tokens:
        return "unk"
    from langdetect import DetectorFactory, LangDetectException, detect_langs  # lazy: slow import

    DetectorFactory.seed = 0
    try:
        candidates = detect_langs(" ".join(tokens))
    except LangDetectException:
        return "unk"
    if not candidates:
        return "unk"
    top = candidates[0]
    if top.lang == "en" or any(c.lang == "en" and c.prob >= 0.15 for c in candidates):
        return "en"
    if len(tokens) >= confident_tokens and top.prob >= min_prob:
        return top.lang
    return "unk"
