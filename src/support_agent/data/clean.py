"""Text normalisation shared by retrieval, sampling and the validity checks.

Tweets in this dump are anonymised: customer handles appear as `@115712`-style numeric mentions,
brand handles keep their names (`@SpotifyCares`). Keep that distinction in mind below.
"""
from __future__ import annotations

import html
import re

MENTION_RE = re.compile(r"@\w+")
CUSTOMER_MENTION_RE = re.compile(r"@\d+\b")
URL_RE = re.compile(r"https?://\S+|\bwww\.\S+|\bt\.co/\S+", re.I)
WS_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")
PLACEHOLDERS = {"url", "brand"}


def clean_text(text: str, brand: str | None = None) -> str:
    """HTML-unescape, replace URLs with <url>, the brand handle with <brand>, drop other @mentions."""
    t = html.unescape(str(text))
    t = URL_RE.sub(" <url> ", t)
    if brand:
        t = re.sub(rf"@{re.escape(brand)}\b", " <brand> ", t, flags=re.I)
    t = MENTION_RE.sub(" ", t)
    return WS_RE.sub(" ", t).strip()


def content_tokens(text: str) -> list[str]:
    """Alphabetic tokens excluding placeholders — used for 'is there anything to classify here?'."""
    return [w for w in WORD_RE.findall(clean_text(text)) if w.lower() not in PLACEHOLDERS]


def normalize_for_dedup(text: str) -> str:
    """Aggressive normalisation so near-identical tweets collapse (outage bursts, copy-paste templates)."""
    t = clean_text(text).lower()
    t = re.sub(r"<url>|<brand>", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return WS_RE.sub(" ", t).strip()


def detect_lang(text: str, min_tokens: int = 3) -> str:
    """ISO-639-1 code, or 'unk' when there is too little text to decide."""
    tokens = content_tokens(text)
    if len(tokens) < min_tokens:
        return "unk"
    from langdetect import DetectorFactory, LangDetectException, detect  # lazy: slow import

    DetectorFactory.seed = 0
    try:
        return detect(" ".join(tokens))
    except LangDetectException:
        return "unk"
