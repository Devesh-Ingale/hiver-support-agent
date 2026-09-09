"""Regex tagging of what a brand reply *does* (its action type).

Used three ways: to profile brands (how much of their public replying is boilerplate), to
down-weight boilerplate in retrieval, and as the 'action-type agreement' diagnostic between a
drafted reply and the brand's real one. Heuristic by design; the report quantifies the mix.
"""
from __future__ import annotations

import re

from .clean import URL_RE

DM_RE = re.compile(
    r"\b(dm|dms|dm'd|direct message|private message|pm us|send (?:us )?a (?:direct |private )?message|"
    r"message us (?:privately|directly)|reach out (?:to us )?(?:via|in|by) dm|slide into our dm)\b",
    re.I,
)
STEPS_RE = re.compile(
    r"\b(try|tried|reinstall|re-install|restart|reboot|log ?out|sign ?out|log ?in|sign ?in|logging (?:out|in)|"
    r"clear(?:ing)? (?:the |your )?cache|update|updating|uninstall|install(?:ing)?|turn (?:it )?(?:off|on)|toggle|"
    r"reset|power cycle|force (?:close|quit)|clean install|offline mode|disable|enable|switch(?:ing)? (?:to|off|on)|"
    r"go to|head to|tap|select|open (?:the|your)|settings)\b",
    re.I,
)
QUESTION_RE = re.compile(
    r"\?|\b(which|what|when|where|how|could you|can you|let us know|tell us|do you|are you|have you|did you|"
    r"is (?:it|this|that)|does (?:it|this))\b",
    re.I,
)
APOLOGY_RE = re.compile(r"\b(sorry|apolog\w+|regret|that'?s not (?:good|great|ideal|right))\b", re.I)

# Priority order when several patterns match: the first is the reply's primary action.
PRIORITY = ("dm_redirect", "steps", "link", "clarifying_question", "apology_only", "other")


def tag_reply(text: str) -> set[str]:
    """All action tags present in a brand reply."""
    t = str(text)
    tags: set[str] = set()
    if DM_RE.search(t):
        tags.add("dm_redirect")
    if STEPS_RE.search(t):
        tags.add("steps")
    if URL_RE.search(t) or "<url>" in t:
        tags.add("link")
    if QUESTION_RE.search(t):
        tags.add("clarifying_question")
    if APOLOGY_RE.search(t):
        tags.add("apology")
    return tags


def primary_reply_type(text: str) -> str:
    tags = tag_reply(text)
    for label in PRIORITY[:4]:
        if label in tags:
            return label
    if tags == {"apology"}:
        return "apology_only"
    return "other"
