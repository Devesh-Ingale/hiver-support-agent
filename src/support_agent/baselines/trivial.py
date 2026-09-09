"""Trivial baselines: majority intent, a fixed template reply, and always-escalate / always-auto.

Each trivially wins one metric (always-escalate has perfect escalation recall; always-auto has 100 %
automation), which is exactly why the report needs a cost-weighted view.
"""
from __future__ import annotations

from collections import Counter

from .rows import make_row

TEMPLATE_REPLY = "Sorry to hear that! Please DM us the details and we'll take a look."


def majority_intent(labels: list[dict]) -> str:
    counts = Counter(r["intent_primary"] for r in labels if not r.get("excluded") and r.get("intent_primary"))
    if not counts:
        raise ValueError("no labels to compute a majority class from")
    return counts.most_common(1)[0][0]


def trivial_rows(items: list[dict], majority: str, always_escalate: bool) -> list[dict]:
    system = "trivial_escalate" if always_escalate else "trivial_auto"
    decision = "escalate" if always_escalate else "auto"
    reason = "baseline policy: escalate everything" if always_escalate else "baseline policy: auto-handle everything"
    return [
        make_row(item, system=system, provider="rule", model="trivial", intent=majority, reply=TEMPLATE_REPLY,
                 decision=decision, reason_code="none", reason=reason, intent_confidence=1.0)
        for item in items
    ]
