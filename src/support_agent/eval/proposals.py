"""Model-assisted labelling: two models propose a label per tweet; the human accepts or overrides.

Disclosure matters more than mechanics here (DECISIONS.md #21). Neither proposer is the system under
test, the human sees both proposals and any disagreement, and every accepted/overridden decision is
recorded so the report can quantify acceptance and anchoring risk.
"""
from __future__ import annotations

from ..llm.base import LLMClient
from ..taxonomy import Taxonomy
from .label_cli import LABELLER_REASONS

PROPOSAL_SYSTEM = """You are annotating customer tweets sent to {brand} on Twitter to build an evaluation set. For each tweet you assign:
- intent: the single best primary intent from the list; intent_secondary only when the tweet clearly raises a second, separate issue (else "none").
- escalate: true when the escalation checklist says a human must handle the tweet; false when a grounded public first reply is acceptable. Apply the checklist literally, not charitably.
- reason_code: the escalation reason when escalate is true ("none" otherwise). Prefer a hard code when one applies; use anger_churn only for strong anger or a credible cancel threat; non_english / no_actionable_content are for tweets that are not English or have nothing to act on.
- anger: 0 calm or neutral, 1 annoyed or sarcastic, 2 furious, insulting or threatening to leave.

# Intents
{intents}

# Escalation checklist
{policy}

Return only the JSON object."""

PROPOSAL_USER = "Tweet:\n{text}\n\nReturn the JSON object."


def proposal_schema(intents: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": list(intents)},
            "intent_secondary": {"type": "string", "enum": list(intents) + ["none"]},
            "escalate": {"type": "boolean"},
            "reason_code": {"type": "string", "enum": list(LABELLER_REASONS) + ["none"]},
            "anger": {"type": "integer", "minimum": 0, "maximum": 2},
        },
        "required": ["intent", "intent_secondary", "escalate", "reason_code", "anger"],
        "additionalProperties": False,
    }


def propose_label(llm: LLMClient, taxonomy: Taxonomy, text: str) -> dict:
    system = PROPOSAL_SYSTEM.format(brand=taxonomy.brand, intents=taxonomy.intents_markdown(), policy=taxonomy.policy_markdown())
    resp = llm.complete(system, PROPOSAL_USER.format(text=text), schema=proposal_schema(taxonomy.intent_ids), temperature=0.0, max_tokens=200)
    obj = resp.json() or {}
    intent = obj.get("intent") if obj.get("intent") in taxonomy.intent_ids else None
    secondary = obj.get("intent_secondary") if obj.get("intent_secondary") in taxonomy.intent_ids and obj.get("intent_secondary") != intent else None
    escalate = bool(obj.get("escalate")) if isinstance(obj.get("escalate"), bool) else None
    reason = obj.get("reason_code") if obj.get("reason_code") in LABELLER_REASONS else "none"
    if escalate and reason == "none":
        reason = None                       # escalated without a valid reason: leave it to the human
    if escalate is False:
        reason = "none"
    anger = obj.get("anger") if obj.get("anger") in (0, 1, 2) else 0
    return {"model": llm.model, "intent": intent, "intent_secondary": secondary, "escalate": escalate, "reason_code": reason,
            "anger": anger, "parse_failed": intent is None or escalate is None}


def merge_proposals(item_id: str, a: dict, b: dict) -> dict:
    """One row per item: both proposals plus agreement flags the labeller shows as a warning."""
    return {
        "item_id": item_id, "a": a, "b": b,
        "agree_intent": a["intent"] == b["intent"] and a["intent"] is not None,
        "agree_escalate": a["escalate"] == b["escalate"] and a["escalate"] is not None,
        "agree_reason": a["reason_code"] == b["reason_code"],
    }


def proposal_accuracy(proposals: dict[str, dict], labels: list[dict]) -> dict:
    """How often each proposer matches *blind* human labels — the anchoring-risk number for the report."""
    rows = [r for r in labels if not r.get("excluded") and r["item_id"] in proposals and not r.get("assisted")]
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    for key in ("a", "b"):
        p = [proposals[r["item_id"]][key] for r in rows]
        out[key] = {
            "model": p[0]["model"],
            "intent_accuracy": sum(x["intent"] == r["intent_primary"] for x, r in zip(p, rows)) / len(rows),
            "escalate_accuracy": sum(x["escalate"] == bool(r["escalate"]) for x, r in zip(p, rows)) / len(rows),
            "reason_accuracy": sum((x["reason_code"] or "none") == r["reason_code"] for x, r in zip(p, rows)) / len(rows),
        }
    agree = [proposals[r["item_id"]] for r in rows]
    out["models_agree_intent"] = sum(g["agree_intent"] for g in agree) / len(rows)
    out["models_agree_escalate"] = sum(g["agree_escalate"] for g in agree) / len(rows)
    both = [(g, r) for g, r in zip(agree, rows) if g["agree_intent"]]
    out["intent_accuracy_when_models_agree"] = (sum(g["a"]["intent"] == r["intent_primary"] for g, r in both) / len(both)) if both else None
    return out
