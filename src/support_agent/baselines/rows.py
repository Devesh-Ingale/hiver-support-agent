"""Shared row shape so baseline outputs are interchangeable with agent outputs in every metric."""
from __future__ import annotations

from ..agent.checks import check_reply


def make_row(item: dict, *, system: str, provider: str, model: str, intent: str | None, reply: str,
             decision: str | None, reason_code: str | None, reason: str, evidence: list[dict] | None = None,
             evidence_texts: list[str] | None = None, retrieval_max_sim: float | None = None,
             intent_confidence: float | None = None, extra: dict | None = None) -> dict:
    violations = check_reply(reply, evidence_texts or []).codes if reply is not None else []
    row = {
        "item_id": item["item_id"], "system": system, "provider": provider, "model": model, "prompt_version": None,
        "text": item.get("root_text", item.get("text")), "lang": item.get("lang"),
        "intent": intent, "intent_secondary": None, "intent_confidence": intent_confidence,
        "reply": reply, "decision": decision, "reason_code": reason_code, "reason": reason,
        "evidence_ids": [e["doc_id"] for e in (evidence or [])],
        "model_decision": decision, "model_reason_code": reason_code, "forced_reason": None,
        "violations": violations, "retrieval_max_sim": retrieval_max_sim, "evidence": evidence or [],
        "parse_failed": False, "repaired": False, "intent_invalid": False,
        "latency_ms": 0, "cached": None, "input_tokens": None, "output_tokens": None, "raw_text": None,
    }
    if extra:
        row.update(extra)
    return row
