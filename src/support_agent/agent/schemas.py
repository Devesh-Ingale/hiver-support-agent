"""The agent's output contract: one JSON object per customer message, schema-enforced at generation time."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

DECISIONS = ("auto", "escalate")

# Why a message must (hard), should (soft) or is routed to a human for non-content reasons (routing).
HARD_REASONS = ("account_or_pii", "payment_refund", "legal_safety_threat", "explicit_human_request")
SOFT_REASONS = ("anger_churn",)
ROUTING_REASONS = ("non_english", "no_actionable_content", "no_relevant_resolution", "draft_invalid")
REASON_CODES = HARD_REASONS + SOFT_REASONS + ROUTING_REASONS + ("none",)

NO_SECONDARY = "none"


class AgentOutput(BaseModel):
    """Validated form of what the model returns (plus what the post-processor changed)."""

    intent: str
    intent_secondary: str | None = None
    intent_confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    reply: str
    reason: str
    reason_code: str
    decision: str

    @field_validator("intent_secondary", mode="before")
    @classmethod
    def _none_secondary(cls, v: Any) -> Any:
        return None if v in (None, "", NO_SECONDARY) else v

    @field_validator("decision")
    @classmethod
    def _decision(cls, v: str) -> str:
        v = str(v).strip().lower()
        if v not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}, got {v!r}")
        return v

    @field_validator("reason_code")
    @classmethod
    def _reason(cls, v: str) -> str:
        v = str(v).strip().lower()
        if v not in REASON_CODES:
            raise ValueError(f"reason_code must be one of {REASON_CODES}, got {v!r}")
        return v

    @field_validator("reply", "reason", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> str:
        return "" if v is None else str(v).strip()


def output_schema(intents: list[str], evidence_ids: list[str] | None = None) -> dict[str, Any]:
    """JSON schema handed to the model. Flat on purpose (Gemini rejects deep nesting; grammar-constrained
    local decoding stays fast). Property order is generation order: evidence and reply come *before* the
    reason and decision so the model has committed to its draft when it judges whether to send it.
    """
    evidence_enum = {"type": "string"}
    if evidence_ids:
        evidence_enum = {"type": "string", "enum": list(evidence_ids)}
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": list(intents)},
            "intent_secondary": {"type": "string", "enum": list(intents) + [NO_SECONDARY]},
            "intent_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_ids": {"type": "array", "items": evidence_enum},
            "reply": {"type": "string"},
            "reason": {"type": "string"},
            "reason_code": {"type": "string", "enum": list(REASON_CODES)},
            "decision": {"type": "string", "enum": list(DECISIONS)},
        },
        "required": ["intent", "intent_secondary", "intent_confidence", "evidence_ids", "reply", "reason",
                     "reason_code", "decision"],
        "additionalProperties": False,
    }


def fallback_output(other_intent: str, why: str) -> AgentOutput:
    """What the pipeline emits when the model's output could not be parsed even after one repair attempt."""
    return AgentOutput(
        intent=other_intent, intent_secondary=None, intent_confidence=0.0, evidence_ids=[], reply="",
        reason=f"model output unusable ({why}); routed to a human", reason_code="draft_invalid", decision="escalate",
    )
