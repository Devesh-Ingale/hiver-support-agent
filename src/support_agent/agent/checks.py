"""Deterministic post-processing: hard escalation rules and tweet-validity checks.

Everything here is plain regex/Python on purpose. It is the part of the system that must be
inspectable, unit-tested and easy to change live; the LLM's judgment is layered *under* it, never over it.
Rule patterns live in the versioned policy file, not in code, so the policy can be frozen before labelling.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..data.clean import CUSTOMER_MENTION_RE, SIGNOFF_RE, URL_RE, content_tokens
from .schemas import AgentOutput

# --- validity checks on a draft (independent of the policy file) ---------------------------------------

PROMISE_RE = re.compile(
    r"\b(will be (?:fixed|resolved|refunded|credited)|you(?:'ll| will) (?:get|receive|be) (?:a )?(?:full )?refund(?:ed)?|"
    r"we(?:'ll| will) (?:refund|credit|fix|resolve|reimburse)|guarantee|within \d+ (?:minutes|hours|days|weeks)|"
    r"by (?:tomorrow|tonight|end of (?:the )?(?:day|week))|(?:fixed|resolved) (?:soon|shortly|asap)|full refund|"
    r"the (?:team|engineers) (?:are|is) (?:fixing|working on) (?:it|this))\b",
    re.I,
)
SENSITIVE_RE = re.compile(
    r"\b(password|passcode|card number|credit card|debit card|cvv|social security|ssn|account number|"
    r"date of birth|home address|pin code|security code|bank details)\b",
    re.I,
)
DM_CONTEXT_RE = re.compile(r"\b(dm|direct message|private message|privately)\b", re.I)


@dataclass
class Violations:
    codes: list[str] = field(default_factory=list)

    def add(self, code: str) -> None:
        if code not in self.codes:
            self.codes.append(code)

    def __bool__(self) -> bool:
        return bool(self.codes)


def check_reply(reply: str, evidence_texts: list[str], max_chars: int = 280) -> Violations:
    """Return the validity violations of a draft. Empty means the draft is safe to send as a tweet."""
    v = Violations()
    text = reply or ""
    if not text.strip():
        v.add("empty_reply")
        return v
    if len(text) > max_chars:
        v.add("too_long")
    if CUSTOMER_MENTION_RE.search(text):
        v.add("anonymised_handle")
    allowed_urls = {u.rstrip(".,)") for e in evidence_texts for u in URL_RE.findall(e or "")}
    for url in URL_RE.findall(text):
        if url.rstrip(".,)") not in allowed_urls:
            v.add("url_not_in_evidence")
            break
    if SIGNOFF_RE.search(text):
        v.add("agent_signoff")
    if PROMISE_RE.search(text):
        v.add("promise")
    if SENSITIVE_RE.search(text) and not DM_CONTEXT_RE.search(text):
        v.add("sensitive_request_public")
    if "|||" in text or "[E" in text:
        v.add("formatting_artifact")   # copied a turn separator or an evidence tag out of the prompt
    return v


# --- hard escalation rules from the policy ------------------------------------------------------------


@dataclass
class HardRules:
    """reason_code -> compiled patterns; matched against the *customer's* message."""

    patterns: dict[str, list[re.Pattern[str]]]

    @classmethod
    def from_config(cls, config: dict[str, list[str]]) -> "HardRules":
        return cls({code: [re.compile(p, re.I) for p in pats] for code, pats in config.items()})

    def match(self, message: str) -> str | None:
        for code, pats in self.patterns.items():
            if any(p.search(message) for p in pats):
                return code
        return None


@dataclass
class PostProcessResult:
    output: AgentOutput
    model_decision: str
    forced_reason: str | None          # rule that overrode the model, if any
    violations: list[str]              # validity violations of the *model's* draft (before any override)
    retrieval_max_sim: float | None


def post_process(output: AgentOutput, message: str, *, lang: str, evidence_texts: list[str],
                 retrieval_max_sim: float | None, hard_rules: HardRules, min_similarity: float,
                 max_chars: int = 280, min_content_tokens: int = 3) -> PostProcessResult:
    """Apply the policy on top of the model's answer. The model can only make things *more* cautious.

    Order matters and is deliberate: content-independent routing first (language, empty message), then
    hard policy rules on the message, then the retrieval floor, then validity of the draft itself.
    """
    model_decision = output.decision
    violations = check_reply(output.reply, evidence_texts, max_chars=max_chars)
    forced: str | None = None

    if lang not in ("en", "unk") :
        forced = "non_english"
    elif len(content_tokens(message)) < min_content_tokens:
        forced = "no_actionable_content"
    else:
        hit = hard_rules.match(message)
        if hit:
            forced = hit
        elif retrieval_max_sim is not None and retrieval_max_sim < min_similarity:
            forced = "no_relevant_resolution"
        elif output.decision == "auto" and violations:
            forced = "draft_invalid"

    if forced:
        # a model that already escalated for a substantive reason keeps its reason; we only add caution —
        # except for facts about the message itself (language, no content), where the objective code wins
        keep_model_reason = (output.decision == "escalate" and output.reason_code not in ("none", "draft_invalid")
                             and forced not in ("non_english", "no_actionable_content"))
        output = output.model_copy(update={
            "decision": "escalate",
            "reason_code": output.reason_code if keep_model_reason else forced,
            "reason": output.reason if keep_model_reason else _describe(forced, violations),
        })
    # an escalation the model made with reason_code "none" is left as-is: metrics count it as "unspecified"

    return PostProcessResult(output=output, model_decision=model_decision, forced_reason=forced,
                             violations=violations.codes, retrieval_max_sim=retrieval_max_sim)


def _describe(code: str, violations: Violations) -> str:
    return {
        "non_english": "message is not in English; routed to a human",
        "no_actionable_content": "message has no actionable text (image-only or empty); a human should look",
        "no_relevant_resolution": "no sufficiently similar historical resolution found; routed to a human",
        "draft_invalid": f"draft failed validity checks ({', '.join(violations.codes)}); routed to a human",
    }.get(code, f"policy rule '{code}' matched the customer's message")
