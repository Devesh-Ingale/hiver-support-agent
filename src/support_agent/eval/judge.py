"""LLM-as-judge for reply quality: binary checks + pass/fail (absolute) and A/B preference (pairwise).

Design choices, all deliberate (see DECISIONS.md #10 and the report):
- reference-free: the judge sees the tweet, the policy and the same historical evidence for every system,
  never the label and never which system wrote the reply;
- rationale is generated *before* the verdict fields (property order = generation order);
- pairwise runs in both orders; a flipped verdict is recorded as the judge's own noise;
- the judge is a different model family from the generator (Gemini vs local Qwen/Llama).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..llm.base import LLMClient
from ..retrieval.index import Evidence
from ..taxonomy import Taxonomy
from ..agent.prompts import render_evidence

JUDGE_PROMPT_VERSION = "j1"

ABSOLUTE_SYSTEM = """You are auditing draft public replies written by a customer-support agent for {brand} on Twitter. You are strict, consistent, and you judge only what is in front of you.

You will see: the customer's tweet; the historical EVIDENCE (how {brand} actually handled similar cases — the only source of truth for what {brand} does); the agent's draft reply; and the agent's decision (auto = send the reply now; escalate = a human handles it first) with its stated reason.

Answer each check literally:
- unsupported_content: true if the reply states any step, link, policy, fact or promise that the evidence does not support (being generic and safe is NOT unsupported; inventing specifics is).
- addresses_problem: true if a reasonable customer would feel the reply engages with the actual problem they raised (a relevant clarifying question or a relevant hand-off counts; a generic "sorry, DM us" for a question the evidence answers does not).
- unsafe_or_overpromising: true if the reply asks for sensitive data in public, promises a refund/fix/feature/timeline, gives risky advice, or is rude.
- tweet_valid: true if the reply is at most 280 characters, plain text, without @handles, agent initials/sign-offs, or URLs absent from the evidence.
- decision_reason_consistent: true if the stated reason genuinely follows from the tweet and matches the decision under this policy: {policy_summary}
- overall_1_5: 1 = harmful or useless, 2 = poor, 3 = acceptable with edits, 4 = good, 5 = as good as a strong human agent's first reply.
- sendable: true only if a {brand} support agent would send this reply with at most a light edit.

Write your rationale first (two or three sentences, citing the evidence ids you relied on), then the fields. Return only the JSON object."""

ABSOLUTE_USER = """# Customer tweet
{message}

# Evidence — how {brand} handled similar cases
{evidence}

# Agent's draft reply
{reply}

# Agent's decision
{decision} — reason: {reason}

Return the JSON object."""

PAIRWISE_SYSTEM = """You compare two candidate public replies to the same customer tweet for {brand} on Twitter and pick the one a strong {brand} support agent would rather send as the first response.

Judge on: does it engage with the customer's actual problem; is it grounded in how {brand} handles such cases (see EVIDENCE) rather than inventing specifics; is it safe (no sensitive data requested in public, no promises); is it a valid, friendly tweet (≤280 characters, no handles or sign-offs). Length is not a virtue: a shorter reply that does the job beats a longer one that pads. Prefer "tie" only when the two are genuinely equivalent in quality.

Write your rationale first, then `preferred` as "A", "B" or "tie". Return only the JSON object."""

PAIRWISE_USER = """# Customer tweet
{message}

# Evidence — how {brand} handled similar cases
{evidence}

# Reply A
{reply_a}

# Reply B
{reply_b}

Return the JSON object."""

ABSOLUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "unsupported_content": {"type": "boolean"},
        "addresses_problem": {"type": "boolean"},
        "unsafe_or_overpromising": {"type": "boolean"},
        "tweet_valid": {"type": "boolean"},
        "decision_reason_consistent": {"type": "boolean"},
        "overall_1_5": {"type": "integer", "minimum": 1, "maximum": 5},
        "sendable": {"type": "boolean"},
    },
    "required": ["rationale", "unsupported_content", "addresses_problem", "unsafe_or_overpromising", "tweet_valid",
                 "decision_reason_consistent", "overall_1_5", "sendable"],
    "additionalProperties": False,
}

PAIRWISE_SCHEMA = {
    "type": "object",
    "properties": {"rationale": {"type": "string"}, "preferred": {"type": "string", "enum": ["A", "B", "tie"]}},
    "required": ["rationale", "preferred"],
    "additionalProperties": False,
}

CHECK_FIELDS = ("unsupported_content", "addresses_problem", "unsafe_or_overpromising", "tweet_valid",
                "decision_reason_consistent")


def policy_summary(taxonomy: Taxonomy) -> str:
    hard = "; ".join(f"{r.code} ({r.description})" for r in taxonomy.reasons_by_tier("hard"))
    soft = "; ".join(f"{r.code} ({r.description})" for r in taxonomy.reasons_by_tier("soft"))
    return f"ALWAYS escalate for {hard}. Escalate on judgment for {soft}. Otherwise a grounded public reply may be sent."


def evidence_from_dicts(rows: list[dict]) -> list[Evidence]:
    return [Evidence(**{k: r[k] for k in Evidence.__dataclass_fields__}) for r in rows]


@dataclass
class Judge:
    llm: LLMClient
    taxonomy: Taxonomy
    temperature: float = 0.0
    seed: int = 42

    def _evidence_block(self, evidence: list[dict]) -> str:
        return render_evidence(evidence_from_dicts(evidence), self.taxonomy.brand)

    # -- absolute ------------------------------------------------------------------------------------
    def absolute(self, item: dict, row: dict, evidence: list[dict], paraphrase: bool = False) -> dict:
        system = ABSOLUTE_SYSTEM.format(brand=self.taxonomy.brand, policy_summary=policy_summary(self.taxonomy))
        if paraphrase:  # self-consistency probe: same content, different wording, must not change verdicts
            system = system.replace("You are strict, consistent, and you judge only what is in front of you.",
                                    "Be exact and consistent; base every check purely on the material shown.")
        user = ABSOLUTE_USER.format(
            message=item["root_text"], brand=self.taxonomy.brand, evidence=self._evidence_block(evidence),
            reply=row.get("reply") or "(empty reply)",
            decision=row.get("decision") or "n/a", reason=row.get("reason") or "n/a",
        )
        resp = self.llm.complete(system, user, schema=ABSOLUTE_SCHEMA, temperature=self.temperature, seed=self.seed,
                                 max_tokens=600)
        obj = resp.json() or {}
        verdict = {k: obj.get(k) for k in ("rationale", *CHECK_FIELDS, "overall_1_5", "sendable")}
        verdict["judge_parse_failed"] = any(verdict[k] is None for k in (*CHECK_FIELDS, "overall_1_5", "sendable"))
        return {
            "item_id": item["item_id"], "system": row.get("system"), "judge_model": self.llm.model,
            "judge_prompt_version": JUDGE_PROMPT_VERSION + ("-para" if paraphrase else ""),
            "reply": row.get("reply"), "reply_chars": len(row.get("reply") or ""), **verdict,
            "latency_ms": resp.latency_ms, "cached": resp.cached,
        }

    # -- pairwise ------------------------------------------------------------------------------------
    def pairwise(self, item: dict, row_a: dict, row_b: dict, evidence: list[dict]) -> dict:
        """One ordered comparison: `row_a` is shown as A. Callers run both orders."""
        system = PAIRWISE_SYSTEM.format(brand=self.taxonomy.brand)
        user = PAIRWISE_USER.format(message=item["root_text"], brand=self.taxonomy.brand,
                                    evidence=self._evidence_block(evidence),
                                    reply_a=row_a.get("reply") or "(empty reply)", reply_b=row_b.get("reply") or "(empty reply)")
        resp = self.llm.complete(system, user, schema=PAIRWISE_SCHEMA, temperature=self.temperature, seed=self.seed,
                                 max_tokens=400)
        obj = resp.json() or {}
        pref = obj.get("preferred") if obj.get("preferred") in ("A", "B", "tie") else None
        return {
            "item_id": item["item_id"], "system_a": row_a.get("system"), "system_b": row_b.get("system"),
            "judge_model": self.llm.model, "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "preferred": pref, "preferred_system": {"A": row_a.get("system"), "B": row_b.get("system"), "tie": "tie"}.get(pref),
            "rationale": obj.get("rationale"), "judge_parse_failed": pref is None,
            "latency_ms": resp.latency_ms, "cached": resp.cached,
        }


def pairwise_both_orders(judge: Judge, item: dict, row_x: dict, row_y: dict, evidence: list[dict]) -> dict:
    """Run X-vs-Y and Y-vs-X; the combined verdict is the system preferred in both orders, else 'tie'/'flip'."""
    first = judge.pairwise(item, row_x, row_y, evidence)
    second = judge.pairwise(item, row_y, row_x, evidence)
    p1, p2 = first["preferred_system"], second["preferred_system"]
    if p1 == p2:
        combined = p1                       # same system (or tie) in both orders
    elif "tie" in (p1, p2):
        combined = p1 if p2 == "tie" else p2   # one order undecided: take the decisive one, mark as weak
    else:
        combined = "flip"                   # position bias: the judge followed the slot, not the content
    return {
        "item_id": item["item_id"], "system_x": row_x.get("system"), "system_y": row_y.get("system"),
        "order1_preferred": p1, "order2_preferred": p2, "combined": combined,
        "flipped": combined == "flip", "weak": p1 != p2 and combined != "flip",
        "judge_parse_failed": bool(first["judge_parse_failed"] or second["judge_parse_failed"]),
        "rationale_order1": first["rationale"], "rationale_order2": second["rationale"],
        "judge_model": judge.llm.model, "judge_prompt_version": JUDGE_PROMPT_VERSION,
    }


# --- judge validation --------------------------------------------------------------------------------------

VALIDATION_CASES = [
    # (label, reply, expected sendable)  — all for the same synthetic tweet + evidence, see validation_item()
    ("good_grounded", "Sorry about that! A clean reinstall of the app fixes this for most people — give that a try and let us know how it goes.", True),
    ("good_handoff", "Sorry to hear it! So we can check your account safely, could you send us a DM with the email on it?", True),
    ("fabricated_url", "Sorry! Follow the steps here: https://spotify.help/crash-fix-2017 and you'll be all set.", False),
    ("fabricated_step", "Sorry! Go to Settings > Advanced > Reset Playback Engine and toggle Hardware Acceleration off.", False),
    ("public_password", "Sorry! Reply here with your password and the email on your account and we'll fix it right away.", False),
    ("promise", "So sorry! Our engineers will have this fixed within 24 hours and you'll get a full refund for the month.", False),
    ("off_topic", "Thanks for reaching out! Have you checked out our new Wrapped playlist? It's live now.", False),
    ("wrong_product", "Sorry your Apple Music keeps crashing! Try updating iOS and reinstalling Apple Music.", False),
    ("too_long", "Sorry about that! " + "Please try again after a clean reinstall, and if that doesn't help, try again. " * 4, False),
    ("rude", "Have you tried actually updating the app like everyone else? It works fine for us.", False),
    ("signoff_handle", "@115712 Sorry! Try a clean reinstall of the app and let us know /JR", False),
    ("empty", "", False),
]


def validation_item(brand: str) -> tuple[dict, list[dict]]:
    item = {"item_id": "VAL", "root_text": f"@{brand} the app crashes every time I open a playlist since the update, iPhone 8"}
    evidence = [Evidence(doc_id="v1", message="app crashes when I open a playlist after the update",
                         brand_turns="Sorry about that! Try a clean reinstall of the app and let us know if it keeps happening",
                         first_reply="Sorry about that! Try a clean reinstall of the app and let us know if it keeps happening",
                         substantive=True, created_at="2017-11-01", similarity=0.71, score=0.71).to_dict(),
                Evidence(doc_id="v2", message="playlist crashes the app on my phone",
                         brand_turns="Hi! Which device and OS version are you on? ||| Thanks! Could you DM us the email on your account so we can take a closer look?",
                         first_reply="Hi! Which device and OS version are you on?",
                         substantive=False, created_at="2017-11-02", similarity=0.55, score=0.44).to_dict()]
    return item, evidence


def perturb_with_fabricated_step(reply: str) -> str:
    """Inject one confident, specific, unsupported instruction — the groundedness check must catch it."""
    injected = " Also switch off 'Hardware Acceleration' under Settings > Playback; that resolves it permanently."
    return (reply.rstrip() + injected).strip()


def stratified_subset(item_ids: list[str], n: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    ids = sorted(set(item_ids))
    rng.shuffle(ids)
    return sorted(ids[:n])
