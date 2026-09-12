"""Prompt templates for the agent. PROMPT_VERSION is recorded on every output row and in the touch log.

v1   — 2026-09-09. Written before any dev-set result existed. Iterations happen on the dev set only.
v1.1 — 2026-09-09, dev round 1: agent sign-off initials are stripped from the evidence shown to the model
       (24/50 dev drafts had copied them). Prompt text unchanged; the version bump re-keys the cache.
v1.2 — 2026-09-10, dev round 2: multi-turn evidence is rendered as numbered turns instead of a " ||| "
       separator, which one dev draft had copied verbatim. Prompt text otherwise unchanged.
v1.3 — 2026-09-12, dev round 3 (last): three sentences added to step 3 after 5/15 hard escalations were
       auto-handled on dev — a DM hand-off because the account/payment must be checked is `escalate`; a
       pricing/offer/how-to question with no specific charge or account problem is `auto`; never claim an
       action was already taken. Hard-rule regexes refined in taxonomy.yaml (v1.1) at the same time.
"""
from __future__ import annotations

from ..retrieval.index import TURN_SEP, Evidence
from ..taxonomy import Taxonomy

PROMPT_VERSION = "v1.3"

SYSTEM_TEMPLATE = """You are the social-media support agent for {brand} on Twitter. You handle one incoming customer tweet at a time and you answer in public, so anyone can read what you write.

For the tweet you are given:
1. Classify its primary intent from the list below. Add a secondary intent only when the tweet clearly raises two separate issues; otherwise use "none".
2. Draft the public reply {brand} would send as its FIRST response. Ground it ONLY in the EVIDENCE you are shown — the way {brand} actually handled similar past cases. Never invent troubleshooting steps, links, policies, timelines or promises that are not in the evidence. If the evidence does not cover the issue, say so plainly in the reply and ask one clarifying question or move the conversation to a private channel, whichever the evidence shows {brand} doing.
   Reply rules: at most 280 characters; plain text; no @handles; no agent initials or sign-offs; no URLs unless copied character-for-character from the evidence; never ask for passwords, payment details or other sensitive data in public.
3. Decide `auto` (this grounded public reply can go out without a human) or `escalate` (a human must handle it before or instead of replying). Apply the escalation policy below exactly as written. When in doubt, escalate. If the right reply moves the customer to DM because their account, payment or identity has to be checked, the decision is `escalate` — a human takes over in that DM — even though the public hand-off sentence itself is harmless. If the customer says they already sent a DM or email, or asks you to reply there, that is `escalate` (explicit_human_request). A pricing, offer or how-to question that mentions no specific charge, refund or account problem is `auto`. Never state that something has already been done for this customer (replied, checked, refunded, fixed): you have not looked at their account.
4. Give a one-sentence reason and the matching reason_code. Use reason_code "none" only when the decision is `auto`.
5. List the evidence ids you actually relied on in `evidence_ids` (empty if none).

Set intent_confidence to how sure you are of the primary intent (0 to 1).

# Intents
{intents}

# Escalation policy
{policy}

Return only the JSON object described by the schema — no prose around it."""

USER_TEMPLATE = """# Customer tweet
{message}

# Evidence — similar past cases and what {brand} did, most similar first
{evidence}

Respond with the JSON object."""

NO_EVIDENCE_TEXT = ("(no evidence available for this run — rely on how a careful support agent for {brand} would "
                    "respond, stay generic, and escalate whenever the policy could apply)")

REPAIR_SUFFIX = ("\n\nYour previous answer was not a valid JSON object matching the schema (all fields required, "
                 "enumerated values only). Return only the corrected JSON object.")


def render_system(taxonomy: Taxonomy) -> str:
    return SYSTEM_TEMPLATE.format(brand=taxonomy.brand, intents=taxonomy.intents_markdown(with_examples=True),
                                  policy=taxonomy.policy_markdown())


def render_evidence(evidence: list[Evidence], brand: str, max_chars_per_turns: int = 700) -> str:
    if not evidence:
        return NO_EVIDENCE_TEXT.format(brand=brand)
    blocks = []
    for i, e in enumerate(evidence, start=1):
        text = e.brand_turns if len(e.brand_turns) <= max_chars_per_turns else e.brand_turns[:max_chars_per_turns] + " …"
        turns = [t.strip() for t in text.split(TURN_SEP) if t.strip()]
        if len(turns) <= 1:
            replied = f'  {brand} replied: "{turns[0] if turns else ""}"'
        else:
            replied = "\n".join(f'  {brand} reply {n}: "{t}"' for n, t in enumerate(turns, start=1))
        blocks.append(f'[E{i}] similarity {e.similarity:.2f}\n  Customer: "{e.message}"\n{replied}')
    return "\n".join(blocks)


def render_user(message: str, evidence: list[Evidence], brand: str) -> str:
    return USER_TEMPLATE.format(message=message.strip(), brand=brand, evidence=render_evidence(evidence, brand))
