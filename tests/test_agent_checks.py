from __future__ import annotations

import pytest
from pydantic import ValidationError

from support_agent.agent.checks import HardRules, check_reply, post_process
from support_agent.agent.schemas import AgentOutput, REASON_CODES, fallback_output, output_schema

INTENTS = ["playback_issue", "payment_billing", "account_access", "other"]
RULES = HardRules.from_config({
    "account_or_pii": [r"\b(hacked|someone (?:else )?(?:is )?using my account|unauthori[sz]ed)\b"],
    "payment_refund": [r"\b(refund|charged twice|double[- ]charged|chargeback)\b"],
    "legal_safety_threat": [r"\b(lawyer|lawsuit|sue you|legal action)\b"],
    "explicit_human_request": [r"\b(speak (?:to|with) (?:a )?(?:human|person|real person)|talk to (?:a )?human)\b"],
})


def make_output(**over) -> AgentOutput:
    base = dict(intent="playback_issue", intent_secondary="none", intent_confidence=0.8, evidence_ids=["e1"],
                reply="Sorry about that! Try a clean reinstall of the app and let us know if it persists.",
                reason="known playback fix from history", reason_code="none", decision="auto")
    base.update(over)
    return AgentOutput(**base)


def test_schema_shape_and_generation_order():
    schema = output_schema(INTENTS, evidence_ids=["e1", "e2"])
    assert list(schema["properties"]) == ["intent", "intent_secondary", "intent_confidence", "evidence_ids",
                                          "reply", "reason", "reason_code", "decision"]
    assert schema["properties"]["intent"]["enum"] == INTENTS
    assert schema["properties"]["evidence_ids"]["items"]["enum"] == ["e1", "e2"]
    assert set(schema["properties"]["reason_code"]["enum"]) == set(REASON_CODES)
    assert schema["additionalProperties"] is False


def test_output_validation():
    out = make_output(intent_secondary="none")
    assert out.intent_secondary is None
    with pytest.raises(ValidationError):
        make_output(decision="maybe")
    with pytest.raises(ValidationError):
        make_output(reason_code="because")
    with pytest.raises(ValidationError):
        make_output(intent_confidence=1.5)
    fb = fallback_output("other", "bad json")
    assert fb.decision == "escalate" and fb.reason_code == "draft_invalid" and fb.reply == ""


def test_check_reply_catches_each_violation():
    ev = ["@1 Here's how: https://t.co/good1 /JR"]
    assert check_reply("", ev).codes == ["empty_reply"]
    assert "too_long" in check_reply("x" * 281, ev).codes
    assert "anonymised_handle" in check_reply("@115712 sorry about that", ev).codes
    assert "url_not_in_evidence" in check_reply("See https://t.co/other", ev).codes
    assert "url_not_in_evidence" not in check_reply("See https://t.co/good1.", ev).codes
    assert "agent_signoff" in check_reply("Try a reinstall and let us know /JR", ev).codes
    assert "agent_signoff" in check_reply("Try a reinstall and let us know. ^AB", ev).codes
    assert "agent_signoff" in check_reply("Could you DM us? We'll look backstage /LO https://t.co/good1", ev).codes
    assert "agent_signoff" not in check_reply("Try the A/B toggle and the I/O settings, then reply", ev).codes
    assert "promise" in check_reply("It will be fixed within 24 hours", ev).codes
    assert "promise" in check_reply("You'll get a full refund", ev).codes
    assert "sensitive_request_public" in check_reply("Reply with your password and card number", ev).codes
    assert "sensitive_request_public" not in check_reply("DM us and we'll reset your password from there", ev).codes
    assert "formatting_artifact" in check_reply("Try a reinstall. ||| Thanks! Let us know", ev).codes
    assert "formatting_artifact" in check_reply("As shown in [E1], try a reinstall", ev).codes
    assert "claims_action_taken" in check_reply("Hey there! We've already replied to your DM. Check your inbox 🙂", ev).codes
    assert "claims_action_taken" in check_reply("Hi! We've just sent a DM your way. Let's carry on there", ev).codes
    assert "claims_action_taken" in check_reply("Sorry! We've passed this on to the team", ev).codes
    assert "claims_action_taken" not in check_reply("Could you send us a DM with your account email? We'll take a look there.", ev).codes
    assert not check_reply("Sorry to hear that! Try logging out and back in, then let us know how it goes.", ev)


def run(out, message, **kw):
    params = dict(lang="en", evidence_texts=[], retrieval_max_sim=0.6, hard_rules=RULES, min_similarity=0.2)
    params.update(kw)
    return post_process(out, message, **params)


def test_post_process_hard_rules_override_auto():
    r = run(make_output(), "@Brand I was charged twice for premium, I want a refund")
    assert r.output.decision == "escalate" and r.output.reason_code == "payment_refund"
    assert r.model_decision == "auto" and r.forced_reason == "payment_refund"
    r = run(make_output(), "someone is using my account, I think I've been hacked")
    assert r.output.reason_code == "account_or_pii"
    r = run(make_output(), "I want to speak to a human being, not a bot")
    assert r.output.reason_code == "explicit_human_request"


def test_post_process_keeps_model_reason_when_it_already_escalated():
    out = make_output(decision="escalate", reason_code="legal_safety_threat", reason="threatens legal action")
    r = run(out, "I'll get my lawyer involved, I was charged twice")
    assert r.output.reason_code == "legal_safety_threat" and r.output.reason == "threatens legal action"
    assert r.forced_reason == "payment_refund"  # the rule still fired; we just did not overwrite a good reason


def test_post_process_routing_rules():
    r = run(make_output(), "@Brand la aplicación no funciona en mi teléfono", lang="es")
    assert r.output.reason_code == "non_english"
    r = run(make_output(), "@Brand 😩😩", lang="unk")
    assert r.output.reason_code == "no_actionable_content"
    # objective message facts override even a model that escalated for its own reason
    r = run(make_output(decision="escalate", reason_code="anger_churn", reason="distress emojis"), "@Brand 😩😩", lang="unk")
    assert r.output.reason_code == "no_actionable_content" and r.forced_reason == "no_actionable_content"
    r = run(make_output(), "the app crashes when I open a playlist", retrieval_max_sim=0.05)
    assert r.output.reason_code == "no_relevant_resolution"
    r = run(make_output(), "the app crashes when I open a playlist", retrieval_max_sim=None)
    assert r.output.decision == "auto"  # no retrieval score available (e.g. no-RAG ablation) -> no floor


def test_post_process_invalid_draft_is_not_sent():
    out = make_output(reply="Try this https://t.co/madeup and it will be fixed within 24 hours /JR")
    r = run(out, "the app crashes when I open a playlist", evidence_texts=["@1 try a reinstall"])
    assert r.output.decision == "escalate" and r.output.reason_code == "draft_invalid"
    assert set(r.violations) == {"url_not_in_evidence", "promise", "agent_signoff"}
    assert "url_not_in_evidence" in r.output.reason


def test_post_process_drafted_handoff_is_an_escalation():
    out = make_output(intent="account_settings_change", reply="Hey! To change your country, could you DM us your account's email? We'll take a look backstage.")
    r = run(out, "I moved to Vietnam and cannot change my country setting, it only shows US")
    assert r.output.decision == "escalate" and r.output.reason_code == "account_or_pii" and r.forced_reason == "dm_handoff"
    assert "hands the customer to DM" in r.output.reason
    out = make_output(intent="payment_billing", reply="Sorry about that! Send us a DM with the email on the account and we'll check the charge.")
    r = run(out, "I think the amount taken this month is wrong")   # no hard-rule keyword, so the hand-off rule decides
    assert r.output.decision == "escalate" and r.output.reason_code == "payment_refund" and r.forced_reason == "dm_handoff"
    # a public troubleshooting reply is not a hand-off
    r = run(make_output(reply="Try logging out and back in, then restart the app."), "the app crashes when I open a playlist")
    assert r.output.decision == "auto" and r.forced_reason is None
    # the model already escalating is left alone (no double counting as dm_handoff)
    r = run(make_output(decision="escalate", reason_code="account_or_pii", reply="Could you DM us your email?"), "cannot change my country")
    assert r.forced_reason is None and r.output.reason_code == "account_or_pii"


def test_post_process_clean_auto_passes_through():
    r = run(make_output(), "the app crashes when I open a playlist")
    assert r.output.decision == "auto" and r.forced_reason is None and r.violations == []
    assert r.output == make_output()
