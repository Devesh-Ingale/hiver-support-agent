from __future__ import annotations

import json

import pandas as pd
import pytest

from support_agent.agent import prompts
from support_agent.agent.pipeline import Agent, AgentConfig
from support_agent.llm.base import LLMResponse
from support_agent.retrieval.index import TfidfIndex


class ScriptedLLM:
    """Returns the given texts in order; records the prompts it was asked."""

    provider = "fake"
    model = "fake-1"

    def __init__(self, *texts: str):
        self.texts = list(texts)
        self.calls: list[tuple[str, str, dict | None]] = []

    def complete(self, system, user, *, schema=None, temperature=0.0, seed=42, max_tokens=1024):
        self.calls.append((system, user, schema))
        text = self.texts.pop(0) if self.texts else "{}"
        return LLMResponse(text=text, provider=self.provider, model=self.model, latency_ms=3)


GOOD = {
    "intent": "playback_issue", "intent_secondary": "none", "intent_confidence": 0.9, "evidence_ids": ["E1"],
    "reply": "Sorry about that! Try a clean reinstall of the app and let us know if it keeps happening.",
    "reason": "known playback fix in the evidence", "reason_code": "none", "decision": "auto",
}


@pytest.fixture()
def index() -> TfidfIndex:
    docs = pd.DataFrame([
        ("d1", "app crashes when I open a playlist", "@1 Sorry! Try a clean reinstall of the app /JR", True),
        ("d2", "charged twice for premium", "@2 Please DM us your account email /AB", False),
        ("d3", "songs skip in offline mode", "@3 Toggle offline mode off and on /CD", True),
    ], columns=["doc_id", "message", "brand_turns", "substantive"]).assign(first_reply=lambda d: d.brand_turns, created_at="2017-11-01")
    return TfidfIndex(min_df=1).fit(docs)


def test_taxonomy_loader_and_rendering(taxonomy):
    assert taxonomy.intent_ids == ["playback_issue", "payment_billing", "account_access", "other_unclear"]
    assert taxonomy.other_intent == "other_unclear"
    assert taxonomy.hard_rules().match("I was charged twice!!") == "payment_refund"
    md = taxonomy.policy_markdown()
    assert "HARD" in md and "`anger_churn`" in md and "never" in md.lower()
    assert "Login trouble is account_access" in taxonomy.intents_markdown()


def test_happy_path_records_everything(taxonomy, index):
    llm = ScriptedLLM(json.dumps(GOOD))
    agent = Agent(llm, taxonomy, AgentConfig(system="main", k=2, min_similarity=0.1), index=index)
    row = agent.handle({"item_id": "t1", "text": "@SpotifyCares my app crashes when I open a playlist"})
    assert row["decision"] == "auto" and row["intent"] == "playback_issue" and row["forced_reason"] is None
    assert row["evidence_ids"] == ["d1"] and row["evidence"][0]["doc_id"] == "d1"
    assert row["retrieval_max_sim"] > 0.1 and row["lang"] == "en" and row["violations"] == []
    assert row["prompt_version"] == prompts.PROMPT_VERSION and row["system"] == "main"
    assert row["parse_failed"] is False and row["repaired"] is False
    system_prompt, user_prompt, schema = llm.calls[0]
    assert "SpotifyCares" in system_prompt and "# Escalation policy" in system_prompt
    assert "[E1]" in user_prompt and "Try a clean reinstall" in user_prompt and "@1" not in user_prompt
    n_ev = len(row["evidence"])
    assert 1 <= n_ev <= 2   # the tiny corpus has no lexical overlap beyond d1, so fewer than k is expected
    assert schema["properties"]["evidence_ids"]["items"]["enum"] == [f"E{i}" for i in range(1, n_ev + 1)]


def test_policy_overrides_model(taxonomy, index):
    llm = ScriptedLLM(json.dumps(GOOD))
    agent = Agent(llm, taxonomy, AgentConfig(system="main", k=2), index=index)
    row = agent.handle({"item_id": "t2", "text": "@SpotifyCares charged twice this month, I want a refund"})
    assert row["decision"] == "escalate" and row["reason_code"] == "payment_refund"
    assert row["model_decision"] == "auto" and row["forced_reason"] == "payment_refund"
    assert row["reply"] == GOOD["reply"]  # the draft is kept so reply quality can still be judged


def test_repair_then_fallback(taxonomy, index):
    llm = ScriptedLLM("not json", "still not json")
    agent = Agent(llm, taxonomy, AgentConfig(system="main"), index=index)
    row = agent.handle({"item_id": "t3", "text": "@SpotifyCares songs skip in offline mode", "lang": "en"})
    assert len(llm.calls) == 2 and llm.calls[1][1].endswith(prompts.REPAIR_SUFFIX)
    assert row["parse_failed"] is True and row["repaired"] is True
    assert row["decision"] == "escalate" and row["reason_code"] == "draft_invalid" and row["intent"] == "other_unclear"


def test_repair_succeeds_on_second_try(taxonomy, index):
    llm = ScriptedLLM("garbage", json.dumps(GOOD))
    agent = Agent(llm, taxonomy, AgentConfig(system="main"), index=index)
    row = agent.handle({"item_id": "t4", "text": "@SpotifyCares app crashes when I open a playlist"})
    assert row["parse_failed"] is False and row["repaired"] is True and row["decision"] == "auto"


def test_unknown_intent_maps_to_other(taxonomy, index):
    llm = ScriptedLLM(json.dumps({**GOOD, "intent": "made_up", "intent_secondary": "also_made_up"}))
    agent = Agent(llm, taxonomy, AgentConfig(system="main"), index=index)
    row = agent.handle({"item_id": "t5", "text": "@SpotifyCares app crashes when I open a playlist"})
    assert row["intent"] == "other_unclear" and row["intent_secondary"] is None and row["intent_invalid"] is True


def test_no_rag_ablation_has_no_evidence_and_no_floor(taxonomy):
    llm = ScriptedLLM(json.dumps({**GOOD, "evidence_ids": []}))
    agent = Agent(llm, taxonomy, AgentConfig(system="no_rag", use_retrieval=False, min_similarity=0.9))
    row = agent.handle({"item_id": "t6", "text": "@SpotifyCares app crashes when I open a playlist"})
    assert row["evidence"] == [] and row["retrieval_max_sim"] is None and row["decision"] == "auto"
    _, user_prompt, schema = llm.calls[0]
    assert "no evidence available" in user_prompt and "enum" not in schema["properties"]["evidence_ids"]["items"]


def test_low_similarity_routes_to_human(taxonomy, index):
    llm = ScriptedLLM(json.dumps(GOOD))
    agent = Agent(llm, taxonomy, AgentConfig(system="main", min_similarity=0.99), index=index)
    row = agent.handle({"item_id": "t7", "text": "@SpotifyCares app keeps crashing whenever I open playlists", "lang": "en"})
    assert 0 < row["retrieval_max_sim"] < 0.99
    assert row["decision"] == "escalate" and row["reason_code"] == "no_relevant_resolution"


def test_non_english_routes_without_calling_policy_rules(taxonomy, index):
    llm = ScriptedLLM(json.dumps(GOOD))
    agent = Agent(llm, taxonomy, AgentConfig(system="main"), index=index)
    row = agent.handle({"item_id": "t8", "text": "@SpotifyCares la aplicación se cierra cuando abro una lista", "lang": "es"})
    assert row["reason_code"] == "non_english" and row["decision"] == "escalate"
