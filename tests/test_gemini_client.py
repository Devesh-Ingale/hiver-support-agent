from __future__ import annotations

import pytest

from support_agent.llm import gemini_client as gc
from support_agent.llm.base import QuotaExhausted

QUOTA_MSG = ("You exceeded your current quota, please check your plan and billing details. "
             "* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, "
             "limit: 20, model: gemini-3.8-flash\nPlease retry in 3.153477113s.")


def test_quota_message_parsing():
    assert gc.describe_quota_error(QUOTA_MSG) == "quota metric generativelanguage.googleapis.com/generate_content_free_tier_requests limit 20 (gemini-3.8-flash)"
    assert gc.describe_quota_error("something else") == "quota exceeded"
    assert float(gc.RETRY_HINT_RE.search(QUOTA_MSG).group(1)) == pytest.approx(3.153477113)


class FakeAPIError(gc.errors.APIError):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        self.status = "RESOURCE_EXHAUSTED"
        self.details = []
        self.response = None


def make_client(tmp_path, monkeypatch, responses):
    """Build a GeminiClient whose generate_content pops scripted results (exceptions or objects)."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    sleeps = []
    monkeypatch.setattr(gc.time, "sleep", lambda s: sleeps.append(s))
    client = gc.GeminiClient("gemini-test", rpm=1000, daily_limit=100, counter_path=tmp_path / "calls.json")

    def fake_generate(user, config):
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    client._generate = fake_generate
    return client, sleeps


class Resp:
    text = '{"ok": true}'

    class usage_metadata:
        prompt_token_count = 10
        candidates_token_count = 5


def test_persistent_429_becomes_quota_exhausted_quickly(tmp_path, monkeypatch):
    client, sleeps = make_client(tmp_path, monkeypatch, [FakeAPIError(429, QUOTA_MSG)] * 5)
    with pytest.raises(QuotaExhausted, match="limit 20"):
        client.complete("sys", "user")
    assert len(sleeps) == gc.MAX_429_IN_A_ROW - 1            # waited on the hint, then gave up instead of backing off for minutes
    assert all(s < 10 for s in sleeps)


def test_transient_429_then_success(tmp_path, monkeypatch):
    client, sleeps = make_client(tmp_path, monkeypatch, [FakeAPIError(429, "Please retry in 4s."), Resp()])
    out = client.complete("sys", "user")
    assert out.json() == {"ok": True} and len(sleeps) == 1 and 4 < sleeps[0] < 7


def test_503_backs_off_then_succeeds(tmp_path, monkeypatch):
    client, sleeps = make_client(tmp_path, monkeypatch, [FakeAPIError(503, "overloaded"), FakeAPIError(503, "overloaded"), Resp()])
    out = client.complete("sys", "user")
    assert out.text == '{"ok": true}' and sleeps == [5.0, 10.0] and client.calls_today() == 3
