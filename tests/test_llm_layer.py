from __future__ import annotations

import json

import pytest

from support_agent.llm.base import CachedLLM, LLMResponse, QuotaExhausted, parse_json_object
from support_agent.llm.batch import read_jsonl, run_batch
from support_agent.llm.cache import ResponseCache
from support_agent.llm.rate_limit import DailyCounter, RateLimiter


class FakeClient:
    provider = "fake"
    model = "fake-1"

    def __init__(self):
        self.calls = 0

    def complete(self, system, user, *, schema=None, temperature=0.0, seed=42, max_tokens=1024):
        self.calls += 1
        return LLMResponse(text=json.dumps({"echo": user, "n": self.calls}), provider=self.provider,
                           model=self.model, latency_ms=1)


def test_parse_json_object_is_lenient():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! Here it is: {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}
    assert parse_json_object("[1, 2]") is None
    assert parse_json_object("not json at all") is None
    assert parse_json_object("") is None


def test_cache_hits_on_identical_request_and_misses_on_any_change(tmp_path):
    client = FakeClient()
    llm = CachedLLM(client, ResponseCache(tmp_path / "cache.sqlite"))
    a = llm.complete("sys", "hello", schema={"type": "object"})
    b = llm.complete("sys", "hello", schema={"type": "object"})
    assert client.calls == 1 and a.cached is False and b.cached is True and a.text == b.text
    llm.complete("sys", "hello", schema={"type": "object"}, temperature=0.5)
    llm.complete("sys!", "hello", schema={"type": "object"})
    llm.complete("sys", "hello")
    assert client.calls == 4
    assert llm.complete("sys", "hello").json() == {"echo": "hello", "n": 4}


def test_rate_limiter_spaces_calls():
    now = [0.0]
    slept = []
    limiter = RateLimiter(rpm=2, clock=lambda: now[0], sleep=lambda s: (slept.append(s), now.__setitem__(0, now[0] + s)))
    assert limiter.wait() == 0.0
    assert limiter.wait() == 0.0
    waited = limiter.wait()  # third call inside the same minute must wait for the first to age out
    assert waited > 59.9 and slept and slept[0] == waited


def test_daily_counter_persists(tmp_path):
    counter = DailyCounter(tmp_path / "calls.json", limit=3)
    assert counter.remaining() == 3
    counter.increment(); counter.increment()
    assert DailyCounter(tmp_path / "calls.json", limit=3).used() == 2
    assert counter.remaining() == 1


def test_run_batch_resumes_and_survives_errors(tmp_path):
    out = tmp_path / "out.jsonl"
    items = [{"item_id": i} for i in range(5)]

    def fn(item):
        if item["item_id"] == 2:
            raise ValueError("boom")
        if item["item_id"] == 4:
            raise QuotaExhausted("budget")
        return {"item_id": item["item_id"], "ok": True}

    rows = run_batch(items, fn, out)
    assert [r["item_id"] for r in rows] == [0, 1, 2, 3]          # stopped cleanly before item 4
    assert "error" in rows[2] and "ValueError" in rows[2]["error"]

    calls = []
    rows = run_batch(items, lambda it: (calls.append(it["item_id"]), {"item_id": it["item_id"], "ok": True})[1], out)
    assert calls == [4]                                            # only the missing item is re-run
    assert len(rows) == 5 and read_jsonl(out) == rows
