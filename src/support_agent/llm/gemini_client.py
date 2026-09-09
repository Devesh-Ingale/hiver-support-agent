"""Gemini through the google-genai SDK, paced for the free tier and resumable when the quota bites."""
from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from ..config import ROOT
from .base import LLMError, LLMResponse, QuotaExhausted
from .rate_limit import DailyCounter, RateLimiter

log = logging.getLogger(__name__)

RETRYABLE = {429, 500, 502, 503, 504}
RETRY_HINT_RE = re.compile(r"retry in (\d+(?:\.\d+)?)\s*s", re.I)
QUOTA_LIMIT_RE = re.compile(r"metric:\s*(\S+),\s*limit:\s*(\d+)(?:,\s*model:\s*(\S+))?", re.I)
MAX_429_IN_A_ROW = 3   # per-minute limits clear within a minute; anything longer is the daily quota


def describe_quota_error(message: str) -> str:
    m = QUOTA_LIMIT_RE.search(message or "")
    return f"quota metric {m.group(1)} limit {m.group(2)} ({m.group(3) or 'model'})" if m else "quota exceeded"


class GeminiClient:
    provider = "gemini"

    def __init__(self, model: str, api_key: str | None = None, rpm: int = 10, daily_limit: int = 1400,
                 counter_path: Path | None = None, max_retries: int = 6):
        load_dotenv(ROOT / ".env")   # explicit path: works from any cwd and from stdin/`-c` scripts
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is not set (put it in .env; see .env.example)")
        self.model = model
        self.max_retries = max_retries
        # SDK-internal retries off: they turn a 429/503 into silent minutes of waiting; we handle retries here
        self._client = genai.Client(api_key=key, http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1), timeout=120_000))
        self._limiter = RateLimiter(rpm)
        self._counter = DailyCounter(counter_path or Path("outputs/gemini_daily_calls.json"), daily_limit)

    def _config(self, system: str, schema: dict | None, temperature: float, seed: int, max_tokens: int):
        kwargs: dict = dict(system_instruction=system, temperature=temperature, max_output_tokens=max_tokens)
        if schema is not None:
            kwargs.update(response_mime_type="application/json", response_json_schema=schema)
        try:
            return types.GenerateContentConfig(seed=seed, **kwargs)
        except Exception:  # SDK version without `seed`
            return types.GenerateContentConfig(**kwargs)

    def complete(self, system: str, user: str, *, schema: dict | None = None, temperature: float = 0.0,
                 seed: int = 42, max_tokens: int = 1024) -> LLMResponse:
        config = self._config(system, schema, temperature, seed, max_tokens)
        delay = 5.0
        rate_limited = 0
        for attempt in range(self.max_retries + 1):
            if self._counter.remaining() <= 0:
                raise QuotaExhausted(f"daily budget of {self._counter.limit} Gemini calls reached; resume tomorrow")
            self._limiter.wait()
            self._counter.increment()
            t0 = time.perf_counter()
            try:
                resp = self._generate(user, config)
            except errors.APIError as e:
                message = str(e.message)
                if e.code == 429:
                    rate_limited += 1
                    if rate_limited >= MAX_429_IN_A_ROW:
                        # a per-minute limit clears within the hint; a limit that persists is the day's quota
                        raise QuotaExhausted(f"gemini {self.model}: {describe_quota_error(message)} — stop and resume tomorrow") from e
                    hint = RETRY_HINT_RE.search(message)
                    wait = min(max(float(hint.group(1)), 2.0) if hint else 20.0, 65.0) + 1.0
                    log.warning("gemini %s -> 429 (%s); waiting %.0fs", self.model, describe_quota_error(message), wait)
                    time.sleep(wait)
                    continue
                if e.code in RETRYABLE and attempt < self.max_retries:
                    log.warning("gemini %s -> HTTP %s; retry %d in %.0fs", self.model, e.code, attempt + 1, delay)
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
                    continue
                raise LLMError(f"gemini {self.model}: HTTP {e.code} {message}") from e
            latency_ms = int((time.perf_counter() - t0) * 1000)
            usage = getattr(resp, "usage_metadata", None)
            return LLMResponse(
                text=resp.text or "",
                provider=self.provider,
                model=self.model,
                latency_ms=latency_ms,
                input_tokens=getattr(usage, "prompt_token_count", None),
                output_tokens=getattr(usage, "candidates_token_count", None),
            )
        raise LLMError(f"gemini {self.model}: gave up after {self.max_retries} retries")

    def _generate(self, user: str, config):
        """The one line that talks to Google; tests stub this."""
        return self._client.models.generate_content(model=self.model, contents=user, config=config)

    def calls_today(self) -> int:
        return self._counter.used()
