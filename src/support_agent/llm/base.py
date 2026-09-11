"""The one interface every model speaks: complete(system, user, schema) -> LLMResponse."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .cache import ResponseCache


class LLMError(RuntimeError):
    """Provider call failed after retries (or was refused). Carries the provider's message."""


class QuotaExhausted(LLMError):
    """Daily call budget reached — stop cleanly so the run can resume tomorrow."""


class ProviderUnavailable(LLMError):
    """The provider cannot be reached at all (server down, no network) — stop the run instead of failing every item."""


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_ms: int
    cached: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None

    def json(self) -> dict[str, Any] | None:
        """Lenient parse of the response as a JSON object; None when it is not one."""
        return parse_json_object(self.text)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LLMResponse":
        return cls(**d)


class LLMClient(Protocol):
    provider: str
    model: str

    def complete(self, system: str, user: str, *, schema: dict | None = None, temperature: float = 0.0,
                 seed: int = 42, max_tokens: int = 1024) -> LLMResponse: ...


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I | re.M)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object out of model text, tolerating code fences and leading/trailing prose."""
    if not text:
        return None
    candidate = _FENCE_RE.sub("", text.strip())
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(candidate[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


class CachedLLM:
    """Wraps any client so identical requests are answered from SQLite instead of the provider.

    The cache key covers provider, model, both prompts, the schema and the sampling parameters, so a
    prompt edit or model change is automatically a cache miss.
    """

    def __init__(self, client: LLMClient, cache: ResponseCache):
        self._client = client
        self._cache = cache
        self.provider = client.provider
        self.model = client.model

    def complete(self, system: str, user: str, *, schema: dict | None = None, temperature: float = 0.0,
                 seed: int = 42, max_tokens: int = 1024) -> LLMResponse:
        key = self._cache.key(provider=self.provider, model=self.model, system=system, user=user,
                              schema=schema, temperature=temperature, seed=seed, max_tokens=max_tokens)
        hit = self._cache.get(key)
        if hit is not None:
            response = LLMResponse.from_dict(hit)
            response.cached = True
            return response
        response = self._client.complete(system, user, schema=schema, temperature=temperature,
                                         seed=seed, max_tokens=max_tokens)
        self._cache.put(key, self.provider, self.model, response.to_dict())
        return response
