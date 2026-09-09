"""Provider-neutral LLM access: local Ollama models and Gemini, behind one interface, always cached."""
from __future__ import annotations

from ..config import Settings
from .base import CachedLLM, LLMClient, LLMError, LLMResponse
from .cache import ResponseCache


def make_client(which: str, settings: Settings, model: str | None = None) -> LLMClient:
    """`which` is 'local' (Ollama) or 'gemini'. Every client is wrapped in the shared SQLite cache."""
    cache = ResponseCache(settings.paths.cache_db)
    if which == "local":
        from .ollama_client import OllamaClient

        client: LLMClient = OllamaClient(model or settings.local_model)
    elif which == "gemini":
        from .gemini_client import GeminiClient

        client = GeminiClient(model or settings.judge_model, rpm=settings.gemini_rpm,
                              counter_path=settings.paths.outputs / "gemini_daily_calls.json")
    else:
        raise ValueError(f"unknown provider {which!r}; expected 'local' or 'gemini'")
    return CachedLLM(client, cache)


__all__ = ["CachedLLM", "LLMClient", "LLMError", "LLMResponse", "ResponseCache", "make_client"]
