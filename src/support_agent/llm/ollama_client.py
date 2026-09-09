"""Local models through the Ollama server (default http://localhost:11434)."""
from __future__ import annotations

import time

import ollama

from .base import LLMError, LLMResponse


class OllamaClient:
    provider = "ollama"

    def __init__(self, model: str, host: str | None = None, num_ctx: int = 8192, keep_alive: str = "15m",
                 think: bool | None = False):
        self.model = model
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.think = think
        self._client = ollama.Client(host=host) if host else ollama.Client()

    def complete(self, system: str, user: str, *, schema: dict | None = None, temperature: float = 0.0,
                 seed: int = 42, max_tokens: int = 1024) -> LLMResponse:
        kwargs: dict = dict(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            options={"temperature": temperature, "seed": seed, "num_ctx": self.num_ctx, "num_predict": max_tokens},
            keep_alive=self.keep_alive,
        )
        if schema is not None:
            kwargs["format"] = schema
        if self.think is not None:
            kwargs["think"] = self.think
        t0 = time.perf_counter()
        try:
            resp = self._client.chat(**kwargs)
        except ollama.ResponseError as e:
            # models without a thinking mode reject the `think` flag; retry once without it
            if "think" in str(e).lower() and "think" in kwargs:
                kwargs.pop("think")
                self.think = None
                try:
                    resp = self._client.chat(**kwargs)
                except ollama.ResponseError as e2:
                    raise LLMError(f"ollama {self.model}: {e2}") from e2
            else:
                raise LLMError(f"ollama {self.model}: {e}") from e
        except Exception as e:  # connection refused etc.
            raise LLMError(f"ollama {self.model}: {e}") from e
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return LLMResponse(
            text=resp.message.content or "",
            provider=self.provider,
            model=self.model,
            latency_ms=latency_ms,
            input_tokens=getattr(resp, "prompt_eval_count", None),
            output_tokens=getattr(resp, "eval_count", None),
        )

    def available_models(self) -> list[str]:
        return [m.model for m in self._client.list().models]
