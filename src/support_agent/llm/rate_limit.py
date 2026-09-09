"""Client-side pacing for the Gemini free tier: requests-per-minute bucket + persisted daily counter."""
from __future__ import annotations

import json
import time
from collections import deque
from datetime import date
from pathlib import Path


class RateLimiter:
    """Blocks so that no more than `rpm` calls start in any rolling 60-second window."""

    def __init__(self, rpm: int, clock=time.monotonic, sleep=time.sleep):
        self.rpm = max(1, rpm)
        self._clock = clock
        self._sleep = sleep
        self._starts: deque[float] = deque()

    def wait(self) -> float:
        now = self._clock()
        while self._starts and now - self._starts[0] >= 60.0:
            self._starts.popleft()
        waited = 0.0
        if len(self._starts) >= self.rpm:
            waited = 60.0 - (now - self._starts[0]) + 0.05
            self._sleep(waited)
            now = self._clock()
            while self._starts and now - self._starts[0] >= 60.0:
                self._starts.popleft()
        self._starts.append(now)
        return waited


class DailyCounter:
    """Counts provider calls per UTC day in a small JSON file so runs can stop before the quota does."""

    def __init__(self, path: Path, limit: int):
        self.path = path
        self.limit = limit
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, int]:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {}

    @property
    def today(self) -> str:
        return date.today().isoformat()

    def used(self) -> int:
        return self._load().get(self.today, 0)

    def remaining(self) -> int:
        return max(0, self.limit - self.used())

    def increment(self) -> int:
        counts = self._load()
        counts[self.today] = counts.get(self.today, 0) + 1
        self.path.write_text(json.dumps(counts, indent=2), encoding="utf-8")
        return counts[self.today]
