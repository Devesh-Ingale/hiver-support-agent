"""SQLite response cache. Nothing is ever paid for, or generated, twice."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ResponseCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            " key TEXT PRIMARY KEY, provider TEXT, model TEXT, created_at TEXT, response TEXT)"
        )
        self._conn.commit()

    @staticmethod
    def key(**request: Any) -> str:
        payload = json.dumps(request, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, provider: str, model: str, response: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO responses (key, provider, model, created_at, response) VALUES (?, ?, ?, ?, ?)",
            (key, provider, model, datetime.now(timezone.utc).isoformat(), json.dumps(response, ensure_ascii=False)),
        )
        self._conn.commit()

    def stats(self) -> dict[str, int]:
        rows = self._conn.execute("SELECT provider || ':' || model, COUNT(*) FROM responses GROUP BY 1").fetchall()
        return {name: n for name, n in rows}

    def close(self) -> None:
        self._conn.close()
