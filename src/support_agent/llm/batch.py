"""Resumable JSONL batch runner: every finished item is appended immediately, reruns skip done items."""
from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .base import QuotaExhausted

log = logging.getLogger(__name__)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_batch(items: list[dict[str, Any]], fn: Callable[[dict[str, Any]], dict[str, Any]], out_path: Path,
              id_key: str = "item_id", desc: str = "items") -> list[dict[str, Any]]:
    """Apply fn to every item not already present in out_path; append results as they complete.

    fn must return a dict containing id_key. A QuotaExhausted error stops the run cleanly (the file is
    consistent and the next invocation resumes); any other exception is recorded as an error row so a
    single bad item cannot kill a long run.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {row[id_key] for row in read_jsonl(out_path)}
    todo = [it for it in items if it[id_key] not in done]
    log.info("%s: %d done, %d to do -> %s", desc, len(done), len(todo), out_path)
    with out_path.open("a", encoding="utf-8") as f:
        for item in tqdm(todo, desc=desc, unit="item"):
            try:
                row = fn(item)
            except QuotaExhausted as e:
                log.warning("stopping: %s", e)
                break
            except Exception as e:  # noqa: BLE001 — keep the batch alive, record the failure
                log.exception("item %s failed", item[id_key])
                row = {id_key: item[id_key], "error": f"{type(e).__name__}: {e}"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    return read_jsonl(out_path)
