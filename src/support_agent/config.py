"""Paths and settings. One dataclass, optionally overridden by `settings.yaml` at the repo root."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    raw: Path = ROOT / "data" / "raw"
    interim: Path = ROOT / "data" / "interim"
    processed: Path = ROOT / "data" / "processed"
    golden: Path = ROOT / "data" / "golden"
    outputs: Path = ROOT / "outputs"
    runs: Path = ROOT / "outputs" / "runs"
    judge: Path = ROOT / "outputs" / "judge"
    human: Path = ROOT / "outputs" / "human"
    results: Path = ROOT / "outputs" / "results"
    report: Path = ROOT / "report"
    figures: Path = ROOT / "report" / "figures"
    taxonomy_dir: Path = ROOT / "src" / "support_agent" / "taxonomy"
    cache_db: Path = ROOT / "outputs" / "llm_cache.sqlite"

    @property
    def twcs_csv(self) -> Path:
        return self.raw / "twcs.csv"

    @property
    def threads_all(self) -> Path:
        return self.interim / "threads_all.parquet"


@dataclass
class Settings:
    brand: str = ""                      # set after the brand-selection checkpoint
    seed: int = 42
    # models
    local_model: str = "qwen3:4b-instruct"
    judge_model: str = "gemini-3.8-flash"
    gemini_agent_model: str = "gemini-3.1-flash-lite"   # the hosted-model comparison row; must differ from the judge
    gemini_rpm: int = 10                 # free-tier default; raise if AI Studio shows more
    # retrieval / agent
    retrieval_k: int = 5
    min_retrieval_similarity: float = 0.0   # "no relevant resolution" threshold, tuned on dev
    max_reply_chars: int = 280
    # golden set
    n_test_random: int = 120             # Part A
    n_test_targeted: int = 80            # Part B
    n_dev: int = 50
    # escalation cost matrix
    cost_missed_hard: float = 10.0
    cost_missed_soft: float = 3.0
    cost_unnecessary_escalation: float = 1.0
    paths: Paths = field(default_factory=Paths)


def load_settings(path: Path | None = None) -> Settings:
    """Defaults, overridden by keys in settings.yaml (if present)."""
    settings = Settings()
    path = path or (ROOT / "settings.yaml")
    if path.exists():
        overrides = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        known = {f.name for f in dataclasses.fields(Settings)} - {"paths"}
        unknown = set(overrides) - known
        if unknown:
            raise KeyError(f"unknown settings in {path.name}: {sorted(unknown)}")
        for key, value in overrides.items():
            setattr(settings, key, value)
    return settings
