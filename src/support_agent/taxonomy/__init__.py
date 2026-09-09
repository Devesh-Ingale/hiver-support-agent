"""The frozen artefacts the whole evaluation hangs on: intent taxonomy and escalation policy.

Both live in `taxonomy.yaml` next to this file (versioned, committed before labelling starts). The same
definitions are shown to the human labeller and given verbatim to the model, so the metric measures
"does the system apply this policy the way a person would".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..agent.checks import HardRules

DEFAULT_PATH = Path(__file__).with_name("taxonomy.yaml")


@dataclass
class Intent:
    id: str
    name: str
    definition: str
    examples: list[str] = field(default_factory=list)
    edge_rules: list[str] = field(default_factory=list)
    typical_action: str = ""      # what the brand usually does for this intent (auto-handleable or not)


@dataclass
class Reason:
    code: str
    tier: str                     # hard | soft | routing
    description: str
    patterns: list[str] = field(default_factory=list)   # regexes for hard rules; empty for judgment calls


@dataclass
class Taxonomy:
    brand: str
    version: str
    intents: list[Intent]
    reasons: list[Reason]
    auto_ok: list[str]            # what a grounded public first reply may do
    never: list[str]              # what a public reply must never do
    good_reply: list[str]         # what "good" means for this brand, in the labeller's and judge's words

    @property
    def intent_ids(self) -> list[str]:
        return [i.id for i in self.intents]

    @property
    def other_intent(self) -> str:
        for i in self.intents:
            if i.id.startswith("other"):
                return i.id
        return self.intents[-1].id

    def hard_rules(self) -> HardRules:
        return HardRules.from_config({r.code: r.patterns for r in self.reasons if r.tier == "hard" and r.patterns})

    def reasons_by_tier(self, tier: str) -> list[Reason]:
        return [r for r in self.reasons if r.tier == tier]

    # -- rendering for prompts and the labeller -------------------------------------------------------
    def intents_markdown(self, with_examples: bool = True) -> str:
        lines = []
        for i in self.intents:
            lines.append(f"- `{i.id}` — **{i.name}**: {i.definition}")
            for rule in i.edge_rules:
                lines.append(f"    - rule: {rule}")
            if with_examples and i.examples:
                lines.append("    - e.g. " + " | ".join(f'"{e}"' for e in i.examples[:3]))
        return "\n".join(lines)

    def policy_markdown(self) -> str:
        parts = ["**A public first reply is GOOD for this brand when it:**"]
        parts += [f"- {g}" for g in self.good_reply]
        parts.append("\n**It must NEVER:**")
        parts += [f"- {n}" for n in self.never]
        parts.append("\n**AUTO-HANDLE (send a grounded public reply) when:**")
        parts += [f"- {a}" for a in self.auto_ok]
        parts.append("\n**ESCALATE to a human — HARD (always, no exceptions):**")
        parts += [f"- `{r.code}`: {r.description}" for r in self.reasons_by_tier("hard")]
        parts.append("\n**ESCALATE — SOFT (judgment call):**")
        parts += [f"- `{r.code}`: {r.description}" for r in self.reasons_by_tier("soft")]
        parts.append("\n**ROUTE to a human for non-content reasons:**")
        parts += [f"- `{r.code}`: {r.description}" for r in self.reasons_by_tier("routing")]
        return "\n".join(parts)


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    path = path or DEFAULT_PATH
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    intents = [Intent(**i) for i in raw["intents"]]
    reasons = [Reason(**r) for r in raw["policy"]["reasons"]]
    ids = [i.id for i in intents]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate intent ids in taxonomy")
    if not any(i.startswith("other") for i in ids):
        raise ValueError("taxonomy needs an `other`/`other_unclear` intent")
    return Taxonomy(
        brand=raw["brand"], version=str(raw["version"]), intents=intents, reasons=reasons,
        auto_ok=raw["policy"].get("auto_ok", []), never=raw["policy"].get("never", []),
        good_reply=raw["policy"].get("good_reply", []),
    )
