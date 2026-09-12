"""Blind, keyboard-driven labelling of golden-set candidates.

Shows the tweet plus the frozen taxonomy and policy — never a model output, cluster id, sampling part or
the brand's real reply. Every label is appended immediately (resume-safe) with the seconds it took.
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from ..agent.schemas import HARD_REASONS, SOFT_REASONS
from ..llm.batch import read_jsonl
from ..taxonomy import Taxonomy

LABELLER_REASONS = HARD_REASONS + SOFT_REASONS + ("non_english", "no_actionable_content")
FLAG_KEYS = {"a": "ambiguous", "m": "multi_intent", "n": "noise_or_spam", "i": "image_only",
             "r": "reply_to_other_customer", "p": "pii_present", "s": "sarcasm"}

COMPACT_HELP = ("one code per tweet:  <intent>[/<secondary>][e<reason>][a<0-2>][c<1-3>][f<flags>] [n:<note>]   "
                "e.g.  1  ·  5e2  ·  3/9a1c2  ·  1fam n:two issues   |   x = exclude · ? = sheet · q = quit")

_CODE_RE = re.compile(r"^(?P<intent>\d+)(?:/(?P<secondary>\d+))?(?P<rest>[a-z0-9\s]*)$", re.I)
_TOKEN_RE = re.compile(r"e(?P<reason>\d)|a(?P<anger>[0-2])|c(?P<conf>[1-3])|f(?P<flags>[amnirps]+)", re.I)


def parse_compact_code(code: str, n_intents: int) -> dict:
    """Turn '5e2a1c2fam n:note' into label fields. Raises ValueError with a helpful message on bad input.

    Grammar: intent number; optional /secondary; then any of e<reason 1-7>, a<anger 0-2>, c<confidence 1-3>,
    f<flag letters>; an optional 'n:' starts the free-text note. Whitespace is ignored before the note.
    """
    code, _, note = code.partition("n:")
    code = code.strip()
    m = _CODE_RE.match(code)
    if not m:
        raise ValueError("start with the intent number, e.g. 1, 5e2, 3/9a1c2")
    intent = int(m.group("intent"))
    if not 1 <= intent <= n_intents:
        raise ValueError(f"intent must be 1-{n_intents}")
    secondary = int(m.group("secondary")) if m.group("secondary") else None
    if secondary is not None and (not 1 <= secondary <= n_intents or secondary == intent):
        raise ValueError(f"secondary must be 1-{n_intents} and differ from the primary")
    rest = re.sub(r"\s+", "", m.group("rest"))
    fields = {"intent": intent, "secondary": secondary, "reason": None, "anger": 0, "confidence": 3, "flags": []}
    pos = 0
    while pos < len(rest):
        t = _TOKEN_RE.match(rest, pos)
        if not t:
            raise ValueError(f"cannot read {rest[pos:]!r}: use e<1-7> a<0-2> c<1-3> f<{''.join(FLAG_KEYS)}>")
        if t.group("reason"):
            r = int(t.group("reason"))
            if not 1 <= r <= len(LABELLER_REASONS):
                raise ValueError(f"reason must be 1-{len(LABELLER_REASONS)}")
            fields["reason"] = LABELLER_REASONS[r - 1]
        elif t.group("anger"):
            fields["anger"] = int(t.group("anger"))
        elif t.group("conf"):
            fields["confidence"] = int(t.group("conf"))
        elif t.group("flags"):
            fields["flags"] = [FLAG_KEYS[ch] for ch in dict.fromkeys(t.group("flags").lower())]
        pos = t.end()
    fields["note"] = note.strip() or None
    return fields


def build_label_row(taxonomy: Taxonomy, cand: dict, fields: dict, round_no: int, seconds: float, labeller: str = "author",
                    extra: dict | None = None) -> dict:
    """The stored label row for parsed compact-code fields. `labeller` says who decided: author | assistant."""
    intents = taxonomy.intents
    intent_id = intents[fields["intent"] - 1].id
    secondary_id = intents[fields["secondary"] - 1].id if fields.get("secondary") else None
    escalate = fields.get("reason") is not None
    row = {
        "item_id": cand["item_id"], "root_id": cand.get("root_id"), "round": round_no, "taxonomy_version": taxonomy.version,
        "labelled_at": datetime.now(timezone.utc).isoformat(), "labeller": labeller, "excluded": False, "exclusion_reason": None,
        "intent_primary": intent_id, "intent_secondary": secondary_id, "escalate": escalate,
        "reason_code": fields.get("reason") or "none", "anger_0_2": fields.get("anger", 0),
        "labeller_confidence_1_3": fields.get("confidence", 3), "quality_flags": fields.get("flags", []),
        "needs_reply": "noise_or_spam" not in fields.get("flags", []), "note": fields.get("note"),
        "label_seconds": round(seconds, 1),
    }
    if extra:
        row.update(extra)
    return row


class Quit(Exception):
    pass


class Labeller:
    def __init__(self, taxonomy: Taxonomy, candidates: list[dict], out_path: Path, round_no: int = 1,
                 input_fn: Callable[[str], str] = input, print_fn: Callable[[str], None] = print,
                 clock: Callable[[], float] = time.monotonic, compact: bool = True,
                 proposals: dict[str, dict] | None = None):
        self.taxonomy = taxonomy
        self.candidates = candidates
        self.out_path = out_path
        self.round_no = round_no
        self._input = input_fn
        self._print = print_fn
        self._clock = clock
        self.compact = compact
        self.proposals = proposals or {}     # item_id -> merged proposal row (model-assisted mode)
        self.intents = taxonomy.intents
        self._intent_number = {it.id: n for n, it in enumerate(self.intents, start=1)}

    # -- display -------------------------------------------------------------------------------------
    def cheat_sheet(self) -> str:
        lines = [f"=== {self.taxonomy.brand} — taxonomy v{self.taxonomy.version} ===", "INTENTS"]
        for n, it in enumerate(self.intents, start=1):
            lines.append(f"  {n:>2}. {it.id:<26} {it.definition}")
            for rule in it.edge_rules:
                lines.append(f"      · {rule}")
        lines.append("ESCALATION REASONS (only asked when you escalate)")
        for n, code in enumerate(LABELLER_REASONS, start=1):
            desc = next((r.description for r in self.taxonomy.reasons if r.code == code), "")
            tier = next((r.tier for r in self.taxonomy.reasons if r.code == code), "")
            lines.append(f"  {n:>2}. {code:<24} [{tier}] {desc}")
        lines.append("FLAGS: " + "  ".join(f"{k}={v}" for k, v in FLAG_KEYS.items()))
        if self.compact:
            lines.append("CODE: " + COMPACT_HELP)
            lines.append("      defaults: no secondary · auto (no e) · anger 0 · confidence 3 · no flags · no note")
            if self.proposals:
                lines.append("ASSISTED: two models propose a label; Enter = accept the shown code · type a code = override · "
                             "a ⚠ marks a model disagreement — look harder there")
        else:
            lines.append("KEYS: number = choose · enter = default/none · x = exclude item · ? = show this sheet · q = quit (progress is saved)")
        lines.append("")
        lines.append(self.taxonomy.policy_markdown())
        return "\n".join(lines)

    # -- main loop -----------------------------------------------------------------------------------
    def run(self, limit: int | None = None) -> int:
        done = {r["item_id"] for r in read_jsonl(self.out_path) if r.get("round") == self.round_no}
        todo = [c for c in self.candidates if c["item_id"] not in done]
        if limit is not None:
            todo = todo[:limit]
        self._print(self.cheat_sheet())
        self._print(f"\n{len(done)} already labelled in round {self.round_no}; {len(todo)} to go.\n")
        labelled = 0
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("a", encoding="utf-8") as f:
            for i, cand in enumerate(todo, start=1):
                try:
                    row = self.label_one(cand, progress=f"{i}/{len(todo)}")
                except Quit:
                    self._print("\nstopped; progress saved.")
                    break
                import json

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                labelled += 1
        return labelled

    def label_one(self, cand: dict, progress: str = "") -> dict:
        t0 = self._clock()
        self._print("\n" + "─" * 88)
        self._print(f"[{progress}] {cand['item_id']}   {str(cand.get('root_created_at', ''))[:16]}")
        self._print(f"  {cand['root_text']}")
        self._print("─" * 88)

        row = {
            "item_id": cand["item_id"], "root_id": cand.get("root_id"), "round": self.round_no,
            "taxonomy_version": self.taxonomy.version, "labelled_at": datetime.now(timezone.utc).isoformat(),
            "excluded": False, "exclusion_reason": None,
        }
        if self.compact:
            proposal = self.proposals.get(cand["item_id"])
            default_code = self._show_proposal(proposal) if proposal else None
            fields = self._ask_compact(default_code)
            if fields is None:  # excluded
                row.update(excluded=True, exclusion_reason=self._ask_text("exclusion reason"),
                           label_seconds=round(self._clock() - t0, 1))
                return row
            extra = None
            if proposal:
                extra = {"assisted": True, "accepted_proposal": fields.get("accepted_default", False),
                         "proposal_a": proposal["a"], "proposal_b": proposal["b"],
                         "models_agree_intent": proposal["agree_intent"], "models_agree_escalate": proposal["agree_escalate"]}
            return build_label_row(self.taxonomy, cand, fields, self.round_no, self._clock() - t0, labeller="author", extra=extra)

        primary = self._ask_intent("intent # (primary)", allow_exclude=True)
        if primary is None:  # excluded
            row.update(excluded=True, exclusion_reason=self._ask_text("exclusion reason"),
                       label_seconds=round(self._clock() - t0, 1))
            return row
        secondary = self._ask_intent("secondary intent # (enter = none)", allow_none=True)
        escalate = self._ask_yes_no("escalate to a human? [y/n]")
        reason = self._ask_reason() if escalate else "none"
        anger = self._ask_int("anger 0=calm 1=annoyed 2=furious", 0, 2, default=0)
        confidence = self._ask_int("your confidence 1=guess 2=fairly sure 3=sure", 1, 3, default=3)
        flags = self._ask_flags()
        note = self._ask_text("note (enter = none)", allow_empty=True)
        row.update(
            intent_primary=primary, intent_secondary=secondary, escalate=escalate, reason_code=reason,
            anger_0_2=anger, labeller_confidence_1_3=confidence, quality_flags=flags,
            needs_reply="noise_or_spam" not in flags, note=note or None,
            label_seconds=round(self._clock() - t0, 1),
        )
        return row

    # -- prompts -------------------------------------------------------------------------------------
    def _prompt(self, text: str) -> str:
        answer = self._input(f"  {text}: ").strip()
        if answer.lower() == "q":
            raise Quit()
        if answer == "?":
            self._print(self.cheat_sheet())
            return self._prompt(text)
        return answer

    def proposal_code(self, p: dict) -> str | None:
        """Compact code equivalent of one model's proposal (None if the proposal is unusable)."""
        if not p or p.get("intent") not in self._intent_number or p.get("escalate") is None:
            return None
        code = str(self._intent_number[p["intent"]])
        if p.get("intent_secondary") in self._intent_number:
            code += f"/{self._intent_number[p['intent_secondary']]}"
        if p["escalate"]:
            if p.get("reason_code") not in LABELLER_REASONS:
                return None
            code += f"e{LABELLER_REASONS.index(p['reason_code']) + 1}"
        if p.get("anger"):
            code += f"a{p['anger']}"
        return code

    def _show_proposal(self, proposal: dict) -> str | None:
        """Print both proposals; return the code offered as the Enter default (model A's), or None."""
        a, b = proposal["a"], proposal["b"]
        code_a, code_b = self.proposal_code(a), self.proposal_code(b)
        if code_a and code_a == code_b:
            self._print(f"  proposal: {code_a}   ({a['intent']}{', escalate ' + a['reason_code'] if a['escalate'] else ', auto'})  — both models agree")
            return code_a
        marks = []
        if not proposal["agree_intent"]:
            marks.append("intent")
        if not proposal["agree_escalate"]:
            marks.append("escalate")
        self._print(f"  ⚠ models disagree on {' & '.join(marks) or 'details'}:")
        self._print(f"    A {a['model']}: {code_a or '?'}  ({a['intent']}{', escalate ' + str(a['reason_code']) if a['escalate'] else ', auto'})")
        self._print(f"    B {b['model']}: {code_b or '?'}  ({b['intent']}{', escalate ' + str(b['reason_code']) if b['escalate'] else ', auto'})")
        return code_a

    def _ask_compact(self, default_code: str | None = None) -> dict | None:
        """One code per tweet; None means the item is excluded. Enter accepts `default_code` when one is offered."""
        while True:
            answer = self._prompt(f"code [enter = {default_code}]" if default_code else "code")
            if answer == "" and default_code:
                fields = parse_compact_code(default_code, len(self.intents))
                fields["accepted_default"] = True
                return fields
            if answer.lower() == "x":
                return None
            try:
                fields = parse_compact_code(answer, len(self.intents))
                fields["accepted_default"] = False
                return fields
            except ValueError as e:
                self._print(f"    ? {e}\n    {COMPACT_HELP}")

    def _ask_intent(self, text: str, allow_exclude: bool = False, allow_none: bool = False) -> str | None:
        while True:
            answer = self._prompt(text)
            if allow_none and answer == "":
                return None
            if allow_exclude and answer.lower() == "x":
                return None
            if answer.isdigit() and 1 <= int(answer) <= len(self.intents):
                return self.intents[int(answer) - 1].id
            if answer in self.taxonomy.intent_ids:
                return answer
            self._print(f"    ? enter 1-{len(self.intents)}" + (", x to exclude" if allow_exclude else "") + (", or enter for none" if allow_none else ""))

    def _ask_reason(self) -> str:
        while True:
            answer = self._prompt("reason # (see sheet)")
            if answer.isdigit() and 1 <= int(answer) <= len(LABELLER_REASONS):
                return LABELLER_REASONS[int(answer) - 1]
            if answer in LABELLER_REASONS:
                return answer
            self._print(f"    ? enter 1-{len(LABELLER_REASONS)}")

    def _ask_yes_no(self, text: str) -> bool:
        while True:
            answer = self._prompt(text).lower()
            if answer in ("y", "yes"):
                return True
            if answer in ("n", "no"):
                return False
            self._print("    ? y or n")

    def _ask_int(self, text: str, lo: int, hi: int, default: int) -> int:
        while True:
            answer = self._prompt(f"{text} (enter = {default})")
            if answer == "":
                return default
            if answer.isdigit() and lo <= int(answer) <= hi:
                return int(answer)
            self._print(f"    ? {lo}-{hi}")

    def _ask_flags(self) -> list[str]:
        while True:
            answer = self._prompt("flags (letters, enter = none)").lower().replace(" ", "").replace(",", "")
            if all(ch in FLAG_KEYS for ch in answer):
                return [FLAG_KEYS[ch] for ch in dict.fromkeys(answer)]
            self._print("    ? letters from: " + " ".join(f"{k}={v}" for k, v in FLAG_KEYS.items()))

    def _ask_text(self, text: str, allow_empty: bool = False) -> str:
        while True:
            answer = self._prompt(text)
            if answer or allow_empty:
                return answer
            self._print("    ? please type something")


def select_for_relabel(labels: list[dict], n: int, seed: int = 42) -> list[str]:
    """Item ids for the blind round-2 pass: a random subset of non-excluded round-1 labels."""
    import random

    ids = sorted({r["item_id"] for r in labels if r.get("round") == 1 and not r.get("excluded")})
    rng = random.Random(seed)
    rng.shuffle(ids)
    return sorted(ids[:n])


def self_agreement(round1: list[dict], round2: list[dict]) -> dict:
    """Cohen's kappa / agreement between the two blind passes, for intents and for the escalation call."""
    from .agreement import binary_agreement, kappa_with_ci

    r1 = {r["item_id"]: r for r in round1 if not r.get("excluded")}
    r2 = {r["item_id"]: r for r in round2 if not r.get("excluded")}
    common = sorted(set(r1) & set(r2))
    if not common:
        return {"n": 0}
    intents = kappa_with_ci([r1[i]["intent_primary"] for i in common], [r2[i]["intent_primary"] for i in common])
    escalate = binary_agreement([r1[i]["escalate"] for i in common], [r2[i]["escalate"] for i in common])
    hard = [i for i in common if r1[i]["reason_code"] in HARD_REASONS or r2[i]["reason_code"] in HARD_REASONS]
    soft = [i for i in common if r1[i]["reason_code"] in SOFT_REASONS or r2[i]["reason_code"] in SOFT_REASONS]
    return {
        "n": len(common),
        "intent": {**intents, "percent_agreement": sum(r1[i]["intent_primary"] == r2[i]["intent_primary"] for i in common) / len(common)},
        "escalate": escalate,
        "escalate_hard_items": {"n": len(hard), "percent_agreement": (sum(r1[i]["escalate"] == r2[i]["escalate"] for i in hard) / len(hard)) if hard else float("nan")},
        "escalate_soft_items": {"n": len(soft), "percent_agreement": (sum(r1[i]["escalate"] == r2[i]["escalate"] for i in soft) / len(soft)) if soft else float("nan")},
    }
