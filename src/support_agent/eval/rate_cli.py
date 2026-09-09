"""Human rating CLI for the judge-agreement study — same rubric as the LLM judge, blinded to system.

Two modes, both resume-safe and order-randomised:
- pairwise: main vs simple replies for the same tweet, shown as A/B in a random order -> outputs/human/pairs.jsonl
- absolute: replies stratified across systems, one at a time -> outputs/human/absolute.jsonl
The rater must finish before looking at any judge output (the report states this).
"""
from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from ..agent.prompts import render_evidence
from ..eval.judge import CHECK_FIELDS, evidence_from_dicts
from ..llm.batch import read_jsonl

CHECK_QUESTIONS = {
    "unsupported_content": "does the reply state any step/link/policy/promise the evidence does NOT support? [y/n]",
    "addresses_problem": "does it engage with the customer's actual problem? [y/n]",
    "unsafe_or_overpromising": "does it ask for sensitive data publicly, promise a refund/fix/timeline, or is it rude? [y/n]",
    "tweet_valid": "is it a valid tweet (<=280 chars, no @handles/sign-offs, no URLs not in evidence)? [y/n]",
    "decision_reason_consistent": "does the stated reason follow from the tweet and match the decision? [y/n]",
}


class Quit(Exception):
    pass


def plan_pairs(main_rows: list[dict], other_rows: list[dict], n: int, seed: int) -> list[dict]:
    """Items where both systems produced a non-empty reply; random subset; random A/B assignment."""
    other = {r["item_id"]: r for r in other_rows if r.get("reply")}
    cands = [r for r in main_rows if r.get("reply") and r["item_id"] in other]
    rng = random.Random(seed)
    rng.shuffle(cands)
    plan = []
    for r in cands[:n]:
        flip = rng.random() < 0.5
        a, b = (other[r["item_id"]], r) if flip else (r, other[r["item_id"]])
        plan.append({"item_id": r["item_id"], "a_system": a["system"], "b_system": b["system"], "reply_a": a["reply"], "reply_b": b["reply"]})
    return plan


def plan_absolute(rows_by_system: dict[str, list[dict]], n_per_system: int, seed: int) -> list[dict]:
    """n items per system, no item repeated across systems, shuffled so the rater cannot infer the source."""
    rng = random.Random(seed)
    used: set[str] = set()
    plan = []
    for system, rows in rows_by_system.items():
        pool = [r for r in rows if r.get("reply") and r["item_id"] not in used]
        rng.shuffle(pool)
        for r in pool[:n_per_system]:
            used.add(r["item_id"])
            plan.append({"item_id": r["item_id"], "system": system, "reply": r["reply"], "decision": r.get("decision"), "reason": r.get("reason")})
    rng.shuffle(plan)
    for i, p in enumerate(plan, start=1):
        p["rating_id"] = f"R{i:03d}"
    return plan


class Rater:
    def __init__(self, brand: str, items: dict[str, dict], evidence_by_item: dict[str, list[dict]], out_path: Path,
                 input_fn: Callable[[str], str] = input, print_fn: Callable[[str], None] = print,
                 clock: Callable[[], float] = time.monotonic):
        self.brand = brand
        self.items = items
        self.evidence_by_item = evidence_by_item
        self.out_path = out_path
        self._input, self._print, self._clock = input_fn, print_fn, clock

    def _prompt(self, text: str) -> str:
        answer = self._input(f"  {text}: ").strip()
        if answer.lower() == "q":
            raise Quit()
        return answer

    def _yes_no(self, text: str) -> bool:
        while True:
            a = self._prompt(text).lower()
            if a in ("y", "yes"):
                return True
            if a in ("n", "no"):
                return False
            self._print("    ? y or n")

    def _choice(self, text: str, options: tuple[str, ...]) -> str:
        while True:
            a = self._prompt(text)
            for opt in options:
                if a.lower() == opt.lower():
                    return opt
            self._print(f"    ? one of {options}")

    def _int(self, text: str, lo: int, hi: int) -> int:
        while True:
            a = self._prompt(text)
            if a.isdigit() and lo <= int(a) <= hi:
                return int(a)
            self._print(f"    ? {lo}-{hi}")

    def _show_context(self, item_id: str, header: str) -> None:
        item = self.items[item_id]
        self._print("\n" + "═" * 88)
        self._print(f"{header}")
        self._print(f"CUSTOMER: {item['root_text']}")
        self._print("─" * 88)
        self._print("EVIDENCE (how the brand handled similar cases):")
        self._print(render_evidence(evidence_from_dicts(self.evidence_by_item.get(item_id, [])), self.brand))
        self._print("─" * 88)

    def run_pairs(self, plan: list[dict]) -> int:
        done = {(r["item_id"]) for r in read_jsonl(self.out_path)}
        todo = [p for p in plan if p["item_id"] not in done]
        self._print(f"PAIRWISE: {len(done)} done, {len(todo)} to go. Pick the reply a strong {self.brand} agent would rather send. q = quit.")
        n = 0
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("a", encoding="utf-8") as f:
            for i, p in enumerate(todo, start=1):
                t0 = self._clock()
                self._show_context(p["item_id"], f"[pair {i}/{len(todo)}]")
                self._print(f"REPLY A: {p['reply_a']}\n\nREPLY B: {p['reply_b']}\n")
                try:
                    pref = self._choice("preferred (A / B / tie)", ("A", "B", "tie"))
                except Quit:
                    self._print("stopped; progress saved.")
                    break
                preferred_system = {"A": p["a_system"], "B": p["b_system"], "tie": "tie"}[pref]
                f.write(json.dumps({**p, "preferred": pref, "preferred_system": preferred_system,
                                    "seconds": round(self._clock() - t0, 1), "rated_at": datetime.now(timezone.utc).isoformat()},
                                   ensure_ascii=False) + "\n")
                f.flush()
                n += 1
        return n

    def run_absolute(self, plan: list[dict]) -> int:
        done = {r["rating_id"] for r in read_jsonl(self.out_path)}
        todo = [p for p in plan if p["rating_id"] not in done]
        self._print(f"ABSOLUTE: {len(done)} done, {len(todo)} to go. Same checks as the judge. q = quit.")
        n = 0
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("a", encoding="utf-8") as f:
            for i, p in enumerate(todo, start=1):
                t0 = self._clock()
                self._show_context(p["item_id"], f"[reply {i}/{len(todo)}]")
                self._print(f"DRAFT REPLY: {p['reply']}\nDECISION: {p.get('decision') or 'n/a'} — reason: {p.get('reason') or 'n/a'}\n")
                try:
                    answers = {k: self._yes_no(q) for k, q in CHECK_QUESTIONS.items()}
                    overall = self._int("overall 1-5 (1 harmful/useless, 3 acceptable with edits, 5 strong human agent)", 1, 5)
                    sendable = self._yes_no("would a brand agent send this with at most a light edit? [y/n]")
                except Quit:
                    self._print("stopped; progress saved.")
                    break
                row = {**p, **answers, "overall_1_5": overall, "sendable": sendable,
                       "seconds": round(self._clock() - t0, 1), "rated_at": datetime.now(timezone.utc).isoformat()}
                assert all(k in row for k in CHECK_FIELDS)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
        return n
