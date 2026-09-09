"""Command-line entry point: `python -m support_agent.cli <command>` (or `support-agent <command>`)."""
from __future__ import annotations

import argparse
import logging
import sys

from .config import load_settings

LATER = {
    "prepare": "clean, language-id, dedup and time-split the chosen brand's threads",
    "taxonomy": "cluster pre-cutoff messages and propose intents for the taxonomy",
    "sample": "draw the golden-set candidates (Part A / Part B / dev)",
    "label": "blind labelling CLI for the golden set",
    "index": "build the TF-IDF retrieval index over the historical corpus",
    "run": "run an agent system over the test set",
    "baselines": "run the trivial and simple baselines",
    "judge": "LLM-as-judge runs",
    "rate": "human rating CLI for the judge-agreement study",
    "eval": "recompute every metric from committed outputs",
    "figures": "render report figures",
    "docx": "render REPORT.md to report/report.docx",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="support-agent", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("data", help="download twcs.csv from Kaggle (needs ~/.kaggle/kaggle.json)")
    s.add_argument("--force", action="store_true", help="re-download even if present")

    s = sub.add_parser("threads", help="reconstruct threads -> data/interim/threads_all.parquet")
    s.add_argument("--nrows", type=int, default=None, help="debug: only read the first N rows")

    s = sub.add_parser("profile", help="profile the largest brands -> outputs/results/brand_profile.{csv,md}")
    s.add_argument("--top", type=int, default=15)

    s = sub.add_parser("smoke", help="run a handful of sample tweets through a model (no retrieval) to check JSON validity and latency")
    s.add_argument("--provider", choices=["local", "gemini"], default="local")
    s.add_argument("--model", default=None, help="override the configured model id")
    s.add_argument("--n", type=int, default=6)
    s.add_argument("--taxonomy", default="example", help="'example' or a path to a taxonomy yaml")

    for name, help_text in LATER.items():
        sub.add_parser(name, help=help_text + " (not implemented yet)")
    return p


SMOKE_TWEETS = [
    "@SpotifyCares my app crashes every time I open a playlist on my iPhone, been like this since the update",
    "@SpotifyCares I was charged twice for premium this month?? I want a refund",
    "@SpotifyCares someone is using my account, songs I never played are in my history. I think I've been hacked",
    "@SpotifyCares bring back the old shuffle, the new one plays the same 10 songs",
    "@SpotifyCares songs keep skipping in offline mode on android",
    "@SpotifyCares this is the third time I'm asking, I want to speak to a real person",
    "@SpotifyCares 😩😩",
    "@SpotifyCares thanks for the quick fix yesterday, you guys are great",
]


def cmd_smoke(args, settings) -> None:
    import statistics
    from pathlib import Path

    from .agent.pipeline import Agent, AgentConfig
    from .llm import make_client
    from .taxonomy import load_taxonomy

    path = settings.paths.taxonomy_dir / "taxonomy.example.yaml" if args.taxonomy == "example" else Path(args.taxonomy)
    taxonomy = load_taxonomy(path)
    llm = make_client(args.provider, settings, model=args.model)
    agent = Agent(llm, taxonomy, AgentConfig(system=f"smoke:{llm.model}", use_retrieval=False))
    print(f"model={llm.model} provider={llm.provider} prompt={agent.system_prompt.count(' ')} words\n")
    rows = []
    for i, text in enumerate(SMOKE_TWEETS[: args.n]):
        row = agent.handle({"item_id": f"S{i}", "text": text, "lang": "en"})
        rows.append(row)
        flag = "PARSE-FAIL" if row["parse_failed"] else ("repaired" if row["repaired"] else "ok")
        print(f"[{i}] {text}\n    -> {row['intent']} ({row['intent_confidence']:.2f}) | {row['decision']} [{row['reason_code']}]"
              f" | {row['latency_ms']} ms | {flag}{' cached' if row['cached'] else ''}"
              f"\n    reply: {row['reply']!r}\n    reason: {row['reason']}"
              + (f"\n    violations: {row['violations']}" if row["violations"] else ""))
    fresh = [r["latency_ms"] for r in rows if not r["cached"]]
    print(f"\nparse failures: {sum(r['parse_failed'] for r in rows)}/{len(rows)}; repaired: {sum(r['repaired'] for r in rows)}; "
          f"median latency (uncached): {statistics.median(fresh) if fresh else float('nan'):.0f} ms; "
          f"draft violations: {sum(bool(r['violations']) for r in rows)}")


def cmd_data(args, settings) -> None:
    from .data.download import download

    csv = download(settings.paths.raw, force=args.force)
    print(f"dataset at {csv}")


def cmd_threads(args, settings) -> None:
    from .data.threads import build_and_save

    threads = build_and_save(settings.paths.twcs_csv, settings.paths.threads_all, nrows=args.nrows)
    roots = threads["root_inbound"].sum()
    print(f"{len(threads):,} threads; {roots:,} customer-rooted; "
          f"{threads['orphan_root'].sum():,} orphan roots; {threads['has_brand_reply'].mean():.1%} got a brand reply")


def cmd_profile(args, settings) -> None:
    import pandas as pd

    from .data.profile_brands import profile_brands, to_markdown

    threads = pd.read_parquet(settings.paths.threads_all)
    profile = profile_brands(threads, top_n=args.top, seed=settings.seed)
    settings.paths.results.mkdir(parents=True, exist_ok=True)
    profile.to_csv(settings.paths.results / "brand_profile.csv", index=False)
    md = to_markdown(profile)
    (settings.paths.results / "brand_profile.md").write_text(md, encoding="utf-8")
    print(md)


COMMANDS = {"data": cmd_data, "threads": cmd_threads, "profile": cmd_profile, "smoke": cmd_smoke}


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles and emoji-laden tweets
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    handler = COMMANDS.get(args.cmd)
    if handler is None:
        print(f"`{args.cmd}` is not implemented yet: {LATER[args.cmd]}", file=sys.stderr)
        return 2
    handler(args, settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
