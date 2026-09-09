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

    for name, help_text in LATER.items():
        sub.add_parser(name, help=help_text + " (not implemented yet)")
    return p


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


COMMANDS = {"data": cmd_data, "threads": cmd_threads, "profile": cmd_profile}


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
