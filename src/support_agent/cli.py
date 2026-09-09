"""Command-line entry point: `python -m support_agent.cli <command>` (or `support-agent <command>`).

Data stage:   data -> threads -> profile -> prepare --brand X -> taxonomy -> sample -> label -> index
Model stage:  smoke | run --system main|no_rag|main_gemini [--split dev|test] | baselines
Proof stage:  judge | rate | eval | figures | docx
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import ROOT, load_settings

log = logging.getLogger(__name__)

LATER = {
    "judge": "LLM-as-judge runs (absolute, pairwise, validation, consistency)",
    "rate": "human rating CLI for the judge-agreement study",
    "eval": "recompute every metric from committed outputs",
    "figures": "render report figures",
    "docx": "render REPORT.md to report/report.docx",
}

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


# ----------------------------------------------------------------------------------------------------------
# parser
# ----------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="support-agent", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("data", help="download twcs.csv from Kaggle (needs ~/.kaggle/kaggle.json)")
    s.add_argument("--force", action="store_true")

    s = sub.add_parser("threads", help="reconstruct threads -> data/interim/threads_all.parquet")
    s.add_argument("--nrows", type=int, default=None, help="debug: only read the first N rows")

    s = sub.add_parser("profile", help="profile the largest brands -> outputs/results/brand_profile.{csv,md}")
    s.add_argument("--top", type=int, default=15)

    s = sub.add_parser("prepare", help="clean, language-id, near-dup and time-split one brand -> data/processed/")
    s.add_argument("--brand", required=True)
    s.add_argument("--window-days", type=int, default=14, help="golden window length before the tail buffer")
    s.add_argument("--tail-buffer-days", type=int, default=2)

    s = sub.add_parser("taxonomy", help="cluster pre-cutoff messages -> outputs/results/taxonomy_clusters.md (+ LLM proposal)")
    s.add_argument("--k", type=int, default=20)
    s.add_argument("--sample", type=int, default=3000)
    s.add_argument("--propose", action="store_true", help="ask the local model to draft intent names/definitions")

    s = sub.add_parser("sample", help="draw golden-set candidates -> data/golden/candidates.jsonl + sampling_note.md")
    s.add_argument("--seed", type=int, default=None)

    s = sub.add_parser("label", help="blind labelling CLI -> data/golden/labels_round{N}.jsonl")
    s.add_argument("--round", type=int, default=1, choices=[1, 2])
    s.add_argument("--relabel", type=int, default=40, help="round 2: how many round-1 items to re-label blind")
    s.add_argument("--limit", type=int, default=None, help="stop after N items (pilot)")
    s.add_argument("--parts", default="A,B,dev", help="which sampling parts to label (round 1)")
    s.add_argument("--finalize", action="store_true", help="join labels with candidates -> data/golden/{test,dev}.jsonl")

    s = sub.add_parser("index", help="build the TF-IDF retrieval index over the historical corpus")

    s = sub.add_parser("smoke", help="run sample tweets through a model (no retrieval) to check JSON validity and latency")
    s.add_argument("--provider", choices=["local", "gemini"], default="local")
    s.add_argument("--model", default=None)
    s.add_argument("--n", type=int, default=6)
    s.add_argument("--taxonomy", default="example", help="'example', 'real', or a path to a taxonomy yaml")

    s = sub.add_parser("run", help="run an agent system over the dev or test set -> outputs/runs/")
    s.add_argument("--system", required=True, choices=["main", "no_rag", "main_gemini"])
    s.add_argument("--split", default="test", choices=["test", "dev"])
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--seed", type=int, default=None, help="override the sampling seed (variance re-run)")
    s.add_argument("--tag", default="", help="suffix for the output file, e.g. seed7")

    s = sub.add_parser("baselines", help="trivial + simple baselines and the human reference rows -> outputs/runs/")
    s.add_argument("--split", default="test", choices=["test", "dev"])

    for name, help_text in LATER.items():
        sub.add_parser(name, help=help_text + " (not implemented yet)")
    return p


# ----------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------
def _require(path: Path, hint: str) -> Path:
    if not path.exists():
        sys.exit(f"missing {path}\n  -> {hint}")
    return path


def _processed(settings, name: str) -> Path:
    return settings.paths.processed / f"{settings.brand}_{name}"


def _load_taxonomy(settings, which: str = "real"):
    from .taxonomy import DEFAULT_PATH, load_taxonomy

    if which == "example":
        return load_taxonomy(settings.paths.taxonomy_dir / "taxonomy.example.yaml")
    if which == "real":
        return load_taxonomy(_require(DEFAULT_PATH, "write src/support_agent/taxonomy/taxonomy.yaml (see taxonomy.example.yaml)"))
    return load_taxonomy(Path(which))


def _load_split(settings, split: str) -> list[dict]:
    from .llm.batch import read_jsonl

    path = _require(settings.paths.golden / f"{split}.jsonl", "run `label --finalize` after labelling")
    return read_jsonl(path)


def _write_settings_brand(brand: str) -> None:
    import yaml

    path = ROOT / "settings.yaml"
    current = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    current = current or {}
    current["brand"] = brand
    path.write_text(yaml.safe_dump(current, sort_keys=True), encoding="utf-8")


# ----------------------------------------------------------------------------------------------------------
# data stage
# ----------------------------------------------------------------------------------------------------------
def cmd_data(args, settings) -> None:
    from .data.download import download

    print(f"dataset at {download(settings.paths.raw, force=args.force)}")


def cmd_threads(args, settings) -> None:
    from .data.threads import build_and_save

    threads = build_and_save(_require(settings.paths.twcs_csv, "run `data` first"), settings.paths.threads_all, nrows=args.nrows)
    print(f"{len(threads):,} threads; {int(threads['root_inbound'].sum()):,} customer-rooted; "
          f"{int(threads['orphan_root'].sum()):,} orphan roots; {threads['has_brand_reply'].mean():.1%} got a brand reply")


def cmd_profile(args, settings) -> None:
    import pandas as pd

    from .data.profile_brands import profile_brands, to_markdown

    threads = pd.read_parquet(_require(settings.paths.threads_all, "run `threads` first"))
    profile = profile_brands(threads, top_n=args.top, seed=settings.seed)
    settings.paths.results.mkdir(parents=True, exist_ok=True)
    profile.to_csv(settings.paths.results / "brand_profile.csv", index=False)
    md = to_markdown(profile)
    (settings.paths.results / "brand_profile.md").write_text(md, encoding="utf-8")
    print(md)


def cmd_prepare(args, settings) -> None:
    import pandas as pd

    from .data.prepare import prepare_brand

    threads = pd.read_parquet(_require(settings.paths.threads_all, "run `threads` first"))
    corpus, pool, stats = prepare_brand(threads, args.brand, golden_window_days=args.window_days, tail_buffer_days=args.tail_buffer_days)
    settings.paths.processed.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(settings.paths.processed / f"{args.brand}_corpus.parquet", index=False)
    pool.to_parquet(settings.paths.processed / f"{args.brand}_pool.parquet", index=False)
    (settings.paths.processed / f"{args.brand}_split_stats.json").write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
    _write_settings_brand(args.brand)
    print(json.dumps(stats.to_dict(), indent=2))
    print(f"\nbrand set to {args.brand!r} in settings.yaml")


def cmd_taxonomy(args, settings) -> None:
    import pandas as pd

    from .taxonomy.induce import cluster_messages, clusters_markdown, propose_intents, sample_for_induction

    corpus = pd.read_parquet(_require(_processed(settings, "corpus.parquet"), "run `prepare --brand X` first"))
    messages = sample_for_induction(corpus, n=args.sample, seed=settings.seed)
    clusters = cluster_messages(messages, k=args.k, seed=settings.seed)
    md = clusters_markdown(clusters, settings.brand, len(messages))
    settings.paths.results.mkdir(parents=True, exist_ok=True)
    (settings.paths.results / "taxonomy_clusters.md").write_text(md, encoding="utf-8")
    print(md[:6000] + ("\n… (truncated; full sheet in outputs/results/taxonomy_clusters.md)" if len(md) > 6000 else ""))
    if args.propose:
        from .llm import make_client

        proposal = propose_intents(make_client("local", settings), md, settings.brand)
        (settings.paths.results / "taxonomy_proposal.md").write_text(proposal, encoding="utf-8")
        print("\n=== local model's proposal (outputs/results/taxonomy_proposal.md) ===\n" + proposal)


def cmd_sample(args, settings) -> None:
    import pandas as pd

    from .eval.golden import SampleConfig, candidate_records, sample_golden, sampling_note
    from .llm.batch import write_jsonl

    pool = pd.read_parquet(_require(_processed(settings, "pool.parquet"), "run `prepare --brand X` first"))
    split_stats = json.loads(_processed(settings, "split_stats.json").read_text(encoding="utf-8"))
    cfg = SampleConfig(n_random=settings.n_test_random, n_targeted=settings.n_test_targeted, n_dev=settings.n_dev,
                       seed=args.seed if args.seed is not None else settings.seed)
    sample, stats = sample_golden(pool, cfg)
    settings.paths.golden.mkdir(parents=True, exist_ok=True)
    write_jsonl(settings.paths.golden / "candidates.jsonl", candidate_records(sample))
    (settings.paths.golden / "sampling_note.md").write_text(sampling_note(stats, split_stats, cfg), encoding="utf-8")
    (settings.paths.golden / "sampling_stats.json").write_text(
        json.dumps({"pool_size": stats.pool_size, "excluded": stats.excluded, "eligible": stats.eligible, "counts": stats.counts}, indent=2),
        encoding="utf-8")
    print(json.dumps(stats.counts, indent=2))


def cmd_label(args, settings) -> None:
    from .eval.label_cli import Labeller, select_for_relabel, self_agreement
    from .llm.batch import read_jsonl, write_jsonl

    taxonomy = _load_taxonomy(settings, "real")
    candidates = read_jsonl(_require(settings.paths.golden / "candidates.jsonl", "run `sample` first"))
    r1_path = settings.paths.golden / "labels_round1.jsonl"

    if args.finalize:
        labels = {r["item_id"]: r for r in read_jsonl(_require(r1_path, "label round 1 first"))}
        by_part = {"test": [], "dev": []}
        skipped = 0
        for cand in candidates:
            lab = labels.get(cand["item_id"])
            if lab is None or lab.get("excluded"):
                skipped += lab is not None
                continue
            merged = {**cand, **{k: v for k, v in lab.items() if k not in ("item_id", "root_id")}}
            merged["lang"] = cand.get("lang")
            (by_part["dev"] if cand["part"] == "dev" else by_part["test"]).append(merged) if cand["part"] in ("A", "B", "dev") else None
        write_jsonl(settings.paths.golden / "test.jsonl", by_part["test"])
        write_jsonl(settings.paths.golden / "dev.jsonl", by_part["dev"])
        print(f"test: {len(by_part['test'])} items, dev: {len(by_part['dev'])} items, excluded during labelling: {skipped}")
        r2_path = settings.paths.golden / "labels_round2.jsonl"
        if r2_path.exists():
            agg = self_agreement(read_jsonl(r1_path), read_jsonl(r2_path))
            (settings.paths.golden / "self_agreement.json").write_text(json.dumps(agg, indent=2, default=str), encoding="utf-8")
            print("self-agreement:", json.dumps(agg, indent=2, default=str))
        return

    if args.round == 1:
        parts = set(args.parts.split(","))
        todo = [c for c in candidates if c["part"] in parts]
        out = r1_path
    else:
        round1 = read_jsonl(_require(r1_path, "label round 1 first"))
        ids = set(select_for_relabel(round1, n=args.relabel, seed=settings.seed))
        todo = [c for c in candidates if c["item_id"] in ids]
        out = settings.paths.golden / "labels_round2.jsonl"
    n = Labeller(taxonomy, todo, out, round_no=args.round).run(limit=args.limit)
    print(f"\nlabelled {n} items this session -> {out}")


def cmd_index(args, settings) -> None:
    import pandas as pd

    from .retrieval.index import TfidfIndex

    corpus = pd.read_parquet(_require(_processed(settings, "corpus.parquet"), "run `prepare --brand X` first"))
    index = TfidfIndex().fit(corpus)
    path = index.save(_processed(settings, "index.joblib"))
    print(f"indexed {len(index):,} historical threads -> {path}")


# ----------------------------------------------------------------------------------------------------------
# model stage
# ----------------------------------------------------------------------------------------------------------
def _agent_for(system: str, settings, taxonomy, seed: int):
    from .agent.pipeline import Agent, AgentConfig
    from .llm import make_client
    from .retrieval.index import TfidfIndex

    use_retrieval = system != "no_rag"
    index = TfidfIndex.load(_require(_processed(settings, "index.joblib"), "run `index` first")) if use_retrieval else None
    llm = make_client("gemini" if system == "main_gemini" else "local", settings)
    config = AgentConfig(system=system, use_retrieval=use_retrieval, k=settings.retrieval_k,
                         min_similarity=settings.min_retrieval_similarity, max_reply_chars=settings.max_reply_chars, seed=seed)
    return Agent(llm, taxonomy, config, index=index)


def cmd_smoke(args, settings) -> None:
    import statistics

    from .agent.pipeline import Agent, AgentConfig
    from .llm import make_client

    taxonomy = _load_taxonomy(settings, args.taxonomy)
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


def cmd_run(args, settings) -> None:
    from .llm.batch import run_batch

    taxonomy = _load_taxonomy(settings, "real")
    items = _load_split(settings, args.split)
    if args.limit:
        items = items[: args.limit]
    seed = args.seed if args.seed is not None else settings.seed
    agent = _agent_for(args.system, settings, taxonomy, seed)
    suffix = ("_" + args.split if args.split != "test" else "") + (f"_{args.tag}" if args.tag else "")
    out = settings.paths.runs / f"{args.system}{suffix}.jsonl"
    rows = run_batch([{**it, "text": it["root_text"], "doc_id": str(it["root_id"])} for it in items],
                     agent.handle, out, desc=f"{args.system}/{args.split}")
    ok = [r for r in rows if "error" not in r]
    print(f"{len(ok)}/{len(items)} items done -> {out}; escalate rate {sum(r['decision'] == 'escalate' for r in ok) / max(len(ok), 1):.1%}; "
          f"parse failures {sum(r.get('parse_failed', False) for r in ok)}; forced by rules {sum(bool(r.get('forced_reason')) for r in ok)}")


def cmd_baselines(args, settings) -> None:
    from .baselines.simple import cv_intent_predictions, human_reference_rows, simple_rows
    from .baselines.trivial import majority_intent, trivial_rows
    from .llm.batch import write_jsonl
    from .retrieval.index import TfidfIndex

    taxonomy = _load_taxonomy(settings, "real")
    items = _load_split(settings, args.split)
    dev = _load_split(settings, "dev") if args.split == "test" else []
    suffix = "_" + args.split if args.split != "test" else ""
    index = TfidfIndex.load(_require(_processed(settings, "index.joblib"), "run `index` first"))

    majority = majority_intent(dev or items)   # dev labels decide the majority class; test labels only as a fallback
    for always in (True, False):
        rows = trivial_rows(items, majority, always_escalate=always)
        write_jsonl(settings.paths.runs / f"{rows[0]['system']}{suffix}.jsonl", rows)
    preds, confs = cv_intent_predictions([it["root_text"] for it in items], [it["intent_primary"] for it in items],
                                         extra_texts=[d["root_text"] for d in dev], extra_labels=[d["intent_primary"] for d in dev],
                                         seed=settings.seed, brand=settings.brand)
    rows = simple_rows(items, preds, confs, index, taxonomy.hard_rules(), settings.brand)
    write_jsonl(settings.paths.runs / f"simple{suffix}.jsonl", rows)
    ref = human_reference_rows(items)
    write_jsonl(settings.paths.runs / f"human_ref{suffix}.jsonl", ref)
    print(f"majority intent (from {'dev' if dev else 'test'}): {majority}; simple baseline: {len(rows)} rows; human reference replies: {len(ref)}")


COMMANDS = {
    "data": cmd_data, "threads": cmd_threads, "profile": cmd_profile, "prepare": cmd_prepare, "taxonomy": cmd_taxonomy,
    "sample": cmd_sample, "label": cmd_label, "index": cmd_index, "smoke": cmd_smoke, "run": cmd_run, "baselines": cmd_baselines,
}


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
