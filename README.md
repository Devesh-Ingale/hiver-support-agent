# hiver-support-agent

An AI customer-support agent for one brand from the Kaggle *Customer Support on Twitter* dataset. Given a customer's tweet it (1) classifies the intent, (2) drafts a reply grounded in how the brand historically handled similar issues, and (3) decides whether to auto-handle or escalate to a human, with a reason — plus the evaluation harness, golden set and report that say how far to trust it.

> Work in progress. The reproduction path, results and report are filled in as the project progresses; see `REPORT.md` and `DECISIONS.md`.

## Reproduce the headline numbers (no API keys, no Kaggle account)

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e . --no-deps
python -m support_agent.cli eval                   # recomputes every number/table/figure from committed outputs
```

## Layout

```
src/support_agent/   data · taxonomy · llm · retrieval · agent · baselines · eval · cli
data/processed/      brand-filtered threads and retrieval corpus (committed, small)
data/golden/         hand-labelled test/dev sets + sampling note
outputs/             cached model outputs, judge outputs, human ratings, results
report/              figures, report.docx
tests/               pytest suite
```

## Rebuild from scratch (optional)

- `python -m support_agent.cli data` — downloads `twcs.csv` from Kaggle (needs `~/.kaggle/kaggle.json`).
- `python -m support_agent.cli threads` / `profile` — thread reconstruction and brand profiling.
- Agent runs need a local [Ollama](https://ollama.com) server; judge runs need `GEMINI_API_KEY` in `.env` (see `.env.example`).
