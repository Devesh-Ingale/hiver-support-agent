# hiver-support-agent

An AI customer-support agent for one brand from the Kaggle *Customer Support on Twitter* dataset. Given a customer's tweet it (1) classifies the intent into a small taxonomy derived from the brand's own traffic, (2) drafts a public reply grounded in how the brand historically handled similar issues, and (3) decides whether to auto-handle or escalate to a human, with a stated reason.

The repo is at least as much about the **proof** as the agent: a frozen, hand-labelled test set; trivial and simple baselines; an LLM judge whose agreement with a human is measured, not assumed; bootstrap intervals on every number; a pre-registered acceptance gate; and a report section on what is misleading about the headline number.

> **Status:** pipeline and evaluation harness complete; brand selection, labelling and results are in progress. `REPORT.md` and `DECISIONS.md` are the written deliverables.

## Reproduce the headline numbers — no API keys, no Kaggle account, ~2 minutes

```bash
git clone https://github.com/Devesh-Ingale/hiver-support-agent && cd hiver-support-agent
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e . --no-deps
python -m support_agent.cli eval                       # every metric, table and figure input from committed outputs
python -m support_agent.cli figures                    # report/figures/*.png
python -m pytest                                       # unit tests (synthetic data, no network)
```

`eval` reads the committed golden set (`data/golden/test.jsonl`), the cached model outputs (`outputs/runs/`), judge outputs (`outputs/judge/`) and human ratings (`outputs/human/`), and writes `outputs/results/metrics.json` + `tables.md`. It never calls a model.

## How the system works

```
customer tweet ─► TF-IDF retrieval over the brand's historical threads (k=5, boilerplate down-weighted)
              ─► ONE structured LLM call (local qwen3:4b-instruct via Ollama; JSON-schema enforced)
                   → intent, secondary intent, confidence, evidence ids, draft reply, reason, reason code, decision
              ─► deterministic post-processor (src/support_agent/agent/checks.py)
                   hard escalation rules from the written policy · non-English / empty routing ·
                   "no relevant resolution" similarity floor · tweet-validity checks (≤280 chars, no invented URLs,
                   no @handles, no agent sign-offs, no promises, no public requests for sensitive data)
              ─► run record with both the model's answer and the final answer
```

Systems compared on the same test items: `trivial_escalate` / `trivial_auto` (majority intent + template reply), `simple` (TF-IDF + logistic regression intents by 5-fold CV, nearest-neighbour verbatim brand reply, policy keyword rules), `no_rag` (same prompt, no evidence), **`main`**, `main_gemini` (same prompts on `gemini-3.8-flash`), and `human_ref` (the brand's real replies, judged with the same rubric).

## Rebuild everything from scratch (optional)

| step | command | needs |
|---|---|---|
| download the dump | `python -m support_agent.cli data` | `~/.kaggle/kaggle.json` |
| reconstruct threads | `python -m support_agent.cli threads` | — |
| profile the largest brands | `python -m support_agent.cli profile` | — |
| clean + time-split one brand | `python -m support_agent.cli prepare --brand <Brand>` | — |
| cluster messages for the taxonomy | `python -m support_agent.cli taxonomy [--propose]` | Ollama for `--propose` |
| draw golden-set candidates | `python -m support_agent.cli sample` | — |
| label blind (round 1, then round 2) | `python -m support_agent.cli label [--round 2] [--finalize]` | a human |
| build the retrieval index | `python -m support_agent.cli index` | — |
| run the agent | `python -m support_agent.cli run --system main [--split dev]` | [Ollama](https://ollama.com) + `ollama pull qwen3:4b-instruct` |
| baselines + human reference | `python -m support_agent.cli baselines` | — |
| LLM judge | `python -m support_agent.cli judge --mode absolute|pairwise|validate|perturb|consistency` | `GEMINI_API_KEY` in `.env` |
| human ratings | `python -m support_agent.cli rate --mode pairs|absolute` | a human |
| metrics, figures, report | `python -m support_agent.cli eval` · `figures` · `docx` | pandoc for `docx` |

Every LLM call is cached in `outputs/llm_cache.sqlite` (git-ignored; the JSONL run files are the artefacts of record), so re-running a step never repeats a call. Gemini calls are rate-limited client-side and counted per day for the free tier.

## Layout

```
src/support_agent/
  data/        download · threads (reconstruction) · clean · tags (reply-type regexes) · profile_brands · prepare (time split)
  taxonomy/    taxonomy.yaml (intents + escalation policy, frozen) · loader · induce (clustering aid)
  llm/         provider protocol · ollama_client · gemini_client · sqlite cache · rate limiter · resumable batch runner
  retrieval/   TF-IDF index (word + char n-grams), boilerplate penalty, dedup
  agent/       schemas (output contract) · prompts (versioned) · pipeline · checks (policy + validity)
  baselines/   trivial · simple (CV-LR, kNN verbatim, rules) · human reference rows
  eval/        golden (sampling) · label_cli · rate_cli · judge · metrics · stats · agreement · run_eval · figures
  cli.py       every command above
data/golden/   candidates, labels, test.jsonl, dev.jsonl, sampling_note.md, self_agreement.json
outputs/       runs/ judge/ human/ results/   (committed)
report/        figures/, report.docx
tests/         pytest suite on synthetic data
```

## Citations and borrowed material

- Dataset: Axel Brooks / Thought Vector, *Customer Support on Twitter*, Kaggle (`thoughtvector/customer-support-on-twitter`), licence as stated on the dataset page (CC BY-NC-SA 4.0 at time of writing). Customer handles in the dump are anonymised to numeric ids; brand handles are real.
- Models: Qwen3-4B-Instruct (Alibaba, Apache-2.0) served by Ollama; `gemini-3.8-flash` (Google) as the judge.
- Judge design borrows the rationale-first, rubric-with-binary-checks pattern from G-Eval (Liu et al., 2023) and the position-swapped pairwise protocol from MT-Bench (Zheng et al., 2023).
- Libraries: pandas, scikit-learn, scipy, matplotlib, pydantic, rank_bm25, langdetect, rapidfuzz, google-genai, ollama. Chart palette and mark conventions follow a validated colour-blind-safe reference palette.
- AI coding assistants were used throughout, as the brief allows; every module has unit tests and the author can walk through and modify any part live.

## Data ethics

The tweets are public customer complaints with customer identities anonymised at source. No attempt is made to de-anonymise anyone, replies are never sent, and the model outputs in `outputs/` are generated drafts, not real brand communications.
