# Golden set — how it was sampled

Brand: **SpotifyCares**. Golden window: 2017-11-17 to 2017-12-01 (retrieval corpus and taxonomy induction use only threads before 2017-11-17; a 2-day buffer before the dump's last day avoids truncated threads).

Population: 6418 customer-initiated threads in the window.

Excluded before sampling (counted, never silent):
- orphan root fragment: 8
- near duplicate of kept item: 196
- near empty beyond quota: 69
- non english beyond quota: 203

Eligible core (English, ≥3 content tokens, one representative per near-duplicate cluster): 5930.

Draws (seed 42):
- **Part A, 120 items** — uniform random from the core, plus quotas of up to 6 near-empty/image-only and 6 non-English tweets so routing behaviour is exercised. Headline numbers use Part A.
- **Part B, 80 items** — 2 per TF-IDF topic cluster (20 clusters) for per-intent support, then an oversample of tweets matching escalation keywords (refund, charged, hacked, lawyer, cancel, 'again', 'third time', ...). Per-intent and escalation-recall tables use A+B and say so.
- **Dev, 50 items** — uniform random, disjoint; the only set prompts and thresholds were tuned on.
- **Spares** — 30 % extra, used only to replace items excluded during labelling (each exclusion is logged with its reason).

Resulting counts: A=120, B=80, spare=75, dev=50.

Labelling protocol: taxonomy and escalation policy frozen and committed first; a 30-item pilot (merges only) preceded the main pass; items were shown in random order with no model output, no cluster id and no brand reply visible; 40 items were re-labelled blind on a later day to estimate label noise (Cohen's κ).
