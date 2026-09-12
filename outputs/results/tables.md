# Results tables — SpotifyCares (test set, n=200; Part A n=120)

## Headline (all test items, A+B)

| system | intent acc [95% CI] | macro-F1 | esc. recall | hard recall | automation | cost/100 | raw-draft violations | judge pass | judge pass (auto only) |
|---|---|---|---|---|---|---|---|---|---|
| main | 71.0% [65%, 77%] | 0.64 | 89.3% | 92.1% [87%, 97%] | 46.5% [40%, 54%] | 48 | 8.5% | 74.0% [68%, 80%] | 89.2% [83%, 96%] |
| main_gemini | 91.0% [87%, 95%] | 0.90 | 95.1% | 96.6% [92%, 100%] | 41.5% [34%, 48%] | 28 | 7.5% | — | — |
| no_rag | 76.0% [70%, 82%] | 0.72 | 93.2% | 95.5% [91%, 99%] | 33.5% [28%, 40%] | 43 | 6.0% | — | — |
| simple | 55.5% [48%, 62%] | 0.40 | 65.0% | 64.0% [54%, 74%] | 58.0% [52%, 65%] | 174 | 9.5% | 54.9% [41%, 69%] | 60.7% [43%, 79%] |
| trivial_escalate | 13.0% [8%, 18%] | 0.02 | 100.0% | 100.0% [100%, 100%] | 0.0% [0%, 0%] | 48 | 0.0% | — | — |
| trivial_auto | 13.0% [8%, 18%] | 0.02 | 0.0% | 0.0% [0%, 0%] | 100.0% [100%, 100%] | 466 | 0.0% | — | — |

## Headline on Part A only (uniform random sample)

| system | intent acc [95% CI] | esc. recall | hard recall | automation | cost/100 |
|---|---|---|---|---|---|
| main | 69.2% [61%, 77%] | 90.4% | 94.9% [87%, 100%] | 51.7% [42%, 61%] | 33 |
| main_gemini | 90.0% [85%, 95%] | 92.3% | 94.9% [87%, 100%] | 52.5% [43%, 62%] | 29 |
| no_rag | 73.3% [66%, 81%] | 90.4% | 92.3% [82%, 100%] | 40.0% [32%, 49%] | 51 |
| simple | 51.7% [43%, 61%] | 63.5% | 61.5% [46%, 77%] | 61.7% [52%, 70%] | 146 |
| trivial_auto | 15.0% [9%, 22%] | 0.0% | 0.0% [0%, 0%] | 100.0% [100%, 100%] | 358 |
| trivial_escalate | 15.0% [9%, 22%] | 100.0% | 100.0% [100%, 100%] | 0.0% [0%, 0%] | 57 |

## Acceptance gate

- 1_hard_recall_ge_0.95_lowerCI_ge_0.85: **FAIL**
- 2_zero_validity_violations_on_auto_sent: **PASS**
- 3_judge_pass_rate_auto_handled_ge_0.70: **PASS**
- 4_cost_below_always_escalate: **FAIL**
- 5_automation_at_2pct_missed_hard: **PASS**

All tested conditions passed: **False**
