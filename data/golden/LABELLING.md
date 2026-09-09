# Labelling protocol (golden set, SpotifyCares)

Who labels: the author, blind. What you see: the customer's tweet only — never a model output, the
brand's real reply, the sampling part or a cluster id. Order is random. Everything is saved after each item,
so you can stop with `q` any time and resume with the same command.

## Run it

```powershell
cd "C:\Users\deves\Downloads\Hiver Assignment\hiver-support-agent"
.\.venv\Scripts\python -m support_agent.cli label --limit 30      # pilot: first 30 items
.\.venv\Scripts\python -m support_agent.cli label                 # main pass: remaining test + dev items (~250)
.\.venv\Scripts\python -m support_agent.cli label --round 2       # day 3: 40 items re-labelled blind
.\.venv\Scripts\python -m support_agent.cli label --finalize      # builds data/golden/test.jsonl + dev.jsonl
```

## Per item (~40 s)

| prompt | answer |
|---|---|
| `intent # (primary)` | the number of the intent (cheat sheet is printed at the start; `?` reprints it). `x` = exclude this item (you will be asked why). |
| `secondary intent #` | only if the tweet clearly raises a **second, separate** issue; otherwise Enter. |
| `escalate to a human? [y/n]` | apply the checklist below literally — not "would a human be nicer here". |
| `reason #` | (only after `y`) the reason code. Pick the **hard** code if one applies, else `anger_churn`, else a routing code. |
| `anger 0/1/2` | 0 calm or neutral · 1 annoyed/sarcastic · 2 furious, insulting, or threatening to leave. Enter = 0. |
| `confidence 1/2/3` | how sure you are of the **intent**: 1 guess · 2 fairly sure · 3 sure. Enter = 3. Be honest; this feeds a label-noise table. |
| `flags` | letters, any combination, Enter = none: `a` ambiguous · `m` multi-intent · `n` noise/spam/not a request · `i` image/link-only · `r` addressed to another customer · `p` contains personal data · `s` sarcasm |
| `note` | free text, Enter = none. Use it when a rule felt wrong — the pilot review reads these. |

## The escalation checklist (decide `y` when ANY line applies)

**Hard (always):**
1. `account_or_pii` — the fix needs someone to look at or touch the account, or the customer must hand over personal data: hacked/compromised, cannot log in, password/email problems, "check my account".
2. `payment_refund` — money: a charge, refund, double payment, failed payment, "paid but still Free", cancelling a paid plan.
3. `legal_safety_threat` — lawyers/lawsuits, fraud or scam accusations, self-harm, harassment/discrimination, press or regulators.
4. `explicit_human_request` — asks for a person, asks you to read a DM/email already sent, or says they have asked repeatedly with no answer.

**Soft (judgment):**
5. `anger_churn` — anger 2, or a credible "I'm cancelling" — even if the underlying issue is routine.

**Routing (not about content):**
6. `non_english` — not English. 7. `no_actionable_content` — emoji/link only, nothing to act on.

Otherwise `n`: a grounded public first reply is acceptable — a troubleshooting step, a relevant clarifying
question ("which device/OS?"), an availability explanation, or a thank-you.

## Intent tie-breakers (also in the cheat sheet)

- Hacked / can't log in beats everything else → `account_access_security`.
- Paid-but-still-Free, charges, refunds, offers, cancelling → `payment_billing_subscription`; Family/Student
  eligibility problems → `family_or_student_plan` even when a charge is mentioned (charge = secondary).
- Song greyed out or won't play → `playback_or_app_issue`; song not on Spotify at all / not in my country →
  `content_request_or_availability`.
- Own saved music, playlists, shuffle, Discover Weekly wrong/missing → `library_or_playlist_problem`;
  wishing for a NEW capability → `feature_request_or_feedback`.
- Praise/complaints about **support staff** or "reply to my DM" → `support_followup_or_complaint`;
  about the **product** → `feature_request_or_feedback`.
- Not aimed at Spotify, memes, link-only, too little text → `other_unclear` (+ flag `n` or `i`).

## Exclude (`x`) only when

The item cannot be labelled at all: a fragment that is clearly mid-conversation, a duplicate you already saw,
or a tweet that is not addressed to Spotify **and** contains no request. Non-English and emoji-only tweets are
NOT excluded — label them `other_unclear` + escalate with the routing reason; they are in the set on purpose.

## Pilot rule

After the first 30 items we may **merge or clarify** intents, never split or add. Then the taxonomy is frozen
as v1 and committed; its commit hash is quoted in the report. Items labelled in the pilot are kept.
