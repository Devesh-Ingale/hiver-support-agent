# Dev set (n=50) — iteration numbers, not the headline

label distribution: playback_or_app_issue=12, content_request_or_availability=10, payment_billing_subscription=7, family_or_student_plan=6, feature_request_or_feedback=6, account_access_security=4, support_followup_or_complaint=3, account_settings_change=1, library_or_playlist_problem=1
escalation labels: none=32, account_or_pii=7, payment_refund=5, anger_churn=3, explicit_human_request=3

## main  (n=50, versions [('v1.2', '1')])
- intent accuracy 78.0% [66%, 88%] · lenient 82.0% [70%, 92%] · macro-F1 0.71
- escalation P/R/F1 0.67/0.56/0.61 · hard recall 66.7% [40%, 87%] (n_hard=15) · automation 70.0% [56%, 82%] · cost/100 128 · missed hard 10.0%
- reason-code agreement (both escalated): exact 0.70 on n=10
- raw-draft violations {'url_not_in_evidence': 0.02, 'any': 0.02} · forced by rule {'payment_refund': 4, 'account_or_pii': 1, 'draft_invalid': 1, 'explicit_human_request': 1} · parse failures 0.0%
- weakest intents: account_settings_change F1 0.50 (n=1); library_or_playlist_problem F1 0.50 (n=1); support_followup_or_complaint F1 0.57 (n=3)
- similarity-floor sweep (threshold → automation, missed-hard): 0.00→70%,10.0%; 0.10→70%,10.0%; 0.20→66%,10.0%; 0.30→38%,4.0%; 0.40→16%,4.0%; 0.50→6%,2.0%; 0.60→2%,2.0%; 0.70→0%,0.0%; 0.80→0%,0.0%; 0.90→0%,0.0%; 1.00→0%,0.0%
- best floor at ≤2 % missed hard: {'threshold': 0.48, 'automation_rate': 0.06, 'missed_hard_rate': 0.02, 'achievable': True}

## main_gemini  (n=50, versions [('v1.2', '1')])
- intent accuracy 88.0% [78%, 96%] · lenient 90.0% [82%, 98%] · macro-F1 0.84
- escalation P/R/F1 0.69/0.61/0.65 · hard recall 73.3% [47%, 93%] (n_hard=15) · automation 68.0% [54%, 80%] · cost/100 108 · missed hard 8.0%
- reason-code agreement (both escalated): exact 0.64 on n=11
- raw-draft violations {'any': 0.0} · forced by rule {'payment_refund': 4, 'account_or_pii': 1, 'explicit_human_request': 1} · parse failures 0.0%
- weakest intents: account_access_security F1 0.67 (n=4); account_settings_change F1 0.67 (n=1); library_or_playlist_problem F1 0.67 (n=1)

## no_rag  (n=50, versions [('v1.2', '1')])
- intent accuracy 82.0% [70%, 92%] · lenient 86.0% [76%, 94%] · macro-F1 0.78
- escalation P/R/F1 0.41/0.94/0.58 · hard recall 100.0% [100%, 100%] (n_hard=15) · automation 18.0% [8%, 28%] · cost/100 54 · missed hard 0.0%
- reason-code agreement (both escalated): exact 0.59 on n=17
- raw-draft violations {'anonymised_handle': 0.02, 'too_long': 0.04, 'any': 0.06} · forced by rule {'payment_refund': 4, 'account_or_pii': 1, 'draft_invalid': 1, 'explicit_human_request': 1} · parse failures 0.0%
- weakest intents: library_or_playlist_problem F1 0.50 (n=1); support_followup_or_complaint F1 0.57 (n=3); account_settings_change F1 0.67 (n=1)
