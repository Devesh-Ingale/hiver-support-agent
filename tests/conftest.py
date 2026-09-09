from __future__ import annotations

from pathlib import Path

import pytest

from support_agent.taxonomy import load_taxonomy

TEST_TAXONOMY = """
brand: SpotifyCares
version: test
intents:
  - id: playback_issue
    name: Playback or app malfunction
    definition: Music will not play, skips, crashes, or the app misbehaves.
    examples: ["app crashes when I open a playlist", "songs keep skipping"]
    edge_rules: ["Login trouble is account_access, not playback_issue."]
    typical_action: troubleshooting steps
  - id: payment_billing
    name: Payment and billing
    definition: Charges, refunds, subscription price or payment failures.
    examples: ["charged twice this month"]
  - id: account_access
    name: Account access
    definition: Cannot log in, hacked account, password problems.
  - id: other_unclear
    name: Other / unclear
    definition: Anything else, or too little text to tell.
policy:
  good_reply:
    - gives the first troubleshooting step the brand actually uses for this issue
    - is friendly, short, one clear next step
  never:
    - ask for passwords or payment details in public
    - promise refunds, fixes or timelines
  auto_ok:
    - a known troubleshooting step from the evidence addresses the issue
    - the tweet is praise or feedback that a short acknowledgement answers
  reasons:
    - code: account_or_pii
      tier: hard
      description: needs account access or exchange of personal data
      patterns: ['\\b(hacked|someone (?:else )?(?:is )?using my account|unauthori[sz]ed)\\b']
    - code: payment_refund
      tier: hard
      description: money — charges, refunds, billing disputes
      patterns: ['\\b(refund|charged twice|double[- ]charged|chargeback)\\b']
    - code: legal_safety_threat
      tier: hard
      description: legal threats, safety, harassment, press
      patterns: ['\\b(lawyer|lawsuit|sue you|legal action)\\b']
    - code: explicit_human_request
      tier: hard
      description: customer explicitly asks for a person
      patterns: ['\\b(speak (?:to|with) (?:a )?(?:human|person|real person)|talk to (?:a )?human)\\b']
    - code: anger_churn
      tier: soft
      description: strong anger or a threat to cancel
    - code: non_english
      tier: routing
      description: not in English
    - code: no_actionable_content
      tier: routing
      description: nothing to act on (image-only, empty)
    - code: no_relevant_resolution
      tier: routing
      description: no sufficiently similar historical case
    - code: draft_invalid
      tier: routing
      description: the draft failed validity checks
"""


@pytest.fixture()
def taxonomy_path(tmp_path: Path) -> Path:
    p = tmp_path / "taxonomy.yaml"
    p.write_text(TEST_TAXONOMY, encoding="utf-8")
    return p


@pytest.fixture()
def taxonomy(taxonomy_path):
    return load_taxonomy(taxonomy_path)
