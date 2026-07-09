from __future__ import annotations

from typing import Literal

from pydantic import Field

from .models import StrictModel
from .prompting import format_facts


class InputLeakageAudit(StrictModel):
    risk: Literal["clean", "questionable", "leaky"]
    current_outcome_disclosed: bool
    judicial_evaluation_present: bool
    suspect_snippets: list[str] = Field(default_factory=list, max_length=3)
    rationale: str


def render_input_leakage_prompt(
    *, claim: str, facts: dict[str, str]
) -> str:
    return f"""Audit this legal-case input for outcome leakage.

The target task is to decide the CURRENT DISPUTE described by the claim. Apply a
CONSERVATIVE standard. Earlier procedural history is not automatically leakage,
but any un-attributed adjudicative conclusion makes the input questionable.
Classify:

- clean: objective events, party allegations, documents, and procedural posture
  do not reveal how the current dispute was resolved;
- questionable: the text contains authoritative legal/evidentiary evaluations
  that could shortcut a material issue, even if the final result is unstated;
- leaky: the text states or unmistakably implies the holding, disposition, or
  decisive judicial finding for the current dispute.

Mark questionable for any un-attributed legal rule, contract interpretation,
credibility assessment, relevance or sufficiency assessment, finding about
reasonable conduct, or conclusion that a requirement was or was not met.
Examples include "service was properly effected", "breach was not established",
"the original document was necessary", "could be interpreted as", "not
analogous", "too wide", "not relevant", or "a deliberate decision". Treat
statements explicitly attributed to a party as allegations, not judicial
findings. When uncertain, choose questionable.
Quote at most three short suspect snippets. Do not infer or decide who should
win. Return only the required JSON.

CLAIM:
{claim}

FACTS:
{format_facts(facts)}
"""
