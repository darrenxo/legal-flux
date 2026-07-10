from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import resolve_path
from .io_utils import sha256_text
from .models import NormalizedCase


def load_prompt(config: dict[str, Any], name: str) -> str:
    path = resolve_path(config, "prompts_dir") / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def format_facts(facts: dict[str, str]) -> str:
    return "\n".join(f"{fact_id}: {text}" for fact_id, text in facts.items())


def authority_context(case: NormalizedCase, *, include: bool) -> str:
    if not include:
        return ""
    related_laws = (case.authorities or "").strip()
    relevant_cases = str(case.metadata.get("relevant_cases") or "").strip()
    if not related_laws and not relevant_cases:
        return ""
    lines = ["", "AUTHORITY CONTEXT:"]
    lines.append("RELATED LAWS:")
    lines.append(related_laws or "No related laws supplied.")
    lines.append("")
    lines.append("RELEVANT CASES:")
    lines.append(relevant_cases or "No relevant cases supplied.")
    return "\n".join(lines)


def render_prompt(
    config: dict[str, Any],
    name: str,
    case: NormalizedCase,
    **extra: Any,
) -> tuple[str, str]:
    template = load_prompt(config, name)
    include_authority = bool(
        config.get("legal_flux", {}).get("include_authority_input", False)
    )
    values = {
        "claim": case.claim,
        "requested_remedy": case.requested_remedy or "Not separately specified.",
        "parties": "\n".join(case.parties) or "Not separately specified.",
        "facts": format_facts(case.facts),
        "authorities": case.authorities or "No authorities supplied.",
        "relevant_cases": case.metadata.get("relevant_cases") or "No relevant cases supplied.",
        "authority_context": authority_context(case, include=include_authority),
        "reference_issues": "\n".join(case.reference_issues) or "None supplied.",
        "gold_answer": case.gold_answer,
        **{
            key: (
                json.dumps(value, ensure_ascii=False, indent=2)
                if not isinstance(value, str)
                else value
            )
            for key, value in extra.items()
        },
    }
    prompt = template.format(**values)
    return prompt, sha256_text(prompt)
