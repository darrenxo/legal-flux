from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from .io_utils import split_facts
from .models import CaseState, Element, Issue, NormalizedCase


LEGALHK_PARQUET_URL = (
    "https://huggingface.co/api/datasets/weijiezz/LegalHK/"
    "parquet/default/train/0.parquet"
)


def download_file(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with httpx.stream("GET", url, follow_redirects=True, timeout=600) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    partial.replace(destination)


def normalize_legalhk_case(index: Any, row: pd.Series, *, split: str) -> NormalizedCase:
    facts = split_facts(row["more_facts"])
    issues = [line.strip() for line in str(row["issues"]).splitlines() if line.strip()]
    reference_state = legalhk_reference_state(row, facts, issues)
    return NormalizedCase(
        dataset="legalhk",
        case_id=f"legalhk-{index}",
        claim=row["plaintiff_claim"],
        requested_remedy=row["plaintiff_claim"],
        parties=[
            f"Plaintiff: {row['plaintiff']}",
            f"Defendant: {row['defendant']}",
        ],
        facts=facts,
        authorities=None,
        gold_answer=str(row["support&reject"]).strip().lower(),
        reference_issues=issues,
        reference_state=reference_state,
        metadata={
            "selection_split": split,
            "lawsuit_type": row["lawsuit_type"],
            "issue_count": len(issues),
            "fact_characters": len(row["more_facts"]),
            "has_defense": bool(row["has_defense"]),
            "leakage_screen": "auto_pass",
            "license_warning": "processed release license unknown",
        },
    )


def legalhk_reference_state(
    row: pd.Series, facts: dict[str, str], issues: list[str]
) -> CaseState:
    if not issues:
        issues = ["Whether the plaintiff's requested relief should be granted."]
    return CaseState(
        claims=[row["plaintiff_claim"]],
        requested_remedies=[row["plaintiff_claim"]],
        issues=[
            Issue(
                issue_id=f"I{index}",
                issue=issue,
                rule_or_test=(
                    str(row["related_laws"]).strip()
                    or "Resolve under the applicable law."
                ),
                burden_on="plaintiff",
                elements=[
                    Element(
                        element_id=f"I{index}E1",
                        element="Facts and law sufficient to resolve the issue",
                        supporting_fact_ids=list(facts),
                        opposing_fact_ids=[],
                        missing_information=[],
                    )
                ],
                defenses=[],
            )
            for index, issue in enumerate(issues, start=1)
        ],
    )


def legalhk_index(case_id: str) -> int:
    prefix = "legalhk-"
    if not str(case_id).startswith(prefix):
        raise ValueError(f"Invalid LegalHK case ID: {case_id}")
    return int(str(case_id)[len(prefix) :])
