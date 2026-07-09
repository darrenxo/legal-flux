from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import pandas as pd


FAMILY_PATTERNS: dict[str, re.Pattern[str]] = {
    "contract_performance": re.compile(
        r"\b(contract|agreement|breach|sale and purchase|s&p|specific performance|"
        r"termination|repudiat|condition precedent|requisition)\b",
        re.I,
    ),
    "debt_payment": re.compile(
        r"\b(debt|loan|invoice|unpaid|arrears|deposit|cheque|promissory|"
        r"money due|sum due|rent arrears|payment)\b",
        re.I,
    ),
    "property_possession": re.compile(
        r"\b(property|land|premises|tenan|lease|landlord|possession|occupation|"
        r"title|conveyance|licen[cs]e)\b",
        re.I,
    ),
    "tort_negligence_damage": re.compile(
        r"\b(negligence|duty of care|tort|accident|injur|damage|causation|"
        r"nuisance|defamation|collision)\b",
        re.I,
    ),
    "employment_compensation": re.compile(
        r"\b(labou?r|employment|employee|employer|wages?|severance|"
        r"employee'?s compensation|work injury|dismissal)\b",
        re.I,
    ),
    "company_insolvency": re.compile(
        r"\b(company|companies ordinance|shareholder|director|liquidat|winding"
        r" up|insolven|creditor|unfair prejudice|scheme of arrangement)\b",
        re.I,
    ),
    "procedure_appeal": re.compile(
        r"\b(appeal|leave|extension of time|set aside|strike out|security for"
        r" costs|summary judgment|judicial review|case stated|interlocutory|"
        r"jurisdiction|procedural)\b",
        re.I,
    ),
    "public_criminal_immigration": re.compile(
        r"\b(criminal|conviction|sentence|sentencing|prosecution|hksar|bail|"
        r"offence|charge|trafficking|immigration|refugee|torture claim|"
        r"non-refoulement|deportation)\b",
        re.I,
    ),
    "trust_probate_family": re.compile(
        r"\b(trust|estate|probate|will|inheritance|matrimonial|divorce|"
        r"maintenance|family)\b",
        re.I,
    ),
}

EVIDENCE_PATTERN = re.compile(
    r"\b(evidence|witness|credib|proof|prove|burden|conflict|contradict|"
    r"allegation|testimony|expert|documentary)\b",
    re.I,
)
DEFENSE_PATTERN = re.compile(
    r"\b(defen[cs]e|counterclaim|set-?off|limitation|estoppel|laches|"
    r"mitigation|contributory|waiver|illegality|jurisdictional objection)\b",
    re.I,
)
REMEDY_DISCRETION_PATTERN = re.compile(
    r"\b(specific performance|injunction|declaration|declaratory|stay|"
    r"set aside|leave|extension|security for costs|equitable|discretion)\b",
    re.I,
)


def profile_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, row in frame.iterrows():
        profile = profile_row(row.to_dict())
        profile["row_index"] = index
        rows.append(profile)
    return pd.DataFrame(rows)


def profile_row(row: dict[str, Any]) -> dict[str, Any]:
    text = _row_text(row)
    issues = split_items(row.get("issues", ""))
    related_laws = split_items(row.get("related_laws", ""))
    relevant_cases = split_items(row.get("relevant_cases", ""))
    facts = str(row.get("more_facts") or "")
    fact_count = _estimate_fact_count(facts)
    fact_characters = len(facts)
    families = _families(text)
    demands = _demands(
        text=text,
        issue_count=len(issues),
        law_count=len(related_laws),
        precedent_count=len(relevant_cases),
        fact_count=fact_count,
        fact_characters=fact_characters,
    )
    trajectory = _trajectory(families, demands)
    return {
        "lawsuit_type": str(row.get("lawsuit_type") or ""),
        "gold_answer": str(row.get("support&reject") or row.get("gold_answer") or ""),
        "issue_count": len(issues),
        "related_law_count": len(related_laws),
        "relevant_case_count": len(relevant_cases),
        "fact_count_estimate": fact_count,
        "fact_characters": fact_characters,
        "template_families": "|".join(families),
        "reasoning_demands": "|".join(demands),
        "trajectory": "|".join(trajectory),
        "trajectory_signature": " > ".join(trajectory),
        "family_count": len(families),
        "demand_count": len(demands),
        "trajectory_length": len(trajectory),
    }


def split_items(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    parts = [part.strip(" -\t") for part in re.split(r"[\n;]+", text)]
    return [part for part in parts if part]


def label_counts(series: pd.Series) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for value in series.fillna(""):
        for label in str(value).split("|"):
            if label:
                counter[label] += 1
    return pd.DataFrame(
        [{"label": key, "count": count} for key, count in counter.most_common()]
    )


def normalized_entropy(values: pd.Series) -> float:
    counts = values.value_counts()
    if len(counts) <= 1:
        return 0.0
    total = counts.sum()
    entropy = -sum((count / total) * math.log(count / total) for count in counts)
    return entropy / math.log(len(counts))


def _row_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in (
            "plaintiff_claim",
            "claim",
            "requested_remedy",
            "lawsuit_type",
            "more_facts",
            "issues",
            "related_laws",
            "relevant_cases",
        )
    )


def _families(text: str) -> list[str]:
    matched = [name for name, pattern in FAMILY_PATTERNS.items() if pattern.search(text)]
    return matched or ["general_civil_reasoning"]


def _demands(
    *,
    text: str,
    issue_count: int,
    law_count: int,
    precedent_count: int,
    fact_count: int,
    fact_characters: int,
) -> list[str]:
    demands = []
    if issue_count == 0:
        demands.append("issue_spotting_gap")
    elif issue_count == 1:
        demands.append("focused_issue_resolution")
    elif issue_count >= 3:
        demands.append("multi_issue_composition")
    else:
        demands.append("dual_issue_resolution")

    if law_count:
        demands.append("supplied_rule_extraction")
    else:
        demands.append("rule_recall_or_doctrine_identification")
    if precedent_count:
        demands.append("precedent_or_analogy_handling")
    if "procedure_appeal" in _families(text):
        demands.append("procedural_threshold_check")
    if EVIDENCE_PATTERN.search(text):
        demands.append("evidence_and_burden_assessment")
    if DEFENSE_PATTERN.search(text):
        demands.append("defense_or_counterargument_check")
    if REMEDY_DISCRETION_PATTERN.search(text):
        demands.append("remedy_discretion_check")
    if fact_count >= 12 or fact_characters >= 2200:
        demands.append("long_fact_filtering")
    return demands


def _trajectory(families: list[str], demands: list[str]) -> list[str]:
    trajectory = ["case_profile"]
    if "issue_spotting_gap" in demands:
        trajectory.append("issue_spotting")
    elif "multi_issue_composition" in demands:
        trajectory.append("issue_decomposition")
    else:
        trajectory.append("issue_confirmation")
    if "long_fact_filtering" in demands:
        trajectory.append("material_fact_filtering")
    if "procedural_threshold_check" in demands:
        trajectory.append("procedural_threshold")
    if "supplied_rule_extraction" in demands:
        trajectory.append("rule_extraction")
    else:
        trajectory.append("rule_or_doctrine_identification")
    for family in families[:3]:
        trajectory.append(f"domain_template:{family}")
    if "precedent_or_analogy_handling" in demands:
        trajectory.append("precedent_or_analogy_check")
    if "evidence_and_burden_assessment" in demands:
        trajectory.append("evidence_burden_assessment")
    if "defense_or_counterargument_check" in demands:
        trajectory.append("defense_counterargument_check")
    if "remedy_discretion_check" in demands:
        trajectory.append("remedy_discretion_check")
    if "multi_issue_composition" in demands:
        trajectory.append("issue_composition")
    trajectory.append("final_decision")
    return trajectory


def _estimate_fact_count(facts: str) -> int:
    text = str(facts or "").strip()
    if not text:
        return 0
    numbered = re.findall(r"\bF\d+\b", text)
    if numbered:
        return len(set(numbered))
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return len([sentence for sentence in sentences if sentence.strip()])
