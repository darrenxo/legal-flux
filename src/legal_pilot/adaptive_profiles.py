from __future__ import annotations

import re
from typing import Any

import pandas as pd


FAMILY_PATTERNS: dict[str, re.Pattern[str]] = {
    "contract_performance": re.compile(
        r"\b(contracts?|agreements?|breach(?:es|ed|ing)?|sale and purchase|s&p|"
        r"specific performance|terminat\w*|repudiat\w*|condition precedent|"
        r"requisitions?)\b",
        re.I,
    ),
    "debt_payment": re.compile(
        r"\b(debts?|loans?|invoices?|unpaid|arrears|deposits?|cheques?|"
        r"promissory|money due|sum due|rent arrears|payments?)\b",
        re.I,
    ),
    "property_possession": re.compile(
        r"\b(propert\w*|land|premises|tenan\w*|leas\w*|landlords?|"
        r"possess\w*|occupation|titles?|conveyanc\w*|licen[cs]\w*)\b",
        re.I,
    ),
    "tort_negligence_damage": re.compile(
        r"\b(negligence|duty of care|torts?|accidents?|injur\w*|damag\w*|"
        r"causation|nuisance|defamation|collisions?)\b",
        re.I,
    ),
    "employment_compensation": re.compile(
        r"\b(labou?r|employment|employees?|employers?|wages?|severance|"
        r"employee'?s compensation|work injur\w*|dismiss\w*)\b",
        re.I,
    ),
    "company_insolvency": re.compile(
        r"\b(compan(?:y|ies)|companies ordinance|shareholders?|directors?|"
        r"liquidat\w*|winding up|insolven\w*|creditors?|unfair prejudice|"
        r"scheme of arrangement)\b",
        re.I,
    ),
    "procedure_appeal": re.compile(
        r"\b(appeals?|leave|extension of time|set aside|strike out|security for"
        r" costs|summary judgment|judicial review|case stated|interlocutory|"
        r"jurisdiction\w*|procedural)\b",
        re.I,
    ),
    "criminal_procedure": re.compile(
        r"\b(criminal|conviction|sentence|sentencing|prosecution|hksar|bail|"
        r"offen[cs]\w*|charg\w*|trafficking|indict\w*)\b",
        re.I,
    ),
    "immigration_non_refoulement": re.compile(
        r"\b(immigration|refugee\w*|torture claim|non-refoulement|"
        r"deport\w*|removal order|screening decision)\b",
        re.I,
    ),
    "public_law_judicial_review": re.compile(
        r"\b(judicial review|public law|administrative law|director of|"
        r"commissioner|secretary for|tribunal|board|public officer|"
        r"leave to apply for judicial review)\b",
        re.I,
    ),
    "trust_probate_family": re.compile(
        r"\b(trusts?|estates?|probate|will|inheritance|matrimonial|divorc\w*|"
        r"maintenance|famil\w*)\b",
        re.I,
    ),
}

EVIDENCE_PATTERN = re.compile(
    r"\b(evidence|witness\w*|credib\w*|proof|prov\w*|burden\w*|conflict\w*|"
    r"contradict\w*|alleg\w*|testimony|experts?|documentary)\b",
    re.I,
)
DEFENSE_PATTERN = re.compile(
    r"\b(defen[cs]\w*|counterclaim\w*|set-?offs?|limitation\w*|estoppel|"
    r"laches|mitigat\w*|contributory|waiv\w*|illegal\w*|"
    r"jurisdictional objection\w*)\b",
    re.I,
)
REMEDY_DISCRETION_PATTERN = re.compile(
    r"\b(specific performance|injunction\w*|declar\w*|stay\w*|set aside|"
    r"leave|extension\w*|security for costs|equitable|discretion\w*)\b",
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
    return matched or ["general_legal_reasoning"]


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
