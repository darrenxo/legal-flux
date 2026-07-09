from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Any

import pandas as pd


OUTCOME_PATTERNS = {
    "explicit_court_outcome": re.compile(
        r"\bthe (?:court|tribunal|judge) "
        r"(?:held|found|concluded|determined|ordered|awarded|dismissed|"
        r"allowed|granted|rejected|ruled|decided)\b",
        re.IGNORECASE,
    ),
    "claim_disposition": re.compile(
        r"\b(?:claim|claims|appeal|application|counterclaim) "
        r"(?:was|were|is|are|be) "
        r"(?:dismissed|allowed|granted|rejected|struck out)\b",
        re.IGNORECASE,
    ),
    "liability_conclusion": re.compile(
        r"\b(?:plaintiff|defendant|respondent|appellant) "
        r"(?:was|is|were|are) (?:not )?liable\b",
        re.IGNORECASE,
    ),
    "award_or_order": re.compile(
        r"\b(?:shall pay|awarded damages|damages (?:of|in the sum of)|"
        r"judgment (?:for|against)|final order)\b",
        re.IGNORECASE,
    ),
    "credibility_finding": re.compile(
        r"\b(?:truthful|reliable|evasive|unreliable|credible) witness\b|"
        r"\b(?:accepted|rejected) (?:the )?(?:evidence|testimony)\b",
        re.IGNORECASE,
    ),
    "judgment_entered": re.compile(
        r"\bjudg(?:e)?ment\b.{0,100}\b(?:entered|given|granted|"
        r"in favou?r|against|liability|damages|rescinding|ordering)\b",
        re.IGNORECASE,
    ),
    "admitted_liability": re.compile(
        r"\badmitted liability\b",
        re.IGNORECASE,
    ),
    "judicial_credibility": re.compile(
        r"\b(?:court|tribunal|judge)\b.{0,80}\b(?:preferred|accepted|rejected|"
        r"not satisfied|found)\b.{0,80}\b(?:evidence|testimony|witness|honest|"
        r"credible|reliable)\b",
        re.IGNORECASE,
    ),
    "appeal_disposition": re.compile(
        r"\bappeal\b.{0,60}\b(?:dismissed|allowed|upheld|rejected)\b|"
        r"\bretrial was ordered\b",
        re.IGNORECASE,
    ),
    "costs_awarded": re.compile(
        r"\bcosts were awarded\b",
        re.IGNORECASE,
    ),
    "evaluative_conclusion": re.compile(
        r"\b(?:no|insufficient) evidence to support\b|"
        r"\bfailed to (?:provide|adduce|establish|prove) evidence\b|"
        r"\bevidence\b.{0,60}\bdid not support\b|"
        r"\ballegations?\b.{0,40}\b(?:vague|imprecise|unsupported)\b|"
        r"\bthe (?:court|tribunal|judge|master|registrar)\b.{0,100}"
        r"\b(?:received|made|found|considered|preferred|accepted|rejected|"
        r"ordered|granted|dismissed|allowed|concluded|determined)\b",
        re.IGNORECASE,
    ),
    "legal_or_credibility_conclusion": re.compile(
        r"\b(?:is|are|was|were) (?:only )?(?:not )?entitled to\b|"
        r"\bevidence\b.{0,50}\bmore credible\b|"
        r"\b(?:not been|not being|was not|were not) honest\b|"
        r"\b(?:demonstrated|showed|established)\b.{0,40}\bexaggerat\w*\b|"
        r"\b(?:credible|credibility|dishonest|exaggeration)\b",
        re.IGNORECASE,
    ),
    "party_evaluation": re.compile(
        r"\b(?:plaintiff|defendant|applicant|respondent)(?:'s)?\b.{0,80}"
        r"\b(?:contentions? were inconsistent|equally responsible)\b|"
        r"\b(?:contentions? were inconsistent|equally responsible)\b",
        re.IGNORECASE,
    ),
}

CRIMINAL_TERMS = re.compile(
    r"\b(?:criminal|conviction|convicted|sentence|sentencing|prosecution|"
    r"prosecutor|bail|offence|offense|indictment|charge|trafficking|"
    r"imprisonment|custodial|habeas corpus|non-refoulement|refugee|"
    r"torture claim|immigration|deportation|removal order|persecution|"
    r"criminal conviction|case stated)\b",
    re.IGNORECASE,
)

STRICT_EVALUATION_TERMS = re.compile(
    r"\b(?:court|tribunal|judge|master|registrar|judg(?:e)?ment|appeal|held|"
    r"ordered|dismissed|allowed|granted|rejected|ruled|decided|found|"
    r"concluded|determined|liable|liability|entitled|credible|credibility|"
    r"honest|dishonest|exaggerat\w*|unchallenged|no evidence|"
    r"insufficient evidence|failed to prove|failed to establish|"
    r"did not support|not enforceable|enforceable|binding|valid|invalid|"
    r"satisfactory|unsatisfactory|struck out|set aside|summary judgment|"
    r"default judgment|interlocutory judgment|consent order|costs order|"
    r"leave to defend)\b",
    re.IGNORECASE,
)


def explicit_leakage_reasons(
    facts: str,
    *,
    judgment_decision: str,
    ngram_size: int,
    overlap_threshold: float,
) -> list[str]:
    reasons = [
        name for name, pattern in OUTCOME_PATTERNS.items() if pattern.search(facts)
    ]
    decision_ngrams = _ngrams(judgment_decision, ngram_size)
    if decision_ngrams:
        fact_ngrams = _ngrams(facts, ngram_size)
        overlap = len(decision_ngrams & fact_ngrams) / len(decision_ngrams)
        if overlap >= overlap_threshold:
            reasons.append("judgment_text_overlap")
    return reasons


def is_civil_legalhk_row(*, plaintiff: str, lawsuit_type: str, claim: str) -> bool:
    if str(plaintiff).strip().upper() == "HKSAR":
        return False
    text = f"{lawsuit_type} {claim}"
    return not bool(CRIMINAL_TERMS.search(text))


def strict_evaluation_reasons(facts: str) -> list[str]:
    return (
        ["strict_judicial_or_evaluative_language"]
        if STRICT_EVALUATION_TERMS.search(str(facts))
        else []
    )


def select_legalhk_splits(
    frame: pd.DataFrame,
    *,
    smoke_count: int,
    evaluation_count: int,
    seed: int,
    smoke_indices: list[Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if evaluation_count % 2:
        raise ValueError("evaluation_count must be even for exact class balance.")
    prepared = _add_strata(frame)
    if smoke_indices is not None:
        if len(smoke_indices) != smoke_count:
            raise ValueError("Reviewed smoke IDs must match smoke_count.")
        missing = set(smoke_indices) - set(prepared.index)
        if missing:
            raise ValueError(
                f"Reviewed smoke IDs are not eligible: {sorted(missing, key=str)}"
            )
        smoke = prepared.loc[smoke_indices].copy()
        remaining = prepared.drop(index=smoke_indices)
        evaluation_indices = _balanced_evaluation_indices(
            remaining, count=evaluation_count, seed=seed
        )
        return smoke, prepared.loc[evaluation_indices].copy()

    evaluation_indices = _balanced_evaluation_indices(
        prepared, count=evaluation_count, seed=seed
    )
    remaining = prepared.drop(index=evaluation_indices)
    smoke_targets = {
        "support": smoke_count // 2 + smoke_count % 2,
        "reject": smoke_count // 2,
    }
    smoke_indices = []
    for offset, outcome in enumerate(("support", "reject"), start=2):
        subset = remaining[remaining["support&reject"] == outcome]
        smoke_indices.extend(
            _stratified_indices(
                subset,
                count=smoke_targets[outcome],
                seed=seed + offset,
            )
        )
    if len(smoke_indices) != smoke_count:
        raise ValueError("Not enough eligible rows for the smoke split.")
    return prepared.loc[smoke_indices].copy(), prepared.loc[evaluation_indices].copy()


def _balanced_evaluation_indices(
    prepared: pd.DataFrame, *, count: int, seed: int
) -> list[Any]:
    if count % 2:
        raise ValueError("evaluation_count must be even for exact class balance.")
    per_class = count // 2
    evaluation_indices: list[Any] = []
    for offset, outcome in enumerate(("support", "reject")):
        subset = prepared[prepared["support&reject"] == outcome]
        evaluation_indices.extend(
            _stratified_indices(subset, count=per_class, seed=seed + offset)
        )
    if len(evaluation_indices) != count:
        raise ValueError("Not enough eligible rows for balanced evaluation.")
    return evaluation_indices


def _add_strata(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["issue_count"] = result["issues"].map(
        lambda value: len(
            [line for line in str(value).splitlines() if line.strip()]
        )
    )
    result["issue_bucket"] = result["issue_count"].map(
        lambda count: "none" if count == 0 else "few" if count <= 2 else "many"
    )
    lengths = result["more_facts"].astype(str).str.len()
    ranked = lengths.rank(method="first")
    result["length_bucket"] = pd.qcut(
        ranked, q=3, labels=["short", "medium", "long"]
    ).astype(str)
    if "has_defense" not in result:
        result["has_defense"] = result["issues"].astype(str).str.contains(
            r"defen[cs]e|contributory|counterclaim|exception|whether.*liable",
            case=False,
            regex=True,
        )
    return result


def _stratified_indices(
    frame: pd.DataFrame, *, count: int, seed: int
) -> list[Any]:
    groups: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
    for index, row in frame.iterrows():
        key = (
            row["issue_bucket"],
            row["length_bucket"],
            bool(row["has_defense"]),
            str(row["lawsuit_type"]),
        )
        groups[key].append(index)
    rng = random.Random(seed)
    for indices in groups.values():
        rng.shuffle(indices)
    chosen: list[Any] = []
    keys = sorted(groups, key=str)
    while len(chosen) < count and any(groups.values()):
        for key in keys:
            if groups[key] and len(chosen) < count:
                chosen.append(groups[key].pop())
    return chosen


def _ngrams(text: str, size: int) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    if size <= 0 or len(tokens) < size:
        return set()
    return {
        tuple(tokens[index : index + size])
        for index in range(len(tokens) - size + 1)
    }
