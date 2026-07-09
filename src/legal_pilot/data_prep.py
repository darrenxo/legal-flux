from __future__ import annotations

import json
import random
import shutil
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import httpx
import pandas as pd

from .config import resolve_path
from .io_utils import split_facts, write_jsonl
from .legalhk_selection import (
    explicit_leakage_reasons,
    is_civil_legalhk_row,
    select_legalhk_splits,
    strict_evaluation_reasons,
)
from .models import CaseState, Element, Issue, NormalizedCase


OPENEXEMPT_ARCHIVES = {
    "baseline": "baseline_robustness.tar.gz",
    "distractor": "distractor_robustness.tar.gz",
    "obfuscation": "obfuscation_robustness.tar.gz",
    "sycophancy": "sycophancy_robustness.tar.gz",
    "reasoning_decomposition": "reasoning_decomposition.tar.gz",
}
OPENEXEMPT_BASE_URL = (
    "https://huggingface.co/datasets/SergioServantez/OpenExempt/"
    "resolve/main/data/{filename}?download=true"
)
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


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe archive member: {member.name}")
        handle.extractall(destination, filter="data")


def prepare_datasets(config: dict[str, Any]) -> dict[str, Any]:
    raw_dir = resolve_path(config, "raw_dir")
    processed_dir = resolve_path(config, "processed_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    datasets = config["data"].get("datasets", ["openexempt", "legalhk"])
    openexempt = (
        prepare_openexempt(
            raw_dir,
            base_count=config["data"]["openexempt_base_cases"],
            seed=config["project"]["seed"],
        )
        if "openexempt" in datasets
        else []
    )
    legalhk: list[NormalizedCase] = []
    legalhk_selection: dict[str, Any] = {}
    review_rows: list[dict[str, Any]] = []
    if "legalhk" in datasets:
        legalhk, legalhk_selection, review_rows = prepare_legalhk(
            raw_dir,
            smoke_count=config["data"].get("smoke_cases", 0),
            evaluation_count=config["data"].get(
                "evaluation_cases", config["data"].get("legalhk_cases", 0)
            ),
            seed=config["project"]["seed"],
            max_characters=config["data"]["max_input_characters"],
            ngram_size=config["data"].get("decision_overlap_ngram", 6),
            overlap_threshold=config["data"].get(
                "decision_overlap_threshold", 0.12
            ),
            smoke_case_ids=config["data"].get("smoke_case_ids"),
            excluded_evaluation_case_ids=config["data"].get(
                "excluded_evaluation_case_ids", []
            ),
        )
    all_cases = openexempt + legalhk
    write_jsonl(
        processed_dir / "cases.jsonl",
        [case.model_dump(mode="json") for case in all_cases],
    )
    if review_rows:
        write_jsonl(processed_dir / "selection_review.jsonl", review_rows)
    smoke_cases = sum(
        case.metadata.get("selection_split") == "smoke" for case in legalhk
    )
    evaluation = [
        case
        for case in legalhk
        if case.metadata.get("selection_split") == "evaluation"
    ]
    manifest = {
        "datasets": datasets,
        "openexempt_cases": len(openexempt),
        "legalhk_cases": len(legalhk),
        "smoke_cases": smoke_cases,
        "evaluation_cases": len(evaluation),
        "evaluation_outcomes": dict(
            sorted(Counter(case.gold_answer for case in evaluation).items())
        ),
        "smoke_evaluation_overlap": 0,
        "total_cases": len(all_cases),
        "leakage_screen": legalhk_selection,
        "notes": _dataset_notes(datasets),
    }
    (processed_dir / "prepare_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _dataset_notes(datasets: list[str]) -> list[str]:
    notes: list[str] = []
    if "openexempt" in datasets:
        notes.append(
                "OpenExempt robustness cases use official symbolic gold labels. "
                "The released suites are controlled strata, not same-case pairs."
        )
    if "legalhk" in datasets:
        notes.append(
                "LegalHK is retained locally only; its processed release has an "
                "unclear license."
        )
    return notes


def prepare_openexempt(
    raw_dir: Path, *, base_count: int, seed: int
) -> list[NormalizedCase]:
    archive_dir = raw_dir / "openexempt_archives"
    extract_dir = raw_dir / "openexempt"
    for suite, filename in OPENEXEMPT_ARCHIVES.items():
        archive = archive_dir / filename
        download_file(OPENEXEMPT_BASE_URL.format(filename=filename), archive)
        suite_destination = extract_dir / suite
        if not suite_destination.exists():
            safe_extract_tar(archive, suite_destination)

    rng = random.Random(seed)
    baseline_rows = _load_openexempt_suite(extract_dir / "baseline", "baseline")
    base_selected = _stratified_sample(
        baseline_rows, count=base_count, key=lambda row: row["task_family"], rng=rng
    )

    challenge_selected: list[dict[str, Any]] = []
    for suite in (
        "distractor",
        "obfuscation",
        "sycophancy",
        "reasoning_decomposition",
    ):
        rows = _load_openexempt_suite(extract_dir / suite, suite)
        challenge_selected.extend(
            _stratified_sample(
                rows,
                count=base_count // 4,
                key=lambda row: row["task_family"],
                rng=rng,
            )
        )

    return [
        _normalize_openexempt(row, variant_id="original")
        for row in base_selected
    ] + [
        _normalize_openexempt(row, variant_id=row["suite"])
        for row in challenge_selected
    ]


def _load_openexempt_suite(root: Path, suite: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for test_path in root.rglob("test.jsonl"):
        task_dir = test_path.parent
        shared_path = task_dir / "shared.json"
        if not shared_path.exists():
            continue
        shared = json.loads(shared_path.read_text(encoding="utf-8"))
        for row in _read_jsonl_iter(test_path):
            rows.append(
                {
                    **row,
                    "suite": suite,
                    "task_family": _task_family(task_dir.name),
                    "task_dir": task_dir.name,
                    "shared": shared,
                }
            )
    return rows


def _normalize_openexempt(
    row: dict[str, Any], *, variant_id: str
) -> NormalizedCase:
    shared = row["shared"]
    facts = split_facts(row.get("facts", ""))
    solved_steps = row.get("solved_steps")
    if solved_steps:
        offset = len(facts)
        for index, line in enumerate(str(solved_steps).splitlines(), start=1):
            if line.strip():
                facts[f"F{offset + index}"] = f"Provided prior result: {line.strip()}"
    claim = shared.get("instruction", f"Solve OpenExempt task {row['task_family']}.")
    oracle = _openexempt_reference_state(row, facts)
    return NormalizedCase(
        dataset="openexempt",
        case_id=row["uid"],
        variant_id=variant_id,
        pair_id=None,
        perturbation_kind=None if variant_id == "original" else variant_id,
        claim=claim,
        requested_remedy="Return the exact answer requested by the task.",
        parties=[],
        facts=facts,
        authorities=shared.get("statutes"),
        gold_answer=str(row.get("solution", "")),
        reference_issues=[oracle.issues[0].issue],
        reference_state=oracle,
        metadata={
            "suite": row["suite"],
            "task_family": row["task_family"],
            "task_dir": row["task_dir"],
            "jurisdiction": row.get("jurisdiction"),
            "official_symbolic_gold": True,
            "paired": False,
        },
    )


def _openexempt_reference_state(
    row: dict[str, Any], facts: dict[str, str]
) -> CaseState:
    family = row["task_family"]
    labels = {
        "allowable_exemptions": (
            "Which exemption jurisdiction applies?",
            "Apply the Bankruptcy Code domicile rule and state opt-out law.",
        ),
        "exemption_classification": (
            "Which statutory exemption categories cover each asset?",
            "Match asset facts to statutory property categories.",
        ),
        "exemption_valuation": (
            "What value is protected by each available exemption?",
            "Apply statutory eligibility and dollar caps.",
        ),
        "nonexempt_assets": (
            "What value remains non-exempt after allocating exemptions?",
            "Allocate exemptions and compute residual non-exempt value.",
        ),
        "optimal_exemptions": (
            "Which exemption schedule minimizes non-exempt value?",
            "Compare allowable schedules and select the optimum.",
        ),
    }
    issue_text, rule = labels.get(
        family, ("What is the legally correct result?", "Apply the supplied statutes.")
    )
    return CaseState(
        claims=[row["shared"].get("instruction", issue_text)],
        requested_remedies=["Exact benchmark answer"],
        issues=[
            Issue(
                issue_id="I1",
                issue=issue_text,
                rule_or_test=rule,
                burden_on="unclear",
                elements=[
                    Element(
                        element_id="E1",
                        element=rule,
                        supporting_fact_ids=list(facts),
                        opposing_fact_ids=[],
                        missing_information=[],
                    )
                ],
                defenses=[],
            )
        ],
    )


def prepare_legalhk(
    raw_dir: Path,
    *,
    smoke_count: int,
    evaluation_count: int,
    seed: int,
    max_characters: int,
    ngram_size: int,
    overlap_threshold: float,
    smoke_case_ids: list[str] | None = None,
    excluded_evaluation_case_ids: list[str] | None = None,
) -> tuple[list[NormalizedCase], dict[str, Any], list[dict[str, Any]]]:
    parquet_path = raw_dir / "legalhk" / "train.parquet"
    download_file(LEGALHK_PARQUET_URL, parquet_path)
    frame = pd.read_parquet(parquet_path)
    frame = frame.fillna("")
    excluded_reasons: Counter[str] = Counter()
    eligible_indices: list[Any] = []
    for index, row in frame.iterrows():
        reasons: list[str] = []
        outcome = str(row["support&reject"]).strip().lower()
        if outcome not in {"support", "reject"}:
            reasons.append("non_binary_outcome")
        if not str(row["plaintiff_claim"]).strip():
            reasons.append("empty_claim")
        if not str(row["more_facts"]).strip():
            reasons.append("empty_facts")
        if len(str(row["plaintiff_claim"])) + len(str(row["more_facts"])) > max_characters:
            reasons.append("input_too_long")
        if not is_civil_legalhk_row(
            plaintiff=str(row["plaintiff"]),
            lawsuit_type=str(row["lawsuit_type"]),
            claim=str(row["plaintiff_claim"]),
        ):
            reasons.append("not_civil")
        reasons.extend(
            explicit_leakage_reasons(
                str(row["more_facts"]),
                judgment_decision=str(row["judgment_decision"]),
                ngram_size=ngram_size,
                overlap_threshold=overlap_threshold,
            )
        )
        if reasons:
            excluded_reasons.update(set(reasons))
        else:
            eligible_indices.append(index)
    eligible = frame.loc[eligible_indices].copy()
    eligible["support&reject"] = (
        eligible["support&reject"].astype(str).str.strip().str.lower()
    )
    eligible["has_defense"] = eligible["issues"].astype(str).str.contains(
        r"defen[cs]e|contributory|counterclaim|exception|whether.*liable",
        case=False,
        regex=True,
    )
    reviewed_smoke_indices = (
        [_legalhk_index(case_id) for case_id in smoke_case_ids]
        if smoke_case_ids is not None
        else None
    )
    if reviewed_smoke_indices is not None:
        missing = set(reviewed_smoke_indices) - set(eligible.index)
        if missing:
            raise ValueError(
                f"Reviewed smoke IDs are not eligible: {sorted(missing)}"
            )
        smoke = eligible.loc[reviewed_smoke_indices].copy()
        evaluation_source = eligible.drop(index=reviewed_smoke_indices)
    else:
        smoke = pd.DataFrame(columns=eligible.columns)
        evaluation_source = eligible
    excluded_evaluation_indices = {
        _legalhk_index(case_id)
        for case_id in (excluded_evaluation_case_ids or [])
    }
    evaluation_source = evaluation_source.drop(
        index=list(excluded_evaluation_indices & set(evaluation_source.index))
    )

    strict_mask = evaluation_source["more_facts"].map(
        lambda value: not strict_evaluation_reasons(str(value))
    )
    strict_evaluation_pool = evaluation_source[strict_mask].copy()
    if reviewed_smoke_indices is not None:
        _, evaluation = select_legalhk_splits(
            strict_evaluation_pool,
            smoke_count=0,
            evaluation_count=evaluation_count,
            seed=seed,
        )
    else:
        smoke, evaluation = select_legalhk_splits(
            strict_evaluation_pool,
            smoke_count=smoke_count,
            evaluation_count=evaluation_count,
            seed=seed,
        )

    cases: list[NormalizedCase] = []
    review_rows: list[dict[str, Any]] = []
    selected_parts = (("smoke", smoke), ("evaluation", evaluation))
    for split, selected in selected_parts:
        for index, row in selected.iterrows():
            case = _normalize_legalhk_case(index, row, split=split)
            cases.append(case)
            review_rows.append(
                {
                    "case_id": case.case_id,
                    "selection_split": split,
                    "claim": case.claim,
                    "requested_remedy": case.requested_remedy,
                    "parties": case.parties,
                    "facts": case.facts,
                    "lawsuit_type": case.metadata["lawsuit_type"],
                    "issue_count": case.metadata["issue_count"],
                    "fact_characters": case.metadata["fact_characters"],
                    "has_defense": case.metadata["has_defense"],
                    "leakage_screen": "auto_pass",
                }
            )
    selection = {
        "status": "low_explicit_leakage_subset",
        "source_rows": len(frame),
        "eligible_rows": len(eligible),
        "excluded_rows": len(frame) - len(eligible),
        "excluded_reasons": dict(sorted(excluded_reasons.items())),
        "selected_smoke": len(smoke),
        "selected_evaluation": len(evaluation),
        "strict_evaluation_pool_rows": len(strict_evaluation_pool),
        "strict_evaluation_excluded_rows": (
            len(evaluation_source) - len(strict_evaluation_pool)
        ),
        "manually_excluded_evaluation_cases": len(
            excluded_evaluation_case_ids or []
        ),
        "smoke_review_status": (
            "manually_reviewed" if smoke_case_ids is not None else "automatic_only"
        ),
        "evaluation_review_status": "local_audit_pending",
        "raw_archive_downloaded": False,
        "limitation": (
            "The processed more_facts field was enhanced from judgments; "
            "screening removes explicit disclosures but cannot remove latent "
            "outcome conditioning."
        ),
    }
    return cases, selection, review_rows


def _legalhk_index(case_id: str) -> int:
    prefix = "legalhk-"
    if not str(case_id).startswith(prefix):
        raise ValueError(f"Invalid LegalHK case ID: {case_id}")
    return int(str(case_id)[len(prefix) :])


def _normalize_legalhk_case(
    index: Any, row: pd.Series, *, split: str
) -> NormalizedCase:
        facts = split_facts(row["more_facts"])
        issues = [line.strip() for line in row["issues"].splitlines() if line.strip()]
        reference_state = _legalhk_reference_state(row, facts, issues)
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
            gold_answer=row["support&reject"].strip().lower(),
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


def _legalhk_reference_state(
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


def _stratified_sample(
    rows: list[dict[str, Any]],
    *,
    count: int,
    key,
    rng: random.Random,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    while len(selected) < count and any(groups.values()):
        for group_key in keys:
            if groups[group_key] and len(selected) < count:
                selected.append(groups[group_key].pop())
    return selected


def _sample_frame_strata(
    frame: pd.DataFrame, *, count: int, seed: int, strata: list[str]
) -> pd.DataFrame:
    rng = random.Random(seed)
    groups: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
    for index, row in frame.iterrows():
        groups[tuple(row[column] for column in strata)].append(index)
    for values in groups.values():
        rng.shuffle(values)
    chosen: list[Any] = []
    keys = sorted(groups, key=str)
    while len(chosen) < count and any(groups.values()):
        for group_key in keys:
            if groups[group_key] and len(chosen) < count:
                chosen.append(groups[group_key].pop())
    return frame.loc[chosen]


def _task_family(name: str) -> str:
    for family in (
        "allowable_exemptions",
        "exemption_classification",
        "exemption_valuation",
        "nonexempt_assets",
        "optimal_exemptions",
    ):
        if family in name:
            return family
    return name


def _line_count(value: str) -> int:
    return len([line for line in str(value).splitlines() if line.strip()])


def _read_jsonl_iter(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
