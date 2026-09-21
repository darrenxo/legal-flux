from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from sklearn.metrics import f1_score


CONDITION = "flux_rf_style"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two scored LegalFlux runs case by case and export full "
            "planner-to-reviewer trace dossiers."
        )
    )
    parser.add_argument("--sft-run-dir", type=Path, required=True)
    parser.add_argument("--dpo-run-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument(
        "--templates",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "templates"
            / "legal_flux_templates_v0.jsonl"
        ),
        help="Template library used by both runs; defaults to the repository v0 pool.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples-per-category", type=int, default=2)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=4,
        help="Maximum executed steps configured for these runs (default: 4).",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object in {path}:{line_number}.")
        rows.append(value)
    return rows


def case_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("dataset") or ""),
        str(row.get("case_id") or ""),
        str(row.get("variant_id") or "original"),
    )


def load_scored_run(run_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows = read_jsonl(run_dir / "scored.jsonl")
    selected = [row for row in rows if row.get("condition") == CONDITION]
    by_case: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in selected:
        key = case_key(row)
        if not all(key):
            raise ValueError(f"Scored row has an incomplete case identity: {key!r}")
        if key in by_case:
            raise ValueError(f"Duplicate scored case identity in {run_dir}: {key!r}")
        by_case[key] = row
    if not by_case:
        raise ValueError(f"No {CONDITION!r} rows found under {run_dir}.")
    return by_case


def load_cases(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    cases: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = case_key(row)
        if key in cases:
            raise ValueError(f"Duplicate normalized case identity: {key!r}")
        cases[key] = row
    return cases


def load_templates(path: Path) -> dict[str, dict[str, Any]]:
    templates: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        template_id = str(row.get("template_id") or "")
        if not template_id:
            raise ValueError(f"Template without template_id in {path}.")
        if template_id in templates:
            raise ValueError(f"Duplicate template_id in {path}: {template_id}")
        templates[template_id] = row
    if not templates:
        raise ValueError(f"No templates found in {path}.")
    return templates


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def template_library_hash(templates: dict[str, dict[str, Any]]) -> str:
    payload = canonical_json(list(templates.values()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_template_library_hash(
    *,
    label: str,
    rows: dict[tuple[str, str, str], dict[str, Any]],
    actual_hash: str,
) -> None:
    recorded = {
        str(row.get("template_pool_hash"))
        for row in rows.values()
        if row.get("template_pool_hash")
    }
    mismatches = sorted(value for value in recorded if value != actual_hash)
    if mismatches:
        raise ValueError(
            f"The supplied template library does not match the {label} run. "
            f"actual={actual_hash!r}, recorded={sorted(recorded)!r}"
        )


def initial_steps(row: dict[str, Any]) -> list[dict[str, Any]]:
    plan = row.get("trajectory_plan")
    if not isinstance(plan, dict):
        return []
    steps = plan.get("planned_steps")
    return steps if isinstance(steps, list) else []


def semantic_steps(steps: Iterable[Any]) -> list[dict[str, Any]]:
    normalized = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        normalized.append(
            {
                "step_name": step.get("step_name"),
                "step_description": step.get("step_description"),
                "template_tags": step.get("template_tags") or [],
            }
        )
    return normalized


def initial_plan_steps_signature(row: dict[str, Any]) -> str:
    return canonical_json(semantic_steps(initial_steps(row)))


def template_ids(row: dict[str, Any]) -> list[str]:
    selected = row.get("selected_templates")
    if isinstance(selected, list):
        return [
            str(item.get("template_id"))
            for item in selected
            if isinstance(item, dict) and item.get("template_id")
        ]
    values = row.get("retrieved_template_ids")
    return [str(value) for value in values] if isinstance(values, list) else []


def artifact_signature(row: dict[str, Any]) -> str:
    artifacts = row.get("executed_steps") or []
    selections = row.get("selected_templates") or []
    reviewer_context = []
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            continue
        selected = (
            selections[index]
            if index < len(selections) and isinstance(selections[index], dict)
            else {}
        )
        reviewer_context.append(
            {
                "step_id": artifact.get("step_id"),
                "step_name": selected.get("step_name"),
                "template_id": artifact.get("template_id"),
                "template_name": selected.get("template_name"),
                "instantiated_result": artifact.get("instantiated_result"),
            }
        )
    return canonical_json(reviewer_context)


def first_artifact_signature(row: dict[str, Any]) -> str:
    artifacts = row.get("executed_steps") or []
    if not artifacts or not isinstance(artifacts[0], dict):
        return canonical_json(None)
    selected = row.get("selected_templates") or []
    selection = selected[0] if selected and isinstance(selected[0], dict) else {}
    return canonical_json(
        {
            "step_id": artifacts[0].get("step_id"),
            "step_name": selection.get("step_name"),
            "template_id": artifacts[0].get("template_id"),
            "template_name": selection.get("template_name"),
            "instantiated_result": artifacts[0].get("instantiated_result"),
        }
    )


def reviewer_context_signatures(
    row: dict[str, Any], *, max_steps: int
) -> list[dict[str, Any]]:
    """Reconstruct the state shown to each stored reviewer call.

    The current reviewer prompt receives the active executed trajectory plus the
    then-current remaining steps. Reviews do not store their input remaining list,
    so replay the runner's deterministic state transitions to recover it.
    """

    remaining = list(initial_steps(row))
    artifacts = row.get("executed_steps") or []
    selections = row.get("selected_templates") or []
    reviews = row.get("trajectory_reviews") or []
    contexts: list[dict[str, Any]] = []
    for review_index, review in enumerate(reviews):
        if not isinstance(review, dict):
            continue
        if review_index < len(artifacts):
            if remaining:
                remaining.pop(0)
            shown_remaining = list(remaining)
            executed_count = review_index + 1
        else:
            # This is the extra forced-finalization call after execution stopped.
            shown_remaining = []
            executed_count = len(artifacts)
        executed_row = {
            "executed_steps": artifacts[:executed_count],
            "selected_templates": selections[:executed_count],
        }
        force_final_answer = (
            review_index >= len(artifacts)
            or not shown_remaining
            or executed_count >= max_steps
        )
        contexts.append(
            {
                "decision": review.get("decision"),
                "force_final_answer": force_final_answer,
                "signature": canonical_json(
                    {
                        "executed_trajectory": json.loads(
                            artifact_signature(executed_row)
                        ),
                        "remaining_steps": shown_remaining,
                        "force_final_answer": force_final_answer,
                    }
                ),
            }
        )
        if review.get("decision") == "revise":
            revised = review.get("revised_remaining_steps")
            remaining = list(revised) if isinstance(revised, list) else []
    return contexts


def first_review_context_signature(row: dict[str, Any], *, max_steps: int) -> str:
    artifacts = row.get("executed_steps") or []
    hashes = row.get("prompt_hashes") or {}
    if artifacts and isinstance(artifacts[0], dict):
        step_id = artifacts[0].get("step_id")
        stored = hashes.get(f"rf_review_{step_id}")
        if stored:
            return f"prompt_hash:{stored}"
    contexts = reviewer_context_signatures(row, max_steps=max_steps)
    return str(contexts[0]["signature"]) if contexts else canonical_json(None)


def final_review_context_signature(row: dict[str, Any], *, max_steps: int) -> str:
    hashes = row.get("prompt_hashes") or {}
    stored_final = hashes.get("rf_review_final")
    if stored_final:
        return f"prompt_hash:{stored_final}"
    reviews = row.get("trajectory_reviews") or []
    artifacts = row.get("executed_steps") or []
    final_indexes = [
        index
        for index, review in enumerate(reviews)
        if isinstance(review, dict) and review.get("decision") == "final_answer"
    ]
    if final_indexes:
        index = final_indexes[-1]
        if index < len(artifacts) and isinstance(artifacts[index], dict):
            step_id = artifacts[index].get("step_id")
            stored = hashes.get(f"rf_review_{step_id}")
            if stored:
                return f"prompt_hash:{stored}"
    contexts = reviewer_context_signatures(row, max_steps=max_steps)
    finals = [item for item in contexts if item["decision"] == "final_answer"]
    return str(finals[-1]["signature"]) if finals else canonical_json(None)


def review_decisions(row: dict[str, Any]) -> list[str]:
    return [
        str(review.get("decision") or "")
        for review in (row.get("trajectory_reviews") or [])
        if isinstance(review, dict)
    ]


def retrieval_similarities(row: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for selected in row.get("selected_templates") or []:
        if not isinstance(selected, dict) or selected.get("similarity") is None:
            continue
        try:
            values.append(float(selected["similarity"]))
        except (TypeError, ValueError):
            continue
    return values


def number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def average(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def is_correct(row: dict[str, Any]) -> bool:
    return row.get("status") == "ok" and bool(row.get("answer_correct"))


def prediction(row: dict[str, Any]) -> str | None:
    if row.get("status") != "ok":
        return None
    value = row.get("prediction")
    if value is None:
        value = (row.get("parsed_json") or {}).get("final_decision")
    return str(value) if value is not None else None


def outcome_bucket(sft: dict[str, Any], dpo: dict[str, Any]) -> str:
    left, right = is_correct(sft), is_correct(dpo)
    if left and right:
        return "both_correct"
    if left:
        return "sft_only"
    if right:
        return "dpo_only"
    return "both_wrong"


def diagnostic_category(
    *,
    bucket: str,
    plan_equal: bool,
    templates_equal: bool,
    artifacts_equal: bool,
    first_review_context_equal: bool,
    final_review_context_equal: bool,
) -> str:
    if bucket not in {"sft_only", "dpo_only"}:
        return bucket
    direction = "regression" if bucket == "sft_only" else "improvement"
    if final_review_context_equal:
        return f"same_final_review_context_{direction}"
    if artifacts_equal:
        return f"same_artifacts_{direction}"
    if first_review_context_equal:
        return f"same_first_review_context_{direction}"
    if plan_equal and templates_equal:
        return f"same_plan_and_templates_{direction}"
    return f"changed_trajectory_{direction}"


def build_pairs(
    sft_rows: dict[tuple[str, str, str], dict[str, Any]],
    dpo_rows: dict[tuple[str, str, str], dict[str, Any]],
    *,
    max_steps: int = 4,
) -> list[dict[str, Any]]:
    paired: list[dict[str, Any]] = []
    for key in sorted(set(sft_rows) & set(dpo_rows), key=case_sort_key):
        sft, dpo = sft_rows[key], dpo_rows[key]
        if sft.get("status") != "ok" or dpo.get("status") != "ok":
            continue
        plan_equal = initial_plan_steps_signature(sft) == initial_plan_steps_signature(
            dpo
        )
        templates_equal = template_ids(sft) == template_ids(dpo)
        artifacts_equal = artifact_signature(sft) == artifact_signature(dpo)
        first_artifact_equal = (
            first_artifact_signature(sft) == first_artifact_signature(dpo)
        )
        first_review_context_equal = (
            first_review_context_signature(sft, max_steps=max_steps)
            == first_review_context_signature(dpo, max_steps=max_steps)
        )
        final_review_context_equal = (
            final_review_context_signature(sft, max_steps=max_steps)
            == final_review_context_signature(dpo, max_steps=max_steps)
        )
        if sft.get("gold_answer") != dpo.get("gold_answer"):
            raise ValueError(
                f"Gold-answer mismatch for paired case {key!r}: "
                f"{sft.get('gold_answer')!r} != {dpo.get('gold_answer')!r}"
            )
        bucket = outcome_bucket(sft, dpo)
        sft_sims = retrieval_similarities(sft)
        dpo_sims = retrieval_similarities(dpo)
        paired.append(
            {
                "dataset": key[0],
                "case_id": key[1],
                "variant_id": key[2],
                "gold_answer": sft.get("gold_answer"),
                "sft_prediction": prediction(sft),
                "dpo_prediction": prediction(dpo),
                "sft_correct": is_correct(sft),
                "dpo_correct": is_correct(dpo),
                "outcome_bucket": bucket,
                "diagnostic_category": diagnostic_category(
                    bucket=bucket,
                    plan_equal=plan_equal,
                    templates_equal=templates_equal,
                    artifacts_equal=artifacts_equal,
                    first_review_context_equal=first_review_context_equal,
                    final_review_context_equal=final_review_context_equal,
                ),
                "initial_plan_equal": plan_equal,
                "template_sequence_equal": templates_equal,
                "first_artifact_equal": first_artifact_equal,
                "first_review_context_equal": first_review_context_equal,
                "executed_artifacts_equal": artifacts_equal,
                "final_review_context_equal": final_review_context_equal,
                "sft_plan_steps": len(initial_steps(sft)),
                "dpo_plan_steps": len(initial_steps(dpo)),
                "sft_executed_steps": len(sft.get("executed_steps") or []),
                "dpo_executed_steps": len(dpo.get("executed_steps") or []),
                "sft_review_decisions": review_decisions(sft),
                "dpo_review_decisions": review_decisions(dpo),
                "sft_template_ids": template_ids(sft),
                "dpo_template_ids": template_ids(dpo),
                "sft_mean_retrieval_similarity": mean(sft_sims) if sft_sims else None,
                "dpo_mean_retrieval_similarity": mean(dpo_sims) if dpo_sims else None,
                "sft_output_tokens": number(sft.get("output_tokens")),
                "dpo_output_tokens": number(dpo.get("output_tokens")),
                "sft_calls": number(sft.get("calls")),
                "dpo_calls": number(dpo.get("calls")),
                "retrieval_similarity_delta": (
                    None
                    if not sft_sims or not dpo_sims
                    else mean(dpo_sims) - mean(sft_sims)
                ),
            }
        )
    return paired


def case_sort_key(key: tuple[str, str, str]) -> tuple[str, int, str, str]:
    match = re.search(r"(\d+)$", key[1])
    number_part = int(match.group(1)) if match else 10**18
    return key[0], number_part, key[1], key[2]


def binary_weighted_f1(rows: list[dict[str, Any]], field: str) -> float:
    return float(
        f1_score(
            [str(row["gold_answer"]) for row in rows],
            [str(row[field]) for row in rows],
            labels=["support", "reject"],
            average="weighted",
            zero_division=0,
        )
    )


def run_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    return {
        "accuracy": mean(bool(row[f"{prefix}_correct"]) for row in rows),
        "weighted_f1": binary_weighted_f1(rows, f"{prefix}_prediction"),
        "mean_plan_steps": average(number(row[f"{prefix}_plan_steps"]) for row in rows),
        "mean_executed_steps": average(
            number(row[f"{prefix}_executed_steps"]) for row in rows
        ),
        "mean_retrieval_similarity": average(
            number(row[f"{prefix}_mean_retrieval_similarity"]) for row in rows
        ),
        "mean_output_tokens": average(
            number(row[f"{prefix}_output_tokens"]) for row in rows
        ),
        "mean_calls": average(number(row[f"{prefix}_calls"]) for row in rows),
    }


def nested_value(row: dict[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def distinct_values(
    rows: dict[tuple[str, str, str], dict[str, Any]], *keys: str
) -> list[Any]:
    values = {
        canonical_json(value): value
        for row in rows.values()
        if (value := nested_value(row, *keys)) is not None
    }
    return [values[key] for key in sorted(values)]


def compare_pair_field(
    pairs: list[dict[str, Any]],
    sft_rows: dict[tuple[str, str, str], dict[str, Any]],
    dpo_rows: dict[tuple[str, str, str], dict[str, Any]],
    *keys: str,
) -> dict[str, int]:
    counts = Counter()
    for comparison in pairs:
        key = (
            str(comparison["dataset"]),
            str(comparison["case_id"]),
            str(comparison["variant_id"]),
        )
        left = nested_value(sft_rows[key], *keys)
        right = nested_value(dpo_rows[key], *keys)
        if left is None or right is None:
            counts["missing"] += 1
        elif canonical_json(left) == canonical_json(right):
            counts["equal"] += 1
        else:
            counts["different"] += 1
    return {
        "equal": counts["equal"],
        "different": counts["different"],
        "missing": counts["missing"],
    }


def build_control_audit(
    *,
    pairs: list[dict[str, Any]],
    sft_rows: dict[tuple[str, str, str], dict[str, Any]],
    dpo_rows: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    fields = {
        "template_pool_hash": ("template_pool_hash",),
        "executor_model": ("role_models", "executor"),
        "inference_runtime": ("inference_runtime",),
        "inference_runtime_version": ("inference_runtime_version",),
        "chat_template_kwargs": ("chat_template_kwargs",),
        "seed": ("seed",),
        "sample_index": ("sample_index",),
        "decoding": ("decoding",),
        "planner_prompt_hash": ("prompt_hashes", "rf_plan"),
    }
    comparisons = {
        name: compare_pair_field(pairs, sft_rows, dpo_rows, *keys)
        for name, keys in fields.items()
    }
    warnings = []
    for name, counts in comparisons.items():
        if counts["different"]:
            warnings.append(
                f"{name} differs for {counts['different']} paired cases; "
                "treat the comparison as confounded."
            )
        if counts["missing"]:
            warnings.append(
                f"{name} is missing on one or both sides for "
                f"{counts['missing']} paired cases."
            )
    return {
        "comparisons": comparisons,
        "warnings": warnings,
        "run_metadata": {
            "sft": {
                "workflow_hashes": distinct_values(sft_rows, "workflow_hash"),
                "model_digests": distinct_values(sft_rows, "model_digest"),
                "planner_models": distinct_values(
                    sft_rows, "role_models", "planner"
                ),
                "executor_models": distinct_values(
                    sft_rows, "role_models", "executor"
                ),
                "reviewer_models": distinct_values(
                    sft_rows, "role_models", "reviewer"
                ),
            },
            "dpo": {
                "workflow_hashes": distinct_values(dpo_rows, "workflow_hash"),
                "model_digests": distinct_values(dpo_rows, "model_digest"),
                "planner_models": distinct_values(
                    dpo_rows, "role_models", "planner"
                ),
                "executor_models": distinct_values(
                    dpo_rows, "role_models", "executor"
                ),
                "reviewer_models": distinct_values(
                    dpo_rows, "role_models", "reviewer"
                ),
            },
        },
    }


def paired_run_rows(
    pairs: list[dict[str, Any]],
    rows: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        rows[
            (
                str(comparison["dataset"]),
                str(comparison["case_id"]),
                str(comparison["variant_id"]),
            )
        ]
        for comparison in pairs
    ]


def trace_diagnostics(rows: list[dict[str, Any]], *, max_steps: int) -> dict[str, Any]:
    retrieval_modes = Counter()
    template_counts = Counter()
    repair_counts = Counter()
    review_paths = Counter()
    review_decisions_counter = Counter()
    plan_length_counts = Counter()
    executed_length_counts = Counter()
    duplicate_step_name_cases = 0
    empty_tag_steps = 0
    total_plan_steps = 0
    planning_analysis_characters = []
    tag_counts = []
    fewer_than_initial_plan = 0
    for row in rows:
        steps = initial_steps(row)
        artifacts = row.get("executed_steps") or []
        reviews = row.get("trajectory_reviews") or []
        plan_length_counts[len(steps)] += 1
        executed_length_counts[len(artifacts)] += 1
        if len(artifacts) < len(steps):
            fewer_than_initial_plan += 1
        names = [
            str(step.get("step_name") or "").strip().casefold()
            for step in steps
            if isinstance(step, dict)
        ]
        if len(names) != len(set(names)):
            duplicate_step_name_cases += 1
        for step in steps:
            if not isinstance(step, dict):
                continue
            tags = step.get("template_tags") or []
            total_plan_steps += 1
            tag_counts.append(len(tags))
            if not tags:
                empty_tag_steps += 1
        plan = row.get("trajectory_plan") or {}
        planning_analysis_characters.append(
            len(str(plan.get("planning_analysis") or ""))
        )
        decisions = review_decisions(row)
        review_paths[">".join(decisions)] += 1
        review_decisions_counter.update(decisions)
        for selected in row.get("selected_templates") or []:
            if not isinstance(selected, dict):
                continue
            retrieval_modes[str(selected.get("retrieval_mode") or "missing")] += 1
            template_counts[str(selected.get("template_id") or "missing")] += 1
        repair_counts.update(str(value) for value in row.get("repair_actions") or [])
    return {
        "prediction_counts": dict(Counter(prediction(row) for row in rows)),
        "plan_length_counts": {str(k): v for k, v in sorted(plan_length_counts.items())},
        "executed_length_counts": {
            str(k): v for k, v in sorted(executed_length_counts.items())
        },
        "max_step_plan_rate": (
            plan_length_counts[max_steps] / len(rows) if rows else None
        ),
        "executed_fewer_than_initial_plan_rate": (
            fewer_than_initial_plan / len(rows) if rows else None
        ),
        "duplicate_step_name_case_rate": (
            duplicate_step_name_cases / len(rows) if rows else None
        ),
        "empty_tag_step_rate": (
            empty_tag_steps / total_plan_steps if total_plan_steps else None
        ),
        "mean_tags_per_planned_step": mean(tag_counts) if tag_counts else None,
        "mean_planning_analysis_characters": (
            mean(planning_analysis_characters)
            if planning_analysis_characters
            else None
        ),
        "retrieval_mode_counts": dict(retrieval_modes.most_common()),
        "top_template_ids": dict(template_counts.most_common(20)),
        "review_decision_counts": dict(review_decisions_counter.most_common()),
        "review_path_counts": dict(review_paths.most_common()),
        "repair_action_counts": dict(repair_counts.most_common()),
    }


def build_summary(
    *,
    sft_rows: dict[tuple[str, str, str], dict[str, Any]],
    dpo_rows: dict[tuple[str, str, str], dict[str, Any]],
    pairs: list[dict[str, Any]],
    max_steps: int = 4,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("No paired successful SFT/DPO cases were found.")
    transitions = Counter(row["outcome_bucket"] for row in pairs)
    categories = Counter(row["diagnostic_category"] for row in pairs)
    flips = Counter(
        (str(row["sft_prediction"]), str(row["dpo_prediction"]))
        for row in pairs
        if row["sft_prediction"] != row["dpo_prediction"]
    )
    sft_metrics = run_metrics(pairs, "sft")
    dpo_metrics = run_metrics(pairs, "dpo")
    sft_paired_rows = paired_run_rows(pairs, sft_rows)
    dpo_paired_rows = paired_run_rows(pairs, dpo_rows)
    excluded_non_ok = [
        {
            "dataset": key[0],
            "case_id": key[1],
            "variant_id": key[2],
            "sft_status": sft_rows[key].get("status"),
            "dpo_status": dpo_rows[key].get("status"),
        }
        for key in sorted(set(sft_rows) & set(dpo_rows), key=case_sort_key)
        if sft_rows[key].get("status") != "ok"
        or dpo_rows[key].get("status") != "ok"
    ]
    return {
        "coverage": {
            "sft_records": len(sft_rows),
            "dpo_records": len(dpo_rows),
            "sft_ok": sum(row.get("status") == "ok" for row in sft_rows.values()),
            "dpo_ok": sum(row.get("status") == "ok" for row in dpo_rows.values()),
            "paired_ok": len(pairs),
            "sft_only_case_ids": sorted(
                [key[1] for key in set(sft_rows) - set(dpo_rows)]
            ),
            "dpo_only_case_ids": sorted(
                [key[1] for key in set(dpo_rows) - set(sft_rows)]
            ),
            "excluded_non_ok": excluded_non_ok,
        },
        "paired_metrics": {
            "sft": sft_metrics,
            "dpo": dpo_metrics,
            "dpo_minus_sft_accuracy": dpo_metrics["accuracy"] - sft_metrics["accuracy"],
            "dpo_minus_sft_weighted_f1": (
                dpo_metrics["weighted_f1"] - sft_metrics["weighted_f1"]
            ),
        },
        "control_audit": build_control_audit(
            pairs=pairs,
            sft_rows=sft_rows,
            dpo_rows=dpo_rows,
        ),
        "trace_diagnostics": {
            "sft": trace_diagnostics(sft_paired_rows, max_steps=max_steps),
            "dpo": trace_diagnostics(dpo_paired_rows, max_steps=max_steps),
        },
        "outcome_transitions": dict(transitions),
        "diagnostic_categories": dict(categories),
        "prediction_flips": {
            f"{left}->{right}": count for (left, right), count in sorted(flips.items())
        },
        "structural_agreement": {
            "initial_plan_equal": sum(row["initial_plan_equal"] for row in pairs),
            "template_sequence_equal": sum(
                row["template_sequence_equal"] for row in pairs
            ),
            "first_artifact_equal": sum(
                row["first_artifact_equal"] for row in pairs
            ),
            "first_review_context_equal": sum(
                row["first_review_context_equal"] for row in pairs
            ),
            "executed_artifacts_equal": sum(
                row["executed_artifacts_equal"] for row in pairs
            ),
            "final_review_context_equal": sum(
                row["final_review_context_equal"] for row in pairs
            ),
            "paired_ok": len(pairs),
        },
    }


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_pair_csv(path: Path, pairs: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(
            {key: csv_value(value) for key, value in row.items()} for row in pairs
        )


def select_examples(
    pairs: list[dict[str, Any]], *, examples_per_category: int
) -> list[dict[str, Any]]:
    if examples_per_category < 1:
        raise ValueError("examples_per_category must be at least 1.")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[str(row["diagnostic_category"])].append(row)
    selected: list[dict[str, Any]] = []
    for category in sorted(grouped):
        candidates = sorted(
            grouped[category],
            key=lambda row: (
                row["retrieval_similarity_delta"] is None,
                row["retrieval_similarity_delta"] or 0.0,
                case_sort_key(
                    (
                        str(row["dataset"]),
                        str(row["case_id"]),
                        str(row["variant_id"]),
                    )
                ),
            ),
        )
        if len(candidates) <= examples_per_category:
            selected.extend(candidates)
            continue
        if examples_per_category == 1:
            indexes = [len(candidates) // 2]
        else:
            indexes = [
                round(index * (len(candidates) - 1) / (examples_per_category - 1))
                for index in range(examples_per_category)
            ]
        selected.extend(candidates[index] for index in indexes)
    return selected


def enriched_template_trace(
    row: dict[str, Any], templates: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    enriched = []
    for selected in row.get("selected_templates") or []:
        if not isinstance(selected, dict):
            continue
        template_id = str(selected.get("template_id") or "")
        if template_id not in templates:
            raise KeyError(
                f"Selected template {template_id!r} is absent from the supplied "
                "template library."
            )
        enriched.append(
            {
                **selected,
                "library_template": templates.get(template_id),
            }
        )
    return enriched


def executed_trajectory(
    row: dict[str, Any], templates: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    selections = enriched_template_trace(row, templates)
    artifacts = row.get("executed_steps") or []
    length = max(len(selections), len(artifacts))
    return [
        {
            "position": index + 1,
            "selected_template": selections[index] if index < len(selections) else None,
            "executed_artifact": artifacts[index] if index < len(artifacts) else None,
        }
        for index in range(length)
    ]


def trace_payload(
    row: dict[str, Any], templates: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return {
        "status": row.get("status"),
        "prediction": prediction(row),
        "answer_correct": row.get("answer_correct"),
        "trajectory_plan": row.get("trajectory_plan"),
        "selected_templates": enriched_template_trace(row, templates),
        "executed_steps": row.get("executed_steps"),
        "executed_trajectory": executed_trajectory(row, templates),
        "trajectory_reviews": row.get("trajectory_reviews"),
        "final_output": row.get("parsed_json"),
        "prompt_hashes": row.get("prompt_hashes"),
        "role_models": row.get("role_models"),
        "model_digest": row.get("model_digest"),
        "inference_runtime": row.get("inference_runtime"),
        "inference_runtime_version": row.get("inference_runtime_version"),
        "decoding": row.get("decoding"),
        "repair_actions": row.get("repair_actions"),
        "schema_errors": row.get("schema_errors"),
        "calls": row.get("calls"),
        "prompt_tokens": row.get("prompt_tokens"),
        "output_tokens": row.get("output_tokens"),
        "raw_response": row.get("raw_response"),
    }


def build_dossiers(
    *,
    selected: list[dict[str, Any]],
    cases: dict[tuple[str, str, str], dict[str, Any]],
    templates: dict[str, dict[str, Any]],
    sft_rows: dict[tuple[str, str, str], dict[str, Any]],
    dpo_rows: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    dossiers = []
    for comparison in selected:
        key = (
            str(comparison["dataset"]),
            str(comparison["case_id"]),
            str(comparison["variant_id"]),
        )
        case = cases.get(key)
        if case is None:
            raise KeyError(f"Normalized case not found: {key!r}")
        dossiers.append(
            {
                "selection_category": comparison["diagnostic_category"],
                "comparison": comparison,
                "case": {
                    "dataset": case.get("dataset"),
                    "case_id": case.get("case_id"),
                    "variant_id": case.get("variant_id"),
                    "claim": case.get("claim"),
                    "requested_remedy": case.get("requested_remedy"),
                    "parties": case.get("parties"),
                    "facts": case.get("facts"),
                    "authorities": case.get("authorities"),
                    "reference_issues": case.get("reference_issues"),
                    "gold_answer": case.get("gold_answer"),
                },
                "sft": trace_payload(sft_rows[key], templates),
                "dpo": trace_payload(dpo_rows[key], templates),
            }
        )
    return dossiers


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def markdown_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def write_dossier_markdown(path: Path, dossiers: list[dict[str, Any]]) -> None:
    lines = [
        "# SFT versus DPO LegalFlux trace dossiers",
        "",
        (
            "These cases were selected deterministically by diagnostic category. "
            "Model-produced strings are preserved in the parsed trace objects; "
            "the companion JSONL also retains each concatenated raw response."
        ),
        (
            "Only `trajectory_plan` is a clean planner output. After a reviewer "
            "returns `revise`, later executed steps are reviewer-authored, so the "
            "full adaptive trajectory measures planner and reviewer jointly."
        ),
        "",
    ]
    for dossier in dossiers:
        case = dossier["case"]
        comparison = dossier["comparison"]
        lines.extend(
            [
                f"## {case['case_id']} — {dossier['selection_category']}",
                "",
                f"**Claim:** {case.get('claim')}",
                "",
                f"**Requested remedy:** {case.get('requested_remedy')}",
                "",
                f"**Gold:** {case.get('gold_answer')}",
                "",
                "**Parties:**",
                "",
            ]
        )
        for party in case.get("parties") or []:
            lines.append(f"- {party}")
        lines.extend(
            [
                "",
                "**Facts:**",
                "",
            ]
        )
        facts = case.get("facts") or {}
        for fact_id, text in facts.items():
            lines.append(f"- **{fact_id}:** {text}")
        lines.extend(
            [
                "",
                "**Authorities:**",
                "",
                "```json",
                markdown_json(case.get("authorities")),
                "```",
                "",
                "**Reference issues:**",
                "",
            ]
        )
        for issue in case.get("reference_issues") or []:
            lines.append(f"- {issue}")
        lines.extend(["", "**Comparison flags:**", "", "```json"])
        lines.append(
            markdown_json(
                {
                    key: comparison[key]
                    for key in (
                        "sft_prediction",
                        "dpo_prediction",
                        "initial_plan_equal",
                        "template_sequence_equal",
                        "first_artifact_equal",
                        "first_review_context_equal",
                        "executed_artifacts_equal",
                        "final_review_context_equal",
                    )
                }
            )
        )
        lines.extend(["```", ""])
        for label in ("sft", "dpo"):
            trace = dossier[label]
            lines.extend(
                [
                    f"### {label.upper()} trace",
                    "",
                    "```json",
                    markdown_json(
                        {
                            key: trace[key]
                            for key in (
                                "prediction",
                                "answer_correct",
                                "trajectory_plan",
                                "executed_trajectory",
                                "trajectory_reviews",
                                "final_output",
                                "prompt_hashes",
                                "role_models",
                                "repair_actions",
                                "schema_errors",
                                "calls",
                                "output_tokens",
                            )
                        }
                    ),
                    "```",
                    "",
                ]
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary["paired_metrics"]
    agreement = summary["structural_agreement"]
    coverage = summary["coverage"]
    audit = summary["control_audit"]
    paired = int(agreement["paired_ok"])

    def percent(value: float) -> str:
        return f"{100 * value:.2f}%"

    lines = [
        "# SFT versus DPO LegalFlux trace comparison",
        "",
        "## Coverage",
        "",
        (
            f"SFT records: {coverage['sft_records']}; DPO records: "
            f"{coverage['dpo_records']}; paired successful cases: {paired}."
        ),
        "",
        "## Paired metrics",
        "",
        "| Run | Accuracy | Weighted F1 | Mean plan steps | Mean executed steps | Mean retrieval similarity | Mean output tokens | Mean calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("sft", "dpo"):
        item = metrics[name]
        similarity = item["mean_retrieval_similarity"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name.upper(),
                    percent(item["accuracy"]),
                    percent(item["weighted_f1"]),
                    f"{item['mean_plan_steps']:.3f}",
                    f"{item['mean_executed_steps']:.3f}",
                    "" if similarity is None else f"{similarity:.4f}",
                    f"{item['mean_output_tokens']:.1f}",
                    f"{item['mean_calls']:.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Control audit",
            "",
            (
                "Planner and reviewer model identities are expected to differ. "
                "The fields below should otherwise match for a clean comparison."
            ),
            "",
            "| Field | Equal | Different | Missing |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, counts in audit["comparisons"].items():
        lines.append(
            f"| {key} | {counts['equal']} | {counts['different']} | "
            f"{counts['missing']} |"
        )
    if audit["warnings"]:
        lines.extend(["", "**Comparison warnings:**", ""])
        lines.extend(f"- {warning}" for warning in audit["warnings"])
    else:
        lines.extend(["", "No control-field mismatches were detected."])
    diagnostics = summary["trace_diagnostics"]
    lines.extend(
        [
            "",
            "## Trace behavior on paired cases",
            "",
            "| Signal | SFT | DPO |",
            "|---|---:|---:|",
        ]
    )
    diagnostic_fields = (
        ("max_step_plan_rate", "Plans using the maximum step count", "percent"),
        (
            "executed_fewer_than_initial_plan_rate",
            "Executed fewer steps than initially planned",
            "percent",
        ),
        ("duplicate_step_name_case_rate", "Plans with duplicate step names", "percent"),
        ("empty_tag_step_rate", "Planned steps with no retrieval tags", "percent"),
        ("mean_tags_per_planned_step", "Mean tags per planned step", "number"),
        (
            "mean_planning_analysis_characters",
            "Mean planning-analysis characters",
            "number",
        ),
    )
    for field, label, kind in diagnostic_fields:
        values = [diagnostics[name][field] for name in ("sft", "dpo")]
        formatted = [
            ""
            if value is None
            else (percent(value) if kind == "percent" else f"{value:.3f}")
            for value in values
        ]
        lines.append(f"| {label} | {formatted[0]} | {formatted[1]} |")
    lines.extend(
        [
            "",
            "**Prediction counts:**",
            "",
            "```json",
            markdown_json(
                {
                    name: diagnostics[name]["prediction_counts"]
                    for name in ("sft", "dpo")
                }
            ),
            "```",
            "",
            "**Retrieval modes:**",
            "",
            "```json",
            markdown_json(
                {
                    name: diagnostics[name]["retrieval_mode_counts"]
                    for name in ("sft", "dpo")
                }
            ),
            "```",
            "",
            "**Reviewer paths:**",
            "",
            "```json",
            markdown_json(
                {
                    name: diagnostics[name]["review_path_counts"]
                    for name in ("sft", "dpo")
                }
            ),
            "```",
        ]
    )
    lines.extend(
        [
            "",
            "## Outcome transitions",
            "",
            "| Bucket | Cases |",
            "|---|---:|",
        ]
    )
    for key, count in sorted(summary["outcome_transitions"].items()):
        lines.append(f"| {key} | {count} |")
    lines.extend(
        [
            "",
            "## Structural agreement",
            "",
            "| Signal | Equal cases | Share |",
            "|---|---:|---:|",
        ]
    )
    for key in (
        "initial_plan_equal",
        "template_sequence_equal",
        "first_artifact_equal",
        "first_review_context_equal",
        "executed_artifacts_equal",
        "final_review_context_equal",
    ):
        count = int(agreement[key])
        lines.append(f"| {key} | {count} | {100 * count / paired:.2f}% |")
    lines.extend(
        [
            "",
            "## Diagnostic categories",
            "",
            "| Category | Cases |",
            "|---|---:|",
        ]
    )
    for key, count in sorted(summary["diagnostic_categories"].items()):
        lines.append(f"| {key} | {count} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "`same_final_review_context_regression` means the final reviewer "
                "received the same reconstructed executed trajectory and remaining "
                "steps, but the SFT run was correct and the DPO run was wrong. This "
                "is the strongest existing-ledger evidence of reviewer/finalizer "
                "regression."
            ),
            "",
            (
                "`same_first_review_context_regression` means both runs gave the "
                "first reviewer the same executed trajectory, remaining steps, and "
                "output mode before their paths diverged. That isolates the first "
                "divergence to reviewer behavior. `same_artifacts_regression` keeps "
                "executed reasoning equal but may still have different remaining-"
                "step context."
            ),
            "",
            (
                "Changed-trajectory cases remain planner/reviewer-confounded. A "
                "DPO-planner + frozen-SFT-reviewer ablation is required to test the "
                "specific hypothesis that DPO improved planning but harmed review."
            ),
            "",
            (
                "The initial `trajectory_plan` is the only uncontaminated planner "
                "output. Any steps introduced after a `revise` review were authored "
                "by the reviewer, not by the original planner."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    sft_rows = load_scored_run(args.sft_run_dir.resolve())
    dpo_rows = load_scored_run(args.dpo_run_dir.resolve())
    cases = load_cases(args.cases.resolve())
    templates = load_templates(args.templates.resolve())
    supplied_template_hash = template_library_hash(templates)
    validate_template_library_hash(
        label="SFT", rows=sft_rows, actual_hash=supplied_template_hash
    )
    validate_template_library_hash(
        label="DPO", rows=dpo_rows, actual_hash=supplied_template_hash
    )
    if args.max_steps < 1:
        raise ValueError("max_steps must be at least 1.")
    pairs = build_pairs(sft_rows, dpo_rows, max_steps=args.max_steps)
    summary = build_summary(
        sft_rows=sft_rows,
        dpo_rows=dpo_rows,
        pairs=pairs,
        max_steps=args.max_steps,
    )
    summary["template_library"] = {
        "path": str(args.templates.resolve()),
        "template_count": len(templates),
        "template_pool_hash": supplied_template_hash,
    }
    selected = select_examples(
        pairs, examples_per_category=args.examples_per_category
    )
    dossiers = build_dossiers(
        selected=selected,
        cases=cases,
        templates=templates,
        sft_rows=sft_rows,
        dpo_rows=dpo_rows,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    pairs_path = output_dir / "paired_cases.csv"
    dossiers_path = output_dir / "selected_dossiers.jsonl"
    summary_markdown_path = output_dir / "summary.md"
    dossier_markdown_path = output_dir / "selected_dossiers.md"

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_pair_csv(pairs_path, pairs)
    write_jsonl(dossiers_path, dossiers)
    write_summary_markdown(summary_markdown_path, summary)
    write_dossier_markdown(dossier_markdown_path, dossiers)

    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "paired_cases": str(pairs_path),
                "selected_dossiers": str(dossiers_path),
                "summary_markdown": str(summary_markdown_path),
                "dossier_markdown": str(dossier_markdown_path),
                "paired_ok": len(pairs),
                "selected_examples": len(dossiers),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
