from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, recall_score

from .config import resolve_path
from .evaluation import bootstrap_ci
from .io_utils import latest_by_run_hash, read_jsonl, sha256_text, write_jsonl
from .runner import load_cases


PRINCIPAL_CONDITIONS = ("direct", "structured", "typed", "validated")


def build_report(config: dict[str, Any]) -> dict[str, Any]:
    run_dir = resolve_path(config, "runs_dir") / config["project"]["run_name"]
    report_dir = resolve_path(config, "reports_dir")
    report_dir.mkdir(parents=True, exist_ok=True)
    scored = read_jsonl(run_dir / "scored.jsonl")
    if not scored:
        raise FileNotFoundError("Scored outputs not found. Run score first.")
    history = read_jsonl(run_dir / "generations.jsonl")
    audits = _load_audit_records(run_dir)

    summary = _condition_summary(scored, history=history)
    strata = _stratum_summary(scored)
    oracle_gap = _oracle_gap(scored)
    paired = paired_condition_comparisons(
        scored, baseline="structured", seed=config["project"]["seed"]
    )
    audit_summary = _audit_summary(audits)
    summary.to_csv(report_dir / "condition_summary.csv", index=False)
    strata.to_csv(report_dir / "stratum_summary.csv", index=False)
    oracle_gap.to_csv(report_dir / "oracle_gap.csv", index=False)
    paired.to_csv(report_dir / "paired_condition_comparisons.csv", index=False)
    audit_summary.to_csv(report_dir / "audit_summary.csv", index=False)

    manual = _write_manual_review_packet(
        config, scored, report_dir, count=40, seed=config["project"]["seed"]
    )
    chart_path = _write_accuracy_chart(summary, report_dir)
    first_errors = Counter(
        row["audit"]["first_error"] for row in audits if row.get("audit")
    )
    recommendation = _recommend(
        summary, oracle_gap, audit_summary=audit_summary
    )
    report_path = report_dir / "pilot_report.md"
    report_path.write_text(
        _render_markdown(
            summary=summary,
            strata=strata,
            oracle_gap=oracle_gap,
            paired=paired,
            audit_summary=audit_summary,
            first_errors=first_errors,
            recommendation=recommendation,
            chart_path=chart_path,
            history=history,
            scored=scored,
            audits=audits,
            manual=manual,
        ),
        encoding="utf-8",
    )
    return {
        "path": str(report_path),
        "chart": str(chart_path),
        "recommendation": recommendation,
        "manual_review": manual,
        "independent_audits": len(audits),
        "api_audits": sum(
            row.get("audit_provider") != "ollama" for row in audits
        ),
        "local_audits": sum(
            row.get("audit_provider") == "ollama" for row in audits
        ),
    }


def _load_audit_records(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("audits*.jsonl")):
        rows.extend(read_jsonl(path))
    latest = latest_by_run_hash(rows)
    selection_path = run_dir / "audit_selection.jsonl"
    allowed_generation_hashes = (
        {
            row.get("run_hash")
            for row in read_jsonl(selection_path)
            if row.get("run_hash")
        }
        if selection_path.exists()
        else None
    )
    return [
        row
        for row in latest
        if row.get("status") in (None, "ok") and row.get("audit")
        and (
            allowed_generation_hashes is None
            or row.get("generation_run_hash") in allowed_generation_hashes
        )
    ]


def _audit_summary(rows: list[dict[str, Any]]) -> pd.DataFrame:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row.get("dataset", ""),
                row.get("condition", ""),
                row.get("audit_model", ""),
            )
        ].append(row)
    records: list[dict[str, Any]] = []
    score_fields = (
        "issue_coverage",
        "rule_fit",
        "factual_grounding",
        "defense_coverage",
        "burden_correctness",
        "final_decision_consistency",
    )
    for (dataset, condition, model), group in sorted(groups.items()):
        audits = [row["audit"] for row in group]
        record: dict[str, Any] = {
            "dataset": dataset,
            "condition": condition,
            "audit_model": model,
            "n": len(group),
        }
        for field in score_fields:
            record[field] = float(
                np.mean([float(audit[field]) for audit in audits])
            )
        record["none_rate"] = float(
            np.mean([audit["first_error"] == "none" for audit in audits])
        )
        records.append(record)
    return pd.DataFrame(records)


def _condition_summary(
    rows: list[dict[str, Any]], history: list[dict[str, Any]] | None = None
) -> pd.DataFrame:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("dataset", ""), row.get("condition", ""))].append(row)
    first_status: dict[str, str] = {}
    for row in history or []:
        run_hash = row.get("run_hash")
        if run_hash and run_hash not in first_status:
            first_status[run_hash] = row.get("status", "error")

    records: list[dict[str, Any]] = []
    for (dataset, condition), group in sorted(groups.items()):
        ok = [row for row in group if row.get("status") == "ok"]
        binary_rows = [
            row for row in group if row.get("gold_answer") in {"support", "reject"}
        ]
        y_true = [str(row["gold_answer"]) for row in binary_rows]
        y_pred = [
            (
                str(row.get("prediction"))
                if row.get("status") == "ok"
                and row.get("prediction") in {"support", "reject"}
                else "__invalid__"
            )
            for row in binary_rows
        ]
        if y_true:
            recalls = recall_score(
                y_true,
                y_pred,
                labels=["support", "reject"],
                average=None,
                zero_division=0,
            )
            macro_f1 = float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=["support", "reject"],
                    average="macro",
                    zero_division=0,
                )
            )
            support_recall = float(recalls[0])
            reject_recall = float(recalls[1])
        else:
            macro_f1 = None
            support_recall = None
            reject_recall = None
        answers_itt = np.array(
            [
                float(row.get("answer_correct", 0.0))
                if row.get("status") == "ok"
                else 0.0
                for row in group
            ]
        )
        ci_low, ci_high = bootstrap_ci(
            answers_itt, seed=20260619, samples=2000
        )
        first_failures = sum(
            first_status.get(row.get("run_hash"), row.get("status")) != "ok"
            for row in group
        )
        records.append(
            {
                "dataset": dataset,
                "condition": condition,
                "planned_n": len(group),
                "usable_n": len(ok),
                "failure_rate": (len(group) - len(ok)) / len(group),
                "first_pass_failure_rate": (
                    first_failures / len(group) if history is not None else None
                ),
                "answer_accuracy_conditional": _mean(ok, "answer_correct"),
                "answer_accuracy_itt": float(answers_itt.mean()),
                "answer_accuracy_itt_ci_low": ci_low,
                "answer_accuracy_itt_ci_high": ci_high,
                "macro_f1_itt": macro_f1,
                "support_recall_itt": support_recall,
                "reject_recall_itt": reject_recall,
                "binary_prediction_valid_rate": _mean(
                    ok, "binary_prediction_valid"
                ),
                "conclusion_with_fact_rate": _mean(
                    ok, "conclusion_with_fact_rate"
                ),
                "valid_fact_reference_rate": _mean(
                    ok, "valid_fact_reference_rate"
                ),
                "unknown_fact_reference_count": _mean(
                    ok, "unknown_fact_reference_count"
                ),
                "issue_coverage_proxy": _mean(ok, "issue_coverage_proxy"),
                "mean_calls": _mean(ok, "calls"),
                "mean_latency_seconds": _mean(ok, "elapsed_seconds"),
                "mean_prompt_tokens": _mean(ok, "prompt_tokens"),
                "mean_output_tokens": _mean(ok, "output_tokens"),
            }
        )
    return pd.DataFrame(records)


def paired_condition_comparisons(
    rows: list[dict[str, Any]],
    *,
    baseline: str,
    seed: int,
    samples: int = 2000,
) -> pd.DataFrame:
    by_condition: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = (
        defaultdict(dict)
    )
    for row in rows:
        condition = str(row.get("condition", ""))
        key = (
            str(row.get("dataset", "")),
            str(row.get("case_id", "")),
            str(row.get("variant_id", "original")),
        )
        by_condition[condition][key] = row
    baseline_rows = by_condition.get(baseline, {})
    records: list[dict[str, Any]] = []
    for condition in sorted(by_condition):
        if condition == baseline:
            continue
        common = sorted(set(baseline_rows) & set(by_condition[condition]))
        if not common:
            continue
        baseline_correct = np.array(
            [_itt_correct(baseline_rows[key]) for key in common], dtype=float
        )
        condition_correct = np.array(
            [_itt_correct(by_condition[condition][key]) for key in common],
            dtype=float,
        )
        differences = condition_correct - baseline_correct
        rng = np.random.default_rng(seed)
        bootstrap = np.array(
            [
                rng.choice(differences, size=len(differences), replace=True).mean()
                for _ in range(samples)
            ]
        )
        baseline_only = int(
            np.sum((baseline_correct == 1) & (condition_correct == 0))
        )
        condition_only = int(
            np.sum((baseline_correct == 0) & (condition_correct == 1))
        )
        discordant = baseline_only + condition_only
        records.append(
            {
                "baseline": baseline,
                "condition": condition,
                "paired_n": len(common),
                "baseline_accuracy": float(baseline_correct.mean()),
                "condition_accuracy": float(condition_correct.mean()),
                "accuracy_difference": float(differences.mean()),
                "difference_ci_low": float(np.quantile(bootstrap, 0.025)),
                "difference_ci_high": float(np.quantile(bootstrap, 0.975)),
                "baseline_only_correct": baseline_only,
                "condition_only_correct": condition_only,
                "mcnemar_exact_p": (
                    float(
                        binomtest(
                            min(baseline_only, condition_only),
                            n=discordant,
                            p=0.5,
                        ).pvalue
                    )
                    if discordant
                    else 1.0
                ),
            }
        )
    return pd.DataFrame(records)


def _itt_correct(row: dict[str, Any]) -> float:
    if row.get("status") != "ok":
        return 0.0
    return float(row.get("answer_correct", 0.0))


def _stratum_summary(rows: list[dict[str, Any]]) -> pd.DataFrame:
    expanded: list[dict[str, Any]] = []
    for row in rows:
        metadata = row.get("metadata") or {}
        if row.get("dataset") == "openexempt":
            stratum = (
                f"{metadata.get('suite', 'unknown')} | "
                f"{metadata.get('task_family', 'unknown')}"
            )
        else:
            stratum = str(metadata.get("lawsuit_type", "unknown"))
        expanded.append({**row, "_stratum": stratum})
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in expanded:
        groups[
            (row.get("dataset", ""), row.get("condition", ""), row["_stratum"])
        ].append(row)
    records = []
    for (dataset, condition, stratum), group in sorted(groups.items()):
        ok = [row for row in group if row.get("status") == "ok"]
        records.append(
            {
                "dataset": dataset,
                "condition": condition,
                "stratum": stratum,
                "planned_n": len(group),
                "usable_n": len(ok),
                "failure_rate": (len(group) - len(ok)) / len(group),
                "answer_accuracy_itt": sum(
                    float(row.get("answer_correct", 0.0)) for row in ok
                )
                / len(group),
                "valid_fact_reference_rate": _mean(
                    ok, "valid_fact_reference_rate"
                ),
            }
        )
    return pd.DataFrame(records)


def _oracle_gap(scored: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    datasets = sorted(
        {
            str(row.get("dataset", ""))
            for row in scored
            if row.get("condition") == "oracle"
        }
    )
    for dataset in datasets:
        by_condition: dict[
            str, dict[tuple[str, str], dict[str, Any]]
        ] = defaultdict(dict)
        for row in scored:
            if row.get("dataset") != dataset:
                continue
            key = (
                str(row.get("case_id", "")),
                str(row.get("variant_id", "original")),
            )
            by_condition[str(row.get("condition", ""))][key] = row
        oracle_keys = set(by_condition.get("oracle", {}))
        common = sorted(
            oracle_keys
            & set(by_condition.get("structured", {}))
            & set(by_condition.get("typed", {}))
            & set(by_condition.get("validated", {}))
        )
        if not common:
            continue
        accuracies = {
            condition: float(
                np.mean(
                    [
                        _itt_correct(by_condition[condition][key])
                        for key in common
                    ]
                )
            )
            for condition in ("structured", "typed", "validated", "oracle")
        }
        structured = accuracies["structured"]
        record: dict[str, Any] = {
            "dataset": dataset,
            "oracle_subset_n": len(common),
            "structured_accuracy_itt": structured,
        }
        for condition in ("typed", "validated", "oracle"):
            value = accuracies[condition]
            record[f"{condition}_accuracy_itt"] = value
            record[f"{condition}_gap_vs_structured"] = value - structured
        oracle_gain = record.get("oracle_gap_vs_structured", np.nan)
        automatic_gain = max(
            record.get("typed_gap_vs_structured", np.nan),
            record.get("validated_gap_vs_structured", np.nan),
        )
        record["automatic_oracle_gain_recovery"] = (
            automatic_gain / oracle_gain
            if pd.notna(oracle_gain) and oracle_gain > 0
            else np.nan
        )
        records.append(record)
    return pd.DataFrame(records)


def _select_manual_case_keys(
    rows: list[dict[str, Any]], *, count: int, seed: int
) -> list[dict[str, Any]]:
    cases: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("condition") not in PRINCIPAL_CONDITIONS:
            continue
        key = (row["dataset"], row["case_id"], row["variant_id"])
        cases.setdefault(
            key,
            {
                "dataset": row["dataset"],
                "case_id": row["case_id"],
                "variant_id": row["variant_id"],
                "metadata": row.get("metadata") or {},
            },
        )
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    datasets = sorted({value["dataset"] for value in cases.values()})
    base_target = count // max(len(datasets), 1)
    for dataset in datasets:
        candidates = [
            value for value in cases.values() if value["dataset"] == dataset
        ]
        strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for value in candidates:
            metadata = value["metadata"]
            label = str(
                metadata.get("suite")
                or metadata.get("lawsuit_type")
                or "unknown"
            )
            strata[label].append(value)
        for values in strata.values():
            rng.shuffle(values)
        target = min(base_target, len(candidates))
        labels = sorted(strata)
        chosen = 0
        while chosen < target and any(strata.values()):
            for label in labels:
                if strata[label] and chosen < target:
                    selected.append(strata[label].pop())
                    chosen += 1
    if len(selected) < count:
        selected_keys = {
            (row["dataset"], row["case_id"], row["variant_id"])
            for row in selected
        }
        remaining = [
            value
            for key, value in cases.items()
            if key not in selected_keys
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
    return selected[:count]


def _write_manual_review_packet(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    report_dir: Path,
    *,
    count: int,
    seed: int,
) -> dict[str, Any]:
    selected = _select_manual_case_keys(rows, count=count, seed=seed)
    row_index = {
        (row["dataset"], row["case_id"], row["variant_id"], row["condition"]): row
        for row in rows
    }
    case_index = {
        (case.dataset, case.case_id, case.variant_id): case
        for case in load_cases(config)
    }
    rng = random.Random(seed)
    packet: list[dict[str, Any]] = []
    reannotation: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    rubric_rows: list[dict[str, Any]] = []

    for index, selected_case in enumerate(selected, start=1):
        case_key = (
            selected_case["dataset"],
            selected_case["case_id"],
            selected_case["variant_id"],
        )
        case = case_index[case_key]
        outputs = []
        condition_rows = []
        for condition in PRINCIPAL_CONDITIONS:
            row = row_index.get((*case_key, condition))
            if row:
                condition_rows.append((condition, row))
        rng.shuffle(condition_rows)
        for condition, row in condition_rows:
            output_id = sha256_text(
                f"{seed}|{index}|{condition}|{row['run_hash']}"
            )[:12]
            outputs.append(
                {
                    "output_id": output_id,
                    "status": row.get("status"),
                    "analysis": row.get("parsed_json"),
                    "error": row.get("schema_errors") if row.get("status") != "ok" else None,
                }
            )
            key_rows.append(
                {
                    "review_case_id": f"M{index:03d}",
                    "output_id": output_id,
                    "source_output_id": "",
                    "dataset": case.dataset,
                    "case_id": case.case_id,
                    "variant_id": case.variant_id,
                    "condition": condition,
                }
            )
            rubric_rows.append(
                {
                    "review_case_id": f"M{index:03d}",
                    "output_id": output_id,
                    "first_error": "",
                    "downstream_symptoms": "",
                    "issue_coverage_0_4": "",
                    "rule_fit_0_4": "",
                    "factual_grounding_0_4": "",
                    "defense_coverage_0_4": "",
                    "burden_correctness_0_4": "",
                    "final_decision_consistency_0_4": "",
                    "notes": "",
                }
            )
        item = {
            "review_case_id": f"M{index:03d}",
            "dataset": case.dataset,
            "claim": case.claim,
            "requested_remedy": case.requested_remedy,
            "facts": case.facts,
            "authorities": case.authorities,
            "outputs": outputs,
        }
        packet.append(item)
        if index <= 10:
            duplicate, mappings = _blind_reannotation(
                item, index=index, seed=seed
            )
            reannotation.append(duplicate)
            source_conditions = {
                row["output_id"]: row["condition"]
                for row in key_rows
                if row["review_case_id"] == f"M{index:03d}"
            }
            for mapping in mappings:
                repeat_output_id = mapping["output_id"]
                source_output_id = mapping["source_output_id"]
                key_rows.append(
                    {
                        "review_case_id": f"R{index:03d}",
                        "output_id": repeat_output_id,
                        "source_output_id": source_output_id,
                        "dataset": case.dataset,
                        "case_id": case.case_id,
                        "variant_id": case.variant_id,
                        "condition": source_conditions[source_output_id],
                    }
                )
                rubric_rows.append(
                    {
                        "review_case_id": f"R{index:03d}",
                        "output_id": repeat_output_id,
                        "first_error": "",
                        "downstream_symptoms": "",
                        "issue_coverage_0_4": "",
                        "rule_fit_0_4": "",
                        "factual_grounding_0_4": "",
                        "defense_coverage_0_4": "",
                        "burden_correctness_0_4": "",
                        "final_decision_consistency_0_4": "",
                        "notes": "",
                    }
                )

    packet_path = report_dir / "manual_review_packet.jsonl"
    repeat_path = report_dir / "manual_review_reannotation.jsonl"
    key_path = report_dir / "manual_review_key.csv"
    rubric_path = report_dir / "manual_review_rubric.csv"
    write_jsonl(packet_path, packet)
    write_jsonl(repeat_path, reannotation)
    _write_csv(key_path, key_rows)
    _write_csv(rubric_path, rubric_rows)
    return {
        "cases": len(packet),
        "reannotation_cases": len(reannotation),
        "packet": str(packet_path),
        "reannotation_packet": str(repeat_path),
        "key": str(key_path),
        "rubric": str(rubric_path),
    }


def _blind_reannotation(
    item: dict[str, Any], *, index: int, seed: int
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    duplicate = json.loads(json.dumps(item))
    duplicate["review_case_id"] = f"R{index:03d}"
    mappings: list[dict[str, str]] = []
    for output in duplicate.get("outputs", []):
        source_output_id = output["output_id"]
        output_id = sha256_text(
            f"repeat|{seed}|{index}|{source_output_id}"
        )[:12]
        output["output_id"] = output_id
        mappings.append(
            {
                "output_id": output_id,
                "source_output_id": source_output_id,
            }
        )
    random.Random(seed + index).shuffle(duplicate["outputs"])
    return duplicate, mappings


def _write_accuracy_chart(summary: pd.DataFrame, report_dir: Path) -> Path:
    chart_path = report_dir / "answer_accuracy_itt.png"
    pivot = summary.pivot(
        index="condition", columns="dataset", values="answer_accuracy_itt"
    )
    axis = pivot.plot(kind="bar", ylim=(0, 1), figsize=(9, 4.5))
    axis.set_ylabel("Intention-to-treat exact accuracy")
    axis.set_title("Legal case-state pilot")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=160)
    plt.close()
    return chart_path


def _recommend(
    summary: pd.DataFrame,
    oracle_gap: pd.DataFrame,
    *,
    audit_summary: pd.DataFrame | None = None,
) -> str:
    indexed = summary.set_index(["dataset", "condition"])
    try:
        structured = float(indexed.loc[
            ("legalhk", "structured"), "answer_accuracy_itt"
        ])
        validated = float(indexed.loc[
            ("legalhk", "validated"), "answer_accuracy_itt"
        ])
        validated_failure = float(
            indexed.loc[("legalhk", "validated"), "failure_rate"]
        )
    except KeyError:
        return "INCOMPLETE: required comparison cells are missing."

    grounding_column = (
        "conclusion_with_fact_rate"
        if "conclusion_with_fact_rate" in summary.columns
        else "valid_fact_reference_rate"
    )
    try:
        grounding_gain = float(
            indexed.loc[("legalhk", "validated"), grounding_column]
        ) - float(indexed.loc[("legalhk", "structured"), grounding_column])
    except (KeyError, TypeError, ValueError):
        grounding_gain = 0.0
    outcome_gain = validated - structured

    if audit_summary is not None and not audit_summary.empty:
        audit_indexed = audit_summary.set_index(["dataset", "condition"])
        try:
            structured_quality = np.mean(
                [
                    audit_indexed.loc[
                        ("legalhk", "structured"), "factual_grounding"
                    ],
                    audit_indexed.loc[
                        ("legalhk", "structured"), "issue_coverage"
                    ],
                ]
            )
            validated_quality = np.mean(
                [
                    audit_indexed.loc[
                        ("legalhk", "validated"), "factual_grounding"
                    ],
                    audit_indexed.loc[
                        ("legalhk", "validated"), "issue_coverage"
                    ],
                ]
            )
        except KeyError:
            pass
        else:
            if validated_quality < structured_quality:
                return (
                    "DO NOT SCALE CURRENT METHOD: the independent judge finds "
                    "lower substantive grounding or issue coverage after "
                    "validation. Refine state induction and fact-to-element "
                    "verification first."
                )

    if (
        outcome_gain >= 0.05
        and grounding_gain >= 0.05
        and validated_failure <= 0.2
    ):
        return (
            "CONDITIONAL GO: validated case states improve both LegalHK outcomes "
            "and evidence-linked analysis. Confirm on a larger held-out LegalHK "
            "sample before adding another dataset."
        )
    if grounding_gain >= 0.05 or outcome_gain >= 0.05:
        return (
            "REFINE: one central LegalHK metric improves, but the outcome and "
            "grounding gains do not yet move together. Inspect paired failures "
            "before scaling."
        )
    if validated_failure > 0.2:
        return "NO-GO: case-state overhead causes excessive unusable outputs."
    return (
        "NO-GO OR REDESIGN: no central LegalHK metric shows a material state "
        "benefit."
    )


def _render_markdown(
    *,
    summary: pd.DataFrame,
    strata: pd.DataFrame,
    oracle_gap: pd.DataFrame,
    paired: pd.DataFrame,
    audit_summary: pd.DataFrame,
    first_errors: Counter,
    recommendation: str,
    chart_path: Path,
    history: list[dict[str, Any]],
    scored: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    manual: dict[str, Any],
) -> str:
    first_by_hash: dict[str, dict[str, Any]] = {}
    for row in history:
        first_by_hash.setdefault(row.get("run_hash", ""), row)
    first_pass_failures = sum(
        row.get("status") != "ok" for row in first_by_hash.values()
    )
    final_failures = sum(row.get("status") != "ok" for row in scored)
    repair_counts = Counter(
        action
        for row in scored
        for action in (row.get("repair_actions") or [])
    )
    repairs = "\n".join(
        f"- {name}: {count}" for name, count in repair_counts.most_common()
    ) or "- None recorded."
    errors = "\n".join(
        f"- {name}: {count}" for name, count in first_errors.most_common()
    ) or "- Pending: independent audit has not been run."
    audit_models = Counter(row.get("audit_model", "unknown") for row in audits)
    audit_note = (
        ", ".join(f"{model}: {count}" for model, count in audit_models.items())
        if audits
        else "Pending."
    )
    indexed = summary.set_index(["dataset", "condition"])

    def metric(condition: str, name: str) -> float | None:
        try:
            value = indexed.loc[("legalhk", condition), name]
        except KeyError:
            return None
        return None if pd.isna(value) else float(value)

    structured_accuracy = metric("structured", "answer_accuracy_itt")
    validated_accuracy = metric("validated", "answer_accuracy_itt")
    structured_grounding = metric("structured", "conclusion_with_fact_rate")
    validated_grounding = metric("validated", "conclusion_with_fact_rate")
    structured_latency = metric("structured", "mean_latency_seconds")
    validated_latency = metric("validated", "mean_latency_seconds")
    outcome_gain = (
        validated_accuracy - structured_accuracy
        if structured_accuracy is not None and validated_accuracy is not None
        else None
    )
    grounding_gain = (
        validated_grounding - structured_grounding
        if structured_grounding is not None and validated_grounding is not None
        else None
    )
    latency_ratio = (
        validated_latency / structured_latency
        if structured_latency and validated_latency is not None
        else None
    )
    diagnostic_lines = [
        (
            f"- LegalHK failure-adjusted outcome accuracy difference: "
            f"{outcome_gain:+.1%}."
            if outcome_gain is not None
            else "- Outcome difference unavailable."
        ),
        (
            f"- Conclusion-with-cited-fact difference: {grounding_gain:+.1%}."
            if grounding_gain is not None
            else "- Evidence-link difference unavailable."
        ),
        (
            f"- Validated/Structured latency ratio: {latency_ratio:.2f}x."
            if latency_ratio is not None
            else "- Latency ratio unavailable."
        ),
    ]
    audit_interpretation = "- Insufficient LegalHK audit cells for comparison."
    if not audit_summary.empty:
        audit_indexed = audit_summary.set_index(["dataset", "condition"])
        try:
            audit_interpretation = (
                "- LegalHK Validated minus Structured: "
                f"{audit_indexed.loc[('legalhk', 'validated'), 'factual_grounding'] - audit_indexed.loc[('legalhk', 'structured'), 'factual_grounding']:+.2f} factual grounding; "
                f"{audit_indexed.loc[('legalhk', 'validated'), 'issue_coverage'] - audit_indexed.loc[('legalhk', 'structured'), 'issue_coverage']:+.2f} issue coverage."
            )
        except KeyError:
            pass
    successes = [
        row
        for row in scored
        if row.get("status") == "ok" and row.get("answer_correct") == 1.0
    ][:5]
    failures = [row for row in scored if row.get("status") != "ok"][:5]
    examples = "\n".join(
        f"- Success: `{row['case_id']}` — {row['condition']}" for row in successes
    )
    examples += ("\n" if examples else "") + "\n".join(
        f"- Failure: `{row['case_id']}` — {row['condition']} "
        f"({row.get('error_type', 'error')})"
        for row in failures
    )
    return f"""# LegalHK-Only Case-State Diagnostic Pilot

## Decision

**{recommendation}**

## Run integrity

- Planned run hashes: {len(scored)}
- First-pass failures: {first_pass_failures}
- Final retained failures: {final_failures}
- Historical records retained: {len(history)}
- Independent audit: {audit_note}

## Aggregate results

{_markdown_table(summary)}

![Intention-to-treat accuracy]({chart_path.name})

## Paired condition comparisons

{_markdown_table(paired)}

Errors and non-binary outputs count as incorrect. Every comparison uses the same
LegalHK cases under Structured and the named condition.

## Oracle-gap analysis

{_markdown_table(oracle_gap)}

Oracle uses sanitized reference issues and laws as an upper-bound diagnostic.

## Diagnostic criteria

{chr(10).join(diagnostic_lines)}

## Performance by lawsuit type

Full table: `stratum_summary.csv` ({len(strata)} rows).

## Structured-output and repair diagnostics

{repairs}

## Independent model audit

{_markdown_table(audit_summary)}

{audit_interpretation}

### First-error distribution

{errors}

## Manual review

- {manual['cases']} blindly presented case instances across four principal conditions.
- {manual['reannotation_cases']} cases exported for consistency re-annotation.
- Packet: `manual_review_packet.jsonl`
- Hidden condition key: `manual_review_key.csv`
- Rubric: `manual_review_rubric.csv`

## Representative records

{examples or "- No examples available."}

## Interpretation and limitations

- This is a LegalHK-only diagnostic and does not establish cross-dataset or
  cross-jurisdiction generalization.
- Selected facts pass an explicit-outcome screen, but LegalHK fact summaries
  were enhanced from judgments and may retain latent outcome conditioning.
- Valid fact IDs measure engineering hygiene, not substantive legal grounding.
- Sixty-four cases can reveal large effects and pipeline failures but cannot
  precisely establish a small five-point improvement.
"""


def _render_legacy_markdown(
    *,
    summary: pd.DataFrame,
    strata: pd.DataFrame,
    oracle_gap: pd.DataFrame,
    audit_summary: pd.DataFrame,
    first_errors: Counter,
    recommendation: str,
    chart_path: Path,
    history: list[dict[str, Any]],
    scored: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    manual: dict[str, Any],
) -> str:
    first_by_hash: dict[str, dict[str, Any]] = {}
    for row in history:
        first_by_hash.setdefault(row.get("run_hash", ""), row)
    first_pass_failures = sum(
        row.get("status") != "ok" for row in first_by_hash.values()
    )
    final_failures = sum(row.get("status") != "ok" for row in scored)
    repair_counts = Counter(
        action
        for row in scored
        for action in (row.get("repair_actions") or [])
    )
    repairs = "\n".join(
        f"- {name}: {count}" for name, count in repair_counts.most_common()
    ) or "- None recorded."
    errors = "\n".join(
        f"- {name}: {count}" for name, count in first_errors.most_common()
    ) or "- Pending: independent audit has not been run."
    successes = [
        row
        for row in scored
        if row.get("status") == "ok" and row.get("answer_correct") == 1.0
    ][:5]
    failures = [row for row in scored if row.get("status") != "ok"][:5]
    examples = "\n".join(
        f"- Success: `{row['dataset']}/{row['case_id']}` — {row['condition']}"
        for row in successes
    )
    examples += ("\n" if examples else "") + "\n".join(
        f"- Failure: `{row['dataset']}/{row['case_id']}` — {row['condition']} "
        f"({row.get('error_type', 'error')})"
        for row in failures
    )
    paired_available = any(row.get("pair_id") for row in scored)
    paired_note = (
        "Paired robustness metrics are available in the scored records."
        if paired_available
        else (
            "Unavailable: the public OpenExempt robustness suites are controlled "
            "strata, not released same-case counterfactual pairs."
        )
    )
    audit_models = Counter(row.get("audit_model", "unknown") for row in audits)
    audit_note = (
        ", ".join(f"{model}: {count}" for model, count in audit_models.items())
        if audits
        else "Pending."
    )
    audit_interpretation = "- Insufficient audit cells for comparison."
    if not audit_summary.empty:
        audit_indexed = audit_summary.set_index(["dataset", "condition"])
        try:
            audit_interpretation = "\n".join(
                [
                    (
                        "- LegalHK validated minus structured: "
                        f"{audit_indexed.loc[('legalhk', 'validated'), 'factual_grounding'] - audit_indexed.loc[('legalhk', 'structured'), 'factual_grounding']:+.2f} factual grounding; "
                        f"{audit_indexed.loc[('legalhk', 'validated'), 'issue_coverage'] - audit_indexed.loc[('legalhk', 'structured'), 'issue_coverage']:+.2f} issue coverage."
                    ),
                    (
                        "- OpenExempt validated minus structured: "
                        f"{audit_indexed.loc[('openexempt', 'validated'), 'factual_grounding'] - audit_indexed.loc[('openexempt', 'structured'), 'factual_grounding']:+.2f} factual grounding; "
                        f"{audit_indexed.loc[('openexempt', 'validated'), 'issue_coverage'] - audit_indexed.loc[('openexempt', 'structured'), 'issue_coverage']:+.2f} issue coverage."
                    ),
                    (
                        "- LegalHK oracle minus structured: "
                        f"{audit_indexed.loc[('legalhk', 'oracle'), 'factual_grounding'] - audit_indexed.loc[('legalhk', 'structured'), 'factual_grounding']:+.2f} factual grounding; "
                        f"{audit_indexed.loc[('legalhk', 'oracle'), 'issue_coverage'] - audit_indexed.loc[('legalhk', 'structured'), 'issue_coverage']:+.2f} issue coverage."
                    ),
                ]
            )
        except KeyError:
            pass
    indexed = summary.set_index(["dataset", "condition"])
    hk_outcome_gain = (
        indexed.loc[("legalhk", "validated"), "answer_accuracy_itt"]
        - indexed.loc[("legalhk", "structured"), "answer_accuracy_itt"]
    )
    oe_grounding_gain = (
        indexed.loc[
            ("openexempt", "validated"), "valid_fact_reference_rate"
        ]
        - indexed.loc[
            ("openexempt", "structured"), "valid_fact_reference_rate"
        ]
    )
    oe_unknown_reduction = 1 - (
        indexed.loc[
            ("openexempt", "validated"), "unknown_fact_reference_count"
        ]
        / indexed.loc[
            ("openexempt", "structured"), "unknown_fact_reference_count"
        ]
    )
    hk_cost_ratio = (
        indexed.loc[("legalhk", "validated"), "mean_latency_seconds"]
        / indexed.loc[("legalhk", "structured"), "mean_latency_seconds"]
    )
    oe_cost_ratio = (
        indexed.loc[("openexempt", "validated"), "mean_latency_seconds"]
        / indexed.loc[("openexempt", "structured"), "mean_latency_seconds"]
    )
    return f"""# Legal Case-State Diagnostic Pilot

## Decision

**{recommendation}**

## Run integrity

- Planned run hashes: {len(scored)}
- First-pass failures: {first_pass_failures}
- Final retained failures after one controlled recovery pass: {final_failures}
- Historical records retained: {len(history)}
- Independent audit: {audit_note}
- ChatGPT UI frontier audit: exported as a supplementary, non-API-equivalent packet.

## Aggregate results

{_markdown_table(summary)}

![Intention-to-treat accuracy]({chart_path.name})

## Oracle-gap analysis

{_markdown_table(oracle_gap)}

The oracle comparison is an upper-bound diagnostic on a subset, not a paired
causal estimate.

## Go/no-go criteria

- OpenExempt valid fact-reference rate: {oe_grounding_gain:+.1%} for validated vs structured.
- OpenExempt nonexistent fact references: {oe_unknown_reduction:.1%} reduction.
- LegalHK failure-adjusted outcome accuracy: {hk_outcome_gain:+.1%}.
- Validated/structured latency ratio: {hk_cost_ratio:.2f}x on LegalHK and {oe_cost_ratio:.2f}x on OpenExempt.
- Paired counterfactual criterion: unavailable in the public release.

The grounding criterion passes, but the five-point outcome threshold, paired
robustness criterion, and under-two-times cost criterion do not all pass.

## Performance by stratum

Full table: `stratum_summary.csv` ({len(strata)} rows).

## Structured-output and repair diagnostics

{repairs}

Failure rates remain part of intention-to-treat metrics; repaired outputs are
not silently treated as pristine.

## Paired robustness

{paired_note}

## Independent model audit

{_markdown_table(audit_summary)}

{audit_interpretation}

### First-error distribution

{errors}

## Manual review

- {manual['cases']} blindly presented case instances across four principal conditions.
- {manual['reannotation_cases']} cases exported separately for consistency re-annotation.
- Packet: `manual_review_packet.jsonl`
- Hidden condition key: `manual_review_key.csv`
- Rubric: `manual_review_rubric.csv`

## Representative records

{examples or "- No examples available."}

## Interpretation

- OpenExempt exact accuracy is zero across conditions; inspect task-format fit and
  deduction quality before treating grounding improvements as outcome success.
- Validated states sharply reduce nonexistent fact references relative to
  structured prompting.
- LegalHK outcome gains are promising but based on 24 cases and coexist with
  structured-output failures.
- The next method should focus on a fact–element verifier and more reliable
  constrained decoding before fine-tuning or branching search.
"""


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and not pd.isna(row.get(key))
    ]
    return float(np.mean(values)) if values else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No data."
    columns = [str(column) for column in frame.columns]
    rows = [
        ["" if pd.isna(value) else _format_cell(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(value.replace("|", "\\|") for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
