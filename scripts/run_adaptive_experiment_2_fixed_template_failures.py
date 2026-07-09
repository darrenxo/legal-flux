from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from legal_pilot.adaptive_profiles import profile_row
from legal_pilot.io_utils import read_jsonl


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "adaptive_trajectory"
CASE_PATH = ROOT / "data" / "processed" / "legalhk_only" / "cases.jsonl"
SCORED_PATH = ROOT / "runs" / "legalhk_only" / "diagnostic" / "scored.jsonl"
AUDIT_PATH = (
    ROOT / "runs" / "legalhk_only" / "diagnostic" / "audits_local_gpt_oss_20b.jsonl"
)
BOT_SCORED_PATH = ROOT / "runs" / "legal_bot" / "diagnostic" / "scored.jsonl"

FIXED_CONDITIONS = ("structured", "typed", "validated")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cases = {row["case_id"]: row for row in read_jsonl(CASE_PATH)}
    scored = [row for row in read_jsonl(SCORED_PATH) if row.get("case_id") in cases]
    frame = pd.DataFrame(_enrich(row, cases[row["case_id"]]) for row in scored)

    condition_summary = _condition_summary(frame)
    regression_summary = _regression_summary(frame)
    profile_regressions = _profile_regressions(frame)
    audit_summary, audit_errors = _audit_tables()
    examples = _representative_examples(frame, cases)
    bot_summary = _bot_summary()

    condition_summary.to_csv(
        REPORT_DIR / "experiment2_condition_summary.csv", index=False
    )
    regression_summary.to_csv(
        REPORT_DIR / "experiment2_fixed_vs_direct_regressions.csv", index=False
    )
    profile_regressions.to_csv(
        REPORT_DIR / "experiment2_regressions_by_profile_label.csv", index=False
    )
    audit_summary.to_csv(REPORT_DIR / "experiment2_audit_summary.csv", index=False)
    audit_errors.to_csv(
        REPORT_DIR / "experiment2_audit_first_error_counts.csv", index=False
    )
    pd.DataFrame(examples).to_json(
        REPORT_DIR / "experiment2_representative_failures.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    bot_summary.to_csv(REPORT_DIR / "experiment2_bot_summary.csv", index=False)

    report = _markdown(
        condition_summary=condition_summary,
        regression_summary=regression_summary,
        profile_regressions=profile_regressions,
        audit_summary=audit_summary,
        audit_errors=audit_errors,
        examples=examples,
        bot_summary=bot_summary,
    )
    (REPORT_DIR / "experiment2_fixed_template_failure_report.md").write_text(
        report, encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "records": int(len(frame)),
                "conditions": sorted(frame["condition"].dropna().unique()),
                "representative_failures": len(examples),
                "report": str(
                    REPORT_DIR / "experiment2_fixed_template_failure_report.md"
                ),
            },
            indent=2,
        )
    )


def _enrich(row: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    facts_text = " ".join(case.get("facts", {}).values())
    reference_state = case.get("reference_state") or {}
    related_laws = "\n".join(
        issue.get("rule_or_test", "")
        for issue in reference_state.get("issues", [])
        if isinstance(issue, dict)
    )
    profile = profile_row(
        {
            "plaintiff_claim": case.get("claim"),
            "lawsuit_type": case.get("metadata", {}).get("lawsuit_type"),
            "more_facts": facts_text,
            "issues": "\n".join(case.get("reference_issues", [])),
            "related_laws": related_laws,
            "relevant_cases": "",
            "support&reject": case.get("gold_answer"),
        }
    )
    return {
        **row,
        "prediction": row.get("prediction")
        or (row.get("parsed_json") or {}).get("final_decision"),
        "correct": float(row.get("answer_correct") or 0.0),
        "reference_issue_count": len(case.get("reference_issues", [])),
        "profile_families": profile["template_families"],
        "profile_demands": profile["reasoning_demands"],
        "profile_trajectory": profile["trajectory_signature"],
        "profile_trajectory_length": profile["trajectory_length"],
    }


def _condition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("condition", dropna=False)
        .agg(
            rows=("run_hash", "count"),
            ok_rows=("status", lambda values: int((values == "ok").sum())),
            accuracy=("correct", "mean"),
            binary_valid=("binary_prediction_valid", "mean"),
            mean_issue_conclusions=("issue_conclusion_count", "mean"),
            mean_trajectory_length=("profile_trajectory_length", "mean"),
            mean_output_tokens=("output_tokens", "mean"),
            mean_calls=("calls", "mean"),
        )
        .reset_index()
        .sort_values("accuracy", ascending=False)
    )


def _regression_summary(frame: pd.DataFrame) -> pd.DataFrame:
    pivot = frame.pivot_table(
        index="case_id",
        columns="condition",
        values="correct",
        aggfunc="max",
    )
    rows = []
    for condition in FIXED_CONDITIONS:
        subset = pivot[["direct", condition]].dropna()
        direct = subset["direct"].astype(float)
        fixed = subset[condition].astype(float)
        rows.append(
            {
                "condition": condition,
                "paired_cases": int(len(subset)),
                "direct_correct_fixed_wrong": int(((direct == 1) & (fixed == 0)).sum()),
                "direct_wrong_fixed_correct": int(((direct == 0) & (fixed == 1)).sum()),
                "both_correct": int(((direct == 1) & (fixed == 1)).sum()),
                "both_wrong": int(((direct == 0) & (fixed == 0)).sum()),
                "fixed_regression_rate_when_direct_correct": float(
                    ((direct == 1) & (fixed == 0)).sum() / max((direct == 1).sum(), 1)
                ),
                "fixed_recovery_rate_when_direct_wrong": float(
                    ((direct == 0) & (fixed == 1)).sum() / max((direct == 0).sum(), 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def _profile_regressions(frame: pd.DataFrame) -> pd.DataFrame:
    direct = frame[frame["condition"] == "direct"][
        ["case_id", "correct"]
    ].rename(columns={"correct": "direct_correct"})
    rows = []
    for condition in FIXED_CONDITIONS:
        fixed = frame[frame["condition"] == condition].merge(direct, on="case_id")
        fixed["regression"] = (fixed["direct_correct"] == 1) & (fixed["correct"] == 0)
        for field in ("profile_families", "profile_demands"):
            counter_total: Counter[str] = Counter()
            counter_regressions: Counter[str] = Counter()
            for _, row in fixed.iterrows():
                for label in str(row[field]).split("|"):
                    if not label:
                        continue
                    counter_total[label] += 1
                    if row["regression"]:
                        counter_regressions[label] += 1
            for label, total in counter_total.items():
                regressions = counter_regressions[label]
                rows.append(
                    {
                        "condition": condition,
                        "profile_field": field,
                        "label": label,
                        "cases": total,
                        "direct_correct_fixed_wrong": regressions,
                        "regression_rate": regressions / total if total else 0.0,
                    }
                )
    return (
        pd.DataFrame(rows)
        .sort_values(["regression_rate", "cases"], ascending=False)
        .reset_index(drop=True)
    )


def _audit_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = [
        row for row in read_jsonl(AUDIT_PATH) if row.get("status") == "ok" and row.get("audit")
    ]
    rows = [{**row, **row["audit"]} for row in records]
    frame = pd.DataFrame(rows)
    score_columns = [
        "issue_coverage",
        "rule_fit",
        "factual_grounding",
        "defense_coverage",
        "burden_correctness",
        "final_decision_consistency",
    ]
    summary = (
        frame.groupby("condition", dropna=False)[score_columns]
        .mean(numeric_only=True)
        .reset_index()
    )
    errors = (
        frame.groupby(["condition", "first_error"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["condition", "count"], ascending=[True, False])
    )
    return summary, errors


def _representative_examples(
    frame: pd.DataFrame, cases: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    direct = frame[frame["condition"] == "direct"][
        ["case_id", "prediction", "correct"]
    ].rename(columns={"prediction": "direct_prediction", "correct": "direct_correct"})
    examples = []
    for condition in FIXED_CONDITIONS:
        fixed = frame[frame["condition"] == condition].merge(direct, on="case_id")
        regressions = fixed[
            (fixed["direct_correct"] == 1) & (fixed["correct"] == 0)
        ].head(3)
        for _, row in regressions.iterrows():
            case = cases[row["case_id"]]
            examples.append(
                {
                    "case_id": row["case_id"],
                    "condition": condition,
                    "gold": case.get("gold_answer"),
                    "direct_prediction": row["direct_prediction"],
                    "fixed_prediction": row["prediction"],
                    "lawsuit_type": case.get("metadata", {}).get("lawsuit_type", ""),
                    "reference_issues": case.get("reference_issues", []),
                    "profile_families": row["profile_families"],
                    "profile_demands": row["profile_demands"],
                    "profile_trajectory": row["profile_trajectory"],
                    "final_rationale": (row.get("parsed_json") or {}).get(
                        "final_rationale", ""
                    )[:500],
                }
            )
    return examples


def _bot_summary() -> pd.DataFrame:
    if not BOT_SCORED_PATH.exists():
        return pd.DataFrame()
    rows = read_jsonl(BOT_SCORED_PATH)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["correct"] = frame["answer_correct"].fillna(0.0)
    return (
        frame.groupby(["condition", "phase"], dropna=False)
        .agg(
            rows=("run_hash", "count"),
            ok_rows=("status", lambda values: int((values == "ok").sum())),
            accuracy=("correct", "mean"),
            fallback_rate=("fallback_used", "mean"),
            mean_retrieval_similarity=("retrieval_similarity", "mean"),
            mean_calls=("calls", "mean"),
        )
        .reset_index()
        .sort_values(["phase", "accuracy"], ascending=[True, False])
    )


def _markdown(
    *,
    condition_summary: pd.DataFrame,
    regression_summary: pd.DataFrame,
    profile_regressions: pd.DataFrame,
    audit_summary: pd.DataFrame,
    audit_errors: pd.DataFrame,
    examples: list[dict[str, Any]],
    bot_summary: pd.DataFrame,
) -> str:
    top_profile = _markdown_table(profile_regressions.head(15))
    example_lines = "\n".join(
        f"- `{item['case_id']}` / {item['condition']}: gold `{item['gold']}`, "
        f"Direct `{item['direct_prediction']}`, fixed `{item['fixed_prediction']}`; "
        f"trajectory `{item['profile_trajectory']}`."
        for item in examples[:9]
    )
    if not example_lines:
        example_lines = "- No fixed-template regressions found."
    bot_text = (
        _markdown_table(bot_summary)
        if not bot_summary.empty
        else "No Legal-BoT diagnostic scored file found."
    )
    return f"""# Experiment 2: Fixed-Template Failure Pilot

## Scope

This post-hoc pilot uses the existing LegalHK-only diagnostic run. It treats
`structured`, `typed`, and `validated` as fixed higher-structure conditions and
compares them against `direct` on the same 64 evaluation cases. It also reads
the local `gpt-oss:20b` audit labels and the existing single-template BoT run.

## Condition summary

{_markdown_table(condition_summary)}

## Fixed-template regressions against Direct

{_markdown_table(regression_summary)}

`direct_correct_fixed_wrong` is especially important: it marks cases where a
fixed higher-level structure appears to have harmed a case the direct model got
right.

## Audit score summary

{_markdown_table(audit_summary)}

## Audit first-error distribution

{_markdown_table(audit_errors)}

## Profile labels most associated with fixed-template regressions

{top_profile}

## Representative fixed-template regressions

{example_lines}

## Existing single-template BoT diagnostic

{bot_text}

## Interpretation

The existing evidence supports a cautious motivation for adaptive trajectories:
fixed structure often improves surface organization and fact references, but it
can impose the wrong case frame, over-decompose simple cases, or create a
case-state whose issue composition is worse than the direct answer. The
adaptive method should therefore be evaluated as a controller that can choose a
short trajectory, a procedural trajectory, an evidence/burden trajectory, a
domain-specific trajectory, or a multi-issue composition trajectory, instead of
forcing every case through the same structure.
"""


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = [str(row[column]).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
