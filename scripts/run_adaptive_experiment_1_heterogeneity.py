from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from legal_pilot.adaptive_profiles import (
    label_counts,
    normalized_entropy,
    profile_frame,
)
from legal_pilot.legalhk_selection import (
    explicit_leakage_reasons,
    is_civil_legalhk_row,
    strict_evaluation_reasons,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "legalhk" / "train.parquet"
REPORT_DIR = ROOT / "reports" / "adaptive_trajectory"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(RAW_PATH)
    profiles = profile_frame(frame)
    profiles["is_civil"] = [
        is_civil_legalhk_row(
            plaintiff=str(row.plaintiff),
            lawsuit_type=str(row.lawsuit_type),
            claim=str(row.plaintiff_claim),
        )
        for row in frame.itertuples(index=False)
    ]
    leakage_reasons = [
        explicit_leakage_reasons(
            str(row.more_facts),
            judgment_decision=str(row.judgment_decision),
            ngram_size=6,
            overlap_threshold=0.12,
        )
        for row in frame.itertuples(index=False)
    ]
    strict_reasons = [
        strict_evaluation_reasons(str(row.more_facts))
        for row in frame.itertuples(index=False)
    ]
    profiles["explicit_leakage_reason_count"] = [len(item) for item in leakage_reasons]
    profiles["strict_reason_count"] = [len(item) for item in strict_reasons]
    profiles["support_reject_binary"] = profiles["gold_answer"].isin(["support", "reject"])
    profiles["low_explicit_leakage_civil_candidate"] = (
        profiles["is_civil"]
        & profiles["support_reject_binary"]
        & profiles["explicit_leakage_reason_count"].eq(0)
        & profiles["strict_reason_count"].eq(0)
        & profiles["fact_characters"].between(1, 48000)
    )

    profile_path = REPORT_DIR / "experiment1_case_profiles.csv"
    profiles.to_csv(profile_path, index=False)

    outputs = {
        "family_counts_full": label_counts(profiles["template_families"]),
        "demand_counts_full": label_counts(profiles["reasoning_demands"]),
        "trajectory_counts_full": _count_series(profiles["trajectory_signature"]),
        "family_counts_candidate": label_counts(
            profiles.loc[
                profiles["low_explicit_leakage_civil_candidate"],
                "template_families",
            ]
        ),
        "demand_counts_candidate": label_counts(
            profiles.loc[
                profiles["low_explicit_leakage_civil_candidate"],
                "reasoning_demands",
            ]
        ),
        "trajectory_counts_candidate": _count_series(
            profiles.loc[
                profiles["low_explicit_leakage_civil_candidate"],
                "trajectory_signature",
            ]
        ),
        "by_lawsuit_type": _by_lawsuit_type(profiles),
    }
    for name, table in outputs.items():
        table.to_csv(REPORT_DIR / f"experiment1_{name}.csv", index=False)

    summary = _summary(frame, profiles)
    (REPORT_DIR / "experiment1_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (REPORT_DIR / "experiment1_heterogeneity_summary.md").write_text(
        _markdown(summary, outputs), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def _count_series(series: pd.Series) -> pd.DataFrame:
    return (
        series.value_counts()
        .rename_axis("trajectory_signature")
        .reset_index(name="count")
    )


def _by_lawsuit_type(profiles: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        profiles.groupby("lawsuit_type", dropna=False)
        .agg(
            cases=("row_index", "count"),
            unique_trajectories=("trajectory_signature", "nunique"),
            mean_trajectory_length=("trajectory_length", "mean"),
            mean_issue_count=("issue_count", "mean"),
            mean_demand_count=("demand_count", "mean"),
        )
        .sort_values(["cases", "unique_trajectories"], ascending=False)
        .reset_index()
    )
    return grouped.head(100)


def _summary(frame: pd.DataFrame, profiles: pd.DataFrame) -> dict:
    candidate = profiles[profiles["low_explicit_leakage_civil_candidate"]]
    return {
        "raw_rows": int(len(frame)),
        "civil_rows": int(profiles["is_civil"].sum()),
        "support_reject_rows": int(profiles["support_reject_binary"].sum()),
        "low_explicit_leakage_civil_candidates": int(len(candidate)),
        "unique_lawsuit_types": int(profiles["lawsuit_type"].nunique(dropna=False)),
        "unique_template_family_sets_full": int(profiles["template_families"].nunique()),
        "unique_trajectory_signatures_full": int(
            profiles["trajectory_signature"].nunique()
        ),
        "unique_trajectory_signatures_candidate": int(
            candidate["trajectory_signature"].nunique()
        ),
        "trajectory_entropy_full": normalized_entropy(
            profiles["trajectory_signature"]
        ),
        "trajectory_entropy_candidate": normalized_entropy(
            candidate["trajectory_signature"]
        ),
        "mean_trajectory_length_full": float(profiles["trajectory_length"].mean()),
        "mean_trajectory_length_candidate": float(
            candidate["trajectory_length"].mean()
        ),
        "mean_demand_count_full": float(profiles["demand_count"].mean()),
        "mean_demand_count_candidate": float(candidate["demand_count"].mean()),
        "median_issue_count_full": float(profiles["issue_count"].median()),
        "median_issue_count_candidate": float(candidate["issue_count"].median()),
        "rows_with_3plus_demands_full": int((profiles["demand_count"] >= 3).sum()),
        "rows_with_3plus_demands_candidate": int(
            (candidate["demand_count"] >= 3).sum()
        ),
    }


def _markdown(summary: dict, outputs: dict[str, pd.DataFrame]) -> str:
    top_families = _markdown_table(outputs["family_counts_full"].head(12))
    top_demands = _markdown_table(outputs["demand_counts_full"].head(12))
    top_trajectories = _markdown_table(outputs["trajectory_counts_full"].head(10))
    candidate_trajectories = _markdown_table(
        outputs["trajectory_counts_candidate"].head(10)
    )
    return f"""# Experiment 1: LegalHK Heterogeneity Profile

## Scope

- Raw LegalHK rows profiled: {summary['raw_rows']}
- Civil rows by deterministic screen: {summary['civil_rows']}
- Binary support/reject rows: {summary['support_reject_rows']}
- Low-explicit-leakage civil candidates: {summary['low_explicit_leakage_civil_candidates']}

## Main heterogeneity indicators

- Unique lawsuit types: {summary['unique_lawsuit_types']}
- Unique trajectory signatures, full cache: {summary['unique_trajectory_signatures_full']}
- Unique trajectory signatures, low-leakage civil candidate subset: {summary['unique_trajectory_signatures_candidate']}
- Normalized trajectory entropy, full cache: {summary['trajectory_entropy_full']:.3f}
- Normalized trajectory entropy, candidate subset: {summary['trajectory_entropy_candidate']:.3f}
- Mean trajectory length, full cache: {summary['mean_trajectory_length_full']:.2f}
- Mean trajectory length, candidate subset: {summary['mean_trajectory_length_candidate']:.2f}
- Rows with at least three reasoning-demand labels, full cache: {summary['rows_with_3plus_demands_full']}
- Rows with at least three reasoning-demand labels, candidate subset: {summary['rows_with_3plus_demands_candidate']}

## Top template-family labels, full cache

{top_families}

## Top reasoning-demand labels, full cache

{top_demands}

## Top trajectory signatures, full cache

{top_trajectories}

## Top trajectory signatures, low-leakage civil candidates

{candidate_trajectories}

## Interpretation

The deterministic profile is not a gold legal taxonomy. It is a scalable
screening layer for whether LegalHK visibly contains many case types and
reasoning-demand combinations before we spend model calls on trajectory
planning. The large number of trajectory signatures and high entropy support
running adaptive template selection as a hypothesis worth testing.
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
