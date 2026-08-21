from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


VALID_LABELS = {"support", "reject"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare direct and SFT LegalFlux performance across deterministic "
            "LegalHK complexity bins."
        )
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--legalhk-parquet", type=Path, required=True)
    parser.add_argument("--direct-scored", type=Path, required=True)
    parser.add_argument("--sft-scored", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--direct-condition", default="direct")
    parser.add_argument("--sft-condition", default="flux_rf_style")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def legalhk_index(case_id: str) -> int:
    match = re.fullmatch(r"legalhk-(\d+)", str(case_id))
    if not match:
        raise ValueError(f"Invalid LegalHK case ID: {case_id!r}")
    return int(match.group(1))


def load_predictions(path: Path, *, condition: str) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    for row in read_jsonl(path):
        if row.get("status") != "ok" or row.get("condition") != condition:
            continue
        prediction = str(row.get("prediction", "")).strip().lower()
        gold = str(row.get("gold_answer", "")).strip().lower()
        if prediction not in VALID_LABELS or gold not in VALID_LABELS:
            continue
        case_id = str(row.get("case_id", ""))
        if case_id in selected:
            raise ValueError(f"Duplicate {condition} prediction for {case_id}")
        selected[case_id] = {"prediction": prediction, "gold": gold}
    if not selected:
        raise ValueError(f"No valid {condition!r} predictions found in {path}")
    return selected


def word_count(value: Any) -> int:
    return len(re.findall(r"\S+", str(value or "").strip()))


def build_paired_frame(args: argparse.Namespace) -> pd.DataFrame:
    cases = {
        row["case_id"]: row
        for row in read_jsonl(args.cases)
        if row.get("dataset") == "legalhk"
        and (row.get("metadata") or {}).get("selection_split") == "trajectory_dev"
    }
    direct = load_predictions(args.direct_scored, condition=args.direct_condition)
    sft = load_predictions(args.sft_scored, condition=args.sft_condition)
    paired_ids = sorted(set(cases) & set(direct) & set(sft), key=legalhk_index)
    if not paired_ids:
        raise ValueError("No paired trajectory-dev cases were found.")

    raw = pd.read_parquet(args.legalhk_parquet).fillna("")
    records: list[dict[str, Any]] = []
    for case_id in paired_ids:
        case = cases[case_id]
        index = legalhk_index(case_id)
        if index not in raw.index:
            raise ValueError(f"Raw LegalHK row {index} is missing for {case_id}")
        row = raw.loc[index]
        direct_row = direct[case_id]
        sft_row = sft[case_id]
        if direct_row["gold"] != sft_row["gold"]:
            raise ValueError(f"Gold-label disagreement for {case_id}")

        issues = [str(item).strip() for item in case.get("reference_issues") or []]
        reasoning = str(row.get("court_reasoning", "")).strip()
        judgment = str(row.get("judgment_decision", "")).strip()
        combined_gold = " ".join(part for part in (reasoning, judgment) if part)
        records.append(
            {
                "case_id": case_id,
                "gold": direct_row["gold"],
                "direct_prediction": direct_row["prediction"],
                "sft_prediction": sft_row["prediction"],
                "fact_count": len(case.get("facts") or {}),
                "issue_count": len(issues) if issues else pd.NA,
                "gold_reasoning_judgment_words": (
                    word_count(combined_gold) if combined_gold else pd.NA
                ),
            }
        )

    frame = pd.DataFrame(records)
    expected = len(cases)
    if len(frame) != expected:
        raise ValueError(
            f"Expected {expected} paired trajectory-dev cases but found {len(frame)}. "
            "Use scored ledgers from complete runs over the same split."
        )
    return frame


def weighted_f1(gold: pd.Series, prediction: pd.Series) -> float:
    total = len(gold)
    weighted_sum = 0.0
    for label in ("support", "reject"):
        true_positive = int(((gold == label) & (prediction == label)).sum())
        false_positive = int(((gold != label) & (prediction == label)).sum())
        false_negative = int(((gold == label) & (prediction != label)).sum())
        support = int((gold == label).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        label_f1 = 2 * true_positive / denominator if denominator else 0.0
        weighted_sum += support * label_f1
    return weighted_sum / total if total else 0.0


def interval_labels(values: pd.Series, *, unit: str) -> pd.Series:
    categories = pd.qcut(values, q=4, duplicates="drop")
    category_order = list(categories.cat.categories)
    labels: dict[Any, str] = {}
    for position, interval in enumerate(category_order, start=1):
        selected = values[categories == interval]
        lower = int(selected.min())
        upper = int(selected.max())
        span = str(lower) if lower == upper else f"{lower}–{upper}"
        labels[interval] = f"Q{position}: {span} {unit}"
    return categories.map(labels).astype(str)


def issue_labels(values: pd.Series) -> pd.Series:
    numeric = values.astype(int)
    return numeric.map(lambda value: str(value) if value <= 3 else "4+")


def quartile_order(labels: pd.Series) -> list[str]:
    return sorted(
        set(labels),
        key=lambda label: int(str(label).split(":", maxsplit=1)[0][1:]),
    )


def summarize_bins(
    frame: pd.DataFrame,
    *,
    feature: str,
    bin_column: str,
    order: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label in order:
        group = frame[frame[bin_column] == label]
        if group.empty:
            continue
        direct_accuracy = float(
            (group["direct_prediction"] == group["gold"]).mean()
        )
        sft_accuracy = float((group["sft_prediction"] == group["gold"]).mean())
        direct_f1 = weighted_f1(group["gold"], group["direct_prediction"])
        sft_f1 = weighted_f1(group["gold"], group["sft_prediction"])
        rows.append(
            {
                "complexity_measure": feature,
                "bin": label,
                "n": len(group),
                "direct_accuracy": direct_accuracy,
                "direct_weighted_f1": direct_f1,
                "sft_legalflux_accuracy": sft_accuracy,
                "sft_legalflux_weighted_f1": sft_f1,
                "sft_minus_direct_accuracy": sft_accuracy - direct_accuracy,
                "sft_minus_direct_weighted_f1": sft_f1 - direct_f1,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> list[str]:
    lines = [
        "| Bin | n | Direct accuracy | Direct weighted F1 | SFT LegalFlux accuracy | SFT LegalFlux weighted F1 | Accuracy gap (SFT − direct) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            "| {bin} | {n:,} | {da:.2%} | {df1:.2%} | {sa:.2%} | {sf1:.2%} | {gap:+.2%} |".format(
                bin=row.bin,
                n=row.n,
                da=row.direct_accuracy,
                df1=row.direct_weighted_f1,
                sa=row.sft_legalflux_accuracy,
                sf1=row.sft_legalflux_weighted_f1,
                gap=row.sft_minus_direct_accuracy,
            )
        )
    return lines


def main() -> None:
    args = parse_args()
    frame = build_paired_frame(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    facts = frame[frame["fact_count"].notna()].copy()
    facts["complexity_bin"] = interval_labels(facts["fact_count"], unit="facts")
    fact_order = quartile_order(facts["complexity_bin"])

    issues = frame[frame["issue_count"].notna()].copy()
    issues["complexity_bin"] = issue_labels(issues["issue_count"])
    issue_order = [label for label in ("1", "2", "3", "4+") if label in set(issues["complexity_bin"])]

    gold_length = frame[frame["gold_reasoning_judgment_words"].notna()].copy()
    gold_length["complexity_bin"] = interval_labels(
        gold_length["gold_reasoning_judgment_words"], unit="words"
    )
    length_order = quartile_order(gold_length["complexity_bin"])

    tables = {
        "facts": summarize_bins(
            facts,
            feature="number_of_facts",
            bin_column="complexity_bin",
            order=fact_order,
        ),
        "issues": summarize_bins(
            issues,
            feature="number_of_issues",
            bin_column="complexity_bin",
            order=issue_order,
        ),
        "gold_length": summarize_bins(
            gold_length,
            feature="gold_reasoning_plus_judgment_words",
            bin_column="complexity_bin",
            order=length_order,
        ),
    }

    for name, table in tables.items():
        table.to_csv(args.output_dir / f"complexity_by_{name}.csv", index=False)
    frame.to_csv(args.output_dir / "paired_case_complexity.csv", index=False)

    markdown = [
        "# LegalHK trajectory-dev complexity analysis",
        "",
        f"Paired cases: {len(frame):,}. Fact-count and gold-text-length bins are approximately equal-frequency quartiles. Issue-count bins are 1, 2, 3, and 4+. Cases missing a measure are excluded only from that measure.",
    ]
    titles = {
        "facts": "Number of facts",
        "issues": "Number of issues",
        "gold_length": "Gold court-reasoning + judgment length",
    }
    for name in ("facts", "issues", "gold_length"):
        markdown.extend(["", f"## {titles[name]}", "", *markdown_table(tables[name])])
    (args.output_dir / "complexity_analysis.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    summary = {
        "paired_cases": len(frame),
        "valid_fact_cases": len(facts),
        "valid_issue_cases": len(issues),
        "valid_gold_length_cases": len(gold_length),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "complexity_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print((args.output_dir / "complexity_analysis.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
