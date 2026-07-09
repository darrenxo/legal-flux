from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "legal_flux" / "trajectory_dev"
REPORT_DIR = ROOT / "reports" / "legal_flux"
CONDITION_ORDER = [
    "direct",
    "structured",
    "flux_fixed",
    "flux_adaptive",
    "flux_adaptive_no_review",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def latest_scored_rows() -> list[dict[str, Any]]:
    plan = json.loads((RUN_DIR / "run_plan.json").read_text(encoding="utf-8"))
    allowed = {
        job["run_hash"]
        for job in plan.get("jobs", [])
        if isinstance(job, dict) and job.get("run_hash")
    }
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(RUN_DIR / "scored.jsonl"):
        if row.get("run_hash") in allowed:
            latest[row["run_hash"]] = row
    return list(latest.values())


def pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{100 * value:.1f}%"


def avg(values: list[float]) -> float | None:
    return mean(values) if values else None


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No rows._\n"
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out) + "\n"


def short(text: Any, limit: int = 220) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def profile(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    return metadata.get("legal_flux_profile") or {}


def condition_summary(rows: list[dict[str, Any]]) -> str:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    table = []
    for condition in CONDITION_ORDER:
        items = by_condition[condition]
        if not items:
            continue
        table.append(
            [
                condition,
                len(items),
                pct(avg([float(x["answer_correct"]) for x in items])),
                pct(avg([float(x["binary_prediction_valid"]) for x in items])),
                pct(avg([float(x["issue_coverage_proxy"] or 0.0) for x in items])),
                f"{avg([float(x.get('calls') or 0.0) for x in items]):.2f}",
                f"{avg([float(x.get('elapsed_seconds') or 0.0) for x in items]):.2f}",
            ]
        )
    return md_table(
        [
            "condition",
            "n",
            "answer acc",
            "binary valid",
            "issue coverage",
            "avg calls",
            "avg sec",
        ],
        table,
    )


def pivot_by_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_case[row["case_id"]][row["condition"]] = row
    return by_case


def direct_adaptive_delta(by_case: dict[str, dict[str, dict[str, Any]]]) -> tuple[str, list[dict[str, Any]]]:
    counts = Counter()
    deltas: list[dict[str, Any]] = []
    for case_id, conditions in sorted(by_case.items()):
        direct = conditions.get("direct")
        adaptive = conditions.get("flux_adaptive")
        fixed = conditions.get("flux_fixed")
        structured = conditions.get("structured")
        no_review = conditions.get("flux_adaptive_no_review")
        if not direct or not adaptive:
            continue
        direct_ok = bool(direct.get("answer_correct"))
        adaptive_ok = bool(adaptive.get("answer_correct"))
        if direct_ok and adaptive_ok:
            bucket = "both_correct"
        elif direct_ok and not adaptive_ok:
            bucket = "direct_only"
        elif adaptive_ok and not direct_ok:
            bucket = "adaptive_only"
        else:
            bucket = "both_wrong"
        counts[bucket] += 1
        deltas.append(
            {
                "case_id": case_id,
                "bucket": bucket,
                "gold": direct.get("gold_answer"),
                "direct_prediction": direct.get("prediction"),
                "adaptive_prediction": adaptive.get("prediction"),
                "fixed_prediction": fixed.get("prediction") if fixed else "",
                "no_review_prediction": no_review.get("prediction") if no_review else "",
                "structured_prediction": structured.get("prediction") if structured else "",
                "lawsuit_type": (direct.get("metadata") or {}).get("lawsuit_type") or "(blank)",
                "template_families": profile(direct).get("template_families") or "",
                "reasoning_demands": profile(direct).get("reasoning_demands") or "",
                "adaptive_trajectory_length": adaptive.get("trajectory_length"),
                "adaptive_review_count": adaptive.get("review_count"),
                "adaptive_calls": adaptive.get("calls"),
                "adaptive_repairs": ";".join(adaptive.get("repair_actions") or []),
                "adaptive_final_rationale": short((adaptive.get("parsed_json") or {}).get("final_rationale"), 400),
                "direct_final_rationale": short((direct.get("parsed_json") or {}).get("final_rationale"), 400),
            }
        )
    table = md_table(
        ["bucket", "count", "share"],
        [
            [bucket, counts[bucket], pct(counts[bucket] / sum(counts.values()))]
            for bucket in ("both_correct", "direct_only", "adaptive_only", "both_wrong")
        ],
    )
    return table, deltas


def prediction_distribution(rows: list[dict[str, Any]]) -> str:
    by_condition: dict[str, Counter[str]] = defaultdict(Counter)
    by_gold_condition: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        condition = row["condition"]
        prediction = str(row.get("prediction"))
        by_condition[condition][prediction] += 1
        by_gold_condition[(condition, str(row.get("gold_answer")))][prediction] += 1
    dist_rows = []
    for condition in CONDITION_ORDER:
        total = sum(by_condition[condition].values())
        if not total:
            continue
        dist_rows.append(
            [
                condition,
                by_condition[condition].get("support", 0),
                by_condition[condition].get("reject", 0),
                by_condition[condition].get("mixed", 0),
                by_condition[condition].get("unresolved", 0),
                by_condition[condition].get("None", 0),
                pct((total - by_condition[condition].get("support", 0) - by_condition[condition].get("reject", 0)) / total),
            ]
        )
    by_gold_rows = []
    for condition in CONDITION_ORDER:
        for gold in ("support", "reject"):
            counter = by_gold_condition[(condition, gold)]
            total = sum(counter.values())
            if not total:
                continue
            if not total:
                continue
            by_gold_rows.append(
                [
                    condition,
                    gold,
                    total,
                    counter.get("support", 0),
                    counter.get("reject", 0),
                    counter.get("mixed", 0),
                    counter.get("unresolved", 0),
                    pct((counter.get("mixed", 0) + counter.get("unresolved", 0)) / total),
                ]
            )
    return (
        md_table(
            ["condition", "support", "reject", "mixed", "unresolved", "none", "non-binary share"],
            dist_rows,
        )
        + "\n"
        + md_table(
            ["condition", "gold", "n", "pred support", "pred reject", "pred mixed", "pred unresolved", "mixed/unresolved share"],
            by_gold_rows,
        )
    )


def adaptive_failure_reasons(rows: list[dict[str, Any]], by_case: dict[str, dict[str, dict[str, Any]]]) -> str:
    adaptive = [row for row in rows if row["condition"] == "flux_adaptive"]
    wrong = [row for row in adaptive if not row.get("answer_correct")]
    reason_counts = Counter()
    for row in wrong:
        prediction = row.get("prediction")
        case_conditions = by_case[row["case_id"]]
        if prediction not in {"support", "reject"}:
            reason_counts["non_binary_prediction"] += 1
        elif prediction != row.get("gold_answer"):
            reason_counts["opposite_binary_prediction"] += 1
        if case_conditions.get("direct", {}).get("answer_correct"):
            reason_counts["direct_was_correct"] += 1
        if case_conditions.get("flux_fixed", {}).get("answer_correct"):
            reason_counts["fixed_was_correct"] += 1
        if case_conditions.get("structured", {}).get("answer_correct"):
            reason_counts["structured_was_correct"] += 1
    rows_out = [
        [reason, count, pct(count / len(wrong) if wrong else 0.0)]
        for reason, count in reason_counts.most_common()
    ]
    return md_table(["adaptive wrong subtype", "count", "share of adaptive wrong"], rows_out)


def lawsuit_type_delta(rows: list[dict[str, Any]]) -> str:
    by_type_condition: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        lawsuit_type = (row.get("metadata") or {}).get("lawsuit_type") or "(blank)"
        by_type_condition[(lawsuit_type, row["condition"])].append(row)
    lawsuit_types = sorted({key[0] for key in by_type_condition})
    table = []
    for lawsuit_type in lawsuit_types:
        direct = by_type_condition.get((lawsuit_type, "direct"), [])
        adaptive = by_type_condition.get((lawsuit_type, "flux_adaptive"), [])
        no_review = by_type_condition.get((lawsuit_type, "flux_adaptive_no_review"), [])
        fixed = by_type_condition.get((lawsuit_type, "flux_fixed"), [])
        if len(direct) < 5:
            continue
        direct_acc = avg([float(row["answer_correct"]) for row in direct]) or 0.0
        adaptive_acc = avg([float(row["answer_correct"]) for row in adaptive]) or 0.0
        no_review_acc = avg([float(row["answer_correct"]) for row in no_review])
        fixed_acc = avg([float(row["answer_correct"]) for row in fixed])
        table.append(
            [
                lawsuit_type,
                len(direct),
                pct(direct_acc),
                pct(fixed_acc) if fixed_acc is not None else "",
                pct(adaptive_acc),
                pct(no_review_acc) if no_review_acc is not None else "",
                f"{100 * (adaptive_acc - direct_acc):+.1f} pp",
            ]
        )
    table.sort(key=lambda row: float(str(row[-1]).split()[0]), reverse=True)
    return md_table(
        [
            "lawsuit_type",
            "n",
            "direct acc",
            "fixed acc",
            "adaptive acc",
            "no-review acc",
            "adaptive-direct",
        ],
        table[:20],
    )


def trajectory_correlates(rows: list[dict[str, Any]]) -> str:
    adaptive = [row for row in rows if row["condition"] == "flux_adaptive"]
    no_review = [row for row in rows if row["condition"] == "flux_adaptive_no_review"]
    buckets = {
        "adaptive_correct": [row for row in adaptive if row.get("answer_correct")],
        "adaptive_wrong": [row for row in adaptive if not row.get("answer_correct")],
        "no_review_correct": [row for row in no_review if row.get("answer_correct")],
        "no_review_wrong": [row for row in no_review if not row.get("answer_correct")],
    }
    table = []
    for name, items in buckets.items():
        if not items:
            table.append([name, 0, "", "", "", "", ""])
            continue
        table.append(
            [
                name,
                len(items),
                f"{avg([float(row.get('trajectory_length') or 0) for row in items]):.2f}",
                f"{avg([float(row.get('review_count') or 0) for row in items]):.2f}",
                f"{avg([float(row.get('calls') or 0) for row in items]):.2f}",
                f"{avg([len(row.get('repair_actions') or []) for row in items]):.2f}",
                pct(avg([float(row.get('binary_prediction_valid') or 0) for row in items])),
            ]
        )
    return md_table(
        ["bucket", "n", "avg trajectory len", "avg reviews", "avg calls", "avg repair count", "binary valid"],
        table,
    )


def template_outcomes(rows: list[dict[str, Any]]) -> str:
    stats: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        if row["condition"] != "flux_adaptive":
            continue
        seen = {
            step.get("template_id")
            for step in (row.get("executed_steps") or [])
            if step.get("template_id")
        }
        for template_id in seen:
            stats[template_id].append(int(bool(row.get("answer_correct"))))
    table = []
    for template_id, outcomes in stats.items():
        if len(outcomes) < 8:
            continue
        table.append([template_id, len(outcomes), pct(mean(outcomes))])
    table.sort(key=lambda row: (float(row[2].rstrip("%")), row[1]))
    return (
        "**Lowest adaptive template-associated accuracies, min 8 uses**\n"
        + md_table(["template", "uses", "case acc when used"], table[:12])
        + "\n**Highest adaptive template-associated accuracies, min 8 uses**\n"
        + md_table(["template", "uses", "case acc when used"], list(reversed(table[-12:])))
    )


def examples_section(deltas: list[dict[str, Any]], bucket: str, limit: int = 8) -> str:
    rows = [row for row in deltas if row["bucket"] == bucket][:limit]
    table = [
        [
            row["case_id"],
            row["gold"],
            row["direct_prediction"],
            row["adaptive_prediction"],
            row.get("no_review_prediction", ""),
            row["fixed_prediction"],
            row["lawsuit_type"],
            short(row["reasoning_demands"], 120),
            row["adaptive_trajectory_length"],
            row["adaptive_review_count"],
            short(row["adaptive_final_rationale"], 220),
        ]
        for row in rows
    ]
    return md_table(
        [
            "case",
            "gold",
            "direct",
            "adaptive",
            "no review",
            "fixed",
            "lawsuit_type",
            "reasoning_demands",
            "traj len",
            "reviews",
            "adaptive rationale",
        ],
        table,
    )


def write_delta_csv(deltas: list[dict[str, Any]]) -> Path:
    path = REPORT_DIR / "trajectory_dev_case_deltas.csv"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(deltas[0]))
        writer.writeheader()
        writer.writerows(deltas)
    return path


def main() -> None:
    rows = latest_scored_rows()
    by_case = pivot_by_case(rows)
    delta_table, deltas = direct_adaptive_delta(by_case)
    delta_csv = write_delta_csv(deltas)
    report = [
        "# LegalFlux Trajectory-Dev Error Analysis",
        "",
        "## Condition Summary",
        condition_summary(rows),
        "## Direct vs Adaptive Outcome Buckets",
        delta_table,
        "## Prediction Distribution",
        prediction_distribution(rows),
        "## Adaptive Failure Subtypes",
        adaptive_failure_reasons(rows, by_case),
        "## Trajectory Correlates",
        trajectory_correlates(rows),
        "## Lawsuit Type Deltas",
        lawsuit_type_delta(rows),
        "## Adaptive Template Associations",
        template_outcomes(rows),
        "## Examples: Direct Correct, Adaptive Wrong",
        examples_section(deltas, "direct_only"),
        "## Examples: Adaptive Correct, Direct Wrong",
        examples_section(deltas, "adaptive_only"),
        "## Artifacts",
        f"- Case-level delta CSV: `{delta_csv.relative_to(ROOT)}`",
        f"- Scored rows: `{(RUN_DIR / 'scored.jsonl').relative_to(ROOT)}`",
        f"- Aggregate metrics: `{(RUN_DIR / 'aggregate.csv').relative_to(ROOT)}`",
        "",
    ]
    report_path = REPORT_DIR / "trajectory_dev_error_analysis.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "delta_csv": str(delta_csv)}, indent=2))


if __name__ == "__main__":
    main()
