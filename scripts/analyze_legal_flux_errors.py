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
TEMPLATE_POOL = ROOT / "templates" / "legal_flux_templates_v0.jsonl"
ADAPTIVE_CONDITION = "flux_rf_style"
CONDITION_ORDER = ["direct", "structured", ADAPTIVE_CONDITION]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
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


def latest_generation_rows() -> dict[str, dict[str, Any]]:
    plan = json.loads((RUN_DIR / "run_plan.json").read_text(encoding="utf-8"))
    allowed = {
        job["run_hash"]
        for job in plan.get("jobs", [])
        if isinstance(job, dict) and job.get("run_hash")
    }
    latest: dict[str, dict[str, Any]] = {}
    path = RUN_DIR / "generations.jsonl"
    if not path.exists():
        return latest
    for row in read_jsonl(path):
        if row.get("run_hash") in allowed:
            latest[row["run_hash"]] = row
    return latest


def template_lookup() -> dict[str, dict[str, Any]]:
    if not TEMPLATE_POOL.exists():
        return {}
    return {row["template_id"]: row for row in read_jsonl(TEMPLATE_POOL)}


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
    value = " ".join(str(text or "").replace("|", "/").split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


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
                f"{avg([float(x.get('prompt_tokens') or 0.0) for x in items]):.0f}",
                f"{avg([float(x.get('output_tokens') or 0.0) for x in items]):.0f}",
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
            "avg prompt tok",
            "avg output tok",
        ],
        table,
    )


def pivot_by_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_case[row["case_id"]][row["condition"]] = row
    return by_case


def direct_adaptive_delta(
    by_case: dict[str, dict[str, dict[str, Any]]],
    templates: dict[str, dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    counts = Counter()
    deltas: list[dict[str, Any]] = []
    for case_id, conditions in sorted(by_case.items()):
        direct = conditions.get("direct")
        adaptive = conditions.get(ADAPTIVE_CONDITION)
        structured = conditions.get("structured")
        if not direct or not adaptive:
            continue
        direct_ok = bool(direct.get("answer_correct"))
        adaptive_ok = bool(adaptive.get("answer_correct"))
        retrieved_ids = adaptive.get("retrieved_template_ids") or []
        template_names = [
            templates.get(template_id, {}).get("template_name", template_id)
            for template_id in retrieved_ids
        ]
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
                "structured_prediction": structured.get("prediction") if structured else "",
                "lawsuit_type": (direct.get("metadata") or {}).get("lawsuit_type") or "(blank)",
                "adaptive_trajectory_length": adaptive.get("trajectory_length"),
                "adaptive_review_count": adaptive.get("review_count"),
                "adaptive_calls": adaptive.get("calls"),
                "adaptive_retrieved_templates": ";".join(retrieved_ids),
                "adaptive_retrieval_modes": "not_logged",
                "adaptive_step_names": ";".join(
                    str(step.get("step_name"))
                    for step in (adaptive.get("trajectory_plan") or {}).get("planned_steps", [])
                ),
                "adaptive_template_names": ";".join(template_names),
                "adaptive_final_rationale": short(
                    (adaptive.get("parsed_json") or {}).get("final_rationale"),
                    500,
                ),
                "direct_final_rationale": short(
                    (direct.get("parsed_json") or {}).get("final_rationale"),
                    500,
                ),
            }
        )
    total = sum(counts.values()) or 1
    table = md_table(
        ["bucket", "count", "share"],
        [
            [bucket, counts[bucket], pct(counts[bucket] / total)]
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
        by_gold_condition[(condition, str(row.get("gold_answer")))] [prediction] += 1
    dist_rows = []
    for condition in CONDITION_ORDER:
        total = sum(by_condition[condition].values())
        if not total:
            continue
        support = by_condition[condition].get("support", 0)
        reject = by_condition[condition].get("reject", 0)
        dist_rows.append(
            [
                condition,
                support,
                reject,
                by_condition[condition].get("mixed", 0),
                by_condition[condition].get("unresolved", 0),
                by_condition[condition].get("None", 0),
                pct((total - support - reject) / total),
            ]
        )
    by_gold_rows = []
    for condition in CONDITION_ORDER:
        for gold in ("support", "reject"):
            counter = by_gold_condition[(condition, gold)]
            total = sum(counter.values())
            if not total:
                continue
            correct = counter.get(gold, 0)
            by_gold_rows.append(
                [
                    condition,
                    gold,
                    total,
                    counter.get("support", 0),
                    counter.get("reject", 0),
                    pct(correct / total),
                ]
            )
    return (
        md_table(
            ["condition", "support", "reject", "mixed", "unresolved", "none", "non-binary share"],
            dist_rows,
        )
        + "\n"
        + md_table(
            ["condition", "gold", "n", "pred support", "pred reject", "gold-specific acc"],
            by_gold_rows,
        )
    )


def adaptive_failure_reasons(
    rows: list[dict[str, Any]], by_case: dict[str, dict[str, dict[str, Any]]]
) -> str:
    adaptive = [row for row in rows if row["condition"] == ADAPTIVE_CONDITION]
    wrong = [row for row in adaptive if not row.get("answer_correct")]
    reason_counts = Counter()
    for row in wrong:
        prediction = row.get("prediction")
        case_conditions = by_case[row["case_id"]]
        if prediction not in {"support", "reject"}:
            reason_counts["non_binary_prediction"] += 1
        elif prediction != row.get("gold_answer"):
            reason_counts["opposite_binary_prediction"] += 1
        if prediction == "support" and row.get("gold_answer") == "reject":
            reason_counts["false_support"] += 1
        if prediction == "reject" and row.get("gold_answer") == "support":
            reason_counts["false_reject"] += 1
        if case_conditions.get("direct", {}).get("answer_correct"):
            reason_counts["direct_was_correct"] += 1
        if case_conditions.get("structured", {}).get("answer_correct"):
            reason_counts["structured_was_correct"] += 1
    rows_out = [
        [reason, count, pct(count / len(wrong) if wrong else 0.0)]
        for reason, count in reason_counts.most_common()
    ]
    return md_table(["adaptive wrong subtype", "count", "share of adaptive wrong"], rows_out)


def pairwise_overlap(by_case: dict[str, dict[str, dict[str, Any]]]) -> str:
    rows = []
    for left, right in (
        ("direct", ADAPTIVE_CONDITION),
        ("direct", "structured"),
        ("structured", ADAPTIVE_CONDITION),
    ):
        both = [conditions for conditions in by_case.values() if left in conditions and right in conditions]
        buckets = Counter(
            (bool(conditions[left].get("answer_correct")), bool(conditions[right].get("answer_correct")))
            for conditions in both
        )
        disagree = sum(
            1
            for conditions in both
            if conditions[left].get("prediction") != conditions[right].get("prediction")
        )
        rows.append(
            [
                f"{left} vs {right}",
                len(both),
                buckets[(True, True)],
                buckets[(True, False)],
                buckets[(False, True)],
                buckets[(False, False)],
                disagree,
            ]
        )
    return md_table(
        [
            "pair",
            "n",
            "both correct",
            "left only",
            "right only",
            "both wrong",
            "prediction disagreements",
        ],
        rows,
    )


def lawsuit_type_delta(rows: list[dict[str, Any]]) -> str:
    by_type_condition: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        lawsuit_type = (row.get("metadata") or {}).get("lawsuit_type") or "(blank)"
        by_type_condition[(lawsuit_type, row["condition"])].append(row)
    lawsuit_types = sorted({key[0] for key in by_type_condition})
    table = []
    for lawsuit_type in lawsuit_types:
        direct = by_type_condition.get((lawsuit_type, "direct"), [])
        adaptive = by_type_condition.get((lawsuit_type, ADAPTIVE_CONDITION), [])
        structured = by_type_condition.get((lawsuit_type, "structured"), [])
        if len(direct) < 5 or not adaptive:
            continue
        direct_acc = avg([float(row["answer_correct"]) for row in direct]) or 0.0
        adaptive_acc = avg([float(row["answer_correct"]) for row in adaptive]) or 0.0
        structured_acc = avg([float(row["answer_correct"]) for row in structured])
        table.append(
            [
                lawsuit_type,
                len(direct),
                pct(direct_acc),
                pct(structured_acc) if structured_acc is not None else "",
                pct(adaptive_acc),
                f"{100 * (adaptive_acc - direct_acc):+.1f} pp",
            ]
        )
    table.sort(key=lambda row: float(str(row[-1]).split()[0]), reverse=True)
    return md_table(
        ["lawsuit_type", "n", "direct acc", "structured acc", "rf acc", "rf-direct"],
        table[:25],
    )


def trajectory_correlates(rows: list[dict[str, Any]]) -> str:
    adaptive = [row for row in rows if row["condition"] == ADAPTIVE_CONDITION]
    buckets = {
        "rf_correct": [row for row in adaptive if row.get("answer_correct")],
        "rf_wrong": [row for row in adaptive if not row.get("answer_correct")],
        "rf_false_support": [
            row
            for row in adaptive
            if row.get("gold_answer") == "reject" and row.get("prediction") == "support"
        ],
        "rf_false_reject": [
            row
            for row in adaptive
            if row.get("gold_answer") == "support" and row.get("prediction") == "reject"
        ],
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


def retrieval_diagnostics(rows: list[dict[str, Any]], templates: dict[str, dict[str, Any]]) -> str:
    adaptive = [row for row in rows if row["condition"] == ADAPTIVE_CONDITION]
    length_counts = Counter()
    review_counts = Counter()
    template_outcomes: dict[str, list[int]] = defaultdict(list)
    tag_counts = Counter()
    step_name_counts = Counter()
    repeated_template_cases = 0
    mode_counts = Counter()
    similarities = []
    for row in adaptive:
        ok = int(bool(row.get("answer_correct")))
        length_counts[int(row.get("trajectory_length") or 0)] += 1
        review_counts[int(row.get("review_count") or 0)] += 1
        retrieved_ids = row.get("retrieved_template_ids") or []
        if len(set(retrieved_ids)) < len(retrieved_ids):
            repeated_template_cases += 1
        for template_id in set(retrieved_ids):
            template_name = templates.get(template_id, {}).get("template_name", template_id)
            template_outcomes[f"{template_id} - {template_name}"].append(ok)
        for selected in row.get("selected_templates") or []:
            mode_counts[str(selected.get("retrieval_mode"))] += 1
            if selected.get("similarity") is not None:
                similarities.append(float(selected["similarity"]))
        for step in (row.get("trajectory_plan") or {}).get("planned_steps", []):
            for tag in step.get("template_tags") or []:
                tag_counts[str(tag)] += 1
            if step.get("step_name"):
                step_name_counts[str(step["step_name"])] += 1
    template_rows = []
    for key, outcomes in template_outcomes.items():
        if len(outcomes) >= 8:
            template_rows.append([key, len(outcomes), pct(mean(outcomes))])
    template_rows.sort(key=lambda row: (float(row[2].rstrip("%")), -row[1]))
    return "\n".join(
        [
            "**Retrieval modes**",
            md_table(["mode", "count"], [[mode, count] for mode, count in mode_counts.most_common()]),
            (
                "**Similarity summary:** "
                + (
                    f"count={len(similarities)}, mean={mean(similarities):.3f}, "
                    f"min={min(similarities):.3f}, max={max(similarities):.3f}"
                    if similarities
                    else "No similarity scores logged."
                )
            ),
            f"**Cases with repeated template IDs in one trajectory:** {repeated_template_cases}/{len(adaptive)}",
            "",
            "**Trajectory length distribution**",
            md_table(["length", "count"], [[length, count] for length, count in sorted(length_counts.items())]),
            "**Review count distribution**",
            md_table(["reviews", "count"], [[count_key, count] for count_key, count in sorted(review_counts.items())]),
            "**Lowest RF template-associated accuracies, min 8 uses**",
            md_table(["template", "uses", "case acc when used"], template_rows[:15]),
            "**Highest RF template-associated accuracies, min 8 uses**",
            md_table(["template", "uses", "case acc when used"], list(reversed(template_rows[-15:]))),
            "**Most common planned step names**",
            md_table(["step name", "count"], [[name, count] for name, count in step_name_counts.most_common(15)]),
            "**Most common planned tags**",
            md_table(["tag", "count"], [[tag, count] for tag, count in tag_counts.most_common(20)]),
        ]
    )


def examples_section(deltas: list[dict[str, Any]], bucket: str, limit: int = 10) -> str:
    rows = [row for row in deltas if row["bucket"] == bucket][:limit]
    table = [
        [
            row["case_id"],
            row["gold"],
            row["direct_prediction"],
            row["adaptive_prediction"],
            row["structured_prediction"],
            row["lawsuit_type"],
            row["adaptive_trajectory_length"],
            row["adaptive_review_count"],
            short(row["adaptive_step_names"], 120),
            short(row["adaptive_template_names"], 160),
            short(row["adaptive_final_rationale"], 220),
        ]
        for row in rows
    ]
    return md_table(
        [
            "case",
            "gold",
            "direct",
            "rf",
            "structured",
            "lawsuit_type",
            "traj len",
            "reviews",
            "planned steps",
            "templates",
            "rf rationale",
        ],
        table,
    )


def write_delta_csv(deltas: list[dict[str, Any]]) -> Path:
    path = REPORT_DIR / "trajectory_dev_rf_case_deltas.csv"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(deltas[0]))
        writer.writeheader()
        writer.writerows(deltas)
    return path


def main() -> None:
    rows = latest_scored_rows()
    generation_rows = latest_generation_rows()
    for row in rows:
        generation = generation_rows.get(row.get("run_hash"))
        if generation and generation.get("selected_templates") is not None:
            row["selected_templates"] = generation.get("selected_templates")
    error_rows = [row for row in rows if row.get("status") == "error"]
    scored_rows = [row for row in rows if "answer_correct" in row]
    templates = template_lookup()
    by_case = pivot_by_case(scored_rows)
    delta_table, deltas = direct_adaptive_delta(by_case, templates)
    delta_csv = write_delta_csv(deltas)
    report = [
        "# LegalFlux RF-Style Trajectory-Dev Error Analysis",
        "",
        "This report uses the latest run hashes in `runs/legal_flux/trajectory_dev/run_plan.json`.",
        "Model-facing inputs in this run used plaintiff claim/task, parties, facts, and supplied authority context.",
        f"Scored rows: {len(scored_rows)}. Error rows excluded from metric tables: {len(error_rows)}.",
        "",
        "## Condition Summary",
        condition_summary(scored_rows),
        "## Direct vs RF Outcome Buckets",
        delta_table,
        "## Pairwise Overlap",
        pairwise_overlap(by_case),
        "## Prediction Distribution",
        prediction_distribution(scored_rows),
        "## RF Failure Subtypes",
        adaptive_failure_reasons(scored_rows, by_case),
        "## Trajectory Correlates",
        trajectory_correlates(scored_rows),
        "## Lawsuit Type Deltas",
        lawsuit_type_delta(scored_rows),
        "## Retrieval And Template Diagnostics",
        retrieval_diagnostics(scored_rows, templates),
        "## Examples: Direct Correct, RF Wrong",
        examples_section(deltas, "direct_only"),
        "## Examples: RF Correct, Direct Wrong",
        examples_section(deltas, "adaptive_only"),
        "## Artifacts",
        f"- Case-level delta CSV: `{delta_csv.relative_to(ROOT)}`",
        f"- Scored rows: `{(RUN_DIR / 'scored.jsonl').relative_to(ROOT)}`",
        f"- Aggregate metrics: `{(RUN_DIR / 'aggregate.csv').relative_to(ROOT)}`",
        "",
    ]
    report_path = REPORT_DIR / "trajectory_dev_rf_error_analysis.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "delta_csv": str(delta_csv)}, indent=2))


if __name__ == "__main__":
    main()
