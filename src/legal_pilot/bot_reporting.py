from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

from .config import resolve_path
from .evaluation import bootstrap_ci
from .io_utils import read_jsonl


CONDITION_DESCRIPTIONS = {
    "direct": "one concise decision prompt with no explicit template",
    "bot_full": "TF-IDF distiller, retrieval, merge manager",
    "bot_no_distiller": "raw case text drives TF-IDF retrieval",
    "bot_no_buffer": "generic fallback for every case",
    "bot_no_manager": "fixed TF-IDF hand-seeded buffer",
    "bot_generic_init": "TF-IDF dynamic growth from an empty legal buffer",
    "semantic_qwen_fixed": "Qwen profiles with fixed BGE-M3 legal seeds",
    "semantic_qwen_dynamic": "Qwen profiles with append-only BGE-M3 updates",
    "semantic_raw_fixed": "raw case text with fixed BGE-M3 legal seeds",
    "semantic_frontier_generic": "frontier profiles with generic reasoning only",
    "semantic_frontier_fixed": "frontier profiles with fixed BGE-M3 legal seeds",
    "semantic_frontier_dynamic": (
        "frontier profiles with append-only BGE-M3 updates"
    ),
}


def bot_condition_summary(rows: list[dict[str, Any]]) -> pd.DataFrame:
    expanded = list(rows)
    expanded.extend({**row, "phase": "all"} for row in rows)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in expanded:
        groups[(str(row.get("condition", "")), str(row.get("phase", "")))].append(
            row
        )
    records: list[dict[str, Any]] = []
    for (condition, phase), group in sorted(groups.items()):
        answers = np.array(
            [
                float(row.get("answer_correct", 0.0))
                if row.get("status") == "ok"
                else 0.0
                for row in group
            ]
        )
        ci_low, ci_high = bootstrap_ci(
            answers, seed=20260619, samples=2000
        )
        y_true = [
            str(row.get("gold_answer"))
            for row in group
            if row.get("gold_answer") in {"support", "reject"}
        ]
        y_pred = [
            (
                str(row.get("prediction"))
                if row.get("status") == "ok"
                and row.get("prediction") in {"support", "reject"}
                else "__invalid__"
            )
            for row in group
            if row.get("gold_answer") in {"support", "reject"}
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
        else:
            recalls = [np.nan, np.nan]
            macro_f1 = np.nan
        ok = [row for row in group if row.get("status") == "ok"]
        records.append(
            {
                "condition": condition,
                "phase": phase,
                "planned_n": len(group),
                "usable_n": len(ok),
                "failure_rate": (len(group) - len(ok)) / len(group),
                "answer_accuracy_itt": float(answers.mean()),
                "answer_accuracy_ci_low": ci_low,
                "answer_accuracy_ci_high": ci_high,
                "macro_f1_itt": macro_f1,
                "support_recall_itt": float(recalls[0]),
                "reject_recall_itt": float(recalls[1]),
                "valid_fact_reference_rate": _mean(
                    ok, "valid_fact_reference_rate"
                ),
                "conclusion_with_fact_rate": _mean(
                    ok, "conclusion_with_fact_rate"
                ),
                "fallback_rate": _mean(ok, "fallback_used"),
                "mean_retrieval_similarity": _mean(
                    ok, "retrieval_similarity"
                ),
                "mean_calls": _mean(ok, "calls"),
                "mean_latency_seconds": _mean(ok, "elapsed_seconds"),
                "mean_prompt_tokens": _mean(ok, "prompt_tokens"),
                "mean_output_tokens": _mean(ok, "output_tokens"),
            }
        )
    return pd.DataFrame(records)


def adaptation_curve(
    rows: list[dict[str, Any]], *, bins: int = 4
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    conditions = sorted(
        {
            str(row.get("condition"))
            for row in rows
            if row.get("phase") == "adaptation"
        }
    )
    for condition in conditions:
        group = sorted(
            [
                row
                for row in rows
                if row.get("phase") == "adaptation"
                and row.get("condition") == condition
            ],
            key=lambda row: int(row.get("stream_index", -1)),
        )
        for round_index, indices in enumerate(
            np.array_split(np.arange(len(group)), bins), start=1
        ):
            if len(indices) == 0:
                continue
            chunk = [group[int(index)] for index in indices]
            records.append(
                {
                    "condition": condition,
                    "round": round_index,
                    "start_stream_index": int(
                        chunk[0].get("stream_index", 0)
                    ),
                    "end_stream_index": int(
                        chunk[-1].get("stream_index", 0)
                    ),
                    "n": len(chunk),
                    "answer_accuracy_itt": float(
                        np.mean(
                            [
                                float(row.get("answer_correct", 0.0))
                                if row.get("status") == "ok"
                                else 0.0
                                for row in chunk
                            ]
                        )
                    ),
                    "fallback_rate": _mean(chunk, "fallback_used"),
                    "mean_buffer_size_after": _mean(
                        chunk, "buffer_size_after"
                    ),
                    "new_templates": sum(
                        row.get("buffer_update_action") == "new"
                        for row in chunk
                    ),
                    "merged_templates": sum(
                        row.get("buffer_update_action") == "merge"
                        for row in chunk
                    ),
                }
            )
    return pd.DataFrame(records)


def buffer_summary(rows: list[dict[str, Any]]) -> pd.DataFrame:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("phase") == "adaptation":
            groups[str(row.get("condition", ""))].append(row)
    records = []
    for condition, group in sorted(groups.items()):
        retrieval_rows = [
            row for row in group if row.get("fallback_used") is not None
        ]
        retrieved = [
            row.get("retrieved_template_id")
            for row in retrieval_rows
            if row.get("retrieved_template_id")
        ]
        records.append(
            {
                "condition": condition,
                "adaptation_n": len(group),
                "new_templates": sum(
                    row.get("buffer_update_action") == "new" for row in group
                ),
                "merged_templates": sum(
                    row.get("buffer_update_action") == "merge" for row in group
                ),
                "rejected_updates": sum(
                    row.get("buffer_update_action") == "reject"
                    for row in group
                ),
                "fallback_rate": _mean(retrieval_rows, "fallback_used"),
                "retrieval_reuse_rate": (
                    float(
                        np.mean(
                            [
                                not bool(row.get("fallback_used"))
                                for row in retrieval_rows
                            ]
                        )
                    )
                    if retrieval_rows
                    else None
                ),
                "unique_retrieved_templates": len(set(retrieved)),
                "final_buffer_size": max(
                    [
                        int(row.get("buffer_size_after") or 0)
                        for row in group
                    ],
                    default=0,
                ),
            }
        )
    return pd.DataFrame(records)


def paired_bot_comparisons(
    rows: list[dict[str, Any]],
    *,
    baseline: str,
    seed: int,
    samples: int = 2000,
) -> pd.DataFrame:
    by_condition: dict[
        str, dict[tuple[str, str, str], dict[str, Any]]
    ] = defaultdict(dict)
    for row in rows:
        key = (
            str(row.get("phase", "")),
            str(row.get("case_id", "")),
            str(row.get("variant_id", "original")),
        )
        by_condition[str(row.get("condition", ""))][key] = row
    base = by_condition.get(baseline, {})
    records = []
    for condition in sorted(by_condition):
        if condition == baseline:
            continue
        common = sorted(set(base) & set(by_condition[condition]))
        if not common:
            continue
        base_correct = np.array([_itt(base[key]) for key in common])
        other_correct = np.array(
            [_itt(by_condition[condition][key]) for key in common]
        )
        difference = other_correct - base_correct
        rng = np.random.default_rng(seed)
        boot = np.array(
            [
                rng.choice(
                    difference, size=len(difference), replace=True
                ).mean()
                for _ in range(samples)
            ]
        )
        records.append(
            {
                "baseline": baseline,
                "condition": condition,
                "paired_n": len(common),
                "baseline_accuracy": float(base_correct.mean()),
                "condition_accuracy": float(other_correct.mean()),
                "accuracy_difference": float(difference.mean()),
                "difference_ci_low": float(np.quantile(boot, 0.025)),
                "difference_ci_high": float(np.quantile(boot, 0.975)),
            }
        )
    return pd.DataFrame(records)


def recommend_bot_next_step(
    summary: pd.DataFrame, curve: pd.DataFrame
) -> str:
    holdout = summary[summary["phase"] == "holdout"].set_index("condition")

    def accuracy(condition: str) -> float | None:
        if condition not in holdout.index:
            return None
        value = holdout.loc[condition, "answer_accuracy_itt"]
        return None if pd.isna(value) else float(value)

    full = accuracy("bot_full")
    semantic_dynamic = accuracy("semantic_qwen_dynamic")
    semantic_fixed = accuracy("semantic_qwen_fixed")
    semantic_raw = accuracy("semantic_raw_fixed")
    frontier_dynamic = accuracy("semantic_frontier_dynamic")
    frontier_fixed = accuracy("semantic_frontier_fixed")
    frontier_generic = accuracy("semantic_frontier_generic")
    fixed = accuracy("bot_no_manager")
    no_buffer = accuracy("bot_no_buffer")
    direct = accuracy("direct")
    if frontier_fixed is not None:
        if semantic_fixed is not None and frontier_fixed >= semantic_fixed + 0.05:
            return (
                "FRONTIER DISTILLER HELPS: with the same semantic retrieval "
                "and fixed buffer, frontier profiles improve held-out accuracy "
                "by at least five points over Qwen profiles."
            )
        if (
            frontier_dynamic is not None
            and frontier_dynamic <= frontier_fixed
        ):
            return (
                "NO EVIDENCE FOR FRONTIER ONLINE UPDATES: semantic retrieval "
                "with a fixed buffer is at least as strong as the append-only "
                "dynamic buffer."
            )
        if direct is not None and frontier_fixed < direct:
            return (
                "REFINE FRONTIER SEMANTIC BOT: the stronger distiller has not "
                "surpassed Direct on the frozen holdout."
            )
        return (
            "PROMISING FRONTIER SEMANTIC BOT: inspect paired gains and "
            "template routing before scaling."
        )
    if semantic_dynamic is not None:
        if (
            semantic_fixed is not None
            and semantic_dynamic <= semantic_fixed
        ):
            return (
                "NO EVIDENCE FOR SEMANTIC ONLINE UPDATES: the append-only "
                "buffer does not beat the fixed semantic buffer or Direct."
            )
        if direct is not None and semantic_dynamic < direct:
            return (
                "REFINE SEMANTIC BOT: semantic retrieval improves some "
                "components but remains below Direct on the frozen holdout."
            )
        if (
            semantic_raw is not None
            and semantic_dynamic >= semantic_raw + 0.05
        ):
            return (
                "SEMANTIC DISTILLATION HELPS: Qwen profiles plus append-only "
                "retrieval beat raw-case semantic routing by at least five "
                "points and are competitive with Direct."
            )
        return (
            "PROMISING SEMANTIC BOT: results are competitive with Direct, but "
            "component gains require paired inspection."
        )
    if full is None:
        return "INCOMPLETE: full BoT holdout results are unavailable."
    if fixed is not None and full <= fixed:
        return (
            "NO EVIDENCE FOR ONLINE UPDATES: the dynamically managed buffer "
            "does not beat the fixed seeded buffer on held-out cases."
        )
    if no_buffer is not None and full <= no_buffer:
        return (
            "NO EVIDENCE FOR TEMPLATE RETRIEVAL: full BoT does not beat the "
            "generic-template ablation on held-out cases."
        )
    full_curve = curve[curve["condition"] == "bot_full"].sort_values("round")
    if len(full_curve) >= 2 and float(
        full_curve.iloc[-1]["answer_accuracy_itt"]
    ) <= float(full_curve.iloc[0]["answer_accuracy_itt"]):
        return (
            "NO ONLINE LEARNING TREND: updates may help individual cases, but "
            "accuracy does not improve across the adaptation stream."
        )
    if direct is not None and full >= direct + 0.05:
        return (
            "GO: full Legal-BoT beats Direct by at least five points on the "
            "frozen holdout; inspect grounding and template reuse before scaling."
        )
    return (
        "REFINE: the buffer components show some incremental value, but full "
        "BoT has not clearly surpassed the Direct baseline."
    )


def build_bot_report(config: dict[str, Any]) -> dict[str, Any]:
    run_dir = (
        resolve_path(config, "runs_dir") / config["project"]["run_name"]
    )
    rows = read_jsonl(run_dir / "scored.jsonl")
    if not rows:
        raise FileNotFoundError("BoT scored outputs not found. Run bot-score.")
    report_dir = resolve_path(config, "reports_dir")
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = bot_condition_summary(rows)
    curve = adaptation_curve(rows)
    buffers = buffer_summary(rows)
    paired = paired_bot_comparisons(
        rows, baseline="direct", seed=config["project"]["seed"]
    )
    recommendation = recommend_bot_next_step(summary, curve)
    summary.to_csv(report_dir / "condition_summary.csv", index=False)
    curve.to_csv(report_dir / "adaptation_curve.csv", index=False)
    buffers.to_csv(report_dir / "buffer_summary.csv", index=False)
    paired.to_csv(report_dir / "paired_vs_direct.csv", index=False)
    chart = _write_curve_chart(curve, report_dir)
    update_counts = Counter(
        row.get("buffer_update_action", "missing") for row in rows
    )
    path = report_dir / "legal_bot_report.md"
    path.write_text(
        _render_report(
            summary=summary,
            curve=curve,
            buffers=buffers,
            paired=paired,
            recommendation=recommendation,
            chart=chart,
            update_counts=update_counts,
        ),
        encoding="utf-8",
    )
    return {
        "path": str(path),
        "chart": str(chart),
        "recommendation": recommendation,
        "records": len(rows),
    }


def _write_curve_chart(curve: pd.DataFrame, report_dir: Path) -> Path:
    path = report_dir / "adaptation_curve.png"
    fig, axis = plt.subplots(figsize=(8, 4.5))
    if not curve.empty:
        for condition, group in curve.groupby("condition"):
            axis.plot(
                group["round"],
                group["answer_accuracy_itt"],
                marker="o",
                label=condition,
            )
        axis.legend(fontsize=7, ncol=2)
    axis.set_xlabel("Adaptation-stream round")
    axis.set_ylabel("Failure-adjusted accuracy")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _render_report(
    *,
    summary: pd.DataFrame,
    curve: pd.DataFrame,
    buffers: pd.DataFrame,
    paired: pd.DataFrame,
    recommendation: str,
    chart: Path,
    update_counts: Counter,
) -> str:
    compared = "\n".join(
        f"- `{condition}`: {CONDITION_DESCRIPTIONS.get(condition, condition)}."
        for condition in summary["condition"].drop_duplicates()
    )
    return f"""# Legal Buffer-of-Thought Diagnostic Pilot

## Decision

**{recommendation}**

## What is compared

{compared}

Each adaptation case is predicted and scored before any update. The buffer is
then frozen for the holdout phase.

## Condition results

{_markdown_table(summary)}

## Paired comparisons against Direct

{_markdown_table(paired)}

## Online adaptation curve

![Adaptation curve]({chart.name})

{_markdown_table(curve)}

## Buffer behavior

{_markdown_table(buffers)}

Recorded update actions: {json.dumps(dict(update_counts), sort_keys=True)}.

## Interpretation limits

- This is a 64-case LegalHK direction-finding pilot, not a publication-scale
  estimate.
- Correctness-gated updates use the LegalHK outcome only after prediction; the
  outcome is never included in a model prompt or template.
- LegalHK fact summaries may retain latent judgment-derived cues.
- The hand-seeded templates are reasoning structures, not legal authorities.
"""


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and not pd.isna(row.get(key))
    ]
    return float(np.mean(values)) if values else None


def _itt(row: dict[str, Any]) -> float:
    return (
        float(row.get("answer_correct", 0.0))
        if row.get("status") == "ok"
        else 0.0
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No data."
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])
