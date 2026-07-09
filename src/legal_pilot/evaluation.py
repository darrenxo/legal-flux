from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import resolve_path
from .io_utils import latest_by_run_hash, read_jsonl, write_jsonl
from .models import FinalAnalysis, NormalizedCase
from .runner import load_cases
from .scoring import score_record


def score_run(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    run_name = "smoke" if smoke else config["project"]["run_name"]
    run_dir = resolve_path(config, "runs_dir") / run_name
    generation_path = run_dir / "generations.jsonl"
    rows = latest_by_run_hash(read_jsonl(generation_path))
    rows = _filter_to_run_plan(rows, run_dir)
    if smoke:
        freeze_path = resolve_path(config, "processed_dir") / "frozen_manifest.json"
        if freeze_path.exists():
            frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
            allowed = set(frozen.get("smoke_run_hashes", []))
            rows = [row for row in rows if row.get("run_hash") in allowed]
    cases = {
        (case.dataset, case.case_id, case.variant_id): case
        for case in load_cases(config)
    }

    scored: list[dict[str, Any]] = []
    for row in rows:
        value = dict(row)
        if row.get("status") == "ok":
            case = cases[(row["dataset"], row["case_id"], row["variant_id"])]
            try:
                analysis = FinalAnalysis.model_validate(row["parsed_json"])
                metrics = score_record(case, analysis)
                value.update(metrics)
                value["prediction"] = analysis.final_decision
            except Exception as exc:
                value["status"] = "score_error"
                value["score_error"] = str(exc)
        scored.append(value)
    write_jsonl(run_dir / "scored.jsonl", scored)

    ok = [row for row in scored if row.get("status") == "ok"]
    frame = pd.DataFrame(ok)
    aggregate = _aggregate_frame(frame)
    aggregate.to_csv(run_dir / "aggregate.csv", index=False)
    summary = {
        "records": len(scored),
        "ok_records": len(ok),
        "error_records": len(scored) - len(ok),
        "aggregate_path": str(run_dir / "aggregate.csv"),
        "scored_path": str(run_dir / "scored.jsonl"),
    }
    (run_dir / "score_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _filter_to_run_plan(
    rows: list[dict[str, Any]], run_dir: Path
) -> list[dict[str, Any]]:
    plan_path = run_dir / "run_plan.json"
    if not plan_path.exists():
        return rows
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    allowed = {
        job["run_hash"]
        for job in plan.get("jobs", [])
        if isinstance(job, dict) and job.get("run_hash")
    }
    return [row for row in rows if row.get("run_hash") in allowed]


def _aggregate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    numeric = [
        column
        for column in (
            "answer_correct",
            "binary_prediction_valid",
            "conclusion_with_fact_rate",
            "valid_fact_reference_rate",
            "unknown_fact_reference_count",
            "issue_coverage_proxy",
            "elapsed_seconds",
            "prompt_tokens",
            "output_tokens",
            "calls",
        )
        if column in frame.columns
    ]
    grouped = (
        frame.groupby(["dataset", "condition"], dropna=False)[numeric]
        .mean(numeric_only=True)
        .reset_index()
    )
    counts = (
        frame.groupby(["dataset", "condition"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    grouped = counts.merge(grouped, on=["dataset", "condition"], how="left")
    for metric in ("answer_correct", "valid_fact_reference_rate"):
        if metric in frame.columns:
            intervals = []
            for keys, group in frame.groupby(["dataset", "condition"], dropna=False):
                low, high = bootstrap_ci(group[metric].dropna().to_numpy(), seed=20260619)
                intervals.append(
                    {
                        "dataset": keys[0],
                        "condition": keys[1],
                        f"{metric}_ci_low": low,
                        f"{metric}_ci_high": high,
                    }
                )
            grouped = grouped.merge(
                pd.DataFrame(intervals), on=["dataset", "condition"], how="left"
            )
    return grouped


def bootstrap_ci(
    values: np.ndarray, *, seed: int, samples: int = 2000
) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    means = np.array(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)]
    )
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
