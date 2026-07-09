from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import resolve_path
from .evaluation import _aggregate_frame
from .io_utils import latest_by_run_hash, read_jsonl, write_jsonl
from .models import FinalAnalysis
from .runner import load_cases
from .scoring import score_record


def score_legal_flux_run(
    config: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    normalized_phase = phase.replace("-", "_")
    run_dir = resolve_path(config, "runs_dir") / normalized_phase
    rows = latest_by_run_hash(read_jsonl(run_dir / "generations.jsonl"))
    rows = _filter_to_run_plan(rows, run_dir)
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
                value.update(score_record(case, analysis))
                value["prediction"] = analysis.final_decision
                value["trajectory_length"] = len(row.get("executed_steps") or [])
                value["review_count"] = len(row.get("trajectory_reviews") or [])
            except Exception as exc:
                value["status"] = "score_error"
                value["score_error"] = str(exc)
        scored.append(value)
    write_jsonl(run_dir / "scored.jsonl", scored)
    ok = [row for row in scored if row.get("status") == "ok"]
    frame = pd.DataFrame(ok)
    aggregate = _aggregate_frame(frame)
    if not frame.empty and {"dataset", "condition", "trajectory_length"}.issubset(frame.columns):
        trajectory = (
            frame.groupby(["dataset", "condition"], dropna=False)[
                ["trajectory_length", "review_count"]
            ]
            .mean(numeric_only=True)
            .reset_index()
        )
        aggregate = aggregate.merge(
            trajectory,
            on=["dataset", "condition"],
            how="left",
            suffixes=("", "_flux"),
        )
    aggregate.to_csv(run_dir / "aggregate.csv", index=False)
    summary = {
        "phase": normalized_phase,
        "records": len(scored),
        "ok_records": len(ok),
        "error_records": len(scored) - len(ok),
        "aggregate_path": str(run_dir / "aggregate.csv"),
        "scored_path": str(run_dir / "scored.jsonl"),
    }
    (run_dir / "score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _filter_to_run_plan(
    rows: list[dict[str, Any]],
    run_dir: Path,
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
