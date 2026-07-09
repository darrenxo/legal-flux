from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .config import resolve_path
from .io_utils import latest_by_run_hash, read_jsonl, write_jsonl
from .models import FinalAnalysis, NormalizedCase
from .runner import load_cases
from .scoring import score_record


def enrich_bot_record(
    row: dict[str, Any], case: NormalizedCase
) -> dict[str, Any]:
    value = dict(row)
    if row.get("status") != "ok":
        return value
    analysis = FinalAnalysis.model_validate(row["parsed_json"])
    value.update(score_record(case, analysis))
    value["prediction"] = analysis.final_decision
    retrieval = row.get("retrieval") or {}
    template = retrieval.get("template") or {}
    update = row.get("buffer_update") or {}
    before = int(row.get("buffer_size_before") or 0)
    after = int(row.get("buffer_size_after") or 0)
    value.update(
        {
            "retrieval_similarity": retrieval.get("similarity"),
            "fallback_used": float(bool(retrieval.get("used_fallback")))
            if retrieval
            else None,
            "retrieved_template_id": template.get("template_id"),
            "buffer_update_action": update.get("action"),
            "buffer_growth": after - before,
        }
    )
    return value


def score_bot_run(
    config: dict[str, Any], *, smoke: bool = False
) -> dict[str, Any]:
    run_dir = (
        resolve_path(config, "runs_dir")
        / ("smoke" if smoke else config["project"]["run_name"])
    )
    records = latest_by_run_hash(read_jsonl(run_dir / "generations.jsonl"))
    records = _filter_to_plan(records, run_dir)
    cases = {
        (case.dataset, case.case_id, case.variant_id): case
        for case in load_cases(config)
    }
    scored: list[dict[str, Any]] = []
    for row in records:
        key = (row["dataset"], row["case_id"], row["variant_id"])
        try:
            scored.append(enrich_bot_record(row, cases[key]))
        except Exception as exc:
            value = dict(row)
            value["status"] = "score_error"
            value["score_error"] = str(exc)
            scored.append(value)
    path = run_dir / "scored.jsonl"
    write_jsonl(path, scored)
    ok = [row for row in scored if row.get("status") == "ok"]
    frame = pd.DataFrame(ok)
    aggregate_path = run_dir / "aggregate.csv"
    if frame.empty:
        aggregate_path.write_text("", encoding="utf-8")
    else:
        numeric = [
            column
            for column in (
                "answer_correct",
                "binary_prediction_valid",
                "conclusion_with_fact_rate",
                "valid_fact_reference_rate",
                "unknown_fact_reference_count",
                "issue_coverage_proxy",
                "retrieval_similarity",
                "fallback_used",
                "buffer_growth",
                "elapsed_seconds",
                "prompt_tokens",
                "output_tokens",
                "calls",
            )
            if column in frame.columns
        ]
        aggregate = (
            frame.groupby(["condition", "phase"], dropna=False)[numeric]
            .mean(numeric_only=True)
            .reset_index()
        )
        aggregate.to_csv(aggregate_path, index=False)
    summary = {
        "records": len(scored),
        "ok_records": len(ok),
        "error_records": len(scored) - len(ok),
        "scored_path": str(path),
        "aggregate_path": str(aggregate_path),
    }
    (run_dir / "score_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _filter_to_plan(
    records: list[dict[str, Any]], run_dir
) -> list[dict[str, Any]]:
    path = run_dir / "run_plan.json"
    if not path.exists():
        return records
    plan = json.loads(path.read_text(encoding="utf-8"))
    allowed = {
        item["run_hash"]
        for item in plan.get("jobs", [])
        if isinstance(item, dict) and item.get("run_hash")
    }
    return [row for row in records if row.get("run_hash") in allowed]
