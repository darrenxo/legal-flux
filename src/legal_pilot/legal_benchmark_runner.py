from __future__ import annotations

import json
import re
import sys
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from .clients import GenerationClient, GenerationResponseError, build_generation_client
from .config import resolve_path
from .io_utils import (
    atomic_write_json,
    canonical_json,
    latest_by_run_hash,
    read_jsonl,
    sha256_text,
    write_jsonl,
)
from .ledger import JsonlLedger, make_run_hash
from .legal_benchmark_data import (
    BenchmarkCase,
    benchmark_path,
    select_benchmark_cases,
)


BENCHMARK_CONDITIONS = ("direct", "structured")


def generate_legal_benchmarks(
    config: dict[str, Any],
    *,
    datasets: list[str] | None,
    subset: str,
    conditions: list[str] | None,
    run_tag: str,
    case_limit: int | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
    dry_run: bool = False,
    fail_on_errors: bool = False,
) -> dict[str, Any]:
    normalized_tag = _validated_run_tag(run_tag)
    selected_conditions = _validated_conditions(conditions)
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1.")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards).")
    cases = select_benchmark_cases(
        config,
        datasets=datasets,
        subset=subset,
        case_limit=case_limit,
    )
    jobs = [
        {"case": case, "condition": condition}
        for case in sorted(cases, key=lambda value: (value.dataset, value.case_id))
        for condition in selected_conditions
    ]
    jobs = [
        job for index, job in enumerate(jobs) if index % num_shards == shard_index
    ]
    run_dir = benchmark_run_dir(
        config,
        subset=subset,
        run_tag=normalized_tag,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    if dry_run:
        return {
            "subset": subset,
            "run_tag": normalized_tag,
            "jobs": len(jobs),
            "cases": len({(job["case"].dataset, job["case"].case_id) for job in jobs}),
            "datasets": sorted({job["case"].dataset for job in jobs}),
            "conditions": selected_conditions,
            "num_shards": num_shards,
            "shard_index": shard_index,
            "run_dir": str(run_dir),
            "dry_run": True,
        }

    client = build_generation_client(config)
    model_name = str(config["model"]["name"])
    try:
        model_info = client.model_info(model_name)
        if not model_info:
            raise RuntimeError(
                f"Model {model_name!r} is not exposed by "
                f"{config['model'].get('provider', 'ollama')} at "
                f"{config['model']['base_url']}."
            )
        model_digest = str(model_info.get("digest") or model_info.get("id") or model_name)
        planned = [
            _planned_job(
                config,
                job["case"],
                condition=job["condition"],
                subset=subset,
                model_digest=model_digest,
            )
            for job in jobs
        ]
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            run_dir / "run_plan.json",
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "subset": subset,
                "run_tag": normalized_tag,
                "model_name": model_name,
                "model_digest": model_digest,
                "model_runtime": _model_runtime_metadata(config, model_info),
                "num_shards": num_shards,
                "shard_index": shard_index,
                "job_count": len(planned),
                "jobs": planned,
            },
        )
        ledger = JsonlLedger(run_dir / "generations.jsonl")
        concurrency = max(1, int(config["model"].get("concurrency", 1)))
        completed = skipped = errors = 0
        if concurrency == 1:
            records = (
                _run_job(
                    client,
                    config,
                    job["case"],
                    condition=job["condition"],
                    subset=subset,
                    model_digest=model_digest,
                    ledger=ledger,
                )
                for job in jobs
            )
            for record in records:
                completed, skipped, errors = _update_counts(
                    record,
                    completed=completed,
                    skipped=skipped,
                    errors=errors,
                )
                _print_progress(completed, skipped, errors, len(jobs))
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(
                        _run_job,
                        client,
                        config,
                        job["case"],
                        condition=job["condition"],
                        subset=subset,
                        model_digest=model_digest,
                        ledger=ledger,
                    )
                    for job in jobs
                ]
                for future in as_completed(futures):
                    completed, skipped, errors = _update_counts(
                        future.result(),
                        completed=completed,
                        skipped=skipped,
                        errors=errors,
                    )
                    _print_progress(completed, skipped, errors, len(jobs))
    finally:
        client.close()

    result = {
        "subset": subset,
        "run_tag": normalized_tag,
        "jobs": len(jobs),
        "completed": completed,
        "skipped": skipped,
        "errors": errors,
        "model_name": model_name,
        "model_digest": model_digest,
        "model_runtime": _model_runtime_metadata(config, model_info),
        "num_shards": num_shards,
        "shard_index": shard_index,
        "concurrency": concurrency,
        "run_dir": str(run_dir),
    }
    if fail_on_errors and errors:
        raise RuntimeError(
            f"Benchmark generation recorded {errors} error(s) in {run_dir}; "
            "successful ledger records were preserved."
        )
    return result


def score_legal_benchmarks(
    config: dict[str, Any],
    *,
    subset: str,
    run_tag: str,
) -> dict[str, Any]:
    normalized_tag = _validated_run_tag(run_tag)
    root = benchmark_path(config, "runs_dir") / subset / normalized_tag
    generation_paths = sorted((root / "shards").glob("*/generations.jsonl"))
    if not generation_paths:
        unsharded = root / "generations.jsonl"
        if not unsharded.exists():
            raise FileNotFoundError(f"No benchmark generation ledger found under {root}.")
        generation_paths = [unsharded]
    rows = latest_by_run_hash(
        row for path in generation_paths for row in read_jsonl(path)
    )
    rows = _filter_to_plans(rows, [path.parent / "run_plan.json" for path in generation_paths])
    scored: list[dict[str, Any]] = []
    for row in rows:
        value = dict(row)
        parsed = row.get("parsed_json") or {}
        prediction = parsed.get("final_decision") if row.get("status") == "ok" else None
        value["prediction"] = prediction
        value["answer_correct"] = (
            int(prediction == row.get("gold_label")) if prediction is not None else None
        )
        scored.append(value)
    write_jsonl(root / "scored.jsonl", scored)
    ok = [row for row in scored if row.get("status") == "ok"]
    aggregate, matrices = _aggregate_scores(ok)
    aggregate.to_csv(root / "aggregate.csv", index=False)
    atomic_write_json(root / "confusion_matrices.json", matrices)
    paired = _paired_comparisons(ok)
    paired.to_csv(root / "paired_comparison.csv", index=False)
    summary = {
        "subset": subset,
        "run_tag": normalized_tag,
        "records": len(scored),
        "ok_records": len(ok),
        "error_records": len(scored) - len(ok),
        "generation_files": [str(path) for path in generation_paths],
        "aggregate_path": str(root / "aggregate.csv"),
        "paired_path": str(root / "paired_comparison.csv"),
        "confusion_matrices_path": str(root / "confusion_matrices.json"),
        "scored_path": str(root / "scored.jsonl"),
    }
    atomic_write_json(root / "score_summary.json", summary)
    return summary


def render_benchmark_prompt(
    config: dict[str, Any],
    case: BenchmarkCase,
    condition: str,
) -> tuple[str, dict[str, Any]]:
    if condition not in BENCHMARK_CONDITIONS:
        raise ValueError(f"Unsupported benchmark condition: {condition}")
    template_path = resolve_path(config, "prompts_dir") / f"benchmark_{condition}.txt"
    template = template_path.read_text(encoding="utf-8")
    max_characters = int(config["benchmarks"].get("max_input_characters", 48000))
    strategy = str(config["benchmarks"].get("input_truncation", "head"))
    case_text, truncation = _truncate_input(
        case.input_text,
        max_characters=max_characters,
        strategy=strategy,
    )
    label_lines = "\n".join(
        f'- "{label}": {case.label_descriptions[label]}' for label in case.labels
    )
    prompt = template.format(
        task_instruction=case.task_instruction,
        labels=label_lines,
        case_text=case_text,
    )
    return prompt, {
        **truncation,
        "prompt_sha256": sha256_text(prompt),
        "prompt_characters": len(prompt),
    }


def benchmark_run_dir(
    config: dict[str, Any],
    *,
    subset: str,
    run_tag: str,
    num_shards: int,
    shard_index: int,
) -> Path:
    root = benchmark_path(config, "runs_dir") / subset / run_tag
    if num_shards == 1:
        return root
    return root / "shards" / f"shard-{shard_index:05d}-of-{num_shards:05d}"


def _planned_job(
    config: dict[str, Any],
    case: BenchmarkCase,
    *,
    condition: str,
    subset: str,
    model_digest: str,
) -> dict[str, Any]:
    prompt, prompt_metadata = render_benchmark_prompt(config, case, condition)
    del prompt
    return {
        "run_hash": _benchmark_run_hash(
            config,
            case,
            condition=condition,
            subset=subset,
            model_digest=model_digest,
            prompt_sha256=prompt_metadata["prompt_sha256"],
        ),
        "dataset": case.dataset,
        "case_id": case.case_id,
        "condition": condition,
        "source_split": case.source_split,
        "prompt_sha256": prompt_metadata["prompt_sha256"],
    }


def _run_job(
    client: GenerationClient,
    config: dict[str, Any],
    case: BenchmarkCase,
    *,
    condition: str,
    subset: str,
    model_digest: str,
    ledger: JsonlLedger,
) -> dict[str, Any] | None:
    prompt, prompt_metadata = render_benchmark_prompt(config, case, condition)
    run_hash = _benchmark_run_hash(
        config,
        case,
        condition=condition,
        subset=subset,
        model_digest=model_digest,
        prompt_sha256=prompt_metadata["prompt_sha256"],
    )
    if ledger.contains(run_hash):
        return None
    base = {
        "run_hash": run_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": case.dataset,
        "case_id": case.case_id,
        "source_split": case.source_split,
        "subset": subset,
        "condition": condition,
        "gold_label": case.gold_label,
        "labels": case.labels,
        "model_name": str(config["model"]["name"]),
        "model_digest": model_digest,
        "model_runtime": str(config["model"].get("runtime_variant") or "unspecified"),
        "temperature": float(config["model"].get("temperature", 0.0)),
        "seed": int(config["model"].get("seed", config["project"]["seed"])),
        "prompt_sha256": prompt_metadata["prompt_sha256"],
        "input": prompt_metadata,
        "metadata": case.metadata,
    }
    try:
        schema = _response_schema(case.labels, condition)
        max_tokens_key = (
            "direct_max_tokens" if condition == "direct" else "structured_max_tokens"
        )
        response = client.generate(
            model=str(config["model"]["name"]),
            prompt=prompt,
            schema=schema,
            temperature=float(config["model"].get("temperature", 0.0)),
            seed=int(config["model"].get("seed", config["project"]["seed"])),
            context_length=int(config["model"]["context_length"]),
            max_tokens=int(config["benchmarks"].get(max_tokens_key, 800)),
            think=False,
        )
        parsed = _validate_response(response.parsed, case.labels, condition)
        record = {
            **base,
            "status": "ok",
            "raw_response": response.raw_text,
            "parsed_json": parsed,
            "elapsed_seconds": response.elapsed_seconds,
            "prompt_tokens": response.prompt_tokens,
            "output_tokens": response.output_tokens,
            "finish_reason": response.metadata.get("finish_reason")
            or response.metadata.get("done_reason"),
            "json_repair_applied": bool(
                response.metadata.get("json_repair_applied")
            ),
        }
    except Exception as exc:
        record = {
            **base,
            "status": "error",
            "raw_response": (
                exc.raw_text if isinstance(exc, GenerationResponseError) else None
            ),
            "parsed_json": None,
            "elapsed_seconds": None,
            "prompt_tokens": None,
            "output_tokens": None,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    ledger.append(record)
    return record


def _response_schema(labels: list[str], condition: str) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "final_decision": {"type": "string", "enum": labels},
    }
    required = ["final_decision"]
    if condition == "direct":
        properties["final_rationale"] = {"type": "string"}
        required.append("final_rationale")
    elif condition == "structured":
        properties["irac_reasoning"] = {"type": "string"}
        required.append("irac_reasoning")
    else:
        raise ValueError(condition)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def _validate_response(
    parsed: dict[str, Any] | None,
    labels: list[str],
    condition: str,
) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError("Model response did not parse to a JSON object.")
    decision = str(parsed.get("final_decision") or "").strip().lower()
    if decision not in labels:
        raise ValueError(f"Invalid final_decision {decision!r}; expected {labels!r}.")
    result: dict[str, Any] = {"final_decision": decision}
    reasoning_key = "final_rationale" if condition == "direct" else "irac_reasoning"
    reasoning = str(parsed.get(reasoning_key) or "").strip()
    if not reasoning:
        raise ValueError(f"Model response lacks non-empty {reasoning_key}.")
    result[reasoning_key] = reasoning
    return result


def _benchmark_run_hash(
    config: dict[str, Any],
    case: BenchmarkCase,
    *,
    condition: str,
    subset: str,
    model_digest: str,
    prompt_sha256: str,
) -> str:
    return make_run_hash(
        workflow="legal_benchmark_direct_structured_v1",
        dataset=case.dataset,
        case_id=case.case_id,
        source_split=case.source_split,
        subset=subset,
        condition=condition,
        gold_label=case.gold_label,
        input_sha256=sha256_text(case.input_text),
        prompt_sha256=prompt_sha256,
        model_name=config["model"]["name"],
        model_digest=model_digest,
        runtime_variant=config["model"].get("runtime_variant"),
        context_length=config["model"]["context_length"],
        temperature=config["model"].get("temperature", 0.0),
        seed=config["model"].get("seed", config["project"]["seed"]),
    )


def _truncate_input(
    text: str,
    *,
    max_characters: int,
    strategy: str,
) -> tuple[str, dict[str, Any]]:
    if max_characters < 1:
        raise ValueError("benchmarks.max_input_characters must be positive.")
    original = len(text)
    if original <= max_characters:
        return text, {
            "original_characters": original,
            "used_characters": original,
            "truncated": False,
            "truncation_strategy": "none",
        }
    if strategy == "head":
        used = text[:max_characters]
    elif strategy == "head_tail":
        head = max_characters * 3 // 4
        tail = max_characters - head
        used = text[:head] + "\n\n[... middle omitted ...]\n\n" + text[-tail:]
    else:
        raise ValueError(f"Unsupported benchmark input truncation: {strategy}")
    return used, {
        "original_characters": original,
        "used_characters": len(used),
        "truncated": True,
        "truncation_strategy": strategy,
    }


def _aggregate_scores(
    rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not rows:
        return pd.DataFrame(), {}
    records: list[dict[str, Any]] = []
    matrices: dict[str, Any] = {}
    frame = pd.DataFrame(rows)
    for (dataset, condition), group in frame.groupby(["dataset", "condition"]):
        labels = list(group.iloc[0]["labels"])
        gold = group["gold_label"].astype(str).tolist()
        predictions = group["prediction"].astype(str).tolist()
        correct = np.asarray([left == right for left, right in zip(gold, predictions, strict=True)], dtype=float)
        low, high = _bootstrap_mean_ci(correct, seed=20260619)
        majority_count = max(Counter(gold).values())
        records.append(
            {
                "dataset": dataset,
                "condition": condition,
                "n": len(group),
                "accuracy": accuracy_score(gold, predictions),
                "accuracy_ci_low": low,
                "accuracy_ci_high": high,
                "macro_f1": f1_score(
                    gold,
                    predictions,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                ),
                "weighted_f1": f1_score(
                    gold,
                    predictions,
                    labels=labels,
                    average="weighted",
                    zero_division=0,
                ),
                "balanced_accuracy": balanced_accuracy_score(gold, predictions),
                "majority_accuracy": majority_count / len(group),
                "mean_elapsed_seconds": group["elapsed_seconds"].mean(),
                "mean_prompt_tokens": group["prompt_tokens"].mean(),
                "mean_output_tokens": group["output_tokens"].mean(),
                "truncation_rate": group["input"].map(
                    lambda value: bool(value.get("truncated"))
                ).mean(),
            }
        )
        matrix = confusion_matrix(gold, predictions, labels=labels)
        matrices[f"{dataset}/{condition}"] = {
            "labels": labels,
            "matrix": matrix.tolist(),
        }
    return pd.DataFrame(records).sort_values(["dataset", "condition"]), matrices


def _paired_comparisons(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    records: list[dict[str, Any]] = []
    for dataset, group in frame.groupby("dataset"):
        pivot = group.pivot_table(
            index="case_id",
            columns="condition",
            values="answer_correct",
            aggfunc="first",
        )
        if not {"direct", "structured"}.issubset(pivot.columns):
            continue
        paired = pivot[["direct", "structured"]].dropna()
        direct = paired["direct"].astype(float).to_numpy()
        structured = paired["structured"].astype(float).to_numpy()
        deltas = structured - direct
        low, high = _bootstrap_mean_ci(deltas, seed=20260620)
        structured_only = int(np.sum((structured == 1) & (direct == 0)))
        direct_only = int(np.sum((structured == 0) & (direct == 1)))
        discordant = structured_only + direct_only
        p_value = (
            float(
                binomtest(
                    min(structured_only, direct_only),
                    n=discordant,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )
            if discordant
            else 1.0
        )
        records.append(
            {
                "dataset": dataset,
                "paired_n": len(paired),
                "direct_accuracy": direct.mean(),
                "structured_accuracy": structured.mean(),
                "structured_minus_direct": deltas.mean(),
                "delta_ci_low": low,
                "delta_ci_high": high,
                "structured_only_correct": structured_only,
                "direct_only_correct": direct_only,
                "mcnemar_exact_p": p_value,
            }
        )
    return pd.DataFrame(records)


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    samples: int = 2000,
) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None, None
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)]
    )
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _filter_to_plans(
    rows: list[dict[str, Any]],
    plan_paths: list[Path],
) -> list[dict[str, Any]]:
    allowed: set[str] = set()
    for path in plan_paths:
        if not path.exists():
            continue
        plan = json.loads(path.read_text(encoding="utf-8"))
        allowed.update(
            job["run_hash"]
            for job in plan.get("jobs", [])
            if isinstance(job, dict) and job.get("run_hash")
        )
    return [row for row in rows if not allowed or row.get("run_hash") in allowed]


def _model_runtime_metadata(
    config: dict[str, Any],
    model_info: dict[str, Any],
) -> dict[str, Any]:
    details = model_info.get("details") or {}
    return {
        "provider": config["model"].get("provider", "ollama"),
        "runtime_variant": config["model"].get("runtime_variant"),
        "inference_runtime": config["model"].get("inference_runtime"),
        "inference_runtime_version": config["model"].get(
            "inference_runtime_version"
        ),
        "format": details.get("format"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
    }


def _validated_conditions(conditions: list[str] | None) -> list[str]:
    selected = list(conditions or BENCHMARK_CONDITIONS)
    unknown = sorted(set(selected) - set(BENCHMARK_CONDITIONS))
    if unknown:
        raise ValueError(f"Unsupported benchmark conditions: {unknown}")
    if not selected:
        raise ValueError("At least one benchmark condition is required.")
    return list(dict.fromkeys(selected))


def _validated_run_tag(value: str) -> str:
    text = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", text):
        raise ValueError(
            "Benchmark run_tag must start with an alphanumeric character and "
            "contain only letters, digits, dot, underscore, or hyphen."
        )
    return text


def _update_counts(
    record: dict[str, Any] | None,
    *,
    completed: int,
    skipped: int,
    errors: int,
) -> tuple[int, int, int]:
    if record is None:
        skipped += 1
    elif record.get("status") == "ok":
        completed += 1
    else:
        errors += 1
    return completed, skipped, errors


def _print_progress(completed: int, skipped: int, errors: int, total: int) -> None:
    processed = completed + skipped + errors
    print(
        f"benchmark progress {processed}/{total} "
        f"(ok={completed}, skipped={skipped}, errors={errors})",
        file=sys.stderr,
        flush=True,
    )
