from __future__ import annotations

import csv
import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audit_selection import balanced_select
from .clients import OllamaClient, OpenAIAuditClient
from .config import resolve_path
from .io_utils import read_jsonl, sha256_text, write_jsonl
from .ledger import JsonlLedger, make_run_hash
from .models import AuditResult
from .prompting import render_prompt
from .runner import load_cases


def select_audit(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    run_name = "smoke" if smoke else config["project"]["run_name"]
    run_dir = resolve_path(config, "runs_dir") / run_name
    rows = read_jsonl(run_dir / "scored.jsonl")
    selected = balanced_select(
        rows,
        limit=min(config["audit"]["max_outputs"], len(rows)),
        seed=config["project"]["seed"],
    )
    # Keep condition in the local selection record, but the audit prompt never includes it.
    write_jsonl(run_dir / "audit_selection.jsonl", selected)
    return {
        "selected": len(selected),
        "path": str(run_dir / "audit_selection.jsonl"),
    }


def run_audit(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    run_name = "smoke" if smoke else config["project"]["run_name"]
    run_dir = resolve_path(config, "runs_dir") / run_name
    selected = read_jsonl(run_dir / "audit_selection.jsonl")
    cases = {
        (case.dataset, case.case_id, case.variant_id): case
        for case in load_cases(config)
    }
    client = OpenAIAuditClient(
        config["audit"]["model"], config["audit"]["reasoning_effort"]
    )
    ledger = JsonlLedger(run_dir / "audits.jsonl")
    completed = 0
    skipped = 0
    for row in selected:
        case = cases[(row["dataset"], row["case_id"], row["variant_id"])]
        prompt, prompt_hash = render_prompt(
            config,
            "audit",
            case,
            generated_output=row["parsed_json"],
        )
        audit_hash = make_run_hash(
            generation_run_hash=row["run_hash"],
            audit_model=config["audit"]["model"],
            prompt_hash=prompt_hash,
        )
        if ledger.contains(audit_hash):
            skipped += 1
            continue
        parsed, metadata = client.audit(prompt, AuditResult)
        ledger.append(
            {
                "run_hash": audit_hash,
                "generation_run_hash": row["run_hash"],
                "dataset": row["dataset"],
                "case_id": row["case_id"],
                "variant_id": row["variant_id"],
                "condition": row["condition"],
                "audit_model": config["audit"]["model"],
                "prompt_hash": prompt_hash,
                "audit": parsed.model_dump(mode="json"),
                "usage": metadata,
            }
        )
        completed += 1
    return {
        "completed": completed,
        "skipped": skipped,
        "path": str(run_dir / "audits.jsonl"),
    }


def run_local_audit(
    config: dict[str, Any],
    *,
    smoke: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    run_name = "smoke" if smoke else config["project"]["run_name"]
    run_dir = resolve_path(config, "runs_dir") / run_name
    selected = read_jsonl(run_dir / "audit_selection.jsonl")
    if limit is not None:
        selected = selected[:limit]
    cases = {
        (case.dataset, case.case_id, case.variant_id): case
        for case in load_cases(config)
    }
    settings = config["audit"]["local"]
    client = OllamaClient(
        settings.get("base_url", config["model"]["base_url"]),
        settings.get("timeout_seconds", config["model"]["timeout_seconds"]),
    )
    model_info = client.model_info(settings["model"])
    if not model_info:
        client.close()
        raise RuntimeError(
            f"Local audit model {settings['model']!r} is not installed."
        )
    digest = model_info.get("digest", "unknown")
    model_slug = re.sub(r"[^a-z0-9]+", "_", settings["model"].lower()).strip("_")
    ledger = JsonlLedger(run_dir / f"audits_local_{model_slug}.jsonl")
    schema = json.loads(
        (resolve_path(config, "schemas_dir") / "audit_result.json").read_text(
            encoding="utf-8"
        )
    )
    completed = 0
    skipped = 0
    try:
        for row in selected:
            case = cases[(row["dataset"], row["case_id"], row["variant_id"])]
            prompt, prompt_hash = render_prompt(
                config,
                "audit",
                case,
                generated_output=row["parsed_json"],
            )
            audit_hash = make_run_hash(
                generation_run_hash=row["run_hash"],
                audit_model=settings["model"],
                audit_model_digest=digest,
                reasoning_effort=settings["reasoning_effort"],
                prompt_hash=prompt_hash,
            )
            if ledger.contains(audit_hash):
                skipped += 1
                continue
            base = {
                "run_hash": audit_hash,
                "generation_run_hash": row["run_hash"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dataset": row["dataset"],
                "case_id": row["case_id"],
                "variant_id": row["variant_id"],
                "condition": row["condition"],
                "audit_provider": "ollama",
                "audit_model": settings["model"],
                "audit_model_digest": digest,
                "reasoning_effort": settings["reasoning_effort"],
                "prompt_hash": prompt_hash,
                "decoding": {
                    "temperature": settings["temperature"],
                    "seed": settings["seed"],
                    "context_length": settings["context_length"],
                    "max_tokens": settings["max_tokens"],
                },
            }
            try:
                response = client.generate(
                    model=settings["model"],
                    prompt=prompt,
                    schema=schema,
                    temperature=settings["temperature"],
                    seed=settings["seed"],
                    context_length=settings["context_length"],
                    max_tokens=settings["max_tokens"],
                    think=settings["reasoning_effort"],
                )
                parsed = AuditResult.model_validate(response.parsed)
                ledger.append(
                    {
                        **base,
                        "status": "ok",
                        "audit": parsed.model_dump(mode="json"),
                        "raw_response": response.raw_text,
                        "usage": {
                            "elapsed_seconds": response.elapsed_seconds,
                            "input_tokens": response.prompt_tokens,
                            "output_tokens": response.output_tokens,
                            **response.metadata,
                        },
                    }
                )
            except Exception as exc:
                ledger.append(
                    {
                        **base,
                        "status": "error",
                        "audit": None,
                        "raw_response": getattr(exc, "raw_text", None),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            completed += 1
    finally:
        client.close()
    return {
        "selected": len(selected),
        "completed": completed,
        "skipped": skipped,
        "model": settings["model"],
        "model_digest": digest,
        "path": str(ledger.path),
    }


def export_chatgpt_audit(
    config: dict[str, Any], *, batch_size: int = 10
) -> dict[str, Any]:
    run_dir = (
        resolve_path(config, "runs_dir") / config["project"]["run_name"]
    )
    selected = read_jsonl(run_dir / "audit_selection.jsonl")
    cases = {
        (case.dataset, case.case_id, case.variant_id): case
        for case in load_cases(config)
    }
    items, key = _build_chatgpt_audit_items(
        selected, cases, seed=config["project"]["seed"]
    )
    output_dir = resolve_path(config, "reports_dir") / "chatgpt_frontier_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("batch_*.jsonl"):
        stale.unlink()
    batches = _batch_items(items, size=batch_size)
    for index, batch in enumerate(batches, start=1):
        write_jsonl(output_dir / f"batch_{index:02d}.jsonl", batch)
    _write_audit_key(output_dir / "hidden_key.csv", key)
    (output_dir / "audit_result.schema.json").write_text(
        (
            resolve_path(config, "schemas_dir") / "audit_result.json"
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (output_dir / "INSTRUCTIONS.md").write_text(
        _chatgpt_instructions(len(batches)),
        encoding="utf-8",
    )
    return {
        "items": len(items),
        "batches": len(batches),
        "batch_size": batch_size,
        "directory": str(output_dir),
        "instructions": str(output_dir / "INSTRUCTIONS.md"),
    }


def _build_chatgpt_audit_items(
    selected: list[dict[str, Any]],
    cases: dict[tuple[str, str, str], Any],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for row in selected:
        case = cases[(row["dataset"], row["case_id"], row["variant_id"])]
        audit_id = sha256_text(f"chatgpt|{seed}|{row['run_hash']}")[:16]
        items.append(
            {
                "audit_id": audit_id,
                "task": case.claim,
                "facts": case.facts,
                "reference_issues": case.reference_issues,
                "gold_answer": case.gold_answer,
                "generated_output": row["parsed_json"],
            }
        )
        key.append(
            {
                "audit_id": audit_id,
                "generation_run_hash": row["run_hash"],
                "dataset": row["dataset"],
                "case_id": row["case_id"],
                "variant_id": row["variant_id"],
                "condition": row["condition"],
            }
        )
    return items, key


def _batch_items(
    items: list[dict[str, Any]], *, size: int
) -> list[list[dict[str, Any]]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    return [items[index : index + size] for index in range(0, len(items), size)]


def _write_audit_key(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _chatgpt_instructions(batch_count: int) -> str:
    return f"""# Manual ChatGPT frontier audit

This is supplementary evidence, not an API-equivalent evaluation.

Use the strongest reasoning setting available in ChatGPT (currently High for
Plus). Use a fresh Temporary Chat for each of the {batch_count} batch files so
memory and earlier judgments do not influence later batches.

For each batch:

1. Upload one `batch_XX.jsonl` file and `audit_result.schema.json`.
2. Paste this instruction:

   Act as a blinded legal-analysis evaluator. Evaluate every JSONL record in
   the uploaded batch independently. You are not told the experimental
   condition. Apply the supplied facts, reference issues, gold answer, and
   generated output. Return one JSONL record per input record, in the same
   order. Each output must contain `audit_id` plus every field required by
   `audit_result.schema.json`. Use integer scores 0–4. Identify the first
   causally responsible error and list downstream symptoms separately. Do not
   skip records. Do not include prose outside the JSONL.

3. Save the response as `response_XX.jsonl`.
4. Check that the number of returned rows equals the input row count.

Important limitations:

- ChatGPT UI does not expose an immutable model digest, seed, token accounting,
  or guaranteed JSON-schema enforcement.
- UI usage limits can vary.
- Record the displayed model/reasoning setting and the date for each batch.
- Do not upload `hidden_key.csv`; it contains condition labels.
"""
