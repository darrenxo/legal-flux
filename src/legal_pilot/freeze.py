from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .clients import OllamaClient
from .config import resolve_path
from .io_utils import latest_by_run_hash, read_jsonl, sha256_text
from .jobs import build_jobs
from .ledger import make_run_hash
from .runner import _preview_prompt, load_cases


def freeze_phase_two(config: dict[str, Any]) -> dict[str, Any]:
    run_dir = resolve_path(config, "runs_dir") / "smoke"
    history = read_jsonl(run_dir / "generations.jsonl")
    if not history:
        raise RuntimeError("No smoke records found. Run smoke first.")

    client = OllamaClient(
        config["model"]["base_url"], config["model"]["timeout_seconds"]
    )
    try:
        model_info = client.model_info(config["model"]["name"])
    finally:
        client.close()
    if not model_info:
        raise RuntimeError("Configured Ollama model is not available.")
    model_digest = model_info.get("digest", "unknown")

    cases = load_cases(config)
    jobs = build_jobs(cases, config, smoke=True)
    expected_hashes = {
        make_run_hash(
            dataset=job["case"].dataset,
            case_id=job["case"].case_id,
            variant_id=job["case"].variant_id,
            condition=job["condition"],
            prompt_hash=_preview_prompt(
                config, job["case"], job["condition"]
            )[1],
            model_digest=model_digest,
            seed=job["seed"],
            sample_index=job["sample_index"],
        )
        for job in jobs
    }
    current = {
        row["run_hash"]: row
        for row in latest_by_run_hash(history)
        if row.get("run_hash") in expected_hashes
    }
    missing = expected_hashes - current.keys()
    if missing:
        raise RuntimeError(
            f"Smoke run is missing {len(missing)} current prompt/model jobs."
        )
    rows = [current[run_hash] for run_hash in sorted(expected_hashes)]
    errors = [row for row in rows if row.get("status") != "ok"]
    if errors:
        raise RuntimeError(
            f"Current smoke candidate has {len(errors)} failed records."
        )

    prompt_dir = resolve_path(config, "prompts_dir")
    schema_dir = resolve_path(config, "schemas_dir")
    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "config": _public_config(config),
        "model": {
            "name": config["model"]["name"],
            "digest": model_digest,
            "size": model_info.get("size"),
        },
        "prompt_hashes": {
            path.name: sha256_text(path.read_text(encoding="utf-8"))
            for path in sorted(prompt_dir.glob("*.txt"))
        },
        "schema_hashes": {
            path.name: sha256_text(path.read_text(encoding="utf-8"))
            for path in sorted(schema_dir.glob("*.json"))
        },
        "case_ids": [
            {
                "dataset": case.dataset,
                "case_id": case.case_id,
                "variant_id": case.variant_id,
            }
            for case in cases
        ],
        "smoke_run_hashes": [row["run_hash"] for row in rows],
        "smoke_history_records": len(history),
    }
    path = resolve_path(config, "processed_dir") / "frozen_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"path": str(path), "records": len(rows), "model": manifest["model"]}


def _public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if not key.startswith("_")
    }
