from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .bot import build_bot_plan, seed_templates
from .clients import OllamaClient
from .config import resolve_path
from .io_utils import canonical_json, latest_by_run_hash, read_jsonl, sha256_text
from .ledger import make_run_hash
from .runner import load_cases
from .semantic_setup import (
    embedding_runtime_info,
    resolve_project_file,
)


def bot_workflow_hash(config: dict[str, Any]) -> str:
    return sha256_text(canonical_json(bot_workflow_components(config)))


def bot_workflow_components(config: dict[str, Any]) -> dict[str, Any]:
    project_root = resolve_path(config, "prompts_dir").parent
    prompt_root = resolve_path(config, "prompts_dir") / "bot"
    schema_root = resolve_path(config, "schemas_dir") / "bot"
    shared_prompts = [
        resolve_path(config, "prompts_dir") / "direct.txt",
    ]
    shared_schemas = [
        resolve_path(config, "schemas_dir") / "direct_analysis.json",
        resolve_path(config, "schemas_dir") / "final_analysis.json",
    ]
    implementation_files = [
        project_root / "src" / "legal_pilot" / name
        for name in (
            "bot.py",
            "bot_runner.py",
            "embeddings.py",
            "frontier_profiles.py",
            "models.py",
            "semantic_setup.py",
        )
    ]
    components = {
        "bot_config": config["bot"],
        "model": {
            key: config["model"][key]
            for key in (
                "name",
                "context_length",
                "temperature",
                "seed",
                "analysis_max_tokens",
                "distill_max_tokens",
                "template_max_tokens",
            )
        },
        "prompts": {
            **{
                path.name: sha256_text(path.read_text(encoding="utf-8"))
                for path in shared_prompts
            },
            **{
                f"bot/{path.name}": sha256_text(
                    path.read_text(encoding="utf-8")
                )
                for path in sorted(prompt_root.glob("*.txt"))
            },
        },
        "schemas": {
            **{
                path.name: sha256_text(path.read_text(encoding="utf-8"))
                for path in shared_schemas
            },
            **{
                f"bot/{path.name}": sha256_text(
                    path.read_text(encoding="utf-8")
                )
                for path in sorted(schema_root.glob("*.json"))
            },
        },
        "implementation": {
            path.name: sha256_text(path.read_text(encoding="utf-8"))
            for path in implementation_files
        },
        "seed_templates": [
            item.model_dump(mode="json") for item in seed_templates()
        ],
    }
    profile_file = config["bot"].get("frontier_profiles_file")
    if profile_file:
        profile_path = resolve_project_file(profile_file)
        components["frontier_profiles"] = (
            sha256_text(profile_path.read_text(encoding="utf-8"))
            if profile_path.exists()
            else None
        )
    return components


def bot_main_plan_hash(
    config: dict[str, Any], cases=None
) -> str:
    source_cases = load_cases(config) if cases is None else cases
    plan = build_bot_plan(source_cases, config, smoke=False)
    payload = [
        {
            "case_id": item.case.case_id,
            "variant_id": item.case.variant_id,
            "gold_answer": item.case.gold_answer,
            "condition": item.condition,
            "phase": item.phase,
            "stream_index": item.stream_index,
        }
        for item in plan
    ]
    return sha256_text(canonical_json(payload))


def bot_run_hash(
    *,
    case_id: str,
    condition: str,
    phase: str,
    stream_index: int,
    model_digest: str,
    embedding_digest: str = "builtin-tfidf",
    workflow_hash: str,
    seed: int,
) -> str:
    return make_run_hash(
        dataset="legalhk",
        case_id=case_id,
        variant_id="original",
        condition=condition,
        phase=phase,
        stream_index=stream_index,
        model_digest=model_digest,
        embedding_digest=embedding_digest,
        workflow_hash=workflow_hash,
        seed=seed,
    )


def freeze_bot_phase(config: dict[str, Any]) -> dict[str, Any]:
    run_dir = resolve_path(config, "runs_dir") / "smoke"
    records = latest_by_run_hash(read_jsonl(run_dir / "generations.jsonl"))
    if not records:
        raise RuntimeError("No BoT smoke records found. Run bot-smoke first.")
    client = OllamaClient(
        config["model"]["base_url"], config["model"]["timeout_seconds"]
    )
    try:
        model_info = client.model_info(config["model"]["name"])
    finally:
        client.close()
    if not model_info:
        raise RuntimeError("Configured Ollama model is not available.")
    digest = model_info.get("digest", "unknown")
    embedding = embedding_runtime_info(config)
    embedding_digest = embedding["digest"]
    workflow_hash = bot_workflow_hash(config)
    plan = build_bot_plan(load_cases(config), config, smoke=True)
    expected = {
        bot_run_hash(
            case_id=item.case.case_id,
            condition=item.condition,
            phase=item.phase,
            stream_index=item.stream_index,
            model_digest=digest,
            embedding_digest=embedding_digest,
            workflow_hash=workflow_hash,
            seed=config["model"]["seed"],
        )
        for item in plan
    }
    current = {
        row["run_hash"]: row
        for row in records
        if row.get("run_hash") in expected
    }
    missing = expected - current.keys()
    failures = [
        row for row in current.values() if row.get("status") != "ok"
    ]
    if missing or failures:
        raise RuntimeError(
            "BoT smoke is not freeze-ready: "
            f"{len(missing)} missing and {len(failures)} failed records."
        )
    split = build_bot_plan(load_cases(config), config, smoke=False)
    main_plan_hash = bot_main_plan_hash(config)
    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": config["model"]["name"],
            "digest": digest,
            "size": model_info.get("size"),
        },
        "workflow_hash": workflow_hash,
        "embedding": embedding,
        "main_plan_hash": main_plan_hash,
        "smoke_run_hashes": sorted(expected),
        "main_stream": [
            {
                "case_id": item.case.case_id,
                "condition": item.condition,
                "phase": item.phase,
                "stream_index": item.stream_index,
            }
            for item in split
        ],
    }
    path = _freeze_manifest_path(config)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(path),
        "records": len(expected),
        "model": manifest["model"],
        "workflow_hash": workflow_hash,
        "embedding": embedding,
        "main_plan_hash": main_plan_hash,
    }


def assert_bot_frozen(
    config: dict[str, Any],
    *,
    model_digest: str,
    embedding_digest: str = "builtin-tfidf",
    workflow_hash: str,
) -> None:
    path = _freeze_manifest_path(config)
    if not path.exists():
        raise RuntimeError("BoT phase 2 is not frozen. Run bot-freeze first.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("model", {}).get("digest") != model_digest:
        raise RuntimeError("Ollama model digest differs from the BoT freeze.")
    if manifest.get("workflow_hash") != workflow_hash:
        raise RuntimeError("BoT prompts, schemas, seeds, or settings changed.")
    if (
        manifest.get("embedding", {}).get("digest", "builtin-tfidf")
        != embedding_digest
    ):
        raise RuntimeError(
            "Embedding model digest differs from the BoT freeze."
        )
    if manifest.get("main_plan_hash") != bot_main_plan_hash(config):
        raise RuntimeError("The frozen LegalHK case stream has changed.")


def _freeze_manifest_path(config: dict[str, Any]):
    configured = config["bot"].get("freeze_manifest_file")
    if configured:
        return resolve_project_file(configured)
    return resolve_path(config, "processed_dir") / "bot_frozen_manifest.json"
