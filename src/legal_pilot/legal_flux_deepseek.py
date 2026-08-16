from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from json_repair import repair_json

from .io_utils import read_jsonl, sha256_text, write_jsonl
from .legal_flux import resolve_project_file, sanitize_flux_template, validate_template_pool
from .models import LegalFluxTemplate


class TemplateApiClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        ...


class DeepSeekTemplateClient:
    def __init__(self, config: dict[str, Any]):
        deepseek_config = config.get("deepseek", {})
        api_key_env = str(deepseek_config.get("api_key_env", "DEEPSEEK_API_KEY"))
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set.")
        from openai import OpenAI

        self.model = str(deepseek_config.get("model", "deepseek-v4-pro"))
        self.temperature = float(deepseek_config.get("temperature", 0.2))
        self.max_tokens = int(deepseek_config.get("max_tokens", 24000))
        self.reasoning_effort = deepseek_config.get("reasoning_effort")
        self.thinking = deepseek_config.get("thinking", "disabled")
        self.client = OpenAI(
            api_key=api_key,
            base_url=str(deepseek_config.get("base_url", "https://api.deepseek.com")),
            timeout=float(deepseek_config.get("timeout_seconds", 600)),
        )

    def complete(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        extra_body: dict[str, Any] = {}
        if self.thinking in {"enabled", "disabled"}:
            extra_body["thinking"] = {"type": self.thinking}
        if extra_body:
            request["extra_body"] = extra_body
        if self.reasoning_effort and self.thinking != "disabled":
            request["reasoning_effort"] = self.reasoning_effort
        start = time.perf_counter()
        response = self.client.chat.completions.create(**request)
        usage = getattr(response, "usage", None)
        message = response.choices[0].message
        return {
            "content": message.content or "",
            "metadata": {
                "response_id": getattr(response, "id", None),
                "model": getattr(response, "model", self.model),
                "elapsed_seconds": time.perf_counter() - start,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            },
        }


def run_deepseek_template_workflow(
    config: dict[str, Any],
    *,
    stage: str = "candidates",
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
    client: TemplateApiClient | None = None,
) -> dict[str, Any]:
    normalized_stage = stage.replace("-", "_")
    if normalized_stage == "all":
        result: dict[str, Any] = {
            "stage": "all",
            "candidates": generate_deepseek_template_candidates(
                config, limit=limit, force=force, dry_run=dry_run, client=client
            ),
        }
        if dry_run:
            result["merge"] = _dry_stage_or_blocked(
                lambda: generate_deepseek_template_merge(
                    config, force=force, dry_run=True, client=client
                ),
                stage="merge",
            )
            result["audit"] = _dry_stage_or_blocked(
                lambda: generate_deepseek_template_audit(
                    config, force=force, dry_run=True, client=client
                ),
                stage="audit",
            )
            return result
        result["merge"] = generate_deepseek_template_merge(
            config, force=force, dry_run=dry_run, client=client
        )
        result["audit"] = generate_deepseek_template_audit(
            config, force=force, dry_run=dry_run, client=client
        )
        return result
    if normalized_stage == "candidates":
        return generate_deepseek_template_candidates(
            config, limit=limit, force=force, dry_run=dry_run, client=client
        )
    if normalized_stage == "merge":
        return generate_deepseek_template_merge(
            config, force=force, dry_run=dry_run, client=client
        )
    if normalized_stage == "audit":
        return generate_deepseek_template_audit(
            config, force=force, dry_run=dry_run, client=client
        )
    raise ValueError("stage must be one of: candidates, merge, audit, all.")


def _dry_stage_or_blocked(call: Any, *, stage: str) -> dict[str, Any]:
    try:
        return call()
    except RuntimeError as exc:
        return {"stage": stage, "dry_run": True, "blocked": str(exc)}


def generate_deepseek_template_candidates(
    config: dict[str, Any],
    *,
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
    client: TemplateApiClient | None = None,
) -> dict[str, Any]:
    batch_root = _batch_root(config)
    prompt = _read_text(batch_root / "prompts" / "01_generate_candidate_templates.md")
    schema_text = _read_text(batch_root / "legal_flux_template.schema.json")
    manifest = _batch_manifest(batch_root)
    batches = manifest.get("batches", [])
    if limit is not None:
        batches = batches[:limit]
    output_dir = _deepseek_root(config) / "03_candidate_templates"
    raw_dir = output_dir / "raw"
    planned = [
        _candidate_plan_row(batch_root, batch, prompt, schema_text)
        for batch in batches
    ]
    if dry_run:
        return {
            "stage": "candidates",
            "dry_run": True,
            "planned_calls": len(planned),
            "output_dir": str(output_dir),
            "calls": planned,
        }
    client = client or DeepSeekTemplateClient(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for batch in batches:
        batch_id = str(batch["batch_id"])
        output_path = output_dir / f"{batch_id}_candidates.jsonl"
        raw_path = raw_dir / f"{batch_id}_raw.txt"
        if output_path.exists() and not force:
            records.append(
                {
                    "batch_id": batch_id,
                    "status": "skipped_existing",
                    "output_path": str(output_path),
                    "template_count": len(read_jsonl(output_path)),
                }
            )
            continue
        messages = _candidate_messages(batch_root, batch, prompt, schema_text)
        response = client.complete(messages)
        raw_text = str(response["content"])
        raw_path.write_text(raw_text, encoding="utf-8")
        templates = _coerce_templates(
            _extract_json_payloads(raw_text),
            source=f"DeepSeek candidate response for {batch_id}",
            sanitize=False,
        )
        write_jsonl(output_path, [template.model_dump(mode="json") for template in templates])
        records.append(
            {
                "batch_id": batch_id,
                "status": "ok",
                "output_path": str(output_path),
                "raw_path": str(raw_path),
                "template_count": len(templates),
                "metadata": response.get("metadata", {}),
                "prompt_sha256": sha256_text(messages[-1]["content"]),
                "output_sha256": sha256_text(output_path.read_text(encoding="utf-8")),
            }
        )
    return _write_stage_manifest(
        _deepseek_root(config),
        "candidates",
        {
            "stage": "candidates",
            "output_dir": str(output_dir),
            "batch_count": len(batches),
            "records": records,
        },
    )


def generate_deepseek_template_merge(
    config: dict[str, Any],
    *,
    force: bool = False,
    dry_run: bool = False,
    client: TemplateApiClient | None = None,
) -> dict[str, Any]:
    batch_root = _batch_root(config)
    output_root = _deepseek_root(config)
    candidate_dir = output_root / "03_candidate_templates"
    output_path = output_root / "legal_flux_templates_deepseek_merged.jsonl"
    raw_path = output_root / "legal_flux_templates_deepseek_merged_raw.txt"
    candidate_paths = sorted(candidate_dir.glob("*_candidates.jsonl"))
    if not candidate_paths:
        raise RuntimeError("No DeepSeek candidate files found. Run candidates stage first.")
    prompt_text = _merge_prompt_text(batch_root, candidate_paths)
    if dry_run:
        return {
            "stage": "merge",
            "dry_run": True,
            "candidate_files": len(candidate_paths),
            "prompt_characters": len(prompt_text),
            "output_path": str(output_path),
        }
    if output_path.exists() and not force:
        return {
            "stage": "merge",
            "status": "skipped_existing",
            "output_path": str(output_path),
            "template_count": len(read_jsonl(output_path)),
        }
    client = client or DeepSeekTemplateClient(config)
    output_root.mkdir(parents=True, exist_ok=True)
    response = client.complete(_messages(prompt_text))
    raw_text = str(response["content"])
    raw_path.write_text(raw_text, encoding="utf-8")
    templates = _coerce_templates(
        _extract_json_payloads(raw_text),
        source="DeepSeek merge response",
        sanitize=True,
    )
    validate_template_pool(templates)
    write_jsonl(output_path, [template.model_dump(mode="json") for template in templates])
    return _write_stage_manifest(
        output_root,
        "merge",
        {
            "stage": "merge",
            "status": "ok",
            "candidate_files": len(candidate_paths),
            "template_count": len(templates),
            "output_path": str(output_path),
            "raw_path": str(raw_path),
            "metadata": response.get("metadata", {}),
            "prompt_sha256": sha256_text(prompt_text),
            "output_sha256": sha256_text(output_path.read_text(encoding="utf-8")),
        },
    )


def generate_deepseek_template_audit(
    config: dict[str, Any],
    *,
    force: bool = False,
    dry_run: bool = False,
    client: TemplateApiClient | None = None,
) -> dict[str, Any]:
    batch_root = _batch_root(config)
    output_root = _deepseek_root(config)
    pool_path = output_root / "legal_flux_templates_deepseek_merged.jsonl"
    raw_path = output_root / "legal_flux_coverage_audit_deepseek.md"
    gap_path = output_root / "legal_flux_gap_fill_deepseek.jsonl"
    if not pool_path.exists():
        raise RuntimeError("No merged DeepSeek template pool found. Run merge stage first.")
    prompt_text = _audit_prompt_text(batch_root, pool_path)
    if dry_run:
        return {
            "stage": "audit",
            "dry_run": True,
            "prompt_characters": len(prompt_text),
            "audit_path": str(raw_path),
            "gap_fill_path": str(gap_path),
        }
    if raw_path.exists() and not force:
        return {
            "stage": "audit",
            "status": "skipped_existing",
            "audit_path": str(raw_path),
            "gap_fill_path": str(gap_path) if gap_path.exists() else None,
        }
    client = client or DeepSeekTemplateClient(config)
    response = client.complete(_messages(prompt_text))
    raw_text = str(response["content"])
    raw_path.write_text(raw_text, encoding="utf-8")
    gap_templates = _coerce_templates(
        _extract_json_payloads(raw_text),
        source="DeepSeek audit gap-fill response",
        sanitize=True,
        allow_empty=True,
    )
    write_jsonl(gap_path, [template.model_dump(mode="json") for template in gap_templates])
    return _write_stage_manifest(
        output_root,
        "audit",
        {
            "stage": "audit",
            "status": "ok",
            "audit_path": str(raw_path),
            "gap_fill_path": str(gap_path),
            "gap_fill_count": len(gap_templates),
            "metadata": response.get("metadata", {}),
            "prompt_sha256": sha256_text(prompt_text),
            "audit_sha256": sha256_text(raw_text),
        },
    )


def _batch_root(config: dict[str, Any]) -> Path:
    return resolve_project_file(
        config["legal_flux"].get(
            "chatgpt_batch_dir",
            "reports/legal_flux/template_distillation/chatgpt_batches",
        )
    )


def _deepseek_root(config: dict[str, Any]) -> Path:
    return resolve_project_file(
        config["legal_flux"].get(
            "deepseek_template_dir",
            "reports/legal_flux/template_distillation/deepseek_api",
        )
    )


def _batch_manifest(batch_root: Path) -> dict[str, Any]:
    path = batch_root / "batch_manifest.json"
    if not path.exists():
        raise RuntimeError("Batch manifest not found. Run flux-export-chatgpt-batches first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_plan_row(
    batch_root: Path,
    batch: dict[str, Any],
    prompt: str,
    schema_text: str,
) -> dict[str, Any]:
    batch_text = _read_text(_batch_file_path(batch, batch_root))
    content = _candidate_user_content(batch, prompt, schema_text, batch_text)
    return {
        "batch_id": batch["batch_id"],
        "kind": batch.get("kind"),
        "case_count": batch.get("case_count"),
        "prompt_characters": len(content),
    }


def _candidate_messages(
    batch_root: Path,
    batch: dict[str, Any],
    prompt: str,
    schema_text: str,
) -> list[dict[str, str]]:
    batch_text = _read_text(_batch_file_path(batch, batch_root))
    return _messages(_candidate_user_content(batch, prompt, schema_text, batch_text))


def _batch_file_path(batch: dict[str, Any], batch_root: Path) -> Path:
    path = Path(batch["path"])
    if path.is_absolute() and path.exists():
        return path
    if not path.is_absolute():
        candidate = batch_root / path
        if candidate.exists():
            return candidate
    subdir = {
        "homogeneous": "01_homogeneous_batches",
        "mixed": "02_mixed_contrast_batches",
        "semantic_family": "01_semantic_family_batches",
    }.get(str(batch.get("kind", "")))
    if subdir:
        candidate = batch_root / subdir / path.name
        if candidate.exists():
            return candidate
    matches = sorted(batch_root.rglob(path.name))
    return matches[0] if matches else path


def _candidate_user_content(
    batch: dict[str, Any],
    prompt: str,
    schema_text: str,
    batch_text: str,
) -> str:
    return f"""{prompt}

BATCH_ID:
{batch["batch_id"]}

BATCH LABEL:
{batch.get("label", "")}

TEMPLATE SCHEMA:
```json
{schema_text}
```

BATCH JSONL:
```jsonl
{batch_text}
```
"""


def _merge_prompt_text(batch_root: Path, candidate_paths: list[Path]) -> str:
    prompt = _read_text(batch_root / "prompts" / "02_merge_deduplicate_templates.md")
    schema_text = _read_text(batch_root / "legal_flux_template.schema.json")
    manifest_text = _read_text(batch_root / "batch_manifest.json")
    coverage_text = _read_text(batch_root / "coverage_summary.json")
    candidate_sections = []
    for path in candidate_paths:
        candidate_sections.append(
            f"## {path.name}\n\n```jsonl\n{_read_text(path)}\n```"
        )
    return f"""{prompt}

TEMPLATE SCHEMA:
```json
{schema_text}
```

BATCH MANIFEST:
```json
{manifest_text}
```

COVERAGE SUMMARY:
```json
{coverage_text}
```

CANDIDATE TEMPLATE FILES:
{chr(10).join(candidate_sections)}
"""


def _audit_prompt_text(batch_root: Path, pool_path: Path) -> str:
    prompt = _read_text(batch_root / "prompts" / "03_coverage_audit_and_gap_fill.md")
    schema_text = _read_text(batch_root / "legal_flux_template.schema.json")
    manifest_text = _read_text(batch_root / "batch_manifest.json")
    pool_text = _read_text(pool_path)
    return f"""{prompt}

TEMPLATE SCHEMA:
```json
{schema_text}
```

BATCH MANIFEST:
```json
{manifest_text}
```

FINAL TEMPLATE POOL:
```jsonl
{pool_text}
```
"""


def _messages(content: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You create reusable high-level LegalFlux reasoning templates. "
                "Return the exact requested artifact only; do not include hidden "
                "chain-of-thought."
            ),
        },
        {"role": "user", "content": content},
    ]


def _extract_json_payloads(text: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    cleaned_lines = [
        line
        for line in text.splitlines()
        if not line.strip().startswith("```")
    ]
    cleaned = "\n".join(cleaned_lines).strip()
    try:
        parsed = json.loads(cleaned)
        return _payloads_from_value(parsed)
    except Exception:
        pass
    try:
        parsed = repair_json(cleaned, return_objects=True)
        payloads.extend(_payloads_from_value(parsed))
    except Exception:
        pass
    for line in cleaned.splitlines():
        stripped = line.strip().rstrip(",")
        if not stripped.startswith("{"):
            continue
        try:
            payloads.extend(_payloads_from_value(json.loads(stripped)))
            continue
        except Exception:
            pass
        try:
            payloads.extend(_payloads_from_value(repair_json(stripped, return_objects=True)))
        except Exception:
            continue
    return _dedupe_payloads(payloads)


def _payloads_from_value(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("templates"), list):
            return [item for item in value["templates"] if isinstance(item, dict)]
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _dedupe_payloads(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for payload in payloads:
        key = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append(payload)
    return unique


def _coerce_templates(
    payloads: list[dict[str, Any]],
    *,
    source: str,
    sanitize: bool,
    allow_empty: bool = False,
) -> list[LegalFluxTemplate]:
    templates: list[LegalFluxTemplate] = []
    errors: list[str] = []
    for index, payload in enumerate(payloads, start=1):
        try:
            template = LegalFluxTemplate.model_validate(payload)
            templates.append(sanitize_flux_template(template) if sanitize else template)
        except Exception as exc:
            errors.append(f"record {index}: {exc}")
    if not templates and not allow_empty:
        details = "; ".join(errors[:5]) if errors else "no JSON template objects found"
        raise ValueError(f"{source} did not contain valid LegalFlux templates: {details}")
    return templates


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def _write_stage_manifest(
    output_root: Path,
    stage: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    path = output_root / f"{stage}_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest_path": str(path)}
