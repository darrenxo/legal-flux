from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def discover_project_root() -> Path:
    configured = os.environ.get("LEGAL_FLUX_ROOT")
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path.cwd(),
        Path(__file__).resolve().parents[2],
    ]
    for candidate in candidates:
        if candidate and (candidate / "configs" / "legal_flux.yaml").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate configs/legal_flux.yaml. Run the CLI from the "
        "legal_flux directory or set LEGAL_FLUX_ROOT."
    )


PROJECT_ROOT = discover_project_root()


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "legal_flux.yaml"
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    data = _load_config_file(config_path, seen=set())
    _apply_runtime_environment(data)
    data["_config_path"] = str(config_path.resolve())
    data["_project_root"] = str(PROJECT_ROOT)
    return data


def resolve_path(config: dict[str, Any], key: str) -> Path:
    root = Path(config["_project_root"])
    return (root / config["paths"][key]).resolve()


def resolve_project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (Path(config["_project_root"]) / path).resolve()


def _load_config_file(path: Path, *, seen: set[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Cyclic config inheritance involving {resolved}.")
    seen.add(resolved)
    data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    parent = data.pop("extends", None)
    if parent is None:
        return data
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    base = _load_config_file(parent_path, seen=seen)
    return _deep_merge(base, data)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_runtime_environment(config: dict[str, Any]) -> None:
    work_root = os.environ.get("LEGAL_FLUX_WORK_ROOT")
    if work_root:
        work_path = Path(work_root).expanduser()
        config.setdefault("paths", {}).update(
            {
                "processed_dir": str(
                    work_path / "data" / "processed" / "legal_flux"
                ),
                "runs_dir": str(work_path / "runs" / "legal_flux"),
                "reports_dir": str(work_path / "reports" / "legal_flux"),
            }
        )
        config.setdefault("training", {}).setdefault("template_sft", {})[
            "output_dir"
        ] = str(
            work_path
            / "runs"
            / "legal_flux"
            / "training"
            / "template_structure_sft"
        )
        config.setdefault("training", {}).setdefault("trajectory_dpo", {})[
            "output_dir"
        ] = str(
            work_path
            / "runs"
            / "legal_flux"
            / "training"
            / "trajectory_dpo"
        )
    model = config.setdefault("model", {})
    if os.environ.get("LEGAL_FLUX_MODEL_BASE_URL"):
        model["base_url"] = os.environ["LEGAL_FLUX_MODEL_BASE_URL"]
    if os.environ.get("LEGAL_FLUX_MODEL_NAME"):
        model["name"] = os.environ["LEGAL_FLUX_MODEL_NAME"]
    if os.environ.get("LEGAL_FLUX_MODEL_CONCURRENCY"):
        model["concurrency"] = int(os.environ["LEGAL_FLUX_MODEL_CONCURRENCY"])
    if os.environ.get("LEGAL_FLUX_VLLM_VERSION"):
        model["inference_runtime"] = "vllm"
        model["inference_runtime_version"] = os.environ[
            "LEGAL_FLUX_VLLM_VERSION"
        ]
    if os.environ.get("LEGAL_FLUX_RETRIEVAL_DEVICE"):
        config.setdefault("xsim", {})["device"] = os.environ[
            "LEGAL_FLUX_RETRIEVAL_DEVICE"
        ]
    legal_flux = config.setdefault("legal_flux", {})
    for role in ("planner", "executor", "reviewer"):
        environment_name = f"LEGAL_FLUX_{role.upper()}_MODEL"
        if os.environ.get(environment_name):
            legal_flux[f"{role}_model"] = os.environ[environment_name]
    if os.environ.get("LEGAL_FLUX_SOURCE_CHECKPOINT"):
        legal_flux["source_checkpoint"] = os.environ[
            "LEGAL_FLUX_SOURCE_CHECKPOINT"
        ]
    if os.environ.get("LEGAL_FLUX_DPO_SFT_CHECKPOINT"):
        config.setdefault("dpo", {})["source_checkpoint"] = os.environ[
            "LEGAL_FLUX_DPO_SFT_CHECKPOINT"
        ]
