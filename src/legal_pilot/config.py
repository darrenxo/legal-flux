from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def discover_project_root() -> Path:
    configured = os.environ.get("LEGAL_PILOT_ROOT")
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path.cwd(),
        Path(__file__).resolve().parents[2],
    ]
    for candidate in candidates:
        if candidate and (candidate / "configs" / "pilot.yaml").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate configs/pilot.yaml. Run the CLI from the "
        "legal_case_state_pilot directory or set LEGAL_PILOT_ROOT."
    )


PROJECT_ROOT = discover_project_root()


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "pilot.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["_config_path"] = str(config_path.resolve())
    data["_project_root"] = str(PROJECT_ROOT)
    return data


def resolve_path(config: dict[str, Any], key: str) -> Path:
    root = Path(config["_project_root"])
    return (root / config["paths"][key]).resolve()
