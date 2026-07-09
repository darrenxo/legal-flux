from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Any

import httpx

from .clients import OllamaClient


def environment_check(config: dict[str, Any]) -> dict[str, Any]:
    packages = [
        "pydantic",
        "jsonschema",
        "pandas",
        "pyarrow",
        "datasets",
        "httpx",
        "tenacity",
        "numpy",
        "scipy",
        "sklearn",
        "matplotlib",
        "openai",
        "pytest",
    ]
    result: dict[str, Any] = {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "supported": _python_supported(sys.version_info[:2]),
        },
        "platform": platform.platform(),
        "packages": {
            package: bool(importlib.util.find_spec(package)) for package in packages
        },
        "nvidia_smi": None,
        "ollama": {"reachable": False, "model": None},
        "openai_key_present": bool(os.environ.get("OPENAI_API_KEY")),
    }
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        completed = subprocess.run(
            [
                nvidia,
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        result["nvidia_smi"] = completed.stdout.strip() or completed.stderr.strip()
    try:
        client = OllamaClient(
            config["model"]["base_url"], config["model"]["timeout_seconds"]
        )
        try:
            result["ollama"] = {
                "reachable": True,
                "model": client.model_info(config["model"]["name"]),
            }
        finally:
            client.close()
    except (httpx.HTTPError, OSError) as exc:
        result["ollama"]["error"] = str(exc)

    result["ready_for_smoke"] = bool(
        result["python"]["supported"]
        and all(result["packages"].values())
        and result["nvidia_smi"]
        and result["ollama"]["reachable"]
        and result["ollama"]["model"]
    )
    return result


def _python_supported(version: tuple[int, int]) -> bool:
    return (3, 11) <= tuple(version) < (3, 13)
