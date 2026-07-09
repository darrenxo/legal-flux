from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def latest_by_run_hash(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    unhashed: list[dict[str, Any]] = []
    for row in rows:
        run_hash = row.get("run_hash")
        if run_hash:
            latest[run_hash] = row
        else:
            unhashed.append(row)
    return [*latest.values(), *unhashed]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def split_facts(text: str) -> dict[str, str]:
    lines = [line.strip(" \t-*") for line in text.splitlines() if line.strip()]
    if lines and lines[0].lower() in {"facts:", "facts"}:
        lines = lines[1:]
    return {f"F{index}": line for index, line in enumerate(lines, start=1)}


def normalize_answer(value: str | None) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).strip().lower().split())
    for prefix in ("final answer:", "answer:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text.strip(" .")
