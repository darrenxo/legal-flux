from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .io_utils import canonical_json, sha256_text


def make_run_hash(**parts: Any) -> str:
    return sha256_text(canonical_json(parts))


class JsonlLedger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._completed = self._load_hashes()

    def _load_hashes(self) -> set[str]:
        if not self.path.exists():
            return set()
        hashes: set[str] = set()
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if value.get("run_hash") and value.get("status") in (None, "ok"):
                    hashes.add(value["run_hash"])
        return hashes

    def contains(self, run_hash: str) -> bool:
        return run_hash in self._completed

    def append(self, record: dict[str, Any]) -> None:
        run_hash = record.get("run_hash")
        if not run_hash:
            raise ValueError("Ledger records require run_hash.")
        with self._lock:
            if run_hash in self._completed:
                return
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            if record.get("status") in (None, "ok"):
                self._completed.add(run_hash)
