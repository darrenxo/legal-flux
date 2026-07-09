from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def balanced_select(
    rows: list[dict[str, Any]], *, limit: int, seed: int
) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row.get("status") == "ok"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        groups[(row.get("dataset", ""), row.get("condition", ""))].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)

    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    while len(selected) < limit and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop())
    return selected

