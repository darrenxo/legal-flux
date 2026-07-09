from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from legal_pilot.io_utils import sha256_text, write_jsonl
from legal_pilot.models import LegalFluxTemplate


DEFAULT_BASE = Path(
    "reports/legal_flux/template_distillation/chatgpt_batches/legal_flux_templates_final.jsonl"
)
DEFAULT_GAPS = Path(
    "reports/legal_flux/template_distillation/chatgpt_batches/legal_flux_gap_fill_proposed.jsonl"
)
DEFAULT_OUTPUT = Path(
    "reports/legal_flux/template_distillation/chatgpt_batches/legal_flux_templates_audited.jsonl"
)
DEFAULT_MANIFEST = Path(
    "reports/legal_flux/template_distillation/chatgpt_batches/legal_flux_templates_audited_manifest.json"
)

DROP_DUPLICATE_IDS = {
    "LF001": "Duplicate of LF022; LF022 keeps dependency consistency and double-counting checks.",
    "LF003": "Duplicate of LF002; LF002 is the more general long-fact filtering template.",
    "LF005": "Duplicate of LF008; LF008 has broader rule-to-element extraction coverage.",
    "LF052": "Duplicate of LF049; LF049 preserves preconditions and implied cooperation/prevention analysis.",
    "LF060": "Duplicate of LF059; LF059 preserves set-off and counterclaim treatment.",
    "LF071": "Duplicate of LF043; LF043 is the more general composite-remedy template.",
    "LF108": "Duplicate of LF106; LF106 already covers winding-up standing as a gateway.",
    "LF116": "Duplicate of LF113; LF113 more fully covers sanction procedure and stakeholder fairness.",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the audited LegalFlux template pool after coverage audit."
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--gaps", type=Path, default=DEFAULT_GAPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    base_templates = _load_templates(args.base)
    gap_templates = _load_templates(args.gaps)
    kept = [
        template
        for template in base_templates
        if template.template_id not in DROP_DUPLICATE_IDS
    ]
    combined = [*kept, *gap_templates]
    renumbered = []
    id_map: list[dict[str, str]] = []
    for index, template in enumerate(combined, start=1):
        new_id = f"LF{index:03d}"
        id_map.append(
            {
                "old_template_id": template.template_id,
                "new_template_id": new_id,
                "template_name": template.template_name,
            }
        )
        renumbered.append(template.model_copy(update={"template_id": new_id}))
    _validate_pool(renumbered)
    write_jsonl(
        args.output,
        [template.model_dump(mode="json") for template in renumbered],
    )
    output_text = args.output.read_text(encoding="utf-8")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base": str(args.base.resolve()),
        "gap_fill": str(args.gaps.resolve()),
        "output": str(args.output.resolve()),
        "base_template_count": len(base_templates),
        "dropped_duplicate_count": len(DROP_DUPLICATE_IDS),
        "gap_fill_count": len(gap_templates),
        "audited_template_count": len(renumbered),
        "dropped_duplicate_ids": DROP_DUPLICATE_IDS,
        "id_map": id_map,
        "output_sha256": sha256_text(output_text),
        "note": (
            "Template text was preserved from source files. The script removes "
            "audit-flagged duplicate records, appends validated gap-fill records, "
            "and renumbers template_id values sequentially."
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2))
    return 0


def _load_templates(path: Path) -> list[LegalFluxTemplate]:
    templates = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            templates.append(LegalFluxTemplate.model_validate(payload))
        except Exception as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return templates


def _validate_pool(templates: list[LegalFluxTemplate]) -> None:
    ids = [template.template_id for template in templates]
    duplicate_ids = [template_id for template_id, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Duplicate template IDs after audit: {duplicate_ids}")
    expected = [f"LF{index:03d}" for index in range(1, len(templates) + 1)]
    if ids != expected:
        raise ValueError("Audited template IDs are not sequential.")
    for template in templates:
        if len(template.knowledge_tags) < 2:
            raise ValueError(f"{template.template_id} has fewer than two tags.")
        if len(template.reasoning_flow) < 2:
            raise ValueError(f"{template.template_id} has fewer than two flow steps.")


if __name__ == "__main__":
    raise SystemExit(main())
