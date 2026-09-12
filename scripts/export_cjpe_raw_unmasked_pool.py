from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from legal_pilot.io_utils import atomic_write_json, read_jsonl, sha256_text
from legal_pilot.legal_flux import template_pool_hash, write_template_pool
from legal_pilot.models import LegalFluxTemplate


EXECUTABLE_FIELDS = (
    "template_name",
    "knowledge_tags",
    "description",
    "application_scenario",
    "reasoning_flow",
    "example_application",
)


def reconstruct_raw_unmasked_pool(input_root: Path) -> list[LegalFluxTemplate]:
    final_rows = read_jsonl(input_root / "legal_flux_templates_gemini_final.jsonl")
    final_lineage = _read_json(
        input_root / "legal_flux_templates_gemini_final_lineage.json"
    )
    initial_drafts = _read_json(
        input_root / "legal_flux_templates_gemini_merged_raw.txt"
    )["templates"]
    adjudication_merges = _read_json(
        input_root / "legal_flux_gap_adjudication_raw.txt"
    )["merged_templates"]

    gap_rows: dict[str, dict[str, Any]] = {}
    for path in sorted((input_root / "04_gap_audits").glob("*_gap_candidates.jsonl")):
        for row in read_jsonl(path):
            candidate_id = str(row["candidate_id"])
            if candidate_id in gap_rows:
                raise ValueError(f"Duplicate gap candidate ID: {candidate_id}")
            gap_rows[candidate_id] = row

    merge_by_sources = {
        tuple(str(value) for value in row["source_candidate_ids"]): row
        for row in adjudication_merges
    }
    templates: list[LegalFluxTemplate] = []
    for masked_row in final_rows:
        template_id = str(masked_row["template_id"])
        source_ids = [str(value) for value in final_lineage[template_id]]
        if source_ids[0].startswith("LF"):
            source_index = int(source_ids[0][2:]) - 1
            source_row = initial_drafts[source_index]
        elif len(source_ids) == 1:
            source_row = gap_rows[source_ids[0]]
        else:
            try:
                source_row = merge_by_sources[tuple(source_ids)]
            except KeyError as exc:
                raise ValueError(
                    f"No raw adjudication merge matches {template_id}: {source_ids}"
                ) from exc
        templates.append(
            LegalFluxTemplate(
                template_id=template_id,
                **{field: source_row[field] for field in EXECUTABLE_FIELDS},
            )
        )
    return templates


def export_raw_unmasked_pool(
    input_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    templates = reconstruct_raw_unmasked_pool(input_root)
    write_template_pool(output_path, templates)
    manifest = {
        "variant": "cjpe_gemini31_pro_raw_unmasked",
        "template_count": len(templates),
        "source_final_pool": _portable_path(
            input_root / "legal_flux_templates_gemini_final.jsonl"
        ),
        "source_final_lineage": _portable_path(
            input_root / "legal_flux_templates_gemini_final_lineage.json"
        ),
        "output_path": _portable_path(output_path),
        "output_sha256": sha256_text(output_path.read_text(encoding="utf-8")),
        "template_pool_hash": template_pool_hash(templates),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    atomic_write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct the CJPE Gemini pool before numeric sanitization."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "reports/legal_flux/template_distillation/cjpe_gemini31_pro_api"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("templates/legal_flux_templates_cjpe_raw_unmasked.jsonl"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = export_raw_unmasked_pool(args.input_root.resolve(), args.output.resolve())
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
