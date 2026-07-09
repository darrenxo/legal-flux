from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_INPUT_DIR = Path(
    "reports/legal_flux/template_distillation/chatgpt_batches/03_candidate_templates"
)
DEFAULT_OUTPUT = Path(
    "reports/legal_flux/template_distillation/chatgpt_batches/all_candidate_templates.jsonl"
)
DEFAULT_MANIFEST = Path(
    "reports/legal_flux/template_distillation/chatgpt_batches/candidate_concat_manifest.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Byte-preserving concatenation of ChatGPT candidate-template JSONL files."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--glob", default="*_candidates.jsonl")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output = args.output.resolve()
    manifest_path = args.manifest.resolve()
    sources = [
        path.resolve()
        for path in sorted(input_dir.glob(args.glob), key=lambda item: item.name)
        if path.resolve() != output
    ]
    if not sources:
        raise SystemExit(f"No candidate files found in {input_dir} matching {args.glob!r}.")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with output.open("wb") as destination:
        for index, source in enumerate(sources):
            data = source.read_bytes()
            if index:
                destination.write(b"\n")
            destination.write(data)
            ended_with_newline = data.endswith((b"\n", b"\r\n"))
            if not ended_with_newline:
                destination.write(b"\n")
            records.append(
                {
                    "path": str(source),
                    "bytes": len(data),
                    "sha256": _sha256(data),
                    "nonempty_line_count": _nonempty_line_count(data),
                    "ended_with_newline": ended_with_newline,
                }
            )

    output_data = output.read_bytes()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "byte_concatenation_no_json_rewrite",
        "input_dir": str(input_dir),
        "glob": args.glob,
        "output": str(output),
        "source_file_count": len(records),
        "source_nonempty_line_count": sum(
            record["nonempty_line_count"] for record in records
        ),
        "output_bytes": len(output_data),
        "output_sha256": _sha256(output_data),
        "sources": records,
        "note": (
            "Source file bytes are copied as-is. The script only inserts newline "
            "separators between files and after files that lack a trailing newline."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2))
    return 0


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _nonempty_line_count(data: bytes) -> int:
    return sum(1 for line in data.splitlines() if line.strip())


if __name__ == "__main__":
    raise SystemExit(main())
