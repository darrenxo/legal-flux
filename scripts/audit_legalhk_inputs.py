from __future__ import annotations

import argparse
import json
from pathlib import Path

from legal_pilot.clients import OllamaClient
from legal_pilot.input_leakage_audit import (
    InputLeakageAudit,
    render_input_leakage_prompt,
)
from legal_pilot.io_utils import read_jsonl
from legal_pilot.io_utils import sha256_text
from legal_pilot.ledger import JsonlLedger, make_run_hash


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--seed", type=int, default=20260619)
    args = parser.parse_args()

    rows = [
        row
        for row in read_jsonl(Path(args.input))
        if row.get("selection_split") == args.split
    ]
    ledger = JsonlLedger(Path(args.output))
    client = OllamaClient(args.base_url, timeout_seconds=1200)
    completed = 0
    skipped = 0
    try:
        model_info = client.model_info(args.model)
        if not model_info:
            raise RuntimeError(f"Model {args.model!r} is not installed.")
        digest = model_info["digest"]
        schema = InputLeakageAudit.model_json_schema()
        for row in rows:
            prompt = render_input_leakage_prompt(
                claim=row["claim"], facts=row["facts"]
            )
            prompt_hash = sha256_text(prompt)
            run_hash = make_run_hash(
                task="input_leakage_audit_v2",
                case_id=row["case_id"],
                model_digest=digest,
                prompt_hash=prompt_hash,
                seed=args.seed,
            )
            if ledger.contains(run_hash):
                skipped += 1
                continue
            response = client.generate(
                model=args.model,
                prompt=prompt,
                schema=schema,
                temperature=0.0,
                seed=args.seed,
                context_length=16384,
                max_tokens=3200,
                think="medium",
            )
            audit = InputLeakageAudit.model_validate(response.parsed)
            ledger.append(
                {
                    "run_hash": run_hash,
                    "status": "ok",
                    "case_id": row["case_id"],
                    "selection_split": args.split,
                    "audit_model": args.model,
                    "audit_model_digest": digest,
                    "audit": audit.model_dump(mode="json"),
                    "elapsed_seconds": response.elapsed_seconds,
                    "prompt_tokens": response.prompt_tokens,
                    "output_tokens": response.output_tokens,
                    "thinking_characters": response.metadata.get(
                        "thinking_characters"
                    ),
                    "thinking_sha256": response.metadata.get(
                        "thinking_sha256"
                    ),
                }
            )
            completed += 1
    finally:
        client.close()
    print(
        json.dumps(
            {
                "cases": len(rows),
                "completed": completed,
                "skipped": skipped,
                "output": str(Path(args.output).resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
