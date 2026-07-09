from __future__ import annotations

import argparse
import json
import sys

from .audit import (
    export_chatgpt_audit,
    run_audit,
    run_local_audit,
    select_audit,
)
from .config import load_config
from .data_prep import prepare_datasets
from .environment import environment_check
from .evaluation import score_run
from .freeze import freeze_phase_two
from .reporting import build_report
from .runner import run_generation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Legal case-state diagnostic pilot")
    parser.add_argument("--config", default=None, help="Path to pilot YAML config.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("env-check")
    subparsers.add_parser("prepare")
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--dry-run", action="store_true")
    subparsers.add_parser("freeze")
    generate = subparsers.add_parser("generate")
    generate.add_argument("--dry-run", action="store_true")
    score = subparsers.add_parser("score")
    score.add_argument("--smoke", action="store_true")
    select = subparsers.add_parser("select-audit")
    select.add_argument("--smoke", action="store_true")
    audit = subparsers.add_parser("audit")
    audit.add_argument("--smoke", action="store_true")
    local_audit = subparsers.add_parser("audit-local")
    local_audit.add_argument("--smoke", action="store_true")
    local_audit.add_argument("--limit", type=int, default=None)
    export_audit = subparsers.add_parser("export-chatgpt-audit")
    export_audit.add_argument("--batch-size", type=int, default=10)
    subparsers.add_parser("report")
    bot_smoke = subparsers.add_parser("bot-smoke")
    bot_smoke.add_argument("--dry-run", action="store_true")
    subparsers.add_parser("bot-freeze")
    bot_generate = subparsers.add_parser("bot-generate")
    bot_generate.add_argument("--dry-run", action="store_true")
    bot_score = subparsers.add_parser("bot-score")
    bot_score.add_argument("--smoke", action="store_true")
    subparsers.add_parser("bot-report")
    subparsers.add_parser("bot-embedding-check")
    subparsers.add_parser("bot-export-frontier")
    import_frontier = subparsers.add_parser("bot-import-frontier")
    import_frontier.add_argument("--input", required=True)
    subparsers.add_parser("flux-prepare")
    subparsers.add_parser("flux-export-templates")
    subparsers.add_parser("flux-export-chatgpt-batches")
    flux_import = subparsers.add_parser("flux-import-templates")
    flux_import.add_argument("--input", required=True)
    flux_smoke = subparsers.add_parser("flux-smoke")
    flux_smoke.add_argument("--dry-run", action="store_true")
    subparsers.add_parser("flux-freeze")
    flux_generate = subparsers.add_parser("flux-generate")
    flux_generate.add_argument(
        "--phase",
        choices=["trajectory-dev", "trajectory_dev", "final-test", "final_test"],
        default="trajectory-dev",
    )
    flux_generate.add_argument("--dry-run", action="store_true")
    flux_score = subparsers.add_parser("flux-score")
    flux_score.add_argument(
        "--phase",
        choices=["smoke", "trajectory-dev", "trajectory_dev", "final-test", "final_test"],
        default="trajectory-dev",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "env-check":
        result = environment_check(config)
    elif args.command == "prepare":
        result = prepare_datasets(config)
    elif args.command == "smoke":
        result = run_generation(config, smoke=True, dry_run=args.dry_run)
    elif args.command == "freeze":
        result = freeze_phase_two(config)
    elif args.command == "generate":
        result = run_generation(config, smoke=False, dry_run=args.dry_run)
    elif args.command == "score":
        result = score_run(config, smoke=args.smoke)
    elif args.command == "select-audit":
        result = select_audit(config, smoke=args.smoke)
    elif args.command == "audit":
        result = run_audit(config, smoke=args.smoke)
    elif args.command == "audit-local":
        result = run_local_audit(
            config, smoke=args.smoke, limit=args.limit
        )
    elif args.command == "export-chatgpt-audit":
        result = export_chatgpt_audit(
            config, batch_size=args.batch_size
        )
    elif args.command == "report":
        result = build_report(config)
    elif args.command == "bot-smoke":
        from .bot_runner import run_bot_generation

        result = run_bot_generation(
            config, smoke=True, dry_run=args.dry_run
        )
    elif args.command == "bot-freeze":
        from .bot_freeze import freeze_bot_phase

        result = freeze_bot_phase(config)
    elif args.command == "bot-generate":
        from .bot_runner import run_bot_generation

        result = run_bot_generation(
            config, smoke=False, dry_run=args.dry_run
        )
    elif args.command == "bot-score":
        from .bot_evaluation import score_bot_run

        result = score_bot_run(config, smoke=args.smoke)
    elif args.command == "bot-report":
        from .bot_reporting import build_bot_report

        result = build_bot_report(config)
    elif args.command == "bot-embedding-check":
        from .semantic_setup import embedding_check

        result = embedding_check(config)
    elif args.command == "bot-export-frontier":
        from .semantic_setup import export_frontier_inputs

        result = export_frontier_inputs(config)
    elif args.command == "bot-import-frontier":
        from .semantic_setup import import_frontier_profiles

        result = import_frontier_profiles(config, input_path=args.input)
    elif args.command == "flux-prepare":
        from .legal_flux_setup import prepare_legal_flux

        result = prepare_legal_flux(config)
    elif args.command == "flux-export-templates":
        from .legal_flux_setup import export_legal_flux_template_inputs

        result = export_legal_flux_template_inputs(config)
    elif args.command == "flux-export-chatgpt-batches":
        from .legal_flux_chatgpt import export_legal_flux_chatgpt_batches

        result = export_legal_flux_chatgpt_batches(config)
    elif args.command == "flux-import-templates":
        from .legal_flux_setup import import_legal_flux_templates

        result = import_legal_flux_templates(config, input_path=args.input)
    elif args.command == "flux-smoke":
        from .legal_flux_runner import run_legal_flux_generation

        result = run_legal_flux_generation(
            config,
            phase="smoke",
            dry_run=args.dry_run,
        )
    elif args.command == "flux-freeze":
        from .legal_flux_setup import freeze_legal_flux_phase

        result = freeze_legal_flux_phase(config)
    elif args.command == "flux-generate":
        from .legal_flux_runner import run_legal_flux_generation

        result = run_legal_flux_generation(
            config,
            phase=args.phase,
            dry_run=args.dry_run,
        )
    elif args.command == "flux-score":
        from .legal_flux_evaluation import score_legal_flux_run

        result = score_legal_flux_run(config, phase=args.phase)
    else:
        raise AssertionError(args.command)
    # Keep CLI JSON portable across Windows consoles that still default to
    # legacy encodings such as cp1252. JSON consumers decode these escapes.
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
