from __future__ import annotations

import argparse
import json
import sys

from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LegalFlux case-level adaptive template-trajectory pilot"
    )
    parser.add_argument("--config", default=None, help="Path to LegalFlux YAML config.")
    subparsers = parser.add_subparsers(dest="command", required=True)

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

    if args.command == "flux-prepare":
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

        result = run_legal_flux_generation(config, phase="smoke", dry_run=args.dry_run)
    elif args.command == "flux-freeze":
        from .legal_flux_setup import freeze_legal_flux_phase

        result = freeze_legal_flux_phase(config)
    elif args.command == "flux-generate":
        from .legal_flux_runner import run_legal_flux_generation

        result = run_legal_flux_generation(
            config, phase=args.phase, dry_run=args.dry_run
        )
    elif args.command == "flux-score":
        from .legal_flux_evaluation import score_legal_flux_run

        result = score_legal_flux_run(config, phase=args.phase)
    else:
        raise AssertionError(args.command)

    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
