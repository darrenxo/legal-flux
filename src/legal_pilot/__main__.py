from __future__ import annotations

import argparse
import json
import sys

from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LegalFlux case-level adaptive template-trajectory experiments"
    )
    parser.add_argument("--config", default=None, help="Path to LegalFlux YAML config.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("flux-prepare")
    subparsers.add_parser("flux-export-templates")
    subparsers.add_parser("flux-export-template-batches")
    subparsers.add_parser("flux-export-chatgpt-batches")

    flux_import = subparsers.add_parser("flux-import-templates")
    flux_import.add_argument("--input", required=True)

    flux_smoke = subparsers.add_parser("flux-smoke")
    flux_smoke.add_argument("--dry-run", action="store_true")

    subparsers.add_parser("flux-freeze")

    flux_generate = subparsers.add_parser("flux-generate")
    flux_generate.add_argument(
        "--phase",
        choices=[
            "planner-train",
            "planner_train",
            "trajectory-dev",
            "trajectory_dev",
            "final-test",
            "final_test",
        ],
        default="trajectory-dev",
    )
    flux_generate.add_argument("--dry-run", action="store_true")
    flux_generate.add_argument("--samples", type=int, default=None)
    flux_generate.add_argument(
        "--case-limit",
        type=int,
        default=None,
        help="Limit the number of cases drawn from the selected phase.",
    )

    flux_score = subparsers.add_parser("flux-score")
    flux_score.add_argument(
        "--phase",
        choices=[
            "smoke",
            "planner-train",
            "planner_train",
            "trajectory-dev",
            "trajectory_dev",
            "final-test",
            "final_test",
        ],
        default="trajectory-dev",
    )
    subparsers.add_parser("flux-export-template-sft")
    trajectory_dpo = subparsers.add_parser("flux-export-trajectory-dpo")
    trajectory_dpo.add_argument("--phase", default="planner-train")
    deepseek_templates = subparsers.add_parser("flux-deepseek-templates")
    deepseek_templates.add_argument(
        "--stage",
        choices=["candidates", "merge", "audit", "all"],
        default="candidates",
    )
    deepseek_templates.add_argument("--limit", type=int, default=None)
    deepseek_templates.add_argument("--force", action="store_true")
    deepseek_templates.add_argument("--dry-run", action="store_true")
    gemini_templates = subparsers.add_parser("flux-gemini-templates")
    gemini_templates.add_argument(
        "--stage",
        choices=["candidates", "merge", "audit", "all"],
        default="candidates",
    )
    gemini_templates.add_argument("--limit", type=int, default=None)
    gemini_templates.add_argument("--force", action="store_true")
    gemini_templates.add_argument("--dry-run", action="store_true")
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
    elif args.command in {"flux-export-template-batches", "flux-export-chatgpt-batches"}:
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
            config,
            phase=args.phase,
            dry_run=args.dry_run,
            sample_count=args.samples,
            case_limit=args.case_limit,
        )
    elif args.command == "flux-score":
        from .legal_flux_evaluation import score_legal_flux_run

        result = score_legal_flux_run(config, phase=args.phase)
    elif args.command == "flux-export-template-sft":
        from .legal_flux_training import export_template_structure_sft

        result = export_template_structure_sft(config)
    elif args.command == "flux-export-trajectory-dpo":
        from .legal_flux_training import export_trajectory_dpo

        result = export_trajectory_dpo(config, phase=args.phase)
    elif args.command == "flux-deepseek-templates":
        from .legal_flux_deepseek import run_deepseek_template_workflow

        result = run_deepseek_template_workflow(
            config,
            stage=args.stage,
            limit=args.limit,
            force=args.force,
            dry_run=args.dry_run,
        )
    elif args.command == "flux-gemini-templates":
        from .legal_flux_gemini import run_gemini_template_workflow

        result = run_gemini_template_workflow(
            config,
            stage=args.stage,
            limit=args.limit,
            force=args.force,
            dry_run=args.dry_run,
        )
    else:
        raise AssertionError(args.command)

    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
