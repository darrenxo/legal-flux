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
    flux_generate.add_argument(
        "--conditions",
        nargs="+",
        choices=["direct", "structured", "flux_rf_style"],
        default=None,
        help="Optional subset of configured comparison conditions.",
    )
    flux_generate.add_argument("--num-shards", type=int, default=1)
    flux_generate.add_argument("--shard-index", type=int, default=0)
    flux_generate.add_argument(
        "--run-tag",
        default=None,
        help="Store this run under trajectory phase experiments/<run-tag>.",
    )
    flux_generate.add_argument(
        "--case-ids-file",
        default=None,
        help="Optional JSON file containing the exact case IDs to run.",
    )
    flux_generate.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="Exit nonzero after preserving the ledger if any generation fails.",
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
    flux_score.add_argument("--run-tag", default=None)
    subparsers.add_parser("flux-export-template-sft")
    template_sft = subparsers.add_parser("flux-train-template-sft")
    template_sft.add_argument("--dry-run", action="store_true")
    template_sft.add_argument("--resume-from-checkpoint", default=None)
    template_sft.add_argument("--learning-rate", type=float, default=None)
    template_sft.add_argument("--num-train-epochs", type=int, default=None)
    template_sft.add_argument("--output-dir", default=None)
    vllm_adapter = subparsers.add_parser("flux-prepare-vllm-adapter")
    vllm_adapter.add_argument("checkpoints", nargs="+")
    vllm_adapter.add_argument("--output-name", default="vllm_text_only")
    dev_tune = subparsers.add_parser("flux-export-dev-tune")
    dev_tune.add_argument("--count", type=int, default=256)
    sft_grid = subparsers.add_parser("flux-summarize-sft-grid")
    sft_grid.add_argument("--phase", default="trajectory-dev")
    sft_grid.add_argument("--prefix", default="sft-")
    xsim = subparsers.add_parser("flux-build-xsim")
    xsim.add_argument(
        "--stage",
        choices=["dense", "rerank", "all"],
        default="all",
    )
    xsim.add_argument("--case-limit", type=int, default=None)
    xsim.add_argument("--force", action="store_true")
    dpo_data = subparsers.add_parser("flux-build-dpo-data")
    dpo_data.add_argument(
        "--stage",
        choices=["sample", "evaluate", "all"],
        default="all",
    )
    dpo_data.add_argument("--case-limit", type=int, default=None)
    dpo_data.add_argument("--force", action="store_true")
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
            conditions=args.conditions,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            run_tag=args.run_tag,
            case_ids_file=args.case_ids_file,
            fail_on_errors=args.fail_on_errors,
        )
    elif args.command == "flux-score":
        from .legal_flux_evaluation import score_legal_flux_run

        result = score_legal_flux_run(
            config,
            phase=args.phase,
            run_tag=args.run_tag,
        )
    elif args.command == "flux-export-template-sft":
        from .legal_flux_training import export_template_structure_sft

        result = export_template_structure_sft(config)
    elif args.command == "flux-train-template-sft":
        from .legal_flux_sft import train_template_structure_sft

        result = train_template_structure_sft(
            config,
            dry_run=args.dry_run,
            resume_from_checkpoint=args.resume_from_checkpoint,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            output_dir=args.output_dir,
        )
    elif args.command == "flux-export-dev-tune":
        from .legal_flux_sft import export_trajectory_dev_tune_subset

        result = export_trajectory_dev_tune_subset(config, count=args.count)
    elif args.command == "flux-prepare-vllm-adapter":
        from .legal_flux_sft import prepare_vllm_text_adapter

        result = {
            "adapters": [
                prepare_vllm_text_adapter(
                    checkpoint,
                    output_name=args.output_name,
                )
                for checkpoint in args.checkpoints
            ]
        }
    elif args.command == "flux-summarize-sft-grid":
        from .legal_flux_sft import summarize_sft_checkpoint_grid

        result = summarize_sft_checkpoint_grid(
            config,
            phase=args.phase,
            prefix=args.prefix,
        )
    elif args.command == "flux-build-xsim":
        from .legal_flux_xsim import build_xsim

        result = build_xsim(
            config,
            stage=args.stage,
            case_limit=args.case_limit,
            force=args.force,
        )
    elif args.command == "flux-build-dpo-data":
        from .legal_flux_dpo import build_dpo_data

        result = build_dpo_data(
            config,
            stage=args.stage,
            case_limit=args.case_limit,
            force=args.force,
        )
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
