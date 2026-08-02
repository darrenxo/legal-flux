from __future__ import annotations

import csv
import inspect
import json
import math
import os
import random
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import resolve_path, resolve_project_path
from .io_utils import read_jsonl, sha256_text, write_jsonl
from .legal_flux_training import export_template_structure_sft
from .runner import load_cases


def train_template_structure_sft(
    config: dict[str, Any],
    *,
    dry_run: bool = False,
    resume_from_checkpoint: str | None = None,
    learning_rate: float | None = None,
    num_train_epochs: int | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    runtime_config = _with_sft_overrides(
        config,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        output_dir=output_dir,
    )
    settings = template_sft_settings(runtime_config)
    data = prepare_template_sft_splits(runtime_config)
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    effective_batch_size = (
        settings["per_device_train_batch_size"]
        * settings["gradient_accumulation_steps"]
        * world_size
    )
    optimizer_steps = (
        math.ceil(data["train_examples"] / effective_batch_size)
        * settings["num_train_epochs"]
    )
    preflight = {
        "task": "template_structure_sft",
        "dry_run": dry_run,
        "model_name_or_path": settings["model_name_or_path"],
        "finetuning_type": settings["finetuning_type"],
        "train_examples": data["train_examples"],
        "eval_examples": data["eval_examples"],
        "train_file": data["train_file"],
        "eval_file": data["eval_file"],
        "output_dir": str(settings["output_dir"]),
        "world_size": world_size,
        "effective_batch_size": effective_batch_size,
        "estimated_optimizer_steps": optimizer_steps,
        "settings": _json_safe_settings(settings),
        "warnings": _sft_warnings(data["total_examples"]),
        "checkpoint_selection": (
            "Select checkpoints by LegalFlux accuracy on trajectory_dev; "
            "template reconstruction holdout is disabled by default."
        ),
    }
    if dry_run:
        return preflight

    try:
        import torch
        import transformers
        import trl
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Template SFT requires the training dependencies. Install the project "
            "with `pip install -e .[train]`."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Template SFT requires a CUDA GPU.")

    train_rows = _trainer_rows(Path(data["train_file"]))
    eval_rows = _trainer_rows(Path(data["eval_file"]))
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None
    use_bf16 = bool(settings["bf16"]) and torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16
    sft_config_kwargs = _template_sft_config_kwargs(
        settings,
        model_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
        has_eval=eval_dataset is not None,
    )
    _validate_constructor_kwargs(
        SFTConfig,
        sft_config_kwargs,
        component="TRL SFTConfig",
    )
    training_args = SFTConfig(**sft_config_kwargs)
    peft_config = None
    if settings["finetuning_type"] == "lora":
        lora_config_kwargs = _template_lora_config_kwargs(settings)
        _validate_constructor_kwargs(
            LoraConfig,
            lora_config_kwargs,
            component="PEFT LoraConfig",
        )
        peft_config = LoraConfig(**lora_config_kwargs)
    processing_class = _load_text_tokenizer(AutoTokenizer, settings)
    trainer = SFTTrainer(
        model=settings["model_name_or_path"],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=processing_class,
    )
    train_result = trainer.train(
        resume_from_checkpoint=resume_from_checkpoint or None
    )
    final_dir = settings["output_dir"] / "final"
    trainer.save_model(str(final_dir))
    if trainer.processing_class is not None:
        trainer.processing_class.save_pretrained(final_dir)

    manifest = {
        **preflight,
        "dry_run": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "final_checkpoint": str(final_dir),
        "train_metrics": train_result.metrics,
        "library_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
        },
        "dataset_hashes": {
            "train": sha256_text(Path(data["train_file"]).read_text(encoding="utf-8")),
            "eval": sha256_text(Path(data["eval_file"]).read_text(encoding="utf-8")),
        },
    }
    manifest_path = settings["output_dir"] / "training_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest_path": str(manifest_path)}


def _template_sft_config_kwargs(
    settings: dict[str, Any],
    *,
    model_dtype: Any,
    use_bf16: bool,
    use_fp16: bool,
    has_eval: bool,
) -> dict[str, Any]:
    model_init_kwargs: dict[str, Any] = {
        "dtype": model_dtype,
        "trust_remote_code": settings["trust_remote_code"],
    }
    if settings["attn_implementation"]:
        model_init_kwargs["attn_implementation"] = settings["attn_implementation"]

    return dict(
        output_dir=str(settings["output_dir"]),
        model_init_kwargs=model_init_kwargs,
        max_length=settings["max_length"],
        completion_only_loss=True,
        packing=False,
        num_train_epochs=settings["num_train_epochs"],
        per_device_train_batch_size=settings["per_device_train_batch_size"],
        per_device_eval_batch_size=settings["per_device_eval_batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        learning_rate=settings["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=settings["warmup_ratio"],
        optim="adamw_torch_fused",
        weight_decay=settings["weight_decay"],
        max_grad_norm=settings["max_grad_norm"],
        gradient_checkpointing=settings["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=settings["tf32"],
        eval_strategy="epoch" if has_eval else "no",
        save_strategy="epoch",
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        save_total_limit=settings["save_total_limit"],
        logging_steps=settings["logging_steps"],
        report_to=settings["report_to"],
        seed=settings["seed"],
        data_seed=settings["seed"],
        dataset_num_proc=settings["dataset_num_proc"],
        remove_unused_columns=True,
    )


def _validate_constructor_kwargs(
    constructor: Any,
    kwargs: dict[str, Any],
    *,
    component: str,
) -> None:
    parameters = inspect.signature(constructor).parameters.values()
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return
    supported = {parameter.name for parameter in parameters}
    unsupported = sorted(set(kwargs) - supported)
    if unsupported:
        names = ", ".join(unsupported)
        raise RuntimeError(
            f"Installed {component} does not support these configured arguments: "
            f"{names}. Check the installed training dependency versions."
        )


def _load_text_tokenizer(auto_tokenizer: Any, settings: dict[str, Any]) -> Any:
    tokenizer = auto_tokenizer.from_pretrained(
        settings["model_name_or_path"],
        trust_remote_code=settings["trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise RuntimeError(
                "The text tokenizer has neither a padding token nor an EOS token."
            )
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _template_lora_config_kwargs(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "r": settings["lora_r"],
        "lora_alpha": settings["lora_alpha"],
        "lora_dropout": settings["lora_dropout"],
        "target_modules": settings["lora_target_modules"],
        "exclude_modules": settings["lora_exclude_modules"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def prepare_template_sft_splits(config: dict[str, Any]) -> dict[str, Any]:
    export = export_template_structure_sft(config)
    source_path = Path(export["output_path"])
    rows = read_jsonl(source_path)
    settings = template_sft_settings(config)
    eval_fraction = settings["eval_fraction"]
    if not 0.0 <= eval_fraction < 1.0:
        raise ValueError("Template SFT eval_fraction must be in [0, 1).")
    eval_count = round(len(rows) * eval_fraction)
    if eval_count >= len(rows):
        raise ValueError("Template SFT requires at least one training example.")
    shuffled = list(rows)
    random.Random(settings["seed"]).shuffle(shuffled)
    eval_ids = {row["template_id"] for row in shuffled[:eval_count]}
    train_rows = [row for row in rows if row["template_id"] not in eval_ids]
    eval_rows = [row for row in rows if row["template_id"] in eval_ids]
    output_dir = source_path.parent
    train_path = output_dir / "template_structure_sft_train.jsonl"
    eval_path = output_dir / "template_structure_sft_eval.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    split_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": settings["seed"],
        "eval_fraction": settings["eval_fraction"],
        "total_examples": len(rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "train_file": str(train_path),
        "eval_file": str(eval_path),
        "held_out_unit": "template_id" if eval_rows else None,
        "selection_split": (
            "internal_template_eval" if eval_rows else "trajectory_dev"
        ),
        "source_manifest": export["manifest_path"],
    }
    manifest_path = output_dir / "template_structure_sft_split_manifest.json"
    manifest_path.write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**split_manifest, "manifest_path": str(manifest_path)}


def export_trajectory_dev_tune_subset(
    config: dict[str, Any],
    *,
    count: int = 256,
) -> dict[str, Any]:
    cases = [
        case
        for case in load_cases(config)
        if case.metadata.get("selection_split") == "trajectory_dev"
    ]
    if not 1 <= count <= len(cases):
        raise ValueError(
            f"count must be between 1 and the {len(cases)} trajectory-dev cases."
        )
    seed = int(config["project"]["seed"])
    groups: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for case in cases:
        groups[_dev_tune_stratum(case)].append(case)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    quotas = {
        key: min(len(group), math.floor(count * len(group) / len(cases)))
        for key, group in groups.items()
    }
    remaining = count - sum(quotas.values())
    priorities = sorted(
        groups,
        key=lambda key: (
            -(count * len(groups[key]) / len(cases) - quotas[key]),
            key,
        ),
    )
    while remaining:
        progressed = False
        for key in priorities:
            if quotas[key] >= len(groups[key]):
                continue
            quotas[key] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise RuntimeError("Could not allocate the requested development subset.")
    selected_ids = {
        case.case_id
        for key, group in groups.items()
        for case in group[: quotas[key]]
    }
    ordered_ids = [case.case_id for case in cases if case.case_id in selected_ids]
    output_dir = resolve_path(config, "processed_dir") / "planner_training"
    output_path = output_dir / f"trajectory_dev_tune_{count}.json"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "trajectory_dev",
        "seed": seed,
        "count": count,
        "case_ids": ordered_ids,
        "stratified_by": ["gold_answer", "broad_domain", "authority_bucket"],
        "label_counts": dict(
            sorted(
                Counter(
                    case.gold_answer for case in cases if case.case_id in selected_ids
                ).items()
            )
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**payload, "output_path": str(output_path)}


def summarize_sft_checkpoint_grid(
    config: dict[str, Any],
    *,
    phase: str = "trajectory_dev",
    prefix: str = "sft-",
) -> dict[str, Any]:
    normalized_phase = phase.replace("-", "_")
    experiments_dir = (
        resolve_path(config, "runs_dir") / normalized_phase / "experiments"
    )
    rows: list[dict[str, Any]] = []
    for aggregate_path in sorted(experiments_dir.glob(f"{prefix}*/aggregate.csv")):
        with aggregate_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("condition") != "flux_rf_style":
                    continue
                rows.append(
                    {
                        "run_tag": aggregate_path.parent.name,
                        **row,
                    }
                )
    if not rows:
        raise FileNotFoundError(
            f"No scored flux_rf_style experiments matching {prefix!r} under "
            f"{experiments_dir}."
        )
    rows.sort(
        key=lambda row: (
            -_float_value(row.get("answer_correct"), default=float("-inf")),
            -_float_value(row.get("weighted_f1"), default=float("-inf")),
            _float_value(row.get("calls"), default=float("inf")),
            row["run_tag"],
        )
    )
    reports_dir = resolve_path(config, "reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "sft_checkpoint_grid.csv"
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = reports_dir / "sft_checkpoint_grid.json"
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": normalized_phase,
        "prefix": prefix,
        "experiments": len(rows),
        "best_run_tag": rows[0]["run_tag"],
        "best_accuracy": _float_value(
            rows[0].get("answer_correct"), default=float("-inf")
        ),
        "output_path": str(output_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**summary, "summary_path": str(summary_path)}


def template_sft_settings(config: dict[str, Any]) -> dict[str, Any]:
    values = config.get("training", {}).get("template_sft", {})
    model_name = str(
        values.get("model_name_or_path") or "Qwen/Qwen3.5-9B"
    ).strip()
    finetuning_type = str(values.get("finetuning_type", "lora"))
    if finetuning_type not in {"lora", "full"}:
        raise ValueError("training.template_sft.finetuning_type must be lora or full.")
    output_dir = resolve_project_path(
        config,
        values.get(
            "output_dir",
            "runs/legal_flux/training/template_structure_sft",
        ),
    )
    return {
        "model_name_or_path": model_name,
        "output_dir": output_dir,
        "finetuning_type": finetuning_type,
        "eval_fraction": float(values.get("eval_fraction", 0.0)),
        "num_train_epochs": int(values.get("num_train_epochs", 6)),
        "learning_rate": float(
            values.get("learning_rate", 1e-4 if finetuning_type == "lora" else 1e-5)
        ),
        "per_device_train_batch_size": int(
            values.get("per_device_train_batch_size", 1)
        ),
        "per_device_eval_batch_size": int(
            values.get("per_device_eval_batch_size", 1)
        ),
        "gradient_accumulation_steps": int(
            values.get("gradient_accumulation_steps", 16)
        ),
        "warmup_ratio": float(values.get("warmup_ratio", 0.05)),
        "weight_decay": float(values.get("weight_decay", 0.01)),
        "max_grad_norm": float(values.get("max_grad_norm", 1.0)),
        "max_length": int(values.get("max_length", 1024)),
        "gradient_checkpointing": bool(values.get("gradient_checkpointing", True)),
        "bf16": bool(values.get("bf16", True)),
        "tf32": bool(values.get("tf32", True)),
        "save_total_limit": int(values.get("save_total_limit", 6)),
        "logging_steps": int(values.get("logging_steps", 5)),
        "dataset_num_proc": int(values.get("dataset_num_proc", 4)),
        "report_to": values.get("report_to", "none"),
        "seed": int(values.get("seed", config["project"]["seed"])),
        "trust_remote_code": bool(values.get("trust_remote_code", False)),
        "attn_implementation": values.get("attn_implementation", "sdpa"),
        "lora_r": int(values.get("lora_r", 32)),
        "lora_alpha": int(values.get("lora_alpha", 64)),
        "lora_dropout": float(values.get("lora_dropout", 0.05)),
        "lora_target_modules": values.get("lora_target_modules", "all-linear"),
        "lora_exclude_modules": list(
            values.get(
                "lora_exclude_modules",
                ["visual", "vision_model", "vision_tower", "merger"],
            )
        ),
    }


def _trainer_rows(path: Path) -> list[dict[str, Any]]:
    return [
        {
            "prompt": row["prompt"],
            "completion": row["completion"],
        }
        for row in read_jsonl(path)
    ]


def _dev_tune_stratum(case: Any) -> tuple[str, str, str]:
    return (
        case.gold_answer,
        str(case.metadata.get("broad_domain") or "unknown"),
        str(case.metadata.get("authority_bucket") or "unknown"),
    )


def _float_value(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sft_warnings(example_count: int) -> list[str]:
    warnings = [
        "All templates are used for SFT by default. Hyperparameters and epoch "
        "checkpoints should be selected with case-level trajectory_dev results."
    ]
    if example_count < 1000:
        warnings.append(
            f"Only {example_count} structure examples are available. ReasonFlux "
            "reports 15K examples extended from about 500 templates, so this is "
            "a first checkpoint rather than a scale-matched replication."
        )
    return warnings


def _json_safe_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in settings.items()
    }


def _with_sft_overrides(
    config: dict[str, Any],
    *,
    learning_rate: float | None,
    num_train_epochs: int | None,
    output_dir: str | None,
) -> dict[str, Any]:
    runtime = deepcopy(config)
    values = runtime.setdefault("training", {}).setdefault("template_sft", {})
    if learning_rate is not None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        values["learning_rate"] = learning_rate
    if num_train_epochs is not None:
        if num_train_epochs < 1:
            raise ValueError("num_train_epochs must be at least 1.")
        values["num_train_epochs"] = num_train_epochs
    if output_dir is not None:
        values["output_dir"] = output_dir
    return runtime
