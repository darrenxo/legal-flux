from __future__ import annotations

import inspect
import json
import math
import os
import random
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import resolve_path, resolve_project_path
from .io_utils import read_jsonl, sha256_text, write_jsonl
from .legal_flux import load_template_pool, template_pool_hash
from .legal_flux_dpo import dpo_construction_workflow_hash
from .models import LegalFluxAbstractPlan


def train_trajectory_dpo(
    config: dict[str, Any],
    *,
    dry_run: bool = False,
    resume_from_checkpoint: str | None = None,
    model_name_or_path: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    runtime_config = _with_dpo_training_overrides(
        config,
        model_name_or_path=model_name_or_path,
        output_dir=output_dir,
    )
    settings = trajectory_dpo_settings(runtime_config)
    data = prepare_trajectory_dpo_splits(runtime_config)
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
        "task": "trajectory_dpo",
        "dry_run": dry_run,
        "model_name_or_path": settings["model_name_or_path"],
        "reference_policy": (
            "The unchanged initial SFT adapter loaded at DPO trainer startup."
        ),
        "train_examples": data["train_examples"],
        "eval_examples": data["eval_examples"],
        "train_file": data["train_file"],
        "eval_file": data["eval_file"],
        "source_file": data["source_file"],
        "output_dir": str(settings["output_dir"]),
        "world_size": world_size,
        "effective_batch_size": effective_batch_size,
        "estimated_optimizer_steps": optimizer_steps,
        "settings": _json_safe_settings(settings),
        "objective": (
            "DPO on the current planner prompt with preferred and rejected "
            "canonical LegalFlux trajectory-plan JSON completions."
        ),
    }
    if dry_run:
        return preflight

    try:
        import torch
        import transformers
        import trl
        from datasets import Dataset
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Trajectory DPO requires the training dependencies. Install the "
            "project with `pip install -e .[train]`."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Trajectory DPO requires a CUDA GPU.")

    use_bf16 = bool(settings["bf16"]) and torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16
    model_kwargs: dict[str, Any] = {
        "is_trainable": True,
        "dtype": model_dtype,
        "trust_remote_code": settings["trust_remote_code"],
    }
    if settings["attn_implementation"]:
        model_kwargs["attn_implementation"] = settings["attn_implementation"]
    tokenizer = AutoTokenizer.from_pretrained(
        _tokenizer_source(settings["model_name_or_path"]),
        trust_remote_code=settings["trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise RuntimeError(
                "The DPO tokenizer has neither a padding token nor an EOS token."
            )
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    train_rows = _dpo_trainer_rows(
        Path(data["train_file"]),
        tokenizer=tokenizer,
        chat_template_kwargs=settings["chat_template_kwargs"],
    )
    eval_rows = _dpo_trainer_rows(
        Path(data["eval_file"]),
        tokenizer=tokenizer,
        chat_template_kwargs=settings["chat_template_kwargs"],
    )
    _validate_dpo_token_lengths(
        [*train_rows, *eval_rows],
        tokenizer=tokenizer,
        max_length=settings["max_length"],
    )
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None
    dpo_config_kwargs = _trajectory_dpo_config_kwargs(
        settings,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
        has_eval=eval_dataset is not None,
    )
    _validate_constructor_kwargs(
        DPOConfig,
        dpo_config_kwargs,
        component="TRL DPOConfig",
    )
    training_args = DPOConfig(**dpo_config_kwargs)
    model = AutoPeftModelForCausalLM.from_pretrained(
        settings["model_name_or_path"],
        **model_kwargs,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
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
            "source": sha256_text(Path(data["source_file"]).read_text(encoding="utf-8")),
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


def prepare_trajectory_dpo_splits(config: dict[str, Any]) -> dict[str, Any]:
    settings = trajectory_dpo_settings(config)
    source_path = (
        resolve_path(config, "processed_dir")
        / "planner_training"
        / "trajectory_dpo.jsonl"
    )
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Trajectory DPO data does not exist at {source_path}. Run "
            "`flux-export-trajectory-dpo` first."
        )
    source_manifest_path = source_path.with_name("trajectory_dpo_manifest.json")
    if source_manifest_path.is_file():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        context_config = _config_for_recorded_collection(config, source_manifest)
        expected = {
            "workflow_hash": dpo_construction_workflow_hash(context_config),
            "template_pool_hash": template_pool_hash(load_template_pool(config)),
        }
        mismatches = {
            key: (source_manifest.get(key), value)
            for key, value in expected.items()
            if source_manifest.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "Trajectory DPO pairs were exported from a different pipeline. "
                f"Rebuild them before training. Mismatches: {mismatches}"
            )
        source_checkpoint = str(source_manifest.get("source_checkpoint") or "")
        if source_checkpoint and not _same_model_source(
            settings["model_name_or_path"], source_checkpoint
        ):
            raise RuntimeError(
                "Trajectory preferences were collected with SFT checkpoint "
                f"{source_checkpoint}, but DPO training was asked to update "
                f"{settings['model_name_or_path']}. Use the same checkpoint."
            )
    rows = read_jsonl(source_path)
    validated = [_validated_dpo_row(row) for row in rows]
    if not validated:
        raise ValueError("Trajectory DPO training requires at least one preference pair.")
    eval_fraction = settings["eval_fraction"]
    if not 0.0 <= eval_fraction < 1.0:
        raise ValueError("training.trajectory_dpo.eval_fraction must be in [0, 1).")
    shuffled = list(validated)
    random.Random(settings["seed"]).shuffle(shuffled)
    eval_count = round(len(shuffled) * eval_fraction)
    if eval_count >= len(shuffled):
        raise ValueError("Trajectory DPO requires at least one training pair.")
    eval_ids = {row["id"] for row in shuffled[:eval_count]}
    train_rows = [row for row in validated if row["id"] not in eval_ids]
    eval_rows = [row for row in validated if row["id"] in eval_ids]
    output_dir = source_path.parent
    train_path = output_dir / "trajectory_dpo_train.jsonl"
    eval_path = output_dir / "trajectory_dpo_eval.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": settings["seed"],
        "eval_fraction": eval_fraction,
        "total_examples": len(validated),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "source_file": str(source_path),
        "source_manifest": (
            str(source_manifest_path) if source_manifest_path.is_file() else None
        ),
        "train_file": str(train_path),
        "eval_file": str(eval_path),
        "held_out_unit": "anchor_case" if eval_rows else None,
    }
    manifest_path = output_dir / "trajectory_dpo_split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest_path": str(manifest_path)}


def trajectory_dpo_settings(config: dict[str, Any]) -> dict[str, Any]:
    values = config.get("training", {}).get("trajectory_dpo", {})
    model_name = str(values.get("model_name_or_path") or "").strip()
    if not model_name:
        raise ValueError(
            "training.trajectory_dpo.model_name_or_path must identify the selected "
            "template-structure SFT adapter checkpoint."
        )
    return {
        "model_name_or_path": model_name,
        "output_dir": resolve_project_path(
            config,
            values.get(
                "output_dir",
                "runs/legal_flux/training/trajectory_dpo",
            ),
        ),
        "eval_fraction": float(values.get("eval_fraction", 0.0)),
        "num_train_epochs": int(values.get("num_train_epochs", 1)),
        "learning_rate": float(values.get("learning_rate", 1e-5)),
        "beta": float(values.get("beta", 0.1)),
        "loss_type": str(values.get("loss_type", "sigmoid")),
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
        "max_length": int(values.get("max_length", 6144)),
        "gradient_checkpointing": bool(values.get("gradient_checkpointing", True)),
        "bf16": bool(values.get("bf16", True)),
        "tf32": bool(values.get("tf32", True)),
        "save_total_limit": int(values.get("save_total_limit", 2)),
        "logging_steps": int(values.get("logging_steps", 5)),
        "dataset_num_proc": int(values.get("dataset_num_proc", 4)),
        "report_to": values.get("report_to", "none"),
        "seed": int(values.get("seed", config["project"]["seed"])),
        "trust_remote_code": bool(values.get("trust_remote_code", False)),
        "attn_implementation": values.get("attn_implementation", "sdpa"),
        "precompute_ref_log_probs": bool(
            values.get("precompute_ref_log_probs", False)
        ),
        "chat_template_kwargs": dict(
            values.get("chat_template_kwargs", {"enable_thinking": False})
        ),
    }


def _trajectory_dpo_config_kwargs(
    settings: dict[str, Any],
    *,
    use_bf16: bool,
    use_fp16: bool,
    has_eval: bool,
) -> dict[str, Any]:
    return {
        "output_dir": str(settings["output_dir"]),
        "max_length": settings["max_length"],
        "truncation_mode": "keep_start",
        "beta": settings["beta"],
        "loss_type": settings["loss_type"],
        "num_train_epochs": settings["num_train_epochs"],
        "per_device_train_batch_size": settings["per_device_train_batch_size"],
        "per_device_eval_batch_size": settings["per_device_eval_batch_size"],
        "gradient_accumulation_steps": settings["gradient_accumulation_steps"],
        "learning_rate": settings["learning_rate"],
        "lr_scheduler_type": "cosine",
        "warmup_ratio": settings["warmup_ratio"],
        "optim": "adamw_torch_fused",
        "weight_decay": settings["weight_decay"],
        "max_grad_norm": settings["max_grad_norm"],
        "gradient_checkpointing": settings["gradient_checkpointing"],
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "bf16": use_bf16,
        "fp16": use_fp16,
        "tf32": settings["tf32"],
        "eval_strategy": "epoch" if has_eval else "no",
        "save_strategy": "epoch",
        "load_best_model_at_end": has_eval,
        "metric_for_best_model": "eval_loss" if has_eval else None,
        "greater_is_better": False if has_eval else None,
        "save_total_limit": settings["save_total_limit"],
        "logging_steps": settings["logging_steps"],
        "report_to": settings["report_to"],
        "seed": settings["seed"],
        "data_seed": settings["seed"],
        "dataset_num_proc": settings["dataset_num_proc"],
        "remove_unused_columns": True,
        "precompute_ref_log_probs": settings["precompute_ref_log_probs"],
    }


def _validated_dpo_row(row: dict[str, Any]) -> dict[str, Any]:
    identifier = str(row.get("id") or "").strip()
    prompt = str(row.get("prompt") or "")
    chosen = str(row.get("chosen") or "")
    rejected = str(row.get("rejected") or "")
    if not identifier or not prompt.strip() or not chosen.strip() or not rejected.strip():
        raise ValueError("Every DPO row requires nonempty id, prompt, chosen, rejected.")
    chosen_plan = LegalFluxAbstractPlan.model_validate(json.loads(chosen))
    rejected_plan = LegalFluxAbstractPlan.model_validate(json.loads(rejected))
    if chosen_plan == rejected_plan:
        raise ValueError(f"DPO row {identifier} has identical chosen/rejected plans.")
    if float(row.get("chosen_reward", 0.0)) <= float(
        row.get("rejected_reward", 0.0)
    ):
        raise ValueError(
            f"DPO row {identifier} does not have a strictly preferred reward."
        )
    return {
        **row,
        "id": identifier,
        "prompt": prompt,
        "chosen": json.dumps(
            chosen_plan.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        "rejected": json.dumps(
            rejected_plan.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
    }


def _dpo_trainer_rows(
    path: Path,
    *,
    tokenizer: Any,
    chat_template_kwargs: dict[str, Any],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_jsonl(path):
        rendered_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(row["prompt"])}],
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        rows.append(
            {
                "prompt": str(rendered_prompt),
                "chosen": str(row["chosen"]),
                "rejected": str(row["rejected"]),
            }
        )
    return rows


def _validate_dpo_token_lengths(
    rows: list[dict[str, str]],
    *,
    tokenizer: Any,
    max_length: int,
) -> None:
    longest = 0
    longest_id = ""
    for row_index, row in enumerate(rows):
        for response_name in ("chosen", "rejected"):
            encoded = tokenizer(
                row["prompt"] + row[response_name],
                add_special_tokens=False,
                truncation=False,
            )
            token_count = len(encoded["input_ids"]) + 1
            if token_count > longest:
                longest = token_count
                longest_id = f"row {row_index} {response_name}"
    if longest > max_length:
        raise ValueError(
            f"DPO max_length={max_length} would truncate {longest_id} at "
            f"{longest} tokens. Increase training.trajectory_dpo.max_length; "
            "trajectory completions must not be silently truncated."
        )


def _tokenizer_source(model_name_or_path: str) -> str:
    path = Path(model_name_or_path)
    if not path.is_dir() or (path / "tokenizer_config.json").is_file():
        return model_name_or_path
    adapter_config = path / "adapter_config.json"
    if adapter_config.is_file():
        payload = json.loads(adapter_config.read_text(encoding="utf-8"))
        base_model = str(payload.get("base_model_name_or_path") or "").strip()
        if base_model:
            return base_model
    return model_name_or_path


def _config_for_recorded_collection(
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    runtime = deepcopy(config)
    legal_flux = runtime.setdefault("legal_flux", {})
    dpo = runtime.setdefault("dpo", {})
    for role in ("planner", "executor", "reviewer"):
        recorded = manifest.get(f"{role}_model")
        if recorded is not None:
            legal_flux[f"{role}_model"] = recorded
            dpo[f"{role}_model"] = recorded
    source_checkpoint = manifest.get("source_checkpoint")
    if source_checkpoint:
        dpo["source_checkpoint"] = source_checkpoint
    else:
        dpo.pop("source_checkpoint", None)
    return runtime


def _same_model_source(left: str, right: str) -> bool:
    if left == right:
        return True
    left_path = Path(left).expanduser()
    right_path = Path(right).expanduser()
    if left_path.exists() and right_path.exists():
        return left_path.resolve() == right_path.resolve()
    return False


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


def _with_dpo_training_overrides(
    config: dict[str, Any],
    *,
    model_name_or_path: str | None,
    output_dir: str | None,
) -> dict[str, Any]:
    runtime = deepcopy(config)
    values = runtime.setdefault("training", {}).setdefault("trajectory_dpo", {})
    if model_name_or_path is not None:
        if not model_name_or_path.strip():
            raise ValueError("model_name_or_path must not be empty.")
        values["model_name_or_path"] = model_name_or_path
    if output_dir is not None:
        values["output_dir"] = output_dir
    return runtime


def _json_safe_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in settings.items()
    }
