from __future__ import annotations

import inspect
import json
import math
import os
import random
import warnings
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import resolve_path, resolve_project_path
from .io_utils import read_jsonl, sha256_text, write_jsonl
from .legal_flux import load_template_pool, template_pool_hash
from .legal_flux_dpo import dpo_construction_workflow_hash
from .models import LegalFluxAbstractPlan
from .prompting import render_prompt
from .runner import load_cases


DPO_POLICY_ADAPTER_NAME = "default"
DPO_REFERENCE_ADAPTER_NAME = "ref"


def train_trajectory_dpo(
    config: dict[str, Any],
    *,
    dry_run: bool = False,
    validate_model_load: bool = False,
    resume_from_checkpoint: str | None = None,
    model_name_or_path: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    if dry_run and validate_model_load:
        raise ValueError("dry_run and validate_model_load are mutually exclusive.")
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
        "policy_adapter_source": _dpo_policy_adapter_source(
            settings["model_name_or_path"]
        ),
        "reference_policy": (
            "TRL 0.29 copies the selected SFT `default` adapter to a frozen "
            "`ref` adapter before DPO training starts."
        ),
        "policy_adapter_name": DPO_POLICY_ADAPTER_NAME,
        "reference_adapter_name": DPO_REFERENCE_ADAPTER_NAME,
        "train_examples": data["train_examples"],
        "eval_examples": data["eval_examples"],
        "train_file": data["train_file"],
        "eval_file": data["eval_file"],
        "source_file": data["source_file"],
        "source_artifact_audit": data.get("source_artifact_audit"),
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
        import peft
        import trl
        from datasets import Dataset
        from peft import (
            PeftConfig,
            PeftModel,
            get_peft_model_state_dict,
            load_peft_weights,
        )
        from safetensors import safe_open
        from packaging.version import Version
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Trajectory DPO requires the training dependencies. Install the "
            "project with `pip install -e .[train]`."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Trajectory DPO requires a CUDA GPU.")
    trl_version = Version(trl.__version__)
    if not Version("0.29") <= trl_version < Version("0.30"):
        raise RuntimeError(
            "Trajectory DPO requires TRL >=0.29,<0.30 so the trainer copies "
            "the initial SFT adapter into its native frozen `ref` adapter; "
            f"found TRL {trl.__version__}."
        )

    use_bf16 = bool(settings["bf16"]) and torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16
    model_kwargs: dict[str, Any] = {
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
    adapter_source = str(preflight["policy_adapter_source"])
    adapter_config = PeftConfig.from_pretrained(adapter_source)
    base_model_source = str(adapter_config.base_model_name_or_path or "").strip()
    if not base_model_source:
        raise RuntimeError(
            f"The policy adapter at {adapter_source} does not identify its base model."
        )
    with safe_open(
        Path(adapter_source) / "adapter_model.safetensors",
        framework="pt",
        device="cpu",
    ) as handle:
        adapter_key_mapping = _dpo_adapter_key_mapping(list(handle.keys()))
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_source,
        **model_kwargs,
    )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error",
                message=r"Found missing adapter keys while loading the checkpoint.*",
                category=UserWarning,
            )
            model = PeftModel.from_pretrained(
                base_model,
                adapter_source,
                is_trainable=True,
                config=adapter_config,
                key_mapping=adapter_key_mapping,
            )
    except UserWarning as exc:
        raise RuntimeError(
            "The selected SFT policy adapter did not load completely. Refusing "
            "to start DPO from newly initialized LoRA tensors."
        ) from exc
    _validate_initial_sft_policy_adapter(model)
    initial_policy_validation = _validate_loaded_policy_state(
        source_state=load_peft_weights(
            adapter_source,
            device="cpu",
            key_mapping=adapter_key_mapping,
        ),
        loaded_state=get_peft_model_state_dict(
            model,
            adapter_name=DPO_POLICY_ADAPTER_NAME,
        ),
        adapter_source=adapter_source,
    )
    print(
        json.dumps(
            {"initial_policy_validation": initial_policy_validation},
            ensure_ascii=False,
        ),
        flush=True,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    _validate_trl_created_reference_adapter(model)
    if validate_model_load:
        return {
            **preflight,
            "dry_run": False,
            "validate_model_load": True,
            "initial_policy_validation": initial_policy_validation,
            "reference_adapter_validation": "passed",
        }
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
        "initial_policy_validation": initial_policy_validation,
        "train_metrics": train_result.metrics,
        "library_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
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
    rows = read_jsonl(source_path)
    source_artifact_audit: dict[str, Any] | None = None
    source_manifest_path = source_path.with_name("trajectory_dpo_manifest.json")
    if source_manifest_path.is_file():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        context_config = _config_for_recorded_collection(config, source_manifest)
        source_artifact_audit = _validate_dpo_source_artifact(
            source_path=source_path,
            rows=rows,
            source_manifest=source_manifest,
            current_workflow_hash=dpo_construction_workflow_hash(context_config),
            current_template_pool_hash=template_pool_hash(load_template_pool(config)),
            current_prompt_hashes=_current_dpo_prompt_hashes(config, rows),
            model_name_or_path=settings["model_name_or_path"],
        )
    validated = [_validated_dpo_row(row) for row in rows]
    if not validated:
        raise ValueError("Trajectory DPO training requires at least one preference pair.")
    if source_artifact_audit is not None:
        source_artifact_audit = {
            **source_artifact_audit,
            "canonical_pairs_verified": len(validated),
        }
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
        "source_artifact_audit": source_artifact_audit,
    }
    manifest_path = output_dir / "trajectory_dpo_split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest_path": str(manifest_path)}


def _validate_dpo_source_artifact(
    *,
    source_path: Path,
    rows: list[dict[str, Any]],
    source_manifest: dict[str, Any],
    current_workflow_hash: str,
    current_template_pool_hash: str,
    current_prompt_hashes: dict[str, str],
    model_name_or_path: str,
) -> dict[str, Any]:
    """Validate a frozen DPO export without coupling training to unrelated code."""
    recorded_output_hash = str(source_manifest.get("output_sha256") or "").strip()
    actual_output_hash = sha256_text(source_path.read_text(encoding="utf-8"))
    if not recorded_output_hash or recorded_output_hash != actual_output_hash:
        raise RuntimeError(
            "The exported trajectory DPO pair file does not match its manifest: "
            f"recorded output_sha256={recorded_output_hash or None!r}, "
            f"actual={actual_output_hash!r}."
        )

    recorded_pair_count = source_manifest.get("pairs")
    if recorded_pair_count is not None and int(recorded_pair_count) != len(rows):
        raise RuntimeError(
            "The exported trajectory DPO pair count does not match its manifest: "
            f"recorded={recorded_pair_count}, actual={len(rows)}."
        )

    recorded_template_hash = str(
        source_manifest.get("template_pool_hash") or ""
    ).strip()
    if recorded_template_hash != current_template_pool_hash:
        raise RuntimeError(
            "Trajectory DPO pairs were evaluated with a different template pool: "
            f"recorded={recorded_template_hash or None!r}, "
            f"current={current_template_pool_hash!r}."
        )

    source_checkpoint = str(source_manifest.get("source_checkpoint") or "").strip()
    if not source_checkpoint:
        raise RuntimeError(
            "The trajectory DPO manifest has no source SFT checkpoint."
        )
    if not _same_model_source(model_name_or_path, source_checkpoint):
        raise RuntimeError(
            "Trajectory preferences were collected with SFT checkpoint "
            f"{source_checkpoint}, but DPO training was asked to update "
            f"{model_name_or_path}. Use the same checkpoint."
        )

    role_models = {
        field: str(source_manifest.get(field) or "").strip()
        for field in ("planner_model", "executor_model", "reviewer_model")
    }
    missing_roles = sorted(field for field, value in role_models.items() if not value)
    if missing_roles:
        raise RuntimeError(
            "The trajectory DPO manifest is missing model-role provenance: "
            f"{missing_roles}."
        )

    prompt_hash_errors = [
        str(row.get("id") or "<missing-id>")
        for row in rows
        if str(row.get("prompt_hash") or "")
        != sha256_text(str(row.get("prompt") or ""))
    ]
    if prompt_hash_errors:
        raise RuntimeError(
            "The exported trajectory DPO data contains prompt/hash mismatches; "
            f"count={len(prompt_hash_errors)}, first={prompt_hash_errors[:5]}."
        )

    current_prompt_errors = [
        str(row.get("id") or "<missing-id>")
        for row in rows
        if current_prompt_hashes.get(str(row.get("id") or ""))
        != str(row.get("prompt_hash") or "")
    ]
    if current_prompt_errors:
        raise RuntimeError(
            "The exported trajectory DPO prompts do not match the current "
            "planner prompt and case inputs; "
            f"count={len(current_prompt_errors)}, first={current_prompt_errors[:5]}."
        )

    recorded_workflow_hash = str(
        source_manifest.get("workflow_hash") or ""
    ).strip()
    if not recorded_workflow_hash:
        raise RuntimeError("The trajectory DPO manifest has no workflow hash.")
    workflow_matches = recorded_workflow_hash == current_workflow_hash
    compatibility_mode = (
        "exact_workflow" if workflow_matches else "verified_immutable_export"
    )
    if not workflow_matches:
        warnings.warn(
            "The current LegalFlux workflow hash differs from the workflow that "
            "exported the DPO pairs. Proceeding with the immutable export because "
            "its file hash, prompt hashes, template pool, source checkpoint, and "
            "model-role provenance all passed validation. Both workflow hashes "
            "will be preserved in the training manifest.",
            UserWarning,
        )

    return {
        "status": "verified",
        "compatibility_mode": compatibility_mode,
        "pair_count": len(rows),
        "output_sha256": actual_output_hash,
        "prompt_hashes_verified": len(rows),
        "current_prompt_hashes_verified": len(rows),
        "template_pool_hash": current_template_pool_hash,
        "recorded_workflow_hash": recorded_workflow_hash,
        "current_workflow_hash": current_workflow_hash,
        "workflow_hash_matches": workflow_matches,
        "source_checkpoint": source_checkpoint,
        **role_models,
    }


def _current_dpo_prompt_hashes(
    config: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, str]:
    identifiers = [str(row.get("id") or "").strip() for row in rows]
    if any(not identifier for identifier in identifiers):
        raise RuntimeError("Every trajectory DPO pair must have a nonempty id.")
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("Trajectory DPO pair ids must be unique.")

    cases = {(case.case_id, case.variant_id): case for case in load_cases(config)}
    hashes: dict[str, str] = {}
    missing_cases: list[str] = []
    max_steps = int(config["legal_flux"].get("max_steps", 4))
    for identifier, row in zip(identifiers, rows, strict=True):
        case_key = (
            str(row.get("case_id") or ""),
            str(row.get("variant_id") or "original"),
        )
        case = cases.get(case_key)
        if case is None:
            missing_cases.append(identifier)
            continue
        _, hashes[identifier] = render_prompt(
            config,
            "legal_flux/rf_plan",
            case,
            max_steps=max_steps,
        )
    if missing_cases:
        raise RuntimeError(
            "Could not reconstruct the current planner prompt for exported DPO "
            f"pairs; missing cases={len(missing_cases)}, first={missing_cases[:5]}."
        )
    return hashes


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
        "sync_ref_model": False,
    }


def _dpo_policy_adapter_source(model_name_or_path: str) -> str:
    """Prefer the prepared text-only SFT adapter for causal-LM DPO."""
    checkpoint = Path(model_name_or_path).expanduser()
    if not checkpoint.is_dir():
        return model_name_or_path
    text_adapter = checkpoint / "vllm_text_only"
    required = (
        text_adapter / "adapter_config.json",
        text_adapter / "adapter_model.safetensors",
    )
    if all(path.is_file() for path in required):
        return str(text_adapter.resolve())
    return str(checkpoint.resolve())


def _dpo_adapter_key_mapping(adapter_keys: list[str]) -> dict[str, str] | None:
    """Map Qwen3.5 conditional-wrapper text keys onto its causal-LM model."""
    if any(".language_model." in key for key in adapter_keys):
        return {r"^(?:model\.)?language_model\.": "model."}
    return None


def _validate_loaded_policy_state(
    *,
    source_state: dict[str, Any],
    loaded_state: dict[str, Any],
    adapter_source: str,
) -> dict[str, Any]:
    """Require every source SFT tensor to load exactly before DPO starts."""
    source_keys = set(source_state)
    loaded_keys = set(loaded_state)
    missing = sorted(source_keys - loaded_keys)
    unexpected = sorted(loaded_keys - source_keys)
    if missing or unexpected:
        hint = (
            " Run `flux-prepare-vllm-adapter` on the original SFT checkpoint "
            "and use its text-only child for DPO."
        )
        raise RuntimeError(
            "The loaded DPO policy does not have the same adapter tensors as "
            f"{adapter_source}: missing={len(missing)}, "
            f"unexpected={len(unexpected)}."
            + hint
        )

    mismatched: list[str] = []
    nonzero_lora_b = 0
    lora_b_tensors = 0
    for key in sorted(source_keys):
        expected = source_state[key].detach().cpu()
        actual = loaded_state[key].detach().cpu()
        if expected.shape != actual.shape:
            mismatched.append(key)
            continue
        expected = expected.to(dtype=actual.dtype)
        if not expected.equal(actual):
            mismatched.append(key)
        if ".lora_B." in key:
            lora_b_tensors += 1
            if bool(expected.count_nonzero()):
                nonzero_lora_b += 1
    if mismatched:
        raise RuntimeError(
            "The selected SFT policy adapter was not loaded exactly before "
            f"DPO; {len(mismatched)} tensors differ. First examples: "
            f"{mismatched[:5]}"
        )
    if not lora_b_tensors or not nonzero_lora_b:
        raise RuntimeError(
            "The selected SFT policy has no nonzero LoRA-B tensors and is "
            "equivalent to a newly initialized adapter. Refusing to start DPO."
        )
    return {
        "adapter_source": adapter_source,
        "tensor_count": len(source_keys),
        "lora_b_tensor_count": lora_b_tensors,
        "nonzero_lora_b_tensor_count": nonzero_lora_b,
        "all_tensors_exact": True,
    }


def _validate_initial_sft_policy_adapter(model: Any) -> None:
    """Require the original SFT adapter expected by TRL 0.29's copy path."""
    adapters = getattr(model, "peft_config", {})
    if DPO_POLICY_ADAPTER_NAME not in adapters:
        raise RuntimeError(
            "The selected SFT checkpoint did not load the expected policy adapter "
            f"{DPO_POLICY_ADAPTER_NAME!r}; available adapters: {sorted(adapters)}."
        )
    if DPO_REFERENCE_ADAPTER_NAME in adapters:
        raise RuntimeError(
            f"Reference adapter name {DPO_REFERENCE_ADAPTER_NAME!r} is already "
            "present in "
            "the selected checkpoint. Use the original SFT checkpoint, not a DPO "
            "output checkpoint."
        )
    policy_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if f".{DPO_POLICY_ADAPTER_NAME}." in name
    ]
    if not policy_parameters:
        raise RuntimeError("Could not identify the selected SFT policy parameters.")
    if not any(parameter.requires_grad for parameter in policy_parameters):
        raise RuntimeError("The selected SFT policy adapter has no trainable parameters.")


def _validate_trl_created_reference_adapter(model: Any) -> None:
    """Verify TRL copied the initial policy into a separate frozen adapter."""
    adapters = getattr(model, "peft_config", {})
    required = {DPO_POLICY_ADAPTER_NAME, DPO_REFERENCE_ADAPTER_NAME}
    if not required.issubset(adapters):
        raise RuntimeError(
            "TRL did not create both required DPO adapters; available adapters: "
            f"{sorted(adapters)}."
        )
    parameters = dict(model.named_parameters())
    policy_names = [
        name
        for name in parameters
        if f".{DPO_POLICY_ADAPTER_NAME}." in name
    ]
    reference_names = [
        name
        for name in parameters
        if f".{DPO_REFERENCE_ADAPTER_NAME}." in name
    ]
    if not policy_names or not reference_names:
        raise RuntimeError(
            "Could not identify both the policy and TRL-created reference "
            "adapter parameters."
        )
    if not any(parameters[name].requires_grad for name in policy_names):
        raise RuntimeError("The DPO policy adapter has no trainable parameters.")
    if any(parameters[name].requires_grad for name in reference_names):
        raise RuntimeError("The DPO reference adapter must remain frozen.")
    for policy_name in policy_names:
        reference_name = policy_name.replace(
            f".{DPO_POLICY_ADAPTER_NAME}.",
            f".{DPO_REFERENCE_ADAPTER_NAME}.",
        )
        if reference_name not in parameters:
            raise RuntimeError(
                "TRL's DPO reference adapter is missing the counterpart of "
                f"{policy_name}."
            )
        policy_value = parameters[policy_name].detach()
        reference_value = parameters[reference_name].detach()
        if not policy_value.equal(reference_value):
            raise RuntimeError(
                "TRL's DPO reference adapter was not initialized identically "
                "to the selected SFT policy."
            )


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
