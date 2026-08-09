from __future__ import annotations

import json

import pytest

from legal_pilot.legal_flux_sft import (
    _is_vision_adapter_key,
    _lora_module_name,
    prepare_vllm_text_adapter,
)


def test_vllm_text_adapter_key_classification():
    text_key = "base_model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
    vision_key = "base_model.model.visual.blocks.0.attn.qkv.lora_B.weight"

    assert not _is_vision_adapter_key(text_key)
    assert _is_vision_adapter_key(vision_key)
    assert _lora_module_name(text_key) == "q_proj"
    assert _lora_module_name(vision_key) == "qkv"


def test_prepare_vllm_text_adapter_removes_only_vision_tensors(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    checkpoint = tmp_path / "checkpoint-30"
    checkpoint.mkdir()
    config = {
        "r": 2,
        "target_modules": ["q_proj", "qkv"],
        "task_type": "CAUSAL_LM",
    }
    (checkpoint / "adapter_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    text_prefix = "base_model.model.language_model.layers.0.self_attn.q_proj"
    vision_prefix = "base_model.model.visual.blocks.0.attn.qkv"
    source_tensors = {
        f"{text_prefix}.lora_A.weight": torch.ones((2, 3)),
        f"{text_prefix}.lora_B.weight": torch.ones((3, 2)),
        f"{vision_prefix}.lora_A.weight": torch.ones((2, 3)),
        f"{vision_prefix}.lora_B.weight": torch.zeros((3, 2)),
    }
    safetensors_torch.save_file(
        source_tensors, checkpoint / "adapter_model.safetensors"
    )

    result = prepare_vllm_text_adapter(checkpoint)

    output = checkpoint / "vllm_text_only"
    prepared_tensors = safetensors_torch.load_file(
        output / "adapter_model.safetensors"
    )
    prepared_config = json.loads(
        (output / "adapter_config.json").read_text(encoding="utf-8")
    )
    assert set(prepared_tensors) == {
        f"{text_prefix}.lora_A.weight",
        f"{text_prefix}.lora_B.weight",
    }
    assert prepared_config["target_modules"] == ["q_proj"]
    assert result["retained_tensor_count"] == 2
    assert result["removed_tensor_count"] == 2
    assert result["removed_lora_b_max_abs"] == 0.0
    assert set(safetensors_torch.load_file(checkpoint / "adapter_model.safetensors")) == set(
        source_tensors
    )


def test_prepare_vllm_text_adapter_refuses_to_overwrite(tmp_path):
    checkpoint = tmp_path / "checkpoint-30"
    output = checkpoint / "vllm_text_only"
    output.mkdir(parents=True)
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"not-read")

    with pytest.raises(FileExistsError):
        prepare_vllm_text_adapter(checkpoint)
