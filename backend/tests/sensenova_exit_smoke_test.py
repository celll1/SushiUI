from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.probes.sensenova_real_checkpoint import (
    EXPECTED_TARGETS,
    _inspect_saved_lora,
    _parse_args,
    _take_denoise_tensor,
    trainer_exit_smoke_config,
)


def test_sensenova_trainer_exit_smoke_config_pins_the_phase1_contract():
    config = trainer_exit_smoke_config()

    constructor = config["constructor"]
    assert constructor["lora_rank"] == 1
    assert constructor["lora_alpha"] == 1
    assert constructor["lora_dtype"] == "fp32"
    assert constructor["weight_dtype"] == "bf16"
    assert constructor["training_dtype"] == "bf16"
    assert constructor["attention_backend"] == "native"
    assert constructor["blocks_to_swap"] == 0

    train_config = config["train_config"]
    assert train_config["gradient_checkpointing"] is True
    assert train_config["batch_size"] == 1
    assert train_config["multi_noise_timesteps"] == 1
    assert train_config["use_reference_images"] is False

    train = config["train"]
    assert train["total_steps"] == 3
    assert train["save_every_n_steps"] == 3
    assert train["sample_every_n_steps"] == 0
    assert train["enable_bucketing"] is False
    assert train["base_resolutions"] == [64]
    assert train["text_encoding_mode"] == "onthefly_gpu"
    assert train["latent_encoding_mode"] == "onthefly_gpu"


def test_sensenova_exit_smoke_inspector_accepts_the_real_882_tensor_contract(tmp_path):
    from safetensors.torch import save_file

    tensors = {}
    for index in range(EXPECTED_TARGETS):
        target = f"language_model.model.layers.{index // 7}.target_{index}"
        tensors[f"{target}.lora_down.weight"] = torch.ones((1, 1), dtype=torch.bfloat16)
        tensors[f"{target}.lora_up.weight"] = torch.ones((1, 1), dtype=torch.bfloat16)
        tensors[f"{target}.alpha"] = torch.tensor(1.0, dtype=torch.float32)
    path = tmp_path / "sensenova_exit_smoke_step_000003.safetensors"
    save_file(
        tensors,
        str(path),
        metadata={
            "tensor_kind": "neo_hf_lora",
            "model_type": "sensenova",
            "modelspec.architecture": "sensenova",
            "lora_targets": "generation",
            "lora_rank": "1",
            "lora_alpha": "1",
            "step": "3",
            "epoch": "2",
        },
    )

    inspected = _inspect_saved_lora(path)

    assert inspected["tensor_count"] == 882
    assert inspected["target_count"] == EXPECTED_TARGETS
    assert inspected["parameter_tensor_count"] == 588
    assert inspected["metadata"]["epoch"] == "2"
    assert inspected["finite"] is True
    assert len(inspected["tensor_sha256"]) == 64
    assert len(inspected["parameter_sha256"]) == 64


def test_sensenova_exit_smoke_parser_keeps_legacy_and_internal_modes(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["probe", "--model-path", "checkpoint", "--checkpointing", "on"],
    )
    legacy = _parse_args()
    assert legacy.checkpointing == "on"
    assert legacy.trainer_exit_smoke is False

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--model-path",
            "checkpoint",
            "--smoke-arm",
            "trainer",
            "--smoke-workdir",
            "work",
            "--smoke-arm-json",
            "result.json",
            "--smoke-cfg-scale",
            "1",
            "--smoke-timestep-shift",
            "1",
            "--smoke-cfg-norm",
            "none",
        ],
    )
    internal = _parse_args()
    assert internal.smoke_arm == "trainer"
    assert internal.checkpointing is None


def test_sensenova_runtime_result_can_be_serialized_without_the_tensor():
    result = {"width": 64, "denoise_tensor": torch.zeros(1), "lora_restored": 294}

    tensor = _take_denoise_tensor(result)

    assert torch.equal(tensor, torch.zeros(1))
    assert "denoise_tensor" not in result
    json.dumps(result)
