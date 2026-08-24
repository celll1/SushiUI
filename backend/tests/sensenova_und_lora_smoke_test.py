"""CPU contract tests for the Phase U-1 exit-smoke probe.

No checkpoint, no CUDA: these pin the parts of
``core.training.probes.sensenova_und_lora`` that decide WHAT gets measured, so a
drift in the contract fails here instead of an hour into a GPU arm.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.models.sensenova.sensenova_lora import und_gradient_unreachable_paths
from core.training.probes.sensenova_und_lora import (
    EXPECTED_BOTH_TARGETS,
    EXPECTED_BOTH_TENSORS,
    EXPECTED_LAYERS,
    EXPECTED_TARGETS,
    MNT_TOTAL_STEPS,
    MNT_VALUE,
    PRE_U1_COMMIT,
    _compare_regression,
    _dead_up_grad_targets,
    _inspect_saved_und_lora,
    _library_backend_root,
    _und_config,
)


def test_the_und_artefact_contract_is_1764_tensors_over_both_branches():
    assert EXPECTED_TARGETS == 294
    assert EXPECTED_BOTH_TARGETS == 588
    assert EXPECTED_BOTH_TENSORS == 1764


def test_und_config_keeps_the_phase1_geometry_and_only_moves_mnt_and_steps():
    baseline = _und_config(total_steps=3, mnt=1)
    assert baseline["train"]["base_resolutions"] == [64]
    assert baseline["train_config"]["batch_size"] == 1
    assert baseline["train_config"]["gradient_checkpointing"] is True
    assert baseline["train_config"]["sensenova_mot_phase_eviction"] is False
    assert baseline["train"]["total_steps"] == 3
    assert baseline["train"]["save_every_n_steps"] == 3

    mnt = _und_config(total_steps=MNT_TOTAL_STEPS, mnt=MNT_VALUE)
    assert mnt["train"]["multi_noise_timesteps"] == MNT_VALUE
    assert mnt["train_config"]["multi_noise_timesteps"] == MNT_VALUE
    assert mnt["train"]["total_steps"] == MNT_TOTAL_STEPS
    assert mnt["train"]["save_every_n_steps"] == MNT_TOTAL_STEPS
    # Everything else must be the SAME contract, or the MNT arm stops being a
    # controlled comparison against the 3-step one.
    assert mnt["constructor"] == baseline["constructor"]
    assert mnt["train"]["base_resolutions"] == baseline["train"]["base_resolutions"]


@pytest.mark.parametrize(
    "argv, expected_suffix",
    [
        (["probe", "--arm", "regression"], None),
        (["probe", "--library-root", "/tmp/old"], "old"),
        (["probe", "--library-root=/tmp/old"], "old"),
    ],
)
def test_library_root_is_read_from_argv_before_any_core_import(
    monkeypatch, argv, expected_suffix
):
    monkeypatch.setattr(sys, "argv", argv)
    root = _library_backend_root()
    assert root.name == "backend"
    if expected_suffix is None:
        assert root == Path(__file__).resolve().parents[1]
    else:
        assert root.parent.name == expected_suffix


def _write_und_lora(path: Path, *, step: int, targets, dead) -> None:
    from safetensors.torch import save_file

    state = {}
    for name in targets:
        state[f"{name}.lora_down.weight"] = torch.ones(1, 4)
        state[f"{name}.lora_up.weight"] = (
            torch.zeros(4, 1) if name in dead else torch.ones(4, 1)
        )
        state[f"{name}.alpha"] = torch.tensor(1.0)
    save_file(
        state,
        str(path),
        metadata={
            "tensor_kind": "neo_hf_lora",
            "model_type": "sensenova",
            "lora_targets": "generation+understanding",
            "lora_rank": "1",
            "lora_alpha": "1",
            "step": str(step),
            "epoch": "2",
        },
    )


def _both_branch_target_names() -> list[str]:
    names = []
    for index in range(EXPECTED_LAYERS):
        prefix = f"language_model.model.layers.{index}"
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            names.append(f"{prefix}.self_attn.{attr}_mot_gen")
            names.append(f"{prefix}.self_attn.{attr}")
        for attr in ("gate_proj", "up_proj", "down_proj"):
            names.append(f"{prefix}.mlp_mot_gen.{attr}")
            names.append(f"{prefix}.mlp.{attr}")
    return names


def test_saved_und_inspector_accepts_the_real_1764_tensor_contract(tmp_path):
    dead = und_gradient_unreachable_paths(EXPECTED_LAYERS)
    path = tmp_path / "und.safetensors"
    _write_und_lora(path, step=3, targets=_both_branch_target_names(), dead=dead)

    report = _inspect_saved_und_lora(path, expected_step=3)

    assert report["tensor_count"] == EXPECTED_BOTH_TENSORS
    assert report["target_count"] == EXPECTED_BOTH_TARGETS
    assert report["metadata"]["lora_targets"] == "generation+understanding"
    assert set(report["unreachable_und_targets"]) == dead
    assert all(
        entry["present"] and entry["lora_up_nonzero"] == 0
        for entry in report["unreachable_und_targets"].values()
    )


def test_saved_und_inspector_rejects_a_generation_only_file(tmp_path):
    generation = [
        name for name in _both_branch_target_names() if "mot_gen" in name
    ]
    path = tmp_path / "gen.safetensors"
    _write_und_lora(path, step=3, targets=generation, dead=set())

    with pytest.raises(AssertionError, match="LoRA tensors"):
        _inspect_saved_und_lora(path, expected_step=3)


class _FakeLoRALayer:
    def __init__(self, gradient):
        self.lora_up = torch.nn.Linear(1, 1, bias=False)
        self.lora_up.weight.grad = gradient


def test_dead_up_grad_targets_names_both_absent_and_all_zero_gradients():
    layers = {
        "reached": _FakeLoRALayer(torch.ones(1, 1)),
        "all_zero": _FakeLoRALayer(torch.zeros(1, 1)),
        "no_grad": _FakeLoRALayer(None),
    }

    assert _dead_up_grad_targets(layers) == ["all_zero", "no_grad"]


def _regression_record(loss: float, digest: str) -> dict:
    return {
        "losses": [loss],
        "grad_digests": [{"up": {"sha256": digest}, "down": {"sha256": digest}}],
        "lora_parameter_sha256": digest,
        "checkpoint": {"tensor_sha256": digest, "tensor_count": EXPECTED_TARGETS * 3},
        "library": {"sensenova_ops": "unused"},
    }


def test_regression_comparison_is_bit_exact_not_approximate():
    same = _compare_regression(
        _regression_record(0.5, "aa"), _regression_record(0.5, "aa")
    )
    assert same["losses_bit_exact"] is True
    assert same["grad_sha256_bit_exact"] is True
    assert same["lora_parameter_sha256_equal"] is True
    assert same["saved_tensor_sha256_equal"] is True

    # One ULP apart is a FAILURE here: the criterion is bit-exactness.
    drifted = _compare_regression(
        _regression_record(0.5, "aa"),
        _regression_record(0.5 + 2 ** -52, "aa"),
    )
    assert drifted["losses_bit_exact"] is False

    rehashed = _compare_regression(
        _regression_record(0.5, "aa"), _regression_record(0.5, "bb")
    )
    assert rehashed["grad_sha256_bit_exact"] is False


def test_the_regression_baseline_is_the_commit_immediately_before_u1():
    import subprocess

    repo = Path(__file__).resolve().parents[2]
    resolved = subprocess.run(
        ["git", "rev-parse", PRE_U1_COMMIT],
        cwd=str(repo), capture_output=True, text=True, check=True,
    ).stdout.strip()
    subject = subprocess.run(
        ["git", "log", "-1", "--format=%s", resolved],
        cwd=str(repo), capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert subject == "Prove the understanding branch is reachable by gradient"


def test_arm_results_stay_json_serialisable():
    # The driver writes every arm result with ``default=str``; Path objects are
    # the only non-JSON type any arm returns, and they must survive.
    payload = {"checkpoint_path": Path("a/b.safetensors"), "loss": 0.5}
    assert json.loads(json.dumps(payload, default=str))["checkpoint_path"].endswith(
        "b.safetensors"
    )
