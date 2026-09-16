"""CPU contracts for Chimera bridge alignment and full U-Net training."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from core.models.sensenova_sdxl_chimera.conditioning_bridge import (
    ChimeraBridgeConfig,
    ConditioningBridge,
)
from core.training.adapters.sensenova_sdxl_chimera_adapter import (
    SenseNovaSDXLChimeraFullParameterAdapter,
)
from core.training.ops.sensenova_sdxl_chimera_ops import bridge_alignment_loss


def _module() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.SiLU(), torch.nn.Linear(8, 4))


def _trainer(stage: str):
    bridge = ConditioningBridge(ChimeraBridgeConfig(
        hidden_size=8,
        kv_width=4,
        selected_layers=(0,),
        context_tokens=3,
        context_dim=6,
        pooled_dim=5,
        bridge_dim=8,
        num_heads=2,
    ))
    return SimpleNamespace(
        config={
            "chimera_training_stage": stage,
            "chimera_clip_hidden_weight": 0.5,
            "chimera_clip_pooled_weight": 0.25,
        },
        chimera_understanding=_module(),
        vae=_module(),
        unet=_module(),
        condition_bridge=bridge,
        chimera_teacher=None,
        unet_lr=2e-5,
        chimera_bridge_lr=3e-5,
        learning_rate=1e-4,
    )


@pytest.mark.parametrize(
    "stage,unet_trainable,bridge_trainable,groups",
    [
        ("bridge_align", False, True, ["condition_bridge"]),
        ("unet", True, False, ["unet"]),
        ("joint", True, True, ["unet", "condition_bridge"]),
    ],
)
def test_stage_freezing_and_optimizer_groups(stage, unet_trainable, bridge_trainable, groups):
    trainer = _trainer(stage)
    adapter = SenseNovaSDXLChimeraFullParameterAdapter(trainer)
    adapter.prepare_models_for_training()

    assert all(p.requires_grad is unet_trainable for p in trainer.unet.parameters())
    assert all(p.requires_grad is bridge_trainable for p in trainer.condition_bridge.parameters())
    assert not any(p.requires_grad for p in trainer.chimera_understanding.parameters())
    assert not any(p.requires_grad for p in trainer.vae.parameters())
    actual = adapter.arch_param_groups()
    assert [group["name"] for group in actual] == groups
    assert [group["lr"] for group in actual] == [
        value for enabled, value in (
            (unet_trainable, 2e-5), (bridge_trainable, 3e-5)
        ) if enabled
    ]


def test_bridge_alignment_loss_updates_only_student_graph():
    trainer = _trainer("bridge_align")
    student = torch.randn(2, 3, 6, requires_grad=True)
    pooled = torch.randn(2, 5, requires_grad=True)
    teacher = torch.randn(2, 3, 6)
    teacher_pooled = torch.randn(2, 5)
    loss, metrics = bridge_alignment_loss(trainer, student, {
        "pooled_text_embeds": pooled,
        "teacher_hidden": teacher,
        "teacher_pooled": teacher_pooled,
    })
    loss.backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert pooled.grad is not None and torch.isfinite(pooled.grad).all()
    assert set(metrics) == {
        "hidden_mse", "hidden_rms_mse", "pooled_mse",
        "hidden_cosine", "pooled_cosine",
    }


def test_directory_checkpoint_entry_is_discoverable_and_sized(tmp_path):
    from core.training.base_trainer import (
        _checkpoint_aux_base,
        _checkpoint_member_files,
        _checkpoint_set_bytes,
        _list_checkpoint_entries,
    )

    entry = tmp_path / "run_step_000003"
    entry.mkdir()
    (entry / "chimera.json").write_text("{}", encoding="utf-8")
    (entry / "config.json").write_text("{}", encoding="utf-8")
    (entry / "model.safetensors").write_bytes(b"weights")
    assert _list_checkpoint_entries(tmp_path) == [entry]
    assert _checkpoint_aux_base(entry) == "run_step_000003"
    assert len(_checkpoint_member_files(entry)) == 3
    assert _checkpoint_set_bytes(entry) == sum(
        path.stat().st_size for path in entry.iterdir()
    )
