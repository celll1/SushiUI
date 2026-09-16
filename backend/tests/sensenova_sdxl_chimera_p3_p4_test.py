"""CPU contracts for Chimera bridge alignment and full U-Net training."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from diffusers import UNet2DConditionModel

from api.arch_capabilities import TRAINING_FEATURE_UNSUPPORTED
from core.models.sensenova_sdxl_chimera.conditioning_bridge import (
    ChimeraBridgeConfig,
    ConditioningBridge,
)
from core.models.sensenova_sdxl_chimera.attention_processor import ChimeraAttnProcessor
from core.training.adapters.sensenova_sdxl_chimera_adapter import (
    SenseNovaSDXLChimeraFullParameterAdapter,
)
from core.training import repa as repa_module
from core.training.arch import ARCH_REGISTRY
from core.training.base_trainer import BaseTrainer
from core.training.ops.sensenova_sdxl_chimera_ops import bridge_alignment_loss, train_step
from core.training.train_runner import _apply_chimera_training_contract


def _module() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.SiLU(), torch.nn.Linear(8, 4))


def _tiny_unet() -> UNet2DConditionModel:
    return UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(8, 16),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=6,
        attention_head_dim=2,
        norm_num_groups=4,
        addition_embed_type="text_time",
        addition_time_embed_dim=4,
        projection_class_embeddings_input_dim=29,
    )


class _RepaTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(()), requires_grad=False)

    def forward(self, pixel_values):
        batch = pixel_values.shape[0]
        features = torch.linspace(0, 1, 4 * 4 * 7, device=pixel_values.device)
        return SimpleNamespace(
            last_hidden_state=features.reshape(1, 16, 7).expand(batch, -1, -1)
        )


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


@pytest.mark.parametrize("stage", ["unet", "joint"])
def test_repa_uses_the_chimera_unet_spatial_menu(stage):
    unet = _tiny_unet()
    trainer = SimpleNamespace(
        config={"chimera_training_stage": stage, "repa_align_depth": -1},
        unet=unet,
        use_condition_images=False,
    )

    tap = ARCH_REGISTRY["sensenova_sdxl_chimera"]().repa_tap(trainer)

    assert tap.module is unet
    assert tap.depth == 3
    assert tap.site_labels == ("down_blocks[1]", "mid_block", "up_blocks[0]")


def test_capabilities_offer_repa_for_chimera():
    assert "repa" not in TRAINING_FEATURE_UNSUPPORTED.get(
        "sensenova_sdxl_chimera", {}
    )


def test_repa_refuses_bridge_only_training():
    trainer = SimpleNamespace(
        config={"chimera_training_stage": "bridge_align"},
        unet=_tiny_unet(),
        use_condition_images=False,
    )
    with pytest.raises(ValueError, match="freezes the U-Net"):
        ARCH_REGISTRY["sensenova_sdxl_chimera"]().repa_tap(trainer)


@pytest.mark.parametrize("stage", ["unet", "joint"])
def test_training_contract_accepts_pixel_repa_for_diffusion_stages(stage):
    config = {
        "chimera_training_stage": stage,
        "repa_enable": True,
        "repa_target_source": "pixel",
    }
    with patch(
        "core.model_loader.ModelLoader.detect_model_type",
        return_value="sensenova_sdxl_chimera",
    ):
        assert _apply_chimera_training_contract("artifact", "full_finetune", config)


@pytest.mark.parametrize(
    "stage,target,match",
    [
        ("bridge_align", "pixel", "freezes the U-Net"),
        ("unet", "latent_stem", "repa_target_source='pixel'"),
    ],
)
def test_training_contract_refuses_inapplicable_repa(stage, target, match):
    config = {
        "chimera_training_stage": stage,
        "repa_enable": True,
        "repa_target_source": target,
    }
    with patch(
        "core.model_loader.ModelLoader.detect_model_type",
        return_value="sensenova_sdxl_chimera",
    ), pytest.raises(ValueError, match=match):
        _apply_chimera_training_contract("artifact", "full_finetune", config)


def test_base_trainer_passes_repa_pixels_into_the_chimera_step():
    source = inspect.getsource(BaseTrainer._execute_forward_backward)
    chimera_branch = source.split(
        "elif self.is_sensenova_sdxl_chimera:", 1
    )[1].split("elif self.is_zimage:", 1)[0]

    assert "repa_pixels=mnt_repa_pixels" in chimera_branch


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


def test_unet_step_retains_attention_context_through_backward():
    class FakeUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))
            self.attn_processors = {"fake": ChimeraAttnProcessor()}

        def forward(self, sample, *_args, **_kwargs):
            return (sample * self.scale,)

    unet = FakeUNet()
    trainer = SimpleNamespace(
        config={"chimera_training_stage": "unet"},
        device="cpu",
        training_dtype=torch.float32,
        unet=unet,
    )
    ctx = SimpleNamespace(
        text_embeddings=torch.randn(1, 3, 6),
        attention_mask={
            "pooled_text_embeds": torch.randn(1, 5),
            "context_positions": torch.zeros(1, 3, 3),
        },
        latents=torch.randn(1, 4, 2, 2),
        timesteps=torch.tensor([0.5]),
        time_ids=None,
    )
    loss, _value, _recon = train_step(trainer, ctx)
    assert unet.attn_processors["fake"].context is not None
    loss.backward()
    assert unet.scale.grad is not None


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_repa_loss_reaches_chimera_unet_and_projector(gradient_checkpointing):
    unet = _tiny_unet().train()
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    site_width = repa_module.spatial_site_width(
        *repa_module.spatial_tap_sites(unet)[1]
    )
    projector = repa_module.RepaProjector(site_width, 7, hidden=12)
    logged = []
    trainer = SimpleNamespace(
        config={"chimera_training_stage": "unet"},
        device=torch.device("cpu"),
        training_dtype=torch.float32,
        unet=unet,
        repa_enable=True,
        repa_align_depth=1,
        repa_encoder=_RepaTeacher(),
        repa_projector=projector,
        repa_size=32,
        repa_weight=0.5,
        repa_target_source="pixel",
        _repa_tap_module=unet,
        _ensure_repa_on_device=lambda: None,
        log_extra_metric=lambda name, value: logged.append((name, value)),
    )
    ctx = SimpleNamespace(
        text_embeddings=torch.randn(1, 3, 6),
        attention_mask={
            "pooled_text_embeds": torch.randn(1, 5),
            "context_positions": torch.zeros(1, 3, 3),
        },
        latents=torch.randn(1, 4, 8, 8),
        timesteps=torch.tensor([0.5]),
        time_ids=None,
        repa_pixels=torch.rand(1, 3, 32, 32).mul(2).sub(1),
    )

    loss, diffusion_loss, _recon = train_step(trainer, ctx)
    loss.backward()

    assert float(loss.detach()) > diffusion_loss
    assert any(name == "repa_loss" for name, _value in logged)
    assert any(parameter.grad is not None for parameter in unet.parameters())
    assert all(parameter.grad is not None for parameter in projector.parameters())
    assert sum(len(module._forward_hooks) for module in unet.modules()) == 0
    assert unet._repa_tap_out is None


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
