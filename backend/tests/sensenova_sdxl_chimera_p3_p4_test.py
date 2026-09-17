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
from core.training.chimera_prefix_prefetch import (
    ChimeraPrefixPrefetcher,
    prefix_nbytes,
)
from core.training.arch import ARCH_REGISTRY
from core.training.base_trainer import BaseTrainer
from core.training.base_trainer import resident_fused_backward_eligible
from core.training.ops.sensenova_sdxl_chimera_ops import (
    bridge_alignment_loss,
    collate_aux,
    sync_training_stage,
    train_step,
    training_stage_for_step,
)
from core.training.train_runner import _apply_chimera_training_contract
from core.models.sensenova_sdxl_chimera.understanding import UnderstandingPrefix


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
        alignment_tokens=3,
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


@pytest.mark.parametrize("target", ["unet", "joint"])
def test_staged_bridge_alignment_switches_exactly_at_completed_step(target):
    trainer = _trainer(target)
    trainer.config["chimera_bridge_align_steps"] = 3
    adapter = SenseNovaSDXLChimeraFullParameterAdapter(trainer)
    adapter.prepare_models_for_training()

    # Optimizer setup runs before the first sync. Future-stage parameters need
    # requires_grad here so fused-backward hooks are installed for the switch.
    assert all(p.requires_grad for p in trainer.unet.parameters())
    assert all(p.requires_grad for p in trainer.condition_bridge.parameters())
    assert training_stage_for_step(trainer, 0) == "bridge_align"
    assert training_stage_for_step(trainer, 2) == "bridge_align"
    assert training_stage_for_step(trainer, 3) == target
    assert [group["name"] for group in adapter.arch_param_groups()] == [
        "unet", "condition_bridge"
    ]

    sync_training_stage(trainer, 0)
    assert not any(p.requires_grad for p in trainer.unet.parameters())
    assert all(p.requires_grad for p in trainer.condition_bridge.parameters())
    sync_training_stage(trainer, 3)
    assert all(p.requires_grad for p in trainer.unet.parameters())
    assert all(p.requires_grad is (target == "joint")
               for p in trainer.condition_bridge.parameters())


def test_chimera_full_finetune_is_resident_fused_backward_eligible():
    trainer = SimpleNamespace(
        is_sensenova=False,
        is_sensenova_sdxl_chimera=True,
        trains_base_weights=True,
        num_optimizer_groups=0,
        config={},
    )
    assert resident_fused_backward_eligible(trainer, "lion8bit_ringbuffer")
    assert resident_fused_backward_eligible(trainer, "adamw8bit_ringbuffer")
    assert not resident_fused_backward_eligible(trainer, "lion")
    trainer.num_optimizer_groups = 2
    assert not resident_fused_backward_eligible(trainer, "lion8bit_ringbuffer")


def test_staged_training_contract_sets_live_encoding_and_requires_alignment_weights():
    config = {
        "chimera_training_stage": "unet",
        "chimera_bridge_align_steps": 8,
        "gradient_accumulation_steps": 2,
        "chimera_clip_hidden_weight": 0.5,
        "chimera_clip_pooled_weight": 0.25,
        "chimera_conditioning_cache": True,
    }
    with patch(
        "core.model_loader.ModelLoader.detect_model_type",
        return_value="sensenova_sdxl_chimera",
    ):
        assert _apply_chimera_training_contract("artifact", "full_finetune", config)
    assert config["text_encoding_mode"] == "onthefly_gpu"
    assert config["chimera_bridge_align_steps"] == 8


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"chimera_training_stage": "bridge_align", "chimera_bridge_align_steps": 2},
         "already remains"),
        ({"chimera_training_stage": "unet", "chimera_bridge_align_steps": 3,
          "gradient_accumulation_steps": 2}, "divisible"),
        ({"chimera_training_stage": "joint", "chimera_bridge_align_steps": 10,
          "total_steps": 10}, "smaller than total_steps"),
    ],
)
def test_staged_training_contract_refuses_invalid_boundaries(updates, match):
    config = {
        "chimera_clip_hidden_weight": 0.5,
        "chimera_clip_pooled_weight": 0.25,
        **updates,
    }
    with patch(
        "core.model_loader.ModelLoader.detect_model_type",
        return_value="sensenova_sdxl_chimera",
    ), pytest.raises(ValueError, match=match):
        _apply_chimera_training_contract("artifact", "full_finetune", config)


def test_prefix_prefetch_keeps_raw_kv_and_reuses_null_prefix(monkeypatch):
    def capture(_transformer, _tokenizer, prompt, selected_layers):
        length = len(prompt) + 1
        return UnderstandingPrefix(
            hidden_states=torch.full((1, length, 8), float(length)),
            layer_kv={
                layer: (
                    torch.full((1, 2, length, 2), float(layer)),
                    torch.full((1, 2, length, 2), float(layer + 1)),
                )
                for layer in selected_layers
            },
            attention_mask=torch.ones(1, length, dtype=torch.bool),
            positions=torch.zeros(1, length, 3),
        )

    monkeypatch.setattr(
        "core.training.chimera_prefix_prefetch.capture_chimera_prompt_prefix",
        capture,
    )
    batches = [
        [({"caption": "cat", "image_path": "a"}, None)],
        [({"caption": "long cat", "image_path": "b"}, None)],
    ]
    prefetcher = ChimeraPrefixPrefetcher(
        transformer=_module(), tokenizer=object(), selected_layers=(1, 3),
        batches=batches, device="cpu", depth=1,
    )
    prefetcher.start()
    prefetcher.activate_batch(0)
    cat = prefetcher.take("cat", "cpu")
    assert cat is not None and cat.hidden_states.shape == (1, 4, 8)
    assert not torch.is_inference(cat.hidden_states)
    bridge = ConditioningBridge(ChimeraBridgeConfig(
        hidden_size=8, kv_width=4, selected_layers=(1, 3),
        context_dim=16, pooled_dim=8, bridge_dim=8, num_heads=2,
        alignment_tokens=3,
    ))
    bridge(
        cat.hidden_states, cat.layer_kv, cat.attention_mask, cat.positions
    ).encoder_hidden_states.square().mean().backward()
    assert any(parameter.grad is not None for parameter in bridge.parameters())
    assert set(cat.layer_kv) == {1, 3}
    assert prefix_nbytes(cat) > cat.hidden_states.numel() * cat.hidden_states.element_size()
    assert prefetcher.take("changed caption", "cpu") is None
    null_first = prefetcher.take("", "cpu")
    prefetcher.activate_batch(1)
    assert prefetcher.take("long cat", "cpu") is not None
    assert prefetcher.take("", "cpu") is null_first
    prefetcher.stop()
    assert prefetcher.stats.hits == 4
    assert prefetcher.stats.misses == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prefix_prefetch_cuda_stream_hands_complete_prefix_to_main(monkeypatch):
    def capture(transformer, _tokenizer, prompt, selected_layers):
        anchor = next(transformer.parameters())
        length = len(prompt) + 1
        hidden = torch.ones(1, length, 8, device=anchor.device) * anchor.flatten()[0]
        return UnderstandingPrefix(
            hidden_states=hidden,
            layer_kv={layer: (hidden.view(1, 2, length, 4),
                              hidden.view(1, 2, length, 4) + 1)
                      for layer in selected_layers},
            attention_mask=torch.ones(1, length, dtype=torch.bool, device=anchor.device),
            positions=torch.zeros(1, length, 3, device=anchor.device),
        )

    monkeypatch.setattr(
        "core.training.chimera_prefix_prefetch.capture_chimera_prompt_prefix",
        capture,
    )
    prefetcher = ChimeraPrefixPrefetcher(
        transformer=_module(), tokenizer=object(), selected_layers=(1,),
        batches=[[({"caption": "gpu", "image_path": "a"}, None)]],
        device="cuda", depth=1,
    )
    prefetcher.start()
    prefetcher.activate_batch(0)
    prefix = prefetcher.take("gpu", "cuda")
    assert prefix is not None and prefix.hidden_states.device.type == "cuda"
    assert torch.isfinite(prefix.hidden_states).all()
    prefetcher.stop()


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
        "alignment_hidden_states": student,
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


def test_variable_context_aux_collation_right_pads_positions_and_mask():
    def item(length):
        return {
            "pooled_text_embeds": torch.randn(1, 5),
            "alignment_hidden_states": torch.randn(1, 3, 6),
            "context_positions": torch.randn(1, length, 3),
            "context_attention_mask": torch.ones(1, length, dtype=torch.bool),
        }

    result = collate_aux([item(2), item(5)])
    assert result["context_positions"].shape == (2, 5, 3)
    assert result["context_attention_mask"].shape == (2, 5)
    assert result["context_attention_mask"].tolist() == [
        [True, True, False, False, False],
        [True, True, True, True, True],
    ]
    assert torch.count_nonzero(result["context_positions"][0, 2:]) == 0
    assert result["alignment_hidden_states"].shape == (2, 3, 6)


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
            "context_attention_mask": torch.ones(1, 3, dtype=torch.bool),
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
            "context_attention_mask": torch.ones(1, 3, dtype=torch.bool),
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


def test_chimera_diffusion_step_writes_debug_latents(tmp_path):
    unet = _tiny_unet().train()
    trainer = SimpleNamespace(
        config={"chimera_training_stage": "unet"},
        device=torch.device("cpu"),
        training_dtype=torch.float32,
        unet=unet,
        repa_enable=False,
        log_prefix="[test]",
        log_extra_metric=lambda *_args: None,
    )
    debug_dir = tmp_path / "step_001000"
    ctx = SimpleNamespace(
        text_embeddings=torch.randn(1, 3, 6),
        attention_mask={
            "pooled_text_embeds": torch.randn(1, 5),
            "context_positions": torch.zeros(1, 3, 3),
            "context_attention_mask": torch.ones(1, 3, dtype=torch.bool),
        },
        latents=torch.randn(1, 4, 8, 8),
        timesteps=torch.tensor([0.5]),
        time_ids=None,
        repa_pixels=None,
        debug_save_path=debug_dir,
        debug_captions=["native prefix"],
        debug_reference_image_paths=[None],
    )

    train_step(trainer, ctx)

    saved = torch.load(debug_dir / "latents_t0.5000.pt", map_location="cpu")
    assert saved["model_type"] == "sensenova_sdxl_chimera"
    assert saved["prediction_type"] == "flow_velocity"
    assert saved["caption"] == "native prefix"
    assert saved["latents"].shape == saved["noisy_latents"].shape
    assert saved["predicted_latent"].shape == saved["latents"].shape


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
