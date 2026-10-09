"""Aligned halos retain exact core coverage and deterministic resume plans."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = lambda *a, **k: None
torch._C._cuda_init = lambda *a, **k: None

from core.models.anima.partition_training import region_count, training_plan, backward_regions
from core.models.anima.checkpointing import configure, partition_scope


def trainer(**config):
    spec = {"regions": [
        {"id": "r0", "parent": None, "down_factor": [2, 2]},
        {"id": "r1", "parent": "r0", "down_factor": [2, 2]}]}
    return SimpleNamespace(config={"dit_partition_training_enabled": True,
        "dit_partition_mode": "fixed", "dit_partition_fixed_count": 4,
        "dit_partition_split_ratio_min": .35, "dit_partition_split_ratio_max": .65,
        **config}, transformer=SimpleNamespace(_residual_unet_spec=spec),
        _current_epoch=3, _current_batch_position=19)


@pytest.mark.parametrize("count", [2, 4])
@pytest.mark.parametrize("halo", [0, 2])
def test_aligned_inputs_keep_single_core_coverage(count, halo):
    owner = trainer(dit_partition_fixed_count=count, dit_partition_halo_tokens=halo)
    clean = torch.zeros(2, 64, 48, 80)
    plan = training_plan(owner, clean)
    assert plan == training_plan(owner, clean)
    assert plan == training_plan(trainer(**owner.config), clean)
    coverage = torch.zeros(24, 40)
    for region in plan.regions:
        core, box = region.core, region.input
        assert all(value % 4 == 0 for value in (box.top, box.left, box.height, box.width))
        coverage[core.top:core.bottom, core.left:core.right] += 1
    assert torch.equal(coverage, torch.ones_like(coverage))
    assert sum(r.core.tokens for r in plan.regions) == plan.full_tokens


def test_threshold_uses_replacement_codec_grid_and_not_batch_size():
    owner = trainer(dit_partition_mode="token_threshold", dit_partition_full_frame_max_tokens=256)
    assert region_count(owner, torch.zeros(4, 64, 32, 32)) == 1
    assert region_count(owner, torch.zeros(1, 16, 32, 64)) == 2
    assert region_count(owner, torch.zeros(1, 64, 64, 64)) == 4
    with pytest.raises(ValueError, match="even"):
        region_count(owner, torch.zeros(1, 64, 31, 32))
    with pytest.raises(ValueError, match="alignment"):
        training_plan(trainer(), torch.zeros(1, 64, 50, 80))


def test_checkpoint_prefix_keeps_offload_and_restores_after_exception():
    model = SimpleNamespace(blocks=[SimpleNamespace(gradient_checkpoint_mode="async_cpu_offload")
                                   for _ in range(4)])
    config = {"dit_gradient_checkpointing_blocks": 2,
              "dit_partition_gradient_checkpointing_blocks": 3}
    configure(model, config)
    modes = lambda: [b.gradient_checkpoint_mode for b in model.blocks]
    assert modes() == ["async_cpu_offload"] * 2 + ["none"] * 2
    with pytest.raises(RuntimeError, match="backward failed"):
        with partition_scope(model, config):
            assert modes() == ["async_cpu_offload"] * 3 + ["none"]
            with partition_scope(model, {"dit_partition_gradient_checkpointing_blocks": 1}):
                assert modes() == ["async_cpu_offload"] + ["none"] * 3
            assert modes() == ["async_cpu_offload"] * 3 + ["none"]
            raise RuntimeError("backward failed")
    assert modes() == ["async_cpu_offload"] * 2 + ["none"] * 2
    with pytest.raises(ValueError, match="between"):
        configure(model, {"dit_gradient_checkpointing_blocks": 5})


def test_conflicting_legacy_partition_settings_fail():
    with pytest.raises(ValueError, match="conflicts"):
        region_count(trainer(qwen_partition_fixed_count=2), torch.zeros(1, 64, 48, 80))


def test_swapped_tail_blocks_override_checkpoint_prefix_limit():
    model = SimpleNamespace(blocks=[SimpleNamespace(gradient_checkpoint_mode="async_cpu_offload")
                                   for _ in range(8)])
    configure(model, {"dit_gradient_checkpointing_blocks": 2, "blocks_to_swap": 3})
    assert [b.gradient_checkpoint_mode for b in model.blocks] == (
        ["async_cpu_offload"]*2 + ["none"]*3 + ["async_cpu_offload"]*3)
    model.blocks[-1].gradient_checkpoint_mode = "none"
    with pytest.raises(ValueError, match="require gradient checkpointing"):
        configure(model, {"blocks_to_swap": 1})


@pytest.mark.parametrize("extra, message", [
    ({"gradient_checkpointing": False}, "requires gradient checkpointing"),
    ({"dit_partition_training_enabled": True}, "conventional backward"),
    ({"use_ema": True}, "does not support EMA")])
def test_api_rejects_anima_fused_backward_combinations(extra, message):
    from api.routes import TrainingRunCreateRequest
    with pytest.raises(ValueError, match=message):
        TrainingRunCreateRequest(dataset_id=1, run_name="invalid_swap",
            base_model_path="M:/model/anima/anima-base-v1.0.safetensors",
            training_method="full_finetune", total_steps=1, blocks_to_swap=2, **extra)


def test_api_refuses_global_without_partition_before_config_write():
    from api.routes import TrainingRunCreateRequest
    with pytest.raises(ValueError, match="require partition"):
        TrainingRunCreateRequest(dataset_id=1, run_name="invalid_global",
            base_model_path="M:/model/anima/anima-base-v1.0.safetensors",
            training_method="full_finetune", total_steps=1,
            dit_partition_global_adapter_enabled=True,
            dit_partition_training_enabled=False)


def test_api_config_keeps_anima_partition_global_and_checkpoint_settings():
    from unittest.mock import patch
    from api.routes import TrainingRunCreateRequest
    from core.training.training_config import _build_train_section
    from core.training.arch.anima import AnimaArchHandler
    with patch("core.model_loader.ModelLoader.detect_model_type", return_value="anima"):
        request = TrainingRunCreateRequest(training_method="full_finetune", base_model_path="anima.safetensors",
            total_steps=1, dit_rewiring_mode="configure", dit_partition_training_enabled=True,
            dit_partition_global_adapter_enabled=True, dit_partition_global_rank=64,
            dit_partition_global_tokens=16, dit_gradient_checkpointing_blocks=24,
            dit_partition_gradient_checkpointing_blocks=24, train_unet=True)
    section = _build_train_section(request.model_dump(), total_steps=1, epochs=None,
        train_unet=True, train_text_encoder=False, arch="anima")
    for name in ("dit_partition_training_enabled", "dit_partition_global_adapter_enabled",
                 "dit_partition_global_rank", "dit_partition_global_tokens",
                 "dit_gradient_checkpointing_blocks", "dit_partition_gradient_checkpointing_blocks"):
        assert section[name] == getattr(request, name)
    assert AnimaArchHandler().supports_dit_partition_training
    assert AnimaArchHandler().dit_partition_adapter() is not None
    bad = request.model_dump()
    bad["crop_decode_loss_enable"] = True
    with pytest.raises(ValueError, match="crop_decode"):
        _build_train_section(bad, total_steps=1, epochs=None,
            train_unet=True, train_text_encoder=False, arch="anima")
    bad = request.model_dump()
    bad["training_method"] = "lora"
    with pytest.raises(ValueError, match="full_finetune"):
        _build_train_section(bad, total_steps=1, epochs=None,
            train_unet=True, train_text_encoder=False, arch="anima")


class PointwiseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.3))
        self.blocks = []
        self.calls = []

    def forward(self, x, context, **kwargs):
        self.calls.append(kwargs["partition_box"])
        return self.scale * x + context[:, :1, :1].reshape(-1, 1, 1, 1, 1)


@pytest.mark.parametrize("weight", [0., .3])
@pytest.mark.parametrize("count", [2, 4])
def test_core_backward_matches_full_canvas_and_shared_conditioning_graph(weight, count):
    torch.manual_seed(19)
    clean, noise = torch.randn(2, 3, 24, 40), torch.randn(2, 3, 24, 40)
    sigma = torch.tensor([.2, .7])
    conditioning_parameter = torch.nn.Parameter(torch.tensor([.2, -.1]))
    prompt = (conditioning_parameter * 2).reshape(2, 1, 1)
    owner = trainer(dit_partition_fixed_count=count, dit_partition_halo_tokens=2)
    owner.transformer = PointwiseModel()
    owner.device = torch.device("cpu")
    owner.training_dtype = torch.float32
    owner.mixed_precision = False
    owner.reconstruction_loss_weight = weight
    owner._collect_concept_sample_loss = True
    plan = training_plan(owner, clean)
    aux = {name: torch.ones(2, 1, dtype=torch.long)
           for name in ("source_mask", "t5_input_ids", "t5_attn_mask")}
    actual = backward_regions(owner, clean=clean, prompt_embeds=prompt, anima_aux=aux,
        sigma=sigma, noise=noise, plan=plan, backward_scale=.25)
    actual_grads = owner.transformer.scale.grad.clone(), conditioning_parameter.grad.clone()
    owner.transformer.scale.grad = None
    conditioning_parameter.grad = None
    sigma_view = sigma[:, None, None, None]
    noisy = (1 - sigma_view) * clean + sigma_view * noise
    prediction = owner.transformer.scale * noisy + (conditioning_parameter * 2)[:, None, None, None]
    pred_items = (prediction - (noise - clean)).square().mean((1, 2, 3))
    recon_items = ((noisy - sigma_view * prediction) - clean).square().mean((1, 2, 3))
    combined = (1 - weight) * pred_items + weight * recon_items
    (combined.mean() * .25).backward()
    torch.testing.assert_close(torch.tensor(actual), torch.stack((combined.mean(), pred_items.mean(),
        recon_items.mean() if weight else torch.tensor(0.))).detach())
    torch.testing.assert_close(actual_grads[0], owner.transformer.scale.grad)
    torch.testing.assert_close(actual_grads[1], conditioning_parameter.grad)
    torch.testing.assert_close(owner._pending_concept_sample_loss, combined.detach())
    assert len(owner.transformer.calls) == count
