"""Native Ming U-Net target packing, gradients and metadata restoration."""

import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def forbidden_cuda(*args, **kwargs):
    raise AssertionError("Ming rewiring test attempted CUDA initialization")


torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda._lazy_init = forbidden_cuda
torch._C._cuda_init = forbidden_cuda
torch.set_num_threads(1)

from core.models.ming_image_01.model_definition import create_transformer
from core.models.ming_image_01.residual_unet import (
    configure_training_topology, default_topology, model_topology_metadata,
    native_config, restore_topology, run_rewired_ming, validate_topology_identity,
)
from core.rewiring.spec import OPERATORS, build_spec
from core.rewiring.artifacts import read_directory_topology


def test_training_pixel_alignment_tracks_nested_topology():
    from types import SimpleNamespace
    from core.training.base_trainer import BaseTrainer
    model = create_transformer(config("zero_masked"), device="cpu", dtype=torch.float32)
    owner = SimpleNamespace(arch=SimpleNamespace(pixel_align=16), is_ming_image_01=True, transformer=model)
    assert BaseTrainer._arch_pixel_align(owner) == 16
    restore_topology(model, spec_for(model))
    assert BaseTrainer._arch_pixel_align(owner) == 64
    owner.is_ming_image_01 = False
    assert BaseTrainer._arch_pixel_align(owner) == 16


@pytest.mark.parametrize("count,halo", [(2, 0), (4, 0), (4, 2)])
def test_random_partition_inputs_align_to_nested_unet_without_changing_cores(count, halo):
    from types import SimpleNamespace
    from core.models.ming_image_01.partition_training import training_plan
    from core.models.ming_image_01.partition import MingPartitionAdapter
    from core.rewiring.spec import preview_plan
    model = create_transformer(config("zero_masked"), device="cpu", dtype=torch.float32)
    spec = spec_for(model)
    restore_topology(model, spec)
    settings = dict(dit_partition_training_enabled=True, dit_partition_mode="fixed",
                    dit_partition_fixed_count=count, dit_partition_halo_tokens=halo,
                    dit_partition_seed=42, dit_partition_split_ratio_min=.35, dit_partition_split_ratio_max=.65)
    owner = SimpleNamespace(config=settings, transformer=model)
    clean = torch.zeros(1, 16, 1, 32, 48)
    for occurrence in range(8):
        owner._current_batch_position = occurrence
        plan = training_plan(owner, clean)
        original = MingPartitionAdapter().build_fixed_plan(16, 24, count=count, halo=halo,
            seed=42, epoch=0, occurrence=occurrence, split_ratio_min=.35, split_ratio_max=.65)
        assert [region.core for region in plan.regions] == [region.core for region in original.regions]
        for region in plan.regions:
            assert region.input.top % 4 == region.input.left % 4 == 0
            preview_plan(spec, region.input.height, region.input.width)


def config(padding):
    return {"dim": 24, "n_layers": 6, "n_refiner_layers": 1, "n_heads": 3,
            "n_kv_heads": 3, "cap_feat_dim": 8, "in_channels": 16,
            "axes_dims": [2, 2, 4], "axes_lens": [128, 32, 32],
            "alignment_padding_mode": padding, "multi_frame_output": padding == "learned"}


def spec_for(model):
    return build_spec("ming_image_01", [{"name": "main", "block_count": len(model.layers)}], [
        dict(id="r0", stream="main", parent=None, start_block=1, end_block=5, down_factor=[2, 2], **OPERATORS),
        dict(id="r1", stream="main", parent="r0", start_block=2, end_block=4, down_factor=[2, 2], **OPERATORS),
    ], native_config(model))


@pytest.mark.parametrize("padding", ["zero_masked", "learned"])
def test_native_nested_rewiring_gradients_reference_caption_and_batch_padding(padding):
    torch.manual_seed(23)
    model = create_transformer(config(padding), device="cpu", dtype=torch.float32)
    if padding == "learned":
        torch.nn.init.constant_(model.x_pad_token, 0.02)
        torch.nn.init.constant_(model.cap_pad_token, 0.03)
    restore_topology(model, spec_for(model))
    frames = 3 if padding == "learned" else 1
    images = [torch.randn(16, frames, 16, 16, requires_grad=True) for _ in range(2)]
    references = [torch.randn(16, 1, 16, 16), torch.randn(16, 2, 16, 16)]
    queries = [torch.randn(5, 8, requires_grad=True), torch.randn(5, 8, requires_grad=True)]
    captions = [torch.randn(3, 24, requires_grad=True), torch.randn(35, 24, requires_grad=True)]
    lengths = []
    handles = [layer.register_forward_pre_hook(lambda _m, args: lengths.append(args[0].shape[1])) for layer in model.layers]
    output = model(images, torch.tensor([0.2, 0.6]), queries, cap_feats_2=captions, ref_x=references).sample
    for handle in handles:
        handle.remove()
    prefix = lengths[0] - frames * 64
    assert lengths == [prefix + frames * count for count in (64, 16, 4, 4, 16, 64)]
    assert output[0].shape[1] == (frames + 1 if padding == "learned" else 1)
    assert output[1].shape[1] == (frames + 2 if padding == "learned" else 1)
    sum(value[:, :frames].square().mean() for value in output).backward()
    assert all(image.grad is not None and torch.isfinite(image.grad).all() for image in images)
    assert all(value.grad is not None and value.grad.abs().sum() > 0 for value in queries + captions)
    assert all(layer.attention.to_q.weight.grad.abs().sum() > 0 for layer in model.layers)
    assert not torch.cuda.is_initialized()


def test_transport_identity_preserves_each_frame_and_auxiliary_update():
    model = create_transformer(config("learned"), device="cpu", dtype=torch.float32)
    class Add(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = torch.nn.Identity()

        def forward(self, value, mask, positions, modulation):
            assert mask.shape == value.shape[:2]
            assert positions.shape[:2] == value.shape[:2]
            return value + 1
    model.layers = torch.nn.ModuleList([Add() for _ in range(6)])
    frames, height, width, extras = 3, 8, 8, 7
    count = frames * height * width
    hidden = torch.randn(2, count + extras, 24, requires_grad=True)
    positions = []
    for start in (2, 5):
        positions.append(model.create_coordinate_grid((frames, height, width), (start, 0, 0), device="cpu").flatten(0, 2))
    rotary = torch.cat([model.rope_embedder(torch.stack(positions).flatten(0, 1)).reshape(2, count, -1),
                        torch.ones(2, extras, 4, dtype=torch.complex64)], dim=1)
    mask = torch.ones(2, count + extras, dtype=torch.bool)
    mask[0, -2:] = False
    output = run_rewired_ming(model, hidden, torch.zeros(2, 24), rotary, mask, positions,
                             (16, frames, 16, 16), 2, spec_for(model))
    torch.testing.assert_close(output, hidden + 6, rtol=1e-6, atol=1e-6)
    output.sum().backward()
    torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))


def test_metadata_profile_identity_and_training_resume(tmp_path):
    model = create_transformer(config("zero_masked"), device="cpu", dtype=torch.float32)
    spec = spec_for(model)
    restore_topology(model, spec)
    model.save_config(tmp_path)
    save_file(model.state_dict(), tmp_path / "diffusion_pytorch_model.safetensors", metadata=model_topology_metadata(model))
    restored = read_directory_topology(tmp_path)
    assert restored == spec
    clone = create_transformer(config("zero_masked"), device="cpu", dtype=torch.float32)
    resolved = configure_training_topology(clone, {"dit_rewiring_mode": "inherit"}, resuming=True, resume_spec=restored)
    assert resolved == clone._residual_unet_spec == spec
    with pytest.raises(ValueError, match="identity|config"):
        validate_topology_identity(spec, config("learned"))
    with pytest.raises(ValueError, match="Resume"):
        configure_training_topology(clone, {"dit_rewiring_mode": "configure", "dit_rewiring_spec": spec},
                                    resuming=True, resume_spec=None)


def test_default_main_depth_and_parameters_unchanged():
    settings = {**config("zero_masked"), "n_layers": 30}
    spec = default_topology(settings)
    assert [(r["start_block"], r["end_block"]) for r in spec["regions"]] == [(4, 26), (13, 17)]
    model = create_transformer(config("learned"), device="cpu", dtype=torch.float32)
    before = set(model.state_dict())
    count = sum(parameter.numel() for parameter in model.parameters())
    restore_topology(model, spec_for(model))
    assert set(model.state_dict()) == before
    assert sum(parameter.numel() for parameter in model.parameters()) == count


def test_checkpointing_runs_native_rewired_backward():
    model = create_transformer(config("zero_masked"), device="cpu", dtype=torch.float32)
    model.enable_gradient_checkpointing()
    restore_topology(model, spec_for(model))
    image = torch.randn(16, 1, 16, 16, requires_grad=True)
    model([image], torch.tensor([0.2]), [torch.randn(4, 8)], cap_feats_2=[torch.randn(3, 24)]).sample[0].square().mean().backward()
    assert image.grad is not None
    assert model.layers[2].attention.to_q.weight.grad.abs().sum() > 0



@pytest.mark.parametrize("padding", ["zero_masked", "learned"])
def test_native_parameter_census_matches_actual_parameters(padding):
    from core.rewiring.parameters import parameter_counts
    model = create_transformer(config(padding), device="cpu", dtype=torch.float32)
    census = parameter_counts("ming_image_01", native_config(model))
    assert census["total"] == sum(parameter.numel() for parameter in model.parameters())
    assert census["blocks"][0]["count"] == sum(parameter.numel() for parameter in model.layers[0].parameters())
    assert census["added"] == 0
