"""Native Ming hybrid optimization, independent layers and strict package restoration."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from ming_image_01_rewiring_test import config, spec_for
from ming_image_01_conditioning_test import write_tiny_package
from core.models.ming_image_01.model_definition import create_transformer
from core.models.ming_image_01.residual_unet import restore_topology
from core.rewiring.hybrid_spec import build_hybrid_spec


def hybrid_for(model):
    return build_hybrid_spec(spec_for(model), stream_widths={"main": model.dim}, boundaries=[
        dict(id="outer", region="r0", rank=6, kernel=3), dict(id="inner", region="r1", rank=6, kernel=3)],
        refiners=[dict(id="entry", stream="main", after_block=0, rank=6, kernel=3),
                  dict(id="exit", stream="main", after_block=5, rank=6, kernel=3)],
        skips=[dict(id="outer_skip", stream="main", source_after=0, destination_before=5, rank=6),
               dict(id="middle_skip", stream="main", source_after=1, destination_before=4, rank=6)])


@pytest.mark.parametrize("padding", ["zero_masked", "learned"])
def test_native_hybrid_matches_baseline_then_trains_every_branch(padding):
    torch.manual_seed(112)
    model = create_transformer(config(padding), device="cpu", dtype=torch.float32)
    restore_topology(model, spec_for(model))
    baseline = deepcopy(model)
    restore_topology(model, hybrid_for(model))
    frames = 3 if padding == "learned" else 1
    images = [torch.randn(16, frames, 16, 24) for _ in range(2)]
    queries = [torch.randn(5, 8) for _ in range(2)]
    captions = [torch.randn(n, 24) for n in (3, 35)]
    references = [torch.randn(16, 1, 16, 24), torch.randn(16, 2, 16, 24)]
    sigma = torch.tensor([.2, .6])
    def forward(module):
        return module(images, sigma, queries, cap_feats_2=captions, ref_x=references).sample
    for actual, expected in zip(forward(model), forward(baseline)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    parameters = list(model.rewiring_modules.parameters())
    identity = [id(p) for p in parameters]
    restore_topology(model, model._residual_unet_spec)
    assert identity == [id(p) for p in model.rewiring_modules.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        sum(output[:, :frames].square().mean() for output in forward(model)).backward()
        optimizer.step()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in parameters)
    from core.models.ming_image_01.vae_alignment import resize_backbone
    module_identity = model.rewiring_modules
    resize_backbone(model, 64)
    assert model.rewiring_modules is module_identity
    assert model._residual_unet_spec["method"] == "hybrid_unet_ssdit"
    from core.models.ming_image_01.partition_training import training_plan
    trainer = SimpleNamespace(transformer=model, config=dict(dit_partition_training_enabled=True,
        dit_partition_mode="fixed", dit_partition_fixed_count=2, dit_partition_halo_tokens=0))
    plan = training_plan(trainer, torch.zeros(1, 64, frames, 64, 96))
    assert all(region.input.height % 4 == region.input.width % 4 == 0 for region in plan.regions)
    assert not torch.cuda.is_initialized()


def test_live_convolutions_do_not_mix_adjacent_layer_grids():
    from hybrid_rewiring_test import execute
    torch.manual_seed(22)
    model = create_transformer(config("learned"), device="cpu", dtype=torch.float32)
    restore_topology(model, hybrid_for(model))
    modules = model.rewiring_modules
    with torch.no_grad():
        for parameter in modules.parameters():
            parameter.normal_(0, .1)
    layers = torch.randn(2, 3, 8 * 12, 24)
    together = modules.forward_stream(layers.flatten(1, 2), "main", 24, 12, 0, None,
        execute, lambda *args: None, grid_groups=3).reshape_as(layers)
    independent = modules.forward_stream(layers.flatten(0, 1), "main", 8, 12, 0, None,
        execute, lambda *args: None).reshape_as(layers)
    torch.testing.assert_close(together, independent, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("profile", ["generation_edit", "layer_decompose"])
@pytest.mark.parametrize("quantized", [False, True])
def test_real_bundle_loader_restores_hybrid_weights_and_optimizer_ownership(tmp_path, profile, quantized):
    from core.models.ming_image_01.loader import load_components
    from core.models.ming_image_01.checkpoint import save_bundle
    from core.training.adapters.ming_image_01_adapter import MingImage01FullParameterAdapter
    source = tmp_path / "source"
    source.mkdir()
    write_tiny_package(source, profile=profile)
    components = load_components(source, dtype=torch.float32, conditioner_attention="eager")
    restore_topology(components.transformer, hybrid_for(components.transformer))
    components.transformer.requires_grad_(True)
    trainer = SimpleNamespace(ming_image_01_components=components, config={}, unet_lr=.001, learning_rate=.001)
    adapter = MingImage01FullParameterAdapter(trainer, conditioner_groups=set(), conditioner_lrs={})
    groups = adapter.arch_param_groups()
    owned = [id(p) for group in groups for p in group["params"]]
    assert len(owned) == len(set(owned))
    assert all(id(p) in owned for p in components.transformer.rewiring_modules.parameters())
    with torch.no_grad():
        for p in components.transformer.rewiring_modules.parameters():
            p.normal_(0, .1)
    if quantized:
        from core.models.common.int8_runtime_quantize import quantize_linears_in_place
        quantize_linears_in_place(components.transformer, arch="ming_image_01", compute_dtype=torch.float32)
        with pytest.raises(NotImplementedError, match="quantized"):
            adapter.arch_param_groups()
    exported = save_bundle(components, tmp_path / "export", max_shard_bytes=4096)
    restored = load_components(exported, dtype=torch.float32, conditioner_attention="eager")
    expected, actual = components.transformer.state_dict(), restored.transformer.state_dict()
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    assert restored.transformer._residual_unet_spec == components.transformer._residual_unet_spec
    assert not any(p.device.type == "meta" for p in restored.transformer.parameters())
    frames = 2 if profile == "layer_decompose" else 1
    images, queries = [torch.randn(16, frames, 16, 24)], [torch.randn(5, 8)]
    captions, references, sigma = [torch.randn(3, 24)], [torch.randn(16, 1, 16, 24)], torch.tensor([.4])
    with torch.no_grad():
        before = components.transformer(images, sigma, queries, cap_feats_2=captions, ref_x=references).sample
        after = restored.transformer(images, sigma, queries, cap_feats_2=captions, ref_x=references).sample
    torch.testing.assert_close(before[0], after[0], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("damage", ["missing", "shape", "nan"])
def test_bad_hybrid_weights_are_rejected(tmp_path, damage):
    from core.models.ming_image_01.artifact import inspect_artifact
    from core.models.ming_image_01.loader import load_components
    from core.models.ming_image_01.checkpoint import save_bundle
    from core.models.ming_image_01.quantization import prepare_quantized_transformer, load_quantized_transformer
    from core.rewiring.artifacts import read_directory_topology
    source = tmp_path / "source"
    source.mkdir()
    write_tiny_package(source)
    components = load_components(source, dtype=torch.float32, conditioner_attention="eager")
    restore_topology(components.transformer, hybrid_for(components.transformer))
    exported = save_bundle(components, tmp_path / "export")
    path = exported.parent / "transformer/diffusion_pytorch_model.safetensors"
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata()
    state = load_file(str(path))
    key = next(key for key in state if key.startswith("rewiring_modules."))
    if damage == "missing":
        del state[key]
    elif damage == "shape":
        state[key] = state[key].flatten()[:1]
    else:
        state[key].fill_(float("nan"))
    save_file(state, str(path), metadata=metadata)
    artifact = inspect_artifact(exported)
    topology = read_directory_topology(artifact.components["transformer"])
    with pytest.raises(ValueError, match="Ming"):
        model = prepare_quantized_transformer(artifact, torch.float32, topology=topology)
        load_quantized_transformer(model, artifact, torch.float32)
