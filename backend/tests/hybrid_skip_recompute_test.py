"""Encoder replay preserves stochastic forward semantics and every gradient."""
from copy import deepcopy

import pytest
import torch

from hybrid_rewiring_test import declaration
from core.rewiring.hybrid_modules import HybridModules, HybridExecution
from core.rewiring.hybrid_spec import build_hybrid_spec, legacy_topology
from core.rewiring.runtime import run_rewired_stream
from hybrid_anima_test import native, forward as anima_forward


@pytest.fixture(autouse=True)
def full_precision_convolution_oracle():
    old_conv, old_matmul = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old_conv, old_matmul


def replay_spec(spec):
    return build_hybrid_spec(legacy_topology(spec), stream_widths=spec["stream_widths"],
        boundaries=spec["boundaries"], refiners=spec["refiners"],
        skips=[dict(item, storage="recompute") for item in spec["skips"]])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("checkpoint_mode", ["none", "normal", "cpu"])
@pytest.mark.parametrize("groups", [1, 2])
def test_stochastic_prefix_replay_preserves_output_rng_coordinates_and_gradients(device, checkpoint_mode, groups):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA transfer verification")
    torch.manual_seed(409)
    spec = declaration()
    resident = HybridModules(spec).to(device)
    for module in resident.modules():
        if hasattr(module, "expand"):
            torch.nn.init.normal_(module.expand.weight, std=.02)
        if hasattr(module, "restore"):
            torch.nn.init.normal_(module.restore.weight, std=.02)
    replayed = HybridModules(replay_spec(spec)).to(device)
    replayed.load_state_dict(deepcopy(resident.state_dict()))
    block = torch.nn.Linear(8, 8).to(device)
    replay_block = deepcopy(block)
    value = torch.randn(2, 3+8*groups*12, 8, device=device, requires_grad=True)
    other = value.detach().clone().requires_grad_()
    condition = torch.randn(2, 1, 8, device=device, requires_grad=True)
    other_condition = condition.detach().clone().requires_grad_()
    options = dict(checkpoint_modules=checkpoint_mode != "none", checkpoint_offload=checkpoint_mode == "cpu")
    calls = []
    def coarse(context, value, height, width, prefix):
        return dict(context, stride=12//width)
    def callback(module, conditioning, log):
        def execute(index, hidden, context):
            log.append(index)
            scale = (context["origin"] + context["stride"])*.001
            return hidden + module(hidden).tanh()*.02 + conditioning*.01 + torch.rand_like(hidden)*scale
        return execute
    def seed():
        torch.manual_seed(714)
    seed()
    reference = resident.forward_stream(value, "main", 8*groups, 12, 3,
        dict(origin=5, stride=1), callback(block, condition, []), coarse, grid_groups=groups, **options)
    expected_cpu_rng = torch.get_rng_state()
    expected_cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None
    seed()
    execution = HybridExecution(replayed, "main", groups, **options)
    execution.configure_replay(other, 8*groups, 12, 3, dict(origin=5, stride=1),
        callback(replay_block, other_condition, calls), coarse)
    actual = run_rewired_stream(other, legacy_topology(spec), "main", 8*groups, 12, 3,
        dict(origin=5, stride=1), callback(replay_block, other_condition, calls), coarse,
        grid_groups=groups, extensions=execution)
    assert len(calls) == 7+1+2
    assert not execution.saved
    assert execution.replay_input.device.type == "cpu" and execution.replay_input.requires_grad
    assert torch.equal(torch.get_rng_state(), expected_cpu_rng)
    if expected_cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(), expected_cuda_rng)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    reference.square().mean().backward()
    actual.square().mean().backward()
    assert len(calls) > 10, "Backward must recompute actual encoder blocks"
    tolerance = dict(rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(other.grad, value.grad, **tolerance)
    torch.testing.assert_close(other_condition.grad, condition.grad, **tolerance)
    for left, right in zip(block.parameters(), replay_block.parameters()):
        torch.testing.assert_close(left.grad, right.grad, **tolerance)
    for (name, left), (_, right) in zip(resident.named_parameters(), replayed.named_parameters()):
        assert left.grad is not None and right.grad is not None, name
        torch.testing.assert_close(left.grad, right.grad, **tolerance)
        assert torch.isfinite(right.grad).all() and right.grad.abs().sum()>0, name


def test_replay_source_after_a_destination_is_rejected():
    spec = declaration()
    spec["skips"].extend([
        dict(id="earlier", stream="main", source_after=2, destination_before=3, rank=3),
        dict(id="late", stream="main", source_after=3, destination_before=4, rank=3)])
    with pytest.raises(ValueError, match="precedes|precede"):
        replay_spec(spec)


def test_block_swap_gate_applies_at_configuration_and_execution():
    from core.rewiring.training import validate_training_features
    spec = replay_spec(declaration())
    config = dict(training_method="full_finetune", dit_rewiring_spec=spec, blocks_to_swap=2)
    with pytest.raises(ValueError, match="encoder replay"):
        validate_training_features(config, "anima")
    with pytest.raises(ValueError, match="encoder replay"):
        HybridModules(spec).forward_stream(torch.randn(1, 96, 8), "main", 8, 12, 0,
            None, lambda i, h, c: h, lambda *a: None, allow_encoder_replay=False)


def test_replay_preview_counts_shared_host_input_and_no_live_skip_payload():
    from core.rewiring.spec import preview_plan, checkpoint_metadata, read_checkpoint_metadata
    from core.rewiring.hybrid_spec import hybrid_preview
    spec = replay_spec(declaration())
    preview = preview_plan(spec, 8, 12)
    assert read_checkpoint_metadata(checkpoint_metadata(spec)) == spec
    entries = preview["hybrid_activation_layout"]["entries"]
    root = [entry for entry in entries if entry["kind"] == "encoder_replay_input"]
    assert len(root) == 1 and root[0]["elements"] == 8*12*8
    skips = [entry for entry in entries if entry["kind"] == "compressed_skip"]
    assert all(entry["elements"] == 0 and entry["recomputed_elements"] > 0 for entry in skips)
    detailed = hybrid_preview(spec, 8, 12, batch_size=1, element_bytes=4)
    assert all(entry["stored_bytes"] == 0 for entry in detailed["learned_skip_edges"])


def compare_native(model, spec, forward, restore):
    for module in model.rewiring_modules.modules():
        with torch.no_grad():
            if hasattr(module, "expand"):
                module.expand.weight.normal_(0, .02)
            if hasattr(module, "restore"):
                module.restore.weight.normal_(0, .02)
    replayed = deepcopy(model)
    restore(replayed, replay_spec(spec))
    reference, actual = forward(model), forward(replayed)
    if isinstance(reference, (list, tuple)):
        reference, actual = torch.stack(reference), torch.stack(actual)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    reference.square().mean().backward()
    actual.square().mean().backward()
    for (name, left), (_, right) in zip(model.named_parameters(), replayed.named_parameters()):
        assert (left.grad is None) == (right.grad is None), name
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-7, msg=name)
    for name, parameter in replayed.rewiring_modules.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


@pytest.mark.parametrize("mode", ["none", "standard", "cpu_offload"])
def test_native_anima_replay_and_internal_checkpointing(native, mode):
    from core.models.anima.residual_unet import restore_topology
    model, spec = native
    model.train().requires_grad_(True)
    for block in model.blocks:
        block.gradient_checkpoint_mode = mode
    compare_native(model, spec, anima_forward, restore_topology)


def test_native_anima_partition_and_global_replay(native):
    from hybrid_anima_test import test_native_partition_and_global_adapter_backward
    from core.models.anima.residual_unet import restore_topology
    model, spec = native
    spec = replay_spec(spec)
    restore_topology(model, spec)
    with torch.no_grad():
        for module in model.rewiring_modules.modules():
            if hasattr(module, "expand"):
                module.expand.weight.normal_(0, .02)
            if hasattr(module, "restore"):
                module.restore.weight.normal_(0, .02)
    test_native_partition_and_global_adapter_backward((model, spec))


@pytest.mark.parametrize("checkpointed", [False, True])
def test_native_ming_replay_and_caption_prefix(checkpointed):
    from hybrid_ming_test import hybrid_for
    from ming_image_01_rewiring_test import config
    from core.models.ming_image_01.model_definition import create_transformer
    from core.models.ming_image_01.residual_unet import restore_topology
    model = create_transformer(config("zero_masked"), device="cpu", dtype=torch.float32)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim >= 2:
                parameter.normal_(0, .02)
    spec = hybrid_for(model)
    restore_topology(model, spec)
    model.train().requires_grad_(True)
    if checkpointed:
        model.enable_gradient_checkpointing()
    images = [torch.randn(16, 1, 16, 24)]
    queries = [torch.randn(3, 8)]
    compare_native(model, spec, lambda module: module(images, torch.tensor([.4]), queries).sample, restore_topology)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_native_qwen_replay_and_causal_prefix(checkpointed):
    from hybrid_qwen_test import model_and_spec
    from qwen_partition_full_checkpoint_test import _forward
    from core.models.qwen_image_21.residual_unet import restore_topology
    model, spec, _ = model_and_spec()
    restore_topology(model, spec)
    model.train().requires_grad_(True)
    if checkpointed:
        model.enable_gradient_checkpointing()
    grid = torch.randn(1, 8, 12, 8)
    compare_native(model, spec, lambda module: _forward(module, grid), restore_topology)
