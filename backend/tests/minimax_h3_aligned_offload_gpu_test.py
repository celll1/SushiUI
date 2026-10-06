"""Opt-in real CUDA Full updates with target-native H3 alignment and block swap."""

from copy import deepcopy
import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("SUSHI_VAE_SWAP_GPU_TESTS") != "1",
                              reason="Coordinate GPU use and set SUSHI_VAE_SWAP_GPU_TESTS=1")


def device_tree(value):
    if torch.is_tensor(value):
        return value.detach().clone().cuda().requires_grad_(value.requires_grad)
    if isinstance(value, dict):
        return {key: device_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [device_tree(item) for item in value]
    return value


@pytest.mark.parametrize("rewired", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_actual_h3_full_updates_match_resident_fused_optimizer(rewired, dtype):
    from minimax_h3_video_alignment_test import setup
    from core.training.ops.minimax_h3_ops import setup_block_swap
    from core.training.optimizers.adafactor_fused import patch_adafactor_fused
    from transformers import Adafactor
    model, _, inputs, _ = setup(4, rewired, dtype)
    model.requires_grad_(True).train().enable_gradient_checkpointing()
    from core.training.model_staging import stage_dense_block_model
    swapped = deepcopy(model)
    stage_dense_block_model(swapped, swapped.transformer_blocks, 5, "cuda")
    assert all(p.device.type == "cpu" for layer in swapped.transformer_blocks[2:] for p in layer.parameters())
    resident = model.cuda()
    trainer = SimpleNamespace(transformer=swapped, is_minimax_h3=True, blocks_to_swap=5,
        block_swap_ring_size=2, gradient_checkpointing=True, use_pinned_memory=True,
        layer_offload_conductor=None, device=torch.device("cuda"), log_prefix="[H3 GPU verification]")
    setup_block_swap(trainer)
    conductor = trainer.layer_offload_conductor
    optimizers, handles = [], []
    updated = {}
    # Match the autograd view nodes introduced by the conductor's module hooks.
    for index in conductor.swappable:
        handles.append(resident.transformer_blocks[index].register_full_backward_hook(lambda *args: None))
    for current in (resident, swapped):
        optimizer = Adafactor(current.parameters(), lr=1e-3, relative_step=False,
                              scale_parameter=False, warmup_init=False)
        patch_adafactor_fused(optimizer)
        group = optimizer.param_groups[0]
        for parameter in current.parameters():
            def update(tensor, opt=optimizer, param_group=group, capture=current is swapped):
                assert tensor.device.type == "cuda"
                opt.step_param(tensor, param_group)
                if capture:
                    updated[id(tensor)] = tensor.detach().clone()
                tensor.grad = None
            handles.append(parameter.register_post_accumulate_grad_hook(update))
        optimizers.append(optimizer)
    conductor.register_optimizer_hooks()
    max_relative = 0.
    try:
        for _ in range(3):
            losses = []
            for current in (resident, swapped):
                output = current(**device_tree(inputs))
                loss = sum(value.float().square().mean() for value in output)
                assert torch.isfinite(loss)
                losses.append(loss.detach())
                loss.backward()
                if current is swapped:
                    conductor.finish_backward()
                    conductor.flush()
                    assert not conductor.engine.active
                    assert all(p.device.type == "cpu" for i in conductor.swappable
                               for p in swapped.transformer_blocks[i].parameters())
                    for i in conductor.swappable:
                        for parameter in swapped.transformer_blocks[i].parameters():
                            torch.testing.assert_close(parameter, updated[id(parameter)].cpu(), rtol=0, atol=0)
            tolerance = (1e-5, 1e-7) if dtype == torch.float32 else (8e-3, 5e-5)
            torch.testing.assert_close(*losses, rtol=tolerance[0], atol=tolerance[1])
            for (name, expected), (actual_name, actual) in zip(resident.named_parameters(), swapped.named_parameters()):
                assert name == actual_name
                if dtype == torch.float32:
                    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=tolerance[0], atol=tolerance[1])
                delta = actual.detach().cpu().float()-expected.detach().cpu().float()
                relative = delta.norm()/expected.detach().cpu().float().norm().clamp_min(1e-12)
                max_relative = max(max_relative, relative.item())
                # Near-zero elements amplify BF16 reduction-order differences.
                assert relative < (1e-6 if dtype == torch.float32 else torch.finfo(dtype).eps), (name, relative.item())
                assert actual.grad is None and expected.grad is None
            pair = swapped.transformer._video_latent_alignment
            assert all(optimizers[1].state[p]["step"] >= 1 for p in pair.parameters())
        assert all(optimizers[1].state[p]["step"] == 3 for p in swapped.parameters())
        assert conductor.engine.stats().d2h_bytes > 0
        print(dict(dtype=str(dtype), unet=rewired, max_parameter_relative_l2=max_relative,
                   transfer_stats=conductor.engine.stats()))
        torch.cuda.synchronize()
    finally:
        conductor.abort_step()
        conductor.cleanup()
        for handle in handles:
            handle.remove()
