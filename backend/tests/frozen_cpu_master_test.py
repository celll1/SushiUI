"""Frozen generation masters preserve CPU storage and repeated forward values."""

from copy import deepcopy
import os

import pytest
import torch

from core.memory_management.block_offloading import TransformerBlockOffloader
from core.memory_management.frozen_module_offload_conductor import (
    _FrozenBundleStore, _owned_tensors, FrozenModuleOffloadConductor,
)


def test_frozen_store_borrows_cpu_storage_and_preserves_shared_owner_guard():
    block = torch.nn.Linear(8, 8)
    tensors = list(_owned_tensors(block))
    pointers = [tensor.data_ptr() for _, _, tensor in tensors]
    store = _FrozenBundleStore(device=torch.device("cpu"), use_pinned_memory=False)
    store.add(0, tensors)
    assert [plane.data_ptr() for plane in store.masters[0].values()] == pointers
    store.point(0, store.masters[0])
    assert [tensor.data_ptr() for _, _, tensor in tensors] == pointers
    with pytest.raises(ValueError, match="shared"):
        store.add(1, tensors)


def test_block_h2d_borrows_cpu_weights_and_restores_views():
    blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(4)])
    reference = deepcopy(blocks)
    pointers = [block.weight.data_ptr() for block in blocks]
    conductor = TransformerBlockOffloader(blocks=blocks, blocks_to_swap=3,
        device=torch.device("cpu"), target_dtype=torch.float32, use_pinned_memory=False,
        supports_backward=False, h2d_only=True, ring_size=2)
    try:
        conductor.prepare_block_devices_before_forward()
        conductor._h2d_setup()
        assert [block.weight.data_ptr() for block in blocks] == pointers
        with torch.no_grad():
            for _ in range(3):
                actual = expected = torch.randn(2, 8)
                for index, (block, oracle) in enumerate(zip(blocks, reference)):
                    conductor.wait_for_block(index)
                    actual = block(actual)
                    conductor.submit_move_blocks_forward(index)
                    expected = oracle(expected)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        conductor.cleanup()
    assert [block.weight.data_ptr() for block in blocks] == pointers


@pytest.mark.skipif(os.environ.get("SUSHI_VAE_SWAP_GPU_TESTS") != "1", reason="GPU opt-in")
@pytest.mark.parametrize("driver", ["block", "hooks"])
@pytest.mark.parametrize("pinned", [False, True])
def test_frozen_generation_gpu_parity(driver, pinned):
    assert torch.cuda.is_available()
    torch.manual_seed(137)
    model = torch.nn.Sequential(*[torch.nn.Linear(16, 16) for _ in range(7)]).eval()
    reference = deepcopy(model).cuda()
    pointers = [module.weight.data_ptr() for module in model]
    if driver == "block":
        conductor = TransformerBlockOffloader(blocks=list(model), blocks_to_swap=5,
            device=torch.device("cuda"), target_dtype=torch.float32, use_pinned_memory=pinned,
            supports_backward=False, h2d_only=True, ring_size=2)
        conductor.prepare_block_devices_before_forward()
        conductor._h2d_setup()
    else:
        conductor = FrozenModuleOffloadConductor(root=model, modules=list(model), blocks_to_swap=5,
            device=torch.device("cuda"), use_pinned_memory=pinned, ring_size=2)
        conductor.register_hooks()
    try:
        with torch.no_grad():
            for _ in range(3):
                inputs = torch.randn(2, 16, device="cuda")
                if driver == "block":
                    actual = inputs
                    for index, module in enumerate(model):
                        conductor.wait_for_block(index)
                        actual = module(actual)
                        conductor.submit_move_blocks_forward(index)
                else:
                    actual = model(inputs)
                torch.testing.assert_close(actual, reference(inputs), rtol=0, atol=0)
    finally:
        conductor.cleanup()
    if not pinned:
        assert [model[index].weight.data_ptr() for index in range(2, 7)] == pointers[2:]
