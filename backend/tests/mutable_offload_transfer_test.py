import copy
import io
from types import SimpleNamespace

import pytest
import torch

from core.memory_management.layer_offload_conductor import LayerOffloadConductor
from core.memory_management.offload_transfer_engine import MutableLruTransferEngine
from core.training.base_trainer import BaseTrainer


def test_trainer_abort_releases_every_mutable_conductor():
    calls = []
    conductors = [
        SimpleNamespace(abort_step=lambda key=key: calls.append(key))
        for key in ("conditional", "unconditional")
    ]
    trainer = SimpleNamespace(_layer_offload_conductors=lambda: conductors)

    BaseTrainer._abort_layer_offload_step(trainer)

    assert calls == ["conditional", "unconditional"]


def test_mutable_engine_writes_dirty_slot_back_before_reuse():
    masters = {
        0: {torch.float32: torch.tensor([1.0, 2.0])},
        1: {torch.float32: torch.tensor([3.0, 4.0])},
    }
    pointed = {}

    def point(key, bundle):
        pointed[key] = bundle[torch.float32]

    engine = MutableLruTransferEngine(
        keys=[0, 1], masters=masters, ring_size=1,
        device=torch.device("cpu"), point_bundle=point,
    )
    engine.acquire(0)
    pointed[0].add_(10)
    engine.release(0, dirty=True)
    engine.acquire(1)
    engine.release(1, dirty=False)

    assert torch.equal(masters[0][torch.float32], torch.tensor([11.0, 12.0]))
    assert engine.stats().d2h_bytes == 8
    assert engine.stats().d2h_submissions == 1


def test_mutable_engine_never_evicts_an_active_slot():
    masters = {
        0: {torch.float32: torch.ones(2)},
        1: {torch.float32: torch.ones(2)},
    }
    engine = MutableLruTransferEngine(
        keys=[0, 1], masters=masters, ring_size=1,
        device=torch.device("cpu"), point_bundle=lambda key, bundle: None,
    )
    engine.acquire(0)
    with pytest.raises(RuntimeError, match="every GPU slot is still active"):
        engine.prefetch(1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_checkpointed_mutable_conductor_matches_resident_update():
    torch.manual_seed(1234)
    resident = torch.nn.ModuleList([
        torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.SiLU(), torch.nn.Linear(64, 32))
        for _ in range(4)
    ]).cuda()
    swapped = copy.deepcopy(resident).cpu()
    target = torch.randn(3, 32, device="cuda")
    sample = torch.randn(3, 32, device="cuda", requires_grad=True)
    lr = 1e-3

    def install_updates(module):
        handles = []
        for parameter in module.parameters():
            def update(tensor):
                tensor.data.add_(tensor.grad, alpha=-lr)
                tensor.grad = None
            handles.append(parameter.register_post_accumulate_grad_hook(update))
        return handles

    resident_handles = install_updates(resident)
    conductor = LayerOffloadConductor(
        swapped, blocks_to_swap=3, device=torch.device("cuda"), ring_size=2,
    )
    conductor.register_hooks()
    swapped_handles = install_updates(swapped)
    conductor.register_optimizer_hooks()

    def run(layers, value):
        for layer in layers:
            value = torch.utils.checkpoint.checkpoint(layer, value, use_reentrant=False)
        return torch.nn.functional.mse_loss(value, target)

    resident_loss = run(resident, sample.clone())
    swapped_loss = run(swapped, sample.clone())
    resident_loss.backward()
    swapped_loss.backward()
    conductor.finish_backward()
    conductor.flush()

    assert torch.equal(resident_loss, swapped_loss)
    for left, right in zip(resident.parameters(), swapped.parameters()):
        assert torch.equal(left.detach().cpu(), right.detach().cpu())
    assert all(state == "cpu" for idx, state in conductor.layer_states.items() if idx >= 1)
    assert conductor.engine.stats().d2h_bytes > 0

    checkpoint = io.BytesIO()
    torch.save({key: value.detach().cpu() for key, value in swapped.state_dict().items()}, checkpoint)
    checkpoint.seek(0)
    resumed = copy.deepcopy(swapped).cpu()
    resumed.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    for left, right in zip(resident.parameters(), resumed.parameters()):
        assert torch.equal(left.detach().cpu(), right.detach().cpu())

    conductor.cleanup()
    for handle in (*resident_handles, *swapped_handles):
        handle.remove()
