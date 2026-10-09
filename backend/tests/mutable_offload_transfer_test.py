import copy
import io
from types import SimpleNamespace

import pytest
import torch

from core.memory_management.layer_offload_conductor import LayerOffloadConductor
from core.memory_management.branched_layer_offload_conductor import BranchedLayerOffloadConductor
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


def test_mutable_ring_pause_preserves_updated_masters_and_reloads_fresh_slots():
    masters={0:{torch.float32:torch.tensor([1.,2.])},1:{torch.float32:torch.tensor([3.,4.])}}
    pointed={}
    engine=MutableLruTransferEngine(keys=[0,1],masters=masters,ring_size=1,device=torch.device("cpu"),
        point_bundle=lambda key,bundle:pointed.update({key:bundle[torch.float32]}))
    engine.acquire(0)
    with pytest.raises(RuntimeError,match="active layers"):
        engine.park_slots()
    pointed[0].add_(5)
    engine.release(0,dirty=True)
    engine.prefetch(1)
    engine.park_slots()
    assert engine.slots_parked and not engine.slots and not engine.key_slot
    assert pointed[0].data_ptr()==masters[0][torch.float32].data_ptr()
    assert torch.equal(masters[0][torch.float32],torch.tensor([6.,7.]))
    engine.acquire(0)
    assert not engine.slots_parked and len(engine.slots)==1
    assert torch.equal(pointed[0],masters[0][torch.float32])
    engine.release(0,dirty=False)
    engine.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("frozen_input", [False, True])
@pytest.mark.parametrize("internal_checkpoint", [False, True])
def test_checkpointed_mutable_conductor_matches_resident_update(frozen_input, internal_checkpoint):
    class InternalBlock(torch.nn.Sequential):
        def set_offload_recompute_callback(self, callback):
            self.callback = callback

        def body(self, value):
            callback = getattr(self, "callback", None)
            if callback is not None and torch._C._current_graph_task_id() >= 0:
                callback()
            return super().forward(value)

        def forward(self, value):
            return torch.utils.checkpoint.checkpoint(self.body, value, use_reentrant=False)

    torch.manual_seed(1234)
    block_type = InternalBlock if internal_checkpoint else torch.nn.Sequential
    resident = torch.nn.ModuleList([
        block_type(torch.nn.Linear(32, 64), torch.nn.SiLU(), torch.nn.Linear(64, 32))
        for _ in range(4)
    ]).cuda()
    if frozen_input:
        resident[0].requires_grad_(False)
    swapped = copy.deepcopy(resident).cpu()
    target = torch.randn(3, 32, device="cuda")
    sample = torch.randn(3, 32, device="cuda", requires_grad=not frozen_input)
    lr = 1e-3

    def install_updates(module):
        handles = []
        for parameter in module.parameters():
            if not parameter.requires_grad:
                continue
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
            value = layer(value) if internal_checkpoint else torch.utils.checkpoint.checkpoint(layer, value, use_reentrant=False)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("frozen_base", [False, True])
def test_branched_conductor_waits_for_active_branch_leaf_updates(frozen_base):
    class AdapterBranch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(16, 16).requires_grad_(False)
            self.adapter = torch.nn.Linear(16, 16, bias=False)

        def forward(self, value):
            return self.base(value) + self.adapter(value)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = (AdapterBranch() if frozen_base else
                      torch.nn.Sequential(torch.nn.Linear(16, 24), torch.nn.SiLU(), torch.nn.Linear(24, 16)))
            self.b = copy.deepcopy(self.a)

        def forward(self, value, *, branch):
            return getattr(self, branch)(value)

    torch.manual_seed(47)
    resident = torch.nn.ModuleList([Block() for _ in range(3)]).cuda()
    resident[0].requires_grad_(False)
    swapped = copy.deepcopy(resident).cpu()
    conductor = BranchedLayerOffloadConductor(root=swapped, layers=swapped,
        branches=("a", "b"), classify_module=lambda path, module: path.split(".")[0] if path else None,
        resolve_call_branches=lambda args, kwargs: (kwargs["branch"],),
        blocks_to_swap=2, device=torch.device("cuda"), ring_size=2, frozen_base_only=frozen_base)
    conductor.register_hooks()
    handles = []
    for layers in (resident, swapped):
        for parameter in layers.parameters():
            if parameter.requires_grad:
                def update(tensor):
                    tensor.grad = tensor.grad.contiguous()
                    tensor.data.add_(tensor.grad, alpha=-.001)
                    tensor.grad = None
                handles.append(parameter.register_post_accumulate_grad_hook(update))
    conductor.register_optimizer_hooks()
    try:
        for branch in ("a", "b"):
            sample = torch.randn(2, 16, device="cuda")
            target = torch.randn_like(sample)
            def loss(layers):
                value = sample
                for layer in layers:
                    value = torch.utils.checkpoint.checkpoint(layer, value, branch=branch, use_reentrant=False)
                return (value-target).square().mean()
            expected, actual = loss(resident), loss(swapped)
            expected.backward()
            actual.backward()
            conductor.finish_backward()
            conductor.flush()
            assert torch.equal(expected, actual)
            for left, right in zip(resident.parameters(), swapped.parameters()):
                assert torch.equal(left.detach().cpu(), right.detach().cpu())
    finally:
        conductor.abort_step()
        conductor.cleanup()
        for handle in handles:
            handle.remove()
