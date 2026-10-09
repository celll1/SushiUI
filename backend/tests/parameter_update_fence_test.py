"""A missing gradient is not proof that its leaf update has completed."""
import torch
from types import SimpleNamespace

from core.memory_management.parameter_update_fence import ParameterUpdateFence
from core.memory_management.layer_offload_conductor import LayerOffloadConductor
from core.memory_management.branched_layer_offload_conductor import BranchedLayerOffloadConductor


def test_fence_waits_for_late_leaves_and_ignores_unused_parameters():
    p, q, unused = (torch.nn.Parameter(torch.tensor(float(i))) for i in (1, 2, 3))
    fence = ParameterUpdateFence((p, q, unused))
    events = []
    for parameter in (p, q, unused):
        def update(tensor):
            tensor.data.add_(tensor.grad, alpha=-.1)
            tensor.grad = None
        parameter.register_post_accumulate_grad_hook(update)
        def observe(tensor):
            fence.observe_accumulation(tensor)
            events.append(fence.ready())
        parameter.register_post_accumulate_grad_hook(observe)
    for _ in range(2):
        events.clear()
        output = p.square() + q.square()
        def before_leaves(gradient):
            fence.backward_complete()
            assert p.grad is None and q.grad is None
            assert not fence.ready()
            return gradient
        output.register_hook(before_leaves)
        output.backward()
        assert events == [False, True]
        assert fence.expected == {id(p), id(q)}
        assert unused.item() == 3
        fence.reset()


def test_fence_holds_accumulated_gradients_until_group_update():
    p, q = (torch.nn.Parameter(torch.tensor(1.)) for _ in range(2))
    fence = ParameterUpdateFence((p, q))
    for parameter in (p, q):
        parameter.register_post_accumulate_grad_hook(fence.observe_accumulation)
    output = p + q
    output.register_hook(lambda gradient: (fence.backward_complete(), gradient)[1])
    output.backward()
    assert not fence.ready()
    p.grad = q.grad = None
    assert fence.ready()


def test_no_parameter_updates_when_backward_requests_only_input_gradient():
    parameter = torch.nn.Parameter(torch.tensor(2.))
    value = torch.tensor(3., requires_grad=True)
    fence = ParameterUpdateFence((parameter,))
    output = parameter * value
    output.register_hook(lambda gradient: (fence.backward_complete(), gradient)[1])
    output.backward(inputs=(value,))
    assert fence.expected == set() and fence.ready()
    assert parameter.grad is None and value.grad.item() == 2


def test_speculative_prefetch_does_not_compete_with_pending_updates():
    calls = []
    conductor = LayerOffloadConductor.__new__(LayerOffloadConductor)
    conductor.enable_prefetch, conductor.ring_size, conductor.swappable = True, 2, (0,1,2)
    conductor.engine = SimpleNamespace(active={1,2}, prefetch=calls.append)
    conductor._prefetch_neighbor(2, backward=True)
    assert calls == []
    conductor.engine.active.remove(1)
    conductor._prefetch_neighbor(2, backward=True)
    assert calls == [1]
    branched = BranchedLayerOffloadConductor.__new__(BranchedLayerOffloadConductor)
    branched.enable_prefetch, branched.ring_size = True, 2
    branched.swappable_layer_indices = (0,1,2)
    branched._masters = {('a',1): {}}
    branched.engine = SimpleNamespace(active={('a',1),('a',2)}, prefetch=calls.append)
    branched._prefetch_neighbor(('a',2), backward=True)
    assert calls == [1]
