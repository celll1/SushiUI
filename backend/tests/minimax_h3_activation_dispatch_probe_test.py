import sys
from pathlib import Path

import torch

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.training.probes.minimax_h3_activation_dispatch import (
    _compare_gradients,
    _optimizer_smoke,
)


def test_optimizer_smoke_restores_live_parameters_and_is_repeatable():
    layer = torch.nn.Linear(2, 1, bias=False)
    layers = {"test": layer}
    initial = layer.weight.detach().clone()
    gradients = {"test.weight": torch.tensor([[0.25, -0.5]])}

    first = _optimizer_smoke(layers, gradients)
    second = _optimizer_smoke(layers, gradients)

    assert torch.equal(layer.weight, initial)
    assert not torch.equal(first["test.weight"], initial)
    assert _compare_gradients(first, second)["exact"]
