"""Split latent-IO weights keep the merged checkpoint layout."""
import pytest
import torch
from torch import nn

from core.models.components.split_linear import SplitLinear


@pytest.mark.parametrize("axis", [0, 1])
def test_forward_state_and_gradients_match_merged_linear(axis):
    torch.manual_seed(3)
    linear = nn.Linear(10, 6, bias=False)
    fresh = [1, 4, 5] if axis == 0 else [0, 7, 9]
    split = SplitLinear(linear, fresh, axis=axis)
    value = torch.randn(4, 10)
    torch.testing.assert_close(split(value), linear(value))
    assert list(split.state_dict()) == ["weight"]
    assert torch.equal(split.state_dict()["weight"], linear.weight)
    split(value).square().sum().backward()
    linear(value).square().sum().backward()
    grad = torch.cat([split.weight_kept.grad, split.weight_fresh.grad], axis).index_select(axis, split._inverse)
    torch.testing.assert_close(grad, linear.weight.grad)
    other = nn.Linear(10, 6, bias=False)
    split.load_state_dict(other.state_dict())
    torch.testing.assert_close(split.merged().weight, other.weight)


def test_rejects_bias_and_improper_subsets():
    with pytest.raises(ValueError):
        SplitLinear(nn.Linear(4, 4), [1], axis=0)
    with pytest.raises(ValueError):
        SplitLinear(nn.Linear(4, 4, bias=False), [0, 1, 2, 3], axis=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_wraps_a_cuda_linear():
    linear = nn.Linear(8, 4, bias=False).cuda()
    split = SplitLinear(linear, [1, 2], axis=1)
    value = torch.randn(3, 8, device="cuda")
    torch.testing.assert_close(split(value), linear(value))
    assert split.state_dict()["weight"].device.type == "cuda"
