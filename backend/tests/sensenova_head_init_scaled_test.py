"""The 'scaled' head init must land x0_pred at unit variance from any activation scale."""
import math
import pytest
import torch

from core.models.sensenova.latent_space import (
    LATENT_HEAD_TARGET_STD, _calibrate_output_scale,
)
from core.models.sensenova.vendor.modeling_fm_modules import ConvDecoder


def _seeded_head(out_channels=4, shuffle=2):
    torch.manual_seed(0)
    head = ConvDecoder(input_dim=256, hidden_dim=64, out_channels=out_channels,
                       shuffle=shuffle)
    conv = head.conv2
    std = 1.0 / math.sqrt(conv.in_channels * 9)
    with torch.no_grad():
        torch.nn.init.trunc_normal_(conv.weight, std=std, a=-3*std, b=3*std)
        conv.bias.zero_()
    return head


@pytest.mark.parametrize("activation_scale", [0.01, 1.0, 40.0])
def test_calibration_hits_target_from_any_activation_scale(activation_scale):
    head = _seeded_head()
    _calibrate_output_scale(head.conv2, LATENT_HEAD_TARGET_STD)
    torch.manual_seed(1)
    x = torch.randn(2, 256, 6, 6) * activation_scale
    out = head(x)
    assert out.std().item() == pytest.approx(LATENT_HEAD_TARGET_STD, rel=0.02)


def test_calibration_runs_once_and_leaves_no_hook():
    head = _seeded_head()
    _calibrate_output_scale(head.conv2, LATENT_HEAD_TARGET_STD)
    torch.manual_seed(1)
    head(torch.randn(2, 256, 6, 6) * 7.0)
    assert not head.conv2._forward_pre_hooks
    weight = head.conv2.weight.detach().clone()
    head(torch.randn(2, 256, 6, 6) * 0.001)          # a later, differently scaled batch
    assert torch.equal(head.conv2.weight, weight)


def test_calibrated_forward_stays_differentiable_and_consistent():
    """The rescale happens before the graph is built, so the gradient matches the
    weights the parameter actually holds."""
    head = _seeded_head()
    _calibrate_output_scale(head.conv2, LATENT_HEAD_TARGET_STD)
    torch.manual_seed(1)
    x = torch.randn(1, 256, 6, 6) * 3.0
    out = head(x)
    out.pow(2).mean().backward()
    conv = head.conv2
    ref = ConvDecoder(input_dim=256, hidden_dim=64, out_channels=4, shuffle=2)
    ref.load_state_dict(head.state_dict())
    ref.zero_grad()
    ref(x).pow(2).mean().backward()
    assert torch.allclose(conv.weight.grad, ref.conv2.weight.grad, atol=1e-6)


def test_zero_init_predicts_a_constant():
    """The behaviour 'scaled' exists to replace, asserted rather than assumed."""
    head = _seeded_head()
    with torch.no_grad():
        head.conv2.weight.zero_(); head.conv2.bias.zero_()
    out = head(torch.randn(2, 256, 6, 6) * 5.0)
    assert out.std().item() == 0.0
