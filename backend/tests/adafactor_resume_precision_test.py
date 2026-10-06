"""Adafactor resume preserves FP32 moments and the next parameter update."""

import pytest
import torch
from transformers import Adafactor

from core.training.optimizers.adafactor_fused import patch_adafactor_fused, patch_adafactor_state_loading


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("beta1", [None, .9])
@pytest.mark.parametrize("fused", [False, True])
def test_resume_preserves_moments_and_next_update(dtype, beta1, fused):
    values = [torch.linspace(-1, 1, 12).reshape(3, 4), torch.linspace(-1, 1, 4)]
    original = [torch.nn.Parameter(value.to(dtype)) for value in values]
    kwargs = dict(lr=.01, relative_step=False, scale_parameter=False, warmup_init=False, beta1=beta1)
    optimizer = Adafactor(original, **kwargs)
    patch = patch_adafactor_fused if fused else patch_adafactor_state_loading
    patch(optimizer)
    for parameter in original:
        parameter.grad = torch.linspace(.02, .5, parameter.numel()).reshape(parameter.shape).to(dtype)
    optimizer.step()
    saved = optimizer.state_dict()
    snapshot = {index: {key: value.clone() if isinstance(value, torch.Tensor) else value
                       for key, value in state.items()} for index, state in saved["state"].items()}
    restored = [torch.nn.Parameter(parameter.detach().clone()) for parameter in original]
    resumed = Adafactor(restored, **kwargs)
    patch(resumed)
    patch(resumed)
    resumed.load_state_dict(saved)
    for left, right in zip(original, restored):
        for key, value in optimizer.state[left].items():
            actual = resumed.state[right][key]
            if isinstance(value, torch.Tensor):
                assert actual.dtype == value.dtype
                torch.testing.assert_close(actual, value, rtol=0, atol=0)
            else:
                assert actual == value
        for parameter in (left, right):
            parameter.grad = torch.linspace(-.3, .7, parameter.numel()).reshape(parameter.shape).to(dtype)
    resumed.step()
    for index, state in snapshot.items():
        for key, value in state.items():
            actual = saved["state"][index][key]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual, value, rtol=0, atol=0)
            else:
                assert actual == value
    optimizer.step()
    for left, right in zip(original, restored):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        for key, value in optimizer.state[left].items():
            actual = resumed.state[right][key]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual, value, rtol=0, atol=0)
            else:
                assert actual == value


def test_factory_patches_restore_without_replacing_standard_step():
    from core.training.optimizer_factory import OptimizerFactory
    parameter = torch.nn.Parameter(torch.ones(3, 4, dtype=torch.bfloat16))
    optimizer = OptimizerFactory.create_optimizer("adafactor", [parameter], learning_rate=.01)
    assert optimizer.load_state_dict.__func__.__name__ == "adafactor_load_state_dict"
    assert optimizer.step.__func__ is Adafactor.step
