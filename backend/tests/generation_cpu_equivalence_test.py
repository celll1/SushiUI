from __future__ import annotations

import asyncio
import inspect
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from api.routes import _run_generation_in_executor  # noqa: E402
from core.inference.schedule_utils import snapshot_schedule_scalars  # noqa: E402
from core.inference.schedulers import SAMPLER_MAP  # noqa: E402


_SCHEDULERS_WITHOUT_PRED_ORIGINAL_SAMPLE = {
    "dpmpp_2m", "dpmpp_sde", "pndm", "unipc",
}


def _seeded_kernel():
    generator = torch.Generator(device="cpu").manual_seed(912_2026)
    sample = torch.randn((2, 4, 8, 8), generator=generator).requires_grad_(True)
    weight = torch.linspace(-0.75, 0.75, 16).reshape(4, 4)
    result = torch.einsum("ij,bjhw->bihw", weight, sample).tanh()
    continuation = torch.randn((17,), generator=generator)
    return result, continuation, result.requires_grad


def test_no_grad_executor_is_bit_exact_and_preserves_rng_advancement():
    with torch.enable_grad():
        expected, expected_continuation, expected_requires_grad = _seeded_kernel()

    async def run():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await _run_generation_in_executor(
                loop, executor, _seeded_kernel
            )

    actual, actual_continuation, actual_requires_grad = asyncio.run(run())

    assert expected_requires_grad
    assert not actual_requires_grad
    assert torch.equal(actual, expected)
    assert torch.equal(actual_continuation, expected_continuation)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_schedule_snapshot_is_bit_exact_to_per_element_item(dtype):
    source = torch.linspace(1.0, -0.125, 18, dtype=dtype)[::2]
    expected = [float(value.item()) for value in source]

    assert snapshot_schedule_scalars(source) == expected


def _step(scheduler, model_output, timestep, sample, generator):
    kwargs = {}
    if "generator" in inspect.signature(scheduler.step).parameters:
        kwargs["generator"] = generator
    return scheduler.step(model_output, timestep, sample, **kwargs)


def _state_tensors(value, seen=None):
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _state_tensors(item, seen)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _state_tensors(item, seen)


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


@pytest.mark.parametrize("sampler_name", tuple(SAMPLER_MAP))
def test_scheduler_preview_x0_does_not_alias_live_next_step_state(sampler_name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", UserWarning)
        scheduler = SAMPLER_MAP[sampler_name]()
        scheduler.set_timesteps(4, device="cpu")
        generator = torch.Generator(device="cpu").manual_seed(1234)
        sample = torch.randn((1, 4, 8, 8), generator=generator)
        observed_pred_x0 = 0
        for timestep in tuple(scheduler.timesteps):
            model_output = torch.randn(sample.shape, generator=generator)
            scheduler.scale_model_input(sample, timestep)
            output = _step(scheduler, model_output, timestep, sample, generator)
            pred_x0 = getattr(output, "pred_original_sample", None)
            if pred_x0 is not None:
                observed_pred_x0 += 1
                assert not _shares_storage(pred_x0, output.prev_sample)
                for name, value in vars(scheduler).items():
                    if name in {"timesteps", "sigmas", "alphas_cumprod"}:
                        continue
                    for state_tensor in _state_tensors(value):
                        assert not _shares_storage(pred_x0, state_tensor), (
                            f"{sampler_name} preview x0 aliases scheduler state {name}"
                        )
            sample = output.prev_sample

    assert (observed_pred_x0 == 0) == (
        sampler_name in _SCHEDULERS_WITHOUT_PRED_ORIGINAL_SAMPLE
    )
