"""Safety policy tests for MiniMax-H3's guarded FBCache controller."""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND = os.path.join(REPO_ROOT, "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from core.inference.fbcache import FirstBlockCache  # noqa: E402
from core.models.minimax_h3 import h3_pipeline_ops as ops  # noqa: E402


def _prime(cache, indicator, guard=None):
    assert cache.use_cache(indicator, 0, guard_indicator=guard) is False
    cache.store(torch.ones(1))


def test_temporal_guard_vetoes_a_global_mean_hit():
    cache = FirstBlockCache(threshold=0.08)
    previous = torch.ones(1, 100, 2)
    previous_frames = previous.reshape(10, 10, 2)
    _prime(cache, previous, previous_frames)

    current = previous.clone()
    current[:, :10] *= 1.5
    current_frames = current.reshape(10, 10, 2)

    assert FirstBlockCache._rel_l1(current, previous) < cache.threshold
    assert cache.use_cache(current, 1, guard_indicator=current_frames) is False


def test_consecutive_hit_cap_and_tail_force_real_steps():
    cache = FirstBlockCache(
        threshold=0.08,
        warmup_steps=1,
        max_consecutive_hits=2,
        total_steps=6,
        tail_steps=1,
    )
    indicator = torch.ones(1, 4, 2)
    _prime(cache, indicator)

    decisions = []
    for step in range(1, 6):
        decisions.append(cache.use_cache(indicator, step))
        if not decisions[-1]:
            cache.store(torch.ones(1))

    assert decisions == [True, True, False, True, False]
    assert (cache.n_hits, cache.n_miss) == (3, 3)


def test_default_controller_policy_is_unchanged_for_other_architectures():
    cache = FirstBlockCache(threshold=0.08, warmup_steps=0)
    indicator = torch.ones(1, 2)
    _prime(cache, indicator)

    assert [cache.use_cache(indicator, step) for step in range(1, 6)] == [True] * 5


class _Scheduler:
    def __init__(self, *, fail_step=False):
        self.timesteps = torch.tensor([0.5])
        self.fail_step = fail_step

    def set_shift(self, shift):
        self.shift = shift

    def set_timesteps(self, steps, device=None):
        if device is not None:
            self.timesteps = self.timesteps.to(device)

    def set_begin_index(self, index):
        self.begin_index = index

    def step(self, velocity, timestep, sample, return_dict=False):
        if self.fail_step:
            raise RuntimeError("scheduler failed")
        return (sample,)


class _Transformer:
    def __init__(self, num_video, num_audio, *, fail_forward=False):
        self.num_video = num_video
        self.num_audio = num_audio
        self.fail_forward = fail_forward
        self.attached = None

    def attach_fbcache(self, cache, **_geometry):
        self.attached = cache

    def __call__(self, **_kwargs):
        assert self.attached is not None
        if self.fail_forward:
            raise RuntimeError("transformer failed")
        return (
            torch.zeros(1, self.num_video, 96),
            torch.zeros(1, self.num_audio, 32),
        )


def _run_failing_denoise(*, fail_forward=False, fail_scheduler=False):
    layout = ops.build_packed_layout(3, 2, 4, 4, 5)
    num_video = int(layout["video_indices"].numel())
    num_audio = int(layout["audio_indices"].numel())
    transformer = _Transformer(num_video, num_audio, fail_forward=fail_forward)
    params = {
        "fbcache_enable": True,
        "fbcache_threshold": 0.08,
        "fbcache_warmup_steps": 1,
        "spectrum_enable": False,
    }
    with pytest.raises(RuntimeError):
        ops.denoise(
            transformer,
            _Scheduler(fail_step=fail_scheduler),
            _Scheduler(),
            prompt_embeds=torch.zeros(1, 3, 8),
            layout=layout,
            video_rows=torch.zeros(num_video, 96),
            audio_rows=torch.zeros(num_audio, 32),
            num_inference_steps=1,
            device="cpu",
            spectrum_params=params,
        )
    return transformer


def test_transformer_exception_detaches_fbcache():
    transformer = _run_failing_denoise(fail_forward=True)
    assert transformer.attached is None


def test_scheduler_exception_sees_fbcache_already_detached():
    transformer = _run_failing_denoise(fail_scheduler=True)
    assert transformer.attached is None
