"""MiniMax-H3 Spectrum forecasts video and audio as one opt-in step decision."""

import sys
from pathlib import Path

import torch

_BACKEND = str(Path(__file__).resolve().parents[1])
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core.models.minimax_h3 import h3_pipeline_ops as ops  # noqa: E402


class _Scheduler:
    def __init__(self, count: int):
        self.timesteps = torch.linspace(0.9, 0.0, count)

    def set_shift(self, shift):
        self.shift = shift

    def set_timesteps(self, steps, device=None):
        if device is not None:
            self.timesteps = self.timesteps.to(device)

    def set_begin_index(self, index):
        self.begin_index = index

    def step(self, velocity, timestep, sample, return_dict=False):
        return (sample - 0.01 * velocity,)


def _params(enabled=True):
    return {
        "spectrum_enable": enabled,
        "spectrum_w": 0.5,
        "spectrum_w_decay": 0.0,
        "spectrum_delta_cap": 0.0,
        "spectrum_m": 4,
        "spectrum_lam": 0.1,
        "spectrum_warmup_steps": 3,
        "spectrum_window_size": 4,
        "spectrum_flex_window": 0.75,
        "spectrum_tail": 0.12,
        "spectrum_feature_mode": "output",
        "spectrum_max_cache": 0,
    }


def _run(*, enabled=True, block_swap_on=False):
    steps = 10
    layout = ops.build_packed_layout(3, 2, 4, 4, 5)
    num_video = int(layout["video_indices"].numel())
    num_audio = int(layout["audio_indices"].numel())
    calls = []

    def transformer(**kwargs):
        calls.append(kwargs)
        value = float(len(calls))
        return (
            torch.full((1, num_video, 96), value),
            torch.full((1, num_audio, 32), value * 10.0),
        )

    progress = []
    video, audio = ops.denoise(
        transformer,
        _Scheduler(steps),
        _Scheduler(steps),
        prompt_embeds=torch.zeros(1, 3, 8),
        layout=layout,
        video_rows=torch.zeros(num_video, 96),
        audio_rows=torch.zeros(num_audio, 32),
        num_inference_steps=steps,
        device="cpu",
        progress_callback=lambda completed, total: progress.append((completed, total)),
        spectrum_params=_params(enabled),
        block_swap_on=block_swap_on,
    )
    return calls, progress, video, audio


def test_spectrum_skips_whole_transformer_calls_for_both_outputs(capsys):
    calls, progress, video, audio = _run()

    assert len(calls) == 8
    assert progress == [(i, 10) for i in range(1, 11)]
    assert torch.isfinite(video).all()
    assert torch.isfinite(audio).all()
    output = capsys.readouterr().out
    assert "paired video/audio final-output forecasting" in output
    assert "8 anchor(s), 2 forecast(s) of 10 step(s)" in output


def test_spectrum_is_disabled_when_block_swap_is_active(capsys):
    calls, progress, _, _ = _run(block_swap_on=True)

    assert len(calls) == 10
    assert len(progress) == 10
    assert "Block Swap is enabled" in capsys.readouterr().out


def test_spectrum_off_keeps_the_exact_forward_count():
    calls, progress, _, _ = _run(enabled=False)

    assert len(calls) == 10
    assert len(progress) == 10


def test_generation_threads_request_params_and_block_swap_state_to_the_loop():
    source = (Path(_BACKEND) / "core" / "pipeline_backends" / "minimax_h3.py").read_text(
        encoding="utf-8")

    assert "spectrum_params=params" in source
    assert 'block_swap_on=int(params.get("blocks_to_swap", 0) or 0) > 0' in source
