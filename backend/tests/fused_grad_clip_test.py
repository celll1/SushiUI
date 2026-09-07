"""The per-parameter clip the fused backward pass can actually apply, and the
spike record that says which sample produced one.

Reproduces run127's shape: a long stretch of ordinary gradients, then one
thousands of times larger. Before this the large one entered unclipped -- there
is no global norm to clip against when each parameter is updated the moment its
own gradient exists.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.grad_spike_log import GradSpikeLog  # noqa: E402
from core.training.optimizers.fused_grad_clip import (  # noqa: E402
    FusedGradClipper,
    apply_fused_grad_clip,
    attach_fused_grad_clipper,
    get_fused_grad_clipper,
)


def _param(value: float, numel: int = 8) -> torch.nn.Parameter:
    p = torch.nn.Parameter(torch.zeros(numel))
    p.grad = torch.full((numel,), value)
    return p


def _norm(param) -> float:
    return float(torch.linalg.vector_norm(param.grad, ord=2))


def _settle(clipper, param, value: float, steps: int) -> None:
    for _ in range(steps):
        param.grad = torch.full_like(param, value)
        clipper.clip(param)


def test_an_ordinary_gradient_passes_through_untouched():
    clipper = FusedGradClipper(factor=8.0, warmup_steps=20)
    param = _param(1.0)
    _settle(clipper, param, 1.0, 40)

    param.grad = torch.full_like(param, 1.5)
    before = _norm(param)
    clipper.clip(param)
    assert _norm(param) == pytest.approx(before)
    assert clipper.take_step_summary() is None


def test_the_run127_spike_is_bounded_to_the_parameters_own_scale():
    """A gradient 4,500x the usual is cut to factor x the running scale."""
    clipper = FusedGradClipper(factor=8.0, warmup_steps=20)
    param = _param(1.0)
    _settle(clipper, param, 1.0, 40)
    scale = float(clipper._scale[id(param)])

    param.grad = torch.full_like(param, 4500.0)
    clipper.clip(param)

    assert _norm(param) == pytest.approx(scale * 8.0, rel=1e-4)
    summary = clipper.take_step_summary()
    assert summary["clipped_parameters"] == 1
    assert summary["worst_ratio"] > 1000


def test_a_sustained_burst_cannot_walk_the_bar_up_behind_itself():
    """A clipped step contributes the previous scale, not the threshold. Feeding
    the threshold back multiplies the scale by 1.07 per clipped step at the
    defaults, which defeats the clip in a few hundred steps."""
    clipper = FusedGradClipper(factor=8.0, warmup_steps=20)
    param = _param(1.0)
    _settle(clipper, param, 1.0, 40)
    before = float(clipper._scale[id(param)])

    for _ in range(200):
        param.grad = torch.full_like(param, 4500.0)
        clipper.clip(param)

    assert float(clipper._scale[id(param)]) == pytest.approx(before, rel=1e-4)
    assert _norm(param) == pytest.approx(before * 8.0, rel=1e-3)


def test_an_unclipped_gradient_still_moves_the_scale_both_ways():
    """The burst guard must not freeze the scale for ordinary drift."""
    clipper = FusedGradClipper(factor=8.0, warmup_steps=20)
    param = _param(1.0)
    _settle(clipper, param, 1.0, 40)
    settled = float(clipper._scale[id(param)])

    _settle(clipper, param, 3.0, 300)
    assert float(clipper._scale[id(param)]) > settled * 2.5
    _settle(clipper, param, 0.5, 300)
    assert float(clipper._scale[id(param)]) < settled


def test_nothing_is_clipped_during_warmup():
    """The scale is built from too few gradients to call anything an outlier."""
    clipper = FusedGradClipper(factor=8.0, warmup_steps=50)
    param = _param(1.0)
    _settle(clipper, param, 1.0, 10)

    param.grad = torch.full_like(param, 4500.0)
    clipper.clip(param)
    assert _norm(param) == pytest.approx(4500.0 * (8 ** 0.5), rel=1e-4)
    assert clipper.take_step_summary() is None


def test_each_parameter_has_its_own_scale():
    """A parameter whose gradients are legitimately 1000x another's must not be
    clipped against the other's history."""
    clipper = FusedGradClipper(factor=8.0, warmup_steps=20)
    small, large = _param(1.0), _param(1000.0)
    for _ in range(40):
        small.grad = torch.full_like(small, 1.0)
        large.grad = torch.full_like(large, 1000.0)
        clipper.clip(small)
        clipper.clip(large)

    large.grad = torch.full_like(large, 1500.0)
    before = _norm(large)
    clipper.clip(large)
    assert _norm(large) == pytest.approx(before)


def test_the_offender_is_named_without_a_per_parameter_sync():
    clipper = FusedGradClipper(factor=8.0, warmup_steps=5)
    quiet, loud = _param(1.0), _param(1.0)
    clipper.name_parameters({id(quiet): "layers.0.attn.q", id(loud): "layers.7.mlp.down"})
    for _ in range(20):
        quiet.grad = torch.full_like(quiet, 1.0)
        loud.grad = torch.full_like(loud, 1.0)
        clipper.clip(quiet)
        clipper.clip(loud)

    quiet.grad = torch.full_like(quiet, 30.0)
    loud.grad = torch.full_like(loud, 9000.0)
    clipper.clip(quiet)
    clipper.clip(loud)

    summary = clipper.take_step_summary()
    assert summary["clipped_parameters"] == 2
    assert summary["worst_parameter"] == "layers.7.mlp.down"


def test_the_summary_resets_between_steps():
    clipper = FusedGradClipper(factor=8.0, warmup_steps=5)
    param = _param(1.0)
    _settle(clipper, param, 1.0, 20)
    param.grad = torch.full_like(param, 4500.0)
    clipper.clip(param)
    assert clipper.take_step_summary()["clipped_parameters"] == 1
    assert clipper.take_step_summary() is None


def test_a_zero_factor_is_refused_rather_than_silently_disabling():
    """0 means the feature is off and setup_fused_grad_clip builds nothing; a
    clipper that accepted it would clip every gradient to zero."""
    with pytest.raises(ValueError, match="positive factor"):
        FusedGradClipper(factor=0.0, warmup_steps=10)


def test_the_hook_entry_point_is_inert_without_a_clipper():
    """Every fused site calls this; with the feature off it must cost an
    attribute lookup and change nothing."""
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=0.1)
    param = _param(4500.0)
    assert get_fused_grad_clipper(optimizer) is None
    apply_fused_grad_clip(optimizer, param)
    assert _norm(param) == pytest.approx(4500.0 * (8 ** 0.5), rel=1e-4)

    clipper = FusedGradClipper(factor=8.0, warmup_steps=2)
    attach_fused_grad_clipper(optimizer, clipper)
    for _ in range(10):
        param.grad = torch.full_like(param, 1.0)
        apply_fused_grad_clip(optimizer, param)
    param.grad = torch.full_like(param, 4500.0)
    apply_fused_grad_clip(optimizer, param)
    assert _norm(param) < 4500.0


def test_a_gradient_free_parameter_is_skipped():
    clipper = FusedGradClipper(factor=8.0, warmup_steps=2)
    param = torch.nn.Parameter(torch.zeros(4))
    clipper.clip(param)
    assert clipper.take_step_summary() is None


# --- the spike record ---------------------------------------------------------


def _batch(path: str, caption: str = "a caption"):
    return [({"image_path": path, "width": 1216, "height": 1728,
              "caption": caption}, object())]


def test_an_ordinary_step_writes_nothing(tmp_path):
    log = GradSpikeLog(tmp_path, factor=8.0, min_history=10, window=50)
    for step in range(40):
        assert log.observe(3.8, step=step, batch=_batch("x.png")) is None
    assert not log.path.exists()


def test_a_spike_records_the_batch_that_produced_it(tmp_path):
    log = GradSpikeLog(tmp_path, factor=8.0, min_history=10, window=50)
    for step in range(40):
        log.observe(3.8, step=step, batch=_batch("ordinary.png"))

    record = log.observe(
        17334.53, step=12355, epoch=0, loss=15.639, learning_rate=7.187e-06,
        batch=_batch("M:/data/offender.png", caption="the caption"),
        timesteps=torch.tensor([0.83, 0.21]))

    assert record is not None
    assert record["ratio"] == pytest.approx(17334.53 / 3.8, rel=1e-3)
    assert record["batch"][0]["path"] == "M:/data/offender.png"
    assert record["batch"][0]["caption"] == "the caption"
    assert record["batch"][0]["width"] == 1216
    assert record["timesteps"] == pytest.approx([0.83, 0.21], rel=1e-3)

    written = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
    assert len(written) == 1 and written[0]["step"] == 12355


def test_the_baseline_is_a_median_so_a_burst_cannot_hide_its_successors(tmp_path):
    """run127 spiked at 12335, 12343 and 12355. With a mean baseline the first
    would lift the bar over the third."""
    log = GradSpikeLog(tmp_path, factor=8.0, min_history=10, window=50)
    for step in range(40):
        log.observe(3.8, step=step, batch=_batch("x.png"))

    assert log.observe(5216.0, step=12335, batch=_batch("a.png")) is not None
    assert log.observe(2670.0, step=12343, batch=_batch("b.png")) is not None
    assert log.observe(17334.0, step=12355, batch=_batch("c.png")) is not None
    assert log.baseline() == pytest.approx(3.8)


def test_a_clipped_step_is_recorded_whatever_its_norm(tmp_path):
    """The clip is why the norm stayed down; judging the step by the resulting
    number would hide exactly the events this exists to catch."""
    log = GradSpikeLog(tmp_path, factor=8.0, min_history=10, window=50)
    for step in range(40):
        log.observe(3.8, step=step, batch=_batch("x.png"))

    record = log.observe(4.0, step=500, batch=_batch("clipped.png"),
                         clip_summary={"clipped_parameters": 3,
                                       "worst_ratio": 912.0,
                                       "worst_parameter": "layers.7.mlp.down"})
    assert record is not None
    assert record["clip"]["worst_parameter"] == "layers.7.mlp.down"
    assert record["ratio"] == pytest.approx(4.0 / 3.8, rel=1e-3)


def test_nothing_is_called_a_spike_before_the_window_fills(tmp_path):
    log = GradSpikeLog(tmp_path, factor=8.0, min_history=50, window=200)
    for step in range(49):
        assert log.observe(3.8, step=step, batch=_batch("x.png")) is None
    assert log.observe(17334.0, step=49, batch=_batch("early.png")) is None
    assert log.baseline() is not None


def test_a_zero_factor_disables_the_log(tmp_path):
    log = GradSpikeLog(tmp_path, factor=0.0)
    assert not log.enabled
    assert log.observe(17334.0, step=1, batch=_batch("x.png")) is None
    assert not log.path.exists()


def test_a_non_finite_norm_never_reaches_the_json(tmp_path):
    """json.dumps writes bare NaN, which is not valid JSON to a reader."""
    log = GradSpikeLog(tmp_path, factor=8.0, min_history=10, window=50)
    for step in range(40):
        log.observe(3.8, step=step, batch=_batch("x.png"))

    assert log.observe(float("nan"), step=100, batch=_batch("x.png")) is None
    record = log.observe(100.0, step=101, loss=float("inf"),
                         batch=_batch("x.png"))
    assert record["loss"] is None
    json.loads(log.path.read_text(encoding="utf-8").splitlines()[-1])


def test_the_file_stops_growing_but_the_count_does_not(tmp_path):
    """A run that spikes on every step is documented by the first records; the
    file must not grow beside a 33 GB checkpoint set."""
    log = GradSpikeLog(tmp_path, factor=8.0, min_history=10, window=50,
                       max_records=5)
    for step in range(40):
        log.observe(3.8, step=step, batch=_batch("x.png"))
    for step in range(100, 130):
        log.observe(17334.0, step=step, batch=_batch("spike.png"))

    written = log.path.read_text(encoding="utf-8").splitlines()
    assert len(written) == 5
    # Counted well past the cap. Not all 30: once the trailing window is mostly
    # spikes the median moves to them and they stop being outliers, which is the
    # median baseline behaving as intended for a sustained regime change.
    assert 5 < log.spikes_recorded < 30
    assert log.baseline() > 1000


# --- reach: a site that forgets the call is silently unprotected ---------------


_HOOK_SITES = (
    ("core/training/base_trainer.py", "the trainer's own per-parameter loop"),
    ("core/training/optimizers/fused_optimizer_groups.py", "Block Swap's optimizer groups"),
    ("core/training/optimizers/lion8bit_ringbuffer.py", "Lion8bit_RingBuffer"),
    ("core/training/optimizers/adamw8bit_ringbuffer.py", "AdamW8bit_RingBuffer"),
)


@pytest.mark.parametrize("relative,description", _HOOK_SITES)
def test_every_fused_hook_site_clips(relative, description):
    """The hooks are registered in four places and only the optimizer is in
    scope in all of them. A site that records the gradient norm but never clips
    leaves those parameters unprotected with nothing to show for it, so the
    check is: wherever the norm is recorded, the clip is applied too."""
    source = (Path(__file__).resolve().parents[1] / relative).read_text(encoding="utf-8")
    recorded = source.count("record_fused_grad_norm(")
    clipped = source.count("apply_fused_grad_clip(")
    assert recorded >= 1, f"{description}: no grad-norm recording site found"
    assert clipped == recorded, (
        f"{description}: {recorded} gradient-norm recording site(s) but "
        f"{clipped} clip site(s) -- a fused hook that records but does not clip "
        f"is silently unprotected")


def test_a_relora_reinit_forgets_the_scales_it_invalidated():
    """Merging the adapters into the base and re-initialising them changes what
    every gradient means; the old scale would clip the new adapters against the
    old ones' history."""
    clipper = FusedGradClipper(factor=8.0, warmup_steps=20)
    param = _param(1.0)
    _settle(clipper, param, 1.0, 40)
    assert clipper._scale

    clipper.reset_scales()
    assert not clipper._scale and not clipper._seen

    # Warmup restarts: the next large gradient is observed, not clipped.
    param.grad = torch.full_like(param, 4500.0)
    clipper.clip(param)
    assert _norm(param) == pytest.approx(4500.0 * (8 ** 0.5), rel=1e-4)
