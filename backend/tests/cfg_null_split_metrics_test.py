"""The null / conditional split of the charted loss and grad norm.

An item trained against the aligned null optimizes the caption-free marginal,
so the run's charted loss is a blend of two populations mixed at the drop rate
and cannot answer "is the conditional branch still improving". These pin what
the split emits, and -- more importantly -- what it REFUSES to emit when the
step cannot be attributed. Nothing here loads a checkpoint or a database.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/cfg_null_split_metrics_test.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.training.base_trainer import BaseTrainer  # noqa: E402
from core.training.metric_registry import EXTRA_METRIC_DEFS  # noqa: E402


class _StubTrainer:
    _log_cfg_null_loss_split = BaseTrainer._log_cfg_null_loss_split
    _log_cfg_null_grad_split = BaseTrainer._log_cfg_null_grad_split
    stash_cfg_null_per_sample_loss = BaseTrainer.stash_cfg_null_per_sample_loss
    log_extra_metric = BaseTrainer.log_extra_metric

    def __init__(self, rate=0.1):
        self._extra_metrics = {}
        self._rate = rate

    def cfg_null_drop_rate(self):
        return self._rate

    def drain(self):
        out, self._extra_metrics = self._extra_metrics, {}
        return out


def _mask(*bits):
    return torch.tensor([bool(b) for b in bits])


# ---------------------------------------------------------------------------
# Batch size 1: the scalar loss IS the item's loss
# ---------------------------------------------------------------------------

def test_a_null_step_at_batch_one_is_attributed_to_null():
    t = _StubTrainer()
    t._log_cfg_null_loss_split(_mask(1), 0.42)
    out = t.drain()
    assert out["loss_null"] == 0.42
    assert "loss_cond" not in out
    assert out["cfg_null_frac"] == 1.0


def test_a_conditional_step_at_batch_one_is_attributed_to_cond():
    t = _StubTrainer()
    t._log_cfg_null_loss_split(_mask(0), 0.09)
    out = t.drain()
    assert out["loss_cond"] == 0.09
    assert "loss_null" not in out
    assert out["cfg_null_frac"] == 0.0


# ---------------------------------------------------------------------------
# Mixed batches: exact with per-item loss, silent without it
# ---------------------------------------------------------------------------

def test_a_mixed_batch_splits_exactly_when_the_arch_stashed_per_item_loss():
    t = _StubTrainer()
    t._last_loss_per_sample = torch.tensor([1.0, 3.0, 0.2, 0.4])
    t._log_cfg_null_loss_split(_mask(1, 1, 0, 0), 1.15)
    out = t.drain()
    assert out["loss_null"] == 2.0
    assert abs(out["loss_cond"] - 0.3) < 1e-6
    assert out["cfg_null_frac"] == 0.5


def test_a_mixed_batch_without_per_item_loss_attributes_neither_side():
    """The batch mean is a blend; assigning it to the larger side would put a
    number on the chart that is not that side's loss."""
    t = _StubTrainer()
    t._log_cfg_null_loss_split(_mask(1, 0, 0, 0), 0.5)
    out = t.drain()
    assert "loss_null" not in out
    assert "loss_cond" not in out
    assert out["cfg_null_frac"] == 0.25


def test_a_homogeneous_batch_without_per_item_loss_is_still_attributed():
    t = _StubTrainer()
    t._log_cfg_null_loss_split(_mask(1, 1, 1), 0.7)
    out = t.drain()
    assert out["loss_null"] == 0.7
    assert "loss_cond" not in out


def test_a_stale_per_item_loss_of_the_wrong_length_is_not_used():
    t = _StubTrainer()
    t._last_loss_per_sample = torch.tensor([1.0, 2.0])
    t._log_cfg_null_loss_split(_mask(1, 1, 1, 1), 0.6)
    out = t.drain()
    assert out["loss_null"] == 0.6  # fell back to the scalar, not indexed


def test_the_per_item_loss_is_consumed_so_it_cannot_leak_into_the_next_step():
    t = _StubTrainer()
    t._last_loss_per_sample = torch.tensor([1.0, 3.0])
    t._log_cfg_null_loss_split(_mask(1, 0), 2.0)
    t.drain()
    assert t._last_loss_per_sample is None
    t._log_cfg_null_loss_split(_mask(1, 0), 2.0)
    out = t.drain()
    assert "loss_null" not in out and "loss_cond" not in out


# ---------------------------------------------------------------------------
# The stash itself
# ---------------------------------------------------------------------------

def test_the_stash_records_one_mse_per_item():
    t = _StubTrainer()
    pred = torch.tensor([[0.0, 0.0], [1.0, 3.0]])
    target = torch.zeros_like(pred)
    t.stash_cfg_null_per_sample_loss(pred, target)
    assert torch.allclose(t._last_loss_per_sample, torch.tensor([0.0, 5.0]))


def test_the_stash_is_skipped_at_batch_one():
    """Nothing to split, and the extra elementwise pass is not free."""
    t = _StubTrainer()
    t.stash_cfg_null_per_sample_loss(torch.zeros(1, 4), torch.zeros(1, 4))
    assert getattr(t, "_last_loss_per_sample", None) is None


def test_the_stash_is_skipped_when_the_mechanism_is_off():
    t = _StubTrainer(rate=0.0)
    t.stash_cfg_null_per_sample_loss(torch.zeros(4, 4), torch.ones(4, 4))
    assert getattr(t, "_last_loss_per_sample", None) is None


# ---------------------------------------------------------------------------
# Grad norm: labelled, never split
# ---------------------------------------------------------------------------

def test_a_grad_norm_from_an_all_null_window_is_labelled_null():
    t = _StubTrainer()
    t._log_cfg_null_loss_split(_mask(1), 0.4)
    t.drain()
    t._log_cfg_null_grad_split(1.25)
    assert t.drain()["gnorm_null"] == 1.25


def test_a_grad_norm_from_a_mixed_accumulation_window_is_not_labelled():
    """Two batches accumulated into one step, drawn differently: the norm is
    one number for both and belongs to neither."""
    t = _StubTrainer()
    t._log_cfg_null_loss_split(_mask(1), 0.4)
    t._log_cfg_null_loss_split(_mask(0), 0.1)
    t.drain()
    t._log_cfg_null_grad_split(1.25)
    out = t.drain()
    assert "gnorm_null" not in out and "gnorm_cond" not in out


def test_the_window_is_drained_so_a_later_step_is_not_labelled_from_it():
    t = _StubTrainer()
    t._log_cfg_null_loss_split(_mask(1), 0.4)
    t.drain()
    t._log_cfg_null_grad_split(1.0)
    t.drain()
    t._log_cfg_null_grad_split(2.0)
    assert t.drain() == {}


def test_no_grad_label_without_the_mechanism():
    t = _StubTrainer()
    t._log_cfg_null_grad_split(1.0)
    assert t.drain() == {}


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------

def test_every_emitted_series_is_registered_for_the_chart():
    for name in ("loss_null", "loss_cond", "cfg_null_frac",
                 "gnorm_null", "gnorm_cond"):
        assert name in EXTRA_METRIC_DEFS, name


def test_the_two_loss_series_share_the_primary_axis_with_loss():
    """They are the same quantity as the main series; putting them on the
    secondary axis would let them scale independently of it."""
    for name in ("loss_null", "loss_cond"):
        assert "axis" not in EXTRA_METRIC_DEFS[name]


def test_the_draw_fraction_is_on_its_own_axis():
    """A [0,1] fraction pooled into a loss Y-range flattens the losses."""
    assert EXTRA_METRIC_DEFS["cfg_null_frac"]["axis"] == "right"


def test_the_loss_split_and_the_grad_label_land_on_the_same_step():
    """_log_metrics_to_db merges the accumulator per step, and the grad label is
    emitted before that step's row is written -- so a reader gets one row
    carrying both, not a loss row and a norm row one step apart."""
    t = _StubTrainer()
    t._log_cfg_null_loss_split(_mask(1), 0.4)
    t._log_cfg_null_grad_split(1.25)
    out = t.drain()
    assert out["loss_null"] == 0.4
    assert out["gnorm_null"] == 1.25
    assert out["cfg_null_frac"] == 1.0


def test_a_stale_stash_cannot_survive_into_the_next_batch():
    """Batch size is constant, so a per-item tensor parked by a batch that was
    then skipped would pass the length check and be read against the NEXT
    batch's labels. The train loop clears it where the label is drawn; this
    pins that the clear is what the loop does."""
    source = (BACKEND / "core" / "training"
              / "base_trainer.py").read_text(encoding="utf-8")
    draw = source.index("cfg_drop_mask = self.sample_cfg_drop_mask(len(batch))")
    assert "self._last_loss_per_sample = None" in source[draw:draw + 600]
