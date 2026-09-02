"""What a resume keeps when a param group changes SIZE, not just count.

``_load_one_optimizer_state``'s fallback used to salvage state only when the
sole difference was a trailing param GROUP added or removed (the REPA projector
case). A group that GREW -- ``sensenova_train_fm_modules`` appends 16 fm_modules
params to the END of the generation group, 294 -> 310 -- reset every moment of
all 16.2B trained parameters instead.

The saved ``state`` is keyed by a FLAT parameter index in param_groups order, so
once any group changes size every LATER group's offsets shift: the entries must
be re-keyed, not filtered. ``test_growing_group_remaps_later_groups`` is the one
that fails if they are merely filtered.

CPU only; no model, no dataset.

Run:
    venv/Scripts/python.exe -m pytest backend/tests/optimizer_state_group_prefix_remap_test.py -v
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.base_trainer import BaseTrainer


class _Trainer:
    _load_one_optimizer_state = BaseTrainer._load_one_optimizer_state
    _remap_optimizer_state_by_group_prefix = BaseTrainer._remap_optimizer_state_by_group_prefix
    _optimizer_state_entry_fits_param = staticmethod(
        BaseTrainer._optimizer_state_entry_fits_param)

    def __init__(self):
        self.log_prefix = "[test]"
        self.device = torch.device("cpu")
        self.optimizer_state_host_resident = False
        self._optimizer_state_partially_fresh = False


def _params(n, numel=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [nn.Parameter(torch.rand(numel, generator=generator)) for _ in range(n)]


def _adamw(groups):
    return torch.optim.AdamW([{"params": g, "lr": 1e-4} for g in groups])


def _stepped(groups, scale=1.0):
    """An optimizer with one real step behind it, and its state_dict."""
    optimizer = _adamw(groups)
    for i, group in enumerate(groups):
        for j, p in enumerate(group):
            p.grad = torch.full_like(p, scale * (i + 1) * (j + 1))
    optimizer.step()
    return optimizer, optimizer.state_dict()


def _load(trainer, optimizer, saved, label="opt.pt"):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        ok = trainer._load_one_optimizer_state(optimizer, saved, label)
    return ok, buffer.getvalue()


def _moment(optimizer, param, key="exp_avg"):
    return optimizer.state[param][key]


# ---------------------------------------------------------------------------
# Unchanged behaviour
# ---------------------------------------------------------------------------

def test_identical_layout_loads_fully_and_sets_no_flag():
    model = _params(3)
    _, saved = _stepped([model])
    expected = [saved["state"][i]["exp_avg"].clone() for i in range(3)]

    live = _adamw([model])
    trainer = _Trainer()
    ok, printed = _load(trainer, live, saved)

    assert ok is True
    assert "Partial optimizer state load OK" not in printed
    for p, want in zip(model, expected):
        assert torch.equal(_moment(live, p), want)
    assert trainer._optimizer_state_partially_fresh is False


def test_trailing_group_added_keeps_the_model_groups():
    """The shipped REPA case: unchanged salvage, projector fresh."""
    model = _params(2)
    _, saved = _stepped([[model[0]], [model[1]]])
    expected = [saved["state"][i]["exp_avg"].clone() for i in range(2)]

    projector = _params(1, seed=7)
    live = _adamw([[model[0]], [model[1]], projector])
    trainer = _Trainer()
    ok, printed = _load(trainer, live, saved)

    assert ok is True
    assert "Partial optimizer state load OK" in printed
    for p, want in zip(model, expected):
        assert torch.equal(_moment(live, p), want)
    assert projector[0] not in live.state
    assert trainer._optimizer_state_partially_fresh is True


def test_trailing_group_removed_keeps_everything_live():
    model = _params(2)
    projector = _params(1, seed=7)
    _, saved = _stepped([[model[0]], [model[1]], projector])
    expected = [saved["state"][i]["exp_avg"].clone() for i in range(2)]

    live = _adamw([[model[0]], [model[1]]])
    trainer = _Trainer()
    ok, printed = _load(trainer, live, saved)

    assert ok is True
    assert "Partial optimizer state load OK" in printed
    for p, want in zip(model, expected):
        assert torch.equal(_moment(live, p), want)
    # Nothing live came up empty, so no re-warmup is warranted.
    assert trainer._optimizer_state_partially_fresh is False


# ---------------------------------------------------------------------------
# The fm_modules case: a LEADING group grows by appending
# ---------------------------------------------------------------------------

def test_growing_group_remaps_later_groups():
    """Generation group 294 -> 310 (fm_modules appended); understanding group
    unchanged. The 294 keep their exact state, the 16 are fresh, and the
    understanding group's state is re-keyed onto its NEW offsets."""
    generation = _params(294, numel=4, seed=1)
    fm_modules = _params(16, numel=4, seed=2)
    understanding = _params(8, numel=4, seed=3)

    _, saved = _stepped([generation, understanding])
    saved_generation = [saved["state"][i]["exp_avg"].clone() for i in range(294)]
    saved_understanding = [saved["state"][294 + i]["exp_avg"].clone() for i in range(8)]

    live = _adamw([generation + fm_modules, understanding])
    trainer = _Trainer()
    ok, printed = _load(trainer, live, saved)

    assert ok is True
    assert "Partial optimizer state load OK" in printed
    for p, want in zip(generation, saved_generation):
        assert torch.equal(_moment(live, p), want)
    for p in fm_modules:
        assert p not in live.state
    # Read from the stale indices this would be off by 16 and every value wrong.
    for p, want in zip(understanding, saved_understanding):
        assert torch.equal(_moment(live, p), want)
    assert trainer._optimizer_state_partially_fresh is True


def test_growing_group_reports_the_counts():
    generation = _params(6, numel=4, seed=1)
    fm_modules = _params(2, numel=4, seed=2)
    understanding = _params(3, numel=4, seed=3)
    _, saved = _stepped([generation, understanding])

    live = _adamw([generation + fm_modules, understanding])
    trainer = _Trainer()
    ok, printed = _load(trainer, live, saved)

    assert ok is True
    assert "9 of 11 live parameter tensor(s) kept their saved state, 2 start fresh" in printed
    assert "group 0: 6 kept / 2 fresh (6 saved -> 8 live)" in printed
    assert "group 1: 3 kept / 0 fresh (3 saved -> 3 live)" in printed


def test_shrinking_group_keeps_the_leading_prefix():
    generation = _params(6, numel=4, seed=1)
    dropped = _params(2, numel=4, seed=2)
    understanding = _params(3, numel=4, seed=3)

    _, saved = _stepped([generation + dropped, understanding])
    saved_generation = [saved["state"][i]["exp_avg"].clone() for i in range(6)]
    saved_understanding = [saved["state"][8 + i]["exp_avg"].clone() for i in range(3)]

    live = _adamw([generation, understanding])
    trainer = _Trainer()
    ok, printed = _load(trainer, live, saved)

    assert ok is True
    for p, want in zip(generation, saved_generation):
        assert torch.equal(_moment(live, p), want)
    for p, want in zip(understanding, saved_understanding):
        assert torch.equal(_moment(live, p), want)
    assert trainer._optimizer_state_partially_fresh is False


# ---------------------------------------------------------------------------
# The safety guard
# ---------------------------------------------------------------------------

def test_size_mismatch_refuses_the_remap():
    """Growth that is not append-only: the leading params are DIFFERENT
    tensors, which positional matching cannot detect except by size."""
    saved_params = _params(2, numel=8, seed=1)
    _, saved = _stepped([saved_params])

    live_params = [nn.Parameter(torch.rand(5)), nn.Parameter(torch.rand(8)),
                   nn.Parameter(torch.rand(8))]
    live = _adamw([live_params])
    trainer = _Trainer()
    ok, printed = _load(trainer, live, saved)

    assert ok is False
    assert "does not fit the live parameter" in printed
    assert "fresh optimizer state" in printed
    assert len(live.state) == 0


def test_non_tensor_entries_do_not_break_the_check():
    """torch keeps ``step`` as a 0-dim tensor; other optimizers keep an int."""
    param = nn.Parameter(torch.rand(8))
    entry = {"step": 4, "exp_avg": torch.zeros(8), "exp_avg_sq": torch.zeros(8)}
    assert BaseTrainer._optimizer_state_entry_fits_param(entry, param) is True
    entry["step"] = torch.tensor(4.0)
    assert BaseTrainer._optimizer_state_entry_fits_param(entry, param) is True


def test_eight_bit_uint8_state_fits_by_numel():
    """AdamW8bit_RingBuffer: uint8 buffers of p.numel(), small absmax blocks."""
    param = nn.Parameter(torch.rand(4096))
    entry = {
        "exp_avg": torch.zeros(4096, dtype=torch.uint8),
        "exp_avg_sq": torch.zeros(4096, dtype=torch.uint8),
        "absmax1": torch.zeros(16),
        "absmax2": torch.zeros(16),
        "is_8bit": True,
        "step": 7,
    }
    assert BaseTrainer._optimizer_state_entry_fits_param(entry, param) is True
    entry["z"] = torch.zeros(4096, dtype=torch.uint8)  # Schedule-Free
    assert BaseTrainer._optimizer_state_entry_fits_param(entry, param) is True
    assert BaseTrainer._optimizer_state_entry_fits_param(
        entry, nn.Parameter(torch.rand(2048))) is False


def test_factored_adafactor_state_fits_by_shape():
    """Adafactor keeps NO parameter-sized tensor when factored: row/col only."""
    param = nn.Parameter(torch.rand(6, 10))
    entry = {
        "step": 3,
        "RMS": 0.0,
        "exp_avg_sq_row": torch.zeros(6),
        "exp_avg_sq_col": torch.zeros(10),
    }
    assert BaseTrainer._optimizer_state_entry_fits_param(entry, param) is True
    assert BaseTrainer._optimizer_state_entry_fits_param(
        entry, nn.Parameter(torch.rand(10, 6))) is False


def test_no_overlapping_groups_is_refused():
    param = nn.Parameter(torch.rand(4))
    live = _adamw([[param]])
    saved = {"state": {}, "param_groups": []}
    trainer = _Trainer()
    ok, printed = _load(trainer, live, saved)

    assert ok is False
    assert "fresh optimizer state" in printed


def test_nothing_salvaged_is_reported_as_a_failure():
    """A layout that overlaps but whose checkpoint holds no state for the
    overlap is not a partial load: ``False`` re-arms the warmup by itself."""
    live = _adamw([_params(1, numel=4)])
    saved = {"state": {}, "param_groups": [{"params": [0, 1], "lr": 1e-4}]}
    trainer = _Trainer()
    ok, printed = _load(trainer, live, saved)

    assert ok is False
    assert "fresh optimizer state" in printed


def test_partially_fresh_flag_tracks_only_actual_freshness():
    """Cross-check of the flag against the cases above, in one place."""
    cases = []

    model = _params(2)
    _, saved = _stepped([[model[0]], [model[1]]])

    trainer = _Trainer()
    _load(trainer, _adamw([[model[0]], [model[1]]]), saved)
    cases.append(("identical", trainer._optimizer_state_partially_fresh, False))

    trainer = _Trainer()
    _load(trainer, _adamw([[model[0]], [model[1]], _params(1, seed=7)]), saved)
    cases.append(("group added", trainer._optimizer_state_partially_fresh, True))

    trainer = _Trainer()
    _load(trainer, _adamw([[model[0]], [model[1], *_params(3, seed=8)]]), saved)
    cases.append(("group grew", trainer._optimizer_state_partially_fresh, True))

    trainer = _Trainer()
    _load(trainer, _adamw([[model[0]]]), saved)
    cases.append(("group removed", trainer._optimizer_state_partially_fresh, False))

    for name, actual, expected in cases:
        assert actual is expected, name


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
