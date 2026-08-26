"""Guard: train_runner's epochs->steps save-cadence arithmetic must stay
routed through one function.

train_runner.py had four network_type branches (lora, relora, full_finetune,
controlnet) each independently computing::

    steps_per_epoch = (len(dataset_items) + batch_size - 1) // batch_size
    save_every_n_steps = save_every * steps_per_epoch

All four read the exact same `dataset_items`/`train_config` (set once before
the branches split), so they never actually disagreed for any input -- this
is pure de-duplication into `_resolve_save_every_n_steps`, not a bug fix.
These tests pin the extracted function's behaviour (ceil division, unit
passthrough, boundary cases) and confirm the duplicate inline formula does
not reappear.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training.train_runner import _resolve_save_every_n_steps

BACKEND = Path(__file__).resolve().parent.parent
TRAIN_RUNNER_SRC = (BACKEND / "core" / "training" / "train_runner.py").read_text(encoding="utf-8")


def test_exact_division_no_partial_batch():
    # 100 items / batch 10 = exactly 10 batches/epoch, 3 epochs -> 30 steps.
    assert _resolve_save_every_n_steps("epochs", 3, 100, 10) == 30


def test_partial_last_batch_counts_as_a_step_ceil_division():
    # 105 items / batch 10 = 10 full batches + 1 partial -> 11 batches/epoch.
    assert _resolve_save_every_n_steps("epochs", 2, 105, 10) == 22


def test_single_item_smaller_than_batch_size_is_one_batch():
    assert _resolve_save_every_n_steps("epochs", 1, 1, 10) == 1


def test_steps_unit_passes_through_unchanged_regardless_of_dataset_size():
    assert _resolve_save_every_n_steps("steps", 500, 105, 10) == 500
    assert _resolve_save_every_n_steps("steps", 500, 0, 1) == 500


def test_batch_size_one_is_identity_on_item_count():
    assert _resolve_save_every_n_steps("epochs", 1, 37, 1) == 37


def test_save_every_zero_epochs_yields_zero_steps():
    assert _resolve_save_every_n_steps("epochs", 0, 105, 10) == 0


def test_all_four_network_type_branches_call_the_shared_helper():
    """Regression pin: the inline ceil-division formula must not be
    reintroduced at any of the four call sites (lora/relora/full_finetune/
    controlnet); they must all route through `_resolve_save_every_n_steps`.
    """
    assert TRAIN_RUNNER_SRC.count("_resolve_save_every_n_steps(") == 5  # def + 4 call sites
    assert "steps_per_epoch = (len(dataset_items)" not in TRAIN_RUNNER_SRC


def test_vae_decoder_branch_is_unaffected():
    """The vae_decoder branch computes total_steps from its own config
    resolver, not from `_resolve_save_every_n_steps`; this extraction must
    not have pulled it in."""
    idx = TRAIN_RUNNER_SRC.index("elif network_type == 'vae_decoder':")
    block = TRAIN_RUNNER_SRC[idx:idx + 2000]
    assert "_resolve_save_every_n_steps" not in block
    assert 'run.total_steps = int(vae_cfg["total_steps"])' in block
