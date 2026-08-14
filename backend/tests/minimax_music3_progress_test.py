"""MiniMax Music 3 txt2aud backend: progress-weighting math (weight-free).

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_music3_progress_test.py -v

Covers `compute_progress_budget`/`combined_progress`/`estimate_num_chunks`
in `core.pipeline_backends.minimax_music3` -- see that module's docstring
("Progress weighting -- the basis, and its limits") for the forward-call-
count heuristic these pin. Nothing here loads a model.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.models.minimax_music3.defaults import CHUNK_FRAMES, CHUNK_HOP
from core.pipeline_backends.minimax_music3 import (
    PROGRESS_TOTAL_UNITS,
    combined_progress,
    compute_progress_budget,
    estimate_num_chunks,
)


def test_estimate_num_chunks_matches_prepare_chunks_single_window():
    assert estimate_num_chunks(CHUNK_FRAMES, CHUNK_FRAMES, CHUNK_HOP) == 1
    assert estimate_num_chunks(CHUNK_FRAMES - 1, CHUNK_FRAMES, CHUNK_HOP) == 1


def test_estimate_num_chunks_matches_prepare_chunks_multi_window():
    # Pinned against `MiniMaxMusic3Pipeline.prepare_chunks`'s own test fixture
    # (minimax_music3_chunk_geometry_test.py): 550 frames -> [0, 100, 200, 300, 400].
    assert estimate_num_chunks(550, CHUNK_FRAMES, CHUNK_HOP) == 5
    assert estimate_num_chunks(CHUNK_FRAMES + 1, CHUNK_FRAMES, CHUNK_HOP) == len(
        range(0, (CHUNK_FRAMES + 1) - CHUNK_HOP, CHUNK_HOP)
    )


def test_budgets_sum_to_total_units():
    ar_budget, flow_budget = compute_progress_budget(
        max_frames=7500, num_codebooks=8, num_inference_steps=30,
        chunk_frames=CHUNK_FRAMES, chunk_hop=CHUNK_HOP,
    )
    assert ar_budget + flow_budget == PROGRESS_TOTAL_UNITS
    assert ar_budget >= 0 and flow_budget >= 0


def test_long_song_ar_dominates_the_budget():
    """A 300s song (design doc: 7,500 AR steps * 8 sub-calls each) against a
    typical flow configuration must have AR claim the overwhelming majority
    of the combined progress bar -- the qualitative behavior the module
    docstring's forward-call heuristic is built to reproduce (the whole
    reason a naive "count each stage's own reported ticks equally" approach
    was rejected: it would make the AR stage look frozen)."""
    ar_budget, flow_budget = compute_progress_budget(
        max_frames=7500, num_codebooks=8, num_inference_steps=30,
        chunk_frames=CHUNK_FRAMES, chunk_hop=CHUNK_HOP,
    )
    assert ar_budget > flow_budget
    assert ar_budget / PROGRESS_TOTAL_UNITS > 0.9


def test_short_song_still_gives_flow_a_nonzero_budget():
    # A short clip's AR stage is proportionally cheaper (fewer frames) while
    # the flow stage still runs at least one chunk * num_inference_steps.
    ar_budget, flow_budget = compute_progress_budget(
        max_frames=25, num_codebooks=8, num_inference_steps=30,
        chunk_frames=CHUNK_FRAMES, chunk_hop=CHUNK_HOP,
    )
    assert flow_budget > 0


def test_zero_length_request_does_not_divide_by_zero():
    ar_budget, flow_budget = compute_progress_budget(
        max_frames=0, num_codebooks=8, num_inference_steps=0,
        chunk_frames=CHUNK_FRAMES, chunk_hop=CHUNK_HOP,
    )
    assert ar_budget + flow_budget == PROGRESS_TOTAL_UNITS


def test_combined_progress_ar_stage_stays_within_its_budget():
    ar_budget, flow_budget = 6000, 4000
    for step in range(0, 101, 10):
        combined = combined_progress("ar", step, 100, ar_budget, flow_budget)
        assert 0 <= combined <= ar_budget


def test_combined_progress_flow_stage_starts_at_ar_budget_and_ends_at_total():
    ar_budget, flow_budget = 6000, 4000
    assert combined_progress("flow", 0, 100, ar_budget, flow_budget) == ar_budget
    assert combined_progress("flow", 100, 100, ar_budget, flow_budget) == PROGRESS_TOTAL_UNITS


def test_combined_progress_is_monotonic_across_the_full_two_stage_sequence():
    ar_budget, flow_budget = compute_progress_budget(
        max_frames=500, num_codebooks=8, num_inference_steps=10,
        chunk_frames=CHUNK_FRAMES, chunk_hop=CHUNK_HOP,
    )
    ar_total = 500
    flow_total = estimate_num_chunks(500, CHUNK_FRAMES, CHUNK_HOP) * 10

    previous = -1
    for step in range(0, ar_total + 1):
        combined = combined_progress("ar", step, ar_total, ar_budget, flow_budget)
        assert combined >= previous
        previous = combined
    for step in range(0, flow_total + 1):
        combined = combined_progress("flow", step, flow_total, ar_budget, flow_budget)
        assert combined >= previous
        previous = combined
    assert previous == PROGRESS_TOTAL_UNITS


def test_combined_progress_handles_ar_early_stop_without_reaching_its_full_budget():
    # AR ending before `max_frames` (the checkpoint's own end-of-audio token)
    # legitimately never reaches `ar_budget` -- the combined series is still
    # well-formed (non-negative, monotonic within what ran); the CALLER
    # (the backend's own final unconditional callback) is what forces the
    # series to PROGRESS_TOTAL_UNITS at the very end, not this function.
    ar_budget, flow_budget = 6000, 4000
    combined = combined_progress("ar", 40, 100, ar_budget, flow_budget)  # stopped at 40% of its own total
    assert combined == round(ar_budget * 0.4)
    assert combined < ar_budget


def test_combined_progress_rejects_an_unknown_stage():
    with pytest.raises(ValueError):
        combined_progress("decode", 1, 1, 5000, 5000)


def test_combined_progress_clamps_an_out_of_range_step():
    ar_budget, flow_budget = 6000, 4000
    # step > total must not overshoot the stage's own budget.
    assert combined_progress("ar", 999, 100, ar_budget, flow_budget) == ar_budget
    # negative step must not underflow.
    assert combined_progress("ar", -5, 100, ar_budget, flow_budget) == 0
