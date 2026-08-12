"""Partial audio pin: a SUBSET of the audio track is conditioning, not the whole one.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_audio_pin_test.py -v

WHY THIS FILE EXISTS
--------------------
`h3_pipeline_ops.build_packed_layout` already reached a WHOLE-track audio pin
(`pin_target_audio`, ia2v) and a PARTIAL video pin (`pinned_video_frames`, temporal
inpaint). `scratchpad/minimax_h3_ai_probe_results.md` measured that the released
fl2va weights honour a PARTIAL audio pin too, at the decoder floor, and this file
pins the mechanism that probe monkeypatched: `pinned_audio_latents` on the same
builder, permuting the CHANNEL-MAJOR audio row block so an arbitrary temporal SET
(not a channel prefix) can be the conditioning prefix.

Most of the ops-level assertions below are a direct port of the real-function
checks in `scratchpad/ai/ai1_audio_pin_harness.py --selftest` ("the real layout,
on CPU, with the pin applied"), which is the probe's own guarantee that these
properties hold against the SHIPPED code rather than a hand re-derivation.
"""

import inspect
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.generation_utils import plan_audio_pin_latents  # noqa: E402
from core.models.minimax_h3 import h3_pipeline_ops as ops  # noqa: E402

NUM_TEXT_TOKENS = 11
LATENT_FRAMES = 7
LATENT_HEIGHT, LATENT_WIDTH = 24, 40
NUM_AUDIO_LATENTS = 37     # ops.audio_latent_frames(22) -- a 22-frame clip's grid


def _layout(pinned_audio_latents=(), **kwargs):
    return ops.build_packed_layout(
        NUM_TEXT_TOKENS, LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH, NUM_AUDIO_LATENTS,
        pinned_audio_latents=pinned_audio_latents, **kwargs,
    )


def _middle_third_pinned():
    """The complement of the probe's own "middle third free" split."""
    third = NUM_AUDIO_LATENTS // 3
    free_lo, free_hi = third, NUM_AUDIO_LATENTS - third
    return tuple(t for t in range(NUM_AUDIO_LATENTS) if not free_lo <= t < free_hi)


# --------------------------------------------------------------------------
# build_packed_layout: the permutation and what it does to num_condition_audio_rows
# --------------------------------------------------------------------------

def test_no_pin_leaves_the_permutation_none_and_pins_nothing():
    """NEGATIVE CONTROL: the unpinned layout is not touched by this feature."""
    layout = _layout(())
    assert layout["audio_row_permutation"] is None
    assert layout["audio_row_order"] is None
    assert layout["num_condition_audio_rows"] == 0


def test_pinned_prefix_is_the_channel_major_rows_of_the_pinned_latents():
    pinned_latents = _middle_third_pinned()
    layout = _layout(pinned_latents)
    n_cond = len(pinned_latents) * ops.AUDIO_CHANNELS
    assert layout["num_condition_audio_rows"] == n_cond
    perm = layout["audio_row_permutation"]
    assert perm is not None and perm.numel() == 2 * NUM_AUDIO_LATENTS
    expected_prefix = torch.tensor(ops.audio_pin_row_indices(pinned_latents, NUM_AUDIO_LATENTS))
    assert torch.equal(perm[:n_cond], expected_prefix)


def test_the_permutation_is_a_bijection_of_the_row_block():
    pinned_latents = _middle_third_pinned()
    layout = _layout(pinned_latents)
    perm = layout["audio_row_permutation"]
    assert torch.equal(torch.sort(perm).values, torch.arange(2 * NUM_AUDIO_LATENTS))
    assert torch.equal(torch.argsort(perm), layout["audio_row_order"])


def test_pinned_rows_land_at_the_conditioning_timestep_and_free_rows_stay_on_schedule():
    pinned_latents = _middle_third_pinned()
    layout = _layout(pinned_latents)
    uniq, index = ops.build_row_timesteps(layout, video_timestep=0.1, audio_timestep=0.1)
    row_time = uniq[index]
    n_cond = layout["num_condition_audio_rows"]
    pinned_rows = layout["audio_indices"][:n_cond]
    free_rows = layout["audio_indices"][n_cond:]
    assert bool((row_time[pinned_rows] == ops.AUDIO_COND_TIMESTEP).all())
    assert bool((row_time[free_rows] == 0.1).all())
    # NEGATIVE CONTROL: the free rows are NOT also pinned.
    assert free_rows.numel() > 0


def test_no_video_row_is_pinned_by_the_audio_permutation():
    pinned_latents = _middle_third_pinned()
    layout = _layout(pinned_latents)
    uniq, index = ops.build_row_timesteps(layout, video_timestep=0.1, audio_timestep=0.1)
    row_time = uniq[index]
    assert bool((row_time[layout["video_indices"]] == 0.1).all())
    assert layout["num_condition_video_rows"] == 0


def test_permute_then_unpermute_is_the_identity_on_a_synthetic_channel_major_block():
    """The draw-time substitution's permute and the decode-time un-permute cancel."""
    pinned_latents = _middle_third_pinned()
    layout = _layout(pinned_latents)
    perm, inverse = layout["audio_row_permutation"], layout["audio_row_order"]
    rows = torch.arange(2 * NUM_AUDIO_LATENTS * 32, dtype=torch.float32).reshape(
        2 * NUM_AUDIO_LATENTS, 32)
    packed = rows[perm]
    restored = packed[inverse]
    assert torch.equal(restored, rows)
    # And the packed prefix really does carry the pinned rows, in the ORDER the
    # permutation names them.
    assert torch.equal(packed[:len(pinned_latents) * ops.AUDIO_CHANNELS],
                       rows[torch.tensor(ops.audio_pin_row_indices(pinned_latents, NUM_AUDIO_LATENTS))])


# --------------------------------------------------------------------------
# substitute_and_permute_audio_rows: THE HELPER the draw-time call site uses.
#
# WHY THESE EXIST: `test_permute_then_unpermute_is_the_identity_on_a_synthetic_
# channel_major_block` above reads `audio_row_permutation` / `audio_row_order`
# straight off the layout and never touches production code, and
# `minimax_h3_audio_pin_test.py`'s backend-wiring tests (further down) stub
# `_generate_minimax_h3` wholesale -- so a bug in the actual substitute-then-
# permute step, or a swap of WHICH layout key each of the two call sites uses,
# is invisible to either. These tests drive the real helper
# `_generate_minimax_h3`'s draw site calls, with `free_rows` and `source_rows`
# given DISJOINT value ranges so "packed the source" and "packed the free draw"
# cannot be confused by a shape/dtype-only check.
# --------------------------------------------------------------------------

def _free_and_source_rows():
    free_rows = torch.arange(2 * NUM_AUDIO_LATENTS * 4, dtype=torch.float32).reshape(
        2 * NUM_AUDIO_LATENTS, 4)
    source_rows = free_rows + 1000.0
    return free_rows, source_rows


def test_substitute_and_permute_packs_the_pinned_source_into_the_prefix():
    """The PREFIX of the packed block is the pinned SOURCE rows, in the order
    the permutation names them -- not the free draw that occupied those rows
    before the substitution."""
    pinned_latents = _middle_third_pinned()
    layout = _layout(pinned_latents)
    free_rows, source_rows = _free_and_source_rows()

    packed = ops.substitute_and_permute_audio_rows(
        free_rows.clone(), source_rows, pinned_latents, NUM_AUDIO_LATENTS,
        layout["audio_row_permutation"])

    n_cond = len(pinned_latents) * ops.AUDIO_CHANNELS
    pin_indices = torch.tensor(ops.audio_pin_row_indices(pinned_latents, NUM_AUDIO_LATENTS))
    assert torch.equal(packed[:n_cond], source_rows[pin_indices])
    assert bool((packed[:n_cond] >= 1000.0).all())


def test_substitute_and_permute_leaves_the_free_rows_as_the_original_draw():
    """NEGATIVE CONTROL: the rows AFTER the prefix are still the free draw."""
    pinned_latents = _middle_third_pinned()
    layout = _layout(pinned_latents)
    free_rows, source_rows = _free_and_source_rows()

    packed = ops.substitute_and_permute_audio_rows(
        free_rows.clone(), source_rows, pinned_latents, NUM_AUDIO_LATENTS,
        layout["audio_row_permutation"])

    n_cond = len(pinned_latents) * ops.AUDIO_CHANNELS
    assert bool((packed[n_cond:] < 1000.0).all())


def test_the_decode_unpermute_restores_channel_major_order_of_the_substituted_block():
    """The FULL round trip a real request goes through: pin + pack at the draw
    site, then un-permute with `audio_row_order` at the decode site -- and the
    result must be the ORIGINAL channel-major block with only the pinned rows
    replaced by the source, not merely SOME permutation's round trip.

    THE MUTANT THIS EXISTS FOR: swapping which layout key is used at the draw
    site (`audio_row_permutation`) and the decode site (`audio_row_order`) for
    one another. Both are shape/dtype-identical bijections of the same row
    block, so a swap raises nothing; it silently reorders every generated
    audio row in the decoded output. This test drives the exact helper the
    draw site calls, un-permutes with the exact key the decode site reads, and
    checks CONTENT equality against the expected channel-major array -- a test
    that only checks `perm(inverse(x)) == x` cannot distinguish "the two keys
    were used correctly" from "the two keys were swapped and still compose to
    the identity as a pair", which is exactly the class of bug this closes.
    """
    pinned_latents = _middle_third_pinned()
    layout = _layout(pinned_latents)
    permutation, order = layout["audio_row_permutation"], layout["audio_row_order"]
    free_rows, source_rows = _free_and_source_rows()
    pin_indices = torch.tensor(ops.audio_pin_row_indices(pinned_latents, NUM_AUDIO_LATENTS))
    expected = free_rows.clone()
    expected[pin_indices] = source_rows[pin_indices]

    packed = ops.substitute_and_permute_audio_rows(
        free_rows.clone(), source_rows, pinned_latents, NUM_AUDIO_LATENTS, permutation)
    restored = packed[order]
    assert torch.equal(restored, expected)

    # NEGATIVE CONTROL for the swap itself: un-permuting with the DRAW-time
    # permutation instead of the decode-time order does NOT restore the
    # channel-major block for this (non-involutory) pinned latent set.
    wrongly_restored = packed[permutation]
    assert not torch.equal(wrongly_restored, expected)


def test_the_backend_draw_and_decode_sites_use_the_correct_layout_keys():
    """Anchors WHICH layout key each of the two production call sites reads.

    Companion to the content tests above: those catch the bug numerically,
    against synthetic data; this one pins the shipped source text directly,
    so a future edit that moves the calls around still has to keep the right
    key at each site or fail here immediately, without needing a GPU or a
    loaded model to exercise the real denoise loop.
    """
    from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin

    source = inspect.getsource(MiniMaxH3Mixin._generate_minimax_h3)

    draw_call = source[source.index("ops.substitute_and_permute_audio_rows("):]
    draw_call = draw_call[:draw_call.index(")\n") + 1]
    assert 'layout["audio_row_permutation"]' in draw_call
    assert 'layout["audio_row_order"]' not in draw_call

    decode_unpermute = source[source.index('audio_row_order = layout["audio_row_order"]'):]
    decode_unpermute = decode_unpermute[:decode_unpermute.index("full_audio_rows = ") + 200]
    assert 'layout["audio_row_order"]' in decode_unpermute
    assert 'layout["audio_row_permutation"]' not in decode_unpermute


def test_unpack_audio_rows_on_the_restored_block_is_channel_correct():
    rows = torch.arange(2 * NUM_AUDIO_LATENTS * 32, dtype=torch.float32).reshape(
        2 * NUM_AUDIO_LATENTS, 32)
    unpacked = ops.unpack_audio_rows(rows, NUM_AUDIO_LATENTS)
    assert tuple(unpacked.shape) == (2, 32, NUM_AUDIO_LATENTS)
    assert torch.equal(unpacked[1, :, 3], rows[NUM_AUDIO_LATENTS + 3])
    assert torch.equal(unpacked[0, :, 0], rows[0])


# --------------------------------------------------------------------------
# pin_target_audio's whole-track case is now the P=all-latents degenerate case
# --------------------------------------------------------------------------

def test_pin_target_audio_generalises_to_an_identity_permutation():
    """The whole-track shorthand is bitwise unchanged: the identity permutation
    of an already-ascending `audio_indices` reorders nothing."""
    layout = _layout((), pin_target_audio=True)
    perm = layout["audio_row_permutation"]
    assert perm is not None
    assert torch.equal(perm, torch.arange(2 * NUM_AUDIO_LATENTS))
    assert torch.equal(layout["audio_row_order"], torch.arange(2 * NUM_AUDIO_LATENTS))
    assert layout["num_condition_audio_rows"] == 2 * NUM_AUDIO_LATENTS
    # ... and matches the pre-partial-pin values bitwise (no `pinned_audio_latents`).
    free_layout = _layout(())
    assert torch.equal(layout["audio_indices"], free_layout["audio_indices"])
    for key in ("position_ids", "token_tags", "video_indices", "text_indices"):
        assert torch.equal(layout[key], free_layout[key]), key


def test_pin_target_audio_and_pinned_audio_latents_are_mutually_exclusive():
    with pytest.raises(ValueError, match="pin_target_audio"):
        _layout((0,), pin_target_audio=True)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def test_pinned_audio_latents_out_of_range_is_refused():
    with pytest.raises(ValueError, match="outside this clip"):
        _layout((NUM_AUDIO_LATENTS,))


def test_pinned_audio_latents_duplicates_are_refused():
    with pytest.raises(ValueError, match="distinct"):
        _layout((0, 0))


def test_pinned_audio_latents_non_integer_is_refused():
    for bad in (1.5, "0", True, None):
        with pytest.raises(ValueError, match="integer index"):
            _layout((bad,))


def test_audio_pin_row_indices_is_channel_major():
    """Row `channel * T + latent`, in the order `latents` was given."""
    assert ops.audio_pin_row_indices((2, 5), 10) == (2, 5, 12, 15)
    assert ops.audio_pin_row_indices((), 10) == ()


# --------------------------------------------------------------------------
# ref2va's layout carries the same (always-None) keys, for a uniform dict shape
# --------------------------------------------------------------------------

def test_ref2va_layout_carries_the_same_none_keys():
    layout = ops.build_ref2va_packed_layout(
        [ops.TEXT_TAG] * 10, [("video", True)], [(7, 24, 40)], [74], 7, 24, 40, 37)
    assert layout["audio_row_permutation"] is None
    assert layout["audio_row_order"] is None


# --------------------------------------------------------------------------
# plan_audio_pin_latents: the route/backend-shared snap arithmetic
# --------------------------------------------------------------------------

def test_free_and_pinned_are_exact_complements():
    free, pinned = plan_audio_pin_latents(40, 85, 207, fps=24.0, latents_per_second=40.0)
    assert set(free) | set(pinned) == set(range(207))
    assert not (set(free) & set(pinned))
    assert free == tuple(sorted(free))
    assert pinned == tuple(sorted(pinned))


def test_free_span_uses_floor_low_and_ceil_high():
    lo = int(math.floor(40 * 40.0 / 24.0))
    hi = int(math.ceil(85 * 40.0 / 24.0))
    free, pinned = plan_audio_pin_latents(40, 85, 207, fps=24.0, latents_per_second=40.0)
    assert free == tuple(range(lo, hi))
    assert pinned == tuple(t for t in range(207) if not lo <= t < hi)


def test_a_whole_clip_range_pins_nothing():
    """The degenerate case `_generate_vidinpaint_minimax_h3` falls back on."""
    free, pinned = plan_audio_pin_latents(0, 207 * 24 // 40 + 1, 207, fps=24.0, latents_per_second=40.0)
    assert pinned == ()
    assert free == tuple(range(207))


def test_an_empty_audio_grid_pins_and_frees_nothing():
    assert plan_audio_pin_latents(0, 10, 0, fps=24.0, latents_per_second=40.0) == ((), ())


def test_the_free_span_is_clamped_to_the_audio_grid():
    """A request range past the clip's own end must not overrun the grid."""
    free, pinned = plan_audio_pin_latents(0, 10_000, 207, fps=24.0, latents_per_second=40.0)
    assert free == tuple(range(207))
    assert pinned == ()
    free, pinned = plan_audio_pin_latents(-5, 10, 207, fps=24.0, latents_per_second=40.0)
    assert free[0] == 0


# --------------------------------------------------------------------------
# Backend wiring: `_generate_vidinpaint_minimax_h3`'s `regenerate_range` mode
# --------------------------------------------------------------------------

GENERATED_VALUE = 7
SOURCE_VALUE = 200
CLIP = 124
LATENTS = 37


def _plan(start, end, clip_frames=CLIP, arch="minimax_h3"):
    from api.generation_utils import plan_video_inpaint_span
    return plan_video_inpaint_span(
        {"regenerate_start_frame": start, "regenerate_end_frame": end},
        arch, clip_frames=clip_frames)


def _source_clip(width=64, height=32, frames=CLIP):
    clip = np.full((frames, height, width, 3), SOURCE_VALUE, dtype=np.uint8)
    for index in range(frames):
        clip[index, 0, 0, 0] = index % 251
    return clip


def _audio_runner(width=64, height=32, *, pinned_audio_return=None):
    """Mirrors `minimax_h3_temporal_inpaint_route_test.py`'s `_runner`, extended
    to also stub `_minimax_h3_inpaint_pinned_audio` -- which shells out to
    ffmpeg -- so the wiring around it can be tested without real audio bytes."""
    from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin

    captured = {}

    class Runner(MiniMaxH3Mixin):
        minimax_h3_components = {
            "variant": "fl2va", "audio_sample_rate": 32000,
            "fps": 24.0, "audio_latent_rate": 40.0,
        }
        current_model_info = {"type": "minimax_h3", "variant": "fl2va"}

        def _generate_minimax_h3(self, params, **kwargs):
            captured.update(kwargs)
            captured["params"] = params
            frames = np.full((int(params["num_frames"]), height, width, 3),
                             GENERATED_VALUE, dtype=np.uint8)
            return frames, None, None, 4242

        def _minimax_h3_inpaint_pinned_audio(self, *args, **kwargs):
            captured["pinned_audio_prepared"] = True
            return pinned_audio_return

    return Runner(), captured


def test_regenerate_range_pins_the_preserved_spans_as_conditioning():
    """THE PIN. `regenerate_range` now conditions on the input track outside
    the regenerate range, instead of generating audio blind."""
    runner, captured = _audio_runner(pinned_audio_return=torch.zeros(2, 10))
    clip = _source_clip()
    params = {"width": 64, "height": 32, "frame_rate": 24.0,
              "regenerate_start_frame": 40, "regenerate_end_frame": 85,
              "inpaint_video_audio_mode": "regenerate_range"}

    runner._generate_vidinpaint_minimax_h3(params, clip, 24.0, b"fake-wav-bytes")

    assert captured.get("pinned_audio_prepared") is True
    assert captured["input_audio"] is not None
    assert len(captured["pinned_audio_latents"]) > 0

    plan = _plan(40, 85)
    num_audio_latents = ops.audio_latent_frames(CLIP, fps=24.0, latents_per_second=40.0)
    _free, expected_pinned = plan_audio_pin_latents(
        plan["start_frame"], plan["end_frame"], num_audio_latents,
        fps=24.0, latents_per_second=40.0)
    assert tuple(sorted(captured["pinned_audio_latents"])) == expected_pinned


def test_regenerate_range_falls_back_to_regenerate_when_nothing_is_left_to_pin(monkeypatch):
    """The degenerate case: the snapped range covers the whole audio grid."""
    import api.generation_utils as gu

    monkeypatch.setattr(gu, "plan_audio_pin_latents", lambda *a, **kw: ((), ()))
    runner, captured = _audio_runner(pinned_audio_return=torch.zeros(2, 10))
    clip = _source_clip()
    params = {"width": 64, "height": 32, "frame_rate": 24.0,
              "regenerate_start_frame": 40, "regenerate_end_frame": 85,
              "inpaint_video_audio_mode": "regenerate_range"}

    runner._generate_vidinpaint_minimax_h3(params, clip, 24.0, b"fake-wav-bytes")

    assert captured.get("pinned_audio_prepared") is None, \
        "the whole-grid degenerate case must not even try to prepare a pin"
    assert captured["input_audio"] is None
    assert captured["pinned_audio_latents"] == ()


def test_regenerate_range_falls_back_when_the_window_extraction_fails():
    """`_minimax_h3_inpaint_pinned_audio` returning None (already warned) must
    leave the range generating unconditioned, exactly the pre-partial-pin
    `regenerate_range` behaviour -- the output-level splice still runs."""
    runner, captured = _audio_runner(pinned_audio_return=None)
    clip = _source_clip()
    params = {"width": 64, "height": 32, "frame_rate": 24.0,
              "regenerate_start_frame": 40, "regenerate_end_frame": 85,
              "inpaint_video_audio_mode": "regenerate_range"}

    runner._generate_vidinpaint_minimax_h3(params, clip, 24.0, b"fake-wav-bytes")

    assert captured.get("pinned_audio_prepared") is True
    assert captured["input_audio"] is None
    assert captured["pinned_audio_latents"] == ()


def test_preserve_input_remains_a_whole_track_pin():
    """NEGATIVE CONTROL: `preserve_input` did not change -- still every latent."""
    runner, captured = _audio_runner(pinned_audio_return=torch.zeros(2, 10))
    clip = _source_clip()
    params = {"width": 64, "height": 32, "frame_rate": 24.0,
              "regenerate_start_frame": 40, "regenerate_end_frame": 85,
              "inpaint_video_audio_mode": "preserve_input"}

    runner._generate_vidinpaint_minimax_h3(params, clip, 24.0, b"fake-wav-bytes")

    assert captured.get("pinned_audio_prepared") is True
    assert captured["input_audio"] is not None
    assert captured["pinned_audio_latents"] == ()


def test_plain_regenerate_pins_no_audio_at_all():
    """NEGATIVE CONTROL: the default mode is untouched by this feature."""
    runner, captured = _audio_runner()
    clip = _source_clip()
    params = {"width": 64, "height": 32, "frame_rate": 24.0,
              "regenerate_start_frame": 40, "regenerate_end_frame": 85,
              "inpaint_video_audio_mode": "regenerate"}

    runner._generate_vidinpaint_minimax_h3(params, clip, 24.0, None)

    assert captured.get("pinned_audio_prepared") is None
    assert captured["input_audio"] is None
    assert captured["pinned_audio_latents"] == ()
