"""ref2va references x temporal-inpaint pins: the extended builder and the
decision table, OPENED (phase B-3-open).

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_inpaint_reference_layout_test.py -v

WHY THIS FILE EXISTS
--------------------
`minimax_h3_inpaint_refs_design.md` (Option B, Gate registration (B)) fixed a
layout extension and a decision table BEFORE any GPU probe ran (phase B-1),
then wired the whole path end to end while keeping the gate shut (phase
B-2a). Phase B-3-open flips the ONE switch the gate function names: the
`ref2va` row now serves every request, opened at the repo owner's explicit
instruction so the endpoint is reachable for hands-on verification through
the real UI, NOT because the design's §6.2 GPU arms (P/C/P-seam) were run --
they were not, and every generation on this path still carries a
`minimax_h3_undocumented_conditioning` warning saying so. Both halves below
remain pure arithmetic (no model, no GPU), pinned here exactly the way
`minimax_h3_outpaint_reference_gate_test.py` pins the outpaint equivalents:

* `resolve_minimax_h3_inpaint_reference_gate` (api.generation_utils) is the
  refusal table -- `ref2va` now returns `None` (allow) unconditionally, the
  same shape as `fl2va`'s no-references row. The interior pin this endpoint
  needs is STILL measured on fl2va only (`minimax_h3_ti_probe_results.md`:
  pinned 3.12 RMS, floor 3.15, control 75.69) and STILL unmeasured on ref2va
  -- opening the gate did not change that fact, only whether the path is
  reachable.
* `build_ref2va_packed_layout`'s `pinned_video_frames` / `pinned_audio_latents`
  parameters (core.models.minimax_h3.h3_pipeline_ops) extend the reference
  builder with the same permutation mechanism `build_packed_layout` already
  uses for fl2va's temporal inpaint, restricted to the TARGET block --
  reference (and anchor) rows already lead both index lists unconditionally,
  so the pin only ever reorders rows at or past `video_start`/`audio_start`.
  `position_ids` is untouched: the permutation only changes which physical
  rows the index lists call "conditioning", never the rotary time any
  physical row carries. This builder-level arithmetic was already reachable
  in phase B-1/B-2a directly against the builder; opening the gate makes it
  reachable through the route/backend as well, which is what this file's
  gate-table tests now assert.

Sequence length / row budget is DELIBERATELY absent from both the layout
tests and the refusal table (owner correction, recorded in the design doc's
Gate registration (B) section): it is not a threshold anything refuses on.
"""

import os
import sys

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from api.error_handlers import ValidationError  # noqa: E402
from api.generation_utils import (  # noqa: E402
    minimax_h3_inpaint_reference_row_count_warning,
    resolve_minimax_h3_inpaint_reference_gate,
)
from core.models.minimax_h3 import h3_pipeline_ops as ops  # noqa: E402


# ---------------------------------------------------------------------------
# The decision table (minimax_h3_inpaint_refs_design.md, Gate registration (B))
# ---------------------------------------------------------------------------

def _gate(variant, *, has_images=False, has_videos=False, has_audios=False,
          has_vision_conditioning=False):
    return resolve_minimax_h3_inpaint_reference_gate(
        variant, has_reference_images=has_images, has_reference_videos=has_videos,
        has_reference_audios=has_audios, has_vision_conditioning=has_vision_conditioning)


def test_fl2va_no_refs_is_allowed():
    assert _gate("fl2va") is None


def test_fl2va_with_any_reference_kind_is_refused():
    with pytest.raises(ValidationError, match="fl2va"):
        _gate("fl2va", has_images=True)
    with pytest.raises(ValidationError, match="fl2va"):
        _gate("fl2va", has_videos=True)
    with pytest.raises(ValidationError, match="fl2va"):
        _gate("fl2va", has_audios=True, has_images=True)  # paired, still refused on fl2va


def test_ref2va_is_allowed_regardless_of_references():
    """Phase B-3-open: ref2va serves every request, with or without
    references -- the interior pin is still unmeasured on these weights, but
    that is now stated by a `warnings[]` entry at generation time, not by a
    refusal here.
    """
    assert _gate("ref2va") is None
    assert _gate("ref2va", has_images=True) is None
    assert _gate("ref2va", has_videos=True, has_audios=True) is None


def test_hybrid_is_refused_regardless_of_references():
    with pytest.raises(ValidationError, match="hybrid"):
        _gate("hybrid")
    with pytest.raises(ValidationError, match="hybrid"):
        _gate("hybrid", has_images=True)


def test_unidentified_variant_refuses_references_but_allows_a_plain_request():
    assert _gate("") is None
    assert _gate(None) is None
    with pytest.raises(ValidationError):
        _gate("", has_images=True)
    with pytest.raises(ValidationError):
        _gate("some-other-checkpoint", has_videos=True)


def test_audio_only_reference_set_is_refused_with_the_pairing_rule():
    """The model's own limit (h3_references.py:148-152), checked before the
    variant/partition rows -- and independent of them, so it fires even on a
    variant that would otherwise allow a plain request.
    """
    with pytest.raises(ValidationError, match="on its own"):
        _gate("ref2va", has_audios=True)
    with pytest.raises(ValidationError, match="on its own"):
        _gate("", has_audios=True)
    with pytest.raises(ValidationError, match="on its own"):
        _gate("fl2va", has_audios=True)


def test_audio_paired_with_a_vision_reference_passes_on_ref2va():
    """Phase B-3-open: a paired audio+image reference set is refused by
    NEITHER the pairing rule NOR the (now-open) ref2va partition row -- the
    gate returns `None`. Before B-3-open this was refused by the partition
    row alone (the pairing rule never fires when a vision reference is
    paired); now nothing refuses it.
    """
    assert _gate("ref2va", has_audios=True, has_images=True) is None


def test_audio_only_with_vision_conditioning_from_a_pin_passes_on_ref2va():
    """SECOND owner correction: `has_vision_conditioning=True` (a temporal-
    inpaint pin or a keyframe anchor already supplying real vision
    conditioning) satisfies the pairing rule's own premise the same way a
    paired image/video reference would -- so an audio-only reference set is
    NOT refused by the pairing check here. Phase B-3-open: it is also not
    refused by the ref2va partition row anymore, so the gate returns `None`
    outright -- this endpoint always passes `has_vision_conditioning=True`
    (it always pins the frames outside the regenerate range), which is
    exactly the shape this test names.
    """
    assert _gate("ref2va", has_audios=True, has_vision_conditioning=True) is None


def test_audio_only_without_vision_conditioning_still_trips_the_pairing_rule():
    """NEGATIVE CONTROL for the above: without `has_vision_conditioning`, an
    audio-only reference set is refused by the pairing rule exactly as
    before -- the default is `False`, so a caller that does not know about
    pins/anchors gets today's behaviour verbatim.
    """
    with pytest.raises(ValidationError, match="on its own"):
        _gate("ref2va", has_audios=True, has_vision_conditioning=False)


def test_validate_references_audio_alone_rule_respects_the_new_flag():
    """`h3_references.validate_references` itself (NOT `/generate/ref2vid`,
    which has its own separate inline copy of this rule, `routes.py:4382-
    4389`, and is not exercised by this test): the default
    (`has_vision_conditioning=False`) reproduces the pre-extension refusal
    verbatim; passing `True` is what an inpaint/keyframe caller opts into,
    and only that caller's audio-only set is allowed through.
    """
    from core.models.minimax_h3.h3_references import MiniMaxH3Reference, validate_references

    audio_only = [MiniMaxH3Reference(kind="audio", sample_rate=32000)]
    with pytest.raises(ValueError, match="on its own"):
        validate_references(audio_only)
    with pytest.raises(ValueError, match="on its own"):
        validate_references(audio_only, has_vision_conditioning=False)
    validate_references(audio_only, has_vision_conditioning=True)  # does not raise


def test_the_gate_never_names_ref2vid_as_a_destination():
    """Mirrors `resolve_minimax_h3_outpaint_reference_gate`'s own test of the
    same name: the invariant is on the TOP-LEVEL message (what a caller acts
    on), not `detail` -- outpaint's own fl2va-with-refs message cites
    `/generate/ref2vid` in its `detail` for context ("mirror of
    /generate/ref2vid's own partition gate"), and that citation is
    informative, not a reroute.
    """
    refusals = []
    for kwargs in (
        {"has_images": True},                       # fl2va, hybrid refuse; ref2va allows: 2
        {"has_videos": True, "has_audios": True},    # paired audio: fl2va, hybrid refuse; ref2va allows: 2
        {},                                          # plain request: only hybrid refuses: 1
    ):
        for variant in ("fl2va", "ref2va", "hybrid"):
            try:
                _gate(variant, **kwargs)
            except ValidationError as exc:
                refusals.append(str(exc))
    assert len(refusals) == 5
    assert not any("ref2vid" in message.lower() for message in refusals)


# ---------------------------------------------------------------------------
# The extended layout builder (minimax_h3_inpaint_refs_design.md, §1/Option B)
# ---------------------------------------------------------------------------

# A small, readable geometry shared by the pin-mechanism tests: one image
# reference (1 latent frame, same canvas as the target), a 6-latent-frame
# target, patch (1, 2, 2) -> rows_per_frame = (4//2)*(4//2) = 4.
_NUM_TEXT_TOKENS = 5
_LAT_H, _LAT_W = 4, 4
_TARGET_FRAMES = 6
_NUM_AUDIO_LATENTS = 3
_REFERENCE_BLOCKS = [("image", False)]
_CONDITION_SHAPES = [(1, _LAT_H, _LAT_W)]


def _build(pinned_video_frames=(), pinned_audio_latents=()):
    return ops.build_ref2va_packed_layout(
        text_token_tags=[1] * _NUM_TEXT_TOKENS,
        reference_blocks=_REFERENCE_BLOCKS,
        condition_latent_shapes=_CONDITION_SHAPES,
        reference_audio_row_counts=[],
        num_latent_frames=_TARGET_FRAMES,
        latent_height=_LAT_H, latent_width=_LAT_W,
        num_audio_latents=_NUM_AUDIO_LATENTS,
        pinned_video_frames=pinned_video_frames,
        pinned_audio_latents=pinned_audio_latents,
    )


def test_standalone_audio_reference_plus_a_video_pin_lays_out_correctly():
    """The layout shape SECOND correction's `has_vision_conditioning=True`
    makes legal: a standalone AUDIO reference (no image/video reference at
    all) alongside a temporal-inpaint pin. The builder itself never enforces
    the pairing rule (that is `validate_references`'s job, at the
    orchestration layer, tested above) -- this test only pins the ARITHMETIC
    of the resulting layout: the audio reference's rows lead `audio_indices`
    (the audio prefix), the pinned frames' rows lead `video_indices` (the
    video prefix), and the two counts are independent, exactly as the
    ordinary reference-video case already is.
    """
    audio_ref_latents = 2
    audio_ref_rows = audio_ref_latents * ops.AUDIO_CHANNELS  # channel-major: 4 rows
    layout = ops.build_ref2va_packed_layout(
        text_token_tags=[1] * _NUM_TEXT_TOKENS,
        reference_blocks=[("audio", True)],
        condition_latent_shapes=[],
        reference_audio_row_counts=[audio_ref_rows],
        num_latent_frames=_TARGET_FRAMES,
        latent_height=_LAT_H, latent_width=_LAT_W,
        num_audio_latents=_NUM_AUDIO_LATENTS,
        pinned_video_frames=(0, 2),
    )
    rows_per_frame = layout["rows_per_frame"]

    # Video prefix: no visual reference block exists (audio-only), so the
    # ENTIRE video conditioning prefix is the pin -- 2 pinned frames.
    assert int(layout["num_condition_video_rows"]) == 2 * rows_per_frame
    video_indices = layout["video_indices"]
    expected_lead_frames = torch.cat([
        torch.arange(0 * rows_per_frame, 1 * rows_per_frame),
        torch.arange(2 * rows_per_frame, 3 * rows_per_frame),
    ])
    # The leading video rows are the pin's own physical rows, offset to where
    # the target video block starts (there is no reference video row ahead of
    # it in THIS shape, so video_start == the physical row right after the
    # audio reference + target audio rows).
    video_start = int(video_indices[: rows_per_frame].min())
    assert torch.equal(video_indices[: 2 * rows_per_frame], video_start + expected_lead_frames)

    # Audio prefix: the standalone reference's own 4 rows, unconditionally
    # ahead of every target audio row (no audio pin requested here).
    assert int(layout["num_condition_audio_rows"]) == audio_ref_rows
    audio_indices = layout["audio_indices"]
    assert torch.equal(audio_indices[:audio_ref_rows], torch.arange(_NUM_TEXT_TOKENS, _NUM_TEXT_TOKENS + audio_ref_rows))

    # The two prefixes are independent counts, per the per-modality COUNT
    # contract `build_row_timesteps` relies on.
    assert int(layout["num_condition_video_rows"]) != int(layout["num_condition_audio_rows"])


def test_reference_audio_block_plus_an_audio_pin_sums_both_counts():
    """The combination the owner's ask actually depends on: a reference AUDIO
    block (the steering track) coexisting with an audio PIN on the target's
    own preserved span (`regenerate_range`'s partial audio pin). Nothing else
    in this file exercises both at once.

    `num_condition_audio_rows` = the reference's own rows + the pinned rows,
    and `audio_indices` orders them [reference rows | pinned target rows |
    free target rows] -- the reference block, built first in the loop, always
    leads; the pin only reorders the TARGET portion behind it.
    """
    audio_ref_latents = 2
    audio_ref_rows = audio_ref_latents * ops.AUDIO_CHANNELS  # 4 rows
    layout = ops.build_ref2va_packed_layout(
        text_token_tags=[1] * _NUM_TEXT_TOKENS,
        reference_blocks=[("audio", True)],
        condition_latent_shapes=[],
        reference_audio_row_counts=[audio_ref_rows],
        num_latent_frames=_TARGET_FRAMES,
        latent_height=_LAT_H, latent_width=_LAT_W,
        num_audio_latents=_NUM_AUDIO_LATENTS,
        pinned_audio_latents=(1,),  # 2 rows (both channels of latent 1)
    )
    pinned_rows = ops.audio_pin_row_indices((1,), _NUM_AUDIO_LATENTS)  # channel-major

    assert int(layout["num_condition_audio_rows"]) == audio_ref_rows + len(pinned_rows)

    audio_indices = layout["audio_indices"]
    # Reference rows lead, unconditionally.
    assert torch.equal(audio_indices[:audio_ref_rows],
                        torch.arange(_NUM_TEXT_TOKENS, _NUM_TEXT_TOKENS + audio_ref_rows))
    # The pinned target rows follow immediately, before any free target row.
    target_prefix = audio_indices[audio_ref_rows: audio_ref_rows + len(pinned_rows)]
    # The target block's permutation spans every original row index once
    # (0..num_target_audio_rows-1), so its minimum -- taken over the WHOLE
    # permuted target slice, not just the pinned prefix -- recovers the
    # target block's own physical start.
    target_start = int(audio_indices[audio_ref_rows:].min())
    assert torch.equal(target_prefix, target_start + torch.tensor(pinned_rows, dtype=torch.long))


def test_build_row_timesteps_pins_reference_and_pinned_rows_but_not_free_ones():
    """The CONTRACT (`build_row_timesteps`), not the arithmetic restated: on a
    pinned ref2va layout, physical rows named by the reference+pin prefix sit
    at the conditioning timestep for their modality; every other row sits at
    the request's own (generated) timestep.
    """
    layout = _build(pinned_video_frames=(0, 2), pinned_audio_latents=(1,))
    video_timestep, audio_timestep = 0.4, 0.7
    unique_timesteps, inverse = ops.build_row_timesteps(layout, video_timestep, audio_timestep)
    row_timesteps = unique_timesteps[inverse]

    n_cond_video = int(layout["num_condition_video_rows"])
    n_cond_audio = int(layout["num_condition_audio_rows"])
    video_indices = layout["video_indices"]
    audio_indices = layout["audio_indices"]

    cond_video_rows = video_indices[:n_cond_video]
    free_video_rows = video_indices[n_cond_video:]
    cond_audio_rows = audio_indices[:n_cond_audio]
    free_audio_rows = audio_indices[n_cond_audio:]

    assert torch.all(row_timesteps[cond_video_rows] == max(video_timestep, ops.VISUAL_COND_TIMESTEP))
    assert torch.all(row_timesteps[free_video_rows] == video_timestep)
    assert torch.all(row_timesteps[cond_audio_rows] == ops.AUDIO_COND_TIMESTEP)
    assert torch.all(row_timesteps[free_audio_rows] == audio_timestep)
    # The reference block's own rows are a strict subset of the conditioning
    # prefix (the pin's rows are the rest of it) -- both pinned at the same
    # timestep, since `build_row_timesteps` makes no distinction between them.
    assert row_timesteps[_NUM_TEXT_TOKENS] == max(video_timestep, ops.VISUAL_COND_TIMESTEP)


def test_conditioning_rows_lead_both_index_lists():
    """video_indices = [ref video rows | pinned target rows | free target
    rows]; audio_indices likewise (no reference audio rows in this geometry,
    so just [pinned target audio | free target audio]).
    """
    layout = _build(pinned_video_frames=(0, 2), pinned_audio_latents=(1,))
    rows_per_frame = layout["rows_per_frame"]
    n_cond_video = int(layout["num_condition_video_rows"])
    n_cond_audio = int(layout["num_condition_audio_rows"])

    # The leading `rows_per_frame` rows of video_indices are the image
    # reference's own rows (physical rows [text_tokens, text_tokens+4)).
    video_indices = layout["video_indices"]
    assert torch.equal(video_indices[:rows_per_frame],
                        torch.arange(_NUM_TEXT_TOKENS, _NUM_TEXT_TOKENS + rows_per_frame))

    # n_cond_video = 4 (image ref) + 2*4 (two pinned frames) = 12.
    assert n_cond_video == rows_per_frame + 2 * rows_per_frame
    # n_cond_audio = 2 (one pinned latent x 2 channels).
    assert n_cond_audio == 2

    # Nothing named as conditioning appears again in the free tail.
    conditioning_video_rows = set(video_indices[:n_cond_video].tolist())
    free_video_rows = set(video_indices[n_cond_video:].tolist())
    assert conditioning_video_rows.isdisjoint(free_video_rows)
    assert len(conditioning_video_rows) == n_cond_video

    audio_indices = layout["audio_indices"]
    conditioning_audio_rows = set(audio_indices[:n_cond_audio].tolist())
    free_audio_rows = set(audio_indices[n_cond_audio:].tolist())
    assert conditioning_audio_rows.isdisjoint(free_audio_rows)


def test_per_modality_counts_are_independent():
    """Pinning only video must not move the audio count, and vice versa."""
    video_only = _build(pinned_video_frames=(0,))
    assert int(video_only["num_condition_audio_rows"]) == 0
    assert int(video_only["num_condition_video_rows"]) == 4 + 1 * 4  # ref + 1 pinned frame

    audio_only = _build(pinned_audio_latents=(0, 2))
    assert int(audio_only["num_condition_video_rows"]) == 4  # ref rows only
    assert int(audio_only["num_condition_audio_rows"]) == 2 * 2  # 2 latents x 2 channels


def test_pinned_target_rows_position_ids_bitwise_equal_to_the_unpinned_build():
    """position_ids is a pure function of the physical row layout, which the
    pin never changes -- so a pinned build's position_ids tensor is bitwise
    IDENTICAL, in full, to the unpinned build's, not merely equal at the
    pinned rows.
    """
    unpinned = _build()
    pinned = _build(pinned_video_frames=(0, 3, 5), pinned_audio_latents=(1,))
    assert torch.equal(unpinned["position_ids"], pinned["position_ids"])
    assert torch.equal(unpinned["token_tags"], pinned["token_tags"])


def test_video_permutation_round_trip_offset_past_the_reference_rows():
    layout = _build(pinned_video_frames=(0, 2))
    rows_per_frame = layout["rows_per_frame"]
    permutation = layout["video_row_permutation"]
    order = layout["video_row_order"]

    # argsort round-trip: order undoes permutation.
    assert torch.equal(order[permutation], torch.arange(permutation.numel()))
    assert torch.equal(permutation[order], torch.arange(permutation.numel()))

    # The permutation is relative to the TARGET block only (0..num_video_rows),
    # never the reference block's own rows -- draw-time code (minimax_h3.py)
    # applies it to a target-only noise tensor.
    num_target_video_rows = _TARGET_FRAMES * rows_per_frame
    assert int(permutation.max()) == num_target_video_rows - 1
    assert int(permutation.min()) == 0

    # Frames 0 and 2 (pinned) lead the permutation, each contributing
    # `rows_per_frame` rows, in ascending frame order.
    expected_lead = torch.cat([
        torch.arange(0 * rows_per_frame, 1 * rows_per_frame),
        torch.arange(2 * rows_per_frame, 3 * rows_per_frame),
    ])
    assert torch.equal(permutation[: 2 * rows_per_frame], expected_lead)

    # In the FULL video_indices list, the permuted target block sits at its
    # offset PAST every reference-block row (the image reference's 4 rows).
    video_indices = layout["video_indices"]
    video_start = int(video_indices[rows_per_frame:].min())  # first target row's physical index
    assert torch.equal(
        video_indices[rows_per_frame:],
        video_start + permutation,
    )


def test_audio_permutation_round_trip():
    layout = _build(pinned_audio_latents=(1,))
    permutation = layout["audio_row_permutation"]
    order = layout["audio_row_order"]
    # Both directions, like the video sibling test.
    assert torch.equal(order[permutation], torch.arange(permutation.numel()))
    assert torch.equal(permutation[order], torch.arange(permutation.numel()))

    # Channel-major: latent 1 of a 3-latent, 2-channel grid is rows [1, 4]
    # (channel*num_audio_latents + latent), matching
    # `h3_pipeline_ops.audio_pin_row_indices`.
    assert permutation[0].item() == 1
    assert permutation[1].item() == 4

    audio_indices = layout["audio_indices"]
    audio_start = int(audio_indices.min())
    assert torch.equal(audio_indices, audio_start + permutation)


def test_no_pins_returns_none_permutations_like_before_this_parameter_existed():
    layout = _build()
    assert layout["video_row_permutation"] is None
    assert layout["video_row_order"] is None
    assert layout["audio_row_permutation"] is None
    assert layout["audio_row_order"] is None


def test_reference_only_shape_is_bit_identical_with_the_new_parameters_at_their_defaults():
    """Calling with the new parameters explicitly empty must reproduce calling
    without them at all -- the reference-only shape this builder already
    shipped is not touched by this parameter pair existing.
    """
    explicit_empty = ops.build_ref2va_packed_layout(
        text_token_tags=[1] * _NUM_TEXT_TOKENS,
        reference_blocks=_REFERENCE_BLOCKS,
        condition_latent_shapes=_CONDITION_SHAPES,
        reference_audio_row_counts=[],
        num_latent_frames=_TARGET_FRAMES,
        latent_height=_LAT_H, latent_width=_LAT_W,
        num_audio_latents=_NUM_AUDIO_LATENTS,
        pinned_video_frames=(),
        pinned_audio_latents=(),
    )
    no_kwargs_at_all = ops.build_ref2va_packed_layout(
        text_token_tags=[1] * _NUM_TEXT_TOKENS,
        reference_blocks=_REFERENCE_BLOCKS,
        condition_latent_shapes=_CONDITION_SHAPES,
        reference_audio_row_counts=[],
        num_latent_frames=_TARGET_FRAMES,
        latent_height=_LAT_H, latent_width=_LAT_W,
        num_audio_latents=_NUM_AUDIO_LATENTS,
    )
    for key in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices",
                "num_condition_video_rows", "num_condition_audio_rows", "rows_per_frame"):
        left, right = explicit_empty[key], no_kwargs_at_all[key]
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right), key
        else:
            assert left == right, key
    assert explicit_empty["video_row_permutation"] is None
    assert no_kwargs_at_all["video_row_permutation"] is None
    assert explicit_empty["audio_row_permutation"] is None
    assert no_kwargs_at_all["audio_row_permutation"] is None


def test_reference_and_anchor_shape_is_unaffected_by_the_new_parameters():
    """The pre-extension reference+anchor (C5) shape is bit-identical whether
    or not the new (empty) pin parameters are passed.
    """
    with_pin_kwargs = ops.build_ref2va_packed_layout(
        text_token_tags=[1] * _NUM_TEXT_TOKENS,
        reference_blocks=_REFERENCE_BLOCKS,
        condition_latent_shapes=_CONDITION_SHAPES,
        reference_audio_row_counts=[],
        num_latent_frames=_TARGET_FRAMES,
        latent_height=_LAT_H, latent_width=_LAT_W,
        num_audio_latents=_NUM_AUDIO_LATENTS,
        keyframe_anchors=("first",),
        pinned_video_frames=(),
        pinned_audio_latents=(),
    )
    without_pin_kwargs = ops.build_ref2va_packed_layout(
        text_token_tags=[1] * _NUM_TEXT_TOKENS,
        reference_blocks=_REFERENCE_BLOCKS,
        condition_latent_shapes=_CONDITION_SHAPES,
        reference_audio_row_counts=[],
        num_latent_frames=_TARGET_FRAMES,
        latent_height=_LAT_H, latent_width=_LAT_W,
        num_audio_latents=_NUM_AUDIO_LATENTS,
        keyframe_anchors=("first",),
    )
    for key in ("position_ids", "token_tags", "video_indices", "audio_indices",
                "num_condition_video_rows", "num_condition_audio_rows"):
        left, right = with_pin_kwargs[key], without_pin_kwargs[key]
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right), key
        else:
            assert left == right, key


def test_pinning_video_with_a_keyframe_anchor_present_is_refused():
    """Same mutual-exclusion rule `build_packed_layout` enforces: an anchor
    and a pin both reserve the SAME conditioning-prefix slot for the target
    block, so combining them is refused rather than silently choosing one.
    """
    with pytest.raises(ValueError, match="keyframe anchors"):
        _ = ops.build_ref2va_packed_layout(
            text_token_tags=[1] * _NUM_TEXT_TOKENS,
            reference_blocks=_REFERENCE_BLOCKS,
            condition_latent_shapes=_CONDITION_SHAPES,
            reference_audio_row_counts=[],
            num_latent_frames=_TARGET_FRAMES,
            latent_height=_LAT_H, latent_width=_LAT_W,
            num_audio_latents=_NUM_AUDIO_LATENTS,
            keyframe_anchors=("first",),
            pinned_video_frames=(0,),
        )


def test_an_image_reference_lands_outside_the_earliest_pinned_frames_binding_radius():
    """The layout-level regression transferred from
    `build_outpaint_references`'s own `test_an_image_reference_lands_outside_
    the_anchors_binding_radius`: an image reference must not land inside the
    binding radius of the earliest pinned frame's own time -- here read
    directly off `position_ids`, since the pin never moves it (unlike an
    anchor, which occupies its own rows, a pin's frame 0 IS the target's own
    frame 0, so its time is `_anchor_rotary_time("first", ...)`'s value by
    construction, and the same A/B geometry the outpaint test used transfers
    verbatim).

    CAVEAT (cannot fail from a bug in the code under test, as written): the
    fixture hand-builds `reference_blocks` with the image already packed
    first, so this test only pins the ARITHMETIC of a layout built in the
    safe order -- it cannot regress a wrong PACKING ORDER, because there is
    no `build_inpaint_references`-style ordering helper yet (that is B-2's
    job, the way `build_outpaint_references` is outpaint's). Once B-2 adds
    one, this test's fixture should be replaced by that helper's actual
    output, the way the outpaint regression test consumes
    `build_outpaint_references` rather than hand-building its input.
    """
    head_video_lat_frames = 37   # minimax_h3_latent_frames(124)
    target_lat_frames = 42       # minimax_h3_latent_frames(141)
    lh, lw = 24, 40
    num_text_tokens = 50

    layout = ops.build_ref2va_packed_layout(
        text_token_tags=[1] * num_text_tokens,
        reference_blocks=[("image", False), ("video", False)],
        condition_latent_shapes=[(1, lh, lw), (head_video_lat_frames, lh, lw)],
        reference_audio_row_counts=[],
        num_latent_frames=target_lat_frames,
        latent_height=lh, latent_width=lw,
        num_audio_latents=0,
        pinned_video_frames=(0,),
    )
    pos = layout["position_ids"]
    rows_per_frame = layout["rows_per_frame"]

    # The image reference is the first block: its rows sit right after the
    # text span.
    image_time = float(pos[num_text_tokens, 0])

    # Frame 0's own time is read directly off position_ids at the target
    # block's own physical start -- unaffected by the pin permutation.
    video_indices = layout["video_indices"]
    num_condition_video_rows = int(layout["num_condition_video_rows"])
    # The physical start of the target video block: the reference block's
    # row count (image + video reference), since there is no anchor here.
    physical_target_start = num_text_tokens + rows_per_frame + head_video_lat_frames * rows_per_frame
    frame0_time = float(pos[physical_target_start, 0])
    # ref video's own rows (image + video reference blocks) + 1 pinned frame.
    expected_condition_rows = rows_per_frame + head_video_lat_frames * rows_per_frame + rows_per_frame
    assert num_condition_video_rows == expected_condition_rows

    binding_radius = (10.0 / 3.0)  # A1: argmin within +/-2 frames == +/-(2*5/3) rotary units
    assert abs(frame0_time - image_time) > binding_radius, (
        f"image reference at t={image_time} is within the pinned frame's own binding radius "
        f"({binding_radius}) of frame 0 at t={frame0_time} -- it will compete with the pin for "
        f"the target origin instead of conditioning the whole span")


# ---------------------------------------------------------------------------
# B-2a: the decode-side permutation offset (`num_pinned_video_rows` /
# `num_pinned_audio_rows`, `minimax_h3.py`'s un-permute at the video/audio
# decode sites). The permutations this builder returns are TARGET-BLOCK-
# RELATIVE (this file's docstring above); a decode site that un-permutes in
# FULL-row space without first skipping the reference/anchor prefix would
# silently read a reference row as if it were a clip row. These tests
# reproduce the decode-site formula directly against fabricated "denoised
# rows" tensors, without a model, and fail if the offset is dropped.
# ---------------------------------------------------------------------------

def _decode_unpermute(full_rows, layout, *, modality):
    """A COPY of the formula `minimax_h3.py`'s decode phase uses (NOT the
    shipped code -- this reimplements it against a fabricated tensor so the
    arithmetic can be checked without a GPU/loaded model). `full_rows` is
    FULL-row space: [reference/anchor rows | possibly-pinned target rows],
    the shape `video_rows`/`audio_rows` has after the draw-time
    reference/anchor prefix is concatenated onto the (already permuted)
    target block.

    Being a copy, this cannot by itself catch the shipped site drifting from
    it -- `test_decode_unpermute_formula_matches_the_shipped_source` below
    provides that coupling by asserting the real source contains the same
    subtraction/slice pattern.
    """
    n_cond = int(layout[f"num_condition_{modality}_rows"])
    num_pinned = int(layout.get(f"num_pinned_{modality}_rows", 0) or 0)
    order = layout[f"{modality}_row_order"]
    if order is None:
        return full_rows[n_cond:]
    n_cond_reference = n_cond - num_pinned
    return full_rows[n_cond_reference:][order]


def test_decode_unpermute_formula_matches_the_shipped_source():
    """Coupling for `_decode_unpermute` above: it is a copy of the formula,
    not a call into the shipped site, so this asserts the real source
    (`MiniMaxH3Mixin._generate_minimax_h3`) still uses the identical
    subtraction-then-slice-then-permute pattern on both the video and audio
    decode branches -- catching the class of drift a pure copy cannot.
    """
    import inspect
    from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin

    source = inspect.getsource(MiniMaxH3Mixin._generate_minimax_h3)
    video_branch = source[source.index("video_row_order = layout[\"video_row_order\"]"):]
    video_branch = video_branch[:video_branch.index("decode_start")]
    assert "n_cond_video - int(layout.get(\"num_pinned_video_rows\"" in video_branch
    assert "][video_row_order" in video_branch

    audio_branch = source[source.index("audio_row_order = layout[\"audio_row_order\"]"):]
    audio_branch = audio_branch[:audio_branch.index("audio_latents = ops.unpack_audio_rows")]
    assert "n_cond_audio - int(layout.get(\"num_pinned_audio_rows\"" in audio_branch
    assert "][audio_row_order" in audio_branch


def test_decode_unpermute_recovers_frame_major_order_past_the_reference_prefix():
    """A ref2va layout with ONE image reference (4 rows) plus a video pin on
    frames (0, 2) of a 6-frame target. Fabricate `full_rows` as
    [reference sentinel rows | pinned-permuted target rows] and assert the
    decode formula recovers ascending (frame-major) target values -- and,
    separately, that skipping the offset (the bug this test guards against)
    would NOT.
    """
    layout = _build(pinned_video_frames=(0, 2))
    rows_per_frame = layout["rows_per_frame"]
    num_target_video_rows = _TARGET_FRAMES * rows_per_frame
    num_reference_video_rows = rows_per_frame  # the one image reference's own rows

    # Reference rows: sentinel values far outside the target value range, so
    # any leak into the "clip" slice is unmistakable.
    reference_rows = torch.full((num_reference_video_rows, 1), -999.0)
    # Target rows in TRUE frame-major order (what the VAE must decode).
    frame_major_target = torch.arange(num_target_video_rows, dtype=torch.float32).unsqueeze(1)
    # The draw applies `video_row_permutation` to the frame-major block
    # before it is written by the denoise loop -- reproduce that.
    permuted_target = frame_major_target[layout["video_row_permutation"]]
    full_rows = torch.cat([reference_rows, permuted_target], dim=0)

    recovered = _decode_unpermute(full_rows, layout, modality="video")
    assert torch.equal(recovered, frame_major_target), (
        "decode-site un-permute did not recover frame-major target order -- the "
        "reference/anchor prefix offset is wrong")
    assert not torch.any(recovered == -999.0), (
        "a reference row leaked into the decoded clip -- the offset was not applied")

    # The regression this test exists to catch: un-permuting in FULL-row space
    # WITHOUT the reference-prefix offset (the pre-fix bug) reads a reference
    # row as if it were a clip row for the first `num_reference_video_rows`
    # positions of the un-permuted result.
    buggy = full_rows[layout["video_row_order"]]
    assert not torch.equal(buggy, frame_major_target), (
        "the offset-less formula accidentally matches the correct one on this "
        "fixture -- the fixture no longer exercises the bug this test guards "
        "against")
    assert torch.any(buggy == -999.0), (
        "the offset-less formula was expected to leak a reference sentinel row")


def test_decode_unpermute_audio_recovers_channel_major_order_past_the_reference_prefix():
    """Mirrored on the audio side: a reference AUDIO block (4 rows) plus an
    audio pin on latent 1 of a 3-latent target.
    """
    audio_ref_latents = 2
    audio_ref_rows = audio_ref_latents * ops.AUDIO_CHANNELS  # 4 rows
    layout = ops.build_ref2va_packed_layout(
        text_token_tags=[1] * _NUM_TEXT_TOKENS,
        reference_blocks=[("audio", True)],
        condition_latent_shapes=[],
        reference_audio_row_counts=[audio_ref_rows],
        num_latent_frames=_TARGET_FRAMES,
        latent_height=_LAT_H, latent_width=_LAT_W,
        num_audio_latents=_NUM_AUDIO_LATENTS,
        pinned_audio_latents=(1,),
    )
    num_target_audio_rows = _NUM_AUDIO_LATENTS * ops.AUDIO_CHANNELS

    reference_rows = torch.full((audio_ref_rows, 1), -999.0)
    channel_major_target = torch.arange(num_target_audio_rows, dtype=torch.float32).unsqueeze(1)
    permuted_target = channel_major_target[layout["audio_row_permutation"]]
    full_rows = torch.cat([reference_rows, permuted_target], dim=0)

    recovered = _decode_unpermute(full_rows, layout, modality="audio")
    assert torch.equal(recovered, channel_major_target)
    assert not torch.any(recovered == -999.0)

    buggy = full_rows[layout["audio_row_order"]]
    assert not torch.equal(buggy, channel_major_target)
    assert torch.any(buggy == -999.0)


def test_num_pinned_video_rows_is_zero_without_a_pin_and_equal_to_the_condition_count_on_fl2va():
    """`num_pinned_video_rows` is the new key both builders return. On
    `build_packed_layout` (fl2va) a pin always REPLACES the anchor count
    (mutual exclusion), so it equals `num_condition_video_rows` whenever a pin
    is present, and both builders agree it is 0 when nothing is pinned --
    which is exactly what makes today's fl2va/no-reference decode paths
    unaffected by this change.
    """
    unpinned_ref2va = _build()
    assert int(unpinned_ref2va["num_pinned_video_rows"]) == 0
    assert int(unpinned_ref2va["num_pinned_audio_rows"]) == 0

    pinned_ref2va = _build(pinned_video_frames=(0,), pinned_audio_latents=(1,))
    # ref2va: the pin's share is a SUBSET of the (reference + pin) total.
    assert int(pinned_ref2va["num_pinned_video_rows"]) < int(pinned_ref2va["num_condition_video_rows"])

    unpinned_fl2va = ops.build_packed_layout(
        _NUM_TEXT_TOKENS, _TARGET_FRAMES, _LAT_H, _LAT_W, _NUM_AUDIO_LATENTS,
    )
    assert int(unpinned_fl2va["num_pinned_video_rows"]) == 0
    assert int(unpinned_fl2va["num_pinned_audio_rows"]) == 0

    pinned_fl2va = ops.build_packed_layout(
        _NUM_TEXT_TOKENS, _TARGET_FRAMES, _LAT_H, _LAT_W, _NUM_AUDIO_LATENTS,
        pinned_video_frames=(0, 2), pinned_audio_latents=(1,),
    )
    # fl2va: the pin's share equals the WHOLE condition count -- no reference
    # prefix exists there, so the decode offset is always 0 on this builder.
    assert int(pinned_fl2va["num_pinned_video_rows"]) == int(pinned_fl2va["num_condition_video_rows"])
    assert int(pinned_fl2va["num_pinned_audio_rows"]) == int(pinned_fl2va["num_condition_audio_rows"])


# ---------------------------------------------------------------------------
# B-2: the row-count WARNING (owner correction, design doc §6) -- never a
# refusal. `minimax_h3_inpaint_reference_row_count_warning` is the pure
# formatter `_generate_minimax_h3` calls once the layout (and therefore
# `ops.packed_row_counts`) is known, before the DiT is staged.
# ---------------------------------------------------------------------------

def test_row_count_warning_reports_the_computed_numbers():
    """Against a REAL built layout (one image reference, a 6-frame target,
    frames 0 and 2 pinned): `condition_video` (12) is the reference's own 4
    rows PLUS the pin's 8, not the reference's alone, and the free
    `target_video` (16) is not "every frame of the clip" -- the pinned 8 are
    also the clip's own rows. The message must report the clip's total
    (pinned + free = 24 video + 6 audio = 30) and the reference's own share
    (12 + 0 condition - 8 pinned = 4, the image reference's actual
    `rows_per_frame`), not the raw, uncorrected `row_counts` fields.
    """
    layout = _build(pinned_video_frames=(0, 2))
    row_counts = ops.packed_row_counts(layout)
    num_pinned_video_rows = int(layout["num_pinned_video_rows"])
    num_pinned_audio_rows = int(layout["num_pinned_audio_rows"])

    # Pin the raw shape this test's correctness depends on, so a fixture
    # change fails loudly here rather than silently changing what is
    # asserted below.
    assert row_counts == {
        "text": 5, "condition_video": 12, "target_video": 16,
        "condition_audio": 0, "target_audio": 6, "total": 39,
    }
    assert num_pinned_video_rows == 8
    assert num_pinned_audio_rows == 0

    message, code = minimax_h3_inpaint_reference_row_count_warning(
        row_counts, num_references=1,
        num_pinned_video_rows=num_pinned_video_rows,
        num_pinned_audio_rows=num_pinned_audio_rows,
    )

    assert code == "minimax_h3_inpaint_reference_row_count"
    assert "39 row" in message
    assert "30 row" in message  # the CLIP's own rows: pinned (8) + free (16+6)
    assert "4 row" in message   # the REFERENCE's own rows: 12 condition - 8 pinned
    assert "1 reference" in message
    # The numbers that would be wrong under the pre-fix formula (target_video
    # alone as "the clip", condition_video+condition_audio alone as "the
    # references") must not appear as the reported clip/reference counts.
    assert "16 row" not in message
    assert "12 row" not in message
    # Factual only -- no threshold language.
    for banned in ("recommended", "should", "limit of", "maximum of"):
        assert banned not in message.lower()


def test_row_count_warning_never_raises_on_a_well_formed_row_counts_dict():
    """The correction this function exists to satisfy: sequence length is
    NEVER a refusal. This only covers WELL-FORMED input (every key
    `packed_row_counts` produces present) -- an incomplete `row_counts` dict
    still raises `KeyError`, which is a caller bug, not a refusal.
    """
    for row_counts, num_references in (
        (ops.packed_row_counts(_build()), 1),
        (ops.packed_row_counts(_build(pinned_video_frames=(0, 1, 2, 3, 4, 5))), 3),
        ({"text": 0, "condition_video": 0, "target_video": 0,
          "condition_audio": 0, "target_audio": 0, "total": 0}, 0),
    ):
        message, code = minimax_h3_inpaint_reference_row_count_warning(
            row_counts, num_references=num_references)
        assert isinstance(message, str) and message
        assert code == "minimax_h3_inpaint_reference_row_count"


def test_row_count_warning_names_the_outpaint_contrast():
    """The design's specific factual claim (§3 Option B "Costs/risks"): this
    layout carries every clip frame, unlike outpaint's generated-span-only
    layout -- named so a reader does not assume inpaint is cheap the way
    outpaint's partial span is.
    """
    layout = _build(pinned_video_frames=(0,))
    row_counts = ops.packed_row_counts(layout)
    message, _code = minimax_h3_inpaint_reference_row_count_warning(row_counts, num_references=2)
    assert "outpaint" in message.lower()
    assert "every frame" in message.lower()


# ---------------------------------------------------------------------------
# PHASE B-3-open: the SAMPLING LOOP's own un-permute (ops.denoise()'s preview
# path), on a ref2va reference+pin layout -- reachable through the route only
# now that the gate is open. The decode-side offset arithmetic is already
# pinned numerically above (test_decode_unpermute_recovers_frame_major_order_
# past_the_reference_prefix); this is its twin inside the loop, which
# `_generate_minimax_h3` feeds `video_row_order` into via `ops.denoise()`.
# ---------------------------------------------------------------------------

class _StubScheduler:
    """Same fixed-step stub `minimax_h3_temporal_inpaint_test.py` uses."""

    def __init__(self, timesteps):
        self.timesteps = torch.tensor(timesteps, dtype=torch.float32)

    def set_shift(self, shift):
        pass

    def set_timesteps(self, steps, device=None):
        pass

    def set_begin_index(self, index):
        pass

    def step(self, velocity, timestep, sample, return_dict=False):
        return (sample - 0.25 * velocity,)


def test_the_sampling_loop_preview_un_permutes_past_a_reference_prefix():
    """A ref2va layout with ONE image reference (4 rows) plus a video pin on
    frames (0, 2) of the same 6-frame target the decode-side test above uses
    -- run through `ops.denoise()` itself (not just the offset formula in
    isolation), asserting the preview latents come back frame-major and the
    reference's own 4 rows never leak into them.

    This is the loop-internal twin of the decode fix: `denoise()` computes
    its own `n_cond_reference_video_rows` (h3_pipeline_ops.py, `denoise`'s
    docstring) independently of the decode site in `_generate_minimax_h3`,
    so a regression in one does not fail a test of the other.
    """
    layout = _build(pinned_video_frames=(0, 2))
    rows_per_frame = layout["rows_per_frame"]
    num_reference_video_rows = rows_per_frame  # the one image reference's own rows
    num_target_video_rows = _TARGET_FRAMES * rows_per_frame
    channels = 24
    row_width = channels * 4  # patch (1, 2, 2) -> patch volume 4, the default `denoise()` uses

    # Reference rows: sentinel values far outside the target value range.
    reference_rows = torch.full((num_reference_video_rows, row_width), -999.0)
    frame_major_target = torch.arange(
        num_target_video_rows, dtype=torch.float32).unsqueeze(1).expand(-1, row_width).clone()
    for frame in range(_TARGET_FRAMES):
        frame_major_target[frame * rows_per_frame:(frame + 1) * rows_per_frame] += 1000.0 * frame
    permuted_target = frame_major_target[layout["video_row_permutation"]]
    packed = torch.cat([reference_rows, permuted_target], dim=0).clone()
    before = packed.clone()

    n_audio = layout["audio_indices"].numel()
    velocity = torch.full((1, packed.shape[0], packed.shape[1]), 2.0)
    seen = []
    ops.denoise(
        lambda **kw: (velocity, torch.zeros(1, n_audio, 32)),
        _StubScheduler([0.75]), _StubScheduler([0.75]),
        prompt_embeds=torch.zeros(1, 3, 8), layout=layout,
        video_rows=packed, audio_rows=torch.zeros(n_audio, 32),
        num_inference_steps=1, device="cpu",
        step_callback=lambda *a: seen.append(a),
        preview_latent_shape=(_TARGET_FRAMES, _LAT_H, _LAT_W),
        video_row_order=layout["video_row_order"],
        latent_channels=channels,
    )
    _index, _total, latents, _extra, pred_x0 = seen[0]
    assert latents.shape == (1, channels, _TARGET_FRAMES, _LAT_H, _LAT_W) == pred_x0.shape

    n_cond = layout["num_condition_video_rows"]
    n_cond_reference = n_cond - int(layout["num_pinned_video_rows"])  # the offset itself
    assert n_cond_reference == num_reference_video_rows
    stepped = before.clone()
    stepped[n_cond:] = before[n_cond:] - 0.25 * velocity[0, n_cond:]
    expected_latents = ops.unpatchify_video_rows(
        stepped[n_cond_reference:][layout["video_row_order"]],
        _TARGET_FRAMES, _LAT_H, _LAT_W, latent_channels=channels)
    assert torch.equal(latents, expected_latents), (
        "the sampling loop's own preview un-permute did not recover frame-major "
        "order past the reference prefix")

    # Negative control: skipping the reference-prefix offset -- indexing the
    # FULL packed rows (reference block included) by video_row_order, which
    # is target-block-relative -- must leak a reference sentinel into the
    # result the way the pre-fix bug did, proving this fixture still
    # exercises the offset rather than passing by coincidence.
    buggy = stepped[layout["video_row_order"]]
    assert torch.any(buggy == -999.0), (
        "the fixture no longer exercises the missing-offset bug this test guards against")
