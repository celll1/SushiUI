"""MiniMax-H3 temporal inpaint: the pinned-frame permutation is a contract.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_temporal_inpaint_test.py -v

WHAT IS BEING PINNED HERE
-------------------------
`build_packed_layout(pinned_video_frames=...)` regenerates part of a clip by
PERMUTING `video_indices` so the preserved latent frames' rows lead, and
counting them as conditioning. The prefix count then addresses an arbitrary
index set with no index-set machinery: the transformer scatters rows with
`index_copy` and reads them back with `index_select`, attention is full and
unmasked, and everything else is addressed by sequence POSITION, so a
permutation of the index block together with the same permutation of the rows
is a bitwise no-op (measured on the real vendored transformer, and on the
released fl2va weights: preserved-span RMS 3.12 against a VAE round-trip floor
of 3.15, control 75.69 -- `scratchpad/minimax_h3_ti_probe_results.md`).

Three things therefore have to be true, and each has a test that fails if a
plausible "simplification" breaks it:

* **nothing changes when nothing is pinned.** The default is the identity
  permutation, so K0.3's recorded layouts are byte-identical -- including the
  case where a pin IS requested but happens to be the identity (a whole-clip or
  leading-prefix pin), which must not perturb a single tensor either.
* **the prefix addresses the REQUESTED frames.** The mutant that matters is a
  permutation that is built and then not applied: the layout still looks
  well-formed and still pins `len(pinned)` frames' worth of rows -- just the
  WRONG frames (0..k-1). Only the rotary clock of the prefix rows can see that,
  which is what `test_the_prefix_carries_the_pinned_frames_clock` reads.
* **the preview un-permutes.** With a pin the conditioning prefix is clip
  content, so `denoise`'s preview takes every video row and restores frame-major
  order. A preview that ignores the permutation, or applies it the wrong way
  round, produces a garbled but correctly-shaped latent -- so the test asserts
  both the right answer AND that the two wrong ones differ from it.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.models.minimax_h3 import h3_pipeline_ops as ops  # noqa: E402

_TENSORS = ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices")

# (label, text tokens, T_lat, lat H, lat W, audio latents, anchors)
GEOMETRIES = [
    ("t2va min clip T=22 384x640", 16, 7, 24, 40, 37, ()),
    ("t2va T=124 768x1344", 64, 37, 48, 84, 207, ()),
    ("fl2va 2 images (first+last)", 16, 7, 24, 40, 37, ("first", "last")),
    ("probe canvas T=124 640x384", 37, 37, 24, 40, 207, ()),
    ("trainer callsite", 11, 1, 4, 6, 0, ()),
]

# The P-TI-1 probe's own request: 640x384 (24x40 latents, 240 rows/frame),
# 124 frames (37 latent frames), latent frames 12-24 pinned = pixel 39-84.
TI1 = dict(text=37, t_lat=37, lh=24, lw=40, n_aud=207)
TI1_PINNED = tuple(range(12, 25))


def _layout(geometry, **kwargs):
    _, text, t_lat, lh, lw, n_aud, anchors = geometry
    return ops.build_packed_layout(text, t_lat, lh, lw, n_aud, keyframe_anchors=anchors, **kwargs)


# ---------------------------------------------------------------------------
# The default is the identity, byte for byte
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("geometry", GEOMETRIES, ids=[g[0] for g in GEOMETRIES])
def test_no_pinned_frames_is_bitwise_the_shipped_layout(geometry):
    """An absent argument and an empty one are the same layout, byte for byte.

    K0.3's recorded index tables and the anchor digests in
    `minimax_h3_layout_test.py` are what this keeps true; here it is asserted
    directly against the no-argument build so a failure names the pin.
    """
    shipped = _layout(geometry)
    empty = _layout(geometry, pinned_video_frames=())
    for key in _TENSORS:
        assert torch.equal(shipped[key], empty[key]), key
    for key in ("sequence_length", "num_condition_video_rows", "num_condition_audio_rows",
                "rows_per_frame"):
        assert shipped[key] == empty[key], key
    assert shipped["video_row_permutation"] is None
    assert shipped["video_row_order"] is None
    assert empty["video_row_permutation"] is None


@pytest.mark.parametrize("pinned,label", [
    (tuple(range(37)), "the whole clip"),
    ((0, 1, 2), "a leading prefix"),
    ((0,), "the first frame alone"),
], ids=lambda x: x if isinstance(x, str) else "")
def test_an_identity_permutation_changes_nothing_but_the_count(pinned, label):
    """A pin whose frames ALREADY lead must not perturb a tensor.

    `video_indices` is built ascending, so pinning frames 0..k-1 is the identity
    permutation. Only `num_condition_video_rows` may move -- if any tensor moves
    too, the permutation is being applied where it should be a no-op.
    """
    plain = ops.build_packed_layout(37, 37, 24, 40, 207)
    pin = ops.build_packed_layout(37, 37, 24, 40, 207, pinned_video_frames=pinned)
    for key in _TENSORS:
        assert torch.equal(plain[key], pin[key]), key
    rows_per_frame = plain["rows_per_frame"]
    assert plain["num_condition_video_rows"] == 0
    assert pin["num_condition_video_rows"] == len(pinned) * rows_per_frame
    identity = torch.arange(plain["video_indices"].numel())
    assert torch.equal(pin["video_row_permutation"], identity)
    assert torch.equal(pin["video_row_order"], identity)


# ---------------------------------------------------------------------------
# The prefix addresses the requested frames -- the mutant that matters
# ---------------------------------------------------------------------------

def test_the_prefix_carries_the_pinned_frames_clock():
    """NEGATIVE CONTROL: a permutation that is built and then not applied.

    Such a mutant produces a layout that passes every SET property -- the index
    block is still a cover, the tags are still right, the count is still
    `len(pinned) * rows_per_frame` -- while pinning latent frames 0..k-1 instead
    of the requested span. The rotary clock is what separates them: each latent
    frame sits at its own time on the packed sequence's temporal axis (distinct
    in float32 by at least 5/3 per frame), so reading the prefix rows' time axis
    names exactly which frames were pinned.
    """
    layout = ops.build_packed_layout(TI1["text"], TI1["t_lat"], TI1["lh"], TI1["lw"],
                                     TI1["n_aud"], pinned_video_frames=TI1_PINNED)
    rows_per_frame = layout["rows_per_frame"]
    n_cond = layout["num_condition_video_rows"]
    assert n_cond == len(TI1_PINNED) * rows_per_frame == 3120

    times = layout["position_ids"][:, 0]
    clock = (float(TI1["text"]) + torch.cat([
        torch.zeros(1, dtype=torch.float64),
        torch.tensor([ops.ROPE_FRAME_RESCALE * ops.ROPE_FRAMES_PER_LATENT[i % 5]
                      for i in range(TI1["t_lat"])], dtype=torch.float64)[:-1].cumsum(0),
    ])).to(torch.float32)
    assert len(set(clock.tolist())) == TI1["t_lat"], "the frame clock is not injective in float32"

    pinned_times = times[layout["video_indices"][:n_cond]]
    free_times = times[layout["video_indices"][n_cond:]]
    assert sorted(set(pinned_times.tolist())) == sorted(clock[list(TI1_PINNED)].tolist())
    assert sorted(set(free_times.tolist())) == sorted(
        clock[[f for f in range(TI1["t_lat"]) if f not in set(TI1_PINNED)]].tolist())
    # ... and the mutant's answer is genuinely a different set, so the check is
    # not vacuous at this geometry.
    assert sorted(clock[list(TI1_PINNED)].tolist()) != sorted(clock[:len(TI1_PINNED)].tolist())


def test_a_mid_span_pin_keeps_every_k03_invariant_except_ascending():
    """Permutation-of / cover / disjointness / tags -- the M2 invariant set.

    The single property a pin gives up is that `video_indices` is ascending,
    which the ordinary layouts still have and which nothing consumes (the
    transformer pairs `index_copy` with `index_select` on the same tensor).
    """
    plain = ops.build_packed_layout(TI1["text"], TI1["t_lat"], TI1["lh"], TI1["lw"], TI1["n_aud"])
    pin = ops.build_packed_layout(TI1["text"], TI1["t_lat"], TI1["lh"], TI1["lw"], TI1["n_aud"],
                                  pinned_video_frames=TI1_PINNED)
    assert pin["sequence_length"] == plain["sequence_length"]
    assert torch.equal(pin["video_indices"].sort().values, plain["video_indices"])
    assert not bool((torch.diff(pin["video_indices"]) > 0).all()), \
        "a mid-span pin left video_indices ascending -- it was not applied"

    blocks = [pin["text_indices"], pin["audio_indices"], pin["video_indices"]]
    assert torch.equal(torch.cat(blocks).sort().values, torch.arange(pin["sequence_length"]))
    tags = pin["token_tags"]
    assert (tags[pin["video_indices"]] == ops.VIDEO_TAG).all()
    assert (tags[pin["audio_indices"]] == ops.AUDIO_TAG).all()
    assert (tags[pin["text_indices"]] == ops.TEXT_TAG).all()
    assert torch.equal(pin["token_tags"], plain["token_tags"])
    assert torch.equal(pin["position_ids"], plain["position_ids"])


def test_build_row_timesteps_pins_the_permuted_prefix_and_is_itself_unchanged():
    """The pin rides the SHIPPED `build_row_timesteps`, which V1 does not touch.

    It reads `video_indices[:n_cond_video]`, an index set, so permuting the
    block is all it takes to move the 0.999 pin onto the clip's own rows.
    """
    pin = ops.build_packed_layout(TI1["text"], TI1["t_lat"], TI1["lh"], TI1["lw"], TI1["n_aud"],
                                  pinned_video_frames=TI1_PINNED)
    unique, index = ops.build_row_timesteps(pin, 0.1, 0.1)
    row_timesteps = unique[index]
    n_cond = pin["num_condition_video_rows"]
    assert sorted(round(float(t), 4) for t in unique.tolist()) == [0.1, 0.999]
    assert (row_timesteps[pin["video_indices"][:n_cond]] == ops.VISUAL_COND_TIMESTEP).all()
    free = row_timesteps[pin["video_indices"][n_cond:]]
    assert torch.allclose(free, torch.full_like(free, 0.1))


def test_the_permutation_and_its_inverse_round_trip():
    """`packed = frame_major[permutation]` and `frame_major = packed[order]`.

    The two keys are not interchangeable at this geometry -- asserted, because a
    caller that swaps them would silently scramble the clip if they happened to
    be equal here.
    """
    pin = ops.build_packed_layout(TI1["text"], TI1["t_lat"], TI1["lh"], TI1["lw"], TI1["n_aud"],
                                  pinned_video_frames=TI1_PINNED)
    permutation, order = pin["video_row_permutation"], pin["video_row_order"]
    assert not torch.equal(permutation, order), "the permutation is its own inverse here"
    rows = torch.randn(permutation.numel(), 3, generator=torch.Generator().manual_seed(0))
    assert torch.equal(rows[permutation][order], rows)
    plain = ops.build_packed_layout(TI1["text"], TI1["t_lat"], TI1["lh"], TI1["lw"], TI1["n_aud"])
    assert torch.equal(plain["video_indices"][permutation], pin["video_indices"])


def test_the_layout_is_a_function_of_the_pinned_SET():
    """Request order does not make a second layout for the same frames."""
    kwargs = dict(num_text_tokens=16, num_latent_frames=7, latent_height=24,
                  latent_width=40, num_audio_latents=37)
    ascending = ops.build_packed_layout(**kwargs, pinned_video_frames=(2, 3, 4))
    shuffled = ops.build_packed_layout(**kwargs, pinned_video_frames=(4, 2, 3))
    for key in _TENSORS + ("video_row_permutation", "video_row_order"):
        assert torch.equal(ascending[key], shuffled[key]), key


@pytest.mark.parametrize("frames,match", [
    ((7,), "outside this clip"),
    ((-1,), "outside this clip"),
    ((0, 0), "distinct"),
    ((1.0,), "integer LATENT-frame index"),
    (("first",), "integer LATENT-frame index"),
    ((True,), "integer LATENT-frame index"),
    ((None,), "integer LATENT-frame index"),
], ids=repr)
def test_a_frame_the_builder_cannot_pin_is_refused(frames, match):
    """7 latent frames means 0..6; everything else is a refusal, not a clamp."""
    with pytest.raises(ValueError, match=match):
        ops.build_packed_layout(16, 7, 24, 40, 37, pinned_video_frames=frames)


def test_anchors_and_pinned_frames_are_refused_together():
    """Both want the same prefix, and only one of them can have it.

    Anchors reserve EXTRA rows ahead of the clip; the pin re-uses the prefix for
    rows OF the clip. Composing them was never measured and would silently
    mis-address the count, so it is a refusal rather than a guess.
    """
    with pytest.raises(ValueError, match="keyframe anchors with pinned video frames"):
        ops.build_packed_layout(16, 7, 24, 40, 37, keyframe_anchors=("first",),
                                pinned_video_frames=(3,))


def test_pinned_frames_are_keyword_only_so_the_trainer_callsite_cannot_see_them():
    """`training/ops/minimax_h3_ops.py` calls this positionally with no keywords."""
    import inspect
    parameter = inspect.signature(ops.build_packed_layout).parameters["pinned_video_frames"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == ()
    trainer_shaped = ops.build_packed_layout(11, 1, 4, 6, 0)
    assert trainer_shaped["num_condition_video_rows"] == 0
    assert trainer_shaped["video_row_order"] is None


def test_matches_the_ti1_probe_harness_arithmetic():
    """The shipped layout equals the one the measured P-TI-1 arm actually ran.

    The harness (session scratchpad `ti/ti1_pin_harness.py`) monkeypatched this
    function in its own process: it built the unpinned layout, gathered
    `video_indices` through `cat(arange(f*rpf, (f+1)*rpf) for f in pinned + rest)`
    and set the count to `len(pinned) * rpf`. Reproducing that bitwise is a free
    end-to-end check of this phase against something already shown to work on
    the released weights.
    """
    text, t_lat, lh, lw, n_aud = (TI1["text"], TI1["t_lat"], TI1["lh"], TI1["lw"], TI1["n_aud"])
    for pinned in (TI1_PINNED, (0, 1, 5, 6), (36,), (0, 36)):
        harness = ops.build_packed_layout(text, t_lat, lh, lw, n_aud)
        rows_per_frame = harness["rows_per_frame"]
        order = list(pinned) + [f for f in range(t_lat) if f not in set(pinned)]
        permutation = torch.cat([torch.arange(f * rows_per_frame, (f + 1) * rows_per_frame)
                                 for f in order])
        shipped = ops.build_packed_layout(text, t_lat, lh, lw, n_aud, pinned_video_frames=pinned)
        assert torch.equal(shipped["video_indices"], harness["video_indices"][permutation]), pinned
        assert shipped["num_condition_video_rows"] == len(pinned) * rows_per_frame
        assert torch.equal(shipped["video_row_permutation"], permutation)
        assert torch.equal(shipped["video_row_order"], torch.argsort(permutation))


# ---------------------------------------------------------------------------
# The preview -- the one denoise-side change, and the actual defect surface
# ---------------------------------------------------------------------------

class _StubScheduler:
    """Steps by `sample - 0.25 * velocity`, so a preview read from the POST-step
    rows cannot accidentally look right (the `minimax_h3_layout_test` stub)."""

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


def _run_denoise(layout, video_rows, *, video_row_order, preview_shape, channels):
    n_audio = layout["audio_indices"].numel()
    velocity = torch.full((1, video_rows.shape[0], video_rows.shape[1]), 2.0)
    seen = []
    out_video, _ = ops.denoise(
        lambda **kw: (velocity, torch.zeros(1, n_audio, 32)),
        _StubScheduler([0.75]), _StubScheduler([0.75]),
        prompt_embeds=torch.zeros(1, 3, 8), layout=layout,
        video_rows=video_rows, audio_rows=torch.zeros(n_audio, 32),
        num_inference_steps=1, device="cpu",
        step_callback=lambda *a: seen.append(a),
        preview_latent_shape=preview_shape, video_row_order=video_row_order,
        latent_channels=channels,
    )
    return out_video, seen, velocity


def test_the_preview_of_a_pinned_request_is_frame_major():
    """NEGATIVE CONTROL: the preview must un-permute, the right way round.

    Two mutants produce a correctly-shaped, plausible-looking latent preview and
    are invisible to any shape or finiteness check: ignoring `video_row_order`
    entirely, and applying the forward permutation instead of its inverse. Both
    are asserted to differ from the answer here -- and the permutation is
    asserted not to be an involution at this geometry, or the second assertion
    would be vacuous.
    """
    t_lat, lh, lw, channels = 5, 4, 4, 24
    pinned = (1, 3)
    layout = ops.build_packed_layout(3, t_lat, lh, lw, 5, pinned_video_frames=pinned)
    order = layout["video_row_order"]
    permutation = layout["video_row_permutation"]
    assert not torch.equal(order, permutation)

    frame_major = torch.randn(layout["video_indices"].numel(), channels * 4,
                              generator=torch.Generator().manual_seed(1))
    # Each latent frame's rows carry a distinct offset, so ANY row that lands in
    # the wrong frame moves the preview.
    rows_per_frame = layout["rows_per_frame"]
    for frame in range(t_lat):
        frame_major[frame * rows_per_frame:(frame + 1) * rows_per_frame] += 100.0 * frame
    packed = frame_major[permutation].clone()
    # `denoise` writes the generated rows IN PLACE, so the comparison needs the
    # pre-step tensor.
    before = packed.clone()

    _, seen, velocity = _run_denoise(layout, packed, video_row_order=order,
                                     preview_shape=(t_lat, lh, lw), channels=channels)
    index, total, latents, extra, pred_x0 = seen[0]
    assert (index, total, extra) == (0, 1, None)
    assert latents.shape == (1, channels, t_lat, lh, lw) == pred_x0.shape

    n_cond = layout["num_condition_video_rows"]
    stepped = before.clone()
    stepped[n_cond:] = before[n_cond:] - 0.25 * velocity[0, n_cond:]
    expected = ops.unpatchify_video_rows(stepped[order], t_lat, lh, lw, latent_channels=channels)
    assert torch.equal(latents, expected)

    # x0 = x_t + sigma * v on the generated rows (sigma = 0.25), and a pinned row
    # previews as itself -- taken from the PRE-step tensor either way.
    x0 = before.clone()
    x0[n_cond:] = before[n_cond:] + 0.25 * velocity[0, n_cond:]
    assert torch.allclose(
        pred_x0, ops.unpatchify_video_rows(x0[order], t_lat, lh, lw, latent_channels=channels),
        atol=1e-6)

    # The two mutants, explicitly.
    ignored = ops.unpatchify_video_rows(stepped, t_lat, lh, lw, latent_channels=channels)
    wrong_way = ops.unpatchify_video_rows(stepped[permutation], t_lat, lh, lw,
                                          latent_channels=channels)
    assert not torch.equal(latents, ignored), "the preview is blind to a missing un-permute"
    assert not torch.equal(latents, wrong_way), "the preview is blind to an inverted un-permute"


def test_the_pinned_rows_are_never_written_by_the_loop():
    """The write slice is untouched: pinned rows come back bit-identical.

    This is the whole mechanism -- the prefix count is the protection, and it is
    the SHIPPED slice `video_rows[n_cond_video:]` that provides it.
    """
    t_lat, lh, lw, channels = 5, 4, 4, 24
    layout = ops.build_packed_layout(3, t_lat, lh, lw, 5, pinned_video_frames=(1, 3))
    n_cond = layout["num_condition_video_rows"]
    packed = torch.randn(layout["video_indices"].numel(), channels * 4,
                         generator=torch.Generator().manual_seed(2))
    before = packed.clone()
    out, _, velocity = _run_denoise(layout, packed, video_row_order=layout["video_row_order"],
                                    preview_shape=(t_lat, lh, lw), channels=channels)
    assert torch.equal(out[:n_cond], before[:n_cond])
    assert not torch.equal(out[n_cond:], before[n_cond:])


def test_the_preview_without_a_permutation_is_the_shipped_one():
    """No `video_row_order` means the shipped behaviour, byte for byte.

    Anchor conditioning rows stay OUT of the preview -- with anchors the prefix
    is not clip content, and unpatchifying it into the clip's geometry would not
    even have the right row count.
    """
    t_lat, lh, lw, channels = 2, 4, 4, 24
    layout = ops.build_packed_layout(3, t_lat, lh, lw, 5, keyframe_anchors=("first",))
    n_cond = layout["num_condition_video_rows"]
    packed = torch.randn(layout["video_indices"].numel(), channels * 4,
                         generator=torch.Generator().manual_seed(3))
    before = packed.clone()
    _, seen, velocity = _run_denoise(layout, packed, video_row_order=None,
                                     preview_shape=(t_lat, lh, lw), channels=channels)
    _, _, latents, _, pred_x0 = seen[0]
    assert latents.shape == (1, channels, t_lat, lh, lw)
    assert torch.equal(latents, ops.unpatchify_video_rows(
        before[n_cond:] - 0.25 * velocity[0, n_cond:], t_lat, lh, lw, latent_channels=channels))
    assert torch.allclose(pred_x0, ops.unpatchify_video_rows(
        before[n_cond:] + 0.25 * velocity[0, n_cond:], t_lat, lh, lw, latent_channels=channels),
        atol=1e-6)


def test_a_row_order_that_does_not_fit_the_rows_is_refused():
    """A mismatched order would scramble the preview rather than fail."""
    layout = ops.build_packed_layout(3, 2, 4, 4, 5)
    with pytest.raises(ValueError, match="video_row_order"):
        ops.denoise(
            lambda **kw: None, _StubScheduler([0.75]), _StubScheduler([0.75]),
            prompt_embeds=torch.zeros(1, 3, 8), layout=layout,
            video_rows=torch.zeros(layout["video_indices"].numel(), 96),
            audio_rows=torch.zeros(layout["audio_indices"].numel(), 32),
            num_inference_steps=1, device="cpu",
            video_row_order=torch.arange(3))
