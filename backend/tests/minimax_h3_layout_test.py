"""MiniMax-H3: the packed-sequence layout and the noise draw order are contracts.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/minimax_h3_layout_test.py -v

WHY THIS FILE EXISTS
--------------------
Two of the shipped sampler's invariants cannot be checked by running it: they
are agreements with a checkpoint, and getting either wrong produces a model that
loads perfectly, runs to completion and generates noise.

* **The packed-sequence layout.** ``h3_pipeline_ops.build_packed_layout`` is a
  port of the diffusers ``minimax-h3`` ``MiniMaxH3PrepareLayoutStep``. During
  Phase 0 that port was validated against a SECOND, independent port of
  ComfyUI's ``PackedLayout`` on six fixed shape tuples; the resulting index
  tables are reproduced here as literals so the shipped function is pinned to
  them rather than to a re-derivation of itself.

  **The comparison is on index ORDER, not the index set.** Ordering the audio
  rows row-major instead of channel-major yields the SAME index set and is
  invisible to set equality — it was the one mutant Phase 0's set-equality
  checks could not catch. ``test_audio_rows_are_channel_major`` is the direct
  guard, and every index comparison below uses ``torch.equal`` on the ordered
  tensors.

* **The noise draw order.** One generator, three kinds of draw: conditions (in
  packed order), then the video noise as a 5-D latent tensor, then the audio
  noise directly in row layout. The SHA-256 digests below were recorded in
  Phase 0 against an independent reimplementation of the upstream blocks
  (12/12 exact, CPU and CUDA generators). Any change to the draw order, the
  shapes, the dtype or the layout of the audio draw changes them.

The recorded values are the contract; if a change here needs them updated, the
change is wrong.
"""

import hashlib
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.models.minimax_h3 import h3_pipeline_ops as ops  # noqa: E402


# Phase 0's six fixed shape tuples: (text tokens, latent frames, latent H,
# latent W, audio latents, keyframe anchors) -> the recorded row counts.
CASES = [
    # label                       text  T_lat  lh  lw  n_aud  anchors             S      video  audio  text
    ("t2va min clip T=22 384x640",  16,     7, 24, 40,    37, (),                1770,  1680,    74,   16),
    ("t2va T=124 768x1344",         64,    37, 48, 84,   207, (),               37774, 37296,   414,   64),
    ("fl2va 1 image (first)",       16,     7, 24, 40,    37, ("first",),        2010,  1920,    74,   16),
    ("fl2va 2 images (first+last)", 16,     7, 24, 40,    37, ("first", "last"), 2250,  2160,    74,   16),
    ("T=39 square 512x512",          1,    12, 32, 32,    65, (),                3203,  3072,   130,    1),
    ("T=56 tall 256x768",          256,    17, 16, 48,    93, ("first", "last"), 4090,  3648,   186,  256),
]


def _layout(case):
    _, text, t_lat, lh, lw, n_aud, anchors = case[:7]
    return ops.build_packed_layout(text, t_lat, lh, lw, n_aud, keyframe_anchors=anchors)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_recorded_row_counts(case):
    """The sequence length and the three per-modality row counts, as recorded."""
    label, text, t_lat, lh, lw, n_aud, anchors, seq_len, n_video, n_audio, n_text = case
    layout = _layout(case)
    assert layout["sequence_length"] == seq_len
    assert layout["video_indices"].numel() == n_video
    assert layout["audio_indices"].numel() == n_audio
    assert layout["text_indices"].numel() == n_text


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_indices_are_ordered_disjoint_and_cover(case):
    """The three index blocks tile [0, S) exactly, each in ascending order.

    ``index_copy`` scatters by index and ``index_select`` gathers by index, so a
    layout that covers the sequence but permutes a modality's rows round-trips
    through the transformer and lands every row in the wrong place.
    """
    layout = _layout(case)
    seq_len = layout["sequence_length"]
    video, audio, text = (layout["video_indices"], layout["audio_indices"], layout["text_indices"])
    for name, block in (("video", video), ("audio", audio), ("text", text)):
        assert torch.equal(block, block.sort().values), f"{name} indices are not ascending"
    combined = torch.cat([text, audio, video]).sort().values
    assert torch.equal(combined, torch.arange(seq_len))


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_layout_order_is_text_conditions_audio_video(case):
    """The block ORDER of the sequence, as literal index ranges.

    ``[text | keyframe conditions | audio | video]``, with the conditioning rows
    LEADING the video index block — which is what lets the loop protect them by
    never writing the first ``num_condition_video_rows`` entries.
    """
    label, text, t_lat, lh, lw, n_aud, anchors = case[:7]
    layout = _layout(case)
    rows_per_frame = (lh // 2) * (lw // 2)
    n_cond = len(anchors) * rows_per_frame
    cond_start = text
    audio_start = cond_start + n_cond
    video_start = audio_start + n_aud * ops.AUDIO_CHANNELS

    assert layout["num_condition_video_rows"] == n_cond
    assert torch.equal(layout["text_indices"], torch.arange(text))
    assert torch.equal(layout["audio_indices"],
                       torch.arange(audio_start, video_start))
    assert torch.equal(
        layout["video_indices"],
        torch.cat([torch.arange(cond_start, audio_start),
                   torch.arange(video_start, layout["sequence_length"])]),
    )


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_modality_tags(case):
    """Every row's tag, which indexes the transformer's AdaLN table."""
    layout = _layout(case)
    tags = layout["token_tags"]
    assert (tags[layout["text_indices"]] == ops.TEXT_TAG).all()
    assert (tags[layout["audio_indices"]] == ops.AUDIO_TAG).all()
    # Conditioning rows are VIDEO rows even though they are not generated.
    assert (tags[layout["video_indices"]] == ops.VIDEO_TAG).all()


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_audio_rows_are_channel_major(case):
    """THE mutant set equality cannot catch.

    Channel-major means the first ``n_aud`` audio rows are channel 0 and the
    next ``n_aud`` are channel 1. Two consequences are checked, because the
    index block alone is identical under either ordering:

    * the rotary WIDTH coordinate is pinned to the low extreme of the width grid
      for the first half and the high extreme for the second — a row-major
      interleave would alternate;
    * the rotary TIME coordinate restarts at the beginning for the second half.
    """
    label, text, t_lat, lh, lw, n_aud, anchors = case[:7]
    layout = _layout(case)
    audio_pos = layout["position_ids"][layout["audio_indices"]]

    width_low = audio_pos[:n_aud, 2]
    width_high = audio_pos[n_aud:, 2]
    assert torch.equal(width_low, width_low[:1].expand_as(width_low))
    assert torch.equal(width_high, width_high[:1].expand_as(width_high))
    assert width_low[0] < width_high[0], "the two stereo channels share a width coordinate"

    time_axis = audio_pos[:, 0]
    assert torch.equal(time_axis[:n_aud], time_axis[n_aud:]), \
        "the second stereo channel does not restart the audio clock (rows are not channel-major)"
    assert float(time_axis[0]) == float(text), "the audio clock does not start after the text span"


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_index_copy_index_select_roundtrip(case):
    """Scatter-then-gather is the identity for every modality, bitwise."""
    layout = _layout(case)
    seq_len = layout["sequence_length"]
    generator = torch.Generator().manual_seed(0)
    buffer = torch.zeros(1, seq_len, 4, dtype=torch.float64)
    blocks = {}
    for name in ("text_indices", "audio_indices", "video_indices"):
        index = layout[name]
        block = torch.randn(1, index.numel(), 4, generator=generator, dtype=torch.float64)
        blocks[name] = block
        buffer = buffer.index_copy(1, index, block)
    for name, block in blocks.items():
        assert torch.equal(buffer.index_select(1, layout[name]), block)


def test_patchify_roundtrip_is_the_identity():
    """``unpatchify_video_rows`` really inverts ``patchify_video_latents``."""
    latents = torch.randn(1, 24, 7, 24, 40, generator=torch.Generator().manual_seed(3))
    rows = ops.patchify_video_latents(latents)
    assert rows.shape == (1, 7 * 12 * 20, 24 * 4)
    back = ops.unpatchify_video_rows(rows[0], 7, 24, 40)
    assert torch.equal(back, latents)


def test_unpack_audio_rows_is_channel_major():
    """Row ``ch * T + t`` of the sequence is channel ``ch``, latent ``t``."""
    num_latents, channels, dim = 5, ops.AUDIO_CHANNELS, 32
    rows = torch.arange(channels * num_latents * dim, dtype=torch.float32).reshape(
        channels * num_latents, dim)
    unpacked = ops.unpack_audio_rows(rows, num_latents)   # [ch, C, T]
    assert unpacked.shape == (channels, dim, num_latents)
    for ch in range(channels):
        for t in range(num_latents):
            assert torch.equal(unpacked[ch, :, t], rows[ch * num_latents + t])


# ---------------------------------------------------------------------------
# Noise draw order
# ---------------------------------------------------------------------------

# Geometry of the recorded digests: T = 22 @ 384x640 -> T_lat 7, 24x40 latents,
# T_aud 37.
_NOISE_VIDEO_SHAPE = (1, 24, 7, 24, 40)
_NOISE_AUDIO_LATENTS = 37
_NOISE_COND_SHAPE = (1, 24, 1, 24, 40)

# Phase 0's recorded SHA-256 prefixes (CUDA generator, seed 0).
RECORDED_CUDA_SEED0 = {
    ("t2va", "video"): "3ae7541dda186ad428ec1197b239c2ba",
    ("t2va", "audio"): "5056f21549b5d70455a5d30d5cf22fdf",
    ("fl2va1", "cond0"): "e452ab1ae534d0e6ca4123b121fe6fd6",
    ("fl2va1", "video"): "bc7493bbb6f102d4e6fc8bedd22e1367",
    ("fl2va1", "audio"): "4a56e2c2b7611a6e3c1414da68a9ca12",
    ("fl2va2", "cond1"): "5ac8b4b05dfc7428985235c68cda0175",
}


def _sha(tensor):
    return hashlib.sha256(
        tensor.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()).hexdigest()[:32]


def _draw(device, condition_shapes=()):
    generator = torch.Generator(device=device).manual_seed(0)
    return ops.draw_noise(
        generator,
        video_latent_shape=_NOISE_VIDEO_SHAPE,
        num_audio_latents=_NOISE_AUDIO_LATENTS,
        condition_shapes=condition_shapes,
        device=device,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the recorded digests use a CUDA generator")
def test_recorded_noise_digests_cuda_seed0():
    """The exact bytes Phase 0 recorded, for t2va and both fl2va shapes."""
    conditions, video, audio = _draw("cuda")
    assert conditions == []
    assert _sha(video) == RECORDED_CUDA_SEED0[("t2va", "video")]
    assert _sha(audio) == RECORDED_CUDA_SEED0[("t2va", "audio")]

    conditions, video, audio = _draw("cuda", (_NOISE_COND_SHAPE,))
    assert _sha(conditions[0]) == RECORDED_CUDA_SEED0[("fl2va1", "cond0")]
    assert _sha(video) == RECORDED_CUDA_SEED0[("fl2va1", "video")]
    assert _sha(audio) == RECORDED_CUDA_SEED0[("fl2va1", "audio")]

    conditions, _video, _audio = _draw("cuda", (_NOISE_COND_SHAPE, _NOISE_COND_SHAPE))
    assert _sha(conditions[0]) == RECORDED_CUDA_SEED0[("fl2va1", "cond0")], \
        "the FIRST condition's draw must not depend on how many follow it"
    assert _sha(conditions[1]) == RECORDED_CUDA_SEED0[("fl2va2", "cond1")]


def test_draw_order_is_conditions_then_video_then_audio():
    """Device-independent: the ORDER, checked by replaying the same generator.

    Runs on CPU too, so the contract is still pinned on a machine with no GPU
    (where the recorded CUDA digests cannot be reproduced at all).
    """
    conditions, video, audio = _draw("cpu", (_NOISE_COND_SHAPE,))
    replay = torch.Generator(device="cpu").manual_seed(0)
    expected_cond = torch.randn(_NOISE_COND_SHAPE, generator=replay, dtype=torch.float32)
    expected_video = torch.randn(_NOISE_VIDEO_SHAPE, generator=replay, dtype=torch.float32)
    expected_audio = torch.randn((_NOISE_AUDIO_LATENTS * ops.AUDIO_CHANNELS, 32),
                                 generator=replay, dtype=torch.float32)
    assert torch.equal(conditions[0], expected_cond)
    assert torch.equal(video, expected_video)
    assert torch.equal(audio, expected_audio)


def test_audio_noise_is_drawn_in_row_layout():
    """Drawn as ``[ch*T, 32]`` directly, NOT as ``[ch, 32, T]`` then permuted.

    Both produce a tensor of the same shape from the same generator; only the
    element ORDER differs, so this is another failure that would look correct
    everywhere except in the output.
    """
    _conditions, _video, audio = _draw("cpu")
    replay = torch.Generator(device="cpu").manual_seed(0)
    torch.randn(_NOISE_VIDEO_SHAPE, generator=replay, dtype=torch.float32)
    permuted = torch.randn((ops.AUDIO_CHANNELS, 32, _NOISE_AUDIO_LATENTS),
                           generator=replay, dtype=torch.float32).permute(0, 2, 1).reshape(-1, 32)
    assert audio.shape == permuted.shape
    assert not torch.equal(audio, permuted), \
        "the audio noise was drawn in [ch, C, T] and permuted, not in row layout"


def test_audio_enable_does_not_perturb_the_draw_sequence():
    """``audio_enable`` gates the DECODE, never a draw.

    ``draw_noise`` takes no such flag on purpose; this pins that it never grows
    one, because skipping the audio draw would shift every later draw and change
    the video for the same seed.
    """
    import inspect
    assert "audio_enable" not in inspect.signature(ops.draw_noise).parameters


# ---------------------------------------------------------------------------
# Visual conditioning (fl2va)
# ---------------------------------------------------------------------------

class _ScaleNoiseOnly:
    """The vendored scheduler's `scale_noise`, isolated (no weights needed)."""

    def scale_noise(self, sample, timestep, noise):
        from core.models.minimax_h3.vendor.scheduling_minimax_h3 import MiniMaxH3Scheduler
        return MiniMaxH3Scheduler.scale_noise(self, sample, timestep, noise)


def test_condition_rows_are_noised_at_the_level_they_are_pinned_at():
    """The anchor's CONTENT and the timestep the model is told agree.

    ``build_condition_rows`` mixes each anchor with its own draw at
    ``keyframe_noise_aug`` (``x_t = t*x0 + (1-t)*noise``), and
    ``build_row_timesteps`` then pins those rows at the same ``t`` for every
    step. Building a clean anchor and declaring it at 0.999 -- or noising at one
    level and declaring another -- is off-distribution for a model trained with
    noise-augmented anchors, and nothing downstream would notice.
    """
    latent = torch.randn(1, 24, 1, 4, 6, generator=torch.Generator().manual_seed(1))
    noise = torch.randn_like(latent)
    rows = ops.build_condition_rows(_ScaleNoiseOnly(), [latent], [noise])

    t = ops.VISUAL_COND_TIMESTEP
    expected = ops.patchify_video_latents(t * latent + (1.0 - t) * noise)[0]
    assert rows.shape == (1 * 2 * 3, 24 * 4)
    assert torch.allclose(rows, expected, atol=1e-6)

    # ... and the row-timestep plan pins exactly those rows at that level.
    layout = ops.build_packed_layout(3, 2, 4, 6, 5, keyframe_anchors=("first",))
    unique, index = ops.build_row_timesteps(layout, 0.5, 0.4)
    cond_rows = layout["video_indices"][:layout["num_condition_video_rows"]]
    assert layout["num_condition_video_rows"] == rows.shape[0]
    assert torch.allclose(unique[index[cond_rows]], torch.full((rows.shape[0],), t), atol=1e-6)


def test_condition_rows_lead_the_video_block_in_packed_order():
    """Anchor i occupies rows ``[i*rows_per_frame, (i+1)*rows_per_frame)``.

    Two anchors in the wrong ORDER (last before first) keeps every index, every
    count and every tag, and moves the two rotary anchor times onto each other's
    rows -- the fl2va analogue of the row-major/channel-major audio mutant.
    """
    latents = [torch.full((1, 24, 1, 4, 6), float(v)) for v in (1.0, 2.0)]
    zeros = [torch.zeros_like(l) for l in latents]
    rows = ops.build_condition_rows(_ScaleNoiseOnly(), latents, zeros)
    rows_per_frame = 2 * 3
    assert rows.shape[0] == 2 * rows_per_frame
    # `scale_noise` at t = 0.999 against zero noise scales, so compare ratios.
    assert torch.allclose(rows[:rows_per_frame], rows[:rows_per_frame][0, 0].expand_as(rows[:rows_per_frame]))
    assert float(rows[rows_per_frame, 0]) == pytest.approx(2.0 * float(rows[0, 0]))

    layout = ops.build_packed_layout(3, 2, 4, 6, 5, keyframe_anchors=("first", "last"))
    position = layout["position_ids"]
    cond = layout["video_indices"][:layout["num_condition_video_rows"]]
    first_time = position[cond[:rows_per_frame], 0]
    last_time = position[cond[rows_per_frame:], 0]
    assert torch.equal(first_time, first_time[:1].expand_as(first_time))
    assert torch.equal(last_time, last_time[:1].expand_as(last_time))
    # "first" sits at the start of the media clock (right after the text span)
    # and "last" at the far end of it.
    assert float(first_time[0]) == 3.0
    assert float(last_time[0]) > float(first_time[0])
    generated_time = position[layout["video_indices"][layout["num_condition_video_rows"]:], 0]
    assert float(generated_time.min()) == float(first_time[0])
    # The "last" anchor is placed by the reference's own pairwise-sum formula
    # (`sum(spans) - ROPE_FRAME_RESCALE`), which lands PAST the last generated
    # frame's rotary time rather than on it. Recorded as the contract rather
    # than "corrected": K0.3 compared this layout against two independent ports
    # on the fl2va cases and they agree exactly.
    assert float(last_time[0]) > float(generated_time.max())


def test_the_first_keyframe_is_stretched_and_every_later_one_is_cover_cropped():
    """The two anchors are put onto the canvas DIFFERENTLY, on purpose.

    Both independent reference implementations do this (diffusers
    ``MiniMaxH3ResizeStep``: `index == 0` plain resize, else cover-crop;
    ComfyUI ``nodes_minimax_h3``: `"disabled"` vs `"center"`), because the first
    keyframe is the geometry anchor while the follower has no say in the canvas.
    Stretching the follower hands the model a distorted anchor it is then pinned
    to for the whole loop, and nothing downstream can notice.

    The arithmetic is checked against MiniMax's own, not against
    ``VaeImageProcessor(resize_mode="crop")``: the two differ by one pixel on
    many aspect ratios.
    """
    from PIL import Image as PILImage
    from core.pipeline_backends.minimax_h3 import MiniMaxH3Mixin

    fit = MiniMaxH3Mixin._minimax_h3_fit_keyframe
    width, height = 640, 384

    # A SQUARE follower on a 5:3 canvas: a stretch would keep the full image and
    # squash it; the cover-crop scales to the width and cuts the top/bottom.
    square = PILImage.new("RGB", (512, 512))
    for y in range(512):                      # vertical ramp, so a crop is visible
        for x in range(0, 512, 64):
            square.putpixel((x, y), (y // 2, y // 2, y // 2))

    stretched = fit(square, width, height, 0)
    cropped = fit(square, width, height, 1)
    assert stretched.size == (width, height)
    assert cropped.size == (width, height)

    # MiniMax's own follower arithmetic, recomputed here.
    scale = max(width / 512, height / 512)                 # 1.25
    resized_size = (max(width, round(512 * scale)), max(height, round(512 * scale)))
    assert resized_size == (640, 640)
    left, top = max(0, (resized_size[0] - width) // 2), max(0, (resized_size[1] - height) // 2)
    assert (left, top) == (0, 128)
    expected = square.resize(resized_size, PILImage.LANCZOS).crop(
        (left, top, left + width, top + height))
    assert list(cropped.getdata()) == list(expected.getdata())

    # ... and the two paths really differ, so removing the crop fails this test
    # rather than silently passing on a same-aspect image.
    assert list(cropped.getdata()) != list(stretched.getdata())
    # The crop keeps the MIDDLE of the vertical ramp; the stretch keeps its ends.
    assert cropped.getpixel((0, 0))[0] > stretched.getpixel((0, 0))[0]

    # A same-aspect follower is untouched by either path (identical geometry),
    # and an already-exact-size frame is returned as-is.
    exact = PILImage.new("RGB", (width, height), (7, 7, 7))
    assert fit(exact, width, height, 1) is not None
    assert list(fit(exact, width, height, 1).getdata()) == list(exact.getdata())


class _StubPosteriorVae(torch.nn.Module):
    """A VAE whose encode returns a posterior with a known mean and std."""

    def __init__(self, std=3.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))   # gives it a dtype
        self.std = std

    def encode(self, x, return_dict=True):
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        from diffusers.models.modeling_outputs import AutoencoderKLOutput
        b, _c, t, h, w = x.shape
        mean = torch.full((b, 24, t, h, w), 2.0)
        logvar = torch.full_like(mean, 2.0 * float(np.log(self.std)))
        dist = DiagonalGaussianDistribution(torch.cat([mean, logvar], dim=1))
        return AutoencoderKLOutput(latent_dist=dist) if return_dict else (dist,)


def test_condition_encode_samples_under_a_fixed_seed_and_leaves_the_request_rng_alone():
    """The released recipe: SAMPLE the posterior, under its own fixed seed.

    Three separate contracts, each of which a "read the mode" shortcut breaks:

    * it is a sample, not the mean (the mean would be a different conditioning
      latent from the one MiniMax-H3's own implementations produce);
    * the seed is FIXED (42, `KEYFRAME_ENCODE_SEED`), so the encode is
      reproducible across runs of the same request;
    * the generator is a FRESH one, so the request generator's draw sequence --
      the contract K0.6 recorded -- is not perturbed by conditioning.
    """
    vae = _StubPosteriorVae()
    image = (np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3))
    kwargs = dict(latents_mean=[0.0] * 24, latents_std=[1.0] * 24,
                  pixel_mean=(0.5, 0.5, 0.5), pixel_std=(0.5, 0.5, 0.5), device="cpu")

    first = ops.encode_condition_images(vae, [image], **kwargs)[0]
    again = ops.encode_condition_images(vae, [image], **kwargs)[0]
    assert torch.equal(first, again), "the conditioning encode is not reproducible"
    assert not torch.allclose(first, torch.full_like(first, 2.0)), \
        "the posterior was read at its mean instead of sampled"
    # Exactly the upstream recipe: mean + std * N(0, 1) drawn from seed 42, then
    # rounded through fp16.
    expected = (2.0 + vae.std * torch.randn(
        first.shape, generator=torch.Generator().manual_seed(ops.KEYFRAME_ENCODE_SEED))
    ).to(torch.float16).float()
    assert torch.equal(first, expected)

    # The REQUEST generator is untouched: the draws after an encode are the
    # draws that would have happened without one.
    generator = torch.Generator().manual_seed(0)
    ops.encode_condition_images(vae, [image, image], **kwargs)
    after = torch.randn(8, generator=generator)
    assert torch.equal(after, torch.randn(8, generator=torch.Generator().manual_seed(0)))


def test_build_condition_rows_requires_one_draw_per_condition():
    """A missing draw would silently shift the whole seed sequence."""
    latent = torch.zeros(1, 24, 1, 4, 6)
    with pytest.raises(ValueError, match="own noise draw"):
        ops.build_condition_rows(_ScaleNoiseOnly(), [latent, latent], [torch.zeros_like(latent)])
    assert ops.build_condition_rows(_ScaleNoiseOnly(), [], []).numel() == 0


# ---------------------------------------------------------------------------
# Frame geometry
# ---------------------------------------------------------------------------

def test_latent_frame_and_audio_latent_geometry():
    """The two closed forms, on the values Phase 0 measured."""
    from core.models.minimax_h3.loader import minimax_h3_latent_frames

    assert [minimax_h3_latent_frames(t) for t in (22, 39, 56, 73, 90, 107, 124, 141, 192)] == \
        [7, 12, 17, 22, 27, 32, 37, 42, 57]
    assert [ops.audio_latent_frames(t) for t in (22, 39, 124, 192)] == [37, 65, 207, 320]


def test_temporal_spec_matches_the_measured_grid():
    """Route validation's spec agrees with the VAE's own arithmetic."""
    from core.models.components.wiring import MINIMAX_H3_TEMPORAL as spec
    from core.models.minimax_h3.loader import minimax_h3_latent_frames

    for num_frames in (22, 39, 124, 141, 345):
        assert spec.is_valid_length(num_frames)
        assert spec.latent_frames(num_frames) == minimax_h3_latent_frames(num_frames)
    # 5 is ON the 17n+5 grid and is NOT decodable: 2 latent frames, and the
    # decoder needs 7. The spec must refuse it.
    assert not spec.is_valid_length(5)
    assert not spec.is_valid_length(21)
    # Snapping rounds UP to the next encodable length, matching MiniMax-H3's own
    # `align_num_frames`; an over-long request clamps to the largest length in
    # the production range, and a too-short one to the floor.
    assert spec.snap_length(130) == 141
    assert spec.snap_length(125) == 141
    assert spec.snap_length(141) == 141
    assert spec.snap_length(400) == 345
    assert spec.snap_length(30) == 124
    # The smoke gate lowers the floor to the VAE's, and rounding is still up.
    assert spec.snap_length(30, smoke=True) == 39
    assert spec.snap_length(2, smoke=True) == 22


def test_suggested_lengths_are_offerable_clip_lengths():
    """A length can be VALID without being worth suggesting."""
    from core.models.components.wiring import LTX2_TEMPORAL, MINIMAX_H3_TEMPORAL

    # LTX-2.3 accepts a 1-frame clip (unchanged) but must not OFFER one.
    assert LTX2_TEMPORAL.is_valid_length(1)
    assert LTX2_TEMPORAL.suggested_lengths(4) == [9, 17, 25, 33]
    assert MINIMAX_H3_TEMPORAL.suggested_lengths(3) == [124, 141, 158]


# ---------------------------------------------------------------------------
# The preview hook
# ---------------------------------------------------------------------------

class _StubScheduler:
    """Just enough scheduler to drive the loop, with an identifiable step."""

    def __init__(self, timesteps):
        self.timesteps = torch.tensor(timesteps, dtype=torch.float32)

    def set_shift(self, shift):
        self.shift = shift

    def set_timesteps(self, steps, device=None):
        pass

    def set_begin_index(self, index):
        pass

    def step(self, velocity, timestep, sample, return_dict=False):
        # Deliberately NOT `sample + sigma * velocity`, so a preview computed
        # from the post-step tensor cannot accidentally look right.
        return (sample - 0.25 * velocity,)


def test_step_callback_gets_x0_from_x_t_as_unpatchified_latents():
    """The preview estimate is ``x_t + sigma_t * v_t``, in LATENT space.

    Two failure modes this pins, both of which produce a plausible-looking
    preview: reading ``x_{t+1}`` (the tensor the scheduler just wrote) instead
    of ``x_t`` (the one the velocity was predicted from), and handing the
    callback PACKED ROWS, which no preview decoder in this repo can read.
    """
    latent_frames, latent_height, latent_width, channels = 2, 4, 4, 24
    layout = ops.build_packed_layout(3, latent_frames, latent_height, latent_width, 5)
    num_video_rows = layout["video_indices"].numel()
    num_audio_rows = layout["audio_indices"].numel()
    row_width = channels * 4

    torch.manual_seed(0)
    video_rows = torch.randn(num_video_rows, row_width)
    audio_rows = torch.randn(num_audio_rows, 32)
    video_rows_before = video_rows.clone()
    velocity = torch.full((1, num_video_rows, row_width), 2.0)

    def transformer(**kwargs):
        return velocity, torch.zeros(1, num_audio_rows, 32)

    seen = []
    ops.denoise(
        transformer, _StubScheduler([0.75]), _StubScheduler([0.75]),
        prompt_embeds=torch.zeros(1, 3, 8), layout=layout,
        video_rows=video_rows, audio_rows=audio_rows, num_inference_steps=1,
        device="cpu", step_callback=lambda *a: seen.append(a),
        preview_latent_shape=(latent_frames, latent_height, latent_width),
        latent_channels=channels,
    )

    assert len(seen) == 1
    index, total, latents, extra, pred_x0 = seen[0]
    assert (index, total, extra) == (0, 1, None)
    # Latents, not rows: [1, C, T, H, W].
    assert latents.shape == (1, channels, latent_frames, latent_height, latent_width)
    assert pred_x0.shape == latents.shape
    # x0 = x_t + sigma * v, from the PRE-step rows (sigma = 1 - t = 0.25).
    expected = ops.unpatchify_video_rows(video_rows_before + 0.25 * velocity[0],
                                         latent_frames, latent_height, latent_width,
                                         latent_channels=channels)
    assert torch.allclose(pred_x0, expected, atol=1e-6)
    # ... and the "latents" argument is the post-step state, also unpatchified.
    assert torch.allclose(
        latents,
        ops.unpatchify_video_rows(video_rows, latent_frames, latent_height, latent_width,
                                  latent_channels=channels),
        atol=1e-6)


def test_step_callback_without_preview_geometry_is_refused():
    """A preview cannot be built from rows alone; the loop says so up front."""
    layout = ops.build_packed_layout(3, 2, 4, 4, 5)
    with pytest.raises(ValueError, match="preview_latent_shape"):
        ops.denoise(
            lambda **kwargs: None, _StubScheduler([0.75]), _StubScheduler([0.75]),
            prompt_embeds=torch.zeros(1, 3, 8), layout=layout,
            video_rows=torch.zeros(layout["video_indices"].numel(), 96),
            audio_rows=torch.zeros(layout["audio_indices"].numel(), 32),
            num_inference_steps=1, device="cpu", step_callback=lambda *a: None)


# ---------------------------------------------------------------------------
# ref2va: where the reference blocks sit in the packed order is a contract too
# ---------------------------------------------------------------------------
# `build_ref2va_packed_layout` is a port of the diffusers `minimax-h3` block
# `MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence`. It was
# validated against a SECOND, independent port -- ComfyUI's `PackedLayout` with
# its `refs` branch -- on the seven configurations below: identical sequence
# length, identical ORDERED index tensors, identical conditioning row counts,
# and a float64 position grid agreeing to <= 1e-4 after our fp32 cast. The
# numbers recorded here are that comparison's, so the shipped function is pinned
# to the cross-check rather than to a re-derivation of itself.
#
# The layout is `[text | reference blocks | target audio | target video]`, one
# block per reference IN REQUEST ORDER, and a video reference's soundtrack packs
# immediately BEFORE its own video rows. What that buys is the invariant the
# denoise loop already relies on: every reference row precedes every generated
# row of its own modality, so `video_indices` and `audio_indices` both lead with
# their conditioning rows and the loop pins the anchors by never writing the
# first `num_condition_*_rows` entries.

# (label, text tokens, target, refs, S, cond_video_rows, cond_audio_rows) where
# a ref is ("image", (T_lat, H_lat, W_lat)) | ("video", shape, audio_latents) |
# ("audio", audio_latents).
_REF_TARGET = (7, 24, 40, 37)   # T=22 @ 384x640
_REF_BIG = (37, 48, 84, 207)    # T=124 @ 768x1344

REF_CASES = [
    ("1 image ref (2048 short edge, 4:3)", 40, _REF_TARGET,
     [("image", (1, 128, 96))], 4866, 3072, 0),
    ("3 image refs, mixed shapes", 120, _REF_TARGET,
     [("image", (1, 128, 96)), ("image", (1, 96, 128)), ("image", (1, 64, 64))], 9042, 7168, 0),
    ("1 video ref, no soundtrack", 200, _REF_TARGET,
     [("video", (7, 34, 60), 0)], 5524, 3570, 0),
    ("1 video ref WITH soundtrack", 200, _REF_TARGET,
     [("video", (7, 34, 60), 37)], 5598, 3570, 74),
    ("image + video+audio + standalone audio (T=124)", 512, _REF_BIG,
     [("image", (1, 128, 96)), ("video", (12, 48, 84), 65), ("audio", 40)], 53600, 15168, 210),
    ("9 image refs (the limit)", 64, _REF_TARGET,
     [("image", (1, 64, 64))] * 9, 11034, 9216, 0),
    ("interleaved: image, audio, video+audio", 300, _REF_TARGET,
     [("image", (1, 96, 96)), ("audio", 20), ("video", (7, 34, 60), 37)], 8042, 5874, 114),
]


def _ref_layout(case):
    _label, text_len, target, refs = case[:4]
    blocks, shapes, audio_rows = [], [], []
    for ref in refs:
        if ref[0] == "image":
            blocks.append(("image", False))
            shapes.append(ref[1])
        elif ref[0] == "video":
            blocks.append(("video", ref[2] > 0))
            shapes.append(ref[1])
            if ref[2] > 0:
                audio_rows.append(ref[2] * 2)
        else:
            blocks.append(("audio", True))
            audio_rows.append(ref[1] * 2)
    latent_t, latent_h, latent_w, audio_t = target
    return ops.build_ref2va_packed_layout(
        [ops.TEXT_TAG] * text_len, blocks, shapes, audio_rows,
        latent_t, latent_h, latent_w, audio_t)


@pytest.mark.parametrize("case", REF_CASES, ids=[c[0] for c in REF_CASES])
def test_ref2va_recorded_row_counts(case):
    """Sequence length and the two conditioning row counts, as cross-checked."""
    _label, text_len, _target, _refs, seq_len, cond_video, cond_audio = case
    layout = _ref_layout(case)
    assert layout["sequence_length"] == seq_len
    assert layout["num_condition_video_rows"] == cond_video
    assert layout["num_condition_audio_rows"] == cond_audio
    assert layout["text_indices"].numel() == text_len


@pytest.mark.parametrize("case", REF_CASES, ids=[c[0] for c in REF_CASES])
def test_ref2va_indices_are_ordered_disjoint_and_cover(case):
    """Ascending within each modality, disjoint across them, covering [0, S)."""
    layout = _ref_layout(case)
    seq_len = layout["sequence_length"]
    blocks = [layout["text_indices"], layout["audio_indices"], layout["video_indices"]]
    for name, block in zip(("text", "audio", "video"), blocks):
        assert torch.equal(block, block.sort().values), f"{name} indices are not ascending"
    combined = torch.cat(blocks).sort().values
    assert torch.equal(combined, torch.arange(seq_len))


@pytest.mark.parametrize("case", REF_CASES, ids=[c[0] for c in REF_CASES])
def test_ref2va_reference_rows_lead_and_precede_the_generated_rows(case):
    """The invariant the denoise loop pins the anchors with.

    Every reference row sits between the text span and the generated rows, so
    the leading `num_condition_*_rows` entries of each index block are exactly
    the conditioning ones -- which is what makes `rows[n_cond:] = step(...)`
    protect them without re-imposing anything.
    """
    _label, text_len, target, _refs, seq_len, cond_video, cond_audio = case
    layout = _ref_layout(case)
    latent_t, latent_h, latent_w, audio_t = target
    generated_video = latent_t * (latent_h // 2) * (latent_w // 2)
    generated_audio = audio_t * ops.AUDIO_CHANNELS

    video_indices, audio_indices = layout["video_indices"], layout["audio_indices"]
    assert video_indices.numel() == cond_video + generated_video
    assert audio_indices.numel() == cond_audio + generated_audio
    if cond_video:
        assert int(video_indices[cond_video - 1]) < int(video_indices[cond_video])
        assert int(video_indices[:cond_video].max()) < int(video_indices[cond_video:].min())
        assert int(video_indices.min()) >= text_len
    if cond_audio:
        assert int(audio_indices[:cond_audio].max()) < int(audio_indices[cond_audio:].min())
    # The generated rows are the LAST two contiguous blocks: target audio, then
    # target video.
    assert torch.equal(video_indices[cond_video:], torch.arange(seq_len - generated_video, seq_len))
    assert torch.equal(
        audio_indices[cond_audio:],
        torch.arange(seq_len - generated_video - generated_audio, seq_len - generated_video))


@pytest.mark.parametrize("case", REF_CASES, ids=[c[0] for c in REF_CASES])
def test_ref2va_modality_tags(case):
    """A reference's rows carry its modality's tag, which AdaLN keys off."""
    layout = _ref_layout(case)
    tags = layout["token_tags"]
    assert (tags[layout["text_indices"]] == ops.TEXT_TAG).all()
    assert (tags[layout["audio_indices"]] == ops.AUDIO_TAG).all()
    assert (tags[layout["video_indices"]] == ops.VIDEO_TAG).all()


def test_ref2va_vision_block_rows_stay_tagged_video():
    """A reference's vision block lives in the TEXT span but is tagged video.

    `build_ref2va_presentation` emits `<|image_pad|>` runs tagged 0 (video); the
    layout must carry those tags through rather than overwriting the whole text
    span with the text tag.
    """
    text_tags = [ops.TEXT_TAG] * 5 + [ops.VIDEO_TAG] * 7 + [ops.TEXT_TAG] * 3
    layout = ops.build_ref2va_packed_layout(
        text_tags, [("image", False)], [(1, 8, 8)], [], 7, 24, 40, 37)
    assert torch.equal(layout["token_tags"][:15], torch.tensor(text_tags))


def test_ref2va_reference_order_is_semantic():
    """Reordering the references is a different request, not the same one."""
    tags = [ops.TEXT_TAG] * 40
    first = ops.build_ref2va_packed_layout(
        tags, [("image", False), ("audio", True)], [(1, 64, 64)], [40], 7, 24, 40, 37)
    second = ops.build_ref2va_packed_layout(
        tags, [("audio", True), ("image", False)], [(1, 64, 64)], [40], 7, 24, 40, 37)
    assert first["sequence_length"] == second["sequence_length"]
    # Same rows, different places: the index tables and the rotary clock differ.
    assert not torch.equal(first["video_indices"], second["video_indices"])
    assert not torch.equal(first["position_ids"], second["position_ids"])


def test_ref2va_video_soundtrack_packs_before_its_own_video_rows():
    """A video reference's audio rows lead its video rows and share their clock."""
    tags = [ops.TEXT_TAG] * 10
    layout = ops.build_ref2va_packed_layout(
        tags, [("video", True)], [(7, 34, 60)], [74], 7, 24, 40, 37)
    assert int(layout["audio_indices"][0]) == 10
    assert int(layout["video_indices"][0]) == 10 + 74
    position_ids = layout["position_ids"]
    assert float(position_ids[10, 0]) == 10.0
    assert float(position_ids[10 + 74, 0]) == 10.0


def test_ref2va_image_reference_advances_the_clock_by_one():
    """An image takes a single integer rotary slot, not a latent frame's 5/3."""
    tags = [ops.TEXT_TAG] * 4
    layout = ops.build_ref2va_packed_layout(
        tags, [("image", False), ("image", False)], [(1, 8, 8), (1, 8, 8)], [], 7, 24, 40, 37)
    position_ids = layout["position_ids"]
    rows_per_image = (8 // 2) * (8 // 2)
    assert float(position_ids[4, 0]) == 4.0
    assert float(position_ids[4 + rows_per_image, 0]) == 5.0


def test_ref2va_layout_matches_the_t2va_builder_with_no_references():
    """No references == the t2va layout, so the two builders cannot drift."""
    plain = ops.build_packed_layout(12, 7, 24, 40, 37)
    empty = ops.build_ref2va_packed_layout([ops.TEXT_TAG] * 12, [], [], [], 7, 24, 40, 37)
    assert empty["sequence_length"] == plain["sequence_length"]
    for key in ("video_indices", "audio_indices", "text_indices", "token_tags", "position_ids"):
        assert torch.equal(empty[key], plain[key]), key
