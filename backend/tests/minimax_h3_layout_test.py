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
    assert spec.snap_length(130) == 124
    assert spec.snap_length(400) == 345
