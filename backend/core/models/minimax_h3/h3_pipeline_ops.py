"""Sampling operations for MiniMax-H3 — the denoise loop this repo owns.

MiniMax-H3 ships as a diffusers **Modular** pipeline and nothing else: there is
no stock ``DiffusionPipeline`` to drive (its ``model_index.json`` declares only
``MiniMaxH3ModularPipeline``). SushiUI needs per-step progress, cancellation,
latent preview, its own strictly-sequential offload sequencing and a block-loop
wrapper for block swap — none of which the Modular stack's
``ComponentsManager`` auto-offload would tolerate. So, following the Anima
precedent (``core/models/anima/anima_pipeline_ops.py``), the model classes are
vendored and the loop lives here.

WHAT THIS MODULE OWES ITS SOURCES
---------------------------------
Every contract below is either a verbatim port of the diffusers ``minimax-h3``
modular blocks (Apache-2.0; ``before_denoise.py`` / ``denoise.py`` /
``decoders.py`` / ``encoders.py``) or was cross-checked in the K0 conformance
suite against a second, independently written implementation of the same
layout. The two places a re-derivation is easy to get subtly wrong are called
out inline:

1. **Packed-sequence assembly** (``build_packed_layout``). The order of the
   layout is ``[text | keyframe conditions | audio | video]``, the audio rows
   are **channel-major** (all of channel 0's latents, then all of channel 1's),
   and the row ORDER matters as much as the row set: a row-major audio block
   holds the same index SET and is invisible to set equality, so K0.3's
   recorded index tables — reproduced by ``minimax_h3_layout_test.py`` — are
   compared as ordered tensors.
2. **Noise draw order** (``draw_noise``). One generator, three kinds of draw,
   in this order: one per visual condition (in packed order, each at its own
   latent shape), then the video noise as a 5-D **latent** tensor, then the
   audio noise **directly in row layout**. Drawing the audio as ``[ch, C, T]``
   and permuting it into rows gives different numbers from the same seed.
   ``audio_enable=False`` skips the audio DECODE, never a draw.
3. **The velocity convention is the opposite of the usual flow-matching one**:
   ``x0 = x_t + sigma * v`` (note the ``+``). The vendored scheduler implements
   it; nothing here recomputes it.
4. **The two schedules.** Video runs at ``shift = 12.0``, audio at
   ``shift = 3.0``, and both are stepped once per loop iteration — the video
   rows on the video grid, the audio rows on the audio grid. A single-schedule
   sampler has to integrate the *video* grid alone and scale the audio velocity
   by ``d(sigma_a)/d(sigma_v)``; K0.4 verified that slope against fp64 autograd
   (3.05e-16) and the two formulations agree to first order. This module follows the diffusers
   reference (two schedulers), because that is the implementation the vendored
   transformer and scheduler came from and it is exact on each stream's own
   grid rather than first-order-accurate on the other's.
5. **Conditioning rows are pinned**, at ``t = max(t_video, 0.999)`` for visual
   conditioning and ``t = 1.0`` for audio references, for every step. The loop
   only ever writes the GENERATED rows, so the anchors ride through unchanged.
   The anchors themselves are built at the SAME level they are pinned at:
   ``encode_condition_images`` produces the clean latent (posterior SAMPLED
   under the fixed ``KEYFRAME_ENCODE_SEED``, then rounded through fp16) and
   ``build_condition_rows`` mixes in that condition's own noise draw through the
   scheduler's ``scale_noise`` (``x_t = t*x0 + (1-t)*noise``) at
   ``keyframe_noise_aug`` — the released model was trained with slightly noised
   anchors, so feeding it an exactly-clean one is off-distribution.

THE TEXT ENCODER IS STREAMED, AND THE SHAPE OF THAT IS LOAD-BEARING
-------------------------------------------------------------------
``encode_prompt`` builds each decoder layer's fp32 GPU parameters from the
memory-mapped CPU tensors and calls the layer through
``torch.func.functional_call``, never writing anything back. The obvious
alternative — ``layer.to(cuda, fp32)`` then ``layer.to("cpu", bf16)`` — detaches
every parameter from the file mapping, so the mmap pages AND an anonymous copy
of the 25 B-parameter decoder are both resident: **73.08 GB peak RSS, pagefile
growth and 46 s/prompt, against 49.82 GB flat and 13.5 s/prompt** for the
functional shape (K0.7, on this box). It is not a micro-optimisation; it is the
difference between running and thrashing a 93.6 GB machine.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from core.inference.cancellation import raise_if_cancelled


# --------------------------------------------------------------------------
# Checkpoint contracts. These values index the transformer's AdaLN table and
# its rotary grid, so they are not preferences.
# --------------------------------------------------------------------------
VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2

# The `t` a visual conditioning anchor is held at (the released model was
# trained with slightly noised anchors, so exactly 1.0 is off-distribution) and
# the `t` an audio reference is held at.
VISUAL_COND_TIMESTEP = 0.999
AUDIO_COND_TIMESTEP = 1.0

# The seed the visual-conditioning posterior is SAMPLED under. Fixed by the
# released implementation (diffusers `components.keyframe_encode_seed = 42`) and
# deliberately independent of the request seed: the conditioning encode is
# reproducible, and it consumes none of the request generator's draws, so the
# recorded draw order (K0.6) is untouched.
KEYFRAME_ENCODE_SEED = 42

# Sigma shifts of the two schedules.
SHIFT_VIDEO = 12.0
SHIFT_AUDIO = 3.0

# Rotary geometry of the packed sequence.
ROPE_SPATIAL_SCALE = 32.0
ROPE_FRAME_RESCALE = 5.0 / 3.0
ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)

# Stereo, carried channel-major in the packed sequence and as two batch items at
# the (mono) audio VAE boundary.
AUDIO_CHANNELS = 2

# Which Qwen3-VL hidden state conditions the transformer. The released text
# encoder file is truncated to exactly this many decoder layers and its declared
# output is the UNNORMALISED hidden state after the last one, which is why the
# loader replaces the final norm with an Identity.
TEXT_ENCODER_LAYER = 50


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def audio_latent_frames(num_frames: int, fps: float = 24.0,
                        latents_per_second: float = 40.0) -> int:
    """Audio latents per channel covering ``num_frames`` video frames.

    ``round(T / 24 * 40)``, exact on every frame count MiniMax-H3 generates
    (MEASURED: 22 -> 37, 39 -> 65, 124 -> 207, 192 -> 320).
    """
    return int(round(num_frames / fps * latents_per_second))


# --------------------------------------------------------------------------
# Packing / unpacking
# --------------------------------------------------------------------------

def patchify_video_latents(latents: torch.Tensor,
                           patch_size: Tuple[int, int, int] = (1, 2, 2)) -> torch.Tensor:
    """``[B, C, T, H, W]`` -> ``[B, T'*H'*W', C*pt*ph*pw]``, frame-major.

    Verbatim shape math from the diffusers block of the same name; the
    equivalent single-einsum form is ``"nctrhpwq->nthwcrpq"``.
    """
    pt, ph, pw = patch_size
    b, c, t, h, w = latents.shape
    x = latents.reshape(b, c, t // pt, pt, h // ph, ph, w // pw, pw)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(b, -1, c * pt * ph * pw).contiguous()


def unpatchify_video_rows(rows: torch.Tensor, num_latent_frames: int, latent_height: int,
                          latent_width: int, latent_channels: int = 24,
                          patch_size: Tuple[int, int, int] = (1, 2, 2)) -> torch.Tensor:
    """The exact inverse of :func:`patchify_video_latents`."""
    pt, ph, pw = patch_size
    x = rows.reshape(-1, num_latent_frames // pt, latent_height // ph, latent_width // pw,
                     latent_channels, pt, ph, pw)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return x.reshape(-1, latent_channels, num_latent_frames, latent_height, latent_width).contiguous()


def unpack_audio_rows(rows: torch.Tensor, num_audio_latents: int,
                      channels: int = AUDIO_CHANNELS) -> torch.Tensor:
    """Channel-major rows ``[ch*T, C]`` -> ``[ch, C, T]`` for the mono audio VAE."""
    return rows.reshape(channels, num_audio_latents, rows.shape[-1]).permute(0, 2, 1).contiguous()


# --------------------------------------------------------------------------
# Packed-sequence layout
# --------------------------------------------------------------------------

def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    # `np.linspace(..., endpoint=False)` computes `(arange*ratio)/n + left`; the
    # hand-written form `arange*(ratio/n) + left` is the same value with a
    # different float64 association, <= 1 ulp apart and unobservable through the
    # model (the rope casts to float32 and the two grids are then bitwise equal,
    # K0.3 supplementary). The diffusers form is used because the vendored
    # transformer is the diffusers one -- do not "fix" it to the other form.
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def _frame_position_grid(latent_height: int, latent_width: int, patch_h: int,
                         patch_w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    sqrt_area = float(np.sqrt(latent_height * latent_width))
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([g.reshape(-1) for g in grids], dim=-1), width_grid


def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % len(ROPE_FRAMES_PER_LATENT)]
         for i in range(num_latent_frames)],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _fill_audio_positions(
    position_ids: torch.Tensor,
    rows: slice,
    num_audio_latents: int,
    rotary_time: float,
    width_grid: torch.Tensor,
) -> None:
    """Place one CHANNEL-MAJOR audio block on the rotary grid.

    Audio rows share the video's clock (one unit per latent: 40 latents/s ==
    24 fps * 5/3), carry no height coordinate, and are pinned to the two
    extremes of the width grid of THEIR OWN block -- the target grid for the
    generated rows and for a standalone audio reference, the video's own grid
    for a video reference's soundtrack.
    """
    time = rotary_time + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[rows, 0] = time.repeat(AUDIO_CHANNELS)
    position_ids[rows, 2] = torch.cat([
        torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
        torch.full((num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
    ])


def _clip_pixel_frames(num_latent_frames: int) -> int:
    """How many PIXEL frames ``num_latent_frames`` latent frames cover.

    The video VAE's ``(1, 4, 4, 4, 4)`` temporal chunking, summed. MEASURED
    against the loader's own ``minimax_h3_latent_frames`` inverse: 7 -> 22,
    12 -> 39, 37 -> 124.
    """
    return sum(ROPE_FRAMES_PER_LATENT[i % len(ROPE_FRAMES_PER_LATENT)]
               for i in range(num_latent_frames))


def is_frame_index_anchor(anchor: "int | str") -> bool:
    """True when ``anchor`` is an integer PIXEL-frame index rather than a string.

    THE ONE PREDICATE, because two of them drifted: this module accepts
    ``np.integer`` (a `-1` sentinel resolved through numpy, a frame index read
    out of an array) while the backend's ``_minimax_h3_fit_keyframe`` tested
    ``isinstance(anchor, int)`` alone, so a ``np.int64(0)`` would have been
    PLACED at frame 0 and then cover-cropped instead of stretched -- a silent
    disagreement about which anchor sets the canvas geometry. Both callers now
    ask here.

    ``bool`` is excluded explicitly: it is an ``int`` subclass, and ``True``
    would otherwise mean frame 1.
    """
    return isinstance(anchor, (int, np.integer)) and not isinstance(anchor, bool)


def _anchor_rotary_time(anchor: "int | str", num_text_tokens: int,
                        num_latent_frames: int) -> float:
    """Where one keyframe anchor sits on the packed sequence's time axis.

    That axis is literally PIXEL-FRAME time: ``t(f) = num_text_tokens +
    ROPE_FRAME_RESCALE * f``, measured exact for T in {5, 22, 39, 124, 192, 345}
    against the video latent grid, so ``"first"`` and ``"last"`` are two
    evaluations of one frame-index function and an integer ``f`` addresses any
    frame of the clip. There is no grid to snap an anchor to.

    The two STRING branches are kept verbatim rather than expressed through the
    integer formula, and this is load-bearing rather than conservative:
    ``"last"`` is numpy's PAIRWISE sum of the per-latent spans, and
    ``(5/3)*(T-1)`` differs from it in the last float64 ulp on most clip lengths
    (51.00000000000001 vs 51.0 at 7 latent frames). The float32 ``position_ids``
    agree after the cast on every geometry measured, so re-routing the strings
    would be invisible in a layout digest and visible only in the float64
    arithmetic. ``minimax_h3_layout_test`` therefore asserts that difference
    directly, in the test whose name says so. Do not merge these branches.
    """
    if isinstance(anchor, str):
        if anchor == "first":
            return float(num_text_tokens)
        if anchor == "last":
            # numpy's PAIRWISE summation, because that is how the reference
            # computes this anchor; a sequential sum differs in the last ulp
            # from 16 latent frames onwards.
            spans = np.ones(num_latent_frames, dtype=np.float64) * ROPE_FRAME_RESCALE
            for offset in range(len(ROPE_FRAMES_PER_LATENT)):
                spans[offset::len(ROPE_FRAMES_PER_LATENT)] *= ROPE_FRAMES_PER_LATENT[offset]
            return float(num_text_tokens) + float(spans.sum()) - ROPE_FRAME_RESCALE
        raise ValueError(
            f"A keyframe anchor must be 'first', 'last' or an integer pixel-frame "
            f"index, got {anchor!r}.")

    if not is_frame_index_anchor(anchor):
        raise ValueError(
            f"A keyframe anchor must be 'first', 'last' or an integer pixel-frame "
            f"index, got {anchor!r}.")

    frame = int(anchor)
    num_pixel_frames = _clip_pixel_frames(num_latent_frames)
    if frame < 0:
        raise ValueError(
            f"A keyframe anchor's frame index must be >= 0, got {frame}. A request's "
            f"-1 ('the last frame') is a SENTINEL the caller resolves -- to 'last', or "
            f"to the clip's own last index -- before the layout is built; placed "
            f"literally it would sit one frame before the clip's origin.")
    if frame >= num_pixel_frames:
        raise ValueError(
            f"A keyframe anchor at frame {frame} is outside this clip: "
            f"{num_latent_frames} latent frames cover {num_pixel_frames} pixel frame(s), "
            f"so the last addressable index is {num_pixel_frames - 1}.")
    return float(num_text_tokens) + ROPE_FRAME_RESCALE * float(frame)


def _validated_pinned_frames(
    pinned_video_frames: Sequence[int],
    num_latent_frames: int,
    keyframe_anchors: Sequence["int | str"],
) -> Tuple[int, ...]:
    """The pinned LATENT frames, ascending, so the layout is a function of the set."""
    if len(keyframe_anchors):
        raise ValueError(
            "MiniMax-H3 cannot combine keyframe anchors with pinned video frames: an anchor "
            "reserves its own conditioning rows ahead of the clip, and the pin re-uses the same "
            "prefix count for rows of the clip itself. Pass one or the other.")
    frames: List[int] = []
    for frame in pinned_video_frames:
        if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)):
            raise ValueError(
                f"A pinned video frame must be an integer LATENT-frame index, got {frame!r}. "
                f"Pixel frames are expanded to latent-frame groups by the caller.")
        frame = int(frame)
        if not 0 <= frame < num_latent_frames:
            raise ValueError(
                f"A pinned video frame at latent frame {frame} is outside this clip: it has "
                f"{num_latent_frames} latent frame(s), so the last addressable index is "
                f"{num_latent_frames - 1}.")
        frames.append(frame)
    if len(set(frames)) != len(frames):
        raise ValueError(f"Pinned video frames must be distinct, got {list(pinned_video_frames)!r}.")
    return tuple(sorted(frames))


def _validated_pinned_video_row_indices(
    pinned_video_row_indices: Sequence[int],
    num_video_rows: int,
    *,
    pinned_video_frames: Sequence[int],
    keyframe_anchors: Sequence["int | str"],
) -> Tuple[int, ...]:
    """Validate and canonicalize arbitrary frame-major video row pins."""
    if len(pinned_video_frames) or len(keyframe_anchors):
        raise ValueError(
            "MiniMax-H3 cannot combine pinned video row indices with pinned video frames "
            "or keyframe anchors: pass one pinning scheme only.")

    rows: List[int] = []
    for row in pinned_video_row_indices:
        if isinstance(row, bool) or not isinstance(row, (int, np.integer)):
            raise ValueError(
                f"A pinned video row must be an integer frame-major row index, got {row!r}.")
        row = int(row)
        if not 0 <= row < num_video_rows:
            raise ValueError(
                f"A pinned video row at index {row} is outside this clip: it has "
                f"{num_video_rows} video row(s), so the last addressable index is "
                f"{num_video_rows - 1}.")
        rows.append(row)
    if len(set(rows)) != len(rows):
        raise ValueError(
            f"Pinned video row indices must be distinct, got "
            f"{list(pinned_video_row_indices)!r}.")
    return tuple(sorted(rows))


def _validated_pinned_audio_latents(
    pinned_audio_latents: Sequence[int],
    num_audio_latents: int,
) -> Tuple[int, ...]:
    """The pinned audio LATENT indices (temporal, per-channel), ascending."""
    latents: List[int] = []
    for latent in pinned_audio_latents:
        if isinstance(latent, bool) or not isinstance(latent, (int, np.integer)):
            raise ValueError(
                f"A pinned audio latent must be an integer index, got {latent!r}.")
        latent = int(latent)
        if not 0 <= latent < num_audio_latents:
            raise ValueError(
                f"A pinned audio latent at index {latent} is outside this clip's audio grid: it "
                f"has {num_audio_latents} latent(s) per channel, so the last addressable index is "
                f"{num_audio_latents - 1}.")
        latents.append(latent)
    if len(set(latents)) != len(latents):
        raise ValueError(f"Pinned audio latents must be distinct, got {list(pinned_audio_latents)!r}.")
    return tuple(sorted(latents))


def audio_pin_row_indices(
    latents: Sequence[int],
    num_audio_latents: int,
    channels: int = AUDIO_CHANNELS,
) -> Tuple[int, ...]:
    """The CHANNEL-MAJOR row indices of a set of audio LATENT positions.

    Audio rows are channel-major (``row = channel * num_audio_latents +
    latent``, see :func:`_fill_audio_positions`), so naming a temporal latent
    span always means naming BOTH channels' rows for it -- a "half" prefix of
    the row block would pin one stereo channel's entire timeline instead of
    half the clip (see :func:`build_packed_layout`'s docstring). ``latents`` is
    walked in ITS OWN order for every channel, so a caller that hands over an
    already-sorted sequence gets an ascending, channel-major row list back --
    which is what both :func:`build_packed_layout` (to build the permutation)
    and the backend substitution site (to address the ORIGINAL, unpermuted row
    block) need.
    """
    latent_tuple = tuple(int(t) for t in latents)
    return tuple(channel * num_audio_latents + latent
                for channel in range(channels) for latent in latent_tuple)


def pin_video_rows(
    video_rows: torch.Tensor,
    source_rows: torch.Tensor,
    pinned_row_indices: Sequence[int],
    scheduler: Any,
    timestep: float,
) -> torch.Tensor:
    """Replace selected frame-major rows with scheduler-conditioned source rows."""

    if video_rows.ndim != 2 or source_rows.shape != video_rows.shape:
        raise ValueError(
            f"video and source rows must have the same [rows, channels] shape, got "
            f"{tuple(video_rows.shape)} and {tuple(source_rows.shape)}."
        )
    pinned = _validated_pinned_video_row_indices(
        pinned_row_indices,
        video_rows.shape[0],
        pinned_video_frames=(),
        keyframe_anchors=(),
    )
    if not pinned:
        return video_rows
    indices = torch.tensor(pinned, dtype=torch.long, device=video_rows.device)
    source = source_rows.to(video_rows.device, video_rows.dtype)
    video_rows[indices] = scheduler.scale_noise(
        source[indices], timestep, video_rows[indices]
    )
    return video_rows


def substitute_and_permute_audio_rows(
    audio_rows: torch.Tensor,
    source_rows: torch.Tensor,
    pinned_latents: Sequence[int],
    num_audio_latents: int,
    permutation: Optional[torch.Tensor],
) -> torch.Tensor:
    """The draw-time half of a PARTIAL audio pin: substitute, then pack.

    Mirrors :func:`pin_video_rows`'s "substitute in ORIGINAL row space first"
    recipe, plus the pack step audio alone needs: the audio block is
    CHANNEL-MAJOR, so an arbitrary temporal SET of pinned latents is not a
    contiguous prefix until it is moved there by the layout's own permutation
    (:func:`build_packed_layout`'s ``audio_row_permutation``) -- video's pins
    never need this because ``pinned_video_row_indices`` / ``pinned_video_frames``
    already name a frame-major set the layout addresses directly, with no
    accompanying reorder of the row storage itself.

    ``permutation`` MUST be the SAME layout's ``audio_row_permutation`` --
    passing its ``audio_row_order`` (the INVERSE, meant for the decode-time
    un-permute back to channel-major order) here has the same shape and dtype
    as the correct tensor, so nothing raises; it silently packs the wrong rows
    into the conditioning prefix instead.

    Both ``audio_rows`` and ``source_rows`` are addressed in ORIGINAL
    (unpermuted, channel-major) row space before the pack, so the free rows'
    own draw (K0.6's recorded order) is untouched by which latents are pinned.
    Mutates ``audio_rows`` in place before permuting it (matching the call
    site's previous inline behaviour) and returns the packed tensor.
    """
    if permutation is None:
        raise RuntimeError(
            "MiniMax-H3 partial audio pin: the layout built no permutation for it.")
    pin_rows = torch.tensor(
        audio_pin_row_indices(sorted(int(t) for t in pinned_latents), num_audio_latents),
        dtype=torch.long, device=audio_rows.device,
    )
    source = source_rows.to(audio_rows.device, audio_rows.dtype)
    audio_rows[pin_rows] = source[pin_rows]
    return audio_rows[permutation.to(audio_rows.device)]


def build_packed_layout(
    num_text_tokens: int,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    *,
    patch_size: Tuple[int, int, int] = (1, 2, 2),
    keyframe_anchors: Sequence["int | str"] = (),
    pinned_video_frames: Sequence[int] = (),
    pinned_video_row_indices: Sequence[int] = (),
    pin_target_audio: bool = False,
    pinned_audio_latents: Sequence[int] = (),
    text_token_tags: Optional[torch.Tensor] = None,
    device: Optional[torch.device | str] = None,
) -> Dict[str, Any]:
    """The ``[text | conditions | audio | video]`` layout of one request.

    Port of ``MiniMaxH3PrepareLayoutStep.build_packed_sequence``. K0.3 compared
    it against a second, independently written packed-layout implementation on
    six shape tuples: identical indices, identical tags, and a tiny packed
    forward through both assemblies bitwise identical.

    ``keyframe_anchors`` takes ``"first"``, ``"last"`` or an integer PIXEL-frame
    index, one per anchor, in packed order; see :func:`_anchor_rotary_time` for
    what an index means and why the two strings are not the same code path.
    Placement costs nothing structurally -- an anchor occupies
    ``rows_per_frame`` rows wherever it sits, and every other tensor here is
    independent of its time -- so the string cases stay byte-identical.

    ``pin_target_audio`` is ia2v (``/generate/img2vid``'s ``input_audio``): the
    generated clip's OWN audio rows are supplied at their true value and never
    denoised. It is the WHOLE-TRACK special case of ``pinned_audio_latents``
    below (every latent named, in order), kept as its own flag because ia2v
    never needs to name a subset -- passing both raises. It changes exactly one
    number in the returned dict from the unpinned case -- ``num_condition_audio_rows``
    becomes every audio row instead of 0 -- because a pinned whole track needs
    no rows of its own: the target audio rows already sit on the target's clock
    (``_fill_audio_positions``), and ``build_row_timesteps`` pins
    ``audio_indices[:n_cond_audio]`` at ``AUDIO_COND_TIMESTEP`` (1.0, which is
    EXACTLY CLEAN under this model's ``x_t = t*x0 + (1-t)*noise``), leaving
    ``denoise`` an empty slice to write. Every tensor built here is identical
    either way, which is why this is a flag and not a second builder.

    ``pinned_audio_latents`` is a PARTIAL audio pin: the latent indices (into
    ``[0, num_audio_latents)``, per channel) that are supplied at their true
    value and never denoised, while the rest of the track is generated. The
    audio rows are CHANNEL-MAJOR, so a "half" prefix of the row block would pin
    one stereo channel's entire timeline rather than half the clip -- naming an
    arbitrary temporal SET is reached the same way ``pinned_video_frames``
    reaches a partial clip, by permuting ``audio_indices``: the pinned latents'
    rows (both channels, via :func:`audio_pin_row_indices`) are moved to the
    front, ``num_condition_audio_rows`` becomes their count, and
    ``audio_row_permutation`` / ``audio_row_order`` record the permutation and
    its inverse for the caller's draw-time substitution and decode-time
    un-permute (mirroring ``video_row_permutation`` / ``video_row_order``
    exactly). Measured in ``scratchpad/minimax_h3_ai_probe_results.md``, with
    video left free (its own §4) -- the same shape with video ALSO pinned on
    the same range, the actual ``regenerate_range`` configuration when both
    this parameter and ``pinned_video_frames``/``pinned_video_row_indices``
    are supplied together, was not measured there.

    ``pinned_video_frames`` is temporal inpaint: the LATENT frames that are
    supplied at (near) their true value and never denoised while the rest of the
    clip is regenerated around them. Their rows are permuted to the front of
    ``video_indices`` and counted as conditioning, which is what lets a PREFIX
    count address an arbitrary index SET -- a permutation of the index block
    together with the same permutation of the rows is a bitwise no-op in the
    transformer (``index_copy`` / ``index_select``, everything else addressed by
    sequence position), so ``build_row_timesteps``, ``denoise``'s write slice
    and the scheduler need no change. Callers permute their rows with
    ``video_row_permutation`` and restore frame-major order with
    ``video_row_order``; both are ``None`` when nothing is pinned, and the only
    invariant given up is that ``video_indices`` is ascending, which nothing
    consumes. Measured in ``scratchpad/minimax_h3_ti_probe_results.md``.

    ``pinned_video_row_indices`` is spatial inpaint: zero-based indices into
    the frame-major flattened video-row block. The selected rows are moved to
    the conditioning prefix, with the remaining rows following them. It is
    mutually exclusive with ``pinned_video_frames`` and ``keyframe_anchors``;
    unlike a frame pin, its conditioning count is the exact number of rows.

    Returns the tensors the transformer reads by name plus the two conditioning
    row counts, which the loop needs to know which rows it may write.
    """
    _, patch_h, patch_w = patch_size
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_condition_rows = len(keyframe_anchors) * rows_per_frame
    num_audio_rows = num_audio_latents * AUDIO_CHANNELS
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows

    condition_start = num_text_tokens
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows

    # 1. The (t, h, w) rotary grid. Text rows sit on the time axis at their row
    # index and the media rows continue from there, so the prompt length shifts
    # the whole media clock.
    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    frame_grid, width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    for index, anchor in enumerate(keyframe_anchors):
        anchor_time = _anchor_rotary_time(anchor, num_text_tokens, num_latent_frames)
        rows = slice(condition_start + index * rows_per_frame,
                     condition_start + (index + 1) * rows_per_frame)
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    # Audio rows are CHANNEL-MAJOR and share the video's rotary clock.
    _fill_audio_positions(position_ids, slice(audio_start, video_start), num_audio_latents,
                          float(num_text_tokens), width_grid)

    video_position_ids = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_position_ids[:, :, 0] = _temporal_position_grid(num_latent_frames, float(num_text_tokens))[:, None]
    video_position_ids[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_position_ids.reshape(-1, 3)

    # 2. Row indices and modality tags. Conditioning rows are VIDEO rows: they
    # lead the video index block, which is why the loop can protect them by
    # simply never writing the first `num_condition_rows` entries.
    video_indices = torch.cat([torch.arange(condition_start, audio_start),
                               torch.arange(video_start, sequence_length)])
    audio_indices = torch.arange(audio_start, video_start)
    text_indices = torch.arange(num_text_tokens)

    num_condition_video_rows = num_condition_rows
    video_row_permutation: Optional[torch.Tensor] = None
    video_row_order: Optional[torch.Tensor] = None
    if len(pinned_video_row_indices):
        pinned_rows = _validated_pinned_video_row_indices(
            pinned_video_row_indices,
            num_video_rows,
            pinned_video_frames=pinned_video_frames,
            keyframe_anchors=keyframe_anchors,
        )
        pinned_set = set(pinned_rows)
        free_rows = tuple(row for row in range(num_video_rows) if row not in pinned_set)
        video_row_permutation = torch.tensor(
            (*pinned_rows, *free_rows), dtype=torch.long)
        video_row_order = torch.argsort(video_row_permutation)
        video_indices = video_indices[video_row_permutation]
        num_condition_video_rows = len(pinned_rows)
    elif len(pinned_video_frames):
        pinned = _validated_pinned_frames(pinned_video_frames, num_latent_frames,
                                          keyframe_anchors)
        free = [frame for frame in range(num_latent_frames) if frame not in set(pinned)]
        video_row_permutation = torch.cat([
            torch.arange(frame * rows_per_frame, (frame + 1) * rows_per_frame)
            for frame in (*pinned, *free)])
        video_row_order = torch.argsort(video_row_permutation)
        video_indices = video_indices[video_row_permutation]
        num_condition_video_rows = len(pinned) * rows_per_frame

    if pin_target_audio and len(pinned_audio_latents):
        raise ValueError(
            "MiniMax-H3 cannot combine pin_target_audio with pinned_audio_latents: pass the "
            "whole-track pin via pin_target_audio=True, or a subset via pinned_audio_latents, "
            "not both.")
    if pin_target_audio:
        # The whole-track case, generalised through the same permutation path
        # a partial pin takes: every latent named, ascending, which is an
        # IDENTITY permutation of an already-ascending `audio_indices` -- so
        # this is bitwise unchanged from the pre-partial-pin behaviour.
        pinned_audio_latents = tuple(range(num_audio_latents))

    num_condition_audio_rows = 0
    audio_row_permutation: Optional[torch.Tensor] = None
    audio_row_order: Optional[torch.Tensor] = None
    if len(pinned_audio_latents):
        pinned_latents = _validated_pinned_audio_latents(pinned_audio_latents, num_audio_latents)
        pinned_latent_set = set(pinned_latents)
        free_latents = tuple(t for t in range(num_audio_latents) if t not in pinned_latent_set)
        pinned_rows = audio_pin_row_indices(pinned_latents, num_audio_latents)
        free_rows = audio_pin_row_indices(free_latents, num_audio_latents)
        audio_row_permutation = torch.tensor((*pinned_rows, *free_rows), dtype=torch.long)
        audio_row_order = torch.argsort(audio_row_permutation)
        audio_indices = audio_indices[audio_row_permutation]
        num_condition_audio_rows = len(pinned_rows)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    if text_token_tags is None:
        token_tags[text_indices] = TEXT_TAG
    else:
        # A keyframe's vision block is tagged VIDEO even though it lives in the
        # text span (fl2va / ref2va presentations).
        token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = AUDIO_TAG
    token_tags[video_indices] = VIDEO_TAG

    layout: Dict[str, Any] = {
        "sequence_length": sequence_length,
        # float32: the rope casts to float32 anyway (and the two candidate
        # float64 spatial grids are bitwise equal after that cast), so the
        # sequence-length x 3 float64 buffer is not carried to the GPU.
        "position_ids": position_ids.to(torch.float32),
        "token_tags": token_tags,
        "video_indices": video_indices,
        "audio_indices": audio_indices,
        "text_indices": text_indices,
        "num_condition_video_rows": num_condition_video_rows,
        "num_condition_audio_rows": num_condition_audio_rows,
        "rows_per_frame": rows_per_frame,
        # frame-major rows -> packed rows, and back.
        "video_row_permutation": video_row_permutation,
        "video_row_order": video_row_order,
        # channel-major audio rows -> packed rows, and back. Same contract as
        # the video pair above: both None when nothing is pinned.
        "audio_row_permutation": audio_row_permutation,
        "audio_row_order": audio_row_order,
    }
    if device is not None:
        for key in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices",
                    "video_row_permutation", "video_row_order",
                    "audio_row_permutation", "audio_row_order"):
            if layout[key] is not None:
                layout[key] = layout[key].to(device)
    return layout


def build_ref2va_packed_layout(
    text_token_tags: Sequence[int],
    reference_blocks: Sequence[Tuple[str, bool]],
    condition_latent_shapes: Sequence[Tuple[int, int, int]],
    reference_audio_row_counts: Sequence[int],
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    *,
    patch_size: Tuple[int, int, int] = (1, 2, 2),
    keyframe_anchors: Sequence["int | str"] = (),
    device: Optional[torch.device | str] = None,
) -> Dict[str, Any]:
    """The ``[text | reference blocks | keyframe anchors | target audio | target video]`` layout.

    Port of ``MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence``.
    Returns the same dict shape as :func:`build_packed_layout`, so the denoise
    loop, the timestep plan and the decode path do not branch on the workflow.

    WHERE THE REFERENCE ROWS SIT, AND WHY IT MATTERS THAT THEY SIT THERE
    -------------------------------------------------------------------
    One block per reference, in REQUEST order, between the text span and the
    generated rows -- never after them. Within a block:

    * an image reference contributes ``ref_video_rows`` and advances the shared
      rotary clock by exactly 1.0 (a single integer slot, NOT a latent frame's
      5/3 units);
    * a standalone audio reference contributes channel-major audio rows on the
      TARGET width grid and advances the clock by its latent count;
    * a video reference contributes its soundtrack's audio rows FIRST and its
      video rows second, both from the same clock origin (so the two are rotary-
      aligned exactly as the generated audio and video are) on the video's own
      width grid, and advances the clock by the larger of the two spans.

    The generated audio and video rows then start from the origin the reference
    blocks left behind. Because every reference row precedes every generated row
    of its own modality, ``video_indices`` and ``audio_indices`` both list their
    conditioning rows FIRST -- which is the invariant ``build_row_timesteps``
    and the denoise loop rely on to pin the anchors and write only what is
    generated. Nothing else about the loop changes for ``ref2va``.

    Args:
        text_token_tags: per-row modality tag of the presentation's text span
            (text rows are 1; a reference's vision block is tagged 0/video).
        reference_blocks: ``(kind, has_audio)`` per reference in packed order,
            where kind is ``"image"``, ``"video"`` or ``"audio"``. ``has_audio``
            is what decides whether a VIDEO reference reserves soundtrack rows
            ahead of its video rows; it is always true for an audio reference
            and always false for an image one.
        condition_latent_shapes: ``(T_lat, H_lat, W_lat)`` per VISUAL reference,
            in packed order -- the shape the VAE actually produced, so the
            layout and the encoded conditioning can never disagree.
        reference_audio_row_counts: packed row count per AUDIO-BEARING
            reference, in packed order.
        keyframe_anchors: ``"first"``, ``"last"`` or an integer pixel-frame
            index per anchor (same vocabulary as :func:`build_packed_layout`),
            placed AFTER every reference block and before the target audio/
            video, at ``rows_per_frame`` (the TARGET canvas's) rows each. Each
            anchor's rotary time is computed from the rotary clock the
            reference loop leaves behind, not from ``num_text_tokens`` -- the
            care ``minimax_h3_conditioning_design.md`` §1.1 names, because that
            clock is where the target's own rows already start. Anchors do not
            advance the clock themselves, matching :func:`build_packed_layout`,
            where a keyframe never shifts the generated video's own origin.
            Empty by default, which reproduces the pre-C5 layout bitwise.
    """
    _, patch_h, patch_w = patch_size
    text_tags = torch.as_tensor(list(text_token_tags), dtype=torch.long)
    num_text_tokens = int(text_tags.numel())
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_target_video_rows = num_latent_frames * rows_per_frame
    num_target_audio_rows = num_audio_latents * AUDIO_CHANNELS

    visual_geometry = iter(tuple(shape) for shape in condition_latent_shapes)
    audio_row_counts = iter(int(count) for count in reference_audio_row_counts)
    num_condition_video_rows = sum(
        frames * (height // patch_h) * (width // patch_w)
        for frames, height, width in (tuple(shape) for shape in condition_latent_shapes)
    ) + len(keyframe_anchors) * rows_per_frame
    num_condition_audio_rows = sum(int(count) for count in reference_audio_row_counts)
    sequence_length = (num_text_tokens + num_condition_video_rows + num_condition_audio_rows
                       + num_target_audio_rows + num_target_video_rows)

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)
    target_frame_grid, target_width_grid = _frame_position_grid(
        latent_height, latent_width, patch_h, patch_w)

    video_index_blocks: List[torch.Tensor] = []
    audio_index_blocks: List[torch.Tensor] = []
    cursor = num_text_tokens
    rotary_time = float(num_text_tokens)
    for kind, has_audio in reference_blocks:
        if kind == "image":
            frames, height, width = next(visual_geometry)
            num_rows = frames * (height // patch_h) * (width // patch_w)
            rows = slice(cursor, cursor + num_rows)
            cursor = rows.stop
            video_index_blocks.append(torch.arange(rows.start, rows.stop))
            frame_grid, _ = _frame_position_grid(height, width, patch_h, patch_w)
            position_ids[rows, 0] = rotary_time
            position_ids[rows, 1:] = frame_grid.repeat(frames, 1)
            rotary_time += 1.0
        elif kind == "audio":
            num_rows = next(audio_row_counts)
            rows = slice(cursor, cursor + num_rows)
            cursor = rows.stop
            audio_index_blocks.append(torch.arange(rows.start, rows.stop))
            reference_audio_latents = num_rows // AUDIO_CHANNELS
            _fill_audio_positions(position_ids, rows, reference_audio_latents, rotary_time,
                                  target_width_grid)
            rotary_time += float(reference_audio_latents)
        elif kind == "video":
            num_audio_rows = next(audio_row_counts) if has_audio else 0
            frames, height, width = next(visual_geometry)
            num_video_rows = frames * (height // patch_h) * (width // patch_w)
            audio_rows = slice(cursor, cursor + num_audio_rows)
            video_rows = slice(audio_rows.stop, audio_rows.stop + num_video_rows)
            cursor = video_rows.stop
            if num_audio_rows:
                audio_index_blocks.append(torch.arange(audio_rows.start, audio_rows.stop))
            video_index_blocks.append(torch.arange(video_rows.start, video_rows.stop))

            frame_grid, width_grid = _frame_position_grid(height, width, patch_h, patch_w)
            reference_audio_latents = num_audio_rows // AUDIO_CHANNELS
            if num_audio_rows:
                _fill_audio_positions(position_ids, audio_rows, reference_audio_latents,
                                      rotary_time, width_grid)
            frame_time = _temporal_position_grid(frames, rotary_time)
            position_ids[video_rows, 0] = frame_time.repeat_interleave(frame_grid.shape[0])
            position_ids[video_rows, 1:] = frame_grid.repeat(frames, 1)
            # SEQUENTIAL float64 sum, deliberately NOT the pairwise sum the
            # `"last"` keyframe anchor uses: the reference implementation keeps
            # both, one per call site, and the two differ in the last ulp from
            # 16 latent frames onwards.
            video_span = sum(
                ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[index % len(ROPE_FRAMES_PER_LATENT)]
                for index in range(frames))
            rotary_time += max(float(reference_audio_latents), video_span)
        else:
            raise ValueError(f"A reference must be an 'image', a 'video' or an 'audio', got {kind!r}.")

    # Keyframe anchors (C5: anchors x references, merged builder), one more
    # block kind after every reference. `rotary_time` is left where the
    # reference loop put it -- an anchor's own time is computed FROM that
    # origin, but placing one does not move it, so the target audio/video below
    # still starts where it would with no anchors at all.
    for anchor in keyframe_anchors:
        anchor_time = _anchor_rotary_time(anchor, rotary_time, num_latent_frames)
        rows = slice(cursor, cursor + rows_per_frame)
        cursor = rows.stop
        video_index_blocks.append(torch.arange(rows.start, rows.stop))
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = target_frame_grid

    audio_start = cursor
    video_start = audio_start + num_target_audio_rows
    _fill_audio_positions(position_ids, slice(audio_start, video_start), num_audio_latents,
                          rotary_time, target_width_grid)
    frame_time = _temporal_position_grid(num_latent_frames, rotary_time)
    position_ids[video_start:, 0] = frame_time.repeat_interleave(target_frame_grid.shape[0])
    position_ids[video_start:, 1:] = target_frame_grid.repeat(num_latent_frames, 1)

    video_indices = torch.cat(video_index_blocks + [torch.arange(video_start, sequence_length)])
    audio_indices = torch.cat(audio_index_blocks + [torch.arange(audio_start, video_start)])
    text_indices = torch.arange(num_text_tokens)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_tags
    token_tags[audio_indices] = AUDIO_TAG
    token_tags[video_indices] = VIDEO_TAG

    layout: Dict[str, Any] = {
        "sequence_length": sequence_length,
        "position_ids": position_ids.to(torch.float32),
        "token_tags": token_tags,
        "video_indices": video_indices,
        "audio_indices": audio_indices,
        "text_indices": text_indices,
        "num_condition_video_rows": num_condition_video_rows,
        "num_condition_audio_rows": num_condition_audio_rows,
        "rows_per_frame": rows_per_frame,
        # Same dict shape as `build_packed_layout`; ref2va pins nothing -- a
        # reference soundtrack conditions through its own leading block
        # (`reference_audio_row_counts`), never through this permutation pair.
        "video_row_permutation": None,
        "video_row_order": None,
        "audio_row_permutation": None,
        "audio_row_order": None,
    }
    if device is not None:
        for key in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices"):
            layout[key] = layout[key].to(device)
    return layout


def packed_row_counts(layout: Dict[str, Any]) -> Dict[str, int]:
    """Count generated and conditioning rows in a completed packed layout."""
    condition_video = int(layout["num_condition_video_rows"])
    condition_audio = int(layout["num_condition_audio_rows"])
    video = int(layout["video_indices"].numel())
    audio = int(layout["audio_indices"].numel())
    text = int(layout["text_indices"].numel())
    return {
        "text": text,
        "condition_video": condition_video,
        "target_video": video - condition_video,
        "condition_audio": condition_audio,
        "target_audio": audio - condition_audio,
        "total": text + video + audio,
    }


def build_row_timesteps(
    layout: Dict[str, Any],
    video_timestep: float,
    audio_timestep: float,
    *,
    keyframe_noise_aug: float = VISUAL_COND_TIMESTEP,
    audio_cond_timestep: float = AUDIO_COND_TIMESTEP,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(distinct timesteps, per-row index into them)`` for one step.

    One forward serves every noise level in the sequence at once: the generated
    video and audio rows sit on their own schedules while the conditioning rows
    stay pinned. Text rows never reach an output head and inherit the video
    timestep. Port of ``MiniMaxH3SetTimestepsStep.build_row_timesteps``.
    """
    video_indices = layout["video_indices"]
    audio_indices = layout["audio_indices"]
    n_cond_video = layout["num_condition_video_rows"]
    n_cond_audio = layout["num_condition_audio_rows"]

    sequence_length = int(layout["sequence_length"])
    row_timesteps = torch.full((sequence_length,), float(video_timestep), dtype=torch.float32,
                               device=video_indices.device)
    row_timesteps[video_indices[:n_cond_video]] = max(float(video_timestep), float(keyframe_noise_aug))
    row_timesteps[audio_indices[n_cond_audio:]] = float(audio_timestep)
    row_timesteps[audio_indices[:n_cond_audio]] = float(audio_cond_timestep)
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)


# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------

def draw_noise(
    generator: torch.Generator,
    *,
    video_latent_shape: Tuple[int, ...],
    num_audio_latents: int,
    condition_shapes: Sequence[Tuple[int, ...]] = (),
    device: torch.device | str = "cuda",
    audio_latent_channels: int = 32,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """One generator, three kinds of draw, in the order the reference uses.

    ``(condition_noises, video_noise, audio_noise)``:

    1. one draw per visual condition, in packed order, at the condition's OWN
       latent shape (they do not share one on ``ref2va``);
    2. the video noise as a 5-D LATENT tensor ``(1, C, T, H, W)``;
    3. the audio noise DIRECTLY in row layout ``(T_aud * channels, 32)``.

    Every draw is float32. K0.6 hashed all three against an independent
    reimplementation of the upstream blocks and matched 12/12 exactly, on both
    a CPU and a CUDA generator, for t2va, fl2va(1), fl2va(2) and
    ``audio_enable=False`` — the last of which must not perturb the sequence,
    which is why NOTHING here is conditional on the audio flag.
    """
    condition_noises = [
        torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
        for shape in condition_shapes
    ]
    video_noise = torch.randn(video_latent_shape, generator=generator, device=device,
                              dtype=torch.float32)
    audio_noise = torch.randn((num_audio_latents * AUDIO_CHANNELS, audio_latent_channels),
                              generator=generator, device=device, dtype=torch.float32)
    return condition_noises, video_noise, audio_noise


# --------------------------------------------------------------------------
# Visual conditioning (fl2va: first and/or last frame)
# --------------------------------------------------------------------------

@torch.no_grad()
def encode_condition_images(
    vae,
    images: Sequence[np.ndarray],
    *,
    latents_mean: Sequence[float],
    latents_std: Sequence[float],
    pixel_mean: Sequence[float],
    pixel_std: Sequence[float],
    device: torch.device | str = "cuda",
    encode_seed: int = KEYFRAME_ENCODE_SEED,
) -> List[torch.Tensor]:
    """Keyframe images ``uint8 [H, W, 3]`` -> normalized latents ``[1, 24, 1, h, w]``.

    The exact inverse of :func:`decode_video`'s pixel handling, and it has the
    same two traps:

    * the video VAE takes **ImageNet-normalised RGB over a [0, 1] base**, not
      ``[-1, 1]`` like every other VAE in this repo, so the image is divided by
      255 and then normalised with the ImageNet constants;
    * the latents carry 24 PER-CHANNEL mean/std vectors and are normalised as
      ``(z - mean) / std`` (decode applies the inverse). The fp32 config values
      are used, not the fp16 copies in the weight file.

    A single frame goes through the SPATIAL encoder alone: the vendored
    ``AutoencoderKLMiniMaxH3._encode`` special-cases ``num_frames == 1`` and
    returns exactly one latent frame, which is the geometry the packed layout
    reserves ``rows_per_frame`` rows for.

    Two more steps of the released recipe, both easy to drop and neither
    optional (diffusers ``encode_vae_condition``, whose docstring says "every
    part of it is needed to reproduce its conditioning"):

    * the posterior is **SAMPLED, not read at its mode**, under a FRESH CPU
      generator seeded at ``encode_seed`` (42 upstream) rather than from the
      request's generator. That is what makes the sample both deterministic and
      invisible to the recorded draw order: the request generator is never
      touched, so K0.6's one-draw-per-condition sequence is unchanged (the
      layout test asserts this);
    * the sampled latent is **rounded to float16 and back** before
      normalisation — about 11 bits of every conditioning latent. Upstream does
      it explicitly; this port would otherwise inherit it only implicitly from
      the loader's fp16 VAE cast and would diverge silently if that cast ever
      changed, so it is written out here.
    """
    return [
        encode_visual_condition(
            vae, np.asarray(image)[None],
            latents_mean=latents_mean, latents_std=latents_std,
            pixel_mean=pixel_mean, pixel_std=pixel_std,
            device=device, encode_seed=encode_seed,
        )
        for image in images
    ]


@torch.no_grad()
def encode_visual_condition(
    vae,
    frames: np.ndarray,
    *,
    latents_mean: Sequence[float],
    latents_std: Sequence[float],
    pixel_mean: Sequence[float],
    pixel_std: Sequence[float],
    device: torch.device | str = "cuda",
    encode_seed: int = KEYFRAME_ENCODE_SEED,
) -> torch.Tensor:
    """One visual condition: ``uint8 [T, H, W, 3]`` -> ``[1, 24, T_lat, h, w]``.

    The single recipe behind BOTH conditioning paths -- an ``fl2va`` keyframe
    (``T == 1``) and a ``ref2va`` image or video reference -- so the two cannot
    drift apart. ``T == 1`` goes through the SPATIAL encoder alone (the vendored
    ``_encode`` special-cases it and returns exactly one latent frame); a frame
    stack goes through the temporal chunking, which is what turns ``17n + 5``
    frames into ``5n + 2`` latent frames.

    See :func:`encode_condition_images` for the two easily-dropped steps of the
    released recipe this implements (the seeded posterior SAMPLE and the float16
    round trip) and the two pixel/latent conventions it owes the VAE.
    """
    torch_device = torch.device(device)
    vae_dtype = next(vae.parameters()).dtype
    pix_mean = torch.tensor(list(pixel_mean), device=torch_device).view(1, -1, 1, 1, 1)
    pix_std = torch.tensor(list(pixel_std), device=torch_device).view(1, -1, 1, 1, 1)
    mean = torch.tensor(list(latents_mean), device=torch_device).view(1, -1, 1, 1, 1)
    std = torch.tensor(list(latents_std), device=torch_device).view(1, -1, 1, 1, 1)

    pixels = torch.from_numpy(np.ascontiguousarray(frames)).to(torch_device, torch.float32)
    # [T, H, W, 3] -> [1, 3, T, H, W], the 5-D clip shape the VAE takes.
    pixels = pixels.permute(3, 0, 1, 2)[None] / 255.0
    pixels = (pixels - pix_mean) / pix_std
    posterior = vae.encode(pixels.to(vae_dtype), return_dict=True).latent_dist
    # A CPU generator against CUDA parameters is deliberate and is what upstream
    # does: `randn_tensor` draws on the generator's device and moves the result,
    # so the sample is identical on either device.
    latent = posterior.sample(generator=torch.Generator().manual_seed(int(encode_seed)))
    latent = latent.to(torch.float16).float()
    return (latent - mean) / std


def build_condition_rows(
    scheduler,
    condition_latents: Sequence[torch.Tensor],
    condition_noises: Sequence[torch.Tensor],
    *,
    keyframe_noise_aug: float = VISUAL_COND_TIMESTEP,
    patch_size: Tuple[int, int, int] = (1, 2, 2),
) -> torch.Tensor:
    """The packed rows of the visual conditioning anchors, in packed order.

    Each anchor is noised to ``keyframe_noise_aug`` with its OWN draw from
    :func:`draw_noise` — the same level :func:`build_row_timesteps` then pins
    that anchor's rows at for every step, so what the model is told about the
    row and what the row contains agree. ``scale_noise`` is the vendored
    scheduler's own forward process (``x_t = t*x0 + (1-t)*noise``, MiniMax-H3's
    ``t`` convention where ``t = 1`` is clean).

    Returns ``[num_conditions * rows_per_frame, C*pt*ph*pw]``, i.e. exactly the
    block the layout reserves at the head of the video index range. An empty
    condition list returns an empty tensor so the t2va path needs no branch.
    """
    if len(condition_latents) != len(condition_noises):
        raise ValueError(
            f"Every visual condition needs its own noise draw: got {len(condition_latents)} "
            f"latent(s) and {len(condition_noises)} noise tensor(s).")
    rows = [
        patchify_video_latents(
            scheduler.scale_noise(latent, keyframe_noise_aug, noise.to(latent.device, latent.dtype)),
            patch_size,
        )[0]
        for latent, noise in zip(condition_latents, condition_noises)
    ]
    if not rows:
        return torch.zeros(0, 0)
    return torch.cat(rows, dim=0)


# --------------------------------------------------------------------------
# Text encoding
# --------------------------------------------------------------------------

@torch.no_grad()
def encode_prompt(
    text_encoder,
    tokenizer,
    prompt: str,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    layer: int = TEXT_ENCODER_LAYER,
    max_tokens: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """``(prompt_embeds [1, S, 5120] on CPU, token count)``.

    The presentation of a ``t2va`` request is the prompt VERBATIM — no chat
    template, no special tokens (``add_special_tokens=False``). The conditioning
    is the hidden state after decoder layer ``layer``, read WITHOUT the final
    norm; the released file is truncated to exactly ``layer`` layers and the
    loader installs an ``nn.Identity`` in place of that norm, so running the
    stack to the end is the same read.

    Each layer's parameters are materialised on the GPU in float32 and the layer
    is called through ``torch.func.functional_call``; the CPU weights are never
    touched, so they stay memory-mapped (see the module docstring for the 73 GB
    vs 49.8 GB measurement behind that).
    """
    language_model = text_encoder.model.language_model
    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not token_ids:
        # An empty prompt would build a zero-row text span, and the media rows'
        # rotary clock starts at the text length, so the sequence would still be
        # valid -- but `index_copy` with an empty index and the AdaLN row for a
        # modality that is not present are both untested paths. Refuse instead.
        raise ValueError("MiniMax-H3 needs a non-empty prompt: its packed sequence conditions on "
                         "the prompt's own rows, and an empty prompt produces none.")
    if max_tokens is not None and len(token_ids) > max_tokens:
        token_ids = token_ids[:max_tokens]

    hidden = encode_presentation(text_encoder, token_ids, device=device, dtype=dtype, layer=layer)
    return hidden, len(token_ids)


# Floating dtypes this helper is allowed to WIDEN to float32. Deliberately an
# allow-list, not "is_floating_point() and dtype is not float8_*": a new float8
# variant (or any other narrow float format a future quant module buffers)
# must opt IN by being added here, not be silently promoted by default. Every
# dtype outside this set -- narrow floats included -- passes through
# unchanged, keyed on DTYPE alone so this stays correct for any module (a
# `ConvRotInt8Linear`'s int8 codes, an `Nvfp4Linear`'s packed uint8 codes and
# float8_e4m3fn block scales, or a future module class this file never names).
_WIDEN_TO_FLOAT32_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32, torch.float64})


def _gpu_module_params(module, device) -> Dict[str, torch.Tensor]:
    """One module's parameters and buffers, on ``device``, narrow floats widened.

    The dict `torch.func.functional_call` runs the module with. Building it
    instead of moving the module is the whole point (see the module docstring):
    the CPU tensors stay attached to the memory-mapped file, so a 48 GiB encoder
    never materialises an anonymous copy of itself.

    Ordinary bf16/fp16 parameters and buffers are widened to float32 (the
    precision `functional_call` runs the module in). A buffer whose dtype is
    NOT in ``_WIDEN_TO_FLOAT32_DTYPES`` keeps its own dtype instead: that
    includes every non-floating buffer (a mask, an index, an int8/uint8
    quantization code) AND, since ``torch.float8_e4m3fn.is_floating_point()``
    is True, a narrow float8 block scale (``Nvfp4Linear.weight_scale``) --
    which a plain ``is_floating_point()`` branch would otherwise widen into a
    float32 tensor `comfy_kitchen.dequantize_nvfp4` does not accept as its
    block-scale argument, corrupting the dequant silently rather than erroring.
    """
    gpu_params = {
        name: tensor.to(device, torch.float32) if tensor.dtype in _WIDEN_TO_FLOAT32_DTYPES
        else tensor.to(device)
        for name, tensor in module.named_parameters()
    }
    gpu_params.update({
        name: tensor.to(device, torch.float32) if tensor.dtype in _WIDEN_TO_FLOAT32_DTYPES
        else tensor.to(device)
        for name, tensor in module.named_buffers()
    })
    return gpu_params


@torch.no_grad()
def _encode_vision_blocks(
    text_encoder,
    vision_inputs: Dict[str, torch.Tensor],
    device: torch.device | str,
) -> Dict[str, Any]:
    """Run the Qwen3-VL vision tower over a presentation's blocks.

    Returns ``{"image_embeds", "video_embeds", "image_deepstack",
    "video_deepstack"}`` (each ``None`` when that modality is absent). The tower
    is 1.1 GB of the 48 GiB file (MEASURED) and is called through
    ``functional_call`` for the same reason the decoder layers are: moving it
    would detach its weights from the file mapping for the rest of the process.

    ``deepstack`` is not optional decoration. Qwen3-VL feeds three intermediate
    vision-tower feature maps back into the FIRST decoder layers at the visual
    rows (``Qwen3VLTextModel._deepstack_process``); a presentation encoded
    without them is a different conditioning, silently.
    """
    visual = text_encoder.model.visual
    merge_area = visual.spatial_merge_size ** 2
    gpu_params = _gpu_module_params(visual, device)
    result: Dict[str, Any] = {"image_embeds": None, "video_embeds": None,
                              "image_deepstack": None, "video_deepstack": None}
    try:
        for prefix, grid_key, out_key, deep_key in (
                ("pixel_values", "image_grid_thw", "image_embeds", "image_deepstack"),
                ("pixel_values_videos", "video_grid_thw", "video_embeds", "video_deepstack")):
            pixels = vision_inputs.get(prefix)
            if pixels is None:
                continue
            grid = vision_inputs[grid_key].to(device)
            output = torch.func.functional_call(
                visual, gpu_params,
                args=(pixels.to(device, torch.float32),),
                kwargs=dict(grid_thw=grid),
            )
            merged = output.pooler_output if hasattr(output, "pooler_output") else output[1]
            deepstack = output.deepstack_features if hasattr(output, "deepstack_features") else output[2]
            # `get_image_features` splits the merged rows per image/video and
            # then concatenates them again; the split exists for callers that
            # need per-item embeddings, so the concatenation is the same tensor.
            split_sizes = (grid.prod(-1) // merge_area).tolist()
            if sum(split_sizes) != merged.shape[0]:  # pragma: no cover - shape contract
                raise RuntimeError(
                    f"The Qwen3-VL vision tower produced {merged.shape[0]} merged row(s) where its "
                    f"grid says {sum(split_sizes)}.")
            result[out_key] = merged
            result[deep_key] = deepstack
    finally:
        del gpu_params
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


@torch.no_grad()
def encode_presentation(
    text_encoder,
    token_ids: Sequence[int],
    *,
    vision_inputs: Optional[Dict[str, torch.Tensor]] = None,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    layer: int = TEXT_ENCODER_LAYER,
) -> torch.Tensor:
    """The layer-``layer`` hidden state of one tokenized presentation.

    ``[1, S, 5120]`` on the CPU. This is MiniMax-H3's conditioning for every
    workflow: a ``t2va`` presentation is the prompt verbatim, and an ``fl2va`` /
    ``ref2va`` one additionally carries a label and a vision block per keyframe
    or reference (``vision_inputs`` holds those blocks' pixels by the
    conditioner's own parameter names). The read is WITHOUT the final norm --
    the released file is truncated to exactly ``layer`` layers and the loader
    installs an ``nn.Identity`` in place of that norm.

    THREE THINGS THE VISION PATH OWES ``Qwen3VLModel.forward`` and that a
    hand-written layer loop drops by default:

    1. the merged vision rows are scattered into the embeddings at the
       modality's placeholder tokens (``<|image_pad|>`` / ``<|video_pad|>``);
    2. **deepstack** -- three intermediate tower feature maps are ADDED to the
       visual rows after each of the first decoder layers;
    3. **mrope** -- with a vision block present the position ids are the 3-D
       ``(t, h, w)`` grid ``get_rope_index`` builds, not ``arange``.

    Each decoder layer's parameters are materialised on the GPU in float32 and
    the layer is called through ``torch.func.functional_call``; the CPU weights
    are never touched, so they stay memory-mapped (73 GB vs 49.8 GB peak RSS,
    K0.7 — see the module docstring).
    """
    model = text_encoder.model
    language_model = model.language_model
    config = text_encoder.config
    token_list = list(token_ids)
    if not token_list:
        raise ValueError("MiniMax-H3 needs a non-empty presentation to condition on.")

    input_ids = torch.tensor([token_list], dtype=torch.long)
    sequence_length = input_ids.shape[1]
    hidden = language_model.embed_tokens(input_ids).to(device, torch.float32)

    visual_mask = None
    deepstack_embeds = None
    if vision_inputs:
        vision = _encode_vision_blocks(text_encoder, vision_inputs, device)
        device_ids = input_ids.to(device)
        image_mask = (device_ids == config.image_token_id) if vision["image_embeds"] is not None else None
        video_mask = (device_ids == config.video_token_id) if vision["video_embeds"] is not None else None
        for mask, embeds, name in ((image_mask, vision["image_embeds"], "image"),
                                   (video_mask, vision["video_embeds"], "video")):
            if mask is None:
                continue
            if int(mask.sum()) != embeds.shape[0]:
                raise RuntimeError(
                    f"MiniMax-H3's presentation reserves {int(mask.sum())} {name} placeholder "
                    f"token(s) but the vision tower produced {embeds.shape[0]} row(s).")
            hidden = hidden.masked_scatter(mask[..., None].expand_as(hidden), embeds.to(hidden.dtype))

        # Deepstack, joined exactly as `Qwen3VLModel.forward` joins it when both
        # modalities are present: one row block per visual position, in sequence
        # order, with each modality writing its own rows.
        if image_mask is not None and video_mask is not None:
            visual_mask = image_mask | video_mask
            image_joint, video_joint = image_mask[visual_mask], video_mask[visual_mask]
            deepstack_embeds = []
            for image_embed, video_embed in zip(vision["image_deepstack"], vision["video_deepstack"]):
                joint = image_embed.new_zeros(int(visual_mask.sum()), image_embed.shape[-1])
                joint[image_joint] = image_embed
                joint[video_joint] = video_embed
                deepstack_embeds.append(joint)
        elif image_mask is not None:
            visual_mask, deepstack_embeds = image_mask, vision["image_deepstack"]
        elif video_mask is not None:
            visual_mask, deepstack_embeds = video_mask, vision["video_deepstack"]

        # mrope: the (t, h, w) grid, computed by the conditioner's own helper so
        # the vision blocks' spatial layout is the one it was trained with.
        position_ids, _deltas = model.get_rope_index(
            input_ids,
            vision_inputs.get("image_grid_thw"),
            vision_inputs.get("video_grid_thw"),
            attention_mask=torch.ones_like(input_ids),
        )
        position_ids = position_ids.to(device)
    else:
        # Qwen3-VL's rotary expands a 2-D `position_ids` to its three mrope
        # sections, so a text-only presentation gets ordinary RoPE.
        position_ids = torch.arange(sequence_length, device=device).unsqueeze(0)

    # The rotary module is moved for the length of this call and PUT BACK: it is
    # the one submodule this function would otherwise mutate, and leaving it on
    # the GPU makes any later CPU-side forward through the same object fail on a
    # device mismatch (its `inv_freq` is a computed non-persistent buffer, so
    # moving it costs nothing and detaches nothing from the file mapping).
    rotary = language_model.rotary_emb
    rotary_device = next((b.device for b in rotary.buffers()), None)
    try:
        cos, sin = rotary.to(device)(hidden, position_ids)
    finally:
        if rotary_device is not None:
            rotary.to(rotary_device)
    # The 1-D positions the attention (and the causal mask) use: mrope's three
    # sections share one text clock, which is `position_ids[0]`.
    text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
    causal_mask = torch.full((1, 1, sequence_length, sequence_length), float("-inf"),
                             device=device, dtype=torch.float32).triu(1)

    for index, decoder_layer in enumerate(language_model.layers):
        gpu_params = _gpu_module_params(decoder_layer, device)
        result = torch.func.functional_call(
            decoder_layer, gpu_params,
            kwargs=dict(hidden_states=hidden, position_embeddings=(cos, sin),
                        attention_mask=causal_mask, position_ids=text_position_ids,
                        past_key_values=None, use_cache=False),
        )
        hidden = result[0] if isinstance(result, tuple) else result
        if deepstack_embeds is not None and index < len(deepstack_embeds):
            hidden = hidden.clone()
            hidden[visual_mask, :] = hidden[visual_mask, :] + deepstack_embeds[index].to(hidden.dtype)
        del gpu_params
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not torch.isfinite(hidden).all():
        raise RuntimeError("The MiniMax-H3 text encoder produced non-finite hidden states.")
    return hidden.to("cpu", dtype)


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

@torch.no_grad()
def denoise(
    transformer,
    scheduler,
    audio_scheduler,
    *,
    prompt_embeds: torch.Tensor,
    layout: Dict[str, Any],
    video_rows: torch.Tensor,
    audio_rows: torch.Tensor,
    num_inference_steps: int,
    device: torch.device | str = "cuda",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    substep_reporter: Optional[Any] = None,
    step_callback: Optional[Callable[..., None]] = None,
    preview_latent_shape: Optional[Tuple[int, int, int]] = None,
    video_row_order: Optional[torch.Tensor] = None,
    latent_channels: int = 24,
    patch_size: Tuple[int, int, int] = (1, 2, 2),
    keyframe_noise_aug: float = VISUAL_COND_TIMESTEP,
    spectrum_params: Optional[Dict[str, Any]] = None,
    block_swap_on: bool = False,
    label: str = "MiniMax-H3",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the packed denoise loop. Returns the final ``(video, audio)`` rows.

    ONE forward per step — the checkpoint is guidance-distilled, so there is no
    unconditional branch to batch or to run twice.

    Note on the step count: ``num_inference_steps`` counts SIGMA GRID POINTS,
    terminal ``0`` included, so it drives ``num_inference_steps - 1`` model
    evaluations. That is the scheduler's own contract (and K0.4 confirmed the
    grid's duplicate-collapse never fires at any step count this integration
    uses, so the mapping is exactly 1:1).

    ``substep_reporter`` is an optional
    ``core.inference.substep_progress.SubStepReporter`` whose forward hooks tick
    progress from inside a step (one step is ~150s here); this loop only tells
    it where the step boundaries are. Its hooks are attached and removed by the
    caller.

    ``step_callback`` is the latent-preview hook, called as
    ``(i, total, latents, None, pred_x0)`` with BOTH tensors unpatchified to
    ``[1, C, T_lat, H_lat, W_lat]`` — the packed row layout is meaningless to
    every preview decoder in this repo. It therefore needs the latent geometry:
    pass ``preview_latent_shape=(T_lat, H_lat, W_lat)`` whenever a callback is
    given (a missing shape is a ValueError rather than a silently packed
    preview).

    ``video_row_order`` is the layout's own, for a temporal-inpaint request:
    with pinned frames the conditioning prefix is clip content, so the preview
    takes EVERY video row and restores frame-major order
    (``frame_major = video_rows[video_row_order]``) instead of unpatchifying the
    generated tail as if it were the whole clip. Nothing else reads it.
    """
    if step_callback is not None and preview_latent_shape is None:
        raise ValueError(
            "denoise(step_callback=...) also needs preview_latent_shape=(T_lat, H_lat, W_lat): the "
            "preview estimate is handed over as latents, not as packed rows.")
    if video_row_order is not None and video_row_order.numel() != video_rows.shape[0]:
        raise ValueError(
            f"denoise(video_row_order=...) orders {video_row_order.numel()} row(s) but was given "
            f"{video_rows.shape[0]} video row(s); it must be the layout's own permutation.")

    torch_device = torch.device(device)
    scheduler.set_shift(SHIFT_VIDEO)
    audio_scheduler.set_shift(SHIFT_AUDIO)
    scheduler.set_timesteps(num_inference_steps, device=torch_device)
    audio_scheduler.set_timesteps(num_inference_steps, device=torch_device)
    # Both schedulers advance their own `_step_index` once per iteration, so
    # they stay in lock-step; pinning the start makes that explicit rather than
    # dependent on `index_for_timestep` finding the right row.
    scheduler.set_begin_index(0)
    audio_scheduler.set_begin_index(0)

    timesteps = scheduler.timesteps
    audio_timesteps = audio_scheduler.timesteps
    total_steps = len(timesteps)

    spectrum_video = spectrum_audio = None
    if spectrum_params and spectrum_params.get("spectrum_enable", False):
        if block_swap_on:
            print("[Spectrum] MiniMax-H3 disabled: Block Swap is enabled "
                  "(forecast skips desync swap rotation)")
        else:
            from core.inference.spectrum_forecaster import build_output_forecaster

            spectrum_video = build_output_forecaster(
                spectrum_params, total_steps, label="MiniMax-H3 video")
            if spectrum_video is not None:
                spectrum_audio = build_output_forecaster(
                    spectrum_params, total_steps, label="MiniMax-H3 audio")
                if spectrum_audio is None:
                    print("[Spectrum] MiniMax-H3: audio forecaster failed to build; "
                          "disabling Spectrum entirely")
                    spectrum_video = None
                else:
                    print(f"[Spectrum] MiniMax-H3: {len(spectrum_video.anchors)}/{total_steps} "
                          f"actual passes (paired video/audio final-output forecasting)")

    fbcache = None
    if spectrum_params:
        from core.inference.fbcache import build_fbcache, fbcache_active

        if fbcache_active(spectrum_params):
            if spectrum_params.get("spectrum_enable", False):
                print("[FBCache] MiniMax-H3 disabled: Spectrum is enabled "
                      "(same trajectory-redundancy target)")
            elif block_swap_on:
                print("[FBCache] MiniMax-H3 disabled: Block Swap is enabled "
                      "(cache hits desync swap rotation)")
            elif not hasattr(transformer, "attach_fbcache"):
                print("[FBCache] MiniMax-H3 disabled: transformer wrapper is unavailable")
            else:
                fbcache = build_fbcache(
                    spectrum_params,
                    label="MiniMax-H3 guarded video/audio",
                    max_consecutive_hits=2,
                    total_steps=total_steps,
                    tail_steps=1,
                )

    n_cond_video = layout["num_condition_video_rows"]
    n_cond_audio = layout["num_condition_audio_rows"]
    layout_kwargs = dict(
        token_tags=layout["token_tags"],
        position_ids=layout["position_ids"],
        video_indices=layout["video_indices"],
        audio_indices=layout["audio_indices"],
        text_indices=layout["text_indices"],
    )

    def call_transformer(step_idx, unique_timesteps, timestep_indices):
        kwargs = dict(
            hidden_states=video_rows[None],
            audio_hidden_states=audio_rows[None],
            encoder_hidden_states=prompt_embeds,
            timestep=unique_timesteps.to(torch_device),
            timestep_indices=timestep_indices.to(torch_device),
            return_dict=False,
            **layout_kwargs,
        )
        if fbcache is None:
            return transformer(**kwargs)
        transformer.attach_fbcache(
            fbcache,
            rows_per_frame=int(layout["rows_per_frame"]),
            condition_video_rows=int(layout["num_condition_video_rows"]),
        )
        transformer._fbcache_step = step_idx
        try:
            return transformer(**kwargs)
        finally:
            transformer.attach_fbcache(None)

    for i, timestep in enumerate(timesteps):
        raise_if_cancelled()
        if substep_reporter is not None:
            substep_reporter.begin_step(i, total_steps)
        unique_timesteps, timestep_indices = build_row_timesteps(
            layout, float(timestep), float(audio_timesteps[i]),
            keyframe_noise_aug=keyframe_noise_aug,
        )

        spectrum_skip = spectrum_video is not None and not spectrum_video.is_anchor(i)
        if spectrum_skip:
            video_velocity = spectrum_video.forecast(i)
            audio_velocity = spectrum_audio.forecast(i)
        else:
            video_velocity, audio_velocity = call_transformer(
                i, unique_timesteps, timestep_indices)
            if spectrum_video is not None:
                spectrum_video.record(i, video_velocity)
                spectrum_audio.record(i, audio_velocity)

        # x0 = x_t + sigma_t * v_t -- the H3 convention, `+` not `-`. It reads
        # x_t, the latent this step's velocity was predicted FROM, so it is
        # computed BEFORE the scheduler overwrites those rows with x_{t+1}.
        pred_x0_rows = None
        if step_callback is not None:
            sigma = 1.0 - float(timestep)
            pred_x0_rows = (video_rows[n_cond_video:].float()
                            + sigma * video_velocity[0, n_cond_video:].float())
            if video_row_order is not None:
                # A pinned row is already (near) x0 and is never stepped, so it
                # previews as itself; an anchor row is not clip content at all
                # and stays out of the preview.
                pred_x0_rows = torch.cat([video_rows[:n_cond_video].float(), pred_x0_rows])

        # Only the GENERATED rows are ever written, so any conditioning anchor
        # survives the whole loop by construction rather than by re-imposition.
        video_rows[n_cond_video:] = scheduler.step(
            video_velocity[0, n_cond_video:].float(), timestep,
            video_rows[n_cond_video:], return_dict=False)[0]
        audio_rows[n_cond_audio:] = audio_scheduler.step(
            audio_velocity[0, n_cond_audio:].float(), audio_timesteps[i],
            audio_rows[n_cond_audio:], return_dict=False)[0]

        if progress_callback is not None:
            try:
                progress_callback(i + 1, total_steps)
            except Exception as exc:  # progress must never take a generation down
                print(f"[{label}] progress_callback raised: {exc}")
        if step_callback is not None:
            try:
                latent_frames, latent_height, latent_width = preview_latent_shape
                unpack = lambda rows: unpatchify_video_rows(  # noqa: E731
                    rows if video_row_order is None else rows[video_row_order],
                    latent_frames, latent_height, latent_width,
                    latent_channels=latent_channels, patch_size=patch_size)
                clip = video_rows if video_row_order is not None else video_rows[n_cond_video:]
                step_callback(i, total_steps, unpack(clip), None, unpack(pred_x0_rows))
            except Exception as exc:
                print(f"[{label}] step_callback raised: {exc}")

    if spectrum_video is not None:
        video_stats = spectrum_video.stats()
        audio_stats = spectrum_audio.stats()
        if video_stats != audio_stats:
            raise RuntimeError(
                "MiniMax-H3 Spectrum video/audio forecasters left lock-step: "
                f"video={video_stats}, audio={audio_stats}")
        print(f"[Spectrum] MiniMax-H3 summary: {video_stats['anchors']} anchor(s), "
              f"{video_stats['forecasts']} forecast(s) of {video_stats['total']} step(s)")

    if fbcache is not None:
        print(f"[FBCache] MiniMax-H3 summary: {fbcache.n_hits} hit(s), "
              f"{fbcache.n_miss} miss(es); temporal guard on")

    return video_rows, audio_rows


# --------------------------------------------------------------------------
# Decode
# --------------------------------------------------------------------------

@torch.no_grad()
def decode_video(
    vae,
    latents: torch.Tensor,
    *,
    latents_mean: Sequence[float],
    latents_std: Sequence[float],
    pixel_mean: Sequence[float],
    pixel_std: Sequence[float],
    device: torch.device | str = "cuda",
) -> np.ndarray:
    """Normalized latents ``[1, 24, T, H, W]`` -> ``uint8 [T, H, W, 3]`` RGB.

    Three conventions, all measured, all easy to get wrong:

    * the latents carry 24 PER-CHANNEL mean/std vectors (no scalar scaling
      factor) and are denormalized as ``z * std + mean``. The fp32 values from
      the config are used, not the fp16 copies inside the weight file;
    * the VAE emits **ImageNet-normalised RGB over a [0, 1] base**, not
      ``[-1, 1]`` like every other VAE in this repo, so the pixel side is
      reverted with the ImageNet constants;
    * the spatial TILING POLICY is pinned by the loader and is NOT a memory
      knob here: flipping it moves the decode by rel-RMS 0.212 on the same
      input (K0.5 supplementary). Nothing in this function touches it.

    Two notes on the fp16 weights the loader casts to (MEASURED, Phase 2, with
    the tiling policy held fixed on both arms: PSNR 77.74 dB / rel-RMS 2.764e-04
    against a full-fp32 decode, 2.2 s / 5.19 GB against 7.3 s / 13.33 GB):

    * upstream instead keeps fp32 weights and wraps this decode in
      ``torch.autocast(float16)``, which is not the same computation (autocast
      leaves the norms and reductions in fp32). The difference from our shape is
      bounded by the fp32 A/B above and has not been separately measured;
    * fp16 weights make ``torch.rms_norm`` fall off its fused path
      ("Mismatch dtype between input and weight") inside the decoder. It is
      numerically harmless -- the A/B above IS the fp16 path -- and costs only a
      little of the 2.2 s.
    """
    torch_device = torch.device(device)
    mean = torch.tensor(list(latents_mean), device=torch_device).view(1, -1, 1, 1, 1)
    std = torch.tensor(list(latents_std), device=torch_device).view(1, -1, 1, 1, 1)
    latents = latents.to(torch_device, torch.float32) * std + mean

    vae_dtype = next(vae.parameters()).dtype
    frames = vae.decode(latents.to(vae_dtype), return_dict=False)[0]

    pix_mean = torch.tensor(list(pixel_mean), device=frames.device).view(1, -1, 1, 1, 1)
    pix_std = torch.tensor(list(pixel_std), device=frames.device).view(1, -1, 1, 1, 1)

    # The pixel-side denormalization (mean/std/clamp/permute/round) is purely elementwise and
    # per-frame -- there is no cross-frame reduction here -- so it is safe to walk the temporal
    # axis in the VAE's own natural chunk unit instead of promoting the whole clip to fp32 on GPU
    # and taking a second full-size contiguous copy for the permute. This keeps only one chunk's
    # worth of fp32 pixels resident at a time; the numerical result is byte-identical to doing it
    # in one shot because every op below is elementwise/layout-only.
    tokens_chunk_size = getattr(vae, "tokens_chunk_size", None)
    temporal_compression_ratio = getattr(vae, "temporal_compression_ratio", None)
    if tokens_chunk_size and temporal_compression_ratio:
        chunk_frames = tokens_chunk_size * temporal_compression_ratio
    else:
        chunk_frames = 17

    total_frames = frames.shape[2]
    out_chunks: list[np.ndarray] = []
    for start in range(0, total_frames, chunk_frames):
        end = min(start + chunk_frames, total_frames)
        chunk = frames[:, :, start:end]
        chunk = (chunk.float() * pix_std + pix_mean).clamp(0, 1)
        # [1, 3, t, H, W] -> [t, H, W, 3] uint8
        chunk = chunk[0].permute(1, 2, 3, 0).contiguous().cpu().numpy()
        out_chunks.append((chunk * 255.0).round().astype(np.uint8))
        del chunk

    return out_chunks[0] if len(out_chunks) == 1 else np.concatenate(out_chunks, axis=0)


@torch.no_grad()
def decode_audio(
    audio_vae,
    audio_latents: torch.Tensor,
    *,
    latents_mean: Sequence[float],
    latents_std: Sequence[float],
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Normalized audio latents ``[ch, 32, T]`` -> waveform ``[ch, samples]``.

    The autoencoder is MONO: stereo is carried as two BATCH items, which is what
    the channel-major packing in the sequence unpacks into.
    """
    torch_device = torch.device(device)
    mean = torch.tensor(list(latents_mean), device=torch_device).view(1, -1, 1)
    std = torch.tensor(list(latents_std), device=torch_device).view(1, -1, 1)
    latents = audio_latents.to(torch_device, torch.float32) * std + mean

    vae_dtype = next(audio_vae.parameters()).dtype
    waveform = audio_vae.decode(latents.to(vae_dtype), return_dict=False)[0]
    # [ch, 1, samples] -> [ch, samples]
    return waveform.float().reshape(waveform.shape[0], -1).cpu()


def trim_audio_to_video(audio: torch.Tensor, num_frames: int, fps: float = 24.0,
                        sample_rate: int = 32000) -> torch.Tensor:
    """Trim the decoded waveform to the video's own duration.

    ``T_aud = round(T / 24 * 40)`` rounds at the audio latent grid and the
    decoder emits exactly 800 samples per latent, so the soundtrack is up to
    half a latent (12.5 ms) longer than the video. The mux would otherwise
    produce a file whose audio outlives its last frame.
    """
    wanted = int(round(num_frames / fps * sample_rate))
    if audio.shape[-1] <= wanted:
        return audio
    return audio[..., :wanted].contiguous()
