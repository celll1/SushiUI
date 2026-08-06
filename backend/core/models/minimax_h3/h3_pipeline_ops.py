"""Sampling operations for MiniMax-H3 — the denoise loop this repo owns.

MiniMax-H3 ships as a diffusers **Modular** pipeline and nothing else: there is
no stock ``DiffusionPipeline`` to drive (its ``model_index.json`` declares only
``MiniMaxH3ModularPipeline``). SushiUI needs per-step progress, cancellation,
latent preview, its own strictly-sequential offload sequencing and, later, a
block-loop wrapper for block swap / FBCache — none of which the Modular stack's
``ComponentsManager`` auto-offload would tolerate. So, following the Anima
precedent (``core/models/anima/anima_pipeline_ops.py``), the model classes are
vendored and the loop lives here.

WHAT THIS MODULE OWES ITS SOURCES
---------------------------------
Every contract below is either a verbatim port of the diffusers ``minimax-h3``
modular blocks (Apache-2.0; ``before_denoise.py`` / ``denoise.py`` /
``decoders.py`` / ``encoders.py``) or was verified against an independent port
of ComfyUI's ``comfy/ldm/minimax/model.py`` in the K0 conformance suite. The
two places a re-derivation is easy to get subtly wrong are called out inline:

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
   rows on the video grid, the audio rows on the audio grid. ComfyUI instead
   integrates the *video* grid alone and scales the audio velocity by
   ``d(sigma_a)/d(sigma_v)``, because its sampler only knows one schedule; K0.4
   verified that slope against fp64 autograd (3.05e-16) and the two
   formulations agree to first order. This module follows the diffusers
   reference (two schedulers), because that is the implementation the vendored
   transformer and scheduler came from and it is exact on each stream's own
   grid rather than first-order-accurate on the other's.
5. **Conditioning rows are pinned**, at ``t = max(t_video, 0.999)`` for visual
   conditioning and ``t = 1.0`` for audio references, for every step. The loop
   only ever writes the GENERATED rows, so the anchors ride through unchanged.
   The anchors themselves are built at the SAME level they are pinned at:
   ``encode_condition_images`` produces the clean latent and
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

    Verbatim shape math from the diffusers block of the same name (and
    equivalent to ComfyUI's ``einsum("nctrhpwq->nthwcrpq")``).
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
    # `np.linspace(..., endpoint=False)` computes `(arange*ratio)/n + left`,
    # while ComfyUI computes `arange*(ratio/n) + left`. Same value, different
    # float64 association, <= 1 ulp apart; unobservable through the model (the
    # rope casts to float32 and the two grids are then bitwise equal, K0.3
    # supplementary). The diffusers form is used because the vendored
    # transformer is the diffusers one -- do not "fix" this to match ComfyUI.
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


def build_packed_layout(
    num_text_tokens: int,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    *,
    patch_size: Tuple[int, int, int] = (1, 2, 2),
    keyframe_anchors: Sequence[str] = (),
    text_token_tags: Optional[torch.Tensor] = None,
    device: Optional[torch.device | str] = None,
) -> Dict[str, Any]:
    """The ``[text | conditions | audio | video]`` layout of one request.

    Port of ``MiniMaxH3PrepareLayoutStep.build_packed_sequence``. K0.3 compared
    it against an independent port of ComfyUI's ``PackedLayout`` on six shape
    tuples: identical indices, identical tags, and a tiny packed forward through
    both assemblies bitwise identical.

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
        if anchor == "first":
            anchor_time = float(num_text_tokens)
        elif anchor == "last":
            # numpy's PAIRWISE summation, because that is how the reference
            # computes this anchor; a sequential sum differs in the last ulp
            # from 16 latent frames onwards.
            spans = np.ones(num_latent_frames, dtype=np.float64) * ROPE_FRAME_RESCALE
            for offset in range(len(ROPE_FRAMES_PER_LATENT)):
                spans[offset::len(ROPE_FRAMES_PER_LATENT)] *= ROPE_FRAMES_PER_LATENT[offset]
            anchor_time = float(num_text_tokens) + float(spans.sum()) - ROPE_FRAME_RESCALE
        else:
            raise ValueError(f"A keyframe anchor must be 'first' or 'last', got {anchor!r}.")
        rows = slice(condition_start + index * rows_per_frame,
                     condition_start + (index + 1) * rows_per_frame)
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    # Audio rows are CHANNEL-MAJOR and share the video's rotary clock (one unit
    # per latent: 40 latents/s == 24 fps * 5/3). They carry no height coordinate
    # and are pinned to the two extremes of the width grid, one per channel.
    audio_time = float(num_text_tokens) + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[audio_start:video_start, 0] = audio_time.repeat(AUDIO_CHANNELS)
    position_ids[audio_start:video_start, 2] = torch.cat([
        torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
        torch.full((num_audio_rows - num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
    ])

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
        "num_condition_video_rows": num_condition_rows,
        "num_condition_audio_rows": 0,
        "rows_per_frame": rows_per_frame,
    }
    if device is not None:
        for key in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices"):
            layout[key] = layout[key].to(device)
    return layout


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

    The posterior is read at its **mode**, not sampled. Two reasons, both
    deliberate: the request generator's draw sequence is a recorded contract
    (K0.6 hashes one draw per condition, and a posterior sample would either add
    a draw or silently consume the global RNG), and it matches how every other
    keyframe/reference encode in this repo reads a latent distribution
    (``ltx2``'s img2vid keyframe, ``krea2``, ``lens``, ``ideogram4``, ACE-Step's
    reference audio). MiniMax's own reference wrapper (`klvae.encode_base`)
    samples the posterior from the global RNG instead; the difference is one
    posterior sigma of jitter on an anchor the sampler then holds fixed.
    """
    torch_device = torch.device(device)
    vae_dtype = next(vae.parameters()).dtype
    pix_mean = torch.tensor(list(pixel_mean), device=torch_device).view(1, -1, 1, 1, 1)
    pix_std = torch.tensor(list(pixel_std), device=torch_device).view(1, -1, 1, 1, 1)
    mean = torch.tensor(list(latents_mean), device=torch_device).view(1, -1, 1, 1, 1)
    std = torch.tensor(list(latents_std), device=torch_device).view(1, -1, 1, 1, 1)

    latents: List[torch.Tensor] = []
    for image in images:
        pixels = torch.from_numpy(np.ascontiguousarray(image)).to(torch_device, torch.float32)
        # [H, W, 3] -> [1, 3, 1, H, W], the 5-D single-frame clip shape.
        pixels = pixels.permute(2, 0, 1)[None, :, None] / 255.0
        pixels = (pixels - pix_mean) / pix_std
        posterior = vae.encode(pixels.to(vae_dtype), return_dict=True).latent_dist
        latent = posterior.mode().float()
        latents.append((latent - mean) / std)
    return latents


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

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    sequence_length = input_ids.shape[1]

    hidden = language_model.embed_tokens(input_ids.cpu()).to(device, torch.float32)
    position_ids = torch.arange(sequence_length, device=device).unsqueeze(0)
    # Qwen3-VL's rotary expands a 2-D `position_ids` to its three mrope
    # sections, so a text-only presentation gets ordinary RoPE.
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
    causal_mask = torch.full((1, 1, sequence_length, sequence_length), float("-inf"),
                             device=device, dtype=torch.float32).triu(1)

    for decoder_layer in language_model.layers:
        gpu_params = {name: tensor.to(device, torch.float32)
                      for name, tensor in decoder_layer.named_parameters()}
        gpu_params.update({
            # A non-float buffer (a mask, an index) must keep its dtype.
            name: tensor.to(device, torch.float32) if tensor.is_floating_point() else tensor.to(device)
            for name, tensor in decoder_layer.named_buffers()
        })
        result = torch.func.functional_call(
            decoder_layer, gpu_params,
            kwargs=dict(hidden_states=hidden, position_embeddings=(cos, sin),
                        attention_mask=causal_mask, position_ids=position_ids,
                        past_key_values=None, use_cache=False),
        )
        hidden = result[0] if isinstance(result, tuple) else result
        del gpu_params
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not torch.isfinite(hidden).all():
        raise RuntimeError("The MiniMax-H3 text encoder produced non-finite hidden states.")
    return hidden.to("cpu", dtype), sequence_length


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
    step_callback: Optional[Callable[..., None]] = None,
    preview_latent_shape: Optional[Tuple[int, int, int]] = None,
    latent_channels: int = 24,
    patch_size: Tuple[int, int, int] = (1, 2, 2),
    keyframe_noise_aug: float = VISUAL_COND_TIMESTEP,
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

    ``step_callback`` is the latent-preview hook, called as
    ``(i, total, latents, None, pred_x0)`` with BOTH tensors unpatchified to
    ``[1, C, T_lat, H_lat, W_lat]`` — the packed row layout is meaningless to
    every preview decoder in this repo. It therefore needs the latent geometry:
    pass ``preview_latent_shape=(T_lat, H_lat, W_lat)`` whenever a callback is
    given (a missing shape is a ValueError rather than a silently packed
    preview).
    """
    if step_callback is not None and preview_latent_shape is None:
        raise ValueError(
            "denoise(step_callback=...) also needs preview_latent_shape=(T_lat, H_lat, W_lat): the "
            "preview estimate is handed over as latents, not as packed rows.")

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

    n_cond_video = layout["num_condition_video_rows"]
    n_cond_audio = layout["num_condition_audio_rows"]
    layout_kwargs = dict(
        token_tags=layout["token_tags"],
        position_ids=layout["position_ids"],
        video_indices=layout["video_indices"],
        audio_indices=layout["audio_indices"],
        text_indices=layout["text_indices"],
    )

    for i, timestep in enumerate(timesteps):
        raise_if_cancelled()
        unique_timesteps, timestep_indices = build_row_timesteps(
            layout, float(timestep), float(audio_timesteps[i]),
            keyframe_noise_aug=keyframe_noise_aug,
        )

        video_velocity, audio_velocity = transformer(
            hidden_states=video_rows[None],
            audio_hidden_states=audio_rows[None],
            encoder_hidden_states=prompt_embeds,
            timestep=unique_timesteps.to(torch_device),
            timestep_indices=timestep_indices.to(torch_device),
            return_dict=False,
            **layout_kwargs,
        )

        # x0 = x_t + sigma_t * v_t -- the H3 convention, `+` not `-`. It reads
        # x_t, the latent this step's velocity was predicted FROM, so it is
        # computed BEFORE the scheduler overwrites those rows with x_{t+1}.
        pred_x0_rows = None
        if step_callback is not None:
            sigma = 1.0 - float(timestep)
            pred_x0_rows = (video_rows[n_cond_video:].float()
                            + sigma * video_velocity[0, n_cond_video:].float())

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
                    rows, latent_frames, latent_height, latent_width,
                    latent_channels=latent_channels, patch_size=patch_size)
                step_callback(i, total_steps, unpack(video_rows[n_cond_video:]), None,
                              unpack(pred_x0_rows))
            except Exception as exc:
                print(f"[{label}] step_callback raised: {exc}")

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
    frames = (frames.float() * pix_std + pix_mean).clamp(0, 1)
    # [1, 3, T, H, W] -> [T, H, W, 3] uint8
    frames = frames[0].permute(1, 2, 3, 0).contiguous().cpu().numpy()
    return (frames * 255.0).round().astype(np.uint8)


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
