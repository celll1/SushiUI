"""MiniMax-H3 ``ref2va`` references — the media side of omni-reference generation.

``ref2va`` conditions on an ordered list of references: up to 9 images, 3 videos
(each optionally carrying its own soundtrack) and 3 standalone audio clips, at
most 12 in total. This module owns everything that happens to that media BEFORE
the packed sequence exists:

* validating the request against the released checkpoint's limits;
* normalising every reference onto MiniMax-H3's own rates and resolutions
  (24 fps, the audio VAE's 32 kHz, an image's own 2048-pixel short edge or the
  generation canvas);
* encoding the visual references through the video VAE and the soundtracks
  through the audio VAE;
* building the tokenized *presentation* — the labelled prompt the Qwen3-VL
  conditioner reads, with one vision block per image and one per merged frame
  pair of a video.

WHERE THIS COMES FROM
---------------------
Every rule below is a port of the diffusers ``minimax-h3`` branch modular
blocks (Apache-2.0): ``before_encoder.MiniMaxH3Ref2VASetupStep`` (normalisation
+ limits), ``encoders.MiniMaxH3Ref2VAReferenceEncoderStep`` (VAE encoding) and
``encoders.MiniMaxH3Ref2VATextEncoderStep`` (presentation). Where ComfyUI's
``nodes_minimax_h3.MiniMaxH3ReferenceToVideo`` does the same thing differently,
the difference is called out inline and the diffusers form is the one
implemented — it is the reference this integration's vendored model code came
from.

THE ORDER OF THE LIST IS SEMANTIC, TWICE OVER
---------------------------------------------
It fixes the ``<Picture i>`` / ``<Audio j>`` / ``<Video k>`` labels the prompt
refers to, and it advances the shared audio/video rotary clock of the packed
sequence. A different order is a different request, so nothing here sorts or
regroups the references: images, videos and audio arrive interleaved exactly as
the caller listed them and stay that way through the layout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# The released checkpoint's limits (README + the diffusers setup block's
# defaults). They bound the request, not the architecture: a fine-tune that
# packs more references would raise them here.
# ---------------------------------------------------------------------------
MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_AUDIOS = 3
MAX_REFERENCES = 12

# Canvas rule of the released checkpoint. A video reference is put on it (with
# its OWN aspect ratio); an image reference is not -- images are encoded at a
# short edge of their own, with no area cap, which is what
# ``reference_image_size="max"`` selects.
CANVAS_MULTIPLE = 32
CANVAS_SHORT_EDGE = 768
CANVAS_MAX_PIXELS = 768 * 1344
REFERENCE_IMAGE_SHORT_EDGE = 2048
MIN_ASPECT_RATIO = 1.0 / 4.0
MAX_ASPECT_RATIO = 4.0

# The rate the CONDITIONER reads a reference video at (the video VAE still sees
# every 24 fps frame). Qwen3-VL then merges the sampled frames in groups of its
# temporal patch, and each group becomes one timestamped vision block.
REFERENCE_VIDEO_SAMPLE_FPS = 2.0

# The video VAE's chunking: `17 * n + 5` pixel frames -> `5 * n + 2` latent
# frames. A reference video is snapped DOWN onto that grid so the encode needs
# no padding.
VAE_FRAMES_PER_CHUNK = 17
VAE_LATENTS_PER_CHUNK = 5
# Shortest reference video this integration accepts, and it is a floor imposed
# from two directions at once: the video VAE cannot decode -- or usefully encode
# -- fewer than 22 frames (the 17n+5 grid's first multi-chunk point), and the
# conditioner needs at least 13 frames at 24 fps to fill one merged vision
# block. Upstream's snap-down arithmetic silently produces a shorter clip than
# it claims below this, so it is refused with the reason instead.
MIN_REFERENCE_VIDEO_FRAMES = 22


@dataclass
class MiniMaxH3Reference:
    """One reference of a ``ref2va`` request, in the order the model reads it.

    Exactly one of the three modalities is populated:

    * ``kind="image"``: ``image`` is a PIL image;
    * ``kind="video"``: ``frames`` is ``uint8 [T, H, W, 3]`` at ``fps``, with an
      optional ``audio`` soundtrack ``[channels, samples]`` at ``sample_rate``;
    * ``kind="audio"``: ``audio`` alone.

    ``fps`` and ``sample_rate`` are not decoration: MiniMax-H3 resamples a
    reference onto its own 24 fps and onto the audio VAE's 32 kHz, so media
    whose real rate was lost on the way in is conditioned on at the wrong speed
    with nothing to raise about it.
    """

    kind: str
    image: Optional[Image.Image] = None
    frames: Optional[np.ndarray] = None
    fps: Optional[float] = None
    audio: Optional[torch.Tensor] = None
    sample_rate: Optional[int] = None
    # Internal callers may preserve a pre-normalized video canvas instead of
    # applying the released reference-video upscale rule again.
    video_canvas: Optional[Tuple[int, int]] = None
    # Carried through for messages and for the gallery row; never read by the
    # model.
    label: Optional[str] = None

    @property
    def has_audio(self) -> bool:
        """Whether this reference contributes audio rows to the packed sequence."""
        return self.audio is not None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_references(references: Sequence[MiniMaxH3Reference]) -> None:
    """Refuse a reference list the released checkpoint cannot serve.

    Raises ``ValueError`` with the limit that was exceeded. These are the
    model's limits, not this repo's, and they are validated server-side rather
    than trusted from the client.
    """
    if not references:
        raise ValueError(
            "ref2va needs at least one reference; use /generate/txt2vid for a text-only request.")
    kinds = [reference.kind for reference in references]
    for kind, limit in (("image", MAX_REFERENCE_IMAGES),
                        ("video", MAX_REFERENCE_VIDEOS),
                        ("audio", MAX_REFERENCE_AUDIOS)):
        if kinds.count(kind) > limit:
            raise ValueError(
                f"MiniMax-H3 accepts at most {limit} {kind} reference(s), got {kinds.count(kind)}.")
    if len(kinds) > MAX_REFERENCES:
        raise ValueError(
            f"MiniMax-H3 accepts at most {MAX_REFERENCES} references in total, got {len(kinds)}.")
    if set(kinds) == {"audio"}:
        raise ValueError(
            "An audio reference has to be paired with at least one image or video reference and "
            "cannot be used on its own: a standalone soundtrack never reaches the conditioner, so a "
            "request built from audio alone conditions the vision stream on nothing.")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def resolve_canvas_size(
    aspect_width: float,
    aspect_height: float,
    *,
    multiple: int = CANVAS_MULTIPLE,
    short_edge: int = CANVAS_SHORT_EDGE,
    max_pixels: int = CANVAS_MAX_PIXELS,
) -> Tuple[int, int]:
    """``(height, width)`` of MiniMax-H3's canvas for one aspect ratio.

    Port of the diffusers ``resolve_canvas_size``: short edge at ``short_edge``,
    area capped at ``max_pixels``, both axes then rounded to ``multiple`` -- so
    the final area can land slightly above the pre-rounding budget. Only the
    RATIO of the first two arguments matters.
    """
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}.")
    ratio = aspect_width / aspect_height
    if not MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:
        raise ValueError(
            f"MiniMax-H3 supports aspect ratios from 1:4 to 4:1, got "
            f"{aspect_width}:{aspect_height} ({ratio:g}).")
    if ratio >= 1.0:
        width, height = short_edge * ratio, float(short_edge)
    else:
        width, height = float(short_edge), short_edge / ratio
    area = width * height
    if area > max_pixels:
        scale = (max_pixels / area) ** 0.5
        width, height = width * scale, height * scale
    return (max(multiple, round(height / multiple) * multiple),
            max(multiple, round(width / multiple) * multiple))


def snap_reference_video_frames(num_frames: int) -> int:
    """The largest ``17 * n + 5`` at or below ``num_frames``.

    The video VAE encodes a reference without padding only on its own grid, and
    upstream snaps DOWN rather than up so a reference is never invented past its
    own end.
    """
    if num_frames < MIN_REFERENCE_VIDEO_FRAMES:
        raise ValueError(
            f"A MiniMax-H3 reference video must run at least {MIN_REFERENCE_VIDEO_FRAMES} frames at "
            f"24 fps ({MIN_REFERENCE_VIDEO_FRAMES / 24.0:.2f} s) -- the video VAE's own 17n+5 chunk "
            f"grid starts there and the conditioner needs at least 13 frames to fill one vision "
            f"block -- got {num_frames}.")
    return ((num_frames - VAE_LATENTS_PER_CHUNK) // VAE_FRAMES_PER_CHUNK) * VAE_FRAMES_PER_CHUNK \
        + VAE_LATENTS_PER_CHUNK


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _as_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"A reference image must be (height, width, 3) RGB pixels, got {tuple(array.shape)}.")
    if array.dtype != np.uint8:
        array = (array * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


def normalize_reference_image(
    image,
    *,
    reference_short_edge: int = REFERENCE_IMAGE_SHORT_EDGE,
    multiple: int = CANVAS_MULTIPLE,
    canvas: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    """One image reference onto the resolution MiniMax-H3 encodes it at.

    TWO sizings exist, and they come from the two reference implementations:

    * ``canvas is None`` -- the released recipe (diffusers
      ``MiniMaxH3Ref2VASetupStep``): a short edge of the image's OWN, 2048 for
      the released checkpoint, **upscaling included and with no area cap**. An
      image reference never binds the generated geometry, so it is not put on
      the generation canvas.
    * ``canvas=(height, width)`` -- ComfyUI's ``ref_image_size="match"``: an
      aspect-preserving scale, DOWN ONLY, to the generation's pixel area.
      Cheaper, because a reference's rows ride through every sampling step.

    Both round each axis to ``multiple``. The resize is PIL's own LANCZOS in
    both implementations, which is why nothing here goes through
    ``VaeImageProcessor`` (its array path interpolates with ``F.interpolate``).
    """
    image = _as_pil(image)
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"A reference image must have a positive size, got {image.size}.")
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"A reference image must be within 1:4 and 4:1, got {width}x{height}.")

    if canvas is None:
        scale = reference_short_edge / min(width, height)
    else:
        canvas_height, canvas_width = canvas
        scale = min(1.0, math.sqrt((canvas_width * canvas_height) / (width * height)))
    target_width = max(multiple, round(width * scale / multiple) * multiple)
    target_height = max(multiple, round(height * scale / multiple) * multiple)
    if image.size == (target_width, target_height):
        return image
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def normalize_reference_video(
    frames,
    source_fps: float,
    num_frames: int,
    *,
    target_fps: float = 24.0,
    multiple: int = CANVAS_MULTIPLE,
    short_edge: int = CANVAS_SHORT_EDGE,
    max_pixels: int = CANVAS_MAX_PIXELS,
    canvas: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """One video reference onto 24 fps, its own canvas and the generated length.

    Two passes, in the reference implementation's order: the constant-frame-rate
    resample first (whole frames dropped and duplicated, as ``ffmpeg``'s ``fps``
    filter does), the LANCZOS rescale second. Frames handed over at 24 fps and
    already on the canvas their own aspect ratio resolves to flow through
    untouched, which is the parity-exact route.
    """
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = (frames * 255.0).round().clip(0, 255).astype(np.uint8)
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(
            f"A reference video must be (num_frames, height, width, 3) RGB frames, got "
            f"{tuple(frames.shape)}.")
    if source_fps <= 0:
        raise ValueError(f"A reference video must have a positive frame rate, got {source_fps}.")

    if source_fps != target_fps:
        scale = target_fps / source_fps
        slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
        frames = np.repeat(
            frames, np.diff(slots, append=math.floor(frames.shape[0] * scale + 0.5)), axis=0)

    frames = frames[:num_frames]
    if frames.shape[0] < MIN_REFERENCE_VIDEO_FRAMES:
        raise ValueError(
            f"A MiniMax-H3 reference video must run at least {MIN_REFERENCE_VIDEO_FRAMES} frames "
            f"once resampled to {target_fps:g} fps and truncated to the generated length, got "
            f"{frames.shape[0]}.")
    if canvas is None:
        height, width = resolve_canvas_size(
            frames.shape[2], frames.shape[1], multiple=multiple, short_edge=short_edge,
            max_pixels=max_pixels)
    else:
        height, width = (int(value) for value in canvas)
        if height <= 0 or width <= 0 or height % multiple or width % multiple:
            raise ValueError(
                f"A reference video canvas must contain positive multiples of {multiple}, got "
                f"{width}x{height}.")
    if frames.shape[1:3] == (height, width):
        return frames
    return np.stack([
        np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS))
        for frame in frames
    ])


def normalize_reference_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    target_sample_rate: int,
    max_duration: float,
) -> torch.Tensor:
    """One soundtrack onto the audio VAE's sample rate, as stereo.

    The truncation happens at the SOURCE rate and the resample is a single pass,
    which is the reference implementation's order. A mono waveform is upmixed by
    repeating its channel.
    """
    waveform = torch.as_tensor(waveform)
    if waveform.ndim != 2 or waveform.shape[0] not in (1, 2):
        raise ValueError(
            "A reference soundtrack must be a (channels, num_samples) mono or stereo waveform, got "
            f"{tuple(waveform.shape)}.")
    waveform = waveform.to(torch.float32)[:, : int(max_duration * sample_rate)]
    if waveform.shape[-1] == 0:
        raise ValueError("A reference soundtrack must carry at least one sample.")
    if waveform.shape[0] != 2:
        waveform = waveform.expand(2, -1).contiguous()
    if sample_rate == target_sample_rate:
        return waveform
    try:
        import torchaudio
    except ImportError as error:  # pragma: no cover - torchaudio is a repo dependency
        raise ImportError(
            f"Resampling a MiniMax-H3 reference soundtrack from {sample_rate} Hz to "
            f"{target_sample_rate} Hz needs torchaudio.") from error
    return torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)


def pinned_audio_sample_counts(
    num_frames: int,
    *,
    fps: float = 24.0,
    sample_rate: int = 32000,
    latent_rate: float = 40.0,
) -> Tuple[int, int, int]:
    """How many samples an ia2v track must carry: ``(required, grid, clip)``.

    TWO exact lengths, and they are not the same number:

    * ``grid`` — ``num_audio_latents * (sample_rate / latent_rate)``, the
      samples the audio VAE turns into EXACTLY the audio latents the packed
      layout reserves rows for. ``num_audio_latents`` is ``round(T/fps*40)``, so
      the grid can land either side of the clip's own duration (124 frames ->
      207 latents -> 5.175 s against a 5.167 s clip; 5 frames -> 8 latents ->
      0.200 s against 0.208 s).
    * ``clip`` — ``round(num_frames / fps * sample_rate)``, what
      :func:`h3_pipeline_ops.trim_audio_to_video` muxes alongside the video.

    ``required`` is the max of the two, because BOTH slices are taken from the
    supplied waveform: one is encoded and pinned, the other is muxed. Padding
    the shortfall is not offered -- a partly-silent pinned timeline is exactly
    the mixed shape that was never measured -- so a shorter track is refused by
    the caller with these numbers in the message.
    """
    from .h3_pipeline_ops import audio_latent_frames

    num_audio_latents = audio_latent_frames(num_frames, fps=fps, latents_per_second=latent_rate)
    grid = int(round(num_audio_latents * sample_rate / latent_rate))
    clip = int(round(num_frames / fps * sample_rate))
    return max(grid, clip), grid, clip


def prepare_pinned_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    num_frames: int,
    fps: float = 24.0,
    target_sample_rate: int = 32000,
    latent_rate: float = 40.0,
) -> torch.Tensor:
    """One uploaded track as the ia2v condition: 32 kHz stereo, exact length.

    Returns ``[2, required_samples]`` float32 at ``target_sample_rate``, head
    aligned -- the first ``required_samples`` of the supplied track and nothing
    else. The caller encodes ``[:grid]`` of it and muxes ``[:clip]`` of it (see
    :func:`pinned_audio_sample_counts`); both are slices of the SOURCE, which is
    what makes the returned soundtrack sample-exact rather than a VAE round
    trip.

    Raises ``ValueError`` naming both durations when the track is too short.
    Longer is fine and is trimmed here; the trim is head-aligned because the
    audio's own clock is the clip's clock and there is no offset to express.

    The truncation happens at the SOURCE rate and the resample is a single
    pass, the same order :func:`normalize_reference_audio` uses -- with two
    seconds of margin kept before the resample so the windowed-sinc filter's
    tail never decides the last sample of the pinned track.
    """
    waveform = torch.as_tensor(waveform)
    if waveform.ndim != 2 or waveform.shape[0] not in (1, 2):
        raise ValueError(
            "An input audio track must be a (channels, num_samples) mono or stereo waveform, got "
            f"{tuple(waveform.shape)}.")
    waveform = waveform.to(torch.float32)
    if waveform.shape[-1] == 0:
        raise ValueError("An input audio track must carry at least one sample.")

    required, _grid, _clip = pinned_audio_sample_counts(
        num_frames, fps=fps, sample_rate=target_sample_rate, latent_rate=latent_rate)
    required_seconds = required / float(target_sample_rate)
    supplied_seconds = waveform.shape[-1] / float(sample_rate)
    if supplied_seconds + 1e-9 < required_seconds:
        raise ValueError(
            f"the input audio track is shorter than the clip: it runs "
            f"{supplied_seconds:.3f}s and this request needs at least "
            f"{required_seconds:.3f}s ({num_frames} frames at {fps:g} fps, whose audio grid is "
            f"{required} sample(s) at {target_sample_rate} Hz). The track conditions the WHOLE "
            f"clip, so a shorter one would leave part of the timeline unconditioned; supply a "
            f"longer track, or shorten the clip.")

    if waveform.shape[0] != 2:
        waveform = waveform.expand(2, -1).contiguous()
    keep = int(math.ceil(required_seconds * sample_rate)) + 2 * int(sample_rate)
    waveform = waveform[:, :keep]

    if sample_rate != target_sample_rate:
        try:
            import torchaudio
        except ImportError as error:  # pragma: no cover - torchaudio is a repo dependency
            raise ImportError(
                f"Resampling an input audio track from {sample_rate} Hz to "
                f"{target_sample_rate} Hz needs torchaudio.") from error
        waveform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)

    if waveform.shape[-1] < required:
        raise ValueError(
            f"the input audio track is shorter than the clip: it resampled to "
            f"{waveform.shape[-1]} sample(s) at {target_sample_rate} Hz where this request needs "
            f"{required} ({required_seconds:.3f}s). Supply a longer track, or shorten the clip.")
    return waveform[:, :required].contiguous()


def normalize_references(
    references: Sequence[MiniMaxH3Reference],
    *,
    num_frames: int,
    fps: float = 24.0,
    audio_sample_rate: int = 32000,
    reference_image_short_edge: int = REFERENCE_IMAGE_SHORT_EDGE,
    image_canvas: Optional[Tuple[int, int]] = None,
    multiple: int = CANVAS_MULTIPLE,
) -> List[MiniMaxH3Reference]:
    """Every reference onto MiniMax-H3's own rates and resolutions, IN ORDER.

    ``image_canvas`` selects the image sizing (see
    :func:`normalize_reference_image`): ``None`` is the released 2048-short-edge
    recipe, a ``(height, width)`` pair is ComfyUI's "match the generation area".
    Every soundtrack is truncated to the generated duration, because that is what
    the packed layout can hold rows for.
    """
    max_duration = num_frames / fps
    normalized: List[MiniMaxH3Reference] = []
    for index, reference in enumerate(references):
        label = reference.label or f"reference {index + 1}"
        waveform = None
        if reference.has_audio:
            source_rate = reference.sample_rate or audio_sample_rate
            waveform = normalize_reference_audio(
                reference.audio, int(source_rate),
                target_sample_rate=audio_sample_rate, max_duration=max_duration)

        if reference.kind == "image":
            normalized.append(MiniMaxH3Reference(
                kind="image",
                image=normalize_reference_image(
                    reference.image, reference_short_edge=reference_image_short_edge,
                    multiple=multiple, canvas=image_canvas),
                label=label,
            ))
        elif reference.kind == "video":
            frames = normalize_reference_video(
                reference.frames, float(reference.fps or fps), num_frames,
                target_fps=fps, multiple=multiple, canvas=reference.video_canvas)
            normalized.append(MiniMaxH3Reference(
                kind="video", frames=frames, fps=fps, audio=waveform,
                sample_rate=None if waveform is None else audio_sample_rate,
                video_canvas=(frames.shape[1], frames.shape[2]), label=label))
        elif reference.kind == "audio":
            normalized.append(MiniMaxH3Reference(
                kind="audio", audio=waveform, sample_rate=audio_sample_rate, label=label))
        else:
            raise ValueError(
                f"A MiniMax-H3 reference must be an 'image', a 'video' or an 'audio', got "
                f"{reference.kind!r}.")
    return normalized


# ---------------------------------------------------------------------------
# VAE encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_reference_visuals(
    vae,
    references: Sequence[MiniMaxH3Reference],
    *,
    latents_mean: Sequence[float],
    latents_std: Sequence[float],
    pixel_mean: Sequence[float],
    pixel_std: Sequence[float],
    device: torch.device | str = "cuda",
    encode_seed: int = 42,
) -> List[torch.Tensor]:
    """The video-VAE conditioning latents of the image and video references.

    One ``[1, C, T_lat, H_lat, W_lat]`` tensor per VISUAL reference, in packed
    order, skipping audio-only ones. An image is a single frame and goes through
    the spatial encoder alone; a video is snapped down onto the ``17n+5`` grid
    first so the temporal chunking needs no padding.

    Shares :func:`h3_pipeline_ops.encode_visual_condition`'s recipe with the
    ``fl2va`` keyframes -- ImageNet-normalised pixels, a posterior SAMPLED under
    the fixed ``encode_seed`` (not the request's generator), rounded through
    float16, then per-channel normalised.
    """
    from .h3_pipeline_ops import encode_visual_condition

    latents: List[torch.Tensor] = []
    for reference in references:
        if reference.kind == "image":
            pixels = np.asarray(reference.image, dtype=np.uint8)[None]
        elif reference.kind == "video":
            frames = reference.frames
            pixels = frames[:snap_reference_video_frames(int(frames.shape[0]))]
        else:
            continue
        latents.append(encode_visual_condition(
            vae, pixels,
            latents_mean=latents_mean, latents_std=latents_std,
            pixel_mean=pixel_mean, pixel_std=pixel_std,
            device=device, encode_seed=encode_seed,
        ))
    return latents


@torch.no_grad()
def encode_reference_audio_rows(
    audio_vae,
    references: Sequence[MiniMaxH3Reference],
    *,
    latents_mean: Sequence[float],
    latents_std: Sequence[float],
    audio_latent_channels: int = 32,
    device: torch.device | str = "cuda",
) -> List[torch.Tensor]:
    """The audio-VAE conditioning rows of every audio-bearing reference.

    One ``[num_audio_latents * 2, audio_latent_channels]`` CHANNEL-MAJOR tensor
    per reference that carries sound (a standalone audio reference, or a video
    reference's soundtrack), in packed order -- one entry each rather than one
    concatenated block, because the packed layout is built from the row count of
    each.

    Two conventions that differ from the visual side and are easy to carry over
    by mistake: the autoencoder is MONO, so the two stereo channels are two
    BATCH items; and a soundtrack takes the posterior **mode**, never a sample,
    and is never noised -- it conditions clean, pinned at ``t = 1.0``.
    """
    mean = torch.tensor(list(latents_mean)).view(1, 1, -1)
    std = torch.tensor(list(latents_std)).view(1, 1, -1)
    rows: List[torch.Tensor] = []
    for reference in references:
        if not reference.has_audio:
            continue
        waveform = reference.audio.to(device)[:, None]
        posterior = audio_vae.encode(waveform, return_dict=False)[0]
        # [ch, C, T] -> [ch, T, C] -> channel-major rows [ch * T, C]
        latents = posterior.mode().float().cpu().transpose(1, 2)
        rows.append(((latents - mean) / std).reshape(-1, audio_latent_channels))
    return rows


# ---------------------------------------------------------------------------
# The conditioner's presentation
# ---------------------------------------------------------------------------

def sample_reference_video_blocks(
    frames: np.ndarray,
    *,
    fps: float,
    sample_fps: float,
    temporal_patch: int,
) -> Tuple[List[np.ndarray], List[float]]:
    """The frames the CONDITIONER sees, and the timestamp labelling each block.

    Every ``fps / sample_fps``-th frame, deduplicated; Qwen3-VL then merges the
    sampled frames in groups of ``temporal_patch`` (repeating the last one when
    the count does not divide), and a merged group is labelled with the MEAN of
    its timestamps. ``"{:.1f}"`` rounds half to even, so the first block of a
    2 fps pair renders as ``"<0.2 seconds>"`` rather than ``"<0.3 seconds>"``.
    """
    stride = fps / sample_fps
    indices: List[int] = []
    cursor = 0.0
    while round(cursor) < frames.shape[0]:
        if not indices or round(cursor) > indices[-1]:
            indices.append(round(cursor))
        cursor += stride
    if len(indices) < temporal_patch:
        minimum = round((temporal_patch - 1) * stride) + 1
        raise ValueError(
            f"A reference video is read at {sample_fps:g} fps and its sampled frames are merged in "
            f"groups of {temporal_patch}, so it must run at least {minimum} frames at {fps:g} fps "
            f"({minimum / fps:.2g} s), got {frames.shape[0]}.")

    timestamps = [index / sample_fps for index in range(len(indices))]
    timestamps += [timestamps[-1]] * (-len(timestamps) % temporal_patch)
    block_timestamps = [
        (timestamps[index] + timestamps[index + temporal_patch - 1]) / 2
        for index in range(0, len(timestamps), temporal_patch)
    ]
    return [frames[index] for index in indices], block_timestamps


def build_ref2va_presentation(
    tokenizer,
    processor,
    prompt: str,
    references: Sequence[MiniMaxH3Reference],
    *,
    fps: float = 24.0,
    sample_fps: float = REFERENCE_VIDEO_SAMPLE_FPS,
    text_tag: int = 1,
    video_tag: int = 0,
) -> Tuple[List[int], List[int], Dict[str, torch.Tensor]]:
    """``(token_ids, per-row modality tags, vision tensors)`` of a ref2va request.

    Every reference prepends a label, numbered PER MODALITY and emitted in
    packed order: ``"<Picture i>: "`` plus a vision block for an image,
    ``"<Audio j>: "`` alone for audio (a waveform never reaches the
    conditioner), and ``"<Video k>: "`` plus one timestamped vision block per
    merged frame pair for a video. A video that carries sound is labelled
    ``"<Audio j>: "`` BEFORE ``"<Video k>: "``, mirroring the order its rows are
    packed in. The prompt follows verbatim -- no chat template, no special
    tokens.

    The rows of a vision block are tagged VIDEO rather than text: that tag is
    what the transformer's AdaLN modulation keys off, and it is the reason the
    text span of a ref2va sequence is not uniformly tagged.

    The vision tensors are batched PER MODALITY while the presentation is
    tokenized in request order. The two agree because the filtering preserves
    relative order within each modality and Qwen3-VL fills the n-th pad RUN of a
    modality with the n-th entry of that modality's batch.
    """
    if processor is None:
        raise RuntimeError(
            "MiniMax-H3 ref2va needs the Qwen3-VL processor (official/processor/): it is what turns "
            "a reference image or video into the vision tensors the conditioner reads.")

    merge_size = processor.image_processor.merge_size ** 2
    vision_inputs: Dict[str, torch.Tensor] = {}

    image_token_counts: List[int] = []
    images = [reference.image for reference in references if reference.kind == "image"]
    if images:
        features = processor.image_processor(images=images, return_tensors="pt")
        vision_inputs["pixel_values"] = features["pixel_values"]
        vision_inputs["image_grid_thw"] = features["image_grid_thw"]
        image_token_counts = [int(grid.prod()) // merge_size for grid in features["image_grid_thw"]]

    video_token_counts: List[int] = []
    video_timestamps: List[List[float]] = []
    videos = [reference for reference in references if reference.kind == "video"]
    if videos:
        temporal_patch = processor.video_processor.temporal_patch_size
        sampled = [
            sample_reference_video_blocks(
                reference.frames, fps=fps, sample_fps=sample_fps, temporal_patch=temporal_patch)
            for reference in videos
        ]
        video_timestamps = [timestamps for _frames, timestamps in sampled]
        features = processor.video_processor(
            videos=[np.stack(frames) for frames, _timestamps in sampled],
            do_sample_frames=False, return_tensors="pt")
        vision_inputs["pixel_values_videos"] = features["pixel_values_videos"]
        vision_inputs["video_grid_thw"] = features["video_grid_thw"]
        video_token_counts = [
            int(grid[1]) * int(grid[2]) // merge_size for grid in features["video_grid_thw"]]
        for timestamps, grid in zip(video_timestamps, features["video_grid_thw"]):
            if int(grid[0]) != len(timestamps):
                raise RuntimeError(
                    f"The processor merged a reference video into {int(grid[0])} vision block(s) but "
                    f"MiniMax-H3 labels {len(timestamps)} of them.")

    token_ids: List[int] = []
    token_tags: List[int] = []

    def emit_text(value: str) -> None:
        ids = tokenizer(value, add_special_tokens=False)["input_ids"]
        token_ids.extend(ids)
        token_tags.extend([text_tag] * len(ids))

    def emit_vision(pad_token: str, num_tokens: int) -> None:
        ids = ([tokenizer.convert_tokens_to_ids("<|vision_start|>")]
               + [tokenizer.convert_tokens_to_ids(pad_token)] * num_tokens
               + [tokenizer.convert_tokens_to_ids("<|vision_end|>")])
        token_ids.extend(ids)
        token_tags.extend([video_tag] * len(ids))

    counts = {"image": 0, "video": 0, "audio": 0}
    for reference in references:
        if reference.has_audio:
            counts["audio"] += 1
            emit_text(f"<Audio {counts['audio']}>: ")
        if reference.kind == "image":
            counts["image"] += 1
            emit_text(f"<Picture {counts['image']}>: ")
            emit_vision("<|image_pad|>", image_token_counts[counts["image"] - 1])
        elif reference.kind == "video":
            counts["video"] += 1
            emit_text(f"<Video {counts['video']}>: ")
            for timestamp in video_timestamps[counts["video"] - 1]:
                emit_text(f"<{timestamp:.1f} seconds>")
                emit_vision("<|video_pad|>", video_token_counts[counts["video"] - 1])
    emit_text(prompt)
    return token_ids, token_tags, vision_inputs


def decode_audio_bytes(data: bytes) -> Tuple[torch.Tensor, int]:
    """An uploaded audio file as ``([channels, samples] float32, sample_rate)``.

    soundfile (libsndfile) first, ``torchaudio.load`` as the fallback -- the
    same two readers, in the same order, that the ACE-Step backend's
    ``_acestep_load_reference_audio`` uses. Nothing is resampled or normalised
    here: MiniMax-H3 resamples a reference onto the audio VAE's own rate later
    (:func:`normalize_reference_audio`), and it needs the file's TRUE rate to do
    that, so the rate is returned rather than assumed.
    """
    import io

    try:
        import soundfile as sf

        samples, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return torch.from_numpy(samples.T).contiguous(), int(sample_rate)
    except Exception as exc:  # noqa: BLE001 - any failure falls through to torchaudio
        print(f"[MiniMax-H3] soundfile could not read a reference audio "
              f"({type(exc).__name__}: {exc}); trying torchaudio")

    import torchaudio

    waveform, sample_rate = torchaudio.load(io.BytesIO(data))
    return waveform.float(), int(sample_rate)


def describe_references(references: Sequence[MiniMaxH3Reference]) -> str:
    """A one-line, log-safe summary of a reference list, in packed order."""
    parts = []
    for reference in references:
        if reference.kind == "image" and reference.image is not None:
            parts.append(f"image {reference.image.size[0]}x{reference.image.size[1]}")
        elif reference.kind == "video" and reference.frames is not None:
            frames = reference.frames
            parts.append(
                f"video {frames.shape[2]}x{frames.shape[1]}x{frames.shape[0]}"
                + ("+audio" if reference.has_audio else ""))
        elif reference.kind == "audio" and reference.audio is not None:
            parts.append(f"audio {reference.audio.shape[-1]} sample(s)")
        else:
            parts.append(reference.kind)
    return ", ".join(parts)
