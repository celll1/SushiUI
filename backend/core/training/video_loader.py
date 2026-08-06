"""
Video clip loader for temporal (video) training (P4b).

Decodes a fixed-length clip of frames from a video file via ``cv2.VideoCapture``
and returns a ``[T, C, H, W]`` float tensor normalised to match the image
VAE-encode path used elsewhere in training (``(x/255 - 0.5) * 2`` -> ``[-1, 1]``,
RGB channel order).

Design constraints (see ltx2_video_dataset_spec.md, P4b):
  - Decoder is cv2 ONLY (no new pip dependency; decord/av unreliable on Windows).
  - ``clip_length`` must be valid for the ARCH's ``TemporalSpec`` — LTX-2.3's
    ``8*k + 1`` (1, 9, 17, 25, ...) when no spec is passed, MiniMax-H3's
    ``17*n + 5`` with a 22-frame decodable floor when its spec is.
  - Frames are sampled either by INDEX (``start_frame + i*stride``, the LTX-2.3
    rule, source fps inherited) or by TIMESTAMP (the source frame nearest
    ``start_time + i*stride/fps_fixed``) for an arch with a fixed frame rate.
    Each frame is resized to ``(target_w, target_h)``.
  - Robust to short videos: if the requested window does not fit, the last valid
    frame is repeated (loop-last-frame) and a note is logged once.
  - Non-ASCII paths: cv2.VideoCapture on Windows/ffmpeg can fail to open unicode
    paths; fall back to a temp ASCII copy (analogous to P4a's imencode fallback).

Phase 6a (MiniMax-H3): every public function takes an optional
``spec: TemporalSpec``. ``spec=None`` is LTX-2.3 and behaves exactly as this
module always has — the golden suite ``backend/tests/temporal_bucketing_test.py``
pins that, cache digests included.
"""

import os
import shutil
import tempfile
from typing import List, NamedTuple, Optional

import torch

from core.models.components.wiring import LTX2_TEMPORAL, TemporalSpec
from core.training.bucketing import (
    clip_cache_key_extras,
    clip_span,
    is_valid_clip_length,
)


# LTX temporal compression: a clip of ``L`` frames encodes to ``(L-1)//8 + 1``
# latent frames. Valid pixel clip lengths are ``8*k + 1``. Retained as the
# ``spec=None`` fallback constant.
LTX_TEMPORAL_COMPRESSION = 8


def _spec_or_ltx(spec: Optional[TemporalSpec]) -> TemporalSpec:
    return spec if spec is not None else LTX2_TEMPORAL


def clip_latent_frames(clip_length: int, spec: Optional[TemporalSpec] = None) -> int:
    """Latent temporal length for a pixel ``clip_length``.

    LTX-2.3 (``spec=None``): ``(L-1)//8 + 1``. MiniMax-H3:
    ``ceil(T/17)*5 - 3`` (measured; ComfyUI's own formula agrees only ON the
    grid). The closed form lives on the spec, not here.
    """
    return int(_spec_or_ltx(spec).latent_frames(int(clip_length)))


def expected_audio_rows(clip_length: int, stride: int = 1,
                        spec: Optional[TemporalSpec] = None,
                        latents_per_second: float = 40.0,
                        channels: int = 2) -> int:
    """Rows a full audio window for this clip should have, or 0 when the arch has
    no fixed rate (and therefore no window-level audio latent).

    ``2 * round(T / fps * latents_per_second)`` -- the closed form measured on
    MiniMax-H3 (22 -> 37 per channel, 39 -> 65), stated here so a short read can
    be RECOGNISED as short instead of silently becoming the batch's row count.
    """
    sp = _spec_or_ltx(spec)
    duration = sp.clip_duration(clip_length, stride)
    if duration is None:
        return 0
    return int(round(duration * float(latents_per_second))) * int(channels)


class ClipWindow(NamedTuple):
    """A sampled clip window, addressed BOTH ways.

    ``start_frame`` is the source frame index the window begins at (what the
    index-sampled LTX-2.3 path uses and what the cache key has always carried);
    ``start_time`` is the same instant in seconds, which is the only addressing
    unit that survives resampling — a 24 fps target window can begin between two
    source frames of a 30 fps video.

    It is a NamedTuple so ``start, _ = sample_clip_window(...)`` and
    ``window.start_frame`` both read naturally.
    """

    start_frame: int
    start_time: float


def sample_clip_window(
    num_frames: int,
    clip_length: int,
    stride: int = 1,
    training: bool = True,
    spec: Optional[TemporalSpec] = None,
    source_fps: Optional[float] = None,
) -> ClipWindow:
    """Choose a window so a ``clip_length``-frame clip (with ``stride``) fits
    inside a video of ``num_frames`` frames.

    The clip spans ``clip_span(clip_length, stride, spec, source_fps)`` SOURCE
    frames — for a fixed-fps arch that is a duration converted into source
    frames, not the frame count itself. The returned start is clamped so
    ``start + span <= num_frames`` when possible.

    Args:
        num_frames: Total frames in the source video.
        clip_length: Number of frames to sample.
        stride: Gap between consecutively sampled frames (>= 1).
        training: Random start when True, centered start when False (val).
        spec: Per-arch temporal spec; None = LTX-2.3.
        source_fps: Source frame rate. Required to express the start in seconds
            and to size the span for a fixed-fps arch; without it ``start_time``
            is 0.0 and the span falls back to the index form.

    Returns:
        ``ClipWindow(start_frame, start_time)``. If the window cannot fit
        (short video), ``(0, 0.0)``; ``load_clip`` then loop-pads the tail.
    """
    stride = max(1, int(stride))
    clip_length = max(1, int(clip_length))
    num_frames = max(0, int(num_frames))

    span = clip_span(clip_length, stride, spec, source_fps)
    max_start = num_frames - span

    if max_start <= 0:
        # Window does not fit (or exactly fills): start at 0 (load_clip loop-pads).
        return ClipWindow(0, 0.0)

    if training:
        import random
        start_frame = random.randint(0, max_start)
    else:
        # Validation: centered window.
        start_frame = max_start // 2
    start_time = (start_frame / float(source_fps)) if source_fps else 0.0
    return ClipWindow(start_frame, start_time)


def plan_source_indices(
    clip_length: int,
    start_frame: int = 0,
    stride: int = 1,
    spec: Optional[TemporalSpec] = None,
    start_time: Optional[float] = None,
    source_fps: Optional[float] = None,
    num_frames: Optional[int] = None,
) -> List[int]:
    """The SOURCE frame indices ``load_clip`` will decode. Pure function.

    Index policy (LTX-2.3, ``spec.fps_fixed is None``):
        ``start_frame + i*stride`` — unchanged, and the clip inherits the
        source's frame rate.

    Timestamp policy (``spec.fps_fixed`` set, e.g. MiniMax-H3's 24.0):
        target frame ``i`` lives at ``start_time + i*stride/fps_fixed`` seconds;
        the chosen source frame is the one whose timestamp is NEAREST that
        instant (``round(t * source_fps)``), clamped into the video. Nearest
        frame, not interpolation: it matches the quality tradeoff the index path
        already makes, and no optical flow is invented.

        Sources SLOWER than the target repeat frames by the same rule; sources
        faster drop them (30 fps -> [0,1,2,4,5,6,8,...] for a 24 fps target).

        Falls back to the index policy when ``source_fps`` is unknown — there is
        nothing to resample against, and silently pretending otherwise is how a
        window gets mislabelled.
    """
    clip_length = max(1, int(clip_length))
    stride = max(1, int(stride))
    start_frame = max(0, int(start_frame))
    sp = _spec_or_ltx(spec)

    if sp.fps_fixed is None or not source_fps:
        return [start_frame + i * stride for i in range(clip_length)]

    t0 = float(start_time) if start_time is not None else (start_frame / float(source_fps))
    hi = (int(num_frames) - 1) if num_frames else None
    out: List[int] = []
    for i in range(clip_length):
        t = t0 + (i * stride) / float(sp.fps_fixed)
        j = int(round(t * float(source_fps)))
        if j < 0:
            j = 0
        if hi is not None and j > hi:
            j = hi
        out.append(j)
    return out


def _open_capture(video_path: str):
    """Open a cv2.VideoCapture, with a temp ASCII-copy fallback for unicode paths.

    Returns ``(cap, tmp_path_or_None)``. Caller must release ``cap`` and, if
    ``tmp_path`` is not None, remove that temp file.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        return cap, None

    # Fallback: some cv2/ffmpeg builds cannot open non-ASCII paths. Copy to a
    # temp ASCII path and open that (mirrors P4a's non-ASCII handling intent).
    try:
        cap.release()
    except Exception:
        pass
    try:
        ext = os.path.splitext(video_path)[1] or ".mp4"
        fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix="clip_")
        os.close(fd)
        shutil.copyfile(video_path, tmp_path)
        cap = cv2.VideoCapture(tmp_path)
        if cap.isOpened():
            return cap, tmp_path
        try:
            cap.release()
        except Exception:
            pass
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    except Exception as e:  # noqa: BLE001
        print(f"[VideoLoader] non-ASCII fallback failed for {video_path}: {e}")

    return None, None


def load_clip(
    video_path: str,
    clip_length: int,
    start_frame: int = 0,
    stride: int = 1,
    target_w: int = 512,
    target_h: int = 512,
    spec: Optional[TemporalSpec] = None,
    start_time: Optional[float] = None,
    source_fps: Optional[float] = None,
) -> torch.Tensor:
    """Decode a ``clip_length``-frame clip and return a ``[T, C, H, W]`` tensor.

    Source frames are chosen by ``plan_source_indices`` — by INDEX for LTX-2.3
    (``start_frame + i*stride``) or by TIMESTAMP for a fixed-fps arch — and each
    is resized to ``(target_w, target_h)``, converted BGR->RGB, and normalised to
    ``[-1, 1]`` (``(x/255 - 0.5) * 2``) to match the image VAE-encode path
    (base_trainer.encode_image). Archs whose VAE wants a different pixel
    convention (MiniMax-H3 wants ImageNet-normalised RGB over [0,1]) convert
    inside their own ``vae_encode_clip``; this contract does not change.

    Short-video robustness: if a requested source frame is past the end (or a
    read fails), the last successfully decoded frame is repeated so the returned
    tensor always has exactly ``clip_length`` frames. A note is logged once.

    Args:
        video_path: Path to the video file.
        clip_length: Number of frames (must be valid for ``spec``).
        start_frame: First source frame index to sample.
        stride: Gap between sampled frames (>= 1).
        target_w: Output width (callers pass ÷32-aligned dims).
        target_h: Output height.
        spec: Per-arch temporal spec; None = LTX-2.3 (``8*k+1``, index sampling).
        start_time: Window start in seconds, for the timestamp path. Defaults to
            ``start_frame / source_fps``.
        source_fps: Source frame rate. Read from the container when omitted (the
            timestamp path needs it; the index path ignores it).

    Returns:
        Float tensor ``[T=clip_length, C=3, H=target_h, W=target_w]`` in [-1, 1].

    Raises:
        RuntimeError: if the video cannot be opened or no frame can be decoded.
        ValueError: if ``clip_length`` is not valid for ``spec``.
    """
    try:
        import cv2
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"[VideoLoader] cv2 unavailable: {e}")

    _spec = _spec_or_ltx(spec)
    if not is_valid_clip_length(clip_length, spec):
        raise ValueError(
            f"[VideoLoader] clip_length must be {_spec.frame_multiple}*k+"
            f"{_spec.frame_offset} and >= {_spec.min_decodable_frames}, "
            f"got {clip_length}"
        )

    clip_length = int(clip_length)
    stride = max(1, int(stride))
    start_frame = max(0, int(start_frame))
    target_w = int(target_w)
    target_h = int(target_h)

    cap, tmp_path = _open_capture(video_path)
    if cap is None:
        raise RuntimeError(f"[VideoLoader] cannot open video: {video_path}")

    frames = []  # list of [C,H,W] tensors
    last_good = None
    padded = False
    try:
        import numpy as np

        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception:
            total = 0

        # Fixed-fps archs resample against the SOURCE rate; read it from the
        # container when the caller did not pass it (the caller usually has it
        # from the dataset probe, which is more reliable than cv2's).
        eff_source_fps = source_fps
        if _spec.fps_fixed is not None and not eff_source_fps:
            try:
                probed = float(cap.get(cv2.CAP_PROP_FPS))
                eff_source_fps = probed if probed and probed > 0 else None
            except Exception:
                eff_source_fps = None
            if not eff_source_fps:
                print(f"[VideoLoader] WARNING: no source fps for {video_path}; "
                      f"falling back to index sampling (clip will NOT be "
                      f"resampled to {_spec.fps_fixed} fps)")

        # Desired source indices for this clip (index or timestamp policy).
        wanted = plan_source_indices(
            clip_length, start_frame, stride, spec,
            start_time=start_time, source_fps=eff_source_fps,
            num_frames=(total if total > 0 else None),
        )

        for idx in wanted:
            frame = None
            # Seek + read. VP8/webm seeking can be imprecise; if the read fails
            # we fall back to loop-last-frame below.
            if total <= 0 or idx < total:
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ok, bgr = cap.read()
                    if ok and bgr is not None:
                        frame = bgr
                except Exception:
                    frame = None

            if frame is None:
                # Short video / unreadable frame: repeat last good frame.
                if last_good is not None:
                    frames.append(last_good.clone())
                    padded = True
                    continue
                else:
                    # No frame decoded yet; try a plain sequential read as a
                    # last resort before giving up.
                    try:
                        ok, bgr = cap.read()
                        if ok and bgr is not None:
                            frame = bgr
                    except Exception:
                        frame = None
                    if frame is None:
                        continue

            # BGR -> RGB, resize, normalise to [-1, 1].
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if (rgb.shape[1], rgb.shape[0]) != (target_w, target_h):
                rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
            arr = rgb.astype(np.float32) / 255.0
            arr = (arr - 0.5) * 2.0  # [-1, 1]
            t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [C,H,W]
            frames.append(t)
            last_good = t
    finally:
        try:
            cap.release()
        except Exception:
            pass
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    if len(frames) == 0:
        raise RuntimeError(
            f"[VideoLoader] decoded 0 frames from {video_path} "
            f"(start={start_frame}, len={clip_length}, stride={stride})"
        )

    # Ensure exactly clip_length frames (loop-pad any shortfall with last frame).
    if len(frames) < clip_length:
        padded = True
        while len(frames) < clip_length:
            frames.append(frames[-1].clone())
    elif len(frames) > clip_length:
        frames = frames[:clip_length]

    if padded:
        print(
            f"[VideoLoader] NOTE: loop-padded short clip for {video_path} "
            f"(start={start_frame}, len={clip_length}, stride={stride}); "
            f"tail frames repeated."
        )

    clip = torch.stack(frames, dim=0)  # [T, C, H, W]
    return clip


def encode_and_cache_clip(
    *,
    cache,
    video_path: str,
    width: int,
    height: int,
    clip_start: int,
    clip_length: int,
    stride: int,
    vae_encode_clip,
    fps: Optional[float] = None,
    device: str = "cuda",
    skip_existing: bool = True,
    spec: Optional[TemporalSpec] = None,
    start_time: Optional[float] = None,
    source_fps: Optional[float] = None,
    tiling_policy: Optional[str] = None,
    audio_prep_version: Optional[str] = None,
    audio_encode_window=None,
    return_record: bool = False,
):
    """Encode-integration SEAM for P5 (ltx2 arch handler).

    This is the single clean entry point P5 calls to turn a video-clip window
    into a cached 5D latent. It deliberately does NOT load or reference the LTX
    VAE (that would collide with the running backend's GPU/model state) — P5
    passes ``vae_encode_clip``, a callable it owns that runs the loaded LTX
    video VAE.

    Flow:
      1. Cache hit -> return the cached 5D latent ``[1, C, T, H', W']``.
      2. Miss -> ``load_clip(...) -> [T, C, H, W]`` (RGB, [-1, 1]).
      3. ``vae_encode_clip(clip)`` -> 5D latent ``[1, C, T, H', W']`` (P5 owns
         the pixel layout adaptation, e.g. ``[1, C, T, H, W]`` permute, and the
         LTX latent-mean/std normalisation).
      4. ``cache.save_clip_latent(...)`` and return the latent.

    Expected ``vae_encode_clip`` signature (P5 implements):
        ``vae_encode_clip(clip: torch.Tensor[T, C, H, W]) -> torch.Tensor[1, C, T, H', W']``

    Args:
        cache: A ``LatentCache`` instance.
        video_path/width/height: Source + target dims.
        clip_start/clip_length/stride: Clip window (part of the cache key).
        vae_encode_clip: P5-owned callable performing the LTX VAE encode.
        fps: Source fps (folded into the cache key when provided).
        device: Device to load a cache-hit latent onto.
        skip_existing: Passed through to ``save_clip_latent``.
        spec: Per-arch temporal spec; None = LTX-2.3 (index sampling, and the
            cache key is byte-identical to the pre-Phase-6a one).
        start_time: Window start in seconds (fixed-fps archs).
        source_fps: Source frame rate (fixed-fps archs). Defaults to ``fps``.
        tiling_policy: VAE spatial-tiling token. MUST identify the same policy
            the arch uses at GENERATION time — the K0.5 / Phase 0T measurements
            show flipping it changes the latents by far more than any tolerance
            in this integration, so a cache built under one policy is not
            interchangeable with the other.
        audio_prep_version: Audio preprocessing token (Phase 6b's window-level
            video+audio record).
        audio_encode_window: OPTIONAL callable ``(start_sec, duration_sec) ->
            audio latent or None``, making the written record WINDOW-LEVEL (video
            AND audio under one key) instead of video-only. It is called with the
            window's own start time and the arch's own clip duration, i.e. the
            SAME timestamps the frames were sampled at, which is what makes A/V
            alignment a property of the construction. ``None`` (every LTX-2.3
            call) writes exactly the record this function always wrote.
        return_record: Return the whole record dict
            ``{"latents", "audio_latents", "has_audio"}`` instead of just the 5D
            video latent. Off by default so existing callers are unchanged.

    Returns:
        5D latent tensor ``[1, C, T, H', W']``, or the record dict when
        ``return_record``.
    """
    _spec = _spec_or_ltx(spec)
    eff_source_fps = source_fps if source_fps is not None else fps
    # Derived by `bucketing.clip_cache_key_extras`, the ONE place the policy half
    # of a clip cache key is built -- shared with `VideoBucketManager.
    # clip_cache_params` so the two can never disagree about which policy fields
    # belong in the key. Empty for an index-sampled arch with no tiling/audio
    # policy, which is what keeps existing LTX-2.3 cache files addressable.
    key_extras = clip_cache_key_extras(
        spec, source_fps=eff_source_fps, start_time=start_time,
        tiling_policy=tiling_policy, audio_prep_version=audio_prep_version,
    )

    if return_record:
        record = cache.load_clip_record(
            video_path, width, height, clip_start, clip_length, stride, fps,
            device=device, **key_extras,
        )
        if record is not None and record.get("latents") is not None:
            return record
    else:
        cached = cache.load_clip_latent(
            video_path, width, height, clip_start, clip_length, stride, fps,
            device=device, **key_extras,
        )
        if cached is not None:
            return cached

    clip = load_clip(
        video_path, clip_length, clip_start, stride, target_w=width, target_h=height,
        spec=spec, start_time=start_time, source_fps=eff_source_fps,
    )  # [T, C, H, W]

    latents = vae_encode_clip(clip)  # P5-owned; -> [1, C, T, H', W']

    if latents.dim() != 5:
        raise ValueError(
            f"[VideoLoader] vae_encode_clip must return a 5D latent "
            f"[1, C, T, H', W'], got {latents.dim()}D {tuple(latents.shape)}"
        )

    # The audio half of the SAME window, cut by the same timestamps.
    #
    # Sized by `TemporalSpec.clip_duration` (`clip_length*stride / fps_fixed`, the
    # duration the clip occupies on the OUTPUT timeline) because that is what the
    # audio latent count is defined against: `T_aud = round(T/fps*40)`, measured.
    # Sizing it from `bucketing.clip_span` instead -- which measures first sample
    # to LAST sample, i.e. `clip_length - 1` gaps -- would return fewer rows than
    # the packed layout's geometry contract calls for.
    #
    # The two agree exactly at stride 1 on a source already at `fps_fixed`
    # (22 frames: span 22 source frames, duration 22/24 s = 22 source frames).
    # They can differ by up to ONE target frame when the source is RESAMPLED
    # (22 frames of a 30 fps source: span reserves 27, the duration wants 27.5) or
    # when stride > 1. `sample_clip_window` places windows using the span, so a
    # window sitting hard against the end of a source can ask for a fraction of a
    # frame of audio past EOF; ffmpeg then returns a short read and the record
    # holds fewer than `2*round(T/fps*40)` rows.
    #
    # Deliberately NOT shortened to fit (that would desynchronise the audio from
    # the frames it was cut with) and deliberately NOT raised (one edge window
    # must not kill a run). The short read is reported with its numbers, and the
    # collation zero-pads it to the batch shape while still reporting the sample
    # as having audio -- so a systematically short dataset is visible in the log
    # instead of being silently absorbed.
    audio_latents = None
    has_audio = None
    if audio_encode_window is not None:
        duration = _spec.clip_duration(clip_length, stride)
        if duration is None:
            raise ValueError(
                "[VideoLoader] audio_encode_window needs a fixed-frame-rate arch: "
                "clip_duration is undefined when spec.fps_fixed is None")
        start_sec = float(start_time) if start_time is not None else (
            (clip_start / float(eff_source_fps)) if eff_source_fps else 0.0)
        audio_latents = audio_encode_window(start_sec, float(duration))
        has_audio = audio_latents is not None
        if audio_latents is not None:
            expected_rows = expected_audio_rows(clip_length, stride, _spec)
            if expected_rows and int(audio_latents.shape[0]) < expected_rows:
                print(f"[VideoLoader] NOTE: audio window for {video_path} "
                      f"(start {start_sec:.3f}s, {duration:.3f}s) returned "
                      f"{int(audio_latents.shape[0])} of {expected_rows} rows -- the "
                      f"requested window runs past the end of the source's audio "
                      f"track. The clip's frames are unaffected; the missing rows "
                      f"are zero-padded at collation.")

    cache.save_clip_latent(
        video_path, width, height, clip_start, clip_length, stride,
        latents, fps=fps, skip_existing=skip_existing,
        audio_latents=audio_latents, has_audio=has_audio, **key_extras,
    )
    if return_record:
        return {"latents": latents, "audio_latents": audio_latents,
                "has_audio": bool(has_audio) if has_audio is not None else None}
    return latents
