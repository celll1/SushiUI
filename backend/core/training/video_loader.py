"""
Video clip loader for temporal (video) training (P4b).

Decodes a fixed-length clip of frames from a video file via ``cv2.VideoCapture``
and returns a ``[T, C, H, W]`` float tensor normalised to match the image
VAE-encode path used elsewhere in training (``(x/255 - 0.5) * 2`` -> ``[-1, 1]``,
RGB channel order).

Design constraints (see ltx2_video_dataset_spec.md, P4b):
  - Decoder is cv2 ONLY (no new pip dependency; decord/av unreliable on Windows).
  - ``clip_length`` must be a valid LTX temporal count: ``8*k + 1`` (1, 9, 17, 25, ...).
  - Frames are sampled starting at ``start_frame`` with ``stride`` between sampled
    frames; each frame is resized to ``(target_w, target_h)`` (callers pass
    ÷32-aligned dims for LTX).
  - Robust to short videos: if the requested window does not fit, the last valid
    frame is repeated (loop-last-frame) and a note is logged once.
  - Non-ASCII paths: cv2.VideoCapture on Windows/ffmpeg can fail to open unicode
    paths; fall back to a temp ASCII copy (analogous to P4a's imencode fallback).
"""

import os
import shutil
import tempfile
from typing import Optional

import torch


# LTX temporal compression: a clip of ``L`` frames encodes to ``(L-1)//8 + 1``
# latent frames. Valid pixel clip lengths are ``8*k + 1``.
LTX_TEMPORAL_COMPRESSION = 8


def is_valid_ltx_clip_length(clip_length: int) -> bool:
    """True if ``clip_length`` is a valid LTX pixel clip length (``8*k + 1``)."""
    try:
        cl = int(clip_length)
    except (TypeError, ValueError):
        return False
    return cl >= 1 and (cl - 1) % LTX_TEMPORAL_COMPRESSION == 0


def ltx_latent_frames(clip_length: int) -> int:
    """Latent temporal length for a pixel ``clip_length``: ``(L-1)//8 + 1``."""
    return (int(clip_length) - 1) // LTX_TEMPORAL_COMPRESSION + 1


def sample_clip_window(
    num_frames: int,
    clip_length: int,
    stride: int = 1,
    training: bool = True,
) -> int:
    """Choose a ``start_frame`` so a ``clip_length``-frame clip (with ``stride``)
    fits inside a video of ``num_frames`` frames.

    The clip spans ``span = (clip_length - 1) * stride + 1`` source frames. The
    returned start is clamped so ``start + span <= num_frames`` when possible.

    Args:
        num_frames: Total frames in the source video.
        clip_length: Number of frames to sample (LTX ``8*k + 1``).
        stride: Gap between consecutively sampled frames (>= 1).
        training: Random start when True, centered start when False (val).

    Returns:
        A non-negative integer start frame index. If the window cannot fit
        (short video), returns 0; ``load_clip`` then loop-pads the tail.
    """
    stride = max(1, int(stride))
    clip_length = max(1, int(clip_length))
    num_frames = max(0, int(num_frames))

    span = (clip_length - 1) * stride + 1
    max_start = num_frames - span

    if max_start <= 0:
        # Window does not fit (or exactly fills): start at 0 (load_clip loop-pads).
        return 0

    if training:
        import random
        return random.randint(0, max_start)
    # Validation: centered window.
    return max_start // 2


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
) -> torch.Tensor:
    """Decode a ``clip_length``-frame clip and return a ``[T, C, H, W]`` tensor.

    Frames are sampled at indices ``start_frame, start_frame+stride, ...`` and
    each is resized to ``(target_w, target_h)``, converted BGR->RGB, and
    normalised to ``[-1, 1]`` (``(x/255 - 0.5) * 2``) to match the image
    VAE-encode path (base_trainer.encode_image).

    Short-video robustness: if a requested source frame is past the end (or a
    read fails), the last successfully decoded frame is repeated so the returned
    tensor always has exactly ``clip_length`` frames. A note is logged once.

    Args:
        video_path: Path to the video file.
        clip_length: Number of frames (must be LTX ``8*k + 1``).
        start_frame: First source frame index to sample.
        stride: Gap between sampled frames (>= 1).
        target_w: Output width (callers pass ÷32-aligned dims for LTX).
        target_h: Output height.

    Returns:
        Float tensor ``[T=clip_length, C=3, H=target_h, W=target_w]`` in [-1, 1].

    Raises:
        RuntimeError: if the video cannot be opened or no frame can be decoded.
        ValueError: if ``clip_length`` is not a valid LTX count.
    """
    try:
        import cv2
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"[VideoLoader] cv2 unavailable: {e}")

    if not is_valid_ltx_clip_length(clip_length):
        raise ValueError(
            f"[VideoLoader] clip_length must be 8*k+1 (LTX), got {clip_length}"
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

        # Desired source indices for this clip.
        wanted = [start_frame + i * stride for i in range(clip_length)]

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

    Returns:
        5D latent tensor ``[1, C, T, H', W']``.
    """
    cached = cache.load_clip_latent(
        video_path, width, height, clip_start, clip_length, stride, fps, device=device
    )
    if cached is not None:
        return cached

    clip = load_clip(
        video_path, clip_length, clip_start, stride, target_w=width, target_h=height
    )  # [T, C, H, W]

    latents = vae_encode_clip(clip)  # P5-owned; -> [1, C, T, H', W']

    if latents.dim() != 5:
        raise ValueError(
            f"[VideoLoader] vae_encode_clip must return a 5D latent "
            f"[1, C, T, H', W'], got {latents.dim()}D {tuple(latents.shape)}"
        )

    cache.save_clip_latent(
        video_path, width, height, clip_start, clip_length, stride,
        latents, fps=fps, skip_existing=skip_existing,
    )
    return latents
