"""
Temporally-consistent clip augmentation for video (LTX-2) training — P4c.

A video clip is a ``[T, C, H, W]`` tensor (from ``video_loader.load_clip``). Any
spatial augmentation MUST be applied IDENTICALLY to every frame of the clip — a
per-frame-independent crop/flip would break temporal coherence and desynchronise
motion from the caption. This module therefore computes ONE plan per clip and
applies it across all T frames at once.

Two axes only (no temporal cropping — all T frames are kept):
  - spatial crop:  a single crop box (in clip-pixel space) sliced from dims (H, W),
                   shared by every frame, then resized to the target bucket size.
  - horizontal flip: a single flip decision, applied to every frame.

The "same transform across all T" property is STRUCTURAL: slicing / flipping /
interpolating over the (C, H, W) dims with T as the leading (batch) dim cannot
differ between frames. ``apply_clip_augment`` is therefore provably identical to
applying the same plan to each frame independently (asserted in the P4c test).

Crop-box aspect logic is shared with the image path via
``crop_planner._max_window_for_aspect`` (largest window of the target aspect that
fits inside the source), so image and video crops stay geometrically consistent.
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from core.training.crop_planner import _max_window_for_aspect


@dataclass(frozen=True)
class ClipAugmentPlan:
    """One augmentation decision for an ENTIRE clip (applied to every frame).

    Attributes:
        crop_box: ``(x, y, w, h)`` crop window in clip-pixel space (dims W, H).
        hflip: If True, horizontally flip every frame.
        target_w, target_h: Output size every frame is resized to (÷32 bucket).
    """
    crop_box: Tuple[int, int, int, int]
    hflip: bool
    target_w: int
    target_h: int


def plan_clip_augment(
    clip_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
    training: bool = True,
    flip_prob: float = 0.5,
    position_mode: str = "random",
    rng: Optional["_random.Random"] = None,
) -> ClipAugmentPlan:
    """Compute a single augmentation plan for a clip.

    Args:
        clip_hw: Source clip ``(H, W)`` (the loaded clip's spatial size).
        target_hw: Desired output ``(H, W)`` (the ÷32 spatial bucket).
        training: Random crop position + random flip when True; centered crop and
            no flip when False (validation) unless ``position_mode`` overrides.
        flip_prob: Probability of a horizontal flip (training only).
        position_mode: ``"random"`` (train default) or ``"center"`` crop placement.
        rng: RNG for reproducibility (defaults to the global ``random`` module).

    Returns:
        A ``ClipAugmentPlan`` (the same plan is applied to every frame by
        ``apply_clip_augment``).
    """
    r = rng if rng is not None else _random
    H, W = int(clip_hw[0]), int(clip_hw[1])
    th, tw = int(target_hw[0]), int(target_hw[1])
    H = max(1, H)
    W = max(1, W)
    th = max(1, th)
    tw = max(1, tw)

    # Largest window of the TARGET aspect that fits inside the source clip.
    cw, ch = _max_window_for_aspect(W, H, tw, th)
    cw = max(1, min(cw, W))
    ch = max(1, min(ch, H))

    mx, my = W - cw, H - ch
    if training and position_mode == "random":
        x = r.randint(0, mx) if mx > 0 else 0
        y = r.randint(0, my) if my > 0 else 0
    else:
        # Centered window (validation / center mode).
        x = mx // 2
        y = my // 2

    hflip = bool(training) and (r.random() < float(flip_prob))

    return ClipAugmentPlan(
        crop_box=(int(x), int(y), int(cw), int(ch)),
        hflip=hflip,
        target_w=tw,
        target_h=th,
    )


def apply_clip_augment(clip: torch.Tensor, plan: ClipAugmentPlan) -> torch.Tensor:
    """Apply ``plan`` to EVERY frame of ``clip`` identically.

    Args:
        clip: ``[T, C, H, W]`` float tensor (from ``load_clip``).
        plan: A ``ClipAugmentPlan`` from ``plan_clip_augment``.

    Returns:
        ``[T, C, target_h, target_w]`` float tensor. The crop, resize and flip are
        applied over the (C, H, W) dims with T as the batch dim, so the transform
        is byte-for-byte identical for every frame.
    """
    if clip.dim() != 4:
        raise ValueError(
            f"[video_augment] apply_clip_augment expects [T, C, H, W], "
            f"got {clip.dim()}D {tuple(clip.shape)}"
        )

    T, C, H, W = clip.shape
    x, y, cw, ch = plan.crop_box
    # Clamp the crop window into bounds (robust to bucket/clip size drift).
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    cw = max(1, min(int(cw), W - x))
    ch = max(1, min(int(ch), H - y))

    # Crop: identical spatial slice for all T frames.
    out = clip[:, :, y:y + ch, x:x + cw]

    # Resize to the target bucket size (T acts as the batch dim -> same per frame).
    if (out.shape[2], out.shape[3]) != (plan.target_h, plan.target_w):
        out = F.interpolate(
            out.float(),
            size=(plan.target_h, plan.target_w),
            mode="bilinear",
            align_corners=False,
        ).to(clip.dtype)

    # Horizontal flip: mirror the W dim for every frame.
    if plan.hflip:
        out = torch.flip(out, dims=[3])

    return out.contiguous()
