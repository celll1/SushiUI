"""
Epoch-dynamic crop & bucketing planner (SDXL only).

For each (item, epoch), deterministically decides how the image is presented this
epoch, along two independent axes:

  - crop axis:   full image (minimal aspect-fitting crop only) vs random crop
  - bucket axis: largest-fitting bucket (least downscale) vs a smaller bucket

This yields a 2x2 mix whose proportions are controlled by two probabilities:

  | (crop \ bucket) | max-fitting bucket | smaller bucket |
  |-----------------|--------------------|----------------|
  | full image      | (1) full -> max    | (3) full -> smaller |
  | random crop     | (2) crop -> max    | (4) crop -> smaller |

  crop_full_image_prob = P(full image)            -> (1)+(3)
  crop_max_bucket_prob = P(largest-fitting bucket) -> (1)+(2)

The decision is a pure function of (seed, epoch, image_path) via an independent
SHA256-seeded RNG, decoupled from the global RNG stream, so a resumed run
regenerates identical crops regardless of where it was interrupted.

SDXL micro-conditioning (time_ids) uses the kohya convention:
    time_ids = [original_h, original_w, crop_top, crop_left, target_h, target_w]
    - original_size = full original image size
    - crop_top_left = crop window top-left in original-image pixels
    - target_size   = output bucket size
For the full-image case at the max bucket this matches standard aspect-ratio
bucketing (cover + minimal crop), so it is backward compatible in spirit.

See docs/EPOCH_DYNAMIC_CROP_BUCKETING_DESIGN.md.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.training.bucketing import (
    BucketManager,
    BucketResolution,
    get_bucket_for_image_size,
    get_bucket_sizes,
)


@dataclass(frozen=True)
class CropSpec:
    """Per-(item, epoch) crop decision.

    Attributes:
        is_full: True if the full image is used (only an aspect-fitting / divisibility
            crop, no content crop).
        crop_box: (cx, cy, cw, ch) crop window in original-image pixels.
        bucket_w, bucket_h: Output bucket size (the encoded latent's pixel target).
        time_ids: kohya-style SDXL time_ids (orig_h, orig_w, crop_top, crop_left,
            target_h, target_w).
        fallback: True if a crop was requested but constraints could not be met and the
            planner fell back to the full image (for logging).
    """
    is_full: bool
    crop_box: Tuple[int, int, int, int]
    bucket_w: int
    bucket_h: int
    time_ids: Tuple[int, int, int, int, int, int]
    fallback: bool = False


def _max_window_for_aspect(ow: int, oh: int, bw: int, bh: int) -> Tuple[int, int]:
    """Largest (cw, ch) of the bucket's aspect ratio that fits inside (ow, oh)."""
    a_b = bw / bh
    if (ow / oh) >= a_b:
        # Original is wider than (or equal to) the bucket aspect -> height-limited.
        ch = oh
        cw = min(ow, int(round(oh * a_b)))
    else:
        # Width-limited.
        cw = ow
        ch = min(oh, int(round(ow / a_b)))
    return max(1, cw), max(1, ch)


class CropPlanner:
    """Deterministic per-epoch crop + bucket planner for SDXL training."""

    def __init__(
        self,
        config: dict,
        base_resolutions: List[int],
        multi_resolution_mode: str = "max",
        divisibility: int = 8,
    ):
        from api.param_defaults import TRAINING_DEFAULTS

        def _cfg(key):
            return config.get(key, TRAINING_DEFAULTS[key])

        self.enable: bool = bool(_cfg("crop_augment_enable"))
        # Mix proportions (the 2x2 axes).
        self.full_image_prob: float = float(_cfg("crop_full_image_prob"))
        self.max_bucket_prob: float = float(_cfg("crop_max_bucket_prob"))
        # Random-crop controls.
        self.min_area_ratio: float = float(_cfg("crop_min_area_ratio"))
        self.min_short_side_px: int = int(_cfg("crop_min_short_side_px"))
        self.aspect_mode: str = str(_cfg("crop_aspect_mode"))          # "source" | "free"
        self.position_mode: str = str(_cfg("crop_position_mode"))      # "random" | "corner"
        # Smaller-bucket controls.
        self.smaller_bucket_mode: str = str(_cfg("crop_smaller_bucket_mode"))  # "base_res" | "scale_range"
        _ssr = _cfg("crop_smaller_scale_range")
        self.smaller_scale_lo: float = float(_ssr[0])
        self.smaller_scale_hi: float = min(1.0, float(_ssr[1]))
        # Full-image (minimal crop) position.
        self.full_position_mode: str = str(_cfg("full_crop_position_mode"))   # "center" | "fixed_corner" | "random"
        # Conditioning + seed.
        self.microcond_mode: str = str(_cfg("crop_microcond_mode"))

        seed = int(_cfg("crop_plan_seed"))
        if seed == 0:
            seed = int(config.get("seed", 0) or 0)
        self.seed: int = seed

        self.base_resolutions = sorted(base_resolutions)
        self.multi_resolution_mode = multi_resolution_mode
        self.divisibility = divisibility

        # Internal BucketManager only for its precomputed per-resolution bucket lists.
        self._bm = BucketManager(
            base_resolutions=self.base_resolutions,
            divisibility=divisibility,
            strategy="crop",
            multi_resolution_mode=multi_resolution_mode,
        )

        # Cached per-epoch batch counts (filled by precompute()).
        self._batches_per_epoch: List[int] = []

    # ------------------------------------------------------------------ RNG
    def _item_rng(self, epoch: int, image_path: str) -> random.Random:
        """Independent RNG seeded by (seed, epoch, image_path). Pure function -> resume
        regenerates identical crops regardless of interruption point."""
        h = hashlib.sha256(f"{self.seed}|{epoch}|{image_path}".encode("utf-8")).digest()
        return random.Random(int.from_bytes(h[:8], "big"))

    # --------------------------------------------------------- bucket selection
    def _select_bucket(self, rw: int, rh: int, use_max: bool, rng: random.Random) -> BucketResolution:
        """Pick the output bucket for a region of pixel size (rw, rh).

        use_max=True  -> largest base-resolution bucket the region fits without upscaling
                         (least downscale; native when the region already <= a bucket).
        use_max=False -> a smaller bucket: a smaller base_resolution (base_res mode) or a
                         quantized downscale of the max bucket (scale_range mode/fallback).
        """
        cand = []  # (res, bucket, scale)  scale > 1 means upscaling
        for res in self.base_resolutions:
            bl = self._bm.bucket_lists[res]
            b = get_bucket_for_image_size(rw, rh, bl, divisibility=self.divisibility)
            scale = max(b.width / rw, b.height / rh)
            cand.append((res, b, scale))

        fitting = [(res, b) for (res, b, s) in cand if s <= 1.0 + 1e-6]
        if not fitting:
            # Region smaller than every bucket -> least upscaling is the only option;
            # max and smaller collapse to it.
            _, b, _ = min(cand, key=lambda c: c[2])
            return b

        by_res = {res: b for (res, b) in fitting}
        max_res = max(by_res)
        if use_max:
            return by_res[max_res]

        # Smaller bucket.
        if self.smaller_bucket_mode == "base_res":
            smaller = sorted(res for res in by_res if res < max_res)
            if smaller:
                chosen = smaller[rng.randrange(len(smaller))]
                return by_res[chosen]
            # No smaller base_resolution available -> fall through to scale_range.

        # scale_range (explicit, or base_res fallback when single base resolution).
        d = rng.uniform(self.smaller_scale_lo, self.smaller_scale_hi)
        eff = max(self.divisibility, int(round(max_res * d / 64)) * 64)
        return get_bucket_for_image_size(
            rw, rh, get_bucket_sizes(eff, self.divisibility), divisibility=self.divisibility
        )

    # ----------------------------------------------------------------- windows
    def _sample_crop_window(self, rng: random.Random, ow: int, oh: int) -> Optional[Tuple[int, int]]:
        """Sample a random crop window (cw, ch) satisfying min area + min short side.
        Returns None if the image cannot hold a valid window (caller -> full fallback)."""
        if min(ow, oh) < self.min_short_side_px:
            return None
        img_area = ow * oh
        area_min = max(self.min_area_ratio * img_area, float(self.min_short_side_px) ** 2)
        area_max = float(img_area)
        if area_min > area_max:
            return None
        A = rng.uniform(area_min, area_max)

        if self.aspect_mode == "free":
            ms2 = float(self.min_short_side_px) ** 2
            a_lo = max(A / (oh * oh), ms2 / A)        # ch <= oh ; cw >= min_short
            a_hi = min((ow * ow) / A, A / ms2)        # cw <= ow ; ch >= min_short
            if a_lo > a_hi:
                # No free aspect fits this area -> use source aspect.
                a = ow / oh
            else:
                a = math.exp(rng.uniform(math.log(a_lo), math.log(a_hi)))
        else:
            a = ow / oh  # "source": preserve image aspect

        cw = int(round(math.sqrt(A * a)))
        ch = int(round(math.sqrt(A / a)))
        cw = max(self.min_short_side_px, min(ow, cw))
        ch = max(self.min_short_side_px, min(oh, ch))
        return cw, ch

    def _place(self, rng: random.Random, ow: int, oh: int, cw: int, ch: int, mode: str) -> Tuple[int, int]:
        """Top-left (cx, cy) of a (cw, ch) window in (ow, oh) per the position mode."""
        mx, my = ow - cw, oh - ch
        if mx <= 0 and my <= 0:
            return 0, 0
        if mode == "center":
            return mx // 2, my // 2
        if mode == "fixed_corner":
            return 0, 0
        if mode == "corner":
            corner = rng.randrange(4)
            return (mx if corner in (1, 3) else 0), (my if corner in (2, 3) else 0)
        # "random"
        cx = rng.randint(0, mx) if mx > 0 else 0
        cy = rng.randint(0, my) if my > 0 else 0
        return cx, cy

    # ----------------------------------------------------------------- specs
    def _spec_from(self, ow: int, oh: int, crop_box: Tuple[int, int, int, int],
                   bucket: BucketResolution, is_full: bool, fallback: bool = False) -> CropSpec:
        cx, cy, cw, ch = crop_box
        time_ids = (oh, ow, cy, cx, bucket.height, bucket.width)
        return CropSpec(
            is_full=is_full,
            crop_box=(cx, cy, cw, ch),
            bucket_w=bucket.width,
            bucket_h=bucket.height,
            time_ids=time_ids,
            fallback=fallback,
        )

    def _full_spec(self, ow: int, oh: int, use_max: bool, rng: random.Random,
                   fallback: bool = False) -> CropSpec:
        """Full image: pick a bucket (max-fit or smaller) by the image aspect, then take
        the aspect-fitting cover window (minimal crop) positioned per full_position_mode."""
        bucket = self._select_bucket(ow, oh, use_max, rng)
        cw, ch = _max_window_for_aspect(ow, oh, bucket.width, bucket.height)
        cx, cy = self._place(rng, ow, oh, cw, ch, self.full_position_mode)
        return self._spec_from(ow, oh, (cx, cy, cw, ch), bucket, is_full=True, fallback=fallback)

    def spec_for(self, epoch: int, image_path: str, ow: int, oh: int) -> CropSpec:
        """Return the CropSpec for (item, epoch). Pure function of (seed, epoch, path)."""
        if ow <= 0 or oh <= 0:
            ow, oh = max(1, ow), max(1, oh)

        # Disabled -> standard full image at the max-fitting bucket, centered minimal crop.
        if not self.enable:
            bucket = self._select_bucket(ow, oh, True, random.Random(0))
            cw, ch = _max_window_for_aspect(ow, oh, bucket.width, bucket.height)
            cx, cy = (ow - cw) // 2, (oh - ch) // 2
            return self._spec_from(ow, oh, (cx, cy, cw, ch), bucket, is_full=True)

        rng = self._item_rng(epoch, image_path)

        # Two independent axis decisions (fixed draw order for determinism).
        full = rng.random() < self.full_image_prob
        use_max = rng.random() < self.max_bucket_prob

        if full:
            return self._full_spec(ow, oh, use_max, rng)

        # Random crop.
        win = self._sample_crop_window(rng, ow, oh)
        if win is None:
            # Image too small to satisfy constraints -> full-image fallback.
            return self._full_spec(ow, oh, use_max, rng, fallback=True)
        cw, ch = win
        cx, cy = self._place(rng, ow, oh, cw, ch, self.position_mode)
        bucket = self._select_bucket(cw, ch, use_max, rng)
        return self._spec_from(ow, oh, (cx, cy, cw, ch), bucket, is_full=False)

    # --------------------------------------------------------- step accounting
    def precompute(self, items: List[Tuple[str, int, int]], num_epochs: int, batch_size: int) -> None:
        """Precompute per-epoch batch counts for exact step accounting.

        items: list of (image_path, ow, oh). Counts are the standard bucketed batch
        count (sum over buckets of ceil(count / batch_size)); priority/VE adjustments
        are layered on by the trainer.
        """
        self._batches_per_epoch = []
        for epoch in range(num_epochs):
            hist: Dict[Tuple[int, int], int] = {}
            for image_path, ow, oh in items:
                spec = self.spec_for(epoch, image_path, ow, oh)
                key = (spec.bucket_w, spec.bucket_h)
                hist[key] = hist.get(key, 0) + 1
            batches = sum((c + batch_size - 1) // batch_size for c in hist.values())
            self._batches_per_epoch.append(batches)

    def batches_per_epoch(self, epoch: int) -> int:
        return self._batches_per_epoch[epoch]

    def steps_per_epoch(self, epoch: int, multi_noise_timesteps: int = 1) -> int:
        return self._batches_per_epoch[epoch] * max(1, multi_noise_timesteps)

    def step_offsets(self, multi_noise_timesteps: int = 1) -> List[int]:
        """Cumulative step offsets, length num_epochs + 1; offsets[-1] = total_steps."""
        offsets = [0]
        for b in self._batches_per_epoch:
            offsets.append(offsets[-1] + b * max(1, multi_noise_timesteps))
        return offsets

    def epoch_for_step(self, global_step: int, multi_noise_timesteps: int = 1) -> int:
        """Epoch index containing global_step (for resume), via the offsets table."""
        import bisect
        offsets = self.step_offsets(multi_noise_timesteps)
        e = bisect.bisect_right(offsets, global_step) - 1
        return max(0, min(e, len(self._batches_per_epoch) - 1))

    # ------------------------------------------------------------- fingerprint
    def fingerprint(self, dataset_fingerprint: Optional[str] = None, num_epochs: int = 0) -> str:
        """Hash of the crop plan parameters; a change invalidates a saved resume plan."""
        parts = [
            f"seed={self.seed}",
            f"enable={self.enable}",
            f"full_prob={self.full_image_prob}",
            f"max_bucket_prob={self.max_bucket_prob}",
            f"min_area={self.min_area_ratio}",
            f"min_short={self.min_short_side_px}",
            f"aspect={self.aspect_mode}",
            f"pos={self.position_mode}",
            f"smaller_mode={self.smaller_bucket_mode}",
            f"smaller_scale={self.smaller_scale_lo},{self.smaller_scale_hi}",
            f"full_pos={self.full_position_mode}",
            f"micro={self.microcond_mode}",
            f"res={self.base_resolutions}",
            f"mrm={self.multi_resolution_mode}",
            f"epochs={num_epochs}",
            f"ds={dataset_fingerprint}",
        ]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
