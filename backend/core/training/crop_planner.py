"""
Epoch-dynamic crop & bucketing planner (SDXL only).

For each (item, epoch), deterministically decides whether to train on the full image
or a constrained random crop, and which bucket the result maps to. The decision is a
pure function of (seed, epoch, image_path), independent of the global RNG stream, so a
resumed run regenerates identical crops regardless of where it was interrupted.

SDXL micro-conditioning (time_ids) uses the kohya convention:
    time_ids = [original_h, original_w, crop_top, crop_left, target_h, target_w]
    - original_size = full original image size
    - crop_top_left = crop window top-left in original-image pixels
    - target_size   = output bucket size
For the full-image case this reduces to the existing micro-conditioning behavior
(crop=(0,0), original=full, target=bucket), so it is backward compatible.

See docs/EPOCH_DYNAMIC_CROP_BUCKETING_DESIGN.md.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.training.bucketing import BucketManager, BucketResolution


@dataclass(frozen=True)
class CropSpec:
    """Per-(item, epoch) crop decision.

    Attributes:
        is_full: True if the full image is used (no crop).
        crop_box: (cx, cy, cw, ch) crop window in original-image pixels.
        bucket_w, bucket_h: Output bucket size (the encoded latent's pixel target).
        time_ids: kohya-style SDXL time_ids (orig_h, orig_w, crop_top, crop_left,
                  target_h, target_w).
        fallback: True if a crop was requested but constraints could not be met and
                  the planner fell back to the full image (for logging).
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
        self.full_image_prob: float = float(_cfg("crop_full_image_prob"))
        self.min_area_ratio: float = float(_cfg("crop_min_area_ratio"))
        self.min_short_side_px: int = int(_cfg("crop_min_short_side_px"))
        _sr = _cfg("crop_scale_range")
        self.scale_min: float = float(_sr[0])
        self.scale_max: float = min(1.0, float(_sr[1]))  # never upscale beyond max window
        self.position_mode: str = str(_cfg("crop_position_mode"))
        self.microcond_mode: str = str(_cfg("crop_microcond_mode"))

        seed = int(_cfg("crop_plan_seed"))
        # 0 = derive from the global training seed (resolved by the caller and passed
        # via config["crop_plan_seed"] or config["seed"]). Fall back to 0.
        if seed == 0:
            seed = int(config.get("seed", 0) or 0)
        self.seed: int = seed

        self.base_resolutions = sorted(base_resolutions)
        self.multi_resolution_mode = multi_resolution_mode
        self.divisibility = divisibility

        # Internal BucketManager used purely for select_bucket() (no state mutation).
        self._bm = BucketManager(
            base_resolutions=self.base_resolutions,
            divisibility=divisibility,
            strategy="crop",
            multi_resolution_mode=multi_resolution_mode,
        )
        # Aspect candidate set: buckets of the largest base resolution (aspects are the
        # same across resolutions; select_bucket() re-picks the actual resolution).
        self._aspect_buckets: List[BucketResolution] = list(
            self._bm.bucket_lists[max(self.base_resolutions)]
        )

        # Cached per-epoch batch counts (filled by precompute()).
        self._batches_per_epoch: List[int] = []

    # ------------------------------------------------------------------ RNG
    def _item_rng(self, epoch: int, image_path: str) -> random.Random:
        """Independent RNG seeded by (seed, epoch, image_path). Pure function -> resume
        regenerates identical crops regardless of interruption point."""
        h = hashlib.sha256(f"{self.seed}|{epoch}|{image_path}".encode("utf-8")).digest()
        return random.Random(int.from_bytes(h[:8], "big"))

    # ----------------------------------------------------------------- specs
    def _full_spec(self, ow: int, oh: int, fallback: bool = False) -> CropSpec:
        b = self._bm.select_bucket(ow, oh)
        time_ids = (oh, ow, 0, 0, b.height, b.width)
        return CropSpec(
            is_full=True,
            crop_box=(0, 0, ow, oh),
            bucket_w=b.width,
            bucket_h=b.height,
            time_ids=time_ids,
            fallback=fallback,
        )

    def spec_for(self, epoch: int, image_path: str, ow: int, oh: int) -> CropSpec:
        """Return the CropSpec for (item, epoch). Pure function of (seed, epoch, path).

        Args:
            epoch: Epoch index (0-based).
            image_path: Item key (also the RNG seed component).
            ow, oh: Original image pixel size.
        """
        if ow <= 0 or oh <= 0:
            return self._full_spec(max(1, ow), max(1, oh))

        # When disabled, always full image (== current micro-conditioning behavior).
        if not self.enable:
            return self._full_spec(ow, oh)

        rng = self._item_rng(epoch, image_path)

        # 1) Full-image vs crop decision.
        if rng.random() < self.full_image_prob:
            return self._full_spec(ow, oh)

        # 2) Feasible aspect candidates: max-window must satisfy both constraints
        #    (shrinking only reduces short-side and area, so the max window is the most
        #    permissive). Dedup by (cw, ch) to avoid biasing toward duplicate aspects.
        min_area = self.min_area_ratio * ow * oh
        feasible: List[Tuple[int, int]] = []
        seen = set()
        for b in self._aspect_buckets:
            cw_max, ch_max = _max_window_for_aspect(ow, oh, b.width, b.height)
            if (cw_max, ch_max) in seen:
                continue
            seen.add((cw_max, ch_max))
            if min(cw_max, ch_max) < self.min_short_side_px:
                continue
            if cw_max * ch_max < min_area:
                continue
            feasible.append((cw_max, ch_max))

        if not feasible:
            # Image too small to satisfy constraints for any aspect -> full image.
            return self._full_spec(ow, oh, fallback=True)

        # 3) Pick an aspect (max window), then a scale that honors the constraints.
        cw_max, ch_max = rng.choice(feasible)
        short_max = min(cw_max, ch_max)
        area_max = cw_max * ch_max
        # Lower bound on scale so both constraints hold; never exceed 1.0 (max window).
        s_lo = max(
            self.scale_min,
            self.min_short_side_px / short_max,
            math.sqrt(min_area / area_max),
        )
        s_hi = self.scale_max
        if s_lo >= s_hi:
            # Constraints push the scale above the configured range -> honor constraints.
            s = min(1.0, s_lo)
        else:
            s = rng.uniform(s_lo, s_hi)

        cw = max(1, min(ow, int(round(cw_max * s))))
        ch = max(1, min(oh, int(round(ch_max * s))))

        # 4) Position within the original image.
        if self.position_mode == "center":
            cx = (ow - cw) // 2
            cy = (oh - ch) // 2
        else:
            cx = rng.randint(0, ow - cw) if ow - cw > 0 else 0
            cy = rng.randint(0, oh - ch) if oh - ch > 0 else 0

        # 5) Bucket the crop window (re-pick resolution via no-upscale logic).
        b = self._bm.select_bucket(cw, ch)
        # kohya time_ids: original = full image, crop_top_left in original pixels.
        time_ids = (oh, ow, cy, cx, b.height, b.width)
        return CropSpec(
            is_full=False,
            crop_box=(cx, cy, cw, ch),
            bucket_w=b.width,
            bucket_h=b.height,
            time_ids=time_ids,
        )

    # --------------------------------------------------------- step accounting
    def precompute(
        self,
        items: List[Tuple[str, int, int]],
        num_epochs: int,
        batch_size: int,
    ) -> None:
        """Precompute per-epoch batch counts for exact step accounting.

        Args:
            items: List of (image_path, ow, oh) for every training item.
            num_epochs: Total epochs.
            batch_size: Items per batch.

        Note: this is the *standard bucketed* batch count (sum over buckets of
        ceil(count / batch_size)). Priority-training / VE reference-separation
        adjustments are layered on by the trainer; this is the canonical baseline.
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
        """Cumulative step offsets, length num_epochs + 1. offsets[e] = steps before
        epoch e; offsets[-1] = total_steps."""
        offsets = [0]
        for b in self._batches_per_epoch:
            offsets.append(offsets[-1] + b * max(1, multi_noise_timesteps))
        return offsets

    def epoch_for_step(self, global_step: int, multi_noise_timesteps: int = 1) -> int:
        """Epoch index containing global_step (for resume), via the offsets table."""
        import bisect
        offsets = self.step_offsets(multi_noise_timesteps)
        # offsets is sorted ascending; find the last offset <= global_step.
        e = bisect.bisect_right(offsets, global_step) - 1
        return max(0, min(e, len(self._batches_per_epoch) - 1))

    # ------------------------------------------------------------- fingerprint
    def fingerprint(self, dataset_fingerprint: Optional[str] = None, num_epochs: int = 0) -> str:
        """Hash of the crop plan parameters. A change here invalidates a resume's saved
        crop plan (forces a fresh fallback)."""
        parts = [
            f"seed={self.seed}",
            f"enable={self.enable}",
            f"full_prob={self.full_image_prob}",
            f"min_area={self.min_area_ratio}",
            f"min_short={self.min_short_side_px}",
            f"scale={self.scale_min},{self.scale_max}",
            f"pos={self.position_mode}",
            f"micro={self.microcond_mode}",
            f"res={self.base_resolutions}",
            f"mrm={self.multi_resolution_mode}",
            f"epochs={num_epochs}",
            f"ds={dataset_fingerprint}",
        ]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
