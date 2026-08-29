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

See docs/guides/DYNAMIC_CROP_BUCKETING.md.
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
        _ssr = _cfg("crop_smaller_scale_range") or TRAINING_DEFAULTS["crop_smaller_scale_range"]
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
    def precompute(self, items: List[Tuple[str, int, int]], num_epochs: int, batch_size: int,
                   sample_epochs: int = 4, max_sample_items: int = 50_000) -> None:
        """Precompute per-epoch batch counts for step accounting (for the progress total).

        items: list of (image_path, ow, oh). The per-epoch batch count is the standard
        bucketed count (sum over buckets of ceil(count / batch_size)).

        A full O(num_epochs * num_items) pass is prohibitive at scale (e.g. 500 epochs x
        3M items = 1.5B planner evals). Two reductions, both valid because this only feeds
        the progress-bar total (the real per-epoch step count is whatever the epoch loop
        builds at runtime):
          - epoch sampling: compute `sample_epochs` epochs, mean for the rest (the fixed
            mix probabilities make per-epoch counts statistically stable).
          - item sampling: estimate the bucket distribution from a strided subset of up to
            `max_sample_items` items and scale the per-bucket counts back to the full size.
        """
        n = len(items)
        if n > max_sample_items:
            stride = n / max_sample_items
            sample = [items[int(i * stride)] for i in range(max_sample_items)]
            scale = n / len(sample)
        else:
            sample = items
            scale = 1.0

        sample_n = max(1, min(num_epochs, sample_epochs))
        sampled: List[int] = []
        for epoch in range(sample_n):
            hist: Dict[Tuple[int, int], float] = {}
            for image_path, ow, oh in sample:
                spec = self.spec_for(epoch, image_path, ow, oh)
                key = (spec.bucket_w, spec.bucket_h)
                hist[key] = hist.get(key, 0) + 1
            # Scale per-bucket counts to the full dataset, then ceil into batches.
            batches = sum(int(math.ceil(c * scale / batch_size)) for c in hist.values())
            sampled.append(batches)
        mean_b = round(sum(sampled) / len(sampled))
        # Exact (sampled) for the sampled epochs, mean for the remainder.
        self._batches_per_epoch = [
            (sampled[e] if e < sample_n else mean_b) for e in range(num_epochs)
        ]

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


class OutpaintControlPlanner:
    """Deterministic per-(item, epoch) crop-rectangle sampler for the outpaint-native
    ControlNet's self-supervised crop->full conditioning (PART B).

    The sampled rect is the KNOWN region (the simulated "placed input"); the FULL
    image is the training target. The rect + the full image are then handed to
    ``core.utils.crop_mask_condition.build_crop_mask_condition`` to build the 4-ch
    conditioning the ControlNet sees. Sibling to :class:`CropPlanner`: it reuses the
    same independent SHA256(seed|epoch|image_path) RNG so a resumed run regenerates
    identical crops regardless of the interruption point.

    Anchor modes (which teach the real outpaint geometry -- extend AWAY from what is
    given):
      - border:   rect spans one FULL canvas axis (cw == W or ch == H) with a partial
                  band on the other axis, flush to one edge -- the exact inverse of
                  ``build_outpaint_canvas``'s single-direction extend (the dominant
                  inference case); a small sub-fraction places the band interior
                  (two-parallel-sides extend)
      - interior: rect free-floating inside the canvas (extend in all directions;
                  also covers the "frame"/all-around extend)
      - edge:     rect flush against one canvas edge (one-directional extend, but
                  NOT axis-spanning)
      - corner:   rect flush into one canvas corner (two-directional extend)

    This is a genuinely different sampling law from :class:`CropPlanner` (area-frac +
    anchor, not min-area-ratio + aspect bucketing), so it is a standalone class rather
    than a config mode of CropPlanner; it deliberately shares only the determinism
    contract.
    """

    # ---- Border/side mode probabilities (hardcoded; could be promoted to config
    # params later -- param plumbing is intentionally NOT added here because the
    # param surface is being edited in parallel elsewhere).
    #
    # BORDER_MODE_PROB: fraction of samples whose known rect spans one full canvas
    # axis. Single-direction extend (known image flush across the entire width or
    # height) is the canonical inference geometry, so it gets the single largest
    # share; the remaining probability mass keeps the pre-existing edge/corner/
    # interior partition intact (scaled by 1 - BORDER_MODE_PROB) so multi-direction
    # extend stays well covered.
    BORDER_MODE_PROB = 0.35
    # Within border mode: probability the band is flush to one edge (single-direction
    # extend). The remainder places the band interior along the partial axis, i.e.
    # the two-parallel-sides extend case.
    BORDER_FLUSH_PROB = 0.8

    def __init__(
        self,
        seed: int = 0,
        min_area: float = 0.15,
        max_area: float = 0.8,
        edge_anchor_prob: float = 0.34,
        corner_anchor_prob: float = 0.33,
        aspect_jitter: float = 0.25,
        snap: int = 8,
        edge_feather_min_px: float = 0.0,
        edge_feather_max_px: float = 0.0,
    ):
        self.seed = int(seed)
        # Clamp to a sane sub-full range so a generate region always exists.
        self.min_area = float(max(0.02, min(0.95, min_area)))
        self.max_area = float(max(self.min_area, min(0.95, max_area)))
        e = float(max(0.0, edge_anchor_prob))
        c = float(max(0.0, corner_anchor_prob))
        if e + c > 1.0:
            # Normalize into [0,1]; interior takes whatever is left (>=0).
            s = e + c
            e, c = e / s, c / s
        self.edge_prob = e
        self.corner_prob = c
        self.aspect_jitter = float(max(0.0, aspect_jitter))
        self.snap = max(1, int(snap))
        # R1 (scratchpad/outpaint_boundary_structure_fix.md D3-R1): per-sample
        # randomized crop_mask_condition edge_feather_px range. Both default to
        # 0.0 -> feather_for() always returns 0.0 with no RNG draw at all, i.e.
        # byte-identical to before this feature existed unless a caller opts in.
        self.edge_feather_min_px = float(max(0.0, edge_feather_min_px))
        self.edge_feather_max_px = float(max(self.edge_feather_min_px, edge_feather_max_px))

    def _item_rng(self, epoch: int, image_path: str) -> random.Random:
        """Independent RNG seeded by (seed, epoch, image_path) -- identical scheme to
        :meth:`CropPlanner._item_rng` so the two planners are jointly resume-safe."""
        h = hashlib.sha256(f"{self.seed}|{epoch}|{image_path}".encode("utf-8")).digest()
        return random.Random(int.from_bytes(h[:8], "big"))

    def _snap(self, v: int, lo: int, hi: int) -> int:
        v = int(round(v / self.snap)) * self.snap
        return max(lo, min(hi, v))

    def _border_rect(self, rng: random.Random, W: int, H: int) -> Tuple[int, int, int, int]:
        """Known rect spanning one FULL canvas axis (cw == W or ch == H) with a partial
        band on the other axis. Flush placement (BORDER_FLUSH_PROB) inverts
        ``build_outpaint_canvas``'s single-direction extend: the known region is the
        original image and the generate region is a border strip on the opposite side.
        Interior placement covers the two-parallel-sides extend. Half-open rect,
        strictly inside the canvas, always leaves a non-empty generate region."""
        horizontal = rng.random() < 0.5  # True: cw == W (extend top/bottom)
        # Band fraction on the partial axis. Since the other axis is full-span, this
        # IS the area fraction, so area coverage stays within [min_area, max_area].
        frac = rng.uniform(self.min_area, self.max_area)
        if horizontal:
            cw = W
            ch = self._snap(int(round(frac * H)), self.snap, max(self.snap, H - self.snap))
            if ch >= H:  # tiny-canvas guard (H <= snap): keep the generate region non-empty
                ch = max(1, H - 1)
            m = H - ch
        else:
            ch = H
            cw = self._snap(int(round(frac * W)), self.snap, max(self.snap, W - self.snap))
            if cw >= W:
                cw = max(1, W - 1)
            m = W - cw

        if rng.random() < self.BORDER_FLUSH_PROB or m < 2 * self.snap:
            # Single-direction extend: band flush to one edge, generate strip opposite.
            off = 0 if rng.random() < 0.5 else m
        else:
            # Two-parallel-sides extend: band interior, generate strips on both sides
            # (offset snapped into [snap, m - snap] so neither strip collapses).
            off = self._snap(rng.randint(self.snap, m - self.snap), self.snap, m - self.snap)

        if horizontal:
            return (0, off, W, off + ch)
        return (off, 0, off + cw, H)

    def rect_for(self, epoch: int, image_path: str, canvas_w: int, canvas_h: int) -> Tuple[int, int, int, int]:
        """Return the known-region rect (x0, y0, x1, y1) in canvas pixels, half-open.
        Pure function of (seed, epoch, image_path, canvas_w, canvas_h). Guarantees a
        non-empty generate region (the rect never covers the whole canvas)."""
        W, H = max(1, int(canvas_w)), max(1, int(canvas_h))
        rng = self._item_rng(epoch, image_path)

        # Mode draw FIRST (fixed draw position for determinism): border/side mode
        # produces a full-axis-spanning known rect matching the single-direction
        # extend geometry used at inference (see class docstring).
        if rng.random() < self.BORDER_MODE_PROB:
            return self._border_rect(rng, W, H)

        area = rng.uniform(self.min_area, self.max_area) * (W * H)
        # Aspect around the canvas aspect with multiplicative jitter.
        a = (W / H) * math.exp(rng.uniform(-self.aspect_jitter, self.aspect_jitter))
        cw = int(round(math.sqrt(area * a)))
        ch = int(round(math.sqrt(area / a)))
        # Snap + clamp; keep at least one snap of generate room on the larger axis so a
        # generate region always exists even after aspect jitter hits a clamp.
        cw = self._snap(cw, self.snap, max(self.snap, W - self.snap))
        ch = self._snap(ch, self.snap, max(self.snap, H - self.snap))
        cw = min(cw, W); ch = min(ch, H)
        if cw >= W and ch >= H:
            # Degenerate (no generate region) -> shrink the larger axis by one snap.
            if W >= H:
                cw = max(self.snap, W - self.snap)
            else:
                ch = max(self.snap, H - self.snap)

        mx, my = W - cw, H - ch  # placement margins (>= 0)
        r = rng.random()
        if r < self.edge_prob and (mx > 0 or my > 0):
            # Flush against one canvas edge; free along the parallel axis.
            edge = rng.randrange(4)  # 0 top, 1 bottom, 2 left, 3 right
            if edge in (0, 1):
                x0 = rng.randint(0, mx) if mx > 0 else 0
                y0 = 0 if edge == 0 else my
            else:
                y0 = rng.randint(0, my) if my > 0 else 0
                x0 = 0 if edge == 2 else mx
        elif r < self.edge_prob + self.corner_prob:
            # Flush into one canvas corner.
            corner = rng.randrange(4)  # 0 TL, 1 TR, 2 BL, 3 BR
            x0 = 0 if corner in (0, 2) else mx
            y0 = 0 if corner in (0, 1) else my
        else:
            # Interior: random free placement.
            x0 = rng.randint(0, mx) if mx > 0 else 0
            y0 = rng.randint(0, my) if my > 0 else 0

        x0 = self._snap(x0, 0, max(0, W - cw))
        y0 = self._snap(y0, 0, max(0, H - ch))
        return (x0, y0, x0 + cw, y0 + ch)

    def feather_for(self, epoch: int, image_path: str) -> float:
        """Per-sample randomized ``crop_mask_condition.build_crop_mask_condition``
        ``edge_feather_px`` draw (R1, ``scratchpad/outpaint_boundary_structure_fix.md``
        D3-R1): the known/unknown boundary is the one thing held constant across
        every sample by :meth:`rect_for` (position/size/aspect/anchor mode are
        already randomized) -- a ControlNet trained on it learns to render the
        rect perimeter as scene structure. Drawing a different edge softness per
        sample removes that invariant.

        Uses an INDEPENDENT RNG stream from :meth:`rect_for` (its own SHA256 salt,
        ``"edge_feather"``) so this draw can never perturb the rect draw order or
        values -- adding/removing/reordering calls to this method relative to
        ``rect_for`` never desyncs either stream. Pure function of
        (seed, epoch, image_path, edge_feather_min_px, edge_feather_max_px) ->
        resume-deterministic (identical to :meth:`rect_for`'s determinism
        contract).

        Returns ``edge_feather_min_px`` (0.0 by default) with NO ``random.Random``
        construction at all when ``edge_feather_max_px <= edge_feather_min_px``
        (the default 0.0/0.0) -- i.e. calling this is a total no-op unless the
        caller opts into a real range.
        """
        lo, hi = self.edge_feather_min_px, self.edge_feather_max_px
        if hi <= lo:
            return lo
        h = hashlib.sha256(f"{self.seed}|{epoch}|{image_path}|edge_feather".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(h[:8], "big"))
        return float(rng.uniform(lo, hi))
