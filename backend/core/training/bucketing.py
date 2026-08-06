"""
Aspect Ratio Bucketing for Training

Based on ai-toolkit implementation with enhancements:
- Multiple resolution support
- Resize vs crop strategies
- Random bucket assignment for multi-resolution
"""

from typing import List, Dict, Tuple, Optional, Literal, Union
from dataclasses import dataclass
import math


# Type alias for bucket keys: either just resolution, or (resolution, has_reference)
BucketKey = Union["BucketResolution", Tuple["BucketResolution", bool]]


@dataclass
class BucketResolution:
    """Bucket resolution definition"""
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def __hash__(self):
        return hash((self.width, self.height))

    def __eq__(self, other):
        if isinstance(other, BucketResolution):
            return self.width == other.width and self.height == other.height
        return False


# SDXL base resolutions (1024x1024 base)
RESOLUTIONS_1024: List[BucketResolution] = [
    # Base resolution
    BucketResolution(1024, 1024),
    # Widescreen
    BucketResolution(2048, 512),
    BucketResolution(1984, 512),
    BucketResolution(1920, 512),
    BucketResolution(1856, 512),
    BucketResolution(1792, 576),
    BucketResolution(1728, 576),
    BucketResolution(1664, 576),
    BucketResolution(1600, 640),
    BucketResolution(1536, 640),
    BucketResolution(1472, 704),
    BucketResolution(1408, 704),
    BucketResolution(1344, 704),
    BucketResolution(1344, 768),
    BucketResolution(1280, 768),
    BucketResolution(1216, 832),
    BucketResolution(1152, 832),
    BucketResolution(1152, 896),
    BucketResolution(1088, 896),
    BucketResolution(1088, 960),
    BucketResolution(1024, 960),
    # Portrait
    BucketResolution(960, 1024),
    BucketResolution(960, 1088),
    BucketResolution(896, 1088),
    BucketResolution(896, 1152),
    BucketResolution(832, 1152),
    BucketResolution(832, 1216),
    BucketResolution(768, 1280),
    BucketResolution(768, 1344),
    BucketResolution(704, 1408),
    BucketResolution(704, 1472),
    BucketResolution(640, 1536),
    BucketResolution(640, 1600),
    BucketResolution(576, 1664),
    BucketResolution(576, 1728),
    BucketResolution(576, 1792),
    BucketResolution(512, 1856),
    BucketResolution(512, 1920),
    BucketResolution(512, 1984),
    BucketResolution(512, 2048),
    # Extra wides
    BucketResolution(8192, 128),
    BucketResolution(128, 8192),
]


def get_bucket_sizes(resolution: int = 512, divisibility: int = 8) -> List[BucketResolution]:
    """
    Generate bucket sizes for a given base resolution.

    Args:
        resolution: Base resolution (e.g., 512, 768, 1024)
        divisibility: All dimensions must be divisible by this (default: 8 for VAE)

    Returns:
        List of bucket resolutions scaled from SDXL base
    """
    scaler = resolution / 1024

    bucket_list = []
    for bucket in RESOLUTIONS_1024:
        width = int(bucket.width * scaler)
        height = int(bucket.height * scaler)

        # Ensure divisibility
        if width % divisibility != 0:
            width = width - (width % divisibility)
        if height % divisibility != 0:
            height = height - (height % divisibility)

        bucket_list.append(BucketResolution(width, height))

    return bucket_list


def get_resolution_from_area(width: int, height: int) -> int:
    """
    Calculate square resolution from image area.

    Args:
        width: Image width
        height: Image height

    Returns:
        Square resolution with same pixel count
    """
    num_pixels = width * height
    square_resolution = int(num_pixels ** 0.5)
    return square_resolution


def get_bucket_for_image_size(
    width: int,
    height: int,
    bucket_list: Optional[List[BucketResolution]] = None,
    resolution: Optional[int] = None,
    divisibility: int = 8
) -> BucketResolution:
    """
    Find the best bucket for an image size.

    Args:
        width: Image width
        height: Image height
        bucket_list: Pre-generated bucket list (optional)
        resolution: Base resolution if bucket_list not provided
        divisibility: Dimension divisibility requirement

    Returns:
        Best matching bucket resolution
    """
    if bucket_list is None and resolution is None:
        # Auto-detect resolution from image area
        resolution = get_resolution_from_area(width, height)

    if bucket_list is None:
        # Use smaller of requested resolution and image resolution
        real_resolution = get_resolution_from_area(width, height)
        resolution = min(resolution, real_resolution)
        bucket_list = get_bucket_sizes(resolution=resolution, divisibility=divisibility)

    # Check for exact match first
    for bucket in bucket_list:
        if bucket.width == width and bucket.height == height:
            return bucket

    # Find closest bucket (minimize cropped pixels)
    closest_bucket = None
    min_removed_pixels = float("inf")

    for bucket in bucket_list:
        scale_w = bucket.width / width
        scale_h = bucket.height / height

        # Use larger scale to minimize crop amount
        scale = max(scale_w, scale_h)

        new_width = int(width * scale)
        new_height = int(height * scale)

        # Calculate pixels that would be cropped
        removed_pixels = (new_width - bucket.width) * new_height + (new_height - bucket.height) * new_width

        if removed_pixels < min_removed_pixels:
            min_removed_pixels = removed_pixels
            closest_bucket = bucket

    if closest_bucket is None:
        raise ValueError(f"No suitable bucket found for image size {width}x{height}")

    return closest_bucket


class BucketManager:
    """
    Manages aspect ratio bucketing for training datasets.

    Supports multiple resolutions with configurable assignment strategies.
    Optionally separates items by reference image availability.
    """

    def __init__(
        self,
        base_resolutions: List[int],
        divisibility: int = 8,
        strategy: Literal["resize", "crop", "random_crop"] = "resize",
        multi_resolution_mode: Literal["max", "random"] = "max",
        separate_by_reference: bool = False
    ):
        """
        Initialize bucket manager.

        Args:
            base_resolutions: List of base resolutions (e.g., [512, 768, 1024])
            divisibility: All dimensions must be divisible by this
            strategy: How to handle oversized images ("resize", "crop", "random_crop")
            multi_resolution_mode: How to assign images to resolutions when multiple specified
                - "max": Use largest resolution that fits the image (default)
                - "random": Randomly select from available resolutions
            separate_by_reference: If True, separate items with/without reference images
                into different buckets. This ensures batches contain either all items
                with reference images or all items without, enabling proper reference
                image conditioning during training.
        """
        self.base_resolutions = sorted(base_resolutions)
        self.divisibility = divisibility
        self.strategy = strategy
        self.multi_resolution_mode = multi_resolution_mode
        self.separate_by_reference = separate_by_reference

        # Generate bucket lists for each resolution
        self.bucket_lists: Dict[int, List[BucketResolution]] = {}
        for res in base_resolutions:
            self.bucket_lists[res] = get_bucket_sizes(res, divisibility)

        # Track which images go to which buckets
        # Key is BucketResolution when separate_by_reference=False
        # Key is (BucketResolution, has_reference) when separate_by_reference=True
        self.buckets: Dict[BucketKey, List[Dict]] = {}

    def select_bucket(
        self,
        width: int,
        height: int,
        target_resolution: Optional[int] = None,
        rng: Optional["random.Random"] = None,
    ) -> BucketResolution:
        """Select the best bucket for a (width, height) WITHOUT mutating state.

        Pure function form of the resolution/bucket selection used by
        assign_image_to_bucket. Used by CropPlanner so per-epoch crop sizes can be
        re-bucketed deterministically.

        Args:
            width, height: Image (or crop) pixel size.
            target_resolution: Force a specific base resolution (or None for auto).
            rng: RNG for multi_resolution_mode == "random" (pass a deterministic
                 random.Random for reproducibility; None uses the global random module).

        Returns:
            BucketResolution
        """
        # Determine which resolution to use
        if target_resolution is not None:
            bucket_list = self.bucket_lists.get(target_resolution)
            if bucket_list is None:
                raise ValueError(f"Resolution {target_resolution} not in base_resolutions")
        else:
            # Multi-resolution mode
            if self.multi_resolution_mode == "random":
                # Randomly select from all available resolutions
                import random as _random
                _rng = rng if rng is not None else _random
                target_resolution = _rng.choice(self.base_resolutions)
                bucket_list = self.bucket_lists[target_resolution]
            else:
                # "max" mode: assign the image to the LARGEST base resolution it can
                # fill WITHOUT upscaling, so high-resolution images train at high
                # resolution (the previous implementation compared crop ratios, which
                # are ~equal across resolutions for the same aspect, so the ascending
                # loop's strict `<` always kept the smallest resolution — every image
                # collapsed to the 512 bucket regardless of its real size).
                #
                # For each base resolution take its best (min-crop) aspect bucket, then
                # pick the largest resolution whose bucket the image downscales into
                # (scale <= 1 = no upscaling). If the image is smaller than every bucket
                # (all would upscale), fall back to the least-upscaling resolution.
                candidates = []  # (resolution, bucket, scale)
                for res in self.base_resolutions:
                    bl = self.bucket_lists[res]
                    candidate_bucket = get_bucket_for_image_size(
                        width, height, bl, divisibility=self.divisibility
                    )
                    scale = max(candidate_bucket.width / width,
                                candidate_bucket.height / height)  # > 1 means upscaling
                    candidates.append((res, candidate_bucket, scale))

                fitting = [c for c in candidates if c[2] <= 1.0 + 1e-6]
                if fitting:
                    _, bucket, _ = max(fitting, key=lambda c: c[0])
                else:
                    _, bucket, _ = min(candidates, key=lambda c: c[2])

                return bucket

        # Resolution-specific bucket list path
        return get_bucket_for_image_size(width, height, bucket_list, divisibility=self.divisibility)

    def assign_image_to_bucket(
        self,
        image_path: str,
        width: int,
        height: int,
        caption: str = "",
        target_resolution: Optional[int] = None,
        dataset_unique_id: Optional[str] = None,
        has_reference: bool = False,
        reference_images: Optional[list] = None,
        forced_bucket: Optional[BucketResolution] = None,
    ) -> Tuple[BucketKey, Dict]:
        """
        Assign an image to the best bucket.

        Args:
            image_path: Path to image file
            width: Image width
            height: Image height
            caption: Image caption
            target_resolution: Specific resolution to use (or None for auto)
            dataset_unique_id: Unique ID for dataset (for cache management)
            has_reference: Whether this item has reference images (only used if separate_by_reference=True)
            forced_bucket: Place the item directly into this bucket (skip selection).
                Used by CropPlanner for per-epoch re-bucketing of a known crop size.

        Returns:
            Tuple of (bucket_key, image_info)
            bucket_key is BucketResolution or (BucketResolution, has_reference)
        """
        # Pure bucket selection (no state mutation) — shared with CropPlanner.
        if forced_bucket is not None:
            bucket = forced_bucket
        else:
            bucket = self.select_bucket(width, height, target_resolution=target_resolution)

        # Create image info
        image_info = {
            "image_path": image_path,
            "caption": caption,
            "original_width": width,
            "original_height": height,
            "bucket_width": bucket.width,
            "bucket_height": bucket.height,
            "target_resolution": target_resolution,
            "has_reference": has_reference,  # Track reference status
        }
        # Store actual reference image paths (for VE conditioning and ControlNet)
        if reference_images:
            image_info["reference_images"] = reference_images

        # Add dataset_unique_id if provided (for cache management)
        if dataset_unique_id is not None:
            image_info["dataset_unique_id"] = dataset_unique_id

        # Determine bucket key based on separate_by_reference setting
        if self.separate_by_reference:
            bucket_key: BucketKey = (bucket, has_reference)
        else:
            bucket_key = bucket

        # Add to bucket
        if bucket_key not in self.buckets:
            self.buckets[bucket_key] = []
        self.buckets[bucket_key].append(image_info)

        return bucket_key, image_info

    def get_bucket_counts(self) -> Dict[str, int]:
        """Get count of images in each bucket."""
        result = {}
        for bucket_key, images in self.buckets.items():
            if isinstance(bucket_key, tuple):
                # (BucketResolution, has_reference)
                bucket, has_ref = bucket_key
                ref_suffix = "+ref" if has_ref else ""
                key = f"{bucket.width}x{bucket.height}{ref_suffix}"
            else:
                # BucketResolution only
                key = f"{bucket_key.width}x{bucket_key.height}"
            result[key] = len(images)
        return result

    def get_all_items(self) -> List[Dict]:
        """Get all items across all buckets (shuffled)."""
        import random
        all_items = []
        for images in self.buckets.values():
            all_items.extend(images)
        random.shuffle(all_items)
        return all_items

    def get_items_by_bucket(self) -> Dict[BucketKey, List[Dict]]:
        """Get items grouped by bucket."""
        return self.buckets.copy()

    def shuffle_buckets(self):
        """Shuffle items within each bucket."""
        import random
        for bucket_items in self.buckets.values():
            random.shuffle(bucket_items)

    def build_batch_indices(self, batch_size: int) -> List[List[Dict]]:
        """
        Build batch indices for training.

        Groups items from the same bucket into batches of batch_size.
        This ensures all items in a batch have the same resolution.
        When separate_by_reference=True, batches also have uniform reference status.

        Args:
            batch_size: Number of items per batch

        Returns:
            List of batches, where each batch is a list of item dicts
        """
        batch_list = []

        # Process each bucket separately
        # Bucket key is either BucketResolution or (BucketResolution, has_reference)
        for bucket_key, items in self.buckets.items():
            # Split items in this bucket into batches
            for start_idx in range(0, len(items), batch_size):
                end_idx = min(start_idx + batch_size, len(items))
                batch = items[start_idx:end_idx]
                batch_list.append(batch)

        # Shuffle the batches (not the items within batches)
        import random
        random.shuffle(batch_list)

        return batch_list

    def get_reference_statistics(self) -> Dict[str, int]:
        """
        Get statistics about reference image distribution.

        Returns:
            Dict with keys: 'with_reference', 'without_reference', 'total'
        """
        with_ref = 0
        without_ref = 0

        for bucket_key, items in self.buckets.items():
            if isinstance(bucket_key, tuple):
                # (BucketResolution, has_reference) - can directly check key
                _, has_ref = bucket_key
                if has_ref:
                    with_ref += len(items)
                else:
                    without_ref += len(items)
            else:
                # BucketResolution only - check individual items
                for item in items:
                    if item.get("has_reference", False):
                        with_ref += 1
                    else:
                        without_ref += 1

        return {
            "with_reference": with_ref,
            "without_reference": without_ref,
            "total": with_ref + without_ref
        }


# ============================================================================
# Temporal (video) bucketing — P4c, ADDITIVE.
#
# Everything below is used ONLY by the video-clip training path (LTX-2). It does
# not touch the image `BucketManager` above, so image bucketing / batching stays
# byte-for-byte unchanged (running image trainers are unaffected).
#
# A video item's bucket is the PAIR:
#     (spatial bucket ÷pixel_align-aligned, clip-length in frames)
# Batches built by `VideoBucketManager.build_batch_indices` are uniform in BOTH
# the spatial bucket AND the frame count, so the 5D latent tensors
# [1, C, T, H', W'] can stack.
#
# TEMPORAL SPEC (Phase 6a). Which clip lengths are valid, what a clip length
# means in latent frames, and whether the arch has a FIXED frame rate are
# per-architecture facts, declared once in
# `core.models.components.wiring.TemporalSpec` and passed in explicitly here.
# `spec=None` keeps every function on the LTX-2.3 rule it has always used
# (`8*k + 1`, source fps preserved, ÷32), so nothing about LTX-2.3 changes by
# the parameter's existence.
# ============================================================================

from core.models.components.wiring import (  # noqa: E402
    LTX2_TEMPORAL,
    TemporalSpec,
)

# LTX temporal compression: a clip of L pixel frames -> (L-1)//8 + 1 latent
# frames. Valid pixel clip lengths are 8*k + 1. Kept as the module fallback for
# `spec=None` callers.
_LTX_TEMPORAL_COMPRESSION = 8

# Default allowed clip lengths (all 8*k + 1). Configurable per call.
DEFAULT_CLIP_LENGTHS: List[int] = [9, 17, 25, 33, 49]

# LTX spatial divisibility (transformer patch / VAE spatial compression is 32).
LTX_SPATIAL_DIVISIBILITY = 32


def _spec_or_ltx(spec: Optional[TemporalSpec]) -> TemporalSpec:
    """The spec to apply — the caller's, or LTX-2.3's (the historical default)."""
    return spec if spec is not None else LTX2_TEMPORAL


def is_valid_clip_length(clip_length: int, spec: Optional[TemporalSpec] = None) -> bool:
    """True if ``clip_length`` is a valid pixel clip length for ``spec``.

    ``spec=None`` is the LTX-2.3 rule ``8*k + 1``. MiniMax-H3's rule is
    ``17*n + 5`` with a HARD decodable floor of 22 frames.

    The ``int()`` coercion (and its except branch) is pre-existing behaviour and
    is preserved verbatim: ``"9"`` and ``9.5`` are accepted, ``"x"``/``None``
    are not.
    """
    try:
        cl = int(clip_length)
    except (TypeError, ValueError):
        return False
    return _spec_or_ltx(spec).is_valid_length(cl)


def clip_span(
    clip_length: int,
    stride: int,
    spec: Optional[TemporalSpec] = None,
    source_fps: Optional[float] = None,
) -> int:
    """Number of SOURCE frames a ``clip_length``-frame clip (with ``stride``)
    spans.

    Index-sampled archs (LTX-2.3): ``(clip_length - 1) * stride + 1`` — the
    sampled indices ARE source indices.

    Fixed-fps archs (MiniMax-H3, 24 fps): the clip occupies
    ``clip_length*stride / fps_fixed`` seconds of the SOURCE timeline, which is
    a different number of source frames whenever the source is not already at
    the target rate — a 22-frame 24 fps clip spans 28 frames of a 30 fps
    source. Without ``source_fps`` there is nothing to convert with, so the
    index form is used (the pessimistic direction only for sources slower than
    the target).
    """
    clip_length = max(1, int(clip_length))
    stride = max(1, int(stride))
    sp = _spec_or_ltx(spec)
    if sp.fps_fixed is not None and source_fps:
        # (clip_length - 1) target-frame gaps of 1/fps_fixed seconds each.
        seconds = ((clip_length - 1) * stride) / float(sp.fps_fixed)
        return int(round(seconds * float(source_fps))) + 1
    return (clip_length - 1) * stride + 1


def pick_clip_length(
    num_frames: int,
    stride: int = 1,
    allowed_clip_lengths: Optional[List[int]] = None,
    spec: Optional[TemporalSpec] = None,
    source_fps: Optional[float] = None,
) -> int:
    """Pick the clip length for a video of ``num_frames`` frames at ``stride``.

    Chooses the LARGEST allowed length whose source span fits inside the video
    (``span <= num_frames``), so short videos get short clips and long videos get
    the longest configured clip. If no allowed length fits (video shorter than
    even the smallest span), returns the SMALLEST allowed length — `load_clip`
    then loop-pads the tail so the clip still has exactly that many frames.

    Args:
        num_frames: Total frames in the source video.
        stride: Gap between sampled frames (>= 1).
        allowed_clip_lengths: Candidate lengths. Defaults to the spec's
            ``default_clip_lengths`` (LTX-2.3: ``DEFAULT_CLIP_LENGTHS``).
        spec: Per-arch temporal spec; None = LTX-2.3.
        source_fps: Source frame rate, used only by fixed-fps archs to convert
            the clip's duration into source frames.

    Returns:
        A clip length valid for ``spec``.
    """
    stride = max(1, int(stride))
    num_frames = max(0, int(num_frames))
    sp = _spec_or_ltx(spec)
    default_lengths = list(sp.default_clip_lengths) if spec is not None else DEFAULT_CLIP_LENGTHS
    allowed = allowed_clip_lengths or default_lengths
    # Keep only lengths valid for this arch, ascending, deduped.
    valid = sorted({int(c) for c in allowed if is_valid_clip_length(c, spec)})
    if not valid:
        # Nothing usable was configured. For LTX-2.3 the historical fallback is
        # [1] (a still); for an arch with a hard decodable floor a 1-frame clip
        # is not loadable at all, so fall back to its own shortest valid length.
        valid = [1] if spec is None else [sp.snap_length(sp.min_decodable_frames, smoke=True)]

    fitting = [c for c in valid if clip_span(c, stride, spec, source_fps) <= num_frames]
    if fitting:
        return max(fitting)
    return valid[0]


def get_video_spatial_bucket(
    width: int,
    height: int,
    resolution: Optional[int] = None,
    divisibility: Optional[int] = None,
    spec: Optional[TemporalSpec] = None,
) -> BucketResolution:
    """Best ÷``divisibility``-aligned spatial bucket for a (width, height) clip.

    Reuses the standard aspect-ratio bucket set but forces the VIDEO arch's
    alignment (LTX-2.3 and MiniMax-H3 both require %32), never the SD ÷8/÷64
    sets. ``divisibility`` defaults to ``spec.pixel_align`` and then to 32.
    Pure function; no state mutation.
    """
    if divisibility is None:
        divisibility = _spec_or_ltx(spec).pixel_align or LTX_SPATIAL_DIVISIBILITY
    return get_bucket_for_image_size(
        width, height, resolution=resolution, divisibility=int(divisibility)
    )


class VideoBucketManager:
    """Temporal bucketing for LTX video clips (P4c).

    Buckets are keyed by the PAIR ``(spatial_bucket, clip_length)``. A batch drawn
    from a single bucket is therefore uniform in BOTH the ÷32 spatial size AND the
    frame count, which is required for the 5D latents to stack.

    This is a standalone sibling of ``BucketManager`` (it reuses its per-resolution
    bucket lists for spatial selection) and never mutates the image path.
    """

    def __init__(
        self,
        base_resolutions: List[int],
        divisibility: Optional[int] = None,
        allowed_clip_lengths: Optional[List[int]] = None,
        stride: int = 1,
        multi_resolution_mode: Literal["max", "random"] = "max",
        temporal_spec: Optional[TemporalSpec] = None,
    ):
        # `temporal_spec=None` is the LTX-2.3 rule this class shipped with.
        self.temporal_spec = temporal_spec
        sp = _spec_or_ltx(temporal_spec)
        if divisibility is None:
            divisibility = sp.pixel_align or LTX_SPATIAL_DIVISIBILITY
        if divisibility % _LTX_TEMPORAL_COMPRESSION and divisibility not in (8, 16, 32, 64):
            pass  # allow any divisibility, but video callers pass 32
        self.divisibility = int(divisibility)
        self.stride = max(1, int(stride))
        default_lengths = (list(sp.default_clip_lengths) if temporal_spec is not None
                           else DEFAULT_CLIP_LENGTHS)
        allowed = allowed_clip_lengths or default_lengths
        self.allowed_clip_lengths = sorted(
            {int(c) for c in allowed if is_valid_clip_length(c, temporal_spec)}
        ) or ([1] if temporal_spec is None
              else [sp.snap_length(sp.min_decodable_frames, smoke=True)])

        # Reuse BucketManager only for its precomputed ÷div spatial bucket lists.
        self._bm = BucketManager(
            base_resolutions=base_resolutions,
            divisibility=self.divisibility,
            strategy="crop",
            multi_resolution_mode=multi_resolution_mode,
        )

        # Key: (BucketResolution, clip_length) -> list of item dicts.
        self.buckets: Dict[Tuple[BucketResolution, int], List[Dict]] = {}

    def select_spatial_bucket(
        self, width: int, height: int, target_resolution: Optional[int] = None,
    ) -> BucketResolution:
        """÷div spatial bucket for a clip (no state mutation)."""
        return self._bm.select_bucket(width, height, target_resolution=target_resolution)

    def pick_clip_length(
        self,
        num_frames: int,
        stride: Optional[int] = None,
        source_fps: Optional[float] = None,
    ) -> int:
        """Clip length for a video of ``num_frames`` frames (uses this manager's
        allowed set + stride + temporal spec)."""
        return pick_clip_length(
            num_frames,
            self.stride if stride is None else stride,
            self.allowed_clip_lengths,
            spec=self.temporal_spec,
            source_fps=source_fps,
        )

    def assign_video_to_bucket(
        self,
        video_path: str,
        width: int,
        height: int,
        num_frames: int,
        caption: str = "",
        stride: Optional[int] = None,
        fps: Optional[float] = None,
        target_resolution: Optional[int] = None,
        dataset_unique_id: Optional[str] = None,
    ) -> Tuple[Tuple[BucketResolution, int], Dict]:
        """Assign a video item to a ``(spatial_bucket, clip_length)`` bucket.

        The chosen spatial bucket (÷div) and clip length flow into the returned
        info dict so the caller can build the P4b clip cache key (compute_clip_hash)
        from the ACTUAL window + bucket used.

        Returns ``((BucketResolution, clip_length), video_info)``.
        """
        eff_stride = self.stride if stride is None else max(1, int(stride))
        spatial = self.select_spatial_bucket(width, height, target_resolution=target_resolution)
        clip_length = self.pick_clip_length(num_frames, eff_stride, source_fps=fps)

        video_info = {
            "video_path": video_path,
            "item_type": "video",
            "caption": caption,
            "original_width": int(width),
            "original_height": int(height),
            "num_frames": int(num_frames),
            "bucket_width": spatial.width,
            "bucket_height": spatial.height,
            "clip_length": int(clip_length),
            "stride": int(eff_stride),
            "fps": (None if fps is None else float(fps)),
            "target_resolution": target_resolution,
        }
        # Fixed-fps archs (MiniMax-H3): the clip is RESAMPLED to the arch's rate,
        # so the item carries both numbers. ``fps`` keeps its meaning (the SOURCE
        # rate, which is what the resampler and the cache key need) and
        # ``target_fps`` is what the clip actually plays at. LTX-2.3 items never
        # gain this key, so everything reading ``fps`` is unaffected.
        sp = _spec_or_ltx(self.temporal_spec)
        if sp.fps_fixed is not None:
            video_info["target_fps"] = float(sp.fps_fixed)
        if dataset_unique_id is not None:
            video_info["dataset_unique_id"] = dataset_unique_id

        key = (spatial, int(clip_length))
        self.buckets.setdefault(key, []).append(video_info)
        return key, video_info

    def clip_cache_params(
        self,
        video_info: Dict,
        clip_start: int,
        start_time: Optional[float] = None,
        tiling_policy: Optional[str] = None,
        audio_prep_version: Optional[str] = None,
    ) -> Dict:
        """Build the argument dict for ``LatentCache.compute_clip_hash`` from a
        bucket assignment + the sampled window start, so the cache key reflects the
        ACTUAL window + bucket used this step.

        The returned keys match ``compute_clip_hash`` / ``save_clip_latent`` /
        ``load_clip_latent`` parameter names exactly.

        For an index-sampled arch (LTX-2.3, ``temporal_spec=None``) the returned
        dict is EXACTLY the seven historical keys — no extra field appears, so
        existing cache files stay addressable.

        For a fixed-fps arch the resampling and VAE-tiling policies join the key:
        the same window decoded at a different target rate, or encoded with
        tiling flipped, is a DIFFERENT latent (K0.5/Phase 0T measured rel-RMS
        0.355 / 0.0952 for tiling alone), and must not be served from one cache
        entry.
        """
        params = {
            "video_path": video_info["video_path"],
            "width": int(video_info["bucket_width"]),
            "height": int(video_info["bucket_height"]),
            "clip_start": int(clip_start),
            "clip_length": int(video_info["clip_length"]),
            "stride": int(video_info["stride"]),
            "fps": video_info.get("fps"),
        }
        sp = _spec_or_ltx(self.temporal_spec)
        if sp.fps_fixed is not None:
            params["source_fps"] = video_info.get("fps")
            params["target_fps"] = float(sp.fps_fixed)
            params["resample_policy"] = sp.resample_policy
            params["start_time"] = (None if start_time is None else float(start_time))
        if tiling_policy is not None:
            params["tiling_policy"] = tiling_policy
        if audio_prep_version is not None:
            params["audio_prep_version"] = audio_prep_version
        return params

    def get_bucket_counts(self) -> Dict[str, int]:
        """Count of items per ``WxHxLf`` bucket (for logging)."""
        result: Dict[str, int] = {}
        for (spatial, clip_length), items in self.buckets.items():
            key = f"{spatial.width}x{spatial.height}x{clip_length}f"
            result[key] = len(items)
        return result

    def get_items_by_bucket(self) -> Dict[Tuple[BucketResolution, int], List[Dict]]:
        return self.buckets.copy()

    def shuffle_buckets(self):
        import random
        for items in self.buckets.values():
            random.shuffle(items)

    def build_batch_indices(self, batch_size: int) -> List[List[Dict]]:
        """Build batches uniform in BOTH spatial bucket AND clip length.

        Each bucket key is ``(spatial_bucket, clip_length)``, so chunking within a
        bucket can never mix spatial sizes or frame counts — every batch stacks
        into a single 5D tensor.
        """
        batch_list: List[List[Dict]] = []
        for _key, items in self.buckets.items():
            for start_idx in range(0, len(items), batch_size):
                batch_list.append(items[start_idx:start_idx + batch_size])
        import random
        random.shuffle(batch_list)
        return batch_list
