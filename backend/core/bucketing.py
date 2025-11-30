"""
Aspect Ratio Bucketing for Training

Based on ai-toolkit implementation with enhancements:
- Multiple resolution support
- Resize vs crop strategies
- Random bucket assignment for multi-resolution
"""

from typing import List, Dict, Tuple, Optional, Literal
from dataclasses import dataclass
import math


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
    """

    def __init__(
        self,
        base_resolutions: List[int],
        divisibility: int = 8,
        strategy: Literal["resize", "crop", "random_crop"] = "resize",
        multi_resolution_mode: Literal["max", "random"] = "max"
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
        """
        self.base_resolutions = sorted(base_resolutions)
        self.divisibility = divisibility
        self.strategy = strategy
        self.multi_resolution_mode = multi_resolution_mode

        # Generate bucket lists for each resolution
        self.bucket_lists: Dict[int, List[BucketResolution]] = {}
        for res in base_resolutions:
            self.bucket_lists[res] = get_bucket_sizes(res, divisibility)

        # Track which images go to which buckets
        self.buckets: Dict[BucketResolution, List[Dict]] = {}

    def assign_image_to_bucket(
        self,
        image_path: str,
        width: int,
        height: int,
        caption: str = "",
        target_resolution: Optional[int] = None
    ) -> Tuple[BucketResolution, Dict]:
        """
        Assign an image to the best bucket.

        Args:
            image_path: Path to image file
            width: Image width
            height: Image height
            caption: Image caption
            target_resolution: Specific resolution to use (or None for auto)

        Returns:
            Tuple of (bucket_resolution, image_info)
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
                import random
                target_resolution = random.choice(self.base_resolutions)
                bucket_list = self.bucket_lists[target_resolution]
            else:
                # "max" mode: Find best bucket across all resolutions
                # Try each resolution and pick the bucket with minimum cropping
                best_bucket = None
                best_resolution = None
                min_crop_ratio = float("inf")

                for res in self.base_resolutions:
                    bucket_list = self.bucket_lists[res]
                    candidate_bucket = get_bucket_for_image_size(width, height, bucket_list, divisibility=self.divisibility)

                    # Calculate crop ratio (how much we need to crop relative to original)
                    scale_w = candidate_bucket.width / width
                    scale_h = candidate_bucket.height / height
                    scale = max(scale_w, scale_h)  # Scale to fit

                    scaled_width = width * scale
                    scaled_height = height * scale

                    crop_width = scaled_width - candidate_bucket.width
                    crop_height = scaled_height - candidate_bucket.height
                    crop_ratio = (crop_width + crop_height) / (width + height)

                    if crop_ratio < min_crop_ratio:
                        min_crop_ratio = crop_ratio
                        best_bucket = candidate_bucket
                        best_resolution = res

                bucket = best_bucket
                target_resolution = best_resolution
                bucket_list = None  # Already have the bucket

        # Find best bucket if not already determined
        if bucket_list is not None:
            bucket = get_bucket_for_image_size(width, height, bucket_list, divisibility=self.divisibility)

        # Create image info
        image_info = {
            "image_path": image_path,
            "caption": caption,
            "original_width": width,
            "original_height": height,
            "bucket_width": bucket.width,
            "bucket_height": bucket.height,
            "target_resolution": target_resolution,
        }

        # Add to bucket
        if bucket not in self.buckets:
            self.buckets[bucket] = []
        self.buckets[bucket].append(image_info)

        return bucket, image_info

    def get_bucket_counts(self) -> Dict[str, int]:
        """Get count of images in each bucket."""
        return {
            f"{bucket.width}x{bucket.height}": len(images)
            for bucket, images in self.buckets.items()
        }

    def get_all_items(self) -> List[Dict]:
        """Get all items across all buckets (shuffled)."""
        import random
        all_items = []
        for images in self.buckets.values():
            all_items.extend(images)
        random.shuffle(all_items)
        return all_items

    def get_items_by_bucket(self) -> Dict[BucketResolution, List[Dict]]:
        """Get items grouped by bucket."""
        return self.buckets.copy()
