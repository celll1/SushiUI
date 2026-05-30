"""Resolution helpers for Lens (microsoft/Lens).

Lens uses RoPE-based positional encoding (LensEmbedRope) with dynamically
computed frequencies.  There is no fixed learned embedding table, so the
model works at **any** resolution whose width and height are multiples of
the VAE spatial downscale factor (16).

``align_to_grid`` is the primary helper for inference and training.
``find_nearest_bucket`` is kept for callers that explicitly need one of
the 18 predefined training buckets (e.g. preview fallback).
"""

from core.models.lens.vendor.resolution import RESOLUTION_BUCKETS


def align_to_grid(width: int, height: int, multiple: int = 16) -> tuple[int, int]:
    """Align (width, height) to the nearest multiples of *multiple*.

    Lens's VAE downscales by 16× spatially, so any (W, H) that are
    multiples of 16 produce valid latent grids.  This is the preferred
    helper for arbitrary-resolution inference.

    Args:
        width:    Requested image width in pixels.
        height:   Requested image height in pixels.
        multiple: Grid size; 16 for Lens (VAE factor × patch size).

    Returns:
        (aligned_width, aligned_height) — each clamped to at least *multiple*.
    """
    aligned_w = max(multiple, round(width / multiple) * multiple)
    aligned_h = max(multiple, round(height / multiple) * multiple)
    return aligned_w, aligned_h


def find_nearest_bucket(width: int, height: int) -> tuple[int, int]:
    """Snap user-specified (width, height) to the nearest Lens training bucket.

    There are 18 predefined buckets (2 base resolutions × 9 aspect ratios).
    This function is retained for callers that need an exact bucket match
    (e.g. latent-preview fallback when the actual resolution is unknown).
    For normal inference, prefer ``align_to_grid``.

    Returns:
        (snapped_width, snapped_height)
    """
    target_area = width * height
    best_w, best_h = width, height
    best_dist = float("inf")

    for base_res, ratio_map in RESOLUTION_BUCKETS.items():
        for _ratio_str, (bucket_h, bucket_w) in ratio_map.items():
            # Distance metric: relative area difference + aspect ratio delta
            bucket_area = bucket_w * bucket_h
            area_dist = abs(target_area - bucket_area) / max(target_area, bucket_area)
            ar_target = width / max(height, 1)
            ar_bucket = bucket_w / max(bucket_h, 1)
            ar_dist = abs(ar_target - ar_bucket)
            dist = area_dist + 0.5 * ar_dist
            if dist < best_dist:
                best_dist = dist
                best_w, best_h = bucket_w, bucket_h

    return best_w, best_h
