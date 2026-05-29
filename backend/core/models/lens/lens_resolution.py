"""Resolution bucket helpers for Lens (microsoft/Lens)."""

from core.models.lens.vendor.resolution import RESOLUTION_BUCKETS


def find_nearest_bucket(width: int, height: int) -> tuple[int, int]:
    """Snap user-specified (width, height) to the nearest Lens resolution bucket.

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
