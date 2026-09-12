"""Shared post-processing for SigLIP2 tagger inference.

Both the loaded inference model (``SigLIP2InferenceManager.predict``) and the
live training model (``TaggerTrainerHandle.predict``) run the same per-tag
best-threshold filtering and OOD (out-of-distribution) threshold adjustment.
Keeping that logic here means the "Use training model" path produces identical
results to the inference path instead of a simplified copy that silently ignores
per-tag thresholds and OOD.

All functions are pure (numpy only) so they can be called from either manager
without sharing any object state.

Tag metrics are looked up by **tag name**, not by index, so the two paths stay
correct even when their vocabularies differ (e.g. the training model expanded
its head via Danbooru augmentation since the inference checkpoint / metrics file
was produced — new tags simply have no metric entry and fall back to the global
threshold instead of raising an IndexError or mis-aligning).
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# Per-tag metric resolver: tag name -> (best_thr, best_f1, n_pos) or None when
# the tag has no reliable metrics. Any of the three values may be NaN/None.
MetricsResolver = Callable[[str], Optional[Tuple[Optional[float], Optional[float], Optional[float]]]]


def mahalanobis(emb: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray) -> float:
    """Use float64 so loaded and live-training OOD distances stay comparable."""
    diff = emb.astype(np.float64) - mu.astype(np.float64)
    return float(np.sqrt(max(0.0, diff @ cov_inv.astype(np.float64) @ diff)))


def ood_threshold_scale(distance: Optional[float], p50: float, p95: float) -> float:
    """Ramp factor in [0, 1] describing how out-of-distribution an image is.

    0 for distance <= p95 (in-dist tail, no penalty), rising linearly to 1 at
    ``p95 + 2*(p95 - p50)``. Used to raise Character/Copyright thresholds for
    OOD images without penalising borderline in-dist ones.
    """
    if distance is None:
        return 0.0
    tail = max(p95 - p50, 1e-6)
    return max(0.0, min(1.0, (distance - p95) / (2.0 * tail)))


def calibration_table_to_name_map(
    calibration_table: Optional[np.ndarray],
    idx_to_tag: Dict[int, str],
) -> Optional[Dict[str, np.ndarray]]:
    """Convert an index-aligned [V, n_bins] calibration table into a
    ``{tag_name: row}`` map so it can be applied to a model with a different
    vocabulary index order. Returns None when no table is available."""
    if calibration_table is None:
        return None
    out: Dict[str, np.ndarray] = {}
    for idx, tag in idx_to_tag.items():
        if 0 <= idx < calibration_table.shape[0]:
            out[tag] = calibration_table[idx]
    return out


def apply_calibration_by_name(
    raw_probs: np.ndarray,
    idx_to_tag: Dict[int, str],
    name_calibration: Optional[Dict[str, np.ndarray]],
    n_bins: int,
) -> Optional[np.ndarray]:
    """Map raw sigmoid probs through a per-tag calibration table keyed by name.

    Returns calibrated probs (same shape as *raw_probs*) or None when no
    calibration data is available. Tags absent from *name_calibration* (e.g.
    newly-added head rows) keep their raw probability.
    """
    if not name_calibration:
        return None
    cal = raw_probs.astype(np.float32).copy()
    bin_idx = np.clip((raw_probs * n_bins).astype(np.int32), 0, n_bins - 1)
    for i in range(len(raw_probs)):
        tag = idx_to_tag.get(i)
        row = name_calibration.get(tag) if tag is not None else None
        if row is None:
            continue
        v = float(row[bin_idx[i]])
        if not math.isnan(v):
            cal[i] = v
    return cal


def select_tag_response(
    raw_probs: np.ndarray,
    idx_to_tag: Dict[int, str],
    tag_to_category: Dict[str, str],
    *,
    threshold: float,
    cal_probs: Optional[np.ndarray] = None,
    use_per_tag_threshold: bool = False,
    get_metrics: Optional[MetricsResolver] = None,
    min_best_thr: float = 0.30,
    min_best_f1: float = 0.05,
    min_samples_for_per_tag: int = 5,
    ood_t: float = 0.0,
    missing_tag_prefix: Optional[str] = None,
    default_category: str = "General",
) -> Tuple[List[Dict], Optional[Dict], Optional[Dict], bool]:
    """Build the filtered response without materializing every vocabulary row."""
    selected: List[Tuple[int, str, str]] = []
    quality: Optional[Tuple[int, str, str]] = None
    rating: Optional[Tuple[int, str, str]] = None
    use_metrics = bool(use_per_tag_threshold and get_metrics is not None)

    for idx in range(len(raw_probs)):
        tag = idx_to_tag.get(idx)
        if tag is None:
            if missing_tag_prefix is None:
                continue
            tag = f"{missing_tag_prefix}{idx}__"
        category = tag_to_category.get(tag, default_category)

        if category == "Quality":
            if quality is None or raw_probs[idx] > raw_probs[quality[0]]:
                quality = (idx, tag, category)
            continue
        if category == "Rating":
            if rating is None or raw_probs[idx] > raw_probs[rating[0]]:
                rating = (idx, tag, category)
            continue

        tag_threshold = threshold
        if use_metrics:
            metrics = get_metrics(tag)
            if metrics is not None:
                best_thr, best_f1, n_pos = metrics
                if (
                    n_pos is not None
                    and int(n_pos) >= min_samples_for_per_tag
                    and best_thr is not None
                    and not math.isnan(float(best_thr))
                ):
                    if (
                        best_f1 is not None
                        and not math.isnan(float(best_f1))
                        and float(best_f1) < min_best_f1
                    ):
                        continue
                    tag_threshold = max(float(best_thr), min_best_thr)
        if use_metrics and ood_t > 0.0 and category in ("Character", "Copyright"):
            tag_threshold += ood_t * (0.85 - tag_threshold)
        if raw_probs[idx] >= tag_threshold:
            selected.append((idx, tag, category))

    def make_item(row: Tuple[int, str, str]) -> Dict:
        idx, tag, category = row
        probability = float(raw_probs[idx])
        item = {
            "tag": tag,
            "prob": probability,
            "raw_prob": probability,
            "category": category,
        }
        if cal_probs is not None:
            item["cal_prob"] = float(cal_probs[idx])
        return item

    selected.sort(key=lambda row: raw_probs[row[0]], reverse=True)
    return (
        [make_item(row) for row in selected],
        make_item(quality) if quality is not None else None,
        make_item(rating) if rating is not None else None,
        use_metrics,
    )
