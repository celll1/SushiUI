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
    """Mahalanobis distance between *emb* and the in-distribution reference.

    Mirrors ``SigLIP2InferenceManager._compute_mahalanobis`` exactly so the
    training-model OOD distance is comparable to the inference-model one.
    """
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


def select_tags(
    all_items: List[Dict],
    *,
    threshold: float,
    use_per_tag_threshold: bool,
    get_metrics: Optional[MetricsResolver],
    min_best_thr: float,
    min_best_f1: float,
    min_samples_for_per_tag: int,
    ood_t: float = 0.0,
) -> Tuple[List[Dict], bool]:
    """Filter *all_items* into the final tag list.

    Each item is a dict with at least ``tag``, ``category``, ``raw_prob`` and
    ``prob`` keys. Quality / Rating categories are always excluded here (the
    caller selects their per-image top-1 separately).

    When *use_per_tag_threshold* is True and *get_metrics* is provided, each
    tag is filtered by its own ``best_thr`` (clamped to *min_best_thr*) and
    dropped entirely when ``best_f1 < min_best_f1``. Tags without metrics fall
    back to the global *threshold*. For OOD images (*ood_t* > 0) the
    Character/Copyright thresholds are raised toward 0.85.

    Returns ``(filtered_sorted_by_prob_desc, used_best_thr)``.
    """
    filtered: List[Dict] = []
    used_best_thr = False

    if use_per_tag_threshold and get_metrics is not None:
        used_best_thr = True
        for it in all_items:
            cat = it["category"]
            if cat in ("Quality", "Rating"):
                continue
            thr_t = threshold  # fallback for tags without reliable metrics
            m = get_metrics(it["tag"])
            if m is not None:
                best_thr, best_f1, n_pos = m
                if (
                    n_pos is not None
                    and int(n_pos) >= min_samples_for_per_tag
                    and best_thr is not None
                    and not math.isnan(float(best_thr))
                ):
                    # Skip unreliable detectors (best_f1 below floor)
                    if best_f1 is not None and not math.isnan(float(best_f1)) \
                            and float(best_f1) < min_best_f1:
                        continue
                    # Clamp best_thr to the minimum to suppress noise-level FPs
                    thr_t = max(float(best_thr), min_best_thr)
            # OOD dynamic threshold: raise Character/Copyright thresholds toward
            # 0.85 proportionally to how far the image is from the train dist.
            if ood_t > 0.0 and cat in ("Character", "Copyright"):
                thr_t = thr_t + ood_t * (0.85 - thr_t)
            if it["raw_prob"] >= thr_t:
                filtered.append(it)
    else:
        filtered = [
            it for it in all_items
            if it["raw_prob"] >= threshold
            and it["category"] not in ("Quality", "Rating")
        ]

    filtered.sort(key=lambda x: x["prob"], reverse=True)
    return filtered, used_best_thr
