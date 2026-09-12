"""Behavior contracts for shared tagger response selection."""

import numpy as np

from core.tagger.tag_selection import select_tag_response


def test_global_selection_preserves_top_categories_order_and_calibration():
    raw = np.array([0.7, 0.7, 0.2, 0.9, 0.8], dtype=np.float32)
    calibrated = np.array([0.6, 0.5, 0.4, 0.3, 0.2], dtype=np.float32)
    idx_to_tag = {0: "first", 1: "second", 2: "quality_a", 3: "quality_b", 4: "safe"}
    categories = {
        "first": "General", "second": "Character",
        "quality_a": "Quality", "quality_b": "Quality", "safe": "Rating",
    }

    tags, quality, rating, used = select_tag_response(
        raw, idx_to_tag, categories, threshold=0.5, cal_probs=calibrated,
        ood_t=1.0,
    )

    assert [item["tag"] for item in tags] == ["first", "second"]
    assert tags[0]["cal_prob"] == float(calibrated[0])
    assert quality["tag"] == "quality_b"
    assert rating["tag"] == "safe"
    assert used is False


def test_per_tag_thresholds_keep_fallbacks_and_drop_unreliable_tags():
    raw = np.array([0.4, 0.6, 0.8, 0.84], dtype=np.float32)
    idx_to_tag = {0: "floor", 1: "weak", 2: "few", 3: "ood_character"}
    categories = {"ood_character": "Character"}
    metrics = {
        "floor": (0.1, 0.8, 20),
        "weak": (0.4, 0.01, 20),
        "few": (0.9, 0.9, 2),
        "ood_character": (0.5, 0.9, 20),
    }

    tags, _, _, used = select_tag_response(
        raw,
        idx_to_tag,
        categories,
        threshold=0.7,
        use_per_tag_threshold=True,
        get_metrics=metrics.get,
        min_best_thr=0.3,
        min_best_f1=0.05,
        min_samples_for_per_tag=5,
        ood_t=1.0,
    )

    assert [item["tag"] for item in tags] == ["few", "floor"]
    assert used is True


def test_missing_rows_follow_each_callers_existing_policy():
    raw = np.array([0.9], dtype=np.float32)

    loaded, _, _, _ = select_tag_response(
        raw, {}, {}, threshold=0.5,
        missing_tag_prefix="__unk_", default_category="Unknown",
    )
    training, _, _, _ = select_tag_response(raw, {}, {}, threshold=0.5)

    assert loaded == [{
        "tag": "__unk_0__", "prob": float(raw[0]),
        "raw_prob": float(raw[0]), "category": "Unknown",
    }]
    assert training == []
