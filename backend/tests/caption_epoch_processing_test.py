"""Regression coverage for per-item, per-epoch caption augmentation."""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training import caption_processor
from core.training.caption_processor import process_caption, process_caption_with_tag_data
from core.training.train_runner import _process_cached_items


TAG_DATA = [
    {"tag": "alpha", "category": "General"},
    {"tag": "beta", "category": "General"},
    {"tag": "gamma", "category": "General"},
    {"tag": "delta", "category": "General"},
    {"tag": "epsilon", "category": "General"},
]


def _natural_item(caption: str = "A natural language caption.", path: str = "natural.png") -> dict:
    return {
        "image_path": path,
        "raw_caption": caption,
        "tag_data": None,
        "is_tags_format": False,
        "width": 64,
        "height": 64,
    }


def test_natural_language_whole_caption_dropout_zero_and_one():
    kept = _process_cached_items(
        [_natural_item()],
        epoch_num=0,
        caption_config={
            "caption_dropout_rate": 0.0,
            "shuffle_tokens": True,
            "token_dropout_rate": 1.0,
            "tag_dropout_rate": 1.0,
        },
    )
    dropped = _process_cached_items(
        [_natural_item()], epoch_num=0, caption_config={"caption_dropout_rate": 1.0}
    )

    assert kept[0]["caption"] == "A natural language caption."
    assert dropped[0]["caption"] == ""


def test_natural_language_dropout_draws_once_per_item_per_epoch(monkeypatch):
    draws = iter((0.25, 0.75, 0.75, 0.25))
    monkeypatch.setattr(caption_processor.random, "random", lambda: next(draws))
    items = [
        _natural_item("first", "first.png"),
        _natural_item("second", "second.png"),
    ]

    epoch_zero = _process_cached_items(
        items, epoch_num=0, caption_config={"caption_dropout_rate": 0.5}
    )
    epoch_one = _process_cached_items(
        items, epoch_num=1, caption_config={"caption_dropout_rate": 0.5}
    )

    assert [item["caption"] for item in epoch_zero] == ["", "second"]
    assert [item["caption"] for item in epoch_one] == ["first", ""]


def test_fast_per_epoch_shuffle_is_stable_and_does_not_touch_global_rng():
    config = {"shuffle_tokens": True, "shuffle_per_epoch": True}
    random.seed(2468)
    state_before = random.getstate()

    first = process_caption_with_tag_data(TAG_DATA, 3, "sample.png", config)
    second = process_caption_with_tag_data(TAG_DATA, 3, "sample.png", config)

    assert first == second
    assert random.getstate() == state_before


def test_fast_per_epoch_tag_dropout_does_not_touch_global_rng():
    config = {"tag_dropout_rate": 0.5, "tag_dropout_per_epoch": True}
    random.seed(1357)
    state_before = random.getstate()

    first = process_caption_with_tag_data(TAG_DATA, 2, "sample.png", config)
    second = process_caption_with_tag_data(TAG_DATA, 2, "sample.png", config)

    assert first == second
    assert random.getstate() == state_before


def test_per_epoch_shuffle_changes_between_epochs_and_matches_legacy_path():
    config = {"shuffle_tokens": True, "shuffle_per_epoch": True}
    raw_caption = ", ".join(item["tag"] for item in TAG_DATA)
    fast_results = {
        process_caption_with_tag_data(TAG_DATA, epoch, "sample.png", config)
        for epoch in range(8)
    }

    assert len(fast_results) > 1
    for epoch in range(8):
        fast = process_caption_with_tag_data(TAG_DATA, epoch, "sample.png", config)
        legacy = process_caption(
            raw_caption,
            epoch_num=epoch,
            item_path="sample.png",
            normalize_tags=False,
            shuffle_tokens=True,
            shuffle_per_epoch=True,
        )
        assert fast == legacy


def test_non_per_epoch_shuffle_keeps_using_the_global_random_stream():
    random.seed(97531)
    state_before = random.getstate()

    process_caption_with_tag_data(
        TAG_DATA, 0, "sample.png", {"shuffle_tokens": True, "shuffle_per_epoch": False}
    )

    assert random.getstate() != state_before
