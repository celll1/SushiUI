"""Concept batches preserve coverage and resume identity across placements."""

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training.concept_batch_order import ConceptOrderConfig, build_concept_batch_plan


def _items():
    dataset = SimpleNamespace(unique_id="test")
    out = []
    for concept in ("hatsune_miku", "kagamine_rin", "megurine_luka"):
        for i in range(10):
            out.append(({
                "image_path": f"{concept}_{i}.png", "raw_caption": concept,
                "tag_data": json.dumps([{"tag": concept, "category": "Character"}]),
                "is_tags_format": True,
                "bucket_width": 512 if i % 2 else 768,
                "bucket_height": 512,
            }, dataset))
    for i in range(30):
        out.append(({
            "image_path": f"background_{i}.png", "raw_caption": "landscape",
            "tag_data": json.dumps([{"tag": "landscape", "category": "General"}]),
            "is_tags_format": True, "bucket_width": 512, "bucket_height": 512,
        }, dataset))
    return out


def _config(**changes):
    defaults = {"enabled": True, "min_items_per_concept": 2}
    return ConceptOrderConfig.parse({**defaults, **changes})


def _paths(plan):
    return [item["image_path"] for batch in plan.batches
            for item, _ in batch]


def test_front_and_spread_cover_every_item_once_and_rebuild_exactly():
    items = _items()
    for placement in ("front", "spread"):
        config = _config(background_placement=placement, local_swap_window=2)
        original = build_concept_batch_plan(items, 4, config, 0, 42)
        rebuilt = build_concept_batch_plan(list(reversed(items)), 4, config, 0, 42)
        assert original.digest == rebuilt.digest
        assert _paths(original) == _paths(rebuilt)
        assert sorted(_paths(original)) == sorted(item["image_path"] for item, _ in items)
        assert original.concept_count == 3
        assert original.focus_items == 30
        assert original.background_items == 30
        for batch in original.batches:
            assert len({(item["bucket_width"], item["bucket_height"])
                        for item, _ in batch}) == 1
        assert _paths(original)[7:] == _paths(rebuilt)[7:]


def test_spread_places_background_throughout_epoch():
    items = _items()
    front = build_concept_batch_plan(items, 4, _config(background_placement="front"), 0, 3)
    spread = build_concept_batch_plan(items, 4, _config(background_placement="spread"), 0, 3)
    front_last_focus = max(i for i, name in enumerate(front.labels) if name is not None)
    spread_last_focus = max(i for i, name in enumerate(spread.labels) if name is not None)
    assert spread_last_focus > front_last_focus


def test_caption_aliases_match_boundaries_without_partial_name_hits():
    dataset = SimpleNamespace(unique_id="test")
    items = [
        ({"image_path": "miku.png", "raw_caption": "Hatsune Miku standing",
          "is_tags_format": False, "width": 512, "height": 512}, dataset),
        ({"image_path": "other.png", "raw_caption": "Hatsune Mikuru standing",
          "is_tags_format": False, "width": 512, "height": 512}, dataset),
    ]
    config = _config(min_items_per_concept=1, match_natural_language=True,
                     caption_aliases={"hatsune_miku": ["Miku Hatsune"]})
    plan = build_concept_batch_plan(items, 1, config, 0, 1)
    assert plan.focus_items == 1
    assert plan.background_items == 1


def test_replay_is_explicit_and_changes_plan_digest():
    items = _items()
    base = build_concept_batch_plan(items, 4, _config(), 0, 8)
    replay = build_concept_batch_plan(items, 4, _config(replay_interval=2), 0, 8)
    assert replay.replay_batches > 0
    assert len(replay.batches) > len(base.batches)
    assert replay.digest != base.digest
    assert sorted(_paths(base)) == sorted(item["image_path"] for item, _ in items)


def test_plan_state_detects_caption_and_reference_changes_on_resume():
    items = _items()
    config = _config()
    original = build_concept_batch_plan(items, 4, config, 0, 8)
    state = original.state(config, 8, None)
    rebuilt = build_concept_batch_plan(list(reversed(items)), 4, config, 0, 8)
    assert state == rebuilt.state(config, 8, None)
    changed = [(dict(item), dataset) for item, dataset in items]
    changed[0][0]["raw_caption"] = "different caption"
    assert build_concept_batch_plan(changed, 4, config, 0, 8).state(config, 8, None) != state
    changed[0][0]["raw_caption"] = items[0][0]["raw_caption"]
    changed[0][0]["reference_images"] = ["reference.png"]
    assert build_concept_batch_plan(changed, 4, config, 0, 8).state(config, 8, None) != state
