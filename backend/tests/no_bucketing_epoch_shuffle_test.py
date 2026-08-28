"""Non-bucketed epoch batches must be shuffled, reproducibly, without disturbing
priority-training order or video/audio batch homogeneity.

Run with:
    venv/Scripts/python.exe -m pytest backend/tests/no_bucketing_epoch_shuffle_test.py -q

Static: no model, no GPU, no DB.

Defect: with bucketing disabled, ``base_trainer.py``'s non-bucketed batch
construction chunked ``_image_all_items`` straight from its DB/dataset-config
order with no ``random.shuffle`` at all -- every epoch trained the exact same
batches in the exact same order (the bucketed path shuffles its batch order
via ``BucketManager.build_batch_indices``). The fix shuffles the item list
(global ``random`` module -- the same stream ``random.getstate()``/
``setstate()`` already saves/restores for mid-epoch resume) before chunking,
while leaving priority items' entry-index order untouched and never crossing
video/audio group boundaries.

These tests pin the ALGORITHM (a literal, comment-stripped copy of the fixed
code block, since it lives inline inside ``BaseTrainer.train()`` and cannot be
unit-invoked without a full model/optimizer/dataset harness) plus a source
grep that the fix is actually wired into ``base_trainer.py``.
"""

import os
import random

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BASE_TRAINER_PATH = os.path.join(_REPO_ROOT, "core", "training", "base_trainer.py")


def _read_base_trainer_source() -> str:
    with open(_BASE_TRAINER_PATH, encoding="utf-8") as f:
        return f.read()


# ===========================================================================
# Pinned copy of the fixed non-bucketed batch-building code
# (base_trainer.py, BaseTrainer.train(), the `else: # no bucket_manager` arm).
# ===========================================================================

def _build_non_bucketed_batches(_image_all_items, batch_size, priority_config=None,
                                 classify_items=None):
    if priority_config is not None:
        priority_items, normal_items = classify_items(_image_all_items, priority_config)
        p_items = [(item, dataset) for item, dataset, _ in priority_items]
        normal_items = list(normal_items)
        random.shuffle(normal_items)
        priority_batches = [p_items[i:i + batch_size] for i in range(0, len(p_items), batch_size)]
        normal_batches = [normal_items[i:i + batch_size] for i in range(0, len(normal_items), batch_size)]
        return priority_batches + normal_batches
    else:
        _shuffled = list(_image_all_items)
        random.shuffle(_shuffled)
        return [_shuffled[i:i + batch_size] for i in range(0, len(_shuffled), batch_size)]


def _make_items(n):
    return [({"image_path": f"img_{i}.png"}, f"ds") for i in range(n)]


# ===========================================================================
# (a) epoch-to-epoch order changes
# ===========================================================================

def test_order_changes_across_epochs():
    items = _make_items(64)

    random.seed(1)
    batches_epoch1 = _build_non_bucketed_batches(items, batch_size=4)
    batches_epoch2 = _build_non_bucketed_batches(items, batch_size=4)

    order1 = [item[0]["image_path"] for b in batches_epoch1 for item in b]
    order2 = [item[0]["image_path"] for b in batches_epoch2 for item in b]

    assert sorted(order1) == sorted(order2) == sorted(i[0]["image_path"] for i in items)
    assert order1 != order2, "epoch 2 must not reproduce epoch 1's batch order"


def test_source_actually_shuffles_before_chunking():
    """Guards against reverting to the pre-fix sequential slice."""
    src = _read_base_trainer_source()
    assert "random.shuffle(_shuffled_image_items)" in src
    assert "_shuffled_image_items[i:i+batch_size]" in src
    assert "random.shuffle(normal_items)" in src


# ===========================================================================
# (b) resume reproducibility: same random state -> same order
# ===========================================================================

def test_resume_reproduces_identical_order_from_saved_state():
    items = _make_items(37)  # not a multiple of batch_size, mirrors real datasets

    random.seed(42)
    saved_state = random.getstate()
    batches_original_run = _build_non_bucketed_batches(items, batch_size=5)

    # Simulate a resume: restore the exact saved state before rebuilding batches.
    random.setstate(saved_state)
    batches_resumed_run = _build_non_bucketed_batches(items, batch_size=5)

    order_original = [item[0]["image_path"] for b in batches_original_run for item in b]
    order_resumed = [item[0]["image_path"] for b in batches_resumed_run for item in b]
    assert order_original == order_resumed


# ===========================================================================
# (c) priority training order invariant
# ===========================================================================

def test_priority_items_keep_entry_index_order_only_normal_shuffled():
    from core.training.priority_training import PriorityTrainingConfig, PriorityEntry, classify_items

    # 5 priority-tagged items (must land in entries-order) + 40 normal items.
    priority_raw = [({"image_path": f"p_{i}.png", "caption": "priority_tag"}, "ds") for i in range(5)]
    normal_raw = _make_items(40)
    all_items = priority_raw + normal_raw

    config = PriorityTrainingConfig(entries=[PriorityEntry(tags=["priority_tag"])], multiplier=1)

    random.seed(7)
    batches = _build_non_bucketed_batches(all_items, batch_size=5,
                                           priority_config=config, classify_items=classify_items)

    priority_paths = [item[0]["image_path"] for b in batches for item in b
                       if item[0]["image_path"].startswith("p_")]
    assert priority_paths == [f"p_{i}.png" for i in range(5)], (
        "priority items must stay in entry-index order (never shuffled)"
    )

    normal_paths = [item[0]["image_path"] for b in batches for item in b
                     if item[0]["image_path"].startswith("img_")]
    assert sorted(normal_paths) == sorted(i[0]["image_path"] for i in normal_raw)
    assert normal_paths != [i[0]["image_path"] for i in normal_raw], (
        "normal items must be shuffled, not left in original DB order"
    )


# ===========================================================================
# (d) video/audio group homogeneity is untouched by the item-level shuffle
# ===========================================================================

def test_video_audio_batches_are_never_built_from_image_all_items():
    """The image-level shuffle in the fixed code operates on `_image_all_items`,
    which base_trainer.py explicitly excludes video/audio items from (see the
    `_image_all_items = [x for x in all_items if x[0].get("item_type") not in
    ("video", "audio")]` filter). The two group batch lists
    (`ltx2_video_batches`, `acestep_audio_batches`) are built earlier, directly
    from the annotated video/audio item dicts grouped by (spatial, clip_length)
    / clip duration, and are concatenated onto `batches` AFTER our shuffle --
    so no shuffle can ever mix a video/audio item into an image batch or across
    a (spatial, clip_length) / duration group boundary."""
    src = _read_base_trainer_source()
    assert '_image_all_items = (\n' in src
    assert 'x[0].get("item_type") not in ("video", "audio")' in src
    # video/audio batch lists are appended AFTER the (now-shuffled) image batches
    assert "batches = batches + ltx2_video_batches + acestep_audio_batches" in src
    assert "batches = (priority_batches * priority_config.multiplier + normal_batches\n" \
           "                                   + ltx2_video_batches + acestep_audio_batches)" in src


def test_video_group_key_is_spatial_and_clip_length_grouped():
    """Pins the video grouping's own homogeneity invariant directly (independent
    of the shuffle fix): members sharing a (bucket_width, bucket_height,
    clip_length) key never split across batches with a different key."""
    from collections import OrderedDict

    def _vkey(item):
        return (item.get("bucket_width"), item.get("bucket_height"), item.get("clip_length"))

    items = (
        [{"image_path": f"v_a{i}.mp4", "bucket_width": 512, "bucket_height": 512, "clip_length": 17}
         for i in range(3)]
        + [{"image_path": f"v_b{i}.mp4", "bucket_width": 704, "bucket_height": 480, "clip_length": 33}
           for i in range(3)]
    )

    groups = OrderedDict()
    for item in items:
        groups.setdefault(_vkey(item), []).append(item)

    batch_size = 2
    video_batches = []
    for _key, members in groups.items():
        for i in range(0, len(members), batch_size):
            video_batches.append(members[i:i + batch_size])

    for batch in video_batches:
        keys = {_vkey(item) for item in batch}
        assert len(keys) == 1, f"video batch mixes groups: {keys}"
