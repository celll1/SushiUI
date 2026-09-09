from pathlib import Path
import random
import sys


BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


from core.training.bucketing import BucketManager  # noqa: E402
from core.training.sensenova_tasks import (  # noqa: E402
    build_task_homogeneous_batches,
    required_caption_types,
)
from core.training.train_runner import _process_cached_items  # noqa: E402


VIEWS = [
    {
        "task": "i2t_caption",
        "target_caption_types": ["natural_language"],
        "hint_caption_types": ["tags"],
        "weight": 3,
    },
    {
        "task": "i2t_tags",
        "target_caption_types": ["tags"],
        "hint_caption_types": [],
        "weight": 1,
    },
]


def _batches(count_a=100, count_b=10):
    pairs = []
    for dataset, count in (("large", count_a), ("small", count_b)):
        for index in range(count):
            pairs.append(({
                "image_path": f"{dataset}-{index}.png",
                "width": 384,
                "height": 384,
                "_sensenova_task_views": VIEWS,
            }, dataset))
    return [pairs[index:index + 8] for index in range(0, len(pairs), 8)]


def _signature(batches):
    return [
        [(item["image_path"], item["_sensenova_task"]) for item, _dataset in batch]
        for batch in batches
    ]


def test_caption_source_union_is_stable_and_deduplicated():
    assert required_caption_types(VIEWS) == ["natural_language", "tags"]


def test_task_batches_are_homogeneous_and_resume_reproducible():
    rng = random.Random(1234)
    state = rng.getstate()
    first = build_task_homogeneous_batches(_batches(), 8, rng)
    rng.setstate(state)
    resumed = build_task_homogeneous_batches(_batches(), 8, rng)
    assert _signature(first) == _signature(resumed)
    assert all(len({item["_sensenova_task"] for item, _ in batch}) == 1 for batch in first)


def test_multiple_datasets_remain_item_proportional():
    result = build_task_homogeneous_batches(_batches(100, 10), 8, random.Random(9))
    counts = {"large": 0, "small": 0}
    for batch in result:
        for _item, dataset in batch:
            counts[dataset] += 1
    assert counts == {"large": 100, "small": 10}


def test_selected_view_does_not_mutate_persistent_bucket_item():
    source = _batches(1, 0)[0][0][0]
    result = build_task_homogeneous_batches([[(source, "large")]], 1, random.Random(1))
    assert "_sensenova_task" not in source
    assert result[0][0][0]["_sensenova_task"] in {"i2t_caption", "i2t_tags"}


def test_bucket_metadata_survives_bucket_record_copy():
    manager = BucketManager([384])
    _, item = manager.assign_image_to_bucket(
        "x.png", 384, 384,
        item_metadata={"_sensenova_task_views": VIEWS, "_captions_by_type": {"tags": {"content": "1girl"}}},
    )
    assert item["_sensenova_task_views"] == VIEWS
    assert item["_captions_by_type"]["tags"]["content"] == "1girl"


def test_epoch_caption_processing_preserves_auxiliary_sources():
    sources = {"natural_language": {"content": "A person.", "is_tags_format": False}}
    result = _process_cached_items(
        [{
            "image_path": "x.png",
            "raw_caption": "primary",
            "tag_data": None,
            "is_tags_format": False,
            "width": 64,
            "height": 64,
            "_captions_by_type": sources,
        }],
        epoch_num=0,
        caption_config={"caption_dropout_rate": 0},
    )
    assert result[0]["_captions_by_type"] == sources

