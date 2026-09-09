from pathlib import Path
import random
import sys
from types import SimpleNamespace

import torch


BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


from core.training.bucketing import BucketManager  # noqa: E402
from core.training.base_trainer import BaseTrainer  # noqa: E402
from core.training.sensenova_tasks import (  # noqa: E402
    build_text_supervision,
    build_task_homogeneous_batches,
    canonicalize_tags,
    keep_hints_for_example,
    eligible_task_views,
    required_caption_types,
    resolve_text_target,
    task_views_signature,
    task_step_metrics,
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
                "_captions_by_type": {
                    "natural_language": {"content": "A person.", "is_tags_format": False},
                    "tags": {"content": "1girl", "is_tags_format": True},
                },
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


def test_task_view_signature_covers_order_weights_and_sources():
    base = [{"dataset_id": 1, "task_views": VIEWS}]
    assert task_views_signature(base) == task_views_signature(base)
    changed = [{"dataset_id": 1, "task_views": [dict(VIEWS[0], weight=4), VIEWS[1]]}]
    assert task_views_signature(base) != task_views_signature(changed)
    assert task_views_signature(base) != task_views_signature([
        {"dataset_id": 1, "task_views": list(reversed(VIEWS))}
    ])


def test_item_eligibility_tracks_targets_and_ti2i_reference():
    item = _batches(1, 0)[0][0][0]
    assert [view["task"] for view in eligible_task_views(item)] == [
        "i2t_caption", "i2t_tags"
    ]
    item["_captions_by_type"]["tags"]["content"] = ""
    assert [view["task"] for view in eligible_task_views(item)] == ["i2t_caption"]
    item["_captions_by_type"]["natural_language"]["content"] = ""
    assert eligible_task_views(item) == []


def test_task_metrics_report_cumulative_draws_and_throughput():
    draws = {"i2t_caption": 3}
    metrics = task_step_metrics(
        "i2t_caption", 2, 0.5, draws, target_tokens=40
    )
    assert draws == {"i2t_caption": 5}
    assert metrics == {
        "task_items_i2t_caption": 2.0,
        "task_draws_i2t_caption": 5.0,
        "task_items_per_second_i2t_caption": 4.0,
        "i2t_target_tokens": 40.0,
        "task_target_tokens_per_second_i2t_caption": 80.0,
    }


def test_all_task_loss_and_throughput_metrics_are_registered():
    from core.training.metric_registry import EXTRA_METRIC_DEFS

    for task in ("i2t_caption", "i2t_tags", "i2t_caption_tags", "t2i", "ti2i"):
        assert f"task_draws_{task}" in EXTRA_METRIC_DEFS
        assert f"task_items_per_second_{task}" in EXTRA_METRIC_DEFS
    for task in ("i2t_caption", "i2t_tags", "i2t_caption_tags"):
        assert f"loss_ce_{task}" in EXTRA_METRIC_DEFS
        assert f"task_target_tokens_per_second_{task}" in EXTRA_METRIC_DEFS
    for task in ("t2i", "ti2i"):
        assert f"loss_flow_{task}" in EXTRA_METRIC_DEFS


def test_selected_view_does_not_mutate_persistent_bucket_item():
    source = _batches(1, 0)[0][0][0]
    result = build_task_homogeneous_batches([[(source, "large")]], 1, random.Random(1))
    assert "_sensenova_task" not in source
    assert result[0][0][0]["_sensenova_task"] in {"i2t_caption", "i2t_tags"}


def test_same_flow_task_with_different_loss_weights_forms_separate_batches():
    items = []
    for index, loss_weight in enumerate((0.5, 2.0)):
        items.append(({
            "image_path": f"flow-{index}.png",
            "width": 384,
            "height": 384,
            "_sensenova_task_views": [{
                "task": "t2i",
                "target_caption_types": ["natural_language"],
                "loss_weight": loss_weight,
            }],
            "_captions_by_type": {
                "natural_language": {"content": "A person."},
            },
        }, "dataset"))
    batches = build_task_homogeneous_batches([items], 8, random.Random(1))
    assert len(batches) == 2
    assert {
        batch[0][0]["_sensenova_task_view"]["loss_weight"] for batch in batches
    } == {0.5, 2.0}


def test_flow_task_loss_weight_scales_backward_but_not_reported_loss():
    class Harness:
        _microbatch_two_stage = BaseTrainer._microbatch_two_stage
        _slice_aux = staticmethod(lambda _value, _lo, _hi: None)

        def __init__(self):
            self.scales = []

        def _execute_forward_backward(self, **kwargs):
            self.scales.append(kwargs["loss_scale"])
            return 2.0, 3.0, 0.0

    batch = {
        "mnt_latents": torch.zeros(2, 1, 1, 1),
        "mnt_text_embeddings": None,
        "mnt_pooled_embeddings": None,
        "mnt_repa_pixels": None,
        "mnt_attention_mask": None,
        "timesteps": torch.zeros(2),
        "debug_save_path": None,
        "batch_captions": None,
        "batch_reference_paths": None,
        "alphas_cumprod_cached": None,
        "use_condition_images": False,
        "condition_images_batch": None,
        "reference_latents_nested": None,
        "lens_latent_shape": None,
        "mnt_time_ids": None,
        "loss_weight_maps_batch": None,
        "sensenova_prefix": None,
        "sensenova_text_batch": None,
        "cfg_drop_mask": None,
        "_sensenova_task_loss_weight": 2.5,
    }
    harness = Harness()
    loss, pred, _recon = harness._microbatch_two_stage(1, 2, batch)
    assert harness.scales == [1.25, 1.25]
    assert loss == 2.0
    assert pred == 3.0


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


class _CharTokenizer:
    def __call__(self, text, return_tensors="pt", return_offsets_mapping=False):
        result = {
            "input_ids": torch.tensor([[ord(char) for char in text]], dtype=torch.long),
            "attention_mask": torch.ones(1, len(text), dtype=torch.long),
        }
        if return_offsets_mapping:
            result["offset_mapping"] = torch.tensor(
                [[[index, index + 1] for index in range(len(text))]], dtype=torch.long
            )
        return result


def test_text_supervision_masks_prompt_and_serializes_canonical_target():
    transformer = SimpleNamespace(
        template="internvl2_5",
        system_message="system",
        downsample_ratio=0.5,
    )
    captions = {
        "natural_language": {"content": "A cat sits.", "is_tags_format": False},
        "tags": {"content": "cat, 1girl, cat", "is_tags_format": True},
    }
    result = build_text_supervision(
        transformer,
        _CharTokenizer(),
        torch.tensor([[4, 4]]),
        {
            "task": "i2t_caption_tags",
            "target_caption_types": ["natural_language", "tags"],
            "hint_caption_types": ["tags"],
            "hint_dropout": 0,
            "prompt_template_version": 1,
        },
        captions,
        image_path="cat.png",
        epoch=2,
        run_seed=10,
    )
    assert result["target"] == '{"caption":"A cat sits.","tags":["1girl","cat"]}'
    active = result["labels"][0] != -100
    supervised = "".join(chr(value) for value in result["input_ids"][0, active].tolist())
    assert supervised.startswith(result["target"])
    assert "Input hint tags" not in supervised
    assert result["target_tokens"] == int(active.sum())


def test_hint_dropout_is_seeded_per_example():
    kwargs = dict(
        run_seed=42, epoch=3, image_path="x.png", task="i2t_caption",
        template_version=1, dropout=0.5,
    )
    assert keep_hints_for_example(**kwargs) == keep_hints_for_example(**kwargs)
    assert keep_hints_for_example(**{**kwargs, "dropout": 0}) is True
    assert keep_hints_for_example(**{**kwargs, "dropout": 1}) is False


def test_target_sources_are_explicit_and_missing_is_an_error():
    assert canonicalize_tags(["z, a, z"]) == ["a", "z"]
    try:
        resolve_text_target("i2t_caption", {}, ["natural_language"])
    except ValueError as exc:
        assert "missing target caption" in str(exc)
    else:
        raise AssertionError("missing target source was accepted")
