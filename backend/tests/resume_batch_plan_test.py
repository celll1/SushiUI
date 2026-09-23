"""A dataset-selection rebase never replays completed logical occurrences."""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training.resume_batch_plan import (
    load_plan, make_plan, rebase_plan, resolve_suffix, save_plan,
)


def _dataset(name, *paths):
    return SimpleNamespace(unique_id=name, items=[
        {"image_path": path, "width": 512, "height": 512,
         "raw_caption": path, "is_tags_format": False}
        for path in paths
    ])


def _batches(*datasets):
    return [[(item, dataset)] for dataset in datasets for item in dataset.items]


def _ids(batches):
    return [(entry[0], entry[1], entry[2])
            for batch in batches for entry in batch]


def test_add_remove_and_readd_preserve_completed_occurrences(tmp_path):
    a = _dataset("a", "a0", "a1")
    b = _dataset("b", "b0", "b1")
    c = _dataset("c", "c0")
    original = make_plan(_batches(a, b), [a, b], "normal", "same", 0)
    reference = save_plan(tmp_path, "run", original)
    assert load_plan(tmp_path, reference) == original

    candidate = make_plan(_batches(a, c), [a, c], "normal", "same", 0)
    revised, stats = rebase_plan(original, 2, candidate)
    assert stats["removed_datasets"] == ["b"]
    assert _ids(revised["batches"][:2]) == _ids(original["batches"][:2])
    assert _ids(revised["batches"][2:]) == [("c", "c0", 0)]
    assert [pair[0]["image_path"]
            for batch in resolve_suffix(revised, 2, _batches(a, c))
            for pair in batch] == ["c0"]

    # The same dataset is selected again after its first occurrence was trained.
    resumed = make_plan(_batches(a, b, c), [a, b, c], "normal", "same", 0)
    again, _ = rebase_plan(revised, 3, resumed)
    assert _ids(again["batches"][:3]) == _ids(revised["batches"][:3])
    assert _ids(again["batches"][3:]) == [("b", "b0", 0), ("b", "b1", 0)]


def test_priority_repeats_are_distinct_obligations():
    a = _dataset("a", "a0")
    b = _dataset("b", "b0")
    original = make_plan([[(a.items[0], a)]] * 3, [a], "priority", "same", 0,
                         {("a", "a0")})
    candidate = make_plan([[(b.items[0], b)]] * 3 + [[(a.items[0], a)]] * 3,
                          [a, b], "priority", "same", 0, {("a", "a0"), ("b", "b0")})
    revised, _ = rebase_plan(original, 1, candidate)
    assert _ids(revised["batches"][:1]) == [("a", "a0", 0)]
    assert sorted(_ids(revised["batches"][1:])) == sorted([
        ("a", "a0", 1), ("a", "a0", 2),
        ("b", "b0", 0), ("b", "b0", 1), ("b", "b0", 2)])


def test_concept_addition_keeps_old_replay_without_new_replay():
    a = _dataset("a", "a0")
    b = _dataset("b", "b0")
    original = make_plan([[(a.items[0], a)]] * 2, [a], "concept", "same", 0)
    candidate = make_plan([[(b.items[0], b)]] * 2 + [[(a.items[0], a)]] * 2,
                          [a, b], "concept", "same", 0)
    candidate["placement"] = "front"
    revised, stats = rebase_plan(original, 1, candidate)
    assert stats["added_occurrences"] == 1
    assert _ids(revised["batches"]) == [
        ("a", "a0", 0), ("b", "b0", 0), ("a", "a0", 1)]


def test_internal_change_and_corrupt_ledger_fail(tmp_path):
    a = _dataset("a", "a0")
    original = make_plan(_batches(a), [a], "normal", "same", 0)
    changed = _dataset("a", "different")
    with pytest.raises(ValueError, match="changed internally"):
        rebase_plan(original, 0, make_plan(_batches(changed), [changed],
                                           "normal", "same", 0))
    reference = save_plan(tmp_path, "run", original)
    (tmp_path / reference["file"]).write_bytes(b"bad")
    with pytest.raises(ValueError, match="ledger size mismatch"):
        load_plan(tmp_path, reference)


def test_unchanged_selection_keeps_plan_revision():
    a = _dataset("a", "a0", "a1")
    original = make_plan(_batches(a), [a], "normal", "same", 0)
    resumed = make_plan(list(reversed(_batches(a))), [a], "normal", "same", 0)
    same, stats = rebase_plan(original, 1, resumed)
    assert same == original
    assert stats["remaining_batches"] == 1


def test_caption_change_is_reported_for_normal_order_but_not_replayed():
    a = _dataset("a", "a0", "a1")
    original = make_plan(_batches(a), [a], "normal", "same", 0)
    edited = _dataset("a", "a0", "a1")
    edited.items[1]["raw_caption"] = "new training text"
    resumed = make_plan(_batches(edited), [edited], "normal", "same", 0)
    revised, stats = rebase_plan(original, 1, resumed)
    assert stats["caption_changed"] == ["a"]
    assert _ids(revised["batches"][1:]) == [("a", "a1", 0)]
    with pytest.raises(ValueError, match="changed internally"):
        rebase_plan(make_plan(_batches(a), [a], "priority", "same", 0), 1,
                    make_plan(_batches(edited), [edited], "priority", "same", 0))


def test_changed_bucket_conditions_fail_before_training():
    a = _dataset("a", "a0")
    original = make_plan(_batches(a), [a], "normal", "same", 0)
    candidate = _batches(a)
    candidate[0][0][0]["bucket_width"] = 768
    with pytest.raises(ValueError, match="conditions changed"):
        resolve_suffix(original, 0, candidate)


def test_training_request_policy_validation():
    from api.param_defaults import TRAINING_DEFAULTS
    from api.routes import TrainingRunCreateRequest

    request = TrainingRunCreateRequest(
        training_method="lora", base_model_path="model", total_steps=10)
    assert request.resume_dataset_change_policy == TRAINING_DEFAULTS[
        "resume_dataset_change_policy"]
    with pytest.raises(ValueError):
        TrainingRunCreateRequest(training_method="lora", base_model_path="model",
                                 total_steps=10, resume_dataset_change_policy="unknown")
    with pytest.raises(ValueError, match="image generation training"):
        TrainingRunCreateRequest(training_method="vae_decoder", base_model_path="model",
                                 total_steps=10,
                                 resume_dataset_change_policy="rebase_remaining")
