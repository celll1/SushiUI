"""Exposure counts only completed passes and survives checkpoint restore."""

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.training.concept_exposure import ConceptExposure


def test_counts_repeated_items_and_restores_from_checkpoint(tmp_path):
    ds = SimpleNamespace(unique_id="dataset-1")
    a = ({"image_path": "a.png", "_concept_exposure_group": "miku"}, ds)
    b = ({"image_path": "b.png", "_concept_exposure_group": "miku"}, ds)
    other = ({"image_path": "other.png"}, ds)
    tracker = ConceptExposure(tmp_path, "concept")
    tracker.set_epoch({"miku": 2}, {"miku": "Miku"})
    tracker.record([a, other], 1, 1)
    tracker.record([a, b], 2, 0)  # skipped batch
    tracker.record([a, b], 3, 2)  # two completed noise passes
    tracker.publish(wait=True)

    data = json.loads((tmp_path / "concept_exposure.json").read_text(encoding="utf-8"))
    assert data["counts"] == {"miku": 5}
    assert data["target_items"] == {"miku": 2}
    assert data["last_step"] == {"miku": 3}

    saved = tracker.state()
    tracker.record([a], 5, 1)
    assert saved["counts"] == {"miku": 5}
    assert saved["last_step"] == {"miku": 3}

    resumed = ConceptExposure(tmp_path, "concept", saved)
    resumed.set_epoch({"miku": 2}, {"miku": "Miku"})
    resumed.record([b], 4, 1)
    assert resumed.state()["counts"] == {"miku": 6}
    assert resumed.state()["target_items"] == {"miku": 2}


def test_api_orders_snapshot_without_trainer_access(tmp_path):
    from api.routes import get_training_concept_exposure

    class Db:
        def query(self, _model):
            return self

        def filter(self, _condition):
            return self

        def first(self):
            return SimpleNamespace(output_dir=str(tmp_path))

    data = {"mode": "concept", "counts": {"miku": 10, "rin": 5},
            "last_step": {"miku": 2, "rin": 7},
            "target_items": {"miku": 2, "rin": 5},
            "names": {"miku": "Miku", "rin": "Rin"}}
    (tmp_path / "concept_exposure.json").write_text(json.dumps(data), encoding="utf-8")
    top = get_training_concept_exposure(1, order="top", limit=1, db=Db())
    latest = get_training_concept_exposure(1, order="latest", limit=1, db=Db())
    assert top["rows"][0]["name"] == "Miku"
    assert top["rows"][0]["mean_passes_per_image"] == 5
    assert latest["rows"][0]["name"] == "Rin"


def test_concept_tags_survive_dataset_rebase_independent_of_focus_assignment(tmp_path):
    from core.training.concept_batch_order import ConceptOrderConfig, build_concept_batch_plan
    from core.training.resume_batch_plan import make_plan, rebase_plan, resolve_suffix

    item = {"image_path": "a.png", "width": 512, "height": 512,
            "raw_caption": "miku, rin", "is_tags_format": True,
            "tag_data": [{"tag": "miku", "category": "Character"},
                         {"tag": "rin", "category": "Character"}]}
    a = SimpleNamespace(unique_id="a", items=[item])
    b = SimpleNamespace(unique_id="b", items=[
        {"image_path": "b.png", "width": 512, "height": 512,
         "raw_caption": "rin", "is_tags_format": True,
         "tag_data": [{"tag": "rin", "category": "Character"}]}
    ])
    config = ConceptOrderConfig.parse({"enabled": True, "min_items_per_concept": 1,
                                       "local_swap_window": 0})
    old = build_concept_batch_plan([(item, a)], 1, config, 0, 7)
    candidate = build_concept_batch_plan([(item, a), (b.items[0], b)], 1, config, 0, 7)
    old_ledger = make_plan(old.batches, [a], "concept", "same", 0)
    new_ledger = make_plan(candidate.batches, [a, b], "concept", "same", 0)
    revised, _ = rebase_plan(old_ledger, 0, new_ledger)
    suffix = resolve_suffix(revised, 0, candidate.batches)

    tracker = ConceptExposure(tmp_path, "concept")
    tracker.set_epoch({"miku": 1, "rin": 2}, {"miku": "miku", "rin": "rin"})
    for batch in suffix:
        tracker.record(batch, 1, 1)
    assert tracker.state()["counts"] == {"miku": 1, "rin": 2}


def test_timestep_histogram_tracks_per_item_repeats_mnt_and_resume(tmp_path):
    from core.training.concept_exposure import TIMESTEP_BIN_EDGES
    ds = SimpleNamespace(unique_id="dataset")
    a = ({"image_path": "a", "_concept_exposure_groups": ("miku", "rin")}, ds)
    b = ({"image_path": "b", "_concept_exposure_group": "miku"}, ds)
    old = {"mode": "concept", "counts": {"miku": 7}}
    tracker = ConceptExposure(tmp_path, "concept", old)
    tracker.record([a, b], 1, 1, [0, 0.98])
    tracker.record([a, a], 2, 1, [0.95, 1])
    state = tracker.state()
    assert state["counts"] == {"miku": 11, "rin": 3}
    assert len(TIMESTEP_BIN_EDGES) == 37
    assert sum(state["timestep_counts"]["miku"]) == 4
    assert state["timestep_counts"]["miku"][0] == 1
    assert state["timestep_counts"]["miku"][-1] == 1
    tracker.record([a], 3, 1, [0.8])
    assert sum(state["timestep_counts"]["miku"]) == 4  # Snapshot is isolated from later updates.
    tracker.publish(wait=True)
    disk = json.loads(tracker.path.read_text(encoding="utf-8"))
    assert disk["version"] == 2
    resumed = ConceptExposure(tmp_path, "concept", disk)
    resumed.record([b], 4, 1, [0.99])
    assert sum(resumed.state()["timestep_counts"]["miku"]) == 6
    assert resumed.state()["counts"]["miku"] == 13


def test_timestep_convention_unknown_values_and_invalid_resume(tmp_path):
    import pytest
    from core.training.concept_exposure import TIMESTEP_BIN_EDGES
    ds = SimpleNamespace(unique_id="dataset")
    item = ({"image_path": "a", "_concept_exposure_group": "priority-0"}, ds)
    tracker = ConceptExposure(tmp_path, "priority")
    tracker.record([item], 1, 1, [0], convention="t1")
    assert tracker.state()["timestep_counts"]["priority-0"][-1] == 1
    tracker.record([item], 2, 1, [1], convention="t1")
    assert tracker.state()["timestep_counts"]["priority-0"][0] == 1
    for value in (float("nan"), float("inf"), -0.1, 1.1):
        tracker.record([item], 3, 1, [value])
    assert sum(tracker.state()["timestep_counts"]["priority-0"]) == 2
    before = tracker.state()
    tracker.record([item], 4, 0, [0.5])
    assert tracker.state() == before
    with pytest.raises(ValueError):
        tracker.record([item], 4, 1, [])
    with pytest.raises(ValueError):
        ConceptExposure(tmp_path, "priority", {**before, "timestep_bin_edges": [0, 1]})
    with pytest.raises(ValueError):
        ConceptExposure(tmp_path, "priority", {**before, "timestep_counts": {"x": [1]}})


def test_api_histogram_preserves_legacy_unknown_and_known_counts(tmp_path):
    from api.routes import get_training_concept_exposure
    from core.training.concept_exposure import TIMESTEP_BIN_EDGES
    class Db:
        def query(self, _model): return self
        def filter(self, _condition): return self
        def first(self): return SimpleNamespace(output_dir=str(tmp_path))
    bins = [0] * (len(TIMESTEP_BIN_EDGES) - 1)
    bins[-1] = 3
    data = {"mode": "priority", "counts": {"old": 5, "new": 8},
            "timestep_bin_edges": list(TIMESTEP_BIN_EDGES), "timestep_counts": {"new": bins}}
    (tmp_path / "concept_exposure.json").write_text(json.dumps(data), encoding="utf-8")
    result = get_training_concept_exposure(1, order="top", limit=10, db=Db())
    assert result["timestep_bin_edges"] == list(TIMESTEP_BIN_EDGES)
    new, old = result["rows"]
    assert new["timestep_recorded_passes"] == 3
    assert new["timestep_unknown_passes"] == 5
    assert old["timestep_recorded_passes"] == 0
    assert old["timestep_unknown_passes"] == 5
