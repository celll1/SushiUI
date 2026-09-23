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
