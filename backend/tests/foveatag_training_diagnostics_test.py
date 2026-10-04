"""Step-native snapshots and history must not substitute evaluation samples."""
import json
from types import SimpleNamespace

import pytest
import torch

from api import tagger_diagnostics as service
from api.param_defaults import TAGGER_DIAGNOSTIC_DEFAULTS
from core.tagger.foveatag.joint_training import FoveaJointTrainer
from core.tagger.foveatag.router_data import FoveaRouterBatch


def joint_run(tmp_path):
    return SimpleNamespace(run_id="step-native", output_dir=str(tmp_path),
        config={"architecture": "foveatag_joint_review", "joint_review": {"max_source_pixels": 100000}})


def snapshot(path, step=2000):
    return {"step": step, "epoch": 0,
        "source": {"source_path": str(path), "source_index": 17},
        "probabilities": torch.tensor([.8, .1, .9]), "positive": torch.tensor([True, True, False]),
        "known": torch.tensor([True, True, False]), "boxes": torch.tensor([[.5,.5,.4,.5]]),
        "random_slots": torch.tensor([False]), "threshold": .5, "local_size": 32,
        "prediction_origin": "training_forward_before_optimizer_update"}


def test_exact_interval_independent_of_checkpoint_and_disabled(tmp_path):
    run = joint_run(tmp_path)
    service.configure(run, {**TAGGER_DIAGNOSTIC_DEFAULTS, "enabled": True, "interval_steps": 1001})
    assert not service.training_due(run, 1000)
    assert service.training_due(run, 1001)
    assert service.training_due(run, 2002)
    service.configure(run, {**TAGGER_DIAGNOSTIC_DEFAULTS, "enabled": False})
    assert not service.training_due(run, 15000)


def test_no_source_skips_without_fixed_sample_or_subprocess(tmp_path, monkeypatch):
    run = joint_run(tmp_path)
    service.configure(run, {**TAGGER_DIAGNOSTIC_DEFAULTS, "enabled": True})
    monkeypatch.setattr(service, "_jobs", {})
    monkeypatch.setattr(service.subprocess, "Popen", lambda *a, **k: pytest.fail("No image may be substituted"))
    assert not service.schedule_training(run, {"step": 1000, "skip_reason": "no_source_metadata_in_step_batch"})
    assert service.status(run)["attempts"][0]["status"] == "skipped"
    with pytest.raises(ValueError, match="No captured"):
        service.schedule(run, force=True)


def test_snapshot_has_no_optimizer_copy_and_preserves_rng(tmp_path, monkeypatch):
    run = joint_run(tmp_path)
    service.configure(run, {**TAGGER_DIAGNOSTIC_DEFAULTS, "enabled": True})
    monkeypatch.setattr(service, "_jobs", {})
    monkeypatch.setattr(service.subprocess, "Popen", lambda *a, **k: SimpleNamespace(poll=lambda: None))
    rng = torch.get_rng_state().clone()
    assert service.schedule_training(run, snapshot(tmp_path / "unopened.png"))
    job = next(service.directory(run).glob("job-*"))
    assert (job / "training_snapshot.pt").is_file()
    assert not (job / "checkpoint.pt").exists()
    assert torch.equal(rng, torch.get_rng_state())
    assert not service.schedule_training(run, snapshot(tmp_path / "unopened.png", 3000))
    assert service.status(run)["attempts"][-1]["reason"] == "diagnostic_worker_busy"


def test_capture_uses_matching_batch_row_boxes_masks_and_exploration():
    from core.tagger.foveatag.training_metrics import TrainingMetricWindow
    captured = []
    trainer = SimpleNamespace(diagnostic_callback=captured.append, diagnostic_epoch=2,
        diagnostic_threshold=.4, selection_budget=3, random_count=1,
        diagnostic_metric_window=TrainingMetricWindow())
    trainer.diagnostic_metric_window.add({"tp": 12, "fp": 3, "fn": 4, "known_negative": 20}, 8, 999, .4)
    batch = SimpleNamespace(
        diagnostic_sources=({}, {"source_path": "actual.png"}),
        labels=torch.tensor([[0.,1.,0.],[1.,0.,1.]]), loss_masks=torch.tensor([[1.,1.,1.],[1.,0.,1.]]),
        candidate_boxes=torch.arange(32).reshape(2,4,4).float(),
        views=SimpleNamespace(local_images=torch.empty(2,4,3,32,32)))
    logits = torch.tensor([[9.,8.,7.],[-1.,2.,3.]])
    indices = torch.tensor([[0,1,2],[3,-1,1]])
    rng = torch.get_rng_state().clone()
    FoveaJointTrainer._capture_diagnostic(trainer, batch, logits, indices, 1000)
    data = captured[0]
    assert data["source"]["source_path"] == "actual.png"
    assert torch.equal(data["probabilities"], logits[1].sigmoid())
    assert torch.equal(data["boxes"], batch.candidate_boxes[1, [3,1]])
    assert data["random_slots"].tolist() == [False, True]
    assert data["known"].tolist() == [True, False, True]
    assert data["step"] == 1000 and data["epoch"] == 2
    assert data["training_window"]["samples"] == 8
    assert data["training_window"]["fp"] == 3
    assert torch.equal(rng, torch.get_rng_state())
    batch.diagnostic_sources = ()
    FoveaJointTrainer._capture_diagnostic(trainer, batch, logits, indices, 2000)
    assert captured[-1]["skip_reason"] == "no_source_metadata_in_step_batch"


def test_batch_transfer_retains_source_provenance():
    from core.tagger.foveatag.batch import FoveaViewTensorBatch
    views = FoveaViewTensorBatch(torch.zeros(1,3,32,32), torch.zeros(1,8),
        torch.zeros(1,2,3,16,16), torch.zeros(1,2,8), torch.ones(1,2,dtype=torch.bool))
    batch = FoveaRouterBatch(views, torch.zeros(1,2,4), torch.zeros(1,3), torch.ones(1,3),
        ({"source_path": "actual.png"},))
    assert batch.to("cpu").diagnostic_sources == batch.diagnostic_sources
    assert batch.stage_for_routing("cpu").diagnostic_sources == batch.diagnostic_sources


def test_source_collation_follows_training_sample_without_rating_filter():
    from PIL import Image
    from core.tagger.tagger_dataset import RawTaggerSample
    from core.tagger.foveatag.router_data import FoveaRouterDataset, collate_router_samples
    from core.tagger.foveatag.views import ViewConfig
    raw = RawTaggerSample(Image.new("RGB", (80,64)), torch.tensor([1.,0.,1.]),
        torch.ones(3), dataset_id=29, source_path="actual.png")
    source = SimpleNamespace(get_raw_item=lambda index: raw)
    dataset = FoveaRouterDataset(source, candidate_count=2,
        view_config=ViewConfig(overview_size=32, local_size=16))
    sample = dataset.sample(17, sample_seed=42)
    assert sample.diagnostic_source == {"source_path": "actual.png", "source_index": 17,
        "sample_seed": 42, "dataset_id": 29}
    assert collate_router_samples([None, sample]).diagnostic_sources == (sample.diagnostic_source,)
    raw.label[1] = 1
    assert dataset.sample(17, sample_seed=42).diagnostic_source == sample.diagnostic_source


def test_live_capture_is_wired_to_post_step_interval_not_log_or_checkpoint():
    import inspect
    code = inspect.getsource(FoveaJointTrainer.train_batch)
    assert "self.diagnostic_due(global_step + 1)" in code
    assert "self._capture_diagnostic(batch, logits, selected, global_step + 1)" in code
    from api.routes import start_tagger_training_run
    from api.param_defaults import TAGGER_TRAINING_START_DEFAULTS
    assert inspect.signature(start_tagger_training_run).parameters["continuous"].default == TAGGER_TRAINING_START_DEFAULTS["continuous"]


def test_render_keeps_entire_history_and_actual_source_without_opening_output(tmp_path):
    from PIL import Image
    import subprocess
    import sys
    import core.tagger.foveatag.diagnostic_worker as worker
    run = joint_run(tmp_path)
    root = service.directory(run)
    job = root / "job-synthetic"
    job.mkdir(parents=True)
    source = tmp_path / "source.png"
    Image.new("RGB", (80,64), "white").save(source)
    vocabulary = tmp_path / "vocab.json"
    vocabulary.write_text(json.dumps({"tag_to_idx": {"fixture_alpha": 0, "fixture_beta": 1, "fixture_gamma": 2}}))
    run.config["joint_review"]["vocabulary_path"] = str(vocabulary)
    service.write_json(root / "index.json", {"items": [{"filename": f"old{i}.png", "step": i} for i in range(20)]})
    captured = snapshot(source)
    captured["training_window"] = {"start_step": 1001, "end_step": 2000,
        "samples": 8000, "batches": 1000, "threshold": .5, "tp": 12000, "fp": 3000,
        "fn": 4000, "known_negative": 20000, "precision": .8, "recall": .75,
        "fp_rate": .15, "fp_per_image": .375, "scope": "training_annotations_pre_update"}
    torch.save(captured, job / "training_snapshot.pt")
    service.write_json(job / "request.json", {"training_snapshot": True, "arguments": run.config["joint_review"]})
    completed = subprocess.run([sys.executable, worker.__file__, str(job)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    assert completed.returncode == 0
    state = service.status(run)
    assert len(state["items"]) == 21
    item = state["items"][0]
    assert item["step"] == 2000 and item["sample_id"] == 17
    assert item["prediction_origin"] == "training_forward_before_optimizer_update"
    assert item["tag_labels"] == "vocabulary"
    assert item["training_window"] == captured["training_window"]
    assert item["counts"] == {"TP": 1, "FP": 0, "FN": 1, "TN": 0}
    assert item["local_boxes"][0]["x"] == pytest.approx(24)
    assert (root / item["filename"]).stat().st_size > 0
    assert "source_path" not in json.dumps(item)


def test_named_rows_align_vocabulary_indices_and_known_tp_fp_fn():
    import numpy as np
    from core.tagger.foveatag.diagnostic_worker import named_probability_rows, probability_groups
    probs = np.array([.8, .1, .9, .9])
    groups = probability_groups(probs, np.array([True, True, False, False]), np.array([True, True, True, False]), .5)
    rows = named_probability_rows(probs, groups, {"fixture_beta": 1, "fixture_gamma": 2, "fixture_alpha": 0, "fixture_unknown": 3})
    assert [row[0] for row in rows] == ["TP: fixture_alpha", "FP: fixture_gamma", "FN: fixture_beta"]
    with pytest.raises(ValueError, match="vocabulary"):
        named_probability_rows(probs, groups, {"fixture_alpha": 0})
