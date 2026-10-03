"""Branch progress and catalog import must preserve comparison lineage."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from core.tagger.foveatag.ui_bridge import import_experiment, project_progress


@pytest.fixture
def arguments():
    return {"epochs": 10, "epoch_samples": 240000, "batch_size": 16}


@pytest.mark.parametrize("mode,offset", [("overview_only", 0), ("overview_placebo", 150000), ("source_native", 300000)])
def test_branch_steps_and_progress(arguments, mode, offset):
    [(event, values)] = project_progress("train", {
        "input_mode": mode, "epoch": 2, "global_step": 31000,
        "sample_offset": 16000, "loss": 0.01, "learning_rate": 0.0001,
    }, arguments)
    assert event == "step"
    assert values["step"] == offset + 31000
    assert values["epoch"] == offset // 15000 + 3
    assert values["lr"] == 0.0001
    assert values["progress"] == pytest.approx((offset // 15000 + 2 + 16000 / 240000) / 30)


def test_validation_uses_aggregate_detection_metrics(arguments):
    [(event, values)] = project_progress("validation", {
        "input_mode": "source_native", "epoch": 0, "global_step": 15000,
        "mean_loss": 0.002, "macro_f1": 0.4, "macro_precision": 0.3,
        "macro_recall": 0.6, "threshold": 0.35,
    }, arguments)
    assert event == "epoch"
    assert values == {"step": 315000, "epoch": 21, "loss": 0.002,
                      "f1": 0.4, "precision": 0.3, "recall": 0.6, "threshold": 0.35}


def test_unknown_branch_refused(arguments):
    with pytest.raises(ValueError, match="known branch"):
        project_progress("train", {"input_mode": "other"}, arguments)


def test_checkpoint_does_not_fabricate_step_or_metric(arguments):
    assert project_progress("checkpoint_saved", {"input_mode": "overview_only"}, arguments) == []


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    import core.tagger.foveatag.phase2_gate as gate
    monkeypatch.setattr(gate, "load_passed_phase2_gate", lambda path: {"decision": "pass"})
    monkeypatch.setattr(gate, "require_phase2_encoder", lambda decision, path: None)
    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path
    write("comparison_summary.json", {"kind": "foveatag_phase3_supervised_comparison", "active_mode": "overview_only"})
    encoder = tmp_path / "encoder.safetensors"
    encoder.write_bytes(b"accepted-encoder")
    vocabulary = write("vocabulary.json", {"num_tags": 2})
    gate_path = write("gate.json", {"decision": "pass"})
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("shared_data_contract.json", "shared_training_contract.json", "shared_initialization.safetensors"):
        (tmp_path / name).write_bytes(b"unchanged")
    write("shared_initialization_metadata.json", {"initialization_contract": {
        "vocabulary_sha256": sha(vocabulary), "pretrained_encoder_sha256": sha(encoder),
        "phase2_gate_sha256": sha(gate_path),
    }})
    contract = {"input_mode": "overview_only", "datasets_db": "snapshot.db",
        "train_counts": {"1": 10}, "validation_counts": {"1": 2}, "domain_weights": {"1": 1.0},
        "holdout_manifests": [], "photo_dataset_id": None, "photo_loss_weight": 0.25,
        "epochs": 10, "epoch_samples": 240000, "batch_size": 16, "workers": 8,
        "learning_rate": 1e-4, "weight_decay": 1e-4, "warmup_steps": 5000,
        "seed": 3, "max_source_pixels": 40000000, "views": {"local_views": 4},
    }
    for role, name in (("shared_data_contract", "shared_data_contract.json"), ("shared_training_contract", "shared_training_contract.json"), ("shared_initialization", "shared_initialization.safetensors")):
        contract[f"{role}_sha256"] = sha(tmp_path / name)
    payload = {"kind": "foveatag_supervised_training", "run_contract": contract,
        "optimizer": {"state": {1: {"step": 30000}}}, "scheduler": {},
        "python_rng_state": (1,), "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": [],
        "epoch": 2, "global_step": 30000, "sample_offset": 0,
        "best_validation_f1": 0.2, "best_threshold": 0.35, "history": [],
    }
    checkpoint = tmp_path / "overview_only/latest.pt"
    checkpoint.parent.mkdir()
    torch.save(payload, checkpoint)
    request = dict(comparison_dir=str(tmp_path), phase2_gate=str(gate_path), pretrained_encoder=str(encoder),
        vocabulary_path=str(vocabulary), checkpoint_every_steps=1000, log_every=100)
    return request, payload, checkpoint


def test_import_recovers_exact_checkpoint_contract(experiment):
    request, payload, checkpoint = experiment
    result = import_experiment(request)
    assert result["snapshot"]["step"] == 30000
    assert result["snapshot"]["epoch"] == 2
    assert result["arguments"]["epochs"] == payload["run_contract"]["epochs"]
    assert result["arguments"]["train_counts"] == payload["run_contract"]["train_counts"]
    assert result["arguments"]["learning_rate"] == 1e-4
    assert result["snapshot"]["checkpoint"] == str(checkpoint)


@pytest.mark.parametrize("name", ["optimizer", "scheduler", "python_rng_state", "torch_rng_state", "cuda_rng_state"])
def test_missing_resume_state_rejected(experiment, name):
    request, payload, checkpoint = experiment
    payload[name] = None
    if name == "optimizer":
        payload[name] = {}
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="missing"):
        import_experiment(request)


def test_changed_common_artifact_rejected(experiment):
    request, _, _ = experiment
    (Path(request["comparison_dir"]) / "shared_training_contract.json").write_text("changed")
    with pytest.raises(ValueError, match="shared_training_contract changed"):
        import_experiment(request)


def test_wrong_vocabulary_rejected(experiment, tmp_path):
    request, _, _ = experiment
    path = tmp_path / "other.json"
    path.write_text('{"num_tags": 3}')
    request["vocabulary_path"] = str(path)
    with pytest.raises(ValueError, match="vocabulary differs"):
        import_experiment(request)
