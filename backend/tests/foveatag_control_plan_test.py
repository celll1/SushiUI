"""CPU tests for the executable Phase-1 residual diagnostic plan."""

import hashlib
import json
from pathlib import Path

import pytest

from core.tagger.foveatag.control_plan import (
    execute_control_evaluation_plan,
    execute_control_training_plan,
    load_control_plan,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(json.dumps(content), encoding="utf-8")
    return path


def _plan(tmp_path):
    repo = tmp_path / "repo"
    checkpoint = _write(tmp_path / "checkpoint.safetensors", b"checkpoint")
    vocabulary = _write(tmp_path / "vocabulary.json", {})
    base = _write(tmp_path / "base.safetensors", b"base")
    database = _write(tmp_path / "datasets.db", b"database")
    manifest = _write(repo / "docs/manifest.json", {})
    baseline = _write(repo / "docs/baseline.json", {})
    plan_path = tmp_path / "plan.json"
    raw = {
        "format_version": 1,
        "kind": "foveatag_phase1_residual_diagnostic_plan",
        "model": {
            "checkpoint": {"path": str(checkpoint), "bytes": checkpoint.stat().st_size, "sha256": _sha(checkpoint)},
            "vocabulary": {"path": str(vocabulary), "bytes": vocabulary.stat().st_size, "sha256": _sha(vocabulary)},
            "base_vision_encoder": {"path": str(base), "bytes": base.stat().st_size, "sha256": _sha(base)},
        },
        "data": {
            "datasets_db": {"path": str(database), "bytes": database.stat().st_size, "mtime_ns": database.stat().st_mtime_ns},
            "train_counts": {"29": 20, "16": 4},
            "validation_counts": {"29": 2, "16": 1},
            "domain_weights": {"29": 0.85, "16": 0.15},
            "photo_dataset_id": 16,
            "photo_loss_weight": 0.25,
            "holdout_manifests": [{"repo_path": "docs/manifest.json", "bytes": manifest.stat().st_size, "sha256": _sha(manifest)}],
        },
        "training": {
            "epochs": 1, "epoch_samples": 24, "batch_size": 4,
            "optimizer_steps": 6, "local_views": 4, "learning_rate": 1e-4,
            "weight_decay": 1e-4, "warmup_steps": 2, "seed": 7,
            "max_source_pixels": 40000000,
            "loss": {"kind": "asymmetric_loss", "gamma_negative": 4.0, "gamma_positive": 1.0, "negative_clip": 0.05, "epsilon": 1e-6, "detach_focal_weight": False, "reduction": "mean"},
        },
        "evaluation": {
            "groups": ["fine_detail", "coarse"],
            "benchmarks": [{"name": "ground", "manifest_repo_path": "docs/manifest.json", "fixed_baseline_report_repo_path": "docs/baseline.json", "fixed_baseline_report_sha256": _sha(baseline)}],
        },
        "outputs": {"training_dir": str(tmp_path / "training"), "evaluation_dir": str(tmp_path / "evaluation")},
    }
    _write(plan_path, raw)
    return plan_path, repo, raw


def test_plan_drives_training_and_evaluation_without_retyping_values(tmp_path):
    path, repo, _raw = _plan(tmp_path)
    plan = load_control_plan(path, repo_root=repo)
    training_calls = []
    evaluation_calls = []

    assert execute_control_training_plan(
        plan, runner=lambda **kwargs: training_calls.append(kwargs) or {"status": "complete"}
    )["status"] == "complete"
    assert execute_control_evaluation_plan(
        plan, runner=lambda **kwargs: evaluation_calls.append(kwargs) or {"status": "complete"}
    )["status"] == "complete"

    train = training_calls[0]
    assert train["train_counts"] == {29: 20, 16: 4}
    assert train["epoch_samples"] == 24
    assert train["max_source_pixels"] == 40000000
    evaluate = evaluation_calls[0]
    assert evaluate["local_views"] == 4
    assert [item.name for item in evaluate["benchmarks"]] == ["ground"]


def test_plan_rejects_changed_external_artifact(tmp_path):
    path, repo, raw = _plan(tmp_path)
    Path(raw["model"]["checkpoint"]["path"]).write_bytes(b"changed")

    with pytest.raises(ValueError, match="checkpoint size changed"):
        load_control_plan(path, repo_root=repo)


def test_plan_rejects_inconsistent_step_budget(tmp_path):
    path, repo, raw = _plan(tmp_path)
    raw["training"]["optimizer_steps"] = 5
    _write(path, raw)

    with pytest.raises(ValueError, match="optimizer_steps"):
        load_control_plan(path, repo_root=repo)


def test_training_plan_requires_explicit_resume_for_existing_output(tmp_path):
    path, repo, _raw = _plan(tmp_path)
    plan = load_control_plan(path, repo_root=repo)
    _write(plan.training_dir / "latest.pt", b"checkpoint")

    with pytest.raises(FileExistsError, match="explicit resume"):
        execute_control_training_plan(
            plan, runner=lambda **_kwargs: pytest.fail("must not run")
        )
