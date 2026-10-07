"""Exercise the shipped calibration/sample statements without loading a model."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def _startup_statements():
    source = Path(__file__).resolve().parents[1] / "core/training/base_trainer.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "BaseTrainer")
    train = next(node for node in trainer.body if isinstance(node, ast.FunctionDef)
                 and node.name == "train")
    statements = []
    for node in train.body:
        calls = [part for part in ast.walk(node) if isinstance(part, ast.Call)
                 and isinstance(part.func, ast.Attribute)]
        if any(call.func.attr in {"calibrate_before_training", "_run_step0_sample_if_due"}
               for call in calls):
            statements.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Attribute) and target.attr == "_vae_alignment_progress_callback"
            for target in node.targets
        ):
            statements.append(node)
    assert len(statements) == 3
    return ast.Module(body=statements, type_ignores=[])


@pytest.mark.parametrize("already_calibrated", [False, True])
def test_calibration_precedes_sample_and_cleans_progress_callback(already_calibrated):
    events = []
    callback = object()
    datasets, buckets = object(), object()
    trainer = SimpleNamespace(calibrated=already_calibrated)

    def calibrate(owner, data, manager):
        assert data is datasets and manager is buckets
        assert owner._vae_alignment_progress_callback is callback
        events.append("inherit" if owner.calibrated else "calibrate")
        owner.calibrated = True

    def sample(**kwargs):
        assert trainer.calibrated
        assert not hasattr(trainer, "_vae_alignment_progress_callback")
        events.append("sample")

    trainer.arch = SimpleNamespace(calibrate_before_training=calibrate)
    trainer._run_step0_sample_if_due = sample
    tree = _startup_statements()
    namespace = {node.id: None for node in ast.walk(tree) if isinstance(node, ast.Name)}
    namespace.update(self=trainer, datasets=datasets, bucket_manager=buckets,
                     progress_callback=callback)
    exec(compile(tree, "startup", "exec"), namespace)
    assert events == ["inherit" if already_calibrated else "calibrate", "sample"]


def test_failed_calibration_never_samples_and_cleans_progress_callback():
    def fail(*args):
        raise RuntimeError("calibration failed")

    trainer = SimpleNamespace(arch=SimpleNamespace(calibrate_before_training=fail))
    tree = _startup_statements()
    namespace = {node.id: None for node in ast.walk(tree) if isinstance(node, ast.Name)}
    namespace["self"] = trainer
    with pytest.raises(RuntimeError, match="calibration failed"):
        exec(compile(tree, "startup", "exec"), namespace)
    assert not hasattr(trainer, "_vae_alignment_progress_callback")
