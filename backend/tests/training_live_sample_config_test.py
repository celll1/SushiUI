import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi import HTTPException

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from api import routes
from core.training import training_sample_rpc


class _Session:
    def __init__(self, run):
        self.run = run

    def query(self, _model):
        return self

    def filter(self, *_args):
        return self

    def first(self):
        return self.run

    def commit(self):
        pass

    def rollback(self):
        pass


def test_live_sample_edit_preserves_training_config_and_tracks_application(tmp_path, monkeypatch):
    original = {
        "config": {"process": [{
            "train": {"batch_size": 1, "lr": 1e-4},
            "sample": {
                "sample_every": 100, "prompts": [{"positive": "old", "negative": ""}],
                "width": 1024, "height": 1024, "sample_steps": 28,
                "guidance_scale": 1.0, "seed": 42, "sampler": "euler",
            },
        }]},
    }
    yaml_text = yaml.safe_dump(original)
    run = SimpleNamespace(
        id=44, run_name="sample-live-test", output_dir=str(tmp_path),
        base_model_path="unused", config_yaml=yaml_text,
    )
    config_path = tmp_path / "sample-live-test_config.yaml"
    config_path.write_text(yaml_text, encoding="utf-8")
    monkeypatch.setattr(routes, "_training_sample_support", lambda _run: ("sdxl", None))
    monkeypatch.setattr(routes.training_process_manager, "processes", {
        44: SimpleNamespace(is_running=True),
    })
    db = _Session(run)
    request = routes.TrainingLiveSampleConfigUpdate(
        expected_revision=0, sample_every=20,
        prompts=[{"positive": "pink dress", "negative": ""}],
        width=1024, height=1024, sample_steps=30,
        guidance_scale=3.0, seed=42,
    )
    status = asyncio.run(routes.update_live_training_sample_config(44, request, db))
    assert status["desired_revision"] == 1
    assert status["pending"] is True
    assert status["config"]["prompts"][0]["positive"] == "pink dress"
    assert training_sample_rpc.read_live_config(tmp_path, 44)["revision"] == 1
    assert training_sample_rpc.read_live_config(tmp_path, 45) is None
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))["config"]["process"][0]
    assert saved["train"] == original["config"]["process"][0]["train"]
    assert saved["sample"]["sampler"] == "euler"
    assert saved["sample"]["sample_every"] == 20

    with pytest.raises(HTTPException) as stale:
        asyncio.run(routes.update_live_training_sample_config(44, request, db))
    assert stale.value.status_code == 409

    training_sample_rpc.write_live_applied(tmp_path, {
        "run_id": 44, "revision": 1, "step": 12,
    })
    applied = asyncio.run(routes.get_live_training_sample_config(44, db))
    assert applied["pending"] is False
    assert applied["applied_step"] == 12


def test_live_sample_edit_rejects_empty_positive_prompt():
    with pytest.raises(ValueError, match="non-empty positive"):
        routes.TrainingLiveSampleConfigUpdate(
            expected_revision=0, sample_every=10,
            prompts=[{"positive": "  ", "negative": ""}],
            width=1024, height=1024, sample_steps=28,
            guidance_scale=1.0, seed=-1,
        )


def test_live_sample_edit_refuses_nonrunning_run(tmp_path, monkeypatch):
    run = SimpleNamespace(
        id=45, run_name="idle", output_dir=str(tmp_path),
        base_model_path="unused", config_yaml="config:\n  process:\n  - sample: {}\n",
    )
    monkeypatch.setattr(routes, "_training_sample_support", lambda _run: ("sdxl", None))
    monkeypatch.setattr(routes.training_process_manager, "processes", {})
    request = routes.TrainingLiveSampleConfigUpdate(
        expected_revision=0, sample_every=10,
        prompts=[{"positive": "test", "negative": ""}],
        width=1024, height=1024, sample_steps=28,
        guidance_scale=1.0, seed=-1,
    )
    with pytest.raises(HTTPException) as stopped:
        asyncio.run(routes.update_live_training_sample_config(45, request, _Session(run)))
    assert stopped.value.status_code == 409
    assert training_sample_rpc.read_live_config(tmp_path, 45) is None


def test_trainer_adopts_live_settings_before_sample_decision():
    source = (BACKEND / "core" / "training" / "base_trainer.py").read_text(encoding="utf-8")
    loop = source[source.index("# Adopt sample-only edits before deciding"):
                  source.index("should_generate_sample = False", source.index("# Adopt sample-only edits before deciding"))]
    for field in ("sample_every", "prompts", "width", "height", "sample_steps",
                  "guidance_scale", "seed"):
        assert f'values["{field}"]' in loop
    assert loop.index("write_live_applied") < loop.index("self._live_sample_revision = next_revision")
