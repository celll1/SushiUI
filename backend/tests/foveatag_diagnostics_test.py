"""Anonymous diagnostics must not mutate comparison state or expose labels."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from api import tagger_diagnostics as service
from api.param_defaults import TAGGER_DIAGNOSTIC_DEFAULTS
from core.tagger.foveatag.diagnostic_worker import probability_groups


def run_at(root):
    root.mkdir(exist_ok=True)
    branch = root / "overview_only"
    branch.mkdir(exist_ok=True)
    (branch / "latest.pt").write_bytes(b"synthetic checkpoint")
    return SimpleNamespace(run_id="synthetic", output_dir=str(root), current_step=20000,
        latest_checkpoint_path=str(branch / "latest.pt"),
        config={"architecture": "foveatag_phase3", "foveatag": {"arguments": {}}})


def test_settings_default_off_and_outside_training_contract(tmp_path):
    run = run_at(tmp_path)
    before = json.dumps(run.config)
    assert service.status(run)["settings"] == TAGGER_DIAGNOSTIC_DEFAULTS
    assert not service.schedule(run)
    service.configure(run, {**TAGGER_DIAGNOSTIC_DEFAULTS, "enabled": True})
    assert json.dumps(run.config) == before


def test_periodic_interval_and_single_worker(tmp_path, monkeypatch):
    run = run_at(tmp_path)
    service.configure(run, {**TAGGER_DIAGNOSTIC_DEFAULTS, "enabled": True})
    service.write_json(service.directory(run) / "index.json", {"items": [{"mode": "overview_only", "catalog_step": 19000}], "error": None})
    assert not service.schedule(run)
    calls = []
    monkeypatch.setattr(service.subprocess, "Popen", lambda *a, **k: calls.append((a, k)) or SimpleNamespace(poll=lambda: None))
    monkeypatch.setattr(service, "_jobs", {})
    service.write_json(service.directory(run) / "index.json", {"items": [{"mode": "overview_only", "catalog_step": 4000}], "error": None})
    assert service.schedule(run)
    assert not service.schedule(run, force=True)
    assert len(calls) == 1


def test_image_route_only_serves_published_contained_names(tmp_path):
    run = run_at(tmp_path)
    root = service.directory(run)
    service.write_json(root / "index.json", {"items": [{"filename": "panel.png"}], "error": None})
    (root / "panel.png").write_bytes(b"not opened")
    assert service.image_path(run, "panel.png") == root / "panel.png"
    with pytest.raises(ValueError):
        service.image_path(run, "../panel.png")
    with pytest.raises(FileNotFoundError):
        service.image_path(run, "unpublished.png")


def test_groups_ignore_unknown_and_retain_fn():
    groups = probability_groups(np.array([.8, .1, .8, .1, .9]),
        np.array([True, True, False, False, True]), np.array([True, True, True, True, False]), .5)
    assert {key: int(value.sum()) for key, value in groups.items()} == {"TP": 1, "FP": 1, "FN": 1, "TN": 1}


@pytest.mark.parametrize("mode,preview,native", [("overview_only", False, False), ("overview_only", True, False), ("overview_placebo", False, False), ("source_native", False, True), ("routed", False, True)])
def test_render_synthetic_general_and_exact_view_geometry(tmp_path, mode, preview, native):
    import sqlite3
    from PIL import Image
    from core.tagger.foveatag.artifact import FoveaArtifactConfig
    from core.tagger.foveatag.encoder import FoveaEncoderConfig
    from core.tagger.foveatag.model import FoveaTagger
    from core.tagger.foveatag.views import ViewConfig
    cfg = FoveaArtifactConfig(num_tags=3, feature_dim=16, pool_heads=2,
        encoder=FoveaEncoderConfig(widths=(8,16,32,48), depths=(1,1,1,1), attention_heads=(2,3)),
        views=ViewConfig(overview_size=32, local_size=32, local_views=2))
    model = FoveaTagger(3, encoder_config=cfg.encoder, feature_dim=16, pool_heads=2)
    if mode == "routed":
        from dataclasses import replace
        from core.tagger.foveatag.artifact import InferenceMode
        cfg = replace(cfg, views=replace(cfg.views, overview_resize_mode="stretch", local_border_mode="shift_inside"),
            inference_modes={"balanced": InferenceMode(32,2)})
    root = tmp_path / "diagnostics"
    job = root / "job-synthetic"
    job.mkdir(parents=True)
    source = tmp_path / "source.png"
    Image.new("RGB", (100,80), "white").save(source)
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps({"tag_to_idx": {"general":0, "anonymous_a":1, "anonymous_b":2},
        "tag_to_category": {"general":"Rating", "anonymous_a":"General", "anonymous_b":"General"}}))
    database = tmp_path / "samples.db"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE dataset_items(id INTEGER,dataset_id INTEGER,image_path TEXT)")
        db.execute("CREATE TABLE dataset_captions(item_id INTEGER,is_tags_format INTEGER,tag_data TEXT,content TEXT)")
        db.execute("INSERT INTO dataset_items VALUES(1,29,?)", (str(source),))
        db.execute("INSERT INTO dataset_captions VALUES(1,1,NULL,'general, anonymous_a')")
    (tmp_path / "shared_initialization_metadata.json").write_text(json.dumps({"initialization_contract":{"model":cfg.to_dict()}}))
    torch.save({"model":model.state_dict(), "run_contract":{"input_mode":mode}, "global_step":1000,
        "epoch":0, "best_threshold":.35}, job / "checkpoint.pt")
    if mode == "routed":
        from core.tagger.foveatag.router import EvidenceRouter
        router = EvidenceRouter(cfg.encoder.widths[2], routing_dim=cfg.routing_dim, routing_slots=cfg.routing_slots)
        torch.save({"kind":"foveatag_joint_training", "model":model.state_dict(), "router":router.state_dict(),
            "run_contract":{"artifact_config":cfg.to_dict()}, "global_step":1000, "epoch":0, "best_threshold":.35}, job / "checkpoint.pt")
    args = {"vocabulary_path":str(vocab_path), "datasets_db":str(database), "train_counts":{"29":1},
        "photo_dataset_id":16, "max_source_pixels":40000000, "seed":20260929}
    (job / "request.json").write_text(json.dumps({"comparison":str(tmp_path), "catalog_step":1000,
        "arguments":args, "focus_preview":preview}))
    import subprocess
    import sys
    import core.tagger.foveatag.diagnostic_worker as worker
    completed = subprocess.run([sys.executable, worker.__file__, str(job)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    assert completed.returncode == 0
    item = json.loads((root / "index.json").read_text())["items"][0]
    assert item["native_focus_used"] is native
    assert item["learned_routing"] is (mode == "routed")
    assert item["focus_preview"] is preview
    assert len(item["local_boxes"]) == (0 if mode == "overview_only" and not preview else 2)
    assert (root / item["filename"]).stat().st_size > 0
    assert "anonymous_a" not in (root / "index.json").read_text()
    if mode == "overview_only" and not preview:
        with sqlite3.connect(database) as db:
            db.execute("INSERT INTO dataset_items VALUES(0,29,?)", (str(source),))
            db.execute("INSERT INTO dataset_captions VALUES(0,1,NULL,'general')")
        job.mkdir()
        torch.save({"model":model.state_dict(), "run_contract":{"input_mode":mode}, "global_step":2000,
            "epoch":0, "best_threshold":.35}, job / "checkpoint.pt")
        (job / "request.json").write_text(json.dumps({"comparison":str(tmp_path), "catalog_step":2000,
            "arguments":args, "focus_preview":preview}))
        completed = subprocess.run([sys.executable, worker.__file__, str(job)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        assert completed.returncode == 0
        assert json.loads((root / "index.json").read_text())["items"][0]["sample_id"] == 1
