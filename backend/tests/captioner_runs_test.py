"""Caption run persistence and lifecycle checks without loading pretrained weights."""

import threading
from pathlib import Path
import sys

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.captioner_training_routes import (CaptionRunService, CaptionerRunCreateRequest,
                                         CaptionerRunStartRequest, CaptionerTrainingConfig)
from api.error_handlers import APIError
from api.param_defaults import CAPTIONER_TRAINING_DEFAULTS
from core.captioner.storage import write_json
from database.models import CaptionerTrainingRun, TrainingBase


@pytest.fixture
def setup(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'runs.db'}", connect_args={"check_same_thread": False})
    TrainingBase.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    tagger = tmp_path / "tagger.safetensors"
    vocab = tmp_path / "vocab.json"
    tagger.write_bytes(b"test stub")
    vocab.write_text("{}")
    request = CaptionerRunCreateRequest(tagger={"checkpoint_path": str(tagger), "vocab_path": str(vocab)},
        dataset_configs=[{"dataset_id": 1, "input_tag_caption_types": ["tags"],
                          "target_caption_types": ["natural"], "filters": {}}], config={})
    yield factory, request, tmp_path
    engine.dispose()


def test_defaults_strict_types_and_extra_fields():
    assert CaptionerTrainingConfig().model_dump() == CAPTIONER_TRAINING_DEFAULTS
    for values in ({"epochs": True}, {"epochs": "10"}, {"unknown": 1}, {"source_weights": {"predicted": "1"}}):
        with pytest.raises(ValueError):
            CaptionerTrainingConfig(**values)


def test_training_openapi_parity():
    import yaml
    from api.captioner_training_routes import (router, CaptionerDatasetConfig,
        CaptionerTaggerConfig, CaptionerExportRequest, CaptionerSampleRequest)
    spec = yaml.safe_load((Path(__file__).resolve().parents[2] / "openapi.yaml").read_text(encoding="utf-8"))
    schemas = spec["components"]["schemas"]
    for model in (CaptionerTrainingConfig, CaptionerRunCreateRequest, CaptionerRunStartRequest,
                  CaptionerDatasetConfig, CaptionerTaggerConfig, CaptionerExportRequest, CaptionerSampleRequest):
        runtime = model.model_json_schema()
        declared = schemas[model.__name__]
        assert set(runtime["properties"]) == set(declared["properties"])
        assert set(runtime.get("required", [])) == set(declared.get("required", []))
        for key, value in runtime["properties"].items():
            if "default" in value:
                assert value["default"] == declared["properties"][key]["default"]
    for route in router.routes:
        assert route.path in spec["paths"]
        assert set(m.lower() for m in route.methods) <= set(spec["paths"][route.path])


def test_run_defaults_are_detached():
    from api.captioner_training_routes import run_defaults
    from api.param_defaults import CAPTIONER_RUN_DEFAULTS
    returned = run_defaults()
    assert returned == CAPTIONER_RUN_DEFAULTS
    returned["run_name"] = "changed"
    assert run_defaults() == CAPTIONER_RUN_DEFAULTS


def test_persistence_interruption_and_exclusive_directory(setup):
    factory, request, root = setup
    service = CaptionRunService(factory, factory, root)
    run = service.create(request)
    assert run["status"] == "pending"
    with factory() as db:
        db.get(CaptionerTrainingRun, run["run_id"]).status = "running"
        db.commit()
    restarted = CaptionRunService(factory, factory, root)
    assert restarted.get(run["run_id"])["status"] == "interrupted"
    assert restarted.events(run["run_id"], 0, 500)["events"][-1]["payload"]["status"] == "interrupted"
    request.output_dir = run["output_dir"]
    with pytest.raises(APIError):
        service.create(request)
    with pytest.raises(APIError):
        service.start(run["run_id"], CaptionerRunStartRequest(resume=True))


def test_stop_resume_and_single_worker(setup):
    factory, request, root = setup
    entered = threading.Event()
    unloaded = threading.Event()
    instances = []

    class Manager:
        def unload(self):
            unloaded.set()

    class Trainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            instances.append(self)

        def run(self):
            opts = self.kwargs
            entered.set()
            assert opts["stop_event"].wait(5)
            checkpoint = Path(opts["output_dir"]) / "checkpoint_test"
            checkpoint.mkdir(exist_ok=True)
            write_json(checkpoint / "manifest.json", {})
            write_json(Path(opts["output_dir"]) / "latest.json", {"checkpoint": checkpoint.name})
            opts["progress"]({"checkpoint": str(checkpoint), "step": 1})
            return {"stopped": True, "step": 1}

    service = CaptionRunService(factory, factory, root, loader=lambda _: Manager(), trainer=Trainer)
    run = service.create(request)
    started = service.start(run["run_id"], CaptionerRunStartRequest())
    worker = service.workers[run["run_id"]][1]
    assert entered.wait(5)
    with pytest.raises(APIError):
        service.start(run["run_id"], CaptionerRunStartRequest())
    service.stop(run["run_id"])
    worker.join(5)
    assert not worker.is_alive() and unloaded.is_set()
    assert service.get(run["run_id"])["status"] == "stopped"
    assert service.get(run["run_id"])["checkpoints"]
    with pytest.raises(APIError):
        service.start(run["run_id"], CaptionerRunStartRequest())
    for name in ("../checkpoint_test", "C:\\outside", "exports", "missing"):
        with pytest.raises(APIError):
            service.start(run["run_id"], CaptionerRunStartRequest(resume=True, checkpoint=name))
    entered.clear()
    resumed = service.start(run["run_id"], CaptionerRunStartRequest(resume=True))
    worker = service.workers[run["run_id"]][1]
    assert entered.wait(5)
    service.stop(run["run_id"])
    worker.join(5)
    assert not worker.is_alive()
    assert resumed["resume_seq"] == started["resume_seq"] + 1
    assert instances[-1].kwargs["resume_path"].endswith("checkpoint_test")
    events = service.events(run["run_id"], 0, 500)["events"]
    assert service.events(run["run_id"], events[-1]["id"], 500)["events"] == []


def test_worker_failure_releases_activity_and_allows_retry(setup):
    from core.model_state_coordinator import model_state_coordinator
    factory, request, root = setup

    def failing_loader(_):
        raise ValueError("incompatible checkpoint")

    service = CaptionRunService(factory, factory, root, loader=failing_loader)
    run = service.create(request)
    with service.lock:
        service.start(run["run_id"], CaptionerRunStartRequest())
        worker = service.workers[run["run_id"]][1]
    worker.join(5)
    assert not worker.is_alive()
    final = service.get(run["run_id"])
    assert final["status"] == "failed" and "incompatible" in final["error"]
    assert not service.workers
    assert not model_state_coordinator.snapshot()["activities"]


def test_real_trainer_through_rest_and_export(setup, monkeypatch):
    import asyncio
    import httpx
    import torch
    from fastapi import FastAPI
    import api.captioner_training_routes as routes
    from captioner_training_test import database, manager
    from core.captioner.artifact import load_bundle

    factory, request, root = setup
    dataset_factory, datasets = database(root)
    request.dataset_configs = [routes.CaptionerDatasetConfig(**value) for value in datasets]
    request.config = CaptionerTrainingConfig(hidden_dim=12, heads=3, layers=1, ffn_dim=24,
        visual_queries=2, max_length=32, max_num_patches=4, max_concepts=2, vocab_size=280,
        tokenizer_min_frequency=1, epochs=1, batch_size=2, near_duplicate_distance=0,
        dropout=0.0, save_every_steps=1, use_tag_aliases=False)
    unloaded = []

    def load(_):
        tagger = manager()
        tagger.unload = lambda: unloaded.append(True)
        return tagger

    service = CaptionRunService(factory, dataset_factory, root, loader=load)
    monkeypatch.setattr(routes, "service", service)
    app = FastAPI()
    app.include_router(routes.router)

    async def call(method, path, **kwargs):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        created = asyncio.run(call("POST", "/captioner-training/runs", json=request.model_dump()))
        assert created.status_code == 200, created.text
        key = created.json()["run_id"]
        started = asyncio.run(call("POST", f"/captioner-training/runs/{key}/start", json={}))
        assert started.status_code == 200, started.text
        with service.lock:
            current = service.workers.get(key)
        if current:
            current[1].join(20)
            assert not current[1].is_alive()
        result = asyncio.run(call("GET", f"/captioner-training/runs/{key}"))
        assert result.json()["status"] == "completed", result.text
        assert unloaded and result.json()["checkpoints"]
        exported = asyncio.run(call("POST", f"/captioner-training/runs/{key}/export", json={}))
        assert exported.status_code == 200, exported.text
        bundle = Path(exported.json()["path"])
        assert not (bundle / "resume.pt").exists()
        load_bundle(bundle)
        events = asyncio.run(call("GET", f"/captioner-training/runs/{key}/events")).json()["events"]
        assert any("loss" in row["payload"] for row in events)
        assert any("export" in row["payload"] for row in events)
    finally:
        torch.set_num_threads(threads)
