"""Persisted batch lifecycle with a real small decoder and isolated databases."""

import asyncio
from pathlib import Path
import threading

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from captioner_api_test import environment, inference_bundle
import api.captioner_batch_routes as routes
from api.error_handlers import APIError
from core.captioner.inference import CaptionInferenceManager
from core.captioner.storage import file_hash
from database.models import CaptionerBatch, CaptionerBatchItem, DatasetItem, TrainingBase


@pytest.fixture
def batch_environment(environment, inference_bundle, tmp_path, monkeypatch):
    client, datasets, image, original = environment
    inference, bundle, content = inference_bundle
    image.write_bytes(content)
    other = tmp_path / "other.png"
    other.write_bytes(content)
    with datasets() as db:
        rows = [DatasetItem(dataset_id=original["dataset_id"], item_type="single", base_name="other", image_path="other.png"),
                DatasetItem(dataset_id=original["dataset_id"], item_type="single", base_name="missing", image_path="missing.png")]
        db.add_all(rows)
        db.commit()
        ids = [original["item_id"], *[row.id for row in rows]]
    engine = create_engine(f"sqlite:///{tmp_path / 'batch.db'}", connect_args={"check_same_thread": False})
    TrainingBase.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    managers, loads = [], []
    def manager():
        instance = CaptionInferenceManager(inference.tagger)
        load = instance.load
        def track(path):
            loads.append(path)
            return load(path)
        instance.load = track
        managers.append(instance)
        return instance
    service = routes.CaptionBatchService(factory, datasets, manager_factory=manager)
    monkeypatch.setattr(routes, "service", service)
    client.app.include_router(routes.router, prefix="/api/v1")
    request = {"bundle_path": str(bundle), "dataset_id": original["dataset_id"], "item_ids": ids, "max_new_tokens": 3}
    yield client, service, request, managers, loads, image
    engine.dispose()


async def run_job(client, service, job_id, retry_failed=False):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=client.app, raise_app_exceptions=False), base_url="http://test") as http:
        started = await http.post(f"/api/v1/captioner/batches/{job_id}/start", json={"retry_failed": retry_failed})
        assert started.status_code == 200, started.text
        record = service.workers.get(job_id)
        if record:
            await asyncio.wait_for(record["task"], 10)


def test_batch_results_persist_and_bundle_loaded_once(batch_environment):
    client, service, request, managers, loads, image = batch_environment
    created = client.post("/api/v1/captioner/batches", json=request)
    assert created.status_code == 200, created.text
    key = created.json()["job_id"]
    assert created.json()["status"] == "pending"
    asyncio.run(run_job(client, service, key))
    status = client.get(f"/api/v1/captioner/batches/{key}").json()
    assert status["status"] == "completed" and status["counts"] == {"completed": 2, "failed": 1}
    rows = client.get(f"/api/v1/captioner/batches/{key}/results").json()["items"]
    assert [row["item_id"] for row in rows] == request["item_ids"]
    assert rows[0]["preview"]["image_hash"] == file_hash(image)
    assert rows[2]["error_code"] == "captioner_item_not_found"
    assert len(loads) == 1 and managers[0].model is None
    assert not image.with_suffix(".json").exists()
    restarted = routes.CaptionBatchService(service.db, service.datasets)
    assert restarted.results(key, 0, 200)["items"] == rows
    assert restarted.results(key, 1, 1)["items"] == rows[1:2]


def test_batch_stop_resume_and_explicit_retry(batch_environment):
    client, service, request, managers, loads, _ = batch_environment
    entered = threading.Event()
    original_factory = service.manager_factory
    def blocking_factory():
        manager = original_factory()
        predict = manager.predict
        calls = []
        def block(*args, cancelled, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                entered.set()
                for _ in range(2000):
                    if cancelled():
                        raise InterruptedError("Caption generation cancelled")
                    threading.Event().wait(0.001)
                raise AssertionError("Batch did not stop")
            return predict(*args, cancelled=cancelled, **kwargs)
        manager.predict = block
        return manager
    service.manager_factory = blocking_factory
    key = service.create(routes.CaptionerBatchCreateRequest(**request))["job_id"]
    async def exercise():
        service.start(key, False)
        task = service.workers[key]["task"]
        assert await asyncio.to_thread(entered.wait, 5)
        with pytest.raises(APIError):
            service.start(key, False)
        assert service.get(key)["status"] == "running"
        service.stop(key)
        await asyncio.wait_for(task, 10)
    asyncio.run(exercise())
    assert service.get(key)["status"] == "stopped"
    assert service.get(key)["counts"] == {"completed": 1, "pending": 2}
    assert managers[0].model is None
    first = service.results(key, 0, 200)["items"][0]
    service.manager_factory = original_factory
    asyncio.run(run_job(client, service, key))
    rows = service.results(key, 0, 200)["items"]
    assert rows[0] == first and rows[1]["attempts"] == 2 and rows[2]["status"] == "failed"
    asyncio.run(run_job(client, service, key, True))
    rows = service.results(key, 0, 200)["items"]
    assert rows[0] == first and rows[2]["attempts"] == 2


def test_batch_restart_and_changed_bundle_refusal(batch_environment):
    _, service, request, _, _, _ = batch_environment
    key = service.create(routes.CaptionerBatchCreateRequest(**request))["job_id"]
    with service.db() as db:
        db.get(CaptionerBatch, key).status = "running"
        db.get(CaptionerBatchItem, (key, request["item_ids"][0])).status = "running"
        db.commit()
    assert service.get(key)["status"] == "interrupted"
    assert service.results(key, 0, 200)["items"][0]["status"] == "pending"
    manifest = Path(request["bundle_path"]) / "manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    async def exercise():
        service.start(key, False)
        await service.workers[key]["task"]
    asyncio.run(exercise())
    state = service.get(key)
    assert state["status"] == "failed" and state["error_code"] == "captioner_batch_bundle_changed"
    assert state["counts"] == {"pending": 3}


def test_batch_schema_defaults_and_invalid_inputs(batch_environment):
    import yaml
    from api.param_defaults import CAPTIONER_BATCH_DEFAULTS
    client, _, request, _, _, _ = batch_environment
    assert client.get("/api/v1/schema/captioner-batch-defaults").json() == CAPTIONER_BATCH_DEFAULTS
    assert client.post("/api/v1/captioner/batches", json={**request, "item_ids": [True]}).status_code == 422
    assert client.post("/api/v1/captioner/batches", json={**request, "item_ids": [1, 1]}).status_code == 400
    assert client.post("/api/v1/captioner/batches", json={**request, "max_new_tokens": 1000}).status_code == 400
    spec = yaml.safe_load((Path(__file__).resolve().parents[2] / "openapi.yaml").read_text(encoding="utf-8"))
    for model in (routes.CaptionerBatchCreateRequest, routes.CaptionerBatchStartRequest, routes.CaptionerBatchSaveRequest):
        runtime, declared = model.model_json_schema(), spec["components"]["schemas"][model.__name__]
        assert set(runtime["properties"]) == set(declared["properties"])
        assert set(runtime.get("required", [])) == set(declared.get("required", []))
        for key, value in runtime["properties"].items():
            if "default" in value:
                assert value["default"] == declared["properties"][key]["default"]
    for route in routes.router.routes:
        assert route.path in spec["paths"]


def test_batch_task_cancellation_drains_worker(batch_environment):
    from core.model_state_coordinator import model_state_coordinator
    _, service, request, managers, _, _ = batch_environment
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    factory = service.manager_factory
    def blocked():
        manager = factory()
        def predict(*args, cancelled, **kwargs):
            entered.set()
            try:
                assert release.wait(5)
                assert cancelled()
                raise InterruptedError("Caption generation cancelled")
            finally:
                exited.set()
        manager.predict = predict
        return manager
    service.manager_factory = blocked
    key = service.create(routes.CaptionerBatchCreateRequest(**request))["job_id"]
    async def exercise():
        service.start(key, False)
        task = service.workers[key]["task"]
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done() and model_state_coordinator.snapshot()["active_generations"] == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(exercise())
    assert exited.is_set() and managers[0].model is None
    assert not service.workers and model_state_coordinator.snapshot()["active_generations"] == 0
    assert service.get(key)["status"] == "interrupted"
    assert service.get(key)["counts"] == {"pending": 3}


def completed_batch(environment):
    client, service, request, _, _, _ = environment
    key = service.create(routes.CaptionerBatchCreateRequest(**request))["job_id"]
    asyncio.run(run_job(client, service, key))
    rows = service.results(key, 0, 200)["items"]
    return client, service, request, key, rows


def save_payload(row, caption="Reviewed natural caption."):
    return {"caption": caption, "dataset_revision": row["preview"]["dataset_revision"],
            "image_hash": row["preview"]["image_hash"]}


def test_batch_saves_advance_siblings_and_keep_receipts(batch_environment):
    import json
    client, service, _, key, rows = completed_batch(batch_environment)
    image = batch_environment[-1]
    tags = image.with_suffix(".txt").read_text(encoding="utf-8")
    first = client.post(f"/api/v1/captioner/batches/{key}/items/{rows[0]['item_id']}/save", json=save_payload(rows[0]))
    assert first.status_code == 200, first.text
    updated = service.results(key, 0, 200)["items"]
    assert updated[0]["saved"]["caption"] == "Reviewed natural caption."
    assert updated[1]["preview"]["dataset_revision"] == first.json()["dataset_revision"]
    second = client.post(f"/api/v1/captioner/batches/{key}/items/{rows[1]['item_id']}/save", json=save_payload(updated[1], "Second reviewed caption."))
    assert second.status_code == 200, second.text
    assert image.with_suffix(".txt").read_text(encoding="utf-8") == tags
    assert json.loads(image.with_suffix(".json").read_text(encoding="utf-8"))["compact_caption"] == "Reviewed natural caption."
    retry = client.post(f"/api/v1/captioner/batches/{key}/items/{rows[0]['item_id']}/save", json=save_payload(rows[0]))
    assert retry.status_code == 200 and retry.json() == first.json()
    changed = client.post(f"/api/v1/captioner/batches/{key}/items/{rows[0]['item_id']}/save", json=save_payload(rows[0], "Different caption."))
    assert changed.status_code == 409
    wrong_image = client.post(f"/api/v1/captioner/batches/{key}/items/{rows[0]['item_id']}/save",
                             json={**save_payload(rows[0]), "image_hash": "0" * 64})
    assert wrong_image.status_code == 409 and wrong_image.json()["code"] == "captioner_stale_image"
    restarted = routes.CaptionBatchService(service.db, service.datasets)
    assert restarted.results(key, 0, 200)["items"][0]["saved"]


def test_batch_save_refuses_external_edits(batch_environment):
    from database.models import Dataset
    client, service, request, key, rows = completed_batch(batch_environment)
    with service.datasets() as db:
        dataset = db.get(Dataset, request["dataset_id"])
        dataset.revision += 1
        db.commit()
    response = client.post(f"/api/v1/captioner/batches/{key}/items/{rows[0]['item_id']}/save", json=save_payload(rows[0]))
    assert response.status_code == 409 and response.json()["code"] == "captioner_stale_revision"
    assert not service.results(key, 0, 200)["items"][0]["pending_save"]
    assert not batch_environment[-1].with_suffix(".json").exists()


def test_batch_save_recovers_failed_receipt_without_second_write(batch_environment, monkeypatch):
    from database.models import Dataset
    client, service, request, key, rows = completed_batch(batch_environment)
    original = service._receipt
    def fail(*args):
        raise RuntimeError("Receipt database unavailable")
    monkeypatch.setattr(service, "_receipt", fail)
    url = f"/api/v1/captioner/batches/{key}/items/{rows[0]['item_id']}/save"
    response = client.post(url, json=save_payload(rows[0]))
    assert response.status_code == 500
    assert service.results(key, 0, 200)["items"][0]["pending_save"]
    with service.datasets() as db:
        saved_revision = db.get(Dataset, request["dataset_id"]).revision
    monkeypatch.setattr(service, "_receipt", original)
    recovered = client.post(url, json=save_payload(rows[0]))
    assert recovered.status_code == 200, recovered.text
    with service.datasets() as db:
        assert db.get(Dataset, request["dataset_id"]).revision == saved_revision
    assert not service.results(key, 0, 200)["items"][0]["pending_save"]


def test_batch_uncertain_save_refuses_changed_sidecar(batch_environment, monkeypatch):
    client, service, _, key, rows = completed_batch(batch_environment)
    original = service._receipt
    monkeypatch.setattr(service, "_receipt", lambda *a: (_ for _ in ()).throw(RuntimeError("Receipt failed")))
    url = f"/api/v1/captioner/batches/{key}/items/{rows[0]['item_id']}/save"
    assert client.post(url, json=save_payload(rows[0])).status_code == 500
    sidecar = batch_environment[-1].with_suffix(".json")
    sidecar.write_text('{"compact_caption":"external edit"}', encoding="utf-8")
    monkeypatch.setattr(service, "_receipt", original)
    response = client.post(url, json=save_payload(rows[0]))
    assert response.status_code == 409 and response.json()["code"] == "captioner_batch_save_uncertain"
    assert "external edit" in sidecar.read_text(encoding="utf-8")
