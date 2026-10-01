"""CPU REST contract tests against isolated dataset databases."""

import json
import asyncio
import io
import threading
from uuid import uuid4
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.captioner_routes import router
from api.error_handlers import APIError
from api.param_defaults import CAPTIONER_DEFAULTS, CAPTIONER_TRAINING_DEFAULTS
from core.captioner.storage import file_hash
from database import get_datasets_db
from database.models import Dataset, DatasetBase, DatasetCaption, DatasetItem


class ASGIClient:
    def __init__(self, app):
        self.app = app

    def request(self, method, path, **kwargs):
        async def send():
            transport = httpx.ASGITransport(app=self.app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(method, path, **kwargs)
        return asyncio.run(send())

    def get(self, path):
        return self.request("GET", path)

    def post(self, path, **kwargs):
        return self.request("POST", path, **kwargs)


@pytest.fixture
def environment(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'dataset.db'}", connect_args={"check_same_thread": False})
    DatasetBase.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    image = tmp_path / "image.png"
    image.write_bytes(b"preview image")
    image.with_suffix(".txt").write_text("blue hair, girl", encoding="utf-8")
    with factory() as db:
        dataset = Dataset(name="test", path=str(tmp_path), tag_statistics={"girl": {"count": 1}})
        db.add(dataset)
        db.flush()
        item = DatasetItem(dataset_id=dataset.id, item_type="single", base_name="image", image_path="image.png")
        db.add(item)
        db.flush()
        db.add(DatasetCaption(item_id=item.id, caption_type="tags", is_tags_format=True, content="blue hair, girl"))
        db.commit()
        payload = {"dataset_id": dataset.id, "item_id": item.id, "dataset_revision": dataset.revision,
                   "image_hash": file_hash(image), "caption": "A girl with blue hair.",
                   "caption_type": "natural_language"}
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    def session():
        with factory() as db:
            yield db
    app.dependency_overrides[get_datasets_db] = session
    from fastapi.responses import JSONResponse
    @app.exception_handler(APIError)
    async def error(request, exc):
        return JSONResponse(status_code=exc.status_code, content={"message": exc.message, "code": exc.code})
    yield ASGIClient(app), factory, image, payload
    engine.dispose()


def test_save_creates_json_and_preserves_tags(environment):
    client, factory, image, payload = environment
    image.with_suffix(".json").write_text(json.dumps({"tags": "original", "other": {"value": 1}}), encoding="utf-8")
    response = client.post("/api/v1/captioner/save", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["dataset_revision"] == payload["dataset_revision"] + 1
    assert image.with_suffix(".txt").read_text(encoding="utf-8") == "blue hair, girl"
    assert json.loads(image.with_suffix(".json").read_text(encoding="utf-8")) == {
        "tags": "original", "other": {"value": 1}, "compact_caption": payload["caption"],
        "_sushi_compact_caption": {"version": 1, "caption_type": "natural_language"}}
    from utils.caption_detector import scan_json_fields
    rescanned = scan_json_fields(json.loads(image.with_suffix(".json").read_text(encoding="utf-8")), {"original"})
    natural_fields = [entry for entry in rescanned if entry["source_field"] == "compact_caption"]
    assert len(natural_fields) == 1 and natural_fields[0]["caption_type"] == "natural_language"
    assert not natural_fields[0]["is_tags_format"]
    with factory() as db:
        natural = db.query(DatasetCaption).filter(DatasetCaption.source_field == "compact_caption").one()
        assert not natural.is_tags_format and natural.content == payload["caption"]
        assert db.query(Dataset).one().tag_statistics == {"girl": {"count": 1}}
        assert db.query(DatasetCaption).filter(DatasetCaption.caption_type == "tags").one().content == "blue hair, girl"
    assert client.post("/api/v1/captioner/save", json=payload).status_code == 409


def test_save_rejects_changed_image_and_tag_destination(environment):
    client, _, image, payload = environment
    assert client.post("/api/v1/captioner/save", json={**payload, "caption_type": "tags"}).status_code == 400
    assert client.post("/api/v1/captioner/save", json={**payload, "item_id": 999}).status_code == 404
    image.write_bytes(b"changed")
    response = client.post("/api/v1/captioner/save", json=payload)
    assert response.status_code == 409 and response.json()["code"] == "captioner_stale_image"
    assert not image.with_suffix(".json").exists()


def test_save_failure_restores_sidecar_and_revision(environment, monkeypatch):
    client, factory, image, payload = environment
    from sqlalchemy.orm import Session
    def fail(self):
        raise RuntimeError("simulated commit failure")
    monkeypatch.setattr(Session, "commit", fail)
    assert client.post("/api/v1/captioner/save", json=payload).status_code == 500
    assert not image.with_suffix(".json").exists()
    assert image.with_suffix(".txt").read_text(encoding="utf-8") == "blue hair, girl"
    with factory() as db:
        assert db.query(Dataset).one().revision == payload["dataset_revision"]
        assert db.query(DatasetCaption).count() == 1


def test_corrupt_json_is_not_replaced_and_defaults_match(environment):
    client, factory, image, payload = environment
    image.with_suffix(".json").write_text("[broken", encoding="utf-8")
    assert client.post("/api/v1/captioner/save", json=payload).status_code == 400
    assert image.with_suffix(".json").read_text(encoding="utf-8") == "[broken"
    assert client.get("/api/v1/schema/captioner-defaults").json() == CAPTIONER_DEFAULTS
    assert client.get("/api/v1/schema/captioner-training-defaults").json() == CAPTIONER_TRAINING_DEFAULTS
    with factory() as db:
        assert db.query(Dataset).one().revision == payload["dataset_revision"]


def test_save_rejects_existing_natural_column_without_source(environment):
    client, factory, image, payload = environment
    with factory() as db:
        db.add(DatasetCaption(item_id=payload["item_id"], caption_type="natural_language",
                             is_tags_format=False, content="Original natural caption", source_field=None))
        db.commit()
    response = client.post("/api/v1/captioner/save", json=payload)
    assert response.status_code == 400 and response.json()["code"] == "captioner_field_conflict"
    assert not image.with_suffix(".json").exists()


def test_save_rejects_json_field_collision_not_yet_indexed(environment):
    client, _, image, payload = environment
    previous = json.dumps({"natural_language": "Original natural caption"})
    image.with_suffix(".json").write_text(previous, encoding="utf-8")
    assert client.post("/api/v1/captioner/save", json=payload).status_code == 400
    assert image.with_suffix(".json").read_text(encoding="utf-8") == previous


def test_openapi_has_unique_keys_and_save_contract_matches():
    import yaml
    from yaml.nodes import MappingNode, SequenceNode
    path = Path(__file__).resolve().parents[2] / "openapi.yaml"
    tree = yaml.compose(path.read_text(encoding="utf-8"))
    visited = set()
    def verify(node):
        if id(node) in visited:
            return
        visited.add(id(node))
        if isinstance(node, MappingNode):
            keys = [key.value for key, _ in node.value]
            assert len(keys) == len(set(keys)), f"Duplicate YAML keys at line {node.start_mark.line + 1}"
            for _, value in node.value:
                verify(value)
        elif isinstance(node, SequenceNode):
            for value in node.value:
                verify(value)
    verify(tree)
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    schemas = spec["components"]["schemas"]
    from api.captioner_routes import CaptionerSaveRequest
    runtime = CaptionerSaveRequest.model_json_schema()
    assert set(schemas["CaptionerSaveRequest"]["required"]) == set(runtime["required"])
    assert set(schemas["CaptionerSaveRequest"]["properties"]) == set(runtime["properties"])
    assert schemas["CaptionerSaveRequest"]["properties"]["caption_type"]["default"] == CAPTIONER_DEFAULTS["save_caption_type"]
    for route in router.routes:
        assert route.path in spec["paths"]


@pytest.fixture
def inference_bundle(tmp_path, monkeypatch):
    import api.captioner_routes as routes
    from captioner_training_test import manager
    from core.tagger.siglip2_inference_manager import SigLIP2InferenceManager
    from core.captioner.artifact import save_bundle, tagger_identity
    from core.captioner.concepts import ConceptSelector
    from core.captioner.model import CaptionConfig, CompactCaptioner
    from core.captioner.tokenizer import CaptionTokenizer
    from core.captioner.inference import CaptionInferenceManager
    from PIL import Image
    tagger = SigLIP2InferenceManager()
    for key, value in vars(manager()).items():
        setattr(tagger, key, value)
    tagger.device = "cpu"
    identity = tagger_identity(tagger, max_num_patches=4)
    tokenizer = CaptionTokenizer.train(["A girl."], vocab_size=280, min_frequency=1)
    model = CompactCaptioner(CaptionConfig(3, 3, 2, tokenizer.vocab_size, 12, 3, 1, 24, 2, 32, 0.0,
                                          9, 0, 1, 2))
    selector = ConceptSelector(identity["names"], identity["categories"], max_concepts=2,
                               allowed_categories=["General"])
    root = tmp_path / "bundle"
    save_bundle(root, model=model, tokenizer=tokenizer, selector=selector, identity=identity,
                training_state=None, snapshot_hash="synthetic", aliases={}, language="en")
    inference = CaptionInferenceManager(tagger)
    monkeypatch.setattr(routes, "_caption_manager", inference)
    image = io.BytesIO()
    Image.new("RGB", (8, 8), (30, 60, 90)).save(image, format="PNG")
    return inference, root, image.getvalue()


def test_upload_and_item_preview_then_explicit_save(environment, inference_bundle, monkeypatch):
    client, factory, path, payload = environment
    inference, bundle, content = inference_bundle
    import database
    monkeypatch.setattr(database, "DatasetsSessionLocal", factory)
    response = client.post("/api/v1/captioner/load", json={"path": str(bundle)})
    assert response.status_code == 200 and response.json()["loaded"]
    calls = []
    original = inference.tagger.model.forward_features
    def count(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(inference.tagger.model, "forward_features", count)
    response = client.post("/api/v1/captioner/predict", files={"image": ("image.png", content, "image/png")},
        data={"request_id": str(uuid4()), "tag_source": "merged", "tags": '["blue hair"]', "max_new_tokens": "3"})
    assert response.status_code == 200, response.text
    assert len(calls) == 1 and response.json()["used_tags"][0]["name"] == "blue hair"
    path.write_bytes(content)
    preview = client.post("/api/v1/captioner/predict-item", json={"dataset_id": payload["dataset_id"],
        "item_id": payload["item_id"], "request_id": str(uuid4()), "max_new_tokens": 3})
    assert preview.status_code == 200, preview.text
    assert len(calls) == 2 and preview.json()["image_hash"] == file_hash(path)
    assert not path.with_suffix(".json").exists()
    saved = client.post("/api/v1/captioner/save", json={**payload, "image_hash": preview.json()["image_hash"],
        "dataset_revision": preview.json()["dataset_revision"], "caption": "Reviewed natural caption."})
    assert saved.status_code == 200, saved.text
    assert client.post("/api/v1/captioner/unload").json() == {"loaded": False}
    assert inference.tagger.model is not None


def test_inference_refusals_and_upload_limit(environment, inference_bundle, monkeypatch):
    client, _, _, _ = environment
    inference, bundle, content = inference_bundle
    request = {"request_id": str(uuid4()), "max_new_tokens": "3"}
    files = {"image": ("image.png", content, "image/png")}
    response = client.post("/api/v1/captioner/predict", data=request, files=files)
    assert response.status_code == 400 and response.json()["code"] == "captioner_not_loaded"
    assert client.post("/api/v1/captioner/load", json={"path": str(bundle)}).status_code == 200
    assert client.post("/api/v1/captioner/predict", data={**request, "max_new_tokens": "99"}, files=files).status_code == 400
    assert client.post("/api/v1/captioner/predict", data={**request, "tags": "broken"}, files=files).status_code == 400
    monkeypatch.setitem(CAPTIONER_DEFAULTS, "max_image_bytes", 1)
    assert client.post("/api/v1/captioner/predict", data=request, files=files).status_code == 413
    assert client.post(f"/api/v1/captioner/requests/{uuid4()}/cancel").status_code == 404


@pytest.fixture
def sample_run_environment(environment, inference_bundle, tmp_path, monkeypatch):
    import shutil
    import api.captioner_training_routes as routes
    from database.models import TrainingBase
    client, datasets, image, payload = environment
    inference, bundle, content = inference_bundle
    engine = create_engine(f"sqlite:///{tmp_path / 'sample_runs.db'}", connect_args={"check_same_thread": False})
    TrainingBase.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    service = routes.CaptionRunService(factory, datasets, tmp_path)
    monkeypatch.setattr(routes, "service", service)
    monkeypatch.setattr(routes, "get_siglip2_inference_manager", lambda: inference.tagger)
    run = service.create(routes.CaptionerRunCreateRequest(
        tagger={"checkpoint_path": str(bundle / "model.safetensors"), "vocab_path": str(bundle / "tokenizer.json")},
        dataset_configs=[{"dataset_id": payload["dataset_id"], "input_tag_caption_types": ["tags"],
                          "target_caption_types": ["natural"], "filters": {}}], config={}))
    checkpoint = Path(run["output_dir"]) / "checkpoint_sample"
    shutil.copytree(bundle, checkpoint)
    from core.captioner.storage import write_json
    write_json(Path(run["output_dir"]) / "latest.json", {"checkpoint": checkpoint.name})
    image.write_bytes(content)
    client.app.include_router(routes.router, prefix="/api/v1")
    request = {"dataset_id": payload["dataset_id"], "item_id": payload["item_id"],
               "request_id": str(uuid4()), "max_new_tokens": 3}
    yield client, service, run, request, inference, bundle, image
    engine.dispose()


def test_sample_isolated_model_and_durable_preview(sample_run_environment, monkeypatch):
    import api.captioner_training_routes as routes
    client, service, run, request, inference, bundle, image = sample_run_environment
    inference.load(bundle)
    preview_model = inference.model
    calls, instances = [], []
    original = inference.tagger.model.forward_features
    monkeypatch.setattr(inference.tagger.model, "forward_features", lambda *args: (calls.append(1), original(*args))[1])
    from core.captioner.inference import CaptionInferenceManager
    def create(tagger):
        manager = CaptionInferenceManager(tagger)
        instances.append(manager)
        return manager
    monkeypatch.setattr(routes, "CaptionInferenceManager", create)
    response = client.post(f"/api/v1/captioner-training/runs/{run['run_id']}/sample", json=request)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["run_id"] == run["run_id"] and result["checkpoint"] == "checkpoint_sample"
    assert result["image_hash"] == file_hash(image) and len(calls) == 1
    with service.datasets() as db:
        assert result["dataset_revision"] == db.get(Dataset, request["dataset_id"]).revision
    assert inference.model is preview_model and inference.status()["path"] == str(bundle)
    assert instances[0].model is None and inference.tagger.model is not None
    assert not image.with_suffix(".json").exists()
    events = service.events(run["run_id"], 0, 500)["events"]
    assert events[-1]["payload"]["sample"] == result
    invalid = client.post(f"/api/v1/captioner-training/runs/{run['run_id']}/sample",
                          json={**request, "request_id": str(uuid4()), "checkpoint": "../bundle"})
    assert invalid.status_code == 400


def test_sample_cancel_cleans_temporary_decoder(sample_run_environment, monkeypatch):
    import api.captioner_training_routes as routes
    import api.captioner_routes as inference_routes
    from core.captioner.inference import CaptionInferenceManager
    from core.model_state_coordinator import model_state_coordinator
    client, service, run, request, inference, _, _ = sample_run_environment
    entered, exited = threading.Event(), threading.Event()
    instances = []
    def create(tagger):
        manager = CaptionInferenceManager(tagger)
        instances.append(manager)
        def predict(*args, cancelled, **kwargs):
            entered.set()
            try:
                for _ in range(2000):
                    if cancelled():
                        raise InterruptedError("Caption generation cancelled")
                    threading.Event().wait(0.001)
                raise AssertionError("Sample was not cancelled")
            finally:
                exited.set()
        manager.predict = predict
        return manager
    monkeypatch.setattr(routes, "CaptionInferenceManager", create)
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=client.app, raise_app_exceptions=False), base_url="http://test") as http:
            task = asyncio.create_task(http.post(f"/api/v1/captioner-training/runs/{run['run_id']}/sample", json=request))
            assert await asyncio.to_thread(entered.wait, 3)
            cancel = await http.post(f"/api/v1/captioner/requests/{request['request_id']}/cancel")
            assert cancel.status_code == 200
            response = await task
            assert response.status_code == 409 and response.json()["code"] == "captioner_cancelled"
    asyncio.run(exercise())
    assert exited.is_set() and instances[0].model is None
    assert request["request_id"] not in inference_routes._requests
    assert model_state_coordinator.snapshot()["active_generations"] == 0
    assert not any("sample" in row["payload"] for row in service.events(run["run_id"], 0, 500)["events"])


def test_sample_corrupt_bundle_reports_artifact_error(sample_run_environment):
    client, service, run, request, inference, _, _ = sample_run_environment
    path = Path(run["output_dir"]) / "checkpoint_sample" / "config.json"
    path.write_text("{}", encoding="utf-8")
    response = client.post(f"/api/v1/captioner-training/runs/{run['run_id']}/sample", json=request)
    assert response.status_code == 400 and response.json()["code"] == "captioner_invalid_artifact"
    assert not inference.status()["loaded"] and inference.tagger.model is not None
    assert not any("sample" in row["payload"] for row in service.events(run["run_id"], 0, 500)["events"])


def test_bundle_load_does_not_consume_training_rng(inference_bundle):
    import torch
    from core.captioner.artifact import load_bundle
    _, bundle, _ = inference_bundle
    before = torch.get_rng_state().clone()
    model, _, _, _, _ = load_bundle(bundle)
    assert torch.equal(before, torch.get_rng_state())
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_explicit_cancel_waits_for_worker_and_releases_lifecycle(environment, inference_bundle, monkeypatch):
    client, _, _, _ = environment
    inference, bundle, content = inference_bundle
    inference.load(bundle)
    started, exited = threading.Event(), threading.Event()
    def predict(*args, cancelled, **kwargs):
        started.set()
        try:
            for _ in range(1000):
                if cancelled():
                    raise InterruptedError("Caption generation cancelled")
                threading.Event().wait(0.001)
            raise AssertionError("Cancellation did not reach worker")
        finally:
            exited.set()
    monkeypatch.setattr(inference, "predict", predict)
    import api.captioner_routes as routes
    from core.model_state_coordinator import model_state_coordinator
    async def exercise():
        transport = httpx.ASGITransport(app=client.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            request_id = str(uuid4())
            task = asyncio.create_task(http.post("/api/v1/captioner/predict",
                data={"request_id": request_id}, files={"image": ("image.png", content)}))
            assert await asyncio.to_thread(started.wait, 2)
            assert model_state_coordinator.snapshot()["active_generations"] == 1
            cancel = await http.post(f"/api/v1/captioner/requests/{request_id}/cancel")
            assert cancel.status_code == 200
            result = await task
            assert result.status_code == 409 and result.json()["code"] == "captioner_cancelled"
            assert exited.is_set() and request_id not in routes._requests
            assert model_state_coordinator.snapshot()["active_generations"] == 0
    asyncio.run(exercise())


def test_strict_gpu_admission_requires_pause_ack_without_touching_cuda(monkeypatch):
    import core.gpu_coordinator as module
    from types import SimpleNamespace
    monkeypatch.setattr(module, "_generation_device_index", lambda: 0)
    coordinator = module.GPUCoordinator()
    handle = SimpleNamespace(device_index=0, pause_event=threading.Event(), resumed_event=threading.Event())
    handle.pause_event.set()
    coordinator._handles = [handle]
    monkeypatch.setattr(coordinator, "_begin_pause_cycle", lambda *args: [])
    async def exercise():
        with pytest.raises(RuntimeError, match="not acknowledged"):
            async with coordinator.generation_slot(3, timeout=0, require_pause_ack=True):
                pytest.fail("Unacknowledged trainer was admitted")
        assert coordinator._active_generations == 0
        assert not handle.pause_event.is_set()
    asyncio.run(exercise())


def test_repeated_task_cancellation_keeps_slot_until_worker_exits(inference_bundle, monkeypatch):
    inference, bundle, content = inference_bundle
    inference.load(bundle)
    entered, release, cancellation = threading.Event(), threading.Event(), threading.Event()
    def predict(*args, cancelled, **kwargs):
        entered.set()
        while not cancelled():
            threading.Event().wait(0.001)
        cancellation.set()
        assert release.wait(2)
        raise InterruptedError("cancelled")
    monkeypatch.setattr(inference, "predict", predict)
    import api.captioner_routes as routes
    from core.model_state_coordinator import model_state_coordinator
    from types import SimpleNamespace
    async def disconnected():
        return False
    async def exercise():
        task = asyncio.create_task(routes._predict_caption(SimpleNamespace(is_disconnected=disconnected), uuid4(),
            content, tag_source="predicted", tags=[], max_new_tokens=3, device="cpu"))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            assert await asyncio.to_thread(cancellation.wait, 2)
            task.cancel()
            await asyncio.sleep(0)
            assert model_state_coordinator.snapshot()["active_generations"] == 1
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert model_state_coordinator.snapshot()["active_generations"] == 0
    asyncio.run(exercise())


def test_cancel_during_gpu_pause_wait_drains_pause_worker(monkeypatch):
    import core.gpu_coordinator as module
    from types import SimpleNamespace
    coordinator = module.GPUCoordinator()
    started, release = threading.Event(), threading.Event()
    handle = SimpleNamespace(device_index=0, pause_event=threading.Event(), resumed_event=threading.Event())
    coordinator._handles = [handle]
    def begin(*args):
        handle.pause_event.set()
        started.set()
        assert release.wait(2)
        handle.resumed_event.set()
        return [handle]
    resumed = []
    def grace():
        resumed.extend(coordinator._currently_paused)
        handle.pause_event.clear()
        coordinator._currently_paused.clear()
    monkeypatch.setattr(coordinator, "_begin_pause_cycle", begin)
    monkeypatch.setattr(coordinator, "_start_grace_timer", grace)
    async def enter():
        async with coordinator.generation_slot(3, require_pause_ack=True):
            pytest.fail("Cancelled admission reached inference")
    async def exercise():
        task = asyncio.create_task(enter())
        try:
            assert await asyncio.to_thread(started.wait, 2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert coordinator._active_generations == 1
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert coordinator._active_generations == 0 and resumed == [handle]
        assert not handle.pause_event.is_set()
    asyncio.run(exercise())
