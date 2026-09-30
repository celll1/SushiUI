"""CPU REST contract tests against isolated dataset databases."""

import json
import asyncio
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
