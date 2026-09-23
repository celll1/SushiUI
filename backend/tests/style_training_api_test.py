"""/style-training/* lifecycle, defaults parity and storage, with the trainer stubbed (no GPU, temp SQLite)."""

from dataclasses import fields
from pathlib import Path
import asyncio
import re
import sys
import threading
import time

import httpx
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import style_training_fixture  # noqa: F401  (CUDA stubs before the trainer stack is imported)

from fastapi import FastAPI
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

import api.style_training_routes as style_routes
from api.error_handlers import register_error_handlers
from api.param_defaults import STYLE_TRAINING_DEFAULTS
from api.schema_routes import router as schema_router
from core.style_encoder.style_trainer import StyleTrainingConfig
from database import get_training_db
from database.auto_migrate import auto_migrate
from database.models import StyleTrainingMetrics, StyleTrainingRun, TaggerTrainingRun, TrainingBase
from database.training_detail_store import (
    RUN_DB_V2,
    STYLE_RUN_DB_FILENAME,
    initialize_style_detail_database,
    open_style_detail_session,
)

REPO = Path(__file__).resolve().parents[2]


class _Client:
    """TestClient is unusable with the installed httpx 0.28 / starlette 0.35; this drives the same ASGI app."""

    def __init__(self, app):
        self.app = app

    def request(self, method, url, **kwargs):
        async def go():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(method, url, **kwargs)
        return asyncio.run(go())

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def patch(self, url, **kwargs):
        return self.request("PATCH", url, **kwargs)

    def delete(self, url, **kwargs):
        return self.request("DELETE", url, **kwargs)


class _StubTrainer:
    def __init__(self):
        self.stopped = threading.Event()

    def stop(self):
        self.stopped.set()


class _Stub:
    """Stands in for run_style_training: emits the trainer's events, then waits for stop or release."""

    def __init__(self, fail=None):
        self.started = threading.Event()
        self.release = threading.Event()
        self.fail = fail
        self.calls = []

    def __call__(self, run_id, config, output_dir, callback, holder):
        self.calls.append((run_id, config, output_dir))
        trainer = _StubTrainer()
        holder.append(trainer)
        callback(run_id, "phase", {"phase": "dataset", "message": "Loading datasets..."})
        callback(run_id, "dataset_progress", {"step": 1, "total": 2, "message": "Loading dataset 1"})
        callback(run_id, "dataset_stats", {"index_hash": "hash-a", "epoch_length": 4, "ingest": {"records": 8}})
        callback(run_id, "resume", {"mode": "exact", "resumed_from_step": 0, "epoch": 0, "step_in_epoch": 0})
        callback(run_id, "warning", {"message": "kept schedule", "ignored": {"epochs": {"saved": 2, "requested": 3}}})
        callback(run_id, "step", {"step": 1, "epoch": 0, "step_in_epoch": 1, "total_steps": 4, "progress": 0.25,
                                  "index_hash": "hash-a", "loss": 2.0, "supcon_loss": 2.0, "grad_norm": 0.5,
                                  "lr": 1e-4, "images": 8})
        callback(run_id, "epoch", {"epoch": 0, "step": 1, "loss": 2.0})
        callback(run_id, "validation", {"epoch": 0, "global_step": 1, "index_hash": "hash-a",
                                        "held_out_artists": {"map_at_r": 0.4, "recall_at_1": 0.5,
                                                             "recall_at_5": 0.9}})
        callback(run_id, "checkpoint", {"name": "best", "step": 1, "epoch": 0, "path": f"{output_dir}/best.st"})
        callback(run_id, "checkpoint", {"name": "latest", "step": 1, "epoch": 0, "path": f"{output_dir}/latest.st"})
        self.started.set()
        if self.fail is not None:
            raise self.fail
        while not (trainer.stopped.is_set() or self.release.is_set()):
            time.sleep(0.01)
        best = {"map_at_r": 0.4, "global_step": 1, "epoch": 0, "index_hash": "hash-a"}
        if trainer.stopped.is_set():
            return {"stopped": True, "global_step": 1, "total_steps": 4, "best": best}
        return {"global_step": 4, "total_steps": 4, "best": best, "evaluation": None}


@pytest.fixture
def api(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{(tmp_path / 'training.db').as_posix()}",
                           connect_args={"check_same_thread": False})
    TrainingBase.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False)

    def training_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    register_error_handlers(app)
    app.include_router(style_routes.router, prefix="/api/v1")
    app.include_router(schema_router, prefix="/api/v1")
    app.dependency_overrides[get_training_db] = training_db
    monkeypatch.setattr(style_routes, "_session_factory", lambda: factory)
    stub = _Stub()
    monkeypatch.setattr(style_routes, "_trainer_entry", lambda: stub)
    yield _Client(app), factory, stub, tmp_path
    for entry in list(style_routes._active.values()):
        entry.stop_requested = True
        entry.forward_stop()
    _wait_idle()


def _wait_idle(timeout=10.0):
    deadline = time.monotonic() + timeout
    while style_routes._active and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not style_routes._active
    style_routes._detail_writer.get().submit(lambda: None).result(timeout=10)


def _body(tmp_path, name="run", **overrides):
    return {"run_name": name, "output_dir": str(tmp_path / name), "dataset_configs": [{"dataset_id": 1}],
            "vision_encoder_path": "google/siglip2-so400m-patch16-naflex", **overrides}


def _create(client, tmp_path, **overrides):
    response = client.post("/api/v1/style-training/runs", json=_body(tmp_path, **overrides))
    assert response.status_code == 200, response.text
    return response.json()


def _error(response, status):
    assert response.status_code == status, response.text
    body = response.json()
    assert {"error", "detail", "status_code", "timestamp", "path"} <= body.keys() and body["status_code"] == status
    return body


# ------------------------------------------------------------------------- parity


def test_defaults_endpoint_returns_exactly_the_defaults_dict(api):
    client, *_ = api
    response = client.get("/api/v1/schema/style-training-defaults")
    assert response.status_code == 200 and response.json() == STYLE_TRAINING_DEFAULTS


def test_request_model_defaults_and_trainer_config_have_the_same_keys():
    trainer_keys = {f.name for f in fields(StyleTrainingConfig)} - {"dataset_configs"}
    create = style_routes.StyleTrainingRunCreateRequest.model_fields
    route_only = set(style_routes.RUN_FIELDS) | {"dataset_configs"}
    assert set(create) - route_only == set(STYLE_TRAINING_DEFAULTS) == trainer_keys
    assert all(create[k].default == STYLE_TRAINING_DEFAULTS[k] for k in STYLE_TRAINING_DEFAULTS)
    assert set(style_routes.StyleTrainingRunUpdateRequest.model_fields) == set(create) - {"output_dir"}


def test_openapi_and_frontend_declare_the_same_config_keys_and_defaults():
    spec = yaml.safe_load((REPO / "openapi.yaml").read_text(encoding="utf-8"))
    schemas = spec["components"]["schemas"]
    fields_ = schemas["StyleTrainingConfigFields"]["properties"]
    assert set(fields_) == set(STYLE_TRAINING_DEFAULTS) and not any("default" in v for v in fields_.values())
    overlay = schemas["StyleTrainingConfig"]["allOf"][1]["properties"]
    assert {k: v["default"] for k, v in overlay.items()} == STYLE_TRAINING_DEFAULTS
    update = schemas["StyleTrainingRunUpdateRequest"]["allOf"]
    assert update[0] == {"$ref": "#/components/schemas/StyleTrainingConfigFields"}
    assert not any("default" in v for v in update[1]["properties"].values())
    source = (REPO / "frontend" / "src" / "utils" / "api.ts").read_text(encoding="utf-8")
    body = re.search(r"export interface StyleTrainingConfig \{(.*?)\n\}", source, re.S).group(1)
    assert set(re.findall(r"^\s+(\w+):", body, re.M)) == set(STYLE_TRAINING_DEFAULTS)
    refs = set(re.findall(r"#/components/schemas/(\w+)", (REPO / "openapi.yaml").read_text(encoding="utf-8")))
    assert {r for r in refs if r.startswith("Style")} <= set(spec["components"]["schemas"])


def test_the_defaults_are_a_valid_config_and_load_the_repo_release():
    cfg = StyleTrainingConfig.from_dict({**STYLE_TRAINING_DEFAULTS, "dataset_configs": [{"dataset_id": 1}]})
    assert cfg.vision_encoder_path == "" and cfg.encoder_source() == STYLE_TRAINING_DEFAULTS["vision_encoder_repo"]


@pytest.mark.parametrize("overrides,message", [
    ({"style_token_dim": 1150}, "not divisible by style_pool_heads"),
    ({"style_images_per_artist": 5}, "must not exceed style_min_images_per_artist"),
    ({"style_mask_ratio_min": 0.9}, "style_mask_ratio_min must not exceed"),
    ({"style_feature_layers": [21, 21]}, "repeats a layer"),
    ({"attn_implementation": "flash_attention_2"}, "flash_attention_2"),
    ({"dataset_configs": []}, "dataset_configs"),
])
def test_cross_field_constraints_are_refused_at_create_and_patch(api, overrides, message):
    client, _, _, tmp_path = api
    body = _error(client.post("/api/v1/style-training/runs", json=_body(tmp_path, **overrides)), 400)
    assert message in body["detail"]
    run_id = _create(client, tmp_path, name="ok")["run_id"]
    body = _error(client.patch(f"/api/v1/style-training/runs/{run_id}", json=overrides), 400)
    assert message in body["detail"]


def test_create_with_only_the_datasets_uses_every_default(api):
    client, _, _, tmp_path = api
    run = client.post("/api/v1/style-training/runs",
                      json={"dataset_configs": [{"dataset_id": 1}], "output_dir": str(tmp_path / "d")})
    assert run.status_code == 200, run.text
    assert run.json()["config"] == {**STYLE_TRAINING_DEFAULTS, "dataset_configs": [{"dataset_id": 1}]}


# ---------------------------------------------------------------------- lifecycle


def test_create_start_metrics_stop_delete(api):
    client, factory, stub, tmp_path = api
    run = _create(client, tmp_path)
    run_id = run["run_id"]
    assert run["status"] == "pending" and not run["active"]
    assert run["config"] == {**STYLE_TRAINING_DEFAULTS, "dataset_configs": [{"dataset_id": 1}],
                             "vision_encoder_path": "google/siglip2-so400m-patch16-naflex"}
    assert (tmp_path / "run" / STYLE_RUN_DB_FILENAME).is_file()
    assert [r["run_id"] for r in client.get("/api/v1/style-training/runs").json()] == [run_id]

    started = client.post(f"/api/v1/style-training/runs/{run_id}/start")
    assert started.status_code == 200 and started.json()["message"] == "started"
    assert stub.started.wait(10)
    assert stub.calls[0][1] == run["config"] and stub.calls[0][2] == str(tmp_path / "run")

    _error(client.post(f"/api/v1/style-training/runs/{run_id}/start"), 409)
    _error(client.delete(f"/api/v1/style-training/runs/{run_id}"), 409)
    _error(client.patch(f"/api/v1/style-training/runs/{run_id}", json={"epochs": 3}), 409)

    live = client.get(f"/api/v1/style-training/runs/{run_id}").json()
    assert live["status"] == "running" and live["active"]
    assert (live["current_step"], live["total_steps"], live["progress"]) == (1, 4, 0.25)
    assert (live["index_hash"], live["best_map_at_r"], live["best_step"]) == ("hash-a", 0.4, 1)
    assert live["last_resume_mode"] == "exact" and live["latest_checkpoint_path"].endswith("latest.st")

    stop = client.post(f"/api/v1/style-training/runs/{run_id}/stop")
    assert stop.status_code == 200 and stop.json()["message"] == "stop_requested"
    assert stop.json()["run"]["status"] in ("stopping", "stopped")  # the stub may finish before the reply
    _wait_idle()

    detail = client.get(f"/api/v1/style-training/runs/{run_id}").json()
    assert detail["status"] == "stopped" and not detail["active"] and detail["detail_error"] is None
    assert detail["dataset_stats"]["index_hash"] == "hash-a"
    assert [w["message"] for w in detail["warnings"]] == ["kept schedule"]
    assert detail["warnings"][0]["ignored"] == {"epochs": {"saved": 2, "requested": 3}}

    metrics = client.get(f"/api/v1/style-training/runs/{run_id}/metrics").json()
    (row,) = metrics  # step, epoch and validation events of step 1 merge into one row
    assert row["index_hash"] == "hash-a" and row["resume_seq"] == 0
    assert (row["loss"], row["epoch_loss"], row["val_map_at_r"], row["val_recall_at_5"]) == (2.0, 2.0, 0.4, 0.9)
    assert row["evaluation"]["held_out_artists"]["map_at_r"] == 0.4
    with factory() as central:
        assert central.query(StyleTrainingMetrics).count() == 0

    stub.started.clear()
    assert client.post(f"/api/v1/style-training/runs/{run_id}/start").status_code == 200
    assert stub.started.wait(10)
    stub.release.set()
    _wait_idle()
    resumed = client.get(f"/api/v1/style-training/runs/{run_id}").json()
    assert resumed["status"] == "completed" and resumed["progress"] == 1.0 and resumed["current_step"] == 4
    seqs = [m["resume_seq"] for m in client.get(f"/api/v1/style-training/runs/{run_id}/metrics").json()]
    assert seqs == [0, 1]
    _error(client.post(f"/api/v1/style-training/runs/{run_id}/stop"), 409)

    assert client.delete(f"/api/v1/style-training/runs/{run_id}").json() == {"deleted": run_id}
    _error(client.get(f"/api/v1/style-training/runs/{run_id}"), 404)
    assert (tmp_path / "run" / STYLE_RUN_DB_FILENAME).is_file()  # artifacts and history are kept


def test_a_failing_trainer_marks_the_run_failed(api, monkeypatch):
    client, _, _, tmp_path = api
    failing = _Stub(fail=RuntimeError("no optimizer step"))
    monkeypatch.setattr(style_routes, "_trainer_entry", lambda: failing)
    run_id = _create(client, tmp_path)["run_id"]
    assert client.post(f"/api/v1/style-training/runs/{run_id}/start").status_code == 200
    _wait_idle()
    run = client.get(f"/api/v1/style-training/runs/{run_id}").json()
    assert run["status"] == "failed" and run["error_message"] == "no optimizer step"


def test_refusals(api):
    client, factory, _, tmp_path = api
    body = _error(client.post("/api/v1/style-training/runs", json=_body(tmp_path, optimizer="sgd")), 400)
    assert "optimizer" in body["detail"]
    _error(client.post("/api/v1/style-training/runs", json=_body(tmp_path, not_a_key=1)), 422)
    _error(client.post("/api/v1/style-training/runs", json=_body(tmp_path, lora_rank=True)), 422)

    run_id = _create(client, tmp_path)["run_id"]
    _error(client.post("/api/v1/style-training/runs", json=_body(tmp_path)), 409)  # same output_dir
    _error(client.get("/api/v1/style-training/runs/missing"), 404)
    _error(client.post("/api/v1/style-training/runs/missing/start"), 404)
    _error(client.post(f"/api/v1/style-training/runs/{run_id}/stop"), 409)
    _error(client.patch(f"/api/v1/style-training/runs/{run_id}", json={"output_dir": "x"}), 422)
    _error(client.patch(f"/api/v1/style-training/runs/{run_id}", json={"lora_rank": 0}), 400)
    _error(client.get(f"/api/v1/style-training/runs/{run_id}/metrics", params={"max_points": -1}), 422)

    import torch
    patched = client.patch(f"/api/v1/style-training/runs/{run_id}", json={"gpu_index": torch.cuda.device_count()})
    assert patched.status_code == 200 and patched.json()["config"]["gpu_index"] == torch.cuda.device_count()
    assert "Selected GPU is not available" in _error(
        client.post(f"/api/v1/style-training/runs/{run_id}/start"), 400)["error"]
    run = client.get(f"/api/v1/style-training/runs/{run_id}").json()
    assert run["status"] == "pending" and not style_routes._active


def test_patch_changes_only_the_given_fields(api):
    client, _, _, tmp_path = api
    run = _create(client, tmp_path)
    patched = client.patch(f"/api/v1/style-training/runs/{run['run_id']}",
                           json={"run_name": "renamed", "epochs": 3, "gpu_index": None,
                                 "dataset_configs": [{"dataset_id": 2, "caption_types": ["tags"]}]}).json()
    assert patched["run_name"] == "renamed" and patched["total_epochs"] == 3
    assert patched["config"] == {**run["config"], "epochs": 3,
                                 "dataset_configs": [{"dataset_id": 2, "caption_types": ["tags"]}]}
    assert patched["dataset_configs"] == [{"dataset_id": 2, "caption_types": ["tags"]}]


@pytest.mark.parametrize("body", [{"lora_rank": True}, {"epochs": "3"}, {"gradient_checkpointing": 1}])
def test_patch_keeps_strict_types(api, body):
    client, _, _, tmp_path = api
    run = _create(client, tmp_path)
    assert client.patch(f"/api/v1/style-training/runs/{run['run_id']}", json=body).status_code == 422


def test_a_stale_active_status_blocks_neither_stop_nor_start(api):
    client, factory, stub, tmp_path = api
    run_id = _create(client, tmp_path)["run_id"]
    with factory() as db:
        db.query(StyleTrainingRun).filter_by(run_id=run_id).update({"status": "running"})
        db.commit()
    stop = client.post(f"/api/v1/style-training/runs/{run_id}/stop")
    assert stop.json()["message"] == "marked_stopped" and stop.json()["run"]["status"] == "stopped"
    with factory() as db:
        db.query(StyleTrainingRun).filter_by(run_id=run_id).update({"status": "starting"})
        db.commit()
    assert client.post(f"/api/v1/style-training/runs/{run_id}/start").status_code == 200
    assert stub.started.wait(10)


def test_concurrent_starts_launch_one_trainer(api):
    client, _, stub, tmp_path = api
    run_id = _create(client, tmp_path)["run_id"]

    async def burst():
        transport = httpx.ASGITransport(app=client.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            return await asyncio.gather(*(c.post(f"/api/v1/style-training/runs/{run_id}/start") for _ in range(6)))

    codes = sorted(r.status_code for r in asyncio.run(burst()))
    assert codes == [200] + [409] * 5
    assert stub.started.wait(10) and len(stub.calls) == 1


def test_default_output_dir_is_under_the_root(api, monkeypatch):
    client, _, _, tmp_path = api
    monkeypatch.setattr(style_routes.settings, "root_dir", str(tmp_path / "root"))
    body = _body(tmp_path)
    del body["output_dir"]
    run = client.post("/api/v1/style-training/runs", json=body).json()
    assert Path(run["output_dir"]) == tmp_path / "root" / "style_models" / run["run_id"]
    assert run["run_name"] == "run"


def test_metrics_decimation_keeps_every_evaluation_row(api):
    client, factory, _, tmp_path = api
    run_id = _create(client, tmp_path)["run_id"]
    with factory() as central:
        run = central.query(StyleTrainingRun).filter_by(run_id=run_id).one()
        local = open_style_detail_session(run)
    for step in range(1, 301):
        local.add(StyleTrainingMetrics(run_id=run_id, resume_seq=0, step=step, loss=1.0, index_hash="h",
                                       evaluation={"s": step} if step % 100 == 0 else None))
    local.commit()
    local.close()
    rows = client.get(f"/api/v1/style-training/runs/{run_id}/metrics", params={"max_points": 60}).json()
    assert [r["step"] for r in rows if r["evaluation"]] == [100, 200, 300]
    assert len(rows) < 80 and rows[-1]["step"] == 300
    assert len(client.get(f"/api/v1/style-training/runs/{run_id}/metrics", params={"max_points": 0}).json()) == 300
    since = client.get(f"/api/v1/style-training/runs/{run_id}/metrics", params={"since_step": 250}).json()
    assert [r["step"] for r in since] == list(range(250, 301))


def test_missing_detail_database_is_reported_not_hidden(api):
    client, _, _, tmp_path = api
    run_id = _create(client, tmp_path)["run_id"]
    (tmp_path / "run" / STYLE_RUN_DB_FILENAME).unlink()
    detail = client.get(f"/api/v1/style-training/runs/{run_id}").json()
    assert "unavailable" in detail["detail_error"] and detail["dataset_stats"] is None
    _error(client.get(f"/api/v1/style-training/runs/{run_id}/metrics"), 404)


# ---------------------------------------------------------------------- migration


def test_catalogue_migration_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'old.db').as_posix()}")
    TaggerTrainingRun.__table__.create(engine)  # a training.db from before style runs
    with engine.begin() as conn:  # and a style table from an older schema, missing two columns
        conn.execute(text("CREATE TABLE style_training_runs (id INTEGER PRIMARY KEY, run_id VARCHAR NOT NULL, "
                          "run_name VARCHAR NOT NULL, training_method VARCHAR NOT NULL, output_dir VARCHAR NOT NULL)"))
    for _ in range(2):
        TrainingBase.metadata.create_all(engine)
    first = auto_migrate(engine, TrainingBase, "old.db")
    assert "style_training_runs.last_resume_mode" in first and "style_training_runs.warnings" in first
    assert auto_migrate(engine, TrainingBase, "old.db") == []
    inspector = inspect(engine)
    for model in (StyleTrainingRun, StyleTrainingMetrics):
        assert {c["name"] for c in inspector.get_columns(model.__tablename__)} == \
            {c.name for c in model.__table__.columns}
    assert {"style_training_runs", "style_training_metrics"} <= set(inspector.get_table_names())
    unique = [i for i in inspector.get_indexes("style_training_metrics") if i["unique"]] + \
        inspector.get_unique_constraints("style_training_metrics")
    assert any(i["column_names"] == ["run_id", "resume_seq", "step"] for i in unique)


def test_detail_database_initialisation_and_open_are_idempotent_and_migrate(tmp_path):
    run = StyleTrainingRun(run_id="style-1", run_name="r", training_method="lora", output_dir=str(tmp_path / "o"),
                           detail_store=RUN_DB_V2, detail_schema_version=2, detail_state="ready",
                           detail_db_name=STYLE_RUN_DB_FILENAME)
    initialize_style_detail_database(run)
    initialize_style_detail_database(run)
    db = open_style_detail_session(run)
    assert db.query(StyleTrainingRun).count() == 1
    db.close()

    old = tmp_path / "old_detail"
    old.mkdir()
    legacy = create_engine(f"sqlite:///{(old / STYLE_RUN_DB_FILENAME).as_posix()}")
    with legacy.begin() as conn:
        conn.execute(text("CREATE TABLE style_training_runs (id INTEGER PRIMARY KEY, run_id VARCHAR NOT NULL, "
                          "run_name VARCHAR NOT NULL, training_method VARCHAR NOT NULL, output_dir VARCHAR NOT NULL)"))
        conn.execute(text("INSERT INTO style_training_runs (id, run_id, run_name, training_method, output_dir) "
                          "VALUES (1, 'style-2', 'r', 'lora', 'x')"))
    legacy.dispose()
    older = StyleTrainingRun(run_id="style-2", output_dir=str(old), detail_store=RUN_DB_V2, detail_state="ready",
                             detail_db_name=STYLE_RUN_DB_FILENAME)
    db = open_style_detail_session(older)
    local = db.query(StyleTrainingRun).one()
    local.warnings = [{"message": "m"}]
    db.add(StyleTrainingMetrics(run_id="style-2", resume_seq=0, step=1, index_hash="h"))
    db.commit()
    assert db.query(StyleTrainingMetrics).count() == 1
    db.close()


# ------------------------------------------------- tagger extraction (unchanged behaviour)


def test_tagger_detail_store_messages_are_unchanged(tmp_path):
    from types import SimpleNamespace
    from database.training_detail_store import (
        DetailStoreError, initialize_tagger_detail_database, open_tagger_detail_session,
        open_tagger_history_session, tagger_detail_db_path,
    )
    with pytest.raises(DetailStoreError, match="^Tagger training run has no output directory$"):
        tagger_detail_db_path(SimpleNamespace(output_dir=None))
    with pytest.raises(DetailStoreError, match=r"^Invalid tagger detail database name: '\.\./x'$"):
        tagger_detail_db_path(SimpleNamespace(output_dir="o", detail_db_name="../x"))
    with pytest.raises(DetailStoreError, match="^Tagger run does not use a v2 detail database$"):
        open_tagger_detail_session(SimpleNamespace(detail_store=None, output_dir="o"))
    v2 = dict(detail_store=RUN_DB_V2, detail_schema_version=2, detail_state="ready", vision_encoder_path="m",
              run_name="n")
    with pytest.raises(DetailStoreError, match="^Tagger detail database is unavailable: "):
        open_tagger_detail_session(TaggerTrainingRun(run_id="a", output_dir=str(tmp_path / "none"), **v2))
    initialize_tagger_detail_database(TaggerTrainingRun(run_id="a", output_dir=str(tmp_path), **v2))
    with pytest.raises(DetailStoreError, match="^Tagger detail database identity mismatch at "):
        open_tagger_detail_session(TaggerTrainingRun(run_id="b", output_dir=str(tmp_path), **v2))
    with pytest.raises(DetailStoreError, match="^Tagger detail database identity mismatch at "):
        initialize_tagger_detail_database(TaggerTrainingRun(run_id="b", output_dir=str(tmp_path), **v2))

    engine = create_engine("sqlite://")
    TrainingBase.metadata.create_all(engine)
    catalog = sessionmaker(bind=engine)()
    with pytest.raises(DetailStoreError, match="^Tagger training run 'x' is missing from the catalogue$"):
        open_tagger_history_session(catalog, "x")
    catalog.add(TaggerTrainingRun(run_id="x", output_dir=str(tmp_path), **{**v2, "detail_state": "migrating"}))
    catalog.commit()
    with pytest.raises(DetailStoreError, match="^Tagger run 'x' detail store is not ready: 'migrating'$"):
        open_tagger_history_session(catalog, "x")
    catalog.close()


def _legacy_tagger_decimation(rows, max_points):
    """The inline algorithm get_tagger_training_metrics used before it was extracted, verbatim."""
    from collections import defaultdict as _defaultdict
    groups = _defaultdict(list)
    for m in rows:
        groups[m.resume_seq].append(m)

    def _is_validation_row(m) -> bool:
        return getattr(m, "f1", None) is not None or getattr(m, "threshold", None) is not None

    data = []
    if max_points > 0 and len(rows) > max_points:
        per_group_max = max(50, max_points // max(1, len(groups)))
        for seq in sorted(groups):
            g_all = groups[seq]
            g_train = [m for m in g_all if not _is_validation_row(m)]
            g_val = [m for m in g_all if _is_validation_row(m)]
            if len(g_train) > per_group_max:
                step_size = max(1, len(g_train) // per_group_max)
                indices = list(range(0, len(g_train), step_size))
                if indices[-1] != len(g_train) - 1:
                    indices.append(len(g_train) - 1)
                g_train = [g_train[i] for i in indices]
            merged = sorted(g_train + g_val, key=lambda m: m.step)
            data.extend(m.to_dict() for m in merged)
    else:
        for seq in sorted(groups):
            data.extend(m.to_dict() for m in groups[seq])
    return data


def test_extracted_decimation_matches_the_tagger_algorithm():
    import random
    from api.training_run_helpers import decimate_metric_rows
    from database.models import TaggerTrainingMetrics
    rng = random.Random(0)
    for trial in range(40):
        rows = []
        for seq in range(rng.randint(1, 4)):
            for step in range(1, rng.randint(2, 700)):
                val = rng.random() < 0.02
                rows.append(TaggerTrainingMetrics(run_id="r", resume_seq=seq, step=step, loss=float(step),
                                                  f1=0.5 if val else None, threshold=None))
        for max_points in (0, 10, 100, 2000):
            expected = _legacy_tagger_decimation(rows, max_points)
            actual = decimate_metric_rows(rows, max_points, lambda m: m.f1 is not None or m.threshold is not None)
            assert actual == expected, (trial, max_points)


def test_gpu_index_validation_is_the_tagger_check():
    import torch
    from api.error_handlers import ValidationError
    from api.training_run_helpers import validate_gpu_index
    validate_gpu_index(None)
    with pytest.raises(ValidationError) as refused:
        validate_gpu_index(torch.cuda.device_count())
    assert refused.value.message == "Selected GPU is not available"
    assert refused.value.detail == (f"gpu_index={torch.cuda.device_count()!r} but this machine exposes "
                                    f"{torch.cuda.device_count()} CUDA device(s).")


# ------------------------------------------------------------------- audit fixes


def test_null_is_a_422_and_a_bound_a_400_in_both_create_and_patch(api):
    client, _, _, tmp_path = api
    _error(client.post("/api/v1/style-training/runs", json=_body(tmp_path, lora_rank=None)), 422)
    _error(client.post("/api/v1/style-training/runs", json=_body(tmp_path, gpu_index=-1)), 400)
    run_id = _create(client, tmp_path)["run_id"]
    _error(client.patch(f"/api/v1/style-training/runs/{run_id}", json={"lora_rank": None}), 422)
    _error(client.patch(f"/api/v1/style-training/runs/{run_id}", json={"gpu_index": -1}), 400)
    assert client.patch(f"/api/v1/style-training/runs/{run_id}", json={"gpu_index": None}).status_code == 200


def _mutate_after_first_read(monkeypatch, factory, run_id, **config_changes):
    """Commits `config_changes` from another session right after the handler's first, unlocked read."""
    real, calls = style_routes._get_run, []

    def get_run(db, rid):
        run = real(db, rid)
        calls.append(rid)
        if len(calls) == 1:
            with factory() as other:
                row = other.query(StyleTrainingRun).filter_by(run_id=run_id).one()
                row.config = {**row.config, **config_changes}
                other.commit()
        return run

    monkeypatch.setattr(style_routes, "_get_run", get_run)


def test_start_launches_the_config_committed_before_it_took_the_lock(api, monkeypatch):
    client, factory, stub, tmp_path = api
    run_id = _create(client, tmp_path)["run_id"]
    _mutate_after_first_read(monkeypatch, factory, run_id, epochs=7)
    assert client.post(f"/api/v1/style-training/runs/{run_id}/start").status_code == 200
    assert stub.started.wait(10) and stub.calls[0][1]["epochs"] == 7


def test_concurrent_patches_do_not_lose_each_others_fields(api, monkeypatch):
    client, factory, _, tmp_path = api
    run_id = _create(client, tmp_path)["run_id"]
    _mutate_after_first_read(monkeypatch, factory, run_id, num_workers=9)
    patched = client.patch(f"/api/v1/style-training/runs/{run_id}", json={"epochs": 3}).json()
    assert (patched["config"]["num_workers"], patched["config"]["epochs"]) == (9, 3)


def test_stop_never_overwrites_a_final_status_the_thread_already_wrote(api, monkeypatch):
    client, factory, stub, tmp_path = api
    written, gate = threading.Event(), threading.Event()
    real = style_routes._update_run

    def update_run(factory_, run_id, only_if=None, **values):
        real(factory_, run_id, only_if, **values)
        if values.get("status") == "completed":
            written.set()
            gate.wait(10)  # the entry is still registered: stop sees an active run

    monkeypatch.setattr(style_routes, "_update_run", update_run)
    run_id = _create(client, tmp_path)["run_id"]
    stub.release.set()
    assert client.post(f"/api/v1/style-training/runs/{run_id}/start").status_code == 200
    assert written.wait(10)
    stop = client.post(f"/api/v1/style-training/runs/{run_id}/stop")
    assert stop.json()["message"] == "stop_requested" and stop.json()["run"]["status"] == "completed"
    gate.set()
    _wait_idle()
    assert client.get(f"/api/v1/style-training/runs/{run_id}").json()["status"] == "completed"


def test_an_output_dir_of_any_run_kind_is_refused(api):
    from database.models import TrainingRun
    client, factory, _, tmp_path = api
    with factory() as db:
        db.add(TaggerTrainingRun(run_id="tag-1", run_name="t", vision_encoder_path="m",
                                 output_dir=str(tmp_path / "tagger")))
        db.add(TrainingRun(run_name="lora", training_method="lora", base_model_path="m", total_steps=1,
                           output_dir=str(tmp_path / "lora")))
        db.commit()
    body = _error(client.post("/api/v1/style-training/runs", json=_body(tmp_path, name="tagger")), 409)
    assert "tagger training run tag-1" in body["detail"]
    body = _error(client.post("/api/v1/style-training/runs", json=_body(tmp_path, name="lora")), 409)
    assert "training run 1" in body["detail"]


def _drop_config_keys(factory, run_id, *keys):
    with factory() as db:
        row = db.query(StyleTrainingRun).filter_by(run_id=run_id).one()
        row.config = {k: v for k, v in row.config.items() if k not in keys}
        db.commit()


def test_a_stored_config_missing_new_keys_is_filled_with_a_warning(api):
    client, factory, stub, tmp_path = api
    run_id = _create(client, tmp_path, style_temperature=0.2)["run_id"]
    _drop_config_keys(factory, run_id, "style_temperature", "style_eval_chunk_size")
    started = client.post(f"/api/v1/style-training/runs/{run_id}/start").json()
    assert started["filled_defaults"] == ["style_temperature", "style_eval_chunk_size"]
    assert stub.started.wait(10) and stub.calls[0][1]["style_temperature"] == STYLE_TRAINING_DEFAULTS["style_temperature"]
    stub.release.set()
    _wait_idle()
    run = client.get(f"/api/v1/style-training/runs/{run_id}").json()
    assert run["config"]["style_eval_chunk_size"] == STYLE_TRAINING_DEFAULTS["style_eval_chunk_size"]
    assert any(w.get("filled_defaults") == ["style_temperature", "style_eval_chunk_size"] for w in run["warnings"])


def test_a_missing_resume_identity_key_is_refused_once_the_run_has_checkpoints(api):
    client, factory, _, tmp_path = api
    run_id = _create(client, tmp_path, style_embed_dim=256)["run_id"]
    _drop_config_keys(factory, run_id, "style_embed_dim", "style_split_seed")
    (tmp_path / "run" / "latest_state.json").write_text("{}", encoding="utf-8")
    for response in (client.patch(f"/api/v1/style-training/runs/{run_id}", json={"epochs": 3}),
                     client.post(f"/api/v1/style-training/runs/{run_id}/start")):
        body = _error(response, 409)
        assert "style_embed_dim" in body["detail"] and "style_split_seed" in body["detail"]
    patched = client.patch(f"/api/v1/style-training/runs/{run_id}",
                           json={"style_embed_dim": 256, "style_split_seed": 0}).json()
    assert patched["filled_defaults"] == [] and patched["config"]["style_embed_dim"] == 256
    assert not style_routes._active
