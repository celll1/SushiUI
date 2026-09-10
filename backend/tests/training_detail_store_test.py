"""Compatibility and path-safety tests for training detail-store routing."""

from types import SimpleNamespace

import pytest
from sqlalchemy import text

from database.training_detail_store import (
    CENTRAL_V1,
    RUN_DB_V2,
    DetailStoreError,
    detail_db_path,
    initialize_run_detail_database,
    open_run_detail_session,
    resolve_detail_store,
)
from database.models import TrainingMetrics, TrainingRun


def _run(**overrides):
    values = {
        "detail_store": None,
        "detail_schema_version": None,
        "detail_state": None,
        "detail_db_name": None,
        "output_dir": "output/run",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_null_store_is_legacy_without_touching_output_directory():
    location = resolve_detail_store(_run(output_dir=None))
    assert location.kind == CENTRAL_V1
    assert location.path is None


def test_v2_uses_fixed_name_beneath_output_directory():
    location = resolve_detail_store(_run(
        detail_store=RUN_DB_V2,
        detail_schema_version=2,
        detail_state="ready",
    ))
    assert location.path == detail_db_path(_run(detail_store=RUN_DB_V2))
    assert location.path.name == "training_run.db"
    assert location.schema_version == 2
    assert location.state == "ready"


@pytest.mark.parametrize("name", ("../other.db", "nested/run.db", "", ".", ".."))
def test_v2_rejects_non_local_database_names(name):
    with pytest.raises(DetailStoreError):
        resolve_detail_store(_run(detail_store=RUN_DB_V2, detail_db_name=name))


def test_unknown_store_is_not_silently_treated_as_legacy():
    with pytest.raises(DetailStoreError):
        resolve_detail_store(_run(detail_store="future_v9"))


def _model_run(tmp_path, *, run_uuid="run-uuid"):
    return TrainingRun(
        id=7,
        run_id=run_uuid,
        run_name="run",
        training_method="lora",
        base_model_path="model",
        output_dir=str(tmp_path),
        total_steps=10,
        detail_store=RUN_DB_V2,
        detail_schema_version=2,
        detail_state="ready",
        detail_db_name="training_run.db",
    )


def test_run_database_contains_only_run_owned_tables(tmp_path):
    run = _model_run(tmp_path)
    factory = initialize_run_detail_database(run)
    db = factory()
    db.add(TrainingMetrics(run_id=run.id, step=1, loss=0.25))
    db.commit()
    tables = {
        row[0] for row in db.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ))
    }
    db.close()

    assert {"training_runs", "training_metrics", "training_checkpoints", "training_samples"} <= tables
    assert "training_presets" not in tables
    opened = open_run_detail_session(run)
    assert opened.query(TrainingMetrics).one().loss == 0.25
    opened.close()


def test_existing_run_database_rejects_another_run_identity(tmp_path):
    initialize_run_detail_database(_model_run(tmp_path, run_uuid="first"))
    with pytest.raises(DetailStoreError, match="identity mismatch"):
        initialize_run_detail_database(_model_run(tmp_path, run_uuid="second"))
