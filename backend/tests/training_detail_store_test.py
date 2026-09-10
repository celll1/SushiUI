"""Compatibility and path-safety tests for training detail-store routing."""

from types import SimpleNamespace

import pytest
from sqlalchemy import text

from database.training_detail_store import (
    CENTRAL_V1,
    RUN_DB_V2,
    DetailStoreError,
    detail_db_path,
    delete_run_database_metrics_after,
    initialize_run_detail_database,
    mirror_metrics_to_run_database,
    migrate_terminal_run_to_v2,
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


def test_metric_mirror_repairs_tail_and_same_step_updates(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from database.models import TrainingBase

    engine = create_engine("sqlite:///:memory:")
    TrainingBase.metadata.create_all(engine)
    central = sessionmaker(bind=engine)()
    run = _model_run(tmp_path)
    central.add(run)
    central.add_all([
        TrainingMetrics(run_id=run.id, step=1, loss=1.0),
        TrainingMetrics(run_id=run.id, step=2, loss=2.0),
    ])
    central.commit()
    initialize_run_detail_database(run)

    mirror_metrics_to_run_database(run, central, [1])
    central.query(TrainingMetrics).filter_by(run_id=run.id, step=1).one().loss = 0.5
    central.commit()
    mirror_metrics_to_run_database(run, central, [1, 2])

    local = open_run_detail_session(run)
    values = [
        row.loss for row in local.query(TrainingMetrics)
        .order_by(TrainingMetrics.step).all()
    ]
    local.close()
    central.close()
    assert values == [0.5, 2.0]


def test_run_database_rewind_deletes_future_metrics(tmp_path):
    run = _model_run(tmp_path)
    factory = initialize_run_detail_database(run)
    local = factory()
    local.add_all([
        TrainingMetrics(run_id=run.id, step=step, loss=float(step))
        for step in range(1, 5)
    ])
    local.commit()
    local.close()

    assert delete_run_database_metrics_after(run, 2) == 2
    opened = open_run_detail_session(run)
    assert [row.step for row in opened.query(TrainingMetrics).all()] == [1, 2]
    opened.close()


def test_terminal_migration_copies_and_retains_central_history(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from database.models import (
        TrainingBase,
        TrainingCheckpoint,
        TrainingSample,
    )

    engine = create_engine("sqlite:///:memory:")
    TrainingBase.metadata.create_all(engine)
    central = sessionmaker(bind=engine)()
    run = _model_run(tmp_path)
    run.detail_store = None
    run.detail_schema_version = None
    run.detail_state = None
    run.detail_db_name = None
    run.status = "completed"
    central.add(run)
    central.add_all([
        TrainingMetrics(run_id=run.id, step=1, loss=0.5),
        TrainingMetrics(run_id=run.id, step=2, loss=0.25),
        TrainingCheckpoint(run_id=run.id, checkpoint_name="step-2", step=2,
                           file_path="step-2.safetensors"),
        TrainingSample(run_id=run.id, step=2, prompt="p", image_path="p.png"),
    ])
    central.commit()

    result = migrate_terminal_run_to_v2(central, run, batch_size=1)

    assert result["training_metrics"] == 2
    assert central.query(TrainingMetrics).filter_by(run_id=run.id).count() == 2
    assert run.detail_store == RUN_DB_V2
    local = open_run_detail_session(run)
    assert local.query(TrainingMetrics).count() == 2
    assert local.query(TrainingCheckpoint).count() == 1
    assert local.query(TrainingSample).count() == 1
    local.close()
    central.close()


def test_nonterminal_migration_is_refused(tmp_path):
    run = _model_run(tmp_path)
    run.detail_store = None
    run.status = "running"
    with pytest.raises(DetailStoreError, match="only terminal runs"):
        migrate_terminal_run_to_v2(None, run)
