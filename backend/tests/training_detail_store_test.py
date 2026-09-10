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
    migrate_terminal_run_to_v2,
    open_run_detail_session,
    open_training_history_session,
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


def test_v2_history_session_writes_only_to_run_database(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from database.models import TrainingBase

    engine = create_engine("sqlite:///:memory:")
    TrainingBase.metadata.create_all(engine)
    central = sessionmaker(bind=engine)()
    run = _model_run(tmp_path)
    central.add(run)
    central.commit()
    initialize_run_detail_database(run)

    history, owned, selected = open_training_history_session(central, run.id)
    assert owned is True
    assert selected.run_id == run.run_id
    history.add(TrainingMetrics(run_id=run.id, step=1, loss=0.25))
    history.commit()
    history.close()

    assert central.query(TrainingMetrics).count() == 0
    local = open_run_detail_session(run)
    assert local.query(TrainingMetrics).one().loss == 0.25
    local.close()
    central.close()


def test_legacy_history_session_keeps_central_writes(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from database.models import TrainingBase

    engine = create_engine("sqlite:///:memory:")
    TrainingBase.metadata.create_all(engine)
    central = sessionmaker(bind=engine)()
    run = _model_run(tmp_path)
    run.detail_store = None
    run.detail_schema_version = None
    run.detail_state = None
    run.detail_db_name = None
    central.add(run)
    central.commit()

    history, owned, _ = open_training_history_session(central, run.id)
    assert history is central
    assert owned is False
    history.add(TrainingMetrics(run_id=run.id, step=1, loss=0.5))
    history.commit()

    assert central.query(TrainingMetrics).one().loss == 0.5
    central.close()


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
