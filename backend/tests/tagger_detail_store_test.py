"""Tagger metrics use the same legacy-compatible per-run storage boundary."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import torch

torch.cuda.get_device_capability = lambda *args, **kwargs: (8, 9)
torch.cuda._lazy_init = lambda *args, **kwargs: None
torch._C._cuda_init = lambda *args, **kwargs: None

from database.models import TaggerTrainingMetrics, TaggerTrainingRun, TrainingBase
from database.training_detail_store import (
    RUN_DB_V2,
    initialize_tagger_detail_database,
    migrate_terminal_tagger_run_to_v2,
    open_tagger_detail_session,
)


def test_tagger_detail_executor_is_shared():
    from api import routes

    first = routes._get_tagger_detail_executor()
    assert routes._get_tagger_detail_executor() is first


def test_tagger_callback_writes_v2_metrics_only_to_run_database(tmp_path):
    from api import routes

    engine = create_engine(
        f"sqlite:///{tmp_path / 'catalog.db'}",
        connect_args={"check_same_thread": False},
    )
    TrainingBase.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    central = factory()
    run = TaggerTrainingRun(
        run_id="tagger-direct",
        run_name="tagger-direct",
        status="running",
        vision_encoder_path="model",
        output_dir=str(tmp_path / "output"),
        detail_store=RUN_DB_V2,
        detail_schema_version=2,
        detail_state="ready",
        detail_db_name="tagger_training_run.db",
    )
    central.add(run)
    central.commit()
    initialize_tagger_detail_database(run)
    central.close()

    callback = routes._make_tagger_progress_callback(run.run_id, factory)
    callback(run.run_id, "step", {
        "step": 3,
        "epoch": 1,
        "loss": 0.25,
        "lr": 1e-4,
        "progress": 0.3,
    })
    callback(run.run_id, "train_f1", {
        "step": 3,
        "train_f1": 0.75,
        "train_precision": 0.8,
        "train_recall": 0.7,
    })
    routes._get_tagger_detail_executor().submit(lambda: None).result(timeout=10)

    central = factory()
    stored_run = central.query(TaggerTrainingRun).filter_by(
        run_id=run.run_id
    ).one()
    assert stored_run.current_step == 3
    assert central.query(TaggerTrainingMetrics).count() == 0
    local = open_tagger_detail_session(stored_run)
    metric = local.query(TaggerTrainingMetrics).one()
    assert (metric.step, metric.epoch, metric.loss, metric.train_f1) == (
        3, 1, 0.25, 0.75
    )
    local.close()
    central.close()


def test_terminal_tagger_migration_retains_central_rows(tmp_path):
    engine = create_engine("sqlite:///:memory:")
    TrainingBase.metadata.create_all(engine)
    central = sessionmaker(bind=engine)()
    run = TaggerTrainingRun(
        run_id="legacy-tagger",
        run_name="tagger",
        status="completed",
        vision_encoder_path="model",
        output_dir=str(tmp_path),
    )
    central.add(run)
    central.add_all([
        TaggerTrainingMetrics(run_id=run.run_id, resume_seq=0, step=1, loss=1.0),
        TaggerTrainingMetrics(run_id=run.run_id, resume_seq=1, step=1, loss=0.5),
    ])
    central.commit()

    result = migrate_terminal_tagger_run_to_v2(central, run, batch_size=1)

    assert result["tagger_training_metrics"] == 2
    assert central.query(TaggerTrainingMetrics).count() == 2
    local = open_tagger_detail_session(run)
    assert local.query(TaggerTrainingMetrics).count() == 2
    local.close()
    central.close()
