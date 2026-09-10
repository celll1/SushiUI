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
    mirror_tagger_metrics_to_run_database,
    open_tagger_detail_session,
)


def test_tagger_detail_executor_is_shared():
    from api import routes

    first = routes._get_tagger_detail_executor()
    assert routes._get_tagger_detail_executor() is first


def test_tagger_metric_mirror_preserves_resume_key_and_updates(tmp_path):
    engine = create_engine("sqlite:///:memory:")
    TrainingBase.metadata.create_all(engine)
    central = sessionmaker(bind=engine)()
    run = TaggerTrainingRun(
        run_id="tagger-uuid",
        run_name="tagger",
        status="pending",
        vision_encoder_path="model",
        output_dir=str(tmp_path),
        detail_store=RUN_DB_V2,
        detail_schema_version=2,
        detail_state="ready",
        detail_db_name="tagger_training_run.db",
    )
    central.add(run)
    central.add(TaggerTrainingMetrics(
        run_id=run.run_id, resume_seq=1, step=4, loss=0.5
    ))
    central.commit()
    initialize_tagger_detail_database(run)

    mirror_tagger_metrics_to_run_database(run, central, [(1, 4)])
    row = central.query(TaggerTrainingMetrics).one()
    row.f1 = 0.75
    central.commit()
    mirror_tagger_metrics_to_run_database(run, central, [(1, 4)])

    local = open_tagger_detail_session(run)
    mirrored = local.query(TaggerTrainingMetrics).one()
    assert (mirrored.resume_seq, mirrored.step, mirrored.loss, mirrored.f1) == (
        1, 4, 0.5, 0.75
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
