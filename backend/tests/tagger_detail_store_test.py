"""Tagger metrics use the same legacy-compatible per-run storage boundary."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import TaggerTrainingMetrics, TaggerTrainingRun, TrainingBase
from database.training_detail_store import (
    RUN_DB_V2,
    initialize_tagger_detail_database,
    mirror_tagger_metrics_to_run_database,
    open_tagger_detail_session,
)


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
