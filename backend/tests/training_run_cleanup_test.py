"""Training-run deletion removes detail rows without relying on FK defaults."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import (
    TaggerTrainingMetrics,
    TaggerTrainingRun,
    TrainingBase,
    TrainingMetrics,
    TrainingRun,
)
from database.training_cleanup import (
    delete_tagger_training_run_record,
    delete_training_run_record,
)


def _session():
    engine = create_engine("sqlite:///:memory:")
    TrainingBase.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_diffusion_run_delete_removes_metrics_explicitly():
    db = _session()
    run = TrainingRun(
        run_id="run-uuid",
        run_name="run",
        training_method="lora",
        base_model_path="model",
        output_dir="output",
        total_steps=10,
    )
    db.add(run)
    db.flush()
    db.add(TrainingMetrics(run_id=run.id, step=1))
    db.commit()

    delete_training_run_record(db, run)
    db.commit()

    assert db.query(TrainingRun).count() == 0
    assert db.query(TrainingMetrics).count() == 0
    db.close()


def test_tagger_run_delete_removes_string_keyed_metrics_explicitly():
    db = _session()
    run = TaggerTrainingRun(
        run_id="tagger-uuid",
        run_name="tagger",
        vision_encoder_path="model",
    )
    db.add(run)
    db.add(TaggerTrainingMetrics(run_id=run.run_id, step=1))
    db.commit()

    delete_tagger_training_run_record(db, run)
    db.commit()

    assert db.query(TaggerTrainingRun).count() == 0
    assert db.query(TaggerTrainingMetrics).count() == 0
    db.close()
