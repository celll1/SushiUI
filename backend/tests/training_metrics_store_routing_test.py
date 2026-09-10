"""The unchanged metrics API reads legacy and run-owned stores correctly."""

import asyncio

import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

torch.cuda.get_device_capability = lambda *args, **kwargs: (8, 9)
torch.cuda._lazy_init = lambda *args, **kwargs: None
torch._C._cuda_init = lambda *args, **kwargs: None

from api.routes import get_training_metrics_db  # noqa: E402
from database.models import TrainingBase, TrainingMetrics, TrainingRun  # noqa: E402
from database.training_detail_store import (  # noqa: E402
    RUN_DB_V2,
    initialize_run_detail_database,
    open_run_detail_session,
)


def _central(tmp_path, *, v2):
    engine = create_engine("sqlite:///:memory:")
    TrainingBase.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    run = TrainingRun(
        run_id="run-uuid",
        run_name="run",
        training_method="lora",
        base_model_path="model",
        output_dir=str(tmp_path),
        total_steps=10,
        detail_store=RUN_DB_V2 if v2 else None,
        detail_schema_version=2 if v2 else None,
        detail_state="ready" if v2 else None,
        detail_db_name="training_run.db" if v2 else None,
    )
    db.add(run)
    db.flush()
    db.add(TrainingMetrics(run_id=run.id, step=1, loss=0.1))
    db.commit()
    return db, run


def _metrics(db, run):
    return asyncio.run(get_training_metrics_db(run.id, 1000, db))


def test_legacy_metrics_stay_in_central_database(tmp_path):
    central, run = _central(tmp_path, v2=False)
    assert _metrics(central, run)["loss"][0]["value"] == 0.1
    central.close()


def test_v2_metrics_come_from_run_database(tmp_path):
    central, run = _central(tmp_path, v2=True)
    initialize_run_detail_database(run)
    local = open_run_detail_session(run)
    local.add(TrainingMetrics(run_id=run.id, step=1, loss=0.2))
    local.commit()
    local.close()

    assert _metrics(central, run)["loss"][0]["value"] == 0.2
    central.close()


def test_missing_v2_database_falls_back_to_retained_central_rows(tmp_path):
    central, run = _central(tmp_path, v2=True)
    assert _metrics(central, run)["loss"][0]["value"] == 0.1
    central.close()


def test_empty_v2_database_falls_back_to_first_central_batch(tmp_path):
    central, run = _central(tmp_path, v2=True)
    initialize_run_detail_database(run)
    assert _metrics(central, run)["loss"][0]["value"] == 0.1
    central.close()
