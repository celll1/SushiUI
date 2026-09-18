from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from database import ensure_training_schema
from database.auto_migrate import auto_migrate
from database.models import TrainingBase, TrainingMetrics


def test_training_worker_reconciles_an_old_catalog_before_orm_use(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'training.db'}")
    TrainingBase.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO training_runs "
            "(id, run_id, run_name, training_method, base_model_path, total_steps, output_dir) "
            "VALUES (127, 'uuid-127', 'run127', 'full_finetune', 'model', 10000, 'output')"
        ))
        for column in (
            "detail_store", "detail_schema_version", "detail_state", "detail_db_name"
        ):
            connection.execute(text(f"ALTER TABLE training_runs DROP COLUMN {column}"))

    ensure_training_schema(engine)

    columns = {column["name"] for column in inspect(engine).get_columns("training_runs")}
    assert {
        "detail_store", "detail_schema_version", "detail_state", "detail_db_name"
    } <= columns
    with engine.connect() as connection:
        assert connection.execute(text(
            "SELECT run_name FROM training_runs WHERE id = 127"
        )).scalar_one() == "run127"


def test_train_runner_reconciles_before_opening_the_catalog():
    source = (Path(__file__).parents[1] / "core" / "training" / "train_runner.py").read_text(
        encoding="utf-8"
    )
    worker = source[source.index("def main()") :]
    assert worker.index("ensure_training_schema()") < worker.index("get_training_db()")


def test_training_metrics_migration_keeps_overlapping_resume_steps(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE training_metrics (
                id INTEGER NOT NULL PRIMARY KEY,
                run_id INTEGER NOT NULL,
                step INTEGER NOT NULL,
                resume_seq INTEGER NOT NULL DEFAULT 0,
                loss FLOAT,
                CONSTRAINT uq_run_step UNIQUE (run_id, step)
            )
        """))
        connection.execute(text(
            "INSERT INTO training_metrics (run_id, step, resume_seq, loss) "
            "VALUES (1, 8000, 0, 1.0)"
        ))

    auto_migrate(
        engine, TrainingBase, "legacy.db", model_classes=(TrainingMetrics,)
    )

    with engine.begin() as connection:
        connection.execute(text(
            "INSERT INTO training_metrics (run_id, step, resume_seq, loss) "
            "VALUES (1, 8000, 1, 2.0)"
        ))
    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(text(
                "INSERT INTO training_metrics (run_id, step, resume_seq, loss) "
                "VALUES (1, 8000, 1, 3.0)"
            ))
