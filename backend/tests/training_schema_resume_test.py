from pathlib import Path

from sqlalchemy import create_engine, inspect, text

from database import ensure_training_schema
from database.models import TrainingBase


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
