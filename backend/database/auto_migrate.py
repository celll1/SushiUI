"""
Automatic database migration system

This module automatically detects and applies schema changes to the database.
It compares the current database schema with the model definitions and adds
any missing columns.

Usage:
    from database.auto_migrate import auto_migrate_all_databases
    auto_migrate_all_databases()
"""

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError
from database.models import GalleryBase, DatasetBase, TrainingBase
import logging
import os

logger = logging.getLogger(__name__)


def get_model_columns(model_class):
    """
    Extract column definitions from a SQLAlchemy model class

    Returns:
        dict: {column_name: column_type_string}
    """
    columns = {}
    for column in model_class.__table__.columns:
        # Get SQLite type representation
        col_type = column.type.compile(dialect=create_engine('sqlite://').dialect)

        # Add nullable constraint
        nullable = "" if column.nullable else " NOT NULL"

        # Add default value if present
        default = ""
        if column.default is not None:
            if hasattr(column.default, 'arg'):
                if callable(column.default.arg):
                    # Skip callable defaults (like datetime.utcnow)
                    pass
                else:
                    default = f" DEFAULT {column.default.arg}"

        columns[column.name] = f"{col_type}{nullable}{default}"

    return columns


def get_db_columns(engine, table_name):
    """
    Get existing columns from database table

    Returns:
        set: Set of column names that exist in the database
    """
    inspector = inspect(engine)

    # Check if table exists
    if not inspector.has_table(table_name):
        return set()

    # Get columns
    columns = inspector.get_columns(table_name)
    return {col['name'] for col in columns}


def auto_migrate(engine, base, db_name="database"):
    """
    Automatically migrate database schema to match model definitions.

    Returns a list of strings describing applied changes (empty = nothing changed).
    """
    applied = []

    try:
        with engine.connect() as conn:
            models = [mapper.class_ for mapper in base.registry.mappers]

            for model_class in models:
                table_name = model_class.__tablename__
                model_columns = get_model_columns(model_class)
                db_columns = get_db_columns(engine, table_name)

                if not db_columns:
                    # Will be created by create_all(); no action needed here
                    continue

                missing_columns = set(model_columns.keys()) - db_columns
                for col_name in missing_columns:
                    col_definition = model_columns[col_name]
                    try:
                        sql = f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_definition}"
                        conn.execute(text(sql))
                        conn.commit()
                        applied.append(f"{table_name}.{col_name}")
                        print(f"[AutoMigrate] {db_name}: added column {table_name}.{col_name}")
                    except OperationalError as e:
                        print(f"[AutoMigrate] {db_name}: failed to add {table_name}.{col_name}: {e}")

                # Per-table index reconciliation.  auto_migrate only adds
                # columns; constraint / index changes need explicit DROP +
                # CREATE.  Each statement is idempotent so reruns are no-ops.
                _reconcile_indices(conn, db_name, table_name, applied)

        return applied

    except Exception as e:
        print(f"[AutoMigrate] Error during migration for {db_name}: {e}")
        import traceback
        traceback.print_exc()
        return []


def _reconcile_indices(conn, db_name: str, table_name: str, applied: list) -> None:
    """Drop legacy indices and create the new ones for tables that changed
    their UNIQUE / Index definitions in the model.

    SQLite stores ``UniqueConstraint(..., name="X")`` as an index of that
    name (visible in ``sqlite_master`` with ``sql LIKE 'CREATE UNIQUE INDEX%'``),
    so we can swap them out via DROP / CREATE INDEX without rebuilding the
    table.  All statements are idempotent.
    """
    # Map of table → (legacy_index_names_to_drop, new_index_sql_to_create)
    plan = {
        "tagger_training_metrics": (
            ("uq_tagger_run_step", "idx_tagger_run_step"),
            (
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_tagger_run_resume_step "
                "ON tagger_training_metrics(run_id, resume_seq, step)",
                "CREATE INDEX IF NOT EXISTS idx_tagger_run_resume_step "
                "ON tagger_training_metrics(run_id, resume_seq, step)",
            ),
        ),
    }
    if table_name not in plan:
        return
    legacy_names, create_sqls = plan[table_name]

    rows = conn.execute(text(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=:t"
    ), {"t": table_name}).fetchall()
    existing = {r[0] for r in rows}

    changed = False
    for old in legacy_names:
        if old in existing:
            try:
                conn.execute(text(f"DROP INDEX {old}"))
                changed = True
            except OperationalError:
                pass

    for sql in create_sqls:
        # Extract index name from "CREATE [UNIQUE] INDEX IF NOT EXISTS <name> ..."
        idx_name = sql.split("IF NOT EXISTS")[1].split("ON")[0].strip()
        if idx_name in existing and not changed:
            # Already present from a prior run; skip silently.
            continue
        try:
            conn.execute(text(sql))
            changed = True
        except OperationalError as e:
            print(f"[AutoMigrate] {db_name}: failed to create index for {table_name}: {e}")

    if changed:
        conn.commit()
        print(f"[AutoMigrate] {db_name}: rebuilt {table_name} indices")
        applied.append(f"{table_name}.indices")


def auto_migrate_all_databases():
    """
    Run auto-migration for all databases (gallery.db, datasets.db, training.db).
    Prints a single summary line; only prints details when columns are added.
    """
    from config.settings import settings

    gallery_db_path  = os.path.join(settings.root_dir, "gallery.db")
    datasets_db_path = os.path.join(settings.root_dir, "datasets.db")
    training_db_path = os.path.join(settings.root_dir, "training.db")

    gallery_engine  = create_engine(f"sqlite:///{gallery_db_path}",  connect_args={"check_same_thread": False})
    datasets_engine = create_engine(f"sqlite:///{datasets_db_path}", connect_args={"check_same_thread": False})
    training_engine = create_engine(f"sqlite:///{training_db_path}", connect_args={"check_same_thread": False})

    changes = (
        auto_migrate(gallery_engine,  GalleryBase,  "gallery.db")  +
        auto_migrate(datasets_engine, DatasetBase,  "datasets.db") +
        auto_migrate(training_engine, TrainingBase, "training.db")
    )

    if not changes:
        print("[AutoMigrate] All schemas up to date")


if __name__ == "__main__":
    # Allow running migration manually
    auto_migrate_all_databases()
