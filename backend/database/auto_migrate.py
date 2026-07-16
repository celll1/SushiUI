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

                # One-shot data backfills for columns that were replaced by a
                # generic channel (idempotent; skipped once nothing to move).
                _backfill_data(conn, db_name, table_name, applied)

        return applied

    except Exception as e:
        print(f"[AutoMigrate] Error during migration for {db_name}: {e}")
        import traceback
        traceback.print_exc()
        return []


def _backfill_data(conn, db_name: str, table_name: str, applied: list) -> None:
    """Move data from legacy dedicated columns into their generic replacement.

    training_metrics.repa_loss (a former dedicated column) is now stored in the
    generic ``extra_metrics`` JSON dict under the ``repa_loss`` key. Old DBs
    still carry the orphaned column with historical values; copy them once so
    MiniT2I runs trained before the change still chart. Idempotent: only rows
    whose extra_metrics does not yet hold repa_loss are touched, and the whole
    step is skipped when the legacy column is gone (fresh DBs never had it).
    """
    if table_name != "training_metrics":
        return
    try:
        cols = get_db_columns(conn.engine, table_name)
        if "repa_loss" not in cols or "extra_metrics" not in cols:
            return  # Fresh DB (no legacy column) or column-add failed — nothing to do.
        result = conn.execute(text(
            "UPDATE training_metrics "
            "SET extra_metrics = json_set(coalesce(extra_metrics, '{}'), '$.repa_loss', repa_loss) "
            "WHERE repa_loss IS NOT NULL "
            "AND json_extract(coalesce(extra_metrics, '{}'), '$.repa_loss') IS NULL"
        ))
        conn.commit()
        moved = getattr(result, "rowcount", -1)
        if moved and moved > 0:
            applied.append(f"{table_name}.repa_loss->extra_metrics ({moved} rows)")
            print(f"[AutoMigrate] {db_name}: backfilled repa_loss into extra_metrics ({moved} rows)")
    except OperationalError as e:
        print(f"[AutoMigrate] {db_name}: repa_loss backfill skipped: {e}")


def _reconcile_indices(conn, db_name: str, table_name: str, applied: list) -> None:
    """Drop legacy indices and create the new ones for tables that changed
    their UNIQUE / Index definitions in the model.

    SQLite stores named UniqueConstraints defined in CREATE TABLE as inline
    CONSTRAINT clauses, which are NOT visible as separate index rows in
    sqlite_master.  Such inline constraints can only be removed by rebuilding
    the entire table.  This function handles both cases:

    1. Standalone ``CREATE [UNIQUE] INDEX`` entries — drop by name and recreate.
    2. Inline ``CONSTRAINT … UNIQUE (…)`` clauses in the CREATE TABLE DDL —
       detected via the table's DDL text and handled by a full table rebuild.
    """
    # -------------------------------------------------------------------
    # Plan: per-table list of (inline_constraint_fragments_to_remove,
    #        legacy_standalone_index_names_to_drop,
    #        new_index_sqls_to_create)
    # -------------------------------------------------------------------
    plan = {
        # tagger_training_metrics was originally created with an inline
        # CONSTRAINT uq_tagger_run_step UNIQUE (run_id, step).
        # That constraint is embedded in the table DDL and cannot be dropped
        # with DROP INDEX — only a table rebuild can remove it.
        "tagger_training_metrics": (
            # inline DDL fragments whose presence triggers a rebuild
            ("CONSTRAINT uq_tagger_run_step",),
            # standalone legacy index names to drop
            ("uq_tagger_run_step", "idx_tagger_run_step"),
            # new indices to create after the rebuild (or instead of the legacy ones)
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
    inline_triggers, legacy_names, create_sqls = plan[table_name]

    # ---- Check for inline constraint fragments in the CREATE TABLE DDL ----
    ddl_row = conn.execute(text(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=:t"
    ), {"t": table_name}).fetchone()
    table_ddl = ddl_row[0] if ddl_row else ""

    needs_rebuild = any(frag in table_ddl for frag in inline_triggers)
    if needs_rebuild:
        _rebuild_table_remove_inline_constraints(conn, db_name, table_name, create_sqls, applied)
        return

    # ---- Normal path: drop standalone legacy indices, create new ones ----
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
        idx_name = sql.split("IF NOT EXISTS")[1].split("ON")[0].strip()
        if idx_name in existing and not changed:
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


def _rebuild_table_remove_inline_constraints(
    conn, db_name: str, table_name: str, create_sqls: tuple, applied: list
) -> None:
    """Rebuild *table_name* without any inline UNIQUE/CHECK constraints.

    SQLite does not support ALTER TABLE DROP CONSTRAINT, so removing an
    inline table constraint requires recreating the table:
        1. CREATE … _new with correct schema (no inline constraints)
        2. INSERT INTO _new SELECT * FROM old
        3. DROP old / RENAME _new → old
        4. CREATE INDEX …

    Column list is derived from the live table to stay schema-agnostic.
    """
    # Fetch column names in ordinal order
    col_rows = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
    cols = [r[1] for r in col_rows]  # r[1] = name
    cols_csv = ", ".join(cols)

    # Build a minimal CREATE TABLE DDL: same columns, no inline constraints.
    # Each column keeps its type, NOT NULL, and DEFAULT from the live schema.
    col_defs = []
    for r in col_rows:
        # r = (cid, name, type, notnull, dflt_value, pk)
        cid, name, ctype, notnull, dflt, pk = r
        part = f"{name} {ctype}"
        if pk:
            part += " NOT NULL PRIMARY KEY"
        elif notnull:
            part += " NOT NULL"
        if dflt is not None:
            part += f" DEFAULT {dflt}"
        col_defs.append(part)

    new_table = f"{table_name}_new"
    create_ddl = f"CREATE TABLE {new_table} (\n    " + ",\n    ".join(col_defs) + "\n)"

    try:
        conn.execute(text(f"DROP TABLE IF EXISTS {new_table}"))
        conn.execute(text(create_ddl))
        conn.execute(text(
            f"INSERT INTO {new_table} ({cols_csv}) SELECT {cols_csv} FROM {table_name}"
        ))
        conn.execute(text(f"DROP TABLE {table_name}"))
        conn.execute(text(f"ALTER TABLE {new_table} RENAME TO {table_name}"))

        for sql in create_sqls:
            try:
                conn.execute(text(sql))
            except OperationalError as e:
                print(f"[AutoMigrate] {db_name}: index creation after rebuild failed: {e}")

        conn.commit()
        print(f"[AutoMigrate] {db_name}: rebuilt {table_name} (removed inline constraints)")
        applied.append(f"{table_name}.rebuild")
    except Exception as e:
        conn.rollback()
        print(f"[AutoMigrate] {db_name}: table rebuild for {table_name} failed: {e}")


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
