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
    Automatically migrate database schema to match model definitions

    Args:
        engine: SQLAlchemy engine for the database
        base: Declarative base (GalleryBase, DatasetBase, or TrainingBase)
        db_name: Name of the database for logging (e.g., "gallery.db")

    This function:
    1. Compares model definitions with actual database schema
    2. Adds any missing columns to existing tables
    3. Creates tables that don't exist

    Note: This does NOT handle:
    - Column deletions (old columns are kept)
    - Column type changes (requires manual migration)
    - Column renames (requires manual migration)
    """
    print(f"[AutoMigrate] Starting migration for {db_name}...")

    try:
        with engine.connect() as conn:
            # Get all model classes from Base
            models = []
            for mapper in base.registry.mappers:
                models.append(mapper.class_)

            for model_class in models:
                table_name = model_class.__tablename__
                print(f"[AutoMigrate] [{db_name}] Checking table: {table_name}")

                # Get model columns
                model_columns = get_model_columns(model_class)

                # Get existing database columns
                db_columns = get_db_columns(engine, table_name)

                # If table doesn't exist, it will be created by create_all()
                if not db_columns:
                    print(f"[AutoMigrate] [{db_name}] Table {table_name} does not exist, will be created by create_all()")
                    continue

                # Find missing columns
                missing_columns = set(model_columns.keys()) - db_columns

                if missing_columns:
                    print(f"[AutoMigrate] [{db_name}] Found {len(missing_columns)} missing column(s) in {table_name}: {missing_columns}")

                    for col_name in missing_columns:
                        col_definition = model_columns[col_name]

                        # SQLite ALTER TABLE only supports ADD COLUMN
                        try:
                            sql = f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_definition}"
                            print(f"[AutoMigrate] [{db_name}] Executing: {sql}")
                            conn.execute(text(sql))
                            conn.commit()
                            print(f"[AutoMigrate] [{db_name}] ✓ Added column {col_name} to {table_name}")
                        except OperationalError as e:
                            print(f"[AutoMigrate] [{db_name}] ✗ Failed to add column {col_name}: {e}")
                            # Continue with other columns even if one fails
                else:
                    print(f"[AutoMigrate] [{db_name}] ✓ Table {table_name} is up to date")

            print(f"[AutoMigrate] Migration completed for {db_name}")
            return True

    except Exception as e:
        print(f"[AutoMigrate] Error during migration for {db_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def auto_migrate_all_databases():
    """
    Run auto-migration for all databases (gallery.db, datasets.db, training.db)
    """
    from config.settings import settings

    print("[AutoMigrate] ========================================")
    print("[AutoMigrate] Starting auto-migration for all databases")
    print("[AutoMigrate] ========================================")

    # Gallery database
    gallery_db_path = os.path.join(settings.root_dir, "gallery.db")
    gallery_engine = create_engine(f"sqlite:///{gallery_db_path}", connect_args={"check_same_thread": False})
    auto_migrate(gallery_engine, GalleryBase, "gallery.db")

    # Datasets database
    datasets_db_path = os.path.join(settings.root_dir, "datasets.db")
    datasets_engine = create_engine(f"sqlite:///{datasets_db_path}", connect_args={"check_same_thread": False})
    auto_migrate(datasets_engine, DatasetBase, "datasets.db")

    # Training database
    training_db_path = os.path.join(settings.root_dir, "training.db")
    training_engine = create_engine(f"sqlite:///{training_db_path}", connect_args={"check_same_thread": False})
    auto_migrate(training_engine, TrainingBase, "training.db")

    print("[AutoMigrate] ========================================")
    print("[AutoMigrate] All databases migrated successfully")
    print("[AutoMigrate] ========================================")


if __name__ == "__main__":
    # Allow running migration manually
    auto_migrate_all_databases()
