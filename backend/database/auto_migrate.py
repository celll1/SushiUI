"""
Automatic database migration system

This module automatically detects and applies schema changes to the database.
It compares the current database schema with the model definitions and adds
any missing columns.

Usage:
    from database.auto_migrate import auto_migrate
    auto_migrate()
"""

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError
from config.settings import settings
from database.models import Base
import logging

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


def auto_migrate():
    """
    Automatically migrate database schema to match model definitions

    This function:
    1. Compares model definitions with actual database schema
    2. Adds any missing columns to existing tables
    3. Creates tables that don't exist

    Note: This does NOT handle:
    - Column deletions (old columns are kept)
    - Column type changes (requires manual migration)
    - Column renames (requires manual migration)
    """
    print("[Database] Starting automatic migration...")

    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})

    try:
        with engine.connect() as conn:
            # Get all model classes from Base
            models = [cls for cls in Base.__subclasses__()]

            for model_class in models:
                table_name = model_class.__tablename__
                print(f"[Database] Checking table: {table_name}")

                # Get model columns
                model_columns = get_model_columns(model_class)

                # Get existing database columns
                db_columns = get_db_columns(engine, table_name)

                # If table doesn't exist, create it
                if not db_columns:
                    print(f"[Database] Table {table_name} does not exist, creating...")
                    model_class.__table__.create(bind=engine)
                    print(f"[Database] ✓ Table {table_name} created")
                    continue

                # Find missing columns
                missing_columns = set(model_columns.keys()) - db_columns

                if missing_columns:
                    print(f"[Database] Found {len(missing_columns)} missing column(s) in {table_name}: {missing_columns}")

                    for col_name in missing_columns:
                        col_definition = model_columns[col_name]

                        # SQLite ALTER TABLE only supports ADD COLUMN
                        try:
                            sql = f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_definition}"
                            print(f"[Database] Executing: {sql}")
                            conn.execute(text(sql))
                            conn.commit()
                            print(f"[Database] ✓ Added column {col_name} to {table_name}")
                        except OperationalError as e:
                            print(f"[Database] ✗ Failed to add column {col_name}: {e}")
                            # Continue with other columns even if one fails
                else:
                    print(f"[Database] ✓ Table {table_name} is up to date")

            print("[Database] Automatic migration completed successfully")
            return True

    except Exception as e:
        print(f"[Database] Error during automatic migration: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_migration():
    """
    Verify that all model columns exist in the database

    Returns:
        bool: True if database matches models, False otherwise
    """
    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})

    models = [cls for cls in Base.__subclasses__()]
    all_ok = True

    for model_class in models:
        table_name = model_class.__tablename__
        model_columns = set(get_model_columns(model_class).keys())
        db_columns = get_db_columns(engine, table_name)

        missing = model_columns - db_columns
        if missing:
            print(f"[Database] Table {table_name} is missing columns: {missing}")
            all_ok = False

    return all_ok


if __name__ == "__main__":
    # Allow running migration manually
    auto_migrate()

    # Verify migration
    if verify_migration():
        print("[Database] ✓ All tables are up to date")
    else:
        print("[Database] ✗ Some tables need migration")
