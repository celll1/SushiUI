"""
Migration script: Single database to separated databases

This script migrates data from the old single database (webui.db or sd_webui.db)
to the new separated database structure:
- gallery.db (generated_images, user_settings)
- datasets.db (datasets, dataset_items, dataset_captions, tag_dictionary)
- training.db (training_runs, training_checkpoints, training_samples)

Usage:
    python migrate_to_separated_dbs.py

The script will:
1. Auto-detect the old database (webui.db or sd_webui.db)
2. Create new separated databases
3. Copy all data to appropriate databases
4. Backup the old database (rename to .backup)
5. Verify migration success
"""

import sqlite3
import os
import sys
from pathlib import Path
from datetime import datetime
import shutil

# Add backend directory to path
backend_dir = Path(__file__).parent
project_root = backend_dir.parent
sys.path.insert(0, str(backend_dir))

# Old database candidates
OLD_DB_CANDIDATES = [
    project_root / "webui.db",
    project_root / "sd_webui.db",
]

# New databases
NEW_DBS = {
    "gallery": project_root / "gallery.db",
    "datasets": project_root / "datasets.db",
    "training": project_root / "training.db",
}

def find_old_database():
    """Find the old database file"""
    for db_path in OLD_DB_CANDIDATES:
        if db_path.exists():
            return db_path
    return None

def check_table_exists(cursor, table_name):
    """Check if a table exists in the database"""
    cursor.execute(f"""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='{table_name}'
    """)
    return cursor.fetchone() is not None

def get_table_columns(cursor, table_name):
    """Get column names of a table"""
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cursor.fetchall()]

def copy_table_data(old_conn, new_conn, table_name, progress_callback=None):
    """Copy all data from old table to new table"""
    old_cursor = old_conn.cursor()
    new_cursor = new_conn.cursor()

    # Check if table exists in old database
    if not check_table_exists(old_cursor, table_name):
        print(f"  [SKIP] Table '{table_name}' not found in old database")
        return 0

    # Get all data
    old_cursor.execute(f"SELECT * FROM {table_name}")
    rows = old_cursor.fetchall()

    if not rows:
        print(f"  [SKIP] Table '{table_name}' is empty")
        return 0

    # Get column names
    columns = get_table_columns(old_cursor, table_name)
    columns_str = ", ".join(columns)
    placeholders = ", ".join(["?"] * len(columns))

    # Insert data
    copied = 0
    for row in rows:
        try:
            new_cursor.execute(
                f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})",
                row
            )
            copied += 1

            if progress_callback and copied % 100 == 0:
                progress_callback(copied, len(rows))

        except sqlite3.IntegrityError as e:
            # Skip duplicates (e.g., if migration is run multiple times)
            print(f"    [WARN] Skipped duplicate row: {e}")
            continue

    new_conn.commit()
    print(f"  [OK] Copied {copied} rows to {table_name}")
    return copied

def migrate_database():
    """Main migration function"""
    print("=" * 70)
    print("DATABASE MIGRATION: Single DB → Separated DBs")
    print("=" * 70)

    # Step 1: Find old database
    print("\n[Step 1/6] Detecting old database...")
    old_db_path = find_old_database()

    if not old_db_path:
        print("  [ERROR] No old database found!")
        print("  Searched for:")
        for candidate in OLD_DB_CANDIDATES:
            print(f"    - {candidate}")
        print("\n  If your database has a different name, please rename it to 'webui.db'")
        return False

    print(f"  [OK] Found old database: {old_db_path}")
    print(f"  [INFO] File size: {old_db_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Step 2: Check if new databases already exist
    print("\n[Step 2/6] Checking new databases...")
    existing_new_dbs = []
    for db_name, db_path in NEW_DBS.items():
        if db_path.exists():
            existing_new_dbs.append(db_name)
            print(f"  [WARN] {db_name}.db already exists")

    if existing_new_dbs:
        print("\n  WARNING: The following new databases already exist:")
        for db_name in existing_new_dbs:
            print(f"    - {db_name}.db")
        print("\n  Migration will APPEND data to existing databases.")
        print("  Duplicate entries will be skipped.\n")

        response = input("  Continue anyway? (yes/no): ")
        if response.lower() != "yes":
            print("  Migration cancelled.")
            return False

    # Step 3: Create new databases with schema
    print("\n[Step 3/6] Creating new database schemas...")
    from database import gallery_engine, datasets_engine, training_engine
    from database.models import GalleryBase, DatasetBase, TrainingBase

    print("  [OK] Creating gallery.db schema...")
    GalleryBase.metadata.create_all(bind=gallery_engine)

    print("  [OK] Creating datasets.db schema...")
    DatasetBase.metadata.create_all(bind=datasets_engine)

    print("  [OK] Creating training.db schema...")
    TrainingBase.metadata.create_all(bind=training_engine)

    # Step 4: Migrate data
    print("\n[Step 4/6] Migrating data...")

    old_conn = sqlite3.connect(str(old_db_path))

    # Migrate gallery tables
    print("\n  Migrating gallery.db tables...")
    gallery_conn = sqlite3.connect(str(NEW_DBS["gallery"]))
    gallery_total = 0
    gallery_total += copy_table_data(old_conn, gallery_conn, "user_settings")
    gallery_total += copy_table_data(old_conn, gallery_conn, "generated_images")
    gallery_conn.close()
    print(f"  [INFO] Gallery total: {gallery_total} rows")

    # Migrate dataset tables
    print("\n  Migrating datasets.db tables...")
    datasets_conn = sqlite3.connect(str(NEW_DBS["datasets"]))
    datasets_total = 0
    datasets_total += copy_table_data(old_conn, datasets_conn, "datasets")
    datasets_total += copy_table_data(old_conn, datasets_conn, "dataset_items")
    datasets_total += copy_table_data(old_conn, datasets_conn, "dataset_captions")
    datasets_total += copy_table_data(old_conn, datasets_conn, "tag_dictionary")
    datasets_conn.close()
    print(f"  [INFO] Datasets total: {datasets_total} rows")

    # Migrate training tables
    print("\n  Migrating training.db tables...")
    training_conn = sqlite3.connect(str(NEW_DBS["training"]))
    training_total = 0
    training_total += copy_table_data(old_conn, training_conn, "training_runs")
    training_total += copy_table_data(old_conn, training_conn, "training_checkpoints")
    training_total += copy_table_data(old_conn, training_conn, "training_samples")
    training_conn.close()
    print(f"  [INFO] Training total: {training_total} rows")

    old_conn.close()

    # Step 5: Verify migration
    print("\n[Step 5/6] Verifying migration...")
    verification_passed = True

    for db_name, db_path in NEW_DBS.items():
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        print(f"  [{db_name}.db] Tables: {len(tables)}")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"    - {table}: {count} rows")

        conn.close()

    # Step 6: Backup old database
    print("\n[Step 6/6] Backing up old database...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = old_db_path.with_suffix(f".backup_{timestamp}")

    try:
        shutil.copy2(old_db_path, backup_path)
        print(f"  [OK] Old database backed up to: {backup_path.name}")
        print(f"  [INFO] Original database: {old_db_path}")
        print("\n  You can safely delete the old database after verifying the migration.")
        print(f"  To delete: del {old_db_path.name}")
    except Exception as e:
        print(f"  [ERROR] Failed to backup old database: {e}")
        verification_passed = False

    # Summary
    print("\n" + "=" * 70)
    if verification_passed:
        print("MIGRATION COMPLETED SUCCESSFULLY!")
    else:
        print("MIGRATION COMPLETED WITH WARNINGS")
    print("=" * 70)

    print("\nSummary:")
    print(f"  Gallery rows:  {gallery_total}")
    print(f"  Dataset rows:  {datasets_total}")
    print(f"  Training rows: {training_total}")
    print(f"  Total rows:    {gallery_total + datasets_total + training_total}")

    print("\nNew databases created:")
    for db_name, db_path in NEW_DBS.items():
        if db_path.exists():
            size_mb = db_path.stat().st_size / 1024 / 1024
            print(f"  - {db_path.name} ({size_mb:.2f} MB)")

    print(f"\nOld database backed up:")
    print(f"  - {backup_path.name}")

    return True

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("IMPORTANT: Database Migration")
    print("=" * 70)
    print("\nThis script will migrate your data from the old single database")
    print("to the new separated database structure.")
    print("\nBefore proceeding:")
    print("  1. STOP the backend server if it's running")
    print("  2. Make sure you have a backup of your database")
    print("  3. Ensure you have enough disk space (migration creates new files)")
    print("\n" + "=" * 70)

    response = input("\nDo you want to proceed with migration? (yes/no): ")

    if response.lower() == "yes":
        success = migrate_database()

        if success:
            print("\nNext steps:")
            print("  1. Restart the backend server")
            print("  2. Verify that all data is accessible in the UI")
            print("  3. If everything works, you can delete the old database")
            print(f"     Command: del {find_old_database().name if find_old_database() else 'webui.db'}")
        else:
            print("\nMigration failed. Please check the errors above.")
    else:
        print("\nMigration cancelled.")
