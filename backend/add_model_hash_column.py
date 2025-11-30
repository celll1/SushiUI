"""
Migration script to add missing columns to database tables

This script automatically detects and adds missing columns to database tables
when new fields are added to the models. It works with the separated databases:
- gallery.db (generated_images table)
- datasets.db (datasets, dataset_items, etc.)
- training.db (training_runs, training_checkpoints, etc.)
"""
import sqlite3
import os

# Database paths (relative to project root)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Define databases and their tables to check
DATABASES = {
    "gallery.db": {
        "generated_images": [
            ("image_hash", "TEXT"),
            ("source_image_hash", "TEXT"),
            ("mask_data", "TEXT"),
            ("lora_names", "TEXT"),
            ("model_hash", "TEXT"),
            ("cfg_schedule_type", "TEXT"),
            ("cfg_schedule_min", "REAL"),
            ("cfg_schedule_max", "REAL"),
            ("cfg_schedule_power", "REAL"),
            ("cfg_rescale_snr_alpha", "REAL"),
            ("dynamic_threshold_percentile", "REAL"),
            ("dynamic_threshold_mimic_scale", "REAL"),
        ],
        "user_settings": [
            # Add any new user_settings columns here if needed
        ],
    },
    "datasets.db": {
        "datasets": [
            # Add any new dataset columns here if needed
        ],
        "dataset_items": [
            # Add any new dataset_items columns here if needed
        ],
    },
    "training.db": {
        "training_runs": [
            # Add any new training_runs columns here if needed
        ],
        "training_checkpoints": [
            # Add any new training_checkpoints columns here if needed
        ],
    },
}

def migrate_database(db_path, tables_to_migrate):
    """Add missing columns to tables in the specified database"""
    if not os.path.exists(db_path):
        print(f"  Database not found: {db_path}")
        return False

    print(f"\nProcessing database: {db_path}")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        for table_name, columns_to_add in tables_to_migrate.items():
            if not columns_to_add:
                continue  # Skip empty column lists

            # Check if table exists
            cursor.execute(f"""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='{table_name}'
            """)
            if not cursor.fetchone():
                print(f"  [SKIP] Table '{table_name}' does not exist")
                continue

            # Check existing columns
            cursor.execute(f"PRAGMA table_info({table_name})")
            existing_columns = [row[1] for row in cursor.fetchall()]

            # Add missing columns
            added_count = 0
            for column_name, column_type in columns_to_add:
                if column_name in existing_columns:
                    print(f"  [SKIP] {table_name}.{column_name} already exists")
                else:
                    cursor.execute(f"""
                        ALTER TABLE {table_name}
                        ADD COLUMN {column_name} {column_type}
                    """)
                    print(f"  [OK] Added {table_name}.{column_name} ({column_type})")
                    added_count += 1

            if added_count > 0:
                print(f"  [INFO] Added {added_count} column(s) to {table_name}")

        conn.commit()
        conn.close()
        print(f"  [DONE] Completed processing {db_path}")
        return True

    except Exception as e:
        print(f"  [ERROR] Error processing {db_path}: {e}")
        if conn:
            conn.close()
        return False

def main():
    """Run migrations on all databases"""
    print("=" * 60)
    print("Database Migration Script")
    print("=" * 60)
    print("\nThis script will add missing columns to database tables.")
    print("It's safe to run multiple times - existing columns are skipped.\n")

    success_count = 0
    total_count = 0

    for db_name, tables in DATABASES.items():
        db_path = os.path.join(project_root, db_name)
        total_count += 1

        if migrate_database(db_path, tables):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"Migration completed: {success_count}/{total_count} databases processed")
    print("=" * 60)

if __name__ == "__main__":
    main()
