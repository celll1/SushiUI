"""
Database migration: Add dataset_configs to training_runs table

This migration adds support for multiple datasets in training runs.
"""

import sqlite3
import json
from pathlib import Path

def migrate():
    db_path = Path(__file__).parent.parent.parent / "training.db"

    if not db_path.exists():
        print(f"[Migration] training.db not found at {db_path}, skipping migration")
        return

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Check if dataset_configs column already exists
    cursor.execute("PRAGMA table_info(training_runs)")
    columns = [col[1] for col in cursor.fetchall()]

    if "dataset_configs" in columns:
        print("[Migration] dataset_configs column already exists, skipping")
        conn.close()
        return

    print("[Migration] Adding dataset_configs column to training_runs table...")

    try:
        # Add dataset_configs column
        cursor.execute("ALTER TABLE training_runs ADD COLUMN dataset_configs TEXT")

        # Migrate existing data: convert dataset_id to dataset_configs
        cursor.execute("SELECT id, dataset_id FROM training_runs WHERE dataset_id IS NOT NULL")
        rows = cursor.fetchall()

        for row_id, dataset_id in rows:
            # Convert single dataset_id to dataset_configs format
            dataset_configs = [
                {
                    "dataset_id": dataset_id,
                    "caption_types": [],  # Empty = use all caption types
                    "filters": {}
                }
            ]
            cursor.execute(
                "UPDATE training_runs SET dataset_configs = ? WHERE id = ?",
                (json.dumps(dataset_configs), row_id)
            )

        conn.commit()
        print(f"[Migration] Successfully migrated {len(rows)} training runs")

    except Exception as e:
        print(f"[Migration] Error: {e}")
        conn.rollback()
        raise

    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
