"""
Migration: Add tag_data column to dataset_captions table

This migration adds a tag_data TEXT column to store pre-categorized tags
in JSON format: [{"tag": "1girl", "category": "General"}, ...]
"""
import sqlite3
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))


def migrate_add_tag_data_column(db_path: str):
    """
    Add tag_data column to dataset_captions table.

    Args:
        db_path: Path to datasets.db
    """
    print(f"[Migration] Connecting to {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if column already exists
        cursor.execute("PRAGMA table_info(dataset_captions)")
        columns = [row[1] for row in cursor.fetchall()]

        if "tag_data" in columns:
            print("[Migration] tag_data column already exists, skipping...")
            return

        # Add tag_data column
        print("[Migration] Adding tag_data column to dataset_captions table...")
        cursor.execute("""
            ALTER TABLE dataset_captions
            ADD COLUMN tag_data TEXT
        """)

        conn.commit()
        print("[Migration] Successfully added tag_data column")

    except Exception as e:
        print(f"[Migration] ERROR: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    # Default database path
    db_path = backend_dir.parent / "datasets.db"

    if len(sys.argv) > 1:
        db_path = sys.argv[1]

    migrate_add_tag_data_column(str(db_path))
