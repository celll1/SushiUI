"""
Migration: Add caption_processing column to datasets table

This migration adds the caption_processing JSON column to store
caption manipulation settings for training (dropout, shuffle, etc.)
"""
import sqlite3
import json
from pathlib import Path

def migrate():
    """Add caption_processing column to datasets table"""
    db_path = Path(__file__).parent.parent.parent / "datasets.db"

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # Check if column already exists
        cursor.execute("PRAGMA table_info(datasets)")
        columns = [row[1] for row in cursor.fetchall()]

        if "caption_processing" in columns:
            print("Column 'caption_processing' already exists, skipping migration")
            return

        # Add caption_processing column
        print("Adding 'caption_processing' column to datasets table...")
        cursor.execute("""
            ALTER TABLE datasets
            ADD COLUMN caption_processing TEXT DEFAULT '{}'
        """)

        # Initialize existing rows with empty dict
        cursor.execute("""
            UPDATE datasets
            SET caption_processing = '{}'
            WHERE caption_processing IS NULL
        """)

        conn.commit()
        print("[OK] Migration completed successfully")

    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Migration failed: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
