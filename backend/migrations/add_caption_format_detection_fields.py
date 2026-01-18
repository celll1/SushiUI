"""
Add caption format detection fields to DatasetCaption table.

This migration adds:
- field_category: "training" | "metadata"
- is_tags_format: True (tags) | False (natural language/metadata)
- tag_match_rate: 0.0-1.0 (percentage of tokens matching taglist)
"""

import sqlite3
import os

def migrate():
    """Add caption format detection fields to datasets.db"""
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "datasets.db")

    if not os.path.exists(db_path):
        print(f"[Migration] Database not found: {db_path}")
        print("[Migration] Skipping migration (will be created with new schema)")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(dataset_captions)")
        columns = [row[1] for row in cursor.fetchall()]

        if "field_category" in columns and "is_tags_format" in columns and "tag_match_rate" in columns:
            print("[Migration] Caption format detection fields already exist, skipping")
            return

        print("[Migration] Adding caption format detection fields to dataset_captions table...")

        # Add field_category column
        if "field_category" not in columns:
            cursor.execute('ALTER TABLE dataset_captions ADD COLUMN field_category TEXT DEFAULT "training"')
            print("[Migration] Added field_category column")

        # Add is_tags_format column
        if "is_tags_format" not in columns:
            cursor.execute('ALTER TABLE dataset_captions ADD COLUMN is_tags_format INTEGER DEFAULT 0')
            print("[Migration] Added is_tags_format column")

        # Add tag_match_rate column
        if "tag_match_rate" not in columns:
            cursor.execute('ALTER TABLE dataset_captions ADD COLUMN tag_match_rate REAL DEFAULT 0.0')
            print("[Migration] Added tag_match_rate column")

        # Update existing captions to have default values
        # Set existing captions to tags format (backward compatibility)
        cursor.execute('UPDATE dataset_captions SET is_tags_format = 1 WHERE is_tags_format = 0')
        print("[Migration] Set existing captions to tags format (backward compatibility)")

        conn.commit()
        print("[Migration] Successfully added caption format detection fields")

    except Exception as e:
        print(f"[Migration] Error: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
