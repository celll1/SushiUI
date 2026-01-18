#!/usr/bin/env python3
"""
Migration: Remove tags_ordered captions from dataset database

tags_ordered was used for category-ordered tags, but is no longer needed
since tag ordering is now handled dynamically in caption processing.

Date: 2025-12-25
"""

import sqlite3
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    db_path = "datasets.db"

    print(f"[Migration] Connecting to {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Count tags_ordered captions before deletion
        cursor.execute("""
            SELECT COUNT(*)
            FROM dataset_captions
            WHERE caption_type = 'tags_ordered'
        """)
        count_before = cursor.fetchone()[0]
        print(f"[Migration] Found {count_before:,} tags_ordered captions")

        if count_before == 0:
            print("[Migration] No tags_ordered captions found. Nothing to do.")
            return

        # Delete tags_ordered captions
        print("[Migration] Deleting tags_ordered captions...")
        cursor.execute("""
            DELETE FROM dataset_captions
            WHERE caption_type = 'tags_ordered'
        """)
        deleted = cursor.rowcount
        print(f"[Migration] Deleted {deleted:,} captions")

        # Commit changes
        conn.commit()
        print("[Migration] Migration completed successfully")

        # Verify deletion
        cursor.execute("""
            SELECT COUNT(*)
            FROM dataset_captions
            WHERE caption_type = 'tags_ordered'
        """)
        count_after = cursor.fetchone()[0]
        print(f"[Migration] Remaining tags_ordered captions: {count_after}")

        # Show remaining caption types
        cursor.execute("""
            SELECT caption_type, field_category, COUNT(*) as count
            FROM dataset_captions
            GROUP BY caption_type, field_category
            ORDER BY count DESC
        """)
        print("\n[Migration] Remaining caption types:")
        for row in cursor.fetchall():
            print(f"  {row[0]:20s} ({row[1]:10s}): {row[2]:,}")

    except Exception as e:
        print(f"[Migration] Error: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    main()
