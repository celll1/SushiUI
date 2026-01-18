"""
Clear existing tag_data to rebuild with correct categories
"""
import sqlite3
import sys
from pathlib import Path

def clear_tag_data(db_path: str):
    """
    Clear tag_data column to NULL for all captions.
    """
    # Fix Windows cp932 encoding issue: force UTF-8 for stdout/stderr
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    print(f"[ClearTagData] Connecting to {db_path}")
    conn = sqlite3.connect(db_path)

    try:
        cursor = conn.cursor()

        # Count captions with tag_data
        cursor.execute("""
            SELECT COUNT(*)
            FROM dataset_captions
            WHERE tag_data IS NOT NULL AND tag_data != ''
        """)
        count = cursor.fetchone()[0]

        print(f"[ClearTagData] Found {count:,} captions with tag_data")

        if count == 0:
            print("[ClearTagData] No tag_data to clear")
            return

        # Clear tag_data
        cursor.execute("""
            UPDATE dataset_captions
            SET tag_data = NULL
            WHERE tag_data IS NOT NULL AND tag_data != ''
        """)

        conn.commit()
        print(f"[ClearTagData] Successfully cleared tag_data for {count:,} captions")

    except Exception as e:
        print(f"[ClearTagData] ERROR: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    # Default database path
    backend_dir = Path(__file__).parent.parent
    db_path = backend_dir.parent / "datasets.db"

    if len(sys.argv) > 1:
        db_path = sys.argv[1]

    clear_tag_data(str(db_path))
