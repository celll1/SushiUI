"""
Build tag_data for existing captions in database

This script reads existing "tags" captions, looks up categories from TagDictionary,
and populates the tag_data column with pre-categorized tag information.
"""
import sqlite3
import json
import sys
from pathlib import Path
from typing import List, Dict

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))


def load_tag_dictionary(conn: sqlite3.Connection) -> Dict[str, str]:
    """
    Load tag dictionary from database.

    Returns:
        Dictionary mapping tag -> category
    """
    cursor = conn.cursor()
    cursor.execute("SELECT tag, category FROM tag_dictionary")
    tag_dict = {row[0]: row[1] for row in cursor.fetchall()}
    print(f"[BuildTagData] Loaded {len(tag_dict)} tags from tag_dictionary")
    return tag_dict


def build_tag_data_from_content(content: str, tag_dict: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Build tag_data from comma-separated tags string.

    Args:
        content: Comma-separated tags (e.g., "1girl, long_hair, smile")
        tag_dict: Dictionary mapping tag -> category

    Returns:
        List of {"tag": "1girl", "category": "General"}
    """
    if not content:
        return []

    tags = [tag.strip() for tag in content.split(",") if tag.strip()]
    tag_data = []

    for tag in tags:
        # Normalize tag for lookup (lowercase, replace spaces with underscores)
        normalized_tag = tag.lower().replace(" ", "_")

        # Look up category
        category = tag_dict.get(normalized_tag, "General")  # Default to "General" if not found

        tag_data.append({
            "tag": tag,  # Use original tag (preserve casing)
            "category": category
        })

    return tag_data


def build_tag_data_for_existing_captions(db_path: str):
    """
    Build tag_data for all existing "tags" captions.

    Args:
        db_path: Path to datasets.db
    """
    print(f"[BuildTagData] Connecting to {db_path}")
    conn = sqlite3.connect(db_path)

    try:
        # Load tag dictionary
        tag_dict = load_tag_dictionary(conn)

        # Get all "tags" captions without tag_data
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, content
            FROM dataset_captions
            WHERE caption_type = 'tags'
            AND (tag_data IS NULL OR tag_data = '')
        """)

        captions = cursor.fetchall()
        total = len(captions)
        print(f"[BuildTagData] Found {total} captions to process")

        if total == 0:
            print("[BuildTagData] No captions to process")
            return

        # Process each caption
        updated = 0
        for idx, (caption_id, content) in enumerate(captions):
            # Build tag_data
            tag_data = build_tag_data_from_content(content, tag_dict)

            # Update database
            cursor.execute("""
                UPDATE dataset_captions
                SET tag_data = ?
                WHERE id = ?
            """, (json.dumps(tag_data), caption_id))

            updated += 1

            # Progress update every 1000 items
            if (idx + 1) % 1000 == 0:
                conn.commit()
                progress_pct = ((idx + 1) / total) * 100.0
                print(f"[BuildTagData] Processed {idx + 1}/{total} ({progress_pct:.1f}%)")

        # Final commit
        conn.commit()
        print(f"[BuildTagData] Successfully updated {updated} captions")

    except Exception as e:
        print(f"[BuildTagData] ERROR: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    # Default database path
    db_path = backend_dir.parent / "datasets.db"

    if len(sys.argv) > 1:
        db_path = sys.argv[1]

    build_tag_data_for_existing_captions(str(db_path))
