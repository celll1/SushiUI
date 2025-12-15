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


def normalize_tag_for_matching(tag: str) -> str:
    """
    Normalize tag for matching (same logic as frontend normalizeTagForMatching).

    Handles various escape patterns:
    - character_name_(series) → character name (series)
    - character name (series) → character name (series)
    - character name \\(series\\) → character name (series)
    - character_name_\\(series\\) → character name (series)

    Args:
        tag: Tag string in any format

    Returns:
        Normalized tag (lowercase, spaces, no escapes)
    """
    normalized = tag.strip()

    # Remove excessive escaping: \\ → nothing
    normalized = normalized.replace("\\\\", "")
    normalized = normalized.replace("\\", "")

    # Normalize underscores to spaces
    normalized = normalized.replace("_", " ")

    # Lowercase for matching
    normalized = normalized.lower()

    return normalized


def load_tag_dictionary_from_json() -> Dict[str, str]:
    """
    Load tag dictionary from taglist JSON files.

    Returns:
        Dictionary mapping normalized_tag -> category
    """
    taglist_dir = Path(__file__).parent.parent.parent / "taglist"

    if not taglist_dir.exists():
        print(f"[BuildTagData] ERROR: taglist directory not found at {taglist_dir}")
        return {}

    category_map = {
        "General": "General",
        "Character": "Character",
        "Artist": "Artist",
        "Copyright": "Copyright",
        "Meta": "Meta",
        "Model": "Model"
    }

    tag_dict = {}

    for category_name, category_key in category_map.items():
        json_path = taglist_dir / f"{category_name}.json"

        if not json_path.exists():
            print(f"[BuildTagData] WARNING: {category_name}.json not found, skipping")
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                tags_data = json.load(f)

            # tags_data is a dict: {tag: count}
            for tag in tags_data.keys():
                # Normalize tag using the same logic as frontend
                normalized_tag = normalize_tag_for_matching(tag)
                tag_dict[normalized_tag] = category_key

            print(f"[BuildTagData] Loaded {len(tags_data)} tags from {category_name}.json")

        except Exception as e:
            print(f"[BuildTagData] ERROR loading {category_name}.json: {e}")

    # Add special tags (Rating and Quality)
    # These are not in category JSON files but need special handling
    special_tags = {
        # Rating tags
        "general": "Rating",
        "sensitive": "Rating",
        "questionable": "Rating",
        "explicit": "Rating",

        # Quality tags
        "masterpiece": "Quality",
        "best quality": "Quality",  # Normalized form
        "high quality": "Quality",
        "normal quality": "Quality",
        "low quality": "Quality",
        "worst quality": "Quality",
    }

    tag_dict.update(special_tags)

    print(f"[BuildTagData] Total tags loaded: {len(tag_dict)} (including {len(special_tags)} special tags)")
    return tag_dict


def build_tag_data_from_content(content: str, tag_dict: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Build tag_data from comma-separated tags string.

    Args:
        content: Comma-separated tags (e.g., "1girl, long_hair, smile")
        tag_dict: Dictionary mapping normalized_tag -> category

    Returns:
        List of {"tag": "1girl", "category": "General"}
    """
    if not content:
        return []

    tags = [tag.strip() for tag in content.split(",") if tag.strip()]
    tag_data = []

    for tag in tags:
        # Normalize tag for lookup using the same logic as frontend
        normalized_tag = normalize_tag_for_matching(tag)

        # Look up category (return "Unknown" if not found)
        category = tag_dict.get(normalized_tag, "Unknown")

        tag_data.append({
            "tag": tag,  # Use original tag (preserve casing and format)
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
        # Load tag dictionary from JSON files
        tag_dict = load_tag_dictionary_from_json()

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
