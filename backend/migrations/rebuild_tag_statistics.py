"""
Rebuild tag_statistics for existing datasets to include category information
"""
import sqlite3
import json
import sys
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def rebuild_tag_statistics(db_path: str):
    """
    Rebuild tag_statistics for all datasets to include category information.
    """
    print(f"[RebuildStats] Connecting to {db_path}")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        # Get all datasets
        cursor.execute("SELECT id, name FROM datasets")
        datasets = cursor.fetchall()

        print(f"[RebuildStats] Found {len(datasets)} datasets")

        for dataset_id, dataset_name in datasets:
            print(f"\n[RebuildStats] Processing dataset: {dataset_name} (ID: {dataset_id})")

            # Count tag occurrences and extract categories from tag_data
            tag_counts = {}
            tag_categories = {}

            # Get all captions for this dataset
            cursor.execute("""
                SELECT dc.tag_data, dc.content
                FROM dataset_captions dc
                JOIN dataset_items di ON dc.item_id = di.id
                WHERE di.dataset_id = ?
                  AND dc.caption_type = 'tags'
            """, (dataset_id,))

            captions = cursor.fetchall()
            print(f"[RebuildStats] Processing {len(captions)} captions...")

            for tag_data_json, content in captions:
                # Extract categories from tag_data if available
                if tag_data_json:
                    try:
                        tag_data = json.loads(tag_data_json)
                        for item in tag_data:
                            tag = item.get("tag", "").strip()
                            category = item.get("category", "Unknown")
                            if tag:
                                tag_counts[tag] = tag_counts.get(tag, 0) + 1
                                # Store category (first occurrence wins)
                                if tag not in tag_categories:
                                    tag_categories[tag] = category
                    except:
                        # Fallback: parse from content (no category info)
                        if content:
                            tags = content.split(",")
                            for tag in tags:
                                tag = tag.strip()
                                if tag:
                                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
                                    if tag not in tag_categories:
                                        tag_categories[tag] = "Unknown"
                else:
                    # No tag_data: parse from content (no category info)
                    if content:
                        tags = content.split(",")
                        for tag in tags:
                            tag = tag.strip()
                            if tag:
                                tag_counts[tag] = tag_counts.get(tag, 0) + 1
                                if tag not in tag_categories:
                                    tag_categories[tag] = "Unknown"

            # Build statistics
            statistics = {}
            for tag, count in tag_counts.items():
                statistics[tag] = {
                    "count": count,
                    "category": tag_categories.get(tag, "Unknown")
                }

            # Update database
            statistics_json = json.dumps(statistics)
            cursor.execute("""
                UPDATE datasets
                SET tag_statistics = ?
                WHERE id = ?
            """, (statistics_json, dataset_id))

            conn.commit()

            print(f"[RebuildStats] Updated {len(statistics)} tags for dataset '{dataset_name}'")

            # Show sample
            sample_tags = list(statistics.items())[:5]
            print(f"[RebuildStats] Sample tags:")
            for tag, stats in sample_tags:
                print(f"  {tag}: count={stats['count']}, category={stats['category']}")

        print(f"\n[RebuildStats] Completed rebuilding statistics for {len(datasets)} datasets")

    except Exception as e:
        print(f"[RebuildStats] ERROR: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    # Default database path
    db_path = Path(__file__).parent.parent.parent / "datasets.db"

    if len(sys.argv) > 1:
        db_path = Path(sys.argv[1])

    if not db_path.exists():
        print(f"❌ ERROR: Database not found at {db_path}")
        sys.exit(1)

    rebuild_tag_statistics(str(db_path))
