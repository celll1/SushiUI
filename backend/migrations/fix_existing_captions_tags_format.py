"""
Fix existing captions to have correct is_tags_format value.

This migration:
1. Sets caption_type="tags" to is_tags_format=True (Danbooru tags)
2. Sets caption_type="tags_ordered" to is_tags_format=True (legacy pre-ordered tags)
3. Sets metadata fields (author, savedAt, metrics.*) to field_category="metadata"
4. Sets natural language fields (text, etc.) to is_tags_format=False
"""

import sqlite3
import os

def migrate():
    """Fix existing captions in datasets.db"""
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "datasets.db")

    if not os.path.exists(db_path):
        print(f"[Migration] Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        print("[Migration] Fixing existing captions format detection...")

        # 1. Set "tags" and "tags_ordered" to tags format
        cursor.execute('''
            UPDATE dataset_captions
            SET is_tags_format = 1, field_category = 'training'
            WHERE caption_type IN ('tags', 'tags_ordered')
        ''')
        tags_updated = cursor.rowcount
        print(f"[Migration] Set {tags_updated} tags/tags_ordered captions to tags format")

        # 2. Set metadata fields to field_category="metadata"
        metadata_patterns = [
            'author', 'savedAt', 'timestamp', 'created', 'updated', 'published',
            'metrics.%', 'stats.%', 'source', 'url', 'link', 'id'
        ]

        for pattern in metadata_patterns:
            cursor.execute(f'''
                UPDATE dataset_captions
                SET field_category = 'metadata', is_tags_format = 0
                WHERE caption_type LIKE ? AND field_category != 'metadata'
            ''', (pattern,))

        cursor.execute('''
            SELECT COUNT(*) FROM dataset_captions WHERE field_category = 'metadata'
        ''')
        metadata_count = cursor.fetchone()[0]
        print(f"[Migration] Set {metadata_count} captions to metadata category")

        # 3. Set natural language fields (text, description, etc.) to is_tags_format=False
        nl_types = ['text', 'description', 'caption']
        for nl_type in nl_types:
            cursor.execute('''
                UPDATE dataset_captions
                SET is_tags_format = 0, field_category = 'training'
                WHERE caption_type = ? AND caption_type NOT IN ('tags', 'tags_ordered')
            ''', (nl_type,))

        # 4. Show final statistics
        cursor.execute('''
            SELECT caption_type, field_category, is_tags_format, COUNT(*)
            FROM dataset_captions
            GROUP BY caption_type, field_category, is_tags_format
            ORDER BY caption_type
        ''')

        print("\n[Migration] Final statistics:")
        print(f"{'Caption Type':<20} | {'Category':<10} | {'Tags Format':<12} | Count")
        print("-" * 70)
        for row in cursor.fetchall():
            caption_type, category, is_tags, count = row
            print(f"{caption_type:<20} | {category:<10} | {bool(is_tags)!s:<12} | {count}")

        conn.commit()
        print("\n[Migration] Successfully fixed existing captions")

    except Exception as e:
        print(f"[Migration] Error: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
