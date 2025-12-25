"""
Taglist loader for caption format detection.
"""

import os
import json
from typing import Set


def load_all_tags(root_dir: str) -> Set[str]:
    """
    Load all tags from taglist JSON files.

    Args:
        root_dir: Root directory of the application (where taglist/ folder is)

    Returns:
        Set of all tags (lowercase, normalized)
    """
    taglist_dir = os.path.join(root_dir, "taglist")

    if not os.path.exists(taglist_dir):
        print(f"[TaglistLoader] Warning: taglist directory not found: {taglist_dir}")
        return set()

    all_tags = set()

    # Taglist files
    taglist_files = [
        "artist.json",
        "character.json",
        "copyright.json",
        "general.json",
        "meta.json",
        # Quality/Rating are not tags in Danbooru sense, skip them
    ]

    for filename in taglist_files:
        file_path = os.path.join(taglist_dir, filename)

        if not os.path.exists(file_path):
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tags = json.load(f)

            # Normalize tags (lowercase, strip)
            for tag in tags:
                normalized = tag.lower().strip()
                if normalized:
                    all_tags.add(normalized)

        except Exception as e:
            print(f"[TaglistLoader] Error loading {filename}: {e}")
            continue

    print(f"[TaglistLoader] Loaded {len(all_tags)} tags from taglist")
    return all_tags
