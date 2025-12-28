"""
Taglist loader for caption format detection.

MIGRATED TO USE TaglistCache singleton (Phase 3):
- Uses server-side cache instead of repeated file reads
- Automatic mtime-based invalidation
- Consistent tag normalization
"""

from typing import Set
from utils.taglist_cache import taglist_cache


def load_all_tags(root_dir: str) -> Set[str]:
    """
    Load all tags from taglist JSON files using TaglistCache singleton.

    Args:
        root_dir: Root directory of the application (where taglist/ folder is)

    Returns:
        Set of all tags (lowercase, normalized)
    """
    # Initialize cache with root directory
    taglist_cache.initialize(root_dir)

    # Collect all tags from all categories
    all_tags = set()

    categories = ["general", "character", "artist", "copyright", "meta", "model"]

    for category in categories:
        category_tags = taglist_cache.get_category_tags(category)

        # Normalize tags (cache already stores normalized keys in category_map)
        for tag in category_tags.keys():
            normalized = taglist_cache._normalize_tag(tag)
            if normalized:
                all_tags.add(normalized)

    print(f"[TaglistLoader] Loaded {len(all_tags)} tags from taglist (via TaglistCache)")
    return all_tags
