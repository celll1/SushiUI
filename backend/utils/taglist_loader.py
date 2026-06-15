"""
Taglist loader for caption format detection.

MIGRATED TO USE TaglistCache singleton (Phase 3):
- Uses server-side cache instead of repeated file reads
- Automatic mtime-based invalidation
- Consistent tag normalization
"""

from typing import Set
from utils.taglist_cache import taglist_cache


def load_all_tags(root_dir: str, include_gelbooru: bool = True) -> Set[str]:
    """
    Load all known tags (normalized, space form) for caption format detection.

    Includes the Danbooru taglist, the Gelbooru supplement (when present), and
    the deprecated-alias keys — so a sidecar dominated by gelbooru tags or
    deprecated tag spellings (e.g. "twin_tails", "hand_on_hip") still scores a
    high tag-match rate and is correctly recognised as tags format rather than
    being misread as natural language.

    Args:
        root_dir: Root directory of the application (where taglist/ folder is)
        include_gelbooru: also recognise Gelbooru-supplement tags (graceful if the
                          taglist_gel/ directory is absent)

    Returns:
        Set of all known tags (lowercase, space form)
    """
    # Initialize cache (loads Gelbooru supplement + alias table when available).
    taglist_cache.initialize(root_dir, enable_gelbooru=include_gelbooru)

    # _category_map already holds normalized keys for Danbooru + (optionally)
    # Gelbooru tags. Use it directly so the supplement is included.
    all_tags = set(taglist_cache._category_map.keys())

    # Add deprecated-alias keys (the spellings a tagger may emit) in space form.
    for danbooru_key in getattr(taglist_cache, "_aliases", {}):
        all_tags.add(danbooru_key.lower().replace("_", " "))

    print(f"[TaglistLoader] Loaded {len(all_tags)} tags for format detection "
          f"(gelbooru={'on' if include_gelbooru else 'off'}, via TaglistCache)")
    return all_tags
