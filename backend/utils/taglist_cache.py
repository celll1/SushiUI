"""
Unified taglist cache singleton for high-performance tag operations.

This module provides a centralized caching layer for taglist data, replacing
scattered file loading patterns across the codebase.

Features:
- Singleton pattern (shared across all components)
- Automatic mtime-based invalidation
- O(1) tag categorization lookup
- O(1) prefix search with 2-character index
- Memory-efficient storage (~200MB for 1.5M tags)

Usage:
    from utils.taglist_cache import taglist_cache

    # Get tag category
    category = taglist_cache.get_category("1girl")  # -> "General"

    # Search by prefix
    results = taglist_cache.search_prefix("hats", category="character", limit=20)

    # Get all tags for a category
    tags = taglist_cache.get_category_tags("artist")
"""

import os
import json
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
import threading


class TaglistCache:
    """
    Singleton cache for taglist data with automatic mtime-based invalidation.

    Attributes:
        _cache: Dict[category, Dict[tag, count]] - Full tag data by category
        _category_map: Dict[normalized_tag, category] - Tag -> category lookup
        _prefix_index: Dict[(category, prefix), List[Tuple[tag, count]]] - Prefix search index
        _mtimes: Dict[category, float] - File modification times for invalidation
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._cache: Dict[str, Dict[str, int]] = {}
        self._category_map: Dict[str, str] = {}
        self._prefix_index: Dict[Tuple[str, str], List[Tuple[str, int]]] = {}
        self._mtimes: Dict[str, float] = {}
        self._root_dir: Optional[str] = None
        self._initialized = True

    def initialize(self, root_dir: str):
        """
        Initialize cache with root directory.

        Args:
            root_dir: Root directory of the application (where taglist/ folder is)
        """
        self._root_dir = root_dir
        self._load_all_categories()

    def _normalize_tag(self, tag: str) -> str:
        """
        Normalize tag for matching: lowercase, replace underscores with spaces.

        Args:
            tag: Raw tag string

        Returns:
            Normalized tag string
        """
        return tag.lower().replace('_', ' ').strip()

    def _get_taglist_path(self, category: str) -> str:
        """
        Get file path for a category's taglist JSON.

        Args:
            category: Category name (general, character, artist, etc.)

        Returns:
            Absolute path to JSON file
        """
        if not self._root_dir:
            raise RuntimeError("TaglistCache not initialized. Call initialize(root_dir) first.")

        # Category name to filename mapping
        filename_map = {
            "general": "General.json",
            "character": "Character.json",
            "artist": "Artist.json",
            "copyright": "Copyright.json",
            "meta": "Meta.json",
            "model": "Model.json",
        }

        filename = filename_map.get(category.lower())
        if not filename:
            raise ValueError(f"Unknown category: {category}")

        return os.path.join(self._root_dir, "taglist", filename)

    def _should_reload(self, category: str) -> bool:
        """
        Check if category data should be reloaded based on file mtime.

        Args:
            category: Category name

        Returns:
            True if file has been modified since last load
        """
        try:
            file_path = self._get_taglist_path(category)
            if not os.path.exists(file_path):
                return False

            current_mtime = os.path.getmtime(file_path)
            cached_mtime = self._mtimes.get(category)

            return cached_mtime is None or current_mtime > cached_mtime
        except Exception as e:
            print(f"[TaglistCache] Error checking mtime for {category}: {e}")
            return False

    def _load_category(self, category: str):
        """
        Load a single category's taglist from disk.

        Args:
            category: Category name
        """
        try:
            file_path = self._get_taglist_path(category)

            if not os.path.exists(file_path):
                print(f"[TaglistCache] Category file not found: {file_path}")
                return

            # Load JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                tags_data = json.load(f)

            # Store in cache
            self._cache[category] = tags_data

            # Update mtime
            self._mtimes[category] = os.path.getmtime(file_path)

            # Update category map (normalized tag -> category)
            for tag in tags_data.keys():
                normalized = self._normalize_tag(tag)
                if normalized:
                    self._category_map[normalized] = category.capitalize()

            # Build prefix index (2-character prefixes for fast search)
            self._build_prefix_index(category, tags_data)

            print(f"[TaglistCache] Loaded {len(tags_data)} tags for category '{category}'")

        except Exception as e:
            print(f"[TaglistCache] Error loading category '{category}': {e}")

    def _build_prefix_index(self, category: str, tags_data: Dict[str, int]):
        """
        Build 2-character prefix index for fast autocomplete.

        Args:
            category: Category name
            tags_data: Dict of tag -> count
        """
        # Clear existing index for this category
        keys_to_remove = [k for k in self._prefix_index.keys() if k[0] == category]
        for key in keys_to_remove:
            del self._prefix_index[key]

        # Build new index
        for tag, count in tags_data.items():
            normalized = self._normalize_tag(tag)
            if len(normalized) >= 2:
                prefix = normalized[:2]
                key = (category, prefix)
                if key not in self._prefix_index:
                    self._prefix_index[key] = []
                self._prefix_index[key].append((tag, count))

        # Sort by count (descending) for each prefix
        for key in self._prefix_index:
            if key[0] == category:
                self._prefix_index[key].sort(key=lambda x: x[1], reverse=True)

    def _load_all_categories(self):
        """Load all category taglists from disk."""
        categories = ["general", "character", "artist", "copyright", "meta", "model"]

        for category in categories:
            if self._should_reload(category):
                self._load_category(category)

    def reload_if_needed(self):
        """Check all categories and reload if any files have been modified."""
        self._load_all_categories()

    def get_category(self, tag: str) -> str:
        """
        Get category for a tag (fast O(1) lookup).

        Args:
            tag: Tag string (case-insensitive)

        Returns:
            Category name ("General", "Character", "Artist", etc.)
            Defaults to "General" if tag not found
        """
        if not self._root_dir:
            raise RuntimeError("TaglistCache not initialized. Call initialize(root_dir) first.")

        # Reload if needed
        self.reload_if_needed()

        normalized = self._normalize_tag(tag)
        return self._category_map.get(normalized, "General")

    def get_categories_batch(self, tags: List[str]) -> Dict[str, str]:
        """
        Get categories for multiple tags (batch operation).

        Args:
            tags: List of tag strings

        Returns:
            Dict mapping tag -> category
        """
        if not self._root_dir:
            raise RuntimeError("TaglistCache not initialized. Call initialize(root_dir) first.")

        # Reload if needed
        self.reload_if_needed()

        result = {}
        for tag in tags:
            normalized = self._normalize_tag(tag)
            result[tag] = self._category_map.get(normalized, "General")

        return result

    def search_prefix(
        self,
        prefix: str,
        category: Optional[str] = None,
        limit: int = 20
    ) -> List[Tuple[str, int, str]]:
        """
        Search tags by prefix (fast O(1) lookup with prefix index).

        Args:
            prefix: Prefix string (case-insensitive, minimum 2 characters)
            category: Category filter (None = search all categories)
            limit: Maximum number of results

        Returns:
            List of (tag, count, category) tuples, sorted by count descending
        """
        if not self._root_dir:
            raise RuntimeError("TaglistCache not initialized. Call initialize(root_dir) first.")

        # Reload if needed
        self.reload_if_needed()

        if len(prefix) < 2:
            return []

        normalized_prefix = self._normalize_tag(prefix)[:2]

        results = []

        if category:
            # Search single category
            key = (category.lower(), normalized_prefix)
            candidates = self._prefix_index.get(key, [])

            for tag, count in candidates:
                if self._normalize_tag(tag).startswith(self._normalize_tag(prefix)):
                    results.append((tag, count, category.capitalize()))
        else:
            # Search all categories
            categories = ["general", "character", "artist", "copyright", "meta", "model"]
            for cat in categories:
                key = (cat, normalized_prefix)
                candidates = self._prefix_index.get(key, [])

                for tag, count in candidates:
                    if self._normalize_tag(tag).startswith(self._normalize_tag(prefix)):
                        results.append((tag, count, cat.capitalize()))

        # Sort by count descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:limit]

    def get_category_tags(self, category: str) -> Dict[str, int]:
        """
        Get all tags for a category.

        Args:
            category: Category name

        Returns:
            Dict of tag -> count for the category
        """
        if not self._root_dir:
            raise RuntimeError("TaglistCache not initialized. Call initialize(root_dir) first.")

        # Reload if needed
        if self._should_reload(category.lower()):
            self._load_category(category.lower())

        return self._cache.get(category.lower(), {})

    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics for all categories.

        Returns:
            Dict of category -> tag count
        """
        if not self._root_dir:
            raise RuntimeError("TaglistCache not initialized. Call initialize(root_dir) first.")

        # Reload if needed
        self.reload_if_needed()

        return {
            cat.capitalize(): len(tags)
            for cat, tags in self._cache.items()
        }


# Global singleton instance
taglist_cache = TaglistCache()
