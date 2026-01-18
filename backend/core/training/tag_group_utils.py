"""
Tag group utilities for caption processing.

Supports:
- Loading tag groups from JSON files (Character, General, Copyright, etc.)
- Tag categorization
- Tag group-based shuffle
- Per-category dropout rates
- Tag normalization (handle various escape patterns)

MIGRATED TO USE TaglistCache singleton (Phase 3):
- Replaces direct JSON file loading with server-side cache
- Eliminates repeated 50MB file reads during training
- Automatic mtime-based cache invalidation
"""
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Set, Optional
from utils.taglist_cache import taglist_cache


# Person count tags (for exclusion from General group shuffle/dropout)
PERSON_COUNT_TAGS = {
    'no_humans', 'no humans',
    'solo',
    'group',
    'still_life', 'still life',
    'multiple_girls', 'multiple girls',
    'multiple_boys', 'multiple boys',
    'multiple_others', 'multiple others',
    'solo_focus', 'solo focus',
    'male_focus', 'male focus',
    'other_focus', 'other focus',
    '1girl', '2girls', '3girls', '4girls', '5girls', '6+girls',
    '1boy', '2boys', '3boys', '4boys', '5boys', '6+boys',
    '1other', '2others', '3others', '4others', '5others', '6+others',
}


def normalize_tag_for_matching(tag: str) -> str:
    """
    Normalize tag for matching purposes only (does not modify the original tag).

    Handles various escape patterns:
    - djibril_(makai_tenshi_djibril)  → djibril (makai tenshi djibril)
    - djibril (makai tenshi djibril)  → djibril (makai tenshi djibril)
    - djibril \\(makai tenshi djibril\\) → djibril (makai tenshi djibril)
    - djibril_\\(makai_tenshi_djibril\\) → djibril (makai tenshi djibril)

    Args:
        tag: Tag string

    Returns:
        Normalized tag for matching (lowercase, standardized format)
    """
    normalized = tag.strip()

    # Remove excessive escaping: \\ → nothing
    normalized = normalized.replace('\\\\', '')
    normalized = normalized.replace('\\', '')

    # Normalize underscores to spaces
    normalized = normalized.replace('_', ' ')

    # Lowercase for matching
    normalized = normalized.lower()

    return normalized


def normalize_tag_for_output(tag: str) -> str:
    """
    Normalize tag for output (standardize to escaped parentheses format).

    Target format: "djibril \\(makai tenshi djibril\\)"

    Patterns to handle:
    - djibril_(makai_tenshi_djibril) → djibril \\(makai tenshi djibril\\)
    - djibril (makai tenshi djibril) → djibril \\(makai tenshi djibril\\)
    - djibril \\(makai tenshi djibril\\) → djibril \\(makai tenshi djibril\\) (keep)
    - djibril_\\(makai_tenshi_djibril\\) → djibril \\(makai tenshi djibril\\)

    Args:
        tag: Tag string

    Returns:
        Normalized tag for output
    """
    normalized = tag.strip()

    # Remove excessive escaping first
    normalized = normalized.replace('\\\\', '\\')

    # Check if tag contains parentheses
    if '(' in normalized or ')' in normalized:
        # Remove existing backslashes before parentheses
        normalized = normalized.replace('\\(', '(')
        normalized = normalized.replace('\\)', ')')

        # Replace underscores with spaces (but only inside parentheses)
        # Pattern: text_(content) → text (content)
        # Pattern: text_\(content\) → text (content)

        # First, extract parts before and after parentheses
        match = re.match(r'^([^(]+)\((.+)\)$', normalized)
        if match:
            prefix = match.group(1).strip()
            content = match.group(2).strip()

            # Replace underscores with spaces
            prefix = prefix.replace('_', ' ')
            content = content.replace('_', ' ')

            # Rebuild with escaped parentheses
            normalized = f"{prefix} \\({content}\\)"
        else:
            # If pattern doesn't match, just replace underscores and escape parentheses
            normalized = normalized.replace('_', ' ')
            normalized = normalized.replace('(', '\\(')
            normalized = normalized.replace(')', '\\)')
    else:
        # No parentheses: just replace underscores with spaces (no escape needed)
        normalized = normalized.replace('_', ' ')

    return normalized


class TagGroupManager:
    """
    Manage tag groups for caption processing.

    MIGRATED TO USE TaglistCache singleton (Phase 3):
    - Uses shared cache instead of per-instance loading
    - Eliminates 50MB file reads on every TagGroupManager instantiation
    """

    def __init__(self, tag_group_dir: str = "taglist"):
        """
        Initialize tag group manager.

        Args:
            tag_group_dir: Directory containing tag group JSON files
        """
        tag_path = Path(tag_group_dir)

        # If relative path, resolve from project root (parent of backend)
        if not tag_path.is_absolute():
            # Get project root
            # __file__ = backend/core/training/tag_group_utils.py
            # .parent = backend/core/training
            # .parent.parent = backend/core
            # .parent.parent.parent = backend
            # .parent.parent.parent.parent = project root
            project_root = Path(__file__).parent.parent.parent.parent
            tag_path = project_root / tag_group_dir

        self.tag_group_dir = tag_path
        self.tag_groups: Dict[str, Set[str]] = {}
        self._normalized_rating_quality: Set[str] = set()  # Fast O(1) lookup for Rating/Quality

        # Initialize TaglistCache (will use singleton if already initialized)
        taglist_cache.initialize(str(project_root))

        self.load_tag_groups()

    def load_tag_groups(self):
        """
        Load tag groups using TaglistCache singleton.

        MIGRATED: Uses shared cache instead of per-instance JSON file loading.
        """
        print(f"[TagGroupManager] Loading tag groups via TaglistCache (no file reads)")

        # Add hardcoded Rating and Quality tags (these don't have JSON files in taglist)
        rating_tags = {
            'general', 'sensitive', 'questionable', 'explicit',
            'rating:general', 'rating:sensitive', 'rating:questionable', 'rating:explicit'
        }
        quality_tags = {
            'best quality', 'high quality', 'great quality', 'normal quality',
            'low quality', 'worst quality', 'masterpiece', 'amazing quality'
        }

        self.tag_groups['Rating'] = rating_tags
        self.tag_groups['Quality'] = quality_tags

        # Build normalized Rating/Quality set for O(1) lookup in get_tag_group()
        for tag in rating_tags:
            self._normalized_rating_quality.add(self._normalize_tag(tag))
        for tag in quality_tags:
            self._normalized_rating_quality.add(self._normalize_tag(tag))

        print(f"[TagGroupManager] Added hardcoded Rating ({len(rating_tags)} tags) and Quality ({len(quality_tags)} tags)")

        # Load other categories from TaglistCache
        categories = ["general", "character", "artist", "copyright", "meta", "model"]
        total_tags = 0

        for category in categories:
            category_tags_dict = taglist_cache.get_category_tags(category)
            tags = set(category_tags_dict.keys())
            self.tag_groups[category.capitalize()] = tags
            total_tags += len(tags)
            print(f"[TagGroupManager] Loaded {len(tags)} tags for group '{category.capitalize()}' (via cache)")

        # Add stats from cache
        cache_stats = taglist_cache.get_stats()
        print(f"[TagGroupManager] Total loaded: {len(self.tag_groups)} tag groups, {total_tags} tags from cache")
        print(f"[TagGroupManager] Cache stats: {cache_stats}")

    def _normalize_tag(self, tag: str) -> str:
        """
        Normalize tag for comparison.

        Args:
            tag: Tag string

        Returns:
            Normalized tag (lowercase, standardized format)
        """
        return normalize_tag_for_matching(tag)

    def get_tag_group(self, tag: str) -> Optional[str]:
        """
        Get group name for a tag using TaglistCache.

        PERFORMANCE CRITICAL: Called millions of times during training.
        Optimized to O(1) lookups only.

        Args:
            tag: Tag string

        Returns:
            Group name or None if not found
        """
        normalized = self._normalize_tag(tag)

        # O(1) check for Rating/Quality (hardcoded tags not in TaglistCache)
        if normalized in self._normalized_rating_quality:
            # Determine if Rating or Quality
            for rating_tag in self.tag_groups['Rating']:
                if self._normalize_tag(rating_tag) == normalized:
                    return 'Rating'
            return 'Quality'

        # O(1) lookup in TaglistCache's category_map (no reload_if_needed overhead)
        category = taglist_cache._category_map.get(normalized)
        if category:
            return category  # Already capitalized (e.g., "General", "Character")

        # Not found in either Rating/Quality or TaglistCache
        return None

    def is_person_count_tag(self, tag: str) -> bool:
        """
        Check if tag is a person count tag.

        Args:
            tag: Tag string

        Returns:
            True if tag is a person count tag
        """
        normalized = self._normalize_tag(tag)
        return normalized in PERSON_COUNT_TAGS

    def categorize_tags(self, tags: List[str]) -> Dict[str, List[str]]:
        """
        Categorize tags by group.

        Args:
            tags: List of tags

        Returns:
            Dict mapping group name to list of tags
        """
        categorized = {}
        for tag in tags:
            group = self.get_tag_group(tag)
            if group is None:
                group = "Unknown"

            if group not in categorized:
                categorized[group] = []
            categorized[group].append(tag)

        return categorized

    def shuffle_by_groups(
        self,
        tokens: List[str],
        groups_to_shuffle: List[str],
        keep_first_n: int = 0,
        exclude_person_count: bool = False,
        shuffle_together: bool = False,
        rng: Optional[random.Random] = None,
    ) -> List[str]:
        """
        Shuffle tokens by tag groups.

        Args:
            tokens: List of tokens (comma-separated tags)
            groups_to_shuffle: List of group names to shuffle (e.g., ["Character", "General"])
            keep_first_n: Number of first tokens to keep unshuffled
            exclude_person_count: Exclude person count tags from General group shuffling
            shuffle_together: Shuffle all selected groups together (vs within each group)
            rng: Random number generator (for reproducibility)

        Returns:
            Shuffled token list
        """
        if rng is None:
            rng = random.Random()

        if not groups_to_shuffle or not tokens or len(tokens) <= keep_first_n:
            return tokens

        # Split into fixed and shuffleable parts
        fixed_tokens = tokens[:keep_first_n]
        working_tokens = tokens[keep_first_n:]

        # Categorize tokens by group
        categorized = {}
        person_count_tokens = []  # Person count tags (1girl, 2boys, etc.)
        non_shuffleable = []

        for token in working_tokens:
            tag_stripped = token.strip()
            if not tag_stripped:
                non_shuffleable.append(token)
                continue

            group = self.get_tag_group(tag_stripped)

            # Check if this is a person count tag (should be excluded from shuffle)
            if exclude_person_count and group == "General" and self.is_person_count_tag(tag_stripped):
                person_count_tokens.append(token)
                continue

            # Check if this tag should be shuffled
            should_shuffle = group in groups_to_shuffle

            if should_shuffle:
                if group not in categorized:
                    categorized[group] = []
                categorized[group].append(token)
            else:
                non_shuffleable.append(token)

        # No tags to shuffle
        if not categorized and not person_count_tokens:
            return tokens

        # Shuffle
        if shuffle_together:
            # Shuffle all selected groups together
            all_shuffleable = []
            for group_tokens in categorized.values():
                all_shuffleable.extend(group_tokens)
            rng.shuffle(all_shuffleable)
            return fixed_tokens + all_shuffleable + non_shuffleable
        else:
            # Shuffle within each group, and insert person count tags at the start of General group
            shuffled_parts = []
            person_count_inserted = False

            for group in groups_to_shuffle:
                # If this is the General group, prepend person count tags (shuffled)
                if group == "General" and person_count_tokens and not person_count_inserted:
                    rng.shuffle(person_count_tokens)  # Shuffle person count tags among themselves
                    shuffled_parts.extend(person_count_tokens)
                    person_count_inserted = True

                if group in categorized:
                    group_tokens = categorized[group]
                    rng.shuffle(group_tokens)
                    shuffled_parts.extend(group_tokens)

            # If General group was not in shuffle_tag_groups, append person count tags at the end
            if person_count_tokens and not person_count_inserted:
                rng.shuffle(person_count_tokens)  # Shuffle person count tags among themselves
                shuffled_parts.extend(person_count_tokens)

            return fixed_tokens + shuffled_parts + non_shuffleable


# Global cache for tag group managers
_tag_group_manager_cache: Dict[str, TagGroupManager] = {}


def get_tag_group_manager(tag_group_dir: str = "taglist") -> TagGroupManager:
    """
    Get or create tag group manager (cached).

    Args:
        tag_group_dir: Directory containing tag group JSON files

    Returns:
        TagGroupManager instance
    """
    if tag_group_dir not in _tag_group_manager_cache:
        _tag_group_manager_cache[tag_group_dir] = TagGroupManager(tag_group_dir)

    return _tag_group_manager_cache[tag_group_dir]
