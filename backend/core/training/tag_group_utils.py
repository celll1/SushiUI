"""
Tag group utilities for caption processing.

Supports:
- Loading tag groups from JSON files (Character, General, Copyright, etc.)
- Tag categorization
- Tag group-based shuffle
- Per-category dropout rates
- Tag normalization (handle various escape patterns)
"""
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Set, Optional


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
    """Manage tag groups for caption processing."""

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
        self._tag_to_group_cache: Dict[str, str] = {}
        self.load_tag_groups()

    def load_tag_groups(self):
        """Load tag groups from JSON files."""
        print(f"[TagGroupManager] Attempting to load tag groups from: {self.tag_group_dir.absolute()}")

        # Add hardcoded Rating and Quality tags (these don't have JSON files)
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

        # Build cache for Rating and Quality
        for tag in rating_tags:
            self._tag_to_group_cache[self._normalize_tag(tag)] = 'Rating'
        for tag in quality_tags:
            self._tag_to_group_cache[self._normalize_tag(tag)] = 'Quality'

        print(f"[TagGroupManager] Added hardcoded Rating ({len(rating_tags)} tags) and Quality ({len(quality_tags)} tags)")

        if not self.tag_group_dir.exists():
            print(f"[TagGroupManager] ERROR: Tag group directory not found: {self.tag_group_dir.absolute()}")
            print(f"[TagGroupManager] Category ordering will not work without tag group files!")
            return

        json_files = list(self.tag_group_dir.glob("*.json"))
        print(f"[TagGroupManager] Found {len(json_files)} JSON files")

        for json_file in json_files:
            group_name = json_file.stem
            print(f"[TagGroupManager] Loading group '{group_name}' from {json_file.name}")
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # JSON format: {"tag_name": count, ...}
                    tags = set(data.keys())
                    self.tag_groups[group_name] = tags

                    # Build cache: tag -> group (skip if already cached to preserve priority)
                    for tag in tags:
                        normalized_tag = self._normalize_tag(tag)
                        # Only add if not already in cache (Rating/Quality have priority)
                        if normalized_tag not in self._tag_to_group_cache:
                            self._tag_to_group_cache[normalized_tag] = group_name

                print(f"[TagGroupManager] Loaded {len(tags)} tags for group '{group_name}'")
            except Exception as e:
                print(f"[TagGroupManager] Failed to load {json_file}: {e}")

        print(f"[TagGroupManager] Total loaded: {len(self.tag_groups)} tag groups, {len(self._tag_to_group_cache)} tags in cache")
        if len(self.tag_groups) == 0:
            print(f"[TagGroupManager] WARNING: No tag groups loaded! Category ordering will not work!")

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
        Get group name for a tag.

        Args:
            tag: Tag string

        Returns:
            Group name or None if not found
        """
        normalized = self._normalize_tag(tag)
        return self._tag_to_group_cache.get(normalized)

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
        non_shuffleable = []

        for token in working_tokens:
            tag_stripped = token.strip()
            if not tag_stripped:
                non_shuffleable.append(token)
                continue

            group = self.get_tag_group(tag_stripped)

            # Check if this tag should be shuffled
            should_shuffle = group in groups_to_shuffle
            if should_shuffle and exclude_person_count and group == "General":
                should_shuffle = not self.is_person_count_tag(tag_stripped)

            if should_shuffle:
                if group not in categorized:
                    categorized[group] = []
                categorized[group].append(token)
            else:
                non_shuffleable.append(token)

        # No tags to shuffle
        if not categorized:
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
            # Shuffle within each group, then reconstruct
            shuffled_parts = []
            for group in groups_to_shuffle:
                if group in categorized:
                    group_tokens = categorized[group]
                    rng.shuffle(group_tokens)
                    shuffled_parts.extend(group_tokens)

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
