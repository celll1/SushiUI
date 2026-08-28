"""
Caption processing utilities for dataset captions during training.

Supports:
- Category ordering (reorder tags by category)
- Caption dropout
- Token dropout
- Token shuffle (per-epoch or random, with tag group support)
- Tag-level dropout with category-specific rates
"""
import random
import hashlib
from typing import List, Dict, Optional
from pathlib import Path

# Debug flag: log reordered tokens only once
_logged_reordered_tokens = False


def _caption_rng(item_path: str, epoch_num: int, operation: str):
    """Return an isolated, stable RNG for one item/epoch operation."""
    seed_str = f"{item_path}_{operation}_epoch{epoch_num}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
    return random.Random(seed)


def apply_caption_dropout(caption: str, caption_dropout_rate: float = 0.0) -> str:
    """Apply whole-caption dropout without interpreting the caption format."""
    if caption_dropout_rate > 0 and random.random() < caption_dropout_rate:
        return ""
    return caption


def process_caption(
    caption: str,
    epoch_num: int = 0,
    item_path: str = "",
    # Tag normalization (standardize tag format)
    normalize_tags: bool = True,  # Normalize tags to standard format (default: True)
    # Category ordering (processed first, before all other operations)
    category_order: Optional[List[str]] = None,  # Order of categories (e.g., ["Rating", "Quality", "Character", ...])
    # Caption dropout
    caption_dropout_rate: float = 0.0,
    # Token dropout
    token_dropout_rate: float = 0.0,
    keep_tokens: int = 0,
    # Token shuffle
    shuffle_tokens: bool = False,
    shuffle_per_epoch: bool = False,
    shuffle_keep_first_n: int = 0,
    shuffle_tag_groups: Optional[List[str]] = None,  # Tag groups to shuffle (e.g., ["Character", "General"])
    shuffle_groups_together: bool = False,  # Shuffle all groups together vs within each group
    tag_group_dir: str = "taglist",  # Directory containing tag group JSON files
    exclude_person_count_from_shuffle: bool = False,  # Exclude person count tags from General shuffle
    # Tag dropout
    tag_dropout_rate: float = 0.0,
    tag_dropout_per_epoch: bool = False,
    tag_dropout_keep_first_n: int = 0,
    tag_dropout_category_rates: Optional[Dict[str, float]] = None,
    tag_dropout_exclude_person_count: bool = False,
) -> str:
    """
    Process a caption with tag normalization, category ordering, dropout, and shuffle operations.

    Processing order:
    1. Category ordering (reorder tags by category)
    2. Caption dropout
    3. Token dropout
    4. Tag dropout
    5. Token shuffle
    6. Tag normalization (if enabled)

    Args:
        caption: Raw caption string (comma-separated tokens)
        epoch_num: Current epoch number (for per-epoch consistency)
        item_path: Path to the dataset item (for per-epoch consistency)
        normalize_tags: Normalize tags to standard format (default: True)
                        Target: "tag_name \\(qualifier\\)" for tags with parentheses
        category_order: Order of categories (e.g., ["Rating", "Quality", "Character", ...])
        caption_dropout_rate: Probability to drop entire caption (0.0-1.0)
        token_dropout_rate: Probability to drop each token (0.0-1.0)
        keep_tokens: Number of first tokens to always keep (immune to token dropout)
        shuffle_tokens: Whether to shuffle tokens
        shuffle_per_epoch: If True, shuffle is consistent per epoch (reproducible)
        shuffle_keep_first_n: Number of first tokens to keep unshuffled
        tag_dropout_rate: Tag-level dropout probability (0.0-1.0)
        tag_dropout_per_epoch: If True, tag dropout is consistent per epoch
        tag_dropout_keep_first_n: Number of first tags to keep (immune to tag dropout)
        tag_dropout_category_rates: Per-category dropout rates (e.g., {"character": 0.1})
        tag_dropout_exclude_person_count: Exclude person count tags (1girl, 2boys, etc.) from dropout

    Returns:
        Processed caption string
    """
    if not caption or not caption.strip():
        return ""

    # Split into tokens
    token_list = [t.strip() for t in caption.split(',') if t.strip()]

    if not token_list:
        return ""

    # Step 1: Category ordering (reorder tags by category)
    # This is done FIRST, before any dropout or shuffle
    if category_order and len(category_order) > 0:
        from core.training.tag_group_utils import get_tag_group_manager
        # enable_gelbooru=True for training to reduce "Unknown" tags
        tag_manager = get_tag_group_manager(tag_group_dir, enable_gelbooru=True)

        # Group tokens by category
        categorized: Dict[str, List[str]] = {}
        unknown_tags: List[str] = []

        for token in token_list:
            tag_group = tag_manager.get_tag_group(token)
            if tag_group:
                if tag_group not in categorized:
                    categorized[tag_group] = []
                categorized[tag_group].append(token)
            else:
                unknown_tags.append(token)

        # Debug: Log categorization results (once)
        global _logged_reordered_tokens
        if not _logged_reordered_tokens:
            print(f"[CaptionProcessor] Categorized groups found: {list(categorized.keys())}")
            print(f"[CaptionProcessor] Category order specified: {category_order}")
            for cat in category_order:
                if cat in categorized:
                    print(f"  {cat}: {categorized[cat][:3]}")

        # Rebuild token_list in category order
        reordered_tokens = []
        for category in category_order:
            if category in categorized:
                reordered_tokens.extend(categorized[category])

        # Add unknown tags at the end
        reordered_tokens.extend(unknown_tags)

        # Debug: Log reordered tokens only once
        if not _logged_reordered_tokens:
            print(f"[CaptionProcessor] Example reordered tokens: {reordered_tokens[:10]}")
            _logged_reordered_tokens = True

        token_list = reordered_tokens

    # Step 2: Caption dropout (全キャプションをドロップ)
    if not apply_caption_dropout(caption, caption_dropout_rate):
        return ""

    # Token dropout (個別トークンをドロップ)
    if token_dropout_rate > 0:
        new_token_list = []
        for idx, token in enumerate(token_list):
            # keep_tokens 以内のトークンは常に保持
            if idx < keep_tokens:
                new_token_list.append(token)
            elif token_dropout_rate >= 1.0:
                # 100%ドロップアウト
                pass
            else:
                # 確率的にドロップアウト
                if random.random() > token_dropout_rate:
                    new_token_list.append(token)
        token_list = new_token_list

    # Tag-level dropout (タグ単位でのドロップアウト)
    if tag_dropout_rate > 0 or tag_dropout_category_rates:
        # Initialize tag manager if category-specific rates are provided
        tag_manager = None
        if tag_dropout_category_rates:
            from core.training.tag_group_utils import get_tag_group_manager
            # enable_gelbooru=True for training to reduce "Unknown" tags
            tag_manager = get_tag_group_manager(tag_group_dir, enable_gelbooru=True)

        new_token_list = []
        for idx, token in enumerate(token_list):
            # tag_dropout_keep_first_n 以内のタグは常に保持
            if idx < tag_dropout_keep_first_n:
                new_token_list.append(token)
                continue

            # 人数タグ（1girl, 2boys など）を除外
            if tag_dropout_exclude_person_count:
                if tag_manager:
                    if tag_manager.is_person_count_tag(token):
                        new_token_list.append(token)
                        continue
                elif _is_person_count_tag(token):
                    new_token_list.append(token)
                    continue

            # カテゴリ別のドロップアウト確率を決定
            dropout_rate = tag_dropout_rate

            # カテゴリ別のドロップアウト確率を適用
            if tag_manager and tag_dropout_category_rates:
                tag_group = tag_manager.get_tag_group(token)
                if tag_group and tag_group in tag_dropout_category_rates:
                    dropout_rate = tag_dropout_category_rates[tag_group]

            # エポックごとに一貫したドロップアウト
            if tag_dropout_per_epoch:
                random_gen = _caption_rng(item_path, epoch_num, f"tag_dropout_{token}")
                rand = random_gen.random()
            else:
                rand = random.random()

            # ドロップアウト確率に基づいて保持/削除
            if rand > dropout_rate:
                new_token_list.append(token)

        token_list = new_token_list

    # Token shuffle (トークンをシャッフル)
    if shuffle_tokens and len(token_list) > 1:
        # Tag group-based shuffle
        if shuffle_tag_groups:
            from core.training.tag_group_utils import get_tag_group_manager

            # enable_gelbooru=True for training to reduce "Unknown" tags
            tag_manager = get_tag_group_manager(tag_group_dir, enable_gelbooru=True)

            if shuffle_per_epoch:
                # エポックごとに一貫したシャッフル（再現性あり）
                rng = _caption_rng(item_path, epoch_num, "shuffle")
            else:
                # 完全ランダムシャッフル
                rng = random.Random()

            token_list = tag_manager.shuffle_by_groups(
                tokens=token_list,
                groups_to_shuffle=shuffle_tag_groups,
                keep_first_n=shuffle_keep_first_n,
                exclude_person_count=exclude_person_count_from_shuffle,
                shuffle_together=shuffle_groups_together,
                rng=rng,
            )
        else:
            # Simple shuffle (all tokens)
            keep_first_n = shuffle_keep_first_n
            fixed_tokens = token_list[:keep_first_n]
            shuffleable_tokens = token_list[keep_first_n:]

            if shuffleable_tokens:
                if shuffle_per_epoch:
                    # エポックごとに一貫したシャッフル（再現性あり）
                    rng = _caption_rng(item_path, epoch_num, "shuffle")
                    rng.shuffle(shuffleable_tokens)
                else:
                    # 完全ランダムシャッフル
                    random.shuffle(shuffleable_tokens)

                token_list = fixed_tokens + shuffleable_tokens

    # Step 6: Tag normalization (normalize tags to standard format)
    if normalize_tags:
        from core.training.tag_group_utils import normalize_tag_for_output
        token_list = [normalize_tag_for_output(token) for token in token_list]

    # 再結合
    return ', '.join(token_list)


def _is_person_count_tag(tag: str) -> bool:
    """
    Check if a tag is a person count tag (e.g., "1girl", "2boys").

    IMPORTANT: This must match PERSON_COUNT_TAGS in tag_group_utils.py.
    Do NOT use .endswith() as it incorrectly matches "magical girl", "sailor girl", etc.

    Args:
        tag: Tag string

    Returns:
        True if tag is a person count tag
    """
    # Normalize: lowercase, underscores to spaces
    tag_normalized = tag.lower().strip().replace('_', ' ')

    # Person count tags (must match tag_group_utils.PERSON_COUNT_TAGS)
    # These are special tags that indicate person count or focus
    PERSON_COUNT_TAGS = {
        'no humans', 'no_humans',
        'solo',
        'group',
        'still life', 'still_life',
        'multiple girls', 'multiple_girls',
        'multiple boys', 'multiple_boys',
        'multiple others', 'multiple_others',
        'solo focus', 'solo_focus',
        'male focus', 'male_focus',
        'other focus', 'other_focus',
        '1girl', '2girls', '3girls', '4girls', '5girls', '6+girls',
        '1boy', '2boys', '3boys', '4boys', '5boys', '6+boys',
        '1other', '2others', '3others', '4others', '5others', '6+others',
    }

    return tag_normalized in PERSON_COUNT_TAGS


def get_default_caption_processing_config() -> Dict[str, any]:
    """
    Get default caption processing configuration.

    Returns:
        Dict with default caption processing settings
    """
    return {
        "caption_dropout_rate": 0.0,
        "token_dropout_rate": 0.0,
        "keep_tokens": 0,
        "shuffle_tokens": False,
        "shuffle_per_epoch": False,
        "shuffle_keep_first_n": 0,
        "tag_dropout_rate": 0.0,
        "tag_dropout_per_epoch": False,
        "tag_dropout_keep_first_n": 0,
        "tag_dropout_category_rates": {},
        "tag_dropout_exclude_person_count": False,
    }


def process_caption_with_tag_data(
    tag_data: List[Dict[str, str]],
    epoch_num: int,
    item_path: str,
    caption_config: Dict,
) -> str:
    """
    Fast per-epoch caption processing using pre-categorized tag_data.

    Args:
        tag_data: List of {"tag": "1girl", "category": "General"} dicts
        epoch_num: Current epoch number (for per-epoch shuffle/dropout seed)
        item_path: Image path (for per-epoch shuffle/dropout seed)
        caption_config: Caption processing configuration

    Returns:
        Processed caption string (comma-separated tags)
    """
    # Extract tags with categories
    tags_with_categories = [(item["tag"], item.get("category", "")) for item in tag_data]

    # Step 1: Category ordering (reorder tags by category)
    # This is done FIRST, before dropout and shuffle
    category_order = caption_config.get("category_order", None)
    if category_order and len(category_order) > 0:
        # Group tags by category
        categorized: Dict[str, List[tuple]] = {}
        unknown_tags: List[tuple] = []

        for tag, category in tags_with_categories:
            if category in category_order:
                if category not in categorized:
                    categorized[category] = []
                categorized[category].append((tag, category))
            else:
                unknown_tags.append((tag, category))

        # Rebuild tags_with_categories in category order
        reordered_tags = []
        for category in category_order:
            if category in categorized:
                reordered_tags.extend(categorized[category])

        # Add unknown tags at the end
        reordered_tags.extend(unknown_tags)

        tags_with_categories = reordered_tags

    # Apply tag dropout (category-aware)
    tag_dropout_rate = caption_config.get("tag_dropout_rate", 0.0)
    tag_dropout_per_epoch = caption_config.get("tag_dropout_per_epoch", False)
    tag_dropout_keep_first_n = caption_config.get("tag_dropout_keep_first_n", 0)
    tag_dropout_category_rates = caption_config.get("tag_dropout_category_rates", {})
    tag_dropout_exclude_person_count = caption_config.get("tag_dropout_exclude_person_count", False)

    if tag_dropout_rate > 0:
        tag_dropout_rng = (
            _caption_rng(item_path, epoch_num, "tag_dropout")
            if tag_dropout_per_epoch else random
        )

        filtered_tags = []
        for idx, (tag, category) in enumerate(tags_with_categories):
            # Keep first N tags
            if idx < tag_dropout_keep_first_n:
                filtered_tags.append((tag, category))
                continue

            # Exclude person count tags if enabled
            if tag_dropout_exclude_person_count and _is_person_count_tag(tag):
                filtered_tags.append((tag, category))
                continue

            # Category-specific dropout rate
            category_rate = tag_dropout_category_rates.get(category, tag_dropout_rate)

            if tag_dropout_rng.random() >= category_rate:
                filtered_tags.append((tag, category))

        tags_with_categories = filtered_tags

    # Apply shuffle (category-aware)
    shuffle_tokens = caption_config.get("shuffle_tokens", False)
    shuffle_per_epoch = caption_config.get("shuffle_per_epoch", False)
    shuffle_keep_first_n = caption_config.get("shuffle_keep_first_n", 0)
    shuffle_tag_groups = caption_config.get("shuffle_tag_groups", None)
    shuffle_groups_together = caption_config.get("shuffle_groups_together", False)
    exclude_person_count_from_shuffle = caption_config.get("exclude_person_count_from_shuffle", False)

    if shuffle_tokens:
        shuffle_rng = (
            _caption_rng(item_path, epoch_num, "shuffle")
            if shuffle_per_epoch else random
        )

        # Split into kept and shuffled parts
        kept_tags = tags_with_categories[:shuffle_keep_first_n]
        tags_to_shuffle = tags_with_categories[shuffle_keep_first_n:]

        if shuffle_tag_groups and len(shuffle_tag_groups) > 0:
            # Category-aware shuffle
            shuffle_tag_groups_set = set(shuffle_tag_groups)  # For O(1) lookup
            groups_dict = {group: [] for group in shuffle_tag_groups}
            person_count_tags = []  # Person count tags (1girl, 2boys, etc.)
            non_shuffled_tags = []  # Tags not in shuffle_tag_groups (preserve category order)

            for tag, category in tags_to_shuffle:
                # Exclude person count tags if enabled (will be placed at the start of General group)
                if exclude_person_count_from_shuffle and category == "General" and _is_person_count_tag(tag):
                    person_count_tags.append((tag, category))
                elif category in shuffle_tag_groups_set:
                    groups_dict[category].append((tag, category))
                else:
                    # Non-shuffled tags: preserve category order
                    non_shuffled_tags.append((tag, category))

            # Shuffle within each group
            for group in groups_dict:
                shuffle_rng.shuffle(groups_dict[group])

            # Rebuild tags in category_order (if available) or original order
            shuffled_tags = []
            if shuffle_groups_together:
                # Shuffle all selected groups together
                all_group_tags = []
                for group_tags in groups_dict.values():
                    all_group_tags.extend(group_tags)
                shuffle_rng.shuffle(all_group_tags)
                shuffled_tags.extend(all_group_tags)
                # Append non-shuffled tags at the end
                shuffled_tags.extend(non_shuffled_tags)
            else:
                # Preserve category order: iterate through all tags and insert in correct position
                # Use category_order if available, otherwise use original tag order
                category_order = caption_config.get("category_order", None)

                if category_order:
                    # Rebuild in category_order
                    for category in category_order:
                        # Insert person count tags before General group tags
                        if category == "General" and person_count_tags:
                            shuffle_rng.shuffle(person_count_tags)
                            shuffled_tags.extend(person_count_tags)
                            person_count_tags = []  # Clear to avoid duplicates

                        # Add shuffled tags from this category
                        if category in groups_dict:
                            shuffled_tags.extend(groups_dict[category])

                        # Add non-shuffled tags from this category
                        category_non_shuffled = [t for t in non_shuffled_tags if t[1] == category]
                        shuffled_tags.extend(category_non_shuffled)

                    # If General was not in category_order, append person count tags at the end
                    if person_count_tags:
                        shuffle_rng.shuffle(person_count_tags)
                        shuffled_tags.extend(person_count_tags)

                    # Add any remaining non-shuffled tags (categories not in category_order)
                    categorized_categories = set(category_order)
                    remaining_non_shuffled = [t for t in non_shuffled_tags if t[1] not in categorized_categories]
                    shuffled_tags.extend(remaining_non_shuffled)
                else:
                    # No category_order: fallback to original behavior (may break category order)
                    # Insert person count tags at the start of General group
                    person_count_inserted = False
                    for group in shuffle_tag_groups:
                        if group == "General" and person_count_tags and not person_count_inserted:
                            shuffle_rng.shuffle(person_count_tags)
                            shuffled_tags.extend(person_count_tags)
                            person_count_inserted = True

                        shuffled_tags.extend(groups_dict[group])

                    # Append person count tags at the end if not inserted
                    if person_count_tags:
                        shuffle_rng.shuffle(person_count_tags)
                        shuffled_tags.extend(person_count_tags)

                    # Append non-shuffled tags
                    shuffled_tags.extend(non_shuffled_tags)

            tags_with_categories = kept_tags + shuffled_tags
        else:
            # Simple shuffle
            shuffle_rng.shuffle(tags_to_shuffle)
            tags_with_categories = kept_tags + tags_to_shuffle

    # Extract tags only (discard categories)
    tags = [tag for tag, _ in tags_with_categories]

    # Step: Tag normalization (normalize tags to standard format)
    # This converts underscores to spaces and escapes parentheses
    normalize_tags = caption_config.get("normalize_tags", True)
    if normalize_tags:
        from core.training.tag_group_utils import normalize_tag_for_output
        tags = [normalize_tag_for_output(tag) for tag in tags]

    # Apply caption dropout
    caption_dropout_rate = caption_config.get("caption_dropout_rate", 0.0)
    return apply_caption_dropout(", ".join(tags), caption_dropout_rate)
