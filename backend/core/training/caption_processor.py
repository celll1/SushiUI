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
        tag_manager = get_tag_group_manager(tag_group_dir)

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
    if caption_dropout_rate > 0:
        if random.random() < caption_dropout_rate:
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
            tag_manager = get_tag_group_manager(tag_group_dir)

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
                seed_value = hash(f"{item_path}_{token}_{epoch_num}") % (2**32)
                random_gen = random.Random(seed_value)
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

            tag_manager = get_tag_group_manager(tag_group_dir)

            if shuffle_per_epoch:
                # エポックごとに一貫したシャッフル（再現性あり）
                seed_str = f"{item_path}_{epoch_num}"
                seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
                rng = random.Random(seed)
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
                    seed_str = f"{item_path}_{epoch_num}"
                    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
                    rng = random.Random(seed)
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

    Args:
        tag: Tag string

    Returns:
        True if tag is a person count tag
    """
    tag_lower = tag.lower().strip()

    # 人数タグのパターン
    # 例: 1girl, 2girls, 1boy, 2boys, 3girls, multiple_girls, solo, etc.
    person_count_patterns = [
        'solo', 'multiple_girls', 'multiple_boys',
        '1girl', '2girls', '3girls', '4girls', '5girls', '6+girls',
        '1boy', '2boys', '3boys', '4boys', '5boys', '6+boys',
    ]

    return tag_lower in person_count_patterns


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
        # Determine random seed
        if tag_dropout_per_epoch:
            seed_str = f"{item_path}_tag_dropout_epoch{epoch_num}"
            seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
            random.seed(seed)

        filtered_tags = []
        for idx, (tag, category) in enumerate(tags_with_categories):
            # Keep first N tags
            if idx < tag_dropout_keep_first_n:
                filtered_tags.append((tag, category))
                continue

            # Exclude person count tags if enabled
            if tag_dropout_exclude_person_count and tag.endswith(("girl", "girls", "boy", "boys", "other", "others")):
                filtered_tags.append((tag, category))
                continue

            # Category-specific dropout rate
            category_rate = tag_dropout_category_rates.get(category, tag_dropout_rate)

            if random.random() >= category_rate:
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
        # Determine random seed
        if shuffle_per_epoch:
            seed_str = f"{item_path}_shuffle_epoch{epoch_num}"
            seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
            random.seed(seed)

        # Split into kept and shuffled parts
        kept_tags = tags_with_categories[:shuffle_keep_first_n]
        tags_to_shuffle = tags_with_categories[shuffle_keep_first_n:]

        if shuffle_tag_groups and len(shuffle_tag_groups) > 0:
            # Category-aware shuffle
            groups_dict = {group: [] for group in shuffle_tag_groups}
            person_count_tags = []  # Person count tags (1girl, 2boys, etc.)
            other_excluded_tags = []  # Non-group tags (not in shuffle_tag_groups)

            for tag, category in tags_to_shuffle:
                # Exclude person count tags if enabled (will be placed at the start of General group)
                if exclude_person_count_from_shuffle and category == "General" and tag.endswith(("girl", "girls", "boy", "boys", "other", "others")):
                    person_count_tags.append((tag, category))
                elif category in groups_dict:
                    groups_dict[category].append((tag, category))
                else:
                    other_excluded_tags.append((tag, category))

            # Shuffle each group
            shuffled_tags = []
            if shuffle_groups_together:
                # Shuffle all groups together
                all_group_tags = []
                for group_tags in groups_dict.values():
                    all_group_tags.extend(group_tags)
                random.shuffle(all_group_tags)
                shuffled_tags.extend(all_group_tags)
            else:
                # Shuffle within each group, and insert person count tags at the start of General group
                for group in shuffle_tag_groups:
                    group_tags = groups_dict[group]
                    random.shuffle(group_tags)

                    # If this is the General group, prepend person count tags (shuffled)
                    if group == "General" and person_count_tags:
                        random.shuffle(person_count_tags)  # Shuffle person count tags among themselves
                        shuffled_tags.extend(person_count_tags)
                        person_count_tags = []  # Clear to avoid duplicates

                    shuffled_tags.extend(group_tags)

                # If General group was not in shuffle_tag_groups, append person count tags at the end
                if person_count_tags:
                    random.shuffle(person_count_tags)  # Shuffle person count tags among themselves
                    shuffled_tags.extend(person_count_tags)

            shuffled_tags.extend(other_excluded_tags)
            tags_with_categories = kept_tags + shuffled_tags
        else:
            # Simple shuffle
            random.shuffle(tags_to_shuffle)
            tags_with_categories = kept_tags + tags_to_shuffle

    # Extract tags only (discard categories)
    tags = [tag for tag, _ in tags_with_categories]

    # Apply caption dropout
    caption_dropout_rate = caption_config.get("caption_dropout_rate", 0.0)
    if caption_dropout_rate > 0 and random.random() < caption_dropout_rate:
        return ""

    return ", ".join(tags)
