"""
Caption format detection and classification.

Classifies caption fields as:
- metadata: Display-only fields (timestamps, author, metrics, etc.)
- tags: Danbooru-style tags (training with tag processing)
- natural_language: Natural language text (training without tag processing)
"""

import re
from typing import Tuple, List, Dict, Any


def classify_field(field_name: str, content: str, taglist: set) -> Tuple[str, bool, float]:
    """
    Classify field as: 'metadata', 'tags', or 'natural_language'

    Args:
        field_name: JSON field name or field path (e.g., "metrics.retweets")
        content: Field content
        taglist: Set of known tags for matching

    Returns:
        (field_category, is_tags_format, tag_match_rate)

        field_category: 'metadata' | 'training'
        is_tags_format: True if tags format, False if natural language or metadata
        tag_match_rate: 0.0-1.0 (percentage of tokens matching taglist)
    """
    # Step 1: Metadata detection (highest priority)
    if is_metadata_field(field_name, content):
        return ('metadata', False, 0.0)

    # Step 2: Training field detection (tags vs natural language)
    is_tags, match_rate = detect_caption_format(content, taglist)

    return ('training', is_tags, match_rate)


def is_metadata_field(field_name: str, content: str) -> bool:
    """
    Detect if field is metadata (not for training)

    Metadata includes: timestamps, author, metrics, IDs, URLs, etc.
    """
    # Metadata field name patterns
    metadata_patterns = [
        'saved', 'author', 'timestamp', 'date', 'time',
        'metrics', 'stats', 'count', 'id', 'url', 'link',
        'created', 'updated', 'modified', 'published',
        'retweet', 'like', 'impression', 'view', 'follow',
        'user', 'username', 'userid', 'source', 'origin'
    ]

    field_lower = field_name.lower()
    if any(pattern in field_lower for pattern in metadata_patterns):
        return True

    # Content-based detection
    content_stripped = content.strip()
    content_lower = content_stripped.lower()

    # Empty content
    if not content_stripped:
        return True

    # Boolean values (True, False, true, false, 0, 1)
    if content_lower in ('true', 'false', '0', '1'):
        return True

    # ISO timestamps (e.g., "2025-08-29T07:00:21.893Z")
    if re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', content_stripped):
        return True

    # Pure numbers (e.g., metrics values: "28", "71")
    if content_stripped.isdigit():
        return True

    # Floating point numbers
    try:
        float(content_stripped)
        return True
    except ValueError:
        pass

    # URLs
    if content_stripped.startswith(('http://', 'https://', 'www.')):
        return True

    # Very short strings (likely IDs, not captions)
    if len(content_stripped) < 3:
        return True

    return False


def detect_caption_format(content: str, taglist: set) -> Tuple[bool, float]:
    """
    Detect if content is Danbooru tags format vs natural language

    Args:
        content: Caption content
        taglist: Set of known tags for matching

    Returns:
        (is_tags_format, tag_match_rate)

    Algorithm:
    1. Split by comma (tags are comma-separated)
    2. Heuristics to avoid false positives:
       - Too few tokens → natural language
       - Too many multi-word tokens → natural language
       - "The character is shinichi kudo, he is ..." → natural language
    3. Match tokens against taglist
    4. High match rate (≥70%) → tags format
    """
    content = content.strip()

    # Empty content
    if not content:
        return (False, 0.0)

    # Split by comma (tags are comma-separated)
    tokens = [t.strip() for t in content.split(',')]

    # Heuristic 1: Too few tokens (likely natural language sentence)
    if len(tokens) < 3:
        return (False, 0.0)

    # Heuristic 2: Tokens contain multiple words (natural language, not tags)
    # Tags can have 2-3 words (e.g., "long hair"), but not full sentences
    multi_word_tokens = [t for t in tokens if len(t.split()) > 3]
    if len(multi_word_tokens) / len(tokens) > 0.3:  # 30%+ are multi-word
        return (False, 0.0)

    # Heuristic 3: Check for sentence patterns (capitalized start, periods)
    # "The character is shinichi kudo, he is ..." → has capitals and periods
    if has_sentence_pattern(content):
        return (False, 0.0)

    # Heuristic 4: Sentence-ending punctuation (periods, semicolons, em-dashes)
    # Tags almost never contain these; natural language frequently does.
    # Count only tokens that END with a period (not tokens like "..." or ";d" which
    # are valid Danbooru tags).  Also skip tokens that are entirely punctuation.
    sentence_period_count = sum(
        1 for t in tokens
        if t.endswith('.') and not all(ch in '.\u2026' for ch in t)
    )
    # Semicolons inside tokens (e.g. ";d", ";t") are emote tags \u2014 count only
    # tokens whose entire content is punctuation-heavy natural-language fragments.
    sentence_semi_count = sum(
        1 for t in tokens
        if ';' in t and len(t.split()) > 1  # multi-word token containing ";"
    )
    em_dash_count = content.count('\u2014')
    sentence_punct_count = sentence_period_count + sentence_semi_count + em_dash_count
    if sentence_punct_count >= 2:
        return (False, 0.0)

    # Heuristic 5: Average words per token
    # Tags average ~1.5 words per comma-separated token
    # Natural language with commas averages higher
    total_words = sum(len(t.split()) for t in tokens)
    avg_words_per_token = total_words / len(tokens)
    if avg_words_per_token > 3.0:
        return (False, 0.0)

    # Heuristic 4: Match against taglist
    matched = 0
    for token in tokens:
        # Normalize token (lowercase, strip)
        normalized = token.lower().strip()

        # Skip empty tokens
        if not normalized:
            continue

        # Check exact match
        if normalized in taglist:
            matched += 1
            continue

        # Check with underscores replaced by spaces
        if normalized.replace('_', ' ') in taglist:
            matched += 1
            continue

        # Check with spaces replaced by underscores
        if normalized.replace(' ', '_') in taglist:
            matched += 1
            continue

    match_rate = matched / len(tokens) if tokens else 0.0

    # Heuristic 5: High match rate = tags format
    is_tags = match_rate >= 0.70  # 70%+ tags recognized

    return (is_tags, match_rate)


def has_sentence_pattern(text: str) -> bool:
    """
    Detect if text has natural language sentence patterns

    Examples:
    - "The character is shinichi kudo, he is ..." → True
    - "1girl, long hair, blue eyes" → False
    """
    # Check for capitalized words at sentence start (after period or comma)
    # Natural language: "The character is X, She is ..."
    # Tags: "1girl, long hair, blue eyes"

    # Split by punctuation
    sentences = re.split(r'[,.]', text)

    # Count capitalized starts
    capitalized_count = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence[0].isupper() and len(sentence.split()) > 1:
            # "The character" → capitalized + multi-word
            capitalized_count += 1

    # If 30%+ of segments are capitalized multi-word, likely natural language
    if len(sentences) > 0 and capitalized_count / len(sentences) > 0.3:
        return True

    # Check for common sentence starters
    sentence_starters = ['the ', 'a ', 'an ', 'this ', 'that ', 'these ', 'those ',
                         'he ', 'she ', 'it ', 'they ', 'in ', 'on ', 'with ']
    text_lower = text.lower()
    if any(text_lower.startswith(starter) for starter in sentence_starters):
        return True

    # Check for period-then-capital pattern (multiple sentences)
    # e.g., "She is smiling. Her hair is long."
    if re.search(r'\.\s+[A-Z]', text):
        return True

    # Check for semicolon usage (common in descriptive captions, rare in tags).
    # Exception: Danbooru emote tags like ";d", ";t", ";p", ";q" are single
    # comma-separated tokens containing only punctuation + one letter.
    if ';' in text:
        tokens_local = [t.strip() for t in text.split(',')]
        non_emote_semis = [
            t for t in tokens_local
            if ';' in t and not re.match(r'^[;:><oO0\-\^\+\*~\.!?x\|pPdDqQtTvVwW]+$', t)
        ]
        if non_emote_semis:
            return True

    return False


def scan_json_fields(json_data: Dict[str, Any], taglist: set, prefix: str = "") -> List[Dict[str, Any]]:
    """
    Recursively scan JSON fields and classify each

    IMPORTANT: Only ONE tags field is allowed per item.
    If multiple tags-format fields are found, only the first one is used.

    Args:
        json_data: JSON object to scan
        taglist: Set of known tags
        prefix: Current field path (for nested fields)

    Returns:
        List of caption dicts: [
            {
                "caption_type": "tags",  # Only ONE tags field (first tags-format field found)
                "content": "1girl, long hair, ...",
                "field_category": "training",
                "is_tags_format": True,
                "tag_match_rate": 0.85,
                "source_field": "tags"  # Original field name
            },
            {
                "caption_type": "text",
                "content": "A girl with long hair...",
                "field_category": "training",
                "is_tags_format": False,
                "tag_match_rate": 0.15,
                "source_field": "text"
            },
            {
                "caption_type": "savedAt",
                "content": "2025-08-29T07:00:21.893Z",
                "field_category": "metadata",
                "is_tags_format": False,
                "tag_match_rate": 0.0,
                "source_field": "savedAt"
            }
        ]
    """
    results = []
    found_tags_field = False  # Track if we've already found a tags field

    for key, value in json_data.items():
        # Build field path
        field_path = f"{prefix}.{key}" if prefix else key

        # Handle nested objects (recursion)
        if isinstance(value, dict):
            # Pass found_tags_field state to nested recursion
            nested_results, found_tags_in_nested = _scan_json_fields_internal(
                value, taglist, prefix=field_path, found_tags_field=found_tags_field
            )
            results.extend(nested_results)
            if found_tags_in_nested:
                found_tags_field = True
            continue

        # Handle lists (skip for now)
        if isinstance(value, list):
            continue

        # Convert to string
        content = str(value)

        # Classify field
        field_category, is_tags_format, match_rate = classify_field(field_path, content, taglist)

        # Enforce single tags field constraint
        if is_tags_format and field_category == "training":
            if found_tags_field:
                # Skip this tags field (already found one)
                print(f"[CaptionDetector] Skipping duplicate tags field: {field_path} (first tags field already used)")
                continue
            else:
                # First tags field - use it with caption_type="tags"
                results.append({
                    "caption_type": "tags",  # Normalized to "tags"
                    "content": content,
                    "field_category": field_category,
                    "is_tags_format": is_tags_format,
                    "tag_match_rate": match_rate,
                    "source_field": field_path  # Original field name
                })
                found_tags_field = True
                continue

        # Non-tags field: use original field name as caption_type
        results.append({
            "caption_type": field_path,
            "content": content,
            "field_category": field_category,
            "is_tags_format": is_tags_format,
            "tag_match_rate": match_rate,
            "source_field": field_path
        })

    return results


def _scan_json_fields_internal(json_data: Dict[str, Any], taglist: set, prefix: str = "", found_tags_field: bool = False) -> tuple:
    """
    Internal recursive helper for scan_json_fields
    Returns: (results, found_tags_field)
    """
    results = []

    for key, value in json_data.items():
        field_path = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            nested_results, found_tags_in_nested = _scan_json_fields_internal(
                value, taglist, prefix=field_path, found_tags_field=found_tags_field
            )
            results.extend(nested_results)
            if found_tags_in_nested:
                found_tags_field = True
            continue

        if isinstance(value, list):
            continue

        content = str(value)
        field_category, is_tags_format, match_rate = classify_field(field_path, content, taglist)

        if is_tags_format and field_category == "training":
            if found_tags_field:
                print(f"[CaptionDetector] Skipping duplicate tags field: {field_path}")
                continue
            else:
                results.append({
                    "caption_type": "tags",
                    "content": content,
                    "field_category": field_category,
                    "is_tags_format": is_tags_format,
                    "tag_match_rate": match_rate,
                    "source_field": field_path
                })
                found_tags_field = True
                continue

        results.append({
            "caption_type": field_path,
            "content": content,
            "field_category": field_category,
            "is_tags_format": is_tags_format,
            "tag_match_rate": match_rate,
            "source_field": field_path
        })

    return (results, found_tags_field)
