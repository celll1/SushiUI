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
    sentence_starters = ['the ', 'a ', 'an ', 'this ', 'that ', 'these ', 'those ', 'he ', 'she ', 'it ', 'they ']
    text_lower = text.lower()
    if any(text_lower.startswith(starter) for starter in sentence_starters):
        return True

    return False


def scan_json_fields(json_data: Dict[str, Any], taglist: set, prefix: str = "") -> List[Dict[str, Any]]:
    """
    Recursively scan JSON fields and classify each

    Args:
        json_data: JSON object to scan
        taglist: Set of known tags
        prefix: Current field path (for nested fields)

    Returns:
        List of caption dicts: [
            {
                "caption_type": "tags",
                "content": "1girl, long hair, ...",
                "field_category": "training",
                "is_tags_format": True,
                "tag_match_rate": 0.85,
                "source_field": "tags"
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
            },
            {
                "caption_type": "metrics.retweets",
                "content": "28",
                "field_category": "metadata",
                "is_tags_format": False,
                "tag_match_rate": 0.0,
                "source_field": "metrics.retweets"
            }
        ]
    """
    results = []

    for key, value in json_data.items():
        # Build field path
        field_path = f"{prefix}.{key}" if prefix else key

        # Handle nested objects (recursion)
        if isinstance(value, dict):
            nested_results = scan_json_fields(value, taglist, prefix=field_path)
            results.extend(nested_results)
            continue

        # Handle lists (skip for now)
        if isinstance(value, list):
            continue

        # Convert to string
        content = str(value)

        # Classify field
        field_category, is_tags_format, match_rate = classify_field(field_path, content, taglist)

        results.append({
            "caption_type": field_path,  # Use full path as caption_type (e.g., "metrics.retweets")
            "content": content,
            "field_category": field_category,
            "is_tags_format": is_tags_format,
            "tag_match_rate": match_rate,
            "source_field": field_path
        })

    return results
