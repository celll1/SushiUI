"""
Query → concrete-tag resolver for Danbooru Query-mode vocabulary expansion.

A user Danbooru query (e.g. ``"blue_* score:>=50"``) is parsed into its positive
tag tokens (metatags like ``score:`` / ``order:`` and exclusions like ``-tag`` /
``!tag`` are dropped). Each token — exact or wildcard — is resolved against the
Danbooru tags API (``name_matches``) into concrete tags with their post_count and
category. Results are filtered to the eligible categories and a post_count floor,
then capped to the top-K by post_count.

This lets Query mode add the tags a user actually crafted the query to capture
(matching the "I built this query for those tags" intent), bounded so a broad
wildcard cannot resolve into an unbounded flood (top_k + min_count + categories).
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from .tag_vocabulary import normalize_tag


# Danbooru metatag prefixes that are search operators, not tags. A token
# containing ':' is treated as a metatag and skipped during resolution.
def _is_metatag(token: str) -> bool:
    return ":" in token


def extract_tag_tokens(query: str) -> List[str]:
    """Return the positive tag tokens of a query (underscore form, wildcards kept).

    Drops: metatags (``score:>=50``, ``order:random``, ``rating:g`` …) and
    exclusions (``-tag``, ``!tag``). Keeps positive tag tokens including those
    with ``*`` wildcards.
    """
    tokens: List[str] = []
    for tok in (query or "").split():
        t = tok.strip()
        if not t:
            continue
        if t[0] in ("-", "!"):       # exclusion — not a collection/expansion target
            continue
        if _is_metatag(t):           # metatag operator, not a tag
            continue
        tokens.append(t)
    return tokens


class QueryResolver:
    """Resolves query tag tokens/wildcards to concrete Danbooru tags."""

    def __init__(
        self,
        client,
        min_count: int = 200,
        categories: List[int] = (0, 3, 4),
        top_k: int = 50,
    ) -> None:
        self._client = client
        self._min_count = max(0, int(min_count))
        self._categories: Set[int] = set(int(c) for c in (categories or []))
        self._top_k = int(top_k)

    def resolve_query(self, query: str) -> List[Tuple[str, int, int]]:
        """Resolve one query string.

        Returns a list of ``(normalized_tag, post_count, category)`` for tags that
        match the query's positive tokens, meet the post_count floor, and fall in
        an eligible category — capped to the top-K by post_count.
        """
        seen: Dict[str, Tuple[str, int, int]] = {}
        for token in extract_tag_tokens(query):
            try:
                rows = self._client.fetch_tags_by_name(
                    token, min_count=self._min_count, limit=200, page=1
                )
            except Exception as exc:
                print(f"[QueryResolver] resolve error for {token!r}: {exc}")
                continue
            for row in rows:
                name = row.get("name")
                if not name:
                    continue
                cat = int(row.get("category", -1))
                if self._categories and cat not in self._categories:
                    continue
                count = int(row.get("post_count", 0) or 0)
                if count < self._min_count:
                    continue
                norm = normalize_tag(name)
                if not norm:
                    continue
                # Keep the highest post_count if a tag matches multiple tokens.
                prev = seen.get(norm)
                if prev is None or count > prev[1]:
                    seen[norm] = (norm, count, cat)

        resolved = sorted(seen.values(), key=lambda r: -r[1])
        if self._top_k > 0:
            resolved = resolved[: self._top_k]
        return resolved
