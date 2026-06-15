"""
Comma-containing tag resolver for SigLIP2 tagger training.

Some booru tags contain literal commas in their *name* — almost all are
Gelbooru copyright titles (visual-novel / movie titles), e.g.

    "godzilla, mothra and king ghidorah: giant monsters all-out attack"
    "ghidorah, the three-headed monster"

Captions are stored comma-separated, so when such a tag is written into a
caption and later split on commas it breaks into fragments:

    "godzilla"  +  "mothra and king ghidorah: giant monsters all-out attack"

The leading fragment usually collides with a real standalone tag
("godzilla") while the trailing fragment becomes an orphan that pollutes the
vocabulary as an "Unknown" tag.

This module reconstructs the original tag as a single **comma-free** canonical
vocabulary entry (the comma cannot survive comma-separated captions, so the
inference-facing tag must not contain one):

    "godzilla mothra and king ghidorah: giant monsters all-out attack"

Resolution runs in priority order on an *ordered* token list:

  1. Order-aware re-merge — adjacent fragments whose comma-join matches a known
     comma-tag are merged into the canonical comma-free form (recovers head+tail).
  2. Tail-fragment alias — a leftover orphan fragment equal to the *unique*
     trailing piece of a known comma-tag is aliased to that canonical form.

An orphan *leading* fragment is left unchanged (it is usually a legitimate
standalone tag, so rewriting it would be destructive).
"""

from __future__ import annotations

import html
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .tag_vocabulary import normalize_tag

_WS_RE = re.compile(r"\s+")


def comma_free(norm_tag: str) -> str:
    """Comma-free canonical form: drop commas and collapse whitespace.

    Input is expected to already be normalized (lowercase, space form).
    """
    return _WS_RE.sub(" ", norm_tag.replace(",", " ")).strip()


class CommaTagResolver:
    """Resolves comma-split tag fragments back to a single comma-free tag."""

    def __init__(self) -> None:
        # parts[0] -> list of (parts, canonical, category), longest-first
        self._by_first: Dict[str, List[Tuple[List[str], str, str]]] = defaultdict(list)
        # unique, non-colliding trailing fragment -> (canonical, category)
        self._tail: Dict[str, Tuple[str, str]] = {}
        # canonical comma-free form -> category
        self._canonical_cat: Dict[str, str] = {}
        # canonical comma-free form -> the normalized constituent parts
        # (the comma-split fragments). Used for vocab-lineage / head-weight
        # inheritance: a merged canonical can inherit from its old fragments.
        self._canonical_parts: Dict[str, List[str]] = {}

    def __len__(self) -> int:
        return len(self._canonical_cat)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build_from_category_map(cls, category_map: Dict[str, str]) -> "CommaTagResolver":
        """Build from a normalized ``tag -> category`` map.

        Pass ``taglist_cache._category_map`` (keys are normalized, so commas in
        tag names are preserved). Only multi-part comma-tags (content on both
        sides of a comma) are used; trailing-comma noise like ``"breasts,"`` is
        ignored because it splits to a single non-empty part and is harmless.
        """
        r = cls()
        tail_owners: Dict[str, set] = defaultdict(set)

        for norm_tag, category in category_map.items():
            if "," not in norm_tag:
                continue
            # The Gelbooru taglist stores HTML entities un-decoded ("&amp;",
            # "&#039;") whereas captions contain the literal characters, so
            # decode before matching against caption tokens.
            decoded = html.unescape(norm_tag)
            parts = [normalize_tag(p) for p in decoded.split(",")]
            parts = [p for p in parts if p]
            if len(parts) < 2:
                continue  # trailing/leading-only comma -> harmless, skip

            canonical = comma_free(decoded)
            if not canonical:
                continue

            r._by_first[parts[0]].append((parts, canonical, category))
            r._canonical_cat[canonical] = category
            r._canonical_parts[canonical] = parts
            tail_owners[parts[-1]].add(canonical)

        # Longest-match first so "a, b, c" wins over "a, b".
        for first in r._by_first:
            r._by_first[first].sort(key=lambda x: len(x[0]), reverse=True)

        # Tail-alias only for tails that are (a) unambiguous (own exactly one
        # canonical) and (b) not themselves a real standalone tag — otherwise
        # aliasing would corrupt a legitimate tag.
        for tail, owners in tail_owners.items():
            if len(owners) != 1:
                continue
            if tail in category_map:
                continue
            canonical = next(iter(owners))
            r._tail[tail] = (canonical, r._canonical_cat[canonical])

        return r

    @classmethod
    def build_from_taglist_cache(cls, root_dir: str, use_gelbooru: bool = True) -> "CommaTagResolver":
        """Initialize the shared taglist cache (optionally with the Gelbooru
        supplement) and build the resolver from its category map.

        Almost all comma-tags live in the Gelbooru taglist, so with
        ``use_gelbooru=False`` the resolver only covers the few Danbooru ones.
        Returns an empty resolver (a no-op) if the cache cannot be initialized.
        """
        try:
            from utils.taglist_cache import taglist_cache
            taglist_cache.initialize(root_dir, enable_gelbooru=bool(use_gelbooru))
            return cls.build_from_category_map(taglist_cache._category_map)
        except Exception as e:
            print(f"[CommaTagResolver] Could not build from taglist cache: {e}")
            return cls()

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def category_of(self, canonical: str) -> Optional[str]:
        """Category for a canonical comma-free tag, or None if not a comma-tag."""
        return self._canonical_cat.get(canonical)

    def canonical_parts(self) -> Dict[str, List[str]]:
        """Map each canonical comma-free tag to its normalized constituent parts
        (the comma-split fragments, in original order: ``[head, ..., tail]``).

        Used to build the vocab lineage so a merged canonical can inherit head
        weights from its old fragment tags.
        """
        return dict(self._canonical_parts)

    def resolve(self, tokens: List[str]) -> List[str]:
        """Resolve an ordered list of normalized tokens to canonical tokens.

        Re-merges adjacent comma-split fragments, then aliases leftover orphan
        tail fragments. Tokens that match nothing are returned unchanged.
        """
        if not self._by_first and not self._tail:
            return tokens

        out: List[str] = []
        i, n = 0, len(tokens)
        while i < n:
            tok = tokens[i]

            merged: Optional[Tuple[str, int]] = None
            candidates = self._by_first.get(tok)
            if candidates:
                for parts, canonical, _category in candidates:
                    span = len(parts)
                    if i + span <= n and tokens[i:i + span] == parts:
                        merged = (canonical, span)
                        break

            if merged is not None:
                out.append(merged[0])
                i += merged[1]
                continue

            tail = self._tail.get(tok)
            if tail is not None:
                out.append(tail[0])
            else:
                out.append(tok)
            i += 1

        return out
