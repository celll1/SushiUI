"""Per-caption artist and content labels (design §5)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional, Protocol

from core.tagger.tag_vocabulary import CATEGORY_ORDER, normalize_tag

_CANONICAL = {name.lower(): name for name in CATEGORY_ORDER}
_NOT_EXPLICIT = {"", "unknown", "__lookup__"}


class CategoryResolver(Protocol):
    """In production this is `get_tag_group_manager()` (tag_group_utils)."""

    def get_tag_group(self, tag: str) -> Optional[str]: ...


class AliasResolver(Protocol):
    """`TagAliasResolver`."""

    def resolve(self, tag: str) -> str: ...


class CommaResolver(Protocol):
    """`CommaTagResolver`."""

    def resolve(self, tokens: list[str]) -> list[str]: ...

    def category_of(self, canonical: str) -> Optional[str]: ...


@dataclass(frozen=True)
class StyleLabels:
    artists: frozenset[str]
    characters: frozenset[str]
    copyrights: frozenset[str]
    has_signature: bool


@dataclass
class LabelStats:
    items: int = 0
    items_without_artist: int = 0
    items_with_multiple_artists: int = 0
    artist_tags_explicit: int = 0
    artist_tags_manager: int = 0
    artist_tags_comma: int = 0
    category_disagreements: int = 0
    general_overridden_by_manager: int = 0
    unresolved_tags: int = 0
    non_copyright_tags_dropped: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def _explicit_category(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if stripped.lower() in _NOT_EXPLICIT:
        return None
    return _CANONICAL.get(stripped.lower(), stripped)


def caption_rows(tag_data: Any, content: Optional[str]) -> list[tuple[str, Optional[str]]]:
    """(raw tag, explicit category or None). Same source precedence as `extract_caption_tags`."""
    rows: list[tuple[str, Optional[str]]] = []
    if tag_data:
        try:
            raw = json.loads(tag_data) if isinstance(tag_data, str) else tag_data
        except (json.JSONDecodeError, TypeError):
            raw = None
        if isinstance(raw, list):
            for row in raw:
                if isinstance(row, dict) and isinstance(row.get("tag"), str):
                    rows.append((row["tag"], _explicit_category(row.get("category"))))
    if not rows and content:
        rows = [(tag.strip(), None) for tag in content.split(",") if tag.strip()]
    return rows


def canonical_tokens(rows: list[tuple[str, Optional[str]]], comma_resolver: Optional[CommaResolver],
                     alias_resolver: Optional[AliasResolver]) -> list[tuple[str, str, Optional[str], Optional[str]]]:
    """(canonical, pre-alias name, explicit category, comma-resolver category) per token.

    Mirrors the tagger vocabulary path: comma re-merge before aliasing, and a comma-tag's category is
    authoritative and never aliased. A merged token has no single explicit category.
    """
    names: list[str] = []
    explicit_of: dict[str, Optional[str]] = {}
    for raw, explicit in rows:
        name = normalize_tag(raw)
        if name:
            names.append(name)
            if explicit_of.get(name) is None:
                explicit_of[name] = explicit
    if comma_resolver is not None:
        names = comma_resolver.resolve(names)
    out = []
    for name in names:
        comma_category = comma_resolver.category_of(name) if comma_resolver is not None else None
        if comma_category is not None:
            out.append((name, name, None, comma_category))
        else:
            canonical = alias_resolver.resolve(name) if alias_resolver is not None else name
            out.append((canonical, name, explicit_of.get(name), None))
    return out


class StyleLabelResolver:
    def __init__(
        self,
        category_resolver: CategoryResolver,
        comma_resolver: Optional[CommaResolver],
        alias_resolver: Optional[AliasResolver],
        signature_tags: Iterable[str],
        non_copyright_tags: Iterable[str],
    ) -> None:
        """`non_copyright_tags` (e.g. `original`) are Copyright-category tags that name no shared work;
        they are dropped from `copyrights` after alias canonicalisation."""
        self._categories = category_resolver
        self._comma = comma_resolver
        self._aliases = alias_resolver
        self._signature = frozenset(normalize_tag(tag) for tag in signature_tags)
        self._non_copyright = frozenset(
            alias_resolver.resolve(normalize_tag(tag)) if alias_resolver is not None else normalize_tag(tag)
            for tag in non_copyright_tags)
        self._lookup_cache: dict[str, Optional[str]] = {}
        self.stats = LabelStats()

    def _lookup(self, name: str) -> Optional[str]:
        try:
            return self._lookup_cache[name]
        except KeyError:
            group = self._categories.get_tag_group(name)
            self._lookup_cache[name] = group
            return group

    def resolve(self, tag_data: Any, content: Optional[str]) -> StyleLabels:
        return self.resolve_captions(((tag_data, content),))

    def resolve_captions(self, captions: Iterable[tuple[Any, Optional[str]]]) -> StyleLabels:
        """One item's labels from all of its (tag_data, content) captions. Comma re-merge runs per
        caption, so fragments at the end of one caption never join the start of the next. Each canonical
        name is resolved once, with the first explicit and first comma category any caption gives it."""
        artists: set[str] = set()
        characters: set[str] = set()
        copyrights: set[str] = set()
        has_signature = False
        stats = self.stats
        merged: dict[str, list] = {}  # canonical -> [pre-alias names, explicit, comma category]
        for tag_data, content in captions:
            for name, original, explicit, comma_category in canonical_tokens(
                    caption_rows(tag_data, content), self._comma, self._aliases):
                entry = merged.setdefault(name, [[], None, None])
                if original not in entry[0]:
                    entry[0].append(original)
                if entry[1] is None:
                    entry[1] = explicit
                if entry[2] is None:
                    entry[2] = comma_category
        for name, (originals, explicit, comma_category) in merged.items():
            source = "explicit"
            if comma_category is not None:
                category = comma_category
                source = "comma"
            else:
                # get_tag_group has no alias fallback and taglists mostly lack deprecated names.
                looked_up = self._lookup(name)
                for original in originals:
                    if looked_up is not None:
                        break
                    if original != name:
                        looked_up = self._lookup(original)
                if explicit is None:
                    category = looked_up
                    source = "manager"
                    if category is None:
                        stats.unresolved_tags += 1
                elif explicit == "General" and looked_up is not None and looked_up != "General":
                    # The tag_data backfill (api/batch_operations.py) writes General for taglist misses.
                    category = looked_up
                    source = "manager"
                    stats.general_overridden_by_manager += 1
                else:
                    category = explicit
                    if looked_up is not None and looked_up != explicit:
                        stats.category_disagreements += 1
            if name in self._signature or not self._signature.isdisjoint(originals):
                has_signature = True
            if category == "Artist":
                if source == "explicit":
                    stats.artist_tags_explicit += 1
                elif source == "manager":
                    stats.artist_tags_manager += 1
                else:
                    stats.artist_tags_comma += 1
                artists.add(name)
            elif category == "Character":
                characters.add(name)
            elif category == "Copyright":
                if name in self._non_copyright:
                    stats.non_copyright_tags_dropped += 1
                else:
                    copyrights.add(name)
        stats.items += 1
        if not artists:
            stats.items_without_artist += 1
        elif len(artists) > 1:
            stats.items_with_multiple_artists += 1
        return StyleLabels(frozenset(artists), frozenset(characters), frozenset(copyrights), has_signature)
