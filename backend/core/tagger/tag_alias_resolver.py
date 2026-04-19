"""
Tag alias resolver for SigLIP2 tagger training.

Resolves deprecated/alias tags to their canonical form using
tagother/tag_aliases.json (danbooru-format alias table).
"""

from __future__ import annotations

import json
from typing import Dict

from .tag_vocabulary import normalize_tag


class TagAliasResolver:
    """Resolves deprecated/alias tags to their canonical (space-form) tag.

    The alias file (tagother/tag_aliases.json) uses lowercase + underscore +
    unescaped keys, matching danbooru's standard format.  This class converts
    any raw tag representation to that key format before looking it up.

    Lookup key pipeline:
        raw tag  →  strip  →  lower  →  unescape  →  replace(" ", "_")
        = danbooru_key  →  alias dict lookup  →  normalize_tag()
        = final vocabulary key (space form)
    """

    def __init__(self, aliases: Dict[str, str]) -> None:
        self._aliases = aliases  # danbooru_key -> canonical_danbooru_key

    @classmethod
    def load(cls, path: str) -> "TagAliasResolver":
        """Load alias table from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    def __len__(self) -> int:
        return len(self._aliases)

    def to_danbooru_key(self, tag: str) -> str:
        """Convert a raw tag to danbooru lookup key format.

        Steps: strip → lower → unescape /( /) and \( \) → replace spaces with '_'
        """
        t = tag.strip().lower()
        # Unescape Danbooru wiki-link parens: /( /) and SD-style \( \) \/ (loop for multi-layer)
        while True:
            prev = t
            t = (t.replace("/(", "(").replace("/)", ")")
                  .replace("\\(", "(").replace("\\)", ")").replace("\\/", "/"))
            if t == prev:
                break
        return t.replace(" ", "_")

    def resolve(self, tag: str) -> str:
        """Return canonical vocabulary key (space form) for a raw tag.

        If the tag has an alias entry, returns the canonical form.
        Otherwise returns normalize_tag(tag) unchanged.
        """
        key = self.to_danbooru_key(tag)
        canonical = self._aliases.get(key, key)  # fallback: identity
        return normalize_tag(canonical)
