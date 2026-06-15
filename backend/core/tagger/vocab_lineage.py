"""
Vocabulary lineage for SigLIP2 tagger head-weight / optimizer-state migration.

When the vocabulary changes between runs (resume / ``init_head_from``), tags
can be **renamed** (deprecated-alias resolution) or **merged** (comma-split
fragments re-joined into a comma-free canonical tag). Plain tag-name alignment
in ``_inherit_head`` / ``migrate_head_optimizer_state`` only handles tags whose
name is unchanged, so renamed/merged tags would lose their learned head rows.

This module builds a lineage map ``new_tag -> [old_predecessor, ...]`` so the
migration code can inherit a predecessor's row when an exact-name match is
absent. Predecessors are ordered **most-specific first**, so the migration can
take the first one that exists in the old vocabulary (tail-priority policy):

  * Comma merge — ``"godzilla mothra and king ghidorah: ..."`` inherits from
    its fragments ordered ``[tail, ..., head]`` i.e. the unique, most-specific
    trailing fragment first, the broad leading fragment ("godzilla") last.
  * Alias rename — the new canonical inherits from the deprecated old tag(s).

Only entries that are *actionable* are kept: the new tag must be in the new
vocab and absent from the old vocab (exact-name match takes precedence), and
each predecessor must exist in the old vocab.
"""

from __future__ import annotations

from typing import Dict, List


def build_vocab_lineage(
    old_tag_to_idx: Dict[str, int],
    new_tag_to_idx: Dict[str, int],
    alias_resolver=None,
    comma_resolver=None,
) -> Dict[str, List[str]]:
    """Build ``new_tag -> [old_predecessor, ...]`` (most-specific first).

    ``alias_resolver`` and ``comma_resolver`` are duck-typed: the former needs
    ``resolve(tag) -> canonical``; the latter needs ``canonical_parts() ->
    {canonical: [part, ...]}``. Either may be None.
    """
    lineage: Dict[str, List[str]] = {}

    def _add(new_tag: str, predecessor: str) -> None:
        # Exact-name match is handled directly by the migrator — skip those.
        if new_tag in old_tag_to_idx or new_tag not in new_tag_to_idx:
            return
        if predecessor == new_tag or predecessor not in old_tag_to_idx:
            return
        preds = lineage.setdefault(new_tag, [])
        if predecessor not in preds:
            preds.append(predecessor)

    # 1. Comma merges — tail first (most specific), head last.
    if comma_resolver is not None and hasattr(comma_resolver, "canonical_parts"):
        for canonical, parts in comma_resolver.canonical_parts().items():
            for predecessor in reversed(parts):
                _add(canonical, predecessor)

    # 2. Alias renames — old deprecated tag -> new canonical.
    if alias_resolver is not None:
        for old_tag in old_tag_to_idx:
            try:
                canonical = alias_resolver.resolve(old_tag)
            except Exception:
                continue
            if canonical != old_tag:
                _add(canonical, old_tag)

    return lineage
