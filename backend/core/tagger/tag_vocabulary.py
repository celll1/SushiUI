"""
Tag vocabulary management for SigLIP2 tagger training.

Collects tags from sushiUI Dataset (DatasetCaption with is_tags_format=True),
builds tag->index mapping with category information.
"""

from __future__ import annotations

import fnmatch
import json
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple

# Category sort order for vocabulary organization
CATEGORY_ORDER: List[str] = [
    "General", "Character", "Copyright", "Artist", "Meta", "Rating", "Quality", "Model"
]

# Quality tag groups (same as tagutl/lora.py)
QUALITY_TAG_GROUPS: Dict[str, List[str]] = {
    "high_quality_group": ["best quality", "high quality", "normal quality", "medium quality"],
    "low_quality_group":  ["low quality", "bad quality", "worst quality"],
}

# Rating tags
#
# IMPORTANT: these must match the *actual* tokens used in the training
# vocabulary and emitted by the Danbooru client (``_RATING_MAP``), which are
# the bare words ``general/sensitive/questionable/explicit`` (NO ``rating:``
# prefix). Using the prefixed form here leaves ``rating_indices`` empty, which
# silently disables rating loss-masking: samples without any rating annotation
# would then train all four rating tags as negatives (false negatives).
RATING_TAGS: List[str] = ["general", "sensitive", "questionable", "explicit"]


def normalize_tag(tag: str) -> str:
    """Normalize tag for consistent matching.

    Steps applied in order:
    1. Strip leading/trailing whitespace
    2. Replace underscores with spaces  (danbooru convention)
    3. Lowercase
    4. Remove Danbooru wiki link escaping  e.g. /(tag/) → (tag)
       used in wiki text to denote literal parentheses in tag names
    5. Remove backslash escaping           e.g. \\( → (  so that
       "fate \\(series\\)" and "fate (series)" collapse to the same key
    """
    tag = tag.strip().replace("_", " ").lower()
    # Unescape parenthesis conventions (loop to handle multiple layers):
    #   /( /)  — Danbooru wiki-link syntax (literal parens in tag names)
    #   \( \)  — SD/booru caption backslash escaping
    #   \/     — backslash before slash (e.g. fate\/extra → fate/extra)
    while True:
        prev = tag
        tag = (tag.replace("/(", "(").replace("/)", ")")
                   .replace("\\(", "(").replace("\\)", ")").replace("\\/", "/"))
        if tag == prev:
            break
    return tag


class TagVocabulary:
    """
    Maps tag strings to integer indices.

    Attributes
    ----------
    tag_to_idx   : Dict[str, int]   normalized tag -> index
    idx_to_tag   : Dict[int, str]   index -> normalized tag
    tag_to_category : Dict[str, str]  normalized tag -> category name
    rating_indices  : List[int]  indices of rating tags
    quality_indices : Dict[str, List[int]]  group_name -> indices
    """

    def __init__(self) -> None:
        self.tag_to_idx: Dict[str, int] = {}
        self.idx_to_tag: Dict[int, str] = {}
        self.tag_to_category: Dict[str, str] = {}
        self.rating_indices: List[int] = []
        self.quality_indices: Dict[str, List[int]] = {k: [] for k in QUALITY_TAG_GROUPS}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build_from_dataset_ids(
        cls,
        dataset_ids: List[int],
        datasets_db,
        min_count: int = 1,
        excluded_categories: Optional[List[str]] = None,
        ban_tags: Optional[List[str]] = None,
        alias_resolver=None,
        use_gelbooru_categories: bool = True,
        comma_resolver=None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> "TagVocabulary":
        """Build vocabulary by scanning DatasetCaption rows for given dataset IDs.

        Parameters
        ----------
        dataset_ids         : list of Dataset.id values
        datasets_db         : SQLAlchemy session for datasets.db
        min_count           : minimum occurrence count to include a tag
        excluded_categories : categories to exclude entirely (e.g. ["Artist"])
        ban_tags            : tag patterns to exclude; supports fnmatch wildcards
                              (e.g. ["some tag", "prefix_*", "bad*"])
        alias_resolver      : optional TagAliasResolver; when provided, deprecated
                              tags are resolved to canonical form before counting
        use_gelbooru_categories : when True, also resolve "Unknown" tag categories
                              against the Gelbooru taglist supplement (taglist_gel/)
                              in addition to the Danbooru taglist. Danbooru always
                              takes precedence; Gelbooru only fills tags Danbooru
                              does not know, reducing the count of "Unknown" tags.
        comma_resolver      : optional CommaTagResolver. When provided, comma-split
                              tag fragments are re-merged / aliased into a single
                              comma-free canonical tag per caption (order-aware
                              re-merge, then tail-fragment alias). See
                              core/tagger/comma_tag_resolver.py.
        """
        from database.models import DatasetItem, DatasetCaption

        tag_counts: Dict[str, int] = defaultdict(int)
        tag_categories: Dict[str, str] = {}

        # Single JOIN query — avoids N+1 (one query per item for lazy caption load).
        # Stream with yield_per instead of .all() so we can report progress and
        # avoid materialising millions of caption rows in memory at once.
        _base_q = (
            datasets_db.query(DatasetCaption)
            .join(DatasetItem, DatasetCaption.item_id == DatasetItem.id)
            .filter(
                DatasetItem.dataset_id.in_(dataset_ids),
                DatasetCaption.is_tags_format == True,
            )
        )
        _total = 0
        if progress_callback is not None:
            # Emit before count() (which itself scans) so the bar isn't blank.
            try:
                progress_callback(0, 1, "Building vocabulary: scanning captions...")
            except Exception:
                pass
            try:
                _total = _base_q.count()
            except Exception:
                _total = 0
            try:
                progress_callback(0, max(1, _total), f"Building vocabulary: 0/{_total:,} captions")
            except Exception:
                pass
        _last_emit = 0.0
        for _i, caption in enumerate(_base_q.yield_per(5000)):
            tags_with_cats = _parse_caption_tags(caption)
            # Normalize tokens preserving order; remember each token's source
            # category for the non-comma path below.
            norm_tokens: List[str] = []
            src_cat: Dict[str, str] = {}
            for tag, category in tags_with_cats:
                nt = normalize_tag(tag)
                if not nt:
                    continue
                norm_tokens.append(nt)
                src_cat.setdefault(nt, category)
            # Re-merge / alias comma-split fragments into comma-free canonical
            # tags (order-aware). Must run BEFORE deprecated-alias resolution so
            # comma-tag parts are matched in their original form.
            if comma_resolver is not None:
                norm_tokens = comma_resolver.resolve(norm_tokens)
            for nt in norm_tokens:
                comma_cat = comma_resolver.category_of(nt) if comma_resolver else None
                if comma_cat is not None:
                    # Canonical comma-free tag: authoritative category, not aliased.
                    norm = nt
                    category = comma_cat
                else:
                    norm = alias_resolver.resolve(nt) if alias_resolver else nt
                    category = src_cat.get(nt, "General")
                tag_counts[norm] += 1
                if norm not in tag_categories:
                    tag_categories[norm] = category
            # Throttled progress (~3x/sec) over the caption scan.
            if progress_callback is not None and _i % 5000 == 0:
                _now = time.monotonic()
                if _now - _last_emit >= 0.3:
                    _last_emit = _now
                    try:
                        progress_callback(_i, max(1, _total), f"Building vocabulary: {_i:,}/{_total:,} captions")
                    except Exception:
                        pass

        # Resolve "__lookup__" sentinels AND "Unknown" tags via taglist_cache.
        # "Unknown" may appear in tag_data when captions were built before a tag was
        # added to the local taglist — re-check so newly added tags get proper categories.
        lookup_tags = [t for t, c in tag_categories.items() if c in ("__lookup__", "Unknown")]
        if lookup_tags:
            try:
                from utils.taglist_cache import taglist_cache
                # Optionally enable the Gelbooru taglist supplement so that tags
                # absent from the local Danbooru taglist still get a category
                # (Danbooru takes precedence; Gelbooru only fills the gaps).
                # initialize() is idempotent: Danbooru categories are mtime-gated
                # and the Gelbooru supplement is a one-time latch, so re-calling
                # it here is cheap and safe.
                if use_gelbooru_categories:
                    try:
                        from config import settings as _settings
                        taglist_cache.initialize(_settings.root_dir, enable_gelbooru=True)
                    except Exception as _e:
                        print(f"[TagVocabulary] Could not enable Gelbooru category supplement: {_e}")
                resolved = taglist_cache.get_categories_batch(lookup_tags)
                for norm_tag in lookup_tags:
                    original = tag_categories[norm_tag]
                    found = resolved.get(norm_tag)
                    if found:
                        tag_categories[norm_tag] = found
                    elif original == "__lookup__":
                        tag_categories[norm_tag] = "General"
                    # else: keep "Unknown" if taglist doesn't know it either
            except Exception:
                # taglist_cache not initialized or unavailable — fall back to General for sentinels
                for norm_tag in lookup_tags:
                    if tag_categories[norm_tag] == "__lookup__":
                        tag_categories[norm_tag] = "General"

        # Filter by min_count
        filtered: Dict[str, int] = {t: c for t, c in tag_counts.items() if c >= min_count}

        # Filter by excluded_categories
        if excluded_categories:
            excl: Set[str] = {c.strip() for c in excluded_categories}
            filtered = {t: c for t, c in filtered.items()
                        if tag_categories.get(t, "General") not in excl}

        # Filter by ban_tags (fnmatch wildcards supported).
        # Tags are stored with spaces ("bad id") but patterns are typically
        # written with underscores ("bad_id", "bad_*_id") following Danbooru
        # convention.  Match against both the original form and the
        # underscore-normalised form so either notation works.
        if ban_tags:
            ban_patterns = [p.strip() for p in ban_tags if p.strip()]
            def _is_banned(tag: str) -> bool:
                tag_u = tag.replace(" ", "_")
                return any(
                    fnmatch.fnmatch(tag, pat) or fnmatch.fnmatch(tag_u, pat)
                    for pat in ban_patterns
                )
            filtered = {t: c for t, c in filtered.items() if not _is_banned(t)}

        # Sort: category order first, then alphabetically within each category
        def _sort_key(tag: str) -> tuple:
            cat = tag_categories.get(tag, "General")
            cat_rank = CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else len(CATEGORY_ORDER)
            return (cat_rank, tag)

        selected = sorted(filtered.keys(), key=_sort_key)

        vocab = cls()
        for idx, tag in enumerate(selected):
            vocab.tag_to_idx[tag] = idx
            vocab.idx_to_tag[idx] = tag
            vocab.tag_to_category[tag] = tag_categories.get(tag, "General")

        vocab._build_special_indices()
        return vocab

    @classmethod
    def from_dict(cls, data: dict) -> "TagVocabulary":
        """Restore from a plain dict (e.g. loaded from JSON)."""
        vocab = cls()
        vocab.tag_to_idx = {k: int(v) for k, v in data["tag_to_idx"].items()}
        vocab.idx_to_tag = {int(k): v for k, v in data["idx_to_tag"].items()}
        vocab.tag_to_category = data.get("tag_to_category", {})
        vocab._build_special_indices()
        return vocab

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        # Build categories section: category -> sorted list of tags
        categories: Dict[str, List[str]] = {}
        for tag, cat in self.tag_to_category.items():
            categories.setdefault(cat, []).append(tag)
        # Sort within each category alphabetically; sort category keys by CATEGORY_ORDER
        sorted_categories: Dict[str, List[str]] = {}
        cat_keys = sorted(
            categories.keys(),
            key=lambda c: CATEGORY_ORDER.index(c) if c in CATEGORY_ORDER else len(CATEGORY_ORDER),
        )
        for cat in cat_keys:
            sorted_categories[cat] = sorted(categories[cat])

        return {
            "tag_to_idx": self.tag_to_idx,
            "idx_to_tag": {str(k): v for k, v in self.idx_to_tag.items()},
            "tag_to_category": self.tag_to_category,
            "num_tags": len(self.tag_to_idx),
            "categories": sorted_categories,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_tags(self) -> int:
        return len(self.tag_to_idx)

    def category_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for cat in self.tag_to_category.values():
            counts[cat] += 1
        return dict(counts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def add_tags(
        self,
        new_tags: List[str],
        category: str = "General",
    ) -> List[Tuple[str, int]]:
        """Add new tags to the vocabulary in place.

        Each tag is normalized before insertion.  Tags already present are
        silently skipped.  New indices are assigned sequentially starting from
        the current ``num_tags``.

        After insertion ``_build_special_indices()`` is re-run so that
        ``rating_indices`` and ``quality_indices`` stay consistent.

        Parameters
        ----------
        new_tags : raw tag strings (will be normalized internally)
        category : default category assigned to new tags

        Returns
        -------
        List of ``(normalized_tag, new_index)`` for tags that were actually added.
        """
        added: List[Tuple[str, int]] = []
        for raw in new_tags:
            norm = normalize_tag(raw)
            if not norm or norm in self.tag_to_idx:
                continue
            idx = len(self.tag_to_idx)
            self.tag_to_idx[norm] = idx
            self.idx_to_tag[idx] = norm
            self.tag_to_category[norm] = category
            added.append((norm, idx))

        if added:
            # Re-build rating/quality index lists to pick up any newly added
            # special tags (unlikely but correct).
            self._build_special_indices()

        return added

    def _build_special_indices(self) -> None:
        """Populate rating_indices and quality_indices from current tag_to_idx.

        Also corrects tag_to_category for any Quality/Rating tag that was
        mis-classified (e.g. danbooru stores 'bad quality' as General).
        """
        self.rating_indices = []
        self.quality_indices = {k: [] for k in QUALITY_TAG_GROUPS}

        # Pre-build normalized lookup sets for O(1) membership test
        _rating_norms: Set[str] = {normalize_tag(r) for r in RATING_TAGS}
        _quality_norms: Dict[str, Set[str]] = {
            gname: {normalize_tag(t) for t in gtags}
            for gname, gtags in QUALITY_TAG_GROUPS.items()
        }
        _all_quality_norms: Set[str] = {n for s in _quality_norms.values() for n in s}

        for tag, idx in self.tag_to_idx.items():
            norm = normalize_tag(tag)
            if norm in _rating_norms:
                self.rating_indices.append(idx)
                # Correct category if mis-classified
                if self.tag_to_category.get(tag) != "Rating":
                    self.tag_to_category[tag] = "Rating"
            if norm in _all_quality_norms:
                # Correct category if mis-classified (e.g. danbooru General)
                if self.tag_to_category.get(tag) != "Quality":
                    self.tag_to_category[tag] = "Quality"
                for group_name, group_norms in _quality_norms.items():
                    if norm in group_norms:
                        self.quality_indices[group_name].append(idx)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_caption_tags(caption) -> List[Tuple[str, str]]:
    """Extract (tag, category) pairs from a DatasetCaption row.

    Prefers tag_data JSON (has category info); falls back to content string
    with category="__lookup__" sentinel so the caller can batch-resolve
    categories via taglist_cache.
    """
    if caption.tag_data:
        try:
            raw = json.loads(caption.tag_data) if isinstance(caption.tag_data, str) else caption.tag_data
            if isinstance(raw, list):
                result = []
                for item in raw:
                    if isinstance(item, dict) and "tag" in item:
                        result.append((item["tag"], item.get("category", "General")))
                return result
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: parse comma-separated content.
    # Use "__lookup__" sentinel so build_from_dataset_ids can resolve via taglist_cache.
    if caption.content:
        tags = [t.strip() for t in caption.content.split(",") if t.strip()]
        return [(t, "__lookup__") for t in tags]

    return []
