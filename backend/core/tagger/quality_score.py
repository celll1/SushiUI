"""Map a Danbooru post score to a quality tag.

Shared by both online-augmentation paths:
  - tagger training     (core/tagger/danbooru_sampler.py)
  - image-gen training  (core/training/danbooru_image_augment.py)

The default thresholds follow the Animagine XL 3.0 convention (absolute Danbooru
score -> quality tier). Absolute thresholds are used rather than dataset-wide
percentiles because the online collector fetches posts incrementally and cannot
rank the whole dataset.

  Quality tag      Danbooru score
  masterpiece      > 150
  best quality     100 - 150
  high quality     75 - 100
  medium quality   25 - 75
  normal quality   0 - 25
  low quality      -5 - 0      (negative tier)
  worst quality    < -5        (negative tier)

A tier whose *minimum* score is negative is a "negative tier" and is only
emitted when ``attach_negative`` is set. Thresholds are fully overridable per run.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

# Ordered high -> low. Each entry is (minimum_inclusive_score, quality_tag).
# Lower bounds are inclusive, so e.g. score 150 -> "best quality", 151 ->
# "masterpiece", matching the Animagine XL 3.0 ranges above.
DEFAULT_QUALITY_THRESHOLDS: List[Tuple[int, str]] = [
    (151, "masterpiece"),
    (100, "best quality"),
    (75, "high quality"),
    (25, "medium quality"),
    (0, "normal quality"),
    (-5, "low quality"),
    (-10 ** 9, "worst quality"),
]


def parse_quality_thresholds(spec: str) -> List[Tuple[int, str]]:
    """Parse a user threshold spec into a descending ``(min_score, tag)`` list.

    Each rule is one ``<min_score> <tag name>`` entry (e.g. ``"150 masterpiece"``),
    separated by newlines and/or commas. The tag may contain spaces
    (``"100 best quality"``). Blank or unparseable lines are skipped. When the
    spec yields no valid rule, :data:`DEFAULT_QUALITY_THRESHOLDS` is returned so
    an empty/typo'd config falls back to the built-in default rather than
    silently disabling the feature.
    """
    if not spec or not spec.strip():
        return list(DEFAULT_QUALITY_THRESHOLDS)

    rules: List[Tuple[int, str]] = []
    for line in spec.replace(",", "\n").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        score_tok, tag = parts[0], parts[1].strip()
        try:
            score = int(float(score_tok))
        except ValueError:
            continue
        if tag:
            rules.append((score, tag))

    if not rules:
        return list(DEFAULT_QUALITY_THRESHOLDS)

    rules.sort(key=lambda r: r[0], reverse=True)
    return rules


def score_to_quality_tag(
    score: Optional[int],
    thresholds: Optional[List[Tuple[int, str]]] = None,
    attach_negative: bool = False,
) -> Optional[str]:
    """Return the quality tag for ``score``, or ``None`` if none applies.

    Parameters
    ----------
    score : Danbooru community score (``post["score"]``). ``None`` -> no tag.
    thresholds : descending ``(min_score, tag)`` rules; defaults to
        :data:`DEFAULT_QUALITY_THRESHOLDS`.
    attach_negative : when False, tiers whose minimum score is negative
        (e.g. ``low quality`` / ``worst quality``) yield ``None`` instead of a tag.
    """
    if score is None:
        return None
    try:
        score = int(score)
    except (TypeError, ValueError):
        return None

    rules = thresholds if thresholds is not None else DEFAULT_QUALITY_THRESHOLDS
    for min_score, tag in rules:
        if score >= min_score:
            if min_score < 0 and not attach_negative:
                return None
            return tag
    return None
