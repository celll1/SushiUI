"""On-demand consistency checks for indexed datasets and their source files."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


_MAX_SAMPLE_NAMES = 20
_MAX_JSON_SIDECAR_BYTES = 16 * 1024 * 1024


def _sample(counter: Counter[str], name: str) -> None:
    if len(counter) < _MAX_SAMPLE_NAMES or name in counter:
        counter[name] += 1


def _sidecar_state(image_path: str, indexed_at: datetime | None) -> tuple[bool, bool, bool]:
    media = Path(image_path)
    candidates = (media.with_suffix(".txt"), media.with_suffix(".json"))
    existing = [path for path in candidates if path.is_file()]
    invalid = False
    for path in existing:
        try:
            if path.suffix == ".json":
                if path.stat().st_size > _MAX_JSON_SIDECAR_BYTES:
                    invalid = True
                else:
                    with path.open("r", encoding="utf-8") as handle:
                        if not isinstance(json.load(handle), dict):
                            invalid = True
            else:
                path.read_text(encoding="utf-8")
        except (OSError, UnicodeError, json.JSONDecodeError):
            invalid = True
    try:
        stale = bool(
            indexed_at
            and any(datetime.utcfromtimestamp(path.stat().st_mtime) > indexed_at for path in existing)
        )
    except OSError:
        invalid = True
        stale = False
    return bool(existing), invalid, stale


def inspect_dataset_rows(
    rows: Iterable[Any],
    *,
    last_scanned_at: datetime | None,
) -> dict[str, Any]:
    """Inspect projected item rows without retaining the dataset in memory."""
    counts: Counter[str] = Counter()
    samples: dict[str, Counter[str]] = {
        key: Counter()
        for key in (
            "missing_media",
            "missing_sidecar",
            "invalid_sidecar",
            "stale_sidecar",
            "duplicate_stem",
            "reference_failure",
            "metadata_gap",
        )
    }
    stem_counts: Counter[str] = Counter()

    for row in rows:
        counts["total_items"] += 1
        name = row.base_name
        media_exists = os.path.isfile(row.image_path)
        if not media_exists:
            counts["missing_media"] += 1
            _sample(samples["missing_media"], name)

        stem = os.path.normcase(os.path.normpath(os.path.splitext(row.image_path)[0]))
        if stem_counts[stem]:
            counts["duplicate_stem"] += 1
            _sample(samples["duplicate_stem"], name)
        stem_counts[stem] += 1

        if row.total_captions:
            counts["captioned_items"] += 1
        if row.total_tags:
            counts["tagged_items"] += 1

        has_sidecar, invalid_sidecar, stale_sidecar = _sidecar_state(
            row.image_path, row.indexed_at or last_scanned_at
        )
        if row.has_file_caption and not has_sidecar:
            counts["missing_sidecar"] += 1
            _sample(samples["missing_sidecar"], name)
        if invalid_sidecar:
            counts["invalid_sidecar"] += 1
            _sample(samples["invalid_sidecar"], name)
        if stale_sidecar:
            counts["stale_sidecar"] += 1
            _sample(samples["stale_sidecar"], name)

        if row.item_type == "reference":
            related = row.related_images if isinstance(row.related_images, Mapping) else {}
            references = related.get("reference") or []
            if not references or any(not os.path.isfile(path) for path in references):
                counts["reference_failure"] += 1
                _sample(samples["reference_failure"], name)

        if row.item_type in ("single", "reference") and (not row.width or not row.height):
            counts["metadata_gap"] += 1
            _sample(samples["metadata_gap"], name)

    issue_keys = (
        "missing_media",
        "missing_sidecar",
        "invalid_sidecar",
        "stale_sidecar",
        "duplicate_stem",
        "reference_failure",
        "metadata_gap",
    )
    return {
        "counts": {key: counts[key] for key in ("total_items", "captioned_items", "tagged_items", *issue_keys)},
        "samples": {key: list(samples[key]) for key in issue_keys},
        "healthy": not any(counts[key] for key in issue_keys),
    }
